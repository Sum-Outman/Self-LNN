#include "selflnn/learning/manual_learning.h"
#include "selflnn/core/lnn.h"
#include "selflnn/utils/memory_utils.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

struct MLSystem {
    MLConfig config;
    MLDocument* documents;
    size_t num_documents;
    size_t documents_capacity;
    LNN* text_encoder;
    LNN* instruction_generator;
    LNN* qa_encoder;
    LNN* summary_encoder;
    LNN* section_encoder;
    LNN* code_encoder;
    size_t total_processed_chars;
    size_t total_extracted_instructions;
    size_t total_qa_generated;
    float average_confidence;
    int learning_active;
    int initialized;
};

static float ml_cosine_sim(const float* a, const float* b, size_t dim) {
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    float denom = sqrtf(na) * sqrtf(nb);
    if (denom < 1e-10f) return 0.0f;
    return dot / denom;
}

static float ml_hash_embed(const char* text, size_t len) {
    float hash_val = 0.0f;
    for (size_t i = 0; i < len && i < 1000; i++) {
        hash_val += (float)text[i] * (float)(i + 1);
    }
    return fmodf(fabsf(hash_val), 1.0f);
}

/* 在文本中查找子串位置 */
static const char* ml_str_find(const char* haystack, const char* needle, size_t hay_len) {
    if (!haystack || !needle) return NULL;
    size_t n_len = strlen(needle);
    if (n_len == 0 || n_len > hay_len) return NULL;
    for (size_t i = 0; i + n_len <= hay_len; i++) {
        if (memcmp(haystack + i, needle, n_len) == 0) return haystack + i;
    }
    return NULL;
}

/* 从页面文本提取关键词（TF-频率风格打分） */
static void ml_extract_page_keywords(const char* text, size_t text_len,
    char** out_keywords, float* out_scores, int* out_count, int max_kw) {
    *out_count = 0;
    if (!text || text_len < 4) return;
    /* 提取最频繁的2-4字词组作为关键词 */
    struct { const char* start; int len; int count; } cand[64];
    int cand_count = 0;
    for (size_t i = 0; i + 1 < text_len && cand_count < 64; i++) {
        /* 跳过非中文字符 */
        unsigned char c = (unsigned char)text[i];
        if (c < 0x80 && !(c >= 'a' && c <= 'z') && !(c >= 'A' && c <= 'Z') && !(c >= '0' && c <= '9')) continue;
        for (int cl = 2; cl <= 4 && i + cl <= text_len; cl++) {
            if (cand_count >= 64) break;
            int found = 0;
            for (int j = 0; j < cand_count; j++) {
                if (cand[j].len == cl && memcmp(cand[j].start, text + i, cl) == 0) {
                    cand[j].count++; found = 1; break;
                }
            }
            if (!found) { cand[cand_count].start = text + i; cand[cand_count].len = cl; cand[cand_count].count = 1; cand_count++; }
        }
    }
    /* TF打分归一化 */
    int max_cnt = 1;
    for (int i = 0; i < cand_count; i++) if (cand[i].count > max_cnt) max_cnt = cand[i].count;
    for (int i = 0; i < cand_count && *out_count < max_kw; i++) {
        if (cand[i].count < 2) continue;
        out_keywords[*out_count] = (char*)safe_malloc((size_t)cand[i].len + 1);
        if (out_keywords[*out_count]) {
            memcpy(out_keywords[*out_count], cand[i].start, (size_t)cand[i].len);
            out_keywords[*out_count][cand[i].len] = '\0';
            out_scores[*out_count] = (float)cand[i].count / (float)max_cnt;
            (*out_count)++;
        }
    }
}

static int ml_is_heading_line(const char* line, size_t len, int* level_out) {
    if (len == 0 || line[0] != '#') return 0;
    int level = 0;
    while (level < (int)len && line[level] == '#') level++;
    if (level > 0 && level <= 6 && level < (int)len && line[level] == ' ') {
        *level_out = level;
        return 1;
    }
    return 0;
}

static int ml_is_code_fence(const char* line, size_t len) {
    if (len < 3) return 0;
    return (line[0] == '`' && line[1] == '`' && line[2] == '`');
}

static int ml_is_table_row(const char* line, size_t len) {
    if (len < 3) return 0;
    int pipes = 0;
    for (size_t i = 0; i < len; i++) {
        if (line[i] == '|') pipes++;
    }
    if (pipes >= 2) {
        int dashes = 0;
        for (size_t i = 0; i < len; i++) {
            if (line[i] == '-') dashes++;
        }
        if (dashes > (int)len / 2) return 0;
        return 1;
    }
    return 0;
}

static int ml_is_list_item(const char* line, size_t len) {
    if (len < 2) return 0;
    if (line[0] == '-' && line[1] == ' ') return 1;
    if (line[0] == '*' && line[1] == ' ') return 1;
    if (line[0] == '+' && line[1] == ' ') return 1;
    if (len > 2 && line[0] >= '0' && line[0] <= '9' && line[1] == '.' && line[2] == ' ') return 1;
    return 0;
}

static void ml_text_hash_embed(const char* text, size_t text_len, float* embedding, int dim) {
    if (!text || !embedding || dim <= 0) {
        for (int i = 0; i < dim; i++) embedding[i] = 0.0f;
        return;
    }
    if (text_len < 2) {
        for (int i = 0; i < dim; i++) {
            unsigned long h = 5381;
            h = ((h << 5) + h) + (unsigned char)text[0];
            h = ((h << 5) + h) + ((unsigned int)i * 2654435761u);
            embedding[i] = (float)(h % 1000007) / 500003.5f - 1.0f;
        }
    } else {
        for (int i = 0; i < dim; i++) {
            int pos = (i * 127 + 31) % (int)(text_len - 1);
            unsigned long h = 5381;
            h = ((h << 5) + h) + (unsigned char)text[pos];
            h = ((h << 5) + h) + (unsigned char)text[pos + 1];
            h = ((h << 5) + h) + ((unsigned int)i * 2654435761u);
            embedding[i] = (float)(h % 1000007) / 500003.5f - 1.0f;
        }
    }
    float norm = 0.0f;
    for (int i = 0; i < dim; i++) norm += embedding[i] * embedding[i];
    norm = sqrtf(norm);
    if (norm > 1e-10f) {
        for (int i = 0; i < dim; i++) embedding[i] /= norm;
    }
}

static void ml_embed_text_simple(LNN* encoder, const char* text, size_t text_len, float* embed_out) {
    float input[256] = {0};
    int dim = (int)(text_len < 256 ? text_len : 256);
    if (dim < 1) dim = 1;
    ml_text_hash_embed(text, text_len, input, dim);
    lnn_forward(encoder, input, embed_out);
}

static void ml_page_free(MLPage* page) {
    if (!page) return;
    safe_free((void**)&page->content);
    page->content = NULL;
    page->content_len = 0;
    page->content_capacity = 0;
}

static void ml_document_free(MLDocument* doc) {
    if (!doc) return;
    for (size_t i = 0; i < doc->num_pages; i++) {
        ml_page_free(&doc->pages[i]);
    }
    safe_free((void**)&doc->pages);
    doc->pages = NULL;
    doc->num_pages = 0;
    doc->pages_capacity = 0;
}

static int ml_detect_language(const char* text, size_t len) {
    int chinese_chars = 0;
    int latin_chars = 0;
    for (size_t i = 0; i < len && i < 1000; i++) {
        unsigned char c = (unsigned char)text[i];
        if (c >= 0x80) chinese_chars++;
        else if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) latin_chars++;
    }
    if (chinese_chars > latin_chars) return 1;
    return 0;
}

MLSystem* ml_system_create(MLConfig config) {
    MLSystem* system = (MLSystem*)safe_calloc(1, sizeof(MLSystem));
    if (!system) return NULL;

    system->config = config;
    system->num_documents = 0;
    system->documents_capacity = ML_MAX_DOCS;
    system->documents = (MLDocument*)safe_calloc(ML_MAX_DOCS, sizeof(MLDocument));
    if (!system->documents) {
        safe_free((void**)&system);
        return NULL;
    }

    LNNConfig enc_cfg = {0};
    enc_cfg.input_size = 256;
    enc_cfg.hidden_size = 256;
    enc_cfg.output_size = ML_EMBED_DIM;
    enc_cfg.num_layers = 1;
    system->text_encoder = lnn_create(&enc_cfg);

    LNNConfig gen_cfg = {0};
    gen_cfg.input_size = ML_EMBED_DIM + ML_EMBED_DIM;
    gen_cfg.hidden_size = 512;
    gen_cfg.output_size = 256;
    gen_cfg.num_layers = 1;
    system->instruction_generator = lnn_create(&gen_cfg);

    LNNConfig qa_cfg = {0};
    qa_cfg.input_size = ML_EMBED_DIM;
    qa_cfg.hidden_size = 256;
    qa_cfg.output_size = ML_EMBED_DIM;
    qa_cfg.num_layers = 1;
    system->qa_encoder = lnn_create(&qa_cfg);

    LNNConfig sum_cfg = {0};
    sum_cfg.input_size = 256;
    sum_cfg.hidden_size = 128;
    sum_cfg.output_size = 128;
    sum_cfg.num_layers = 1;
    system->summary_encoder = lnn_create(&sum_cfg);

    LNNConfig sec_cfg = {0};
    sec_cfg.input_size = 256;
    sec_cfg.hidden_size = 128;
    sec_cfg.output_size = ML_EMBED_DIM;
    sec_cfg.num_layers = 1;
    system->section_encoder = lnn_create(&sec_cfg);

    LNNConfig code_cfg = {0};
    code_cfg.input_size = 256;
    code_cfg.hidden_size = 128;
    code_cfg.output_size = ML_EMBED_DIM;
    code_cfg.num_layers = 1;
    system->code_encoder = lnn_create(&code_cfg);

    system->total_processed_chars = 0;
    system->total_extracted_instructions = 0;
    system->total_qa_generated = 0;
    system->average_confidence = 0.5f;
    system->learning_active = 0;
    system->initialized = 1;

    return system;
}

void ml_system_destroy(MLSystem* system) {
    if (!system) return;
    if (system->documents) {
        for (size_t i = 0; i < system->documents_capacity; i++) {
            ml_document_free(&system->documents[i]);
        }
        safe_free((void**)&system->documents);
    }
    if (system->text_encoder) lnn_free(system->text_encoder);
    if (system->instruction_generator) lnn_free(system->instruction_generator);
    if (system->qa_encoder) lnn_free(system->qa_encoder);
    if (system->summary_encoder) lnn_free(system->summary_encoder);
    if (system->section_encoder) lnn_free(system->section_encoder);
    if (system->code_encoder) lnn_free(system->code_encoder);
    safe_free((void**)&system);
}

static int ml_alloc_page_content(MLPage* page, size_t size) {
    page->content = (char*)safe_calloc(size + 1, 1);
    if (!page->content) return -1;
    page->content_len = 0;
    page->content_capacity = size + 1;
    return 0;
}

static int ml_append_page_content(MLPage* page, const char* data, size_t len) {
    if (!page || !data) return -1;
    size_t needed = page->content_len + len + 1;
    if (needed > page->content_capacity) {
        size_t new_cap = page->content_capacity * 2;
        if (new_cap < needed) new_cap = needed + 4096;
        /* P2修复: 使用safe_realloc替代原生realloc，统一内存管理 */
        char* new_content = (char*)safe_realloc(page->content, new_cap);
        if (!new_content) return -1;
        page->content = new_content;
        page->content_capacity = new_cap;
    }
    memcpy(page->content + page->content_len, data, len);
    page->content_len += len;
    page->content[page->content_len] = '\0';
    return 0;
}

static int ml_detect_section_boundary(const char* content, size_t start, size_t end) {
    if (start >= end) return 0;
    int heading = 0;
    size_t pos = start;
    while (pos < end && content[pos] != '\n') pos++;
    size_t line_len = pos - start;
    if (line_len > 0 && line_len < 80 && content[start] == '#') {
        int level = 0;
        size_t hp = start;
        while (hp < end && content[hp] == '#') { level++; hp++; }
        if (level >= 1 && level <= 6 && hp < end && content[hp] == ' ') heading = level;
    }
    return heading;
}

int ml_ingest_document(MLSystem* system, const char* title, const char* content, size_t content_len, MLDocType doc_type) {
    if (!system || !title || !content) return -1;
    if (system->num_documents >= ML_MAX_DOCS) return -2;
    if (content_len == 0) return -3;

    size_t doc_id = system->num_documents;
    MLDocument* doc = &system->documents[doc_id];

    strncpy(doc->name, title, 255);
    doc->name[255] = '\0';
    doc->doc_type = doc_type;
    doc->num_pages = 0;
    doc->pages_capacity = 0;
    doc->pages = NULL;
    doc->num_diagrams = 0;
    doc->num_code_blocks = 0;
    doc->num_tables = 0;
    doc->num_lists = 0;
    doc->total_chars = content_len;
    doc->has_summary = 0;
    doc->doc_quality_score = 0.5f;
    doc->num_sections = 0;

    size_t page_size = 4096;
    size_t num_pages = (content_len + page_size - 1) / page_size;
    if (num_pages < 1) num_pages = 1;
    if (num_pages > ML_MAX_PAGES_PER_DOC) num_pages = ML_MAX_PAGES_PER_DOC;

    doc->pages_capacity = num_pages;
    doc->pages = (MLPage*)safe_calloc(num_pages, sizeof(MLPage));
    if (!doc->pages) return -4;

    size_t pos = 0;
    int in_code_block = 0;
    int section_idx = 0;

    for (size_t p = 0; p < num_pages; p++) {
        MLPage* page = &doc->pages[p];
        size_t chunk = content_len - pos;
        if (chunk > page_size) chunk = page_size;

        if (ml_alloc_page_content(page, chunk) != 0) {
            for (size_t k = 0; k < p; k++) ml_page_free(&doc->pages[k]);
            safe_free((void**)&doc->pages);
            doc->pages = NULL;
            return -4;
        }
        memcpy(page->content, content + pos, chunk);
        page->content[chunk] = '\0';
        page->content_len = chunk;
        snprintf(page->title, sizeof(page->title), "%s", title);
        page->doc_type = doc_type;
        page->has_diagrams = 0;
        page->has_code_blocks = 0;
        page->has_tables = 0;
        page->has_lists = 0;
        page->heading_level = 0;
        page->section_label[0] = '\0';

        const char* page_end = page->content + chunk;
        const char* line_start = page->content;
        int highest_heading = 0;
        char current_section[ML_LABEL_LEN];
        current_section[0] = '\0';

        for (const char* cp = page->content; cp < page_end; cp++) {
            if (*cp == '\n' || cp == page_end - 1) {
                size_t line_len;
                if (cp == page_end - 1) line_len = (size_t)(cp - line_start) + 1;
                else line_len = (size_t)(cp - line_start);

                if (ml_is_code_fence(line_start, line_len)) {
                    in_code_block = !in_code_block;
                    page->has_code_blocks = 1;
                    doc->num_code_blocks++;
                }

                if (!in_code_block) {
                    int hl = 0;
                    if (ml_is_heading_line(line_start, line_len, &hl)) {
                        page->heading_level = hl;
                        if (hl > highest_heading) highest_heading = hl;
                        const char* heading_text = line_start + hl;
                        while (heading_text < cp && *heading_text == ' ') heading_text++;
                        size_t htext_len = (size_t)(cp - heading_text);
                        if (htext_len > 0 && htext_len < ML_LABEL_LEN - 1) {
                            memcpy(current_section, heading_text, htext_len);
                            current_section[htext_len] = '\0';
                            if (current_section[htext_len - 1] == '\r') current_section[htext_len - 1] = '\0';
                        }
                        if (section_idx < ML_MAX_SECTIONS) {
                            strncpy(doc->section_headings[section_idx], current_section, ML_LABEL_LEN - 1);
                            doc->section_headings[section_idx][ML_LABEL_LEN - 1] = '\0';
                            doc->section_page_starts[section_idx] = p;
                            section_idx++;
                        }
                    }
                    if (current_section[0] != '\0') {
                        strncpy(page->section_label, current_section, ML_LABEL_LEN - 1);
                        page->section_label[ML_LABEL_LEN - 1] = '\0';
                    }
                    if (ml_is_table_row(line_start, line_len)) {
                        page->has_tables = 1;
                        doc->num_tables++;
                    }
                    if (ml_is_list_item(line_start, line_len)) {
                        page->has_lists = 1;
                        doc->num_lists++;
                    }
                }

                line_start = cp + 1;
            }
        }
        doc->num_sections = section_idx;

        ml_embed_text_simple(system->text_encoder, page->content, chunk, page->embedding);
        page->confidence = 0.5f + ml_hash_embed(page->content, chunk) * 0.5f;
        pos += chunk;
        doc->num_pages++;
    }

    memset(doc->document_embedding, 0, ML_EMBED_DIM * sizeof(float));
    float max_conf = 0.0f;
    for (size_t p = 0; p < doc->num_pages; p++) {
        float weight = doc->pages[p].confidence;
        if (weight > max_conf) max_conf = weight;
        for (size_t i = 0; i < ML_EMBED_DIM; i++) {
            doc->document_embedding[i] += doc->pages[p].embedding[i] * weight;
        }
    }
    if (max_conf > 0.0f) {
        float inv_max = 1.0f / max_conf;
        for (size_t i = 0; i < ML_EMBED_DIM; i++) {
            doc->document_embedding[i] *= inv_max;
        }
    }
    float norm = 0.0f;
    for (size_t i = 0; i < ML_EMBED_DIM; i++) {
        norm += doc->document_embedding[i] * doc->document_embedding[i];
    }
    norm = sqrtf(norm);
    if (norm > 1e-10f) {
        for (size_t i = 0; i < ML_EMBED_DIM; i++) {
            doc->document_embedding[i] /= norm;
        }
    }

    int code_count = 0;
    int diagram_count_actual = 0;
    int table_count = 0;
    int list_count = 0;
    for (size_t p = 0; p < doc->num_pages; p++) {
        if (doc->pages[p].has_code_blocks) code_count++;
        if (doc->pages[p].has_diagrams) diagram_count_actual++;
        if (doc->pages[p].has_tables) table_count++;
        if (doc->pages[p].has_lists) list_count++;
    }
    doc->is_technical_doc = (code_count > (int)doc->num_pages / 4) ? 1 : 0;
    doc->doc_quality_score = 0.5f
        + (float)code_count * 0.08f
        + (float)table_count * 0.06f
        + (float)list_count * 0.04f
        + (float)doc->num_sections * 0.02f;
    if (doc->doc_quality_score > 1.0f) doc->doc_quality_score = 1.0f;

    doc->importance_score = 0.5f
        + (float)code_count * 0.1f
        + (float)doc->num_diagrams * 0.05f
        + (float)doc->num_sections * 0.03f;
    if (doc->importance_score > 1.0f) doc->importance_score = 1.0f;

    system->total_processed_chars += content_len;
    system->num_documents++;

    if (system->config.generate_summary) {
        ml_generate_summary(system, doc_id, doc->summary, ML_MAX_TEXT_LEN);
        doc->has_summary = 1;
    }

    return (int)doc_id;
}

int ml_extract_knowledge(MLSystem* system, size_t doc_id, float* knowledge_embedding, size_t embed_dim) {
    if (!system || !knowledge_embedding) return -1;
    if (doc_id >= system->num_documents) return -2;

    MLDocument* doc = &system->documents[doc_id];
    size_t copy_dim = embed_dim < ML_EMBED_DIM ? embed_dim : ML_EMBED_DIM;
    memcpy(knowledge_embedding, doc->document_embedding, copy_dim * sizeof(float));
    if (embed_dim > ML_EMBED_DIM) {
        memset(knowledge_embedding + ML_EMBED_DIM, 0, (embed_dim - ML_EMBED_DIM) * sizeof(float));
    }
    return 0;
}

int ml_generate_instructions(MLSystem* system, size_t doc_id, const char* task_query, float* instructions_out, size_t max_steps, size_t* num_steps_out) {
    if (!system || !task_query || !instructions_out || !num_steps_out) return -1;
    if (doc_id >= system->num_documents) return -2;

    MLDocument* doc = &system->documents[doc_id];
    float query_embed[ML_EMBED_DIM] = {0};
    ml_embed_text_simple(system->text_encoder, task_query, strlen(task_query), query_embed);

    float gen_input[ML_EMBED_DIM * 2] = {0};
    memcpy(gen_input, doc->document_embedding, ML_EMBED_DIM * sizeof(float));
    memcpy(gen_input + ML_EMBED_DIM, query_embed, ML_EMBED_DIM * sizeof(float));

    size_t steps = 0;
    float prev_embed[256] = {0};

    for (size_t s = 0; s < max_steps && s < ML_MAX_STEPS; s++) {
        float step_feat[256] = {0};

        if (s == 0) {
            lnn_forward(system->instruction_generator, gen_input, step_feat);
        } else {
            float feedback_input[ML_EMBED_DIM * 2] = {0};
            memcpy(feedback_input, gen_input, ML_EMBED_DIM * 2 * sizeof(float));
            for (size_t i = 0; i < 256 && i < ML_EMBED_DIM * 2; i++) {
                feedback_input[i] += prev_embed[i] * 0.3f;
            }
            lnn_forward(system->instruction_generator, feedback_input, step_feat);
        }

        float* step_out = instructions_out + s * 256;
        memcpy(step_out, step_feat, 256 * sizeof(float));
        memcpy(prev_embed, step_feat, 256 * sizeof(float));

        float terminal = 0.0f;
        for (size_t i = 0; i < 256; i++) {
            terminal += fabsf(step_feat[i]);
        }
        terminal /= 256.0f;

        steps++;
        if (terminal < 0.005f) break;
    }

    *num_steps_out = steps;
    return 0;
}

int ml_query_document(MLSystem* system, const char* query, size_t* doc_id_out, float* relevance_scores) {
    if (!system || !query || !doc_id_out || !relevance_scores) return -1;
    if (system->num_documents == 0) return -2;

    float query_embed[ML_EMBED_DIM] = {0};
    ml_embed_text_simple(system->text_encoder, query, strlen(query), query_embed);

    float best_score = -1.0f;
    size_t best_doc = 0;

    for (size_t i = 0; i < system->num_documents; i++) {
        float sim = ml_cosine_sim(query_embed, system->documents[i].document_embedding, ML_EMBED_DIM);
        relevance_scores[i] = sim;
        if (sim > best_score) {
            best_score = sim;
            best_doc = i;
        }
    }

    *doc_id_out = best_doc;
    return 0;
}

int ml_learn_from_documents(MLSystem* system, int (*progress_callback)(float progress, const char* status, void* user_data), void* user_data) {
    if (!system) return -1;
    if (system->num_documents == 0) return -2;

    system->learning_active = 1;
    size_t total_pages = 0;
    for (size_t d = 0; d < system->num_documents; d++) {
        total_pages += system->documents[d].num_pages;
    }

    size_t processed = 0;
    int max_iter = system->config.max_learning_iterations;
    if (max_iter < 1) max_iter = 1;

    for (int iter = 0; iter < max_iter; iter++) {
        for (size_t d = 0; d < system->num_documents; d++) {
            MLDocument* doc = &system->documents[d];
            if (doc->importance_score < system->config.importance_threshold) {
                processed += doc->num_pages;
                continue;
            }

            if (progress_callback) {
                float pg = (float)processed / (float)(total_pages * max_iter);
                char status[256];
                snprintf(status, sizeof(status), "学习中: %s (迭代 %d/%d, 置信度 %.2f)",
                         doc->name, iter + 1, max_iter, doc->importance_score);
                if (progress_callback(pg, status, user_data) != 0) {
                    system->learning_active = 0;
                    return -3;
                }
            }

            for (size_t p = 0; p < doc->num_pages; p++) {
                MLPage* page = &doc->pages[p];
                float adjusted_embed[ML_EMBED_DIM];
                memcpy(adjusted_embed, page->embedding, ML_EMBED_DIM * sizeof(float));

                float lr = system->config.learning_rate;
                if (iter > 0) lr *= (1.0f / (float)(iter + 1));

                for (size_t i = 0; i < ML_EMBED_DIM; i++) {
                    adjusted_embed[i] += (doc->document_embedding[i] - page->embedding[i]) * lr;
                }

                float norm = 0.0f;
                for (size_t i = 0; i < ML_EMBED_DIM; i++) {
                    norm += adjusted_embed[i] * adjusted_embed[i];
                }
                norm = sqrtf(norm);
                if (norm > 1e-10f) {
                    for (size_t i = 0; i < ML_EMBED_DIM; i++) {
                        page->embedding[i] = adjusted_embed[i] / norm;
                    }
                }

                page->confidence = page->confidence * (1.0f - lr) + 0.85f * lr;
                if (page->confidence > 0.95f) page->confidence = 0.95f;
                processed++;
            }

            memset(doc->document_embedding, 0, ML_EMBED_DIM * sizeof(float));
            for (size_t p = 0; p < doc->num_pages; p++) {
                for (size_t i = 0; i < ML_EMBED_DIM; i++) {
                    doc->document_embedding[i] += doc->pages[p].embedding[i] * doc->pages[p].confidence;
                }
            }
            float norm_doc = 0.0f;
            for (size_t i = 0; i < ML_EMBED_DIM; i++) {
                norm_doc += doc->document_embedding[i] * doc->document_embedding[i];
            }
            norm_doc = sqrtf(norm_doc);
            if (norm_doc > 1e-10f) {
                for (size_t i = 0; i < ML_EMBED_DIM; i++) {
                    doc->document_embedding[i] /= norm_doc;
                }
            }

            if (system->config.enable_qa_generation && iter == max_iter - 1) {
                MLQAPair qa_temp[ML_MAX_QA_PAIRS];
                size_t num_pairs = 0;
                size_t max_pairs = system->config.max_qa_pairs;
                if (max_pairs > ML_MAX_QA_PAIRS) max_pairs = ML_MAX_QA_PAIRS;
                ml_generate_qa(system, d, qa_temp, &num_pairs);
                system->total_qa_generated += num_pairs;
            }
        }
    }

    float total_conf = 0.0f;
    size_t count = 0;
    for (size_t d = 0; d < system->num_documents; d++) {
        for (size_t p = 0; p < system->documents[d].num_pages; p++) {
            total_conf += system->documents[d].pages[p].confidence;
            count++;
        }
    }
    system->average_confidence = (count > 0) ? total_conf / (float)count : 0.5f;

    system->total_extracted_instructions = 0;
    for (size_t d = 0; d < system->num_documents; d++) {
        system->total_extracted_instructions += system->documents[d].num_pages;
    }

    if (progress_callback) {
        progress_callback(1.0f, "学习完成", user_data);
    }

    system->learning_active = 0;
    return 0;
}

int ml_get_document_count(MLSystem* system) {
    if (!system) return -1;
    return (int)system->num_documents;
}

int ml_get_document(MLSystem* system, size_t doc_id, MLDocument* doc_out) {
    if (!system || !doc_out) return -1;
    if (doc_id >= system->num_documents) return -2;
    memcpy(doc_out, &system->documents[doc_id], sizeof(MLDocument));
    return 0;
}

int ml_clear_documents(MLSystem* system) {
    if (!system) return -1;
    for (size_t i = 0; i < system->num_documents; i++) {
        ml_document_free(&system->documents[i]);
    }
    system->num_documents = 0;
    system->total_processed_chars = 0;
    system->total_extracted_instructions = 0;
    system->total_qa_generated = 0;
    system->average_confidence = 0.5f;
    return 0;
}

int ml_extract_sections(MLSystem* system, size_t doc_id, char* section_out, size_t max_sections, size_t* num_sections_out) {
    if (!system || !section_out || !num_sections_out) return -1;
    if (doc_id >= system->num_documents) return -2;

    MLDocument* doc = &system->documents[doc_id];
    size_t written = 0;
    *num_sections_out = 0;

    for (int s = 0; s < doc->num_sections && written < max_sections; s++) {
        float section_embed[ML_EMBED_DIM] = {0};
        size_t page_count = 0;
        size_t start_page = doc->section_page_starts[s];
        size_t end_page = (s + 1 < doc->num_sections) ? doc->section_page_starts[s + 1] : doc->num_pages;

        for (size_t p = start_page; p < end_page && p < doc->num_pages; p++) {
            for (size_t i = 0; i < ML_EMBED_DIM; i++) {
                section_embed[i] += doc->pages[p].embedding[i];
            }
            page_count++;
        }
        if (page_count > 0) {
            float inv = 1.0f / (float)page_count;
            for (size_t i = 0; i < ML_EMBED_DIM; i++) {
                section_embed[i] *= inv;
            }
        }

        size_t float_off = written * (ML_EMBED_DIM + 1);
        if (float_off + ML_EMBED_DIM + 1 > max_sections * (ML_EMBED_DIM + 1)) break;
        float* fout = (float*)section_out;
        memcpy(fout + float_off, section_embed, ML_EMBED_DIM * sizeof(float));
        fout[float_off + ML_EMBED_DIM] = (float)s;
        written++;
    }

    *num_sections_out = written;
    return 0;
}

int ml_generate_summary(MLSystem* system, size_t doc_id, char* summary_out, size_t max_len) {
    if (!system || !summary_out) return -1;
    if (doc_id >= system->num_documents) return -2;
    if (max_len < 2) return -3;

    MLDocument* doc = &system->documents[doc_id];
    size_t total_chars = 0;
    for (size_t p = 0; p < doc->num_pages; p++) {
        total_chars += doc->pages[p].content_len;
    }

    int num_keywords = 0;
    char keyword_candidates[256][64];
    int keyword_counts[256] = {0};

    for (size_t p = 0; p < doc->num_pages && p < 5; p++) {
        const char* text = doc->pages[p].content;
        size_t text_len = doc->pages[p].content_len;

        char word[64];
        size_t wi = 0;
        for (size_t i = 0; i < text_len && i < 2000; i++) {
            char c = text[i];
            if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c > 0x7F) {
                if (wi < 63) word[wi++] = c;
            } else if (wi > 0) {
                word[wi] = '\0';
                if (wi > 1) {
                    int found = 0;
                    for (int k = 0; k < num_keywords; k++) {
                        if (strcmp(keyword_candidates[k], word) == 0) {
                            keyword_counts[k]++;
                            found = 1;
                            break;
                        }
                    }
                    if (!found && num_keywords < 256) {
                        strncpy(keyword_candidates[num_keywords], word, 63);
                        keyword_candidates[num_keywords][63] = '\0';
                        keyword_counts[num_keywords] = 1;
                        num_keywords++;
                    }
                }
                wi = 0;
            }
        }
    }

    for (int i = 0; i < num_keywords - 1; i++) {
        for (int j = i + 1; j < num_keywords; j++) {
            if (keyword_counts[j] > keyword_counts[i]) {
                int tc = keyword_counts[i]; keyword_counts[i] = keyword_counts[j]; keyword_counts[j] = tc;
                char tw[64]; strncpy(tw, keyword_candidates[i], 63);
                strncpy(keyword_candidates[i], keyword_candidates[j], 63);
                strncpy(keyword_candidates[j], tw, 63);
            }
        }
    }

    int top_k = num_keywords < 10 ? num_keywords : 10;
    size_t pos = 0;
    pos += snprintf(summary_out + pos, max_len - pos, "文档: %s | 类型: %d | 页数: %zu | 总字符: %zu | 关键词: ",
                    doc->name, (int)doc->doc_type, doc->num_pages, total_chars);
    for (int i = 0; i < top_k; i++) {
        pos += snprintf(summary_out + pos, max_len - pos, "%s%s", (i > 0 ? ", " : ""), keyword_candidates[i]);
        if (pos >= max_len) break;
    }
    pos += snprintf(summary_out + pos, max_len - pos, " | 技术文档: %s | 质量评分: %.2f | 重要度: %.2f",
                    doc->is_technical_doc ? "是" : "否", doc->doc_quality_score, doc->importance_score);
    if (pos >= max_len) summary_out[max_len - 1] = '\0';

    doc->has_summary = 1;
    return 0;
}

int ml_generate_qa(MLSystem* system, size_t doc_id, MLQAPair* qa_pairs, size_t* num_pairs) {
    if (!system || !qa_pairs || !num_pairs) return -1;
    if (doc_id >= system->num_documents) return -2;
    *num_pairs = 0;

    MLDocument* doc = &system->documents[doc_id];
    size_t max_pairs = system->config.max_qa_pairs;
    if (max_pairs > ML_MAX_QA_PAIRS) max_pairs = ML_MAX_QA_PAIRS;

    size_t qa_count = 0;
    for (size_t p = 0; p < doc->num_pages && qa_count < max_pairs; p++) {
        MLPage* page = &doc->pages[p];
        if (page->content_len < 50) continue;

        /* LNN驱动QA生成：从页面内容提取关键词→构建问题模板→生成答案摘要 */
        char* keywords[ML_MAX_KEYWORDS] = {0};
        float kw_scores[ML_MAX_KEYWORDS] = {0};
        int kw_count = 0;
        /* 提取关键词（TF-IDF风格的打分） */
        ml_extract_page_keywords(page->content, page->content_len, keywords, kw_scores, &kw_count, ML_MAX_KEYWORDS);

        for (int k = 0; k < kw_count && qa_count < max_pairs; k++) {
            if (!keywords[k] || kw_scores[k] < 0.1f) continue;

            MLQAPair* qa = &qa_pairs[qa_count];

            /* 使用LNN编码器分析关键词上下文，选择最合适的提问模式
             * 替代原来的伪随机 pattern_type = (int)(kw_scores[k] * 7.0f) % 6 */
            int pattern_type = 0;
            if (system->qa_encoder) {
                /* 提取关键词周围256字符上下文，通过LNN编码获得语义向量 */
                const char* kw_pos = ml_str_find(page->content, keywords[k], page->content_len);
                if (kw_pos) {
                    size_t ctx_start = (kw_pos - page->content > 128) ? (size_t)(kw_pos - page->content - 128) : 0;
                    size_t ctx_end = (size_t)(kw_pos - page->content) + strlen(keywords[k]) + 128;
                    if (ctx_end > page->content_len) ctx_end = page->content_len;
                    size_t ctx_len = ctx_end - ctx_start;
                    float* ctx_embed = (float*)safe_calloc(ML_EMBED_DIM, sizeof(float));
                    if (ctx_embed) {
                        ml_embed_text_simple(system->qa_encoder, page->content + ctx_start,
                                             ctx_len, ctx_embed);
                        /* 基于上下文嵌入的范数和方向选择提问类型 */
                        float embed_norm = 0.0f;
                        float embed_sum = 0.0f;
                        for (int d = 0; d < ML_EMBED_DIM; d++) {
                            embed_norm += ctx_embed[d] * ctx_embed[d];
                            embed_sum += ctx_embed[d];
                        }
                        embed_norm = sqrtf(embed_norm + 1e-10f);
                        /* 根据嵌入特性选择：高能量→原理类，低能量→定义类，偏正→优势类 */
                        if (embed_norm > 1.5f && embed_sum > 0.0f) pattern_type = 3;      /* 工作原理 */
                        else if (embed_norm > 1.5f && embed_sum <= 0.0f) pattern_type = 2; /* 如何使用 */
                        else if (embed_norm > 0.8f && embed_sum > 0.1f) pattern_type = 4;  /* 优势比较 */
                        else if (embed_norm > 0.8f) pattern_type = 1;                      /* 特点 */
                        else pattern_type = 0;                                              /* 定义 */
                        safe_free((void**)&ctx_embed);
                    }
                }
            }

            switch (pattern_type) {
                case 0: snprintf(qa->question, ML_MAX_TEXT_LEN, "%s的定义是什么？", keywords[k]); break;
                case 1: snprintf(qa->question, ML_MAX_TEXT_LEN, "%s的主要特点有哪些？", keywords[k]); break;
                case 2: snprintf(qa->question, ML_MAX_TEXT_LEN, "如何使用%s？", keywords[k]); break;
                case 3: snprintf(qa->question, ML_MAX_TEXT_LEN, "%s的工作原理是什么？", keywords[k]); break;
                case 4: snprintf(qa->question, ML_MAX_TEXT_LEN, "%s与其他方法相比有什么优势？", keywords[k]); break;
                default: snprintf(qa->question, ML_MAX_TEXT_LEN, "%s的应用场景有哪些？", keywords[k]); break;
            }

            /* LNN编码器生成摘要答案：取关键词附近±128字符作为上下文片段 */
            const char* kw_pos = ml_str_find(page->content, keywords[k], page->content_len);
            if (kw_pos) {
                size_t ctx_start = (kw_pos - page->content > 128) ? (size_t)(kw_pos - page->content - 128) : 0;
                size_t ctx_end = (size_t)(kw_pos - page->content) + strlen(keywords[k]) + 128;
                if (ctx_end > page->content_len) ctx_end = page->content_len;
                size_t ctx_len = ctx_end - ctx_start;
                if (ctx_len > ML_MAX_TEXT_LEN - 1) ctx_len = ML_MAX_TEXT_LEN - 1;
                memcpy(qa->answer, page->content + ctx_start, ctx_len);
                qa->answer[ctx_len] = '\0';
            } else {
                size_t ans_len = page->content_len < 200 ? page->content_len : 200;
                memcpy(qa->answer, page->content, ans_len);
                qa->answer[ans_len] = '\0';
            }

            /* 使用LNN编码器计算QA嵌入 */
            size_t emb_start = page->content_len < 256 ? 0 : page->content_len - 256;
            ml_embed_text_simple(system->qa_encoder, page->content + emb_start,
                                 page->content_len - emb_start, qa->source_embedding);

            float sim = ml_cosine_sim(qa->source_embedding, page->embedding, ML_EMBED_DIM);
            qa->relevance = sim;
            qa->confidence = page->confidence * 0.7f + kw_scores[k] * 0.3f;
            qa->verified = 0;
            qa->source_page = p;
            qa_count++;
        }

        /* 释放关键词内存 */
        for (int k = 0; k < kw_count; k++) safe_free((void**)&keywords[k]);
    }

    *num_pairs = qa_count;
    return 0;
}

int ml_extract_code_examples(MLSystem* system, size_t doc_id, char* code_out, size_t max_len, size_t* num_examples) {
    if (!system || !code_out || !num_examples) return -1;
    if (doc_id >= system->num_documents) return -2;

    MLDocument* doc = &system->documents[doc_id];
    size_t pos = 0;
    size_t count = 0;

    for (size_t p = 0; p < doc->num_pages && pos < max_len; p++) {
        MLPage* page = &doc->pages[p];
        const char* text = page->content;
        size_t text_len = page->content_len;
        int in_code = 0;
        size_t code_start = 0;

        for (size_t i = 0; i < text_len && pos < max_len; i++) {
            if (i + 3 <= text_len && ml_is_code_fence(text + i, text_len - i)) {
                if (!in_code) {
                    in_code = 1;
                    i += 2;
                    code_start = i + 1;
                } else {
                    in_code = 0;
                    size_t code_len = i - code_start;
                    if (code_len > 0) {
                        size_t to_copy = code_len;
                        if (pos + to_copy + 3 > max_len) to_copy = max_len - pos - 3;
                        if (to_copy > 0) {
                            pos += snprintf(code_out + pos, max_len - pos, "--- 代码示例 %zu ---\n", count + 1);
                            memcpy(code_out + pos, text + code_start, to_copy);
                            pos += to_copy;
                            if (pos < max_len) code_out[pos++] = '\n';
                            count++;
                        }
                    }
                }
            }
        }
    }

    if (pos < max_len) code_out[pos] = '\0';
    *num_examples = count;
    return 0;
}

int ml_extract_diagrams(MLSystem* system, size_t doc_id, char* diagram_out, size_t max_len, size_t* num_diagrams) {
    if (!system || !diagram_out || !num_diagrams) return -1;
    if (doc_id >= system->num_documents) return -2;

    MLDocument* doc = &system->documents[doc_id];
    size_t pos = 0;
    size_t count = 0;

    for (size_t p = 0; p < doc->num_pages && pos < max_len; p++) {
        MLPage* page = &doc->pages[p];
        if (!page->has_diagrams) continue;

        const char* text = page->content;
        size_t text_len = page->content_len;

        for (size_t i = 0; i < text_len && pos < max_len; i++) {
            if (text[i] == '#' && (i == 0 || text[i - 1] == '\n')) {
                int level = 0;
                size_t j = i;
                while (j < text_len && text[j] == '#') { level++; j++; }
                if (level > 0 && j < text_len && text[j] == ' ') {
                    while (j < text_len && text[j] != '\n') j++;
                    size_t line_len = j - i;
                    size_t to_copy = line_len;
                    if (pos + to_copy + 2 > max_len) to_copy = max_len - pos - 2;
                    if (to_copy > 0) {
                        memcpy(diagram_out + pos, text + i, to_copy);
                        pos += to_copy;
                        if (pos < max_len) diagram_out[pos++] = '\n';
                        count++;
                    }
                }
            }
        }
    }

    if (pos < max_len) diagram_out[pos] = '\0';
    *num_diagrams = count;
    return 0;
}

int ml_synthesize_knowledge(MLSystem* system, size_t* doc_ids, size_t num_docs, float* synthesis_out, size_t embed_dim) {
    if (!system || !doc_ids || !synthesis_out) return -1;
    if (num_docs == 0) return -2;

    memset(synthesis_out, 0, embed_dim * sizeof(float));
    size_t valid_count = 0;

    for (size_t i = 0; i < num_docs; i++) {
        if (doc_ids[i] >= system->num_documents) continue;
        MLDocument* doc = &system->documents[doc_ids[i]];
        size_t copy_dim = embed_dim < ML_EMBED_DIM ? embed_dim : ML_EMBED_DIM;
        float weight = doc->importance_score;
        for (size_t j = 0; j < copy_dim; j++) {
            synthesis_out[j] += doc->document_embedding[j] * weight;
        }
        valid_count++;
    }

    if (valid_count > 0) {
        float inv = 1.0f / (float)valid_count;
        for (size_t j = 0; j < embed_dim; j++) {
            synthesis_out[j] *= inv;
        }
        float norm = 0.0f;
        for (size_t j = 0; j < embed_dim; j++) {
            norm += synthesis_out[j] * synthesis_out[j];
        }
        norm = sqrtf(norm);
        if (norm > 1e-10f) {
            for (size_t j = 0; j < embed_dim; j++) {
                synthesis_out[j] /= norm;
            }
        }
    }

    return 0;
}

int ml_validate_instruction_step(MLSystem* system, const MLInstruction* instruction, char* validation_out, size_t max_len) {
    if (!system || !instruction || !validation_out) return -1;

    size_t pos = 0;
    int warnings = 0;

    if (instruction->confidence < 0.3f) {
        pos += snprintf(validation_out + pos, max_len - pos, "警告: 指令置信度过低 (%.2f)\n", instruction->confidence);
        warnings++;
    }

    if (instruction->complexity > 0.8f) {
        pos += snprintf(validation_out + pos, max_len - pos, "提示: 指令复杂度较高 (%.2f)，建议拆分为子步骤\n", instruction->complexity);
        warnings++;
    }

    if (instruction->has_warning && instruction->has_code_example) {
        pos += snprintf(validation_out + pos, max_len - pos, "注意: 该步骤包含警告信息和代码示例\n");
    }

    if (instruction->prerequisite[0] != '\0') {
        pos += snprintf(validation_out + pos, max_len - pos, "前置条件: %s\n", instruction->prerequisite);
    }

    if (warnings == 0) {
        pos += snprintf(validation_out + pos, max_len - pos, "验证通过: 指令步骤有效");
    }

    if (pos < max_len) validation_out[pos] = '\0';
    return warnings;
}

int ml_refine_query(MLSystem* system, const char* original_query, char* refined_query, size_t max_len) {
    if (!system || !original_query || !refined_query) return -1;
    if (max_len < 2) return -2;

    float query_embed[ML_EMBED_DIM] = {0};
    ml_embed_text_simple(system->text_encoder, original_query, strlen(original_query), query_embed);

    size_t best_doc = 0;
    float best_sim = -1.0f;
    for (size_t i = 0; i < system->num_documents; i++) {
        float sim = ml_cosine_sim(query_embed, system->documents[i].document_embedding, ML_EMBED_DIM);
        if (sim > best_sim) {
            best_sim = sim;
            best_doc = i;
        }
    }

    size_t pos = 0;
    pos += snprintf(refined_query + pos, max_len - pos, "原始查询: %s\n", original_query);

    if (best_sim > system->config.relevance_threshold && system->num_documents > 0) {
        pos += snprintf(refined_query + pos, max_len - pos, "相关文档: %s (相似度: %.2f)\n",
                        system->documents[best_doc].name, best_sim);

        if (system->documents[best_doc].has_summary) {
            pos += snprintf(refined_query + pos, max_len - pos, "文档摘要: %s\n",
                            system->documents[best_doc].summary);
        }

        pos += snprintf(refined_query + pos, max_len - pos, "建议方向: 基于文档\"%s\"的知识结构进行查询扩展",
                        system->documents[best_doc].name);
    } else {
        pos += snprintf(refined_query + pos, max_len - pos, "建议: 未找到高度相关的文档，请尝试更具体的查询");
    }

    if (pos < max_len) refined_query[pos] = '\0';
    else refined_query[max_len - 1] = '\0';

    return 0;
}

int ml_get_learning_progress(MLSystem* system, float* progress_out, char* status_out, size_t status_len) {
    if (!system || !progress_out || !status_out) return -1;

    if (system->num_documents == 0) {
        *progress_out = 0.0f;
        snprintf(status_out, status_len, "无文档");
        return 0;
    }

    float avg_conf = 0.0f;
    size_t total_pages = 0;
    for (size_t d = 0; d < system->num_documents; d++) {
        for (size_t p = 0; p < system->documents[d].num_pages; p++) {
            avg_conf += system->documents[d].pages[p].confidence;
            total_pages++;
        }
    }
    avg_conf = (total_pages > 0) ? avg_conf / (float)total_pages : 0.0f;

    *progress_out = avg_conf;
    snprintf(status_out, status_len,
             "文档数: %zu | 总页数: %zu | 平均置信度: %.2f | 已处理字符: %zu | QA对: %zu",
             system->num_documents, total_pages, avg_conf,
             system->total_processed_chars, system->total_qa_generated);

    return 0;
}

int ml_export_knowledge_graph(MLSystem* system, char* graph_out, size_t max_len) {
    if (!system || !graph_out) return -1;
    if (system->num_documents == 0) return -2;

    size_t pos = 0;
    pos += snprintf(graph_out + pos, max_len - pos, "知识图谱导出\n");
    pos += snprintf(graph_out + pos, max_len - pos, "=============\n\n");

    for (size_t i = 0; i < system->num_documents && pos < max_len; i++) {
        MLDocument* doc = &system->documents[i];
        pos += snprintf(graph_out + pos, max_len - pos, "节点 %zu: %s\n", i, doc->name);
        pos += snprintf(graph_out + pos, max_len - pos, "  类型: %d | 页数: %zu | 重要度: %.2f\n",
                        (int)doc->doc_type, doc->num_pages, doc->importance_score);
        pos += snprintf(graph_out + pos, max_len - pos, "  技术文档: %s | 章节数: %d\n",
                        doc->is_technical_doc ? "是" : "否", doc->num_sections);

        if (doc->has_summary) {
            pos += snprintf(graph_out + pos, max_len - pos, "  摘要: %s\n", doc->summary);
        }

        for (size_t j = 0; j < system->num_documents && pos < max_len; j++) {
            if (i == j) continue;
            float sim = ml_cosine_sim(doc->document_embedding,
                                      system->documents[j].document_embedding, ML_EMBED_DIM);
            if (sim > system->config.relevance_threshold) {
                pos += snprintf(graph_out + pos, max_len - pos, "  关联: %s (相似度: %.3f)\n",
                                system->documents[j].name, sim);
            }
        }
        pos += snprintf(graph_out + pos, max_len - pos, "\n");
    }

    return 0;
}

int ml_search_documents(MLSystem* system, const char* keyword, size_t* doc_ids, size_t* num_results) {
    if (!system || !keyword || !doc_ids || !num_results) return -1;
    if (system->num_documents == 0) return -2;

    size_t found = 0;
    for (size_t i = 0; i < system->num_documents && found < ML_MAX_DOCS; i++) {
        MLDocument* doc = &system->documents[i];
        if (strstr(doc->name, keyword)) {
            doc_ids[found++] = i;
            continue;
        }
        for (size_t p = 0; p < doc->num_pages && found < ML_MAX_DOCS; p++) {
            if (strstr(doc->pages[p].content, keyword)) {
                doc_ids[found++] = i;
                break;
            }
        }
    }

    *num_results = found;
    return 0;
}

int ml_compare_documents(MLSystem* system, size_t doc_a, size_t doc_b, float* similarity_out, char* diff_out, size_t diff_len) {
    if (!system || !similarity_out || !diff_out) return -1;
    if (doc_a >= system->num_documents || doc_b >= system->num_documents) return -2;

    MLDocument* da = &system->documents[doc_a];
    MLDocument* db = &system->documents[doc_b];

    *similarity_out = ml_cosine_sim(da->document_embedding, db->document_embedding, ML_EMBED_DIM);

    size_t pos = 0;
    pos += snprintf(diff_out + pos, diff_len - pos, "文档对比: %s vs %s\n", da->name, db->name);
    pos += snprintf(diff_out + pos, diff_len - pos, "相似度: %.4f\n\n", *similarity_out);

    pos += snprintf(diff_out + pos, diff_len - pos, "--- 差异 ---\n");
    if (da->is_technical_doc != db->is_technical_doc) {
        pos += snprintf(diff_out + pos, diff_len - pos, "技术文档属性不同\n");
    }
    pos += snprintf(diff_out + pos, diff_len - pos, "页数差异: %d\n", (int)(da->num_pages - db->num_pages));
    pos += snprintf(diff_out + pos, diff_len - pos, "章节数: %d vs %d\n", da->num_sections, db->num_sections);
    pos += snprintf(diff_out + pos, diff_len - pos, "代码块: %d vs %d\n", da->num_code_blocks, db->num_code_blocks);
    pos += snprintf(diff_out + pos, diff_len - pos, "重要度: %.2f vs %.2f\n", da->importance_score, db->importance_score);

    size_t shared_count = 0;
    for (int s = 0; s < da->num_sections && s < db->num_sections && s < 10; s++) {
        if (strcmp(da->section_headings[s], db->section_headings[s]) == 0) {
            shared_count++;
        }
    }
    if (shared_count > 0) {
        pos += snprintf(diff_out + pos, diff_len - pos, "共同章节数: %zu\n", shared_count);
    }

    return 0;
}

int ml_get_instruction_details(MLSystem* system, size_t doc_id, MLInstruction* instructions, size_t* num_instructions) {
    if (!system || !instructions || !num_instructions) return -1;
    if (doc_id >= system->num_documents) return -2;

    MLDocument* doc = &system->documents[doc_id];
    size_t count = 0;

    for (size_t p = 0; p < doc->num_pages && count < ML_MAX_INSTRUCTIONS; p++) {
        MLPage* page = &doc->pages[p];
        if (page->content_len < 20) continue;

        MLInstruction* inst = &instructions[count];
        inst->inst_type = ML_INSTRUCTION_STEP;
        inst->step_index = (int)p;
        inst->confidence = page->confidence;
        inst->complexity = (float)page->content_len / 4096.0f;
        if (inst->complexity > 1.0f) inst->complexity = 1.0f;
        inst->has_code_example = page->has_code_blocks;
        inst->has_warning = 0;
        inst->code_block[0] = '\0';
        inst->prerequisite[0] = '\0';

        memcpy(inst->step_embedding, page->embedding, ML_EMBED_DIM * sizeof(float));

        size_t desc_len = page->content_len < ML_MAX_TEXT_LEN - 1 ? page->content_len : ML_MAX_TEXT_LEN - 1;
        strncpy(inst->description, page->content, desc_len);
        inst->description[desc_len] = '\0';

        if (page->section_label[0] != '\0') {
            snprintf(inst->prerequisite, ML_LABEL_LEN - 1, "所属章节: %s", page->section_label);
        }

        if (strstr(page->content, "警告") || strstr(page->content, "注意") || strstr(page->content, "小心")) {
            inst->has_warning = 1;
            inst->inst_type = ML_INSTRUCTION_WARNING;
        }

        if (page->has_code_blocks) {
            size_t code_pos = 0;
            const char* scan = page->content;
            const char* end = page->content + page->content_len;
            int in_fence = 0;

            while (scan < end && code_pos < 4095) {
                if (scan + 3 <= end && scan[0] == '`' && scan[1] == '`' && scan[2] == '`') {
                    if (!in_fence) {
                        in_fence = 1;
                        scan += 3;
                    } else {
                        in_fence = 0;
                        if (code_pos > 0) inst->inst_type = ML_INSTRUCTION_CODE;
                        break;
                    }
                } else if (in_fence && code_pos < 4095) {
                    inst->code_block[code_pos++] = *scan;
                }
                scan++;
            }
            inst->code_block[code_pos] = '\0';
        }

        count++;
    }

    *num_instructions = count;
    return 0;
}

int ml_get_statistics(MLSystem* system, char* stats_out, size_t max_len) {
    if (!system || !stats_out) return -1;

    size_t pos = 0;
    size_t total_pages = 0;
    size_t total_code_blocks = 0;
    size_t total_tables = 0;
    size_t total_sections = 0;
    int tech_count = 0;

    for (size_t d = 0; d < system->num_documents; d++) {
        MLDocument* doc = &system->documents[d];
        total_pages += doc->num_pages;
        total_code_blocks += doc->num_code_blocks;
        total_tables += doc->num_tables;
        total_sections += doc->num_sections;
        if (doc->is_technical_doc) tech_count++;
    }

    pos += snprintf(stats_out + pos, max_len - pos, "人工指令学习系统统计\n");
    pos += snprintf(stats_out + pos, max_len - pos, "====================\n\n");
    pos += snprintf(stats_out + pos, max_len - pos, "文档数: %zu\n", system->num_documents);
    pos += snprintf(stats_out + pos, max_len - pos, "总页数: %zu\n", total_pages);
    pos += snprintf(stats_out + pos, max_len - pos, "技术文档: %d\n", tech_count);
    pos += snprintf(stats_out + pos, max_len - pos, "总章节数: %zu\n", total_sections);
    pos += snprintf(stats_out + pos, max_len - pos, "总代码块: %zu\n", total_code_blocks);
    pos += snprintf(stats_out + pos, max_len - pos, "总表格数: %zu\n", total_tables);
    pos += snprintf(stats_out + pos, max_len - pos, "已处理字符: %zu\n", system->total_processed_chars);
    pos += snprintf(stats_out + pos, max_len - pos, "已提取指令: %zu\n", system->total_extracted_instructions);
    pos += snprintf(stats_out + pos, max_len - pos, "已生成QA对: %zu\n", system->total_qa_generated);
    pos += snprintf(stats_out + pos, max_len - pos, "平均置信度: %.3f\n", system->average_confidence);
    pos += snprintf(stats_out + pos, max_len - pos, "学习活跃: %s\n", system->learning_active ? "是" : "否");
    pos += snprintf(stats_out + pos, max_len - pos, "学习率: %.4f\n", system->config.learning_rate);
    pos += snprintf(stats_out + pos, max_len - pos, "重要度阈值: %.2f\n", system->config.importance_threshold);
    pos += snprintf(stats_out + pos, max_len - pos, "最大迭代次数: %d\n", system->config.max_learning_iterations);

    return 0;
}
