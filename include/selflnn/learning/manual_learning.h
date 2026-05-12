#ifndef SELFLNN_MANUAL_LEARNING_H
#define SELFLNN_MANUAL_LEARNING_H

#include "selflnn/core/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ML_MAX_DOCS 128
#define ML_MAX_PAGES_PER_DOC 1024
#define ML_MAX_TEXT_LEN 65536
#define ML_MAX_STEPS 2048
#define ML_EMBED_DIM 512
#define ML_MAX_INSTRUCTIONS 512
#define ML_MAX_SECTIONS 256
#define ML_MAX_QA_PAIRS 1024
#define ML_MAX_KEYWORDS 64
#define ML_LABEL_LEN 256

typedef enum {
    ML_DOC_TYPE_PDF = 0,
    ML_DOC_TYPE_HTML = 1,
    ML_DOC_TYPE_TEXT = 2,
    ML_DOC_TYPE_MARKDOWN = 3,
    ML_DOC_TYPE_IMAGE = 4
} MLDocType;

typedef enum {
    ML_INSTRUCTION_FLOW = 0,
    ML_INSTRUCTION_STEP = 1,
    ML_INSTRUCTION_WARNING = 2,
    ML_INSTRUCTION_NOTE = 3,
    ML_INSTRUCTION_TIP = 4,
    ML_INSTRUCTION_CODE = 5,
    ML_INSTRUCTION_DIAGRAM = 6
} MLInstructionType;

typedef struct {
    char title[256];
    char* content;
    size_t content_len;
    size_t content_capacity;
    MLDocType doc_type;
    float embedding[ML_EMBED_DIM];
    float confidence;
    int has_diagrams;
    int has_code_blocks;
    int has_tables;
    int has_lists;
    int heading_level;
    char section_label[ML_LABEL_LEN];
} MLPage;

typedef struct MLDocument {
    char name[256];
    MLPage* pages;
    size_t num_pages;
    size_t pages_capacity;
    MLDocType doc_type;
    float document_embedding[ML_EMBED_DIM];
    float importance_score;
    int is_technical_doc;
    int num_diagrams;
    int num_code_blocks;
    int num_tables;
    int num_lists;
    size_t total_chars;
    char summary[ML_MAX_TEXT_LEN];
    int has_summary;
    float doc_quality_score;
    int num_sections;
    char section_headings[ML_MAX_SECTIONS][ML_LABEL_LEN];
    size_t section_page_starts[ML_MAX_SECTIONS];
} MLDocument;

typedef struct {
    char question[ML_MAX_TEXT_LEN];
    char answer[ML_MAX_TEXT_LEN];
    float relevance;
    int verified;
    float confidence;
    size_t source_page;
    float source_embedding[ML_EMBED_DIM];
} MLQAPair;

typedef struct {
    MLInstructionType inst_type;
    float step_embedding[ML_EMBED_DIM];
    char description[ML_MAX_TEXT_LEN];
    int step_index;
    float confidence;
    float complexity;
    int has_code_example;
    int has_warning;
    char code_block[4096];
    char prerequisite[ML_LABEL_LEN];
} MLInstruction;

typedef struct {
    float learning_rate;
    float importance_threshold;
    int extract_diagrams;
    int extract_code;
    int generate_summary;
    int enable_qa_generation;
    size_t max_qa_pairs;
    int enable_hierarchy_extraction;
    int enable_cross_doc_synthesis;
    int enable_step_validation;
    int max_learning_iterations;
    float relevance_threshold;
} MLConfig;

#define ML_CONFIG_DEFAULT { \
    0.001f, 0.3f, 1, 1, 1, 1, 50, 1, 1, 1, 10, 0.15f \
}

typedef struct MLSystem MLSystem;

MLSystem* ml_system_create(MLConfig config);
void ml_system_destroy(MLSystem* system);

int ml_ingest_document(MLSystem* system, const char* title, const char* content, size_t content_len, MLDocType doc_type);
int ml_extract_knowledge(MLSystem* system, size_t doc_id, float* knowledge_embedding, size_t embed_dim);
int ml_generate_instructions(MLSystem* system, size_t doc_id, const char* task_query, float* instructions_out, size_t max_steps, size_t* num_steps_out);
int ml_query_document(MLSystem* system, const char* query, size_t* doc_id_out, float* relevance_scores);
int ml_learn_from_documents(MLSystem* system, int (*progress_callback)(float progress, const char* status, void* user_data), void* user_data);
int ml_get_document_count(MLSystem* system);
int ml_get_document(MLSystem* system, size_t doc_id, MLDocument* doc_out);
int ml_clear_documents(MLSystem* system);

int ml_extract_sections(MLSystem* system, size_t doc_id, char* section_out, size_t max_sections, size_t* num_sections_out);
int ml_generate_summary(MLSystem* system, size_t doc_id, char* summary_out, size_t max_len);
int ml_generate_qa(MLSystem* system, size_t doc_id, MLQAPair* qa_pairs, size_t* num_pairs);
int ml_extract_code_examples(MLSystem* system, size_t doc_id, char* code_out, size_t max_len, size_t* num_examples);
int ml_extract_diagrams(MLSystem* system, size_t doc_id, char* diagram_out, size_t max_len, size_t* num_diagrams);
int ml_synthesize_knowledge(MLSystem* system, size_t* doc_ids, size_t num_docs, float* synthesis_out, size_t embed_dim);
int ml_validate_instruction_step(MLSystem* system, const MLInstruction* instruction, char* validation_out, size_t max_len);
int ml_refine_query(MLSystem* system, const char* original_query, char* refined_query, size_t max_len);
int ml_get_learning_progress(MLSystem* system, float* progress_out, char* status_out, size_t status_len);
int ml_export_knowledge_graph(MLSystem* system, char* graph_out, size_t max_len);
int ml_search_documents(MLSystem* system, const char* keyword, size_t* doc_ids, size_t* num_results);
int ml_compare_documents(MLSystem* system, size_t doc_a, size_t doc_b, float* similarity_out, char* diff_out, size_t diff_len);
int ml_get_instruction_details(MLSystem* system, size_t doc_id, MLInstruction* instructions, size_t* num_instructions);
int ml_get_statistics(MLSystem* system, char* stats_out, size_t max_len);

#ifdef __cplusplus
}
#endif

#endif
