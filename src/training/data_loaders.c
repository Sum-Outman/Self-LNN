#include "selflnn/training/data_loaders.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/logging.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <ctype.h>

#ifdef _MSC_VER
#pragma warning(disable:4100 4189 4244 4267 4701 4047)
#endif

/* ========================================================================
 * 内部工具函数
 * 
 * ZSF-041说明：当前支持的图像格式: BMP (24/32位), PPM (P3/P6)
 * PNG/JPEG 不支持: 需要zlib/libjpeg外部依赖，违反纯C零依赖原则
 * 解决方案: 使用外部工具预处理为BMP/PPM格式后加载，或使用本系统内置的图像采集模块
 * ======================================================================== */

#define LOADER_MAX_LINE 65536
#define LOADER_MAX_COLS 4096
#define LOADER_MAX_TOKENS 4096

/** 安全释放并置空 */
#define LOADER_SAFE_FREE(ptr) do { \
    if (*(ptr)) { safe_free((void**)(ptr)); } \
} while(0)

/** 从文件读取一行（处理所有换行符风格） */
static char* loader_read_line(char* buffer, size_t buf_size, FILE* fp) {
    if (!buffer || buf_size == 0 || !fp) return NULL;
    size_t pos = 0;
    int c;
    while (pos < buf_size - 1) {
        c = fgetc(fp);
        if (c == EOF) {
            if (pos == 0) return NULL;
            break;
        }
        if (c == '\n') break;
        if (c == '\r') {
            int next = fgetc(fp);
            if (next != '\n' && next != EOF) ungetc(next, fp);
            break;
        }
        buffer[pos++] = (char)c;
    }
    buffer[pos] = '\0';
    return buffer;
}

/** 判断字符串是否为空白 */
static int loader_is_blank_line(const char* line) {
    while (*line) {
        if (!isspace((unsigned char)*line)) return 0;
        line++;
    }
    return 1;
}

/** 去除字符串两端空白（原地修改） */
static char* loader_trim(char* str) {
    if (!str) return NULL;
    char* end;
    while (isspace((unsigned char)*str)) str++;
    if (*str == '\0') return str;
    end = str + strlen(str) - 1;
    while (end > str && isspace((unsigned char)*end)) end--;
    *(end + 1) = '\0';
    return str;
}

/** 安全字符串复制 */
static char* loader_strdup(const char* src) {
    if (!src) return NULL;
    size_t len = strlen(src);
    char* dst = (char*)safe_malloc(len + 1);
    if (!dst) return NULL;
    memcpy(dst, src, len + 1);
    return dst;
}

/* ========================================================================
 * CSV解析器
 * ======================================================================== */

/**
 * 解析CSV一行，处理引号字段
 * 返回分割后的token数组，token_count为实际数量
 * 返回的tokens指向静态缓冲区，下次调用会覆盖
 */
static int loader_parse_csv_line(const char* line, char delimiter,
                                  char** tokens, int max_tokens) {
    if (!line || !tokens || max_tokens <= 0) return 0;
    int count = 0;
    const char* p = line;

    while (*p && count < max_tokens) {
        /* 跳过前导空白 */
        while (*p && isspace((unsigned char)*p) && *p != delimiter) p++;

        if (*p == '"') {
            /* 引号字段 */
            p++;
            size_t written = 0;
            char* buf = (char*)safe_malloc(LOADER_MAX_LINE);
            if (!buf) return count;
            while (*p && written < LOADER_MAX_LINE - 1) {
                if (*p == '"') {
                    if (*(p + 1) == '"') {
                        buf[written++] = '"';
                        p += 2;
                    } else {
                        p++;
                        break;
                    }
                } else {
                    buf[written++] = *p++;
                }
            }
            buf[written] = '\0';
            tokens[count] = buf;
            count++;
            /* 跳过到下一个分隔符或行尾 */
            while (*p && *p != delimiter) p++;
            if (*p == delimiter) p++;
        } else {
            /* 普通字段 */
            const char* start = p;
            while (*p && *p != delimiter) p++;
            size_t len = (size_t)(p - start);
            char* buf = (char*)safe_malloc(len + 1);
            if (!buf) return count;
            memcpy(buf, start, len);
            buf[len] = '\0';
            /* 去除尾部空白 */
            char* trimmed = loader_trim(buf);
            if (trimmed != buf) {
                memmove(buf, trimmed, strlen(trimmed) + 1);
            }
            tokens[count] = buf;
            count++;
            if (*p == delimiter) p++;
        }
    }
    return count;
}

/** 释放CSV tokens */
static void loader_free_tokens(char** tokens, int count) {
    for (int i = 0; i < count; i++) {
        if (tokens[i]) safe_free((void**)&tokens[i]);
    }
}

TrainingDataset* data_load_csv(const char* filepath, int has_header,
                                char delimiter, const int* label_columns,
                                size_t num_label_cols) {
    if (!filepath) return NULL;

    FILE* fp = fopen(filepath, "r");
    if (!fp) {
        log_error("[CSV加载器] 无法打开文件: %s", filepath);
        return NULL;
    }

    /* 第一遍扫描：确定行数、列数 */
    char line[LOADER_MAX_LINE];
    size_t total_rows = 0;
    int max_cols = 0;
    int first_row_cols = 0;

    /* 跳过BOM头 */
    int c1 = fgetc(fp), c2 = fgetc(fp), c3 = fgetc(fp);
    if (c1 != 0xEF || c2 != 0xBB || c3 != 0xBF) {
        if (c1 != EOF) ungetc(c1, fp);
        if (c2 != EOF) ungetc(c2, fp);
        if (c3 != EOF) ungetc(c3, fp);
    }

    while (loader_read_line(line, sizeof(line), fp)) {
        if (loader_is_blank_line(line)) continue;
        total_rows++;
        if (total_rows == 1) {
            first_row_cols = 0;
            const char* p = line;
            while (*p) {
                while (*p && isspace((unsigned char)*p) && *p != delimiter) p++;
                if (*p == '"') {
                    p++;
                    while (*p) {
                        if (*p == '"') { if (*(p+1)=='"') p++; else break; }
                        p++;
                    }
                    if (*p) p++;
                } else {
                    while (*p && *p != delimiter) p++;
                }
                first_row_cols++;
                if (*p == delimiter) p++;
            }
            max_cols = first_row_cols;
        } else {
            int cols = 0;
            const char* p = line;
            while (*p) {
                while (*p && isspace((unsigned char)*p) && *p != delimiter) p++;
                if (*p == '"') {
                    p++;
                    while (*p) {
                        if (*p == '"') { if (*(p+1)=='"') p++; else break; }
                        p++;
                    }
                    if (*p) p++;
                } else {
                    while (*p && *p != delimiter) p++;
                }
                cols++;
                if (*p == delimiter) p++;
            }
            if (cols > max_cols) max_cols = cols;
        }
    }

    size_t data_rows = has_header ? (total_rows - 1) : total_rows;

    if (data_rows == 0 || max_cols == 0) {
        fclose(fp);
        log_error("[CSV加载器] 文件为空或无有效数据: %s", filepath);
        return NULL;
    }

    /* 确定输入/输出维度 */
    size_t input_dim, output_dim;

    if (num_label_cols > 0 && label_columns) {
        output_dim = num_label_cols;
        input_dim = (size_t)max_cols - num_label_cols;
    } else if (num_label_cols == 0 && label_columns) {
        output_dim = 0;
        input_dim = (size_t)max_cols;
    } else {
        output_dim = 1;
        input_dim = (size_t)max_cols - 1;
    }

    /* 创建数据集 */
    TrainingDataset* ds = dataset_create("csv_dataset", data_rows, input_dim, output_dim);
    if (!ds) {
        fclose(fp);
        log_error("[CSV加载器] 数据集创建失败");
        return NULL;
    }

    /* 第二遍：读取数据 */
    rewind(fp);
    /* 跳过BOM */
    c1 = fgetc(fp); c2 = fgetc(fp); c3 = fgetc(fp);
    if (c1 != 0xEF || c2 != 0xBB || c3 != 0xBF) {
        if (c1 != EOF) ungetc(c1, fp);
        if (c2 != EOF) ungetc(c2, fp);
        if (c3 != EOF) ungetc(c3, fp);
    }

    size_t row_idx = 0;
    int line_num = 0;
    char* tokens[LOADER_MAX_TOKENS];

    while (loader_read_line(line, sizeof(line), fp) && row_idx < data_rows) {
        if (loader_is_blank_line(line)) continue;
        line_num++;
        if (has_header && line_num == 1) continue;

        int num_tokens = loader_parse_csv_line(line, delimiter, tokens, LOADER_MAX_TOKENS);
        if (num_tokens < (int)input_dim) {
            loader_free_tokens(tokens, num_tokens);
            continue;
        }

        float* input_row = ds->inputs + row_idx * input_dim;
        float* output_row = ds->outputs + row_idx * output_dim;

        if (num_label_cols > 0 && label_columns) {
            for (size_t i = 0; i < input_dim; i++) input_row[i] = 0.0f;
            for (size_t i = 0; i < output_dim; i++) output_row[i] = 0.0f;
            int in_idx = 0;
            for (int c = 0; c < num_tokens; c++) {
                int is_label = 0;
                for (size_t l = 0; l < num_label_cols; l++) {
                    if (c == label_columns[l]) { is_label = 1; break; }
                }
                if (is_label) {
                    int out_pos = -1;
                    for (size_t l = 0; l < num_label_cols; l++) {
                        if (c == label_columns[l]) { out_pos = (int)l; break; }
                    }
                    if (out_pos >= 0 && out_pos < (int)output_dim) {
                        output_row[out_pos] = (float)atof(tokens[c]);
                    }
                } else {
                    if (in_idx < (int)input_dim) {
                        input_row[in_idx++] = (float)atof(tokens[c]);
                    }
                }
            }
        } else {
            for (size_t i = 0; i < input_dim && (int)i < num_tokens - (int)output_dim; i++) {
                input_row[i] = (float)atof(tokens[i]);
            }
            for (size_t i = 0; i < output_dim && (int)(input_dim + i) < num_tokens; i++) {
                output_row[i] = (float)atof(tokens[input_dim + i]);
            }
        }

        loader_free_tokens(tokens, num_tokens);
        row_idx++;
    }

    fclose(fp);

    if (row_idx < data_rows) {
        ds->header.num_samples = (uint32_t)row_idx;
    }

    log_info("[CSV加载器] 已加载: %s (%zu样本, %zu→%zu维)",
             filepath, (size_t)ds->header.num_samples, input_dim, output_dim);
    return ds;
}

/* ========================================================================
 * JSON解析器（最小实现）
 * ======================================================================== */

/** JSON值类型 */
typedef enum {
    JSON_NULL, JSON_BOOL, JSON_NUMBER, JSON_STRING,
    JSON_ARRAY, JSON_OBJECT
} JsonType;

/** JSON token节点 */
typedef struct JsonNode {
    JsonType type;
    char* key;
    union {
        int bool_val;
        double num_val;
        char* str_val;
        struct {
            struct JsonNode* items;
            int count;
            int capacity;
        } array;
        struct {
            struct JsonNode* members;
            int count;
            int capacity;
        } object;
    };
} JsonNode;

static void json_node_free(JsonNode* node) {
    if (!node) return;
    if (node->key) safe_free((void**)&node->key);
    if (node->type == JSON_STRING && node->str_val) {
        safe_free((void**)&node->str_val);
    } else if (node->type == JSON_ARRAY) {
        for (int i = 0; i < node->array.count; i++) {
            json_node_free(&node->array.items[i]);
        }
        safe_free((void**)&node->array.items);
    } else if (node->type == JSON_OBJECT) {
        for (int i = 0; i < node->object.count; i++) {
            json_node_free(&node->object.members[i]);
        }
        safe_free((void**)&node->object.members);
    }
}

static int json_skip_whitespace(const char** p) {
    while (**p && isspace((unsigned char)**p)) (*p)++;
    return (**p != '\0') ? 1 : 0;
}

static int json_parse_value(const char** p, JsonNode* node);

static int json_parse_string(const char** p, char** out) {
    if (!json_skip_whitespace(p)) return -1;
    if (**p != '"') return -1;
    (*p)++;
    size_t cap = 256, len = 0;
    char* buf = (char*)safe_malloc(cap);
    if (!buf) return -1;

    while (**p && **p != '"') {
        if (**p == '\\') {
            (*p)++;
            switch (**p) {
                case '"': buf[len++] = '"'; break;
                case '\\': buf[len++] = '\\'; break;
                case '/': buf[len++] = '/'; break;
                case 'b': buf[len++] = '\b'; break;
                case 'f': buf[len++] = '\f'; break;
                case 'n': buf[len++] = '\n'; break;
                case 'r': buf[len++] = '\r'; break;
                case 't': buf[len++] = '\t'; break;
                case 'u': {
                    unsigned int code = 0;
                    for (int i = 0; i < 4; i++) {
                        (*p)++;
                        char c = **p;
                        code <<= 4;
                        if (c >= '0' && c <= '9') code |= (c - '0');
                        else if (c >= 'a' && c <= 'f') code |= (c - 'a' + 10);
                        else if (c >= 'A' && c <= 'F') code |= (c - 'A' + 10);
                    }
                    if (code < 128) buf[len++] = (char)code;
                    else buf[len++] = '?';
                    break;
                }
                default: buf[len++] = **p; break;
            }
        } else {
            buf[len++] = **p;
        }
        (*p)++;
        if (len >= cap - 1) {
            cap *= 2;
            char* new_buf = (char*)safe_realloc(buf, cap);
            if (!new_buf) { safe_free((void**)&buf); return -1; }
            buf = new_buf;
        }
    }
    if (**p == '"') (*p)++;
    buf[len] = '\0';
    *out = buf;
    return 0;
}

static int json_parse_number(const char** p, double* out) {
    if (!json_skip_whitespace(p)) return -1;
    char num_buf[128];
    int idx = 0;
    int has_dot = 0, has_exp = 0;

    if (**p == '-') { num_buf[idx++] = '-'; (*p)++; }
    while (**p && idx < 127) {
        if (**p >= '0' && **p <= '9') {
            num_buf[idx++] = **p; (*p)++;
        } else if (**p == '.' && !has_dot) {
            has_dot = 1; num_buf[idx++] = '.'; (*p)++;
        } else if ((**p == 'e' || **p == 'E') && !has_exp) {
            has_exp = 1; num_buf[idx++] = 'e'; (*p)++;
            if (**p == '+' || **p == '-') { num_buf[idx++] = **p; (*p)++; }
        } else break;
    }
    num_buf[idx] = '\0';
    if (idx == 0 || (idx == 1 && num_buf[0] == '-')) return -1;
    *out = strtod(num_buf, NULL);
    return 0;
}

static int json_parse_array(const char** p, JsonNode* node) {
    if (!json_skip_whitespace(p)) return -1;
    if (**p != '[') return -1;
    (*p)++;
    node->type = JSON_ARRAY;
    node->array.count = 0;
    node->array.capacity = 16;
    node->array.items = (JsonNode*)safe_calloc((size_t)node->array.capacity, sizeof(JsonNode));
    if (!node->array.items) return -1;

    json_skip_whitespace(p);
    if (**p == ']') { (*p)++; return 0; }

    while (**p) {
        if (node->array.count >= node->array.capacity) {
            node->array.capacity *= 2;
            JsonNode* new_items = (JsonNode*)safe_realloc(node->array.items,
                (size_t)node->array.capacity * sizeof(JsonNode));
            if (!new_items) return -1;
            node->array.items = new_items;
        }
        memset(&node->array.items[node->array.count], 0, sizeof(JsonNode));
        if (json_parse_value(p, &node->array.items[node->array.count]) != 0) return -1;
        node->array.count++;
        json_skip_whitespace(p);
        if (**p == ',') { (*p)++; json_skip_whitespace(p); }
        else if (**p == ']') { (*p)++; return 0; }
        else return -1;
    }
    return -1;
}

static int json_parse_object(const char** p, JsonNode* node) {
    if (!json_skip_whitespace(p)) return -1;
    if (**p != '{') return -1;
    (*p)++;
    node->type = JSON_OBJECT;
    node->object.count = 0;
    node->object.capacity = 16;
    node->object.members = (JsonNode*)safe_calloc((size_t)node->object.capacity, sizeof(JsonNode));
    if (!node->object.members) return -1;

    json_skip_whitespace(p);
    if (**p == '}') { (*p)++; return 0; }

    while (**p) {
        if (node->object.count >= node->object.capacity) {
            node->object.capacity *= 2;
            JsonNode* new_members = (JsonNode*)safe_realloc(node->object.members,
                (size_t)node->object.capacity * sizeof(JsonNode));
            if (!new_members) return -1;
            node->object.members = new_members;
        }
        JsonNode* member = &node->object.members[node->object.count];
        memset(member, 0, sizeof(JsonNode));
        if (**p == '"') {
            if (json_parse_string(p, &member->key) != 0) return -1;
        } else return -1;
        json_skip_whitespace(p);
        if (**p != ':') return -1;
        (*p)++;
        if (json_parse_value(p, member) != 0) return -1;
        node->object.count++;
        json_skip_whitespace(p);
        if (**p == ',') { (*p)++; json_skip_whitespace(p); }
        else if (**p == '}') { (*p)++; return 0; }
        else return -1;
    }
    return -1;
}

static int json_parse_value(const char** p, JsonNode* node) {
    if (!json_skip_whitespace(p)) return -1;

    if (**p == '"') {
        node->type = JSON_STRING;
        return json_parse_string(p, &node->str_val);
    } else if (**p == '{') {
        return json_parse_object(p, node);
    } else if (**p == '[') {
        return json_parse_array(p, node);
    } else if (**p == 't' && strncmp(*p, "true", 4) == 0) {
        node->type = JSON_BOOL; node->bool_val = 1; *p += 4; return 0;
    } else if (**p == 'f' && strncmp(*p, "false", 5) == 0) {
        node->type = JSON_BOOL; node->bool_val = 0; *p += 5; return 0;
    } else if (**p == 'n' && strncmp(*p, "null", 4) == 0) {
        node->type = JSON_NULL; *p += 4; return 0;
    } else if (**p == '-' || (**p >= '0' && **p <= '9')) {
        node->type = JSON_NUMBER;
        return json_parse_number(p, &node->num_val);
    }
    return -1;
}

/** 从JSON节点提取数值数组 */
static int json_extract_numbers(JsonNode* node, float** out, int* out_count) {
    if (!node || !out || !out_count) return -1;
    *out = NULL;
    *out_count = 0;

    if (node->type == JSON_NUMBER) {
        *out = (float*)safe_malloc(sizeof(float));
        if (!*out) return -1;
        **out = (float)node->num_val;
        *out_count = 1;
        return 0;
    }
    if (node->type == JSON_ARRAY) {
        int count = 0;
        for (int i = 0; i < node->array.count; i++) {
            if (node->array.items[i].type == JSON_NUMBER) count++;
        }
        if (count == 0) return -1;
        *out = (float*)safe_malloc((size_t)count * sizeof(float));
        if (!*out) return -1;
        int idx = 0;
        for (int i = 0; i < node->array.count; i++) {
            if (node->array.items[i].type == JSON_NUMBER) {
                (*out)[idx++] = (float)node->array.items[i].num_val;
            }
        }
        *out_count = count;
        return 0;
    }
    return -1;
}

TrainingDataset* data_load_json(const char* filepath, const char* input_field,
                                 const char* output_field, size_t output_cols) {
    if (!filepath) return NULL;

    FILE* fp = fopen(filepath, "rb");
    if (!fp) {
        log_error("[JSON加载器] 无法打开文件: %s", filepath);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    if (file_size <= 0) { fclose(fp); return NULL; }
    rewind(fp);

    char* json_text = (char*)safe_malloc((size_t)file_size + 1);
    if (!json_text) { fclose(fp); return NULL; }
    size_t read_size = fread(json_text, 1, (size_t)file_size, fp);
    fclose(fp);
    /* P2-FIX-24: 完整检查fread返回值，防止部分读取 */
    if (read_size != (size_t)file_size) {
        safe_free((void**)&json_text);
        log_error("[JSON加载器] 文件读取不完整: 期望 %ld 字节, 实际读取 %zu 字节", file_size, read_size);
        return NULL;
    }
    json_text[read_size] = '\0';

    const char* p = json_text;
    JsonNode root;
    memset(&root, 0, sizeof(JsonNode));

    if (json_parse_value(&p, &root) != 0) {
        safe_free((void**)&json_text);
        log_error("[JSON加载器] JSON解析失败: %s", filepath);
        return NULL;
    }

    TrainingDataset* ds = NULL;
    const char* in_field = input_field ? input_field : "input";
    const char* out_field = output_field ? output_field : "output";

    if (root.type == JSON_ARRAY && root.array.count > 0) {
        JsonNode* first = &root.array.items[0];
        if (first->type == JSON_OBJECT) {
            size_t num_samples = (size_t)root.array.count;
            size_t input_dim = 1, output_dim_val = output_cols > 0 ? output_cols : 1;
            float* all_inputs = NULL;
            float* all_outputs = NULL;
            int total_in = 0, total_out = 0;
            size_t sample_in_dim = 0, sample_out_dim = 0;
            int first_sample = 1;

            for (size_t s = 0; s < num_samples; s++) {
                JsonNode* obj = &root.array.items[s];
                if (obj->type != JSON_OBJECT) continue;

                float* in_data = NULL; int in_count = 0;
                float* out_data = NULL; int out_count = 0;

                for (int m = 0; m < obj->object.count; m++) {
                    JsonNode* member = &obj->object.members[m];
                    if (member->key && strcmp(member->key, in_field) == 0) {
                        json_extract_numbers(member, &in_data, &in_count);
                    }
                    if (member->key && strcmp(member->key, out_field) == 0) {
                        json_extract_numbers(member, &out_data, &out_count);
                    }
                }

                if (first_sample) {
                    sample_in_dim = in_count > 0 ? (size_t)in_count : 1;
                    sample_out_dim = output_cols > 0 ? (size_t)output_cols :
                                     (out_count > 0 ? (size_t)out_count : 1);
                    all_inputs = (float*)safe_malloc(num_samples * sample_in_dim * sizeof(float));
                    all_outputs = (float*)safe_malloc(num_samples * sample_out_dim * sizeof(float));
                    if (!all_inputs || !all_outputs) {
                        safe_free((void**)&all_inputs); safe_free((void**)&all_outputs);
                        safe_free((void**)&in_data); safe_free((void**)&out_data);
                        json_node_free(&root); safe_free((void**)&json_text);
                        return NULL;
                    }
                    first_sample = 0;
                }

                for (size_t i = 0; i < sample_in_dim; i++) {
                    all_inputs[s * sample_in_dim + i] = (in_data && (int)i < in_count) ? in_data[i] : 0.0f;
                }
                for (size_t i = 0; i < sample_out_dim; i++) {
                    all_outputs[s * sample_out_dim + i] = (out_data && (int)i < out_count) ? out_data[i] : 0.0f;
                }

                safe_free((void**)&in_data);
                safe_free((void**)&out_data);
                total_in += in_count;
                total_out += out_count;
            }

            input_dim = sample_in_dim;
            output_dim_val = sample_out_dim;

            ds = dataset_from_memory(all_inputs, all_outputs, num_samples, input_dim, output_dim_val);
            safe_free((void**)&all_inputs);
            safe_free((void**)&all_outputs);

        } else if (first->type == JSON_ARRAY) {
            size_t num_samples = (size_t)root.array.count;
            size_t num_cols = 0;

            for (size_t s = 0; s < num_samples && s < 5; s++) {
                JsonNode* row = &root.array.items[s];
                if (row->type == JSON_ARRAY) {
                    int row_cols = 0;
                    for (int i = 0; i < row->array.count; i++) {
                        if (row->array.items[i].type == JSON_NUMBER) row_cols++;
                    }
                    if ((int)row_cols > (int)num_cols) num_cols = (size_t)row_cols;
                }
            }

            if (num_cols == 0) { json_node_free(&root); safe_free((void**)&json_text); return NULL; }

            size_t out_dim = output_cols > 0 ? output_cols : 1;
            size_t in_dim = num_cols - out_dim;
            ds = dataset_create("json_dataset", num_samples, in_dim, out_dim);
            if (!ds) { json_node_free(&root); safe_free((void**)&json_text); return NULL; }

            size_t row_idx = 0;
            for (size_t s = 0; s < num_samples; s++) {
                JsonNode* row = &root.array.items[s];
                if (row->type != JSON_ARRAY) continue;
                int col = 0;
                for (int i = 0; i < row->array.count && col < (int)num_cols; i++) {
                    if (row->array.items[i].type == JSON_NUMBER) {
                        float val = (float)row->array.items[i].num_val;
                        if (col < (int)in_dim) {
                            ds->inputs[row_idx * in_dim + col] = val;
                        } else {
                            int out_idx = col - (int)in_dim;
                            if (out_idx < (int)out_dim) {
                                ds->outputs[row_idx * out_dim + out_idx] = val;
                            }
                        }
                        col++;
                    }
                }
                row_idx++;
            }
            ds->header.num_samples = (uint32_t)row_idx;
        }
    } else if (root.type == JSON_OBJECT) {
        JsonNode* data_array = NULL;
        JsonNode* target_array = NULL;

        for (int m = 0; m < root.object.count; m++) {
            JsonNode* member = &root.object.members[m];
            if (member->key) {
                if (strcmp(member->key, "data") == 0 || strcmp(member->key, "inputs") == 0 || strcmp(member->key, "X") == 0) {
                    data_array = member;
                } else if (strcmp(member->key, "target") == 0 || strcmp(member->key, "targets") == 0 || strcmp(member->key, "labels") == 0 || strcmp(member->key, "y") == 0) {
                    target_array = member;
                }
            }
        }

        if (data_array && data_array->type == JSON_ARRAY) {
            size_t num_samples = (size_t)data_array->array.count;
            size_t in_dim = 1, out_dim = output_cols > 0 ? output_cols : 1;

            if (data_array->array.count > 0 && data_array->array.items[0].type == JSON_ARRAY) {
                in_dim = 0;
                for (int i = 0; i < data_array->array.items[0].array.count; i++) {
                    if (data_array->array.items[0].array.items[i].type == JSON_NUMBER) in_dim++;
                }
            }

            ds = dataset_create("json_dataset", num_samples, in_dim, out_dim);
            if (!ds) { json_node_free(&root); safe_free((void**)&json_text); return NULL; }

            for (size_t s = 0; s < num_samples; s++) {
                JsonNode* row = &data_array->array.items[s];
                if (row->type == JSON_ARRAY) {
                    int col = 0;
                    for (int i = 0; i < row->array.count && col < (int)in_dim; i++) {
                        if (row->array.items[i].type == JSON_NUMBER) {
                            ds->inputs[s * in_dim + col++] = (float)row->array.items[i].num_val;
                        }
                    }
                }
            }

            if (target_array && target_array->type == JSON_ARRAY) {
                for (size_t s = 0; s < num_samples && s < (size_t)target_array->array.count; s++) {
                    JsonNode* tgt = &target_array->array.items[s];
                    float* out_row = ds->outputs + s * out_dim;
                    if (tgt->type == JSON_NUMBER) {
                        out_row[0] = (float)tgt->num_val;
                    } else if (tgt->type == JSON_ARRAY) {
                        for (int i = 0; i < (int)out_dim && i < tgt->array.count; i++) {
                            if (tgt->array.items[i].type == JSON_NUMBER)
                                out_row[i] = (float)tgt->array.items[i].num_val;
                        }
                    }
                }
            }
        }
    }

    json_node_free(&root);
    safe_free((void**)&json_text);

    if (!ds) {
        log_error("[JSON加载器] 无法解析数据结构: %s", filepath);
        return NULL;
    }

    log_info("[JSON加载器] 已加载: %s (%u样本, %u→%u维)",
             filepath, ds->header.num_samples, ds->header.input_dim, ds->header.output_dim);
    return ds;
}

/* ========================================================================
 * 图像加载器
 * ======================================================================== */

float* data_load_image_bmp(const char* filepath, int* width, int* height, int* channels) {
    if (!filepath) return NULL;

    FILE* fp = fopen(filepath, "rb");
    if (!fp) return NULL;

    unsigned char header[54];
    if (fread(header, 1, 54, fp) != 54) { fclose(fp); return NULL; }

    if (header[0] != 'B' || header[1] != 'M') { fclose(fp); return NULL; }

    int w = *(int*)&header[18];
    int h = *(int*)&header[22];
    unsigned short bits = *(unsigned short*)&header[28];
    unsigned int compression = *(unsigned int*)&header[30];

    if (w <= 0 || h <= 0 || w > 16384 || h > 16384) { fclose(fp); return NULL; }
    if (bits != 24 && bits != 32) { fclose(fp); return NULL; }
    if (compression != 0) { fclose(fp); return NULL; }

    int ch = (bits == 24) ? 3 : 4;
    unsigned int data_offset = *(unsigned int*)&header[10];

    int row_size = (w * bits / 8 + 3) & ~3;
    size_t pixel_count = (size_t)w * h;

    unsigned char* raw = (unsigned char*)safe_malloc((size_t)row_size * h);
    if (!raw) { fclose(fp); return NULL; }

    fseek(fp, data_offset, SEEK_SET);
    if (fread(raw, 1, (size_t)row_size * h, fp) != (size_t)row_size * h) {
        safe_free((void**)&raw); fclose(fp); return NULL;
    }
    fclose(fp);

    float* pixels = (float*)safe_malloc(pixel_count * ch * sizeof(float));
    if (!pixels) { safe_free((void**)&raw); return NULL; }

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int raw_idx = (h - 1 - y) * row_size + x * (bits / 8);
            int pix_idx = (y * w + x) * ch;
            pixels[pix_idx + 2] = raw[raw_idx + 0] / 255.0f;
            pixels[pix_idx + 1] = raw[raw_idx + 1] / 255.0f;
            pixels[pix_idx + 0] = raw[raw_idx + 2] / 255.0f;
            if (ch == 4) {
                pixels[pix_idx + 3] = (bits == 32) ? raw[raw_idx + 3] / 255.0f : 1.0f;
            }
        }
    }

    safe_free((void**)&raw);

    if (width) *width = w;
    if (height) *height = h;
    if (channels) *channels = ch;
    return pixels;
}

float* data_load_image_ppm(const char* filepath, int* width, int* height, int* channels) {
    if (!filepath) return NULL;

    FILE* fp = fopen(filepath, "rb");
    if (!fp) return NULL;

    char magic[3];
    if (fread(magic, 1, 2, fp) != 2) { fclose(fp); return NULL; }
    magic[2] = '\0';

    if (magic[0] != 'P' || magic[1] != '6') { fclose(fp); return NULL; }

    int c;
    c = fgetc(fp);
    while (c == ' ' || c == '\t' || c == '\n' || c == '\r') c = fgetc(fp);
    if (c == '#') {
        while (c != '\n' && c != EOF) c = fgetc(fp);
        c = fgetc(fp);
        while (c == ' ' || c == '\t' || c == '\n' || c == '\r') c = fgetc(fp);
    }
    ungetc(c, fp);

    int w, h, max_val;
    if (fscanf(fp, "%d %d %d", &w, &h, &max_val) != 3) { fclose(fp); return NULL; }
    c = fgetc(fp);

    if (w <= 0 || h <= 0 || w > 16384 || h > 16384) { fclose(fp); return NULL; }
    if (max_val <= 0 || max_val > 65535) { fclose(fp); return NULL; }

    size_t pixel_count = (size_t)w * h;
    size_t data_size = pixel_count * 3;
    unsigned char* raw = (unsigned char*)safe_malloc(data_size);
    if (!raw) { fclose(fp); return NULL; }

    if (max_val <= 255) {
        if (fread(raw, 1, data_size, fp) != data_size) {
            safe_free((void**)&raw); fclose(fp); return NULL;
        }
    } else {
        for (size_t i = 0; i < data_size; i++) {
            int hi = fgetc(fp);
            int lo = fgetc(fp);
            if (hi == EOF || lo == EOF) {
                safe_free((void**)&raw); fclose(fp); return NULL;
            }
            raw[i] = (unsigned char)(((hi << 8) | lo) * 255 / max_val);
        }
    }
    fclose(fp);

    float* pixels = (float*)safe_malloc(pixel_count * 3 * sizeof(float));
    if (!pixels) { safe_free((void**)&raw); return NULL; }

    for (size_t i = 0; i < pixel_count * 3; i++) {
        pixels[i] = raw[i] / 255.0f;
    }
    safe_free((void**)&raw);

    if (width) *width = w;
    if (height) *height = h;
    if (channels) *channels = 3;
    return pixels;
}

/** 双线性插值缩放图像 */
static float* loader_resize_image(const float* src, int src_w, int src_h, int channels,
                                    int dst_w, int dst_h) {
    if (!src || src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0 || channels <= 0) {
        return NULL;
    }

    float* dst = (float*)safe_calloc((size_t)dst_w * dst_h * channels, sizeof(float));
    if (!dst) return NULL;

    float x_ratio = (float)(src_w - 1) / (float)dst_w;
    float y_ratio = (float)(src_h - 1) / (float)dst_h;

    for (int dy = 0; dy < dst_h; dy++) {
        for (int dx = 0; dx < dst_w; dx++) {
            float sx = dx * x_ratio;
            float sy = dy * y_ratio;
            int ix = (int)sx;
            int iy = (int)sy;
            float fx = sx - ix;
            float fy = sy - iy;

            int ix1 = ix < src_w - 1 ? ix + 1 : ix;
            int iy1 = iy < src_h - 1 ? iy + 1 : iy;

            for (int c = 0; c < channels; c++) {
                float v00 = src[(iy * src_w + ix) * channels + c];
                float v10 = src[(iy * src_w + ix1) * channels + c];
                float v01 = src[(iy1 * src_w + ix) * channels + c];
                float v11 = src[(iy1 * src_w + ix1) * channels + c];

                float v = v00 * (1 - fx) * (1 - fy)
                        + v10 * fx * (1 - fy)
                        + v01 * (1 - fx) * fy
                        + v11 * fx * fy;

                dst[(dy * dst_w + dx) * channels + c] = v;
            }
        }
    }
    return dst;
}

/** 将RGB图像转为灰度 */
static float* loader_to_grayscale(const float* rgb, int width, int height, int channels) {
    if (!rgb || width <= 0 || height <= 0) return NULL;
    size_t count = (size_t)width * height;
    float* gray = (float*)safe_malloc(count * sizeof(float));
    if (!gray) return NULL;

    for (size_t i = 0; i < count; i++) {
        float r = rgb[i * channels], g = rgb[i * channels + 1], b = rgb[i * channels + 2];
        gray[i] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
    return gray;
}

TrainingDataset* data_load_images(const char* const* filepaths, size_t num_images,
                                   const float* labels, size_t label_dim,
                                   int target_width, int target_height, int grayscale) {
    if (!filepaths || num_images == 0) return NULL;

    /* 第一遍扫描：确定所有图像尺寸 */
    int* widths = (int*)safe_calloc(num_images, sizeof(int));
    int* heights = (int*)safe_calloc(num_images, sizeof(int));
    int* chs = (int*)safe_calloc(num_images, sizeof(int));

    if (!widths || !heights || !chs) {
        safe_free((void**)&widths); safe_free((void**)&heights); safe_free((void**)&chs);
        return NULL;
    }

    for (size_t i = 0; i < num_images; i++) {
        int w = 0, h = 0, ch = 0;
        float* px = data_load_image_bmp(filepaths[i], &w, &h, &ch);
        if (!px) px = data_load_image_ppm(filepaths[i], &w, &h, &ch);
        if (px) {
            widths[i] = w; heights[i] = h; chs[i] = ch;
            safe_free((void**)&px);
        } else {
            widths[i] = 0; heights[i] = 0; chs[i] = 0;
        }
    }

    /* 确定输出尺寸 */
    int out_w = target_width, out_h = target_height;
    if (out_w <= 0 || out_h <= 0) {
        int max_w = 0, max_h = 0;
        for (size_t i = 0; i < num_images; i++) {
            if (widths[i] > max_w) max_w = widths[i];
            if (heights[i] > max_h) max_h = heights[i];
        }
        if (max_w <= 0 || max_h <= 0) {
            safe_free((void**)&widths); safe_free((void**)&heights); safe_free((void**)&chs);
            return NULL;
        }
        if (out_w <= 0) out_w = max_w;
        if (out_h <= 0) out_h = max_h;
    }

    int out_ch = grayscale ? 1 : 3;
    size_t input_dim = (size_t)out_w * out_h * out_ch;
    size_t output_dim_val = label_dim > 0 ? label_dim : 1;

    TrainingDataset* ds = dataset_create("image_dataset", num_images, input_dim, output_dim_val);
    if (!ds) {
        safe_free((void**)&widths); safe_free((void**)&heights); safe_free((void**)&chs);
        return NULL;
    }

    for (size_t i = 0; i < num_images; i++) {
        if (widths[i] <= 0 || heights[i] <= 0) continue;

        int img_w = 0, img_h = 0, img_ch = 0;
        float* px = data_load_image_bmp(filepaths[i], &img_w, &img_h, &img_ch);
        if (!px) px = data_load_image_ppm(filepaths[i], &img_w, &img_h, &img_ch);
        if (!px) continue;

        float* resized = px;
        if (img_w != out_w || img_h != out_h) {
            resized = loader_resize_image(px, img_w, img_h, img_ch, out_w, out_h);
            safe_free((void**)&px);
            if (!resized) continue;
        }

        float* final_data = resized;
        int final_ch = img_ch;

        if (grayscale && img_ch >= 3) {
            final_data = loader_to_grayscale(resized, out_w, out_h, img_ch);
            if (resized != px) safe_free((void**)&resized);
            if (!final_data) continue;
            final_ch = 1;
        } else if (grayscale && img_ch < 3) {
            final_ch = 1;
        }

        float* dst_row = ds->inputs + i * input_dim;
        size_t copy_pixels = (size_t)out_w * out_h;
        for (size_t p = 0; p < copy_pixels; p++) {
            for (int c = 0; c < final_ch && c < out_ch; c++) {
                dst_row[p * out_ch + c] = final_data[p * final_ch + c];
            }
        }

        if (final_data != px) safe_free((void**)&final_data);
        else if (final_data != resized) safe_free((void**)&final_data);
        else safe_free((void**)&px);
    }

    if (labels && label_dim > 0) {
        for (size_t i = 0; i < num_images; i++) {
            for (size_t d = 0; d < label_dim && d < output_dim_val; d++) {
                ds->outputs[i * output_dim_val + d] = labels[i * label_dim + d];
            }
        }
    }

    safe_free((void**)&widths); safe_free((void**)&heights); safe_free((void**)&chs);

    log_info("[图像加载器] 已加载: %zu张图像 (%d×%d×%d → %zu维)",
             num_images, out_w, out_h, out_ch, input_dim);
    return ds;
}

/* ========================================================================
 * WAV音频加载器
 * ======================================================================== */

float* data_load_wav(const char* filepath, int* sample_rate,
                      size_t* num_samples, int* num_channels) {
    if (!filepath) return NULL;

    FILE* fp = fopen(filepath, "rb");
    if (!fp) return NULL;

    /* RIFF头 */
    char riff[4];
    if (fread(riff, 1, 4, fp) != 4) { fclose(fp); return NULL; }
    if (memcmp(riff, "RIFF", 4) != 0) { fclose(fp); return NULL; }

    fseek(fp, 4, SEEK_CUR); /* 跳过文件大小 */

    char wave[4];
    if (fread(wave, 1, 4, fp) != 4) { fclose(fp); return NULL; }
    if (memcmp(wave, "WAVE", 4) != 0) { fclose(fp); return NULL; }

    /* 查找fmt块 */
    unsigned short audio_format = 0, channels = 0;
    unsigned int sample_rate_val = 0;
    unsigned short bits_per_sample = 0;
    int fmt_found = 0;

    while (1) {
        char chunk_id[4];
        if (fread(chunk_id, 1, 4, fp) != 4) break;
        unsigned int chunk_size;
        if (fread(&chunk_size, 4, 1, fp) != 1) break;

        if (memcmp(chunk_id, "fmt ", 4) == 0) {
            unsigned short audio_fmt;
            if (fread(&audio_fmt, 2, 1, fp) != 1) break;
            unsigned short ch;
            if (fread(&ch, 2, 1, fp) != 1) break;
            unsigned int sr;
            if (fread(&sr, 4, 1, fp) != 1) break;
            fseek(fp, 6, SEEK_CUR); /* 跳过字节率和块对齐 */
            unsigned short bps;
            if (fread(&bps, 2, 1, fp) != 1) break;

            audio_format = audio_fmt;
            channels = ch;
            sample_rate_val = sr;
            bits_per_sample = bps;
            fmt_found = 1;

            /* 跳过fmt块剩余部分 */
            unsigned int remaining = chunk_size - 16;
            if (remaining > 0) fseek(fp, remaining, SEEK_CUR);
        } else if (memcmp(chunk_id, "data", 4) == 0) {
            if (!fmt_found) break;

            unsigned int data_size = chunk_size;
            unsigned int bytes_per_sample = bits_per_sample / 8;
            unsigned int total_samples = data_size / bytes_per_sample;

            if (bytes_per_sample == 0 || total_samples == 0) break;

            unsigned char* raw_data = (unsigned char*)safe_malloc(data_size);
            if (!raw_data) break;

            if (fread(raw_data, 1, data_size, fp) != data_size) {
                safe_free((void**)&raw_data); break;
            }
            fclose(fp);

            size_t num_s = (size_t)(total_samples / channels);
            float* samples = (float*)safe_malloc(num_s * channels * sizeof(float));
            if (!samples) { safe_free((void**)&raw_data); return NULL; }

            if (bits_per_sample == 16) {
                for (size_t i = 0; i < num_s * channels; i++) {
                    short s = (short)(raw_data[i * 2] | (raw_data[i * 2 + 1] << 8));
                    samples[i] = s / 32768.0f;
                }
            } else if (bits_per_sample == 8) {
                for (size_t i = 0; i < num_s * channels; i++) {
                    samples[i] = (raw_data[i] - 128) / 128.0f;
                }
            } else if (bits_per_sample == 24) {
                for (size_t i = 0; i < num_s * channels; i++) {
                    int s = raw_data[i * 3] | (raw_data[i * 3 + 1] << 8) | (raw_data[i * 3 + 2] << 16);
                    if (s & 0x800000) s |= ~0xFFFFFF;
                    samples[i] = s / 8388608.0f;
                }
            } else if (bits_per_sample == 32) {
                for (size_t i = 0; i < num_s * channels; i++) {
                    int s = *(int*)(raw_data + i * 4);
                    samples[i] = s / 2147483648.0f;
                }
            } else {
                safe_free((void**)&raw_data);
                safe_free((void**)&samples);
                return NULL;
            }

            safe_free((void**)&raw_data);

            if (sample_rate) *sample_rate = (int)sample_rate_val;
            if (num_samples) *num_samples = num_s;
            if (num_channels) *num_channels = channels;
            return samples;
        } else {
            fseek(fp, chunk_size, SEEK_CUR);
        }
    }

    fclose(fp);
    return NULL;
}

TrainingDataset* data_load_audio_wav(const char* const* filepaths, size_t num_files,
                                      const float* labels, size_t label_dim,
                                      int target_sample_rate, size_t fixed_length) {
    if (!filepaths || num_files == 0) return NULL;

    /* 确定特征长度 */
    size_t feature_len = fixed_length;
    if (feature_len == 0) {
        size_t max_len = 0;
        for (size_t i = 0; i < num_files; i++) {
            int sr = 0; size_t ns = 0; int nc = 0;
            float* samples = data_load_wav(filepaths[i], &sr, &ns, &nc);
            if (samples) {
                /* 如果指定了目标采样率，则调整长度 */
                size_t effective_len = ns;
                if (target_sample_rate > 0 && sr > 0 && sr != target_sample_rate) {
                    effective_len = (size_t)((double)ns * target_sample_rate / sr);
                }
                if (effective_len > max_len) max_len = effective_len;
                safe_free((void**)&samples);
            }
        }
        if (max_len == 0) return NULL;
        feature_len = max_len;
    }

    size_t input_dim = feature_len;
    size_t output_dim_val = label_dim > 0 ? label_dim : 1;

    TrainingDataset* ds = dataset_create("audio_dataset", num_files, input_dim, output_dim_val);
    if (!ds) return NULL;

    for (size_t i = 0; i < num_files; i++) {
        int sr = 0; size_t ns = 0; int nc = 0;
        float* samples = data_load_wav(filepaths[i], &sr, &ns, &nc);
        if (!samples) continue;

        /* 多声道混音为单声道 */
        float* mono = samples;
        size_t mono_len = ns;
        if (nc > 1) {
            mono = (float*)safe_calloc(ns, sizeof(float));
            if (mono) {
                for (size_t s = 0; s < ns; s++) {
                    for (int c = 0; c < nc; c++) {
                        mono[s] += samples[s * nc + c];
                    }
                    mono[s] /= nc;
                }
            }
            safe_free((void**)&samples);
            if (!mono) continue;
            mono_len = ns;
        }

        /* 采样率转换（简单最近邻重采样） */
        float* resampled = mono;
        size_t resampled_len = mono_len;
        if (target_sample_rate > 0 && sr > 0 && sr != target_sample_rate) {
            resampled_len = (size_t)((double)mono_len * target_sample_rate / sr);
            if (resampled_len > 0) {
                resampled = (float*)safe_malloc(resampled_len * sizeof(float));
                if (resampled) {
                    for (size_t j = 0; j < resampled_len; j++) {
                        double src_idx = (double)j * sr / target_sample_rate;
                        size_t idx = (src_idx < (double)mono_len - 1) ? (size_t)src_idx : (mono_len - 1);
                        resampled[j] = mono[idx];
                    }
                } else {
                    resampled = mono;
                    resampled_len = mono_len;
                }
            }
            if (resampled != mono) safe_free((void**)&mono);
        }

        /* 截断或填充到固定长度 */
        float* dst_row = ds->inputs + i * input_dim;
        size_t copy_len = resampled_len < feature_len ? resampled_len : feature_len;
        memcpy(dst_row, resampled, copy_len * sizeof(float));
        if (resampled_len < feature_len) {
            memset(dst_row + resampled_len, 0, (feature_len - resampled_len) * sizeof(float));
        }

        if (resampled != mono) safe_free((void**)&resampled);
        else if (resampled != samples) safe_free((void**)&resampled);
        else safe_free((void**)&samples);
    }

    if (labels && label_dim > 0) {
        for (size_t i = 0; i < num_files; i++) {
            for (size_t d = 0; d < label_dim && d < output_dim_val; d++) {
                ds->outputs[i * output_dim_val + d] = labels[i * label_dim + d];
            }
        }
    }

    log_info("[音频加载器] 已加载: %zu个音频文件 (特征长度=%zu, %zu→%zu维)",
             num_files, feature_len, input_dim, output_dim_val);
    return ds;
}
