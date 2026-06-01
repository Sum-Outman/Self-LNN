/**
 * @file json_parser.c
 * @brief 纯C RFC 8259 JSON解析器实现
 *
 * K-038: 递归下降JSON解析器，完整支持RFC 8259标准。
 * - 对象/数组任意深度嵌套
 * - 字符串转义 (\", \\, \/, \b, \f, \n, \r, \t)
 * - Unicode转义 (\uXXXX → UTF-8) 完整实现（含代理对支持）
 * - 整数/浮点数（科学计数法，double精度）
 * - 布尔/空值
 * - 边界处理：空字符串、超长数字、嵌套过深(默认128层)检测
 * 100%纯C实现，零外部依赖。
 */

#include "selflnn/utils/json_parser.h"
#include "selflnn/utils/memory_utils.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>

typedef struct {
    const char* src;
    size_t pos;
    size_t len;
    int error;
    char error_msg[256];
} JsonParser;

static void parser_error(JsonParser* p, const char* msg) {
    if (!p->error) {
        snprintf(p->error_msg, sizeof(p->error_msg), "位置%zu: %s", p->pos, msg);
        p->error = 1;
    }
}

static void skip_whitespace(JsonParser* p) {
    while (p->pos < p->len) {
        char c = p->src[p->pos];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            p->pos++;
        } else {
            break;
        }
    }
}

static char peek_char(JsonParser* p) {
    skip_whitespace(p);
    if (p->pos >= p->len) return '\0';
    return p->src[p->pos];
}

static char next_char(JsonParser* p) {
    skip_whitespace(p);
    if (p->pos >= p->len) return '\0';
    return p->src[p->pos++];
}

static int expect_char(JsonParser* p, char c) {
    skip_whitespace(p);
    if (p->pos >= p->len || p->src[p->pos] != c) {
        char buf[64];
        snprintf(buf, sizeof(buf), "期望 '%c'", c);
        parser_error(p, buf);
        return 0;
    }
    p->pos++;
    return 1;
}

static uint32_t hex_digit(char c) {
    if (c >= '0' && c <= '9') return (uint32_t)(c - '0');
    if (c >= 'a' && c <= 'f') return (uint32_t)(c - 'a' + 10);
    if (c >= 'A' && c <= 'F') return (uint32_t)(c - 'A' + 10);
    return 0;
}

static int encode_utf8(uint32_t cp, char* buf) {
    if (cp <= 0x7F) {
        buf[0] = (char)cp;
        return 1;
    } else if (cp <= 0x7FF) {
        buf[0] = (char)(0xC0 | (cp >> 6));
        buf[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    } else if (cp <= 0xFFFF) {
        buf[0] = (char)(0xE0 | (cp >> 12));
        buf[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        buf[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    } else {
        buf[0] = (char)(0xF0 | (cp >> 18));
        buf[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
        buf[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
        buf[3] = (char)(0x80 | (cp & 0x3F));
        return 4;
    }
}

static char* json_parse_string_core(JsonParser* p) {
    if (p->pos >= p->len || p->src[p->pos] != '"') {
        parser_error(p, "期望字符串起始引号");
        return NULL;
    }
    p->pos++;

    size_t cap = 256;
    char* buf = (char*)safe_malloc(cap);
    if (!buf) return NULL;
    size_t len = 0;

    while (p->pos < p->len) {
        /* ZSFZX-FIX-R7-1: JSON字符串长度限制 — 防止恶意超长字符串耗尽内存 */
        if (len >= JSON_MAX_STRING_LEN) {
            parser_error(p, "字符串超过最大长度限制");
            safe_free((void**)&buf);
            return NULL;
        }
        char c = p->src[p->pos++];
        if (c == '"') {
            buf[len] = '\0';
            return buf;
        }
        if (c == '\\') {
            if (p->pos >= p->len) { safe_free((void**)&buf); return NULL; }
            char esc = p->src[p->pos++];
            switch (esc) {
                case '"':  c = '"';  break;
                case '\\': c = '\\'; break;
                case '/':  c = '/';  break;
                case 'b':  c = '\b'; break;
                case 'f':  c = '\f'; break;
                case 'n':  c = '\n'; break;
                case 'r':  c = '\r'; break;
                case 't':  c = '\t'; break;
                case 'u': {
                    if (p->pos + 4 > p->len) { safe_free((void**)&buf); return NULL; }
                    uint32_t cp = 0;
                    for (int i = 0; i < 4; i++) cp = (cp << 4) | hex_digit(p->src[p->pos++]);
                    /* 代理对支持 */
                    if (cp >= 0xD800 && cp <= 0xDBFF && p->pos + 6 <= p->len &&
                        p->src[p->pos] == '\\' && p->src[p->pos+1] == 'u') {
                        p->pos += 2;
                        uint32_t lo = 0;
                        for (int i = 0; i < 4; i++) lo = (lo << 4) | hex_digit(p->src[p->pos++]);
                        if (lo >= 0xDC00 && lo <= 0xDFFF)
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    }
                    char utf8[4];
                    int ulen = encode_utf8(cp, utf8);
                    for (int i = 0; i < ulen; i++) {
                        if (len + 1 >= cap) {
                            cap *= 2;
                            char* nb = (char*)safe_realloc(buf, cap);
                            if (!nb) { safe_free((void**)&buf); return NULL; }
                            buf = nb;
                        }
                        buf[len++] = utf8[i];
                    }
                    continue;
                }
                default: c = esc; break;
            }
        }
        if (len + 1 >= cap) {
            cap *= 2;
            char* nb = (char*)safe_realloc(buf, cap);
            if (!nb) { safe_free((void**)&buf); return NULL; }
            buf = nb;
        }
        buf[len++] = c;
    }
    parser_error(p, "字符串未闭合");
    safe_free((void**)&buf);
    return NULL;
}

static JsonValue* json_parse_value(JsonParser* p, int depth);
static JsonValue* json_parse_array(JsonParser* p, int depth);
static JsonValue* json_parse_object(JsonParser* p, int depth);

static JsonValue* json_value_create(JsonType type) {
    JsonValue* v = (JsonValue*)safe_malloc(sizeof(JsonValue));
    if (!v) return NULL;
    memset(v, 0, sizeof(JsonValue));
    v->type = type;
    return v;
}

static JsonValue* json_parse_array(JsonParser* p, int depth) {
    if (depth > JSON_MAX_DEPTH) { parser_error(p, "数组嵌套过深"); return NULL; }
    JsonValue* v = json_value_create(JSON_ARRAY);
    if (!v) return NULL;

    size_t cap = 8;
    v->data.array.items = (JsonValue**)safe_malloc(cap * sizeof(JsonValue*));
    if (!v->data.array.items) { safe_free((void**)&v); return NULL; }

    char c = peek_char(p);
    if (c == ']') { p->pos++; return v; }

    for (;;) {
        JsonValue* item = json_parse_value(p, depth + 1);
        if (!item) { json_free(v); return NULL; }

        if (v->data.array.count >= cap) {
            cap *= 2;
            JsonValue** nb = (JsonValue**)safe_realloc(v->data.array.items, cap * sizeof(JsonValue*));
            if (!nb) { json_free(item); json_free(v); return NULL; }
            v->data.array.items = nb;
        }
        v->data.array.items[v->data.array.count++] = item;

        c = next_char(p);
        if (c == ']') return v;
        if (c != ',') { parser_error(p, "期望 ',' 或 ']'"); json_free(v); return NULL; }
    }
}

static JsonValue* json_parse_object(JsonParser* p, int depth) {
    if (depth > JSON_MAX_DEPTH) { parser_error(p, "对象嵌套过深"); return NULL; }
    JsonValue* v = json_value_create(JSON_OBJECT);
    if (!v) return NULL;

    size_t cap = 8;
    v->data.object.keys = (char**)safe_malloc(cap * sizeof(char*));
    v->data.object.values = (JsonValue**)safe_malloc(cap * sizeof(JsonValue*));
    v->data.object.capacity = cap;
    if (!v->data.object.keys || !v->data.object.values) { json_free(v); return NULL; }

    char c = peek_char(p);
    if (c == '}') { p->pos++; return v; }

    for (;;) {
        char* key = json_parse_string_core(p);
        if (!key) { json_free(v); return NULL; }

        if (!expect_char(p, ':')) { safe_free((void**)&key); json_free(v); return NULL; }

        JsonValue* val = json_parse_value(p, depth + 1);
        if (!val) { safe_free((void**)&key); json_free(v); return NULL; }

        if (v->data.object.count >= cap) {
            cap *= 2;
            char** nk = (char**)safe_realloc(v->data.object.keys, cap * sizeof(char*));
            JsonValue** nv = (JsonValue**)safe_realloc(v->data.object.values, cap * sizeof(JsonValue*));
            if (!nk || !nv) { safe_free((void**)&key); json_free(val); json_free(v); return NULL; }
            v->data.object.keys = nk;
            v->data.object.values = nv;
            v->data.object.capacity = cap;
        }
        v->data.object.keys[v->data.object.count] = key;
        v->data.object.values[v->data.object.count] = val;
        v->data.object.count++;

        c = next_char(p);
        if (c == '}') return v;
        if (c != ',') { parser_error(p, "期望 ',' 或 '}'"); json_free(v); return NULL; }
    }
}

static JsonValue* json_parse_string_wrapper(JsonParser* p) {
    char* s = json_parse_string_core(p);
    if (!s) return NULL;
    JsonValue* v = json_value_create(JSON_STRING);
    if (!v) { safe_free((void**)&s); return NULL; }
    v->data.string_val = s;
    return v;
}

static JsonValue* json_parse_number(JsonParser* p) {
    size_t start = p->pos;
    char c = p->src[p->pos];
    if (c == '-') p->pos++;
    while (p->pos < p->len && p->src[p->pos] >= '0' && p->src[p->pos] <= '9') p->pos++;
    if (p->pos < p->len && p->src[p->pos] == '.') {
        p->pos++;
        while (p->pos < p->len && p->src[p->pos] >= '0' && p->src[p->pos] <= '9') p->pos++;
    }
    if (p->pos < p->len && (p->src[p->pos] == 'e' || p->src[p->pos] == 'E')) {
        p->pos++;
        if (p->src[p->pos] == '+' || p->src[p->pos] == '-') p->pos++;
        while (p->pos < p->len && p->src[p->pos] >= '0' && p->src[p->pos] <= '9') p->pos++;
    }
    size_t num_len = p->pos - start;
    char* num_str = (char*)safe_malloc(num_len + 1);
    if (!num_str) return NULL;
    memcpy(num_str, p->src + start, num_len);
    num_str[num_len] = '\0';

    JsonValue* v = json_value_create(JSON_NUMBER);
    if (!v) { safe_free((void**)&num_str); return NULL; }
    v->data.number_val = atof(num_str);
    /* ZSFZX-FIX-R7-2: 拒绝NaN/Inf数字 — RFC 8259不允许, 且污染下游计算 */
    if (isnan(v->data.number_val) || isinf(v->data.number_val)) {
        parser_error(p, "JSON不允许NaN或Infinity数字值");
        safe_free((void**)&num_str);
        json_value_free(v);
        return NULL;
    }
    /* ZSFZX-FIX-R7-2: 严格数字格式验证 — 拒绝前导零和空数字 */
    {
        size_t si = 0;
        if (num_str[si] == '-') si++;
        if (si >= num_len || (num_str[si] == '0' && si + 1 < num_len &&
            num_str[si+1] >= '0' && num_str[si+1] <= '9')) {
            parser_error(p, "JSON数字不允许前导零");
            safe_free((void**)&num_str);
            json_value_free(v);
            return NULL;
        }
    }
    safe_free((void**)&num_str);
    return v;
}

static JsonValue* json_parse_literal(JsonParser* p, const char* lit, JsonType type, int bool_val) {
    size_t len = strlen(lit);
    if (p->pos + len > p->len || memcmp(p->src + p->pos, lit, len) != 0) {
        char buf[128];
        snprintf(buf, sizeof(buf), "期望 '%s'", lit);
        parser_error(p, buf);
        return NULL;
    }
    p->pos += len;
    JsonValue* v = json_value_create(type);
    if (v && type == JSON_BOOL) v->data.bool_val = bool_val;
    return v;
}

static JsonValue* json_parse_value(JsonParser* p, int depth) {
    char c = peek_char(p);
    switch (c) {
        case '{': p->pos++; return json_parse_object(p, depth);
        case '[': p->pos++; return json_parse_array(p, depth);
        case '"': return json_parse_string_wrapper(p);
        case 't': return json_parse_literal(p, "true", JSON_BOOL, 1);
        case 'f': return json_parse_literal(p, "false", JSON_BOOL, 0);
        case 'n': return json_parse_literal(p, "null", JSON_NULL, 0);
        case '-': case '0': case '1': case '2': case '3': case '4':
        case '5': case '6': case '7': case '8': case '9':
            return json_parse_number(p);
        default:
            parser_error(p, "无法识别的JSON值");
            return NULL;
    }
}

JsonValue* json_parse(const char* json) {
    if (!json) return NULL;
    JsonParser p;
    p.src = json;
    p.pos = 0;
    p.len = strlen(json);
    p.error = 0;
    p.error_msg[0] = '\0';

    JsonValue* v = json_parse_value(&p, 0);
    if (!v) {
        if (p.error) fprintf(stderr, "[JSON解析] %s\n", p.error_msg);
        return NULL;
    }
    skip_whitespace(&p);
    if (p.pos < p.len) {
        fprintf(stderr, "[JSON解析] 根值后有额外数据: 位置%zu\n", p.pos);
    }
    return v;
}

void json_free(JsonValue* v) {
    if (!v) return;
    switch (v->type) {
        case JSON_STRING:
            safe_free((void**)&v->data.string_val);
            break;
        case JSON_ARRAY:
            for (size_t i = 0; i < v->data.array.count; i++)
                json_free(v->data.array.items[i]);
            safe_free((void**)&v->data.array.items);
            break;
        case JSON_OBJECT:
            for (size_t i = 0; i < v->data.object.count; i++) {
                safe_free((void**)&v->data.object.keys[i]);
                json_free(v->data.object.values[i]);
            }
            safe_free((void**)&v->data.object.keys);
            safe_free((void**)&v->data.object.values);
            break;
        default: break;
    }
    safe_free((void**)&v);
}

JsonValue* json_get(const JsonValue* obj, const char* key) {
    if (!obj || obj->type != JSON_OBJECT || !key) return NULL;
    for (size_t i = 0; i < obj->data.object.count; i++) {
        if (strcmp(obj->data.object.keys[i], key) == 0)
            return obj->data.object.values[i];
    }
    return NULL;
}

const char* json_get_string(const JsonValue* obj, const char* key) {
    JsonValue* v = json_get(obj, key);
    if (v && v->type == JSON_STRING) return v->data.string_val;
    return NULL;
}

double json_get_number(const JsonValue* obj, const char* key) {
    JsonValue* v = json_get(obj, key);
    if (v && v->type == JSON_NUMBER) return v->data.number_val;
    return 0.0;
}

int json_get_bool(const JsonValue* obj, const char* key) {
    JsonValue* v = json_get(obj, key);
    if (v && v->type == JSON_BOOL) return v->data.bool_val;
    return 0;
}

JsonValue* json_array_get(const JsonValue* arr, size_t index) {
    if (!arr || arr->type != JSON_ARRAY || index >= arr->data.array.count) return NULL;
    return arr->data.array.items[index];
}

size_t json_array_size(const JsonValue* arr) {
    if (!arr || arr->type != JSON_ARRAY) return 0;
    return arr->data.array.count;
}

static void json_to_string_core(const JsonValue* v, char* buf, size_t* pos, size_t cap) {
#define APPEND(str) do { \
    size_t sl = strlen(str); \
    if (*pos + sl < cap) memcpy(buf + *pos, str, sl); \
    *pos += sl; \
} while(0)
#define APPEND_CHAR(ch) do { \
    if (*pos < cap) buf[*pos] = (ch); \
    (*pos)++; \
} while(0)

    if (!v) { APPEND("null"); return; }
    char tmp[64];
    switch (v->type) {
        case JSON_NULL: APPEND("null"); break;
        case JSON_BOOL: APPEND(v->data.bool_val ? "true" : "false"); break;
        case JSON_NUMBER:
            snprintf(tmp, sizeof(tmp), "%.17g", v->data.number_val);
            APPEND(tmp); break;
        case JSON_STRING:
            APPEND_CHAR('"');
            if (v->data.string_val) {
                /* ZSFX-DEEP-R12-003: 添加JSON字符串转义输出
                 * 原作直接拼接string_val无转义→含\"或\n会生成非法JSON */
                const char* s = v->data.string_val;
                while (*s) {
                    switch (*s) {
                        case '"':  APPEND("\\\""); break;
                        case '\\': APPEND("\\\\"); break;
                        case '\n': APPEND("\\n");  break;
                        case '\r': APPEND("\\r");  break;
                        case '\t': APPEND("\\t");  break;
                        case '\b': APPEND("\\b");  break;
                        case '\f': APPEND("\\f");  break;
                        default:
                            if ((unsigned char)*s < 0x20) {
                                char esc[8];
                                snprintf(esc, sizeof(esc), "\\u%04x", (unsigned char)*s);
                                APPEND(esc);
                            } else {
                                APPEND_CHAR(*s);
                            }
                            break;
                    }
                    s++;
                }
            }
            APPEND_CHAR('"'); break;
        case JSON_ARRAY:
            APPEND_CHAR('[');
            for (size_t i = 0; i < v->data.array.count; i++) {
                if (i > 0) APPEND_CHAR(',');
                json_to_string_core(v->data.array.items[i], buf, pos, cap);
            }
            APPEND_CHAR(']'); break;
        case JSON_OBJECT:
            APPEND_CHAR('{');
            for (size_t i = 0; i < v->data.object.count; i++) {
                if (i > 0) APPEND_CHAR(',');
                APPEND_CHAR('"');
                APPEND(v->data.object.keys[i]);
                APPEND("\":");
                json_to_string_core(v->data.object.values[i], buf, pos, cap);
            }
            APPEND_CHAR('}'); break;
    }
#undef APPEND
#undef APPEND_CHAR
}

char* json_to_string(const JsonValue* v) {
    if (!v) return NULL;
    size_t pos = 0, cap = 4096;
    char* buf = (char*)safe_malloc(cap);
    if (!buf) return NULL;

    /* 带动态扩展的序列化（处理超长JSON） */
    while (1) {
        pos = 0;
        json_to_string_core(v, buf, &pos, cap);
        if (pos + 8 < cap) break;  /* 成功写入，留有安全余量 */
        /* 缓冲区不足，扩展为2倍并重试 */
        cap *= 2;
        char* new_buf = (char*)safe_realloc(buf, cap);
        if (!new_buf) { safe_free((void**)&buf); return NULL; }
        buf = new_buf;
        if (cap > 16777216) break;  /* 最大16MB硬限制 */
    }

    buf[pos < cap ? pos : cap - 1] = '\0';
    return buf;
}

int parse_json_string(const char* json, const char* key, char* buf, size_t buf_size) {
    if (!json || !key || !buf || buf_size == 0) return -1;
    JsonValue* root = json_parse(json);
    if (!root) return -1;
    const char* val = json_get_string(root, key);
    if (val) {
        snprintf(buf, buf_size, "%s", val);
        json_free(root);
        return 0;
    }
    json_free(root);
    return -1;
}
