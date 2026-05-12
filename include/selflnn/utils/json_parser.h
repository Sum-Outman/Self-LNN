/**
 * @file json_parser.h
 * @brief 纯C RFC 8259 JSON解析器
 *
 * 递归下降JSON解析器，100%纯C实现，零外部依赖。
 * 支持：对象/数组嵌套、字符串转义、Unicode转义(\uXXXX)、
 *       布尔/空值、整数/浮点数、嵌套深度可达64层。
 *
 * 使用示例：
 *   JsonValue* root = json_parse(json_string);
 *   char* val = json_get_string(root, "key");
 *   int num = json_get_int(root, "count");
 *   JsonValue* arr = json_get(root, "items");
 *   for (int i = 0; i < json_array_size(arr); i++)
 *       printf("%s\n", json_array_get(arr, i)->string_val);
 *   json_free(root);
 */

#ifndef SELFLNN_JSON_PARSER_H
#define SELFLNN_JSON_PARSER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define JSON_MAX_DEPTH 64
#define JSON_MAX_STRING_LEN (64 * 1024)

typedef enum {
    JSON_NULL = 0,
    JSON_BOOL,
    JSON_NUMBER,
    JSON_STRING,
    JSON_ARRAY,
    JSON_OBJECT
} JsonType;

typedef struct JsonValue {
    JsonType type;
    union {
        int bool_val;
        double number_val;
        char* string_val;
        struct {
            struct JsonValue** items;
            size_t count;
        } array;
        struct {
            char** keys;
            struct JsonValue** values;
            size_t count;
            size_t capacity;
        } object;
    } data;
} JsonValue;

JsonValue* json_parse(const char* json);
void json_free(JsonValue* v);

JsonValue* json_get(const JsonValue* obj, const char* key);
const char* json_get_string(const JsonValue* obj, const char* key);
double json_get_number(const JsonValue* obj, const char* key);
int json_get_bool(const JsonValue* obj, const char* key);

JsonValue* json_array_get(const JsonValue* arr, size_t index);
size_t json_array_size(const JsonValue* arr);

char* json_to_string(const JsonValue* v);

/**
 * @brief 从JSON字符串中提取指定键对应的字符串值
 * @param json 原始JSON字符串
 * @param key 键名
 * @param buf 输出缓冲区
 * @param buf_size 缓冲区大小
 * @return 成功返回0，失败返回-1
 */
int parse_json_string(const char* json, const char* key, char* buf, size_t buf_size);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_JSON_PARSER_H */
