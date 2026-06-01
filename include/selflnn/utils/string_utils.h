/**
 * @file string_utils.h
 * @brief 字符串工具接口
 * 
 * 字符串操作、编码转换、解析等工具函数。
 */

#ifndef SELFLNN_STRING_UTILS_H
#define SELFLNN_STRING_UTILS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 字符串编码类型
 */
typedef enum {
    STRING_ENCODING_UTF8 = 0,      /**< UTF-8编码 */
    STRING_ENCODING_UTF16 = 1,     /**< UTF-16编码 */
    STRING_ENCODING_UTF32 = 2,     /**< UTF-32编码 */
    STRING_ENCODING_ASCII = 3,     /**< ASCII编码 */
    STRING_ENCODING_GBK = 4,       /**< GBK编码 */
    STRING_ENCODING_GB2312 = 5     /**< GB2312编码 */
} StringEncoding;

/**
 * @brief 字符串比较选项
 */
typedef enum {
    STRING_COMPARE_CASE_SENSITIVE = 0,     /**< 区分大小写 */
    STRING_COMPARE_CASE_INSENSITIVE = 1    /**< 不区分大小写 */
} StringCompareOption;

/**
 * @brief 字符串分割选项
 */
typedef enum {
    STRING_SPLIT_KEEP_EMPTY = 0,           /**< 保留空字符串 */
    STRING_SPLIT_SKIP_EMPTY = 1            /**< 跳过空字符串 */
} StringSplitOption;

/**
 * @brief 字符串替换选项
 */
typedef enum {
    STRING_REPLACE_ALL = 0,                /**< 替换所有匹配 */
    STRING_REPLACE_FIRST = 1,              /**< 只替换第一个匹配 */
    STRING_REPLACE_LAST = 2                /**< 只替换最后一个匹配 */
} StringReplaceOption;

/**
 * @brief 动态字符串结构
 */
typedef struct {
    char* data;            /**< 字符串数据 */
    size_t length;         /**< 当前长度（不包括终止符） */
    size_t capacity;       /**< 分配容量（包括终止符） */
} DynamicString;

/* 基础字符串操作 */

/**
 * @brief 安全复制字符串
 * 
 * @param dest 目标缓冲区
 * @param src 源字符串
 * @param dest_size 目标缓冲区大小
 * @return size_t 复制的字符数（不包括终止符）
 */
size_t string_copy_safe(char* dest, const char* src, size_t dest_size);

/**
 * @brief 安全连接字符串
 * 
 * @param dest 目标缓冲区
 * @param src 源字符串
 * @param dest_size 目标缓冲区大小
 * @return size_t 连接后的总长度（不包括终止符）
 */
size_t string_concat_safe(char* dest, const char* src, size_t dest_size);

/**
 * @brief 比较字符串
 * 
 * @param str1 字符串1
 * @param str2 字符串2
 * @param option 比较选项
 * @return int 相等返回0，str1<str2返回负数，str1>str2返回正数
 */
int string_compare(const char* str1, const char* str2, StringCompareOption option);

/**
 * @brief 查找子字符串
 * 
 * @param haystack 主字符串
 * @param needle 子字符串
 * @param case_sensitive 是否区分大小写
 * @return const char* 找到的位置，未找到返回NULL
 */
const char* string_find(const char* haystack, const char* needle, int case_sensitive);

/**
 * @brief 反向查找子字符串
 * 
 * @param haystack 主字符串
 * @param needle 子字符串
 * @param case_sensitive 是否区分大小写
 * @return const char* 找到的位置，未找到返回NULL
 */
const char* string_find_reverse(const char* haystack, const char* needle, int case_sensitive);

/**
 * @brief 判断字符串是否以指定前缀开头
 * 
 * @param str 字符串
 * @param prefix 前缀
 * @param case_sensitive 是否区分大小写
 * @return int 是前缀返回1，否则返回0
 */
int string_starts_with(const char* str, const char* prefix, int case_sensitive);

/**
 * @brief 判断字符串是否以指定后缀结尾
 * 
 * @param str 字符串
 * @param suffix 后缀
 * @param case_sensitive 是否区分大小写
 * @return int 是后缀返回1，否则返回0
 */
int string_ends_with(const char* str, const char* suffix, int case_sensitive);

/* 字符串转换 */

/**
 * @brief 转换为小写
 * 
 * @param str 字符串（原地修改）
 */
void string_to_lower(char* str);

/**
 * @brief 转换为大写
 * 
 * @param str 字符串（原地修改）
 */
void string_to_upper(char* str);

/**
 * @brief 去除首尾空白字符
 * 
 * @param str 字符串（原地修改）
 * @return char* 处理后的字符串
 */
char* string_trim(char* str);

/**
 * @brief 去除左侧空白字符
 * 
 * @param str 字符串（原地修改）
 * @return char* 处理后的字符串
 */
char* string_trim_left(char* str);

/**
 * @brief 去除右侧空白字符
 * 
 * @param str 字符串（原地修改）
 * @return char* 处理后的字符串
 */
char* string_trim_right(char* str);

/* 字符串分割与连接 */

/**
 * @brief 分割字符串
 * 
 * @param str 输入字符串
 * @param delimiter 分隔符
 * @param option 分割选项
 * @param count 输出分割数量
 * @return char** 分割结果数组，调用者负责释放
 */
char** string_split(const char* str, const char* delimiter, 
                   StringSplitOption option, size_t* count);

/**
 * @brief 释放分割结果
 * 
 * @param parts 分割结果数组
 * @param count 分割数量
 */
void string_split_free(char** parts, size_t count);

/**
 * @brief 连接字符串数组
 * 
 * @param parts 字符串数组
 * @param count 字符串数量
 * @param delimiter 分隔符
 * @return char* 连接后的字符串，调用者负责释放
 */
char* string_join(const char** parts, size_t count, const char* delimiter);

/* 字符串替换 */

/**
 * @brief 替换字符串中的子串
 * 
 * @param str 输入字符串
 * @param old_substr 旧子串
 * @param new_substr 新子串
 * @param option 替换选项
 * @return char* 替换后的字符串，调用者负责释放
 */
char* string_replace(const char* str, const char* old_substr, 
                    const char* new_substr, StringReplaceOption option);

/* 动态字符串操作 */

/**
 * @brief 创建动态字符串
 * 
 * @param initial_capacity 初始容量
 * @return DynamicString* 动态字符串句柄
 */
DynamicString* string_dynamic_create(size_t initial_capacity);

/**
 * @brief 从C字符串创建动态字符串
 * 
 * @param str C字符串
 * @return DynamicString* 动态字符串句柄
 */
DynamicString* string_dynamic_from_cstr(const char* str);

/**
 * @brief 释放动态字符串
 * 
 * @param ds 动态字符串句柄
 */
void string_dynamic_free(DynamicString* ds);

/**
 * @brief 清空动态字符串
 * 
 * @param ds 动态字符串句柄
 */
void string_dynamic_clear(DynamicString* ds);

/**
 * @brief 追加字符串到动态字符串
 * 
 * @param ds 动态字符串句柄
 * @param str 要追加的字符串
 * @return int 成功返回0，失败返回-1
 */
int string_dynamic_append(DynamicString* ds, const char* str);

/**
 * @brief 追加字符到动态字符串
 * 
 * @param ds 动态字符串句柄
 * @param ch 要追加的字符
 * @return int 成功返回0，失败返回-1
 */
int string_dynamic_append_char(DynamicString* ds, char ch);

/**
 * @brief 获取动态字符串的C字符串
 * 
 * @param ds 动态字符串句柄
 * @return const char* C字符串
 */
const char* string_dynamic_cstr(const DynamicString* ds);

/**
 * @brief 获取动态字符串长度
 * 
 * @param ds 动态字符串句柄
 * @return size_t 字符串长度
 */
size_t string_dynamic_length(const DynamicString* ds);

/**
 * @brief 获取动态字符串容量
 * 
 * @param ds 动态字符串句柄
 * @return size_t 字符串容量
 */
size_t string_dynamic_capacity(const DynamicString* ds);

/* 编码转换 */

/**
 * @brief 转换字符串编码
 * 
 * @param src 源字符串
 * @param src_encoding 源编码
 * @param dst_encoding 目标编码
 * @param dst_size 目标缓冲区大小
 * @param dst 目标缓冲区
 * @return size_t 转换后的字符数（不包括终止符）
 */
size_t string_convert_encoding(const char* src, StringEncoding src_encoding,
                              StringEncoding dst_encoding,
                              size_t dst_size, char* dst);

/**
 * @brief UTF-8到UTF-16转换
 * 
 * @param utf8_str UTF-8字符串
 * @param utf16_str UTF-16缓冲区
 * @param utf16_size UTF-16缓冲区大小（以16位字符计）
 * @return size_t 转换后的UTF-16字符数
 */
size_t string_utf8_to_utf16(const char* utf8_str, unsigned short* utf16_str, size_t utf16_size);

/**
 * @brief UTF-16到UTF-8转换
 * 
 * @param utf16_str UTF-16字符串
 * @param utf16_len UTF-16字符串长度（以16位字符计）
 * @param utf8_str UTF-8缓冲区
 * @param utf8_size UTF-8缓冲区大小
 * @return size_t 转换后的UTF-8字节数
 */
size_t string_utf16_to_utf8(const unsigned short* utf16_str, size_t utf16_len,
                           char* utf8_str, size_t utf8_size);

/* 格式化字符串 */

/**
 * @brief 安全格式化字符串
 * 
 * @param buffer 输出缓冲区
 * @param buffer_size 缓冲区大小
 * @param format 格式字符串
 * @param ... 参数
 * @return int 写入的字符数（不包括终止符）
 */
int string_format_safe(char* buffer, size_t buffer_size, const char* format, ...);

/**
 * @brief 动态格式化字符串
 * 
 * @param format 格式字符串
 * @param ... 参数
 * @return char* 格式化后的字符串，调用者负责释放
 */
char* string_format_dynamic(const char* format, ...);

/* 字符串检查 */

/**
 * @brief 检查字符串是否只包含数字
 * 
 * @param str 字符串
 * @return int 只包含数字返回1，否则返回0
 */
int string_is_numeric(const char* str);

/**
 * @brief 检查字符串是否只包含字母
 * 
 * @param str 字符串
 * @return int 只包含字母返回1，否则返回0
 */
int string_is_alpha(const char* str);

/**
 * @brief 检查字符串是否只包含字母和数字
 * 
 * @param str 字符串
 * @return int 只包含字母和数字返回1，否则返回0
 */
int string_is_alnum(const char* str);

/**
 * @brief 检查字符串是否只包含空白字符
 * 
 * @param str 字符串
 * @return int 只包含空白字符返回1，否则返回0
 */
int string_is_whitespace(const char* str);

/* 实用函数 */

/**
 * @brief 计算字符串长度（安全版本）
 * 
 * @param str 字符串
 * @param max_len 最大检查长度
 * @return size_t 字符串长度（不超过max_len）
 */
size_t string_length_safe(const char* str, size_t max_len);

/**
 * @brief 复制字符串（分配新内存）
 * 
 * @param str 源字符串
 * @return char* 新字符串，调用者负责释放。NULL输入返回空字符串
 */
char* string_duplicate(const char* str);

/**
 * @brief 复制字符串（分配新内存，可空版本）
 * 
 * 与string_duplicate功能相同，但NULL输入返回NULL而非空字符串，
 * 用于需要NULL传播的场景。
 * 
 * @param str 源字符串
 * @return char* 新字符串，调用者负责释放。NULL输入返回NULL
 */
char* string_duplicate_nullable(const char* str);

/**
 * @brief 复制字符串数组（分配新内存）
 * 
 * @param src 源字符串数组
 * @param count 字符串数量
 * @return char** 新字符串数组，调用者负责释放。失败返回NULL
 */
char** string_duplicate_array(const char** src, size_t count);

/**
 * @brief 释放字符串数组
 * 
 * @param array 字符串数组
 * @param count 字符串数量
 */
void string_array_free(char** array, size_t count);

/**
 * @brief 反转字符串
 * 
 * @param str 字符串（原地修改）
 */
void string_reverse(char* str);

/**
 * @brief 计算字符串哈希值
 * 
 * @param str 字符串
 * @param seed 种子值
 * @return unsigned int 哈希值
 */
unsigned int string_hash(const char* str, unsigned int seed);

/* 安全字符串复制 */
#ifndef safe_strdup
#define safe_strdup(s) ((s) ? _strdup(s) : NULL)
#endif

#ifdef __cplusplus
}
#endif

#endif // SELFLNN_STRING_UTILS_H