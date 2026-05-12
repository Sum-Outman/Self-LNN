/**
 * @file string_utils.c
 * @brief 字符串工具库实现
 * 
 * 字符串操作、编码转换、解析等工具函数。
 */

#include "selflnn/utils/string_utils.h"
#include "selflnn/utils/memory_utils.h"

#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdarg.h>
#include <stdio.h>
#include <wchar.h>
#include <locale.h>
#include <errno.h>
#ifdef _WIN32
#include <windows.h>
#endif

/* ========================= 基础字符串操作 ========================= */

size_t string_copy_safe(char* dest, const char* src, size_t dest_size) {
    if (!dest || !src || dest_size == 0) {
        return 0;
    }
    
    size_t i = 0;
    for (; i < dest_size - 1 && src[i] != '\0'; i++) {
        dest[i] = src[i];
    }
    dest[i] = '\0';
    
    return i;
}

size_t string_concat_safe(char* dest, const char* src, size_t dest_size) {
    if (!dest || !src || dest_size == 0) {
        return 0;
    }
    
    size_t dest_len = strlen(dest);
    if (dest_len >= dest_size) {
        return dest_len;
    }
    
    size_t i = 0;
    for (; i < dest_size - dest_len - 1 && src[i] != '\0'; i++) {
        dest[dest_len + i] = src[i];
    }
    dest[dest_len + i] = '\0';
    
    return dest_len + i;
}

int string_compare(const char* str1, const char* str2, StringCompareOption option) {
    if (!str1 && !str2) return 0;
    if (!str1) return -1;
    if (!str2) return 1;
    
    if (option == STRING_COMPARE_CASE_SENSITIVE) {
        return strcmp(str1, str2);
    } else {
        #ifdef _WIN32
            return _stricmp(str1, str2);
        #else
            return strcasecmp(str1, str2);
        #endif
    }
}

const char* string_find(const char* haystack, const char* needle, int case_sensitive) {
    if (!haystack || !needle) return NULL;
    
    if (case_sensitive) {
        return strstr(haystack, needle);
    } else {

        size_t haystack_len = strlen(haystack);
        size_t needle_len = strlen(needle);
        
        if (needle_len == 0) return haystack;
        if (needle_len > haystack_len) return NULL;
        
        for (size_t i = 0; i <= haystack_len - needle_len; i++) {
            int match = 1;
            for (size_t j = 0; j < needle_len; j++) {
                char h = haystack[i + j];
                char n = needle[j];
                if (tolower((unsigned char)h) != tolower((unsigned char)n)) {
                    match = 0;
                    break;
                }
            }
            if (match) {
                return haystack + i;
            }
        }
        return NULL;
    }
}

const char* string_find_reverse(const char* haystack, const char* needle, int case_sensitive) {
    if (!haystack || !needle) return NULL;
    
    size_t haystack_len = strlen(haystack);
    size_t needle_len = strlen(needle);
    
    if (needle_len == 0) return haystack + haystack_len;
    if (needle_len > haystack_len) return NULL;
    
    for (size_t i = haystack_len - needle_len + 1; i-- > 0; ) {
        int match = 1;
        for (size_t j = 0; j < needle_len; j++) {
            char h = haystack[i + j];
            char n = needle[j];
            if (case_sensitive) {
                if (h != n) {
                    match = 0;
                    break;
                }
            } else {
                if (tolower((unsigned char)h) != tolower((unsigned char)n)) {
                    match = 0;
                    break;
                }
            }
        }
        if (match) {
            return haystack + i;
        }
    }
    return NULL;
}

int string_starts_with(const char* str, const char* prefix, int case_sensitive) {
    if (!str || !prefix) return 0;
    
    size_t str_len = strlen(str);
    size_t prefix_len = strlen(prefix);
    
    if (prefix_len > str_len) return 0;
    
    if (case_sensitive) {
        return strncmp(str, prefix, prefix_len) == 0;
    } else {
        #ifdef _WIN32
            return _strnicmp(str, prefix, prefix_len) == 0;
        #else
            return strncasecmp(str, prefix, prefix_len) == 0;
        #endif
    }
}

int string_ends_with(const char* str, const char* suffix, int case_sensitive) {
    if (!str || !suffix) return 0;
    
    size_t str_len = strlen(str);
    size_t suffix_len = strlen(suffix);
    
    if (suffix_len > str_len) return 0;
    
    const char* str_suffix = str + (str_len - suffix_len);
    
    if (case_sensitive) {
        return strcmp(str_suffix, suffix) == 0;
    } else {
        #ifdef _WIN32
            return _stricmp(str_suffix, suffix) == 0;
        #else
            return strcasecmp(str_suffix, suffix) == 0;
        #endif
    }
}

/* ========================= 字符串转换 ========================= */

void string_to_lower(char* str) {
    if (!str) return;
    
    for (char* p = str; *p; p++) {
        *p = (char)tolower((unsigned char)*p);
    }
}

void string_to_upper(char* str) {
    if (!str) return;
    
    for (char* p = str; *p; p++) {
        *p = (char)toupper((unsigned char)*p);
    }
}

char* string_trim(char* str) {
    if (!str) return NULL;
    
    // 去除左侧空白
    char* start = str;
    while (*start && isspace((unsigned char)*start)) {
        start++;
    }
    
    // 去除右侧空白
    char* end = str + strlen(str) - 1;
    while (end >= start && isspace((unsigned char)*end)) {
        end--;
    }
    
    // 移动字符串
    if (start != str) {
        size_t len = end - start + 1;
        memmove(str, start, len);
        str[len] = '\0';
    } else if (end < start) {
        // 全部是空白
        str[0] = '\0';
    } else {
        // 截断
        str[end - start + 1] = '\0';
    }
    
    return str;
}

char* string_trim_left(char* str) {
    if (!str) return NULL;
    
    char* start = str;
    while (*start && isspace((unsigned char)*start)) {
        start++;
    }
    
    if (start != str) {
        size_t len = strlen(start);
        memmove(str, start, len + 1); // 包括终止符
    }
    
    return str;
}

char* string_trim_right(char* str) {
    if (!str) return NULL;
    
    char* end = str + strlen(str) - 1;
    while (end >= str && isspace((unsigned char)*end)) {
        end--;
    }
    
    *(end + 1) = '\0';
    
    return str;
}

/* ========================= 字符串分割与连接 ========================= */

char** string_split(const char* str, const char* delimiter, 
                   StringSplitOption option, size_t* count) {
    if (!str || !delimiter || !count) {
        return NULL;
    }
    
    size_t delimiter_len = strlen(delimiter);
    if (delimiter_len == 0) {
        // 按字符分割：每个字符作为一个部分
        size_t str_len = strlen(str);
        char** result = (char**)safe_malloc(str_len * sizeof(char*));
        if (!result) return NULL;
        
        size_t part_count = 0;
        for (size_t i = 0; i < str_len; i++) {
            // 为每个字符分配字符串
            result[part_count] = (char*)safe_malloc(2 * sizeof(char));
            if (!result[part_count]) {
                // 清理已分配的内存
                for (size_t j = 0; j < part_count; j++) {
                    safe_free((void**)&result[j]);
                }
                safe_free((void**)&result);
                return NULL;
            }
            result[part_count][0] = str[i];
            result[part_count][1] = '\0';
            part_count++;
        }
        
        *count = part_count;
        return result;
    }
    
    // 第一次遍历：计算分割数量
    size_t num_parts = 0;
    const char* current = str;
    const char* next;
    
    while (*current) {
        next = strstr(current, delimiter);
        if (!next) {
            // 最后一个部分
            if (option == STRING_SPLIT_KEEP_EMPTY || 
                (current[0] != '\0' && !(option == STRING_SPLIT_SKIP_EMPTY && current[0] == '\0'))) {
                num_parts++;
            }
            break;
        }
        
        size_t part_len = next - current;
        if (option == STRING_SPLIT_KEEP_EMPTY || 
            (part_len > 0 && !(option == STRING_SPLIT_SKIP_EMPTY && part_len == 0))) {
            num_parts++;
        }
        
        current = next + delimiter_len;
    }
    
    if (num_parts == 0) {
        *count = 0;
        return NULL;
    }
    
    // 分配内存
    char** parts = (char**)safe_malloc(num_parts * sizeof(char*));
    if (!parts) {
        *count = 0;
        return NULL;
    }
    
    // 第二次遍历：提取子字符串
    current = str;
    size_t part_index = 0;
    
    while (*current && part_index < num_parts) {
        next = strstr(current, delimiter);
        if (!next) {
            next = current + strlen(current);
        }
        
        size_t part_len = next - current;
        
        if (option == STRING_SPLIT_KEEP_EMPTY || 
            (part_len > 0 && !(option == STRING_SPLIT_SKIP_EMPTY && part_len == 0))) {
            // 分配并复制子字符串
            parts[part_index] = (char*)safe_malloc(part_len + 1);
            if (!parts[part_index]) {
                // 内存分配失败，清理已分配的部分
                for (size_t i = 0; i < part_index; i++) {
                    safe_free((void**)&parts[i]);
                }
                safe_free((void**)&parts);
                *count = 0;
                return NULL;
            }
            
            if (part_len > 0) {
                memcpy(parts[part_index], current, part_len);
            }
            parts[part_index][part_len] = '\0';
            part_index++;
        }
        
        if (*next == '\0') {
            break;
        }
        
        current = next + delimiter_len;
    }
    
    *count = num_parts;
    return parts;
}

void string_split_free(char** parts, size_t count) {
    if (!parts) return;
    
    for (size_t i = 0; i < count; i++) {
        safe_free((void**)&parts[i]);
    }
    safe_free((void**)&parts);
}

char* string_join(const char** parts, size_t count, const char* delimiter) {
    if (!parts || count == 0 || !delimiter) {
        char* empty = (char*)safe_malloc(1);
        if (empty) empty[0] = '\0';
        return empty;
    }
    
    // 计算总长度
    size_t total_len = 0;
    size_t delimiter_len = strlen(delimiter);
    
    for (size_t i = 0; i < count; i++) {
        if (parts[i]) {
            total_len += strlen(parts[i]);
        }
        if (i < count - 1) {
            total_len += delimiter_len;
        }
    }
    
    // 分配内存
    char* result = (char*)safe_malloc(total_len + 1);
    if (!result) {
        return NULL;
    }
    
    // 连接字符串
    char* dest = result;
    for (size_t i = 0; i < count; i++) {
        if (parts[i]) {
            size_t part_len = strlen(parts[i]);
            memcpy(dest, parts[i], part_len);
            dest += part_len;
        }
        
        if (i < count - 1) {
            memcpy(dest, delimiter, delimiter_len);
            dest += delimiter_len;
        }
    }
    *dest = '\0';
    
    return result;
}

/* ========================= 字符串替换 ========================= */

char* string_replace(const char* str, const char* old_substr, 
                    const char* new_substr, StringReplaceOption option) {
    if (!str || !old_substr || !new_substr) {
        return string_duplicate(str ? str : "");
    }
    
    size_t str_len = strlen(str);
    size_t old_len = strlen(old_substr);
    size_t new_len = strlen(new_substr);
    
    if (old_len == 0) {
        return string_duplicate(str);
    }
    
    // 计算替换次数
    size_t replace_count = 0;
    const char* search_start = str;
    const char* found;
    
    if (option == STRING_REPLACE_ALL) {
        while ((found = strstr(search_start, old_substr)) != NULL) {
            replace_count++;
            search_start = found + old_len;
        }
    } else {
        // 只替换第一次或最后一次
        replace_count = 1;
    }
    
    if (replace_count == 0) {
        return string_duplicate(str);
    }
    
    // 计算新字符串长度
    size_t new_str_len = str_len + replace_count * (new_len - old_len);
    
    // 分配内存
    char* result = (char*)safe_malloc(new_str_len + 1);
    if (!result) {
        return NULL;
    }
    
    // 执行替换
    if (option == STRING_REPLACE_ALL || option == STRING_REPLACE_FIRST) {
        // 替换所有或第一次
        const char* src = str;
        char* dest = result;
        size_t replacements_done = 0;
        
        while (*src) {
            if ((option == STRING_REPLACE_ALL || replacements_done == 0) &&
                strstr(src, old_substr) == src) {
                // 找到匹配
                memcpy(dest, new_substr, new_len);
                dest += new_len;
                src += old_len;
                replacements_done++;
                
                if (option == STRING_REPLACE_FIRST && replacements_done >= 1) {
                    // 只替换第一次，复制剩余部分
                    size_t remaining_len = strlen(src);
                    memcpy(dest, src, remaining_len);
                    dest += remaining_len;
                    break;
                }
            } else {
                *dest++ = *src++;
            }
        }
        *dest = '\0';
    } else if (option == STRING_REPLACE_LAST) {
        // 替换最后一次
        const char* last_occurrence = NULL;
        const char* search = str;
        
        while ((search = strstr(search, old_substr)) != NULL) {
            last_occurrence = search;
            search += old_len;
        }
        
        if (!last_occurrence) {
            // 没有找到
            memcpy(result, str, str_len + 1);
            return result;
        }
        
        // 复制最后一次出现之前的部分
        size_t before_len = last_occurrence - str;
        memcpy(result, str, before_len);
        
        // 复制新子串
        memcpy(result + before_len, new_substr, new_len);
        
        // 复制剩余部分
        const char* after_old = last_occurrence + old_len;
        size_t after_len = str_len - (after_old - str);
        memcpy(result + before_len + new_len, after_old, after_len + 1); // 包括终止符
    }
    
    return result;
}

/* ========================= 动态字符串操作 ========================= */

DynamicString* string_dynamic_create(size_t initial_capacity) {
    if (initial_capacity == 0) {
        initial_capacity = 16;
    }
    
    DynamicString* ds = (DynamicString*)safe_malloc(sizeof(DynamicString));
    if (!ds) {
        return NULL;
    }
    
    ds->data = (char*)safe_malloc(initial_capacity);
    if (!ds->data) {
        safe_free((void**)&ds);
        return NULL;
    }
    
    ds->data[0] = '\0';
    ds->length = 0;
    ds->capacity = initial_capacity;
    
    return ds;
}

DynamicString* string_dynamic_from_cstr(const char* str) {
    if (!str) {
        return string_dynamic_create(16);
    }
    
    size_t len = strlen(str);
    size_t capacity = len + 1;
    if (capacity < 16) capacity = 16;
    
    DynamicString* ds = string_dynamic_create(capacity);
    if (!ds) {
        return NULL;
    }
    
    memcpy(ds->data, str, len + 1);
    ds->length = len;
    
    return ds;
}

void string_dynamic_free(DynamicString* ds) {
    if (!ds) return;
    
    safe_free((void**)&ds->data);
    safe_free((void**)&ds);
}

void string_dynamic_clear(DynamicString* ds) {
    if (!ds || !ds->data) return;
    
    ds->data[0] = '\0';
    ds->length = 0;
}

int string_dynamic_append(DynamicString* ds, const char* str) {
    if (!ds || !str) return -1;
    
    size_t str_len = strlen(str);
    size_t new_len = ds->length + str_len;
    
    // 检查是否需要扩容
    if (new_len + 1 > ds->capacity) {
        size_t new_capacity = ds->capacity * 2;
        while (new_capacity <= new_len + 1) {
            new_capacity *= 2;
        }
        
        char* new_data = (char*)safe_malloc(new_capacity);
        if (!new_data) {
            return -1;
        }
        
        memcpy(new_data, ds->data, ds->length + 1); // 包括终止符
        safe_free((void**)&ds->data);
        ds->data = new_data;
        ds->capacity = new_capacity;
    }
    
    // 追加字符串
    memcpy(ds->data + ds->length, str, str_len + 1); // 包括终止符
    ds->length = new_len;
    
    return 0;
}

int string_dynamic_append_char(DynamicString* ds, char ch) {
    if (!ds) return -1;
    
    // 检查是否需要扩容
    if (ds->length + 2 > ds->capacity) {
        size_t new_capacity = ds->capacity * 2;
        char* new_data = (char*)safe_malloc(new_capacity);
        if (!new_data) {
            return -1;
        }
        
        memcpy(new_data, ds->data, ds->length + 1); // 包括终止符
        safe_free((void**)&ds->data);
        ds->data = new_data;
        ds->capacity = new_capacity;
    }
    
    // 追加字符
    ds->data[ds->length] = ch;
    ds->data[ds->length + 1] = '\0';
    ds->length++;
    
    return 0;
}

const char* string_dynamic_cstr(const DynamicString* ds) {
    if (!ds || !ds->data) return "";
    return ds->data;
}

size_t string_dynamic_length(const DynamicString* ds) {
    if (!ds) return 0;
    return ds->length;
}

size_t string_dynamic_capacity(const DynamicString* ds) {
    if (!ds) return 0;
    return ds->capacity;
}

/* ========================= 编码转换 ========================= */

size_t string_convert_encoding(const char* src, StringEncoding src_encoding,
                              StringEncoding dst_encoding,
                              size_t dst_size, char* dst) {
    if (!src || !dst || dst_size == 0) {
        return 0;
    }
    
    // 如果源编码和目标编码相同，直接复制
    if (src_encoding == dst_encoding) {
        return string_copy_safe(dst, src, dst_size);
    }
    
    // 支持UTF-8和UTF-16之间的转换
    #ifdef _WIN32
        UINT src_cp = 0;
        UINT dst_cp = 0;
        
        switch (src_encoding) {
            case STRING_ENCODING_UTF8: src_cp = CP_UTF8; break;
            case STRING_ENCODING_UTF16: src_cp = 1200; break; // UTF-16LE
            case STRING_ENCODING_GBK: src_cp = 936; break;
            case STRING_ENCODING_GB2312: src_cp = 936; break;
            case STRING_ENCODING_ASCII: src_cp = CP_ACP; break;
            default: src_cp = CP_ACP; break;
        }
        
        switch (dst_encoding) {
            case STRING_ENCODING_UTF8: dst_cp = CP_UTF8; break;
            case STRING_ENCODING_UTF16: dst_cp = 1200; break;
            case STRING_ENCODING_GBK: dst_cp = 936; break;
            case STRING_ENCODING_GB2312: dst_cp = 936; break;
            case STRING_ENCODING_ASCII: dst_cp = CP_ACP; break;
            default: dst_cp = CP_ACP; break;
        }
        
        // 首先计算所需缓冲区大小
        int src_len = (int)strlen(src);
        int required_size = MultiByteToWideChar(src_cp, 0, src, src_len, NULL, 0);
        if (required_size <= 0) {
            dst[0] = '\0';
            return 0;
        }
        
        wchar_t* wide_buf = (wchar_t*)safe_malloc((required_size + 1) * sizeof(wchar_t));
        if (!wide_buf) {
            dst[0] = '\0';
            return 0;
        }
        
        // 转换为宽字符
        int converted = MultiByteToWideChar(src_cp, 0, src, src_len, wide_buf, required_size);
        wide_buf[converted] = 0;
        
        if (converted <= 0) {
            safe_free((void**)&wide_buf);
            dst[0] = '\0';
            return 0;
        }
        
        // 从宽字符转换到目标编码
        int dst_result = WideCharToMultiByte(dst_cp, 0, wide_buf, converted, dst, (int)dst_size - 1, NULL, NULL);
        safe_free((void**)&wide_buf);
        
        if (dst_result <= 0) {
            dst[0] = '\0';
            return 0;
        }
        
        dst[dst_result] = '\0';
        return (size_t)dst_result;
    #else
        // 非Windows平台：使用libiconv或简单转换
        // 基础转换：支持UTF-8到UTF-16和UTF-16到UTF-8的基本转换
        if (src_encoding == STRING_ENCODING_UTF8 && dst_encoding == STRING_ENCODING_UTF16) {
            // 使用mbstowcs
            size_t src_len = strlen(src);
            size_t max_chars = dst_size / sizeof(unsigned short);
            if (max_chars == 0) {
                dst[0] = '\0';
                return 0;
            }
            
            size_t converted = mbstowcs((wchar_t*)dst, src, max_chars - 1);
            if (converted == (size_t)-1) {
                dst[0] = '\0';
                return 0;
            }
            
            // 确保以空字符结尾
            ((unsigned short*)dst)[converted] = 0;
            return converted * sizeof(unsigned short);
        } else if (src_encoding == STRING_ENCODING_UTF16 && dst_encoding == STRING_ENCODING_UTF8) {
            // 使用wcstombs
            size_t src_len = 0;
            const unsigned short* src_utf16 = (const unsigned short*)src;
            while (src_utf16[src_len] != 0) src_len++;
            
            size_t converted = wcstombs(dst, (const wchar_t*)src_utf16, dst_size - 1);
            if (converted == (size_t)-1) {
                dst[0] = '\0';
                return 0;
            }
            
            dst[converted] = '\0';
            return converted;
        } else {
            // 不支持的编码转换
            dst[0] = '\0';
            return 0;
        }
    #endif
}

size_t string_utf8_to_utf16(const char* utf8_str, unsigned short* utf16_str, size_t utf16_size) {
    if (!utf8_str || !utf16_str || utf16_size == 0) {
        return 0;
    }
    
    #ifdef _WIN32
        // 使用Windows API进行UTF-8到UTF-16转换
        int required_size = MultiByteToWideChar(CP_UTF8, 0, utf8_str, -1, NULL, 0);
        if (required_size <= 0 || (size_t)required_size > utf16_size) {
            utf16_str[0] = 0;
            return 0;
        }
        
        int converted = MultiByteToWideChar(CP_UTF8, 0, utf8_str, -1, (LPWSTR)utf16_str, (int)utf16_size);
        if (converted <= 0) {
            utf16_str[0] = 0;
            return 0;
        }
        
        // 返回字符数（不包括空终止符）
        return (size_t)(converted - 1);
    #else
        // 使用标准C函数mbstowcs
        size_t max_chars = utf16_size - 1; // 预留空终止符
        size_t converted = mbstowcs((wchar_t*)utf16_str, utf8_str, max_chars);
        if (converted == (size_t)-1) {
            utf16_str[0] = 0;
            return 0;
        }
        
        // 确保以空字符结尾
        utf16_str[converted] = 0;
        return converted;
    #endif
}

size_t string_utf16_to_utf8(const unsigned short* utf16_str, size_t utf16_len,
                           char* utf8_str, size_t utf8_size) {
    (void)utf16_len;  // 在某些平台上未使用
    if (!utf16_str || !utf8_str || utf8_size == 0) {
        return 0;
    }
    
    #ifdef _WIN32
        // 使用Windows API进行UTF-16到UTF-8转换
        int required_size = WideCharToMultiByte(CP_UTF8, 0, (LPCWSTR)utf16_str, -1, NULL, 0, NULL, NULL);
        if (required_size <= 0 || (size_t)required_size > utf8_size) {
            utf8_str[0] = '\0';
            return 0;
        }
        
        int converted = WideCharToMultiByte(CP_UTF8, 0, (LPCWSTR)utf16_str, -1, utf8_str, (int)utf8_size, NULL, NULL);
        if (converted <= 0) {
            utf8_str[0] = '\0';
            return 0;
        }
        
        // 返回字节数（不包括空终止符）
        return (size_t)(converted - 1);
    #else
        // 使用标准C函数wcstombs
        size_t max_bytes = utf8_size - 1; // 预留空终止符
        size_t converted = wcstombs(utf8_str, (const wchar_t*)utf16_str, max_bytes);
        if (converted == (size_t)-1) {
            utf8_str[0] = '\0';
            return 0;
        }
        
        // 确保以空字符结尾
        utf8_str[converted] = '\0';
        return converted;
    #endif
}

/* ========================= 格式化字符串 ========================= */

int string_format_safe(char* buffer, size_t buffer_size, const char* format, ...) {
    if (!buffer || !format || buffer_size == 0) {
        return 0;
    }
    
    va_list args;
    va_start(args, format);
    int result = vsnprintf(buffer, buffer_size, format, args);
    va_end(args);
    
    if (result < 0 || (size_t)result >= buffer_size) {
        buffer[buffer_size - 1] = '\0';
        return (int)(buffer_size - 1);
    }
    
    return result;
}

char* string_format_dynamic(const char* format, ...) {
    if (!format) {
        char* empty = (char*)safe_malloc(1);
        if (empty) empty[0] = '\0';
        return empty;
    }
    
    va_list args;
    va_start(args, format);
    
    // 第一次调用：计算所需长度
    va_list args_copy;
    va_copy(args_copy, args);
    int len = vsnprintf(NULL, 0, format, args_copy);
    va_end(args_copy);
    
    if (len < 0) {
        va_end(args);
        return NULL;
    }
    
    // 分配内存
    char* buffer = (char*)safe_malloc((size_t)len + 1);
    if (!buffer) {
        va_end(args);
        return NULL;
    }
    
    // 第二次调用：实际格式化
    vsnprintf(buffer, (size_t)len + 1, format, args);
    va_end(args);
    
    return buffer;
}

/* ========================= 字符串检查 ========================= */

int string_is_numeric(const char* str) {
    if (!str || *str == '\0') {
        return 0;
    }
    
    for (const char* p = str; *p; p++) {
        if (!isdigit((unsigned char)*p)) {
            return 0;
        }
    }
    return 1;
}

int string_is_alpha(const char* str) {
    if (!str || *str == '\0') {
        return 0;
    }
    
    for (const char* p = str; *p; p++) {
        if (!isalpha((unsigned char)*p)) {
            return 0;
        }
    }
    return 1;
}

int string_is_alnum(const char* str) {
    if (!str || *str == '\0') {
        return 0;
    }
    
    for (const char* p = str; *p; p++) {
        if (!isalnum((unsigned char)*p)) {
            return 0;
        }
    }
    return 1;
}

int string_is_whitespace(const char* str) {
    if (!str) {
        return 1;
    }
    
    for (const char* p = str; *p; p++) {
        if (!isspace((unsigned char)*p)) {
            return 0;
        }
    }
    return 1;
}

/* ========================= 实用函数 ========================= */

size_t string_length_safe(const char* str, size_t max_len) {
    if (!str) {
        return 0;
    }
    
    size_t len = 0;
    while (len < max_len && str[len] != '\0') {
        len++;
    }
    return len;
}

char* string_duplicate(const char* str) {
    if (!str) {
        char* empty = (char*)safe_malloc(1);
        if (empty) empty[0] = '\0';
        return empty;
    }
    
    size_t len = strlen(str);
    char* copy = (char*)safe_malloc(len + 1);
    if (!copy) {
        return NULL;
    }
    
    memcpy(copy, str, len + 1);
    return copy;
}

void string_reverse(char* str) {
    if (!str || *str == '\0') {
        return;
    }
    
    size_t len = strlen(str);
    for (size_t i = 0; i < len / 2; i++) {
        char temp = str[i];
        str[i] = str[len - 1 - i];
        str[len - 1 - i] = temp;
    }
}

unsigned int string_hash(const char* str, unsigned int seed) {
    if (!str) {
        return seed;
    }
    
    unsigned int hash = seed;
    while (*str) {
        hash = (hash * 31) + (unsigned char)*str;
        str++;
    }
    return hash;
}

char* string_duplicate_nullable(const char* str) {
    if (!str) return NULL;
    size_t len = strlen(str);
    char* copy = (char*)safe_malloc(len + 1);
    if (!copy) return NULL;
    memcpy(copy, str, len);
    copy[len] = '\0';
    return copy;
}

char** string_duplicate_array(const char** src, size_t count) {
    if (!src || count == 0) return NULL;
    char** dest = (char**)safe_malloc(count * sizeof(char*));
    if (!dest) return NULL;
    for (size_t i = 0; i < count; i++) {
        if (src[i]) {
            dest[i] = string_duplicate_nullable(src[i]);
            if (!dest[i]) {
                for (size_t j = 0; j < i; j++) {
                    safe_free((void**)&dest[j]);
                }
                safe_free((void**)&dest);
                return NULL;
            }
        } else {
            dest[i] = NULL;
        }
    }
    return dest;
}

void string_array_free(char** array, size_t count) {
    if (!array) return;
    for (size_t i = 0; i < count; i++) {
        safe_free((void**)&array[i]);
    }
    safe_free((void**)&array);
}