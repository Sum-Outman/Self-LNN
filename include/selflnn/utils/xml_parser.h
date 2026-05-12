/**
 * @file xml_parser.h
 * @brief 纯C XML 1.0 解析器
 *
 * 递归下降XML解析器，100%纯C实现，零外部依赖。
 * 支持：元素嵌套、属性、自闭合标签、注释、CDATA、文本节点。
 */

#ifndef SELFLNN_XML_PARSER_H
#define SELFLNN_XML_PARSER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define XML_MAX_ATTRS 32

typedef enum { XML_ELEMENT = 0, XML_TEXT = 1 } XmlNodeType;

typedef struct XmlNode {
    XmlNodeType type;
    struct XmlNode* parent;
    struct XmlNode* children;
    struct XmlNode* next;
    char* name;
    char* text;
    char* attr_names[XML_MAX_ATTRS];
    char* attr_values[XML_MAX_ATTRS];
    int attr_count;
} XmlNode;

XmlNode* xml_parse(const char* xml_text);
XmlNode* xml_parse_file(const char* filepath);
void xml_free(XmlNode* node);

XmlNode* xml_find_child(const XmlNode* parent, const char* name);
XmlNode* xml_find_child_attr(const XmlNode* parent, const char* name, const char* attr, const char* val);
XmlNode* xml_find_child_r(const XmlNode* parent, const char* name, const char* attr, const char* val);
const char* xml_get_attr(const XmlNode* node, const char* attr);
const char* xml_get_text(const XmlNode* node);

int xml_read_float(const XmlNode* node, const char* attr, float* val);
int xml_read_vec3(const XmlNode* node, const char* attr, float v[3]);
int xml_read_vec4(const XmlNode* node, const char* attr, float v[4]);

XmlNode* xml_get_root(const XmlNode* node);

#ifdef __cplusplus
}
#endif

#endif
