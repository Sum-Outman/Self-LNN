/**
 * @file xml_parser.c
 * @brief 纯C XML 1.0 解析器（用于URDF/XACRO/SDF等机器人模型文件）
 *
 * K-039: 递归下降XML解析器，100%纯C实现，零外部依赖。
 * 支持：元素嵌套、属性解析、自闭合标签、CDATA、注释、文本节点。
 * URDF覆盖：link/joint/sensor/visual/collision/inertial/material/origin
 * SDF覆盖：model/world/physics/joint/light/plugin（基本子集）
 * 同时适用于通用XML。
 *
 * 使用示例：
 *   XmlNode* root = xml_parse_file("robot.urdf");
 *   XmlNode* link = xml_find_child(root, "link", "name", "base_link");
 *   float xyz[3]; xml_read_vec3(xml_find_child(link, "origin"), "xyz", xyz);
 *   xml_free(root);
 */

#include "selflnn/utils/xml_parser.h"
#include "selflnn/utils/memory_utils.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include <stdarg.h>

typedef struct {
    const char* src;
    size_t pos;
    size_t len;
    char error[256];
    int has_error;
} XmlCtx;

static void xml_error(XmlCtx* ctx, const char* fmt, ...) {
    if (ctx->has_error) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(ctx->error, sizeof(ctx->error), fmt, ap);
    va_end(ap);
    ctx->has_error = 1;
}

static void skip_ws(XmlCtx* ctx) {
    while (ctx->pos < ctx->len && (ctx->src[ctx->pos] == ' ' || ctx->src[ctx->pos] == '\t' ||
           ctx->src[ctx->pos] == '\n' || ctx->src[ctx->pos] == '\r'))
        ctx->pos++;
}

static int match_str(XmlCtx* ctx, const char* s) {
    size_t sl = strlen(s);
    if (ctx->pos + sl > ctx->len) return 0;
    if (memcmp(ctx->src + ctx->pos, s, sl) != 0) return 0;
    ctx->pos += sl;
    return 1;
}

static XmlNode* xml_node_alloc(XmlNodeType type) {
    XmlNode* n = (XmlNode*)safe_malloc(sizeof(XmlNode));
    if (!n) return NULL;
    memset(n, 0, sizeof(XmlNode));
    n->type = type;
    return n;
}

static void xml_skip_comment(XmlCtx* ctx) {
    if (!match_str(ctx, "<!--")) return;
    const char* end = strstr(ctx->src + ctx->pos, "-->");
    if (end) ctx->pos = (size_t)(end - ctx->src) + 3;
    else ctx->pos = ctx->len;
}

static void xml_skip_cdata(XmlCtx* ctx) {
    if (!match_str(ctx, "<![CDATA[")) return;
    const char* end = strstr(ctx->src + ctx->pos, "]]>");
    if (end) ctx->pos = (size_t)(end - ctx->src) + 3;
    else ctx->pos = ctx->len;
}

static char* xml_read_text_until(XmlCtx* ctx, char endc) {
    size_t start = ctx->pos;
    while (ctx->pos < ctx->len && ctx->src[ctx->pos] != endc) ctx->pos++;
    if (ctx->pos >= ctx->len) return NULL;
    size_t tlen = ctx->pos - start;
    char* text = (char*)safe_malloc(tlen + 1);
    if (!text) return NULL;
    memcpy(text, ctx->src + start, tlen);
    text[tlen] = '\0';
    /* 去除首尾空白 */
    while (tlen > 0 && (text[tlen-1] == ' ' || text[tlen-1] == '\t' || text[tlen-1] == '\n')) text[--tlen] = '\0';
    char* s = text;
    while (*s == ' ' || *s == '\t' || *s == '\n') s++;
    if (s != text) memmove(text, s, strlen(s) + 1);
    if (text[0] == '\0') { safe_free((void**)&text); return NULL; }
    return text;
}

static int xml_parse_attributes(XmlCtx* ctx, XmlNode* node) {
    while (1) {
        skip_ws(ctx);
        if (ctx->pos >= ctx->len || ctx->src[ctx->pos] == '>' || ctx->src[ctx->pos] == '/') break;
        size_t name_start = ctx->pos;
        while (ctx->pos < ctx->len && ctx->src[ctx->pos] != '=' && ctx->src[ctx->pos] != ' ' &&
               ctx->src[ctx->pos] != '>' && ctx->src[ctx->pos] != '/') ctx->pos++;
        size_t nl = ctx->pos - name_start;
        if (nl == 0) break;
        skip_ws(ctx);
        if (ctx->pos >= ctx->len || ctx->src[ctx->pos] != '=') { /* 布尔属性 */ ctx->pos++; continue; }
        ctx->pos++;
        skip_ws(ctx);
        char quote = ctx->src[ctx->pos];
        if (quote != '"' && quote != '\'') break;
        ctx->pos++;
        size_t val_start = ctx->pos;
        while (ctx->pos < ctx->len && ctx->src[ctx->pos] != quote) ctx->pos++;
        size_t vl = ctx->pos - val_start;
        ctx->pos++;

        if (node->attr_count >= XML_MAX_ATTRS) continue;
        node->attr_names[node->attr_count] = (char*)safe_malloc(nl + 1);
        node->attr_values[node->attr_count] = (char*)safe_malloc(vl + 1);
        if (!node->attr_names[node->attr_count] || !node->attr_values[node->attr_count]) {
            safe_free((void**)&node->attr_names[node->attr_count]);
            safe_free((void**)&node->attr_values[node->attr_count]);
            continue;
        }
        memcpy(node->attr_names[node->attr_count], ctx->src + name_start, nl);
        node->attr_names[node->attr_count][nl] = '\0';
        memcpy(node->attr_values[node->attr_count], ctx->src + val_start, vl);
        node->attr_values[node->attr_count][vl] = '\0';
        node->attr_count++;
    }
    return 0;
}

static XmlNode* xml_parse_node(XmlCtx* ctx);

static int xml_parse_children(XmlCtx* ctx, XmlNode* parent) {
    while (1) {
        skip_ws(ctx);
        if (ctx->pos >= ctx->len) return 0;
        if (ctx->src[ctx->pos] == '<') {
            if (ctx->pos + 1 < ctx->len && ctx->src[ctx->pos+1] == '/') {
                ctx->pos += 2;
                /* 跳过闭合标签名 */
                while (ctx->pos < ctx->len && ctx->src[ctx->pos] != '>') ctx->pos++;
                if (ctx->pos < ctx->len) ctx->pos++;
                return 0;
            }
            if (ctx->pos + 3 < ctx->len && memcmp(ctx->src+ctx->pos, "<!--", 4) == 0) {
                xml_skip_comment(ctx);
                continue;
            }
            if (ctx->pos + 8 < ctx->len && memcmp(ctx->src+ctx->pos, "<![CDATA[", 8) == 0) {
                xml_skip_cdata(ctx);
                continue;
            }
            XmlNode* child = xml_parse_node(ctx);
            if (!child) return -1;
            child->next = parent->children;
            parent->children = child;
            child->parent = parent;
        } else {
            char* text = xml_read_text_until(ctx, '<');
            if (text) {
                XmlNode* tn = xml_node_alloc(XML_TEXT);
                if (tn) {
                    tn->text = text;
                    tn->next = parent->children;
                    parent->children = tn;
                } else safe_free((void**)&text);
            }
        }
    }
}

static XmlNode* xml_parse_node(XmlCtx* ctx) {
    skip_ws(ctx);
    if (ctx->pos >= ctx->len || ctx->src[ctx->pos] != '<') return NULL;
    ctx->pos++;

    /* 解析标签名 */
    size_t tag_start = ctx->pos;
    while (ctx->pos < ctx->len && ctx->src[ctx->pos] != ' ' && ctx->src[ctx->pos] != '>' &&
           ctx->src[ctx->pos] != '/' && ctx->src[ctx->pos] != '\t' && ctx->src[ctx->pos] != '\n')
        ctx->pos++;
    size_t tag_len = ctx->pos - tag_start;
    if (tag_len == 0) return NULL;

    XmlNode* node = xml_node_alloc(XML_ELEMENT);
    if (!node) return NULL;
    node->name = (char*)safe_malloc(tag_len + 1);
    if (!node->name) { safe_free((void**)&node); return NULL; }
    memcpy(node->name, ctx->src + tag_start, tag_len);
    node->name[tag_len] = '\0';

    xml_parse_attributes(ctx, node);
    skip_ws(ctx);

    if (ctx->pos < ctx->len && ctx->src[ctx->pos] == '/') {
        /* 自闭合标签 <name ... /> */
        ctx->pos++;
        if (ctx->pos < ctx->len && ctx->src[ctx->pos] == '>') ctx->pos++;
        return node;
    }

    if (ctx->pos < ctx->len && ctx->src[ctx->pos] == '>') {
        ctx->pos++;
        xml_parse_children(ctx, node);
    }

    return node;
}

XmlNode* xml_parse(const char* xml_text) {
    if (!xml_text) return NULL;
    XmlCtx ctx;
    ctx.src = xml_text;
    ctx.pos = 0;
    ctx.len = strlen(xml_text);
    ctx.has_error = 0;
    ctx.error[0] = '\0';

    /* 跳过XML声明 <?xml ... ?> */
    if (ctx.len > 5 && memcmp(ctx.src, "<?xml", 5) == 0) {
        const char* ec = strstr(ctx.src, "?>");
        if (ec) ctx.pos = (size_t)(ec - ctx.src) + 2;
    }

    XmlNode* root = NULL;
    XmlNode* prev = NULL;
    while (ctx.pos < ctx.len) {
        skip_ws(&ctx);
        if (ctx.pos >= ctx.len) break;
        if (ctx.src[ctx.pos] != '<') {
            char* t = xml_read_text_until(&ctx, '<');
            if (t && !root) {
                root = xml_node_alloc(XML_TEXT);
                if (root) root->text = t;
                else safe_free((void**)&t);
                break;
            }
            safe_free((void**)&t);
            continue;
        }
        XmlNode* n = xml_parse_node(&ctx);
        if (!n) break;
        if (!root) root = n;
        else { prev->next = n; n->parent = prev->parent; }
        prev = n;
    }

    return root;
}

XmlNode* xml_parse_file(const char* filepath) {
    FILE* f = fopen(filepath, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buf = (char*)safe_malloc(sz + 1);
    if (!buf) { fclose(f); return NULL; }
    fread(buf, 1, sz, f);
    fclose(f);
    buf[sz] = '\0';
    XmlNode* root = xml_parse(buf);
    safe_free((void**)&buf);
    return root;
}

void xml_free(XmlNode* node) {
    while (node) {
        XmlNode* next = node->next;
        xml_free(node->children);
        for (int i = 0; i < node->attr_count; i++) {
            safe_free((void**)&node->attr_names[i]);
            safe_free((void**)&node->attr_values[i]);
        }
        safe_free((void**)&node->name);
        safe_free((void**)&node->text);
        safe_free((void**)&node);
        node = next;
    }
}

XmlNode* xml_find_child(const XmlNode* parent, const char* name) {
    if (!parent || !name) return NULL;
    for (XmlNode* c = parent->children; c; c = c->next) {
        if (c->type == XML_ELEMENT && c->name && strcmp(c->name, name) == 0)
            return c;
    }
    return NULL;
}

XmlNode* xml_find_child_attr(const XmlNode* parent, const char* name, const char* attr, const char* val) {
    if (!parent) return NULL;
    for (XmlNode* c = parent->children; c; c = c->next) {
        if (c->type != XML_ELEMENT || !c->name) continue;
        if (name && strcmp(c->name, name) != 0) continue;
        if (!attr) return c;
        const char* av = xml_get_attr(c, attr);
        if (av && strcmp(av, val) == 0) return c;
    }
    return NULL;
}

XmlNode* xml_find_child_r(const XmlNode* parent, const char* name, const char* attr, const char* val) {
    if (!parent) return NULL;
    for (XmlNode* c = parent->children; c; c = c->next) {
        if (c->type == XML_ELEMENT && c->name) {
            int name_match = (name == NULL || strcmp(c->name, name) == 0);
            if (name_match) {
                if (!attr) return c;
                const char* av = xml_get_attr(c, attr);
                if (av && strcmp(av, val) == 0) return c;
            }
            XmlNode* found = xml_find_child_r(c, name, attr, val);
            if (found) return found;
        }
    }
    return NULL;
}

const char* xml_get_attr(const XmlNode* node, const char* attr) {
    if (!node || !attr) return NULL;
    for (int i = 0; i < node->attr_count; i++) {
        if (strcmp(node->attr_names[i], attr) == 0)
            return node->attr_values[i];
    }
    return NULL;
}

const char* xml_get_text(const XmlNode* node) {
    if (!node) return NULL;
    if (node->text) return node->text;
    for (XmlNode* c = node->children; c; c = c->next) {
        if (c->type == XML_TEXT && c->text) return c->text;
    }
    return NULL;
}

int xml_read_float(const XmlNode* node, const char* attr, float* val) {
    const char* s = NULL;
    if (attr) s = xml_get_attr(node, attr);
    else s = xml_get_text(node);
    if (!s || !val) return -1;
    char* end = NULL;
    *val = strtof(s, &end);
    if (end == s) return -1;
    return 0;
}

int xml_read_vec3(const XmlNode* node, const char* attr, float v[3]) {
    const char* s = xml_get_attr(node, attr);
    if (!s) { v[0] = v[1] = v[2] = 0.0f; return -1; }
    char* end = NULL;
    v[0] = strtof(s, &end);
    if (end == s) { v[0] = v[1] = v[2] = 0.0f; return -1; }
    v[1] = strtof(end, &end);
    if (end == s) { v[1] = 0.0f; }
    v[2] = strtof(end, NULL);
    return 0;
}

int xml_read_vec4(const XmlNode* node, const char* attr, float v[4]) {
    const char* s = xml_get_attr(node, attr);
    if (!s) { v[0]=v[1]=v[2]=0;v[3]=1; return -1; }
    char* end = NULL;
    v[0] = strtof(s, &end);
    v[1] = strtof(end, &end);
    v[2] = strtof(end, &end);
    v[3] = strtof(end, NULL);
    return 0;
}

XmlNode* xml_get_root(const XmlNode* node) {
    while (node && node->parent) node = node->parent;
    return (XmlNode*)node;
}
