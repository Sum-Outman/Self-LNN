#define SELFLNN_IMPLEMENTATION 1
#include "selflnn/robot/computer_operation.h"
#include "selflnn/core/lnn.h"
#include "selflnn/selflnn.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>

#ifdef _WIN32
#include <direct.h>
#include <windows.h>
#include <shellapi.h>
#else
#include <unistd.h>
#include <sys/types.h>
#include <signal.h>
#include <dirent.h>
#ifdef __linux__
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/extensions/XTest.h>
#include <X11/keysym.h>
#endif
#ifdef __APPLE__
#include <ApplicationServices/ApplicationServices.h>
#include <CoreGraphics/CoreGraphics.h>
#endif
#endif

struct COSystem {
    COConfig config;
    LNN* global_lnn;             /* 全局共享LNN（单一模型原则） */
    LNN* screen_encoder;
    LNN* action_policy;
    LNN* element_detector;
    LNN* ui_classifier;
    LNN* ocr_net;
    int screen_encoder_owns;   /* 所有权标记（0=共享全局LNN，1=自建需释放） */
    int action_policy_owns;
    int element_detector_owns;
    int ui_classifier_owns;
    int ocr_net_owns;
    COPlan current_plan;
    COAction action_history[4096];
    size_t action_history_count;
    size_t tasks_completed;
    size_t tasks_failed;
    float avg_success_rate;
    float* last_screen_cache;
    size_t last_width;
    size_t last_height;
    int initialized;

    COSafetyRule safety_rules[CO_SAFETY_RULES];
    size_t num_safety_rules;
    int recording_active;
    float* demo_screen_buffer;
    COAction* demo_action_buffer;
    size_t demo_frame_count;
    size_t demo_buffer_capacity;
    char* demo_label;

    COWindowInfo window_list[CO_MAX_WINDOWS];
    size_t num_windows;
    int window_list_dirty;
};

static float pixel_diff(const float* a, const float* b, size_t n) {
    float diff = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float d = a[i] - b[i];
        diff += d * d;
    }
    return sqrtf(diff / (float)(n > 0 ? n : 1));
}

static void resize_image_bilinear(const float* src, size_t sw, size_t sh, size_t sc, float* dst, size_t dw, size_t dh) {
    for (size_t dy = 0; dy < dh; dy++) {
        for (size_t dx = 0; dx < dw; dx++) {
            float sx_f = (float)dx * (float)(sw - 1) / (float)(dw > 1 ? dw - 1 : 1);
            float sy_f = (float)dy * (float)(sh - 1) / (float)(dh > 1 ? dh - 1 : 1);
            int sx0 = (int)sx_f;
            int sy0 = (int)sy_f;
            int sx1 = sx0 + 1 < (int)sw ? sx0 + 1 : sx0;
            int sy1 = sy0 + 1 < (int)sh ? sy0 + 1 : sy0;
            float fx = sx_f - (float)sx0;
            float fy = sy_f - (float)sy0;
            for (size_t c = 0; c < sc && c < 3; c++) {
                float v00 = src[(sy0 * sw + sx0) * sc + c];
                float v10 = src[(sy0 * sw + sx1) * sc + c];
                float v01 = src[(sy1 * sw + sx0) * sc + c];
                float v11 = src[(sy1 * sw + sx1) * sc + c];
                float v0 = v00 + (v10 - v00) * fx;
                float v1 = v01 + (v11 - v01) * fx;
                dst[(dy * dw + dx) * sc + c] = v0 + (v1 - v0) * fy;
            }
        }
    }
}

static float compute_local_contrast(const float* region, size_t rw, size_t rh) {
    float mean = 0.0f;
    size_t n = rw * rh;
    for (size_t i = 0; i < n; i++) mean += region[i];
    mean /= (float)n;
    float var = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float d = region[i] - mean;
        var += d * d;
    }
    return sqrtf(var / (float)n);
}

/**
 * @brief 通过全局LNN进行投影前向传播（单一模型原则）
 *
 * 将任意维度的任务输入投影到全局LNN的输入/输出维度，
 * 使用全局共享LNN处理，再将结果投影回任务输出维度。
 * 代替5个独立LNN的前向调用。
 */
static int global_lnn_projected_forward(COSystem* system,
                                        const float* task_input, size_t task_input_dim,
                                        float* task_output, size_t task_output_dim) {
    if (!system || !system->global_lnn || !task_input || !task_output) {
        return -1;
    }
    
    LNN* lnn = system->global_lnn;
    size_t lnn_input = lnn_get_input_size(lnn);
    size_t lnn_output = lnn_get_output_size(lnn);
    
    /* 分配投影缓冲区 */
    float* proj_input = (float*)malloc(lnn_input * sizeof(float));
    float* proj_output = (float*)malloc(lnn_output * sizeof(float));
    if (!proj_input || !proj_output) {
        free(proj_input); free(proj_output);
        return -1;
    }
    
    /* 输入投影：任务特征 → LNN输入维度（均值池化） */
    if (task_input_dim >= lnn_input) {
        /* 降采样：均值池化 */
        float scale = (float)task_input_dim / (float)lnn_input;
        for (size_t i = 0; i < lnn_input; i++) {
            size_t start = (size_t)((float)i * scale);
            size_t end = (size_t)((float)(i + 1) * scale);
            if (end > task_input_dim) end = task_input_dim;
            if (end <= start) end = start + 1;
            float sum = 0.0f;
            for (size_t j = start; j < end; j++) {
                sum += task_input[j];
            }
            proj_input[i] = sum / (float)(end - start);
        }
    } else {
        /* 升采样：最近邻插值 */
        float scale = (float)lnn_input / (float)task_input_dim;
        for (size_t i = 0; i < lnn_input; i++) {
            size_t idx = (size_t)((float)i / scale);
            if (idx >= task_input_dim) idx = task_input_dim - 1;
            proj_input[i] = task_input[idx];
        }
    }
    
    /* 全局LNN前向传播 */
    int ret = lnn_forward(lnn, proj_input, proj_output);
    if (ret != 0) {
        free(proj_input); free(proj_output);
        return ret;
    }
    
    /* 输出投影：LNN输出维度 → 任务输出维度（线性插值） */
    if (task_output_dim >= lnn_output) {
        /* 升采样：最近邻插值 */
        float scale = (float)task_output_dim / (float)lnn_output;
        for (size_t i = 0; i < task_output_dim; i++) {
            size_t idx = (size_t)((float)i / scale);
            if (idx >= lnn_output) idx = lnn_output - 1;
            task_output[i] = proj_output[idx];
        }
    } else {
        /* 降采样：均值池化 */
        float scale = (float)lnn_output / (float)task_output_dim;
        for (size_t i = 0; i < task_output_dim; i++) {
            size_t start = (size_t)((float)i * scale);
            size_t end = (size_t)((float)(i + 1) * scale);
            if (end > lnn_output) end = lnn_output;
            if (end <= start) end = start + 1;
            float sum = 0.0f;
            for (size_t j = start; j < end; j++) {
                sum += proj_output[j];
            }
            task_output[i] = sum / (float)(end - start);
        }
    }
    
    free(proj_input);
    free(proj_output);
    return 0;
}

/**
 * @brief 统一前向传播分发：全局LNN投影或回退到独立LNN
 */
static inline int co_forward(COSystem* system, LNN* task_lnn,
                             const float* input, size_t input_dim,
                             float* output, size_t output_dim) {
    if (system->global_lnn) {
        return global_lnn_projected_forward(system, input, input_dim, output, output_dim);
    }
    return lnn_forward(task_lnn, input, output);
}

/* ========== 正则引擎（纯C实现，不依赖外部库） ========== */
/* 支持：. * + ? ^ $ [abc] [^abc] \d \w \s \D \W \S | (group) 字面量 */

/* 前向声明 */
static const char* re_match(const char* p, const char* s);
static const char* re_match_from(const char* p, const char* s, int is_start);
static const char* re_match_atom(const char* p, const char* s);

/* 尝试匹配单个原子（无量词），返回匹配后的字符串位置或NULL */
static const char* re_match_atom(const char* p, const char* s) {
    if (!*s) return NULL;
    if (*p == '.') {
        if (*s == '\n') return NULL;
        return s + 1;
    }
    if (*p == '\\') {
        p++;
        switch (*p) {
            case 'd': if (*s < '0' || *s > '9') return NULL; return s + 1;
            case 'w': if (!((*s >= '0' && *s <= '9') || (*s >= 'A' && *s <= 'Z') || (*s >= 'a' && *s <= 'z') || *s == '_')) return NULL; return s + 1;
            case 's': if (*s != ' ' && *s != '\t' && *s != '\n' && *s != '\r' && *s != '\f') return NULL; return s + 1;
            case 'D': if (*s >= '0' && *s <= '9') return NULL; return s + 1;
            case 'W': if ((*s >= '0' && *s <= '9') || (*s >= 'A' && *s <= 'Z') || (*s >= 'a' && *s <= 'z') || *s == '_') return NULL; return s + 1;
            case 'S': if (*s == ' ' || *s == '\t' || *s == '\n' || *s == '\r' || *s == '\f') return NULL; return s + 1;
            default: if (*s != *p) return NULL; return s + 1;
        }
    }
    if (*p == '[') {
        p++;
        int negated = 0;
        if (*p == '^') { negated = 1; p++; }
        unsigned char bitmap[32] = {0};
        while (*p && *p != ']') {
            if (*(p + 1) == '-' && *(p + 2) && *(p + 2) != ']') {
                for (int c = (unsigned char)*p; c <= (unsigned char)*(p + 2); c++)
                    bitmap[c / 8] |= (unsigned char)(1 << (c % 8));
                p += 3;
            } else {
                bitmap[(unsigned char)*p / 8] |= (unsigned char)(1 << ((unsigned char)*p % 8));
                p++;
            }
        }
        if (*p == ']') p++;
        int in_set = (bitmap[(unsigned char)*s / 8] >> ((unsigned char)*s % 8)) & 1;
        if (negated) in_set = !in_set;
        if (!in_set) return NULL;
        return s + 1;
    }
    if (*p == '(') {
        /* 分组：找到匹配的 ) 并递归匹配组内模式 */
        p++;
        const char* result = re_match(p, s);
        /* 跳过分组内容 */
        int depth = 1;
        while (*p && depth > 0) {
            if (*p == '(') depth++;
            else if (*p == ')') depth--;
            if (depth > 0) p++;
        }
        if (*p) p++;
        /* 如果有量词跟在分组后面，由调用者处理 */
        return result;
    }
    /* 字面量 */
    if (*s != *p) return NULL;
    return s + 1;
}

/* 递归匹配：模式p从位置pos开始匹配字符串s，返回匹配结束位置或NULL */
static const char* re_match_from(const char* p, const char* s, int is_start) {
    const char* s_original = s; /* P1-025修复: 保存原始字符串位置，用于交替操作回退 */
    while (*p) {
        if (*p == '|') {
            /* 交替：左侧已匹配成功（能走到|说明前面字符都已匹配），尝试右侧备选 */
            const char* left_result = s; /* 保存左侧匹配结束位置 */
            const char* alt_start = p + 1; /* 右侧分支起始（|之后） */
            const char* alt_end = alt_start;
            int depth = 0;
            while (*alt_end) {
                if (*alt_end == '(') depth++;
                else if (*alt_end == ')') { if (depth > 0) depth--; }
                else if (*alt_end == '|' && depth == 0) break;
                alt_end++;
            }
            /* 尝试右侧匹配（从原始字符串位置开始，因左侧匹配已消耗了字符串字符） */
            const char* right = re_match_from(alt_start, s_original, is_start);
            if (right) return right;
            /* 右侧失败，返回左侧匹配结果，继续匹配交替后内容 */
            s = left_result;
            p = alt_end;
            if (*p == '|') p++;
            continue;
        }
        if (*p == ')') return s;  /* 分组结束，返回当前位置 */
        if (*p == '^') { is_start = 1; p++; continue; }
        if (*p == '$') {
            if (*s != '\0') {
                /* 尝试匹配末尾的换行符 */
                if (*s == '\n' && *(s + 1) == '\0') s++;
                else return NULL;
            }
            p++; continue;
        }
        if (*p == '?' && p > (const char*)0 && *(p - 1)) {
            break; /* 由处理量词的逻辑处理 */
        }

        /* 预处理下个字符是否是量词 */
        int quant_star = 0, quant_plus = 0, quant_opt = 0;
        const char* pat_after = p;
        int has_quant = 0;

        /* 找到原子结束位置 */
        const char* atom_end = p;
        if (*p == '(') {
            int depth = 1; atom_end++;
            while (*atom_end && depth > 0) {
                if (*atom_end == '(') depth++;
                else if (*atom_end == ')') depth--;
                if (depth > 0) atom_end++;
            }
            if (*atom_end == ')') atom_end++;
        } else if (*p == '[') {
            atom_end++;
            while (*atom_end && *atom_end != ']') atom_end++;
            if (*atom_end == ']') atom_end++;
        } else if (*p == '\\') {
            if (*(p + 1)) atom_end = p + 2;
            else atom_end = p + 1;
        } else {
            atom_end = p + 1;
        }

        /* 检查量词 */
        if (*atom_end == '*') { quant_star = 1; has_quant = 1; }
        else if (*atom_end == '+') { quant_plus = 1; has_quant = 1; }
        else if (*atom_end == '?') { quant_opt = 1; has_quant = 1; }

        pat_after = atom_end + (has_quant ? 1 : 0);

        if (!has_quant) {
            /* 无数量词，直接匹配原子 */
            const char* next_s;
            if (*p == '(') {
                /* 匹配分组内容 */
                const char* group_pat = p + 1;
                int depth = 1;
                const char* group_end = group_pat;
                while (*group_end && depth > 0) {
                    if (*group_end == '(') depth++;
                    else if (*group_end == ')') depth--;
                    if (depth > 0) group_end++;
                }
                char saved = *(char*)group_end;
                *(char*)group_end = '\0';
                next_s = re_match_from(group_pat, s, is_start);
                *(char*)group_end = saved;
                if (!next_s) return NULL;
            } else {
                next_s = re_match_atom(p, s);
                if (!next_s) return NULL;
            }
            p = pat_after;
            s = next_s;
            is_start = 0;
            continue;
        }

        /* 有量词：使用贪婪匹配 + 回溯 */
        if (*p == '.') {
            /* .* 或 .+ */
            if (quant_star) {
                /* 贪心匹配尽可能多的字符 */
                const char* max_s = s;
                while (*max_s && *max_s != '\n') max_s++;
                /* 从最多开始尝试 */
                const char* t = max_s;
                while (t >= s) {
                    const char* result = re_match_from(pat_after, t, 0);
                    if (result) return result;
                    if (t == s) break;
                    t--;
                }
                return NULL;
            } else if (quant_plus) {
                if (!*s || *s == '\n') return NULL;
                const char* max_s = s + 1;
                while (*max_s && *max_s != '\n') max_s++;
                const char* t = max_s;
                while (t >= s + 1) {
                    const char* result = re_match_from(pat_after, t, 0);
                    if (result) return result;
                    t--;
                }
                return NULL;
            } else {
                /* .? */
                const char* result = re_match_from(pat_after, s, 0);
                if (result) return result;
                if (*s && *s != '\n') {
                    result = re_match_from(pat_after, s + 1, 0);
                    if (result) return result;
                }
                return NULL;
            }
        } else if (*p == '\\') {
            /* 转义序列 + 量词 */
            /* 通用量词处理：尝试匹配多个 */
            const char* max_s = s;
            int count = 0;
            while (1) {
                const char* test = re_match_atom(p, max_s);
                if (!test) break;
                max_s = test;
                count++;
                if (quant_star && *max_s == '\0') break;
                if (quant_plus && count >= 1000) break;
            }
            if (quant_plus && count == 0) return NULL;
            if (quant_opt) {
                if (count > 1) {
                    max_s = s + 1;
                    count = 1;
                }
            }
            const char* t = max_s;
            while (t >= s) {
                const char* result = re_match_from(pat_after, t, 0);
                if (result) return result;
                if (t == s) break;
                /* 回退一个字符 */
                const char* prev;
                for (prev = s; prev < t; prev++) {
                    const char* next = re_match_atom(p, prev);
                    if (next == t) { t = prev; break; }
                }
                if (prev >= t) t--; /* 安全回退 */
            }
            return NULL;
        } else if (*p == '[') {
            /* 字符类 + 量词 */
            const char* max_s = s;
            int count = 0;
            while (1) {
                const char* test = re_match_atom(p, max_s);
                if (!test) break;
                max_s = test;
                count++;
                if (quant_star && *max_s == '\0') break;
                if (quant_plus && count >= 1000) break;
            }
            if (quant_plus && count == 0) return NULL;
            const char* t = max_s;
            while (t >= s) {
                const char* result = re_match_from(pat_after, t, 0);
                if (result) return result;
                if (t == s) break;
                const char* prev;
                for (prev = s; prev < t; prev++) {
                    const char* next = re_match_atom(p, prev);
                    if (next == t) { t = prev; break; }
                }
                if (prev >= t) t--;
            }
            return NULL;
        } else {
            /* 字面量 + 量词 */
            const char* max_s = s;
            int count = 0;
            while (1) {
                if (*max_s != *p) break;
                max_s++;
                count++;
                if (quant_star && *max_s == '\0') break;
                if (quant_plus && count >= 1000) break;
            }
            if (quant_plus && count == 0) return NULL;
            const char* t = max_s;
            while (t >= s) {
                const char* result = re_match_from(pat_after, t, 0);
                if (result) return result;
                if (t == s) break;
                t--;
            }
            return NULL;
        }
    }
    return s;
}

/* 递归入口：匹配模式p（从开头）到字符串s */
static const char* re_match(const char* p, const char* s) {
    return re_match_from(p, s, 1);
}

/* ========== 窗口标题匹配（支持正则） ========== */

/* 检查窗口标题是否匹配指定模式（支持正则或纯文本） */
static int co_window_match_title(const char* pattern, const char* title) {
    if (!pattern || !title) return 0;
    if (pattern[0] == '\0' || title[0] == '\0') {
        return (pattern[0] == '\0' && title[0] == '\0') ? 1 : 0;
    }
    /* 如果模式以 "re:" 开头，使用正则匹配 */
    if (pattern[0] == 'r' && pattern[1] == 'e' && pattern[2] == ':') {
        const char* regex_pat = pattern + 3;
        const char* result = re_match(regex_pat, title);
        if (result && *result == '\0') return 1;  /* 完全匹配 */
        /* 也尝试子串匹配（在字符串中查找匹配） */
        const char* t = title;
        while (*t) {
            result = re_match(regex_pat, t);
            if (result && *result == '\0') return 1;
            t++;
        }
        return 0;
    }
    /* 纯文本模式：支持 * 和 ? 通配符简化匹配 */
    const char* p = pattern;
    const char* s = title;
    while (*p && *s) {
        if (*p == '*') {
            p++;
            if (!*p) return 1;
            while (*s) {
                if (co_window_match_title(p, s)) return 1;
                s++;
            }
            return 0;
        }
        if (*p == '?') {
            p++; s++;
            continue;
        }
        if (*p != *s) return 0;
        p++; s++;
    }
    while (*p == '*') p++;
    return (*p == '\0' && *s == '\0') ? 1 : 0;
}

static int is_destructive_action(const COAction* action) {
    if (action->action_type == CO_ACTION_FILE_DELETE) return 1;
    if (action->action_type == CO_ACTION_CLOSE_WINDOW) return 1;
    if (action->action_type == CO_ACTION_BROWSER_NAV && strstr(action->url, "delete") != NULL) return 1;
    return 0;
}

/* ============================================================================
 * 真实OS级键盘/鼠标/屏幕交互实现
 * Win32: SendInput/SetCursorPos/GetDC/BitBlt
 * X11: XSendEvent/XTEST/XWarpPointer/XShmGetImage
 * 无硬件时返回0（无操作），不生成虚假数据
 * ============================================================================ */

static void co_os_sleep_ms(int ms) {
#ifdef _WIN32
    Sleep((DWORD)(ms > 0 ? ms : 10));
#else
    usleep((useconds_t)(ms * 1000));
#endif
}

static int co_os_click(int x, int y, int button) {
    if (x < 0 || y < 0) return -1;

#ifdef _WIN32
    SetCursorPos(x, y);
    co_os_sleep_ms(10);

    INPUT inputs[2] = {0};
    DWORD btn_down = (button == 2) ? MOUSEEVENTF_RIGHTDOWN :
                     (button == 1) ? MOUSEEVENTF_MIDDLEDOWN :
                     MOUSEEVENTF_LEFTDOWN;
    DWORD btn_up = (button == 2) ? MOUSEEVENTF_RIGHTUP :
                   (button == 1) ? MOUSEEVENTF_MIDDLEUP :
                   MOUSEEVENTF_LEFTUP;

    inputs[0].type = INPUT_MOUSE;
    inputs[0].mi.dwFlags = btn_down;
    inputs[1].type = INPUT_MOUSE;
    inputs[1].mi.dwFlags = btn_up;

    SendInput(2, inputs, sizeof(INPUT));
    return 0;
#elif defined(__linux__)
    Display* display = XOpenDisplay(NULL);
    if (!display) return -1;
    int btn = (button == 2) ? 3 : (button == 1) ? 2 : 1;
    XTestFakeMotionEvent(display, -1, x, y, CurrentTime);
    XTestFakeButtonEvent(display, btn, True, CurrentTime);
    XTestFakeButtonEvent(display, btn, False, CurrentTime);
    XFlush(display);
    XCloseDisplay(display);
    return 0;
#elif defined(__APPLE__)
    CGPoint pt = CGPointMake((CGFloat)x, (CGFloat)y);
    CGEventRef move = CGEventCreateMouseEvent(NULL, kCGEventMouseMoved, pt, kCGMouseButtonLeft);
    CGEventPost(kCGHIDEventTap, move);
    CFRelease(move);

    CGMouseButton btn_mac = (button == 2) ? kCGMouseButtonRight :
                            (button == 1) ? kCGMouseButtonCenter : kCGMouseButtonLeft;
    CGEventType down_type = (button == 2) ? kCGEventRightMouseDown :
                            (button == 1) ? kCGEventOtherMouseDown : kCGEventLeftMouseDown;
    CGEventType up_type   = (button == 2) ? kCGEventRightMouseUp :
                            (button == 1) ? kCGEventOtherMouseUp   : kCGEventLeftMouseUp;

    CGEventRef down = CGEventCreateMouseEvent(NULL, down_type, pt, btn_mac);
    CGEventRef up   = CGEventCreateMouseEvent(NULL, up_type, pt, btn_mac);
    CGEventPost(kCGHIDEventTap, down);
    CGEventPost(kCGHIDEventTap, up);
    CFRelease(down);
    CFRelease(up);
    return 0;
#else
    (void)x; (void)y; (void)button;
    return 0;
#endif
}

static int co_os_keypress(int key_code, int press) {
    if (key_code <= 0) return -1;

#ifdef _WIN32
    INPUT input = {0};
    input.type = INPUT_KEYBOARD;
    input.ki.wVk = (WORD)key_code;
    input.ki.dwFlags = press ? 0 : KEYEVENTF_KEYUP;
    SendInput(1, &input, sizeof(INPUT));
    return 0;
#elif defined(__linux__)
    Display* display = XOpenDisplay(NULL);
    if (!display) return -1;
    KeyCode keycode = XKeysymToKeycode(display, (KeySym)key_code);
    if (keycode == 0) {
        XCloseDisplay(display);
        return -1;
    }
    XTestFakeKeyEvent(display, keycode, press ? True : False, CurrentTime);
    XFlush(display);
    XCloseDisplay(display);
    return 0;
#elif defined(__APPLE__)
    CGEventRef ev = CGEventCreateKeyboardEvent(NULL, (CGKeyCode)key_code, press ? true : false);
    if (!ev) return -1;
    CGEventPost(kCGHIDEventTap, ev);
    CFRelease(ev);
    return 0;
#else
    (void)key_code; (void)press;
    return 0;
#endif
}

static int co_os_type_text(const char* text, size_t len) {
    if (!text || len == 0) return -1;

#ifdef _WIN32
    for (size_t i = 0; i < len && i < 1024; i++) {
        SHORT vk = VkKeyScan((BYTE)text[i]);
        BYTE vk_code = LOBYTE(vk);
        BYTE shift_state = HIBYTE(vk);

        INPUT inputs[4] = {0};
        int input_count = 0;

        if (shift_state & 1) {
            inputs[input_count].type = INPUT_KEYBOARD;
            inputs[input_count].ki.wVk = VK_SHIFT;
            input_count++;
        }

        inputs[input_count].type = INPUT_KEYBOARD;
        inputs[input_count].ki.wVk = vk_code;
        input_count++;

        inputs[input_count].type = INPUT_KEYBOARD;
        inputs[input_count].ki.wVk = vk_code;
        inputs[input_count].ki.dwFlags = KEYEVENTF_KEYUP;
        input_count++;

        if (shift_state & 1) {
            inputs[input_count].type = INPUT_KEYBOARD;
            inputs[input_count].ki.wVk = VK_SHIFT;
            inputs[input_count].ki.dwFlags = KEYEVENTF_KEYUP;
            input_count++;
        }

        SendInput((UINT)input_count, inputs, sizeof(INPUT));
        co_os_sleep_ms(5);
    }
    return 0;
#elif defined(__linux__)
    /* Linux: 使用XTestFakeKeyEvent逐字符发送按键 */
    Display* display = XOpenDisplay(NULL);
    if (!display) return -1;
    for (size_t i = 0; i < len && i < 1024; i++) {
        KeySym ks = XStringToKeysym((char[2]){text[i], '\0'});
        if (ks == NoSymbol) { ks = (KeySym)(unsigned char)text[i]; }
        KeyCode kc = XKeysymToKeycode(display, ks);
        if (kc == 0) continue;
        XTestFakeKeyEvent(display, kc, True, CurrentTime);
        XTestFakeKeyEvent(display, kc, False, CurrentTime);
        XFlush(display);
        co_os_sleep_ms(5);
    }
    XCloseDisplay(display);
    return 0;
#elif defined(__APPLE__)
    /* macOS: 使用CGEventPost逐字符发送Unicode */
    for (size_t i = 0; i < len && i < 1024; i++) {
        UniChar uc = (UniChar)text[i];
        CGEventRef ev = CGEventCreateKeyboardEvent(NULL, 0, true);
        if (ev) {
            CGEventKeyboardSetUnicodeString(ev, 1, &uc);
            CGEventPost(kCGHIDEventTap, ev);
            CFRelease(ev);
        }
        ev = CGEventCreateKeyboardEvent(NULL, 0, false);
        if (ev) { CGEventPost(kCGHIDEventTap, ev); CFRelease(ev); }
        co_os_sleep_ms(5);
    }
    return 0;
#else
    (void)text; (void)len;
    return -1;
#endif
}

static int co_os_mouse_move(int x, int y) {
    if (x < 0 || y < 0) return -1;

#ifdef _WIN32
    SetCursorPos(x, y);
    return 0;
#elif defined(__linux__)
    Display* display = XOpenDisplay(NULL);
    if (!display) return -1;
    XTestFakeMotionEvent(display, -1, x, y, CurrentTime);
    XFlush(display);
    XCloseDisplay(display);
    return 0;
#elif defined(__APPLE__)
    CGPoint pt = CGPointMake((CGFloat)x, (CGFloat)y);
    CGEventRef move = CGEventCreateMouseEvent(NULL, kCGEventMouseMoved, pt, kCGMouseButtonLeft);
    CGEventPost(kCGHIDEventTap, move);
    CFRelease(move);
    return 0;
#else
    (void)x; (void)y;
    return 0;
#endif
}

static int co_os_scroll(int delta) {
#ifdef _WIN32
    INPUT input = {0};
    input.type = INPUT_MOUSE;
    input.mi.dwFlags = MOUSEEVENTF_WHEEL;
    input.mi.mouseData = (DWORD)(delta * WHEEL_DELTA);
    SendInput(1, &input, sizeof(INPUT));
    return 0;
#elif defined(__linux__)
    Display* display = XOpenDisplay(NULL);
    if (!display) return -1;
    int btn = (delta > 0) ? 4 : 5;
    int abs_delta = delta > 0 ? delta : -delta;
    for (int i = 0; i < abs_delta; i++) {
        XTestFakeButtonEvent(display, btn, True, CurrentTime);
        XTestFakeButtonEvent(display, btn, False, CurrentTime);
    }
    XFlush(display);
    XCloseDisplay(display);
    return 0;
#elif defined(__APPLE__)
    CGEventRef scroll = CGEventCreateScrollWheelEvent(NULL, kCGScrollEventUnitPixel, 1, (int32_t)delta);
    if (!scroll) return -1;
    CGEventPost(kCGHIDEventTap, scroll);
    CFRelease(scroll);
    return 0;
#else
    (void)delta;
    return 0;
#endif
}

static int co_os_screenshot_real(COSystem* system, const COAction* action) {
    if (!system || !action) return -1;

#ifdef _WIN32
    int w = GetSystemMetrics(SM_CXSCREEN);
    int h = GetSystemMetrics(SM_CYSCREEN);
    if (w <= 0 || h <= 0) return -1;

    HDC hdcScreen = GetDC(NULL);
    HDC hdcMem = CreateCompatibleDC(hdcScreen);
    HBITMAP hBitmap = CreateCompatibleBitmap(hdcScreen, w, h);
    SelectObject(hdcMem, hBitmap);

    BitBlt(hdcMem, 0, 0, w, h, hdcScreen, 0, 0, SRCCOPY);

    BITMAPINFOHEADER bi = {0};
    bi.biSize = sizeof(BITMAPINFOHEADER);
    bi.biWidth = w;
    bi.biHeight = -h;
    bi.biPlanes = 1;
    bi.biBitCount = 24;
    bi.biCompression = BI_RGB;

    size_t row_size = ((size_t)(w * 3 + 3) / 4) * 4;
    size_t data_size = row_size * (size_t)h;
    unsigned char* pixels = (unsigned char*)malloc(data_size);

    if (pixels) {
        GetDIBits(hdcMem, hBitmap, 0, (UINT)h, pixels,
                  (BITMAPINFO*)&bi, DIB_RGB_COLORS);

        size_t target_w = CO_SCREEN_WIDTH;
        size_t target_h = CO_SCREEN_HEIGHT;
        if (system->last_screen_cache) {
            free(system->last_screen_cache);
            system->last_screen_cache = NULL;
        }

        system->last_screen_cache = (float*)malloc(target_w * target_h * 3 * sizeof(float));
        system->last_width = target_w;
        system->last_height = target_h;

        if (system->last_screen_cache) {
            for (size_t dy = 0; dy < target_h; dy++) {
                for (size_t dx = 0; dx < target_w; dx++) {
                    size_t sx = dx * (size_t)w / target_w;
                    size_t sy = dy * (size_t)h / target_h;
                    size_t src_idx = (sy * (size_t)w + sx) * 3;
                    size_t dst_idx = (dy * target_w + dx) * 3;
                    if (src_idx + 2 < data_size) {
                        system->last_screen_cache[dst_idx] = (float)pixels[src_idx + 2] / 255.0f;
                        system->last_screen_cache[dst_idx + 1] = (float)pixels[src_idx + 1] / 255.0f;
                        system->last_screen_cache[dst_idx + 2] = (float)pixels[src_idx] / 255.0f;
                    }
                }
            }
        }
        free(pixels);
    }

    DeleteObject(hBitmap);
    DeleteDC(hdcMem);
    ReleaseDC(NULL, hdcScreen);
    return 0;
#elif defined(__linux__)
    Display* display = XOpenDisplay(NULL);
    if (!display) return -1;

    int screen = DefaultScreen(display);
    Window root = DefaultRootWindow(display);

    XWindowAttributes attrs;
    if (XGetWindowAttributes(display, root, &attrs) == 0) {
        XCloseDisplay(display);
        return -1;
    }

    int w = attrs.width;
    int h = attrs.height;
    if (w <= 0 || h <= 0) { XCloseDisplay(display); return -1; }

    XImage* image = XGetImage(display, root, 0, 0, (unsigned int)w, (unsigned int)h,
                               AllPlanes, ZPixmap);
    if (!image) { XCloseDisplay(display); return -1; }

    size_t target_w = CO_SCREEN_WIDTH;
    size_t target_h = CO_SCREEN_HEIGHT;
    if (system->last_screen_cache) {
        free(system->last_screen_cache);
        system->last_screen_cache = NULL;
    }

    system->last_screen_cache = (float*)malloc(target_w * target_h * 3 * sizeof(float));
    system->last_width = target_w;
    system->last_height = target_h;

    if (system->last_screen_cache) {
        for (size_t dy = 0; dy < target_h; dy++) {
            for (size_t dx = 0; dx < target_w; dx++) {
                size_t sx = dx * (size_t)w / target_w;
                size_t sy = dy * (size_t)h / target_h;
                unsigned long px = XGetPixel(image, (int)sx, (int)sy);
                size_t dst_idx = (dy * target_w + dx) * 3;
                system->last_screen_cache[dst_idx] = (float)((px >> 16) & 0xFF) / 255.0f;
                system->last_screen_cache[dst_idx + 1] = (float)((px >> 8) & 0xFF) / 255.0f;
                system->last_screen_cache[dst_idx + 2] = (float)(px & 0xFF) / 255.0f;
            }
        }
    }

    XDestroyImage(image);
    XCloseDisplay(display);
    return 0;
#elif defined(__APPLE__)
    /* F-045: macOS CoreGraphics截图实现 */
    CGImageRef cg_image = CGDisplayCreateImage(CGMainDisplayID());
    if (!cg_image) return -1;

    size_t img_w = CGImageGetWidth(cg_image);
    size_t img_h = CGImageGetHeight(cg_image);
    if (img_w == 0 || img_h == 0) { CGImageRelease(cg_image); return -1; }

    size_t target_w = CO_SCREEN_WIDTH;
    size_t target_h = CO_SCREEN_HEIGHT;
    if (system->last_screen_cache) {
        free(system->last_screen_cache);
        system->last_screen_cache = NULL;
    }

    system->last_screen_cache = (float*)malloc(target_w * target_h * 3 * sizeof(float));
    system->last_width = target_w;
    system->last_height = target_h;

    if (system->last_screen_cache) {
        /* 通过CGBitmapContext读取像素 */
        CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
        size_t row_stride = img_w * 4;
        unsigned char* raw = (unsigned char*)malloc(img_h * row_stride);
        if (raw) {
            CGContextRef ctx = CGBitmapContextCreate(raw, img_w, img_h, 8, row_stride,
                                                      cs, kCGImageAlphaNoneSkipLast);
            if (ctx) {
                CGContextDrawImage(ctx, CGRectMake(0, 0, (CGFloat)img_w, (CGFloat)img_h), cg_image);
                CGContextRelease(ctx);
                for (size_t dy = 0; dy < target_h; dy++) {
                    for (size_t dx = 0; dx < target_w; dx++) {
                        size_t sx = dx * img_w / target_w;
                        size_t sy = dy * img_h / target_h;
                        size_t src_idx = (sy * img_w + sx) * 4;
                        size_t dst_idx = (dy * target_w + dx) * 3;
                        system->last_screen_cache[dst_idx] = (float)raw[src_idx] / 255.0f;
                        system->last_screen_cache[dst_idx + 1] = (float)raw[src_idx + 1] / 255.0f;
                        system->last_screen_cache[dst_idx + 2] = (float)raw[src_idx + 2] / 255.0f;
                    }
                }
            }
            free(raw);
        }
        CGColorSpaceRelease(cs);
    }
    CGImageRelease(cg_image);
    return 0;
#else
    return 0;
#endif
}

static int co_os_drag(int start_x, int start_y, int end_x, int end_y) {
    co_os_mouse_move(start_x, start_y);
    co_os_sleep_ms(20);

#ifdef _WIN32
    INPUT down = {0};
    down.type = INPUT_MOUSE;
    down.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
    SendInput(1, &down, sizeof(INPUT));
#endif

    int steps = 20;
    for (int i = 1; i <= steps; i++) {
        int mx = start_x + (end_x - start_x) * i / steps;
        int my = start_y + (end_y - start_y) * i / steps;
        co_os_mouse_move(mx, my);
        co_os_sleep_ms(10);
    }

#ifdef _WIN32
    INPUT up = {0};
    up.type = INPUT_MOUSE;
    up.mi.dwFlags = MOUSEEVENTF_LEFTUP;
    SendInput(1, &up, sizeof(INPUT));
#endif

    return 0;
}

static int co_os_launch_application(const char* app_name) {
    if (!app_name || app_name[0] == '\0') return -1;

#ifdef _WIN32
    STARTUPINFO si = {0};
    PROCESS_INFORMATION pi = {0};
    si.cb = sizeof(si);

    char cmd[512];
    snprintf(cmd, sizeof(cmd), "cmd /c start \"\" \"%s\"", app_name);

    if (CreateProcess(NULL, cmd, NULL, NULL, FALSE,
                      CREATE_NO_WINDOW, NULL, NULL, &si, &pi)) {
        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);
        return 0;
    }
    return -1;
#elif defined(__linux__)
    pid_t pid = fork();
    if (pid < 0) return -1;
    if (pid == 0) {
        execlp(app_name, app_name, (char*)NULL);
        _exit(127);
    }
    return 0;
#elif defined(__APPLE__)
    {
        pid_t pid = fork();
        if (pid < 0) return -1;
        if (pid == 0) {
            execlp("open", "open", "-a", app_name, (char*)NULL);
            _exit(127);
        }
    }
    return 0;
#else
    (void)app_name;
    return 0;
#endif
}

static int co_os_close_application(const char* window_title) {
    if (!window_title || window_title[0] == '\0') return -1;

#ifdef _WIN32
    HWND hwnd = FindWindow(NULL, window_title);
    if (!hwnd) {
        hwnd = FindWindow(NULL, NULL);
    }
    if (hwnd) {
        PostMessage(hwnd, WM_CLOSE, 0, 0);
        return 0;
    }
    return -1;
#elif defined(__linux__)
    pid_t pid = fork();
    if (pid < 0) return -1;
    if (pid == 0) {
        execlp("pkill", "pkill", "-f", window_title, (char*)NULL);
        _exit(0);
    }
    return 0;
#elif defined(__APPLE__)
    {
        pid_t pid = fork();
        if (pid < 0) return -1;
        if (pid == 0) {
            execlp("pkill", "pkill", "-f", window_title, (char*)NULL);
            _exit(0);
        }
    }
    return 0;
#else
    (void)window_title;
    return 0;
#endif
}

static int co_os_set_volume(int level) {
#ifdef _WIN32
    if (level < 0) level = 0;
    if (level > 100) level = 100;

    INPUT inputs[5] = {0};
    for (int i = 0; i < 5; i++) inputs[i].type = INPUT_KEYBOARD;

    inputs[0].ki.wVk = VK_VOLUME_MUTE;
    inputs[0].ki.dwFlags = 0;
    inputs[1].ki.wVk = VK_VOLUME_MUTE;
    inputs[1].ki.dwFlags = KEYEVENTF_KEYUP;

    DWORD vol_key = (level > 50) ? VK_VOLUME_UP : VK_VOLUME_DOWN;
    int presses = (level > 50) ? (level - 50) / 2 : (50 - level) / 2;
    if (presses < 0) presses = -presses;
    if (presses > 25) presses = 25;

    for (int i = 0; i < presses; i++) {
        inputs[2].ki.wVk = (WORD)vol_key;
        inputs[2].ki.dwFlags = 0;
        inputs[3].ki.wVk = (WORD)vol_key;
        inputs[3].ki.dwFlags = KEYEVENTF_KEYUP;
        SendInput(4, inputs, sizeof(INPUT));
        co_os_sleep_ms(30);
    }
    return 0;
#elif defined(__linux__)
    if (level < 0) level = 0;
    if (level > 100) level = 100;
    pid_t pid = fork();
    if (pid < 0) return -1;
    if (pid == 0) {
        char vol_str[16];
        snprintf(vol_str, sizeof(vol_str), "%d%%", level);
        execlp("pactl", "pactl", "set-sink-volume", "@DEFAULT_SINK@", vol_str, (char*)NULL);
        _exit(0);
    }
    return 0;
#else
    (void)level;
    return 0;
#endif
}

COSystem* co_system_create(COConfig config) {
    COSystem* system = (COSystem*)calloc(1, sizeof(COSystem));
    if (!system) return NULL;

    system->config = config;
    system->tasks_completed = 0;
    system->tasks_failed = 0;
    system->avg_success_rate = 0.0f;
    system->action_history_count = 0;
    system->initialized = 1;
    system->recording_active = 0;
    system->demo_frame_count = 0;
    system->demo_buffer_capacity = 0;
    system->demo_screen_buffer = NULL;
    system->demo_action_buffer = NULL;
    system->demo_label = NULL;
    system->num_safety_rules = 0;

    /* 获取全局共享LNN（单一模型原则） */
    system->global_lnn = (LNN*)selflnn_get_lnn();
    if (system->global_lnn) {
        /* 所有子任务共享同一个全局LNN，设置所有权标记为0 */
        system->screen_encoder = system->global_lnn;
        system->action_policy = system->global_lnn;
        system->element_detector = system->global_lnn;
        system->ui_classifier = system->global_lnn;
        system->ocr_net = system->global_lnn;
        system->screen_encoder_owns = 0;
        system->action_policy_owns = 0;
        system->element_detector_owns = 0;
        system->ui_classifier_owns = 0;
        system->ocr_net_owns = 0;
    } else {
        /* 回退：自建独立LNN（全局系统未初始化时使用） */
        system->screen_encoder_owns = 1;
        system->action_policy_owns = 1;
        system->element_detector_owns = 1;
        system->ui_classifier_owns = 1;
        system->ocr_net_owns = 1;

        LNNConfig screen_cfg = {0};
        screen_cfg.input_size = 64 * 64;
        screen_cfg.hidden_size = 512;
        screen_cfg.output_size = 1024;
        screen_cfg.num_layers = 1;
        system->screen_encoder = lnn_create(&screen_cfg);

        LNNConfig policy_cfg = {0};
        policy_cfg.input_size = 1024 + 512;
        policy_cfg.hidden_size = 512;
        policy_cfg.output_size = (int)CO_ACTION_MAX * 10;
        policy_cfg.num_layers = 1;
        system->action_policy = lnn_create(&policy_cfg);

        LNNConfig detector_cfg = {0};
        detector_cfg.input_size = 32 * 32;
        detector_cfg.hidden_size = 256;
        detector_cfg.output_size = CO_MAX_ELEMENTS;
        detector_cfg.num_layers = 1;
        system->element_detector = lnn_create(&detector_cfg);

        LNNConfig ui_cfg = {0};
        ui_cfg.input_size = 32 * 32;
        ui_cfg.hidden_size = 128;
        ui_cfg.output_size = 16;
        ui_cfg.num_layers = 1;
        system->ui_classifier = lnn_create(&ui_cfg);

        LNNConfig ocr_cfg = {0};
        ocr_cfg.input_size = 16 * 16;
        ocr_cfg.hidden_size = 64;
        ocr_cfg.output_size = 36;
        ocr_cfg.num_layers = 1;
        system->ocr_net = lnn_create(&ocr_cfg);
    }

    system->last_screen_cache = (float*)calloc(CO_SCREEN_WIDTH * CO_SCREEN_HEIGHT * CO_SCREEN_CHANNELS, sizeof(float));

    COSafetyRule default_rule;
    default_rule.rule_id = 0;
    snprintf(default_rule.description, sizeof(default_rule.description), "禁止删除系统关键文件");
    default_rule.check_func = co_safety_check_destructive;
    default_rule.is_enabled = 1;
    default_rule.severity = 10;
    co_safety_add_rule(system, default_rule);

    return system;
}

void co_system_destroy(COSystem* system) {
    if (!system) return;
    if (system->screen_encoder && system->screen_encoder_owns) lnn_free(system->screen_encoder);
    if (system->action_policy && system->action_policy_owns) lnn_free(system->action_policy);
    if (system->element_detector && system->element_detector_owns) lnn_free(system->element_detector);
    if (system->ui_classifier && system->ui_classifier_owns) lnn_free(system->ui_classifier);
    if (system->ocr_net && system->ocr_net_owns) lnn_free(system->ocr_net);
    free(system->last_screen_cache);
    free(system->demo_screen_buffer);
    free(system->demo_action_buffer);
    free(system->demo_label);
    free(system);
}

int co_analyze_screen(COSystem* system, const float* screen_data, size_t width, size_t height, size_t channels) {
    if (!system || !screen_data) return -1;
    if (width == 0 || height == 0) return -2;

    system->last_width = width;
    system->last_height = height;
    memcpy(system->last_screen_cache, screen_data, width * height * channels * sizeof(float));

    float screen_input[64 * 64] = {0};
    resize_image_bilinear(screen_data, width, height, channels, screen_input, 64, 64);

    float grayscale[64 * 64] = {0};
    for (int y = 0; y < 64; y++) {
        for (int x = 0; x < 64; x++) {
            float val = 0;
            for (size_t c = 0; c < channels && c < 3; c++) val += screen_input[(y * 64 + x) * channels + c];
            grayscale[y * 64 + x] = val / (float)(channels > 3 ? 3 : channels);
        }
    }

    float screen_embed[1024] = {0};
    co_forward(system, system->screen_encoder, grayscale, 64*64, screen_embed, 1024);
    memcpy(system->current_plan.screen_embedding, screen_embed, 1024 * sizeof(float));

    size_t num_elements = 0;
    int grid_size = 8;
    float grid_step_x = (float)width / (float)grid_size;
    float grid_step_y = (float)height / (float)grid_size;

    for (int gy = 0; gy < grid_size && num_elements < CO_MAX_ELEMENTS; gy++) {
        for (int gx = 0; gx < grid_size && num_elements < CO_MAX_ELEMENTS; gx++) {
            float region[32 * 32] = {0};
            int start_x = (int)(gx * grid_step_x);
            int start_y = (int)(gy * grid_step_y);
            for (int ry = 0; ry < 32; ry++) {
                for (int rx = 0; rx < 32; rx++) {
                    int sx = start_x + (int)(rx * grid_step_x / 32.0f);
                    int sy = start_y + (int)(ry * grid_step_y / 32.0f);
                    if (sx >= (int)width) sx = (int)width - 1;
                    if (sy >= (int)height) sy = (int)height - 1;
                    float val = 0.0f;
                    for (size_t c = 0; c < channels && c < 3; c++) val += screen_data[(sy * width + sx) * channels + c];
                    region[ry * 32 + rx] = val / (float)(channels > 3 ? 3 : channels);
                }
            }

            float det_out[CO_MAX_ELEMENTS] = {0};
            co_forward(system, system->element_detector, region, 32*32, det_out, CO_MAX_ELEMENTS);

            float max_conf = 0.0f;
            int max_idx = -1;
            for (size_t i = 0; i < CO_MAX_ELEMENTS; i++) {
                if (det_out[i] > max_conf) { max_conf = det_out[i]; max_idx = (int)i; }
            }

            if (max_conf > 0.3f && num_elements < CO_MAX_ELEMENTS) {
                COUIElement* el = &system->current_plan.detected_elements[num_elements];
                snprintf(el->label, CO_LABEL_LEN, "element_%d", max_idx);
                el->bbox[0] = (float)start_x;
                el->bbox[1] = (float)start_y;
                el->bbox[2] = (float)grid_step_x;
                el->bbox[3] = (float)grid_step_y;
                el->confidence = max_conf;
                el->is_visible = 1;
                float contrast = compute_local_contrast(region, 32, 32);
                if (contrast > 0.15f) el->ui_type = CO_UI_BUTTON;
                else if (contrast > 0.08f) el->ui_type = CO_UI_TEXT_INPUT;
                else el->ui_type = CO_UI_UNKNOWN;
                el->is_interactive = (el->ui_type == CO_UI_BUTTON || el->ui_type == CO_UI_TEXT_INPUT || el->ui_type == CO_UI_LINK);
                system->current_plan.element_positions[num_elements][0] = (float)start_x;
                system->current_plan.element_positions[num_elements][1] = (float)start_y;
                system->current_plan.element_positions[num_elements][2] = (float)grid_step_x;
                system->current_plan.element_positions[num_elements][3] = (float)grid_step_y;
                num_elements++;
            }
        }
    }

    system->current_plan.num_elements = num_elements;

    if (system->config.enable_ocr) {
        size_t num_ocr = 0;
        co_ocr_recognize(system, screen_data, width, height, system->current_plan.ocr_results, &num_ocr);
        system->current_plan.num_ocr_results = num_ocr;
    }

    return (int)num_elements;
}

int co_plan_task(COSystem* system, const char* task_description, COPlan* plan_out) {
    if (!system || !task_description || !plan_out) return -1;
    memset(&system->current_plan, 0, sizeof(COPlan));
    strncpy(system->current_plan.task_description, task_description, 255);
    system->current_plan.task_description[255] = '\0';

    float policy_input[1024 + 512] = {0};
    memcpy(policy_input, system->current_plan.screen_embedding, 1024 * sizeof(float));
    memcpy(policy_input + 1024, system->current_plan.task_embedding, 512 * sizeof(float));

    float policy_out[CO_ACTION_MAX * 10] = {0};
    co_forward(system, system->action_policy, policy_input, 1024+512, policy_out, CO_ACTION_MAX * 10);

    system->current_plan.num_actions = 0;
    for (int i = 0; i < CO_ACTION_MAX && system->current_plan.num_actions < CO_MAX_ACTIONS; i++) {
        float* act_out = policy_out + i * 10;
        COActionType atype = (COActionType)((int)act_out[0] % (int)CO_ACTION_MAX);
        if (act_out[9] < 0.1f) continue;

        COAction* action = &system->current_plan.actions[system->current_plan.num_actions];
        memset(action, 0, sizeof(COAction));
        action->action_type = atype;
        action->x = (int)(act_out[1] * system->last_width);
        action->y = (int)(act_out[2] * system->last_height);
        action->target_x = (int)(act_out[3] * system->last_width);
        action->target_y = (int)(act_out[4] * system->last_height);
        action->key_code = (int)act_out[5];
        action->modifier_flags = (int)act_out[6];
        action->duration = act_out[7];
        action->confidence = act_out[9];
        action->safety_checked = 0;

        if (atype == CO_ACTION_TYPE_TEXT && system->current_plan.num_ocr_results > 0) {
            strncpy(action->text, system->current_plan.ocr_results[0].text, 255);
        }
        if (atype == CO_ACTION_FILE_OPEN || atype == CO_ACTION_FILE_SAVE || atype == CO_ACTION_FILE_DELETE) {
            if (system->current_plan.num_ocr_results > 0) {
                strncpy(action->file_path, system->current_plan.ocr_results[0].text, CO_FILE_PATH_LEN - 1);
            }
        }
        if (atype == CO_ACTION_BROWSER_NAV && system->current_plan.num_ocr_results > 0) {
            strncpy(action->url, system->current_plan.ocr_results[0].text, 1023);
        }

        system->current_plan.num_actions++;
    }

    system->current_plan.is_plan_complete = 1;
    memcpy(plan_out, &system->current_plan, sizeof(COPlan));
    return (int)system->current_plan.num_actions;
}

static int co_close_window_by_title(COSystem* system, const char* title) {
    (void)system;
    if (!title || !title[0]) return -1;
#ifdef _WIN32
    HWND hwnd = FindWindowA(NULL, title);
    if (hwnd) {
        PostMessage(hwnd, WM_CLOSE, 0, 0);
        return 0;
    }
    return -1;
#elif defined(__linux__)
    Display* dpy = XOpenDisplay(NULL);
    if (!dpy) return -1;
    Window root = DefaultRootWindow(dpy);
    Atom net_close = XInternAtom(dpy, "_NET_CLOSE_WINDOW", False);
    XEvent ev = {0};
    ev.xclient.type = ClientMessage;
    ev.xclient.window = root;
    ev.xclient.message_type = net_close;
    ev.xclient.format = 32;
    XSendEvent(dpy, root, False, SubstructureRedirectMask | SubstructureNotifyMask, &ev);
    XFlush(dpy);
    XCloseDisplay(dpy);
    return 0;
#else
    return -1;
#endif
}

static int co_focus_window_by_title(COSystem* system, const char* title) {
    (void)system;
    if (!title || !title[0]) return -1;
#ifdef _WIN32
    HWND hwnd = FindWindowA(NULL, title);
    if (hwnd) {
        SetForegroundWindow(hwnd);
        return 0;
    }
    return -1;
#elif defined(__linux__)
    Display* dpy = XOpenDisplay(NULL);
    if (!dpy) return -1;
    Window root = DefaultRootWindow(dpy);
    Atom net_active = XInternAtom(dpy, "_NET_ACTIVE_WINDOW", False);
    XEvent ev = {0};
    ev.xclient.type = ClientMessage;
    ev.xclient.window = root;
    ev.xclient.message_type = net_active;
    ev.xclient.format = 32;
    XSendEvent(dpy, root, False, SubstructureRedirectMask | SubstructureNotifyMask, &ev);
    XFlush(dpy);
    XCloseDisplay(dpy);
    return 0;
#else
    return -1;
#endif
}

int co_execute_action(COSystem* system, const COAction* action) {
    if (!system || !action) return -1;

    /* 安全检查 */
    if (system->config.enable_safety_check) {
        char safety_reason[256];
        int safety_ret = co_safety_validate_action(system, action, safety_reason, sizeof(safety_reason));
        if (safety_ret != 0) return safety_ret;
    }

    switch (action->action_type) {
        case CO_ACTION_CLICK:
            return co_os_click(action->x, action->y, 0);
        case CO_ACTION_DOUBLE_CLICK:
            if (co_os_click(action->x, action->y, 0) != 0) return -1;
            co_os_sleep_ms(50);
            return co_os_click(action->x, action->y, 0);
        case CO_ACTION_RIGHT_CLICK:
            return co_os_click(action->x, action->y, 2);
        case CO_ACTION_DRAG: {
            int x1 = action->x, y1 = action->y, x2 = action->target_x, y2 = action->target_y;
            co_os_mouse_move(x1, y1);
            co_os_sleep_ms(10);

#ifdef _WIN32
            INPUT down = {0};
            down.type = INPUT_MOUSE;
            down.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
            SendInput(1, &down, sizeof(INPUT));
#elif defined(__linux__)
            Display* dpy = XOpenDisplay(NULL);
            if (dpy) {
                XTestFakeButtonEvent(dpy, 1, True, CurrentTime);
                XFlush(dpy);
                XCloseDisplay(dpy);
            }
#endif
            co_os_sleep_ms(5);

            int steps = (int)(action->duration * 60.0f);
            if (steps < 5) steps = 5;
            for (int s = 0; s < steps; s++) {
                int mx = x1 + (x2 - x1) * s / steps;
                int my = y1 + (y2 - y1) * s / steps;
                co_os_mouse_move(mx, my);
                co_os_sleep_ms((int)(action->duration * 1000.0f / (float)steps));
            }

#ifdef _WIN32
            INPUT up = {0};
            up.type = INPUT_MOUSE;
            up.mi.dwFlags = MOUSEEVENTF_LEFTUP;
            SendInput(1, &up, sizeof(INPUT));
#elif defined(__linux__)
            dpy = XOpenDisplay(NULL);
            if (dpy) {
                XTestFakeButtonEvent(dpy, 1, False, CurrentTime);
                XFlush(dpy);
                XCloseDisplay(dpy);
            }
#endif
            return 0;
        }
        case CO_ACTION_SCROLL:
            return co_os_scroll(action->target_y);
        case CO_ACTION_KEYPRESS:
            co_os_keypress(action->key_code, 1);
            co_os_sleep_ms(30);
            return co_os_keypress(action->key_code, 0);
        case CO_ACTION_KEY_COMBO: {
            int keys[8] = {0};
            int kc = 0;
            if (action->modifier_flags & 1) { co_os_keypress(0xA0, 1); keys[kc++] = 0xA0; }
            if (action->modifier_flags & 2) { co_os_keypress(0xA2, 1); keys[kc++] = 0xA2; }
            if (action->modifier_flags & 4) { co_os_keypress(0x5B, 1); keys[kc++] = 0x5B; }
            co_os_keypress(action->key_code, 1);
            co_os_sleep_ms(20);
            co_os_keypress(action->key_code, 0);
            for (int ki = kc - 1; ki >= 0; ki--) co_os_keypress(keys[ki], 0);
            return 0;
        }
        case CO_ACTION_TYPE_TEXT:
            if (action->text[0]) {
                return co_os_type_text(action->text, strlen(action->text));
            }
            return 0;
        case CO_ACTION_MOVE:
            return co_os_mouse_move(action->x, action->y);
        case CO_ACTION_HOVER:
            co_os_mouse_move(action->x, action->y);
            co_os_sleep_ms((int)(action->duration * 1000.0f));
            return 0;
        case CO_ACTION_LAUNCH_APP: {
            if (!action->file_path[0]) return -1;
#ifdef _WIN32
            ShellExecute(NULL, "open", action->file_path, NULL, NULL, SW_SHOWNORMAL);
#elif defined(__linux__)
            pid_t pid = fork();
            if (pid == 0) {
                execlp(action->file_path, action->file_path, (char*)NULL);
                _exit(1);
            }
#endif
            return 0;
        }
        case CO_ACTION_CLOSE_WINDOW: {
            if (action->window_title[0]) {
                co_close_window_by_title(system, action->window_title);
            }
            return 0;
        }
        case CO_ACTION_SWITCH_WINDOW: {
            if (action->window_title[0]) {
                co_focus_window_by_title(system, action->window_title);
            }
            return 0;
        }
        case CO_ACTION_FILE_OPEN: {
            if (!action->file_path[0]) return -1;
            return co_file_open(system, action->file_path);
        }
        case CO_ACTION_FILE_SAVE: {
            if (!action->file_path[0]) return -1;
            return co_file_save(system, action->file_path, action->text, strlen(action->text));
        }
        case CO_ACTION_FILE_DELETE: {
            if (!action->file_path[0]) return -1;
            return co_file_delete(system, action->file_path);
        }
        case CO_ACTION_BROWSER_NAV: {
            if (!action->url[0]) return -1;
            return co_browser_navigate(system, action->url);
        }
        case CO_ACTION_SCREENSHOT:
            return co_os_screenshot_real(system, action);
        default:
            return -1;
    }
}

int co_execute_plan(COSystem* system, COPlan* plan, int (*progress_callback)(float, const char*, void*), void* user_data) {
    if (!system || !plan) return -1;
    if (plan->num_actions == 0) return -2;

    int success_count = 0;
    int fail_count = 0;

    for (size_t i = 0; i < plan->num_actions; i++) {
        COAction* action = &plan->actions[i];
        if (action->confidence < system->config.error_tolerance) {
            fail_count++;
            continue;
        }
        int retries = 0;
        int action_success = 0;
        while (retries <= system->config.max_retries && !action_success) {
            if (co_execute_action(system, action) == 0) action_success = 1;
            else retries++;
        }
        if (action_success) success_count++;
        else fail_count++;

        if (progress_callback) {
            float progress = (float)(i + 1) / (float)plan->num_actions;
            char status[128];
            snprintf(status, 128, "执行动作 %zu/%zu: %s", i + 1, plan->num_actions, action_success ? "成功" : "失败");
            if (progress_callback(progress, status, user_data)) break;
        }
    }

    if (success_count > fail_count) system->tasks_completed++;
    else system->tasks_failed++;

    float total = (float)(system->tasks_completed + system->tasks_failed);
    if (total > 0) system->avg_success_rate = (float)system->tasks_completed / total;
    plan->is_plan_complete = 1;
    return success_count > 0 ? 0 : -3;
}

int co_recognize_ui_element(COSystem* system, const float* screen_region, size_t region_width, size_t region_height, char* element_label, size_t label_buf_size, float* confidence) {
    if (!system || !screen_region || !element_label || !confidence) return -1;

    float region_input[32 * 32] = {0};
    resize_image_bilinear(screen_region, region_width, region_height, 1, region_input, 32, 32);

    float classifier_out[16] = {0};
    co_forward(system, system->ui_classifier, region_input, 32*32, classifier_out, 16);

    int best_idx = 0;
    float best_conf = classifier_out[0];
    for (int i = 1; i < 16; i++) {
        if (classifier_out[i] > best_conf) { best_conf = classifier_out[i]; best_idx = i; }
    }

    *confidence = best_conf;
    const char* ui_names[] = {"button", "text_input", "label", "link", "checkbox", "radio", "dropdown", "slider", "icon", "image", "table", "menu", "scrollbar", "dialog", "window", "unknown"};
    const char* name = (best_idx >= 0 && best_idx < 16) ? ui_names[best_idx] : "unknown";
    snprintf(element_label, label_buf_size, "%s", name);

    float contrast = compute_local_contrast(region_input, 32, 32);
    if (contrast < 0.05f && best_conf < 0.3f) {
        *confidence = 0.0f;
        snprintf(element_label, label_buf_size, "empty");
        return -3;
    }

    return best_idx;
}

int co_ocr_recognize(COSystem* system, const float* screen_data, size_t width, size_t height, COOCRResult* results, size_t* num_results) {
    if (!system || !screen_data || !results || !num_results) return -1;

    size_t max_results = *num_results > 0 ? *num_results : CO_MAX_OCR_RESULTS;
    size_t found = 0;

    float regions[CO_MAX_OCR_RESULTS * 4];
    size_t num_regions = CO_MAX_OCR_RESULTS;
    co_detect_text_region(system, screen_data, width, height, regions, &num_regions);

    for (size_t r = 0; r < num_regions && found < max_results; r++) {
        float* reg = &regions[r * 4];
        int rx = (int)reg[0];
        int ry = (int)reg[1];
        int rw = (int)(reg[2] - reg[0]);
        int rh = (int)(reg[3] - reg[1]);
        if (rw < 4 || rh < 4) continue;
        if (rx + rw > (int)width) rw = (int)width - rx;
        if (ry + rh > (int)height) rh = (int)height - ry;
        if (rw <= 0 || rh <= 0) continue;

        int num_chars = rw / 8;
        if (num_chars > 32) num_chars = 32;
        if (num_chars < 1) num_chars = 1;

        COOCRResult* ocr = &results[found];
        memset(ocr, 0, sizeof(COOCRResult));
        ocr->bbox[0] = (float)rx;
        ocr->bbox[1] = (float)ry;
        ocr->bbox[2] = (float)(rx + rw);
        ocr->bbox[3] = (float)(ry + rh);
        ocr->is_recognized = 1;

        for (int ci = 0; ci < num_chars && ci < CO_OCR_TEXT_LEN - 1; ci++) {
            int char_x = rx + ci * 8;
            int char_y = ry;
            int cw = (rw - ci * 8) > 8 ? 8 : (rw - ci * 8);
            int ch = rh > 16 ? 16 : rh;
            if (cw < 2 || ch < 2) { ocr->text[ci] = ' '; continue; }

            float char_input[16 * 16] = {0};
            for (int py = 0; py < 16 && py < ch; py++) {
                for (int px = 0; px < 16 && px < cw; px++) {
                    int sx = char_x + (int)((float)px * (float)cw / 16.0f);
                    int sy = char_y + (int)((float)py * (float)ch / 16.0f);
                    if (sx >= (int)width) sx = (int)width - 1;
                    if (sy >= (int)height) sy = (int)height - 1;
                    float val = 0;
                    size_t idx = (size_t)(sy * width + sx);
                    if (idx * 3 + 2 < width * height * 3) {
                        val = (screen_data[idx * 3] + screen_data[idx * 3 + 1] + screen_data[idx * 3 + 2]) / 3.0f;
                    }
                    char_input[py * 16 + px] = val;
                }
            }

            float ocr_out[36] = {0};
            co_forward(system, system->ocr_net, char_input, 16*16, ocr_out, 36);

            int best_char = 0;
            float best_char_conf = ocr_out[0];
            for (int ci2 = 1; ci2 < 36; ci2++) {
                if (ocr_out[ci2] > best_char_conf) { best_char_conf = ocr_out[ci2]; best_char = ci2; }
            }

            if (best_char_conf > 0.4f) {
                if (best_char < 10) ocr->text[ci] = (char)('0' + best_char);
                else ocr->text[ci] = (char)('A' + best_char - 10);
                ocr->confidence += best_char_conf;
            } else {
                ocr->text[ci] = '?';
            }
        }

        ocr->text[num_chars] = '\0';
        if (num_chars > 0) ocr->confidence /= (float)num_chars;
        found++;
    }

    *num_results = found;
    return (int)found;
}

int co_classify_ui_element(COSystem* system, const float* region_data, size_t rw, size_t rh, COUIElement* element_out) {
    if (!system || !region_data || !element_out) return -1;

    float input[32 * 32] = {0};
    resize_image_bilinear(region_data, rw, rh, 1, input, 32, 32);

    float out[16] = {0};
    co_forward(system, system->ui_classifier, input, 32*32, out, 16);

    int best = 0;
    element_out->confidence = out[0];
    for (int i = 1; i < 16; i++) {
        if (out[i] > element_out->confidence) { element_out->confidence = out[i]; best = i; }
    }

    element_out->ui_type = (COUIElementType)best;
    element_out->is_visible = 1;
    element_out->is_interactive = (best <= CO_UI_SLIDER || best == CO_UI_MENU);
    snprintf(element_out->label, CO_LABEL_LEN, "ui_%d", best);

    float contrast = compute_local_contrast(input, 32, 32);
    if (contrast < 0.03f) { element_out->is_visible = 0; element_out->confidence *= 0.5f; }

    return best;
}

int co_detect_text_region(COSystem* system, const float* screen_data, size_t width, size_t height, float* regions_out, size_t* num_regions) {
    if (!system || !screen_data || !regions_out || !num_regions) return -1;

    size_t max_regions = *num_regions;
    size_t found = 0;
    int step = 16;

    for (size_t y = 0; y + 16 <= height && found < max_regions; y += step) {
        for (size_t x = 0; x + 32 <= width && found < max_regions; x += step) {
            float region[32 * 16];
            for (int ry = 0; ry < 16; ry++) {
                for (int rx = 0; rx < 32; rx++) {
                    size_t sx = x + rx;
                    size_t sy = y + ry;
                    if (sx >= width) sx = width - 1;
                    if (sy >= height) sy = height - 1;
                    size_t idx = (sy * width + sx) * 3;
                    float val = (screen_data[idx] + screen_data[idx + 1] + screen_data[idx + 2]) / 3.0f;
                    region[ry * 32 + rx] = val;
                }
            }

            float contrast = compute_local_contrast(region, 32, 16);
            if (contrast > 0.12f) {
                int merged = 0;
                for (size_t i = 0; i < found; i++) {
                    float* existing = &regions_out[i * 4];
                    float ex = existing[0], ey = existing[1];
                    float ex2 = existing[2], ey2 = existing[3];
                    float fx = (float)x, fy = (float)y;
                    float fx2 = fx + 32.0f, fy2 = fy + 16.0f;
                    if (fx <= ex2 + 8 && fx2 >= ex - 8 && fy <= ey2 + 4 && fy2 >= ey - 4) {
                        if (fx < ex) existing[0] = fx;
                        if (fy < ey) existing[1] = fy;
                        if (fx2 > ex2) existing[2] = fx2;
                        if (fy2 > ey2) existing[3] = fy2;
                        merged = 1;
                        break;
                    }
                }
                if (!merged) {
                    float* new_reg = &regions_out[found * 4];
                    new_reg[0] = (float)x; new_reg[1] = (float)y;
                    new_reg[2] = (float)(x + 32); new_reg[3] = (float)(y + 16);
                    found++;
                }
            }
        }
    }

    *num_regions = found;
    return (int)found;
}

int co_browser_navigate(COSystem* system, const char* url) {
    if (!system || !url) return -1;
    if (!system->config.enable_browser_control) return -5;
    COAction action;
    memset(&action, 0, sizeof(COAction));
    action.action_type = CO_ACTION_BROWSER_NAV;
    strncpy(action.url, url, 1023);
    action.confidence = 1.0f;
    action.safety_checked = 1;
    return co_execute_action(system, &action);
}

int co_browser_click_link(COSystem* system, const char* link_text) {
    if (!system || !link_text) return -1;
    if (!system->config.enable_browser_control) return -5;
    for (size_t i = 0; i < system->current_plan.num_elements; i++) {
        COUIElement* el = &system->current_plan.detected_elements[i];
        if (el->ui_type == CO_UI_LINK && strstr(el->associated_text, link_text)) {
            COAction action;
            memset(&action, 0, sizeof(COAction));
            action.action_type = CO_ACTION_CLICK;
            action.x = (int)(el->bbox[0] + el->bbox[2] / 2);
            action.y = (int)(el->bbox[1] + el->bbox[3] / 2);
            action.confidence = el->confidence;
            return co_execute_action(system, &action);
        }
    }
    return -3;
}

int co_browser_fill_form(COSystem* system, const char* field_name, const char* value) {
    if (!system || !field_name || !value) return -1;
    if (!system->config.enable_browser_control) return -5;
    for (size_t i = 0; i < system->current_plan.num_elements; i++) {
        COUIElement* el = &system->current_plan.detected_elements[i];
        if (el->ui_type == CO_UI_TEXT_INPUT && strstr(el->associated_text, field_name)) {
            COAction click_action;
            memset(&click_action, 0, sizeof(COAction));
            click_action.action_type = CO_ACTION_CLICK;
            click_action.x = (int)(el->bbox[0] + el->bbox[2] / 2);
            click_action.y = (int)(el->bbox[1] + el->bbox[3] / 2);
            click_action.confidence = el->confidence;
            co_execute_action(system, &click_action);

            COAction type_action;
            memset(&type_action, 0, sizeof(COAction));
            type_action.action_type = CO_ACTION_TYPE_TEXT;
            strncpy(type_action.text, value, 255);
            type_action.confidence = 1.0f;
            return co_execute_action(system, &type_action);
        }
    }
    return -3;
}

int co_browser_get_tabs(COSystem* system, COBrowserTab* tabs, size_t* num_tabs) {
    if (!system || !tabs || !num_tabs) return -1;
    size_t max_tabs = *num_tabs;
    size_t count = 0;
#ifdef _WIN32
    /* 枚举浏览器窗口：通过窗口类名查找已知浏览器 */
    static const char* browser_classes[] = {
        "MozillaWindowClass",      /* Firefox */
        "Chrome_WidgetWin_1",      /* Chrome */
        "Chrome_WidgetWin_0",
        "ApplicationFrameWindow",  /* Edge */
        NULL
    };
    HWND hwnd = NULL;
    for (int bc = 0; browser_classes[bc] != NULL && count < max_tabs; bc++) {
        hwnd = FindWindowExA(NULL, hwnd, browser_classes[bc], NULL);
        while (hwnd && count < max_tabs) {
            char title[256] = {0};
            GetWindowTextA(hwnd, title, sizeof(title) - 1);
            if (title[0] != '\0') {
                tabs[count].tab_id = (int)((size_t)hwnd & 0x7FFFFFFF);
                snprintf(tabs[count].title, sizeof(tabs[count].title), "%s", title);
                tabs[count].url[0] = '\0';
                tabs[count].is_active = (hwnd == GetForegroundWindow()) ? 1 : 0;
                count++;
            }
            hwnd = FindWindowExA(NULL, hwnd, browser_classes[bc], NULL);
        }
    }
#elif defined(__linux__)
    /* Linux下通过wmctrl或xdotool查询，简化实现使用X11查询 */
    Display* dpy = XOpenDisplay(NULL);
    if (dpy) {
        Window root = DefaultRootWindow(dpy);
        Window parent, *children;
        unsigned int nchildren;
        if (XQueryTree(dpy, root, &root, &parent, &children, &nchildren) && children) {
            for (unsigned int i = 0; i < nchildren && count < max_tabs; i++) {
                char* wm_name = NULL;
                XTextProperty tp;
                if (XGetWMName(dpy, children[i], &tp) && tp.value) {
                    wm_name = (char*)tp.value;
                }
                if (wm_name && wm_name[0]) {
                    tabs[count].tab_id = (int)i;
                    snprintf(tabs[count].title, sizeof(tabs[count].title), "%s", wm_name);
                    tabs[count].url[0] = '\0';
                    tabs[count].is_active = 0;
                    count++;
                    XFree(tp.value);
                }
            }
            XFree(children);
        }
        XCloseDisplay(dpy);
    }
#endif
    *num_tabs = count;
    return (int)count;
}

int co_browser_switch_tab(COSystem* system, int tab_id) {
    if (!system) return -1;
    if (!system->config.enable_browser_control) return -5;
    COAction action;
    memset(&action, 0, sizeof(COAction));
    action.action_type = CO_ACTION_SWITCH_WINDOW;
    action.key_code = tab_id;
    action.modifier_flags = 2;
    action.confidence = 1.0f;
    return co_execute_action(system, &action);
}

int co_browser_close_tab(COSystem* system, int tab_id) {
    if (!system) return -1;
    COAction action;
    memset(&action, 0, sizeof(COAction));
    action.action_type = CO_ACTION_CLOSE_WINDOW;
    action.key_code = tab_id;
    action.confidence = 1.0f;
    return co_execute_action(system, &action);
}

int co_browser_scroll(COSystem* system, int dx, int dy) {
    if (!system) return -1;
    COAction action;
    memset(&action, 0, sizeof(COAction));
    action.action_type = CO_ACTION_SCROLL;
    action.target_x = dx;
    action.target_y = dy;
    action.confidence = 1.0f;
    return co_execute_action(system, &action);
}

int co_browser_execute_js(COSystem* system, const char* js_code, char* result_out, size_t result_size) {
    if (!system || !js_code) return -1;
    if (!system->config.enable_browser_control) return -5;
    /* 尝试通过键盘注入执行JS：
     * 1. 将JS代码写入剪贴板
     * 2. 发送F12打开开发者工具
     * 3. 切换到控制台标签
     * 4. 粘贴并执行
     */
    int success = 0;
#ifdef _WIN32
    if (OpenClipboard(NULL)) {
        HGLOBAL hMem = GlobalAlloc(GMEM_MOVEABLE, strlen(js_code) + 1);
        if (hMem) {
            char* clip_data = (char*)GlobalLock(hMem);
            if (clip_data) {
                strcpy(clip_data, js_code);
                GlobalUnlock(hMem);
                EmptyClipboard();
                SetClipboardData(CF_TEXT, hMem);
                /* 发送Ctrl+V粘贴 */
                keybd_event(VK_CONTROL, 0, 0, 0);
                keybd_event('V', 0, 0, 0);
                keybd_event('V', 0, KEYEVENTF_KEYUP, 0);
                keybd_event(VK_CONTROL, 0, KEYEVENTF_KEYUP, 0);
                /* 发送Enter执行 */
                keybd_event(VK_RETURN, 0, 0, 0);
                keybd_event(VK_RETURN, 0, KEYEVENTF_KEYUP, 0);
                success = 1;
            }
            GlobalFree(hMem);
        }
        CloseClipboard();
    }
#elif defined(__linux__) && defined(X11_AVAILABLE)
    /* Linux下使用xdotool或xclip注入 */
    char cmd[2048];
    snprintf(cmd, sizeof(cmd), "echo '%s' | xclip -selection clipboard 2>/dev/null && "
             "xdotool key ctrl+v Return 2>/dev/null", js_code);
    success = (system(cmd) == 0) ? 1 : 0;
#endif
    if (result_out && result_size > 0) {
        snprintf(result_out, result_size, "{\"status\":\"%s\",\"method\":\"clipboard_injection\"}",
                 success ? "executed" : "failed");
    }
    return success ? 0 : -3;
}

int co_file_open(COSystem* system, const char* path) {
    if (!system || !path) return -1;
    if (!system->config.enable_file_operations) return -5;
    COAction action;
    memset(&action, 0, sizeof(COAction));
    action.action_type = CO_ACTION_FILE_OPEN;
    strncpy(action.file_path, path, CO_FILE_PATH_LEN - 1);
    action.confidence = 1.0f;
    return co_execute_action(system, &action);
}

int co_file_save(COSystem* system, const char* path, const char* content, size_t content_len) {
    if (!system || !path) return -1;
    if (!system->config.enable_file_operations) return -5;
    if (content && content_len > 0) {
        FILE* f = fopen(path, "wb");
        if (!f) return -4;
        size_t written = fwrite(content, 1, content_len, f);
        fclose(f);
        return written > 0 ? 0 : -4;
    }
    COAction action;
    memset(&action, 0, sizeof(COAction));
    action.action_type = CO_ACTION_FILE_SAVE;
    strncpy(action.file_path, path, CO_FILE_PATH_LEN - 1);
    action.confidence = 1.0f;
    return co_execute_action(system, &action);
}

int co_file_delete(COSystem* system, const char* path) {
    if (!system || !path) return -1;
    if (!system->config.enable_file_operations) return -5;
    char safety_reason[256];
    COAction check_action;
    memset(&check_action, 0, sizeof(COAction));
    check_action.action_type = CO_ACTION_FILE_DELETE;
    strncpy(check_action.file_path, path, CO_FILE_PATH_LEN - 1);
    if (system->config.enable_safety_check && co_safety_validate_action(system, &check_action, safety_reason, sizeof(safety_reason)) != 0) {
        return -6;
    }
    COAction action;
    memset(&action, 0, sizeof(COAction));
    action.action_type = CO_ACTION_FILE_DELETE;
    strncpy(action.file_path, path, CO_FILE_PATH_LEN - 1);
    action.confidence = 1.0f;
    action.safety_checked = 1;
    return co_execute_action(system, &action);
}

int co_file_list_directory(COSystem* system, const char* dir_path, COFileInfo* files, size_t* num_files) {
    if (!system || !dir_path || !files || !num_files) return -1;
    if (!system->config.enable_file_operations) return -5;
    size_t max = *num_files;
    size_t count = 0;
#ifdef _WIN32
    WIN32_FIND_DATAA find_data;
    char search_path[1024];
    snprintf(search_path, sizeof(search_path), "%s\\*", dir_path);
    HANDLE hFind = FindFirstFileA(search_path, &find_data);
    if (hFind == INVALID_HANDLE_VALUE) {
        *num_files = 0;
        return 0;
    }
    do {
        if (strcmp(find_data.cFileName, ".") == 0 || strcmp(find_data.cFileName, "..") == 0) continue;
        if (count >= max) break;
        snprintf(files[count].name, sizeof(files[count].name), "%s", find_data.cFileName);
        snprintf(files[count].path, sizeof(files[count].path), "%s\\%s", dir_path, find_data.cFileName);
        files[count].size_bytes = ((int64_t)find_data.nFileSizeHigh << 32) | find_data.nFileSizeLow;
        files[count].is_directory = (find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) ? 1 : 0;
        files[count].last_modified = 0;
        count++;
    } while (FindNextFileA(hFind, &find_data));
    FindClose(hFind);
#else
    struct dirent* entry;
    DIR* dp = opendir(dir_path);
    if (!dp) {
        *num_files = 0;
        return 0;
    }
    while ((entry = readdir(dp)) != NULL) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) continue;
        if (count >= max) break;
        snprintf(files[count].name, sizeof(files[count].name), "%s", entry->d_name);
        snprintf(files[count].path, sizeof(files[count].path), "%s/%s", dir_path, entry->d_name);
        files[count].is_directory = (entry->d_type == DT_DIR) ? 1 : 0;
        files[count].size = 0;
        files[count].last_modified = 0;
        count++;
    }
    closedir(dp);
#endif
    *num_files = count;
    return (int)count;
}

int co_file_copy(COSystem* system, const char* src, const char* dst) {
    if (!system || !src || !dst) return -1;
    if (!system->config.enable_file_operations) return -5;
    FILE* fsrc = fopen(src, "rb");
    if (!fsrc) return -4;
    FILE* fdst = fopen(dst, "wb");
    if (!fdst) { fclose(fsrc); return -4; }
    char buf[8192];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), fsrc)) > 0) fwrite(buf, 1, n, fdst);
    fclose(fsrc);
    fclose(fdst);
    return 0;
}

int co_file_move(COSystem* system, const char* src, const char* dst) {
    if (!system || !src || !dst) return -1;
    if (!system->config.enable_file_operations) return -5;
    if (rename(src, dst) == 0) return 0;
    if (co_file_copy(system, src, dst) == 0 && remove(src) == 0) return 0;
    return -4;
}

int co_file_create_directory(COSystem* system, const char* path) {
    if (!system || !path) return -1;
    (void)system;
#ifdef _WIN32
    return _mkdir(path) == 0 ? 0 : -4;
#else
    return mkdir(path, 0755) == 0 ? 0 : -4;
#endif
}

int co_launch_app(COSystem* system, const char* app_path, const char* args) {
    if (!system || !app_path) return -1;
    if (!system->config.enable_app_control) return -5;
    COAction action;
    memset(&action, 0, sizeof(COAction));
    action.action_type = CO_ACTION_LAUNCH_APP;
    strncpy(action.file_path, app_path, CO_FILE_PATH_LEN - 1);
    if (args) strncpy(action.text, args, 255);
    action.confidence = 1.0f;
    return co_execute_action(system, &action);
}

int co_close_app(COSystem* system, const char* app_name) {
    if (!system || !app_name) return -1;
    if (!system->config.enable_app_control) return -5;
    COAction action;
    memset(&action, 0, sizeof(COAction));
    action.action_type = CO_ACTION_CLOSE_WINDOW;
    strncpy(action.window_title, app_name, 127);
    action.confidence = 1.0f;
    /* 在窗口列表中查找匹配的窗口，存入实际标题 */
    for (size_t i = 0; i < system->num_windows; i++) {
        if (co_window_match_title(app_name, system->window_list[i].title)) {
            strncpy(action.window_title, system->window_list[i].title, 127);
            break;
        }
    }
    return co_execute_action(system, &action);
}

int co_list_windows(COSystem* system, COWindowInfo* windows, size_t* num_windows) {
    if (!system || !windows || !num_windows) return -1;
    size_t count = (system->num_windows < CO_MAX_WINDOWS) ? system->num_windows : CO_MAX_WINDOWS;
    for (size_t i = 0; i < count; i++) {
        memcpy(&windows[i], &system->window_list[i], sizeof(COWindowInfo));
    }
    *num_windows = count;
    return 0;
}

int co_switch_window(COSystem* system, const char* window_title) {
    if (!system || !window_title) return -1;
    if (!system->config.enable_app_control) return -5;
    COAction action;
    memset(&action, 0, sizeof(COAction));
    action.action_type = CO_ACTION_SWITCH_WINDOW;
    strncpy(action.window_title, window_title, 127);
    action.modifier_flags = 1;
    action.confidence = 1.0f;
    /* 在窗口列表中查找匹配的窗口 */
    for (size_t i = 0; i < system->num_windows; i++) {
        if (co_window_match_title(window_title, system->window_list[i].title)) {
            strncpy(action.window_title, system->window_list[i].title, 127);
            action.confidence = system->window_list[i].is_active ? 1.0f : 0.8f;
            action.target_x = (int)(system->window_list[i].position[0] +
                           system->window_list[i].position[2] / 2.0f);
            action.target_y = (int)(system->window_list[i].position[1] +
                           system->window_list[i].position[3] / 2.0f);
            break;
        }
    }
    return co_execute_action(system, &action);
}

int co_get_active_window(COSystem* system, COWindowInfo* window_out) {
    if (!system || !window_out) return -1;
    for (size_t i = 0; i < system->num_windows; i++) {
        if (system->window_list[i].is_active) {
            memcpy(window_out, &system->window_list[i], sizeof(COWindowInfo));
            return 0;
        }
    }
    memset(window_out, 0, sizeof(COWindowInfo));
    return 0;
}

int co_minimize_window(COSystem* system, const char* window_title) {
    if (!system || !window_title) return -1;
    if (!system->config.enable_app_control) return -5;
    COAction action;
    memset(&action, 0, sizeof(COAction));
    action.action_type = CO_ACTION_CLOSE_WINDOW;
    strncpy(action.window_title, window_title, 127);
    action.text[0] = 'm';
    action.confidence = 1.0f;
    /* 查找匹配窗口位置 */
    for (size_t i = 0; i < system->num_windows; i++) {
        if (co_window_match_title(window_title, system->window_list[i].title)) {
            strncpy(action.window_title, system->window_list[i].title, 127);
            action.target_x = (int)(system->window_list[i].position[0] +
                           system->window_list[i].position[2] / 2.0f);
            action.target_y = (int)(system->window_list[i].position[1] + 10);
            system->window_list[i].is_minimized = 1;
            break;
        }
    }
    return co_execute_action(system, &action);
}

int co_maximize_window(COSystem* system, const char* window_title) {
    if (!system || !window_title) return -1;
    COAction action;
    memset(&action, 0, sizeof(COAction));
    action.action_type = CO_ACTION_SWITCH_WINDOW;
    strncpy(action.window_title, window_title, 127);
    action.text[0] = 'M';
    action.confidence = 1.0f;
    for (size_t i = 0; i < system->num_windows; i++) {
        if (co_window_match_title(window_title, system->window_list[i].title)) {
            strncpy(action.window_title, system->window_list[i].title, 127);
            action.target_x = (int)(system->window_list[i].position[0] +
                           system->window_list[i].position[2] / 2.0f);
            action.target_y = (int)(system->window_list[i].position[1] + 10);
            system->window_list[i].is_minimized = 0;
            break;
        }
    }
    return co_execute_action(system, &action);
}

/**
 * @brief 根据标题模式查找匹配的窗口
 *
 * 支持两种匹配模式：
 * 1. 纯文本通配模式（默认）：* 匹配任意字符，? 匹配单个字符
 * 2. 正则模式：以 "re:" 开头的模式使用正则引擎匹配
 *
 * @param system COSystem实例
 * @param pattern 标题匹配模式（纯文本通配或 "re:" + 正则表达式）
 * @param windows 输出匹配的窗口信息数组
 * @param num_windows 输入缓冲区大小，输出实际匹配数量
 * @return int 成功返回0，失败返回-1
 */
int co_find_windows_by_title(COSystem* system, const char* pattern,
                              COWindowInfo* windows, size_t* num_windows) {
    if (!system || !pattern || !windows || !num_windows) return -1;
    size_t max_out = *num_windows;
    size_t found = 0;
    for (size_t i = 0; i < system->num_windows && found < max_out; i++) {
        if (co_window_match_title(pattern, system->window_list[i].title)) {
            memcpy(&windows[found], &system->window_list[i], sizeof(COWindowInfo));
            found++;
        }
    }
    *num_windows = found;
    return 0;
}

/**
 * @brief 设置/更新窗口信息（供外部系统集成使用）
 *
 * 当外部系统检测到窗口状态变化时，调用此函数更新内部窗口列表。
 * 支持新增窗口和更新已有窗口信息。
 *
 * @param system COSystem实例
 * @param info 窗口信息
 * @return int 成功返回0，失败返回-1
 */
int co_set_window_info(COSystem* system, const COWindowInfo* info) {
    if (!system || !info) return -1;
    /* 查找是否已存在相同PID的窗口 */
    for (size_t i = 0; i < system->num_windows; i++) {
        if (system->window_list[i].pid == info->pid) {
            memcpy(&system->window_list[i], info, sizeof(COWindowInfo));
            return 0;
        }
    }
    /* 新增窗口 */
    if (system->num_windows >= CO_MAX_WINDOWS) return -2;
    memcpy(&system->window_list[system->num_windows], info, sizeof(COWindowInfo));
    system->num_windows++;
    return 0;
}

int co_simulate_keypress(COSystem* system, int key_code, int modifier_flags) {
    if (!system) return -1;
    COAction action;
    memset(&action, 0, sizeof(COAction));
    action.action_type = CO_ACTION_KEYPRESS;
    action.key_code = key_code;
    action.modifier_flags = modifier_flags;
    action.confidence = 1.0f;
    return co_execute_action(system, &action);
}

int co_simulate_key_combo(COSystem* system, const int* keys, size_t num_keys) {
    if (!system || !keys || num_keys == 0) return -1;
    int modifiers = 0;
    int main_key = keys[0];
    if (num_keys > 1) {
        for (size_t i = 0; i < num_keys - 1; i++) {
            if (keys[i] == 17) modifiers |= 1;
            else if (keys[i] == 16) modifiers |= 2;
            else if (keys[i] == 18) modifiers |= 4;
            else main_key = keys[i];
        }
        main_key = keys[num_keys - 1];
    }
    return co_simulate_keypress(system, main_key, modifiers);
}

int co_simulate_type_text(COSystem* system, const char* text, float speed) {
    if (!system || !text) return -1;
    COAction action;
    memset(&action, 0, sizeof(COAction));
    action.action_type = CO_ACTION_TYPE_TEXT;
    strncpy(action.text, text, 255);
    action.duration = speed;
    action.confidence = 1.0f;
    return co_execute_action(system, &action);
}

int co_simulate_mouse_move(COSystem* system, int x, int y) {
    if (!system) return -1;
    COAction action;
    memset(&action, 0, sizeof(COAction));
    action.action_type = CO_ACTION_MOVE;
    action.x = x;
    action.y = y;
    action.confidence = 1.0f;
    return co_execute_action(system, &action);
}

int co_simulate_mouse_click(COSystem* system, int x, int y, int button) {
    if (!system) return -1;
    co_simulate_mouse_move(system, x, y);
    COAction action;
    memset(&action, 0, sizeof(COAction));
    action.action_type = (button == 2) ? CO_ACTION_RIGHT_CLICK : CO_ACTION_CLICK;
    action.x = x;
    action.y = y;
    if (button == 0) action.modifier_flags = 0;
    action.confidence = 1.0f;
    return co_execute_action(system, &action);
}

int co_simulate_mouse_drag(COSystem* system, int x1, int y1, int x2, int y2) {
    if (!system) return -1;
    COAction action;
    memset(&action, 0, sizeof(COAction));
    action.action_type = CO_ACTION_DRAG;
    action.x = x1;
    action.y = y1;
    action.target_x = x2;
    action.target_y = y2;
    action.confidence = 1.0f;
    return co_execute_action(system, &action);
}

int co_simulate_mouse_scroll(COSystem* system, int clicks) {
    if (!system) return -1;
    COAction action;
    memset(&action, 0, sizeof(COAction));
    action.action_type = CO_ACTION_SCROLL;
    action.target_y = clicks;
    action.confidence = 1.0f;
    return co_execute_action(system, &action);
}

int co_simulate_mouse_double_click(COSystem* system, int x, int y) {
    if (!system) return -1;
    co_simulate_mouse_move(system, x, y);
    COAction action;
    memset(&action, 0, sizeof(COAction));
    action.action_type = CO_ACTION_DOUBLE_CLICK;
    action.x = x;
    action.y = y;
    action.confidence = 1.0f;
    return co_execute_action(system, &action);
}

int co_screenshot_capture(COSystem* system, float* data_out, size_t* width, size_t* height) {
    if (!system || !data_out || !width || !height) return -1;

    COAction screenshot_action;
    memset(&screenshot_action, 0, sizeof(COAction));
    screenshot_action.action_type = CO_ACTION_SCREENSHOT;
    screenshot_action.confidence = 1.0f;

    int result = co_os_screenshot_real(system, &screenshot_action);

    if (result == 0 && system->last_screen_cache) {
        size_t copy_w = system->last_width > 0 ? system->last_width : CO_SCREEN_WIDTH;
        size_t copy_h = system->last_height > 0 ? system->last_height : CO_SCREEN_HEIGHT;
        size_t copy_size = copy_w * copy_h * CO_SCREEN_CHANNELS;
        memcpy(data_out, system->last_screen_cache, copy_size * sizeof(float));
        *width = copy_w;
        *height = copy_h;
        return 0;
    }

    if (system->last_screen_cache) {
        size_t copy_w = system->last_width > 0 ? system->last_width : CO_SCREEN_WIDTH;
        size_t copy_h = system->last_height > 0 ? system->last_height : CO_SCREEN_HEIGHT;
        size_t copy_size = copy_w * copy_h * CO_SCREEN_CHANNELS;
        memcpy(data_out, system->last_screen_cache, copy_size * sizeof(float));
        *width = copy_w;
        *height = copy_h;
        return 0;
    }

    return -1;
}

int co_screenshot_compare(COSystem* system, const float* before, const float* after, size_t w, size_t h, float* diff_out) {
    if (!system || !before || !after || !diff_out) return -1;
    size_t n = w * h * CO_SCREEN_CHANNELS;
    float mse = pixel_diff(before, after, n);
    *diff_out = mse;
    return mse > 0.05f ? 1 : 0;
}

int co_screenshot_find_change(COSystem* system, const float* before, const float* after, size_t w, size_t h, float* changed_region) {
    if (!system || !before || !after || !changed_region) return -1;
    int grid = 8;
    float gw = (float)w / (float)grid;
    float gh = (float)h / (float)grid;
    float min_x = (float)w, min_y = (float)h, max_x = 0, max_y = 0;
    int found_change = 0;

    for (int gy = 0; gy < grid; gy++) {
        for (int gx = 0; gx < grid; gx++) {
            int sx = (int)(gx * gw);
            int sy = (int)(gy * gh);
            int ex = (int)((gx + 1) * gw);
            int ey = (int)((gy + 1) * gh);
            float sum_diff = 0;
            int count = 0;
            for (int py = sy; py < ey && py < (int)h; py++) {
                for (int px = sx; px < ex && px < (int)w; px++) {
                    size_t idx = (size_t)(py * w + px) * CO_SCREEN_CHANNELS;
                    float d = 0;
                    for (size_t c = 0; c < CO_SCREEN_CHANNELS; c++) {
                        float diff = before[idx + c] - after[idx + c];
                        d += diff * diff;
                    }
                    sum_diff += d;
                    count++;
                }
            }
            if (count > 0 && sum_diff / (float)count > 0.01f) {
                if ((float)sx < min_x) min_x = (float)sx;
                if ((float)sy < min_y) min_y = (float)sy;
                if ((float)ex > max_x) max_x = (float)ex;
                if ((float)ey > max_y) max_y = (float)ey;
                found_change = 1;
            }
        }
    }

    if (found_change) {
        changed_region[0] = min_x;
        changed_region[1] = min_y;
        changed_region[2] = max_x;
        changed_region[3] = max_y;
        return 1;
    }
    return 0;
}

int co_safety_validate_action(COSystem* system, const COAction* action, char* reason_out, size_t reason_size) {
    if (!system || !action) return -1;
    if (!system->config.enable_safety_check) return 0;

    for (size_t i = 0; i < system->num_safety_rules; i++) {
        if (!system->safety_rules[i].is_enabled) continue;
        if (system->safety_rules[i].check_func) {
            if (system->safety_rules[i].check_func(action, system) != 0) {
                if (reason_out && reason_size > 0) {
                    snprintf(reason_out, reason_size, "安全规则 %d 阻止: %s", system->safety_rules[i].rule_id, system->safety_rules[i].description);
                }
                return -1;
            }
        }
    }

    if (is_destructive_action(action)) {
        if (action->confidence < system->config.safety_confirm_threshold) {
            if (reason_out && reason_size > 0) {
                snprintf(reason_out, reason_size, "破坏性操作置信度不足: %.2f < %.2f", (double)action->confidence, (double)system->config.safety_confirm_threshold);
            }
            return -2;
        }
    }

    return 0;
}

int co_safety_add_rule(COSystem* system, COSafetyRule rule) {
    if (!system) return -1;
    if (system->num_safety_rules >= CO_SAFETY_RULES) return -2;
    rule.rule_id = (int)system->num_safety_rules;
    system->safety_rules[system->num_safety_rules++] = rule;
    return rule.rule_id;
}

int co_safety_remove_rule(COSystem* system, int rule_id) {
    if (!system) return -1;
    for (size_t i = 0; i < system->num_safety_rules; i++) {
        if (system->safety_rules[i].rule_id == rule_id) {
            for (size_t j = i; j < system->num_safety_rules - 1; j++) system->safety_rules[j] = system->safety_rules[j + 1];
            system->num_safety_rules--;
            return 0;
        }
    }
    return -3;
}

int co_safety_check_destructive(const COAction* action, const void* context) {
    (void)context;
    if (!action) return -1;
    if (is_destructive_action(action)) {
        if (action->file_path[0] != '\0') {
            const char* critical_paths[] = {"C:\\Windows", "C:\\Program Files", "/etc", "/usr", "/bin", "/boot"};
            for (size_t i = 0; i < 6; i++) {
                if (strstr(action->file_path, critical_paths[i]) == action->file_path) return -1;
            }
        }
    }
    return 0;
}

int co_record_demo_start(COSystem* system) {
    if (!system) return -1;
    if (!system->config.enable_demo_recording) return -5;
    if (system->recording_active) return -3;

    {
        size_t screen_bytes = (size_t)CO_MAX_DEMO_FRAMES * CO_SCREEN_WIDTH * CO_SCREEN_HEIGHT * CO_SCREEN_CHANNELS * sizeof(float);
        size_t action_bytes = (size_t)CO_MAX_DEMO_FRAMES * sizeof(COAction);
        float* new_screen = (float*)realloc(system->demo_screen_buffer, screen_bytes);
        COAction* new_action = (COAction*)realloc(system->demo_action_buffer, action_bytes);
        if (!new_screen || !new_action) {
            if (new_screen) system->demo_screen_buffer = new_screen;
            if (new_action) system->demo_action_buffer = new_action;
            return -4;
        }
        system->demo_screen_buffer = new_screen;
        system->demo_action_buffer = new_action;
        system->demo_buffer_capacity = CO_MAX_DEMO_FRAMES;
    }

    system->demo_frame_count = 0;
    system->recording_active = 1;
    return 0;
}

int co_record_demo_frame(COSystem* system, const float* screen_data, size_t w, size_t h, const COAction* action) {
    if (!system || !screen_data || !action) return -1;
    if (!system->recording_active) return -2;
    if (system->demo_frame_count >= system->demo_buffer_capacity) return -3;

    size_t offset = system->demo_frame_count * CO_SCREEN_WIDTH * CO_SCREEN_HEIGHT * CO_SCREEN_CHANNELS;
    for (size_t y = 0; y < h && y < CO_SCREEN_HEIGHT; y++) {
        for (size_t x = 0; x < w && x < CO_SCREEN_WIDTH; x++) {
            size_t src_idx = (y * w + x) * 3;
            size_t dst_idx = offset + (y * CO_SCREEN_WIDTH + x) * 3;
            system->demo_screen_buffer[dst_idx] = screen_data[src_idx];
            system->demo_screen_buffer[dst_idx + 1] = screen_data[src_idx + 1];
            system->demo_screen_buffer[dst_idx + 2] = screen_data[src_idx + 2];
        }
    }

    memcpy(&system->demo_action_buffer[system->demo_frame_count], action, sizeof(COAction));
    system->demo_frame_count++;
    return 0;
}

int co_record_demo_stop(COSystem* system) {
    if (!system) return -1;
    if (!system->recording_active) return -2;
    system->recording_active = 0;
    return (int)system->demo_frame_count;
}

int co_record_demo_save(COSystem* system, const char* filepath) {
    if (!system || !filepath) return -1;
    FILE* f = fopen(filepath, "wb");
    if (!f) return -4;

    uint32_t magic = 0x444D4F43;
    uint32_t num_frames = (uint32_t)system->demo_frame_count;
    uint32_t sw = CO_SCREEN_WIDTH;
    uint32_t sh = CO_SCREEN_HEIGHT;
    fwrite(&magic, sizeof(magic), 1, f);
    fwrite(&num_frames, sizeof(num_frames), 1, f);
    fwrite(&sw, sizeof(sw), 1, f);
    fwrite(&sh, sizeof(sh), 1, f);

    for (size_t i = 0; i < system->demo_frame_count; i++) {
        size_t offset = i * CO_SCREEN_WIDTH * CO_SCREEN_HEIGHT * CO_SCREEN_CHANNELS;
        fwrite(&system->demo_screen_buffer[offset], sizeof(float), CO_SCREEN_WIDTH * CO_SCREEN_HEIGHT * CO_SCREEN_CHANNELS, f);
        fwrite(&system->demo_action_buffer[i], sizeof(COAction), 1, f);
    }

    if (system->demo_label) {
        uint32_t label_len = (uint32_t)strlen(system->demo_label);
        fwrite(&label_len, sizeof(label_len), 1, f);
        fwrite(system->demo_label, 1, label_len, f);
    }

    fclose(f);
    return 0;
}

int co_record_demo_load(COSystem* system, const char* filepath) {
    if (!system || !filepath) return -1;
    FILE* f = fopen(filepath, "rb");
    if (!f) return -4;

    uint32_t magic, num_frames, sw, sh;
    if (fread(&magic, sizeof(magic), 1, f) != 1 || magic != 0x444D4F43) { fclose(f); return -5; }
    fread(&num_frames, sizeof(num_frames), 1, f);
    fread(&sw, sizeof(sw), 1, f);
    fread(&sh, sizeof(sh), 1, f);

    if (num_frames > CO_MAX_DEMO_FRAMES) num_frames = CO_MAX_DEMO_FRAMES;

    free(system->demo_screen_buffer);
    free(system->demo_action_buffer);
    free(system->demo_label);

    system->demo_buffer_capacity = num_frames;
    system->demo_frame_count = num_frames;
    system->demo_screen_buffer = (float*)malloc(num_frames * CO_SCREEN_WIDTH * CO_SCREEN_HEIGHT * CO_SCREEN_CHANNELS * sizeof(float));
    system->demo_action_buffer = (COAction*)malloc(num_frames * sizeof(COAction));
    system->demo_label = NULL;

    for (uint32_t i = 0; i < num_frames; i++) {
        size_t offset = i * CO_SCREEN_WIDTH * CO_SCREEN_HEIGHT * CO_SCREEN_CHANNELS;
        fread(&system->demo_screen_buffer[offset], sizeof(float), CO_SCREEN_WIDTH * CO_SCREEN_HEIGHT * CO_SCREEN_CHANNELS, f);
        fread(&system->demo_action_buffer[i], sizeof(COAction), 1, f);
    }

    if (fread(&magic, sizeof(magic), 1, f) == 1) {
        uint32_t label_len = magic;
        if (label_len > 0 && label_len < 1024) {
            system->demo_label = (char*)calloc(label_len + 1, 1);
            fread(system->demo_label, 1, label_len, f);
        }
    }

    fclose(f);
    return (int)num_frames;
}

int co_record_demo_replay(COSystem* system, int (*progress_cb)(float, const char*, void*), void* user_data) {
    if (!system) return -1;
    if (system->demo_frame_count == 0) return -2;

    COPlan replay_plan;
    memset(&replay_plan, 0, sizeof(COPlan));
    replay_plan.num_actions = system->demo_frame_count;

    float last_frame[CO_SCREEN_WIDTH * CO_SCREEN_HEIGHT * CO_SCREEN_CHANNELS];
    int has_last_frame = 0;

    for (size_t i = 0; i < system->demo_frame_count && i < CO_MAX_ACTIONS; i++) {
        memcpy(&replay_plan.actions[i], &system->demo_action_buffer[i], sizeof(COAction));
        size_t offset = i * CO_SCREEN_WIDTH * CO_SCREEN_HEIGHT * CO_SCREEN_CHANNELS;

        if (has_last_frame) {
            float diff = 0;
            int has_change = co_screenshot_compare(system, last_frame, &system->demo_screen_buffer[offset], CO_SCREEN_WIDTH, CO_SCREEN_HEIGHT, &diff);
            if (!has_change) continue;
        }

        memcpy(last_frame, &system->demo_screen_buffer[offset], CO_SCREEN_WIDTH * CO_SCREEN_HEIGHT * CO_SCREEN_CHANNELS * sizeof(float));
        has_last_frame = 1;
    }

    return co_execute_plan(system, &replay_plan, progress_cb, user_data);
}

int co_get_task_stats(COSystem* system, size_t* completed, size_t* failed, float* success_rate) {
    if (!system) return -1;
    if (completed) *completed = system->tasks_completed;
    if (failed) *failed = system->tasks_failed;
    if (success_rate) *success_rate = system->avg_success_rate;
    return 0;
}

int co_reset_stats(COSystem* system) {
    if (!system) return -1;
    system->tasks_completed = 0;
    system->tasks_failed = 0;
    system->avg_success_rate = 0.0f;
    system->action_history_count = 0;
    return 0;
}

int co_get_system_status(COSystem* system, char* status_out, size_t status_size) {
    if (!system || !status_out) return -1;
    snprintf(status_out, status_size,
        "{"
        "\"initialized\":%d,"
        "\"tasks_completed\":%zu,"
        "\"tasks_failed\":%zu,"
        "\"success_rate\":%.3f,"
        "\"elements_detected\":%zu,"
        "\"ocr_results\":%zu,"
        "\"recording\":%d,"
        "\"demo_frames\":%zu,"
        "\"safety_rules\":%zu"
        "}",
        system->initialized,
        system->tasks_completed,
        system->tasks_failed,
        (double)system->avg_success_rate,
        system->current_plan.num_elements,
        system->current_plan.num_ocr_results,
        system->recording_active,
        system->demo_frame_count,
        system->num_safety_rules);
    return 0;
}

int co_learn_from_demo(COSystem* system, const float* screen_sequence, const COAction* action_sequence, size_t num_frames, size_t width, size_t height, const char* task_label) {
    if (!system || !screen_sequence || !action_sequence || !task_label) return -1;
    if (num_frames == 0) return -2;

    size_t learn_samples = num_frames < 100 ? num_frames : 100;
    float screen_input[64 * 64] = {0};

    for (size_t s = 0; s < learn_samples; s++) {
        size_t frame_idx = s * num_frames / learn_samples;
        const float* frame = screen_sequence + frame_idx * width * height * 3;
        resize_image_bilinear(frame, width, height, 3, screen_input, 64, 64);

        float screen_embed[1024] = {0};
        co_forward(system, system->screen_encoder, screen_input, 64*64, screen_embed, 1024);

        float policy_input[1024 + 512] = {0};
        memcpy(policy_input, screen_embed, 1024 * sizeof(float));

        const COAction* action = &action_sequence[frame_idx < num_frames ? frame_idx : num_frames - 1];
        float policy_target[CO_ACTION_MAX * 10] = {0};
        policy_target[0] = (float)action->action_type;
        policy_target[1] = (float)action->x / (float)(width > 0 ? width : 1);
        policy_target[2] = (float)action->y / (float)(height > 0 ? height : 1);
        policy_target[3] = (float)action->target_x / (float)(width > 0 ? width : 1);
        policy_target[4] = (float)action->target_y / (float)(height > 0 ? height : 1);
        policy_target[5] = (float)action->key_code;
        policy_target[6] = (float)action->modifier_flags;
        policy_target[7] = action->duration;
        policy_target[9] = 1.0f;

        float policy_out[CO_ACTION_MAX * 10] = {0};
        co_forward(system, system->action_policy, policy_input, 1024+512, policy_out, CO_ACTION_MAX * 10);
    }

    free(system->demo_label);
    system->demo_label = (char*)malloc(strlen(task_label) + 1);
    if (system->demo_label) strcpy(system->demo_label, task_label);

    return 0;
}
