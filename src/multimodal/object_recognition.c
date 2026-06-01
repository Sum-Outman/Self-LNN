/**
 * @file object_recognition.c
 * @brief 增强物体识别与场景理解完整实现
 * 使用真实特征提取算法（HOG + 归一化互相关模板匹配）
 */
#include "selflnn/multimodal/object_recognition.h"
#include "selflnn/multimodal/depth_estimation.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/secure_random.h" /* CfC Xavier初始化 */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* 梯度直方图方向数 */
#define HOG_BINS 9
/* 每个cell的像素数 */
#define HOG_CELL_SIZE 8

/* CfC深度学习管道常量 —— HOG特征维数=输入层，隐藏层=64，输出=类别数 */
#define CFC_OR_INPUT_DIM  128
#define CFC_OR_HIDDEN_DIM 64

struct ObjectRecognizer {
    float category_templates[OR_MAX_CATEGORIES][128];
    int category_count;
    char category_names[OR_MAX_CATEGORIES][64];
    int initialized;
/* 显式训练状态标志。
     * 0=未训练(使用HSV启发式默认类别模板)
     * 1=已训练(类别模板通过or_train_from_examples从真实数据学习)
     * 替代原有内联原型向量norm>0.25f的启发式判断。 */
    int is_trained;
    SceneType last_scene;
    /* 深度估计集成 —— H-010: 接入真实深度替代硬编码物距 */
    DepthEstimator* depth_estimator;
    const float* depth_map;
    int depth_map_w;
    int depth_map_h;
    int has_depth_map;
/* CfC深度学习管道 —— 输入层=HOG特征128维，隐藏层64维，输出层=类别数 */
    int is_cfc_trained;
    float cfc_hidden[CFC_OR_HIDDEN_DIM];
    float cfc_W_gx[CFC_OR_HIDDEN_DIM * CFC_OR_INPUT_DIM];
    float cfc_W_ax[CFC_OR_HIDDEN_DIM * CFC_OR_INPUT_DIM];
    float cfc_W_gh[CFC_OR_HIDDEN_DIM * CFC_OR_HIDDEN_DIM];
    float cfc_W_ah[CFC_OR_HIDDEN_DIM * CFC_OR_HIDDEN_DIM];
    float cfc_b_g[CFC_OR_HIDDEN_DIM];
    float cfc_b_a[CFC_OR_HIDDEN_DIM];
    float cfc_W_cls[OR_MAX_CATEGORIES * CFC_OR_HIDDEN_DIM];
    float cfc_b_cls[OR_MAX_CATEGORIES];
    float cfc_logits[OR_MAX_CATEGORIES];
    float cfc_tau;
    float cfc_dt;
};

/**
 * @brief 计算图像梯度（Sobel算子，真实实现）
 */
static void compute_gradients(const float* image, int w, int h, int ch,
                              float* grad_mag, float* grad_ori, int* g_w, int* g_h) {
    int gw = w - 2, gh = h - 2;
    if (gw < 1 || gh < 1) { *g_w = 0; *g_h = 0; return; }
    *g_w = gw; *g_h = gh;
    
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            /* 对每个通道取灰度值（加权平均） */
            float gy_p1 = 0, gy_m1 = 0, gx_p1 = 0, gx_m1 = 0;
            for (int c = 0; c < ch; c++) {
                float wc = (c == 0) ? 0.3f : ((c == 1) ? 0.59f : 0.11f);
                gy_p1 += image[((y + 1) * w + x) * ch + c] * wc;
                gy_m1 += image[((y - 1) * w + x) * ch + c] * wc;
                gx_p1 += image[(y * w + (x + 1)) * ch + c] * wc;
                gx_m1 += image[(y * w + (x - 1)) * ch + c] * wc;
            }
            float gx = gx_p1 - gx_m1;
            float gy = gy_p1 - gy_m1;
            int idx = (y - 1) * gw + (x - 1);
            grad_mag[idx] = sqrtf(gx * gx + gy * gy);
            grad_ori[idx] = atan2f(gy, gx);
        }
    }
}

/**
 * @brief 计算HOG特征描述子（真实实现）
 */
static void compute_hog_features(const float* grad_mag, const float* grad_ori,
                                  int gw, int gh, float* features, int max_features) {
    int cells_x = gw / HOG_CELL_SIZE;
    int cells_y = gh / HOG_CELL_SIZE;
    if (cells_x < 1) cells_x = 1;
    if (cells_y < 1) cells_y = 1;
    
    int total_cells = cells_x * cells_y;
    int feature_count = total_cells * HOG_BINS;
    if (feature_count > max_features) feature_count = max_features;
    
    memset(features, 0, feature_count * sizeof(float));
    
    for (int cy = 0; cy < cells_y; cy++) {
        for (int cx = 0; cx < cells_x; cx++) {
            float cell_hist[HOG_BINS] = {0};
            float cell_max = 0;
            
            for (int dy = 0; dy < HOG_CELL_SIZE; dy++) {
                for (int dx = 0; dx < HOG_CELL_SIZE; dx++) {
                    int gx = cx * HOG_CELL_SIZE + dx;
                    int gy = cy * HOG_CELL_SIZE + dy;
                    if (gx >= gw || gy >= gh) continue;
                    int idx = gy * gw + gx;
                    float mag = grad_mag[idx];
                    float ori = grad_ori[idx] + (float)M_PI; /* 转到[0, 2π] */
                    float bin_f = ori * (float)HOG_BINS / (2.0f * (float)M_PI);
                    int bin0 = (int)bin_f % HOG_BINS;
                    int bin1 = (bin0 + 1) % HOG_BINS;
                    float w1 = bin_f - floorf(bin_f);
                    float w0 = 1.0f - w1;
                    cell_hist[bin0] += mag * w0;
                    cell_hist[bin1] += mag * w1;
                }
            }
            
            /* 找最大值用于归一化 */
            for (int b = 0; b < HOG_BINS; b++) {
                if (cell_hist[b] > cell_max) cell_max = cell_hist[b];
            }
            
            /* L2归一化 */
            float l2_norm = 0;
            for (int b = 0; b < HOG_BINS; b++) {
                l2_norm += cell_hist[b] * cell_hist[b];
            }
            l2_norm = sqrtf(l2_norm) + 1e-6f;
            
            int feat_base = (cy * cells_x + cx) * HOG_BINS;
            for (int b = 0; b < HOG_BINS && (feat_base + b) < feature_count; b++) {
                features[feat_base + b] = cell_hist[b] / l2_norm;
            }
        }
    }
}

/**
 * @brief 计算两向量的归一化互相关（真实模板匹配）
 */
static float normalized_cross_correlation(const float* a, const float* b, int len) {
    float sum_a = 0, sum_b = 0, sum_ab = 0, sum_aa = 0, sum_bb = 0;
    for (int i = 0; i < len; i++) {
        sum_a += a[i];
        sum_b += b[i];
    }
    float mean_a = sum_a / (float)len;
    float mean_b = sum_b / (float)len;
    for (int i = 0; i < len; i++) {
        float da = a[i] - mean_a;
        float db = b[i] - mean_b;
        sum_ab += da * db;
        sum_aa += da * da;
        sum_bb += db * db;
    }
    float denom = sqrtf(sum_aa * sum_bb);
    if (denom < 1e-10f) return 0;
    return sum_ab / denom;
}

/* ======================================================================== */
/* CfC深度学习基础数学函数 —— 从image_recognition_deep.c移植  */
/* ======================================================================== */

static float _or_cfc_sig(float x) { return 1.0f / (1.0f + expf(-x)); }

static float _or_cfc_tanh_f(float x) {
    float e2x = expf(2.0f * x);
    return (e2x - 1.0f) / (e2x + 1.0f);
}

/* Xavier初始化 —— 适用于tanh/sigmoid激活函数层 */
static void _or_cfc_xavier_init(float* w, int fan_in, int fan_out) {
    float scale = sqrtf(2.0f / (float)(fan_in + fan_out));
    for (int i = 0; i < fan_in * fan_out; i++) {
        float u1 = secure_random_float();
        float u2 = secure_random_float();
        if (u1 < 1e-7f) u1 = 1e-7f;
        w[i] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2) * scale;
    }
}

/* 向量运算库 */
static void _or_cfc_mat_vec_mul(const float* mat, const float* vec, float* out, int r, int c) {
    for (int i = 0; i < r; i++) {
        out[i] = 0.0f;
        for (int j = 0; j < c; j++) out[i] += mat[i * c + j] * vec[j];
    }
}
static void _or_cfc_vec_add(float* a, const float* b, int n) { for (int i = 0; i < n; i++) a[i] += b[i]; }
static void _or_cfc_vec_hadamard(float* a, const float* b, int n) { for (int i = 0; i < n; i++) a[i] *= b[i]; }
static void _or_cfc_vec_sigmoid(float* a, int n) { for (int i = 0; i < n; i++) a[i] = _or_cfc_sig(a[i]); }
static void _or_cfc_vec_tanh(float* a, int n) { for (int i = 0; i < n; i++) a[i] = _or_cfc_tanh_f(a[i]); }
static void _or_cfc_vec_scale(float* a, float s, int n) { for (int i = 0; i < n; i++) a[i] *= s; }
static void _or_cfc_vec_copy(float* d, const float* s, int n) { memcpy(d, s, n * sizeof(float)); }

static float _or_cfc_vec_dot(const float* a, const float* b, int n) {
    float s = 0.0f; for (int i = 0; i < n; i++) s += a[i] * b[i]; return s;
}

/* Softmax归一化 */
static void _or_cfc_softmax(float* logits, int n) {
    float mv = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > mv) mv = logits[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { logits[i] = expf(logits[i] - mv); sum += logits[i]; }
    if (sum > 1e-10f) for (int i = 0; i < n; i++) logits[i] /= sum;
}

/* CfC ODE步进 —— 核心连续时间液态神经网络前向传播
 * 公式: h(t+dt) = h(t)*exp(-dt/τ) + (1-exp(-dt/τ))*σ(W_gx·x+W_gh·h+b_g)⊙tanh(W_ax·x+W_ah·h+b_a) */
static void _or_cfc_ode_step(const float* in, int in_dim,
                              const float* W_gx, const float* W_ax,
                              const float* W_gh, const float* W_ah,
                              const float* b_g, const float* b_a,
                              float* h, int h_dim, float tau, float dt) {
    float* gate = (float*)safe_malloc((size_t)h_dim * sizeof(float) * 4);
    if (!gate) return;
    float* act  = gate + h_dim;
    float* t1   = act + h_dim;
    float* t2   = t1 + h_dim;
    memset(gate, 0, h_dim * sizeof(float));
    memset(act, 0, h_dim * sizeof(float));
    _or_cfc_mat_vec_mul(W_gx, in, t1, h_dim, in_dim);
    _or_cfc_mat_vec_mul(W_gh, h, t2, h_dim, h_dim);
    _or_cfc_vec_add(t1, t2, h_dim); _or_cfc_vec_add(t1, b_g, h_dim); _or_cfc_vec_sigmoid(t1, h_dim);
    _or_cfc_vec_copy(gate, t1, h_dim);
    _or_cfc_mat_vec_mul(W_ax, in, t1, h_dim, in_dim);
    _or_cfc_mat_vec_mul(W_ah, h, t2, h_dim, h_dim);
    _or_cfc_vec_add(t1, t2, h_dim); _or_cfc_vec_add(t1, b_a, h_dim); _or_cfc_vec_tanh(t1, h_dim);
    _or_cfc_vec_copy(act, t1, h_dim);
    _or_cfc_vec_hadamard(gate, act, h_dim);
    float decay = expf(-dt / tau);
    for (int i = 0; i < h_dim; i++) h[i] = h[i] * decay + (1.0f - decay) * gate[i];
    safe_free((void**)&gate);
}

/* CfC ODE步反向传播 —— 计算参数梯度用于训练 */
static void _or_cfc_ode_step_backward(const float* in, int in_dim,
                                       const float* W_gx, const float* W_ax,
                                       const float* W_gh, const float* W_ah,
                                       const float* b_g, const float* b_a,
                                       float* dW_gx, float* dW_ax,
                                       float* dW_gh, float* dW_ah,
                                       float* db_g, float* db_a,
                                       const float* h, const float* dL_dh_new,
                                       float* dL_dh, float* dL_din,
                                       int h_dim, float tau, float dt) {
    float* buf = (float*)safe_malloc((size_t)h_dim * sizeof(float) * 8);
    if (!buf) return;
    float* pre_gate = buf + h_dim * 0;
    float* pre_act  = buf + h_dim * 1;
    float* gate     = buf + h_dim * 2;
    float* act      = buf + h_dim * 3;
    float* t1       = buf + h_dim * 4;
    float* t2       = buf + h_dim * 5;
    float* d_driver = buf + h_dim * 6;
    float* d_pre_gate = buf + h_dim * 7;
    float* d_gate   = d_pre_gate;
    float* d_act    = pre_gate;
    float* d_pre_act = pre_act;
    (void)d_gate; (void)d_act; (void)d_pre_act;
    memset(pre_gate, 0, h_dim * sizeof(float));
    memset(pre_act, 0, h_dim * sizeof(float));
    _or_cfc_mat_vec_mul(W_gx, in, t1, h_dim, in_dim);
    _or_cfc_mat_vec_mul(W_gh, h, t2, h_dim, h_dim);
    for (int i = 0; i < h_dim; i++) pre_gate[i] = t1[i] + t2[i] + b_g[i];
    memcpy(gate, pre_gate, h_dim * sizeof(float));
    _or_cfc_vec_sigmoid(gate, h_dim);
    _or_cfc_mat_vec_mul(W_ax, in, t1, h_dim, in_dim);
    _or_cfc_mat_vec_mul(W_ah, h, t2, h_dim, h_dim);
    for (int i = 0; i < h_dim; i++) pre_act[i] = t1[i] + t2[i] + b_a[i];
    memcpy(act, pre_act, h_dim * sizeof(float));
    _or_cfc_vec_tanh(act, h_dim);
    float decay = expf(-dt / tau);
    for (int i = 0; i < h_dim; i++) {
        d_driver[i] = (1.0f - decay) * dL_dh_new[i];
        d_gate[i] = d_driver[i] * act[i];
        d_act[i] = d_driver[i] * gate[i];
    }
    for (int i = 0; i < h_dim; i++) {
        d_pre_gate[i] = d_gate[i] * gate[i] * (1.0f - gate[i]);
        d_pre_act[i] = d_act[i] * (1.0f - act[i] * act[i]);
    }
    for (int i = 0; i < h_dim; i++) {
        for (int j = 0; j < in_dim; j++) {
            dW_gx[i * in_dim + j] += d_pre_gate[i] * in[j];
            dW_ax[i * in_dim + j] += d_pre_act[i] * in[j];
        }
        for (int j = 0; j < h_dim; j++) {
            dW_gh[i * h_dim + j] += d_pre_gate[i] * h[j];
            dW_ah[i * h_dim + j] += d_pre_act[i] * h[j];
        }
        db_g[i] += d_pre_gate[i];
        db_a[i] += d_pre_act[i];
    }
    if (dL_din) {
        for (int j = 0; j < in_dim; j++) {
            float sum = 0.0f;
            for (int i = 0; i < h_dim; i++)
                sum += W_gx[i * in_dim + j] * d_pre_gate[i] + W_ax[i * in_dim + j] * d_pre_act[i];
            dL_din[j] = sum;
        }
    }
    if (dL_dh) {
        for (int i = 0; i < h_dim; i++) {
            float sum = 0.0f;
            for (int k = 0; k < h_dim; k++)
                sum += W_gh[k * h_dim + i] * d_pre_gate[k] + W_ah[k * h_dim + i] * d_pre_act[k];
            dL_dh[i] = decay * dL_dh_new[i] + sum;
        }
    }
    safe_free((void**)&buf);
}

ObjectRecognizer* object_recognizer_create(void) {
    ObjectRecognizer* or_obj = (ObjectRecognizer*)safe_calloc(1, sizeof(ObjectRecognizer));
    if (!or_obj) return NULL;
    or_obj->initialized = 1;
    or_obj->depth_estimator = NULL;
    or_obj->depth_map = NULL;
    or_obj->depth_map_w = 0;
    or_obj->depth_map_h = 0;
    or_obj->has_depth_map = 0;

/* 初始化CfC深度学习管道 —— Xavier初始化权重，零初始化偏置和隐藏状态 */
    or_obj->is_cfc_trained = 0;
    or_obj->cfc_tau = 0.1f;
    or_obj->cfc_dt = 0.05f;
    memset(or_obj->cfc_hidden, 0, CFC_OR_HIDDEN_DIM * sizeof(float));
    memset(or_obj->cfc_b_g, 0, CFC_OR_HIDDEN_DIM * sizeof(float));
    memset(or_obj->cfc_b_a, 0, CFC_OR_HIDDEN_DIM * sizeof(float));
    memset(or_obj->cfc_b_cls, 0, OR_MAX_CATEGORIES * sizeof(float));
    memset(or_obj->cfc_logits, 0, OR_MAX_CATEGORIES * sizeof(float));
    _or_cfc_xavier_init(or_obj->cfc_W_gx, CFC_OR_INPUT_DIM, CFC_OR_HIDDEN_DIM);
    _or_cfc_xavier_init(or_obj->cfc_W_ax, CFC_OR_INPUT_DIM, CFC_OR_HIDDEN_DIM);
    _or_cfc_xavier_init(or_obj->cfc_W_gh, CFC_OR_HIDDEN_DIM, CFC_OR_HIDDEN_DIM);
    _or_cfc_xavier_init(or_obj->cfc_W_ah, CFC_OR_HIDDEN_DIM, CFC_OR_HIDDEN_DIM);
    _or_cfc_xavier_init(or_obj->cfc_W_cls, CFC_OR_HIDDEN_DIM, OR_MAX_CATEGORIES);

    /* 预定义类别名称，基于HSV颜色空间生成具有基本区分能力的初始模板 */
    const char* cats[] = {"人","车辆","动物","家具","电子设备","食物","工具","建筑物","植物","自然景观"};
    int cat_count = sizeof(cats) / sizeof(cats[0]);

    /* 类别典型HSV参数：{色相均值°,色相宽度°,饱和度均值,饱和度宽度,明度均值,明度宽度,纹理复杂度,边缘密度,色相偏移°,备用} */
    float hsv_params[10][10] = {
        {22.0f,25.0f,0.45f,0.25f,0.55f,0.30f,0.35f,0.20f,8.0f,0.0f},   /* 人：肤色基调 */
        {200.0f,180.0f,0.12f,0.12f,0.65f,0.25f,0.15f,0.55f,0.0f,0.0f}, /* 车辆：金属色 */
        {35.0f,50.0f,0.40f,0.30f,0.48f,0.32f,0.55f,0.30f,15.0f,0.0f},  /* 动物：暖色皮毛 */
        {28.0f,30.0f,0.35f,0.20f,0.50f,0.28f,0.40f,0.35f,5.0f,0.0f},   /* 家具：木质调 */
        {220.0f,120.0f,0.08f,0.10f,0.32f,0.22f,0.10f,0.70f,0.0f,0.0f}, /* 电子设备：暗色调 */
        {18.0f,55.0f,0.65f,0.30f,0.58f,0.28f,0.50f,0.25f,22.0f,0.0f},  /* 食物：高饱和暖色 */
        {180.0f,160.0f,0.08f,0.10f,0.55f,0.30f,0.12f,0.65f,0.0f,0.0f}, /* 工具：金属色 */
        {30.0f,60.0f,0.10f,0.12f,0.52f,0.28f,0.30f,0.45f,10.0f,0.0f},  /* 建筑物：灰棕调 */
        {120.0f,80.0f,0.55f,0.35f,0.42f,0.30f,0.65f,0.38f,0.0f,0.0f},  /* 植物：绿色调 */
        {190.0f,130.0f,0.40f,0.35f,0.50f,0.35f,0.50f,0.30f,0.0f,0.0f}  /* 自然景观：蓝绿调 */
    };

    for (int i = 0; i < cat_count && i < OR_MAX_CATEGORIES; i++) {
        snprintf(or_obj->category_names[i], 64, "%s", cats[i]);
        memset(or_obj->category_templates[i], 0, 128 * sizeof(float));

        /* 基于HSV颜色空间生成初始模板特征：
         * 128维 = 16色相区间 × 4饱和等级 × 2明度等级 */
        float* p = hsv_params[i];
        float h_mean = p[0], h_width = p[1];
        float s_mean = p[2], s_width = p[3];
        float v_mean = p[4], v_width = p[5];
        float texture = p[6], edge_density = p[7];
        float h_shift = p[8];

        for (int h_bin = 0; h_bin < 16; h_bin++) {
            float bin_c = (float)h_bin * 22.5f;
            float h_diff = bin_c - (h_mean + h_shift);
            while (h_diff > 180.0f) h_diff -= 360.0f;
            while (h_diff < -180.0f) h_diff += 360.0f;
            float h_r = expf(-(h_diff * h_diff) / (2.0f * h_width * h_width));

            for (int s_bin = 0; s_bin < 4; s_bin++) {
                float s_c = (float)s_bin * 0.25f + 0.125f;
                float s_diff = s_c - s_mean;
                float s_r = expf(-(s_diff * s_diff) / (2.0f * s_width * s_width));

                for (int v_bin = 0; v_bin < 2; v_bin++) {
                    float v_c = (float)v_bin * 0.5f + 0.25f;
                    float v_diff = v_c - v_mean;
                    float v_r = expf(-(v_diff * v_diff) / (2.0f * v_width * v_width));

                    int fi = h_bin * 8 + s_bin * 2 + v_bin;
                    if (fi >= 0 && fi < 128) {
                        float cv = h_r * s_r * v_r;
                        float tm = 0.3f + 0.7f * (1.0f - texture * 0.5f);
                        float eb = 1.0f + edge_density * ((float)(h_bin % 4) / 8.0f);
                        or_obj->category_templates[i][fi] = cv * tm * eb;
                    }
                }
            }
        }

        /* L2归一化到目标范数0.5，确保后续训练可用（训练检查阈值0.25） */
        float l2 = 0.0f;
        for (int d = 0; d < 128; d++) l2 += or_obj->category_templates[i][d] * or_obj->category_templates[i][d];
        l2 = sqrtf(l2);
        if (l2 > 1e-10f) {
            float tn = 0.5f;
            for (int d = 0; d < 128; d++) or_obj->category_templates[i][d] = or_obj->category_templates[i][d] * tn / l2;
        } else {
            /* 退化保护：最小范数模板 */
            for (int d = 0; d < 128; d++)
                or_obj->category_templates[i][d] = (float)((d + i * 13) % 128) * 0.003f;
        }
        or_obj->category_count++;
    }
    or_obj->last_scene = SCENE_UNKNOWN;
    return or_obj;
}

void object_recognizer_free(ObjectRecognizer* or_obj) { safe_free((void**)&or_obj); }

/* ======================================================================== */
/* CfC深度学习前向传播 —— 核心分类管道                      */
/* ======================================================================== */

/**
 * @brief CfC前向传播 —— 将HOG特征输入CfC液态神经网络
 * 管道: HOG特征(128维) → CfC ODE步进(隐藏层64维) → 线性分类头 → softmax概率
 * @param or_obj 识别器句柄
 * @param hog_features HOG特征向量 [128]
 * @param probs 输出类别概率 [category_count]，调用方分配
 * @return 最优类别ID，失败返回-1
 */
static int object_cfc_forward(ObjectRecognizer* or_obj, const float* hog_features, float* probs) {
    if (!or_obj || !hog_features || !probs) return -1;
    if (!or_obj->is_cfc_trained) return -1;
    int hd = CFC_OR_HIDDEN_DIM;
    int in_dim = CFC_OR_INPUT_DIM;
    int nc = or_obj->category_count;
    if (nc <= 0 || nc > OR_MAX_CATEGORIES) return -1;

    /* 归一化输入HOG特征 */
    float input_norm[CFC_OR_INPUT_DIM];
    memcpy(input_norm, hog_features, in_dim * sizeof(float));
    float in_norm = 0.0f;
    for (int i = 0; i < in_dim; i++) in_norm += input_norm[i] * input_norm[i];
    in_norm = sqrtf(in_norm) + 1e-8f;
    for (int i = 0; i < in_dim; i++) input_norm[i] /= in_norm;

    /* CfC ODE步进: 隐藏状态演化 */
    memset(or_obj->cfc_hidden, 0, hd * sizeof(float));
    _or_cfc_ode_step(input_norm, in_dim,
                     or_obj->cfc_W_gx, or_obj->cfc_W_ax,
                     or_obj->cfc_W_gh, or_obj->cfc_W_ah,
                     or_obj->cfc_b_g, or_obj->cfc_b_a,
                     or_obj->cfc_hidden, hd, or_obj->cfc_tau, or_obj->cfc_dt);

    /* 线性分类头: logits = W_cls · hidden + b_cls */
    memset(or_obj->cfc_logits, 0, nc * sizeof(float));
    _or_cfc_mat_vec_mul(or_obj->cfc_W_cls, or_obj->cfc_hidden, or_obj->cfc_logits, nc, hd);
    _or_cfc_vec_add(or_obj->cfc_logits, or_obj->cfc_b_cls, nc);

    /* Softmax概率输出 */
    memcpy(probs, or_obj->cfc_logits, nc * sizeof(float));
    _or_cfc_softmax(probs, nc);

    /* 返回最优类别索引 */
    int best = 0;
    for (int i = 1; i < nc; i++) if (probs[i] > probs[best]) best = i;
    return best;
}

/* ======================================================================== */
/* CfC深度学习训练 —— SGD + 交叉熵损失                      */
/* ======================================================================== */

/**
 * @brief CfC训练函数 —— 使用SGD优化器 + 交叉熵损失
 * 训练流程:
 *   1. 前向传播: HOG特征 → CfC ODE → 线性分类头 → softmax概率
 *   2. 损失计算: 交叉熵 L = -log(p_correct)
 *   3. 反向传播: 分类头梯度 → CfC ODE反向传播 → 权重更新
 * @param or_obj 识别器句柄
 * @param features 训练样本特征矩阵 [samples × dim]，行主序
 * @param labels 训练标签 [samples]
 * @param samples 样本数
 * @param dim 特征维度(应为128)
 * @param categories 类别数
 * @param epochs 训练轮数
 * @param lr 学习率
 * @return 0成功，-1失败
 */
int object_cfc_train(ObjectRecognizer* or_obj, const float* features,
                     const int* labels, int samples, int dim, int categories,
                     int epochs, float lr) {
    if (!or_obj || !features || !labels) return -1;
    if (samples <= 0 || dim < CFC_OR_INPUT_DIM || categories <= 0) return -1;
    if (categories > OR_MAX_CATEGORIES) categories = OR_MAX_CATEGORIES;
    int hd = CFC_OR_HIDDEN_DIM;
    int in_dim = CFC_OR_INPUT_DIM;
    int feat_dim = dim < in_dim ? dim : in_dim;

    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;
        int valid_samples = 0;

        for (int s = 0; s < samples; s++) {
            int label = labels[s];
            if (label < 0 || label >= categories) continue;

            const float* sample_features = &features[s * dim];

            /* 归一化输入 */
            float input_norm[CFC_OR_INPUT_DIM];
            memcpy(input_norm, sample_features, feat_dim * sizeof(float));
            if (feat_dim < in_dim)
                memset(input_norm + feat_dim, 0, (in_dim - feat_dim) * sizeof(float));
            float in_norm = 0.0f;
            for (int i = 0; i < in_dim; i++) in_norm += input_norm[i] * input_norm[i];
            in_norm = sqrtf(in_norm) + 1e-8f;
            for (int i = 0; i < in_dim; i++) input_norm[i] /= in_norm;

            /* 1. 前向传播: CfC ODE步进 */
            float hidden_before[CFC_OR_HIDDEN_DIM];
            memset(hidden_before, 0, hd * sizeof(float));
            _or_cfc_ode_step(input_norm, in_dim,
                             or_obj->cfc_W_gx, or_obj->cfc_W_ax,
                             or_obj->cfc_W_gh, or_obj->cfc_W_ah,
                             or_obj->cfc_b_g, or_obj->cfc_b_a,
                             hidden_before, hd, or_obj->cfc_tau, or_obj->cfc_dt);

            /* 2. 线性分类头 */
            memset(or_obj->cfc_logits, 0, categories * sizeof(float));
            _or_cfc_mat_vec_mul(or_obj->cfc_W_cls, hidden_before, or_obj->cfc_logits, categories, hd);
            _or_cfc_vec_add(or_obj->cfc_logits, or_obj->cfc_b_cls, categories);

            /* 3. Softmax */
            float probs[OR_MAX_CATEGORIES];
            memcpy(probs, or_obj->cfc_logits, categories * sizeof(float));
            _or_cfc_softmax(probs, categories);

            /* 4. 交叉熵损失 */
            total_loss += -logf(probs[label] + 1e-10f);
            valid_samples++;

            /* 5. 反向传播: 分类头梯度 dL/dlogits = probs - one_hot(label) */
            float grad_logits[OR_MAX_CATEGORIES];
            for (int i = 0; i < categories; i++)
                grad_logits[i] = probs[i] - (i == label ? 1.0f : 0.0f);

            /* 6. 更新分类头权重: W_cls -= lr * grad_logits × hidden^T */
            for (int i = 0; i < categories; i++) {
                for (int j = 0; j < hd; j++) {
                    or_obj->cfc_W_cls[i * hd + j] -= lr * grad_logits[i] * hidden_before[j];
                }
                or_obj->cfc_b_cls[i] -= lr * grad_logits[i];
            }

            /* 7. CfC ODE参数反向传播: dL/dh = W_cls^T × grad_logits */
            float dL_dh[CFC_OR_HIDDEN_DIM];
            memset(dL_dh, 0, hd * sizeof(float));
            for (int j = 0; j < hd; j++) {
                for (int k = 0; k < categories; k++) {
                    dL_dh[j] += or_obj->cfc_W_cls[k * hd + j] * grad_logits[k];
                }
            }

            /* 8. CfC ODE步反向传播 —— 累积参数梯度 */
            float dW_gx[CFC_OR_HIDDEN_DIM * CFC_OR_INPUT_DIM];
            float dW_ax[CFC_OR_HIDDEN_DIM * CFC_OR_INPUT_DIM];
            float dW_gh[CFC_OR_HIDDEN_DIM * CFC_OR_HIDDEN_DIM];
            float dW_ah[CFC_OR_HIDDEN_DIM * CFC_OR_HIDDEN_DIM];
            float db_g[CFC_OR_HIDDEN_DIM];
            float db_a[CFC_OR_HIDDEN_DIM];
            memset(dW_gx, 0, sizeof(dW_gx)); memset(dW_ax, 0, sizeof(dW_ax));
            memset(dW_gh, 0, sizeof(dW_gh)); memset(dW_ah, 0, sizeof(dW_ah));
            memset(db_g, 0, sizeof(db_g)); memset(db_a, 0, sizeof(db_a));

            float zero_h[CFC_OR_HIDDEN_DIM];
            memset(zero_h, 0, hd * sizeof(float));

            _or_cfc_ode_step_backward(input_norm, in_dim,
                                      or_obj->cfc_W_gx, or_obj->cfc_W_ax,
                                      or_obj->cfc_W_gh, or_obj->cfc_W_ah,
                                      or_obj->cfc_b_g, or_obj->cfc_b_a,
                                      dW_gx, dW_ax, dW_gh, dW_ah,
                                      db_g, db_a,
                                      zero_h, dL_dh,
                                      NULL, NULL,
                                      hd, or_obj->cfc_tau, or_obj->cfc_dt);

            /* 9. 应用CfC参数更新 (SGD) */
            float cfc_lr = lr * 0.5f;
            for (int i = 0; i < hd * in_dim; i++) {
                or_obj->cfc_W_gx[i] -= cfc_lr * dW_gx[i];
                or_obj->cfc_W_ax[i] -= cfc_lr * dW_ax[i];
            }
            for (int i = 0; i < hd * hd; i++) {
                or_obj->cfc_W_gh[i] -= cfc_lr * dW_gh[i];
                or_obj->cfc_W_ah[i] -= cfc_lr * dW_ah[i];
            }
            for (int i = 0; i < hd; i++) {
                or_obj->cfc_b_g[i] -= cfc_lr * db_g[i];
                or_obj->cfc_b_a[i] -= cfc_lr * db_a[i];
            }
        }

        /* 早停: 平均损失低于阈值则停止 */
        if (valid_samples > 0 && total_loss / (float)valid_samples < 0.05f) break;
    }

/* 训练完成后标记CfC管道已训练 */
    or_obj->is_cfc_trained = 1;
    return 0;
}

/* 多尺度滑动窗口检测 */
int or_detect_objects(ObjectRecognizer* or_obj, const float* image, int w, int h, int ch,
    DetectedObject* out, int max_count) {
    if (!or_obj || !image || !out || w <= 0 || h <= 0 || ch <= 0) return 0;

    int count = 0;
    int scales[] = {32, 64, 96};
    int scale_count = sizeof(scales) / sizeof(scales[0]);

    /* 分配梯度缓冲区（最大尺寸） */
    int max_gw = w - 2, max_gh = h - 2;
    if (max_gw < 1 || max_gh < 1) return 0;
    float* grad_mag = (float*)safe_malloc((size_t)(max_gw * max_gh) * sizeof(float));
    float* grad_ori = (float*)safe_malloc((size_t)(max_gw * max_gh) * sizeof(float));
    if (!grad_mag || !grad_ori) {
        if (grad_mag) safe_free((void**)&grad_mag);
        if (grad_ori) safe_free((void**)&grad_ori);
        return 0;
    }

    /* 计算全图梯度一次（用于多尺度滑动窗口） */
    int gw, gh;
    compute_gradients(image, w, h, ch, grad_mag, grad_ori, &gw, &gh);

    for (int s = 0; s < scale_count && count < max_count; s++) {
        int win_size = scales[s];
        int step = win_size / 3;
        /* 确保窗口不小于两个cell */
        if (win_size < HOG_CELL_SIZE * 2) continue;

        for (int y = 0; y + win_size <= h && count < max_count; y += step) {
            for (int x = 0; x + win_size <= w && count < max_count; x += step) {
                /* 提取窗口内梯度数据计算HOG特征 */
                int wg_start = (x > 0) ? x - 1 : 0;
                int wg_end = (x + win_size - 2 < gw) ? x + win_size - 2 : gw;
                int hg_start = (y > 0) ? y - 1 : 0;
                int hg_end = (y + win_size - 2 < gh) ? y + win_size - 2 : gh;
                int wg_w = wg_end - wg_start;
                int wg_h = hg_end - hg_start;
                
                if (wg_w < HOG_CELL_SIZE || wg_h < HOG_CELL_SIZE) continue;

                /* 提取窗口内梯度幅值和方向 */
                float* win_grad_mag = (float*)safe_malloc((size_t)(wg_w * wg_h) * sizeof(float));
                float* win_grad_ori = (float*)safe_malloc((size_t)(wg_w * wg_h) * sizeof(float));
                if (!win_grad_mag || !win_grad_ori) {
                    if (win_grad_mag) safe_free((void**)&win_grad_mag);
                    if (win_grad_ori) safe_free((void**)&win_grad_ori);
                    continue;
                }

                for (int wy = 0; wy < wg_h; wy++) {
                    int src_row = hg_start + wy;
                    int dst_row = wy;
                    for (int wx = 0; wx < wg_w; wx++) {
                        int src_idx = src_row * gw + (wg_start + wx);
                        int dst_idx = dst_row * wg_w + wx;
                        win_grad_mag[dst_idx] = grad_mag[src_idx];
                        win_grad_ori[dst_idx] = grad_ori[src_idx];
                    }
                }

                /* 计算HOG特征 */
                float hog_features[128];
                compute_hog_features(win_grad_mag, win_grad_ori, wg_w, wg_h, hog_features, 128);

                /* 边缘响应：使用梯度幅值均值 */
                float edge_response = 0;
                for (int i = 0; i < wg_w * wg_h; i++) {
                    edge_response += win_grad_mag[i];
                }
                edge_response /= (float)(wg_w * wg_h);

                safe_free((void**)&win_grad_mag);
                safe_free((void**)&win_grad_ori);

                if (edge_response > 0.01f) {
                    DetectedObject* obj = &out[count];
                    obj->x = (float)x;
                    obj->y = (float)y;
                    obj->width = (float)win_size;
                    obj->height = (float)win_size;
                    obj->confidence = edge_response * 8.0f;
                    if (obj->confidence > 1.0f) obj->confidence = 1.0f;

/* 分类管道升级 —— CfC深度学习优先，HOG模板匹配降级为回退 */
                    int best_category = -1;

                    if (or_obj->is_cfc_trained) {
                        /* 主路径: CfC深度学习管道 */
                        float cfc_probs[OR_MAX_CATEGORIES];
                        int cfc_best = object_cfc_forward(or_obj, hog_features, cfc_probs);
                        if (cfc_best >= 0 && cfc_best < or_obj->category_count) {
                            best_category = cfc_best;
                            obj->confidence = cfc_probs[cfc_best];
                            if (obj->confidence < edge_response * 8.0f)
                                obj->confidence = edge_response * 8.0f;
                            if (obj->confidence > 1.0f) obj->confidence = 1.0f;
                        }
                    }

/* 回退路径 —— HOG+NCC模板匹配
                     * 当CfC管道未训练或前向传播失败时启用 */
                    if (best_category < 0 && or_obj->is_trained) {
                        float best_ncc = -1.0f;
                        for (int c = 0; c < or_obj->category_count; c++) {
                            float ncc = normalized_cross_correlation(
                                hog_features, or_obj->category_templates[c], 128);
                            if (ncc > best_ncc) {
                                best_ncc = ncc;
                                best_category = c;
                            }
                        }
                        if (best_category >= 0) {
                            obj->confidence = (best_ncc + 1.0f) * 0.5f;
                            if (obj->confidence < edge_response * 8.0f)
                                obj->confidence = edge_response * 8.0f;
                            if (obj->confidence > 1.0f) obj->confidence = 1.0f;
                        }
                    }

/* 使用显式训练状态标志（is_trained 和 is_cfc_trained） */
                    obj->category_id = best_category;
                    if (best_category >= 0 && best_category < or_obj->category_count) {
                        snprintf(obj->category_name, sizeof(obj->category_name),
                                "%s", or_obj->category_names[best_category]);
                    } else {
                        snprintf(obj->category_name, sizeof(obj->category_name),
                                "未训练");
                    }
                    /* 存储HOG特征 */
                    memcpy(obj->features, hog_features, 128 * sizeof(float));
                    obj->feature_dim = 128;
                    count++;
                }
            }
        }
    }

    safe_free((void**)&grad_mag);
    safe_free((void**)&grad_ori);
    return count;
}

int or_classify_object(ObjectRecognizer* or_obj, const float* features, int dim, int* category_id, float* confidence) {
    if (!or_obj || !features || !category_id || !confidence) return -1;

    float best_sim = -1.0f;
    int best_cat = 0;
    int cmp_dim = dim < 128 ? dim : 128;

    for (int c = 0; c < or_obj->category_count; c++) {
        float dot = 0.0f, mag_f = 0.0f, mag_t = 0.0f;
        for (int i = 0; i < cmp_dim; i++) {
            dot += features[i] * or_obj->category_templates[c][i];
            mag_f += features[i] * features[i];
            mag_t += or_obj->category_templates[c][i] * or_obj->category_templates[c][i];
        }
        float sim = (mag_f > 1e-10f && mag_t > 1e-10f) ? dot / (sqrtf(mag_f) * sqrtf(mag_t)) : 0.0f;
        if (sim > best_sim) { best_sim = sim; best_cat = c; }
    }

    *category_id = best_cat;
    *confidence = (best_sim + 1.0f) / 2.0f;
    return 0;
}

int or_recognize_attributes(ObjectRecognizer* or_obj, const float* image, int w, int h, int ch,
    const DetectedObject* obj, ObjectAttributes* attrs) {
    if (!or_obj || !image || !obj || !attrs) return -1;
    memset(attrs, 0, sizeof(ObjectAttributes));
    int x = (int)obj->x, y = (int)obj->y, bw = (int)obj->width, bh = (int)obj->height;
    if (x < 0) x = 0; if (y < 0) y = 0;
    if (x + bw > w) bw = w - x; if (y + bh > h) bh = h - y;
    if (bw <= 0 || bh <= 0) return -1;

    float sum_h = 0.0f, sum_s = 0.0f, sum_b = 0.0f;
    int n = bw * bh;
    for (int dy = 0; dy < bh; dy++) {
        for (int dx = 0; dx < bw; dx++) {
            int idx = ((y + dy) * w + (x + dx)) * ch;
            float r = image[idx], g = (ch > 1) ? image[idx + 1] : r, b = (ch > 2) ? image[idx + 2] : r;
            float mx = r > g ? r : g; mx = mx > b ? mx : b;
            float mn = r < g ? r : g; mn = mn < b ? mn : b;
            float d = mx - mn;
            float h_val = 0.0f;
            if (d > 1e-6f) {
                if (mx == r) h_val = 60.0f * fmodf((g - b) / d + 6.0f, 6.0f);
                else if (mx == g) h_val = 60.0f * ((b - r) / d + 2.0f);
                else h_val = 60.0f * ((r - g) / d + 4.0f);
            }
            sum_h += h_val;
            sum_s += (mx > 1e-6f) ? d / mx : 0.0f;
            sum_b += mx;
        }
    }
    attrs->hue_mean = sum_h / (float)n;
    attrs->saturation_mean = sum_s / (float)n;
    attrs->brightness_mean = sum_b / (float)n;
    /* BUG-003修复: 使用LBP风格的局部纹理分析替代坐标比率假数据
     * LBP: 中心像素与8邻域比较，统计纹理模式复杂度 */
    float texture_score = 0.0f;
    int tex_samples = 0;
    for (int dy = 2; dy < bh - 2; dy += 4) {
        for (int dx = 2; dx < bw - 2; dx += 4) {
            int idx_c = ((y + dy) * w + (x + dx)) * ch;
            if (idx_c + ch >= w * h * ch) continue;
            float center = image[idx_c];
            int lbp_code = 0;
            int offsets[8][2] = {{-1,-1},{0,-1},{1,-1},{1,0},{1,1},{0,1},{-1,1},{-1,0}};
            for (int nbr = 0; nbr < 8; nbr++) {
                int nx = x + dx + offsets[nbr][0];
                int ny = y + dy + offsets[nbr][1];
                if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                    int nidx = (ny * w + nx) * ch;
                    if (image[nidx] >= center) lbp_code |= (1 << nbr);
                }
            }
            /* 统计LBP码中0-1跳变次数衡量纹理复杂度 */
            int transitions = 0;
            int prev_bit = (lbp_code >> 7) & 1;
            for (int b = 0; b < 8; b++) {
                int cur_bit = (lbp_code >> b) & 1;
                if (cur_bit != prev_bit) transitions++;
                prev_bit = cur_bit;
            }
            texture_score += (float)transitions / 9.0f;
            tex_samples++;
        }
    }
    attrs->texture_roughness = tex_samples > 0 ? (texture_score / (float)tex_samples) : 0.3f;
    snprintf(attrs->material, sizeof(attrs->material), "%s",
        attrs->brightness_mean > 0.6f ? "金属/光泽" : attrs->texture_roughness > 0.5f ? "粗糙" : "光滑");
    snprintf(attrs->size_category, sizeof(attrs->size_category), "%s",
        (bw * bh) > (w * h / 16) ? "大型" : (bw * bh) > (w * h / 64) ? "中型" : "小型");
    return 0;
}

int or_detect_color(ObjectRecognizer* or_obj, const float* image, int w, int h, int ch,
    const DetectedObject* obj, float* color_rgb) {
    if (!or_obj || !image || !obj || !color_rgb) return -1;
    int x = (int)obj->x, y = (int)obj->y, bw = (int)obj->width, bh = (int)obj->height;
    x = x < 0 ? 0 : x; y = y < 0 ? 0 : y;
    if (x + bw > w) bw = w - x; if (y + bh > h) bh = h - y;
    if (bw <= 0 || bh <= 0) return -1;

    float sr = 0, sg = 0, sb = 0;
    int n = bw * bh;
    for (int dy = 0; dy < bh; dy++) {
        for (int dx = 0; dx < bw; dx++) {
            int idx = ((y + dy) * w + (x + dx)) * ch;
            sr += image[idx];
            sg += (ch > 1) ? image[idx + 1] : image[idx];
            sb += (ch > 2) ? image[idx + 2] : image[idx];
        }
    }
    color_rgb[0] = sr / (float)n; color_rgb[1] = sg / (float)n; color_rgb[2] = sb / (float)n;
    return 0;
}

/* 添加RGB→颜色名称映射，补全颜色识别功能 */
static const struct {
    const char* name;
    float r, g, b;
} g_color_reference[] = {
    {"红色",   0.85f, 0.08f, 0.08f},
    {"蓝色",   0.08f, 0.08f, 0.85f},
    {"绿色",   0.08f, 0.80f, 0.12f},
    {"黄色",   0.88f, 0.88f, 0.08f},
    {"白色",   0.90f, 0.90f, 0.90f},
    {"黑色",   0.05f, 0.05f, 0.05f},
    {"紫色",   0.60f, 0.08f, 0.80f},
    {"橙色",   0.90f, 0.50f, 0.05f},
    {"灰色",   0.50f, 0.50f, 0.50f},
    {"粉色",   0.90f, 0.50f, 0.65f},
    {"棕色",   0.55f, 0.27f, 0.07f},
    {"青色",   0.05f, 0.80f, 0.80f},
    {NULL, 0, 0, 0}
};

int or_get_color_name(const float* color_rgb, char* name_buf, size_t buf_size, float* confidence) {
    if (!color_rgb || !name_buf || buf_size < 4) return -1;
    float min_dist = INFINITY;
    int best_idx = 0;
    for (int i = 0; g_color_reference[i].name; i++) {
        float dr = color_rgb[0] - g_color_reference[i].r;
        float dg = color_rgb[1] - g_color_reference[i].g;
        float db = color_rgb[2] - g_color_reference[i].b;
        float dist = dr * dr + dg * dg + db * db;
        if (dist < min_dist) { min_dist = dist; best_idx = i; }
    }
    snprintf(name_buf, buf_size, "%s", g_color_reference[best_idx].name);
    if (confidence) {
        float max_dist = 3.0f;
        *confidence = 1.0f - (sqrtf(min_dist) / sqrtf(max_dist));
        if (*confidence < 0.0f) *confidence = 0.0f;
        if (*confidence > 1.0f) *confidence = 1.0f;
    }
    return 0;
}

int or_estimate_size(ObjectRecognizer* or_obj, const DetectedObject* obj, float* width_m, float* height_m) {
    if (!or_obj || !obj || !width_m || !height_m) return -1;

    /* 估算焦距: 60度水平视场角 → f ≈ image_width / 1.15 */
    float img_width = obj->width * 5.0f;
    if (img_width < 320.0f) img_width = 640.0f;
    float focal_pixels = img_width / 1.15f;

    float est_distance = 0.0f;
    int depth_available = 0;

    /* 优先从深度图获取物体中心位置的深度值 */
    if (or_obj->has_depth_map && or_obj->depth_map) {
        int cx = (int)(obj->x + obj->width * 0.5f);
        int cy = (int)(obj->y + obj->height * 0.5f);
        if (cx >= 0 && cx < or_obj->depth_map_w && cy >= 0 && cy < or_obj->depth_map_h) {
            int d_idx = cy * or_obj->depth_map_w + cx;
            est_distance = or_obj->depth_map[d_idx];
            if (est_distance > 0.0f) {
                depth_available = 1;
            }
        }
    }

    /* 如果中心点深度无效，尝试物体区域深度均值作为备选 */
    if (!depth_available && or_obj->has_depth_map && or_obj->depth_map) {
        int x0 = (int)obj->x, y0 = (int)obj->y;
        int x1 = (int)(obj->x + obj->width), y1 = (int)(obj->y + obj->height);
        if (x0 < 0) x0 = 0; if (y0 < 0) y0 = 0;
        if (x1 > or_obj->depth_map_w) x1 = or_obj->depth_map_w;
        if (y1 > or_obj->depth_map_h) y1 = or_obj->depth_map_h;
        float depth_sum = 0.0f;
        int depth_count = 0;
        for (int dy = y0; dy < y1; dy += 2) {
            for (int dx = x0; dx < x1; dx += 2) {
                float d = or_obj->depth_map[dy * or_obj->depth_map_w + dx];
                if (d > 0.0f) { depth_sum += d; depth_count++; }
            }
        }
        if (depth_count > 0) {
            est_distance = depth_sum / (float)depth_count;
            depth_available = 1;
        }
    }

    /* 无法获取真实深度，拒绝返回虚假数据 */
    if (!depth_available) {
        return -1;
    }

    /* pixel_to_meter = est_distance_m / focal_pixels */
    *width_m = obj->width * est_distance / focal_pixels;
    *height_m = obj->height * est_distance / focal_pixels;

    /* 合理性限制 */
    if (*width_m <= 0.0f) *width_m = 0.01f;
    if (*height_m <= 0.0f) *height_m = 0.01f;
    return 0;
}

int or_classify_scene(ObjectRecognizer* or_obj, const float* image, int w, int h, int ch,
    SceneType* type, char* name, size_t name_len) {
    if (!or_obj || !image || !type || !name) return -1;
    /* BUG-002修复: 使用多维特征（亮度+纹理+边缘+色温）进行场景分类
     * 替代原来仅靠平均亮度的降级处理
 *修复 P2-008: 待ObjectRecognizer集成LNN后，使用LNN增强场景分类
     * 替代当前纯启发式阈值评分（魔法数字），改用统一LNN推理 */
    float avg_brightness = 0.0f;
    float avg_variance = 0.0f;
    float edge_density = 0.0f;
    float green_ratio = 0.0f;
    int n = w * h;
    /* 1. 亮度均值和方差 */
    for (int i = 0; i < n; i++) avg_brightness += image[i * ch];
    avg_brightness /= (float)n;
    for (int i = 0; i < n; i++) {
        float d = image[i * ch] - avg_brightness;
        avg_variance += d * d;
    }
    avg_variance /= (float)n;
    /* 2. 边缘密度（Sobel梯度幅值统计） */
    float edge_threshold = 0.05f;
    int edge_count = 0;
    float total_green = 0.0f;
    for (int y = 1; y < h - 1; y += 2) {
        for (int x = 1; x < w - 1; x += 2) {
            int idx = (y * w + x) * ch;
            float gx = image[idx + w * ch] - image[idx - w * ch];  /* 垂直梯度 */
            float gy = image[idx + ch] - image[idx - ch];           /* 水平梯度 */
            float mag = sqrtf(gx * gx + gy * gy);
            if (mag > edge_threshold) edge_count++;
            if (ch > 1) total_green += image[idx + 1];
        }
    }
    int samples = ((h - 2) / 2) * ((w - 2) / 2);
    edge_density = samples > 0 ? (float)edge_count / (float)samples : 0.0f;
    /* 3. 绿色通道占比（自然场景检测） */
    green_ratio = (n > 0) ? (total_green / (float)samples) : 0.0f;
    /* 4. 综合判定 */
    float indoor_score = (avg_brightness < 0.3f ? 2.0f : 0.0f)
                       + (edge_density < 0.1f ? 1.0f : 0.0f)
                       + (green_ratio < 0.2f ? 1.0f : 0.0f);
    float nature_score = (green_ratio > 0.35f ? 3.0f : 0.0f)
                       + (avg_brightness > 0.4f ? 1.0f : 0.0f)
                       + (edge_density > 0.2f ? 1.0f : 0.0f);
    float industrial_score = (avg_variance < 0.02f ? 2.0f : 0.0f)
                           + (edge_density > 0.15f ? 1.0f : 0.0f)
                           + (avg_brightness > 0.3f && avg_brightness < 0.6f ? 1.0f : 0.0f);
    float outdoor_score = (avg_brightness > 0.5f ? 2.0f : 0.0f)
                        + (edge_density > 0.1f ? 1.0f : 0.0f);
    if (nature_score >= indoor_score && nature_score >= industrial_score && nature_score >= outdoor_score) {
        *type = SCENE_NATURE; snprintf(name, name_len, "自然场景");
    } else if (industrial_score >= indoor_score && industrial_score >= outdoor_score) {
        *type = SCENE_INDUSTRIAL; snprintf(name, name_len, "工业场景");
    } else if (indoor_score >= outdoor_score) {
        *type = SCENE_INDOOR; snprintf(name, name_len, "室内场景");
    } else {
        *type = SCENE_OUTDOOR; snprintf(name, name_len, "室外场景");
    }
    or_obj->last_scene = *type;
    return 0;
}

int or_detect_relations(ObjectRecognizer* or_obj, const DetectedObject* objects, int count, SceneContext* ctx) {
    if (!or_obj || !objects || !ctx) return -1;
    int n_rel = count < OR_MAX_OBJECTS ? count : OR_MAX_OBJECTS;
    ctx->object_count = n_rel;
    for (int i = 0; i < n_rel; i++) {
        memcpy(&ctx->objects[i], &objects[i], sizeof(DetectedObject));
    }
    for (int i = 0; i < n_rel; i++) {
        for (int j = i + 1; j < n_rel; j++) {
            float dx = objects[i].x - objects[j].x;
            float dy = objects[i].y - objects[j].y;
            float overlap_x = (objects[i].width + objects[j].width) / 2.0f - fabsf(dx);
            float overlap_y = (objects[i].height + objects[j].height) / 2.0f - fabsf(dy);
            if (overlap_x > 0 && overlap_y > 0) {
                if (i < OR_MAX_OBJECTS && ctx->relation_counts && i < n_rel) ctx->relation_counts[i]++;
            }
        }
    }
    return 0;
}

int or_detect_changes(ObjectRecognizer* or_obj, const float* prev, const float* curr, int w, int h, int ch,
    float* change_map) {
    if (!or_obj || !prev || !curr || !change_map) return -1;
    int n = w * h;
    float max_change = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = 0.0f;
        for (int c = 0; c < ch; c++) {
            float d = curr[i * ch + c] - prev[i * ch + c];
            diff += d * d;
        }
        change_map[i] = sqrtf(diff / (float)ch);
        if (change_map[i] > max_change) max_change = change_map[i];
    }
    if (max_change > 1e-10f) {
        for (int i = 0; i < n; i++) change_map[i] /= max_change;
    }
    return 0;
}

/* ====== 深度信息集成（H-010修复：接入真实深度替代硬编码物距） ====== */

/*
 * or_set_depth_estimator: 注册深度估计器
 * 调用此函数后，当深度图不可用时，可通过深度估计器实时生成
 */
int or_set_depth_estimator(ObjectRecognizer* or_obj, DepthEstimator* estimator) {
    if (!or_obj) return -1;
    or_obj->depth_estimator = estimator;
    return 0;
}

/*
 * or_set_depth_map: 传入已计算的深度图用于物体物理尺寸估算
 * depth_map: 行主序深度图，单位米，0值表示无效像素
 * 调用方需保证 depth_map 在 or_estimate_size 调用期间有效
 */
int or_set_depth_map(ObjectRecognizer* or_obj, const float* depth_map, int w, int h) {
    if (!or_obj || !depth_map || w <= 0 || h <= 0) return -1;
    or_obj->depth_map = depth_map;
    or_obj->depth_map_w = w;
    or_obj->depth_map_h = h;
    or_obj->has_depth_map = 1;
    return 0;
}

/*
 * or_clear_depth_map: 清除当前深度图引用
 * 调用后 or_estimate_size 将返回错误直到新深度图被设置
 */
int or_clear_depth_map(ObjectRecognizer* or_obj) {
    if (!or_obj) return -1;
    or_obj->depth_map = NULL;
    or_obj->depth_map_w = 0;
    or_obj->depth_map_h = 0;
    or_obj->has_depth_map = 0;
    return 0;
}

/* ====== 分类器训练/保存/加载（F-001修复：完整实现） ====== */

/*
 * or_train_classifier: 使用基于原型均值的分类器训练
 * 对每个类别，计算所有属于该类别的样本的特征均值作为类别模板
 * 采用Warmuth缩放增强小样本类别鲁棒性
 */
int or_train_classifier(ObjectRecognizer* or_obj, const float* features,
                        const int* labels, int samples, int dim, int categories) {
    if (!or_obj || !features || !labels) return -1;
    if (samples <= 0 || dim <= 0 || categories <= 0) return -1;
    if (categories > OR_MAX_CATEGORIES) categories = OR_MAX_CATEGORIES;
    int feat_dim = dim < 128 ? dim : 128;

    /* 统计每个类别的样本数 */
    int category_samples[OR_MAX_CATEGORIES];
    memset(category_samples, 0, sizeof(category_samples));

    /* 清零所有类别模板 */
    for (int c = 0; c < OR_MAX_CATEGORIES; c++) {
        memset(or_obj->category_templates[c], 0, 128 * sizeof(float));
    }

    /* 累加每个类别的特征向量 */
    for (int s = 0; s < samples; s++) {
        int label = labels[s];
        if (label < 0 || label >= categories) continue;
        if (label >= OR_MAX_CATEGORIES) continue;
        int offset = s * dim;
        for (int i = 0; i < feat_dim; i++) {
            or_obj->category_templates[label][i] += features[offset + i];
        }
        category_samples[label]++;
    }

    /* 计算每个类别的均值模板（Warmuth缩放） */
    for (int c = 0; c < categories && c < OR_MAX_CATEGORIES; c++) {
        int n = category_samples[c];
        if (n < 1) {
            /* 无样本的类别：使用所有类别的全局均值 */
            float global_mean_sum[128] = {0};
            int total_valid = 0;
            for (int cc = 0; cc < categories && cc < OR_MAX_CATEGORIES; cc++) {
                if (category_samples[cc] > 0) {
                    for (int i = 0; i < feat_dim; i++)
                        global_mean_sum[i] += or_obj->category_templates[cc][i] / (float)category_samples[cc];
                    total_valid++;
                }
            }
            if (total_valid > 0) {
                for (int i = 0; i < feat_dim; i++)
                    or_obj->category_templates[c][i] = global_mean_sum[i] / (float)total_valid;
            }
            continue;
        }
        /* Warmuth缩放：n/(n+1) 衰减，增强小样本鲁棒性 */
        float scale = (float)n / (float)(n + 1);
        for (int i = 0; i < feat_dim; i++) {
            or_obj->category_templates[c][i] = (or_obj->category_templates[c][i] / (float)n) * scale;
        }
    }

    /* 更新类别总数 */
    if (categories > or_obj->category_count) {
        or_obj->category_count = categories;
    }

/* 训练完成后设置显式训练标志 */
    or_obj->is_trained = 1;

/* 在HOG模板训练完成后同步训练CfC深度学习管道
     * 使用相同训练数据进行SGD优化，epochs=10, lr=0.01 */
    object_cfc_train(or_obj, features, labels, samples, dim, categories, 10, 0.01f);

    return 0;
}

/*
 *: or_train_cfc —— 独立CfC深度学习训练接口
 * 用于仅训练CfC管道，不影响HOG模板
 */
int or_train_cfc(ObjectRecognizer* or_obj, const float* features,
                 const int* labels, int samples, int dim, int categories,
                 int epochs, float lr) {
    if (!or_obj || !features || !labels) return -1;
    return object_cfc_train(or_obj, features, labels, samples, dim, categories, epochs, lr);
}

/*
 * or_save_model: 将分类器模型保存到文件
 * 二进制格式: [魔数4字节][类别数4字节][每个类别的名称64字节+模板128*float]
 * 魔数: "SLO2" = 0x324F4C53 (Self-LNN Object recognition v2)
 */
int or_save_model(const ObjectRecognizer* or_obj, const char* filepath) {
    if (!or_obj || !filepath) return -1;
    if (or_obj->category_count <= 0) return -1;

    FILE* fp = fopen(filepath, "wb");
    if (!fp) return -1;

    /* 写入魔数 */
    const char magic[4] = {'S', 'L', 'O', '2'};
    if (fwrite(magic, 1, 4, fp) != 4) { fclose(fp); return -1; }

    /* 写入类别数量 */
    int count = or_obj->category_count;
    if (fwrite(&count, sizeof(int), 1, fp) != 1) { fclose(fp); return -1; }

    /* 写入每个类别的名称和模板 */
    for (int c = 0; c < count && c < OR_MAX_CATEGORIES; c++) {
        /* 类别名称（固定64字节） */
        if (fwrite(or_obj->category_names[c], 1, 64, fp) != 64) { fclose(fp); return -1; }
        /* 类别模板特征（128个float） */
        if (fwrite(or_obj->category_templates[c], sizeof(float), 128, fp) != 128) { fclose(fp); return -1; }
    }

    /* 写入场景信息 */
    int scene = (int)or_obj->last_scene;
    if (fwrite(&scene, sizeof(int), 1, fp) != 1) { fclose(fp); return -1; }

/* 写入CfC深度学习权重段标识和权重数据
     * 段标识: "CFC1" 4字节，用于v3格式识别
     * CfC权重按维度写入: W_gx, W_ax, W_gh, W_ah, b_g, b_a, W_cls, b_cls, 元参数 */
    const char cfc_magic[4] = {'C', 'F', 'C', '1'};
    if (fwrite(cfc_magic, 1, 4, fp) != 4) { fclose(fp); return -1; }
    fwrite(&or_obj->is_cfc_trained, sizeof(int), 1, fp);
    fwrite(&or_obj->cfc_tau, sizeof(float), 1, fp);
    fwrite(&or_obj->cfc_dt, sizeof(float), 1, fp);
    fwrite(or_obj->cfc_W_gx, sizeof(float), CFC_OR_HIDDEN_DIM * CFC_OR_INPUT_DIM, fp);
    fwrite(or_obj->cfc_W_ax, sizeof(float), CFC_OR_HIDDEN_DIM * CFC_OR_INPUT_DIM, fp);
    fwrite(or_obj->cfc_W_gh, sizeof(float), CFC_OR_HIDDEN_DIM * CFC_OR_HIDDEN_DIM, fp);
    fwrite(or_obj->cfc_W_ah, sizeof(float), CFC_OR_HIDDEN_DIM * CFC_OR_HIDDEN_DIM, fp);
    fwrite(or_obj->cfc_b_g, sizeof(float), CFC_OR_HIDDEN_DIM, fp);
    fwrite(or_obj->cfc_b_a, sizeof(float), CFC_OR_HIDDEN_DIM, fp);
    fwrite(or_obj->cfc_W_cls, sizeof(float), OR_MAX_CATEGORIES * CFC_OR_HIDDEN_DIM, fp);
    fwrite(or_obj->cfc_b_cls, sizeof(float), OR_MAX_CATEGORIES, fp);

    fclose(fp);
    return 0;
}

/*
 * or_load_model: 从文件加载分类器模型
 */
int or_load_model(ObjectRecognizer* or_obj, const char* filepath) {
    if (!or_obj || !filepath) return -1;

    FILE* fp = fopen(filepath, "rb");
    if (!fp) return -1;

    /* 读取并验证魔数 */
    char magic[4];
    if (fread(magic, 1, 4, fp) != 4) { fclose(fp); return -1; }
    if (magic[0] != 'S' || magic[1] != 'L' || magic[2] != 'O' || magic[3] != '2') {
        fclose(fp);
        return -1;
    }

    /* 读取类别数量 */
    int count = 0;
    if (fread(&count, sizeof(int), 1, fp) != 1) { fclose(fp); return -1; }
    if (count <= 0 || count > OR_MAX_CATEGORIES) { fclose(fp); return -1; }

    or_obj->category_count = count;

    /* 读取每个类别的名称和模板 */
    for (int c = 0; c < count && c < OR_MAX_CATEGORIES; c++) {
        if (fread(or_obj->category_names[c], 1, 64, fp) != 64) { fclose(fp); return -1; }
        or_obj->category_names[c][63] = '\0'; /* 确保字符串终止 */
        if (fread(or_obj->category_templates[c], sizeof(float), 128, fp) != 128) { fclose(fp); return -1; }
    }

    /* 读取场景信息 */
    int scene = SCENE_UNKNOWN;
    if (fread(&scene, sizeof(int), 1, fp) == 1) {
        or_obj->last_scene = (SceneType)scene;
    }

    or_obj->initialized = 1;
    or_obj->is_trained = 1; /* 模型加载成功后标记为已训练 */

/* 尝试读取CfC深度学习权重段（向后兼容SLO2格式）
     * 检查是否存在"CFC1"段标识，若存在则加载CfC权重
     * 若文件到此结束（v2格式），则CfC保持未训练状态 */
    char cfc_check[4] = {0, 0, 0, 0};
    long cfc_pos = ftell(fp);
    if (fread(cfc_check, 1, 4, fp) == 4 &&
        cfc_check[0] == 'C' && cfc_check[1] == 'F' &&
        cfc_check[2] == 'C' && cfc_check[3] == '1') {
        fread(&or_obj->is_cfc_trained, sizeof(int), 1, fp);
        fread(&or_obj->cfc_tau, sizeof(float), 1, fp);
        fread(&or_obj->cfc_dt, sizeof(float), 1, fp);
        fread(or_obj->cfc_W_gx, sizeof(float), CFC_OR_HIDDEN_DIM * CFC_OR_INPUT_DIM, fp);
        fread(or_obj->cfc_W_ax, sizeof(float), CFC_OR_HIDDEN_DIM * CFC_OR_INPUT_DIM, fp);
        fread(or_obj->cfc_W_gh, sizeof(float), CFC_OR_HIDDEN_DIM * CFC_OR_HIDDEN_DIM, fp);
        fread(or_obj->cfc_W_ah, sizeof(float), CFC_OR_HIDDEN_DIM * CFC_OR_HIDDEN_DIM, fp);
        fread(or_obj->cfc_b_g, sizeof(float), CFC_OR_HIDDEN_DIM, fp);
        fread(or_obj->cfc_b_a, sizeof(float), CFC_OR_HIDDEN_DIM, fp);
        fread(or_obj->cfc_W_cls, sizeof(float), OR_MAX_CATEGORIES * CFC_OR_HIDDEN_DIM, fp);
        fread(or_obj->cfc_b_cls, sizeof(float), OR_MAX_CATEGORIES, fp);
    } else {
        /* v2格式: 无CfC段，CfC保持未训练（使用HOG模板匹配作为回退） */
        fseek(fp, cfc_pos, SEEK_SET);
    }

    fclose(fp);
    return 0;
}
