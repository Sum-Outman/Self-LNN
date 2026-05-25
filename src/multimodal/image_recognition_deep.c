#include "selflnn/multimodal/image_recognition_deep.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/secure_random.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <float.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define PCFC_MAX_CH 8
#define PCFC_HD 32

static float _sig(float x) { return 1.0f / (1.0f + expf(-x)); }

static float _tanh_f(float x) {
    float e2x = expf(2.0f * x);
    return (e2x - 1.0f) / (e2x + 1.0f);
}

/* K-011修复：使用加密安全随机数替代rand() */
/* Xavier初始化：适用于tanh/sigmoid激活函数层
 * 公式：std = sqrt(2.0 / (fan_in + fan_out)) */
static void _xavier_init(float* w, int fan_in, int fan_out) {
    float scale = sqrtf(2.0f / (float)(fan_in + fan_out));
    for (int i = 0; i < fan_in * fan_out; i++) {
        float u1 = secure_random_float();
        float u2 = secure_random_float();
        if (u1 < 1e-7f) u1 = 1e-7f;
        w[i] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2) * scale;
    }
}

/* He/Kaiming初始化：适用于ReLU及变体激活函数层
 * 公式：std = sqrt(2.0 / fan_in)
 * 对ReLU激活的层使用此初始化可以避免梯度消失，保持前向传播方差 */
static void _he_init(float* w, int fan_in, int fan_out) {
    float scale = sqrtf(2.0f / (float)fan_in);
    for (int i = 0; i < fan_in * fan_out; i++) {
        float u1 = secure_random_float();
        float u2 = secure_random_float();
        if (u1 < 1e-7f) u1 = 1e-7f;
        w[i] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2) * scale;
    }
}

static void _mat_vec_mul(const float* mat, const float* vec, float* out, int r, int c) {
    for (int i = 0; i < r; i++) {
        out[i] = 0.0f;
        for (int j = 0; j < c; j++) out[i] += mat[i * c + j] * vec[j];
    }
}

static void _vec_add(float* a, const float* b, int n) { for (int i = 0; i < n; i++) a[i] += b[i]; }
static void _vec_hadamard(float* a, const float* b, int n) { for (int i = 0; i < n; i++) a[i] *= b[i]; }
static void _vec_sigmoid(float* a, int n) { for (int i = 0; i < n; i++) a[i] = _sig(a[i]); }
static void _vec_tanh(float* a, int n) { for (int i = 0; i < n; i++) a[i] = _tanh_f(a[i]); }
static void _vec_scale(float* a, float s, int n) { for (int i = 0; i < n; i++) a[i] *= s; }
static void _vec_copy(float* d, const float* s, int n) { memcpy(d, s, n * sizeof(float)); }

static float _vec_dot(const float* a, const float* b, int n) {
    float s = 0.0f; for (int i = 0; i < n; i++) s += a[i] * b[i]; return s;
}

static float _vec_norm(const float* a, int n) { return sqrtf(_vec_dot(a, a, n)); }

static void _vec_normalize(float* a, int n) {
    float norm = _vec_norm(a, n);
    if (norm > 1e-10f) _vec_scale(a, 1.0f / norm, n);
}

static float _l2_dist(const float* a, const float* b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) { float d = a[i] - b[i]; s += d * d; }
    return sqrtf(s);
}

static float _cos_sim(const float* a, const float* b, int n) {
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (int i = 0; i < n; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    float denom = sqrtf(na * nb);
    return (denom > 1e-10f) ? dot / denom : 0.0f;
}

static void _softmax(float* logits, int n) {
    float mv = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > mv) mv = logits[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { logits[i] = expf(logits[i] - mv); sum += logits[i]; }
    if (sum > 1e-10f) for (int i = 0; i < n; i++) logits[i] /= sum;
}

static void _cfc_ode_step(const float* in, int in_dim,
                           const float* W_gx, const float* W_ax,
                           const float* W_gh, const float* W_ah,
                           const float* b_g, const float* b_a,
                           float* h, int h_dim, float tau, float dt) {
    float gate[256], act[256], t1[256], t2[256];
    memset(gate, 0, h_dim * sizeof(float));
    memset(act, 0, h_dim * sizeof(float));
    _mat_vec_mul(W_gx, in, t1, h_dim, in_dim);
    _mat_vec_mul(W_gh, h, t2, h_dim, h_dim);
    _vec_add(t1, t2, h_dim); _vec_add(t1, b_g, h_dim); _vec_sigmoid(t1, h_dim);
    _vec_copy(gate, t1, h_dim);
    _mat_vec_mul(W_ax, in, t1, h_dim, in_dim);
    _mat_vec_mul(W_ah, h, t2, h_dim, h_dim);
    _vec_add(t1, t2, h_dim); _vec_add(t1, b_a, h_dim); _vec_tanh(t1, h_dim);
    _vec_copy(act, t1, h_dim);
    _vec_hadamard(gate, act, h_dim);
    float decay = expf(-dt / tau);
    for (int i = 0; i < h_dim; i++) h[i] = h[i] * decay + (1.0f - decay) * gate[i];
}

static void _patch_cfc_encode(const float* img, int w, int h, int ch,
                               int px, int py, int pw, int ph,
                               const float* W_gx, const float* W_ax,
                               const float* W_gh, const float* W_ah,
                               const float* b_g, const float* b_a,
                               float* hidden, int hd, float tau, float dt) {
    int in_dim = (ch < PCFC_MAX_CH) ? ch : PCFC_MAX_CH;
    float inp[PCFC_MAX_CH];
    for (int dy = 0; dy < ph; dy++) {
        for (int dx = 0; dx < pw; dx++) {
            int ix = px + dx, iy = py + dy;
            if (ix >= 0 && ix < w && iy >= 0 && iy < h) {
                for (int c = 0; c < in_dim; c++)
                    inp[c] = img[(iy * w + ix) * ch + c];
            } else {
                memset(inp, 0, in_dim * sizeof(float));
            }
            _cfc_ode_step(inp, in_dim, W_gx, W_ax, W_gh, W_ah,
                          b_g, b_a, hidden, hd, tau, dt);
        }
    }
}

/* ======================================================================== */
/*  细粒度分类                                                              */
/* ======================================================================== */

struct IRDFineClassifier {
    IRDFineConfig cfg;
    int training_completed;
    float W_gx_p[256 * 32], W_ax_p[256 * 32], W_gh_p[256 * 256], W_ah_p[256 * 256];
    float b_g_p[256], b_a_p[256];
    float W_gx_part[256 * 256], W_ax_part[256 * 256];
    float W_gh_part[256 * 256], W_ah_part[256 * 256];
    float b_g_part[256], b_a_part[256];
    float W_fine[256 * 256], b_fine[256];
    float W_coarse[64 * 256], b_coarse[64];
    float bilin_W[256 * 256], bilin_proj[256 * 2];
    float pW_gx[PCFC_HD * PCFC_MAX_CH], pW_ax[PCFC_HD * PCFC_MAX_CH];
    float pW_gh[PCFC_HD * PCFC_HD], pW_ah[PCFC_HD * PCFC_HD];
    float pb_g[PCFC_HD], pb_a[PCFC_HD];
};

IRDFineConfig ird_fine_get_default_config(void) {
    IRDFineConfig c; memset(&c, 0, sizeof(c));
    c.input_dim = 32; c.num_fine_categories = 64; c.num_coarse_categories = 10;
    c.num_parts = 8; c.patch_size = 16; c.feature_dim = 256;
    c.cfc_time_constant = 0.1f; c.cfc_delta_t = 0.05f;
    c.enable_bilinear_pooling = 1; c.bilinear_normalization = 1.0f;
    c.enable_part_refinement = 1; c.max_refinement_iterations = 3; c.discriminative_threshold = 0.3f;
    return c;
}

IRDFineClassifier* ird_fine_create(const IRDFineConfig* config) {
    if (!config) return NULL;
    IRDFineClassifier* c = (IRDFineClassifier*)safe_calloc(1, sizeof(IRDFineClassifier));
    if (!c) return NULL;
    memcpy(&c->cfg, config, sizeof(IRDFineConfig));
    int hd = config->feature_dim, id = config->input_dim;
    _xavier_init(c->W_gx_p, id, hd); _xavier_init(c->W_ax_p, id, hd);
    _xavier_init(c->W_gh_p, hd, hd); _xavier_init(c->W_ah_p, hd, hd);
    memset(c->b_g_p, 0, hd * sizeof(float)); memset(c->b_a_p, 0, hd * sizeof(float));
    _xavier_init(c->W_gx_part, hd, hd); _xavier_init(c->W_ax_part, hd, hd);
    _xavier_init(c->W_gh_part, hd, hd); _xavier_init(c->W_ah_part, hd, hd);
    memset(c->b_g_part, 0, hd * sizeof(float)); memset(c->b_a_part, 0, hd * sizeof(float));
    _xavier_init(c->W_fine, hd, config->num_fine_categories);
    memset(c->b_fine, 0, config->num_fine_categories * sizeof(float));
    _xavier_init(c->W_coarse, hd, config->num_coarse_categories);
    memset(c->b_coarse, 0, config->num_coarse_categories * sizeof(float));
    _xavier_init(c->bilin_W, hd, hd); _xavier_init(c->bilin_proj, 2, hd);
    _xavier_init(c->pW_gx, PCFC_MAX_CH, PCFC_HD);
    _xavier_init(c->pW_ax, PCFC_MAX_CH, PCFC_HD);
    _xavier_init(c->pW_gh, PCFC_HD, PCFC_HD);
    _xavier_init(c->pW_ah, PCFC_HD, PCFC_HD);
    memset(c->pb_g, 0, PCFC_HD * sizeof(float));
    memset(c->pb_a, 0, PCFC_HD * sizeof(float));
    c->training_completed = 0;
    return c;
}

int ird_fine_mark_trained(IRDFineClassifier* c) {
    if (!c) return -1;
    c->training_completed = 1;
    return 0;
}

int ird_fine_is_trained(const IRDFineClassifier* c) {
    return c ? c->training_completed : 0;
}

void ird_fine_free(IRDFineClassifier* c) { safe_free((void**)&c); }

static void _extract_patches_cfc(IRDFineClassifier* clf, const float* img,
                                  int w, int h, int ch, float* pf, int* np) {
    int ps = clf->cfg.patch_size, st = ps / 2;
    int cols = (w - ps) / st + 1, rows = (h - ps) / st + 1;
    if (cols < 1) cols = 1; if (rows < 1) rows = 1;
    *np = cols * rows; if (*np > 256) *np = 256;
    int hd = clf->cfg.feature_dim;
    float hidden[256], patch_h[PCFC_HD];
    memset(hidden, 0, hd * sizeof(float));
    float tau = clf->cfg.cfc_time_constant, dt = clf->cfg.cfc_delta_t;
    int pi = 0;
    for (int ry = 0; ry < rows && pi < *np; ry++) {
        for (int rx = 0; rx < cols && pi < *np; rx++) {
            memset(patch_h, 0, PCFC_HD * sizeof(float));
            _patch_cfc_encode(img, w, h, ch, rx * st, ry * st, ps, ps,
                              clf->pW_gx, clf->pW_ax, clf->pW_gh, clf->pW_ah,
                              clf->pb_g, clf->pb_a, patch_h, PCFC_HD, tau, dt);
            _cfc_ode_step(patch_h, PCFC_HD, clf->W_gx_p, clf->W_ax_p,
                          clf->W_gh_p, clf->W_ah_p, clf->b_g_p, clf->b_a_p,
                          hidden, hd, tau, dt);
            _vec_copy(&pf[pi * hd], hidden, hd);
            pi++;
        }
    }
    *np = pi;
}

static void _saliency(const float* pf, int np, int hd, float* sal, int* ns, float th) {
    float mx = 0.0f;
    for (int i = 0; i < np; i++) { float n = _vec_norm(&pf[i * hd], hd); sal[i] = n; if (n > mx) mx = n; }
    *ns = 0;
    if (mx > 1e-6f) { for (int i = 0; i < np; i++) { sal[i] /= mx; if (sal[i] > th) (*ns)++; } }
}

static void _agg_global(const float* pf, int np, int hd, const float* sal, float* gf) {
    float tw = 0.0f; memset(gf, 0, hd * sizeof(float));
    for (int i = 0; i < np; i++) { float w = sal[i]; for (int j = 0; j < hd; j++) gf[j] += w * pf[i * hd + j]; tw += w; }
    if (tw > 1e-10f) for (int j = 0; j < hd; j++) gf[j] /= tw;
}

static void _bilinear(const float* gf, const float* lf, float* bl, int hd,
                       const float* W) {
    float pg[256], pl[256]; memset(pg, 0, hd * sizeof(float)); memset(pl, 0, hd * sizeof(float));
    _mat_vec_mul(W, gf, pg, hd, hd); _mat_vec_mul(W, lf, pl, hd, hd);
    for (int i = 0; i < hd; i++) bl[i] = pg[i] * pl[i];
    for (int i = 0; i < hd; i++) bl[i] = copysignf(sqrtf(fabsf(bl[i]) + 1e-8f), bl[i]);
    _vec_normalize(bl, hd);
}

static void _locate_parts(IRDFineClassifier* clf, const float* pf, int np,
                           int cols, int ps, int st,
                           IRDDiscriminativePart* parts, int* nump) {
    int hd = clf->cfg.feature_dim;
    float sal[256]; int ns;
    _saliency(pf, np, hd, sal, &ns, clf->cfg.discriminative_threshold);
    int maxp = clf->cfg.num_parts; if (maxp > 32) maxp = 32;
    int sel = 0, used[256]; memset(used, 0, np * sizeof(int));
    for (int it = 0; it < maxp && sel < maxp; it++) {
        int bi = -1; float bs = -1.0f;
        for (int i = 0; i < np; i++) { if (used[i]) continue; if (sal[i] > bs) { bs = sal[i]; bi = i; } }
        if (bi < 0 || bs < clf->cfg.discriminative_threshold * 0.5f) break;
        int rx = bi % cols, ry = bi / cols, cx = rx * st, cy = ry * st;
        int too_close = 0;
        for (int p = 0; p < sel; p++) {
            float dx = (float)cx - parts[p].x, dy = (float)cy - parts[p].y;
            if (sqrtf(dx * dx + dy * dy) < (float)ps * 0.7f) too_close = 1;
        }
        if (too_close) { used[bi] = 1; continue; }
        for (int i = 0; i < np; i++) {
            int prx = i % cols, pry = i / cols;
            if (sqrtf((float)((rx-prx)*(rx-prx)+(ry-pry)*(ry-pry))) * (float)st < (float)ps * 0.5f) used[i] = 1;
        }
        parts[sel].part_id = sel; parts[sel].part_type = IRD_PART_OTHER;
        parts[sel].x = (float)cx; parts[sel].y = (float)cy;
        parts[sel].width = (float)ps; parts[sel].height = (float)ps;
        parts[sel].discriminative_score = bs;
        _vec_copy(parts[sel].feature_vector, &pf[bi * hd], hd);
        memcpy(parts[sel].saliency_map, sal, np * sizeof(float));
        parts[sel].num_salient_patches = ns;
        sel++;
    }
    *nump = sel;
}

/* ZSFX-004修复: 多特征组合未训练回退分类器
 * 使用 Sobel梯度方向直方图(6方向) + LBP纹理模式(8模式) + HSV颜色直方图(12色相桶)
 * 实现至少10种场景类型的细粒度分类，置信度基于特征匹配得分 */
#define FALLBACK_GRAD_DIRS     6
#define FALLBACK_LBP_PATTERNS  8
#define FALLBACK_HUE_BINS      12
#define FALLBACK_NUM_SCENES    10
#define FALLBACK_FEAT_DIM      (FALLBACK_GRAD_DIRS + FALLBACK_LBP_PATTERNS + FALLBACK_HUE_BINS)

/* 场景原型的特征向量（预定义10种场景的归一化特征模板） */
static const float g_scene_prototypes[FALLBACK_NUM_SCENES][FALLBACK_FEAT_DIM] = {
    /* 0: 文字/文档 - 高对比度、双边纹理、黑白为主 */
    {0.08f,0.07f,0.06f,0.07f,0.08f,0.06f, 0.45f,0.08f,0.05f,0.12f,0.08f,0.35f,0.06f,0.04f, 0.65f,0.15f,0.03f,0.02f,0.03f,0.01f,0.01f,0.01f,0.02f,0.03f,0.02f,0.02f},
    /* 1: 室内场景 - 均匀梯度分布、规则纹理、暖色调 */
    {0.12f,0.11f,0.10f,0.13f,0.12f,0.10f, 0.04f,0.12f,0.06f,0.16f,0.14f,0.10f,0.08f,0.05f, 0.08f,0.12f,0.18f,0.22f,0.15f,0.08f,0.05f,0.04f,0.03f,0.02f,0.01f,0.02f},
    /* 2: 室外自然 - 水平渐变多、树叶/草纹理、绿色为主 */
    {0.18f,0.10f,0.06f,0.15f,0.09f,0.05f, 0.02f,0.04f,0.08f,0.28f,0.20f,0.10f,0.06f,0.03f, 0.03f,0.04f,0.07f,0.25f,0.30f,0.14f,0.07f,0.04f,0.02f,0.01f,0.01f,0.02f},
    /* 3: 城市建筑 - 垂直+水平渐变主导、规则纹理、灰蓝调 */
    {0.22f,0.06f,0.04f,0.23f,0.05f,0.03f, 0.02f,0.03f,0.05f,0.15f,0.25f,0.18f,0.10f,0.05f, 0.05f,0.08f,0.15f,0.18f,0.15f,0.10f,0.08f,0.06f,0.05f,0.04f,0.03f,0.03f},
    /* 4: 人物肖像 - 中心渐变分布、光滑纹理、肤色暖调 */
    {0.10f,0.09f,0.11f,0.12f,0.10f,0.09f, 0.02f,0.06f,0.10f,0.22f,0.18f,0.14f,0.08f,0.03f, 0.02f,0.05f,0.12f,0.28f,0.20f,0.12f,0.08f,0.05f,0.03f,0.02f,0.01f,0.02f},
    /* 5: 夜景/暗光 - 低梯度响应、噪声纹理、暗蓝色调 */
    {0.02f,0.02f,0.03f,0.02f,0.03f,0.02f, 0.30f,0.20f,0.12f,0.08f,0.06f,0.05f,0.04f,0.02f, 0.02f,0.04f,0.08f,0.05f,0.03f,0.02f,0.03f,0.06f,0.12f,0.18f,0.20f,0.17f},
    /* 6: 天空/水景 - 水平渐变主导、光滑低纹理、蓝色为主 */
    {0.25f,0.08f,0.04f,0.05f,0.03f,0.02f, 0.01f,0.03f,0.18f,0.30f,0.22f,0.08f,0.04f,0.01f, 0.01f,0.02f,0.03f,0.04f,0.05f,0.08f,0.18f,0.28f,0.18f,0.08f,0.03f,0.02f},
    /* 7: 食物/餐桌 - 中高梯度圆形分布、细密纹理、暖色调 */
    {0.06f,0.08f,0.10f,0.09f,0.07f,0.06f, 0.03f,0.08f,0.15f,0.20f,0.16f,0.14f,0.07f,0.03f, 0.02f,0.06f,0.15f,0.25f,0.18f,0.12f,0.08f,0.05f,0.03f,0.02f,0.02f,0.02f},
    /* 8: 运动/高速 - 方向性渐变单一主导、低纹理、全色温 */
    {0.25f,0.05f,0.03f,0.03f,0.02f,0.02f, 0.02f,0.08f,0.25f,0.20f,0.12f,0.08f,0.05f,0.03f, 0.05f,0.10f,0.15f,0.15f,0.12f,0.10f,0.08f,0.07f,0.06f,0.05f,0.04f,0.03f},
    /* 9: 纹理/图案 - 全方向梯度均匀、LBP高响应、全色域 */
    {0.12f,0.10f,0.09f,0.11f,0.10f,0.12f, 0.01f,0.01f,0.02f,0.05f,0.14f,0.25f,0.22f,0.14f, 0.05f,0.08f,0.10f,0.12f,0.14f,0.11f,0.09f,0.08f,0.07f,0.06f,0.05f,0.05f}
};
static const char* g_scene_names[FALLBACK_NUM_SCENES] = {
    "文字/文档", "室内场景", "室外自然", "城市建筑", "人物肖像",
    "夜景/暗光", "天空/水景", "食物/餐桌", "运动/高速", "纹理/图案"
};

/* 提取Sobel梯度方向直方图(6方向: 0°,30°,60°,90°,120°,150°) */
static void _extract_gradient_hist(const float* gray, int w, int h,
                                    float ghist[FALLBACK_GRAD_DIRS]) {
    memset(ghist, 0, FALLBACK_GRAD_DIRS * sizeof(float));
    int total = 0;
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            int idx = y * w + x;
            float gx = gray[y * w + (x + 1)] - gray[y * w + (x - 1)];
            float gy = gray[(y + 1) * w + x] - gray[(y - 1) * w + x];
            float mag = sqrtf(gx * gx + gy * gy + 1e-8f);
            if (mag < 0.02f) continue;
            float angle = atan2f(gy, gx) * 180.0f / (float)M_PI;
            if (angle < 0) angle += 180.0f;
            /* 量化到6个方向: 0-29,30-59,60-89,90-119,120-149,150-179 */
            int bin = (int)(angle / 30.0f);
            if (bin < 0) bin = 0;
            if (bin >= FALLBACK_GRAD_DIRS) bin = FALLBACK_GRAD_DIRS - 1;
            ghist[bin] += mag;
            total++;
        }
    }
    if (total > 0) {
        float inv = 1.0f / (float)total;
        for (int i = 0; i < FALLBACK_GRAD_DIRS; i++) ghist[i] *= inv;
    }
    /* L1归一化 */
    float sum = 0;
    for (int i = 0; i < FALLBACK_GRAD_DIRS; i++) sum += ghist[i];
    if (sum > 1e-8f) for (int i = 0; i < FALLBACK_GRAD_DIRS; i++) ghist[i] /= sum;
}

/* 提取LBP纹理模式(8邻域,模式统计) */
static void _extract_lbp_patterns(const float* gray, int w, int h,
                                   float lbp[FALLBACK_LBP_PATTERNS + 1]) {
    /* lbp[0..7]: 均匀模式计数, lbp[8]: 0 or 1模式总占比 */
    memset(lbp, 0, (FALLBACK_LBP_PATTERNS + 1) * sizeof(float));
    int total = 0;
    int radius = 1;
    /* LBP采样点: 8邻域按逆时针排列 (左上开始) */
    int dx[8] = {-1, 0, 1, 1, 1, 0, -1, -1};
    int dy[8] = {-1,-1,-1, 0, 1, 1, 1, 0};
    for (int y = radius; y < h - radius; y++) {
        for (int x = radius; x < w - radius; x++) {
            int ci = y * w + x;
            float center = gray[ci];
            int code = 0;
            int ones = 0;
            for (int k = 0; k < 8; k++) {
                int ni = (y + dy[k]) * w + (x + dx[k]);
                if (gray[ni] >= center) { code |= (1 << k); ones++; }
            }
            /* 统计该码的跳变次数作为LBP模式类别
             * 均匀模式(0次跳变) → lbp[0], 1次跳变 → lbp[1], ..., 7次跳变 → lbp[7] */
            int transitions = 0;
            int prev = (code >> 7) & 1;
            for (int k = 0; k < 8; k++) {
                int cur = (code >> k) & 1;
                if (cur != prev) transitions++;
                prev = cur;
            }
            if (transitions < 0) transitions = 0;
            if (transitions >= FALLBACK_LBP_PATTERNS) transitions = FALLBACK_LBP_PATTERNS - 1;
            lbp[transitions] += 1.0f;
            lbp[FALLBACK_LBP_PATTERNS] += (float)ones / 8.0f;
            total++;
        }
    }
    if (total > 0) {
        float inv = 1.0f / (float)total;
        for (int i = 0; i < FALLBACK_LBP_PATTERNS; i++) lbp[i] *= inv;
        lbp[FALLBACK_LBP_PATTERNS] *= inv;
    }
    /* L1归一化 */
    float sum = 0;
    for (int i = 0; i < FALLBACK_LBP_PATTERNS; i++) sum += lbp[i];
    if (sum > 1e-8f) for (int i = 0; i < FALLBACK_LBP_PATTERNS; i++) lbp[i] /= sum;
}

/* 提取HSV颜色直方图(12色相桶) - 从RGB图像计算 */
static void _extract_hsv_hist(const float* img, int w, int h, int ch,
                               float hhist[FALLBACK_HUE_BINS]) {
    memset(hhist, 0, FALLBACK_HUE_BINS * sizeof(float));
    int total = 0;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int idx = y * w + x;
            float r = (ch >= 1) ? img[(size_t)idx * ch + 0] : 0.0f;
            float g = (ch >= 2) ? img[(size_t)idx * ch + 1] : r;
            float b = (ch >= 3) ? img[(size_t)idx * ch + 2] : r;
            /* RGB → HSV 色相计算 */
            float mx = (r > g) ? ((r > b) ? r : b) : ((g > b) ? g : b);
            float mn = (r < g) ? ((r < b) ? r : b) : ((g < b) ? g : b);
            float chroma = mx - mn;
            float hue = 0.0f;
            if (chroma > 1e-7f) {
                if (mx == r) {
                    hue = 60.0f * (g - b) / chroma;
                    if (hue < 0) hue += 360.0f;
                } else if (mx == g) {
                    hue = 60.0f * ((b - r) / chroma + 2.0f);
                } else {
                    hue = 60.0f * ((r - g) / chroma + 4.0f);
                }
            }
            int bin = (int)(hue / 30.0f);
            if (bin < 0) bin = 0;
            if (bin >= FALLBACK_HUE_BINS) bin = FALLBACK_HUE_BINS - 1;
            /* 使用饱和度加权 */
            float sat = (mx > 1e-7f) ? chroma / mx : 0.0f;
            float val = mx;
            float weight = sat * val + 0.1f;
            hhist[bin] += weight;
            total++;
        }
    }
    if (total > 0) {
        /* 归一化 */
        float sum = 0;
        for (int i = 0; i < FALLBACK_HUE_BINS; i++) sum += hhist[i];
        if (sum > 1e-8f)
            for (int i = 0; i < FALLBACK_HUE_BINS; i++) hhist[i] /= sum;
    }
}

/* 计算两个特征向量的余弦相似度 */
static float _cosine_sim(const float* a, const float* b, int dim) {
    float dot = 0, na = 0, nb = 0;
    for (int i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    float denom = sqrtf(na * nb + 1e-10f);
    return dot / denom;
}

int ird_fine_classify(IRDFineClassifier* clf, const float* img,
                       int w, int h, int ch, IRDFineGrainedResult* res) {
    if (!clf || !img || !res) return -1;
    memset(res, 0, sizeof(IRDFineGrainedResult));

    /* ZSFX-004修复: 未训练状态使用多特征组合回退分类
     * Sobel梯度方向直方图(6方向) + LBP纹理模式(8模式) + HSV颜色直方图(12色相桶)
     * 与10种场景原型做余弦相似度匹配，置信度基于匹配得分 */
    if (!clf->training_completed) {
        /* 根据图像尺寸分配灰度图缓冲区 */
        int total_px = w * h;
        /* 限制最大图像尺寸防止栈溢出，大图做降采样 */
        int dw = w, dh = h;
        int ds_w = w, ds_h = h;
        float* gray = NULL;
        float* gray_small = NULL;
        const float* use_gray = NULL;
        int use_w, use_h;
        /* 如果图像太大，降采样到最大边长256 */
        if (w > 256 || h > 256) {
            ds_w = (w > h) ? 256 : (256 * w / h);
            ds_h = (h > w) ? 256 : (256 * h / w);
            if (ds_w < 4) ds_w = 4; if (ds_h < 4) ds_h = 4;
            gray = (float*)malloc((size_t)total_px * sizeof(float));
            gray_small = (float*)malloc((size_t)(ds_w * ds_h) * sizeof(float));
            if (!gray || !gray_small) {
                if (gray) free(gray);
                if (gray_small) free(gray_small);
                snprintf(res->coarse_category_name, 64, "内存不足");
                res->coarse_confidence = 0.0f;
                return -1;
            }
            /* 转为灰度图 */
            for (int i = 0; i < total_px; i++) {
                float r = (ch >= 1) ? img[(size_t)i * ch + 0] : 0.0f;
                float gval = (ch >= 2) ? img[(size_t)i * ch + 1] : r;
                float b = (ch >= 3) ? img[(size_t)i * ch + 2] : r;
                gray[i] = 0.299f * r + 0.587f * gval + 0.114f * b;
            }
            /* 最近邻降采样到ds_w×ds_h */
            for (int y = 0; y < ds_h; y++) {
                for (int x = 0; x < ds_w; x++) {
                    int sx = x * w / ds_w;
                    int sy = y * h / ds_h;
                    if (sx >= w) sx = w - 1;
                    if (sy >= h) sy = h - 1;
                    gray_small[y * ds_w + x] = gray[sy * w + sx];
                }
            }
            use_gray = gray_small;
            use_w = ds_w;
            use_h = ds_h;
        } else {
            gray = (float*)malloc((size_t)total_px * sizeof(float));
            if (!gray) {
                snprintf(res->coarse_category_name, 64, "内存不足");
                res->coarse_confidence = 0.0f;
                return -1;
            }
            for (int i = 0; i < total_px; i++) {
                float r = (ch >= 1) ? img[(size_t)i * ch + 0] : 0.0f;
                float gval = (ch >= 2) ? img[(size_t)i * ch + 1] : r;
                float b = (ch >= 3) ? img[(size_t)i * ch + 2] : r;
                gray[i] = 0.299f * r + 0.587f * gval + 0.114f * b;
            }
            use_gray = gray;
            use_w = w;
            use_h = h;
        }

        /* 提取三组特征 */
        float ghist[FALLBACK_GRAD_DIRS];
        float lbp[FALLBACK_LBP_PATTERNS + 1];  /* +1用于均值 */
        float hhist[FALLBACK_HUE_BINS];
        _extract_gradient_hist(use_gray, use_w, use_h, ghist);
        _extract_lbp_patterns(use_gray, use_w, use_h, lbp);
        /* HSV使用原始分辨率图像(降采样版本也可) */
        if (gray_small) {
            /* 对降采样后的图重新构建伪RGB用于HSV提取 */
            float* ds_img = (float*)malloc((size_t)(ds_w * ds_h) * 3 * sizeof(float));
            if (ds_img) {
                for (int i = 0; i < ds_w * ds_h; i++) {
                    float v = gray_small[i];
                    ds_img[i * 3 + 0] = v;
                    ds_img[i * 3 + 1] = v;
                    ds_img[i * 3 + 2] = v;
                }
                _extract_hsv_hist(ds_img, ds_w, ds_h, 3, hhist);
                free(ds_img);
            } else {
                memset(hhist, 0, FALLBACK_HUE_BINS * sizeof(float));
            }
        } else {
            _extract_hsv_hist(img, w, h, ch, hhist);
        }

        /* 释放灰度缓冲区 */
        if (gray) free(gray);
        if (gray_small) free(gray_small);

        /* 组合特征向量: [梯度6 + LBP8 + 色调12] = 26维 */
        float feat[FALLBACK_FEAT_DIM];
        memcpy(feat, ghist, FALLBACK_GRAD_DIRS * sizeof(float));
        memcpy(feat + FALLBACK_GRAD_DIRS, lbp, FALLBACK_LBP_PATTERNS * sizeof(float));
        memcpy(feat + FALLBACK_GRAD_DIRS + FALLBACK_LBP_PATTERNS, hhist, FALLBACK_HUE_BINS * sizeof(float));

        /* 与所有场景原型计算余弦相似度，选最佳匹配 */
        int best_scene = 0;
        float best_sim = -1.0f;
        for (int s = 0; s < FALLBACK_NUM_SCENES; s++) {
            float sim = _cosine_sim(feat, g_scene_prototypes[s], FALLBACK_FEAT_DIM);
            if (sim > best_sim) { best_sim = sim; best_scene = s; }
        }
        /* 计算置信度: 对余弦相似度做sigmoid映射到[0.1, 0.85]区间 */
        float raw_conf = 1.0f / (1.0f + expf(-(best_sim - 0.3f) * 8.0f));
        res->coarse_confidence = 0.1f + raw_conf * 0.75f;
        if (res->coarse_confidence > 0.85f) res->coarse_confidence = 0.85f;
        if (res->coarse_confidence < 0.1f) res->coarse_confidence = 0.1f;

        snprintf(res->coarse_category_name, 64, "%s", g_scene_names[best_scene]);
        snprintf(res->fine_category_name, 64, "未训练-多特征分类(%s)", g_scene_names[best_scene]);
        return 0;
    }
    int hd = clf->cfg.feature_dim, ps = clf->cfg.patch_size, st = ps / 2;
    int cols = (w - ps) / st + 1;
    if (cols < 1) cols = 1;
    float pf[256 * 256]; int np;
    _extract_patches_cfc(clf, img, w, h, ch, pf, &np);
    float sal[256]; int ns;
    _saliency(pf, np, hd, sal, &ns, clf->cfg.discriminative_threshold);
    _agg_global(pf, np, hd, sal, res->global_feature);
    _locate_parts(clf, pf, np, cols, ps, st, res->parts, &res->num_parts);
    memset(res->local_feature, 0, hd * sizeof(float));
    for (int p = 0; p < res->num_parts; p++) {
        for (int j = 0; j < hd; j++)
            res->local_feature[j] += res->parts[p].feature_vector[j] * res->parts[p].discriminative_score;
    }
    if (res->num_parts > 0) {
        float tw = 0.0f; for (int p = 0; p < res->num_parts; p++) tw += res->parts[p].discriminative_score;
        if (tw > 1e-10f) _vec_scale(res->local_feature, 1.0f / tw, hd);
    }
    if (clf->cfg.enable_bilinear_pooling)
        _bilinear(res->global_feature, res->local_feature, res->bilinear_feature, hd, clf->bilin_W);
    else
        _vec_copy(res->bilinear_feature, res->global_feature, hd);

    float clog[64];
    _mat_vec_mul(clf->W_coarse, res->bilinear_feature, clog, clf->cfg.num_coarse_categories, hd);
    _vec_add(clog, clf->b_coarse, clf->cfg.num_coarse_categories);
    _softmax(clog, clf->cfg.num_coarse_categories);

    float flog[256];
    _mat_vec_mul(clf->W_fine, res->bilinear_feature, flog, clf->cfg.num_fine_categories, hd);
    _vec_add(flog, clf->b_fine, clf->cfg.num_fine_categories);
    _softmax(flog, clf->cfg.num_fine_categories);

    int bc = 0; float bcc = 0.0f;
    for (int i = 0; i < clf->cfg.num_coarse_categories; i++) { if (clog[i] > bcc) { bcc = clog[i]; bc = i; } }
    int bf = 0; float bfc = 0.0f;
    for (int i = 0; i < clf->cfg.num_fine_categories; i++) { if (flog[i] > bfc) { bfc = flog[i]; bf = i; } }
    res->fine_category_id = bf; res->coarse_category_id = bc;
    res->fine_confidence = bfc; res->coarse_confidence = bcc;
    if (bc < 64) snprintf(res->coarse_category_name, 64, "%s", clf->cfg.coarse_names[bc]);
    if (bf < 256) snprintf(res->fine_category_name, 64, "%s", clf->cfg.fine_names[bf]);
    return 0;
}

int ird_fine_classify_batch(IRDFineClassifier* c, const float* im,
                             int w, int h, int ch, int bs, IRDFineGrainedResult* rs) {
    if (!c || !im || !rs) return -1;
    for (int i = 0; i < bs; i++) { int r = ird_fine_classify(c, &im[i * w * h * ch], w, h, ch, &rs[i]); if (r) return r; }
    return 0;
}

int ird_fine_locate_parts(IRDFineClassifier* clf, const float* img,
                           int w, int h, int ch, IRDDiscriminativePart* parts, int* np) {
    if (!clf || !img || !parts || !np) return -1;
    int ps = clf->cfg.patch_size, st = ps / 2;
    int cols = (w - ps) / st + 1;
    if (cols < 1) cols = 1;
    float pf[256 * 256]; int npp;
    _extract_patches_cfc(clf, img, w, h, ch, pf, &npp);
    _locate_parts(clf, pf, npp, cols, ps, st, parts, np);
    return 0;
}

int ird_fine_train(IRDFineClassifier* clf, const float* im, const int* lb,
                    int n, int w, int h, int ch, int ep, float lr) {
    if (!clf || !im || !lb || n <= 0) return -1;
    int hd = clf->cfg.feature_dim, nf = clf->cfg.num_fine_categories;
    for (int e = 0; e < ep; e++) {
        float tl = 0.0f;
        for (int s = 0; s < n; s++) {
            IRDFineGrainedResult r;
            int ret = ird_fine_classify(clf, &im[s * w * h * ch], w, h, ch, &r);
            if (ret) continue;
            int la = lb[s]; if (la < 0 || la >= nf) continue;
            float log[256];
            _mat_vec_mul(clf->W_fine, r.bilinear_feature, log, nf, hd);
            _vec_add(log, clf->b_fine, nf); _softmax(log, nf);
            tl += -logf(log[la] + 1e-10f);
            float grad[256];
            for (int i = 0; i < nf; i++) grad[i] = log[i] - (i == la ? 1.0f : 0.0f);
            for (int i = 0; i < nf; i++)
                for (int j = 0; j < hd; j++) clf->W_fine[i * hd + j] -= lr * grad[i] * r.bilinear_feature[j];
            for (int i = 0; i < nf; i++) clf->b_fine[i] -= lr * grad[i];
        }
        if (tl / (float)n < 0.01f) break;
    }
    return 0;
}

int ird_fine_save(const IRDFineClassifier* c, const char* p) {
    if (!c || !p) return -1; FILE* f = fopen(p, "wb"); if (!f) return -1;
    fwrite(&c->cfg, sizeof(IRDFineConfig), 1, f);
    fwrite(c->W_gx_p, sizeof(float), 256 * 32, f); fwrite(c->W_ax_p, sizeof(float), 256 * 32, f);
    fwrite(c->W_gh_p, sizeof(float), 256 * 256, f); fwrite(c->W_ah_p, sizeof(float), 256 * 256, f);
    fwrite(c->b_g_p, sizeof(float), 256, f); fwrite(c->b_a_p, sizeof(float), 256, f);
    fwrite(c->pW_gx, sizeof(float), PCFC_HD * PCFC_MAX_CH, f);
    fwrite(c->pW_ax, sizeof(float), PCFC_HD * PCFC_MAX_CH, f);
    fwrite(c->pW_gh, sizeof(float), PCFC_HD * PCFC_HD, f);
    fwrite(c->pW_ah, sizeof(float), PCFC_HD * PCFC_HD, f);
    fwrite(c->pb_g, sizeof(float), PCFC_HD, f);
    fwrite(c->pb_a, sizeof(float), PCFC_HD, f);
    fclose(f); return 0;
}

int ird_fine_load(IRDFineClassifier* c, const char* p) {
    if (!c || !p) return -1; FILE* f = fopen(p, "rb"); if (!f) return -1;
    fread(&c->cfg, sizeof(IRDFineConfig), 1, f);
    fread(c->W_gx_p, sizeof(float), 256 * 32, f); fread(c->W_ax_p, sizeof(float), 256 * 32, f);
    fread(c->W_gh_p, sizeof(float), 256 * 256, f); fread(c->W_ah_p, sizeof(float), 256 * 256, f);
    fread(c->b_g_p, sizeof(float), 256, f); fread(c->b_a_p, sizeof(float), 256, f);
    fread(c->pW_gx, sizeof(float), PCFC_HD * PCFC_MAX_CH, f);
    fread(c->pW_ax, sizeof(float), PCFC_HD * PCFC_MAX_CH, f);
    fread(c->pW_gh, sizeof(float), PCFC_HD * PCFC_HD, f);
    fread(c->pW_ah, sizeof(float), PCFC_HD * PCFC_HD, f);
    fread(c->pb_g, sizeof(float), PCFC_HD, f);
    fread(c->pb_a, sizeof(float), PCFC_HD, f);
    fclose(f); return 0;
}

/* ======================================================================== */
/*  开放集识别                                                              */
/* ======================================================================== */

struct IRDOpenSetRecognizer {
    IRDOpenSetConfig cfg;
    float prototypes[256 * 256], W_gx[256 * 32], W_ax[256 * 32];
    float W_gh[256 * 256], W_ah[256 * 256], b_g[256], b_a[256];
    float hidden[256], inv_temp, reject_threshold;
    int num_known_classes, total_samples;
    float weibull_k[256], weibull_lambda[256], weibull_thresh;
    float nndr_threshold;
    float pW_gx[PCFC_HD * PCFC_MAX_CH], pW_ax[PCFC_HD * PCFC_MAX_CH];
    float pW_gh[PCFC_HD * PCFC_HD], pW_ah[PCFC_HD * PCFC_HD];
    float pb_g[PCFC_HD], pb_a[PCFC_HD];
};

IRDOpenSetConfig ird_open_set_get_default_config(void) {
    IRDOpenSetConfig c; memset(&c, 0, sizeof(c));
    c.feature_dim = 256; c.input_dim = 32; c.num_known = 128;
    c.cfc_time_constant = 0.1f; c.cfc_delta_t = 0.05f;
    c.rejection_threshold = 0.4f; c.nndr_threshold = 0.7f;
    c.rbf_gamma = 0.5f; c.weibull_fit_size = 20;
    c.temperature = 0.1f;
    return c;
}

IRDOpenSetRecognizer* ird_open_set_create(const IRDOpenSetConfig* config) {
    if (!config) return NULL;
    IRDOpenSetRecognizer* rec = (IRDOpenSetRecognizer*)safe_calloc(1, sizeof(IRDOpenSetRecognizer));
    if (!rec) return NULL;
    memcpy(&rec->cfg, config, sizeof(IRDOpenSetConfig));
    rec->inv_temp = 1.0f / (config->temperature + 1e-6f);
    rec->reject_threshold = config->rejection_threshold;
    rec->nndr_threshold = config->nndr_threshold;
    rec->weibull_thresh = 0.5f;
    int hd = config->feature_dim, id = config->input_dim;
    _xavier_init(rec->W_gx, id, hd); _xavier_init(rec->W_ax, id, hd);
    memset(rec->b_g, 0, hd * sizeof(float)); memset(rec->b_a, 0, hd * sizeof(float));
    memset(rec->hidden, 0, hd * sizeof(float));
    _xavier_init(rec->pW_gx, PCFC_MAX_CH, PCFC_HD);
    _xavier_init(rec->pW_ax, PCFC_MAX_CH, PCFC_HD);
    _xavier_init(rec->pW_gh, PCFC_HD, PCFC_HD);
    _xavier_init(rec->pW_ah, PCFC_HD, PCFC_HD);
    memset(rec->pb_g, 0, PCFC_HD * sizeof(float));
    memset(rec->pb_a, 0, PCFC_HD * sizeof(float));
    rec->num_known_classes = 0; rec->total_samples = 0;
    return rec;
}

void ird_open_set_free(IRDOpenSetRecognizer* recognizer) {
    if (!recognizer) return;
    memset(recognizer, 0, sizeof(IRDOpenSetRecognizer));
    safe_free((void**)&recognizer);
}

int ird_open_set_set_threshold(IRDOpenSetRecognizer* recognizer, float threshold) {
    if (!recognizer) return -1;
    if (threshold < 0.0f || threshold > 1.0f) return -1;
    recognizer->reject_threshold = threshold;
    return 0;
}

int ird_open_set_update_rejection(IRDOpenSetRecognizer* recognizer, float threshold) {
    if (!recognizer) return -1;
    if (threshold >= 0.0f && threshold <= 1.0f) {
        recognizer->reject_threshold = threshold;
    }
    float max_thresh = recognizer->reject_threshold;
    for (int i = 0; i < recognizer->num_known_classes; i++) {
        float w = recognizer->weibull_lambda[i];
        float k = recognizer->weibull_k[i];
        if (k > 0.0f && w > 0.0f) {
            float p = 1.0f - expf(-powf(recognizer->reject_threshold / w, k));
            if (p > max_thresh) max_thresh = p;
        }
    }
    recognizer->weibull_thresh = max_thresh;
    return 0;
}

int ird_open_set_learn_known(IRDOpenSetRecognizer* recognizer, const float* features, const int* labels, int n) {
    if (!recognizer || !features || !labels || n <= 0) return -1;
    int hd = recognizer->cfg.feature_dim;
    int max_classes = 256;
    int class_counts[256];
    float class_sums[256 * 256];
    int class_to_idx[65536];
    memset(class_counts, 0, sizeof(class_counts));
    memset(class_sums, 0, sizeof(class_sums));
    memset(class_to_idx, 0xFF, sizeof(class_to_idx));

    int num_unique = 0;
    for (int i = 0; i < n && i < max_classes * 1000; i++) {
        int lbl = labels[i];
        if (lbl < 0) continue;
        if (class_to_idx[lbl % 65536] == -1 || class_counts[lbl % 65536] == 0) {
            if (num_unique >= max_classes) break;
            class_to_idx[lbl % 65536] = num_unique;
            num_unique++;
        }
        int ci = class_to_idx[lbl % 65536];
        class_counts[ci]++;
        float* sum_ptr = &class_sums[ci * hd];
        for (int d = 0; d < hd; d++) sum_ptr[d] += features[i * hd + d];
    }

    recognizer->num_known_classes = num_unique;
    for (int ci = 0; ci < num_unique; ci++) {
        float* proto = &recognizer->prototypes[ci * hd];
        float* sum_ptr = &class_sums[ci * hd];
        float cnt = (float)(class_counts[ci] > 0 ? class_counts[ci] : 1);
        for (int d = 0; d < hd; d++) proto[d] = sum_ptr[d] / cnt;
        _vec_normalize(proto, hd);
        recognizer->total_samples += class_counts[ci];
    }

    float all_dists[256];
    int dist_count = 0;
    for (int ci = 0; ci < num_unique; ci++) {
        float* proto = &recognizer->prototypes[ci * hd];
        float min_d = FLT_MAX;
        for (int i = 0; i < n; i++) {
            int lbl = labels[i]; if (lbl < 0) continue;
            int ci_check = class_to_idx[lbl % 65536];
            if (ci_check == ci) {
                float d = _l2_dist(&features[i * hd], proto, hd);
                if (d < min_d) min_d = d;
            }
        }
        if (min_d < FLT_MAX && dist_count < 256) {
            all_dists[dist_count++] = min_d;
        }
    }
    if (dist_count > 0) ird_open_set_fit_weibull(recognizer, all_dists, dist_count);

    return 0;
}

int ird_open_set_fit_weibull(IRDOpenSetRecognizer* recognizer, const float* known_distances, int num_samples) {
    if (!recognizer || !known_distances || num_samples <= 0) return -1;
    if (num_samples > 256) num_samples = 256;

    float sorted[256];
    memcpy(sorted, known_distances, num_samples * sizeof(float));
    for (int i = 0; i < num_samples - 1; i++)
        for (int j = i + 1; j < num_samples; j++)
            if (sorted[j] < sorted[i]) { float t = sorted[i]; sorted[i] = sorted[j]; sorted[j] = t; }

    float sum_log = 0.0f, sum_pow = 0.0f;
    float eps = 1e-6f;
    for (int i = 0; i < num_samples; i++) {
        float x = sorted[i] + eps;
        sum_log += logf(x);
        sum_pow += x;
    }

    float mean_log = sum_log / (float)num_samples;
    float mean_x = sum_pow / (float)num_samples;

    float k = 1.2f;
    for (int iter = 0; iter < 20; iter++) {
        float numer = 0.0f, denom = 0.0f;
        for (int i = 0; i < num_samples; i++) {
            float x = sorted[i] + eps;
            float xk = powf(x, k);
            numer += xk * logf(x);
            denom += xk;
        }
        float gk = numer / (denom + eps) - 1.0f / k - mean_log;
        float gk_prime = 0.0f;
        for (int i = 0; i < num_samples; i++) {
            float x = sorted[i] + eps;
            float xk = powf(x, k);
            gk_prime += xk * logf(x) * logf(x) / denom -
                        (xk * logf(x)) * (xk * logf(x)) / (denom * denom + eps);
        }
        gk_prime += 1.0f / (k * k + eps);
        if (fabsf(gk_prime) < 1e-10f) break;
        float dk = gk / gk_prime;
        k -= dk;
        if (k < 0.1f) k = 0.1f;
        if (k > 10.0f) k = 10.0f;
        if (fabsf(dk) < 1e-6f) break;
    }

    float sum_k = 0.0f;
    for (int i = 0; i < num_samples; i++) sum_k += powf(sorted[i] + eps, k);
    float lambda = powf(sum_k / (float)num_samples, 1.0f / (k + eps));

    for (int i = 0; i < recognizer->num_known_classes && i < 256; i++) {
        recognizer->weibull_k[i] = k;
        recognizer->weibull_lambda[i] = lambda;
    }

    ird_open_set_update_rejection(recognizer, recognizer->reject_threshold);
    return 0;
}

int ird_open_set_predict(IRDOpenSetRecognizer* recognizer, const float* features, IRDOpenSetPrediction* prediction, int top_k) {
    if (!recognizer || !features || !prediction) return -1;
    memset(prediction, 0, sizeof(IRDOpenSetPrediction));

    int hd = recognizer->cfg.feature_dim;
    int nc = recognizer->num_known_classes;

    float query_h[256];
    memset(query_h, 0, sizeof(query_h));
    _cfc_ode_step(features, hd, recognizer->W_gx, recognizer->W_ax,
                  recognizer->W_gh, recognizer->W_ah,
                  recognizer->b_g, recognizer->b_a,
                  query_h, hd, recognizer->cfg.cfc_time_constant, recognizer->cfg.cfc_delta_t);

    float dists[256];
    float min_dist = FLT_MAX;
    int best_class = -1;
    float second_min = FLT_MAX;

    for (int ci = 0; ci < nc; ci++) {
        dists[ci] = _l2_dist(query_h, &recognizer->prototypes[ci * hd], hd);
        if (dists[ci] < min_dist) {
            second_min = min_dist;
            min_dist = dists[ci];
            best_class = ci;
        } else if (dists[ci] < second_min) {
            second_min = dists[ci];
        }
    }

    float ratio = (second_min < FLT_MAX && second_min > 1e-10f) ? (min_dist / second_min) : 0.0f;
    float score = expf(-min_dist * recognizer->inv_temp);
    float weibull_prob = 1.0f;
    if (nc > 0 && best_class >= 0 && best_class < nc) {
        float w = recognizer->weibull_lambda[best_class];
        float k = recognizer->weibull_k[best_class];
        if (w > 0.0f && k > 0.0f) {
            weibull_prob = 1.0f - expf(-powf(min_dist / w, k));
        }
    }

    prediction->assigned_class = best_class;
    prediction->confidence = score;
    prediction->distance_to_known = min_dist;
    prediction->nndr_score = ratio;
    prediction->evt_score = weibull_prob;
    prediction->is_unknown = (ratio > recognizer->nndr_threshold ||
                               weibull_prob < recognizer->weibull_thresh ||
                               score < recognizer->reject_threshold) ? 1 : 0;

    if (top_k <= 0) top_k = 1;
    if (top_k > 10) top_k = 10;
    if (top_k > nc) top_k = nc;
    for (int i = 0; i < nc; i++) for (int j = i + 1; j < nc; j++)
        if (dists[j] < dists[i]) { float t = dists[i]; dists[i] = dists[j]; dists[j] = t; }

    prediction->top_k = top_k;
    for (int ki = 0; ki < top_k && ki < 10; ki++) {
        prediction->top_confidence[ki] = (ki < nc) ? expf(-dists[ki] * recognizer->inv_temp) : 0.0f;
    }

    return 0;
}

static int _os_extract_image_features(IRDOpenSetRecognizer* recognizer, const float* image,
                                        int w, int h, int ch, float* features) {
    if (!recognizer || !image || !features) return -1;
    int hd = recognizer->cfg.feature_dim;
    memset(features, 0, hd * sizeof(float));
    memset(recognizer->hidden, 0, hd * sizeof(float));

    int ps = 16, st = 8;
    int patch_count = 0;
    for (int y = 0; y + ps <= h; y += st) {
        for (int x = 0; x + ps <= w; x += st) {
            float patch_vec[256];
            int pidx = 0;
            for (int py = y; py < y + ps && pidx < hd; py++) {
                for (int px = x; px < x + ps && pidx < hd; px++) {
                    float sum_ch = 0.0f;
                    for (int cc = 0; cc < ch; cc++)
                        sum_ch += image[(py * w + px) * ch + cc];
                    patch_vec[pidx++] = sum_ch / (float)ch;
                }
            }
            while (pidx < hd) patch_vec[pidx++] = 0.0f;

            _cfc_ode_step(patch_vec, hd, recognizer->W_gx, recognizer->W_ax,
                          recognizer->W_gh, recognizer->W_ah,
                          recognizer->b_g, recognizer->b_a,
                          recognizer->hidden, hd,
                          recognizer->cfg.cfc_time_constant, recognizer->cfg.cfc_delta_t);
            _vec_add(features, recognizer->hidden, hd);
            patch_count++;
        }
    }
    if (patch_count > 0) _vec_scale(features, 1.0f / (float)patch_count, hd);
    _vec_normalize(features, hd);
    return 0;
}

int ird_open_set_predict_image(IRDOpenSetRecognizer* recognizer, const float* image,
                                 int w, int h, int ch, IRDOpenSetPrediction* prediction) {
    if (!recognizer || !image || !prediction) return -1;
    float features[256];
    if (_os_extract_image_features(recognizer, image, w, h, ch, features) != 0) return -1;
    return ird_open_set_predict(recognizer, features, prediction, 5);
}

int ird_open_set_save(const IRDOpenSetRecognizer* recognizer, const char* path) {
    if (!recognizer || !path) return -1;
    FILE* file = fopen(path, "wb");
    if (!file) return -1;
    const char magic[9] = "SELFIRD1";
    fwrite(magic, 1, 8, file);
    fwrite(&recognizer->cfg, sizeof(IRDOpenSetConfig), 1, file);
    fwrite(&recognizer->inv_temp, sizeof(float), 1, file);
    fwrite(&recognizer->reject_threshold, sizeof(float), 1, file);
    fwrite(&recognizer->num_known_classes, sizeof(int), 1, file);
    fwrite(&recognizer->total_samples, sizeof(int), 1, file);
    fwrite(&recognizer->weibull_thresh, sizeof(float), 1, file);
    fwrite(&recognizer->nndr_threshold, sizeof(float), 1, file);
    int hd = recognizer->cfg.feature_dim;
    if (recognizer->num_known_classes > 0) {
        fwrite(recognizer->prototypes, sizeof(float), recognizer->num_known_classes * hd, file);
    }
    fwrite(recognizer->W_gx, sizeof(float), hd * recognizer->cfg.input_dim, file);
    fwrite(recognizer->W_ax, sizeof(float), hd * recognizer->cfg.input_dim, file);
    fwrite(recognizer->W_gh, sizeof(float), hd * hd, file);
    fwrite(recognizer->W_ah, sizeof(float), hd * hd, file);
    fwrite(recognizer->b_g, sizeof(float), hd, file);
    fwrite(recognizer->b_a, sizeof(float), hd, file);
    fwrite(recognizer->hidden, sizeof(float), hd, file);
    fwrite(recognizer->weibull_k, sizeof(float), hd, file);
    fwrite(recognizer->weibull_lambda, sizeof(float), hd, file);
    fwrite(recognizer->pW_gx, sizeof(float), PCFC_HD * PCFC_MAX_CH, file);
    fwrite(recognizer->pW_ax, sizeof(float), PCFC_HD * PCFC_MAX_CH, file);
    fwrite(recognizer->pW_gh, sizeof(float), PCFC_HD * PCFC_HD, file);
    fwrite(recognizer->pW_ah, sizeof(float), PCFC_HD * PCFC_HD, file);
    fwrite(recognizer->pb_g, sizeof(float), PCFC_HD, file);
    fwrite(recognizer->pb_a, sizeof(float), PCFC_HD, file);
    fclose(file);
    return 0;
}

int ird_open_set_load(IRDOpenSetRecognizer* recognizer, const char* path) {
    if (!recognizer || !path) return -1;
    FILE* file = fopen(path, "rb");
    if (!file) return -1;
    char magic[8];
    if (fread(magic, 1, 8, file) != 8 || memcmp(magic, "SELFIRD1", 8) != 0) {
        fclose(file); return -1;
    }
    fread(&recognizer->cfg, sizeof(IRDOpenSetConfig), 1, file);
    fread(&recognizer->inv_temp, sizeof(float), 1, file);
    fread(&recognizer->reject_threshold, sizeof(float), 1, file);
    fread(&recognizer->num_known_classes, sizeof(int), 1, file);
    fread(&recognizer->total_samples, sizeof(int), 1, file);
    fread(&recognizer->weibull_thresh, sizeof(float), 1, file);
    fread(&recognizer->nndr_threshold, sizeof(float), 1, file);
    int hd = recognizer->cfg.feature_dim;
    if (recognizer->num_known_classes > 0) {
        fread(recognizer->prototypes, sizeof(float), recognizer->num_known_classes * hd, file);
    }
    fread(recognizer->W_gx, sizeof(float), hd * recognizer->cfg.input_dim, file);
    fread(recognizer->W_ax, sizeof(float), hd * recognizer->cfg.input_dim, file);
    fread(recognizer->W_gh, sizeof(float), hd * hd, file);
    fread(recognizer->W_ah, sizeof(float), hd * hd, file);
    fread(recognizer->b_g, sizeof(float), hd, file);
    fread(recognizer->b_a, sizeof(float), hd, file);
    fread(recognizer->hidden, sizeof(float), hd, file);
    fread(recognizer->weibull_k, sizeof(float), hd, file);
    fread(recognizer->weibull_lambda, sizeof(float), hd, file);
    fread(recognizer->pW_gx, sizeof(float), PCFC_HD * PCFC_MAX_CH, file);
    fread(recognizer->pW_ax, sizeof(float), PCFC_HD * PCFC_MAX_CH, file);
    fread(recognizer->pW_gh, sizeof(float), PCFC_HD * PCFC_HD, file);
    fread(recognizer->pW_ah, sizeof(float), PCFC_HD * PCFC_HD, file);
    fread(recognizer->pb_g, sizeof(float), PCFC_HD, file);
    fread(recognizer->pb_a, sizeof(float), PCFC_HD, file);
    fclose(file);
    return 0;
}

/* ======================================================================== */
/*  少样本识别                                                              */
/* ======================================================================== */

struct IRDFewShotRecognizer {
    IRDFewShotConfig cfg;
    float W_embed[32 * 256], b_embed[256];
    float W_gx[256 * 32], W_ax[256 * 32], W_gh[256 * 256], W_ah[256 * 256];
    float b_g[256], b_a[256], tau, dt;
    float pW_gx[PCFC_HD * PCFC_MAX_CH], pW_ax[PCFC_HD * PCFC_MAX_CH];
    float pW_gh[PCFC_HD * PCFC_HD], pW_ah[PCFC_HD * PCFC_HD];
    float pb_g[PCFC_HD], pb_a[PCFC_HD];
    float support_features[IRD_MAX_SUPPORT_SAMPLES * IRD_SEMANTIC_DIM];
    int support_labels[IRD_MAX_SUPPORT_SAMPLES];
    float prototypes[IRD_MAX_PROTOTYPES * IRD_SEMANTIC_DIM];
    int prototype_counts[IRD_MAX_PROTOTYPES];
    float finetune_W[256 * 32], finetune_b[256];
    char class_names[256][64];
    int num_support, num_classes, initialized;
};

IRDFewShotConfig ird_few_shot_get_default_config(void) {
    IRDFewShotConfig c; memset(&c, 0, sizeof(c));
    c.embedding_dim = 256; c.input_dim = 32;
    c.max_support = IRD_MAX_SUPPORT_SAMPLES; c.max_way = IRD_MAX_PROTOTYPES;
    c.cfc_time_constant = 0.1f; c.cfc_delta_t = 0.05f;
    c.finetune_lr = 0.001f; c.finetune_epochs = 5;
    c.distance_metric = 0; return c;
}

IRDFewShotRecognizer* ird_few_shot_create(const IRDFewShotConfig* config) {
    if (!config) return NULL;
    IRDFewShotRecognizer* rec = (IRDFewShotRecognizer*)safe_calloc(1, sizeof(IRDFewShotRecognizer));
    if (!rec) return NULL;
    memcpy(&rec->cfg, config, sizeof(IRDFewShotConfig));
    int hd = config->embedding_dim, id = config->input_dim;
    _xavier_init(rec->W_embed, id, hd);
    memset(rec->b_embed, 0, hd * sizeof(float));
    _xavier_init(rec->W_gx, id, hd); _xavier_init(rec->W_ax, id, hd);
    _xavier_init(rec->W_gh, hd, hd); _xavier_init(rec->W_ah, hd, hd);
    memset(rec->b_g, 0, hd * sizeof(float)); memset(rec->b_a, 0, hd * sizeof(float));
    rec->tau = config->cfc_time_constant; rec->dt = config->cfc_delta_t;
    _xavier_init(rec->pW_gx, PCFC_MAX_CH, PCFC_HD);
    _xavier_init(rec->pW_ax, PCFC_MAX_CH, PCFC_HD);
    _xavier_init(rec->pW_gh, PCFC_HD, PCFC_HD);
    _xavier_init(rec->pW_ah, PCFC_HD, PCFC_HD);
    memset(rec->pb_g, 0, PCFC_HD * sizeof(float));
    memset(rec->pb_a, 0, PCFC_HD * sizeof(float));
    rec->num_support = 0; rec->num_classes = 0; rec->initialized = 0;
    return rec;
}

void ird_few_shot_free(IRDFewShotRecognizer* rec) { safe_free((void**)&rec); }

static int _embed_image_cfc(IRDFewShotRecognizer* rec, const float* img,
                              int w, int h, int ch, float* embedding) {
    if (!rec || !img || !embedding) return -1;
    int hd = rec->cfg.embedding_dim;
    float tau = rec->tau, dt = rec->dt;
    memset(embedding, 0, hd * sizeof(float));
    int ps = 16, st = 8;
    float patch_h[PCFC_HD];
    for (int y = 0; y + ps <= h; y += st) {
        for (int x = 0; x + ps <= w; x += st) {
            memset(patch_h, 0, PCFC_HD * sizeof(float));
            _patch_cfc_encode(img, w, h, ch, x, y, ps, ps,
                              rec->pW_gx, rec->pW_ax, rec->pW_gh, rec->pW_ah,
                              rec->pb_g, rec->pb_a, patch_h, PCFC_HD, tau, dt);
            _cfc_ode_step(patch_h, PCFC_HD, rec->W_gx, rec->W_ax,
                          rec->W_gh, rec->W_ah, rec->b_g, rec->b_a,
                          embedding, hd, tau, dt);
        }
    }
    _vec_normalize(embedding, hd);
    return 0;
}

int ird_few_shot_set_support(IRDFewShotRecognizer* rec, const float* images,
                               const int* labels, int n, int w, int h, int ch) {
    if (!rec || !images || !labels || n <= 0) return -1;
    if (n > rec->cfg.max_support) n = rec->cfg.max_support;
    int hd = rec->cfg.embedding_dim;
    for (int i = 0; i < n; i++) {
        float emb[256];
        if (_embed_image_cfc(rec, &images[i * w * h * ch], w, h, ch, emb) != 0) continue;
        memcpy(&rec->support_features[i * hd], emb, hd * sizeof(float));
        rec->support_labels[i] = labels[i];
    }
    rec->num_support = n;
    int max_lb = -1;
    for (int i = 0; i < n; i++) if (labels[i] > max_lb) max_lb = labels[i];
    rec->num_classes = max_lb + 1;
    ird_few_shot_compute_prototypes(rec);
    rec->initialized = 1;
    return 0;
}

int ird_few_shot_set_support_features(IRDFewShotRecognizer* rec,
                                        const float* features, const int* labels, int n) {
    if (!rec || !features || !labels || n <= 0) return -1;
    if (n > rec->cfg.max_support) n = rec->cfg.max_support;
    int hd = rec->cfg.embedding_dim;
    memcpy(rec->support_features, features, n * hd * sizeof(float));
    memcpy(rec->support_labels, labels, n * sizeof(int));
    rec->num_support = n;
    int max_lb = -1;
    for (int i = 0; i < n; i++) if (labels[i] > max_lb) max_lb = labels[i];
    rec->num_classes = max_lb + 1;
    ird_few_shot_compute_prototypes(rec);
    rec->initialized = 1;
    return 0;
}

int ird_few_shot_compute_prototypes(IRDFewShotRecognizer* rec) {
    if (!rec || rec->num_support <= 0) return -1;
    int hd = rec->cfg.embedding_dim, nc = rec->num_classes;
    memset(rec->prototypes, 0, nc * hd * sizeof(float));
    memset(rec->prototype_counts, 0, nc * sizeof(int));
    for (int i = 0; i < rec->num_support; i++) {
        int lb = rec->support_labels[i];
        if (lb < 0 || lb >= nc) continue;
        for (int j = 0; j < hd; j++)
            rec->prototypes[lb * hd + j] += rec->support_features[i * hd + j];
        rec->prototype_counts[lb]++;
    }
    for (int ci = 0; ci < nc; ci++) {
        if (rec->prototype_counts[ci] > 0) {
            float inv = 1.0f / (float)rec->prototype_counts[ci];
            for (int j = 0; j < hd; j++) rec->prototypes[ci * hd + j] *= inv;
            _vec_normalize(&rec->prototypes[ci * hd], hd);
        }
    }
    return 0;
}

int ird_few_shot_predict(IRDFewShotRecognizer* rec, const float* image,
                           int w, int h, int ch, IRDFewShotPrediction* prediction) {
    if (!rec || !image || !prediction) return -1;
    memset(prediction, 0, sizeof(IRDFewShotPrediction));
    if (!rec->initialized || rec->num_classes <= 0) return -1;
    float emb[256];
    if (_embed_image_cfc(rec, image, w, h, ch, emb) != 0) return -1;
    return ird_few_shot_predict_from_features(rec, emb, prediction);
}

int ird_few_shot_predict_from_features(IRDFewShotRecognizer* rec,
                                         const float* features,
                                         IRDFewShotPrediction* prediction) {
    if (!rec || !features || !prediction) return -1;
    memset(prediction, 0, sizeof(IRDFewShotPrediction));
    if (!rec->initialized || rec->num_classes <= 0) return -1;
    int hd = rec->cfg.embedding_dim, nc = rec->num_classes;
    int dm = rec->cfg.distance_metric;
    float dists[256];
    for (int ci = 0; ci < nc; ci++) {
        dists[ci] = (dm == 0) ? _l2_dist(features, &rec->prototypes[ci * hd], hd)
                              : (1.0f - _cos_sim(features, &rec->prototypes[ci * hd], hd));
    }
    int si[256];
    for (int ci = 0; ci < nc; ci++) si[ci] = ci;
    for (int i = 0; i < nc - 1; i++)
        for (int j = i + 1; j < nc; j++)
            if (dists[si[j]] < dists[si[i]]) { int t = si[i]; si[i] = si[j]; si[j] = t; }

    int best = si[0];
    prediction->class_id = best; prediction->confidence = 1.0f / (1.0f + dists[best]);
    prediction->prototype_distance = dists[best];
    if (best < nc) snprintf(prediction->class_name, 64, "%s", rec->class_names[best]);
    int topk = (nc < prediction->top_k) ? nc : prediction->top_k;
    if (topk > 10) topk = 10;
    for (int ki = 0; ki < topk; ki++) {
        prediction->top_predictions[ki].class_id = si[ki];
        prediction->top_predictions[ki].confidence = 1.0f / (1.0f + dists[si[ki]]);
        if (si[ki] < nc)
            snprintf(prediction->top_predictions[ki].class_name, 64, "%s", rec->class_names[si[ki]]);
    }
    return 0;
}

int ird_few_shot_finetune(IRDFewShotRecognizer* rec, const float* images,
                            const int* labels, int n, int w, int h, int ch) {
    if (!rec || !images || !labels || n <= 0 || !rec->initialized) return -1;
    int hd = rec->cfg.embedding_dim, nc = rec->num_classes;
    float lr = rec->cfg.finetune_lr;
    int ep = rec->cfg.finetune_epochs;
    memcpy(rec->finetune_W, rec->W_embed, hd * rec->cfg.input_dim * sizeof(float));
    memcpy(rec->finetune_b, rec->b_embed, hd * sizeof(float));

    for (int e = 0; e < ep; e++) {
        float total_loss = 0.0f;
        for (int s = 0; s < n; s++) {
            const float* img = &images[s * w * h * ch];
            int lb = labels[s]; if (lb < 0 || lb >= nc) continue;
            float emb[256];
            if (_embed_image_cfc(rec, img, w, h, ch, emb) != 0) continue;

            float probs[256]; float max_p = -1e10f;
            for (int ci = 0; ci < nc; ci++) {
                float sim = _cos_sim(emb, &rec->prototypes[ci * hd], hd);
                probs[ci] = sim; if (sim > max_p) max_p = sim;
            }
            float sum_e = 0.0f;
            for (int ci = 0; ci < nc; ci++) { probs[ci] = expf(probs[ci] - max_p); sum_e += probs[ci]; }
            if (sum_e < 1e-10f) sum_e = 1e-10f;
            for (int ci = 0; ci < nc; ci++) probs[ci] /= sum_e;
            total_loss += -logf(probs[lb] + 1e-10f);

            for (int j = 0; j < hd; j++) {
                float grad = (probs[lb] - 1.0f) * emb[j] * lr;
                for (int k = 0; k < rec->cfg.input_dim; k++)
                    rec->finetune_W[j * rec->cfg.input_dim + k] -= grad * img[k % (w * h * ch)];
                rec->finetune_b[j] -= grad;
            }
        }
        if (total_loss / (float)n < 0.01f) break;
    }
    return 0;
}

int ird_few_shot_episodic_train(IRDFewShotRecognizer* rec,
                                  const IRDFewShotEpisode* episodes, int num_episodes,
                                  int w, int h, int ch) {
    if (!rec || !episodes || num_episodes <= 0) return -1;
    int hd = rec->cfg.embedding_dim;
    for (int epi = 0; epi < num_episodes; epi++) {
        const IRDFewShotEpisode* ep = &episodes[epi];
        if (ep->num_support <= 0) continue;

        int nc = ep->num_classes;
        int ns = ep->num_support;
        float* ep_support = (float*)safe_calloc(ns * hd, sizeof(float));
        if (!ep_support) continue;
        memcpy(ep_support, ep->support_features, ns * hd * sizeof(float));

        ird_few_shot_set_support_features(rec, ep_support, ep->support_labels, ns);
        rec->num_classes = nc;
        rec->initialized = 1;

        float* query_protos = (float*)safe_calloc(nc * hd, sizeof(float));
        if (query_protos) {
            memcpy(query_protos, ep->prototypes, nc * hd * sizeof(float));
            memcpy(rec->prototypes, query_protos, nc * hd * sizeof(float));
            safe_free((void**)&query_protos);
        } else {
            ird_few_shot_compute_prototypes(rec);
        }
        safe_free((void**)&ep_support);
    }
    rec->cfg.finetune_epochs = 1;
    return 0;
}

int ird_few_shot_add_to_support(IRDFewShotRecognizer* rec, const float* image,
                                  int label, int w, int h, int ch) {
    if (!rec || !image || rec->num_support >= rec->cfg.max_support) return -1;
    int hd = rec->cfg.embedding_dim, idx = rec->num_support;
    float emb[256];
    if (_embed_image_cfc(rec, image, w, h, ch, emb) != 0) return -1;
    memcpy(&rec->support_features[idx * hd], emb, hd * sizeof(float));
    rec->support_labels[idx] = label;
    rec->num_support++;
    if (label + 1 > rec->num_classes) rec->num_classes = label + 1;
    ird_few_shot_compute_prototypes(rec);
    rec->initialized = 1;
    return 0;
}

int ird_few_shot_clear_support(IRDFewShotRecognizer* rec) {
    if (!rec) return -1;
    int hd = rec->cfg.embedding_dim;
    memset(rec->support_features, 0, rec->cfg.max_support * hd * sizeof(float));
    memset(rec->support_labels, 0, rec->cfg.max_support * sizeof(int));
    memset(rec->prototypes, 0, rec->cfg.max_way * hd * sizeof(float));
    memset(rec->prototype_counts, 0, rec->cfg.max_way * sizeof(int));
    rec->num_support = 0; rec->num_classes = 0; rec->initialized = 0;
    return 0;
}

int ird_few_shot_update_prototype(IRDFewShotRecognizer* rec, int label,
                                    const float* new_prototype) {
    if (!rec || !new_prototype || label < 0 || label >= rec->cfg.max_way) return -1;
    int hd = rec->cfg.embedding_dim;
    memcpy(&rec->prototypes[label * hd], new_prototype, hd * sizeof(float));
    _vec_normalize(&rec->prototypes[label * hd], hd);
    if (label + 1 > rec->num_classes) rec->num_classes = label + 1;
    rec->initialized = 1;
    return 0;
}

int ird_few_shot_save(const IRDFewShotRecognizer* rec, const char* path) {
    if (!rec || !path) return -1;
    FILE* f = fopen(path, "wb"); if (!f) return -1;
    fwrite(&rec->cfg, sizeof(IRDFewShotConfig), 1, f);
    fwrite(rec->W_embed, sizeof(float), 32 * 256, f);
    fwrite(rec->b_embed, sizeof(float), 256, f);
    fwrite(rec->W_gx, sizeof(float), 256 * 32, f);
    fwrite(rec->W_ax, sizeof(float), 256 * 32, f);
    fwrite(rec->W_gh, sizeof(float), 256 * 256, f);
    fwrite(rec->W_ah, sizeof(float), 256 * 256, f);
    fwrite(rec->b_g, sizeof(float), 256, f);
    fwrite(rec->b_a, sizeof(float), 256, f);
    fwrite(rec->pW_gx, sizeof(float), PCFC_HD * PCFC_MAX_CH, f);
    fwrite(rec->pW_ax, sizeof(float), PCFC_HD * PCFC_MAX_CH, f);
    fwrite(rec->pW_gh, sizeof(float), PCFC_HD * PCFC_HD, f);
    fwrite(rec->pW_ah, sizeof(float), PCFC_HD * PCFC_HD, f);
    fwrite(rec->pb_g, sizeof(float), PCFC_HD, f);
    fwrite(rec->pb_a, sizeof(float), PCFC_HD, f);
    fclose(f); return 0;
}

int ird_few_shot_load(IRDFewShotRecognizer* rec, const char* path) {
    if (!rec || !path) return -1;
    FILE* f = fopen(path, "rb"); if (!f) return -1;
    fread(&rec->cfg, sizeof(IRDFewShotConfig), 1, f);
    fread(rec->W_embed, sizeof(float), 32 * 256, f);
    fread(rec->b_embed, sizeof(float), 256, f);
    fread(rec->W_gx, sizeof(float), 256 * 32, f);
    fread(rec->W_ax, sizeof(float), 256 * 32, f);
    fread(rec->W_gh, sizeof(float), 256 * 256, f);
    fread(rec->W_ah, sizeof(float), 256 * 256, f);
    fread(rec->b_g, sizeof(float), 256, f);
    fread(rec->b_a, sizeof(float), 256, f);
    fread(rec->pW_gx, sizeof(float), PCFC_HD * PCFC_MAX_CH, f);
    fread(rec->pW_ax, sizeof(float), PCFC_HD * PCFC_MAX_CH, f);
    fread(rec->pW_gh, sizeof(float), PCFC_HD * PCFC_HD, f);
    fread(rec->pW_ah, sizeof(float), PCFC_HD * PCFC_HD, f);
    fread(rec->pb_g, sizeof(float), PCFC_HD, f);
    fread(rec->pb_a, sizeof(float), PCFC_HD, f);
    rec->tau = rec->cfg.cfc_time_constant; rec->dt = rec->cfg.cfc_delta_t;
    rec->num_support = 0; rec->num_classes = 0; rec->initialized = 1;
    fclose(f); return 0;
}

/* ======================================================================== */
/*  零样本识别                                                              */
/* ======================================================================== */

struct IRDZeroShotRecognizer {
    IRDZeroShotConfig cfg;
    float W_vis_sem[256 * 256], b_vis_sem[256];
    float W_attr_pred[128 * 256], b_attr_pred[128];
    float class_attributes[256 * 128], semantic_prototypes[256 * 256];
    float W_gx[256 * 32], W_ax[256 * 32], W_gh[256 * 256], W_ah[256 * 256];
    float b_g[256], b_a[256], hidden[256];
    float margin;
    int num_seen_classes, num_unseen_classes, num_total_classes;
    float pW_gx[PCFC_HD * PCFC_MAX_CH], pW_ax[PCFC_HD * PCFC_MAX_CH];
    float pW_gh[PCFC_HD * PCFC_HD], pW_ah[PCFC_HD * PCFC_HD];
    float pb_g[PCFC_HD], pb_a[PCFC_HD];
};

IRDZeroShotConfig ird_zero_shot_get_default_config(void) {
    IRDZeroShotConfig c; memset(&c, 0, sizeof(c));
    c.feature_dim = 256; c.input_dim = 32; c.semantic_dim = 256; c.attribute_dim = 128;
    c.cfc_time_constant = 0.1f; c.cfc_delta_t = 0.05f;
    c.margin = 0.1f; c.learning_rate = 0.001f;
    c.max_seen_classes = 256; c.max_unseen_classes = 256;
    return c;
}

IRDZeroShotRecognizer* ird_zero_shot_create(const IRDZeroShotConfig* config) {
    if (!config) return NULL;
    IRDZeroShotRecognizer* rec = (IRDZeroShotRecognizer*)safe_calloc(1, sizeof(IRDZeroShotRecognizer));
    if (!rec) return NULL;
    memcpy(&rec->cfg, config, sizeof(IRDZeroShotConfig));
    int hd = config->feature_dim, sd = config->semantic_dim, ad = config->attribute_dim;
    _xavier_init(rec->W_vis_sem, hd, sd);
    memset(rec->b_vis_sem, 0, sd * sizeof(float));
    _xavier_init(rec->W_attr_pred, hd, ad);
    memset(rec->b_attr_pred, 0, ad * sizeof(float));
    _xavier_init(rec->W_gx, config->input_dim, hd);
    _xavier_init(rec->W_ax, config->input_dim, hd);
    _xavier_init(rec->W_gh, hd, hd);
    _xavier_init(rec->W_ah, hd, hd);
    memset(rec->b_g, 0, hd * sizeof(float));
    memset(rec->b_a, 0, hd * sizeof(float));
    memset(rec->hidden, 0, hd * sizeof(float));
    _xavier_init(rec->pW_gx, PCFC_MAX_CH, PCFC_HD);
    _xavier_init(rec->pW_ax, PCFC_MAX_CH, PCFC_HD);
    _xavier_init(rec->pW_gh, PCFC_HD, PCFC_HD);
    _xavier_init(rec->pW_ah, PCFC_HD, PCFC_HD);
    memset(rec->pb_g, 0, PCFC_HD * sizeof(float));
    memset(rec->pb_a, 0, PCFC_HD * sizeof(float));
    rec->margin = config->margin;
    rec->num_seen_classes = 0; rec->num_unseen_classes = 0; rec->num_total_classes = 0;
    return rec;
}

void ird_zero_shot_free(IRDZeroShotRecognizer* rec) { safe_free((void**)&rec); }

static void _map_visual_to_semantic(const float* vis, float* sem_out, int hd, int sd,
                                      const float* W, const float* b) {
    _mat_vec_mul(W, vis, sem_out, sd, hd);
    _vec_add(sem_out, b, sd);
    _vec_normalize(sem_out, sd);
}

static void _predict_attributes(const float* vis, float* attrs, int hd, int ad,
                                  const float* W, const float* b) {
    _mat_vec_mul(W, vis, attrs, ad, hd);
    _vec_add(attrs, b, ad);
    for (int i = 0; i < ad; i++) attrs[i] = _sig(attrs[i]);
}

int ird_zero_shot_set_class_attributes(IRDZeroShotRecognizer* rec, const float* attributes,
                                         int num_classes, int class_offset) {
    if (!rec || !attributes || num_classes <= 0) return -1;
    int ad = rec->cfg.attribute_dim, start = class_offset;
    if (start + num_classes > 256) return -1;
    for (int ci = 0; ci < num_classes; ci++)
        memcpy(&rec->class_attributes[(start + ci) * ad], &attributes[ci * ad], ad * sizeof(float));
    rec->num_seen_classes = start + num_classes;
    rec->num_total_classes = rec->num_seen_classes + rec->num_unseen_classes;
    return 0;
}

int ird_zero_shot_set_semantic_prototypes(IRDZeroShotRecognizer* rec, const float* prototypes,
                                            int num_classes, int class_offset) {
    if (!rec || !prototypes || num_classes <= 0) return -1;
    int sd = rec->cfg.semantic_dim;
    for (int ci = 0; ci < num_classes; ci++)
        memcpy(&rec->semantic_prototypes[(class_offset + ci) * sd],
               &prototypes[ci * sd], sd * sizeof(float));
    return 0;
}

int ird_zero_shot_learn_mapping(IRDZeroShotRecognizer* rec, const float* visual_features,
                                  const float* semantic_features, int num_samples) {
    if (!rec || !visual_features || !semantic_features || num_samples <= 0) return -1;
    int hd = rec->cfg.feature_dim, sd = rec->cfg.semantic_dim;
    float lr = rec->cfg.learning_rate, margin = rec->margin;
    float loss = 0.0f;
    for (int s = 0; s < num_samples; s++) {
        const float* vis = &visual_features[s * hd];
        const float* sem_pos = &semantic_features[s * sd];
        float pred_sem[256];
        _map_visual_to_semantic(vis, pred_sem, hd, sd, rec->W_vis_sem, rec->b_vis_sem);
        float pos_sim = _cos_sim(pred_sem, sem_pos, sd);
        int neg_idx = (s + 1) % num_samples;
        const float* sem_neg = &semantic_features[neg_idx * sd];
        float neg_sim = _cos_sim(pred_sem, sem_neg, sd);
        float hinge = margin - pos_sim + neg_sim;
        if (hinge > 0.0f) {
            loss += hinge;
            float grad_scale = lr * 1.0f;
            for (int i = 0; i < sd; i++) {
                for (int j = 0; j < hd; j++) {
                    float g = grad_scale * (sem_pos[i] - sem_neg[i]) * vis[j];
                    rec->W_vis_sem[i * hd + j] -= g;
                }
                rec->b_vis_sem[i] -= grad_scale * (sem_pos[i] - sem_neg[i]);
            }
        }
    }
    return (int)(loss * 1000.0f);
}

int ird_zero_shot_add_seen_class(IRDZeroShotRecognizer* rec, int class_id,
                                   const float* semantic_prototype) {
    if (!rec || !semantic_prototype || class_id < 0 || class_id >= 256) return -1;
    memcpy(&rec->semantic_prototypes[class_id * rec->cfg.semantic_dim],
           semantic_prototype, rec->cfg.semantic_dim * sizeof(float));
    if (class_id >= rec->num_seen_classes) rec->num_seen_classes = class_id + 1;
    rec->num_total_classes = rec->num_seen_classes + rec->num_unseen_classes;
    return 0;
}

int ird_zero_shot_add_unseen_class(IRDZeroShotRecognizer* rec, int class_id,
                                     const float* semantic_prototype) {
    if (!rec || !semantic_prototype || class_id < 0 || class_id >= 256) return -1;
    memcpy(&rec->semantic_prototypes[class_id * rec->cfg.semantic_dim],
           semantic_prototype, rec->cfg.semantic_dim * sizeof(float));
    int unseen_slot = class_id - rec->num_seen_classes;
    if (unseen_slot >= rec->num_unseen_classes) rec->num_unseen_classes = unseen_slot + 1;
    rec->num_total_classes = rec->num_seen_classes + rec->num_unseen_classes;
    return 0;
}

int ird_zero_shot_predict_from_features(IRDZeroShotRecognizer* rec, const float* features,
                                          IRDZeroShotPrediction* pred) {
    if (!rec || !features || !pred) return -1;
    memset(pred, 0, sizeof(IRDZeroShotPrediction));
    int hd = rec->cfg.feature_dim, sd = rec->cfg.semantic_dim, ad = rec->cfg.attribute_dim;
    int nc = rec->num_total_classes; if (nc <= 0) return -1;
    float pred_sem[256];
    _map_visual_to_semantic(features, pred_sem, hd, sd, rec->W_vis_sem, rec->b_vis_sem);
    memcpy(pred->semantic_embedding, pred_sem, sd * sizeof(float));
    float attrs[128];
    _predict_attributes(features, attrs, hd, ad, rec->W_attr_pred, rec->b_attr_pred);
    memcpy(pred->attribute_prediction, attrs, ad * sizeof(float));

    float sims[256];
    for (int ci = 0; ci < nc; ci++) {
        sims[ci] = _cos_sim(pred_sem, &rec->semantic_prototypes[ci * sd], sd);
    }
    int best = 0; float bs = sims[0];
    for (int ci = 1; ci < nc; ci++) { if (sims[ci] > bs) { bs = sims[ci]; best = ci; } }
    pred->class_id = best; pred->confidence = bs;
    /* 零样本类别名称：使用预定义中文类别表 */
    static const char* default_zs_categories[] = {
        "物体","动物","人","车辆","建筑","植物","食物","工具","电子设备","家具",
        "服装","书籍","乐器","运动器材","厨房用具","办公用品","玩具","艺术品","自然景观","城市景观",
        "室内场景","室外场景","文字","图表","人脸","手写","标志","条形码","二维码","医疗影像",
        "卫星图像","显微镜图像","红外图像","X光图像","超声波","雷达图像","热成像","夜视","水下","航空"
    };
    int ncats = (int)(sizeof(default_zs_categories)/sizeof(default_zs_categories[0]));
    if (best < ncats) {
        snprintf(pred->class_name, 64, "%s", default_zs_categories[best]);
    } else {
        snprintf(pred->class_name, 64, "类别_%d", best);
    }

    int topk = (nc < 10) ? nc : 10;
    int si[256];
    for (int ci = 0; ci < nc; ci++) si[ci] = ci;
    for (int i = 0; i < nc - 1; i++)
        for (int j = i + 1; j < nc; j++)
            if (sims[si[j]] > sims[si[i]]) { int t = si[i]; si[i] = si[j]; si[j] = t; }
    pred->top_k = topk;
    for (int ki = 0; ki < topk; ki++) {
        pred->top_predictions[ki].class_id = si[ki];
        pred->top_predictions[ki].similarity = sims[si[ki]];
        if (si[ki] < ncats) {
            snprintf(pred->top_predictions[ki].class_name, 64, "%s", default_zs_categories[si[ki]]);
        } else {
            snprintf(pred->top_predictions[ki].class_name, 64, "类别_%d", si[ki]);
        }
    }
    return 0;
}

int ird_zero_shot_predict(IRDZeroShotRecognizer* rec, const float* img,
                            int w, int h, int ch, IRDZeroShotPrediction* pred) {
    if (!rec || !img || !pred) return -1;
    int hd = rec->cfg.feature_dim;
    float tau = rec->cfg.cfc_time_constant, dt = rec->cfg.cfc_delta_t;
    memset(rec->hidden, 0, hd * sizeof(float));
    int ps = 16, st = 8;
    float patch_h[PCFC_HD];
    for (int y = 0; y + ps <= h; y += st) {
        for (int x = 0; x + ps <= w; x += st) {
            memset(patch_h, 0, PCFC_HD * sizeof(float));
            _patch_cfc_encode(img, w, h, ch, x, y, ps, ps,
                              rec->pW_gx, rec->pW_ax, rec->pW_gh, rec->pW_ah,
                              rec->pb_g, rec->pb_a, patch_h, PCFC_HD, tau, dt);
            _cfc_ode_step(patch_h, PCFC_HD, rec->W_gx, rec->W_ax, rec->W_gh, rec->W_ah,
                          rec->b_g, rec->b_a, rec->hidden, hd, tau, dt);
        }
    }
    return ird_zero_shot_predict_from_features(rec, rec->hidden, pred);
}

int ird_zero_shot_save(const IRDZeroShotRecognizer* rec, const char* path) {
    if (!rec || !path) return -1;
    FILE* f = fopen(path, "wb"); if (!f) return -1;
    fwrite(&rec->cfg, sizeof(IRDZeroShotConfig), 1, f);
    fwrite(rec->W_vis_sem, sizeof(float), 256 * 256, f);
    fwrite(rec->b_vis_sem, sizeof(float), 256, f);
    fwrite(rec->class_attributes, sizeof(float), 256 * 128, f);
    fwrite(rec->semantic_prototypes, sizeof(float), 256 * 256, f);
    fwrite(rec->W_gx, sizeof(float), 256 * 32, f);
    fwrite(rec->W_ax, sizeof(float), 256 * 32, f);
    fwrite(rec->W_gh, sizeof(float), 256 * 256, f);
    fwrite(rec->W_ah, sizeof(float), 256 * 256, f);
    fwrite(rec->b_g, sizeof(float), 256, f);
    fwrite(rec->b_a, sizeof(float), 256, f);
    fwrite(rec->pW_gx, sizeof(float), PCFC_HD * PCFC_MAX_CH, f);
    fwrite(rec->pW_ax, sizeof(float), PCFC_HD * PCFC_MAX_CH, f);
    fwrite(rec->pW_gh, sizeof(float), PCFC_HD * PCFC_HD, f);
    fwrite(rec->pW_ah, sizeof(float), PCFC_HD * PCFC_HD, f);
    fwrite(rec->pb_g, sizeof(float), PCFC_HD, f);
    fwrite(rec->pb_a, sizeof(float), PCFC_HD, f);
    fclose(f); return 0;
}

int ird_zero_shot_load(IRDZeroShotRecognizer* rec, const char* path) {
    if (!rec || !path) return -1;
    FILE* f = fopen(path, "rb"); if (!f) return -1;
    fread(&rec->cfg, sizeof(IRDZeroShotConfig), 1, f);
    fread(rec->W_vis_sem, sizeof(float), 256 * 256, f);
    fread(rec->b_vis_sem, sizeof(float), 256, f);
    fread(rec->class_attributes, sizeof(float), 256 * 128, f);
    fread(rec->semantic_prototypes, sizeof(float), 256 * 256, f);
    fread(rec->W_gx, sizeof(float), 256 * 32, f);
    fread(rec->W_ax, sizeof(float), 256 * 32, f);
    fread(rec->W_gh, sizeof(float), 256 * 256, f);
    fread(rec->W_ah, sizeof(float), 256 * 256, f);
    fread(rec->b_g, sizeof(float), 256, f);
    fread(rec->b_a, sizeof(float), 256, f);
    fread(rec->pW_gx, sizeof(float), PCFC_HD * PCFC_MAX_CH, f);
    fread(rec->pW_ax, sizeof(float), PCFC_HD * PCFC_MAX_CH, f);
    fread(rec->pW_gh, sizeof(float), PCFC_HD * PCFC_HD, f);
    fread(rec->pW_ah, sizeof(float), PCFC_HD * PCFC_HD, f);
    fread(rec->pb_g, sizeof(float), PCFC_HD, f);
    fread(rec->pb_a, sizeof(float), PCFC_HD, f);
    rec->num_seen_classes = rec->cfg.max_seen_classes;
    rec->num_unseen_classes = rec->cfg.max_unseen_classes;
    rec->num_total_classes = rec->num_seen_classes + rec->num_unseen_classes;
    rec->margin = rec->cfg.margin;
    fclose(f); return 0;
}

/* ======================================================================== */
/*  深度识别管理器                                                          */
/* ======================================================================== */

struct IRDDeepManager {
    IRDDeepManagerConfig cfg;
    IRDFineClassifier* fine_classifier;
    IRDOpenSetRecognizer* open_set_recognizer;
    IRDZeroShotRecognizer* zero_shot_recognizer;
    IRDFewShotRecognizer* few_shot_recognizer;
    int current_mode;
};

IRDDeepManagerConfig ird_deep_manager_get_default_config(void) {
    IRDDeepManagerConfig c; memset(&c, 0, sizeof(c));
    c.default_image_width = 224; c.default_image_height = 224;
    c.default_channels = 3; c.max_batch_size = 16;
    c.fine_grain_enabled = 1; c.open_set_enabled = 1;
    c.zero_shot_enabled = 1; c.few_shot_enabled = 1;
    c.default_recognition_mode = IRD_MODE_FINE_GRAINED;
    return c;
}

IRDDeepManager* ird_deep_manager_create(const IRDDeepManagerConfig* config) {
    if (!config) return NULL;
    IRDDeepManager* m = (IRDDeepManager*)safe_calloc(1, sizeof(IRDDeepManager));
    if (!m) return NULL;
    memcpy(&m->cfg, config, sizeof(IRDDeepManagerConfig));
    m->current_mode = config->default_recognition_mode;
    if (config->fine_grain_enabled) {
        IRDFineConfig fc = ird_fine_get_default_config();
        fc.input_dim = 32; fc.feature_dim = 256;
        m->fine_classifier = ird_fine_create(&fc);
    }
    if (config->open_set_enabled) {
        IRDOpenSetConfig oc = ird_open_set_get_default_config();
        oc.feature_dim = 256; oc.input_dim = 32;
        m->open_set_recognizer = ird_open_set_create(&oc);
    }
    if (config->zero_shot_enabled) {
        IRDZeroShotConfig zc = ird_zero_shot_get_default_config();
        zc.feature_dim = 256; zc.input_dim = 32;
        m->zero_shot_recognizer = ird_zero_shot_create(&zc);
    }
    if (config->few_shot_enabled) {
        IRDFewShotConfig fsc = ird_few_shot_get_default_config();
        fsc.embedding_dim = 256; fsc.input_dim = 32;
        m->few_shot_recognizer = ird_few_shot_create(&fsc);
    }
    return m;
}

void ird_deep_manager_free(IRDDeepManager* manager) {
    if (!manager) return;
    if (manager->fine_classifier) ird_fine_free(manager->fine_classifier);
    if (manager->open_set_recognizer) ird_open_set_free(manager->open_set_recognizer);
    if (manager->zero_shot_recognizer) ird_zero_shot_free(manager->zero_shot_recognizer);
    if (manager->few_shot_recognizer) ird_few_shot_free(manager->few_shot_recognizer);
    safe_free((void**)&manager);
}

int ird_deep_manager_recognize(IRDDeepManager* manager, const float* image,
                                 int w, int h, int ch, IRDDeepRecognitionResult* result) {
    if (!manager || !image || !result) return -1;
    memset(result, 0, sizeof(IRDDeepRecognitionResult));
    result->primary_mode = manager->current_mode;

    switch (manager->current_mode) {
    case IRD_MODE_FINE_GRAINED:
        if (manager->fine_classifier) {
            result->has_fine_result = 1;
            ird_fine_classify(manager->fine_classifier, image, w, h, ch, &result->fine_result);
            result->best_class_id = result->fine_result.fine_category_id;
            result->primary_confidence = result->fine_result.fine_confidence;
            result->is_unknown = 0;
        }
        break;
    case IRD_MODE_OPEN_SET:
        if (manager->open_set_recognizer) {
            result->has_open_set_result = 1;
            IRDOpenSetPrediction* osp = &result->open_set_result;
            ird_open_set_predict_image(manager->open_set_recognizer, image, w, h, ch, osp);
            result->best_class_id = osp->assigned_class;
            result->primary_confidence = osp->confidence;
            result->is_unknown = osp->is_unknown;
        }
        break;
    case IRD_MODE_ZERO_SHOT:
        if (manager->zero_shot_recognizer) {
            result->has_zero_shot_result = 1;
            ird_zero_shot_predict(manager->zero_shot_recognizer, image, w, h, ch, &result->zero_shot_result);
            result->best_class_id = result->zero_shot_result.class_id;
            result->primary_confidence = result->zero_shot_result.confidence;
            result->is_unknown = 0;
        }
        break;
    case IRD_MODE_FEW_SHOT:
        if (manager->few_shot_recognizer) {
            result->has_few_shot_result = 1;
            ird_few_shot_predict(manager->few_shot_recognizer, image, w, h, ch, &result->few_shot_result);
            result->best_class_id = result->few_shot_result.class_id;
            result->primary_confidence = result->few_shot_result.confidence;
            result->is_unknown = 0;
        }
        break;
    default:
        return -1;
    }
    return 0;
}

int ird_deep_manager_set_mode(IRDDeepManager* manager, int mode) {
    if (!manager) return -1;
    if (mode < IRD_MODE_FINE_GRAINED || mode > IRD_MODE_FEW_SHOT) return -1;
    manager->current_mode = mode;
    return 0;
}

/* ======================================================================== */
/*  预训练权重加载与保存（统一二进制格式）                                     */
/* ======================================================================== */

/* 统一权重文件魔数：SELFIRD3 = SELF IRD v3 */
#define IRD_WEIGHTS_MAGIC "SELFIRD3"

/* 保存深度识别管理器所有子模型权重到单个二进制文件
 * 文件格式：[魔数8字节][管理器配置][子模型存在标记4×int][逐个子模型权重数据]
 * 返回0成功，-1失败 */
int ird_save_model_weights(const IRDDeepManager* manager, const char* path) {
    if (!manager || !path) return -1;
    FILE* f = fopen(path, "wb");
    if (!f) return -1;

    /* 写入魔数标识 */
    const char magic[9] = IRD_WEIGHTS_MAGIC;
    fwrite(magic, 1, 8, f);

    /* 写入管理器配置 */
    fwrite(&manager->cfg, sizeof(IRDDeepManagerConfig), 1, f);

    /* 写入各子模型存在标记 */
    int flags[4];
    flags[0] = (manager->fine_classifier != NULL) ? 1 : 0;
    flags[1] = (manager->open_set_recognizer != NULL) ? 1 : 0;
    flags[2] = (manager->zero_shot_recognizer != NULL) ? 1 : 0;
    flags[3] = (manager->few_shot_recognizer != NULL) ? 1 : 0;
    fwrite(flags, sizeof(int), 4, f);

    /* 写入当前识别模式 */
    fwrite(&manager->current_mode, sizeof(int), 1, f);

    /* ---- 细粒度分类器权重 ---- */
    if (flags[0]) {
        const IRDFineClassifier* c = manager->fine_classifier;
        fwrite(&c->cfg, sizeof(IRDFineConfig), 1, f);
        fwrite(&c->training_completed, sizeof(int), 1, f);
        fwrite(c->W_gx_p, sizeof(float), 256 * 32, f);
        fwrite(c->W_ax_p, sizeof(float), 256 * 32, f);
        fwrite(c->W_gh_p, sizeof(float), 256 * 256, f);
        fwrite(c->W_ah_p, sizeof(float), 256 * 256, f);
        fwrite(c->b_g_p, sizeof(float), 256, f);
        fwrite(c->b_a_p, sizeof(float), 256, f);
        fwrite(c->W_gx_part, sizeof(float), 256 * 256, f);
        fwrite(c->W_ax_part, sizeof(float), 256 * 256, f);
        fwrite(c->W_gh_part, sizeof(float), 256 * 256, f);
        fwrite(c->W_ah_part, sizeof(float), 256 * 256, f);
        fwrite(c->b_g_part, sizeof(float), 256, f);
        fwrite(c->b_a_part, sizeof(float), 256, f);
        fwrite(c->W_fine, sizeof(float), 256 * 256, f);
        fwrite(c->b_fine, sizeof(float), 256, f);
        fwrite(c->W_coarse, sizeof(float), 64 * 256, f);
        fwrite(c->b_coarse, sizeof(float), 64, f);
        fwrite(c->bilin_W, sizeof(float), 256 * 256, f);
        fwrite(c->bilin_proj, sizeof(float), 256 * 2, f);
        fwrite(c->pW_gx, sizeof(float), PCFC_HD * PCFC_MAX_CH, f);
        fwrite(c->pW_ax, sizeof(float), PCFC_HD * PCFC_MAX_CH, f);
        fwrite(c->pW_gh, sizeof(float), PCFC_HD * PCFC_HD, f);
        fwrite(c->pW_ah, sizeof(float), PCFC_HD * PCFC_HD, f);
        fwrite(c->pb_g, sizeof(float), PCFC_HD, f);
        fwrite(c->pb_a, sizeof(float), PCFC_HD, f);
    }

    /* ---- 开放集识别器权重 ---- */
    if (flags[1]) {
        const IRDOpenSetRecognizer* r = manager->open_set_recognizer;
        fwrite(&r->cfg, sizeof(IRDOpenSetConfig), 1, f);
        fwrite(&r->inv_temp, sizeof(float), 1, f);
        fwrite(&r->reject_threshold, sizeof(float), 1, f);
        fwrite(&r->num_known_classes, sizeof(int), 1, f);
        fwrite(&r->total_samples, sizeof(int), 1, f);
        fwrite(&r->weibull_thresh, sizeof(float), 1, f);
        fwrite(&r->nndr_threshold, sizeof(float), 1, f);
        fwrite(r->prototypes, sizeof(float), 256 * 256, f);
        fwrite(r->W_gx, sizeof(float), 256 * 32, f);
        fwrite(r->W_ax, sizeof(float), 256 * 32, f);
        fwrite(r->W_gh, sizeof(float), 256 * 256, f);
        fwrite(r->W_ah, sizeof(float), 256 * 256, f);
        fwrite(r->b_g, sizeof(float), 256, f);
        fwrite(r->b_a, sizeof(float), 256, f);
        fwrite(r->hidden, sizeof(float), 256, f);
        fwrite(r->weibull_k, sizeof(float), 256, f);
        fwrite(r->weibull_lambda, sizeof(float), 256, f);
        fwrite(r->pW_gx, sizeof(float), PCFC_HD * PCFC_MAX_CH, f);
        fwrite(r->pW_ax, sizeof(float), PCFC_HD * PCFC_MAX_CH, f);
        fwrite(r->pW_gh, sizeof(float), PCFC_HD * PCFC_HD, f);
        fwrite(r->pW_ah, sizeof(float), PCFC_HD * PCFC_HD, f);
        fwrite(r->pb_g, sizeof(float), PCFC_HD, f);
        fwrite(r->pb_a, sizeof(float), PCFC_HD, f);
    }

    /* ---- 零样本识别器权重 ---- */
    if (flags[2]) {
        const IRDZeroShotRecognizer* z = manager->zero_shot_recognizer;
        fwrite(&z->cfg, sizeof(IRDZeroShotConfig), 1, f);
        fwrite(&z->margin, sizeof(float), 1, f);
        fwrite(&z->num_seen_classes, sizeof(int), 1, f);
        fwrite(&z->num_unseen_classes, sizeof(int), 1, f);
        fwrite(&z->num_total_classes, sizeof(int), 1, f);
        fwrite(z->W_vis_sem, sizeof(float), 256 * 256, f);
        fwrite(z->b_vis_sem, sizeof(float), 256, f);
        fwrite(z->W_attr_pred, sizeof(float), 128 * 256, f);
        fwrite(z->b_attr_pred, sizeof(float), 128, f);
        fwrite(z->class_attributes, sizeof(float), 256 * 128, f);
        fwrite(z->semantic_prototypes, sizeof(float), 256 * 256, f);
        fwrite(z->W_gx, sizeof(float), 256 * 32, f);
        fwrite(z->W_ax, sizeof(float), 256 * 32, f);
        fwrite(z->W_gh, sizeof(float), 256 * 256, f);
        fwrite(z->W_ah, sizeof(float), 256 * 256, f);
        fwrite(z->b_g, sizeof(float), 256, f);
        fwrite(z->b_a, sizeof(float), 256, f);
        fwrite(z->hidden, sizeof(float), 256, f);
        fwrite(z->pW_gx, sizeof(float), PCFC_HD * PCFC_MAX_CH, f);
        fwrite(z->pW_ax, sizeof(float), PCFC_HD * PCFC_MAX_CH, f);
        fwrite(z->pW_gh, sizeof(float), PCFC_HD * PCFC_HD, f);
        fwrite(z->pW_ah, sizeof(float), PCFC_HD * PCFC_HD, f);
        fwrite(z->pb_g, sizeof(float), PCFC_HD, f);
        fwrite(z->pb_a, sizeof(float), PCFC_HD, f);
    }

    /* ---- 少样本识别器权重 ---- */
    if (flags[3]) {
        const IRDFewShotRecognizer* fs = manager->few_shot_recognizer;
        fwrite(&fs->cfg, sizeof(IRDFewShotConfig), 1, f);
        fwrite(&fs->tau, sizeof(float), 1, f);
        fwrite(&fs->dt, sizeof(float), 1, f);
        fwrite(&fs->num_support, sizeof(int), 1, f);
        fwrite(&fs->num_classes, sizeof(int), 1, f);
        fwrite(&fs->initialized, sizeof(int), 1, f);
        fwrite(fs->W_embed, sizeof(float), 32 * 256, f);
        fwrite(fs->b_embed, sizeof(float), 256, f);
        fwrite(fs->W_gx, sizeof(float), 256 * 32, f);
        fwrite(fs->W_ax, sizeof(float), 256 * 32, f);
        fwrite(fs->W_gh, sizeof(float), 256 * 256, f);
        fwrite(fs->W_ah, sizeof(float), 256 * 256, f);
        fwrite(fs->b_g, sizeof(float), 256, f);
        fwrite(fs->b_a, sizeof(float), 256, f);
        fwrite(fs->pW_gx, sizeof(float), PCFC_HD * PCFC_MAX_CH, f);
        fwrite(fs->pW_ax, sizeof(float), PCFC_HD * PCFC_MAX_CH, f);
        fwrite(fs->pW_gh, sizeof(float), PCFC_HD * PCFC_HD, f);
        fwrite(fs->pW_ah, sizeof(float), PCFC_HD * PCFC_HD, f);
        fwrite(fs->pb_g, sizeof(float), PCFC_HD, f);
        fwrite(fs->pb_a, sizeof(float), PCFC_HD, f);
        fwrite(fs->support_features, sizeof(float), IRD_MAX_SUPPORT_SAMPLES * IRD_SEMANTIC_DIM, f);
        fwrite(fs->support_labels, sizeof(int), IRD_MAX_SUPPORT_SAMPLES, f);
        fwrite(fs->prototypes, sizeof(float), IRD_MAX_PROTOTYPES * IRD_SEMANTIC_DIM, f);
        fwrite(fs->prototype_counts, sizeof(int), IRD_MAX_PROTOTYPES, f);
        fwrite(fs->finetune_W, sizeof(float), 256 * 32, f);
        fwrite(fs->finetune_b, sizeof(float), 256, f);
        fwrite(fs->class_names, sizeof(char), 256 * 64, f);
    }

    fclose(f);
    return 0;
}

/* 从二进制文件加载预训练权重到深度识别管理器所有子模型
 * 文件格式与 ird_save_model_weights 严格对应
 * 加载时会覆盖现有权重，用于恢复训练状态或部署预训练模型
 * 返回0成功，-1失败 */
int ird_load_pretrained_weights(IRDDeepManager* manager, const char* path) {
    if (!manager || !path) return -1;
    FILE* f = fopen(path, "rb");
    if (!f) return -1;

    /* 验证魔数标识 */
    char magic[8];
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, IRD_WEIGHTS_MAGIC, 8) != 0) {
        fclose(f);
        return -1;
    }

    /* 读取管理器配置 */
    IRDDeepManagerConfig loaded_cfg;
    fread(&loaded_cfg, sizeof(IRDDeepManagerConfig), 1, f);
    memcpy(&manager->cfg, &loaded_cfg, sizeof(IRDDeepManagerConfig));

    /* 读取子模型存在标记 */
    int flags[4];
    fread(flags, sizeof(int), 4, f);

    /* 读取当前识别模式 */
    fread(&manager->current_mode, sizeof(int), 1, f);

    /* ---- 细粒度分类器权重 ---- */
    if (flags[0] && manager->fine_classifier) {
        IRDFineClassifier* c = manager->fine_classifier;
        fread(&c->cfg, sizeof(IRDFineConfig), 1, f);
        fread(&c->training_completed, sizeof(int), 1, f);
        fread(c->W_gx_p, sizeof(float), 256 * 32, f);
        fread(c->W_ax_p, sizeof(float), 256 * 32, f);
        fread(c->W_gh_p, sizeof(float), 256 * 256, f);
        fread(c->W_ah_p, sizeof(float), 256 * 256, f);
        fread(c->b_g_p, sizeof(float), 256, f);
        fread(c->b_a_p, sizeof(float), 256, f);
        fread(c->W_gx_part, sizeof(float), 256 * 256, f);
        fread(c->W_ax_part, sizeof(float), 256 * 256, f);
        fread(c->W_gh_part, sizeof(float), 256 * 256, f);
        fread(c->W_ah_part, sizeof(float), 256 * 256, f);
        fread(c->b_g_part, sizeof(float), 256, f);
        fread(c->b_a_part, sizeof(float), 256, f);
        fread(c->W_fine, sizeof(float), 256 * 256, f);
        fread(c->b_fine, sizeof(float), 256, f);
        fread(c->W_coarse, sizeof(float), 64 * 256, f);
        fread(c->b_coarse, sizeof(float), 64, f);
        fread(c->bilin_W, sizeof(float), 256 * 256, f);
        fread(c->bilin_proj, sizeof(float), 256 * 2, f);
        fread(c->pW_gx, sizeof(float), PCFC_HD * PCFC_MAX_CH, f);
        fread(c->pW_ax, sizeof(float), PCFC_HD * PCFC_MAX_CH, f);
        fread(c->pW_gh, sizeof(float), PCFC_HD * PCFC_HD, f);
        fread(c->pW_ah, sizeof(float), PCFC_HD * PCFC_HD, f);
        fread(c->pb_g, sizeof(float), PCFC_HD, f);
        fread(c->pb_a, sizeof(float), PCFC_HD, f);
    } else if (flags[0]) {
        /* 文件中存在权重但管理器未初始化该子模型，跳过对应数据 */
        size_t skip_size = sizeof(IRDFineConfig) + sizeof(int)
            + (256 * 32 + 256 * 32 + 256 * 256 + 256 * 256) * sizeof(float)
            + (256 + 256) * sizeof(float)
            + (256 * 256 + 256 * 256 + 256 * 256 + 256 * 256) * sizeof(float)
            + (256 + 256) * sizeof(float)
            + (256 * 256 + 256) * sizeof(float)
            + (64 * 256 + 64) * sizeof(float)
            + (256 * 256 + 256 * 2) * sizeof(float)
            + (PCFC_HD * PCFC_MAX_CH + PCFC_HD * PCFC_MAX_CH) * sizeof(float)
            + (PCFC_HD * PCFC_HD + PCFC_HD * PCFC_HD) * sizeof(float)
            + (PCFC_HD + PCFC_HD) * sizeof(float);
        fseek(f, (long)skip_size, SEEK_CUR);
    }

    /* ---- 开放集识别器权重 ---- */
    if (flags[1] && manager->open_set_recognizer) {
        IRDOpenSetRecognizer* r = manager->open_set_recognizer;
        fread(&r->cfg, sizeof(IRDOpenSetConfig), 1, f);
        fread(&r->inv_temp, sizeof(float), 1, f);
        fread(&r->reject_threshold, sizeof(float), 1, f);
        fread(&r->num_known_classes, sizeof(int), 1, f);
        fread(&r->total_samples, sizeof(int), 1, f);
        fread(&r->weibull_thresh, sizeof(float), 1, f);
        fread(&r->nndr_threshold, sizeof(float), 1, f);
        fread(r->prototypes, sizeof(float), 256 * 256, f);
        fread(r->W_gx, sizeof(float), 256 * 32, f);
        fread(r->W_ax, sizeof(float), 256 * 32, f);
        fread(r->W_gh, sizeof(float), 256 * 256, f);
        fread(r->W_ah, sizeof(float), 256 * 256, f);
        fread(r->b_g, sizeof(float), 256, f);
        fread(r->b_a, sizeof(float), 256, f);
        fread(r->hidden, sizeof(float), 256, f);
        fread(r->weibull_k, sizeof(float), 256, f);
        fread(r->weibull_lambda, sizeof(float), 256, f);
        fread(r->pW_gx, sizeof(float), PCFC_HD * PCFC_MAX_CH, f);
        fread(r->pW_ax, sizeof(float), PCFC_HD * PCFC_MAX_CH, f);
        fread(r->pW_gh, sizeof(float), PCFC_HD * PCFC_HD, f);
        fread(r->pW_ah, sizeof(float), PCFC_HD * PCFC_HD, f);
        fread(r->pb_g, sizeof(float), PCFC_HD, f);
        fread(r->pb_a, sizeof(float), PCFC_HD, f);
    } else if (flags[1]) {
        size_t skip_size = sizeof(IRDOpenSetConfig) + sizeof(float) * 2
            + sizeof(int) * 2 + sizeof(float) * 2
            + (256 * 256 + 256 * 32 + 256 * 32 + 256 * 256 + 256 * 256) * sizeof(float)
            + (256 + 256 + 256 + 256 + 256) * sizeof(float)
            + (PCFC_HD * PCFC_MAX_CH + PCFC_HD * PCFC_MAX_CH) * sizeof(float)
            + (PCFC_HD * PCFC_HD + PCFC_HD * PCFC_HD) * sizeof(float)
            + (PCFC_HD + PCFC_HD) * sizeof(float);
        fseek(f, (long)skip_size, SEEK_CUR);
    }

    /* ---- 零样本识别器权重 ---- */
    if (flags[2] && manager->zero_shot_recognizer) {
        IRDZeroShotRecognizer* z = manager->zero_shot_recognizer;
        fread(&z->cfg, sizeof(IRDZeroShotConfig), 1, f);
        fread(&z->margin, sizeof(float), 1, f);
        fread(&z->num_seen_classes, sizeof(int), 1, f);
        fread(&z->num_unseen_classes, sizeof(int), 1, f);
        fread(&z->num_total_classes, sizeof(int), 1, f);
        fread(z->W_vis_sem, sizeof(float), 256 * 256, f);
        fread(z->b_vis_sem, sizeof(float), 256, f);
        fread(z->W_attr_pred, sizeof(float), 128 * 256, f);
        fread(z->b_attr_pred, sizeof(float), 128, f);
        fread(z->class_attributes, sizeof(float), 256 * 128, f);
        fread(z->semantic_prototypes, sizeof(float), 256 * 256, f);
        fread(z->W_gx, sizeof(float), 256 * 32, f);
        fread(z->W_ax, sizeof(float), 256 * 32, f);
        fread(z->W_gh, sizeof(float), 256 * 256, f);
        fread(z->W_ah, sizeof(float), 256 * 256, f);
        fread(z->b_g, sizeof(float), 256, f);
        fread(z->b_a, sizeof(float), 256, f);
        fread(z->hidden, sizeof(float), 256, f);
        fread(z->pW_gx, sizeof(float), PCFC_HD * PCFC_MAX_CH, f);
        fread(z->pW_ax, sizeof(float), PCFC_HD * PCFC_MAX_CH, f);
        fread(z->pW_gh, sizeof(float), PCFC_HD * PCFC_HD, f);
        fread(z->pW_ah, sizeof(float), PCFC_HD * PCFC_HD, f);
        fread(z->pb_g, sizeof(float), PCFC_HD, f);
        fread(z->pb_a, sizeof(float), PCFC_HD, f);
    } else if (flags[2]) {
        size_t skip_size = sizeof(IRDZeroShotConfig) + sizeof(float)
            + sizeof(int) * 3
            + (256 * 256 + 256 + 128 * 256 + 128) * sizeof(float)
            + (256 * 128 + 256 * 256 + 256 * 32 + 256 * 32) * sizeof(float)
            + (256 * 256 + 256 * 256 + 256 + 256 + 256) * sizeof(float)
            + (PCFC_HD * PCFC_MAX_CH + PCFC_HD * PCFC_MAX_CH) * sizeof(float)
            + (PCFC_HD * PCFC_HD + PCFC_HD * PCFC_HD) * sizeof(float)
            + (PCFC_HD + PCFC_HD) * sizeof(float);
        fseek(f, (long)skip_size, SEEK_CUR);
    }

    /* ---- 少样本识别器权重 ---- */
    if (flags[3] && manager->few_shot_recognizer) {
        IRDFewShotRecognizer* fs = manager->few_shot_recognizer;
        fread(&fs->cfg, sizeof(IRDFewShotConfig), 1, f);
        fread(&fs->tau, sizeof(float), 1, f);
        fread(&fs->dt, sizeof(float), 1, f);
        fread(&fs->num_support, sizeof(int), 1, f);
        fread(&fs->num_classes, sizeof(int), 1, f);
        fread(&fs->initialized, sizeof(int), 1, f);
        fread(fs->W_embed, sizeof(float), 32 * 256, f);
        fread(fs->b_embed, sizeof(float), 256, f);
        fread(fs->W_gx, sizeof(float), 256 * 32, f);
        fread(fs->W_ax, sizeof(float), 256 * 32, f);
        fread(fs->W_gh, sizeof(float), 256 * 256, f);
        fread(fs->W_ah, sizeof(float), 256 * 256, f);
        fread(fs->b_g, sizeof(float), 256, f);
        fread(fs->b_a, sizeof(float), 256, f);
        fread(fs->pW_gx, sizeof(float), PCFC_HD * PCFC_MAX_CH, f);
        fread(fs->pW_ax, sizeof(float), PCFC_HD * PCFC_MAX_CH, f);
        fread(fs->pW_gh, sizeof(float), PCFC_HD * PCFC_HD, f);
        fread(fs->pW_ah, sizeof(float), PCFC_HD * PCFC_HD, f);
        fread(fs->pb_g, sizeof(float), PCFC_HD, f);
        fread(fs->pb_a, sizeof(float), PCFC_HD, f);
        fread(fs->support_features, sizeof(float), IRD_MAX_SUPPORT_SAMPLES * IRD_SEMANTIC_DIM, f);
        fread(fs->support_labels, sizeof(int), IRD_MAX_SUPPORT_SAMPLES, f);
        fread(fs->prototypes, sizeof(float), IRD_MAX_PROTOTYPES * IRD_SEMANTIC_DIM, f);
        fread(fs->prototype_counts, sizeof(int), IRD_MAX_PROTOTYPES, f);
        fread(fs->finetune_W, sizeof(float), 256 * 32, f);
        fread(fs->finetune_b, sizeof(float), 256, f);
        fread(fs->class_names, sizeof(char), 256 * 64, f);
    } else if (flags[3]) {
        size_t skip_size = sizeof(IRDFewShotConfig) + sizeof(float) * 2
            + sizeof(int) * 3
            + (32 * 256 + 256) * sizeof(float)
            + (256 * 32 + 256 * 32 + 256 * 256 + 256 * 256 + 256 + 256) * sizeof(float)
            + (PCFC_HD * PCFC_MAX_CH + PCFC_HD * PCFC_MAX_CH) * sizeof(float)
            + (PCFC_HD * PCFC_HD + PCFC_HD * PCFC_HD) * sizeof(float)
            + (PCFC_HD + PCFC_HD) * sizeof(float)
            + IRD_MAX_SUPPORT_SAMPLES * IRD_SEMANTIC_DIM * sizeof(float)
            + IRD_MAX_SUPPORT_SAMPLES * sizeof(int)
            + IRD_MAX_PROTOTYPES * IRD_SEMANTIC_DIM * sizeof(float)
            + IRD_MAX_PROTOTYPES * sizeof(int)
            + (256 * 32 + 256) * sizeof(float)
            + 256 * 64 * sizeof(char);
        fseek(f, (long)skip_size, SEEK_CUR);
    }

    fclose(f);
    return 0;
}


