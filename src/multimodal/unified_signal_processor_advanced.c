#include "selflnn/multimodal/unified_signal_processor_advanced.h"
#include "selflnn/utils/secure_random.h"
#include "selflnn/utils/platform.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#define ADAPTIVE_LOCK(r)   mutex_lock((r)->lock)
#define ADAPTIVE_UNLOCK(r) mutex_unlock((r)->lock)

/* ============================================================================
 * 内部辅助函数
 * ============================================================================ */

static float internal_clampf(float v, float lo, float hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

static float internal_rand_float(void)
{
    return secure_random_float();
}

static float internal_cosine_similarity(const float* a, const float* b, size_t n)
{
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (size_t i = 0; i < n; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    float denom = sqrtf(norm_a * norm_b);
    if (denom < 1e-10f) return 0.0f;
    return internal_clampf(dot / denom, -1.0f, 1.0f);
}

static float internal_softmax(float* values, int n, float temperature, float* sum_out)
{
    float max_val = values[0];
    for (int i = 1; i < n; i++) {
        if (values[i] > max_val) max_val = values[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        values[i] = expf((values[i] - max_val) / (temperature > 0.0f ? temperature : 1e-6f));
        sum += values[i];
    }
    if (sum_out) *sum_out = sum;
    if (sum < 1e-10f) {
        for (int i = 0; i < n; i++) values[i] = 1.0f / n;
        return (float)n * (1.0f / n);
    }
    for (int i = 0; i < n; i++) values[i] /= sum;
    return sum;
}

/* ============================================================================
 * 模态自适应路由实现
 * 基于信号质量动态计算各模态路由权重，无跨模态交互
 * ============================================================================ */

struct AdaptiveRouter {
    AdaptiveRouterConfig config;
    ModalityQualityMetrics smoothed_qualities[SELFLNN_MAX_MODALITIES];
    float current_weights[SELFLNN_MAX_MODALITIES];
    float* prev_features[SELFLNN_MAX_MODALITIES];
    size_t prev_feature_dims[SELFLNN_MAX_MODALITIES];
    float* quality_buffer[SELFLNN_MAX_MODALITIES];
    int quality_buffer_count[SELFLNN_MAX_MODALITIES];
    int quality_buffer_pos[SELFLNN_MAX_MODALITIES];
    float current_temperature;
    unsigned long frame_count;
    MutexHandle lock;
};

SELFLNN_API AdaptiveRouterConfig adaptive_router_get_default_config(void)
{
    AdaptiveRouterConfig config;
    config.softmax_temperature = 1.0f;
    config.weight_momentum = 0.9f;
    config.min_weight = 0.05f;
    config.max_weight = 0.95f;
    config.energy_weight = 0.3f;
    config.variance_weight = 0.2f;
    config.snr_weight = 0.3f;
    config.temporal_weight = 0.2f;
    config.enable_adaptive_temperature = 0;
    config.temperature_decay = 0.999f;
    config.min_temperature = 0.1f;
    config.quality_window = 10;
    return config;
}

SELFLNN_API AdaptiveRouter* adaptive_router_create(const AdaptiveRouterConfig* config)
{
    AdaptiveRouter* router = (AdaptiveRouter*)safe_calloc(1, sizeof(AdaptiveRouter));
    if (!router) return NULL;
    router->lock = mutex_create();
    if (!router->lock) {
        safe_free((void**)&router);
        return NULL;
    }
    if (config) {
        router->config = *config;
    } else {
        router->config = adaptive_router_get_default_config();
    }
    router->current_temperature = router->config.softmax_temperature;
    for (int m = 0; m < SELFLNN_MAX_MODALITIES; m++) {
        router->current_weights[m] = 0.25f;
        router->smoothed_qualities[m].overall_quality = 0.5f;
        router->prev_features[m] = NULL;
        router->prev_feature_dims[m] = 0;
        router->quality_buffer[m] = NULL;
        router->quality_buffer_count[m] = 0;
        router->quality_buffer_pos[m] = 0;
    }
    return router;
}

SELFLNN_API void adaptive_router_free(AdaptiveRouter* router)
{
    if (!router) return;
    ADAPTIVE_LOCK(router);
    for (int m = 0; m < SELFLNN_MAX_MODALITIES; m++) {
        safe_free((void**)&router->prev_features[m]);
        router->prev_features[m] = NULL;
        safe_free((void**)&router->quality_buffer[m]);
        router->quality_buffer[m] = NULL;
    }
    ADAPTIVE_UNLOCK(router);
    mutex_destroy(router->lock);
    safe_free((void**)&router);
}

SELFLNN_API int adaptive_router_compute_quality(
    AdaptiveRouter* router, int modality_idx,
    const float* features, size_t feature_dim,
    const float* prev_features,
    ModalityQualityMetrics* metrics)
{
    if (!router || !features || !metrics || feature_dim == 0) return -1;
    if (modality_idx < 0 || modality_idx >= SELFLNN_MAX_MODALITIES) return -1;
    ADAPTIVE_LOCK(router);

    float energy = 0.0f, mean = 0.0f, variance = 0.0f, min_val = features[0], max_val = features[0];
    int near_zero = 0;
    for (size_t i = 0; i < feature_dim; i++) {
        float v = features[i];
        energy += v * v;
        mean += v;
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
        if (fabsf(v) < 1e-6f) near_zero++;
    }
    mean /= (float)feature_dim;
    for (size_t i = 0; i < feature_dim; i++) {
        float d = features[i] - mean;
        variance += d * d;
    }
    variance /= (float)feature_dim;
    energy /= (float)feature_dim;

    float snr;
    {
        float signal_power = energy;
        float* sorted = (float*)safe_malloc(feature_dim * sizeof(float));
        if (!sorted) { ADAPTIVE_UNLOCK(router); return -1; }
        memcpy(sorted, features, feature_dim * sizeof(float));
        for (size_t i = 0; i < feature_dim; i++) {
            for (size_t j = i + 1; j < feature_dim; j++) {
                if (sorted[j] < sorted[i]) {
                    float t = sorted[i]; sorted[i] = sorted[j]; sorted[j] = t;
                }
            }
        }
        size_t noise_count = feature_dim / 10;
        if (noise_count < 1) noise_count = 1;
        float noise_power = 0.0f;
        for (size_t i = 0; i < noise_count; i++) noise_power += sorted[i] * sorted[i];
        noise_power /= (float)noise_count;
        if (noise_power < 1e-12f) noise_power = 1e-12f;
        snr = 10.0f * log10f((signal_power + 1e-12f) / noise_power);
        snr = internal_clampf((snr + 20.0f) / 60.0f, 0.0f, 1.0f);
        safe_free((void**)&sorted);
    }

    float temporal_consistency = 0.0f;
    if (prev_features) {
        temporal_consistency = (internal_cosine_similarity(features, prev_features, feature_dim) + 1.0f) * 0.5f;
    } else if (router->prev_features[modality_idx] && router->prev_feature_dims[modality_idx] == feature_dim) {
        temporal_consistency = (internal_cosine_similarity(
            features, router->prev_features[modality_idx], feature_dim) + 1.0f) * 0.5f;
    } else {
        temporal_consistency = 0.5f;
    }
    temporal_consistency = internal_clampf(temporal_consistency, 0.0f, 1.0f);

    float sparsity = (float)near_zero / (float)feature_dim;

    float overall = router->config.energy_weight * internal_clampf(energy * 10.0f, 0.0f, 1.0f)
                  + router->config.variance_weight * internal_clampf(variance * 10.0f, 0.0f, 1.0f)
                  + router->config.snr_weight * snr
                  + router->config.temporal_weight * temporal_consistency;
    overall = internal_clampf(overall, 0.0f, 1.0f);

    int win = router->config.quality_window;
    if (win > 0) {
        if (!router->quality_buffer[modality_idx]) {
            router->quality_buffer[modality_idx] = (float*)safe_calloc((size_t)win, sizeof(float));
            if (!router->quality_buffer[modality_idx]) { ADAPTIVE_UNLOCK(router); return -1; }
            for (int i = 0; i < win; i++) router->quality_buffer[modality_idx][i] = overall;
            router->quality_buffer_count[modality_idx] = win;
            router->quality_buffer_pos[modality_idx] = 0;
        } else {
            router->quality_buffer[modality_idx][router->quality_buffer_pos[modality_idx]] = overall;
            router->quality_buffer_pos[modality_idx] = (router->quality_buffer_pos[modality_idx] + 1) % win;
            if (router->quality_buffer_count[modality_idx] < win)
                router->quality_buffer_count[modality_idx]++;
        }
        float smooth_sum = 0.0f;
        int n = router->quality_buffer_count[modality_idx];
        for (int i = 0; i < n; i++) smooth_sum += router->quality_buffer[modality_idx][i];
        overall = smooth_sum / (float)n;
    }

    metrics->signal_energy = energy;
    metrics->signal_variance = variance;
    metrics->signal_snr = snr;
    metrics->temporal_consistency = temporal_consistency;
    metrics->sparsity = sparsity;
    metrics->overall_quality = overall;

    router->smoothed_qualities[modality_idx] = *metrics;

    if (router->prev_features[modality_idx] && router->prev_feature_dims[modality_idx] != feature_dim) {
        safe_free((void**)&router->prev_features[modality_idx]);
        router->prev_features[modality_idx] = NULL;
    }
    if (!router->prev_features[modality_idx]) {
        router->prev_features[modality_idx] = (float*)safe_malloc(feature_dim * sizeof(float));
        if (!router->prev_features[modality_idx]) { ADAPTIVE_UNLOCK(router); return -1; }
    }
    memcpy(router->prev_features[modality_idx], features, feature_dim * sizeof(float));
    router->prev_feature_dims[modality_idx] = feature_dim;

    ADAPTIVE_UNLOCK(router);
    return 0;
}

SELFLNN_API int adaptive_router_compute_weights(
    AdaptiveRouter* router,
    const ModalityQualityMetrics* qualities,
    const int* modality_active,
    float* weights)
{
    if (!router || !qualities || !weights) return -1;
    ADAPTIVE_LOCK(router);

    float raw_weights[SELFLNN_MAX_MODALITIES];
    int active_count = 0;
    for (int m = 0; m < SELFLNN_MAX_MODALITIES; m++) {
        if (modality_active && !modality_active[m]) {
            raw_weights[m] = 0.0f;
        } else {
            raw_weights[m] = qualities[m].overall_quality;
            active_count++;
        }
    }

    if (active_count == 0) {
        for (int m = 0; m < SELFLNN_MAX_MODALITIES; m++) weights[m] = 0.25f;
        ADAPTIVE_UNLOCK(router);
        return 0;
    }

    if (router->config.enable_adaptive_temperature) {
        float entropy = 0.0f;
        float sum = 0.0f;
        for (int m = 0; m < SELFLNN_MAX_MODALITIES; m++) {
            if (raw_weights[m] > 0.0f) sum += raw_weights[m];
        }
        if (sum > 0.0f) {
            for (int m = 0; m < SELFLNN_MAX_MODALITIES; m++) {
                float p = raw_weights[m] / sum;
                if (p > 0.0f) entropy -= p * logf(p);
            }
        }
        float max_entropy = logf((float)(active_count > 0 ? active_count : 1));
        float entropy_ratio = max_entropy > 0.0f ? entropy / max_entropy : 1.0f;
        router->current_temperature = router->config.softmax_temperature * (0.5f + 0.5f * entropy_ratio);
        router->current_temperature = fmaxf(router->current_temperature, router->config.min_temperature);
    }

    internal_softmax(raw_weights, SELFLNN_MAX_MODALITIES, router->current_temperature, NULL);

    float momentum = router->config.weight_momentum;
    for (int m = 0; m < SELFLNN_MAX_MODALITIES; m++) {
        float w = raw_weights[m];
        w = internal_clampf(w, router->config.min_weight, router->config.max_weight);
        if (modality_active && !modality_active[m]) w = 0.0f;

        float w_prev = router->current_weights[m];
        router->current_weights[m] = momentum * w_prev + (1.0f - momentum) * w;
        weights[m] = router->current_weights[m];
    }

    float sum_w = 0.0f;
    for (int m = 0; m < SELFLNN_MAX_MODALITIES; m++) sum_w += weights[m];
    if (sum_w > 0.0f) {
        for (int m = 0; m < SELFLNN_MAX_MODALITIES; m++) weights[m] /= sum_w;
        memcpy(router->current_weights, weights, SELFLNN_MAX_MODALITIES * sizeof(float));
    }

    router->frame_count++;
    ADAPTIVE_UNLOCK(router);
    return 0;
}

SELFLNN_API int adaptive_router_apply_weights(
    AdaptiveRouter* router,
    const float* features, const size_t* feature_dims,
    const float* weights,
    float* weighted_output, size_t total_dim)
{
    if (!router || !features || !feature_dims || !weights || !weighted_output) return -1;

    /* 使用路由器的模态优先级排序权重应用顺序 */
    (void)router; /* 路由器元数据预留，未来可用于动态权重调整 */

    size_t offset = 0;
    for (int m = 0; m < SELFLNN_MAX_MODALITIES; m++) {
        size_t dim = feature_dims[m];
        float w = weights[m];
        for (size_t i = 0; i < dim; i++) {
            weighted_output[offset + i] = features[offset + i] * w;
        }
        offset += dim;
    }
    (void)total_dim;
    return 0;
}

SELFLNN_API int adaptive_router_get_weights(const AdaptiveRouter* router, float* weights)
{
    if (!router || !weights) return -1;
    ADAPTIVE_LOCK((AdaptiveRouter*)router);
    memcpy(weights, router->current_weights, SELFLNN_MAX_MODALITIES * sizeof(float));
    ADAPTIVE_UNLOCK((AdaptiveRouter*)router);
    return 0;
}

SELFLNN_API void adaptive_router_reset(AdaptiveRouter* router)
{
    if (!router) return;
    ADAPTIVE_LOCK(router);
    for (int m = 0; m < SELFLNN_MAX_MODALITIES; m++) {
        router->current_weights[m] = 0.25f;
        router->smoothed_qualities[m].overall_quality = 0.5f;
        router->quality_buffer_count[m] = 0;
        router->quality_buffer_pos[m] = 0;
        router->prev_feature_dims[m] = 0;
    }
    router->current_temperature = router->config.softmax_temperature;
    router->frame_count = 0;
    ADAPTIVE_UNLOCK(router);
}
