#include "selflnn/core/loss.h"
#include <math.h>
#include <float.h>
#include <string.h>

float loss_compute(const float* predictions, const float* targets, int n, LossType loss_type)
{
    return loss_compute_ex(predictions, targets, n, loss_type, NULL);
}

/* 损失函数默认超参数——可通过setter函数在运行时修改 */
static float g_default_focal_gamma = 2.0f;
static float g_default_focal_alpha = 0.25f;
static float g_default_dice_smooth = 1e-6f;
static float g_default_triplet_margin = 1.0f;
static float g_default_quantile_tau = 0.5f;

/* 可配置超参数setter函数 */
void loss_set_default_focal_gamma(float gamma) { g_default_focal_gamma = gamma; }
void loss_set_default_focal_alpha(float alpha) { g_default_focal_alpha = alpha; }
void loss_set_default_dice_smooth(float smooth) { g_default_dice_smooth = smooth; }
void loss_set_default_triplet_margin(float margin) { g_default_triplet_margin = margin; }
void loss_set_default_quantile_tau(float tau) { g_default_quantile_tau = tau; }

static float get_gamma(const LossConfig* c) {
    return (c && c->focal_gamma > 0.0f) ? c->focal_gamma : g_default_focal_gamma;
}
static float get_alpha(const LossConfig* c) {
    return (c && c->focal_alpha > 0.0f) ? c->focal_alpha : g_default_focal_alpha;
}
static float get_smooth(const LossConfig* c) {
    return (c && c->dice_smooth > 0.0f) ? c->dice_smooth : g_default_dice_smooth;
}
static float get_margin(const LossConfig* c) {
    return (c && c->triplet_margin > 0.0f) ? c->triplet_margin : g_default_triplet_margin;
}
static float get_tau(const LossConfig* c) {
    return (c && c->quantile_tau > 0.0f) ? c->quantile_tau : g_default_quantile_tau;
}

float loss_compute_ex(const float* predictions, const float* targets, int n,
                       LossType loss_type, const LossConfig* config)
{
    if (!predictions || !targets || n <= 0) return 0.0f;

    float loss = 0.0f;
    int i;

    switch (loss_type)
    {
        case LOSS_MSE:
        {
            for (i = 0; i < n; i++)
            {
                float diff = predictions[i] - targets[i];
                loss += diff * diff;
            }
            loss /= (float)n;
            break;
        }
        case LOSS_MAE:
        {
            for (i = 0; i < n; i++)
            {
                loss += fabsf(predictions[i] - targets[i]);
            }
            loss /= (float)n;
            break;
        }
        case LOSS_HUBER:
        {
            float delta = 1.0f;
            for (i = 0; i < n; i++)
            {
                float diff = predictions[i] - targets[i];
                float abs_diff = fabsf(diff);
                if (abs_diff <= delta)
                {
                    loss += 0.5f * diff * diff;
                }
                else
                {
                    loss += delta * (abs_diff - 0.5f * delta);
                }
            }
            loss /= (float)n;
            break;
        }
        case LOSS_CATEGORICAL_CROSSENTROPY:
        {
            /* P2-055修复: 多分类CrossEntropy仅限制预测值，target保持原始one-hot值 */
            for (i = 0; i < n; i++)
            {
                float p = predictions[i];
                if (p < FLT_EPSILON) p = FLT_EPSILON;
                if (p > 1.0f - FLT_EPSILON) p = 1.0f - FLT_EPSILON;
                loss -= targets[i] * logf(p);
            }
            loss /= (float)n;
            break;
        }
        case LOSS_BINARY_CROSSENTROPY:
        {
            for (i = 0; i < n; i++)
            {
                float p = predictions[i];
                if (p < FLT_EPSILON) p = FLT_EPSILON;
                if (p > 1.0f - FLT_EPSILON) p = 1.0f - FLT_EPSILON;
                float t = targets[i];
                loss -= t * logf(p) + (1.0f - t) * logf(1.0f - p);
            }
            loss /= (float)n;
            break;
        }
        case LOSS_KLD:
        {
            for (i = 0; i < n; i++)
            {
                float p = predictions[i];
                if (p < FLT_EPSILON) p = FLT_EPSILON;
                float q = targets[i];
                if (q < FLT_EPSILON) q = FLT_EPSILON;
                loss += p * logf(p / q);
            }
            loss /= (float)n;
            break;
        }
        case LOSS_COSINE:
        {
            float dot = 0.0f, norm_p = 0.0f, norm_t = 0.0f;
            for (i = 0; i < n; i++)
            {
                dot += predictions[i] * targets[i];
                norm_p += predictions[i] * predictions[i];
                norm_t += targets[i] * targets[i];
            }
            norm_p = sqrtf(norm_p > FLT_EPSILON ? norm_p : FLT_EPSILON);
            norm_t = sqrtf(norm_t > FLT_EPSILON ? norm_t : FLT_EPSILON);
            loss = 1.0f - dot / (norm_p * norm_t);
            break;
        }
        case LOSS_CONTRASTIVE:
        {
            float margin = 1.0f;
            for (i = 0; i < n; i += 2)
            {
                float diff = predictions[i] - targets[i];
                float d_sq = diff * diff;
                float d = sqrtf(d_sq + FLT_EPSILON);
                float y = targets[i + 1];
                loss += y * d_sq + (1.0f - y) * (margin > d ? (margin - d) * (margin - d) : 0.0f);
            }
            loss /= (float)(n > 1 ? n / 2 : 1);
            break;
        }
        case LOSS_FOCAL:
        {
            float gamma = get_gamma(config);
            float alpha = get_alpha(config);
            for (i = 0; i < n; i++)
            {
                float p = predictions[i];
                if (p < 1e-7f) p = 1e-7f;
                if (p > 1.0f - 1e-7f) p = 1.0f - 1e-7f;
                float t = targets[i];
                float pt = (t > 0.5f) ? p : 1.0f - p;
                float at = (t > 0.5f) ? alpha : 1.0f - alpha;
                float log_pt = logf(pt > 1e-7f ? pt : 1e-7f);
                loss -= at * powf(1.0f - pt, gamma) * log_pt;
            }
            loss /= (float)n;
            break;
        }
        case LOSS_DICE:
        {
            float smooth = get_smooth(config);
            float intersection = 0.0f, sum_p2 = 0.0f, sum_t2 = 0.0f;
            for (i = 0; i < n; i++)
            {
                float p = predictions[i];
                float t = targets[i];
                intersection += p * t;
                sum_p2 += p * p;
                sum_t2 += t * t;
            }
            float denom = sum_p2 + sum_t2 + smooth;
            loss = 1.0f - 2.0f * intersection / denom;
            break;
        }
        case LOSS_TRIPLET:
        {
            float margin = get_margin(config);
            int num_triplets = n / 3;
            if (num_triplets < 1) return 0.0f;
            for (i = 0; i < num_triplets * 3; i += 3)
            {
                float diff_ap = predictions[i] - predictions[i + 1];
                float diff_an = predictions[i] - predictions[i + 2];
                float d_ap = diff_ap * diff_ap;
                float d_an = diff_an * diff_an;
                loss += (d_ap - d_an + margin > 0.0f) ? (d_ap - d_an + margin) : 0.0f;
            }
            loss /= (float)num_triplets;
            break;
        }
        case LOSS_QUANTILE:
        {
            float tau = get_tau(config);
            for (i = 0; i < n; i++)
            {
                float diff = targets[i] - predictions[i];
                loss += (diff >= 0.0f) ? tau * diff : (tau - 1.0f) * diff;
            }
            loss /= (float)n;
            break;
        }
        default:
            break;
    }

    if (isnan(loss) || isinf(loss)) {
        /* FIX-006: NaN/Inf损失时返回大值作为哨兵，外部调用者应据此跳过批次 */
        return 1e6f;
    }
    return loss;
}

void loss_gradient_ex(const float* predictions, const float* targets, int n, float* gradients,
                       LossType loss_type, const LossConfig* config)
{
    if (!predictions || !targets || n <= 0 || !gradients) return;

    int i;

    switch (loss_type)
    {
        case LOSS_MSE:
        {
            float scale = 2.0f / (float)n;
            for (i = 0; i < n; i++)
            {
                gradients[i] = scale * (predictions[i] - targets[i]);
            }
            break;
        }
        case LOSS_MAE:
        {
            float scale = 1.0f / (float)n;
            for (i = 0; i < n; i++)
            {
                float diff = predictions[i] - targets[i];
                gradients[i] = scale * (diff >= 0.0f ? 1.0f : -1.0f);
            }
            break;
        }
        case LOSS_HUBER:
        {
            float delta = 1.0f;
            float scale = 1.0f / (float)n;
            for (i = 0; i < n; i++)
            {
                float diff = predictions[i] - targets[i];
                float abs_diff = fabsf(diff);
                if (abs_diff <= delta)
                {
                    gradients[i] = scale * diff;
                }
                else
                {
                    gradients[i] = scale * delta * (diff >= 0.0f ? 1.0f : -1.0f);
                }
            }
            break;
        }
        case LOSS_CATEGORICAL_CROSSENTROPY:
        {
            float scale = 1.0f / (float)n;
            for (i = 0; i < n; i++)
            {
                float p = predictions[i];
                if (p < FLT_EPSILON) p = FLT_EPSILON;
                if (p > 1.0f - FLT_EPSILON) p = 1.0f - FLT_EPSILON;
                gradients[i] = scale * (-targets[i] / p);
            }
            break;
        }
        case LOSS_BINARY_CROSSENTROPY:
        {
            float scale = 1.0f / (float)n;
            for (i = 0; i < n; i++)
            {
                float p = predictions[i];
                if (p < FLT_EPSILON) p = FLT_EPSILON;
                if (p > 1.0f - FLT_EPSILON) p = 1.0f - FLT_EPSILON;
                gradients[i] = scale * (p - targets[i]) / (p * (1.0f - p) + FLT_EPSILON);
            }
            break;
        }
        case LOSS_KLD:
        {
            float scale = 1.0f / (float)n;
            for (i = 0; i < n; i++)
            {
                float p = predictions[i];
                if (p < FLT_EPSILON) p = FLT_EPSILON;
                float q = targets[i];
                if (q < FLT_EPSILON) q = FLT_EPSILON;
                gradients[i] = scale * (logf(p / q) + 1.0f);
            }
            break;
        }
        case LOSS_COSINE:
        {
            float dot = 0.0f, norm_p = 0.0f, norm_t = 0.0f;
            for (i = 0; i < n; i++)
            {
                dot += predictions[i] * targets[i];
                norm_p += predictions[i] * predictions[i];
                norm_t += targets[i] * targets[i];
            }
            norm_p = sqrtf(norm_p > FLT_EPSILON ? norm_p : FLT_EPSILON);
            norm_t = sqrtf(norm_t > FLT_EPSILON ? norm_t : FLT_EPSILON);
            float denom = norm_p * norm_t;
            if (denom > FLT_EPSILON)
            {
                float norm_p_sq = norm_p * norm_p;
                for (i = 0; i < n; i++)
                {
                    gradients[i] = -(targets[i] / denom - dot * predictions[i] / (norm_p_sq * denom));
                }
            }
            break;
        }
        case LOSS_CONTRASTIVE:
        {
            float margin = 1.0f;
            for (i = 0; i < n; i += 2)
            {
                float diff = predictions[i] - targets[i];
                float d_sq = diff * diff;
                float d = sqrtf(d_sq + FLT_EPSILON);
                float y = targets[i + 1];
                float grad = y * 2.0f * diff;
                if (margin > d)
                {
                    grad += (1.0f - y) * 2.0f * (margin - d) * (-diff / (d + FLT_EPSILON));
                }
                gradients[i] = grad;
                gradients[i + 1] = 0.0f;
            }
            break;
        }
        case LOSS_FOCAL:
        {
            float gamma = get_gamma(config);
            float alpha = get_alpha(config);
            float inv_n = 1.0f / (float)n;
            for (i = 0; i < n; i++)
            {
                float p = predictions[i];
                if (p < FLT_EPSILON) p = FLT_EPSILON;
                if (p > 1.0f - FLT_EPSILON) p = 1.0f - FLT_EPSILON;
                float t = (targets[i] > 0.5f) ? 1.0f : 0.0f;
                float pt = (t > 0.5f) ? p : 1.0f - p;
                float at = (t > 0.5f) ? alpha : 1.0f - alpha;
                float grad_mod = at * powf(1.0f - pt, gamma);
                float dpt_dp = (t > 0.5f) ? 1.0f : -1.0f;
                float inner = gamma * logf(pt + FLT_EPSILON) / (1.0f - pt + FLT_EPSILON) - 1.0f / (pt + FLT_EPSILON);
                gradients[i] = -inv_n * grad_mod * inner * dpt_dp;
            }
            break;
        }
        case LOSS_DICE:
        {
            float smooth = get_smooth(config);
            float intersection = 0.0f, sum_p2 = 0.0f, sum_t2 = 0.0f;
            for (i = 0; i < n; i++)
            {
                float p = predictions[i];
                float t = targets[i];
                intersection += p * t;
                sum_p2 += p * p;
                sum_t2 += t * t;
            }
            float denom = sum_p2 + sum_t2 + smooth;
            float denom_sq = denom * denom;
            for (i = 0; i < n; i++)
            {
                float t = targets[i];
                float p = predictions[i];
                gradients[i] = -2.0f * (t * denom - 2.0f * p * intersection) / (denom_sq + FLT_EPSILON);
            }
            break;
        }
        case LOSS_TRIPLET:
        {
            float margin = get_margin(config);
            int num_triplets = n / 3;
            if (num_triplets < 1) return;
            float inv_n = 1.0f / (float)num_triplets;
            for (i = 0; i < num_triplets * 3; i += 3)
            {
                float a = predictions[i], p = predictions[i + 1], neg = predictions[i + 2];
                float d_ap = (a - p) * (a - p);
                float d_an = (a - neg) * (a - neg);
                float loss_val = d_ap - d_an + margin;
                if (loss_val > 0.0f)
                {
                    gradients[i] = inv_n * 2.0f * ((a - p) - (a - neg));
                    gradients[i + 1] = -inv_n * 2.0f * (a - p);
                    gradients[i + 2] = inv_n * 2.0f * (a - neg);
                }
                else
                {
                    gradients[i] = 0.0f;
                    gradients[i + 1] = 0.0f;
                    gradients[i + 2] = 0.0f;
                }
            }
            break;
        }
        case LOSS_QUANTILE:
        {
            float tau = get_tau(config);
            float inv_n = 1.0f / (float)n;
            for (i = 0; i < n; i++)
            {
                float diff = targets[i] - predictions[i];
                gradients[i] = inv_n * ((diff >= 0.0f) ? -tau : -(tau - 1.0f));
            }
            break;
        }
        default:
            break;
    }

    /* FIX-006: 检测并清零所有非有限梯度值，防止NaN/Inf传播到参数更新 */
    for (i = 0; i < n; i++) {
        if (!isfinite(gradients[i])) {
            gradients[i] = 0.0f;
        }
    }
}

void loss_gradient(const float* predictions, const float* targets, int n, float* gradients, LossType loss_type)
{
    loss_gradient_ex(predictions, targets, n, gradients, loss_type, NULL);
}

/* ==========================================================================
 * ZSFWS-024: 多模态损失函数实现
 * 解决多模态输出多尺度差异（视觉[-1,1]、文本离散ID、传感器变范围）
 * ========================================================================== */

/**
 * @brief 解析模态类型→默认损失函数映射
 */
LossType loss_get_default_for_modality(ModalityType modality)
{
    switch (modality) {
        case MODALITY_VISUAL:
            return LOSS_HUBER;                    /* [-1,1]浮点，Huber鲁棒于异常像素 */
        case MODALITY_TEXT_LOGITS:
            return LOSS_CATEGORICAL_CROSSENTROPY; /* 离散token logits，CE最优 */
        case MODALITY_TEXT_EMBED:
            return LOSS_COSINE;                   /* 嵌入向量，方向相似度优于L2 */
        case MODALITY_SENSOR:
            return LOSS_HUBER;                    /* 变范围传感器，Huber抗噪声 */
        case MODALITY_CONTROL:
            return LOSS_MSE;                      /* 连续控制信号，MSE精确 */
        case MODALITY_AUDIO:
            return LOSS_MAE;                      /* 频域特征，MAE抗尖峰干扰 */
        case MODALITY_CUSTOM:
        default:
            return LOSS_MSE;                      /* 自定义回退到MSE */
    }
}

/**
 * @brief 验证多模态段描述的完整性
 * @return 0=有效, -1=有越界段, -2=有重叠段
 */
static int validate_multimodal_segments(const MultimodalLossSegment* segments,
                                         int num_segments, int total_length)
{
    if (!segments || num_segments <= 0) return -1;
    if (total_length <= 0) return -1;

    /* 检查每段的边界和重叠 */
    for (int i = 0; i < num_segments; i++) {
        const MultimodalLossSegment* seg = &segments[i];
        if (seg->start_index < 0 || seg->length <= 0) return -1;
        if (seg->start_index + seg->length > total_length) return -1;
        if (seg->weight < 0.0f) return -1;        /* 负权重无效 */

        /* 检查与前一段是否重叠（段需按start_index升序） */
        if (i > 0) {
            const MultimodalLossSegment* prev = &segments[i - 1];
            if (seg->start_index < prev->start_index + prev->length) {
                return -2;                          /* 段重叠 */
            }
        }
    }
    return 0;
}

/**
 * @brief 计算多模态损失（带每段独立梯度输出）
 * 
 * 遍历所有模态段，对每段使用其模态类型对应的最佳损失函数计算损失值。
 * 
 * @return float 加权总损失，验证失败返回1e6f
 */
float loss_compute_multimodal(const float* predictions, const float* targets,
                               int total_length,
                               const MultimodalLossSegment* segments,
                               int num_segments)
{
    if (!predictions || !targets || !segments || num_segments <= 0 || total_length <= 0) {
        return 1e6f;
    }

    /* 验证段描述 */
    if (validate_multimodal_segments(segments, num_segments, total_length) != 0) {
        return 1e6f;
    }

    float total_loss = 0.0f;
    float total_weight = 0.0f;

    for (int s = 0; s < num_segments; s++) {
        const MultimodalLossSegment* seg = &segments[s];
        const float* seg_pred = predictions + seg->start_index;
        const float* seg_target = targets + seg->start_index;

        /* 确定该段使用的损失类型 */
        LossType loss_type;
        const LossConfig* loss_cfg;
        if (seg->modality == MODALITY_CUSTOM) {
            loss_type = seg->custom_loss_type;
        } else {
            loss_type = loss_get_default_for_modality(seg->modality);
        }

        /* 若config全零则传NULL使用默认值 */
        if (seg->loss_config.focal_gamma > 0.0f || seg->loss_config.focal_alpha > 0.0f ||
            seg->loss_config.dice_smooth > 0.0f || seg->loss_config.triplet_margin > 0.0f ||
            seg->loss_config.quantile_tau > 0.0f) {
            loss_cfg = &seg->loss_config;
        } else {
            loss_cfg = NULL;
        }

        float seg_loss = loss_compute_ex(seg_pred, seg_target, seg->length, loss_type, loss_cfg);
        float seg_weight = seg->weight > 0.0f ? seg->weight : 1.0f;

        total_loss += seg_weight * seg_loss;
        total_weight += seg_weight;
    }

    /* 归一化：除以总权重以保持损失量纲一致性 */
    if (total_weight > 0.0f) {
        total_loss /= total_weight;
    }

    if (isnan(total_loss) || isinf(total_loss)) {
        return 1e6f;
    }
    return total_loss;
}

/**
 * @brief 计算多模态损失梯度
 * 
 * 对每个模态段独立计算梯度，写入统一梯度数组对应位置。
 * 先全局清零再逐段填充，确保未被覆盖的区域梯度为0。
 */
void loss_gradient_multimodal(const float* predictions, const float* targets,
                               int total_length, float* gradients,
                               const MultimodalLossSegment* segments,
                               int num_segments)
{
    if (!predictions || !targets || !gradients || !segments || num_segments <= 0 || total_length <= 0) {
        return;
    }

    if (validate_multimodal_segments(segments, num_segments, total_length) != 0) {
        return;
    }

    /* 全局清零：未覆盖区域梯度归零，防止跨模态泄漏 */
    memset(gradients, 0, (size_t)total_length * sizeof(float));

    for (int s = 0; s < num_segments; s++) {
        const MultimodalLossSegment* seg = &segments[s];
        const float* seg_pred = predictions + seg->start_index;
        const float* seg_target = targets + seg->start_index;
        float* seg_grad = gradients + seg->start_index;

        /* 确定损失类型 */
        LossType loss_type;
        const LossConfig* loss_cfg;
        if (seg->modality == MODALITY_CUSTOM) {
            loss_type = seg->custom_loss_type;
        } else {
            loss_type = loss_get_default_for_modality(seg->modality);
        }

        if (seg->loss_config.focal_gamma > 0.0f || seg->loss_config.focal_alpha > 0.0f ||
            seg->loss_config.dice_smooth > 0.0f || seg->loss_config.triplet_margin > 0.0f ||
            seg->loss_config.quantile_tau > 0.0f) {
            loss_cfg = &seg->loss_config;
        } else {
            loss_cfg = NULL;
        }

        float seg_weight = seg->weight > 0.0f ? seg->weight : 1.0f;

        /* 计算该段的原始梯度 */
        loss_gradient_ex(seg_pred, seg_target, seg->length, seg_grad, loss_type, loss_cfg);

        /* 应用段权重缩放梯度 */
        if (seg_weight != 1.0f) {
            for (int i = 0; i < seg->length; i++) {
                seg_grad[i] *= seg_weight;
            }
        }
    }
}
