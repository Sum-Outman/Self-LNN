#include "selflnn/core/loss.h"
#include <math.h>
#include <float.h>

float loss_compute(const float* predictions, const float* targets, int n, LossType loss_type)
{
    return loss_compute_ex(predictions, targets, n, loss_type, NULL);
}

static float default_focal_gamma(void) { return 2.0f; }
static float default_focal_alpha(void) { return 0.25f; }
static float default_dice_smooth(void) { return 1e-6f; }
static float default_triplet_margin(void) { return 1.0f; }
static float default_quantile_tau(void) { return 0.5f; }

static float get_gamma(const LossConfig* c) {
    return (c && c->focal_gamma > 0.0f) ? c->focal_gamma : default_focal_gamma();
}
static float get_alpha(const LossConfig* c) {
    return (c && c->focal_alpha > 0.0f) ? c->focal_alpha : default_focal_alpha();
}
static float get_smooth(const LossConfig* c) {
    return (c && c->dice_smooth > 0.0f) ? c->dice_smooth : default_dice_smooth();
}
static float get_margin(const LossConfig* c) {
    return (c && c->triplet_margin > 0.0f) ? c->triplet_margin : default_triplet_margin();
}
static float get_tau(const LossConfig* c) {
    return (c && c->quantile_tau > 0.0f) ? c->quantile_tau : default_quantile_tau();
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
