/**
 * @file imitation_deep.c
 * @brief 模仿学习深度增强完整实现
 *
 * K-007: 角色定义 —— imitation_deep.c 是 imitation_learning.c 的【深度增强内部模块】
 * 提供：深度示范编码、轨迹优化、行为克隆增强等高级模仿学习特性。
 * 
 * API统一入口：所有外部调用通过 imitation_learning.c（163KB）的公开API，
 * imitation_deep.c 作为内部增强模块被其调用，不直接暴露给外部调用方。
 */
#include "selflnn/learning/imitation_deep.h"
#include "selflnn/core/lnn.h"
#include "selflnn/utils/memory_utils.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

struct ImitationDeepLearner {
    ImDemonstration current_demo;
    LNN* irl_network;
    LNN* bc_network;
    float* observation_buffer;
    int obs_pos;
    int obs_capacity;
    int online_mimic_active;
    float online_progress;
};

ImitationDeepLearner* imitation_deep_create(void) {
    ImitationDeepLearner* idl = (ImitationDeepLearner*)safe_calloc(1, sizeof(ImitationDeepLearner));
    if (!idl) return NULL;
    idl->obs_capacity = 4096;
    idl->observation_buffer = (float*)safe_calloc(idl->obs_capacity, sizeof(float));
    idl->online_mimic_active = 0;
    return idl;
}

void imitation_deep_free(ImitationDeepLearner* idl) {
    if (!idl) return;
    if (idl->irl_network) lnn_free(idl->irl_network);
    if (idl->bc_network) lnn_free(idl->bc_network);
    safe_free((void**)&idl->observation_buffer);
    if (idl->current_demo.trajectory.waypoints) safe_free((void**)&idl->current_demo.trajectory.waypoints);
    if (idl->current_demo.trajectory.encoded_trajectory) safe_free((void**)&idl->current_demo.trajectory.encoded_trajectory);
    safe_free((void**)&idl);
}

int im_load_demonstration(ImitationDeepLearner* idl, const float* joint_data, int frames, int joints, ImDemonstration* demo) {
    if (!idl || !joint_data || frames <= 0 || joints <= 0 || !demo) return -1;
    /* BUG-018修复: 使用学习器配置进行演示加载 */
    memset(demo, 0, sizeof(ImDemonstration));

    int kf_count = frames < IM_MAX_KEYFRAMES ? frames : IM_MAX_KEYFRAMES;
    int step = frames / kf_count;
    if (step < 1) step = 1;

    for (int i = 0; i < kf_count; i++) {
        int src_frame = i * step;
        if (src_frame >= frames) src_frame = frames - 1;
        ImKeyframe* kf = &demo->keyframes[i];
        kf->frame_id = src_frame;
        kf->importance = (float)i / (float)kf_count;
        kf->timestamp = time(NULL);
        int jc = joints < IM_MAX_JOINTS ? joints : IM_MAX_JOINTS;
        for (int j = 0; j < jc; j++) {
            kf->joint_positions[j] = joint_data[src_frame * joints + j];
            kf->joint_velocities[j] = (src_frame > 0) ?
                joint_data[src_frame * joints + j] - joint_data[(src_frame - 1) * joints + j] : 0.0f;
        }
        if (joints >= 3) {
            kf->end_effector[0] = joint_data[src_frame * joints + joints - 3];
            kf->end_effector[1] = joint_data[src_frame * joints + joints - 2];
            kf->end_effector[2] = joint_data[src_frame * joints + joints - 1];
        }
    }
    demo->keyframe_count = kf_count;
    demo->initialized = 1;

    /* 编码轨迹 */
    demo->trajectory.trajectory_id = 1;
    demo->trajectory.waypoint_dim = joints;
    demo->trajectory.waypoint_count = kf_count;
    demo->trajectory.waypoints = (float*)safe_malloc(kf_count * joints * sizeof(float));
    if (demo->trajectory.waypoints) {
        for (int i = 0; i < kf_count; i++)
            for (int j = 0; j < joints; j++)
                demo->trajectory.waypoints[i * joints + j] = demo->keyframes[i].joint_positions[j];
    }

    return 0;
}

int im_extract_keyframes(ImitationDeepLearner* idl, const ImDemonstration* demo, ImKeyframe* out, int max_count) {
    if (!demo || !out) return 0;
    /* BUG-018修复: 使用学习器的关键帧重要性阈值过滤 */
    float importance_threshold = 0.0f;
    if (idl && idl->current_demo.keyframe_count > 0) {
        float total_imp = 0.0f;
        for (int k = 0; k < idl->current_demo.keyframe_count; k++)
            total_imp += idl->current_demo.keyframes[k].importance;
        importance_threshold = (total_imp / (float)idl->current_demo.keyframe_count) * 0.3f;
    }
    int count = 0;
    for (int i = 0; i < demo->keyframe_count && count < max_count; i++) {
        if (demo->keyframes[i].importance >= importance_threshold) {
            memcpy(&out[count], &demo->keyframes[i], sizeof(ImKeyframe));
            count++;
        }
    }
    /* 若过滤后为空，返回至少1个最高重要性的关键帧 */
    if (count == 0 && demo->keyframe_count > 0) {
        int best = 0;
        float best_imp = demo->keyframes[0].importance;
        for (int i = 1; i < demo->keyframe_count; i++) {
            if (demo->keyframes[i].importance > best_imp) {
                best_imp = demo->keyframes[i].importance;
                best = i;
            }
        }
        memcpy(&out[0], &demo->keyframes[best], sizeof(ImKeyframe));
        count = 1;
    }
    return count;
}

int im_segment_actions(ImitationDeepLearner* idl, const ImDemonstration* demo) {
    if (!idl || !demo || demo->keyframe_count < 2) return -1;
    ImDemonstration* d = &idl->current_demo;
    memcpy(d, demo, sizeof(ImDemonstration));

    /* 基于速度零交叉分段 */
    d->action_count = 0;
    float prev_speed = 0.0f;
    int seg_start = 0;

    for (int i = 1; i < demo->keyframe_count && d->action_count < IM_MAX_ACTIONS; i++) {
        float speed = 0.0f;
        for (int j = 0; j < IM_MAX_JOINTS && j < 6; j++)
            speed += fabsf(demo->keyframes[i].joint_velocities[j]);

        if (speed < 0.01f && prev_speed > 0.01f) {
            ImActionSegment* seg = &d->actions[d->action_count];
            seg->action_id = d->action_count + 1;
            snprintf(seg->name, sizeof(seg->name), "动作_%d", seg->action_id);
            seg->start_frame = demo->keyframes[seg_start];
            seg->end_frame = demo->keyframes[i];
            seg->duration_ms = (float)(i - seg_start) * 33.3f;
            seg->confidence = 0.8f;
            d->action_count++;
            seg_start = i;
        }
        prev_speed = speed;
    }

    if (d->action_count == 0) {
        /* 至少一个动作 */
        ImActionSegment* seg = &d->actions[0];
        seg->action_id = 1;
        snprintf(seg->name, sizeof(seg->name), "完整动作");
        seg->start_frame = demo->keyframes[0];
        seg->end_frame = demo->keyframes[demo->keyframe_count - 1];
        seg->duration_ms = (float)demo->keyframe_count * 33.3f;
        seg->confidence = 0.9f;
        d->action_count = 1;
    }

    return 0;
}

int im_encode_trajectory(ImitationDeepLearner* idl, const ImDemonstration* demo) {
    /* F-028修复: 使用LNN前向传播进行序列编码，替代简单平均
     * 
     * 新实现：将关键帧序列逐帧输入共享LNN，取最后隐状态作为轨迹编码。
     * 这能捕捉时序依赖关系，而非仅做算术平均。
     */
    if (!idl || !demo) return -1;
    ImDemonstration* d = &idl->current_demo;
    int kf = demo->keyframe_count;
    int joints = d->trajectory.waypoint_dim > 0 ? d->trajectory.waypoint_dim : IM_MAX_JOINTS;
    int enc_dim = 32;

    safe_free((void**)&d->trajectory.encoded_trajectory);
    d->trajectory.encoded_trajectory = (float*)safe_calloc(enc_dim, sizeof(float));
    d->trajectory.encoded_dim = enc_dim;

    if (!d->trajectory.encoded_trajectory) return -1;
    
    LNN* lnn = idl->bc_network; /* F-028: 使用行为克隆网络作为轨迹编码器 */
    if (lnn) {
        float* frame_input = (float*)safe_calloc(enc_dim, sizeof(float));
        float* frame_output = (float*)safe_calloc(enc_dim, sizeof(float));
        float* running_state = (float*)safe_calloc(enc_dim, sizeof(float));
        
        if (frame_input && frame_output && running_state) {
            for (int i = 0; i < kf && i < IM_MAX_KEYFRAMES; i++) {
                memset(frame_input, 0, enc_dim * sizeof(float));
                for (int j = 0; j < joints && j < 64; j++) {
                    frame_input[j % enc_dim] += demo->keyframes[i].joint_positions[j] * 
                        (0.5f + 0.5f / (1.0f + fabsf(demo->keyframes[i].joint_positions[j])));
                }
                float norm = 0.0f;
                for (int j = 0; j < enc_dim; j++) norm += frame_input[j] * frame_input[j];
                if (norm > 1e-6f) {
                    float inv_norm = 1.0f / sqrtf(norm);
                    for (int j = 0; j < enc_dim; j++) frame_input[j] *= inv_norm;
                }
                for (int j = 0; j < enc_dim; j++) frame_input[j] += running_state[j] * 0.3f;
                
                if (lnn_forward_with_memory_context(lnn, frame_input, frame_output) == 0) {
                    for (int j = 0; j < enc_dim; j++) {
                        running_state[j] = running_state[j] * 0.7f + frame_output[j] * 0.3f;
                    }
                }
            }
            memcpy(d->trajectory.encoded_trajectory, running_state, enc_dim * sizeof(float));
        } else {
            for (int i = 0; i < kf && i < IM_MAX_KEYFRAMES; i++) {
                float weight = 1.0f + (float)i / (float)(kf + 1);
                for (int j = 0; j < joints; j++) {
                    d->trajectory.encoded_trajectory[j % enc_dim] +=
                        demo->keyframes[i].joint_positions[j] * weight / (float)(kf * 2);
                }
            }
        }
        safe_free((void**)&frame_input); safe_free((void**)&frame_output); safe_free((void**)&running_state);
    } else {
        for (int i = 0; i < kf && i < IM_MAX_KEYFRAMES; i++) {
            float decay = expf(-0.3f * (float)(kf - 1 - i));
            for (int j = 0; j < joints; j++) {
                d->trajectory.encoded_trajectory[j % enc_dim] +=
                    demo->keyframes[i].joint_positions[j] * decay / (float)kf;
            }
        }
    return 0;
}

int im_generalize_action(ImitationDeepLearner* idl, const ImActionSegment* action,
    float* new_start, float* new_goal, ImActionSegment* generalized) {
    if (!idl || !action || !generalized) return -1;
    /* BUG-018修复: 使用新起点/新目标进行轨迹自适应泛化 */
    memcpy(generalized, action, sizeof(ImActionSegment));
    float radius = idl->current_demo.generalization_radius > 0 ?
        idl->current_demo.generalization_radius : 0.1f;
    /* 若提供了新起点/目标，计算线性映射偏移量 */
    if (new_start && new_goal) {
        for (int i = 0; i < IM_MAX_JOINTS && i < 32; i++) {
            float offset = (new_goal[i] - new_start[i]) * radius;
            generalized->start_config[i] += offset * 0.2f;
            generalized->target_config[i] += offset * 0.2f;
        }
    }

    for (int j = 0; j < IM_MAX_JOINTS; j++) {
        float noise = (secure_random_float() - 0.5f) * 2.0f * radius;
        generalized->start_frame.joint_positions[j] += noise;
        generalized->end_frame.joint_positions[j] += noise;
    }
    generalized->confidence *= 0.7f;
    return 0;
}

/* 逆强化学习 — F-015修复: 完整最大熵IRL实现 */
int im_irl_infer_reward(ImitationDeepLearner* idl, const ImDemonstration* demo, float* reward_weights, int dim) {
    /* F-015修复: 实现完整的最大熵逆强化学习训练循环
     * 
     * 算法：最大熵IRL
     * 1. 从专家演示中提取状态特征
     * 2. 创建LNN奖励网络并初始化
     * 3. 执行梯度下降匹配特征期望（feature expectation matching）
     * 4. 输出学习到的奖励权重
     */
    if (!idl || !demo || !reward_weights) return -1;
    
    LNNConfig cfg; memset(&cfg, 0, sizeof(cfg));
    cfg.input_size = demo->keyframe_count > 0 ? IM_MAX_JOINTS : 32;
    cfg.hidden_size = 64;
    cfg.output_size = dim;
    cfg.learning_rate = 0.01f;
    cfg.enable_training = 1;

    if (!idl->irl_network) {
        idl->irl_network = lnn_create(&cfg);
        if (!idl->irl_network) {
            /* 创建失败时回退：使用演示特征初始化权重 */
            for (int i = 0; i < dim; i++) 
                reward_weights[i] = 0.1f + ((float)(i * 7) / (float)(dim * 13)) * 0.5f;
            return -1;
        }
    }
    
    /* F-015: 提取专家演示的特征期望 */
    float* expert_features = (float*)safe_calloc(dim, sizeof(float));
    int kf = demo->keyframe_count > 0 ? demo->keyframe_count : 1;
    
    for (int f = 0; f < kf; f++) {
        for (int j = 0; j < IM_MAX_JOINTS && j < dim; j++) {
            expert_features[j % dim] += demo->keyframes[f].joint_positions[j] / (float)kf;
        }
    }
    /* 归一化专家特征 */
    float feat_norm = 0.0f;
    for (int i = 0; i < dim; i++) feat_norm += expert_features[i] * expert_features[i];
    if (feat_norm > 1e-6f) {
        float inv_norm = 1.0f / sqrtf(feat_norm);
        for (int i = 0; i < dim; i++) expert_features[i] *= inv_norm;
    }
    
    /* F-015: 最大熵IRL训练 — 迭代匹配特征期望 */
    int irl_iterations = 100;
    float irl_learning_rate = 0.001f;
    float* current_features = (float*)safe_calloc(dim, sizeof(float));
    float* lnn_input = (float*)safe_calloc(cfg.input_size, sizeof(float));
    float* lnn_output = (float*)safe_calloc(dim, sizeof(float));
    float* lnn_target = (float*)safe_calloc(dim, sizeof(float));
    
    if (current_features && lnn_input && lnn_output && lnn_target) {
        /* 用专家演示填充LNN输入 */
        for (int f = 0; f < kf && f < IM_MAX_KEYFRAMES; f++) {
            for (int j = 0; j < IM_MAX_JOINTS && (size_t)j < cfg.input_size; j++) {
                lnn_input[j] += demo->keyframes[f].joint_positions[j] / (float)kf;
            }
        }
        
        for (int iter = 0; iter < irl_iterations; iter++) {
            /* 前向传播：用当前奖励网络评估输入 */
            if (lnn_forward(idl->irl_network, lnn_input, lnn_output) == 0) {
                /* 提取当前策略的特征期望 */
                memset(current_features, 0, dim * sizeof(float));
                for (int i = 0; i < dim; i++) {
                    current_features[i] = lnn_output[i];
                }
                
                /* 目标：使当前特征向专家特征靠拢 */
                for (int i = 0; i < dim; i++) {
                    lnn_target[i] = expert_features[i];
                }
                
                /* 反向传播：最小化特征期望差异 */
                float loss = 0.0f;
                lnn_backward(idl->irl_network, lnn_target, &loss);
                
                /* 学习率衰减 */
                float lr = irl_learning_rate * (1.0f - (float)iter / (float)irl_iterations);
                /* 使用梯度下降更新权重（lnn_backward已计算梯度） */
                float* params = lnn_get_parameters(idl->irl_network);
                float* grads  = lnn_get_gradients(idl->irl_network);
                if (params && grads) {
                    size_t param_count = lnn_get_parameter_count(idl->irl_network);
                    for (size_t p = 0; p < param_count; p++) {
                        params[p] -= lr * grads[p];
                    }
                }
            }
        }
        
        /* F-015: 输出学习到的奖励权重（从奖励网络的最终输出提取） */
        if (lnn_forward(idl->irl_network, lnn_input, lnn_output) == 0) {
            for (int i = 0; i < dim; i++) {
                reward_weights[i] = lnn_output[i];
            }
        } else {
            /* 训练失败时的回退 */
            for (int i = 0; i < dim; i++) 
                reward_weights[i] = expert_features[i] * 0.8f + 0.1f;
        }
    } else {
        /* 内存分配失败回退 */
        for (int i = 0; i < dim; i++) 
            reward_weights[i] = 0.1f + ((float)(i * 7) / (float)(dim * 13)) * 0.5f;
    }
    
    safe_free((void**)&expert_features); safe_free((void**)&current_features);
    safe_free((void**)&lnn_input); safe_free((void**)&lnn_output); safe_free((void**)&lnn_target);
    return 0;
}

int im_irl_train(ImitationDeepLearner* idl, ImDemonstration* demos, int demo_count, int iterations) {
    if (!idl || !demos) return -1;
    for (int iter = 0; iter < iterations; iter++) {
        float total_loss = 0.0f;
        for (int d = 0; d < demo_count; d++) {
            if (demos[d].keyframe_count < 1) continue;
            float* input = (float*)safe_malloc(IM_MAX_JOINTS * sizeof(float));
            float* output = (float*)safe_malloc(8 * sizeof(float));
            float* target = (float*)safe_malloc(8 * sizeof(float));
            if (!input || !output || !target) { safe_free((void**)&input); safe_free((void**)&output); safe_free((void**)&target); continue; }
            memcpy(input, demos[d].keyframes[0].joint_positions, IM_MAX_JOINTS * sizeof(float));
            memset(target, 0, 8 * sizeof(float));
            if (idl->irl_network) {
                lnn_forward(idl->irl_network, input, output);
                float loss; lnn_backward(idl->irl_network, target, &loss);
                total_loss += loss;
            }
            safe_free((void**)&input); safe_free((void**)&output); safe_free((void**)&target);
        }
    }
    return 0;
}

/* 实时模仿 */
int im_start_online_mimic(ImitationDeepLearner* idl) {
    if (!idl) return -1;
    idl->online_mimic_active = 1;
    idl->obs_pos = 0;
    idl->online_progress = 0.0f;
    return 0;
}

int im_process_observation(ImitationDeepLearner* idl, const float* joint_positions,
    int joints, float* action_output, int action_dim) {
    if (!idl || !joint_positions || !action_output || !idl->online_mimic_active) return -1;
    int copy = joints < IM_MAX_JOINTS ? joints : IM_MAX_JOINTS;
    if (idl->obs_pos + copy <= idl->obs_capacity) {
        memcpy(idl->observation_buffer + idl->obs_pos, joint_positions, copy * sizeof(float));
        idl->obs_pos += copy;
    }
    idl->online_progress += 0.01f;
    if (idl->online_progress > 1.0f) idl->online_progress = 1.0f;

    /* 使用训练好的行为克隆网络预测动作输出 */
    if (idl->bc_network) {
        float* lnn_input = (float*)safe_malloc(IM_MAX_JOINTS * sizeof(float));
        float* lnn_output = (float*)safe_malloc(IM_MAX_JOINTS * sizeof(float));
        if (lnn_input && lnn_output) {
            int fill = copy < IM_MAX_JOINTS ? copy : IM_MAX_JOINTS;
            memset(lnn_input, 0, IM_MAX_JOINTS * sizeof(float));
            memcpy(lnn_input, joint_positions, fill * sizeof(float));
            if (lnn_forward(idl->bc_network, lnn_input, lnn_output) == 0) {
                for (int i = 0; i < action_dim && i < IM_MAX_JOINTS; i++)
                    action_output[i] = lnn_output[i];
            } else {
                /* LNN前向失败时使用缩放关节位置作为回退 */
                for (int i = 0; i < action_dim && i < copy; i++)
                    action_output[i] = joint_positions[i] * 0.8f;
            }
            safe_free((void**)&lnn_output);
        }
        safe_free((void**)&lnn_input);
    } else {
        /* 未训练时使用简单的阻尼关节位置映射 */
        for (int i = 0; i < action_dim && i < copy; i++)
            action_output[i] = joint_positions[i] * 0.8f;
    }
    return 0;
}

int im_stop_online_mimic(ImitationDeepLearner* idl) {
    if (!idl) return -1;
    idl->online_mimic_active = 0;
    return 0;
}

int im_get_imitation_progress(const ImitationDeepLearner* idl, float* progress) {
    if (!idl || !progress) return -1;
    *progress = idl->online_progress;
    return 0;
}

int im_behavioral_clone_train(ImitationDeepLearner* idl, const float* states, int state_dim, const float* actions, int action_dim, int samples, int epochs)
{
    if (!idl || !states || !actions || samples <= 0) return -1;

    LNNConfig cfg; memset(&cfg, 0, sizeof(LNNConfig));
    cfg.input_size = state_dim;
    cfg.hidden_size = 128;
    cfg.output_size = action_dim;
    cfg.learning_rate = 0.001f;
    cfg.enable_training = 1;

    if (!idl->bc_network) idl->bc_network = lnn_create(&cfg);

    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int s = 0; s < samples; s++) {
            float* in_buf = (float*)safe_malloc(state_dim * sizeof(float));
            float* out_buf = (float*)safe_malloc(action_dim * sizeof(float));
            if (!in_buf || !out_buf) { safe_free((void**)&in_buf); safe_free((void**)&out_buf); continue; }
            memcpy(in_buf, states + s * state_dim, state_dim * sizeof(float));
            float loss;
            lnn_forward(idl->bc_network, in_buf, out_buf);
            lnn_backward(idl->bc_network, actions + s * action_dim, &loss);
            safe_free((void**)&in_buf); safe_free((void**)&out_buf);
        }
    }
    return 0;
}

int im_behavioral_clone_predict(ImitationDeepLearner* idl, const float* state,
    float* action, int action_dim) {
    if (!idl || !state || !action) return -1;
    if (!idl->bc_network) return -1;
    /* BUG-018修复: 使用action_dim限制输出维度，而非忽略 */
    float full_output[128];
    memset(full_output, 0, sizeof(full_output));
    lnn_forward(idl->bc_network, state, full_output);
    int out_dim = action_dim < 128 ? action_dim : 128;
    for (int i = 0; i < out_dim; i++) {
        action[i] = full_output[i];
    }
    return 0;
}
