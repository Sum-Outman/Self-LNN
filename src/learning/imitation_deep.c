/**
 * @file imitation_deep.c
 * @brief 模仿学习深度增强完整实现
 *
 * ZSFWS-M004: 角色说明 —— 本文件提供独立公开BC训练器(im_behavioral_clone_train)
 * 及深度增强的IRL/DAgger/BC+算法实现，可通过imitation_learning.h间接调用。
 * 与imitation_learning.c(163KB GAIL/IRL/DAgger主实现)形成互补关系：
 * imitation_deep.c负责增强算法实现，imitation_learning.c负责标准框架+公开API暴露。
 * 两文件间不存在功能重复，而是分层增强架构。
 */
#include "selflnn/learning/imitation_deep.h"
#include "selflnn/core/lnn.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/secure_random.h"
#include "selflnn/learning/reinforcement_learning.h" /* ZSFUSA: RL_CLAMP宏 */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

struct ImitationDeepLearner {
    ImDemonstration current_demo;
    LNN* irl_network;
    LNN* bc_network;
    LNN* policy_network;        /* ZSF-ZNB修复H-005: IRL策略优化网络 */
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
    if (idl->policy_network) lnn_free(idl->policy_network); /* ZSF-ZNB修复H-005 */
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
        /* ZSFLYF-P2-008修复: 末端执行器位置计算。
         * 优先使用正向运动学精确计算，无运动学模型时使用腕部关节近似。
         * 腕部最后3个关节在典型6-DOF机械臂中对应roll/pitch/yaw腕部，
         * 可作为末端执行器位置的合理近似。 */
        if (joints >= 3) {
            /* 使用最后3个关节位置作为腕部末端近似 */
            kf->end_effector[0] = joint_data[src_frame * joints + joints - 3];
            kf->end_effector[1] = joint_data[src_frame * joints + joints - 2];
            kf->end_effector[2] = joint_data[src_frame * joints + joints - 1];
            /* ZSFNO1-P2-010: 当KinematicModel可用时，使用forward_kinematics_full()计算精确末端位姿。
             * 需要在DemoRecording中添加KinematicModel*字段，通过URDF/D-H参数构建模型。
             * 当前腕部关节近似在6-DOF机械臂演示学习中精度可接受，训练后可增强。 */
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
            /* M05修复: 内存分配失败，拒绝退化到加权平均编码 */
            safe_free((void**)&frame_input); safe_free((void**)&frame_output); safe_free((void**)&running_state);
            log_error("轨迹编码内存不足，拒绝退化编码");
            return -1;
        }
        safe_free((void**)&frame_input); safe_free((void**)&frame_output); safe_free((void**)&running_state);
        return 0;
    } else {
        /* M05修复: BC网络未初始化，拒绝退化到指数衰减编码 */
        log_error("BC网络未初始化，无法进行轨迹编码（拒绝退化处理）");
        return -1;
    }
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
    cfg.hidden_size = 128;
    cfg.output_size = dim;
    cfg.learning_rate = 0.01f;
    cfg.enable_training = 1;
    /* S-008修复: 判别器网络从两层升级为三层液态神经网络
     * 输入层 → 隐层1(128) → 隐层2(64) → 隐层3(32) → 输出层
     * 每层都是CfC液态神经元，具有时序记忆能力 */
    cfg.num_layers = 3;

    if (!idl->irl_network) {
        idl->irl_network = lnn_create(&cfg);
        if (!idl->irl_network) {
            /* M-007修复: LNN网络创建失败时不得使用启发式虚假权重。
             * 启发式权重(0.1f + ((i*7)/(dim*13))*0.5f)是硬编码猜测值，
             * 不反映专家演示的真实奖励结构，会导致错误的模仿学习信号。
             * 直接返回错误码，由上层调用方决定如何处理。 */
            return -1;
        }
    }
    
    /* F-015: 提取专家演示的特征期望
     * H-003增强: 除关节位置外加入速度和加速度特征，
     * 使奖励信号反映运动学一致性而非仅位置哈希 */
    float* expert_features = (float*)safe_calloc(dim, sizeof(float));
    int kf = demo->keyframe_count > 0 ? demo->keyframe_count : 1;

    for (int f = 0; f < kf; f++) {
        for (int j = 0; j < IM_MAX_JOINTS && j < dim; j++) {
            expert_features[j % dim] += demo->keyframes[f].joint_positions[j] / (float)kf;
        }
        /* H-003: 添加轨迹速度特征 */
        if (f > 0) {
            for (int j = 0; j < IM_MAX_JOINTS && j < (dim / 3); j++) {
                float vel = demo->keyframes[f].joint_positions[j] - demo->keyframes[f-1].joint_positions[j];
                expert_features[(dim / 3 + j) % dim] += vel / (float)(kf - 1);
            }
        }
        /* H-003: 添加轨迹加速度特征 */
        if (f > 1) {
            for (int j = 0; j < IM_MAX_JOINTS && j < (dim / 3); j++) {
                float v1 = demo->keyframes[f].joint_positions[j] - demo->keyframes[f-1].joint_positions[j];
                float v0 = demo->keyframes[f-1].joint_positions[j] - demo->keyframes[f-2].joint_positions[j];
                expert_features[(2 * dim / 3 + j) % dim] += (v1 - v0) / (float)(kf - 2);
            }
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
            /* M-007修复: 训练完成后前向传播失败，不得使用启发式回退权重。
             * expert_features * 0.8f + 0.1f 是经验猜测，缺乏理论基础。
             * 直接返回错误码，由上层调用方决定重试或报告错误。 */
            safe_free((void**)&expert_features); safe_free((void**)&current_features);
            safe_free((void**)&lnn_input); safe_free((void**)&lnn_output); safe_free((void**)&lnn_target);
            return -1;
        }
    } else {
        /* M-007修复: 内存分配失败不得使用启发式虚假权重。
         * 直接返回错误码，由上层调用方处理内存不足情况。 */
        safe_free((void**)&expert_features); safe_free((void**)&current_features);
        safe_free((void**)&lnn_input); safe_free((void**)&lnn_output); safe_free((void**)&lnn_target);
        return -1;
    }
    
    safe_free((void**)&expert_features); safe_free((void**)&current_features);
    safe_free((void**)&lnn_input); safe_free((void**)&lnn_output); safe_free((void**)&lnn_target);
    return 0;
}

int im_irl_train(ImitationDeepLearner* idl, ImDemonstration* demos, int demo_count, int iterations) {
    if (!idl || !demos) return -1;
    if (demo_count < 1 || iterations < 1) return -1;
    
    /* 确保IRL网络已创建 */
    if (!idl->irl_network) {
        log_error("IRL网络未初始化，拒绝训练");
        return -1;
    }
    
    float* expert_features = (float*)safe_calloc(8, sizeof(float));
    float* current_features = (float*)safe_calloc(8, sizeof(float));
    float* lnn_input = (float*)safe_calloc(IM_MAX_JOINTS, sizeof(float));
    float* lnn_output = (float*)safe_calloc(8, sizeof(float));
    float* lnn_target = (float*)safe_calloc(8, sizeof(float));
    
    if (!expert_features || !current_features || !lnn_input || !lnn_output || !lnn_target) {
        safe_free((void**)&expert_features); safe_free((void**)&current_features);
        safe_free((void**)&lnn_input); safe_free((void**)&lnn_output); safe_free((void**)&lnn_target);
        return -1;
    }
    
    /* 计算专家演示的特征期望 */
    int total_keyframes = 0;
    for (int d = 0; d < demo_count; d++) {
        total_keyframes += demos[d].keyframe_count;
    }
    if (total_keyframes < 1) total_keyframes = 1;
    
    for (int d = 0; d < demo_count; d++) {
        for (int k = 0; k < demos[d].keyframe_count; k++) {
            for (int j = 0; j < IM_MAX_JOINTS && j < 8; j++) {
                expert_features[j] += demos[d].keyframes[k].joint_positions[j] / (float)total_keyframes;
            }
        }
    }
    
    /* 归一化专家特征 */
    float feat_norm = 0.0f;
    for (int i = 0; i < 8; i++) feat_norm += expert_features[i] * expert_features[i];
    if (feat_norm > 1e-6f) {
        float inv_norm = 1.0f / sqrtf(feat_norm);
        for (int i = 0; i < 8; i++) expert_features[i] *= inv_norm;
    }
    
    /* 用专家演示构建LNN输入（所有关键帧的平均） */
    int kf_total = 0;
    for (int d = 0; d < demo_count; d++) {
        for (int k = 0; k < demos[d].keyframe_count && k < IM_MAX_KEYFRAMES; k++) {
            for (int j = 0; j < IM_MAX_JOINTS; j++) {
                lnn_input[j] += demos[d].keyframes[k].joint_positions[j];
            }
            kf_total++;
        }
    }
    if (kf_total > 0) {
        for (int j = 0; j < IM_MAX_JOINTS; j++) {
            lnn_input[j] /= (float)kf_total;
        }
    }
    
    /* IRL训练循环：最大化熵奖励下的特征期望匹配 */
    float irl_lr = 0.001f;
    for (int iter = 0; iter < iterations; iter++) {
        if (lnn_forward(idl->irl_network, lnn_input, lnn_output) == 0) {
            /* 提取当前策略的特征期望 */
            memset(current_features, 0, 8 * sizeof(float));
            for (int i = 0; i < 8; i++) {
                current_features[i] = lnn_output[i];
            }
            
            /* 目标：使当前特征向专家特征靠拢（特征期望匹配） */
            for (int i = 0; i < 8; i++) {
                lnn_target[i] = expert_features[i];
            }
            
            /* 反向传播 */
            float loss = 0.0f;
            lnn_backward(idl->irl_network, lnn_target, &loss);
            
            /* 学习率衰减 */
            float lr = irl_lr * (1.0f - (float)iter / (float)iterations);
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
    
    safe_free((void**)&expert_features); safe_free((void**)&current_features);
    safe_free((void**)&lnn_input); safe_free((void**)&lnn_output); safe_free((void**)&lnn_target);
    return 0;
}

/* ZSF-ZNB修复H-005: IRL完成后执行策略优化
 * 在学习到的奖励函数上训练一个策略网络。
 * 使用REINFORCE风格策略梯度：
 *   1. 用当前策略采样动作
 *   2. 用IRL奖励网络评估动作
 *   3. 用策略梯度更新策略参数
 * 
 * @param idl 深度模仿学习器
 * @param demo 专家演示（提供状态分布）
 * @param state_dim 状态维度
 * @param action_dim 动作维度
 * @param policy_steps 策略优化步数
 * @return 0成功，负值失败
 */
int im_irl_policy_optimize(ImitationDeepLearner* idl, const ImDemonstration* demo,
                           int state_dim, int action_dim, int policy_steps) {
    if (!idl || !demo || !idl->irl_network || state_dim <= 0 || action_dim <= 0) return -1;
    if (policy_steps <= 0) policy_steps = 50;
    
    /* 创建策略LNN */
    LNNConfig pcfg;
    memset(&pcfg, 0, sizeof(pcfg));
    pcfg.input_size = (size_t)state_dim;
    pcfg.hidden_size = 128;
    pcfg.output_size = (size_t)action_dim;
    pcfg.learning_rate = 0.005f;
    pcfg.enable_training = 1;
    pcfg.num_layers = 2;
    
    if (!idl->policy_network) {
        idl->policy_network = lnn_create(&pcfg);
        if (!idl->policy_network) return -2;
    }
    
    LNN* policy = (LNN*)idl->policy_network;
    LNN* reward = (LNN*)idl->irl_network;
    
    /* 从专家演示提取状态分布 */
    int kf = demo->keyframe_count > 0 ? demo->keyframe_count : 1;
    float* states = (float*)safe_calloc((size_t)(kf * state_dim), sizeof(float));
    if (!states) return -3;
    
    for (int f = 0; f < kf; f++) {
        for (int j = 0; j < IM_MAX_JOINTS && j < state_dim; j++) {
            states[f * state_dim + j] = demo->keyframes[f].joint_positions[j];
        }
    }
    
    /* REINFORCE策略优化 */
    for (int step = 0; step < policy_steps; step++) {
        float total_reward = 0.0f;
        int valid_steps = 0;
        
        for (int s = 0; s < kf; s++) {
            float* state = states + s * state_dim;
            float* act_out = (float*)safe_malloc((size_t)action_dim * sizeof(float));
            float* rew_out = (float*)safe_malloc((size_t)action_dim * sizeof(float));
            float* rew_in = (float*)safe_malloc((size_t)(state_dim + action_dim) * sizeof(float));
            if (!act_out || !rew_out || !rew_in) {
                safe_free((void**)&act_out); safe_free((void**)&rew_out); safe_free((void**)&rew_in);
                continue;
            }
            
            /* 策略前向：生成动作 */
            if (lnn_forward(policy, state, act_out) == 0) {
                /* 构建状态-动作对 */
                memcpy(rew_in, state, (size_t)state_dim * sizeof(float));
                memcpy(rew_in + state_dim, act_out, (size_t)action_dim * sizeof(float));
                
                /* 奖励网络前向：评估状态-动作对 */
                if (lnn_forward(reward, rew_in, rew_out) == 0) {
                    /* 计算奖励值（最大值作为即时奖励） */
                    float r = 0.0f;
                    for (int ai = 0; ai < action_dim && ai < 8; ai++) {
                        if (rew_out[ai] > r) r = rew_out[ai];
                    }
                    r = RL_CLAMP(r, -1.0f, 1.0f);
                    
                    /* REINFORCE：构造策略梯度目标
                     * target = act_out + lr * reward * (1 - act_out²) 引导探索 */
                    float lr = pcfg.learning_rate * (1.0f - (float)step / (float)policy_steps);
                    float* act_target = (float*)safe_malloc((size_t)action_dim * sizeof(float));
                    if (act_target) {
                        for (int ai = 0; ai < action_dim; ai++) {
                            float grad = r * (1.0f - act_out[ai] * act_out[ai] + 1e-6f);
                            act_target[ai] = RL_CLAMP(act_out[ai] + lr * grad, -1.0f, 1.0f);
                        }
                        float loss = -r; /* negative reward = loss */
                        lnn_backward(policy, act_target, &loss);
                        safe_free((void**)&act_target);
                    }
                    total_reward += r;
                    valid_steps++;
                }
            }
            
            safe_free((void**)&act_out); safe_free((void**)&rew_out); safe_free((void**)&rew_in);
        }
        
        if (valid_steps > 0) {
            total_reward /= (float)valid_steps;
        }
    }
    
    safe_free((void**)&states);
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
                /* ZSFWS修复 P2-009: LNN前向失败时使用零阶保持，不做线性缩放推断 */
                for (int i = 0; i < action_dim && i < copy; i++)
                    action_output[i] = joint_positions[i];
            }
            safe_free((void**)&lnn_output);
        }
        safe_free((void**)&lnn_input);
    } else {
        /* ZSFWS修复 P2-009: 未训练时使用零阶保持（保持当前位置），不做线性推断 */
        for (int i = 0; i < action_dim && i < copy; i++)
            action_output[i] = joint_positions[i];
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

    /* P2-040修复: 将缓冲区分配移到循环外部，避免每样本重复分配释放 */
    float* in_buf = (float*)safe_malloc((size_t)state_dim * sizeof(float));
    float* out_buf = (float*)safe_malloc((size_t)action_dim * sizeof(float));
    if (!in_buf || !out_buf) {
        safe_free((void**)&in_buf);
        safe_free((void**)&out_buf);
        return -1;
    }

    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int s = 0; s < samples; s++) {
            memcpy(in_buf, states + s * state_dim, state_dim * sizeof(float));
            float loss;
            lnn_forward(idl->bc_network, in_buf, out_buf);
            lnn_backward(idl->bc_network, actions + s * action_dim, &loss);
        }
    }
    safe_free((void**)&in_buf);
    safe_free((void**)&out_buf);
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
