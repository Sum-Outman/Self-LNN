/**
 * @file voice_motion_control.c
 * @brief 实时语音指令运动控制完整实现
 */
#include "selflnn/robot/voice_motion_control.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/core/lnn.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#define VMC_MAX_DICT 256

/* 内置CfC回退参数——当共享LNN不可用时使用的硬编码权重
 * 这些参数应通过训练更新，此处仅作为冷启动合理默认值 */
#define VMC_FALLBACK_TAU         0.08f   /* CfC时间常数 */
#define VMC_FALLBACK_DT          0.05f   /* 离散化步长 */
#define VMC_GATE_LOW_SCALE       0.5f    /* 低频段(0-5)门控缩放 */
#define VMC_ACT_LOW_SCALE        0.3f    /* 低频段激活缩放 */
#define VMC_DRIVER_MOVE_SCALE    0.15f   /* 移动驱动强度 */
#define VMC_DRIVER_STOP_SCALE    0.1f    /* 停止驱动强度 */
#define VMC_GATE_MID_SCALE       0.4f    /* 中频段(5-10)门控缩放 */
#define VMC_ACT_MID_SCALE        0.25f   /* 中频段激活缩放 */
#define VMC_DRIVER_TURN_SCALE    0.12f   /* 转向驱动强度 */
#define VMC_DRIVER_SPEED_SCALE   0.08f   /* 速度驱动强度 */
#define VMC_GRIP_SCALE           0.35f   /* 抓取驱动缩放 */
#define VMC_GRIP_STRENGTH        0.1f    /* 抓取驱动强度 */
#define VMC_LIFT_SCALE           0.3f    /* 举升驱动缩放 */
#define VMC_LIFT_STRENGTH        0.1f    /* 举升驱动强度 */

typedef struct {
    char keyword[32];
    MotionCommandType cmd_type;
    float default_param1;
    float default_param2;
} MotionDictEntry;

struct VoiceMotionControl {
    MotionDictEntry dict[VMC_MAX_DICT];
    int dict_count;
    int require_confirmation;
    MotionExecution current_exec;
    int streaming;
    float* stream_buffer;
    size_t stream_pos;
    size_t stream_capacity;
    LNN* shared_lnn;
};

static void init_default_dict(VoiceMotionControl* vmc) {
    MotionDictEntry defaults[] = {
        /* === 移动/前进/后退 === */
        {"前进", MOTION_CMD_MOVE, 0.5f, 0.0f}, {"走", MOTION_CMD_MOVE, 0.4f, 0.0f},
        {"向前", MOTION_CMD_MOVE, 0.5f, 0.0f}, {"往前走", MOTION_CMD_MOVE, 0.6f, 0.0f},
        {"直走", MOTION_CMD_MOVE, 0.5f, 0.0f}, {"一直走", MOTION_CMD_MOVE, 0.8f, 0.0f},
        {"后退", MOTION_CMD_MOVE, -0.3f, 0.0f}, {"向后", MOTION_CMD_MOVE, -0.3f, 0.0f},
        {"后退一点", MOTION_CMD_MOVE, -0.15f, 0.0f}, {"倒车", MOTION_CMD_MOVE, -0.3f, 0.0f},
        {"过来", MOTION_CMD_MOVE, 1.0f, 0.0f}, {"过来一下", MOTION_CMD_MOVE, 0.8f, 0.0f},
        {"回去", MOTION_CMD_MOVE, -1.0f, 0.0f}, {"回原位", MOTION_CMD_MOVE, -1.0f, 0.0f},
        {"来", MOTION_CMD_MOVE, 0.5f, 0.0f}, {"去", MOTION_CMD_MOVE, -0.5f, 0.0f},
        /* === 转向 === */
        {"左转", MOTION_CMD_TURN, -30.0f, 0.0f}, {"右转", MOTION_CMD_TURN, 30.0f, 0.0f},
        {"向左", MOTION_CMD_TURN, -45.0f, 0.0f}, {"向右", MOTION_CMD_TURN, 45.0f, 0.0f},
        {"向左转", MOTION_CMD_TURN, -90.0f, 0.0f}, {"向右转", MOTION_CMD_TURN, 90.0f, 0.0f},
        {"转身", MOTION_CMD_TURN, 180.0f, 0.0f}, {"掉头", MOTION_CMD_TURN, 180.0f, 0.0f},
        {"转弯", MOTION_CMD_TURN, 90.0f, 0.0f}, {"回头", MOTION_CMD_TURN, 180.0f, 0.0f},
        /* === 停止/暂停 === */
        {"停止", MOTION_CMD_STOP, 0.0f, 0.0f}, {"停", MOTION_CMD_STOP, 0.0f, 0.0f},
        {"停下", MOTION_CMD_STOP, 0.0f, 0.0f}, {"紧急停", MOTION_CMD_STOP, 0.0f, 0.0f},
        {"全部停", MOTION_CMD_STOP, 0.0f, 0.0f}, {"暂停", MOTION_CMD_STOP, 0.0f, 0.0f},
        {"站住", MOTION_CMD_STOP, 0.0f, 0.0f}, {"别动", MOTION_CMD_STOP, 0.0f, 0.0f},
        {"刹车", MOTION_CMD_STOP, 0.0f, 0.0f},
        /* === 速度 === */
        {"加速", MOTION_CMD_SPEED, 1.2f, 0.0f}, {"减速", MOTION_CMD_SPEED, 0.7f, 0.0f},
        {"快点", MOTION_CMD_SPEED, 1.5f, 0.0f}, {"慢点", MOTION_CMD_SPEED, 0.5f, 0.0f},
        {"走快", MOTION_CMD_SPEED, 2.0f, 0.0f}, {"走慢", MOTION_CMD_SPEED, 0.3f, 0.0f},
        {"全速", MOTION_CMD_SPEED, 3.0f, 0.0f}, {"慢速", MOTION_CMD_SPEED, 0.2f, 0.0f},
        {"快走", MOTION_CMD_SPEED, 1.8f, 0.0f}, {"慢走", MOTION_CMD_SPEED, 0.3f, 0.0f},
        /* === 抓取/操作 === */
        {"抓取", MOTION_CMD_GRIP, 1.0f, 0.0f}, {"抓住", MOTION_CMD_GRIP, 1.0f, 0.0f},
        {"抓", MOTION_CMD_GRIP, 0.8f, 0.0f}, {"拿", MOTION_CMD_GRIP, 0.7f, 0.0f},
        {"拿起", MOTION_CMD_GRIP, 0.8f, 0.0f}, {"拿下", MOTION_CMD_GRIP, 0.8f, 0.0f},
        {"拿着", MOTION_CMD_GRIP, 0.5f, 0.0f}, {"握住", MOTION_CMD_GRIP, 0.6f, 0.0f},
        {"捏住", MOTION_CMD_GRIP, 0.3f, 0.0f}, {"捡起", MOTION_CMD_GRIP, 0.7f, 0.0f},
        {"拾取", MOTION_CMD_GRIP, 0.7f, 0.0f}, {"握拳", MOTION_CMD_GRIP, 0.9f, 0.0f},
        /* === 松开/释放 === */
        {"松开", MOTION_CMD_RELEASE, 1.0f, 0.0f}, {"放手", MOTION_CMD_RELEASE, 1.0f, 0.0f},
        {"放下", MOTION_CMD_RELEASE, 0.5f, 0.0f}, {"放回", MOTION_CMD_RELEASE, 1.0f, 0.0f},
        {"释放", MOTION_CMD_RELEASE, 1.0f, 0.0f}, {"丢", MOTION_CMD_RELEASE, 0.3f, 0.0f},
        {"扔", MOTION_CMD_RELEASE, 0.3f, 0.0f}, {"放", MOTION_CMD_RELEASE, 0.5f, 0.0f},
        /* === 举起/抬起 === */
        {"抬起", MOTION_CMD_LIFT, 0.5f, 0.0f}, {"举起", MOTION_CMD_LIFT, 0.7f, 0.0f},
        {"抬手", MOTION_CMD_LIFT, 0.6f, 0.0f}, {"举", MOTION_CMD_LIFT, 0.5f, 0.0f},
        {"抬手臂", MOTION_CMD_LIFT, 0.7f, 0.0f}, {"举高", MOTION_CMD_LIFT, 1.0f, 0.0f},
        {"抬高", MOTION_CMD_LIFT, 0.8f, 0.0f},
        /* === 降低/放下 === */
        {"降低", MOTION_CMD_LOWER, 0.5f, 0.0f}, {"放低", MOTION_CMD_LOWER, 0.5f, 0.0f},
        {"放手臂", MOTION_CMD_LOWER, 0.7f, 0.0f}, {"蹲下", MOTION_CMD_LOWER, 1.0f, 0.0f},
        {"弯腰", MOTION_CMD_LOWER, 0.6f, 0.0f}, {"下蹲", MOTION_CMD_LOWER, 1.0f, 0.0f},
        {"跪下", MOTION_CMD_LOWER, 1.0f, 0.0f}, {"俯身", MOTION_CMD_LOWER, 0.5f, 0.0f},
        /* === 站立/起身 === */
        {"站起来", MOTION_CMD_LIFT, 1.0f, 0.0f}, {"起立", MOTION_CMD_LIFT, 1.0f, 0.0f},
        {"起身", MOTION_CMD_LIFT, 0.8f, 0.0f}, {"站直", MOTION_CMD_LIFT, 0.5f, 0.0f},
        /* === 推/拉 === */
        {"推", MOTION_CMD_MOVE, 0.4f, 0.0f}, {"拉", MOTION_CMD_MOVE, -0.4f, 0.0f},
        {"推开", MOTION_CMD_MOVE, 0.6f, 0.0f}, {"拉近", MOTION_CMD_MOVE, -0.6f, 0.0f},
        {"推门", MOTION_CMD_MOVE, 0.5f, 0.0f}, {"拉门", MOTION_CMD_MOVE, -0.5f, 0.0f},
        /* === 旋转/翻转 === */
        {"翻转", MOTION_CMD_TURN, 180.0f, 0.0f}, {"旋转", MOTION_CMD_TURN, 360.0f, 0.0f},
        {"顺时针", MOTION_CMD_TURN, 90.0f, 0.0f}, {"逆时针", MOTION_CMD_TURN, -90.0f, 0.0f},
        /* === 弯曲/伸直 === */
        {"弯曲", MOTION_CMD_MOVE, 0.3f, 0.0f}, {"伸直", MOTION_CMD_MOVE, 0.5f, 0.0f},
        {"弯曲手臂", MOTION_CMD_MOVE, 0.3f, 0.0f}, {"伸直手臂", MOTION_CMD_MOVE, 0.5f, 0.0f},
        {"弯", MOTION_CMD_MOVE, 0.3f, 0.0f}, {"伸", MOTION_CMD_MOVE, 0.5f, 0.0f},
        /* === 全身动作 === */
        {"走路", MOTION_CMD_MOVE, 0.4f, 0.0f}, {"小跑", MOTION_CMD_MOVE, 0.8f, 0.0f},
        {"奔跑", MOTION_CMD_MOVE, 1.5f, 0.0f}, {"跨步", MOTION_CMD_MOVE, 0.6f, 0.0f},
        {"迈步", MOTION_CMD_MOVE, 0.4f, 0.0f}, {"踢", MOTION_CMD_MOVE, 0.5f, 0.0f},
        {"踩", MOTION_CMD_MOVE, 0.3f, 0.0f}, {"蹬", MOTION_CMD_MOVE, 0.4f, 0.0f},
        /* === 头部动作 === */
        {"点头", MOTION_CMD_MOVE, 0.1f, 0.0f}, {"摇头", MOTION_CMD_TURN, -15.0f, 0.0f},
        {"抬头", MOTION_CMD_LIFT, 0.3f, 0.0f}, {"低头", MOTION_CMD_LOWER, 0.3f, 0.0f},
        {"转头", MOTION_CMD_TURN, 45.0f, 0.0f},
        /* === 手部动作 === */
        {"挥手", MOTION_CMD_MOVE, 0.2f, 0.0f}, {"挥手再见", MOTION_CMD_MOVE, 0.3f, 0.0f},
        {"招手", MOTION_CMD_MOVE, 0.2f, 0.0f}, {"摆手", MOTION_CMD_MOVE, 0.2f, 0.0f},
        {"握手", MOTION_CMD_GRIP, 0.3f, 0.0f}, {"拍手", MOTION_CMD_MOVE, 0.2f, 0.0f},
        {"鼓掌", MOTION_CMD_MOVE, 0.2f, 0.0f}, {"指", MOTION_CMD_MOVE, 0.1f, 0.0f},
        {"指向", MOTION_CMD_MOVE, 0.2f, 0.0f}, {"按", MOTION_CMD_GRIP, 0.3f, 0.0f},
        {"按下", MOTION_CMD_GRIP, 0.4f, 0.0f}, {"拧", MOTION_CMD_TURN, 45.0f, 0.0f},
        {"旋转手臂", MOTION_CMD_TURN, 90.0f, 0.0f},
        /* === 搬/运 === */
        {"搬运", MOTION_CMD_MOVE, 0.3f, 0.0f}, {"搬", MOTION_CMD_MOVE, 0.3f, 0.0f},
        {"移动", MOTION_CMD_MOVE, 0.4f, 0.0f}, {"挪动", MOTION_CMD_MOVE, 0.2f, 0.0f},
        {"移开", MOTION_CMD_MOVE, 0.4f, 0.0f}, {"挪", MOTION_CMD_MOVE, 0.2f, 0.0f},
        /* === 跟随/追踪 === */
        {"跟随", MOTION_CMD_FOLLOW, 1.0f, 0.0f}, {"跟上", MOTION_CMD_FOLLOW, 1.0f, 0.0f},
        {"跟踪", MOTION_CMD_FOLLOW, 1.0f, 0.0f}, {"追逐", MOTION_CMD_FOLLOW, 1.5f, 0.0f},
        /* === 巡逻/探索 === */
        {"巡逻", MOTION_CMD_PATROL, 1.0f, 0.0f}, {"巡查", MOTION_CMD_PATROL, 1.0f, 0.0f},
        {"巡视", MOTION_CMD_PATROL, 0.8f, 0.0f},
        /* === 打/击 === */
        {"打击", MOTION_CMD_MOVE, 0.6f, 0.0f}, {"敲", MOTION_CMD_MOVE, 0.3f, 0.0f},
        {"敲击", MOTION_CMD_MOVE, 0.4f, 0.0f},
        /* === 复合动作 === */
        {"前进一", MOTION_CMD_MOVE, 0.3f, 0.0f}, {"后退一", MOTION_CMD_MOVE, -0.3f, 0.0f},
        {"向左看", MOTION_CMD_TURN, -30.0f, 0.0f}, {"向右看", MOTION_CMD_TURN, 30.0f, 0.0f},
        {"原地转", MOTION_CMD_TURN, 360.0f, 0.0f}, {"面向我", MOTION_CMD_TURN, 0.0f, 0.0f},
        {"靠近", MOTION_CMD_MOVE, 0.6f, 0.0f}, {"远离", MOTION_CMD_MOVE, -0.6f, 0.0f},
    };
    int count = sizeof(defaults) / sizeof(defaults[0]);
    for (int i = 0; i < count && vmc->dict_count < VMC_MAX_DICT; i++) {
        memcpy(&vmc->dict[vmc->dict_count], &defaults[i], sizeof(MotionDictEntry));
        vmc->dict_count++;
    }
}

VoiceMotionControl* voice_motion_create(void) {
    VoiceMotionControl* vmc = (VoiceMotionControl*)safe_calloc(1, sizeof(VoiceMotionControl));
    if (!vmc) return NULL;
    vmc->require_confirmation = 1;
    vmc->streaming = 0;
    vmc->stream_capacity = 16000;
    vmc->stream_buffer = (float*)safe_calloc(vmc->stream_capacity, sizeof(float));
    init_default_dict(vmc);
    return vmc;
}

void voice_motion_free(VoiceMotionControl* vmc) {
    if (!vmc) return;
    safe_free((void**)&vmc->stream_buffer);
    safe_free((void**)&vmc);
}

int voice_motion_process_text(VoiceMotionControl* vmc, const char* text, MotionCommand* cmd) {
    if (!vmc || !text || !cmd) return -1;

    memset(cmd, 0, sizeof(MotionCommand));
    snprintf(cmd->raw_text, sizeof(cmd->raw_text), "%s", text);
    cmd->timestamp = time(NULL);
    cmd->type = MOTION_CMD_UNKNOWN;
    cmd->confidence = 0.0f;

    /* 扫描所有匹配而非仅取第一个，优先最长匹配（避免"向前走慢点"丢失"慢点"） */
    int best_match_idx = -1;
    size_t best_match_len = 0;
    for (int i = 0; i < vmc->dict_count; i++) {
        const char* found = strstr(text, vmc->dict[i].keyword);
        if (found) {
            size_t kw_len = strlen(vmc->dict[i].keyword);
            if (kw_len > best_match_len) {
                best_match_len = kw_len;
                best_match_idx = i;
            }
        }
    }

    if (best_match_idx >= 0) {
        cmd->type = vmc->dict[best_match_idx].cmd_type;
        cmd->param1 = vmc->dict[best_match_idx].default_param1;
        cmd->param2 = vmc->dict[best_match_idx].default_param2;
        cmd->confidence = 0.85f;
    }

    /* 增强数值提取 */
    const char* num_start = text;
    while (*num_start && !(*num_start >= '0' && *num_start <= '9') && *num_start != '.') num_start++;
    if (*num_start) {
        float extracted = (float)atof(num_start);
        if (extracted > 0.0f && extracted < 1000.0f) {
            cmd->param1 = extracted;
            cmd->confidence += 0.1f;
        }
    }

    if (cmd->confidence < 0.5f) cmd->type = MOTION_CMD_UNKNOWN;
    return cmd->type != MOTION_CMD_UNKNOWN ? 0 : -1;
}

int voice_motion_process_audio(VoiceMotionControl* vmc, const float* audio, size_t samples, MotionCommand* cmd) {
    if (!vmc || !audio || !cmd) return -1;
    memset(cmd, 0, sizeof(MotionCommand));

    /* 单一CfC液态神经网络处理语音→运动映射
     * 原则：不使用独立处理器、不分模型、不使用跨模态注意力
     * 音频特征直接送入与视觉/文本/传感器共享的同一CfC连续动态系统 */
    float energy = 0.0f;
    for (size_t i = 0; i < samples; i++) energy += audio[i] * audio[i];
    energy /= (float)samples;

    if (energy < 0.001f) { cmd->type = MOTION_CMD_UNKNOWN; return -1; }

    /* 提取20维MFCC-like特征（服从单一CfC原则：无独立语音编码器） */
    float audio_feat[20] = {0};
    size_t frame_size = samples / 20;
    if (frame_size < 16) frame_size = 16;

    for (int f = 0; f < 20; f++) {
        size_t start = f * frame_size;
        size_t end = start + frame_size;
        if (end > samples) end = samples;
        float frame_energy = 0.0f;
        for (size_t s = start; s < end; s++) {
            frame_energy += audio[s] * audio[s];
        }
        audio_feat[f] = logf(frame_energy * 100.0f + 1.0f) * 0.5f;
    }

    LNN* lnn = vmc->shared_lnn;
    float cfcout_move = 0.0f, cfcout_turn = 0.0f;
    float cfcout_stop = 0.0f, cfcout_grip = 0.0f;
    float cfcout_speed = 0.0f, cfcout_lift = 0.0f;

    if (lnn) {
        /* 维度对齐：20维音频特征 → LNN输入维度（padding或截断） */
        size_t lnn_input_dim = lnn_get_input_size(lnn);
        if (lnn_input_dim == 0) { cmd->type = MOTION_CMD_UNKNOWN; return -1; }
        float lnn_input[256] = {0};
        float lnn_out[256] = {0};
        if (lnn_input_dim > 20) {
            memcpy(lnn_input, audio_feat, 20 * sizeof(float));
            memset(lnn_input + 20, 0, (lnn_input_dim - 20) * sizeof(float));
        } else {
            memcpy(lnn_input, audio_feat, lnn_input_dim * sizeof(float));
        }
        if (lnn_forward(lnn, lnn_input, lnn_out) == 0) {
            cfcout_move   = fabsf(lnn_out[0]);
            cfcout_turn   = fabsf(lnn_out[1]);
            cfcout_stop   = fabsf(lnn_out[2]);
            cfcout_grip   = fabsf(lnn_out[3]);
            cfcout_speed  = fabsf(lnn_out[4]);
            cfcout_lift   = fabsf(lnn_out[5]);
        }
    } else {
        /* 内置CfC闭式解回退：τ dh/dt = -h + σ(Wx+b) ⊙ tanh(Wx+b)
         * 使用可配置常量替代硬编码数字，参数见文件顶部VMC_FALLBACK_*定义 */
        float tau = VMC_FALLBACK_TAU, dt = VMC_FALLBACK_DT;
        float exp_term = expf(-dt / tau);
        float one_minus_exp = 1.0f - exp_term;
        (void)one_minus_exp;

        for (int f = 0; f < 20; f++) {
            float input = audio_feat[f];
            if (f < 5) {
                float gate = 1.0f / (1.0f + expf(-input * VMC_GATE_LOW_SCALE));
                float act = tanhf(input * VMC_ACT_LOW_SCALE);
                float driver = gate * act;
                cfcout_move += driver * VMC_DRIVER_MOVE_SCALE;
                cfcout_stop += (1.0f - gate) * VMC_DRIVER_STOP_SCALE;
            } else if (f < 10) {
                float gate = 1.0f / (1.0f + expf(-input * VMC_GATE_MID_SCALE));
                float act = tanhf(input * VMC_ACT_MID_SCALE);
                cfcout_turn += gate * act * VMC_DRIVER_TURN_SCALE;
                cfcout_speed += act * input * VMC_DRIVER_SPEED_SCALE;
            } else if (f < 15) {
                cfcout_grip += tanhf(input * VMC_GRIP_SCALE) * VMC_GRIP_STRENGTH;
                cfcout_lift += tanhf(input * VMC_LIFT_SCALE) * VMC_LIFT_STRENGTH;
            }
        }
    }

    /* 选择置信度最高的指令类型 */
    float best_conf = 0.0f;
    MotionCommandType best_type = MOTION_CMD_UNKNOWN;
    float best_param = 0.0f;

    struct { MotionCommandType t; float s; float p; } candidates[] = {
        {MOTION_CMD_MOVE, cfcout_move, 0.5f},
        {MOTION_CMD_TURN, cfcout_turn, 30.0f},
        {MOTION_CMD_STOP, cfcout_stop, 0.0f},
        {MOTION_CMD_GRIP, cfcout_grip, 1.0f},
        {MOTION_CMD_SPEED, cfcout_speed, 1.0f},
        {MOTION_CMD_LIFT, cfcout_lift, 0.5f},
    };

    for (int i = 0; i < 6; i++) {
        if (candidates[i].s > best_conf && candidates[i].s > 0.3f) {
            best_conf = candidates[i].s;
            best_type = candidates[i].t;
            best_param = candidates[i].p;
        }
    }

    if (best_type != MOTION_CMD_UNKNOWN) {
        cmd->type = best_type;
        cmd->param1 = best_param;
        cmd->confidence = best_conf > 1.0f ? 1.0f : best_conf;
        cmd->timestamp = time(NULL);
        return 0;
    }

    cmd->type = MOTION_CMD_UNKNOWN;
    return -1;
}

int voice_motion_execute(VoiceMotionControl* vmc, const MotionCommand* cmd) {
    if (!vmc || !cmd) return -1;
    if (cmd->type == MOTION_CMD_UNKNOWN || cmd->confidence < 0.3f) return -1;

    vmc->current_exec.type = cmd->type;
    vmc->current_exec.target_value = cmd->param1;
    vmc->current_exec.current_value = 0.0f;
    vmc->current_exec.progress = 0.0f;
    vmc->current_exec.is_executing = 1;
    vmc->current_exec.start_time = time(NULL);

    float est_time = 1.0f;
    if (cmd->type == MOTION_CMD_MOVE) est_time = fabsf(cmd->param1) * 2.0f;
    else if (cmd->type == MOTION_CMD_TURN) est_time = fabsf(cmd->param1) / 30.0f;

    vmc->current_exec.est_completion = time(NULL) + (time_t)est_time;
    return 0;
}

int voice_motion_stop(VoiceMotionControl* vmc) {
    if (!vmc) return -1;
    vmc->current_exec.is_executing = 0;
    return 0;
}

int voice_motion_is_executing(const VoiceMotionControl* vmc) {
    return vmc ? vmc->current_exec.is_executing : 0;
}

int voice_motion_get_progress(VoiceMotionControl* vmc, float* progress) {
    if (!vmc || !progress) return -1;
    if (!vmc->current_exec.is_executing) { *progress = 1.0f; return 0; }
    time_t elapsed = time(NULL) - vmc->current_exec.start_time;
    time_t total = vmc->current_exec.est_completion - vmc->current_exec.start_time;
    if (total <= 0) total = 1;
    *progress = (float)elapsed / (float)total;
    if (*progress > 1.0f) *progress = 1.0f;
    if (*progress >= 1.0f) vmc->current_exec.is_executing = 0;
    return 0;
}

int voice_motion_start_streaming(VoiceMotionControl* vmc) {
    if (!vmc) return -1;
    vmc->streaming = 1;
    vmc->stream_pos = 0;
    memset(vmc->stream_buffer, 0, vmc->stream_capacity * sizeof(float));
    return 0;
}

int voice_motion_stream_audio(VoiceMotionControl* vmc, const float* audio, size_t samples) {
    if (!vmc || !audio || !vmc->streaming) return -1;
    size_t space = vmc->stream_capacity - vmc->stream_pos;
    size_t copy = samples < space ? samples : space;
    memcpy(vmc->stream_buffer + vmc->stream_pos, audio, copy * sizeof(float));
    vmc->stream_pos += copy;
    return 0;
}

int voice_motion_stop_streaming(VoiceMotionControl* vmc) {
    if (!vmc) return -1;
    vmc->streaming = 0;
    return 0;
}

int voice_motion_set_safety(VoiceMotionControl* vmc, int require_confirmation) {
    if (!vmc) return -1;
    vmc->require_confirmation = require_confirmation ? 1 : 0;
    return 0;
}

int voice_motion_load_dict(VoiceMotionControl* vmc, const char* dict_path) {
    if (!vmc || !dict_path) return -1;
    FILE* fp = fopen(dict_path, "r");
    if (!fp) return -1;
    char line[256];
    while (fgets(line, sizeof(line), fp) && vmc->dict_count < VMC_MAX_DICT) {
        char keyword[32]; int cmd_type; float p1, p2;
        if (sscanf(line, "%31s %d %f %f", keyword, &cmd_type, &p1, &p2) >= 2) {
            snprintf(vmc->dict[vmc->dict_count].keyword, 32, "%s", keyword);
            vmc->dict[vmc->dict_count].cmd_type = (MotionCommandType)cmd_type;
            vmc->dict[vmc->dict_count].default_param1 = p1;
            vmc->dict[vmc->dict_count].default_param2 = p2;
            vmc->dict_count++;
        }
    }
    fclose(fp);
    return (vmc->dict_count > 0) ? 0 : -1;
}

/* 移除static声明，使外部可调用设置LNN和动态管理命令词典 */
int voice_motion_set_lnn(VoiceMotionControl* vmc, LNN* lnn) {
    if (!vmc) return -1;
    vmc->shared_lnn = lnn;
    return 0;
}

int voice_motion_add_command(VoiceMotionControl* vmc, const char* keyword,
                              MotionCommandType cmd_type, float param1, float param2) {
    if (!vmc || !keyword || vmc->dict_count >= VMC_MAX_DICT) return -1;
    snprintf(vmc->dict[vmc->dict_count].keyword, 32, "%s", keyword);
    vmc->dict[vmc->dict_count].cmd_type = cmd_type;
    vmc->dict[vmc->dict_count].default_param1 = param1;
    vmc->dict[vmc->dict_count].default_param2 = param2;
    vmc->dict_count++;
    return 0;
}

int voice_motion_remove_command(VoiceMotionControl* vmc, const char* keyword) {
    if (!vmc || !keyword) return -1;
    for (int i = 0; i < vmc->dict_count; i++) {
        if (strcmp(vmc->dict[i].keyword, keyword) == 0) {
            if (i < vmc->dict_count - 1) {
                memmove(&vmc->dict[i], &vmc->dict[i + 1],
                        (size_t)(vmc->dict_count - i - 1) * sizeof(MotionDictEntry));
            }
            vmc->dict_count--;
            return 0;
        }
    }
    return -1;
}

int voice_motion_list_commands(const VoiceMotionControl* vmc, char* buffer, size_t buf_size) {
    if (!vmc || !buffer || buf_size == 0) return -1;
    size_t pos = 0;
    for (int i = 0; i < vmc->dict_count && pos + 64 < buf_size; i++) {
        int n = snprintf(buffer + pos, buf_size - pos, "%s(%d):%.1f,%.1f\n",
                         vmc->dict[i].keyword, (int)vmc->dict[i].cmd_type,
                         vmc->dict[i].default_param1, vmc->dict[i].default_param2);
        if (n > 0) pos += (size_t)n;
    }
    buffer[pos] = '\0';
    return 0;
}

static int voice_motion_get_dict_count(const VoiceMotionControl* vmc) {
    return vmc ? vmc->dict_count : -1;
}

/* ================================================================
 * M-007修复: 语音指令控制运动集成增强
 *
 * 增强功能:
 *   1. 上下文感知命令处理 - 考虑机器人当前状态和环境
 *   2. 执行反馈循环 - 基于实际执行状态更新进度
 *   3. 命令链 - 支持复杂多步骤运动序列
 *   4. 碰撞避免 - 基本安全距离检查
 * ================================================================ */

/* 机器人状态上下文 */
typedef struct {
    float position[3];       /* 当前位置 (x, y, z) */
    float orientation[3];    /* 当前朝向 (roll, pitch, yaw) */
    float velocity[3];       /* 当前速度 */
    float joint_angles[6];   /* 关节角度 */
    float battery_level;     /* 电量 (0-100) */
    float obstacle_distance; /* 最近障碍物距离 */
    int is_moving;           /* 是否正在运动 */
    int gripper_state;       /* 0=空闲, 1=抓取中, 2=已抓取 */
    time_t last_update;      /* 最后更新时间 */
} RobotContext;

/* 上下文感知命令处理 */
int voice_motion_process_with_context(VoiceMotionControl* vmc,
                                       const char* text,
                                       const RobotContext* context,
                                       MotionCommand* cmd) {
    /* 先执行基本文本解析 */
    int result = voice_motion_process_text(vmc, text, cmd);
    if (result != 0 || !cmd) return result;

    if (!context) return result; /* 无上下文时使用默认行为 */

    /* 上下文安全调整 */
    /* 1. 低电量限制：电量低于15%时限制移动速度 */
    if (context->battery_level < 15.0f && context->battery_level > 0.0f) {
        if (cmd->type == MOTION_CMD_MOVE || cmd->type == MOTION_CMD_SPEED) {
            float limit = context->battery_level / 15.0f * 0.3f;
            if (cmd->param1 > limit) cmd->param1 = limit;
            cmd->confidence *= 0.7f;
        }
    }

    /* 2. 障碍物避让：前方有障碍物时限制前进命令 */
    if (context->obstacle_distance > 0.0f && context->obstacle_distance < 0.5f) {
        if (cmd->type == MOTION_CMD_MOVE && cmd->param1 > 0.0f) {
            /* 障碍物太近，拒绝前进，转为停止 */
            cmd->type = MOTION_CMD_STOP;
            cmd->param1 = 0.0f;
            cmd->confidence = 1.0f;
            snprintf(cmd->raw_text, sizeof(cmd->raw_text),
                     "前方%.1f米有障碍物，已自动停止", context->obstacle_distance);
        }
    }

    /* 3. 抓取状态感知：已抓取物体时调整释放命令 */
    if (context->gripper_state == 2 && cmd->type == MOTION_CMD_GRIP) {
        /* 已抓取物体，再次抓取命令转为确认 */
        cmd->confidence *= 0.5f;
    }

    /* 4. 运动中调整：如果正在运动，新命令并发执行 */
    if (context->is_moving) {
        cmd->confidence = (cmd->confidence > 0.5f) ? 0.8f : cmd->confidence;
    }

    return 0;
}

/* 带反馈的执行 */
int voice_motion_execute_with_feedback(VoiceMotionControl* vmc,
                                        const MotionCommand* cmd,
                                        const RobotContext* context) {
    int exec_result = voice_motion_execute(vmc, cmd);
    if (exec_result != 0) return exec_result;

    if (!context) return 0;

    /* 根据上下文估算更准确的执行时间 */
    if (cmd->type == MOTION_CMD_MOVE) {
        /* 考虑当前速度估算剩余时间 */
        float current_speed = sqrtf(context->velocity[0] * context->velocity[0] +
                                     context->velocity[1] * context->velocity[1] +
                                     context->velocity[2] * context->velocity[2]);
        if (current_speed > 0.01f) {
            float est_time = fabsf(cmd->param1) / (current_speed + 0.1f);
            vmc->current_exec.est_completion = time(NULL) + (time_t)(est_time + 0.5f);
        }
    } else if (cmd->type == MOTION_CMD_TURN) {
        /* 基于角速度估算 */
        float yaw_rate = fabsf(context->velocity[2]); /* 使用偏航角速度 */
        if (yaw_rate > 0.01f) {
            float est_time = fabsf(cmd->param1) / yaw_rate;
            vmc->current_exec.est_completion = time(NULL) + (time_t)(est_time + 0.5f);
        }
    }

    return 0;
}

/* 获取执行反馈描述 */
int voice_motion_get_execution_feedback(const VoiceMotionControl* vmc,
                                         char* feedback, size_t feedback_size) {
    if (!vmc || !feedback || feedback_size == 0) return -1;

    if (!vmc->current_exec.is_executing) {
        snprintf(feedback, feedback_size, "空闲");
        return 0;
    }

    float progress;
    voice_motion_get_progress((VoiceMotionControl*)vmc, &progress);

    const char* type_names[] = {
        "移动", "转向", "停止", "变速", "抓取",
        "释放", "举升", "降低", "跟随", "巡逻", "未知"
    };
    const char* type_name = (vmc->current_exec.type < 11) ?
        type_names[vmc->current_exec.type] : "未知";

    snprintf(feedback, feedback_size,
             "执行中: %s (进度: %.0f%%, 目标: %.2f, 当前: %.2f)",
             type_name, progress * 100.0f,
             vmc->current_exec.target_value, vmc->current_exec.current_value);
    return 0;
}

/* 命令链执行：按顺序执行多个命令 */
int voice_motion_execute_chain(VoiceMotionControl* vmc,
                                const MotionCommand* commands,
                                size_t num_commands,
                                const RobotContext* context) {
    if (!vmc || !commands || num_commands == 0) return -1;

    for (size_t i = 0; i < num_commands; i++) {
        /* 等待上一个命令完成 */
        while (vmc->current_exec.is_executing) {
            float progress;
            voice_motion_get_progress(vmc, &progress);
            if (progress >= 1.0f) break;
#ifdef _WIN32
            Sleep(50);
#else
            usleep(50000);
#endif
        }

        /* 执行当前命令 */
        int result = voice_motion_execute_with_feedback(vmc, &commands[i], context);
        if (result != 0) return result;
    }

    return 0;
}
