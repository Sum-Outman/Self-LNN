/**
 * @file voice_motion_control.h
 * @brief 实时语音指令运动控制系统接口
 */

#ifndef SELFLNN_VOICE_MOTION_CONTROL_H
#define SELFLNN_VOICE_MOTION_CONTROL_H

#include <stddef.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    MOTION_CMD_MOVE = 0,
    MOTION_CMD_TURN = 1,
    MOTION_CMD_STOP = 2,
    MOTION_CMD_SPEED = 3,
    MOTION_CMD_GRIP = 4,
    MOTION_CMD_RELEASE = 5,
    MOTION_CMD_LIFT = 6,
    MOTION_CMD_LOWER = 7,
    MOTION_CMD_FOLLOW = 8,
    MOTION_CMD_PATROL = 9,
    MOTION_CMD_UNKNOWN = 10
} MotionCommandType;

typedef struct {
    MotionCommandType type;
    float param1;
    float param2;
    float param3;
    float confidence;
    char raw_text[256];
    time_t timestamp;
} MotionCommand;

typedef struct {
    MotionCommandType type;
    float current_value;
    float target_value;
    float progress;
    int is_executing;
    time_t start_time;
    time_t est_completion;
} MotionExecution;

typedef struct VoiceMotionControl VoiceMotionControl;

VoiceMotionControl* voice_motion_create(void);
void voice_motion_free(VoiceMotionControl* vmc);

int voice_motion_process_text(VoiceMotionControl* vmc, const char* text, MotionCommand* cmd);
int voice_motion_process_audio(VoiceMotionControl* vmc, const float* audio, size_t samples, MotionCommand* cmd);
int voice_motion_execute(VoiceMotionControl* vmc, const MotionCommand* cmd);
int voice_motion_stop(VoiceMotionControl* vmc);
int voice_motion_is_executing(const VoiceMotionControl* vmc);
int voice_motion_get_progress(VoiceMotionControl* vmc, float* progress);

/* 连续语音控制 */
int voice_motion_start_streaming(VoiceMotionControl* vmc);
int voice_motion_stream_audio(VoiceMotionControl* vmc, const float* audio, size_t samples);
int voice_motion_stop_streaming(VoiceMotionControl* vmc);

/* 安全确认 */
int voice_motion_set_safety(VoiceMotionControl* vmc, int require_confirmation);

/* 多语言词典加载 */
int voice_motion_load_dict(VoiceMotionControl* vmc, const char* dict_path);

/* LNN集成接口和动态词典管理 */
struct LNN;
int voice_motion_set_lnn(VoiceMotionControl* vmc, struct LNN* lnn);
int voice_motion_add_command(VoiceMotionControl* vmc, const char* keyword,
                              MotionCommandType cmd_type, float param1, float param2);
int voice_motion_remove_command(VoiceMotionControl* vmc, const char* keyword);
int voice_motion_list_commands(const VoiceMotionControl* vmc, char* buffer, size_t buf_size);

/* M-007修复: 上下文感知与反馈增强接口 */
typedef struct {
    float position[3];
    float orientation[3];
    float velocity[3];
    float joint_angles[6];
    float battery_level;
    float obstacle_distance;
    int is_moving;
    int gripper_state;
    time_t last_update;
} RobotContext;

int voice_motion_process_with_context(VoiceMotionControl* vmc,
                                       const char* text,
                                       const RobotContext* context,
                                       MotionCommand* cmd);
int voice_motion_execute_with_feedback(VoiceMotionControl* vmc,
                                        const MotionCommand* cmd,
                                        const RobotContext* context);
int voice_motion_get_execution_feedback(const VoiceMotionControl* vmc,
                                         char* feedback, size_t feedback_size);
int voice_motion_execute_chain(VoiceMotionControl* vmc,
                                const MotionCommand* commands,
                                size_t num_commands,
                                const RobotContext* context);

#ifdef __cplusplus
}
#endif
#endif
