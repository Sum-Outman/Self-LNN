#ifndef SELFLNN_GAIT_GENERATOR_H
#define SELFLNN_GAIT_GENERATOR_H

#include "selflnn/robot/kinematics.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * 步态生成器 - 人形机器人双足行走步态规划
 * 基于ZMP预览控制 + LIPM线性倒立摆 + Bézier曲线足部轨迹插值
 * 支持前进/后退/侧行/转弯，可配置步长、步高、步行速度
 * ============================================================================ */

#define GAIT_MAX_FOOTSTEPS_PER_CYCLE 64
#define GAIT_MAX_LEGS 16
#define GAIT_ZMP_SUPPORT_POINTS 32
#define GAIT_BEZIER_CONTROL_POINTS 8
#define GAIT_IK_MAX_ITER 100
#define GAIT_IK_TOLERANCE 1e-6f
#define GAIT_LIPM_PREVIEW_HORIZON 20

/* --------------------------------------------------------------------------
 * 步态方向枚举
 * -------------------------------------------------------------------------- */
typedef enum {
    GAIT_FORWARD = 0,       /* 前进 */
    GAIT_BACKWARD = 1,      /* 后退 */
    GAIT_LEFTWARD = 2,      /* 左侧行 */
    GAIT_RIGHTWARD = 3,     /* 右侧行 */
    GAIT_TURN_LEFT = 4,     /* 左转弯 */
    GAIT_TURN_RIGHT = 5,    /* 右转弯 */
    GAIT_DIAGONAL_FL = 6,   /* 左前斜向 */
    GAIT_DIAGONAL_FR = 7,   /* 右前斜向 */
    GAIT_IN_PLACE = 8       /* 原地踏步 */
} GaitDirection;

/* --------------------------------------------------------------------------
 * 步态相位枚举
 * -------------------------------------------------------------------------- */
typedef enum {
    GAIT_PHASE_DOUBLE_SUPPORT = 0,   /* 双支撑相 */
    GAIT_PHASE_LEFT_SWING = 1,       /* 左腿摆动相 */
    GAIT_PHASE_RIGHT_SWING = 2,      /* 右腿摆动相 */
    GAIT_PHASE_STOPPED = 3           /* 停止 */
} GaitPhase;

/* --------------------------------------------------------------------------
 * 摆动腿轨迹类型
 * -------------------------------------------------------------------------- */
typedef enum {
    SWING_TRAJ_BEZIER = 0,       /* Bézier曲线 */
    SWING_TRAJ_CUBIC_SPLINE = 1, /* 三次样条插值 */
    SWING_TRAJ_CYCLOID = 2,      /* 摆线 */
    SWING_TRAJ_MIN_JERK = 3      /* 最小加加速度 */
} SwingTrajectoryType;

/* --------------------------------------------------------------------------
 * 支撑多边形顶点
 * -------------------------------------------------------------------------- */
typedef struct {
    float x; /* 支撑域顶点X坐标 (m) */
    float y; /* 支撑域顶点Y坐标 (m) */
} SupportVertex;

/* --------------------------------------------------------------------------
 * 步态配置参数
 * -------------------------------------------------------------------------- */
typedef struct {
    float step_length;              /* 步长 (m)，默认0.3 */
    float step_height;              /* 抬脚高度 (m)，默认0.08 */
    float step_width;               /* 左右脚间距 (m)，默认0.15 */
    float step_frequency;           /* 步频 (Hz)，默认2.0 */
    float double_support_ratio;     /* 双支撑相占周期比例，默认0.2 */
    float com_height;               /* 质心高度 (m)，默认0.8 */
    float foot_clearance;           /* 脚底离地安全距离 (m)，默认0.02 */
    float gravity;                  /* 重力加速度 (m/s²)，默认9.81 */
    float max_step_length;          /* 最大步长 (m)，默认0.4 */
    float max_turn_angle;           /* 最大转弯角 (rad/步)，默认0.3 */
    float zmp_safety_margin;        /* ZMP安全边界 (m)，默认0.03 */
    float force_landing_threshold;  /* 落地力传感器阈值 (N)，默认50.0 */
    float swing_duty_ratio;         /* 摆动相占空比，默认0.5 */
    int enable_zmp_preview;         /* 是否启用ZMP预览控制，1=启用 */
    int enable_com_compensation;    /* 是否启用质心补偿，1=启用 */
    int ik_max_iterations;          /* IK求解最大迭代次数 */
    float ik_tolerance;             /* IK求解容差 */
    float hip_ankle_offset;         /* 髋-踝水平偏移 (m)，默认0.0 */
    float hip_yaw_offset;           /* 髋偏航角度偏移 (rad) */
} GaitConfig;

/* --------------------------------------------------------------------------
 * 脚步（单脚着陆状态）
 * -------------------------------------------------------------------------- */
typedef struct {
    Vec3 position;                  /* 脚步世界坐标位置 (m) */
    float orientation;              /* 脚步偏航角 (rad) */
    int leg_index;                  /* 腿编号（0=左腿, 1=右腿） */
    int is_support;                 /* 1=支撑脚, 0=摆动脚 */
    float time_start;               /* 该步开始时间 (s) */
    float time_end;                 /* 该步结束时间 (s) */
    int step_index;                 /* 步序编号 */
    float contact_force;            /* 接触力传感器读数 (N) */
    int landed;                     /* 1=已落地 */
} Footstep;

/* --------------------------------------------------------------------------
 * ZMP零力矩点
 * -------------------------------------------------------------------------- */
typedef struct {
    float x;          /* ZMP X坐标 (m) */
    float y;          /* ZMP Y坐标 (m) */
    float z;          /* ZMP Z坐标（地面=0） */
    float pressure;   /* ZMP点压力值 (N) */
    float cop_x;      /* 压力中心X (m) */
    float cop_y;      /* 压力中心Y (m) */
    int is_valid;     /* ZMP是否在支撑域内 */
} ZMPPoint;

/* --------------------------------------------------------------------------
 * LIPM线性倒立摆状态
 * -------------------------------------------------------------------------- */
typedef struct {
    float com_x;        /* 质心X位置 (m) */
    float com_y;        /* 质心Y位置 (m) */
    float com_z;        /* 质心Z高度 (m)，恒定 */
    float com_vx;       /* 质心X速度 (m/s) */
    float com_vy;       /* 质心Y速度 (m/s) */
    float com_ax;       /* 质心X加速度 (m/s²) */
    float com_ay;       /* 质心Y加速度 (m/s²) */
    float zmp_x;        /* 当前ZMP X参考 (m) */
    float zmp_y;        /* 当前ZMP Y参考 (m) */
    float zmp_ref_x[GAIT_LIPM_PREVIEW_HORIZON]; /* ZMP X预览轨迹 */
    float zmp_ref_y[GAIT_LIPM_PREVIEW_HORIZON]; /* ZMP Y预览轨迹 */
    float com_ref_x[GAIT_LIPM_PREVIEW_HORIZON]; /* CoM X预览轨迹 */
    float com_ref_y[GAIT_LIPM_PREVIEW_HORIZON]; /* CoM Y预览轨迹 */
    float omega;        /* LIPM自然频率 ω = sqrt(g/zc) (rad/s) */
    int preview_index;  /* 当前预览步序 */
} LIPMState;

/* --------------------------------------------------------------------------
 * 步态周期
 * -------------------------------------------------------------------------- */
typedef struct {
    Footstep footsteps[GAIT_MAX_FOOTSTEPS_PER_CYCLE]; /* 脚步序列 */
    int num_footsteps;          /* 当前脚步数量 */
    float cycle_time;           /* 完整周期时间 (s) */
    float elapsed_time;         /* 当前周期已过时间 (s) */
    float single_support_time;  /* 单支撑相时间 (s) */
    float double_support_time;  /* 双支撑相时间 (s) */
    int current_step_index;     /* 当前步序 */
    int cycle_complete;         /* 1=周期完成 */
    GaitPhase current_phase;    /* 当前相位 */
} GaitCycle;

/* --------------------------------------------------------------------------
 * 行走参数（高层指令）
 * -------------------------------------------------------------------------- */
typedef struct {
    Vec3 direction;             /* 行走方向向量（单位向量） */
    float speed;                /* 行走速度 (m/s)，默认0.5 */
    float turn_rate;            /* 转弯速率 (rad/s)，默认0.0 */
    GaitDirection gait_type;    /* 步态类型 */
    float target_distance;      /* 目标行走距离 (m)，-1=无限 */
    float target_angle;         /* 目标转角度 (rad)，0=不限 */
    int num_steps;              /* 目标步数，-1=无限 */
    int stop_after_cycle;       /* 完成当前周期后停止，1=是 */
} WalkParameter;

/* --------------------------------------------------------------------------
 * 步态生成器实例（不透明类型）
 * -------------------------------------------------------------------------- */
typedef struct GaitGenerator GaitGenerator;

/* ============================================================================
 * 步态状态（对外暴露的步态运行状态）
 * ============================================================================ */
typedef struct {
    GaitPhase phase;                    /* 当前行走相位 */
    float phase_progress;               /* 当前相位进度 (0~1) */
    float left_foot_pos[3];             /* 左脚位置 (x,y,z) */
    float right_foot_pos[3];            /* 右脚位置 (x,y,z) */
    float com_position[3];              /* 质心位置 (x,y,z) */
    float com_velocity[3];              /* 质心速度 (vx,vy,vz) */
    float zmp[2];                       /* 当前ZMP (x,y) */
    float left_foot_target[3];          /* 左脚目标落点 */
    float right_foot_target[3];         /* 右脚目标落点 */
    int left_foot_on_ground;            /* 1=左脚着地 */
    int right_foot_on_ground;           /* 1=右脚着地 */
    float left_foot_force;              /* 左脚受力 (N) */
    float right_foot_force;             /* 右脚受力 (N) */
    float cycle_time;                   /* 当前周期总时间 (s) */
    float elapsed_time;                 /* 周期已过时间 (s) */
    int step_count;                     /* 已完成步数 */
    float stability_margin;             /* ZMP稳定裕度 (m) */
    int is_stable;                      /* 1=当前步态稳定 */
    float left_joint_angles[6];         /* 左腿6维关节角度 (rad) */
    float right_joint_angles[6];        /* 右腿6维关节角度 (rad) */
    float swing_foot_trajectory[3];     /* 当前摆动脚笛卡尔位置 */
    float swing_foot_velocity[3];       /* 当前摆动脚速度 */
} GaitState;

/* ============================================================================
 * 对外API函数声明
 * ============================================================================ */

/**
 * @brief 创建步态生成器实例
 * @param config 步态配置参数（NULL则使用默认配置）
 * @param kin_model 运动学模型（用于IK解算）
 * @return 步态生成器句柄，失败返回NULL
 */
GaitGenerator* gait_generator_create(const GaitConfig* config, const KinematicModel* kin_model);

/**
 * @brief 释放步态生成器实例
 * @param gen 步态生成器句柄
 */
void gait_generator_free(GaitGenerator* gen);

/**
 * @brief 生成行走周期 - 计算完整的双足行走步态循环
 * @param gen 步态生成器句柄
 * @param params 行走参数
 * @param cycle 输出步态周期
 * @return 0=成功, -1=失败
 */
int gait_generate_walk_cycle(GaitGenerator* gen, const WalkParameter* params, GaitCycle* cycle);

/**
 * @brief 计算零力矩点(ZMP) - 基于多面体支撑域方法
 * @param gen 步态生成器句柄
 * @param support_vertices 支撑多边形顶点数组
 * @param num_vertices 顶点数量
 * @param forces 各支撑点的力 (N)
 * @param num_forces 力数组大小
 * @param zmp 输出ZMP点
 * @return 0=ZMP在支撑域内, 1=ZMP在安全边界内, -1=ZMP在支撑域外(即将倾倒)
 */
int gait_compute_zmp(GaitGenerator* gen,
                      const SupportVertex* support_vertices, int num_vertices,
                      const float* forces, int num_forces,
                      ZMPPoint* zmp);

/**
 * @brief 线性倒立摆(LIPM)更新 - 前向和侧向动力学
 * @param gen 步态生成器句柄
 * @param lipm 当前LIPM状态（输入输出）
 * @param dt 时间步长 (s)
 * @param target_zmp_x ZMP X目标 (m)
 * @param target_zmp_y ZMP Y目标 (m)
 * @return 0=成功, -1=失败
 */
int gait_lipm_update(GaitGenerator* gen, LIPMState* lipm,
                      float dt, float target_zmp_x, float target_zmp_y);

/**
 * @brief 足部轨迹规划 - 使用Bézier曲线/三次样条插值
 * @param gen 步态生成器句柄
 * @param start_pos 起始脚位置 (x,y,z)
 * @param end_pos 目标脚位置 (x,y,z)
 * @param t 归一化时间 (0~1)
 * @param step_height 抬脚高度 (m)
 * @param traj_type 轨迹类型
 * @param out_pos 输出足部位置 (x,y,z)
 * @param out_vel 输出足部速度 (vx,vy,vz)，NULL则忽略
 * @return 0=成功, -1=失败
 */
int gait_foot_trajectory(GaitGenerator* gen,
                          const float* start_pos, const float* end_pos,
                          float t, float step_height,
                          SwingTrajectoryType traj_type,
                          float* out_pos, float* out_vel);

/**
 * @brief 步态规划 - 基于ZMP预览控制规划下一步落点
 * @param gen 步态生成器句柄
 * @param params 行走参数
 * @param current_left_foot 当前左脚位置 (x,y,z)
 * @param current_right_foot 当前右脚位置 (x,y,z)
 * @param com_pos 当前质心位置
 * @param swing_leg 摆动腿编号（0=左, 1=右）
 * @param num_preview_steps 预览步数
 * @param next_footsteps 输出预测的下一步脚步位置数组
 * @return 规划步数（>0=成功, <=0=失败）
 */
int gait_step_planning(GaitGenerator* gen,
                        const WalkParameter* params,
                        const float* current_left_foot,
                        const float* current_right_foot,
                        const float* com_pos,
                        int swing_leg,
                        int num_preview_steps,
                        float* next_footsteps);

/**
 * @brief 摆动腿轨迹生成 - 生成完整的摆动相轨迹
 * @param gen 步态生成器句柄
 * @param start_pos 起始脚位置 (x,y,z)
 * @param target_pos 目标落点 (x,y,z)
 * @param swing_time 摆动时间 (s)
 * @param dt 采样时间步长 (s)
 * @param traj_type 轨迹类型
 * @param out_traj 输出轨迹点数组 [num_points * 3]
 * @param max_points 最大轨迹点数
 * @param num_points 输出实际轨迹点数
 * @return 0=成功, -1=失败
 */
int gait_generate_swing_leg_trajectory(GaitGenerator* gen,
                                        const float* start_pos,
                                        const float* target_pos,
                                        float swing_time, float dt,
                                        SwingTrajectoryType traj_type,
                                        float* out_traj,
                                        int max_points, int* num_points);

/**
 * @brief 脚部落地检测 - 基于ZMP和力传感器阈值
 * @param gen 步态生成器句柄
 * @param zmp 当前ZMP
 * @param swing_foot_force 摆动脚力传感器值 (N)，4分量 [fx,fy,fz,mag]
 * @param swing_foot_pos 摆动脚当前位置 (x,y,z)
 * @param target_landing_pos 目标落点 (x,y,z)
 * @return 1=已落地, 0=未落地, -1=异常着陆
 */
int gait_detect_foot_landing(GaitGenerator* gen,
                              const ZMPPoint* zmp,
                              const float* swing_foot_force,
                              const float* swing_foot_pos,
                              const float* target_landing_pos);

/**
 * @brief 更新步态状态 - 驱动步态状态机
 * @param gen 步态生成器句柄
 * @param dt 时间步长 (s)
 * @param state 输入输出：当前步态状态
 * @return 0=成功, -1=失败
 */
int gait_update_state(GaitGenerator* gen, float dt, GaitState* state);

/**
 * @brief 获取腿部关节角度 - 通过逆运动学(IK)求解
 * @param gen 步态生成器句柄
 * @param leg 腿编号（0=左, 1=右）
 * @param foot_target 脚部目标位置 (x,y,z)
 * @param hip_position 髋部位置 (x,y,z)
 * @param joint_angles 输出关节角度 (rad) [6维]
 * @return 0=成功, -1=IK未收敛
 */
int gait_get_leg_joint_angles(GaitGenerator* gen,
                               int leg,
                               const float* foot_target,
                               const float* hip_position,
                               float* joint_angles);

/* ============================================================================
 * 辅助函数
 * ============================================================================ */

/**
 * @brief 获取默认步态配置
 * @return 默认GaitConfig
 */
GaitConfig gait_config_default(void);

/**
 * @brief 获取默认行走参数
 * @return 默认WalkParameter（前进模式）
 */
WalkParameter walk_parameter_default(void);

/**
 * @brief 重置步态生成器状态
 * @param gen 步态生成器句柄
 */
void gait_generator_reset(GaitGenerator* gen);

/**
 * @brief 设置步态配置
 * @param gen 步态生成器句柄
 * @param config 新配置
 * @return 0=成功, -1=失败
 */
int gait_generator_set_config(GaitGenerator* gen, const GaitConfig* config);

/**
 * @brief 获取当前步态配置
 * @param gen 步态生成器句柄
 * @param config 输出配置
 * @return 0=成功, -1=失败
 */
int gait_generator_get_config(const GaitGenerator* gen, GaitConfig* config);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_GAIT_GENERATOR_H */
