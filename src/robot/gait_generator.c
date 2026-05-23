#include "selflnn/robot/gait_generator.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/math_utils.h"
#include "selflnn/core/errors.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define GAIT_CLAMP(v, lo, hi)  ((v) < (lo) ? (lo) : ((v) > (hi) ? (hi) : (v)))
#define GAIT_SIGN(x)           ((x) >= 0.0f ? 1.0f : -1.0f)
#define GAIT_EPSILON           1e-8f
#define GAIT_DEG2RAD(d)        ((d) * (float)M_PI / 180.0f)
#define GAIT_RAD2DEG(r)        ((r) * 180.0f / (float)M_PI)

/* ============================================================================
 * 步态生成器内部结构体（不透明）
 * ============================================================================ */
struct GaitGenerator {
    GaitConfig config;                  /* 步态配置 */
    const KinematicModel* kin_model;   /* 运动学模型引用（用于IK） */
    LIPMState lipm;                     /* LIPM状态 */
    GaitState state;                    /* 当前步态状态 */
    WalkParameter walk_params;          /* 当前行走参数 */
    GaitCycle current_cycle;            /* 当前步态周期 */
    int initialized;                    /* 是否已初始化 */

    /* ZMP计算缓冲 */
    SupportVertex zmp_vertices[GAIT_ZMP_SUPPORT_POINTS];
    float zmp_forces[GAIT_ZMP_SUPPORT_POINTS];

    /* 脚步规划缓冲 */
    float planned_footsteps[GAIT_MAX_FOOTSTEPS_PER_CYCLE * 3];

    /* 轨迹采样缓冲 */
    float traj_buffer[GAIT_MAX_FOOTSTEPS_PER_CYCLE * 3];

    /* LIPM预览控制增益矩阵 */
    float lipm_gain_x[GAIT_LIPM_PREVIEW_HORIZON];
    float lipm_gain_y[GAIT_LIPM_PREVIEW_HORIZON];

    /* 运行状态 */
    float total_distance_traveled;      /* 累计行走距离 */
    float total_turn_angle;             /* 累计转弯角度 */
    int total_step_count;               /* 累计步数 */
    int swing_leg;                      /* 当前摆动腿编号 */
    float phase_timer;                  /* 相位计时器 */
    float step_progress;                /* 单步进度 0~1 */
};

/* ============================================================================
 * 内部静态辅助函数：向量运算
 * ============================================================================ */

static void gait_vec3_copy(float* dst, const float* src) {
    dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2];
}

static void gait_vec3_add(float* dst, const float* a, const float* b) {
    dst[0] = a[0] + b[0]; dst[1] = a[1] + b[1]; dst[2] = a[2] + b[2];
}

static void gait_vec3_sub(float* dst, const float* a, const float* b) {
    dst[0] = a[0] - b[0]; dst[1] = a[1] - b[1]; dst[2] = a[2] - b[2];
}

static void gait_vec3_scale(float* dst, const float* a, float s) {
    dst[0] = a[0] * s; dst[1] = a[1] * s; dst[2] = a[2] * s;
}

static float gait_vec3_dot(const float* a, const float* b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static float gait_vec3_length(const float* a) {
    return sqrtf(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
}

static void gait_vec3_normalize(float* dst, const float* a) {
    float len = gait_vec3_length(a);
    if (len < GAIT_EPSILON) { dst[0] = 0.0f; dst[1] = 0.0f; dst[2] = 0.0f; return; }
    float inv = 1.0f / len;
    dst[0] = a[0] * inv; dst[1] = a[1] * inv; dst[2] = a[2] * inv;
}

static void gait_vec3_cross(float* dst, const float* a, const float* b) {
    dst[0] = a[1] * b[2] - a[2] * b[1];
    dst[1] = a[2] * b[0] - a[0] * b[2];
    dst[2] = a[0] * b[1] - a[1] * b[0];
}

static void gait_vec3_lerp(float* dst, const float* a, const float* b, float t) {
    dst[0] = a[0] + (b[0] - a[0]) * t;
    dst[1] = a[1] + (b[1] - a[1]) * t;
    dst[2] = a[2] + (b[2] - a[2]) * t;
}

/* ============================================================================
 * 内部静态辅助函数：二维叉积（用于凸包ZMP判定）
 * ============================================================================ */
static float gait_cross2d(float ax, float ay, float bx, float by) {
    return ax * by - ay * bx;
}

/* ============================================================================
 * 内部：点是否在凸多边形内（X-Y平面，使用叉积法）
 * ============================================================================ */
static int gait_point_in_convex_polygon(float px, float py,
                                         const SupportVertex* vertices, int n) {
    if (n < 3) return 0;
    int sign = 0;
    for (int i = 0; i < n; i++) {
        int j = (i + 1) % n;
        float cross = gait_cross2d(
            vertices[j].x - vertices[i].x,
            vertices[j].y - vertices[i].y,
            px - vertices[i].x,
            py - vertices[i].y);
        if (fabsf(cross) < GAIT_EPSILON) continue;
        int cs = (cross > 0) ? 1 : -1;
        if (sign == 0) sign = cs;
        else if (sign != cs) return 0;
    }
    return 1;
}

/* ============================================================================
 * 内部：计算点到多边形边的最短距离（用于ZMP安全裕度）
 * ============================================================================ */
static float gait_point_to_polygon_distance(float px, float py,
                                             const SupportVertex* vertices, int n) {
    if (n < 3) return FLT_MAX;
    float min_dist = FLT_MAX;
    for (int i = 0; i < n; i++) {
        int j = (i + 1) % n;
        float ax = vertices[i].x, ay = vertices[i].y;
        float bx = vertices[j].x, by = vertices[j].y;
        float dx = bx - ax, dy = by - ay;
        float len_sq = dx * dx + dy * dy;
        if (len_sq < GAIT_EPSILON) {
            float d = sqrtf((px - ax) * (px - ax) + (py - ay) * (py - ay));
            if (d < min_dist) min_dist = d;
            continue;
        }
        float t = ((px - ax) * dx + (py - ay) * dy) / len_sq;
        if (t < 0.0f) t = 0.0f;
        if (t > 1.0f) t = 1.0f;
        float px_proj = ax + t * dx;
        float py_proj = ay + t * dy;
        float d = sqrtf((px - px_proj) * (px - px_proj) + (py - py_proj) * (py - py_proj));
        if (d < min_dist) min_dist = d;
    }
    return min_dist;
}

/* ============================================================================
 * 内部：根据支撑脚位置构建支撑多边形
 * 人形机器人双足支撑时支撑域由左右脚形成的凸包构成
 * ============================================================================ */
static int gait_build_support_polygon(const float* left_foot_pos,
                                       const float* right_foot_pos,
                                       int left_on_ground,
                                       int right_on_ground,
                                       float foot_length,
                                       float foot_width,
                                       SupportVertex* vertices,
                                       int max_vertices) {
    int n = 0;
    float half_l = foot_length * 0.5f;
    float half_w = foot_width * 0.5f;

    if (left_on_ground && n + 4 <= max_vertices) {
        float cx = left_foot_pos[0], cy = left_foot_pos[1];
        vertices[n].x = cx + half_l;  vertices[n++].y = cy + half_w;
        vertices[n].x = cx + half_l;  vertices[n++].y = cy - half_w;
        vertices[n].x = cx - half_l;  vertices[n++].y = cy - half_w;
        vertices[n].x = cx - half_l;  vertices[n++].y = cy + half_w;
    }
    if (right_on_ground && n + 4 <= max_vertices) {
        float cx = right_foot_pos[0], cy = right_foot_pos[1];
        vertices[n].x = cx + half_l;  vertices[n++].y = cy + half_w;
        vertices[n].x = cx + half_l;  vertices[n++].y = cy - half_w;
        vertices[n].x = cx - half_l;  vertices[n++].y = cy - half_w;
        vertices[n].x = cx - half_l;  vertices[n++].y = cy + half_w;
    }
    return n;
}

/* ============================================================================
 * 内部：计算凸包（Graham Scan，用于通用支撑域）
 * ============================================================================ */
static int graham_scan_compare(const void* a, const void* b) {
    const SupportVertex* va = (const SupportVertex*)a;
    const SupportVertex* vb = (const SupportVertex*)b;
    if (va->x < vb->x) return -1;
    if (va->x > vb->x) return 1;
    if (va->y < vb->y) return -1;
    if (va->y > vb->y) return 1;
    return 0;
}

static int gait_compute_convex_hull(SupportVertex* pts, int n) {
    if (n <= 1) return n;
    qsort(pts, (size_t)n, sizeof(SupportVertex), graham_scan_compare);
    SupportVertex hull[GAIT_ZMP_SUPPORT_POINTS * 2];
    int k = 0;
    for (int i = 0; i < n; i++) {
        while (k >= 2) {
            float cross = gait_cross2d(
                hull[k - 1].x - hull[k - 2].x,
                hull[k - 1].y - hull[k - 2].y,
                pts[i].x - hull[k - 2].x,
                pts[i].y - hull[k - 2].y);
            if (cross <= 0.0f) break;
            k--;
        }
        hull[k++] = pts[i];
    }
    int lower_start = k + 1;
    for (int i = n - 2; i >= 0; i--) {
        while (k >= lower_start) {
            float cross = gait_cross2d(
                hull[k - 1].x - hull[k - 2].x,
                hull[k - 1].y - hull[k - 2].y,
                pts[i].x - hull[k - 2].x,
                pts[i].y - hull[k - 2].y);
            if (cross <= 0.0f) break;
            k--;
        }
        hull[k++] = pts[i];
    }
    int result_n = (k > 1) ? k - 1 : k;
    memcpy(pts, hull, (size_t)result_n * sizeof(SupportVertex));
    return result_n;
}

/* ============================================================================
 * 内部：Bézier曲线计算 - 立方Bézier B(t) = Σ C(n,i) (1-t)^(n-i) t^i P_i
 * 使用德卡斯特里奥(de Casteljau)算法
 * ============================================================================ */
static float gait_bezier_eval(const float* control_points, int num_points, float t) {
    float temp[GAIT_BEZIER_CONTROL_POINTS];
    memcpy(temp, control_points, (size_t)num_points * sizeof(float));
    for (int r = 1; r < num_points; r++) {
        for (int i = 0; i < num_points - r; i++) {
            temp[i] = (1.0f - t) * temp[i] + t * temp[i + 1];
        }
    }
    return temp[0];
}

static float gait_bezier_eval_derivative(const float* control_points, int num_points, float t) {
    if (num_points <= 1) return 0.0f;
    float deriv_ctrl[GAIT_BEZIER_CONTROL_POINTS];
    int n = num_points - 1;
    for (int i = 0; i < n; i++) {
        deriv_ctrl[i] = (float)n * (control_points[i + 1] - control_points[i]);
    }
    return gait_bezier_eval(deriv_ctrl, n, t);
}

/* ============================================================================
 * 内部：三次样条插值系数计算
 * ============================================================================ */
static void gait_cubic_spline_coeffs(const float* x, const float* y, int n,
                                      float* a, float* b, float* c, float* d) {
    float h[GAIT_BEZIER_CONTROL_POINTS];
    float alpha[GAIT_BEZIER_CONTROL_POINTS];
    float l[GAIT_BEZIER_CONTROL_POINTS], mu[GAIT_BEZIER_CONTROL_POINTS], z[GAIT_BEZIER_CONTROL_POINTS];

    for (int i = 0; i < n; i++) a[i] = y[i];
    for (int i = 0; i < n - 1; i++) h[i] = x[i + 1] - x[i];

    for (int i = 1; i < n - 1; i++) {
        alpha[i] = (3.0f / h[i]) * (a[i + 1] - a[i]) - (3.0f / h[i - 1]) * (a[i] - a[i - 1]);
    }
    l[0] = 1.0f; mu[0] = 0.0f; z[0] = 0.0f;
    for (int i = 1; i < n - 1; i++) {
        l[i] = 2.0f * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1];
        mu[i] = h[i] / l[i];
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
    }
    l[n - 1] = 1.0f; z[n - 1] = 0.0f; c[n - 1] = 0.0f;
    for (int j = n - 2; j >= 0; j--) {
        c[j] = z[j] - mu[j] * c[j + 1];
        b[j] = (a[j + 1] - a[j]) / h[j] - h[j] * (c[j + 1] + 2.0f * c[j]) / 3.0f;
        d[j] = (c[j + 1] - c[j]) / (3.0f * h[j]);
    }
}

static float gait_cubic_spline_eval(const float* x, const float* a, const float* b,
                                     const float* c, const float* d, int n, float t) {
    int i;
    for (i = 0; i < n - 2; i++) {
        if (t <= x[i + 1]) break;
    }
    if (i >= n - 1) i = n - 2;
    float dx = t - x[i];
    return a[i] + b[i] * dx + c[i] * dx * dx + d[i] * dx * dx * dx;
}

/* ============================================================================
 * 内部：LIPM预览控制增益计算
 * 基于Kajita等人提出的ZMP预览控制方法
 * 公式: u(k) = -Gi * Σ e(i) - Gx * x(k)
 * 其中e(i) = zmp_ref(i) - zmp_actual(i)
 * ============================================================================ */
static void gait_lipm_compute_preview_gains(float omega, float dt,
                                             int horizon, float* gains) {
    float A[4] = {1.0f, dt, 0.0f, 1.0f};
    float B[2] = {dt * dt * 0.5f, dt};
    float C[2] = {1.0f, 0.0f};

    /* 基于LIPM离散化: 
     * [x(k+1)]   = [1 dt dt²/2] [x(k)  ]
     * [vx(k+1)]   = [0 1  dt  ] [vx(k) ]
     * [ax(k+1)]   = [0 0   1  ] [ax(k) ] + [0 0 1]^T * jerk
     * zmp = [1 0 -zc/g] * state
     * 使用Riccati方程求解最优增益 */
    float zc_over_g = 1.0f / omega; /* zc/g近似 */
    (void)zc_over_g;

    /* 简化预览增益：指数衰减 */
    for (int i = 0; i < horizon; i++) {
        float k = (float)i;
        float decay = expf(-omega * k * dt);
        gains[i] = -decay * omega * dt;
    }
}

/* ============================================================================
 * 内部：LIPM轨道能量计算
 * E = 0.5 * v² - 0.5 * ω² * (x - zmp)²
 * ============================================================================ */
static float gait_lipm_orbital_energy(float pos, float vel, float zmp, float omega) {
    return 0.5f * vel * vel - 0.5f * omega * omega * (pos - zmp) * (pos - zmp);
}

/* ============================================================================
 * 内部：LIPM捕获点(Capture Point)计算
 * CP = x + v/ω (在恒定ZMP假设下)
 * ============================================================================ */
static float gait_lipm_capture_point(float pos, float vel, float omega) {
    return pos + vel / omega;
}

/* ============================================================================
 * 内部：腿部逆运动学(IK) - 6自由度人形腿
 * 关节顺序: [髋偏航, 髋横滚, 髋俯仰, 膝俯仰, 踝俯仰, 踝横滚]
 * 使用解析几何方法求解
 * ============================================================================ */
static int gait_leg_ik(const float* hip_pos, const float* foot_target,
                        float thigh_len, float shin_len,
                        int is_left_leg, float* joint_angles) {
    /* 相对髋部坐标 */
    float dx = foot_target[0] - hip_pos[0];
    float dy = foot_target[1] - hip_pos[1];
    float dz = foot_target[2] - hip_pos[2];

    /* 髋偏航角 (绕Z轴旋转) */
    float hip_yaw = atan2f(dy, dx);

    /* 投影到髋偏航后的X-Z平面 */
    float d_horiz = sqrtf(dx * dx + dy * dy);
    if (d_horiz < GAIT_EPSILON) d_horiz = GAIT_EPSILON;

    /* 髋-踝距离 */
    float D = sqrtf(d_horiz * d_horiz + dz * dz);
    if (D < GAIT_EPSILON) D = GAIT_EPSILON;

    /* 余弦定理求膝角 */
    float l1 = thigh_len;
    float l2 = shin_len;
    float cos_knee = (D * D - l1 * l1 - l2 * l2) / (2.0f * l1 * l2);
    if (cos_knee > 1.0f) cos_knee = 1.0f;
    if (cos_knee < -1.0f) cos_knee = -1.0f;
    float knee_angle = (float)M_PI - acosf(cos_knee); /* 膝俯仰角：伸直=0 */

    /* 髋俯仰角 */
    float alpha = atan2f(-dz, d_horiz);
    float cos_beta = (D * D + l1 * l1 - l2 * l2) / (2.0f * D * l1);
    if (cos_beta > 1.0f) cos_beta = 1.0f;
    if (cos_beta < -1.0f) cos_beta = -1.0f;
    float beta = acosf(cos_beta);
    float hip_pitch = alpha + beta;

    /* 踝俯仰角 = 保持脚底水平 */
    float ankle_pitch = -(hip_pitch + knee_angle);

    /* 髋横滚角（左右向倾斜补偿） */
    float hip_roll;
    if (is_left_leg) {
        hip_roll = atan2f(fabsf(dz) * 0.15f, d_horiz) * -1.0f;
    } else {
        hip_roll = atan2f(fabsf(dz) * 0.15f, d_horiz);
    }
    hip_roll = GAIT_CLAMP(hip_roll, -0.5f, 0.5f);

    /* 踝横滚角 = -髋横滚（保持脚底水平） */
    float ankle_roll = -hip_roll;

    /* 输出关节角度 [髋偏航, 髋横滚, 髋俯仰, 膝俯仰, 踝俯仰, 踝横滚] */
    joint_angles[0] = hip_yaw;
    joint_angles[1] = hip_roll;
    joint_angles[2] = hip_pitch;
    joint_angles[3] = knee_angle;
    joint_angles[4] = ankle_pitch;
    joint_angles[5] = ankle_roll;

    return 0;
}

/* ============================================================================
 * 默认步态配置
 * ============================================================================ */
GaitConfig gait_config_default(void) {
    GaitConfig cfg;
    memset(&cfg, 0, sizeof(GaitConfig));
    cfg.step_length = 0.3f;
    cfg.step_height = 0.08f;
    cfg.step_width = 0.15f;
    cfg.step_frequency = 2.0f;
    cfg.double_support_ratio = 0.2f;
    cfg.com_height = 0.8f;
    cfg.foot_clearance = 0.02f;
    cfg.gravity = 9.81f;
    cfg.max_step_length = 0.5f;
    cfg.max_turn_angle = 0.3f;
    cfg.zmp_safety_margin = 0.03f;
    cfg.force_landing_threshold = 50.0f;
    cfg.swing_duty_ratio = 0.5f;
    cfg.enable_zmp_preview = 1;
    cfg.enable_com_compensation = 1;
    cfg.ik_max_iterations = GAIT_IK_MAX_ITER;
    cfg.ik_tolerance = GAIT_IK_TOLERANCE;
    cfg.hip_ankle_offset = 0.0f;
    cfg.hip_yaw_offset = 0.0f;
    return cfg;
}

/* ============================================================================
 * 默认行走参数
 * ============================================================================ */
WalkParameter walk_parameter_default(void) {
    WalkParameter wp;
    memset(&wp, 0, sizeof(WalkParameter));
    wp.direction.x = 1.0f;
    wp.direction.y = 0.0f;
    wp.direction.z = 0.0f;
    wp.speed = 0.5f;
    wp.turn_rate = 0.0f;
    wp.gait_type = GAIT_FORWARD;
    wp.target_distance = -1.0f;
    wp.target_angle = 0.0f;
    wp.num_steps = -1;
    wp.stop_after_cycle = 0;
    return wp;
}

/* ============================================================================
 * 步态生成器创建/销毁
 * ============================================================================ */
GaitGenerator* gait_generator_create(const GaitConfig* config, const KinematicModel* kin_model) {
    GaitGenerator* gen = (GaitGenerator*)safe_malloc(sizeof(GaitGenerator));
    if (!gen) return NULL;
    memset(gen, 0, sizeof(GaitGenerator));

    if (config) {
        gen->config = *config;
    } else {
        gen->config = gait_config_default();
    }
    gen->kin_model = kin_model;

    /* 初始化LIPM */
    memset(&gen->lipm, 0, sizeof(LIPMState));
    gen->lipm.com_z = gen->config.com_height;
    gen->lipm.omega = sqrtf(gen->config.gravity / gen->config.com_height);

    /* 计算LIPM预览控制增益 */
    float dt = 1.0f / gen->config.step_frequency / 10.0f;
    gait_lipm_compute_preview_gains(gen->lipm.omega, dt,
                                     GAIT_LIPM_PREVIEW_HORIZON, gen->lipm_gain_x);

    /* 初始化步态状态 */
    memset(&gen->state, 0, sizeof(GaitState));
    gen->state.phase = GAIT_PHASE_DOUBLE_SUPPORT;
    gen->state.is_stable = 1;
    gen->state.stability_margin = gen->config.zmp_safety_margin;

    /* 默认初始脚步位置 */
    gen->state.left_foot_pos[1] = gen->config.step_width * 0.5f;
    gen->state.right_foot_pos[1] = -gen->config.step_width * 0.5f;
    gen->state.left_foot_on_ground = 1;
    gen->state.right_foot_on_ground = 1;

    gen->swing_leg = 1; /* 先抬右腿 */
    gen->initialized = 1;

    return gen;
}

void gait_generator_free(GaitGenerator* gen) {
    if (!gen) return;
    safe_free((void**)&gen);
}

void gait_generator_reset(GaitGenerator* gen) {
    if (!gen) return;
    memset(&gen->lipm, 0, sizeof(LIPMState));
    gen->lipm.com_z = gen->config.com_height;
    gen->lipm.omega = sqrtf(gen->config.gravity / gen->config.com_height);
    memset(&gen->state, 0, sizeof(GaitState));
    gen->state.phase = GAIT_PHASE_DOUBLE_SUPPORT;
    gen->state.is_stable = 1;
    gen->state.stability_margin = gen->config.zmp_safety_margin;
    gen->state.left_foot_pos[1] = gen->config.step_width * 0.5f;
    gen->state.right_foot_pos[1] = -gen->config.step_width * 0.5f;
    gen->state.left_foot_on_ground = 1;
    gen->state.right_foot_on_ground = 1;
    gen->swing_leg = 1;
    gen->phase_timer = 0.0f;
    gen->step_progress = 0.0f;
    gen->total_distance_traveled = 0.0f;
    gen->total_turn_angle = 0.0f;
    gen->total_step_count = 0;
    gen->walk_params = walk_parameter_default();
}

int gait_generator_set_config(GaitGenerator* gen, const GaitConfig* config) {
    if (!gen || !config) return -1;
    gen->config = *config;
    gen->lipm.omega = sqrtf(gen->config.gravity / gen->config.com_height);
    gen->lipm.com_z = gen->config.com_height;
    float dt = 1.0f / gen->config.step_frequency / 10.0f;
    gait_lipm_compute_preview_gains(gen->lipm.omega, dt,
                                     GAIT_LIPM_PREVIEW_HORIZON, gen->lipm_gain_x);
    return 0;
}

int gait_generator_get_config(const GaitGenerator* gen, GaitConfig* config) {
    if (!gen || !config) return -1;
    *config = gen->config;
    return 0;
}

/* ============================================================================
 * gait_compute_zmp - ZMP零力矩点计算
 * 基于多面体支撑域方法:
 * ZMP_x = Σ(F_i * x_i) / Σ(F_i)
 * ZMP_y = Σ(F_i * y_i) / Σ(F_i)
 * 然后检查ZMP是否在支撑多边形内及安全边界内
 * ============================================================================ */
int gait_compute_zmp(GaitGenerator* gen,
                      const SupportVertex* support_vertices, int num_vertices,
                      const float* forces, int num_forces,
                      ZMPPoint* zmp) {
    if (!gen || !zmp) return -1;

    memset(zmp, 0, sizeof(ZMPPoint));
    zmp->z = 0.0f;

    /* 如果有力传感器数据，使用力加权平均计算ZMP */
    if (forces && num_forces > 0 && num_vertices > 0) {
        float sum_f = 0.0f;
        float sum_fx = 0.0f;
        float sum_fy = 0.0f;
        float sum_pressure = 0.0f;

        int n = (num_forces < num_vertices) ? num_forces : num_vertices;
        for (int i = 0; i < n; i++) {
            float f = forces[i];
            if (f < 0.0f) f = 0.0f;
            sum_f += f;
            sum_fx += f * support_vertices[i].x;
            sum_fy += f * support_vertices[i].y;
            sum_pressure += f;
        }

        if (sum_f > GAIT_EPSILON) {
            zmp->x = sum_fx / sum_f;
            zmp->y = sum_fy / sum_f;
            zmp->pressure = sum_pressure;
            zmp->cop_x = zmp->x;
            zmp->cop_y = zmp->y;
        } else {
            /* 无力数据时使用支撑多边形几何中心 */
            float cx = 0.0f, cy = 0.0f;
            for (int i = 0; i < num_vertices; i++) {
                cx += support_vertices[i].x;
                cy += support_vertices[i].y;
            }
            zmp->x = cx / (float)num_vertices;
            zmp->y = cy / (float)num_vertices;
            zmp->pressure = 0.0f;
        }
    } else if (num_vertices >= 3) {
        /* 使用支撑多边形重心作为默认ZMP */
        float cx = 0.0f, cy = 0.0f;
        for (int i = 0; i < num_vertices; i++) {
            cx += support_vertices[i].x;
            cy += support_vertices[i].y;
        }
        zmp->x = cx / (float)num_vertices;
        zmp->y = cy / (float)num_vertices;
        zmp->pressure = 0.0f;
    } else {
        zmp->is_valid = 0;
        return -1;
    }

    /* 检查ZMP是否在支撑多边形内 */
    if (num_vertices >= 3) {
        int inside = gait_point_in_convex_polygon(zmp->x, zmp->y,
                                                   support_vertices, num_vertices);
        float margin = gait_point_to_polygon_distance(zmp->x, zmp->y,
                                                       support_vertices, num_vertices);

        if (inside && margin >= gen->config.zmp_safety_margin) {
            zmp->is_valid = 1;
            return 0;
        } else if (inside && margin < gen->config.zmp_safety_margin) {
            zmp->is_valid = 1;
            return 1;
        } else {
            zmp->is_valid = 0;
            return -1;
        }
    }

    zmp->is_valid = 0;
    return -1;
}

/* ============================================================================
 * gait_lipm_update - 线性倒立摆(LIPM)更新
 * 3D LIPM动力学:
 *   前向(X): d²x/dt² = (g/zc) * (x - zmp_x)
 *   侧向(Y): d²y/dt² = (g/zc) * (y - zmp_y)
 * 使用半隐式欧拉积分和ZMP预览控制
 * ============================================================================ */
int gait_lipm_update(GaitGenerator* gen, LIPMState* lipm,
                      float dt, float target_zmp_x, float target_zmp_y) {
    if (!gen || !lipm) return -1;

    float omega_sq = lipm->omega * lipm->omega;

    /* 更新ZMP参考 */
    lipm->zmp_x = target_zmp_x;
    lipm->zmp_y = target_zmp_y;

    /* LIPM解析解（恒定ZMP假设下的精确解）:
     * x(t+dt) = x(t)*cosh(ω*dt) + v(t)/ω*sinh(ω*dt) + zmp*(1-cosh(ω*dt))
     * v(t+dt) = x(t)*ω*sinh(ω*dt) + v(t)*cosh(ω*dt) - zmp*ω*sinh(ω*dt) */
    float w_dt = lipm->omega * dt;
    float cosh_wt = coshf(w_dt);
    float sinh_wt = sinhf(w_dt);

    /* X方向更新 */
    float x_old = lipm->com_x;
    float vx_old = lipm->com_vx;
    lipm->com_x = x_old * cosh_wt + vx_old / lipm->omega * sinh_wt
                + target_zmp_x * (1.0f - cosh_wt);
    lipm->com_vx = x_old * lipm->omega * sinh_wt + vx_old * cosh_wt
                 - target_zmp_x * lipm->omega * sinh_wt;
    lipm->com_ax = omega_sq * (lipm->com_x - target_zmp_x);

    /* Y方向更新 */
    float y_old = lipm->com_y;
    float vy_old = lipm->com_vy;
    lipm->com_y = y_old * cosh_wt + vy_old / lipm->omega * sinh_wt
                + target_zmp_y * (1.0f - cosh_wt);
    lipm->com_vy = y_old * lipm->omega * sinh_wt + vy_old * cosh_wt
                 - target_zmp_y * lipm->omega * sinh_wt;
    lipm->com_ay = omega_sq * (lipm->com_y - target_zmp_y);

    /* ZMP预览控制 - 基于未来ZMP参考调整当前质心加速度 */
    if (gen->config.enable_zmp_preview) {
        int idx = lipm->preview_index;
        if (idx >= 0 && idx < GAIT_LIPM_PREVIEW_HORIZON) {
            float gain_x = gen->lipm_gain_x[idx];
            float gain_y = gen->lipm_gain_y[idx];
            float zmp_error_x = lipm->zmp_ref_x[idx] - lipm->zmp_x;
            float zmp_error_y = lipm->zmp_ref_y[idx] - lipm->zmp_y;
            lipm->com_ax += gain_x * zmp_error_x;
            lipm->com_ay += gain_y * zmp_error_y;
            lipm->com_vx += lipm->com_ax * dt * 0.5f;
            lipm->com_vy += lipm->com_ay * dt * 0.5f;
        }
    }

    lipm->preview_index++;
    if (lipm->preview_index >= GAIT_LIPM_PREVIEW_HORIZON) {
        lipm->preview_index = 0;
    }

    return 0;
}

/* ============================================================================
 * gait_foot_trajectory - 足部轨迹规划
 * 支持四种轨迹类型: Bézier曲线, 三次样条, 摆线, 最小加加速度
 * ============================================================================ */
int gait_foot_trajectory(GaitGenerator* gen,
                          const float* start_pos, const float* end_pos,
                          float t, float step_height,
                          SwingTrajectoryType traj_type,
                          float* out_pos, float* out_vel) {
    if (!gen || !start_pos || !end_pos || !out_pos) return -1;

    float tt = GAIT_CLAMP(t, 0.0f, 1.0f);

    switch (traj_type) {
    case SWING_TRAJ_BEZIER: {
        /* 立方Bézier，4个控制点: 
         * P0=起点, P1=起点+前向偏移, P2=终点+前向偏移, P3=终点
         * Z轴单独Bézier控制抬起高度 */
        float mid_x = (start_pos[0] + end_pos[0]) * 0.5f;
        float mid_y = (start_pos[1] + end_pos[1]) * 0.5f;

        float ctrl_x[4] = { start_pos[0], start_pos[0] + (mid_x - start_pos[0]) * 0.5f,
                            end_pos[0] - (end_pos[0] - mid_x) * 0.5f, end_pos[0] };
        float ctrl_y[4] = { start_pos[1], start_pos[1] + (mid_y - start_pos[1]) * 0.5f,
                            end_pos[1] - (end_pos[1] - mid_y) * 0.5f, end_pos[1] };
        float ctrl_z[4] = { start_pos[2], start_pos[2] + step_height,
                            end_pos[2] + step_height, end_pos[2] };

        out_pos[0] = gait_bezier_eval(ctrl_x, 4, tt);
        out_pos[1] = gait_bezier_eval(ctrl_y, 4, tt);
        out_pos[2] = gait_bezier_eval(ctrl_z, 4, tt);

        if (out_vel) {
            out_vel[0] = gait_bezier_eval_derivative(ctrl_x, 4, tt);
            out_vel[1] = gait_bezier_eval_derivative(ctrl_y, 4, tt);
            out_vel[2] = gait_bezier_eval_derivative(ctrl_z, 4, tt);
        }
        break;
    }
    case SWING_TRAJ_CUBIC_SPLINE: {
        /* 三次样条：4个控制时刻 [0, 0.33, 0.67, 1.0] */
        float tx[4] = { 0.0f, 0.33f, 0.67f, 1.0f };
        float h = step_height;
        float yx[4] = { start_pos[0],
                        start_pos[0] + (end_pos[0] - start_pos[0]) * 0.45f,
                        start_pos[0] + (end_pos[0] - start_pos[0]) * 0.75f,
                        end_pos[0] };
        float yy[4] = { start_pos[1],
                        start_pos[1] + (end_pos[1] - start_pos[1]) * 0.45f,
                        start_pos[1] + (end_pos[1] - start_pos[1]) * 0.75f,
                        end_pos[1] };
        float yz[4] = { start_pos[2], start_pos[2] + h,
                        end_pos[2] + h * 0.5f, end_pos[2] };

        float ax[4], bx[4], cx[4], dx[4];
        float ay[4], by[4], cy[4], dy[4];
        float az[4], bz[4], cz[4], dz[4];

        gait_cubic_spline_coeffs(tx, yx, 4, ax, bx, cx, dx);
        gait_cubic_spline_coeffs(tx, yy, 4, ay, by, cy, dy);
        gait_cubic_spline_coeffs(tx, yz, 4, az, bz, cz, dz);

        out_pos[0] = gait_cubic_spline_eval(tx, ax, bx, cx, dx, 4, tt);
        out_pos[1] = gait_cubic_spline_eval(tx, ay, by, cy, dy, 4, tt);
        out_pos[2] = gait_cubic_spline_eval(tx, az, bz, cz, dz, 4, tt);

        if (out_vel) {
            float dt_eps = 0.001f;
            float t2 = GAIT_CLAMP(tt + dt_eps, 0.0f, 1.0f);
            float t1 = GAIT_CLAMP(tt - dt_eps, 0.0f, 1.0f);
            float inv_2dt = 1.0f / (2.0f * dt_eps);
            out_vel[0] = (gait_cubic_spline_eval(tx, ax, bx, cx, dx, 4, t2) -
                          gait_cubic_spline_eval(tx, ax, bx, cx, dx, 4, t1)) * inv_2dt;
            out_vel[1] = (gait_cubic_spline_eval(tx, ay, by, cy, dy, 4, t2) -
                          gait_cubic_spline_eval(tx, ay, by, cy, dy, 4, t1)) * inv_2dt;
            out_vel[2] = (gait_cubic_spline_eval(tx, az, bz, cz, dz, 4, t2) -
                          gait_cubic_spline_eval(tx, az, bz, cz, dz, 4, t1)) * inv_2dt;
        }
        break;
    }
    case SWING_TRAJ_CYCLOID: {
        /* 摆线轨迹: 
         * x(s) = (1-s)*x0 + s*x1 (线性插值)
         * z(s) = z0 + step_height * (1 - cos(2πs)) + (z1-z0)*s */
        float s = tt;
        float phase = 2.0f * (float)M_PI * s;
        out_pos[0] = start_pos[0] + (end_pos[0] - start_pos[0]) * s;
        out_pos[1] = start_pos[1] + (end_pos[1] - start_pos[1]) * s;
        out_pos[2] = start_pos[2] + step_height * (1.0f - cosf(phase))
                   + (end_pos[2] - start_pos[2]) * s;

        if (out_vel) {
            out_vel[0] = end_pos[0] - start_pos[0];
            out_vel[1] = end_pos[1] - start_pos[1];
            out_vel[2] = step_height * 2.0f * (float)M_PI * sinf(phase)
                       + (end_pos[2] - start_pos[2]);
        }
        break;
    }
    case SWING_TRAJ_MIN_JERK: {
        /* 最小加加速度轨迹 (Flash & Hogan):
         * s(t) = 10t³ - 15t⁴ + 6t⁵
         * 速度: 30t² - 60t³ + 30t⁴ */
        float s_val = 10.0f * tt * tt * tt - 15.0f * tt * tt * tt * tt
                    + 6.0f * tt * tt * tt * tt * tt;
        float ds_val = 30.0f * tt * tt - 60.0f * tt * tt * tt
                     + 30.0f * tt * tt * tt * tt;

        out_pos[0] = start_pos[0] + (end_pos[0] - start_pos[0]) * s_val;
        out_pos[1] = start_pos[1] + (end_pos[1] - start_pos[1]) * s_val;

        /* Z轴: 使用正弦包络 + 最小加加速度平滑 */
        float sin_lift = sinf((float)M_PI * tt);
        out_pos[2] = start_pos[2] + step_height * sin_lift * s_val
                   + (end_pos[2] - start_pos[2]) * s_val;

        if (out_vel) {
            out_vel[0] = (end_pos[0] - start_pos[0]) * ds_val;
            out_vel[1] = (end_pos[1] - start_pos[1]) * ds_val;
            float dsin = (float)M_PI * cosf((float)M_PI * tt);
            out_vel[2] = step_height * (dsin * s_val + sin_lift * ds_val)
                       + (end_pos[2] - start_pos[2]) * ds_val;
        }
        break;
    }
    default:
        return -1;
    }

    return 0;
}

/* ============================================================================
 * gait_step_planning - 步态规划
 * 基于ZMP预览控制和LIPM动力学规划未来N步的脚部落点
 * ============================================================================ */
int gait_step_planning(GaitGenerator* gen,
                        const WalkParameter* params,
                        const float* current_left_foot,
                        const float* current_right_foot,
                        const float* com_pos,
                        int swing_leg,
                        int num_preview_steps,
                        float* next_footsteps) {
    if (!gen || !params || !current_left_foot || !current_right_foot
        || !com_pos || !next_footsteps) return -1;

    float step_len = gen->config.step_length;
    float step_wid = gen->config.step_width;
    float turn_per_step = 0.0f;

    /* 根据步态方向设置步长和转弯 */
    float dir_x = params->direction.x;
    float dir_y = params->direction.y;

    switch (params->gait_type) {
    case GAIT_FORWARD:
        dir_x = 1.0f; dir_y = 0.0f;
        break;
    case GAIT_BACKWARD:
        dir_x = -1.0f; dir_y = 0.0f;
        break;
    case GAIT_LEFTWARD:
        dir_x = 0.0f; dir_y = 1.0f;
        break;
    case GAIT_RIGHTWARD:
        dir_x = 0.0f; dir_y = -1.0f;
        break;
    case GAIT_TURN_LEFT:
        dir_x = cosf(turn_per_step); dir_y = sinf(turn_per_step);
        turn_per_step = gen->config.max_turn_angle * 0.3f;
        break;
    case GAIT_TURN_RIGHT:
        dir_x = cosf(-turn_per_step); dir_y = sinf(-turn_per_step);
        turn_per_step = -gen->config.max_turn_angle * 0.3f;
        break;
    case GAIT_DIAGONAL_FL:
        dir_x = 0.707f; dir_y = 0.707f;
        break;
    case GAIT_DIAGONAL_FR:
        dir_x = 0.707f; dir_y = -0.707f;
        break;
    case GAIT_IN_PLACE:
        step_len = 0.0f;
        break;
    default:
        break;
    }

    /* 根据速度调整步长 */
    float speed_factor = params->speed / 0.5f;
    step_len *= speed_factor;
    step_len = GAIT_CLAMP(step_len, 0.0f, gen->config.max_step_length);

    /* 计算每一步的落点 */
    float lfx = current_left_foot[0], lfy = current_left_foot[1];
    float rfx = current_right_foot[0], rfy = current_right_foot[1];

    int preview = (num_preview_steps > 0) ? num_preview_steps : 4;
    if (preview > GAIT_MAX_FOOTSTEPS_PER_CYCLE)
        preview = GAIT_MAX_FOOTSTEPS_PER_CYCLE;

    float accum_angle = 0.0f;

    for (int i = 0; i < preview; i++) {
        int leg = (swing_leg + i) % 2;
        float* step = next_footsteps + i * 3;

        if (leg == 0) {
            /* 左腿下一步：交叉落在右腿前方另一侧 */
            float nx = rfx + dir_x * step_len;
            float ny = rfy + dir_y * step_len + step_wid * 0.5f;
            /* 应用累计转弯角度 */
            if (fabsf(turn_per_step) > GAIT_EPSILON) {
                accum_angle += turn_per_step;
                float ca = cosf(accum_angle), sa = sinf(accum_angle);
                float rx = nx - com_pos[0], ry = ny - com_pos[1];
                nx = com_pos[0] + rx * ca - ry * sa;
                ny = com_pos[1] + rx * sa + ry * ca;
            }
            step[0] = nx;
            step[1] = ny;
            step[2] = 0.0f;
        } else {
            /* 右腿下一步：交叉落在左腿前方另一侧 */
            float nx = lfx + dir_x * step_len;
            float ny = lfy + dir_y * step_len - step_wid * 0.5f;
            if (fabsf(turn_per_step) > GAIT_EPSILON) {
                accum_angle += turn_per_step;
                float ca = cosf(accum_angle), sa = sinf(accum_angle);
                float rx = nx - com_pos[0], ry = ny - com_pos[1];
                nx = com_pos[0] + rx * ca - ry * sa;
                ny = com_pos[1] + rx * sa + ry * ca;
            }
            step[0] = nx;
            step[1] = ny;
            step[2] = 0.0f;
        }
    }

    return preview;
}

/* ============================================================================
 * gait_generate_swing_leg_trajectory - 生成完整的摆动相轨迹
 * 将摆动时间离散化，逐点采样足部轨迹
 * ============================================================================ */
int gait_generate_swing_leg_trajectory(GaitGenerator* gen,
                                        const float* start_pos,
                                        const float* target_pos,
                                        float swing_time, float dt,
                                        SwingTrajectoryType traj_type,
                                        float* out_traj,
                                        int max_points, int* num_points) {
    if (!gen || !start_pos || !target_pos || !out_traj || !num_points) return -1;
    if (swing_time < GAIT_EPSILON || dt < GAIT_EPSILON) return -1;

    int n = (int)(swing_time / dt);
    if (n <= 0) n = 1;
    if (n > max_points) n = max_points;

    for (int i = 0; i < n; i++) {
        float t = (float)i / (float)(n - 1 > 0 ? n - 1 : 1);
        float pos[3], vel[3];
        int ret = gait_foot_trajectory(gen, start_pos, target_pos,
                                        t, gen->config.step_height,
                                        traj_type, pos, vel);
        if (ret != 0) return -1;
        out_traj[i * 3 + 0] = pos[0];
        out_traj[i * 3 + 1] = pos[1];
        out_traj[i * 3 + 2] = pos[2];
    }

    *num_points = n;
    return 0;
}

/* ============================================================================
 * gait_detect_foot_landing - 脚部落地检测
 * 综合ZMP位置和力传感器阈值判断脚是否已落地
 * ============================================================================ */
int gait_detect_foot_landing(GaitGenerator* gen,
                              const ZMPPoint* zmp,
                              const float* swing_foot_force,
                              const float* swing_foot_pos,
                              const float* target_landing_pos) {
    if (!gen || !zmp || !swing_foot_force || !swing_foot_pos || !target_landing_pos)
        return -1;

    float force_mag = swing_foot_force[3]; /* 合力大小 */
    float force_z = swing_foot_force[2];   /* Z方向力 */
    float threshold = gen->config.force_landing_threshold;

    /* 条件1: 力传感器检测到显著垂直接触力 */
    int force_condition = (force_z > threshold || force_mag > threshold * 1.5f);

    /* 条件2: 摆动脚高度接近目标地面 */
    float height_diff = fabsf(swing_foot_pos[2] - target_landing_pos[2]);
    int height_condition = (height_diff < gen->config.foot_clearance * 2.0f);

    /* 条件3: ZMP向支撑域边界移动（表示负载转移） */
    float zmp_to_foot_x = zmp->x - target_landing_pos[0];
    float zmp_to_foot_y = zmp->y - target_landing_pos[1];
    float zmp_dist = sqrtf(zmp_to_foot_x * zmp_to_foot_x + zmp_to_foot_y * zmp_to_foot_y);
    int zmp_condition = (zmp_dist < gen->config.step_length * 0.8f);

    /* 异常着陆检测 */
    if (force_z > threshold * 5.0f) {
        return -1; /* 冲击力过大，可能摔倒 */
    }

    if (force_condition && height_condition) {
        return 1; /* 正常着陆 */
    }
    if (height_condition && zmp_condition) {
        return 1; /* ZMP和高度都满足条件 */
    }

    return 0; /* 未着陆 */
}

/* ============================================================================
 * gait_generate_walk_cycle - 生成行走周期
 * 计算双支撑相→单支撑相→摆动相的完整周期
 * ============================================================================ */
int gait_generate_walk_cycle(GaitGenerator* gen, const WalkParameter* params, GaitCycle* cycle) {
    if (!gen || !params || !cycle) return -1;

    memset(cycle, 0, sizeof(GaitCycle));

    gen->walk_params = *params;

    /* 步态周期时间 T = 1 / (步频 * 2) （每步半个周期） */
    float cycle_period = 1.0f / gen->config.step_frequency;
    cycle->cycle_time = cycle_period * 2.0f; /* 完整双步周期 */

    /* 时间分配 */
    float ds_ratio = GAIT_CLAMP(gen->config.double_support_ratio, 0.1f, 0.4f);
    cycle->double_support_time = cycle_period * ds_ratio;
    cycle->single_support_time = cycle_period - cycle->double_support_time;
    cycle->elapsed_time = 0.0f;
    cycle->current_step_index = 0;
    cycle->cycle_complete = 0;

    /* 使用当前脚步位置进行规划 */
    float lf[3], rf[3], com[3];
    gait_vec3_copy(lf, gen->state.left_foot_pos);
    gait_vec3_copy(rf, gen->state.right_foot_pos);
    com[0] = gen->lipm.com_x;
    com[1] = gen->lipm.com_y;
    com[2] = gen->lipm.com_z;

    /* 规划4步 */
    float planned[4 * 3];
    int num_planned = gait_step_planning(gen, params, lf, rf, com,
                                          gen->swing_leg, 4, planned);
    if (num_planned <= 0) return -1;

    /* 填充步态周期的脚步序列 */
    int num_steps = (num_planned < GAIT_MAX_FOOTSTEPS_PER_CYCLE)
                    ? num_planned : GAIT_MAX_FOOTSTEPS_PER_CYCLE;
    cycle->num_footsteps = num_steps;

    for (int i = 0; i < num_steps; i++) {
        int leg = (gen->swing_leg + i) % 2;
        cycle->footsteps[i].position.x = planned[i * 3 + 0];
        cycle->footsteps[i].position.y = planned[i * 3 + 1];
        cycle->footsteps[i].position.z = planned[i * 3 + 2];
        cycle->footsteps[i].orientation = 0.0f;
        cycle->footsteps[i].leg_index = leg;
        cycle->footsteps[i].is_support = 0;
        cycle->footsteps[i].time_start = (float)i * cycle_period;
        cycle->footsteps[i].time_end = (float)(i + 1) * cycle_period;
        cycle->footsteps[i].step_index = i;
        cycle->footsteps[i].contact_force = 0.0f;
        cycle->footsteps[i].landed = 0;
    }

    cycle->current_phase = GAIT_PHASE_DOUBLE_SUPPORT;
    return 0;
}

/* ============================================================================
 * gait_update_state - 更新步态状态机
 * 驱动双支撑相↔单支撑相(左/右摆动)的状态转换
 * ============================================================================ */
int gait_update_state(GaitGenerator* gen, float dt, GaitState* state) {
    if (!gen || !state) return -1;

    if (dt <= 0.0f) return 0;

    float cycle_period = 1.0f / gen->config.step_frequency;
    float ds_time = cycle_period * GAIT_CLAMP(gen->config.double_support_ratio, 0.1f, 0.4f);
    float ss_time = cycle_period - ds_time;
    float swing_time = ss_time;

    gen->phase_timer += dt;
    state->elapsed_time = gen->phase_timer;

    /* 更新支撑脚的ZMP目标 */
    float target_zmp_x, target_zmp_y;
    float mid_foot_x = (gen->state.left_foot_pos[0] + gen->state.right_foot_pos[0]) * 0.5f;
    float mid_foot_y = (gen->state.left_foot_pos[1] + gen->state.right_foot_pos[1]) * 0.5f;

    switch (gen->state.phase) {
    case GAIT_PHASE_DOUBLE_SUPPORT: {
        /* 双支撑相: ZMP在两脚之间 */
        target_zmp_x = mid_foot_x;
        target_zmp_y = mid_foot_y;

        /* 双支撑相计时 */
        if (gen->phase_timer >= ds_time) {
            /* 进入摆动相 */
            gen->phase_timer = 0.0f;
            if (gen->swing_leg == 0) {
                gen->state.phase = GAIT_PHASE_LEFT_SWING;
            } else {
                gen->state.phase = GAIT_PHASE_RIGHT_SWING;
            }
        }
        break;
    }
    case GAIT_PHASE_LEFT_SWING: {
        /* 左腿摆动相: ZMP在右脚 */
        if (gen->state.right_foot_on_ground) {
            target_zmp_x = gen->state.right_foot_pos[0];
            target_zmp_y = gen->state.right_foot_pos[1];
        } else {
            target_zmp_x = mid_foot_x;
            target_zmp_y = mid_foot_y;
        }

        /* 计算摆动腿轨迹进度 */
        float swing_progress = gen->phase_timer / swing_time;
        swing_progress = GAIT_CLAMP(swing_progress, 0.0f, 1.0f);
        state->phase_progress = swing_progress;

        /* 使用Bézier轨迹计算摆动脚位置 */
        float target_lf[3];
        gait_vec3_copy(target_lf, gen->state.left_foot_target);
        if (gait_vec3_length(target_lf) < GAIT_EPSILON) {
            gait_vec3_copy(target_lf, gen->state.left_foot_pos);
            target_lf[0] += gen->config.step_length;
        }
        float swing_pos[3], swing_vel[3];
        gait_foot_trajectory(gen, gen->state.left_foot_pos, target_lf,
                              swing_progress, gen->config.step_height,
                              SWING_TRAJ_BEZIER, swing_pos, swing_vel);
        gait_vec3_copy(state->swing_foot_trajectory, swing_pos);
        gait_vec3_copy(state->swing_foot_velocity, swing_vel);
        gait_vec3_copy(state->left_foot_pos, swing_pos);
        state->left_foot_on_ground = 0;

        /* 检查摆动是否完成 */
        if (gen->phase_timer >= swing_time) {
            gen->phase_timer = 0.0f;
            gen->swing_leg = 1;
            gen->state.phase = GAIT_PHASE_DOUBLE_SUPPORT;
            gen->state.left_foot_on_ground = 1;
            gait_vec3_copy(gen->state.left_foot_pos, target_lf);
            gen->total_step_count++;
        }
        break;
    }
    case GAIT_PHASE_RIGHT_SWING: {
        /* 右腿摆动相: ZMP在左脚 */
        if (gen->state.left_foot_on_ground) {
            target_zmp_x = gen->state.left_foot_pos[0];
            target_zmp_y = gen->state.left_foot_pos[1];
        } else {
            target_zmp_x = mid_foot_x;
            target_zmp_y = mid_foot_y;
        }

        float swing_progress = gen->phase_timer / swing_time;
        swing_progress = GAIT_CLAMP(swing_progress, 0.0f, 1.0f);
        state->phase_progress = swing_progress;

        float target_rf[3];
        gait_vec3_copy(target_rf, gen->state.right_foot_target);
        if (gait_vec3_length(target_rf) < GAIT_EPSILON) {
            gait_vec3_copy(target_rf, gen->state.right_foot_pos);
            target_rf[0] += gen->config.step_length;
        }
        float swing_pos[3], swing_vel[3];
        gait_foot_trajectory(gen, gen->state.right_foot_pos, target_rf,
                              swing_progress, gen->config.step_height,
                              SWING_TRAJ_BEZIER, swing_pos, swing_vel);
        gait_vec3_copy(state->swing_foot_trajectory, swing_pos);
        gait_vec3_copy(state->swing_foot_velocity, swing_vel);
        gait_vec3_copy(state->right_foot_pos, swing_pos);
        state->right_foot_on_ground = 0;

        if (gen->phase_timer >= swing_time) {
            gen->phase_timer = 0.0f;
            gen->swing_leg = 0;
            gen->state.phase = GAIT_PHASE_DOUBLE_SUPPORT;
            gen->state.right_foot_on_ground = 1;
            gait_vec3_copy(gen->state.right_foot_pos, target_rf);
            gen->total_step_count++;
        }
        break;
    }
    case GAIT_PHASE_STOPPED:
        target_zmp_x = mid_foot_x;
        target_zmp_y = mid_foot_y;
        break;
    default:
        return -1;
    }

    /* LIPM更新 */
    gait_lipm_update(gen, &gen->lipm, dt, target_zmp_x, target_zmp_y);

    /* 更新输出状态 */
    state->phase = gen->state.phase;
    state->cycle_time = cycle_period * 2.0f;
    state->com_position[0] = gen->lipm.com_x;
    state->com_position[1] = gen->lipm.com_y;
    state->com_position[2] = gen->lipm.com_z;
    state->com_velocity[0] = gen->lipm.com_vx;
    state->com_velocity[1] = gen->lipm.com_vy;
    state->com_velocity[2] = 0.0f;
    state->zmp[0] = gen->lipm.zmp_x;
    state->zmp[1] = gen->lipm.zmp_y;
    state->step_count = gen->total_step_count;

    /* 计算ZMP稳定裕度 */
    SupportVertex verts[GAIT_ZMP_SUPPORT_POINTS];
    int nverts = gait_build_support_polygon(
        gen->state.left_foot_pos, gen->state.right_foot_pos,
        gen->state.left_foot_on_ground, gen->state.right_foot_on_ground,
        0.25f, 0.1f, verts, GAIT_ZMP_SUPPORT_POINTS);

    if (nverts >= 3) {
        nverts = gait_compute_convex_hull(verts, nverts);
        int inside = gait_point_in_convex_polygon(gen->lipm.zmp_x, gen->lipm.zmp_y,
                                                   verts, nverts);
        state->stability_margin = gait_point_to_polygon_distance(
            gen->lipm.zmp_x, gen->lipm.zmp_y, verts, nverts);
        state->is_stable = (inside && state->stability_margin >= gen->config.zmp_safety_margin) ? 1 : 0;
    } else {
        state->stability_margin = 0.0f;
        state->is_stable = 0;
    }

    /* 计算摆动腿IK */
    float hip_left[3] = { gen->lipm.com_x,
                          gen->lipm.com_y + gen->config.step_width * 0.25f,
                          gen->lipm.com_z - 0.1f };
    float hip_right[3] = { gen->lipm.com_x,
                           gen->lipm.com_y - gen->config.step_width * 0.25f,
                           gen->lipm.com_z - 0.1f };

    gait_get_leg_joint_angles(gen, 0, gen->state.left_foot_pos, hip_left,
                               state->left_joint_angles);
    gait_get_leg_joint_angles(gen, 1, gen->state.right_foot_pos, hip_right,
                               state->right_joint_angles);

    /* 同步GaitState的其他字段 */
    gait_vec3_copy(state->left_foot_target, gen->state.left_foot_target);
    gait_vec3_copy(state->right_foot_target, gen->state.right_foot_target);
    state->left_foot_on_ground = gen->state.left_foot_on_ground;
    state->right_foot_on_ground = gen->state.right_foot_on_ground;
    state->left_foot_force = gen->state.left_foot_force;
    state->right_foot_force = gen->state.right_foot_force;

    return 0;
}

/* ============================================================================
 * gait_get_leg_joint_angles - 腿部逆运动学求解
 * 对给定足部位置和髋部位置，计算6维关节角度
 * ============================================================================ */
int gait_get_leg_joint_angles(GaitGenerator* gen,
                               int leg,
                               const float* foot_target,
                               const float* hip_position,
                               float* joint_angles) {
    if (!gen || !foot_target || !hip_position || !joint_angles) return -1;

    float thigh_len = gen->config.com_height * 0.45f;  /* 大腿长度 */
    float shin_len = gen->config.com_height * 0.40f;   /* 小腿长度 */

    int is_left = (leg == 0) ? 1 : 0;
    int ret = gait_leg_ik(hip_position, foot_target,
                           thigh_len, shin_len, is_left, joint_angles);

    /* 如果运动学模型可用，使用模型进行更精确的IK约束 */
    if (gen->kin_model && gen->kin_model->joint_count >= 6) {
        float ik_res[6];
        memcpy(ik_res, joint_angles, 6 * sizeof(float));

        /* 关节限位裁剪 */
        for (int i = 0; i < 6 && i < gen->kin_model->joint_count; i++) {
            if (gen->kin_model->joints[i].joint_type != JOINT_TYPE_CONTINUOUS) {
                float lo = gen->kin_model->joints[i].joint_limit_lower;
                float hi = gen->kin_model->joints[i].joint_limit_upper;
                joint_angles[i] = GAIT_CLAMP(ik_res[i], lo, hi);
            }
        }
    }

    return ret;
}
