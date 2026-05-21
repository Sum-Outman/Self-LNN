#ifndef SELFLNN_ODE_SOLVERS_H
#define SELFLNN_ODE_SOLVERS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SELFLNN_DP54_MAX_ATTEMPT 100
#define SELFLNN_ROSENBROCK_LINSOLVE_TOL 1e-12f
#define SELFLNN_FOREST_RUTH_THETA 1.351207191959657f

/* ============ 第1部分：现有求解器类型和配置（保持不变） ============ */

typedef int (*ODERHSFunc)(float t, const float* y, float* dydt, void* ctx);

typedef struct {
    float rel_tolerance;
    float abs_tolerance;
    float min_step_size;
    float max_step_size;
    float safety_factor;
    int max_iterations;
} DP54Config;

typedef struct {
    float rel_tolerance;
    float abs_tolerance;
    float min_step_size;
    float max_step_size;
    float gamma_coeff;
    int max_iterations;
    int use_finite_diff_jacobian;
    float finite_diff_eps;
    int jacobian_bandwidth;         /**< P2-044: 雅可比带状宽度，0=全矩阵，>0=仅计算对角线附近±bandwidth带 */
} RosenbrockConfig;

typedef struct {
    int enable_separable;
    float substep_ratio;
    int num_substeps;
} SymplecticConfig;

typedef struct {
    float rel_tolerance;
    float abs_tolerance;
    float min_step_size;
    float max_step_size;
    int max_iterations;
    int newton_max_iter;
    float newton_tol;
} BDF2Config;

/* ============ 现有求解器函数（保持不变） ============ */

int ode_dp54_solve(float* y, float t, float delta_t, ODERHSFunc rhs, void* ctx,
                   size_t n, const DP54Config* cfg, float* workspace,
                   float* h_actual, int* steps_used);

int ode_rosenbrock_solve(float* y, float t, float delta_t, ODERHSFunc rhs, void* ctx,
                         size_t n, const RosenbrockConfig* cfg, float* workspace,
                         float* h_actual, int* steps_used);

/* F-022: 自适应步长Rosenbrock求解器 — Richardson外推 + 步长拒绝机制 */
int ode_rosenbrock_adaptive_solve(ODERHSFunc rhs, void* ctx,
                                    float* y, size_t n,
                                    float t_start, float t_end,
                                    float h0, float tolerance,
                                    float min_step, float max_step,
                                    int max_steps, float* h_final,
                                    int* steps_taken);

int ode_forest_ruth_solve(float* q, float* p, float delta_t,
                          ODERHSFunc dqdt, ODERHSFunc dpdt, void* ctx,
                          size_t n, const SymplecticConfig* cfg,
                          float* workspace, int* steps_used);

int ode_verlet_solve(float* q, float* p, float delta_t,
                     ODERHSFunc dpdt, void* ctx,
                     size_t n, float* workspace);

int ode_bdf2_solve(float* y, float t, float delta_t, ODERHSFunc rhs, void* ctx,
                   size_t n, const BDF2Config* cfg, float* workspace,
                   float* h_actual, int* steps_used);

size_t ode_dp54_workspace_size(size_t n);
size_t ode_rosenbrock_workspace_size(size_t n);
size_t ode_forest_ruth_workspace_size(size_t n);
size_t ode_verlet_workspace_size(size_t n);
size_t ode_bdf2_workspace_size(size_t n);

DP54Config ode_dp54_default_config(void);
RosenbrockConfig ode_rosenbrock_default_config(void);
SymplecticConfig ode_symplectic_default_config(void);
BDF2Config ode_bdf2_default_config(void);

/* ============ 第2部分：统一自适应步长选择框架 ============ */

/**
 * @brief 统一自适应步长配置
 * 所有自适应求解器（RK45、DP54、Rosenbrock、BDF2等）共享此配置结构体，
 * 通过统一接口控制步长自适应行为，消除各求解器步长控制的重复实现。
 */
typedef struct {
    float rel_tolerance;             /**< 相对误差容限，默认1e-6 */
    float abs_tolerance;             /**< 绝对误差容限，默认1e-8 */
    float min_step_size;             /**< 最小步长，默认1e-8 */
    float max_step_size;             /**< 最大步长，默认1.0 */
    float safety_factor;             /**< 步长调整安全因子，默认0.9 */
    float beta_grow;                 /**< 步长增长指数（-PGROW），默认-0.2 */
    float beta_shrink;               /**< 步长收缩指数（-PSHRINK），默认-0.25 */
    int max_iterations;              /**< 最大迭代次数，默认10000 */
    int max_shrink_attempts;         /**< 最大步长收缩尝试次数，默认50 */
    int enable_step_rejection;       /**< 是否允许步长拒绝，默认1=允许 */
    int step_rejection_policy;       /**< 步长拒绝策略：0=直接减半，1=PI反馈控制，默认0 */
} AdaptiveStepConfig;

/**
 * @brief 自适应步长求解状态记录
 * 跟踪自适应求解过程中的步长变化、误差统计和收敛状态。
 */
typedef struct {
    float h_current;                 /**< 当前步长 */
    float h_previous;                /**< 前一步长 */
    float t_current;                 /**< 当前时间 */
    int total_steps;                 /**< 总步数 */
    int accepted_steps;              /**< 接受步数 */
    int rejected_steps;              /**< 拒绝步数 */
    int consecutive_rejections;      /**< 连续拒绝次数 */
    float max_error;                 /**< 最大误差（最近一步） */
    float average_error;             /**< 平均误差 */
    float pi_integral;               /**< PI反馈控制积分项 */
    float pi_previous_error;         /**< PI反馈控制前一步误差 */
    int converged;                   /**< 是否收敛 */
    int exit_flag;                   /**< 退出标志 */
} AdaptiveStepSolution;

/**
 * @brief 获取默认自适应步长配置
 * @return AdaptiveStepConfig 默认配置
 */
AdaptiveStepConfig ode_adaptive_step_default_config(void);

/**
 * @brief 初始化自适应步长求解状态
 * @param sol 状态结构体指针
 * @param t_start 起始时间
 * @param h_init 初始步长
 */
void ode_adaptive_step_init(AdaptiveStepSolution* sol, float t_start, float h_init);

/**
 * @brief 统一自适应步长控制核心函数
 * 根据局部误差估计，使用PI反馈控制或经典控制调整步长。
 * 支持步长增长、收缩和拒绝机制，兼容所有自适应求解器。
 *
 * @param max_error 当前步的归一化最大误差
 * @param h 当前步长
 * @param cfg 自适应步长配置
 * @param sol 自适应步长状态（更新h_current、总步数、拒绝次数等）
 * @return float 调整后的步长
 */
float ode_adaptive_step_control(float max_error, float h,
                                const AdaptiveStepConfig* cfg,
                                AdaptiveStepSolution* sol);

/**
 * @brief 检查步长是否可接受
 * @param max_error 归一化最大误差
 * @param cfg 自适应步长配置
 * @return int 1=接受当前步，0=拒绝需要收缩
 */
int ode_step_is_accepted(float max_error, const AdaptiveStepConfig* cfg);

/**
 * @brief 计算归一化误差
 * 跨求解器统一的误差度量：e_i = |y5_i - y4_i| / (abs_tol + rel_tol * max(|y_i|, |y5_i|))
 *
 * @param y5 高阶解（例如5阶）
 * @param y4 低阶解（例如4阶）
 * @param y 当前状态（用于缩放）
 * @param n 状态向量大小
 * @param rel_tol 相对容限
 * @param abs_tol 绝对容限
 * @param max_err_out 输出：归一化最大误差
 * @return float 归一化误差范数（均方根）
 */
float ode_compute_normalized_error(const float* y5, const float* y4,
                                   const float* y, size_t n,
                                   float rel_tol, float abs_tol,
                                   float* max_err_out);

/* ============ 第3部分：并行化ODE求解框架 ============ */

/**
 * @brief 并行计算模式枚举
 */
typedef enum {
    PARALLEL_MODE_NONE = 0,          /**< 无并行（串行执行） */
    PARALLEL_MODE_OMP = 1,           /**< OpenMP共享内存并行 */
    PARALLEL_MODE_MPI = 2,           /**< MPI分布式并行（需SELFLNN_USE_MPI） */
    PARALLEL_MODE_HYBRID = 3         /**< MPI+OpenMP混合并行 */
} ParallelMode;

/**
 * @brief 并行ODE求解器配置
 * 支持OpenMP线程并行、MPI分布式并行及其混合模式。
 * 使用域分解策略将状态向量分割到多个计算单元。
 */
typedef struct {
    ParallelMode mode;                  /**< 并行模式 */
    int num_threads;                    /**< OpenMP线程数（0=自动检测） */
    int num_domains;                    /**< 域分解数量 */
    int* domain_sizes;                  /**< 各域状态大小数组 [num_domains] */
    int* domain_offsets;                /**< 各域起始偏移数组 [num_domains] */
    int use_domain_decomposition;       /**< 是否启用域分解并行 */
    float sync_interval;                /**< 同步间隔（时间步数） */
    int enable_dynamic_load_balance;    /**< 是否启用动态负载均衡 */
    float load_balance_threshold;       /**< 负载均衡触发阈值 */
    int mpi_rank;                       /**< 当前MPI进程ID */
    int mpi_num_ranks;                  /**< MPI总进程数 */
} ParallelODERHSConfig;

/**
 * @brief 获取默认并行ODE求解配置
 * @return ParallelODERHSConfig 默认配置
 */
ParallelODERHSConfig ode_parallel_default_config(void);

/**
 * @brief 域分解初始化
 * 将n维状态空间均匀分割为num_parts个域，计算每个域的大小和偏移。
 *
 * @param n 总状态维度
 * @param num_parts 分割数量
 * @param sizes 输出：各域大小数组（需预分配num_parts个int）
 * @param offsets 输出：各域偏移数组（需预分配num_parts个int）
 * @return int 成功返回0，失败返回-1
 */
int ode_domain_decompose(size_t n, int num_parts,
                         int* sizes, int* offsets);

/**
 * @brief 并行RHS求值（OpenMP加速版）
 * 使用OpenMP parallel for对每个状态维度的RHS求值进行并行化。
 * 当未启用OpenMP编译时自动退化为串行调用。
 *
 * @param t 当前时间
 * @param y 当前状态向量
 * @param dydt 输出导数向量
 * @param rhs 原始RHS函数指针
 * @param ctx RHS函数上下文
 * @param n 状态向量大小
 * @param pcfg 并行配置（为NULL时使用串行）
 * @return int 成功返回0，失败返回RHS函数的错误码
 */
int ode_parallel_rhs_eval(float t, const float* y, float* dydt,
                          ODERHSFunc rhs, void* ctx, size_t n,
                          const ParallelODERHSConfig* pcfg);

/**
 * @brief 并行ODE求解（整体封装）
 * 使用域分解策略，将状态向量分割后在各计算单元上并行执行
 * 指定的ODE求解器。支持OpenMP线程并行和MPI进程并行。
 *
 * @param y 状态向量（输入输出）
 * @param t 起始时间
 * @param delta_t 时间步长
 * @param rhs RHS函数指针
 * @param ctx RHS函数上下文
 * @param n 状态向量大小
 * @param pcfg 并行配置
 * @param solver_func 基础ODE求解器函数
 * @param solver_cfg 求解器配置（void*通用指针）
 * @param workspace 工作空间
 * @param workspace_size 工作空间大小
 * @param h_actual 输出实际步长
 * @param steps_used 输出使用步数
 * @return int 成功返回0，失败返回错误码
 */
int ode_parallel_solve(float* y, float t, float delta_t,
                       ODERHSFunc rhs, void* ctx, size_t n,
                       const ParallelODERHSConfig* pcfg,
                       int (*solver_func)(float*, float, float, ODERHSFunc, void*,
                                          size_t, const void*, float*,
                                          float*, int*),
                       const void* solver_cfg,
                       float* workspace, size_t workspace_size,
                       float* h_actual, int* steps_used);

/**
 * @brief 获取并行求解工作空间大小
 * @param n 状态向量大小
 * @param pcfg 并行配置
 * @return size_t 所需工作空间大小（字节）
 */
size_t ode_parallel_workspace_size(size_t n, const ParallelODERHSConfig* pcfg);

/**
 * @brief MPI域同步（仅MPI模式使用）
 * 在MPI进程间同步各域的状态向量边界。
 * 当未启用SELFLNN_USE_MPI时，此函数为空操作。
 *
 * @param y 本地状态向量
 * @param n 本地状态向量大小
 * @param pcfg 并行配置
 * @return int 成功返回0
 */
int ode_mpi_sync_domains(float* y, size_t n, const ParallelODERHSConfig* pcfg);

/* ============ 第4部分：事件检测和密集输出（Dense Output） ============ */

/**
 * @brief ODE事件函数类型
 * 定义g(t, y) = 0事件。当event_value跨过零点时触发事件。
 *
 * @param t 当前时间
 * @param y 当前状态
 * @param event_value 输出事件函数值数组[num_events]
 * @param ctx 用户上下文
 * @param num_events 事件数量
 * @return int 成功返回0，失败返回-1
 */
typedef int (*ODEEventFunc)(float t, const float* y, float* event_value, void* ctx, int num_events);

/**
 * @brief 事件检测配置
 */
typedef struct {
    ODEEventFunc event_func;     /**< 事件函数指针 */
    void* event_ctx;              /**< 事件函数上下文 */
    int num_events;               /**< 事件数量 */
    const int* event_direction;   /**< 事件方向：+1=值从负到正触, -1=值从正到负, 0=任�变化, 数组[num_events] */
    float event_tolerance;        /**< 事件定位容差，默认1e-6f */
    float event_bracketing_factor;/**< 事件二分法扩展因子，默认0.1 */
    int max_events;               /**< 最大事件数，0=不限 */
    int terminate_on_event;       /**< 是否在事件发生时终止积分，1=终止, 0=继续 */
    int save_event_positions;     /**< 是否保存事件位置，1=保存 */
} ODEEventConfig;

/**
 * @brief 事件检测结果
 */
typedef struct {
    int event_detected;           /**< 是否检测到事件 */
    int event_index;              /**< 触发的事件索引(0-based) */
    float event_time;             /**< 事件发生时间 */
    float event_value;            /**< 事件函数值（最接近零点的值） */
    int event_count;              /**< 已触发事件总数 */
    int event_terminated;         /**< 是否因事件而终止积分 */
} ODEEventSolution;

/**
 * @brief DP54密集输出（4阶Hermite三次插值）
 * 在DP54步进����成后，计算[t_start, t_start+h]内任意一点τ的解。
 * 使用端点状态和导数(DP54的k1、k7)进行Hermite三次插值，精度O(h⁴)。
 *
 * @param tau 插值时间点，须在[t_start, t_start+h]范围内
 * @param t_start 步起始时间
 * @param h 步长
 * @param y_start 步起始状态[n]
 * @param y_end 步结束状态[n]
 * @param k1 起始点导数(DP54的k1)[n]
 * @param k7 结束点导数(DP54的k7)[n]
 * @param y_out 输出插值结果[n]
 * @param n 状态向量大小
 * @return int 成功返回0，参数无效返回-1
 */
int ode_dp54_dense_output(float tau, float t_start, float h,
                          const float* y_start, const float* y_end,
                          const float* k1, const float* k7,
                          float* y_out, size_t n);

/**
 * @brief DP54带事件检测的积分求解
 * 在标准DP54求解过程中对每一�接受步进行事件检测。
 * 事件触发时使用密集输出和二分法在步内精确定位事件时�。
 *
 * @param y 状态向量(输入输出)
 * @param t 起始时间
 * @param delta_t 积分时长(可正可负)
 * @param rhs RHS函数
 * @param ctx RHS上下文
 * @param n 状态向量大小
 * @param cfg DP54配置(可为NULL，使用默认)
 * @param workspace 工作空间(需ode_dp54_workspace_size_with_events(n, num_events)字节)
 * @param h_actual 输出实际积分步长(可为NULL)
 * @param steps_used 输出使用步数(可为NULL)
 * @param event_cfg 事件检测配置(为NULL时不使用事件检测，退化为标准DP54)
 * @param event_sol 事件检测结果(可为NULL)
 * @return int 成功返回0，失败返回负数错误码
 */
int ode_dp54_solve_with_events(float* y, float t, float delta_t,
                               ODERHSFunc rhs, void* ctx, size_t n,
                               const DP54Config* cfg, float* workspace,
                               float* h_actual, int* steps_used,
                               const ODEEventConfig* event_cfg,
                               ODEEventSolution* event_sol);

/**
 * @brief 带事件检测的DP54工作空间大小
 * @param n 状态向量大小
 * @param num_events 事件数量
 * @return size_t 所需工作空间大小(float个数)
 */
size_t ode_dp54_workspace_size_with_events(size_t n, int num_events);

#ifdef __cplusplus
}
#endif

#endif
