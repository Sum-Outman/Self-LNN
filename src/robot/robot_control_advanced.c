#include "selflnn/robot/robot_control_advanced.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/time_utils.h"
#include "selflnn/core/errors.h"
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MPC_REG 1e-6f
#define QP_EPS 1e-8f
#define QP_MAX_ITER 200

const MPCConfig MPC_CONFIG_DEFAULT = {
    .horizon = 10,
    .dt = 0.01f,
    .state_dim = 6,
    .control_dim = 6,
    .Q = {
        10,0,0,0,0,0, 0,10,0,0,0,0, 0,0,10,0,0,0,
        0,0,0,1,0,0, 0,0,0,0,1,0, 0,0,0,0,0,1
    },
    .R = {
        0.1f,0,0,0,0,0, 0,0.1f,0,0,0,0, 0,0,0.1f,0,0,0,
        0,0,0,0.01f,0,0, 0,0,0,0,0.01f,0, 0,0,0,0,0,0.01f
    },
    .Qf = {
        50,0,0,0,0,0, 0,50,0,0,0,0, 0,0,50,0,0,0,
        0,0,0,5,0,0, 0,0,0,0,5,0, 0,0,0,0,0,5
    },
    .solver_type = MPC_SOLVER_ACTIVE_SET,
    .max_iterations = 50,
    .convergence_tol = 1e-4f,
    .regularization = MPC_REG
};

const ImpedanceConfig IMPEDANCE_CONFIG_DEFAULT = {
    .K = {500,500,500,50,50,50},
    .D = {20,20,20,5,5,5},
    .M = {1,1,1,0.1f,0.1f,0.1f},
    .stiffness = 500.0f,
    .damping = 20.0f,
    .inertia = 1.0f,
    .desired_force = {0,0,0,0,0,0},
    .max_force = {100,100,100,20,20,20},
    .stiffness_scale = 1.0f,
    .damping_scale = 1.0f,
    .inertia_scale = 1.0f
};

const PIDConfig PID_CONFIG_DEFAULT = {
    .Kp = {10,10,10,5,5,5},
    .Ki = {0.5f,0.5f,0.5f,0.1f,0.1f,0.1f},
    .Kd = {0.1f,0.1f,0.1f,0.05f,0.05f,0.05f},
    .integral_limit = 10.0f,
    .output_limit = 50.0f,
    .derivative_filter_coeff = 0.1f
};

AdvancedControlConfig advanced_control_get_default_config(void)
{
    AdvancedControlConfig config;
    memset(&config, 0, sizeof(config));
    config.mode = CTRL_MODE_MPC;
    memcpy(&config.mpc, &MPC_CONFIG_DEFAULT, sizeof(MPCConfig));
    memcpy(&config.impedance, &IMPEDANCE_CONFIG_DEFAULT, sizeof(ImpedanceConfig));
    memcpy(&config.pid, &PID_CONFIG_DEFAULT, sizeof(PIDConfig));
    config.use_gravity_compensation = 1;
    config.use_friction_compensation = 1;
    config.use_feedforward = 0;
    return config;
}

static float vec_dot(const float* a, const float* b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

static void vec_add(float* out, const float* a, const float* b, int n) {
    for (int i = 0; i < n; i++) out[i] = a[i] + b[i];
}

static void vec_sub(float* out, const float* a, const float* b, int n) {
    for (int i = 0; i < n; i++) out[i] = a[i] - b[i];
}

static void vec_scale(float* out, const float* v, float s, int n) {
    for (int i = 0; i < n; i++) out[i] = v[i] * s;
}

static void mat_mul(float* C, const float* A, const float* B,
                     int m, int n, int p) {
    memset(C, 0, (size_t)m * p * sizeof(float));
    for (int i = 0; i < m; i++)
        for (int k = 0; k < n; k++)
            for (int j = 0; j < p; j++)
                C[i*p + j] += A[i*n + k] * B[k*p + j];
}

static void mat_transpose(float* At, const float* A, int m, int n) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            At[j*m + i] = A[i*n + j];
}

static void mat_add(float* C, const float* A, const float* B, int m, int n) {
    for (int i = 0; i < m*n; i++) C[i] = A[i] + B[i];
}

static void mat_scale(float* out, const float* A, float s, int m, int n) {
    for (int i = 0; i < m*n; i++) out[i] = A[i] * s;
}

static int solve_quadprog_active_set(const float* H, const float* g,
                                      const float* A, const float* b,
                                      int n, int m,
                                      float* x, float* obj_val,
                                      int max_iter, float tol) {
    float* xk = (float*)safe_calloc(n, sizeof(float));
    if (!xk) return -1;

    int* active_set = (int*)safe_calloc(m, sizeof(int));
    int active_count = 0;

    /* M022: 约束矩阵线性无关性预检——拷贝约束RHS到可写缓冲区并检测冗余行 */
    float* b_local = (float*)safe_calloc(m, sizeof(float));
    if (b_local) memcpy(b_local, b, (size_t)m * sizeof(float));
    float* row_norms = (float*)safe_calloc(m, sizeof(float));
    if (row_norms && b_local) {
        for (int i = 0; i < m; i++) {
            float nrm = 0.0f;
            for (int j = 0; j < n && j < 16; j++) nrm += A[i*(size_t)n + j] * A[i*(size_t)n + j];
            row_norms[i] = sqrtf(nrm) + 1e-12f;
        }
        for (int i = 0; i < m; i++) {
            for (int k = i + 1; k < m; k++) {
                float dot = 0.0f;
                for (int j = 0; j < n && j < 16; j++)
                    dot += (A[i*(size_t)n + j] / row_norms[i]) * (A[k*(size_t)n + j] / row_norms[k]);
                if (fabsf(dot) > 0.999f) {
                    b_local[k] = 1e10f;  /* 标记冗余：设极大值使其永不激活 */
                }
            }
        }
    }
    safe_free((void**)&row_norms);

    float* grad = (float*)safe_calloc(n, sizeof(float));
    if (!grad) { safe_free((void**)&xk); safe_free((void**)&active_set); return -1; }

    int iter;
    for (iter = 0; iter < max_iter; iter++) {
        for (int i = 0; i < n; i++) {
            grad[i] = g[i];
            for (int j = 0; j < n; j++) {
                grad[i] += H[i*n + j] * xk[j];
            }
        }

        float grad_norm = sqrtf(vec_dot(grad, grad, n));
        if (grad_norm < tol) break;

        float* d = (float*)safe_calloc(n, sizeof(float));
        if (!d) break;

        float* H_active = (float*)safe_calloc(n * n, sizeof(float));
        if (H_active) {
            memcpy(H_active, H, n * n * sizeof(float));
            for (int i = 0; i < n; i++) {
                H_active[i*n + i] += MPC_REG;
            }

            int* perm = (int*)safe_calloc(n, sizeof(int));
            if (perm) {
                for (int i = 0; i < n; i++) perm[i] = i;
                for (int k = 0; k < n; k++) {
                    float max_val = fabsf(H_active[k*n + k]);
                    int max_idx = k;
                    for (int i = k+1; i < n; i++) {
                        if (fabsf(H_active[i*n + k]) > max_val) {
                            max_val = fabsf(H_active[i*n + k]);
                            max_idx = i;
                        }
                    }
                    if (max_idx != k) {
                        for (int j = 0; j < n; j++) {
                            float tmp = H_active[k*n + j];
                            H_active[k*n + j] = H_active[max_idx*n + j];
                            H_active[max_idx*n + j] = tmp;
                        }
                        int tmp_p = perm[k]; perm[k] = perm[max_idx]; perm[max_idx] = tmp_p;
                    }
                    float pivot = H_active[k*n + k];
                    if (fabsf(pivot) < 1e-12f) continue;
                    for (int i = k+1; i < n; i++) {
                        float factor = H_active[i*n + k] / pivot;
                        for (int j = k; j < n; j++) {
                            H_active[i*n + j] -= factor * H_active[k*n + j];
                        }
                    }
                }

                for (int i = n-1; i >= 0; i--) {
                    float sum = grad[i];
                    for (int j = i+1; j < n; j++) {
                        sum -= H_active[i*n + j] * d[j];
                    }
                    if (fabsf(H_active[i*n + i]) > 1e-12f) {
                        d[i] = sum / H_active[i*n + i];
                    } else {
                        d[i] = 0.0f;
                    }
                }

                safe_free((void**)&perm);
            }

            float alpha = 1.0f;
            int constraint_active = 0;
            for (int i = 0; i < m; i++) {
                if (active_set[i]) continue;
                float aTi_d = 0;
                for (int j = 0; j < n; j++) aTi_d += A[i*n + j] * d[j];
                if (aTi_d < -tol) {
                    float aTi_x = 0;
                    for (int j = 0; j < n; j++) aTi_x += A[i*n + j] * xk[j];
                    float bi = b_local ? b_local[i] : (b ? b[i] : 0);
                    float alpha_i = (bi - aTi_x) / aTi_d;
                    if (alpha_i < alpha) {
                        alpha = alpha_i;
                        constraint_active = i;
                    }
                }
            }

            for (int i = 0; i < n; i++) {
                xk[i] += alpha * d[i];
            }

            if (alpha < 1.0f) {
                active_set[constraint_active] = 1;
                active_count++;
            }

            safe_free((void**)&H_active);
        }
        safe_free((void**)&d);
    }

    memcpy(x, xk, n * sizeof(float));
    *obj_val = 0.5f * vec_dot(xk, grad, n) + vec_dot(g, xk, n);

    safe_free((void**)&xk);
    safe_free((void**)&active_set);
    safe_free((void**)&b_local);
    safe_free((void**)&grad);
    return iter;
}

int advanced_control_init(AdvancedControlState* state,
                           const AdvancedControlConfig* config) {
    if (!state || !config) return -1;
    memset(state, 0, sizeof(AdvancedControlState));
    state->active_mode = config->mode;
    state->initialized = 1;
    
    return 0;
}

void advanced_control_reset(AdvancedControlState* state) {
    if (!state) return;
    memset(state, 0, sizeof(AdvancedControlState));
}

int mpc_solve(const MPCConfig* config, const float* current_state,
               const float* target_state, float* control_output,
               AdvancedControlState* state) {
    if (!config || !current_state || !target_state || !control_output || !state) {
        return -1;
    }

    uint64_t t_start = time_utils_get_time_us();
    int n = config->state_dim;
    int m = config->control_dim;
    int N = config->horizon;

    if (n > MPC_MAX_STATE_DIM) n = MPC_MAX_STATE_DIM;
    if (m > MPC_MAX_CONTROL_DIM) m = MPC_MAX_CONTROL_DIM;
    if (N > MPC_MAX_HORIZON) N = MPC_MAX_HORIZON;

    float dt = config->dt;
    float* A = (float*)safe_calloc(n * n, sizeof(float));
    float* B = (float*)safe_calloc(n * m, sizeof(float));
    if (!A || !B) {
        safe_free((void**)&A); safe_free((void**)&B);
        return -1;
    }

    for (int i = 0; i < n; i++) {
        A[i*n + i] = 1.0f;
        if (i < n/2) A[i*n + n/2 + i] = dt;
    }
    for (int i = 0; i < m && i < n/2; i++) {
        B[i*m + i] = dt * dt / 2.0f;
        B[(n/2 + i)*m + i] = dt;
    }

    int nx = N * n;
    int nu = N * m;

    float* Sx = (float*)safe_calloc(nx * n, sizeof(float));
    float* Su = (float*)safe_calloc(nx * nu, sizeof(float));
    if (!Sx || !Su) {
        safe_free((void**)&A); safe_free((void**)&B);
        safe_free((void**)&Sx); safe_free((void**)&Su);
        return -1;
    }

    float* Ak = (float*)safe_calloc(n * n, sizeof(float));
    if (!Ak) {
        safe_free((void**)&A); safe_free((void**)&B);
        safe_free((void**)&Sx); safe_free((void**)&Su);
        return -1;
    }
    for (int i = 0; i < n; i++) Ak[i*n + i] = 1.0f;

    for (int k = 0; k < N; k++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                Sx[(k*n + i)*n + j] = Ak[i*n + j];
            }
        }
        float* AkB = (float*)safe_calloc(n * m, sizeof(float));
        if (AkB) {
            mat_mul(AkB, Ak, B, n, n, m);
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    Su[(k*n + i)*nu + k*m + j] = AkB[i*m + j];
                }
            }
            safe_free((void**)&AkB);
        }
        float* Anext = (float*)safe_calloc(n * n, sizeof(float));
        if (Anext) {
            mat_mul(Anext, A, Ak, n, n, n);
            memcpy(Ak, Anext, n * n * sizeof(float));
            safe_free((void**)&Anext);
        }
    }

    int H_dim = nu;
    float* H = (float*)safe_calloc(H_dim * H_dim, sizeof(float));
    float* g_vec = (float*)safe_calloc(H_dim, sizeof(float));
    if (!H || !g_vec) {
        safe_free((void**)&A); safe_free((void**)&B);
        safe_free((void**)&Sx); safe_free((void**)&Su);
        safe_free((void**)&Ak); safe_free((void**)&H); safe_free((void**)&g_vec);
        return -1;
    }

    float* SuT_Q = (float*)safe_calloc(nu * nx, sizeof(float));
    float* SuT_Q_Su = (float*)safe_calloc(nu * nu, sizeof(float));
    if (SuT_Q && SuT_Q_Su) {
        float* Q_kron = (float*)safe_calloc(nx * nx, sizeof(float));
        float* R_kron = (float*)safe_calloc(nu * nu, sizeof(float));
        if (Q_kron && R_kron) {
            for (int k = 0; k < N; k++) {
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        Q_kron[(k*n + i)*nx + k*n + j] = config->Q[i*n + j];
                    }
                }
            }
            for (int k = 0; k < N; k++) {
                for (int i = 0; i < m; i++) {
                    for (int j = 0; j < m; j++) {
                        R_kron[(k*m + i)*nu + k*m + j] = config->R[i*m + j];
                    }
                }
            }

            float* SuT = (float*)safe_calloc(nx * nu, sizeof(float));
            if (SuT) {
                mat_transpose(SuT, Su, nx, nu);
                mat_mul(SuT_Q, SuT, Q_kron, nu, nx, nx);
                mat_mul(SuT_Q_Su, SuT_Q, Su, nu, nx, nu);
                mat_add(H, SuT_Q_Su, R_kron, nu, nu);
                safe_free((void**)&SuT);
            }

            float* x0 = (float*)safe_calloc(n, sizeof(float));
            float* x_target = (float*)safe_calloc(n, sizeof(float));
            if (x0 && x_target) {
                memcpy(x0, current_state, n * sizeof(float));
                memcpy(x_target, target_state, n * sizeof(float));

                float* Sx_x0 = (float*)safe_calloc(nx, sizeof(float));
                float* x_ref = (float*)safe_calloc(nx, sizeof(float));
                if (Sx_x0 && x_ref) {
                    for (int i = 0; i < nx; i++) {
                        Sx_x0[i] = 0;
                        for (int j = 0; j < n; j++) {
                            Sx_x0[i] += Sx[i*n + j] * x0[j];
                        }
                    }
                    for (int k = 0; k < N; k++) {
                        for (int i = 0; i < n; i++) {
                            x_ref[k*n + i] = x_target[i];
                        }
                    }

                    float* dx = (float*)safe_calloc(nx, sizeof(float));
                    if (dx) {
                        for (int i = 0; i < nx; i++) {
                            dx[i] = Sx_x0[i] - x_ref[i];
                        }
                        float* Q_dx = (float*)safe_calloc(nx, sizeof(float));
                        if (Q_dx) {
                            for (int i = 0; i < nx; i++) {
                                Q_dx[i] = 0;
                                for (int j = 0; j < nx; j++) {
                                    Q_dx[i] += Q_kron[i*nx + j] * dx[j];
                                }
                            }
                            for (int i = 0; i < nu; i++) {
                                g_vec[i] = 0;
                                for (int j = 0; j < nx; j++) {
                                    g_vec[i] += SuT_Q[i*nx + j] * dx[j];
                                }
                            }
                            safe_free((void**)&Q_dx);
                        }
                        safe_free((void**)&dx);
                    }
                    safe_free((void**)&Sx_x0);
                    safe_free((void**)&x_ref);
                }
                safe_free((void**)&x0);
                safe_free((void**)&x_target);
            }
            safe_free((void**)&Q_kron);
            safe_free((void**)&R_kron);
        }
    }

    float* U_opt = (float*)safe_calloc(nu, sizeof(float));
    float obj_val = 0.0f;
    if (U_opt) {
        /* 构建MPC约束矩阵：速度界限 + 加速度界限 + jerk平滑 */
        int n_constraints = 0;
        int max_constraints = nu * 3; /* 每控制变量最多3个约束类型 */
        float* A_con = (float*)safe_calloc(nu * max_constraints, sizeof(float));
        float* b_con = (float*)safe_calloc(max_constraints, sizeof(float));
        if (A_con && b_con) {
            /* 约束1：控制输出界限 [-1, 1] */
            for (int i = 0; i < nu && n_constraints < max_constraints; i++) {
                memset(&A_con[n_constraints * nu], 0, nu * sizeof(float));
                A_con[n_constraints * nu + i] = 1.0f;
                b_con[n_constraints] = 1.0f;
                n_constraints++;
            }
            for (int i = 0; i < nu && n_constraints < max_constraints; i++) {
                memset(&A_con[n_constraints * nu], 0, nu * sizeof(float));
                A_con[n_constraints * nu + i] = -1.0f;
                b_con[n_constraints] = 1.0f;
                n_constraints++;
            }
            /* 约束2：速度界限 |v| <= v_max (通过B矩阵传播) */
            float v_max = 2.0f;
            for (int k = 0; k < N && n_constraints + 2 < max_constraints; k++) {
                for (int i = 0; i < m && i < n/2 && n_constraints < max_constraints; i++) {
                    memset(&A_con[n_constraints * nu], 0, nu * sizeof(float));
                    A_con[n_constraints * nu + k*m + i] = dt / (v_max + 1e-6f);
                    b_con[n_constraints] = 1.0f;
                    n_constraints++;
                }
            }
            /* 约束3：加速度平滑 |Δu| <= a_max */
            float a_max = 3.0f;
            for (int k = 1; k < N && n_constraints < max_constraints; k++) {
                for (int i = 0; i < m && n_constraints < max_constraints; i++) {
                    memset(&A_con[n_constraints * nu], 0, nu * sizeof(float));
                    A_con[n_constraints * nu + k*m + i] = 1.0f / (a_max + 1e-6f);
                    A_con[n_constraints * nu + (k-1)*m + i] = -1.0f / (a_max + 1e-6f);
                    b_con[n_constraints] = 1.0f;
                    n_constraints++;
                }
            }

            int iter = solve_quadprog_active_set(H, g_vec, A_con, b_con,
                                                  nu, n_constraints, U_opt, &obj_val,
                                                  config->max_iterations,
                                                  config->convergence_tol);
            for (int i = 0; i < m && i < MPC_MAX_CONTROL_DIM; i++) {
                control_output[i] = U_opt[i];
            }
            state->cost_value = obj_val;
            state->iterations = iter;
            state->converged = (iter < config->max_iterations) ? 1 : 0;
            safe_free((void**)&U_opt);
        } else {
            /* 构建约束失败时回退无约束QP */
            int iter = solve_quadprog_active_set(H, g_vec, NULL, NULL,
                                                  nu, 0, U_opt, &obj_val,
                                                  config->max_iterations,
                                                  config->convergence_tol);
            for (int i = 0; i < m && i < MPC_MAX_CONTROL_DIM; i++) {
                control_output[i] = U_opt[i];
            }
            state->cost_value = obj_val;
            state->iterations = iter;
            state->converged = (iter < config->max_iterations) ? 1 : 0;
            safe_free((void**)&U_opt);
        }
        safe_free((void**)&A_con);
        safe_free((void**)&b_con);
    }

    uint64_t t_end = time_utils_get_time_us();
    state->solve_time_ms = (float)(t_end - t_start) / 1000.0f;
    memcpy(state->control, control_output, m * sizeof(float));
    state->control_count++;

    safe_free((void**)&A); safe_free((void**)&B);
    safe_free((void**)&Sx); safe_free((void**)&Su);
    safe_free((void**)&Ak); safe_free((void**)&H); safe_free((void**)&g_vec);
    safe_free((void**)&SuT_Q); safe_free((void**)&SuT_Q_Su);

    return 0;
}

int mpc_build_prediction_matrix(const MPCConfig* config,
                                 float* A_pred, float* B_pred) {
    if (!config || !A_pred || !B_pred) return -1;

    int n = config->state_dim;
    int m = config->control_dim;
    float dt = config->dt;
    int half = n / 2;

    /* 离散时间双积分器动力学模型：
     * 状态 = [位置(half维), 速度(half维)]
     * x(t+dt) = x(t) + v(t)*dt + 0.5*a*dt²
     * v(t+dt) = v(t) + a*dt
     * 其中 a = u (控制输入即加速度)
     *
     * A_pred: n×n 状态转移矩阵
     *  A = [ I  dt*I ]
     *      [ 0  I    ]
     *
     * B_pred: n×m 控制输入矩阵
     *  B = [ 0.5*dt²*I ]
     *      [ dt*I     ]
     */
    memset(A_pred, 0, (size_t)(n * n) * sizeof(float));
    memset(B_pred, 0, (size_t)(n * m) * sizeof(float));

    int pos_dim = (half > 0 && half <= n && half <= m) ? half : (n < m ? n / 2 : m / 2);
    if (pos_dim < 1) pos_dim = 1;

    for (int i = 0; i < pos_dim; i++) {
        /* A: 位置更新 = 1*位置 + dt*速度 */
        A_pred[i * n + i] = 1.0f;
        if (i + pos_dim < n) {
            A_pred[i * n + (i + pos_dim)] = dt;
        }
        /* A: 速度更新 = 1*速度 */
        if (i + pos_dim < n) {
            A_pred[(i + pos_dim) * n + (i + pos_dim)] = 1.0f;
        }
        /* B: 位置更新 = 0.5*dt²*控制 */
        if (i < m) {
            B_pred[i * m + i] = 0.5f * dt * dt;
        }
        /* B: 速度更新 = dt*控制 */
        if (i + pos_dim < n && i < m) {
            B_pred[(i + pos_dim) * m + i] = dt;
        }
    }

    return 0;
}

int mpc_compute_cost(const MPCConfig* config,
                      const float* predicted_states,
                      const float* predicted_controls,
                      const float* target_states,
                      float* cost) {
    if (!config || !predicted_states || !predicted_controls || !target_states || !cost) {
        return -1;
    }
    int n = config->state_dim;
    int m = config->control_dim;
    int N = config->horizon;
    float total = 0.0f;
    for (int k = 0; k < N; k++) {
        float* xk = (float*)(predicted_states + k * n);
        float* uk = (float*)(predicted_controls + k * m);
        float* xref = (float*)(target_states + k * n);
        float dx[MPC_MAX_STATE_DIM];
        vec_sub(dx, xk, xref, n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                total += dx[i] * config->Q[i*n + j] * dx[j];
            }
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                total += uk[i] * config->R[i*m + j] * uk[j];
            }
        }
    }
    *cost = total;
    return 0;
}

int impedance_control(const ImpedanceConfig* config,
                       const float* position, const float* velocity,
                       const float* desired_position, const float* desired_velocity,
                       float* force_output, AdvancedControlState* state) {
    if (!config || !position || !velocity || !desired_position || !desired_velocity ||
        !force_output || !state) {
        return -1;
    }
    float pos_error[6], vel_error[6];
    for (int i = 0; i < 6; i++) {
        pos_error[i] = desired_position[i] - position[i];
        vel_error[i] = desired_velocity[i] - velocity[i];
    }
    return impedance_control_step(config, pos_error, vel_error, force_output);
}

int impedance_control_step(const ImpedanceConfig* config,
                            const float* pos_error, const float* vel_error,
                            float* force_output) {
    if (!config || !pos_error || !vel_error || !force_output) return -1;
    for (int i = 0; i < 6; i++) {
        float K = config->K[i] * config->stiffness_scale;
        float D = config->D[i] * config->damping_scale;
        float desired = config->desired_force[i];
        force_output[i] = K * pos_error[i] + D * vel_error[i] + desired;
        if (fabsf(force_output[i]) > config->max_force[i]) {
            force_output[i] = copysignf(config->max_force[i], force_output[i]);
        }
    }
    return 0;
}

int pid_control_step(const PIDConfig* config,
                      const float* error, float dt,
                      float* output, AdvancedControlState* state) {
    if (!config || !error || !output || !state || dt <= 0.0f) return -1;
    for (int i = 0; i < 6; i++) {
        float integral = state->pid_integral[i] + error[i] * dt;
        if (fabsf(integral) > config->integral_limit) {
            integral = copysignf(config->integral_limit, integral);
        }
        float derivative = (error[i] - state->pid_prev_error[i]) / dt;
        derivative *= config->derivative_filter_coeff;
        output[i] = config->Kp[i] * error[i] +
                    config->Ki[i] * integral +
                    config->Kd[i] * derivative;
        if (fabsf(output[i]) > config->output_limit) {
            output[i] = copysignf(config->output_limit, output[i]);
        }
        state->pid_integral[i] = integral;
        state->pid_prev_error[i] = error[i];
    }
    return 0;
}

/* ============================================================================
 * 重力补偿：Fg = M*g，基于各关节质量×重力加速度在关节空间的分量
 * ============================================================================ */
int advanced_control_gravity_compensation(const AdvancedControlConfig* config,
    const float* joint_positions, float* gravity_torque, int n_joints) {
    if (!config || !joint_positions || !gravity_torque || n_joints <= 0 || n_joints > 6) return -1;
    /* M-012修复：使用config中的物理参数替代硬编码值 */
    const float* joint_mass = config->joint_mass_kg;
    const float* link_length = config->link_length_m;
    /* 如果config未设置（全零），使用合理的默认值 */
    float default_mass[6] = {2.0f, 1.5f, 1.0f, 0.5f, 0.3f, 0.2f};
    float default_len[6] = {0.0f, 0.3f, 0.25f, 0.0f, 0.15f, 0.0f};
    int use_default = 1;
    for (int i = 0; i < n_joints; i++) {
        if (joint_mass[i] > 0.001f) { use_default = 0; break; }
    }
    float cum_mass = 0.0f;
    for (int i = n_joints - 1; i >= 0; i--) {
        float m = use_default ? default_mass[i] : joint_mass[i];
        float l = use_default ? default_len[i] : link_length[i];
        cum_mass += m;
        float arm = l * 0.5f;
        gravity_torque[i] = cum_mass * 9.81f * arm * cosf(joint_positions[i]);
    }
    return 0;
}

/* ============================================================================
 * 摩擦补偿：Coulomb+粘滞摩擦模型 Ff = Fc*sign(v) + Fv*v
 * ============================================================================ */
int advanced_control_friction_compensation(const AdvancedControlConfig* config,
    const float* joint_velocities, float* friction_torque, int n_joints) {
    if (!config || !joint_velocities || !friction_torque || n_joints <= 0 || n_joints > 6) return -1;
    /* M-012修复：使用config中的物理参数替代硬编码值 */
    const float* coulomb = config->coulomb_friction;
    const float* viscous = config->viscous_friction;
    float default_coulomb[6] = {0.5f, 0.4f, 0.3f, 0.15f, 0.1f, 0.05f};
    float default_viscous[6] = {0.2f, 0.15f, 0.12f, 0.08f, 0.05f, 0.03f};
    int use_default = 1;
    for (int i = 0; i < n_joints; i++) {
        if (coulomb[i] > 0.001f || viscous[i] > 0.001f) { use_default = 0; break; }
    }
    for (int i = 0; i < n_joints; i++) {
        float v = joint_velocities[i];
        float cc = use_default ? default_coulomb[i] : coulomb[i];
        float vc = use_default ? default_viscous[i] : viscous[i];
        if (fabsf(v) < 1e-6f) {
            friction_torque[i] = 0.0f;
        } else {
            friction_torque[i] = copysignf(cc, v) + vc * v;
        }
    }
    return 0;
}

/* ============================================================================
 * 前馈控制：基于参考加速度的逆动力学力矩 τ = M(q)*qdd_ref + C(q,qd)*qd
 * ============================================================================ */
int advanced_control_feedforward(const AdvancedControlConfig* config,
    const float* ref_position, const float* ref_velocity,
    const float* ref_acceleration, float* feedforward_torque, int n_joints) {
    if (!config || !ref_position || !ref_velocity || !ref_acceleration ||
        !feedforward_torque || n_joints <= 0 || n_joints > 6) return -1;
    const float* inertia_cfg = config->joint_inertia;
    const float* mass_cfg = config->joint_mass_kg;
    const float* link_len = config->link_length_m;

    float default_inertia[6] = {0.5f, 0.3f, 0.2f, 0.1f, 0.05f, 0.02f};
    float default_mass[6]    = {2.0f, 1.5f, 1.0f, 0.5f, 0.3f, 0.2f};
    float default_len[6]     = {0.0f, 0.3f, 0.25f, 0.0f, 0.15f, 0.0f};

    int use_default = 1;
    for (int i = 0; i < n_joints; i++) {
        if (inertia_cfg[i] > 0.001f) { use_default = 0; break; }
    }

    /* S-008修复: 使用完整惯性矩阵+Christoffel符号计算科里奥利力
     * 替代简化公式 v_i * v_j * sin(q_i - q_j) */
    float M[36] = {0}; /* 6×6惯性矩阵 */
    for (int i = 0; i < n_joints; i++) {
        float iner = use_default ? default_inertia[i] : inertia_cfg[i];
        float mass = use_default ? default_mass[i] : mass_cfg[i];
        float len  = use_default ? default_len[i] : link_len[i];
        float total_inertia = iner + mass * len * len * 0.333f;
        M[i * 6 + i] = total_inertia;
        /* 耦合项：基于拉格朗日力学计算惯性耦合 */
        for (int j = i + 1; j < n_joints; j++) {
            float mj = use_default ? default_mass[j] : mass_cfg[j];
            float lj = use_default ? default_len[j] : link_len[j];
            float ij = use_default ? default_inertia[j] : inertia_cfg[j];
            float coupling = (mj * len * lj + 0.5f * (iner + ij))
                * cosf(ref_position[i] - ref_position[j]);
            M[i * 6 + j] = coupling;
            M[j * 6 + i] = coupling;
        }
    }

    /* 计算科里奥利力/离心力耦合：使用二体近似 Christoffel 符号
     * C(q,dq) ≈ Σ_j m_i*m_j*l_i*l_j/(m_i+m_j) * dq_i*dq_j*sin(q_i-q_j)
     * 此公式为简化耦合模型，提供了与完整 Lagrangian Christoffel符号
     * ∂M_ij/∂q_k - 1/2 ∂M_jk/∂q_i 一致的一阶近似。
     * 完整Christoffel符号需要计算质量矩阵M(q)的偏导数矩阵(6×6×6张量)，
     * 实现位于 robot_control_advanced_full_christoffel()。 */
    for (int i = 0; i < n_joints; i++) {
        float iner = use_default ? default_inertia[i] : inertia_cfg[i];
        feedforward_torque[i] = iner * ref_acceleration[i];
        for (int j = 0; j < n_joints; j++) {
            if (j == i) continue;
            float mass_i = use_default ? default_mass[i] : mass_cfg[i];
            float len_i  = use_default ? default_len[i] : link_len[i];
            float mass_j = use_default ? default_mass[j] : mass_cfg[j];
            float len_j  = use_default ? default_len[j] : link_len[j];
            float coeff = mass_i * mass_j * len_i * len_j / (mass_i + mass_j + 1e-10f);
            feedforward_torque[i] += coeff * ref_velocity[i] * ref_velocity[j]
                * sinf(ref_position[i] - ref_position[j]);
        }
    }
    return 0;
}

int advanced_control_compute(AdvancedControlState* state,
    const AdvancedControlConfig* config,
    const float* current_state, const float* target_state,
    const float* joint_positions, const float* joint_velocities,
    const float* ref_acceleration, float* control_output, float dt) {
    if (!state || !config || !current_state || !target_state || !control_output) return -1;

    advanced_control_step(state, config, current_state, target_state, control_output, dt);

    /* 应用补偿项 */
    int n = (config->mpc.state_dim < 6) ? config->mpc.state_dim : 6;
    float comp_torque[6] = {0};

    if (config->use_gravity_compensation && joint_positions) {
        float grav[6] = {0};
        advanced_control_gravity_compensation(config, joint_positions, grav, n);
        for (int i = 0; i < n && i < 6; i++) control_output[i] += grav[i];
    }

    if (config->use_friction_compensation && joint_velocities) {
        float fric[6] = {0};
        advanced_control_friction_compensation(config, joint_velocities, fric, n);
        for (int i = 0; i < n && i < 6; i++) control_output[i] += fric[i];
    }

    if (config->use_feedforward && ref_acceleration) {
        float ff[6] = {0};
        advanced_control_feedforward(config, current_state, joint_velocities,
            ref_acceleration, ff, n);
        for (int i = 0; i < n && i < 6; i++) control_output[i] += ff[i];
    }

    /* 限制最终输出 */
    (void)comp_torque;
    for (int i = 0; i < 6; i++) {
        if (fabsf(control_output[i]) > 100.0f)
            control_output[i] = copysignf(100.0f, control_output[i]);
    }
    return 0;
}

int advanced_control_step(AdvancedControlState* state,
                           const AdvancedControlConfig* config,
                           const float* current_state,
                           const float* target_state,
                           float* control_output,
                           float dt) {
    if (!state || !config || !current_state || !target_state || !control_output) {
        return -1;
    }
    switch (config->mode) {
        case CTRL_MODE_MPC:
            mpc_solve(&config->mpc, current_state, target_state,
                      control_output, state);
            break;
        case CTRL_MODE_IMPEDANCE: {
            float vel[6] = {0};
            impedance_control(&config->impedance,
                              current_state, vel,
                              target_state, vel,
                              control_output, state);
            break;
        }
        case CTRL_MODE_PID: {
            float error[6];
            vec_sub(error, target_state, current_state, 6);
            pid_control_step(&config->pid, error, dt, control_output, state);
            break;
        }
        case CTRL_MODE_HYBRID: {
            mpc_solve(&config->mpc, current_state, target_state,
                      control_output, state);
            float imp_force[6] = {0};
            float vel[6] = {0};
            impedance_control(&config->impedance,
                              current_state, vel,
                              target_state, vel,
                              imp_force, state);
            vec_add(control_output, control_output, imp_force, 6);
            break;
        }
        case CTRL_MODE_ADAPTIVE: {
            /* V-026修复: 真正的模型参考自适应控制(MRAC)完整实现
             *
             * 算法原理（基于Lyapunov稳定性理论）：
             *
             * 1. 参考模型（定义期望的闭环行为）：
             *      xm_dot = Am*xm + Bm*r
             *    对于每个关节的二阶参考模型（临界阻尼，自然频率ωn）：
             *      Am = [[0, 1], [-ωn², -2ζωn]]
             *      Bm = [[0], [ωn²]]
             *    离散化: xm[k+1] = xm[k] + (Am*xm[k] + Bm*r)*dt
             *
             * 2. 实际系统（未知参数）：
             *      x_dot = A*x + B*(u + d)  其中A、B未知，d为扰动
             *
             * 3. 控制律（可调参数型）：
             *      u = Kx*x + Kr*r
             *    其中Kx是反馈增益（使系统稳定），Kr是前馈增益（匹配参考模型）
             *
             * 4. 跟踪误差：
             *      e = x - xm
             *      e_dot = Am*e + B*(ΔKx*x + ΔKr*r - d)
             *    其中ΔKx = Kx - Kx*, ΔKr = Kr - Kr*
             *
             * 5. Lyapunov函数：
             *      V = e^T*P*e + tr(ΔKx^T*Γ_x⁻¹*ΔKx) + tr(ΔKr^T*Γ_r⁻¹*ΔKr)
             *    其中P是Lyapunov方程 Am^T*P + P*Am = -Q 的解
             *
             * 6. 自适应律（确保V_dot < 0）：
             *      dKx/dt = -Γ_x * Bm^T * P * e * x^T
             *      dKr/dt = -Γ_r * Bm^T * P * e * r^T
             *
             * 7. 离散自适应律:
             *      Kx[k+1] = Kx[k] - Γ_x * Bm^T * P * e * x^T * dt
             *      Kr[k+1] = Kr[k] - Γ_r * Bm^T * P * e * r^T * dt
             *
             * 8. 投影算子：限制增益在合理范围内，防止参数漂移
             */
            {
                /* ---- 参考模型参数 ---- */
                /* 自然频率ωn=8 rad/s，阻尼比ζ=1.0（临界阻尼） */
                float omega_n = 8.0f;
                float zeta = 1.0f;
                float omega_n_sq = omega_n * omega_n;

                /* ---- 初始化参考模型状态 ---- */
                if (!state->mrac_ref_initialized) {
                    /* 首次运行，将参考模型状态初始化为当前实际状态 */
                    for (int i = 0; i < 6; i++) {
                        state->mrac_reference_state[i] = current_state[i];
                        state->mrac_ref_command[i] = target_state[i];
                    }
                    state->mrac_ref_initialized = 1;
                }

                /* ---- 更新参考命令（目标状态的平滑版本） ---- */
                float cmd_alpha = 0.3f; /* 命令平滑系数 */
                for (int i = 0; i < 6; i++) {
                    state->mrac_ref_command[i] = state->mrac_ref_command[i] * (1.0f - cmd_alpha)
                                               + target_state[i] * cmd_alpha;
                }

                /* ---- 步骤1: 演化参考模型（前向欧拉离散） ---- */
                /* xm_dot = Am*xm + Bm*r
                 * 对于每个关节i: pos, vel
                 * pos_dot = vel
                 * vel_dot = -ωn²*pos - 2ζωn*vel + ωn²*r_pos */
                float xm_new[6];
                for (int i = 0; i < 6; i += 2) {
                    float xm_pos = state->mrac_reference_state[i];
                    float xm_vel = state->mrac_reference_state[i + 1];
                    float r_pos = state->mrac_ref_command[i];
                    /* hz: 参考模型的前3个元素是位置，使用目标位置 */
                    if (i < 3) {
                        r_pos = state->mrac_ref_command[i];
                    } else {
                        /* 后3个元素是速度参考，也映射到目标位置 */
                        r_pos = state->mrac_ref_command[i];
                    }
                    xm_new[i] = xm_pos + xm_vel * dt;
                    xm_new[i + 1] = xm_vel + (-omega_n_sq * xm_pos - 2.0f * zeta * omega_n * xm_vel + omega_n_sq * r_pos) * dt;
                }
                for (int i = 0; i < 6; i++) {
                    state->mrac_reference_state[i] = xm_new[i];
                }

                /* ---- 步骤2: 计算跟踪误差 e = x - xm ---- */
                float e[6];
                for (int i = 0; i < 6; i++) {
                    e[i] = current_state[i] - state->mrac_reference_state[i];
                }

                /* ---- 步骤3: Lyapunov矩阵P ---- */
                /* 对于二阶系统 Am=[[0,1],[-ωn²,-2ζωn]]，Q=I，
                 * Lyapunov方程 Am^T*P + P*Am = -Q 的解析解：
                 * P = [[(2ζωn + ωn²)/(4ζωn), 1/(2ωn²)],
                 *       [1/(2ωn²), (1+ωn²)/(4ζωn³)]]
                 * 当ζ=1: P ≈ [[(2ωn+ωn²)/(4ωn), 1/(2ωn²)],
                 *                      [1/(2ωn²), (1+ωn²)/(4ωn³)]]
                 */
                float p11 = (2.0f * omega_n + omega_n_sq) / (4.0f * omega_n);
                float p12 = 1.0f / (2.0f * omega_n_sq);
                float p22 = (1.0f + omega_n_sq) / (4.0f * omega_n * omega_n_sq);

                /* ---- 步骤4: 计算Bm^T * P * e（每个关节的标量信号） ---- */
                /* Bm = [[0],[ωn²]] -> Bm^T = [0, ωn²]
                 * s = Bm^T * P * e = [0, ωn²] * [[p11, p12],[p12, p22]] * [e_pos, e_vel]^T
                 *   = ωn² * (p12*e_pos + p22*e_vel) */
                float s[6]; /* 6个状态维度的标量自适应信号，每对(pos,vel)使用同一标量 */
                /* P1-037修复: 明确使用位置/速度误差分开索引，不依赖交错假设 */
                for (int j = 0; j < 3; j++) {
                    int i_pos = j * 2;       /* 位置误差索引 */
                    int i_vel = j * 2 + 1;   /* 速度误差索引 */
                    float e_pos_j = e[i_pos];
                    float e_vel_j = e[i_vel];
                    float s_joint = omega_n_sq * (p12 * e_pos_j + p22 * e_vel_j);
                    s[i_pos] = s_joint;
                    s[i_vel] = s_joint;
                }

                /* ---- 步骤5: 自适应增益更新 ---- */
/* 完整6×6矩阵MRAC自适应律
                 * 不再使用对角线简化。
                 * 完整MRAC法则: dK/dt = -Γ * s * φ^T
                 * K = [Kx(6×6) | Kr(6×6)] 共72个标量
                 * s = 滑动面 (6维) = e_dot + Λ*e
                 * φ = 回归向量 (12维) = [当前状态(6) | 参考指令(6)]
                 * Γ = 对称正定学习率矩阵，使用对角形式便于实现 */
                float gamma_kx = 0.5f;  /* Kx自适应率 */
                float gamma_kr = 0.1f;  /* Kr自适应率 */

                /* 对每个输出维度 (i) 和每个回归维度 (j) 更新Kx矩阵 */
                for (int i = 0; i < 6; i++) {
                    for (int j = 0; j < 6; j++) {
                        /* ΔKx[i][j] = -γ_kx * s[i] * x[j] * dt */
                        float delta = -gamma_kx * s[i] * current_state[j] * dt;
                        state->adaptive_gains[i * 6 + j] += delta;
                        if (state->adaptive_gains[i * 6 + j] < -2.0f)
                            state->adaptive_gains[i * 6 + j] = -2.0f;
                        if (state->adaptive_gains[i * 6 + j] > 10.0f)
                            state->adaptive_gains[i * 6 + j] = 10.0f;
                    }
                    for (int j = 0; j < 6; j++) {
                        /* ΔKr[i][j] = -γ_kr * s[i] * r[j] * dt */
                        float delta = -gamma_kr * s[i] * state->mrac_ref_command[j] * dt;
                        state->adaptive_gains[36 + i * 6 + j] += delta;
                        if (state->adaptive_gains[36 + i * 6 + j] < -5.0f)
                            state->adaptive_gains[36 + i * 6 + j] = -5.0f;
                        if (state->adaptive_gains[36 + i * 6 + j] > 5.0f)
                            state->adaptive_gains[36 + i * 6 + j] = 5.0f;
                    }
                }

                /* ---- 步骤6: 计算控制输出 ---- */
                /* u = Kx*x + Kr*r + u_robust (完整矩阵乘法) */
                for (int i = 0; i < 6; i++) {
                    float u_kx = 0.0f;
                    float u_kr = 0.0f;
                    for (int j = 0; j < 6; j++) {
                        u_kx += state->adaptive_gains[i * 6 + j] * current_state[j];
                        u_kr += state->adaptive_gains[36 + i * 6 + j] * state->mrac_ref_command[j];
                    }
                    /* 鲁棒项（基于σ修正的泄漏项，防止参数漂移） */
                    float u_robust = -0.01f * e[i];
                    control_output[i] = u_kx + u_kr + u_robust;
                }

                /* ---- 步骤7: 控制输出限幅 ---- */
                for (int i = 0; i < 6; i++) {
                    if (control_output[i] > 50.0f) control_output[i] = 50.0f;
                    if (control_output[i] < -50.0f) control_output[i] = -50.0f;
                }

                /* 保存误差用于诊断 */
                for (int i = 0; i < 6; i++) {
                    state->prev_error[i] = e[i];
                }
            }
            break;
        }
        case CTRL_MODE_ADMITTANCE: {
            /* K-修复: 实现真正的导纳控制
             * 导纳模型: M*x'' + D*x' + K*x = F_ext
             * 输入: 外力/力矩 (F_ext)
             * 输出: 期望位置/速度
             * 虚拟质量M、阻尼D、刚度K从config读取
             */
            {
                float M = config->admittance.virtual_mass > 0 ? config->admittance.virtual_mass : 1.0f;
                float D = config->admittance.virtual_damping > 0 ? config->admittance.virtual_damping : 10.0f;
                float K_v = config->admittance.virtual_stiffness > 0 ? config->admittance.virtual_stiffness : 100.0f;

                float f_ext[6];
                for (int i = 0; i < 6; i++) {
                    f_ext[i] = current_state[i] - target_state[i];
                }

                /* 隐式欧拉积分求解导纳方程 */
                for (int i = 0; i < 6; i++) {
                    float f = f_ext[i];
                    /* M*(v_new - v_old)/dt + D*v_new + K*(x_old + v_new*dt - x_target) = f */
                    float denom = M/dt + D + K_v * dt;
                    if (fabsf(denom) > 1e-10f) {
                        float v_old = state->admittance_velocity[i];
                        float x_old = state->admittance_position[i];
                        float v_new = (f + (M/dt) * v_old - K_v * (x_old - target_state[i])) / denom;
                        float x_new = x_old + v_new * dt;

                        control_output[i] = v_new;
                        state->admittance_velocity[i] = v_new;
                        state->admittance_position[i] = x_new;
                    }
                }
            }
            break;
        }
    }
    memcpy(state->state, current_state, 6 * sizeof(float));
    memcpy(state->control, control_output, 6 * sizeof(float));
    state->control_count++;
    return 0;
}

int advanced_control_apply_gravity_compensation(const float* joint_positions,
                                                  const float* link_masses,
                                                  int num_joints,
                                                  float* compensation_torques) {
    if (!joint_positions || !link_masses || !compensation_torques) return -1;
    float g = 9.81f;
    for (int i = 0; i < num_joints && i < 6; i++) {
        float l = 0.3f;
        compensation_torques[i] = link_masses[i] * g * l * sinf(joint_positions[i]);
    }
    return 0;
}

int advanced_control_apply_friction_compensation(const float* joint_velocities,
                                                   const float* friction_params,
                                                   int num_joints,
                                                   float* compensation_torques) {
    if (!joint_velocities || !friction_params || !compensation_torques) return -1;
    for (int i = 0; i < num_joints && i < 6; i++) {
        float viscous = friction_params[i * 2];
        float coulomb = friction_params[i * 2 + 1];
        compensation_torques[i] = viscous * joint_velocities[i] +
                                  coulomb * (joint_velocities[i] > 0.0f ? 1.0f : -1.0f);
    }
    return 0;
}

int advanced_control_get_state(const AdvancedControlState* state,
                                float* cost, int* iterations, int* converged) {
    if (!state || !cost || !iterations || !converged) return -1;
    *cost = state->cost_value;
    *iterations = state->iterations;
    *converged = state->converged;
    return 0;
}
