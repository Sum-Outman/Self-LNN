#ifndef SELFLNN_ROBOT_CONTROL_ADVANCED_H
#define SELFLNN_ROBOT_CONTROL_ADVANCED_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MPC_MAX_HORIZON 50
#define MPC_MAX_STATE_DIM 12
#define MPC_MAX_CONTROL_DIM 6
#define MPC_MAX_CONSTRAINTS 20

typedef enum {
    CTRL_MODE_PID = 0,
    CTRL_MODE_MPC = 1,
    CTRL_MODE_IMPEDANCE = 2,
    CTRL_MODE_ADMITTANCE = 3,
    CTRL_MODE_HYBRID = 4,
    CTRL_MODE_ADAPTIVE = 5
} AdvancedControlMode;

typedef enum {
    MPC_SOLVER_ACTIVE_SET = 0,
    MPC_SOLVER_INTERIOR_POINT = 1,
    MPC_SOLVER_GRADIENT = 2
} MPCSolverType;

typedef struct {
    float Q[MPC_MAX_STATE_DIM * MPC_MAX_STATE_DIM];
    float R[MPC_MAX_CONTROL_DIM * MPC_MAX_CONTROL_DIM];
    float Qf[MPC_MAX_STATE_DIM * MPC_MAX_STATE_DIM];
    int horizon;
    float dt;
    int state_dim;
    int control_dim;
    float state_limits_low[MPC_MAX_STATE_DIM];
    float state_limits_high[MPC_MAX_STATE_DIM];
    float control_limits_low[MPC_MAX_CONTROL_DIM];
    float control_limits_high[MPC_MAX_CONTROL_DIM];
    float delta_control_limits[MPC_MAX_CONTROL_DIM];
    MPCSolverType solver_type;
    int max_iterations;
    float convergence_tol;
    float regularization;
} MPCConfig;

typedef struct {
    float K[6];
    float D[6];
    float M[6];
    float stiffness;
    float damping;
    float inertia;
    float desired_force[6];
    float max_force[6];
    float stiffness_scale;
    float damping_scale;
    float inertia_scale;
} ImpedanceConfig;

typedef struct {
    float Kp[6];
    float Ki[6];
    float Kd[6];
    float integral_limit;
    float output_limit;
    float derivative_filter_coeff;
} PIDConfig;

typedef struct {
    AdvancedControlMode mode;
    MPCConfig mpc;
    ImpedanceConfig impedance;
    PIDConfig pid;
    float target_position[3];
    float target_orientation[4];
    float target_velocity[3];
    float target_force[6];
    int use_gravity_compensation;
    int use_friction_compensation;
    int use_feedforward;
    float feedforward_torques[6];
} AdvancedControlConfig;

typedef struct {
    int initialized;
    AdvancedControlMode active_mode;
    float* predicted_states;
    float* predicted_controls;
    int prediction_horizon;
    float cost_value;
    int iterations;
    int converged;
    float solve_time_ms;
    float state[MPC_MAX_STATE_DIM];
    float control[MPC_MAX_CONTROL_DIM];
    float impedance_force[6];
    float pid_integral[6];
    float pid_prev_error[6];
    float desired_stiffness[6];
    float desired_damping[6];
    float estimated_disturbance[6];
    int control_count;
} AdvancedControlState;

extern const MPCConfig MPC_CONFIG_DEFAULT;
extern const ImpedanceConfig IMPEDANCE_CONFIG_DEFAULT;
extern const PIDConfig PID_CONFIG_DEFAULT;

AdvancedControlConfig advanced_control_get_default_config(void);

int advanced_control_init(AdvancedControlState* state,
                           const AdvancedControlConfig* config);
void advanced_control_reset(AdvancedControlState* state);

int mpc_solve(const MPCConfig* config, const float* current_state,
               const float* target_state, float* control_output,
               AdvancedControlState* state);
int mpc_build_prediction_matrix(const MPCConfig* config,
                                 float* A_pred, float* B_pred);
int mpc_compute_cost(const MPCConfig* config,
                      const float* predicted_states,
                      const float* predicted_controls,
                      const float* target_states,
                      float* cost);

int impedance_control(const ImpedanceConfig* config,
                       const float* position, const float* velocity,
                       const float* desired_position, const float* desired_velocity,
                       float* force_output, AdvancedControlState* state);
int impedance_control_step(const ImpedanceConfig* config,
                            const float* pos_error, const float* vel_error,
                            float* force_output);

int pid_control_step(const PIDConfig* config,
                      const float* error, float dt,
                      float* output, AdvancedControlState* state);

int advanced_control_step(AdvancedControlState* state,
                           const AdvancedControlConfig* config,
                           const float* current_state,
                           const float* target_state,
                           float* control_output,
                           float dt);
int advanced_control_apply_gravity_compensation(const float* joint_positions,
                                                  const float* link_masses,
                                                  int num_joints,
                                                  float* compensation_torques);
int advanced_control_apply_friction_compensation(const float* joint_velocities,
                                                   const float* friction_params,
                                                   int num_joints,
                                                   float* compensation_torques);
int advanced_control_get_state(const AdvancedControlState* state,
                                float* cost, int* iterations, int* converged);

/* 高级控制完整入口：MPC/Impedance/PID → 重力+摩擦+前馈补偿 */
int advanced_control_compute(AdvancedControlState* state,
    const AdvancedControlConfig* config,
    const float* current_state, const float* target_state,
    const float* joint_positions, const float* joint_velocities,
    const float* ref_acceleration, float* control_output, float dt);

int advanced_control_gravity_compensation(const AdvancedControlConfig* config,
    const float* joint_positions, float* gravity_torque, int n_joints);

int advanced_control_friction_compensation(const AdvancedControlConfig* config,
    const float* joint_velocities, float* friction_torque, int n_joints);

int advanced_control_feedforward(const AdvancedControlConfig* config,
    const float* ref_position, const float* ref_velocity,
    const float* ref_acceleration, float* feedforward_torque, int n_joints);

#ifdef __cplusplus
}
#endif

#endif
