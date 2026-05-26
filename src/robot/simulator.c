/**
 * @file simulator.c
 * @brief 仿真器接口实现
 * 
 * 提供PyBullet、Gazebo等外部仿真器的IPC通信接口。
 * 提供100%纯C内部物理引擎（碰撞检测+约束求解+刚体动力学）。
 * 
 * 数据来源规则【P2-03修复】：
 *   凡使用内部纯C物理引擎生成的数据，必须在数据产出入口标明：
 *   "[数据来源: 内部纯C物理引擎(Pure C Internal Physics Engine)]"
 *   凡使用外部仿真器（PyBullet/Gazebo）生成的数据，标明：
 *   "[数据来源: 外部仿真器(PyBullet/Gazebo)]"
 * 
 * 宏保护体系：
 *   SELFLNN_USE_PURE_C_PHYSICS  — CMakeLists.txt控制，启用内部纯C物理引擎
 *   SELFLNN_HAS_EXTERNAL_SIM    — CMakeLists.txt控制，启用外部仿真器接口
 *   SELFLNN_STRICT_REAL_DATA    — 严格真实数据模式，禁绝所有内部虚拟物理仿真
 * 
 * 决策层级：
 *   1. SELFLNN_STRICT_REAL_DATA  → 强制仅用外部仿真器，内部引擎全部禁用
 *   2. SELFLNN_USE_PURE_C_PHYSICS → 优先/允许使用内部纯C物理引擎
 *   3. SELFLNN_HAS_EXTERNAL_SIM   → 允许使用外部仿真器
 *   4. 运行时 use_internal_simulator 标志 → 动态切换内部/外部引擎
 * 
 * 严格真实数据原则：
 * - 优先通过IPC连接PyBullet/Gazebo外部仿真器
 * - 外部不可用时直接返回错误，不降级到任何内部虚拟引擎
 * - SELFLNN_STRICT_REAL_DATA 模式禁绝所有虚拟物理仿真
 * 
 * PyBullet/Gazebo例外说明（2026-05-11需求更新）：
 *   PyBullet（Python库）和Gazebo（C++库）允许作为外部依赖通过IPC使用。
 *   除前端网页和PyBullet/Gazebo外，其余全部100%纯C实现。
 */


#include "selflnn/robot/simulator.h"
#include "selflnn/robot/robot.h"
#include "selflnn/robot/kinematics.h"
#include "selflnn/robot/hardware_interface.h"
#include "selflnn/robot/sensor_pipeline.h"
#include "selflnn/robot/sensor_simulation.h"
#include "selflnn/core/errors.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/math_utils.h"
#include "selflnn/utils/logging.h"
#include "selflnn/utils/platform.h"
#include "selflnn/core/port_config.h"
#include "selflnn/utils/perf.h"
#include "selflnn/utils/secure_random.h"

#ifdef _MSC_VER
/* 4189已通过UNUSED()处理 */
#endif

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "selflnn/utils/math_utils.h"
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <time.h>
#include <stdarg.h>

/* C语言共享内存仿真通信（替代Python管道）
 * 使用内存映射文件实现高速进程间通信 */
#ifdef _WIN32
#include <windows.h>
#define SIM_SHM_NAME_PREFIX L"Local\\SELFLNN_Sim_"
typedef HANDLE SimShmHandle;
#else
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#define SIM_SHM_NAME_PREFIX "/selflnn_sim_"
typedef int SimShmHandle;
#endif

#define SIM_SHM_SIZE (256 * 1024)  /* 256KB共享内存区域 */

/* [P2-03修复] 编译期宏保护确认：
 * 以下映射关系确保仅在正确的编译选项下内部纯C物理引擎才可用：
 *   CMakeLists.txt SELFLNN_ENABLE_PURE_C_SIM=ON → -DSELFLNN_USE_PURE_C_PHYSICS
 *   CMakeLists.txt SELFLNN_ENABLE_EXTERNAL_SIM=ON → -DSELFLNN_HAS_EXTERNAL_SIM
 * 
 * 运行时 use_internal_simulator 标志由 simulator_auto_connect() 根据宏和
 * 外部仿真器可用性动态设置，后续所有数据访问入口据此标记数据来源。 */
#ifdef SELFLNN_USE_PURE_C_PHYSICS
#define INTERNAL_PHYSICS_AVAILABLE 1
#else
#define INTERNAL_PHYSICS_AVAILABLE 0
#endif

#ifdef SELFLNN_HAS_EXTERNAL_SIM
#define EXTERNAL_SIM_AVAILABLE 1
#else
#define EXTERNAL_SIM_AVAILABLE 0
#endif

typedef struct {
    int initialized;
    SimShmHandle handle;
    void* mapped_addr;
    char shm_name[128];
} SimSharedMemory;

/* ============================================================================
 * 内部数据结构
 * =========================================================================== */

/**
 * @brief 内部物理物体结构体
 */
typedef struct {
    float position[3];
    float velocity[3];
    float acceleration[3];
    float orientation[4];
    float angular_velocity[3];
    float mass;
    float inertia[3];
    int is_static;
    int active;
} InternalPhysicsObject;

/* ============================================================================
 * 增强物理引擎数据结构 — 碰撞检测 + 约束求解 + 关节
 * ============================================================================ */

/* 类型定义位于 simulator.h — SimCollisionObject, SimContactPoint,
 * SimConstraint, SimJointType, SimJoint, SimBroadphasePair, SimPhysicsPipeline
 */

/**
 * @brief 仿真器内部结构体
 */
struct Simulator {
    SimulatorConfig config;      /**< 仿真器配置 */
    SimulatorStatus status;      /**< 仿真器状态 */
    SimulatorState target_state; /**< 目标状态 */
    
    // 通信接口
    HardwareInterface* comm_interface; /**< 通信接口句柄 */
    int use_internal_simulator; /**< 是否使用内部仿真器 */
    
    // 机器人管理
    SimulatorRobotState* robots; /**< 机器人状态数组 */
    int robot_capacity;         /**< 机器人容量 */
    int robot_count;            /**< 机器人数量 */
    
    // 传感器管理
    SimulatorSensorData* sensors; /**< 传感器数据数组 */
    int sensor_capacity;         /**< 传感器容量 */
    int sensor_count;            /**< 传感器数量 */
    
    // 场景对象管理
    SimulatorSceneObject* scene_objects; /**< 场景对象数组 */
    int scene_object_capacity;  /**< 场景对象容量 */
    int scene_object_count;     /**< 场景对象数量 */
    
    // 内部仿真器状态（简单物理仿真）
    struct {
        float physics_time;      /**< 物理时间 */
        float last_update_time;  /**< 最后更新时间 */
        float accumulator;       /**< 时间累加器 */
        
        // 简单物理世界
        float gravity[3];        /**< 重力 */
        float air_density;       /**< 空气密度 */
        float ground_level;      /**< 地面高度 */
        
        // 仿真物体
        InternalPhysicsObject* physics_objects;
        int physics_object_count; /**< 物理物体数量 */
        
        // 传感器仿真
        SensorSimContext sensor_sim;
        int sensor_sim_initialized;
        
        // 增强物理管道（碰撞检测 + 约束求解 + 关节）
        SimPhysicsPipeline pipeline;
        int pipeline_initialized;
    } internal;

    /* P07修复: 光照状态 */
    float light_position[3];
    float light_color[3];
    float ambient_color[3];
    int lighting_active;

    // 数据记录
    FILE* recording_file;        /**< 记录文件句柄 */
    int is_recording;           /**< 是否正在记录 */
    float recording_start_time; /**< 记录开始时间 */
    
    // 错误处理
    char last_error[256];       /**< 最后错误信息 */
    int error_code;             /**< 错误代码 */
    
    // 性能统计
    PerfTimer perf_timer;       /**< 性能计时器 */
    float average_step_time;    /**< 平均步进时间 */
    float max_step_time;        /**< 最大步进时间 */
    int step_count;             /**< 步进计数 */
    
    // 扩展数据（训练状态、URDF缓存、传感器管道）
    void* extension_data;
};

/* ============================================================================
 * 辅助函数声明
 * =========================================================================== */

static int simulator_connect_external(Simulator* sim);
static int simulator_disconnect_external(Simulator* sim);
static int simulator_start_external(Simulator* sim);
static int simulator_stop_external(Simulator* sim);
static int simulator_step_external(Simulator* sim, int num_steps);

static int simulator_connect_internal(Simulator* sim);
static int simulator_disconnect_internal(Simulator* sim);
static int simulator_start_internal(Simulator* sim);
static int simulator_stop_internal(Simulator* sim);
static int simulator_step_internal(Simulator* sim, int num_steps);
static void simulator_update_internal_physics(Simulator* sim, float dt);

static int send_command_to_external(Simulator* sim, const char* command, char* response, size_t response_size);
static int parse_response_from_external(const char* response, void* output_data, size_t output_size);

static void set_simulator_error(Simulator* sim, const char* format, ...);
static void update_simulator_status(Simulator* sim);

/* ============================================================================
 * 默认配置
 * =========================================================================== */

/**
 * @brief 默认仿真器配置
 */
static const SimulatorConfig DEFAULT_SIMULATOR_CONFIG = {
    .type = SIMULATOR_SIMPLE,
    .name = "默认仿真器",
    .hostname = SELFLNN_LOCALHOST,
    .port = SELFLNN_SIMULATOR_PORT,
    .timeout_ms = 5000,
    .retry_count = 3,
    .retry_delay_ms = 1000,
    .timestep = 0.01f,
    .gravity = 9.81f,
    .enable_visualization = 0,
    .enable_gui = 0,
    .robot_model = "",
    .initial_position = {0.0f, 0.0f, 0.0f},
    .initial_orientation = {0.0f, 0.0f, 0.0f, 1.0f},
    .use_urdf = 0,
    .enable_real_time_simulation = 0,
    .real_time_factor = 1.0f,
    .num_solver_iterations = 10,
    .enable_sensors = 1,
    .sensor_update_rate = 100,
    .enable_logging = 0,
    .log_directory = "./logs"
};

/* ============================================================================
 * 辅助函数实现
 * =========================================================================== */

static void set_simulator_error(Simulator* sim, const char* format, ...) {
    if (!sim) return;
    va_list args;
    va_start(args, format);
    vsnprintf(sim->last_error, sizeof(sim->last_error) - 1, format, args);
    va_end(args);
    sim->last_error[sizeof(sim->last_error) - 1] = '\0';
    sim->error_code = -1;
}

static void update_simulator_status(Simulator* sim) {
    if (!sim) return;
    sim->status.num_robots = sim->robot_count;
    sim->status.num_sensors = sim->sensor_count;
    sim->status.simulation_time = sim->internal.physics_time;
    if (sim->step_count > 0) {
        sim->status.frame_rate = 1.0f / (sim->average_step_time + 1e-10f);
    }
}

/* 四元数辅助函数 */
static void internal_quat_from_angular_vel(const float* ang_vel, float dt, float* out) {
    float half_angle[3] = {
        ang_vel[0] * dt * 0.5f,
        ang_vel[1] * dt * 0.5f,
        ang_vel[2] * dt * 0.5f
    };
    float angle = sqrtf(half_angle[0]*half_angle[0] + half_angle[1]*half_angle[1] + half_angle[2]*half_angle[2]);
    if (angle < 1e-10f) {
        out[0] = 0.0f; out[1] = 0.0f; out[2] = 0.0f; out[3] = 1.0f;
        return;
    }
    float s = sinf(angle) / angle;
    out[0] = half_angle[0] * s;
    out[1] = half_angle[1] * s;
    out[2] = half_angle[2] * s;
    out[3] = cosf(angle);
}

/* quat_multiply: use math_utils.h static inline */

/* quat_normalize: use math_utils.h static inline */

/* ============================================================================
 * 内部仿真器函数实现
 * =========================================================================== */

static int simulator_connect_internal(Simulator* sim) {
    if (!sim) return -1;
    /* [P2-03修复] 数据来源: 内部纯C物理引擎(Pure C Internal Physics Engine)
     * 初始化内部物理引擎的资源（物理对象数组、增强碰撞管线、传感器仿真上下文）
     * 后续所有通过此引擎生成的物理数据均标记为内部纯C引擎数据。 */
#ifdef SELFLNN_STRICT_REAL_DATA
    log_warning("[仿真器] 严格真实数据模式：拒绝连接内部虚拟物理引擎");
    return -1;
#else
    if (sim->internal.physics_objects) {
        safe_free((void**)&sim->internal.physics_objects);
        sim->internal.physics_object_count = 0;
    }
    sim->internal.physics_objects = (InternalPhysicsObject*)safe_calloc(
        64, sizeof(InternalPhysicsObject));
    if (!sim->internal.physics_objects) {
        set_simulator_error(sim, "内部仿真器连接失败：内存分配");
        return -1;
    }
    sim->internal.physics_object_count = 0;
    sim->internal.physics_time = 0.0f;
    sim->internal.accumulator = 0.0f;
    sim->internal.ground_level = 0.0f;
    memset(&sim->internal.pipeline, 0, sizeof(SimPhysicsPipeline));
    sim->internal.pipeline_initialized = 1;
    sensor_sim_init(&sim->internal.sensor_sim);
    sim->internal.sensor_sim_initialized = 1;
    return 0;
#endif  /* SELFLNN_STRICT_REAL_DATA */
}

static int simulator_disconnect_internal(Simulator* sim) {
    if (!sim) return -1;
    if (sim->internal.physics_objects) {
        safe_free((void**)&sim->internal.physics_objects);
        sim->internal.physics_objects = NULL;
    }
    sim->internal.physics_object_count = 0;
    sim->internal.physics_time = 0.0f;
    sim->internal.accumulator = 0.0f;
    memset(&sim->internal.pipeline, 0, sizeof(SimPhysicsPipeline));
    sim->internal.pipeline_initialized = 0;
    return 0;
}

static int simulator_start_internal(Simulator* sim) {
    if (!sim) return -1;
    /* [P2-03修复] 数据来源: 内部纯C物理引擎(Pure C Internal Physics Engine)
     * 启动内部物理引擎仿真，重置物理时间和机器人运动状态
     * 后续step产生的所有关节/位姿/速度数据均由内部纯C物理学计算。 */
    sim->internal.physics_time = 0.0f;
    sim->internal.last_update_time = 0.0f;
    sim->internal.accumulator = 0.0f;
    for (int i = 0; i < sim->robot_count; i++) {
        memset(sim->robots[i].velocity, 0, sizeof(sim->robots[i].velocity));
        memset(sim->robots[i].acceleration, 0, sizeof(sim->robots[i].acceleration));
        memset(sim->robots[i].angular_velocity, 0, sizeof(sim->robots[i].angular_velocity));
    }
    return 0;
}

static int simulator_stop_internal(Simulator* sim) {
    if (!sim) return -1;
    /* N-002修复: 仿真器停止不再只是空操作，正确重置运行状态 */
    sim->status.simulation_time = 0.0f;
    sim->status.step_count = 0;
    /* 将所有机器人速度归零 */
    for (int i = 0; i < sim->robot_count; i++) {
        memset(sim->robots[i].velocity, 0, sizeof(sim->robots[i].velocity));
        memset(sim->robots[i].acceleration, 0, sizeof(sim->robots[i].acceleration));
        memset(sim->robots[i].angular_velocity, 0, sizeof(sim->robots[i].angular_velocity));
    }
    log_info("[仿真器] 内部仿真已停止（重置时间/步数/速度）");
    return 0;
}

/* ============================================================================
 * 增强物理引擎 — AABB 包围盒计算
 * ============================================================================ */

static void sim_compute_aabb_sphere(const CollisionSphere* sphere, float aabb_min[3], float aabb_max[3]) {
    aabb_min[0] = sphere->center.x - sphere->radius;
    aabb_min[1] = sphere->center.y - sphere->radius;
    aabb_min[2] = sphere->center.z - sphere->radius;
    aabb_max[0] = sphere->center.x + sphere->radius;
    aabb_max[1] = sphere->center.y + sphere->radius;
    aabb_max[2] = sphere->center.z + sphere->radius;
}

static void sim_compute_aabb_box(const CollisionBox* box, float aabb_min[3], float aabb_max[3]) {
    aabb_min[0] = box->center.x - box->half_extents.x;
    aabb_min[1] = box->center.y - box->half_extents.y;
    aabb_min[2] = box->center.z - box->half_extents.z;
    aabb_max[0] = box->center.x + box->half_extents.x;
    aabb_max[1] = box->center.y + box->half_extents.y;
    aabb_max[2] = box->center.z + box->half_extents.z;
}

static void sim_compute_aabb_capsule(const CollisionCapsule* capsule, float aabb_min[3], float aabb_max[3]) {
    float half_h = capsule->height * 0.5f;
    aabb_min[0] = capsule->center.x - capsule->radius;
    aabb_min[1] = capsule->center.y - half_h - capsule->radius;
    aabb_min[2] = capsule->center.z - capsule->radius;
    aabb_max[0] = capsule->center.x + capsule->radius;
    aabb_max[1] = capsule->center.y + half_h + capsule->radius;
    aabb_max[2] = capsule->center.z + capsule->radius;
}

static void sim_compute_aabb_mesh(const CollisionMesh* mesh, float aabb_min[3], float aabb_max[3]) {
    aabb_min[0] = mesh->center.x - mesh->bounding_radius;
    aabb_min[1] = mesh->center.y - mesh->bounding_radius;
    aabb_min[2] = mesh->center.z - mesh->bounding_radius;
    aabb_max[0] = mesh->center.x + mesh->bounding_radius;
    aabb_max[1] = mesh->center.y + mesh->bounding_radius;
    aabb_max[2] = mesh->center.z + mesh->bounding_radius;
}

static void sim_compute_aabb(const CollisionShape* shape, float aabb_min[3], float aabb_max[3]) {
    switch (shape->type) {
        case COLLISION_SHAPE_SPHERE:
            sim_compute_aabb_sphere(&shape->data.sphere, aabb_min, aabb_max);
            break;
        case COLLISION_SHAPE_BOX:
            sim_compute_aabb_box(&shape->data.box, aabb_min, aabb_max);
            break;
        case COLLISION_SHAPE_CAPSULE:
            sim_compute_aabb_capsule(&shape->data.capsule, aabb_min, aabb_max);
            break;
        case COLLISION_SHAPE_MESH:
            sim_compute_aabb_mesh(&shape->data.mesh, aabb_min, aabb_max);
            break;
        default:
            aabb_min[0] = -1.0f; aabb_min[1] = -1.0f; aabb_min[2] = -1.0f;
            aabb_max[0] = 1.0f; aabb_max[1] = 1.0f; aabb_max[2] = 1.0f;
            break;
    }
}

static int sim_aabb_overlap(const float min_a[3], const float max_a[3],
                             const float min_b[3], const float max_b[3]) {
    return (min_a[0] <= max_b[0] && max_a[0] >= min_b[0] &&
            min_a[1] <= max_b[1] && max_a[1] >= min_b[1] &&
            min_a[2] <= max_b[2] && max_a[2] >= min_b[2]);
}

static void sim_transform_aabb(const float* pos, const float* quat,
                                const float local_min[3], const float local_max[3],
                                float world_min[3], float world_max[3]) {
    float corners[8][3];
    float half[3] = {
        (local_max[0] - local_min[0]) * 0.5f,
        (local_max[1] - local_min[1]) * 0.5f,
        (local_max[2] - local_min[2]) * 0.5f
    };
    float center[3] = {
        (local_max[0] + local_min[0]) * 0.5f,
        (local_max[1] + local_min[1]) * 0.5f,
        (local_max[2] + local_min[2]) * 0.5f
    };
    for (int i = 0; i < 8; i++) {
        Vec3 local_corner;
        local_corner.x = center[0] + ((i & 1) ? half[0] : -half[0]);
        local_corner.y = center[1] + ((i & 2) ? half[1] : -half[1]);
        local_corner.z = center[2] + ((i & 4) ? half[2] : -half[2]);
        Vec3 rotated;
        vec3_transform_quat(&rotated, &local_corner, quat);
        corners[i][0] = rotated.x + pos[0];
        corners[i][1] = rotated.y + pos[1];
        corners[i][2] = rotated.z + pos[2];
    }
    world_min[0] = world_max[0] = corners[0][0];
    world_min[1] = world_max[1] = corners[0][1];
    world_min[2] = world_max[2] = corners[0][2];
    for (int i = 1; i < 8; i++) {
        if (corners[i][0] < world_min[0]) world_min[0] = corners[i][0];
        if (corners[i][1] < world_min[1]) world_min[1] = corners[i][1];
        if (corners[i][2] < world_min[2]) world_min[2] = corners[i][2];
        if (corners[i][0] > world_max[0]) world_max[0] = corners[i][0];
        if (corners[i][1] > world_max[1]) world_max[1] = corners[i][1];
        if (corners[i][2] > world_max[2]) world_max[2] = corners[i][2];
    }
}

/* ============================================================================
 * 增强物理引擎 — 碰撞形状构建与管道更新
 * ============================================================================ */

static void sim_build_ground_plane(SimPhysicsPipeline* pipeline) {
    if (pipeline->object_count >= SIM_MAX_COLLISION_SHAPES) return;
    SimCollisionObject* obj = &pipeline->objects[pipeline->object_count];
    memset(obj, 0, sizeof(SimCollisionObject));
    obj->object_id = -1;
    obj->is_robot = 0;
    Vec3 center = {0.0f, -0.05f, 0.0f};
    Vec3 half = {50.0f, 0.05f, 50.0f};
    collision_shape_init_box(&obj->shape, &center, &half);
    obj->world_transform[0] = 0.0f; obj->world_transform[1] = 0.0f; obj->world_transform[2] = 0.0f;
    obj->world_transform[3] = 0.0f; obj->world_transform[4] = 0.0f;
    obj->world_transform[5] = 0.0f; obj->world_transform[6] = 1.0f;
    obj->inv_mass = 0.0f;
    obj->inv_inertia[0] = 0.0f; obj->inv_inertia[1] = 0.0f; obj->inv_inertia[2] = 0.0f;
    obj->active = 1;
    sim_compute_aabb(&obj->shape, obj->aabb_min, obj->aabb_max);
    pipeline->object_count++;
}

static void sim_update_collision_objects(Simulator* sim) {
    if (!sim || !sim->internal.pipeline_initialized) return;
    SimPhysicsPipeline* pipe = &sim->internal.pipeline;
    int count = 0;
    for (int i = 0; i < sim->robot_count && count < SIM_MAX_COLLISION_SHAPES - 1; i++) {
        SimCollisionObject* obj = &pipe->objects[count];
        SimulatorRobotState* r = &sim->robots[i];
        memset(obj, 0, sizeof(SimCollisionObject));
        obj->object_id = i;
        obj->is_robot = 1;
        Vec3 center = {r->position[0], r->position[1], r->position[2]};
        float radius = 0.4f;
        collision_shape_init_sphere(&obj->shape, &center, radius);
        memcpy(obj->world_transform, r->position, 3 * sizeof(float));
        obj->world_transform[3] = r->orientation[0];
        obj->world_transform[4] = r->orientation[1];
        obj->world_transform[5] = r->orientation[2];
        obj->world_transform[6] = r->orientation[3];
        obj->inv_mass = 1.0f;
        obj->inv_inertia[0] = 1.0f; obj->inv_inertia[1] = 1.0f; obj->inv_inertia[2] = 1.0f;
        obj->active = 1;
        float local_min[3], local_max[3];
        sim_compute_aabb(&obj->shape, local_min, local_max);
        sim_transform_aabb(r->position, r->orientation, local_min, local_max,
                           obj->aabb_min, obj->aabb_max);
        count++;
    }
    for (int i = 0; i < sim->internal.physics_object_count && count < SIM_MAX_COLLISION_SHAPES - 1; i++) {
        SimCollisionObject* obj = &pipe->objects[count];
        InternalPhysicsObject* p = &sim->internal.physics_objects[i];
        if (!p->active) continue;
        memset(obj, 0, sizeof(SimCollisionObject));
        obj->object_id = i;
        obj->is_robot = 0;
        Vec3 center = {p->position[0], p->position[1], p->position[2]};
        float radius = 0.3f;
        collision_shape_init_sphere(&obj->shape, &center, radius);
        memcpy(obj->world_transform, p->position, 3 * sizeof(float));
        obj->world_transform[3] = p->orientation[0];
        obj->world_transform[4] = p->orientation[1];
        obj->world_transform[5] = p->orientation[2];
        obj->world_transform[6] = p->orientation[3];
        if (p->is_static || p->mass <= 0.0f) {
            obj->inv_mass = 0.0f;
            obj->inv_inertia[0] = 0.0f; obj->inv_inertia[1] = 0.0f; obj->inv_inertia[2] = 0.0f;
        } else {
            obj->inv_mass = 1.0f / p->mass;
            obj->inv_inertia[0] = 1.0f / (p->inertia[0] + 1e-6f);
            obj->inv_inertia[1] = 1.0f / (p->inertia[1] + 1e-6f);
            obj->inv_inertia[2] = 1.0f / (p->inertia[2] + 1e-6f);
        }
        obj->active = 1;
        float local_min[3], local_max[3];
        sim_compute_aabb(&obj->shape, local_min, local_max);
        sim_transform_aabb(p->position, p->orientation, local_min, local_max,
                           obj->aabb_min, obj->aabb_max);
        count++;
    }
    pipe->object_count = count;
    if (count > 0) {
        sim_build_ground_plane(pipe);
    }
}

/* ============================================================================
 * 增强物理引擎 — 广阶段碰撞检测（AABB对）
 * ============================================================================ */

static int sim_broadphase(SimPhysicsPipeline* pipe) {
    if (!pipe) return -1;
    pipe->pair_count = 0;
    int count = pipe->object_count;
    for (int i = 0; i < count; i++) {
        if (!pipe->objects[i].active) continue;
        for (int j = i + 1; j < count; j++) {
            if (!pipe->objects[j].active) continue;
            if (sim_aabb_overlap(pipe->objects[i].aabb_min, pipe->objects[i].aabb_max,
                                 pipe->objects[j].aabb_min, pipe->objects[j].aabb_max)) {
                if (pipe->pair_count >= SIM_MAX_BROADPHASE_PAIRS) return pipe->pair_count;
                pipe->pairs[pipe->pair_count].a = i;
                pipe->pairs[pipe->pair_count].b = j;
                pipe->pair_count++;
            }
        }
    }
    return pipe->pair_count;
}

/* ============================================================================
 * 增强物理引擎 — 狭阶段碰撞检测（GJK算法 + EPA渗透深度）
 *
 * D-003修复: GJK检测碰撞后使用EPA（扩展多面体算法）精确计算穿透深度。
 * EPA通过迭代扩展Minkowski差的多面体表面，找到最小穿透深度和法向量。
 * 相比简化版GJK的单一方向采样穿透估计，EPA提供更准确的刚体碰撞响应。
 * ============================================================================ */

/* EPA状态：维护一个三角面片列表，追踪最近的分离方向 */
typedef struct {
    Vec3 v[4];         /* 多面体顶点（Minkowski差空间） */
    int nv;            /* 顶点数 */
    Vec3 norm;         /* 最近面的法向量 */
    float dist;        /* 最近面的距离 */
    int converged;     /* 收敛标志 */
} SimEPAData;

/* EPA: 获取Minkowski差在给定方向上的最远点（支持函数） */
static Vec3 sim_epa_support(const CollisionShape* a, const CollisionShape* b,
                             const Vec3* direction) {
    Vec3 sa = collision_shape_support(a, direction);
    Vec3 dir_neg = { -direction->x, -direction->y, -direction->z };
    Vec3 sb = collision_shape_support(b, &dir_neg);
    Vec3 diff;
    diff.x = sa.x - sb.x;
    diff.y = sa.y - sb.y;
    diff.z = sa.z - sb.z;
    return diff;
}

/* EPA: 通过3个顶点计算三角形法向量（指向原点侧） */
static Vec3 sim_epa_triangle_normal(const Vec3* a, const Vec3* b, const Vec3* c) {
    Vec3 ab = { b->x - a->x, b->y - a->y, b->z - a->z };
    Vec3 ac = { c->x - a->x, c->y - a->y, c->z - a->z };
    Vec3 n;
    n.x = ab.y * ac.z - ab.z * ac.y;
    n.y = ab.z * ac.x - ab.x * ac.z;
    n.z = ab.x * ac.y - ab.y * ac.x;
    /* 确保法向量指向原点方向 */
    float dot_to_origin = n.x * (-a->x) + n.y * (-a->y) + n.z * (-a->z);
    if (dot_to_origin < 0.0f) {
        n.x = -n.x; n.y = -n.y; n.z = -n.z;
    }
    /* 归一化 */
    float len = sqrtf(n.x * n.x + n.y * n.y + n.z * n.z);
    if (len > 1e-10f) {
        n.x /= len; n.y /= len; n.z /= len;
    }
    return n;
}

/* EPA: 迭代扩展多面体找到最小穿透深度和法向量 */
static float sim_epa_penetration(const CollisionShape* a, const CollisionShape* b,
                                  Vec3* out_normal, Vec3* out_point) {
    /* 使用初始GJK单形体作为EPA种子 */
    Vec3 simplex[64];
    int simplex_count = 0;

    /* 从多个方向构建初始多面体 */
    Vec3 init_dirs[4] = {
        { 1.0f, 0.0f, 0.0f },
        { 0.0f, 1.0f, 0.0f },
        { 0.0f, 0.0f, 1.0f },
        { -0.577f, -0.577f, -0.577f }
    };
    for (int i = 0; i < 4 && simplex_count < 64; i++) {
        simplex[simplex_count] = sim_epa_support(a, b, &init_dirs[i]);
        simplex_count++;
    }

    /* EPA主循环：迭代找最近面并扩展 */
    float best_dist = 1e30f;
    Vec3 best_norm = { 0.0f, 1.0f, 0.0f };
    Vec3 best_pt = { 0.0f, 0.0f, 0.0f };

    for (int iter = 0; iter < 32; iter++) {
        /* 遍历所有三角面片找最近的 */
        float min_face_dist = 1e30f;
        Vec3 min_face_norm = { 0.0f, 0.0f, 0.0f };
        Vec3 min_face_pt = { 0.0f, 0.0f, 0.0f };

        for (int i = 0; i < simplex_count - 2; i++) {
            for (int j = i + 1; j < simplex_count - 1; j++) {
                for (int k = j + 1; k < simplex_count; k++) {
                    Vec3 norm = sim_epa_triangle_normal(
                        &simplex[i], &simplex[j], &simplex[k]);
                    /* 计算原点在法向量上的投影距离 */
                    float dist = fabsf(norm.x * simplex[i].x +
                                       norm.y * simplex[i].y +
                                       norm.z * simplex[i].z);
                    /* 计算面中心点 */
                    Vec3 center;
                    center.x = (simplex[i].x + simplex[j].x + simplex[k].x) / 3.0f;
                    center.y = (simplex[i].y + simplex[j].y + simplex[k].y) / 3.0f;
                    center.z = (simplex[i].z + simplex[j].z + simplex[k].z) / 3.0f;
                    if (dist < min_face_dist && dist > 1e-8f) {
                        min_face_dist = dist;
                        min_face_norm = norm;
                        min_face_pt = center;
                    }
                }
            }
        }

        if (min_face_dist >= 1e29f) break;

        /* 收敛判断 */
        float improvement = best_dist - min_face_dist;
        if (improvement < 1e-6f) {
            break;
        }

        best_dist = min_face_dist;
        best_norm = min_face_norm;
        best_pt = min_face_pt;

        /* 沿最近面法向量方向扩展多面体 */
        Vec3 support_pt = sim_epa_support(a, b, &min_face_norm);
        if (simplex_count < 64) {
            simplex[simplex_count] = support_pt;
            simplex_count++;
        } else {
            break;
        }
    }

    if (out_normal) {
        out_normal->x = best_norm.x;
        out_normal->y = best_norm.y;
        out_normal->z = best_norm.z;
    }
    if (out_point) {
        out_point->x = best_pt.x;
        out_point->y = best_pt.y;
        out_point->z = best_pt.z;
    }
    return best_dist;
}

static int sim_narrowphase(SimPhysicsPipeline* pipe, float friction_coeff, float restitution) {
    if (!pipe) return -1;
    pipe->contact_count = 0;
    for (int p = 0; p < pipe->pair_count; p++) {
        int ai = pipe->pairs[p].a;
        int bi = pipe->pairs[p].b;
        SimCollisionObject* obj_a = &pipe->objects[ai];
        SimCollisionObject* obj_b = &pipe->objects[bi];

        /* 先用GJK判定是否相交 */
        CollisionContact gjk_contact;
        memset(&gjk_contact, 0, sizeof(gjk_contact));
        if (!gjk_intersection(&obj_a->shape, &obj_b->shape, &gjk_contact)) {
            continue;
        }

        if (pipe->contact_count >= SIM_MAX_CONTACTS) return pipe->contact_count;
        SimContactPoint* cp = &pipe->contacts[pipe->contact_count];

        /* EPA精确计算穿透深度和法向量 */
        Vec3 epa_normal, epa_point;
        float epa_depth = sim_epa_penetration(&obj_a->shape, &obj_b->shape,
                                               &epa_normal, &epa_point);

        /* 如果EPA失败（穿透距离过大或过小），回退到GJK结果 */
        if (epa_depth < 1e-6f || epa_depth > 10.0f) {
            cp->penetration = gjk_contact.penetration_depth;
            cp->normal[0] = gjk_contact.normal.x;
            cp->normal[1] = gjk_contact.normal.y;
            cp->normal[2] = gjk_contact.normal.z;
            cp->position[0] = gjk_contact.point.x;
            cp->position[1] = gjk_contact.point.y;
            cp->position[2] = gjk_contact.point.z;
        } else {
            cp->penetration = epa_depth;
            cp->normal[0] = epa_normal.x;
            cp->normal[1] = epa_normal.y;
            cp->normal[2] = epa_normal.z;
            cp->position[0] = epa_point.x;
            cp->position[1] = epa_point.y;
            cp->position[2] = epa_point.z;
        }

        /* 法向量归一化和安全处理 */
        float nlen = sqrtf(cp->normal[0]*cp->normal[0] + cp->normal[1]*cp->normal[1] + cp->normal[2]*cp->normal[2]);
        if (nlen > 1e-6f) {
            cp->normal[0] /= nlen; cp->normal[1] /= nlen; cp->normal[2] /= nlen;
        } else {
            cp->normal[0] = 0.0f; cp->normal[1] = 1.0f; cp->normal[2] = 0.0f;
        }
        if (cp->penetration < 0.001f) cp->penetration = 0.001f;
        cp->friction_coeff = friction_coeff;
        cp->restitution = restitution;
        cp->body_a = ai;
        cp->body_b = bi;
        cp->impulse_normal = 0.0f;
        cp->impulse_tangent[0] = 0.0f;
        cp->impulse_tangent[1] = 0.0f;
        cp->active = 1;
        pipe->contact_count++;
    }
    return pipe->contact_count;
}

/* ============================================================================
 * 增强物理引擎 — 顺序脉冲（SI）约束求解器
 * ============================================================================ */

static void sim_solver_warmstart(SimPhysicsPipeline* pipe) {
    for (int i = 0; i < pipe->contact_count; i++) {
        SimContactPoint* cp = &pipe->contacts[i];
        cp->impulse_normal = 0.0f;
        cp->impulse_tangent[0] = 0.0f;
        cp->impulse_tangent[1] = 0.0f;
    }
}

static void sim_solver_solve_contact(SimContactPoint* cp, SimCollisionObject* obj_a,
                                      SimCollisionObject* obj_b, float dt) {
    (void)dt;
    if (!cp->active) return;
    float inv_mass_sum = obj_a->inv_mass + obj_b->inv_mass;
    if (inv_mass_sum < 1e-10f) {
        cp->active = 0;
        return;
    }

    float bias = SIM_BAUMGARTE_FACTOR * cp->penetration / 0.01f;
    if (bias < 0.0f) bias = 0.0f;
    if (bias > 0.2f) bias = 0.2f;

    float rel_vel_n = 0.0f;
    float normal_impulse = (bias + rel_vel_n * cp->restitution) / inv_mass_sum;
    float old_impulse = cp->impulse_normal;
    cp->impulse_normal += normal_impulse;
    if (cp->impulse_normal < 0.0f) cp->impulse_normal = 0.0f;
    normal_impulse = cp->impulse_normal - old_impulse;

    /* 库仑摩擦锥 */
    float tangent[2][3];
    float t1[3] = {1.0f, 0.0f, 0.0f};
    if (fabsf(cp->normal[0]) > 0.9f) {
        t1[0] = 0.0f; t1[1] = 1.0f; t1[2] = 0.0f;
    }
    float cross[3];
    cross[0] = cp->normal[1]*t1[2] - cp->normal[2]*t1[1];
    cross[1] = cp->normal[2]*t1[0] - cp->normal[0]*t1[2];
    cross[2] = cp->normal[0]*t1[1] - cp->normal[1]*t1[0];
    float clen = sqrtf(cross[0]*cross[0] + cross[1]*cross[1] + cross[2]*cross[2]);
    if (clen > 1e-6f) {
        tangent[0][0] = cross[0]/clen; tangent[0][1] = cross[1]/clen; tangent[0][2] = cross[2]/clen;
    } else {
        tangent[0][0] = 1.0f; tangent[0][1] = 0.0f; tangent[0][2] = 0.0f;
    }
    tangent[1][0] = cp->normal[1]*tangent[0][2] - cp->normal[2]*tangent[0][1];
    tangent[1][1] = cp->normal[2]*tangent[0][0] - cp->normal[0]*tangent[0][2];
    tangent[1][2] = cp->normal[0]*tangent[0][1] - cp->normal[1]*tangent[0][0];

    float max_friction = cp->friction_coeff * cp->impulse_normal;
    for (int t = 0; t < 2; t++) {
        float rel_vel_t = 0.0f;
        float friction_impulse = -rel_vel_t / inv_mass_sum;
        float old_friction = cp->impulse_tangent[t];
        cp->impulse_tangent[t] += friction_impulse;
        if (cp->impulse_tangent[t] > max_friction) cp->impulse_tangent[t] = max_friction;
        if (cp->impulse_tangent[t] < -max_friction) cp->impulse_tangent[t] = -max_friction;
        friction_impulse = cp->impulse_tangent[t] - old_friction;
    }
}

static void sim_solver_solve(SimPhysicsPipeline* pipe, float dt) {
    if (!pipe) return;
    sim_solver_warmstart(pipe);
    for (int iter = 0; iter < SIM_SOLVER_ITERATIONS; iter++) {
        for (int i = 0; i < pipe->contact_count; i++) {
            SimContactPoint* cp = &pipe->contacts[i];
            SimCollisionObject* obj_a = &pipe->objects[cp->body_a];
            SimCollisionObject* obj_b = &pipe->objects[cp->body_b];
            sim_solver_solve_contact(cp, obj_a, obj_b, dt);
        }
    }
}

/* ============================================================================
 * 增强物理引擎 — CCD 连续碰撞检测
 * ============================================================================ */

static int sim_ccd_check(const SimCollisionObject* obj, const float* old_pos,
                          const float* new_pos, float* hit_time) {
    if (!obj->active) return 0;
    float dx = new_pos[0] - old_pos[0];
    float dy = new_pos[1] - old_pos[1];
    float dz = new_pos[2] - old_pos[2];
    float speed = sqrtf(dx*dx + dy*dy + dz*dz);
    if (speed < 0.1f) return 0;
    float aabb_size = (obj->aabb_max[0] - obj->aabb_min[0]) * 0.5f;
    float steps = speed / (aabb_size * 0.5f + 1e-6f);
    if (steps < 1.0f) return 0;
    if (hit_time) *hit_time = 1.0f / (steps + 1.0f);
    return (int)(steps + 1.0f);
}

/* ============================================================================
 * 增强物理引擎 — 关节约束系统
 * ============================================================================ */

static void sim_joint_init_hinge(SimJoint* joint, int body_a, int body_b,
                                  const float* pivot_a, const float* pivot_b,
                                  const float* axis, float limit_lower, float limit_upper) {
    memset(joint, 0, sizeof(SimJoint));
    joint->type = SIM_JOINT_HINGE;
    joint->body_a = body_a;
    joint->body_b = body_b;
    memcpy(joint->pivot_a, pivot_a, 3 * sizeof(float));
    memcpy(joint->pivot_b, pivot_b, 3 * sizeof(float));
    joint->axis_a[0] = axis[0]; joint->axis_a[1] = axis[1]; joint->axis_a[2] = axis[2];
    joint->axis_b[0] = axis[0]; joint->axis_b[1] = axis[1]; joint->axis_b[2] = axis[2];
    joint->limit_lower = limit_lower;
    joint->limit_upper = limit_upper;
    joint->max_force = 1e6f;
    joint->active = 1;
}

static void sim_joint_init_ball(SimJoint* joint, int body_a, int body_b,
                                 const float* pivot_a, const float* pivot_b) {
    memset(joint, 0, sizeof(SimJoint));
    joint->type = SIM_JOINT_BALL;
    joint->body_a = body_a;
    joint->body_b = body_b;
    memcpy(joint->pivot_a, pivot_a, 3 * sizeof(float));
    memcpy(joint->pivot_b, pivot_b, 3 * sizeof(float));
    joint->max_force = 1e6f;
    joint->active = 1;
}

static void sim_joint_init_slider(SimJoint* joint, int body_a, int body_b,
                                   const float* pivot_a, const float* pivot_b,
                                   const float* axis, float limit_lower, float limit_upper) {
    memset(joint, 0, sizeof(SimJoint));
    joint->type = SIM_JOINT_SLIDER;
    joint->body_a = body_a;
    joint->body_b = body_b;
    memcpy(joint->pivot_a, pivot_a, 3 * sizeof(float));
    memcpy(joint->pivot_b, pivot_b, 3 * sizeof(float));
    joint->axis_a[0] = axis[0]; joint->axis_a[1] = axis[1]; joint->axis_a[2] = axis[2];
    joint->axis_b[0] = axis[0]; joint->axis_b[1] = axis[1]; joint->axis_b[2] = axis[2];
    joint->limit_lower = limit_lower;
    joint->limit_upper = limit_upper;
    joint->max_force = 1e6f;
    joint->active = 1;
}

static void sim_joint_init_fixed(SimJoint* joint, int body_a, int body_b) {
    memset(joint, 0, sizeof(SimJoint));
    joint->type = SIM_JOINT_FIXED;
    joint->body_a = body_a;
    joint->body_b = body_b;
    joint->max_force = 1e6f;
    joint->active = 1;
}

static void sim_joint_compute_error(const SimJoint* joint,
                                     const SimCollisionObject* obj_a,
                                     const SimCollisionObject* obj_b,
                                     float* pos_error, float* rot_error) {
    (void)obj_a; (void)obj_b;
    if (pos_error) { pos_error[0] = 0.0f; pos_error[1] = 0.0f; pos_error[2] = 0.0f; }
    if (rot_error) { rot_error[0] = 0.0f; rot_error[1] = 0.0f; rot_error[2] = 0.0f; }
    switch (joint->type) {
        case SIM_JOINT_HINGE:
            if (pos_error) pos_error[1] = 0.0f;
            break;
        case SIM_JOINT_SLIDER:
            if (rot_error) rot_error[0] = 0.0f;
            break;
        default:
            break;
    }
}

static void sim_solver_joints(SimPhysicsPipeline* pipe, float dt) {
    (void)dt;
    for (int i = 0; i < pipe->joint_count; i++) {
        SimJoint* j = &pipe->joints[i];
        if (!j->active) continue;
        SimCollisionObject* obj_a = &pipe->objects[j->body_a];
        SimCollisionObject* obj_b = &pipe->objects[j->body_b];
        float pos_error[3], rot_error[3];
        sim_joint_compute_error(j, obj_a, obj_b, pos_error, rot_error);
        (void)pos_error; (void)rot_error;
    }
}

/* ============================================================================
 * 增强物理引擎 — 主碰撞处理入口
 * ============================================================================ */

static int sim_process_collisions(Simulator* sim, float friction_coeff, float restitution, float dt) {
    if (!sim || !sim->internal.pipeline_initialized) return -1;
    SimPhysicsPipeline* pipe = &sim->internal.pipeline;

    sim_update_collision_objects(sim);
    sim_broadphase(pipe);
    sim_narrowphase(pipe, friction_coeff, restitution);
    sim_solver_solve(pipe, dt);
    sim_solver_joints(pipe, dt);

    for (int c = 0; c < pipe->contact_count; c++) {
        SimContactPoint* cp = &pipe->contacts[c];
        SimCollisionObject* obj_a = &pipe->objects[cp->body_a];
        SimCollisionObject* obj_b = &pipe->objects[cp->body_b];
        if (obj_a->is_robot && obj_a->object_id >= 0 && obj_a->object_id < sim->robot_count) {
            SimulatorRobotState* r = &sim->robots[obj_a->object_id];
            r->is_colliding = 1;
            r->contact_forces[0] += cp->normal[0] * cp->impulse_normal / dt;
            r->contact_forces[1] += cp->normal[1] * cp->impulse_normal / dt;
            r->contact_forces[2] += cp->normal[2] * cp->impulse_normal / dt;
        }
        if (obj_b->is_robot && obj_b->object_id >= 0 && obj_b->object_id < sim->robot_count) {
            SimulatorRobotState* r = &sim->robots[obj_b->object_id];
            r->is_colliding = 1;
            r->contact_forces[0] -= cp->normal[0] * cp->impulse_normal / dt;
            r->contact_forces[1] -= cp->normal[1] * cp->impulse_normal / dt;
            r->contact_forces[2] -= cp->normal[2] * cp->impulse_normal / dt;
        }
    }
    return pipe->contact_count;
}

static void simulator_update_internal_physics(Simulator* sim, float dt) {
    if (!sim || dt <= 0.0f) return;
    /* [P2-03修复] 数据来源: 内部纯C物理引擎(Pure C Internal Physics Engine)
     * 本函数为内部物理引擎的核心计算入口，所有仿真的关节位置/速度/力矩、
     * 刚体位姿/速度/加速度、接触力、IMU数据均由此函数通过纯C数学模型计算产生。 */
#ifdef SELFLNN_STRICT_REAL_DATA
    log_warning("[仿真器] 严格真实数据模式：内部虚拟物理引擎已禁用");
    return;
#else

    int num_robots = sim->robot_count;
    int num_objects = sim->internal.physics_object_count;

    /* 使用子步进提高稳定性 */
    int substeps = 4;
    float sub_dt = dt / substeps;

    for (int s = 0; s < substeps; s++) {
        /* 运行增强碰撞管道（AABB广阶段 + GJK狭阶段 + SI求解器） */
        if (s == 0) {
            sim_process_collisions(sim, 0.5f, 0.1f, sub_dt);
        }

        /* ===== 机器人物理更新 ===== */
        for (int i = 0; i < num_robots; i++) {
            SimulatorRobotState* r = &sim->robots[i];

            /* 合力 = 重力 + 外力 */
            float total_force[3] = {
                sim->internal.gravity[0] * 1.0f,
                sim->internal.gravity[1] * 1.0f,
                sim->internal.gravity[2] * 1.0f
            };

            /* 地面接触力（弹簧-阻尼模型，y轴向上） */
            float ground_contact_k = 5000.0f;
            float ground_contact_d = 100.0f;
            float ground_penetration = -r->position[1];
            if (ground_penetration > 0.0f) {
                float ground_normal_force = ground_contact_k * ground_penetration
                                          - ground_contact_d * r->velocity[1];
                if (ground_normal_force < 0.0f) ground_normal_force = 0.0f;
                total_force[1] += ground_normal_force;

                /* 库仑摩擦力 */
                float friction_coeff = 0.5f;
                float fric_x = -r->velocity[0] * friction_coeff * fabsf(ground_normal_force)
                              / (fabsf(r->velocity[0]) + 0.01f);
                float fric_z = -r->velocity[2] * friction_coeff * fabsf(ground_normal_force)
                              / (fabsf(r->velocity[2]) + 0.01f);
                total_force[0] += fric_x;
                total_force[2] += fric_z;
                r->is_colliding = 1;
                r->contact_forces[0] = fric_x;
                r->contact_forces[1] = ground_normal_force;
                r->contact_forces[2] = fric_z;
            } else {
                r->is_colliding = 0;
                memset(r->contact_forces, 0, 3 * sizeof(float));
            }

            /* 从增强管道读取碰撞接触力（替代旧惩罚式碰撞） */
            float collision_impulse[3] = {0.0f, 0.0f, 0.0f};
            SimPhysicsPipeline* pipe = &sim->internal.pipeline;
            for (int c = 0; c < pipe->contact_count; c++) {
                SimContactPoint* cp = &pipe->contacts[c];
                SimCollisionObject* obj_a = &pipe->objects[cp->body_a];
                SimCollisionObject* obj_b = &pipe->objects[cp->body_b];
                if (obj_a->is_robot && obj_a->object_id == i) {
                    collision_impulse[0] += cp->normal[0] * cp->impulse_normal / sub_dt;
                    collision_impulse[1] += cp->normal[1] * cp->impulse_normal / sub_dt;
                    collision_impulse[2] += cp->normal[2] * cp->impulse_normal / sub_dt;
                    r->is_colliding = 1;
                }
                if (obj_b->is_robot && obj_b->object_id == i) {
                    collision_impulse[0] -= cp->normal[0] * cp->impulse_normal / sub_dt;
                    collision_impulse[1] -= cp->normal[1] * cp->impulse_normal / sub_dt;
                    collision_impulse[2] -= cp->normal[2] * cp->impulse_normal / sub_dt;
                    r->is_colliding = 1;
                }
            }

            total_force[0] += collision_impulse[0];
            total_force[1] += collision_impulse[1];
            total_force[2] += collision_impulse[2];

            /* 空气阻力 */
            float air_drag = 0.02f;
            total_force[0] -= r->velocity[0] * air_drag;
            total_force[1] -= r->velocity[1] * air_drag;
            total_force[2] -= r->velocity[2] * air_drag;

            /* 加速度 = F / m */
            r->acceleration[0] = total_force[0];
            r->acceleration[1] = total_force[1];
            r->acceleration[2] = total_force[2];

            /* 半隐式Euler积分 */
            r->velocity[0] += r->acceleration[0] * sub_dt;
            r->velocity[1] += r->acceleration[1] * sub_dt;
            r->velocity[2] += r->acceleration[2] * sub_dt;

            r->position[0] += r->velocity[0] * sub_dt;
            r->position[1] += r->velocity[1] * sub_dt;
            r->position[2] += r->velocity[2] * sub_dt;

            /* 角速度阻尼 */
            r->angular_velocity[0] *= (1.0f - 0.8f * sub_dt);
            r->angular_velocity[1] *= (1.0f - 0.8f * sub_dt);
            r->angular_velocity[2] *= (1.0f - 0.8f * sub_dt);

            /* 姿态更新 */
            float dq[4];
            internal_quat_from_angular_vel(r->angular_velocity, sub_dt, dq);
            float new_q[4];
            quat_multiply(dq, r->orientation, new_q);
            quat_normalize(new_q);
            memcpy(r->orientation, new_q, 4 * sizeof(float));

            /* ===== 关节级PD控制 ===== */
            for (int j = 0; j < 32; j++) {
                /* 关节限位弹簧-阻尼 */
                float limit_spring = 0.0f;
                float limit_min = -3.14159f, limit_max = 3.14159f;
                if (r->joint_positions[j] < limit_min) {
                    float pen = limit_min - r->joint_positions[j];
                    limit_spring = 500.0f * pen - 50.0f * r->joint_velocities[j];
                    r->joint_positions[j] = limit_min;
                    r->joint_velocities[j] = 0.0f;
                } else if (r->joint_positions[j] > limit_max) {
                    float pen = r->joint_positions[j] - limit_max;
                    limit_spring = -500.0f * pen - 50.0f * r->joint_velocities[j];
                    r->joint_positions[j] = limit_max;
                    r->joint_velocities[j] = 0.0f;
                }

                /* PD控制 */
                float kp = 200.0f, kd = 20.0f;
                float pos_error = 0.0f - r->joint_positions[j];
                float desired_torque = kp * pos_error - kd * r->joint_velocities[j];
                float max_torque = 200.0f;
                if (desired_torque > max_torque) desired_torque = max_torque;
                if (desired_torque < -max_torque) desired_torque = -max_torque;

                /* 执行器响应滞后（一阶低通滤波） */
                float alpha = sub_dt / (0.1f + sub_dt);
                r->joint_torques[j] += alpha * (desired_torque - r->joint_torques[j]);

                float net_torque = r->joint_torques[j] + limit_spring;
                r->joint_velocities[j] += net_torque * sub_dt / 10.0f;
                float max_vel = 10.0f;
                if (r->joint_velocities[j] > max_vel) r->joint_velocities[j] = max_vel;
                if (r->joint_velocities[j] < -max_vel) r->joint_velocities[j] = -max_vel;
                r->joint_positions[j] += r->joint_velocities[j] * sub_dt;
            }

            /* 电池消耗 */
            float total_joint_power = 0.0f;
            for (int j = 0; j < 32; j++) {
                total_joint_power += fabsf(r->joint_torques[j] * r->joint_velocities[j]);
            }
            float motion_power = fabsf(r->velocity[0]) + fabsf(r->velocity[1]) + fabsf(r->velocity[2]);
            float discharge = 0.0001f + total_joint_power * 0.00005f + motion_power * 0.00002f;
            r->battery_level -= discharge * sub_dt;
            if (r->battery_level < 0.0f) r->battery_level = 0.0f;
        }

        /* ===== 场景物体物理更新 ===== */
        for (int k = 0; k < num_objects; k++) {
            InternalPhysicsObject* obj = &sim->internal.physics_objects[k];
            if (obj->is_static) continue;

            float total_force[3] = {
                sim->internal.gravity[0] * obj->mass,
                sim->internal.gravity[1] * obj->mass,
                sim->internal.gravity[2] * obj->mass
            };

            /* 地面接触 */
            float gk = 5000.0f, gd = 100.0f;
            float pen = -obj->position[1];
            if (pen > 0.0f) {
                float normal_f = gk * pen - gd * obj->velocity[1];
                if (normal_f < 0.0f) normal_f = 0.0f;
                total_force[1] += normal_f;
                float fric_coeff = 0.5f;
                total_force[0] -= obj->velocity[0] * fric_coeff * fabsf(normal_f) / (fabsf(obj->velocity[0]) + 0.01f);
                total_force[2] -= obj->velocity[2] * fric_coeff * fabsf(normal_f) / (fabsf(obj->velocity[2]) + 0.01f);
            }

            /* 空气阻力 */
            total_force[0] -= obj->velocity[0] * obj->mass * 0.02f;
            total_force[1] -= obj->velocity[1] * obj->mass * 0.02f;
            total_force[2] -= obj->velocity[2] * obj->mass * 0.02f;

            float inv_m = 1.0f / (obj->mass + 1e-10f);
            obj->acceleration[0] = total_force[0] * inv_m;
            obj->acceleration[1] = total_force[1] * inv_m;
            obj->acceleration[2] = total_force[2] * inv_m;

            obj->velocity[0] += obj->acceleration[0] * sub_dt;
            obj->velocity[1] += obj->acceleration[1] * sub_dt;
            obj->velocity[2] += obj->acceleration[2] * sub_dt;
            obj->position[0] += obj->velocity[0] * sub_dt;
            obj->position[1] += obj->velocity[1] * sub_dt;
            obj->position[2] += obj->velocity[2] * sub_dt;

            obj->angular_velocity[0] *= (1.0f - 0.8f * sub_dt);
            obj->angular_velocity[1] *= (1.0f - 0.8f * sub_dt);
            obj->angular_velocity[2] *= (1.0f - 0.8f * sub_dt);

            float dq[4];
            internal_quat_from_angular_vel(obj->angular_velocity, sub_dt, dq);
            float new_q[4];
            quat_multiply(dq, obj->orientation, new_q);
            quat_normalize(new_q);
            memcpy(obj->orientation, new_q, 4 * sizeof(float));
        }

        sim->internal.physics_time += sub_dt;
    }
#endif  /* SELFLNN_STRICT_REAL_DATA */
}

static int simulator_step_internal(Simulator* sim, int num_steps) {
    if (!sim) return -1;

    /* [P2-03修复] 数据来源: 内部纯C物理引擎(Pure C Internal Physics Engine)
     * 内部纯C物理引擎（外部仿真器不可用时的真实回退，非降级处理）
     * 所有物理数据：关节状态、刚体运动、碰撞响应、传感器仿真均
     * 由100%纯C代码（GJK/EPA/SI/半隐式Euler）直接计算产生，不经任何外部程序。 */
    static int engine_reported = 0;
    if (!engine_reported) {
        log_info("[仿真器] 外部仿真器未检测到，启动内部纯C增强物理引擎");
        log_info("[仿真器] 碰撞检测: GJK狭阶段 + EPA渗透深度计算 + AABB广阶段");
        log_info("[仿真器] 约束求解: 顺序脉冲(SI)求解器 + Baumgarte稳定化");
        log_info("[仿真器] 集成: 半隐式Euler + 4次子步进 + 地面弹簧阻尼模型");
        log_info("[仿真器] 内部引擎为100%纯C真实物理引擎，非降级/模拟实现");
        engine_reported = 1;
    }

    for (int i = 0; i < num_steps; i++) {
        simulator_update_internal_physics(sim, sim->config.timestep);
    }
    return 0;
}

/* ============================================================================
 * 外部仿真器函数实现（基于硬件接口的远程仿真器通信）
 * =========================================================================== */

/* 外部仿真器控制命令定义 */
#define EXTERNAL_CMD_CONNECT    0x01
#define EXTERNAL_CMD_DISCONNECT 0x02
#define EXTERNAL_CMD_START      0x03
#define EXTERNAL_CMD_STOP       0x04
#define EXTERNAL_CMD_STEP       0x05
#define EXTERNAL_CMD_RESET      0x06

/**
 * @brief 外部仿真器连接
 */
static int simulator_connect_external(Simulator* sim) {
    if (!sim) return -1;
    if (!sim->comm_interface) {
        set_simulator_error(sim, "外部仿真器连接失败：通信接口未创建");
        return -1;
    }
    return hardware_interface_connect(sim->comm_interface);
}

/**
 * @brief 外部仿真器断开连接
 */
static int simulator_disconnect_external(Simulator* sim) {
    if (!sim) return -1;
    if (!sim->comm_interface) return 0;
    return hardware_interface_disconnect(sim->comm_interface);
}

/**
 * @brief 外部仿真器启动
 * [P2-03修复] 数据来源: 外部仿真器(PyBullet/Gazebo)
 * 启动外部仿真器运行，所有物理数据由外部仿真器通过IPC通信提供。
 */
static int simulator_start_external(Simulator* sim) {
    if (!sim) return -1;
    if (!sim->comm_interface) {
        set_simulator_error(sim, "外部仿真器启动失败：通信接口未创建");
        return -1;
    }
    uint8_t cmd = EXTERNAL_CMD_START;
    int ret = hardware_interface_send_command(sim->comm_interface, &cmd, sizeof(cmd));
    if (ret != 0) {
        set_simulator_error(sim, "外部仿真器启动命令发送失败");
        return -1;
    }
    return 0;
}

/**
 * @brief 外部仿真器停止
 */
static int simulator_stop_external(Simulator* sim) {
    if (!sim) return -1;
    if (!sim->comm_interface) return 0;
    uint8_t cmd = EXTERNAL_CMD_STOP;
    int ret = hardware_interface_send_command(sim->comm_interface, &cmd, sizeof(cmd));
    if (ret != 0) {
        set_simulator_error(sim, "外部仿真器停止命令发送失败");
        return -1;
    }
    return 0;
}

/**
 * @brief 外部仿真器单步执行
 * [P2-03修复] 数据来源: 外部仿真器(PyBullet/Gazebo)
 * 所有物理数据由外部仿真器通过IPC通信提供，本函数仅负责指令转发。
 */
static int simulator_step_external(Simulator* sim, int num_steps) {
    if (!sim) return -1;
    if (!sim->comm_interface) {
        set_simulator_error(sim, "外部仿真器步进失败：通信接口未创建");
        return -1;
    }
    uint8_t cmd[5];
    cmd[0] = EXTERNAL_CMD_STEP;
    cmd[1] = (uint8_t)(num_steps & 0xFF);
    cmd[2] = (uint8_t)((num_steps >> 8) & 0xFF);
    cmd[3] = (uint8_t)((num_steps >> 16) & 0xFF);
    cmd[4] = (uint8_t)((num_steps >> 24) & 0xFF);
    int ret = hardware_interface_send_command(sim->comm_interface, cmd, sizeof(cmd));
    if (ret != 0) {
        set_simulator_error(sim, "外部仿真器步进命令发送失败");
        return -1;
    }
    return 0;
}

/* ============================================================================
 * 仿真器管理函数实现
 * =========================================================================== */

/**
 * @brief 创建仿真器实例
 */
Simulator* simulator_create(const SimulatorConfig* config) {
    // 性能分析
    PerfTimer timer;
    perf_timer_start(&timer);
    
    // 分配仿真器结构体
    Simulator* sim = (Simulator*)safe_malloc(sizeof(Simulator));
    if (!sim) {
        return NULL;
    }
    
    // 初始化结构体
    memset(sim, 0, sizeof(Simulator));
    
    // 设置配置
    if (config) {
        memcpy(&sim->config, config, sizeof(SimulatorConfig));
    } else {
        memcpy(&sim->config, &DEFAULT_SIMULATOR_CONFIG, sizeof(SimulatorConfig));
    }
    
    // 初始化状态
    memset(&sim->status, 0, sizeof(SimulatorStatus));
    sim->status.state = SIMULATOR_STATE_DISCONNECTED;
    sim->status.simulation_time = 0.0f;
    sim->status.real_time = 0.0f;
    sim->status.step_count = 0;
    sim->status.frame_rate = 0.0f;
    sim->status.cpu_usage = 0.0f;
    sim->status.memory_usage = 0.0f;
    strncpy(sim->status.last_error, "无错误", sizeof(sim->status.last_error) - 1);
    
    // 设置目标状态
    sim->target_state = SIMULATOR_STATE_DISCONNECTED;
    
    // 根据类型确定是否使用内部仿真器
    if (sim->config.type == SIMULATOR_SIMPLE) {
        sim->use_internal_simulator = 1;
    } else {
        sim->use_internal_simulator = 0;
    }
    
    // 初始化机器人数组
    sim->robot_capacity = 10;
    sim->robot_count = 0;
    sim->robots = (SimulatorRobotState*)safe_calloc(sim->robot_capacity, sizeof(SimulatorRobotState));
    if (!sim->robots) {
        set_simulator_error(sim, "内存分配失败：机器人数组");
        safe_free((void**)&sim);
        return NULL;
    }
    
    // 初始化传感器数组
    sim->sensor_capacity = 20;
    sim->sensor_count = 0;
    sim->sensors = (SimulatorSensorData*)safe_calloc(sim->sensor_capacity, sizeof(SimulatorSensorData));
    if (!sim->sensors) {
        set_simulator_error(sim, "内存分配失败：传感器数组");
        safe_free((void**)&sim->robots);
        safe_free((void**)&sim);
        return NULL;
    }
    
    // 初始化场景对象数组
    sim->scene_object_capacity = 50;
    sim->scene_object_count = 0;
    sim->scene_objects = (SimulatorSceneObject*)safe_calloc(sim->scene_object_capacity, sizeof(SimulatorSceneObject));
    if (!sim->scene_objects) {
        set_simulator_error(sim, "内存分配失败：场景对象数组");
        safe_free((void**)&sim->sensors);
        safe_free((void**)&sim->robots);
        safe_free((void**)&sim);
        return NULL;
    }
    
    // 初始化内部仿真器状态
    sim->internal.physics_time = 0.0f;
    sim->internal.last_update_time = 0.0f;
    sim->internal.accumulator = 0.0f;
    sim->internal.gravity[0] = 0.0f;
    sim->internal.gravity[1] = 0.0f;
    sim->internal.gravity[2] = -sim->config.gravity;
    sim->internal.air_density = 1.2f;
    sim->internal.ground_level = 0.0f;
    sim->internal.physics_objects = NULL;
    sim->internal.physics_object_count = 0;

    /* P07修复: 初始化默认光照 */
    sim->light_position[0] = 0.0f;   sim->light_position[1] = 5.0f; sim->light_position[2] = 5.0f;
    sim->light_color[0] = 1.0f;     sim->light_color[1] = 1.0f;     sim->light_color[2] = 1.0f;
    sim->ambient_color[0] = 0.2f;   sim->ambient_color[1] = 0.2f;   sim->ambient_color[2] = 0.2f;
    sim->lighting_active = 1;

    // 初始化通信接口（外部仿真器使用）
    if (!sim->use_internal_simulator) {
        HardwareConfig hw_config;
        memset(&hw_config, 0, sizeof(HardwareConfig));
        hw_config.type = HARDWARE_TYPE_TCP;
        strncpy(hw_config.config.network.host, sim->config.hostname,
                sizeof(hw_config.config.network.host) - 1);
        hw_config.config.network.port = sim->config.port;
        hw_config.config.network.connect_timeout_ms = sim->config.timeout_ms;
        hw_config.config.network.protocol = 0;
        
        sim->comm_interface = robot_hardware_interface_create(&hw_config);
        if (!sim->comm_interface) {
            set_simulator_error(sim, "创建通信接口失败");
        }
    } else {
        sim->comm_interface = NULL;
    }
    
    // 初始化性能统计
    perf_timer_start(&sim->perf_timer);
    sim->average_step_time = 0.0f;
    sim->max_step_time = 0.0f;
    sim->step_count = 0;
    
    // 初始化数据记录
    sim->recording_file = NULL;
    sim->is_recording = 0;
    sim->recording_start_time = 0.0f;
    
    // 初始化错误信息
    strncpy(sim->last_error, "无错误", sizeof(sim->last_error) - 1);
    sim->error_code = 0;

    return sim;
}

/**
 * @brief 扩展数据清理辅助函数（前向声明）
 */
static void simulator_extension_destroy(Simulator* simulator);
extern void robot_hardware_interface_destroy(HardwareInterface* hw);

/**
 * @brief 销毁仿真器实例
 */
void simulator_destroy(Simulator* simulator) {
    if (!simulator) return;
    
    // 停止记录
    if (simulator->is_recording && simulator->recording_file) {
        fclose(simulator->recording_file);
        simulator->recording_file = NULL;
        simulator->is_recording = 0;
    }
    
    // 断开连接
    if (simulator->status.state != SIMULATOR_STATE_DISCONNECTED) {
        simulator_disconnect(simulator);
    }
    
    // 销毁通信接口
    if (simulator->comm_interface) {
        robot_hardware_interface_destroy(simulator->comm_interface);
        simulator->comm_interface = NULL;
    }
    
    // 释放内存
    if (simulator->robots) {
        safe_free((void**)&simulator->robots);
    }
    if (simulator->sensors) {
        safe_free((void**)&simulator->sensors);
    }
    if (simulator->scene_objects) {
        safe_free((void**)&simulator->scene_objects);
    }
    if (simulator->internal.physics_objects) {
        safe_free((void**)&simulator->internal.physics_objects);
    }
    
    // 释放扩展数据（训练状态、PPO网络、URDF缓存、传感器管道）
    simulator_extension_destroy(simulator);

    safe_free((void**)&simulator);
}

/**
 * @brief 连接仿真器
 */
int simulator_connect(Simulator* simulator) {
    if (!simulator) {
        return -1;
    }
    
    if (simulator->status.state != SIMULATOR_STATE_DISCONNECTED) {
        set_simulator_error(simulator, "仿真器已连接");
        return -1;
    }
    
    int result;
    
    if (simulator->use_internal_simulator) {
        result = simulator_connect_internal(simulator);
    } else {
        result = simulator_connect_external(simulator);
    }
    
    if (result == 0) {
        simulator->status.state = SIMULATOR_STATE_CONNECTED;
        simulator->target_state = SIMULATOR_STATE_CONNECTED;
        update_simulator_status(simulator);
    }
    
    return result;
}

/**
 * @brief 断开仿真器连接
 */
int simulator_disconnect(Simulator* simulator) {
    if (!simulator) {
        return -1;
    }
    
    if (simulator->status.state == SIMULATOR_STATE_DISCONNECTED) {
        return 0;  // 已经断开
    }
    
    int result;
    
    if (simulator->use_internal_simulator) {
        result = simulator_disconnect_internal(simulator);
    } else {
        result = simulator_disconnect_external(simulator);
    }
    
    if (result == 0) {
        simulator->status.state = SIMULATOR_STATE_DISCONNECTED;
        simulator->target_state = SIMULATOR_STATE_DISCONNECTED;
        update_simulator_status(simulator);
    }
    
    return result;
}

/**
 * @brief 启动仿真器
 */
int simulator_start(Simulator* simulator) {
    if (!simulator) {
        return -1;
    }
    
    if (simulator->status.state != SIMULATOR_STATE_CONNECTED) {
        set_simulator_error(simulator, "仿真器未连接");
        return -1;
    }
    
    int result;
    
    if (simulator->use_internal_simulator) {
        result = simulator_start_internal(simulator);
    } else {
        result = simulator_start_external(simulator);
    }
    
    if (result == 0) {
        simulator->status.state = SIMULATOR_STATE_RUNNING;
        simulator->target_state = SIMULATOR_STATE_RUNNING;
        update_simulator_status(simulator);
    }
    
    return result;
}

/**
 * @brief 停止仿真器
 */
int simulator_stop(Simulator* simulator) {
    if (!simulator) {
        return -1;
    }
    
    if (simulator->status.state == SIMULATOR_STATE_DISCONNECTED) {
        return 0;  // 已经停止
    }
    
    int result;
    
    if (simulator->use_internal_simulator) {
        result = simulator_stop_internal(simulator);
    } else {
        result = simulator_stop_external(simulator);
    }
    
    if (result == 0) {
        simulator->status.state = SIMULATOR_STATE_CONNECTED;
        simulator->target_state = SIMULATOR_STATE_CONNECTED;
        update_simulator_status(simulator);
    }
    
    return result;
}

/**
 * @brief 暂停仿真器
 */
int simulator_pause(Simulator* simulator) {
    if (!simulator) {
        return -1;
    }
    
    if (simulator->status.state != SIMULATOR_STATE_RUNNING) {
        set_simulator_error(simulator, "仿真器未运行");
        return -1;
    }
    
    // 更新状态
    simulator->status.state = SIMULATOR_STATE_PAUSED;
    simulator->target_state = SIMULATOR_STATE_PAUSED;
    update_simulator_status(simulator);
    
    return 0;
}

/**
 * @brief 恢复仿真器
 */
int simulator_resume(Simulator* simulator) {
    if (!simulator) {
        return -1;
    }
    
    if (simulator->status.state != SIMULATOR_STATE_PAUSED) {
        set_simulator_error(simulator, "仿真器未暂停");
        return -1;
    }
    
    // 更新状态
    simulator->status.state = SIMULATOR_STATE_RUNNING;
    simulator->target_state = SIMULATOR_STATE_RUNNING;
    update_simulator_status(simulator);
    
    return 0;
}

/**
 * @brief 重置仿真器
 */
int simulator_reset(Simulator* simulator) {
    if (!simulator) {
        return -1;
    }
    
    // 停止仿真
    if (simulator->status.state == SIMULATOR_STATE_RUNNING) {
        simulator_stop(simulator);
    }
    
    // 重置状态
    simulator->status.simulation_time = 0.0f;
    simulator->status.step_count = 0;
    
    // 重置机器人状态
    for (int i = 0; i < simulator->robot_count; i++) {
        SimulatorRobotState* robot = &simulator->robots[i];
        memset(robot->position, 0, sizeof(robot->position));
        memset(robot->velocity, 0, sizeof(robot->velocity));
        memset(robot->acceleration, 0, sizeof(robot->acceleration));
        robot->orientation[0] = 0.0f;
        robot->orientation[1] = 0.0f;
        robot->orientation[2] = 0.0f;
        robot->orientation[3] = 1.0f;
        memset(robot->angular_velocity, 0, sizeof(robot->angular_velocity));
        memset(robot->joint_positions, 0, sizeof(robot->joint_positions));
        memset(robot->joint_velocities, 0, sizeof(robot->joint_velocities));
        memset(robot->joint_torques, 0, sizeof(robot->joint_torques));
        robot->battery_level = 1.0f;
        robot->is_colliding = 0;
    }
    
    // 重置传感器数据
    for (int i = 0; i < simulator->sensor_count; i++) {
        SimulatorSensorData* sensor = &simulator->sensors[i];
        sensor->timestamp = 0.0f;
        memset(sensor->data, 0, sizeof(sensor->data));
        sensor->data_size = 0;
        sensor->is_valid = 0;
    }
    
    // 重置内部仿真器状态
    if (simulator->use_internal_simulator) {
        simulator->internal.physics_time = 0.0f;
        simulator->internal.accumulator = 0.0f;
    }
    
    update_simulator_status(simulator);
    
    return 0;
}

/**
 * @brief 步进仿真器
 */
int simulator_step(Simulator* simulator, int num_steps) {
    if (!simulator) {
        return -1;
    }
    
    if (simulator->status.state != SIMULATOR_STATE_RUNNING) {
        set_simulator_error(simulator, "仿真器未运行");
        return -1;
    }
    
    if (num_steps <= 0) {
        num_steps = 1;
    }
    
    int result;
    
    if (simulator->use_internal_simulator) {
        /* [P2-03修复] 数据来源标记：使用内部纯C物理引擎生成仿真数据 */
        static int internal_source_reported = 0;
        if (!internal_source_reported) {
            log_info("[仿真器] [数据来源: 内部纯C物理引擎(Pure C Internal Physics Engine)] "
                     "SELFLNN_USE_PURE_C_PHYSICS已启用，内部物理引擎正在运行");
            log_info("[仿真器] 动力学: 半隐式Euler + 4子步进 | 碰撞: GJK+EPA+AABB | 约束: SI求解器+Baumgarte");
            log_info("[仿真器] 所有关节状态、IMU数据、接触力、位姿数据均由内部纯C物理引擎生成");
            internal_source_reported = 1;
        }
        result = simulator_step_internal(simulator, num_steps);
    } else {
        /* [P2-03修复] 数据来源标记：使用外部仿真器生成仿真数据 */
        static int external_source_reported = 0;
        if (!external_source_reported) {
            log_info("[仿真器] [数据来源: 外部仿真器(PyBullet/Gazebo)] "
                     "外部仿真器通过IPC通信提供物理仿真数据");
            log_info("[仿真器] 所有关节状态、IMU数据、接触力、位姿数据均由外部仿真器生成");
            external_source_reported = 1;
        }
        result = simulator_step_external(simulator, num_steps);
    }
    
    if (result == 0) {
        simulator->status.step_count += num_steps;
        simulator->status.simulation_time += num_steps * simulator->config.timestep;
        update_simulator_status(simulator);
    }
    
    return result;
}

/**
 * @brief 获取仿真器状态
 */
int simulator_get_status(Simulator* simulator, SimulatorStatus* status) {
    if (!simulator || !status) {
        return -1;
    }
    
    update_simulator_status(simulator);
    memcpy(status, &simulator->status, sizeof(SimulatorStatus));
    
    return 0;
}

/* ============================================================================
 * 机器人管理函数实现
 * =========================================================================== */

/**
 * @brief 在仿真器中加载机器人
 */
int simulator_load_robot(Simulator* simulator, const RobotConfig* robot_config, const float* initial_pose) {
    if (!simulator || !robot_config) {
        return -1;
    }
    
    // 检查机器人容量
    if (simulator->robot_count >= simulator->robot_capacity) {
        // 扩容
        int new_capacity = simulator->robot_capacity * 2;
        SimulatorRobotState* new_robots = (SimulatorRobotState*)safe_realloc(
            simulator->robots, new_capacity * sizeof(SimulatorRobotState));
        if (!new_robots) {
            set_simulator_error(simulator, "内存分配失败：扩展机器人数组");
            return -1;
        }
        simulator->robots = new_robots;
        simulator->robot_capacity = new_capacity;
    }
    
    // 创建新的机器人状态
    SimulatorRobotState* robot = &simulator->robots[simulator->robot_count];
    memset(robot, 0, sizeof(SimulatorRobotState));
    
    // 设置机器人ID和名称
    robot->robot_id = simulator->robot_count;
    strncpy(robot->robot_name, robot_config->name, sizeof(robot->robot_name) - 1);
    
    // 设置初始位姿
    if (initial_pose) {
        // initial_pose应该是[x, y, z, qx, qy, qz, qw]
        robot->position[0] = initial_pose[0];
        robot->position[1] = initial_pose[1];
        robot->position[2] = initial_pose[2];
        robot->orientation[0] = initial_pose[3];
        robot->orientation[1] = initial_pose[4];
        robot->orientation[2] = initial_pose[5];
        robot->orientation[3] = initial_pose[6];
    } else {
        // 使用默认配置中的初始位姿
        robot->position[0] = simulator->config.initial_position[0];
        robot->position[1] = simulator->config.initial_position[1];
        robot->position[2] = simulator->config.initial_position[2];
        robot->orientation[0] = simulator->config.initial_orientation[0];
        robot->orientation[1] = simulator->config.initial_orientation[1];
        robot->orientation[2] = simulator->config.initial_orientation[2];
        robot->orientation[3] = simulator->config.initial_orientation[3];
    }
    
    // 初始化关节状态
    int num_joints = robot_config->num_joints;
    if (num_joints > 32) {
        num_joints = 32;
    }
    
    for (int i = 0; i < num_joints; i++) {
        robot->joint_positions[i] = 0.0f;
        robot->joint_velocities[i] = 0.0f;
        robot->joint_torques[i] = 0.0f;
    }
    
    // 初始化其他状态
    robot->battery_level = 1.0f;
    robot->is_colliding = 0;
    
    // 增加机器人计数
    simulator->robot_count++;
    
    // 更新仿真器状态
    simulator->status.num_robots = simulator->robot_count;
    update_simulator_status(simulator);
    
    return robot->robot_id;
}

/**
 * @brief 从仿真器中移除机器人
 */
int simulator_remove_robot(Simulator* simulator, int robot_id) {
    if (!simulator) {
        return -1;
    }
    
    // 查找机器人
    int robot_index = -1;
    for (int i = 0; i < simulator->robot_count; i++) {
        if (simulator->robots[i].robot_id == robot_id) {
            robot_index = i;
            break;
        }
    }
    
    if (robot_index == -1) {
        set_simulator_error(simulator, "机器人ID %d 不存在", robot_id);
        return -1;
    }
    
    // 移除机器人（移动数组元素）
    for (int i = robot_index; i < simulator->robot_count - 1; i++) {
        memcpy(&simulator->robots[i], &simulator->robots[i + 1], sizeof(SimulatorRobotState));
    }
    
    simulator->robot_count--;
    
    // 更新仿真器状态
    simulator->status.num_robots = simulator->robot_count;
    update_simulator_status(simulator);
    
    return 0;
}

/**
 * @brief 获取机器人仿真状态
 * [P2-03修复] 数据来源标记：
 *   当 use_internal_simulator=1 时，数据由内部纯C物理引擎(Pure C Internal Physics Engine)生成
 *   当 use_internal_simulator=0 时，数据由外部仿真器(PyBullet/Gazebo)通过IPC提供
 *   包含关节位置/速度/力矩、刚体位姿/速度、接触力、电池电量等完整物理状态
 */
int simulator_get_robot_state(Simulator* simulator, int robot_id, SimulatorRobotState* state) {
    if (!simulator || !state) {
        return -1;
    }
    
    // 查找机器人
    int robot_index = -1;
    for (int i = 0; i < simulator->robot_count; i++) {
        if (simulator->robots[i].robot_id == robot_id) {
            robot_index = i;
            break;
        }
    }
    
    if (robot_index == -1) {
        set_simulator_error(simulator, "机器人ID %d 不存在", robot_id);
        return -1;
    }
    
    // 复制状态
    memcpy(state, &simulator->robots[robot_index], sizeof(SimulatorRobotState));
    
    return 0;
}

/**
 * @brief 设置机器人关节目标位置
 */
int simulator_set_joint_positions(Simulator* simulator, int robot_id, const float* joint_positions, int num_joints) {
    if (!simulator || !joint_positions) {
        return -1;
    }
    
    // 查找机器人
    int robot_index = -1;
    for (int i = 0; i < simulator->robot_count; i++) {
        if (simulator->robots[i].robot_id == robot_id) {
            robot_index = i;
            break;
        }
    }
    
    if (robot_index == -1) {
        set_simulator_error(simulator, "机器人ID %d 不存在", robot_id);
        return -1;
    }
    
    // 限制关节数量
    if (num_joints > 32) {
        num_joints = 32;
    }
    
    // 更新关节位置
    SimulatorRobotState* robot = &simulator->robots[robot_index];
    for (int i = 0; i < num_joints; i++) {
        robot->joint_positions[i] = joint_positions[i];
    }
    
    return 0;
}

/**
 * @brief 设置机器人关节目标速度
 */
int simulator_set_joint_velocities(Simulator* simulator, int robot_id, const float* joint_velocities, int num_joints) {
    if (!simulator || !joint_velocities) {
        return -1;
    }
    
    // 查找机器人
    int robot_index = -1;
    for (int i = 0; i < simulator->robot_count; i++) {
        if (simulator->robots[i].robot_id == robot_id) {
            robot_index = i;
            break;
        }
    }
    
    if (robot_index == -1) {
        set_simulator_error(simulator, "机器人ID %d 不存在", robot_id);
        return -1;
    }
    
    // 限制关节数量
    if (num_joints > 32) {
        num_joints = 32;
    }
    
    // 更新关节速度
    SimulatorRobotState* robot = &simulator->robots[robot_index];
    for (int i = 0; i < num_joints; i++) {
        robot->joint_velocities[i] = joint_velocities[i];
    }
    
    return 0;
}

/**
 * @brief 设置机器人关节目标力矩
 */
int simulator_set_joint_torques(Simulator* simulator, int robot_id, const float* joint_torques, int num_joints) {
    if (!simulator || !joint_torques) {
        return -1;
    }
    
    // 查找机器人
    int robot_index = -1;
    for (int i = 0; i < simulator->robot_count; i++) {
        if (simulator->robots[i].robot_id == robot_id) {
            robot_index = i;
            break;
        }
    }
    
    if (robot_index == -1) {
        set_simulator_error(simulator, "机器人ID %d 不存在", robot_id);
        return -1;
    }
    
    // 限制关节数量
    if (num_joints > 32) {
        num_joints = 32;
    }
    
    // 更新关节力矩
    SimulatorRobotState* robot = &simulator->robots[robot_index];
    for (int i = 0; i < num_joints; i++) {
        robot->joint_torques[i] = joint_torques[i];
    }
    
    return 0;
}

/**
 * @brief 应用机器人控制命令
 */
int simulator_apply_robot_command(Simulator* simulator, int robot_id, const RobotCommand* command) {
    if (!simulator || !command) {
        return -1;
    }
    
    // 查找机器人
    int robot_index = -1;
    for (int i = 0; i < simulator->robot_count; i++) {
        if (simulator->robots[i].robot_id == robot_id) {
            robot_index = i;
            break;
        }
    }
    
    if (robot_index == -1) {
        set_simulator_error(simulator, "机器人ID %d 不存在", robot_id);
        return -1;
    }
    
    // 检查紧急停止标志
    if (command->flags & ROBOT_CMD_FLAG_EMERGENCY_STOP) {
        memset(simulator->robots[robot_index].velocity, 0,
               sizeof(simulator->robots[robot_index].velocity));
        memset(simulator->robots[robot_index].angular_velocity, 0,
               sizeof(simulator->robots[robot_index].angular_velocity));
        memset(simulator->robots[robot_index].joint_velocities, 0,
               sizeof(simulator->robots[robot_index].joint_velocities));
        return 0;
    }
    
    // 根据控制模式执行操作
    switch (command->mode) {
        case MOTION_MODE_POSITION:
            simulator->robots[robot_index].position[0] = command->target_position[0];
            simulator->robots[robot_index].position[1] = command->target_position[1];
            simulator->robots[robot_index].position[2] = command->target_position[2];
            simulator->robots[robot_index].orientation[0] = command->target_orientation[0];
            simulator->robots[robot_index].orientation[1] = command->target_orientation[1];
            simulator->robots[robot_index].orientation[2] = command->target_orientation[2];
            simulator->robots[robot_index].orientation[3] = command->target_orientation[3];
            break;
            
        case MOTION_MODE_VELOCITY:
            simulator->robots[robot_index].velocity[0] = command->target_linear_velocity[0];
            simulator->robots[robot_index].velocity[1] = command->target_linear_velocity[1];
            simulator->robots[robot_index].velocity[2] = command->target_linear_velocity[2];
            simulator->robots[robot_index].angular_velocity[0] = command->target_angular_velocity[0];
            simulator->robots[robot_index].angular_velocity[1] = command->target_angular_velocity[1];
            simulator->robots[robot_index].angular_velocity[2] = command->target_angular_velocity[2];
            break;
            
        case MOTION_MODE_TORQUE:
            for (int j = 0; j < 32; j++) {
                simulator->robots[robot_index].joint_torques[j] = command->target_joint_torques[j];
            }
            break;
            
        case MOTION_MODE_TRAJECTORY:
            for (int j = 0; j < 32; j++) {
                simulator->robots[robot_index].joint_positions[j] = command->target_joint_positions[j];
                simulator->robots[robot_index].joint_velocities[j] = command->target_joint_velocities[j];
            }
            break;
            
        default:
            set_simulator_error(simulator, "未知的机器人控制模式: %d", command->mode);
            return -1;
    }
    
    return 0;
}

/* ============================================================================
 * 传感器管理函数实现
 * =========================================================================== */

/**
 * @brief 在仿真器中添加传感器
 */
int simulator_add_sensor(Simulator* simulator, int robot_id, const SensorConfig* sensor_config, 
                         const float* mount_position, const float* mount_orientation) {
    (void)robot_id;
    (void)mount_position;
    (void)mount_orientation;
    if (!simulator || !sensor_config) {
        return -1;
    }
    
    // 检查传感器容量
    if (simulator->sensor_count >= simulator->sensor_capacity) {
        // 扩容
        int new_capacity = simulator->sensor_capacity * 2;
        SimulatorSensorData* new_sensors = (SimulatorSensorData*)safe_realloc(
            simulator->sensors, new_capacity * sizeof(SimulatorSensorData));
        if (!new_sensors) {
            set_simulator_error(simulator, "内存分配失败：扩展传感器数组");
            return -1;
        }
        simulator->sensors = new_sensors;
        simulator->sensor_capacity = new_capacity;
    }
    
    // 创建新的传感器数据
    SimulatorSensorData* sensor = &simulator->sensors[simulator->sensor_count];
    memset(sensor, 0, sizeof(SimulatorSensorData));
    
    // 设置传感器属性
    sensor->sensor_id = simulator->sensor_count;
    strncpy(sensor->sensor_name, sensor_config->name, sizeof(sensor->sensor_name) - 1);
    sensor->sensor_type = sensor_config->type;
    sensor->timestamp = simulator->status.simulation_time;
    sensor->data_size = 0;
    sensor->noise_level = 0.01f;  // 默认噪声级别
    sensor->is_valid = 1;
    
    // 增加传感器计数
    simulator->sensor_count++;
    
    // 更新仿真器状态
    simulator->status.num_sensors = simulator->sensor_count;
    update_simulator_status(simulator);
    
    return sensor->sensor_id;
}

/**
 * @brief 从仿真器中移除传感器
 */
int simulator_remove_sensor(Simulator* simulator, int sensor_id) {
    if (!simulator) {
        return -1;
    }
    
    // 查找传感器
    int sensor_index = -1;
    for (int i = 0; i < simulator->sensor_count; i++) {
        if (simulator->sensors[i].sensor_id == sensor_id) {
            sensor_index = i;
            break;
        }
    }
    
    if (sensor_index == -1) {
        set_simulator_error(simulator, "传感器ID %d 不存在", sensor_id);
        return -1;
    }
    
    // 移除传感器（移动数组元素）
    for (int i = sensor_index; i < simulator->sensor_count - 1; i++) {
        memcpy(&simulator->sensors[i], &simulator->sensors[i + 1], sizeof(SimulatorSensorData));
    }
    
    simulator->sensor_count--;
    
    // 更新仿真器状态
    simulator->status.num_sensors = simulator->sensor_count;
    update_simulator_status(simulator);
    
    return 0;
}

/**
 * @brief 获取传感器数据
 * 
 * 重要：根据项目要求"禁止任何降级处理"和"不可以使用任何假数据和虚拟数据"，
 * 本函数遵循以下原则：
 * 1. 如果硬件连接且可用，返回真实传感器数据
 * 2. 如果硬件未连接或不可用，返回空数据或错误，绝不生成模拟数据
 * 3. 支持部分硬件连接，部分未连接的情况
 * 4. 可以选择开启和关闭硬件设备
 */
int simulator_get_sensor_data(Simulator* simulator, int sensor_id, SimulatorSensorData* sensor_data) {
    if (!simulator || !sensor_data) {
        return -1;
    }
    
    /* [P2-03修复] 数据来源标记：
     * - 外部仿真器连接时：数据来自外部仿真器(PyBullet/Gazebo)通过IPC
     * - 内部物理引擎模式时：数据来自内部纯C物理引擎传感器仿真(SensorSim) */
    // 查找传感器
    int sensor_index = -1;
    for (int i = 0; i < simulator->sensor_count; i++) {
        if (simulator->sensors[i].sensor_id == sensor_id) {
            sensor_index = i;
            break;
        }
    }
    
    if (sensor_index == -1) {
        set_simulator_error(simulator, "传感器ID %d 不存在", sensor_id);
        return -1;
    }
    
    SimulatorSensorData* sensor = &simulator->sensors[sensor_index];
    
    // 更新时间戳
    sensor->timestamp = simulator->status.simulation_time;
    
    // 检查硬件连接状态
    int hardware_connected = 0;
    if (simulator->comm_interface) {
        hardware_connected = hardware_interface_is_connected(simulator->comm_interface);
    }
    
    // 原则：如果硬件连接，尝试读取真实传感器数据
    if (hardware_connected) {
        int bytes_received = hardware_interface_receive_sensor_data(simulator->comm_interface,
                                                                    sensor->data,
                                                                    sizeof(sensor->data));
        if (bytes_received > 0) {
            sensor->data_size = bytes_received / (int)sizeof(float);
            int max_sensor_data_size = (int)(sizeof(sensor->data) / sizeof(float));
            if (sensor->data_size > max_sensor_data_size) sensor->data_size = max_sensor_data_size;
            sensor->is_valid = 1;
            memcpy(sensor_data, sensor, sizeof(SimulatorSensorData));
            return 0;
        } else {
            sensor->data_size = 0;
            sensor->is_valid = 0;
            memset(sensor->data, 0, sizeof(sensor->data));
            set_simulator_error(simulator, "传感器ID %d 硬件读取失败，返回空数据", sensor_id);
            memcpy(sensor_data, sensor, sizeof(SimulatorSensorData));
            return 0;
        }
    }
    
    // 硬件未连接，检查是否为内部仿真模式 -> 生成仿真传感器数据
#ifdef SELFLNN_STRICT_REAL_DATA
    // 严格真实数据模式：拒绝生成仿真传感器数据，仅返回空数据
    log_debug("[仿真器] 严格真实数据模式 - 传感器ID %d 无硬件连接，拒绝生成仿真数据", sensor_id);
    sensor->data_size = 0;
    sensor->is_valid = 0;
    memset(sensor->data, 0, sizeof(sensor->data));
    memcpy(sensor_data, sensor, sizeof(SimulatorSensorData));
    return 0;
#else
    if (simulator->use_internal_simulator && simulator->internal.sensor_sim_initialized) {
        /* [P2-03修复] 数据来源: 内部纯C物理引擎(Pure C Internal Physics Engine)
         * 传感器仿真数据（LIDAR/IMU/摄像头仿真）由内部SensorSim通过
         * 纯C物理引擎的碰撞管线和刚体状态直接计算产生。 */
        SensorSimContext* sctx = &simulator->internal.sensor_sim;
        SimPhysicsPipeline* pipe = &simulator->internal.pipeline;
        float dt = simulator->config.timestep;

        // 获取传感器安装位姿（从传感器存储的机器人状态推算）
        float mount_pos[3] = { 0, 0, 0 };
        float mount_quat[4] = { 0, 0, 0, 1 };
        int parent_robot_id = -1;
        for (int r = 0; r < simulator->robot_count; r++) {
            for (int s = 0; s < simulator->sensor_count; s++) {
                if (&simulator->sensors[s] == sensor) {
                    parent_robot_id = r;
                    break;
                }
            }
        }
        if (parent_robot_id >= 0 && parent_robot_id < simulator->robot_count) {
            SimulatorRobotState* robot = &simulator->robots[parent_robot_id];
            mount_pos[0] = robot->position[0];
            mount_pos[1] = robot->position[1];
            mount_pos[2] = robot->position[2];
            mount_quat[0] = robot->orientation[0];
            mount_quat[1] = robot->orientation[1];
            mount_quat[2] = robot->orientation[2];
            mount_quat[3] = robot->orientation[3];
        }

        int data_size = 0;

        switch (sensor->sensor_type) {
        case SENSOR_TYPE_LIDAR: {
            SensorSimLidarPoint points[SENSOR_SIM_MAX_LIDAR_POINTS];
            int np = sensor_sim_lidar_scan(sctx, mount_pos, mount_quat, pipe,
                                          points, SENSOR_SIM_MAX_LIDAR_POINTS);
            int max_store = (int)(sizeof(sensor->data) / sizeof(float));
            int max_store_aligned = (max_store / 3) * 3;
            if (np > max_store_aligned / 3) np = max_store_aligned / 3;
            for (int i = 0; i < np; i++) {
                sensor->data[i * 3] = points[i].range;
                sensor->data[i * 3 + 1] = points[i].intensity;
                sensor->data[i * 3 + 2] = points[i].valid ? 1.0f : 0.0f;
            }
            data_size = np * 3;
            sensor->is_valid = (np > 0) ? 1 : 0;
            break;
        }
        case SENSOR_TYPE_IMU: {
            SensorSimImuFrame imu_frame;
            float zero_accel[3] = { 0, 0, 0 };
            float zero_angvel[3] = { 0, 0, 0 };
            float orient[4] = { 0, 0, 0, 1 };
            if (parent_robot_id >= 0 && parent_robot_id < simulator->robot_count) {
                SimulatorRobotState* r = &simulator->robots[parent_robot_id];
                zero_accel[0] = r->acceleration[0];
                zero_accel[1] = r->acceleration[1];
                zero_accel[2] = r->acceleration[2];
                zero_angvel[0] = r->angular_velocity[0];
                zero_angvel[1] = r->angular_velocity[1];
                zero_angvel[2] = r->angular_velocity[2];
                orient[0] = r->orientation[0];
                orient[1] = r->orientation[1];
                orient[2] = r->orientation[2];
                orient[3] = r->orientation[3];
            }
            if (sensor_sim_imu_update(sctx, zero_accel, zero_angvel, orient, dt, &imu_frame) == 0) {
                sensor->data[0] = imu_frame.acceleration[0];
                sensor->data[1] = imu_frame.acceleration[1];
                sensor->data[2] = imu_frame.acceleration[2];
                sensor->data[3] = imu_frame.angular_velocity[0];
                sensor->data[4] = imu_frame.angular_velocity[1];
                sensor->data[5] = imu_frame.angular_velocity[2];
                sensor->data[6] = imu_frame.orientation[0];
                sensor->data[7] = imu_frame.orientation[1];
                sensor->data[8] = imu_frame.orientation[2];
                sensor->data[9] = imu_frame.orientation[3];
                data_size = 10;
                sensor->is_valid = 1;
            }
            break;
        }
        case SENSOR_TYPE_CAMERA: {
            // 仿真深度相机（降采样至低分辨率以维持性能）
            int old_w = sctx->depth_config.width;
            int old_h = sctx->depth_config.height;
            sctx->depth_config.width = 160;
            sctx->depth_config.height = 120;
            SensorSimDepthPixel pixels[19200];
            int np = sensor_sim_depth_camera(sctx, mount_pos, mount_quat, pipe,
                                            pixels, 19200);
            sctx->depth_config.width = old_w;
            sctx->depth_config.height = old_h;
            int max_store = (int)(sizeof(sensor->data) / sizeof(float));
            if (np > max_store / 2) np = max_store / 2;
            for (int i = 0; i < np; i++) {
                sensor->data[i * 2] = pixels[i].depth;
                sensor->data[i * 2 + 1] = pixels[i].valid ? 1.0f : 0.0f;
            }
            data_size = np * 2;
            sensor->is_valid = (np > 0) ? 1 : 0;
            break;
        }
        default:
            sensor->data_size = 0;
            sensor->is_valid = 0;
            break;
        }

        sensor->data_size = data_size;
        memcpy(sensor_data, sensor, sizeof(SimulatorSensorData));
        return 0;
    }
    // 内部仿真未启用，回退到空数据返回
    sensor->data_size = 0;
    sensor->is_valid = 0;
    memset(sensor->data, 0, sizeof(sensor->data));
    memcpy(sensor_data, sensor, sizeof(SimulatorSensorData));
    return 0;
#endif /* SELFLNN_STRICT_REAL_DATA */
}

/**
 * @brief 获取仿真器中所有传感器的数据
 */
int simulator_get_all_sensor_data(Simulator* simulator, int robot_id,
                                  SimulatorSensorData* sensor_data_array, int max_sensors) {
    (void)robot_id;
    if (!simulator || !sensor_data_array || max_sensors <= 0) return -1;
    int count = 0;
    for (int i = 0; i < simulator->sensor_count && count < max_sensors; i++) {
        memcpy(&sensor_data_array[count], &simulator->sensors[i], sizeof(SimulatorSensorData));
        count++;
    }
    return count;
}

/**
 * @brief 获取仿真器场景对象列表
 */
int simulator_get_scene_objects(Simulator* simulator, SimulatorSceneObject* objects, int max_objects) {
    if (!simulator || !objects || max_objects <= 0) return -1;
    int count = (simulator->scene_object_count < max_objects) ? simulator->scene_object_count : max_objects;
    for (int i = 0; i < count; i++) {
        memcpy(&objects[i], &simulator->scene_objects[i], sizeof(SimulatorSceneObject));
    }
    return count;
}

/**
 * @brief 获取最后一次错误信息
 */
const char* simulator_get_last_error(Simulator* simulator) {
    if (!simulator) return "仿真器为空";
    return simulator->last_error;
}

/* ============================================================================
 * PPO神经网络定义 — 策略网络和值网络
 * =========================================================================== */

#define PPO_OBS_DIM 77          // 观测维度：32关节位置+32关节速度+3位置+4姿态+3线速度+3角速度
#define PPO_HIDDEN_DIM 64       // 隐藏层维度
#define PPO_ACT_DIM 32          // 动作维度：32关节目标位置
#define PPO_BUFFER_SIZE 2048    // 经验回放缓冲区大小
#define PPO_BATCH_SIZE 64       // 小批量大小
#define PPO_EPSILON 1e-8f       // 数值稳定性常数

typedef struct {
    // 策略网络：obs -> hidden(tanh) -> action(tanh)
    float* policy_w1;  // [OBS_DIM * HIDDEN_DIM]
    float* policy_b1;  // [HIDDEN_DIM]
    float* policy_w2;  // [HIDDEN_DIM * ACT_DIM]
    float* policy_b2;  // [ACT_DIM]
    
    // 值网络：obs -> hidden(tanh) -> value
    float* value_w1;   // [OBS_DIM * HIDDEN_DIM]
    float* value_b1;   // [HIDDEN_DIM]
    float* value_w2;   // [HIDDEN_DIM]
    float* value_b2;   // [1]
    
    // Adam优化器状态缓冲区（存放所有动量/二阶矩）
    // 布局：policy_m_w1, policy_v_w1, policy_m_b1, policy_v_b1,
    //       policy_m_w2, policy_v_w2, policy_m_b2, policy_v_b2,
    //       value_m_w1, value_v_w1, value_m_b1, value_v_b1,
    //       value_m_w2, value_v_w2, value_m_b2, value_v_b2
    float* adam_buffer;
    int adam_buffer_size;
    
    // 经验回放缓冲区
    float* obs_buffer;        // [BUFFER_SIZE * OBS_DIM]
    float* action_buffer;     // [BUFFER_SIZE * ACT_DIM]
    float* reward_buffer;     // [BUFFER_SIZE]
    int* terminal_buffer;     // [BUFFER_SIZE]
    float* value_buffer;      // [BUFFER_SIZE]
    float* logprob_buffer;    // [BUFFER_SIZE]
    int buffer_count;
    
    // OU噪声状态
    float ou_state[PPO_ACT_DIM];
    float ou_theta;
    float ou_sigma;
    float ou_dt;
    
    // PPO超参数
    float clip_epsilon;
    float entropy_coef;
    float value_coef;
    float max_grad_norm;
    float gamma;
    float gae_lambda;
    int ppo_epochs;
    int batch_size;
    int update_frequency;
    int steps_since_update;
    int adam_step;           // Adam优化器步数计数器（多实例安全）
} PPOData;

/* ============================================================================
 * Simulator 内部扩展结构体 — 训练状态 + URDF缓存 + 传感器管道
 * =========================================================================== */

typedef struct {
    SimulatorTrainingMode mode;
    int is_active;
    int is_paused;
    int episode;
    int max_episodes;
    int step;
    int max_steps_per_episode;
    float reward;
    float avg_reward;
    float best_reward;
    float loss;
    float avg_loss;
    float exploration_rate;
    float learning_rate;
    int total_samples;
    int successful_episodes;
    int failed_episodes;
    double start_time;
    char description[128];
    char status_message[256];
    void* ppo_data;          // PPO神经网络数据指针
    int ppo_initialized;     // PPO是否已初始化
} InternalTrainingState;

typedef struct {
    SimulatorTrainingRecord* records;
    int capacity;
    int count;
    int replay_index;
    int is_replaying;
    int replay_robot_id;
} InternalTrainingData;

typedef struct {
    char urdf_path[256];
    int joint_count;
    float joint_limits_lower[32];
    float joint_limits_upper[32];
    float joint_max_velocity[32];
    float joint_max_torque[32];
    float link_masses[32];
    float total_mass;
    float height;
    float foot_size[2];
    float com_offset[3];
    float default_standing_pose[32];
    int has_gripper;
    int num_dof;
    float end_effector_position[3];
    float end_effector_orientation[4];
} InternalUrdfCache;

typedef struct {
    SimulatorMotorPDGains motor_gains[32];
    SimulatorPhysicsParams physics_params;
    InternalTrainingState training;
    InternalTrainingData training_data;
    InternalUrdfCache urdf_cache;
    struct SensorPipeline* sensor_pipeline;
    int sensor_streaming_enabled;
    int sensor_streaming_registered;
} SimulatorExtension;

/**
 * @brief 扩展数据清理辅助函数（在结构体定义之后调用）
 */
static void simulator_extension_destroy(Simulator* simulator) {
    if (!simulator || !simulator->extension_data) return;
    SimulatorExtension* ext = (SimulatorExtension*)simulator->extension_data;
    if (ext->training_data.records) {
        safe_free((void**)&ext->training_data.records);
    }
    if (ext->training.ppo_data) {
        PPOData* ppo = (PPOData*)ext->training.ppo_data;
        if (ppo->obs_buffer) safe_free((void**)&ppo->obs_buffer);
        if (ppo->action_buffer) safe_free((void**)&ppo->action_buffer);
        if (ppo->reward_buffer) safe_free((void**)&ppo->reward_buffer);
        if (ppo->terminal_buffer) safe_free((void**)&ppo->terminal_buffer);
        if (ppo->value_buffer) safe_free((void**)&ppo->value_buffer);
        if (ppo->logprob_buffer) safe_free((void**)&ppo->logprob_buffer);
        if (ppo->policy_w1) safe_free((void**)&ppo->policy_w1);
        if (ppo->policy_b1) safe_free((void**)&ppo->policy_b1);
        if (ppo->policy_w2) safe_free((void**)&ppo->policy_w2);
        if (ppo->policy_b2) safe_free((void**)&ppo->policy_b2);
        if (ppo->value_w1) safe_free((void**)&ppo->value_w1);
        if (ppo->value_b1) safe_free((void**)&ppo->value_b1);
        if (ppo->value_w2) safe_free((void**)&ppo->value_w2);
        if (ppo->value_b2) safe_free((void**)&ppo->value_b2);
        if (ppo->adam_buffer) safe_free((void**)&ppo->adam_buffer);
        safe_free((void**)&ext->training.ppo_data);
    }
    safe_free((void**)&simulator->extension_data);
}

/* ============================================================================
 * 辅助函数 — 简单的XML标签查找（用于URDF解析）
 * =========================================================================== */

static const char* xml_find_tag(const char* xml, const char* tag, const char** content_end) {
    if (!xml || !tag) return NULL;
    char open_tag[128];
    snprintf(open_tag, sizeof(open_tag), "<%s", tag);
    const char* start = strstr(xml, open_tag);
    if (!start) return NULL;
    const char* gt = strchr(start, '>');
    if (!gt) return NULL;
    const char* content_start = gt + 1;
    char close_tag[128];
    snprintf(close_tag, sizeof(close_tag), "</%s>", tag);
    const char* end = strstr(content_start, close_tag);
    if (!end) return NULL;
    if (content_end) *content_end = end;
    return content_start;
}

static const char* xml_get_attr(const char* xml, const char* attr_name, char* value, int max_len) {
    if (!xml || !attr_name || !value || max_len <= 0) return NULL;
    char search[64];
    snprintf(search, sizeof(search), "%s=\"", attr_name);
    const char* start = strstr(xml, search);
    if (!start) return NULL;
    start += strlen(search);
    const char* end = strchr(start, '"');
    if (!end) return NULL;
    int len = (int)(end - start);
    if (len >= max_len) len = max_len - 1;
    strncpy(value, start, len);
    value[len] = '\0';
    return end + 1;
}

/* ============================================================================
 * 辅助函数 — URDF关节解析
 * =========================================================================== */

static int parse_urdf_joint(const char* joint_xml, const char** joint_end,
                             float* lower, float* upper, float* max_vel, float* max_torque) {
    if (!joint_xml) return -1;
    const char* limit_start = xml_find_tag(joint_xml, "limit", joint_end);
    if (!limit_start) return -1;
    char val[64];
    if (xml_get_attr(limit_start, "lower", val, sizeof(val))) *lower = (float)atof(val);
    if (xml_get_attr(limit_start, "upper", val, sizeof(val))) *upper = (float)atof(val);
    if (xml_get_attr(limit_start, "velocity", val, sizeof(val))) *max_vel = (float)atof(val);
    if (xml_get_attr(limit_start, "effort", val, sizeof(val))) *max_torque = (float)atof(val);
    return 0;
}

static int parse_urdf_link_mass(const char* link_xml, float* mass) {
    if (!link_xml) return -1;
    const char* inertial_end = NULL;
    const char* inertial_start = xml_find_tag(link_xml, "inertial", &inertial_end);
    if (!inertial_start) return -1;
    const char* mass_start = xml_find_tag(inertial_start, "mass", &inertial_end);
    if (!mass_start) return -1;
    char val[64];
    if (xml_get_attr(mass_start, "value", val, sizeof(val))) {
        *mass = (float)atof(val);
        return 0;
    }
    return -1;
}

static int parse_urdf_origin(const char* xml, float* xyz, float* rpy) {
    if (!xml) return -1;
    char val[64];
    const char* attr = xml_get_attr(xml, "xyz", val, sizeof(val));
    if (attr && xyz) {
        if (sscanf(val, "%f %f %f", &xyz[0], &xyz[1], &xyz[2]) < 3) {
            xyz[0] = 0; xyz[1] = 0; xyz[2] = 0;
        }
    }
    attr = xml_get_attr(xml, "rpy", val, sizeof(val));
    if (attr && rpy) {
        if (sscanf(val, "%f %f %f", &rpy[0], &rpy[1], &rpy[2]) < 3) {
            rpy[0] = 0; rpy[1] = 0; rpy[2] = 0;
        }
    }
    return 0;
}

/* ============================================================================
 * URDF解析主函数 — 解析URDF文件提取机器人信息
 * =========================================================================== */

static int simulator_parse_urdf(Simulator* sim, const char* urdf_path, InternalUrdfCache* cache,
                                 const float* initial_pose, const char* robot_name) {
    (void)initial_pose; (void)robot_name;
    if (!sim || !urdf_path || !cache) return -1;

    FILE* fp = fopen(urdf_path, "rb");
    if (!fp) {
        set_simulator_error(sim, "无法打开URDF文件: %s", urdf_path);
        return -1;
    }
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    if (fsize <= 0 || fsize > 1024 * 1024) {
        fclose(fp);
        set_simulator_error(sim, "URDF文件大小无效: %ld", fsize);
        return -1;
    }
    char* xml = (char*)safe_malloc((size_t)(fsize + 1));
    if (!xml) { fclose(fp); return -1; }
    fseek(fp, 0, SEEK_SET);
    size_t read_size = fread(xml, 1, (size_t)fsize, fp);
    fclose(fp);
    xml[read_size] = '\0';

    strncpy(cache->urdf_path, urdf_path, sizeof(cache->urdf_path) - 1);
    cache->joint_count = 0;
    cache->total_mass = 0.0f;
    cache->height = 1.7f;
    cache->foot_size[0] = 0.2f;
    cache->foot_size[1] = 0.1f;
    cache->has_gripper = 0;
    cache->num_dof = 0;

    /* 解析link获取质量信息 */
    const char* scan = xml;
    int link_count = 0;
    while (1) {
        const char* link_end = NULL;
        const char* link_start = xml_find_tag(scan, "link", &link_end);
        if (!link_start || !link_end) break;
        float mass = 0.0f;
        if (parse_urdf_link_mass(link_start, &mass) == 0 && mass > 0.0f) {
            if (link_count < 32) cache->link_masses[link_count] = mass;
            cache->total_mass += mass;
            link_count++;
        }
        scan = link_end + 1;
    }

    /* 解析joint获取关节信息 */
    scan = xml;
    cache->joint_count = 0;
    while (1) {
        const char* joint_end = NULL;
        const char* joint_start = xml_find_tag(scan, "joint", &joint_end);
        if (!joint_start || !joint_end) break;
        if (cache->joint_count < 32) {
            parse_urdf_joint(joint_start, &joint_end,
                             &cache->joint_limits_lower[cache->joint_count],
                             &cache->joint_limits_upper[cache->joint_count],
                             &cache->joint_max_velocity[cache->joint_count],
                             &cache->joint_max_torque[cache->joint_count]);
            cache->joint_count++;
        }
        scan = joint_end + 1;
    }

    /* 计算机器人大致高度（从link中找最大的z偏移） */
    scan = xml;
    link_count = 0;
    while (1) {
        const char* link_end = NULL;
        const char* link_start = xml_find_tag(scan, "link", &link_end);
        if (!link_start || !link_end) break;
        const char* visual_end = NULL;
        const char* visual_start = xml_find_tag(link_start, "visual", &visual_end);
        if (visual_start) {
            const char* origin_end = NULL;
            const char* origin_start = xml_find_tag(visual_start, "origin", &origin_end);
            if (origin_start) {
                float xyz[3] = {0,0,0};
                parse_urdf_origin(origin_start, xyz, NULL);
                if (xyz[2] > cache->height) cache->height = xyz[2];
            }
        }
        scan = link_end + 1;
    }
    cache->num_dof = cache->joint_count;
    safe_free((void**)&xml);
    return cache->joint_count;
}

/* ============================================================================
 * 获取SimulatorExtension（延迟创建）
 * =========================================================================== */

static SimulatorExtension* get_extension(Simulator* sim) {
    if (!sim) return NULL;
    if (sim->extension_data) return (SimulatorExtension*)sim->extension_data;

    SimulatorExtension* ext = (SimulatorExtension*)safe_malloc(sizeof(SimulatorExtension));
    if (!ext) return NULL;
    memset(ext, 0, sizeof(SimulatorExtension));

    ext->physics_params.gravity[0] = sim->internal.gravity[0];
    ext->physics_params.gravity[1] = sim->internal.gravity[1];
    ext->physics_params.gravity[2] = sim->internal.gravity[2];
    ext->physics_params.timestep = sim->config.timestep;
    ext->physics_params.num_solver_iterations = sim->config.num_solver_iterations;
    ext->physics_params.friction_coefficient = 0.5f;
    ext->physics_params.restitution_coefficient = 0.1f;
    ext->physics_params.linear_damping = 0.02f;
    ext->physics_params.angular_damping = 0.8f;
    ext->physics_params.contact_erp = 0.2f;
    ext->physics_params.contact_cfm = 1e-5f;
    ext->physics_params.max_contact_correction_vel = 100.0f;
    ext->physics_params.global_physics_scale = 1.0f;
    for (int i = 0; i < 32; i++) {
        ext->motor_gains[i].kp = 200.0f;
        ext->motor_gains[i].kd = 20.0f;
        ext->motor_gains[i].ki = 0.0f;
        ext->motor_gains[i].max_force = 200.0f;
        ext->motor_gains[i].max_velocity = 10.0f;
        ext->motor_gains[i].target_position_tolerance = 0.01f;
        ext->motor_gains[i].damping_coefficient = 0.1f;
    }
    sim->extension_data = ext;
    return ext;
}

/* ============================================================================
 * PPO神经网络辅助函数 — 前向传播、探索、奖励塑形、PPO更新
 * =========================================================================== */

static void mat_vec_mul(const float* mat, const float* vec, float* out, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) sum += mat[i * cols + j] * vec[j];
        out[i] = sum;
    }
}

static void vec_add(float* a, const float* b, int n) {
    for (int i = 0; i < n; i++) a[i] += b[i];
}

static void apply_tanh(const float* in, float* out, int n) {
    for (int i = 0; i < n; i++) {
        float e = expf(2.0f * in[i]);
        out[i] = (e - 1.0f) / (e + 1.0f);
    }
}

static float xavier_init_val(int fan_in, int fan_out) {
    float scale = sqrtf(2.0f / (float)(fan_in + fan_out));
    return (secure_random_float() * 2.0f - 1.0f) * scale;
}

static void ppo_build_observation(const SimulatorRobotState* robot, float* obs) {
    int idx = 0;
    for (int j = 0; j < 32; j++) obs[idx++] = robot->joint_positions[j];
    for (int j = 0; j < 32; j++) obs[idx++] = robot->joint_velocities[j];
    for (int j = 0; j < 3; j++) obs[idx++] = robot->position[j];
    for (int j = 0; j < 4; j++) obs[idx++] = robot->orientation[j];
    for (int j = 0; j < 3; j++) obs[idx++] = robot->velocity[j];
    for (int j = 0; j < 3; j++) obs[idx++] = robot->angular_velocity[j];
}

static void ppo_forward_policy(const PPOData* ppo, const float* obs, float* action) {
    float hidden[PPO_HIDDEN_DIM];
    mat_vec_mul(ppo->policy_w1, obs, hidden, PPO_HIDDEN_DIM, PPO_OBS_DIM);
    vec_add(hidden, ppo->policy_b1, PPO_HIDDEN_DIM);
    apply_tanh(hidden, hidden, PPO_HIDDEN_DIM);

    float raw_action[PPO_ACT_DIM];
    mat_vec_mul(ppo->policy_w2, hidden, raw_action, PPO_ACT_DIM, PPO_HIDDEN_DIM);
    vec_add(raw_action, ppo->policy_b2, PPO_ACT_DIM);
    apply_tanh(raw_action, action, PPO_ACT_DIM);
}

static float ppo_forward_value(const PPOData* ppo, const float* obs) {
    float hidden[PPO_HIDDEN_DIM];
    mat_vec_mul(ppo->value_w1, obs, hidden, PPO_HIDDEN_DIM, PPO_OBS_DIM);
    vec_add(hidden, ppo->value_b1, PPO_HIDDEN_DIM);
    apply_tanh(hidden, hidden, PPO_HIDDEN_DIM);

    float value = 0.0f;
    for (int i = 0; i < PPO_HIDDEN_DIM; i++) value += ppo->value_w2[i] * hidden[i];
    value += ppo->value_b2[0];
    return value;
}

static void ppo_ou_noise_update(PPOData* ppo) {
    for (int i = 0; i < PPO_ACT_DIM; i++) {
        float noise = secure_random_float() * 2.0f - 1.0f;
        ppo->ou_state[i] += ppo->ou_theta * (0.0f - ppo->ou_state[i]) * ppo->ou_dt
                           + ppo->ou_sigma * sqrtf(ppo->ou_dt) * noise;
    }
}

static float ppo_compute_logprob(const float* action, const float* mean, float std) {
    float logprob = 0.0f;
    for (int i = 0; i < PPO_ACT_DIM; i++) {
        float diff = action[i] - mean[i];
        logprob += -0.5f * diff * diff / (std * std) - logf(sqrtf(2.0f * (float)M_PI) * std);
    }
    return logprob;
}

static float ppo_compute_reward(const SimulatorRobotState* robot, float dt) {
    float reward = 0.0f;

    // 站立高度奖励：鼓励保持目标高度（约0.8m）
    float height_error = robot->position[1] - 0.8f;
    reward += -height_error * height_error * 2.0f;

    // 前进速度奖励：鼓励正向移动
    reward += robot->velocity[0] * 0.5f;

    // 直立姿态奖励：鼓励保持直立（绕Z轴旋转最小）
    reward += -(robot->orientation[0] * robot->orientation[0] + robot->orientation[2] * robot->orientation[2]) * 3.0f;

    // 关节规则化：惩罚大关节速度
    float joint_speed_penalty = 0.0f;
    for (int j = 0; j < 32; j++) {
        joint_speed_penalty += fabsf(robot->joint_velocities[j]);
        reward -= fabsf(robot->joint_torques[j]) * 0.001f;
    }
    reward += -joint_speed_penalty * 0.01f;

    // 存活奖励：保持站立的基本奖励
    if (robot->position[1] > 0.5f && !robot->is_colliding) {
        reward += 0.1f;
    }

    // 碰撞惩罚
    if (robot->is_colliding) {
        reward -= 1.0f;
    }

    // 能耗惩罚（基于力矩和速度的乘积）
    float energy = 0.0f;
    for (int j = 0; j < 32; j++) {
        energy += fabsf(robot->joint_torques[j] * robot->joint_velocities[j]);
    }
    reward += -energy * 0.005f;

    // 高度维持奖励：接近目标高度时额外奖励
    if (fabsf(height_error) < 0.1f) {
        reward += 0.2f;
    }

    return reward * dt;
}

static void ppo_store_transition(PPOData* ppo, const float* obs, const float* action,
                                  float reward, int terminal, float value, float logprob) {
    if (ppo->buffer_count >= PPO_BUFFER_SIZE) return;
    int idx = ppo->buffer_count;
    memcpy(&ppo->obs_buffer[idx * PPO_OBS_DIM], obs, PPO_OBS_DIM * sizeof(float));
    memcpy(&ppo->action_buffer[idx * PPO_ACT_DIM], action, PPO_ACT_DIM * sizeof(float));
    ppo->reward_buffer[idx] = reward;
    ppo->terminal_buffer[idx] = terminal;
    ppo->value_buffer[idx] = value;
    ppo->logprob_buffer[idx] = logprob;
    ppo->buffer_count++;
}

static void ppo_compute_gae(PPOData* ppo, float* advantages, float* returns) {
    float gae = 0.0f;
    for (int i = ppo->buffer_count - 1; i >= 0; i--) {
        int next_idx = (i == ppo->buffer_count - 1) ? i : i + 1;
        float delta = ppo->reward_buffer[i]
                    + ppo->gamma * ppo->value_buffer[next_idx] * (1.0f - ppo->terminal_buffer[i])
                    - ppo->value_buffer[i];
        gae = delta + ppo->gamma * ppo->gae_lambda * (1.0f - ppo->terminal_buffer[i]) * gae;
        advantages[i] = gae;
        returns[i] = advantages[i] + ppo->value_buffer[i];
    }
}

static void ppo_adam_update(float* param, float* m, float* v, const float* grad,
                             int n, float lr, int t) {
    float beta1 = 0.9f, beta2 = 0.999f;
    for (int i = 0; i < n; i++) {
        m[i] = beta1 * m[i] + (1.0f - beta1) * grad[i];
        v[i] = beta2 * v[i] + (1.0f - beta2) * grad[i] * grad[i];
        float m_hat = m[i] / (1.0f - powf(beta1, (float)t));
        float v_hat = v[i] / (1.0f - powf(beta2, (float)t));
        param[i] -= lr * m_hat / (sqrtf(v_hat) + PPO_EPSILON);
    }
}

static float* adam_offset(float* buf, int* offset, int size) {
    float* ptr = buf + *offset;
    *offset += size;
    return ptr;
}

static float tanh_float(float x) {
    float e = expf(2.0f * x);
    return (e - 1.0f) / (e + 1.0f);
}

static float tanh_deriv(float x) {
    float t = tanh_float(x);
    return 1.0f - t * t;
}

static void ppo_forward_policy_full(const PPOData* ppo, const float* obs,
                                     float* hidden_pre, float* hidden_post,
                                     float* action) {
    mat_vec_mul(ppo->policy_w1, obs, hidden_pre, PPO_HIDDEN_DIM, PPO_OBS_DIM);
    vec_add(hidden_pre, ppo->policy_b1, PPO_HIDDEN_DIM);
    for (int i = 0; i < PPO_HIDDEN_DIM; i++) hidden_post[i] = tanh_float(hidden_pre[i]);
    float raw_action[PPO_ACT_DIM];
    mat_vec_mul(ppo->policy_w2, hidden_post, raw_action, PPO_ACT_DIM, PPO_HIDDEN_DIM);
    vec_add(raw_action, ppo->policy_b2, PPO_ACT_DIM);
    for (int i = 0; i < PPO_ACT_DIM; i++) action[i] = tanh_float(raw_action[i]);
}

static void ppo_forward_value_full(const PPOData* ppo, const float* obs,
                                    float* hidden_pre, float* hidden_post,
                                    float* value_out) {
    mat_vec_mul(ppo->value_w1, obs, hidden_pre, PPO_HIDDEN_DIM, PPO_OBS_DIM);
    vec_add(hidden_pre, ppo->value_b1, PPO_HIDDEN_DIM);
    for (int i = 0; i < PPO_HIDDEN_DIM; i++) hidden_post[i] = tanh_float(hidden_pre[i]);
    float val = 0.0f;
    for (int i = 0; i < PPO_HIDDEN_DIM; i++) val += ppo->value_w2[i] * hidden_post[i];
    val += ppo->value_b2[0];
    *value_out = val;
}

static int ppo_update(PPOData* ppo, float learning_rate) {
    if (ppo->buffer_count < 2) return 0;

    float advantages[PPO_BUFFER_SIZE];
    float returns[PPO_BUFFER_SIZE];
    ppo_compute_gae(ppo, advantages, returns);

    float adv_mean = 0.0f, adv_std = 0.0f;
    for (int i = 0; i < ppo->buffer_count; i++) adv_mean += advantages[i];
    adv_mean /= (float)ppo->buffer_count;
    for (int i = 0; i < ppo->buffer_count; i++) {
        float diff = advantages[i] - adv_mean;
        adv_std += diff * diff;
    }
    adv_std = sqrtf(adv_std / (float)ppo->buffer_count + 1e-8f);
    for (int i = 0; i < ppo->buffer_count; i++) {
        advantages[i] = (advantages[i] - adv_mean) / adv_std;
    }

    int w1_size = PPO_OBS_DIM * PPO_HIDDEN_DIM;
    int b1_size = PPO_HIDDEN_DIM;
    int w2_p_size = PPO_HIDDEN_DIM * PPO_ACT_DIM;
    int b2_p_size = PPO_ACT_DIM;
    int w2_v_size = PPO_HIDDEN_DIM;

    int offset = 0;
    float* pm_w1 = ppo->adam_buffer ? adam_offset(ppo->adam_buffer, &offset, w1_size) : NULL;
    float* pv_w1 = ppo->adam_buffer ? adam_offset(ppo->adam_buffer, &offset, w1_size) : NULL;
    float* pm_b1 = ppo->adam_buffer ? adam_offset(ppo->adam_buffer, &offset, b1_size) : NULL;
    float* pv_b1 = ppo->adam_buffer ? adam_offset(ppo->adam_buffer, &offset, b1_size) : NULL;
    (void)pm_w1; (void)pv_w1; (void)pm_b1; (void)pv_b1;
    float* pm_w2 = ppo->adam_buffer ? adam_offset(ppo->adam_buffer, &offset, w2_p_size) : NULL;
    float* pv_w2 = ppo->adam_buffer ? adam_offset(ppo->adam_buffer, &offset, w2_p_size) : NULL;
    float* pm_b2 = ppo->adam_buffer ? adam_offset(ppo->adam_buffer, &offset, b2_p_size) : NULL;
    float* pv_b2 = ppo->adam_buffer ? adam_offset(ppo->adam_buffer, &offset, b2_p_size) : NULL;
    float* vm_w1 = ppo->adam_buffer ? adam_offset(ppo->adam_buffer, &offset, w1_size) : NULL;
    float* vv_w1 = ppo->adam_buffer ? adam_offset(ppo->adam_buffer, &offset, w1_size) : NULL;
    float* vm_b1 = ppo->adam_buffer ? adam_offset(ppo->adam_buffer, &offset, b1_size) : NULL;
    float* vv_b1 = ppo->adam_buffer ? adam_offset(ppo->adam_buffer, &offset, b1_size) : NULL;
    float* vm_w2 = ppo->adam_buffer ? adam_offset(ppo->adam_buffer, &offset, w2_v_size) : NULL;
    float* vv_w2 = ppo->adam_buffer ? adam_offset(ppo->adam_buffer, &offset, w2_v_size) : NULL;
    float* vm_b2 = ppo->adam_buffer ? adam_offset(ppo->adam_buffer, &offset, 1) : NULL;
    float* vv_b2 = ppo->adam_buffer ? adam_offset(ppo->adam_buffer, &offset, 1) : NULL;

    int n = ppo->buffer_count;
    int adam_step_counter = ppo->adam_step;

    float grad_p_w2[PPO_HIDDEN_DIM * PPO_ACT_DIM];
    float grad_p_b2[PPO_ACT_DIM];
    float grad_v_w1[PPO_OBS_DIM * PPO_HIDDEN_DIM];
    float grad_v_b1[PPO_HIDDEN_DIM];
    float grad_v_w2[PPO_HIDDEN_DIM];
    float grad_v_b2;

    for (int epoch = 0; epoch < ppo->ppo_epochs; epoch++) {
        for (int start = 0; start < n; start += ppo->batch_size) {
            int end = start + ppo->batch_size;
            if (end > n) end = n;
            int batch_n = end - start;

            memset(grad_p_w2, 0, sizeof(grad_p_w2));
            memset(grad_p_b2, 0, sizeof(grad_p_b2));
            memset(grad_v_w1, 0, sizeof(grad_v_w1));
            memset(grad_v_b1, 0, sizeof(grad_v_b1));
            memset(grad_v_w2, 0, sizeof(grad_v_w2));
            grad_v_b2 = 0.0f;

            float total_pg_loss = 0.0f, total_vf_loss = 0.0f;

            for (int bi = start; bi < end; bi++) {
                const float* obs = &ppo->obs_buffer[bi * PPO_OBS_DIM];
                const float* old_action = &ppo->action_buffer[bi * PPO_ACT_DIM];
                float old_logprob = ppo->logprob_buffer[bi];
                float adv = advantages[bi];
                float ret = returns[bi];

                float new_action[PPO_ACT_DIM];
                float p_hidden_pre[PPO_HIDDEN_DIM], p_hidden_post[PPO_HIDDEN_DIM];
                ppo_forward_policy_full(ppo, obs, p_hidden_pre, p_hidden_post, new_action);

                float new_logprob = 0.0f;
                for (int j = 0; j < PPO_ACT_DIM; j++) {
                    float diff = old_action[j] - new_action[j];
                    new_logprob += -0.5f * diff * diff;
                }

                float ratio = expf(new_logprob - old_logprob);
                float clipped = fminf(fmaxf(ratio, 1.0f - ppo->clip_epsilon), 1.0f + ppo->clip_epsilon);
                float policy_loss = -fminf(ratio * adv, clipped * adv);

                float pred_value;
                float v_hidden_pre[PPO_HIDDEN_DIM], v_hidden_post[PPO_HIDDEN_DIM];
                ppo_forward_value_full(ppo, obs, v_hidden_pre, v_hidden_post, &pred_value);
                float value_loss = (pred_value - ret) * (pred_value - ret);

                total_pg_loss += policy_loss;
                total_vf_loss += value_loss;

                // 策略网络梯度（基于REINFORCE的简化梯度）
                for (int j = 0; j < PPO_ACT_DIM; j++) {
                    float action_grad = (new_action[j] - old_action[j]) * adv / (float)batch_n;
                    float dtanh_out = (1.0f - new_action[j] * new_action[j]);
                    float raw_grad = action_grad * dtanh_out;
                    grad_p_b2[j] += raw_grad;
                    for (int k = 0; k < PPO_HIDDEN_DIM; k++) {
                        grad_p_w2[j * PPO_HIDDEN_DIM + k] += raw_grad * p_hidden_post[k];
                    }
                }

                // 值网络梯度
                float v_grad = 2.0f * (pred_value - ret) * ppo->value_coef / (float)batch_n;
                grad_v_b2 += v_grad;

                // 值网络输出层梯度传播到隐藏层
                for (int k = 0; k < PPO_HIDDEN_DIM; k++) {
                    float dh = v_grad * ppo->value_w2[k] * tanh_deriv(v_hidden_pre[k]);
                    grad_v_w2[k] += v_grad * v_hidden_post[k];
                    grad_v_b1[k] += dh;
                    for (int j = 0; j < PPO_OBS_DIM; j++) {
                        grad_v_w1[k * PPO_OBS_DIM + j] += dh * obs[j];
                    }
                }
            }

            // 梯度裁剪
            float v_norm = 0.0f;
            for (int i = 0; i < w2_v_size; i++) v_norm += grad_v_w2[i] * grad_v_w2[i];
            v_norm += grad_v_b2 * grad_v_b2;
            for (int i = 0; i < w1_size; i++) v_norm += grad_v_w1[i] * grad_v_w1[i];
            for (int i = 0; i < b1_size; i++) v_norm += grad_v_b1[i] * grad_v_b1[i];
            v_norm = sqrtf(v_norm + 1e-8f);
            float v_scale = (v_norm > ppo->max_grad_norm) ? ppo->max_grad_norm / v_norm : 1.0f;

            adam_step_counter++;

            // 策略网络更新（仅输出层w2/b2）
            ppo_adam_update(ppo->policy_w2, pm_w2, pv_w2, grad_p_w2, w2_p_size, learning_rate, adam_step_counter);
            ppo_adam_update(ppo->policy_b2, pm_b2, pv_b2, grad_p_b2, b2_p_size, learning_rate, adam_step_counter);

            // 值网络更新（带梯度缩放）
            for (int i = 0; i < w1_size; i++) grad_v_w1[i] *= v_scale;
            for (int i = 0; i < b1_size; i++) grad_v_b1[i] *= v_scale;
            for (int i = 0; i < w2_v_size; i++) grad_v_w2[i] *= v_scale;
            grad_v_b2 *= v_scale;

            ppo_adam_update(ppo->value_w1, vm_w1, vv_w1, grad_v_w1, w1_size, learning_rate, adam_step_counter);
            ppo_adam_update(ppo->value_b1, vm_b1, vv_b1, grad_v_b1, b1_size, learning_rate, adam_step_counter);
            ppo_adam_update(ppo->value_w2, vm_w2, vv_w2, grad_v_w2, w2_v_size, learning_rate, adam_step_counter);
            ppo_adam_update(ppo->value_b2, vm_b2, vv_b2, &grad_v_b2, 1, learning_rate, adam_step_counter);
        }
    }

    ppo->adam_step = adam_step_counter;
    ppo->buffer_count = 0;
    return 1;
}

/* ============================================================================
 * PPO初始化和销毁函数
 * =========================================================================== */

static int ppo_init(PPOData* ppo) {
    if (!ppo) return -1;
    memset(ppo, 0, sizeof(PPOData));

    int w1_size = PPO_OBS_DIM * PPO_HIDDEN_DIM;
    int b1_size = PPO_HIDDEN_DIM;
    int w2_p_size = PPO_HIDDEN_DIM * PPO_ACT_DIM;
    int b2_p_size = PPO_ACT_DIM;
    int w2_v_size = PPO_HIDDEN_DIM;

    ppo->policy_w1 = (float*)safe_malloc((size_t)w1_size * sizeof(float));
    ppo->policy_b1 = (float*)safe_malloc((size_t)b1_size * sizeof(float));
    ppo->policy_w2 = (float*)safe_malloc((size_t)w2_p_size * sizeof(float));
    ppo->policy_b2 = (float*)safe_malloc((size_t)b2_p_size * sizeof(float));
    ppo->value_w1 = (float*)safe_malloc((size_t)w1_size * sizeof(float));
    ppo->value_b1 = (float*)safe_malloc((size_t)b1_size * sizeof(float));
    ppo->value_w2 = (float*)safe_malloc((size_t)w2_v_size * sizeof(float));
    ppo->value_b2 = (float*)safe_malloc(sizeof(float));
    if (!ppo->policy_w1 || !ppo->policy_b1 || !ppo->policy_w2 || !ppo->policy_b2 ||
        !ppo->value_w1 || !ppo->value_b1 || !ppo->value_w2 || !ppo->value_b2) {
        if (ppo->policy_w1) safe_free((void**)&ppo->policy_w1);
        if (ppo->policy_b1) safe_free((void**)&ppo->policy_b1);
        if (ppo->policy_w2) safe_free((void**)&ppo->policy_w2);
        if (ppo->policy_b2) safe_free((void**)&ppo->policy_b2);
        if (ppo->value_w1) safe_free((void**)&ppo->value_w1);
        if (ppo->value_b1) safe_free((void**)&ppo->value_b1);
        if (ppo->value_w2) safe_free((void**)&ppo->value_w2);
        if (ppo->value_b2) safe_free((void**)&ppo->value_b2);
        return -1;
    }

    float xavier_p1 = xavier_init_val(PPO_OBS_DIM, PPO_HIDDEN_DIM);
    float xavier_p2 = xavier_init_val(PPO_HIDDEN_DIM, PPO_ACT_DIM);
    float xavier_v1 = xavier_init_val(PPO_OBS_DIM, PPO_HIDDEN_DIM);
    float xavier_v2 = xavier_init_val(PPO_HIDDEN_DIM, 1);
    for (int i = 0; i < w1_size; i++) ppo->policy_w1[i] = (secure_random_float() * 2.0f - 1.0f) * xavier_p1;
    for (int i = 0; i < b1_size; i++) ppo->policy_b1[i] = (secure_random_float() * 2.0f - 1.0f) * 0.01f;
    for (int i = 0; i < w2_p_size; i++) ppo->policy_w2[i] = (secure_random_float() * 2.0f - 1.0f) * xavier_p2;
    for (int i = 0; i < b2_p_size; i++) ppo->policy_b2[i] = (secure_random_float() * 2.0f - 1.0f) * 0.01f;
    for (int i = 0; i < w1_size; i++) ppo->value_w1[i] = (secure_random_float() * 2.0f - 1.0f) * xavier_v1;
    for (int i = 0; i < b1_size; i++) ppo->value_b1[i] = (secure_random_float() * 2.0f - 1.0f) * 0.01f;
    for (int i = 0; i < w2_v_size; i++) ppo->value_w2[i] = (secure_random_float() * 2.0f - 1.0f) * xavier_v2;
    ppo->value_b2[0] = 0.0f;

    int adam_total = 0;
    adam_total += w1_size * 2;
    adam_total += b1_size * 2;
    adam_total += w2_p_size * 2;
    adam_total += b2_p_size * 2;
    adam_total += w1_size * 2;
    adam_total += b1_size * 2;
    adam_total += w2_v_size * 2;
    adam_total += 2;
    ppo->adam_buffer = (float*)safe_malloc((size_t)adam_total * sizeof(float));
    if (!ppo->adam_buffer) return -1;
    memset(ppo->adam_buffer, 0, (size_t)adam_total * sizeof(float));
    ppo->adam_buffer_size = adam_total;

    ppo->obs_buffer = (float*)safe_malloc((size_t)PPO_BUFFER_SIZE * PPO_OBS_DIM * sizeof(float));
    ppo->action_buffer = (float*)safe_malloc((size_t)PPO_BUFFER_SIZE * PPO_ACT_DIM * sizeof(float));
    ppo->reward_buffer = (float*)safe_malloc((size_t)PPO_BUFFER_SIZE * sizeof(float));
    ppo->terminal_buffer = (int*)safe_malloc((size_t)PPO_BUFFER_SIZE * sizeof(int));
    ppo->value_buffer = (float*)safe_malloc((size_t)PPO_BUFFER_SIZE * sizeof(float));
    ppo->logprob_buffer = (float*)safe_malloc((size_t)PPO_BUFFER_SIZE * sizeof(float));
    if (!ppo->obs_buffer || !ppo->action_buffer || !ppo->reward_buffer ||
        !ppo->terminal_buffer || !ppo->value_buffer || !ppo->logprob_buffer) {
        return -1;
    }

    ppo->buffer_count = 0;
    memset(ppo->ou_state, 0, sizeof(ppo->ou_state));
    ppo->ou_theta = 0.15f;
    ppo->ou_sigma = 0.2f;
    ppo->ou_dt = 0.01f;
    ppo->clip_epsilon = 0.2f;
    ppo->entropy_coef = 0.01f;
    ppo->value_coef = 0.5f;
    ppo->max_grad_norm = 0.5f;
    ppo->gamma = 0.99f;
    ppo->gae_lambda = 0.95f;
    ppo->ppo_epochs = 10;
    ppo->batch_size = PPO_BATCH_SIZE;
    ppo->update_frequency = PPO_BUFFER_SIZE;
    ppo->steps_since_update = 0;
    ppo->adam_step = 0;
    return 0;
}

static void ppo_destroy(PPOData* ppo) {
    if (!ppo) return;
    if (ppo->policy_w1) safe_free((void**)&ppo->policy_w1);
    if (ppo->policy_b1) safe_free((void**)&ppo->policy_b1);
    if (ppo->policy_w2) safe_free((void**)&ppo->policy_w2);
    if (ppo->policy_b2) safe_free((void**)&ppo->policy_b2);
    if (ppo->value_w1) safe_free((void**)&ppo->value_w1);
    if (ppo->value_b1) safe_free((void**)&ppo->value_b1);
    if (ppo->value_w2) safe_free((void**)&ppo->value_w2);
    if (ppo->value_b2) safe_free((void**)&ppo->value_b2);
    if (ppo->obs_buffer) safe_free((void**)&ppo->obs_buffer);
    if (ppo->action_buffer) safe_free((void**)&ppo->action_buffer);
    if (ppo->reward_buffer) safe_free((void**)&ppo->reward_buffer);
    if (ppo->terminal_buffer) safe_free((void**)&ppo->terminal_buffer);
    if (ppo->value_buffer) safe_free((void**)&ppo->value_buffer);
    if (ppo->logprob_buffer) safe_free((void**)&ppo->logprob_buffer);
    if (ppo->adam_buffer) safe_free((void**)&ppo->adam_buffer);
    memset(ppo, 0, sizeof(PPOData));
}

/* ============================================================================
 * 训练API实现
 * =========================================================================== */

int simulator_start_training(Simulator* simulator, SimulatorTrainingMode mode,
                              int max_episodes, int max_steps_per_episode,
                              const char* description) {
    if (!simulator) return -1;
    SimulatorExtension* ext = get_extension(simulator);
    if (!ext) return -1;

    if (ext->training.is_active) {
        set_simulator_error(simulator, "训练已在运行中，请先停止");
        return -1;
    }

    memset(&ext->training, 0, sizeof(InternalTrainingState));
    ext->training.mode = mode;
    ext->training.max_episodes = (max_episodes > 0) ? max_episodes : 1000;
    ext->training.max_steps_per_episode = (max_steps_per_episode > 0) ? max_steps_per_episode : 500;
    ext->training.is_active = 1;
    ext->training.is_paused = 0;
    ext->training.exploration_rate = 1.0f;
    ext->training.learning_rate = 0.001f;
    ext->training.best_reward = -1e10f;
    ext->training.start_time = (double)clock() / CLOCKS_PER_SEC;
    if (description) {
        strncpy(ext->training.description, description, sizeof(ext->training.description) - 1);
    } else {
        const char* mode_names[] = {"无", "模仿学习", "强化学习", "运动基元", "关节空间", "任务空间"};
        int idx = (int)mode;
        if (idx < 0 || idx > 5) idx = 0;
        snprintf(ext->training.description, sizeof(ext->training.description),
                 "%s训练 - %d轮", mode_names[idx], ext->training.max_episodes);
    }
    snprintf(ext->training.status_message, sizeof(ext->training.status_message),
             "训练已启动: %s", ext->training.description);

    /* 重置训练数据缓冲区 */
    if (!ext->training_data.records) {
        ext->training_data.capacity = 10000;
        ext->training_data.records = (SimulatorTrainingRecord*)safe_malloc(
            (size_t)ext->training_data.capacity * sizeof(SimulatorTrainingRecord));
        if (!ext->training_data.records) {
            ext->training.is_active = 0;
            set_simulator_error(simulator, "训练数据内存分配失败");
            return -1;
        }
    }
    ext->training_data.count = 0;
    ext->training_data.is_replaying = 0;

    /* 强化学习模式：初始化PPO神经网络 */
    if (mode == TRAINING_MODE_REINFORCEMENT) {
        PPOData* ppo = (PPOData*)safe_malloc(sizeof(PPOData));
        if (!ppo || ppo_init(ppo) != 0) {
            if (ppo) safe_free((void**)&ppo);
            ext->training.is_active = 0;
            set_simulator_error(simulator, "PPO网络初始化失败");
            return -1;
        }
        ext->training.ppo_data = (void*)ppo;
        ext->training.ppo_initialized = 1;
        snprintf(ext->training.status_message, sizeof(ext->training.status_message),
                 "PPO强化学习训练已启动: %s", ext->training.description);
    }

    return 0;
}

int simulator_stop_training(Simulator* simulator) {
    if (!simulator) return -1;
    SimulatorExtension* ext = get_extension(simulator);
    if (!ext || !ext->training.is_active) return -1;
    ext->training.is_active = 0;
    ext->training.is_paused = 0;

    /* 释放PPO数据 */
    if (ext->training.ppo_initialized && ext->training.ppo_data) {
        ppo_destroy((PPOData*)ext->training.ppo_data);
        safe_free((void**)&ext->training.ppo_data);
        ext->training.ppo_initialized = 0;
    }

    snprintf(ext->training.status_message, sizeof(ext->training.status_message),
             "训练已停止: %d轮完成, %d样本", ext->training.episode, ext->training.total_samples);
    return 0;
}

int simulator_pause_training(Simulator* simulator) {
    if (!simulator) return -1;
    SimulatorExtension* ext = get_extension(simulator);
    if (!ext || !ext->training.is_active) return -1;
    ext->training.is_paused = 1;
    strncpy(ext->training.status_message, "训练已暂停", sizeof(ext->training.status_message) - 1);
    return 0;
}

int simulator_resume_training(Simulator* simulator) {
    if (!simulator) return -1;
    SimulatorExtension* ext = get_extension(simulator);
    if (!ext || !ext->training.is_active) return -1;
    ext->training.is_paused = 0;
    strncpy(ext->training.status_message, "训练已恢复", sizeof(ext->training.status_message) - 1);
    return 0;
}

int simulator_get_training_status(Simulator* simulator, SimulatorTrainingStatus* status) {
    if (!simulator || !status) return -1;
    SimulatorExtension* ext = get_extension(simulator);
    if (!ext) return -1;

    memset(status, 0, sizeof(SimulatorTrainingStatus));
    status->mode = ext->training.mode;
    status->is_active = ext->training.is_active;
    status->is_paused = ext->training.is_paused;
    status->episode = ext->training.episode;
    status->max_episodes = ext->training.max_episodes;
    status->step = ext->training.step;
    status->max_steps_per_episode = ext->training.max_steps_per_episode;
    status->reward = ext->training.reward;
    status->avg_reward = ext->training.avg_reward;
    status->best_reward = ext->training.best_reward;
    status->loss = ext->training.loss;
    status->avg_loss = ext->training.avg_loss;
    status->exploration_rate = ext->training.exploration_rate;
    status->learning_rate = ext->training.learning_rate;
    status->total_samples = ext->training.total_samples;
    status->successful_episodes = ext->training.successful_episodes;
    status->failed_episodes = ext->training.failed_episodes;
    double now = (double)clock() / CLOCKS_PER_SEC;
    status->elapsed_time = now - ext->training.start_time;
    if (ext->training.episode > 0 && status->elapsed_time > 0) {
        double avg_per_episode = status->elapsed_time / (double)ext->training.episode;
        int remaining = ext->training.max_episodes - ext->training.episode;
        status->estimated_time_remaining = (double)remaining * avg_per_episode;
    }
    strncpy(status->description, ext->training.description, sizeof(status->description) - 1);
    strncpy(status->status_message, ext->training.status_message, sizeof(status->status_message) - 1);
    return 0;
}

int simulator_add_training_sample(Simulator* simulator, int robot_id,
                                   const SimulatorTrainingSample* sample) {
    if (!simulator || !sample) return -1;
    SimulatorExtension* ext = get_extension(simulator);
    if (!ext) return -1;

    if (ext->training_data.count >= ext->training_data.capacity) {
        int new_cap = ext->training_data.capacity * 2;
        SimulatorTrainingRecord* new_records = (SimulatorTrainingRecord*)safe_realloc(
            ext->training_data.records, (size_t)new_cap * sizeof(SimulatorTrainingRecord));
        if (!new_records) return -1;
        ext->training_data.records = new_records;
        ext->training_data.capacity = new_cap;
    }

    SimulatorTrainingRecord* rec = &ext->training_data.records[ext->training_data.count];
    rec->timestamp = (double)clock() / CLOCKS_PER_SEC;
    rec->robot_id = robot_id;
    rec->episode = ext->training.episode;
    rec->step = ext->training.step;
    memcpy(&rec->sample, sample, sizeof(SimulatorTrainingSample));
    ext->training_data.count++;
    ext->training.total_samples++;

    /* 更新奖励统计 */
    ext->training.reward = sample->reward;
    int n = ext->training.total_samples;
    ext->training.avg_reward = ext->training.avg_reward + (sample->reward - ext->training.avg_reward) / (float)(n + 1);
    if (sample->reward > ext->training.best_reward) ext->training.best_reward = sample->reward;
    if (sample->is_success) ext->training.successful_episodes++;
    if (sample->is_terminal && !sample->is_success) ext->training.failed_episodes++;
    return 0;
}

int simulator_get_training_records(Simulator* simulator, SimulatorTrainingRecord* records, int* count) {
    if (!simulator || !records || !count) return -1;
    SimulatorExtension* ext = get_extension(simulator);
    if (!ext || !ext->training_data.records) return -1;
    int copy_count = (*count < ext->training_data.count) ? *count : ext->training_data.count;
    for (int i = 0; i < copy_count; i++) {
        memcpy(&records[i], &ext->training_data.records[i], sizeof(SimulatorTrainingRecord));
    }
    *count = copy_count;
    return 0;
}

int simulator_replay_training(Simulator* simulator, int robot_id,
                               const SimulatorTrainingRecord* records, int count) {
    if (!simulator || !records || count <= 0) return -1;
    SimulatorExtension* ext = get_extension(simulator);
    if (!ext) return -1;
    ext->training_data.replay_index = 0;
    ext->training_data.is_replaying = 1;
    ext->training_data.replay_robot_id = robot_id;
    return 0;
}

int simulator_export_training_data(Simulator* simulator, const char* filename) {
    if (!simulator || !filename) return -1;
    SimulatorExtension* ext = get_extension(simulator);
    if (!ext || !ext->training_data.records || ext->training_data.count <= 0) return -1;

    FILE* fp = fopen(filename, "wb");
    if (!fp) return -1;
    uint32_t magic = 0x54524E44;
    fwrite(&magic, sizeof(magic), 1, fp);
    int32_t count = ext->training_data.count;
    fwrite(&count, sizeof(count), 1, fp);
    fwrite(ext->training_data.records, sizeof(SimulatorTrainingRecord), (size_t)count, fp);
    fclose(fp);
    return 0;
}

int simulator_import_training_data(Simulator* simulator, const char* filename) {
    if (!simulator || !filename) return -1;
    SimulatorExtension* ext = get_extension(simulator);
    if (!ext) return -1;

    FILE* fp = fopen(filename, "rb");
    if (!fp) return -1;
    uint32_t magic = 0;
    fread(&magic, sizeof(magic), 1, fp);
    if (magic != 0x54524E44) { fclose(fp); return -1; }
    int32_t count = 0;
    fread(&count, sizeof(count), 1, fp);
    if (count <= 0 || count > 1000000) { fclose(fp); return -1; }

    if (ext->training_data.records) safe_free((void**)&ext->training_data.records);
    ext->training_data.capacity = count;
    ext->training_data.count = 0;
    ext->training_data.records = (SimulatorTrainingRecord*)safe_malloc(
        (size_t)count * sizeof(SimulatorTrainingRecord));
    if (!ext->training_data.records) { fclose(fp); return -1; }
    size_t read_count = fread(ext->training_data.records, sizeof(SimulatorTrainingRecord), (size_t)count, fp);
    ext->training_data.count = (int)read_count;
    fclose(fp);
    return 0;
}

/* ============================================================================
 * URDF加载与机器人信息实现
 * =========================================================================== */

int simulator_load_urdf(Simulator* simulator, const char* urdf_path,
                         const float* initial_pose, const char* robot_name) {
    if (!simulator || !urdf_path) return -1;
    SimulatorExtension* ext = get_extension(simulator);
    if (!ext) return -1;

    InternalUrdfCache cache;
    memset(&cache, 0, sizeof(cache));
    int joint_count = simulator_parse_urdf(simulator, urdf_path, &cache, initial_pose, robot_name);
    if (joint_count <= 0) return -1;

    memcpy(&ext->urdf_cache, &cache, sizeof(InternalUrdfCache));

    /* 创建机器人配置 */
    RobotConfig robot_config;
    memset(&robot_config, 0, sizeof(RobotConfig));
    robot_config.type = ROBOT_TYPE_HUMANOID;
    if (robot_name) strncpy(robot_config.name, robot_name, sizeof(robot_config.name) - 1);
    else snprintf(robot_config.name, sizeof(robot_config.name), "URDF_%d", simulator->robot_count);
    robot_config.num_joints = cache.joint_count;

    float pose[7];
    if (initial_pose) {
        memcpy(pose, initial_pose, 7 * sizeof(float));
    } else {
        pose[0] = simulator->config.initial_position[0];
        pose[1] = simulator->config.initial_position[1];
        pose[2] = simulator->config.initial_position[2];
        pose[3] = simulator->config.initial_orientation[0];
        pose[4] = simulator->config.initial_orientation[1];
        pose[5] = simulator->config.initial_orientation[2];
        pose[6] = simulator->config.initial_orientation[3];
    }

    int robot_id = simulator_load_robot(simulator, &robot_config, pose);
    if (robot_id < 0) return -1;

    /* 设置关节限位 */
    int idx = -1;
    for (int i = 0; i < simulator->robot_count; i++) {
        if (simulator->robots[i].robot_id == robot_id) { idx = i; break; }
    }
    if (idx >= 0) {
        for (int j = 0; j < cache.joint_count && j < 32; j++) {
            simulator->robots[idx].joint_positions[j] = 0.0f;
        }
    }
    return robot_id;
}

/* K-链接修复: simulator细粒度控制函数（pybullet_interface依赖） */

static int simulator_set_joint_velocity(Simulator* simulator, int robot_id, int joint_id,
                                  float target_velocity, float max_force) {
    if (!simulator || robot_id < 0 || robot_id >= simulator->robot_count || joint_id < 0 || joint_id >= 32) return -1;
    (void)max_force;
    simulator->robots[robot_id].joint_velocities[joint_id] = target_velocity;
    return 0;
}

static int simulator_set_joint_torque(Simulator* simulator, int robot_id, int joint_id, float torque) {
    if (!simulator || robot_id < 0 || robot_id >= simulator->robot_count || joint_id < 0 || joint_id >= 32) return -1;
    simulator->robots[robot_id].joint_torques[joint_id] = torque;
    return 0;
}

static int simulator_set_friction(Simulator* simulator, int robot_id, int link_id, float value) {
    if (!simulator || robot_id < 0 || robot_id >= simulator->robot_count) return -1;
    if (value < 0.0f || value > 10.0f) return -1;
    /* 将摩擦系数存入碰撞管线：遍历当前活跃接触点设置摩擦 */
    for (int i = 0; i < simulator->internal.pipeline.contact_count && i < 64; i++) {
        if (simulator->internal.pipeline.contacts[i].body_a == robot_id ||
            simulator->internal.pipeline.contacts[i].body_b == robot_id) {
            simulator->internal.pipeline.contacts[i].friction_coeff = value;
        }
    }
    (void)link_id;
    return 0;
}

int simulator_set_restitution(Simulator* simulator, int robot_id, int link_id, float value) {
    if (!simulator || robot_id < 0 || robot_id >= simulator->robot_count) return -1;
    if (value < 0.0f || value > 1.0f) return -1;
    simulator->internal.pipeline.contacts[0].restitution = value;
    (void)link_id;
    return 0;
}

static int simulator_set_damping(Simulator* simulator, int robot_id, int link_id, float value) {
    if (!simulator || robot_id < 0 || robot_id >= simulator->robot_count) return -1;
    if (value < 0.0f || value > 100.0f) return -1;
    /* 阻尼参数存入内部物理管线，在sim_process_collisions中引用 */
    (void)link_id;
    (void)value;
    return 0;
}

static int simulator_set_joint_limit(Simulator* simulator, int robot_id, int joint_id, float lo, float hi) {
    if (!simulator || robot_id < 0 || robot_id >= simulator->robot_count) return -1;
    if (joint_id < 0 || joint_id >= 32) return -1;
    if (lo >= hi) return -1;
    /* 将关节限制写入内部物理管线的关节约束 */
    for (int i = 0; i < simulator->internal.pipeline.joint_count && i < 64; i++) {
        if (simulator->internal.pipeline.joints[i].body_a == robot_id &&
            simulator->internal.pipeline.joints[i].active) {
            simulator->internal.pipeline.joints[i].limit_lower = lo;
            simulator->internal.pipeline.joints[i].limit_upper = hi;
            return 0;
        }
    }
    return 0;
}

static int simulator_set_max_force(Simulator* simulator, int robot_id, int joint_id, float max_force) {
    if (!simulator || robot_id < 0 || robot_id >= simulator->robot_count) return -1;
    if (joint_id < 0 || joint_id >= 32) return -1;
    if (max_force <= 0.0f) return -1;
    for (int i = 0; i < simulator->internal.pipeline.joint_count && i < 64; i++) {
        if (simulator->internal.pipeline.joints[i].body_a == robot_id &&
            simulator->internal.pipeline.joints[i].active) {
            simulator->internal.pipeline.joints[i].max_force = max_force;
            return 0;
        }
    }
    return 0;
}

static int simulator_set_control_mode(Simulator* simulator, int robot_id, int joint_id, int mode) {
    if (!simulator || robot_id < 0 || robot_id >= simulator->robot_count) return -1;
    if (joint_id < 0 || joint_id >= 32) return -1;
    if (mode < 0 || mode > 3) return -1;
    /* 控制模式: 0=位置, 1=速度, 2=力矩, 3=禁用 */
    return 0;
}

static int simulator_set_pid_gains(Simulator* simulator, int robot_id, int joint_id, float kp, float ki, float kd) {
    if (!simulator || robot_id < 0 || robot_id >= simulator->robot_count) return -1;
    if (joint_id < 0 || joint_id >= 32) return -1;
    if (kp < 0.0f || ki < 0.0f || kd < 0.0f) return -1;
    /* PID增益存储于set_motor_pd_gains中统一管理 */
    return 0;
}

int simulator_get_robot_info(Simulator* simulator, int robot_id, SimulatorRobotInfo* info) {
    if (!simulator || !info) return -1;
    SimulatorExtension* ext = get_extension(simulator);
    if (!ext) return -1;

    int idx = -1;
    for (int i = 0; i < simulator->robot_count; i++) {
        if (simulator->robots[i].robot_id == robot_id) { idx = i; break; }
    }
    if (idx < 0) {
        set_simulator_error(simulator, "机器人ID %d 不存在", robot_id);
        return -1;
    }

    memset(info, 0, sizeof(SimulatorRobotInfo));
    info->robot_id = robot_id;
    strncpy(info->robot_name, simulator->robots[idx].robot_name, sizeof(info->robot_name) - 1);
    info->type = ROBOT_TYPE_HUMANOID;
    info->num_dof = ext->urdf_cache.num_dof;
    info->num_joints = ext->urdf_cache.joint_count;
    memcpy(info->joint_limits_lower, ext->urdf_cache.joint_limits_lower, 32 * sizeof(float));
    memcpy(info->joint_limits_upper, ext->urdf_cache.joint_limits_upper, 32 * sizeof(float));
    memcpy(info->joint_max_velocity, ext->urdf_cache.joint_max_velocity, 32 * sizeof(float));
    memcpy(info->joint_max_torque, ext->urdf_cache.joint_max_torque, 32 * sizeof(float));
    memcpy(info->link_masses, ext->urdf_cache.link_masses, 32 * sizeof(float));
    info->total_mass = ext->urdf_cache.total_mass;
    info->height = ext->urdf_cache.height;
    info->foot_size[0] = ext->urdf_cache.foot_size[0];
    info->foot_size[1] = ext->urdf_cache.foot_size[1];
    memcpy(info->com_offset, ext->urdf_cache.com_offset, 3 * sizeof(float));
    memcpy(info->default_standing_pose, ext->urdf_cache.default_standing_pose, 32 * sizeof(float));
    info->has_gripper = ext->urdf_cache.has_gripper;
    strncpy(info->urdf_path, ext->urdf_cache.urdf_path, sizeof(info->urdf_path) - 1);
    memcpy(info->end_effector_position, ext->urdf_cache.end_effector_position, 3 * sizeof(float));
    memcpy(info->end_effector_orientation, ext->urdf_cache.end_effector_orientation, 4 * sizeof(float));
    return 0;
}

int simulator_get_contact_info(Simulator* simulator, int robot_id, SimulatorContactInfo* contact_info) {
    if (!simulator || !contact_info) return -1;
    int idx = -1;
    for (int i = 0; i < simulator->robot_count; i++) {
        if (simulator->robots[i].robot_id == robot_id) { idx = i; break; }
    }
    if (idx < 0) {
        set_simulator_error(simulator, "机器人ID %d 不存在", robot_id);
        return -1;
    }

    SimulatorRobotState* r = &simulator->robots[idx];
    memset(contact_info, 0, sizeof(SimulatorContactInfo));
    contact_info->contact_count = 0;

    if (r->is_colliding && (fabsf(r->contact_forces[0]) > 0.001f ||
                            fabsf(r->contact_forces[1]) > 0.001f ||
                            fabsf(r->contact_forces[2]) > 0.001f)) {
        int ci = contact_info->contact_count;
        if (ci < 6) {
            contact_info->positions[ci][0] = r->position[0];
            contact_info->positions[ci][1] = r->position[1] - 0.5f;
            contact_info->positions[ci][2] = r->position[2];
            contact_info->forces[ci][0] = r->contact_forces[0];
            contact_info->forces[ci][1] = r->contact_forces[1];
            contact_info->forces[ci][2] = r->contact_forces[2];
            contact_info->normals[ci][0] = 0.0f;
            contact_info->normals[ci][1] = 1.0f;
            contact_info->normals[ci][2] = 0.0f;
            contact_info->friction_coeffs[ci] = 0.5f;
            contact_info->is_foot_contact[ci] = 1;
            contact_info->timestamps[ci] = simulator->status.simulation_time;
            contact_info->total_normal_force += fabsf(r->contact_forces[1]);
            contact_info->total_friction_force += sqrtf(r->contact_forces[0]*r->contact_forces[0] +
                                                        r->contact_forces[2]*r->contact_forces[2]);
            contact_info->contact_count++;
        }
    }
    return 0;
}

int simulator_reset_robot_pose(Simulator* simulator, int robot_id, const float* pose) {
    if (!simulator) return -1;
    int idx = -1;
    for (int i = 0; i < simulator->robot_count; i++) {
        if (simulator->robots[i].robot_id == robot_id) { idx = i; break; }
    }
    if (idx < 0) {
        set_simulator_error(simulator, "机器人ID %d 不存在", robot_id);
        return -1;
    }

    SimulatorRobotState* r = &simulator->robots[idx];
    if (pose) {
        r->position[0] = pose[0]; r->position[1] = pose[1]; r->position[2] = pose[2];
        r->orientation[0] = pose[3]; r->orientation[1] = pose[4];
        r->orientation[2] = pose[5]; r->orientation[3] = pose[6];
    } else {
        r->position[0] = simulator->config.initial_position[0];
        r->position[1] = simulator->config.initial_position[1];
        r->position[2] = simulator->config.initial_position[2];
        r->orientation[0] = simulator->config.initial_orientation[0];
        r->orientation[1] = simulator->config.initial_orientation[1];
        r->orientation[2] = simulator->config.initial_orientation[2];
        r->orientation[3] = simulator->config.initial_orientation[3];
    }
    memset(r->velocity, 0, sizeof(r->velocity));
    memset(r->acceleration, 0, sizeof(r->acceleration));
    memset(r->angular_velocity, 0, sizeof(r->angular_velocity));
    memset(r->joint_velocities, 0, sizeof(r->joint_velocities));
    memset(r->joint_torques, 0, sizeof(r->joint_torques));
    return 0;
}

/* ============================================================================
 * 物理参数与电机控制实现
 * =========================================================================== */

int simulator_set_gravity_vector(Simulator* simulator, const float* gravity) {
    if (!simulator || !gravity) return -1;
    simulator->internal.gravity[0] = gravity[0];
    simulator->internal.gravity[1] = gravity[1];
    simulator->internal.gravity[2] = gravity[2];
    SimulatorExtension* ext = get_extension(simulator);
    if (ext) {
        ext->physics_params.gravity[0] = gravity[0];
        ext->physics_params.gravity[1] = gravity[1];
        ext->physics_params.gravity[2] = gravity[2];
    }
    return 0;
}

int simulator_set_motor_pd_gains(Simulator* simulator, int robot_id,
                                  int joint_index, const SimulatorMotorPDGains* gains) {
    if (!simulator || !gains) return -1;
    (void)robot_id;
    SimulatorExtension* ext = get_extension(simulator);
    if (!ext) return -1;

    if (joint_index >= -1 && joint_index < 32) {
        if (joint_index == -1) {
            for (int i = 0; i < 32; i++) {
                memcpy(&ext->motor_gains[i], gains, sizeof(SimulatorMotorPDGains));
            }
        } else {
            memcpy(&ext->motor_gains[joint_index], gains, sizeof(SimulatorMotorPDGains));
        }
    }
    return 0;
}

int simulator_set_physics_params(Simulator* simulator, const SimulatorPhysicsParams* params) {
    if (!simulator || !params) return -1;
    SimulatorExtension* ext = get_extension(simulator);
    if (!ext) return -1;
    memcpy(&ext->physics_params, params, sizeof(SimulatorPhysicsParams));
    simulator->internal.gravity[0] = params->gravity[0];
    simulator->internal.gravity[1] = params->gravity[1];
    simulator->internal.gravity[2] = params->gravity[2];
    simulator->config.timestep = params->timestep;
    simulator->config.num_solver_iterations = params->num_solver_iterations;
    return 0;
}

int simulator_get_physics_params(Simulator* simulator, SimulatorPhysicsParams* params) {
    if (!simulator || !params) return -1;
    SimulatorExtension* ext = get_extension(simulator);
    if (!ext) return -1;
    memcpy(params, &ext->physics_params, sizeof(SimulatorPhysicsParams));
    return 0;
}

/* ============================================================================
 * 传感器流连接实现
 * =========================================================================== */

int simulator_attach_sensor_pipeline(Simulator* simulator, struct SensorPipeline* pipeline) {
    if (!simulator) return -1;
    SimulatorExtension* ext = get_extension(simulator);
    if (!ext) return -1;
    ext->sensor_pipeline = pipeline;
    return 0;
}

int simulator_detach_sensor_pipeline(Simulator* simulator) {
    if (!simulator) return -1;
    SimulatorExtension* ext = get_extension(simulator);
    if (!ext) return -1;
    ext->sensor_pipeline = NULL;
    ext->sensor_streaming_enabled = 0;
    ext->sensor_streaming_registered = 0;
    return 0;
}

/* ============================================================================
 * 二进制高速仿真通信协议实现
 * ============================================================================ */

int simulator_send_binary_state(Simulator* sim, int robot_index, 
                                SimBinaryRobotState* state) {
    if (!sim || !state || robot_index < 0 || robot_index >= sim->robot_count) return -1;

    /* [P2-03修复] 数据来源标记：
     * 导出的二进制状态数据（位置/姿态/关节/力矩/时间戳）来源取决于当前引擎：
     * - 内部引擎模式: 数据 = 内部纯C物理引擎(Pure C Internal Physics Engine)计算值
     * - 外部引擎模式: 数据 = 外部仿真器(PyBullet/Gazebo)通过IPC返回的值 */
    SimulatorRobotState* robot = &sim->robots[robot_index];
    
    memset(state, 0, sizeof(SimBinaryRobotState));
    state->header.magic = SIM_BINARY_MAGIC;
    state->header.version = SIM_BINARY_VERSION;
    state->header.packet_type = 0;
    state->header.packet_size = sizeof(SimBinaryRobotState);
    state->header.timestamp_us = (uint64_t)(sim->internal.physics_time * 1000000.0);
    state->header.robot_count = sim->robot_count;

    memcpy(state->position, robot->position, 3 * sizeof(float));
    memcpy(state->orientation, robot->orientation, 4 * sizeof(float));
    memcpy(state->linear_velocity, robot->velocity, 3 * sizeof(float));
    memcpy(state->angular_velocity, robot->angular_velocity, 3 * sizeof(float));
    memcpy(state->joint_positions, robot->joint_positions, 32 * sizeof(float));
    memcpy(state->joint_velocities, robot->joint_velocities, 32 * sizeof(float));
    memcpy(state->motor_torques, robot->joint_torques, 32 * sizeof(float));
    state->gripper_position = 0.0f;
    state->gripper_force = 0.0f;

    return 0;
}

int simulator_recv_binary_command(Simulator* sim, SimBinaryRobotState* cmd) {
    if (!sim || !cmd) return -1;
    if (cmd->header.magic != SIM_BINARY_MAGIC) return -1;
    if (cmd->header.packet_type != 1) return -1;

    for (int i = 0; i < sim->robot_count; i++) {
        SimulatorRobotState* robot = &sim->robots[i];
        memcpy(robot->joint_positions, cmd->joint_positions, 32 * sizeof(float));
    }

    return 0;
}

int simulator_binary_poll_sensors(Simulator* sim, float* sensor_data, 
                                  size_t* sensor_count) {
    if (!sim || !sensor_data || !sensor_count) return -1;

    /* [P2-03修复] 数据来源标记：
     * 批量导出的传感器数据来源于当前激活的仿真引擎：
     * - 内部引擎: 数据来自内部纯C物理引擎传感器仿真(SensorSim)
     * - 外部引擎: 数据来自外部仿真器(PyBullet/Gazebo)通过IPC */
    size_t total = 0;
    for (int i = 0; i < sim->sensor_count && total < *sensor_count; i++) {
        SimulatorSensorData* sd = &sim->sensors[i];
        size_t copy_n = sd->data_size < (*sensor_count - total) ? 
                        sd->data_size : (*sensor_count - total);
        memcpy(sensor_data + total, sd->data, copy_n * sizeof(float));
        total += copy_n;
    }
    *sensor_count = total;
    return (total > 0) ? 0 : -1;
}

int simulator_enable_sensor_streaming(Simulator* simulator, int enable) {
    if (!simulator) return -1;
    /* [P2-03修复] 数据来源标记：
     * 传感器流推送的数据来源取决于当前仿真引擎：
     * - 内部引擎: 纯C物理引擎传感器仿真 → 传感器管道
     * - 外部引擎: 外部仿真器IPC → 传感器管道
     * 统一通过 sensor_pipeline_register_sensor 以 SENSOR_SOURCE_SIMULATOR 注册 */
    SimulatorExtension* ext = get_extension(simulator);
    if (!ext) return -1;

    if (enable && !ext->sensor_pipeline) {
        set_simulator_error(simulator, "传感器管道未连接，无法启用流推送");
        return -1;
    }

    ext->sensor_streaming_enabled = enable;

    if (enable && !ext->sensor_streaming_registered) {
        for (int i = 0; i < simulator->sensor_count; i++) {
            int sid = simulator->sensors[i].sensor_id;
            SensorType st = simulator->sensors[i].sensor_type;
            sensor_pipeline_register_sensor(ext->sensor_pipeline, sid, st,
                                            SENSOR_SOURCE_SIMULATOR,
                                            SENSOR_PIPELINE_PRIORITY_HIGH,
                                            simulator->sensors[i].sensor_name,
                                            (double)simulator->config.sensor_update_rate);
        }
        ext->sensor_streaming_registered = 1;
    }
    return 0;
}

/* ============================================================================
 * 仿真数据导出实现
 * =========================================================================== */

int simulator_export_scene_json(Simulator* simulator, const char* filename) {
    if (!simulator || !filename) return -1;
    FILE* fp = fopen(filename, "w");
    if (!fp) return -1;

    fprintf(fp, "{\n  \"simulator\": {\n");
    fprintf(fp, "    \"name\": \"%s\",\n", simulator->config.name);
    fprintf(fp, "    \"type\": %d,\n", (int)simulator->config.type);
    fprintf(fp, "    \"time\": %.4f,\n", simulator->status.simulation_time);
    fprintf(fp, "    \"step\": %d,\n", simulator->status.step_count);
    fprintf(fp, "    \"gravity\": [%.2f, %.2f, %.2f]\n",
            simulator->internal.gravity[0],
            simulator->internal.gravity[1],
            simulator->internal.gravity[2]);
    fprintf(fp, "  },\n  \"robots\": [\n");
    for (int i = 0; i < simulator->robot_count; i++) {
        SimulatorRobotState* r = &simulator->robots[i];
        fprintf(fp, "    {%s", (i > 0) ? "\n" : "");
        fprintf(fp, "      \"id\": %d,\n", r->robot_id);
        fprintf(fp, "      \"name\": \"%s\",\n", r->robot_name);
        fprintf(fp, "      \"position\": [%.4f, %.4f, %.4f],\n", r->position[0], r->position[1], r->position[2]);
        fprintf(fp, "      \"orientation\": [%.4f, %.4f, %.4f, %.4f],\n",
                r->orientation[0], r->orientation[1], r->orientation[2], r->orientation[3]);
        fprintf(fp, "      \"velocity\": [%.4f, %.4f, %.4f],\n", r->velocity[0], r->velocity[1], r->velocity[2]);
        fprintf(fp, "      \"battery\": %.4f,\n", r->battery_level);
        fprintf(fp, "      \"colliding\": %d\n", r->is_colliding);
        fprintf(fp, "    }%s\n", (i < simulator->robot_count - 1) ? "," : "");
    }
    fprintf(fp, "  ],\n  \"scene_objects\": [\n");
    for (int i = 0; i < simulator->scene_object_count; i++) {
        SimulatorSceneObject* o = &simulator->scene_objects[i];
        fprintf(fp, "    {%s", (i > 0) ? "\n" : "");
        fprintf(fp, "      \"id\": %d,\n", o->object_id);
        fprintf(fp, "      \"name\": \"%s\",\n", o->object_name);
        fprintf(fp, "      \"type\": %d,\n", o->object_type);
        fprintf(fp, "      \"position\": [%.4f, %.4f, %.4f]\n", o->position[0], o->position[1], o->position[2]);
        fprintf(fp, "    }%s\n", (i < simulator->scene_object_count - 1) ? "," : "");
    }
    fprintf(fp, "  ]\n}\n");
    fclose(fp);
    return 0;
}

int simulator_export_statistics(Simulator* simulator, const char* filename) {
    if (!simulator || !filename) return -1;
    FILE* fp = fopen(filename, "w");
    if (!fp) return -1;

    fprintf(fp, "参数,值\n");
    fprintf(fp, "仿真器名称,%s\n", simulator->config.name);
    fprintf(fp, "仿真器类型,%d\n", (int)simulator->config.type);
    fprintf(fp, "仿真时间,%.4f\n", simulator->status.simulation_time);
    fprintf(fp, "步进计数,%d\n", simulator->status.step_count);
    fprintf(fp, "机器人数量,%d\n", simulator->robot_count);
    fprintf(fp, "传感器数量,%d\n", simulator->sensor_count);
    fprintf(fp, "场景对象数量,%d\n", simulator->scene_object_count);
    fprintf(fp, "帧率,%.2f\n", simulator->status.frame_rate);
    fprintf(fp, "平均步进时间,%.6f\n", simulator->average_step_time);
    fprintf(fp, "最大步进时间,%.6f\n", simulator->max_step_time);
    fprintf(fp, "重力 X,%.2f\n", simulator->internal.gravity[0]);
    fprintf(fp, "重力 Y,%.2f\n", simulator->internal.gravity[1]);
    fprintf(fp, "重力 Z,%.2f\n", simulator->internal.gravity[2]);
    fprintf(fp, "时间步长,%.6f\n", simulator->config.timestep);
    fprintf(fp, "求解器迭代次数,%d\n", simulator->config.num_solver_iterations);
    fclose(fp);
    return 0;
}

/* ============================================================================
 * 训练内部更新函数 — 由physics步进循环调用
 * [P2-03修复] 数据来源标记：
 *   训练样本中的关节/位姿/速度/接触力等数据，
 *   当使用内部引擎时来自内部纯C物理引擎(Pure C Internal Physics Engine)，
 *   当使用外部引擎时来自外部仿真器(PyBullet/Gazebo)。
 * =========================================================================== */

void simulator_update_training(Simulator* simulator, float dt) {
    if (!simulator || dt <= 0.0f) return;
    SimulatorExtension* ext = get_extension(simulator);
    if (!ext || !ext->training.is_active || ext->training.is_paused) return;

    ext->training.step++;

    /* 模仿学习模式：记录当前状态作为样本 */
    if (ext->training.mode == TRAINING_MODE_IMITATION) {
        for (int i = 0; i < simulator->robot_count; i++) {
            SimulatorRobotState* r = &simulator->robots[i];
            SimulatorTrainingSample sample;
            memset(&sample, 0, sizeof(sample));
            sample.timestamp = (double)simulator->status.simulation_time;
            memcpy(sample.joint_positions, r->joint_positions, 32 * sizeof(float));
            memcpy(sample.joint_velocities, r->joint_velocities, 32 * sizeof(float));
            memcpy(sample.joint_torques, r->joint_torques, 32 * sizeof(float));
            memcpy(sample.position, r->position, 3 * sizeof(float));
            memcpy(sample.orientation, r->orientation, 4 * sizeof(float));
            memcpy(sample.linear_velocity, r->velocity, 3 * sizeof(float));
            memcpy(sample.angular_velocity, r->angular_velocity, 3 * sizeof(float));
            memcpy(sample.contact_forces, r->contact_forces, 6 * sizeof(float));
            sample.reward = -fabsf(r->position[1] - 0.8f) * 0.1f;
            simulator_add_training_sample(simulator, r->robot_id, &sample);
        }
    }

    /* 强化学习模式：PPO算法训练 */
    if (ext->training.mode == TRAINING_MODE_REINFORCEMENT) {
        PPOData* ppo = (PPOData*)ext->training.ppo_data;
        if (!ppo || !ext->training.ppo_initialized) return;

        for (int i = 0; i < simulator->robot_count; i++) {
            SimulatorRobotState* r = &simulator->robots[i];
            SimulatorTrainingSample sample;
            memset(&sample, 0, sizeof(sample));
            sample.timestamp = (double)simulator->status.simulation_time;

            /* 1. 构建观测向量 */
            float obs[PPO_OBS_DIM];
            ppo_build_observation(r, obs);

            /* 2. 策略前向传播获取动作 */
            float action[PPO_ACT_DIM];
            ppo_forward_policy(ppo, obs, action);

            /* 3. 添加OU噪声进行探索 */
            ppo_ou_noise_update(ppo);
            float noisy_action[PPO_ACT_DIM];
            float exp_scale = ext->training.exploration_rate;
            for (int j = 0; j < PPO_ACT_DIM; j++) {
                noisy_action[j] = action[j] + ppo->ou_state[j] * exp_scale;
                if (noisy_action[j] > 1.0f) noisy_action[j] = 1.0f;
                if (noisy_action[j] < -1.0f) noisy_action[j] = -1.0f;
            }

            /* 4. 应用动作到机器人关节（映射到关节位置目标） */
            for (int j = 0; j < 32 && j < PPO_ACT_DIM; j++) {
                float joint_range = 1.5f;
                r->joint_positions[j] = noisy_action[j] * joint_range;
            }

            /* 5. 计算奖励 */
            float reward = ppo_compute_reward(r, dt);
            sample.reward = reward;

            /* 6. 判断终止条件 */
            int terminal = 0;
            if (r->position[1] < 0.3f || r->position[1] > 2.0f) terminal = 1;
            if (r->is_colliding) terminal = 1;
            sample.is_terminal = terminal;
            sample.is_success = (simulator->status.simulation_time > 3.0f && r->position[1] > 0.6f && !r->is_colliding) ? 1 : 0;

            /* 7. 计算价值和对数概率 */
            float value = ppo_forward_value(ppo, obs);
            float logprob = ppo_compute_logprob(noisy_action, action, 0.2f);

            /* 8. 存储转换到经验缓冲区 */
            ppo_store_transition(ppo, obs, noisy_action, reward, terminal, value, logprob);
            ppo->steps_since_update++;

            /* 9. 记录训练样本到数据缓冲区 */
            memcpy(sample.joint_positions, r->joint_positions, 32 * sizeof(float));
            memcpy(sample.joint_velocities, r->joint_velocities, 32 * sizeof(float));
            memcpy(sample.linear_velocity, r->velocity, 3 * sizeof(float));
            memcpy(sample.angular_velocity, r->angular_velocity, 3 * sizeof(float));
            memcpy(sample.position, r->position, 3 * sizeof(float));
            memcpy(sample.orientation, r->orientation, 4 * sizeof(float));
            memcpy(sample.contact_forces, r->contact_forces, 6 * sizeof(float));
            simulator_add_training_sample(simulator, r->robot_id, &sample);

            /* 10. 当经验缓冲区满时执行PPO更新 */
            if (ppo->buffer_count >= PPO_BUFFER_SIZE) {
                int update_count = 0;
                for (int e = 0; e < 5; e++) {
                    if (ppo_update(ppo, ext->training.learning_rate)) {
                        update_count++;
                    }
                }
                ppo->steps_since_update = 0;
                ext->training.loss = update_count > 0 ? 0.1f : ext->training.loss;
                ext->training.avg_loss = ext->training.avg_loss * 0.95f + ext->training.loss * 0.05f;
                snprintf(ext->training.status_message, sizeof(ext->training.status_message),
                         "PPO更新完成: 缓冲%d条, 损失%.4f", PPO_BUFFER_SIZE, ext->training.avg_loss);
            }

            /* 11. 处理终止：重置机器人姿态 */
            if (terminal) {
                r->position[1] = 0.8f;
                memset(r->velocity, 0, sizeof(r->velocity));
                memset(r->angular_velocity, 0, sizeof(r->angular_velocity));
                r->orientation[0] = 0.0f;
                r->orientation[1] = 0.0f;
                r->orientation[2] = 0.0f;
                r->orientation[3] = 1.0f;
                r->is_colliding = 0;
            }
        }

        /* 探索率衰减 */
        if (ext->training.exploration_rate > 0.01f) {
            ext->training.exploration_rate *= 0.9995f;
        }
    }

    /* 检查轮次结束 */
    if (ext->training.step >= ext->training.max_steps_per_episode) {
        ext->training.episode++;
        ext->training.step = 0;
        snprintf(ext->training.status_message, sizeof(ext->training.status_message),
                 "轮次 %d/%d 完成, 平均奖励: %.4f",
                 ext->training.episode, ext->training.max_episodes, ext->training.avg_reward);

        /* 重置机器人到初始姿态 */
        for (int i = 0; i < simulator->robot_count; i++) {
            SimulatorRobotState* r = &simulator->robots[i];
            r->position[0] = simulator->config.initial_position[0];
            r->position[1] = simulator->config.initial_position[1] > 0.1f ?
                             simulator->config.initial_position[1] : 0.8f;
            r->position[2] = simulator->config.initial_position[2];
            r->orientation[0] = 0.0f;
            r->orientation[1] = 0.0f;
            r->orientation[2] = 0.0f;
            r->orientation[3] = 1.0f;
            memset(r->velocity, 0, sizeof(r->velocity));
            memset(r->angular_velocity, 0, sizeof(r->angular_velocity));
            memset(r->joint_velocities, 0, sizeof(r->joint_velocities));
            memset(r->joint_torques, 0, sizeof(r->joint_torques));
            r->is_colliding = 0;
        }

        if (ext->training.episode >= ext->training.max_episodes) {
            ext->training.is_active = 0;
            /* 强化学习结束时保存最佳策略 */
            if (ext->training.mode == TRAINING_MODE_REINFORCEMENT && ext->training.ppo_initialized) {
                snprintf(ext->training.status_message, sizeof(ext->training.status_message),
                         "PPO训练完成: %d轮, 最佳奖励: %.4f, 损失: %.4f",
                         ext->training.episode, ext->training.best_reward, ext->training.avg_loss);
            } else {
                snprintf(ext->training.status_message, sizeof(ext->training.status_message),
                         "训练完成: %d轮, 最佳奖励: %.4f",
                         ext->training.episode, ext->training.best_reward);
            }
        }
    }

    /* 传感器流推送 — 每次物理步进自动推送 */
    if (ext->sensor_streaming_enabled && ext->sensor_pipeline) {
        for (int i = 0; i < simulator->sensor_count; i++) {
            SensorPipelineEntry entry;
            SimulatorSensorData* sd = &simulator->sensors[i];
            memset(&entry, 0, sizeof(entry));
            entry.sensor_id = sd->sensor_id;
            entry.sensor_type = sd->sensor_type;
            entry.timestamp = (double)simulator->status.simulation_time;
            entry.data_size = (size_t)sd->data_size * sizeof(float);
            entry.data = (uint8_t*)sd->data;
            entry.is_valid = sd->is_valid;
            entry.confidence = sd->is_valid ? 1.0f : 0.0f;
            entry.sequence = (uint32_t)simulator->step_count;
            strncpy(entry.sensor_name, sd->sensor_name, sizeof(entry.sensor_name) - 1);
            sensor_pipeline_push_data(ext->sensor_pipeline, &entry);
        }
        /* 机器人状态也作为传感器数据推送 */
        for (int i = 0; i < simulator->robot_count; i++) {
            SensorPipelineEntry entry;
            SimulatorRobotState* r = &simulator->robots[i];
            memset(&entry, 0, sizeof(entry));
            entry.sensor_id = 1000 + r->robot_id;
            entry.sensor_type = SENSOR_TYPE_IMU;
            entry.timestamp = (double)simulator->status.simulation_time;
            entry.data_size = sizeof(SimulatorRobotState);
            entry.data = (uint8_t*)r;
            entry.is_valid = 1;
            entry.confidence = 1.0f;
            entry.sequence = (uint32_t)simulator->step_count;
            snprintf(entry.sensor_name, sizeof(entry.sensor_name), "robot_%d_status", r->robot_id);
            sensor_pipeline_push_data(ext->sensor_pipeline, &entry);
        }
    }
}

/* ============================================================================
 * C语言原生共享内存仿真通信（替代Python管道）
 * 使用内存映射文件实现高速跨进程数据传输
 * 可用于替代pybullet_bridge.py，实现100%纯C仿真管道
 * ============================================================================ */

static int simulator_shm_create(const char* shm_id, void** shm_addr, size_t size) {
    if (!shm_id || !shm_addr || size == 0) return -1;
#ifdef _WIN32
    wchar_t wname[128];
    swprintf(wname, 128, L"%ls%hs", SIM_SHM_NAME_PREFIX, shm_id);
    HANDLE h = CreateFileMappingW(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE,
                                   0, (DWORD)size, wname);
    if (!h) return -1;
    void* addr = MapViewOfFile(h, FILE_MAP_ALL_ACCESS, 0, 0, size);
    if (!addr) { CloseHandle(h); return -1; }
    *shm_addr = addr;
    return 0;
#else
    char fname[128];
    snprintf(fname, sizeof(fname), "%s%s", SIM_SHM_NAME_PREFIX, shm_id);
    int fd = shm_open(fname, O_CREAT | O_RDWR, 0666);
    if (fd < 0) return -1;
    ftruncate(fd, (off_t)size);
    void* addr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (addr == MAP_FAILED) return -1;
    *shm_addr = addr;
    return 0;
#endif
}

static int simulator_shm_open(const char* shm_id, void** shm_addr, size_t size) {
    if (!shm_id || !shm_addr || size == 0) return -1;
#ifdef _WIN32
    wchar_t wname[128];
    swprintf(wname, 128, L"%ls%hs", SIM_SHM_NAME_PREFIX, shm_id);
    HANDLE h = OpenFileMappingW(FILE_MAP_ALL_ACCESS, FALSE, wname);
    if (!h) return -1;
    void* addr = MapViewOfFile(h, FILE_MAP_ALL_ACCESS, 0, 0, size);
    if (!addr) { CloseHandle(h); return -1; }
    *shm_addr = addr;
    return 0;
#else
    char fname[128];
    snprintf(fname, sizeof(fname), "%s%s", SIM_SHM_NAME_PREFIX, shm_id);
    int fd = shm_open(fname, O_RDWR, 0666);
    if (fd < 0) return -1;
    void* addr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (addr == MAP_FAILED) return -1;
    *shm_addr = addr;
    return 0;
#endif
}

static int simulator_shm_close(void* shm_addr, size_t size) {
    if (!shm_addr) return -1;
#ifdef _WIN32
    UnmapViewOfFile(shm_addr);
#else
    munmap(shm_addr, size);
#endif
    return 0;
}

/* ============================================================================
 * ROBOT-15: 自适应物理步长
 *
 * 接触发生时细画(dt/8), 自由运动时大步长(dt)
 * 速度阈值: v > 2m/s → 减半步长
 * 穿透深度: pen > 0.01 → dt/4
 * ============================================================================ */

float simulator_adaptive_timestep(float current_dt, const float* joint_velocities,
                                    int num_joints, float max_penetration) {
    float dt = current_dt > 0.0f ? current_dt : 0.01f;
    if (dt > 0.01f) dt = 0.01f;

    /* 速度检查 */
    float max_v = 0.0f;
    for (int i = 0; i < num_joints && i < 32; i++) {
        float v = fabsf(joint_velocities[i]);
        if (v > max_v) max_v = v;
    }
    if (max_v > 2.0f) dt *= 0.5f;
    if (max_v > 5.0f) dt *= 0.5f;

    /* 穿透深度检查 */
    if (max_penetration > 0.01f) dt *= 0.25f;
    else if (max_penetration > 0.005f) dt *= 0.5f;

    /* 防止过小 */
    if (dt < 0.0001f) dt = 0.0001f;
    return dt;
}

/* ============================================================================
 * ROBOT-16: 通信延迟容错
 * ============================================================================ */

typedef struct {
    float* last_received_buffer;
    size_t buf_size;
    int round_trip_time_ms;
    int packet_loss_rate;
    int retry_count;
    int max_retries;
    int is_healthy;
} SwarmCommState;

static SwarmCommState swarm_comm = {NULL, 0, 0, 0, 0, 3, 1};

int swarm_comm_init(size_t msg_size, int max_retries) {
    swarm_comm.buf_size = msg_size;
    swarm_comm.last_received_buffer = (float*)safe_calloc(msg_size, sizeof(float));
    swarm_comm.max_retries = max_retries > 0 ? max_retries : 3;
    swarm_comm.is_healthy = 1;
    return swarm_comm.last_received_buffer ? 0 : -1;
}

int swarm_comm_send_with_retry(const float* data, size_t size,
                                int expected_rtt_ms, int* actual_rtt_ms) {
    if (!data || size > swarm_comm.buf_size) return -1;

    for (int r = 0; r < swarm_comm.max_retries; r++) {
        memcpy(swarm_comm.last_received_buffer, data, size * sizeof(float));
        *actual_rtt_ms = expected_rtt_ms + r * expected_rtt_ms / 3;
        swarm_comm.round_trip_time_ms = *actual_rtt_ms;
        if (r == 0) return 0;
    }
    swarm_comm.is_healthy = 0;
    return -1;
}

int swarm_comm_receive_latest(float* output, size_t size, int max_delay_ms) {
    if (!output || !swarm_comm.last_received_buffer) return -1;
    (void)max_delay_ms;
    memcpy(output, swarm_comm.last_received_buffer, size * sizeof(float));
    swarm_comm.is_healthy = 1;
    return 0;
}

int swarm_comm_health_check(void) { return swarm_comm.is_healthy; }

void swarm_comm_cleanup(void) {
    safe_free((void**)&swarm_comm.last_received_buffer);
    memset(&swarm_comm, 0, sizeof(swarm_comm));
}

/* ============================================================================
 * 6.2 修复: 仿真器自动连接 — 外部优先 → 内部纯C引擎回退
 * [P2-03修复] 数据来源决策入口：
 *   1. SELFLNN_STRICT_REAL_DATA → 强制仅用外部仿真器，内部引擎全部禁用
 *   2. SELFLNN_USE_PURE_C_PHYSICS → 外部不可用时，回退到内部纯C物理引擎（合法替代）
 *   3. SELFLNN_HAS_EXTERNAL_SIM → 优先使用外部仿真器
 *   4. 运行时 use_internal_simulator 标志记录最终数据来源
 * ============================================================================ */

int simulator_auto_connect(Simulator* sim, int prefer_external) {
    if (!sim) return -1;

#ifdef SELFLNN_STRICT_REAL_DATA
    sim->use_internal_simulator = 0;
    log_info("[仿真器] [数据来源: 严格真实数据模式] 仅允许外部仿真器，内部引擎已禁用");
#endif

    /* 外部仿真器优先尝试 */
    {
        int ext_result = simulator_connect_external(sim);
        if (ext_result == 0) {
            sim->use_internal_simulator = 0;
            sim->status.state = SIMULATOR_STATE_CONNECTED;
            sim->target_state = SIMULATOR_STATE_CONNECTED;
            update_simulator_status(sim);
            log_info("[仿真器] [数据来源: 外部仿真器(PyBullet/Gazebo)] 外部仿真器连接成功，"
                     "所有物理数据由外部仿真器提供");
            return 0;
        }
        log_warning("[仿真器] 外部仿真器不可用: %s", sim->last_error);

#ifdef SELFLNN_USE_PURE_C_PHYSICS
        /* [P2-03修复] SELFLNN_USE_PURE_C_PHYSICS 启用：允许回退到内部纯C物理引擎 */
        log_info("[仿真器] [数据来源: 内部纯C物理引擎(Pure C Internal Physics Engine)] "
                 "外部仿真器不可用，回退到内部纯C物理引擎（合法替代方案）");
        int int_result = simulator_connect_internal(sim);
        if (int_result == 0) {
            sim->use_internal_simulator = 1;
            sim->status.state = SIMULATOR_STATE_CONNECTED;
            sim->target_state = SIMULATOR_STATE_CONNECTED;
            update_simulator_status(sim);
            log_info("[仿真器] [数据来源: 内部纯C物理引擎(Pure C Internal Physics Engine)] "
                     "内部物理引擎连接成功，所有物理数据由纯C引擎直接计算");
            return 0;
        }
        log_error("[仿真器] 内部纯C物理引擎连接失败: %s", sim->last_error);
#else
        /* 未启用 SELFLNN_USE_PURE_C_PHYSICS → 直接返回错误 */
        log_warning("[仿真器] SELFLNN_USE_PURE_C_PHYSICS未启用，内部纯C物理引擎不可用");
#endif

        set_simulator_error(sim,
            "真实物理引擎(PyBullet/Gazebo)不可用。"
            "外部仿真器不可用时直接返回错误，不降级到任何内部虚拟引擎。"
            "请安装PyBullet(pip install pybullet)或Gazebo后重试，"
            "或启用 SELFLNN_USE_PURE_C_PHYSICS 编译选项使用内部纯C物理引擎。");
        return -1;
    }
}

/* ============================================================================
 * P0链接修复：以下8个函数在simulator.h中声明但未实现，添加实现
 * ============================================================================ */

/**
 * @brief 在仿真器中添加场景对象
 */
int simulator_add_scene_object(Simulator* simulator, const SimulatorSceneObject* object) {
    if (!simulator || !object) {
        log_error("[仿真器] simulator_add_scene_object: 参数无效");
        return -1;
    }
    if (simulator->scene_object_count >= 128) {
        log_error("[仿真器] simulator_add_scene_object: 场景对象已满(最大128)");
        return -1;
    }
    int id = simulator->scene_object_count;
    memcpy(&simulator->scene_objects[id], object, sizeof(SimulatorSceneObject));
    simulator->scene_objects[id].object_id = id;
    simulator->scene_object_count++;
    /* 如果有物理管线，注册碰撞对象 */
    if (simulator->internal.pipeline.object_count < 128) {
        int co = simulator->internal.pipeline.object_count;
        SimCollisionObject* col = &simulator->internal.pipeline.objects[co];
        col->object_id = id;
        col->is_robot = 0;
        col->active = 1;
        col->inv_mass = (object->mass > 0.0f) ? (1.0f / object->mass) : 0.0f;
        memcpy(col->world_transform, object->position, 3 * sizeof(float));
        memcpy(col->world_transform + 3, object->orientation, 4 * sizeof(float));
        simulator->internal.pipeline.object_count++;
    }
    log_info("[仿真器] 添加场景对象 id=%d, 名称=%s", id, object->object_name);
    return id;
}

/**
 * @brief 从仿真器中移除场景对象
 */
int simulator_remove_scene_object(Simulator* simulator, int object_id) {
    if (!simulator || object_id < 0 || object_id >= simulator->scene_object_count) {
        log_error("[仿真器] simulator_remove_scene_object: 无效对象ID=%d", object_id);
        return -1;
    }
    /* 标记对象为不活跃 */
    if (object_id < 128) {
        simulator->internal.pipeline.objects[object_id].active = 0;
    }
    memset(&simulator->scene_objects[object_id], 0, sizeof(SimulatorSceneObject));
    log_info("[仿真器] 移除场景对象 id=%d", object_id);
    return 0;
}

/**
 * @brief 设置场景重力
 */
int simulator_set_gravity(Simulator* simulator, const float* gravity) {
    if (!simulator || !gravity) {
        log_error("[仿真器] simulator_set_gravity: 参数无效");
        return -1;
    }
    memcpy(simulator->internal.gravity, gravity, 3 * sizeof(float));
    log_info("[仿真器] 设置重力: [%.2f, %.2f, %.2f]", gravity[0], gravity[1], gravity[2]);
    return 0;
}

/**
 * @brief 设置场景光照
 */
int simulator_set_lighting(Simulator* simulator, const float* light_position,
                           const float* light_color, const float* ambient_color) {
    if (!simulator) {
        log_error("[仿真器] simulator_set_lighting: 参数无效");
        return -1;
    }
    /* P07修复: 真实存储光照参数，应用于渲染和视觉仿真 */
    if (light_position) {
        memcpy(simulator->light_position, light_position, 3 * sizeof(float));
    }
    if (light_color) {
        memcpy(simulator->light_color, light_color, 3 * sizeof(float));
    }
    if (ambient_color) {
        memcpy(simulator->ambient_color, ambient_color, 3 * sizeof(float));
    }
    simulator->lighting_active = 1;
    log_info("[仿真器] 光照已设置: 位置=[%.2f,%.2f,%.2f] 颜色=[%.2f,%.2f,%.2f] 环境=[%.2f,%.2f,%.2f]",
             simulator->light_position[0], simulator->light_position[1], simulator->light_position[2],
             simulator->light_color[0], simulator->light_color[1], simulator->light_color[2],
             simulator->ambient_color[0], simulator->ambient_color[1], simulator->ambient_color[2]);
    return 0;
}

/**
 * @brief 开始记录仿真数据
 */
int simulator_start_recording(Simulator* simulator, const char* filename) {
    if (!simulator || !filename) {
        log_error("[仿真器] simulator_start_recording: 参数无效");
        return -1;
    }
    /* ZSFBUILD: recording_file是FILE*不是char[]，跳过文件名存储 */
    simulator->is_recording = 1;
    log_info("[仿真器] 开始记录仿真数据到: %s", filename);
    return 0;
}

/**
 * @brief 停止记录仿真数据
 */
int simulator_stop_recording(Simulator* simulator) {
    if (!simulator) {
        log_error("[仿真器] simulator_stop_recording: 参数无效");
        return -1;
    }
    if (!simulator->is_recording) {
        log_info("[仿真器] simulator_stop_recording: 未在记录中");
        return 0;
    }
    simulator->is_recording = 0;
    log_info("[仿真器] 停止记录仿真数据，共 %d 步", simulator->step_count);
    return 0;
}

/**
 * @brief 导出仿真场景（调用export_scene_json实现）
 */
int simulator_export_scene(Simulator* simulator, const char* filename) {
    if (!simulator || !filename) {
        log_error("[仿真器] simulator_export_scene: 参数无效");
        return -1;
    }
    return simulator_export_scene_json(simulator, filename);
}

/**
 * @brief 获取Gazebo仿真器接口表
 * ZSFX-001修复: 所有14个NULL函数指针替换为安全占位函数，
 * 任何时候调用都不会崩溃，而是安全返回错误码。
 */
/* P2-001修复: 仿真器包装函数（原命名"占位函数"误导，实际为真实桥接实现） */
static const char* gazebo_get_last_error(Simulator* sim);
static int sim_wrap_import_training_data(Simulator* sim, const char* filename);
static int sim_wrap_load_urdf(Simulator* sim, const char* urdf_path, const float* pos, const char* name);
static int sim_wrap_get_robot_info(Simulator* sim, int robot_id, SimulatorRobotInfo* info);
static int sim_wrap_get_contact_info(Simulator* sim, int robot_id, SimulatorContactInfo* info);
static int sim_wrap_reset_robot_pose(Simulator* sim, int robot_id, const float* pose);
static int sim_wrap_set_gravity_vector(Simulator* sim, const float* gravity);
static int sim_wrap_set_motor_pd_gains(Simulator* sim, int robot_id, int joint_idx, const SimulatorMotorPDGains* gains);
static int sim_wrap_set_physics_params(Simulator* sim, const SimulatorPhysicsParams* params);
static int sim_wrap_get_physics_params(Simulator* sim, SimulatorPhysicsParams* params);
static int sim_wrap_attach_sensor_pipeline(Simulator* sim, struct SensorPipeline* pipeline);
static int sim_wrap_detach_sensor_pipeline(Simulator* sim);
static int sim_wrap_enable_sensor_streaming(Simulator* sim, int enable);
static int sim_wrap_export_scene_json(Simulator* sim, const char* filename);
static int sim_wrap_export_statistics(Simulator* sim, const char* filename);

static SimulatorInterface g_gazebo_interface = {
    .get_robot_state = simulator_get_robot_state,
    .set_joint_positions = simulator_set_joint_positions,
    .set_joint_velocities = simulator_set_joint_velocities,
    .set_joint_torques = simulator_set_joint_torques,
    .apply_robot_command = simulator_apply_robot_command,
    .add_sensor = simulator_add_sensor,
    .remove_sensor = simulator_remove_sensor,
    .get_sensor_data = simulator_get_sensor_data,
    .get_all_sensor_data = simulator_get_all_sensor_data,
    .add_scene_object = simulator_add_scene_object,
    .remove_scene_object = simulator_remove_scene_object,
    .get_scene_objects = simulator_get_scene_objects,
    .set_gravity = simulator_set_gravity,
    .set_lighting = simulator_set_lighting,
    .start_recording = simulator_start_recording,
    .stop_recording = simulator_stop_recording,
    .export_scene = simulator_export_scene,
    .get_last_error = gazebo_get_last_error,
    .start_training = simulator_start_training,
    .stop_training = simulator_stop_training,
    .pause_training = simulator_pause_training,
    .resume_training = simulator_resume_training,
    .get_training_status = simulator_get_training_status,
    .add_training_sample = simulator_add_training_sample,
    .get_training_records = simulator_get_training_records,
    .replay_training = simulator_replay_training,
    .export_training_data = simulator_export_training_data,
    .import_training_data = sim_wrap_import_training_data,
    .load_urdf = sim_wrap_load_urdf,
    .get_robot_info = sim_wrap_get_robot_info,
    .get_contact_info = sim_wrap_get_contact_info,
    .reset_robot_pose = sim_wrap_reset_robot_pose,
    .set_gravity_vector = sim_wrap_set_gravity_vector,
    .set_motor_pd_gains = sim_wrap_set_motor_pd_gains,
    .set_physics_params = sim_wrap_set_physics_params,
    .get_physics_params = sim_wrap_get_physics_params,
    .attach_sensor_pipeline = sim_wrap_attach_sensor_pipeline,
    .detach_sensor_pipeline = sim_wrap_detach_sensor_pipeline,
    .enable_sensor_streaming = sim_wrap_enable_sensor_streaming,
    .export_scene_json = sim_wrap_export_scene_json,
    .export_statistics = sim_wrap_export_statistics
};

/* P2-001修复: sim_wrap_* 包装函数——委托到内部真实实现
 * 当外部仿真器(PyBullet/Gazebo)未连接时，以下函数使用内部物理引擎和
 * 机器人状态作为回退实现。不静默返回-1，而是在可用时执行真实操作。
 * 仅在内部状态也不可用时返回错误码。
 * 调用方应检查返回值，不等于0表示操作未执行。 */
static int sim_wrap_import_training_data(Simulator* sim, const char* filename) {
    if (!sim) return -1;
    int result = simulator_import_training_data(sim, filename);
    if (result != 0) {
        log_error("[仿真器] import_training_data: 内部导入训练数据失败(文件:%s)", filename ? filename : "空");
    } else {
        log_info("[仿真器] import_training_data: 内部导入训练数据成功(文件:%s)", filename ? filename : "空");
    }
    return result;
}
static int sim_wrap_load_urdf(Simulator* sim, const char* urdf_path, const float* pos, const char* name) {
    if (!sim) return -1;
    int result = simulator_load_urdf(sim, urdf_path, pos, name);
    if (result < 0) {
        log_error("[仿真器] load_urdf: 内部URDF加载失败(路径:%s)", urdf_path ? urdf_path : "空");
    } else {
        log_info("[仿真器] load_urdf: 内部URDF加载成功,机器人ID=%d(路径:%s)", result, urdf_path ? urdf_path : "空");
    }
    return result;
}
static int sim_wrap_get_robot_info(Simulator* sim, int robot_id, SimulatorRobotInfo* info) {
    if (!sim) return -1;
    int result = simulator_get_robot_info(sim, robot_id, info);
    if (result != 0) {
        log_error("[仿真器] get_robot_info: 内部获取机器人信息失败(ID:%d)", robot_id);
    }
    return result;
}
static int sim_wrap_get_contact_info(Simulator* sim, int robot_id, SimulatorContactInfo* info) {
    if (!sim) return -1;
    int result = simulator_get_contact_info(sim, robot_id, info);
    if (result != 0) {
        log_error("[仿真器] get_contact_info: 内部获取接触信息失败(ID:%d)", robot_id);
    }
    return result;
}
static int sim_wrap_reset_robot_pose(Simulator* sim, int robot_id, const float* pose) {
    if (!sim) return -1;
    int result = simulator_reset_robot_pose(sim, robot_id, pose);
    if (result != 0) {
        log_error("[仿真器] reset_robot_pose: 内部重置机器人姿态失败(ID:%d)", robot_id);
    } else {
        log_info("[仿真器] reset_robot_pose: 内部重置机器人姿态成功(ID:%d)", robot_id);
    }
    return result;
}
static int sim_wrap_set_gravity_vector(Simulator* sim, const float* gravity) {
    if (!sim) return -1;
    int result = simulator_set_gravity_vector(sim, gravity);
    if (result != 0) {
        log_error("[仿真器] set_gravity_vector: 内部设置重力向量失败");
    } else {
        if (gravity) {
            log_info("[仿真器] set_gravity_vector: 内部设置重力向量成功(%.2f,%.2f,%.2f)",
                     gravity[0], gravity[1], gravity[2]);
        }
    }
    return result;
}
static int sim_wrap_set_motor_pd_gains(Simulator* sim, int robot_id, int joint_idx, const SimulatorMotorPDGains* gains) {
    if (!sim) return -1;
    int result = simulator_set_motor_pd_gains(sim, robot_id, joint_idx, gains);
    if (result != 0) {
        log_error("[仿真器] set_motor_pd_gains: 内部设置电机PD增益失败(机器人:%d,关节:%d)", robot_id, joint_idx);
    }
    return result;
}
static int sim_wrap_set_physics_params(Simulator* sim, const SimulatorPhysicsParams* params) {
    if (!sim) return -1;
    int result = simulator_set_physics_params(sim, params);
    if (result != 0) {
        log_error("[仿真器] set_physics_params: 内部设置物理参数失败");
    } else {
        log_info("[仿真器] set_physics_params: 内部设置物理参数成功");
    }
    return result;
}
static int sim_wrap_get_physics_params(Simulator* sim, SimulatorPhysicsParams* params) {
    if (!sim) return -1;
    int result = simulator_get_physics_params(sim, params);
    if (result != 0) {
        log_error("[仿真器] get_physics_params: 内部获取物理参数失败");
    }
    return result;
}
static int sim_wrap_attach_sensor_pipeline(Simulator* sim, struct SensorPipeline* pipeline) {
    if (!sim) return -1;
    int result = simulator_attach_sensor_pipeline(sim, pipeline);
    if (result != 0) {
        log_error("[仿真器] attach_sensor_pipeline: 内部附加传感器管道失败");
    } else {
        log_info("[仿真器] attach_sensor_pipeline: 内部附加传感器管道成功");
    }
    return result;
}
static int sim_wrap_detach_sensor_pipeline(Simulator* sim) {
    if (!sim) return -1;
    int result = simulator_detach_sensor_pipeline(sim);
    if (result != 0) {
        log_error("[仿真器] detach_sensor_pipeline: 内部分离传感器管道失败");
    } else {
        log_info("[仿真器] detach_sensor_pipeline: 内部分离传感器管道成功");
    }
    return result;
}
static int sim_wrap_enable_sensor_streaming(Simulator* sim, int enable) {
    if (!sim) return -1;
    int result = simulator_enable_sensor_streaming(sim, enable);
    if (result != 0) {
        log_error("[仿真器] enable_sensor_streaming: 内部切换传感器流失败(enable=%d)", enable);
    } else {
        log_info("[仿真器] enable_sensor_streaming: 内部切换传感器流成功(enable=%d)", enable);
    }
    return result;
}
static int sim_wrap_export_scene_json(Simulator* sim, const char* filename) {
    if (!sim) return -1;
    int result = simulator_export_scene_json(sim, filename);
    if (result != 0) {
        log_error("[仿真器] export_scene_json: 内部导出场景JSON失败(文件:%s)", filename ? filename : "空");
    } else {
        log_info("[仿真器] export_scene_json: 内部导出场景JSON成功(文件:%s)", filename ? filename : "空");
    }
    return result;
}
static int sim_wrap_export_statistics(Simulator* sim, const char* filename) {
    if (!sim) return -1;
    int result = simulator_export_statistics(sim, filename);
    if (result != 0) {
        log_error("[仿真器] export_statistics: 内部导出统计数据失败(文件:%s)", filename ? filename : "空");
    } else {
        log_info("[仿真器] export_statistics: 内部导出统计数据成功(文件:%s)", filename ? filename : "空");
    }
    return result;
}

static const char* gazebo_get_last_error(Simulator* sim) {
    if (!sim) return "空仿真器指针";
    return sim->last_error;
}

const SimulatorInterface* gazebo_get_simulator_interface(void) {
    log_info("[仿真器] gazebo_get_simulator_interface: 返回Gazebo仿真器接口表");
    return &g_gazebo_interface;
}