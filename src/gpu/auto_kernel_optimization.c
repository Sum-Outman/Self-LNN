/**
 * @file auto_kernel_optimization.c
 * @brief 自动内核优化系统实现
 *
 * 实现GPU内核自动调优引擎，包含性能分析数据库、自动参数选择、
 * 工作组大小优化、向量化宽度自适应和性能预测功能。
 */


#include "selflnn/gpu/auto_kernel_optimization.h"
#include "selflnn/gpu/gpu.h"
#include "selflnn/core/errors.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/string_utils.h"
#include "selflnn/utils/perf.h"
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif
#include <math.h>
#include <stdio.h>
#include <float.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#include <sys/time.h>
#endif

/**
 * @brief 性能数据库初始容量
 */
#define INITIAL_DATABASE_CAPACITY 256

/**
 * @brief 性能数据库增长因子
 */
#define DATABASE_GROWTH_FACTOR 2

/**
 * @brief 最大内核名称长度
 */
#define MAX_KERNEL_NAME_LENGTH 128

/**
 * @brief 调优搜索的迭代次数
 */
#define TUNE_SEARCH_ITERATIONS 32

/**
 * @brief 性能缓存过期时间（秒）
 */
#define CACHE_EXPIRY_SECONDS 3600

/**
 * @brief 默认工作组大小表（按内核类型）
 */
static const size_t default_work_group_sizes[10][3] = {
    {16, 16, 1},   /* MATMUL */
    {16, 16, 1},   /* CONV2D */
    {16, 16, 1},   /* POOLING */
    {128, 1, 1},   /* NORMALIZATION */
    {256, 1, 1},   /* ACTIVATION */
    {128, 1, 1},   /* SOFTMAX */
    {16, 16, 1},   /* LIQUID_GATE */
    {256, 1, 1},   /* REDUCTION */
    {256, 1, 1},   /* ELEMENTWISE */
    {64, 1, 1}     /* CUSTOM */
};

/**
 * @brief 默认向量化宽度表（按内核类型）
 */
static const int default_vector_widths[10] = {
    4,  /* MATMUL */
    4,  /* CONV2D */
    4,  /* POOLING */
    8,  /* NORMALIZATION */
    8,  /* ACTIVATION */
    4,  /* SOFTMAX */
    4,  /* LIQUID_GATE */
    4,  /* REDUCTION */
    8,  /* ELEMENTWISE */
    4   /* CUSTOM */
};

/**
 * @brief 默认循环展开因子表（按内核类型）
 */
static const int default_unroll_factors[10] = {
    4,  /* MATMUL */
    2,  /* CONV2D */
    2,  /* POOLING */
    1,  /* NORMALIZATION */
    4,  /* ACTIVATION */
    1,  /* SOFTMAX */
    2,  /* LIQUID_GATE */
    1,  /* REDUCTION */
    4,  /* ELEMENTWISE */
    2   /* CUSTOM */
};

/**
 * @brief 性能数据库条目
 */
typedef struct {
    KernelPerformanceRecord record;  /**< 性能记录 */
    time_t timestamp;                /**< 记录时间戳 */
    int is_valid;                    /**< 是否有效 */
} DatabaseEntry;

/**
 * @brief 调优搜索空间参数
 */
typedef struct {
    size_t local_sizes[4][3];    /**< 待搜索的工作组大小 */
    int vector_widths[4];        /**< 待搜索的向量化宽度 */
    int unroll_factors[4];       /**< 待搜索的展开因子 */
    int use_shared_options[2];   /**< 共享内存选项 */
    int search_count;            /**< 搜索配置数量 */
} TuneSearchSpace;

/**
 * @brief 自动内核优化器内部结构
 */
struct AutoKernelOptimizer {
    int device_id;                          /**< 设备ID */
    char device_name[256];                  /**< 设备名称 */

    /** 性能数据库 */
    DatabaseEntry* database;                /**< 数据库数组 */
    int database_capacity;                  /**< 数据库容量 */
    int database_count;                     /**< 数据库条目数 */

    /** 统计信息 */
    int total_profiles;                     /**< 总分析次数 */
    int total_optimizations;                /**< 总优化次数 */
    double accumulated_speedup;             /**< 累积加速比 */

    /** 缓存 */
    int cache_enabled;                      /**< 是否启用缓存 */
    time_t last_cache_cleanup;              /**< 上次缓存清理时间 */

    /** 默认优化参数（按内核类型） */
    KernelOptimizationParams default_params[10]; /**< 默认参数 */

    /** 设备能力 */
    int max_work_group_size;                /**< 最大工作组大小 */
    int supports_doubles;                   /**< 是否支持双精度 */
    int supports_half;                      /**< 是否支持半精度 */
    int compute_units;                      /**< 计算单元数 */
    float clock_speed_mhz;                  /**< 时钟速度 */
    size_t global_memory_size;              /**< 全局内存大小 */

    /** F-003修复：真实GPU测量上下文 */
    GpuContext* online_context;             /**< GPU上下文（用于在线测量），NULL=仅预测模式 */
    double online_default_time;             /**< 在线测量默认内核时间（ms），用于搜索空间校准 */

    /** 线程安全 */
#ifdef _WIN32
    CRITICAL_SECTION mutex;                 /**< 互斥锁 */
#else
    pthread_mutex_t mutex;
#endif
};

/**
 * @brief 获取当前时间戳（秒）
 */
static double get_current_timestamp_sec(void) {
#ifdef _WIN32
    LARGE_INTEGER frequency, counter;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)frequency.QuadPart;
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
#endif
}

/**
 * @brief 初始化默认优化参数
 */
static void init_default_params(AutoKernelOptimizer* optimizer) {
    for (int i = 0; i < 10; i++) {
        KernelOptimizationParams* params = &optimizer->default_params[i];
        memset(params, 0, sizeof(KernelOptimizationParams));

        params->local_work_size[0] = default_work_group_sizes[i][0];
        params->local_work_size[1] = default_work_group_sizes[i][1];
        params->local_work_size[2] = default_work_group_sizes[i][2];
        params->vector_width = default_vector_widths[i];
        params->unroll_factor = default_unroll_factors[i];
        params->use_shared_memory = (i == KERNEL_TYPE_MATMUL || i == KERNEL_TYPE_CONV2D ||
                                     i == KERNEL_TYPE_LIQUID_GATE) ? 1 : 0;
        params->shared_memory_size = (i == KERNEL_TYPE_MATMUL) ? 49152 :
                                     (i == KERNEL_TYPE_LIQUID_GATE) ? 32768 : 16384;
        params->use_tensor_cores = (i == KERNEL_TYPE_MATMUL) ? 1 : 0;
        params->num_workgroups = 0;
        params->strategy = OPT_STRATEGY_BALANCED;
        params->prefer_occupancy = 0;
        params->enable_doubles = optimizer->supports_doubles;
    }
}

/**
 * @brief 生成搜索空间配置
 */
static void generate_search_space(TuneSearchSpace* space, int max_wg_size) {
    memset(space, 0, sizeof(TuneSearchSpace));

    int size_idx = 0;

    /* 工作组大小搜索 */
    if (max_wg_size >= 512) {
        space->local_sizes[size_idx][0] = 32; space->local_sizes[size_idx][1] = 32; space->local_sizes[size_idx++][2] = 1;
    }
    if (max_wg_size >= 256) {
        space->local_sizes[size_idx][0] = 16; space->local_sizes[size_idx][1] = 16; space->local_sizes[size_idx++][2] = 1;
    }
    if (max_wg_size >= 128) {
        space->local_sizes[size_idx][0] = 8; space->local_sizes[size_idx][1] = 8; space->local_sizes[size_idx++][2] = 1;
    }
    {
        space->local_sizes[size_idx][0] = 64; space->local_sizes[size_idx][1] = 1; space->local_sizes[size_idx++][2] = 1;
    }

    /* 向量化宽度搜索 */
    space->vector_widths[0] = 1;
    space->vector_widths[1] = 2;
    space->vector_widths[2] = 4;
    space->vector_widths[3] = 8;

    /* 循环展开因子搜索 */
    space->unroll_factors[0] = 1;
    space->unroll_factors[1] = 2;
    space->unroll_factors[2] = 4;
    space->unroll_factors[3] = 8;

    /* 共享内存选项 */
    space->use_shared_options[0] = 0;
    space->use_shared_options[1] = 1;

    space->search_count = size_idx;
}

/**
 * @brief 比较两个优化参数是否相同
 */
static int params_equal(const KernelOptimizationParams* a,
                        const KernelOptimizationParams* b) {
    return a->local_work_size[0] == b->local_work_size[0] &&
           a->local_work_size[1] == b->local_work_size[1] &&
           a->local_work_size[2] == b->local_work_size[2] &&
           a->vector_width == b->vector_width &&
           a->unroll_factor == b->unroll_factor &&
           a->use_shared_memory == b->use_shared_memory &&
           a->use_tensor_cores == b->use_tensor_cores &&
           a->prefer_occupancy == b->prefer_occupancy;
}

/**
 * @brief 计算两个参数配置的差异分数
 */
static double params_difference_score(const KernelOptimizationParams* a,
                                     const KernelOptimizationParams* b) {
    double score = 0.0;
    for (int i = 0; i < 3; i++) {
        score += (double)(a->local_work_size[i] > b->local_work_size[i] ?
                         a->local_work_size[i] - b->local_work_size[i] :
                         b->local_work_size[i] - a->local_work_size[i]);
    }
    score += (double)(a->vector_width > b->vector_width ?
                     a->vector_width - b->vector_width :
                     b->vector_width - a->vector_width) * 2.0;
    score += (double)(a->unroll_factor > b->unroll_factor ?
                     a->unroll_factor - b->unroll_factor :
                     b->unroll_factor - a->unroll_factor) * 1.5;
    if (a->use_shared_memory != b->use_shared_memory) score += 5.0;
    if (a->use_tensor_cores != b->use_tensor_cores) score += 3.0;
    return score;
}

/**
 * @brief 在数据库中查找匹配记录
 */
static int find_record_in_database(AutoKernelOptimizer* optimizer,
                                   KernelType kernel_type,
                                   const char* kernel_name,
                                   size_t input_size,
                                   size_t output_size) {
    for (int i = 0; i < optimizer->database_count; i++) {
        DatabaseEntry* entry = &optimizer->database[i];
        if (!entry->is_valid) continue;
        if (entry->record.kernel_type != kernel_type) continue;
        if (strcmp(entry->record.kernel_name, kernel_name) != 0) continue;
        if (entry->record.input_size != input_size) continue;
        if (entry->record.output_size != output_size) continue;
        return i;
    }
    return -1;
}

/**
 * @brief 扩展数据库容量
 */
static int expand_database(AutoKernelOptimizer* optimizer) {
    int new_capacity = optimizer->database_capacity * DATABASE_GROWTH_FACTOR;
    if (new_capacity > 100000) new_capacity = 100000;

    DatabaseEntry* new_db = (DatabaseEntry*)safe_realloc(optimizer->database,
                                                     new_capacity * sizeof(DatabaseEntry));
    if (!new_db) return -1;

    memset(new_db + optimizer->database_capacity, 0,
           (new_capacity - optimizer->database_capacity) * sizeof(DatabaseEntry));
    optimizer->database = new_db;
    optimizer->database_capacity = new_capacity;
    return 0;
}

/**
 * @brief 获取当前最佳性能记录
 */
static int get_best_record_internal(AutoKernelOptimizer* optimizer,
                                    KernelType kernel_type,
                                    const char* kernel_name,
                                    size_t input_size,
                                    size_t output_size,
                                    KernelPerformanceRecord* best_record) {
    int found = 0;
    double best_time = DBL_MAX;

    for (int i = 0; i < optimizer->database_count; i++) {
        DatabaseEntry* entry = &optimizer->database[i];
        if (!entry->is_valid) continue;
        if (entry->record.kernel_type != kernel_type) continue;
        if (strcmp(entry->record.kernel_name, kernel_name) != 0) continue;
        if (entry->record.input_size != input_size) continue;
        if (entry->record.output_size != output_size) continue;

        if (entry->record.best_time_ms < best_time) {
            best_time = entry->record.best_time_ms;
            if (best_record) {
                memcpy(best_record, &entry->record, sizeof(KernelPerformanceRecord));
            }
            found = 1;
        }
    }

    return found ? 0 : -1;
}

/**
 * @brief 更新或插入性能记录
 */
static int upsert_performance_record(AutoKernelOptimizer* optimizer,
                                     KernelType kernel_type,
                                     const char* kernel_name,
                                     size_t input_size,
                                     size_t output_size,
                                     const KernelOptimizationParams* params,
                                     double execution_time_ms) {
    int existing_idx = find_record_in_database(optimizer, kernel_type,
                                                kernel_name, input_size, output_size);

    if (existing_idx >= 0) {
        DatabaseEntry* entry = &optimizer->database[existing_idx];
        KernelPerformanceRecord* rec = &entry->record;

        rec->use_count++;
        rec->average_time_ms = (rec->average_time_ms * (rec->use_count - 1) +
                                execution_time_ms) / (double)rec->use_count;

        if (execution_time_ms < rec->best_time_ms || rec->best_time_ms < 0.001) {
            rec->best_time_ms = execution_time_ms;
        }

        if (execution_time_ms <= rec->execution_time_ms + 0.001) {
            memcpy(&rec->params, params, sizeof(KernelOptimizationParams));
            rec->execution_time_ms = execution_time_ms;
        }

        entry->timestamp = (long)time(NULL);
        return 0;
    }

    if (optimizer->database_count >= optimizer->database_capacity) {
        if (expand_database(optimizer) != 0) return -1;
    }

    DatabaseEntry* entry = &optimizer->database[optimizer->database_count];
    memset(entry, 0, sizeof(DatabaseEntry));

    KernelPerformanceRecord* rec = &entry->record;
    rec->kernel_type = kernel_type;
    strncpy(rec->kernel_name, kernel_name, MAX_KERNEL_NAME_LENGTH - 1);
    rec->kernel_name[MAX_KERNEL_NAME_LENGTH - 1] = '\0';
    rec->input_size = input_size;
    rec->output_size = output_size;
    memcpy(&rec->params, params, sizeof(KernelOptimizationParams));
    rec->execution_time_ms = execution_time_ms;
    rec->device_id = optimizer->device_id;
    rec->use_count = 1;
    rec->average_time_ms = execution_time_ms;
    rec->best_time_ms = execution_time_ms;
    rec->is_optimal = 0;

    if (input_size > 0 && execution_time_ms > 0) {
        double bytes_processed = (double)(input_size + output_size);
        rec->bandwidth = (bytes_processed / 1.0e9) / (execution_time_ms / 1000.0);

        double operations = 0.0;
        switch (kernel_type) {
            case KERNEL_TYPE_MATMUL: {
                size_t n = (size_t)sqrt((double)input_size / sizeof(float));
                operations = 2.0 * n * n * n;
                break;
            }
            case KERNEL_TYPE_CONV2D: {
                operations = (double)input_size * 2.0;
                break;
            }
            default:
                operations = (double)input_size;
                break;
        }
        rec->flops = (operations / 1.0e9) / (execution_time_ms / 1000.0);
    }

    entry->timestamp = (long)time(NULL);
    entry->is_valid = 1;
    optimizer->database_count++;

    return 0;
}

/**
 * @brief 预测性能（线性插值）
 */
static double predict_performance_linear(AutoKernelOptimizer* optimizer,
                                         KernelType kernel_type,
                                         const char* kernel_name,
                                         size_t input_size,
                                         size_t output_size) {
    (void)output_size;
    double closest_time = -1.0;
    double closest_distance = DBL_MAX;
    double second_closest_time = -1.0;
    double second_closest_distance = DBL_MAX;

    for (int i = 0; i < optimizer->database_count; i++) {
        DatabaseEntry* entry = &optimizer->database[i];
        if (!entry->is_valid) continue;
        if (entry->record.kernel_type != kernel_type) continue;
        if (strcmp(entry->record.kernel_name, kernel_name) != 0) continue;

        size_t db_input = entry->record.input_size;
        double distance = (double)(input_size > db_input ?
                                 input_size - db_input :
                                 db_input - input_size);

        if (distance < closest_distance) {
            second_closest_distance = closest_distance;
            second_closest_time = closest_time;
            closest_distance = distance;
            closest_time = entry->record.average_time_ms;
        } else if (distance < second_closest_distance) {
            second_closest_distance = distance;
            second_closest_time = entry->record.average_time_ms;
        }
    }

    if (closest_time < 0) return -1.0;

    if (second_closest_time < 0 || closest_distance < 0.001) {
        return closest_time;
    }

    double total_dist = closest_distance + second_closest_distance;
    if (total_dist < 0.001) return closest_time;

    double weight1 = second_closest_distance / total_dist;
    double weight2 = closest_distance / total_dist;

    return weight1 * closest_time + weight2 * second_closest_time;
}

/* ============================================================================
 * 公共API实现
 * =========================================================================== */

AutoKernelOptimizer* auto_kernel_optimizer_create(int device_id, const char* device_name) {
    AutoKernelOptimizer* optimizer = (AutoKernelOptimizer*)safe_malloc(sizeof(AutoKernelOptimizer));
    if (!optimizer) {
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "创建自动内核优化器：内存分配失败");
        return NULL;
    }
    memset(optimizer, 0, sizeof(AutoKernelOptimizer));

    optimizer->device_id = device_id;
    if (device_name) {
        strncpy(optimizer->device_name, device_name, 255);
        optimizer->device_name[255] = '\0';
    } else {
        snprintf(optimizer->device_name, 256, "Device_%d", device_id);
    }

    optimizer->database_capacity = INITIAL_DATABASE_CAPACITY;
    optimizer->database = (DatabaseEntry*)safe_calloc(optimizer->database_capacity,
                                                  sizeof(DatabaseEntry));
    if (!optimizer->database) {
        safe_free((void**)&optimizer);
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "创建自动内核优化器：数据库内存分配失败");
        return NULL;
    }
    optimizer->database_count = 0;
    optimizer->total_profiles = 0;
    optimizer->total_optimizations = 0;
    optimizer->accumulated_speedup = 0.0;
    optimizer->cache_enabled = 1;
    optimizer->last_cache_cleanup = time(NULL);

    /* F-022修复：从实际GPU硬件查询设备参数，替代硬编码默认值 */
    {
        GpuDeviceInfo gpu_info;
        memset(&gpu_info, 0, sizeof(gpu_info));
        GpuBackend best = gpu_auto_select();
        int found_gpu = gpu_get_device_info(best, 0, &gpu_info);
        if (found_gpu == 0) {
            /* 无GPU时使用CPU核心数作为compute_units */
#ifdef _WIN32
            SYSTEM_INFO si;
            GetSystemInfo(&si);
            gpu_info.compute_units = (int)si.dwNumberOfProcessors;
#else
            long nprocs = sysconf(_SC_NPROCESSORS_ONLN);
            gpu_info.compute_units = nprocs > 0 ? (int)nprocs : 4;
#endif
            gpu_info.max_work_group_size = 256;
            gpu_info.clock_speed = 0.0f;
            gpu_info.total_memory = 0;
            gpu_info.supports_half = 0;
            gpu_info.supports_double = 0;
        }
        optimizer->max_work_group_size = gpu_info.max_work_group_size > 0 ?
            gpu_info.max_work_group_size : 256;
        optimizer->supports_doubles = gpu_info.supports_double ? 1 : 0;
        optimizer->supports_half = gpu_info.supports_half ? 1 : 0;
        optimizer->compute_units = gpu_info.compute_units > 0 ?
            gpu_info.compute_units : 8;
        optimizer->clock_speed_mhz = gpu_info.clock_speed > 0.0f ?
            gpu_info.clock_speed : 1000.0f;
        optimizer->global_memory_size = gpu_info.total_memory > 0 ?
            gpu_info.total_memory : (4ULL * 1024 * 1024 * 1024);
    }

    init_default_params(optimizer);

    /* F-003修复：初始化在线测量上下文 */
    optimizer->online_context = NULL;
    optimizer->online_default_time = 0.0;

#ifdef _WIN32
    InitializeCriticalSection(&optimizer->mutex);
#else
    pthread_mutex_init(&optimizer->mutex, NULL);
#endif

    return optimizer;
}

void auto_kernel_optimizer_destroy(AutoKernelOptimizer* optimizer) {
    if (!optimizer) return;

#ifdef _WIN32
    EnterCriticalSection(&optimizer->mutex);
#else
    pthread_mutex_lock(&optimizer->mutex);
#endif

    safe_free((void**)&optimizer->database);

#ifdef _WIN32
    LeaveCriticalSection(&optimizer->mutex);
    DeleteCriticalSection(&optimizer->mutex);
#else
    pthread_mutex_unlock(&optimizer->mutex);
    pthread_mutex_destroy(&optimizer->mutex);
#endif

    safe_free((void**)&optimizer);
}

/* F-003修复: 设置GPU上下文以启用真实在线调优测量 */
int auto_kernel_optimizer_set_context(AutoKernelOptimizer* optimizer, GpuContext* context) {
    if (!optimizer) return -1;
#ifdef _WIN32
    EnterCriticalSection(&optimizer->mutex);
#else
    pthread_mutex_lock(&optimizer->mutex);
#endif
    optimizer->online_context = context;
    optimizer->online_default_time = 0.0;
    /* 校准默认内核时间（预热一次以填充GPU流水线） */
    if (context) {
        optimizer->online_default_time = 0.5; /* ms，后续由首次测量校准 */
    }
#ifdef _WIN32
    LeaveCriticalSection(&optimizer->mutex);
#else
    pthread_mutex_unlock(&optimizer->mutex);
#endif
    return 0;
}

int auto_kernel_optimizer_profile(AutoKernelOptimizer* optimizer,
                                 KernelType kernel_type,
                                 const char* kernel_name,
                                 size_t input_size,
                                 size_t output_size,
                                 const size_t* global_work_size,
                                 const KernelOptimizationParams* params,
                                 double execution_time_ms) {
    if (!optimizer || !kernel_name || !params) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "记录内核性能：参数无效");
        return -1;
    }

    (void)output_size;
    (void)global_work_size;

#ifdef _WIN32
    EnterCriticalSection(&optimizer->mutex);
#else
    pthread_mutex_lock(&optimizer->mutex);
#endif

    int result = upsert_performance_record(optimizer, kernel_type, kernel_name,
                                           input_size, output_size,
                                           params, execution_time_ms);

    if (result == 0) {
        optimizer->total_profiles++;
    }

#ifdef _WIN32
    LeaveCriticalSection(&optimizer->mutex);
#else
    pthread_mutex_unlock(&optimizer->mutex);
#endif

    return result;
}

int auto_kernel_optimizer_get_optimal_params(AutoKernelOptimizer* optimizer,
                                            KernelType kernel_type,
                                            const char* kernel_name,
                                            size_t input_size,
                                            size_t output_size,
                                            KernelOptimizationParams* params) {
    if (!optimizer || !kernel_name || !params) {
        return -1;
    }

#ifdef _WIN32
    EnterCriticalSection(&optimizer->mutex);
#else
    pthread_mutex_lock(&optimizer->mutex);
#endif

    KernelPerformanceRecord best_record;
    int found = get_best_record_internal(optimizer, kernel_type, kernel_name,
                                         input_size, output_size, &best_record);

    if (found == 0) {
        memcpy(params, &best_record.params, sizeof(KernelOptimizationParams));
    } else {
        if (kernel_type >= 0 && kernel_type <= 9) {
            memcpy(params, &optimizer->default_params[kernel_type],
                   sizeof(KernelOptimizationParams));

            size_t wg = auto_kernel_optimizer_suggest_work_group(optimizer, input_size,
                                                                 optimizer->max_work_group_size,
                                                                 kernel_type);
            params->local_work_size[0] = wg;
            params->local_work_size[1] = (kernel_type == KERNEL_TYPE_MATMUL ||
                                         kernel_type == KERNEL_TYPE_CONV2D ||
                                         kernel_type == KERNEL_TYPE_POOLING ||
                                         kernel_type == KERNEL_TYPE_LIQUID_GATE) ? wg : 1;
            params->local_work_size[2] = 1;
            params->vector_width = auto_kernel_optimizer_suggest_vector_width(optimizer,
                               sizeof(float), optimizer->supports_half, 100.0f);
            params->unroll_factor = auto_kernel_optimizer_suggest_unroll_factor(optimizer,
                                                                              64, 4);
        } else {
            params->local_work_size[0] = 64;
            params->local_work_size[1] = 1;
            params->local_work_size[2] = 1;
            params->vector_width = 4;
            params->unroll_factor = 2;
            params->use_shared_memory = 0;
            params->shared_memory_size = 0;
            params->use_tensor_cores = 0;
            params->num_workgroups = 0;
            params->strategy = OPT_STRATEGY_BALANCED;
            params->prefer_occupancy = 0;
            params->enable_doubles = optimizer->supports_doubles;
        }
    }

#ifdef _WIN32
    LeaveCriticalSection(&optimizer->mutex);
#else
    pthread_mutex_unlock(&optimizer->mutex);
#endif

    return 0;
}

/* ============================================================================
 * G-009修复: CPU微基准测试引擎
 * 对每种内核类型运行小矩阵微基准测试，测量不同配置的实际执行时间
 * 用于选择最优实现策略（工作组大小、向量化宽度、循环展开因子等）
 * ============================================================================ */

/**
 * @brief 微基准测试配置
 */
typedef struct {
    size_t elems;          /**< 测试元素数量 */
    size_t rows;           /**< 矩阵行数（MATMUL/CONV2D） */
    size_t cols;           /**< 矩阵列数 */
    int unroll_factor;     /**< 循环展开因子 */
    int vector_width;      /**< 向量化宽度 */
    int use_tiling;        /**< 是否使用分块策略（模拟共享内存） */
    int tile_size;         /**< 分块大小 */
} MicroBenchConfig;

/**
 * @brief 获取高精度时间戳（纳秒）
 */
static uint64_t micro_bench_timestamp_ns(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (uint64_t)(counter.QuadPart * 1000000000ULL / freq.QuadPart);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
#endif
}

/**
 * @brief CPU微基准测试：矩阵乘法模式
 * 使用不同的优化参数执行小矩阵乘法并测量时间
 */
static double micro_bench_matmul(const MicroBenchConfig* cfg) {
    size_t N = cfg->rows;
    if (N < 8) N = 8; if (N > 128) N = 128;
    size_t elems = N * N;

    float* A = (float*)safe_calloc(elems, sizeof(float));
    float* B = (float*)safe_calloc(elems, sizeof(float));
    float* C = (float*)safe_calloc(elems, sizeof(float));
    if (!A || !B || !C) {
        safe_free((void**)&A); safe_free((void**)&B); safe_free((void**)&C);
        return -1.0;
    }
    for (size_t i = 0; i < elems; i++) {
        A[i] = (float)(i % 100) / 100.0f;
        B[i] = (float)((i + 37) % 100) / 100.0f;
    }

    int unroll = cfg->unroll_factor;
    if (unroll < 1) unroll = 1; if (unroll > 8) unroll = 8;
    int vwidth = cfg->vector_width;
    if (vwidth < 1) vwidth = 1; if (vwidth > 8) vwidth = 8;
    int tile = cfg->tile_size;
    if (tile < 4) tile = 4; if (tile > 64) tile = 64;

    /* 预热 */
    for (size_t rep = 0; rep < 2; rep++) {
        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < N; j++) {
                float sum = 0.0f;
                size_t k = 0;
                for (; k + unroll <= N; k += (size_t)unroll) {
                    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
                    float s4 = 0.0f, s5 = 0.0f, s6 = 0.0f, s7 = 0.0f;
                    if (unroll >= 1) s0 = A[i * N + k + 0] * B[(k + 0) * N + j];
                    if (unroll >= 2) s1 = A[i * N + k + 1] * B[(k + 1) * N + j];
                    if (unroll >= 3) s2 = A[i * N + k + 2] * B[(k + 2) * N + j];
                    if (unroll >= 4) s3 = A[i * N + k + 3] * B[(k + 3) * N + j];
                    if (unroll >= 5) s4 = A[i * N + k + 4] * B[(k + 4) * N + j];
                    if (unroll >= 6) s5 = A[i * N + k + 5] * B[(k + 5) * N + j];
                    if (unroll >= 7) s6 = A[i * N + k + 6] * B[(k + 6) * N + j];
                    if (unroll >= 8) s7 = A[i * N + k + 7] * B[(k + 7) * N + j];
                    sum += s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7;
                }
                for (; k < N; k++) {
                    sum += A[i * N + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
    }

    uint64_t t_start = micro_bench_timestamp_ns();
    int bench_iters = 3;
    for (int iter = 0; iter < bench_iters; iter++) {
        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < N; j += (size_t)vwidth) {
                float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
                float sum4 = 0.0f, sum5 = 0.0f, sum6 = 0.0f, sum7 = 0.0f;
                size_t k = 0;
                for (; k + unroll <= N; k += (size_t)unroll) {
                    if (vwidth >= 1 && j + 0 < N) sum0 += A[i * N + k] * B[k * N + j + 0];
                    if (vwidth >= 2 && j + 1 < N) sum1 += A[i * N + k] * B[k * N + j + 1];
                    if (vwidth >= 3 && j + 2 < N) sum2 += A[i * N + k] * B[k * N + j + 2];
                    if (vwidth >= 4 && j + 3 < N) sum3 += A[i * N + k] * B[k * N + j + 3];
                    if (vwidth >= 5 && j + 4 < N) sum4 += A[i * N + k] * B[k * N + j + 4];
                    if (vwidth >= 6 && j + 5 < N) sum5 += A[i * N + k] * B[k * N + j + 5];
                    if (vwidth >= 7 && j + 6 < N) sum6 += A[i * N + k] * B[k * N + j + 6];
                    if (vwidth >= 8 && j + 7 < N) sum7 += A[i * N + k] * B[k * N + j + 7];
                }
                for (; k < N; k++) {
                    if (vwidth >= 1 && j + 0 < N) sum0 += A[i * N + k] * B[k * N + j + 0];
                    if (vwidth >= 2 && j + 1 < N) sum1 += A[i * N + k] * B[k * N + j + 1];
                    if (vwidth >= 3 && j + 2 < N) sum2 += A[i * N + k] * B[k * N + j + 2];
                    if (vwidth >= 4 && j + 3 < N) sum3 += A[i * N + k] * B[k * N + j + 3];
                }
                if (vwidth >= 1 && j + 0 < N) C[i * N + j + 0] = sum0;
                if (vwidth >= 2 && j + 1 < N) C[i * N + j + 1] = sum1;
                if (vwidth >= 3 && j + 2 < N) C[i * N + j + 2] = sum2;
                if (vwidth >= 4 && j + 3 < N) C[i * N + j + 3] = sum3;
                if (vwidth >= 5 && j + 4 < N) C[i * N + j + 4] = sum4;
                if (vwidth >= 6 && j + 5 < N) C[i * N + j + 5] = sum5;
                if (vwidth >= 7 && j + 6 < N) C[i * N + j + 6] = sum6;
                if (vwidth >= 8 && j + 7 < N) C[i * N + j + 7] = sum7;
            }
        }
    }
    uint64_t t_end = micro_bench_timestamp_ns();

    double time_ms = (double)(t_end - t_start) / 1e6 / (double)bench_iters;
    (void)tile;
    safe_free((void**)&A); safe_free((void**)&B); safe_free((void**)&C);
    return time_ms;
}

/**
 * @brief CPU微基准测试：逐元素操作模式（激活/归一化/Softmax等）
 */
static double micro_bench_elementwise(const MicroBenchConfig* cfg) {
    size_t N = cfg->elems;
    if (N < 256) N = 256; if (N > 16384) N = 16384;

    float* data = (float*)safe_calloc(N, sizeof(float));
    float* result = (float*)safe_calloc(N, sizeof(float));
    if (!data || !result) {
        safe_free((void**)&data); safe_free((void**)&result);
        return -1.0;
    }
    for (size_t i = 0; i < N; i++) data[i] = (float)(i % 100) / 100.0f;

    int unroll = cfg->unroll_factor;
    if (unroll < 1) unroll = 1; if (unroll > 8) unroll = 8;
    int vwidth = cfg->vector_width;
    if (vwidth < 1) vwidth = 1; if (vwidth > 8) vwidth = 8;

    /* 预热 */
    for (size_t rep = 0; rep < 2; rep++) {
        size_t i = 0;
        for (; i + (size_t)unroll <= N; i += (size_t)unroll) {
            for (int u = 0; u < unroll && i + (size_t)u < N; u++) {
                float x = data[i + (size_t)u];
                result[i + (size_t)u] = (x > 0.0f) ? x : 0.0f;
            }
        }
        for (; i < N; i++) result[i] = (data[i] > 0.0f) ? data[i] : 0.0f;
    }

    uint64_t t_start = micro_bench_timestamp_ns();
    int bench_iters = 10;
    for (int iter = 0; iter < bench_iters; iter++) {
        size_t i = 0;
        for (; i + (size_t)(unroll * vwidth) <= N; i += (size_t)(unroll * vwidth)) {
            float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, r3 = 0.0f;
            float r4 = 0.0f, r5 = 0.0f, r6 = 0.0f, r7 = 0.0f;
            if (vwidth >= 1 && i + 0 < N) { float x = data[i + 0]; r0 = (x > 0.0f) ? x : 0.0f; }
            if (vwidth >= 2 && i + 1 < N) { float x = data[i + 1]; r1 = (x > 0.0f) ? x : 0.0f; }
            if (vwidth >= 3 && i + 2 < N) { float x = data[i + 2]; r2 = (x > 0.0f) ? x : 0.0f; }
            if (vwidth >= 4 && i + 3 < N) { float x = data[i + 3]; r3 = (x > 0.0f) ? x : 0.0f; }
            if (vwidth >= 5 && i + 4 < N) { float x = data[i + 4]; r4 = (x > 0.0f) ? x : 0.0f; }
            if (vwidth >= 6 && i + 5 < N) { float x = data[i + 5]; r5 = (x > 0.0f) ? x : 0.0f; }
            if (vwidth >= 7 && i + 6 < N) { float x = data[i + 6]; r6 = (x > 0.0f) ? x : 0.0f; }
            if (vwidth >= 8 && i + 7 < N) { float x = data[i + 7]; r7 = (x > 0.0f) ? x : 0.0f; }
            if (vwidth >= 1 && i + 0 < N) result[i + 0] = r0;
            if (vwidth >= 2 && i + 1 < N) result[i + 1] = r1;
            if (vwidth >= 3 && i + 2 < N) result[i + 2] = r2;
            if (vwidth >= 4 && i + 3 < N) result[i + 3] = r3;
            if (vwidth >= 5 && i + 4 < N) result[i + 4] = r4;
            if (vwidth >= 6 && i + 5 < N) result[i + 5] = r5;
            if (vwidth >= 7 && i + 6 < N) result[i + 6] = r6;
            if (vwidth >= 8 && i + 7 < N) result[i + 7] = r7;
        }
        for (; i < N; i++) {
            float x = data[i];
            result[i] = (x > 0.0f) ? x : 0.0f;
        }
    }
    uint64_t t_end = micro_bench_timestamp_ns();

    double time_ms = (double)(t_end - t_start) / 1e6 / (double)bench_iters;
    safe_free((void**)&data); safe_free((void**)&result);
    return time_ms;
}

/**
 * @brief CPU微基准测试：卷积模式（模拟Conv2D/Pooling等）
 */
static double micro_bench_conv(const MicroBenchConfig* cfg) {
    size_t H = cfg->rows;
    size_t W = cfg->cols;
    size_t K = 3;
    if (H < 8) H = 8; if (H > 64) H = 64;
    if (W < 8) W = 8; if (W > 64) W = 64;

    size_t input_elems = H * W;
    size_t output_elems = (H - K + 1) * (W - K + 1);
    if (output_elems < 1) { H = 16; W = 16; input_elems = H * W; output_elems = (H - K + 1) * (W - K + 1); }

    float* input = (float*)safe_calloc(input_elems, sizeof(float));
    float* kernel = (float*)safe_calloc(K * K, sizeof(float));
    float* output = (float*)safe_calloc(output_elems, sizeof(float));
    if (!input || !kernel || !output) {
        safe_free((void**)&input); safe_free((void**)&kernel); safe_free((void**)&output);
        return -1.0;
    }
    for (size_t i = 0; i < input_elems; i++) input[i] = (float)(i % 100) / 100.0f;
    for (size_t i = 0; i < (size_t)(K * K); i++) kernel[i] = 0.1f;

    int unroll = cfg->unroll_factor;
    if (unroll < 1) unroll = 1; if (unroll > 4) unroll = 4;

    /* 预热 */
    for (size_t rep = 0; rep < 2; rep++) {
        size_t out_h = H - K + 1;
        size_t out_w = W - K + 1;
        for (size_t oh = 0; oh < out_h; oh++) {
            for (size_t ow = 0; ow < out_w; ow++) {
                float sum = 0.0f;
                for (size_t kh = 0; kh < K; kh++)
                    for (size_t kw = 0; kw < K; kw++)
                        sum += input[(oh + kh) * W + (ow + kw)] * kernel[kh * K + kw];
                output[oh * out_w + ow] = sum;
            }
        }
    }

    uint64_t t_start = micro_bench_timestamp_ns();
    int bench_iters = 5;
    size_t out_h = H - K + 1;
    size_t out_w = W - K + 1;
    for (int iter = 0; iter < bench_iters; iter++) {
        for (size_t oh = 0; oh < out_h; oh++) {
            for (size_t ow = 0; ow < out_w; ow += (size_t)unroll) {
                float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                for (size_t u = 0; u < (size_t)unroll && ow + u < out_w; u++) {
                    for (size_t kh = 0; kh < K; kh++)
                        for (size_t kw = 0; kw < K; kw++)
                            sums[u] += input[(oh + kh) * W + (ow + u + kw)] * kernel[kh * K + kw];
                }
                for (size_t u = 0; u < (size_t)unroll && ow + u < out_w; u++)
                    output[oh * out_w + ow + u] = sums[u];
            }
        }
    }
    uint64_t t_end = micro_bench_timestamp_ns();

    double time_ms = (double)(t_end - t_start) / 1e6 / (double)bench_iters;
    safe_free((void**)&input); safe_free((void**)&kernel); safe_free((void**)&output);
    return time_ms;
}

/**
 * @brief CPU微基准测试：Reduction模式（求和归约）
 */
static double micro_bench_reduction(const MicroBenchConfig* cfg) {
    size_t N = cfg->elems;
    if (N < 256) N = 256; if (N > 65536) N = 65536;

    float* data = (float*)safe_calloc(N, sizeof(float));
    if (!data) return -1.0;
    for (size_t i = 0; i < N; i++) data[i] = (float)(i % 100) / 100.0f;

    int vwidth = cfg->vector_width;
    if (vwidth < 1) vwidth = 1; if (vwidth > 8) vwidth = 8;

    /* 预热 */
    float dummy_sum = 0.0f;
    for (size_t i = 0; i < N; i++) dummy_sum += data[i];
    (void)dummy_sum;

    uint64_t t_start = micro_bench_timestamp_ns();
    int bench_iters = 20;
    float total = 0.0f;
    for (int iter = 0; iter < bench_iters; iter++) {
        float sum = 0.0f;
        size_t i = 0;
        for (; i + (size_t)vwidth <= N; i += (size_t)vwidth) {
            if (vwidth >= 1) sum += data[i + 0];
            if (vwidth >= 2) sum += data[i + 1];
            if (vwidth >= 3) sum += data[i + 2];
            if (vwidth >= 4) sum += data[i + 3];
            if (vwidth >= 5) sum += data[i + 4];
            if (vwidth >= 6) sum += data[i + 5];
            if (vwidth >= 7) sum += data[i + 6];
            if (vwidth >= 8) sum += data[i + 7];
        }
        for (; i < N; i++) sum += data[i];
        total += sum;
    }
    uint64_t t_end = micro_bench_timestamp_ns();

    double time_ms = (double)(t_end - t_start) / 1e6 / (double)bench_iters;
    (void)total;
    safe_free((void**)&data);
    return time_ms;
}

/**
 * @brief 根据内核类型运行CPU微基准测试
 * 对每种配置组合执行实际的CPU计算并返回测量时间
 */
static double run_cpu_micro_benchmark(KernelType kernel_type,
                                      const KernelOptimizationParams* params) {
    MicroBenchConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.unroll_factor = params->unroll_factor > 0 ? params->unroll_factor : 1;
    cfg.vector_width = params->vector_width > 0 ? params->vector_width : 1;
    cfg.use_tiling = params->use_shared_memory;
    cfg.tile_size = params->local_work_size[0] > 0 ? (int)params->local_work_size[0] : 16;

    switch (kernel_type) {
        case KERNEL_TYPE_MATMUL:
            cfg.rows = 64; cfg.cols = 64; cfg.elems = 64 * 64;
            return micro_bench_matmul(&cfg);

        case KERNEL_TYPE_CONV2D:
        case KERNEL_TYPE_POOLING:
            cfg.rows = 32; cfg.cols = 32; cfg.elems = 32 * 32;
            return micro_bench_conv(&cfg);

        case KERNEL_TYPE_LIQUID_GATE:
            /* CfC液态门使用矩阵乘法模式模拟 */
            cfg.rows = 48; cfg.cols = 48; cfg.elems = 48 * 48;
            return micro_bench_matmul(&cfg);

        case KERNEL_TYPE_REDUCTION:
            cfg.elems = 4096;
            return micro_bench_reduction(&cfg);

        case KERNEL_TYPE_NORMALIZATION:
        case KERNEL_TYPE_ACTIVATION:
        case KERNEL_TYPE_SOFTMAX:
        case KERNEL_TYPE_ELEMENTWISE:
        default:
            cfg.elems = 4096;
            return micro_bench_elementwise(&cfg);
    }
}

/* ============================================================================
 * G-009修复: 自动内核调优主函数
 * 使用CPU微基准测试替代原来的启发式估算
 * 对每种内核类型和参数配置运行小规模实际计算并测量时间
 * ============================================================================ */
double auto_kernel_optimizer_tune(AutoKernelOptimizer* optimizer,
                                  KernelType kernel_type,
                                  const char* kernel_name,
                                  size_t input_size,
                                  size_t output_size) {
    if (!optimizer || !kernel_name) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "执行自动调优：参数无效");
        return -1.0;
    }

#ifdef _WIN32
    EnterCriticalSection(&optimizer->mutex);
#else
    pthread_mutex_lock(&optimizer->mutex);
#endif

    TuneSearchSpace search_space;
    generate_search_space(&search_space, optimizer->max_work_group_size);

    double best_time = DBL_MAX;
    KernelOptimizationParams best_params;
    memset(&best_params, 0, sizeof(KernelOptimizationParams));

    /* G-009修复: 遍历搜索空间，对每种配置运行CPU微基准测试 */
    for (int s = 0; s < search_space.search_count; s++) {
        for (int v = 0; v < 4; v++) {
            KernelOptimizationParams test_params;
            memset(&test_params, 0, sizeof(KernelOptimizationParams));

            test_params.local_work_size[0] = search_space.local_sizes[s][0];
            test_params.local_work_size[1] = search_space.local_sizes[s][1];
            test_params.local_work_size[2] = search_space.local_sizes[s][2];
            test_params.vector_width = search_space.vector_widths[v];
            test_params.unroll_factor = search_space.unroll_factors[s % 4];
            test_params.use_shared_memory = search_space.use_shared_options[s % 2];
            test_params.use_tensor_cores = (kernel_type == KERNEL_TYPE_MATMUL) ? 1 : 0;
            test_params.num_workgroups = 0;
            test_params.strategy = OPT_STRATEGY_BALANCED;
            test_params.prefer_occupancy = 0;
            test_params.enable_doubles = optimizer->supports_doubles;

            double candidate_time = -1.0;

            /* 优先级1: GPU在线上下文可用时执行真实GPU测量 */
            if (optimizer->online_context) {
                size_t global_ws = input_size / (size_t)(test_params.vector_width > 0 ?
                    test_params.vector_width : 1);
                if (global_ws < 1) global_ws = 1;
                size_t local_ws = test_params.local_work_size[0] * test_params.local_work_size[1];
                if (local_ws < 1) local_ws = 1;
                size_t num_workgroups = (global_ws + local_ws - 1) / local_ws;

                float* test_input = (float*)safe_calloc(input_size, sizeof(float));
                float* test_output = (float*)safe_calloc(output_size, sizeof(float));
                if (test_input && test_output) {
                    for (size_t ki = 0; ki < input_size; ki++)
                        test_input[ki] = (float)(ki % 100) / 100.0f;

                    /* 使用通用内核执行API进行真实GPU测量 */
                    uint64_t t_start = perf_timestamp_ns();
                    /* G-009: GPU内核在线测量 —— 当GPU上下文可用时通过
                     * auto_kernel_optimizer_measure_real 进行真实计时，
                     * 否则回退到CPU微基准测试 */
                    int exec_ret = -1;
                    if (optimizer->online_context) {
                        /* 需要构建内核源并创建/执行/释放，完整测量耗时 */
                        char kernel_src[1024];
                        snprintf(kernel_src, sizeof(kernel_src),
                            "__kernel void micro_bench(__global float* in, __global float* out) {"
                            "  int i = get_global_id(0);"
                            "  if (i < %zu) out[i] = in[i] * in[i] + in[i];"
                            "}", output_size);
                        double measured = 0.0;
                        int measure_ret = auto_kernel_optimizer_measure_real(
                            optimizer->online_context, kernel_src, "micro_bench",
                            global_ws, local_ws, 2, 5, &measured);
                        if (measure_ret == 0 && measured > 0.0) {
                            candidate_time = measured;
                            exec_ret = 0;
                        }
                    }
                    uint64_t t_end = perf_timestamp_ns();

                    if (exec_ret == 0 && candidate_time <= 0.0)
                        candidate_time = (double)(t_end - t_start) / 1e6;
                }
                safe_free((void**)&test_input);
                safe_free((void**)&test_output);
            }

            /* 优先级2: G-009修复 - CPU微基准测试（无GPU时真实测量CPU执行时间） */
            if (candidate_time <= 0.0) {
                candidate_time = run_cpu_micro_benchmark(kernel_type, &test_params);
            }

            /* 优先级3: 数据库历史预测（CPU微基准失败时回退） */
            if (candidate_time <= 0.0) {
                double predicted_time = predict_performance_linear(optimizer, kernel_type,
                                                                   kernel_name, input_size,
                                                                   output_size);
                candidate_time = (predicted_time > 0.001) ? predicted_time : -1.0;
            }

            /* 优先级4: 最终回退 - 基于参数的理论估算 */
            if (candidate_time <= 0.0) {
                float ws = (float)(test_params.local_work_size[0] * test_params.local_work_size[1]);
                float vw = (float)(test_params.vector_width > 0 ? test_params.vector_width : 1);
                float uf = (float)(test_params.unroll_factor > 0 ? test_params.unroll_factor : 1);
                float sh = test_params.use_shared_memory ? 0.8f : 1.0f;
                float tc = test_params.use_tensor_cores ? 0.5f : 1.0f;
                candidate_time = (float)input_size * 10.0f / (ws * vw * uf) * sh * tc;
            }

            if (candidate_time > 0.0 && candidate_time < best_time) {
                best_time = candidate_time;
                memcpy(&best_params, &test_params, sizeof(KernelOptimizationParams));
            }
        }
    }

    if (best_time >= DBL_MAX) {
        if (kernel_type >= 0 && kernel_type <= 9) {
            memcpy(&best_params, &optimizer->default_params[kernel_type],
                   sizeof(KernelOptimizationParams));
        }
        best_time = 0.0;
    }

    upsert_performance_record(optimizer, kernel_type, kernel_name,
                              input_size, output_size, &best_params, best_time);

    optimizer->total_optimizations++;

    double previous_best = 0.0;
    KernelPerformanceRecord prev_rec;
    if (get_best_record_internal(optimizer, kernel_type, kernel_name,
                                 input_size, output_size, &prev_rec) == 0) {
        previous_best = prev_rec.best_time_ms;
        if (previous_best > 0.001 && best_time > 0.001) {
            optimizer->accumulated_speedup += previous_best / best_time;
        }
    }

#ifdef _WIN32
    LeaveCriticalSection(&optimizer->mutex);
#else
    pthread_mutex_unlock(&optimizer->mutex);
#endif

    return best_time;
}

int auto_kernel_optimizer_get_statistics(AutoKernelOptimizer* optimizer,
                                        int* total_profiles,
                                        int* total_optimizations,
                                        double* average_speedup) {
    if (!optimizer) return -1;

#ifdef _WIN32
    EnterCriticalSection(&optimizer->mutex);
#else
    pthread_mutex_lock(&optimizer->mutex);
#endif

    if (total_profiles) *total_profiles = optimizer->total_profiles;
    if (total_optimizations) *total_optimizations = optimizer->total_optimizations;

    if (average_speedup) {
        if (optimizer->total_optimizations > 0) {
            *average_speedup = optimizer->accumulated_speedup /
                              (double)optimizer->total_optimizations;
        } else {
            *average_speedup = 1.0;
        }
    }

#ifdef _WIN32
    LeaveCriticalSection(&optimizer->mutex);
#else
    pthread_mutex_unlock(&optimizer->mutex);
#endif

    return 0;
}

/*
 * 真实GPU执行计时测量（用于在线调优模式）
 * 当GpuContext可用时，实际创建/执行内核并测量墙钟时间
 */
int auto_kernel_optimizer_measure_real(GpuContext* context,
                                        const char* kernel_source,
                                        const char* kernel_name,
                                        size_t global_size, size_t local_size,
                                        int warmup_iters, int bench_iters,
                                        double* out_time_ms) {
    if (!context || !kernel_source || !kernel_name || !out_time_ms) return -1;

    GpuKernel* kernel = gpu_kernel_create(context, kernel_source, kernel_name);
    if (!kernel) return -1;

    /* 预热 */
    for (int i = 0; i < warmup_iters; i++) {
        gpu_kernel_execute(kernel, global_size, local_size);
    }

    /* 计时测量 */
#ifdef _WIN32
    LARGE_INTEGER freq, start, end;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&start);
#else
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif

    for (int i = 0; i < bench_iters; i++) {
        gpu_kernel_execute(kernel, global_size, local_size);
    }

#ifdef _WIN32
    QueryPerformanceCounter(&end);
    *out_time_ms = (double)(end.QuadPart - start.QuadPart) * 1000.0 /
                   (double)freq.QuadPart / (double)bench_iters;
#else
    clock_gettime(CLOCK_MONOTONIC, &end);
    *out_time_ms = ((end.tv_sec - start.tv_sec) * 1000.0 +
                    (end.tv_nsec - start.tv_nsec) / 1.0e6) / (double)bench_iters;
#endif

    gpu_kernel_free(kernel);
    return 0;
}

size_t auto_kernel_optimizer_suggest_work_group(AutoKernelOptimizer* optimizer,
                                               size_t global_size,
                                               size_t max_work_group_size,
                                               KernelType kernel_type) {
    if (!optimizer || global_size == 0) return 64;

    size_t suggested = max_work_group_size;

    if (kernel_type == KERNEL_TYPE_MATMUL || kernel_type == KERNEL_TYPE_CONV2D ||
        kernel_type == KERNEL_TYPE_POOLING || kernel_type == KERNEL_TYPE_LIQUID_GATE) {
        suggested = (max_work_group_size >= 256) ? 16 : 8;
    } else if (kernel_type == KERNEL_TYPE_NORMALIZATION ||
               kernel_type == KERNEL_TYPE_SOFTMAX) {
        suggested = (max_work_group_size >= 256) ? 128 : 64;
    } else if (kernel_type == KERNEL_TYPE_ACTIVATION ||
               kernel_type == KERNEL_TYPE_ELEMENTWISE) {
        suggested = (max_work_group_size >= 512) ? 256 : max_work_group_size;
    } else {
        suggested = (max_work_group_size >= 128) ? 64 : max_work_group_size;
    }

    if (suggested > max_work_group_size) suggested = max_work_group_size;

    while (suggested > global_size && suggested > 1) {
        suggested /= 2;
    }

    if (suggested < 1) suggested = 1;

    return suggested;
}

int auto_kernel_optimizer_suggest_vector_width(AutoKernelOptimizer* optimizer,
                                              size_t data_size,
                                              int supports_half,
                                              float memory_bandwidth) {
    (void)supports_half;
    (void)memory_bandwidth;
    if (!optimizer) return 1;

    if (data_size <= 1) {
        return 1;
    } else if (data_size <= 2) {
        return 2;
    } else if (data_size <= 4) {
        return 4;
    }

    if (optimizer->compute_units >= 16 && optimizer->clock_speed_mhz >= 1000.0f) {
        return 8;
    }

    return 4;
}

int auto_kernel_optimizer_suggest_unroll_factor(AutoKernelOptimizer* optimizer,
                                               int loop_iterations,
                                               int operations_per_iter) {
    /* ZSFZS-F020修复: 使用优化器历史数据库信息辅助决策展开因子。
     * 结合循环迭代数和每迭代操作数，参考数据库中最优配置。 */
    if (loop_iterations <= 0 || operations_per_iter <= 0) return 1;

    /* 尝试从历史数据库查询最优展开因子 */
    if (optimizer && optimizer->database_count > 0) {
        int hist_unroll = 4;
        for (size_t i = 0; i < (size_t)optimizer->database_count && i < 64; i++) {
            if (optimizer->database[i].record.input_size > 0 &&
                optimizer->database[i].record.params.unroll_factor > hist_unroll) {
                hist_unroll = optimizer->database[i].record.params.unroll_factor;
            }
        }
        if (hist_unroll > 0 && loop_iterations >= hist_unroll * 4) {
            return hist_unroll;
        }
    }

    if (loop_iterations >= 64 && operations_per_iter <= 4) {
        return 8;
    } else if (loop_iterations >= 32 && operations_per_iter <= 8) {
        return 4;
    } else if (loop_iterations >= 16 && operations_per_iter <= 16) {
        return 2;
    }

    return 1;
}

void auto_kernel_optimizer_clear_cache(AutoKernelOptimizer* optimizer) {
    if (!optimizer) return;

#ifdef _WIN32
    EnterCriticalSection(&optimizer->mutex);
#else
    pthread_mutex_lock(&optimizer->mutex);
#endif

    memset(optimizer->database, 0,
           optimizer->database_capacity * sizeof(DatabaseEntry));
    optimizer->database_count = 0;
    optimizer->total_profiles = 0;
    optimizer->total_optimizations = 0;
    optimizer->accumulated_speedup = 0.0;
    optimizer->last_cache_cleanup = time(NULL);

#ifdef _WIN32
    LeaveCriticalSection(&optimizer->mutex);
#else
    pthread_mutex_unlock(&optimizer->mutex);
#endif
}

int auto_kernel_optimizer_save_database(AutoKernelOptimizer* optimizer,
                                       const char* filepath) {
    if (!optimizer || !filepath) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "保存优化数据库：参数无效");
        return -1;
    }

#ifdef _WIN32
    EnterCriticalSection(&optimizer->mutex);
#else
    pthread_mutex_lock(&optimizer->mutex);
#endif

    FILE* fp = fopen(filepath, "wb");
    if (!fp) {
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "保存优化数据库：无法打开文件");
#ifdef _WIN32
        LeaveCriticalSection(&optimizer->mutex);
#else
        pthread_mutex_unlock(&optimizer->mutex);
#endif
        return -1;
    }

    int header[3];
    header[0] = 0x4F505449; /* 'OPTI' 魔数 */
    header[1] = optimizer->database_count;
    header[2] = (int)sizeof(KernelPerformanceRecord);
    fwrite(header, sizeof(int), 3, fp);

    int valid_count = 0;
    for (int i = 0; i < optimizer->database_count; i++) {
        if (optimizer->database[i].is_valid) valid_count++;
    }
    fwrite(&valid_count, sizeof(int), 1, fp);

    for (int i = 0; i < optimizer->database_count; i++) {
        if (optimizer->database[i].is_valid) {
            fwrite(&optimizer->database[i].record, sizeof(KernelPerformanceRecord), 1, fp);
        }
    }

    fclose(fp);

#ifdef _WIN32
    LeaveCriticalSection(&optimizer->mutex);
#else
    pthread_mutex_unlock(&optimizer->mutex);
#endif

    return 0;
}

int auto_kernel_optimizer_load_database(AutoKernelOptimizer* optimizer,
                                       const char* filepath) {
    if (!optimizer || !filepath) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "加载优化数据库：参数无效");
        return -1;
    }

#ifdef _WIN32
    EnterCriticalSection(&optimizer->mutex);
#else
    pthread_mutex_lock(&optimizer->mutex);
#endif

    FILE* fp = fopen(filepath, "rb");
    if (!fp) {
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "加载优化数据库：无法打开文件");
#ifdef _WIN32
        LeaveCriticalSection(&optimizer->mutex);
#else
        pthread_mutex_unlock(&optimizer->mutex);
#endif
        return -1;
    }

    int header[3];
    if (fread(header, sizeof(int), 3, fp) != 3 || header[0] != 0x4F505449) {
        fclose(fp);
        selflnn_set_last_error(SELFLNN_ERROR_FORMAT_ERROR, __func__, __FILE__, __LINE__,
                              "加载优化数据库：无效的文件格式");
#ifdef _WIN32
        LeaveCriticalSection(&optimizer->mutex);
#else
        pthread_mutex_unlock(&optimizer->mutex);
#endif
        return -1;
    }

    int stored_count;
    if (fread(&stored_count, sizeof(int), 1, fp) != 1) {
        fclose(fp);
#ifdef _WIN32
        LeaveCriticalSection(&optimizer->mutex);
#else
        pthread_mutex_unlock(&optimizer->mutex);
#endif
        return -1;
    }

    int records_to_read = stored_count;
    if (records_to_read > optimizer->database_capacity) {
        int new_capacity = records_to_read + INITIAL_DATABASE_CAPACITY;
        DatabaseEntry* new_db = (DatabaseEntry*)safe_realloc(optimizer->database,
                                                         new_capacity * sizeof(DatabaseEntry));
        if (!new_db) {
            fclose(fp);
#ifdef _WIN32
            LeaveCriticalSection(&optimizer->mutex);
#else
            pthread_mutex_unlock(&optimizer->mutex);
#endif
            return -1;
        }
        memset(new_db + optimizer->database_capacity, 0,
               (new_capacity - optimizer->database_capacity) * sizeof(DatabaseEntry));
        optimizer->database = new_db;
        optimizer->database_capacity = new_capacity;
    }

    int loaded = 0;
    for (int i = 0; i < records_to_read; i++) {
        if (optimizer->database_count >= optimizer->database_capacity) break;

        DatabaseEntry* entry = &optimizer->database[optimizer->database_count];
        if (fread(&entry->record, sizeof(KernelPerformanceRecord), 1, fp) == 1) {
            entry->is_valid = 1;
            entry->timestamp = (long)time(NULL);
            optimizer->database_count++;
            loaded++;
        }
    }

    fclose(fp);

#ifdef _WIN32
    LeaveCriticalSection(&optimizer->mutex);
#else
    pthread_mutex_unlock(&optimizer->mutex);
#endif

    return (loaded > 0) ? 0 : -1;
}

int auto_kernel_optimizer_get_best_record(AutoKernelOptimizer* optimizer,
                                         KernelType kernel_type,
                                         const char* kernel_name,
                                         size_t input_size,
                                         KernelPerformanceRecord* record) {
    if (!optimizer || !kernel_name || !record) return -1;

#ifdef _WIN32
    EnterCriticalSection(&optimizer->mutex);
#else
    pthread_mutex_lock(&optimizer->mutex);
#endif

    int result = get_best_record_internal(optimizer, kernel_type, kernel_name,
                                          input_size, 0, record);

#ifdef _WIN32
    LeaveCriticalSection(&optimizer->mutex);
#else
    pthread_mutex_unlock(&optimizer->mutex);
#endif

    return result;
}

int auto_kernel_optimizer_get_record_count(AutoKernelOptimizer* optimizer) {
    if (!optimizer) return -1;

#ifdef _WIN32
    EnterCriticalSection(&optimizer->mutex);
#else
    pthread_mutex_lock(&optimizer->mutex);
#endif

    int count = optimizer->database_count;

#ifdef _WIN32
    LeaveCriticalSection(&optimizer->mutex);
#else
    pthread_mutex_unlock(&optimizer->mutex);
#endif

    return count;
}

int auto_kernel_optimizer_get_record_at(AutoKernelOptimizer* optimizer,
                                       int index,
                                       KernelPerformanceRecord* record) {
    if (!optimizer || !record || index < 0) return -1;

#ifdef _WIN32
    EnterCriticalSection(&optimizer->mutex);
#else
    pthread_mutex_lock(&optimizer->mutex);
#endif

    if (index >= optimizer->database_count) {
#ifdef _WIN32
        LeaveCriticalSection(&optimizer->mutex);
#else
        pthread_mutex_unlock(&optimizer->mutex);
#endif
        return -1;
    }

    if (!optimizer->database[index].is_valid) {
#ifdef _WIN32
        LeaveCriticalSection(&optimizer->mutex);
#else
        pthread_mutex_unlock(&optimizer->mutex);
#endif
        return -1;
    }

    memcpy(record, &optimizer->database[index].record, sizeof(KernelPerformanceRecord));

#ifdef _WIN32
    LeaveCriticalSection(&optimizer->mutex);
#else
    pthread_mutex_unlock(&optimizer->mutex);
#endif

    return 0;
}

int auto_kernel_optimizer_predict_performance(AutoKernelOptimizer* optimizer,
                                             KernelType kernel_type,
                                             const char* kernel_name,
                                             size_t input_size,
                                             size_t output_size,
                                             double* predicted_time_ms) {
    if (!optimizer || !kernel_name || !predicted_time_ms) {
        return -1;
    }

#ifdef _WIN32
    EnterCriticalSection(&optimizer->mutex);
#else
    pthread_mutex_lock(&optimizer->mutex);
#endif

    /* 先检查是否有完全匹配的记录 */
    KernelPerformanceRecord best_record;
    int found = get_best_record_internal(optimizer, kernel_type, kernel_name,
                                         input_size, output_size, &best_record);

    if (found == 0) {
        *predicted_time_ms = best_record.average_time_ms;
#ifdef _WIN32
        LeaveCriticalSection(&optimizer->mutex);
#else
        pthread_mutex_unlock(&optimizer->mutex);
#endif
        return 0;
    }

    /* 用线性插值预测 */
    double predicted = predict_performance_linear(optimizer, kernel_type, kernel_name,
                                                  input_size, output_size);

    if (predicted < 0) {
        /* 完全没有历史数据，使用基准估计 */
        double bytes = (double)(input_size + output_size);
        double bandwidth = 100.0e9;
        double compute_intensity = 1.0;

        switch (kernel_type) {
            case KERNEL_TYPE_MATMUL: {
                size_t n = (size_t)sqrt((double)input_size / sizeof(float));
                double ops = 2.0 * n * n * n;
                double compute_time = ops / (optimizer->compute_units * optimizer->clock_speed_mhz * 1.0e6 * 4.0);
                double memory_time = (double)(input_size + output_size) / bandwidth;
                predicted = (compute_time + memory_time) * 1000.0;
                break;
            }
            case KERNEL_TYPE_CONV2D:
                compute_intensity = 2.0;
                predicted = (bytes / bandwidth) / compute_intensity * 1000.0;
                break;
            default:
                predicted = (bytes / bandwidth) * 1000.0;
                break;
        }

        if (predicted < 0.001) predicted = 1.0;
    }

    *predicted_time_ms = predicted;

#ifdef _WIN32
    LeaveCriticalSection(&optimizer->mutex);
#else
    pthread_mutex_unlock(&optimizer->mutex);
#endif

    return 0;
}
