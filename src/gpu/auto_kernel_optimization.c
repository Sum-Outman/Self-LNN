/**
 * @file auto_kernel_optimization.c
 * @brief 自动内核优化系统实现
 *
 * 实现GPU内核自动调优引擎，包含性能分析数据库、自动参数选择、
 * 工作组大小优化、向量化宽度自适应和性能预测功能。
 */


#include "selflnn/gpu/auto_kernel_optimization.h"
#include "selflnn/core/errors.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/string_utils.h"
#include <stdlib.h>
#include <string.h>
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
    optimizer->max_work_group_size = 256;
    optimizer->supports_doubles = 0;
    optimizer->supports_half = 1;
    optimizer->compute_units = 8;
    optimizer->clock_speed_mhz = 1000.0f;
    optimizer->global_memory_size = 4ULL * 1024 * 1024 * 1024;

    init_default_params(optimizer);

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

            double predicted_time = predict_performance_linear(optimizer, kernel_type,
                                                               kernel_name, input_size,
                                                               output_size);

            /* 如果数据库中有接近的配置，使用预测值；否则标记需要真实测量 */
            double candidate_time = (predicted_time > 0.001) ? predicted_time : -1.0;

            /* 当预测不可靠时（数据库记录稀少），使用启发式估算 */
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
    (void)optimizer;
    if (loop_iterations <= 0 || operations_per_iter <= 0) return 1;

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
