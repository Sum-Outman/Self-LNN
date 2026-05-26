/**
 * @file api_training.c
 * @brief API训练实现
 * 
 * 通过API进行模型训练的实现。
 */

#define _CRT_NONSTDC_NO_DEPRECATE

#include "selflnn/knowledge/api_training.h"
#include "selflnn/utils/secure_random.h"
#include "selflnn/training/training.h"
#include "selflnn/knowledge/knowledge.h"
#include "selflnn/utils/string_utils.h"
#include "selflnn/utils/json_parser.h"
#include "selflnn/core/errors.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

/* ============================================================================
 * HTTPS/TLS支持：动态加载系统OpenSSL
 * 编译时零依赖，运行时检测libssl可用性
 * ============================================================================ */
#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
typedef HMODULE TLS_LIB_HANDLE;
#define TLS_DLOPEN(path) LoadLibraryA(path)
#define TLS_DLSYM(lib, sym) GetProcAddress((HMODULE)(lib), sym)
#define TLS_DLCLOSE(lib) FreeLibrary((HMODULE)(lib))
static const char* ssl_lib_paths[] = {
    "libssl-3-x64.dll", "libssl-3.dll", "libssl-1_1-x64.dll",
    "libssl-1_1.dll", "ssleay32.dll", NULL
};
static const char* crypto_lib_paths[] = {
    "libcrypto-3-x64.dll", "libcrypto-3.dll", "libcrypto-1_1-x64.dll",
    "libcrypto-1_1.dll", "libeay32.dll", NULL
};
#else
#include <dlfcn.h>
#include <sys/socket.h>
typedef void* TLS_LIB_HANDLE;
#define TLS_DLOPEN(path) dlopen(path, RTLD_LAZY)
#define TLS_DLSYM(lib, sym) dlsym(lib, sym)
#define TLS_DLCLOSE(lib) dlclose(lib)
static const char* ssl_lib_paths[] = {
    "libssl.so.3", "libssl.so.1.1", "libssl.so.30",
    "libssl.so.10", "libssl.so", NULL
};
static const char* crypto_lib_paths[] = {
    "libcrypto.so.3", "libcrypto.so.1.1", "libcrypto.so.30",
    "libcrypto.so.10", "libcrypto.so", NULL
};
#endif

/* OpenSSL函数指针类型（最小接口子集） */
typedef struct ssl_st SSL;
typedef struct ssl_ctx_st SSL_CTX;
typedef struct ssl_method_st SSL_METHOD;
typedef unsigned long (*TLS_ERR_get_error_t)(void);
typedef void (*TLS_ERR_error_string_n_t)(unsigned long, char*, size_t);
typedef SSL_CTX* (*TLS_CTX_new_t)(const SSL_METHOD*);
typedef void (*TLS_CTX_free_t)(SSL_CTX*);
typedef int (*TLS_CTX_set_verify_t)(SSL_CTX*, int, void*);
typedef SSL* (*TLS_new_t)(SSL_CTX*);
typedef void (*TLS_free_t)(SSL*);
typedef int (*TLS_set_fd_t)(SSL*, int);
typedef int (*TLS_connect_t)(SSL*);
typedef int (*TLS_read_t)(SSL*, void*, int);
typedef int (*TLS_write_t)(SSL*, const void*, int);
typedef int (*TLS_shutdown_t)(SSL*);
typedef const SSL_METHOD* (*TLS_client_method_t)(void);

static struct {
    TLS_LIB_HANDLE ssl_lib;
    TLS_LIB_HANDLE crypto_lib;
    TLS_CTX_new_t CTX_new;
    TLS_CTX_free_t CTX_free;
    TLS_CTX_set_verify_t CTX_set_verify;
    TLS_new_t SSL_new;
    TLS_free_t SSL_free;
    TLS_set_fd_t SSL_set_fd;
    TLS_connect_t SSL_connect;
    TLS_read_t SSL_read;
    TLS_write_t SSL_write;
    TLS_shutdown_t SSL_shutdown;
    TLS_client_method_t TLS_client_method;
    TLS_ERR_get_error_t ERR_get_error;
    TLS_ERR_error_string_n_t ERR_error_string_n;
    int available;
} g_tls;

/* I-017修复: TLS通过dlopen动态加载系统SSL库(OpenSSL/LibreSSL/BoringSSL)
 * 设计原则：零编译期硬编码 — 运行时探测多路径，失败时优雅降级到纯HTTP
 * ssl_lib_paths/crypto_lib_paths 在文件头部定义，按平台差异配置 */

static int tls_init(void) {
    if (g_tls.available) return 0;
    if (g_tls.ssl_lib) return -1; /* already tried and failed */
    for (int i = 0; ssl_lib_paths[i]; i++) {
        g_tls.ssl_lib = TLS_DLOPEN(ssl_lib_paths[i]);
        if (g_tls.ssl_lib) break;
    }
    for (int i = 0; crypto_lib_paths[i]; i++) {
        g_tls.crypto_lib = TLS_DLOPEN(crypto_lib_paths[i]);
        if (g_tls.crypto_lib) break;
    }
    if (!g_tls.ssl_lib && !g_tls.crypto_lib) return -1;
    TLS_LIB_HANDLE tls_lib = g_tls.ssl_lib ? g_tls.ssl_lib : g_tls.crypto_lib;
    #define TLS_LOAD_VOID(fn, name) g_tls.fn = (void*)TLS_DLSYM(tls_lib, name)
    if (g_tls.ssl_lib) {
        TLS_LOAD_VOID(TLS_client_method, "TLS_client_method");
        TLS_LOAD_VOID(CTX_new, "SSL_CTX_new");
        TLS_LOAD_VOID(CTX_free, "SSL_CTX_free");
        TLS_LOAD_VOID(SSL_new, "SSL_new");
        TLS_LOAD_VOID(SSL_free, "SSL_free");
        TLS_LOAD_VOID(SSL_set_fd, "SSL_set_fd");
        TLS_LOAD_VOID(SSL_connect, "SSL_connect");
        TLS_LOAD_VOID(SSL_read, "SSL_read");
        TLS_LOAD_VOID(SSL_write, "SSL_write");
        TLS_LOAD_VOID(SSL_shutdown, "SSL_shutdown");
    }
    if (g_tls.crypto_lib) {
        g_tls.ERR_get_error = (void*)TLS_DLSYM(g_tls.crypto_lib, "ERR_get_error");
        g_tls.ERR_error_string_n = (void*)TLS_DLSYM(g_tls.crypto_lib, "ERR_error_string_n");
        if (!g_tls.ssl_lib) {
            g_tls.TLS_client_method = (void*)TLS_DLSYM(g_tls.crypto_lib, "TLS_client_method");
            g_tls.CTX_new = (void*)TLS_DLSYM(g_tls.crypto_lib, "SSL_CTX_new");
        }
    }
    if (!g_tls.SSL_new || !g_tls.SSL_connect || !g_tls.TLS_client_method || !g_tls.CTX_new) {
        return -1;
    }
    g_tls.SSL_set_fd = g_tls.SSL_set_fd ? g_tls.SSL_set_fd : NULL;
    g_tls.SSL_read = g_tls.SSL_read ? g_tls.SSL_read : NULL;
    g_tls.SSL_write = g_tls.SSL_write ? g_tls.SSL_write : NULL;
    g_tls.CTX_set_verify = g_tls.CTX_set_verify ? g_tls.CTX_set_verify : NULL;
    g_tls.available = 1;
    return 0;
}

/* TLS包裹socket: 返回SSL*或NULL（失败时自动close socket） */
static SSL* tls_wrap_socket(int fd, const char* host, SocketHandle* sock_out) {
    if (tls_init() != 0) return NULL;
    SSL_CTX* ctx = g_tls.CTX_new(g_tls.TLS_client_method());
    if (!ctx) return NULL;
    if (g_tls.CTX_set_verify) g_tls.CTX_set_verify(ctx, 0, NULL);
    SSL* ssl = g_tls.SSL_new(ctx);
    if (!ssl) { g_tls.CTX_free(ctx); return NULL; }
    if (g_tls.SSL_set_fd) g_tls.SSL_set_fd(ssl, fd);
    if (g_tls.SSL_connect(ssl) != 1) {
        long err = g_tls.ERR_get_error ? g_tls.ERR_get_error() : 0;
        char errbuf[256] = {0};
        if (g_tls.ERR_error_string_n && err) g_tls.ERR_error_string_n(err, errbuf, sizeof(errbuf));
        g_tls.SSL_free(ssl); g_tls.CTX_free(ctx);
        return NULL;
    }
    g_tls.CTX_free(ctx);
    *sock_out = fd;
    return ssl;
}

static void tls_unwrap(SSL* ssl, int fd) {
    if (!ssl) return;
    if (g_tls.SSL_shutdown) g_tls.SSL_shutdown(ssl);
    g_tls.SSL_free(ssl);
    (void)fd;
}

static int tls_send(SSL* ssl, const void* data, size_t len) {
    if (!ssl || !g_tls.SSL_write) return -1;
    return g_tls.SSL_write(ssl, data, (int)len);
}

static int tls_recv(SSL* ssl, void* buf, size_t len) {
    if (!ssl || !g_tls.SSL_read) return -1;
    return g_tls.SSL_read(ssl, buf, (int)len);
}

#undef TLS_LOAD_VOID

/**
 * @brief 训练会话内部结构
 */
struct TrainingSession {
    TrainingConfig config;           /**< 训练配置 */
    TrainingStatus status;           /**< 当前状态 */
    
    /* 训练状态 */
    int current_iteration;           /**< 当前迭代次数 */
    float current_accuracy;          /**< 当前准确率 */
    float current_loss;              /**< 当前损失 */
    float progress_percentage;       /**< 进度百分比 */
    
    /* 训练数据 */
    void* training_data;             /**< 训练数据 */
    size_t training_data_size;       /**< 训练数据大小 */
    
    /* 训练结果 */
    TrainingResult result;           /**< 训练结果 */
    
    /* 回调函数 */
    void (*callback)(TrainingStateInfo*, void*); /**< 回调函数 */
    void* callback_user_data;        /**< 回调用户数据 */
    
    /* 内部训练器 */
    Trainer* trainer;                /**< 训练器句柄 */
    
    /* 关联的神经网络 */
    LNN* network;                    /**< 要训练的神经网络 */
    
    /* 时间统计 */
    clock_t start_time;              /**< 开始时间 */
    clock_t end_time;                /**< 结束时间 */
};

/* 辅助函数 */

/**
 * @brief 更新训练状态
 * 
 * @param session 训练会话
 * @param status 新状态
 * @param message 状态消息
 */
static void update_training_status(TrainingSession* session,
                                  TrainingStatus status,
                                  const char* message) {
    if (session == NULL) {
        return;
    }
    
    session->status = status;
    
    /* 调用回调函数 */
    if (session->callback != NULL) {
        TrainingStateInfo state;
        memset(&state, 0, sizeof(TrainingStateInfo));
        
        state.status = status;
        state.current_iteration = session->current_iteration;
        state.current_accuracy = session->current_accuracy;
        state.current_loss = session->current_loss;
        state.progress_percentage = session->progress_percentage;
        
        if (message != NULL) {
            state.message = string_duplicate(message);
        }
        
        session->callback(&state, session->callback_user_data);
        
        if (state.message != NULL) {
            safe_free((void**)&state.message);
        }
    }
}

/**
 * @brief 计算进度百分比
 * 
 * @param session 训练会话
 * @return float 进度百分比
 */
static float calculate_progress(const TrainingSession* session) {
    if (session == NULL || session->config.epochs <= 0) {
        return 0.0f;
    }
    
    return (float)session->current_iteration * 100.0f / 
           (float)session->config.epochs;
}

/* 公共API实现 */

TrainingSession* training_session_create(const TrainingConfig* config) {
    if (config == NULL) {
        return NULL;
    }
    
    TrainingSession* session = (TrainingSession*)safe_malloc(sizeof(TrainingSession));
    if (session == NULL) {
        return NULL;
    }
    
    /* 复制配置 */
    memcpy(&session->config, config, sizeof(TrainingConfig));
    
    /* 初始化状态 */
    session->status = TRAINING_STATUS_IDLE;
    session->current_iteration = 0;
    session->current_accuracy = 0.0f;
    session->current_loss = 0.0f;
    session->progress_percentage = 0.0f;
    
    /* 初始化数据 */
    session->training_data = NULL;
    session->training_data_size = 0;
    
    /* 初始化结果 */
    memset(&session->result, 0, sizeof(TrainingResult));
    
    /* 初始化回调 */
    session->callback = NULL;
    session->callback_user_data = NULL;
    
    /* 初始化训练器 */
    session->trainer = NULL;
    session->network = NULL;
    
    /* 初始化时间 */
    session->start_time = 0;
    session->end_time = 0;
    
    return session;
}

void training_session_free(TrainingSession* session) {
    if (session == NULL) {
        return;
    }
    
    /* 停止训练 */
    if (session->status == TRAINING_STATUS_RUNNING ||
        session->status == TRAINING_STATUS_PAUSED) {
        training_session_stop(session);
    }
    
    /* 释放训练数据 */
    if (session->training_data != NULL) {
        safe_free((void**)&session->training_data);
    }
    
    /* 释放训练结果 */
    if (session->result.model_data != NULL) {
        safe_free((void**)&session->result.model_data);
    }
    
    /* 释放训练器 */
    if (session->trainer != NULL) {
        trainer_free(session->trainer);
        session->trainer = NULL;
    }
    
    /* 注意：network由调用者管理，这里只置空 */
    session->network = NULL;
    
    safe_free((void**)&session);
}

int training_session_start(TrainingSession* session, 
                          const void* training_data, size_t data_size) {
    if (session == NULL || training_data == NULL || data_size == 0) {
        return -1;
    }
    
    /* 检查状态 */
    if (session->status != TRAINING_STATUS_IDLE &&
        session->status != TRAINING_STATUS_COMPLETED &&
        session->status != TRAINING_STATUS_FAILED) {
        return -1;
    }
    
    /* 释放旧数据 */
    if (session->training_data != NULL) {
        safe_free((void**)&session->training_data);
    }
    
    /* 复制训练数据 */
    session->training_data = safe_malloc(data_size);
    if (session->training_data == NULL) {
        return -1;
    }
    
    memcpy(session->training_data, training_data, data_size);
    session->training_data_size = data_size;
    
    /* 重置状态 */
    session->current_iteration = 0;
    session->current_accuracy = 0.0f;
    session->current_loss = 0.0f;
    session->progress_percentage = 0.0f;
    
    /* 创建训练器 */
    if (session->trainer != NULL) {
        trainer_free(session->trainer);
        session->trainer = NULL;
    }
    
    /* 检查是否有可用的神经网络 */
    if (session->network == NULL) {
        update_training_status(session, TRAINING_STATUS_FAILED, "未设置神经网络");
        return -1;
    }
    
    /* 使用真实训练器 */
    session->trainer = trainer_create(&session->config, session->network);
    if (session->trainer == NULL) {
        update_training_status(session, TRAINING_STATUS_FAILED, "创建训练器失败");
        return -1;
    }
    
    /* 更新状态 */
    update_training_status(session, TRAINING_STATUS_RUNNING, "训练开始");
    
    /* 记录开始时间 */
    session->start_time = clock();

    /* 实际启动训练：使用训练器在训练数据上执行真实的训练循环 */
    if (session->trainer && session->training_data) {
        float* inputs = NULL;
        float* targets = NULL;
        size_t sample_size = session->config.input_size + session->config.output_size;
        size_t num_samples = session->training_data_size / (sample_size * sizeof(float));

        if (num_samples > 0) {
            inputs = (float*)safe_malloc(num_samples * session->config.input_size * sizeof(float));
            targets = (float*)safe_malloc(num_samples * session->config.output_size * sizeof(float));

            if (inputs && targets) {
                const float* raw_data = (const float*)session->training_data;
                for (size_t i = 0; i < num_samples; i++) {
                    memcpy(inputs + i * session->config.input_size,
                           raw_data + i * sample_size,
                           session->config.input_size * sizeof(float));
                    memcpy(targets + i * session->config.output_size,
                           raw_data + i * sample_size + session->config.input_size,
                           session->config.output_size * sizeof(float));
                }

                int train_result = trainer_train(session->trainer, inputs, targets,
                                                  num_samples, NULL, NULL);

                if (train_result == 0) {
                    TrainingState* state = trainer_get_state(session->trainer);
                    if (state) {
                        session->current_loss = state->current_loss;
                        session->current_accuracy = state->current_accuracy;
                    }
                    session->progress_percentage = 1.0f;
                    update_training_status(session, TRAINING_STATUS_COMPLETED, "训练完成");
                } else {
                    update_training_status(session, TRAINING_STATUS_FAILED, "训练执行失败");
                }

                safe_free((void**)&inputs);
                safe_free((void**)&targets);
                return train_result;
            }

            safe_free((void**)&inputs);
            safe_free((void**)&targets);
        }
    }
    
    return 0;
}

int training_session_pause(TrainingSession* session) {
    if (session == NULL) {
        return -1;
    }
    
    if (session->status != TRAINING_STATUS_RUNNING) {
        return -1;
    }
    
    /* 更新状态 */
    update_training_status(session, TRAINING_STATUS_PAUSED, "训练暂停");
    
    return 0;
}

int training_session_resume(TrainingSession* session) {
    if (session == NULL) {
        return -1;
    }
    
    if (session->status != TRAINING_STATUS_PAUSED) {
        return -1;
    }
    
    /* 更新状态 */
    update_training_status(session, TRAINING_STATUS_RUNNING, "训练恢复");
    
    return 0;
}

int training_session_stop(TrainingSession* session) {
    if (session == NULL) {
        return -1;
    }
    
    if (session->status != TRAINING_STATUS_RUNNING &&
        session->status != TRAINING_STATUS_PAUSED) {
        return -1;
    }
    
    /* 更新状态 */
    update_training_status(session, TRAINING_STATUS_IDLE, "训练停止");
    
    /* 记录结束时间 */
    session->end_time = clock();
    
    /* 计算训练时间 */
    if (session->start_time > 0 && session->end_time > 0) {
        session->result.training_time_ms = 
            (long)((session->end_time - session->start_time) * 1000 / CLOCKS_PER_SEC);
    }
    
    return 0;
}

int training_session_get_state(TrainingSession* session, TrainingStateInfo* state) {
    if (session == NULL || state == NULL) {
        return -1;
    }
    
    state->status = session->status;
    state->current_iteration = session->current_iteration;
    state->current_accuracy = session->current_accuracy;
    state->current_loss = session->current_loss;
    state->progress_percentage = session->progress_percentage;
    
    /* 根据状态设置消息 */
    const char* message = NULL;
    switch (session->status) {
        case TRAINING_STATUS_IDLE:
            message = "空闲";
            break;
        case TRAINING_STATUS_RUNNING:
            message = "运行中";
            break;
        case TRAINING_STATUS_PAUSED:
            message = "暂停";
            break;
        case TRAINING_STATUS_COMPLETED:
            message = "完成";
            break;
        case TRAINING_STATUS_FAILED:
            message = "失败";
            break;
        default:
            message = "未知状态";
            break;
    }
    
    if (message != NULL) {
        state->message = string_duplicate(message);
    }
    
    return 0;
}

int training_session_get_result(TrainingSession* session, TrainingResult* result) {
    if (session == NULL || result == NULL) {
        return -1;
    }
    
    /* 复制结果 */
    memcpy(result, &session->result, sizeof(TrainingResult));
    
    /* 复制模型数据 */
    if (session->result.model_data != NULL && session->result.model_data_size > 0) {
        result->model_data = safe_malloc(session->result.model_data_size);
        if (result->model_data == NULL) {
            return -1;
        }
        memcpy(result->model_data, session->result.model_data, 
               session->result.model_data_size);
        result->model_data_size = session->result.model_data_size;
    }
    
    return 0;
}

int training_session_export_model(TrainingSession* session, 
                                 void* model_data, size_t max_size) {
    if (session == NULL || model_data == NULL || max_size == 0) {
        return -1;
    }
    
    if (session->result.model_data == NULL || session->result.model_data_size == 0) {
        return -1;
    }
    
    if (session->result.model_data_size > max_size) {
        return -1;
    }
    
    memcpy(model_data, session->result.model_data, session->result.model_data_size);
    
    return (int)session->result.model_data_size;
}

int training_session_import_model(TrainingSession* session,
                                 const void* model_data, size_t data_size) {
    if (session == NULL || model_data == NULL || data_size == 0) {
        return -1;
    }
    
    /* 释放旧模型数据 */
    if (session->result.model_data != NULL) {
        safe_free((void**)&session->result.model_data);
    }
    
    /* 复制新模型数据 */
    session->result.model_data = safe_malloc(data_size);
    if (session->result.model_data == NULL) {
        return -1;
    }
    
    memcpy(session->result.model_data, model_data, data_size);
    session->result.model_data_size = data_size;
    
    /* 更新状态 */
    session->status = TRAINING_STATUS_COMPLETED;
    session->result.final_accuracy = session->current_accuracy;
    session->result.final_loss = session->current_loss;
    session->result.total_iterations = session->current_iteration;
    
    return 0;
}

int training_session_set_callback(TrainingSession* session,
                                 void (*callback)(TrainingStateInfo*, void*),
                                 void* user_data) {
    if (session == NULL) {
        return -1;
    }
    
    session->callback = callback;
    session->callback_user_data = user_data;
    
    return 0;
}

int training_session_set_network(TrainingSession* session, LNN* network) {
    if (session == NULL) {
        return -1;
    }
    
    session->network = network;
    return 0;
}

/* ============================================================================
 * OpenAI兼容适配器实现
 * ============================================================================
 *
 * 纯C实现OpenAI API兼容的HTTP客户端，支持：
 * - Chat Completions (POST /v1/chat/completions)
 * - Fine-tuning (POST /v1/fine_tuning/jobs)
 * - List Models (GET /v1/models)
 * - 多提供者支持（OpenAI / Azure / 自定义 / 本地）
 * - 超时重试机制
 * - 平台套接字抽象（platform.h）
 *
 * 不依赖任何第三方HTTP/JSON库。
 * ============================================================================
 */

#include "selflnn/utils/platform.h"

/* 平台相关的套接字头文件 */
#if defined(_WIN32) || defined(_WIN64)
#include <winsock2.h>
#else
#include <sys/socket.h>
#endif

/* ---------- HTTP请求/响应常量 ---------- */
#define OPENAI_HTTP_BUF_SIZE        (64 * 1024)
#define OPENAI_MAX_RESPONSE_SIZE    (256 * 1024)
#define OPENAI_DEFAULT_TIMEOUT_MS   30000
#define OPENAI_MAX_REDIRECTS        5

/* ---------- 指数退避重试常量 ---------- */
#define OPENAI_RETRY_BASE_DELAY_MS  200
#define OPENAI_RETRY_MAX_DELAY_MS   30000
#define OPENAI_RETRY_JITTER_RANGE   0.5f
#define OPENAI_MAX_RETRIES          5
#define OPENAI_RETRY_STATUS_LIMIT   10

/* ---------- 内部辅助：URL解析 ---------- */

/**
 * @brief 解析URL为host/port/path
 *
 * 支持格式: http://host:port/path, https://host:port/path, host:port/path
 *
 * @param url 输入URL
 * @param host 输出主机名缓冲区
 * @param host_size 主机名缓冲区大小
 * @param port 输出端口号
 * @param path 输出路径缓冲区
 * @param path_size 路径缓冲区大小
 * @return int 成功返回0，失败返回-1
 */
static int parse_api_url(const char* url, char* host, size_t host_size,
                         unsigned short* port, char* path, size_t path_size) {
    if (!url || !host || !port || !path) return -1;

    const char* p = url;
    int is_https = 0;

    /* 跳过协议前缀 */
    if (strncmp(p, "https://", 8) == 0) {
        is_https = 1;
        p += 8;
    } else if (strncmp(p, "http://", 7) == 0) {
        p += 7;
    }

    /* 提取主机名 */
    const char* host_start = p;
    const char* colon = strchr(p, ':');
    const char* slash = strchr(p, '/');

    if (colon && (!slash || colon < slash)) {
        /* 有端口号: host:port/path */
        size_t host_len = (size_t)(colon - host_start);
        if (host_len >= host_size) host_len = host_size - 1;
        strncpy(host, host_start, host_len);
        host[host_len] = '\0';

        /* 解析端口 */
        long port_long = strtol(colon + 1, NULL, 10);
        if (port_long <= 0 || port_long > 65535) {
            *port = (unsigned short)(is_https ? 443 : 80);
        } else {
            *port = (unsigned short)port_long;
        }
        p = colon + 1;
        /* 跳过端口数字 */
        while (*p && *p != '/') p++;
    } else if (slash) {
        /* 无端口但有路径: host/path */
        size_t host_len = (size_t)(slash - host_start);
        if (host_len >= host_size) host_len = host_size - 1;
        strncpy(host, host_start, host_len);
        host[host_len] = '\0';
        *port = (unsigned short)(is_https ? 443 : 80);
    } else {
        /* 只有主机: host */
        size_t host_len = strlen(host_start);
        if (host_len >= host_size) host_len = host_size - 1;
        strncpy(host, host_start, host_len);
        host[host_len] = '\0';
        *port = (unsigned short)(is_https ? 443 : 80);
    }

    /* 提取路径 */
    if (*p == '/') {
        size_t path_len = strlen(p);
        if (path_len >= path_size) path_len = path_size - 1;
        strncpy(path, p, path_len);
        path[path_len] = '\0';
    } else {
        path[0] = '/';
        path[1] = '\0';
    }

    return is_https ? 1 : 0; /* 返回1表示HTTPS，0表示HTTP */
}

/* ---------- HTTP客户端（纯Socket） ---------- */

/**
 * @brief 构建HTTP请求并发送
 *
 * @param method HTTP方法 (GET/POST)
 * @param host 主机名
 * @param port 端口
 * @param path URL路径
 * @param headers 额外请求头（每行以\r\n结尾，可NULL）
 * @param body 请求体（可NULL）
 * @param body_len 请求体长度
 * @param response 输出响应缓冲区（调用者负责free）
 * @return int 成功返回0，失败返回-1
 */
static int http_request(const char* method, const char* host, unsigned short port,
                        const char* path, const char* headers,
                        const char* body, size_t body_len,
                        DynamicString* response, int is_https) {
    if (!method || !host || !path || !response) return -1;

    /* 创建套接字 */
    SocketHandle sock = socket_create(0, 1, 0); /* AF_INET, SOCK_STREAM */
    if (sock < 0) return -1;

    /* 设置超时 */
    int timeout_ms = OPENAI_DEFAULT_TIMEOUT_MS;
#if defined(_WIN32) || defined(_WIN64)
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout_ms, sizeof(timeout_ms));
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, (const char*)&timeout_ms, sizeof(timeout_ms));
#else
    struct timeval tv;
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
#endif

    /* 连接服务器 */
    if (socket_connect(sock, host, port) != 0) {
        socket_close(sock);
        return -1;
    }

    /* HTTPS: TLS握手 */
    SSL* tls_ssl = NULL;
    if (is_https) {
        tls_ssl = tls_wrap_socket((int)sock, host, &sock);
        if (!tls_ssl) {
            socket_close(sock);
            return -1;
        }
    }

    /* 构建HTTP请求 */
    DynamicString* request_buf = string_dynamic_create(4096);
    if (!request_buf) {
        socket_close(sock);
        return -1;
    }

    string_dynamic_append(request_buf, method);
    string_dynamic_append(request_buf, " ");
    string_dynamic_append(request_buf, path);
    string_dynamic_append(request_buf, " HTTP/1.1\r\n");
    string_dynamic_append(request_buf, "Host: ");
    string_dynamic_append(request_buf, host);

    /* 非标准端口添加端口号到Host头 */
    if (port != 80 && port != 443) {
        char port_str[16];
        string_format_safe(port_str, sizeof(port_str), ":%hu", port);
        string_dynamic_append(request_buf, port_str);
    }
    string_dynamic_append(request_buf, "\r\n");

    string_dynamic_append(request_buf, "User-Agent: Self-LNN/1.0\r\n");
    string_dynamic_append(request_buf, "Accept: application/json\r\n");
    string_dynamic_append(request_buf, "Connection: close\r\n");

    if (body && body_len > 0) {
        char content_length[32];
        string_format_safe(content_length, sizeof(content_length),
                          "Content-Length: %zu\r\n", body_len);
        string_dynamic_append(request_buf, content_length);
        string_dynamic_append(request_buf, "Content-Type: application/json\r\n");
    }

    if (headers) {
        string_dynamic_append(request_buf, headers);
    }

    string_dynamic_append(request_buf, "\r\n");

    if (body && body_len > 0) {
        string_dynamic_append(request_buf, body);
    }

    /* 发送请求 */
    const char* req_data = string_dynamic_cstr(request_buf);
    size_t req_len = string_dynamic_length(request_buf);
    int sent = tls_ssl ? tls_send(tls_ssl, req_data, req_len) : socket_send(sock, req_data, req_len);
    string_dynamic_free(request_buf);

    if (sent < 0 || (size_t)sent != req_len) {
        if (tls_ssl) tls_unwrap(tls_ssl, (int)sock);
        socket_close(sock);
        return -1;
    }

    /* 接收响应 */
#define HTTP_RECV(buf, sz) (tls_ssl ? tls_recv(tls_ssl, (buf), (sz)) : socket_recv(sock, (buf), (sz)))
#define HTTP_CLOSE() do { if(tls_ssl) tls_unwrap(tls_ssl, (int)sock); socket_close(sock); } while(0)
    char recv_buf[4096];
    int total = 0;

    /* 先接收响应头 */
    char header_buf[8192];
    int header_len = 0;
    int headers_done = 0;
    int content_length_val = -1;
    int chunked = 0;
    int http_status = 0;

    while (header_len < (int)sizeof(header_buf) - 1) {
        int n = HTTP_RECV(recv_buf, sizeof(recv_buf) - 1);
        if (n <= 0) break;

        recv_buf[n] = '\0';

        if (!headers_done) {
            /* 将数据追加到header_buf */
            int copy_len = n;
            if (header_len + copy_len >= (int)sizeof(header_buf) - 1) {
                copy_len = (int)sizeof(header_buf) - 1 - header_len;
            }
            memcpy(header_buf + header_len, recv_buf, copy_len);
            header_len += copy_len;
            header_buf[header_len] = '\0';

            /* 查找头部结束标记 */
            char* header_end = strstr(header_buf, "\r\n\r\n");
            if (header_end) {
                headers_done = 1;

                /* 解析HTTP状态码 */
                if (strncmp(header_buf, "HTTP/", 5) == 0) {
                    http_status = (int)strtol(header_buf + 5, NULL, 10);
                    /* 跳过 "HTTP/x.x " */
                    const char* sp = strchr(header_buf, ' ');
                    if (sp) http_status = (int)strtol(sp + 1, NULL, 10);
                }

                /* 解析Content-Length */
                const char* cl = strstr(header_buf, "Content-Length:");
                if (cl) {
                    const char* val = cl + 15;
                    while (*val == ' ') val++;
                    content_length_val = (int)strtol(val, NULL, 10);
                }

                /* 检查chunked */
                if (strstr(header_buf, "Transfer-Encoding: chunked") != NULL ||
                    strstr(header_buf, "transfer-encoding: chunked") != NULL) {
                    chunked = 1;
                }

                /* 将头部之后的数据（如果有）写入响应 */
                const char* body_start = header_end + 4;
                size_t body_remaining = (size_t)(header_len - (body_start - header_buf));
                if (body_remaining > 0) {
                    string_dynamic_append(response, body_start);
                    total += (int)body_remaining;
                }
            }
        } else {
            /* 已经解析完头部，直接追加数据 */
            string_dynamic_append(response, recv_buf);
            total += n;
        }

        if (headers_done && content_length_val >= 0 && total >= content_length_val) {
            break;
        }
    }

    /* 如果还没接收完，继续接收 */
    if (headers_done && !chunked && content_length_val > 0) {
        while (total < content_length_val) {
            int n = HTTP_RECV(recv_buf, sizeof(recv_buf) - 1);
            if (n <= 0) break;
            recv_buf[n] = '\0';
            string_dynamic_append(response, recv_buf);
            total += n;
        }
    } else if (chunked) {
        /* chunked编码：已在上面的循环中接收完成（因为Connection: close） */
        /* 但可能还需要额外接收 */
        char extra[4096];
        int n;
        while ((n = HTTP_RECV(extra, sizeof(extra) - 1)) > 0) {
            extra[n] = '\0';
            string_dynamic_append(response, extra);
        }
    }

    HTTP_CLOSE();
    return (http_status >= 200 && http_status < 300) ? 0 : -1;
#undef HTTP_RECV
#undef HTTP_CLOSE
}

/* ---------- 指数退避重试辅助 ---------- */

/**
 * @brief 计算指数退避重试延迟（含完整抖动）
 *
 * 使用 Full Jitter 策略：random(0, min(baseDelay * 2^attempt, maxDelay))
 * 避免惊群效应，同时保证最大延迟上限
 *
 * @param attempt 当前重试次数（从0开始）
 * @return unsigned int 延迟毫秒数
 */
static unsigned int get_retry_delay_ms(int attempt) {
    if (attempt < 0) attempt = 0;
    if (attempt > 10) attempt = 10;

    unsigned long long exponential = (unsigned long long)OPENAI_RETRY_BASE_DELAY_MS << attempt;
    if (exponential > OPENAI_RETRY_MAX_DELAY_MS) {
        exponential = OPENAI_RETRY_MAX_DELAY_MS;
    }
    /* K-012修复：使用安全随机数进行指数退避抖动 */
    double jitter = (double)secure_random_float();
    unsigned long long delay = (unsigned long long)((double)exponential * jitter);
    if (delay > OPENAI_RETRY_MAX_DELAY_MS) {
        delay = OPENAI_RETRY_MAX_DELAY_MS;
    }
    if (delay < 1) delay = 1;
    return (unsigned int)delay;
}

/**
 * @brief 判断HTTP状态码是否可重试
 *
 * 可重试状态码: 429 (限流), 500+, 以及连接超时/重置
 *
 * @param http_status HTTP状态码
 * @return int 1=可重试，0=不可重试
 */
static int is_retryable_status(int http_status) {
    if (http_status == 429) return 1;
    if (http_status >= 500 && http_status < 600) return 1;
    return 0;
}

/* ---------- JSON手动构建/解析 ---------- */

/**
 * @brief JSON转义字符串（分配新内存）
 */
static char* json_escape_string(const char* input) {
    if (!input) return string_duplicate("");

    size_t len = strlen(input);
    /* 预计算转义后长度 */
    size_t escaped_len = 0;
    for (size_t i = 0; i < len; i++) {
        switch (input[i]) {
            case '"': case '\\': case '\b': case '\f':
            case '\n': case '\r': case '\t':
                escaped_len += 2;
                break;
            default:
                escaped_len++;
                break;
        }
    }

    char* result = (char*)safe_malloc(escaped_len + 1);
    if (!result) return NULL;

    size_t j = 0;
    for (size_t i = 0; i < len; i++) {
        switch (input[i]) {
            case '"':  result[j++] = '\\'; result[j++] = '"';  break;
            case '\\': result[j++] = '\\'; result[j++] = '\\'; break;
            case '\b': result[j++] = '\\'; result[j++] = 'b';  break;
            case '\f': result[j++] = '\\'; result[j++] = 'f';  break;
            case '\n': result[j++] = '\\'; result[j++] = 'n';  break;
            case '\r': result[j++] = '\\'; result[j++] = 'r';  break;
            case '\t': result[j++] = '\\'; result[j++] = 't';  break;
            default:   result[j++] = input[i];                  break;
        }
    }
    result[j] = '\0';
    return result;
}

/**
 * @brief 从JSON字符串中提取字符串字段值
 *
 * 查找 "key": "value" 模式，返回value的副本
 *
 * @param json JSON字符串
 * @param key 字段名
 * @return char* 字段值副本，未找到返回NULL
 */
static char* json_extract_string(const char* json, const char* key) {
    if (!json || !key) return NULL;

    /* 构建搜索模式: "key": " */
    DynamicString* pattern = string_dynamic_create(strlen(key) + 8);
    if (!pattern) return NULL;
    string_dynamic_append(pattern, "\"");
    string_dynamic_append(pattern, key);
    string_dynamic_append(pattern, "\": \"");
    const char* pat = string_dynamic_cstr(pattern);

    const char* start = strstr(json, pat);
    if (!start) {
        /* 尝试带空格版本: "key" : " */
        string_dynamic_clear(pattern);
        string_dynamic_append(pattern, "\"");
        string_dynamic_append(pattern, key);
        string_dynamic_append(pattern, "\" : \"");
        pat = string_dynamic_cstr(pattern);
        start = strstr(json, pat);
    }
    if (!start) {
        string_dynamic_free(pattern);
        return NULL;
    }

    start += strlen(pat);
    const char* end = strchr(start, '"');
    if (!end) {
        string_dynamic_free(pattern);
        return NULL;
    }

    size_t val_len = (size_t)(end - start);
    char* value = (char*)safe_malloc(val_len + 1);
    if (!value) {
        string_dynamic_free(pattern);
        return NULL;
    }
    memcpy(value, start, val_len);
    value[val_len] = '\0';
    string_dynamic_free(pattern);
    return value;
}

/**
 * @brief 从JSON字符串中提取整数字段值
 *
 * @param json JSON字符串
 * @param key 字段名
 * @param default_val 默认值
 * @return int 整数值
 */
static int json_extract_int(const char* json, const char* key, int default_val) {
    if (!json || !key) return default_val;

    char search_key[256];
    string_format_safe(search_key, sizeof(search_key), "\"%s\":", key);
    const char* start = strstr(json, search_key);
    if (!start) return default_val;

    start += strlen(search_key);
    while (*start == ' ') start++;

    if (*start == '-' || (*start >= '0' && *start <= '9')) {
        return (int)strtol(start, NULL, 10);
    }
    return default_val;
}

/* ---------- OpenAI适配器内部结构 ---------- */

struct OpenAIAdapter {
    OpenAIConfig config;        /**< 适配器配置 */
    int initialized;            /**< 是否已初始化 */
    int last_http_status;       /**< 上次HTTP状态码 */
    char last_error[512];       /**< 上次错误消息 */
};

/* ---------- 辅助：构建Authorization头 ---------- */

/**
 * @brief 根据提供者类型构建Authorization请求头
 *
 * @param adapter 适配器句柄
 * @return char* 请求头字符串（以\r\n结尾），调用者负责释放
 */
static char* build_auth_header(const OpenAIAdapter* adapter) {
    if (!adapter || !adapter->config.api_key[0]) {
        return NULL;
    }

    DynamicString* header = string_dynamic_create(256);

    switch (adapter->config.provider) {
        case OPENAI_PROVIDER_AZURE:
            /* Azure使用api-key头 */
            string_dynamic_append(header, "api-key: ");
            string_dynamic_append(header, adapter->config.api_key);
            string_dynamic_append(header, "\r\n");
            break;
        case OPENAI_PROVIDER_OPENAI:
        case OPENAI_PROVIDER_CUSTOM:
        case OPENAI_PROVIDER_LOCAL:
        default:
            string_dynamic_append(header, "Authorization: Bearer ");
            string_dynamic_append(header, adapter->config.api_key);
            string_dynamic_append(header, "\r\n");
            break;
    }

    char* result = string_duplicate(string_dynamic_cstr(header));
    string_dynamic_free(header);
    return result;
}

/* ---------- 辅助：构建聊天补全请求体 ---------- */

/**
 * @brief 构建聊天补全请求JSON体
 *
 * @param adapter 适配器句柄
 * @param messages 消息数组JSON字符串
 * @return char* JSON请求体，调用者负责释放
 */
static char* build_chat_body(const OpenAIAdapter* adapter, const char* messages) {
    if (!adapter || !messages) return NULL;

    DynamicString* body = string_dynamic_create(4096);
    if (!body) return NULL;

    string_dynamic_append(body, "{\n");
    string_dynamic_append(body, "  \"model\": \"");
    string_dynamic_append(body, adapter->config.model);
    string_dynamic_append(body, "\",\n");
    string_dynamic_append(body, "  \"messages\": ");
    string_dynamic_append(body, messages);
    string_dynamic_append(body, ",\n");

    if (adapter->config.temperature > 0.0f) {
        char temp[16];
        string_format_safe(temp, sizeof(temp), "%.2f", adapter->config.temperature);
        string_dynamic_append(body, "  \"temperature\": ");
        string_dynamic_append(body, temp);
        string_dynamic_append(body, ",\n");
    }

    if (adapter->config.top_p > 0.0f && adapter->config.top_p < 1.0f) {
        char top_p[16];
        string_format_safe(top_p, sizeof(top_p), "%.2f", adapter->config.top_p);
        string_dynamic_append(body, "  \"top_p\": ");
        string_dynamic_append(body, top_p);
        string_dynamic_append(body, ",\n");
    }

    if (adapter->config.max_tokens > 0) {
        char max_tok[16];
        string_format_safe(max_tok, sizeof(max_tok), "%d", adapter->config.max_tokens);
        string_dynamic_append(body, "  \"max_tokens\": ");
        string_dynamic_append(body, max_tok);
        string_dynamic_append(body, ",\n");
    }

    if (adapter->config.presence_penalty != 0.0f) {
        char pp[16];
        string_format_safe(pp, sizeof(pp), "%.2f", adapter->config.presence_penalty);
        string_dynamic_append(body, "  \"presence_penalty\": ");
        string_dynamic_append(body, pp);
        string_dynamic_append(body, ",\n");
    }

    if (adapter->config.frequency_penalty != 0.0f) {
        char fp[16];
        string_format_safe(fp, sizeof(fp), "%.2f", adapter->config.frequency_penalty);
        string_dynamic_append(body, "  \"frequency_penalty\": ");
        string_dynamic_append(body, fp);
        string_dynamic_append(body, ",\n");
    }

    if (adapter->config.stop_count > 0) {
        string_dynamic_append(body, "  \"stop\": [");
        for (int i = 0; i < adapter->config.stop_count; i++) {
            if (i > 0) string_dynamic_append(body, ", ");
            string_dynamic_append(body, "\"");
            string_dynamic_append(body, adapter->config.stop_sequences[i]);
            string_dynamic_append(body, "\"");
        }
        string_dynamic_append(body, "],\n");
    }

    if (adapter->config.stream_enabled) {
        string_dynamic_append(body, "  \"stream\": true,\n");
    }

    /* 移除末尾的逗号和换行，并关闭JSON */
    size_t len = string_dynamic_length(body);
    const char* data = string_dynamic_cstr(body);
    /* 回退去掉最后一个逗号 */
    if (len >= 2 && data[len - 2] == ',') {
        /* 直接修改内部缓冲区 */
        ((char*)data)[len - 2] = '\n';
        ((char*)data)[len - 1] = '}';
        ((char*)data)[len] = '\0';
        body->length = len;
    } else {
        string_dynamic_append(body, "}");
    }

    char* result = string_duplicate(string_dynamic_cstr(body));
    string_dynamic_free(body);
    return result;
}

/* ---------- 辅助：解析聊天补全响应 ---------- */

/**
 * @brief 解析OpenAI聊天补全响应
 *
 * @param response_text HTTP响应体
 * @param result OpenAIResponse输出
 */
static void parse_chat_response(const char* response_text, OpenAIResponse* result) {
    if (!response_text || !result) return;

    JsonValue* root = json_parse(response_text);
    if (root) {
        JsonValue* choices = json_get(root, "choices");
        if (choices && choices->type == JSON_ARRAY && choices->data.array.count > 0) {
            JsonValue* choice0 = choices->data.array.items[0];
            JsonValue* message = json_get(choice0, "message");
            if (message) {
                const char* c = json_get_string(message, "content");
                if (c) { result->content = string_duplicate(c); result->content_length = strlen(c); }
            }
            const char* fr = json_get_string(choice0, "finish_reason");
            if (fr) result->finish_reason = string_duplicate(fr);
        }
        JsonValue* usage = json_get(root, "usage");
        if (usage) {
            result->prompt_tokens = (int)json_get_number(usage, "prompt_tokens");
            result->completion_tokens = (int)json_get_number(usage, "completion_tokens");
        }
        JsonValue* err = json_get(root, "error");
        if (err) {
            const char* msg = json_get_string(err, "message");
            if (msg) string_copy_safe(result->error_message, msg, sizeof(result->error_message));
        }
        json_free(root);
        return;
    }

    /* 回退到旧解析器 */
    char* content = json_extract_string(response_text, "content");
    if (content) { result->content = content; result->content_length = strlen(content); }
    char* finish = json_extract_string(response_text, "finish_reason");
    if (finish) result->finish_reason = finish;
    result->prompt_tokens = json_extract_int(response_text, "prompt_tokens", 0);
    result->completion_tokens = json_extract_int(response_text, "completion_tokens", 0);
    char* error = json_extract_string(response_text, "message");
    if (error) {
        string_copy_safe(result->error_message, error, sizeof(result->error_message));
        safe_free((void**)&error);
    }
}

/* ---------- 辅助：解析模型列表响应 ---------- */

/**
 * @brief 从模型列表响应中提取可读信息
 *
 * @param response_text HTTP响应体
 * @param result OpenAIResponse输出
 */
static void parse_models_response(const char* response_text, OpenAIResponse* result) {
    if (!response_text || !result) return;

    JsonValue* root = json_parse(response_text);
    if (root) {
        JsonValue* data = json_get(root, "data");
        if (data && data->type == JSON_ARRAY && data->data.array.count > 0) {
            const char* id = json_get_string(data->data.array.items[0], "id");
            if (id) { result->content = string_duplicate(id); result->content_length = strlen(id); }
        }
        JsonValue* err = json_get(root, "error");
        if (err) {
            const char* msg = json_get_string(err, "message");
            if (msg) string_copy_safe(result->error_message, msg, sizeof(result->error_message));
        }
        json_free(root);
        return;
    }

    /* 回退 */
    char* first_id = json_extract_string(response_text, "id");
    if (first_id) { result->content = first_id; result->content_length = strlen(first_id); }
    char* error = json_extract_string(response_text, "message");
    if (error) {
        string_copy_safe(result->error_message, error, sizeof(result->error_message));
        safe_free((void**)&error);
    }
}

/* ---------- OpenAI适配器公共API实现 ---------- */

OpenAIAdapter* openai_adapter_create(const OpenAIConfig* config) {
    if (!config) return NULL;

    /* 验证配置 */
    if (!config->api_base[0]) return NULL;
    if (!config->model[0]) return NULL;

    OpenAIAdapter* adapter = (OpenAIAdapter*)safe_malloc(sizeof(OpenAIAdapter));
    if (!adapter) return NULL;

    memcpy(&adapter->config, config, sizeof(OpenAIConfig));
    adapter->initialized = 1;
    adapter->last_http_status = 0;
    adapter->last_error[0] = '\0';

    return adapter;
}

void openai_adapter_free(OpenAIAdapter* adapter) {
    if (!adapter) return;
    adapter->initialized = 0;
    safe_free((void**)&adapter);
}

int openai_adapter_chat_completion(OpenAIAdapter* adapter,
                                    const char* messages,
                                    OpenAIResponse* response) {
    if (!adapter || !adapter->initialized || !messages || !response) {
        return -1;
    }

    /* 初始化响应 */
    memset(response, 0, sizeof(OpenAIResponse));

    /* 解析URL */
    char host[256];
    unsigned short port;
    char path[512];
    int is_https = parse_api_url(adapter->config.api_base, host, sizeof(host),
                                  &port, path, sizeof(path));

    /* 构建Authorization头 */
    char* auth_header = build_auth_header(adapter);

    /* 构建请求体 */
    char* body = build_chat_body(adapter, messages);
    if (!body) {
        if (auth_header) safe_free((void**)&auth_header);
        response->success = 0;
        response->http_status = 0;
        string_copy_safe(response->error_message,
                        "构建请求体失败", sizeof(response->error_message));
        return -1;
    }

    /* 组合额外的请求头 */
    DynamicString* extra_headers = string_dynamic_create(256);
    if (auth_header) {
        string_dynamic_append(extra_headers, auth_header);
        safe_free((void**)&auth_header);
    }

    /* 发送请求（带指数退避+抖动重试） */
    int retries = adapter->config.max_retries > 0 ? adapter->config.max_retries : OPENAI_MAX_RETRIES;
    int http_ok = 0;
    DynamicString* http_response = NULL;

    for (int retry = 0; retry < retries; retry++) {
        if (http_response) string_dynamic_free(http_response);
        http_response = string_dynamic_create(OPENAI_HTTP_BUF_SIZE);
        if (!http_response) {
            string_dynamic_free(extra_headers);
            safe_free((void**)&body);
            return -1;
        }

        /* Azure需要额外的api-version参数 */
        const char* api_path = path;
        char azure_path[512];
        if (adapter->config.provider == OPENAI_PROVIDER_AZURE) {
            string_format_safe(azure_path, sizeof(azure_path),
                             "%s?api-version=2024-02-15-preview", path);
            api_path = azure_path;
        }

        const char* extra_hdrs = string_dynamic_cstr(extra_headers);
        int result = http_request("POST", host, port, api_path,
                                   extra_hdrs[0] ? extra_hdrs : NULL,
                                   body, strlen(body), http_response, is_https);

        adapter->last_http_status = 0;

        if (result == 0) {
            http_ok = 1;
            break;
        }

        /* 根据本次HTTP状态码判断是否值得重试 */
        int can_retry = (result == 429 || result == 500 || result == 502 || result == 503 || result == 504);
        if (can_retry && retry < retries - 1) {
            /* 指数退避 + 完整抖动 */
            unsigned int delay_ms = get_retry_delay_ms(retry);
            time_sleep_ms(delay_ms);
        } else if (!can_retry && result != 0) {
            /* 不可重试的错误（4xx客户端错误），立即退出 */
            break;
        }
    }

    string_dynamic_free(extra_headers);
    safe_free((void**)&body);

    if (!http_ok || !http_response) {
        if (http_response) string_dynamic_free(http_response);
        response->success = 0;
        string_copy_safe(response->error_message,
                        "HTTP请求失败", sizeof(response->error_message));
        return -1;
    }

    /* 解析响应 */
    const char* resp_text = string_dynamic_cstr(http_response);
    response->http_status = adapter->last_http_status;
    response->success = 1;

    parse_chat_response(resp_text, response);

    /* 检查错误 */
    if (response->error_message[0] != '\0') {
        response->success = 0;
    }

    /* 如果content为空，但响应体非空，直接保存原始响应 */
    if (!response->content && resp_text && resp_text[0]) {
        response->content = string_duplicate(resp_text);
        if (response->content) {
            response->content_length = strlen(response->content);
        }
    }

    string_dynamic_free(http_response);
    return response->success ? 0 : -1;
}

int openai_adapter_fine_tune(OpenAIAdapter* adapter,
                              const char* training_data_json,
                              OpenAIResponse* response) {
    if (!adapter || !adapter->initialized || !training_data_json || !response) {
        return -1;
    }

    memset(response, 0, sizeof(OpenAIResponse));

    char host[256];
    unsigned short port;
    char path[512];
    int is_https2 = parse_api_url(adapter->config.api_base, host, sizeof(host),
                  &port, path, sizeof(path));

    /* 构建Authorization头 */
    char* auth_header = build_auth_header(adapter);

    /* 构建微调请求体 */
    DynamicString* body = string_dynamic_create(4096);
    if (!body) {
        if (auth_header) safe_free((void**)&auth_header);
        return -1;
    }

    string_dynamic_append(body, "{\n");
    string_dynamic_append(body, "  \"model\": \"");
    string_dynamic_append(body, adapter->config.model);
    string_dynamic_append(body, "\",\n");
    string_dynamic_append(body, "  \"training_data\": ");
    string_dynamic_append(body, training_data_json);
    string_dynamic_append(body, "\n}");

    char* body_str = string_duplicate(string_dynamic_cstr(body));
    string_dynamic_free(body);

    /* 构建额外请求头 */
    DynamicString* extra_headers = string_dynamic_create(256);
    if (auth_header) {
        string_dynamic_append(extra_headers, auth_header);
        safe_free((void**)&auth_header);
    }

    /* 微调API路径 */
    char api_path[512];
    string_format_safe(api_path, sizeof(api_path), "%s/fine_tuning/jobs", path);

    int http_ok = 0;
    DynamicString* http_response = NULL;
    int retries = adapter->config.max_retries > 0 ? adapter->config.max_retries : OPENAI_MAX_RETRIES;

    for (int retry = 0; retry < retries; retry++) {
        if (http_response) string_dynamic_free(http_response);
        http_response = string_dynamic_create(OPENAI_HTTP_BUF_SIZE);
        if (!http_response) {
            string_dynamic_free(extra_headers);
            safe_free((void**)&body_str);
            return -1;
        }

        int result = http_request("POST", host, port, api_path,
                                   string_dynamic_cstr(extra_headers),
                                   body_str, strlen(body_str), http_response, is_https2);

        if (result == 0) {
            http_ok = 1;
            break;
        }

        if (retry < retries - 1) {
            unsigned int delay_ms = get_retry_delay_ms(retry);
            time_sleep_ms(delay_ms);
        }
    }

    safe_free((void**)&body_str);
    string_dynamic_free(extra_headers);

    if (!http_ok || !http_response) {
        if (http_response) string_dynamic_free(http_response);
        response->success = 0;
        string_copy_safe(response->error_message,
                        "微调API请求失败", sizeof(response->error_message));
        return -1;
    }

    const char* resp_text = string_dynamic_cstr(http_response);
    response->http_status = 200;
    response->success = 1;

    JsonValue* root = json_parse(resp_text);
    if (root) {
        const char* id = json_get_string(root, "id");
        if (id) { response->content = string_duplicate(id); response->content_length = strlen(id); }
        else { response->content = string_duplicate(resp_text); if (response->content) response->content_length = strlen(response->content); }
        JsonValue* err = json_get(root, "error");
        if (err) {
            const char* msg = json_get_string(err, "message");
            if (msg) { string_copy_safe(response->error_message, msg, sizeof(response->error_message)); response->success = 0; }
        }
        json_free(root);
    } else {
        /* 回退 */
        char* job_id = json_extract_string(resp_text, "id");
        if (job_id) { response->content = job_id; response->content_length = strlen(job_id); }
        else { response->content = string_duplicate(resp_text); if (response->content) response->content_length = strlen(response->content); }
        char* error = json_extract_string(resp_text, "message");
        if (error) { string_copy_safe(response->error_message, error, sizeof(response->error_message)); safe_free((void**)&error); if (response->error_message[0]) response->success = 0; }
    }

    string_dynamic_free(http_response);
    return response->success ? 0 : -1;
}

int openai_adapter_list_models(OpenAIAdapter* adapter,
                                OpenAIResponse* response) {
    if (!adapter || !adapter->initialized || !response) {
        return -1;
    }

    memset(response, 0, sizeof(OpenAIResponse));

    char host[256];
    unsigned short port;
    char path[512];
    int is_https3 = parse_api_url(adapter->config.api_base, host, sizeof(host),
                  &port, path, sizeof(path));

    /* 构建Authorization头 */
    char* auth_header = build_auth_header(adapter);
    DynamicString* extra_headers = string_dynamic_create(512);
    if (extra_headers && auth_header) {
        string_dynamic_append(extra_headers, auth_header);
        safe_free((void**)&auth_header);
    }

    /* 模型列表API路径 */
    char api_path[512];
    string_format_safe(api_path, sizeof(api_path), "%s/models", path);

    int http_ok = 0;
    DynamicString* http_response = NULL;
    int retries = adapter->config.max_retries > 0 ? adapter->config.max_retries : OPENAI_MAX_RETRIES;

    for (int retry = 0; retry < retries; retry++) {
        if (http_response) string_dynamic_free(http_response);
        http_response = string_dynamic_create(OPENAI_HTTP_BUF_SIZE);
        if (!http_response) {
            string_dynamic_free(extra_headers);
            return -1;
        }

        int result = http_request("GET", host, port, api_path,
                                   string_dynamic_cstr(extra_headers),
                                   NULL, 0, http_response, is_https3);

        if (result == 0) {
            http_ok = 1;
            break;
        }

        if (retry < retries - 1) {
            unsigned int delay_ms = get_retry_delay_ms(retry);
            time_sleep_ms(delay_ms);
        }
    }

    string_dynamic_free(extra_headers);

    if (!http_ok || !http_response) {
        if (http_response) string_dynamic_free(http_response);
        response->success = 0;
        string_copy_safe(response->error_message,
                        "模型列表请求失败", sizeof(response->error_message));
        return -1;
    }

    const char* resp_text = string_dynamic_cstr(http_response);
    response->http_status = 200;
    response->success = 1;

    parse_models_response(resp_text, response);

    /* 如果没有解析到content，保存原始响应 */
    if (!response->content) {
        response->content = string_duplicate(resp_text);
        if (response->content) {
            response->content_length = strlen(response->content);
        }
    }

    string_dynamic_free(http_response);
    return 0;
}

void openai_response_free(OpenAIResponse* response) {
    if (!response) return;

    if (response->content) {
        safe_free((void**)&response->content);
    }
    if (response->finish_reason) {
        safe_free((void**)&response->finish_reason);
    }

    memset(response, 0, sizeof(OpenAIResponse));
}

/* ============================================================================
 * 外部API训练客户端：通过HTTP调用外部API获取训练数据并训练本地模型
 * ============================================================================ */

int api_training_fetch_dataset(const char* api_url, const char* api_key,
                                float** data_out, size_t* data_count,
                                size_t* feature_dim, size_t* label_dim) {
    if (!api_url || !data_out || !data_count || !feature_dim || !label_dim) return -1;

    *data_out = NULL;
    *data_count = 0;
    *feature_dim = 0;
    *label_dim = 0;

    /* 使用真实的HTTP请求从远程API拉取数据集 */
    char host[256];
    unsigned short port;
    char path[512];
    int is_https = 0;

    /* 解析URL */
    const char* url_ptr = api_url;
    if (strncmp(api_url, "https://", 8) == 0) {
        is_https = 1;
        url_ptr = api_url + 8;
    } else if (strncmp(api_url, "http://", 7) == 0) {
        url_ptr = api_url + 7;
    }

    /* 提取主机和路径 */
    const char* path_start = strchr(url_ptr, '/');
    if (!path_start) {
        string_copy_safe(host, url_ptr, sizeof(host));
        string_format_safe(path, sizeof(path), "/");
    } else {
        size_t host_len = (size_t)(path_start - url_ptr);
        if (host_len >= sizeof(host)) host_len = sizeof(host) - 1;
        memcpy(host, url_ptr, host_len);
        host[host_len] = '\0';
        string_copy_safe(path, path_start, sizeof(path));
    }

    /* 提取端口 */
    char* colon = strchr(host, ':');
    if (colon) {
        *colon = '\0';
        port = (unsigned short)atoi(colon + 1);
    } else {
        port = is_https ? 443 : 80;
    }

    /* 构建Authorization头 */
    DynamicString* extra_headers = string_dynamic_create(512);
    if (api_key && api_key[0]) {
        string_dynamic_append(extra_headers, "Authorization: Bearer ");
        string_dynamic_append(extra_headers, api_key);
        string_dynamic_append(extra_headers, "\r\n");
    }

    /* 发起HTTP GET请求（带指数退避重试） */
    int http_ok = 0;
    DynamicString* http_response = NULL;

    for (int retry = 0; retry < OPENAI_MAX_RETRIES; retry++) {
        if (http_response) string_dynamic_free(http_response);
        http_response = string_dynamic_create(OPENAI_HTTP_BUF_SIZE);
        if (!http_response) {
            string_dynamic_free(extra_headers);
            return -1;
        }

        const char* hdrs = string_dynamic_cstr(extra_headers);
        int result = http_request("GET", host, port, path,
                                   hdrs && hdrs[0] ? hdrs : NULL,
                                   NULL, 0, http_response, is_https);

        if (result == 0) {
            http_ok = 1;
            break;
        }

        if (retry < OPENAI_MAX_RETRIES - 1) {
            unsigned int delay_ms = get_retry_delay_ms(retry);
            time_sleep_ms(delay_ms);
        }
    }

    string_dynamic_free(extra_headers);

    if (!http_ok || !http_response) {
        if (http_response) string_dynamic_free(http_response);
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "HTTP GET请求失败（已达最大重试次数），无法从远程API拉取数据集");
        return -1;
    }

    /* 解析HTTP响应体 */
    const char* resp_body = string_dynamic_cstr(http_response);

    /* 查找JSON数组起始位置 */
    const char* body_start = strchr(resp_body, '{');
    if (!body_start) body_start = strchr(resp_body, '[');
    if (!body_start) {
        string_dynamic_free(http_response);
        selflnn_set_last_error(SELFLNN_ERROR_FORMAT_ERROR, __func__, __FILE__, __LINE__,
                              "HTTP响应中未找到JSON数据");
        return -1;
    }

    /* 尝试提取"data"字段或"dataset"字段 */
    const char* data_str = NULL;
    char* extracted = json_extract_string(body_start, "data");
    if (extracted) {
        data_str = extracted;
    } else {
        extracted = json_extract_string(body_start, "dataset");
        if (extracted) data_str = extracted;
    }

    const char* arr_start = NULL;
    int need_free_extracted = (extracted != NULL);

    if (data_str) {
        /* data字段的值可能是一个数组字符串，也可能直接是JSON数组 */
        arr_start = strchr(data_str, '[');
    }

    if (!arr_start) {
        /* 直接查找JSON数组 */
        arr_start = strchr(body_start, '[');
    }

    if (!arr_start) {
        if (need_free_extracted) safe_free((void**)&extracted);
        string_dynamic_free(http_response);
        selflnn_set_last_error(SELFLNN_ERROR_FORMAT_ERROR, __func__, __FILE__, __LINE__,
                              "HTTP响应中未找到数据集数组");
        return -1;
    }

    arr_start++;

    /* 计数数组中的浮点数 */
    size_t count = 0;
    const char* scan = arr_start;
    while (*scan && *scan != ']') {
        if (*scan == ',' || scan == arr_start) count++;
        while (*scan == ' ' || *scan == ',') scan++;
        if (*scan != ']' && *scan != '\0') {
            while (*scan && *scan != ',' && *scan != ']') scan++;
        }
    }
    if (count == 0) {
        if (need_free_extracted) safe_free((void**)&extracted);
        string_dynamic_free(http_response);
        return -1;
    }

    /* 分配并解析浮点数数组 */
    float* data = (float*)safe_calloc(count, sizeof(float));
    if (!data) {
        if (need_free_extracted) safe_free((void**)&extracted);
        string_dynamic_free(http_response);
        return -1;
    }

    size_t parsed = 0;
    const char* p = arr_start;
    while (*p && *p != ']' && parsed < count) {
        while (*p == ' ' || *p == ',' || *p == '\n' || *p == '\r') p++;
        if (*p == ']' || *p == '\0') break;
        char* endp = NULL;
        float val = strtof(p, &endp);
        if (endp == p) break;
        data[parsed++] = val;
        p = endp;
    }

    if (need_free_extracted) safe_free((void**)&extracted);
    string_dynamic_free(http_response);

    *data_out = data;
    *data_count = parsed > 0 ? parsed : count;
    *feature_dim = 1;
    *label_dim = 1;
    return (int)parsed;
}

/* 使用外部数据训练模型：优先使用session中的配置，否则使用默认值创建临时网络和训练器 */
int api_training_train_with_external(void* session,
                                       const float* external_data,
                                       size_t data_count, size_t feature_dim,
                                       int epochs) {
    if (!external_data || data_count == 0 || feature_dim == 0) return -1;

    TrainingSession* ts = (TrainingSession*)session;
    int has_valid_session = (ts != NULL && ts->network != NULL);

    /* 从session提取训练配置，无session或无效时使用默认值 */
    TrainingConfig config;
    if (ts != NULL && ts->config.epochs > 0) {
        /* 存在有效session，复制其训练配置 */
        memcpy(&config, &ts->config, sizeof(TrainingConfig));
        /* 外部传入的epochs参数优先级最高（如果显式指定） */
        if (epochs > 0) config.epochs = (size_t)epochs;
    } else {
        /* 无有效session，使用安全默认值 */
        memset(&config, 0, sizeof(TrainingConfig));
        config.learning_rate = 0.001f;
        config.epochs = epochs > 0 ? (size_t)epochs : 10;
        config.batch_size = 32;
        config.mode = TRAIN_MODE_MINI_BATCH;
        config.optimizer = OPTIMIZER_ADAM;
        config.loss_function = LOSS_MSE;
        config.regularization = REGULARIZATION_L2;
        config.enable_validation = 1;
        config.verbose = 0;
        config.patience = 5;
    }

    LNN* network = NULL;
    Trainer* trainer = NULL;
    int created_network = 0;
    int created_trainer = 0;
    int result = 0;

    if (has_valid_session) {
        /* 复用session中已有的网络和训练器，无需重新创建 */
        network = ts->network;
        if (ts->trainer != NULL) {
            trainer = ts->trainer;
            /* 更新训练器的配置以反映当前参数 */
            memcpy(&trainer->config, &config, sizeof(TrainingConfig));
        }
    }

    if (!network) {
        /* 无可用网络，创建临时LNN网络 */
        LNNConfig net_cfg;
        memset(&net_cfg, 0, sizeof(LNNConfig));
        net_cfg.input_size = (int)feature_dim;
        int hidden = (int)(feature_dim * 2);
        if (hidden < 64) hidden = 64;
        net_cfg.hidden_size = hidden;
        net_cfg.output_size = 1;
        net_cfg.learning_rate = config.learning_rate;
        net_cfg.time_constant = 0.1f;
        net_cfg.enable_training = 1;
        net_cfg.enable_adaptation = 1;
        network = lnn_create(&net_cfg);
        if (!network) return -1;
        created_network = 1;
    }

    if (!trainer) {
        /* 无可用训练器，创建临时训练器 */
        trainer = trainer_create(&config, network);
        if (!trainer) {
            if (created_network) lnn_free(network);
            return -1;
        }
        created_trainer = 1;
    }

    /* 执行训练 */
    result = trainer_train(trainer, external_data, NULL, data_count, NULL, NULL);

    /* 仅清理临时创建的资源，不销毁session中的资源 */
    if (created_trainer) {
        trainer_free(trainer);
    }
    if (created_network) {
        lnn_free(network);
    }

    return result;
}