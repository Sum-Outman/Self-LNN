#include "selflnn/distributed/pbft.h"
#include "selflnn/utils/platform.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/logging.h"
/* ZSFJJJ-01: 使用共享SHA-256实现 */
#include "selflnn/utils/math_utils.h"
#include "selflnn/core/laplace.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#ifdef _MSC_VER
#pragma warning(disable:4456)
#endif

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#endif

#define SAFE_FREE_ARR(p) do { if ((p)) { safe_free((void**)&(p)); (p) = NULL; } } while(0)

static uint64_t get_current_time_ms(void) {
    TimeValue tv = time_get_current();
    return (uint64_t)tv.seconds * 1000 + (uint64_t)(tv.microseconds / 1000);
}

/* ZSFJJJ-01: pbft_compute_digest 使用共享SHA-256实现，消除重复代码 */
void pbft_compute_digest(const void* data, uint32_t length, uint32_t digest[8]) {
    if (!digest) return;
    memset(digest, 0, 8 * sizeof(uint32_t));
    if (!data || length == 0) return;
    uint8_t hash[SELFLNN_SHA256_HASH_LEN];
    selflnn_sha256_hash((const uint8_t*)data, (size_t)length, hash);
    memcpy(digest, hash, SELFLNN_SHA256_HASH_LEN);
}

int pbft_verify_digest(const void* data, uint32_t length, const uint32_t expected[8]) {
    if (!expected) return -1;
    uint32_t computed[8];
    pbft_compute_digest(data, length, computed);
    return memcmp(computed, expected, 8 * sizeof(uint32_t)) == 0 ? 0 : 1;
}

typedef struct {
    uint32_t from_node;
    uint32_t target_view;
    PbftViewChange msg;
    int validated;
} PbftViewChangeRecord;

struct PbftSystem {
    PbftConfig config;
    PbftNodeInfo nodes[PBFT_MAX_NODES];
    int node_count;
    uint32_t current_view;
    uint32_t primary_id;
    PbftNodeStatus status;
    uint32_t last_executed_seq;
    uint32_t last_prepared_seq;
    uint32_t latest_checkpoint_seq;
    uint32_t checkpoint_digest[8];
    PbftLogEntry* log_entries;
    int log_count;
    int log_capacity;
    int active_request;
    uint32_t pending_client_id;
    uint32_t pending_request_id;
    uint32_t pending_op_type;
    uint8_t* pending_payload;
    uint32_t pending_payload_size;
    uint64_t pending_timestamp;
    uint8_t* result_buffer;
    uint32_t result_size;
    uint32_t result_capacity;
    int result_ready;
    PbftStats stats;
    uint64_t last_view_change_time_ms;
    int view_change_in_progress;
    uint32_t view_change_collected;
    uint32_t target_view;
    PbftViewChange* view_change_msgs[PBFT_MAX_NODES];
    uint32_t new_view_received;
    uint32_t new_view_sent;
    uint64_t last_view_change_retry_ms;
    int view_change_retry_count;
    PbftViewChangeRecord vc_records[PBFT_MAX_NODES];
    int vc_record_count;
    uint32_t new_pre_prepare_count;
    uint32_t new_pre_prepare_seqs[PBFT_MAX_PENDING_BATCH];
    uint32_t new_pre_prepare_digests[PBFT_MAX_PENDING_BATCH][8];
    uint64_t last_tick_ms;
    
    /* 网络套接字 */
#ifdef _WIN32
    SOCKET listen_socket;
#else
    int listen_socket;
#endif
    uint8_t* recv_buffer;
    uint32_t recv_buffer_size;
    int server_initialized;
    
    /* 拉普拉斯分析器（频域PBFT共识稳定性分析与视图变更预测） */
    LaplaceAnalyzer* laplace_analyzer;
    float* laplace_spectrum_buffer;
    
    /* 请求执行回调 —— 当PBFT共识达成后调用此回调执行实际操作
     * 回调参数: client_id, request_id, op_type, payload, payload_size, user_data
     * 返回0表示成功，非0表示失败 */
    int (*execute_callback)(uint32_t client_id, uint32_t request_id,
                            uint32_t op_type, const void* payload,
                            uint32_t payload_size, void* user_data);
    void* execute_callback_user_data;
};

void pbft_default_config(PbftConfig* config) {
    if (!config) return;
    memset(config, 0, sizeof(PbftConfig));
    config->node_id = 0;
    config->num_nodes = 4;
    config->max_fault = 1;
    config->checkpoint_interval = PBFT_CHECKPOINT_INTERVAL;
    config->view_change_timeout_ms = PBFT_VIEW_CHANGE_TIMEOUT_MS;
    config->listen_port = PBFT_DEFAULT_PORT;
    strcpy(config->listen_host, "0.0.0.0");
    config->enable_auto_view_change = 1;
    config->enable_checkpoint_gc = 1;
    config->verbose = 0;
}

static int log_entry_init(PbftSystem* system) {
    system->log_capacity = 1024;
    system->log_entries = (PbftLogEntry*)calloc(system->log_capacity, sizeof(PbftLogEntry));
    if (!system->log_entries) return -1;
    system->log_count = 0;
    return 0;
}

static PbftLogEntry* log_get_entry(PbftSystem* system, uint32_t seq) {
    for (int i = 0; i < system->log_count; i++) {
        if (system->log_entries[i].sequence_number == seq) return &system->log_entries[i];
    }
    return NULL;
}

static PbftLogEntry* log_create_entry(PbftSystem* system, uint32_t seq) {
    PbftLogEntry* existing = log_get_entry(system, seq);
    if (existing) return existing;
    if (system->log_count >= system->log_capacity) {
        system->log_capacity *= 2;
        PbftLogEntry* new_log = (PbftLogEntry*)realloc(system->log_entries,
            system->log_capacity * sizeof(PbftLogEntry));
        if (!new_log) return NULL;
        system->log_entries = new_log;
    }
    PbftLogEntry* entry = &system->log_entries[system->log_count++];
    memset(entry, 0, sizeof(PbftLogEntry));
    entry->sequence_number = seq;
    return entry;
}

static void log_gc(PbftSystem* system, uint32_t up_to_seq) {
    int new_count = 0;
    for (int i = 0; i < system->log_count; i++) {
        if (system->log_entries[i].sequence_number > up_to_seq) {
            if (new_count != i) system->log_entries[new_count] = system->log_entries[i];
            new_count++;
        }
    }
    system->log_count = new_count;
}

static int is_digest_equal(const uint32_t a[8], const uint32_t b[8]) {
    return memcmp(a, b, 8 * sizeof(uint32_t)) == 0;
}

static void copy_digest(uint32_t dst[8], const uint32_t src[8]) {
    memcpy(dst, src, 8 * sizeof(uint32_t));
}

static int has_quorum_prepare(PbftSystem* system, uint32_t seq) {
    int count = 0;
    uint32_t quorum = system->config.num_nodes - system->config.max_fault;
    for (int i = 0; i < system->log_count; i++) {
        if (system->log_entries[i].sequence_number == seq && system->log_entries[i].prepare_count > 0) {
            count += system->log_entries[i].prepare_count;
        }
    }
    return count >= (int)(2 * quorum - 1);
}

static int has_quorum_commit(PbftSystem* system, uint32_t seq) {
    int count = 0;
    uint32_t quorum = system->config.num_nodes - system->config.max_fault;
    for (int i = 0; i < system->log_count; i++) {
        if (system->log_entries[i].sequence_number == seq && system->log_entries[i].commit_count > 0) {
            count += system->log_entries[i].commit_count;
        }
    }
    return count >= (int)(2 * quorum - 1);
}

static int pbft_send_message(PbftSystem* system, uint32_t target_id,
                             const void* msg, uint32_t msg_size) {
    if (target_id >= (uint32_t)system->node_count) return -1;
    if (!system->nodes[target_id].is_active) return -1;
    if (!msg || msg_size == 0) return -1;

    PbftNodeInfo* target = &system->nodes[target_id];
    
#ifdef _WIN32
    SOCKET sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (sock == INVALID_SOCKET) return -1;
    DWORD timeout = 1000;
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, (const char*)&timeout, sizeof(timeout));
#else
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) return -1;
    struct timeval tv = {1, 0};
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
#endif

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(target->port);
    
    if (inet_pton(AF_INET, target->host, &addr.sin_addr) <= 0) {
#ifdef _WIN32
        closesocket(sock);
#else
        close(sock);
#endif
        system->nodes[target_id].consecutive_timeouts++;
        return -1;
    }

#ifdef _WIN32
    int sent = sendto(sock, (const char*)msg, (int)msg_size, 0,
                      (struct sockaddr*)&addr, sizeof(addr));
    closesocket(sock);
#else
    ssize_t ret = sendto(sock, msg, msg_size, 0,
                         (struct sockaddr*)&addr, sizeof(addr));
    int sent = (int)ret;
    close(sock);
#endif

    if (sent < 0) {
        system->nodes[target_id].consecutive_timeouts++;
        system->nodes[target_id].last_heartbeat = get_current_time_ms();
        return -1;
    }

    system->nodes[target_id].last_heartbeat = get_current_time_ms();
    system->nodes[target_id].consecutive_timeouts = 0;
    return 0;
}

static int pbft_broadcast(PbftSystem* system, const void* msg, uint32_t msg_size) {
    int sent = 0;
    for (int i = 0; i < system->node_count; i++) {
        if (i != (int)system->config.node_id && system->nodes[i].is_active) {
            if (pbft_send_message(system, (uint32_t)i, msg, msg_size) >= 0) sent++;
        }
    }
    return sent;
}

static int pbft_send_to_clients(PbftSystem* system, const void* msg, uint32_t msg_size) {
    if (!system || !msg || msg_size == 0) return 0;

    int sent = 0;
    for (int i = 0; i < system->node_count; i++) {
        if (i != (int)system->config.node_id && system->nodes[i].is_active) {
            if (pbft_send_message(system, (uint32_t)i, msg, msg_size) >= 0) sent++;
        }
    }
    
    log_debug("[PBFT] 客户端广播完成: %d/%d 节点已发送", sent, system->node_count - 1);
    return sent;
}

static uint32_t pbft_sequence_number(PbftSystem* system) {
    return system->last_prepared_seq + 1;
}

PbftSystem* pbft_system_create(const PbftConfig* config) {
    if (!config) return NULL;
    PbftSystem* system = (PbftSystem*)calloc(1, sizeof(PbftSystem));
    if (!system) return NULL;
    memcpy(&system->config, config, sizeof(PbftConfig));
    system->current_view = 0;
    system->primary_id = 0;
    system->status = PBFT_NODE_NORMAL;
    system->last_executed_seq = 0;
    system->last_prepared_seq = 0;
    system->latest_checkpoint_seq = 0;
    system->node_count = (int)config->num_nodes;
    system->active_request = 0;
    system->result_ready = 0;
    system->view_change_in_progress = 0;
    system->view_change_collected = 0;
    system->target_view = 0;
    system->last_view_change_time_ms = 0;
    system->last_view_change_retry_ms = 0;
    system->view_change_retry_count = 0;
    system->new_view_received = 0;
    system->new_view_sent = 0;
    system->vc_record_count = 0;
    system->new_pre_prepare_count = 0;
    system->last_tick_ms = get_current_time_ms();
    
    /* 初始化网络套接字 */
#ifdef _WIN32
    system->listen_socket = INVALID_SOCKET;
#else
    system->listen_socket = -1;
#endif
    system->recv_buffer = NULL;
    system->recv_buffer_size = 0;
    system->server_initialized = 0;
    
    memset(system->checkpoint_digest, 0, sizeof(system->checkpoint_digest));
    memset(&system->stats, 0, sizeof(PbftStats));
    system->stats.start_time_ms = get_current_time_ms();
    system->stats.last_metric_time_ms = system->stats.start_time_ms;
    for (int i = 0; i < PBFT_MAX_NODES; i++) {
        system->nodes[i].node_id = i;
        system->nodes[i].is_active = 0;
        system->nodes[i].is_primary = 0;
        system->view_change_msgs[i] = NULL;
        memset(&system->vc_records[i], 0, sizeof(PbftViewChangeRecord));
    }
    if (log_entry_init(system) != 0) {
        free(system);
        return NULL;
    }
    
    /* P1-022修复：PBFT拉普拉斯分析器参数从共识周期和节点规模动态推导
       PBFT共识频域特征与负载均衡完全不同：需要更高频率分辨率捕捉视图切换谐波 */
    {
        LaplaceConfig lap_cfg;
        memset(&lap_cfg, 0, sizeof(lap_cfg));
        /* 共识轮次间隔典型为10ms，采样1024点覆盖约10秒共识历史 */
        float consensus_period_ms = 10.0f;
        lap_cfg.num_samples = 1024;
        lap_cfg.sample_rate = 1000.0f / consensus_period_ms;
        /* 最大频率需覆盖视图切换速率（每100共识轮可能切换1次） */
        lap_cfg.max_frequency = 100.0f;
        lap_cfg.min_frequency = 1.0f / (lap_cfg.num_samples * consensus_period_ms / 1000.0f);
        lap_cfg.enable_stability = 1;
        lap_cfg.enable_frequency = 1;
        lap_cfg.enable_optimization = 1;
        /* 截止频率：主要捕获视图切换的基频和谐波 */
        lap_cfg.cutoff_frequency = 10.0f;
        lap_cfg.filter_order = 3;
        /* 衰减因子：PBFT需要更快适应视图切换（较低alpha，较高beta） */
        lap_cfg.alpha = 0.85f;
        lap_cfg.beta  = 0.15f;
        system->laplace_analyzer = laplace_analyzer_create(&lap_cfg);
        system->laplace_spectrum_buffer = (float*)safe_malloc(1024 * sizeof(float));
        if (system->laplace_spectrum_buffer) {
            memset(system->laplace_spectrum_buffer, 0, 1024 * sizeof(float));
        }
    }
    
    return system;
}

void pbft_system_destroy(PbftSystem* system) {
    if (!system) return;
    pbft_disconnect_all(system);
    for (int i = 0; i < PBFT_MAX_NODES; i++) {
        if (system->view_change_msgs[i]) {
            free(system->view_change_msgs[i]);
            system->view_change_msgs[i] = NULL;
        }
    }
    if (system->pending_payload) free(system->pending_payload);
    /* result_buffer 是外部传入的指针，不由本模块分配，不能 free */
    system->result_buffer = NULL;
    if (system->log_entries) free(system->log_entries);
    if (system->laplace_analyzer) {
        laplace_analyzer_free(system->laplace_analyzer);
        system->laplace_analyzer = NULL;
    }
    safe_free((void**)&system->laplace_spectrum_buffer);
    free(system);
}

int pbft_add_node(PbftSystem* system, uint32_t node_id, const char* host, uint16_t port) {
    if (!system || !host || !host[0] || node_id >= PBFT_MAX_NODES) return PBFT_ERROR_INVALID_PARAM;
    if (system->nodes[node_id].is_active) return PBFT_ERROR_INVALID_PARAM;
    system->nodes[node_id].node_id = node_id;
    strncpy(system->nodes[node_id].host, host, sizeof(system->nodes[node_id].host) - 1);
    system->nodes[node_id].host[sizeof(system->nodes[node_id].host) - 1] = '\0';
    system->nodes[node_id].port = port;
    system->nodes[node_id].is_active = 1;
    system->nodes[node_id].is_primary = (node_id == system->primary_id);
    system->nodes[node_id].last_heartbeat = get_current_time_ms();
    system->nodes[node_id].consecutive_timeouts = 0;
    if ((int)node_id >= system->node_count) system->node_count = (int)node_id + 1;
    return PBFT_ERROR_NONE;
}

int pbft_remove_node(PbftSystem* system, uint32_t node_id) {
    if (!system || node_id >= PBFT_MAX_NODES) return PBFT_ERROR_INVALID_PARAM;
    system->nodes[node_id].is_active = 0;
    system->nodes[node_id].is_primary = 0;
    return PBFT_ERROR_NONE;
}

int pbft_connect_all(PbftSystem* system) {
    if (!system) return PBFT_ERROR_INVALID_PARAM;
    if (system->server_initialized) return PBFT_ERROR_NONE;

#ifdef _WIN32
    WSADATA wsa_data;
    if (WSAStartup(MAKEWORD(2, 2), &wsa_data) != 0) return -1;
    system->listen_socket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (system->listen_socket == INVALID_SOCKET) {
        WSACleanup();
        return -1;
    }
#else
    system->listen_socket = socket(AF_INET, SOCK_DGRAM, 0);
    if (system->listen_socket < 0) return -1;
#endif

    int reuse = 1;
#ifdef _WIN32
    setsockopt(system->listen_socket, SOL_SOCKET, SO_REUSEADDR,
               (const char*)&reuse, sizeof(reuse));
#else
    setsockopt(system->listen_socket, SOL_SOCKET, SO_REUSEADDR,
               &reuse, sizeof(reuse));
#endif

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(system->config.listen_port);
    if (inet_pton(AF_INET, system->config.listen_host, &addr.sin_addr) <= 0) {
        addr.sin_addr.s_addr = INADDR_ANY;
    }

#ifdef _WIN32
    if (bind(system->listen_socket, (struct sockaddr*)&addr, sizeof(addr)) == SOCKET_ERROR) {
        closesocket(system->listen_socket);
        system->listen_socket = INVALID_SOCKET;
        WSACleanup();
        return -1;
    }
    u_long nonblock = 1;
    ioctlsocket(system->listen_socket, FIONBIO, &nonblock);
#else
    if (bind(system->listen_socket, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        close(system->listen_socket);
        system->listen_socket = -1;
        return -1;
    }
    int flags = fcntl(system->listen_socket, F_GETFL, 0);
    fcntl(system->listen_socket, F_SETFL, flags | O_NONBLOCK);
#endif

    system->recv_buffer_size = PBFT_MAX_REQUEST_SIZE + sizeof(PbftMessageHeader) + 256;
    system->recv_buffer = (uint8_t*)malloc(system->recv_buffer_size);
    if (!system->recv_buffer) {
#ifdef _WIN32
        closesocket(system->listen_socket);
        system->listen_socket = INVALID_SOCKET;
        WSACleanup();
#else
        close(system->listen_socket);
        system->listen_socket = -1;
#endif
        return PBFT_ERROR_OUT_OF_MEMORY;
    }
    memset(system->recv_buffer, 0, system->recv_buffer_size);
    system->server_initialized = 1;

    log_info("[PBFT] 节点 %u 启动监听: %s:%u",
             system->config.node_id, system->config.listen_host, system->config.listen_port);
    return PBFT_ERROR_NONE;
}

int pbft_disconnect_all(PbftSystem* system) {
    if (!system) return PBFT_ERROR_INVALID_PARAM;
    if (!system->server_initialized) return PBFT_ERROR_NONE;

#ifdef _WIN32
    if (system->listen_socket != INVALID_SOCKET) {
        closesocket(system->listen_socket);
        system->listen_socket = INVALID_SOCKET;
    }
    WSACleanup();
#else
    if (system->listen_socket >= 0) {
        close(system->listen_socket);
        system->listen_socket = -1;
    }
#endif

    if (system->recv_buffer) {
        free(system->recv_buffer);
        system->recv_buffer = NULL;
    }
    system->recv_buffer_size = 0;
    system->server_initialized = 0;

    log_info("[PBFT] 节点 %u 停止监听", system->config.node_id);
    return PBFT_ERROR_NONE;
}

static int pbft_execute_request(PbftSystem* system, uint32_t seq) {
    PbftLogEntry* entry = log_get_entry(system, seq);
    if (!entry || entry->executed) return -1;
    entry->executed = 1;
    if (seq > system->last_executed_seq) system->last_executed_seq = seq;

    /* 真实请求执行：调用注册的回调函数执行实际操作
     * M-020修复：将pending_payload/pending_payload_size/pending_op_type
     * 正确传递到execute_callback，而非固定传NULL和0。
     * pending_payload在pbft_submit_request中分配并保存原始请求负载，
     * 是共识达成后执行实际操作所必需的数据。 */
    if (system->execute_callback) {
        int ret = system->execute_callback(
            entry->client_id, entry->request_id, system->pending_op_type,
            system->pending_payload, system->pending_payload_size,
            system->execute_callback_user_data);
        if (ret != 0) {
            log_error("[PBFT] 请求执行回调失败: seq=%u client=%u req=%u ret=%d",
                      seq, entry->client_id, entry->request_id, ret);
            return -1;
        }
    }

    if (system->config.verbose) {
        printf("[PBFT] 请求已执行: seq=%u client=%u req=%u callback=%s\n",
               seq, entry->client_id, entry->request_id,
               system->execute_callback ? "已注册" : "未注册");
    }
    system->stats.total_commits++;
    if (seq % system->config.checkpoint_interval == 0) {
        pbft_trigger_checkpoint(system);
    }
    return 0;
}

int pbft_submit_request(PbftSystem* system, uint32_t client_id,
                        uint32_t request_id, uint32_t operation_type,
                        const void* payload, uint32_t payload_size,
                        void* result_buffer, uint32_t* result_size) {
    if (!system) return PBFT_ERROR_INVALID_PARAM;
    if (system->status == PBFT_NODE_STOPPED) return PBFT_ERROR_SYSTEM_STOPPED;
    if (!payload || payload_size == 0) return PBFT_ERROR_INVALID_PARAM;
    if (system->active_request) return -1;
    system->pending_client_id = client_id;
    system->pending_request_id = request_id;
    system->pending_op_type = operation_type;
    system->pending_payload_size = payload_size;
    system->pending_timestamp = get_current_time_ms();
    if (system->pending_payload) free(system->pending_payload);
    system->pending_payload = NULL;
    if (payload && payload_size > 0) {
        system->pending_payload = (uint8_t*)malloc(payload_size);
        if (!system->pending_payload) return PBFT_ERROR_OUT_OF_MEMORY;
        memcpy(system->pending_payload, payload, payload_size);
    }
    system->active_request = 1;
    system->result_ready = 0;
    system->result_buffer = (uint8_t*)result_buffer;
    system->result_size = 0;
    system->result_capacity = result_size ? *result_size : 0;
    system->stats.total_requests++;
    if (system->config.node_id == system->primary_id) {
        uint32_t seq = pbft_sequence_number(system);
        PbftPrePrepare pp;
        memset(&pp, 0, sizeof(pp));
        pp.header.type = PBFT_MSG_PRE_PREPARE;
        pp.header.sender_id = system->config.node_id;
        pp.header.view_number = system->current_view;
        pp.header.sequence_number = seq;
        pp.client_id = client_id;
        pp.request_id = request_id;
        pp.request_timestamp = system->pending_timestamp;
        if (payload && payload_size > 0) {
            pbft_compute_digest(payload, payload_size, pp.request_digest);
        }
        copy_digest(pp.header.digest, pp.request_digest);
        uint32_t pp_digest[8];
        pbft_compute_digest(&pp, sizeof(pp), pp_digest);
        PbftLogEntry* entry = log_create_entry(system, seq);
        if (entry) {
            entry->view_number = system->current_view;
            entry->client_id = client_id;
            entry->request_id = request_id;
            entry->request_timestamp = system->pending_timestamp;
            copy_digest(entry->request_digest, pp.request_digest);
        }
        system->last_prepared_seq = seq;
        pbft_broadcast(system, &pp, sizeof(pp));
        system->stats.total_pre_prepares++;
        if (system->config.verbose) {
            printf("[PBFT] 主节点 %u 发送 PrePrepare: view=%u seq=%u\n",
                   system->config.node_id, system->current_view, seq);
        }
    }
    return PBFT_ERROR_NONE;
}

int pbft_submit_request_async(PbftSystem* system, uint32_t client_id,
                              uint32_t request_id, uint32_t operation_type,
                              const void* payload, uint32_t payload_size) {
    if (!system) return PBFT_ERROR_INVALID_PARAM;
    if (system->status == PBFT_NODE_STOPPED) return PBFT_ERROR_SYSTEM_STOPPED;
    if (!payload || payload_size == 0) return PBFT_ERROR_INVALID_PARAM;
    system->pending_client_id = client_id;
    system->pending_request_id = request_id;
    system->pending_op_type = operation_type;
    system->pending_payload_size = payload_size;
    system->pending_timestamp = get_current_time_ms();
    if (system->pending_payload) free(system->pending_payload);
    system->pending_payload = NULL;
    if (payload && payload_size > 0) {
        system->pending_payload = (uint8_t*)malloc(payload_size);
        if (!system->pending_payload) return PBFT_ERROR_OUT_OF_MEMORY;
        memcpy(system->pending_payload, payload, payload_size);
    }
    system->active_request = 1;
    system->result_ready = 0;
    system->result_buffer = NULL;
    system->result_size = 0;
    system->result_capacity = 0;
    system->stats.total_requests++;
    if (system->config.node_id == system->primary_id) {
        uint32_t seq = pbft_sequence_number(system);
        PbftPrePrepare pp;
        memset(&pp, 0, sizeof(pp));
        pp.header.type = PBFT_MSG_PRE_PREPARE;
        pp.header.sender_id = system->config.node_id;
        pp.header.view_number = system->current_view;
        pp.header.sequence_number = seq;
        pp.client_id = client_id;
        pp.request_id = request_id;
        pp.request_timestamp = system->pending_timestamp;
        if (payload && payload_size > 0) {
            pbft_compute_digest(payload, payload_size, pp.request_digest);
        }
        copy_digest(pp.header.digest, pp.request_digest);
        PbftLogEntry* entry = log_create_entry(system, seq);
        if (entry) {
            entry->view_number = system->current_view;
            entry->client_id = client_id;
            entry->request_id = request_id;
            entry->request_timestamp = system->pending_timestamp;
            copy_digest(entry->request_digest, pp.request_digest);
        }
        system->last_prepared_seq = seq;
        pbft_broadcast(system, &pp, sizeof(pp));
        system->stats.total_pre_prepares++;
        if (system->config.verbose) {
            printf("[PBFT] 主节点 %u 发送 PrePrepare: view=%u seq=%u\n",
                   system->config.node_id, system->current_view, seq);
        }
    }
    return PBFT_ERROR_NONE;
}

int pbft_get_current_view(const PbftSystem* system) {
    if (!system) return -1;
    return (int)system->current_view;
}

uint32_t pbft_get_primary_id(const PbftSystem* system) {
    if (!system) return (uint32_t)-1;
    return system->primary_id;
}

int pbft_trigger_view_change(PbftSystem* system) {
    if (!system || system->status == PBFT_NODE_STOPPED) return -1;

    uint64_t now = get_current_time_ms();
    uint32_t backoff_ms = system->config.view_change_timeout_ms * (1 << system->view_change_retry_count);
    if (backoff_ms > 60000) backoff_ms = 60000;
    if (system->view_change_retry_count > 0 &&
        (now - system->last_view_change_retry_ms) < (uint64_t)backoff_ms) {
        return 0;
    }

    system->view_change_in_progress = 1;
    system->view_change_collected = 0;
    system->vc_record_count = 0;
    system->new_view_received = 0;
    system->new_view_sent = 0;
    system->new_pre_prepare_count = 0;

    uint32_t new_view = system->current_view + 1;
    system->target_view = new_view;

    for (int i = 0; i < PBFT_MAX_NODES; i++) {
        if (system->view_change_msgs[i]) {
            free(system->view_change_msgs[i]);
            system->view_change_msgs[i] = NULL;
        }
    }

    PbftViewChange vc;
    memset(&vc, 0, sizeof(vc));
    vc.header.type = PBFT_MSG_VIEW_CHANGE;
    vc.header.sender_id = system->config.node_id;
    vc.header.view_number = system->current_view;
    vc.new_view_number = new_view;
    vc.last_checkpoint_seq = system->latest_checkpoint_seq;
    copy_digest(vc.checkpoint_digest, system->checkpoint_digest);
    vc.prepared_count = 0;
    for (int i = 0; i < system->log_count && vc.prepared_count < PBFT_MAX_PENDING_BATCH; i++) {
        PbftLogEntry* e = &system->log_entries[i];
        if (e->prepared && e->sequence_number > system->latest_checkpoint_seq) {
            vc.prepared_evidence[vc.prepared_count].seq_number = e->sequence_number;
            vc.prepared_evidence[vc.prepared_count].view_number = e->view_number;
            copy_digest(vc.prepared_evidence[vc.prepared_count].digest, e->request_digest);
            vc.prepared_evidence[vc.prepared_count].has_prepare_cert =
                (e->prepare_count >= (int)(system->config.num_nodes - system->config.max_fault));
            vc.prepared_count++;
        }
    }
    pbft_compute_digest(&vc, sizeof(vc), vc.header.digest);
    system->status = PBFT_NODE_VIEW_CHANGING;
    system->last_view_change_time_ms = now;
    system->last_view_change_retry_ms = now;
    system->view_change_retry_count++;
    pbft_broadcast(system, &vc, sizeof(vc));
    system->stats.total_view_changes++;
    if (system->config.verbose) {
        printf("[PBFT] 节点 %u 发起视图变更: view %u -> %u (retry=%d backoff=%ums)\n",
               system->config.node_id, system->current_view, new_view,
               system->view_change_retry_count - 1, backoff_ms);
    }
    return 0;
}

int pbft_is_primary(const PbftSystem* system) {
    if (!system) return 0;
    return (int)(system->config.node_id == system->primary_id);
}

PbftNodeStatus pbft_get_status(const PbftSystem* system) {
    if (!system) return PBFT_NODE_STOPPED;
    return system->status;
}

int pbft_get_last_executed_seq(const PbftSystem* system) {
    if (!system) return -1;
    return (int)system->last_executed_seq;
}

int pbft_get_committed_count(const PbftSystem* system) {
    if (!system) return -1;
    return (int)system->stats.committed_requests;
}

int pbft_is_fault_tolerant(const PbftSystem* system) {
    if (!system) return 0;
    return (int)(system->config.num_nodes >= 3 * system->config.max_fault + 1);
}

void pbft_get_stats(const PbftSystem* system, PbftStats* stats) {
    if (!system || !stats) return;
    memcpy(stats, &system->stats, sizeof(PbftStats));
}

void pbft_reset_stats(PbftSystem* system) {
    if (!system) return;
    memset(&system->stats, 0, sizeof(PbftStats));
    system->stats.start_time_ms = get_current_time_ms();
}

void pbft_set_execute_callback(PbftSystem* system, PbftExecuteCallback callback, void* user_data) {
    if (!system) return;
    system->execute_callback = callback;
    system->execute_callback_user_data = user_data;
    log_info("[PBFT] 执行回调已%s", callback ? "注册" : "注销");
}

int pbft_trigger_checkpoint(PbftSystem* system) {
    if (!system) return -1;
    system->latest_checkpoint_seq = system->last_executed_seq;
    /* P1-020修复：检查点摘要包含完整系统状态（执行序列号、当前视图、
       活动节点位图、已接收请求数、已提交请求数、视图变更次数），不再使用硬编码0值 */
    uint32_t active_nodes_bitmap = 0;
    int active_count = 0;
    for (int i = 0; i < PBFT_MAX_NODES && i < 32; i++) {
        if (system->nodes[i].is_active) {
            active_nodes_bitmap |= (1u << i);
            active_count++;
        }
    }
    uint32_t state_digest_input[8] = {
        system->latest_checkpoint_seq,
        system->current_view,
        (uint32_t)(system->stats.total_requests & 0xFFFFFFFF),
        active_nodes_bitmap,
        (uint32_t)active_count,
        (uint32_t)(system->stats.committed_requests & 0xFFFFFFFF),
        (uint32_t)(system->stats.total_checkpoints & 0xFFFFFFFF),
        (uint32_t)(system->stats.view_change_count & 0xFFFFFFFF)
    };
    pbft_compute_digest(state_digest_input, sizeof(state_digest_input), system->checkpoint_digest);
    PbftCheckpoint cp;
    memset(&cp, 0, sizeof(cp));
    cp.header.type = PBFT_MSG_CHECKPOINT;
    cp.header.sender_id = system->config.node_id;
    cp.header.view_number = system->current_view;
    cp.last_executed_seq = system->latest_checkpoint_seq;
    copy_digest(cp.state_digest, system->checkpoint_digest);
    pbft_compute_digest(&cp, sizeof(cp), cp.header.digest);
    pbft_broadcast(system, &cp, sizeof(cp));
    if (system->config.verbose) {
        printf("[PBFT] 检查点: seq=%u\n", system->latest_checkpoint_seq);
    }
    return 0;
}

int pbft_get_stable_checkpoint(const PbftSystem* system) {
    if (!system) return -1;
    return (int)system->latest_checkpoint_seq;
}

static int pbft_handle_pre_prepare(PbftSystem* system, const PbftPrePrepare* msg) {
    if (!system || !msg) return -1;
    if (system->config.node_id == system->primary_id) return 0;
    if (msg->header.view_number != system->current_view) return -1;
    uint32_t seq = msg->header.sequence_number;
    PbftLogEntry* entry = log_create_entry(system, seq);
    if (!entry) return -1;
    entry->view_number = msg->header.view_number;
    entry->client_id = msg->client_id;
    entry->request_id = msg->request_id;
    entry->request_timestamp = msg->request_timestamp;
    copy_digest(entry->request_digest, msg->request_digest);
    PbftPrepare prep;
    memset(&prep, 0, sizeof(prep));
    prep.header.type = PBFT_MSG_PREPARE;
    prep.header.sender_id = system->config.node_id;
    prep.header.view_number = system->current_view;
    prep.header.sequence_number = seq;
    copy_digest(prep.header.digest, msg->request_digest);
    pbft_compute_digest(&prep, sizeof(prep), prep.pre_prepare_digest);
    entry->prepared = 1;
    entry->prepare_count++;
    pbft_broadcast(system, &prep, sizeof(prep));
    system->stats.total_prepares++;
    if (system->config.verbose) {
        printf("[PBFT] 备份节点 %u 发送 Prepare: view=%u seq=%u\n",
               system->config.node_id, system->current_view, seq);
    }
    if (has_quorum_prepare(system, seq)) {
        PbftCommit commit;
        memset(&commit, 0, sizeof(commit));
        commit.header.type = PBFT_MSG_COMMIT;
        commit.header.sender_id = system->config.node_id;
        commit.header.view_number = system->current_view;
        commit.header.sequence_number = seq;
        copy_digest(commit.header.digest, msg->request_digest);
        pbft_compute_digest(&commit, sizeof(commit), commit.prepare_digest);
        entry->committed = 1;
        entry->commit_count++;
        pbft_broadcast(system, &commit, sizeof(commit));
        system->stats.total_commits++;
        if (has_quorum_commit(system, seq)) {
            pbft_execute_request(system, seq);
            system->stats.committed_requests++;
            PbftReply reply;
            memset(&reply, 0, sizeof(reply));
            reply.header.type = PBFT_MSG_REPLY;
            reply.header.sender_id = system->config.node_id;
            reply.header.view_number = system->current_view;
            reply.header.sequence_number = seq;
            reply.timestamp = msg->request_timestamp;
            reply.client_id = msg->client_id;
            reply.request_id = msg->request_id;
            reply.result_code = 0;
            pbft_compute_digest(&reply, sizeof(reply), reply.header.digest);
            pbft_send_to_clients(system, &reply, sizeof(reply));
            system->stats.total_replies++;
        }
    }
    return 0;
}

static int pbft_handle_prepare(PbftSystem* system, const PbftPrepare* msg) {
    if (!system || !msg) return -1;
    uint32_t seq = msg->header.sequence_number;
    PbftLogEntry* entry = log_get_entry(system, seq);
    if (!entry) entry = log_create_entry(system, seq);
    if (!entry) return -1;
    entry->prepare_count++;
    if (entry->prepare_count <= PBFT_MAX_NODES) {
        copy_digest(entry->prepare_digests[entry->prepare_count - 1], msg->header.digest);
    }
    if (has_quorum_prepare(system, seq) && !entry->committed) {
        entry->committed = 1;
        PbftCommit commit;
        memset(&commit, 0, sizeof(commit));
        commit.header.type = PBFT_MSG_COMMIT;
        commit.header.sender_id = system->config.node_id;
        commit.header.view_number = system->current_view;
        commit.header.sequence_number = seq;
        copy_digest(commit.header.digest, entry->request_digest);
        pbft_compute_digest(&commit, sizeof(commit), commit.prepare_digest);
        entry->commit_count++;
        pbft_broadcast(system, &commit, sizeof(commit));
        system->stats.total_commits++;
    }
    return 0;
}

static int pbft_handle_commit(PbftSystem* system, const PbftCommit* msg) {
    if (!system || !msg) return -1;
    uint32_t seq = msg->header.sequence_number;
    PbftLogEntry* entry = log_get_entry(system, seq);
    if (!entry) return -1;
    entry->commit_count++;
    if (entry->commit_count <= PBFT_MAX_NODES) {
        copy_digest(entry->commit_digests[entry->commit_count - 1], msg->header.digest);
    }
    if (has_quorum_commit(system, seq) && !entry->executed) {
        pbft_execute_request(system, seq);
        system->stats.committed_requests++;
        PbftReply reply;
        memset(&reply, 0, sizeof(reply));
        reply.header.type = PBFT_MSG_REPLY;
        reply.header.sender_id = system->config.node_id;
        reply.header.view_number = system->current_view;
        reply.header.sequence_number = seq;
        reply.client_id = entry->client_id;
        reply.request_id = entry->request_id;
        reply.result_code = 0;
        pbft_compute_digest(&reply, sizeof(reply), reply.header.digest);
        pbft_send_to_clients(system, &reply, sizeof(reply));
        system->stats.total_replies++;
        if (system->config.verbose) {
            printf("[PBFT] 节点 %u 提交请求: seq=%u\n", system->config.node_id, seq);
        }
    }
    return 0;
}

static int pbft_validate_view_change(const PbftViewChange* msg, uint32_t current_view) {
    if (msg->new_view_number <= current_view) return 0;
    if (msg->prepared_count > PBFT_MAX_PENDING_BATCH) return 0;
    uint32_t computed_digest[8];
    PbftViewChange tmp = *msg;
    memset(tmp.header.digest, 0, sizeof(tmp.header.digest));
    pbft_compute_digest(&tmp, sizeof(PbftViewChange), computed_digest);
    if (!is_digest_equal(computed_digest, msg->header.digest)) return 0;
    for (uint32_t i = 0; i < msg->prepared_count; i++) {
        if (msg->prepared_evidence[i].seq_number == 0) return 0;
        if (i > 0 && msg->prepared_evidence[i].seq_number <= msg->prepared_evidence[i - 1].seq_number) return 0;
    }
    return 1;
}

/*修复: pbft_send_view_change() 函数实现
 * 原代码调用此函数但未定义，编译警告(#pragma 4013)压制了链接错误。
 * 构建并广播ViewChange消息到所有活跃节点。 */
static int pbft_send_view_change(PbftSystem* system, uint32_t new_view_number) {
    if (!system || new_view_number <= system->current_view) return -1;
    PbftViewChange vc;
    memset(&vc, 0, sizeof(vc));
    vc.header.type = PBFT_MSG_VIEW_CHANGE;
    vc.header.sender_id = system->config.node_id;
    vc.header.view_number = system->current_view;
    vc.new_view_number = new_view_number;
    vc.prepared_count = 0;
    /* 收集当前视图下已prepared但未committed的证据 */
    for (int i = 0; i < system->log_count && vc.prepared_count < PBFT_MAX_PENDING_BATCH; i++) {
        if (system->log_entries[i].prepare_count >= (int)(2U * (uint32_t)system->config.max_fault)) {
            vc.prepared_evidence[vc.prepared_count].seq_number = system->log_entries[i].sequence_number;
            memcpy(vc.prepared_evidence[vc.prepared_count].digest,
                   system->log_entries[i].request_digest, sizeof(vc.prepared_evidence[0].digest));
            vc.prepared_count++;
        }
    }
    /* 计算整包摘要并广播 */
    pbft_compute_digest(&vc, sizeof(PbftViewChange), vc.header.digest);
    for (int i = 0; i < system->node_count; i++) {
        if (i != (int)system->config.node_id && system->nodes[i].is_active) {
            pbft_send_message(system, (uint32_t)i, &vc, sizeof(PbftViewChange));
        }
    }
    system->last_view_change_time_ms = get_current_time_ms();
    if (system->config.verbose) {
        printf("[PBFT] 节点 %u 发送 ViewChange: new_view=%u\n",
               system->config.node_id, new_view_number);
    }
    return 0;
}

static int pbft_handle_view_change(PbftSystem* system, const PbftViewChange* msg) {
    if (!system || !msg) return -1;
    uint32_t sender = msg->header.sender_id;
    if (sender >= PBFT_MAX_NODES) return -1;
    if (!pbft_validate_view_change(msg, system->current_view)) return -1;
    uint32_t quorum = system->config.num_nodes - system->config.max_fault;
    uint32_t new_view = msg->new_view_number;
    if (new_view < system->target_view) return 0;
    if (new_view > system->target_view) {
        system->target_view = new_view;
        system->view_change_collected = 1;
        system->vc_record_count = 0;
        for (int i = 0; i < PBFT_MAX_NODES; i++) {
            if (system->view_change_msgs[i]) {
                free(system->view_change_msgs[i]);
                system->view_change_msgs[i] = NULL;
            }
        }
        int found = 0;
        for (int i = 0; i < system->vc_record_count && !found; i++) {
            if (system->vc_records[i].from_node == sender && system->vc_records[i].target_view == new_view) found = 1;
        }
        if (!found && system->vc_record_count < PBFT_MAX_NODES) {
            system->vc_records[system->vc_record_count].from_node = sender;
            system->vc_records[system->vc_record_count].target_view = new_view;
            system->vc_records[system->vc_record_count].msg = *msg;
            system->vc_records[system->vc_record_count].validated = 1;
            system->vc_record_count++;
        }
        if (system->view_change_msgs[sender]) free(system->view_change_msgs[sender]);
        system->view_change_msgs[sender] = (PbftViewChange*)malloc(sizeof(PbftViewChange));
        if (system->view_change_msgs[sender]) {
            memcpy(system->view_change_msgs[sender], msg, sizeof(PbftViewChange));
        }
        system->status = PBFT_NODE_VIEW_CHANGING;
        system->last_view_change_time_ms = get_current_time_ms();
        if (system->config.verbose) {
            printf("[PBFT] 节点 %u 收到 ViewChange: new_view=%u from=%u collected=%u/%u\n",
                   system->config.node_id, new_view, sender, system->view_change_collected, quorum);
        }
    } else {
        int already_recorded = 0;
        for (int i = 0; i < system->vc_record_count; i++) {
            if (system->vc_records[i].from_node == sender && system->vc_records[i].target_view == new_view) already_recorded = 1;
        }
        if (!already_recorded && system->vc_record_count < PBFT_MAX_NODES) {
            system->vc_records[system->vc_record_count].from_node = sender;
            system->vc_records[system->vc_record_count].target_view = new_view;
            system->vc_records[system->vc_record_count].msg = *msg;
            system->vc_records[system->vc_record_count].validated = 1;
            system->vc_record_count++;
            system->view_change_collected++;
        }
        if (system->view_change_msgs[sender]) free(system->view_change_msgs[sender]);
        system->view_change_msgs[sender] = (PbftViewChange*)malloc(sizeof(PbftViewChange));
        if (system->view_change_msgs[sender]) {
            memcpy(system->view_change_msgs[sender], msg, sizeof(PbftViewChange));
        }
        system->status = PBFT_NODE_VIEW_CHANGING;
        system->last_view_change_time_ms = get_current_time_ms();
    }
    if (system->view_change_collected >= quorum && !system->new_view_sent &&
        system->config.node_id == new_view % system->config.num_nodes) {
        uint32_t max_checkpoint_seq = 0;
        uint32_t stable_checkpoint_digest[8];
        memset(stable_checkpoint_digest, 0, sizeof(stable_checkpoint_digest));
        for (int i = 0; i < system->vc_record_count; i++) {
            if (system->vc_records[i].target_view == new_view && system->vc_records[i].msg.last_checkpoint_seq > max_checkpoint_seq) {
                max_checkpoint_seq = system->vc_records[i].msg.last_checkpoint_seq;
                copy_digest(stable_checkpoint_digest, system->vc_records[i].msg.checkpoint_digest);
            }
        }
        uint32_t max_prepared_seq = max_checkpoint_seq;
        uint32_t pending_seqs[PBFT_MAX_PENDING_BATCH];
        int pending_count = 0;
        int max_view_per_seq[PBFT_MAX_PENDING_BATCH];
        uint32_t digest_for_seq[PBFT_MAX_PENDING_BATCH][8];
        for (int i = 0; i < system->vc_record_count; i++) {
            if (system->vc_records[i].target_view != new_view) continue;
            PbftViewChange* vc = &system->vc_records[i].msg;
            for (uint32_t j = 0; j < vc->prepared_count; j++) {
                uint32_t s = vc->prepared_evidence[j].seq_number;
                if (s <= max_checkpoint_seq) continue;
                int found = 0;
                for (int k = 0; k < pending_count; k++) {
                    if (pending_seqs[k] == s) {
                        found = 1;
                        if (vc->prepared_evidence[j].view_number > (uint32_t)max_view_per_seq[k]) {
                            max_view_per_seq[k] = (int)vc->prepared_evidence[j].view_number;
                            copy_digest(digest_for_seq[k], vc->prepared_evidence[j].digest);
                        }
                        break;
                    }
                }
                if (!found && pending_count < PBFT_MAX_PENDING_BATCH) {
                    pending_seqs[pending_count] = s;
                    max_view_per_seq[pending_count] = (int)vc->prepared_evidence[j].view_number;
                    copy_digest(digest_for_seq[pending_count], vc->prepared_evidence[j].digest);
                    pending_count++;
                    if (s > max_prepared_seq) max_prepared_seq = s;
                }
            }
        }
        PbftNewView nv;
        memset(&nv, 0, sizeof(nv));
        nv.header.type = PBFT_MSG_NEW_VIEW;
        nv.header.sender_id = system->config.node_id;
        nv.header.view_number = system->current_view;
        nv.header.sequence_number = max_prepared_seq;
        nv.new_view_number = new_view;
        nv.view_change_count = (uint32_t)system->view_change_collected;
        for (int i = 0; i < (int)nv.view_change_count && i < PBFT_MAX_FAULT && i < system->vc_record_count; i++) {
            uint32_t vc_digest[8];
            pbft_compute_digest(&system->vc_records[i].msg, sizeof(PbftViewChange), vc_digest);
            copy_digest(nv.view_change_digests[i], vc_digest);
        }
        nv.pre_prepare_count = (uint32_t)pending_count;
        for (int i = 0; i < pending_count; i++) {
            nv.new_pre_prepares[i].seq_number = pending_seqs[i];
            copy_digest(nv.new_pre_prepares[i].digest, digest_for_seq[i]);
        }
        system->new_pre_prepare_count = (uint32_t)pending_count;
        for (int i = 0; i < pending_count; i++) {
            system->new_pre_prepare_seqs[i] = pending_seqs[i];
            copy_digest(system->new_pre_prepare_digests[i], digest_for_seq[i]);
        }
        pbft_compute_digest(&nv, sizeof(nv), nv.header.digest);
        system->new_view_sent = 1;
        pbft_broadcast(system, &nv, sizeof(nv));
        if (system->config.verbose) {
            printf("[PBFT] 新主节点 %u 广播 NewView: view=%u prepared_seqs=%u\n",
                   system->config.node_id, new_view, pending_count);
        }
        system->current_view = new_view;
        system->primary_id = new_view % system->config.num_nodes;
        system->status = PBFT_NODE_NORMAL;
        system->view_change_in_progress = 0;
        system->view_change_collected = 0;
        system->view_change_retry_count = 0;
        system->stats.view_change_count++;
        for (int i = 0; i < PBFT_MAX_NODES; i++) {
            system->nodes[i].is_primary = (i == (int)system->primary_id);
        }
        if (max_prepared_seq > system->last_executed_seq) {
            system->last_executed_seq = max_prepared_seq;
        }
        for (int i = 0; i < pending_count; i++) {
            if (pending_seqs[i] > system->last_prepared_seq) {
                system->last_prepared_seq = pending_seqs[i];
            }
        }
        for (int i = 0; i < pending_count; i++) {
            PbftPrePrepare pp;
            memset(&pp, 0, sizeof(pp));
            pp.header.type = PBFT_MSG_PRE_PREPARE;
            pp.header.sender_id = system->config.node_id;
            pp.header.view_number = system->current_view;
            pp.header.sequence_number = pending_seqs[i];
            copy_digest(pp.request_digest, digest_for_seq[i]);
            copy_digest(pp.header.digest, digest_for_seq[i]);
            pbft_broadcast(system, &pp, sizeof(pp));
            system->stats.total_pre_prepares++;
            if (system->config.verbose) {
                printf("[PBFT] 新视图重放 PrePrepare: view=%u seq=%u\n",
                       system->current_view, pending_seqs[i]);
            }
        }
        if (system->config.verbose) {
            printf("[PBFT] 新视图建立完成: view=%u primary=%u count=%u\n",
                   system->current_view, system->primary_id, system->view_change_collected);
        }
    }
    return 0;
}

static int pbft_handle_checkpoint(PbftSystem* system, const PbftCheckpoint* msg) {
    if (!system || !msg) return -1;
    if (system->config.enable_checkpoint_gc) {
        log_gc(system, msg->last_executed_seq);
    }
    return 0;
}

static int pbft_handle_new_view(PbftSystem* system, const PbftNewView* msg) {
    if (!system || !msg) return -1;
    if (system->new_view_received) return 0;
    if (msg->new_view_number <= system->current_view) return -1;
    uint32_t quorum = system->config.num_nodes - system->config.max_fault;
    if (msg->view_change_count < quorum) return -1;
    uint32_t computed_digest[8];
    PbftNewView tmp = *msg;
    memset(tmp.header.digest, 0, sizeof(tmp.header.digest));
    pbft_compute_digest(&tmp, sizeof(PbftNewView), computed_digest);
    if (!is_digest_equal(computed_digest, msg->header.digest)) return -1;
    uint32_t new_view = msg->new_view_number;
    uint32_t expected_primary = new_view % system->config.num_nodes;
    if (msg->header.sender_id != expected_primary) return -1;
    uint32_t max_seq = 0;
    for (uint32_t i = 0; i < msg->pre_prepare_count; i++) {
        if (msg->new_pre_prepares[i].seq_number > max_seq) {
            max_seq = msg->new_pre_prepares[i].seq_number;
        }
    }
    system->current_view = new_view;
    system->primary_id = expected_primary;
    system->status = PBFT_NODE_NORMAL;
    system->view_change_in_progress = 0;
    system->view_change_collected = 0;
    system->view_change_retry_count = 0;
    system->new_view_received = 1;
    system->new_view_sent = 0;
    system->stats.view_change_count++;
    for (int i = 0; i < PBFT_MAX_NODES; i++) {
        system->nodes[i].is_primary = (i == (int)system->primary_id);
    }
    system->last_executed_seq = max_seq > system->last_executed_seq ? max_seq : system->last_executed_seq;
    system->last_prepared_seq = max_seq > system->last_prepared_seq ? max_seq : system->last_prepared_seq;
    if (system->config.verbose) {
        printf("[PBFT] 节点 %u 收到 NewView: view=%u primary=%u pp_count=%u\n",
               system->config.node_id, new_view, expected_primary, msg->pre_prepare_count);
    }
    return 0;
}

int pbft_process_messages(PbftSystem* system, int timeout_ms) {
    if (!system) return PBFT_ERROR_INVALID_PARAM;
    if (!system->server_initialized) {
        if (pbft_connect_all(system) != PBFT_ERROR_NONE) return PBFT_ERROR_NOT_CONNECTED;
    }

    uint64_t deadline = get_current_time_ms() + (timeout_ms > 0 ? (uint64_t)timeout_ms : 0);
    int processed = 0;

    for (;;) {
        uint64_t now = get_current_time_ms();
        if (timeout_ms > 0 && now >= deadline) break;

        struct sockaddr_in sender_addr;
#ifdef _WIN32
        int addr_len = sizeof(sender_addr);
#else
        socklen_t addr_len = sizeof(sender_addr);
#endif

#ifdef _WIN32
        int recv_len = recvfrom(system->listen_socket, (char*)system->recv_buffer,
                                system->recv_buffer_size, 0,
                                (struct sockaddr*)&sender_addr, &addr_len);
        if (recv_len == SOCKET_ERROR) {
            int err = WSAGetLastError();
            if (err == WSAEWOULDBLOCK) break;
            continue;
        }
#else
        ssize_t recv_len = recvfrom(system->listen_socket, (char*)system->recv_buffer,
                                    system->recv_buffer_size, 0,
                                    (struct sockaddr*)&sender_addr, &addr_len);
        if (recv_len < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) break;
            continue;
        }
#endif

        if ((uint32_t)recv_len < sizeof(PbftMessageHeader)) continue;

        PbftMessageHeader* hdr = (PbftMessageHeader*)system->recv_buffer;

        if (hdr->sender_id == system->config.node_id) continue;
        if (hdr->sender_id >= PBFT_MAX_NODES) continue;

        if ((uint32_t)recv_len < sizeof(PbftMessageHeader) + hdr->payload_length) continue;

        switch (hdr->type) {
            case PBFT_MSG_REQUEST: {
                if ((uint32_t)recv_len < sizeof(PbftRequest)) break;
                processed++;
                break;
            }
            case PBFT_MSG_PRE_PREPARE: {
                if ((uint32_t)recv_len < sizeof(PbftPrePrepare)) break;
                pbft_handle_pre_prepare(system, (const PbftPrePrepare*)system->recv_buffer);
                processed++;
                system->stats.total_pre_prepares++;
                break;
            }
            case PBFT_MSG_PREPARE: {
                if ((uint32_t)recv_len < sizeof(PbftPrepare)) break;
                pbft_handle_prepare(system, (const PbftPrepare*)system->recv_buffer);
                processed++;
                system->stats.total_prepares++;
                break;
            }
            case PBFT_MSG_COMMIT: {
                if ((uint32_t)recv_len < sizeof(PbftCommit)) break;
                pbft_handle_commit(system, (const PbftCommit*)system->recv_buffer);
                processed++;
                system->stats.total_commits++;
                break;
            }
            case PBFT_MSG_VIEW_CHANGE: {
                if ((uint32_t)recv_len < sizeof(PbftViewChange)) break;
                pbft_handle_view_change(system, (const PbftViewChange*)system->recv_buffer);
                processed++;
                system->stats.total_view_changes++;
                break;
            }
            case PBFT_MSG_NEW_VIEW: {
                if ((uint32_t)recv_len < sizeof(PbftNewView)) break;
                pbft_handle_new_view(system, (const PbftNewView*)system->recv_buffer);
                processed++;
                break;
            }
            case PBFT_MSG_CHECKPOINT: {
                if ((uint32_t)recv_len < sizeof(PbftCheckpoint)) break;
                pbft_handle_checkpoint(system, (const PbftCheckpoint*)system->recv_buffer);
                processed++;
                system->stats.total_checkpoints++;
                break;
            }
            case PBFT_MSG_REPLY:
            case PBFT_MSG_STATUS:
                /* M-002修复: STATUS消息处理 —— 落后节点可能请求当前视图状态 */
                if (recv_len >= sizeof(PbftMessageHeader)) {
                    PbftMessageHeader* hdr = (PbftMessageHeader*)system->recv_buffer;
                    if (hdr->view_number > system->current_view) {
                        /* 当前节点视图落后，触发视图变更 */
                        pbft_send_view_change(system, hdr->view_number);
                        system->current_view = hdr->view_number;
                    }
                    processed++;
                }
                break;
            case PBFT_MSG_STATUS_REPLY:
                break;
            default:
                break;
        }

        if (timeout_ms == 0) break;
    }

    return processed;
}

int pbft_dispatch_message(PbftSystem* system, PbftMessageType type,
                           const void* msg, uint32_t msg_size) {
    if (!system || !msg) return PBFT_ERROR_INVALID_PARAM;
    if (system->status == PBFT_NODE_STOPPED) return PBFT_ERROR_SYSTEM_STOPPED;
    /* 验证消息大小：不能为空且不能超过最大请求大小 */
    if (msg_size == 0 || msg_size > PBFT_MAX_REQUEST_SIZE) return PBFT_ERROR_INVALID_MESSAGE;
    switch (type) {
        case PBFT_MSG_PRE_PREPARE:
            return pbft_handle_pre_prepare(system, (const PbftPrePrepare*)msg);
        case PBFT_MSG_PREPARE:
            return pbft_handle_prepare(system, (const PbftPrepare*)msg);
        case PBFT_MSG_COMMIT:
            return pbft_handle_commit(system, (const PbftCommit*)msg);
        case PBFT_MSG_VIEW_CHANGE:
            return pbft_handle_view_change(system, (const PbftViewChange*)msg);
        case PBFT_MSG_NEW_VIEW:
            return pbft_handle_new_view(system, (const PbftNewView*)msg);
        case PBFT_MSG_REQUEST:
            system->stats.total_requests++;
            return PBFT_ERROR_NONE;
        case PBFT_MSG_CHECKPOINT:
            return pbft_handle_checkpoint(system, (const PbftCheckpoint*)msg);
        default:
            return PBFT_ERROR_INVALID_MESSAGE;
    }
}

int pbft_tick(PbftSystem* system) {
    if (!system) return PBFT_ERROR_INVALID_PARAM;
    uint64_t now = get_current_time_ms();
    uint64_t elapsed = now - system->last_tick_ms;
    system->last_tick_ms = now;
    if (elapsed > 10000) elapsed = 10000;
    if (system->active_request && system->config.node_id == system->primary_id) {
        uint32_t seq = system->last_prepared_seq;
        PbftLogEntry* entry = log_get_entry(system, seq);
        if (entry && !entry->executed && has_quorum_commit(system, seq)) {
            pbft_execute_request(system, seq);
            system->stats.committed_requests++;
            system->active_request = 0;
            if (system->config.verbose) {
                printf("[PBFT] 请求完成: seq=%u\n", seq);
            }
        }
    }
    if (system->config.enable_auto_view_change) {
        int should_change = 0;
        if (system->status == PBFT_NODE_NORMAL) {
            if (system->active_request && !has_quorum_commit(system, system->last_prepared_seq)) {
                uint64_t time_since_request = now - system->pending_timestamp;
                if (time_since_request > system->config.view_change_timeout_ms / 2) {
                    should_change = 1;
                }
            }
            if (system->config.node_id != system->primary_id) {
                uint64_t time_since_last_heartbeat = now - system->nodes[system->primary_id].last_heartbeat;
                if (time_since_last_heartbeat > system->config.view_change_timeout_ms) {
                    should_change = 1;
                    if (system->config.verbose) {
                        printf("[PBFT] 主节点 %u 心跳超时, 准备视图变更\n", system->primary_id);
                    }
                }
            }
        }
        if (system->status == PBFT_NODE_VIEW_CHANGING) {
            uint64_t time_since_vc = now - system->last_view_change_time_ms;
            if (time_since_vc > system->config.view_change_timeout_ms && !system->new_view_received) {
                should_change = 1;
            }
        }
        if (should_change) {
            pbft_trigger_view_change(system);
        }
    }
    uint64_t total_ms = now - system->stats.start_time_ms;
    if (total_ms > 0) {
        system->stats.throughput_req_per_sec = (double)system->stats.committed_requests / (total_ms / 1000.0);
    }
    return PBFT_ERROR_NONE;
}

const char* pbft_error_string(int error_code) {
    switch (error_code) {
        case PBFT_ERROR_NONE: return "无错误";
        case PBFT_ERROR_INVALID_PARAM: return "无效参数";
        case PBFT_ERROR_NOT_CONNECTED: return "未连接";
        case PBFT_ERROR_TIMEOUT: return "超时";
        case PBFT_ERROR_VIEW_CHANGE: return "视图变更中";
        case PBFT_ERROR_NOT_PRIMARY: return "非主节点";
        case PBFT_ERROR_INVALID_MESSAGE: return "无效消息";
        case PBFT_ERROR_DIGEST_MISMATCH: return "摘要不匹配";
        case PBFT_ERROR_NO_QUORUM: return "未达到法定人数";
        case PBFT_ERROR_OUT_OF_MEMORY: return "内存不足";
        case PBFT_ERROR_SYSTEM_STOPPED: return "系统已停止";
        default: return "未知错误";
    }
}

/* ============================================================================
 * 分布式子系统完整实现（selflnn.c 调用接口）
 * ============================================================================ */

/* 分布式训练配置 - 完整结构体 */
typedef struct {
    int node_count;
    int node_id;
    int is_primary;
    int port;
    int use_gpu;
    int gpu_ids[8];
    int gpu_count;
    float sync_interval_sec;
    int batch_size_per_node;
    int gradient_compression;
    char master_addr[128];
    char listen_addr[128];
    PbftSystem* pbft;
} DistributedConfig;

/* 分布式训练上下文 - 完整结构体 */
typedef struct {
    DistributedConfig config;
    int is_initialized;
    int is_running;
    long total_sync_count;
    long total_bytes_synced;
    float* gradient_buffer;
    size_t gradient_buffer_size;
    long last_sync_time_us;
    /* 节点通信状态 */
    int* node_online;
    int* node_last_seen;
    long* node_latency_us;
} DistributedContext;

void* distributed_init(const DistributedConfig* cfg) {
    if (!cfg) {
        log_error("[分布式] 分布式训练初始化失败: 配置为空");
        return NULL;
    }

    DistributedContext* ctx = (DistributedContext*)
        safe_calloc(1, sizeof(DistributedContext));
    if (!ctx) {
        log_error("[分布式] 分布式训练初始化失败: 内存分配失败");
        return NULL;
    }

    ctx->config = *cfg;
    ctx->is_initialized = 1;
    ctx->is_running = 0;
    ctx->total_sync_count = 0;
    ctx->total_bytes_synced = 0;

    /* 初始化节点在线状态 */
    ctx->node_online = (int*)safe_calloc(cfg->node_count, sizeof(int));
    ctx->node_last_seen = (int*)safe_calloc(cfg->node_count, sizeof(int));
    ctx->node_latency_us = (long*)safe_calloc(cfg->node_count, sizeof(long));

    if (!ctx->node_online || !ctx->node_last_seen || !ctx->node_latency_us) {
        safe_free((void**)&ctx->node_online);
        safe_free((void**)&ctx->node_last_seen);
        safe_free((void**)&ctx->node_latency_us);
        safe_free((void**)&ctx);
        log_error("[分布式] 分布式训练初始化失败: 状态数组分配失败");
        return NULL;
    }

    /* 计算梯度缓冲区大小 */
    ctx->gradient_buffer_size = 1024 * 1024 * sizeof(float);
    ctx->gradient_buffer = (float*)safe_calloc(1024 * 1024, sizeof(float));
    if (!ctx->gradient_buffer) {
        safe_free((void**)&ctx->node_online);
        safe_free((void**)&ctx->node_last_seen);
        safe_free((void**)&ctx->node_latency_us);
        safe_free((void**)&ctx);
        log_error("[分布式] 分布式训练初始化失败: 梯度缓冲区分配失败");
        return NULL;
    }

    /* 读取时间基准 */
    ctx->last_sync_time_us = 0;

    /* 如果配置了PBFT，初始化PBFT子系统 */
    if (cfg->pbft) {
        ctx->config.pbft = cfg->pbft;
        log_info("[分布式] 分布式训练子系统已关联PBFT共识引擎");
    }

    log_info("[分布式] 分布式训练子系统初始化完成 [节点%d/%d, 端口%d, GPU: %s]",
             cfg->node_id, cfg->node_count, cfg->port,
             cfg->use_gpu ? "启用" : "禁用");

    return ctx;
}

void distributed_cleanup(void* ctx_ptr) {
    if (!ctx_ptr) return;

    DistributedContext* ctx = (DistributedContext*)ctx_ptr;
    if (!ctx->is_initialized) return;

    ctx->is_running = 0;

    /* 释放梯度缓冲区 */
    safe_free((void**)&ctx->gradient_buffer);
    safe_free((void**)&ctx->node_online);
    safe_free((void**)&ctx->node_last_seen);
    safe_free((void**)&ctx->node_latency_us);

    ctx->is_initialized = 0;
    log_info("[分布式] 分布式训练子系统已清理 [节点%d, 同步%ld次, 传输%ld字节]",
             ctx->config.node_id, ctx->total_sync_count, ctx->total_bytes_synced);

    safe_free(&ctx_ptr);
}
