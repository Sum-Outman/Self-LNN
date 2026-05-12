#ifndef SELFLNN_PBFT_H
#define SELFLNN_PBFT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* PBFT 系统常数 */
#define PBFT_MAX_NODES 64
#define PBFT_MAX_FAULT 21
#define PBFT_MAX_PENDING_BATCH 256
#define PBFT_DIGEST_SIZE 32
#define PBFT_MAX_REQUEST_SIZE 65536
#define PBFT_CHECKPOINT_INTERVAL 128
#define PBFT_VIEW_CHANGE_TIMEOUT_MS 5000
#define PBFT_DEFAULT_PORT 18765

/* PBFT 节点状态 */
typedef enum {
    PBFT_NODE_NORMAL,
    PBFT_NODE_VIEW_CHANGING,
    PBFT_NODE_STOPPED
} PbftNodeStatus;

/* PBFT 消息类型 */
typedef enum {
    PBFT_MSG_REQUEST,
    PBFT_MSG_PRE_PREPARE,
    PBFT_MSG_PREPARE,
    PBFT_MSG_COMMIT,
    PBFT_MSG_REPLY,
    PBFT_MSG_VIEW_CHANGE,
    PBFT_MSG_NEW_VIEW,
    PBFT_MSG_CHECKPOINT,
    PBFT_MSG_STATUS,
    PBFT_MSG_STATUS_REPLY
} PbftMessageType;

/* PBFT 节点角色 */
typedef enum {
    PBFT_ROLE_PRIMARY,
    PBFT_ROLE_BACKUP,
    PBFT_ROLE_CLIENT
} PbftNodeRole;

/* PBFT 消息头部 */
typedef struct {
    PbftMessageType type;
    uint32_t sender_id;
    uint32_t view_number;
    uint32_t sequence_number;
    uint32_t digest[8];
    uint32_t payload_length;
    uint32_t reserved;
} PbftMessageHeader;

/* PBFT 请求消息 */
typedef struct {
    PbftMessageHeader header;
    uint64_t timestamp;
    uint32_t client_id;
    uint32_t request_id;
    uint32_t operation_type;
    uint32_t payload_size;
} PbftRequest;

/* PBFT 预准备消息 */
typedef struct {
    PbftMessageHeader header;
    uint32_t request_digest[8];
    uint64_t request_timestamp;
    uint32_t client_id;
    uint32_t request_id;
} PbftPrePrepare;

/* PBFT 准备消息 */
typedef struct {
    PbftMessageHeader header;
    uint32_t pre_prepare_digest[8];
} PbftPrepare;

/* PBFT 提交消息 */
typedef struct {
    PbftMessageHeader header;
    uint32_t prepare_digest[8];
} PbftCommit;

/* PBFT 回复消息 */
typedef struct {
    PbftMessageHeader header;
    uint64_t timestamp;
    uint32_t client_id;
    uint32_t request_id;
    uint32_t result_code;
    uint32_t result_size;
} PbftReply;

/* PBFT 视图变更消息 */
typedef struct {
    PbftMessageHeader header;
    uint32_t new_view_number;
    uint32_t last_checkpoint_seq;
    uint32_t checkpoint_digest[8];
    uint32_t prepared_count;
    struct {
        uint32_t seq_number;
        uint32_t view_number;
        uint32_t digest[8];
        int has_prepare_cert;
    } prepared_evidence[PBFT_MAX_PENDING_BATCH];
} PbftViewChange;

/* PBFT 新视图消息 */
typedef struct {
    PbftMessageHeader header;
    uint32_t new_view_number;
    uint32_t view_change_count;
    uint32_t view_change_digests[PBFT_MAX_FAULT][8];
    uint32_t pre_prepare_count;
    struct {
        uint32_t seq_number;
        uint32_t digest[8];
    } new_pre_prepares[PBFT_MAX_PENDING_BATCH];
} PbftNewView;

/* PBFT 检查点消息 */
typedef struct {
    PbftMessageHeader header;
    uint32_t last_executed_seq;
    uint32_t state_digest[8];
} PbftCheckpoint;

/* PBFT 日志条目 */
typedef struct {
    uint32_t sequence_number;
    uint32_t view_number;
    uint32_t request_digest[8];
    uint64_t request_timestamp;
    uint32_t client_id;
    uint32_t request_id;
    int prepared;
    int committed;
    int executed;
    int prepare_count;
    uint32_t prepare_digests[PBFT_MAX_NODES][8];
    int commit_count;
    uint32_t commit_digests[PBFT_MAX_NODES][8];
} PbftLogEntry;

/* PBFT 节点信息 */
typedef struct {
    uint32_t node_id;
    char host[64];
    uint16_t port;
    int is_active;
    int is_primary;
    uint64_t last_heartbeat;
    int consecutive_timeouts;
} PbftNodeInfo;

/* PBFT 节点配置 */
typedef struct {
    uint32_t node_id;
    uint32_t num_nodes;
    uint32_t max_fault;
    uint32_t checkpoint_interval;
    uint32_t view_change_timeout_ms;
    uint16_t listen_port;
    char listen_host[64];
    int enable_auto_view_change;
    int enable_checkpoint_gc;
    int verbose;
} PbftConfig;

/* PBFT 统计信息 */
typedef struct {
    uint64_t total_requests;
    uint64_t total_pre_prepares;
    uint64_t total_prepares;
    uint64_t total_commits;
    uint64_t total_replies;
    uint64_t total_view_changes;
    uint64_t total_checkpoints;
    uint64_t committed_requests;
    uint64_t failed_requests;
    uint64_t view_change_count;
    double avg_commit_latency_ms;
    double throughput_req_per_sec;
    uint64_t start_time_ms;
    uint64_t last_metric_time_ms;
    uint32_t recent_latencies[1024];
    uint32_t recent_latency_count;
} PbftStats;

/* PBFT 系统 */
typedef struct PbftSystem PbftSystem;

/* 创建和销毁 */
PbftSystem* pbft_system_create(const PbftConfig* config);
void pbft_system_destroy(PbftSystem* system);

/* 节点管理 */
int pbft_add_node(PbftSystem* system, uint32_t node_id, const char* host, uint16_t port);
int pbft_remove_node(PbftSystem* system, uint32_t node_id);
int pbft_connect_all(PbftSystem* system);
int pbft_disconnect_all(PbftSystem* system);

/* 客户端请求 */
int pbft_submit_request(PbftSystem* system, uint32_t client_id,
                        uint32_t request_id, uint32_t operation_type,
                        const void* payload, uint32_t payload_size,
                        void* result_buffer, uint32_t* result_size);
int pbft_submit_request_async(PbftSystem* system, uint32_t client_id,
                              uint32_t request_id, uint32_t operation_type,
                              const void* payload, uint32_t payload_size);

/* 视图管理 */
int pbft_get_current_view(const PbftSystem* system);
uint32_t pbft_get_primary_id(const PbftSystem* system);
int pbft_trigger_view_change(PbftSystem* system);
int pbft_is_primary(const PbftSystem* system);

/* 状态查询 */
PbftNodeStatus pbft_get_status(const PbftSystem* system);
int pbft_get_last_executed_seq(const PbftSystem* system);
int pbft_get_committed_count(const PbftSystem* system);
int pbft_is_fault_tolerant(const PbftSystem* system);

/* 统计信息 */
void pbft_get_stats(const PbftSystem* system, PbftStats* stats);
void pbft_reset_stats(PbftSystem* system);

/* 检查点 */
int pbft_trigger_checkpoint(PbftSystem* system);
int pbft_get_stable_checkpoint(const PbftSystem* system);

/* 运行循环 */
int pbft_process_messages(PbftSystem* system, int timeout_ms);
int pbft_tick(PbftSystem* system);

/* 消息分发 */
int pbft_dispatch_message(PbftSystem* system, PbftMessageType type,
                           const void* msg, uint32_t msg_size);

/* 序列化辅助 */
void pbft_compute_digest(const void* data, uint32_t length, uint32_t digest[8]);
int pbft_verify_digest(const void* data, uint32_t length, const uint32_t expected[8]);

/* 配置辅助 */
void pbft_default_config(PbftConfig* config);

/* 错误字符串 */
const char* pbft_error_string(int error_code);

#define PBFT_ERROR_NONE 0
#define PBFT_ERROR_INVALID_PARAM -1
#define PBFT_ERROR_NOT_CONNECTED -2
#define PBFT_ERROR_TIMEOUT -3
#define PBFT_ERROR_VIEW_CHANGE -4
#define PBFT_ERROR_NOT_PRIMARY -5
#define PBFT_ERROR_INVALID_MESSAGE -6
#define PBFT_ERROR_DIGEST_MISMATCH -7
#define PBFT_ERROR_NO_QUORUM -8
#define PBFT_ERROR_OUT_OF_MEMORY -9
#define PBFT_ERROR_SYSTEM_STOPPED -10

#ifdef __cplusplus
}
#endif

#endif
