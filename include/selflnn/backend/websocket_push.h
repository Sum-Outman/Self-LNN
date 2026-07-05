/**
 * @file websocket_push.h
 * @brief WebSocket实时推送服务器接口
 * 
 * 位于backend子系统：提供WebSocket实时数据推送，用于前端仪表盘、
 * 训练监控、系统状态等场景的实时数据推送。
 */

#ifndef SELFLNN_WEBSOCKET_PUSH_H
#define SELFLNN_WEBSOCKET_PUSH_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WS_MAX_CLIENTS 64
#define WS_MAX_MESSAGE_SIZE 262144
#define WS_SEND_BUFFER_SIZE 524288

typedef struct WSPushServer WSPushServer;

/* C-002修复: WebSocket客户端消息回调类型
 * 用于服务器接收客户端WebSocket文本/二进制消息后分发给上层处理
 * 参数: client_index - 发送消息的客户端索引
 *       data         - 消息内容（opcode=0x01为文本UTF-8，0x02为二进制）
 *       data_len     - 消息内容字节长度
 *       opcode       - WebSocket帧opcode（0x01文本或0x02二进制）
 *       user_data    - 注册回调时传入的用户数据指针 */
typedef void (*WSClientMessageHandler)(int client_index, const unsigned char* data,
                                        size_t data_len, uint8_t opcode, void* user_data);

/* C-002修复: 注册客户端消息回调（在ws_push_server_start之前调用） */
void ws_push_set_message_handler(WSPushServer* server, WSClientMessageHandler handler, void* user_data);

typedef enum {
    WS_MSG_TRAINING_PROGRESS = 0,
    WS_MSG_SYSTEM_STATUS = 1,
    WS_MSG_KNOWLEDGE_UPDATE = 2,
    WS_MSG_MODEL_OUTPUT = 3,
    WS_MSG_ERROR = 4,
    WS_MSG_LOG = 5,
    WS_MSG_CUSTOM = 6,
    WS_MSG_DIALOGUE_RESPONSE = 7,
    WS_MSG_DIALOGUE_TOKEN = 8,
    WS_MSG_EVOLUTION_EVENT = 9,
    WS_MSG_SAFETY_ALERT = 10,
    WS_MSG_ROBOT_STATUS = 11,
    WS_MSG_COGNITION_EVENT = 12,
    WS_MSG_DIAGNOSTIC = 13,
    WS_MSG_MULTIMODAL_DATA = 14,
    WS_MSG_TRAINING_METRICS = 15,
    WS_MSG_GPU_STATUS = 16,
    WS_MSG_MEMORY_STATUS = 17,
    WS_MSG_KNOWLEDGE_STATUS = 18,
    WS_MSG_PREDICTION_RESULT = 19,
    WS_MSG_CONCEPT_EVOLUTION = 20,
    WS_MSG_STATE_ACTIVATION_DATA = 21,  /**< L-9修复: 枚举名与映射字符串"state_activation_data"统一 */
    WS_MSG_WEIGHT_DISTRIBUTION = 22,
    WS_MSG_ACTIVATION_STATS = 23,
    WS_MSG_LNN_STATE = 24,
    WS_MSG_METACOGNITION = 25,
    WS_MSG_COUNT
} WSMessageType;

WSPushServer* ws_push_server_create(int port);

int ws_push_server_start(WSPushServer* server);

void ws_push_server_stop(WSPushServer* server);

void ws_push_server_destroy(WSPushServer* server);

int ws_push_broadcast(WSPushServer* server, WSMessageType type, const char* data);

int ws_push_broadcast_json(WSPushServer* server, const char* json_message);

int ws_push_send_to_client(WSPushServer* server, int client_index, const char* json_message);

int ws_push_get_next_available_client(WSPushServer* server);

int ws_push_get_client_count(const WSPushServer* server);

int ws_push_server_poll(WSPushServer* server, int timeout_ms);

#ifdef __cplusplus
}
#endif

#endif
