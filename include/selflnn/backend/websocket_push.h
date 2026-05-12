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

typedef enum {
    WS_MSG_TRAINING_PROGRESS = 0,
    WS_MSG_SYSTEM_STATUS = 1,
    WS_MSG_KNOWLEDGE_UPDATE = 2,
    WS_MSG_MODEL_OUTPUT = 3,
    WS_MSG_ERROR = 4,
    WS_MSG_LOG = 5,
    WS_MSG_CUSTOM = 6,
    WS_MSG_DIALOGUE_RESPONSE = 7,
    WS_MSG_DIALOGUE_TOKEN = 8
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

#ifdef __cplusplus
}
#endif

#endif
