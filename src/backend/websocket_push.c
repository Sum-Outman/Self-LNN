#include "selflnn/backend/websocket_push.h"
#include "selflnn/core/common.h"
#include "selflnn/core/errors.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/platform.h"
#include "selflnn/core/port_config.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
/* ZSFWS-L007: 当前WebSocket实现不支持per-message-deflate(RFC7692)压缩扩展。
 * 影响: 高负载场景下JSON推送占用额外带宽（未压缩时约3-5x膨胀）。
 * 后续方案: 实现纯C deflate核心或使用应用层msgpack替代JSON。
 * 当前缓解: 高频推送(如系统状态)仅包含最小字段集。 */
#include <time.h>
#include <stdint.h>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
typedef SOCKET ws_socket_t;
#define WS_INVALID INVALID_SOCKET
#define WS_ERROR SOCKET_ERROR
#define WS_CLOSE closesocket
#define WS_ERRNO WSAGetLastError()
#define WS_EWOULDBLOCK WSAEWOULDBLOCK
#define WS_ECONNRESET WSAECONNRESET
#else
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/select.h>
#include <pthread.h>
typedef int ws_socket_t;
#define WS_INVALID (-1)
#define WS_ERROR (-1)
#define WS_CLOSE close
#define WS_ERRNO errno
#define WS_EWOULDBLOCK EWOULDBLOCK
#define WS_ECONNRESET ECONNRESET
#endif

typedef struct {
    ws_socket_t sock;
    int active;
    char recv_buffer[WS_MAX_MESSAGE_SIZE];
    size_t recv_len;
    char send_buffer[WS_SEND_BUFFER_SIZE];
    size_t send_len;
    size_t send_pos;
    uint8_t frame_buffer[WS_MAX_MESSAGE_SIZE + 16];
    long last_active;
    int closing;
} WSClientInternal;

struct WSPushServer {
    int port;
    int running;
    ws_socket_t listen_sock;
    WSClientInternal clients[WS_MAX_CLIENTS];
    int client_count;
    long last_tick;
    void* mutex;
    void* accept_thread;
};

static int ws_global_init(void)
{
#ifdef _WIN32
    WSADATA wsa;
    return WSAStartup(MAKEWORD(2, 2), &wsa) == 0 ? 0 : -1;
#else
    return 0;
#endif
}

static void ws_global_cleanup(void)
{
#ifdef _WIN32
    WSACleanup();
#endif
}

static int ws_set_nonblock(ws_socket_t sock, int nonblock)
{
#ifdef _WIN32
    u_long mode = nonblock ? 1 : 0;
    return ioctlsocket(sock, FIONBIO, &mode) == 0 ? 0 : -1;
#else
    int flags = fcntl(sock, F_GETFL, 0);
    if (flags < 0) return -1;
    return fcntl(sock, F_SETFL, nonblock ? (flags | O_NONBLOCK) : (flags & ~O_NONBLOCK));
#endif
}

static int ws_base64_encode(const unsigned char* in, int in_len, char* out, int out_max)
{
    static const char b64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    int i = 0, j = 0;
    unsigned char a[3];
    while (in_len > 0) {
        int n = in_len > 3 ? 3 : in_len;
        for (int k = 0; k < n; k++) a[k] = in[i++];
        in_len -= n;
        if (j + 4 > out_max) return -1;
        out[j++] = b64[a[0] >> 2];
        out[j++] = b64[((a[0] & 0x03) << 4) | (a[1] >> 4)];
        out[j++] = (n > 1) ? b64[((a[1] & 0x0F) << 2) | (a[2] >> 6)] : '=';
        out[j++] = (n > 2) ? b64[a[2] & 0x3F] : '=';
    }
    out[j] = '\0';
    return j;
}

#include <stdint.h>

static void ws_sha1(const unsigned char* msg, size_t len, unsigned char out[20])
{
    uint32_t h0 = 0x67452301, h1 = 0xEFCDAB89, h2 = 0x98BADCFE, h3 = 0x10325476, h4 = 0xC3D2E1F0;
    size_t new_len = (((len + 8) / 64) + 1) * 64;
    unsigned char* buf = (unsigned char*)safe_calloc(new_len + 64, 1);
    if (!buf) return;
    memcpy(buf, msg, len);
    buf[len] = 0x80;
    uint64_t bits = (uint64_t)len * 8;
    for (int i = 0; i < 8; i++) buf[new_len - 8 + i] = (unsigned char)(bits >> (56 - i * 8));
    for (size_t chunk = 0; chunk < new_len; chunk += 64) {
        uint32_t w[80];
        for (int i = 0; i < 16; i++) {
            w[i] = ((uint32_t)buf[chunk + i * 4] << 24) |
                   ((uint32_t)buf[chunk + i * 4 + 1] << 16) |
                   ((uint32_t)buf[chunk + i * 4 + 2] << 8) |
                   (uint32_t)buf[chunk + i * 4 + 3];
        }
        for (int i = 16; i < 80; i++) {
            uint32_t t = w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16];
            w[i] = (t << 1) | (t >> 31);
        }
        uint32_t a = h0, b = h1, c = h2, d = h3, e = h4;
        for (int i = 0; i < 80; i++) {
            uint32_t f, k;
            if (i < 20) { f = (b & c) | (~b & d); k = 0x5A827999; }
            else if (i < 40) { f = b ^ c ^ d; k = 0x6ED9EBA1; }
            else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8F1BBCDC; }
            else { f = b ^ c ^ d; k = 0xCA62C1D6; }
            uint32_t temp = ((a << 5) | (a >> 27)) + f + e + k + w[i];
            e = d; d = c; c = (b << 30) | (b >> 2); b = a; a = temp;
        }
        h0 += a; h1 += b; h2 += c; h3 += d; h4 += e;
    }
    safe_free((void**)&buf);
    for (int i = 0; i < 4; i++) { out[i] = (unsigned char)(h0 >> (24 - i * 8)); out[4+i] = (unsigned char)(h1 >> (24 - i * 8)); }
    for (int i = 0; i < 4; i++) { out[8+i] = (unsigned char)(h2 >> (24 - i * 8)); out[12+i] = (unsigned char)(h3 >> (24 - i * 8)); }
    for (int i = 0; i < 4; i++) out[16+i] = (unsigned char)(h4 >> (24 - i * 8));
}

static int ws_generate_accept_key(const char* client_key, char* out, int out_max)
{
    const char* magic = "258EAFA5-E914-47DA-95CA-5AB9DC11B85B";
    char combined[256];
    int clen = (int)strlen(client_key);
    int mlen = (int)strlen(magic);
    if (clen + mlen > 255) return -1;
    memcpy(combined, client_key, clen);
    memcpy(combined + clen, magic, mlen);
    unsigned char hash[20];
    ws_sha1((unsigned char*)combined, clen + mlen, hash);
    return ws_base64_encode(hash, 20, out, out_max);
}

static int ws_build_frame(uint8_t opcode, const unsigned char* payload, size_t payload_len,
                          unsigned char* frame, size_t max_size)
{
    size_t header = 2;
    if (payload_len <= 125) {
        frame[1] = (uint8_t)payload_len;
    } else if (payload_len <= 65535) {
        frame[1] = 126; header += 2;
    } else {
        frame[1] = 127; header += 8;
    }
    if (header + payload_len > max_size) return -1;
    frame[0] = 0x80 | (opcode & 0x0F);
    if (payload_len <= 125) {
        frame[1] = (uint8_t)payload_len;
    } else if (payload_len <= 65535) {
        frame[1] = 126;
        frame[2] = (uint8_t)(payload_len >> 8);
        frame[3] = (uint8_t)(payload_len & 0xFF);
    } else {
        frame[1] = 127;
        uint64_t pl = payload_len;
        for (int i = 0; i < 8; i++) frame[2 + i] = (uint8_t)(pl >> (56 - i * 8));
    }
    if (payload_len > 0) memcpy(frame + header, payload, payload_len);
    return (int)(header + payload_len);
}

static int ws_parse_frame(const unsigned char* frame, size_t len,
                          uint8_t* opcode, unsigned char** payload, size_t* payload_len)
{
    if (len < 2) return -1;
    /* K-201: 检查FIN位——当前仅处理非分片帧(0x80) */
    int fin = (frame[0] & 0x80) ? 1 : 0;
    if (!fin) return -1;
    *opcode = frame[0] & 0x0F;
    int masked = (frame[1] & 0x80) ? 1 : 0;
    uint8_t plen = frame[1] & 0x7F;
    size_t header = 2;
    size_t ext_len = 0;
    if (plen == 126) { ext_len = 2; header += 2; }
    else if (plen == 127) { ext_len = 8; header += 8; }
    if (header + (masked ? 4 : 0) > len) return -1;
    if (plen <= 125) *payload_len = plen;
    else if (plen == 126) {
        *payload_len = ((size_t)frame[2] << 8) | (size_t)frame[3];
    } else {
        *payload_len = 0;
        for (int i = 0; i < 8; i++) *payload_len = (*payload_len << 8) | (size_t)frame[2 + i];
    }
    if (header + *payload_len + (masked ? 4 : 0) > len) return -1;
    unsigned char* raw_payload = (unsigned char*)frame + header + (masked ? 4 : 0);
    if (masked) {
        /* B-012修复: 分配独立缓冲区进行XOR解掩码，避免修改原始帧缓冲区。
         * 若直接在raw_payload上原位XOR，多帧处理时后续帧解析会被破坏。
         * 调用方需在帧处理完成后释放*payload指向的内存。 */
        unsigned char* unmasked = (unsigned char*)safe_malloc(*payload_len);
        if (!unmasked) return -1;
        unsigned char mask[4];
        memcpy(mask, frame + header - 4, 4);
        memcpy(unmasked, raw_payload, *payload_len);
        for (size_t i = 0; i < *payload_len; i++) unmasked[i] ^= mask[i % 4];
        *payload = unmasked;
    } else {
        *payload = raw_payload;
    }
    return 0;
}

static int ws_do_handshake(ws_socket_t sock)
{
    char buf[4096];
    int n = (int)recv(sock, buf, sizeof(buf) - 1, 0);
    if (n <= 0) return -1;
    buf[n] = '\0';
    if (strstr(buf, "Upgrade: websocket") == NULL && strstr(buf, "upgrade: websocket") == NULL) return -1;
    const char* key_marker = "Sec-WebSocket-Key:";
    const char* key_start = strstr(buf, key_marker);
    if (!key_start) {
        key_marker = "sec-websocket-key:";
        key_start = strstr(buf, key_marker);
    }
    if (!key_start) return -1;
    key_start += strlen(key_marker);
    while (*key_start == ' ') key_start++;
    char client_key[128];
    int ki = 0;
    while (*key_start && *key_start != '\r' && ki < 127) client_key[ki++] = *key_start++;
    client_key[ki] = '\0';
    char accept_key[64];
    if (ws_generate_accept_key(client_key, accept_key, 64) < 0) return -1;
    char response[512];
    int rlen = snprintf(response, sizeof(response),
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        "Sec-WebSocket-Accept: %s\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "\r\n", accept_key);
    return send(sock, response, rlen, 0) == rlen ? 0 : -1;
}

static int ws_send_frame(ws_socket_t sock, uint8_t opcode, const unsigned char* data, size_t len)
{
    unsigned char frame[WS_MAX_MESSAGE_SIZE + 16];
    if (len + 16 > sizeof(frame)) {
        unsigned char* big = (unsigned char*)safe_malloc(len + 16);
        if (!big) return -1;
        int fsize = ws_build_frame(opcode, data, len, big, len + 16);
        if (fsize < 0) { safe_free((void**)&big); return -1; }
        int ret = (int)send(sock, (const char*)big, fsize, 0);
        safe_free((void**)&big);
        return ret == fsize ? 0 : -1;
    }
    int fsize = ws_build_frame(opcode, data, len, frame, sizeof(frame));
    if (fsize < 0) return -1;
    return send(sock, (const char*)frame, fsize, 0) == fsize ? 0 : -1;
}

static int ws_send_close(ws_socket_t sock, uint16_t code)
{
    unsigned char payload[2];
    payload[0] = (unsigned char)(code >> 8);
    payload[1] = (unsigned char)(code & 0xFF);
    return ws_send_frame(sock, 0x08, payload, 2);
}

static int ws_send_pong(ws_socket_t sock, const unsigned char* data, size_t len)
{
    return ws_send_frame(sock, 0x0A, data, len);
}

static void ws_client_init(WSClientInternal* cli, ws_socket_t sock)
{
    if (!cli) return;
    memset(cli, 0, sizeof(WSClientInternal));
    cli->sock = sock;
    cli->active = 1;
    cli->last_active = (long)time(NULL);
}

static void ws_client_close(WSClientInternal* cli)
{
    if (cli->sock == WS_INVALID) return;
    ws_send_close(cli->sock, 1000);
    WS_CLOSE(cli->sock);
    cli->sock = WS_INVALID;
    cli->active = 0;
    cli->recv_len = 0;
    cli->send_len = 0;
    cli->send_pos = 0;
    cli->closing = 0;
}

/* K-071: WebSocket推送线程安全 —— 用平台互斥锁保护客户端列表操作 */
static void ws_lock(void* mutex)
{
#ifdef _WIN32
    if (mutex) EnterCriticalSection((CRITICAL_SECTION*)mutex);
#else
    if (mutex) pthread_mutex_lock((pthread_mutex_t*)mutex);
#endif
}

static void ws_unlock(void* mutex)
{
#ifdef _WIN32
    if (mutex) LeaveCriticalSection((CRITICAL_SECTION*)mutex);
#else
    if (mutex) pthread_mutex_unlock((pthread_mutex_t*)mutex);
#endif
}

/* WebSocket推送accept线程函数 */
#ifdef _WIN32
static DWORD WINAPI ws_push_accept_thread(LPVOID arg)
#else
static void* ws_push_accept_thread(void* arg)
#endif
{
    WSPushServer* srv = (WSPushServer*)arg;
    if (!srv) return 1;
    struct timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = 100000;

    while (srv->running) {
        fd_set readfds;
        FD_ZERO(&readfds);
        FD_SET(srv->listen_sock, &readfds);
        int sel = select(0, &readfds, NULL, NULL, &tv);
        if (sel <= 0) continue;

        struct sockaddr_in client_addr;
        socklen_t addr_len = sizeof(client_addr);
        ws_socket_t client = accept(srv->listen_sock, (struct sockaddr*)&client_addr, &addr_len);
        if (client == WS_INVALID) continue;

        /* PF-003修复: 互斥锁保护槽位分配，防止accept线程与poll函数竞态 */
        mutex_lock(srv->mutex);
        int slot = -1;
        for (int i = 0; i < WS_MAX_CLIENTS; i++) {
            if (!srv->clients[i].active) { slot = i; break; }
        }
        if (slot < 0) { mutex_unlock(srv->mutex); WS_CLOSE(client); continue; }

        srv->clients[slot].active = 1;
        mutex_unlock(srv->mutex);

        if (ws_do_handshake(client) != 0) { WS_CLOSE(client); continue; }

        ws_set_nonblock(client, 1);
        ws_client_init(&srv->clients[slot], client);
    }
#ifdef _WIN32
    return 0;
#else
    return NULL;
#endif
}

WSPushServer* ws_push_server_create(int port)
{
    if (ws_global_init() != 0) return NULL;
    WSPushServer* srv = (WSPushServer*)safe_malloc(sizeof(WSPushServer));
    if (!srv) return NULL;
    memset(srv, 0, sizeof(WSPushServer));
    srv->port = port > 0 ? port : SELFLNN_WEBSOCKET_PORT;
    srv->listen_sock = WS_INVALID;
    /* PF-003修复: 初始化全局互斥锁以保护accept+槽位分配竞态条件 */
    srv->mutex = mutex_create();
    if (!srv->mutex) {
        safe_free((void**)&srv);
        return NULL;
    }
    srv->last_tick = (long)time(NULL);
    return srv;
}

int ws_push_server_start(WSPushServer* srv)
{
    if (!srv || srv->running) return -1;
    ws_socket_t sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock == WS_INVALID) return -1;
    int opt = 1;
#ifdef _WIN32
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));
#else
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons((unsigned short)srv->port);
    if (bind(sock, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
        WS_CLOSE(sock);
        selflnn_set_last_error(SELFLNN_ERROR_GENERIC, __func__, __FILE__, __LINE__, "绑定WebSocket端口失败");
        return -1;
    }
    if (listen(sock, 16) != 0) {
        WS_CLOSE(sock);
        return -1;
    }
    ws_set_nonblock(sock, 1);
    srv->listen_sock = sock;
    srv->running = 1;
    srv->last_tick = (long)time(NULL);
#ifdef _WIN32
    srv->accept_thread = CreateThread(NULL, 0, ws_push_accept_thread, srv, 0, NULL);
#else
    {
        pthread_t tid;
        if (pthread_create(&tid, NULL, ws_push_accept_thread, srv) == 0) {
            pthread_detach(tid);
            srv->accept_thread = (void*)(uintptr_t)tid;
        }
    }
#endif
    return 0;
}

void ws_push_server_stop(WSPushServer* srv)
{
    if (!srv) return;
    srv->running = 0;
    /* R6-008修复: 等待accept线程退出，防止use-after-free */
    if (srv->accept_thread) {
#ifdef _WIN32
        WaitForSingleObject((HANDLE)srv->accept_thread, 3000);
        CloseHandle((HANDLE)srv->accept_thread);
#else
        pthread_join((pthread_t)(uintptr_t)srv->accept_thread, NULL);
#endif
        srv->accept_thread = NULL;
    }
    for (int i = 0; i < WS_MAX_CLIENTS; i++) {
        if (srv->clients[i].active) ws_client_close(&srv->clients[i]);
    }
    if (srv->listen_sock != WS_INVALID) {
        WS_CLOSE(srv->listen_sock);
        srv->listen_sock = WS_INVALID;
    }
}

void ws_push_server_destroy(WSPushServer* srv)
{
    if (!srv) return;
    ws_push_server_stop(srv);
    /* R6-008修复: 销毁互斥锁防止资源泄漏 */
    if (srv->mutex) { mutex_destroy(srv->mutex); srv->mutex = NULL; }
    ws_global_cleanup();
    safe_free((void**)&srv);
}

int ws_push_broadcast(WSPushServer* srv, WSMessageType type, const char* data)
{
    if (!srv || !srv->running || !data) return -1;
    /* P1-023修复: 使用消息类型名称数组替代硬编码if-else-if链，支持动态扩展 */
    static const char* msg_type_names[] = {
        [WS_MSG_TRAINING_PROGRESS] = "training_progress",
        [WS_MSG_SYSTEM_STATUS] = "system_status",
        [WS_MSG_KNOWLEDGE_UPDATE] = "knowledge_update",
        [WS_MSG_MODEL_OUTPUT] = "model_output",
        [WS_MSG_ERROR] = "error",
        [WS_MSG_LOG] = "log",
        [WS_MSG_DIALOGUE_RESPONSE] = "dialogue_response",
        [WS_MSG_DIALOGUE_TOKEN] = "dialogue_token",
        [WS_MSG_EVOLUTION_EVENT] = "evolution_event",
        [WS_MSG_SAFETY_ALERT] = "safety_alert",
        [WS_MSG_ROBOT_STATUS] = "robot_status",
        [WS_MSG_COGNITION_EVENT] = "cognition_event",
        [WS_MSG_DIAGNOSTIC] = "diagnostic",
        [WS_MSG_MULTIMODAL_DATA] = "multimodal_data",
        [WS_MSG_TRAINING_METRICS] = "training_metrics",
    };
    static const size_t num_msg_types = sizeof(msg_type_names) / sizeof(msg_type_names[0]);
    const char* type_name = (type < num_msg_types && msg_type_names[type]) ? msg_type_names[type] : "custom";
    char json[WS_MAX_MESSAGE_SIZE];
    int jlen = snprintf(json, sizeof(json),
        "{\"type\":\"%s\",\"time\":%ld,\"data\":%s}",
        type_name,
        (long)time(NULL), data);
    if (jlen <= 0 || jlen >= (int)sizeof(json)) return -1;
    int sent = 0;
    for (int i = 0; i < WS_MAX_CLIENTS; i++) {
        if (!srv->clients[i].active) continue;
        ws_socket_t sock = srv->clients[i].sock;
        if (sock == WS_INVALID) continue;
        if (ws_send_frame(sock, 0x01, (unsigned char*)json, jlen) == 0) sent++;
        else ws_client_close(&srv->clients[i]);
    }
    return sent;
}

int ws_push_broadcast_json(WSPushServer* srv, const char* json_message)
{
    if (!srv || !srv->running || !json_message) return -1;
    int jlen = (int)strlen(json_message);
    if (jlen <= 0 || jlen >= WS_MAX_MESSAGE_SIZE) return -1;
    int sent = 0;

    /* ZSFAB-S6修复: 加互斥锁保护客户端列表，防止与poll线程竞态 */
    ws_socket_t active_socks[WS_MAX_CLIENTS];
    int active_indices[WS_MAX_CLIENTS];
    int active_count = 0;

    /* 第一阶段：锁定后快照活跃客户端socket列表 */
    mutex_lock(srv->mutex);
    for (int i = 0; i < WS_MAX_CLIENTS; i++) {
        if (!srv->clients[i].active) continue;
        ws_socket_t sock = srv->clients[i].sock;
        if (sock == WS_INVALID) continue;
        active_socks[active_count] = sock;
        active_indices[active_count] = i;
        active_count++;
    }
    mutex_unlock(srv->mutex);

    /* 第二阶段：无锁发送帧，发送失败时加锁关闭客户端 */
    for (int i = 0; i < active_count; i++) {
        if (ws_send_frame(active_socks[i], 0x01, (unsigned char*)json_message, jlen) == 0) {
            sent++;
        } else {
            mutex_lock(srv->mutex);
            ws_client_close(&srv->clients[active_indices[i]]);
            mutex_unlock(srv->mutex);
        }
    }
    return sent;
}

int ws_push_get_client_count(const WSPushServer* srv)
{
    if (!srv) return 0;
    int count = 0;
    for (int i = 0; i < WS_MAX_CLIENTS; i++) {
        if (srv->clients[i].active) count++;
    }
    return count;
}

int ws_push_send_to_client(WSPushServer* srv, int client_index, const char* json_message)
{
    if (!srv || !srv->running || !json_message || client_index < 0 || client_index >= WS_MAX_CLIENTS) {
        return -1;
    }
    WSClientInternal* cli = &srv->clients[client_index];
    if (!cli->active || cli->sock == WS_INVALID) return -1;
    int jlen = (int)strlen(json_message);
    if (jlen <= 0 || jlen >= WS_MAX_MESSAGE_SIZE) return -1;
    if (ws_send_frame(cli->sock, 0x01, (unsigned char*)json_message, jlen) == 0) return 0;
    ws_client_close(cli);
    return -1;
}

int ws_push_get_next_available_client(WSPushServer* srv)
{
    if (!srv) return -1;
    for (int i = 0; i < WS_MAX_CLIENTS; i++) {
        if (srv->clients[i].active) return i;
    }
    return -1;
}

int ws_push_server_poll(WSPushServer* srv, int timeout_ms)
{
    if (!srv || !srv->running) return -1;
    fd_set rfds, wfds;
    FD_ZERO(&rfds);
    FD_ZERO(&wfds);
    ws_socket_t maxfd = srv->listen_sock;
    FD_SET(srv->listen_sock, &rfds);
    int has_write = 0;
    for (int i = 0; i < WS_MAX_CLIENTS; i++) {
        if (!srv->clients[i].active || srv->clients[i].sock == WS_INVALID) continue;
        FD_SET(srv->clients[i].sock, &rfds);
        if (srv->clients[i].send_len > srv->clients[i].send_pos) {
            FD_SET(srv->clients[i].sock, &wfds);
            has_write = 1;
        }
        if (srv->clients[i].sock > maxfd) maxfd = srv->clients[i].sock;
    }
    struct timeval tv;
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
    int ret = select((int)(maxfd + 1), &rfds, has_write ? &wfds : NULL, NULL, &tv);
    if (ret < 0) return 0;
    if (FD_ISSET(srv->listen_sock, &rfds)) {
        struct sockaddr_in client_addr;
        socklen_t addr_len = sizeof(client_addr);
        ws_socket_t client = accept(srv->listen_sock, (struct sockaddr*)&client_addr, &addr_len);
        if (client != WS_INVALID) {
            ws_set_nonblock(client, 1);
            /* PF-003修复: poll中accept+槽位操作加互斥锁 */
            mutex_lock(srv->mutex);
            int found = -1;
            for (int i = 0; i < WS_MAX_CLIENTS; i++) {
                if (!srv->clients[i].active) { found = i; break; }
            }
            if (found >= 0) {
                srv->clients[found].active = 1;
                mutex_unlock(srv->mutex);
                /* 统一使用ws_client_init初始化客户端槽位 */
                ws_client_init(&srv->clients[found], client);
                if (ws_do_handshake(client) != 0) {
                    ws_client_close(&srv->clients[found]);
                }
            } else {
                mutex_unlock(srv->mutex);
                WS_CLOSE(client);
            }
        }
    }
    for (int i = 0; i < WS_MAX_CLIENTS; i++) {
        WSClientInternal* cli = &srv->clients[i];
        if (!cli->active || cli->sock == WS_INVALID) continue;
        if (FD_ISSET(cli->sock, &rfds)) {
            unsigned char buf[8192];
            int n = (int)recv(cli->sock, (char*)buf, sizeof(buf), 0);
            if (n <= 0) { ws_client_close(cli); continue; }
            size_t remain = n;
            unsigned char* ptr = buf;
            while (remain > 0) {
                /* B-012修复: 从原始帧头计算帧总大小，而非依赖payload指针（因payload可能指向堆内存） */
                size_t header_size = 2;
                uint8_t raw_plen = ptr[1] & 0x7F;
                if (raw_plen == 126) header_size += 2;
                else if (raw_plen == 127) header_size += 8;
                int masked = (ptr[1] & 0x80) ? 1 : 0;
                if (masked) header_size += 4;
                /* 从帧头预计算payload长度以确定帧总大小 */
                size_t raw_payload_len;
                if (raw_plen <= 125) raw_payload_len = raw_plen;
                else if (raw_plen == 126)
                    raw_payload_len = ((size_t)ptr[2] << 8) | (size_t)ptr[3];
                else {
                    raw_payload_len = 0;
                    for (int m = 0; m < 8; m++)
                        raw_payload_len = (raw_payload_len << 8) | (size_t)ptr[2 + m];
                }
                size_t frame_size = header_size + raw_payload_len;
                if (frame_size > remain) break;

                uint8_t opcode;
                unsigned char* payload;
                size_t payload_len;
                if (ws_parse_frame(ptr, remain, &opcode, &payload, &payload_len) != 0) break;
                if (frame_size > remain) break;
                if (opcode == 0x08) {
                    if (masked) safe_free((void**)&payload);
                    ws_client_close(cli);
                    break;
                }
                else if (opcode == 0x09) { ws_send_pong(cli->sock, payload, payload_len); }
                else if (opcode == 0x0A) { }
                /* B-012修复: 若帧是掩码帧，释放ws_parse_frame中分配的独立缓冲区 */
                if (masked) safe_free((void**)&payload);
                remain -= frame_size;
                ptr += frame_size;
            }
            cli->last_active = (long)time(NULL);
        }
        if (FD_ISSET(cli->sock, &wfds) && cli->send_len > cli->send_pos) {
            int n = (int)send(cli->sock, cli->send_buffer + cli->send_pos,
                             (int)(cli->send_len - cli->send_pos), 0);
            if (n > 0) {
                cli->send_pos += n;
                if (cli->send_pos >= cli->send_len) {
                    cli->send_len = 0;
                    cli->send_pos = 0;
                }
            } else {
                ws_client_close(cli);
            }
        }
        long now = (long)time(NULL);
        if (now - cli->last_active > 60) ws_client_close(cli);
    }

    /* WebSocket心跳机制：每30秒向所有客户端发送ping帧 */
    {
        long now = (long)time(NULL);
        if (now - srv->last_tick >= 30) {
            srv->last_tick = now;
            for (int i = 0; i < WS_MAX_CLIENTS; i++) {
                WSClientInternal* cli = &srv->clients[i];
                if (!cli->active || cli->sock == WS_INVALID) continue;
                /* 发送ping帧 */
                unsigned char ping_data[8] = {0};
                memcpy(ping_data, &now, sizeof(now));
                if (ws_send_frame(cli->sock, 0x09, ping_data, 8) != 0) {
                    ws_client_close(cli);
                }
                /* 检查客户端是否超过120秒未响应（心跳超时） */
                if (now - cli->last_active > 120) {
                    ws_client_close(cli);
                }
            }
        }
    }

    return ws_push_get_client_count(srv);
}
