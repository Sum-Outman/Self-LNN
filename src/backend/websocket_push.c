#include "selflnn/backend/websocket_push.h"
#include "selflnn/core/common.h"
#include "selflnn/core/errors.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/platform.h"
#include "selflnn/utils/logging.h"  /* DEEP-005: log_warn宏 */
#include "selflnn/core/port_config.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
/* 当前WebSocket实现不支持per-message-deflate(RFC7692)压缩扩展。
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

/* Linux使用epoll，macOS/BSD使用kqueue替代select */
#ifdef __linux__
#include <sys/epoll.h>
#elif defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__NetBSD__)
#include <sys/event.h>
#endif

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
    struct sockaddr_in addr;  /* P2-17修复: 存储客户端地址用于速率限制 */
    char recv_buffer[WS_MAX_MESSAGE_SIZE];
    size_t recv_len;
    char send_buffer[WS_SEND_BUFFER_SIZE];
    size_t send_len;
    size_t send_pos;
    uint8_t frame_buffer[WS_MAX_MESSAGE_SIZE + 16];
    long last_active;
    int closing;
    /* L-006修复: 分片消息支持 —— 累积分片帧到完整消息 */
    uint8_t fragment_buffer[WS_MAX_MESSAGE_SIZE]; /* 分片累积缓冲区 */
    size_t fragment_len;                          /* 已累积的字节数 */
    uint8_t fragment_opcode;                      /* 首帧的opcode（文本或二进制） */
    int fragment_active;                          /* 1=正在累积分片 */
} WSClientInternal;

struct WSPushServer {
    int port;
    /* D-4修复: running改为volatile int，确保accept线程在每次循环迭代中
 * 都从内存重新读取running值，而非使用寄存器缓存。在弱内存序CPU上
 * 配合ws_push_server_stop中的内存屏障，保证线程安全退出。 */
    volatile int running;
    ws_socket_t listen_sock;
    WSClientInternal clients[WS_MAX_CLIENTS];
    int client_count;
    long last_tick;
    void* mutex;
    void* accept_thread;
    /* C-002修复: 客户端消息回调钩子 */
    WSClientMessageHandler msg_handler;
    void* msg_handler_user_data;
/* epoll/kqueue文件描述符（Linux/macOS/BSD高性能I/O多路复用） */
#ifdef __linux__
    int epoll_fd;
#elif defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__NetBSD__)
    int kqueue_fd;
#endif
};

static int ws_global_init(void)
{
#ifdef _WIN32
    /* P3-004修复: 使用原子标志防止多次WSAStartup导致引用计数不匹配 */
    static volatile LONG ws_init_done = 0;
    if (InterlockedCompareExchange(&ws_init_done, 1, 0) != 0) return 0;
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
        /* v9.11修复: 清零a数组防止残留值污染最后不完整组的编码 */
        memset(a, 0, sizeof(a));
        for (int k = 0; k < n; k++) a[k] = in[i++];
        in_len -= n;
        if (j + 5 > out_max) return -1;  /* H-008修复: 边界检查 */
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
    /* P3-002修复: 减少冗余分配64字节；添加大len防御 */
    if (len > 1024 * 1024) {
        /* 超过1MB的WebSocket数据不应进行SHA-1哈希，清零输出并记录警告 */
        memset(out, 0, 20);
        log_warn("[WebSocket] SHA-1输入数据过大(%zu字节)，拒绝处理", len);
        return;
    }
    size_t new_len = (((len + 8) / 64) + 1) * 64;
    unsigned char* buf = (unsigned char*)safe_calloc(new_len, 1);
    if (!buf) {
        /* P-FIX-017: SHA-1分配失败时清零输出，防止调用方使用未初始化哈希值 */
        memset(out, 0, 20);
        return;
    }
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

/* P0-004修复: ws_parse_frame_with_fin —— 正确处理fin_flag参数的分片帧解析
 * 原代码存在两处错误:
 * 1. 函数名误写为ws_parse_frame导致与包装器重复定义
 * 2. 函数体内使用未声明的fin_flag参数
 * 修复: 重命名为ws_parse_frame_with_fin并添加int* fin_flag参数 */
static int ws_parse_frame_with_fin(const unsigned char* frame, size_t len,
                          uint8_t* opcode, unsigned char** payload, size_t* payload_len,
                          int* fin_flag)
{
    if (len < 2) return -1;
    /* L-006修复: 分片消息支持 —— 不再拒绝FIN=0的帧
     * FIN=0 + opcode=0x01/0x02 → 首帧，记录opcode，开始累积分片
     * FIN=0 + opcode=0x00 → 延续帧，追加到 fragment_buffer
     * FIN=1 + opcode=0x00 → 尾帧，追加并完成消息组装
     * FIN=1 + opcode=0x01/0x02 → 完整非分片帧（原有逻辑） */
    int fin = (frame[0] & 0x80) ? 1 : 0;
    *fin_flag = fin;
    *opcode = frame[0] & 0x0F;
    int masked = (frame[1] & 0x80) ? 1 : 0;
    uint8_t plen = frame[1] & 0x7F;
    size_t header = 2;
    if (plen == 126) { header += 2; }
    else if (plen == 127) { header += 8; }
    if (header + (masked ? 4 : 0) > len) return -1;
    if (plen <= 125) *payload_len = plen;
    else if (plen == 126) {
        *payload_len = ((size_t)frame[2] << 8) | (size_t)frame[3];
    } else {
        /* P1修复: 使用uint64_t中间变量解析64位payload长度，
         * 防止32位平台size_t(4字节)移位溢出导致截断。
         * 解析后检查是否超过WS_MAX_MESSAGE_SIZE，拒绝超大帧防止内存耗尽攻击 */
        uint64_t len64 = 0;
        for (int i = 0; i < 8; i++) len64 = (len64 << 8) | (uint64_t)frame[2 + i];
        if (len64 > WS_MAX_MESSAGE_SIZE) return -1;
        *payload_len = (size_t)len64;
    }
    if (header + *payload_len + (masked ? 4 : 0) > len) return -1;
    unsigned char* raw_payload = (unsigned char*)frame + header + (masked ? 4 : 0);
    if (masked) {
        unsigned char* unmasked = (unsigned char*)safe_malloc(*payload_len);
        if (!unmasked) return -1;
        unsigned char mask[4];
        /* P0修复: 掩码密钥位于header之后，而非header-4位置。
         * header变量(2/4/10)仅计基础帧头大小，不含掩码4字节。
         * 原代码 frame+header-4 当header=2时读取frame-2，导致缓冲区下溢。 */
        memcpy(mask, frame + header, 4);
        memcpy(unmasked, raw_payload, *payload_len);
        for (size_t i = 0; i < *payload_len; i++) unmasked[i] ^= mask[i % 4];
        *payload = unmasked;
    } else {
        *payload = raw_payload;
    }
    return 0;
}

/* L-006修复: 向后兼容包装器 —— ws_parse_frame_with_fin 的旧接口
 * P0-004修复: 由于上面的函数已正确命名为ws_parse_frame_with_fin,
 * 此包装器现在能正确调用目标函数 */
static int ws_parse_frame(const unsigned char* frame, size_t len,
                          uint8_t* opcode, unsigned char** payload, size_t* payload_len) {
    int fin_flag;
    return ws_parse_frame_with_fin(frame, len, opcode, payload, payload_len, &fin_flag);
}

/* L-006修复: 将payload追加到客户端的分片缓冲区 */
static int ws_fragment_append(WSClientInternal* cli, const unsigned char* data, size_t len) {
    if (cli->fragment_len + len > WS_MAX_MESSAGE_SIZE) {
        cli->fragment_len = 0; cli->fragment_active = 0; return -1;
    }
    memcpy(cli->fragment_buffer + cli->fragment_len, data, len);
    cli->fragment_len += len;
    return 0;
}

static int ws_do_handshake(ws_socket_t sock)
{
    char buf[4096];
    int total = 0;
    int n;
    /* P2-16修复: 循环读取直到完整HTTP头，处理TCP分片 */
    while (total < (int)sizeof(buf) - 1) {
        n = (int)recv(sock, buf + total, (int)sizeof(buf) - 1 - total, 0);
        if (n <= 0) {
            log_warn("[WebSocket握手] recv失败: n=%d, errno=%d, total=%d", n, (int)WS_ERRNO, total);
            return -1;
        }
        total += n;
        buf[total] = '\0';
        if (strstr(buf, "\r\n\r\n")) break; /* HTTP头结束 */
    }
    log_info("[WebSocket握手] 收到HTTP请求(%d bytes): %.60s", total, buf);

    /* C-003修复: 提取URI路径并校验，仅允许合法WebSocket端点 */
    {
        const char* req_end = strstr(buf, "\r\n");
        if (!req_end) { log_warn("[WebSocket握手] 未找到请求行结束符"); return -1; }
        const char* space1 = strchr(buf, ' ');
        if (!space1) { log_warn("[WebSocket握手] 未找到HTTP方法后的空格"); return -1; }
        const char* uri_start = space1 + 1;
        const char* space2 = strchr(uri_start, ' ');
        if (!space2) { log_warn("[WebSocket握手] 未找到URI后的空格"); return -1; }
        size_t uri_len = (size_t)(space2 - uri_start);
        /* 允许的WebSocket端点: "/ws" */
        if (uri_len != 3 || memcmp(uri_start, "/ws", 3) != 0) {
            log_warn("[WebSocket握手] URI不匹配: '%.*s' (len=%zu)", (int)uri_len, uri_start, uri_len);
            const char* not_found = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n";
            send(sock, not_found, (int)strlen(not_found), 0);
            return -1;
        }
    }

    if (strstr(buf, "Upgrade: websocket") == NULL && strstr(buf, "upgrade: websocket") == NULL) {
        log_warn("[WebSocket握手] 缺少Upgrade: websocket头");
        return -1;
    }
    const char* key_marker = "Sec-WebSocket-Key:";
    const char* key_start = strstr(buf, key_marker);
    if (!key_start) {
        key_marker = "sec-websocket-key:";
        key_start = strstr(buf, key_marker);
    }
    if (!key_start) { log_warn("[WebSocket握手] 缺少Sec-WebSocket-Key头"); return -1; }
    key_start += strlen(key_marker);
    while (*key_start == ' ') key_start++;
    char client_key[128];
    int ki = 0;
    while (*key_start && *key_start != '\r' && ki < 127) client_key[ki++] = *key_start++;
    client_key[ki] = '\0';
    char accept_key[64];
    if (ws_generate_accept_key(client_key, accept_key, 64) < 0) {
        log_warn("[WebSocket握手] 生成Accept-Key失败");
        return -1;
    }
    char response[512];
    int rlen = snprintf(response, sizeof(response),
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        "Sec-WebSocket-Accept: %s\r\n"
        "Access-Control-Allow-Origin: http://localhost:8080\r\n"  /* P3-10修复: 限制CORS来源，移除通配符 */
        "\r\n", accept_key);
    int send_ret = (int)send(sock, response, rlen, 0);
    if (send_ret != rlen) {
        log_warn("[WebSocket握手] send失败: ret=%d, rlen=%d, errno=%d", send_ret, rlen, (int)WS_ERRNO);
        return -1;
    }
    log_info("[WebSocket握手] 成功! Accept-Key=%s", accept_key);
    return 0;
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

static void ws_client_init(WSClientInternal* cli, ws_socket_t sock, struct sockaddr_in addr)
{
    if (!cli) return;
    memset(cli, 0, sizeof(WSClientInternal));
    cli->sock = sock;
    cli->active = 1;
    cli->addr = addr;  /* P2-17修复: 存储客户端地址 */
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
    /* L-006修复: 关闭时清除分片累积状态 */
    cli->fragment_len = 0;
    cli->fragment_active = 0;
    cli->fragment_opcode = 0;
}

/* R5-⑦修复: 实际锁机制使用platform.c的mutex_lock/mutex_unlock */

/* WebSocket推送accept线程函数 */
#ifdef _WIN32
static DWORD WINAPI ws_push_accept_thread(LPVOID arg)
#else
static void* ws_push_accept_thread(void* arg)
#endif
{
    WSPushServer* srv = (WSPushServer*)arg;
    if (!srv) return 1;

#ifdef __linux__
/* Linux epoll等待新连接 */
    struct epoll_event events[1];
    while (srv->running) {
        int nfds = epoll_wait(srv->epoll_fd, events, 1, 100);  /* C-008修复: 去除fd=0回退，epoll_fd无效时不监听stdin */
        if (nfds <= 0) { time_sleep_ms(100); continue; }
#elif defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__NetBSD__)
/* macOS/BSD kqueue等待新连接 */
    struct kevent events[1];
    struct timespec ts = {0, 100000000}; /* 100ms */
    while (srv->running) {
        int nfds = kevent(srv->kqueue_fd, NULL, 0, events, 1, &ts);  /* C-008修复: 去除fd=0回退，kqueue_fd无效时不监听stdin */
        if (nfds <= 0) { time_sleep_ms(100); continue; }
#else
    struct timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = 100000;

    while (srv->running) {
        fd_set readfds;
        FD_ZERO(&readfds);
        FD_SET(srv->listen_sock, &readfds);
        int sel = select(0, &readfds, NULL, NULL, &tv);
        if (sel <= 0) continue;
#endif

        struct sockaddr_in client_addr;
        socklen_t addr_len = sizeof(client_addr);
        ws_socket_t client = accept(srv->listen_sock, (struct sockaddr*)&client_addr, &addr_len);
        if (client == WS_INVALID) continue;

        /* P2-17修复: 每IP连接数限制，防止DoS */
        int ip_count = 0;
        for (int i = 0; i < WS_MAX_CLIENTS; i++) {
            if (srv->clients[i].active && srv->clients[i].addr.sin_addr.s_addr == client_addr.sin_addr.s_addr) {
                ip_count++;
            }
        }
        if (ip_count >= 10) { /* 每IP最多10个连接 */
            WS_CLOSE(client);
            continue;
        }

        /* PF-003修复: 互斥锁保护槽位分配，防止accept线程与poll函数竞态 */
        mutex_lock(srv->mutex);
        int slot = -1;
        for (int i = 0; i < WS_MAX_CLIENTS; i++) {
            if (!srv->clients[i].active) { slot = i; break; }
        }
        if (slot < 0) { mutex_unlock(srv->mutex); WS_CLOSE(client); continue; }

        /* Z-R3-P02修复: 将ws_client_init移到锁内执行，消除竞态窗口。
         * 原代码在mutex_unlock之后才调用ws_client_init，导致另一个线程
         * (如ws_push_broadcast_json)在active=1但sock未设置时访问，
         * 可能使用未初始化的socket导致崩溃或数据损坏。 */
        ws_client_init(&srv->clients[slot], client, client_addr);
        srv->clients[slot].active = 1;
        mutex_unlock(srv->mutex);

        /* v9.11修复: 握手前强制设为阻塞模式。
         * 关键: accept()继承监听socket的非阻塞属性(WSAEventSelect/ioctlsocket)，
         * 必须显式设为阻塞模式才能完成WebSocket握手。 */
        ws_set_nonblock(client, 0);
        if (ws_do_handshake(client) != 0) {
            WS_CLOSE(client);
            ws_client_close(&srv->clients[slot]);  /* C-007修复: 握手失败重置槽位，防止僵尸槽位泄漏 */
            continue;
        }

        ws_set_nonblock(client, 1);
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
/* 初始化epoll/kqueue FD */
#ifdef __linux__
    srv->epoll_fd = -1;
#elif defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__NetBSD__)
    srv->kqueue_fd = -1;
#endif
    return srv;
}

/* C-002修复: 注册客户端消息回调处理器 */
void ws_push_set_message_handler(WSPushServer* server, WSClientMessageHandler handler, void* user_data)
{
    if (!server) return;
    server->msg_handler = handler;
    server->msg_handler_user_data = user_data;
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
/* Linux需要SO_REUSEPORT支持独立端口监听 */
    setsockopt(sock, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt));
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

/* 创建epoll/kqueue I/O多路复用FD */
#ifdef __linux__
    srv->epoll_fd = epoll_create1(0);
    if (srv->epoll_fd >= 0) {
        struct epoll_event ev;
        ev.events = EPOLLIN;
        ev.data.fd = sock;
        epoll_ctl(srv->epoll_fd, EPOLL_CTL_ADD, sock, &ev);
    }
#elif defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__NetBSD__)
    srv->kqueue_fd = kqueue();
    if (srv->kqueue_fd >= 0) {
        struct kevent ev;
        EV_SET(&ev, sock, EVFILT_READ, EV_ADD, 0, 0, NULL);
        kevent(srv->kqueue_fd, &ev, 1, NULL, 0, NULL);
    }
#endif

    srv->running = 1;
    srv->last_tick = (long)time(NULL);
#ifdef _WIN32
    srv->accept_thread = CreateThread(NULL, 0, ws_push_accept_thread, srv, 0, NULL);
#else
    {
        pthread_t tid;
        if (pthread_create(&tid, NULL, ws_push_accept_thread, srv) == 0) {
            /* Z-R3-P01修复: 移除pthread_detach，因为ws_push_server_stop中调用了pthread_join。
             * POSIX规范明确禁止对已分离线程调用pthread_join，导致未定义行为(EINVAL或静默崩溃)。 */
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
/* 清理epoll/kqueue FD */
#ifdef __linux__
    if (srv->epoll_fd >= 0) {
        close(srv->epoll_fd);
        srv->epoll_fd = -1;
    }
#elif defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__NetBSD__)
    if (srv->kqueue_fd >= 0) {
        close(srv->kqueue_fd);
        srv->kqueue_fd = -1;
    }
#endif
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
        [WS_MSG_CUSTOM] = "custom",
        [WS_MSG_DIALOGUE_RESPONSE] = "dialogue_response",
        [WS_MSG_DIALOGUE_TOKEN] = "dialogue_token",
        [WS_MSG_EVOLUTION_EVENT] = "evolution_event",
        [WS_MSG_SAFETY_ALERT] = "safety_alert",
        [WS_MSG_ROBOT_STATUS] = "robot_status",
        [WS_MSG_COGNITION_EVENT] = "cognition_event",
        [WS_MSG_DIAGNOSTIC] = "diagnostic",
        [WS_MSG_MULTIMODAL_DATA] = "multimodal_data",
        [WS_MSG_TRAINING_METRICS] = "training_metrics",
        [WS_MSG_GPU_STATUS] = "gpu_status",
        [WS_MSG_MEMORY_STATUS] = "memory_status",
        [WS_MSG_KNOWLEDGE_STATUS] = "knowledge_status",
        [WS_MSG_PREDICTION_RESULT] = "prediction_result",
        [WS_MSG_CONCEPT_EVOLUTION] = "concept_evolution",
        [WS_MSG_STATE_ACTIVATION_DATA] = "state_activation_data",
        [WS_MSG_WEIGHT_DISTRIBUTION] = "weight_distribution",
        [WS_MSG_ACTIVATION_STATS] = "activation_stats",
        [WS_MSG_LNN_STATE] = "lnn_state",
        [WS_MSG_METACOGNITION] = "metacognition_status",
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
    /* P1修复: 仿照ws_push_broadcast_json()的锁内快照、锁外发送模式，
     * 避免在mutex内进行网络IO导致慢客户端阻塞整个服务器。 */
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

    /* 第二阶段：锁外发送数据，避免慢客户端阻塞mutex */
    for (int i = 0; i < active_count; i++) {
        if (ws_send_frame(active_socks[i], 0x01, (unsigned char*)json, jlen) == 0) {
            sent++;
        } else {
            /* 发送失败：重新加锁关闭该客户端。
             * TOCTOU竞态防护：重新加锁后验证当前槽位的socket是否与快照一致。
             * 在锁外发送期间，原始客户端可能已断开且槽位被新客户端重用，
             * 若不验证socket一致性，将误关闭新客户端。 */
            mutex_lock(srv->mutex);
            if (active_indices[i] < WS_MAX_CLIENTS &&
                srv->clients[active_indices[i]].active &&
                srv->clients[active_indices[i]].sock == active_socks[i]) {
                /* socket一致：确认是同一客户端，安全关闭 */
                ws_client_close(&srv->clients[active_indices[i]]);
            }
            /* socket不一致：槽位已被新客户端重用，跳过关闭操作 */
            mutex_unlock(srv->mutex);
        }
    }
    return sent;
}

int ws_push_broadcast_json(WSPushServer* srv, const char* json_message)
{
    if (!srv || !srv->running || !json_message) return -1;
    int jlen = (int)strlen(json_message);
    if (jlen <= 0 || jlen >= WS_MAX_MESSAGE_SIZE) return -1;
    int sent = 0;

/* 加互斥锁保护客户端列表，防止与poll线程竞态 */
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
            /* 发送失败：重新加锁关闭该客户端。
             * TOCTOU竞态防护：重新加锁后验证当前槽位的socket是否与快照一致。
             * 在锁外发送期间，原始客户端可能已断开且槽位被新客户端重用，
             * 若不验证socket一致性，将误关闭新客户端。 */
            mutex_lock(srv->mutex);
            if (active_indices[i] < WS_MAX_CLIENTS &&
                srv->clients[active_indices[i]].active &&
                srv->clients[active_indices[i]].sock == active_socks[i]) {
                /* socket一致：确认是同一客户端，安全关闭 */
                ws_client_close(&srv->clients[active_indices[i]]);
            }
            /* socket不一致：槽位已被新客户端重用，跳过关闭操作 */
            mutex_unlock(srv->mutex);
        }
    }
    return sent;
}

int ws_push_get_client_count(const WSPushServer* srv)
{
    if (!srv) return 0;
    int count = 0;
    /* H-MED-002: 遍历客户端数组前加锁，防止并发修改 */
    mutex_lock(((WSPushServer*)srv)->mutex);
    for (int i = 0; i < WS_MAX_CLIENTS; i++) {
        if (srv->clients[i].active) count++;
    }
    mutex_unlock(((WSPushServer*)srv)->mutex);
    return count;
}

int ws_push_send_to_client(WSPushServer* srv, int client_index, const char* json_message)
{
    if (!srv || !srv->running || !json_message || client_index < 0 || client_index >= WS_MAX_CLIENTS) {
        return -1;
    }
    WSClientInternal* cli = &srv->clients[client_index];
    /* H-MED-002: 访问cli->active和cli->sock前加锁，防止并发关闭/修改 */
    mutex_lock(srv->mutex);
    if (!cli->active || cli->sock == WS_INVALID) {
        mutex_unlock(srv->mutex);
        return -1;
    }
    int jlen = (int)strlen(json_message);
    if (jlen <= 0 || jlen >= WS_MAX_MESSAGE_SIZE) {
        mutex_unlock(srv->mutex);
        return -1;
    }
    if (ws_send_frame(cli->sock, 0x01, (unsigned char*)json_message, jlen) == 0) {
        mutex_unlock(srv->mutex);
        return 0;
    }
    ws_client_close(cli);
    mutex_unlock(srv->mutex);
    return -1;
}

int ws_push_get_next_available_client(WSPushServer* srv)
{
    if (!srv) return -1;
    /* P3-11修复: 添加互斥锁保护clients数组遍历 */
    mutex_lock(srv->mutex);
    for (int i = 0; i < WS_MAX_CLIENTS; i++) {
        if (srv->clients[i].active) {
            mutex_unlock(srv->mutex);
            return i;
        }
    }
    mutex_unlock(srv->mutex);
    return -1;
}

/* ZSFJJJ-C002修复: 实现 ws_client_process() — WebSocket客户端帧处理
 * C-002修复: 增加server和client_slot参数以支持消息回调分发
 * P0修复: 添加帧头越界读取防护，在读取ptr[1]/ptr[2]/ptr[3]前检查remain
 * P1修复: 使用cli->recv_buffer实现跨TCP段帧重组，不完整帧保留到下次recv */
static void ws_client_process(WSPushServer* server, int client_slot, WSClientInternal* cli) {
    if (!cli || !cli->active || cli->sock == WS_INVALID) return;

    /* P1修复: 使用持久接收缓冲区实现跨TCP段帧重组。
     * 原代码将recv数据读入局部缓冲区直接处理，当WebSocket帧跨越
     * 多个TCP段时不完整帧的数据被丢弃，导致帧丢失和解析错误。
     * 现使用cli->recv_buffer累积数据，完整帧处理后才移除。 */
    unsigned char tmp[8192];
    int n = (int)recv(cli->sock, (char*)tmp, sizeof(tmp), 0);
    if (n <= 0) {
        ws_client_close(cli);
        return;
    }

    /* 将新接收的数据追加到持久接收缓冲区 */
    if (cli->recv_len + (size_t)n > WS_MAX_MESSAGE_SIZE) {
        /* 接收缓冲区溢出，可能是客户端发送了超大帧或恶意数据 */
        cli->recv_len = 0;
        ws_client_close(cli);
        return;
    }
    memcpy(cli->recv_buffer + cli->recv_len, tmp, (size_t)n);
    cli->recv_len += (size_t)n;

    /* 从接收缓冲区解析帧 */
    size_t remain = cli->recv_len;
    size_t consumed = 0;

    while (remain > 0) {
        unsigned char* ptr = (unsigned char*)cli->recv_buffer + consumed;

        /* P0修复: 在读取帧头字段前检查是否有足够字节，防止越界读取。
         * 原代码在检查frame_size > remain之前就读取ptr[1], ptr[2], ptr[3]，
         * 当remain只有1字节时会发生越界读取。 */
        if (remain < 2) break;  /* 至少需要2字节读取opcode和基本payload_len */
        size_t header_size = 2;
        uint8_t raw_plen = ptr[1] & 0x7F;
        if (raw_plen == 126) {
            if (remain < 4) break;  /* 需要额外2字节读取16位extended payload length */
            header_size += 2;
        } else if (raw_plen == 127) {
            if (remain < 10) break;  /* 需要额外8字节读取64位extended payload length */
            header_size += 8;
        }
        int masked = (ptr[1] & 0x80) ? 1 : 0;
        if (masked) header_size += 4;

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
        if (frame_size > remain) break;  /* 帧不完整，等待更多数据到达 */

        uint8_t opcode;
        unsigned char* payload;
        size_t payload_len;
        int fin_flag;
        /* L-006修复: 使用ws_parse_frame_with_fin获取FIN标志，支持分片消息 */
        if (ws_parse_frame_with_fin(ptr, remain, &opcode, &payload, &payload_len, &fin_flag) != 0) break;

        /* 控制帧(opcode>=0x08)不允许分片，直接处理 */
        if (opcode == 0x08) {
            if (masked) safe_free((void**)&payload);
            ws_client_close(cli);
            return;
        } else if (opcode == 0x09) {
            ws_send_pong(cli->sock, payload, payload_len);
        }
        /* opcode 0x0A (pong) 无需处理 */
        /* 数据帧(0x01/0x02)和延续帧(0x00)的分片逻辑 */
        else if (opcode == 0x01 || opcode == 0x02) {
            if (!fin_flag) {
                /* 分片首帧：记录opcode，开始累积 */
                if (!cli->fragment_active) {
                    cli->fragment_active = 1;
                    cli->fragment_opcode = opcode;
                    cli->fragment_len = 0;
                }
                if (ws_fragment_append(cli, payload, payload_len) == 0) {
                    /* 成功累积到缓冲区 */
                } else {
                    /* 分片溢出，重置状态 */
                    cli->fragment_len = 0; cli->fragment_active = 0;
                }
            } else {
                /* FIN=1 + opcode=0x01/0x02 = 完整非分片帧，或分片首帧也是最后一帧 */
                /* C-002修复: 通过回调分发完整消息到上层处理 */
                if (server && server->msg_handler) {
                    server->msg_handler(client_slot, payload, payload_len, opcode,
                                       server->msg_handler_user_data);
                }
            }
        } else if (opcode == 0x00) {
            /* 延续帧 */
            if (cli->fragment_active) {
                if (ws_fragment_append(cli, payload, payload_len) == 0) {
                    if (fin_flag) {
                        /* 尾帧：分片消息组装完成 */
                        /* C-002修复: 通过回调分发组装后的完整消息 */
                        if (server && server->msg_handler) {
                            server->msg_handler(client_slot,
                                               cli->fragment_buffer, cli->fragment_len,
                                               cli->fragment_opcode,
                                               server->msg_handler_user_data);
                        }
                        cli->fragment_active = 0;
                        cli->fragment_len = 0;
                    }
                    /* 中间延续帧：已累积，继续等待后续帧 */
                } else {
                    /* 溢出 */
                    cli->fragment_len = 0; cli->fragment_active = 0;
                }
            }
            /* 未激活分片状态时的孤立延续帧：忽略 */
        }

        if (masked) safe_free((void**)&payload);
        consumed += frame_size;
        remain -= frame_size;
    }

    /* 将未消费的数据移到缓冲区开头，供下次recv继续累积 */
    if (consumed > 0 && remain > 0) {
        memmove(cli->recv_buffer, cli->recv_buffer + consumed, remain);
    }
    cli->recv_len = remain;
    cli->last_active = (long)time(NULL);
}

int ws_push_server_poll(WSPushServer* srv, int timeout_ms)
{
    if (!srv || !srv->running) return -1;

#if defined(__linux__) || defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__NetBSD__)
/* Linux使用epoll，macOS/BSD使用kqueue替代select */
#ifdef __linux__
    if (srv->epoll_fd >= 0) {
        struct epoll_event evs[WS_MAX_CLIENTS + 1];
        int nfds = epoll_wait(srv->epoll_fd, evs, WS_MAX_CLIENTS + 1, timeout_ms);
        if (nfds < 0) return 0;
        for (int i = 0; i < nfds; i++) {
            int fd = evs[i].data.fd;
            if (fd == srv->listen_sock) {
                struct sockaddr_in client_addr;
                socklen_t addr_len = sizeof(client_addr);
                ws_socket_t client = accept(srv->listen_sock, (struct sockaddr*)&client_addr, &addr_len);
                if (client != WS_INVALID) {
                    /* v9.11修复: 先握手再设置非阻塞，防止recv()在非阻塞模式下立即返回失败。
                     * 关键: accept()继承监听socket的非阻塞属性，必须显式设为阻塞模式完成握手。 */
                    mutex_lock(srv->mutex);
                    int found = -1;
                    for (int j = 0; j < WS_MAX_CLIENTS; j++) {
                        if (!srv->clients[j].active) { found = j; break; }
                    }
                    if (found >= 0) {
                        ws_client_init(&srv->clients[found], client, client_addr);
                        srv->clients[found].active = 1;
                        mutex_unlock(srv->mutex);
                        /* 握手前强制设为阻塞模式 */
                        ws_set_nonblock(client, 0);
                        if (ws_do_handshake(client) != 0) {
                            ws_client_close(&srv->clients[found]);
                        } else {
                            /* 握手成功后再设置非阻塞并注册epoll */
                            ws_set_nonblock(client, 1);
                            struct epoll_event ev;
                            ev.events = EPOLLIN | EPOLLRDHUP;
                            ev.data.fd = client;
                            epoll_ctl(srv->epoll_fd, EPOLL_CTL_ADD, client, &ev);
                        }
                    } else {
                        mutex_unlock(srv->mutex);
                        WS_CLOSE(client);
                    }
                }
            } else {
                /* 客户端FD事件处理 */
                for (int j = 0; j < WS_MAX_CLIENTS; j++) {
                    if (srv->clients[j].active && srv->clients[j].sock == fd) {
                        int sock_before = srv->clients[j].sock;
                        ws_client_process(srv, j, &srv->clients[j]);
                        /* P0-8修复：客户端被关闭后从epoll中删除FD，防止FD重用导致数据错乱 */
                        if (!srv->clients[j].active || srv->clients[j].sock == WS_INVALID) {
                            epoll_ctl(srv->epoll_fd, EPOLL_CTL_DEL, sock_before, NULL);
                        }
                        break;
                    }
                }
            }
        }
        return nfds;
    }
#else
    /* kqueue路径 */
    if (srv->kqueue_fd >= 0) {
        struct kevent evs[WS_MAX_CLIENTS + 1];
        struct timespec ts = {timeout_ms / 1000, (timeout_ms % 1000) * 1000000L};
        int nfds = kevent(srv->kqueue_fd, NULL, 0, evs, WS_MAX_CLIENTS + 1, &ts);
        if (nfds < 0) return 0;
        for (int i = 0; i < nfds; i++) {
            int fd = (int)(intptr_t)evs[i].udata;
            if (fd == srv->listen_sock) {
                struct sockaddr_in client_addr;
                socklen_t addr_len = sizeof(client_addr);
                ws_socket_t client = accept(srv->listen_sock, (struct sockaddr*)&client_addr, &addr_len);
                if (client != WS_INVALID) {
                    /* v9.11修复: 先握手再设置非阻塞。
                     * 关键: accept()继承监听socket的非阻塞属性，必须显式设为阻塞模式完成握手。 */
                    mutex_lock(srv->mutex);
                    int found = -1;
                    for (int j = 0; j < WS_MAX_CLIENTS; j++) {
                        if (!srv->clients[j].active) { found = j; break; }
                    }
                    if (found >= 0) {
                        ws_client_init(&srv->clients[found], client, client_addr);
                        srv->clients[found].active = 1;
                        mutex_unlock(srv->mutex);
                        /* 握手前强制设为阻塞模式 */
                        ws_set_nonblock(client, 0);
                        if (ws_do_handshake(client) != 0) {
                            ws_client_close(&srv->clients[found]);
                        } else {
                            /* 握手成功后再设置非阻塞并注册kqueue */
                            ws_set_nonblock(client, 1);
                            struct kevent ev;
                            EV_SET(&ev, client, EVFILT_READ, EV_ADD, 0, 0, (void*)(intptr_t)client);
                            kevent(srv->kqueue_fd, &ev, 1, NULL, 0, NULL);
                        }
                    } else {
                        mutex_unlock(srv->mutex);
                        WS_CLOSE(client);
                    }
                }
            } else {
                for (int j = 0; j < WS_MAX_CLIENTS; j++) {
                    if (srv->clients[j].active && srv->clients[j].sock == fd) {
                        int sock_before = srv->clients[j].sock;
                        ws_client_process(srv, j, &srv->clients[j]);
                        /* C-002修复：客户端被关闭后从kqueue中删除FD，防止FD重用导致数据错乱 */
                        if (!srv->clients[j].active || srv->clients[j].sock == WS_INVALID) {
                            struct kevent ev;
                            EV_SET(&ev, sock_before, EVFILT_READ, EV_DELETE, 0, 0, NULL);
                            kevent(srv->kqueue_fd, &ev, 1, NULL, 0, NULL);
                        }
                        break;
                    }
                }
            }
        }
        return nfds;
    }
#endif
#endif
    /* 回退到select（Windows或不支持epoll/kqueue的场景） */
    fd_set rfds, wfds;
    FD_ZERO(&rfds);
    FD_ZERO(&wfds);
    ws_socket_t maxfd = srv->listen_sock;
    FD_SET(srv->listen_sock, &rfds);
    int has_write = 0;
    for (int i = 0; i < WS_MAX_CLIENTS; i++) {
        if (!srv->clients[i].active || srv->clients[i].sock == WS_INVALID) continue;
        /* P1-15修复: FD_SETSIZE溢出防护，防止栈损坏 */
        if ((int)srv->clients[i].sock < 0 || (int)srv->clients[i].sock >= FD_SETSIZE) {
            /* 客户端socket超出FD_SETSIZE范围，跳过避免栈溢出 */
            continue;
        }
        FD_SET(srv->clients[i].sock, &rfds);
        if (srv->clients[i].send_len > srv->clients[i].send_pos) {
            FD_SET(srv->clients[i].sock, &wfds);
            has_write = 1;
        }
        if (srv->clients[i].sock > maxfd) maxfd = srv->clients[i].sock;
    }
    /* P1-15修复: 确保maxfd不超过FD_SETSIZE */
    if ((int)maxfd >= FD_SETSIZE) maxfd = FD_SETSIZE - 1;
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
            /* v9.11修复: 先握手再设置非阻塞，防止recv()在非阻塞模式下立即返回失败。
             * 关键: Windows上accept()继承监听socket的非阻塞属性，
             * 必须显式设为阻塞模式才能完成握手。 */
            mutex_lock(srv->mutex);
            int found = -1;
            for (int i = 0; i < WS_MAX_CLIENTS; i++) {
                if (!srv->clients[i].active) { found = i; break; }
            }
            if (found >= 0) {
                ws_client_init(&srv->clients[found], client, client_addr);
                srv->clients[found].active = 1;
                mutex_unlock(srv->mutex);
                /* 握手前强制设为阻塞模式 */
                ws_set_nonblock(client, 0);
                if (ws_do_handshake(client) != 0) {
                    ws_client_close(&srv->clients[found]);
                } else {
                    /* 握手成功后再设置非阻塞 */
                    ws_set_nonblock(client, 1);
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
            /* P1修复: select路径也使用持久接收缓冲区实现跨TCP段帧重组。
             * P0修复: 添加帧头越界读取防护。 */
            unsigned char tmp[8192];
            int n = (int)recv(cli->sock, (char*)tmp, sizeof(tmp), 0);
            if (n <= 0) { ws_client_close(cli); continue; }

            /* 将新数据追加到持久接收缓冲区 */
            if (cli->recv_len + (size_t)n > WS_MAX_MESSAGE_SIZE) {
                cli->recv_len = 0;
                ws_client_close(cli);
                continue;
            }
            memcpy(cli->recv_buffer + cli->recv_len, tmp, (size_t)n);
            cli->recv_len += (size_t)n;

            size_t remain = cli->recv_len;
            size_t consumed = 0;
            while (remain > 0) {
                unsigned char* ptr = (unsigned char*)cli->recv_buffer + consumed;

                /* P0修复: 在读取帧头字段前检查是否有足够字节，防止越界读取 */
                if (remain < 2) break;  /* 至少需要2字节读取opcode和基本payload_len */
                size_t header_size = 2;
                uint8_t raw_plen = ptr[1] & 0x7F;
                if (raw_plen == 126) {
                    if (remain < 4) break;  /* 需要额外2字节读取16位extended payload length */
                    header_size += 2;
                } else if (raw_plen == 127) {
                    if (remain < 10) break;  /* 需要额外8字节读取64位extended payload length */
                    header_size += 8;
                }
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
                if (frame_size > remain) break;  /* 帧不完整，等待更多数据到达 */

                uint8_t opcode;
                unsigned char* payload;
                size_t payload_len;
                int fin_flag_select;
                /* L-006修复: select路径也使用ws_parse_frame_with_fin支持分片 */
                if (ws_parse_frame_with_fin(ptr, remain, &opcode, &payload, &payload_len, &fin_flag_select) != 0) break;
                if (opcode == 0x08) {
                    if (masked) safe_free((void**)&payload);
                    ws_client_close(cli);
                    break;
                }
                else if (opcode == 0x09) { ws_send_pong(cli->sock, payload, payload_len); }
                else if (opcode == 0x0A) { }
                /* L-006修复: select路径数据帧分片处理 */
                else if ((opcode == 0x01 || opcode == 0x02) && !fin_flag_select) {
                    /* 分片首帧: 开始累积 */
                    if (!cli->fragment_active) {
                        cli->fragment_active = 1;
                        cli->fragment_opcode = opcode;
                        cli->fragment_len = 0;
                    }
                    ws_fragment_append(cli, payload, payload_len);
                } else if (opcode == 0x00) {
                    /* 延续帧: 追加并检测尾帧 */
                    if (cli->fragment_active) {
                        ws_fragment_append(cli, payload, payload_len);
                        if (fin_flag_select) {
                            cli->fragment_active = 0;
                            cli->fragment_len = 0;
                        }
                    }
                }
                /* B-012修复: 若帧是掩码帧，释放ws_parse_frame中分配的独立缓冲区 */
                if (masked) safe_free((void**)&payload);
                consumed += frame_size;
                remain -= frame_size;
            }
            /* 将未消费的数据移到缓冲区开头，供下次recv继续累积 */
            if (consumed > 0 && remain > 0) {
                memmove(cli->recv_buffer, cli->recv_buffer + consumed, remain);
            }
            cli->recv_len = remain;
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
