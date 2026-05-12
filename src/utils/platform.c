/**
 * @file platform.c
 * @brief 跨平台抽象层实现
 * 
 * 提供跨平台的系统函数实现，解决Windows/Unix头文件冲突问题。
 * 遵循纯C语言原则，不依赖第三方库。
 */

// 禁用Windows CRT安全警告

// Windows平台定义（WIN32_LEAN_AND_MEAN 由 CMake 全局定义）
#include "selflnn/utils/platform.h"
#include "selflnn/utils/memory_utils.h"
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <sys/types.h>
#include <intrin.h>

// Windows特定常量定义
#ifndef IPPROTO_IPV6
#define IPPROTO_IPV6 41
#endif
#else
#define PLATFORM_UNIX
#include <pthread.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/prctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <dirent.h>
#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif
#endif

/* 平台检测 */
PlatformType platform_get_type(void) {
#if defined(_WIN32) || defined(_WIN64)
    return PLATFORM_WINDOWS;
#elif defined(__linux__)
    return PLATFORM_LINUX;
#elif defined(__APPLE__)
    return PLATFORM_MACOS;
#else
    return PLATFORM_UNKNOWN;
#endif
}

/* 线程函数 */
#if defined(_WIN32) || defined(_WIN64)
static DWORD WINAPI thread_wrapper(LPVOID arg) {
    // 使用明确的类型声明而不是typeof
    typedef struct {
        void* (*start_routine)(void*);
        void* arg;
    } ThreadParams;
    
    ThreadParams* params = (ThreadParams*)arg;
    void* result = params->start_routine(params->arg);
    safe_free((void**)&params);
    return (DWORD)(uintptr_t)result;
}
#endif

ThreadHandle thread_create(void* (*start_routine)(void*), void* arg) {
#if defined(_WIN32) || defined(_WIN64)
    typedef struct {
        void* (*start_routine)(void*);
        void* arg;
    } ThreadParams;
    
    ThreadParams* params = safe_malloc(sizeof(ThreadParams));
    if (!params) return NULL;
    params->start_routine = start_routine;
    params->arg = arg;
    
    HANDLE thread = CreateThread(NULL, 0, thread_wrapper, params, 0, NULL);
    if (!thread) {
        safe_free((void**)&params);
        return NULL;
    }
    return (ThreadHandle)thread;
#else
    pthread_t thread;
    if (pthread_create(&thread, NULL, start_routine, arg) != 0) {
        return NULL;
    }
    return (ThreadHandle)(uintptr_t)thread;
#endif
}

int thread_set_name(const char* name) {
    if (!name) return -1;
#if defined(_WIN32) || defined(_WIN64)
    typedef HRESULT (WINAPI *SetThreadDescriptionFunc)(HANDLE, PCWSTR);
    HMODULE hKernel32 = GetModuleHandleW(L"kernel32.dll");
    if (hKernel32) {
        SetThreadDescriptionFunc pSetThreadDescription = (SetThreadDescriptionFunc)
            GetProcAddress(hKernel32, "SetThreadDescription");
        if (pSetThreadDescription) {
            int wlen = MultiByteToWideChar(CP_UTF8, 0, name, -1, NULL, 0);
            if (wlen > 0) {
                wchar_t* wname = (wchar_t*)malloc(wlen * sizeof(wchar_t));
                if (wname) {
                    MultiByteToWideChar(CP_UTF8, 0, name, -1, wname, wlen);
                    pSetThreadDescription(GetCurrentThread(), wname);
                    free(wname);
                }
            }
        }
    }
#elif defined(__APPLE__)
    pthread_setname_np(name);
#else
    /* Linux: prctl PR_SET_NAME 最大15字符 */
    if (strlen(name) > 15) {
        char truncated[16];
        strncpy(truncated, name, 15);
        truncated[15] = '\0';
        prctl(PR_SET_NAME, truncated, 0, 0, 0);
    } else {
        prctl(PR_SET_NAME, name, 0, 0, 0);
    }
#endif
    return 0;
}

/* ========== 目录遍历接口实现 ========== */

#if defined(_WIN32) || defined(_WIN64)
/* Windows内部结构：包装FindFirstFile/FindNextFile句柄 */
typedef struct {
    HANDLE find_handle;
    WIN32_FIND_DATAA find_data;
    char search_path[1024];
    int first_call;
} PlatformDirContext;
#else
/* Unix内部结构：包装opendir/readdir句柄 */
typedef struct {
    DIR* dir;
    char dir_path[1024];
} PlatformDirContext;
#endif

PlatformDirHandle platform_dir_open(const char* dir_path) {
    if (!dir_path || dir_path[0] == '\0') return NULL;

    PlatformDirContext* ctx = (PlatformDirContext*)safe_calloc(1, sizeof(PlatformDirContext));
    if (!ctx) return NULL;

#if defined(_WIN32) || defined(_WIN64)
    size_t path_len = strlen(dir_path);
    if (path_len + 3 > sizeof(ctx->search_path)) {
        safe_free((void**)&ctx);
        return NULL;
    }
    /* 构建搜索路径：dir_path\* */
    snprintf(ctx->search_path, sizeof(ctx->search_path), "%s\\*", dir_path);
    ctx->find_handle = FindFirstFileA(ctx->search_path, &ctx->find_data);
    if (ctx->find_handle == INVALID_HANDLE_VALUE) {
        safe_free((void**)&ctx);
        return NULL;
    }
    ctx->first_call = 1;
#else
    ctx->dir = opendir(dir_path);
    if (!ctx->dir) {
        safe_free((void**)&ctx);
        return NULL;
    }
    strncpy(ctx->dir_path, dir_path, sizeof(ctx->dir_path) - 1);
    ctx->dir_path[sizeof(ctx->dir_path) - 1] = '\0';
#endif

    return (PlatformDirHandle)ctx;
}

int platform_dir_read(PlatformDirHandle handle, PlatformDirEntry* entry) {
    if (!handle || !entry) return -1;

    PlatformDirContext* ctx = (PlatformDirContext*)handle;

#if defined(_WIN32) || defined(_WIN64)
    /* 首次调用时，FindFirstFile已经读取了第一个条目 */
    if (ctx->first_call) {
        ctx->first_call = 0;
    } else {
        if (!FindNextFileA(ctx->find_handle, &ctx->find_data)) {
            return 0;
        }
    }
    /* 跳过 . 和 .. */
    while (strcmp(ctx->find_data.cFileName, ".") == 0 ||
           strcmp(ctx->find_data.cFileName, "..") == 0) {
        if (!FindNextFileA(ctx->find_handle, &ctx->find_data)) {
            return 0;
        }
    }

    /* 填充条目信息 */
    strncpy(entry->name, ctx->find_data.cFileName, sizeof(entry->name) - 1);
    entry->name[sizeof(entry->name) - 1] = '\0';

    /* 构建完整路径 */
    {
        const char* sep = strrchr(ctx->search_path, '\\');
        if (sep) {
            size_t base_len = (size_t)(sep - ctx->search_path);
            size_t name_len = strlen(entry->name);
            if (base_len + 1 + name_len < sizeof(entry->path)) {
                memcpy(entry->path, ctx->search_path, base_len);
                entry->path[base_len] = '\\';
                memcpy(entry->path + base_len + 1, entry->name, name_len + 1);
            }
        }
    }

    entry->is_directory = (ctx->find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) ? 1 : 0;
    entry->is_regular_file = !entry->is_directory;
    entry->file_size = ((long long)ctx->find_data.nFileSizeHigh << 32) | (long long)ctx->find_data.nFileSizeLow;
    entry->last_modified = 0; /* 需要转换FILETIME到time_t，简化处理 */
#else
    struct dirent* dp = readdir(ctx->dir);
    if (!dp) return 0;

    /* 跳过 . 和 .. */
    while (strcmp(dp->d_name, ".") == 0 || strcmp(dp->d_name, "..") == 0) {
        dp = readdir(ctx->dir);
        if (!dp) return 0;
    }

    /* 填充条目信息 */
    strncpy(entry->name, dp->d_name, sizeof(entry->name) - 1);
    entry->name[sizeof(entry->name) - 1] = '\0';

    snprintf(entry->path, sizeof(entry->path), "%s/%s", ctx->dir_path, entry->name);

    /* 获取文件状态信息 */
    struct stat st;
    if (stat(entry->path, &st) == 0) {
        entry->is_directory = S_ISDIR(st.st_mode) ? 1 : 0;
        entry->is_regular_file = S_ISREG(st.st_mode) ? 1 : 0;
        entry->file_size = (long long)st.st_size;
        entry->last_modified = (long)st.st_mtime;
    } else {
        entry->is_directory = 0;
        entry->is_regular_file = 0;
        entry->file_size = 0;
        entry->last_modified = 0;
    }
#endif

    return 1;
}

int platform_dir_close(PlatformDirHandle handle) {
    if (!handle) return -1;

    PlatformDirContext* ctx = (PlatformDirContext*)handle;

#if defined(_WIN32) || defined(_WIN64)
    if (ctx->find_handle != INVALID_HANDLE_VALUE) {
        FindClose(ctx->find_handle);
    }
#else
    if (ctx->dir) {
        closedir(ctx->dir);
    }
#endif

    safe_free((void**)&ctx);
    return 0;
}

/* ========== 文件/目录操作接口实现 ========== */

int platform_dir_create(const char* path) {
    if (!path || path[0] == '\0') return -1;

#if defined(_WIN32) || defined(_WIN64)
    if (CreateDirectoryA(path, NULL)) return 0;
    /* 如果已存在则视为成功 */
    if (GetLastError() == ERROR_ALREADY_EXISTS) return 0;
    return -1;
#else
    if (mkdir(path, 0755) == 0) return 0;
    if (errno == EEXIST) return 0;
    return -1;
#endif
}

int platform_dir_remove(const char* path) {
    if (!path || path[0] == '\0') return -1;

#if defined(_WIN32) || defined(_WIN64)
    if (RemoveDirectoryA(path)) return 0;
    return -1;
#else
    if (rmdir(path) == 0) return 0;
    return -1;
#endif
}

int platform_file_remove(const char* path) {
    if (!path || path[0] == '\0') return -1;

#if defined(_WIN32) || defined(_WIN64)
    if (DeleteFileA(path)) return 0;
    return -1;
#else
    if (unlink(path) == 0) return 0;
    return -1;
#endif
}

long long platform_file_size(const char* path) {
    if (!path || path[0] == '\0') return -1;

#if defined(_WIN32) || defined(_WIN64)
    WIN32_FILE_ATTRIBUTE_DATA attr;
    if (GetFileAttributesExA(path, GetFileExInfoStandard, &attr)) {
        if (attr.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) return -1;
        return ((long long)attr.nFileSizeHigh << 32) | (long long)attr.nFileSizeLow;
    }
    return -1;
#else
    struct stat st;
    if (stat(path, &st) == 0) {
        if (S_ISDIR(st.st_mode)) return -1;
        return (long long)st.st_size;
    }
    return -1;
#endif
}

int platform_path_is_directory(const char* path) {
    if (!path || path[0] == '\0') return 0;

#if defined(_WIN32) || defined(_WIN64)
    DWORD attr = GetFileAttributesA(path);
    if (attr == INVALID_FILE_ATTRIBUTES) return 0;
    return (attr & FILE_ATTRIBUTE_DIRECTORY) ? 1 : 0;
#else
    struct stat st;
    if (stat(path, &st) != 0) return 0;
    return S_ISDIR(st.st_mode) ? 1 : 0;
#endif
}

int thread_join(ThreadHandle thread, void** retval) {
#if defined(_WIN32) || defined(_WIN64)
    DWORD result = WaitForSingleObject((HANDLE)thread, INFINITE);
    if (result != WAIT_OBJECT_0) return -1;
    
    DWORD exit_code;
    if (GetExitCodeThread((HANDLE)thread, &exit_code)) {
        if (retval) *retval = (void*)(uintptr_t)exit_code;
    }
    CloseHandle((HANDLE)thread);
    return 0;
#else
    void* thread_result;
    int rc = pthread_join((pthread_t)(uintptr_t)thread, &thread_result);
    if (rc != 0) return -1;
    if (retval) *retval = thread_result;
    return 0;
#endif
}

void thread_exit(void* retval) {
#if defined(_WIN32) || defined(_WIN64)
    ExitThread((DWORD)(uintptr_t)retval);
#else
    pthread_exit(retval);
#endif
}

/* 互斥锁函数 */
MutexHandle mutex_create(void) {
#if defined(_WIN32) || defined(_WIN64)
    HANDLE mutex = CreateMutex(NULL, FALSE, NULL);
    return (MutexHandle)mutex;
#else
    pthread_mutex_t* mutex = safe_malloc(sizeof(pthread_mutex_t));
    if (!mutex) return NULL;
    if (pthread_mutex_init(mutex, NULL) != 0) {
        safe_free((void**)&mutex);
        return NULL;
    }
    return (MutexHandle)mutex;
#endif
}

int mutex_lock(MutexHandle mutex) {
#if defined(_WIN32) || defined(_WIN64)
    DWORD result = WaitForSingleObject((HANDLE)mutex, INFINITE);
    return (result == WAIT_OBJECT_0) ? 0 : -1;
#else
    return pthread_mutex_lock((pthread_mutex_t*)mutex);
#endif
}

int mutex_unlock(MutexHandle mutex) {
#if defined(_WIN32) || defined(_WIN64)
    return ReleaseMutex((HANDLE)mutex) ? 0 : -1;
#else
    return pthread_mutex_unlock((pthread_mutex_t*)mutex);
#endif
}

int mutex_trylock(MutexHandle mutex) {
#if defined(_WIN32) || defined(_WIN64)
    DWORD result = WaitForSingleObject((HANDLE)mutex, 0);
    return (result == WAIT_OBJECT_0) ? 0 : -1;
#else
    return pthread_mutex_trylock((pthread_mutex_t*)mutex);
#endif
}

void mutex_destroy(MutexHandle mutex) {
#if defined(_WIN32) || defined(_WIN64)
    CloseHandle((HANDLE)mutex);
#else
    pthread_mutex_destroy((pthread_mutex_t*)mutex);
    safe_free((void**)&mutex);
#endif
}

/* 时间相关函数 */
TimeValue time_get_current(void) {
    TimeValue tv = {0};
#if defined(_WIN32) || defined(_WIN64)
    FILETIME ft;
    GetSystemTimeAsFileTime(&ft);
    ULARGE_INTEGER uli;
    uli.LowPart = ft.dwLowDateTime;
    uli.HighPart = ft.dwHighDateTime;
    // Windows FILETIME是100纳秒间隔，从1601-01-01开始
    // 转换为Unix时间戳（1970-01-01开始的微秒）
    uli.QuadPart -= 116444736000000000ULL; // 1601到1970的100纳秒间隔
    tv.seconds = (unsigned long)(uli.QuadPart / 10000000ULL);
    tv.microseconds = (unsigned long)((uli.QuadPart % 10000000ULL) / 10);
#else
    struct timeval current;
    gettimeofday(&current, NULL);
    tv.seconds = (unsigned long)current.tv_sec;
    tv.microseconds = (unsigned long)current.tv_usec;
#endif
    return tv;
}

void time_sleep_ms(unsigned int milliseconds) {
#if defined(_WIN32) || defined(_WIN64)
    Sleep(milliseconds);
#else
    usleep(milliseconds * 1000);
#endif
}

void platform_sleep_ms(unsigned int milliseconds) {
    time_sleep_ms(milliseconds);
}

/* 内存相关函数 */
void* memory_map(size_t size) {
#if defined(_WIN32) || defined(_WIN64)
    HANDLE mapping = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, 
                                      PAGE_READWRITE, 0, (DWORD)size, NULL);
    if (!mapping) return NULL;
    
    void* addr = MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, size);
    CloseHandle(mapping);
    return addr;
#else
    void* addr = mmap(NULL, size, PROT_READ | PROT_WRITE, 
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return (addr == MAP_FAILED) ? NULL : addr;
#endif
}

int memory_unmap(void* addr, size_t size) {
#if defined(_WIN32) || defined(_WIN64)
    (void)size;  // Windows版本未使用size参数
    return UnmapViewOfFile(addr) ? 0 : -1;
#else
    return munmap(addr, size) == 0 ? 0 : -1;
#endif
}

/* 网络函数 */
SocketHandle socket_create(int domain, int socket_type, int protocol) {
#if defined(_WIN32) || defined(_WIN64)
    static int initialized = 0;
    if (!initialized) {
        WSADATA wsaData;
        WSAStartup(MAKEWORD(2, 2), &wsaData);
        initialized = 1;
    }
#endif
    return (SocketHandle)socket(domain, socket_type, protocol);
}

int socket_bind(SocketHandle sock, const char* addr, unsigned short port) {
    struct sockaddr_in sa;
    memset(&sa, 0, sizeof(sa));
    sa.sin_family = AF_INET;
    sa.sin_port = htons(port);
    
    if (addr && strcmp(addr, "0.0.0.0") != 0) {
        inet_pton(AF_INET, addr, &sa.sin_addr);
    } else {
        sa.sin_addr.s_addr = INADDR_ANY;
    }
    
    return bind(sock, (struct sockaddr*)&sa, sizeof(sa)) == 0 ? 0 : -1;
}

int socket_listen(SocketHandle sock, int backlog) {
    return listen(sock, backlog) == 0 ? 0 : -1;
}

SocketHandle socket_accept(SocketHandle sock, char* client_addr, int addr_len) {
    struct sockaddr_in client;
    socklen_t client_len = sizeof(client);
    SocketHandle client_sock = (SocketHandle)accept(sock, (struct sockaddr*)&client, &client_len);
    
    if (client_sock >= 0 && client_addr) {
        inet_ntop(AF_INET, &client.sin_addr, client_addr, addr_len);
    }
    
    return client_sock;
}

int socket_connect(SocketHandle sock, const char* addr, unsigned short port) {
    struct sockaddr_in sa;
    memset(&sa, 0, sizeof(sa));
    sa.sin_family = AF_INET;
    sa.sin_port = htons(port);
    inet_pton(AF_INET, addr, &sa.sin_addr);
    
    return connect(sock, (struct sockaddr*)&sa, sizeof(sa)) == 0 ? 0 : -1;
}

int socket_send(SocketHandle sock, const void* buf, size_t len) {
    // 将size_t安全地转换为int，但限制大小以避免溢出
    if (len > INT_MAX) {
        return -1;  // 数据太大，无法发送
    }
    return send(sock, buf, (int)len, 0);
}

int socket_recv(SocketHandle sock, void* buf, size_t len) {
    // 将size_t安全地转换为int，但限制大小以避免溢出
    if (len > INT_MAX) {
        return -1;  // 缓冲区太大，无法接收
    }
    return recv(sock, buf, (int)len, 0);
}

int socket_setsockopt(SocketHandle sock, int level, int optname, const void* optval, int optlen) {
#if defined(_WIN32) || defined(_WIN64)
    return setsockopt(sock, level, optname, (const char*)optval, optlen) == 0 ? 0 : -1;
#else
    return setsockopt(sock, level, optname, optval, (socklen_t)optlen) == 0 ? 0 : -1;
#endif
}

void socket_close(SocketHandle sock) {
#if defined(_WIN32) || defined(_WIN64)
    closesocket(sock);
#else
    close(sock);
#endif
}

/* 文件路径函数 */
char platform_path_separator(void) {
#if defined(_WIN32) || defined(_WIN64)
    return '\\';
#else
    return '/';
#endif
}

const char* platform_path_separator_str(void) {
#if defined(_WIN32) || defined(_WIN64)
    return "\\";
#else
    return "/";
#endif
}

int platform_path_join(const char* base, const char* component,
                       char* out_buffer, size_t buffer_size) {
    if (!base || !component || !out_buffer || buffer_size < 2) {
        return -1;
    }

    size_t base_len = strlen(base);
    size_t comp_len = strlen(component);

    if (base_len + comp_len + 2 > buffer_size) {
        return -1;
    }

    strcpy(out_buffer, base);

    char sep = platform_path_separator();
    if (base_len > 0 && base[base_len - 1] != sep && base[base_len - 1] != '/' && base[base_len - 1] != '\\') {
        out_buffer[base_len] = sep;
        out_buffer[base_len + 1] = '\0';
    }

    strcat(out_buffer, component);
    return 0;
}

int platform_path_normalize(char* path) {
    if (!path) return -1;

    char sep = platform_path_separator();
    size_t len = strlen(path);

    for (size_t i = 0; i < len; i++) {
        if (path[i] == '/' || path[i] == '\\') {
            path[i] = sep;
        }
    }

    /* 移除重复的分隔符，但保留双斜杠开头的Windows UNC路径 */
    size_t write_idx = 0;
    int is_unc = 0;

#if defined(_WIN32) || defined(_WIN64)
    if (len >= 2 && path[0] == sep && path[1] == sep) {
        is_unc = 1;
        write_idx = 2;
    }
#endif

    for (size_t i = write_idx; i < len; i++) {
        if (path[i] == sep && i > 0 && path[i - 1] == sep) {
            continue;
        }
        path[write_idx++] = path[i];
    }
    path[write_idx] = '\0';

    return 0;
}

int platform_get_temp_dir(char* out_buffer, size_t buffer_size) {
    if (!out_buffer || buffer_size < 2) return -1;

#if defined(_WIN32) || defined(_WIN64)
    DWORD ret = GetTempPathA((DWORD)buffer_size, out_buffer);
    if (ret == 0 || ret > buffer_size) {
        strncpy(out_buffer, "C:\\Windows\\Temp", buffer_size - 1);
        out_buffer[buffer_size - 1] = '\0';
    }
#else
    const char* tmpdir = getenv("TMPDIR");
    if (!tmpdir) tmpdir = getenv("TMP");
    if (!tmpdir) tmpdir = getenv("TEMP");
    if (!tmpdir) tmpdir = "/tmp";

    size_t len = strlen(tmpdir);
    if (len >= buffer_size) return -1;
    strncpy(out_buffer, tmpdir, buffer_size - 1);
    out_buffer[buffer_size - 1] = '\0';
#endif

    /* 确保尾部有分隔符 */
    size_t len = strlen(out_buffer);
    char sep = platform_path_separator();
    if (len > 0 && out_buffer[len - 1] != sep) {
        if (len + 1 < buffer_size) {
            out_buffer[len] = sep;
            out_buffer[len + 1] = '\0';
        }
    }
    return 0;
}

int platform_get_config_dir(const char* app_name,
                            char* out_buffer, size_t buffer_size) {
    if (!out_buffer || buffer_size < 2) return -1;

#if defined(_WIN32) || defined(_WIN64)
    DWORD ret = GetEnvironmentVariableA("APPDATA", out_buffer, (DWORD)buffer_size);
    if (ret == 0 || ret > buffer_size) {
        ret = GetEnvironmentVariableA("USERPROFILE", out_buffer, (DWORD)buffer_size);
        if (ret > 0 && ret < buffer_size) {
            platform_path_join(out_buffer, "AppData\\Roaming", out_buffer, buffer_size);
        } else {
            strncpy(out_buffer, "C:\\", buffer_size - 1);
            out_buffer[buffer_size - 1] = '\0';
        }
    }
#else
    const char* xdg_config = getenv("XDG_CONFIG_HOME");
    if (xdg_config && strlen(xdg_config) > 0) {
        size_t len = strlen(xdg_config);
        if (len >= buffer_size) return -1;
        strncpy(out_buffer, xdg_config, buffer_size - 1);
        out_buffer[buffer_size - 1] = '\0';
    } else {
        const char* home = getenv("HOME");
        if (!home) return -1;
        size_t len = strlen(home);
        if (len + 9 >= buffer_size) return -1;
        platform_path_join(home, ".config", out_buffer, buffer_size);
    }
#endif

    if (app_name && strlen(app_name) > 0) {
        char tmp[1024];
        size_t current_len = strlen(out_buffer);
        if (current_len >= sizeof(tmp)) return -1;
        strncpy(tmp, out_buffer, sizeof(tmp) - 1);
        tmp[sizeof(tmp) - 1] = '\0';
        platform_path_join(tmp, app_name, out_buffer, buffer_size);
    }

    return 0;
}

int platform_get_data_dir(const char* app_name,
                          char* out_buffer, size_t buffer_size) {
    if (!out_buffer || buffer_size < 2) return -1;

#if defined(_WIN32) || defined(_WIN64)
    DWORD ret = GetEnvironmentVariableA("LOCALAPPDATA", out_buffer, (DWORD)buffer_size);
    if (ret == 0 || ret > buffer_size) {
        ret = GetEnvironmentVariableA("APPDATA", out_buffer, (DWORD)buffer_size);
        if (ret > 0 && ret < buffer_size) {
            char tmp[1024];
            strncpy(tmp, out_buffer, sizeof(tmp) - 1);
            tmp[sizeof(tmp) - 1] = '\0';
            platform_path_join(tmp, app_name ? app_name : "", out_buffer, buffer_size);
            return 0;
        }
    }
#else
    const char* xdg_data = getenv("XDG_DATA_HOME");
    if (xdg_data && strlen(xdg_data) > 0) {
        size_t len = strlen(xdg_data);
        if (len >= buffer_size) return -1;
        strncpy(out_buffer, xdg_data, buffer_size - 1);
        out_buffer[buffer_size - 1] = '\0';
    } else {
        const char* home = getenv("HOME");
        if (!home) return -1;
        platform_path_join(home, ".local/share", out_buffer, buffer_size);
    }
#endif

    if (app_name && strlen(app_name) > 0) {
        char tmp[1024];
        size_t current_len = strlen(out_buffer);
        if (current_len >= sizeof(tmp)) return -1;
        strncpy(tmp, out_buffer, sizeof(tmp) - 1);
        tmp[sizeof(tmp) - 1] = '\0';
        platform_path_join(tmp, app_name, out_buffer, buffer_size);
    }

    return 0;
}

int platform_path_exists(const char* path) {
    if (!path) return -1;

#if defined(_WIN32) || defined(_WIN64)
    DWORD attr = GetFileAttributesA(path);
    if (attr == INVALID_FILE_ATTRIBUTES) {
        return 0;
    }
    return 1;
#else
    if (access(path, F_OK) == 0) {
        return 1;
    }
    return 0;
#endif
}

int platform_path_optimized(const char* path, char* out_buffer, size_t buffer_size) {
    if (!path || !out_buffer || buffer_size < 2) return -1;

    /* 首先复制路径 */
    size_t path_len = strlen(path);
    if (path_len >= buffer_size) return -1;
    strncpy(out_buffer, path, buffer_size - 1);
    out_buffer[buffer_size - 1] = '\0';

    /* 规范化分隔符 */
    platform_path_normalize(out_buffer);

#if defined(_WIN32) || defined(_WIN64)
    /* Windows特有优化：对于长路径（超过260字符），添加 \\?\ 前缀 */
    if (path_len >= 260 && path_len + 4 < buffer_size) {
        char tmp[1024];
        strncpy(tmp, out_buffer, sizeof(tmp) - 1);
        tmp[sizeof(tmp) - 1] = '\0';

        if (tmp[0] == '\\' && tmp[1] == '\\') {
            /* UNC路径：\\?\UNC\server\share\... */
            out_buffer[0] = '\\';
            out_buffer[1] = '\\';
            out_buffer[2] = '?';
            out_buffer[3] = '\\';
            out_buffer[4] = 'U';
            out_buffer[5] = 'N';
            out_buffer[6] = 'C';
            /* 跳过原始UNC的双斜杠 */
            if (strlen(tmp) > 2) {
                strcpy(out_buffer + 7, tmp + 2);
            }
        } else {
            /* 普通路径：\\?\C:\... */
            char normalized[1024];
            strncpy(normalized, tmp, sizeof(normalized) - 1);
            normalized[sizeof(normalized) - 1] = '\0';

            out_buffer[0] = '\\';
            out_buffer[1] = '\\';
            out_buffer[2] = '?';
            out_buffer[3] = '\\';
            strcpy(out_buffer + 4, normalized);
        }
    }
#else
    /* Unix特有优化：解析符号链接获取真实路径 */
    char resolved[1024];
    if (realpath(path, resolved) != NULL) {
        size_t resolved_len = strlen(resolved);
        if (resolved_len < buffer_size) {
            strncpy(out_buffer, resolved, buffer_size - 1);
            out_buffer[buffer_size - 1] = '\0';
        }
    }
#endif

    return 0;
}

unsigned int platform_get_cache_line_size(void) {
#if defined(_WIN32) || defined(_WIN64)
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    /* Windows不直接暴露缓存行大小，使用CPUID检测 */
    int cpuInfo[4] = {0};
    __cpuid(cpuInfo, 0x80000006);
    unsigned int cache_line = (unsigned int)(cpuInfo[2] & 0xFF);
    return (cache_line > 0) ? cache_line : 64;
#elif defined(__linux__)
    long size = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
    if (size > 0) return (unsigned int)size;
    return 64;
#elif defined(__APPLE__)
    size_t line_size = 0;
    size_t sizeof_line_size = sizeof(line_size);
    if (sysctlbyname("hw.cachelinesize", &line_size, &sizeof_line_size, NULL, 0) == 0) {
        return (unsigned int)line_size;
    }
    return 64;
#else
    return 64;
#endif
}

unsigned int platform_get_page_size(void) {
#if defined(_WIN32) || defined(_WIN64)
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    return (unsigned int)sysInfo.dwPageSize;
#else
    long size = sysconf(_SC_PAGESIZE);
    if (size > 0) return (unsigned int)size;
    return 4096;
#endif
}

unsigned int platform_get_num_processors(void) {
#if defined(_WIN32) || defined(_WIN64)
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    return (unsigned int)sysInfo.dwNumberOfProcessors;
#else
    long count = sysconf(_SC_NPROCESSORS_ONLN);
    if (count > 0) return (unsigned int)count;
    return 1;
#endif
}

size_t platform_get_optimal_io_buffer_size(void) {
    unsigned int cache_line = platform_get_cache_line_size();
    unsigned int page_size = platform_get_page_size();

    /* 最优IO缓冲区通常是页面大小的倍数，
       在大多数平台上64KB是IO子系统的甜点 */
    size_t optimal = page_size * 16;

    /* 确保是缓存行大小的倍数 */
    if (cache_line > 0 && (optimal % cache_line) != 0) {
        optimal = (optimal / cache_line + 1) * cache_line;
    }

    return (optimal > 0) ? optimal : 65536;
}

/* CRT版本检测函数 */
int platform_detect_crt_version(CRTVersionInfo* info) {
    if (!info) return -1;

    memset(info, 0, sizeof(CRTVersionInfo));

#if defined(_WIN32) || defined(_WIN64)
    /* ========== Windows平台：检测MSVCRT/UCRT ========== */
    HINSTANCE hUcrt = NULL;
    HINSTANCE hMsvcrt = NULL;

    /* 首先尝试ucrtbase.dll（Universal CRT，VS2015+） */
    hUcrt = LoadLibraryA("ucrtbase.dll");
    if (hUcrt) {
        info->type = CRT_UCRT;
        strncpy(info->description, "Universal CRT (ucrtbase.dll)", sizeof(info->description) - 1);

        /* 尝试获取DLL版本信息 */
        char dll_path[MAX_PATH];
        DWORD path_len = GetModuleFileNameA((HMODULE)hUcrt, dll_path, MAX_PATH);
        if (path_len > 0) {
            DWORD handle = 0;
            DWORD ver_size = GetFileVersionInfoSizeA(dll_path, &handle);
            if (ver_size > 0) {
                void* ver_data = safe_malloc(ver_size);
                if (ver_data) {
                    if (GetFileVersionInfoA(dll_path, handle, ver_size, ver_data)) {
                        VS_FIXEDFILEINFO* file_info = NULL;
                        UINT info_len = 0;
                        if (VerQueryValueA(ver_data, "\\", (void**)&file_info, &info_len)) {
                            if (file_info && info_len >= sizeof(VS_FIXEDFILEINFO)) {
                                info->major_version = HIWORD(file_info->dwFileVersionMS);
                                info->minor_version = LOWORD(file_info->dwFileVersionMS);
                                info->patch_version = HIWORD(file_info->dwFileVersionLS);

                                char ver_str[48];
                                snprintf(ver_str, sizeof(ver_str), "Universal CRT v%d.%d.%d",
                                         info->major_version, info->minor_version, info->patch_version);
                                strncpy(info->description, ver_str, sizeof(info->description) - 1);
                            }
                        }
                    }
                    safe_free((void**)&ver_data);
                }
            }
        }
        FreeLibrary(hUcrt);
    }
    /* 尝试msvcrt.dll（旧版CRT，VS2015以前） */
    else if ((hMsvcrt = LoadLibraryA("msvcrt.dll")) != NULL) {
        info->type = CRT_MSVCRT;
        strncpy(info->description, "MSVCRT (msvcrt.dll)", sizeof(info->description) - 1);

        char dll_path[MAX_PATH];
        DWORD path_len = GetModuleFileNameA((HMODULE)hMsvcrt, dll_path, MAX_PATH);
        if (path_len > 0) {
            DWORD handle = 0;
            DWORD ver_size = GetFileVersionInfoSizeA(dll_path, &handle);
            if (ver_size > 0) {
                void* ver_data = safe_malloc(ver_size);
                if (ver_data) {
                    if (GetFileVersionInfoA(dll_path, handle, ver_size, ver_data)) {
                        VS_FIXEDFILEINFO* file_info = NULL;
                        UINT info_len = 0;
                        if (VerQueryValueA(ver_data, "\\", (void**)&file_info, &info_len)) {
                            if (file_info && info_len >= sizeof(VS_FIXEDFILEINFO)) {
                                info->major_version = HIWORD(file_info->dwFileVersionMS);
                                info->minor_version = LOWORD(file_info->dwFileVersionMS);
                                info->patch_version = HIWORD(file_info->dwFileVersionLS);

                                char ver_str[48];
                                snprintf(ver_str, sizeof(ver_str), "MSVCRT v%d.%d.%d",
                                         info->major_version, info->minor_version, info->patch_version);
                                strncpy(info->description, ver_str, sizeof(info->description) - 1);
                            }
                        }
                    }
                    safe_free((void**)&ver_data);
                }
            }
        }
        FreeLibrary(hMsvcrt);
    }
    /* 尝试msvcrt20.dll/msvcrt40.dll等变体 */
    else {
        const char* crt_variants[] = {
            "msvcrt20.dll", "msvcrt40.dll", "msvcrtd.dll", NULL
        };
        for (int i = 0; crt_variants[i] != NULL; i++) {
            HINSTANCE hVar = LoadLibraryA(crt_variants[i]);
            if (hVar) {
                info->type = CRT_MSVCRT;
                snprintf(info->description, sizeof(info->description),
                         "MSVCRT variant (%s)", crt_variants[i]);
                FreeLibrary(hVar);
                break;
            }
        }
        if (info->type == CRT_UNKNOWN) {
            strncpy(info->description, "Unknown Windows CRT", sizeof(info->description) - 1);
        }
    }

#elif defined(__linux__)
    /* ========== Linux平台：检测glibc/musl ========== */
#if defined(__GLIBC__) || defined(__GNU_LIBRARY__)
    info->type = CRT_GLIBC;
    info->major_version = __GLIBC__;
    info->minor_version = __GLIBC_MINOR__;
    /* 尝试通过运行时动态链接获取完整版本号（含patch）*/
    {
        FILE* fp_ver = fopen("/proc/self/maps", "r");
        if (fp_ver) {
            char line[512];
            while (fgets(line, sizeof(line), fp_ver)) {
                if (strstr(line, "libc")) {
                    const char* p = strstr(line, "libc-");
                    if (p) {
                        p += 5;
                        int maj = 0, min = 0, pat = 0;
                        if (sscanf(p, "%d.%d.%d", &maj, &min, &pat) >= 3) {
                            info->patch_version = pat;
                        }
                    }
                    break;
                }
            }
            fclose(fp_ver);
        }
    }
    {
        char ver_str[48];
        if (info->patch_version > 0) {
            snprintf(ver_str, sizeof(ver_str), "glibc %d.%d.%d",
                     info->major_version, info->minor_version, info->patch_version);
        } else {
            snprintf(ver_str, sizeof(ver_str), "glibc %d.%d",
                     info->major_version, info->minor_version);
        }
        strncpy(info->description, ver_str, sizeof(info->description) - 1);
    }
#elif defined(__MUSL__)
    info->type = CRT_MUSL;
    strncpy(info->description, "musl libc", sizeof(info->description) - 1);
#if defined(__MUSL_VERSION_MAJOR) && defined(__MUSL_VERSION_MINOR)
    info->major_version = __MUSL_VERSION_MAJOR;
    info->minor_version = __MUSL_VERSION_MINOR;
    {
        char ver_str[48];
        snprintf(ver_str, sizeof(ver_str), "musl %d.%d",
                 info->major_version, info->minor_version);
        strncpy(info->description, ver_str, sizeof(info->description) - 1);
    }
#endif
#else
    /* 无法通过宏确定具体类型，尝试读取libc.so.6的符号链接 */
    info->type = CRT_GLIBC;
    strncpy(info->description, "glibc (detected via soname)", sizeof(info->description) - 1);

    FILE* fp = fopen("/proc/self/maps", "r");
    if (fp) {
        char line[512];
        while (fgets(line, sizeof(line), fp)) {
            if (strstr(line, "libc") && strstr(line, ".so")) {
                /* 提取路径中的版本号 */
                const char* p = strstr(line, "libc-");
                if (p) {
                    p += 5; /* 跳过 "libc-" */
                    int maj = 0, min = 0, patch = 0;
                    if (sscanf(p, "%d.%d.%d", &maj, &min, &patch) >= 2) {
                        info->major_version = maj;
                        info->minor_version = min;
                        info->patch_version = patch;
                        char ver_str[48];
                        snprintf(ver_str, sizeof(ver_str), "glibc %d.%d.%d",
                                 maj, min, patch);
                        strncpy(info->description, ver_str, sizeof(info->description) - 1);
                    }
                }
                break;
            }
        }
        fclose(fp);
    }
#endif

#elif defined(__APPLE__)
    /* ========== macOS平台：检测dyld/libSystem ========== */
    info->type = CRT_BCRYPT;
    strncpy(info->description, "macOS libSystem (BSD libc)", sizeof(info->description) - 1);

    /* 通过sysctl获取macOS版本，间接推断libc版本 */
    char os_version[64] = {0};
    size_t os_len = sizeof(os_version) - 1;
    if (sysctlbyname("kern.osproductversion", os_version, &os_len, NULL, 0) == 0) {
        int maj = 0, min = 0, patch = 0;
        if (sscanf(os_version, "%d.%d.%d", &maj, &min, &patch) >= 2) {
            /* macOS版本到libSystem版本的映射：
               macOS 10.x → libSystem版本约 x+1
               macOS 11.x → libSystem版本约 11.x
               这是一个近似映射，主要用于信息参考 */
            info->major_version = maj;
            info->minor_version = min;
            info->patch_version = patch;

            char ver_str[64];
            snprintf(ver_str, sizeof(ver_str),
                     "macOS %d.%d.%d / libSystem (BSD libc)",
                     maj, min, patch);
            strncpy(info->description, ver_str, sizeof(info->description) - 1);
        }
    }

    /* 通过sysctl获取内核版本作为补充信息 */
    {
        char kern_ver[256] = {0};
        size_t kern_len = sizeof(kern_ver) - 1;
        if (sysctlbyname("kern.version", kern_ver, &kern_len, NULL, 0) == 0) {
            /* 仅在未获取到macOS版本时使用 */
            if (info->major_version == 0 && info->minor_version == 0) {
                int maj = 0, min = 0;
                if (sscanf(kern_ver, "Darwin Kernel Version %d.%d", &maj, &min) >= 1) {
                    info->major_version = maj;
                    info->minor_version = min;
                }
            }
        }
    }
#else
    /* ========== 未知平台 ========== */
    info->type = CRT_UNKNOWN;
    strncpy(info->description, "Unknown CRT", sizeof(info->description) - 1);
#endif

    return 0;
}
