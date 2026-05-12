/**
 * @file platform.h
 * @brief 跨平台抽象层
 * 
 * 提供跨平台的系统函数抽象，解决Windows/Unix头文件冲突问题。
 * 遵循纯C语言原则，不依赖第三方库。
 */

#ifndef SELFLNN_PLATFORM_H
#define SELFLNN_PLATFORM_H

#include <stddef.h>

/**
 * @brief 标记未使用的参数，避免编译器警告
 * 用于接口回调函数中，接口契约需要但当前未使用的参数。
 */
#if defined(_MSC_VER)
#define UNUSED(x) (void)(x)
#elif defined(__GNUC__) || defined(__clang__)
#define UNUSED(x) (void)(x)
#else
#define UNUSED(x) (void)(x)
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 平台类型枚举
 */
typedef enum {
    PLATFORM_WINDOWS = 0,
    PLATFORM_LINUX = 1,
    PLATFORM_MACOS = 2,
    PLATFORM_UNKNOWN = 3
} PlatformType;

/**
 * @brief 线程句柄抽象
 */
typedef void* ThreadHandle;

/**
 * @brief 互斥锁句柄抽象
 */
typedef void* MutexHandle;

/**
 * @brief 套接字句柄抽象
 */
typedef int SocketHandle;

/**
 * @brief 时间结构体
 */
typedef struct {
    long seconds;      /**< 秒 */
    long microseconds; /**< 微秒 */
} TimeValue;

/**
 * @brief 获取当前平台类型
 * @return 平台类型
 */
PlatformType platform_get_type(void);

/**
 * @brief 线程相关函数
 */
ThreadHandle thread_create(void* (*start_routine)(void*), void* arg);

/**
 * @brief 设置当前线程名称（操作系统级别）
 * 
 * 在Windows上使用SetThreadDescription，Linux上使用prctl(PR_SET_NAME, ...)，
 * macOS上使用pthread_setname_np。
 * 
 * @param name 线程名称（最大15字符，Windows上最大128字符）
 * @return int 0=成功，其他=错误码
 */
int thread_set_name(const char* name);

int thread_join(ThreadHandle thread, void** retval);
void thread_exit(void* retval);

/**
 * @brief 互斥锁相关函数
 */
MutexHandle mutex_create(void);
int mutex_lock(MutexHandle mutex);
int mutex_unlock(MutexHandle mutex);
int mutex_trylock(MutexHandle mutex);
void mutex_destroy(MutexHandle mutex);

/**
 * @brief 时间相关函数
 */
TimeValue time_get_current(void);
void time_sleep_ms(unsigned int milliseconds);

/**
 * @brief 平台级休眠（毫秒级）
 * @param milliseconds 休眠毫秒数
 */
void platform_sleep_ms(unsigned int milliseconds);

/**
 * @brief 内存相关函数（跨平台内存映射）
 */
void* memory_map(size_t size);
int memory_unmap(void* addr, size_t size);

/**
 * @brief 网络相关函数（基础抽象）
 */
SocketHandle socket_create(int domain, int type, int protocol);
int socket_bind(SocketHandle sock, const char* addr, unsigned short port);
int socket_listen(SocketHandle sock, int backlog);
SocketHandle socket_accept(SocketHandle sock, char* client_addr, int addr_len);
int socket_connect(SocketHandle sock, const char* addr, unsigned short port);
int socket_send(SocketHandle sock, const void* buf, size_t len);
int socket_recv(SocketHandle sock, void* buf, size_t len);
int socket_setsockopt(SocketHandle sock, int level, int optname, const void* optval, int optlen);
void socket_close(SocketHandle sock);

/**
 * @brief 获取平台路径分隔符字符
 * @return char Windows返回'\\'，Unix返回'/'
 */
char platform_path_separator(void);

/**
 * @brief 获取平台路径分隔符字符串
 * @return const char* Windows返回"\\"，Unix返回"/"
 */
const char* platform_path_separator_str(void);

/**
 * @brief 连接两个路径组件，自动使用正确的路径分隔符
 * 
 * @param base 基础路径
 * @param component 要附加的路径组件
 * @param out_buffer [out] 输出缓冲区的完整路径
 * @param buffer_size 输出缓冲区大小
 * @return int 成功返回0，失败返回-1
 */
int platform_path_join(const char* base, const char* component,
                       char* out_buffer, size_t buffer_size);

/**
 * @brief 规范化路径分隔符（将所有分隔符转换为当前平台标准）
 * 
 * @param path 要规范化的路径（原地修改）
 * @return int 成功返回0，失败返回-1
 */
int platform_path_normalize(char* path);

/**
 * @brief 获取系统临时目录路径
 * 
 * @param out_buffer [out] 输出缓冲区
 * @param buffer_size 输出缓冲区大小
 * @return int 成功返回0，失败返回-1
 */
int platform_get_temp_dir(char* out_buffer, size_t buffer_size);

/**
 * @brief 获取应用配置目录路径
 * 
 * @param app_name 应用名称，用于创建子目录（可以为NULL）
 * @param out_buffer [out] 输出缓冲区
 * @param buffer_size 输出缓冲区大小
 * @return int 成功返回0，失败返回-1
 */
int platform_get_config_dir(const char* app_name,
                            char* out_buffer, size_t buffer_size);

/**
 * @brief 获取应用数据目录路径
 * 
 * @param app_name 应用名称，用于创建子目录（可以为NULL）
 * @param out_buffer [out] 输出缓冲区
 * @param buffer_size 输出缓冲区大小
 * @return int 成功返回0，失败返回-1
 */
int platform_get_data_dir(const char* app_name,
                          char* out_buffer, size_t buffer_size);

/**
 * @brief 检查路径是否存在
 * 
 * @param path 要检查的路径
 * @return int 存在返回1，不存在返回0，错误返回-1
 */
int platform_path_exists(const char* path);

/**
 * @brief 获取平台优化的路径版本
 * 
 * 将通用的正斜杠路径转换为当前平台的最优格式，
 * 例如Windows上转换为短路径名或使用\\?\前缀来绕过MAX_PATH限制。
 * 
 * @param path 输入路径
 * @param out_buffer [out] 优化后的路径输出缓冲区
 * @param buffer_size 输出缓冲区大小
 * @return int 成功返回0，失败返回-1
 */
int platform_path_optimized(const char* path, char* out_buffer, size_t buffer_size);

/**
 * @brief 获取CPU缓存行大小（用于内存对齐优化）
 * 
 * @return unsigned int 缓存行大小（字节），无法检测时返回64
 */
unsigned int platform_get_cache_line_size(void);

/**
 * @brief 获取系统内存页面大小
 * 
 * @return unsigned int 页面大小（字节）
 */
unsigned int platform_get_page_size(void);

/**
 * @brief 获取CPU核心/逻辑处理器数量
 * 
 * @return unsigned int 处理器核心数，无法检测时返回1
 */
unsigned int platform_get_num_processors(void);

/**
 * @brief 获取平台特有的最优文件缓冲区大小
 * 
 * @return size_t 最优缓冲区大小，默认64KB
 */
size_t platform_get_optimal_io_buffer_size(void);

/**
 * @brief CRT运行时版本类型枚举
 */
typedef enum {
    CRT_UNKNOWN = 0,   /**< 未知CRT类型 */
    CRT_MSVCRT,         /**< Windows MSVCRT (Visual Studio 2015以前) */
    CRT_UCRT,           /**< Windows Universal CRT (Visual Studio 2015+) */
    CRT_GLIBC,          /**< Linux glibc */
    CRT_MUSL,           /**< Linux musl */
    CRT_BCRYPT,         /**< macOS/BSD libc */
    CRT_NEWLIB          /**< 嵌入式newlib */
} CRTVersionType;

/**
 * @brief CRT运行时版本信息结构体
 */
typedef struct {
    CRTVersionType type;            /**< CRT类型 */
    int major_version;              /**< 主版本号 */
    int minor_version;              /**< 次版本号 */
    int patch_version;              /**< 补丁版本号 */
    char description[64];           /**< 可读描述字符串 */
} CRTVersionInfo;

/**
 * @brief 检测当前CRT运行时版本
 * 
 * Windows: 通过检测msvcrt.dll/ucrtbase.dll文件版本
 * Linux:   通过gnu_get_libc_version()或读取libc.so.6
 * macOS:   通过sysctl检测libSystem版本
 * 
 * @param info [out] 输出CRT版本信息结构体
 * @return int 成功返回0，失败返回-1
 */
int platform_detect_crt_version(CRTVersionInfo* info);

/**
 * @brief 目录条目信息结构体
 */
typedef struct {
    char name[256];          /**< 文件/目录名称（不含路径） */
    char path[1024];         /**< 完整路径 */
    int is_directory;        /**< 是否为目录 */
    int is_regular_file;     /**< 是否为普通文件 */
    long long file_size;     /**< 文件大小（字节），目录时为0 */
    long last_modified;      /**< 最后修改时间（Unix时间戳） */
} PlatformDirEntry;

/**
 * @brief 目录遍历句柄（不透明类型）
 */
typedef void* PlatformDirHandle;

/**
 * @brief 打开一个目录准备遍历
 * 
 * @param dir_path 目录路径
 * @return PlatformDirHandle 成功返回非NULL句柄，失败返回NULL
 */
PlatformDirHandle platform_dir_open(const char* dir_path);

/**
 * @brief 读取目录中的下一个条目
 * 
 * @param handle 由platform_dir_open返回的句柄
 * @param entry [out] 输出条目信息
 * @return int 成功读取返回1，没有更多条目返回0，出错返回-1
 */
int platform_dir_read(PlatformDirHandle handle, PlatformDirEntry* entry);

/**
 * @brief 关闭目录遍历句柄
 * 
 * @param handle 由platform_dir_open返回的句柄
 * @return int 成功返回0，失败返回-1
 */
int platform_dir_close(PlatformDirHandle handle);

/**
 * @brief 创建目录（包括父目录）
 * 
 * @param path 要创建的目录路径
 * @return int 成功返回0，失败返回-1
 */
int platform_dir_create(const char* path);

/**
 * @brief 移除空目录
 * 
 * @param path 要移除的目录路径
 * @return int 成功返回0，失败返回-1
 */
int platform_dir_remove(const char* path);

/**
 * @brief 删除文件
 * 
 * @param path 要删除的文件路径
 * @return int 成功返回0，失败返回-1
 */
int platform_file_remove(const char* path);

/**
 * @brief 获取文件大小
 * 
 * @param path 文件路径
 * @return long long 成功返回文件大小（字节），失败返回-1
 */
long long platform_file_size(const char* path);

/**
 * @brief 检查路径是否为目录
 * 
 * @param path 要检查的路径
 * @return int 是目录返回1，不是目录或出错返回0
 */
int platform_path_is_directory(const char* path);

#ifdef __cplusplus
}
#endif

#endif // SELFLNN_PLATFORM_H