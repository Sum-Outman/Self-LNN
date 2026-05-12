#include "selflnn/utils/time_utils.h"
#include "selflnn/utils/platform.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#include <sys/time.h>
#endif

/* 全局高精度频率缓存 */
#ifdef _WIN32
static LARGE_INTEGER g_perf_frequency = {0};
static int g_perf_freq_init = 0;
#endif

uint64_t time_utils_get_time_ms(void)
{
#ifdef _WIN32
    FILETIME ft;
    uint64_t tmp;
    GetSystemTimeAsFileTime(&ft);
    tmp = ((uint64_t)ft.dwHighDateTime << 32) | ft.dwLowDateTime;
    tmp /= 10000;
    return tmp;
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000 + (uint64_t)tv.tv_usec / 1000;
#endif
}

uint64_t time_utils_get_time_us(void)
{
#ifdef _WIN32
    FILETIME ft;
    uint64_t tmp;
    GetSystemTimeAsFileTime(&ft);
    tmp = ((uint64_t)ft.dwHighDateTime << 32) | ft.dwLowDateTime;
    tmp /= 10;
    return tmp;
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000000 + (uint64_t)tv.tv_usec;
#endif
}

/* K-041: 高精度计时器（纳秒级） */
uint64_t time_utils_get_time_ns(void) {
#ifdef _WIN32
    if (!g_perf_freq_init) {
        QueryPerformanceFrequency(&g_perf_frequency);
        g_perf_freq_init = 1;
    }
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    return (uint64_t)((double)counter.QuadPart * 1000000000.0 / (double)g_perf_frequency.QuadPart);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
#endif
}

/* K-041: 单调时钟（不受系统时间调整影响，用于计算时间差） */
uint64_t time_utils_get_time_monotonic_ms(void) {
#ifdef _WIN32
    if (!g_perf_freq_init) {
        QueryPerformanceFrequency(&g_perf_frequency);
        g_perf_freq_init = 1;
    }
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    return (uint64_t)((double)counter.QuadPart * 1000.0 / (double)g_perf_frequency.QuadPart);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000ULL + (uint64_t)ts.tv_nsec / 1000000ULL;
#endif
}

double time_utils_get_time_s(void)
{
#ifdef _WIN32
    FILETIME ft;
    uint64_t tmp;
    GetSystemTimeAsFileTime(&ft);
    tmp = ((uint64_t)ft.dwHighDateTime << 32) | ft.dwLowDateTime;
    return (double)tmp / 10000000.0;
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
#endif
}
