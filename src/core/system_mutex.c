/* ZSFUSA: 系统互斥锁实现 — ZSFLNN-H-016修复: 补齐 trylock/init/destroy */
#include "selflnn/core/common.h"
#include <stdlib.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#endif

void system_mutex_lock(void* mutex_ptr) {
    if (!mutex_ptr) return;
#ifdef _WIN32
    EnterCriticalSection((CRITICAL_SECTION*)mutex_ptr);
#else
    pthread_mutex_lock((pthread_mutex_t*)mutex_ptr);
#endif
}

void system_mutex_unlock(void* mutex_ptr) {
    if (!mutex_ptr) return;
#ifdef _WIN32
    LeaveCriticalSection((CRITICAL_SECTION*)mutex_ptr);
#else
    pthread_mutex_unlock((pthread_mutex_t*)mutex_ptr);
#endif
}

int system_mutex_trylock(void* mutex_ptr) {
    if (!mutex_ptr) return -1;
#ifdef _WIN32
    return TryEnterCriticalSection((CRITICAL_SECTION*)mutex_ptr) ? 0 : -1;
#else
    return pthread_mutex_trylock((pthread_mutex_t*)mutex_ptr) == 0 ? 0 : -1;
#endif
}

void* system_mutex_create(void) {
#ifdef _WIN32
    CRITICAL_SECTION* cs = (CRITICAL_SECTION*)malloc(sizeof(CRITICAL_SECTION));
    if (!cs) return NULL;
    InitializeCriticalSection(cs);
    return cs;
#else
    pthread_mutex_t* mtx = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
    if (!mtx) return NULL;
    pthread_mutex_init(mtx, NULL);
    return mtx;
#endif
}

void system_mutex_destroy(void* mutex_ptr) {
    if (!mutex_ptr) return;
#ifdef _WIN32
    DeleteCriticalSection((CRITICAL_SECTION*)mutex_ptr);
#else
    pthread_mutex_destroy((pthread_mutex_t*)mutex_ptr);
#endif
    free(mutex_ptr);
}
