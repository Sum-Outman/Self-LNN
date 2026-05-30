/**
 * @file gpu_npu.c
 * @brief NPU 硬件检测与驱动扫描工具
 *
 * 提供统一的NPU硬件检测和驱动路径扫描功能。
 * 支持华为昇腾（Ascend）、寒武纪（Cambricon）、谷歌TPU三大NPU平台。
 * 自动探测常见安装路径和环境变量，无需用户手动配置。
 * 本文件仅负责硬件检测和驱动发现，不对检测结果做任何降级处理。
 */

#include "selflnn/gpu/gpu.h"
#include "selflnn/gpu/gpu_npu.h"
#include "selflnn/core/common.h"
#include "selflnn/utils/logging.h"
#include "selflnn/utils/memory_utils.h"
#include "gpu_internal.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <pthread.h>
#endif

/* ============================================================================
 * 常量定义
 * ============================================================================ */

#define NPU_MAX_DEVICES 256
#define NPU_DEVICE_NAME_MAX 128
#define NPU_PATH_MAX 512
#define NPU_MAX_SCAN_PATHS 64

/* ============================================================================
 * 文件/路径工具函数
 * ============================================================================ */

static int npu_file_exists(const char* path) {
#ifdef _WIN32
    DWORD attr = GetFileAttributesA(path);
    return (attr != INVALID_FILE_ATTRIBUTES && !(attr & FILE_ATTRIBUTE_DIRECTORY));
#else
    struct stat st;
    return (stat(path, &st) == 0 && S_ISREG(st.st_mode));
#endif
}

static int npu_dir_exists(const char* path) {
#ifdef _WIN32
    DWORD attr = GetFileAttributesA(path);
    return (attr != INVALID_FILE_ATTRIBUTES && (attr & FILE_ATTRIBUTE_DIRECTORY));
#else
    struct stat st;
    return (stat(path, &st) == 0 && S_ISDIR(st.st_mode));
#endif
}

/* ============================================================================
 * NPU 运行时硬件检测
 * ============================================================================ */

static int g_npu_hardware_detected = -1;
static NpuVendor g_detected_vendor = NPU_VENDOR_UNKNOWN;

static int detect_ascend_hardware(void) {
#ifdef _WIN32
    const char* reg_keys[] = {
        "SYSTEM\\CurrentControlSet\\Services\\Ascend",
        "SYSTEM\\CurrentControlSet\\Services\\Ascend310",
        "SYSTEM\\CurrentControlSet\\Services\\Ascend910",
        "SYSTEM\\CurrentControlSet\\Services\\davinci",
        NULL
    };
    for (int i = 0; reg_keys[i]; i++) {
        HKEY hKey;
        if (RegOpenKeyExA(HKEY_LOCAL_MACHINE, reg_keys[i], 0, KEY_READ, &hKey) == ERROR_SUCCESS) {
            RegCloseKey(hKey);
            return 1;
        }
    }
    if (npu_dir_exists("C:\\Program Files\\Huawei\\Ascend") ||
        npu_dir_exists("C:\\Program Files (x86)\\Huawei\\Ascend"))
        return 1;
    if (npu_file_exists("ascendcl.dll") || npu_file_exists("AscendCL.dll"))
        return 1;
#else
    const char* ascend_dirs[] = {
        "/usr/local/Ascend",
        "/usr/Ascend",
        "/usr/local/Ascend/ascend-toolkit",
        NULL
    };
    for (int i = 0; ascend_dirs[i]; i++) {
        if (npu_dir_exists(ascend_dirs[i])) return 1;
    }
    DIR* dev_dir = opendir("/dev");
    if (dev_dir) {
        struct dirent* entry;
        while ((entry = readdir(dev_dir)) != NULL) {
            if (strstr(entry->d_name, "ascend") ||
                strstr(entry->d_name, "davinci") ||
                strstr(entry->d_name, "Ascend")) {
                closedir(dev_dir);
                return 1;
            }
        }
        closedir(dev_dir);
    }
    if (npu_dir_exists("/sys/class/ascend")) return 1;
    const char* env_vars[] = {"ASCEND_HOME", "ASCEND_TOOLKIT_HOME", "NPU_HOST_LIB", NULL};
    for (int i = 0; env_vars[i]; i++) {
        if (getenv(env_vars[i])) return 1;
    }
    FILE* fp = popen("which npu-smi 2>/dev/null", "r");
    if (fp) {
        char buf[256];
        if (fgets(buf, sizeof(buf), fp) != NULL) {
            pclose(fp);
            return 1;
        }
        pclose(fp);
    }
#endif
    return 0;
}

static int detect_cambricon_hardware(void) {
#ifdef _WIN32
    const char* reg_keys[] = {
        "SYSTEM\\CurrentControlSet\\Services\\Cambricon",
        "SYSTEM\\CurrentControlSet\\Services\\mlu",
        NULL
    };
    for (int i = 0; reg_keys[i]; i++) {
        HKEY hKey;
        if (RegOpenKeyExA(HKEY_LOCAL_MACHINE, reg_keys[i], 0, KEY_READ, &hKey) == ERROR_SUCCESS) {
            RegCloseKey(hKey);
            return 1;
        }
    }
    if (npu_dir_exists("C:\\Program Files\\Cambricon") ||
        npu_dir_exists("C:\\Program Files\\NeuWare"))
        return 1;
    if (npu_file_exists("cnrt.dll") || npu_file_exists("libcnrt.dll"))
        return 1;
#else
    const char* neuware_dirs[] = {
        "/usr/local/neuware",
        "/usr/local/cambricon",
        "/usr/neuware",
        "/opt/neuware",
        NULL
    };
    for (int i = 0; neuware_dirs[i]; i++) {
        if (npu_dir_exists(neuware_dirs[i])) return 1;  /* ZSFA-FIX-F-016: neuw→neu typo */
    }
    DIR* dev_dir = opendir("/dev");
    if (dev_dir) {
        struct dirent* entry;
        while ((entry = readdir(dev_dir)) != NULL) {
            if (strstr(entry->d_name, "cambricon") ||
                strstr(entry->d_name, "mlu") ||
                strstr(entry->d_name, "cn")) {
                closedir(dev_dir);
                return 1;
            }
        }
        closedir(dev_dir);
    }
    const char* env_vars[] = {"NEUWARE_HOME", "CAMBRICON_HOME", "CAMBRICON_PATH", NULL};
    for (int i = 0; env_vars[i]; i++) {
        if (getenv(env_vars[i])) return 1;
    }
    FILE* fp = popen("which cnmon 2>/dev/null", "r");
    if (fp) {
        char buf[256];
        if (fgets(buf, sizeof(buf), fp) != NULL) {
            pclose(fp);
            return 1;
        }
        pclose(fp);
    }
#endif
    return 0;
}

static int detect_tpu_hardware(void) {
#ifdef _WIN32
    if (npu_dir_exists("C:\\Program Files\\Google\\TPU") ||
        npu_dir_exists("C:\\Program Files (x86)\\Google\\TPU"))
        return 1;
    if (npu_file_exists("libtpu.dll")) return 1;
#else
    DIR* dev_dir = opendir("/dev");
    if (dev_dir) {
        struct dirent* entry;
        while ((entry = readdir(dev_dir)) != NULL) {
            if (strstr(entry->d_name, "accel") || strstr(entry->d_name, "tpu")) {
                closedir(dev_dir);
                return 1;
            }
        }
        closedir(dev_dir);
    }
    if (npu_dir_exists("/sys/class/accel")) return 1;
    const char* env_vars[] = {"TPU_HOME", "TPU_LIBRARY_PATH", NULL};
    for (int i = 0; env_vars[i]; i++) {
        if (getenv(env_vars[i])) return 1;
    }
    if (npu_file_exists("/usr/lib/libtpu.so") || npu_file_exists("libtpu.so")) return 1;
#endif
    return 0;
}

static int npu_detect_hardware(void) {
    if (g_npu_hardware_detected >= 0) return g_npu_hardware_detected;

    LOG_INFO("正在检测NPU硬件...");

    if (detect_ascend_hardware()) {
        g_npu_hardware_detected = 1;
        g_detected_vendor = NPU_VENDOR_ASCEND;
        LOG_INFO("检测到华为昇腾NPU硬件");
        return 1;
    }
    if (detect_cambricon_hardware()) {
        g_npu_hardware_detected = 1;
        g_detected_vendor = NPU_VENDOR_CAMBRICON;
        LOG_INFO("检测到寒武纪MLU硬件");
        return 1;
    }
    if (detect_tpu_hardware()) {
        g_npu_hardware_detected = 1;
        g_detected_vendor = NPU_VENDOR_TPU;
        LOG_INFO("检测到谷歌TPU硬件");
        return 1;
    }

    g_npu_hardware_detected = 0;
    g_detected_vendor = NPU_VENDOR_UNKNOWN;
    LOG_INFO("未检测到NPU硬件");
    return 0;
}

/* ============================================================================
 * NPU SDK 动态加载
 * ============================================================================ */

#ifdef _WIN32
typedef HMODULE NpuLibHandle;
#define NPU_DLOPEN(a) LoadLibraryA(a)
#define NPU_DLSYM(a,b) GetProcAddress(a,b)
#define NPU_DLCLOSE(a) FreeLibrary(a)
#else
typedef void* NpuLibHandle;
#define NPU_DLOPEN(a) dlopen(a, RTLD_LAZY | RTLD_LOCAL)
#define NPU_DLSYM(a,b) dlsym(a,b)
#define NPU_DLCLOSE(a) dlclose(a)
#endif

typedef struct {
    NpuLibHandle handle;
    int loaded;
    int device_count;
    char device_name[256];
    char library_path[NPU_PATH_MAX];
} NpuSdkState;

static NpuSdkState g_npu_sdk = {0};
static int g_npu_init_called = 0;

static int npu_try_load_sdk(void) {
    if (g_npu_sdk.loaded) return 1;
    if (!npu_detect_hardware()) return 0;

    const char* lib_names[4][2] = {
        { "ascendcl.dll",       "libascendcl.so"       },
        { "cnrt.dll",           "libcnrt.so"           },
        { "libtpu.dll",         "libtpu.so"            },
        { NULL, NULL }
    };

    int vendor_idx = (int)g_detected_vendor;
    if (vendor_idx < 0 || vendor_idx > 2) return 0;

#ifdef _WIN32
    const char* lib_name = lib_names[vendor_idx][0];
#else
    const char* lib_name = lib_names[vendor_idx][1];
#endif
    if (!lib_name) return 0;

    g_npu_sdk.handle = NPU_DLOPEN(lib_name);
    if (!g_npu_sdk.handle) {
        LOG_WARN("无法加载NPU SDK库: %s", lib_name);
        return 0;
    }

    strncpy(g_npu_sdk.library_path, lib_name, NPU_PATH_MAX - 1);
    g_npu_sdk.loaded = 1;
    /* 尝试从SDK库中获取真实设备数量 */
    g_npu_sdk.device_count = 1;
#ifdef _WIN32
    {
        int (*count_fn)(void) = (int(*)(void))GetProcAddress(g_npu_sdk.handle, "getDeviceCount");
        if (count_fn) { int cnt = count_fn(); if (cnt > 0) g_npu_sdk.device_count = cnt; }
    }
#else
    {
        int (*count_fn)(void) = (int(*)(void))dlsym(g_npu_sdk.handle, "getDeviceCount");
        if (count_fn) { int cnt = count_fn(); if (cnt > 0) g_npu_sdk.device_count = cnt; }
    }
#endif
    snprintf(g_npu_sdk.device_name, sizeof(g_npu_sdk.device_name),
             "NPU-%s", npu_vendor_name(g_detected_vendor));

    LOG_INFO("NPU SDK加载成功: %s (供应商: %s)",
             lib_name, npu_vendor_name(g_detected_vendor));
    return 1;
}

/* ============================================================================
 * 驱动路径扫描辅助函数
 * ============================================================================ */

static void checked_path_add(NpuDriverScanResult* result, const char* path) {
    if (result->paths_checked >= NPU_MAX_SCAN_PATHS) return;
    strncpy(result->checked_paths[result->paths_checked], path, NPU_PATH_MAX - 1);
    result->checked_paths[result->paths_checked][NPU_PATH_MAX - 1] = '\0';
    result->paths_checked++;
}

static void found_path_add(NpuDriverScanResult* result, const char* path) {
    if (result->paths_found >= NPU_MAX_SCAN_PATHS) return;
    strncpy(result->found_paths[result->paths_found], path, NPU_PATH_MAX - 1);
    result->found_paths[result->paths_found][NPU_PATH_MAX - 1] = '\0';
    result->paths_found++;
}

static void try_find_library(NpuDriverScanResult* result,
                              const char* dir, const char* lib_name) {
    char full_path[NPU_PATH_MAX];
    snprintf(full_path, sizeof(full_path), "%s/%s", dir, lib_name);
    checked_path_add(result, full_path);
    if (npu_file_exists(full_path)) {
        strncpy(result->library_path, full_path, NPU_PATH_MAX - 1);
        result->library_path[NPU_PATH_MAX - 1] = '\0';
        result->found = 1;
        found_path_add(result, full_path);
    }
}

static void try_read_version_file(const char* path, char* version_buf,
                                   size_t buf_size) {
    if (!path || !npu_file_exists(path)) return;
    FILE* fp = fopen(path, "r");
    if (!fp) return;
    char line[256];
    while (fgets(line, sizeof(line), fp)) {
        size_t len = strlen(line);
        if (len > 0 && line[len - 1] == '\n') line[len - 1] = '\0';
        if (strlen(line) > 0 && line[0] != '#') {
            strncpy(version_buf, line, buf_size - 1);
            version_buf[buf_size - 1] = '\0';
            break;
        }
    }
    fclose(fp);
}

#ifndef _WIN32
static void scan_dev_directory(NpuDriverScanResult* result,
                                const char* dev_dir, const char* prefix) {
    DIR* dir = opendir(dev_dir);
    if (!dir) return;
    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_name[0] == '.') continue;
        if (strncmp(entry->d_name, prefix, strlen(prefix)) == 0) {
            char full_path[NPU_PATH_MAX];
            snprintf(full_path, sizeof(full_path), "%s/%s", dev_dir, entry->d_name);
            struct stat st;
            if (stat(full_path, &st) == 0 && result->device_count < NPU_MAX_DEVICES) {
                strncpy(result->device_names[result->device_count],
                        entry->d_name, NPU_DEVICE_NAME_MAX - 1);
                result->device_names[result->device_count][NPU_DEVICE_NAME_MAX - 1] = '\0';
                result->device_count++;
            }
        }
    }
    closedir(dir);
}
#endif

/* ============================================================================
 * 昇腾驱动路径扫描
 * ============================================================================ */

#define ASCEND_LIB_NAME "libascendcl.so"
#define ASCEND_LIB_NAME_WIN "ascendcl.dll"

static void scan_ascend_paths(NpuDriverScanResult* result) {
    memset(result, 0, sizeof(NpuDriverScanResult));
    result->vendor = NPU_VENDOR_ASCEND;

    const char* ascend_home = getenv("ASCEND_HOME");
    const char* toolkit_home = getenv("ASCEND_TOOLKIT_HOME");
    const char* ascend_root = getenv("ASCEND_ROOT");
    const char* npu_host_lib = getenv("NPU_HOST_LIB");

    char path[NPU_PATH_MAX];

#ifdef _WIN32
    const char* lib_name = ASCEND_LIB_NAME_WIN;
    if (ascend_home) {
        snprintf(path, sizeof(path), "%s/lib/%s", ascend_home, lib_name);
        try_find_library(result, ascend_home, lib_name);
        if (!result->found) {
            snprintf(path, sizeof(path), "%s/lib64/%s", ascend_home, lib_name);
            checked_path_add(result, path);
            if (npu_file_exists(path)) {
                strncpy(result->library_path, path, NPU_PATH_MAX - 1);
                result->found = 1;
                found_path_add(result, path);
            }
        }
    }
    if (toolkit_home && !result->found) {
        snprintf(path, sizeof(path), "%s/lib/%s", toolkit_home, lib_name);
        try_find_library(result, toolkit_home, lib_name);
    }
    const char* pf_dirs[] = {
        "C:\\Program Files\\Huawei\\Ascend",
        "C:\\Program Files (x86)\\Huawei\\Ascend",
        NULL
    };
    for (int i = 0; pf_dirs[i] && !result->found; i++) {
        snprintf(path, sizeof(path), "%s\\lib\\%s", pf_dirs[i], lib_name);
        checked_path_add(result, path);
        if (npu_file_exists(path)) {
            strncpy(result->library_path, path, NPU_PATH_MAX - 1);
            result->found = 1;
            strncpy(result->install_path, pf_dirs[i], NPU_PATH_MAX - 1);
            found_path_add(result, path);
        }
    }
#else
    const char* lib_name = ASCEND_LIB_NAME;
    if (npu_host_lib) {
        snprintf(path, sizeof(path), "%s/%s", npu_host_lib, lib_name);
        try_find_library(result, npu_host_lib, lib_name);
    }
    if (ascend_home && !result->found) {
        snprintf(path, sizeof(path), "%s/lib64/%s", ascend_home, lib_name);
        try_find_library(result, ascend_home, lib_name);
        if (!result->found) {
            snprintf(path, sizeof(path), "%s/lib/%s", ascend_home, lib_name);
            checked_path_add(result, path);
            if (npu_file_exists(path)) {
                strncpy(result->library_path, path, NPU_PATH_MAX - 1);
                result->found = 1;
                found_path_add(result, path);
            }
        }
    }
    if (toolkit_home && !result->found) {
        snprintf(path, sizeof(path), "%s/lib64/%s", toolkit_home, lib_name);
        try_find_library(result, toolkit_home, lib_name);
    }
    if (ascend_root && !result->found) {
        snprintf(path, sizeof(path), "%s/lib64/%s", ascend_root, lib_name);
        try_find_library(result, ascend_root, lib_name);
    }
    const char* common_paths[] = {
        "/usr/local/Ascend",
        "/usr/local/Ascend/ascend-toolkit",
        "/usr/local/Ascend/ascend-toolkit/latest",
        "/usr/local/Ascend/driver",
        "/usr/Ascend",
        "/usr/local/Ascend/nnae/latest",
        NULL
    };
    for (int i = 0; common_paths[i] && !result->found; i++) {
        if (!npu_dir_exists(common_paths[i])) {
            checked_path_add(result, common_paths[i]);
            continue;
        }
        const char* subdirs[] = { "lib64", "lib", "" };
        for (int j = 0; subdirs[j][0]; j++) {
            char lib_dir[NPU_PATH_MAX];
            if (subdirs[j][0]) {
                snprintf(lib_dir, sizeof(lib_dir), "%s/%s", common_paths[i], subdirs[j]);
            } else {
                snprintf(lib_dir, sizeof(lib_dir), "%s", common_paths[i]);
            }
            snprintf(path, sizeof(path), "%s/%s", lib_dir, lib_name);
            checked_path_add(result, path);
            if (npu_file_exists(path)) {
                strncpy(result->library_path, path, NPU_PATH_MAX - 1);
                strncpy(result->install_path, common_paths[i], NPU_PATH_MAX - 1);
                result->found = 1;
                found_path_add(result, path);
                break;
            }
        }
    }
    const char* ver_paths[] = {
        "/usr/local/Ascend/version.cfg",
        "/usr/local/Ascend/ascend-toolkit/latest/version.cfg",
        "/usr/Ascend/version.cfg",
        NULL
    };
    for (int i = 0; ver_paths[i]; i++) {
        checked_path_add(result, ver_paths[i]);
        try_read_version_file(ver_paths[i], result->version, sizeof(result->version));
        if (result->version[0]) break;
    }
    scan_dev_directory(result, "/dev", "ascend");
    scan_dev_directory(result, "/dev", "davinci");
    scan_dev_directory(result, "/dev", "Ascend");
    scan_dev_directory(result, "/dev/hisi_hdc", "");
    DIR* sys_dir = opendir("/sys/class/ascend");
    if (sys_dir) {
        struct dirent* entry;
        while ((entry = readdir(sys_dir)) != NULL) {
            if (entry->d_name[0] == '.') continue;
            if (result->device_count < NPU_MAX_DEVICES) {
                snprintf(result->device_names[result->device_count],
                         NPU_DEVICE_NAME_MAX, "ascend_%s", entry->d_name);
                result->device_count++;
            }
        }
        closedir(sys_dir);
    } else {
        checked_path_add(result, "/sys/class/ascend");
    }
#endif

    if (result->found) {
        LOG_INFO("昇腾NPU驱动发现: %s", result->library_path);
    }
}

/* ============================================================================
 * 寒武纪驱动路径扫描
 * ============================================================================ */

#define CAMBRICON_LIB_NAME "libcnrt.so"
#define CAMBRICON_LIB_NAME_WIN "cnrt.dll"

static void scan_cambricon_paths(NpuDriverScanResult* result) {
    memset(result, 0, sizeof(NpuDriverScanResult));
    result->vendor = NPU_VENDOR_CAMBRICON;

    const char* neuware_home = getenv("NEUWARE_HOME");
    const char* cambricon_home = getenv("CAMBRICON_HOME");
    const char* cambricon_path = getenv("CAMBRICON_PATH");

    char path[NPU_PATH_MAX];

#ifdef _WIN32
    const char* lib_name = CAMBRICON_LIB_NAME_WIN;
    if (neuware_home) {
        snprintf(path, sizeof(path), "%s/lib/%s", neuware_home, lib_name);
        try_find_library(result, neuware_home, lib_name);
    }
    if (cambricon_home && !result->found) {
        snprintf(path, sizeof(path), "%s/lib/%s", cambricon_home, lib_name);
        try_find_library(result, cambricon_home, lib_name);
    }
    const char* pf_dirs[] = {
        "C:\\Program Files\\Cambricon",
        "C:\\Program Files\\NeuWare",
        NULL
    };
    for (int i = 0; pf_dirs[i] && !result->found; i++) {
        snprintf(path, sizeof(path), "%s\\lib\\%s", pf_dirs[i], lib_name);
        checked_path_add(result, path);
        if (npu_file_exists(path)) {
            strncpy(result->library_path, path, NPU_PATH_MAX - 1);
            result->found = 1;
            strncpy(result->install_path, pf_dirs[i], NPU_PATH_MAX - 1);
            found_path_add(result, path);
        }
    }
#else
    const char* lib_name = CAMBRICON_LIB_NAME;
    if (neuware_home) {
        snprintf(path, sizeof(path), "%s/lib64/%s", neuware_home, lib_name);
        try_find_library(result, neuware_home, lib_name);
        if (!result->found) {
            snprintf(path, sizeof(path), "%s/lib/%s", neuware_home, lib_name);
            checked_path_add(result, path);
            if (npu_file_exists(path)) {
                strncpy(result->library_path, path, NPU_PATH_MAX - 1);
                result->found = 1;
                found_path_add(result, path);
            }
        }
    }
    if (cambricon_home && !result->found) {
        snprintf(path, sizeof(path), "%s/lib64/%s", cambricon_home, lib_name);
        try_find_library(result, cambricon_home, lib_name);
    }
    if (cambricon_path && !result->found) {
        snprintf(path, sizeof(path), "%s/%s", cambricon_path, lib_name);
        try_find_library(result, cambricon_path, lib_name);
    }
    const char* common_paths[] = {
        "/usr/local/neuware",
        "/usr/local/cambricon",
        "/usr/neuware",
        "/opt/neuware",
        "/usr/local/neuware/lib64",
        NULL
    };
    for (int i = 0; common_paths[i] && !result->found; i++) {
        if (!npu_dir_exists(common_paths[i])) {
            checked_path_add(result, common_paths[i]);
            continue;
        }
        const char* subdirs[] = { "lib64", "lib", "" };
        for (int j = 0; subdirs[j][0]; j++) {
            char lib_dir[NPU_PATH_MAX];
            if (subdirs[j][0]) {
                snprintf(lib_dir, sizeof(lib_dir), "%s/%s", common_paths[i], subdirs[j]);
            } else {
                snprintf(lib_dir, sizeof(lib_dir), "%s", common_paths[i]);
            }
            snprintf(path, sizeof(path), "%s/%s", lib_dir, lib_name);
            checked_path_add(result, path);
            if (npu_file_exists(path)) {
                strncpy(result->library_path, path, NPU_PATH_MAX - 1);
                strncpy(result->install_path, common_paths[i], NPU_PATH_MAX - 1);
                result->found = 1;
                found_path_add(result, path);
                break;
            }
        }
    }
    const char* ver_paths[] = {
        "/usr/local/neuware/version.txt",
        "/usr/local/neuware/VERSION",
        "/usr/local/cambricon/version.txt",
        NULL
    };
    for (int i = 0; ver_paths[i]; i++) {
        checked_path_add(result, ver_paths[i]);
        try_read_version_file(ver_paths[i], result->version, sizeof(result->version));
        if (result->version[0]) break;
    }
    scan_dev_directory(result, "/dev", "cambricon");
    scan_dev_directory(result, "/dev", "mlu");
    scan_dev_directory(result, "/dev", "cn");
#endif

    if (result->found) {
        LOG_INFO("寒武纪NPU驱动发现: %s", result->library_path);
    }
}

/* ============================================================================
 * 谷歌TPU驱动路径扫描
 * ============================================================================ */

#define TPU_LIB_NAME "libtpu.so"
#define TPU_LIB_NAME_WIN "libtpu.dll"

static void scan_tpu_paths(NpuDriverScanResult* result) {
    memset(result, 0, sizeof(NpuDriverScanResult));
    result->vendor = NPU_VENDOR_TPU;

    const char* tpu_home = getenv("TPU_HOME");
    const char* tpu_lib_path = getenv("TPU_LIBRARY_PATH");

    char path[NPU_PATH_MAX];

#ifdef _WIN32
    const char* lib_name = TPU_LIB_NAME_WIN;
    if (tpu_home) {
        snprintf(path, sizeof(path), "%s/lib/%s", tpu_home, lib_name);
        try_find_library(result, tpu_home, lib_name);
    }
    if (tpu_lib_path) {
        try_find_library(result, tpu_lib_path, lib_name);
    }
    const char* pf_dirs[] = {
        "C:\\Program Files\\Google\\TPU",
        "C:\\Program Files (x86)\\Google\\TPU",
        NULL
    };
    for (int i = 0; pf_dirs[i] && !result->found; i++) {
        snprintf(path, sizeof(path), "%s\\lib\\%s", pf_dirs[i], lib_name);
        checked_path_add(result, path);
        if (npu_file_exists(path)) {
            strncpy(result->library_path, path, NPU_PATH_MAX - 1);
            result->found = 1;
            strncpy(result->install_path, pf_dirs[i], NPU_PATH_MAX - 1);
            found_path_add(result, path);
        }
    }
#else
    const char* lib_name = TPU_LIB_NAME;
    if (tpu_lib_path) {
        try_find_library(result, tpu_lib_path, lib_name);
    }
    if (tpu_home) {
        snprintf(path, sizeof(path), "%s/lib64/%s", tpu_home, lib_name);
        try_find_library(result, tpu_home, lib_name);
        if (!result->found) {
            snprintf(path, sizeof(path), "%s/lib/%s", tpu_home, lib_name);
            checked_path_add(result, path);
            if (npu_file_exists(path)) {
                strncpy(result->library_path, path, NPU_PATH_MAX - 1);
                result->found = 1;
                found_path_add(result, path);
            }
        }
    }
    const char* common_paths[] = {
        "/usr/local/tpu",
        "/usr/lib",
        "/usr/local/lib",
        "/opt/tpu",
        NULL
    };
    for (int i = 0; common_paths[i] && !result->found; i++) {
        if (!npu_dir_exists(common_paths[i])) {
            checked_path_add(result, common_paths[i]);
            continue;
        }
        snprintf(path, sizeof(path), "%s/%s", common_paths[i], lib_name);
        checked_path_add(result, path);
        if (npu_file_exists(path)) {
            strncpy(result->library_path, path, NPU_PATH_MAX - 1);
            strncpy(result->install_path, common_paths[i], NPU_PATH_MAX - 1);
            result->found = 1;
            found_path_add(result, path);
            break;
        }
    }
    snprintf(path, sizeof(path), "/usr/lib/libtpu.so");
    checked_path_add(result, path);
    if (!result->found && npu_file_exists(path)) {
        strncpy(result->library_path, path, NPU_PATH_MAX - 1);
        result->found = 1;
        found_path_add(result, path);
    }
    scan_dev_directory(result, "/dev", "accel");
    scan_dev_directory(result, "/dev", "tpu");
    DIR* sys_dir = opendir("/sys/class/accel");
    if (sys_dir) {
        struct dirent* entry;
        while ((entry = readdir(sys_dir)) != NULL) {
            if (entry->d_name[0] == '.') continue;
            if (result->device_count < NPU_MAX_DEVICES) {
                snprintf(result->device_names[result->device_count],
                         NPU_DEVICE_NAME_MAX, "accel_%s", entry->d_name);
                result->device_count++;
            }
        }
        closedir(sys_dir);
    } else {
        checked_path_add(result, "/sys/class/accel");
    }
#endif

    if (result->found) {
        LOG_INFO("谷歌TPU驱动发现: %s", result->library_path);
    }
}

/* ============================================================================
 * 公共扫描 API 实现
 * ============================================================================ */

int npu_scan_driver_paths(NpuDriverScanReport* report) {
    if (!report) return -1;
    memset(report, 0, sizeof(NpuDriverScanReport));

    scan_ascend_paths(&report->ascend);
    scan_cambricon_paths(&report->cambricon);
    scan_tpu_paths(&report->tpu);

    report->total_devices = report->ascend.device_count
                          + report->cambricon.device_count
                          + report->tpu.device_count;
    report->any_found = report->ascend.found
                      || report->cambricon.found
                      || report->tpu.found;
    int count = 0;
    if (report->ascend.found) count++;
    if (report->cambricon.found) count++;
    if (report->tpu.found) count++;
    return count;
}

int npu_scan_single_driver(NpuVendor vendor, NpuDriverScanResult* result) {
    if (!result) return -1;
    switch (vendor) {
        case NPU_VENDOR_ASCEND:
            scan_ascend_paths(result);
            return result->found ? 1 : 0;
        case NPU_VENDOR_CAMBRICON:
            scan_cambricon_paths(result);
            return result->found ? 1 : 0;
        case NPU_VENDOR_TPU:
            scan_tpu_paths(result);
            return result->found ? 1 : 0;
        default:
            memset(result, 0, sizeof(NpuDriverScanResult));
            result->vendor = NPU_VENDOR_UNKNOWN;
            return -1;
    }
}

const char* npu_vendor_name(NpuVendor vendor) {
    switch (vendor) {
        case NPU_VENDOR_ASCEND:    return "华为昇腾 (Ascend)";
        case NPU_VENDOR_CAMBRICON: return "寒武纪 (Cambricon)";
        case NPU_VENDOR_TPU:       return "谷歌 TPU";
        default:                   return "未知 NPU";
    }
}

const char* npu_install_guide(NpuVendor vendor) {
    switch (vendor) {
        case NPU_VENDOR_ASCEND:
            return "安装指南:\n"
                   "  1. 从华为官方网站下载 Ascend 驱动和固件\n"
                   "  2. 运行 ./Ascend310-driver-*.run --full\n"
                   "  3. 运行 ./Ascend-acl-*.run --full\n"
                   "  4. 设置环境变量: export ASCEND_HOME=/usr/local/Ascend\n"
                   "  5. 验证: ls /dev/ascend* 或 npu-smi info";
        case NPU_VENDOR_CAMBRICON:
            return "安装指南:\n"
                   "  1. 从寒武纪官方网站下载 NeuWare SDK\n"
                   "  2. 运行 ./cambricon-mlu-driver-*.run\n"
                   "  3. 设置环境变量: export NEUWARE_HOME=/usr/local/neuware\n"
                   "  4. 验证: ls /dev/cambricon* 或 cnmon";
        case NPU_VENDOR_TPU:
            return "安装指南:\n"
                   "  1. 确保已安装 gasket 驱动: sudo modprobe gasket\n"
                   "  2. TPU 设备将显示为 /dev/accel*\n"
                   "  3. 安装 libtpu.so: pip install libtpu-nightly\n"
                   "  4. 设置环境变量: export TPU_LIBRARY_PATH=/path/to/libtpu.so\n"
                   "  5. 验证: ls /dev/accel*";
        default:
            return "未知 NPU 供应商，请参考硬件文档安装驱动。";
    }
}

GpuBackend npu_vendor_to_backend(NpuVendor vendor) {
    switch (vendor) {
        case NPU_VENDOR_ASCEND:    return GPU_BACKEND_ASCEND;
        case NPU_VENDOR_CAMBRICON: return GPU_BACKEND_CAMBRICON;
        case NPU_VENDOR_TPU:       return GPU_BACKEND_TPU;
        default:                   return GPU_BACKEND_CPU;
    }
}

NpuVendor npu_backend_to_vendor(GpuBackend backend) {
    switch (backend) {
        case GPU_BACKEND_ASCEND:    return NPU_VENDOR_ASCEND;
        case GPU_BACKEND_CAMBRICON: return NPU_VENDOR_CAMBRICON;
        case GPU_BACKEND_TPU:       return NPU_VENDOR_TPU;
        default:                    return NPU_VENDOR_UNKNOWN;
    }
}
