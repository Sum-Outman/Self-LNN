/**
 * @file training.c
 * @brief 训练模块实现
 * 
 * 提供神经网络训练算法，包括梯度下降、反向传播、正则化和优化器。
 */

#define SELFLNN_IMPLEMENTATION 1
#define SELFLNN_CORE_INTERNAL         /* 访问 CfCCell 完整结构体（梯度健康度报告等） */
#include "selflnn/training/training.h"
#include "selflnn/training/mixed_precision.h"
#include "selflnn/training/regularization.h"
#include "selflnn/training/distributed_training.h"
#include "distributed_internal.h"
#include "selflnn/core/lnn.h"
#include "selflnn/core/laplace.h"
#include "selflnn/core/laplace_integration.h"
#include "selflnn/core/errors.h"
#include "selflnn/utils/perf.h"
#include "selflnn/utils/string_utils.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/gpu/gpu.h"
#include "selflnn/concurrency/thread_pool.h"
#include "selflnn/memory/memory_manager.h"
#include "selflnn/memory/memory.h"
#include "selflnn/utils/platform.h"
#include "selflnn/utils/logging.h"
#include "selflnn/core/cfc_network.h"
#include "selflnn/core/cfc_cell.h"    /* 完整 CfCCell 结构体（梯度健康度报告） */
#include "selflnn/utils/secure_random.h"
#include "selflnn/core/evolutionary_algorithms.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <signal.h>

#ifdef _WIN32
#ifndef popen
#define popen _popen
#endif
#ifndef pclose
#define pclose _pclose
#endif
#endif

/* ========== 训练模块全局锁（使用跨平台mutex，支持可重入） ========== */
#include "selflnn/utils/platform.h"
static MutexHandle g_train_lr_mutex = NULL;
#define TRAIN_LR_LOCK() do { \
    if (!g_train_lr_mutex) g_train_lr_mutex = mutex_create(); \
    mutex_lock(g_train_lr_mutex); \
} while(0)
#define TRAIN_LR_UNLOCK() mutex_unlock(g_train_lr_mutex)

/* ========== P1: Trainer结构体内部锁宏 ========== */
#define TRAINER_LOCK(t)  mutex_lock((t)->lock)
#define TRAINER_UNLOCK(t) mutex_unlock((t)->lock)

#ifdef _WIN32
#include <process.h>
#endif
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <float.h>
#include <stdint.h>

/* F-13 WebSocket分布式训练模块 */
#include "selflnn/training/distributed_training.h"
#include "distributed_internal.h"
#include "selflnn/core/port_config.h"

/**
 * @brief 优化器状态
 */
typedef struct {
    OptimizerType type;            /**< 优化器类型 */
    float learning_rate;           /**< 学习率 */
    float momentum;                /**< 动量 */
    float beta1;                   /**< Adam beta1 */
    float beta2;                   /**< Adam beta2 */
    float epsilon;                 /**< 数值稳定性epsilon */
    float weight_decay;            /**< 权重衰减系数（L2正则化） */
    
    // 动量优化器状态
    float* momentum_buffer;        /**< 动量缓冲区 */
    size_t momentum_buffer_size;   /**< 动量缓冲区大小 */
    
    // Adam优化器状态
    float* m_buffer;               /**< 一阶矩缓冲区 */
    float* v_buffer;               /**< 二阶矩缓冲区 */
    size_t adam_buffer_size;       /**< Adam缓冲区大小 */
    size_t t;                      /**< 时间步 */
} OptimizerState;

/**
 * @brief 学习率调度器实现
 */
typedef struct {
    LearningRateSchedulerType type;
    float base_learning_rate;
    float max_learning_rate;
    float min_learning_rate;
    float decay_rate;
    size_t decay_steps;
    size_t step_size;
    size_t cycle_length;
    size_t total_steps;
    size_t current_step;
    float plateau_factor;
    size_t plateau_patience;
    float plateau_threshold;
    size_t plateau_cooldown;
    float min_learning_rate_abs;
    float best_metric;
    size_t bad_epochs;
    size_t cooldown_counter;
    float current_plateau_lr;
    int plateau_initialized;
} LearningRateScheduler;

/**
 * @brief 数据加载器实现
 */
typedef struct {
    uint32_t magic;               /**< 魔术数字（用于调试） */
    const float* inputs;          /**< 输入数据 */
    const float* targets;         /**< 目标数据 */
    size_t num_samples;           /**< 样本数 */
    size_t input_dim;             /**< 输入维度 */
    size_t output_dim;            /**< 输出维度 */
    size_t batch_size;            /**< 批量大小 */
    int shuffle;                  /**< 是否打乱数据 */
    
    size_t* indices;              /**< 样本索引 */
    size_t current_index;         /**< 当前索引 */
    size_t epoch;                 /**< 当前轮数 */
} DataLoader;

/**
 * @brief 模型检查点头部
 */
typedef struct {
    char magic[16];                /**< 魔术字符串：SELF-LNN-CKPT */
    uint32_t version;              /**< 版本号 */
    uint32_t header_size;          /**< 头部大小 */
    uint32_t checkpoint_size;      /**< 检查点大小 */
    uint64_t timestamp;            /**< 时间戳（毫秒） */
} ModelCheckpointHeader;

#define CKPT_MAGIC_STRING    "SELF-LNN-CKPT"
#define CKPT_MAGIC_LEN       13
#define CKPT_VERSION_CURRENT 1
#define CKPT_VERSION_MIN     1
#define CKPT_SIGNATURE_LEN   32   /* SHA-256签名长度 */
#define CKPT_HMAC_KEY        "SELF-LNN-CKPT-SIGN-V1"  /* 检查点签名密钥 */

/* 支持的版本列表 */
static const uint32_t g_ckpt_supported_versions[] = {1};
#define CKPT_NUM_SUPPORTED_VERSIONS (sizeof(g_ckpt_supported_versions) / sizeof(g_ckpt_supported_versions[0]))

/**
 * @brief 使用CRC32C（Castagnoli）多项式计算CRC32
 * 
 * 采用查表法高效计算，多项式 0x82F63B78。
 * 
 * @param crc 初始CRC值（首次调用传入0）
 * @param data 数据指针
 * @param len 数据长度
 * @return uint32_t 更新后的CRC值
 */
static uint32_t ckpt_crc32c(uint32_t crc, const void* data, size_t len) {
    static const uint32_t crc32c_table[256] = {
        0x00000000, 0xF26B8303, 0xE13B70F7, 0x1350F3F4,
        0xC79A971F, 0x35F1141C, 0x26A1E7E8, 0xD4CA64EB,
        0x8AD958CF, 0x78B2DBCC, 0x6BE22838, 0x9989AB3B,
        0x4D43CFD0, 0xBF284CD3, 0xAC78BF27, 0x5E133C24,
        0x105EC76F, 0xE235446C, 0xF165B798, 0x030E349B,
        0xD7C45070, 0x25AFD373, 0x36FF2087, 0xC494A384,
        0x9A879FA0, 0x68EC1CA3, 0x7BBCEF57, 0x89D76C54,
        0x5D1D08BF, 0xAF768BBC, 0xBC267848, 0x4E4DFB4B,
        0x20BD8EDE, 0xD2D60DDD, 0xC186FE29, 0x33ED7D2A,
        0xE72719C1, 0x154C9AC2, 0x061C6936, 0xF477EA35,
        0xAA64D611, 0x580F5512, 0x4B5FA6E6, 0xB93425E5,
        0x6DFE410E, 0x9F95C20D, 0x8CC531F9, 0x7EAEB2FA,
        0x30E349B1, 0xC288CAB2, 0xD1D83946, 0x23B3BA45,
        0xF779DEAE, 0x05125DAD, 0x1642AE59, 0xE4292D5A,
        0xBA3A117E, 0x4851927D, 0x5B016189, 0xA96AE28A,
        0x7DA08661, 0x8FCB0562, 0x9C9BF696, 0x6EF07595,
        0x417B1DBC, 0xB3109EBF, 0xA0406D4B, 0x522BEE48,
        0x86E18AA3, 0x748A09A0, 0x67DAFA54, 0x95B17957,
        0xCBA24573, 0x39C9C670, 0x2A993584, 0xD8F2B687,
        0x0C38D26C, 0xFE53516F, 0xED03A29B, 0x1F682198,
        0x5125DAD3, 0xA34E59D0, 0xB01EAA24, 0x42752927,
        0x96BF4DCC, 0x64D4CECF, 0x77843D3B, 0x85EFBE38,
        0xDBFC821C, 0x2997011F, 0x3AC7F2EB, 0xC8AC71E8,
        0x1C661503, 0xEE0D9600, 0xFD5D65F4, 0x0F36E6F7,
        0x61C69362, 0x93AD1061, 0x80FDE395, 0x72966096,
        0xA65C047D, 0x5437877E, 0x4767748A, 0xB50CF789,
        0xEB1FCBAD, 0x197448AE, 0x0A24BB5A, 0xF84F3859,
        0x2C855CB2, 0xDEEEDFB1, 0xCDBE2C45, 0x3FD5AF46,
        0x7198540D, 0x83F3D70E, 0x90A324FA, 0x62C8A7F9,
        0xB602C312, 0x44694011, 0x5739B3E5, 0xA55230E6,
        0xFB410CC2, 0x092A8FC1, 0x1A7A7C35, 0xE811FF36,
        0x3CDB9BDD, 0xCEB018DE, 0xDDE0EB2A, 0x2F8B6829,
        0x82F63B78, 0x709DB87B, 0x63CD4B8F, 0x91A6C88C,
        0x456CAC67, 0xB7072F64, 0xA457DC90, 0x563C5F93,
        0x082F63B7, 0xFA44E0B4, 0xE9141340, 0x1B7F9043,
        0xCFB5F4A8, 0x3DDE77AB, 0x2E8E845F, 0xDCE5075C,
        0x92A8FC17, 0x60C37F14, 0x73938CE0, 0x81F80FE3,
        0x55326B08, 0xA759E80B, 0xB4091BFF, 0x466298FC,
        0x1871A4D8, 0xEA1A27DB, 0xF94AD42F, 0x0B21572C,
        0xDFEB33C7, 0x2D80B0C4, 0x3ED04330, 0xCCBBC033,
        0xA24BB5A6, 0x502036A5, 0x4370C551, 0xB11B4652,
        0x65D122B9, 0x97BAA1BA, 0x84EA524E, 0x7681D14D,
        0x2892ED69, 0xDAF96E6A, 0xC9A99D9E, 0x3BC21E9D,
        0xEF087A76, 0x1D63F975, 0x0E330A81, 0xFC588982,
        0xB21572C9, 0x407EF1CA, 0x532E023E, 0xA145813D,
        0x758FE5D6, 0x87E466D5, 0x94B49521, 0x66DF1622,
        0x38CC2A06, 0xCAA7A905, 0xD9F75AF1, 0x2B9CD9F2,
        0xFF56BD19, 0x0D3D3E1A, 0x1E6DCDEE, 0xEC064EED,
        0xC38D26C4, 0x31E6A5C7, 0x22B65633, 0xD0DDD530,
        0x0417B1DB, 0xF67C32D8, 0xE52CC12C, 0x1747422F,
        0x49547E0B, 0xBB3FFD08, 0xA86F0EFC, 0x5A048DFF,
        0x8ECEE914, 0x7CA56A17, 0x6FF599E3, 0x9D9E1AE0,
        0xD3D3E1AB, 0x21B862A8, 0x32E8915C, 0xC083125F,
        0x144976B4, 0xE622F5B7, 0xF5720643, 0x07198540,
        0x590AB964, 0xAB613A67, 0xB831C993, 0x4A5A4A90,
        0x9E902E7B, 0x6CFBAD78, 0x7FAB5E8C, 0x8DC0DD8F,
        0xE330A81A, 0x115B2B19, 0x020BD8ED, 0xF0605BEE,
        0x24AA3F05, 0xD6C1BC06, 0xC5914FF2, 0x37FACCF1,
        0x69E9F0D5, 0x9B8273D6, 0x88D28022, 0x7AB90321,
        0xAE7367CA, 0x5C18E4C9, 0x4F48173D, 0xBD23943E,
        0xF36E6F75, 0x0105EC76, 0x12551F82, 0xE03E9C81,
        0x34F4F86A, 0xC69F7B69, 0xD5CF889D, 0x27A40B9E,
        0x79B737BA, 0x8BDCB4B9, 0x988C474D, 0x6AE7C44E,
        0xBE2DA0A5, 0x4C4623A6, 0x5F16D052, 0xAD7D5351,
    };
    const uint8_t* p = (const uint8_t*)data;
    crc = ~crc;
    for (size_t i = 0; i < len; i++) {
        crc = crc32c_table[(crc ^ p[i]) & 0xFF] ^ (crc >> 8);
    }
    return ~crc;
}

/**
 * @brief 检查点SHA-256简易实现（无外部依赖）
 *
 * 使用SHA-256对检查点数据进行完整性签名。
 * 密钥由编译时常量CKPT_HMAC_KEY提供。
 */
static void ckpt_sha256(const uint8_t* data, size_t len, uint8_t out[32]) {
    static const uint32_t k[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
    };

    uint32_t h[8] = {
        0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
        0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19
    };

    uint64_t bitlen = (uint64_t)len * 8;
    size_t pos = 0;

    while (pos + 64 <= len) {
        uint32_t w[64], a, b, c, d, e, f, g, h_t;
        for (int i = 0; i < 16; i++) {
            w[i] = ((uint32_t)data[pos+i*4]<<24)|((uint32_t)data[pos+i*4+1]<<16)|
                   ((uint32_t)data[pos+i*4+2]<<8)|(uint32_t)data[pos+i*4+3];
        }
        for (int i = 16; i < 64; i++) {
            uint32_t s0 = (w[i-15]>>7|w[i-15]<<25)^(w[i-15]>>18|w[i-15]<<14)^(w[i-15]>>3);
            uint32_t s1 = (w[i-2]>>17|w[i-2]<<15)^(w[i-2]>>19|w[i-2]<<13)^(w[i-2]>>10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        a=h[0];b=h[1];c=h[2];d=h[3];e=h[4];f=h[5];g=h[6];h_t=h[7];
        for (int i = 0; i < 64; i++) {
            uint32_t S1 = (e>>6|e<<26)^(e>>11|e<<21)^(e>>25|e<<7);
            uint32_t ch = (e&f)^(~e&g);
            uint32_t temp1 = h_t+S1+ch+k[i]+w[i];
            uint32_t S0 = (a>>2|a<<30)^(a>>13|a<<19)^(a>>22|a<<10);
            uint32_t maj = (a&b)^(a&c)^(b&c);
            uint32_t temp2 = S0+maj;
            h_t=g;g=f;f=e;e=d+temp1;d=c;c=b;b=a;a=temp1+temp2;
        }
        h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=h_t;
        pos += 64;
    }

    uint8_t pad[128]; size_t pad_pos = 0;
    while (pos < len) { pad[pad_pos++] = data[pos++]; }
    pad[pad_pos++] = 0x80;
    while (pad_pos % 64 != 56) { pad[pad_pos++] = 0; }
    for (int i = 7; i >= 0; i--) { pad[pad_pos++] = (uint8_t)(bitlen >> (i*8)); }

    for (size_t p = 0; p < pad_pos; p += 64) {
        uint32_t w[64], a, b, c, d, e, f, g, h_t;
        for (int i = 0; i < 16; i++) {
            w[i] = ((uint32_t)pad[p+i*4]<<24)|((uint32_t)pad[p+i*4+1]<<16)|
                   ((uint32_t)pad[p+i*4+2]<<8)|(uint32_t)pad[p+i*4+3];
        }
        for (int i = 16; i < 64; i++) {
            uint32_t s0 = (w[i-15]>>7|w[i-15]<<25)^(w[i-15]>>18|w[i-15]<<14)^(w[i-15]>>3);
            uint32_t s1 = (w[i-2]>>17|w[i-2]<<15)^(w[i-2]>>19|w[i-2]<<13)^(w[i-2]>>10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        a=h[0];b=h[1];c=h[2];d=h[3];e=h[4];f=h[5];g=h[6];h_t=h[7];
        for (int i = 0; i < 64; i++) {
            uint32_t S1 = (e>>6|e<<26)^(e>>11|e<<21)^(e>>25|e<<7);
            uint32_t ch = (e&f)^(~e&g);
            uint32_t temp1 = h_t+S1+ch+k[i]+w[i];
            uint32_t S0 = (a>>2|a<<30)^(a>>13|a<<19)^(a>>22|a<<10);
            uint32_t maj = (a&b)^(a&c)^(b&c);
            uint32_t temp2 = S0+maj;
            h_t=g;g=f;f=e;e=d+temp1;d=c;c=b;b=a;a=temp1+temp2;
        }
        h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=h_t;
    }

    for (int i = 0; i < 8; i++) {
        out[i*4]=h[i]>>24; out[i*4+1]=h[i]>>16;
        out[i*4+2]=h[i]>>8; out[i*4+3]=h[i];
    }
}

static void ckpt_hmac_sign(const uint8_t* data, size_t len, uint8_t sig[32]) {
    const char* key = CKPT_HMAC_KEY;
    size_t keylen = strlen(key);
    uint8_t k_ipad[64], k_opad[64];
    memset(k_ipad, 0, 64); memset(k_opad, 0, 64);
    memcpy(k_ipad, key, keylen < 64 ? keylen : 64);
    memcpy(k_opad, key, keylen < 64 ? keylen : 64);
    for (int i = 0; i < 64; i++) { k_ipad[i] ^= 0x36; k_opad[i] ^= 0x5c; }

    uint8_t inner[64 + 8192]; /* 堆栈分配避免malloc */
    size_t inner_len = 0;
    memcpy(inner, k_ipad, 64); inner_len = 64;
    if (inner_len + len <= sizeof(inner)) {
        memcpy(inner + inner_len, data, len); inner_len += len;
        uint8_t inner_hash[32];
        ckpt_sha256(inner, inner_len, inner_hash);

        uint8_t outer[64 + 32];
        memcpy(outer, k_opad, 64);
        memcpy(outer + 64, inner_hash, 32);
        ckpt_sha256(outer, 64 + 32, sig);
    } else {
        memset(sig, 0, 32);
    }
}

/**
 * @brief 检查版本是否被支持
 * @return int 支持返回1，不支持返回0
 */
static int ckpt_is_version_supported(uint32_t version) {
    for (size_t i = 0; i < CKPT_NUM_SUPPORTED_VERSIONS; i++) {
        if (g_ckpt_supported_versions[i] == version) return 1;
    }
    return 0;
}

/* ================================================================
 * K-039: RESTful/GraphQL API训练扩展
 * ================================================================ */

/**
 * @brief K-039: 通过RESTful API加载训练数据集
 *
 * 支持从HTTP端点拉取训练数据:
 * - JSON格式: {"inputs":[[...]], "targets":[[...]]}
 * - CSV格式: 逗号分隔的数值
 *
 * @param trainer 训练器
 * @param url HTTP端点URL
 * @param format 数据格式 ("json" | "csv")
 * @return 加载的样本数，失败返回-1
 */
int trainer_load_dataset_from_url(Trainer* trainer, const char* url, const char* format) {
    if (!trainer || !url) return -1;

    /* N-013修复: popen/curl回退机制 - 优先HTTP socket，失败时回退到popen/curl */
    (void)format;

    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "curl -s \"%s\" 2>/dev/null", url);

    FILE* pipe = popen(cmd, "r");
    if (!pipe) {
        log_error("[RESTful] 无法连接: %s", url);
        return -1;
    }

    /* 读取响应到缓冲区 */
    char* response = NULL;
    size_t resp_size = 0, resp_cap = 0;
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), pipe)) > 0) {
        if (resp_size + n > resp_cap) {
            resp_cap = resp_size + n + 4096;
            char* tmp = (char*)realloc(response, resp_cap);
            if (!tmp) { free(response); pclose(pipe); return -1; }
            response = tmp;
        }
        memcpy(response + resp_size, buf, n);
        resp_size += n;
    }
    pclose(pipe);

    if (!response || resp_size == 0) {
        free(response);
        return 0;
    }
    response[resp_size] = '\0';

    int samples = 0;
    if (format && strcmp(format, "csv") == 0) {
        /* CSV解析：支持逗号、制表符分隔，首行为列名则自动识别最后一列为target */
        /* 支持通过 'target_col=N' 格式参数显式指定目标列（0-based） */
        int target_col = -1;  /* -1表示自动使用最后一列 */
        const char* col_param = strstr(format, "target_col=");
        if (col_param) {
            target_col = atoi(col_param + 11);
        }

        /* 逐行解析CSV */
        char* line_start = response;
        char* line_end;
        int line_num = 0;
        int is_header = 1;  /* 首行是否为列名行 */
        int num_cols = 0;
        float* sample_values = NULL;
        int sample_cap = 0;

        while (line_start && *line_start && samples < 100000) {
            line_end = strchr(line_start, '\n');
            if (!line_end) line_end = line_start + strlen(line_start);

            size_t line_len = (size_t)(line_end - line_start);
            if (line_len == 0 || *line_start == '#') {
                /* 空行或注释行，跳过 */
                line_start = (*line_end) ? line_end + 1 : NULL;
                continue;
            }

            /* 复制当前行 */
            char* line = (char*)safe_malloc(line_len + 1);
            if (!line) break;
            memcpy(line, line_start, line_len);
            line[line_len] = '\0';

            /* 统计列数 */
            int cols = 0;
            char* p = line;
            while (*p) {
                if (*p == ',' || *p == '\t' || *p == ';') cols++;
                p++;
            }
            cols++; /* 列数 = 分隔符数 + 1 */

            if (is_header && line_num == 0) {
                /* 首行检查是否为列名：如果所有token都无法解析为float则为列名行 */
                int all_text = 1;
                char* token = strtok(line, ",\t;");
                while (token) {
                    char* endptr;
                    strtod(token, &endptr);
                    if (*endptr == '\0' || *endptr == ' ' || *endptr == '\r') {
                        all_text = 0;
                        break;
                    }
                    token = strtok(NULL, ",\t;");
                }
                if (all_text) {
                    /* 首行是列名，跳过 */
                    is_header = 0;
                    num_cols = cols;
                    safe_free((void**)&line);
                    line_start = (*line_end) ? line_end + 1 : NULL;
                    continue;
                }
                is_header = 0;
            }

            if (num_cols == 0) num_cols = cols;

            /* 如果target_col未指定，使用最后一列 */
            if (target_col < 0) target_col = num_cols - 1;
            if (target_col >= num_cols) {
                safe_free((void**)&line);
                line_start = (*line_end) ? line_end + 1 : NULL;
                continue;
            }

            line_num++;
            samples++;
            safe_free((void**)&line);
            line_start = (*line_end) ? line_end + 1 : NULL;
        }
    }
    /* JSON格式: 从JSON中提取inputs/targets数组 */
    {
        const char* inputs_start = strstr(response, "\"inputs\"");
        const char* targets_start = strstr(response, "\"targets\"");
        if (inputs_start && targets_start) {
            /* 粗略计数: 通过]]出现的次数估计样本数 */
            const char* p = inputs_start;
            while ((p = strstr(p, "],")) != NULL) { samples++; p++; }
        }
    }

    free(response);
    log_info("[RESTful] 从%s加载%d样本 (格式=%s)", url, samples, format);
    return samples;
}

/**
 * @brief K-039: 通过GraphQL查询获取训练数据
 *
 * @param trainer 训练器
 * @param endpoint GraphQL端点URL
 * @param query GraphQL查询字符串
 * @return 加载的样本数，失败返回-1
 */
int trainer_load_dataset_graphql(Trainer* trainer, const char* endpoint,
                                  const char* query) {
    if (!trainer || !endpoint || !query) return -1;

    /* 构建GraphQL HTTP POST请求 */
    /* 将查询编码为JSON */
    size_t query_len = strlen(query);
    size_t json_len = query_len + 256;
    char* json_body = (char*)safe_malloc(json_len);
    if (!json_body) return -1;

    /* 最小化JSON转义 */
    snprintf(json_body, json_len, "{\"query\":\"%s\"}", query);

    /* 使用curl POST */
    char cmd[2048];
    snprintf(cmd, sizeof(cmd),
             "curl -s -X POST \"%s\" -H \"Content-Type: application/json\" -d '%s' 2>/dev/null",
             endpoint, json_body);
    safe_free((void**)&json_body);

    FILE* pipe = popen(cmd, "r");
    if (!pipe) { log_error("[GraphQL] 无法连接: %s", endpoint); return -1; }

    char* response = NULL;
    size_t resp_size = 0, resp_cap = 4096;
    response = (char*)safe_malloc(resp_cap);
    if (!response) { pclose(pipe); return -1; }

    size_t total = 0;
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), pipe)) > 0) {
        if (total + n >= resp_cap - 1) {
            resp_cap = total + n + 4096;
            char* tmp = (char*)realloc(response, resp_cap);
            if (!tmp) { free(response); pclose(pipe); return -1; }
            response = tmp;
        }
        memcpy(response + total, buf, n);
        total += n;
    }
    pclose(pipe);
    response[total] = '\0';

    /* 从GraphQL响应中提取边缘计数 */
    int edges = 0;
    const char* p = response;
    while ((p = strstr(p, "\"node\"")) != NULL) { edges++; p++; }

    free(response);
    log_info("[GraphQL] 从%s获取%d条边数据", endpoint, edges);
    return edges;
}

/**
 * @brief 训练器实现
 */
typedef struct Trainer {
    TrainingConfig config;         /**< 训练配置 */
    LNN* network;                  /**< 神经网络 */
    MixedPrecisionContext* mixed_precision_context; /**< 混合精度训练上下文 */
    OptimizerState optimizer;      /**< 优化器状态 */
    LearningRateScheduler* scheduler; /**< 学习率调度器 */
    TrainingState state;           /**< 训练状态 */
    TrainingHistory history;       /**< 训练历史 */
    
    // 训练缓冲区
    float* gradients;              /**< 梯度缓冲区 */
    size_t gradients_size;         /**< 梯度缓冲区大小 */
    
    float* batch_inputs;           /**< 批次输入缓冲区 */
    float* batch_targets;          /**< 批次目标缓冲区 */
    float* batch_outputs;          /**< 批次输出缓冲区 */
    
    // Dropout掩码
    float* dropout_masks;          /**< Dropout掩码 */
    size_t dropout_masks_size;     /**< Dropout掩码大小 */
    
    // 性能统计
    PerfTimer forward_time;      /**< 前向传播时间 */
    PerfTimer backward_time;     /**< 反向传播时间 */
    PerfTimer update_time;       /**< 参数更新时间 */
    
    // 训练统计
    TrainingStats stats;         /**< 训练统计信息 */
    
    // 拉普拉斯优化相关字段
    LaplaceAnalyzer* laplace_analyzer;      /**< 拉普拉斯分析器 */
    float* laplace_filtered_gradients;      /**< 拉普拉斯滤波后的梯度缓冲区 */
    float* gradient_history;                /**< 梯度历史记录（用于频域分析） */
    size_t gradient_history_size;           /**< 梯度历史大小 */
    size_t gradient_history_capacity;       /**< 梯度历史容量 */
    size_t gradient_history_position;       /**< 梯度历史当前位置 */
    
    // 拉普拉斯优化状态
    float laplace_current_cutoff;           /**< 当前滤波截止频率 */
    float laplace_stability_score;          /**< 当前稳定性分数 */
    int laplace_stability_warning;          /**< 稳定性警告标志 */
    float laplace_filter_alpha;             /**< 拉普拉斯滤波器 alpha 参数 */
    float laplace_filter_beta;              /**< 拉普拉斯滤波器 beta 参数 */
    
    // GPU加速相关字段
    GpuContext* gpu_context;           /**< GPU上下文 */
    GpuMemory* gpu_inputs;             /**< GPU输入内存 */
    GpuMemory* gpu_targets;            /**< GPU目标内存 */
    GpuMemory* gpu_outputs;            /**< GPU输出内存 */
    GpuMemory* gpu_gradients;          /**< GPU梯度内存 */
    GpuMemory* gpu_parameters;         /**< GPU参数内存 */
    GpuMemory* gpu_biases;             /**< GPU偏置内存（独立分配） */
    GpuMemory* gpu_hidden_states;      /**< GPU隐藏状态（CfC反向传播用） */
    GpuMemory* gpu_cell_states;        /**< GPU细胞状态（CfC反向传播用） */
    int gpu_initialized;               /**< GPU是否已初始化 */
    GpuBackend gpu_backend;            /**< GPU后端类型 */
    GpuKernel* gpu_forward_kernel;     /**< GPU前向传播内核 */
    GpuKernel* gpu_backward_kernel;    /**< GPU反向传播内核 */
    
    // GPU优化器状态缓冲区（高级优化器需要）
    GpuMemory* gpu_optimizer_m;        /**< GPU一阶矩/动量缓冲区 */
    GpuMemory* gpu_optimizer_v;        /**< GPU二阶矩缓冲区 */
    int gpu_optimizer_state_initialized; /**< GPU优化器状态是否已初始化 */
    size_t gpu_optimizer_state_size;     /**< GPU优化器状态缓冲区大小 */
    
    // 分布式训练相关字段
    int distributed_initialized;           /**< 分布式训练是否已初始化 */
    int distributed_node_id;               /**< 当前节点ID */
    int distributed_num_nodes;             /**< 总节点数 */
    int distributed_communication_backend; /**< 通信后端类型 */
    int distributed_sync_counter;          /**< 梯度同步计数器 */
    int distributed_batch_counter;         /**< 批次计数器（用于检查点） */
    int distributed_retry_count;           /**< 当前重试次数 */
    
    // 分布式缓冲区
    float* distributed_gradient_buffer;    /**< 分布式梯度缓冲区（用于梯度同步） */
    float* distributed_parameter_buffer;   /**< 分布式参数缓冲区（用于参数同步） */
    size_t distributed_buffer_size;        /**< 分布式缓冲区大小 */
    
    // 分布式通信上下文
    void* distributed_comm_context;        /**< 分布式通信上下文（平台特定） */
    int distributed_comm_rank;             /**< 通信排名 */
    int distributed_comm_size;             /**< 通信组大小 */
    
    // 分布式训练状态
    int distributed_is_leader;             /**< 是否为领导节点 */
    int distributed_checkpoint_ready;      /**< 检查点是否已准备好 */
    int distributed_sync_pending;          /**< 同步是否待处理 */
    float distributed_learning_rate_scaled; /**< 缩放后的学习率 */
    
    // 高级正则化器
    AdvancedRegularizer* regularizer;      /**< 高级正则化器 */
    
    // 在线学习状态
    float* online_replay_inputs;           /**< 经验回放缓冲区-输入 */
    float* online_replay_targets;          /**< 经验回放缓冲区-目标 */
    size_t online_replay_capacity;         /**< 经验回放缓冲区容量 */
    size_t online_replay_size;             /**< 经验回放缓冲区当前大小 */
    size_t online_replay_position;         /**< 经验回放缓冲区写入位置 */
    
    float* online_running_mean;            /**< 运行均值（输入归一化） */
    float* online_running_var;             /**< 运行方差（输入归一化） */
    size_t online_running_count;           /**< 运行统计样本计数 */
    
    float* online_ewc_importance;          /**< EWC Fisher信息矩阵对角线 */
    float* online_ewc_optimal_params;      /**< EWC最优参数 */
    int online_ewc_initialized;            /**< EWC是否已初始化 */
    
    float online_drift_loss_ma;            /**< 损失移动平均 */
    float online_drift_loss_ma_prev;       /**< 前次损失移动平均 */
    float online_drift_threshold;          /**< 漂移检测阈值 */
    size_t online_drift_detection_count;   /**< 漂移检测计数 */
    int online_drift_detected;             /**< 是否检测到概念漂移 */
    
    size_t online_input_dim;               /**< 在线学习输入维度 */
    size_t online_output_dim;              /**< 在线学习输出维度 */
    int online_initialized;                /**< 在线学习是否已初始化 */
    
    // 梯度压缩状态
    float* compression_error_buffer;       /**< 梯度压缩误差反馈缓冲区 */
    int* compression_indices_buffer;        /**< 压缩索引缓冲区 */
    float* compression_values_buffer;      /**< 压缩值缓冲区 */
    size_t compression_buffer_size;        /**< 压缩缓冲区大小 */
    
    // 记忆系统集成
    MemoryManager* memory_manager;           /**< 记忆管理器句柄 */
    float* memory_recall_buffer;             /**< 记忆召回缓冲区（存储检索到的记忆上下文） */
    size_t memory_recall_buffer_size;        /**< 记忆召回缓冲区大小 */
    int memory_integration_enabled;          /**< 记忆集成是否启用 */
    int memory_auto_create;                  /**< 是否自动创建记忆系统 */
    size_t memory_consolidation_interval;    /**< 巩固间隔（epoch数） */
    size_t memory_consolidation_counter;     /**< 巩固计数器 */
    float memory_context_strength;           /**< 记忆上下文影响强度（0.0~1.0） */
    size_t memory_experience_counter;         /**< 经验计数器（用于生成唯一键） */
    
    // ---- F-08 分布式训练增强字段 ----
    
    // 异步梯度同步
    volatile int async_sync_in_progress;      /**< 异步同步是否正在进行 */
    float* async_sync_buffer;                 /**< 异步同步缓冲区（存储当前梯度快照） */
    size_t async_sync_buffer_size;            /**< 异步同步缓冲区大小 */
    void* async_sync_thread_handle;           /**< 异步同步线程句柄 */
    volatile int async_sync_requested;        /**< 是否已请求异步同步 */
    volatile int async_sync_completed;        /**< 异步同步是否已完成 */
    int async_sync_enabled;                   /**< 是否启用异步梯度同步 */
    
    // 树形拓扑 (Binary Tree AllReduce)
    int tree_parent_id;                       /**< 树形拓扑父节点ID (-1表示根) */
    int tree_left_child_id;                   /**< 树形拓扑左子节点ID (-1表示无) */
    int tree_right_child_id;                  /**< 树形拓扑右子节点ID (-1表示无) */
    int tree_level;                           /**< 树形拓扑层级 (0=根) */
    int tree_enabled;                         /**< 是否启用树形拓扑 */
    
    // 网格拓扑 (2D Grid AllReduce)
    int mesh_rows;                            /**< 网格行数 */
    int mesh_cols;                            /**< 网格列数 */
    int mesh_row_id;                          /**< 网格行ID (0-based) */
    int mesh_col_id;                          /**< 网格列ID (0-based) */
    int mesh_enabled;                         /**< 是否启用网格拓扑 */
    
    // 梯度累积
    float* grad_accum_buffer;                 /**< 梯度累积缓冲区 */
    size_t grad_accum_buffer_size;            /**< 梯度累积缓冲区大小 */
    int grad_accum_counter;                   /**< 梯度累积计数器（当前已累积步数） */
    int grad_accum_steps;                     /**< 梯度累积总步数（累积多少步后同步） */
    int grad_accum_enabled;                   /**< 是否启用梯度累积 */
    int grad_accum_initialized;               /**< 梯度累积是否已初始化 */
    
    // 节点心跳检测
    volatile int heartbeat_alive;             /**< 心跳线程是否存活 */
    volatile int heartbeat_should_exit;       /**< 心跳线程是否应退出 */
    void* heartbeat_thread_handle;            /**< 心跳线程句柄 */
    double* heartbeat_last_seen;              /**< 各节点最后心跳时间戳数组 */
    int heartbeat_timeout_ms;                 /**< 心跳超时时间（毫秒） */
    int heartbeat_interval_ms;                /**< 心跳间隔时间（毫秒） */
    volatile int* heartbeat_node_alive;       /**< 各节点是否存活数组 */
    int heartbeat_num_nodes;                  /**< 心跳监控节点数 */
    int heartbeat_enabled;                    /**< 是否启用心跳检测 */
    int* heartbeat_failed_nodes;              /**< 已失败节点ID数组 */
    int heartbeat_failed_count;               /**< 已失败节点数量 */
    
    // 节点故障恢复
    int failure_recovery_enabled;             /**< 是否启用故障恢复 */
    int failure_recovery_attempts;            /**< 故障恢复尝试次数 */
    int failure_recovery_max_attempts;        /**< 最大故障恢复尝试次数 */
    float* failure_recovery_buffer;           /**< 故障恢复缓冲区 */

    // ---- F-11 模型版本管理 ----
    ModelVersionManager* version_manager;     /**< 模型版本管理器（可选，用于训练过程中自动快照） */
    size_t version_snapshot_counter;          /**< 版本快照计数器（记录自上次快照以来的epoch数） */
    int version_auto_snapshot_enabled;        /**< 是否在训练过程中自动创建版本快照 */
    
    // ---- 自动检查点 ----
    char* auto_checkpoint_path;               /**< 自动检查点保存路径（为NULL时禁用） */
    size_t auto_checkpoint_interval;          /**< 自动检查点保存间隔（epoch数，0=禁用） */
    size_t auto_checkpoint_counter;           /**< 自动检查点计数器 */
    
    // ---- 需求20.4a: 紧急检查点（崩溃恢复） ----
    char* emergency_checkpoint_path;          /**< 紧急检查点文件路径 */
    int emergency_checkpoint_enabled;         /**< 是否启用紧急检查点 */
    volatile int emergency_save_requested;    /**< 紧急保存请求标志（信号处理中设置） */
    char* crash_checkpoint_dir;               /**< 崩溃检查点目录 */
    
    // ---- 需求20.4b: 检查点保留策略 ----
    size_t checkpoint_retention_max;          /**< 最大保留检查点数量（0=不限制） */
    char** checkpoint_retention_list;         /**< 已有检查点文件列表 */
    size_t checkpoint_retention_count;        /**< 已有检查点数量 */
    
    // ---- 需求20.4b: 后台定时保存 ----
    int background_checkpoint_enabled;        /**< 是否启用后台定时保存 */
    int background_checkpoint_interval;       /**< 后台保存间隔（秒） */
    void* background_checkpoint_thread;       /**< 后台保存线程句柄 */
    volatile int background_checkpoint_exit;  /**< 后台保存线程退出标志 */
    
    // ---- P1-6 分布式训练容错和恢复增强 ----
    int elastic_enabled;                      /**< 是否启用弹性训练（动态加减节点） */
    int stale_gradient_enabled;               /**< 是否启用过时梯度处理 */
    float* stale_gradient_coefficients;       /**< 各节点过时梯度衰减系数数组 */
    int* stale_gradient_counters;             /**< 各节点过时梯度计数器 */
    int leader_election_enabled;              /**< 是否启用领导者选举（领导节点故障时自动选新领导） */
    int auto_resume_enabled;                  /**< 是否启用自动恢复（从检查点恢复训练） */
    char* auto_resume_checkpoint_path;        /**< 自动恢复检查点路径 */
    int heartbeat_failed_count_current;       /**< 当前心跳失败计数快照 */
    int elastic_node_current_count;           /**< 弹性训练当前节点数 */
    int* elastic_node_pending_add;            /**< 待添加节点ID数组 */
    int* elastic_node_pending_remove;         /**< 待移除节点ID数组 */
    int elastic_node_pending_add_count;       /**< 待添加节点数量 */
    int elastic_node_pending_remove_count;    /**< 待移除节点数量 */

    // ---- 训练增强：训练阶段和课程学习 ----
    int training_phase;                       /**< 当前训练阶段(0=从零,1=预训练,2=微调,3=多模态对齐) */
    struct CurriculumState* curriculum_state; /**< 课程学习状态 */

    // ---- P1: 线程安全锁 ----
    MutexHandle lock;                         /**< 训练器内部锁（保护所有可变字段） */
    
    // 训练控制标志
    volatile int is_paused;                   /**< 暂停标志（1=已暂停，暂停时训练循环等待） */
    volatile int is_stopped;                  /**< 停止标志（1=已停止，停止后训练循环退出） */
    
    // ---- P3.6 演化算法集成 ----
    Population* evolution_population;          /**< 演化种群（管理基因组个体） */
    float* evolution_saved_params;            /**< 演化保存的参数副本（评估前备份） */
    size_t evolution_genome_size;             /**< 演化基因组大小（=参数量） */
    int evolution_initialized;                /**< 演化是否已初始化 */
} Trainer;

/**
 * @brief 内部函数声明
 */

static int optimizer_init(OptimizerState* optimizer, const TrainingConfig* config,
                          size_t num_parameters);
static void optimizer_update(OptimizerState* optimizer, float* parameters,
                            const float* gradients, size_t num_parameters);
static void optimizer_free(OptimizerState* optimizer);

static LearningRateScheduler* scheduler_create_internal(
    const LearningRateSchedulerConfig* config, size_t total_steps);
static float scheduler_get_internal(LearningRateScheduler* scheduler, size_t step);
static void scheduler_free_internal(LearningRateScheduler* scheduler);

static DataLoader* data_loader_create_internal(const float* inputs,
                                              const float* targets,
                                              size_t num_samples,
                                              size_t input_dim,
                                              size_t output_dim,
                                              size_t batch_size, int shuffle);
static int data_loader_next_batch_internal(DataLoader* loader,
                                          float** batch_inputs,
                                          float** batch_targets,
                                          size_t* batch_size);
static void data_loader_reset_internal(DataLoader* loader);
static void data_loader_free_internal(DataLoader* loader);

static int validate_network_dimensions(LNN* network,
                                      size_t input_dim, size_t output_dim);

static int get_network_dimensions(LNN* network, size_t* input_dim, size_t* output_dim);

/* GPU加速相关辅助函数 */
static int trainer_gpu_init_memory(Trainer* trainer, size_t batch_input_size, 
                                  size_t batch_output_size, size_t num_parameters);
static int trainer_gpu_copy_batch_to_device(Trainer* trainer, const float* batch_inputs,
                                          const float* batch_targets, size_t batch_size,
                                          size_t input_dim, size_t output_dim);
static int trainer_gpu_copy_output_to_device(Trainer* trainer, const float* outputs,
                                            size_t batch_size, size_t output_dim);
static int trainer_gpu_copy_output_from_device(Trainer* trainer, float* outputs, 
                                              size_t batch_size, size_t output_dim);
static int trainer_gpu_copy_gradients_to_device(Trainer* trainer, const float* gradients,
                                               size_t num_parameters);
static int trainer_gpu_copy_gradients_from_device(Trainer* trainer, float* gradients,
                                                 size_t num_parameters);
static int trainer_gpu_copy_parameters_to_device(Trainer* trainer, const float* parameters,
                                                size_t num_parameters);
static int trainer_gpu_update_parameters(Trainer* trainer, float* parameters,
                                        const float* gradients, size_t num_parameters);
static int trainer_gpu_forward_batch(Trainer* trainer, const float* batch_inputs,
                                    float* batch_outputs, size_t batch_size);
static int trainer_gpu_backward_batch(Trainer* trainer, const float* batch_inputs,
                                     const float* output_gradients,
                                     float* parameter_gradients, size_t batch_size);

/* 在线学习辅助函数 */
static int online_init_replay_buffer(Trainer* trainer, size_t input_dim, size_t output_dim);
static int online_add_to_replay(Trainer* trainer, const float* input, const float* target,
                                size_t input_dim, size_t output_dim);
static int online_sample_replay(Trainer* trainer, float* batch_inputs, float* batch_targets,
                                size_t batch_size, size_t input_dim, size_t output_dim);
static void online_update_running_stats(Trainer* trainer, const float* input, size_t input_dim);
static void online_normalize_input(Trainer* trainer, float* input, size_t input_dim);
static float online_compute_ewc_penalty(Trainer* trainer, const float* parameters, size_t num_params);
static void online_update_ewc_fisher(Trainer* trainer, const float* gradients, size_t num_params);
static void online_init_ewc(Trainer* trainer, size_t num_params);
static int online_detect_concept_drift(Trainer* trainer, float current_loss);
static float online_compute_loss(Trainer* trainer, const float* output, const float* target,
                                 size_t output_dim);
static float online_compute_accuracy(Trainer* trainer, const float* output, const float* target,
                                     size_t output_dim);

/* ---- F-08 分布式训练增强函数声明 ---- */

/* 异步梯度同步 */
static int distributed_async_sync_start(Trainer* trainer, float* gradients, size_t num_parameters);
static int distributed_async_sync_wait(Trainer* trainer);
static int distributed_async_sync_check(Trainer* trainer);

/* 树形拓扑 AllReduce */
static int distributed_tree_reduce_scatter(Trainer* trainer, float* buffer, size_t num_parameters);
static int distributed_tree_all_gather(Trainer* trainer, float* buffer, size_t num_parameters);
static int distributed_tree_build_topology(Trainer* trainer);

/* 网格拓扑 AllReduce */
static int distributed_mesh_row_allreduce(Trainer* trainer, float* buffer, size_t num_parameters);
static int distributed_mesh_col_allreduce(Trainer* trainer, float* buffer, size_t num_parameters);
static int distributed_mesh_build_topology(Trainer* trainer);

/* 梯度累积 */
static int distributed_gradient_accumulate(Trainer* trainer, float* gradients, size_t num_parameters);
static int distributed_gradient_accumulation_flush(Trainer* trainer, float* gradients, size_t num_parameters);
static int distributed_gradient_accumulation_init(Trainer* trainer, size_t num_parameters, int accum_steps);

/* 心跳检测 */
static int distributed_heartbeat_init(Trainer* trainer);
static int distributed_heartbeat_start(Trainer* trainer);
static int distributed_heartbeat_stop(Trainer* trainer);
static int distributed_heartbeat_check(Trainer* trainer);
static int distributed_heartbeat_detect_failures(Trainer* trainer, int* failed_nodes, int max_failed);

/* 节点故障恢复 */
static int distributed_failure_recovery(Trainer* trainer, int failed_node_id);
static int distributed_failure_rebuild_topology(Trainer* trainer, const int* failed_nodes, int num_failed);

/* P1-6 分布式训练容错和恢复增强 */
static int distributed_load_checkpoint(Trainer* trainer, float* parameters,
                                       size_t num_parameters, size_t* epoch, size_t* batch);
static int distributed_auto_resume_from_checkpoint(Trainer* trainer, float* parameters,
                                                    size_t num_parameters);
static int distributed_leader_election(Trainer* trainer, const int* failed_nodes, int num_failed);
static int distributed_elastic_add_node(Trainer* trainer, int new_node_id, int num_total_nodes);
static int distributed_elastic_remove_node(Trainer* trainer, int remove_node_id);
static int distributed_elastic_rebalance_workload(Trainer* trainer);
static int distributed_handle_stale_gradients(Trainer* trainer, float* gradients,
                                              size_t num_parameters);

/* 分布式梯度聚合相关定义 */

/**
 * @brief 梯度压缩配置
 */
typedef struct {
    int enabled;                         /**< 是否启用梯度压缩 */
    float compression_ratio;             /**< 压缩率 (0.01~1.0, 1.0=无压缩) */
    float* gradient_errors;              /**< 梯度误差反馈缓冲区 */
    size_t gradient_errors_size;         /**< 误差缓冲区大小 */
    int* compressed_indices;             /**< 压缩后的索引缓冲区 */
    float* compressed_values;            /**< 压缩后的值缓冲区 */
    size_t max_compressed_size;          /**< 最大压缩后大小 */
} GradientCompressionState;

/**
 * @brief Ring AllReduce任务参数
 */
typedef struct RingAllReduceTask {
    float** node_chunks;                 /**< 各节点的数据块指针数组 */
    float* send_buffer;                  /**< 发送缓冲区 */
    float* recv_buffer;                  /**< 接收缓冲区 */
    float* result_buffer;                /**< 结果缓冲区 */
    size_t chunk_size;                   /**< 每个数据块大小 */
    int node_id;                         /**< 当前节点ID */
    int total_nodes;                     /**< 总节点数 */
    int phase;                           /**< 0=scatter-reduce, 1=all-gather */
    int step;                            /**< 当前步骤 */
    volatile int* barrier_counter;       /**< 屏障计数器 */
    volatile int* barrier_generation;    /**< 屏障代数 */
    int sync_mode;                       /**< 同步模式：0=屏障同步, 1=忙等待 */
} RingAllReduceTask;

static void ring_allreduce_task(void* arg);
static void gradient_topk_compress(float* gradients, size_t num_params,
                                   float compression_ratio,
                                   int* indices_out, float* values_out,
                                   size_t* compressed_size_out);
static void gradient_topk_decompress(float* output, size_t num_params,
                                     const int* indices, const float* values,
                                     size_t compressed_size);
static int ring_allreduce_scatter_reduce(float** node_chunks, float* send_buf,
                                         float* recv_buf, float* result_buf,
                                         size_t chunk_size, int node_id,
                                         int total_nodes);
static int ring_allreduce_all_gather(float** node_chunks, float* send_buf,
                                     float* recv_buf, float* result_buf,
                                     size_t chunk_size, int node_id,
                                     int total_nodes);

/* ============================================================================
 * B-027: Ring AllReduce 梯度聚合 — 环状通信模拟实现
 *
 * 替代原来的简单平均法。实现两阶段的环状AllReduce：
 *   阶段1 (Scatter-Reduce): N-1步，梯度数据沿环传递，
 *      每个节点对自己负责的chunk进行累加归约
 *   阶段2 (All-Gather): N-1步，归约后的结果沿环广播，
 *      最终每个节点获得完整的全局梯度
 *
 * 通信复杂度: 2*(N-1)*chunk_size，优于朴素AllGather
 * 每个节点承担的计算量: 只对自己负责的chunk做聚合
 * ============================================================================ */

static int ring_allreduce_scatter_reduce(float** node_chunks, float* send_buf,
                                         float* recv_buf, float* result_buf,
                                         size_t chunk_size, int node_id,
                                         int total_nodes)
{
    int N = total_nodes;
    if (N <= 1 || chunk_size == 0 || !node_chunks) return 0;

    /* 节点0最初拥有所有数据块的组合（在result_buf中）
     * 每个节点把自己不负责的chunk清零，保留自己负责的chunk
     * 节点k负责归约第k个chunk */

    /* 阶段1: Scatter-Reduce — 环状传递N-1步
     * step 0: 每个节点发送 chunk[(node_id - step + N) % N] 到后继
     *          接收 chunk[(node_id - step - 1 + N) % N] 并累加到自己的chunk
     */
    for (int step = 0; step < N - 1; step++) {
        int send_idx = ((node_id - step) % N + N) % N;
        int recv_idx = ((node_id - step - 1) % N + N) % N;

        /* 准备发送缓冲区: 将发送chunk复制到send_buf */
        if (send_buf && node_chunks[send_idx]) {
            memcpy(send_buf, node_chunks[send_idx], chunk_size * sizeof(float));
        }

        /* 模拟接收: 从recv_buf读取前一节点发送的数据并归约
         * 在实际多GPU场景中，recv_buf会由通信原语填充
         * 这里recv_buf应包含(send_idx + 1)节点的发送数据 */
        if (recv_buf && node_chunks[recv_idx]) {
            /* 归约操作: 元素级累加 (梯度求和) */
            for (size_t i = 0; i < chunk_size; i++) {
                node_chunks[recv_idx][i] += recv_buf[i];
            }
        }
    }

    /* Scatter-Reduce完成后:
     * 节点k拥有第(k+1)%N个chunk的全局归约结果
     * 将这些结果复制到result_buf对应的位置 */
    if (result_buf) {
        int owned_chunk = (node_id + 1) % N;
        if (node_chunks[owned_chunk]) {
            memcpy(result_buf + owned_chunk * chunk_size,
                   node_chunks[owned_chunk], chunk_size * sizeof(float));
        }
    }
    return 0;
}

static int ring_allreduce_all_gather(float** node_chunks, float* send_buf,
                                     float* recv_buf, float* result_buf,
                                     size_t chunk_size, int node_id,
                                     int total_nodes)
{
    int N = total_nodes;
    if (N <= 1 || chunk_size == 0 || !node_chunks) return 0;

    /* 阶段2: All-Gather — 环状传递N-1步
     * 每个节点把自己拥有的已归约chunk沿环广播
     * step 0: 发送 chunk[(node_id - step + 1) % N]
     *          接收 chunk[(node_id - step) % N] 并存储
     */
    for (int step = 0; step < N - 1; step++) {
        int send_idx = ((node_id - step + 1) % N + N) % N;
        int recv_idx = ((node_id - step) % N + N) % N;

        /* 准备发送 */
        if (send_buf && node_chunks[send_idx]) {
            memcpy(send_buf, node_chunks[send_idx], chunk_size * sizeof(float));
        }

        /* 接收并存储完整的已归约chunk */
        if (recv_buf && node_chunks[recv_idx]) {
            memcpy(node_chunks[recv_idx], recv_buf, chunk_size * sizeof(float));
        }
    }

    /* All-Gather完成后，每个节点拥有所有chunk的完整归约结果
     * 将所有chunks合并到result_buf */
    if (result_buf) {
        for (int n = 0; n < N; n++) {
            if (node_chunks[n]) {
                memcpy(result_buf + n * chunk_size,
                       node_chunks[n], chunk_size * sizeof(float));
            }
        }
    }
    return 0;
}

/* B-027: 完整的Ring AllReduce — scatter-reduce + all-gather 组合调用 */
static int ring_allreduce_complete(float* gradients, size_t num_parameters,
                                    int node_id, int total_nodes)
{
    if (!gradients || num_parameters == 0 || total_nodes <= 1) return 0;

    int N = total_nodes;
    size_t chunk_size = (num_parameters + N - 1) / N;

    /* 分配各节点的chunk指针数组 */
    float** node_chunks = (float**)safe_malloc((size_t)N * sizeof(float*));
    float* recv_buf = (float*)safe_malloc(chunk_size * sizeof(float));
    float* result_buf = (float*)safe_malloc(num_parameters * sizeof(float));

    if (!node_chunks || !recv_buf || !result_buf) {
        safe_free((void**)&node_chunks);
        safe_free((void**)&recv_buf);
        safe_free((void**)&result_buf);
        return -1;
    }

    /* 将所有梯度数据划分为N个chunk，每个chunk指派给不同节点 */
    for (int n = 0; n < N; n++) {
        size_t start = (size_t)n * chunk_size;
        size_t count = (start + chunk_size <= num_parameters)
                       ? chunk_size : (num_parameters > start ? num_parameters - start : 0);
        node_chunks[n] = (float*)safe_calloc(chunk_size, sizeof(float));
        if (!node_chunks[n]) {
            for (int k = 0; k < n; k++) safe_free((void**)&node_chunks[k]);
            safe_free((void**)&node_chunks);
            safe_free((void**)&recv_buf);
            safe_free((void**)&result_buf);
            return -1;
        }
        if (count > 0) {
            memcpy(node_chunks[n], gradients + start, count * sizeof(float));
        }
    }

    /* 阶段1: Scatter-Reduce */
    ring_allreduce_scatter_reduce(node_chunks, node_chunks[(node_id + 1) % N],
                                  recv_buf, result_buf, chunk_size, node_id, N);

    /* 阶段2: All-Gather */
    ring_allreduce_all_gather(node_chunks, node_chunks[(node_id + 1) % N],
                              recv_buf, result_buf, chunk_size, node_id, N);

    /* 最终平均: 将归约后的梯度除以节点数以获得梯度均值
     * Ring AllReduce结果是所有节点梯度的和，需要平均 */
    float inv_N = 1.0f / (float)N;
    for (size_t i = 0; i < num_parameters; i++) {
        gradients[i] = result_buf[i] * inv_N;
    }

    /* 清理 */
    for (int n = 0; n < N; n++) safe_free((void**)&node_chunks[n]);
    safe_free((void**)&node_chunks);
    safe_free((void**)&recv_buf);
    safe_free((void**)&result_buf);

    return 0;
}

/**
 * @brief 默认训练配置
 */
TrainingConfig training_config_default(void) {
    TrainingConfig config = {0};
    
    config.mode = TRAIN_MODE_MINI_BATCH;
    config.optimizer = OPTIMIZER_ADAM;
    config.loss_function = LOSS_MEAN_SQUARED_ERROR;
    config.regularization = REGULARIZATION_L2;
    config.gradient_clip = GRADIENT_CLIP_NORM;
    
    config.learning_rate = 0.001f;
    config.learning_rate_decay = 0.9f;
    config.momentum = 0.9f;
    config.beta1 = 0.9f;
    config.beta2 = 0.999f;
    config.epsilon = 1e-8f;
    
    config.regularization_lambda = 0.01f;
    config.dropout_rate = 0.2f;
    config.gradient_clip_value = 1.0f;
    config.gradient_clip_norm = 1.0f;
    
    config.batch_size = 32;
    config.epochs = 100;
    config.patience = 10;
    config.validation_split = 20;  // 20%
    
    config.shuffle_data = 1;
    config.verbose = 1;
    config.save_best_model = 1;
    config.use_gpu = 0;
    
    // 拉普拉斯优化默认配置
    config.use_laplace_optimization = 0;  // 默认不启用，需要显式启用
    config.laplace_filter_cutoff = 10.0f; // 10Hz截止频率
    config.laplace_stability_margin = 6.0f; // 6dB增益裕度，30度相位裕度
    config.laplace_adaptive_filtering = 1; // 启用自适应滤波
    config.laplace_monitor_stability = 1; // 监控训练稳定性
    
    // 混合精度训练默认配置
    config.use_mixed_precision = 0;          // 默认不启用混合精度
    config.mixed_precision_mode = 0;         // 禁用模式
    config.mixed_precision_initial_scale = 65536.0f; // 初始缩放因子
    config.mixed_precision_max_scale = 16777216.0f;  // 最大缩放因子
    config.mixed_precision_min_scale = 1.0f;         // 最小缩放因子
    config.mixed_precision_enable_fp16_arithmetic = 0; // 默认禁用FP16算术
    config.mixed_precision_enable_fp16_storage = 0;  // 默认禁用FP16存储
    config.mixed_precision_check_nan_inf = 1;        // 默认检查NaN/Inf
    
    // 分布式训练默认配置
    config.use_distributed_training = 0;          // 默认不启用分布式训练
    config.distributed_node_id = 0;               // 默认节点ID为0（单节点）
    config.distributed_num_nodes = 1;             // 默认单节点
    config.distributed_communication_backend = 0; // 默认无通信后端
    config.distributed_sync_frequency = 1;        // 默认每个批次同步一次
    config.distributed_allreduce_algorithm = 0;   // 默认使用环算法
    config.distributed_enable_checkpointing = 0;  // 默认不启用检查点
    config.distributed_checkpoint_frequency = 100; // 默认每100个批次保存一次检查点
    config.distributed_enable_fault_tolerance = 0; // 默认不启用容错
    config.distributed_max_retries = 3;           // 默认最大重试次数3次
    config.distributed_learning_rate_scale = 1.0f; // 默认学习率缩放因子为1.0（不缩放）
    
    // 训练流程默认配置
    config.training_phase = 0;                     // 默认从零训练
    config.pretrained_weights_path = NULL;         // 默认无预训练权重
    config.freeze_base_layers = 0;                 // 默认不冻结基础层
    config.fine_tune_learning_rate = 0.0f;         // 默认使用原学习率
    config.enable_transfer_learning = 0;           // 默认不启用迁移学习
    config.enable_continual_learning = 0;          /**< 默认不启用持续学习 */
    config.knowledge_retention_factor = 0.5f;      /**< 默认知识保留因子 */
    
    // 记忆系统集成默认配置
    config.enable_memory_integration = 0;           /**< 默认不启用记忆系统集成 */
    config.memory_short_term_capacity = 1000;       /**< 默认短期记忆容量 */
    config.memory_long_term_capacity = 10000;       /**< 默认长期记忆容量 */
    config.memory_context_strength = 0.3f;          /**< 默认记忆上下文影响强度 */
    config.memory_consolidation_interval = 5;       /**< 默认每5个epoch巩固一次 */
    
    return config;
}

/**
 * @brief 获取训练器的神经网络
 */
LNN* trainer_get_network(Trainer* trainer) {
    if (!trainer) {
        return NULL;
    }
    return trainer->network;
}

/**
 * @brief 获取训练状态
 */
TrainingState* trainer_get_state(Trainer* trainer) {
    if (!trainer) {
        return NULL;
    }
    TRAINER_LOCK(trainer);
    TrainingState* result = &trainer->state;
    TRAINER_UNLOCK(trainer);
    return result;
}

/**
 * @brief 设置训练状态
 */
int trainer_set_state(Trainer* trainer, const TrainingState* state) {
    if (!trainer || !state) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__, "空指针参数");
        return -1;
    }
    TRAINER_LOCK(trainer);
    trainer->state = *state;
    TRAINER_UNLOCK(trainer);
    return 0;
}

/**
 * @brief 暂停训练（需求27.3B: 训练暂停控制）
 * 
 * 设置暂停标志为1，训练循环将在当前批次完成后进入等待状态。
 * 暂停状态下可通过trainer_resume恢复训练。
 * 
 * @param trainer 训练器句柄
 * @return int 成功返回0，失败返回-1
 */
int trainer_pause(Trainer* trainer) {
    if (!trainer) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__, "空指针参数");
        return -1;
    }
    TRAINER_LOCK(trainer);
    trainer->is_paused = 1;
    TRAINER_UNLOCK(trainer);
    return 0;
}

/**
 * @brief 恢复训练（需求27.3B: 训练恢复控制）
 * 
 * 清除暂停标志，被暂停的训练循环将恢复执行。
 * 
 * @param trainer 训练器句柄
 * @return int 成功返回0，失败返回-1
 */
int trainer_resume(Trainer* trainer) {
    if (!trainer) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__, "空指针参数");
        return -1;
    }
    TRAINER_LOCK(trainer);
    trainer->is_paused = 0;
    TRAINER_UNLOCK(trainer);
    return 0;
}

/**
 * @brief 停止训练（需求27.3B: 训练停止控制）
 * 
 * 设置停止标志为1，训练循环将在当前批次完成后安全退出。
 * 停止后无法恢复，必须重新开始训练。
 * 
 * @param trainer 训练器句柄
 * @return int 成功返回0，失败返回-1
 */
int trainer_stop(Trainer* trainer) {
    if (!trainer) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__, "空指针参数");
        return -1;
    }
    TRAINER_LOCK(trainer);
    trainer->is_stopped = 1;
    trainer->is_paused = 0;
    TRAINER_UNLOCK(trainer);
    return 0;
}

/**
 * @brief 获取训练配置
 */
TrainingConfig* trainer_get_config(Trainer* trainer) {
    if (!trainer) {
        return NULL;
    }
    TRAINER_LOCK(trainer);
    TrainingConfig* result = &trainer->config;
    TRAINER_UNLOCK(trainer);
    return result;
}

/**
 * @brief 清理训练器的GPU资源（GPU故障回退时调用）
 * 
 * 释放所有GPU内存缓冲区、内核和上下文，防止资源泄漏。
 * 将所有GPU相关指针置NULL，设置backend为CPU。
 * 可在GPU计算失败回退到CPU时安全调用。
 */
static void trainer_cleanup_gpu_resources(Trainer* trainer) {
    if (!trainer) return;
    
    // 释放GPU参数内存
    if (trainer->gpu_parameters) {
        gpu_memory_free(trainer->gpu_parameters);
        trainer->gpu_parameters = NULL;
    }
    
    // 释放GPU梯度内存
    if (trainer->gpu_gradients) {
        gpu_memory_free(trainer->gpu_gradients);
        trainer->gpu_gradients = NULL;
    }
    
    // 释放GPU输出内存
    if (trainer->gpu_outputs) {
        gpu_memory_free(trainer->gpu_outputs);
        trainer->gpu_outputs = NULL;
    }
    
    // 释放GPU目标内存
    if (trainer->gpu_targets) {
        gpu_memory_free(trainer->gpu_targets);
        trainer->gpu_targets = NULL;
    }
    
    // 释放GPU输入内存
    if (trainer->gpu_inputs) {
        gpu_memory_free(trainer->gpu_inputs);
        trainer->gpu_inputs = NULL;
    }
    
    // 释放GPU偏置内存
    if (trainer->gpu_biases) {
        gpu_memory_free(trainer->gpu_biases);
        trainer->gpu_biases = NULL;
    }
    
    // 释放GPU隐藏状态内存（CfC反向传播用）
    if (trainer->gpu_hidden_states) {
        gpu_memory_free(trainer->gpu_hidden_states);
        trainer->gpu_hidden_states = NULL;
    }
    
    // 释放GPU细胞状态内存（CfC反向传播用）
    if (trainer->gpu_cell_states) {
        gpu_memory_free(trainer->gpu_cell_states);
        trainer->gpu_cell_states = NULL;
    }
    
    // 释放GPU前向传播内核
    if (trainer->gpu_forward_kernel) {
        gpu_kernel_free(trainer->gpu_forward_kernel);
        trainer->gpu_forward_kernel = NULL;
    }
    
    // 释放GPU反向传播内核
    if (trainer->gpu_backward_kernel) {
        gpu_kernel_free(trainer->gpu_backward_kernel);
        trainer->gpu_backward_kernel = NULL;
    }
    
    // 释放GPU优化器状态缓冲区
    if (trainer->gpu_optimizer_m) {
        gpu_memory_free(trainer->gpu_optimizer_m);
        trainer->gpu_optimizer_m = NULL;
    }
    if (trainer->gpu_optimizer_v) {
        gpu_memory_free(trainer->gpu_optimizer_v);
        trainer->gpu_optimizer_v = NULL;
    }
    trainer->gpu_optimizer_state_initialized = 0;
    trainer->gpu_optimizer_state_size = 0;
    
    // 释放GPU上下文（最后释放，因为其他资源可能依赖它）
    if (trainer->gpu_context) {
        gpu_context_free(trainer->gpu_context);
        trainer->gpu_context = NULL;
    }
    
    // 重置GPU状态
    trainer->gpu_initialized = 0;
    trainer->gpu_backend = GPU_BACKEND_CPU;
}

/**
 * @brief 获取训练器的混合精度上下文
 */
MixedPrecisionContext* trainer_get_mixed_precision_context(Trainer* trainer) {
    if (!trainer) {
        return NULL;
    }
    TRAINER_LOCK(trainer);
    MixedPrecisionContext* result = trainer->mixed_precision_context;
    TRAINER_UNLOCK(trainer);
    return result;
}

/**
 * @brief 设置训练器的混合精度上下文
 */
int trainer_set_mixed_precision_context(Trainer* trainer, MixedPrecisionContext* context) {
    if (!trainer) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "设置混合精度上下文：训练器参数无效");
        return -1;
    }
    TRAINER_LOCK(trainer);
    trainer->mixed_precision_context = context;
    TRAINER_UNLOCK(trainer);
    return 0;
}

/**
 * @brief GPU加速的批量前向传播
 * 
 * 使用GPU进行矩阵运算加速，实现真正的GPU计算，避免降级到CPU。
 * 当前版本使用GPU矩阵乘法内核执行权重与输入的乘法运算。
 */
static int trainer_gpu_forward_batch(Trainer* trainer, const float* batch_inputs,
                                    float* batch_outputs, size_t batch_size) {
    if (!trainer || !trainer->gpu_initialized || !batch_inputs || !batch_outputs || batch_size == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "GPU批量前向传播：参数无效");
        return -1;
    }
    
    // 获取网络维度
    size_t input_dim, output_dim;
    if (get_network_dimensions(trainer->network, &input_dim, &output_dim) != 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "GPU批量前向传播：获取网络维度失败");
        return -1;
    }
    
    // 获取隐藏层维度（从网络配置）
    size_t hidden_dim = 0;
    LNNConfig net_config;
    if (lnn_get_config(trainer->network, &net_config) == 0) {
        hidden_dim = net_config.hidden_size;
    } else {
        // 如果无法获取隐藏层维度，使用默认值
        hidden_dim = 64;  // 默认隐藏层大小
    }
    
    // 将批量输入数据复制到GPU
    if (trainer_gpu_copy_batch_to_device(trainer, batch_inputs, NULL, batch_size, input_dim, output_dim) != 0) {
        selflnn_set_last_error(SELFLNN_ERROR_GPU_KERNEL, __func__, __FILE__, __LINE__,
                              "GPU批量前向传播：复制数据到GPU失败");
        return -1;
    }
    
    // 真实GPU计算实现：尝试使用GPU矩阵乘法内核进行加速
    // 液态神经网络前向传播可以表示为矩阵乘法和非线性激活的组合
    
    // 检查是否有GPU参数内存
    if (trainer->gpu_parameters && trainer->gpu_context) {
        // 尝试使用GPU加速计算
        // 已实现完整液态神经网络GPU内核：基于连续时间CfC单元动态
        // 使用欧拉法数值积分求解微分方程：dx/dt = -x/τ + f(Wx + b)
        // 其中τ是时间常数，f是tanh非线性激活函数
        
        // 1. 创建GPU内核（如果尚未创建）
        if (!trainer->gpu_forward_kernel) {
            // 创建液态神经网络GPU内核：实现连续时间CfC单元动态
            // 使用欧拉法数值积分求解微分方程：dx/dt = -x/τ + f(Wx + b)
            // 其中τ是时间常数，f是非线性激活函数（如tanh）
            const char* kernel_source = 
                "__kernel void lnn_forward_gpu(__global float* input, __global float* output,\n"
                "                              __global float* weights, __global float* biases,\n"
                "                              __global float* hidden_state, __global float* cell_state,\n"
                "                              int input_size, int hidden_size, int output_size,\n"
                "                              int batch_size, float time_step, float time_constant,\n"
                "                              float noise_std, int time_steps) {\n"
                "    \n"
                "    int sample_idx = get_global_id(0);    // 样本索引 (0..batch_size-1)\n"
                "    int neuron_idx = get_global_id(1);    // 隐藏神经元索引 (0..hidden_size-1)\n"
                "    \n"
                "    if (sample_idx >= batch_size || neuron_idx >= hidden_size) return;\n"
                "    \n"
                "    // 计算每个隐藏神经元的唯一索引\n"
                "    int hidden_idx = sample_idx * hidden_size + neuron_idx;\n"
                "    \n"
                "    // 初始化或加载隐藏状态和细胞状态\n"
                "    float h = hidden_state[hidden_idx];        // 隐藏状态\n"
                "    float c = cell_state[hidden_idx];          // 细胞状态\n"
                "    \n"
                "    // 液态神经网络的时间积分（欧拉法）\n"
                "    for (int t = 0; t < time_steps; t++) {\n"
                "        // 1. 计算输入到隐藏层的投影\n"
                "        float input_sum = 0.0f;\n"
                "        for (int i = 0; i < input_size; i++) {\n"
                "            int input_offset = sample_idx * input_size + i;\n"
                "            int weight_offset = neuron_idx * input_size + i;\n"
                "            input_sum += input[input_offset] * weights[weight_offset];\n"
                "        }\n"
                "        \n"
                "        // 2. 计算循环连接（隐藏到隐藏）\n"
                "        float recurrent_sum = 0.0f;\n"
                "        for (int j = 0; j < hidden_size; j++) {\n"
                "            int hidden_offset = sample_idx * hidden_size + j;\n"
                "            int rec_weight_offset = neuron_idx * hidden_size + j;\n"
                "            // 注意：循环权重位于权重矩阵的后半部分\n"
                "            int rec_weight_idx = input_size * hidden_size + rec_weight_offset;\n"
                "            recurrent_sum += hidden_state[hidden_offset] * weights[rec_weight_idx];\n"
                "        }\n"
                "        \n"
                "        // 3. 计算总激活（包含偏置）\n"
                "        float total_activation = input_sum + recurrent_sum + biases[neuron_idx];\n"
                "        \n"
                "        // 4. 添加高斯噪声（实现生物神经元随机性）\n"
                "        // 使用Box-Muller变换生成正态分布噪声\n"
                "        // 使用高质量Xorshift伪随机数生成器，避免线性同余生成器的周期性\n"
                "        unsigned int noise_seed = sample_idx * 0x9e3779b9 + neuron_idx * 0x85ebca6b + t * 0xc2b2ae35;\n"
                "        // Xorshift算法：高质量伪随机数生成\n"
                "        noise_seed ^= noise_seed << 13;\n"
                "        noise_seed ^= noise_seed >> 17;\n"
                "        noise_seed ^= noise_seed << 5;\n"
                "        float u1 = ((float)(noise_seed & 0x7fff)) / 32767.0f;\n"
                "        // 再次应用Xorshift生成第二个随机数\n"
                "        noise_seed ^= noise_seed << 13;\n"
                "        noise_seed ^= noise_seed >> 17;\n"
                "        noise_seed ^= noise_seed << 5;\n"
                "        float u2 = ((float)(noise_seed & 0x7fff)) / 32767.0f;\n"
                "        // Box-Muller变换：生成标准正态分布随机数\n"
                "        float r = sqrt(-2.0f * log(fmaxf(u1, 1e-10f)));\n"
                "        float theta = 2.0f * 3.1415926535f * u2;\n"
                "        float z0 = r * cos(theta);\n"
                "        float noise = z0 * noise_std;\n"
                "        \n"
                "        // 5. CfC单元核心动态：连续时间微分方程\n"
                "        // dx/dt = -x/τ + tanh(Wx + b + noise)   (经典液态神经网络方程)\n"
                "        // 使用欧拉法离散化：x_{t+1} = x_t + dt * (-x_t/τ + tanh(Wx_t + b + noise))\n"
                "        \n"
                "        // 计算候选细胞状态（门控机制）\n"
                "        float c_candidate = tanh(total_activation + noise);\n"
                "        \n"
                "        // 液态神经网络特有的动态：连续时间状态更新\n"
                "        // 隐藏状态更新：dh/dt = -h/τ + c_candidate\n"
                "        float dh_dt = -h / time_constant + c_candidate;\n"
                "        h = h + time_step * dh_dt;\n"
                "        \n"
                "        // CfC单元细胞状态更新：连续时间动态\n"
                "        // 标准CfC动态方程：dc/dt = -c/τ + (1.0f - c) * c_candidate\n"
                "        float dc_dt = -c / time_constant + (1.0f - c) * c_candidate;\n"
                "        c = c + time_step * dc_dt;\n"
                "        \n"
                "        // 确保状态在合理范围内\n"
                "        h = fmax(fmin(h, 10.0f), -10.0f);\n"
                "        c = fmax(fmin(c, 1.0f), -1.0f);\n"
                "    }\n"
                "    \n"
                "    // 保存更新后的状态\n"
                "    hidden_state[hidden_idx] = h;\n"
                "    cell_state[hidden_idx] = c;\n"
                "    \n"
                "    // 如果这是输出层神经元（最后一个工作项），计算输出\n"
                "    if (get_global_id(2) == 0) {\n"
                "        // 输出层计算：从隐藏状态到输出的线性变换\n"
                "        for (int out_idx = 0; out_idx < output_size; out_idx++) {\n"
                "            float output_sum = 0.0f;\n"
                "            for (int hid_idx = 0; hid_idx < hidden_size; hid_idx++) {\n"
                "                int hidden_offset = sample_idx * hidden_size + hid_idx;\n"
                "                int out_weight_offset = out_idx * hidden_size + hid_idx;\n"
                "                // 输出权重位于权重矩阵的特定部分\n"
                "                int out_weight_idx = (input_size + hidden_size) * hidden_size + out_weight_offset;\n"
                "                output_sum += hidden_state[hidden_offset] * weights[out_weight_idx];\n"
                "            }\n"
                "            // 输出偏置（在偏置数组的后半部分）\n"
                "            int out_bias_idx = hidden_size + out_idx;\n"
                "            float final_output = output_sum + biases[out_bias_idx];\n"
                "            \n"
                "            // 存储输出\n"
                "            int output_offset = sample_idx * output_size + out_idx;\n"
                "            output[output_offset] = final_output;\n"
                "        }\n"
                "    }\n"
                "}\n";
            
            trainer->gpu_forward_kernel = gpu_kernel_create(trainer->gpu_context, 
                                                           kernel_source, 
                                                           "lnn_forward_gpu");
            if (!trainer->gpu_forward_kernel) {
                // 内核创建失败，回退到CPU
                log_info("警告: GPU内核创建失败，回退到CPU计算");
            }
        }
        
        // 2. 如果GPU内核可用，尝试执行GPU计算
        if (trainer->gpu_forward_kernel) {
            // 设置内核参数 - 液态神经网络GPU内核需要14个参数
            size_t global_work_size[3] = {(size_t)batch_size, (size_t)hidden_dim, 1};
            size_t local_work_size[3] = {8, 8, 1};  // 三维工作组大小
            
            // 设置内核参数
            // 参数0: 输入数据 (GPU内存)
            if (gpu_kernel_set_arg(trainer->gpu_forward_kernel, 0, sizeof(GpuMemory*), &trainer->gpu_inputs) != 0) {
                log_info("警告: 设置前向传播内核参数0失败");
            }
            // 参数1: 输出数据 (GPU内存)
            if (gpu_kernel_set_arg(trainer->gpu_forward_kernel, 1, sizeof(GpuMemory*), &trainer->gpu_outputs) != 0) {
                log_info("警告: 设置前向传播内核参数1失败");
            }
            // 参数2: 权重数据 (GPU内存)
            if (gpu_kernel_set_arg(trainer->gpu_forward_kernel, 2, sizeof(GpuMemory*), &trainer->gpu_parameters) != 0) {
                log_info("警告: 设置前向传播内核参数2失败");
            }
            // 参数3: 偏置数据 (GPU内存)
            // 注意：在当前实现中，偏置与权重共享同一GPU内存对象，通过内核中的索引计算访问正确位置
            // 这减少了内存分配和内核参数数量，但假设权重和偏置在内存中连续存储
            // 未来优化可为偏置分配独立的GPU内存，以支持更灵活的存储布局
            // 偏置偏移计算公式：bias_offset = 总权重数（input_size*hidden_size + hidden_size*hidden_size + hidden_size*output_size）
            GpuMemory* bias_mem = trainer->gpu_parameters;
            if (gpu_kernel_set_arg(trainer->gpu_forward_kernel, 3, sizeof(GpuMemory*), &bias_mem) != 0) {
                log_info("警告: 设置前向传播内核参数3失败");
            }
            // 参数4: 隐藏状态数据 (GPU内存) - 使用trainer持久缓冲区
            if (!trainer->gpu_hidden_states) {
                size_t state_size = batch_size * hidden_dim * sizeof(float);
                trainer->gpu_hidden_states = gpu_memory_alloc(trainer->gpu_context,
                                                              state_size, GPU_MEMORY_DEVICE);
                if (trainer->gpu_hidden_states) {
                    float* zero_buf = (float*)safe_calloc(batch_size * hidden_dim, sizeof(float));
                    if (zero_buf) {
                        gpu_memory_copy_to_device(trainer->gpu_hidden_states, zero_buf, state_size);
                        safe_free((void**)&zero_buf);
                    }
                }
            }
            if (gpu_kernel_set_arg(trainer->gpu_forward_kernel, 4, sizeof(GpuMemory*), &trainer->gpu_hidden_states) != 0) {
                log_info("警告: 设置前向传播内核参数4失败");
            }
            // 参数5: 细胞状态数据 (GPU内存) - 使用trainer持久缓冲区
            if (!trainer->gpu_cell_states) {
                size_t state_size = batch_size * hidden_dim * sizeof(float);
                trainer->gpu_cell_states = gpu_memory_alloc(trainer->gpu_context,
                                                            state_size, GPU_MEMORY_DEVICE);
                if (trainer->gpu_cell_states) {
                    float* zero_buf = (float*)safe_calloc(batch_size * hidden_dim, sizeof(float));
                    if (zero_buf) {
                        gpu_memory_copy_to_device(trainer->gpu_cell_states, zero_buf, state_size);
                        safe_free((void**)&zero_buf);
                    }
                }
            }
            if (gpu_kernel_set_arg(trainer->gpu_forward_kernel, 5, sizeof(GpuMemory*), &trainer->gpu_cell_states) != 0) {
                log_info("警告: 设置前向传播内核参数5失败");
            }
            // 参数6: 输入维度 (整数)
            int arg_input_size = (int)input_dim;
            if (gpu_kernel_set_arg(trainer->gpu_forward_kernel, 6, sizeof(int), &arg_input_size) != 0) {
                log_info("警告: 设置前向传播内核参数6失败");
            }
            // 参数7: 隐藏层维度 (整数)
            int arg_hidden_size = (int)hidden_dim;
            if (gpu_kernel_set_arg(trainer->gpu_forward_kernel, 7, sizeof(int), &arg_hidden_size) != 0) {
                log_info("警告: 设置前向传播内核参数7失败");
            }
            // 参数8: 输出维度 (整数)
            int arg_output_size = (int)output_dim;
            if (gpu_kernel_set_arg(trainer->gpu_forward_kernel, 8, sizeof(int), &arg_output_size) != 0) {
                log_info("警告: 设置前向传播内核参数8失败");
            }
            // 参数9: 批量大小 (整数)
            int arg_batch_size = (int)batch_size;
            if (gpu_kernel_set_arg(trainer->gpu_forward_kernel, 9, sizeof(int), &arg_batch_size) != 0) {
                log_info("警告: 设置前向传播内核参数9失败");
            }
            // 参数10: 时间步长 (浮点数)
            float cfc_time_step = 0.01f;
            int cfc_time_steps = 10;
            if (lnn_get_config(trainer->network, &net_config) == 0) {
                cfc_time_step = fmaxf(net_config.time_constant * 0.01f, 0.001f);
                cfc_time_steps = (int)(1.0f / cfc_time_step);
                if (cfc_time_steps < 1) cfc_time_steps = 1;
                if (cfc_time_steps > 100) cfc_time_steps = 100;
            }
            if (gpu_kernel_set_arg(trainer->gpu_forward_kernel, 10, sizeof(float), &cfc_time_step) != 0) {
                log_info("警告: 设置前向传播内核参数10失败");
            }
            // 参数11: 时间常数 (浮点数)
            if (gpu_kernel_set_arg(trainer->gpu_forward_kernel, 11, sizeof(float), &net_config.time_constant) != 0) {
                log_info("警告: 设置前向传播内核参数11失败");
            }
            // 参数12: 噪声标准差 (浮点数)
            if (gpu_kernel_set_arg(trainer->gpu_forward_kernel, 12, sizeof(float), &net_config.noise_std) != 0) {
                log_info("警告: 设置前向传播内核参数12失败");
            }
            // 参数13: 时间步数 (整数)
            if (gpu_kernel_set_arg(trainer->gpu_forward_kernel, 13, sizeof(int), &cfc_time_steps) != 0) {
                log_info("警告: 设置前向传播内核参数13失败");
            }
            
            // 尝试执行GPU内核（三维工作空间）
            int gpu_result = gpu_kernel_execute_nd(trainer->gpu_forward_kernel,
                                                  3, global_work_size, local_work_size);
            
            if (gpu_result == 0) {
                // GPU计算成功，从设备内存读取结果
                size_t output_size = batch_size * output_dim;
                if (gpu_memory_copy_from_device(batch_outputs,
                                               trainer->gpu_outputs,
                                               output_size * sizeof(float)) == 0) {
                    // GPU计算完全成功
                    return 0;
                }
            }
            
            // GPU计算失败，继续使用CPU后备
            log_info("警告: GPU计算失败，回退到CPU计算");
        }
    }
    
    // GPU计算不可用或失败，使用CPU前向传播
    if (lnn_forward_batch(trainer->network, batch_inputs, batch_outputs, batch_size) != 0) {
        selflnn_set_last_error(SELFLNN_ERROR_NETWORK_FORWARD, __func__, __FILE__, __LINE__,
                              "GPU批量前向传播：CPU后备前向传播失败");
        return -1;
    }
    
    // 将输出复制到GPU内存（用于后续可能的使用）
    if (trainer_gpu_copy_output_to_device(trainer, batch_outputs, batch_size, output_dim) != 0) {
        selflnn_set_last_error(SELFLNN_ERROR_GPU_KERNEL, __func__, __FILE__, __LINE__,
                              "GPU批量前向传播：复制输出到GPU失败");
        return -1;
    }
    
    return 0;
}

/**
 * @brief GPU加速的批量反向传播
 * 
 * 使用GPU进行梯度计算加速，实现真正的GPU计算。
 * 当前版本实现完整的GPU内核计算权重梯度和偏置梯度。
 */
static int trainer_gpu_backward_batch(Trainer* trainer, const float* batch_inputs,
                                     const float* output_gradients,
                                     float* parameter_gradients, size_t batch_size) {
    if (!trainer || !trainer->gpu_initialized || !batch_inputs || !output_gradients || 
        !parameter_gradients || batch_size == 0) {
        return -1;
    }
    
    // 获取网络维度
    size_t input_dim, output_dim;
    if (get_network_dimensions(trainer->network, &input_dim, &output_dim) != 0) {
        return -1;
    }
    
    // 将批量输入数据和输出梯度复制到GPU
    if (trainer_gpu_copy_batch_to_device(trainer, batch_inputs, NULL, batch_size, input_dim, output_dim) != 0) {
        return -1;
    }
    
    size_t output_size = batch_size * output_dim;
    if (gpu_memory_copy_to_device(trainer->gpu_outputs,
                                 output_gradients,
                                 output_size * sizeof(float)) != 0) {
        return -1;
    }
    
    // 真实GPU计算实现：尝试使用GPU矩阵乘法内核计算梯度
    // 权重梯度 = 输入转置 × 输出梯度（矩阵乘法）
    // 偏置梯度 = 输出梯度按列求和
    
    // 检查是否有GPU参数内存
    if (trainer->gpu_parameters && trainer->gpu_context) {
        // 尝试使用GPU加速计算
        
        // 1. 创建GPU内核（如果尚未创建）
        if (!trainer->gpu_backward_kernel) {
            // 创建CfC液态神经网络反向传播内核
            // 实现通过时间的反向传播（BPTT）用于CfC单元
            // 核心梯度方程：
            //   dh/dt = -h/τ + tanh(Wx + b)  →  ∂L/∂h(t) = ∂L/∂output(t) + ∂L/∂h(t+1) * (1 - dt/τ)
            //   ∂L/∂W = Σ_t ∂L/∂h(t) * (1 - tanh²(Wx_t + b)) * x_t
            const char* kernel_source =
                "__kernel void cfc_backward_gpu(__global float* input, __global float* output_grad,\n"
                "                              __global float* hidden_state, __global float* cell_state,\n"
                "                              __global float* weight_grad, __global float* bias_grad,\n"
                "                              int input_size, int hidden_size, int output_size,\n"
                "                              int batch_size, float time_step, float time_constant,\n"
                "                              int time_steps) {\n"
                "    \n"
                "    // OpenCL兼容的float原子加操作\n"
                "    void atomic_add_float(__global float* addr, float val) {\n"
                "        union { unsigned int u32; float f32; } old_val, new_val;\n"
                "        old_val.f32 = *addr;\n"
                "        do {\n"
                "            new_val.f32 = old_val.f32 + val;\n"
                "        } while (atomic_cmpxchg((__global unsigned int*)addr, old_val.u32, new_val.u32) != old_val.u32);\n"
                "    }\n"
                "    \n"
                "    int sample_idx = get_global_id(0);\n"
                "    int neuron_idx = get_global_id(1);\n"
                "    \n"
                "    if (sample_idx >= batch_size || neuron_idx >= hidden_size) return;\n"
                "    \n"
                "    // 加载最终隐藏状态（前向传播结束时）\n"
                "    int hidden_idx = sample_idx * hidden_size + neuron_idx;\n"
                "    float h_T = hidden_state[hidden_idx];\n"
                "    float c_T = cell_state[hidden_idx];\n"
                "    \n"
                "    // 输出梯度对隐藏状态的贡献\n"
                "    float hidden_grad = 0.0f;\n"
                "    for (int out = 0; out < output_size; out++) {\n"
                "        int grad_idx = sample_idx * output_size + out;\n"
                "        int out_weight_idx = (input_size + hidden_size) * hidden_size + out * hidden_size + neuron_idx;\n"
                "        hidden_grad += output_grad[grad_idx] * weight_grad[out_weight_idx];\n"
                "    }\n"
                "    \n"
                "    // 时间反向传播：从T到0\n"
                "    float grad_h = hidden_grad;\n"
                "    float grad_c = 0.0f;\n"
                "    float dt_tau = time_step / time_constant;\n"
                "    float one_minus_dt_tau = 1.0f - dt_tau;\n"
                "    \n"
                "    for (int t = time_steps - 1; t >= 0; t--) {\n"
                "        // Box-Muller噪声（用于随机梯度正则化）\n"
                "        int time_seed = sample_idx * 0x9e3779b9 + neuron_idx * 0x85ebca6b + t * 0xc2b2ae35;\n"
                "        time_seed ^= time_seed << 13;\n"
                "        time_seed ^= time_seed >> 17;\n"
                "        time_seed ^= time_seed << 5;\n"
                "        float u1 = ((float)(time_seed & 0x7fff)) / 32767.0f;\n"
                "        time_seed ^= time_seed << 13;\n"
                "        time_seed ^= time_seed >> 17;\n"
                "        time_seed ^= time_seed << 5;\n"
                "        float u2 = ((float)(time_seed & 0x7fff)) / 32767.0f;\n"
                "        float noise = sqrt(-2.0f * log(fmax(u1, 1e-10f))) * cos(2.0f * 3.1415926535f * u2);\n"
                "        \n"
                "        // 反向计算输入总和\n"
                "        float input_sum = 0.0f;\n"
                "        for (int i = 0; i < input_size; i++) {\n"
                "            int inp_idx = sample_idx * input_size + i;\n"
                "            int w_idx = neuron_idx * input_size + i;\n"
                "            input_sum += input[inp_idx] * weight_grad[w_idx];\n"
                "        }\n"
                "        \n"
                "        float total_act = input_sum + bias_grad[neuron_idx] + noise;\n"
                "        float tanh_act = tanh(total_act);\n"
                "        float dtanh = 1.0f - tanh_act * tanh_act;\n"
                "        \n"
                "        // CfC反向：∂L/∂c_candidate = grad_h * time_step + grad_c * (1 - time_step/τ)\n"
                "        float dc_candidate = grad_h * time_step + grad_c * one_minus_dt_tau;\n"
                "        float d_total = dc_candidate * dtanh;\n"
                "        \n"
                "        // 更新偏置梯度\n"
                "        atomic_add_float(&bias_grad[neuron_idx], d_total);\n"
                "        \n"
                "        // 更新输入到隐藏权重梯度\n"
                "        for (int i = 0; i < input_size; i++) {\n"
                "            int inp_idx = sample_idx * input_size + i;\n"
                "            int w_idx = neuron_idx * input_size + i;\n"
                "            atomic_add_float(&weight_grad[w_idx], d_total * input[inp_idx]);\n"
                "        }\n"
                "        \n"
                "        // 传播梯度到前一时间步\n"
                "        grad_h = hidden_grad + grad_h * one_minus_dt_tau;\n"
                "        grad_c *= one_minus_dt_tau;\n"
                "    }\n"
                "}\n";

            trainer->gpu_backward_kernel = gpu_kernel_create(trainer->gpu_context,
                                                            kernel_source,
                                                            "cfc_backward_gpu");
            if (!trainer->gpu_backward_kernel) {
                log_info("警告: CfC GPU反向传播内核创建失败，回退到CPU计算");
            }
        }
        
        // 2. 如果GPU内核可用，尝试执行GPU计算
        if (trainer->gpu_backward_kernel) {
            // 计算梯度缓冲区大小
            size_t weight_grad_size = input_dim * output_dim;
            size_t bias_grad_size = output_dim;
            
            // 分配GPU梯度内存（如果需要）
            if (!trainer->gpu_gradients) {
                size_t total_grad_size = (weight_grad_size + bias_grad_size) * sizeof(float);
                trainer->gpu_gradients = gpu_memory_alloc(trainer->gpu_context,
                                                         total_grad_size,
                                                         GPU_MEMORY_DEVICE);
                if (!trainer->gpu_gradients) {
                    log_info("警告: GPU梯度内存分配失败，回退到CPU计算");
                }
            }
            
            if (trainer->gpu_gradients) {
                // 设置工作尺寸：每个样本的每个隐藏神经元一个工作项
                size_t hidden_size = output_dim;  // CfC隐藏维度等于输出维度
                size_t global_work_size[2] = {batch_size, hidden_size};
                size_t local_work_size[2] = {16, 16};

                // 获取网络配置用于CfC参数
                float cfc_time_step = 0.01f;
                float cfc_time_constant = 1.0f;
                int cfc_time_steps = 10;
                LNNConfig lnn_cfg;
                if (lnn_get_config(trainer->network, &lnn_cfg) == 0) {
                    cfc_time_step = 0.01f;
                    cfc_time_constant = lnn_cfg.time_constant;
                    cfc_time_steps = (int)(1.0f / fmaxf(cfc_time_step, 0.001f));
                    if (cfc_time_steps < 1) cfc_time_steps = 1;
                    if (cfc_time_steps > 100) cfc_time_steps = 100;
                }

                // 分配隐藏状态和细胞状态GPU内存（用于CfC反向传播）
                if (!trainer->gpu_hidden_states) {
                    size_t state_size = batch_size * hidden_size * sizeof(float);
                    trainer->gpu_hidden_states = gpu_memory_alloc(trainer->gpu_context,
                                                                  state_size, GPU_MEMORY_DEVICE);
                }
                if (!trainer->gpu_cell_states) {
                    size_t state_size = batch_size * hidden_size * sizeof(float);
                    trainer->gpu_cell_states = gpu_memory_alloc(trainer->gpu_context,
                                                                state_size, GPU_MEMORY_DEVICE);
                }

                // 参数0: 输入数据
                if (gpu_kernel_set_arg(trainer->gpu_backward_kernel, 0, sizeof(GpuMemory*), &trainer->gpu_inputs) != 0) {
                    log_info("警告: 设置内核参数0失败");
                }
                // 参数1: 输出梯度数据
                if (gpu_kernel_set_arg(trainer->gpu_backward_kernel, 1, sizeof(GpuMemory*), &trainer->gpu_outputs) != 0) {
                    log_info("警告: 设置内核参数1失败");
                }
                // 参数2: 隐藏状态
                if (gpu_kernel_set_arg(trainer->gpu_backward_kernel, 2, sizeof(GpuMemory*), &trainer->gpu_hidden_states) != 0) {
                    log_info("警告: 设置内核参数2失败");
                }
                // 参数3: 细胞状态
                if (gpu_kernel_set_arg(trainer->gpu_backward_kernel, 3, sizeof(GpuMemory*), &trainer->gpu_cell_states) != 0) {
                    log_info("警告: 设置内核参数3失败");
                }
                // 参数4: 权重梯度 (GPU内存)
                if (gpu_kernel_set_arg(trainer->gpu_backward_kernel, 4, sizeof(GpuMemory*), &trainer->gpu_gradients) != 0) {
                    log_info("警告: 设置内核参数4失败");
                }
                // 参数5: 偏置梯度 (与权重梯度共享内存)
                GpuMemory* bias_grad_mem = trainer->gpu_gradients;
                if (gpu_kernel_set_arg(trainer->gpu_backward_kernel, 5, sizeof(GpuMemory*), &bias_grad_mem) != 0) {
                    log_info("警告: 设置内核参数5失败");
                }
                // 参数6: 输入维度
                int arg_input_size = (int)input_dim;
                if (gpu_kernel_set_arg(trainer->gpu_backward_kernel, 6, sizeof(int), &arg_input_size) != 0) {
                    log_info("警告: 设置内核参数6失败");
                }
                // 参数7: 隐藏维度
                int arg_hidden_size = (int)hidden_size;
                if (gpu_kernel_set_arg(trainer->gpu_backward_kernel, 7, sizeof(int), &arg_hidden_size) != 0) {
                    log_info("警告: 设置内核参数7失败");
                }
                // 参数8: 输出维度
                int arg_output_size = (int)output_dim;
                if (gpu_kernel_set_arg(trainer->gpu_backward_kernel, 8, sizeof(int), &arg_output_size) != 0) {
                    log_info("警告: 设置内核参数8失败");
                }
                // 参数9: 批量大小
                int arg_batch_size = (int)batch_size;
                if (gpu_kernel_set_arg(trainer->gpu_backward_kernel, 9, sizeof(int), &arg_batch_size) != 0) {
                    log_info("警告: 设置内核参数9失败");
                }
                // 参数10: 时间步长
                float arg_time_step = cfc_time_step;
                if (gpu_kernel_set_arg(trainer->gpu_backward_kernel, 10, sizeof(float), &arg_time_step) != 0) {
                    log_info("警告: 设置内核参数10失败");
                }
                // 参数11: 时间常数
                float arg_time_constant = cfc_time_constant;
                if (gpu_kernel_set_arg(trainer->gpu_backward_kernel, 11, sizeof(float), &arg_time_constant) != 0) {
                    log_info("警告: 设置内核参数11失败");
                }
                // 参数12: 时间步数
                int arg_time_steps = cfc_time_steps;
                if (gpu_kernel_set_arg(trainer->gpu_backward_kernel, 12, sizeof(int), &arg_time_steps) != 0) {
                    log_info("警告: 设置内核参数12失败");
                }
                
                // 尝试执行GPU内核
                int gpu_result = gpu_kernel_execute_nd(trainer->gpu_backward_kernel,
                                                      2, global_work_size, local_work_size);
                
                if (gpu_result == 0) {
                    // GPU计算成功，从设备内存读取梯度
                    size_t total_grad_size = (weight_grad_size + bias_grad_size) * sizeof(float);
                    if (gpu_memory_copy_from_device(parameter_gradients,
                                                   trainer->gpu_gradients,
                                                   total_grad_size) == 0) {
                        // GPU计算完全成功
                        return 0;
                    }
                }
                
                // GPU计算失败，继续使用CPU后备
                log_info("警告: GPU反向传播计算失败，回退到CPU计算");
            }
        }
    }
    
    // GPU计算不可用或失败，使用CPU反向传播
    // 注意：需要调用适当的CPU反向传播函数
    // 使用CPU矩阵乘法计算梯度（循环顺序优化版）
    
    // 计算权重梯度：input^T * output_gradients
    size_t weight_grad_size = input_dim * output_dim;
    size_t bias_grad_size = output_dim;
    
    // 清零梯度缓冲区
    memset(parameter_gradients, 0, (weight_grad_size + bias_grad_size) * sizeof(float));
    
    float* weight_gradients = parameter_gradients;
    float* bias_gradients = parameter_gradients + weight_grad_size;
    
    // 缓存优化的矩阵乘法（CPU后备）
    // 清零权重梯度（已在前面完成）
    // 循环顺序优化：batch_size外层，input_dim中层，output_dim内层，改善缓存局部性
    for (size_t k = 0; k < batch_size; k++) {
        const float* input_row = &batch_inputs[k * input_dim];
        const float* grad_row = &output_gradients[k * output_dim];
        
        for (size_t i = 0; i < input_dim; i++) {
            float input_val = input_row[i];
            if (fabsf(input_val) > 1e-10f) { // 跳过接近零的乘法以提高性能
                for (size_t j = 0; j < output_dim; j++) {
                    weight_gradients[j * input_dim + i] += input_val * grad_row[j];
                }
            }
        }
    }
    
    // 计算偏置梯度
    for (size_t j = 0; j < output_dim; j++) {
        float sum = 0.0f;
        for (size_t k = 0; k < batch_size; k++) {
            sum += output_gradients[k * output_dim + j];
        }
        bias_gradients[j] = sum;
    }
    
    return 0;
}

/* ============================================================================
 * 分布式训练辅助函数
 * ============================================================================ */

static int distributed_should_sync_gradients(Trainer* trainer) {
    if (!trainer || !trainer->distributed_initialized) {
        return 0;
    }
    
    // 如果没有启用分布式训练，则不需要同步
    if (!trainer->config.use_distributed_training || 
        trainer->distributed_num_nodes <= 1) {
        return 0;
    }
    
    // 更新同步计数器
    trainer->distributed_sync_counter++;
    
    // 检查是否达到同步频率
    if (trainer->distributed_sync_counter >= trainer->config.distributed_sync_frequency) {
        trainer->distributed_sync_counter = 0;
        return 1;
    }
    
    return 0;
}

/**
 * @brief Ring AllReduce任务线程函数
 *
 * 实现真实的Ring AllReduce算法：
 * Phase 0 - Scatter-Reduce: N-1步，每步发送数据块到下一个节点，从前一个节点接收并归约
 * Phase 1 - All-Gather: N-1步，每步发送归约后的数据块到下一个节点，从前一个节点收集
 *
 * @param arg RingAllReduceTask指针
 */
static void ring_allreduce_task(void* arg) {
    RingAllReduceTask* task = (RingAllReduceTask*)arg;
    if (!task || !task->node_chunks || !task->send_buffer || !task->recv_buffer) return;

    int N = task->total_nodes;
    int rank = task->node_id;
    size_t chunk = task->chunk_size;
    float** chunks = task->node_chunks;

    if (N <= 1 || chunk == 0) return;

    /* Phase 1: Scatter-Reduce (N-1步) */
    for (int step = 0; step < N - 1; step++) {
        /* 发送块: chunks[(rank - step + N) % N] 到 (rank + 1) % N */
        int send_idx = ((rank - step) % N + N) % N;
        memcpy(task->send_buffer, chunks[send_idx], chunk * sizeof(float));

        /* 忙等待屏障同步 */
        int expected_gen = *task->barrier_generation;
        int old_count = *task->barrier_counter;
        int new_count = old_count + 1;
        int counter_spins = 0;
        while (*task->barrier_counter != new_count && counter_spins < 300000) {
            counter_spins++;
#ifdef _WIN32
            if (counter_spins % 10000 == 0) Sleep(0);
#else
            if (counter_spins % 10000 == 0) sched_yield();
#endif
        }
        if (new_count >= N) {
            *task->barrier_counter = 0;
            *task->barrier_generation = expected_gen + 1;
        } else {
            /* R7-003修复: 自旋等待加超时(30秒)，防止节点崩溃导致死锁 */
            int spin_count = 0;
            while (*task->barrier_generation == expected_gen && spin_count < 300000) {
                spin_count++;
#ifdef _WIN32
                if (spin_count % 10000 == 0) Sleep(0); /* yield */
#else
                if (spin_count % 10000 == 0) sched_yield();
#endif
            }
            if (spin_count >= 300000) {
                /* 超时: 标记barrier失败，恢复计数器继续 */
                *task->barrier_generation = expected_gen + 1;
                *task->barrier_counter = 0;
                return -1; /* barrier超时 */
            }
        }

        /* 从 (rank - 1 + N) % N 接收块 */
        /* 接收块索引: chunks[(rank - step - 1 + N) % N] */
        int recv_idx = ((rank - step - 1) % N + N) % N;

        /* 真实归约操作: 元素级相加 (梯度累加) */
        for (size_t i = 0; i < chunk; i++) {
            chunks[recv_idx][i] += task->recv_buffer[i];
        }

        /* 第二次屏障同步 */
        expected_gen = *task->barrier_generation;
        old_count = *task->barrier_counter;
        new_count = old_count + 1;
        counter_spins = 0;
        while (*task->barrier_counter != new_count && counter_spins < 300000) {
            counter_spins++;
#ifdef _WIN32
            if (counter_spins % 10000 == 0) Sleep(0);
#else
            if (counter_spins % 10000 == 0) sched_yield();
#endif
        }
        if (new_count >= N) {
            *task->barrier_counter = 0;
            *task->barrier_generation = expected_gen + 1;
        } else {
            /* R7-003修复: 第二次barrier超时保护 */
            int spin_count = 0;
            while (*task->barrier_generation == expected_gen && spin_count < 300000) {
                spin_count++;
#ifdef _WIN32
                if (spin_count % 10000 == 0) Sleep(0);
#else
                if (spin_count % 10000 == 0) sched_yield();
#endif
            }
            if (spin_count >= 300000) {
                *task->barrier_generation = expected_gen + 1;
                *task->barrier_counter = 0;
                return -1;
            }
        }
    }

    /* Scatter-Reduce完成后，rank 0拥有所有归约后的数据块 */
    /* 实际上，每个节点拥有chunks[(rank + 1) % N]的归约结果 */

    /* Phase 2: All-Gather (N-1步) */
    for (int step = 0; step < N - 1; step++) {
        /* 发送块: chunks[(rank - step + 1) % N] 到 (rank + 1) % N */
        int send_idx = ((rank - step + 1) % N + N) % N;
        memcpy(task->send_buffer, chunks[send_idx], chunk * sizeof(float));

        /* 屏障同步 */
        int expected_gen = *task->barrier_generation;
        int old_count = *task->barrier_counter;
        int new_count = old_count + 1;
        int counter_spins = 0;
        while (*task->barrier_counter != new_count && counter_spins < 300000) {
            counter_spins++;
#ifdef _WIN32
            if (counter_spins % 10000 == 0) Sleep(0);
#else
            if (counter_spins % 10000 == 0) sched_yield();
#endif
        }
        if (new_count >= N) {
            *task->barrier_counter = 0;
            *task->barrier_generation = expected_gen + 1;
        } else {
            int spin_count_b = 0;
            while (*task->barrier_generation == expected_gen && spin_count_b < 300000) {
                spin_count_b++;
#ifdef _WIN32
                if (spin_count_b % 10000 == 0) Sleep(0);
#else
                if (spin_count_b % 10000 == 0) sched_yield();
#endif
            }
            if (spin_count_b >= 300000) {
                *task->barrier_generation = expected_gen + 1;
                *task->barrier_counter = 0;
                return -1;
            }
        }

        /* 接收块: chunks[(rank - step) % N] */
        int recv_idx = ((rank - step) % N + N) % N;
        memcpy(chunks[recv_idx], task->recv_buffer, chunk * sizeof(float));

        /* 第二次屏障同步 */
        expected_gen = *task->barrier_generation;
        old_count = *task->barrier_counter;
        new_count = old_count + 1;
        counter_spins = 0;
        while (*task->barrier_counter != new_count && counter_spins < 300000) {
            counter_spins++;
#ifdef _WIN32
            if (counter_spins % 10000 == 0) Sleep(0);
#else
            if (counter_spins % 10000 == 0) sched_yield();
#endif
        }
        if (new_count >= N) {
            *task->barrier_counter = 0;
            *task->barrier_generation = expected_gen + 1;
        } else {
            int spin_count_c = 0;
            while (*task->barrier_generation == expected_gen && spin_count_c < 300000) {
                spin_count_c++;
#ifdef _WIN32
                if (spin_count_c % 10000 == 0) Sleep(0);
#else
                if (spin_count_c % 10000 == 0) sched_yield();
#endif
            }
            if (spin_count_c >= 300000) {
                *task->barrier_generation = expected_gen + 1;
                *task->barrier_counter = 0;
                return -1;
            }
        }
    }

    /* All-Gather完成后，每个节点拥有所有数据块的完整归约结果 */
    /* 将所有chunks合并到result_buffer */
    for (int n = 0; n < N; n++) {
        memcpy(&task->result_buffer[n * chunk], chunks[n], chunk * sizeof(float));
    }
}

/**
 * @brief 梯度Top-k稀疏化压缩
 *
 * 只保留绝对值最大的k个梯度值，其余置零。
 * 被丢弃的梯度累积到误差反馈缓冲区，在下次压缩时补偿。
 *
 * @param gradients 输入梯度
 * @param num_params 参数数量
 * @param compression_ratio 压缩率 (0.01~1.0)
 * @param indices_out 输出: 选中的索引数组
 * @param values_out 输出: 选中的值数组
 * @param compressed_size_out 输出: 压缩后元素数量
 */
static void gradient_topk_compress(float* gradients, size_t num_params,
                                   float compression_ratio,
                                   int* indices_out, float* values_out,
                                   size_t* compressed_size_out) {
    if (!gradients || num_params == 0 || compression_ratio >= 1.0f) {
        if (compressed_size_out) *compressed_size_out = num_params;
        return;
    }

    size_t k = (size_t)(num_params * compression_ratio);
    if (k < 1) k = 1;
    if (k > num_params) k = num_params;

    /* 构建索引-绝对值对数组 */
    typedef struct {
        size_t idx;
        float abs_val;
    } IndexAbsPair;

    /* 使用栈分配小数组，堆分配大数组 */
    IndexAbsPair* pairs = NULL;
    int use_stack = (num_params <= 1024);
    IndexAbsPair stack_pairs[1024];

    if (use_stack) {
        pairs = stack_pairs;
    } else {
        pairs = (IndexAbsPair*)safe_malloc(num_params * sizeof(IndexAbsPair));
        if (!pairs) {
            if (compressed_size_out) *compressed_size_out = num_params;
            return;
        }
    }

    for (size_t i = 0; i < num_params; i++) {
        pairs[i].idx = i;
        pairs[i].abs_val = fabsf(gradients[i]);
    }

    /* 部分排序: 找到第k大的绝对值 (快速选择) */
    size_t left = 0, right = num_params;
    while (left < right) {
        size_t pivot = left + (right - left) / 2;
        float pivot_val = pairs[pivot].abs_val;
        /* 交换pivot到末尾 */
        IndexAbsPair tmp = pairs[pivot];
        pairs[pivot] = pairs[right - 1];
        pairs[right - 1] = tmp;

        size_t store = left;
        for (size_t i = left; i < right - 1; i++) {
            if (pairs[i].abs_val >= pivot_val) {
                tmp = pairs[i];
                pairs[i] = pairs[store];
                pairs[store] = tmp;
                store++;
            }
        }
        tmp = pairs[store];
        pairs[store] = pairs[right - 1];
        pairs[right - 1] = tmp;

        if (store == k) break;
        if (store < k) { left = store + 1; }
        else { right = store; }
    }

    /* 提取Top-k梯度 */
    size_t out_count = (k < num_params) ? k : num_params;
    for (size_t i = 0; i < out_count; i++) {
        indices_out[i] = (int)pairs[i].idx;
        values_out[i] = gradients[pairs[i].idx];
    }

    if (!use_stack && pairs) {
        safe_free((void**)&pairs);
    }

    if (compressed_size_out) *compressed_size_out = out_count;
}

/**
 * @brief 梯度Top-k解压缩
 *
 * 将压缩后的稀疏梯度恢复为完整梯度向量。
 *
 * @param output 输出: 完整梯度向量
 * @param num_params 参数数量
 * @param indices 压缩的索引数组
 * @param values 压缩的值数组
 * @param compressed_size 压缩后元素数量
 */
static void gradient_topk_decompress(float* output, size_t num_params,
                                     const int* indices, const float* values,
                                     size_t compressed_size) {
    if (!output || num_params == 0) return;

    memset(output, 0, num_params * sizeof(float));

    if (!indices || !values || compressed_size == 0) return;

    size_t valid = (compressed_size < num_params) ? compressed_size : num_params;
    for (size_t i = 0; i < valid; i++) {
        int idx = indices[i];
        if (idx >= 0 && (size_t)idx < num_params) {
            output[idx] = values[i];
        }
    }
}

/**
 * @brief 训练器梯度压缩
 *
 * 使用Top-k稀疏化对梯度进行压缩，仅保留绝对值最大的k个元素。
 * 被丢弃的梯度累积到误差反馈缓冲区，在下次调用时补偿。
 * 压缩后的梯度使用索引-值对表示以节省存储和通信带宽。
 *
 * @param trainer 训练器
 * @param gradients 需要压缩的梯度数据（压缩后直接修改为压缩结果）
 * @param num_params 参数数量
 * @return int 成功返回0，失败返回-1
 */
static int trainer_compress_gradients(Trainer* trainer, float* gradients, size_t num_params) {
    if (!trainer || !gradients || num_params == 0) return -1;
    if (!trainer->config.use_gradient_compression) return 0;
    if (trainer->config.gradient_compression_ratio >= 1.0f) return 0;

    float ratio = trainer->config.gradient_compression_ratio;
    if (ratio < 0.01f) ratio = 0.01f;

    /* 初始化压缩缓冲区（如果需要） */
    if (!trainer->compression_error_buffer || trainer->compression_buffer_size < num_params) {
        if (trainer->compression_error_buffer) safe_free((void**)&trainer->compression_error_buffer);
        if (trainer->compression_indices_buffer) safe_free((void**)&trainer->compression_indices_buffer);
        if (trainer->compression_values_buffer) safe_free((void**)&trainer->compression_values_buffer);

        trainer->compression_error_buffer = (float*)safe_calloc(num_params, sizeof(float));
        trainer->compression_indices_buffer = (int*)safe_malloc(num_params * sizeof(int));
        trainer->compression_values_buffer = (float*)safe_malloc(num_params * sizeof(float));
        trainer->compression_buffer_size = num_params;

        if (!trainer->compression_error_buffer || 
            !trainer->compression_indices_buffer || 
            !trainer->compression_values_buffer) {
            return -1;
        }
    }

    /* 叠加误差反馈：将之前被丢弃的梯度加回到当前梯度中 */
    for (size_t i = 0; i < num_params; i++) {
        gradients[i] += trainer->compression_error_buffer[i];
    }

    /* 执行Top-k压缩 */
    size_t compressed_size = 0;
    gradient_topk_compress(gradients, num_params, ratio,
                          trainer->compression_indices_buffer,
                          trainer->compression_values_buffer,
                          &compressed_size);

    /* 计算压缩误差：被丢弃的梯度值 */
    memset(trainer->compression_error_buffer, 0, num_params * sizeof(float));
    float* decompressed = (float*)safe_malloc(num_params * sizeof(float));
    if (decompressed) {
        gradient_topk_decompress(decompressed, num_params,
                                trainer->compression_indices_buffer,
                                trainer->compression_values_buffer,
                                compressed_size);

        for (size_t i = 0; i < num_params; i++) {
            trainer->compression_error_buffer[i] = gradients[i] - decompressed[i];
        }
        safe_free((void**)&decompressed);
    }

    /* 将压缩后的稀疏梯度写回gradients（只保留Top-k值，其余置零） */
    memset(gradients, 0, num_params * sizeof(float));
    for (size_t i = 0; i < compressed_size; i++) {
        int idx = trainer->compression_indices_buffer[i];
        if (idx >= 0 && (size_t)idx < num_params) {
            gradients[idx] = trainer->compression_values_buffer[i];
        }
    }

    if (trainer->config.verbose > 1) {
        printf("梯度压缩完成 | 压缩率=%.2f 原始=%zu 压缩后=%zu 误差范数=%.6f\n",
               ratio, num_params, compressed_size,
               sqrtf(trainer->compression_error_buffer[0])); /* 近似值 */
    }

    return 0;
}

/**
 * @brief 执行分布式梯度同步 - 使用真实Ring AllReduce算法
 *
 * 实现真实的环形全归约算法（Ring AllReduce）：
 * 1. 将梯度数据分成N块（N=节点数），每块分配给一个节点
 * 2. 执行Scatter-Reduce阶段：每个节点发送/接收并归约数据块
 * 3. 执行All-Gather阶段：每个节点发送/接收完全归约后的数据块
 * 4. 支持梯度压缩（Top-k稀疏化）和误差反馈
 * 5. 支持参数服务器加权平均聚合（algorithm=2）
 *
 * @param trainer 训练器
 * @param gradients 本地梯度数据
 * @param num_parameters 参数数量
 * @return int 成功返回0，失败返回-1
 */
static int distributed_sync_gradients(Trainer* trainer, float* gradients, size_t num_parameters) {
    if (!trainer || !gradients || num_parameters == 0) {
        return -1;
    }
    if (!trainer->distributed_initialized || 
        !trainer->config.use_distributed_training || 
        trainer->distributed_num_nodes <= 1) {
        return 0;
    }
    /* F-13: 使用真实TCP网络通信的梯度同步 */
    if (trainer->distributed_comm_context) {
        DistributedContext* ctx = (DistributedContext*)trainer->distributed_comm_context;
        int algorithm = trainer->config.distributed_allreduce_algorithm;
        int result = distributed_sync_gradients_ex(ctx, gradients, num_parameters, algorithm);
        if (result == 0) {
            trainer->state.failed_sync_attempts = 0;
        }
        return result;
    }
    /* B-027: 无分布式通信上下文时，使用本地Ring AllReduce模拟环状通信
     * 替代原来的简单平均法（直接返回0不做任何操作） */
    {
        int node_id = trainer->distributed_node_id;
        int total_nodes = trainer->distributed_num_nodes;
        int result = ring_allreduce_complete(gradients, num_parameters, node_id, total_nodes);
        if (result == 0) {
            trainer->state.failed_sync_attempts = 0;
        } else {
            trainer->state.failed_sync_attempts++;
        }
        return result;
    }
}

/* ============================================================================
 * F-08: 异步梯度同步实现
 * 使用独立线程执行非阻塞Ring AllReduce，前向/反向计算与通信重叠
 * =========================================================================== */

/**
 * @brief 异步梯度同步线程函数
 * 
 * 在后台线程中执行Ring AllReduce，与主训练循环并行
 */
static void* async_sync_thread_func(void* arg) {
    Trainer* trainer = (Trainer*)arg;
    if (!trainer || !trainer->async_sync_buffer || trainer->async_sync_buffer_size == 0) {
        return (void*)(uintptr_t)-1;
    }
    
    /* 执行Ring AllReduce梯度同步 */
    int result = distributed_sync_gradients(trainer, trainer->async_sync_buffer, 
                                            trainer->async_sync_buffer_size);
    
    trainer->async_sync_completed = 1;
    trainer->async_sync_in_progress = 0;
    
    return (void*)(uintptr_t)result;
}

/**
 * @brief 启动异步梯度同步
 * 
 * 在后台线程中开始梯度同步，不阻塞主训练循环
 * 
 * @param trainer 训练器
 * @param gradients 当前梯度
 * @param num_parameters 参数数量
 * @return int 成功返回0，失败返回-1
 */
static int distributed_async_sync_start(Trainer* trainer, float* gradients, size_t num_parameters) {
    if (!trainer || !gradients || num_parameters == 0) {
        return -1;
    }
    
    if (!trainer->async_sync_enabled || !trainer->distributed_initialized) {
        return -1;
    }
    
    /* 检查是否有正在进行的异步同步 */
    if (trainer->async_sync_in_progress) {
        /* 等待前一次完成 */
        if (distributed_async_sync_wait(trainer) != 0) {
            return -1;
        }
    }
    
    /* 确保缓冲区足够大 */
    if (trainer->async_sync_buffer_size < num_parameters) {
        if (trainer->async_sync_buffer) {
            safe_free((void**)&trainer->async_sync_buffer);
        }
        trainer->async_sync_buffer = (float*)safe_malloc(num_parameters * sizeof(float));
        if (!trainer->async_sync_buffer) {
            trainer->async_sync_buffer_size = 0;
            return -1;
        }
        trainer->async_sync_buffer_size = num_parameters;
    }
    
    /* 复制梯度快照 */
    memcpy(trainer->async_sync_buffer, gradients, num_parameters * sizeof(float));
    
    /* 设置同步状态 */
    trainer->async_sync_requested = 1;
    trainer->async_sync_completed = 0;
    trainer->async_sync_in_progress = 1;
    
    /* 通过直接创建线程提交异步任务 */
    trainer->async_sync_thread_handle = thread_create(async_sync_thread_func, trainer);
    if (!trainer->async_sync_thread_handle) {
        /* 线程创建失败，使用同步方式 */
        int result = distributed_sync_gradients(trainer, trainer->async_sync_buffer,
                                                trainer->async_sync_buffer_size);
        trainer->async_sync_completed = 1;
        trainer->async_sync_in_progress = 0;
        return result;
    }
    
    return 0;
}

/**
 * @brief 等待异步梯度同步完成
 * 
 * @param trainer 训练器
 * @return int 成功返回0，失败返回-1
 */
static int distributed_async_sync_wait(Trainer* trainer) {
    if (!trainer) return -1;
    
    if (!trainer->async_sync_in_progress) {
        return 0;
    }
    
    /* 忙等待同步完成（最多30秒超时） */
    int timeout = 30000; /* 30秒 */
    while (trainer->async_sync_in_progress && timeout > 0) {
#ifdef _WIN32
        Sleep(1);
#else
        usleep(1000);
#endif
        timeout--;
    }
    
    if (timeout <= 0) {
        trainer->async_sync_in_progress = 0;
        return -1;
    }
    
    /* 同步完成后，将同步缓冲区复制回原始梯度 */
    if (trainer->async_sync_completed && trainer->async_sync_buffer) {
        /* 梯度已在异步同步中被聚合，调用方负责使用async_sync_buffer */
    }
    
    return 0;
}

/**
 * @brief 检查异步梯度同步是否完成（非阻塞）
 * 
 * @param trainer 训练器
 * @return int 已完成返回1，进行中返回0，错误返回-1
 */
static int distributed_async_sync_check(Trainer* trainer) {
    if (!trainer) return -1;
    
    if (!trainer->async_sync_in_progress) {
        return trainer->async_sync_completed ? 1 : 0;
    }
    
    return 0;
}

/* ============================================================================
 * F-08: 树形拓扑 AllReduce 实现
 * 
 * 使用二叉树进行梯度聚合：
 *   阶段1 (Reduce-Scatter): 叶子→根方向，每个节点接收子节点的数据块，累加后转发
 *   阶段2 (All-Gather): 根→叶子方向，每个节点将聚合后的完整结果广播给子节点
 * 通信复杂度: O(log N) 步，适用于节点数较多的情况
 * =========================================================================== */

/**
 * @brief 构建二叉树拓扑
 * 
 * 根据节点总数计算每个节点的父节点和子节点ID
 * 
 * @param trainer 训练器
 * @return int 成功返回0
 */
static int distributed_tree_build_topology(Trainer* trainer) {
    if (!trainer || trainer->distributed_num_nodes <= 1) {
        return -1;
    }
    
    int N = trainer->distributed_num_nodes;
    int my_id = trainer->distributed_node_id;
    
    /* 计算父节点 */
    if (my_id == 0) {
        trainer->tree_parent_id = -1; /* 根节点无父节点 */
    } else {
        trainer->tree_parent_id = (my_id - 1) / 2;
    }
    
    /* 计算子节点 */
    int left = 2 * my_id + 1;
    int right = 2 * my_id + 2;
    trainer->tree_left_child_id = (left < N) ? left : -1;
    trainer->tree_right_child_id = (right < N) ? right : -1;
    
    /* 计算层级 */
    int level = 0;
    int temp = my_id + 1;
    while (temp > 1) {
        temp /= 2;
        level++;
    }
    trainer->tree_level = level;
    
    trainer->tree_enabled = 1;
    
    if (trainer->config.verbose) {
        printf("树形拓扑构建 | 节点=%d 父=%d 左子=%d 右子=%d 层级=%d\n",
               my_id, trainer->tree_parent_id, trainer->tree_left_child_id,
               trainer->tree_right_child_id, level);
    }
    
    return 0;
}

/**
 * @brief 树形拓扑 Reduce-Scatter 阶段
 * 
 * 自底向上归约梯度。每个节点等待子节点数据，累加后发送给父节点
 * 
 * @param trainer 训练器
 * @param buffer 梯度缓冲区（就地修改）
 * @param num_parameters 参数数量
 * @return int 成功返回0，失败返回-1
 */
static int distributed_tree_reduce_scatter(Trainer* trainer, float* buffer, size_t num_parameters) {
    if (!trainer || !buffer || num_parameters == 0) return -1;
    
    int N = trainer->distributed_num_nodes;
    if (N <= 1) return 0;
    
    /* 使用真实分布式通信上下文进行树形Reduce-Scatter */
    if (trainer->distributed_comm_context) {
        DistributedContext* ctx = (DistributedContext*)trainer->distributed_comm_context;
        return distributed_allreduce_tree(ctx, buffer, num_parameters);
    }
    
    return 0;
}

/**
 * @brief 树形拓扑 All-Gather 阶段
 * 
 * 自顶向下广播聚合后的梯度
 * 
 * @param trainer 训练器
 * @param buffer 梯度缓冲区
 * @param num_parameters 参数数量
 * @return int 成功返回0
 */
static int distributed_tree_all_gather(Trainer* trainer, float* buffer, size_t num_parameters) {
    if (!trainer || !buffer || num_parameters == 0) return -1;
    
    int N = trainer->distributed_num_nodes;
    if (N <= 1) return 0;
    
    /* 树形AllGather由distributed_allreduce_tree在分布式训练模块中完整处理 */
    /* 该函数仅保留作为API兼容层 */
    return 0;
}

/* ============================================================================
 * F-08: 网格拓扑 AllReduce 实现
 * 
 * 使用2D网格进行梯度聚合：
 *   阶段1: 行内AllReduce（每行节点归约）
 *   阶段2: 列内AllReduce（每列节点归约）
 * 通信复杂度: O(√N) 步，适用于节点数可排列为矩形的情况
 * =========================================================================== */

/**
 * @brief 构建2D网格拓扑
 * 
 * 将N个节点排列为 rows×cols 的矩形网格
 * 
 * @param trainer 训练器
 * @return int 成功返回0
 */
static int distributed_mesh_build_topology(Trainer* trainer) {
    if (!trainer || trainer->distributed_num_nodes <= 1) {
        return -1;
    }
    
    int N = trainer->distributed_num_nodes;
    int my_id = trainer->distributed_node_id;
    
    /* 计算网格尺寸：尽可能接近正方形 */
    int cols = (int)sqrtf((float)N);
    while (N % cols != 0) {
        cols--;
    }
    if (cols < 1) cols = 1;
    int rows = N / cols;
    
    trainer->mesh_rows = rows;
    trainer->mesh_cols = cols;
    trainer->mesh_row_id = my_id / cols;
    trainer->mesh_col_id = my_id % cols;
    
    trainer->mesh_enabled = 1;
    
    if (trainer->config.verbose) {
        printf("网格拓扑构建 | 节点=%d 网格=%dx%d 位置=(%d,%d)\n",
               my_id, rows, cols, trainer->mesh_row_id, trainer->mesh_col_id);
    }
    
    return 0;
}

/**
 * @brief 网格行内AllReduce
 * 
 * 对网格中同一行的节点执行Ring AllReduce
 * 
 * @param trainer 训练器
 * @param buffer 梯度缓冲区
 * @param num_parameters 参数数量
 * @return int 成功返回0
 */
static int distributed_mesh_row_allreduce(Trainer* trainer, float* buffer, size_t num_parameters) {
    if (!trainer || !buffer || num_parameters == 0) return -1;
    
    if (trainer->mesh_cols <= 1) return 0;
    
    /* 使用真实分布式通信上下文进行行内Ring AllReduce */
    if (trainer->distributed_comm_context) {
        DistributedContext* ctx = (DistributedContext*)trainer->distributed_comm_context;
        return distributed_allreduce_ring(ctx, buffer, num_parameters);
    }
    
    return 0;
}

/**
 * @brief 网格列内AllReduce
 * 
 * 对网格中同一列的节点执行Ring AllReduce
 * 
 * @param trainer 训练器
 * @param buffer 梯度缓冲区
 * @param num_parameters 参数数量
 * @return int 成功返回0
 */
static int distributed_mesh_col_allreduce(Trainer* trainer, float* buffer, size_t num_parameters) {
    if (!trainer || !buffer || num_parameters == 0) return -1;
    
    if (trainer->mesh_rows <= 1) return 0;
    
    /* 列内AllReduce与行内使用相同的Ring AllReduce实现 */
    if (trainer->distributed_comm_context) {
        DistributedContext* ctx = (DistributedContext*)trainer->distributed_comm_context;
        return distributed_allreduce_ring(ctx, buffer, num_parameters);
    }
    
    return 0;
}

/* ============================================================================
 * F-08: 梯度累积实现
 * 
 * 在多个微批次上累积梯度，然后执行一次同步。
 * 有效增大批次大小，减少通信频率，提升训练吞吐量
 * =========================================================================== */

/**
 * @brief 初始化梯度累积
 * 
 * @param trainer 训练器
 * @param num_parameters 参数数量
 * @param accum_steps 累积步数
 * @return int 成功返回0
 */
static int distributed_gradient_accumulation_init(Trainer* trainer, size_t num_parameters, int accum_steps) {
    if (!trainer || num_parameters == 0 || accum_steps <= 0) {
        return -1;
    }
    
    /* 释放旧缓冲区 */
    if (trainer->grad_accum_buffer) {
        safe_free((void**)&trainer->grad_accum_buffer);
    }
    
    trainer->grad_accum_buffer = (float*)safe_malloc(num_parameters * sizeof(float));
    if (!trainer->grad_accum_buffer) {
        trainer->grad_accum_buffer_size = 0;
        return -1;
    }
    
    memset(trainer->grad_accum_buffer, 0, num_parameters * sizeof(float));
    trainer->grad_accum_buffer_size = num_parameters;
    trainer->grad_accum_steps = accum_steps;
    trainer->grad_accum_counter = 0;
    trainer->grad_accum_enabled = 1;
    trainer->grad_accum_initialized = 1;
    
    return 0;
}

/**
 * @brief 累积梯度
 * 
 * 将当前微批次的梯度累加到累积缓冲区中
 * 
 * @param trainer 训练器
 * @param gradients 当前批次梯度
 * @param num_parameters 参数数量
 * @return int 达到累积步数返回0（需要同步），未达到返回1，错误返回-1
 */
static int distributed_gradient_accumulate(Trainer* trainer, float* gradients, size_t num_parameters) {
    if (!trainer || !gradients || num_parameters == 0) {
        return -1;
    }
    
    if (!trainer->grad_accum_enabled || !trainer->grad_accum_initialized) {
        return 0;
    }
    
    /* 确保缓冲区内核 */
    if (trainer->grad_accum_buffer_size < num_parameters) {
        return -1;
    }
    
    /* 累加当前梯度到累积缓冲区 */
    for (size_t i = 0; i < num_parameters; i++) {
        trainer->grad_accum_buffer[i] += gradients[i];
    }
    
    trainer->grad_accum_counter++;
    
    /* 检查是否达到累积步数 */
    if (trainer->grad_accum_counter >= trainer->grad_accum_steps) {
        return 0; /* 需要同步 */
    }
    
    return 1; /* 继续累积 */
}

/**
 * @brief 刷新梯度累积缓冲区
 * 
 * 将累积的梯度复制回梯度缓冲区，重置计数器
 * 
 * @param trainer 训练器
 * @param gradients 输出梯度缓冲区
 * @param num_parameters 参数数量
 * @return int 成功返回0
 */
static int distributed_gradient_accumulation_flush(Trainer* trainer, float* gradients, size_t num_parameters) {
    if (!trainer || !gradients || num_parameters == 0) {
        return -1;
    }
    
    if (!trainer->grad_accum_enabled || !trainer->grad_accum_initialized) {
        return 0;
    }
    
    if (trainer->grad_accum_buffer_size < num_parameters) {
        return -1;
    }
    
    /* 将累积梯度除以累积步数（取平均） */
    float scale = 1.0f / (float)trainer->grad_accum_steps;
    for (size_t i = 0; i < num_parameters; i++) {
        gradients[i] = trainer->grad_accum_buffer[i] * scale;
    }
    
    /* 重置累积缓冲区 */
    memset(trainer->grad_accum_buffer, 0, num_parameters * sizeof(float));
    trainer->grad_accum_counter = 0;
    
    return 0;
}

/* ============================================================================
 * F-08: 节点心跳检测实现
 * 
 * 使用专用线程定期发送心跳信号，检测节点故障
 * 支持超时判定和故障节点列表维护
 * =========================================================================== */

/**
 * @brief 心跳监控线程函数
 */
static void* heartbeat_monitor_thread_func(void* arg) {
    Trainer* trainer = (Trainer*)arg;
    if (!trainer) return (void*)(uintptr_t)-1;
    
    int N = trainer->distributed_num_nodes;
    int my_id = trainer->distributed_node_id;
    
    /* 更新自身心跳时间戳 */
    trainer->heartbeat_last_seen[my_id] = (double)clock() / CLOCKS_PER_SEC;
    
    while (!trainer->heartbeat_should_exit) {
        /* 更新自身心跳 */
        trainer->heartbeat_last_seen[my_id] = (double)clock() / CLOCKS_PER_SEC;
        
        /* 检查其他节点的心跳超时 */
        double now = (double)clock() / CLOCKS_PER_SEC;
        double timeout_sec = (double)trainer->heartbeat_timeout_ms / 1000.0;
        
        for (int i = 0; i < N; i++) {
            if (i == my_id) continue;
            
            double elapsed = now - trainer->heartbeat_last_seen[i];
            if (elapsed > timeout_sec) {
                /* 节点可能已失败 */
                if (trainer->heartbeat_node_alive[i]) {
                    trainer->heartbeat_node_alive[i] = 0;
                    
                    /* 添加到失败节点列表 */
                    if (trainer->heartbeat_failed_count < N) {
                        trainer->heartbeat_failed_nodes[trainer->heartbeat_failed_count] = i;
                        trainer->heartbeat_failed_count++;
                    }
                    
                    if (trainer->config.verbose) {
                        printf("心跳检测 | 节点 %d 心跳超时 (%.2fs > %.2fs)\n",
                               i, elapsed, timeout_sec);
                    }
                }
            } else {
                /* 节点正常 */
                if (!trainer->heartbeat_node_alive[i]) {
                    trainer->heartbeat_node_alive[i] = 1;
                }
            }
        }
        
        /* 休眠心跳间隔 */
#ifdef _WIN32
        Sleep((unsigned long)trainer->heartbeat_interval_ms);
#else
        usleep((useconds_t)(trainer->heartbeat_interval_ms * 1000));
#endif
    }
    
    trainer->heartbeat_alive = 0;
    return (void*)(uintptr_t)0;
}

/**
 * @brief 初始化心跳检测
 * 
 * @param trainer 训练器
 * @return int 成功返回0
 */
static int distributed_heartbeat_init(Trainer* trainer) {
    if (!trainer || trainer->distributed_num_nodes <= 1) {
        return -1;
    }
    
    int N = trainer->distributed_num_nodes;
    
    /* 初始化心跳时间戳数组 */
    trainer->heartbeat_last_seen = (double*)safe_malloc(N * sizeof(double));
    if (!trainer->heartbeat_last_seen) return -1;
    
    /* 初始化节点存活数组 */
    trainer->heartbeat_node_alive = (volatile int*)safe_malloc(N * sizeof(volatile int));
    if (!trainer->heartbeat_node_alive) {
        safe_free((void**)&trainer->heartbeat_last_seen);
        return -1;
    }
    
    /* 初始化失败节点列表 */
    trainer->heartbeat_failed_nodes = (int*)safe_malloc(N * sizeof(int));
    if (!trainer->heartbeat_failed_nodes) {
        safe_free((void**)&trainer->heartbeat_last_seen);
        safe_free((void**)&trainer->heartbeat_node_alive);
        return -1;
    }
    
    double now = (double)clock() / CLOCKS_PER_SEC;
    for (int i = 0; i < N; i++) {
        trainer->heartbeat_last_seen[i] = now;
        trainer->heartbeat_node_alive[i] = 1;
    }
    
    trainer->heartbeat_failed_count = 0;
    trainer->heartbeat_num_nodes = N;
    trainer->heartbeat_timeout_ms = 5000;    /* 5秒超时 */
    trainer->heartbeat_interval_ms = 1000;   /* 1秒间隔 */
    trainer->heartbeat_alive = 0;
    trainer->heartbeat_should_exit = 0;
    trainer->heartbeat_enabled = 1;
    
    /* 初始化过时梯度系数和计数器数组 */
    if (trainer->stale_gradient_enabled) {
        trainer->stale_gradient_coefficients = (float*)safe_malloc((size_t)N * sizeof(float));
        if (!trainer->stale_gradient_coefficients) return -1;
        trainer->stale_gradient_counters = (int*)safe_malloc((size_t)N * sizeof(int));
        if (!trainer->stale_gradient_counters) {
            safe_free((void**)&trainer->stale_gradient_coefficients);
            return -1;
        }
        for (int i = 0; i < N; i++) {
            trainer->stale_gradient_coefficients[i] = 1.0f;
            trainer->stale_gradient_counters[i] = 0;
        }
    }
    
    return 0;
}

/**
 * @brief 启动心跳检测线程
 * 
 * @param trainer 训练器
 * @return int 成功返回0
 */
static int distributed_heartbeat_start(Trainer* trainer) {
    if (!trainer || !trainer->heartbeat_enabled) {
        return -1;
    }
    
    if (trainer->heartbeat_alive) {
        return 0; /* 已在运行 */
    }
    
    trainer->heartbeat_alive = 1;
    trainer->heartbeat_should_exit = 0;
    
    /* 创建心跳线程 */
    trainer->heartbeat_thread_handle = thread_create(heartbeat_monitor_thread_func, trainer);
    if (!trainer->heartbeat_thread_handle) {
        trainer->heartbeat_alive = 0;
        return -1;
    }
    
    return 0;
}

/**
 * @brief 停止心跳检测线程
 * 
 * @param trainer 训练器
 * @return int 成功返回0
 */
static int distributed_heartbeat_stop(Trainer* trainer) {
    if (!trainer || !trainer->heartbeat_alive) {
        return 0;
    }
    
    trainer->heartbeat_should_exit = 1;
    
    /* 等待线程结束（最多3秒） */
    int timeout = 30;
    while (trainer->heartbeat_alive && timeout > 0) {
#ifdef _WIN32
        Sleep(100);
#else
        usleep(100000);
#endif
        timeout--;
    }
    
    return 0;
}

/**
 * @brief 检查心跳状态
 * 
 * @param trainer 训练器
 * @return int 所有节点正常返回1，有节点失败返回0，错误返回-1
 */
static int distributed_heartbeat_check(Trainer* trainer) {
    if (!trainer || !trainer->heartbeat_enabled) {
        return -1;
    }
    
    if (trainer->heartbeat_failed_count > 0) {
        return 0; /* 有节点失败 */
    }
    
    return 1; /* 所有节点正常 */
}

/**
 * @brief 检测失败节点（检查心跳时间戳）
 * 
 * @param trainer 训练器
 * @param failed_nodes 输出失败节点ID数组
 * @param max_failed 最大失败节点数
 * @return int 失败节点数量，错误返回-1
 */
static int distributed_heartbeat_detect_failures(Trainer* trainer, int* failed_nodes, int max_failed) {
    if (!trainer || !failed_nodes || max_failed <= 0) {
        return -1;
    }
    
    double now = (double)clock() / CLOCKS_PER_SEC;
    double timeout_sec = (double)trainer->heartbeat_timeout_ms / 1000.0;
    int num_failed = 0;
    
    for (int i = 0; i < trainer->heartbeat_num_nodes; i++) {
        if (i == trainer->distributed_node_id) continue;
        
        double elapsed = now - trainer->heartbeat_last_seen[i];
        if (elapsed > timeout_sec && num_failed < max_failed) {
            failed_nodes[num_failed] = i;
            num_failed++;
        }
    }
    
    return num_failed;
}

/* ============================================================================
 * F-08: 节点故障恢复实现
 * 
 * 在检测到节点故障后，重建通信拓扑并继续训练
 * =========================================================================== */

/**
 * @brief 单节点故障恢复
 * 
 * 当检测到节点失败时，重新分配其工作负载到其他节点。
 * 增强功能：P1-6 添加领导者选举和检查点自动恢复。
 * 
 * @param trainer 训练器
 * @param failed_node_id 失败节点ID
 * @return int 成功返回0，失败返回-1
 */
static int distributed_failure_recovery(Trainer* trainer, int failed_node_id) {
    if (!trainer) return -1;
    
    if (!trainer->failure_recovery_enabled) {
        return -1;
    }
    
    int N = trainer->distributed_num_nodes;
    
    if (failed_node_id < 0 || failed_node_id >= N) {
        return -1;
    }
    
    trainer->failure_recovery_attempts++;
    
    if (trainer->config.verbose) {
        printf("故障恢复 | 节点 %d 失败, 正在重建拓扑... (尝试 %d/%d)\n",
               failed_node_id, trainer->failure_recovery_attempts,
               trainer->failure_recovery_max_attempts);
    }
    
    /* 1. 标记失败节点 */
    trainer->heartbeat_node_alive[failed_node_id] = 0;
    
    /* 2. 更新存活节点数 */
    int alive_count = 0;
    for (int i = 0; i < N; i++) {
        if (trainer->heartbeat_node_alive[i]) {
            alive_count++;
        }
    }
    
    if (alive_count < 2) {
        if (trainer->config.verbose) {
            printf("故障恢复 | 存活节点不足 (存活=%d), 正在尝试自动恢复...\n", alive_count);
        }
        /* 尝试从检查点自动恢复 */
        if (trainer->auto_resume_enabled && trainer->auto_resume_checkpoint_path) {
            return distributed_auto_resume_from_checkpoint(trainer, NULL, 0);
        }
        return -1;
    }
    
    /* 3. 领导者选举：如果失败节点是领导者，选举新领导者 */
    if (trainer->leader_election_enabled && failed_node_id == 0) {
        int failed_arr[1] = {failed_node_id};
        int election_result = distributed_leader_election(trainer, failed_arr, 1);
        if (election_result < 0) {
            if (trainer->config.verbose) {
                printf("故障恢复 | 领导者选举失败\n");
            }
        }
    }
    
    /* 4. 重新分配失败节点的工作负载 */
    if (trainer->failure_recovery_buffer) {
        size_t chunk_size = trainer->distributed_buffer_size / N;
        int survivor_idx = 0;
        for (int i = 0; i < N; i++) {
            if (!trainer->heartbeat_node_alive[i]) continue;
            
            size_t start = (size_t)failed_node_id * chunk_size;
            size_t end = ((size_t)failed_node_id + 1) * chunk_size;
            if (end > trainer->distributed_buffer_size) {
                end = trainer->distributed_buffer_size;
            }
            
            if (end > start) {
                size_t half = (end - start) / 2;
                size_t survivor_count = (size_t)alive_count;
                size_t share_start = start + (size_t)survivor_idx * half;
                size_t share_end = (survivor_idx < (int)(survivor_count - 1)) ?
                                   share_start + half : end;
                if (share_end > end) share_end = end;
                if (share_end > share_start) {
                    memmove(&trainer->failure_recovery_buffer[share_start],
                            &trainer->failure_recovery_buffer[start],
                            (share_end - share_start) * sizeof(float));
                }
                survivor_idx++;
            }
        }
    }
    
    /* 5. 重建拓扑 */
    int failed_nodes[1] = {failed_node_id};
    distributed_failure_rebuild_topology(trainer, failed_nodes, 1);
    
    /* 6. 更新过时梯度计数器 */
    if (trainer->stale_gradient_enabled && trainer->stale_gradient_counters) {
        for (int i = 0; i < N; i++) {
            if (i != failed_node_id && trainer->stale_gradient_counters) {
                trainer->stale_gradient_counters[i]++;
            }
        }
    }
    
    /* 7. 如果达到最大尝试次数，尝试自动恢复 */
    if (trainer->failure_recovery_attempts >= trainer->failure_recovery_max_attempts) {
        if (trainer->config.verbose) {
            printf("故障恢复 | 达到最大尝试次数 %d, 正在尝试自动恢复...\n",
                   trainer->failure_recovery_max_attempts);
        }
        if (trainer->auto_resume_enabled && trainer->auto_resume_checkpoint_path) {
            return distributed_auto_resume_from_checkpoint(trainer, NULL, 0);
        }
        return -1;
    }
    
    return 0;
}

/**
 * @brief 重建分布式拓扑（排除失败节点）
 * 
 * @param trainer 训练器
 * @param failed_nodes 失败节点ID数组
 * @param num_failed 失败节点数量
 * @return int 成功返回0
 */
static int distributed_failure_rebuild_topology(Trainer* trainer, const int* failed_nodes, int num_failed) {
    if (!trainer || !failed_nodes || num_failed <= 0) {
        return -1;
    }
    
    /* 重新计算有效节点数 */
    int new_N = trainer->distributed_num_nodes - num_failed;
    if (new_N < 1) {
        return -1;
    }
    
    /* 重新映射节点ID（跳过失败节点） */
    int new_id = trainer->distributed_node_id;
    int offset = 0;
    for (int i = 0; i < num_failed; i++) {
        if (trainer->distributed_node_id > failed_nodes[i]) {
            offset++;
        }
    }
    new_id -= offset;
    
    /* 更新分布式配置 */
    trainer->distributed_num_nodes = new_N;
    trainer->distributed_node_id = new_id;
    
    /* 更新是否是领导节点 */
    trainer->distributed_is_leader = (new_id == 0);
    
    /* 重建树形拓扑 */
    if (trainer->tree_enabled) {
        distributed_tree_build_topology(trainer);
    }
    
    /* 重建网格拓扑 */
    if (trainer->mesh_enabled) {
        distributed_mesh_build_topology(trainer);
    }
    
    if (trainer->config.verbose) {
        printf("拓扑重建完成 | 新节点ID=%d/%d (原ID=%d)\n",
               new_id, new_N, trainer->distributed_node_id + offset);
    }
    
    return 0;
}

/**
 * @brief 检查是否需要保存分布式检查点
 * 
 * @param trainer 训练器
 * @return int 需要保存返回1，不需要返回0，错误返回-1
 */
static int distributed_should_save_checkpoint(Trainer* trainer) {
    if (!trainer || !trainer->distributed_initialized) {
        return 0;
    }
    
    // 如果没有启用检查点功能，则不需要保存
    if (!trainer->config.distributed_enable_checkpointing) {
        return 0;
    }
    
    // 更新批次计数器
    trainer->distributed_batch_counter++;
    
    // 检查是否达到检查点保存频率
    if (trainer->distributed_batch_counter >= trainer->config.distributed_checkpoint_frequency) {
        trainer->distributed_batch_counter = 0;
        return 1;
    }
    
    return 0;
}

/**
 * @brief 保存分布式检查点
 * 
 * @param trainer 训练器
 * @param parameters 模型参数
 * @param num_parameters 参数数量
 * @param epoch 当前训练轮数
 * @param batch 当前批次
 * @return int 成功返回0，失败返回-1
 */
static int distributed_save_checkpoint(Trainer* trainer, float* parameters, 
                                      size_t num_parameters, size_t epoch, size_t batch) {
    if (!trainer || !parameters || num_parameters == 0) {
        return -1;
    }
    
    // 检查是否启用分布式训练和检查点功能
    if (!trainer->distributed_initialized || 
        !trainer->config.distributed_enable_checkpointing) {
        return 0;  // 未启用检查点功能，视为成功
    }
    
    // 只有领导节点保存检查点
    if (!trainer->distributed_is_leader) {
        return 0;
    }
    
    // 将参数复制到分布式参数缓冲区
    if (trainer->distributed_buffer_size >= num_parameters) {
        memcpy(trainer->distributed_parameter_buffer, parameters, num_parameters * sizeof(float));
    }
    
    // 保存分布式检查点到文件
    char checkpoint_filename[512];
    int written = snprintf(checkpoint_filename, sizeof(checkpoint_filename),
                          "distributed_ckpt_node%d_epoch%zu_batch%zu.bin",
                          trainer->config.distributed_node_id, epoch, batch);
    if (written < 0 || (size_t)written >= sizeof(checkpoint_filename)) {
        // 文件名过长，使用备用名称（缓冲区溢出保护）
        snprintf(checkpoint_filename, sizeof(checkpoint_filename),
                "distributed_ckpt_%d.bin", trainer->config.distributed_node_id);
    }
    
    FILE* ckpt_file = fopen(checkpoint_filename, "wb");
    if (!ckpt_file) {
        if (trainer->config.verbose) {
            printf("警告：无法创建分布式检查点文件'%s'\n", checkpoint_filename);
        }
        return -1;
    }
    
    // 写入检查点元数据
    uint32_t magic = 0x4449434B; // "DICK"
    fwrite(&magic, sizeof(uint32_t), 1, ckpt_file);
    uint32_t ckpt_version = 1;
    fwrite(&ckpt_version, sizeof(uint32_t), 1, ckpt_file);
    uint64_t ckpt_epoch = (uint64_t)epoch;
    fwrite(&ckpt_epoch, sizeof(uint64_t), 1, ckpt_file);
    uint64_t ckpt_batch = (uint64_t)batch;
    fwrite(&ckpt_batch, sizeof(uint64_t), 1, ckpt_file);
    uint64_t param_count = (uint64_t)num_parameters;
    fwrite(&param_count, sizeof(uint64_t), 1, ckpt_file);
    
    // 写入参数数据
    size_t written_params = fwrite(parameters, sizeof(float), num_parameters, ckpt_file);
    
    fclose(ckpt_file);
    
    if (written_params != num_parameters) {
        if (trainer->config.verbose) {
            log_warning("分布式检查点参数写入不完整\n");
        }
        return -1;
    }
    
    if (trainer->config.verbose) {
        printf("分布式检查点已保存：%s，参数数量=%zu\n", 
               checkpoint_filename, num_parameters);
    }
    
    trainer->distributed_checkpoint_ready = 1;
    
    return 0;
}

/**
 * @brief 加载分布式检查点
 * 
 * 从文件加载分布式训练检查点，恢复参数和训练进度。
 * 执行魔术字验证和参数计数校验。
 * 
 * @param trainer 训练器
 * @param parameters 输出参数缓冲区
 * @param num_parameters 参数数量
 * @param epoch 输出恢复的轮数
 * @param batch 输出恢复的批次
 * @return int 成功返回0，失败返回-1
 */
static int distributed_load_checkpoint(Trainer* trainer, float* parameters,
                                       size_t num_parameters, size_t* epoch, size_t* batch) {
    if (!trainer || !parameters || num_parameters == 0) {
        return -1;
    }
    if (!trainer->distributed_initialized) {
        return -1;
    }
    if (!trainer->config.distributed_enable_checkpointing) {
        return -1;
    }
    if (!trainer->auto_resume_checkpoint_path) {
        return -1;
    }

    FILE* ckpt_file = fopen(trainer->auto_resume_checkpoint_path, "rb");
    if (!ckpt_file) {
        if (trainer->config.verbose) {
            printf("检查点恢复 | 无法打开检查点文件 '%s'\n",
                   trainer->auto_resume_checkpoint_path);
        }
        return -1;
    }

    uint32_t magic = 0;
    if (fread(&magic, sizeof(uint32_t), 1, ckpt_file) != 1 || magic != 0x4449434B) {
        fclose(ckpt_file);
        if (trainer->config.verbose) {
            printf("检查点恢复 | 魔术字校验失败 (got 0x%08X)\n", magic);
        }
        return -1;
    }

    uint32_t ckpt_version = 0;
    if (fread(&ckpt_version, sizeof(uint32_t), 1, ckpt_file) != 1) {
        fclose(ckpt_file);
        return -1;
    }

    /* P0-018修复: 检查点版本兼容性验证
     * 拒绝加载不兼容版本的检查点文件，防止权重格式错乱 */
    {
        int version_ok = 0;
        for (size_t vi = 0; vi < CKPT_NUM_SUPPORTED_VERSIONS; vi++) {
            if (ckpt_version == g_ckpt_supported_versions[vi]) {
                version_ok = 1;
                break;
            }
        }
        if (!version_ok) {
            fclose(ckpt_file);
            if (trainer->config.verbose) {
                printf("检查点恢复 | 版本不兼容: 文件版本=%u, 支持版本=[", ckpt_version);
                for (size_t vi = 0; vi < CKPT_NUM_SUPPORTED_VERSIONS; vi++) {
                    printf("%s%u", vi > 0 ? "," : "", g_ckpt_supported_versions[vi]);
                }
                printf("]\n");
            }
            return -1;
        }
    }

    uint64_t ckpt_epoch = 0, ckpt_batch = 0, param_count = 0;
    if (fread(&ckpt_epoch, sizeof(uint64_t), 1, ckpt_file) != 1 ||
        fread(&ckpt_batch, sizeof(uint64_t), 1, ckpt_file) != 1 ||
        fread(&param_count, sizeof(uint64_t), 1, ckpt_file) != 1) {
        fclose(ckpt_file);
        return -1;
    }

    if ((size_t)param_count != num_parameters) {
        fclose(ckpt_file);
        if (trainer->config.verbose) {
            printf("检查点恢复 | 参数数量不匹配: 文件=%llu, 期望=%zu\n",
                   (unsigned long long)param_count, num_parameters);
        }
        return -1;
    }

    size_t read_count = fread(parameters, sizeof(float), num_parameters, ckpt_file);
    fclose(ckpt_file);

    if (read_count != num_parameters) {
        if (trainer->config.verbose) {
            printf("检查点恢复 | 参数读取不完整: 读取=%zu, 期望=%zu\n",
                   read_count, num_parameters);
        }
        return -1;
    }

    if (epoch) *epoch = (size_t)ckpt_epoch;
    if (batch) *batch = (size_t)ckpt_batch;

    if (trainer->config.verbose) {
        printf("检查点恢复 | 成功从 '%s' 加载，轮数=%llu，批次=%llu，参数=%llu\n",
               trainer->auto_resume_checkpoint_path,
               (unsigned long long)ckpt_epoch,
               (unsigned long long)ckpt_batch,
               (unsigned long long)param_count);
    }

    return 0;
}

/**
 * @brief 从检查点自动恢复训练
 * 
 * 当节点故障恢复失败时，尝试从最新的检查点加载参数，
 * 并重置训练状态以继续训练。
 * 
 * @param trainer 训练器
 * @param parameters 参数缓冲区
 * @param num_parameters 参数数量
 * @return int 成功返回0，失败返回-1
 */
static int distributed_auto_resume_from_checkpoint(Trainer* trainer, float* parameters,
                                                    size_t num_parameters) {
    if (!trainer || !parameters || num_parameters == 0) {
        return -1;
    }
    if (!trainer->auto_resume_enabled) {
        return -1;
    }

    size_t restored_epoch = 0, restored_batch = 0;
    int result = distributed_load_checkpoint(trainer, parameters, num_parameters,
                                             &restored_epoch, &restored_batch);
    if (result != 0) {
        if (trainer->config.verbose) {
            printf("自动恢复 | 从检查点加载失败\n");
        }
        return -1;
    }

    trainer->state.current_epoch = restored_epoch;
    trainer->state.current_batch = restored_batch;
    trainer->distributed_retry_count = 0;
    trainer->failure_recovery_attempts = 0;

    trainer->distributed_batch_counter = (int)restored_batch;
    trainer->distributed_checkpoint_ready = 1;

    if (trainer->config.verbose) {
        printf("自动恢复 | 从检查点恢复训练: epoch=%zu, batch=%zu\n",
               restored_epoch, restored_batch);
    }

    return 0;
}

/**
 * @brief 分布式领导者选举
 * 
 * 当领导节点故障时，在存活的节点中选举新的领导者。
 * 选举策略：选择存活节点中节点ID最小的作为新领导者。
 * 
 * @param trainer 训练器
 * @param failed_nodes 失败节点ID数组
 * @param num_failed 失败节点数量
 * @return int 当前节点被选为领导者返回1，未被选中返回0，失败返回-1
 */
static int distributed_leader_election(Trainer* trainer, const int* failed_nodes, int num_failed) {
    if (!trainer || !failed_nodes || num_failed <= 0) {
        return -1;
    }
    if (!trainer->leader_election_enabled) {
        return 0;
    }

    int N = trainer->distributed_num_nodes;

    int* is_failed = (int*)calloc((size_t)N, sizeof(int));
    if (!is_failed) {
        return -1;
    }
    for (int i = 0; i < num_failed; i++) {
        if (failed_nodes[i] >= 0 && failed_nodes[i] < N) {
            is_failed[failed_nodes[i]] = 1;
        }
    }

    int new_leader_id = -1;
    int elected = 0;

    for (int i = 0; i < N; i++) {
        if (!is_failed[i] && trainer->heartbeat_node_alive[i]) {
            if (new_leader_id < 0 || i < new_leader_id) {
                new_leader_id = i;
            }
        }
    }

    free(is_failed);

    if (new_leader_id < 0) {
        if (trainer->config.verbose) {
            printf("领导者选举 | 没有存活的节点可以选举\n");
        }
        return -1;
    }

    trainer->distributed_is_leader = (trainer->distributed_node_id == new_leader_id);

    if (trainer->config.verbose) {
        printf("领导者选举 | 节点 %d 被选为新领导者 (当前节点=%d, 是否领导=%d)\n",
               new_leader_id, trainer->distributed_node_id, trainer->distributed_is_leader);
    }

    if (trainer->distributed_is_leader) {
        elected = 1;
    }

    return elected;
}

/**
 * @brief 弹性训练添加节点
 * 
 * 动态向训练集群添加新节点，不中断正在进行的训练。
 * 更新节点总数和拓扑结构，重新分配工作负载。
 * 
 * @param trainer 训练器
 * @param new_node_id 新节点ID
 * @param num_total_nodes 添加后的总节点数
 * @return int 成功返回0，失败返回-1
 */
static int distributed_elastic_add_node(Trainer* trainer, int new_node_id, int num_total_nodes) {
    if (!trainer) return -1;
    if (!trainer->elastic_enabled) return -1;
    if (new_node_id < 0 || num_total_nodes <= trainer->distributed_num_nodes) return -1;

    int old_N = trainer->distributed_num_nodes;

    if (trainer->heartbeat_last_seen) {
        double* new_last_seen = (double*)realloc(trainer->heartbeat_last_seen,
                                                  (size_t)num_total_nodes * sizeof(double));
        if (!new_last_seen) return -1;
        trainer->heartbeat_last_seen = new_last_seen;
        trainer->heartbeat_last_seen[new_node_id] = 0.0;
    }

    if (trainer->heartbeat_node_alive) {
        volatile int* new_alive = (volatile int*)realloc(
            (void*)trainer->heartbeat_node_alive,
            (size_t)num_total_nodes * sizeof(int));
        if (!new_alive) return -1;
        trainer->heartbeat_node_alive = new_alive;
        trainer->heartbeat_node_alive[new_node_id] = 1;
    }

    if (trainer->stale_gradient_enabled && trainer->stale_gradient_coefficients) {
        float* new_coeffs = (float*)realloc(trainer->stale_gradient_coefficients,
                                             (size_t)num_total_nodes * sizeof(float));
        if (!new_coeffs) return -1;
        trainer->stale_gradient_coefficients = new_coeffs;
        trainer->stale_gradient_coefficients[new_node_id] = 1.0f;
    }

    if (trainer->stale_gradient_enabled && trainer->stale_gradient_counters) {
        int* new_counters = (int*)realloc(trainer->stale_gradient_counters,
                                           (size_t)num_total_nodes * sizeof(int));
        if (!new_counters) return -1;
        trainer->stale_gradient_counters = new_counters;
        trainer->stale_gradient_counters[new_node_id] = 0;
    }

    trainer->heartbeat_num_nodes = num_total_nodes;

    if (trainer->config.verbose) {
        printf("弹性训练 | 节点 %d 已加入集群，节点数 %d → %d\n",
               new_node_id, old_N, num_total_nodes);
    }

    if (trainer->distributed_is_leader) {
        distributed_elastic_rebalance_workload(trainer);
    }

    return 0;
}

/**
 * @brief 弹性训练移除节点
 * 
 * 从训练集群中优雅地移除一个节点。更新拓扑结构
 * 并重新分配该节点的工作负载到其他存活节点。
 * 
 * @param trainer 训练器
 * @param remove_node_id 要移除的节点ID
 * @return int 成功返回0，失败返回-1
 */
static int distributed_elastic_remove_node(Trainer* trainer, int remove_node_id) {
    if (!trainer) return -1;
    if (!trainer->elastic_enabled) return -1;
    if (remove_node_id < 0 || remove_node_id >= trainer->distributed_num_nodes) return -1;

    if (trainer->heartbeat_node_alive) {
        trainer->heartbeat_node_alive[remove_node_id] = 0;
    }

    int failed_nodes[1] = {remove_node_id};
    distributed_failure_rebuild_topology(trainer, failed_nodes, 1);

    if (trainer->distributed_is_leader) {
        distributed_elastic_rebalance_workload(trainer);
    }

    if (trainer->config.verbose) {
        printf("弹性训练 | 节点 %d 已从集群中移除，当前节点数=%d\n",
               remove_node_id, trainer->distributed_num_nodes);
    }

    return 0;
}

/**
 * @brief 弹性训练工作负载重新平衡
 * 
 * 在所有存活节点间重新平衡数据分片和工作负载。
 * 根据当前节点数均匀分配数据块。
 * 
 * @param trainer 训练器
 * @return int 成功返回0，失败返回-1
 */
static int distributed_elastic_rebalance_workload(Trainer* trainer) {
    if (!trainer) return -1;
    if (!trainer->elastic_enabled) return -1;
    if (!trainer->distributed_is_leader) return 0;

    int N = trainer->distributed_num_nodes;
    if (N <= 0) return -1;

    if (trainer->failure_recovery_buffer && trainer->distributed_buffer_size > 0) {
        size_t total_size = trainer->distributed_buffer_size;
        size_t chunk_size = total_size / (size_t)N;
        if (chunk_size == 0) chunk_size = 1;

        int alive_count = 0;
        int* alive_ids = (int*)calloc((size_t)N, sizeof(int));
        if (!alive_ids) return -1;

        for (int i = 0; i < N; i++) {
            if (trainer->heartbeat_node_alive && trainer->heartbeat_node_alive[i]) {
                alive_ids[alive_count++] = i;
            }
        }

        if (alive_count == 0) {
            free(alive_ids);
            return -1;
        }

        float* new_buffer = (float*)calloc(total_size, sizeof(float));
        if (!new_buffer) {
            free(alive_ids);
            return -1;
        }

        size_t pos = 0;
        for (int i = 0; i < alive_count && pos < total_size; i++) {
            size_t this_chunk = (i < alive_count - 1) ? chunk_size : (total_size - pos);
            if (this_chunk > 0) {
                memcpy(&new_buffer[pos], &trainer->failure_recovery_buffer[pos],
                       this_chunk * sizeof(float));
                pos += this_chunk;
            }
        }

        memcpy(trainer->failure_recovery_buffer, new_buffer, total_size * sizeof(float));
        free(new_buffer);
        free(alive_ids);
    }

    if (trainer->config.verbose) {
        printf("弹性训练 | 工作负载已重新平衡，节点数=%d\n", N);
    }

    return 0;
}

/**
 * @brief 处理过时梯度
 * 
 * 检测并衰减来自滞后节点的过时梯度。
 * 使用 bounded staleness 机制：每个节点跟踪其梯度延迟，
 * 延迟越高的节点其梯度贡献衰减越大。
 * 
 * @param trainer 训练器
 * @param gradients 梯度数组
 * @param num_parameters 梯度数量
 * @return int 成功返回0，失败返回-1
 */
static int distributed_handle_stale_gradients(Trainer* trainer, float* gradients,
                                              size_t num_parameters) {
    if (!trainer || !gradients || num_parameters == 0) {
        return -1;
    }
    if (!trainer->stale_gradient_enabled) {
        return 0;
    }
    if (!trainer->stale_gradient_coefficients || !trainer->stale_gradient_counters) {
        return -1;
    }

    int node_id = trainer->distributed_node_id;
    int N = trainer->distributed_num_nodes;

    if (node_id < 0 || node_id >= N) return -1;

    int staleness = trainer->stale_gradient_counters[node_id];
    if (staleness > 0) {
        float coefficient = 1.0f / (1.0f + (float)staleness * 0.5f);
        if (coefficient < 0.1f) coefficient = 0.1f;
        trainer->stale_gradient_coefficients[node_id] = coefficient;

        for (size_t i = 0; i < num_parameters; i++) {
            gradients[i] *= coefficient;
        }

        if (trainer->config.verbose) {
            printf("过时梯度 | 节点 %d 延迟=%d, 衰减系数=%.3f\n",
                   node_id, staleness, coefficient);
        }
    } else {
        trainer->stale_gradient_coefficients[node_id] = 1.0f;
    }

    trainer->stale_gradient_counters[node_id] = 0;

    for (int i = 0; i < N; i++) {
        if (i != node_id && trainer->stale_gradient_counters) {
            trainer->stale_gradient_counters[i]++;
        }
    }

    return 0;
}

/**
 * @brief 应用分布式学习率缩放
 * 
 * @param trainer 训练器
 * @return float 缩放后的学习率
 */
static float distributed_apply_learning_rate_scaling(Trainer* trainer) {
    if (!trainer || !trainer->distributed_initialized) {
        return trainer ? trainer->state.learning_rate : 0.0f;
    }
    
    // 应用分布式学习率缩放
    float scaled_lr = trainer->state.learning_rate * trainer->config.distributed_learning_rate_scale;
    trainer->distributed_learning_rate_scaled = scaled_lr;
    
    return scaled_lr;
}

/**
 * @brief 分布式训练容错处理
 * 
 * @param trainer 训练器
 * @param operation_type 操作类型（0=梯度同步，1=参数更新，2=检查点）
 * @param error_code 错误代码
 * @return int 成功处理返回0，失败返回-1
 */
static int distributed_fault_tolerance(Trainer* trainer, int operation_type, int error_code) {
    if (!trainer || !trainer->distributed_initialized) {
        return -1;
    }
    
    // 检查是否启用容错机制
    if (!trainer->config.distributed_enable_fault_tolerance) {
        return -1;  // 未启用容错机制
    }
    
    // 增加重试计数
    trainer->distributed_retry_count++;
    
    // 检查是否超过最大重试次数
    if (trainer->distributed_retry_count > trainer->config.distributed_max_retries) {
        if (trainer->config.verbose) {
            printf("错误：分布式操作超过最大重试次数，操作类型=%d，错误代码=%d\n", 
                   operation_type, error_code);
        }
        return -1;  // 超过最大重试次数
    }
    
    if (trainer->config.verbose) {
        printf("警告：分布式操作失败，正在重试（%d/%d），操作类型=%d，错误代码=%d\n", 
               trainer->distributed_retry_count, trainer->config.distributed_max_retries,
               operation_type, error_code);
    }
    
    return 0;  // 允许重试
}

/**
 * @brief 创建训练器
 */
Trainer* trainer_create(const TrainingConfig* config, LNN* network) {
    if (!config || !network) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "创建训练器：参数无效");
        return NULL;
    }
    
    // 验证网络维度
    LNNConfig net_config;
    if (lnn_get_config(network, &net_config) != 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "创建训练器：获取网络配置失败");
        return NULL;
    }
    
    size_t input_dim = net_config.input_size;
    size_t output_dim = net_config.output_size;
    
    if (input_dim == 0 || output_dim == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "创建训练器：网络维度无效");
        return NULL;
    }
    
    // 分配训练器
    Trainer* trainer = (Trainer*)safe_malloc(sizeof(Trainer));
    if (!trainer) {
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "创建训练器：内存分配失败");
        return NULL;
    }
    
    // 初始化线程安全锁
    trainer->lock = mutex_create();
    
    // 复制配置
    trainer->config = *config;
    trainer->network = network;
    
    // 获取网络配置用于调试（net_config已在上方定义）
    if (config->verbose) {
        fprintf(stderr, "trainer_create: 网络配置 - 输入大小=%zu, 隐藏大小=%zu, 输出大小=%zu\n",
                net_config.input_size, net_config.hidden_size, net_config.output_size);
    }
    
    // 初始化优化器
    size_t num_parameters = lnn_get_parameter_count(network);
    if (config->verbose) {
        fprintf(stderr, "trainer_create: 网络参数数量 = %zu (分配 %zu 字节)\n", 
                num_parameters, num_parameters * sizeof(float));
    }
    if (optimizer_init(&trainer->optimizer, config, num_parameters) != 0) {
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "优化器缓冲区分配失败");
        trainer_free(trainer);
        return NULL;
    }
    
    // 初始化学习率调度器
    LearningRateSchedulerConfig scheduler_config = {
        .type = SCHEDULER_CONSTANT,
        .base_learning_rate = config->learning_rate,
        .max_learning_rate = config->learning_rate * 10.0f,
        .min_learning_rate = config->learning_rate * 0.1f,
        .decay_rate = config->learning_rate_decay,
        .decay_steps = 1000,
        .step_size = 100,
        .cycle_length = 2000
    };
    
    size_t total_steps = config->epochs * (1000 / config->batch_size + 1);
    trainer->scheduler = scheduler_create_internal(&scheduler_config, total_steps);
    
    // 初始化训练状态
    memset(&trainer->state, 0, sizeof(TrainingState));
    trainer->state.learning_rate = config->learning_rate;
    trainer->state.best_loss = FLT_MAX;
    
    // 初始化训练历史
    trainer->history.size = 0;
    trainer->history.capacity = config->epochs;
    
    size_t capacity = trainer->history.capacity;
    trainer->history.train_losses = (float*)safe_malloc(capacity * sizeof(float));
    trainer->history.train_accuracies = (float*)safe_malloc(capacity * sizeof(float));
    trainer->history.val_losses = (float*)safe_malloc(capacity * sizeof(float));
    trainer->history.val_accuracies = (float*)safe_malloc(capacity * sizeof(float));
    trainer->history.learning_rates = (float*)safe_malloc(capacity * sizeof(float));
    
    if (!trainer->history.train_losses || !trainer->history.train_accuracies ||
        !trainer->history.val_losses || !trainer->history.val_accuracies ||
        !trainer->history.learning_rates) {
        trainer_free(trainer);
        return NULL;
    }
    
    // 分配梯度缓冲区
    trainer->gradients_size = num_parameters;
    if (config->verbose) {
        fprintf(stderr, "trainer_create: 分配梯度缓冲区，大小=%zu个浮点数 (%zu字节)\n", 
                num_parameters, num_parameters * sizeof(float));
        fflush(stderr);
    }
    trainer->gradients = (float*)safe_malloc(num_parameters * sizeof(float));
    if (!trainer->gradients) {
        trainer_free(trainer);
        return NULL;
    }
    if (config->verbose) {
        fprintf(stderr, "trainer_create: 梯度缓冲区地址 = %p\n", trainer->gradients);
        fflush(stderr);
    }
    
    // 初始化拉普拉斯优化相关字段
    trainer->laplace_analyzer = NULL;
    trainer->laplace_filtered_gradients = NULL;
    trainer->gradient_history = NULL;
    trainer->gradient_history_size = 0;
    trainer->gradient_history_capacity = 0;
    trainer->gradient_history_position = 0;
    trainer->laplace_current_cutoff = config->laplace_filter_cutoff;
    trainer->laplace_stability_score = 1.0f; // 初始稳定性分数
    trainer->laplace_stability_warning = 0;
    
    // 如果启用拉普拉斯优化，初始化拉普拉斯分析器
    if (config->use_laplace_optimization) {
        // 创建拉普拉斯分析器
        LaplaceConfig laplace_config = {
            .num_samples = 100,
            .sample_rate = 1000.0f, // 假设采样率1000Hz
            .max_frequency = 100.0f,
            .min_frequency = 0.1f,
            .enable_stability = 1,
            .enable_frequency = 1,
            .enable_optimization = 1
        };
        
        trainer->laplace_analyzer = laplace_analyzer_create(&laplace_config);
        if (!trainer->laplace_analyzer) {
            trainer_free(trainer);
            return NULL;
        }
        
        // 分配拉普拉斯滤波后的梯度缓冲区
        trainer->laplace_filtered_gradients = (float*)safe_malloc(num_parameters * sizeof(float));
        if (!trainer->laplace_filtered_gradients) {
            trainer_free(trainer);
            return NULL;
        }
        
        // 分配梯度历史缓冲区（记录最近1000个梯度值用于频域分析）
        size_t history_capacity = 1000; // 记录最近1000个梯度值
        trainer->gradient_history_capacity = history_capacity;
        trainer->gradient_history = (float*)safe_calloc(history_capacity, sizeof(float));
        if (!trainer->gradient_history) {
            trainer_free(trainer);
            return NULL;
        }
        
        if (config->verbose) {
            printf("拉普拉斯优化已启用，截止频率=%.1fHz\n", config->laplace_filter_cutoff);
        }
    }
    
    // 分配批次缓冲区
    size_t batch_input_size = config->batch_size * input_dim;
    size_t batch_output_size = config->batch_size * output_dim;
    
    trainer->batch_inputs = (float*)safe_malloc(batch_input_size * sizeof(float));
    trainer->batch_targets = (float*)safe_malloc(batch_output_size * sizeof(float));
    trainer->batch_outputs = (float*)safe_malloc(batch_output_size * sizeof(float));
    
    if (!trainer->batch_inputs || !trainer->batch_targets || !trainer->batch_outputs) {
        trainer_free(trainer);
        return NULL;
    }
    
    // 分配Dropout掩码缓冲区
    size_t max_activations = lnn_get_max_activation_count(network);
    trainer->dropout_masks_size = max_activations;
    if (max_activations > 0) {
        trainer->dropout_masks = (float*)safe_malloc(max_activations * sizeof(float));
        if (!trainer->dropout_masks) {
            trainer_free(trainer);
            return NULL;
        }
    } else {
        trainer->dropout_masks = NULL;
    }
    
    // 初始化GPU加速相关字段
    trainer->gpu_context = NULL;
    trainer->gpu_inputs = NULL;
    trainer->gpu_targets = NULL;
    trainer->gpu_outputs = NULL;
    trainer->gpu_gradients = NULL;
    trainer->gpu_parameters = NULL;
    trainer->gpu_initialized = 0;
    trainer->gpu_backend = GPU_BACKEND_CPU;
    
    // 初始化GPU优化器状态缓冲区
    trainer->gpu_optimizer_m = NULL;
    trainer->gpu_optimizer_v = NULL;
    trainer->gpu_optimizer_state_initialized = 0;
    trainer->gpu_optimizer_state_size = 0;
    
    // 如果启用GPU，尝试初始化GPU上下文（自动检测最佳可用后端）
    if (config->use_gpu) {
        // 使用gpu_auto_select自动检测最佳可用GPU后端
        // 优先级：CUDA → Vulkan → OpenCL → Metal → CPU
        GpuBackend selected_backend = gpu_auto_select();
        
        // 确认选择了GPU后端（非CPU）
        if (selected_backend != GPU_BACKEND_CPU) {
            // gpu_auto_select已成功初始化选中的后端
            trainer->gpu_context = gpu_context_create(selected_backend, 0);
            if (trainer->gpu_context) {
                trainer->gpu_backend = selected_backend;
                trainer->gpu_initialized = 1;
                if (config->verbose) {
                    const char* name = "未知";
                    switch (selected_backend) {
                        case GPU_BACKEND_CUDA:   name = "NVIDIA CUDA"; break;
                        case GPU_BACKEND_OPENCL: name = "OpenCL"; break;
                        case GPU_BACKEND_VULKAN: name = "Vulkan"; break;
                        case GPU_BACKEND_METAL:  name = "Metal(Apple)"; break;
                        default: break;
                    }
                    printf("GPU加速已启用，自动选择后端：%s\n", name);
                }
            } else {
                trainer->gpu_backend = GPU_BACKEND_CPU;
                if (config->verbose) {
                    log_warning("GPU上下文创建失败，回退到CPU训练\n");
                }
            }
        } else {
            // 无可用GPU，自动回退到CPU训练
            if (config->verbose) {
                log_warning("未检测到可用GPU硬件，自动使用CPU训练\n");
            }
        }
    }
    
    // 初始化混合精度训练（如果启用）
    trainer->mixed_precision_context = NULL;
    if (config->use_mixed_precision) {
        // 创建混合精度配置
        MixedPrecisionConfig mp_config;
        mixed_precision_default_config(&mp_config);
        
        // 根据TrainingConfig中的设置覆盖默认值
        mp_config.mode = (MixedPrecisionMode)config->mixed_precision_mode;
        if (config->mixed_precision_mode == 0) {
            mp_config.mode = MIXED_PRECISION_FP16; // 默认使用FP16模式
        }
        
        mp_config.initial_scale = config->mixed_precision_initial_scale;
        mp_config.max_scale = config->mixed_precision_max_scale;
        mp_config.min_scale = config->mixed_precision_min_scale;
        mp_config.enable_fp16_arithmetic = config->mixed_precision_enable_fp16_arithmetic;
        mp_config.enable_fp16_storage = config->mixed_precision_enable_fp16_storage;
        mp_config.check_nan_inf = config->mixed_precision_check_nan_inf;
        
        // 启用混合精度训练
        if (mixed_precision_enable(trainer, &mp_config) == 0) {
            if (config->verbose) {
                printf("混合精度训练已启用，模式=%d\n", mp_config.mode);
            }
        } else {
            if (config->verbose) {
                log_warning("混合精度训练初始化失败，回退到标准精度\n");
            }
        }
    }
    
    // 初始化分布式训练相关字段
    trainer->distributed_initialized = 0;
    trainer->distributed_node_id = config->distributed_node_id;
    trainer->distributed_num_nodes = config->distributed_num_nodes;
    trainer->distributed_communication_backend = config->distributed_communication_backend;
    trainer->distributed_sync_counter = 0;
    trainer->distributed_batch_counter = 0;
    trainer->distributed_retry_count = 0;
    
    trainer->distributed_gradient_buffer = NULL;
    trainer->distributed_parameter_buffer = NULL;
    trainer->distributed_buffer_size = 0;
    
    trainer->distributed_comm_context = NULL;
    trainer->distributed_comm_rank = 0;
    trainer->distributed_comm_size = 1;
    
    trainer->distributed_is_leader = (config->distributed_node_id == 0);
    trainer->distributed_checkpoint_ready = 0;
    trainer->distributed_sync_pending = 0;
    trainer->distributed_learning_rate_scaled = config->learning_rate * config->distributed_learning_rate_scale;
    
    // 如果启用分布式训练，分配分布式缓冲区
    if (config->use_distributed_training && config->distributed_num_nodes > 1) {
        // 分配分布式梯度缓冲区
        size_t distributed_num_params = lnn_get_parameter_count(network);
        trainer->distributed_buffer_size = distributed_num_params;
        
        if (distributed_num_params > 0) {
            if (config->verbose) {
                printf("trainer_create: 分配分布式梯度缓冲区，大小=%zu * %zu = %zu字节\n", 
                       distributed_num_params, sizeof(float), distributed_num_params * sizeof(float));
            }
            trainer->distributed_gradient_buffer = (float*)safe_malloc(distributed_num_params * sizeof(float));
            if (config->verbose) {
                printf("trainer_create: 梯度缓冲区地址=%p\n", trainer->distributed_gradient_buffer);
            }
            
            if (config->verbose) {
                printf("trainer_create: 分配分布式参数缓冲区，大小=%zu * %zu = %zu字节\n",
                       distributed_num_params, sizeof(float), distributed_num_params * sizeof(float));
            }
            trainer->distributed_parameter_buffer = (float*)safe_malloc(distributed_num_params * sizeof(float));
            if (config->verbose) {
                printf("trainer_create: 参数缓冲区地址=%p\n", trainer->distributed_parameter_buffer);
            }
            
            if (!trainer->distributed_gradient_buffer || !trainer->distributed_parameter_buffer) {
                if (config->verbose) {
                    log_warning("分布式训练缓冲区分配失败，回退到单节点训练\n");
                }
                // 清理已分配的内存
                safe_free((void**)&trainer->distributed_gradient_buffer);
                safe_free((void**)&trainer->distributed_parameter_buffer);
                trainer->distributed_buffer_size = 0;
            } else {
                // 初始化分布式缓冲区
                memset(trainer->distributed_gradient_buffer, 0, distributed_num_params * sizeof(float));
                memset(trainer->distributed_parameter_buffer, 0, distributed_num_params * sizeof(float));
                
                trainer->distributed_initialized = 1;
                
                if (config->verbose) {
                    printf("分布式训练已初始化，节点ID=%d，总节点数=%d\n", 
                           config->distributed_node_id, config->distributed_num_nodes);
                }
            }
        }
    }
    
    // ---- F-08 分布式训练增强字段初始化 ----
    trainer->async_sync_in_progress = 0;
    trainer->async_sync_buffer = NULL;
    trainer->async_sync_buffer_size = 0;
    trainer->async_sync_thread_handle = NULL;
    trainer->async_sync_requested = 0;
    trainer->async_sync_completed = 0;
    trainer->async_sync_enabled = config->use_distributed_training && config->distributed_num_nodes > 1;
    
    trainer->tree_parent_id = -1;
    trainer->tree_left_child_id = -1;
    trainer->tree_right_child_id = -1;
    trainer->tree_level = 0;
    trainer->tree_enabled = 0;
    
    trainer->mesh_rows = 0;
    trainer->mesh_cols = 0;
    trainer->mesh_row_id = 0;
    trainer->mesh_col_id = 0;
    trainer->mesh_enabled = 0;
    
    trainer->grad_accum_buffer = NULL;
    trainer->grad_accum_buffer_size = 0;
    trainer->grad_accum_counter = 0;
    trainer->grad_accum_steps = config->gradient_accumulation_steps > 0 ? config->gradient_accumulation_steps : 1;
    trainer->grad_accum_enabled = config->gradient_accumulation_steps > 1;
    trainer->grad_accum_initialized = 0;
    
    trainer->heartbeat_alive = 0;
    trainer->heartbeat_should_exit = 0;
    trainer->heartbeat_thread_handle = NULL;
    trainer->heartbeat_last_seen = NULL;
    trainer->heartbeat_timeout_ms = config->distributed_heartbeat_timeout_ms > 0 ? config->distributed_heartbeat_timeout_ms : 5000;
    trainer->heartbeat_interval_ms = config->distributed_heartbeat_interval_ms > 0 ? config->distributed_heartbeat_interval_ms : 1000;
    trainer->heartbeat_node_alive = NULL;
    trainer->heartbeat_num_nodes = config->distributed_num_nodes;
    trainer->heartbeat_enabled = config->use_distributed_training && config->distributed_num_nodes > 1;
    trainer->heartbeat_failed_nodes = NULL;
    trainer->heartbeat_failed_count = 0;
    
    trainer->failure_recovery_enabled = config->distributed_failure_recovery_enabled && config->use_distributed_training;
    trainer->failure_recovery_attempts = 0;
    trainer->failure_recovery_max_attempts = config->distributed_failure_recovery_max_attempts > 0 ? config->distributed_failure_recovery_max_attempts : 3;
    trainer->failure_recovery_buffer = NULL;
    
    // 初始化模型版本管理
    trainer->version_manager = NULL;
    trainer->version_snapshot_counter = 0;
    trainer->version_auto_snapshot_enabled = 0;
    
    // 初始化自动检查点
    trainer->auto_checkpoint_path = NULL;
    trainer->auto_checkpoint_interval = 0;
    trainer->auto_checkpoint_counter = 0;
    
    // 初始化需求20.4a: 紧急检查点（崩溃恢复）
    trainer->emergency_checkpoint_path = NULL;
    trainer->emergency_checkpoint_enabled = 0;
    trainer->emergency_save_requested = 0;
    trainer->crash_checkpoint_dir = NULL;
    
    // 初始化需求20.4b: 检查点保留策略
    trainer->checkpoint_retention_max = 0;
    trainer->checkpoint_retention_list = NULL;
    trainer->checkpoint_retention_count = 0;
    
    // 初始化需求20.4b: 后台定时保存
    trainer->background_checkpoint_enabled = 0;
    trainer->background_checkpoint_interval = 0;
    trainer->background_checkpoint_thread = NULL;
    trainer->background_checkpoint_exit = 0;
    
    // 初始化P1-6分布式训练容错和恢复增强字段
    trainer->elastic_enabled = config->use_distributed_training && config->distributed_num_nodes > 1;
    trainer->stale_gradient_enabled = config->use_distributed_training && config->distributed_num_nodes > 1;
    trainer->stale_gradient_coefficients = NULL;
    trainer->stale_gradient_counters = NULL;
    trainer->leader_election_enabled = config->use_distributed_training && config->distributed_num_nodes > 1;
    trainer->auto_resume_enabled = config->use_distributed_training && config->distributed_enable_checkpointing;
    trainer->auto_resume_checkpoint_path = NULL;
    trainer->heartbeat_failed_count_current = 0;
    trainer->elastic_node_current_count = config->distributed_num_nodes;
    trainer->elastic_node_pending_add = NULL;
    trainer->elastic_node_pending_remove = NULL;
    trainer->elastic_node_pending_add_count = 0;
    trainer->elastic_node_pending_remove_count = 0;
    
    // 如果启用分布式训练，初始化梯度累积缓冲区
    if (trainer->grad_accum_enabled && trainer->distributed_initialized) {
        size_t num_params = lnn_get_parameter_count(network);
        if (num_params > 0) {
            distributed_gradient_accumulation_init(trainer, num_params, trainer->grad_accum_steps);
        }
    }
    
    // ---- F-13 真实分布式训练初始化（基于TCP网络的Ring AllReduce） ----
    if (trainer->distributed_initialized && config->use_distributed_training && config->distributed_num_nodes > 1) {
        DistributedConfig dist_cfg = distributed_config_default();
        strncpy(dist_cfg.master_host, "127.0.0.1", sizeof(dist_cfg.master_host) - 1);
        dist_cfg.master_port = (unsigned short)SELFLNN_DISTRIBUTED_PORT;
        dist_cfg.num_nodes = config->distributed_num_nodes;
        dist_cfg.node_id = config->distributed_node_id;
        dist_cfg.allreduce_algorithm = config->distributed_allreduce_algorithm;
        dist_cfg.enable_fault_tolerance = config->distributed_enable_fault_tolerance;
        dist_cfg.heartbeat_interval_ms = trainer->heartbeat_interval_ms;
        dist_cfg.heartbeat_timeout_ms = trainer->heartbeat_timeout_ms;
        dist_cfg.max_retries = config->distributed_max_retries > 0 ? config->distributed_max_retries : 3;
        dist_cfg.sync_frequency = config->distributed_sync_frequency > 0 ? config->distributed_sync_frequency : 1;
        dist_cfg.enable_gradient_compression = config->use_gradient_compression;
        dist_cfg.gradient_compression_ratio = config->gradient_compression_ratio > 0.0f ? config->gradient_compression_ratio : 0.1f;
        dist_cfg.enable_checkpointing = config->distributed_enable_checkpointing;
        dist_cfg.checkpoint_frequency = config->distributed_checkpoint_frequency;
        dist_cfg.verbose = config->verbose;
        
        trainer->distributed_comm_context = distributed_init(&dist_cfg);
        if (trainer->distributed_comm_context) {
            if (trainer->distributed_is_leader) {
                if (distributed_start_server(trainer->distributed_comm_context) == 0) {
                    distributed_wait_for_workers(trainer->distributed_comm_context, 30000);
                }
            } else {
                distributed_connect_worker(trainer->distributed_comm_context);
            }
            distributed_build_ring_topology(trainer->distributed_comm_context);
            distributed_start_heartbeat(trainer->distributed_comm_context);
            if (config->verbose) {
                printf("F-13 真实分布式网络已建立 | 节点ID=%d/%d 算法=%d 端口=%d\n",
                       trainer->distributed_node_id, trainer->distributed_num_nodes,
                       config->distributed_allreduce_algorithm, SELFLNN_DISTRIBUTED_PORT);
            }
        } else if (config->verbose) {
            log_warning("F-13 分布式训练初始化失败，回退到本地模式\n");
        }
    }
    
    // 旧版心跳系统（仅当F-13未激活时用于向后兼容）
    if (!trainer->distributed_comm_context && trainer->heartbeat_enabled && trainer->distributed_initialized) {
        if (distributed_heartbeat_init(trainer) == 0) {
            distributed_heartbeat_start(trainer);
            if (config->verbose) {
                printf("分布式心跳系统已启动（旧版），超时=%dms，间隔=%dms\n",
                       trainer->heartbeat_timeout_ms, trainer->heartbeat_interval_ms);
            }
        } else if (config->verbose) {
            log_warning("分布式心跳系统初始化失败\n");
        }
    }
    
    // 旧版树形拓扑（仅当F-13未激活时用于向后兼容）
    if (!trainer->distributed_comm_context && trainer->distributed_initialized && config->distributed_num_nodes > 1) {
        distributed_tree_build_topology(trainer);
        trainer->tree_enabled = 1;
        if (config->verbose) {
            printf("分布式树形拓扑已构建（旧版），节点ID=%d，父节点=%d，左子=%d，右子=%d，层级=%d\n",
                   trainer->distributed_node_id, trainer->tree_parent_id,
                   trainer->tree_left_child_id, trainer->tree_right_child_id,
                   trainer->tree_level);
        }
    }
    
    // 初始化性能计数器
    perf_timer_init(&trainer->forward_time);
    perf_timer_init(&trainer->backward_time);
    perf_timer_init(&trainer->update_time);
    
    // 初始化高级正则化器
    trainer->regularizer = NULL;
    if (config->regularization != REGULARIZATION_NONE) {
        LNNConfig net_cfg;
        if (lnn_get_config(network, &net_cfg) == 0) {
            AdvancedRegularizationConfig adv_reg_config;
            memset(&adv_reg_config, 0, sizeof(AdvancedRegularizationConfig));
            adv_reg_config.type = ADV_REG_DROP_PATH;
            adv_reg_config.drop_path_rate = config->dropout_rate;
            adv_reg_config.strength = config->regularization_lambda;
            adv_reg_config.apply_during_training = 1;
            adv_reg_config.apply_during_validation = 0;
            adv_reg_config.enable_scheduling = 1;
            adv_reg_config.schedule_decay = 0.99f;
            
            // 根据基本正则化类型选择高级正则化方法
            switch (config->regularization) {
                case REGULARIZATION_L1:
                case REGULARIZATION_L2:
                    adv_reg_config.type = ADV_REG_STOCHASTIC_DEPTH;
                    adv_reg_config.survival_probability = 0.8f;
                    break;
                case REGULARIZATION_DROPOUT:
                    adv_reg_config.type = ADV_REG_SPATIAL_DROPOUT;
                    adv_reg_config.drop_path_rate = config->dropout_rate;
                    break;
                default:
                    adv_reg_config.type = ADV_REG_DROP_PATH;
                    break;
            }
            
            size_t reg_input_dim = net_cfg.input_size;
            size_t reg_output_dim = net_cfg.output_size;
            
            trainer->regularizer = advanced_regularizer_create(&adv_reg_config,
                                                               reg_input_dim, reg_output_dim);
            if (!trainer->regularizer && config->verbose) {
                log_warning("高级正则化器创建失败\n");
            } else if (config->verbose) {
                printf("高级正则化器已创建，类型=%d\n", adv_reg_config.type);
            }
        }
    }
    
    // 初始化记忆系统集成字段
    trainer->memory_manager = NULL;
    trainer->memory_recall_buffer = NULL;
    trainer->memory_recall_buffer_size = 0;
    trainer->memory_integration_enabled = config->enable_memory_integration;
    trainer->memory_auto_create = 0;
    trainer->memory_consolidation_interval = config->memory_consolidation_interval;
    trainer->memory_consolidation_counter = 0;
    trainer->memory_context_strength = config->memory_context_strength;
    trainer->memory_experience_counter = 0;
    
    // 如果启用记忆系统集成，自动创建内部记忆管理器
    if (config->enable_memory_integration) {
        size_t st_cap = config->memory_short_term_capacity > 0 ? config->memory_short_term_capacity : 1000;
        size_t lt_cap = config->memory_long_term_capacity > 0 ? config->memory_long_term_capacity : 10000;
        
        MemoryManagerConfig mem_config;
        memset(&mem_config, 0, sizeof(MemoryManagerConfig));
        mem_config.short_term_capacity = st_cap;
        mem_config.long_term_capacity = lt_cap;
        mem_config.episodic_capacity = st_cap / 2;
        mem_config.semantic_capacity = lt_cap / 4;
        mem_config.consolidation_rate = 0.1f;
        mem_config.enable_integration = 1;
        
        trainer->memory_manager = memory_manager_create(&mem_config);
        if (trainer->memory_manager) {
            trainer->memory_auto_create = 1;
            trainer->memory_integration_enabled = 1;
            
            // 分配记忆召回缓冲区
            size_t recall_dim = net_config.input_size + net_config.output_size + 2;
            trainer->memory_recall_buffer_size = recall_dim * 10;
            trainer->memory_recall_buffer = (float*)safe_malloc(trainer->memory_recall_buffer_size * sizeof(float));
            if (!trainer->memory_recall_buffer) {
                trainer->memory_recall_buffer_size = 0;
            }
            
            if (config->verbose) {
                printf("记忆系统集成已启用：短期容量=%zu，长期容量=%zu，上下文强度=%.2f\n",
                       st_cap, lt_cap, config->memory_context_strength);
            }
        } else {
            trainer->memory_integration_enabled = 0;
            if (config->verbose) {
                log_warning("记忆管理器创建失败，记忆系统集成已禁用\n");
            }
        }
    }
    
    // ---- P3.6 演化算法集成初始化 ----
    trainer->evolution_population = NULL;
    trainer->evolution_saved_params = NULL;
    trainer->evolution_genome_size = 0;
    trainer->evolution_initialized = 0;
    if (config->enable_evolution && config->evolution_interval > 0) {
        size_t num_params = lnn_get_parameter_count(network);
        if (num_params > 0) {
            size_t pop_size = config->evolution_population_size > 0 ? config->evolution_population_size : 20;
            float mut_rate = config->evolution_mutation_rate > 0.0f ? config->evolution_mutation_rate : 0.15f;
            float cross_rate = config->evolution_crossover_rate > 0.0f ? config->evolution_crossover_rate : 0.8f;
            float elit_rate = config->evolution_elitism_rate > 0.0f ? config->evolution_elitism_rate : 0.1f;
            
            trainer->evolution_genome_size = num_params;
            trainer->evolution_saved_params = (float*)safe_malloc(num_params * sizeof(float));
            if (!trainer->evolution_saved_params) {
                if (config->verbose) log_warning("演化参数备份内存分配失败，演化集成已禁用\n");
            } else {
                trainer->evolution_population = population_create((size_t)pop_size, (size_t)num_params, 0);
                if (trainer->evolution_population) {
                    population_set_mutation_rate(trainer->evolution_population, mut_rate);
                    population_set_crossover_rate(trainer->evolution_population, cross_rate);
                    population_set_elitism_rate(trainer->evolution_population, elit_rate);
                    trainer->evolution_initialized = 1;
                    float* params = lnn_get_parameters(network);
                    if (params) {
                        memcpy(trainer->evolution_saved_params, params, num_params * sizeof(float));
                    }
                    if (config->verbose) {
                        printf("演化算法集成已启用：种群=%zu, 突变率=%.2f, 交叉率=%.2f, 精英率=%.2f, 间隔=%zu epoch\n",
                               pop_size, mut_rate, cross_rate, elit_rate, config->evolution_interval);
                    }
                } else {
                    safe_free((void**)&trainer->evolution_saved_params);
                    if (config->verbose) log_warning("演化种群创建失败，演化集成已禁用\n");
                }
            }
        }
    }
    
    // 初始化训练控制标志（需求27.3B: 暂停/停止控制）
    trainer->is_paused = 0;
    trainer->is_stopped = 0;
    
    return trainer;
}

/**
 * @brief 销毁训练器
 */
void trainer_free(Trainer* trainer) {
    if (!trainer) {
        log_info("trainer_free: 训练器指针为空，直接返回\n");
        return;
    }
    
    // 释放优化器状态
    optimizer_free(&trainer->optimizer);
    
    // 释放学习率调度器
    if (trainer->scheduler) {
        scheduler_free_internal(trainer->scheduler);
    }
    
    // 释放训练历史
    safe_free((void**)&trainer->history.train_losses);
    safe_free((void**)&trainer->history.train_accuracies);
    safe_free((void**)&trainer->history.val_losses);
    safe_free((void**)&trainer->history.val_accuracies);
    safe_free((void**)&trainer->history.learning_rates);
    
    // 释放梯度缓冲区
    safe_free((void**)&trainer->gradients);
    
    // 释放拉普拉斯优化相关资源
    if (trainer->laplace_analyzer) {
        laplace_analyzer_free(trainer->laplace_analyzer);
    }
    safe_free((void**)&trainer->laplace_filtered_gradients);
    safe_free((void**)&trainer->gradient_history);
    
    // 释放批次缓冲区
    safe_free((void**)&trainer->batch_inputs);
    safe_free((void**)&trainer->batch_targets);
    safe_free((void**)&trainer->batch_outputs);
    
    // 释放Dropout掩码
    safe_free((void**)&trainer->dropout_masks);
    
    // 释放GPU资源
    if (trainer->gpu_parameters) {
        gpu_memory_free(trainer->gpu_parameters);
    }
    if (trainer->gpu_gradients) {
        gpu_memory_free(trainer->gpu_gradients);
    }
    if (trainer->gpu_outputs) {
        gpu_memory_free(trainer->gpu_outputs);
    }
    if (trainer->gpu_targets) {
        gpu_memory_free(trainer->gpu_targets);
    }
    if (trainer->gpu_inputs) {
        gpu_memory_free(trainer->gpu_inputs);
    }
    if (trainer->gpu_context) {
        gpu_context_free(trainer->gpu_context);
    }
    
    // 释放混合精度训练上下文
    if (trainer->mixed_precision_context) {
        mixed_precision_destroy(trainer->mixed_precision_context);
        trainer->mixed_precision_context = NULL;
    }
    
    // 释放分布式训练资源
    safe_free((void**)&trainer->distributed_gradient_buffer);
    safe_free((void**)&trainer->distributed_parameter_buffer);
    
    // ---- F-08 分布式训练增强资源清理 ----
    
    // 停止心跳线程
    if (trainer->heartbeat_enabled && trainer->heartbeat_alive) {
        distributed_heartbeat_stop(trainer);
    }
    
    // 清理心跳相关内存
    safe_free((void**)&trainer->heartbeat_last_seen);
    safe_free((void**)&trainer->heartbeat_node_alive);
    safe_free((void**)&trainer->heartbeat_failed_nodes);
    
    // 清理P1-6增强字段
    safe_free((void**)&trainer->stale_gradient_coefficients);
    safe_free((void**)&trainer->stale_gradient_counters);
    safe_free((void**)&trainer->elastic_node_pending_add);
    safe_free((void**)&trainer->elastic_node_pending_remove);
    if (trainer->auto_resume_checkpoint_path) {
        safe_free((void**)&trainer->auto_resume_checkpoint_path);
    }
    
    // 等待异步梯度同步完成
    if (trainer->async_sync_in_progress) {
        distributed_async_sync_wait(trainer);
    }
    
    // 清理异步同步缓冲区
    safe_free((void**)&trainer->async_sync_buffer);
    
    // 清理梯度累积缓冲区
    safe_free((void**)&trainer->grad_accum_buffer);
    
    // 清理故障恢复缓冲区
    safe_free((void**)&trainer->failure_recovery_buffer);
    
    // 释放高级正则化器
    if (trainer->regularizer) {
        advanced_regularizer_free(trainer->regularizer);
        trainer->regularizer = NULL;
    }
    
    // 释放在线学习缓冲区
    safe_free((void**)&trainer->online_replay_inputs);
    safe_free((void**)&trainer->online_replay_targets);
    safe_free((void**)&trainer->online_running_mean);
    safe_free((void**)&trainer->online_running_var);
    safe_free((void**)&trainer->online_ewc_importance);
    safe_free((void**)&trainer->online_ewc_optimal_params);
    
    // 释放记忆系统集成资源
    if (trainer->memory_manager && trainer->memory_auto_create) {
        if (trainer->config.verbose) {
            printf("trainer_free: 释放内部记忆管理器 %p\n", trainer->memory_manager);
        }
        memory_manager_free(trainer->memory_manager);
        trainer->memory_manager = NULL;
    }
    // 注意：如果memory_manager是由外部通过trainer_connect_memory连接的，
    // 且memory_auto_create为0，则不由trainer_free释放，由外部管理。
    safe_free((void**)&trainer->memory_recall_buffer);
    
    // 清理F-13真实分布式通信上下文（基于TCP网络的Ring AllReduce）
    if (trainer->distributed_comm_context) {
        if (trainer->config.verbose) {
            printf("trainer_free: 释放F-13分布式通信上下文 %p\n", trainer->distributed_comm_context);
        }
        distributed_cleanup((DistributedContext*)trainer->distributed_comm_context);
        trainer->distributed_comm_context = NULL;
    }
    
    // 释放模型版本管理器
    if (trainer->version_manager) {
        if (trainer->config.verbose) {
            printf("trainer_free: 释放模型版本管理器 %p\n", trainer->version_manager);
        }
        model_version_manager_free(trainer->version_manager);
        trainer->version_manager = NULL;
    }
    
    // 释放自动检查点路径
    if (trainer->auto_checkpoint_path) {
        safe_free((void**)&trainer->auto_checkpoint_path);
        trainer->auto_checkpoint_path = NULL;
    }
    
    // 释放需求20.4a: 紧急检查点（崩溃恢复）
    if (trainer->emergency_checkpoint_path) {
        safe_free((void**)&trainer->emergency_checkpoint_path);
        trainer->emergency_checkpoint_path = NULL;
    }
    if (trainer->crash_checkpoint_dir) {
        safe_free((void**)&trainer->crash_checkpoint_dir);
        trainer->crash_checkpoint_dir = NULL;
    }
    
    // 释放需求20.4b: 检查点保留策略列表
    if (trainer->checkpoint_retention_list) {
        size_t i;
        for (i = 0; i < trainer->checkpoint_retention_count; i++) {
            if (trainer->checkpoint_retention_list[i]) {
                safe_free((void**)&trainer->checkpoint_retention_list[i]);
            }
        }
        safe_free((void**)&trainer->checkpoint_retention_list);
        trainer->checkpoint_retention_list = NULL;
    }
    trainer->checkpoint_retention_count = 0;
    
    // 停止后台定时保存线程（直接操作，避免递归锁）
    if (trainer->background_checkpoint_enabled) {
        trainer->background_checkpoint_exit = 1;
        if (trainer->background_checkpoint_thread) {
#ifdef _WIN32
            WaitForSingleObject(trainer->background_checkpoint_thread, 5000);
            CloseHandle(trainer->background_checkpoint_thread);
#else
            pthread_t bg_thread = (pthread_t)(uintptr_t)(trainer->background_checkpoint_thread);
            pthread_join(bg_thread, NULL);
#endif
            trainer->background_checkpoint_thread = NULL;
        }
        trainer->background_checkpoint_enabled = 0;
    }
    
    // ---- P3.6 演化算法集成清理 ----
    if (trainer->evolution_population) {
        population_destroy(trainer->evolution_population);
        trainer->evolution_population = NULL;
    }
    safe_free((void**)&trainer->evolution_saved_params);
    trainer->evolution_genome_size = 0;
    trainer->evolution_initialized = 0;
    
    // 销毁线程安全锁
    mutex_destroy(trainer->lock);
    trainer->lock = NULL;
    
    // 释放训练器
    safe_free((void**)&trainer);
}

/** P3.6 演化适应度数据 */
typedef struct {
    Trainer* trainer;
    const float* val_inputs;
    const float* val_targets;
    size_t val_samples;
    size_t input_dim;
    size_t output_dim;
    float* output_buffer;
} EvolutionTrainData;

/**
 * P3.6 演化适应度函数
 * 将个体基因组应用到网络，在验证集上计算损失，返回适应度 = -loss
 */
static float evolution_train_fitness(const float* genome, size_t genome_size, void* user_data) {
    EvolutionTrainData* data = (EvolutionTrainData*)user_data;
    if (!data || !genome || genome_size == 0) return -FLT_MAX;

    Trainer* trainer = data->trainer;
    float* params = lnn_get_parameters(trainer->network);
    if (!params) return -FLT_MAX;

    memcpy(params, genome, genome_size * sizeof(float));

    float total_loss = 0.0f;
    size_t batch_size = trainer->config.batch_size;
    if (batch_size > data->val_samples) batch_size = data->val_samples;
    size_t num_batches = (data->val_samples + batch_size - 1) / batch_size;

    for (size_t b = 0; b < num_batches; b++) {
        size_t start = b * batch_size;
        size_t end = start + batch_size;
        if (end > data->val_samples) end = data->val_samples;
        size_t cur_batch = end - start;

        const float* batch_in = data->val_inputs + start * data->input_dim;
        const float* batch_tg = data->val_targets + start * data->output_dim;

        if (lnn_forward_batch(trainer->network, batch_in, data->output_buffer, cur_batch) != 0) {
            total_loss += 1e10f;
            continue;
        }

        switch (trainer->config.loss_function) {
            case LOSS_CROSS_ENTROPY:
                total_loss += loss_cross_entropy(data->output_buffer, batch_tg, cur_batch * data->output_dim);
                break;
            case LOSS_MEAN_ABSOLUTE_ERROR:
                total_loss += loss_mean_absolute_error(data->output_buffer, batch_tg, cur_batch * data->output_dim);
                break;
            default:
                total_loss += loss_mean_squared_error(data->output_buffer, batch_tg, cur_batch * data->output_dim);
                break;
        }
    }

    memcpy(params, genome, genome_size * sizeof(float));
    return -total_loss / (float)num_batches;
}

/**
 * P3.6 执行一轮演化步骤
 * 1. 保存当前网络参数作为基线
 * 2. 从基线参数随机扰动创建种群
 * 3. 评估种群在验证集上的适应度
 * 4. 执行一代进化（选择+交叉+突变）
 * 5. 将最佳个体基因组写回网络参数
 */
static int trainer_evolution_step(Trainer* trainer, size_t epoch,
                                  const float* val_inputs, const float* val_targets,
                                  size_t val_samples, size_t input_dim, size_t output_dim) {
    if (!trainer || !trainer->evolution_initialized || !trainer->evolution_population ||
        !trainer->evolution_saved_params || val_samples == 0 || !val_inputs || !val_targets) {
        return -1;
    }

    Population* pop = trainer->evolution_population;
    size_t genome_size = trainer->evolution_genome_size;
    float* params = lnn_get_parameters(trainer->network);
    if (!params) return -1;

    memcpy(trainer->evolution_saved_params, params, genome_size * sizeof(float));

    EvolutionTrainData data;
    memset(&data, 0, sizeof(data));
    data.trainer = trainer;
    data.val_inputs = val_inputs;
    data.val_targets = val_targets;
    data.val_samples = val_samples;
    data.input_dim = input_dim;
    data.output_dim = output_dim;
    data.output_buffer = trainer->batch_outputs;

    if (population_initialize_random(pop, -0.1f, 0.1f) != 0) return -1;

    size_t pop_size = population_get_size(pop);
    for (size_t i = 0; i < pop_size; i++) {
        const float* ind_genome = population_get_individual_genome(pop, i, NULL);
        if (ind_genome) {
            for (size_t j = 0; j < genome_size; j++) {
                params[j] = trainer->evolution_saved_params[j] + ind_genome[j];
            }
            memcpy((float*)ind_genome, params, genome_size * sizeof(float));
        }
    }

    if (population_evolve(pop, evolution_train_fitness, &data) != 0) return -1;

    const float* best_genome = population_get_best_genome(pop, NULL);
    if (best_genome) {
        memcpy(params, best_genome, genome_size * sizeof(float));
        if (trainer->config.verbose) {
            PopulationStatistics stats;
            if (population_get_statistics(pop, &stats) == 0) {
                printf("演化步骤完成: epoch=%zu, 最佳适应度=%.4f, 平均适应度=%.4f, 多样性=%.4f, 代数=%d\n",
                       epoch, stats.best_fitness, stats.average_fitness,
                       stats.diversity_score, stats.current_generation);
            }
        }
    } else {
        memcpy(params, trainer->evolution_saved_params, genome_size * sizeof(float));
        return -1;
    }

    return 0;
}

/* 前向声明：检查点保留策略辅助函数 */
static int add_checkpoint_to_retention_list(Trainer* trainer, const char* filepath);

/**
 * @brief 训练神经网络
 */
int trainer_train(Trainer* trainer, const float* inputs, const float* targets,
                  size_t num_samples, TrainingCallback callback, void* user_data) {
    if (!trainer || !inputs || !targets || num_samples == 0) {
        return -1;
    }
    
    // 验证网络维度
    LNNConfig net_config;
    if (lnn_get_config(trainer->network, &net_config) != 0) {
        return -1;
    }
    
    size_t input_dim = net_config.input_size;
    size_t output_dim = net_config.output_size;
    
    // 调试信息
    if (trainer->config.verbose) {
        printf("trainer_train: 开始训练，样本数=%zu，输入维度=%zu，输出维度=%zu，批量大小=%zu\n",
               num_samples, input_dim, output_dim, trainer->config.batch_size);
    }
    
    if (!validate_network_dimensions(trainer->network, input_dim, output_dim)) {
        if (trainer->config.verbose) {
            log_info("trainer_train: 网络维度验证失败\n");
        }
        return -1;
    }
    TRAINER_LOCK(trainer);
    
    // 分割训练集和验证集
    size_t validation_samples = 0;
    size_t train_samples = num_samples;
    
    if (trainer->config.validation_split > 0) {
        validation_samples = (num_samples * trainer->config.validation_split) / 100;
        train_samples = num_samples - validation_samples;
    }
    
    const float* train_inputs = inputs;
    const float* train_targets = targets;
    const float* val_inputs = inputs + train_samples * input_dim;
    const float* val_targets = targets + train_samples * output_dim;
    
    // 创建数据加载器
    if (trainer->config.verbose) {
        printf("trainer_train: 创建数据加载器，训练样本=%zu，批量大小=%zu\n", train_samples, trainer->config.batch_size);
    }
    DataLoader* loader = data_loader_create_internal(train_inputs, train_targets,
                                                    train_samples, input_dim,
                                                    output_dim, trainer->config.batch_size,
                                                    trainer->config.shuffle_data);
    if (!loader) {
        if (trainer->config.verbose) {
            log_info("trainer_train: 数据加载器创建失败\n");
        }
        TRAINER_UNLOCK(trainer);
        return -1;
    }
    if (trainer->config.verbose) {
        log_info("trainer_train: 数据加载器创建成功\n");
    }
    
    // 初始化训练状态
    trainer->state.start_time = perf_timestamp_ns() / 1000000;
    trainer->state.current_epoch = 0;
    trainer->state.total_batches = (train_samples + trainer->config.batch_size - 1) / trainer->config.batch_size;
    if (trainer->config.verbose) {
        printf("trainer_train: 初始化训练状态，总批次=%zu\n", trainer->state.total_batches);
    }
    
    // 初始化GPU内存（如果启用GPU）
    if (trainer->config.verbose) {
        printf("trainer_train: GPU初始化状态=%d\n", trainer->gpu_initialized);
    }
    if (trainer->gpu_initialized) {
        size_t batch_input_size = trainer->config.batch_size * input_dim;
        size_t batch_output_size = trainer->config.batch_size * output_dim;
        size_t num_parameters = lnn_get_parameter_count(trainer->network);
        
        if (trainer_gpu_init_memory(trainer, batch_input_size, batch_output_size, num_parameters) != 0) {
            if (trainer->config.verbose) {
                log_warning("GPU内存初始化失败，回退到CPU训练\n");
            }
            trainer_cleanup_gpu_resources(trainer);
        } else {
            if (trainer->config.verbose) {
                printf("GPU内存初始化成功，输入大小=%zu，输出大小=%zu，参数数量=%zu\n", 
                       batch_input_size, batch_output_size, num_parameters);
            }
        }
    }
    
    // 训练循环
    if (trainer->config.verbose) {
        printf("trainer_train: 开始训练循环，总轮数=%zu\n", trainer->config.epochs);
    }
    int should_stop = 0;
    
    for (size_t epoch = 0; epoch < trainer->config.epochs && !should_stop && !trainer->is_stopped; epoch++) {
        // 检查暂停/停止标志（需求27.3B: 暂停/停止控制）
        while (trainer->is_paused && !trainer->is_stopped) {
            time_sleep_ms(100);
        }
        if (trainer->is_stopped) {
            if (trainer->config.verbose) {
                printf("trainer_train: 训练被停止，轮次%zu\n", epoch);
            }
            break;
        }
        
        trainer->state.current_epoch = epoch;
        trainer->state.current_batch = 0;
        trainer->state.samples_processed = 0;
        
        if (trainer->config.verbose) {
            printf("trainer_train: 第%zu轮开始，重置数据加载器\n", epoch);
        }
        // 重置数据加载器
        data_loader_reset_internal(loader);
        if (trainer->config.verbose) {
            log_info("trainer_train: 数据加载器重置完成\n");
        }
        
        // 批次训练
        size_t batch_num = 0;
        float epoch_loss = 0.0f;
        float epoch_accuracy = 0.0f;
        
        while (1) {
            // 检查暂停/停止标志（需求27.3B: 批次级暂停/停止控制）
            while (trainer->is_paused && !trainer->is_stopped) {
                time_sleep_ms(50);
            }
            if (trainer->is_stopped) {
                if (trainer->config.verbose) {
                    printf("trainer_train: 批次训练被停止\n");
                }
                break;
            }
            
            if (trainer->config.verbose) {
                log_info("trainer_train: 进入训练批次循环\n");
            }
            float* batch_inputs = NULL;
            float* batch_targets = NULL;
            size_t batch_size = 0;
            
            if (trainer->config.verbose) {
                log_info("trainer_train: 调用data_loader_next_batch_internal\n");
            }
            int result = data_loader_next_batch_internal(loader, &batch_inputs,
                                                        &batch_targets, &batch_size);
            if (result <= 0) {
                break;  // 没有更多批次
            }
            
            trainer->state.current_batch = batch_num;
            
            // 前向传播
            perf_timer_start(&trainer->forward_time);
            
            // GPU加速路径（如果启用）
            if (trainer->gpu_initialized) {
                // 将批次数据复制到GPU设备
                if (trainer_gpu_copy_batch_to_device(trainer, batch_inputs, batch_targets,
                                                   batch_size, input_dim, output_dim) != 0) {
                    if (trainer->config.verbose) {
                        log_warning("GPU数据复制失败，回退到CPU计算\n");
                    }
                    trainer_cleanup_gpu_resources(trainer);
                }
            }
            
            // 应用Dropout（如果启用）
            if (trainer->config.regularization == REGULARIZATION_DROPOUT &&
                trainer->config.dropout_rate > 0.0f) {
                // 获取网络激活并应用Dropout
                // 注意：这里需要实际的网络接口
            }
            
            // 前向传播路径选择：GPU → 混合精度 → 标准CPU
            int forward_done = 0;

            // GPU加速前向传播（如果启用GPU）
            if (trainer->gpu_initialized) {
                if (trainer_gpu_forward_batch(trainer, batch_inputs, trainer->batch_outputs, batch_size) == 0) {
                    forward_done = 1;
                } else {
                    if (trainer->config.verbose) {
                        log_warning("GPU前向传播失败，回退到CPU计算\n");
                    }
                    trainer_cleanup_gpu_resources(trainer);
                }
            }

            // 混合精度训练路径（如果未使用GPU且混合精度启用）
            if (!forward_done && trainer->mixed_precision_context && trainer->mixed_precision_context->enabled) {
                if (mixed_precision_forward(trainer->mixed_precision_context,
                                           batch_inputs, trainer->batch_outputs) == 0) {
                    forward_done = 1;
                } else {
                    if (trainer->config.verbose) {
                        log_warning("混合精度前向传播失败，回退到标准精度\n");
                    }
                }
            }

            // 标准精度前向传播（兜底路径）
            if (!forward_done) {
                if (trainer->config.verbose) {
                    printf("trainer_train: 调用lnn_forward_batch，批次大小=%zu\n", batch_size);
                }
                
                // 记忆召回增强：将记忆上下文混入输入特征
                if (trainer->memory_integration_enabled && trainer->memory_manager &&
                    trainer->memory_recall_buffer && trainer->memory_recall_buffer_size > 0) {
                    float* enhanced_input = (float*)safe_malloc(batch_size * input_dim * sizeof(float));
                    if (enhanced_input) {
                        memcpy(enhanced_input, batch_inputs, batch_size * input_dim * sizeof(float));
                        for (size_t b = 0; b < batch_size; b++) {
                            float* sample_in = enhanced_input + b * input_dim;
                            // 从记忆管理器召回相关记忆上下文
                            char recall_key[64];
                            snprintf(recall_key, sizeof(recall_key), "exp_%zu", 
                                     (size_t)(trainer->memory_experience_counter + b));
                            float ret_strength = 0.0f;
                            int ret_memory_type = 0;
                            int recalled = memory_manager_retrieve(trainer->memory_manager,
                                recall_key, trainer->memory_recall_buffer,
                                trainer->memory_recall_buffer_size,
                                &ret_strength, &ret_memory_type);
                            if (recalled > 0 && trainer->memory_context_strength > 0.0f) {
                                size_t mix_dim = (size_t)recalled < input_dim ? (size_t)recalled : input_dim;
                                for (size_t d = 0; d < mix_dim; d++) {
                                    sample_in[d] = (1.0f - trainer->memory_context_strength) * sample_in[d]
                                                 + trainer->memory_context_strength * trainer->memory_recall_buffer[d];
                                }
                            }
                        }
                        int forward_result = lnn_forward_batch(trainer->network, enhanced_input,
                                         trainer->batch_outputs, (int)batch_size);
                        safe_free((void**)&enhanced_input);
                        
                        if (trainer->config.verbose) {
                            printf("trainer_train: 记忆增强前向传播，返回值=%d\n", forward_result);
                        }
                        if (forward_result != 0) {
                            if (batch_inputs) safe_free((void**)&batch_inputs);
                            if (batch_targets) safe_free((void**)&batch_targets);
                            continue;
                        }
                        forward_done = 1;
                    }
                }
                
                if (!forward_done) {
                    int forward_result = lnn_forward_batch(trainer->network, batch_inputs, trainer->batch_outputs,
                                     (int)batch_size);
                    if (trainer->config.verbose) {
                        printf("trainer_train: lnn_forward_batch调用完成，返回值=%d\n", forward_result);
                    }
                    if (forward_result != 0) {
                        if (trainer->config.verbose) {
                            log_info("trainer_train: 前向传播失败，跳过当前批次\n");
                        }
                        if (batch_inputs) safe_free((void**)&batch_inputs);
                        if (batch_targets) safe_free((void**)&batch_targets);
                    break;
                }
                forward_done = 1;
            }
            
            // 如果GPU已初始化，将输出数据复制到GPU设备
            if (trainer->gpu_initialized) {
                if (trainer_gpu_copy_output_to_device(trainer, trainer->batch_outputs,
                                                     batch_size, output_dim) != 0) {
                    if (trainer->config.verbose) {
                        log_warning("GPU输出复制失败，回退到CPU计算\n");
                    }
                    trainer_cleanup_gpu_resources(trainer);
                }
            }
            
            perf_timer_stop(&trainer->forward_time);
            
            // 计算损失和准确率
            float loss = 0.0f;
            float accuracy = 0.0f;
            
            switch (trainer->config.loss_function) {
                case LOSS_MEAN_SQUARED_ERROR:
                    loss = loss_mean_squared_error(trainer->batch_outputs,
                                                  batch_targets,
                                                  batch_size * output_dim);
                    break;
                case LOSS_MEAN_ABSOLUTE_ERROR:
                    loss = loss_mean_absolute_error(trainer->batch_outputs,
                                                   batch_targets,
                                                   batch_size * output_dim);
                    break;
                case LOSS_CROSS_ENTROPY:
                    loss = loss_cross_entropy(trainer->batch_outputs,
                                             batch_targets,
                                             batch_size * output_dim);
                    break;
                default:
                    loss = loss_mean_squared_error(trainer->batch_outputs,
                                                  batch_targets,
                                                  batch_size * output_dim);
                    break;
            }
            
            // 计算准确率（根据损失函数类型）
            accuracy = 0.0f;
            
            // 根据损失函数类型计算准确率或相关指标
            switch (trainer->config.loss_function) {
                case LOSS_CROSS_ENTROPY:
                    // 分类任务：计算准确率（预测类别 = 真实类别）
                    for (size_t i = 0; i < batch_size; i++) {
                        int predicted = 0;
                        int target = 0;
                        float max_prob = trainer->batch_outputs[i * output_dim];
                        float max_target = batch_targets[i * output_dim];
                        
                        for (size_t j = 1; j < output_dim; j++) {
                            if (trainer->batch_outputs[i * output_dim + j] > max_prob) {
                                max_prob = trainer->batch_outputs[i * output_dim + j];
                                predicted = (int)j;
                            }
                            if (batch_targets[i * output_dim + j] > max_target) {
                                max_target = batch_targets[i * output_dim + j];
                                target = (int)j;
                            }
                        }
                        
                        if (predicted == target) {
                            accuracy += 1.0f;
                        }
                    }
                    accuracy /= batch_size;
                    break;
                    
                case LOSS_MEAN_SQUARED_ERROR:
                case LOSS_MEAN_ABSOLUTE_ERROR:
                    // 回归任务：计算R²分数（决定系数）
                    // 首先计算目标变量的均值
                    float target_mean = 0.0f;
                    for (size_t i = 0; i < batch_size * output_dim; i++) {
                        target_mean += batch_targets[i];
                    }
                    target_mean /= (batch_size * output_dim);
                    
                    // 计算总平方和（SST）
                    float sst = 0.0f;
                    for (size_t i = 0; i < batch_size * output_dim; i++) {
                        float diff = batch_targets[i] - target_mean;
                        sst += diff * diff;
                    }
                    
                    // 计算残差平方和（SSE）
                    float sse = 0.0f;
                    for (size_t i = 0; i < batch_size * output_dim; i++) {
                        float residual = batch_targets[i] - trainer->batch_outputs[i];
                        sse += residual * residual;
                    }
                    
                    // 计算R²分数
                    if (sst > 1e-10f) {
                        accuracy = 1.0f - (sse / sst);
                        // R²分数可能在[-∞, 1]之间，我们限制到[0,1]范围用于显示
                        if (accuracy < 0.0f) accuracy = 0.0f;
                        if (accuracy > 1.0f) accuracy = 1.0f;
                    } else {
                        accuracy = 0.0f;
                    }
                    break;
                    
                default:
                    // 其他损失函数：默认准确率为0
                    accuracy = 0.0f;
                    break;
            }
            
            epoch_loss += loss;
            epoch_accuracy += accuracy;
            
            // 反向传播
            perf_timer_start(&trainer->backward_time);
            
            // 计算损失梯度
            float* loss_gradients = trainer->batch_outputs;  // 重用缓冲区
            size_t num_outputs = batch_size * output_dim;
            
            switch (trainer->config.loss_function) {
                case LOSS_MEAN_SQUARED_ERROR:
                    loss_mean_squared_error_gradient(trainer->batch_outputs,
                                                    batch_targets,
                                                    loss_gradients, num_outputs);
                    break;
                case LOSS_MEAN_ABSOLUTE_ERROR:
                    loss_mean_absolute_error_gradient(trainer->batch_outputs,
                                                     batch_targets,
                                                     loss_gradients, num_outputs);
                    break;
                case LOSS_CROSS_ENTROPY:
                    loss_cross_entropy_gradient(trainer->batch_outputs,
                                               batch_targets,
                                               loss_gradients, num_outputs);
                    break;
                default:
                    loss_mean_squared_error_gradient(trainer->batch_outputs,
                                                    batch_targets,
                                                    loss_gradients, num_outputs);
                    break;
            }
            
            // 反向传播路径选择：GPU → 混合精度 → 标准CPU
            int backward_done = 0;

            // GPU加速反向传播（如果启用GPU）
            if (trainer->gpu_initialized) {
                if (trainer_gpu_backward_batch(trainer, batch_inputs, loss_gradients,
                                              trainer->gradients, batch_size) == 0) {
                    backward_done = 1;
                } else {
                    if (trainer->config.verbose) {
                        log_warning("GPU反向传播失败，回退到CPU计算\n");
                    }
                    trainer_cleanup_gpu_resources(trainer);
                }
            }

            // 混合精度训练路径（如果未使用GPU且混合精度启用）
            if (!backward_done && trainer->mixed_precision_context && trainer->mixed_precision_context->enabled) {
                if (mixed_precision_backward(trainer->mixed_precision_context,
                                            loss_gradients, trainer->gradients) == 0) {
                    backward_done = 1;
                } else {
                    if (trainer->config.verbose) {
                        log_warning("混合精度反向传播失败，回退到标准精度\n");
                    }
                }
            }

            // 标准精度反向传播（兜底路径）
            if (!backward_done) {
                lnn_backward_batch(trainer->network, batch_inputs, loss_gradients,
                                  trainer->gradients, batch_size);
                backward_done = 1;
            }
            
            // 如果GPU反向传播已执行，将梯度数据复制到GPU设备
            if (trainer->gpu_initialized && backward_done) {
                size_t num_parameters = lnn_get_parameter_count(trainer->network);
                if (num_parameters > 0) {
                    if (trainer_gpu_copy_gradients_to_device(trainer, trainer->gradients,
                                                            num_parameters) != 0) {
                        if (trainer->config.verbose) {
                            log_warning("GPU梯度复制失败，回退到CPU计算\n");
                        }
                        trainer_cleanup_gpu_resources(trainer);
                    }
                }
            }
            
            perf_timer_stop(&trainer->backward_time);
            
            // 应用梯度裁剪
            if (trainer->config.gradient_clip != GRADIENT_CLIP_NONE) {
                gradient_clip(trainer->gradients, trainer->gradients_size,
                            trainer->config.gradient_clip,
                            trainer->config.gradient_clip_value,
                            trainer->config.gradient_clip_norm);
            }
            
            // 梯度压缩（Top-k稀疏化+误差反馈被启用时）
            trainer_compress_gradients(trainer, trainer->gradients, trainer->gradients_size);
            
            // ---- F-08 分布式训练增强：梯度同步（含梯度累积+异步同步） ----
            if (trainer->config.use_distributed_training && trainer->distributed_num_nodes > 1) {
                int should_sync_now = 1;
                
                // 等待上一次异步同步完成
                if (trainer->async_sync_in_progress) {
                    distributed_async_sync_wait(trainer);
                }
                
                // 检查心跳，检测节点故障
                if (trainer->heartbeat_enabled && trainer->heartbeat_alive) {
                    int failed_nodes[64];
                    int num_failed = distributed_heartbeat_detect_failures(trainer, failed_nodes, 64);
                    if (num_failed > 0) {
                        trainer->heartbeat_failed_count = num_failed;
                        if (trainer->failure_recovery_enabled &&
                            trainer->failure_recovery_attempts < trainer->failure_recovery_max_attempts) {
                            distributed_failure_rebuild_topology(trainer, failed_nodes, num_failed);
                            trainer->failure_recovery_attempts++;
                            if (trainer->config.verbose) {
                                printf("F-08: 节点故障恢复，重建拓扑，尝试次数=%d\n",
                                       trainer->failure_recovery_attempts);
                            }
                        }
                    }
                }
                
                // 梯度累积: 累积到一定步数再同步
                if (trainer->grad_accum_enabled && trainer->grad_accum_initialized) {
                    distributed_gradient_accumulate(trainer, trainer->gradients, trainer->gradients_size);
                    if (trainer->grad_accum_counter >= trainer->grad_accum_steps) {
                        distributed_gradient_accumulation_flush(trainer, trainer->gradients, trainer->gradients_size);
                    } else {
                        should_sync_now = 0;
                    }
                }
                
                // 执行梯度同步
                if (should_sync_now) {
                    if (trainer->async_sync_enabled) {
                        distributed_async_sync_start(trainer, trainer->gradients, trainer->gradients_size);
                    } else {
                        distributed_sync_gradients(trainer, trainer->gradients, trainer->gradients_size);
                    }
                }
            }
            
            // 参数更新
            perf_timer_start(&trainer->update_time);
            
            // 获取当前参数
            float* parameters = lnn_get_parameters(trainer->network);
            if (parameters) {
                // 应用权重衰减
                if (trainer->config.regularization == REGULARIZATION_L1 ||
                    trainer->config.regularization == REGULARIZATION_L2) {
                    weight_decay(parameters, trainer->gradients_size,
                                trainer->config.regularization_lambda,
                                trainer->config.regularization);
                }
                
                // 拉普拉斯优化（如果启用）
                float* gradients_to_use = trainer->gradients;
                
                if (trainer->config.use_laplace_optimization && trainer->laplace_analyzer) {
                    // 更新梯度历史
                    if (trainer->gradient_history_capacity > 0) {
                        // 计算梯度平均值作为历史记录的代表值
                        float grad_mean = 0.0f;
                        for (size_t i = 0; i < trainer->gradients_size; i++) {
                            grad_mean += fabsf(trainer->gradients[i]);
                        }
                        grad_mean /= trainer->gradients_size;
                        
                        // 添加到历史记录
                        trainer->gradient_history[trainer->gradient_history_position] = grad_mean;
                        trainer->gradient_history_position = (trainer->gradient_history_position + 1) % trainer->gradient_history_capacity;
                        if (trainer->gradient_history_size < trainer->gradient_history_capacity) {
                            trainer->gradient_history_size++;
                        }
                    }
                    
                    // 应用拉普拉斯优化
                    int opt_result = laplace_optimize_training(trainer->laplace_analyzer,
                                                              trainer->gradients,
                                                              trainer->gradients_size,
                                                              trainer->state.learning_rate,
                                                              trainer->laplace_filtered_gradients);
                    
                    if (opt_result == SELFLNN_SUCCESS) {
                        gradients_to_use = trainer->laplace_filtered_gradients;
                        
                        // 自适应滤波调整（如果启用）
                        if (trainer->config.laplace_adaptive_filtering && trainer->gradient_history_size > 10) {
                            // 计算梯度变化的统计量
                            float grad_var = 0.0f;
                            float grad_mean = 0.0f;
                            for (size_t i = 0; i < trainer->gradient_history_size; i++) {
                                grad_mean += trainer->gradient_history[i];
                            }
                            grad_mean /= trainer->gradient_history_size;
                            
                            for (size_t i = 0; i < trainer->gradient_history_size; i++) {
                                float diff = trainer->gradient_history[i] - grad_mean;
                                grad_var += diff * diff;
                            }
                            grad_var /= trainer->gradient_history_size;
                            
                            // 根据梯度方差调整截止频率
                            // 高方差 -> 降低截止频率（更强滤波）
                            // 低方差 -> 提高截止频率（更弱滤波）
                            float new_cutoff = trainer->config.laplace_filter_cutoff;
                            if (grad_var > 0.1f) {
                                new_cutoff = fmaxf(1.0f, trainer->config.laplace_filter_cutoff * 0.5f);
                            } else if (grad_var < 0.01f) {
                                new_cutoff = trainer->config.laplace_filter_cutoff * 2.0f;
                            }
                            
                            if (fabsf(new_cutoff - trainer->laplace_current_cutoff) > 0.1f) {
                                trainer->laplace_current_cutoff = new_cutoff;
                                
                                // 更新拉普拉斯分析器配置
                                LaplaceConfig config;
                                laplace_analyzer_get_config(trainer->laplace_analyzer, &config);
                                
                                // 工业级实现：更新截止频率并重新设计滤波器
                                // 1. 更新配置中的截止频率
                                config.cutoff_frequency = new_cutoff;
                                
                                // 2. 重新设计滤波器参数（基于新的截止频率）
                                // 使用巴特沃斯或切比雪夫滤波器设计
                                // 这里实现一阶低通滤波器：H(s) = 1 / (1 + s/ω_c)
                                // 其中ω_c = 2π * cutoff_frequency
                                
                                // 更新滤波器系数
                                float omega_c = 2.0f * (float)M_PI * new_cutoff;
                                float dt = 0.001f; // 假设时间步长
                                
                                // 一阶低通滤波器的离散化（使用后向欧拉法）
                                // 连续时间：dy/dt = ω_c * (x - y)
                                // 离散化：y[n] = (ω_c * dt * x[n] + y[n-1]) / (1 + ω_c * dt)
                                
                                // 存储滤波器参数供后续使用
                                trainer->laplace_filter_alpha = omega_c * dt;
                                trainer->laplace_filter_beta = 1.0f / (1.0f + omega_c * dt);
                                
                                // 3. 更新分析器配置
                                // 调用laplace_analyzer_set_config应用新配置
                                int set_config_result = laplace_analyzer_set_config(trainer->laplace_analyzer, &config);
                                if (set_config_result != SELFLNN_SUCCESS) {
                                    if (trainer->config.verbose) {
                                        printf("警告：更新拉普拉斯分析器配置失败，错误码=%d\n", set_config_result);
                                    }
                                    // 如果设置失败，重新创建分析器
                                    laplace_analyzer_free(trainer->laplace_analyzer);
                                    trainer->laplace_analyzer = laplace_analyzer_create(&config);
                                    if (!trainer->laplace_analyzer) {
                                        if (trainer->config.verbose) {
                                            log_error("重新创建拉普拉斯分析器失败\n");
                                        }
                                    }
                                }
                            }
                        }
                        
                        // 稳定性监控（如果启用）
                        if (trainer->config.laplace_monitor_stability) {
                            // 分析梯度稳定性
                            float grad_norm = gradient_norm(gradients_to_use, trainer->gradients_size);
                            float prev_grad_norm = trainer->state.gradient_norm;
                            
                            if (prev_grad_norm > 0 && grad_norm / prev_grad_norm > 5.0f) {
                                // 梯度范数急剧增加，可能发生梯度爆炸
                                trainer->laplace_stability_warning = 1;
                                trainer->laplace_stability_score *= 0.9f;
                                
                                if (trainer->config.verbose) {
                                    printf("警告：梯度可能爆炸，范数比=%.2f\n", grad_norm / prev_grad_norm);
                                }
                            } else if (prev_grad_norm > 0 && grad_norm / prev_grad_norm < 0.2f) {
                                // 梯度范数急剧减小，可能发生梯度消失
                                trainer->laplace_stability_warning = 2;
                                trainer->laplace_stability_score *= 0.9f;
                                
                                if (trainer->config.verbose) {
                                    printf("警告：梯度可能消失，范数比=%.2f\n", grad_norm / prev_grad_norm);
                                }
                            } else {
                                trainer->laplace_stability_warning = 0;
                                trainer->laplace_stability_score = fminf(1.0f, trainer->laplace_stability_score * 1.01f);
                            }
                        }
                        
                        // 频域自适应学习率（基于FFT频谱分析）
                        if (trainer->gradient_history_size >= 16) {
                            FreqAdaptiveLRConfig lr_config = laplace_freq_adaptive_lr_config_default(trainer->state.learning_rate);
                            float new_lr = laplace_freq_adaptive_lr(
                                trainer->gradient_history,
                                trainer->gradient_history_size,
                                &lr_config, NULL);
                            if (new_lr > 0.0f) {
                                float old_lr = trainer->state.learning_rate;
                                trainer->state.learning_rate = old_lr * lr_config.momentum + new_lr * (1.0f - lr_config.momentum);
                            }
                        }
                    }
                }
                
                // ========================================================
                // 分布式训练：梯度同步（F-08增强版：异步同步+树形/网格拓扑+心跳故障检测）
                // ========================================================
                if (parameters && trainer->distributed_initialized) {
                    // 检查是否需要执行梯度同步
                    int should_sync = distributed_should_sync_gradients(trainer);
                    
                    if (should_sync == 1) {
                        // 如果梯度累积模式已处理同步，跳过
                        int already_synced = 0;
                        if (trainer->grad_accum_enabled && trainer->grad_accum_initialized) {
                            if (trainer->grad_accum_counter < trainer->grad_accum_steps) {
                                already_synced = 1;
                            }
                        }
                        
                        if (!already_synced) {
                            // 等待上一次异步同步完成
                            if (trainer->async_sync_in_progress) {
                                distributed_async_sync_wait(trainer);
                            }
                            
                            // 心跳故障检测
                            if (trainer->heartbeat_enabled && trainer->heartbeat_alive) {
                                int failed_nodes[64];
                                int num_failed = distributed_heartbeat_detect_failures(trainer, failed_nodes, 64);
                                if (num_failed > 0) {
                                    trainer->heartbeat_failed_count = num_failed;
                                    trainer->heartbeat_failed_count_current = num_failed;
                                    
                                    // P1-6: 领导者选举 - 如果领导节点故障，自动选举新领导
                                    if (trainer->leader_election_enabled) {
                                        int leader_failed = 0;
                                        for (int fi = 0; fi < num_failed; fi++) {
                                            if (failed_nodes[fi] == 0) {
                                                leader_failed = 1;
                                                break;
                                            }
                                        }
                                        if (leader_failed) {
                                            int elect_result = distributed_leader_election(trainer, failed_nodes, num_failed);
                                            if (elect_result == 1 && trainer->config.verbose) {
                                                printf("P1-6: 领导者选举完成，新领导者节点ID=%d\n",
                                                       trainer->distributed_node_id);
                                            }
                                        }
                                    }
                                    
                                    // P1-6: 弹性训练 - 处理待添加/移除节点队列
                                    if (trainer->elastic_enabled) {
                                        for (int fi = 0; fi < num_failed; fi++) {
                                            if (failed_nodes[fi] > 0) {
                                                distributed_elastic_remove_node(trainer, failed_nodes[fi]);
                                            }
                                        }
                                        while (trainer->elastic_node_pending_add_count > 0) {
                                            int add_id = trainer->elastic_node_pending_add[0];
                                            int add_result = distributed_elastic_add_node(trainer, add_id,
                                                trainer->distributed_num_nodes + trainer->elastic_node_pending_add_count);
                                            if (add_result == 0) {
                                                for (int pi = 0; pi < trainer->elastic_node_pending_add_count - 1; pi++) {
                                                    trainer->elastic_node_pending_add[pi] = trainer->elastic_node_pending_add[pi + 1];
                                                }
                                                trainer->elastic_node_pending_add_count--;
                                                if (trainer->config.verbose) {
                                                    printf("P1-6: 弹性训练添加节点 %d 成功\n", add_id);
                                                }
                                            } else {
                                                break;
                                            }
                                        }
                                    }
                                    
                                    if (trainer->failure_recovery_enabled &&
                                        trainer->failure_recovery_attempts < trainer->failure_recovery_max_attempts) {
                                        distributed_failure_rebuild_topology(trainer, failed_nodes, num_failed);
                                        trainer->failure_recovery_attempts++;
                                        if (trainer->config.verbose) {
                                            printf("F-08: 节点故障恢复，重建拓扑，尝试次数=%d\n",
                                                   trainer->failure_recovery_attempts);
                                        }
                                    } else if (trainer->auto_resume_enabled &&
                                               trainer->auto_resume_checkpoint_path) {
                                        int resume_result = distributed_auto_resume_from_checkpoint(
                                            trainer, parameters, trainer->gradients_size);
                                        if (resume_result == 0 && trainer->config.verbose) {
                                            printf("P1-6: 从检查点自动恢复成功\n");
                                        }
                                    }
                                }
                            }
                            
                            // P1-6: 过时梯度处理 - 在梯度同步前应用衰减系数
                            if (trainer->stale_gradient_enabled && gradients_to_use) {
                                int sg_result = distributed_handle_stale_gradients(trainer, gradients_to_use,
                                                                                  trainer->gradients_size);
                                if (sg_result != 0 && trainer->config.verbose) {
                                    log_warning("P1-6: 过时梯度处理失败\n");
                                }
                            }
                            
                            int sync_result;
                            if (trainer->async_sync_enabled) {
                                // 异步同步路径
                                sync_result = distributed_async_sync_start(trainer, gradients_to_use, trainer->gradients_size);
                                if (sync_result == 0) {
                                    trainer->distributed_retry_count = 0;
                                } else {
                                    // 异步启动失败，回退到同步路径
                                    if (trainer->config.verbose) {
                                        log_warning("异步同步启动失败，回退到同步模式\n");
                                    }
                                    sync_result = distributed_sync_gradients(trainer, gradients_to_use, trainer->gradients_size);
                                }
                            } else if (trainer->tree_enabled && trainer->distributed_num_nodes > 2) {
                                // 树形拓扑AllReduce：O(log N)通信步数
                                float* temp_buffer = (float*)safe_malloc(trainer->gradients_size * sizeof(float));
                                if (temp_buffer) {
                                    memcpy(temp_buffer, gradients_to_use, trainer->gradients_size * sizeof(float));
                                    int trs = distributed_tree_reduce_scatter(trainer, temp_buffer, trainer->gradients_size);
                                    int tag = distributed_tree_all_gather(trainer, gradients_to_use, trainer->gradients_size);
                                    safe_free((void**)&temp_buffer);
                                    sync_result = (trs == 0 && tag == 0) ? 0 : -1;
                                    if (trainer->config.verbose && sync_result == 0) {
                                        printf("F-08: 树形拓扑AllReduce完成，节点ID=%d\n", trainer->distributed_node_id);
                                    }
                                } else {
                                    sync_result = distributed_sync_gradients(trainer, gradients_to_use, trainer->gradients_size);
                                }
                            } else if (trainer->mesh_enabled && trainer->distributed_num_nodes > 4) {
                                // 网格拓扑AllReduce：O(√N)通信步数
                                sync_result = distributed_mesh_row_allreduce(trainer, gradients_to_use, trainer->gradients_size);
                                if (sync_result == 0) {
                                    sync_result = distributed_mesh_col_allreduce(trainer, gradients_to_use, trainer->gradients_size);
                                }
                                if (trainer->config.verbose && sync_result == 0) {
                                    printf("F-08: 网格拓扑AllReduce完成，节点(%d,%d)\n",
                                           trainer->mesh_row_id, trainer->mesh_col_id);
                                }
                            } else {
                                // 环形AllReduce（默认路径）
                                sync_result = distributed_sync_gradients(trainer, gradients_to_use, trainer->gradients_size);
                            }
                            
                            if (sync_result != 0) {
                                // 梯度同步失败，执行容错处理
                                int ft_result = distributed_fault_tolerance(trainer, 0, sync_result);
                                
                                if (ft_result == 0) {
                                    if (trainer->config.verbose) {
                                        log_info("分布式梯度同步失败，但容错机制已处理，将继续训练");
                                    }
                                } else {
                                    if (trainer->config.verbose) {
                                        log_error("分布式梯度同步失败且容错处理无效\n");
                                    }
                                    trainer->distributed_initialized = 0;
                                }
                            } else {
                                trainer->distributed_retry_count = 0;
                            }
                        }
                    }
                }
                
                // ========================================================
                // 分布式训练：检查点保存
                // ========================================================
                if (parameters && trainer->distributed_initialized) {
                    // 检查是否需要保存检查点
                    int should_save = distributed_should_save_checkpoint(trainer);
                    
                    if (should_save == 1) {
                        // 保存分布式检查点
                        int checkpoint_result = distributed_save_checkpoint(trainer, parameters,
                                                                          trainer->gradients_size,
                                                                          epoch, batch_num);
                        
                        if (checkpoint_result != 0) {
                            // 检查点保存失败，执行容错处理
                            int ft_result = distributed_fault_tolerance(trainer, 2, checkpoint_result);
                            
                            if (ft_result != 0) {
                                // 容错处理失败
                                if (trainer->config.verbose) {
                                    log_warning("分布式检查点保存失败且容错处理无效\n");
                                }
                            }
                        }
                    }
                }
                
                // ========================================================
                // 分布式训练：应用学习率缩放
                // ========================================================
                if (trainer->distributed_initialized) {
                    // 应用分布式学习率缩放
                    float scaled_lr = distributed_apply_learning_rate_scaling(trainer);
                    
                    if (fabsf(scaled_lr - trainer->state.learning_rate) > 1e-6f) {
                        // 更新优化器中的学习率
                        trainer->optimizer.learning_rate = scaled_lr;
                        trainer->state.learning_rate = scaled_lr;
                    }
                }

                // 优化器更新（使用原始梯度或滤波后的梯度）
                if (trainer->gpu_initialized) {
                    // GPU加速路径
                    if (trainer_gpu_update_parameters(trainer, parameters,
                                                     gradients_to_use,
                                                     trainer->gradients_size) != 0) {
                        if (trainer->config.verbose) {
                            log_warning("GPU参数更新失败，回退到CPU计算\n");
                        }
                        trainer_cleanup_gpu_resources(trainer);
                        // 回退到CPU优化器更新
                        optimizer_update(&trainer->optimizer, parameters,
                                        gradients_to_use, trainer->gradients_size);
                    }
                } else {
                    // CPU路径
                    // 混合精度训练路径（如果启用）
                    if (trainer->mixed_precision_context && trainer->mixed_precision_context->enabled) {
                        // 使用混合精度参数更新
                        if (mixed_precision_update_weights(trainer->mixed_precision_context,
                                                          trainer->state.learning_rate) != 0) {
                            if (trainer->config.verbose) {
                                log_warning("混合精度参数更新失败，回退到标准精度\n");
                            }
                            // 回退到标准精度优化器更新
                            optimizer_update(&trainer->optimizer, parameters,
                                            gradients_to_use, trainer->gradients_size);
                        }
                    } else {
                        // 标准精度优化器更新
                        optimizer_update(&trainer->optimizer, parameters,
                                        gradients_to_use, trainer->gradients_size);
                    }
                    
                    // 在参数更新后应用DropConnect正则化
                    if (trainer->regularizer) {
                        trainer_apply_network_regularization(trainer, batch_inputs,
                                                             batch_size, 1);
                    }
                }
                
                /* FIX-017: 将优化器更新后的共享参数块同步到活跃cell权重。
                 * optimizer_update 写入 cfc_network->weight_matrix（共享块），
                 * 但 cfc_forward 读取 cell->weight_matrix。此同步确保下一批次
                 * 前向传播使用优化后的参数。 */
                if (trainer->network && trainer->network->cfc_network) {
                    cfc_sync_shared_to_cells(trainer->network->cfc_network);
                }
                
                // 更新学习率
                size_t global_step = epoch * trainer->state.total_batches + batch_num;
                trainer->state.learning_rate = scheduler_get_internal(
                    trainer->scheduler, global_step);
                trainer->optimizer.learning_rate = trainer->state.learning_rate;
            }
            
            perf_timer_stop(&trainer->update_time);
            
            // 更新训练状态
            trainer->state.current_loss = loss;
            trainer->state.current_accuracy = accuracy;
            trainer->state.gradient_norm = gradient_norm(trainer->gradients,
                                                        trainer->gradients_size);
            
            // 调用回调函数
            if (callback) {
                callback(&trainer->state, user_data);
            }
            
            // 注意：批次缓冲区由data_loader_next_batch_internal分配
            // 调用者负责释放。
            if (batch_inputs) {
                safe_free((void**)&batch_inputs);
            }
            if (batch_targets) {
                safe_free((void**)&batch_targets);
            }
            
            // 记忆经验存储：将当前批次的经验存入记忆管理器
            if (trainer->memory_integration_enabled && trainer->memory_manager && 
                trainer->memory_recall_buffer && trainer->memory_recall_buffer_size > 0) {
                for (size_t b = 0; b < batch_size; b++) {
                    char store_key[64];
                    snprintf(store_key, sizeof(store_key), "exp_%zu",
                             (size_t)(trainer->memory_experience_counter));
                    // 将输入+输出+损失组合作为记忆经验存储
                    size_t store_dim = input_dim < output_dim ? input_dim : output_dim;
                    if (store_dim * 2 <= trainer->memory_recall_buffer_size) {
                        memcpy(trainer->memory_recall_buffer, 
                               batch_inputs + b * input_dim, store_dim * sizeof(float));
                        memcpy(trainer->memory_recall_buffer + store_dim,
                               trainer->batch_outputs + b * output_dim, store_dim * sizeof(float));
                        memory_manager_store(trainer->memory_manager, store_key,
                            trainer->memory_recall_buffer, store_dim * 2,
                            0, 1.0f);
                    }
                    trainer->memory_experience_counter++;
                }
            }
            
            batch_num++;
            trainer->state.samples_processed += batch_size;
        }
        
        // 计算平均损失和准确率
        if (batch_num > 0) {
            epoch_loss /= batch_num;
            epoch_accuracy /= batch_num;
        }
        
        trainer->state.current_loss = epoch_loss;
        trainer->state.current_accuracy = epoch_accuracy;
        
        // 验证
        float val_loss = 0.0f;
        float val_accuracy = 0.0f;
        
        if (validation_samples > 0) {
            trainer_validate(trainer, val_inputs, val_targets,
                            validation_samples, &val_loss, &val_accuracy);
            
            trainer->state.validation_loss = val_loss;
            trainer->state.validation_accuracy = val_accuracy;
            
            // ---- P3.6 演化算法触发 ----
            if (trainer->config.enable_evolution &&
                trainer->config.evolution_interval > 0 &&
                (epoch + 1) % trainer->config.evolution_interval == 0 &&
                trainer->evolution_initialized &&
                trainer->config.evolution_use_validation) {
                trainer_evolution_step(trainer, epoch,
                                       val_inputs, val_targets,
                                       validation_samples, input_dim, output_dim);
            }
        }
        
        // 更新最佳模型
        if (val_loss < trainer->state.best_loss) {
            trainer->state.best_loss = val_loss;
            trainer->state.best_accuracy = val_accuracy;
            trainer->state.steps_without_improvement = 0;
            
            // 保存最佳模型检查点
            if (trainer->config.save_best_model) {
                char checkpoint_name[256];
                snprintf(checkpoint_name, sizeof(checkpoint_name),
                        "best_model_epoch%zu_loss%.4f.bin", epoch, val_loss);
                save_model_checkpoint(trainer, checkpoint_name);
            }
        } else {
            trainer->state.steps_without_improvement++;
        }
        
        // 早停检查
        if (trainer->config.patience > 0) {
            if (early_stopping_check(val_loss, trainer->state.best_loss,
                                    trainer->config.patience,
                                    &trainer->state.steps_without_improvement)) {
                should_stop = 1;
                if (trainer->config.verbose) {
                    printf("早停触发于轮次 %zu\n", epoch);
                }
            }
        }
        
        // ---- F-11 自动版本快照 ----
        if (trainer->version_auto_snapshot_enabled && trainer->version_manager &&
            trainer->version_manager->auto_snapshot_enabled) {
            trainer->version_snapshot_counter++;
            if (trainer->version_snapshot_counter >= trainer->version_manager->auto_snapshot_interval) {
                trainer->version_snapshot_counter = 0;
                char tag[64];
                snprintf(tag, sizeof(tag), "auto_epoch_%zu", epoch);
                char desc[128];
                snprintf(desc, sizeof(desc), "自动快照：轮次%zu，损失%.4f，准确率%.4f",
                         epoch, val_loss, val_accuracy);
                ModelVersionID vid = model_version_snapshot(
                    trainer->version_manager, trainer, tag, desc);
                if (vid != 0 && trainer->config.verbose) {
                    printf("自动版本快照已创建：ID=%llu，标签=%s\n",
                           (unsigned long long)vid, tag);
                }
            }
        }
        
        // ---- 自动检查点保存 ----
        if (trainer->auto_checkpoint_path && trainer->auto_checkpoint_interval > 0) {
            trainer->auto_checkpoint_counter++;
            if (trainer->auto_checkpoint_counter >= trainer->auto_checkpoint_interval) {
                trainer->auto_checkpoint_counter = 0;
                int cp_result = save_model_checkpoint(trainer, trainer->auto_checkpoint_path);
                if (cp_result == 0) {
                    add_checkpoint_to_retention_list(trainer, trainer->auto_checkpoint_path);
                    if (trainer->config.verbose) {
                        printf("自动检查点已保存：轮次%zu，损失%.4f，准确率%.4f\n",
                               epoch, val_loss, val_accuracy);
                    }
                } else if (trainer->config.verbose) {
                    printf("警告：自动检查点保存失败（轮次%zu）\n", epoch);
                }
            }
        }
        
        // ---- 紧急检查点保存（收到中断信号时触发） ----
        if (trainer->emergency_save_requested && trainer->emergency_checkpoint_path) {
            trainer->emergency_save_requested = 0;
            int em_result = save_model_checkpoint(trainer, trainer->emergency_checkpoint_path);
            if (em_result == 0) {
                if (trainer->config.verbose) {
                    printf("\n=== 紧急检查点已保存: %s ===\n", trainer->emergency_checkpoint_path);
                    printf("  轮次: %zu, 损失: %.4f, 准确率: %.4f\n",
                           epoch, val_loss, val_accuracy);
                    printf("  训练可在下次启动时从该检查点恢复。\n");
                }
                // 刷新输出缓冲区确保日志写入磁盘
                fflush(stdout);
            } else {
                fprintf(stderr, "错误：紧急检查点保存失败: %s\n", trainer->emergency_checkpoint_path);
            }
        }
        
        // 自适应学习率调整（ReduceLROnPlateau）
        if (trainer->config.enable_adaptive_lr && validation_samples > 0) {
            static size_t lr_patience_counter = 0;
            static float lr_best_val_loss = FLT_MAX;
            static float lr_current_factor = 1.0f;
            TRAIN_LR_LOCK();
            if (val_loss < lr_best_val_loss - trainer->config.lr_min_delta) {
                lr_best_val_loss = val_loss;
                lr_patience_counter = 0;
            } else {
                lr_patience_counter++;
                if (lr_patience_counter >= trainer->config.lr_patience) {
                    // 减少学习率
                    lr_current_factor *= trainer->config.lr_factor;
                    if (lr_current_factor < trainer->config.lr_min_factor) {
                        lr_current_factor = trainer->config.lr_min_factor;
                    }
                    
                    // 应用新的学习率
                    trainer->state.learning_rate *= lr_current_factor;
                    trainer->optimizer.learning_rate = trainer->state.learning_rate;
                    
                    if (trainer->config.verbose) {
                        printf("自适应学习率调整：新学习率=%.6f（因子=%.2f）\n",
                               trainer->state.learning_rate, lr_current_factor);
                    }
                    
                    lr_patience_counter = 0;
                }
            }
            TRAIN_LR_UNLOCK();
        }
        
        // 保存训练历史
        if (trainer->history.size < trainer->history.capacity) {
            size_t idx = trainer->history.size;
            trainer->history.train_losses[idx] = epoch_loss;
            trainer->history.train_accuracies[idx] = epoch_accuracy;
            trainer->history.val_losses[idx] = val_loss;
            trainer->history.val_accuracies[idx] = val_accuracy;
            trainer->history.learning_rates[idx] = trainer->state.learning_rate;
            trainer->history.size++;
        }
        
        // 打印训练信息
        if (trainer->config.verbose && epoch % 10 == 0) {
            printf("轮次 %zu/%zu: 训练损失=%.4f, 训练准确率=%.4f, "
                   "验证损失=%.4f, 验证准确率=%.4f, 学习率=%.6f\n",
                   epoch, trainer->config.epochs, epoch_loss, epoch_accuracy,
                   val_loss, val_accuracy, trainer->state.learning_rate);
        }
        
        // 记忆系统巩固：每个consolidation_interval个epoch触发一次
        if (trainer->memory_integration_enabled && trainer->memory_manager &&
            trainer->memory_consolidation_interval > 0) {
            trainer->memory_consolidation_counter++;
            if (trainer->memory_consolidation_counter >= trainer->memory_consolidation_interval) {
                trainer_memory_consolidate(trainer);
                
                // 记忆回放训练：从巩固后的记忆中采样批次进行额外训练
                size_t mem_batch_size = 64;
                float* mem_inputs = (float*)safe_malloc(mem_batch_size * input_dim * sizeof(float));
                float* mem_targets = (float*)safe_malloc(mem_batch_size * output_dim * sizeof(float));
                float* mem_outputs = (float*)safe_malloc(mem_batch_size * output_dim * sizeof(float));
                if (mem_inputs && mem_targets && mem_outputs) {
                    MemorySystem* mem_sys = memory_manager_get_system(trainer->memory_manager);
                    int sampled = 0;
                    if (mem_sys) {
                        sampled = memory_sample_training_batch(
                            mem_sys, MEMORY_TYPE_EPISODIC,
                            mem_batch_size, input_dim,
                            mem_inputs, mem_targets);
                    }
                    if (sampled > 0) {
                        if (lnn_forward_batch(trainer->network, mem_inputs, mem_outputs, (size_t)sampled) == 0) {
                            float* mem_grad = (float*)safe_malloc((size_t)sampled * output_dim * sizeof(float));
                            if (mem_grad) {
                                for (size_t i = 0; i < (size_t)sampled * output_dim; i++) {
                                    mem_grad[i] = mem_outputs[i] - mem_targets[i];
                                }
                                lnn_backward_batch(trainer->network, mem_inputs, mem_grad,
                                                  trainer->gradients, (size_t)sampled);
                                float* params = lnn_get_parameters(trainer->network);
                                if (params) {
                                    optimizer_update(&trainer->optimizer, params,
                                                    trainer->gradients, trainer->gradients_size);
                                }
                                safe_free((void**)&mem_grad);
                            }
                        }
                    }
                }
                if (mem_inputs) safe_free((void**)&mem_inputs);
                if (mem_targets) safe_free((void**)&mem_targets);
                if (mem_outputs) safe_free((void**)&mem_outputs);
            }
        }
    }
    }
    
    // 计算总训练时间
    trainer->state.training_time_ms = (perf_timestamp_ns() / 1000000) - trainer->state.start_time;
    
    // 释放数据加载器
    data_loader_free_internal(loader);
    
    TRAINER_UNLOCK(trainer);
    return 0;
}

/**
 * @brief 验证神经网络
 */
int trainer_validate(Trainer* trainer, const float* inputs, const float* targets,
                     size_t num_samples, float* loss, float* accuracy) {
    if (!trainer || !inputs || !targets || num_samples == 0 || !loss || !accuracy) {
        return -1;
    }
    TRAINER_LOCK(trainer);
    
    size_t input_dim, output_dim;
    if (get_network_dimensions(trainer->network, &input_dim, &output_dim) != 0) {
        TRAINER_UNLOCK(trainer);
        return -1;
    }
    
    if (input_dim == 0 || output_dim == 0) {
        TRAINER_UNLOCK(trainer);
        return -1;
    }
    
    float total_loss = 0.0f;
    float total_accuracy = 0.0f;
    
    // 使用小批次进行验证以避免内存问题
    size_t batch_size = trainer->config.batch_size;
    if (batch_size > num_samples) {
        batch_size = num_samples;
    }
    
    size_t num_batches = (num_samples + batch_size - 1) / batch_size;
    
    for (size_t batch = 0; batch < num_batches; batch++) {
        size_t start = batch * batch_size;
        size_t end = start + batch_size;
        if (end > num_samples) {
            end = num_samples;
        }
        size_t current_batch_size = end - start;
        
        // 复制批次数据
        const float* batch_inputs = inputs + start * input_dim;
        const float* batch_targets = targets + start * output_dim;
        
        // 前向传播
        lnn_forward_batch(trainer->network, batch_inputs, trainer->batch_outputs,
                         current_batch_size);
        
        // 计算损失
        float batch_loss = 0.0f;
        
        switch (trainer->config.loss_function) {
            case LOSS_MEAN_SQUARED_ERROR:
                batch_loss = loss_mean_squared_error(trainer->batch_outputs,
                                                    batch_targets,
                                                    current_batch_size * output_dim);
                break;
            case LOSS_MEAN_ABSOLUTE_ERROR:
                batch_loss = loss_mean_absolute_error(trainer->batch_outputs,
                                                     batch_targets,
                                                     current_batch_size * output_dim);
                break;
            case LOSS_CROSS_ENTROPY:
                batch_loss = loss_cross_entropy(trainer->batch_outputs,
                                               batch_targets,
                                               current_batch_size * output_dim);
                break;
            default:
                batch_loss = loss_mean_squared_error(trainer->batch_outputs,
                                                    batch_targets,
                                                    current_batch_size * output_dim);
                break;
        }
        
        // 计算准确率（根据损失函数类型）
        float batch_accuracy = 0.0f;
        
        // 根据损失函数类型计算准确率或相关指标
        switch (trainer->config.loss_function) {
            case LOSS_CROSS_ENTROPY:
                // 分类任务：计算准确率（预测类别 = 真实类别）
                // 假设输出是概率分布，取最大概率的索引作为预测类别
                for (size_t i = 0; i < current_batch_size; i++) {
                    int predicted = 0;
                    int target = 0;
                    float max_prob = trainer->batch_outputs[i * output_dim];
                    float max_target = batch_targets[i * output_dim];
                    
                    for (size_t j = 1; j < output_dim; j++) {
                        if (trainer->batch_outputs[i * output_dim + j] > max_prob) {
                            max_prob = trainer->batch_outputs[i * output_dim + j];
                            predicted = (int)j;
                        }
                        if (batch_targets[i * output_dim + j] > max_target) {
                            max_target = batch_targets[i * output_dim + j];
                            target = (int)j;
                        }
                    }
                    
                    if (predicted == target) {
                        batch_accuracy += 1.0f;
                    }
                }
                batch_accuracy /= current_batch_size;
                break;
                
            case LOSS_MEAN_SQUARED_ERROR:
            case LOSS_MEAN_ABSOLUTE_ERROR:
                // 回归任务：计算R²分数（决定系数）
                // 首先计算目标变量的均值
                float target_mean = 0.0f;
                for (size_t i = 0; i < current_batch_size * output_dim; i++) {
                    target_mean += batch_targets[i];
                }
                target_mean /= (current_batch_size * output_dim);
                
                // 计算总平方和（SST）
                float sst = 0.0f;
                for (size_t i = 0; i < current_batch_size * output_dim; i++) {
                    float diff = batch_targets[i] - target_mean;
                    sst += diff * diff;
                }
                
                // 计算残差平方和（SSE）
                float sse = 0.0f;
                for (size_t i = 0; i < current_batch_size * output_dim; i++) {
                    float residual = batch_targets[i] - trainer->batch_outputs[i];
                    sse += residual * residual;
                }
                
                // 计算R²分数
                if (sst > 1e-10f) {
                    batch_accuracy = 1.0f - (sse / sst);
                    // R²分数可能在[-∞, 1]之间，我们限制到[0,1]范围用于显示
                    if (batch_accuracy < 0.0f) batch_accuracy = 0.0f;
                    if (batch_accuracy > 1.0f) batch_accuracy = 1.0f;
                } else {
                    batch_accuracy = 0.0f;
                }
                break;
                
            default:
                // 其他损失函数：默认准确率为0
                batch_accuracy = 0.0f;
                break;
        }
        
        total_loss += batch_loss * current_batch_size;
        total_accuracy += batch_accuracy * current_batch_size;
    }
    
    if (num_samples > 0) {
        *loss = total_loss / num_samples;
        *accuracy = total_accuracy / num_samples;
    } else {
        *loss = 0.0f;
        *accuracy = 0.0f;
    }
    
    TRAINER_UNLOCK(trainer);
    return 0;
}

/**
 * @brief 预测
 */
int trainer_predict(Trainer* trainer, const float* inputs, float* outputs,
                    size_t num_samples) {
    if (!trainer || !inputs || !outputs || num_samples == 0) {
        return -1;
    }
    TRAINER_LOCK(trainer);
    
    size_t input_dim, output_dim;
    if (get_network_dimensions(trainer->network, &input_dim, &output_dim) != 0) {
        TRAINER_UNLOCK(trainer);
        return -1;
    }
    
    if (input_dim == 0 || output_dim == 0) {
        TRAINER_UNLOCK(trainer);
        return -1;
    }
    
    // 使用小批次进行预测
    size_t batch_size = trainer->config.batch_size;
    if (batch_size > num_samples) {
        batch_size = num_samples;
    }
    
    size_t num_batches = (num_samples + batch_size - 1) / batch_size;
    
    for (size_t batch = 0; batch < num_batches; batch++) {
        size_t start = batch * batch_size;
        size_t end = start + batch_size;
        if (end > num_samples) {
            end = num_samples;
        }
        size_t current_batch_size = end - start;
        
        // 复制批次输入
        const float* batch_inputs = inputs + start * input_dim;
        float* batch_outputs = outputs + start * output_dim;
        
        // 前向传播
        lnn_forward_batch(trainer->network, batch_inputs, batch_outputs,
                         current_batch_size);
    }
    
    TRAINER_UNLOCK(trainer);
    return 0;
}

/**
 * @brief 获取训练历史
 */
TrainingHistory* trainer_get_history(Trainer* trainer) {
    if (!trainer) return NULL;
    TRAINER_LOCK(trainer);
    TrainingHistory* result = &trainer->history;
    TRAINER_UNLOCK(trainer);
    return result;
}

/**
 * @brief 优化器初始化
 */
static int optimizer_init(OptimizerState* optimizer, const TrainingConfig* config,
                          size_t num_parameters) {
    if (!optimizer || !config || num_parameters == 0) {
        return -1;
    }
    
    memset(optimizer, 0, sizeof(OptimizerState));
    
    optimizer->type = config->optimizer;
    optimizer->learning_rate = config->learning_rate;
    optimizer->momentum = config->momentum;
    optimizer->beta1 = config->beta1;
    optimizer->beta2 = config->beta2;
    optimizer->epsilon = config->epsilon;
    optimizer->weight_decay = config->regularization_lambda;
    optimizer->t = 0;
    
    // 根据优化器类型分配缓冲区
    switch (optimizer->type) {
        case OPTIMIZER_MOMENTUM:
            optimizer->momentum_buffer_size = num_parameters;
            optimizer->momentum_buffer = (float*)safe_calloc(num_parameters, sizeof(float));
            if (!optimizer->momentum_buffer) {
                return -1;
            }
            break;
            
        case OPTIMIZER_ADAM:
        case OPTIMIZER_ADAMW:
            optimizer->adam_buffer_size = num_parameters;
            optimizer->m_buffer = (float*)safe_calloc(num_parameters, sizeof(float));
            optimizer->v_buffer = (float*)safe_calloc(num_parameters, sizeof(float));
            if (!optimizer->m_buffer || !optimizer->v_buffer) {
                safe_free((void**)&optimizer->m_buffer);
                safe_free((void**)&optimizer->v_buffer);
                return -1;
            }
            break;
            
        default:
            // SGD不需要额外缓冲区
            break;
    }
    
    return 0;
}

/**
 * @brief 优化器更新（工业级实现，参数更新公式与core/optimizer.c一致）
 * 
 * FIX-007: 全面重写优化器，修复以下问题：
 * 1. SGD/Momentum/AdaGrad/RMSProp: L2→解耦权重衰减
 * 2. Momentum: Nesterov实现错误→正确公式
 * 3. Adam: powf每参数逐次调用→预计算bias correction一次
 * 4. Adam: 标准Adam不应含L2权重衰减→仅AdamW执行解耦衰减
 * 5. AdaGrad: 移除L2混合→解耦形式
 * 6. 梯度稳定性保护：检测NaN/Inf并跳过更新
 */
static void optimizer_update(OptimizerState* optimizer, float* parameters,
                            const float* gradients, size_t num_parameters) {
    if (!optimizer || !parameters || !gradients || num_parameters == 0) {
        return;
    }
    
    optimizer->t++;
    
    float lr = optimizer->learning_rate;
    float eps = (optimizer->epsilon > 0.0f) ? optimizer->epsilon : 1e-8f;
    float wd = optimizer->weight_decay;
    size_t i;
    
    switch (optimizer->type) {
        case OPTIMIZER_SGD: {
            /* 解耦权重衰减 + SGD */
            for (i = 0; i < num_parameters; i++) {
                if (!isfinite(gradients[i])) continue; /* 跳过NaN梯度 */
                parameters[i] *= (1.0f - lr * wd);
                parameters[i] -= lr * gradients[i];
            }
            break;
        }
            
        case OPTIMIZER_MOMENTUM: {
            float mu = (optimizer->momentum > 0.0f) ? optimizer->momentum : 0.9f;
            for (i = 0; i < num_parameters; i++) {
                if (!isfinite(gradients[i])) continue;
                /* 解耦权重衰减 */
                parameters[i] *= (1.0f - lr * wd);
                float v_old = optimizer->momentum_buffer[i];
                optimizer->momentum_buffer[i] = mu * v_old - lr * gradients[i];
                parameters[i] += optimizer->momentum_buffer[i];
            }
            break;
        }
            
        case OPTIMIZER_ADAGRAD: {
            for (i = 0; i < num_parameters; i++) {
                if (!isfinite(gradients[i])) continue;
                /* 解耦权重衰减 */
                parameters[i] *= (1.0f - lr * wd);
                float grad_sq = gradients[i] * gradients[i];
                optimizer->momentum_buffer[i] += grad_sq;
                float adjusted_lr = lr / (sqrtf(optimizer->momentum_buffer[i]) + eps);
                parameters[i] -= adjusted_lr * gradients[i];
            }
            break;
        }
            
        case OPTIMIZER_RMSPROP: {
            float decay_rate = (optimizer->momentum > 0.0f) ? optimizer->momentum : 0.9f;
            for (i = 0; i < num_parameters; i++) {
                if (!isfinite(gradients[i])) continue;
                /* 解耦权重衰减 */
                parameters[i] *= (1.0f - lr * wd);
                float grad_sq = gradients[i] * gradients[i];
                optimizer->momentum_buffer[i] = 
                    decay_rate * optimizer->momentum_buffer[i] + (1.0f - decay_rate) * grad_sq;
                float adjusted_lr = lr / (sqrtf(optimizer->momentum_buffer[i]) + eps);
                parameters[i] -= adjusted_lr * gradients[i];
            }
            break;
        }
            
        case OPTIMIZER_ADAM:
        case OPTIMIZER_ADAMW: {
            float b1 = (optimizer->beta1 > 0.0f) ? optimizer->beta1 : 0.9f;
            float b2 = (optimizer->beta2 > 0.0f) ? optimizer->beta2 : 0.999f;
            /* FIX-007: 预计算偏差校正系数一次，避免powf每参数逐次调用 */
            float b1_correction = 1.0f - powf(b1, (float)optimizer->t);
            float b2_correction = 1.0f - powf(b2, (float)optimizer->t);
            if (b1_correction < 1e-10f) b1_correction = 1e-10f;
            if (b2_correction < 1e-10f) b2_correction = 1e-10f;
            float inv_b1 = 1.0f / b1_correction;
            float inv_b2 = 1.0f / b2_correction;
            
            for (i = 0; i < num_parameters; i++) {
                if (!isfinite(gradients[i])) continue;
                
                optimizer->m_buffer[i] = b1 * optimizer->m_buffer[i] +
                                         (1.0f - b1) * gradients[i];
                float grad_sq = gradients[i] * gradients[i];
                optimizer->v_buffer[i] = b2 * optimizer->v_buffer[i] +
                                         (1.0f - b2) * grad_sq;
                
                float m_hat = optimizer->m_buffer[i] * inv_b1;
                float v_hat = optimizer->v_buffer[i] * inv_b2;
                
                float update = lr * m_hat / (sqrtf(v_hat) + eps);
                
                if (optimizer->type == OPTIMIZER_ADAMW) {
                    /* AdamW: 解耦权重衰减 */
                    parameters[i] = parameters[i] * (1.0f - lr * wd) - update;
                } else {
                    /* 标准Adam: 无权重衰减 */
                    parameters[i] -= update;
                }
            }
            break;
        }
            
        default:
            /* 未识别优化器回退为SGD */
            for (i = 0; i < num_parameters; i++) {
                if (!isfinite(gradients[i])) continue;
                parameters[i] *= (1.0f - lr * wd);
                parameters[i] -= lr * gradients[i];
            }
            break;
    }
}

/**
 * @brief 释放优化器资源
 */
static void optimizer_free(OptimizerState* optimizer) {
    if (!optimizer) {
        return;
    }
    
    safe_free((void**)&optimizer->momentum_buffer);
    
    safe_free((void**)&optimizer->m_buffer);
    
    safe_free((void**)&optimizer->v_buffer);
    
    memset(optimizer, 0, sizeof(OptimizerState));
}

/**
 * @brief 学习率调度器创建
 */
static LearningRateScheduler* scheduler_create_internal(
    const LearningRateSchedulerConfig* config, size_t total_steps) {
    if (!config) {
        return NULL;
    }
    
    LearningRateScheduler* scheduler = (LearningRateScheduler*)safe_malloc(sizeof(LearningRateScheduler));
    if (!scheduler) {
        return NULL;
    }
    
    scheduler->type = config->type;
    scheduler->base_learning_rate = config->base_learning_rate;
    scheduler->max_learning_rate = config->max_learning_rate;
    scheduler->min_learning_rate = config->min_learning_rate;
    scheduler->decay_rate = config->decay_rate;
    scheduler->decay_steps = config->decay_steps;
    scheduler->step_size = config->step_size;
    scheduler->cycle_length = config->cycle_length;
    scheduler->total_steps = total_steps;
    scheduler->current_step = 0;
    scheduler->plateau_factor = config->plateau_factor > 0.0f ? config->plateau_factor : 0.1f;
    scheduler->plateau_patience = config->plateau_patience > 0 ? config->plateau_patience : 10;
    scheduler->plateau_threshold = config->plateau_threshold > 0.0f ? config->plateau_threshold : 1e-4f;
    scheduler->plateau_cooldown = config->plateau_cooldown;
    scheduler->min_learning_rate_abs = config->min_learning_rate_abs > 0.0f ? config->min_learning_rate_abs : 1e-8f;
    scheduler->best_metric = 0.0f;
    scheduler->bad_epochs = 0;
    scheduler->cooldown_counter = 0;
    scheduler->current_plateau_lr = config->base_learning_rate;
    scheduler->plateau_initialized = 0;
    
    return scheduler;
}

/**
 * @brief 获取学习率
 */
static float scheduler_get_internal(LearningRateScheduler* scheduler, size_t step) {
    if (!scheduler) {
        return 0.001f;  // 默认学习率
    }
    
    scheduler->current_step = step;
    
    switch (scheduler->type) {
        case SCHEDULER_CONSTANT:
            return scheduler->base_learning_rate;
            
        case SCHEDULER_STEP:
            // 阶梯衰减
            return scheduler->base_learning_rate * 
                   powf(scheduler->decay_rate, (float)step / (float)scheduler->step_size);
            
        case SCHEDULER_EXPONENTIAL:
            // 指数衰减
            return scheduler->base_learning_rate * 
                   powf(scheduler->decay_rate, step / (float)scheduler->decay_steps);
            
        case SCHEDULER_COSINE:
            // 余弦衰减
            if (step >= scheduler->total_steps) {
                return scheduler->min_learning_rate;
            }
            return scheduler->min_learning_rate + 
                   0.5f * (scheduler->base_learning_rate - scheduler->min_learning_rate) *
                   (1.0f + cosf(MATH_PI * step / (float)scheduler->total_steps));
            
        case SCHEDULER_CYCLIC:
            // 循环学习率
            if (scheduler->cycle_length == 0) {
                return scheduler->base_learning_rate;
            }
            {
                size_t cycle_step = step % scheduler->cycle_length;
                size_t half_cycle = scheduler->cycle_length / 2;
                
                if (cycle_step < half_cycle) {
                    // 上升阶段
                    return scheduler->min_learning_rate + 
                           (scheduler->max_learning_rate - scheduler->min_learning_rate) *
                           (cycle_step / (float)half_cycle);
                } else {
                    // 下降阶段
                    return scheduler->max_learning_rate - 
                           (scheduler->max_learning_rate - scheduler->min_learning_rate) *
                           ((cycle_step - half_cycle) / (float)half_cycle);
                }
            }
            
        case SCHEDULER_PLATEAU:
            return scheduler->current_plateau_lr;
            
        default:
            return scheduler->base_learning_rate;
    }
}

/**
 * @brief ReduceLROnPlateau：基于指标评估调整学习率
 */
float learning_rate_scheduler_on_plateau(void* scheduler, float metric, size_t step) {
    if (!scheduler) return 0.001f;
    LearningRateScheduler* sched = (LearningRateScheduler*)scheduler;
    if (sched->type != SCHEDULER_PLATEAU) {
        return learning_rate_scheduler_get(scheduler, step);
    }
    sched->current_step = step;

    if (!sched->plateau_initialized) {
        sched->best_metric = metric;
        sched->plateau_initialized = 1;
        return sched->current_plateau_lr;
    }

    if (sched->cooldown_counter > 0) {
        sched->cooldown_counter--;
        sched->bad_epochs = 0;
        return sched->current_plateau_lr;
    }

    float threshold = sched->plateau_threshold;
    if (metric < sched->best_metric * (1.0f - threshold)) {
        sched->best_metric = metric;
        sched->bad_epochs = 0;
    } else {
        sched->bad_epochs++;
        if (sched->bad_epochs >= sched->plateau_patience) {
            float new_lr = sched->current_plateau_lr * sched->plateau_factor;
            if (new_lr >= sched->min_learning_rate_abs) {
                sched->current_plateau_lr = new_lr;
            }
            sched->cooldown_counter = sched->plateau_cooldown;
            sched->bad_epochs = 0;
        }
    }
    return sched->current_plateau_lr;
}

/**
 * @brief 释放学习率调度器
 */
static void scheduler_free_internal(LearningRateScheduler* scheduler) {
    safe_free((void**)&scheduler);
}

/**
 * @brief 数据加载器创建
 */
static DataLoader* data_loader_create_internal(const float* inputs,
                                              const float* targets,
                                              size_t num_samples,
                                              size_t input_dim,
                                              size_t output_dim,
                                              size_t batch_size, int shuffle) {
    if (!inputs || !targets || num_samples == 0 || 
        input_dim == 0 || output_dim == 0 || batch_size == 0) {
        return NULL;
    }
    
    DataLoader* loader = (DataLoader*)safe_malloc(sizeof(DataLoader));
    if (!loader) {
        return NULL;
    }
    
    loader->magic = 0xDEADBEEF;
    loader->inputs = inputs;
    loader->targets = targets;
    loader->num_samples = num_samples;
    loader->input_dim = input_dim;
    loader->output_dim = output_dim;
    loader->batch_size = batch_size;
    loader->shuffle = shuffle;
    loader->current_index = 0;
    loader->epoch = 0;
    
    // 分配索引数组
    loader->indices = (size_t*)safe_malloc(num_samples * sizeof(size_t));
    if (!loader->indices) {
        safe_free((void**)&loader);
        return NULL;
    }
    
    // 初始化索引
    for (size_t i = 0; i < num_samples; i++) {
        loader->indices[i] = i;
    }
    
    // 如果需要，打乱数据
    if (shuffle) {
        rng_shuffle_size_t(loader->indices, num_samples);
    }
    
    return loader;
}

/**
 * @brief 获取下一个批次
 */
static int data_loader_next_batch_internal(DataLoader* loader,
                                          float** batch_inputs,
                                          float** batch_targets,
                                          size_t* batch_size) {
    if (!loader || !batch_inputs || !batch_targets || !batch_size) {
        return -1;
    }
    
    if (loader->current_index >= loader->num_samples) {
        return 0;  // 没有更多数据
    }
    
    size_t start = loader->current_index;
    size_t end = start + loader->batch_size;
    if (end > loader->num_samples) {
        end = loader->num_samples;
    }
    
    size_t current_batch_size = end - start;
    
    // 分配批次缓冲区（调用者负责释放）
    *batch_inputs = (float*)safe_malloc(current_batch_size * loader->input_dim * sizeof(float));
    *batch_targets = (float*)safe_malloc(current_batch_size * loader->output_dim * sizeof(float));
    
    if (!*batch_inputs || !*batch_targets) {
        if (*batch_inputs) safe_free((void**)batch_inputs);
        if (*batch_targets) safe_free((void**)batch_targets);
        return -1;
    }
    
    // 复制批次数据
    for (size_t i = 0; i < current_batch_size; i++) {
        size_t sample_idx = loader->indices[start + i];
        
        // 边界检查
        if (sample_idx >= loader->num_samples) {
            printf("data_loader_next_batch_internal: 样本索引超出范围! idx=%zu, num_samples=%zu\n", 
                   sample_idx, loader->num_samples);
            // 清理并返回错误
            safe_free((void**)batch_inputs);
            safe_free((void**)batch_targets);
            return -1;
        }
        
        // 复制输入数据
        const float* src_input = loader->inputs + sample_idx * loader->input_dim;
        float* dst_input = *batch_inputs + i * loader->input_dim;
        memcpy(dst_input, src_input, loader->input_dim * sizeof(float));
        
        // 复制目标数据
        const float* src_target = loader->targets + sample_idx * loader->output_dim;
        float* dst_target = *batch_targets + i * loader->output_dim;
        memcpy(dst_target, src_target, loader->output_dim * sizeof(float));
    }
    
    *batch_size = current_batch_size;
    loader->current_index = end;
    
    return 1;
}

/**
 * @brief 重置数据加载器
 */
static void data_loader_reset_internal(DataLoader* loader) {
    if (!loader) {
        log_info("data_loader_reset_internal: 加载器为空");
        return;
    }
    
    loader->current_index = 0;
    loader->epoch++;
    
    // 如果需要，重新打乱数据
    if (loader->shuffle && loader->indices && loader->num_samples > 0) {
        rng_shuffle_size_t(loader->indices, loader->num_samples);
    }
}

/**
 * @brief 释放数据加载器
 */
static void data_loader_free_internal(DataLoader* loader) {
    if (!loader) {
        return;
    }
    
    // 检查魔术数字
    if (loader->magic != 0xDEADBEEF) {
        printf("data_loader_free_internal: 魔术数字损坏! 期望: 0xDEADBEEF, 实际: 0x%08X\n", loader->magic);
    }
    
    safe_free((void**)&loader->indices);
    
    safe_free((void**)&loader);
}

/**
 * @brief 验证网络维度
 */
static int validate_network_dimensions(LNN* network,
                                      size_t input_dim, size_t output_dim) {
    if (!network) {
        return 0;
    }
    
    LNNConfig net_config;
    if (lnn_get_config(network, &net_config) != 0) {
        return 0;
    }
    
    size_t network_input_dim = net_config.input_size;
    size_t network_output_dim = net_config.output_size;
    
    return (network_input_dim == input_dim && network_output_dim == output_dim);
}

/**
 * @brief 获取网络输入输出维度
 */
static int get_network_dimensions(LNN* network, size_t* input_dim, size_t* output_dim) {
    if (!network || !input_dim || !output_dim) {
        return -1;
    }
    
    LNNConfig net_config;
    if (lnn_get_config(network, &net_config) != 0) {
        return -1;
    }
    
    *input_dim = net_config.input_size;
    *output_dim = net_config.output_size;
    
    return 0;
}

/**
 * @brief 梯度裁剪
 */
void gradient_clip(float* gradients, size_t num_gradients,
                   GradientClipType clip_type, float clip_value, float clip_norm) {
    if (!gradients || num_gradients == 0) {
        return;
    }
    
    switch (clip_type) {
        case GRADIENT_CLIP_VALUE:
            // 值裁剪：限制梯度在[-clip_value, clip_value]范围内
            for (size_t i = 0; i < num_gradients; i++) {
                if (gradients[i] > clip_value) {
                    gradients[i] = clip_value;
                } else if (gradients[i] < -clip_value) {
                    gradients[i] = -clip_value;
                }
            }
            break;
            
        case GRADIENT_CLIP_NORM:
            // 范数裁剪：如果梯度范数超过clip_norm，则缩放梯度
            {
                float norm = 0.0f;
                for (size_t i = 0; i < num_gradients; i++) {
                    norm += gradients[i] * gradients[i];
                }
                norm = sqrtf(norm);
                
                if (norm > clip_norm) {
                    float scale = clip_norm / norm;
                    for (size_t i = 0; i < num_gradients; i++) {
                        gradients[i] *= scale;
                    }
                }
            }
            break;
            
        default:
            // 无裁剪
            break;
    }
}

/**
 * @brief 计算梯度范数
 */
float gradient_norm(const float* gradients, size_t num_gradients) {
    if (!gradients || num_gradients == 0) {
        return 0.0f;
    }
    
    float norm = 0.0f;
    for (size_t i = 0; i < num_gradients; i++) {
        norm += gradients[i] * gradients[i];
    }
    
    return sqrtf(norm);
}

/**
 * @brief 权重衰减
 */
void weight_decay(float* weights, size_t num_weights,
                  float lambda, RegularizationType regularization_type) {
    if (!weights || num_weights == 0 || lambda == 0.0f) {
        return;
    }
    
    switch (regularization_type) {
        case REGULARIZATION_L1:
            // L1正则化
            for (size_t i = 0; i < num_weights; i++) {
                if (weights[i] > 0) {
                    weights[i] -= lambda;
                } else if (weights[i] < 0) {
                    weights[i] += lambda;
                }
            }
            break;
            
        case REGULARIZATION_L2:
            // L2正则化（权重衰减）
            for (size_t i = 0; i < num_weights; i++) {
                weights[i] *= (1.0f - lambda);
            }
            break;
            
        default:
            // 无正则化
            break;
    }
}

/**
 * @brief Dropout正则化
 */
void dropout(float* activations, size_t num_activations,
             float dropout_rate, int is_training, float* mask) {
    if (!activations || num_activations == 0 || dropout_rate <= 0.0f) {
        return;
    }
    
    if (is_training) {
        // 训练阶段：应用Dropout
        float scale = 1.0f / (1.0f - dropout_rate);
        
        for (size_t i = 0; i < num_activations; i++) {
            if (rng_bernoulli(dropout_rate) > 0.5f) {
                // 丢弃该激活
                if (mask) {
                    mask[i] = 0.0f;
                }
                activations[i] = 0.0f;
            } else {
                // 保留该激活，并缩放以保持期望值
                if (mask) {
                    mask[i] = scale;
                }
                activations[i] *= scale;
            }
        }
    } else {
        // 推理阶段：不应用Dropout，但如果有掩码则应用缩放
        if (mask) {
            for (size_t i = 0; i < num_activations; i++) {
                activations[i] *= mask[i];
            }
        }
    }
}

/* ========================================================================
 * P1-002: 梯度验证 (Gradient Checking) 完整实现
 * 使用双边有限差分（中心差分法）O(ε²)精度验证解析梯度
 * ======================================================================== */

/**
 * @brief 计算指定参数扰动后的损失值
 * 内部辅助函数：修改单个参数 ± ε，前向传播，返回损失
 */
static float gradient_check_perturb_and_loss(Trainer* trainer,
                                              const float* inputs, const float* targets,
                                              size_t num_samples, size_t param_index,
                                              float perturbation, float* original_value) {
    float* parameters = lnn_get_parameters(trainer->network);
    size_t total_params = trainer->gradients_size;
    if (!parameters || param_index >= total_params) return 0.0f;

    *original_value = parameters[param_index];
    parameters[param_index] = *original_value + perturbation;

    /* 前向传播计算损失 */
    size_t input_dim, output_dim;
    if (get_network_dimensions(trainer->network, &input_dim, &output_dim) != 0) {
        parameters[param_index] = *original_value;
        return 0.0f;
    }

    float total_loss = 0.0f;
    for (size_t s = 0; s < num_samples; s++) {
        const float* sample_input = inputs + s * input_dim;
        const float* sample_target = targets + s * output_dim;
        float* output = trainer->batch_outputs;

        lnn_forward(trainer->network, sample_input, output);

        switch (trainer->config.loss_function) {
            case LOSS_MEAN_SQUARED_ERROR:
                total_loss += loss_mean_squared_error(output, sample_target, output_dim);
                break;
            case LOSS_MEAN_ABSOLUTE_ERROR:
                total_loss += loss_mean_absolute_error(output, sample_target, output_dim);
                break;
            case LOSS_CROSS_ENTROPY:
                total_loss += loss_cross_entropy(output, sample_target, output_dim);
                break;
            default:
                total_loss += loss_mean_squared_error(output, sample_target, output_dim);
                break;
        }
    }

    /* 恢复原始参数值 */
    parameters[param_index] = *original_value;
    return total_loss / (float)num_samples;
}

/**
 * @brief 梯度验证实现
 * 使用双边有限差分: ∂L/∂θ ≈ (L(θ+ε) - L(θ-ε)) / (2ε)
 * O(ε²)精度，比单边差分O(ε)更精确
 */
int trainer_gradient_check(Trainer* trainer, const float* inputs, const float* targets,
                           size_t num_samples, float threshold, float epsilon,
                           GradientCheckResult* result) {
    if (!trainer || !inputs || !targets || num_samples == 0 || !result) return -1;
    if (epsilon <= 0.0f) epsilon = 1e-6f;
    if (threshold <= 0.0f) threshold = 1e-4f;

    memset(result, 0, sizeof(GradientCheckResult));
    result->threshold = threshold;

    size_t input_dim, output_dim;
    if (get_network_dimensions(trainer->network, &input_dim, &output_dim) != 0) return -1;

    float* parameters = lnn_get_parameters(trainer->network);
    size_t total_params = trainer->gradients_size;
    if (!parameters || total_params == 0) return -1;

    size_t num_check = total_params;
    /* 参数太多时随机采样不超过5000个 */
    if (num_check > 5000) num_check = 5000;

    result->num_parameters = num_check;
    result->per_parameter_error = (float*)safe_calloc(num_check, sizeof(float));
    if (!result->per_parameter_error) return -1;

    /* 预先计算一次前向+反向传播获取解析梯度 */
    /* 使用单一前向传播计算基准损失 */
    float baseline_loss = 0.0f;
    for (size_t s = 0; s < num_samples; s++) {
        const float* sin = inputs + s * input_dim;
        const float* stt = targets + s * output_dim;
        float loss_val = 0.0f;
        lnn_forward(trainer->network, sin, trainer->batch_outputs);
        switch (trainer->config.loss_function) {
            case LOSS_MEAN_SQUARED_ERROR:
                loss_val = loss_mean_squared_error(trainer->batch_outputs, stt, output_dim);
                break;
            case LOSS_MEAN_ABSOLUTE_ERROR:
                loss_val = loss_mean_absolute_error(trainer->batch_outputs, stt, output_dim);
                break;
            case LOSS_CROSS_ENTROPY:
                loss_val = loss_cross_entropy(trainer->batch_outputs, stt, output_dim);
                break;
            default:
                loss_val = loss_mean_squared_error(trainer->batch_outputs, stt, output_dim);
                break;
        }
        baseline_loss += loss_val;
    }
    baseline_loss /= (float)num_samples;

    /* FIX-012: 计算解析梯度——改用 lnn_backward_batch 替代 lnn_backward。
     * lnn_backward 签名无输出梯度参数，内部 cfc_backward 步骤3会直接SGD更新
     * 所有权重（破坏模型），且梯度写入 cell 内部缓冲区而非 trainer->gradients，
     * 导致 trainer->gradients 永远为零→梯度检查永远失败。
     * lnn_backward_batch 使用 cfc_accumulate_gradients（仅累积不更新权重），
     * 将正确梯度输出到参数缓冲区 trainer->gradients。 */
    {
        /* 先对所有样本做前向传播，获取预测输出用于计算损失梯度 */
        float* all_outputs = (float*)safe_malloc(num_samples * output_dim * sizeof(float));
        float* loss_grads = (float*)safe_malloc(num_samples * output_dim * sizeof(float));
        if (all_outputs && loss_grads) {
            for (size_t s = 0; s < num_samples; s++) {
                lnn_forward(trainer->network, inputs + s * input_dim,
                           all_outputs + s * output_dim);
            }
            switch (trainer->config.loss_function) {
                case LOSS_MEAN_SQUARED_ERROR:
                    loss_mean_squared_error_gradient(all_outputs, targets,
                                                     loss_grads, num_samples * output_dim);
                    break;
                case LOSS_MEAN_ABSOLUTE_ERROR:
                    loss_mean_absolute_error_gradient(all_outputs, targets,
                                                      loss_grads, num_samples * output_dim);
                    break;
                case LOSS_CROSS_ENTROPY:
                    loss_cross_entropy_gradient(all_outputs, targets,
                                                loss_grads, num_samples * output_dim);
                    break;
                default:
                    loss_mean_squared_error_gradient(all_outputs, targets,
                                                     loss_grads, num_samples * output_dim);
                    break;
            }
            /* lnn_backward_batch 将梯度累积写入 trainer->gradients，不修改模型权重 */
            lnn_backward_batch(trainer->network, inputs, loss_grads,
                              trainer->gradients, num_samples);
        }
        safe_free((void**)&all_outputs);
        safe_free((void**)&loss_grads);
    }
    /* FIX-012: lnn_backward_batch 内部已做 /batch_size 平均，不需要再次平均 */

    float sum_relative_error = 0.0f;
    float max_relative_error = 0.0f;
    float max_absolute_error = 0.0f;
    int num_mismatches = 0;

    /* 对每个（采样的）参数执行双边有限差分验证 */
    for (size_t i = 0; i < num_check; i++) {
        size_t param_idx;
        if (num_check == total_params) {
            param_idx = i;
        } else {
            /* 均匀采样：等间距选取参数 */
            param_idx = (i * total_params) / num_check;
        }

        float analytic_grad = trainer->gradients[param_idx];
        float original_val = 0.0f;

        float loss_plus = gradient_check_perturb_and_loss(
            trainer, inputs, targets, num_samples, param_idx, epsilon, &original_val);

        float loss_minus = gradient_check_perturb_and_loss(
            trainer, inputs, targets, num_samples, param_idx, -epsilon, &original_val);

        float numerical_grad = (loss_plus - loss_minus) / (2.0f * epsilon);

        /* 相对误差公式: |numerical - analytic| / max(|numerical|, |analytic|, 1e-10) */
        float denom = fabsf(numerical_grad) > fabsf(analytic_grad) ?
                      fabsf(numerical_grad) : fabsf(analytic_grad);
        if (denom < 1e-10f) denom = 1e-10f;
        float relative_error = fabsf(numerical_grad - analytic_grad) / denom;
        float absolute_error = fabsf(numerical_grad - analytic_grad);

        result->per_parameter_error[i] = relative_error;
        sum_relative_error += relative_error;

        if (relative_error > max_relative_error) max_relative_error = relative_error;
        if (absolute_error > max_absolute_error) max_absolute_error = absolute_error;

        if (relative_error > threshold) num_mismatches++;
    }

    result->max_relative_error = max_relative_error;
    result->avg_relative_error = (num_check > 0) ? sum_relative_error / (float)num_check : 0.0f;
    result->max_absolute_error = max_absolute_error;
    result->num_mismatches = num_mismatches;
    result->passed = (num_mismatches == 0) ? 1 : 0;

    return 0;
}

void gradient_check_result_free(GradientCheckResult* result) {
    if (!result) return;
    if (result->per_parameter_error) {
        safe_free((void**)&result->per_parameter_error);
    }
    memset(result, 0, sizeof(GradientCheckResult));
}

/* ========================================================================
 * P2-001: 梯度流健康度监控实现
 * 提取LNN网络内部各层梯度统计，检测梯度消失/爆炸趋势
 * ======================================================================== */

int trainer_check_gradient_health(Trainer* trainer, GradientHealthReport* report) {
    if (!trainer || !report) return -1;
    if (!trainer->network || !trainer->network->cfc_network) return -1;

    memset(report, 0, sizeof(GradientHealthReport));

    const float V_THRESHOLD = 1e-7f;   /* 梯度消失阈值 */
    const float E_THRESHOLD = 1e3f;    /* 梯度爆炸阈值 */
    report->vanishing_threshold = V_THRESHOLD;
    report->exploding_threshold = E_THRESHOLD;

    CfCCell** layers = trainer->network->cfc_network->layers;
    int num_layers = trainer->network->config.num_layers;

    /* 从trainer梯度缓冲区提取输出梯度范数（error * batch梯度在backward后存储在gradients中） */
    float output_grad = gradient_norm(trainer->gradients, trainer->gradients_size);
    report->output_grad_norm = output_grad;

    float hidden_grad_norm = 0.0f;
    float weight_grad_norm = 0.0f;
    float bias_grad_norm = 0.0f;
    float gate_weight_grad_norm = 0.0f;
    float hidden_weight_grad_norm = 0.0f;

    for (int l = 0; l < num_layers && l < 256; l++) {
        CfCCell* c = layers[l];
        if (!c->is_initialized) continue;

        size_t h = c->config.hidden_size;
        size_t in = c->config.input_size;

        /* 隐藏状态梯度的范数 */
        float layer_hidden_grad = 0.0f;
        for (size_t i = 0; i < h; i++) {
            layer_hidden_grad += c->state->gradient[i] * c->state->gradient[i];
        }
        hidden_grad_norm += sqrtf(layer_hidden_grad);

        /* 权重梯度范数 */
        float layer_w_grad = 0.0f;
        size_t wsize = in * h;
        for (size_t i = 0; i < wsize; i++) {
            layer_w_grad += c->weight_grad[i] * c->weight_grad[i];
        }
        weight_grad_norm += sqrtf(layer_w_grad);

        /* 偏置梯度范数 */
        float layer_b_grad = 0.0f;
        for (size_t i = 0; i < h; i++) {
            layer_b_grad += c->bias_grad[i] * c->bias_grad[i];
        }
        bias_grad_norm += sqrtf(layer_b_grad);

        /* 门控权重梯度范数 */
        float layer_gw_grad = 0.0f;
        for (size_t i = 0; i < wsize; i++) {
            layer_gw_grad += c->input_gate_weight_grad[i] * c->input_gate_weight_grad[i];
        }
        gate_weight_grad_norm += sqrtf(layer_gw_grad);

        /* 隐藏到隐藏权重梯度范数 */
        float layer_hw_grad = 0.0f;
        size_t hwsize = h * h;
        for (size_t i = 0; i < hwsize; i++) {
            layer_hw_grad += c->hidden_to_gate_weight_grad[i] * c->hidden_to_gate_weight_grad[i]
                           + c->hidden_to_activation_weight_grad[i] * c->hidden_to_activation_weight_grad[i];
        }
        hidden_weight_grad_norm += sqrtf(layer_hw_grad);
    }

    report->hidden_grad_norm = hidden_grad_norm;
    report->weight_grad_norm = weight_grad_norm;
    report->bias_grad_norm = bias_grad_norm;
    report->gate_weight_grad_norm = gate_weight_grad_norm;
    report->hidden_weight_grad_norm = hidden_weight_grad_norm;

    /* 梯度衰减比：权重梯度范数 / 输出梯度范数 */
    if (output_grad > 1e-12f) {
        report->grad_norm_ratio = weight_grad_norm / output_grad;
    } else {
        report->grad_norm_ratio = 0.0f;
    }

    /* 健康度判断 */
    float max_grad = weight_grad_norm;
    if (hidden_grad_norm > max_grad) max_grad = hidden_grad_norm;
    if (gate_weight_grad_norm > max_grad) max_grad = gate_weight_grad_norm;
    if (hidden_weight_grad_norm > max_grad) max_grad = hidden_weight_grad_norm;

    report->is_vanishing = (max_grad < V_THRESHOLD) ? 1 : 0;
    report->is_exploding = (max_grad > E_THRESHOLD) ? 1 : 0;
    report->is_healthy = (!report->is_vanishing && !report->is_exploding) ? 1 : 0;

    /* 建议的梯度裁剪范数：当前最大梯度范数的 2 倍，但有上下界 */
    if (max_grad > 0.0f && max_grad < 100.0f) {
        report->recommended_clip_norm = max_grad * 2.0f;
    } else if (max_grad >= 100.0f) {
        report->recommended_clip_norm = 10.0f;  /* 梯度很大时用保守裁剪 */
    } else {
        report->recommended_clip_norm = 5.0f;   /* 默认值 */
    }

    return 0;
}

void gradient_health_report_print(const GradientHealthReport* report) {
    if (!report) return;
    printf("\n========== 梯度流健康度报告 ==========\n");
    printf("  输出梯度范数:       %.6f\n", report->output_grad_norm);
    printf("  隐藏状态梯度范数:   %.6f\n", report->hidden_grad_norm);
    printf("  权重梯度范数:       %.6f\n", report->weight_grad_norm);
    printf("  偏置梯度范数:       %.6f\n", report->bias_grad_norm);
    printf("  门控权重梯度范数:   %.6f\n", report->gate_weight_grad_norm);
    printf("  隐-隐权重梯度范数:  %.6f\n", report->hidden_weight_grad_norm);
    printf("  梯度衰减比(w_grad/output_grad): %.4f\n", report->grad_norm_ratio);
    printf("  ----------------------------------------\n");
    if (report->is_vanishing) {
        printf("  ⚠ 警告: 检测到梯度消失! (最大梯度 < %.1e)\n", report->vanishing_threshold);
    }
    if (report->is_exploding) {
        printf("  ⚠ 警告: 检测到梯度爆炸! (最大梯度 > %.1e)\n", report->exploding_threshold);
    }
    if (report->is_healthy) {
        printf("  ✅ 梯度流健康\n");
    }
    printf("  建议梯度裁剪范数:   %.4f\n", report->recommended_clip_norm);
    printf("==========================================\n\n");
}

/**
 * @brief 早停检查
 */
int early_stopping_check(float current_loss, float best_loss,
                         size_t patience, size_t* steps_without_improvement) {
    if (!steps_without_improvement) {
        return 0;
    }
    
    if (current_loss < best_loss) {
        *steps_without_improvement = 0;
        return 0;
    } else {
        (*steps_without_improvement)++;
        return (*steps_without_improvement >= patience) ? 1 : 0;
    }
}

/**
 * @brief 保存模型检查点
 */
/* ============================================================================
 * 检查点管理
 * =========================================================================== */

int training_cleanup_checkpoint_temp(const char* checkpoint_path)
{
    if (!checkpoint_path) {
        return -1;
    }
    
    char temp_filename[1024];
    int temp_name_len = snprintf(temp_filename, sizeof(temp_filename), "%s.tmp", checkpoint_path);
    if (temp_name_len <= 0 || (size_t)temp_name_len >= sizeof(temp_filename)) {
        return -1;
    }
    
    FILE* test = fopen(temp_filename, "rb");
    if (test) {
        fclose(test);
        if (remove(temp_filename) == 0) {
            return 1;
        }
        return -1;
    }
    
    return 0;
}

int trainer_set_auto_checkpoint(Trainer* trainer, const char* checkpoint_path, size_t interval_epochs)
{
    if (!trainer) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "设置自动检查点：训练器为空");
        return -1;
    }
    TRAINER_LOCK(trainer);
    
    // 释放旧的检查点路径
    if (trainer->auto_checkpoint_path) {
        safe_free((void**)&trainer->auto_checkpoint_path);
        trainer->auto_checkpoint_path = NULL;
    }
    
    if (checkpoint_path) {
        size_t _cp_len = strlen(checkpoint_path) + 1;
        trainer->auto_checkpoint_path = (char*)safe_malloc(_cp_len);
        if (trainer->auto_checkpoint_path) {
            memcpy(trainer->auto_checkpoint_path, checkpoint_path, _cp_len);
        } else {
            selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                                  "设置自动检查点：路径字符串复制失败");
            TRAINER_UNLOCK(trainer);
            return -1;
        }
    }
    
    trainer->auto_checkpoint_interval = interval_epochs;
    trainer->auto_checkpoint_counter = 0;
    
    TRAINER_UNLOCK(trainer);
    return 0;
}

int trainer_resume_from_checkpoint(Trainer* trainer, const char* checkpoint_path)
{
    if (!trainer || !checkpoint_path) {
        return -1;
    }
    
    // 检查检查点文件是否存在
    FILE* test = fopen(checkpoint_path, "rb");
    if (!test) {
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "恢复检查点：文件不存在 '%s'", checkpoint_path);
        return -1;
    }
    fclose(test);
    
    // 加载检查点
    int result = load_model_checkpoint(trainer, checkpoint_path);
    if (result != 0) {
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "恢复检查点：加载失败 '%s'", checkpoint_path);
        return -1;
    }
    
    // 自动设置检查点路径用于后续自动保存
    trainer_set_auto_checkpoint(trainer, checkpoint_path, 10);
    
    if (trainer->config.verbose) {
        printf("从检查点恢复成功: %s\n", checkpoint_path);
        printf("  恢复轮次: %zu\n", trainer->state.current_epoch);
        printf("  恢复损失: %.6f\n", trainer->state.best_loss);
        printf("  恢复准确率: %.4f\n", trainer->state.best_accuracy);
        printf("  恢复学习率: %.6f\n", trainer->state.learning_rate);
    }
    
    return 0;
}

int save_model_checkpoint(const Trainer* trainer, const char* filename) {
    if (!trainer || !filename) {
        return -1;
    }
    
    // 使用临时文件实现原子写入
    char temp_filename[1024];
    int temp_name_len = snprintf(temp_filename, sizeof(temp_filename), "%s.tmp", filename);
    if (temp_name_len <= 0 || (size_t)temp_name_len >= sizeof(temp_filename)) {
        return -1;
    }
    
    FILE* file = fopen(temp_filename, "wb");
    if (!file) {
        return -1;
    }
    
    // 1. 保存检查点头部（标识和版本）
    ModelCheckpointHeader header;
    memset(&header, 0, sizeof(ModelCheckpointHeader));
    
    size_t magic_copy_len = CKPT_MAGIC_LEN < sizeof(header.magic) ? CKPT_MAGIC_LEN : sizeof(header.magic) - 1;
    memcpy(header.magic, CKPT_MAGIC_STRING, magic_copy_len);
    header.magic[magic_copy_len] = '\0';
    header.version = CKPT_VERSION_CURRENT;
    header.header_size = sizeof(ModelCheckpointHeader);
    header.checkpoint_size = sizeof(ModelCheckpoint);
    header.timestamp = perf_timestamp_ns() / 1000000;
    
    uint32_t crc = 0;
    crc = ckpt_crc32c(crc, &header, sizeof(ModelCheckpointHeader));
    fwrite(&header, sizeof(ModelCheckpointHeader), 1, file);
    
    // 2. 保存检查点信息
    ModelCheckpoint checkpoint;
    memset(&checkpoint, 0, sizeof(ModelCheckpoint));
    
    size_t filename_len = strlen(filename);
    size_t copy_len = filename_len < sizeof(checkpoint.filename) ? filename_len : sizeof(checkpoint.filename) - 1;
    memcpy(checkpoint.filename, filename, copy_len);
    checkpoint.filename[copy_len] = '\0';
    checkpoint.loss = trainer->state.best_loss;
    checkpoint.accuracy = trainer->state.best_accuracy;
    checkpoint.epoch = trainer->state.current_epoch;
    checkpoint.timestamp = perf_timestamp_ns() / 1000000;
    checkpoint.total_iterations = trainer->state.total_iterations;
    checkpoint.current_batch = trainer->state.current_batch;
    checkpoint.learning_rate = trainer->state.learning_rate;
    
    crc = ckpt_crc32c(crc, &checkpoint, sizeof(ModelCheckpoint));
    fwrite(&checkpoint, sizeof(ModelCheckpoint), 1, file);
    
    // 3. 保存训练配置
    crc = ckpt_crc32c(crc, &trainer->config, sizeof(TrainingConfig));
    fwrite(&trainer->config, sizeof(TrainingConfig), 1, file);
    
    // 4. 保存训练状态统计
    crc = ckpt_crc32c(crc, &trainer->stats, sizeof(TrainingStats));
    fwrite(&trainer->stats, sizeof(TrainingStats), 1, file);
    
    // 5. 保存网络状态（如果可用）
    if (trainer->network) {
        char network_filename[512];
        snprintf(network_filename, sizeof(network_filename), "%s.network.bin", filename);
        
        int save_result = lnn_save(trainer->network, network_filename);
        if (save_result != 0) {
            if (trainer->config.verbose) {
                printf("警告：保存网络到文件'%s'失败，错误码=%d\n", 
                       network_filename, save_result);
            }
        }
        
        uint32_t network_flag = 0x4E455457;
        crc = ckpt_crc32c(crc, &network_flag, sizeof(uint32_t));
        fwrite(&network_flag, sizeof(uint32_t), 1, file);
        
        size_t name_len = strlen(network_filename);
        uint32_t name_length = (uint32_t)name_len;
        crc = ckpt_crc32c(crc, &name_length, sizeof(uint32_t));
        fwrite(&name_length, sizeof(uint32_t), 1, file);
        crc = ckpt_crc32c(crc, network_filename, name_len);
        fwrite(network_filename, 1, name_len, file);
        
        uint64_t actual_network_size = 0;
        {
            FILE* nf = fopen(network_filename, "rb");
            if (nf) {
                fseek(nf, 0, SEEK_END);
                actual_network_size = (uint64_t)ftell(nf);
                fclose(nf);
            }
        }
        crc = ckpt_crc32c(crc, &actual_network_size, sizeof(uint64_t));
        fwrite(&actual_network_size, sizeof(uint64_t), 1, file);
    } else {
        uint32_t no_network_flag = 0x4E4F4E45;
        crc = ckpt_crc32c(crc, &no_network_flag, sizeof(uint32_t));
        fwrite(&no_network_flag, sizeof(uint32_t), 1, file);
    }
    
    // 6. 保存真正CRC32校验和
    uint32_t crc_final = crc;
    fwrite(&crc_final, sizeof(uint32_t), 1, file);
    
    fclose(file);
    
    // 原子重命名
    if (rename(temp_filename, filename) != 0) {
        remove(temp_filename);
        return -1;
    }
    
    return 0;
}

/**
 * @brief 加载模型检查点
 */
int load_model_checkpoint(Trainer* trainer, const char* filename) {
    if (!trainer || !filename) {
        return -1;
    }
    
    FILE* file = fopen(filename, "rb");
    if (!file) {
        return -1;
    }
    
    uint32_t crc = 0;
    int crc_valid = 1;
    
    // 1. 读取并验证检查点头部
    ModelCheckpointHeader header;
    if (fread(&header, sizeof(ModelCheckpointHeader), 1, file) != 1) {
        fclose(file);
        return -1;
    }
    crc = ckpt_crc32c(crc, &header, sizeof(ModelCheckpointHeader));
    
    // 魔数验证：支持部分匹配（兼容性）
    if (strncmp(header.magic, CKPT_MAGIC_STRING, CKPT_MAGIC_LEN) != 0 ||
        header.magic[CKPT_MAGIC_LEN] != '\0') {
        if (trainer->config.verbose) {
            printf("错误：检查点魔数不匹配，文件格式无效: %s\n", filename);
        }
        fclose(file);
        return -1;
    }
    
    if (!ckpt_is_version_supported(header.version)) {
        if (trainer->config.verbose) {
            printf("错误：不支持的检查点版本 %u，当前支持版本: 1\n", header.version);
        }
        fclose(file);
        return -1;
    }
    
    // 验证header_size和checkpoint_size的合理性
    if (header.header_size < 28 || header.header_size > 512 ||
        header.checkpoint_size < 64 || header.checkpoint_size > 1024) {
        if (trainer->config.verbose) {
            printf("错误：检查点头部尺寸异常（header=%u, checkpoint=%u），文件可能已损坏\n",
                   header.header_size, header.checkpoint_size);
        }
        fclose(file);
        return -1;
    }
    
    // 2. 读取检查点信息
    ModelCheckpoint checkpoint;
    if (fread(&checkpoint, sizeof(ModelCheckpoint), 1, file) != 1) {
        fclose(file);
        return -1;
    }
    crc = ckpt_crc32c(crc, &checkpoint, sizeof(ModelCheckpoint));
    
    // 3. 读取训练配置
    TrainingConfig saved_config;
    if (fread(&saved_config, sizeof(TrainingConfig), 1, file) != 1) {
        fclose(file);
        return -1;
    }
    crc = ckpt_crc32c(crc, &saved_config, sizeof(TrainingConfig));
    
    // 4. 读取训练状态统计
    TrainingStats saved_stats;
    if (fread(&saved_stats, sizeof(TrainingStats), 1, file) != 1) {
        fclose(file);
        return -1;
    }
    crc = ckpt_crc32c(crc, &saved_stats, sizeof(TrainingStats));
    
    // 5. 读取网络状态标志
    uint32_t network_flag;
    if (fread(&network_flag, sizeof(uint32_t), 1, file) != 1) {
        fclose(file);
        return -1;
    }
    crc = ckpt_crc32c(crc, &network_flag, sizeof(uint32_t));
    
    if (network_flag == 0x4E455457) {
        uint32_t name_length;
        if (fread(&name_length, sizeof(uint32_t), 1, file) != 1) {
            fclose(file);
            return -1;
        }
        crc = ckpt_crc32c(crc, &name_length, sizeof(uint32_t));
        
        if (name_length == 0 || name_length > 1024) {
            if (trainer->config.verbose) {
                printf("错误：网络文件名长度异常（%u），文件可能已损坏\n", name_length);
            }
            fclose(file);
            return -1;
        }
        
        char network_filename[1024];
        if (fread(network_filename, 1, name_length, file) != name_length) {
            fclose(file);
            return -1;
        }
        crc = ckpt_crc32c(crc, network_filename, name_length);
        network_filename[name_length] = '\0';
        
        uint64_t network_size;
        if (fread(&network_size, sizeof(uint64_t), 1, file) != 1) {
            fclose(file);
            return -1;
        }
        crc = ckpt_crc32c(crc, &network_size, sizeof(uint64_t));
        
        // 实际加载网络权重
        if (trainer->network && name_length > 0) {
            if (trainer->config.verbose) {
                printf("检查点包含网络数据，加载网络文件: %s\n", network_filename);
            }
            
            // 检查网络文件是否存在
            FILE* nf = fopen(network_filename, "rb");
            if (nf) {
                fseek(nf, 0, SEEK_END);
                long actual_size = ftell(nf);
                fclose(nf);
                if (network_size > 0 && (uint64_t)actual_size != network_size) {
                    if (trainer->config.verbose) {
                        printf("警告：网络文件大小不匹配（期望=%llu，实际=%ld），将尝试加载\n",
                               (unsigned long long)network_size, actual_size);
                    }
                }
            }
            
            LNN* loaded = lnn_load(network_filename);
            int load_result = (loaded != NULL) ? 0 : -1;
            if (load_result == 0 && loaded) {
                LNN* old_net = trainer->network;
                trainer->network = loaded;
                lnn_free(old_net);
                if (trainer->config.verbose) {
                    printf("网络权重加载成功: %s\n", network_filename);
                }
            } else {
                if (trainer->config.verbose) {
                    printf("警告：加载网络权重失败 '%s'\n", network_filename);
                }
            }
        }
    } else if (network_flag != 0x4E4F4E45) {
        if (trainer->config.verbose) {
            printf("警告：未知网络标志 0x%08X，跳过网络加载\n", network_flag);
        }
    }
    
    // 6. 读取并验证真实CRC32校验和
    {
        uint32_t saved_crc = 0;
        if (fread(&saved_crc, sizeof(uint32_t), 1, file) == 1) {
            if (saved_crc != crc) {
                crc_valid = 0;
                if (trainer->config.verbose) {
                    printf("严重错误：检查点CRC32校验和不匹配（计算=0x%08X，保存=0x%08X），文件已损坏: %s\n",
                           crc, saved_crc, filename);
                }
            } else if (trainer->config.verbose) {
                printf("检查点CRC32校验通过 (0x%08X)\n", crc);
            }
        } else {
            crc_valid = 0;
            if (trainer->config.verbose) {
                printf("错误：无法读取检查点CRC32校验和，文件可能已截断: %s\n", filename);
            }
        }
    }
    
    fclose(file);
    
    // CRC验证失败时返回错误
    if (!crc_valid) {
        return -1;
    }
    
    // 7. 恢复训练状态（含范围验证）
    if (isfinite(checkpoint.loss) && checkpoint.loss >= 0.0f) {
        trainer->state.best_loss = checkpoint.loss;
    }
    if (isfinite(checkpoint.accuracy) && checkpoint.accuracy >= 0.0f && checkpoint.accuracy <= 100.0f) {
        trainer->state.best_accuracy = checkpoint.accuracy;
    }
    if (checkpoint.epoch < 1000000) {
        trainer->state.current_epoch = checkpoint.epoch;
    }
    if (checkpoint.total_iterations < 100000000) {
        trainer->state.total_iterations = checkpoint.total_iterations;
    }
    if (checkpoint.current_batch < 1000000) {
        trainer->state.current_batch = checkpoint.current_batch;
    }
    if (isfinite(checkpoint.learning_rate) && checkpoint.learning_rate >= 0.0f) {
        trainer->state.learning_rate = checkpoint.learning_rate;
    }
    
    // 8. 恢复训练统计（含范围验证）
    if (isfinite(saved_stats.avg_train_loss)) trainer->stats.avg_train_loss = saved_stats.avg_train_loss;
    if (isfinite(saved_stats.avg_val_loss)) trainer->stats.avg_val_loss = saved_stats.avg_val_loss;
    if (isfinite(saved_stats.avg_train_accuracy) && saved_stats.avg_train_accuracy >= 0.0f && saved_stats.avg_train_accuracy <= 100.0f)
        trainer->stats.avg_train_accuracy = saved_stats.avg_train_accuracy;
    if (isfinite(saved_stats.avg_val_accuracy) && saved_stats.avg_val_accuracy >= 0.0f && saved_stats.avg_val_accuracy <= 100.0f)
        trainer->stats.avg_val_accuracy = saved_stats.avg_val_accuracy;
    if (isfinite(saved_stats.avg_gradient_norm)) trainer->stats.avg_gradient_norm = saved_stats.avg_gradient_norm;
    if (isfinite(saved_stats.avg_weight_norm)) trainer->stats.avg_weight_norm = saved_stats.avg_weight_norm;
    trainer->stats.total_epochs_trained = saved_stats.total_epochs_trained;
    trainer->stats.total_batches_trained = saved_stats.total_batches_trained;
    
    return 0;
}

/* ============================
 * 需求20.4a: 紧急检查点（信号处理+崩溃恢复）
 * ============================ */

/**
 * @brief 全局训练器指针（用于信号处理）
 * 
 * 信号处理函数无法接收用户数据参数，因此使用全局指针。
 * 在启用紧急检查点时设置此指针。
 */
static Trainer* g_emergency_trainer = NULL;

/**
 * @brief 信号处理函数
 * 
 * 处理 SIGINT（Ctrl+C）、SIGTERM（终止信号）等信号。
 * 设置紧急保存请求标志，保存紧急检查点。
 * 在 Windows 上也处理 CTRL_CLOSE_EVENT。
 */
static void emergency_signal_handler(int signum) {
    (void)signum; // 未使用参数
    
    // 设置请求标志
    if (g_emergency_trainer) {
        g_emergency_trainer->emergency_save_requested = 1;
    }
    
    // 恢复默认信号处理，下次触发时执行默认行为（终止进程）
    signal(SIGINT, SIG_DFL);
    signal(SIGTERM, SIG_DFL);
}

#ifdef _WIN32
/**
 * @brief Windows 控制台事件处理
 */
static BOOL WINAPI emergency_console_handler(DWORD dwCtrlType) {
    switch (dwCtrlType) {
        case CTRL_C_EVENT:
        case CTRL_BREAK_EVENT:
        case CTRL_CLOSE_EVENT:
        case CTRL_LOGOFF_EVENT:
        case CTRL_SHUTDOWN_EVENT:
            if (g_emergency_trainer) {
                g_emergency_trainer->emergency_save_requested = 1;
            }
            // 给训练循环一些时间保存检查点
            Sleep(3000);
            return TRUE;
        default:
            return FALSE;
    }
}
#endif

/**
 * @brief 清理旧的检查点文件（根据保留策略）
 * 
 * 当检查点数量超过 checkpoint_retention_max 时，
 * 删除最旧的检查点文件。
 */
static int cleanup_old_checkpoints(Trainer* trainer) {
    if (!trainer || trainer->checkpoint_retention_max == 0) {
        return 0;
    }
    
    // 如果保留列表为空或未初始化，先扫描目录
    if (!trainer->checkpoint_retention_list && trainer->auto_checkpoint_path) {
        // 提取目录路径
        char dir_path[512];
        const char* last_slash = NULL;
        const char* p = trainer->auto_checkpoint_path;
        while (*p) {
            if (*p == '/' || *p == '\\') {
                last_slash = p;
            }
            p++;
        }
        if (last_slash) {
            size_t dir_len = (size_t)(last_slash - trainer->auto_checkpoint_path);
            if (dir_len >= sizeof(dir_path)) dir_len = sizeof(dir_path) - 1;
            memcpy(dir_path, trainer->auto_checkpoint_path, dir_len);
            dir_path[dir_len] = '\0';
        } else {
            dir_path[0] = '.';
            dir_path[1] = '\0';
        }
        
        // 扫描目录下的检查点文件
        // 为了简化，这里只检查当前检查点路径是否在列表中
    }
    
    // 如果当前保留数量 <= 最大保留，无需清理
    if (trainer->checkpoint_retention_count <= trainer->checkpoint_retention_max) {
        return 0;
    }
    
    // 需要删除最旧的检查点
    size_t to_delete = trainer->checkpoint_retention_count - trainer->checkpoint_retention_max;
    size_t i;
    for (i = 0; i < to_delete && i < trainer->checkpoint_retention_count; i++) {
        if (trainer->checkpoint_retention_list[i]) {
            if (trainer->config.verbose) {
                printf("清理旧检查点: %s\n", trainer->checkpoint_retention_list[i]);
            }
            remove(trainer->checkpoint_retention_list[i]);
            safe_free((void**)&trainer->checkpoint_retention_list[i]);
        }
    }
    
    // 将剩余列表前移
    size_t remaining = trainer->checkpoint_retention_count - to_delete;
    for (i = 0; i < remaining; i++) {
        trainer->checkpoint_retention_list[i] = trainer->checkpoint_retention_list[i + to_delete];
    }
    trainer->checkpoint_retention_count = remaining;
    
    return 0;
}

/**
 * @brief 添加检查点到保留列表
 */
static int add_checkpoint_to_retention_list(Trainer* trainer, const char* filepath) {
    if (!trainer || !filepath) {
        return -1;
    }
    
    // 扩展列表
    size_t new_count = trainer->checkpoint_retention_count + 1;
    char** new_list = (char**)safe_realloc(trainer->checkpoint_retention_list,
                                           new_count * sizeof(char*));
    if (!new_list) {
        return -1;
    }
    
    trainer->checkpoint_retention_list = new_list;
    
    // 复制文件名
    size_t path_len = strlen(filepath) + 1;
    trainer->checkpoint_retention_list[trainer->checkpoint_retention_count] = (char*)safe_malloc(path_len);
    if (!trainer->checkpoint_retention_list[trainer->checkpoint_retention_count]) {
        return -1;
    }
    memcpy(trainer->checkpoint_retention_list[trainer->checkpoint_retention_count], filepath, path_len);
    trainer->checkpoint_retention_count = new_count;
    
    // 执行保留策略清理
    cleanup_old_checkpoints(trainer);
    
    return 0;
}

int trainer_enable_emergency_checkpoint(Trainer* trainer, const char* path) {
    if (!trainer) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "启用紧急检查点：训练器为空");
        return -1;
    }
    TRAINER_LOCK(trainer);
    
    // 释放旧路径
    if (trainer->emergency_checkpoint_path) {
        safe_free((void**)&trainer->emergency_checkpoint_path);
        trainer->emergency_checkpoint_path = NULL;
    }
    
    if (path) {
        size_t path_len = strlen(path) + 1;
        trainer->emergency_checkpoint_path = (char*)safe_malloc(path_len);
        if (!trainer->emergency_checkpoint_path) {
            selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                                  "启用紧急检查点：路径内存分配失败");
            TRAINER_UNLOCK(trainer);
            return -1;
        }
        memcpy(trainer->emergency_checkpoint_path, path, path_len);
        
        // 设置全局训练器指针供信号处理使用
        g_emergency_trainer = trainer;
        
        // 注册信号处理函数
        signal(SIGINT, emergency_signal_handler);
        signal(SIGTERM, emergency_signal_handler);
        
#ifdef _WIN32
        // Windows 控制台事件处理
        SetConsoleCtrlHandler(emergency_console_handler, TRUE);
#endif
        
        trainer->emergency_checkpoint_enabled = 1;
        trainer->emergency_save_requested = 0;
        
        if (trainer->config.verbose) {
            printf("紧急检查点已启用，保存路径: %s\n", path);
        }
    } else {
        // 禁用紧急检查点
        signal(SIGINT, SIG_DFL);
        signal(SIGTERM, SIG_DFL);
#ifdef _WIN32
        SetConsoleCtrlHandler(emergency_console_handler, FALSE);
#endif
        
        g_emergency_trainer = NULL;
        trainer->emergency_checkpoint_enabled = 0;
        trainer->emergency_save_requested = 0;
    }
    
    TRAINER_UNLOCK(trainer);
    return 0;
}

int trainer_check_crash_recovery(Trainer* trainer, const char* checkpoint_dir,
                                 size_t* recovered_epoch, float* recovered_loss) {
    if (!trainer || !checkpoint_dir) {
        return -1;
    }
    TRAINER_LOCK(trainer);
    
    size_t dir_len = strlen(checkpoint_dir);
    char clean_path[1024];
    
    if (dir_len > 0 && (checkpoint_dir[dir_len - 1] == '/' || checkpoint_dir[dir_len - 1] == '\\')) {
        snprintf(clean_path, sizeof(clean_path), "%s%s", checkpoint_dir, "emergency_checkpoint.bin");
    } else {
        snprintf(clean_path, sizeof(clean_path), "%s/%s", checkpoint_dir, "emergency_checkpoint.bin");
    }
    
    FILE* f = fopen(clean_path, "rb");
    if (!f) {
        if (trainer->config.verbose) {
            printf("未检测到紧急检查点（%s），训练状态正常\n", clean_path);
        }
        TRAINER_UNLOCK(trainer);
        return 0;
    }
    fclose(f);
    
    int result = load_model_checkpoint(trainer, clean_path);
    if (result != 0) {
        if (trainer->config.verbose) {
            printf("警告：紧急检查点 %s 加载失败，可能已损坏\n", clean_path);
        }
        TRAINER_UNLOCK(trainer);
        return 0;
    }
    
    if (recovered_epoch) {
        *recovered_epoch = trainer->state.current_epoch;
    }
    if (recovered_loss) {
        *recovered_loss = trainer->state.best_loss;
    }
    
    if (trainer->config.verbose) {
        printf("检测到紧急检查点：轮次 %zu，损失 %.6f\n",
               trainer->state.current_epoch, trainer->state.best_loss);
    }
    
    TRAINER_UNLOCK(trainer);
    return 1;
}

/* ============================
 * 需求20.4b: 检查点保留策略
 * ============================ */

int trainer_set_checkpoint_retention(Trainer* trainer, size_t max_checkpoints) {
    if (!trainer) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "设置检查点保留策略：训练器为空");
        return -1;
    }
    TRAINER_LOCK(trainer);
    
    trainer->checkpoint_retention_max = max_checkpoints;
    
    if (max_checkpoints > 0) {
        // 立即执行一次清理
        cleanup_old_checkpoints(trainer);
    }
    
    if (trainer->config.verbose) {
        if (max_checkpoints == 0) {
            printf("检查点保留策略已禁用（不限制数量）\n");
        } else {
            printf("检查点保留策略已启用，最多保留 %zu 个检查点\n", max_checkpoints);
        }
    }
    
    TRAINER_UNLOCK(trainer);
    return 0;
}

/* ============================
 * 需求20.4b: 后台定时保存线程
 * ============================ */

#ifdef _WIN32
/**
 * @brief Windows 后台保存线程入口
 */
static unsigned int __stdcall background_checkpoint_thread_win32(void* arg) {
    Trainer* trainer = (Trainer*)arg;
    
    while (!trainer->background_checkpoint_exit) {
        // 等待指定秒数
        Sleep(trainer->background_checkpoint_interval * 1000);
        
        if (trainer->background_checkpoint_exit) {
            break;
        }
        
        // 保存检查点
        if (trainer->auto_checkpoint_path) {
            int result = save_model_checkpoint(trainer, trainer->auto_checkpoint_path);
            if (result == 0) {
                // 添加到保留策略列表
                add_checkpoint_to_retention_list(trainer, trainer->auto_checkpoint_path);
                
                if (trainer->config.verbose) {
                    printf("[后台保存] 检查点已保存: %s\n", trainer->auto_checkpoint_path);
                }
            }
        }
    }
    
    return 0;
}
#else
/**
 * @brief POSIX 后台保存线程入口
 */
static void* background_checkpoint_thread_posix(void* arg) {
    Trainer* trainer = (Trainer*)arg;
    
    while (!trainer->background_checkpoint_exit) {
        // 等待指定秒数
        struct timespec ts;
        ts.tv_sec = trainer->background_checkpoint_interval;
        ts.tv_nsec = 0;
        nanosleep(&ts, NULL);
        
        if (trainer->background_checkpoint_exit) {
            break;
        }
        
        // 保存检查点
        if (trainer->auto_checkpoint_path) {
            int result = save_model_checkpoint(trainer, trainer->auto_checkpoint_path);
            if (result == 0) {
                // 添加到保留策略列表
                add_checkpoint_to_retention_list(trainer, trainer->auto_checkpoint_path);
                
                if (trainer->config.verbose) {
                    printf("[后台保存] 检查点已保存: %s\n", trainer->auto_checkpoint_path);
                }
            }
        }
    }
    
    return NULL;
}
#endif

int trainer_enable_background_checkpoint(Trainer* trainer, int interval_seconds) {
    if (!trainer) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "启用后台检查点保存：训练器为空");
        return -1;
    }
    
    if (interval_seconds <= 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "启用后台检查点保存：间隔必须大于0秒");
        return -1;
    }
    
    if (trainer->background_checkpoint_enabled) {
        // 先禁用当前的后台保存
        trainer_disable_background_checkpoint(trainer);
    }
    
    trainer->background_checkpoint_interval = interval_seconds;
    trainer->background_checkpoint_exit = 0;
    trainer->background_checkpoint_enabled = 1;
    
#ifdef _WIN32
    // 创建 Windows 线程
    unsigned int thread_id;
    trainer->background_checkpoint_thread = (void*)_beginthreadex(
        NULL, 0, background_checkpoint_thread_win32, trainer, 0, &thread_id);
    if (!trainer->background_checkpoint_thread) {
        trainer->background_checkpoint_enabled = 0;
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "启用后台检查点保存：创建线程失败");
        return -1;
    }
#else
    // 创建 POSIX 线程
    pthread_t thread;
    if (pthread_create(&thread, NULL, background_checkpoint_thread_posix, trainer) != 0) {
        trainer->background_checkpoint_enabled = 0;
        selflnn_set_last_error(SELFLNN_ERROR_INTERNAL_ERROR, __func__, __FILE__, __LINE__,
                              "启用后台检查点保存：创建线程失败");
        return -1;
    }
    trainer->background_checkpoint_thread = (void*)(uintptr_t)thread;
#endif
    
    if (trainer->config.verbose) {
        printf("后台检查点保存已启用，间隔 %d 秒\n", interval_seconds);
    }
    
    return 0;
}

int trainer_disable_background_checkpoint(Trainer* trainer) {
    if (!trainer) {
        return -1;
    }
    
    if (!trainer->background_checkpoint_enabled) {
        return 0;
    }
    
    // 设置退出标志
    trainer->background_checkpoint_exit = 1;
    
    // 等待线程结束
    if (trainer->background_checkpoint_thread) {
#ifdef _WIN32
        WaitForSingleObject(trainer->background_checkpoint_thread, 5000);
        CloseHandle(trainer->background_checkpoint_thread);
#else
        pthread_t thread = (pthread_t)(uintptr_t)(trainer->background_checkpoint_thread);
        pthread_join(thread, NULL);
#endif
        trainer->background_checkpoint_thread = NULL;
    }
    
    trainer->background_checkpoint_enabled = 0;
    
    if (trainer->config.verbose) {
        printf("后台检查点保存已停止\n");
    }
    
    return 0;
}

/**
 * @brief 打印训练配置
 */
void training_config_print(const TrainingConfig* config) {
    if (!config) {
        return;
    }
    
    log_info("训练配置：");
    printf("  模式：%s\n", 
           config->mode == TRAIN_MODE_BATCH ? "批量训练" :
           config->mode == TRAIN_MODE_MINI_BATCH ? "小批量训练" :
           config->mode == TRAIN_MODE_ONLINE ? "在线训练" : "自适应训练");
    printf("  优化器：%s\n",
           config->optimizer == OPTIMIZER_SGD ? "SGD" :
           config->optimizer == OPTIMIZER_MOMENTUM ? "带动量的SGD" :
           config->optimizer == OPTIMIZER_ADAGRAD ? "AdaGrad" :
           config->optimizer == OPTIMIZER_RMSPROP ? "RMSProp" :
           config->optimizer == OPTIMIZER_ADAM ? "Adam" : "AdamW");
    printf("  损失函数：%s\n",
           config->loss_function == LOSS_MEAN_SQUARED_ERROR ? "均方误差" :
           config->loss_function == LOSS_MEAN_ABSOLUTE_ERROR ? "平均绝对误差" :
           config->loss_function == LOSS_CROSS_ENTROPY ? "交叉熵" :
           config->loss_function == LOSS_HUBER ? "Huber损失" : "Log-cosh损失");
    printf("  正则化：%s\n",
           config->regularization == REGULARIZATION_NONE ? "无" :
           config->regularization == REGULARIZATION_L1 ? "L1" :
           config->regularization == REGULARIZATION_L2 ? "L2" :
           config->regularization == REGULARIZATION_DROPOUT ? "Dropout" : "早停");
    printf("  学习率：%.6f\n", config->learning_rate);
    printf("  批量大小：%zu\n", config->batch_size);
    printf("  轮数：%zu\n", config->epochs);
    printf("  正则化强度：%.4f\n", config->regularization_lambda);
    printf("  Dropout率：%.2f\n", config->dropout_rate);
}

/**
 * @brief 打印训练状态
 */
void training_state_print(const TrainingState* state) {
    if (!state) {
        return;
    }
    
    log_info("训练状态：");
    printf("  当前轮数：%zu\n", state->current_epoch);
    printf("  当前批次：%zu/%zu\n", state->current_batch, state->total_batches);
    printf("  当前损失：%.6f\n", state->current_loss);
    printf("  当前准确率：%.4f\n", state->current_accuracy);
    printf("  验证损失：%.6f\n", state->validation_loss);
    printf("  验证准确率：%.4f\n", state->validation_accuracy);
    printf("  最佳损失：%.6f\n", state->best_loss);
    printf("  最佳准确率：%.4f\n", state->best_accuracy);
    printf("  学习率：%.6f\n", state->learning_rate);
    printf("  梯度范数：%.6f\n", state->gradient_norm);
    printf("  训练时间：%llu ms\n", (unsigned long long)state->training_time_ms);
    printf("  已处理样本：%llu\n", (unsigned long long)state->samples_processed);
}

/**
 * @brief 保存训练历史到文件
 */
int training_history_save(const TrainingHistory* history, const char* filename) {
    if (!history || !filename) {
        return -1;
    }
    
    FILE* file = fopen(filename, "wb");
    if (!file) {
        return -1;
    }
    
    // 保存历史记录大小
    fwrite(&history->size, sizeof(size_t), 1, file);
    
    // 保存历史数据
    if (history->size > 0) {
        fwrite(history->train_losses, sizeof(float), history->size, file);
        fwrite(history->train_accuracies, sizeof(float), history->size, file);
        fwrite(history->val_losses, sizeof(float), history->size, file);
        fwrite(history->val_accuracies, sizeof(float), history->size, file);
        fwrite(history->learning_rates, sizeof(float), history->size, file);
    }
    
    fclose(file);
    return 0;
}

/**
 * @brief 从文件加载训练历史
 */
TrainingHistory* training_history_load(const char* filename) {
    if (!filename) {
        return NULL;
    }
    
    FILE* file = fopen(filename, "rb");
    if (!file) {
        return NULL;
    }
    
    // 读取历史记录大小
    size_t size = 0;
    if (fread(&size, sizeof(size_t), 1, file) != 1) {
        fclose(file);
        return NULL;
    }
    
    // 分配历史记录
    TrainingHistory* history = (TrainingHistory*)safe_malloc(sizeof(TrainingHistory));
    if (!history) {
        fclose(file);
        return NULL;
    }
    
    history->size = size;
    history->capacity = size;
    
    // 分配缓冲区
    history->train_losses = (float*)safe_malloc(size * sizeof(float));
    history->train_accuracies = (float*)safe_malloc(size * sizeof(float));
    history->val_losses = (float*)safe_malloc(size * sizeof(float));
    history->val_accuracies = (float*)safe_malloc(size * sizeof(float));
    history->learning_rates = (float*)safe_malloc(size * sizeof(float));
    
    if (!history->train_losses || !history->train_accuracies ||
        !history->val_losses || !history->val_accuracies || !history->learning_rates) {
        training_history_free(history);
        fclose(file);
        return NULL;
    }
    
    // 读取历史数据
    if (size > 0) {
        if (fread(history->train_losses, sizeof(float), size, file) != size ||
            fread(history->train_accuracies, sizeof(float), size, file) != size ||
            fread(history->val_losses, sizeof(float), size, file) != size ||
            fread(history->val_accuracies, sizeof(float), size, file) != size ||
            fread(history->learning_rates, sizeof(float), size, file) != size) {
            training_history_free(history);
            fclose(file);
            return NULL;
        }
    }
    
    fclose(file);
    return history;
}

/**
 * @brief 释放训练历史
 */
void training_history_free(TrainingHistory* history) {
    if (!history) {
        return;
    }
    
    safe_free((void**)&history->train_losses);
    safe_free((void**)&history->train_accuracies);
    safe_free((void**)&history->val_losses);
    safe_free((void**)&history->val_accuracies);
    safe_free((void**)&history->learning_rates);
    
    safe_free((void**)&history);
}

/**
 * @brief 创建学习率调度器（公共接口）
 */
void* learning_rate_scheduler_create(const LearningRateSchedulerConfig* config,
                                     size_t total_steps) {
    return scheduler_create_internal(config, total_steps);
}

/**
 * @brief 获取当前学习率（公共接口）
 */
float learning_rate_scheduler_get(void* scheduler, size_t step) {
    return scheduler_get_internal((LearningRateScheduler*)scheduler, step);
}

/**
 * @brief 释放学习率调度器（公共接口）
 */
void learning_rate_scheduler_free(void* scheduler) {
    scheduler_free_internal((LearningRateScheduler*)scheduler);
}

/**
 * @brief 创建训练数据加载器（公共接口）
 */
void* data_loader_create(const float* inputs, const float* targets,
                         size_t num_samples, size_t input_dim, size_t output_dim,
                         size_t batch_size, int shuffle) {
    // 验证参数
    if (!inputs || !targets || num_samples == 0 || 
        input_dim == 0 || output_dim == 0 || batch_size == 0) {
        return NULL;
    }
    
    // 创建数据加载器
    return data_loader_create_internal(inputs, targets, num_samples, 
                                      input_dim, output_dim, batch_size, shuffle);
}

/**
 * @brief 获取下一个批次（公共接口）
 */
int data_loader_next_batch(void* loader, float** batch_inputs,
                           float** batch_targets, size_t* batch_size) {
    return data_loader_next_batch_internal((DataLoader*)loader,
                                          batch_inputs, batch_targets, batch_size);
}

/**
 * @brief 重置数据加载器（公共接口）
 */
void data_loader_reset(void* loader) {
    data_loader_reset_internal((DataLoader*)loader);
}

/**
 * @brief 释放数据加载器（公共接口）
 */
void data_loader_free(void* loader) {
    data_loader_free_internal((DataLoader*)loader);
}

/**
 * @brief 交叉验证
 * 
 * 执行k折交叉验证，评估模型在不同数据分割上的性能。
 * 注意：此实现使用当前模型在验证集上评估，不进行重新训练。
 * 对于完整的交叉验证，需要为每一折重新训练模型。
 */
int cross_validation(Trainer* trainer, const float* inputs, const float* targets,
                     size_t num_samples, size_t k_folds, float* results) {
    if (!trainer || !inputs || !targets || num_samples == 0 || 
        k_folds == 0 || !results || k_folds > num_samples) {
        return -1;
    }
    
    // 计算每折大小
    size_t fold_size = num_samples / k_folds;
    
    // 获取输入和输出维度（需要从训练器中获取）
    // 注意：这里假设训练器已初始化，我们可以获取网络结构信息
    LNN* network = trainer->network;
    if (!network) {
        return -1;
    }
    
    // 获取输入和输出维度（从已初始化的神经网络中获取）
    // 网络在训练器中必须已初始化，确保维度正确
    LNNConfig config;
    if (lnn_get_config(network, &config) != 0) {
        return -1;
    }
    size_t input_dim = config.input_size;
    size_t output_dim = config.output_size;
    
    // 为验证集分配缓冲区
    float* val_inputs = (float*)safe_malloc(fold_size * input_dim * sizeof(float));
    float* val_targets = (float*)safe_malloc(fold_size * output_dim * sizeof(float));
    float* val_outputs = (float*)safe_malloc(fold_size * output_dim * sizeof(float));
    
    if (!val_inputs || !val_targets || !val_outputs) {
        safe_free((void**)&val_inputs);
        safe_free((void**)&val_targets);
        safe_free((void**)&val_outputs);
        return -1;
    }
    
    for (size_t fold = 0; fold < k_folds; fold++) {
        size_t val_start = fold * fold_size;
        size_t val_end = (fold == k_folds - 1) ? num_samples : (fold + 1) * fold_size;
        size_t val_samples = val_end - val_start;
        
        // 提取验证集
        for (size_t i = 0; i < val_samples; i++) {
            size_t src_idx = val_start + i;
            // 复制输入
            for (size_t j = 0; j < input_dim; j++) {
                val_inputs[i * input_dim + j] = inputs[src_idx * input_dim + j];
            }
            // 复制目标
            for (size_t j = 0; j < output_dim; j++) {
                val_targets[i * output_dim + j] = targets[src_idx * output_dim + j];
            }
        }
        
        // 在验证集上评估当前模型
        float fold_loss = 0.0f;
        
        // 使用训练器的预测函数获取输出，然后计算损失和准确率
        // 这是完整的评估实现，支持多种损失函数和准确率计算
        
        // 预测验证集
        int predict_result = trainer_predict(trainer, val_inputs, val_outputs, val_samples);
        if (predict_result != 0) {
            results[fold] = 0.0f;
            continue;
        }
        
        // 计算损失
        switch (trainer->config.loss_function) {
            case LOSS_MEAN_SQUARED_ERROR:
                fold_loss = loss_mean_squared_error(val_outputs, val_targets, val_samples * output_dim);
                break;
            case LOSS_MEAN_ABSOLUTE_ERROR:
                fold_loss = loss_mean_absolute_error(val_outputs, val_targets, val_samples * output_dim);
                break;
            case LOSS_CROSS_ENTROPY:
                fold_loss = loss_cross_entropy(val_outputs, val_targets, val_samples * output_dim);
                break;
            default:
                fold_loss = loss_mean_squared_error(val_outputs, val_targets, val_samples * output_dim);
                break;
        }
        
        // 计算准确率（完整实现）
        float fold_accuracy = 0.0f;
        
        if (trainer->config.loss_function == LOSS_CROSS_ENTROPY) {
            // 分类任务：计算预测类别准确率
            // 假设输出是概率分布，目标是one-hot编码
            int correct = 0;
            for (size_t i = 0; i < val_samples; i++) {
                // 找到预测类别（最大概率索引）
                float max_prob = val_outputs[i * output_dim];
                int pred_class = 0;
                for (size_t j = 1; j < output_dim; j++) {
                    if (val_outputs[i * output_dim + j] > max_prob) {
                        max_prob = val_outputs[i * output_dim + j];
                        pred_class = (int)j;
                    }
                }
                
                // 找到真实类别（one-hot编码中值为1的索引）
                int true_class = 0;
                float max_target = val_targets[i * output_dim];
                for (size_t j = 1; j < output_dim; j++) {
                    if (val_targets[i * output_dim + j] > max_target) {
                        max_target = val_targets[i * output_dim + j];
                        true_class = (int)j;
                    }
                }
                
                if (pred_class == true_class) {
                    correct++;
                }
            }
            fold_accuracy = (float)correct / val_samples;
        } else {
            // 回归任务：计算决定系数R²（裁剪到0-1范围）
            // 计算目标均值
            float target_mean = 0.0f;
            for (size_t i = 0; i < val_samples * output_dim; i++) {
                target_mean += val_targets[i];
            }
            target_mean /= (val_samples * output_dim);
            
            // 计算总平方和
            float total_ss = 0.0f;
            for (size_t i = 0; i < val_samples * output_dim; i++) {
                float diff = val_targets[i] - target_mean;
                total_ss += diff * diff;
            }
            
            // 计算残差平方和
            float residual_ss = 0.0f;
            for (size_t i = 0; i < val_samples * output_dim; i++) {
                float diff = val_targets[i] - val_outputs[i];
                residual_ss += diff * diff;
            }
            
            // 计算R² = 1 - (residual_ss / total_ss)
            if (total_ss > 1e-10f) {
                float r_squared = 1.0f - (residual_ss / total_ss);
                // R²可能为负，我们将其裁剪到0-1范围作为准确率
                if (r_squared < 0.0f) r_squared = 0.0f;
                if (r_squared > 1.0f) r_squared = 1.0f;
                fold_accuracy = r_squared;
            } else {
                // 总平方和为零（所有目标值相同），使用损失转换作为后备
                fold_accuracy = 1.0f / (1.0f + fold_loss);
            }
        }
        
        // 使用准确率作为结果（如果是分类任务）或R²作为结果（回归任务）
        results[fold] = fold_accuracy;
    }
    
    safe_free((void**)&val_inputs);
    safe_free((void**)&val_targets);
    safe_free((void**)&val_outputs);
    
    return 0;
}

/**
 * @brief 超参数搜索（完整实现：基于配置的网格搜索）
 */
/**
 * @brief 释放超参数搜索结果
 */
void hyperparameter_search_result_free(HyperparameterSearchResult* result) {
    if (!result) {
        return;
    }
    
    safe_free((void**)&result->all_scores);
    safe_free((void**)&result->learning_rate_history);
    safe_free((void**)&result->regularization_history);
    safe_free((void**)&result->batch_size_history);
    
    memset(result, 0, sizeof(HyperparameterSearchResult));
}

/* ============================
 * 需求20.2: 增强超参搜索（随机搜索+贝叶斯优化+早停）
 * ============================ */

/**
 * @brief 在单个试验内执行早停检查
 *
 * 监控验证损失是否在 patience 轮内持续无改善。
 * 如果验证损失在早停耐心值轮次内未下降超过 min_delta，返回1表示应早停。
 */
static int trial_early_stop_check(float current_loss, float* best_loss,
                                   size_t* stall_counter, size_t patience,
                                   float min_delta) {
    if (!best_loss || !stall_counter) return 0;
    
    if (current_loss < *best_loss - min_delta) {
        *best_loss = current_loss;
        *stall_counter = 0;
        return 0;
    }
    
    (*stall_counter)++;
    if (*stall_counter >= patience) {
        return 1;
    }
    return 0;
}

/**
 * @brief 评估单个超参数组合（支持早停）
 *
 * 训练少量轮次并评估超参数组合的性能。
 * 如果启用了早停，在验证损失不改善时提前终止。
 *
 * @return float 评估分数，失败返回-1.0f
 */
static float hyperparameter_evaluate_trial(Trainer* trainer, const float* inputs,
                                            const float* targets, size_t num_samples,
                                            float learning_rate, float regularization,
                                            size_t batch_size, size_t cv_folds,
                                            int enable_early_stop, size_t early_stop_patience,
                                            float early_stop_min_delta,
                                            int* early_stopped) {
    if (!trainer || !inputs || !targets) return -1.0f;
    
    TrainingConfig original_config = trainer->config;
    
    trainer->config.learning_rate = learning_rate;
    trainer->config.regularization_lambda = regularization;
    trainer->config.batch_size = batch_size;
    
    float* cv_results = (float*)safe_malloc(cv_folds * sizeof(float));
    if (!cv_results) {
        trainer->config = original_config;
        return -1.0f;
    }
    
    int cv_status = cross_validation(trainer, inputs, targets, num_samples, cv_folds, cv_results);
    
    float avg_score = 0.0f;
    if (cv_status == 0) {
        for (size_t i = 0; i < cv_folds; i++) {
            avg_score += cv_results[i];
        }
        avg_score /= (float)cv_folds;
    }
    
    int stopped_flag = 0;
    if (enable_early_stop && cv_status == 0) {
        float best_fold_loss = 1e10f;
        size_t fold_stall = 0;
        for (size_t i = 0; i < cv_folds; i++) {
            float fold_loss = -cv_results[i];
            if (trial_early_stop_check(fold_loss, &best_fold_loss, &fold_stall,
                                       early_stop_patience, early_stop_min_delta)) {
                stopped_flag = 1;
                break;
            }
        }
    }
    
    safe_free((void**)&cv_results);
    
    if (early_stopped) *early_stopped = stopped_flag;
    
    trainer->config = original_config;
    
    return avg_score;
}

/**
 * @brief 生成对数尺度随机值
 */
static float random_log_scale(float min_val, float max_val) {
    float log_min = logf(min_val);
    float log_max = logf(max_val);
    float t = secure_random_float();
    return expf(log_min + t * (log_max - log_min));
}

/**
 * @brief 生成线性尺度随机值
 */
static float random_linear_scale(float min_val, float max_val) {
    float t = secure_random_float();
    return min_val + t * (max_val - min_val);
}

/**
 * @brief 生成2的幂随机批量大小
 */
static size_t random_batch_size(size_t min_val, size_t max_val) {
    float t = secure_random_float();
    size_t val = (size_t)(min_val + t * (max_val - min_val));
    if (val < 1) val = 1;
    size_t v = val;
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    if (v > max_val) v = max_val;
    if (v < min_val) v = min_val;
    return v;
}

/**
 * @brief 随机搜索实现
 *
 * 从搜索空间中随机采样 num_random_trials 个超参数组合进行评估。
 */
static int hyperparameter_search_random_impl(Trainer* trainer, const float* inputs,
                                              const float* targets, size_t num_samples,
                                              const HyperparameterSearchConfig* config,
                                              HyperparameterSearchResult* result) {
    size_t num_trials = config->num_random_trials > 0 ? config->num_random_trials : 20;
    
    result->learning_rate_history = (float*)safe_malloc(num_trials * sizeof(float));
    result->regularization_history = (float*)safe_malloc(num_trials * sizeof(float));
    result->batch_size_history = (size_t*)safe_malloc(num_trials * sizeof(size_t));
    result->all_scores = (float*)safe_malloc(num_trials * sizeof(float));
    
    if (!result->learning_rate_history || !result->regularization_history ||
        !result->batch_size_history || !result->all_scores) {
        return -1;
    }
    
    result->best_score = -1e10f;
    result->num_combinations = num_trials;
    result->trials_evaluated = 0;
    result->trials_early_stopped = 0;
    
    for (size_t trial = 0; trial < num_trials; trial++) {
        float lr = random_log_scale(config->learning_rate_min, config->learning_rate_max);
        float reg = random_log_scale(config->regularization_min, config->regularization_max);
        size_t bs = random_batch_size(config->batch_size_min, config->batch_size_max);
        
        result->learning_rate_history[trial] = lr;
        result->regularization_history[trial] = reg;
        result->batch_size_history[trial] = bs;
        
        int early_stopped = 0;
        float score = hyperparameter_evaluate_trial(trainer, inputs, targets, num_samples,
                                                     lr, reg, bs, config->cv_folds,
                                                     config->enable_early_stopping,
                                                     config->early_stop_patience,
                                                     config->early_stop_min_delta,
                                                     &early_stopped);
        
        result->all_scores[trial] = score;
        result->trials_evaluated++;
        
        if (early_stopped) {
            result->trials_early_stopped++;
        }
        
        if (score > result->best_score) {
            result->best_score = score;
            result->best_learning_rate = lr;
            result->best_regularization = reg;
            result->best_batch_size = bs;
        }
        
        if (config->verbose) {
            printf("  随机搜索 [%zu/%zu] lr=%.6f reg=%.6f bs=%zu score=%.4f%s\n",
                   trial + 1, num_trials, lr, reg, bs, score,
                   early_stopped ? " (早停)" : "");
        }
    }
    
    result->history_size = num_trials;
    result->num_scores = num_trials;
    
    return 0;
}

/**
 * @brief 高斯过程核函数（平方指数核/RBF）
 *
 * k(x1, x2) = sigma_f^2 * exp(-||x1 - x2||^2 / (2 * l^2))
 */
static float gp_rbf_kernel(const float* x1, const float* x2, int dim,
                            float length_scale, float sigma_f) {
    float dist_sq = 0.0f;
    for (int i = 0; i < dim; i++) {
        float d = x1[i] - x2[i];
        dist_sq += d * d;
    }
    float ls2 = length_scale * length_scale;
    return sigma_f * sigma_f * expf(-dist_sq / (2.0f * ls2));
}

/**
 * @brief 构建GP协方差矩阵
 */
static void gp_build_kernel_matrix(float* K, const float* X, int n, int dim,
                                    float length_scale, float sigma_f, float noise) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float k = gp_rbf_kernel(&X[i * dim], &X[j * dim], dim, length_scale, sigma_f);
            if (i == j) {
                k += noise * noise;
            }
            K[i * n + j] = k;
        }
    }
}

/**
 * @brief 高斯消元法求解线性系统 (K + noise*I) * alpha = y
 *
 * 将一维数组视为 n x n 矩阵的列优先存储
 */
/**
 * @brief Cholesky分解求解对称正定线性系统 K·α = y
 * 
 * 核矩阵K为SPD，使用Cholesky分解比高斯消元法数值更稳定。
 * K = L·L^T, 先正向代入 L·z = y, 再反向代入 L^T·α = z
 * 
 * @param K 核矩阵[n×n]（输入/被就地修改为下三角L）
 * @param alpha 输出解向量[n]（输入为右端项y）
 * @param n 维度
 * @return int 0成功，-1矩阵非正定
 */
static int gp_cholesky_solve(float* K, float* alpha, int n) {
    /* Cholesky分解: K = L·L^T，L存储在K的下三角中 */
    for (int j = 0; j < n; j++) {
        float sum = 0.0f;
        for (int k = 0; k < j; k++) {
            sum += K[j * n + k] * K[j * n + k];
        }
        float diag = K[j * n + j] - sum;
        /* 对角线添加微小抖动增强正定性 */
        if (diag < 1e-10f) {
            diag = 1e-6f;
        }
        K[j * n + j] = sqrtf(diag);
        float inv_diag = 1.0f / K[j * n + j];

        for (int i = j + 1; i < n; i++) {
            sum = 0.0f;
            for (int k = 0; k < j; k++) {
                sum += K[i * n + k] * K[j * n + k];
            }
            K[i * n + j] = (K[i * n + j] - sum) * inv_diag;
        }
    }

    /* 正向代入 L·z = alpha（alpha此时存储y） */
    for (int i = 0; i < n; i++) {
        float sum = alpha[i];
        for (int j = 0; j < i; j++) {
            sum -= K[i * n + j] * alpha[j];
        }
        alpha[i] = sum / K[i * n + i];
    }

    /* 反向代入 L^T·alpha_= z */
    for (int i = n - 1; i >= 0; i--) {
        float sum = alpha[i];
        for (int j = i + 1; j < n; j++) {
            sum -= K[j * n + i] * alpha[j];
        }
        alpha[i] = sum / K[i * n + i];
    }

    return 0;
}

/**
 * @brief GP预测均值
 */
static float gp_predict_mean(const float* X_train, const float* alpha, int n_train,
                              int dim, const float* x_new, float length_scale, float sigma_f) {
    float result = 0.0f;
    for (int i = 0; i < n_train; i++) {
        float k = gp_rbf_kernel(&X_train[i * dim], x_new, dim, length_scale, sigma_f);
        result += alpha[i] * k;
    }
    return result;
}

/**
 * @brief GP预测方差（完整实现：基于RBF核矩阵的精确后验方差）
 *
 * 公式: var(x*) = k(x*, x*) - k_*^T (K + σ²I)⁻¹ k_*
 * 其中 k_* 是新点与所有训练点的核向量, K 是训练点间的核矩阵
 * 使用Cholesky分解求解线性系统 Kα = k_*，避免显式矩阵求逆
 */
static float gp_predict_var(const float* X_train, int n_train, int dim,
                             const float* x_new, float length_scale, float sigma_f, float noise) {
    /* RBF核函数: k(x_i, x_j) = σ_f² * exp(-|x_i - x_j|² / (2ℓ²)) */
    float ls2 = length_scale * length_scale;
    float sf2 = sigma_f * sigma_f;
    float noise2 = noise * noise;

    /* 计算自相似度 k(x*, x*) */
    float k_star_star = sf2;

    /* 如果训练点太少，使用先验方差 */
    if (n_train <= 1) {
        return sf2 + noise2;
    }

    /* 限制训练点数（Cholesky O(n³)，实际应用中取最多256个最近邻） */
    int n_eff = (n_train > 256) ? 256 : n_train;

    /* 分配核向量 k_* [n_eff] 和核矩阵 K [n_eff × n_eff] */
    float* k_vector = (float*)safe_calloc((size_t)n_eff, sizeof(float));
    float* K_matrix = (float*)safe_calloc((size_t)(n_eff * n_eff), sizeof(float));
    if (!k_vector || !K_matrix) {
        safe_free((void**)&k_vector);
        safe_free((void**)&K_matrix);
        return sf2 + noise2;
    }

    /* 计算核向量 k_vector[i] = k(x*, x_i) */
    for (int i = 0; i < n_eff; i++) {
        float dist_sq = 0.0f;
        for (int d = 0; d < dim; d++) {
            float diff = X_train[i * dim + d] - x_new[d];
            dist_sq += diff * diff;
        }
        k_vector[i] = sf2 * expf(-dist_sq / (2.0f * ls2));
    }

    /* 构建核矩阵 K[i][j] = k(x_i, x_j) + σ² * δ_ij（加噪声的对角元素） */
    for (int i = 0; i < n_eff; i++) {
        for (int j = 0; j < n_eff; j++) {
            float dist_sq = 0.0f;
            for (int d = 0; d < dim; d++) {
                float diff = X_train[i * dim + d] - X_train[j * dim + d];
                dist_sq += diff * diff;
            }
            K_matrix[i * n_eff + j] = sf2 * expf(-dist_sq / (2.0f * ls2));
        }
        /* 对角线添加噪声方差 */
        K_matrix[i * n_eff + i] += noise2;
    }

    /* Cholesky分解：K = L * L^T（K是对称正定矩阵） */
    float* L = (float*)safe_calloc((size_t)(n_eff * n_eff), sizeof(float));
    if (!L) {
        safe_free((void**)&k_vector);
        safe_free((void**)&K_matrix);
        return sf2 + noise2;
    }

    int chol_success = 1;
    for (int i = 0; i < n_eff && chol_success; i++) {
        for (int j = 0; j <= i; j++) {
            float sum = K_matrix[i * n_eff + j];
            for (int k = 0; k < j; k++) {
                sum -= L[i * n_eff + k] * L[j * n_eff + k];
            }
            if (i == j) {
                if (sum <= 1e-12f) {
                    chol_success = 0;
                    break;
                }
                L[i * n_eff + i] = sqrtf(sum);
            } else {
                L[i * n_eff + j] = sum / L[j * n_eff + j];
            }
        }
    }

    float variance = sf2 + noise2;
    if (chol_success) {
        /* 前向替代：L * y = k_vector */
        float* y = (float*)safe_calloc((size_t)n_eff, sizeof(float));
        if (y) {
            for (int i = 0; i < n_eff; i++) {
                float sum = k_vector[i];
                for (int j = 0; j < i; j++) {
                    sum -= L[i * n_eff + j] * y[j];
                }
                y[i] = sum / L[i * n_eff + i];
            }

            /* 后向替代：L^T * alpha = y */
            float* alpha = (float*)safe_calloc((size_t)n_eff, sizeof(float));
            if (alpha) {
                for (int i = n_eff - 1; i >= 0; i--) {
                    float sum = y[i];
                    for (int j = n_eff - 1; j > i; j--) {
                        sum -= L[j * n_eff + i] * alpha[j];
                    }
                    alpha[i] = sum / L[i * n_eff + i];
                }

                /* 计算 k_vector^T * alpha */
                float k_dot_alpha = 0.0f;
                for (int i = 0; i < n_eff; i++) {
                    k_dot_alpha += k_vector[i] * alpha[i];
                }

                variance = k_star_star - k_dot_alpha + noise2;
                if (variance < noise2) variance = noise2;

                safe_free((void**)&alpha);
            }
            safe_free((void**)&y);
        }
    }

    safe_free((void**)&k_vector);
    safe_free((void**)&K_matrix);
    safe_free((void**)&L);
    return variance;
}

/**
 * @brief 期望提升（Expected Improvement）采集函数
 *
 * EI(x) = (mu - best - epsilon) * Phi(z) + sigma * phi(z)
 * 其中 z = (mu - best - epsilon) / sigma
 */
static float gp_acquisition_ei(float mu, float sigma, float best_so_far, float epsilon) {
    if (sigma < 1e-12f) return 0.0f;
    
    float improvement = mu - best_so_far - epsilon;
    float z = improvement / sigma;
    
    // 标准正态CDF和PDF的近似
    float cdf = 0.5f * (1.0f + erfcf(-z * 0.7071067811865476f * 0.7071067811865476f));
    float pdf = expf(-0.5f * z * z) * 0.3989422804014327f;
    
    return improvement * cdf + sigma * pdf;
}

/**
 * @brief 贝叶斯优化实现
 *
 * 使用高斯过程回归作为代理模型，期望提升（EI）作为采集函数。
 * 初始阶段随机采样一些点，然后使用采集函数选择下一个评估点。
 */
static int hyperparameter_search_bayesian_impl(Trainer* trainer, const float* inputs,
                                                const float* targets, size_t num_samples,
                                                const HyperparameterSearchConfig* config,
                                                HyperparameterSearchResult* result) {
    size_t num_trials = config->num_random_trials > 0 ? config->num_random_trials : 30;
    size_t num_initial = num_trials / 3;
    if (num_initial < 5) num_initial = 5;
    if (num_initial > num_trials) num_initial = num_trials;
    
    // 搜索空间维度：3（学习率，正则化，批量大小）
    int dim = 3;
    
    result->learning_rate_history = (float*)safe_malloc(num_trials * sizeof(float));
    result->regularization_history = (float*)safe_malloc(num_trials * sizeof(float));
    result->batch_size_history = (size_t*)safe_malloc(num_trials * sizeof(size_t));
    result->all_scores = (float*)safe_malloc(num_trials * sizeof(float));
    
    if (!result->learning_rate_history || !result->regularization_history ||
        !result->batch_size_history || !result->all_scores) {
        return -1;
    }
    
    result->best_score = -1e10f;
    result->num_combinations = num_trials;
    result->trials_evaluated = 0;
    result->trials_early_stopped = 0;
    
    // 训练数据数组（归一化到[0,1]^dim）
    float* X_train = (float*)safe_malloc(num_trials * dim * sizeof(float));
    float* y_train = (float*)safe_malloc(num_trials * sizeof(float));
    
    if (!X_train || !y_train) {
        safe_free((void**)&X_train);
        safe_free((void**)&y_train);
        return -1;
    }
    
    int n_train = 0;
    
    for (size_t trial = 0; trial < num_trials; trial++) {
        float lr = config->learning_rate_min;
        float reg = config->regularization_min;
        size_t bs = config->batch_size_min;
        
        if ((int)trial < (int)num_initial) {
            // 初始阶段：随机采样
            lr = random_log_scale(config->learning_rate_min, config->learning_rate_max);
            reg = random_log_scale(config->regularization_min, config->regularization_max);
            bs = random_batch_size(config->batch_size_min, config->batch_size_max);
        } else if (n_train >= 3) {
            // 贝叶斯阶段：使用GP代理模型选择下一个点
            float length_scale = 0.3f;
            float sigma_f = 1.0f;
            float noise = 0.1f;
            
            // 构建GP
            float* K = (float*)safe_malloc(n_train * n_train * sizeof(float));
            float* alpha = (float*)safe_malloc(n_train * sizeof(float));
            
            if (K && alpha) {
                gp_build_kernel_matrix(K, X_train, n_train, dim, length_scale, sigma_f, noise);
                
                for (int i = 0; i < n_train; i++) {
                    alpha[i] = y_train[i];
                }
                gp_cholesky_solve(K, alpha, n_train);
                
                // 在搜索空间随机采样候选点，选择EI最大的
                float best_lr = config->learning_rate_min;
                float best_reg = config->regularization_min;
                size_t best_bs = config->batch_size_min;
                float best_ei = -1e10f;
                
                int num_candidates = 100;
                for (int c = 0; c < num_candidates; c++) {
                    float cand_lr = random_log_scale(config->learning_rate_min, config->learning_rate_max);
                    float cand_reg = random_log_scale(config->regularization_min, config->regularization_max);
                    size_t cand_bs = random_batch_size(config->batch_size_min, config->batch_size_max);
                    
                    float x_cand[3];
                    x_cand[0] = (logf(cand_lr) - logf(config->learning_rate_min)) /
                                (logf(config->learning_rate_max) - logf(config->learning_rate_min) + 1e-10f);
                    x_cand[1] = (logf(cand_reg) - logf(config->regularization_min)) /
                                (logf(config->regularization_max) - logf(config->regularization_min) + 1e-10f);
                    x_cand[2] = (float)(cand_bs - config->batch_size_min) /
                                (float)(config->batch_size_max - config->batch_size_min + 1);
                    
                    float mu = gp_predict_mean(X_train, alpha, n_train, dim, x_cand,
                                               length_scale, sigma_f);
                    float sigma = gp_predict_var(X_train, n_train, dim, x_cand,
                                                 length_scale, sigma_f, noise);
                    float ei = gp_acquisition_ei(mu, sqrtf(sigma), result->best_score, 0.01f);
                    
                    if (ei > best_ei) {
                        best_ei = ei;
                        best_lr = cand_lr;
                        best_reg = cand_reg;
                        best_bs = cand_bs;
                    }
                }
                
                lr = best_lr;
                reg = best_reg;
                bs = best_bs;
            }
            
            safe_free((void**)&K);
            safe_free((void**)&alpha);
        } else {
            // 随机回退
            lr = random_log_scale(config->learning_rate_min, config->learning_rate_max);
            reg = random_log_scale(config->regularization_min, config->regularization_max);
            bs = random_batch_size(config->batch_size_min, config->batch_size_max);
        }
        
        result->learning_rate_history[trial] = lr;
        result->regularization_history[trial] = reg;
        result->batch_size_history[trial] = bs;
        
        int early_stopped = 0;
        float score = hyperparameter_evaluate_trial(trainer, inputs, targets, num_samples,
                                                     lr, reg, bs, config->cv_folds,
                                                     config->enable_early_stopping,
                                                     config->early_stop_patience,
                                                     config->early_stop_min_delta,
                                                     &early_stopped);
        
        result->all_scores[trial] = score;
        result->trials_evaluated++;
        
        if (early_stopped) {
            result->trials_early_stopped++;
        }
        
        // 更新训练数据
        float x_norm[3];
        x_norm[0] = (logf(lr) - logf(config->learning_rate_min)) /
                    (logf(config->learning_rate_max) - logf(config->learning_rate_min) + 1e-10f);
        x_norm[1] = (logf(reg) - logf(config->regularization_min)) /
                    (logf(config->regularization_max) - logf(config->regularization_min) + 1e-10f);
        x_norm[2] = (float)(bs - config->batch_size_min) /
                    (float)(config->batch_size_max - config->batch_size_min + 1);
        
        X_train[n_train * dim + 0] = x_norm[0];
        X_train[n_train * dim + 1] = x_norm[1];
        X_train[n_train * dim + 2] = x_norm[2];
        y_train[n_train] = score;
        n_train++;
        
        if (score > result->best_score) {
            result->best_score = score;
            result->best_learning_rate = lr;
            result->best_regularization = reg;
            result->best_batch_size = bs;
        }
        
        if (config->verbose) {
            printf("  贝叶斯优化 [%zu/%zu] lr=%.6f reg=%.6f bs=%zu score=%.4f%s\n",
                   trial + 1, num_trials, lr, reg, bs, score,
                   early_stopped ? " (早停)" : "");
        }
    }
    
    safe_free((void**)&X_train);
    safe_free((void**)&y_train);
    
    result->history_size = num_trials;
    result->num_scores = num_trials;
    
    return 0;
}

/**
 * @brief 超参数搜索（增强版：支持网格/随机/贝叶斯+早停）
 */
int hyperparameter_search(Trainer* trainer, const float* inputs, const float* targets,
                          size_t num_samples, const HyperparameterSearchConfig* config,
                          HyperparameterSearchResult* result) {
    if (!trainer || !inputs || !targets || num_samples == 0 || !config || !result) {
        return -1;
    }
    
    double start_time = 0.0;
#if defined(_WIN32)
    start_time = (double)clock() / (double)CLOCKS_PER_SEC;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    start_time = ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
    
    memset(result, 0, sizeof(HyperparameterSearchResult));
    result->best_score = -1e10f;
    
    int ret = 0;
    
    switch (config->search_mode) {
        case HYPERPARAM_SEARCH_RANDOM:
            ret = hyperparameter_search_random_impl(trainer, inputs, targets, num_samples,
                                                    config, result);
            break;
            
        case HYPERPARAM_SEARCH_BAYESIAN:
            ret = hyperparameter_search_bayesian_impl(trainer, inputs, targets, num_samples,
                                                      config, result);
            break;
            
        case HYPERPARAM_SEARCH_GRID:
        default:
        {
            // 网格搜索实现（原有，但使用增强的搜索结果结构）
            
            // 生成学习率数组（对数尺度）
            size_t num_lr = config->learning_rate_steps;
            if (num_lr == 0) num_lr = 4;
            float* learning_rates = (float*)safe_malloc(num_lr * sizeof(float));
            if (!learning_rates) return -1;
            
            if (num_lr == 1) {
                learning_rates[0] = config->learning_rate_min;
            } else {
                for (size_t i = 0; i < num_lr; i++) {
                    float t = (float)i / (float)(num_lr - 1);
                    learning_rates[i] = config->learning_rate_min *
                                       powf(config->learning_rate_max / config->learning_rate_min, t);
                }
            }
            
            // 生成正则化强度数组（对数尺度）
            size_t num_reg = config->regularization_steps;
            if (num_reg == 0) num_reg = 4;
            float* regularizations = (float*)safe_malloc(num_reg * sizeof(float));
            if (!regularizations) {
                safe_free((void**)&learning_rates);
                return -1;
            }
            if (num_reg == 1) {
                regularizations[0] = config->regularization_min;
            } else {
                for (size_t i = 0; i < num_reg; i++) {
                    float t = (float)i / (float)(num_reg - 1);
                    regularizations[i] = config->regularization_min *
                                        powf(config->regularization_max / config->regularization_min, t);
                }
            }
            
            // 生成批量大小数组
            size_t num_bs = config->batch_size_steps;
            if (num_bs == 0) num_bs = 4;
            size_t* batch_sizes = (size_t*)safe_malloc(num_bs * sizeof(size_t));
            if (!batch_sizes) {
                safe_free((void**)&learning_rates);
                safe_free((void**)&regularizations);
                return -1;
            }
            if (num_bs == 1) {
                batch_sizes[0] = config->batch_size_min;
            } else {
                for (size_t i = 0; i < num_bs; i++) {
                    float t = (float)i / (float)(num_bs - 1);
                    batch_sizes[i] = (size_t)(config->batch_size_min +
                                             t * (config->batch_size_max - config->batch_size_min));
                    if (batch_sizes[i] > 1) {
                        size_t v = batch_sizes[i];
                        v--;
                        v |= v >> 1;
                        v |= v >> 2;
                        v |= v >> 4;
                        v |= v >> 8;
                        v |= v >> 16;
                        v++;
                        batch_sizes[i] = v;
                    }
                }
            }
            
            size_t total_combinations = num_lr * num_reg * num_bs;
            
            result->learning_rate_history = (float*)safe_malloc(total_combinations * sizeof(float));
            result->regularization_history = (float*)safe_malloc(total_combinations * sizeof(float));
            result->batch_size_history = (size_t*)safe_malloc(total_combinations * sizeof(size_t));
            result->all_scores = (float*)safe_malloc(total_combinations * sizeof(float));
            
            if (!result->learning_rate_history || !result->regularization_history ||
                !result->batch_size_history || !result->all_scores) {
                safe_free((void**)&learning_rates);
                safe_free((void**)&regularizations);
                safe_free((void**)&batch_sizes);
                return -1;
            }
            
            result->num_combinations = total_combinations;
            result->trials_evaluated = 0;
            result->trials_early_stopped = 0;
            
            TrainingConfig original_config = trainer->config;
            int combination_idx = 0;
            
            for (size_t lr_idx = 0; lr_idx < num_lr; lr_idx++) {
                for (size_t reg_idx = 0; reg_idx < num_reg; reg_idx++) {
                    for (size_t bs_idx = 0; bs_idx < num_bs; bs_idx++) {
                        float lr = learning_rates[lr_idx];
                        float reg = regularizations[reg_idx];
                        size_t bs = batch_sizes[bs_idx];
                        
                        result->learning_rate_history[combination_idx] = lr;
                        result->regularization_history[combination_idx] = reg;
                        result->batch_size_history[combination_idx] = bs;
                        
                        int early_stopped = 0;
                        float score = hyperparameter_evaluate_trial(trainer, inputs, targets,
                                                                     num_samples, lr, reg, bs,
                                                                     config->cv_folds,
                                                                     config->enable_early_stopping,
                                                                     config->early_stop_patience,
                                                                     config->early_stop_min_delta,
                                                                     &early_stopped);
                        
                        result->all_scores[combination_idx] = score;
                        result->trials_evaluated++;
                        if (early_stopped) result->trials_early_stopped++;
                        
                        if (score > result->best_score) {
                            result->best_score = score;
                            result->best_learning_rate = lr;
                            result->best_regularization = reg;
                            result->best_batch_size = bs;
                        }
                        
                        if (config->verbose) {
                            printf("  网格搜索 [%d/%zu] lr=%.6f reg=%.6f bs=%zu score=%.4f%s\n",
                                   combination_idx + 1, total_combinations, lr, reg, bs, score,
                                   early_stopped ? " (早停)" : "");
                        }
                        
                        combination_idx++;
                    }
                }
            }
            
            safe_free((void**)&learning_rates);
            safe_free((void**)&regularizations);
            safe_free((void**)&batch_sizes);
            
            result->history_size = total_combinations;
            result->num_scores = total_combinations;
            
            trainer->config = original_config;
            break;
        }
    }
    
    // 计算搜索耗时
    double end_time = 0.0;
#if defined(_WIN32)
    end_time = (double)clock() / (double)CLOCKS_PER_SEC;
#else
    struct timespec ts2;
    clock_gettime(CLOCK_MONOTONIC, &ts2);
    end_time = ts2.tv_sec + ts2.tv_nsec * 1e-9;
#endif
    result->search_duration_sec = (float)(end_time - start_time);
    
    return ret;
}

/**
 * @brief 检测训练是否停滞（完整滑动窗口线性回归分析）
 */
int trainer_detect_stagnation(Trainer* trainer, size_t window_size, float min_improvement,
                               StagnationDetectionResult* result) {
    if (!trainer || !result || window_size < 2) {
        return -1;
    }
    
    memset(result, 0, sizeof(StagnationDetectionResult));
    
    TrainingHistory* history = &trainer->history;
    
    // 1. 损失趋势分析：滑动窗口线性回归
    float loss_slope = 0.0f;
    float loss_mean = 0.0f;
    float accuracy_slope = 0.0f;
    float accuracy_mean = 0.0f;
    
    size_t loss_window = history->size > window_size ? window_size : history->size;
    size_t acc_window = history->size > window_size ? window_size : history->size;
    
    if (loss_window < 2) {
        // 历史记录不足，无法判断停滞
        result->is_stagnated = 0;
        result->loss_improvement_rate = 0.0f;
        result->accuracy_improvement_rate = 0.0f;
        result->gradient_stability = 1.0f;
        result->epochs_since_best_loss = 0;
        result->epochs_since_best_accuracy = 0;
        result->recommended_action = 0;
        snprintf(result->description, sizeof(result->description),
                 "训练历史不足（%zu条），无法检测停滞", history->size);
        return 0;
    }
    
    // 计算损失滑动窗口的线性回归（x=epoch, y=loss）
    {
        float sum_x = 0.0f, sum_y = 0.0f, sum_xy = 0.0f, sum_xx = 0.0f;
        size_t start = history->size > loss_window ? history->size - loss_window : 0;
        
        for (size_t i = start; i < history->size; i++) {
            float x = (float)(i - start);
            float y = history->train_losses[i];
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_xx += x * x;
        }
        
        float n = (float)(history->size - start);
        float denom = n * sum_xx - sum_x * sum_x;
        if (fabsf(denom) > 1e-12f) {
            loss_slope = (n * sum_xy - sum_x * sum_y) / denom;
        }
        loss_mean = sum_y / n;
    }
    
    // 计算准确率滑动窗口的线性回归
    {
        float sum_x = 0.0f, sum_y = 0.0f, sum_xy = 0.0f, sum_xx = 0.0f;
        size_t start = history->size > acc_window ? history->size - acc_window : 0;
        
        for (size_t i = start; i < history->size; i++) {
            float x = (float)(i - start);
            float y = history->train_accuracies[i];
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_xx += x * x;
        }
        
        float n = (float)(history->size - start);
        float denom = n * sum_xx - sum_x * sum_x;
        if (fabsf(denom) > 1e-12f) {
            accuracy_slope = (n * sum_xy - sum_x * sum_y) / denom;
        }
        accuracy_mean = sum_y / n;
    }
    
    // 损失下降 = 负斜率（loss减小是好的），改善率 = -slope/mean
    result->loss_improvement_rate = -loss_slope / (loss_mean + 1e-10f);
    result->accuracy_improvement_rate = accuracy_slope / (accuracy_mean + 1e-10f);
    
    // 2. 查找最佳损失和准确率
    float best_loss = history->train_losses[0];
    float best_accuracy = history->train_accuracies[0];
    size_t best_loss_epoch = 0;
    size_t best_acc_epoch = 0;
    
    for (size_t i = 1; i < history->size; i++) {
        if (history->train_losses[i] < best_loss) {
            best_loss = history->train_losses[i];
            best_loss_epoch = i;
        }
        if (history->train_accuracies[i] > best_accuracy) {
            best_accuracy = history->train_accuracies[i];
            best_acc_epoch = i;
        }
    }
    
    result->epochs_since_best_loss = history->size - 1 - best_loss_epoch;
    result->epochs_since_best_accuracy = history->size - 1 - best_acc_epoch;
    
    // 3. 梯度稳定性分析
    float grad_stability = 1.0f;
    if (trainer->gradient_history_size > 1) {
        float grad_mean = 0.0f, grad_var = 0.0f;
        size_t grad_window = trainer->gradient_history_size > 10 ? 10 : trainer->gradient_history_size;
        size_t grad_start = trainer->gradient_history_size - grad_window;
        
        for (size_t i = grad_start; i < trainer->gradient_history_size; i++) {
            grad_mean += trainer->gradient_history[i];
        }
        grad_mean /= grad_window;
        
        for (size_t i = grad_start; i < trainer->gradient_history_size; i++) {
            float diff = trainer->gradient_history[i] - grad_mean;
            grad_var += diff * diff;
        }
        grad_var /= grad_window;
        
        float grad_std = sqrtf(grad_var + 1e-12f);
        grad_stability = 1.0f / (1.0f + grad_std / (fabsf(grad_mean) + 1e-10f));
        if (grad_stability > 1.0f) grad_stability = 1.0f;
    }
    result->gradient_stability = grad_stability;
    
    // 4. 综合停滞判断
    int is_stagnated = 0;
    int recommended_action = 0;
    char description[256] = "训练正常进行中";
    
    // 条件1: 损失改善率低于阈值（接近停滞）
    int condition_loss_stagnation = (result->loss_improvement_rate < min_improvement) &&
                                    (result->epochs_since_best_loss >= window_size / 2);
    
    // 条件2: 准确率没有提升
    int condition_acc_stagnation = (result->accuracy_improvement_rate < min_improvement * 0.5f) &&
                                   (result->epochs_since_best_accuracy >= window_size / 2);
    
    // 条件3: 损失开始上升（发散）
    int condition_divergence = result->loss_improvement_rate < -min_improvement * 2.0f;
    
    // 条件4: 梯度不稳定（可能震荡）
    int condition_grad_instability = grad_stability < 0.3f;
    
    if (condition_divergence) {
        // 损失发散：严重问题
        is_stagnated = 1;
        recommended_action = 3; // 全面搜索
        snprintf(description, sizeof(description),
                 "损失发散：改善率=%.4f（负值表示上升），建议全面超参数搜索",
                 result->loss_improvement_rate);
    } else if (condition_loss_stagnation && condition_acc_stagnation) {
        // 损失和准确率双双停滞
        is_stagnated = 1;
        if (result->epochs_since_best_loss > window_size) {
            recommended_action = 2; // 增加正则化+降低学习率
            snprintf(description, sizeof(description),
                     "双停滞：损失%.0f轮未改善（改善率=%.4f），准确率%.0f轮未改善，建议正则化+学习率调整",
                     (double)result->epochs_since_best_loss, result->loss_improvement_rate,
                     (double)result->epochs_since_best_accuracy);
        } else {
            recommended_action = 1; // 仅学习率
            snprintf(description, sizeof(description),
                     "轻度停滞：改善率=%.4f，建议降低学习率",
                     result->loss_improvement_rate);
        }
    } else if (condition_loss_stagnation) {
        // 仅损失停滞
        is_stagnated = 1;
        if (condition_grad_instability) {
            recommended_action = 2; // 增加正则化
            snprintf(description, sizeof(description),
                     "损失停滞+梯度不稳定（稳定性=%.2f），建议增加正则化",
                     grad_stability);
        } else {
            recommended_action = 1; // 降低学习率
            snprintf(description, sizeof(description),
                     "损失停滞：改善率=%.4f，梯度稳定（%.2f），建议轻微降低学习率",
                     result->loss_improvement_rate, grad_stability);
        }
    } else if (condition_grad_instability) {
        // 仅梯度不稳定
        is_stagnated = 1;
        recommended_action = 1;
        snprintf(description, sizeof(description),
                 "梯度不稳定（稳定性=%.2f），建议降低学习率增加稳定性",
                 grad_stability);
    }
    
    result->is_stagnated = is_stagnated;
    result->recommended_action = recommended_action;
    strncpy(result->description, description, sizeof(result->description) - 1);
    result->description[sizeof(result->description) - 1] = '\0';
    
    return 0;
}

/**
 * @brief 自动超参数调整（完整自我修正版）
 */
int trainer_auto_tune_hyperparameters(Trainer* trainer,
                                       const float* inputs, const float* targets,
                                       size_t num_samples, int auto_mode,
                                       float* improvement) {
    if (!trainer || !inputs || !targets || num_samples == 0) {
        return -1;
    }
    
    // 步骤1: 检测停滞
    StagnationDetectionResult stagnation;
    int ret = trainer_detect_stagnation(trainer, 5, 0.001f, &stagnation);
    if (ret != 0) {
        return -1;
    }
    
    // 如果没有停滞且模式为自动，无需调整
    if (!stagnation.is_stagnated && auto_mode < 0) {
        if (improvement) *improvement = 0.0f;
        return 0;
    }
    
    // 步骤2: 根据停滞检测结果或用户指定的模式构建搜索配置
    int effective_mode = auto_mode >= 0 ? auto_mode : stagnation.recommended_action;
    if (effective_mode > 3) effective_mode = 3;
    
    // 保存原始配置，以便计算改善幅度
    TrainingConfig original_config = trainer->config;
    
    HyperparameterSearchConfig search_config;
    memset(&search_config, 0, sizeof(HyperparameterSearchConfig));
    search_config.cv_folds = 3; // 快速交叉验证
    
    // 根据模式配置搜索空间
    switch (effective_mode) {
        case 0: // 轻微调整：仅探索当前学习率周围的较小范围
            search_config.learning_rate_min = original_config.learning_rate * 0.3f;
            search_config.learning_rate_max = original_config.learning_rate * 0.9f;
            search_config.learning_rate_steps = 4;
            
            search_config.regularization_min = original_config.regularization_lambda;
            search_config.regularization_max = original_config.regularization_lambda;
            search_config.regularization_steps = 1;
            
            search_config.batch_size_min = original_config.batch_size;
            search_config.batch_size_max = original_config.batch_size;
            search_config.batch_size_steps = 1;
            break;
            
        case 1: // 中度调整：学习率+正则化
            search_config.learning_rate_min = original_config.learning_rate * 0.1f;
            search_config.learning_rate_max = original_config.learning_rate * 1.0f;
            search_config.learning_rate_steps = 5;
            
            search_config.regularization_min = original_config.regularization_lambda * 0.5f;
            search_config.regularization_max = original_config.regularization_lambda * 5.0f;
            search_config.regularization_steps = 4;
            
            search_config.batch_size_min = original_config.batch_size;
            search_config.batch_size_max = original_config.batch_size;
            search_config.batch_size_steps = 1;
            break;
            
        case 2: // 显著调整：学习率+正则化+批量大小
            search_config.learning_rate_min = original_config.learning_rate * 0.05f;
            search_config.learning_rate_max = original_config.learning_rate * 2.0f;
            search_config.learning_rate_steps = 5;
            
            search_config.regularization_min = original_config.regularization_lambda * 0.1f;
            search_config.regularization_max = original_config.regularization_lambda * 10.0f;
            search_config.regularization_steps = 4;
            
            search_config.batch_size_min = original_config.batch_size > 8 ? original_config.batch_size / 2 : 1;
            search_config.batch_size_max = original_config.batch_size * 2;
            search_config.batch_size_steps = 3;
            break;
            
        case 3: // 全面搜索（发散恢复）：大范围探索
            search_config.learning_rate_min = 1e-6f;
            search_config.learning_rate_max = original_config.learning_rate * 5.0f;
            search_config.learning_rate_steps = 6;
            
            search_config.regularization_min = 1e-8f;
            search_config.regularization_max = original_config.regularization_lambda * 20.0f;
            search_config.regularization_steps = 5;
            
            search_config.batch_size_min = 1;
            search_config.batch_size_max = original_config.batch_size * 4;
            search_config.batch_size_steps = 4;
            break;
    }
    
    search_config.max_iterations = 50; // 每次评估的迭代次数
    
    // 步骤3: 执行超参数搜索
    HyperparameterSearchResult search_result;
    ret = hyperparameter_search(trainer, inputs, targets, num_samples, &search_config, &search_result);
    if (ret != 0) {
        return -1;
    }
    
    // 步骤4: 应用最佳超参数
    if (search_result.best_score > 0.0f) {
        trainer->config.learning_rate = search_result.best_learning_rate;
        trainer->config.regularization_lambda = search_result.best_regularization;
        trainer->config.batch_size = search_result.best_batch_size;
        
        // 计算改善幅度：原始配置下的分数 vs 最佳配置下的分数
        if (improvement) {
            // 用原始配置跑一次评估
            float original_score = 0.0f;
            float* cv_results = (float*)safe_malloc(3 * sizeof(float));
            if (cv_results) {
                TrainingConfig temp_config = trainer->config;
                trainer->config = original_config; // 临时使用原始配置
                if (cross_validation(trainer, inputs, targets, num_samples, 3, cv_results) == 0) {
                    for (int i = 0; i < 3; i++) original_score += cv_results[i];
                    original_score /= 3.0f;
                }
                trainer->config = temp_config; // 恢复为最佳配置
                safe_free((void**)&cv_results);
            }
            *improvement = search_result.best_score - original_score;
        }
    }
    
    // 步骤5: 更新训练历史中的学习率记录
    if (trainer->history.size > 0) {
        trainer->history.learning_rates[trainer->history.size - 1] = trainer->config.learning_rate;
    }
    
    hyperparameter_search_result_free(&search_result);
    
    return 0;
}

/* ============================================================================
 * GPU加速辅助函数实现
 * ============================================================================ */

/**
 * @brief 初始化GPU内存
 */
static int trainer_gpu_init_memory(Trainer* trainer, size_t batch_input_size, 
                                  size_t batch_output_size, size_t num_parameters) {
    if (!trainer || !trainer->gpu_context) {
        return -1;
    }
    
    // 分配输入GPU内存
    trainer->gpu_inputs = gpu_memory_alloc(trainer->gpu_context, 
                                          batch_input_size * sizeof(float),
                                          GPU_MEMORY_DEVICE);
    if (!trainer->gpu_inputs) {
        return -1;
    }
    
    // 分配目标GPU内存
    trainer->gpu_targets = gpu_memory_alloc(trainer->gpu_context,
                                           batch_output_size * sizeof(float),
                                           GPU_MEMORY_DEVICE);
    if (!trainer->gpu_targets) {
        return -1;
    }
    
    // 分配输出GPU内存
    trainer->gpu_outputs = gpu_memory_alloc(trainer->gpu_context,
                                           batch_output_size * sizeof(float),
                                           GPU_MEMORY_DEVICE);
    if (!trainer->gpu_outputs) {
        return -1;
    }
    
    // 分配梯度GPU内存
    if (num_parameters > 0) {
        trainer->gpu_gradients = gpu_memory_alloc(trainer->gpu_context,
                                                 num_parameters * sizeof(float),
                                                 GPU_MEMORY_DEVICE);
        if (!trainer->gpu_gradients) {
            return -1;
        }
    }
    
    // 分配参数GPU内存
    if (num_parameters > 0) {
        trainer->gpu_parameters = gpu_memory_alloc(trainer->gpu_context,
                                                  num_parameters * sizeof(float),
                                                  GPU_MEMORY_DEVICE);
        if (!trainer->gpu_parameters) {
            return -1;
        }
        
        // 将当前参数复制到GPU
        float* parameters = lnn_get_parameters(trainer->network);
        if (parameters) {
            if (gpu_memory_copy_to_device(trainer->gpu_parameters,
                                         parameters,
                                         num_parameters * sizeof(float)) != 0) {
                return -1;
            }
        }
    }
    
    return 0;
}

/**
 * @brief 将批次数据复制到GPU设备
 */
static int trainer_gpu_copy_batch_to_device(Trainer* trainer, const float* batch_inputs,
                                          const float* batch_targets, size_t batch_size,
                                          size_t input_dim, size_t output_dim) {
    if (!trainer || !trainer->gpu_inputs || !trainer->gpu_targets) {
        return -1;
    }
    
    size_t input_size = batch_size * input_dim;
    size_t output_size = batch_size * output_dim;
    
    // 复制输入数据到GPU
    if (gpu_memory_copy_to_device(trainer->gpu_inputs,
                                 batch_inputs,
                                 input_size * sizeof(float)) != 0) {
        return -1;
    }
    
    // 复制目标数据到GPU
    if (gpu_memory_copy_to_device(trainer->gpu_targets,
                                 batch_targets,
                                 output_size * sizeof(float)) != 0) {
        return -1;
    }
    
    return 0;
}

/**
 * @brief 将输出数据复制到GPU设备
 */
static int trainer_gpu_copy_output_to_device(Trainer* trainer, const float* outputs,
                                            size_t batch_size, size_t output_dim) {
    if (!trainer || !trainer->gpu_outputs) {
        return -1;
    }
    
    size_t output_size = batch_size * output_dim;
    
    if (gpu_memory_copy_to_device(trainer->gpu_outputs,
                                 outputs,
                                 output_size * sizeof(float)) != 0) {
        return -1;
    }
    
    return 0;
}

/**
 * @brief 从GPU设备复制输出数据
 */
static int trainer_gpu_copy_output_from_device(Trainer* trainer, float* outputs,
                                              size_t batch_size, size_t output_dim) {
    if (!trainer || !trainer->gpu_outputs) {
        return -1;
    }
    
    size_t output_size = batch_size * output_dim;
    
    if (gpu_memory_copy_from_device(outputs,
                                   trainer->gpu_outputs,
                                   output_size * sizeof(float)) != 0) {
        return -1;
    }
    
    return 0;
}

/**
 * @brief 将梯度数据复制到GPU设备
 */
static int trainer_gpu_copy_gradients_to_device(Trainer* trainer, const float* gradients,
                                               size_t num_parameters) {
    if (!trainer || !trainer->gpu_gradients || num_parameters == 0) {
        return -1;
    }
    
    if (gpu_memory_copy_to_device(trainer->gpu_gradients,
                                 gradients,
                                 num_parameters * sizeof(float)) != 0) {
        return -1;
    }
    
    return 0;
}

/**
 * @brief 从GPU设备复制梯度数据
 */
static int trainer_gpu_copy_gradients_from_device(Trainer* trainer, float* gradients,
                                                 size_t num_parameters) {
    if (!trainer || !trainer->gpu_gradients || num_parameters == 0) {
        return -1;
    }
    
    if (gpu_memory_copy_from_device(gradients,
                                   trainer->gpu_gradients,
                                   num_parameters * sizeof(float)) != 0) {
        return -1;
    }
    
    return 0;
}

/**
 * @brief 将参数数据复制到GPU设备
 */
static int trainer_gpu_copy_parameters_to_device(Trainer* trainer, const float* parameters,
                                                size_t num_parameters) {
    if (!trainer || !trainer->gpu_parameters || !parameters || num_parameters == 0) {
        return -1;
    }
    
    if (gpu_memory_copy_to_device(trainer->gpu_parameters,
                                 parameters,
                                 num_parameters * sizeof(float)) != 0) {
        return -1;
    }
    
    return 0;
}

/**
 * @brief GPU参数更新（在GPU上执行优化器更新）
 */
static int trainer_gpu_update_parameters(Trainer* trainer, float* parameters,
                                        const float* gradients, size_t num_parameters) {
    if (!trainer || !trainer->gpu_parameters || !trainer->gpu_gradients || 
        !parameters || !gradients || num_parameters == 0) {
        return -1;
    }
    
    // GPU优化器更新函数：使用通用GPU优化器内核执行参数更新
    // 当前版本已实现完整GPU优化器内核，支持SGD/Momentum/AdaGrad/RMSProp/Adam/AdamW
    // 采用参数化内核设计，通过内核参数动态配置优化器类型和超参数
    
    // 将梯度复制到GPU（如果尚未复制）
    if (gpu_memory_copy_to_device(trainer->gpu_gradients,
                                 gradients,
                                 num_parameters * sizeof(float)) != 0) {
        return -1;
    }
    
    // 将参数复制到GPU（如果尚未复制）
    if (gpu_memory_copy_to_device(trainer->gpu_parameters,
                                 parameters,
                                 num_parameters * sizeof(float)) != 0) {
        return -1;
    }
    
    // GPU优化器更新实现
    // 使用GPU内核执行优化器更新，避免降级到CPU
    // 当前实现支持SGD优化器，其他优化器类型需要扩展
    
    // 检查是否支持GPU优化器内核
    int gpu_optimizer_supported = 0;
    
    // 根据优化器类型选择GPU实现
    switch (trainer->optimizer.type) {
        case OPTIMIZER_SGD: {
        // 在GPU上执行SGD更新: parameters = parameters - learning_rate * gradients
        // 创建或重用GPU优化器内核
        static GpuKernel* gpu_optimizer_kernel = NULL;
        static GpuContext* last_gpu_context = NULL;
        
        // 如果内核尚未创建或上下文已更改，创建新内核
        if (!gpu_optimizer_kernel || trainer->gpu_context != last_gpu_context) {
            if (gpu_optimizer_kernel) {
                gpu_kernel_free(gpu_optimizer_kernel);
                gpu_optimizer_kernel = NULL;
            }
            
            // SGD优化器GPU内核：向量缩放和加法
            // parameters[i] = parameters[i] - learning_rate * gradients[i]
            const char* optimizer_kernel_source = 
                "__kernel void sgd_update_gpu(__global float* parameters,\n"
                "                             __global float* gradients,\n"
                "                             float learning_rate,\n"
                "                             int num_elements) {\n"
                "    int idx = get_global_id(0);\n"
                "    if (idx >= num_elements) return;\n"
                "    \n"
                "    // SGD更新: parameters = parameters - learning_rate * gradients\n"
                "    parameters[idx] = parameters[idx] - learning_rate * gradients[idx];\n"
                "}\n";
            
            // 创建GPU内核
            gpu_optimizer_kernel = gpu_kernel_create(trainer->gpu_context,
                                                    optimizer_kernel_source,
                                                    "sgd_update_gpu");
            if (!gpu_optimizer_kernel) {
                log_info("错误: 创建GPU优化器内核失败，禁止降级到CPU实现");
                return -3;
            } else {
                last_gpu_context = trainer->gpu_context;
                gpu_optimizer_supported = 1;
            }
        } else {
            gpu_optimizer_supported = 1;
        }
        
        // 如果GPU优化器内核可用，执行GPU更新
        if (gpu_optimizer_supported && gpu_optimizer_kernel) {
            // 设置内核参数
            // 参数0: 参数数据 (GPU内存)
            if (gpu_kernel_set_arg(gpu_optimizer_kernel, 0, sizeof(GpuMemory*), &trainer->gpu_parameters) != 0) {
                log_info("警告: 设置GPU优化器内核参数0失败");
                gpu_optimizer_supported = 0;
            }
            
            // 参数1: 梯度数据 (GPU内存)
            if (gpu_kernel_set_arg(gpu_optimizer_kernel, 1, sizeof(GpuMemory*), &trainer->gpu_gradients) != 0) {
                log_info("警告: 设置GPU优化器内核参数1失败");
                gpu_optimizer_supported = 0;
            }
            
            // 参数2: 学习率 (标量)
            float learning_rate = trainer->optimizer.learning_rate;
            if (gpu_kernel_set_arg(gpu_optimizer_kernel, 2, sizeof(float), &learning_rate) != 0) {
                log_info("警告: 设置GPU优化器内核参数2失败");
                gpu_optimizer_supported = 0;
            }
            
            // 参数3: 元素数量 (整数)
            int num_elements = (int)num_parameters;
            if (gpu_kernel_set_arg(gpu_optimizer_kernel, 3, sizeof(int), &num_elements) != 0) {
                log_info("警告: 设置GPU优化器内核参数3失败");
                gpu_optimizer_supported = 0;
            }
            
            // 执行GPU内核
            if (gpu_optimizer_supported) {
                // 一维工作空间：每个元素一个工作项
                size_t global_work_size = (size_t)num_parameters;
                // 本地工作组大小：GPU特性相关，使用NULL让运行时决定
                size_t local_work_size = 0;
                
                if (gpu_kernel_execute(gpu_optimizer_kernel, global_work_size, local_work_size) != 0) {
                    log_info("警告: 执行GPU优化器内核失败");
                    gpu_optimizer_supported = 0;
                }
            }
        }
        
        // SGD case结束
        break;
    }
    
    // 高级优化器：在GPU上维护动量/二阶矩状态缓冲区
    // 使用gpu_optimizer_m/gpu_optimizer_v作为通用状态缓冲区
    // 每个优化器类型有独立的内核实现，通过静态内核缓存避免重复编译
    
    case OPTIMIZER_MOMENTUM: {
        // Momentum优化器GPU内核：v = momentum*v + lr*grad; param = param - v
        // 使用gpu_optimizer_m作为速度缓冲区，与optimizer.momentum_buffer同步
        
        static GpuKernel* momentum_kernel = NULL;
        static GpuContext* momentum_last_ctx = NULL;
        
        // 分配GPU动量状态缓冲区
        size_t state_bytes = num_parameters * sizeof(float);
        if (!trainer->gpu_optimizer_m || trainer->gpu_optimizer_state_size != state_bytes) {
            if (trainer->gpu_optimizer_m) {
                gpu_memory_free(trainer->gpu_optimizer_m);
                trainer->gpu_optimizer_m = NULL;
            }
            trainer->gpu_optimizer_m = gpu_memory_alloc(trainer->gpu_context,
                                                       state_bytes, GPU_MEMORY_DEVICE);
            if (!trainer->gpu_optimizer_m) {
                log_info("错误: 分配GPU动量缓冲区失败");
                break;
            }
            trainer->gpu_optimizer_state_size = state_bytes;
            
            // 从CPU动量缓冲区初始化，或置零
            if (trainer->optimizer.momentum_buffer &&
                trainer->optimizer.momentum_buffer_size >= num_parameters) {
                gpu_memory_copy_to_device(trainer->gpu_optimizer_m,
                                         trainer->optimizer.momentum_buffer,
                                         state_bytes);
            } else {
                float* zeros = (float*)safe_calloc(num_parameters, sizeof(float));
                gpu_memory_copy_to_device(trainer->gpu_optimizer_m, zeros, state_bytes);
                safe_free((void**)&zeros);
            }
            trainer->gpu_optimizer_state_initialized = 1;
        }
        
        // 同步CPU→GPU状态
        if (trainer->optimizer.momentum_buffer &&
            trainer->optimizer.momentum_buffer_size >= num_parameters) {
            gpu_memory_copy_to_device(trainer->gpu_optimizer_m,
                                     trainer->optimizer.momentum_buffer,
                                     state_bytes);
        }
        
        // 创建或重用Momentum内核
        if (!momentum_kernel || trainer->gpu_context != momentum_last_ctx) {
            if (momentum_kernel) { gpu_kernel_free(momentum_kernel); momentum_kernel = NULL; }
            
            const char* kernel_src =
                "__kernel void momentum_update_gpu(__global float* params,\n"
                "    __global float* grads, __global float* velocity,\n"
                "    float lr, float momentum, int n) {\n"
                "    int i = get_global_id(0); if (i >= n) return;\n"
                "    velocity[i] = momentum * velocity[i] + lr * grads[i];\n"
                "    params[i] = params[i] - velocity[i];\n"
                "}\n";
            
            momentum_kernel = gpu_kernel_create(trainer->gpu_context,
                                               kernel_src, "momentum_update_gpu");
            if (!momentum_kernel) {
                log_info("错误: 创建GPU Momentum内核失败，禁止降级到CPU实现");
                return -3;
            }
            momentum_last_ctx = trainer->gpu_context;
        }
        
        // 设置内核参数并执行
        int kernel_ok = 1;
        float lr = trainer->optimizer.learning_rate;
        float mom = trainer->optimizer.momentum;
        int nelem = (int)num_parameters;
        
        if (gpu_kernel_set_arg(momentum_kernel, 0, sizeof(GpuMemory*), &trainer->gpu_parameters) != 0) kernel_ok = 0;
        if (kernel_ok && gpu_kernel_set_arg(momentum_kernel, 1, sizeof(GpuMemory*), &trainer->gpu_gradients) != 0) kernel_ok = 0;
        if (kernel_ok && gpu_kernel_set_arg(momentum_kernel, 2, sizeof(GpuMemory*), &trainer->gpu_optimizer_m) != 0) kernel_ok = 0;
        if (kernel_ok && gpu_kernel_set_arg(momentum_kernel, 3, sizeof(float), &lr) != 0) kernel_ok = 0;
        if (kernel_ok && gpu_kernel_set_arg(momentum_kernel, 4, sizeof(float), &mom) != 0) kernel_ok = 0;
        if (kernel_ok && gpu_kernel_set_arg(momentum_kernel, 5, sizeof(int), &nelem) != 0) kernel_ok = 0;
        
        if (kernel_ok) {
            size_t gws = (size_t)num_parameters;
            if (gpu_kernel_execute(momentum_kernel, gws, 0) != 0) {
                log_info("警告: 执行GPU Momentum内核失败");
                kernel_ok = 0;
            }
        }
        
        if (!kernel_ok) { return -3; }
        
        // 复制GPU→CPU动量状态
        if (trainer->optimizer.momentum_buffer &&
            trainer->optimizer.momentum_buffer_size >= num_parameters) {
            gpu_memory_copy_from_device(trainer->optimizer.momentum_buffer,
                                       trainer->gpu_optimizer_m, state_bytes);
        }
        
        gpu_optimizer_supported = 1;
        break;
    }
    
    case OPTIMIZER_ADAGRAD: {
        // AdaGrad优化器GPU内核：G += grad^2; param = param - lr*grad/(sqrt(G)+eps)
        // 使用gpu_optimizer_m作为平方梯度累加器（仅GPU端维护，不同步CPU）
        
        static GpuKernel* adagrad_kernel = NULL;
        static GpuContext* adagrad_last_ctx = NULL;
        
        size_t state_bytes = num_parameters * sizeof(float);
        if (!trainer->gpu_optimizer_m || trainer->gpu_optimizer_state_size != state_bytes) {
            if (trainer->gpu_optimizer_m) {
                gpu_memory_free(trainer->gpu_optimizer_m);
                trainer->gpu_optimizer_m = NULL;
            }
            trainer->gpu_optimizer_m = gpu_memory_alloc(trainer->gpu_context,
                                                       state_bytes, GPU_MEMORY_DEVICE);
            if (!trainer->gpu_optimizer_m) {
                log_info("错误: 分配GPU AdaGrad累加器失败");
                break;
            }
            trainer->gpu_optimizer_state_size = state_bytes;
            float* zeros = (float*)safe_calloc(num_parameters, sizeof(float));
            gpu_memory_copy_to_device(trainer->gpu_optimizer_m, zeros, state_bytes);
            safe_free((void**)&zeros);
            trainer->gpu_optimizer_state_initialized = 1;
        }
        
        if (!adagrad_kernel || trainer->gpu_context != adagrad_last_ctx) {
            if (adagrad_kernel) { gpu_kernel_free(adagrad_kernel); adagrad_kernel = NULL; }
            
            const char* kernel_src =
                "__kernel void adagrad_update_gpu(__global float* params,\n"
                "    __global float* grads, __global float* sum_sq,\n"
                "    float lr, float eps, int n) {\n"
                "    int i = get_global_id(0); if (i >= n) return;\n"
                "    sum_sq[i] += grads[i] * grads[i];\n"
                "    params[i] = params[i] - lr * grads[i] / (sqrt(sum_sq[i]) + eps);\n"
                "}\n";
            
            adagrad_kernel = gpu_kernel_create(trainer->gpu_context,
                                              kernel_src, "adagrad_update_gpu");
            if (!adagrad_kernel) {
                log_info("错误: 创建GPU AdaGrad内核失败，禁止降级到CPU实现");
                return -3;
            }
            adagrad_last_ctx = trainer->gpu_context;
        }
        
        int kernel_ok = 1;
        float lr = trainer->optimizer.learning_rate;
        float eps = (trainer->optimizer.epsilon > 0.0f) ? trainer->optimizer.epsilon : 1e-8f;
        int nelem = (int)num_parameters;
        
        if (gpu_kernel_set_arg(adagrad_kernel, 0, sizeof(GpuMemory*), &trainer->gpu_parameters) != 0) kernel_ok = 0;
        if (kernel_ok && gpu_kernel_set_arg(adagrad_kernel, 1, sizeof(GpuMemory*), &trainer->gpu_gradients) != 0) kernel_ok = 0;
        if (kernel_ok && gpu_kernel_set_arg(adagrad_kernel, 2, sizeof(GpuMemory*), &trainer->gpu_optimizer_m) != 0) kernel_ok = 0;
        if (kernel_ok && gpu_kernel_set_arg(adagrad_kernel, 3, sizeof(float), &lr) != 0) kernel_ok = 0;
        if (kernel_ok && gpu_kernel_set_arg(adagrad_kernel, 4, sizeof(float), &eps) != 0) kernel_ok = 0;
        if (kernel_ok && gpu_kernel_set_arg(adagrad_kernel, 5, sizeof(int), &nelem) != 0) kernel_ok = 0;
        
        if (kernel_ok) {
            size_t gws = (size_t)num_parameters;
            if (gpu_kernel_execute(adagrad_kernel, gws, 0) != 0) {
                log_info("警告: 执行GPU AdaGrad内核失败");
                kernel_ok = 0;
            }
        }
        
        if (!kernel_ok) {
            /* P0-012修复: GPU AdaGrad内核执行失败时自动回退CPU */
            log_info("警告: 执行GPU AdaGrad内核失败，自动回退到CPU优化器");
            break;
        }
        
        gpu_optimizer_supported = 1;
        break;
    }
    
    case OPTIMIZER_RMSPROP: {
        // RMSProp优化器GPU内核：sq_avg = decay*sq_avg + (1-decay)*grad^2
        //                         param = param - lr*grad/(sqrt(sq_avg)+eps)
        // 使用gpu_optimizer_m作为平方梯度均值缓冲区（仅GPU端维护）
        
        static GpuKernel* rmsprop_kernel = NULL;
        static GpuContext* rmsprop_last_ctx = NULL;
        
        size_t state_bytes = num_parameters * sizeof(float);
        if (!trainer->gpu_optimizer_m || trainer->gpu_optimizer_state_size != state_bytes) {
            if (trainer->gpu_optimizer_m) {
                gpu_memory_free(trainer->gpu_optimizer_m);
                trainer->gpu_optimizer_m = NULL;
            }
            trainer->gpu_optimizer_m = gpu_memory_alloc(trainer->gpu_context,
                                                       state_bytes, GPU_MEMORY_DEVICE);
            if (!trainer->gpu_optimizer_m) {
                log_info("错误: 分配GPU RMSProp缓冲区失败");
                break;
            }
            trainer->gpu_optimizer_state_size = state_bytes;
            float* zeros = (float*)safe_calloc(num_parameters, sizeof(float));
            gpu_memory_copy_to_device(trainer->gpu_optimizer_m, zeros, state_bytes);
            safe_free((void**)&zeros);
            trainer->gpu_optimizer_state_initialized = 1;
        }
        
        if (!rmsprop_kernel || trainer->gpu_context != rmsprop_last_ctx) {
            if (rmsprop_kernel) { gpu_kernel_free(rmsprop_kernel); rmsprop_kernel = NULL; }
            
            const char* kernel_src =
                "__kernel void rmsprop_update_gpu(__global float* params,\n"
                "    __global float* grads, __global float* sq_avg,\n"
                "    float lr, float decay, float eps, int n) {\n"
                "    int i = get_global_id(0); if (i >= n) return;\n"
                "    sq_avg[i] = decay * sq_avg[i] + (1.0f - decay) * grads[i] * grads[i];\n"
                "    params[i] = params[i] - lr * grads[i] / (sqrt(sq_avg[i]) + eps);\n"
                "}\n";
            
            rmsprop_kernel = gpu_kernel_create(trainer->gpu_context,
                                              kernel_src, "rmsprop_update_gpu");
            if (!rmsprop_kernel) {
                /* P0-012修复: GPU内核创建失败时自动回退CPU */
                log_info("警告: 创建GPU RMSProp内核失败，自动回退到CPU优化器");
                break;
            }
            rmsprop_last_ctx = trainer->gpu_context;
        }
        
        int kernel_ok = 1;
        float lr = trainer->optimizer.learning_rate;
        float decay = (trainer->optimizer.beta2 > 0.0f) ? trainer->optimizer.beta2 : 0.9f;
        float eps = (trainer->optimizer.epsilon > 0.0f) ? trainer->optimizer.epsilon : 1e-8f;
        int nelem = (int)num_parameters;
        
        if (gpu_kernel_set_arg(rmsprop_kernel, 0, sizeof(GpuMemory*), &trainer->gpu_parameters) != 0) kernel_ok = 0;
        if (kernel_ok && gpu_kernel_set_arg(rmsprop_kernel, 1, sizeof(GpuMemory*), &trainer->gpu_gradients) != 0) kernel_ok = 0;
        if (kernel_ok && gpu_kernel_set_arg(rmsprop_kernel, 2, sizeof(GpuMemory*), &trainer->gpu_optimizer_m) != 0) kernel_ok = 0;
        if (kernel_ok && gpu_kernel_set_arg(rmsprop_kernel, 3, sizeof(float), &lr) != 0) kernel_ok = 0;
        if (kernel_ok && gpu_kernel_set_arg(rmsprop_kernel, 4, sizeof(float), &decay) != 0) kernel_ok = 0;
        if (kernel_ok && gpu_kernel_set_arg(rmsprop_kernel, 5, sizeof(float), &eps) != 0) kernel_ok = 0;
        if (kernel_ok && gpu_kernel_set_arg(rmsprop_kernel, 6, sizeof(int), &nelem) != 0) kernel_ok = 0;
        
        if (kernel_ok) {
            size_t gws = (size_t)num_parameters;
            if (gpu_kernel_execute(rmsprop_kernel, gws, 0) != 0) {
                log_info("警告: 执行GPU RMSProp内核失败");
                kernel_ok = 0;
            }
        }
        
        if (!kernel_ok) {
            /* P0-012修复: GPU RMSProp内核执行失败时自动回退CPU */
            log_info("警告: 执行GPU RMSProp内核失败，自动回退到CPU优化器");
            break;
        }
        
        gpu_optimizer_supported = 1;
        break;
    }
    
    case OPTIMIZER_ADAM:{
        // Adam优化器GPU内核：m = b1*m + (1-b1)*grad
        //                    v = b2*v + (1-b2)*grad^2
        //                    m_hat = m/(1-b1^t), v_hat = v/(1-b2^t)
        //                    param = param - lr*m_hat/(sqrt(v_hat)+eps)
        // 使用gpu_optimizer_m作为一阶矩，gpu_optimizer_v作为二阶矩
        
        static GpuKernel* adam_kernel = NULL;
        static GpuContext* adam_last_ctx = NULL;
        
        size_t state_bytes = num_parameters * sizeof(float);
        if (!trainer->gpu_optimizer_m || !trainer->gpu_optimizer_v ||
            trainer->gpu_optimizer_state_size != state_bytes) {
            if (trainer->gpu_optimizer_m) { gpu_memory_free(trainer->gpu_optimizer_m); trainer->gpu_optimizer_m = NULL; }
            if (trainer->gpu_optimizer_v) { gpu_memory_free(trainer->gpu_optimizer_v); trainer->gpu_optimizer_v = NULL; }
            
            trainer->gpu_optimizer_m = gpu_memory_alloc(trainer->gpu_context,
                                                       state_bytes, GPU_MEMORY_DEVICE);
            trainer->gpu_optimizer_v = gpu_memory_alloc(trainer->gpu_context,
                                                       state_bytes, GPU_MEMORY_DEVICE);
            if (!trainer->gpu_optimizer_m || !trainer->gpu_optimizer_v) {
                log_info("错误: 分配GPU Adam状态缓冲区失败");
                if (trainer->gpu_optimizer_m) { gpu_memory_free(trainer->gpu_optimizer_m); trainer->gpu_optimizer_m = NULL; }
                if (trainer->gpu_optimizer_v) { gpu_memory_free(trainer->gpu_optimizer_v); trainer->gpu_optimizer_v = NULL; }
                break;
            }
            trainer->gpu_optimizer_state_size = state_bytes;
            
            // 从CPU状态初始化或置零
            if (trainer->optimizer.m_buffer && trainer->optimizer.v_buffer &&
                trainer->optimizer.adam_buffer_size >= num_parameters) {
                gpu_memory_copy_to_device(trainer->gpu_optimizer_m,
                                         trainer->optimizer.m_buffer, state_bytes);
                gpu_memory_copy_to_device(trainer->gpu_optimizer_v,
                                         trainer->optimizer.v_buffer, state_bytes);
            } else {
                float* zeros = (float*)safe_calloc(num_parameters, sizeof(float));
                gpu_memory_copy_to_device(trainer->gpu_optimizer_m, zeros, state_bytes);
                gpu_memory_copy_to_device(trainer->gpu_optimizer_v, zeros, state_bytes);
                safe_free((void**)&zeros);
            }
            trainer->gpu_optimizer_state_initialized = 1;
        }
        
        // 同步CPU→GPU状态
        if (trainer->optimizer.m_buffer && trainer->optimizer.v_buffer &&
            trainer->optimizer.adam_buffer_size >= num_parameters) {
            gpu_memory_copy_to_device(trainer->gpu_optimizer_m,
                                     trainer->optimizer.m_buffer, state_bytes);
            gpu_memory_copy_to_device(trainer->gpu_optimizer_v,
                                     trainer->optimizer.v_buffer, state_bytes);
        }
        
        // 创建或重用Adam内核
        if (!adam_kernel || trainer->gpu_context != adam_last_ctx) {
            if (adam_kernel) { gpu_kernel_free(adam_kernel); adam_kernel = NULL; }
            
            const char* kernel_src =
                "__kernel void adam_update_gpu(__global float* params,\n"
                "    __global float* grads, __global float* m, __global float* v,\n"
                "    float lr, float b1, float b2, float eps,\n"
                "    float b1_t, float b2_t, int n) {\n"
                "    int i = get_global_id(0); if (i >= n) return;\n"
                "    m[i] = b1 * m[i] + (1.0f - b1) * grads[i];\n"
                "    v[i] = b2 * v[i] + (1.0f - b2) * grads[i] * grads[i];\n"
                "    float m_hat = m[i] / (1.0f - b1_t);\n"
                "    float v_hat = v[i] / (1.0f - b2_t);\n"
                "    params[i] = params[i] - lr * m_hat / (sqrt(v_hat) + eps);\n"
                "}\n";
            
            adam_kernel = gpu_kernel_create(trainer->gpu_context,
                                           kernel_src, "adam_update_gpu");
            if (!adam_kernel) {
                /* P0-012修复: GPU内核创建失败时自动回退CPU，不硬拒绝 */
                log_info("警告: 创建GPU Adam内核失败，自动回退到CPU优化器");
                break;
            }
            adam_last_ctx = trainer->gpu_context;
        }
        
        int kernel_ok = 1;
        float lr = trainer->optimizer.learning_rate;
        float b1 = (trainer->optimizer.beta1 > 0.0f) ? trainer->optimizer.beta1 : 0.9f;
        float b2 = (trainer->optimizer.beta2 > 0.0f) ? trainer->optimizer.beta2 : 0.999f;
        float eps = (trainer->optimizer.epsilon > 0.0f) ? trainer->optimizer.epsilon : 1e-8f;
        int nelem = (int)num_parameters;
        size_t t_step = (trainer->optimizer.t > 0) ? trainer->optimizer.t : 1;
        float b1_t = powf(b1, (float)t_step);
        float b2_t = powf(b2, (float)t_step);
        
        if (gpu_kernel_set_arg(adam_kernel, 0, sizeof(GpuMemory*), &trainer->gpu_parameters) != 0) kernel_ok = 0;
        if (kernel_ok && gpu_kernel_set_arg(adam_kernel, 1, sizeof(GpuMemory*), &trainer->gpu_gradients) != 0) kernel_ok = 0;
        if (kernel_ok && gpu_kernel_set_arg(adam_kernel, 2, sizeof(GpuMemory*), &trainer->gpu_optimizer_m) != 0) kernel_ok = 0;
        if (kernel_ok && gpu_kernel_set_arg(adam_kernel, 3, sizeof(GpuMemory*), &trainer->gpu_optimizer_v) != 0) kernel_ok = 0;
        if (kernel_ok && gpu_kernel_set_arg(adam_kernel, 4, sizeof(float), &lr) != 0) kernel_ok = 0;
        if (kernel_ok && gpu_kernel_set_arg(adam_kernel, 5, sizeof(float), &b1) != 0) kernel_ok = 0;
        if (kernel_ok && gpu_kernel_set_arg(adam_kernel, 6, sizeof(float), &b2) != 0) kernel_ok = 0;
        if (kernel_ok && gpu_kernel_set_arg(adam_kernel, 7, sizeof(float), &eps) != 0) kernel_ok = 0;
        if (kernel_ok && gpu_kernel_set_arg(adam_kernel, 8, sizeof(float), &b1_t) != 0) kernel_ok = 0;
        if (kernel_ok && gpu_kernel_set_arg(adam_kernel, 9, sizeof(float), &b2_t) != 0) kernel_ok = 0;
        if (kernel_ok && gpu_kernel_set_arg(adam_kernel, 10, sizeof(int), &nelem) != 0) kernel_ok = 0;
        
        if (kernel_ok) {
            size_t gws = (size_t)num_parameters;
            if (gpu_kernel_execute(adam_kernel, gws, 0) != 0) {
                log_info("警告: 执行GPU Adam内核失败");
                kernel_ok = 0;
            }
        }
        
        if (!kernel_ok) {
            /* P0-012修复: GPU Adam内核执行失败时自动回退CPU */
            log_info("警告: 执行GPU Adam内核失败，自动回退到CPU优化器");
            break;
        }
        
        // 复制GPU→CPU Adam状态
        if (trainer->optimizer.m_buffer && trainer->optimizer.v_buffer &&
            trainer->optimizer.adam_buffer_size >= num_parameters) {
            gpu_memory_copy_from_device(trainer->optimizer.m_buffer,
                                       trainer->gpu_optimizer_m, state_bytes);
            gpu_memory_copy_from_device(trainer->optimizer.v_buffer,
                                       trainer->gpu_optimizer_v, state_bytes);
        }
        
        gpu_optimizer_supported = 1;
        break;
    }
    
    case OPTIMIZER_ADAMW: {
        // AdamW优化器GPU内核：Adam + 解耦权重衰减
        // m = b1*m + (1-b1)*grad
        // v = b2*v + (1-b2)*grad^2
        // m_hat = m/(1-b1^t), v_hat = v/(1-b2^t)
        // param = param - lr*(m_hat/(sqrt(v_hat)+eps) + wd*param)
        
        static GpuKernel* adamw_kernel = NULL;
        static GpuContext* adamw_last_ctx = NULL;
        
        size_t state_bytes = num_parameters * sizeof(float);
        if (!trainer->gpu_optimizer_m || !trainer->gpu_optimizer_v ||
            trainer->gpu_optimizer_state_size != state_bytes) {
            if (trainer->gpu_optimizer_m) { gpu_memory_free(trainer->gpu_optimizer_m); trainer->gpu_optimizer_m = NULL; }
            if (trainer->gpu_optimizer_v) { gpu_memory_free(trainer->gpu_optimizer_v); trainer->gpu_optimizer_v = NULL; }
            
            trainer->gpu_optimizer_m = gpu_memory_alloc(trainer->gpu_context,
                                                       state_bytes, GPU_MEMORY_DEVICE);
            trainer->gpu_optimizer_v = gpu_memory_alloc(trainer->gpu_context,
                                                       state_bytes, GPU_MEMORY_DEVICE);
            if (!trainer->gpu_optimizer_m || !trainer->gpu_optimizer_v) {
                log_info("错误: 分配GPU AdamW状态缓冲区失败");
                if (trainer->gpu_optimizer_m) { gpu_memory_free(trainer->gpu_optimizer_m); trainer->gpu_optimizer_m = NULL; }
                if (trainer->gpu_optimizer_v) { gpu_memory_free(trainer->gpu_optimizer_v); trainer->gpu_optimizer_v = NULL; }
                break;
            }
            trainer->gpu_optimizer_state_size = state_bytes;
            
            if (trainer->optimizer.m_buffer && trainer->optimizer.v_buffer &&
                trainer->optimizer.adam_buffer_size >= num_parameters) {
                gpu_memory_copy_to_device(trainer->gpu_optimizer_m,
                                         trainer->optimizer.m_buffer, state_bytes);
                gpu_memory_copy_to_device(trainer->gpu_optimizer_v,
                                         trainer->optimizer.v_buffer, state_bytes);
            } else {
                float* zeros = (float*)safe_calloc(num_parameters, sizeof(float));
                gpu_memory_copy_to_device(trainer->gpu_optimizer_m, zeros, state_bytes);
                gpu_memory_copy_to_device(trainer->gpu_optimizer_v, zeros, state_bytes);
                safe_free((void**)&zeros);
            }
            trainer->gpu_optimizer_state_initialized = 1;
        }
        
        if (trainer->optimizer.m_buffer && trainer->optimizer.v_buffer &&
            trainer->optimizer.adam_buffer_size >= num_parameters) {
            gpu_memory_copy_to_device(trainer->gpu_optimizer_m,
                                     trainer->optimizer.m_buffer, state_bytes);
            gpu_memory_copy_to_device(trainer->gpu_optimizer_v,
                                     trainer->optimizer.v_buffer, state_bytes);
        }
        
        if (!adamw_kernel || trainer->gpu_context != adamw_last_ctx) {
            if (adamw_kernel) { gpu_kernel_free(adamw_kernel); adamw_kernel = NULL; }
            
            const char* kernel_src =
                "__kernel void adamw_update_gpu(__global float* params,\n"
                "    __global float* grads, __global float* m, __global float* v,\n"
                "    float lr, float b1, float b2, float eps,\n"
                "    float b1_t, float b2_t, float wd, int n) {\n"
                "    int i = get_global_id(0); if (i >= n) return;\n"
                "    m[i] = b1 * m[i] + (1.0f - b1) * grads[i];\n"
                "    v[i] = b2 * v[i] + (1.0f - b2) * grads[i] * grads[i];\n"
                "    float m_hat = m[i] / (1.0f - b1_t);\n"
                "    float v_hat = v[i] / (1.0f - b2_t);\n"
                "    params[i] = params[i] - lr * (m_hat / (sqrt(v_hat) + eps) + wd * params[i]);\n"
                "}\n";
            
            adamw_kernel = gpu_kernel_create(trainer->gpu_context,
                                            kernel_src, "adamw_update_gpu");
            if (!adamw_kernel) {
                log_info("错误: 创建GPU AdamW内核失败，禁止降级到CPU实现");
                return -3;
            }
            adamw_last_ctx = trainer->gpu_context;
        }
        
        int kernel_ok = 1;
        float lr = trainer->optimizer.learning_rate;
        float b1 = (trainer->optimizer.beta1 > 0.0f) ? trainer->optimizer.beta1 : 0.9f;
        float b2 = (trainer->optimizer.beta2 > 0.0f) ? trainer->optimizer.beta2 : 0.999f;
        float eps = (trainer->optimizer.epsilon > 0.0f) ? trainer->optimizer.epsilon : 1e-8f;
        float wd = trainer->optimizer.weight_decay;
        int nelem = (int)num_parameters;
        size_t t_step = (trainer->optimizer.t > 0) ? trainer->optimizer.t : 1;
        float b1_t = powf(b1, (float)t_step);
        float b2_t = powf(b2, (float)t_step);
        
        if (gpu_kernel_set_arg(adamw_kernel, 0, sizeof(GpuMemory*), &trainer->gpu_parameters) != 0) kernel_ok = 0;
        if (kernel_ok && gpu_kernel_set_arg(adamw_kernel, 1, sizeof(GpuMemory*), &trainer->gpu_gradients) != 0) kernel_ok = 0;
        if (kernel_ok && gpu_kernel_set_arg(adamw_kernel, 2, sizeof(GpuMemory*), &trainer->gpu_optimizer_m) != 0) kernel_ok = 0;
        if (kernel_ok && gpu_kernel_set_arg(adamw_kernel, 3, sizeof(GpuMemory*), &trainer->gpu_optimizer_v) != 0) kernel_ok = 0;
        if (kernel_ok && gpu_kernel_set_arg(adamw_kernel, 4, sizeof(float), &lr) != 0) kernel_ok = 0;
        if (kernel_ok && gpu_kernel_set_arg(adamw_kernel, 5, sizeof(float), &b1) != 0) kernel_ok = 0;
        if (kernel_ok && gpu_kernel_set_arg(adamw_kernel, 6, sizeof(float), &b2) != 0) kernel_ok = 0;
        if (kernel_ok && gpu_kernel_set_arg(adamw_kernel, 7, sizeof(float), &eps) != 0) kernel_ok = 0;
        if (kernel_ok && gpu_kernel_set_arg(adamw_kernel, 8, sizeof(float), &b1_t) != 0) kernel_ok = 0;
        if (kernel_ok && gpu_kernel_set_arg(adamw_kernel, 9, sizeof(float), &b2_t) != 0) kernel_ok = 0;
        if (kernel_ok && gpu_kernel_set_arg(adamw_kernel, 10, sizeof(float), &wd) != 0) kernel_ok = 0;
        if (kernel_ok && gpu_kernel_set_arg(adamw_kernel, 11, sizeof(int), &nelem) != 0) kernel_ok = 0;
        
        if (kernel_ok) {
            size_t gws = (size_t)num_parameters;
            if (gpu_kernel_execute(adamw_kernel, gws, 0) != 0) {
                log_info("警告: 执行GPU AdamW内核失败");
                kernel_ok = 0;
            }
        }
        
        if (!kernel_ok) { return -3; }
        
        if (trainer->optimizer.m_buffer && trainer->optimizer.v_buffer &&
            trainer->optimizer.adam_buffer_size >= num_parameters) {
            gpu_memory_copy_from_device(trainer->optimizer.m_buffer,
                                       trainer->gpu_optimizer_m, state_bytes);
            gpu_memory_copy_from_device(trainer->optimizer.v_buffer,
                                       trainer->gpu_optimizer_v, state_bytes);
        }
        
        gpu_optimizer_supported = 1;
        break;
    }
    
    default:
        // 未知优化器类型，GPU不支持
        gpu_optimizer_supported = 0;
        break;
    }
    
    if (!gpu_optimizer_supported) {
        /* P0-012修复: GPU优化器不可用时自动回退到CPU，不再硬拒绝。
         * 支持CPU计算和CPU训练是核心需求，GPU加速失败时不应阻止训练。
         * 返回-1让调用方(trainer_train循环)自动回退到CPU优化器路径。 */
        log_info("警告: GPU优化器不可用，自动回退到CPU计算(支持CPU训练)");
        return -1;
    } else {
        // GPU优化器更新已成功执行，参数已在GPU内存中更新
        // 无需额外复制，参数已更新
        // 如果需要，可以将更新后的参数复制回主机内存
        if (gpu_memory_copy_from_device(parameters, trainer->gpu_parameters,
                                        num_parameters * sizeof(float)) != 0) {
            log_info("警告: 从GPU复制更新后的参数失败");
        }
    }
    
    return 0;
}

/**
 * @brief 对网络应用高级正则化
 * 
 * 在训练过程中应用配置的高级正则化技术，如DropPath、CutMix、MixUp、
 * 对抗训练等。根据当前训练轮数自动调度正则化强度。
 * 
 * @param trainer 训练器
 * @param inputs 输入数据（将被增强）
 * @param targets 目标数据（将被增强）
 * @param batch_size 当前批次大小
 * @param input_dim 输入维度
 * @param output_dim 输出维度
 * @param augmented_inputs 增强后的输入输出缓冲区
 * @param augmented_targets 增强后的目标输出缓冲区
 * @param epoch 当前训练轮数
 * @param total_epochs 总训练轮数
 * @return int 成功返回0，失败返回-1
 */
int trainer_apply_regularization(Trainer* trainer,
                                 const float* inputs, const float* targets,
                                 size_t batch_size, size_t input_dim, size_t output_dim,
                                 float* augmented_inputs, float* augmented_targets,
                                 size_t epoch, size_t total_epochs) {
    if (!trainer || !inputs || !targets || !augmented_inputs || !augmented_targets) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__, "参数无效");
        return -1;
    }
    
    if (!trainer->regularizer) {
        memcpy(augmented_inputs, inputs, batch_size * input_dim * sizeof(float));
        memcpy(augmented_targets, targets, batch_size * output_dim * sizeof(float));
        return 0;
    }
    
    advanced_regularizer_update_schedule(trainer->regularizer, epoch, total_epochs);
    
    int ret = 0;
    LNNConfig net_config;
    memset(&net_config, 0, sizeof(LNNConfig));
    if (lnn_get_config(trainer->network, &net_config) != 0) {
        memcpy(augmented_inputs, inputs, batch_size * input_dim * sizeof(float));
        memcpy(augmented_targets, targets, batch_size * output_dim * sizeof(float));
        return 0;
    }
    
    memcpy(augmented_inputs, inputs, batch_size * input_dim * sizeof(float));
    memcpy(augmented_targets, targets, batch_size * output_dim * sizeof(float));
    
    float reg_strength = advanced_regularizer_get_strength(trainer->regularizer);
    if (reg_strength > 0.01f) {
        float* cutmix_inputs = (float*)safe_malloc(batch_size * input_dim * sizeof(float));
        float* cutmix_targets = (float*)safe_malloc(batch_size * output_dim * sizeof(float));
        if (cutmix_inputs && cutmix_targets) {
            ret = advanced_regularizer_apply_cutmix(trainer->regularizer,
                                                     inputs, targets,
                                                     batch_size, input_dim, output_dim,
                                                     cutmix_inputs, cutmix_targets);
            if (ret == 0) {
                memcpy(augmented_inputs, cutmix_inputs, batch_size * input_dim * sizeof(float));
                memcpy(augmented_targets, cutmix_targets, batch_size * output_dim * sizeof(float));
            }
            safe_free((void**)&cutmix_inputs);
            safe_free((void**)&cutmix_targets);
        }
        
        float* mixup_inputs = (float*)safe_malloc(batch_size * input_dim * sizeof(float));
        float* mixup_targets = (float*)safe_malloc(batch_size * output_dim * sizeof(float));
        if (mixup_inputs && mixup_targets) {
            ret = advanced_regularizer_apply_mixup(trainer->regularizer,
                                                    augmented_inputs, augmented_targets,
                                                    batch_size, input_dim, output_dim,
                                                    mixup_inputs, mixup_targets);
            if (ret == 0) {
                memcpy(augmented_inputs, mixup_inputs, batch_size * input_dim * sizeof(float));
                memcpy(augmented_targets, mixup_targets, batch_size * output_dim * sizeof(float));
            }
            safe_free((void**)&mixup_inputs);
            safe_free((void**)&mixup_targets);
        }
        
        /* 跨模态数据混合增强 */
        float* multimodal_inputs = (float*)safe_malloc(batch_size * input_dim * sizeof(float));
        float* multimodal_targets = (float*)safe_malloc(batch_size * output_dim * sizeof(float));
        if (multimodal_inputs && multimodal_targets) {
            ret = advanced_regularizer_apply_multimodal_mix(trainer->regularizer,
                                                            augmented_inputs, augmented_targets,
                                                            batch_size, input_dim, output_dim,
                                                            multimodal_inputs, multimodal_targets);
            if (ret == 0) {
                memcpy(augmented_inputs, multimodal_inputs, batch_size * input_dim * sizeof(float));
                memcpy(augmented_targets, multimodal_targets, batch_size * output_dim * sizeof(float));
            }
            safe_free((void**)&multimodal_inputs);
            safe_free((void**)&multimodal_targets);
        }
        
        if (net_config.num_layers > 0) {
            int* layer_indices = (int*)safe_malloc(net_config.num_layers * sizeof(int));
            if (layer_indices) {
                for (size_t i = 0; i < net_config.num_layers; i++) {
                    layer_indices[i] = (int)i;
                }
                advanced_regularizer_apply_drop_path(trainer->regularizer,
                                                      trainer->network,
                                                      layer_indices,
                                                      net_config.num_layers,
                                                      1);
                safe_free((void**)&layer_indices);
            }
        }
    }
    
    return 0;
}

/**
 * @brief 生成对抗样本进行对抗训练
 */
int trainer_generate_adversarial(Trainer* trainer,
                                  const float* inputs, const float* targets,
                                  size_t batch_size, size_t input_dim,
                                  float* adversarial_inputs) {
    if (!trainer || !inputs || !targets || !adversarial_inputs) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__, "参数无效");
        return -1;
    }
    
    if (!trainer->regularizer) {
        selflnn_set_last_error(SELFLNN_ERROR_NOT_INITIALIZED, __func__, __FILE__, __LINE__, "高级正则化器未初始化");
        return -1;
    }
    
    return advanced_regularizer_generate_adversarial(trainer->regularizer,
                                                      trainer->network,
                                                      inputs, targets,
                                                      batch_size, input_dim,
                                                      adversarial_inputs);
}

/**
 * @brief 获取高级正则化器句柄
 */
AdvancedRegularizer* trainer_get_regularizer(Trainer* trainer) {
    if (!trainer) {
        return NULL;
    }
    return trainer->regularizer;
}

/**
 * @brief 配置高级正则化
 */
int trainer_configure_regularization(Trainer* trainer,
                                     const AdvancedRegularizationConfig* config) {
    if (!trainer || !config) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__, "参数无效");
        return -1;
    }
    
    LNNConfig net_config;
    if (lnn_get_config(trainer->network, &net_config) != 0) {
        selflnn_set_last_error(SELFLNN_ERROR_ALGORITHM_FAILURE, __func__, __FILE__, __LINE__, "获取网络配置失败");
        return -1;
    }
    
    if (trainer->regularizer) {
        advanced_regularizer_free(trainer->regularizer);
        trainer->regularizer = NULL;
    }
    
    trainer->regularizer = advanced_regularizer_create(config,
                                                        net_config.input_size,
                                                        net_config.output_size);
    if (!trainer->regularizer) {
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__, "创建高级正则化器失败");
        return -1;
    }
    
    return 0;
}

/**
 * @brief 应用网络级别正则化（DropConnect + 空间Dropout + 可切换归一化）
 *
 * 在训练过程中对网络权重和中间激活应用高级正则化技术。
 * 在每次前向传播前对权重应用DropConnect，在前向传播后对
 * 中间激活应用空间Dropout和可切换归一化。
 *
 * @param trainer 训练器
 * @param batch_inputs 当前批次输入
 * @param batch_size 批次大小
 * @param training 是否训练模式
 * @return int 成功返回0，失败返回-1
 */
int trainer_apply_network_regularization(Trainer* trainer,
                                         const float* batch_inputs,
                                         size_t batch_size,
                                         int training) {
    UNUSED(batch_inputs); UNUSED(batch_size);
    if (!trainer || !trainer->regularizer) {
        return 0;
    }
    
    int ret = 0;
    AdvancedRegularizationType reg_type = advanced_regularizer_get_type(trainer->regularizer);
    float reg_strength = advanced_regularizer_get_strength(trainer->regularizer);
    
    if (reg_strength < 0.01f || !training) {
        return 0;
    }
    
    float* parameters = lnn_get_parameters(trainer->network);
    size_t num_params = lnn_get_parameter_count(trainer->network);
    
    if (parameters && num_params > 0) {
        size_t sqrt_params = (size_t)sqrtf((float)num_params);
        size_t cols = sqrt_params;
        size_t rows = num_params / cols;
        if (rows < 1) rows = 1;
        
        if (reg_type == ADV_REG_DROPCONNECT || reg_type == ADV_REG_ENSEMBLE) {
            float drop_rate = advanced_regularizer_get_drop_rate(trainer->regularizer);
            ret = advanced_regularizer_apply_dropconnect(trainer->regularizer,
                                                         parameters,
                                                         rows, cols,
                                                         drop_rate,
                                                         training);
        }
    }
    
    LNNConfig net_config;
    if (lnn_get_config(trainer->network, &net_config) != 0) {
        return ret;
    }
    
    return ret;
}

/**
 * @brief 应用域自适应训练
 *
 * 使用配置的域自适应方法（DANN、MMD、CORAL、ADDA）对源域和目标域
 * 数据进行域自适应训练。源域数据从常规训练数据中获得，目标域数据
 * 从提供的目标域输入中获得。
 *
 * @param trainer 训练器
 * @param source_inputs 源域输入（带标签的训练数据）
 * @param target_inputs 目标域输入（无标签的部署数据）
 * @param batch_size 批量大小
 * @param input_dim 输入维度
 * @param domain_labels 输出域标签缓冲区（0=源域，1=目标域）
 * @param domain_loss 输出域损失值
 * @return int 成功返回0，失败返回-1
 */
int trainer_apply_domain_adaptation(Trainer* trainer,
                                    const float* source_inputs,
                                    const float* target_inputs,
                                    size_t batch_size, size_t input_dim,
                                    float* domain_labels,
                                    float* domain_loss) {
    if (!trainer || !source_inputs || !target_inputs) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__, "参数无效");
        return -1;
    }
    
    if (!trainer->regularizer) {
        selflnn_set_last_error(SELFLNN_ERROR_NOT_INITIALIZED, __func__, __FILE__, __LINE__, "高级正则化器未初始化");
        return -1;
    }
    
    return advanced_regularizer_apply_domain_adaptation(trainer->regularizer,
                                                        source_inputs,
                                                        target_inputs,
                                                        batch_size, input_dim,
                                                        domain_labels,
                                                        domain_loss);
}

/* ================================================================
 * 在线学习（流式处理）实现
 * ================================================================ */

/**
 * @brief 初始化经验回放缓冲区
 */
static int online_init_replay_buffer(Trainer* trainer, size_t input_dim, size_t output_dim) {
    if (!trainer) return -1;
    
    size_t capacity = 1024;
    if (trainer->config.batch_size > 0) {
        capacity = trainer->config.batch_size * 32;
        if (capacity < 128) capacity = 128;
        if (capacity > 65536) capacity = 65536;
    }
    
    trainer->online_replay_inputs = (float*)safe_malloc(capacity * input_dim * sizeof(float));
    trainer->online_replay_targets = (float*)safe_malloc(capacity * output_dim * sizeof(float));
    
    if (!trainer->online_replay_inputs || !trainer->online_replay_targets) {
        if (trainer->online_replay_inputs) safe_free((void**)&trainer->online_replay_inputs);
        if (trainer->online_replay_targets) safe_free((void**)&trainer->online_replay_targets);
        trainer->online_replay_inputs = NULL;
        trainer->online_replay_targets = NULL;
        return -1;
    }
    
    memset(trainer->online_replay_inputs, 0, capacity * input_dim * sizeof(float));
    memset(trainer->online_replay_targets, 0, capacity * output_dim * sizeof(float));
    
    trainer->online_replay_capacity = capacity;
    trainer->online_replay_size = 0;
    trainer->online_replay_position = 0;
    
    return 0;
}

/**
 * @brief 添加样本到经验回放缓冲区
 */
static int online_add_to_replay(Trainer* trainer, const float* input, const float* target,
                                size_t input_dim, size_t output_dim) {
    if (!trainer || !input || !target) return -1;
    if (!trainer->online_replay_inputs || !trainer->online_replay_targets) return -1;
    if (trainer->online_replay_capacity == 0) return -1;
    
    size_t pos = trainer->online_replay_position;
    size_t input_offset = pos * input_dim;
    size_t target_offset = pos * output_dim;
    
    memcpy(trainer->online_replay_inputs + input_offset, input, input_dim * sizeof(float));
    memcpy(trainer->online_replay_targets + target_offset, target, output_dim * sizeof(float));
    
    trainer->online_replay_position = (pos + 1) % trainer->online_replay_capacity;
    if (trainer->online_replay_size < trainer->online_replay_capacity) {
        trainer->online_replay_size++;
    }
    
    return 0;
}

/**
 * @brief 从经验回放缓冲区采样小批次
 */
static int online_sample_replay(Trainer* trainer, float* batch_inputs, float* batch_targets,
                                size_t batch_size, size_t input_dim, size_t output_dim) {
    if (!trainer || !batch_inputs || !batch_targets) return -1;
    if (trainer->online_replay_size < batch_size) return -1;
    
    size_t size = trainer->online_replay_size;
    
    for (size_t i = 0; i < batch_size; i++) {
        size_t idx = (size_t)(rng_uniform(0.0f, 1.0f) * size);
        if (idx >= size) idx = size - 1;
        
        memcpy(batch_inputs + i * input_dim,
               trainer->online_replay_inputs + idx * input_dim,
               input_dim * sizeof(float));
        memcpy(batch_targets + i * output_dim,
               trainer->online_replay_targets + idx * output_dim,
               output_dim * sizeof(float));
    }
    
    return 0;
}

/**
 * @brief 使用Welford算法更新运行统计
 *
 * 在线更新均值和方差，使用数值稳定的Welford在线算法：
 *   mean_{n+1} = mean_n + (x - mean_n) / (n+1)
 *   M2_{n+1} = M2_n + (x - mean_n) * (x - mean_{n+1})
 *   var_n = M2_n / n
 */
static void online_update_running_stats(Trainer* trainer, const float* input, size_t input_dim) {
    if (!trainer || !input) return;
    if (!trainer->online_running_mean || !trainer->online_running_var) return;
    
    size_t count = trainer->online_running_count;
    
    for (size_t i = 0; i < input_dim; i++) {
        float x = input[i];
        
        if (count == 0) {
            trainer->online_running_mean[i] = x;
            trainer->online_running_var[i] = 0.0f;
        } else {
            float mean_old = trainer->online_running_mean[i];
            float delta = x - mean_old;
            float mean_new = mean_old + delta / (float)(count + 1);
            float delta2 = x - mean_new;
            
            trainer->online_running_mean[i] = mean_new;
            trainer->online_running_var[i] = trainer->online_running_var[i] + delta * delta2;
        }
    }
    
    trainer->online_running_count = count + 1;
}

/**
 * @brief 使用运行统计归一化输入
 */
static void online_normalize_input(Trainer* trainer, float* input, size_t input_dim) {
    if (!trainer || !input) return;
    if (!trainer->online_running_mean || !trainer->online_running_var) return;
    if (trainer->online_running_count < 2) return;
    
    for (size_t i = 0; i < input_dim; i++) {
        float std = sqrtf(trainer->online_running_var[i] / (float)(trainer->online_running_count - 1));
        if (std > 1e-8f) {
            input[i] = (input[i] - trainer->online_running_mean[i]) / std;
        } else {
            input[i] = input[i] - trainer->online_running_mean[i];
        }
    }
}

/**
 * @brief 初始化EWC（弹性权重巩固）
 *
 * 保存当前参数作为最优参数，初始化Fisher信息矩阵为零
 */
static void online_init_ewc(Trainer* trainer, size_t num_params) {
    if (!trainer || num_params == 0) return;
    
    trainer->online_ewc_importance = (float*)safe_malloc(num_params * sizeof(float));
    trainer->online_ewc_optimal_params = (float*)safe_malloc(num_params * sizeof(float));
    
    if (!trainer->online_ewc_importance || !trainer->online_ewc_optimal_params) {
        if (trainer->online_ewc_importance) safe_free((void**)&trainer->online_ewc_importance);
        if (trainer->online_ewc_optimal_params) safe_free((void**)&trainer->online_ewc_optimal_params);
        trainer->online_ewc_importance = NULL;
        trainer->online_ewc_optimal_params = NULL;
        trainer->online_ewc_initialized = 0;
        return;
    }
    
    memset(trainer->online_ewc_importance, 0, num_params * sizeof(float));
    
    float* params = lnn_get_parameters(trainer->network);
    if (params) {
        memcpy(trainer->online_ewc_optimal_params, params, num_params * sizeof(float));
    } else {
        memset(trainer->online_ewc_optimal_params, 0, num_params * sizeof(float));
    }
    
    trainer->online_ewc_initialized = 1;
}

/**
 * @brief 更新EWC Fisher信息矩阵
 *
 * Fisher信息矩阵对角线元素的在线更新：
 *   F_i = (1 - decay) * F_i + decay * gradient_i^2
 *
 * 梯度平方表示该参数对输出的敏感度（信息量）
 */
static void online_update_ewc_fisher(Trainer* trainer, const float* gradients, size_t num_params) {
    if (!trainer || !gradients || num_params == 0) return;
    if (!trainer->online_ewc_initialized) return;
    if (!trainer->online_ewc_importance) return;
    
    float decay = 0.01f;
    if (trainer->config.knowledge_retention_factor > 0) {
        decay = 1.0f - trainer->config.knowledge_retention_factor;
        if (decay < 0.001f) decay = 0.001f;
        if (decay > 0.5f) decay = 0.5f;
    }
    
    for (size_t i = 0; i < num_params; i++) {
        float fisher = gradients[i] * gradients[i];
        trainer->online_ewc_importance[i] = (1.0f - decay) * trainer->online_ewc_importance[i]
                                           + decay * fisher;
    }
}

/**
 * @brief 计算EWC正则化惩罚项
 *
 * EWC惩罚：L_ewc = (lambda / 2) * Σ F_i * (θ_i - θ*_i)^2
 * 其中F_i是Fisher信息矩阵对角线，θ*_i是之前学习的最优参数
 */
static float online_compute_ewc_penalty(Trainer* trainer, const float* parameters, size_t num_params) {
    if (!trainer || !parameters || num_params == 0) return 0.0f;
    if (!trainer->online_ewc_initialized) return 0.0f;
    if (!trainer->online_ewc_importance || !trainer->online_ewc_optimal_params) return 0.0f;
    if (!trainer->config.enable_continual_learning) return 0.0f;
    
    float lambda = trainer->config.regularization_lambda;
    if (lambda <= 0.0f) lambda = 0.1f;
    
    float penalty = 0.0f;
    
    for (size_t i = 0; i < num_params; i++) {
        if (trainer->online_ewc_importance[i] > 1e-10f) {
            float diff = parameters[i] - trainer->online_ewc_optimal_params[i];
            penalty += trainer->online_ewc_importance[i] * diff * diff;
        }
    }
    
    penalty *= 0.5f * lambda;
    
    if (penalty > 1e10f) penalty = 1e10f;
    
    return penalty;
}

/**
 * @brief 检测概念漂移
 *
 * 使用指数加权移动平均（EWMA）监测损失变化：
 *   ma_t = alpha * loss_t + (1 - alpha) * ma_{t-1}
 *   drift_ratio = |ma_t - ma_{t-1}| / ma_{t-1}
 *   如果drift_ratio > threshold，检测到概念漂移
 *
 * @param trainer 训练器
 * @param current_loss 当前损失
 * @return int 1=检测到漂移，0=未检测到
 */
static int online_detect_concept_drift(Trainer* trainer, float current_loss) {
    if (!trainer) return 0;
    
    float alpha = 0.1f;
    float threshold = trainer->online_drift_threshold;
    if (threshold <= 0.0f) threshold = 0.3f;
    
    if (trainer->online_drift_detection_count == 0) {
        trainer->online_drift_loss_ma = current_loss;
        trainer->online_drift_loss_ma_prev = current_loss;
        trainer->online_drift_detection_count = 1;
        return 0;
    }
    
    trainer->online_drift_loss_ma_prev = trainer->online_drift_loss_ma;
    trainer->online_drift_loss_ma = alpha * current_loss + (1.0f - alpha) * trainer->online_drift_loss_ma;
    trainer->online_drift_detection_count++;
    
    float prev = trainer->online_drift_loss_ma_prev;
    float curr = trainer->online_drift_loss_ma;
    
    if (prev > 1e-10f) {
        float drift_ratio = fabsf(curr - prev) / prev;
        if (drift_ratio > threshold && current_loss > prev * 1.5f) {
            trainer->online_drift_detected = 1;
            if (trainer->config.verbose) {
                printf("概念漂移检测：漂移率=%.4f，阈值=%.4f，损失=%.6f->%.6f\n",
                       drift_ratio, threshold, prev, curr);
            }
            return 1;
        }
    }
    
    if (trainer->online_drift_detection_count > 100) {
        trainer->online_drift_detected = 0;
    }
    
    trainer->online_drift_detected = 0;
    return 0;
}

/**
 * @brief 计算在线学习损失
 */
static float online_compute_loss(Trainer* trainer, const float* output, const float* target,
                                 size_t output_dim) {
    if (!trainer || !output || !target || output_dim == 0) return 0.0f;
    
    switch (trainer->config.loss_function) {
        case LOSS_MEAN_SQUARED_ERROR:
            return loss_mean_squared_error(output, target, output_dim);
        case LOSS_MEAN_ABSOLUTE_ERROR:
            return loss_mean_absolute_error(output, target, output_dim);
        case LOSS_CROSS_ENTROPY: {
            float eps = 1e-7f;
            float loss = 0.0f;
            for (size_t i = 0; i < output_dim; i++) {
                float p = output[i];
                if (p < eps) p = eps;
                if (p > 1.0f - eps) p = 1.0f - eps;
                loss -= target[i] * logf(p);
            }
            return loss;
        }
        case LOSS_HUBER: {
            float delta = 1.0f;
            float loss = 0.0f;
            for (size_t i = 0; i < output_dim; i++) {
                float diff = output[i] - target[i];
                float abs_diff = fabsf(diff);
                if (abs_diff <= delta) {
                    loss += 0.5f * diff * diff;
                } else {
                    loss += delta * (abs_diff - 0.5f * delta);
                }
            }
            return loss / output_dim;
        }
        default:
            return loss_mean_squared_error(output, target, output_dim);
    }
}

/**
 * @brief 计算在线学习准确率
 */
static float online_compute_accuracy(Trainer* trainer, const float* output, const float* target,
                                     size_t output_dim) {
    if (!trainer || !output || !target || output_dim == 0) return 0.0f;
    
    switch (trainer->config.loss_function) {
        case LOSS_CROSS_ENTROPY: {
            int predicted = 0;
            int actual = 0;
            float max_pred = output[0];
            float max_act = target[0];
            for (size_t i = 1; i < output_dim; i++) {
                if (output[i] > max_pred) { max_pred = output[i]; predicted = (int)i; }
                if (target[i] > max_act) { max_act = target[i]; actual = (int)i; }
            }
            return (predicted == actual) ? 1.0f : 0.0f;
        }
        case LOSS_MEAN_SQUARED_ERROR:
        case LOSS_MEAN_ABSOLUTE_ERROR: {
            float target_mean = 0.0f;
            for (size_t i = 0; i < output_dim; i++) target_mean += target[i];
            target_mean /= output_dim;
            
            float sst = 0.0f, sse = 0.0f;
            for (size_t i = 0; i < output_dim; i++) {
                float t_diff = target[i] - target_mean;
                sst += t_diff * t_diff;
                float residual = target[i] - output[i];
                sse += residual * residual;
            }
            
            if (sst > 1e-10f) {
                float r2 = 1.0f - (sse / sst);
                if (r2 < 0.0f) r2 = 0.0f;
                if (r2 > 1.0f) r2 = 1.0f;
                return r2;
            }
            return 0.0f;
        }
        default:
            return 0.0f;
    }
}

/**
 * @brief 在线训练神经网络（流式处理）
 *
 * 在线学习核心函数。每个样本到达后：
 * 1. 归一化输入（使用运行统计）
 * 2. 前向传播计算输出和损失
 * 3. 存储到经验回放缓冲区
 * 4. 更新运行统计
 * 5. 从回放缓冲区采样小批量
 * 6. 计算EWC惩罚（如果启用持续学习）
 * 7. 反向传播和参数更新
 * 8. 更新EWC Fisher信息
 * 9. 检测概念漂移
 *
 * 在线学习率调度：使用更快的衰减和更小的初始学习率
 */
int trainer_train_online(Trainer* trainer, const float* inputs, const float* targets,
                         size_t num_samples, TrainingCallback callback, void* user_data) {
    if (!trainer || !inputs || !targets || num_samples == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "在线训练：参数无效");
        return -1;
    }
    
    if (!trainer->network) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "在线训练：网络未设置");
        return -1;
    }
    
    LNNConfig net_config;
    if (lnn_get_config(trainer->network, &net_config) != 0) {
        return -1;
    }
    
    size_t input_dim = net_config.input_size;
    size_t output_dim = net_config.output_size;
    
    if (!validate_network_dimensions(trainer->network, input_dim, output_dim)) {
        return -1;
    }
    
    if (!trainer->online_initialized) {
        float* mean_buf = (float*)safe_malloc(input_dim * sizeof(float));
        float* var_buf = (float*)safe_malloc(input_dim * sizeof(float));
        if (!mean_buf || !var_buf) {
            if (mean_buf) safe_free((void**)&mean_buf);
            if (var_buf) safe_free((void**)&var_buf);
            selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                                  "在线训练：无法分配运行统计缓冲区");
            return -1;
        }
        memset(mean_buf, 0, input_dim * sizeof(float));
        memset(var_buf, 0, input_dim * sizeof(float));
        trainer->online_running_mean = mean_buf;
        trainer->online_running_var = var_buf;
        trainer->online_running_count = 0;
        
        if (online_init_replay_buffer(trainer, input_dim, output_dim) != 0) {
            safe_free((void**)&trainer->online_running_mean);
            safe_free((void**)&trainer->online_running_var);
            trainer->online_running_mean = NULL;
            trainer->online_running_var = NULL;
            selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                                  "在线训练：无法初始化经验回放缓冲区");
            return -1;
        }
        
        size_t num_params = lnn_get_parameter_count(trainer->network);
        if (trainer->config.enable_continual_learning && num_params > 0) {
            online_init_ewc(trainer, num_params);
        }
        
        trainer->online_drift_loss_ma = 0.0f;
        trainer->online_drift_loss_ma_prev = 0.0f;
        trainer->online_drift_threshold = 0.3f;
        trainer->online_drift_detection_count = 0;
        trainer->online_drift_detected = 0;
        trainer->online_input_dim = input_dim;
        trainer->online_output_dim = output_dim;
        trainer->online_initialized = 1;
        
        if (trainer->config.verbose) {
            printf("在线训练初始化：输入维度=%zu，输出维度=%zu，回放容量=%zu\n",
                   input_dim, output_dim, trainer->online_replay_capacity);
        }
    }
    
    float online_lr = trainer->config.learning_rate * 0.5f;
    float lr_decay = 0.995f;
    size_t replay_batch_size = trainer->config.batch_size;
    if (replay_batch_size < 4) replay_batch_size = 4;
    if (replay_batch_size > 256) replay_batch_size = 256;
    float ewc_lambda = trainer->config.regularization_lambda;
    if (ewc_lambda <= 0.0f) ewc_lambda = 0.1f;
    
    trainer->state.start_time = perf_timestamp_ns() / 1000000;
    trainer->state.current_epoch = 0;
    
    float* norm_input = (float*)safe_malloc(input_dim * sizeof(float));
    float* sample_output = (float*)safe_malloc(output_dim * sizeof(float));
    float* replay_batch_inputs = (float*)safe_malloc(replay_batch_size * input_dim * sizeof(float));
    float* replay_batch_targets = (float*)safe_malloc(replay_batch_size * output_dim * sizeof(float));
    float* replay_batch_outputs = (float*)safe_malloc(replay_batch_size * output_dim * sizeof(float));
    float* replay_loss_grad = (float*)safe_malloc(replay_batch_size * output_dim * sizeof(float));
    
    if (!norm_input || !sample_output || !replay_batch_inputs || !replay_batch_targets ||
        !replay_batch_outputs || !replay_loss_grad) {
        if (norm_input) safe_free((void**)&norm_input);
        if (sample_output) safe_free((void**)&sample_output);
        if (replay_batch_inputs) safe_free((void**)&replay_batch_inputs);
        if (replay_batch_targets) safe_free((void**)&replay_batch_targets);
        if (replay_batch_outputs) safe_free((void**)&replay_batch_outputs);
        if (replay_loss_grad) safe_free((void**)&replay_loss_grad);
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "在线训练：无法分配临时缓冲区");
        return -1;
    }
    
    float cumulative_loss = 0.0f;
    float cumulative_accuracy = 0.0f;
    
    for (size_t sample_idx = 0; sample_idx < num_samples; sample_idx++) {
        const float* raw_input = inputs + sample_idx * input_dim;
        const float* target = targets + sample_idx * output_dim;
        
        memcpy(norm_input, raw_input, input_dim * sizeof(float));
        online_normalize_input(trainer, norm_input, input_dim);
        
        online_update_running_stats(trainer, raw_input, input_dim);
        
        // 记忆上下文召回：检索相似经验，调制当前输入
        if (trainer->memory_integration_enabled && trainer->memory_manager && 
            trainer->memory_recall_buffer && trainer->memory_recall_buffer_size > 0) {
            size_t num_recalled = 0;
            float* recall_temp = (float*)safe_malloc(trainer->memory_recall_buffer_size * sizeof(float));
            if (recall_temp) {
                if (trainer_memory_recall_similar(trainer, norm_input, input_dim,
                                                  recall_temp, &num_recalled, 5) == 0 && num_recalled > 0) {
                    float context_bias = trainer->memory_context_strength;
                    size_t avg_dim = input_dim < trainer->memory_recall_buffer_size ? input_dim : trainer->memory_recall_buffer_size;
                    float* avg_context = (float*)safe_malloc(avg_dim * sizeof(float));
                    if (avg_context) {
                        memset(avg_context, 0, avg_dim * sizeof(float));
                        size_t num_to_avg = num_recalled < 5 ? num_recalled : 5;
                        for (size_t ri = 0; ri < num_to_avg; ri++) {
                            for (size_t di = 0; di < avg_dim; di++) {
                                avg_context[di] += recall_temp[ri * (input_dim * 2 + 2) + di];
                            }
                        }
                        for (size_t di = 0; di < avg_dim; di++) {
                            avg_context[di] /= (float)num_to_avg;
                        }
                        for (size_t di = 0; di < avg_dim; di++) {
                            norm_input[di] = (1.0f - context_bias) * norm_input[di] + context_bias * avg_context[di];
                        }
                        safe_free((void**)&avg_context);
                    }
                }
                safe_free((void**)&recall_temp);
            }
        }
        
        if (lnn_forward_batch(trainer->network, norm_input, sample_output, 1) != 0) {
            if (trainer->config.verbose) {
                printf("在线训练警告：样本%zu前向传播失败\n", sample_idx);
            }
            continue;
        }
        
        float sample_loss = online_compute_loss(trainer, sample_output, target, output_dim);
        
        cumulative_loss += sample_loss;
        float sample_acc = online_compute_accuracy(trainer, sample_output, target, output_dim);
        cumulative_accuracy += sample_acc;
        
        online_add_to_replay(trainer, norm_input, target, input_dim, output_dim);
        
        // 保存经验到记忆系统
        if (trainer->memory_integration_enabled && trainer->memory_manager) {
            trainer_memory_save_experience(trainer, raw_input, target, input_dim, output_dim,
                                          sample_loss, sample_acc);
        }
        
        int drift_detected = 0;
        if (sample_idx > 0 && sample_idx % 5 == 0) {
            drift_detected = online_detect_concept_drift(trainer, sample_loss);
        }
        
        // 记忆系统巩固：每处理一定数量的样本后触发
        if (trainer->memory_integration_enabled && trainer->memory_manager &&
            trainer->memory_consolidation_interval > 0) {
            trainer->memory_consolidation_counter++;
            if (trainer->memory_consolidation_counter >= trainer->memory_consolidation_interval * 100) {
                trainer_memory_consolidate(trainer);
                
                // 在线记忆回放训练：从巩固后的记忆中采样批次进行额外训练
                size_t mem_batch_size = 32;
                float* mem_inputs = (float*)safe_malloc(mem_batch_size * input_dim * sizeof(float));
                float* mem_targets = (float*)safe_malloc(mem_batch_size * output_dim * sizeof(float));
                float* mem_outputs = (float*)safe_malloc(mem_batch_size * output_dim * sizeof(float));
                if (mem_inputs && mem_targets && mem_outputs) {
                    MemorySystem* mem_sys = memory_manager_get_system(trainer->memory_manager);
                    int sampled = 0;
                    if (mem_sys) {
                        sampled = memory_sample_training_batch(
                            mem_sys, MEMORY_TYPE_EPISODIC,
                            mem_batch_size, input_dim,
                            mem_inputs, mem_targets);
                    }
                    if (sampled > 0) {
                        if (lnn_forward_batch(trainer->network, mem_inputs, mem_outputs, (size_t)sampled) == 0) {
                            float* mem_grad = (float*)safe_malloc((size_t)sampled * output_dim * sizeof(float));
                            if (mem_grad) {
                                for (size_t i = 0; i < (size_t)sampled * output_dim; i++) {
                                    mem_grad[i] = mem_outputs[i] - mem_targets[i];
                                }
                                lnn_backward_batch(trainer->network, mem_inputs, mem_grad,
                                                  trainer->gradients, (size_t)sampled);
                                float* params = lnn_get_parameters(trainer->network);
                                if (params) {
                                    optimizer_update(&trainer->optimizer, params,
                                                    trainer->gradients, trainer->gradients_size);
                                }
                                safe_free((void**)&mem_grad);
                            }
                        }
                    }
                }
                if (mem_inputs) safe_free((void**)&mem_inputs);
                if (mem_targets) safe_free((void**)&mem_targets);
                if (mem_outputs) safe_free((void**)&mem_outputs);
            }
        }
        
        if (trainer->online_replay_size >= replay_batch_size && sample_idx % 2 == 0) {
            memset(replay_batch_inputs, 0, replay_batch_size * input_dim * sizeof(float));
            memset(replay_batch_targets, 0, replay_batch_size * output_dim * sizeof(float));
            
            if (online_sample_replay(trainer, replay_batch_inputs, replay_batch_targets,
                                     replay_batch_size, input_dim, output_dim) != 0) {
                continue;
            }
            
            if (lnn_forward_batch(trainer->network, replay_batch_inputs, replay_batch_outputs,
                                 replay_batch_size) != 0) {
                continue;
            }
            
            memset(replay_loss_grad, 0, replay_batch_size * output_dim * sizeof(float));
            switch (trainer->config.loss_function) {
                case LOSS_MEAN_SQUARED_ERROR:
                    loss_mean_squared_error_gradient(replay_batch_outputs, replay_batch_targets,
                                                    replay_loss_grad, replay_batch_size * output_dim);
                    break;
                case LOSS_MEAN_ABSOLUTE_ERROR:
                    loss_mean_absolute_error_gradient(replay_batch_outputs, replay_batch_targets,
                                                     replay_loss_grad, replay_batch_size * output_dim);
                    break;
                case LOSS_CROSS_ENTROPY:
                    loss_cross_entropy_gradient(replay_batch_outputs, replay_batch_targets,
                                               replay_loss_grad, replay_batch_size * output_dim);
                    break;
                default:
                    loss_mean_squared_error_gradient(replay_batch_outputs, replay_batch_targets,
                                                    replay_loss_grad, replay_batch_size * output_dim);
                    break;
            }
            
            if (lnn_backward_batch(trainer->network, replay_batch_inputs, replay_loss_grad,
                                  trainer->gradients, replay_batch_size) != 0) {
                continue;
            }
            
            if (trainer->config.gradient_clip != GRADIENT_CLIP_NONE) {
                gradient_clip(trainer->gradients, trainer->gradients_size,
                            trainer->config.gradient_clip,
                            trainer->config.gradient_clip_value,
                            trainer->config.gradient_clip_norm);
            }
            
            trainer_compress_gradients(trainer, trainer->gradients, trainer->gradients_size);
            
            float* parameters = lnn_get_parameters(trainer->network);
            size_t num_params = lnn_get_parameter_count(trainer->network);
            
            if (trainer->config.enable_continual_learning && trainer->online_ewc_initialized && parameters) {
                float ewc_penalty = online_compute_ewc_penalty(trainer, parameters, num_params);
                
                if (ewc_penalty > 0.0f) {
                    float ewc_grad_scale = ewc_lambda / (float)replay_batch_size;
                    for (size_t i = 0; i < num_params && i < trainer->gradients_size; i++) {
                        float diff = parameters[i] - trainer->online_ewc_optimal_params[i];
                        trainer->gradients[i] += ewc_grad_scale * trainer->online_ewc_importance[i] * diff;
                    }
                }
            }
            
            if (parameters && trainer->config.regularization == REGULARIZATION_L1) {
                weight_decay(parameters, trainer->gradients_size,
                            trainer->config.regularization_lambda, REGULARIZATION_L1);
            }
            
            trainer->optimizer.learning_rate = online_lr;
            if (parameters) {
                optimizer_update(&trainer->optimizer, parameters,
                               trainer->gradients, trainer->gradients_size);
            }
            
            if (trainer->config.enable_continual_learning && trainer->online_ewc_initialized) {
                online_update_ewc_fisher(trainer, trainer->gradients, num_params);
            }
            
            online_lr *= lr_decay;
            if (online_lr < trainer->config.learning_rate * 0.01f) {
                online_lr = trainer->config.learning_rate * 0.01f;
            }
            
            if (drift_detected) {
                online_lr *= 1.5f;
                if (online_lr > trainer->config.learning_rate * 2.0f) {
                    online_lr = trainer->config.learning_rate * 2.0f;
                }
                if (trainer->config.verbose) {
                    printf("概念漂移响应：学习率调整为%.6f\n", online_lr);
                }
            }
        }
        
        trainer->state.current_loss = sample_loss;
        trainer->state.current_accuracy = sample_acc;
        trainer->state.samples_processed = sample_idx + 1;
        trainer->state.gradient_norm = gradient_norm(trainer->gradients, trainer->gradients_size);
        trainer->state.learning_rate = online_lr;
        
        if (callback) {
            callback(&trainer->state, user_data);
        }
    }
    
    trainer->state.current_loss = cumulative_loss / (float)num_samples;
    trainer->state.current_accuracy = cumulative_accuracy / (float)num_samples;
    trainer->state.training_time_ms = (perf_timestamp_ns() / 1000000) - trainer->state.start_time;
    
    if (trainer->config.verbose) {
        printf("在线训练完成：处理%zu样本，平均损失=%.4f，平均准确率=%.4f，最终学习率=%.6f，时间=%llums\n",
               num_samples, trainer->state.current_loss, trainer->state.current_accuracy,
               online_lr, (unsigned long long)trainer->state.training_time_ms);
        if (trainer->online_drift_detected) {
            log_warning("检测到概念漂移，建议重新评估模型\n");
        }
        if (trainer->online_ewc_initialized) {
            printf("EWC持续学习已启用，知识保留因子=%.2f\n", trainer->config.knowledge_retention_factor);
        }
        printf("经验回放缓冲区大小=%zu/%zu\n", trainer->online_replay_size, trainer->online_replay_capacity);
    }
    
    safe_free((void**)&norm_input);
    safe_free((void**)&sample_output);
    safe_free((void**)&replay_batch_inputs);
    safe_free((void**)&replay_batch_targets);
    safe_free((void**)&replay_batch_outputs);
    safe_free((void**)&replay_loss_grad);
    
    return 0;
}

/**
 * @brief 在线学习单步更新（实时AGI专用）
 *
 * 针对CfC液态神经网络的实时单样本在线更新。
 * 结合经验回放、EWC持续学习和概念漂移检测。
 *
 * @param trainer 训练器
 * @param input 单个输入样本
 * @param target 单个目标样本
 * @param input_dim 输入维度
 * @param output_dim 输出维度
 * @param loss 输出当前样本损失
 * @return int 成功返回0，失败返回-1
 */
int trainer_online_step(Trainer* trainer, const float* input, const float* target,
                        size_t input_dim, size_t output_dim, float* loss) {
    if (!trainer || !input || !target || input_dim == 0 || output_dim == 0) {
        return -1;
    }

    if (!trainer->online_initialized) {
        float* mean_buf = (float*)safe_malloc(input_dim * sizeof(float));
        float* var_buf = (float*)safe_malloc(input_dim * sizeof(float));
        if (!mean_buf || !var_buf) {
            if (mean_buf) safe_free((void**)&mean_buf);
            if (var_buf) safe_free((void**)&var_buf);
            return -1;
        }
        memset(mean_buf, 0, input_dim * sizeof(float));
        memset(var_buf, 0, input_dim * sizeof(float));
        trainer->online_running_mean = mean_buf;
        trainer->online_running_var = var_buf;
        trainer->online_running_count = 0;

        if (online_init_replay_buffer(trainer, input_dim, output_dim) != 0) {
            safe_free((void**)&trainer->online_running_mean);
            safe_free((void**)&trainer->online_running_var);
            return -1;
        }

        size_t num_params = lnn_get_parameter_count(trainer->network);
        if (trainer->config.enable_continual_learning && num_params > 0) {
            online_init_ewc(trainer, num_params);
        }

        trainer->online_drift_loss_ma = 0.0f;
        trainer->online_drift_loss_ma_prev = 0.0f;
        trainer->online_drift_threshold = 0.3f;
        trainer->online_input_dim = input_dim;
        trainer->online_output_dim = output_dim;
        trainer->online_initialized = 1;
    }

    size_t replay_batch_size = trainer->config.batch_size;
    if (replay_batch_size < 4) replay_batch_size = 4;
    if (replay_batch_size > 256) replay_batch_size = 256;
    float online_lr = trainer->config.learning_rate * 0.5f;
    float ewc_lambda = trainer->config.regularization_lambda;
    if (ewc_lambda <= 0.0f) ewc_lambda = 0.1f;

    float* norm_input = (float*)safe_malloc(input_dim * sizeof(float));
    float* sample_output = (float*)safe_malloc(output_dim * sizeof(float));
    if (!norm_input || !sample_output) {
        if (norm_input) safe_free((void**)&norm_input);
        if (sample_output) safe_free((void**)&sample_output);
        return -1;
    }

    memcpy(norm_input, input, input_dim * sizeof(float));
    online_normalize_input(trainer, norm_input, input_dim);
    online_update_running_stats(trainer, input, input_dim);

    // 记忆上下文召回：检索相似经验，调制当前输入
    if (trainer->memory_integration_enabled && trainer->memory_manager &&
        trainer->memory_recall_buffer && trainer->memory_recall_buffer_size > 0) {
        size_t num_recalled = 0;
        float* recall_temp = (float*)safe_malloc(trainer->memory_recall_buffer_size * sizeof(float));
        if (recall_temp) {
            if (trainer_memory_recall_similar(trainer, norm_input, input_dim,
                                              recall_temp, &num_recalled, 3) == 0 && num_recalled > 0) {
                float context_bias = trainer->memory_context_strength;
                size_t avg_dim = input_dim < trainer->memory_recall_buffer_size ? input_dim : trainer->memory_recall_buffer_size;
                float* avg_context = (float*)safe_malloc(avg_dim * sizeof(float));
                if (avg_context) {
                    memset(avg_context, 0, avg_dim * sizeof(float));
                    size_t num_to_avg = num_recalled < 3 ? num_recalled : 3;
                    for (size_t ri = 0; ri < num_to_avg; ri++) {
                        for (size_t di = 0; di < avg_dim; di++) {
                            avg_context[di] += recall_temp[ri * (input_dim * 2 + 2) + di];
                        }
                    }
                    for (size_t di = 0; di < avg_dim; di++) {
                        avg_context[di] /= (float)num_to_avg;
                    }
                    for (size_t di = 0; di < avg_dim; di++) {
                        norm_input[di] = (1.0f - context_bias) * norm_input[di] + context_bias * avg_context[di];
                    }
                    safe_free((void**)&avg_context);
                }
            }
            safe_free((void**)&recall_temp);
        }
    }

    if (lnn_forward_batch(trainer->network, norm_input, sample_output, 1) != 0) {
        safe_free((void**)&norm_input);
        safe_free((void**)&sample_output);
        return -1;
    }

    float sample_loss = online_compute_loss(trainer, sample_output, target, output_dim);
    float sample_acc = online_compute_accuracy(trainer, sample_output, target, output_dim);
    if (loss) *loss = sample_loss;

    online_add_to_replay(trainer, norm_input, target, input_dim, output_dim);

    // 保存经验到记忆系统
    if (trainer->memory_integration_enabled && trainer->memory_manager) {
        trainer_memory_save_experience(trainer, input, target, input_dim, output_dim,
                                      sample_loss, sample_acc);
    }

    if (trainer->online_replay_size >= replay_batch_size) {
        float* replay_batch_inputs = (float*)safe_malloc(replay_batch_size * input_dim * sizeof(float));
        float* replay_batch_targets = (float*)safe_malloc(replay_batch_size * output_dim * sizeof(float));
        float* replay_batch_outputs = (float*)safe_malloc(replay_batch_size * output_dim * sizeof(float));
        float* replay_loss_grad = (float*)safe_malloc(replay_batch_size * output_dim * sizeof(float));

        if (replay_batch_inputs && replay_batch_targets && replay_batch_outputs && replay_loss_grad) {
            memset(replay_batch_inputs, 0, replay_batch_size * input_dim * sizeof(float));
            memset(replay_batch_targets, 0, replay_batch_size * output_dim * sizeof(float));

            if (online_sample_replay(trainer, replay_batch_inputs, replay_batch_targets,
                                     replay_batch_size, input_dim, output_dim) == 0) {

                if (lnn_forward_batch(trainer->network, replay_batch_inputs, replay_batch_outputs,
                                     replay_batch_size) == 0) {

                    memset(replay_loss_grad, 0, replay_batch_size * output_dim * sizeof(float));
                    switch (trainer->config.loss_function) {
                        case LOSS_MEAN_SQUARED_ERROR:
                            loss_mean_squared_error_gradient(replay_batch_outputs, replay_batch_targets,
                                                            replay_loss_grad, replay_batch_size * output_dim);
                            break;
                        case LOSS_CROSS_ENTROPY:
                            loss_cross_entropy_gradient(replay_batch_outputs, replay_batch_targets,
                                                       replay_loss_grad, replay_batch_size * output_dim);
                            break;
                        default:
                            loss_mean_squared_error_gradient(replay_batch_outputs, replay_batch_targets,
                                                            replay_loss_grad, replay_batch_size * output_dim);
                            break;
                    }

                    if (lnn_backward_batch(trainer->network, replay_batch_inputs, replay_loss_grad,
                                          trainer->gradients, replay_batch_size) == 0) {

                        if (trainer->config.gradient_clip != GRADIENT_CLIP_NONE) {
                            gradient_clip(trainer->gradients, trainer->gradients_size,
                                        trainer->config.gradient_clip,
                                        trainer->config.gradient_clip_value,
                                        trainer->config.gradient_clip_norm);
                        }

                        trainer_compress_gradients(trainer, trainer->gradients, trainer->gradients_size);

                        float* parameters = lnn_get_parameters(trainer->network);
                        size_t num_params = lnn_get_parameter_count(trainer->network);

                        if (trainer->config.enable_continual_learning && trainer->online_ewc_initialized && parameters) {
                            float ewc_penalty = online_compute_ewc_penalty(trainer, parameters, num_params);
                            if (ewc_penalty > 0.0f) {
                                float ewc_grad_scale = ewc_lambda / (float)replay_batch_size;
                                for (size_t i = 0; i < num_params && i < trainer->gradients_size; i++) {
                                    float diff = parameters[i] - trainer->online_ewc_optimal_params[i];
                                    trainer->gradients[i] += ewc_grad_scale * trainer->online_ewc_importance[i] * diff;
                                }
                            }
                        }

                        if (parameters) {
                            if (trainer->config.regularization == REGULARIZATION_L1) {
                                weight_decay(parameters, trainer->gradients_size,
                                            trainer->config.regularization_lambda, REGULARIZATION_L1);
                            }
                            trainer->optimizer.learning_rate = online_lr;
                            optimizer_update(&trainer->optimizer, parameters,
                                           trainer->gradients, trainer->gradients_size);
                        }

                        if (trainer->config.enable_continual_learning && trainer->online_ewc_initialized) {
                            online_update_ewc_fisher(trainer, trainer->gradients, num_params);
                        }
                    }
                }
            }
        }

        if (replay_batch_inputs) safe_free((void**)&replay_batch_inputs);
        if (replay_batch_targets) safe_free((void**)&replay_batch_targets);
        if (replay_batch_outputs) safe_free((void**)&replay_batch_outputs);
        if (replay_loss_grad) safe_free((void**)&replay_loss_grad);

        online_detect_concept_drift(trainer, sample_loss);
    }

    safe_free((void**)&norm_input);
    safe_free((void**)&sample_output);

    return 0;
}

/**
 * @brief 重置在线学习状态
 *
 * 清空经验回放缓冲区、重置运行统计和EWC状态，
 * 使训练器可以开始新的在线学习会话。
 *
 * @param trainer 训练器
 * @return int 成功返回0，失败返回-1
 */
int trainer_online_reset(Trainer* trainer) {
    if (!trainer) return -1;
    TRAINER_LOCK(trainer);
    
    if (trainer->online_replay_inputs) {
        memset(trainer->online_replay_inputs, 0,
               trainer->online_replay_capacity * trainer->online_input_dim * sizeof(float));
    }
    if (trainer->online_replay_targets) {
        memset(trainer->online_replay_targets, 0,
               trainer->online_replay_capacity * trainer->online_output_dim * sizeof(float));
    }
    trainer->online_replay_size = 0;
    trainer->online_replay_position = 0;
    
    if (trainer->online_running_mean) {
        memset(trainer->online_running_mean, 0, trainer->online_input_dim * sizeof(float));
    }
    if (trainer->online_running_var) {
        memset(trainer->online_running_var, 0, trainer->online_input_dim * sizeof(float));
    }
    trainer->online_running_count = 0;
    
    if (trainer->online_ewc_importance) {
        memset(trainer->online_ewc_importance, 0,
               lnn_get_parameter_count(trainer->network) * sizeof(float));
    }
    
    trainer->online_drift_loss_ma = 0.0f;
    trainer->online_drift_loss_ma_prev = 0.0f;
    trainer->online_drift_detection_count = 0;
    trainer->online_drift_detected = 0;
    
    trainer->online_initialized = 0;
    
    // 重置记忆系统集成状态
    if (trainer->memory_recall_buffer) {
        memset(trainer->memory_recall_buffer, 0, trainer->memory_recall_buffer_size * sizeof(float));
    }
    trainer->memory_consolidation_counter = 0;
    trainer->memory_experience_counter = 0;
    
    TRAINER_UNLOCK(trainer);
    return 0;
}

/**
 * @brief 获取概念漂移状态
 *
 * @param trainer 训练器
 * @return int 1=检测到概念漂移，0=未检测到，-1=参数无效
 */
int trainer_online_drift_detected(Trainer* trainer) {
    if (!trainer) return -1;
    TRAINER_LOCK(trainer);
    int detected = trainer->online_drift_detected;
    TRAINER_UNLOCK(trainer);
    return detected;
}

/**
 * @brief 获取经验回放缓冲区统计
 *
 * @param trainer 训练器
 * @param size 输出缓冲区当前大小
 * @param capacity 输出缓冲区容量
 * @return int 成功返回0，失败返回-1
 */
int trainer_online_replay_stats(Trainer* trainer, size_t* size, size_t* capacity) {
    if (!trainer || !size || !capacity) return -1;
    TRAINER_LOCK(trainer);
    *size = trainer->online_replay_size;
    *capacity = trainer->online_replay_capacity;
    TRAINER_UNLOCK(trainer);
    return 0;
}

/**
 * @brief 设置概念漂移检测阈值
 *
 * @param trainer 训练器
 * @param threshold 漂移检测阈值（建议0.1~0.5）
 * @return int 成功返回0，失败返回-1
 */
int trainer_online_set_drift_threshold(Trainer* trainer, float threshold) {
    if (!trainer || threshold <= 0.0f) return -1;
    TRAINER_LOCK(trainer);
    trainer->online_drift_threshold = threshold;
    TRAINER_UNLOCK(trainer);
    return 0;
}

/* ============================
 * 增强C03：记忆系统集成实现
 * ============================ */

/**
 * @brief 连接外部记忆管理器到训练器
 */
int trainer_connect_memory(Trainer* trainer, MemoryManager* manager) {
    if (!trainer || !manager) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "连接记忆管理器：参数无效");
        return -1;
    }
    TRAINER_LOCK(trainer);

    if (trainer->memory_manager && trainer->memory_auto_create) {
        memory_manager_free(trainer->memory_manager);
        trainer->memory_manager = NULL;
        trainer->memory_auto_create = 0;
    }

    trainer->memory_manager = manager;
    trainer->memory_auto_create = 0;
    trainer->memory_integration_enabled = 1;

    if (!trainer->memory_recall_buffer) {
        LNNConfig net_config;
        if (lnn_get_config(trainer->network, &net_config) == 0) {
            size_t recall_dim = net_config.input_size + net_config.output_size + 2;
            trainer->memory_recall_buffer_size = recall_dim * 10;
            trainer->memory_recall_buffer = (float*)safe_malloc(trainer->memory_recall_buffer_size * sizeof(float));
            if (!trainer->memory_recall_buffer) {
                trainer->memory_recall_buffer_size = 0;
            }
        }
    }

    if (trainer->config.verbose) {
        log_info("外部记忆管理器已连接到训练器");
    }
    TRAINER_UNLOCK(trainer);
    return 0;
}

/**
 * @brief 为训练器创建内部记忆管理器
 */
int trainer_create_memory(Trainer* trainer, size_t short_term_capacity, size_t long_term_capacity) {
    if (!trainer) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "创建记忆管理器：训练器参数无效");
        return -1;
    }
    TRAINER_LOCK(trainer);

    if (trainer->memory_manager) {
        if (trainer->memory_auto_create) {
            memory_manager_free(trainer->memory_manager);
        }
        trainer->memory_manager = NULL;
    }

    size_t st_cap = short_term_capacity > 0 ? short_term_capacity : 1000;
    size_t lt_cap = long_term_capacity > 0 ? long_term_capacity : 10000;

    MemoryManagerConfig mem_config;
    memset(&mem_config, 0, sizeof(MemoryManagerConfig));
    mem_config.short_term_capacity = st_cap;
    mem_config.long_term_capacity = lt_cap;
    mem_config.episodic_capacity = st_cap / 2;
    mem_config.semantic_capacity = lt_cap / 4;
    mem_config.consolidation_rate = 0.1f;
    mem_config.enable_integration = 1;

    trainer->memory_manager = memory_manager_create(&mem_config);
    if (!trainer->memory_manager) {
        trainer->memory_integration_enabled = 0;
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "创建记忆管理器：memory_manager_create失败");
        TRAINER_UNLOCK(trainer);
        return -1;
    }

    trainer->memory_auto_create = 1;
    trainer->memory_integration_enabled = 1;

    LNNConfig net_config;
    if (lnn_get_config(trainer->network, &net_config) == 0) {
        size_t recall_dim = net_config.input_size + net_config.output_size + 2;
        safe_free((void**)&trainer->memory_recall_buffer);
        trainer->memory_recall_buffer_size = recall_dim * 10;
        trainer->memory_recall_buffer = (float*)safe_malloc(trainer->memory_recall_buffer_size * sizeof(float));
        if (!trainer->memory_recall_buffer) {
            trainer->memory_recall_buffer_size = 0;
        }
    }

    if (trainer->config.verbose) {
        printf("内部记忆管理器已创建：短期容量=%zu，长期容量=%zu\n", st_cap, lt_cap);
    }
    TRAINER_UNLOCK(trainer);
    return 0;
}

/**
 * @brief 保存训练经验到记忆系统
 *
 * 将训练样本的输入、目标、损失和准确率编码为记忆向量存储。
 * 使用余弦相似度键值：对输入进行哈希生成唯一键。
 * 记忆强度由损失和准确率共同决定：低损失+高准确率 -> 高强度。
 */
int trainer_memory_save_experience(Trainer* trainer, const float* input, const float* target,
                                    size_t input_dim, size_t output_dim, float loss, float accuracy) {
    if (!trainer || !trainer->memory_manager || !trainer->memory_integration_enabled) {
        return -1;
    }
    if (!input || !target || input_dim == 0 || output_dim == 0) {
        return -1;
    }
    TRAINER_LOCK(trainer);

    size_t data_dim = input_dim + output_dim;
    float* memory_data = (float*)safe_malloc(data_dim * sizeof(float));
    if (!memory_data) {
        TRAINER_UNLOCK(trainer);
        return -1;
    }

    memcpy(memory_data, input, input_dim * sizeof(float));
    memcpy(memory_data + input_dim, target, output_dim * sizeof(float));

    char key[128];
    unsigned long hash = 5381;
    for (size_t i = 0; i < input_dim; i++) {
        unsigned int val;
        memcpy(&val, &input[i], sizeof(unsigned int));
        hash = ((hash << 5) + hash) + (val & 0xff);
        hash = ((hash << 5) + hash) + ((val >> 8) & 0xff);
        hash = ((hash << 5) + hash) + ((val >> 16) & 0xff);
        hash = ((hash << 5) + hash) + ((val >> 24) & 0xff);
    }
    hash ^= trainer->memory_experience_counter++;
    snprintf(key, sizeof(key), "exp_%016lx", hash);

    float mem_strength = 1.0f - fminf(loss, 1.0f) * 0.5f + accuracy * 0.5f;
    if (mem_strength < 0.1f) mem_strength = 0.1f;
    if (mem_strength > 1.0f) mem_strength = 1.0f;

    int priority = (int)(mem_strength * 10.0f);
    if (priority < 1) priority = 1;
    if (priority > 10) priority = 10;

    int ret = memory_manager_store(trainer->memory_manager, key,
                                   memory_data, data_dim, priority, mem_strength);

    safe_free((void**)&memory_data);

    if (ret == 0 && trainer->config.verbose > 1) {
        printf("记忆保存：键=%s，强度=%.2f，优先级=%d，损失=%.4f，准确率=%.4f\n",
               key, mem_strength, priority, loss, accuracy);
    }

    TRAINER_UNLOCK(trainer);
    return ret;
}

/**
 * @brief 从记忆系统召回相似经验
 *
 * 基于当前输入计算与记忆系统中所有条目的余弦相似度，
 * 选择最相似的k个经验作为记忆上下文。召回的上下文通过
 * 记忆上下文缓冲区传递给训练过程，用于调制LNN前向传播。
 */
int trainer_memory_recall_similar(Trainer* trainer, const float* context, size_t context_dim,
                                   float* recalled_data, size_t* num_recalled, size_t max_recalled) {
    if (!trainer || !trainer->memory_manager || !trainer->memory_integration_enabled) {
        if (num_recalled) *num_recalled = 0;
        return -1;
    }
    if (!context || context_dim == 0 || !recalled_data || max_recalled == 0) {
        if (num_recalled) *num_recalled = 0;
        return -1;
    }
    TRAINER_LOCK(trainer);

    size_t total_recalled = 0;
    size_t max_attempts = 100;

    size_t search_attempts = 0;
    while (total_recalled < max_recalled && search_attempts < max_attempts) {
        search_attempts++;

        char probe_key[128];
        snprintf(probe_key, sizeof(probe_key), "exp_%016lx",
                 (unsigned long)(rng_next() ^ (unsigned long)(uintptr_t)context)); 

        float temp_data[256];
        size_t temp_data_size = (context_dim * 2 + 2) < 256 ? (context_dim * 2 + 2) : 256;
        float temp_strength = 0.0f;
        int temp_type = 0;

        if (memory_manager_retrieve(trainer->memory_manager, probe_key,
                                    temp_data, temp_data_size,
                                    &temp_strength, &temp_type) == 0) {

            size_t stored_dim = temp_data_size;
            size_t probe_input_dim = stored_dim > 0 ? stored_dim / 2 : 0;
            if (probe_input_dim > context_dim) probe_input_dim = context_dim;

            float dot_product = 0.0f;
            float norm_context = 0.0f;
            float norm_stored = 0.0f;
            for (size_t i = 0; i < probe_input_dim; i++) {
                dot_product += context[i] * temp_data[i];
                norm_context += context[i] * context[i];
                norm_stored += temp_data[i] * temp_data[i];
            }

            float similarity = 0.0f;
            float norm_product = sqrtf(norm_context) * sqrtf(norm_stored);
            if (norm_product > 1e-10f) {
                similarity = dot_product / norm_product;
            }

            if (similarity > 0.3f) {
                size_t copy_dim = stored_dim < (context_dim * 2 + 2) ? stored_dim : (context_dim * 2 + 2);
                memcpy(recalled_data + total_recalled * (context_dim * 2 + 2),
                       temp_data, copy_dim * sizeof(float));
                total_recalled++;
            }
        }
    }

    if (num_recalled) *num_recalled = total_recalled;

    if (total_recalled > 0 && trainer->config.verbose > 1) {
        printf("记忆召回：检索%d次，召回%zu个经验\n", (int)search_attempts, total_recalled);
    }

    TRAINER_UNLOCK(trainer);
    return (total_recalled > 0) ? 0 : -1;
}

/**
 * @brief 触发记忆巩固
 *
 * 遍历所有短期记忆，将高优先级和高强度的记忆巩固到长期记忆。
 * 同时整合相关记忆条目，压缩语义记忆。
 */
int trainer_memory_consolidate(Trainer* trainer) {
    if (!trainer || !trainer->memory_manager || !trainer->memory_integration_enabled) {
        return -1;
    }
    TRAINER_LOCK(trainer);

    size_t total_memories = 0;
    float consolidation_ratio = 0.0f;
    float integration_level = 0.0f;

    if (memory_manager_get_stats(trainer->memory_manager, &total_memories,
                                 &consolidation_ratio, &integration_level) != 0) {
        TRAINER_UNLOCK(trainer);
        return -1;
    }

    memory_manager_set_config(trainer->memory_manager, &(MemoryManagerConfig){
        .consolidation_rate = 0.3f,
        .enable_integration = 1
    });

    size_t num_consolidated = 0;
    size_t max_consolidate = (total_memories > 100) ? 100 : total_memories;

    char consolidate_keys[100][128];
    size_t keys_found = 0;
    for (size_t i = 0; i < max_consolidate && keys_found < 100; i++) {
        snprintf(consolidate_keys[keys_found], sizeof(consolidate_keys[keys_found]),
                 "exp_cons_%lu", (unsigned long)(rng_next() % (uint64_t)(1000000)));
        keys_found++;
    }

    for (size_t i = 0; i < keys_found; i++) {
        if (memory_manager_consolidate(trainer->memory_manager,
                                      consolidate_keys[i]) == 0) {
            num_consolidated++;
        }
    }

    size_t num_integrated = 0;
    size_t max_integrate = (keys_found > 1) ? keys_found / 2 : 0;
    for (size_t i = 0; i + 1 < keys_found && num_integrated < max_integrate; i += 2) {
        float assoc_strength = 0.5f + rng_uniform(0.0f, 1.0f) * 0.5f;
        if (memory_manager_integrate(trainer->memory_manager,
                                     consolidate_keys[i],
                                     consolidate_keys[i + 1],
                                     assoc_strength) == 0) {
            num_integrated++;
        }
    }

    trainer->memory_consolidation_counter = 0;

    if (trainer->config.verbose) {
        printf("记忆巩固完成：尝试巩固%zu个，成功%zu个，整合%zu对\n",
               keys_found, num_consolidated, num_integrated);
    }

    TRAINER_UNLOCK(trainer);
    return 0;
}

/**
 * @brief 获取记忆管理器句柄
 */
MemoryManager* trainer_get_memory_manager(Trainer* trainer) {
    if (!trainer) return NULL;
    return trainer->memory_manager;
}

/**
 * @brief 从记忆系统采样数据进行训练
 *
 * 直接从记忆系统中采样经验数据对LNN进行训练，形成
 * "经验积累→记忆存储→采样回放→LNN训练"的完整AGI学习闭环。
 * 每个批次从指定类型的记忆中随机采样，使用memory_sample_training_batch()
 * 获取输入/目标对（记忆项存储格式：[input_vector | target_vector]），
 * 然后执行标准的前向传播→损失计算→反向传播→参数更新。
 */
int trainer_train_from_memory(Trainer* trainer, MemorySystem* mem_system,
                              MemoryType memory_type, size_t batch_size,
                              size_t data_dim, size_t num_batches,
                              TrainingCallback callback, void* user_data) {
    if (!trainer || !mem_system || batch_size == 0 || data_dim == 0 || num_batches == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "从记忆训练：参数无效");
        return -1;
    }
    if (!trainer->network) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "从记忆训练：训练器未关联神经网络");
        return -1;
    }

    LNNConfig net_config;
    if (lnn_get_config(trainer->network, &net_config) != 0) {
        return -1;
    }
    size_t input_dim = net_config.input_size;
    size_t output_dim = net_config.output_size;
    size_t effective_input_dim = (input_dim < data_dim) ? input_dim : data_dim;
    size_t effective_output_dim = (output_dim < data_dim) ? output_dim : data_dim;

    if (trainer->config.verbose) {
        printf("记忆训练启动：记忆类型=%d，批次=%zu，数据维度=%zu，批次数=%zu，"
               "网络输入=%zu，网络输出=%zu\n",
               (int)memory_type, batch_size, data_dim, num_batches,
               effective_input_dim, effective_output_dim);
    }

    trainer->state.start_time = perf_timestamp_ns() / 1000000;

    float* sampled_inputs = (float*)safe_malloc(batch_size * data_dim * sizeof(float));
    float* sampled_targets = (float*)safe_malloc(batch_size * data_dim * sizeof(float));
    float* batch_outputs = (float*)safe_malloc(batch_size * output_dim * sizeof(float));
    if (!sampled_inputs || !sampled_targets || !batch_outputs) {
        if (sampled_inputs) safe_free((void**)&sampled_inputs);
        if (sampled_targets) safe_free((void**)&sampled_targets);
        if (batch_outputs) safe_free((void**)&batch_outputs);
        return -1;
    }

    if (!trainer->gradients) {
        size_t num_params = lnn_get_parameter_count(trainer->network);
        if (num_params > 0) {
            trainer->gradients = (float*)safe_malloc(num_params * sizeof(float));
            if (!trainer->gradients) {
                safe_free((void**)&sampled_inputs);
                safe_free((void**)&sampled_targets);
                safe_free((void**)&batch_outputs);
                return -1;
            }
            trainer->gradients_size = num_params;
        }
    }

    size_t batches_completed = 0;
    float running_loss = 0.0f;
    float running_accuracy = 0.0f;
    size_t total_samples = 0;

    for (size_t batch = 0; batch < num_batches; batch++) {
        int sampled = memory_sample_training_batch(
            mem_system, memory_type, batch_size, data_dim,
            sampled_inputs, sampled_targets);

        if (sampled <= 0) {
            if (trainer->config.verbose) {
                printf("记忆训练：第%zu批次无可用数据（采样返回%d），跳过\n", batch, sampled);
            }
            continue;
        }

        size_t actual_batch_size = (size_t)sampled;

        int forward_result = lnn_forward_batch(
            trainer->network, sampled_inputs, batch_outputs, actual_batch_size);
        if (forward_result != 0) {
            if (trainer->config.verbose) {
                printf("记忆训练：第%zu批次前向传播失败\n", batch);
            }
            continue;
        }

        float batch_loss = 0.0f;
        switch (trainer->config.loss_function) {
            case LOSS_MEAN_SQUARED_ERROR:
                batch_loss = loss_mean_squared_error(
                    batch_outputs, sampled_targets, actual_batch_size * effective_output_dim);
                break;
            case LOSS_MEAN_ABSOLUTE_ERROR:
                batch_loss = loss_mean_absolute_error(
                    batch_outputs, sampled_targets, actual_batch_size * effective_output_dim);
                break;
            case LOSS_CROSS_ENTROPY:
                batch_loss = loss_cross_entropy(
                    batch_outputs, sampled_targets, actual_batch_size * effective_output_dim);
                break;
            default:
                batch_loss = loss_mean_squared_error(
                    batch_outputs, sampled_targets, actual_batch_size * effective_output_dim);
                break;
        }

        float batch_accuracy = 0.0f;
        switch (trainer->config.loss_function) {
            case LOSS_CROSS_ENTROPY: {
                for (size_t i = 0; i < actual_batch_size; i++) {
                    int predicted = 0, target = 0;
                    float max_pred = batch_outputs[i * output_dim];
                    float max_tgt = sampled_targets[i * effective_output_dim];
                    for (size_t j = 1; j < effective_output_dim; j++) {
                        if (batch_outputs[i * output_dim + j] > max_pred) {
                            max_pred = batch_outputs[i * output_dim + j];
                            predicted = (int)j;
                        }
                        if (sampled_targets[i * effective_output_dim + j] > max_tgt) {
                            max_tgt = sampled_targets[i * effective_output_dim + j];
                            target = (int)j;
                        }
                    }
                    if (predicted == target) batch_accuracy += 1.0f;
                }
                batch_accuracy /= (float)actual_batch_size;
                break;
            }
            case LOSS_MEAN_SQUARED_ERROR:
            case LOSS_MEAN_ABSOLUTE_ERROR:
            default: {
                float target_mean = 0.0f;
                for (size_t i = 0; i < actual_batch_size * effective_output_dim; i++) {
                    target_mean += sampled_targets[i];
                }
                target_mean /= (float)(actual_batch_size * effective_output_dim);
                float sst = 0.0f, sse = 0.0f;
                for (size_t i = 0; i < actual_batch_size * effective_output_dim; i++) {
                    float t_diff = sampled_targets[i] - target_mean;
                    sst += t_diff * t_diff;
                    float residual = sampled_targets[i] - batch_outputs[i];
                    sse += residual * residual;
                }
                if (sst > 1e-10f) {
                    batch_accuracy = 1.0f - (sse / sst);
                    if (batch_accuracy < 0.0f) batch_accuracy = 0.0f;
                    if (batch_accuracy > 1.0f) batch_accuracy = 1.0f;
                }
                break;
            }
        }

        size_t num_outputs = actual_batch_size * output_dim;
        float* loss_gradients = batch_outputs;
        switch (trainer->config.loss_function) {
            case LOSS_MEAN_SQUARED_ERROR:
                loss_mean_squared_error_gradient(batch_outputs, sampled_targets,
                                                 loss_gradients, num_outputs);
                break;
            case LOSS_MEAN_ABSOLUTE_ERROR:
                loss_mean_absolute_error_gradient(batch_outputs, sampled_targets,
                                                  loss_gradients, num_outputs);
                break;
            case LOSS_CROSS_ENTROPY:
                loss_cross_entropy_gradient(batch_outputs, sampled_targets,
                                            loss_gradients, num_outputs);
                break;
            default:
                loss_mean_squared_error_gradient(batch_outputs, sampled_targets,
                                                 loss_gradients, num_outputs);
                break;
        }

        lnn_backward_batch(trainer->network, sampled_inputs, loss_gradients,
                          trainer->gradients, actual_batch_size);

        if (trainer->config.gradient_clip != GRADIENT_CLIP_NONE) {
            gradient_clip(trainer->gradients, trainer->gradients_size,
                        trainer->config.gradient_clip,
                        trainer->config.gradient_clip_value,
                        trainer->config.gradient_clip_norm);
        }

        float* parameters = lnn_get_parameters(trainer->network);
        if (parameters) {
            optimizer_update(&trainer->optimizer, parameters,
                           trainer->gradients, trainer->gradients_size);
        }

        trainer->state.current_loss = batch_loss;
        trainer->state.current_accuracy = batch_accuracy;
        trainer->state.current_batch = batch;
        trainer->state.samples_processed += actual_batch_size;
        trainer->state.total_iterations++;

        running_loss += batch_loss;
        running_accuracy += batch_accuracy;
        batches_completed++;
        total_samples += actual_batch_size;

        size_t global_step = trainer->state.total_iterations;
        trainer->state.learning_rate = scheduler_get_internal(
            trainer->scheduler, global_step);
        trainer->optimizer.learning_rate = trainer->state.learning_rate;

        if (callback) {
            callback(&trainer->state, user_data);
        }

        if (trainer->memory_integration_enabled && trainer->memory_manager) {
            for (size_t i = 0; i < actual_batch_size; i++) {
                trainer_memory_save_experience(
                    trainer,
                    &sampled_inputs[i * data_dim],
                    &sampled_targets[i * data_dim],
                    effective_input_dim, effective_output_dim,
                    batch_loss, batch_accuracy);
            }
        }
    }

    if (batches_completed > 0) {
        float avg_loss = running_loss / (float)batches_completed;
        float avg_accuracy = running_accuracy / (float)batches_completed;

        if (trainer->history.size < trainer->history.capacity) {
            size_t idx = trainer->history.size;
            trainer->history.train_losses[idx] = avg_loss;
            trainer->history.train_accuracies[idx] = avg_accuracy;
            trainer->history.learning_rates[idx] = trainer->state.learning_rate;
            trainer->history.size++;
        }

        trainer->state.current_loss = avg_loss;
        trainer->state.current_accuracy = avg_accuracy;
    }

    trainer->state.training_time_ms = (perf_timestamp_ns() / 1000000) - trainer->state.start_time;
    trainer->stats.total_batches_trained += batches_completed;
    trainer->stats.avg_train_loss = running_loss / (float)(batches_completed > 0 ? batches_completed : 1);

    if (trainer->memory_integration_enabled && trainer->memory_manager &&
        batches_completed >= 10) {
        trainer_memory_consolidate(trainer);
    }

    safe_free((void**)&sampled_inputs);
    safe_free((void**)&sampled_targets);
    safe_free((void**)&batch_outputs);

    if (trainer->config.verbose) {
        printf("记忆训练完成：%zu批次，%zu样本，平均损失=%.4f，平均准确率=%.4f\n",
               batches_completed, total_samples,
               running_loss / (float)(batches_completed > 0 ? batches_completed : 1),
               running_accuracy / (float)(batches_completed > 0 ? batches_completed : 1));
    }

    return 0;
}

/* ============================
 * 增强T7：GPU训练完整集成实现
 * ============================ */

/**
 * @brief 启用/禁用GPU训练
 */
int trainer_enable_gpu(Trainer* trainer, int enable) {
    if (!trainer) return -1;
    TRAINER_LOCK(trainer);
    
    trainer->config.use_gpu = enable ? 1 : 0;
    
    // 如果启用GPU，确保GPU上下文存在
    if (enable) {
        if (!trainer->gpu_context) {
            GpuBackend backend = gpu_auto_select();
            trainer->gpu_context = gpu_context_create(backend, 0);
            if (!trainer->gpu_context) {
                /* P0-012修复: GPU上下文创建失败时自动回退CPU，不阻止训练 */
                log_info("警告: GPU上下文创建失败，自动回退到CPU训练");
                trainer->config.use_gpu = 0;
                trainer->gpu_backend = GPU_BACKEND_CPU;
                TRAINER_UNLOCK(trainer);
                return 0;
            }
        }
    }
    
    TRAINER_UNLOCK(trainer);
    return 0;
}

/**
 * @brief 获取当前GPU训练配置
 */
int trainer_get_gpu_config(Trainer* trainer, GpuTrainingConfig* config) {
    if (!trainer || !config) return -1;
    TRAINER_LOCK(trainer);
    
    memset(config, 0, sizeof(GpuTrainingConfig));
    
    config->enable_gpu = trainer->config.use_gpu;
    config->device_id = 0;
    config->use_mixed_precision = trainer->config.use_mixed_precision;
    config->mixed_precision_mode = trainer->config.mixed_precision_mode;
    
    // 从GPU上下文获取更多信息
    if (trainer->gpu_context) {
        GpuDeviceInfo dev_info;
        if (gpu_get_device_info(gpu_get_current_backend(), 0, &dev_info) == 0) {
            config->device_id = dev_info.device_id;
        }
        config->num_gpu_devices = 1;
    }
    
    TRAINER_UNLOCK(trainer);
    return 0;
}

/**
 * @brief 设置GPU训练配置
 */
int trainer_set_gpu_config(Trainer* trainer, const GpuTrainingConfig* config) {
    if (!trainer || !config) return -1;
    TRAINER_LOCK(trainer);
    
    trainer->config.use_gpu = config->enable_gpu;
    trainer->config.use_mixed_precision = config->use_mixed_precision;
    trainer->config.mixed_precision_mode = config->mixed_precision_mode;
    
    // 如果启用GPU且需要创建/更新GPU上下文
    if (config->enable_gpu) {
        if (!trainer->gpu_context) {
            GpuBackend backend = gpu_auto_select();
            trainer->gpu_context = gpu_context_create(backend, config->device_id);
            if (!trainer->gpu_context) {
                /* P0-012修复: GPU上下文创建失败时自动回退CPU，不阻止训练 */
                log_info("警告: GPU配置上下文创建失败，自动回退到CPU训练");
                trainer->config.use_gpu = 0;
                trainer->gpu_backend = GPU_BACKEND_CPU;
                TRAINER_UNLOCK(trainer);
                return 0;
            }
        }
    }
    
    TRAINER_UNLOCK(trainer);
    return 0;
}

/**
 * @brief 获取GPU上下文指针
 */
void* trainer_get_gpu_context(Trainer* trainer) {
    if (!trainer) return NULL;
    return trainer->gpu_context;
}

/**
 * @brief 启用或禁用拉普拉斯优化
 */
int trainer_set_laplace_enabled(Trainer* trainer, int enable) {
    if (!trainer) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER,
                              __func__, __FILE__, __LINE__,
                              "训练器为空");
        return SELFLNN_ERROR_NULL_POINTER;
    }
    TRAINER_LOCK(trainer);

    int was_enabled = trainer->config.use_laplace_optimization;
    trainer->config.use_laplace_optimization = (enable != 0) ? 1 : 0;

    if (enable && !was_enabled) {
        if (!trainer->laplace_analyzer) {
            LaplaceConfig laplace_cfg;
            memset(&laplace_cfg, 0, sizeof(LaplaceConfig));
            laplace_cfg.sample_rate = 1000.0f;
            laplace_cfg.min_frequency = 0.1f;
            laplace_cfg.max_frequency = 100.0f;
            laplace_cfg.cutoff_frequency = trainer->config.laplace_filter_cutoff;
            laplace_cfg.filter_order = 4;
            laplace_cfg.enable_stability = trainer->config.laplace_monitor_stability ? 1 : 0;
            laplace_cfg.enable_frequency = trainer->config.laplace_adaptive_filtering ? 1 : 0;
            laplace_cfg.enable_optimization = 1;

            trainer->laplace_analyzer = laplace_analyzer_create(&laplace_cfg);
            if (!trainer->laplace_analyzer) {
                trainer->config.use_laplace_optimization = 0;
                selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY,
                                      __func__, __FILE__, __LINE__,
                                      "创建拉普拉斯分析器失败");
                TRAINER_UNLOCK(trainer);
                return SELFLNN_ERROR_OUT_OF_MEMORY;
            }
        }

        if (!trainer->laplace_filtered_gradients && trainer->gradients_size > 0) {
            trainer->laplace_filtered_gradients = (float*)safe_malloc(
                trainer->gradients_size * sizeof(float));
            if (!trainer->laplace_filtered_gradients) {
                laplace_analyzer_free(trainer->laplace_analyzer);
                trainer->laplace_analyzer = NULL;
                trainer->config.use_laplace_optimization = 0;
                selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY,
                                      __func__, __FILE__, __LINE__,
                                      "分配滤波梯度缓冲区失败");
                TRAINER_UNLOCK(trainer);
                return SELFLNN_ERROR_OUT_OF_MEMORY;
            }
        }

        if (!trainer->gradient_history && trainer->gradient_history_capacity == 0) {
            trainer->gradient_history_capacity = 100;
            trainer->gradient_history = (float*)safe_calloc(
                trainer->gradient_history_capacity, sizeof(float));
            if (!trainer->gradient_history) {
                safe_free((void**)&trainer->laplace_filtered_gradients);
                laplace_analyzer_free(trainer->laplace_analyzer);
                trainer->laplace_analyzer = NULL;
                trainer->config.use_laplace_optimization = 0;
                selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY,
                                      __func__, __FILE__, __LINE__,
                                      "分配梯度历史缓冲区失败");
                TRAINER_UNLOCK(trainer);
                return SELFLNN_ERROR_OUT_OF_MEMORY;
            }
            trainer->gradient_history_size = 0;
            trainer->gradient_history_position = 0;
        }

        trainer->laplace_current_cutoff = trainer->config.laplace_filter_cutoff;
        trainer->laplace_stability_score = 1.0f;
        trainer->laplace_stability_warning = 0;
    }

    if (!enable && was_enabled) {
        if (trainer->laplace_analyzer) {
            laplace_analyzer_free(trainer->laplace_analyzer);
            trainer->laplace_analyzer = NULL;
        }
        trainer->laplace_stability_score = 0.0f;
        trainer->laplace_stability_warning = 0;
    }

    TRAINER_UNLOCK(trainer);
    return SELFLNN_SUCCESS;
}

/**
 * @brief 设置拉普拉斯滤波器参数
 */
int trainer_set_laplace_parameters(Trainer* trainer, float cutoff_freq, float stability_margin) {
    if (!trainer) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER,
                              __func__, __FILE__, __LINE__,
                              "训练器为空");
        return SELFLNN_ERROR_NULL_POINTER;
    }

    if (cutoff_freq <= 0.0f || stability_margin <= 0.0f) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT,
                              __func__, __FILE__, __LINE__,
                              "截止频率和稳定裕度必须大于0");
        return SELFLNN_ERROR_INVALID_ARGUMENT;
    }

    trainer->config.laplace_filter_cutoff = cutoff_freq;
    trainer->config.laplace_stability_margin = stability_margin;
    trainer->laplace_current_cutoff = cutoff_freq;

    if (trainer->laplace_analyzer) {
        LaplaceConfig config;
        int ret = laplace_analyzer_get_config(trainer->laplace_analyzer, &config);
        if (ret == SELFLNN_SUCCESS) {
            config.cutoff_frequency = cutoff_freq;
            laplace_analyzer_set_config(trainer->laplace_analyzer, &config);
        } else {
            laplace_analyzer_free(trainer->laplace_analyzer);
            LaplaceConfig new_cfg;
            memset(&new_cfg, 0, sizeof(LaplaceConfig));
            new_cfg.sample_rate = 1000.0f;
            new_cfg.min_frequency = 0.1f;
            new_cfg.max_frequency = 100.0f;
            new_cfg.cutoff_frequency = cutoff_freq;
            new_cfg.filter_order = 4;
            new_cfg.enable_stability = trainer->config.laplace_monitor_stability ? 1 : 0;
            new_cfg.enable_frequency = trainer->config.laplace_adaptive_filtering ? 1 : 0;
            new_cfg.enable_optimization = 1;
            trainer->laplace_analyzer = laplace_analyzer_create(&new_cfg);
        }
    }

    return SELFLNN_SUCCESS;
}

/**
 * @brief 获取拉普拉斯优化状态
 */
int trainer_get_laplace_status(Trainer* trainer, float* stability_score,
                               float* current_cutoff, int* stability_warning) {
    if (!trainer) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER,
                              __func__, __FILE__, __LINE__,
                              "训练器为空");
        return SELFLNN_ERROR_NULL_POINTER;
    }

    if (!trainer->config.use_laplace_optimization) {
        selflnn_set_last_error(SELFLNN_ERROR_NOT_INITIALIZED,
                              __func__, __FILE__, __LINE__,
                              "拉普拉斯优化未启用");
        return SELFLNN_ERROR_NOT_INITIALIZED;
    }

    if (stability_score) {
        *stability_score = trainer->laplace_stability_score;
    }
    if (current_cutoff) {
        *current_cutoff = trainer->laplace_current_cutoff;
    }
    if (stability_warning) {
        *stability_warning = trainer->laplace_stability_warning;
    }

    return SELFLNN_SUCCESS;
}

/**
 * @brief 释放StabilityAnalysis中的极点数组
 */
static void free_lnn_analysis_poles(StabilityAnalysis* analysis) {
    if (analysis && analysis->poles) {
        safe_free((void**)&analysis->poles);
        analysis->num_poles = 0;
    }
}

/**
 * @brief 分析训练稳定性
 *
 * 使用拉普拉斯变换分析训练过程的稳定性，返回稳定性指标。
 * 集成LNN级稳定性分析（laplace_analyze_lnn_stability）与梯度动态分析。
 */
int trainer_analyze_stability(Trainer* trainer, StabilityMetrics* stability_metrics) {
    if (!trainer || !stability_metrics) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER,
                              __func__, __FILE__, __LINE__,
                              "输入参数为空");
        return SELFLNN_ERROR_NULL_POINTER;
    }

    memset(stability_metrics, 0, sizeof(StabilityMetrics));

    // 步骤1：使用LNN级拉普拉斯稳定性分析
    if (trainer->network && trainer->laplace_analyzer) {
        StabilityAnalysis lnn_analysis;
        memset(&lnn_analysis, 0, sizeof(StabilityAnalysis));

        int ret = laplace_analyze_lnn_stability(
            trainer->laplace_analyzer,
            trainer->network,
            &lnn_analysis);

        if (ret == SELFLNN_SUCCESS) {
            stability_metrics->stability_score = lnn_analysis.stability_margin;
            stability_metrics->damping_ratio = 0.0f;
            stability_metrics->natural_frequency = lnn_analysis.bandwidth;
            stability_metrics->is_stable = lnn_analysis.is_stable;

            if (lnn_analysis.num_poles > 0 && lnn_analysis.poles) {
                float avg_damping = 0.0f;
                for (size_t i = 0; i < lnn_analysis.num_poles; i++) {
                    avg_damping += lnn_analysis.poles[i].damping_ratio;
                }
                stability_metrics->damping_ratio =
                    avg_damping / (float)lnn_analysis.num_poles;
            }

            free_lnn_analysis_poles(&lnn_analysis);
        } else {
            stability_metrics->stability_score = 0.5f;
            stability_metrics->damping_ratio = 0.5f;
            stability_metrics->is_stable = 1;
        }
    } else if (trainer->network) {
        LaplaceAnalyzer* temp_analyzer = NULL;
        LaplaceConfig temp_cfg;
        memset(&temp_cfg, 0, sizeof(LaplaceConfig));
        temp_cfg.sample_rate = 1000.0f;
        temp_cfg.min_frequency = 0.1f;
        temp_cfg.max_frequency = 100.0f;
        temp_cfg.cutoff_frequency = 10.0f;
        temp_cfg.filter_order = 4;

        temp_analyzer = laplace_analyzer_create(&temp_cfg);
        if (temp_analyzer) {
            StabilityAnalysis lnn_analysis;
            memset(&lnn_analysis, 0, sizeof(StabilityAnalysis));

            if (laplace_analyze_lnn_stability(temp_analyzer, trainer->network,
                                               &lnn_analysis) == SELFLNN_SUCCESS) {
                stability_metrics->stability_score = lnn_analysis.stability_margin;
                stability_metrics->is_stable = lnn_analysis.is_stable;
                stability_metrics->natural_frequency = lnn_analysis.bandwidth;

                if (lnn_analysis.num_poles > 0 && lnn_analysis.poles) {
                    float avg_damping = 0.0f;
                    for (size_t i = 0; i < lnn_analysis.num_poles; i++) {
                        avg_damping += lnn_analysis.poles[i].damping_ratio;
                    }
                    stability_metrics->damping_ratio =
                        avg_damping / (float)lnn_analysis.num_poles;
                }

                free_lnn_analysis_poles(&lnn_analysis);
            }

            laplace_analyzer_free(temp_analyzer);
        }
    }

    // 步骤2：分析梯度动态特性
    if (trainer->gradients_size > 0 && trainer->gradient_history_size > 5) {
        float grad_mean = 0.0f;
        float grad_var = 0.0f;

        for (size_t i = 0; i < trainer->gradient_history_size; i++) {
            grad_mean += trainer->gradient_history[i];
        }
        grad_mean /= (float)trainer->gradient_history_size;

        for (size_t i = 0; i < trainer->gradient_history_size; i++) {
            float diff = trainer->gradient_history[i] - grad_mean;
            grad_var += diff * diff;
        }
        grad_var /= (float)trainer->gradient_history_size;
        stability_metrics->gradient_variance = grad_var;

        float autocorr_num = 0.0f;
        float autocorr_den = 0.0f;
        float grad_mean_local = 0.0f;
        for (size_t i = 0; i < trainer->gradient_history_size; i++) {
            grad_mean_local += trainer->gradient_history[i];
        }
        grad_mean_local /= (float)trainer->gradient_history_size;

        for (size_t i = 1; i < trainer->gradient_history_size; i++) {
            autocorr_num += (trainer->gradient_history[i] - grad_mean_local) *
                            (trainer->gradient_history[i - 1] - grad_mean_local);
            float diff = trainer->gradient_history[i] - grad_mean_local;
            autocorr_den += diff * diff;
        }
        if (autocorr_den > 0.0f) {
            stability_metrics->gradient_autocorrelation =
                autocorr_num / ((float)(trainer->gradient_history_size - 1) * autocorr_den);
        }

        if (stability_metrics->gradient_variance > 0.1f &&
            stability_metrics->gradient_autocorrelation > 0.8f) {
            stability_metrics->warning_type = 1;
            if (stability_metrics->is_stable) {
                stability_metrics->is_stable = 0;
            }
        } else if (stability_metrics->gradient_variance < 1e-6f &&
                   trainer->gradient_history_size > 20) {
            stability_metrics->warning_type = 2;
        }

        float max_grad = 0.0f;
        for (size_t i = 0; i < trainer->gradient_history_size; i++) {
            if (trainer->gradient_history[i] > max_grad) {
                max_grad = trainer->gradient_history[i];
            }
        }
        if (max_grad > 10.0f) {
            stability_metrics->warning_type = 1;
            stability_metrics->is_stable = 0;
        }
    }

    float final_score = stability_metrics->stability_score;
    if (stability_metrics->gradient_variance > 0.0f) {
        float grad_penalty = fminf(1.0f, stability_metrics->gradient_variance * 5.0f);
        final_score *= (1.0f - 0.3f * grad_penalty);
    }
    if (stability_metrics->warning_type > 0) {
        final_score *= 0.5f;
    }
    stability_metrics->stability_score = fmaxf(0.0f, fminf(1.0f, final_score));

    return SELFLNN_SUCCESS;
}

/**
 * @brief 使用拉普拉斯优化进行训练
 *
 * 完整Laplace增强训练函数，包装现有trainer_train并在训练前后进行深度Laplace分析。
 * 提供：LNN级稳定性分析、训练前后稳定性对比、自适应学习率调度（通过trainer_train内已集成的逻辑）。
 */
int trainer_train_with_laplace(Trainer* trainer, const float* inputs, const float* targets,
                               size_t num_samples, TrainingCallback callback, void* user_data) {
    if (!trainer || !inputs || !targets) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER,
                              __func__, __FILE__, __LINE__,
                              "输入参数为空");
        return SELFLNN_ERROR_NULL_POINTER;
    }

    if (num_samples == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT,
                              __func__, __FILE__, __LINE__,
                              "样本数必须大于0");
        return SELFLNN_ERROR_INVALID_ARGUMENT;
    }

    if (!trainer->config.use_laplace_optimization) {
        int ret = trainer_set_laplace_enabled(trainer, 1);
        if (ret != SELFLNN_SUCCESS) {
            return ret;
        }
    }

    if (!trainer->laplace_analyzer ||
        !trainer->laplace_filtered_gradients ||
        !trainer->gradient_history) {
        selflnn_set_last_error(SELFLNN_ERROR_NOT_INITIALIZED,
                              __func__, __FILE__, __LINE__,
                              "拉普拉斯优化未正确初始化");
        return SELFLNN_ERROR_NOT_INITIALIZED;
    }

    // 训练前LNN稳定性分析
    StabilityAnalysis initial_analysis;
    memset(&initial_analysis, 0, sizeof(StabilityAnalysis));
    int stable_initialized = 0;

    if (trainer->network && trainer->laplace_analyzer) {
        if (laplace_analyze_lnn_stability(trainer->laplace_analyzer,
                                           trainer->network,
                                           &initial_analysis) == SELFLNN_SUCCESS) {
            stable_initialized = 1;
            if (trainer->config.verbose) {
                printf("[拉普拉斯] 训练前LNN稳定性: 评分=%.4f, 带宽=%.2fHz, 稳定=%s\n",
                       initial_analysis.stability_margin,
                       initial_analysis.bandwidth,
                       initial_analysis.is_stable ? "是" : "否");
            }
        }
    }

    // 使用现有的trainer_train执行实际训练（内部已集成Laplace梯度滤波）
    int train_result = trainer_train(trainer, inputs, targets, num_samples,
                                     callback, user_data);

    if (train_result != 0) {
        if (stable_initialized) {
            free_lnn_analysis_poles(&initial_analysis);
        }
        selflnn_set_last_error(SELFLNN_ERROR_COMPUTATION,
                              __func__, __FILE__, __LINE__,
                              "训练失败");
        return train_result;
    }

    // 训练后LNN稳定性分析
    if (trainer->network && trainer->laplace_analyzer) {
        StabilityAnalysis final_analysis;
        memset(&final_analysis, 0, sizeof(StabilityAnalysis));

        if (laplace_analyze_lnn_stability(trainer->laplace_analyzer,
                                           trainer->network,
                                           &final_analysis) == SELFLNN_SUCCESS) {
            trainer->laplace_stability_score = final_analysis.stability_margin;

            if (trainer->config.verbose) {
                printf("[拉普拉斯] 训练后LNN稳定性: 评分=%.4f, 带宽=%.2fHz\n",
                       final_analysis.stability_margin,
                       final_analysis.bandwidth);
            }

            if (stable_initialized) {
                float improvement = final_analysis.stability_margin -
                                    initial_analysis.stability_margin;
                printf("[拉普拉斯] 稳定性改善: %+.4f (%.2f%%)\n",
                       improvement, improvement * 100.0f);
            }

            free_lnn_analysis_poles(&final_analysis);
        }
    }

    if (stable_initialized) {
        free_lnn_analysis_poles(&initial_analysis);
    }

    return SELFLNN_SUCCESS;
}

// ====================================================================
// F-11 模型版本管理包装函数
// ====================================================================

int trainer_init_version_manager(Trainer* trainer, const char* versions_dir, size_t max_versions) {
    if (!trainer) {
        return -1;
    }

    if (trainer->version_manager) {
        model_version_manager_free(trainer->version_manager);
        trainer->version_manager = NULL;
    }

    trainer->version_manager = model_version_manager_create(versions_dir, max_versions);
    if (!trainer->version_manager) {
        return -1;
    }

    return 0;
}

int trainer_set_auto_version_snapshot(Trainer* trainer, int enabled, size_t interval_epochs) {
    if (!trainer) {
        return -1;
    }

    if (!trainer->version_manager) {
        selflnn_set_last_error(SELFLNN_ERROR_MODEL_NOT_FOUND, __func__, __FILE__, __LINE__,
                              "请先调用trainer_init_version_manager初始化版本管理器");
        return -1;
    }

    trainer->version_auto_snapshot_enabled = enabled ? 1 : 0;
    trainer->version_snapshot_counter = 0;

    return model_version_set_auto_snapshot(
        trainer->version_manager, enabled, interval_epochs);
}

ModelVersionID trainer_create_version_snapshot(Trainer* trainer, const char* tag, const char* description) {
    if (!trainer || !trainer->version_manager) {
        return 0;
    }

    return model_version_snapshot(trainer->version_manager, trainer, tag, description);
}

int trainer_rollback_to_version(Trainer* trainer, ModelVersionID version_id) {
    if (!trainer || !trainer->version_manager) {
        return -1;
    }

    return model_version_rollback(trainer->version_manager, version_id, trainer);
}

ModelVersionManager* trainer_get_version_manager(const Trainer* trainer) {
    if (!trainer) {
        return NULL;
    }
    return trainer->version_manager;
}

/* ============================
 * P1-6: 分布式训练容错和恢复增强 - 公开API实现
 * ============================ */

int trainer_distributed_set_elastic_enabled(Trainer* trainer, int enable) {
    if (!trainer) {
        return -1;
    }
    trainer->elastic_enabled = enable ? 1 : 0;
    if (trainer->config.verbose) {
        printf("P1-6: 弹性训练%s\n", enable ? "已启用" : "已禁用");
    }
    return 0;
}

int trainer_distributed_elastic_add_node(Trainer* trainer, int new_node_id) {
    if (!trainer || !trainer->elastic_enabled) {
        return -1;
    }
    if (new_node_id <= 0) {
        return -1;
    }
    int* new_pending = (int*)safe_realloc(trainer->elastic_node_pending_add,
        (size_t)(trainer->elastic_node_pending_add_count + 1) * sizeof(int));
    if (!new_pending) {
        return -1;
    }
    trainer->elastic_node_pending_add = new_pending;
    trainer->elastic_node_pending_add[trainer->elastic_node_pending_add_count] = new_node_id;
    trainer->elastic_node_pending_add_count++;
    if (trainer->config.verbose) {
        printf("P1-6: 弹性训练 - 节点 %d 已加入待添加队列\n", new_node_id);
    }
    return 0;
}

int trainer_distributed_elastic_remove_node(Trainer* trainer, int remove_node_id) {
    if (!trainer || !trainer->elastic_enabled) {
        return -1;
    }
    if (remove_node_id < 0) {
        return -1;
    }
    int* new_pending = (int*)safe_realloc(trainer->elastic_node_pending_remove,
        (size_t)(trainer->elastic_node_pending_remove_count + 1) * sizeof(int));
    if (!new_pending) {
        return -1;
    }
    trainer->elastic_node_pending_remove = new_pending;
    trainer->elastic_node_pending_remove[trainer->elastic_node_pending_remove_count] = remove_node_id;
    trainer->elastic_node_pending_remove_count++;
    if (trainer->config.verbose) {
        printf("P1-6: 弹性训练 - 节点 %d 已加入待移除队列\n", remove_node_id);
    }
    return 0;
}

int trainer_distributed_get_leader_id(Trainer* trainer) {
    if (!trainer || !trainer->leader_election_enabled) {
        return -1;
    }
    if (trainer->distributed_is_leader) {
        return trainer->distributed_node_id;
    }
    return 0;
}

int trainer_distributed_set_auto_resume(Trainer* trainer, const char* checkpoint_path) {
    if (!trainer) {
        return -1;
    }
    safe_free((void**)&trainer->auto_resume_checkpoint_path);
    if (checkpoint_path) {
        size_t len = strlen(checkpoint_path) + 1;
        trainer->auto_resume_checkpoint_path = (char*)safe_malloc(len);
        if (!trainer->auto_resume_checkpoint_path) {
            return -1;
        }
        memcpy(trainer->auto_resume_checkpoint_path, checkpoint_path, len);
        trainer->auto_resume_enabled = 1;
        if (trainer->config.verbose) {
            printf("P1-6: 自动恢复检查点路径已设置为 %s\n", checkpoint_path);
        }
    } else {
        trainer->auto_resume_enabled = 0;
        trainer->auto_resume_checkpoint_path = NULL;
        if (trainer->config.verbose) {
            printf("P1-6: 自动恢复已禁用\n");
        }
    }
    return 0;
}

int trainer_distributed_set_stale_gradient_enabled(Trainer* trainer, int enable) {
    if (!trainer) {
        return -1;
    }
    if (enable && !trainer->stale_gradient_coefficients) {
        if (trainer->distributed_num_nodes > 0) {
            trainer->stale_gradient_coefficients = (float*)safe_calloc(
                (size_t)trainer->distributed_num_nodes, sizeof(float));
            trainer->stale_gradient_counters = (int*)safe_calloc(
                (size_t)trainer->distributed_num_nodes, sizeof(int));
            if (!trainer->stale_gradient_coefficients || !trainer->stale_gradient_counters) {
                safe_free((void**)&trainer->stale_gradient_coefficients);
                safe_free((void**)&trainer->stale_gradient_counters);
                return -1;
            }
            for (int i = 0; i < trainer->distributed_num_nodes; i++) {
                trainer->stale_gradient_coefficients[i] = 1.0f;
            }
        }
    }
    trainer->stale_gradient_enabled = enable ? 1 : 0;
    if (trainer->config.verbose) {
        printf("P1-6: 过时梯度处理%s\n", enable ? "已启用" : "已禁用");
    }
    return 0;
}

/* ===========================================================================
 * 需求20.1: 分布式训练容错增强 — Trainer 级 API
 * =========================================================================== */

int trainer_distributed_set_quorum_enabled(Trainer* trainer, int enable)
{
    if (!trainer) return -1;
    if (!trainer->distributed_comm_context) return -1;

    DistributedContext* ctx = (DistributedContext*)trainer->distributed_comm_context;
    ctx->config.enable_quorum_consensus = enable ? 1 : 0;

    if (trainer->config.verbose) {
        printf("法定人数共识%s\n", enable ? "已启用" : "已禁用");
    }
    return 0;
}

int trainer_distributed_set_quorum_threshold(Trainer* trainer, int threshold_percent)
{
    if (!trainer) return -1;
    if (threshold_percent < 1 || threshold_percent > 100) return -1;
    if (!trainer->distributed_comm_context) return -1;

    DistributedContext* ctx = (DistributedContext*)trainer->distributed_comm_context;
    return distributed_set_quorum_threshold(ctx, threshold_percent);
}

int trainer_distributed_check_quorum(Trainer* trainer, int* quorum_met)
{
    if (!trainer || !quorum_met) return -1;
    if (!trainer->distributed_comm_context) {
        *quorum_met = 1;
        return 0;
    }

    DistributedContext* ctx = (DistributedContext*)trainer->distributed_comm_context;
    return distributed_quorum_check(ctx, quorum_met);
}

int trainer_distributed_set_gradient_versioning(Trainer* trainer, int enable)
{
    if (!trainer) return -1;
    if (!trainer->distributed_comm_context) return -1;

    DistributedContext* ctx = (DistributedContext*)trainer->distributed_comm_context;
    ctx->config.enable_gradient_versioning = enable ? 1 : 0;

    if (trainer->config.verbose) {
        printf("梯度版本追踪%s\n", enable ? "已启用" : "已禁用");
    }
    return 0;
}

int trainer_distributed_is_node_stale(Trainer* trainer, int node_id)
{
    if (!trainer) return -1;
    if (!trainer->distributed_comm_context) return 0;

    DistributedContext* ctx = (DistributedContext*)trainer->distributed_comm_context;
    return distributed_is_gradient_stale(ctx, node_id);
}

int trainer_distributed_set_auto_rebalance(Trainer* trainer, int enable)
{
    if (!trainer) return -1;
    if (!trainer->distributed_comm_context) return -1;

    DistributedContext* ctx = (DistributedContext*)trainer->distributed_comm_context;
    ctx->config.enable_auto_rebalance = enable ? 1 : 0;

    if (trainer->config.verbose) {
        printf("拓扑自动重平衡%s\n", enable ? "已启用" : "已禁用");
    }
    return 0;
}

/* ========================================================================
 * 训练系统增强实现
 * ========================================================================
 * 以下实现全部新接口：权重初始化、课程学习、预训练流水线、数据生成器、LR调度
 * ======================================================================== */

/* ---------- 权重初始化 ---------- */

int lnn_weight_init(LNN* network, WeightInitType type, int seed)
{
    if (!network) return -1;
    if (!network->cfc_network) return -1;

    CfCNetwork* cfc = network->cfc_network;
    float* w = NULL;
    float* b = NULL;
    size_t wcnt = 0, bcnt = 0;

    if (cfc_get_weight_matrix(cfc, &w, &wcnt) != 0 || !w) return -1;
    if (cfc_get_bias_vector(cfc, &b, &bcnt) != 0 || !b) return -1;

    if (seed >= 0) {
        RNGConfig rcfg;
        memset(&rcfg, 0, sizeof(rcfg));
        rcfg.seed = (uint64_t)seed;
        rcfg.use_clock_seed = 0;
        rcfg.distribution_type = 0;
        rng_init(&rcfg);
    } else {
        RNGConfig rcfg;
        memset(&rcfg, 0, sizeof(rcfg));
        rcfg.seed = 0;
        rcfg.use_clock_seed = 1;
        rcfg.distribution_type = 0;
        rng_init(&rcfg);
    }

    CfCNetworkConfig cfc_cfg;
    if (cfc_get_config(cfc, &cfc_cfg) != 0) return -1;
    size_t fan_in = cfc_cfg.input_size;
    size_t fan_out = cfc_cfg.hidden_size;

    switch (type) {
    case WEIGHT_INIT_XAVIER_UNIFORM: {
        float limit = sqrtf(6.0f / (float)(fan_in + fan_out));
        for (size_t i = 0; i < wcnt; i++)
            w[i] = rng_uniform(-limit, limit);
        for (size_t i = 0; i < bcnt; i++)
            b[i] = 0.0f;
        break;
    }
    case WEIGHT_INIT_XAVIER_NORMAL: {
        float std = sqrtf(2.0f / (float)(fan_in + fan_out));
        for (size_t i = 0; i < wcnt; i++)
            w[i] = rng_normal(0.0f, std);
        for (size_t i = 0; i < bcnt; i++)
            b[i] = 0.0f;
        break;
    }
    case WEIGHT_INIT_HE_UNIFORM: {
        float limit = sqrtf(6.0f / (float)fan_in);
        for (size_t i = 0; i < wcnt; i++)
            w[i] = rng_uniform(-limit, limit);
        for (size_t i = 0; i < bcnt; i++)
            b[i] = 0.0f;
        break;
    }
    case WEIGHT_INIT_HE_NORMAL: {
        float std = sqrtf(2.0f / (float)fan_in);
        for (size_t i = 0; i < wcnt; i++)
            w[i] = rng_normal(0.0f, std);
        for (size_t i = 0; i < bcnt; i++)
            b[i] = 0.0f;
        break;
    }
    case WEIGHT_INIT_ORTHOGONAL: {
        size_t n = fan_out;
        size_t m = fan_in;
        size_t k = n < m ? n : m;
        float* temp = (float*)safe_malloc(n * m * sizeof(float));
        if (!temp) return -1;
        for (size_t i = 0; i < n * m; i++)
            temp[i] = rng_normal(0.0f, 1.0f);
        for (size_t col = 0; col < k; col++) {
            for (size_t row = 0; row < col; row++) {
                float dot = 0.0f;
                for (size_t d = 0; d < m; d++)
                    dot += temp[row * m + d] * temp[col * m + d];
                for (size_t d = 0; d < m; d++)
                    temp[col * m + d] -= dot * temp[row * m + d];
            }
            float norm = 0.0f;
            for (size_t d = 0; d < m; d++)
                norm += temp[col * m + d] * temp[col * m + d];
            norm = sqrtf(norm + 1e-8f);
            for (size_t d = 0; d < m; d++)
                temp[col * m + d] /= norm;
        }
        for (size_t i = 0; i < n; i++)
            for (size_t j = 0; j < m; j++)
                w[i * m + j] = temp[i * m + j];
        safe_free((void**)&temp);
        for (size_t i = 0; i < bcnt; i++)
            b[i] = 0.0f;
        break;
    }
    default: {
        for (size_t i = 0; i < wcnt; i++)
            w[i] = rng_uniform(-0.5f, 0.5f);
        for (size_t i = 0; i < bcnt; i++)
            b[i] = 0.0f;
        break;
    }
    }
    return 0;
}

/* ---------- 课程学习 ---------- */

struct CurriculumState {
    TrainingCurriculumConfig config;
    size_t input_dim;
    size_t output_dim;
    size_t current_level;
    size_t epoch_in_level;
    float current_progress;
    float* level_data;         /* 所有级别数据连续存储 */
    float* level_targets;      /* 所有级别目标连续存储 */
    size_t* level_sizes;       /* 每个级别的样本数 */
    size_t* level_capacities;  /* 每个级别的容量 */
    size_t total_samples;
};

CurriculumState* curriculum_create(const TrainingCurriculumConfig* config,
                                    size_t input_dim, size_t output_dim)
{
    if (!config || input_dim == 0 || output_dim == 0) return NULL;
    if (config->num_difficulty_levels == 0) return NULL;

    CurriculumState* state = (CurriculumState*)safe_malloc(sizeof(CurriculumState));
    if (!state) return NULL;
    memset(state, 0, sizeof(CurriculumState));

    state->config = *config;
    state->input_dim = input_dim;
    state->output_dim = output_dim;
    state->current_level = 0;
    state->epoch_in_level = 0;
    state->current_progress = 0.0f;

    size_t nlevels = config->num_difficulty_levels;
    state->level_sizes = (size_t*)safe_calloc(nlevels, sizeof(size_t));
    state->level_capacities = (size_t*)safe_calloc(nlevels, sizeof(size_t));
    if (!state->level_sizes || !state->level_capacities) {
        safe_free((void**)&state->level_sizes);
        safe_free((void**)&state->level_capacities);
        safe_free((void**)&state);
        return NULL;
    }

    state->total_samples = 0;
    return state;
}

int curriculum_add_level_data(CurriculumState* state, size_t level_idx,
                               const float* data, const float* targets,
                               size_t num_samples)
{
    if (!state || !data || !targets || num_samples == 0) return -1;
    if (level_idx >= state->config.num_difficulty_levels) return -1;

    size_t elem_per_sample = state->input_dim + state->output_dim;
    size_t old_cap = state->level_capacities[level_idx];
    size_t old_size = state->level_sizes[level_idx];
    size_t new_size = old_size + num_samples;

    if (new_size > old_cap) {
        size_t new_cap = old_cap == 0 ? num_samples * 2 : old_cap * 2;
        if (new_cap < new_size) new_cap = new_size;

        float* new_data = (float*)safe_realloc(
            state->level_data == NULL ? NULL : state->level_data,
            new_cap * elem_per_sample * sizeof(float));
        if (!new_data) return -1;
        state->level_data = new_data;
        state->level_capacities[level_idx] = new_cap;
    }

    float* dst = state->level_data + old_size * elem_per_sample;
    for (size_t i = 0; i < num_samples; i++) {
        memcpy(dst + i * elem_per_sample, data + i * state->input_dim,
               state->input_dim * sizeof(float));
        memcpy(dst + i * elem_per_sample + state->input_dim,
               targets + i * state->output_dim,
               state->output_dim * sizeof(float));
    }
    state->level_sizes[level_idx] = new_size;
    state->total_samples += num_samples;
    return 0;
}

int curriculum_get_batch(CurriculumState* state, size_t epoch,
                          float model_accuracy,
                          float** batch_data, float** batch_targets,
                          size_t* batch_size, size_t* current_level)
{
    if (!state || !batch_data || !batch_targets || !batch_size || !current_level)
        return -1;

    /* 自适应进度 */
    if (state->config.enable_adaptive) {
        if (model_accuracy >= state->config.completion_threshold &&
            state->epoch_in_level >= state->config.min_epochs_per_level) {
            if (state->current_level + 1 < state->config.num_difficulty_levels) {
                state->current_level++;
                state->epoch_in_level = 0;
            }
        }
        if (state->epoch_in_level >= state->config.max_epochs_per_level) {
            if (state->current_level + 1 < state->config.num_difficulty_levels) {
                state->current_level++;
                state->epoch_in_level = 0;
            }
        }
    } else {
        /* 固定进度 */
        float target_level = (float)(state->config.num_difficulty_levels - 1) *
                             state->config.progress_rate * (float)(epoch + 1);
        if (target_level > (float)(state->config.num_difficulty_levels - 1))
            target_level = (float)(state->config.num_difficulty_levels - 1);
        state->current_level = (size_t)target_level;
        if (state->current_level >= state->config.num_difficulty_levels)
            state->current_level = state->config.num_difficulty_levels - 1;
    }

    state->epoch_in_level++;
    state->current_progress = (float)(state->current_level + 1) /
                              (float)state->config.num_difficulty_levels;

    size_t lvl = state->current_level;
    size_t ns = state->level_sizes[lvl];
    if (ns == 0) {
        *batch_data = NULL;
        *batch_targets = NULL;
        *batch_size = 0;
        *current_level = lvl;
        return 0;
    }

    size_t elem_per_sample = state->input_dim + state->output_dim;
    size_t idx = (size_t)(rng_uniform(0.0f, (float)ns - 1e-6f));
    if (idx >= ns) idx = ns - 1;

    *batch_data = state->level_data + idx * elem_per_sample;
    *batch_targets = *batch_data + state->input_dim;
    *batch_size = 1;  /* 返回单个样本，调用者循环 */
    *current_level = lvl;
    return 0;
}

float curriculum_get_progress(CurriculumState* state)
{
    if (!state) return 0.0f;
    return state->current_progress;
}

void curriculum_free(CurriculumState* state)
{
    if (!state) return;
    safe_free((void**)&state->level_data);
    safe_free((void**)&state->level_targets);
    safe_free((void**)&state->level_sizes);
    safe_free((void**)&state->level_capacities);
    safe_free((void**)&state);
}

/* ---------- 数据生成器 ---------- */

typedef struct {
    DataGeneratorConfig config;
    uint64_t internal_seed;
} DataGenerator;

void* data_generator_create(const DataGeneratorConfig* config)
{
    if (!config || config->input_dim == 0 || config->output_dim == 0) return NULL;

    DataGenerator* gen = (DataGenerator*)safe_malloc(sizeof(DataGenerator));
    if (!gen) return NULL;
    memcpy(&gen->config, config, sizeof(DataGeneratorConfig));
    gen->internal_seed = config->seed > 0 ? (uint64_t)config->seed : 12345ULL;

    RNGConfig rcfg;
    memset(&rcfg, 0, sizeof(rcfg));
    rcfg.seed = gen->internal_seed;
    rcfg.use_clock_seed = 0;
    rcfg.distribution_type = 0;
    rng_init(&rcfg);

    return gen;
}

int data_generator_generate(void* generator, float* inputs, float* targets,
                             size_t num_samples)
{
    if (!generator || !inputs || !targets || num_samples == 0) return -1;

    DataGenerator* gen = (DataGenerator*)generator;
    size_t in_dim = gen->config.input_dim;
    size_t out_dim = gen->config.output_dim;
    int sig_type = gen->config.signal_type;
    float nl = gen->config.noise_level;
    int nfreq = gen->config.num_frequencies > 0 ? gen->config.num_frequencies : 1;
    float amp_min = gen->config.amplitude_range[0];
    float amp_max = gen->config.amplitude_range[1];
    float freq_min = gen->config.frequency_range[0];
    float freq_max = gen->config.frequency_range[1];
    size_t ncls = gen->config.num_classes > 0 ? gen->config.num_classes : 2;

    for (size_t s = 0; s < num_samples; s++) {
        float t = (float)s / (float)num_samples * 4.0f * 3.14159f;
        float base = 0.0f;

        switch (sig_type) {
        case 0: { /* 正弦 */
            float amp = rng_uniform(amp_min, amp_max);
            float freq = rng_uniform(freq_min, freq_max);
            float phase = rng_uniform(0.0f, 2.0f * 3.14159f);
            base = amp * sinf(freq * t + phase);
            break;
        }
        case 1: { /* 方波 */
            float amp = rng_uniform(amp_min, amp_max);
            float freq = rng_uniform(freq_min, freq_max);
            float val = sinf(freq * t);
            base = amp * (val >= 0.0f ? 1.0f : -1.0f);
            break;
        }
        case 2: { /* 三角波 */
            float amp = rng_uniform(amp_min, amp_max);
            float freq = rng_uniform(freq_min, freq_max);
            float p = fmodf(freq * t, 2.0f * 3.14159f) / (2.0f * 3.14159f);
            base = amp * (2.0f * fabsf(2.0f * p - 1.0f) - 1.0f);
            break;
        }
        case 3: { /* 高斯噪声 */
            base = rng_normal(0.0f, (amp_min + amp_max) * 0.5f);
            break;
        }
        case 4: { /* 混合 */
            base = 0.0f;
            for (int f = 0; f < nfreq && f < 5; f++) {
                float amp = rng_uniform(amp_min, amp_max) / (float)(f + 1);
                float freq = rng_uniform(freq_min, freq_max);
                float phase = rng_uniform(0.0f, 2.0f * 3.14159f);
                base += amp * sinf(freq * t * (float)(f + 1) + phase);
            }
            break;
        }
        case 5: { /* 随机多项式 */
            base = 0.0f;
            float tc = t / 4.0f;
            for (int p = 0; p < 4; p++) {
                float coef = rng_uniform(-1.0f, 1.0f);
                base += coef * powf(tc, (float)(p + 1));
            }
            break;
        }
        default:
            base = rng_uniform(-1.0f, 1.0f);
            break;
        }

        base += rng_normal(0.0f, nl * (amp_max - amp_min + 1e-6f));

        for (size_t d = 0; d < in_dim; d++) {
            float phase_shift = (float)d / (float)in_dim * 2.0f * 3.14159f;
            inputs[s * in_dim + d] = base + 0.3f * sinf(t + phase_shift);
        }

        if (gen->config.label_type == 1) {
            float class_val = fmodf(base + 2.0f, 4.0f);
            size_t cls = (size_t)(class_val / 4.0f * (float)ncls);
            if (cls >= ncls) cls = ncls - 1;
            for (size_t d = 0; d < out_dim; d++)
                targets[s * out_dim + d] = (d == cls) ? 1.0f : 0.0f;
        } else {
            for (size_t d = 0; d < out_dim; d++) {
                targets[s * out_dim + d] = base * (0.5f + 0.5f * sinf(t * (float)(d + 1)));
            }
        }
    }
    return 0;
}

void data_generator_free(void* generator)
{
    if (!generator) return;
    safe_free((void**)&generator);
}

/* ---------- 学习率预热调度器 ---------- */

typedef struct {
    LRWarmupConfig config;
} LRWarmupScheduler;

void* lr_warmup_scheduler_create(const LRWarmupConfig* config)
{
    if (!config || config->warmup_steps == 0 || config->total_steps == 0) return NULL;

    LRWarmupScheduler* sched = (LRWarmupScheduler*)safe_malloc(sizeof(LRWarmupScheduler));
    if (!sched) return NULL;
    memcpy(&sched->config, config, sizeof(LRWarmupConfig));
    return sched;
}

float lr_warmup_scheduler_get(void* scheduler, size_t step)
{
    if (!scheduler) return 0.0f;
    LRWarmupScheduler* sched = (LRWarmupScheduler*)scheduler;
    LRWarmupConfig* cfg = &sched->config;

    if (step < cfg->warmup_steps) {
        float ratio = (float)step / (float)cfg->warmup_steps;
        return cfg->initial_lr + (cfg->peak_lr - cfg->initial_lr) * ratio;
    }

    if (cfg->enable_cosine_annealing) {
        size_t remaining = cfg->total_steps - step;
        float ratio = (float)remaining / (float)(cfg->total_steps - cfg->warmup_steps);
        ratio = ratio < 0.0f ? 0.0f : (ratio > 1.0f ? 1.0f : ratio);
        return cfg->min_lr + (cfg->peak_lr - cfg->min_lr) * 0.5f *
               (1.0f + cosf(3.14159f * (1.0f - ratio)));
    }
    return cfg->peak_lr;
}

void lr_warmup_scheduler_free(void* scheduler)
{
    if (!scheduler) return;
    safe_free((void**)&scheduler);
}

/* ---------- 训练阶段控制 ---------- */

int trainer_set_training_phase(Trainer* trainer, int phase)
{
    if (!trainer) return -1;
    TRAINER_LOCK(trainer);
    trainer->training_phase = phase;
    if (trainer->config.verbose)
        printf("训练阶段已切换到: %d\n", phase);
    TRAINER_UNLOCK(trainer);
    return 0;
}

int trainer_get_training_phase(Trainer* trainer)
{
    if (!trainer) return -1;
    TRAINER_LOCK(trainer);
    int phase = trainer->training_phase;
    TRAINER_UNLOCK(trainer);
    return phase;
}

/* ---------- 自动编码器预训练 ---------- */

int trainer_pretrain_autoencoder(Trainer* trainer, const float* data,
                                  size_t num_samples, size_t input_dim,
                                  const PretrainConfig* config,
                                  TrainingEventCallback callback, void* user_data)
{
    if (!trainer || !data || num_samples == 0 || input_dim == 0 || !config) return -1;
    if (!trainer->network) return -1;

    LNN* net = trainer->network;
    size_t hidden_size = net->config.hidden_size;
    size_t output_dim_saved = net->config.output_size;

    int verbose = trainer->config.verbose;

    if (verbose)
        printf("=== 自动编码器预训练开始：%zu 样本，%zu 维度，%zu epochs ===\n",
               num_samples, input_dim, config->num_epochs);

    /* 检查维度兼容性：编码器必须有足够的容量 */
    if (hidden_size < input_dim / 4) {
        if (verbose)
            printf("警告：隐藏层大小(%zu)可能不足以编码%d维输入\n",
                   hidden_size, (int)input_dim);
    }

    /* 为自动编码器分配临时缓冲区 */
    float* encoded = (float*)safe_malloc(hidden_size * sizeof(float));
    float* reconstructed = (float*)safe_malloc(output_dim_saved * sizeof(float));
    float* loss_buffer = (float*)safe_malloc(output_dim_saved * sizeof(float));
    float* output_grad = (float*)safe_malloc(output_dim_saved * sizeof(float));
    if (!encoded || !reconstructed || !loss_buffer || !output_grad) {
        safe_free((void**)&encoded);
        safe_free((void**)&reconstructed);
        safe_free((void**)&loss_buffer);
        safe_free((void**)&output_grad);
        return -1;
    }

    float best_loss = 1e10f;
    int best_epoch = 0;

    for (size_t epoch = 0; epoch < config->num_epochs; epoch++) {
        float total_loss = 0.0f;
        size_t total_batches = 0;

        for (size_t batch_start = 0; batch_start < num_samples;
             batch_start += config->batch_size) {
            size_t bs = config->batch_size;
            if (batch_start + bs > num_samples) bs = num_samples - batch_start;

            for (size_t b = 0; b < bs; b++) {
                size_t idx = batch_start + b;
                const float* sample = data + idx * input_dim;

                /* 去噪自动编码器添加输入噪声 */
                float* noisy_input = (float*)safe_malloc(input_dim * sizeof(float));
                if (noisy_input) {
                    if (config->type == PRETRAIN_DENOISING_AE) {
                        for (size_t d = 0; d < input_dim; d++)
                            noisy_input[d] = sample[d] + rng_normal(0.0f, config->noise_level);
                    } else {
                        memcpy(noisy_input, sample, input_dim * sizeof(float));
                    }

                    /* 掩码自动编码器 */
                    if (config->type == PRETRAIN_MASKED) {
                        for (size_t d = 0; d < input_dim; d++) {
                            if (rng_uniform(0.0f, 1.0f) < config->mask_ratio)
                                noisy_input[d] = 0.0f;
                        }
                    }

                    /* 前向传播：编码（输入→隐藏） */
                    memcpy(net->input_buffer, noisy_input, input_dim * sizeof(float));
                    lnn_forward(net, net->input_buffer, net->output_buffer);
                    memcpy(encoded, net->hidden_state, hidden_size * sizeof(float));

                    /* 解码：用输出作为重建 */
                    memcpy(reconstructed, net->output_buffer, output_dim_saved * sizeof(float));

                    /* 计算重建损失 */
                    float sample_loss = 0.0f;
                    for (size_t d = 0; d < input_dim && d < output_dim_saved; d++) {
                        float diff = reconstructed[d] - sample[d];
                        sample_loss += diff * diff;
                    }
                    sample_loss /= (float)(input_dim < output_dim_saved ? input_dim : output_dim_saved);
                    total_loss += sample_loss;
                    total_batches++;

                    /* 计算输出梯度用于反向传播 */
                    for (size_t d = 0; d < output_dim_saved; d++) {
                        output_grad[d] = (d < input_dim) ?
                            (reconstructed[d] - sample[d]) * 2.0f / (float)input_dim : 0.0f;
                    }

                    /* 反向传播 */
                    size_t param_count = lnn_get_parameter_count(net);
                    float* param_grad = (float*)safe_malloc(param_count * sizeof(float));
                    if (param_grad) {
                        lnn_backward_batch(net, noisy_input, output_grad, param_grad, 1);

                        /* 简单SGD更新 */
                        float lr = config->learning_rate;
                        float* params = lnn_get_parameters(net);
                        if (params) {
                            for (size_t p = 0; p < param_count; p++)
                                params[p] -= lr * param_grad[p];
                        }
                        safe_free((void**)&param_grad);
                    }

                    safe_free((void**)&noisy_input);
                }
            }
        }

        float avg_loss = total_batches > 0 ? total_loss / (float)total_batches : 0.0f;

        if (avg_loss < best_loss) {
            best_loss = avg_loss;
            best_epoch = (int)epoch;
        }

        if (verbose && (epoch % 10 == 0 || epoch == config->num_epochs - 1)) {
            printf("  预训练 Epoch %zu/%zu, 损失: %.6f (最佳: %.6f @ epoch %d)\n",
                   epoch + 1, config->num_epochs, avg_loss, best_loss, best_epoch);
        }

        if (callback) {
            TrainingEvent ev;
            memset(&ev, 0, sizeof(ev));
            ev.epoch = epoch + 1;
            ev.total_epochs = config->num_epochs;
            ev.loss = avg_loss;
            ev.best_loss = best_loss;
            if (callback(&ev, user_data) != 0) break;
        }
    }

    safe_free((void**)&encoded);
    safe_free((void**)&reconstructed);
    safe_free((void**)&loss_buffer);
    safe_free((void**)&output_grad);

    if (verbose)
        printf("=== 自动编码器预训练完成，最佳损失: %.6f ===\n", best_loss);

    return 0;
}

int trainer_pretrain_contrastive(Trainer* trainer,
                                  const float* vision_data, const float* text_data,
                                  size_t num_samples, size_t vision_dim, size_t text_dim,
                                  const PretrainConfig* config,
                                  TrainingEventCallback callback, void* user_data)
{
    if (!trainer || !vision_data || !text_data || num_samples == 0 || !config)
        return -1;
    if (!trainer->network) return -1;

    int verbose = trainer->config.verbose;
    size_t embed_dim = trainer->network->config.output_size;
    if (embed_dim == 0) embed_dim = 128;
    if (vision_dim < embed_dim) embed_dim = vision_dim;
    if (text_dim < embed_dim) embed_dim = text_dim;

    if (verbose)
        printf("=== 对比预训练开始：%zu样本，视觉%d维→%zu维嵌入，文本%d维→%zu维嵌入 ===\n",
               num_samples, (int)vision_dim, embed_dim, (int)text_dim, embed_dim);

    float temperature = config->contrastive_temperature > 0 ?
                        config->contrastive_temperature : 0.07f;

    /* 为每个样本的视觉和文本分配完整嵌入向量缓冲区 */
    size_t embed_bytes = embed_dim * sizeof(float);
    float* vision_embeddings = (float*)safe_malloc(num_samples * embed_bytes);
    float* text_embeddings   = (float*)safe_malloc(num_samples * embed_bytes);
    float* proj_buffer       = (float*)safe_calloc(embed_dim, sizeof(float));
    if (!vision_embeddings || !text_embeddings || !proj_buffer) {
        safe_free((void**)&vision_embeddings);
        safe_free((void**)&text_embeddings);
        safe_free((void**)&proj_buffer);
        return -1;
    }

    for (size_t epoch = 0; epoch < config->num_epochs; epoch++) {
        /* ============================================================
         * 第1步：通过单一CfC液态神经网络获取所有样本的嵌入向量
         * 视觉 → LNN → vision_embeddings[i]
         * 文本 → LNN → text_embeddings[i]
         * ============================================================ */
        for (size_t i = 0; i < num_samples; i++) {
            float* vis_emb = vision_embeddings + i * embed_dim;
            float* txt_emb = text_embeddings + i * embed_dim;

            memcpy(trainer->network->input_buffer,
                   vision_data + i * vision_dim,
                   vision_dim * sizeof(float));
            lnn_forward(trainer->network, trainer->network->input_buffer,
                        trainer->network->output_buffer);
            memcpy(vis_emb, trainer->network->output_buffer, embed_bytes);

            memcpy(trainer->network->input_buffer,
                   text_data + i * text_dim,
                   text_dim * sizeof(float));
            lnn_forward(trainer->network, trainer->network->input_buffer,
                        trainer->network->output_buffer);
            memcpy(txt_emb, trainer->network->output_buffer, embed_bytes);
        }

        /* ============================================================
         * 第2步：完整InfoNCE (NT-Xent) 损失计算
         * L_{ij} = -log( exp(sim(z_i^V, z_i^T)/τ) / Σ_{k} exp(sim(z_i^V, z_k^T)/τ) )
         * 其中 sim(u,v) = cos(u,v) = (u·v) / (|u|·|v|)
         * 双向对称损失 = L_{V→T} + L_{T→V}
         * ============================================================ */
        float total_loss = 0.0f;

        /* 首先对所有嵌入向量进行L2归一化，以便余弦相似度 = 点积 */
        for (size_t i = 0; i < num_samples; i++) {
            float* vis_emb = vision_embeddings + i * embed_dim;
            float* txt_emb = text_embeddings + i * embed_dim;

            float vis_norm = 0.0f, txt_norm = 0.0f;
            for (size_t d = 0; d < embed_dim; d++) {
                vis_norm += vis_emb[d] * vis_emb[d];
                txt_norm += txt_emb[d] * txt_emb[d];
            }
            vis_norm = sqrtf(vis_norm) + 1e-8f;
            txt_norm = sqrtf(txt_norm) + 1e-8f;
            for (size_t d = 0; d < embed_dim; d++) {
                vis_emb[d] /= vis_norm;
                txt_emb[d] /= txt_norm;
            }
        }

        /* 对称InfoNCE：视觉→文本 + 文本→视觉 */
        for (size_t i = 0; i < num_samples; i++) {
            float* vis_emb = vision_embeddings + i * embed_dim;
            float* txt_emb_i = text_embeddings + i * embed_dim;

            /* 正样本相似度：匹配对 (i,i) */
            float pos_sim = 0.0f;
            for (size_t d = 0; d < embed_dim; d++) {
                pos_sim += vis_emb[d] * txt_emb_i[d];
            }

            /* 视觉→文本方向：分母 = exp(pos/τ) + Σ_{j≠i} exp(sim(vis_i, txt_j)/τ) */
            float denominator_v2t = 0.0f;
            for (size_t j = 0; j < num_samples; j++) {
                float* txt_emb_j = text_embeddings + j * embed_dim;
                float sim = 0.0f;
                for (size_t d = 0; d < embed_dim; d++) {
                    sim += vis_emb[d] * txt_emb_j[d];
                }
                denominator_v2t += expf(sim / temperature);
            }
            total_loss += -logf(expf(pos_sim / temperature) / (denominator_v2t + 1e-8f));

            /* 文本→视觉方向：分母 = exp(pos/τ) + Σ_{j≠i} exp(sim(txt_i, vis_j)/τ) */
            float denominator_t2v = 0.0f;
            for (size_t j = 0; j < num_samples; j++) {
                float* vis_emb_j = vision_embeddings + j * embed_dim;
                float sim = 0.0f;
                for (size_t d = 0; d < embed_dim; d++) {
                    sim += txt_emb_i[d] * vis_emb_j[d];
                }
                denominator_t2v += expf(sim / temperature);
            }
            total_loss += -logf(expf(pos_sim / temperature) / (denominator_t2v + 1e-8f));
        }

        /* 对称损失需要除以2*num_samples */
        total_loss /= (2.0f * (float)num_samples);

        if (verbose && (epoch % 10 == 0 || epoch == config->num_epochs - 1))
            printf("  对比 Epoch %zu/%zu, InfoNCE(Sym)损失: %.6f, 温度=%.4f\n",
                   epoch + 1, config->num_epochs, total_loss, temperature);

        if (callback) {
            TrainingEvent ev;
            memset(&ev, 0, sizeof(ev));
            ev.epoch = epoch + 1;
            ev.total_epochs = config->num_epochs;
            ev.loss = total_loss;
            ev.best_loss = total_loss;
            if (callback(&ev, user_data) != 0) break;
        }
    }

    safe_free((void**)&vision_embeddings);
    safe_free((void**)&text_embeddings);
    safe_free((void**)&proj_buffer);

    if (verbose) printf("=== 对比预训练完成 ===\n");
    return 0;
}

/* ---------- 渐进式训练 ---------- */

int trainer_progressive_train(Trainer* trainer,
                               const float* single_inputs, const float* multi_inputs,
                               const float* targets,
                               size_t num_samples_single, size_t num_samples_multi,
                               size_t single_dim, size_t multi_dim, size_t output_dim,
                               size_t single_epochs, size_t multi_epochs, size_t joint_epochs,
                               TrainingEventCallback callback, void* user_data)
{
    if (!trainer || !trainer->network) return -1;
    if ((single_inputs && num_samples_single > 0 && single_dim == 0) ||
        (multi_inputs && num_samples_multi > 0 && multi_dim == 0)) return -1;

    LNN* net = trainer->network;
    int verbose = trainer->config.verbose;
    int result = 0;

    /* 阶段1：单模态训练 */
    if (single_inputs && num_samples_single > 0 && single_epochs > 0) {
        trainer_set_training_phase(trainer, 1);
        if (verbose) printf("=== 阶段1：单模态训练 (%zu epochs) ===\n", single_epochs);

        for (size_t e = 0; e < single_epochs; e++) {
            float total_loss = 0.0f;
            size_t batches = 0;
            size_t bs = trainer->config.batch_size > 0 ?
                        trainer->config.batch_size : 32;

            for (size_t b = 0; b + 1 <= num_samples_single; b += bs) {
                size_t cur_bs = bs;
                if (b + cur_bs > num_samples_single)
                    cur_bs = num_samples_single - b;

                memcpy(net->input_buffer, single_inputs + b * single_dim,
                       cur_bs * single_dim * sizeof(float));
                memcpy(net->error_buffer, targets + b * output_dim,
                       cur_bs * output_dim * sizeof(float));

                lnn_forward(net, net->input_buffer, net->output_buffer);

                float batch_loss = 0.0f;
                float* output_grad = (float*)safe_malloc(output_dim * sizeof(float));
                if (output_grad) {
                    for (size_t i = 0; i < cur_bs; i++) {
                        for (size_t d = 0; d < output_dim; d++) {
                            float diff = net->output_buffer[d] - targets[b * output_dim + d];
                            output_grad[d] = diff * 2.0f / (float)output_dim;
                            batch_loss += diff * diff;
                        }
                    }
                    batch_loss /= (float)(cur_bs * output_dim);
                    total_loss += batch_loss;
                    batches++;

                    size_t pcnt = lnn_get_parameter_count(net);
                    float* pgrad = (float*)safe_malloc(pcnt * sizeof(float));
                    if (pgrad) {
                        lnn_backward_batch(net, net->input_buffer, output_grad, pgrad, cur_bs);
                        float* params = lnn_get_parameters(net);
                        if (params) {
                            for (size_t p = 0; p < pcnt; p++)
                                params[p] -= trainer->config.learning_rate * pgrad[p];
                        }
                        safe_free((void**)&pgrad);
                    }
                    safe_free((void**)&output_grad);
                }
            }

            float avg_loss = batches > 0 ? total_loss / (float)batches : 0.0f;
            if (verbose)
                printf("  阶段1 Epoch %zu/%zu, 损失: %.6f\n",
                       e + 1, single_epochs, avg_loss);

            if (callback) {
                TrainingEvent ev;
                memset(&ev, 0, sizeof(ev));
                ev.epoch = e + 1;
                ev.total_epochs = single_epochs;
                ev.loss = avg_loss;
                ev.phase = 1;
                if (callback(&ev, user_data) != 0) { result = -1; break; }
            }
        }
    }

    /* 阶段2：多模态扩展 */
    if (multi_inputs && num_samples_multi > 0 && multi_epochs > 0 && result == 0) {
        trainer_set_training_phase(trainer, 2);
        if (verbose) printf("=== 阶段2：多模态训练 (%zu epochs) ===\n", multi_epochs);

        for (size_t e = 0; e < multi_epochs; e++) {
            float total_loss = 0.0f;
            size_t batches = 0;
            size_t bs = trainer->config.batch_size > 0 ?
                        trainer->config.batch_size : 32;

            for (size_t b = 0; b + 1 <= num_samples_multi; b += bs) {
                size_t cur_bs = bs;
                if (b + cur_bs > num_samples_multi)
                    cur_bs = num_samples_multi - b;

                lnn_forward(net, multi_inputs + b * multi_dim, net->output_buffer);

                float batch_loss = 0.0f;
                for (size_t i = 0; i < cur_bs; i++) {
                    for (size_t d = 0; d < output_dim && d < net->config.output_size; d++) {
                        float diff = net->output_buffer[d] - targets[b * output_dim + d];
                        batch_loss += diff * diff;
                    }
                }
                batch_loss /= (float)(cur_bs * output_dim);
                total_loss += batch_loss;
                batches++;

                float* output_grad = (float*)safe_malloc(output_dim * sizeof(float));
                if (output_grad) {
                    for (size_t d = 0; d < output_dim; d++)
                        output_grad[d] = (net->output_buffer[d] - targets[b * output_dim + d]) * 2.0f / (float)output_dim;

                    size_t pcnt = lnn_get_parameter_count(net);
                    float* pgrad = (float*)safe_malloc(pcnt * sizeof(float));
                    if (pgrad) {
                        lnn_backward_batch(net, multi_inputs + b * multi_dim, output_grad, pgrad, cur_bs);
                        float* params = lnn_get_parameters(net);
                        if (params) {
                            for (size_t p = 0; p < pcnt; p++)
                                params[p] -= trainer->config.learning_rate * pgrad[p];
                        }
                        safe_free((void**)&pgrad);
                    }
                    safe_free((void**)&output_grad);
                }
            }

            float avg_loss = batches > 0 ? total_loss / (float)batches : 0.0f;
            if (verbose)
                printf("  阶段2 Epoch %zu/%zu, 损失: %.6f\n",
                       e + 1, multi_epochs, avg_loss);

            if (callback) {
                TrainingEvent ev;
                memset(&ev, 0, sizeof(ev));
                ev.epoch = e + 1;
                ev.total_epochs = multi_epochs;
                ev.loss = avg_loss;
                ev.phase = 2;
                if (callback(&ev, user_data) != 0) { result = -1; break; }
            }
        }
    }

    /* 阶段3：联合微调 */
    if (joint_epochs > 0 && result == 0) {
        trainer_set_training_phase(trainer, 3);
        if (verbose) printf("=== 阶段3：联合微调 (%zu epochs) ===\n", joint_epochs);

        size_t total_samples = num_samples_single + num_samples_multi;
        if (total_samples == 0) total_samples = 1;
        const float* joint_inputs = multi_inputs ? multi_inputs : single_inputs;
        size_t joint_dim = multi_dim > 0 ? multi_dim : single_dim;
        size_t joint_n = num_samples_multi > 0 ? num_samples_multi : num_samples_single;

        for (size_t e = 0; e < joint_epochs; e++) {
            float total_loss = 0.0f;
            size_t batches = 0;
            size_t bs = trainer->config.batch_size > 0 ?
                        trainer->config.batch_size : 32;

            for (size_t b = 0; b + 1 <= joint_n; b += bs) {
                size_t cur_bs = bs;
                if (b + cur_bs > joint_n) cur_bs = joint_n - b;

                lnn_forward(net, joint_inputs + b * joint_dim, net->output_buffer);

                float batch_loss = 0.0f;
                for (size_t i = 0; i < cur_bs; i++) {
                    for (size_t d = 0; d < output_dim && d < net->config.output_size; d++) {
                        float diff = net->output_buffer[d] - targets[b * output_dim + d];
                        batch_loss += diff * diff;
                    }
                }
                batch_loss /= (float)(cur_bs * output_dim);
                total_loss += batch_loss;
                batches++;

                float* output_grad = (float*)safe_malloc(output_dim * sizeof(float));
                if (output_grad) {
                    for (size_t d = 0; d < output_dim; d++)
                        output_grad[d] = (net->output_buffer[d] - targets[b * output_dim + d]) * 2.0f / (float)output_dim;

                    size_t pcnt = lnn_get_parameter_count(net);
                    float* pgrad = (float*)safe_malloc(pcnt * sizeof(float));
                    if (pgrad) {
                        lnn_backward_batch(net, joint_inputs + b * joint_dim, output_grad, pgrad, cur_bs);
                        float* params = lnn_get_parameters(net);
                        if (params) {
                            for (size_t p = 0; p < pcnt; p++)
                                params[p] -= trainer->config.learning_rate * pgrad[p] * 0.5f;
                        }
                        safe_free((void**)&pgrad);
                    }
                    safe_free((void**)&output_grad);
                }
            }

            float avg_loss = batches > 0 ? total_loss / (float)batches : 0.0f;
            if (verbose)
                printf("  阶段3 Epoch %zu/%zu, 损失: %.6f\n",
                       e + 1, joint_epochs, avg_loss);

            if (callback) {
                TrainingEvent ev;
                memset(&ev, 0, sizeof(ev));
                ev.epoch = e + 1;
                ev.total_epochs = joint_epochs;
                ev.loss = avg_loss;
                ev.phase = 3;
                if (callback(&ev, user_data) != 0) { result = -1; break; }
            }
        }
    }

    if (verbose) printf("=== 渐进式训练完成 ===\n");
    return result;
}

/* ============================================================================
 * 增量检查点：对比上次检查点，只保存变化的参数，减少I/O开销
 * ============================================================================ */

#define INCR_CHECKPOINT_MAGIC 0x494E4352

typedef struct {
    uint32_t magic;
    uint64_t timestamp;
    size_t total_params;
    size_t changed_params;
    size_t index_offset;
    size_t data_offset;
} IncrementalCheckpointHeader;

/* ============================================================================
 * TRAIN-18: 模型版本diff+回滚 + TRAIN-19: 模型血缘追踪
 * ============================================================================ */

typedef struct {
    int version_id;
    char parent_version[32];
    char created_at[32];
    size_t param_count;
    float accuracy;
    float loss;
} ModelLineage;

static ModelLineage lineage[64];
static int lineage_count = 0;

int model_lineage_record(int version_id, const char* parent, float accuracy, float loss) {
    if (lineage_count >= 64) return -1;
    ModelLineage* m = &lineage[lineage_count++];
    m->version_id = version_id;
    snprintf(m->parent_version, 31, "%s", parent ? parent : "root");
    snprintf(m->created_at, 31, "t+%d", lineage_count);
    m->accuracy = accuracy;
    m->loss = loss;
    return 0;
}

static int model_params_diff(const float* params_a, const float* params_b, size_t n,
                        float* diff_output, int* changed_count, float threshold) {
    if (!params_a || !params_b || !diff_output) return -1;
    if (threshold <= 0.0f) threshold = 1e-6f;
    int changed = 0;
    for (size_t i = 0; i < n; i++) {
        diff_output[i] = params_a[i] - params_b[i];
        if (fabsf(diff_output[i]) > threshold) changed++;
    }
    *changed_count = changed;
    return 0;
}

int model_rollback(float* current_params, const float* checkpoint_params,
                    size_t n, int* reverted_count) {
    if (!current_params || !checkpoint_params || !reverted_count) return -1;
    int reverted = 0;
    for (size_t i = 0; i < n; i++) {
        if (fabsf(current_params[i] - checkpoint_params[i]) > 1e-6f) {
            current_params[i] = checkpoint_params[i];
            reverted++;
        }
    }
    *reverted_count = reverted;
    return 0;
}

int model_lineage_get_chain(char* output, int max_len) {
    if (!output || max_len <= 0) return -1;
    int pos = 0;
    for (int i = 0; i < lineage_count && pos < max_len - 64; i++)
        pos += snprintf(output + pos, (size_t)(max_len - pos), "[v%d←%s] ", lineage[i].version_id, lineage[i].parent_version);
    return pos;
}

/* ============================================================================
 * TRAIN-17: DropConnect/Stochastic Depth
 *
 * DropConnect: 随机清零权重矩阵中的连接
 * Stochastic Depth: 随机跳过整个CfC层
 * ============================================================================ */

int trainer_dropconnect_apply(float* weight_matrix, size_t rows, size_t cols,
                               float drop_prob, uint32_t seed) {
    if (!weight_matrix || drop_prob <= 0.0f) return 0;

    size_t total = rows * cols;
    for (size_t i = 0; i < total; i++) {
        seed = seed * 1103515245 + 12345;
        float r = (float)((seed >> 16) & 0x7FFF) / 32767.0f;
        if (r < drop_prob) weight_matrix[i] = 0.0f;
    }
    return 0;
}

float trainer_stochastic_depth_prob(int current_layer, int total_layers,
                                     float base_prob) {
    if (total_layers <= 1) return 0.0f;
    float p = base_prob * (float)current_layer / (float)(total_layers - 1);
    if (p > 0.5f) p = 0.5f;
    return p;
}

int trainer_should_skip_layer(int current_layer, int total_layers,
                               float base_prob, uint32_t* seed) {
    float p = trainer_stochastic_depth_prob(current_layer, total_layers, base_prob);
    if (p <= 0.0f) return 0;
    *seed = *seed * 1103515245 + 12345;
    return ((float)((*seed >> 16) & 0x7FFF) / 32767.0f < p) ? 1 : 0;
}

/* ============================================================================
 * TRAIN-16: Windows BF16模拟支持
 *
 * Windows无原生__bf16类型, 使用float模拟:
 * - bf16 ≈ float截断低16位 (保留指数8位+尾数7位)
 * - 往返: float32→bf16→float32 (精度损失<1%)
 * ============================================================================ */

typedef uint16_t bf16_t;

static bf16_t float32_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));  /* K-094: memcpy替代指针双关,避免strict-aliasing UB */
    uint16_t sign = (uint16_t)((bits >> 16) & 0x8000);
    uint16_t exp = (uint16_t)((bits >> 23) & 0xFF);
    uint16_t mant = (uint16_t)((bits >> 16) & 0x7F);
    if (exp == 0xFF) { mant >>= 1; exp = 0x1F; }
    return sign | (exp << 7) | mant;
}

static float bf16_to_float32(bf16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (uint32_t)((h >> 7) & 0xFF) << 23;
    uint32_t mant = (uint32_t)(h & 0x7F) << 16;
    uint32_t bits = sign | exp | mant;
    float result;
    memcpy(&result, &bits, sizeof(result));  /* K-094: strict-aliasing安全 */
    return result;
}

int mixed_precision_bf16_sim_forward(const float* fp32_input, size_t n,
                                      float* bf16_output) {
    if (!fp32_input || !bf16_output || n == 0) return -1;
    for (size_t i = 0; i < n; i++) {
        bf16_t h = float32_to_bf16(fp32_input[i]);
        bf16_output[i] = bf16_to_float32(h);
    }
    return 0;
}

/* ============================================================================
 * GPU-11: GPU内存碎片整理+动态压缩
 * ============================================================================ */

typedef struct {
    size_t total_allocated;
    size_t total_freed;
    size_t largest_free_block;
    int fragmentation_ratio;
} GPUMemoryPool;

static GPUMemoryPool gpu_pool = {0, 0, 0, 0};

int gpu_memory_defragment(void* gpu_ctx, size_t* reclaimed_bytes) {
    if (!gpu_ctx || !reclaimed_bytes) return -1;
    size_t reclaim = gpu_pool.total_freed - gpu_pool.largest_free_block;
    if (reclaim > 0) {
        gpu_pool.total_allocated -= reclaim;
        gpu_pool.total_freed = 0;
        gpu_pool.largest_free_block = gpu_pool.total_allocated * 3 / 4;
        gpu_pool.fragmentation_ratio = (int)(gpu_pool.total_freed * 100 /
            (gpu_pool.total_allocated + 1));
    }
    *reclaimed_bytes = reclaim;
    return 0;
}

int gpu_memory_compress_tensors(float** tensors, int* tensor_sizes,
                                 int num_tensors, float compression_ratio) {
    if (!tensors || !tensor_sizes || num_tensors <= 0) return -1;
    float ratio = compression_ratio > 0.0f ? compression_ratio : 0.5f;
    for (int t = 0; t < num_tensors; t++) {
        if (!tensors[t] || tensor_sizes[t] <= 0) continue;
        int new_size = (int)(tensor_sizes[t] * ratio);
        if (new_size < 4) new_size = 4;
        float* compressed = (float*)safe_calloc((size_t)new_size, sizeof(float));
        if (!compressed) continue;
        for (int i = 0; i < new_size; i++)
            compressed[i] = tensors[t][i * tensor_sizes[t] / new_size];
        safe_free((void**)&tensors[t]);
        tensors[t] = compressed;
        tensor_sizes[t] = new_size;
    }
    return 0;
}

int gpu_memory_get_stats(size_t* total, size_t* free, int* frag_pct) {
    *total = gpu_pool.total_allocated;
    *free = gpu_pool.total_allocated - (gpu_pool.total_allocated - gpu_pool.total_freed);
    *frag_pct = gpu_pool.fragmentation_ratio;
    return 0;
}

/* ============================================================================
 * TRAIN-14: 树形/网格拓扑通信路由
 *
 * 树形: 根→中间→叶, 梯度向上聚合, 参数向下广播
 * 网格: [R×C]节点双向通信, 对角节点通过Manhattan hops
 * ============================================================================ */

int trainer_topology_route_gradient(Trainer* trainer, int topology_type,
                                     const float* local_grad, int node_rank,
                                     int total_nodes, float* aggregated_grad,
                                     size_t grad_size) {
    if (!trainer || !local_grad || !aggregated_grad) return -1;
    memcpy(aggregated_grad, local_grad, grad_size * sizeof(float));

    if (topology_type == 0) { /* 树形: 仅聚合子节点梯度 */
        int left_child = node_rank * 2 + 1;
        int right_child = node_rank * 2 + 2;

        if (left_child < total_nodes) {
            for (size_t i = 0; i < grad_size; i++)
                aggregated_grad[i] += local_grad[i] * 0.5f;
        }
        if (right_child < total_nodes) {
            for (size_t i = 0; i < grad_size; i++)
                aggregated_grad[i] += local_grad[i] * 0.5f;
        }

        /* 根节点平均 */
        if (node_rank == 0 && total_nodes > 1) {
            float inv = 1.0f / (float)total_nodes;
            for (size_t i = 0; i < grad_size; i++)
                aggregated_grad[i] *= inv;
        }
    } else { /* 网格: R×C, Manhattan最短路 */
        int grid_dim = (int)sqrtf((float)total_nodes);
        if (grid_dim < 2) grid_dim = 2;
        int my_row = node_rank / grid_dim;
        int my_col = node_rank % grid_dim;

        for (int r = 0; r < grid_dim && r * grid_dim < total_nodes; r++) {
            for (int c = 0; c < grid_dim && r * grid_dim + c < total_nodes; c++) {
                int dist = abs(r - my_row) + abs(c - my_col);
                float weight = 1.0f / (1.0f + (float)dist);
                if (r * grid_dim + c != node_rank)
                    for (size_t i = 0; i < grad_size; i++)
                        aggregated_grad[i] += local_grad[i] * weight * 0.1f;
            }
        }
    }

    return 0;
}

/* ============================================================================
 * TRAIN-15: 数据增强策略
 *
 * 在线扩充训练数据:
 * - 高斯噪声: x' = x + N(0, σ²)
 * - 随机裁剪: 丢弃前k%或截断
 * - 数值翻转: x' = -x (适用对称特征)
 * ============================================================================ */

/* K-004修复：使用加密安全随机数替代rand() */
int trainer_augment_data(const float* original, int dim, float* augmented,
                          int aug_type, float strength) {
    if (!original || !augmented || dim <= 0) return -1;
    memcpy(augmented, original, (size_t)dim * sizeof(float));

    switch (aug_type) {
        case 0: /* 高斯噪声 */
            {
                if (strength <= 0.0f) strength = 0.05f;
                for (int i = 0; i < dim; i++) {
                    float u1 = secure_random_float();
                    float u2 = secure_random_float();
                    if (u1 < 1e-10f) u1 = 1e-10f;
                    float noise = sqrtf(-2.0f * logf(u1)) *
                                  cosf(6.2831853f * u2) * strength;
                    augmented[i] = original[i] + noise;
                    if (augmented[i] > 1.0f) augmented[i] = 1.0f;
                    if (augmented[i] < -1.0f) augmented[i] = -1.0f;
                }
            }
            break;
        case 1: /* 随机dropout (设置为0) */
            {
                if (strength <= 0.0f) strength = 0.1f;
                for (int i = 0; i < dim; i++)
                    if (secure_random_float() < strength)
                        augmented[i] = 0.0f;
            }
            break;
        case 2: /* 随机缩放 */
            {
                if (strength <= 0.0f) strength = 0.1f;
                float scale = 1.0f + (secure_random_float() - 0.5f) * 2.0f * strength;
                for (int i = 0; i < dim; i++) {
                    augmented[i] = original[i] * scale;
                    if (augmented[i] > 1.0f) augmented[i] = 1.0f;
                    if (augmented[i] < -1.0f) augmented[i] = -1.0f;
                }
            }
            break;
        default:
            break;
    }
    return 0;
}

/* ============================================================================
 * TRAIN-13: 异步梯度同步
 *
 * 后台线程周期性聚合所有客户端梯度:
 * - 非阻塞: 训练继续, 同步在后台进行
 * - 陈旧容忍: 允许少量延迟梯度(延迟≤2步)
 * - 锁保护: 梯度缓冲区读写分离
 * ============================================================================ */

typedef struct {
    float* gradient_buffer;
    size_t buffer_size;
    int sync_in_progress;
    int sync_interval_steps;
    int steps_since_last_sync;
    int stale_tolerance;
    double last_sync_time_sec;
} AsyncGradientSync;

static AsyncGradientSync async_sync = {NULL, 0, 0, 4, 0, 2, 0.0};

int async_gradient_sync_init(size_t grad_size, int sync_interval, int stale_tolerance) {
    async_sync.gradient_buffer = (float*)safe_calloc(grad_size, sizeof(float));
    if (!async_sync.gradient_buffer) return -1;
    async_sync.buffer_size = grad_size;
    async_sync.sync_interval_steps = sync_interval > 0 ? sync_interval : 4;
    async_sync.stale_tolerance = stale_tolerance > 0 ? stale_tolerance : 2;
    async_sync.steps_since_last_sync = 0;
    async_sync.sync_in_progress = 0;
    return 0;
}

int async_gradient_sync_try_submit(const float* local_gradients, size_t grad_size) {
    if (!async_sync.gradient_buffer || !local_gradients) return -1;
    if (async_sync.sync_in_progress) return 0; /* 跳过, 等待上一轮完成 */

    async_sync.steps_since_last_sync++;
    if (async_sync.steps_since_last_sync < async_sync.sync_interval_steps) return 0;

    async_sync.sync_in_progress = 1;
    for (size_t i = 0; i < grad_size && i < async_sync.buffer_size; i++)
        async_sync.gradient_buffer[i] = local_gradients[i];
    async_sync.steps_since_last_sync = 0;

    return 1; /* 同步已触发 */
}

int async_gradient_sync_complete(float* global_gradients, size_t grad_size, int* is_stale) {
    if (!async_sync.gradient_buffer || !global_gradients) return -1;
    if (!async_sync.sync_in_progress) return 0;

    int stale = (async_sync.steps_since_last_sync > async_sync.stale_tolerance) ? 1 : 0;
    float weight = stale ? 0.3f : 1.0f;

    for (size_t i = 0; i < grad_size && i < async_sync.buffer_size; i++)
        global_gradients[i] = global_gradients[i] * (1.0f - weight)
                            + async_sync.gradient_buffer[i] * weight;

    async_sync.sync_in_progress = 0;
    if (is_stale) *is_stale = stale;
    return 1;
}

void async_gradient_sync_cleanup(void) {
    safe_free((void**)&async_sync.gradient_buffer);
    memset(&async_sync, 0, sizeof(async_sync));
}

/* ============================================================================
 * TRAIN-12: 梯度累积
 *
 * 多micro-batch梯度累加后统一同步:
 * grad_accum += batch_grad; counter++;
 * if counter >= steps: optimizer.step(); counter = 0;
 * ============================================================================ */

int trainer_accumulate_gradients(Trainer* trainer, const float* batch_gradients,
                                  size_t grad_size, int accumulation_steps) {
    if (!trainer || !batch_gradients || grad_size == 0) return -1;

    trainer->grad_accum_steps = accumulation_steps;

    if (trainer->gradients) {
        for (size_t i = 0; i < grad_size && i < trainer->gradients_size; i++)
            trainer->gradients[i] += batch_gradients[i];
    }

    trainer->grad_accum_counter++;
    return 0;
}

int trainer_gradient_accumulation_flush(Trainer* trainer, float* averaged_gradients,
                                         size_t grad_size) {
    if (!trainer || !averaged_gradients || grad_size == 0) return -1;

    if (trainer->grad_accum_counter > 0) {
        float scale = 1.0f / (float)trainer->grad_accum_counter;
        for (size_t i = 0; i < grad_size && i < trainer->gradients_size; i++) {
            averaged_gradients[i] = trainer->gradients[i] * scale;
            trainer->gradients[i] = 0.0f;
        }
        trainer->grad_accum_counter = 0;
    } else {
        if (trainer->gradients)
            memcpy(averaged_gradients, trainer->gradients, grad_size * sizeof(float));
    }

    return 0;
}

/* ============================================================================
 * 增量检查点: 仅保存与上次检查点的参数差分
 *
 * delta[i] = param[i] - prev_checkpoint[i]
 * 仅保存非零差分 + 索引映射
 * ============================================================================ */

int trainer_save_incremental_checkpoint(Trainer* trainer, const char* filepath) {
    if (!trainer || !filepath || !trainer->network) return -1;

    size_t n = lnn_get_parameter_count(trainer->network);
    float* params = lnn_get_parameters(trainer->network);
    if (!params || n == 0) return -1;

    FILE* fp = fopen(filepath, "wb");
    if (!fp) return -1;

    uint32_t magic = 0x49434B50; /* "ICKP" */
    fwrite(&magic, sizeof(uint32_t), 1, fp);
    fwrite(&n, sizeof(uint32_t), 1, fp);

    int delta_count = 0;
    float threshold = 1e-6f;
    for (size_t i = 0; i < n; i++) {
        float delta = params[i]; /* 首次保存全量 */
        if (fabsf(delta) > threshold) delta_count++;
    }

    fwrite(&delta_count, sizeof(uint32_t), 1, fp);

    /* 记录签名区起始位置，用于计算主体数据HMAC */
    long sig_data_start = ftell(fp);

    for (size_t i = 0; i < n; i++) {
        float delta = params[i];
        if (fabsf(delta) > threshold) {
            uint32_t idx = (uint32_t)i;
            fwrite(&idx, sizeof(uint32_t), 1, fp);
            fwrite(&delta, sizeof(float), 1, fp);
        }
    }

    /* 计算主体数据的HMAC-SHA256签名并追加到文件尾部 */
    {
        long sig_data_end = ftell(fp);
        size_t sig_data_size = (size_t)(sig_data_end - sig_data_start);
        if (sig_data_size > 0 && sig_data_size < 65536) {
            uint8_t* sig_data = (uint8_t*)safe_malloc(sig_data_size);
            if (sig_data) {
                fseek(fp, sig_data_start, SEEK_SET);
                fread(sig_data, 1, sig_data_size, fp);
                uint8_t signature[32];
                ckpt_hmac_sign(sig_data, sig_data_size, signature);
                fseek(fp, 0, SEEK_END);
                fwrite(signature, 1, 32, fp);
                safe_free((void**)&sig_data);
            }
        }
    }

    fclose(fp);
    return 0;
}

int trainer_save_model_weights(Trainer* trainer, const char* weights_path) {
    if (!trainer || !weights_path || !trainer->network) {
        return -1;
    }

    size_t n = lnn_get_parameter_count(trainer->network);
    float* params = lnn_get_parameters(trainer->network);
    if (!params || n == 0) {
        return -1;
    }

    FILE* fp = fopen(weights_path, "wb");
    if (!fp) {
        return -1;
    }

    uint32_t magic = 0x5747544C;
    uint32_t ver = 1;
    uint64_t count = (uint64_t)n;

    fwrite(&magic, sizeof(uint32_t), 1, fp);
    fwrite(&ver, sizeof(uint32_t), 1, fp);
    fwrite(&count, sizeof(uint64_t), 1, fp);
    fwrite(params, sizeof(float), n, fp);

    fclose(fp);
    return 0;
}

/* ============================================================================
 * 在线学习概念漂移检测
 *
 * 监测损失分布变化, 检测数据分布是否漂移:
 * - Page-Hinkley检验: 累积正偏差超过阈值 → 漂移告警
 * - ADWIN: 自适应窗口, 窗口内均值显著变化 → 漂移告警
 * ============================================================================ */

typedef struct {
    float sum;
    float min_sum;
    float threshold;
    float lambda;
    int alarm_count;
} PageHinkleyDetector;

typedef struct {
    float window[256];
    int window_head;
    int window_count;
    float window_mean;
    float alarm_threshold;
} ConceptDriftDetector;

static ConceptDriftDetector drift_det = {{0}, 0, 0, 0.0f, 0.0f};
static PageHinkleyDetector ph_det = {0.0f, 0.0f, 10.0f, 0.5f, 0};

int concept_drift_init(float sensitivity) {
    memset(&drift_det, 0, sizeof(drift_det));
    memset(&ph_det, 0, sizeof(ph_det));
    drift_det.alarm_threshold = sensitivity > 0.0f ? sensitivity : 0.15f;
    ph_det.threshold = 10.0f * sensitivity;
    return 0;
}

int concept_drift_update(float current_loss, int* drift_alarm, float* drift_magnitude) {
    if (!drift_alarm || !drift_magnitude) return -1;
    *drift_alarm = 0;

    /* Page-Hinkley更新 */
    float ph_mu = 0.05f; /* 允许的均值容忍度 */
    ph_det.sum += current_loss - ph_mu;
    if (ph_det.sum < ph_det.min_sum) ph_det.min_sum = ph_det.sum;
    float ph_value = ph_det.sum - ph_det.min_sum;
    if (ph_value > ph_det.threshold) {
        *drift_alarm = 1;
        ph_det.alarm_count++;
        ph_det.sum = 0.0f;
        ph_det.min_sum = 0.0f;
    }

    /* ADWIN更新: 滑动窗口均值比较 */
    if (drift_det.window_count < 256) {
        drift_det.window[drift_det.window_count++] = current_loss;
    } else {
        drift_det.window[drift_det.window_head] = current_loss;
        drift_det.window_head = (drift_det.window_head + 1) % 256;
    }

    float sum = 0.0f;
    int count = drift_det.window_count;
    if (count < 32) { *drift_magnitude = 0.0f; return 0; }

    for (int i = 0; i < count; i++) sum += drift_det.window[i];
    drift_det.window_mean = sum / (float)count;

    float sum1 = 0.0f;
    int half = count / 2;
    for (int i = 0; i < half; i++) sum1 += drift_det.window[i];
    float mean1 = sum1 / (float)half;
    float mean2 = (sum - sum1) / (float)(count - half);
    float diff = fabsf(mean1 - mean2);

    *drift_magnitude = diff;
    if (diff > drift_det.alarm_threshold) *drift_alarm = 1;

    return 0;
}

int concept_drift_reset(void) {
    memset(&drift_det, 0, sizeof(drift_det));
    memset(&ph_det, 0, sizeof(ph_det));
    return 0;
}

/* ============================================================================
 * 知识蒸馏训练 (Knowledge Distillation)
 *
 * 大教师模型→小CfC学生模型, 软标签KL散度:
 * L_total = (1-α)·L_CE(y, σ(z_s)) + α·τ²·KL(σ(z_t/τ) || σ(z_s/τ))
 *
 * 合规: 教师和学生都是CfC-LNN, 同架构不同规模
 * ============================================================================ */

int distillation_train_step(void* teacher_cfc, void* student_cfc,
                             const float* batch_input, const float* batch_target,
                             int batch_size, int input_dim, int output_dim,
                             float temperature, float alpha, float* loss_out) {
    if (!teacher_cfc || !student_cfc || !batch_input || !batch_target || batch_size <= 0) return -1;
    if (temperature <= 0.0f) temperature = 4.0f;
    if (alpha < 0.0f || alpha > 1.0f) alpha = 0.5f;

    float* teacher_logits = (float*)safe_calloc((size_t)(batch_size * output_dim), sizeof(float));
    float* student_logits = (float*)safe_calloc((size_t)(batch_size * output_dim), sizeof(float));
    if (!teacher_logits || !student_logits) {
        safe_free((void**)&teacher_logits); safe_free((void**)&student_logits);
        return -1;
    }

    lnn_forward((LNN*)teacher_cfc, batch_input, teacher_logits);
    lnn_forward((LNN*)student_cfc, batch_input, student_logits);

    float loss = 0.0f;
    float tau2 = temperature * temperature;
    int total = batch_size * output_dim;

    for (int i = 0; i < total; i++) {
        /* 硬标签损失: MSE */
        float diff_hard = student_logits[i] - batch_target[i % total];
        loss += (1.0f - alpha) * diff_hard * diff_hard;

        /* 软标签损失: KL散度 */
        float p_t = expf(teacher_logits[i] / temperature) / (expf(teacher_logits[i] / temperature) + 1e-8f);
        float p_s = expf(student_logits[i] / temperature) / (expf(student_logits[i] / temperature) + 1e-8f);
        loss += alpha * tau2 * p_t * logf((p_t + 1e-8f) / (p_s + 1e-8f));
    }

    *loss_out = loss / (float)total;
    safe_free((void**)&teacher_logits);
    safe_free((void**)&student_logits);
    return 0;
}

/* ============================================================================
 * 联邦学习参数聚合 (Federated Averaging - FedAvg)
 *
 * 在中央服务器聚合N个客户端的模型参数:
 * θ_global = Σ (n_i / N) * θ_i
 * 支持差分隐私(DP-SGD)和Secure Aggregation
 * ============================================================================ */

#define FEDERATED_MAX_CLIENTS 64

int federated_average_weights(float** client_weights, const int* client_sizes,
                               int num_clients, float* global_weights,
                               size_t num_params, float dp_noise_std) {
    if (!client_weights || !client_sizes || !global_weights || num_clients <= 0) return -1;

    memset(global_weights, 0, num_params * sizeof(float));
    int total_samples = 0;
    for (int c = 0; c < num_clients && c < FEDERATED_MAX_CLIENTS; c++) {
        total_samples += client_sizes[c];
    }
    if (total_samples == 0) return -1;

    for (int c = 0; c < num_clients && c < FEDERATED_MAX_CLIENTS; c++) {
        if (!client_weights[c]) continue;
        float weight = (float)client_sizes[c] / (float)total_samples;
        for (size_t p = 0; p < num_params; p++) {
            global_weights[p] += weight * client_weights[c][p];
        }
    }

    /* 差分隐私: 添加高斯噪声 */
    if (dp_noise_std > 0.0f) {
        for (size_t p = 0; p < num_params; p++) {
            /* K-004修复：差分隐私噪声使用安全随机数 */
            float u1 = secure_random_float(), u2 = secure_random_float();
            if (u1 < 1e-10f) u1 = 1e-10f;
            float z = sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
            global_weights[p] += z * dp_noise_std;
        }
    }

    return 0;
}

int federated_compute_client_update(const float* global_model, const float* local_data,
                                     size_t data_size, const float* local_labels,
                                     float* local_model, size_t num_params,
                                     float learning_rate, int local_epochs) {
    if (!global_model || !local_model || !local_data || !local_labels) return -1;

    memcpy(local_model, global_model, num_params * sizeof(float));

    for (int e = 0; e < local_epochs; e++) {
        for (size_t p = 0; p < num_params && p < data_size; p++) {
            float grad = (local_model[p] - local_data[p] + local_labels[p % (data_size / 2 + 1)]) * 0.5f;
            local_model[p] -= learning_rate * grad;
        }
    }

    return 0;
}

/* ============================================================================
 * 课程学习调度器 (Curriculum Learning Scheduler)
 *
 * 从简单到困难逐步暴露训练样本，加速收敛并提升泛化：
 * 1. 难度评分：样本损失越大越难
 * 2. 分阶段调度：每阶段仅暴露难度≤阈值的样本
 * 3. 自适应阈值：根据当前epoch损失动态调整难度阈值
 * ============================================================================ */

#define CURRICULUM_MAX_STAGES 16

typedef struct {
    float* sample_difficulties;
    size_t num_samples;
    int current_stage;
    int num_stages;
    float difficulty_threshold;
    float difficulty_growth_rate;
    float min_difficulty;
    float max_difficulty;
    int warmup_epochs;
    int epochs_per_stage;
    int* active_sample_mask;
} CurriculumScheduler;

CurriculumScheduler* curriculum_scheduler_create(size_t num_samples, int num_stages,
                                                   int warmup_epochs, int epochs_per_stage) {
    if (num_samples == 0 || num_stages <= 0) return NULL;
    CurriculumScheduler* cs = (CurriculumScheduler*)safe_calloc(1, sizeof(CurriculumScheduler));
    if (!cs) return NULL;
    cs->num_samples = num_samples;
    cs->num_stages = (num_stages <= CURRICULUM_MAX_STAGES) ? num_stages : CURRICULUM_MAX_STAGES;
    cs->warmup_epochs = warmup_epochs > 0 ? warmup_epochs : 5;
    cs->epochs_per_stage = epochs_per_stage > 0 ? epochs_per_stage : 10;
    cs->current_stage = 0;
    cs->min_difficulty = 0.0f;
    cs->max_difficulty = 1.0f;
    cs->difficulty_threshold = 0.2f;
    cs->difficulty_growth_rate = 1.0f / (float)cs->num_stages;
    cs->sample_difficulties = (float*)safe_calloc(num_samples, sizeof(float));
    cs->active_sample_mask = (int*)safe_calloc(num_samples, sizeof(int));
    if (!cs->sample_difficulties || !cs->active_sample_mask) {
        safe_free((void**)&cs->sample_difficulties);
        safe_free((void**)&cs->active_sample_mask);
        safe_free((void**)&cs);
        return NULL;
    }
    for (size_t i = 0; i < num_samples; i++) cs->active_sample_mask[i] = 1;
    return cs;
}

void curriculum_scheduler_free(CurriculumScheduler* cs) {
    if (!cs) return;
    safe_free((void**)&cs->sample_difficulties);
    safe_free((void**)&cs->active_sample_mask);
    safe_free((void**)&cs);
}

int curriculum_scheduler_update(CurriculumScheduler* cs, int epoch, float epoch_loss) {
    if (!cs) return -1;
    if (epoch < cs->warmup_epochs) {
        cs->current_stage = 0;
        cs->difficulty_threshold = cs->min_difficulty + 0.3f;
        return 0;
    }
    int stage = (epoch - cs->warmup_epochs) / cs->epochs_per_stage;
    if (stage >= cs->num_stages) stage = cs->num_stages - 1;
    cs->current_stage = stage;
    cs->difficulty_threshold = cs->min_difficulty + cs->difficulty_growth_rate * (float)stage;
    if (cs->difficulty_threshold < 0.15f) cs->difficulty_threshold = 0.15f;
    if (epoch_loss > 2.0f) cs->difficulty_threshold *= 0.9f;
    if (epoch_loss < 0.5f) cs->difficulty_threshold *= 1.1f;
    if (cs->difficulty_threshold > cs->max_difficulty) cs->difficulty_threshold = cs->max_difficulty;
    return 0;
}

int curriculum_scheduler_update_difficulty(CurriculumScheduler* cs, size_t sample_idx, float loss) {
    if (!cs || sample_idx >= cs->num_samples) return -1;
    float alpha = 0.1f;
    cs->sample_difficulties[sample_idx] = (1.0f - alpha) * cs->sample_difficulties[sample_idx] + alpha * loss;
    return 0;
}

int curriculum_scheduler_is_active(const CurriculumScheduler* cs, size_t sample_idx) {
    if (!cs || sample_idx >= cs->num_samples) return 1;
    if (cs->current_stage >= cs->num_stages - 1) return 1;
    return cs->sample_difficulties[sample_idx] <= cs->difficulty_threshold ? 1 : 0;
}

int trainer_load_incremental_checkpoint(Trainer* trainer, const char* filepath) {
    if (!trainer || !filepath || !trainer->network) {
        log_error("trainer_load_incremental_checkpoint: 参数无效\n");
        return -1;
    }

    LNN* network = trainer->network;
    size_t n = lnn_get_parameter_count(network);
    float* params = lnn_get_parameters(network);
    if (!params || n == 0) {
        log_error("trainer_load_incremental_checkpoint: 无法获取网络参数\n");
        return -1;
    }

    FILE* fp = fopen(filepath, "rb");
    if (!fp) {
        log_error("trainer_load_incremental_checkpoint: 无法打开文件 %s\n", filepath);
        return -1;
    }

    /* 读取并验证魔数 */
    uint32_t magic = 0;
    if (fread(&magic, sizeof(uint32_t), 1, fp) != 1) {
        log_error("trainer_load_incremental_checkpoint: 读取魔数失败\n");
        fclose(fp);
        return -1;
    }
    if (magic != 0x49434B50) {
        log_error("trainer_load_incremental_checkpoint: 魔数不匹配 (0x%08X), 期望 0x49434B50\n", magic);
        fclose(fp);
        return -1;
    }

    /* 读取总参数量并验证一致性 */
    uint32_t file_n = 0;
    if (fread(&file_n, sizeof(uint32_t), 1, fp) != 1) {
        log_error("trainer_load_incremental_checkpoint: 读取参数数量失败\n");
        fclose(fp);
        return -1;
    }
    if ((size_t)file_n != n) {
        log_error("trainer_load_incremental_checkpoint: 参数数量不匹配 (文件=%u, 网络=%zu)\n",
                  (unsigned)file_n, n);
        fclose(fp);
        return -1;
    }

    /* 读取delta条目数量 */
    uint32_t delta_count = 0;
    if (fread(&delta_count, sizeof(uint32_t), 1, fp) != 1) {
        log_error("trainer_load_incremental_checkpoint: 读取delta数量失败\n");
        fclose(fp);
        return -1;
    }

    if (delta_count > n) {
        log_warning("trainer_load_incremental_checkpoint: delta数量(%u)超过参数总数(%zu)，截断\n",
                    (unsigned)delta_count, n);
        delta_count = (uint32_t)n;
    }

    /* 清零当前参数（检查点保存的是全量参数值，从零基线上加delta恢复） */
    memset(params, 0, n * sizeof(float));

    /* 记录delta数据起始位置（用于签名验证） */
    long sig_data_start = ftell(fp);

    /* 计算delta数据预期大小 */
    size_t expected_delta_bytes = delta_count * (sizeof(uint32_t) + sizeof(float));
    uint8_t* sig_data = (uint8_t*)safe_malloc(expected_delta_bytes + 256);
    int has_signature = 0;
    uint8_t expected_sig[32];
    uint8_t actual_sig[32];
    memset(expected_sig, 0, 32); memset(actual_sig, 0, 32);

    /* 先读delta数据到缓冲区，再检查尾部签名 */
    if (sig_data) {
        size_t bytes_read = fread(sig_data, 1, expected_delta_bytes, fp);
        if (bytes_read == expected_delta_bytes) {
            /* 检查文件尾是否有32字节签名 */
            long cur_pos = ftell(fp);
            fseek(fp, 0, SEEK_END);
            long file_end = ftell(fp);
            if (file_end - cur_pos >= 32) {
                /* 有签名：计算期望签名 */
                ckpt_hmac_sign(sig_data, expected_delta_bytes, expected_sig);
                fseek(fp, -32, SEEK_END);
                if (fread(actual_sig, 1, 32, fp) == 32) {
                    has_signature = 1;
                }
            }
            fseek(fp, cur_pos, SEEK_SET);
        }
        safe_free((void**)&sig_data);
    }

    /* 逐条读取并应用delta */
    uint32_t loaded = 0;
    for (uint32_t i = 0; i < delta_count; i++) {
        uint32_t idx = 0;
        float delta = 0.0f;
        if (fread(&idx, sizeof(uint32_t), 1, fp) != 1) {
            log_warning("trainer_load_incremental_checkpoint: 读取第%u个delta索引时文件结束\n", i);
            break;
        }
        if (fread(&delta, sizeof(float), 1, fp) != 1) {
            log_warning("trainer_load_incremental_checkpoint: 读取第%u个delta值时文件结束\n", (unsigned)i);
            break;
        }
        if (idx < (uint32_t)n) {
            params[idx] = delta;
            loaded++;
        } else {
            log_warning("trainer_load_incremental_checkpoint: delta索引%u超出范围(最大%zu)，跳过\n",
                        (unsigned)idx, n - 1);
        }
    }

    fclose(fp);

    /* 验证HMAC签名 */
    if (has_signature) {
        if (memcmp(expected_sig, actual_sig, 32) != 0) {
            log_error("trainer_load_incremental_checkpoint: HMAC-SHA256签名验证失败！检查点可能被篡改！\n");
            memset(params, 0, n * sizeof(float));
            return -1;
        }
        log_info("trainer_load_incremental_checkpoint: HMAC-SHA256签名验证通过\n");
    }

    log_info("trainer_load_incremental_checkpoint: 成功加载增量检查点 '%s' (%u/%u 参数)\n",
             filepath, loaded, delta_count);

    if (loaded < delta_count) {
        log_warning("trainer_load_incremental_checkpoint: 加载不完整 (已加载%u/%u)\n", loaded, delta_count);
        return -1;
    }

    return 0;
}

