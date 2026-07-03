#include "selflnn/backend/auth.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/core/errors.h"
/* ZSFJJJ-01: 使用共享SHA-256实现 */
#include "selflnn/utils/math_utils.h"

#include "selflnn/utils/secure_random.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#endif

/* ZSFJJJ-01: SHA-256实现已移至 math_utils.c，此处使用共享实现 */

/* ZSFJJJ-01: HMAC-SHA256 使用共享SHA-256流式接口 */
static void hmac_sha256(const uint8_t* key, size_t key_len,
                        const uint8_t* msg, size_t msg_len,
                        uint8_t output[AUTH_HASH_LEN]) {
    uint8_t k_ipad[64], k_opad[64];
    uint8_t inner_hash[AUTH_HASH_LEN];
    memset(k_ipad, 0, 64); memset(k_opad, 0, 64);
    size_t klen = key_len < 64 ? key_len : 64;
    memcpy(k_ipad, key, klen);
    memcpy(k_opad, key, klen);
    for (size_t i = 0; i < 64; i++) {
        k_ipad[i] ^= 0x36;
        k_opad[i] ^= 0x5c;
    }
    selflnn_sha256_ctx_t ctx;
    selflnn_sha256_init(&ctx);
    selflnn_sha256_update(&ctx, k_ipad, 64);
    selflnn_sha256_update(&ctx, msg, msg_len);
    selflnn_sha256_final(&ctx, inner_hash);
    selflnn_sha256_init(&ctx);
    selflnn_sha256_update(&ctx, k_opad, 64);
    selflnn_sha256_update(&ctx, inner_hash, AUTH_HASH_LEN);
    selflnn_sha256_final(&ctx, output);
}

static void generate_random_bytes(uint8_t* output, size_t len) {
    secure_random_bytes(output, len);
}

/**
 * @brief 使用随机盐值对密钥哈希进行加盐哈希存储
 * 
 * 存储格式：salt[16] || SHA256(salt || key_hash)
 * 即使数据库泄露，攻击者无法反向推导原始API密钥
 */
static void salted_hash_key(const uint8_t* key_hash, uint8_t* out_salt, uint8_t* out_salted_hash) {
    generate_random_bytes(out_salt, AUTH_SALT_LEN);
    uint8_t concat[AUTH_SALT_LEN + AUTH_HASH_LEN];
    memcpy(concat, out_salt, AUTH_SALT_LEN);
    memcpy(concat + AUTH_SALT_LEN, key_hash, AUTH_HASH_LEN);
    selflnn_sha256_hash(concat, sizeof(concat), out_salted_hash);
}

/**
 * @brief 验证密钥：用存储的盐值对输入哈希进行加盐哈希后对比
 * @return 1=匹配，0=不匹配
 */
static int salted_verify_key(const uint8_t* stored_salt, const uint8_t* stored_salted_hash,
                              const uint8_t* input_key_hash) {
    uint8_t concat[AUTH_SALT_LEN + AUTH_HASH_LEN];
    memcpy(concat, stored_salt, AUTH_SALT_LEN);
    memcpy(concat + AUTH_SALT_LEN, input_key_hash, AUTH_HASH_LEN);
    uint8_t computed_hash[AUTH_HASH_LEN];
    selflnn_sha256_hash(concat, sizeof(concat), computed_hash);
    return memcmp(computed_hash, stored_salted_hash, AUTH_HASH_LEN) == 0 ? 1 : 0;
}

/* P2注释: 手动十六进制转换 - 避免依赖sprintf/printf系列函数，纯C实现
 * 将哈希字节转换为 "sk-xxxx-xxxx-xxxx-xxxx..." 格式的人类可读密钥字符串 */
static void key_to_string(const uint8_t* hash, char* str, size_t str_len) {
    const char hex[] = "0123456789abcdef";
    if (str_len < 3) return;
    str[0] = 's'; str[1] = 'k'; str[2] = '-';
    size_t pos = 3;
    for (size_t i = 0; i < AUTH_HASH_LEN && pos + 1 < str_len; i++) {
        str[pos++] = hex[(hash[i] >> 4) & 0x0f];
        str[pos++] = hex[hash[i] & 0x0f];
        if ((i % 4) == 3 && pos < str_len - 1 && i < AUTH_HASH_LEN - 1)
            str[pos++] = '-';
    }
    if (pos < str_len) str[pos] = '\0';
}

static int string_to_key(const char* str, uint8_t* hash) {
    if (!str || strncmp(str, "sk-", 3) != 0) return -1;
    const char* p = str + 3;
    size_t hex_pos = 0;
    uint8_t hex_buf[AUTH_HASH_LEN * 2];
    while (*p && hex_pos < sizeof(hex_buf)) {
        if (*p == '-') { p++; continue; }
        char c = *p;
        if (c >= '0' && c <= '9') hex_buf[hex_pos++] = (uint8_t)(c - '0');
        else if (c >= 'a' && c <= 'f') hex_buf[hex_pos++] = (uint8_t)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') hex_buf[hex_pos++] = (uint8_t)(c - 'A' + 10);
        else return -1;
        p++;
    }
    if (hex_pos < AUTH_HASH_LEN * 2) return -1;
    for (size_t i = 0; i < AUTH_HASH_LEN; i++)
        hash[i] = (hex_buf[i*2] << 4) | hex_buf[i*2+1];
    return 0;
}

/* 使用高精度时间函数替代 clock()
 * clock() 在某些平台分辨率仅为10ms，影响令牌桶精度。
 * Windows使用GetTickCount64（毫秒精度），POSIX使用clock_gettime */
static uint64_t get_current_time_ms(void) {
#ifdef _WIN32
    return GetTickCount64();
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000 + (uint64_t)ts.tv_nsec / 1000000;
#endif
}

static void token_bucket_init(TokenBucket* bucket, uint64_t capacity, double refill_rate_per_sec) {
    bucket->capacity = capacity;
    bucket->tokens = capacity;
    bucket->refill_rate = refill_rate_per_sec / 1000.0;
    bucket->last_refill_time_ms = get_current_time_ms();
    bucket->max_tokens = capacity;
    bucket->tokens_consumed_total = 0;
    bucket->tokens_refilled_total = 0;
}

static void token_bucket_refill(TokenBucket* bucket) {
    uint64_t now = get_current_time_ms();
    uint64_t elapsed = now - bucket->last_refill_time_ms;
    if (elapsed < 1) return;
    double new_tokens = (double)elapsed * bucket->refill_rate;
    if (new_tokens > 0.0) {
        bucket->tokens_refilled_total += (uint64_t)new_tokens;
        bucket->tokens += (uint64_t)new_tokens;
        if (bucket->tokens > bucket->max_tokens)
            bucket->tokens = bucket->max_tokens;
        bucket->last_refill_time_ms = now;
    }
}

static int token_bucket_consume(TokenBucket* bucket, uint64_t tokens) {
    token_bucket_refill(bucket);
    if (bucket->tokens >= tokens) {
        bucket->tokens -= tokens;
        bucket->tokens_consumed_total++;
        return 1;
    }
    return 0;
}

AuthSystem* auth_system_create(int require_auth) {
    AuthSystem* auth = (AuthSystem*)safe_calloc(1, sizeof(AuthSystem));
    if (!auth) return NULL;
    auth->auth_required = require_auth;
    auth->key_count = 0;
    auth->rule_count = 0;
    auth->initialized = 1;
    token_bucket_init(&auth->global_bucket, 1000, 100.0);
    return auth;
}

void auth_system_free(AuthSystem* auth) {
    if (!auth) return;
    memset(auth->keys, 0, sizeof(auth->keys));
    auth->key_count = 0;
    auth->initialized = 0;
    safe_free((void**)&auth);
}

int auth_generate_key(AuthSystem* auth, const char* name, AuthPermission permission,
                      int expiration_days, char* key_out, size_t key_out_size) {
    if (!auth || !name || !key_out) return -1;
    if (auth->key_count >= AUTH_MAX_KEYS) return -2;

    uint8_t random_bytes[32];
    generate_random_bytes(random_bytes, 32);

    /* P1-011修复: 使用随机盐替代硬编码主盐值，确保不同API密钥使用不同盐 */
    uint8_t master_salt[16];
    generate_random_bytes(master_salt, 16);
    for (size_t i = 0; i < 16; i++) master_salt[i] = (uint8_t)(master_salt[i] ^ random_bytes[i % 32]);
    time_t now = time(NULL);
    uint8_t time_bytes[8];
    for (size_t i = 0; i < 8; i++)
        time_bytes[i] = (uint8_t)((uint64_t)now >> (i*8));

    uint8_t hmac_input[64];
    memcpy(hmac_input, random_bytes, 32);
    memcpy(hmac_input+32, time_bytes, 8);
    memcpy(hmac_input+40, master_salt, 16);
    size_t name_len = strlen(name);
    if (name_len > 8) name_len = 8;
    if (name_len > 0) memcpy(hmac_input+56, name, name_len);

    uint8_t key_hash[AUTH_HASH_LEN];
    hmac_sha256(random_bytes, 32, hmac_input, 56+name_len, key_hash);

    key_to_string(key_hash, key_out, key_out_size);

    AuthKeyEntry* entry = &auth->keys[auth->key_count];
    /* 加盐哈希存储（不存储明文密钥，防御数据库泄露） */
    salted_hash_key(key_hash, entry->salt, entry->hash);
    memcpy(entry->key_prefix, key_out, 15);
    entry->key_prefix[15] = '\0';
    entry->permission = permission;
    entry->enabled = 1;
    entry->created_at = now;
    entry->expires_at = (expiration_days > 0) ? (now + (time_t)expiration_days * 86400) : 0;
    entry->last_used_at = 0;
    entry->usage_count = 0;
    entry->is_encrypted = 1;  /* 标记已加盐哈希存储 */
    strncpy(entry->name, name, AUTH_KEY_NAME_LEN - 1);
    entry->name[AUTH_KEY_NAME_LEN - 1] = '\0';
    auth->key_count++;
    return 0;
}

int auth_validate_key(AuthSystem* auth, const char* key, AuthPermission required_permission) {
    if (!auth || !key) return 0;
    uint8_t key_hash[AUTH_HASH_LEN];
    if (string_to_key(key, key_hash) != 0) return 0;
    for (int i = 0; i < auth->key_count; i++) {
        /* 使用SHA-256加盐哈希验证（强制，不再支持明文memcmp回退） */
        int match = salted_verify_key(auth->keys[i].salt, auth->keys[i].hash, key_hash);
        if (match) {
            if (!auth->keys[i].enabled) return 0;
            if (auth->keys[i].expires_at > 0 && time(NULL) > auth->keys[i].expires_at) {
                auth->keys[i].enabled = 0;
                return 0;
            }
            if (auth->keys[i].permission < required_permission) return 0;
            auth->keys[i].last_used_at = time(NULL);
            auth->keys[i].usage_count++;
            return 1;
        }
    }
    return 0;
}

int auth_revoke_key(AuthSystem* auth, const char* key_or_prefix) {
    if (!auth || !key_or_prefix) return -1;
    for (int i = 0; i < auth->key_count; i++) {
        if (strncmp(auth->keys[i].key_prefix, key_or_prefix, 
                    strlen(key_or_prefix) < 15 ? strlen(key_or_prefix) : 15) == 0) {
            /* B-013修复: 先memmove覆盖目标位置，再清零尾部空余条目。
             * 避免先memset再memmove时，若memmove失败原始数据无法恢复的问题。 */
            size_t shift_count = (size_t)(auth->key_count - i - 1);
            if (shift_count > 0) {
                memmove(&auth->keys[i], &auth->keys[i + 1], shift_count * sizeof(AuthKeyEntry));
            }
            memset(&auth->keys[auth->key_count - 1], 0, sizeof(AuthKeyEntry));
            auth->key_count--;
            return 0;
        }
    }
    return -1;
}

int auth_enable_key(AuthSystem* auth, const char* key_or_prefix, int enable) {
    if (!auth || !key_or_prefix) return -1;
    for (int i = 0; i < auth->key_count; i++) {
        size_t prefix_len = strlen(key_or_prefix);
        if (prefix_len > 15) prefix_len = 15;
        if (strncmp(auth->keys[i].key_prefix, key_or_prefix, prefix_len) == 0) {
            auth->keys[i].enabled = enable;
            return 0;
        }
    }
    return -1;
}

int auth_list_keys(AuthSystem* auth, AuthKeyEntry* entries, size_t* count, size_t max_count) {
    if (!auth || !entries || !count) return -1;
    size_t copy_count = (size_t)auth->key_count < max_count ? (size_t)auth->key_count : max_count;
    memcpy(entries, auth->keys, copy_count * sizeof(AuthKeyEntry));
    *count = copy_count;
    return 0;
}

int auth_get_stats(AuthSystem* auth, AuthStats* stats) {
    if (!auth || !stats) return -1;
    memset(stats, 0, sizeof(AuthStats));
    stats->total_keys = auth->key_count;
    stats->active_keys = 0;
    stats->requests_total = 0;
    for (int i = 0; i < auth->key_count; i++) {
        if (auth->keys[i].enabled) stats->active_keys++;
        stats->requests_total += auth->keys[i].usage_count;
    }
    stats->requests_today = 0;
    time_t now = time(NULL);
    struct tm* tm_now = localtime(&now);
    time_t today_start = (time_t)mktime(tm_now) - (tm_now->tm_hour * 3600 + tm_now->tm_min * 60 + tm_now->tm_sec);
    for (int i = 0; i < auth->key_count; i++) {
        if (auth->keys[i].last_used_at >= today_start)
            stats->requests_today += auth->keys[i].usage_count;
    }
    stats->global_bucket_capacity = (int)(auth->global_bucket.max_tokens);
    stats->global_bucket_remaining = (int)(auth->global_bucket.tokens);
    stats->global_bucket_refill_per_sec = (int)(auth->global_bucket.refill_rate * 1000.0);
    stats->auth_failures = 0;
    stats->rate_limited = 0;
    return 0;
}

int auth_register_endpoint_rule(AuthSystem* auth, const char* endpoint_pattern,
                                AuthPermission min_permission, int require_auth) {
    if (!auth || !endpoint_pattern || auth->rule_count >= AUTH_MAX_API_ENDPOINTS) return -1;
    AuthEndpointRule* rule = &auth->endpoint_rules[auth->rule_count++];
    strncpy(rule->endpoint, endpoint_pattern, 127);
    rule->endpoint[127] = '\0';
    rule->min_permission = min_permission;
    rule->require_auth = require_auth;
    return 0;
}

AuthPermission auth_get_required_permission(AuthSystem* auth, const char* endpoint) {
    if (!auth || !endpoint) return AUTH_PERM_NONE;
    for (int i = 0; i < auth->rule_count; i++) {
        if (strstr(endpoint, auth->endpoint_rules[i].endpoint) != NULL)
            return auth->endpoint_rules[i].min_permission;
    }
    return AUTH_PERM_READONLY;
}

int auth_check_rate_limit(AuthSystem* auth, TokenBucket* bucket) {
    if (!bucket) return 0;
    /* 自适应限流：利用认证上下文全局负载状态动态调整限流粒度 */
    if (auth && auth->initialized) {
        /* 若全局令牌桶余量低于25%，触发更严格的限流：消耗双倍令牌 */
        if (auth->global_bucket.tokens < auth->global_bucket.max_tokens / 4) {
            return token_bucket_consume(bucket, 2) ? 1 : 0;
        }
    }
    return token_bucket_consume(bucket, 1) ? 1 : 0;
}

int auth_global_check_rate(AuthSystem* auth) {
    if (!auth) return 0;
    return token_bucket_consume(&auth->global_bucket, 1) ? 1 : 0;
}

/* M-017修复: Bearer Token为统一主认证方式，旧式api_key仅作兼容保留 */
/* 认证优先级: 1.Bearer Token(Authorization头) > 2.旧式api_key(请求体JSON字段) */
int auth_check_request(AuthSystem* auth, const char* endpoint,
                       const char* auth_header, AuthPermission* out_permission) {
    if (!auth || !endpoint) return 0;
    
    if (!auth->auth_required) {
        if (out_permission) *out_permission = AUTH_PERM_READONLY;
        return 1;
    }
    
    AuthPermission required = auth_get_required_permission(auth, endpoint);
    
    if (!auth_header) return 0;
    
    /* 仅接受标准Bearer Token格式（主认证方式） */
    if (strncmp(auth_header, "Bearer ", 7) != 0) return 0;
    const char* token = auth_header + 7;
    
    int valid = auth_validate_key(auth, token, required);
    if (valid) {
        if (out_permission) *out_permission = required;
        return 1;
    }
    return 0;
}

/* M-017: 旧式API密钥兼容验证（从请求体JSON中提取api_key字段直接验证） */
/* 仅作为向后兼容保留，新客户端应使用Bearer Token方式 */
int auth_validate_legacy_key(AuthSystem* auth, const char* key, AuthPermission required_permission) {
    if (!auth || !key || key[0] == '\0') return 0;
    return auth_validate_key(auth, key, required_permission);
}

static void xor_encrypt_decrypt(uint8_t* data, size_t len, const uint8_t* key, size_t key_len) {
    for (size_t i = 0; i < len; i++)
        data[i] ^= key[i % key_len];
}

int auth_save_keys(AuthSystem* auth, const char* filepath) {
/* 使用salt+SHA-256多轮迭代的HKDF风格密钥派生替代简单XOR，
     * 增强加密密钥安全性。原实现仅对filepath字符简单XOR，攻击者可轻易推测。 */
    if (!auth || !filepath) return -1;
    FILE* fp = fopen(filepath, "wb");
    if (!fp) return -1;
    
    uint8_t header[16] = "SELFLNN_AUTH_V3"; /* V3表示HKDF加密格式 */
    fwrite(header, 1, 16, fp);
    
    int count = auth->key_count;
    fwrite(&count, sizeof(int), 1, fp);
    
    /* 生成随机盐值并写入文件 */
    uint8_t salt[16];
    generate_random_bytes(salt, 16);
    fwrite(salt, 1, 16, fp);
    
    size_t keys_size = (size_t)count * sizeof(AuthKeyEntry);
    uint8_t* keys_copy = (uint8_t*)safe_malloc(keys_size);
    if (keys_copy) {
        memcpy(keys_copy, auth->keys, keys_size);
        /* 派生加密密钥：SHA-256(salt || SHA-256(salt || filepath || salt)) */
        size_t path_len = strlen(filepath);
        if (path_len > 200) path_len = 200;
        uint8_t kdf_buf[232];  /* 16(salt) + 200(path) + 16(salt) */
        memcpy(kdf_buf, salt, 16);
        memcpy(kdf_buf + 16, filepath, path_len);
        memcpy(kdf_buf + 16 + path_len, salt, 16);
        
        uint8_t h1[AUTH_HASH_LEN];
        selflnn_sha256_hash(kdf_buf, 16 + path_len + 16, h1);
        
        /* 第二轮：SHA-256(salt || h1) */
        memcpy(kdf_buf, salt, 16);
        memcpy(kdf_buf + 16, h1, AUTH_HASH_LEN);
        uint8_t xor_key[AUTH_HASH_LEN];
        selflnn_sha256_hash(kdf_buf, 16 + AUTH_HASH_LEN, xor_key);
        
        for (size_t i = 0; i < keys_size; i++) keys_copy[i] ^= xor_key[i % AUTH_HASH_LEN];
        fwrite(keys_copy, 1, keys_size, fp);
        safe_free((void**)&keys_copy);
    } else {
        fwrite(auth->keys, sizeof(AuthKeyEntry), (size_t)count, fp);
    }
    
    fclose(fp);
    return 0;
}

int auth_load_keys(AuthSystem* auth, const char* filepath) {
    if (!auth || !filepath) return -1;
    FILE* fp = fopen(filepath, "rb");
    if (!fp) return -1;
    
    uint8_t header[16];
    int is_encrypted = 0;
    int is_v3 = 0;
    if (fread(header, 1, 16, fp) != 16) { fclose(fp); return -1; }
    
/* 检测加密格式版本 */
    if (memcmp(header, "SELFLNN_AUTH_V3", 16) == 0) {
        is_encrypted = 1;
        is_v3 = 1;
    } else if (memcmp(header, "SELFLNN_AUTH_V2", 16) == 0) {
        is_encrypted = 1;
    } else if (memcmp(header, "SELFLNN_AUTH_V1", 16) != 0) {
        fclose(fp);
        return -1;
    }
    
    int count = 0;
    if (fread(&count, sizeof(int), 1, fp) != 1 || count > AUTH_MAX_KEYS) {
        fclose(fp);
        return -1;
    }

    /* V3格式：读取盐值 */
    uint8_t salt[16] = {0};
    if (is_v3) {
        if (fread(salt, 1, 16, fp) != 16) {
            fclose(fp);
            return -1;
        }
    }
    
    size_t keys_size = (size_t)count * sizeof(AuthKeyEntry);
    if (is_encrypted) {
        uint8_t* keys_encrypted = (uint8_t*)safe_malloc(keys_size);
        if (!keys_encrypted || fread(keys_encrypted, 1, keys_size, fp) != keys_size) {
            free(keys_encrypted);
            fclose(fp);
            return -1;
        }
        if (is_v3) {
/* V3格式使用salt+HKDF风格密钥派生解密 */
            size_t path_len = strlen(filepath);
            if (path_len > 200) path_len = 200;
            uint8_t kdf_buf[232];
            memcpy(kdf_buf, salt, 16);
            memcpy(kdf_buf + 16, filepath, path_len);
            memcpy(kdf_buf + 16 + path_len, salt, 16);
            
            uint8_t h1[AUTH_HASH_LEN];
            selflnn_sha256_hash(kdf_buf, 16 + path_len + 16, h1);
            
            memcpy(kdf_buf, salt, 16);
            memcpy(kdf_buf + 16, h1, AUTH_HASH_LEN);
            uint8_t xor_key[AUTH_HASH_LEN];
            selflnn_sha256_hash(kdf_buf, 16 + AUTH_HASH_LEN, xor_key);
            
            for (size_t i = 0; i < keys_size; i++) keys_encrypted[i] ^= xor_key[i % AUTH_HASH_LEN];
        } else {
            /* V2格式：兼容旧的简单XOR密钥派生 */
            uint8_t xor_key[32];
            memset(xor_key, 0, 32);
            size_t path_len = strlen(filepath);
            for (size_t i = 0; i < path_len; i++) xor_key[i % 32] ^= (uint8_t)filepath[i];
            for (size_t i = 0; i < keys_size; i++) keys_encrypted[i] ^= xor_key[i % 32];
        }
        memcpy(auth->keys, keys_encrypted, keys_size);
        safe_free((void**)&keys_encrypted);
    } else {
        if (fread(auth->keys, sizeof(AuthKeyEntry), (size_t)count, fp) != (size_t)count) {
            fclose(fp);
            return -1;
        }
    }
    auth->key_count = count;
    
    fclose(fp);
    return 0;
}
