#ifndef SELFLNN_BACKEND_AUTH_H
#define SELFLNN_BACKEND_AUTH_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>
#include <time.h>
#include "selflnn/backend/backend.h"

#define AUTH_MAX_KEYS 64
#define AUTH_KEY_STRING_LEN 128
#define AUTH_KEY_NAME_LEN 64
#define AUTH_HASH_LEN 32
#define AUTH_SALT_LEN 16
#define AUTH_MAX_API_ENDPOINTS 256

typedef enum {
    AUTH_PERM_NONE = 0,
    AUTH_PERM_READONLY = 1,
    AUTH_PERM_READWRITE = 2,
    AUTH_PERM_ADMIN = 3
} AuthPermission;

typedef struct {
    uint8_t hash[AUTH_HASH_LEN];       /**< 密钥的加盐SHA-256哈希（不存储明文） */
    uint8_t salt[AUTH_SALT_LEN];       /**< 每密钥独立随机盐值 */
    char key_prefix[16];               /**< 密钥前缀（用于列表显示，不用于验证） */
    AuthPermission permission;
    int enabled;
    time_t created_at;
    time_t expires_at;
    time_t last_used_at;
    int usage_count;
    char name[AUTH_KEY_NAME_LEN];
    int is_encrypted;                  /**< 标记是否已使用加盐哈希存储 */
} AuthKeyEntry;

typedef struct {
    uint64_t capacity;
    uint64_t tokens;
    double refill_rate;
    uint64_t last_refill_time_ms;
    uint64_t max_tokens;
    uint64_t tokens_consumed_total;
    uint64_t tokens_refilled_total;
} TokenBucket;

typedef struct {
    char endpoint[128];
    AuthPermission min_permission;
    int require_auth;
} AuthEndpointRule;

/**
 * @brief 增强认证统计信息
 */
typedef struct {
    int active_keys;                    /**< 活跃密钥数 */
    int total_keys;                     /**< 总密钥数 */
    int requests_today;                 /**< 今日请求数 */
    int requests_total;                 /**< 总请求数 */
    int global_bucket_capacity;         /**< 全局令牌桶容量 */
    int global_bucket_remaining;        /**< 全局令牌桶剩余令牌 */
    int global_bucket_refill_per_sec;   /**< 全局令牌桶每秒补充率 */
    int auth_failures;                  /**< 认证失败次数 */
    int rate_limited;                   /**< 被限流次数 */
} AuthStats;

/**
 * @brief 认证系统实例（不透明类型）
 */
typedef struct {
    AuthKeyEntry keys[AUTH_MAX_KEYS];
    int key_count;
    TokenBucket global_bucket;
    AuthEndpointRule endpoint_rules[AUTH_MAX_API_ENDPOINTS];
    int rule_count;
    int auth_required;
    int initialized;
} AuthSystem;

AuthSystem* auth_system_create(int require_auth);
void auth_system_free(AuthSystem* auth);

int auth_generate_key(AuthSystem* auth, const char* name, AuthPermission permission,
                      int expiration_days, char* key_out, size_t key_out_size);
int auth_validate_key(AuthSystem* auth, const char* key, AuthPermission required_permission);
int auth_revoke_key(AuthSystem* auth, const char* key_or_prefix);
int auth_enable_key(AuthSystem* auth, const char* key_or_prefix, int enable);
int auth_list_keys(AuthSystem* auth, AuthKeyEntry* entries, size_t* count, size_t max_count);
int auth_get_stats(AuthSystem* auth, AuthStats* stats);

int auth_register_endpoint_rule(AuthSystem* auth, const char* endpoint_pattern,
                                AuthPermission min_permission, int require_auth);
AuthPermission auth_get_required_permission(AuthSystem* auth, const char* endpoint);

int auth_check_rate_limit(AuthSystem* auth, TokenBucket* bucket);
int auth_global_check_rate(AuthSystem* auth);
int auth_check_request(AuthSystem* auth, const char* endpoint,
                       const char* auth_header, AuthPermission* out_permission);

int auth_save_keys(AuthSystem* auth, const char* filepath);
int auth_load_keys(AuthSystem* auth, const char* filepath);

#ifdef __cplusplus
}
#endif

#endif
