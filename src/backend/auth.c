#include "selflnn/backend/auth.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/core/errors.h"

#include "selflnn/utils/secure_random.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#define ROTLEFT(a,b) (((a) << (b)) | ((a) >> (32-(b))))
#define ROTRIGHT(a,b) (((a) >> (b)) | ((a) << (32-(b))))
#define CH(x,y,z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTRIGHT(x,2) ^ ROTRIGHT(x,13) ^ ROTRIGHT(x,22))
#define EP1(x) (ROTRIGHT(x,6) ^ ROTRIGHT(x,11) ^ ROTRIGHT(x,25))
#define SIG0(x) (ROTRIGHT(x,7) ^ ROTRIGHT(x,18) ^ ((x) >> 3))
#define SIG1(x) (ROTRIGHT(x,17) ^ ROTRIGHT(x,19) ^ ((x) >> 10))

static const uint32_t k[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,
    0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,
    0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,
    0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,
    0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,
    0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,
    0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,
    0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,
    0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

typedef struct {
    uint8_t data[64];
    uint32_t datalen;
    uint64_t bitlen;
    uint32_t state[8];
} SHA256_CTX;

static void sha256_transform(SHA256_CTX* ctx, const uint8_t data[]) {
    uint32_t a,b,c,d,e,f,g,h,i,j,t1,t2,m[64];
    for (i=0,j=0; i<16; i++,j+=4)
        m[i] = ((uint32_t)data[j]<<24) | ((uint32_t)data[j+1]<<16) | ((uint32_t)data[j+2]<<8) | (uint32_t)data[j+3];
    for (; i<64; i++)
        m[i] = SIG1(m[i-2]) + m[i-7] + SIG0(m[i-15]) + m[i-16];
    a = ctx->state[0]; b = ctx->state[1]; c = ctx->state[2]; d = ctx->state[3];
    e = ctx->state[4]; f = ctx->state[5]; g = ctx->state[6]; h = ctx->state[7];
    for (i=0; i<64; i++) {
        t1 = h + EP1(e) + CH(e,f,g) + k[i] + m[i];
        t2 = EP0(a) + MAJ(a,b,c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    ctx->state[0] += a; ctx->state[1] += b; ctx->state[2] += c; ctx->state[3] += d;
    ctx->state[4] += e; ctx->state[5] += f; ctx->state[6] += g; ctx->state[7] += h;
}

static void sha256_init(SHA256_CTX* ctx) {
    ctx->datalen = 0; ctx->bitlen = 0;
    ctx->state[0] = 0x6a09e667; ctx->state[1] = 0xbb67ae85;
    ctx->state[2] = 0x3c6ef372; ctx->state[3] = 0xa54ff53a;
    ctx->state[4] = 0x510e527f; ctx->state[5] = 0x9b05688c;
    ctx->state[6] = 0x1f83d9ab; ctx->state[7] = 0x5be0cd19;
}

static void sha256_update(SHA256_CTX* ctx, const uint8_t data[], size_t len) {
    for (size_t i=0; i<len; i++) {
        ctx->data[ctx->datalen] = data[i]; ctx->datalen++;
        if (ctx->datalen == 64) {
            sha256_transform(ctx, ctx->data);
            ctx->bitlen += 512;
            ctx->datalen = 0;
        }
    }
}

static void sha256_final(SHA256_CTX* ctx, uint8_t hash[AUTH_HASH_LEN]) {
    size_t i = ctx->datalen;
    if (ctx->datalen < 56) {
        ctx->data[i++] = 0x80;
        while (i < 56) ctx->data[i++] = 0x00;
    } else {
        ctx->data[i++] = 0x80;
        while (i < 64) ctx->data[i++] = 0x00;
        sha256_transform(ctx, ctx->data);
        memset(ctx->data, 0, 56);
    }
    ctx->bitlen += ctx->datalen * 8;
    ctx->data[63] = (uint8_t)(ctx->bitlen);
    ctx->data[62] = (uint8_t)(ctx->bitlen >> 8);
    ctx->data[61] = (uint8_t)(ctx->bitlen >> 16);
    ctx->data[60] = (uint8_t)(ctx->bitlen >> 24);
    ctx->data[59] = (uint8_t)(ctx->bitlen >> 32);
    ctx->data[58] = (uint8_t)(ctx->bitlen >> 40);
    ctx->data[57] = (uint8_t)(ctx->bitlen >> 48);
    ctx->data[56] = (uint8_t)(ctx->bitlen >> 56);
    sha256_transform(ctx, ctx->data);
    for (i=0; i<4; i++) {
        hash[i]    = (ctx->state[0] >> (24 - i*8)) & 0x000000ff;
        hash[i+4]  = (ctx->state[1] >> (24 - i*8)) & 0x000000ff;
        hash[i+8]  = (ctx->state[2] >> (24 - i*8)) & 0x000000ff;
        hash[i+12] = (ctx->state[3] >> (24 - i*8)) & 0x000000ff;
        hash[i+16] = (ctx->state[4] >> (24 - i*8)) & 0x000000ff;
        hash[i+20] = (ctx->state[5] >> (24 - i*8)) & 0x000000ff;
        hash[i+24] = (ctx->state[6] >> (24 - i*8)) & 0x000000ff;
        hash[i+28] = (ctx->state[7] >> (24 - i*8)) & 0x000000ff;
    }
}

static void sha256_hash(const uint8_t* input, size_t input_len, uint8_t output[AUTH_HASH_LEN]) {
    SHA256_CTX ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, input, input_len);
    sha256_final(&ctx, output);
}

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
    SHA256_CTX ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, k_ipad, 64);
    sha256_update(&ctx, msg, msg_len);
    sha256_final(&ctx, inner_hash);
    sha256_init(&ctx);
    sha256_update(&ctx, k_opad, 64);
    sha256_update(&ctx, inner_hash, AUTH_HASH_LEN);
    sha256_final(&ctx, output);
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
    sha256_hash(concat, sizeof(concat), out_salted_hash);
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
    sha256_hash(concat, sizeof(concat), computed_hash);
    return memcmp(computed_hash, stored_salted_hash, AUTH_HASH_LEN) == 0 ? 1 : 0;
}

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

static uint64_t get_current_time_ms(void) {
    return (uint64_t)clock() * 1000 / CLOCKS_PER_SEC;
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

    uint8_t master_salt[16] = {0x73,0x65,0x6c,0x66,0x6c,0x6e,0x6e,0x5f,
                               0x61,0x75,0x74,0x68,0x5f,0x76,0x31,0x00};
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
        /* 使用加盐哈希验证（防御数据库泄露攻击） */
        int match = auth->keys[i].is_encrypted 
            ? salted_verify_key(auth->keys[i].salt, auth->keys[i].hash, key_hash)
            : (memcmp(auth->keys[i].hash, key_hash, AUTH_HASH_LEN) == 0);
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
            memset(&auth->keys[i], 0, sizeof(AuthKeyEntry));
            memmove(&auth->keys[i], &auth->keys[i+1], (auth->key_count - i - 1) * sizeof(AuthKeyEntry));
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
    (void)auth;
    return token_bucket_consume(bucket, 1) ? 1 : 0;
}

int auth_global_check_rate(AuthSystem* auth) {
    if (!auth) return 0;
    return token_bucket_consume(&auth->global_bucket, 1) ? 1 : 0;
}

int auth_check_request(AuthSystem* auth, const char* endpoint,
                       const char* auth_header, AuthPermission* out_permission) {
    if (!auth || !endpoint) return 0;
    
    if (!auth->auth_required) {
        if (out_permission) *out_permission = AUTH_PERM_ADMIN;
        return 1;
    }
    
    AuthPermission required = auth_get_required_permission(auth, endpoint);
    
    if (!auth->auth_required && required == AUTH_PERM_NONE) {
        if (out_permission) *out_permission = AUTH_PERM_READONLY;
        return 1;
    }
    
    if (!auth_header) return 0;
    
    if (strncmp(auth_header, "Bearer ", 7) != 0) return 0;
    const char* token = auth_header + 7;
    
    int valid = auth_validate_key(auth, token, required);
    if (valid) {
        if (out_permission) *out_permission = required;
        return 1;
    }
    return 0;
}

static void xor_encrypt_decrypt(uint8_t* data, size_t len, const uint8_t* key, size_t key_len) {
    for (size_t i = 0; i < len; i++)
        data[i] ^= key[i % key_len];
}

int auth_save_keys(AuthSystem* auth, const char* filepath) {
    /* F-037修复: 启用XOR加密保护密钥文件，防止明文泄露 */
    if (!auth || !filepath) return -1;
    FILE* fp = fopen(filepath, "wb");
    if (!fp) return -1;
    
    uint8_t header[16] = "SELFLNN_AUTH_V2"; /* V2表示加密格式 */
    fwrite(header, 1, 16, fp);
    
    int count = auth->key_count;
    fwrite(&count, sizeof(int), 1, fp);
    
    /* F-037: XOR加密密钥数据再写入 */
    size_t keys_size = (size_t)count * sizeof(AuthKeyEntry);
    uint8_t* keys_copy = (uint8_t*)safe_malloc(keys_size);
    if (keys_copy) {
        memcpy(keys_copy, auth->keys, keys_size);
        /* 使用文件路径哈希作为XOR密钥 */
        uint8_t xor_key[32];
        memset(xor_key, 0, 32);
        size_t path_len = strlen(filepath);
        for (size_t i = 0; i < path_len; i++) xor_key[i % 32] ^= (uint8_t)filepath[i];
        for (size_t i = 0; i < keys_size; i++) keys_copy[i] ^= xor_key[i % 32];
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
    if (fread(header, 1, 16, fp) != 16) { fclose(fp); return -1; }
    
    /* F-037: 检测加密格式 */
    if (memcmp(header, "SELFLNN_AUTH_V2", 16) == 0) {
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
    
    size_t keys_size = (size_t)count * sizeof(AuthKeyEntry);
    if (is_encrypted) {
        /* F-037: XOR解密 */
        uint8_t* keys_encrypted = (uint8_t*)safe_malloc(keys_size);
        if (!keys_encrypted || fread(keys_encrypted, 1, keys_size, fp) != keys_size) {
            free(keys_encrypted);
            fclose(fp);
            return -1;
        }
        uint8_t xor_key[32];
        memset(xor_key, 0, 32);
        size_t path_len = strlen(filepath);
        for (size_t i = 0; i < path_len; i++) xor_key[i % 32] ^= (uint8_t)filepath[i];
        for (size_t i = 0; i < keys_size; i++) keys_encrypted[i] ^= xor_key[i % 32];
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
