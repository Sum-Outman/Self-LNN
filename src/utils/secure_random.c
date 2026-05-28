#include "selflnn/utils/secure_random.h"
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#include <wincrypt.h>
#pragma comment(lib, "bcrypt.lib")
#include <bcrypt.h>
#else
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/random.h>
#include <fcntl.h>
#endif

int secure_random_bytes(uint8_t* buffer, size_t length)
{
    if (!buffer || length == 0) return -1;

#ifdef _WIN32
    NTSTATUS status = BCryptGenRandom(NULL, buffer, (ULONG)length, BCRYPT_USE_SYSTEM_PREFERRED_RNG);
    return (status >= 0) ? 0 : -1;
#elif defined(__linux__) && defined(SYS_getrandom)
    size_t offset = 0;
    while (offset < length) {
        ssize_t ret = syscall(SYS_getrandom, buffer + offset, length - offset, 0);
        if (ret < 0) {
            if (errno == EINTR) continue;
            break;
        }
        offset += (size_t)ret;
    }
    if (offset == length) return 0;
    {
        FILE* f = fopen("/dev/urandom", "rb");
        if (!f) return -1;
        size_t read_count = fread(buffer, 1, length, f);
        fclose(f);
        return (read_count == length) ? 0 : -1;
    }
#else
    {
        FILE* f = fopen("/dev/urandom", "rb");
        if (!f) return -1;
        size_t read_count = fread(buffer, 1, length, f);
        fclose(f);
        return (read_count == length) ? 0 : -1;
    }
#endif
}

float secure_random_float(void)
{
    uint32_t r = 0;
    secure_random_bytes((uint8_t*)&r, sizeof(r));
    /* ZSFX-DEEP-R12-002: 使用double中间精度防止r=2^32-1时float上溢到1.0 */
    return (float)((double)r / 4294967296.0);
}

uint32_t secure_random_int(uint32_t max)
{
    if (max == 0) return 0;
    uint32_t r = 0;
    secure_random_bytes((uint8_t*)&r, sizeof(r));
    return (uint32_t)((uint64_t)r * (uint64_t)max / 4294967296ULL);
}

void secure_random_init(void)
{
    uint8_t dummy;
    secure_random_bytes(&dummy, 1);
}
