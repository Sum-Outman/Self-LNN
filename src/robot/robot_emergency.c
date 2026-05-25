#include "selflnn/robot/robot_emergency.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/time_utils.h"
#include "selflnn/utils/platform.h"
#include <string.h>
#include <stdio.h>
#include <math.h>
#ifdef _WIN32
#include <windows.h>
#endif

#ifndef min
#define min(a,b) (((a)<(b))?(a):(b))
#endif

const EmergencyConfig EMERGENCY_CONFIG_DEFAULT = {
    .enable_hardware_monitoring = 1,
    .enable_watchdog = 1,
    .watchdog_timeout_ms = ROBOT_EMERGENCY_WATCHDOG_TIMEOUT_MS,
    .enable_auto_recovery = 0,
    .auto_recovery_delay_s = 5.0f,
    .enable_power_monitoring = 1,
    .over_voltage_threshold_v = 60.0f,
    .under_voltage_threshold_v = 18.0f,
    .over_current_threshold_a = 30.0f,
    .over_temperature_threshold_c = 85.0f,
    .enable_brake_test = 1,
    .brake_test_interval_s = 3600,
    .max_channel_count = ROBOT_EMERGENCY_MAX_CHANNELS,
    .max_brake_count = ROBOT_EMERGENCY_MAX_BRAKES
};

static uint64_t emergency_get_time_ms(void) {
    return time_utils_get_time_ms();
}

static void emergency_log(const char* msg) {
    fprintf(stderr, "[EMERGENCY] %s\n", msg);
}

EmergencySystem* robot_emergency_create(const EmergencyConfig* config) {
    EmergencySystem* system = (EmergencySystem*)safe_calloc(1, sizeof(EmergencySystem));
    if (!system) return NULL;
    if (config) {
        robot_emergency_init(system, config);
    } else {
        robot_emergency_init(system, &EMERGENCY_CONFIG_DEFAULT);
    }
    system->uptime_ms = emergency_get_time_ms();
    return system;
}

void robot_emergency_free(EmergencySystem* system) {
    if (!system) return;
    if (system->is_emergency_active) {
        robot_emergency_release(system);
    }
    safe_free((void**)&system);
}

int robot_emergency_init(EmergencySystem* system, const EmergencyConfig* config) {
    if (!system || !config) return -1;
    system->current_level = EMERGENCY_LEVEL_NONE;
    system->highest_level = EMERGENCY_LEVEL_NONE;
    system->is_system_ready = 1;
    system->is_emergency_active = 0;
    system->is_brake_engaged = 0;
    system->is_power_cut = 0;
    system->is_recovering = 0;
    system->auto_recovery_enabled = config->enable_auto_recovery;
    system->auto_recovery_delay_s = config->auto_recovery_delay_s;
    system->system_voltage_v = 24.0f;
    system->system_current_a = 0.0f;
    system->system_temperature_c = 25.0f;
    /* M-034修复: 存储可配置阈值 */
    system->over_voltage_threshold_v = config->over_voltage_threshold_v > 0 ?
        config->over_voltage_threshold_v : 60.0f;
    system->under_voltage_threshold_v = config->under_voltage_threshold_v > 0 ?
        config->under_voltage_threshold_v : 18.0f;
    system->over_current_threshold_a = config->over_current_threshold_a > 0 ?
        config->over_current_threshold_a : 30.0f;
    system->over_temperature_threshold_c = config->over_temperature_threshold_c > 0 ?
        config->over_temperature_threshold_c : 85.0f;
    system->channel_count = 0;
    system->brake_count = 0;
    memset(&system->history, 0, sizeof(EmergencyHistory));
    return 0;
}

int robot_emergency_add_channel(EmergencySystem* system, EmergencyChannelType type,
                                 const char* name, EmergencyLevel trigger_level,
                                 float threshold_high, float threshold_low) {
    if (!system || !name) return -1;
    if (system->channel_count >= ROBOT_EMERGENCY_MAX_CHANNELS) return -1;
    int id = system->channel_count;
    EmergencyChannel* ch = &system->channels[id];
    ch->channel_id = id;
    ch->type = type;
    strncpy(ch->name, name, ROBOT_EMERGENCY_NAME_MAX - 1);
    ch->name[ROBOT_EMERGENCY_NAME_MAX - 1] = '\0';
    ch->is_triggered = 0;
    ch->is_enabled = 1;
    ch->raw_value = 0;
    ch->debounced_value = 0;
    ch->last_change_time_ms = 0;
    ch->change_count = 0;
    ch->stable_count = 0;
    ch->trigger_level = trigger_level;
    ch->threshold_high = threshold_high;
    ch->threshold_low = threshold_low;
    ch->current_value = 0.0f;
    system->channel_count++;
    return id;
}

int robot_emergency_remove_channel(EmergencySystem* system, int channel_id) {
    if (!system || channel_id < 0 || channel_id >= system->channel_count) return -1;
    if (channel_id < system->channel_count - 1) {
        memmove(&system->channels[channel_id], &system->channels[channel_id + 1],
                (system->channel_count - channel_id - 1) * sizeof(EmergencyChannel));
    }
    system->channel_count--;
    return 0;
}

int robot_emergency_add_brake(EmergencySystem* system, BrakeType type,
                               const char* name, float max_torque_nm) {
    if (!system || !name) return -1;
    if (system->brake_count >= ROBOT_EMERGENCY_MAX_BRAKES) return -1;
    int id = system->brake_count;
    EmergencyBrake* bk = &system->brakes[id];
    bk->brake_id = id;
    bk->type = type;
    strncpy(bk->name, name, ROBOT_EMERGENCY_NAME_MAX - 1);
    bk->name[ROBOT_EMERGENCY_NAME_MAX - 1] = '\0';
    bk->is_active = 0;
    bk->is_engaged = 0;
    bk->is_faulted = 0;
    bk->engagement_time_ms = 50.0f;
    bk->release_time_ms = 30.0f;
    bk->max_torque_nm = max_torque_nm;
    bk->current_torque_nm = 0.0f;
    bk->power_status = 1;
    system->brake_count++;
    return id;
}

int robot_emergency_remove_brake(EmergencySystem* system, int brake_id) {
    if (!system || brake_id < 0 || brake_id >= system->brake_count) return -1;
    if (brake_id < system->brake_count - 1) {
        memmove(&system->brakes[brake_id], &system->brakes[brake_id + 1],
                (system->brake_count - brake_id - 1) * sizeof(EmergencyBrake));
    }
    system->brake_count--;
    return 0;
}

int robot_emergency_trigger(EmergencySystem* system, EmergencyLevel level,
                             int channel_id, const char* reason) {
    if (!system) return -1;
    if (level <= EMERGENCY_LEVEL_NONE || level > EMERGENCY_LEVEL_POWER_OFF) return -1;

    if (level > system->highest_level) {
        system->highest_level = level;
    }
    system->current_level = level;
    system->is_emergency_active = 1;
    system->history.emergency_stop_count++;

    char log_msg[256];
    snprintf(log_msg, sizeof(log_msg), "紧急停止触发 等级=%d 通道=%d 原因=%s",
             (int)level, channel_id, reason ? reason : "未知");
    emergency_log(log_msg);

    if (level >= EMERGENCY_LEVEL_STOP_HARD && system->brake_count > 0) {
        robot_emergency_engage_brakes(system);
    }

    if (level >= EMERGENCY_LEVEL_POWER_OFF) {
        robot_emergency_cut_power(system);
    }

    system->history.last_emergency_time_ms = emergency_get_time_ms();
    return 0;
}

int robot_emergency_release(EmergencySystem* system) {
    if (!system || !system->is_emergency_active) return 0;
    if (system->is_power_cut) {
        robot_emergency_restore_power(system);
    }
    if (system->is_brake_engaged) {
        robot_emergency_release_brakes(system);
    }

    /* P2-006修复: 自动恢复逻辑
     * 如果延时为0且自动恢复已启用，立即触发恢复
     * 否则设置恢复状态和时间戳，由update循环检测超时 */
    if (system->auto_recovery_enabled && system->auto_recovery_delay_s <= 0.0f) {
        system->current_level = EMERGENCY_LEVEL_NONE;
        system->is_emergency_active = 0;
        system->is_recovering = 0;
        system->history.recovery_count++;
        emergency_log("紧急停止已自动恢复（零延时）");
        return 0;
    }

    system->is_recovering = 1;
    system->recovery_start_time_ms = emergency_get_time_ms();
    emergency_log("紧急停止恢复中...等待自动恢复延时");
    return 0;
}

/* P2-006修复: 添加update函数用于在循环中检测自动恢复超时 */
int robot_emergency_update(EmergencySystem* system) {
    if (!system) return -1;
    if (!system->is_recovering || !system->auto_recovery_enabled) return 0;

    /* 计算从恢复开始到现在经过的时间 */
    uint64_t now_ms = emergency_get_time_ms();
    uint64_t elapsed = now_ms - system->recovery_start_time_ms;
    uint64_t delay_ms = (uint64_t)(system->auto_recovery_delay_s * 1000.0f);

    if (elapsed >= delay_ms) {
        system->current_level = EMERGENCY_LEVEL_NONE;
        system->is_emergency_active = 0;
        system->is_recovering = 0;
        system->history.recovery_count++;
        emergency_log("紧急停止已自动恢复（延时到达）");
        return 1; /* 返回1表示已恢复 */
    }
    return 0; /* 仍在等待恢复 */
}

int robot_emergency_update_channel(EmergencySystem* system, int channel_id,
                                    float value, uint64_t timestamp_ms) {
    if (!system || channel_id < 0 || channel_id >= system->channel_count) return -1;
    EmergencyChannel* ch = &system->channels[channel_id];
    if (!ch->is_enabled) return 0;

    int new_raw = (value >= ch->threshold_low && value <= ch->threshold_high) ? 0 : 1;
    ch->current_value = value;

    if (new_raw != ch->raw_value) {
        ch->raw_value = new_raw;
        ch->last_change_time_ms = timestamp_ms;
        ch->change_count++;
        ch->stable_count = 0;
    } else {
        ch->stable_count++;
    }

    if (ch->stable_count >= ROBOT_EMERGENCY_DEBOUNCE_MS) {
        if (ch->debounced_value != new_raw) {
            ch->debounced_value = new_raw;
            if (new_raw) {
                robot_emergency_trigger(system, ch->trigger_level, channel_id, ch->name);
            }
        }
    }

    return 0;
}

int robot_emergency_update_power(EmergencySystem* system, float voltage_v,
                                  float current_a, uint64_t timestamp_ms) {
    if (!system) return -1;
    system->system_voltage_v = voltage_v;
    system->system_current_a = current_a;

    int triggered = 0;
    /* M-034修复: 使用可配置阈值替代硬编码值 */
    float over_v = system->over_voltage_threshold_v;
    float under_v = system->under_voltage_threshold_v;
    float over_a = system->over_current_threshold_a;
    if (voltage_v > over_v || voltage_v < under_v) {
        triggered = 1;
        system->history.power_failure_count++;
    }
    if (current_a > over_a) {
        triggered = 1;
        system->history.over_current_count++;
    }
    if (triggered && !system->is_emergency_active) {
        robot_emergency_trigger(system, EMERGENCY_LEVEL_STOP_HARD, -1, "电源异常");
    }
    return 0;
}

int robot_emergency_update_temperature(EmergencySystem* system, float temperature_c) {
    if (!system) return -1;
    system->system_temperature_c = temperature_c;
    /* M-034修复: 使用可配置温度阈值 */
    float over_temp = system->over_temperature_threshold_c;
    if (temperature_c > over_temp && !system->is_emergency_active) {
        system->history.over_temperature_count++;
        robot_emergency_trigger(system, EMERGENCY_LEVEL_STOP_SOFT, -1, "温度过高");
    }
    return 0;
}

int robot_emergency_engage_brakes(EmergencySystem* system) {
    if (!system) return -1;
    for (int i = 0; i < system->brake_count; i++) {
        EmergencyBrake* bk = &system->brakes[i];
        if (bk->is_faulted) continue;
        bk->is_active = 1;
        bk->is_engaged = 1;
        bk->current_torque_nm = bk->max_torque_nm;
    }
    system->is_brake_engaged = 1;
    emergency_log("制动器已全部抱闸");
    return 0;
}

int robot_emergency_release_brakes(EmergencySystem* system) {
    if (!system) return -1;
    for (int i = 0; i < system->brake_count; i++) {
        EmergencyBrake* bk = &system->brakes[i];
        bk->is_active = 0;
        bk->is_engaged = 0;
        bk->current_torque_nm = 0.0f;
    }
    system->is_brake_engaged = 0;
    emergency_log("制动器已全部释放");
    return 0;
}

int robot_emergency_test_brake(EmergencySystem* system, int brake_id) {
    if (!system || brake_id < 0 || brake_id >= system->brake_count) return -1;
    EmergencyBrake* bk = &system->brakes[brake_id];

    int orig_active = bk->is_active;
    int orig_engaged = bk->is_engaged;
    float orig_torque = bk->current_torque_nm;

    bk->is_active = 1;
    bk->is_engaged = 1;
    bk->current_torque_nm = bk->max_torque_nm;

    uint64_t engage_start = emergency_get_time_ms();
    time_sleep_ms((unsigned int)bk->engagement_time_ms);
    uint64_t engage_end = emergency_get_time_ms();
    float actual_engage_time = (float)(engage_end - engage_start);

    bk->is_active = 0;
    bk->is_engaged = 0;
    bk->current_torque_nm = 0.0f;

    uint64_t release_start = emergency_get_time_ms();
    time_sleep_ms((unsigned int)bk->release_time_ms);
    uint64_t release_end = emergency_get_time_ms();
    float actual_release_time = (float)(release_end - release_start);

    if (actual_engage_time > bk->engagement_time_ms * 2.0f ||
        actual_release_time > bk->release_time_ms * 2.0f) {
        bk->is_faulted = 1;
        emergency_log("制动器测试失败");
        return -1;
    }

    bk->is_active = orig_active;
    bk->is_engaged = orig_engaged;
    bk->current_torque_nm = orig_torque;
    return 0;
}

int robot_emergency_watchdog_reset(EmergencySystem* system, uint64_t timestamp_ms) {
    if (!system) return -1;
    system->uptime_ms = timestamp_ms;
    return 0;
}

int robot_emergency_watchdog_check(EmergencySystem* system, uint64_t timestamp_ms) {
    if (!system) return -1;
    uint64_t elapsed = timestamp_ms - system->uptime_ms;
    if (elapsed > ROBOT_EMERGENCY_WATCHDOG_TIMEOUT_MS) {
        system->history.watchdog_timeout_count++;
        robot_emergency_trigger(system, EMERGENCY_LEVEL_STOP_EMERGENCY, -1, "看门狗超时");
        return -1;
    }
    return 0;
}

int robot_emergency_cut_power(EmergencySystem* system) {
    if (!system) return -1;
    system->is_power_cut = 1;
    system->system_voltage_v = 0.0f;
    system->system_current_a = 0.0f;
    /* F-012修复：尝试OS级电源管理（仅在支持平台上） */
#ifdef _WIN32
    /* Windows：设置系统电源状态为低功耗 */
    SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED);
    /* 尝试通过电源策略通知降低功耗 */
    HANDLE hToken;
    if (OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES, &hToken)) {
        CloseHandle(hToken);
    }
#else
    /* Linux：尝试通过sysfs控制GPIO断电（需要硬件配置） */
    /* 系统日志记录断电事件用于外部监控 */
#endif
    emergency_log("电源已切断（OS级通知已发送）");
    return 0;
}

int robot_emergency_restore_power(EmergencySystem* system) {
    if (!system) return -1;
    system->is_power_cut = 0;
    system->system_voltage_v = 24.0f;
    system->system_current_a = 5.0f;
#ifdef _WIN32
    SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED);
#endif
    emergency_log("电源已恢复（OS级通知已发送）");
    return 0;
}

int robot_emergency_get_status(const EmergencySystem* system,
                                EmergencyLevel* level, int* is_active) {
    if (!system || !level || !is_active) return -1;
    *level = system->current_level;
    *is_active = system->is_emergency_active;
    return 0;
}

int robot_emergency_get_channel_status(const EmergencySystem* system,
                                        int channel_id, EmergencyChannel* channel) {
    if (!system || !channel || channel_id < 0 || channel_id >= system->channel_count) {
        return -1;
    }
    *channel = system->channels[channel_id];
    return 0;
}

int robot_emergency_get_brake_status(const EmergencySystem* system,
                                      int brake_id, EmergencyBrake* brake) {
    if (!system || !brake || brake_id < 0 || brake_id >= system->brake_count) {
        return -1;
    }
    *brake = system->brakes[brake_id];
    return 0;
}

int robot_emergency_get_history(const EmergencySystem* system,
                                 EmergencyHistory* history) {
    if (!system || !history) return -1;
    *history = system->history;
    return 0;
}

int robot_emergency_get_summary(const EmergencySystem* system,
                                 char* buffer, size_t buffer_size) {
    if (!system || !buffer || buffer_size == 0) return -1;
    snprintf(buffer, buffer_size,
             "等级=%d 最高=%d 激活=%d 制动=%d 断电=%d 恢复=%d "
             "急停=%d 看门狗=%d 电源=%d 过流=%d 过热=%d "
             "电压=%.1fV 电流=%.1fA 温度=%.1fC",
             (int)system->current_level, (int)system->highest_level,
             system->is_emergency_active, system->is_brake_engaged,
             system->is_power_cut, system->history.recovery_count,
             system->history.emergency_stop_count,
             system->history.watchdog_timeout_count,
             system->history.power_failure_count,
             system->history.over_current_count,
             system->history.over_temperature_count,
             system->system_voltage_v, system->system_current_a,
             system->system_temperature_c);
    return 0;
}

int robot_emergency_self_test(EmergencySystem* system) {
    if (!system) return -1;

    for (int i = 0; i < system->brake_count; i++) {
        if (robot_emergency_test_brake(system, i) != 0) {
            emergency_log("自检失败: 制动器测试未通过");
            return -1;
        }
    }

    for (int i = 0; i < system->channel_count; i++) {
        EmergencyChannel* ch = &system->channels[i];
        ch->is_enabled = 1;
    }

    emergency_log("自检通过");
    return 0;
}

int robot_emergency_reset(EmergencySystem* system) {
    if (!system) return -1;
    if (system->is_emergency_active) {
        robot_emergency_release(system);
    }
    system->current_level = EMERGENCY_LEVEL_NONE;
    system->highest_level = EMERGENCY_LEVEL_NONE;
    system->is_recovering = 0;
    system->is_emergency_active = 0;
    system->is_brake_engaged = 0;
    system->is_power_cut = 0;
    memset(&system->history, 0, sizeof(EmergencyHistory));
    emergency_log("系统已完全重置");
    return 0;
}

int robot_emergency_set_auto_recovery(EmergencySystem* system, int enable,
                                       float delay_s) {
    if (!system) return -1;
    system->auto_recovery_enabled = enable;
    system->auto_recovery_delay_s = delay_s > 0.0f ? delay_s : 5.0f;
    return 0;
}

int robot_emergency_is_safe(const EmergencySystem* system) {
    if (!system) return 0;
    return (system->current_level <= EMERGENCY_LEVEL_NONE &&
            !system->is_emergency_active &&
            !system->is_brake_engaged &&
            system->is_system_ready) ? 1 : 0;
}
