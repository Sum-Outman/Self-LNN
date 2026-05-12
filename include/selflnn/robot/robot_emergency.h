#ifndef SELFLNN_ROBOT_EMERGENCY_H
#define SELFLNN_ROBOT_EMERGENCY_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ROBOT_EMERGENCY_MAX_CHANNELS 4
#define ROBOT_EMERGENCY_MAX_BRAKES 8
#define ROBOT_EMERGENCY_MAX_SENSORS 16
#define ROBOT_EMERGENCY_DEBOUNCE_MS 20
#define ROBOT_EMERGENCY_WATCHDOG_TIMEOUT_MS 100
#define ROBOT_EMERGENCY_NAME_MAX 64

typedef enum {
    EMERGENCY_LEVEL_NONE = 0,
    EMERGENCY_LEVEL_WARNING = 1,
    EMERGENCY_LEVEL_STOP_SOFT = 2,
    EMERGENCY_LEVEL_STOP_HARD = 3,
    EMERGENCY_LEVEL_STOP_EMERGENCY = 4,
    EMERGENCY_LEVEL_POWER_OFF = 5
} EmergencyLevel;

typedef enum {
    CHANNEL_TYPE_NONE = 0,
    CHANNEL_TYPE_BUTTON = 1,
    CHANNEL_TYPE_SOFTWARE = 2,
    CHANNEL_TYPE_WATCHDOG = 3,
    CHANNEL_TYPE_SENSOR = 4,
    CHANNEL_TYPE_COMMUNICATION = 5,
    CHANNEL_TYPE_POWER_MONITOR = 6,
    CHANNEL_TYPE_TEMPERATURE = 7,
    CHANNEL_TYPE_CURRENT = 8
} EmergencyChannelType;

typedef enum {
    BRAKE_TYPE_NONE = 0,
    BRAKE_TYPE_ELECTROMAGNETIC = 1,
    BRAKE_TYPE_MECHANICAL = 2,
    BRAKE_TYPE_REGENERATIVE = 3,
    BRAKE_TYPE_HYDRAULIC = 4,
    BRAKE_TYPE_PNEUMATIC = 5
} BrakeType;

typedef enum {
    EMERGENCY_MONITOR_OK = 0,
    EMERGENCY_MONITOR_WARNING = 1,
    EMERGENCY_MONITOR_FAULT = 2,
    EMERGENCY_MONITOR_CRITICAL = 3
} EmergencyMonitorStatus;

typedef struct {
    int channel_id;
    EmergencyChannelType type;
    char name[ROBOT_EMERGENCY_NAME_MAX];
    int is_triggered;
    int is_enabled;
    int raw_value;
    int debounced_value;
    uint64_t last_change_time_ms;
    int change_count;
    int stable_count;
    EmergencyLevel trigger_level;
    float threshold_high;
    float threshold_low;
    float current_value;
} EmergencyChannel;

typedef struct {
    int brake_id;
    BrakeType type;
    char name[ROBOT_EMERGENCY_NAME_MAX];
    int is_active;
    int is_engaged;
    int is_faulted;
    float engagement_time_ms;
    float release_time_ms;
    float max_torque_nm;
    float current_torque_nm;
    int power_status;
} EmergencyBrake;

typedef struct {
    int emergency_stop_count;
    uint64_t last_emergency_time_ms;
    uint64_t total_emergency_duration_ms;
    int recovery_count;
    int watchdog_timeout_count;
    int power_failure_count;
    int communication_loss_count;
    int sensor_fault_count;
    int over_current_count;
    int over_temperature_count;
} EmergencyHistory;

typedef struct {
    EmergencyLevel current_level;
    EmergencyLevel highest_level;
    int is_system_ready;
    int is_emergency_active;
    int is_brake_engaged;
    int is_power_cut;
    int is_recovering;
    int auto_recovery_enabled;
    float auto_recovery_delay_s;
    float system_voltage_v;
    float system_current_a;
    float system_temperature_c;
    uint64_t uptime_ms;
    uint64_t recovery_start_time_ms;
    int channel_count;
    int brake_count;
    EmergencyChannel channels[ROBOT_EMERGENCY_MAX_CHANNELS];
    EmergencyBrake brakes[ROBOT_EMERGENCY_MAX_BRAKES];
    EmergencyHistory history;
} EmergencySystem;

typedef struct {
    int enable_hardware_monitoring;
    int enable_watchdog;
    int watchdog_timeout_ms;
    int enable_auto_recovery;
    float auto_recovery_delay_s;
    int enable_power_monitoring;
    float over_voltage_threshold_v;
    float under_voltage_threshold_v;
    float over_current_threshold_a;
    float over_temperature_threshold_c;
    int enable_brake_test;
    int brake_test_interval_s;
    int max_channel_count;
    int max_brake_count;
} EmergencyConfig;

extern const EmergencyConfig EMERGENCY_CONFIG_DEFAULT;

EmergencySystem* robot_emergency_create(const EmergencyConfig* config);
void robot_emergency_free(EmergencySystem* system);
int robot_emergency_init(EmergencySystem* system, const EmergencyConfig* config);

int robot_emergency_add_channel(EmergencySystem* system, EmergencyChannelType type,
                                 const char* name, EmergencyLevel trigger_level,
                                 float threshold_high, float threshold_low);
int robot_emergency_remove_channel(EmergencySystem* system, int channel_id);

int robot_emergency_add_brake(EmergencySystem* system, BrakeType type,
                               const char* name, float max_torque_nm);
int robot_emergency_remove_brake(EmergencySystem* system, int brake_id);

int robot_emergency_trigger(EmergencySystem* system, EmergencyLevel level,
                             int channel_id, const char* reason);
int robot_emergency_release(EmergencySystem* system);

int robot_emergency_update_channel(EmergencySystem* system, int channel_id,
                                    float value, uint64_t timestamp_ms);
int robot_emergency_update_power(EmergencySystem* system, float voltage_v,
                                  float current_a, uint64_t timestamp_ms);
int robot_emergency_update_temperature(EmergencySystem* system, float temperature_c);

int robot_emergency_engage_brakes(EmergencySystem* system);
int robot_emergency_release_brakes(EmergencySystem* system);
int robot_emergency_test_brake(EmergencySystem* system, int brake_id);

int robot_emergency_watchdog_reset(EmergencySystem* system, uint64_t timestamp_ms);
int robot_emergency_watchdog_check(EmergencySystem* system, uint64_t timestamp_ms);

int robot_emergency_cut_power(EmergencySystem* system);
int robot_emergency_restore_power(EmergencySystem* system);

int robot_emergency_get_status(const EmergencySystem* system,
                                EmergencyLevel* level, int* is_active);
int robot_emergency_get_channel_status(const EmergencySystem* system,
                                        int channel_id, EmergencyChannel* channel);
int robot_emergency_get_brake_status(const EmergencySystem* system,
                                      int brake_id, EmergencyBrake* brake);
int robot_emergency_get_history(const EmergencySystem* system,
                                 EmergencyHistory* history);
int robot_emergency_get_summary(const EmergencySystem* system,
                                 char* buffer, size_t buffer_size);

int robot_emergency_self_test(EmergencySystem* system);
int robot_emergency_reset(EmergencySystem* system);
int robot_emergency_set_auto_recovery(EmergencySystem* system, int enable,
                                       float delay_s);
int robot_emergency_is_safe(const EmergencySystem* system);

#ifdef __cplusplus
}
#endif

#endif
