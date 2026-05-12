/**
 * @file device_protocols.h
 * @brief 设备协议扩展系统接口（Modbus/CAN/OPC-UA/EtherCAT/MQTT）
 */

#ifndef SELFLNN_DEVICE_PROTOCOLS_H
#define SELFLNN_DEVICE_PROTOCOLS_H

#include <stddef.h>
#include <stdint.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    PROTO_MODBUS_RTU = 0,
    PROTO_MODBUS_TCP = 1,
    PROTO_CAN_BUS = 2,
    PROTO_OPC_UA = 3,
    PROTO_ETHERCAT = 4,
    PROTO_MQTT = 5,
    PROTO_SERIAL_RAW = 6,
    PROTO_CUSTOM = 7
} DeviceProtocolType;

typedef enum {
    MODBUS_FC_READ_COILS = 1,
    MODBUS_FC_READ_DISCRETE = 2,
    MODBUS_FC_READ_HOLDING = 3,
    MODBUS_FC_READ_INPUT = 4,
    MODBUS_FC_WRITE_SINGLE_COIL = 5,
    MODBUS_FC_WRITE_SINGLE_REG = 6,
    MODBUS_FC_WRITE_MULTI_COIL = 15,
    MODBUS_FC_WRITE_MULTI_REG = 16
} ModbusFunctionCode;

typedef struct {
    char port[64];
    int baud_rate;
    int data_bits;
    int stop_bits;
    char parity;
    int slave_id;
    int timeout_ms;
    int retries;
} ModbusConfig;

typedef struct {
    char interface[64];
    int bit_rate;
    int is_extended_id;
    int timeout_ms;
} CanConfig;

typedef struct {
    char endpoint_url[256];
    char security_policy[64];
    int session_timeout_ms;
    int use_encryption;
} OpcUaConfig;

typedef struct {
    char master_port[64];
    int cycle_time_us;
    int slave_count;
    int use_dc;
} EthercatConfig;

typedef struct {
    char broker_url[256];
    int port;
    char client_id[64];
    char username[64];
    char password[64];
    int keep_alive_sec;
    int qos;
} MqttConfig;

typedef struct {
    DeviceProtocolType type;
    union {
        ModbusConfig modbus;
        CanConfig can;
        OpcUaConfig opc_ua;
        EthercatConfig ethercat;
        MqttConfig mqtt;
    };
    char device_name[64];
    int is_connected;
    time_t last_data_time;
} DeviceProtocolConfig;

typedef struct {
    uint8_t slave_id;
    uint8_t function_code;
    uint16_t start_address;
    uint16_t quantity;
    uint16_t* data;
    int data_count;
    int error_code;
    char error_msg[128];
} ModbusFrame;

typedef struct {
    uint32_t id;
    uint8_t dlc;
    uint8_t data[8];
    int is_extended;
    int is_remote;
    int is_error;
    time_t timestamp;
} CanFrame;

typedef struct DeviceProtocolManager DeviceProtocolManager;

DeviceProtocolManager* device_protocol_create(void);
void device_protocol_free(DeviceProtocolManager* dpm);

int device_protocol_connect(DeviceProtocolManager* dpm, const DeviceProtocolConfig* config);
int device_protocol_disconnect(DeviceProtocolManager* dpm, const char* device_name);
int device_protocol_is_connected(DeviceProtocolManager* dpm, const char* device_name);

/* Modbus */
int modbus_read_registers(DeviceProtocolManager* dpm, const char* device, uint16_t addr, uint16_t count, uint16_t* out);
int modbus_write_register(DeviceProtocolManager* dpm, const char* device, uint16_t addr, uint16_t value);
int modbus_write_registers(DeviceProtocolManager* dpm, const char* device, uint16_t addr, uint16_t* values, uint16_t count);

/* CAN */
int can_send_frame(DeviceProtocolManager* dpm, const char* device, const CanFrame* frame);
int can_read_frame(DeviceProtocolManager* dpm, const char* device, CanFrame* frame);

/* OPC-UA */
int opcua_read_node(DeviceProtocolManager* dpm, const char* device, const char* node_id, char* value, size_t max_len);
int opcua_write_node(DeviceProtocolManager* dpm, const char* device, const char* node_id, const char* value);

/* MQTT */
int mqtt_publish(DeviceProtocolManager* dpm, const char* device, const char* topic, const char* payload);
int mqtt_subscribe(DeviceProtocolManager* dpm, const char* device, const char* topic);

/* 通用设备扫描 */
int device_protocol_scan(DeviceProtocolManager* dpm, char* devices, int max_count);

#ifdef __cplusplus
}
#endif
#endif
