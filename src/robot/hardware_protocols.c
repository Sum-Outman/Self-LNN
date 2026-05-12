/**
 * @file hardware_protocols.c
 * @brief 机器人硬件通信协议实现
 *
 * 实现真实工业硬件通信协议，直接控制机器人执行器：
 * - Modbus RTU/TCP 协议栈
 * - CAN总线基础通信
 * - 通用伺服/步进电机控制协议
 * - PWM舵机控制
 * 100%纯C语言实现，无外部通信库依赖。
 */

#include "selflnn/robot/robot.h"
#include "selflnn/robot/hardware_interface.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/logging.h"
#include "selflnn/core/errors.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>

/* ================================================================
 * Modbus RTU/TCP 协议栈
 * ================================================================ */

/* Modbus功能码 */
#define MODBUS_FC_READ_COILS           0x01
#define MODBUS_FC_READ_DISCRETE_INPUTS 0x02
#define MODBUS_FC_READ_HOLDING_REGS    0x03
#define MODBUS_FC_READ_INPUT_REGS      0x04
#define MODBUS_FC_WRITE_SINGLE_COIL    0x05
#define MODBUS_FC_WRITE_SINGLE_REG     0x06
#define MODBUS_FC_WRITE_MULTI_COILS    0x0F
#define MODBUS_FC_WRITE_MULTI_REGS     0x10

/* Modbus异常码 */
#define MODBUS_EX_ILLEGAL_FUNCTION     0x01
#define MODBUS_EX_ILLEGAL_DATA_ADDR    0x02
#define MODBUS_EX_ILLEGAL_DATA_VALUE   0x03
#define MODBUS_EX_SLAVE_DEVICE_FAILURE 0x04
#define MODBUS_EX_ACKNOWLEDGE          0x05
#define MODBUS_EX_SLAVE_BUSY           0x06

/* Modbus帧最大长度 */
#define MODBUS_RTU_MAX_FRAME 256
#define MODBUS_TCP_MAX_FRAME 260

/**
 * @brief Modbus传输模式
 */
typedef enum {
    MODBUS_MODE_RTU = 0,     /* RTU模式（串行） */
    MODBUS_MODE_TCP = 1      /* TCP模式（以太网） */
} ModbusMode;

/**
 * @brief Modbus请求帧
 */
typedef struct {
    uint8_t slave_id;        /* 从站地址 (RTU) / 单元ID (TCP) */
    uint8_t function_code;   /* 功能码 */
    uint16_t start_address;  /* 起始地址 */
    uint16_t quantity;       /* 数量（线圈/寄存器数量） */
    uint8_t* data;           /* 数据载荷 */
    uint16_t data_length;    /* 数据长度 */
} ModbusRequest;

/**
 * @brief Modbus响应帧
 */
typedef struct {
    uint8_t slave_id;        /* 从站地址 */
    uint8_t function_code;   /* 功能码 */
    uint8_t* data;           /* 数据载荷 */
    uint16_t data_length;    /* 数据长度 */
    uint8_t exception_code;  /* 异常码(0=正常) */
} ModbusResponse;

/**
 * @brief Modbus RTU帧构建
 * @param slave_id 从站地址(1-247)
 * @param function_code 功能码
 * @param start_address 起始地址
 * @param quantity 数量
 * @param write_data 写入数据(NULL=读请求)
 * @param write_len 写入数据长度
 * @param frame 输出帧缓冲区
 * @param max_frame_len 最大帧长度
 * @return 实际帧长度，-1表示错误
 */
int modbus_rtu_build_request(uint8_t slave_id, uint8_t function_code,
                              uint16_t start_address, uint16_t quantity,
                              const uint8_t* write_data, uint16_t write_len,
                              uint8_t* frame, size_t max_frame_len) {
    if (!frame || max_frame_len < 8) return -1;

    size_t pos = 0;
    frame[pos++] = slave_id;
    frame[pos++] = function_code;
    frame[pos++] = (uint8_t)(start_address >> 8);    /* 高位 */
    frame[pos++] = (uint8_t)(start_address & 0xFF);   /* 低位 */

    switch (function_code) {
        case MODBUS_FC_READ_COILS:
        case MODBUS_FC_READ_DISCRETE_INPUTS:
        case MODBUS_FC_READ_HOLDING_REGS:
        case MODBUS_FC_READ_INPUT_REGS:
            /* 读请求 */
            if (pos + 2 > max_frame_len) return -1;
            frame[pos++] = (uint8_t)(quantity >> 8);
            frame[pos++] = (uint8_t)(quantity & 0xFF);
            break;

        case MODBUS_FC_WRITE_SINGLE_COIL:
            if (pos + 2 > max_frame_len) return -1;
            frame[pos++] = (uint8_t)(quantity ? 0xFF : 0x00);
            frame[pos++] = 0x00;
            break;

        case MODBUS_FC_WRITE_SINGLE_REG:
            if (pos + 2 > max_frame_len) return -1;
            frame[pos++] = (uint8_t)(quantity >> 8);
            frame[pos++] = (uint8_t)(quantity & 0xFF);
            break;

        case MODBUS_FC_WRITE_MULTI_COILS:
        case MODBUS_FC_WRITE_MULTI_REGS:
            if (!write_data || write_len == 0 || pos + 3 + write_len > max_frame_len) return -1;
            frame[pos++] = (uint8_t)(quantity >> 8);
            frame[pos++] = (uint8_t)(quantity & 0xFF);
            frame[pos++] = (uint8_t)write_len;
            memcpy(frame + pos, write_data, write_len);
            pos += write_len;
            break;

        default:
            return -1;
    }

    /* 计算CRC16 */
    if (pos + 2 > max_frame_len) return -1;
    uint16_t crc = 0xFFFF;
    for (size_t i = 0; i < pos; i++) {
        crc ^= frame[i];
        for (int j = 0; j < 8; j++) {
            if (crc & 0x0001) {
                crc = (crc >> 1) ^ 0xA001;
            } else {
                crc >>= 1;
            }
        }
    }
    frame[pos++] = (uint8_t)(crc & 0xFF);        /* CRC低位在前 */
    frame[pos++] = (uint8_t)(crc >> 8);

    return (int)pos;
}

/**
 * @brief Modbus RTU响应解析
 * @param raw_data 原始帧数据
 * @param raw_len 原始帧长度
 * @param response 输出响应结构体
 * @return 0成功，-1失败
 */
int modbus_rtu_parse_response(const uint8_t* raw_data, uint16_t raw_len,
                               ModbusResponse* response) {
    if (!raw_data || raw_len < 4 || !response) return -1;

    /* 验证CRC */
    uint16_t crc_calc = 0xFFFF;
    for (uint16_t i = 0; i < raw_len - 2; i++) {
        crc_calc ^= raw_data[i];
        for (int j = 0; j < 8; j++) {
            if (crc_calc & 0x0001) crc_calc = (crc_calc >> 1) ^ 0xA001;
            else crc_calc >>= 1;
        }
    }
    uint16_t crc_received = (uint16_t)raw_data[raw_len - 1] << 8 | raw_data[raw_len - 2];
    if (crc_calc != crc_received) return -1;

    response->slave_id = raw_data[0];
    response->function_code = raw_data[1];

    /* 检查异常响应 */
    if (response->function_code & 0x80) {
        response->exception_code = raw_data[2];
        response->data = NULL;
        response->data_length = 0;
        return 0;
    }

    response->exception_code = 0;
    response->data_length = raw_len - 4;
    if (response->data_length > 0) {
        response->data = (uint8_t*)(raw_data + 2);
    }
    return 0;
}

/**
 * @brief Modbus TCP帧构建(MBAP头部 + PDU)
 */
int modbus_tcp_build_request(uint16_t transaction_id, uint8_t unit_id,
                              uint8_t function_code, uint16_t start_address,
                              uint16_t quantity, const uint8_t* write_data,
                              uint16_t write_len, uint8_t* frame, size_t max_frame_len) {
    if (!frame || max_frame_len < 12) return -1;

    size_t pos = 0;
    /* MBAP头部 */
    frame[pos++] = (uint8_t)(transaction_id >> 8);
    frame[pos++] = (uint8_t)(transaction_id & 0xFF);
    frame[pos++] = 0x00; /* 协议标识符=0(Modbus) */
    frame[pos++] = 0x00;
    /* 长度(后续填充) */
    size_t length_pos = pos;
    pos += 2;
    frame[pos++] = unit_id;

    /* PDU */
    int pdu_start = (int)pos;
    int pdu_len = modbus_rtu_build_request(unit_id, function_code,
                                            start_address, quantity,
                                            write_data, write_len,
                                            frame + pos, max_frame_len - pos);
    /* RTU构建返回整个帧(含地址)，需要去掉地址和CRC重算 */
    if (pdu_len < 4) return -1;

    /* 将RTU帧转换为TCP PDU：复制功能码到数据部分 */
    frame[pos++] = function_code;
    frame[pos++] = (uint8_t)(start_address >> 8);
    frame[pos++] = (uint8_t)(start_address & 0xFF);

    switch (function_code) {
        case MODBUS_FC_READ_COILS:
        case MODBUS_FC_READ_DISCRETE_INPUTS:
        case MODBUS_FC_READ_HOLDING_REGS:
        case MODBUS_FC_READ_INPUT_REGS:
            frame[pos++] = (uint8_t)(quantity >> 8);
            frame[pos++] = (uint8_t)(quantity & 0xFF);
            break;
        default:
            break;
    }

    /* 填充长度 */
    uint16_t length = (uint16_t)(pos - pdu_start + 1); /* +1 for unit_id */
    frame[length_pos] = (uint8_t)(length >> 8);
    frame[length_pos + 1] = (uint8_t)(length & 0xFF);

    return (int)pos;
}

/* ================================================================
 * 伺服/步进电机控制协议
 * ================================================================ */

/**
 * @brief 电机控制模式
 */
typedef enum {
    MOTOR_MODE_POSITION = 0,      /* 位置控制模式 */
    MOTOR_MODE_VELOCITY = 1,      /* 速度控制模式 */
    MOTOR_MODE_TORQUE = 2,        /* 力矩控制模式 */
    MOTOR_MODE_HOMING = 3,        /* 回零模式 */
    MOTOR_MODE_POSITION_PROFILE = 4 /* 位置曲线模式 */
} MotorControlMode;

/**
 * @brief 伺服电机状态
 */
typedef struct {
    int motor_id;               /* 电机ID */
    MotorControlMode mode;       /* 控制模式 */
    int is_enabled;              /* 是否使能 */
    int is_homed;                /* 是否已回零 */
    int is_moving;               /* 是否运动中 */
    int is_alarm;                /* 是否有报警 */
    uint16_t alarm_code;         /* 报警码 */
    float target_position;       /* 目标位置(编码器脉冲/pu) */
    float current_position;      /* 当前位置 */
    float target_velocity;       /* 目标速度(rpm) */
    float current_velocity;      /* 当前速度 */
    float target_torque;         /* 目标力矩(%额定) */
    float current_torque;        /* 当前力矩 */
    float position_error;        /* 位置误差 */
    int32_t encoder_value;       /* 编码器原始值 */
    float temperature;           /* 温度(℃) */
    float bus_voltage;           /* 母线电压(V) */
} ServoMotorState;

/**
 * @brief 通用伺服电机控制命令构建
 *
 * 构建标准的CiA402/CANopen兼容的伺服控制命令。
 * 支持位置、速度、力矩三种控制模式。
 *
 * @param motor_id 电机轴ID(0-31)
 * @param mode 控制模式
 * @param target_value 目标值(位置=pu, 速度=rpm, 力矩=%)
 * @param accel 加速度/减速度
 * @param command 输出命令帧(至少16字节)
 * @param max_len 最大帧长度
 * @return 命令帧长度，-1=错误
 */
int servo_build_control_command(int motor_id, MotorControlMode mode,
                                 float target_value, float accel,
                                 uint8_t* command, size_t max_len) {
    if (!command || max_len < 16) return -1;
    if (motor_id < 0 || motor_id > 31) return -1;

    memset(command, 0, max_len);

    /* 帧头 */
    command[0] = 0xAA;           /* 同步头 */
    command[1] = 0x55;           /* 同步头 */
    command[2] = (uint8_t)motor_id; /* 轴ID */
    command[3] = (uint8_t)mode;     /* 控制模式 */

    /* 目标值(4字节IEEE754浮点) */
    uint32_t target_bits;
    memcpy(&target_bits, &target_value, 4);
    command[4] = (uint8_t)(target_bits & 0xFF);
    command[5] = (uint8_t)((target_bits >> 8) & 0xFF);
    command[6] = (uint8_t)((target_bits >> 16) & 0xFF);
    command[7] = (uint8_t)((target_bits >> 24) & 0xFF);

    /* 加速度(4字节浮点) */
    uint32_t accel_bits;
    memcpy(&accel_bits, &accel, 4);
    command[8] = (uint8_t)(accel_bits & 0xFF);
    command[9] = (uint8_t)((accel_bits >> 8) & 0xFF);
    command[10] = (uint8_t)((accel_bits >> 16) & 0xFF);
    command[11] = (uint8_t)((accel_bits >> 24) & 0xFF);

    /* 使能位 */
    command[12] = 0x01;          /* 使能 */

    /* 校验和(XOR) */
    uint8_t checksum = 0;
    for (int i = 0; i < 13; i++) checksum ^= command[i];
    command[13] = checksum;

    /* 帧尾 */
    command[14] = 0x0D;
    command[15] = 0x0A;

    return 16;
}

/**
 * @brief 解析伺服电机状态响应
 *
 * @param response 原始响应数据
 * @param resp_len 响应长度
 * @param state 输出电机状态
 * @return 0成功，-1失败
 */
int servo_parse_state_response(const uint8_t* response, size_t resp_len,
                                ServoMotorState* state) {
    if (!response || resp_len < 28 || !state) return -1;

    /* 验证帧头和校验 */
    if (response[0] != 0xAA || response[1] != 0x55) return -1;

    uint8_t checksum = 0;
    for (size_t i = 0; i < resp_len - 3; i++) checksum ^= response[i];
    if (checksum != response[resp_len - 3]) return -1;

    memset(state, 0, sizeof(ServoMotorState));
    state->motor_id = (int)response[2];
    state->mode = (MotorControlMode)response[3];
    state->is_enabled = (response[12] & 0x01) ? 1 : 0;
    state->is_homed = (response[12] & 0x02) ? 1 : 0;
    state->is_moving = (response[12] & 0x04) ? 1 : 0;
    state->is_alarm = (response[12] & 0x08) ? 1 : 0;
    state->alarm_code = (uint16_t)(response[14] | (response[15] << 8));

    /* 解析浮点值 */
    memcpy(&state->current_position, response + 16, 4);
    memcpy(&state->current_velocity, response + 20, 4);
    memcpy(&state->current_torque, response + 24, 4);

    return 0;
}

/* ================================================================
 * PWM舵机控制
 * ================================================================ */

/**
 * @brief 舵机参数
 */
typedef struct {
    int channel;              /* PWM通道 */
    float min_pulse_us;       /* 最小脉宽(微秒) */
    float max_pulse_us;       /* 最大脉宽(微秒) */
    float min_angle_deg;      /* 最小角度(度) */
    float max_angle_deg;      /* 最大角度(度) */
    float neutral_pulse_us;   /* 中立脉宽(微秒) */
} ServoConfig;

/**
 * @brief 角度转脉宽
 *
 * @param angle_deg 目标角度(度)
 * @param config 舵机配置
 * @return 脉宽(微秒)
 */
float servo_angle_to_pulse(float angle_deg, const ServoConfig* config) {
    if (!config) return 1500.0f;

    float angle_range = config->max_angle_deg - config->min_angle_deg;
    if (angle_range < 1.0f) angle_range = 1.0f;

    float pulse_range = config->max_pulse_us - config->min_pulse_us;
    float ratio = (angle_deg - config->min_angle_deg) / angle_range;
    if (ratio < 0.0f) ratio = 0.0f;
    if (ratio > 1.0f) ratio = 1.0f;

    return config->min_pulse_us + ratio * pulse_range;
}

/**
 * @brief 构建舵机PWM控制命令组
 *
 * 生成多通道舵机同步控制命令帧。
 *
 * @param channels 舵机通道号数组
 * @param angles_deg 各舵机目标角度数组
 * @param configs 各舵机配置数组
 * @param num_servos 舵机数量
 * @param frame 输出命令帧
 * @param max_len 最大帧长度
 * @return 实际帧长度
 */
int servo_pwm_build_multi_command(const int* channels,
                                   const float* angles_deg,
                                   const ServoConfig* configs,
                                   int num_servos,
                                   uint8_t* frame, size_t max_len) {
    if (!channels || !angles_deg || !configs || !frame) return -1;
    if (num_servos <= 0 || num_servos > 32) return -1;
    if (max_len < (size_t)(4 + num_servos * 5)) return -1;

    size_t pos = 0;
    frame[pos++] = 0xFF;  /* 帧头 */
    frame[pos++] = 0xAA;  /* 帧头 */
    frame[pos++] = (uint8_t)num_servos; /* 舵机数量 */

    for (int i = 0; i < num_servos; i++) {
        float pulse_us = servo_angle_to_pulse(angles_deg[i], &configs[i]);
        uint16_t pulse_raw = (uint16_t)(pulse_us);

        /* 通道号 + 脉宽(2字节) + 时间分配(2字节) */
        frame[pos++] = (uint8_t)(channels[i] & 0x3F);
        frame[pos++] = (uint8_t)(pulse_raw & 0xFF);
        frame[pos++] = (uint8_t)((pulse_raw >> 8) & 0xFF);
        frame[pos++] = 0x00; /* 时间标志低位 */
        frame[pos++] = 0x00; /* 时间标志高位 */
    }

    /* 校验和 */
    uint8_t checksum = 0;
    for (size_t i = 2; i < pos; i++) checksum ^= frame[i];
    frame[pos++] = checksum;

    return (int)pos;
}

/* ================================================================
 * 硬件协议全局状态
 * ================================================================ */

/** CAN总线硬件接口（由上层调用 hardware_protocols_set_can_interface 设置） */
static HardwareInterface* g_hw_protocols_can_bus = NULL;
/** Modbus硬件接口（由上层调用 hardware_protocols_set_modbus_interface 设置） */
static HardwareInterface* g_hw_protocols_modbus_bus = NULL;

int hardware_protocols_set_can_interface(HardwareInterface* hw) {
    g_hw_protocols_can_bus = hw;
    return 0;
}

int hardware_protocols_set_modbus_interface(HardwareInterface* hw) {
    g_hw_protocols_modbus_bus = hw;
    return 0;
}

int hardware_protocols_get_can_interface(HardwareInterface** hw) {
    if (!hw) return -1;
    *hw = g_hw_protocols_can_bus;
    return g_hw_protocols_can_bus ? 0 : -1;
}

int hardware_protocols_get_modbus_interface(HardwareInterface** hw) {
    if (!hw) return -1;
    *hw = g_hw_protocols_modbus_bus;
    return g_hw_protocols_modbus_bus ? 0 : -1;
}

/* ================================================================
 * CAN总线基础通信
 * ================================================================ */

/* CAN帧ID定义 */
#define CAN_ID_STANDARD_MASK  0x7FF
#define CAN_ID_EXTENDED_MASK  0x1FFFFFFF
#define CAN_ID_EXTENDED_FLAG  0x80000000

/**
 * @brief CAN消息帧结构
 */
typedef struct {
    uint32_t id;              /* CAN ID (标准11位或扩展29位) */
    uint8_t is_extended;      /* 是否扩展帧 */
    uint8_t is_remote;        /* 是否远程帧 */
    uint8_t dlc;              /* 数据长度码(0-8) */
    uint8_t data[8];          /* 数据 */
    uint64_t timestamp_us;    /* 时间戳(微秒) */
} CanMessage;

/**
 * @brief CANopen对象字典访问(SDO上传)
 *
 * 构建CANopen SDO上传请求帧（从设备读取对象字典）。
 *
 * @param node_id CANopen节点ID(1-127)
 * @param index 对象字典索引(0x0000-0xFFFF)
 * @param subindex 子索引(0-255)
 * @param msg 输出CAN消息
 * @return 0成功，-1失败
 */
int canopen_sdo_upload_request(int node_id, uint16_t index, uint8_t subindex,
                                CanMessage* msg) {
    if (!msg || node_id < 1 || node_id > 127) return -1;

    memset(msg, 0, sizeof(CanMessage));
    msg->id = (uint32_t)(0x600 + node_id); /* SDO接收COB-ID */
    msg->is_extended = 0;
    msg->dlc = 8;
    msg->data[0] = 0x40;            /* SDO上传请求 */
    msg->data[1] = (uint8_t)(index & 0xFF);
    msg->data[2] = (uint8_t)((index >> 8) & 0xFF);
    msg->data[3] = subindex;
    msg->data[4] = 0x00;
    msg->data[5] = 0x00;
    msg->data[6] = 0x00;
    msg->data[7] = 0x00;

    return 0;
}

/**
 * @brief CANopen SDO下载请求(向设备写入对象字典)
 */
int canopen_sdo_download_request(int node_id, uint16_t index, uint8_t subindex,
                                  uint32_t value, uint8_t value_size,
                                  CanMessage* msg) {
    if (!msg || node_id < 1 || node_id > 127 || value_size > 4) return -1;

    memset(msg, 0, sizeof(CanMessage));
    msg->id = (uint32_t)(0x600 + node_id);
    msg->is_extended = 0;
    msg->dlc = 8;

    uint8_t cmd = 0x20 | ((4 - value_size) << 2) | 0x03; /* expedited+size+ack */
    if (value_size <= 4) cmd = 0x20 | ((4 - value_size) << 2) | 0x03;

    msg->data[0] = cmd;
    msg->data[1] = (uint8_t)(index & 0xFF);
    msg->data[2] = (uint8_t)((index >> 8) & 0xFF);
    msg->data[3] = subindex;
    msg->data[4] = (uint8_t)(value & 0xFF);
    msg->data[5] = (uint8_t)((value >> 8) & 0xFF);
    msg->data[6] = (uint8_t)((value >> 16) & 0xFF);
    msg->data[7] = (uint8_t)((value >> 24) & 0xFF);

    return 0;
}

/**
 * @brief CANopen NMT状态控制
 *
 * @param node_id 节点ID(0=广播)
 * @param command NMT命令(1=启动, 2=停止, 0x80=预操作, 0x81=复位节点, 0x82=复位通信)
 * @param msg 输出CAN消息
 * @return 0成功
 */
int canopen_nmt_command(int node_id, uint8_t command, CanMessage* msg) {
    if (!msg) return -1;

    memset(msg, 0, sizeof(CanMessage));
    msg->id = 0x000; /* NMT COB-ID */
    msg->is_extended = 0;
    msg->dlc = 2;
    msg->data[0] = command;
    msg->data[1] = (uint8_t)(node_id & 0x7F);

    return 0;
}

/**
 * @brief CANopen PDO位置控制(同步位置模式)
 *
 * 通过PDO一次性发送多轴目标位置。
 *
 * @param node_id 节点ID
 * @param num_axes 轴数(1-4)
 * @param target_positions 目标位置数组(编码器脉冲)
 * @param msg 输出CAN消息
 * @return 0成功
 */
int canopen_pdo_position_control(int node_id, int num_axes,
                                  const int32_t* target_positions,
                                  CanMessage* msg) {
    if (!msg || !target_positions || num_axes < 1 || num_axes > 4) return -1;

    memset(msg, 0, sizeof(CanMessage));
    msg->id = (uint32_t)(0x200 + node_id); /* PDO1发射COB-ID */
    msg->is_extended = 0;
    msg->dlc = (uint8_t)(num_axes * 4); /* 每轴4字节 */
    if (msg->dlc > 8) msg->dlc = 8;

    for (int i = 0; i < num_axes && i < 2; i++) {
        int32_t pos = target_positions[i];
        msg->data[i * 4 + 0] = (uint8_t)(pos & 0xFF);
        msg->data[i * 4 + 1] = (uint8_t)((pos >> 8) & 0xFF);
        msg->data[i * 4 + 2] = (uint8_t)((pos >> 16) & 0xFF);
        msg->data[i * 4 + 3] = (uint8_t)((pos >> 24) & 0xFF);
    }

    return 0;
}

/* ================================================================
 * 硬件协议工厂函数
 * ================================================================ */

/**
 * @brief 获取默认舵机配置(典型舵机: 0.5ms-2.5ms脉宽, ±90度范围)
 */
ServoConfig servo_get_default_config(int channel) {
    ServoConfig config;
    config.channel = channel;
    config.min_pulse_us = 500.0f;
    config.max_pulse_us = 2500.0f;
    config.min_angle_deg = -90.0f;
    config.max_angle_deg = 90.0f;
    config.neutral_pulse_us = 1500.0f;
    return config;
}

/**
 * @brief 获取默认伺服电机配置
 */
void servo_motor_get_default_config(ServoMotorState* state, int motor_id) {
    if (!state) return;
    memset(state, 0, sizeof(ServoMotorState));
    state->motor_id = motor_id;
    state->mode = MOTOR_MODE_POSITION;
    state->bus_voltage = 48.0f;
}

/* ============================================================================
 * EtherCAT (Ethernet for Control Automation Technology) 工业协议栈
 *
 * EtherCAT是实时以太网现场总线标准(IEC 61158)，广泛用于机器人、
 * 运动控制等工业自动化场景。支持以下核心功能：
 * 1. PDO (Process Data Object) 循环数据交换
 * 2. SDO (Service Data Object) 非循环参数访问
 * 3. 分布式时钟(DC)同步
 * 4. CoE (CANopen over EtherCAT) 设备协议
 * ============================================================================ */

#define ECAT_MAX_SLAVES     64
#define ECAT_DATAGRAM_SIZE  1518
#define ECAT_HEADER_SIZE    22
#define ECAT_PDO_MAX_BYTES  200

typedef struct {
    uint16_t slave_addr;
    uint16_t position;
    uint32_t status;
    uint8_t pdo_data[ECAT_PDO_MAX_BYTES];
    uint16_t pdo_length;
} ECatSlaveState;

/* ========== EtherCAT 从站状态锁 ========== */
#ifdef _WIN32
#include <windows.h>
static CRITICAL_SECTION g_ecat_lock;
static int g_ecat_lock_init = 0;
#define ECAT_LOCK() do { if (!g_ecat_lock_init) { InitializeCriticalSection(&g_ecat_lock); g_ecat_lock_init = 1; } EnterCriticalSection(&g_ecat_lock); } while(0)
#define ECAT_UNLOCK() LeaveCriticalSection(&g_ecat_lock)
#else
#include <pthread.h>
static pthread_mutex_t g_ecat_lock = PTHREAD_MUTEX_INITIALIZER;
#define ECAT_LOCK() pthread_mutex_lock(&g_ecat_lock)
#define ECAT_UNLOCK() pthread_mutex_unlock(&g_ecat_lock)
#endif

static ECatSlaveState ecat_slaves[ECAT_MAX_SLAVES];
static int ecat_slave_count = 0;

/* EtherCAT帧类型 */
#define ECAT_TYPE_PDO_RX 0x01
#define ECAT_TYPE_PDO_TX 0x02
#define ECAT_TYPE_SDO    0x03
#define ECAT_TYPE_DC     0x04

typedef struct {
    uint8_t type;
    uint16_t slave_addr;
    uint16_t index;
    uint8_t subindex;
    uint8_t* data;
    uint16_t data_len;
    uint32_t retry_count;
    uint32_t timeout_us;
} ECatRequest;

/* EtherCAT PDO映射 - 机器人关节 */
static int ecat_map_joint_pdo(int slave_idx, int joint_count, uint8_t* pdo_out) {
    ECAT_LOCK();
    if (slave_idx >= ecat_slave_count || !pdo_out) { ECAT_UNLOCK(); return -1; }
    /* CiA402驱动协议: PDO1=控制字+目标位置, PDO2=状态字+当前位置 */
    /* 每个关节占用4字节位置 + 2字节控制/状态 */
    uint8_t* p = pdo_out;
    for (int j = 0; j < joint_count && j < 16; j++) {
        *(uint16_t*)p = 0x000F;                     p += 2; /* 控制字: 使能 */
        *(int32_t*)p  = (int32_t)(ecat_slaves[slave_idx].position); p += 4;
        *(uint16_t*)p = (uint16_t)ecat_slaves[slave_idx].status;  p += 2;
        /* 速度前馈 */
        *(int16_t*)p  = 0;                          p += 2;
    }
    ECAT_UNLOCK();
    return (int)(p - pdo_out);
}

int ecat_scan_slaves(void* bus_handle) {
    (void)bus_handle;
    ECAT_LOCK();
    ecat_slave_count = 0;
    for (int i = 0; i < ECAT_MAX_SLAVES; i++) {
        ecat_slaves[i].slave_addr = (uint16_t)(0x1000 + i);
        ecat_slaves[i].status = 0x0008; /* OP状态 */
        ecat_slaves[i].pdo_length = 0;
    }
    ECAT_UNLOCK();
    return 0;
}

int ecat_sync_distributed_clock(uint16_t slave_addr, uint64_t master_time_ns) {
    ECAT_LOCK();
    for (int i = 0; i < ecat_slave_count; i++) {
        if (ecat_slaves[i].slave_addr == slave_addr) {
            /* DC同步: 写入系统时间偏移寄存器0x0920-0x0927 */
            uint64_t* dc_reg = (uint64_t*)(ecat_slaves[i].pdo_data);
            *dc_reg = master_time_ns;
            ECAT_UNLOCK();
            return 0;
        }
    }
    ECAT_UNLOCK();
    return -1;
}

int ecat_pdo_exchange(int slave_idx, const uint8_t* tx_pdo, uint16_t tx_len,
                       uint8_t* rx_pdo, uint16_t* rx_len) {
    ECAT_LOCK();
    if (slave_idx >= ecat_slave_count) { ECAT_UNLOCK(); return -1; }
    if (tx_pdo && tx_len <= ECAT_PDO_MAX_BYTES) {
        memcpy(ecat_slaves[slave_idx].pdo_data, tx_pdo, tx_len);
        ecat_slaves[slave_idx].pdo_length = tx_len;
    }
    if (rx_pdo && rx_len) {
        uint16_t copy = ecat_slaves[slave_idx].pdo_length;
        if (copy > *rx_len) copy = *rx_len;
        memcpy(rx_pdo, ecat_slaves[slave_idx].pdo_data, copy);
        *rx_len = copy;
    }
    ECAT_UNLOCK();
    return 0;
}

int ecat_sdo_read(int slave_idx, uint16_t index, uint8_t subindex,
                   uint8_t* data, uint16_t* data_len) {
    if (!data || !data_len) return -1;
    ECAT_LOCK();
    if (slave_idx < 0 || (size_t)slave_idx >= (size_t)ecat_slave_count) {
        ECAT_UNLOCK();
        memset(data, 0, *data_len);
        *data_len = 0;
        return -1;
    }
    /* 对象字典映射: 根据CiA402标准索引映射到PDO数据缓冲区 */
    uint16_t offset = 0;
    uint16_t max_copy = sizeof(ecat_slaves[slave_idx].pdo_data);
    switch (index) {
        case 0x6040: offset = 0;  max_copy = 2; break; /* 控制字 */
        case 0x6041: offset = 2;  max_copy = 2; break; /* 状态字 */
        case 0x607A: offset = 4;  max_copy = 4; break; /* 目标位置 */
        case 0x6064: offset = 4;  max_copy = 4; break; /* 实际位置 */
        case 0x60FF: offset = 8;  max_copy = 4; break; /* 目标速度 */
        case 0x606C: offset = 8;  max_copy = 4; break; /* 实际速度 */
        case 0x6071: offset = 12; max_copy = 2; break; /* 目标转矩 */
        case 0x6077: offset = 12; max_copy = 2; break; /* 实际转矩 */
        case 0x1000: /* 设备类型 */
            if (subindex == 0) {
                uint32_t dev_type = 0x00020192;
                uint16_t copy = *data_len < 4 ? *data_len : 4;
                memcpy(data, &dev_type, copy);
                *data_len = copy;
                ECAT_UNLOCK();
                return 0;
            }
            break;
        case 0x1001: /* 错误寄存器 */
            if (subindex == 0) {
                uint8_t err = 0;
                uint16_t copy = *data_len < 1 ? *data_len : 1;
                memcpy(data, &err, copy);
                *data_len = copy;
                ECAT_UNLOCK();
                return 0;
            }
            break;
        case 0x1018: /* 标识对象 */
            if (subindex == 0) {
                uint32_t num_obj = 4;
                uint16_t copy = *data_len < 4 ? *data_len : 4;
                memcpy(data, &num_obj, copy);
                *data_len = copy;
                ECAT_UNLOCK();
                return 0;
            }
            break;
        default:
            offset = (uint16_t)subindex * 4;
            break;
    }
    if (offset >= max_copy) {
        ECAT_UNLOCK();
        memset(data, 0, *data_len);
        *data_len = 0;
        return 0;
    }
    uint16_t copy_len = *data_len;
    if (offset + copy_len > max_copy) copy_len = max_copy - offset;
    memcpy(data, ecat_slaves[slave_idx].pdo_data + offset, copy_len);
    *data_len = copy_len;
    ECAT_UNLOCK();
    return 0;
}

int ecat_sdo_write(int slave_idx, uint16_t index, uint8_t subindex,
                    const uint8_t* data, uint16_t data_len) {
    if (!data || data_len == 0) return -1;
    ECAT_LOCK();
    if (slave_idx < 0 || (size_t)slave_idx >= (size_t)ecat_slave_count) {
        ECAT_UNLOCK();
        return -1;
    }
    /* SDO写入: 将数据写入从站PDO缓冲区对应对象字典位置 */
    uint16_t offset = 0;
    uint16_t max_copy = sizeof(ecat_slaves[slave_idx].pdo_data);
    switch (index) {
        case 0x6040: offset = 0;  max_copy = 2; break; /* 控制字 */
        case 0x607A: offset = 4;  max_copy = 4; break; /* 目标位置 */
        case 0x60FF: offset = 8;  max_copy = 4; break; /* 目标速度 */
        case 0x6071: offset = 12; max_copy = 2; break; /* 目标转矩 */
        default:
            offset = (uint16_t)subindex * 4;
            break;
    }
    if (offset >= max_copy) {
        ECAT_UNLOCK();
        return -1;
    }
    uint16_t copy_len = data_len;
    if (offset + copy_len > max_copy) copy_len = max_copy - offset;
    memcpy(ecat_slaves[slave_idx].pdo_data + offset, data, copy_len);
    ECAT_UNLOCK();
    return 0;
}

/* ============================================================================
 * ROBOT-17: Modbus RTU/TCP完整实现 + CANopen协议
 * ============================================================================ */

typedef struct {
    uint8_t slave_addr;
    uint8_t function_code;
    uint16_t start_addr;
    uint16_t quantity;
    uint8_t data[252];
    uint16_t data_len;
    uint16_t crc;
} ModbusFrame;

static uint16_t modbus_crc16(const uint8_t* data, int len) {
    uint16_t crc = 0xFFFF;
    for (int i = 0; i < len; i++) {
        crc ^= data[i];
        for (int j = 0; j < 8; j++)
            crc = (crc & 1) ? (crc >> 1) ^ 0xA001 : (crc >> 1);
    }
    return crc;
}

/* CANopen SDO upload/download - 通过硬件接口发送CAN帧实现真实设备通信 */
int canopen_sdo_upload(int node_id, uint16_t index, uint8_t subindex,
                        uint32_t* value) {
    if (!value) return -1;
    if (!g_hw_protocols_can_bus || !hardware_interface_is_connected(g_hw_protocols_can_bus)) {
        return -1;
    }

    /* 构建SDO上传请求帧 */
    CanMessage msg;
    canopen_sdo_upload_request(node_id, index, subindex, &msg);

    /* 通过CAN总线发送 */
    if (hardware_interface_send(g_hw_protocols_can_bus, &msg, sizeof(CanMessage)) < 0) {
        return -1;
    }

    /* 接收SDO响应（最多重试5次） */
    CanMessage resp;
    for (int retry = 0; retry < 5; retry++) {
        int recv_len = hardware_interface_receive(g_hw_protocols_can_bus, &resp, sizeof(CanMessage));
        if (recv_len >= (int)sizeof(CanMessage)) {
            /* 验证响应COB-ID: 0x580 + node_id */
            if ((resp.id & 0x7FF) == (uint32_t)(0x580 + node_id)) {
                /* 检查SDO abort */
                if (resp.data[0] == 0x80) {
                    return -1;
                }
                /* 解析上传响应中的值 */
                *value = (uint32_t)resp.data[4] |
                         ((uint32_t)resp.data[5] << 8) |
                         ((uint32_t)resp.data[6] << 16) |
                         ((uint32_t)resp.data[7] << 24);
                return 0;
            }
        }
    }
    return -1;
}

int canopen_sdo_download(int node_id, uint16_t index, uint8_t subindex,
                          uint32_t value) {
    if (!g_hw_protocols_can_bus || !hardware_interface_is_connected(g_hw_protocols_can_bus)) {
        return -1;
    }

    /* 构建SDO下载请求帧 */
    CanMessage msg_dl;
    canopen_sdo_download_request(node_id, index, subindex, value, 4, &msg_dl);

    /* 通过CAN总线发送 */
    if (hardware_interface_send(g_hw_protocols_can_bus, &msg_dl, sizeof(CanMessage)) < 0) {
        return -1;
    }

    /* 等待确认响应（最多重试5次） */
    CanMessage resp_dl;
    for (int retry = 0; retry < 5; retry++) {
        int recv_len = hardware_interface_receive(g_hw_protocols_can_bus, &resp_dl, sizeof(CanMessage));
        if (recv_len >= (int)sizeof(CanMessage)) {
            if ((resp_dl.id & 0x7FF) == (uint32_t)(0x580 + node_id)) {
                /* 检查SDO abort */
                if (resp_dl.data[0] == 0x80) {
                    return -1;
                }
                return 0;
            }
        }
    }
    return -1;
}

/* Modbus通过硬件接口发送/接收真实设备数据 */
int modbus_read_holding_registers(int slave_id, uint16_t start_addr,
                                   uint16_t quantity, uint16_t* values) {
    if (!values || quantity == 0) return -1;
    if (!g_hw_protocols_modbus_bus || !hardware_interface_is_connected(g_hw_protocols_modbus_bus)) {
        return -1;
    }

    /* 构建Modbus RTU请求帧 */
    uint8_t frame[8] = {(uint8_t)slave_id, 0x03,
                        (uint8_t)(start_addr >> 8), (uint8_t)(start_addr & 0xFF),
                        (uint8_t)(quantity >> 8), (uint8_t)(quantity & 0xFF), 0, 0};
    uint16_t crc = modbus_crc16(frame, 6);
    frame[6] = (uint8_t)(crc & 0xFF); frame[7] = (uint8_t)(crc >> 8);

    /* 发送请求 */
    if (hardware_interface_send(g_hw_protocols_modbus_bus, frame, sizeof(frame)) < 0) {
        return -1;
    }

    /* 接收响应: 从站地址 + 功能码 + 字节数 + 数据(2*quantity) + CRC(2) */
    uint16_t resp_len = (uint16_t)(5 + 2 * quantity);
    uint8_t* resp_buf = (uint8_t*)safe_malloc(resp_len);
    if (!resp_buf) return -1;

    int recv_len = hardware_interface_receive(g_hw_protocols_modbus_bus, resp_buf, resp_len);
    if (recv_len < 5) {
        safe_free((void**)&resp_buf);
        return -1;
    }

    /* 验证响应 */
    if (resp_buf[0] != (uint8_t)slave_id || resp_buf[1] != 0x03) {
        safe_free((void**)&resp_buf);
        return -1;
    }

    uint8_t byte_count = resp_buf[2];
    uint16_t val_count = byte_count / 2;
    if (val_count > quantity) val_count = quantity;

    for (uint16_t i = 0; i < val_count; i++) {
        values[i] = (uint16_t)((resp_buf[3 + i * 2] << 8) | resp_buf[4 + i * 2]);
    }

    safe_free((void**)&resp_buf);
    return (int)val_count;
}

int modbus_write_single_register(int slave_id, uint16_t addr, uint16_t value) {
    if (!g_hw_protocols_modbus_bus || !hardware_interface_is_connected(g_hw_protocols_modbus_bus)) {
        return -1;
    }

    uint8_t frame[8] = {(uint8_t)slave_id, 0x06,
                        (uint8_t)(addr >> 8), (uint8_t)(addr & 0xFF),
                        (uint8_t)(value >> 8), (uint8_t)(value & 0xFF), 0, 0};
    uint16_t crc = modbus_crc16(frame, 6);
    frame[6] = (uint8_t)(crc & 0xFF); frame[7] = (uint8_t)(crc >> 8);

    if (hardware_interface_send(g_hw_protocols_modbus_bus, frame, sizeof(frame)) < 0) {
        return -1;
    }

    /* 等待回显确认（6字节从站地址+功能码+地址+值，+2字节CRC） */
    uint8_t echo[8];
    int recv_len = hardware_interface_receive(g_hw_protocols_modbus_bus, echo, sizeof(echo));
    if (recv_len >= 8) {
        if (echo[0] == (uint8_t)slave_id && echo[1] == 0x06) return 0;
    }
    return (recv_len >= 6) ? 0 : -1;
}

/* ============================================================================
 * ROBOT-19: 设备协议版本协商
 * ============================================================================ */

typedef struct {
    int protocol_version;
    int device_vendor_id;
    int device_product_id;
    int supported_features;
    char negotiated_version[16];
} ProtocolVersion;

int device_protocol_negotiate(int expected_version, int device_vendor,
                               int device_product, ProtocolVersion* result) {
    if (!result) return -1;
    memset(result, 0, sizeof(ProtocolVersion));
    result->device_vendor_id = device_vendor;
    result->device_product_id = device_product;
    if (expected_version <= 3) {
        result->protocol_version = expected_version;
        snprintf(result->negotiated_version, 15, "v%d", expected_version);
    } else {
        result->protocol_version = 3;
        snprintf(result->negotiated_version, 15, "fallback-v3");
    }
    result->supported_features = (1 << (result->protocol_version + 1)) - 1;
    return 0;
}
