/**
 * @file hardware_interface.h
 * @brief 硬件接口抽象层
 * 
 * 提供硬件通信接口抽象，支持串口、网络、CAN总线等通信协议。
 * 遵循100%纯C语言原则，不依赖任何第三方库。
 */

#ifndef SELFLNN_HARDWARE_INTERFACE_H
#define SELFLNN_HARDWARE_INTERFACE_H

#include "selflnn/utils/platform.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 硬件接口类型枚举
 */
typedef enum {
    HARDWARE_TYPE_SERIAL = 0,       /**< 串口通信 */
    HARDWARE_TYPE_TCP = 1,          /**< TCP网络通信 */
    HARDWARE_TYPE_UDP = 2,          /**< UDP网络通信 */
    HARDWARE_TYPE_CAN = 3,          /**< CAN总线通信 */
    HARDWARE_TYPE_MODBUS_TCP = 4,   /**< Modbus TCP通信 */
    HARDWARE_TYPE_MODBUS_RTU = 5,   /**< Modbus RTU通信 */
    HARDWARE_TYPE_WEBSOCKET = 6,    /**< WebSocket通信 */
    HARDWARE_TYPE_CUSTOM = 7,       /**< 自定义通信 */
    HARDWARE_TYPE_I2C = 8,          /**< I2C总线通信 */
    HARDWARE_TYPE_SPI = 9,          /**< SPI总线通信 */
    HARDWARE_TYPE_GPIO = 10         /**< GPIO通用输入输出 */
} HardwareType;

/**
 * @brief 串口配置结构体
 */
typedef struct {
    char port_name[64];           /**< 串口名称（如COM1、/dev/ttyUSB0） */
    int baud_rate;                /**< 波特率 */
    int data_bits;                /**< 数据位（5,6,7,8） */
    int stop_bits;                /**< 停止位（1,2） */
    int parity;                   /**< 校验位（0:无校验，1:奇校验，2:偶校验） */
    int flow_control;             /**< 流控制（0:无，1:硬件，2:软件） */
    int timeout_ms;               /**< 超时时间（毫秒） */
} SerialConfig;

/**
 * @brief 网络配置结构体
 */
typedef struct {
    char host[256];               /**< 主机地址 */
    int port;                     /**< 端口号 */
    int protocol;                 /**< 协议类型（TCP/UDP） */
    int connect_timeout_ms;       /**< 连接超时（毫秒） */
    int receive_timeout_ms;       /**< 接收超时（毫秒） */
    int send_timeout_ms;          /**< 发送超时（毫秒） */
    int keepalive;                /**< 是否启用KeepAlive */
} NetworkConfig;

/**
 * @brief CAN总线配置结构体
 */
#define SELFLNN_HARDWARE_INTERFACE_CAN_CONFIG_DEFINED
typedef struct {
    char interface[64];           /**< CAN接口名称（如can0） */
    int bitrate;                  /**< 比特率（如500000） */
    int can_id;                   /**< CAN ID */
    int extended_frame;           /**< 是否使用扩展帧 */
    int loopback;                 /**< 是否启用回环模式 */
    int receive_own_msgs;         /**< 是否接收自己发送的消息 */
} CanConfig;

/**
 * @brief Modbus配置结构体
 */
#define SELFLNN_HARDWARE_INTERFACE_MODBUS_CONFIG_DEFINED
typedef struct {
    int slave_id;                 /**< 从站ID */
    int function_code;            /**< 功能码 */
    int start_address;            /**< 起始地址 */
    int num_registers;            /**< 寄存器数量 */
    int timeout_ms;               /**< 超时时间（毫秒） */
    int retry_count;              /**< 重试次数 */
} ModbusConfig;

/**
 * @brief I2C总线配置结构体
 */
typedef struct {
    int scl_pin;                  /**< SCL时钟引脚号（软件模拟） */
    int sda_pin;                  /**< SDA数据引脚号（软件模拟） */
    int bus_speed_khz;            /**< 总线速度（kHz），标准100kHz/快速400kHz/高速3.4MHz */
    int clock_stretch_enabled;    /**< 是否启用时钟拉伸检测 */
    int multi_master_enabled;     /**< 是否启用多主机模式 */
    int slave_address;            /**< 本机从机地址（多主机模式下使用） */
    int retry_count;              /**< 通信重试次数 */
    int timeout_ms;               /**< 超时时间（毫秒） */
} I2cConfig;

/**
 * @brief SPI总线配置结构体
 */
typedef struct {
    int sclk_pin;                 /**< SCLK时钟引脚号 */
    int mosi_pin;                 /**< MOSI主机输出从机输入引脚号 */
    int miso_pin;                 /**< MISO主机输入从机输出引脚号 */
    int cs_pin;                   /**< CS片选引脚号 */
    int mode;                     /**< SPI模式（0-3），对应CPOL/CPHA组合 */
    int bit_order;                /**< 位序（0:MSB优先,1:LSB优先） */
    int speed_hz;                 /**< 时钟频率（Hz） */
    int bits_per_word;            /**< 每字位数（通常为8） */
    int cs_active_low;            /**< 片选是否低电平有效（默认1） */
    int cs_delay_us;              /**< 片选建立/保持延迟（微秒） */
    int use_dual_mode;            /**< 是否使用Dual SPI模式 */
    int use_quad_mode;            /**< 是否使用Quad SPI模式 */
    int dma_enabled;              /**< 是否启用DMA传输 */
} SpiConfig;

/**
 * @brief GPIO引脚配置结构体
 */
typedef struct {
    int pin;                      /**< 引脚号 */
    int direction;                /**< 方向（0:输入,1:输出） */
    int pull_mode;                /**< 上下拉模式（0:无,1:上拉,2:下拉） */
    int initial_value;            /**< 初始输出值（输出模式时有效） */
    int interrupt_enabled;        /**< 是否启用中断 */
    int interrupt_trigger;        /**< 中断触发方式（0:上升沿,1:下降沿,2:双沿,3:低电平,4:高电平） */
    int debounce_ms;              /**< 去抖时间（毫秒） */
    char label[32];               /**< 引脚标签 */
} GpioConfig;

/**
 * @brief I2C消息结构体（用于多消息事务）
 */
typedef struct {
    uint8_t address;              /**< 从机地址（7位，左对齐） */
    uint8_t* buffer;              /**< 数据缓冲区 */
    size_t length;                /**< 数据长度 */
    int is_read;                  /**< 是否为读操作（0:写,1:读） */
    int no_stop;                  /**< 完成后不发送停止条件（用于重复起始条件） */
} I2cMessage;

/**
 * @brief SPI传输段结构体（用于多段传输）
 */
typedef struct {
    const uint8_t* tx_buffer;     /**< 发送数据缓冲区（可为NULL仅接收） */
    uint8_t* rx_buffer;          /**< 接收数据缓冲区（可为NULL仅发送） */
    size_t length;                /**< 传输长度 */
    int keep_cs_active;           /**< 传输后保持片选有效 */
    int delay_us;                 /**< 传输后延迟（微秒） */
} SpiTransferSegment;

/**
 * @brief GPIO中断事件
 */
typedef struct {
    int pin;                      /**< 触发中断的引脚号 */
    int event_type;               /**< 事件类型（0:上升沿,1:下降沿,2:双沿） */
    int64_t timestamp_us;         /**< 事件时间戳（微秒） */
    int pin_value;                /**< 触发时的引脚电平 */
} GpioInterruptEvent;

/**
 * @brief 硬件运行模式枚举
 *
 * 决定硬件接口在无法连接真实硬件时的行为。
 * HW_MODE_SIMULATION(2) 已永久禁用——不再允许生成仿真传感器数据。
 */
typedef enum {
    HW_MODE_AUTO = 0,         /**< 自动模式：尝试连接真实硬件，失败时返回错误 */
    HW_MODE_REAL = 1          /**< 真实模式：必须连接真实硬件，失败时返回错误 */
} HardwareMode;

/**
 * @brief 硬件接口配置结构体
 */
typedef struct {
    HardwareType type;            /**< 接口类型 */
    HardwareMode mode;            /**< 硬件运行模式（默认HW_MODE_AUTO） */
    union {
        SerialConfig serial;      /**< 串口配置 */
        NetworkConfig network;    /**< 网络配置 */
        CanConfig can;            /**< CAN配置 */
        ModbusConfig modbus;      /**< Modbus配置 */
        I2cConfig i2c;            /**< I2C配置 */
        SpiConfig spi;            /**< SPI配置 */
        GpioConfig gpio;          /**< GPIO配置 */
    } config;
    
    int enable_async;             /**< 是否启用异步通信 */
    int buffer_size;              /**< 缓冲区大小 */
    int enable_logging;           /**< 是否启用日志记录 */
    char log_file[256];           /**< 日志文件路径 */
} HardwareConfig;

/**
 * @brief 硬件接口句柄
 */
typedef struct HardwareInterface HardwareInterface;

/**
 * @brief 创建机器人硬件接口实例
 * 
 * @param config 硬件配置
 * @return HardwareInterface* 硬件接口句柄，失败返回NULL
 */
HardwareInterface* robot_hardware_interface_create(const HardwareConfig* config);

/**
 * @brief 释放硬件接口实例
 * 
 * @param hw 硬件接口句柄
 */
void hardware_interface_free(HardwareInterface* hw);

/**
 * @brief 销毁机器人硬件接口实例（断开连接并释放资源）
 * 
 * @param hw 硬件接口句柄
 */
void robot_hardware_interface_destroy(HardwareInterface* hw);

/**
 * @brief 连接硬件设备
 * 
 * @param hw 硬件接口句柄
 * @return int 成功返回0，失败返回错误码
 */
int hardware_interface_connect(HardwareInterface* hw);

/**
 * @brief 断开硬件设备连接
 * 
 * @param hw 硬件接口句柄
 * @return int 成功返回0，失败返回错误码
 */
int hardware_interface_disconnect(HardwareInterface* hw);

/**
 * @brief 检查连接状态
 * 
 * @param hw 硬件接口句柄
 * @return int 已连接返回1，未连接返回0，错误返回-1
 */
int hardware_interface_is_connected(HardwareInterface* hw);

/**
 * @brief 检查是否处于仿真模式（已永久禁用）
 * 
 * 仿真模式已永久禁用，此函数始终返回0（非仿真模式）。
 * 
 * @param hw 硬件接口句柄
 * @return int 始终返回0（仿真模式已禁用），错误返回-1
 */
int hardware_interface_is_simulation(HardwareInterface* hw);

/**
 * @brief 设置仿真模式下的机器人运动状态（已永久禁用）
 * 
 * 仿真模式已永久禁用，此函数始终返回错误。
 * 
 * @param hw 硬件接口句柄
 * @param linear_velocity 线速度[m/s] (x,y,z)，可为NULL
 * @param angular_velocity 角速度[rad/s] (roll,pitch,yaw)，可为NULL
 * @param linear_acceleration 线加速度[m/s²] (x,y,z)，可为NULL
 * @return int 始终返回-1（仿真模式已禁用）
 */
int hardware_interface_set_simulation_motion(HardwareInterface* hw,
                                             const double linear_velocity[3],
                                             const double angular_velocity[3],
                                             const double linear_acceleration[3]);

/**
 * @brief 设置仿真模式下的机器人地理位置（已永久禁用）
 * 
 * 仿真模式已永久禁用，此函数始终返回错误。
 * 
 * @param hw              硬件接口句柄
 * @param latitude_deg    纬度（度，-90~90，北半球为正）
 * @param longitude_deg   经度（度，-180~180，东半球为正）
 * @param altitude_m      海拔高度（米）
 * @return int            始终返回-1（仿真模式已禁用）
 */
int hardware_interface_set_simulation_position(HardwareInterface* hw,
                                                double latitude_deg,
                                                double longitude_deg,
                                                double altitude_m);

/**
 * @brief 发送数据
 * 
 * @param hw 硬件接口句柄
 * @param data 数据缓冲区
 * @param size 数据大小
 * @return int 成功发送的字节数，失败返回-1
 */
int hardware_interface_send(HardwareInterface* hw, const void* data, size_t size);

/**
 * @brief 接收数据
 * 
 * @param hw 硬件接口句柄
 * @param buffer 接收缓冲区
 * @param size 缓冲区大小
 * @return int 实际接收的字节数，失败返回-1
 */
int hardware_interface_receive(HardwareInterface* hw, void* buffer, size_t size);

/**
 * @brief 发送机器人控制命令
 * 
 * @param hw 硬件接口句柄
 * @param command 机器人控制命令数据
 * @param command_size 命令数据大小（字节）
 * @return int 成功返回发送的字节数，失败返回-1
 */
int hardware_interface_send_command(HardwareInterface* hw, const void* command, size_t command_size);

/**
 * @brief 接收传感器数据
 * 
 * @param hw 硬件接口句柄
 * @param sensor_data 传感器数据输出缓冲区
 * @param max_size 最大数据大小
 * @return int 实际接收的数据大小，失败返回-1
 */
int hardware_interface_receive_sensor_data(HardwareInterface* hw, void* sensor_data, size_t max_size);

/**
 * @brief 获取最后错误信息
 * 
 * @param hw 硬件接口句柄
 * @return const char* 错误信息字符串
 */
const char* hardware_interface_get_last_error(HardwareInterface* hw);

/**
 * @brief 获取硬件统计信息
 * 
 * @param hw 硬件接口句柄
 * @param bytes_sent 发送字节数（输出，可为NULL）
 * @param bytes_received 接收字节数（输出，可为NULL）
 * @param connection_count 连接次数（输出，可为NULL）
 * @param error_count 错误计数（输出，可为NULL）
 * @return int 成功返回0，失败返回-1
 */
int hardware_interface_get_stats(HardwareInterface* hw,
                                size_t* bytes_sent,
                                size_t* bytes_received,
                                size_t* connection_count,
                                size_t* error_count);

/**
 * @brief PWM通道配置结构体
 */
typedef struct {
    int channel;              /**< PWM通道号（0~N） */
    int gpio_pin;             /**< GPIO引脚号 */
    double frequency;         /**< PWM频率（Hz） */
    double duty_cycle;        /**< 占空比（0.0~1.0） */
    int polarity;             /**< 极性（0:正常,1:反向） */
    int enabled;              /**< 是否启用 */
} PwmChannelConfig;

/**
 * @brief 编码器数据
 */
typedef struct {
    int64_t position;         /**< 当前位置（脉冲数） */
    double velocity;          /**< 当前速度（脉冲/秒） */
    double acceleration;      /**< 当前加速度（脉冲/秒²） */
    int direction;            /**< 方向（1:正向,-1:反向,0:停止） */
    double last_update_time;  /**< 最后更新时间（秒） */
    int overflow_count;       /**< 溢出计数 */
} EncoderData;

/**
 * @brief IMU原始数据
 */
typedef struct {
    double accelerometer[3];  /**< 加速度计（m/s²）[x,y,z] */
    double gyroscope[3];      /**< 陀螺仪（rad/s）[x,y,z] */
    double magnetometer[3];   /**< 磁力计（µT）[x,y,z] */
    double temperature;       /**< 温度（°C） */
    double timestamp;         /**< 时间戳（秒） */
    int is_simulated_data; /**< 1=仿真计算数据, 0=真实硬件传感器数据 */
} ImuRawData;

/**
 * @brief IMU姿态数据（四元数+欧拉角）
 */
typedef struct {
    double quaternion[4];     /**< 姿态四元数 [w,x,y,z] */
    double euler_angles[3];   /**< 欧拉角（rad）[roll,pitch,yaw] */
    double rotation_matrix[9]; /**< 旋转矩阵 3x3 */
    double linear_acceleration[3]; /**< 线性加速度（m/s²，去除重力）[x,y,z] */
    double angular_velocity[3]; /**< 角速度（rad/s）[x,y,z] */
    double confidence;        /**< 置信度（0.0~1.0） */
    double timestamp;         /**< 时间戳（秒） */
} ImuOrientation;

/**
 * @brief 初始化PWM通道
 * 
 * @param config PWM通道配置
 * @return int 成功返回0，失败返回-1
 */
int hardware_interface_pwm_init(const PwmChannelConfig* config);

/**
 * @brief 设置PWM占空比
 * 
 * @param channel PWM通道号
 * @param duty_cycle 占空比（0.0~1.0）
 * @return int 成功返回0，失败返回-1
 */
int hardware_interface_pwm_set_duty_cycle(int channel, double duty_cycle);

/**
 * @brief 设置PWM频率
 * 
 * @param channel PWM通道号
 * @param frequency 频率（Hz）
 * @return int 成功返回0，失败返回-1
 */
int hardware_interface_pwm_set_frequency(int channel, double frequency);

/**
 * @brief 启用/禁用PWM通道
 * 
 * @param channel PWM通道号
 * @param enabled 是否启用
 * @return int 成功返回0，失败返回-1
 */
int hardware_interface_pwm_enable(int channel, int enabled);

/**
 * @brief 释放PWM通道
 * 
 * @param channel PWM通道号
 * @return int 成功返回0，失败返回-1
 */
int hardware_interface_pwm_release(int channel);

/**
 * @brief 读取编码器数据
 * 
 * @param channel 编码器通道号
 * @param data 编码器数据输出
 * @return int 成功返回0，失败返回-1
 */
int hardware_interface_encoder_read(int channel, EncoderData* data);

/**
 * @brief 重置编码器计数
 * 
 * @param channel 编码器通道号
 * @return int 成功返回0，失败返回-1
 */
int hardware_interface_encoder_reset(int channel);

/**
 * @brief 设置编码器零位
 * 
 * @param channel 编码器通道号
 * @return int 成功返回0，失败返回-1
 */
int hardware_interface_encoder_set_zero(int channel);

/**
 * @brief 读取IMU原始传感器数据
 * 
 * @param hw 硬件接口句柄（用于通信，可为NULL选择默认I2C）
 * @param data IMU原始数据输出
 * @return int 成功返回0，失败返回-1
 */
int hardware_interface_imu_read_raw(HardwareInterface* hw, ImuRawData* data);

/**
 * @brief 计算IMU姿态（使用Madgwick/Mahony滤波器）
 * 
 * @param raw IMU原始数据
 * @param dt 时间步长（秒）
 * @param orientation 姿态输出
 * @return int 成功返回0，失败返回-1
 */
int hardware_interface_imu_compute_orientation(const ImuRawData* raw, double dt, ImuOrientation* orientation);

/**
 * @brief IMU校准（零偏估计）
 * 
 * @param hw 硬件接口句柄
 * @param samples 采样次数
 * @param gyro_bias 陀螺仪零偏输出[3]
 * @param accel_bias 加速度计零偏输出[3]
 * @param mag_bias 磁力计零偏输出[3]
 * @return int 成功返回0，失败返回-1
 */
int hardware_interface_imu_calibrate(HardwareInterface* hw, int samples,
                                     double gyro_bias[3], double accel_bias[3], double mag_bias[3]);

/**
 * @brief 获取马达控制状态
 * 
 * @param motor_id 马达ID
 * @param current_duty_cycle 当前占空比输出
 * @param current_rpm 当前转速输出（RPM）
 * @param motor_temperature 马达温度输出（°C）
 * @return int 成功返回0，失败返回-1
 */
int hardware_interface_motor_get_status(int motor_id, double* current_duty_cycle,
                                        double* current_rpm, double* motor_temperature);

/**
 * @brief 设置马达PID参数
 * 
 * @param motor_id 马达ID
 * @param kp 比例系数
 * @param ki 积分系数
 * @param kd 微分系数
 * @return int 成功返回0，失败返回-1
 */
int hardware_interface_motor_set_pid(int motor_id, double kp, double ki, double kd);

/**
 * @brief 设置目标转速（速度模式）
 * 
 * @param motor_id 马达ID
 * @param target_rpm 目标转速
 * @return int 成功返回0，失败返回-1
 */
int hardware_interface_motor_set_target_rpm(int motor_id, double target_rpm);

/**
 * @brief 紧急停止所有马达
 * 
 * @return int 成功返回0，失败返回-1
 */
int hardware_interface_motor_emergency_stop(void);

/* ==================== I2C总线接口 ==================== */

/**
 * @brief I2C总线主模式初始化
 * 
 * @param scl_pin SCL引脚号
 * @param sda_pin SDA引脚号
 * @param speed_khz 总线速度（kHz）
 * @return int 成功返回0，失败返回-1
 */
int hardware_interface_i2c_init(int scl_pin, int sda_pin, int speed_khz);

/**
 * @brief I2C产生起始条件
 * 
 * @return int 成功返回0，失败返回-1
 */
int hardware_interface_i2c_start(void);

/**
 * @brief I2C产生停止条件
 * 
 * @return int 成功返回0，失败返回-1
 */
int hardware_interface_i2c_stop(void);

/**
 * @brief I2C发送字节并检查ACK
 * 
 * @param data 要发送的字节
 * @return int 成功（收到ACK）返回0，失败（收到NACK）返回-1
 */
int hardware_interface_i2c_write_byte(uint8_t data);

/**
 * @brief I2C读取字节并发送ACK/NACK
 * 
 * @param ack 发送ACK(0)或NACK(1)
 * @return uint8_t 读取到的字节
 */
uint8_t hardware_interface_i2c_read_byte(int ack);

/**
 * @brief I2C向从机地址发送数据（完整事务：起始+地址+数据+停止）
 * 
 * @param addr 7位从机地址
 * @param data 数据缓冲区
 * @param len 数据长度
 * @return int 成功返回写入字节数，失败返回-1
 */
int hardware_interface_i2c_write(uint8_t addr, const uint8_t* data, size_t len);

/**
 * @brief I2C从从机地址读取数据（完整事务：起始+地址+读取+停止）
 * 
 * @param addr 7位从机地址
 * @param buffer 接收缓冲区
 * @param len 要读取的字节数
 * @return int 成功返回读取字节数，失败返回-1
 */
int hardware_interface_i2c_read(uint8_t addr, uint8_t* buffer, size_t len);

/**
 * @brief I2C组合事务（写后读，支持重复起始条件）
 * 
 * @param messages I2C消息数组
 * @param msg_count 消息数量
 * @return int 成功返回0，失败返回-1
 */
int hardware_interface_i2c_transfer(I2cMessage* messages, size_t msg_count);

/**
 * @brief I2C扫描总线上的设备
 * 
 * @param found_addrs 找到的设备地址数组
 * @param max_count 最大返回数量
 * @return int 找到的设备数量，失败返回-1
 */
int hardware_interface_i2c_scan(uint8_t* found_addrs, size_t max_count);

/**
 * @brief I2C设置总线速度
 * 
 * @param speed_khz 速度（kHz）
 * @return int 成功返回0，失败返回-1
 */
int hardware_interface_i2c_set_speed(int speed_khz);

/* ==================== SPI总线接口 ==================== */

/**
 * @brief SPI总线主模式初始化
 * 
 * @param sclk_pin SCLK时钟引脚号
 * @param mosi_pin MOSI引脚号
 * @param miso_pin MISO引脚号
 * @param cs_pin CS片选引脚号
 * @param mode SPI模式（0-3）
 * @param speed_hz 时钟频率（Hz）
 * @return int 成功返回0，失败返回-1
 */
int hardware_interface_spi_init(int sclk_pin, int mosi_pin, int miso_pin,
                                int cs_pin, int mode, int speed_hz);

/**
 * @brief SPI全双工传输一个字节
 * 
 * @param tx_data 要发送的数据
 * @param rx_data 接收的数据输出（可为NULL）
 * @return int 成功返回0，失败返回-1
 */
int hardware_interface_spi_transfer_byte(uint8_t tx_data, uint8_t* rx_data);

/**
 * @brief SPI全双工批量传输
 * 
 * @param tx_data 发送数据缓冲区（可为NULL仅接收）
 * @param rx_data 接收数据缓冲区（可为NULL仅发送）
 * @param count 传输字节数
 * @return int 成功返回传输字节数，失败返回-1
 */
int hardware_interface_spi_transfer(const uint8_t* tx_data, uint8_t* rx_data, size_t count);

/**
 * @brief SPI多段传输（允许不同段的不同配置）
 * 
 * @param segments 传输段数组
 * @param segment_count 段数量
 * @return int 成功返回0，失败返回-1
 */
int hardware_interface_spi_transfer_multi(SpiTransferSegment* segments, size_t segment_count);

/**
 * @brief SPI设置片选状态
 * 
 * @param active 是否激活片选
 * @return int 成功返回0，失败返回-1
 */
int hardware_interface_spi_set_cs(int active);

/**
 * @brief SPI设置传输模式
 * 
 * @param mode SPI模式（0-3）
 * @return int 成功返回0，失败返回-1
 */
int hardware_interface_spi_set_mode(int mode);

/**
 * @brief SPI设置传输速度
 * 
 * @param speed_hz 速度（Hz）
 * @return int 成功返回0，失败返回-1
 */
int hardware_interface_spi_set_speed(int speed_hz);

/* ==================== GPIO通用接口 ==================== */

/**
 * @brief GPIO引脚初始化
 * 
 * @param pin 引脚号
 * @param direction 方向（0:输入,1:输出）
 * @param pull_mode 上下拉模式（0:无,1:上拉,2:下拉）
 * @return int 成功返回0，失败返回-1
 */
int hardware_interface_gpio_init(int pin, int direction, int pull_mode);

/**
 * @brief GPIO引脚初始化（完整配置）
 * 
 * @param config GPIO配置
 * @return int 成功返回0，失败返回-1
 */
int hardware_interface_gpio_init_full(const GpioConfig* config);

/**
 * @brief 设置GPIO引脚输出值
 * 
 * @param pin 引脚号
 * @param value 输出值（0:低电平,1:高电平）
 * @return int 成功返回0，失败返回-1
 */
int hardware_interface_gpio_set(int pin, int value);

/**
 * @brief 读取GPIO引脚输入值
 * 
 * @param pin 引脚号
 * @return int 引脚电平（0:低电平,1:高电平），失败返回-1
 */
int hardware_interface_gpio_get(int pin);

/**
 * @brief 切换GPIO引脚输出值
 * 
 * @param pin 引脚号
 * @return int 成功返回0，失败返回-1
 */
int hardware_interface_gpio_toggle(int pin);

/**
 * @brief 批量设置多个GPIO引脚
 * 
 * @param pins 引脚号数组
 * @param values 值数组
 * @param count 引脚数量
 * @return int 成功返回0，失败返回-1
 */
int hardware_interface_gpio_set_multi(const int* pins, const int* values, size_t count);

/**
 * @brief 批量读取多个GPIO引脚
 * 
 * @param pins 引脚号数组
 * @param values 值输出数组
 * @param count 引脚数量
 * @return int 成功返回0，失败返回-1
 */
int hardware_interface_gpio_get_multi(const int* pins, int* values, size_t count);

/**
 * @brief 设置GPIO引脚方向
 * 
 * @param pin 引脚号
 * @param direction 方向（0:输入,1:输出）
 * @return int 成功返回0，失败返回-1
 */
int hardware_interface_gpio_set_direction(int pin, int direction);

/**
 * @brief 设置GPIO引脚上下拉模式
 * 
 * @param pin 引脚号
 * @param pull_mode 上下拉模式（0:无,1:上拉,2:下拉）
 * @return int 成功返回0，失败返回-1
 */
int hardware_interface_gpio_set_pull_mode(int pin, int pull_mode);

/**
 * @brief 设置GPIO引脚中断
 * 
 * @param pin 引脚号
 * @param trigger 触发方式（0:上升沿,1:下降沿,2:双沿,3:低电平,4:高电平）
 * @return int 成功返回0，失败返回-1
 */
int hardware_interface_gpio_set_interrupt(int pin, int trigger);

/**
 * @brief 读取GPIO中断事件队列
 * 
 * @param events 事件缓冲区
 * @param max_count 最大事件数量
 * @return int 实际读取的事件数量，失败返回-1
 */
int hardware_interface_gpio_get_interrupt_events(GpioInterruptEvent* events, size_t max_count);

/**
 * @brief 释放GPIO引脚
 * 
 * @param pin 引脚号
 * @return int 成功返回0，失败返回-1
 */
int hardware_interface_gpio_release(int pin);

/**
 * @brief 释放所有GPIO引脚
 * 
 * @return int 成功返回0，失败返回-1
 */
int hardware_interface_gpio_release_all(void);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_HARDWARE_INTERFACE_H */