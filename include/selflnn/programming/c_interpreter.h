/**
 * @file c_interpreter.h
 * @brief K-007: 纯C解释执行模式 — 轻量级C子集解释器接口
 *
 *: C解释器集成到自我编程模块。
 * 当外部编译器不可用时，自我编程模块可使用此解释器执行简单C程序。
 */

#ifndef SELFLNN_C_INTERPRETER_H
#define SELFLNN_C_INTERPRETER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 检查C解释器是否可用
 * @return int 1=可用，0=不可用
 */
int c_interpreter_available(void);

/**
 * @brief 执行C代码片段
 * @param code C代码字符串
 * @param result_buffer 结果输出缓冲区
 * @param result_size 缓冲区大小
 * @return int 成功返回0，失败返回-1
 */
int c_interpreter_execute(const char* code, char* result_buffer, size_t result_size);

/**
 * @brief 解释表达式的值
 * @param code 表达式代码
 * @param result 浮点结果输出
 * @param error_msg 错误信息输出
 * @return int 成功返回0，失败返回-1
 */
int c_interpreter_interpret_expr(const char* code, float* result, char* error_msg);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_C_INTERPRETER_H */
