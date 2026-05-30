#ifndef SELFLNN_HAPTIC_ENHANCE_H
#define SELFLNN_HAPTIC_ENHANCE_H

#include "selflnn/multimodal/haptic_learning.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
 * haptic_enhance.c 独有的函数声明
 * 所有基础版本的函数已在 haptic_learning.h 中声明（无需重复）
 * haptic_enhance.c 中增强版本的实现由链接器自动覆盖同名符号
 * ================================================================ */

/* 全局触觉处理器设置（haptic_enhance.c 独有函数） */
void haptic_enhance_set_global_processor(HapticCfcProcessor* proc);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_HAPTIC_ENHANCE_H */
