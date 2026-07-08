/**
 * @file version.h
 * @brief SELF-LNN版本号 (DEEP-005: 替代CMake生成的version.h.in)
 * 
 * 原系统使用CMake的configure_file从version.h.in生成，MSVC构建时该文件缺失。
 * 此文件为静态版本头，与CMake中的PROJECT_VERSION保持同步。
 * 
 * 版本号格式: MAJOR.MINOR.PATCH.TWEAK
 * - MAJOR: 重大架构变更，不保证向后兼容
 * - MINOR: 功能增量更新，保持向后兼容
 * - PATCH: 缺陷修复和补丁更新
 * - TWEAK: 微调版本号（热修复、构建配置调整等）
 */

#ifndef SELFLNN_VERSION_H
#define SELFLNN_VERSION_H

#define SELFLNN_VERSION_MAJOR 1
#define SELFLNN_VERSION_MINOR 5
#define SELFLNN_VERSION_PATCH 0
/* BLD-012修复: 添加TWEAK和FULL版本宏，与CMake中的PROJECT_VERSION_TWEAK/PROJECT_VERSION_FULL对齐 */
#define SELFLNN_VERSION_TWEAK 0
#define SELFLNN_VERSION_FULL "1.5.0.0"

#define SELFLNN_VERSION_STRING "1.5.0"

#endif /* SELFLNN_VERSION_H */
