# 实用工具库
# xml_parser.c和xml_parser.h已删除（孤儿文件，无外部调用者，~400行死代码）
# kinematics.c已内置轻量级URDF/XML解析器，无需额外XML工具
add_library(utils
    math_utils.c
    memory_utils.c
    perf.c
    platform.c
    string_utils.c
    json_parser.c
    energy_efficiency.c
    logging.c
    secure_random.c
    xorshift_prng.c
    time_utils.c
    config_loader.c
    compat_link_stubs.c
)

target_include_directories(utils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

# Windows下需要链接Winsock库和版本信息库
if(MSVC)
    target_link_libraries(utils PUBLIC ws2_32 version)
    set_source_files_properties(math_utils.c PROPERTIES COMPILE_FLAGS "/FImath.h")
elseif(WIN32)
    target_link_libraries(utils PUBLIC ws2_32 version)
endif()

# L-006修复: 所有内部模块已统一使用utils，此别名保留仅为外部兼容，计划v2.0移除
add_library(selflnn_utils ALIAS utils)

# 安装规则
install(TARGETS utils
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../include/selflnn/utils/
    DESTINATION include/selflnn/utils
    FILES_MATCHING PATTERN "*.h"
)