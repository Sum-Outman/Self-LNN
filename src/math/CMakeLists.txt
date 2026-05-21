# 数学运算库（矩阵运算接口）
add_library(selflnn_math
    matrix_ops.c
)

target_include_directories(selflnn_math PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

# 安装规则
install(TARGETS selflnn_math
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../include/selflnn/math/
    DESTINATION include/selflnn/math
    FILES_MATCHING PATTERN "*.h"
)
