# 产品设计能力库
message(STATUS "构建产品设计库: 包含源文件 product_design.c")

set(PRODUCT_DESIGN_SOURCES
    product_design.c
    product_design_enhanced.c
)

message(STATUS "产品设计库源文件: ${PRODUCT_DESIGN_SOURCES}")

add_library(selflnn_product_design ${PRODUCT_DESIGN_SOURCES})

target_include_directories(selflnn_product_design PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_link_libraries(selflnn_product_design PRIVATE 
    selflnn_core 
    selflnn_memory
    selflnn_utils
    selflnn_reasoning
)

# 安装规则
install(TARGETS selflnn_product_design
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../include/selflnn/product_design/
    DESTINATION include/selflnn/product_design
    FILES_MATCHING PATTERN "*.h"
)