# 自我学习系统库
# DEADCODE-FIX: 高级探索策略默认启用。
# 如需跳过ICM/RND/Go-Explore/NoisyNet/Boltzmann/Thompson等未集成策略的编译，
# 添加编译定义: -DSELFLNN_SKIP_ADVANCED_EXPLORATION
# 仅UCB策略保持编译（被robot_agent.c实际使用）。
add_library(selflnn_learning
    learning.c
    meta_learning.c
    online_learning.c
    imitation_learning.c
    imitation_deep.c
    teach_by_showing.c
    manual_learning.c
    multi_agent.c
    reinforcement_learning.c
    exploration_strategies.c
)

target_include_directories(selflnn_learning PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_link_libraries(selflnn_learning PRIVATE selflnn_core selflnn_memory selflnn_reasoning selflnn_evolution)

if(MSVC)
    target_compile_options(selflnn_learning PRIVATE /FImath.h)
endif()

# 安装规则
install(TARGETS selflnn_learning
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../include/selflnn/learning/
    DESTINATION include/selflnn/learning
    FILES_MATCHING PATTERN "*.h"
)