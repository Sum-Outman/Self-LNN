# ZSFBUILD: MSVC不支持以下文件（头文件成员名不匹配，预存结构断裂）
# ros_gazebo_bridge.c hardware_detector.c 的struct成员名与头文件不兼容
if(NOT MSVC)
    set(ROBOT_EXTRA_SOURCES
        ros_gazebo_bridge.c
        hardware_detector.c
    )
else()
    set(ROBOT_EXTRA_SOURCES "")
endif()

# 机器人控制库
add_library(selflnn_robot
    robot.c
    robot_emergency.c
    robot_calibration.c
    robot_agent.c
    robot_control_advanced.c
    robot_vision_odometry.c
    hardware_interface.c
    simulator.c
    kinematics.c
    sensor_pipeline.c
    humanoid_sensor_fusion.c
    computer_operation.c
    path_planning.c
    swarm_control.c
    voice_motion_control.c
    device_protocols.c
    dynamics.c
    physics_engine.c
    sensor_simulation.c
    ros_bridge.c
    pybullet_bridge.c
    gazebo_bridge.c
    ros_node.c
    ros_robot_controller.c
    hardware_resource_manager.c
    swarm_coordination.c
    ros2_node.c
    ros_protocol.c
    gait_generator.c
    ${ROBOT_EXTRA_SOURCES}
)

target_compile_definitions(selflnn_robot PRIVATE WIN32_LEAN_AND_MEAN)

target_include_directories(selflnn_robot PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_link_libraries(selflnn_robot PRIVATE selflnn_core selflnn_reasoning selflnn_multisystem)

if(MSVC)
    target_link_libraries(selflnn_robot PRIVATE mf mfplat mfreadwrite mfuuid strmiids ole32 uuid mmdevapi propsys)
elseif(MINGW)
    target_link_libraries(selflnn_robot PRIVATE mf mfplat mfreadwrite ole32 strmiids uuid mmdevapi propsys)
endif()

# 安装规则
install(TARGETS selflnn_robot
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../include/selflnn/robot/
    DESTINATION include/selflnn/robot
    FILES_MATCHING PATTERN "*.h"
)