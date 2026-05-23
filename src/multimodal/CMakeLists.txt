# 多模态处理库
add_library(selflnn_multimodal
    multimodal.c
    stereo_calibration.c
    audio.c
    text.c
    sensor.c
    multimodal_manager.c
    dialogue.c
    depth_estimation.c
    point_cloud.c
    vad.c
    slam.c
    slam_frontend.c
    slam_backend.c
    slam_loop_closure.c
    slam_vocabulary.c
    slam_io.c
    audio_semantic.c
    audio_tools.c
    dialogue_deep.c
    speech_recognition.c
    tts.c
    tts_pinyin_real.c
    cfc_ocr_net.c
    ocr.c
    text_detection.c
    multimodal_teaching.c
    dialogue_memory.c
    teaching_loop.c
    object_recognition.c
    haptic_learning.c
    liquid_vision.c
    slam_enhance.c
    haptic_enhance.c
    camera_capture.c
    audio_capture.c
    character_segmentation.c
    image_recognition_deep.c
    unified_signal_processor.c
    unified_signal_processor_advanced.c
    unified_signal_processor_training.c
    data_collection_pipeline.c
    multimodal_unified_input.c
    multimodal_integration.c
    sensor_preprocessor_deep.c
    stereo_3d_reconstruction.c
    stereo_depth_enhance.c
    image_loader.c
    audio_loader.c
    speech_language_model.c
    environment_sound.c
)

target_include_directories(selflnn_multimodal PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_link_libraries(selflnn_multimodal PRIVATE selflnn_core)

if(WIN32)
    target_link_libraries(selflnn_multimodal PRIVATE vfw32.lib)
endif()

# 添加MSVC编译器选项解决PDB文件冲突
if(MSVC)
    target_compile_options(selflnn_multimodal PRIVATE /FS)
endif()

# 安装规则
install(TARGETS selflnn_multimodal
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../include/selflnn/multimodal/
    DESTINATION include/selflnn/multimodal
    FILES_MATCHING PATTERN "*.h"
)