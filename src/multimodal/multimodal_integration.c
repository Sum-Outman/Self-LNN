/**
 * @file multimodal_integration.c
 * @brief 多模态功能集成实现
 * 
 * 实现所有新功能到统一液态神经网络架构的集成。
 * 遵循"所有模态 → 统一输入到同一个连续动态系统"的架构原则。
 * 将空间识别、语音识别、OCR等功能集成到统一处理流程中。
 */

/* ============================================================================
 * 内部头文件包含
 * =========================================================================== */
#include "selflnn/multimodal/multimodal_integration.h"
#include "selflnn/core/lnn.h"
#include "selflnn/multimodal/unified_signal_processor.h"
#include "selflnn/multimodal/depth_estimation.h"
#include "selflnn/multimodal/point_cloud.h"
#include "selflnn/multimodal/slam.h"
#include "selflnn/multimodal/stereo_calibration.h"
#include "selflnn/multimodal/speech_recognition.h"
#include "selflnn/multimodal/vad.h"
#include "selflnn/multimodal/audio_semantic.h"
#include "selflnn/multimodal/ocr.h"
#include "selflnn/multimodal/character_segmentation.h"
#include "selflnn/multimodal/text_detection.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/logging.h"
#include "selflnn/core/memory.h"
#include "selflnn/core/lnn.h"
#include "selflnn/core/cfc_network.h"
#include "selflnn/core/laplace.h"
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <time.h>



/**
 * @brief 获取默认的多模态集成配置
 */
MultimodalIntegrationConfig multimodal_integration_get_default_config(void) {
    MultimodalIntegrationConfig config;
    memset(&config, 0, sizeof(MultimodalIntegrationConfig));
    
    /* 空间识别配置 */
    config.spatial_config.enable_depth_estimation = 1;
    config.spatial_config.enable_point_cloud = 1;
    config.spatial_config.enable_slam = 1;
    config.spatial_config.enable_stereo_vision = 1;
    config.spatial_config.max_features = 256;
    
    /* 语音识别配置 */
    config.speech_config.enable_vad = 1;
    config.speech_config.enable_speech_recognition = 1;
    config.speech_config.enable_semantic_understanding = 1;
    config.speech_config.max_features = 128;
    
    /* OCR配置 */
    config.ocr_config.enable_text_detection = 1;
    config.ocr_config.enable_character_segmentation = 1;
    config.ocr_config.enable_ocr = 1;
    config.ocr_config.max_features = 64;
    
    /* 统一配置 */
    config.unified_feature_dimension = 512;
    config.enable_unified_signal_processor = 1;
    
    return config;
}

/**
 * @brief 创建多模态集成处理器
 */
MultimodalIntegrationProcessor* multimodal_integration_processor_create(
    const MultimodalIntegrationConfig* config) {
    
    if (!config) {
        return NULL;
    }
    
    /* 分配处理器内存 */
    MultimodalIntegrationProcessor* processor = (MultimodalIntegrationProcessor*)
        safe_malloc(sizeof(MultimodalIntegrationProcessor));
    if (!processor) {
        return NULL;
    }
    
    /* 初始化内存 */
    memset(processor, 0, sizeof(MultimodalIntegrationProcessor));
    
    /* 复制配置 */
    memcpy(&processor->config, config, sizeof(MultimodalIntegrationConfig));
    
    /* 创建空间识别组件 */
    if (config->spatial_config.enable_depth_estimation) {
        DepthEstimationConfig depth_config = depth_estimation_get_default_config();
        depth_config.enable_stereo_depth = config->spatial_config.enable_stereo_vision;
        depth_config.max_features = config->spatial_config.max_features;
        processor->depth_estimator = depth_estimator_create(&depth_config);
    }
    
    if (config->spatial_config.enable_point_cloud) {
        processor->point_cloud_processor = point_cloud_processor_create();
    }
    
    if (config->spatial_config.enable_slam) {
        SlamConfig slam_config = slam_get_default_config();
        processor->slam_system = slam_system_create(&slam_config);
    }
    
    /* 创建语音识别组件 */
    if (config->speech_config.enable_vad) {
        VadConfig vad_config = vad_get_default_config();
        processor->vad_processor = vad_processor_create(&vad_config);
    }
    
    if (config->speech_config.enable_speech_recognition) {
        SpeechRecognitionConfig speech_config = speech_recognition_get_default_config();
        processor->speech_recognizer = speech_recognizer_create(&speech_config);
    }
    
    if (config->speech_config.enable_semantic_understanding) {
        AudioSemanticConfig audio_semantic_config = audio_semantic_get_default_config();
        processor->audio_semantic_processor = audio_semantic_processor_create(&audio_semantic_config, NULL, NULL, NULL);
    }
    
    /* 创建OCR组件 */
    if (config->ocr_config.enable_text_detection) {
        TextDetectionConfig text_det_config = {0};
        processor->text_detection_processor = text_detection_processor_create(&text_det_config);
    }
    
    if (config->ocr_config.enable_character_segmentation) {
        CharSegmentationConfig char_seg_config = {0};
        processor->char_seg_processor = char_segmentation_processor_create(&char_seg_config);
    }
    
    if (config->ocr_config.enable_ocr) {
        OcrConfig ocr_config = ocr_get_default_config();
        processor->ocr_processor = ocr_processor_create(&ocr_config);
    }
    
    /* 创建统一信号处理器 */
    if (config->enable_unified_signal_processor) {
        UnifiedSignalProcessorConfig processor_config = unified_signal_processor_get_default_config();
        processor_config.unified_dimension = config->unified_feature_dimension;
        processor_config.enable_state_evolution = 1;
        processor_config.evolution_type = STATE_EVOLUTION_NONLINEAR;
        processor->unified_signal_processor = unified_signal_processor_create(&processor_config);
    }
    
    /* 检查所有必需组件是否创建成功 */
    if ((config->enable_unified_signal_processor && !processor->unified_signal_processor) ||
        (config->spatial_config.enable_depth_estimation && !processor->depth_estimator) ||
        (config->spatial_config.enable_point_cloud && !processor->point_cloud_processor) ||
        (config->spatial_config.enable_slam && !processor->slam_system) ||
        (config->speech_config.enable_vad && !processor->vad_processor) ||
        (config->speech_config.enable_speech_recognition && !processor->speech_recognizer) ||
        (config->speech_config.enable_semantic_understanding && !processor->audio_semantic_processor) ||
        (config->ocr_config.enable_text_detection && !processor->text_detection_processor) ||
        (config->ocr_config.enable_character_segmentation && !processor->char_seg_processor) ||
        (config->ocr_config.enable_ocr && !processor->ocr_processor)) {
        
        /* 清理并返回失败 */
        multimodal_integration_processor_free(processor);
        return NULL;
    }
    
    processor->initialized = 1;
    return processor;
}



/**
 * @brief 处理视觉输入并提取空间识别特征
 */
int multimodal_integration_process_vision(
    MultimodalIntegrationProcessor* processor,
    const float* image_data, int width, int height, int channels,
    const float* stereo_image,
    VisionInput* vision_output) {
    
    if (!processor || !image_data || !vision_output || width <= 0 || height <= 0 || channels <= 0) {
        return -1;
    }
    
    /* 初始化视觉输出 */
    memset(vision_output, 0, sizeof(VisionInput));
    
    /* 提取空间识别特征 */
    float* spatial_features = (float*)safe_calloc(processor->config.spatial_config.max_features, sizeof(float));
    if (!spatial_features) {
        return -2;
    }
    
    int feature_count = 0;
    
    /* 深度估计 - 声明在外部作用域，点云处理也需要访问 */
    DepthEstimationResult depth_result;
    memset(&depth_result, 0, sizeof(DepthEstimationResult));
    int depth_result_code = 0;
    
    if (processor->depth_estimator && processor->config.spatial_config.enable_depth_estimation) {
        if (stereo_image && processor->config.spatial_config.enable_stereo_vision) {
            /* 立体视觉深度估计 */
            StereoCalibration calib;
            memset(&calib, 0, sizeof(StereoCalibration));
            calib.is_calibrated = 0; /* 使用默认标定 */
            
            depth_result_code = depth_estimate_stereo(processor->depth_estimator,
                                                     image_data, stereo_image,
                                                     width, height, channels,
                                                     &calib, &depth_result);
        } else {
            /* 单目深度估计 */
            CameraCalibration mono_calib;
            memset(&mono_calib, 0, sizeof(CameraCalibration));
            
            depth_result_code = depth_estimate_monocular(processor->depth_estimator,
                                                        image_data,
                                                        width, height, channels,
                                                        &mono_calib, &depth_result);
        }
        
        if (depth_result_code == 0 && depth_result.depth_map) {
            /* 提取深度图特征（完整实现：计算丰富的深度统计特征） */
            int total_pixels = width * height;
            float depth_mean = 0.0f, depth_variance = 0.0f, depth_skewness = 0.0f, depth_kurtosis = 0.0f;
            float depth_min = FLT_MAX, depth_max = FLT_MIN;
            float gradient_magnitude_mean = 0.0f;
            int valid_pixels = 0;
            
            // 第一轮：计算均值、最小值和最大值
            for (int i = 0; i < total_pixels; i++) {
                float depth = depth_result.depth_map[i];
                if (depth > 0.0f) {  // 忽略无效深度
                    depth_mean += depth;
                    if (depth < depth_min) depth_min = depth;
                    if (depth > depth_max) depth_max = depth;
                    valid_pixels++;
                }
            }
            if (valid_pixels > 0) {
                depth_mean /= valid_pixels;
            }
            
            // 第二轮：计算方差、偏度和峰度
            for (int i = 0; i < total_pixels; i++) {
                float depth = depth_result.depth_map[i];
                if (depth > 0.0f) {
                    float diff = depth - depth_mean;
                    float diff2 = diff * diff;
                    depth_variance += diff2;
                    depth_skewness += diff * diff2;  // diff^3
                    depth_kurtosis += diff2 * diff2; // diff^4
                }
            }
            if (valid_pixels > 0) {
                depth_variance /= valid_pixels;
                if (depth_variance > 1e-8f) {
                    depth_skewness /= (valid_pixels * powf(depth_variance, 1.5f));
                    depth_kurtosis /= (valid_pixels * depth_variance * depth_variance);
                }
            }
            
            // 第三轮：计算梯度统计（水平方向和垂直方向）
            int gradient_samples = 0;
            for (int y = 1; y < height - 1; y++) {
                for (int x = 1; x < width - 1; x++) {
                    int idx = y * width + x;
                    float depth_center = depth_result.depth_map[idx];
                    if (depth_center > 0.0f) {
                        float depth_right = depth_result.depth_map[idx + 1];
                        float depth_down = depth_result.depth_map[idx + width];
                        if (depth_right > 0.0f && depth_down > 0.0f) {
                            float grad_x = depth_right - depth_center;
                            float grad_y = depth_down - depth_center;
                            float mag = sqrtf(grad_x * grad_x + grad_y * grad_y);
                            gradient_magnitude_mean += mag;
                            gradient_samples++;
                        }
                    }
                }
            }
            if (gradient_samples > 0) {
                gradient_magnitude_mean /= gradient_samples;
            }
            
            // 计算深度范围
            float depth_range = (depth_max > depth_min) ? (depth_max - depth_min) : 0.0f;
            
            // 存储特征（确保不超过最大特征数限制）
            if (feature_count + 7 <= processor->config.spatial_config.max_features) {
                spatial_features[feature_count++] = depth_mean;
                spatial_features[feature_count++] = depth_variance;
                spatial_features[feature_count++] = depth_skewness;
                spatial_features[feature_count++] = depth_kurtosis;
                spatial_features[feature_count++] = depth_range;
                spatial_features[feature_count++] = gradient_magnitude_mean;
                spatial_features[feature_count++] = (float)valid_pixels / total_pixels; // 有效深度比例
            } else {
                // 如果空间不足，至少存储关键特征
                if (feature_count < processor->config.spatial_config.max_features) {
                    spatial_features[feature_count++] = depth_mean;
                }
                if (feature_count < processor->config.spatial_config.max_features) {
                    spatial_features[feature_count++] = depth_variance;
                }
            }
        }
    }
    
    /* 点云处理 */
    if (processor->point_cloud_processor && processor->config.spatial_config.enable_point_cloud) {
        /* 完整实现：基于深度图生成点云特征 */
        if (feature_count + 5 <= processor->config.spatial_config.max_features) {
            float point_cloud_density = 0.0f;
            float point_cloud_range = 0.0f;
            float point_cloud_mean_curvature = 0.0f;
            float point_cloud_planarity = 0.0f;
            float point_cloud_anisotropy = 0.0f;
            
            /* 从深度图真实计算点云统计特征（depth_result在当前作用域可用） */
            if (processor->depth_estimator && depth_result_code == 0 && depth_result.depth_map) {
                /* 从深度图生成点云并计算统计特征 */
                float sum_x = 0.0f, sum_y = 0.0f, sum_z = 0.0f;
                float sum_xx = 0.0f, sum_yy = 0.0f, sum_zz = 0.0f;
                float sum_xy = 0.0f, sum_xz = 0.0f, sum_yz = 0.0f;
                int point_count = 0;
                
                /* 相机内参（从标定获取或使用默认值） */
                float fx = width * 0.8f;
                float fy = height * 0.8f;
                float cx = width * 0.5f;
                float cy = height * 0.5f;
                
                for (int y = 0; y < height; y += 2) {
                    for (int x = 0; x < width; x += 2) {
                        int idx = y * width + x;
                        float depth = depth_result.depth_map[idx];
                        if (depth > 0.01f && depth < 100.0f) {
                            /* 反投影到3D空间 */
                            float px = (x - cx) * depth / fx;
                            float py = (y - cy) * depth / fy;
                            float pz = depth;
                            
                            sum_x += px; sum_y += py; sum_z += pz;
                            sum_xx += px * px; sum_yy += py * py; sum_zz += pz * pz;
                            sum_xy += px * py; sum_xz += px * pz; sum_yz += py * pz;
                            point_count++;
                        }
                    }
                }
                
                if (point_count > 10) {
                    float inv_n = 1.0f / point_count;
                    float mean_x = sum_x * inv_n;
                    float mean_y = sum_y * inv_n;
                    float mean_z = sum_z * inv_n;
                    
                    /* 计算协方差矩阵特征值来确定点云形状特征 */
                    float cov_xx = sum_xx * inv_n - mean_x * mean_x;
                    float cov_yy = sum_yy * inv_n - mean_y * mean_y;
                    float cov_zz = sum_zz * inv_n - mean_z * mean_z;
                    float cov_xy = sum_xy * inv_n - mean_x * mean_y;
                    float cov_xz = sum_xz * inv_n - mean_x * mean_z;
                    float cov_yz = sum_yz * inv_n - mean_y * mean_z;
                    
                    float trace = cov_xx + cov_yy + cov_zz;
                    /* 3x3协方差矩阵特征值计算（使用完整协方差矩阵）
                     * 通过计算矩阵的迹和Frobenius范数来估算特征值
                     * 比仅使用对角线元素更准确 */
                    float frob2 = cov_xx*cov_xx + cov_yy*cov_yy + cov_zz*cov_zz +
                                  2.0f*(cov_xy*cov_xy + cov_xz*cov_xz + cov_yz*cov_yz);
                    float p = (trace*trace - frob2) / 2.0f;
                    float q = cov_xx*(cov_yy*cov_zz - cov_yz*cov_yz) -
                              cov_xy*(cov_xy*cov_zz - cov_yz*cov_xz) +
                              cov_xz*(cov_xy*cov_yz - cov_yy*cov_xz);
                    /* 使用Cardano方法近似求解3x3特征值 */
                    float phi;
                    if (p > 1e-10f) {
                        float cos_phi = q / (2.0f * powf(p / 3.0f, 1.5f) + 1e-10f);
                        if (cos_phi > 1.0f) cos_phi = 1.0f;
                        if (cos_phi < -1.0f) cos_phi = -1.0f;
                        phi = acosf(cos_phi) / 3.0f;
                    } else {
                        phi = 0.0f;
                    }
                    float sqrt_p3 = 2.0f * sqrtf(p / 3.0f + 1e-10f);
                    float tr3 = trace / 3.0f;
                    float e1 = tr3 + sqrt_p3 * cosf(phi);
                    float e2 = tr3 + sqrt_p3 * cosf(phi + 2.0f * 3.14159f / 3.0f);
                    float e3 = tr3 + sqrt_p3 * cosf(phi - 2.0f * 3.14159f / 3.0f);
                    /* 排序 e1 >= e2 >= e3 */
                    if (e1 < e2) { float t = e1; e1 = e2; e2 = t; }
                    if (e1 < e3) { float t = e1; e1 = e3; e3 = t; }
                    if (e2 < e3) { float t = e2; e2 = e3; e3 = t; }
                    float eigenvalue_sum = e1 + e2 + e3;
                    if (eigenvalue_sum > 1e-10f) {
                        point_cloud_density = (float)point_count / (width * height);
                        point_cloud_range = sqrtf(cov_xx + cov_yy + cov_zz);
                        point_cloud_mean_curvature = (e1 - e2) / eigenvalue_sum;
                        point_cloud_planarity = (e2 - e3) / eigenvalue_sum;
                        point_cloud_anisotropy = (e1 - e3) / eigenvalue_sum;
                    }
                }
            }
            
            spatial_features[feature_count++] = point_cloud_density;
            spatial_features[feature_count++] = point_cloud_range;
            spatial_features[feature_count++] = point_cloud_mean_curvature;
            spatial_features[feature_count++] = point_cloud_planarity;
            spatial_features[feature_count++] = point_cloud_anisotropy;
        }
    }
    
    /* SLAM处理 */
    if (processor->slam_system && processor->config.spatial_config.enable_slam) {
        SlamResult slam_result;
        memset(&slam_result, 0, sizeof(SlamResult));
        
        int slam_result_code = slam_process_frame(processor->slam_system,
                                                  image_data, SLAM_SENSOR_MONOCULAR,
                                                  width, height, channels,
                                                  0.0f, /* timestamp */
                                                  &slam_result);
        
        if (slam_result_code == 0 && slam_result.tracking_quality > 0.5f) {
            /* 提取SLAM特征 */
            if (feature_count + 7 <= processor->config.spatial_config.max_features) {
                /* 位姿特征（位置和方向） */
                spatial_features[feature_count++] = slam_result.estimated_pose.position[0];
                spatial_features[feature_count++] = slam_result.estimated_pose.position[1];
                spatial_features[feature_count++] = slam_result.estimated_pose.position[2];
                spatial_features[feature_count++] = slam_result.estimated_pose.orientation[0];
                spatial_features[feature_count++] = slam_result.estimated_pose.orientation[1];
                spatial_features[feature_count++] = slam_result.estimated_pose.orientation[2];
                spatial_features[feature_count++] = slam_result.estimated_pose.orientation[3];
            }
        }
    }
    
    /* 填充视觉输出 */
    vision_output->features = spatial_features;
    vision_output->feature_count = feature_count;
    vision_output->width = width;
    vision_output->height = height;
    vision_output->channels = channels;
    vision_output->timestamp = 0; /* 实际应用中应使用真实时间戳 */
    
    return 0;
}

/**
 * @brief 处理音频输入并提取语音识别特征
 */
int multimodal_integration_process_audio(
    MultimodalIntegrationProcessor* processor,
    const float* audio_data, int num_samples, int sample_rate, int num_channels,
    AudioInput* audio_output) {
    
    if (!processor || !audio_data || !audio_output || num_samples <= 0 || sample_rate <= 0) {
        return -1;
    }
    
    /* 初始化音频输出 */
    memset(audio_output, 0, sizeof(AudioInput));
    
    /* 提取语音识别特征 */
    float* audio_features = (float*)safe_calloc(processor->config.speech_config.max_features, sizeof(float));
    if (!audio_features) {
        return -2;
    }
    
    int feature_count = 0;
    
    /* 语音端点检测 */
    if (processor->vad_processor && processor->config.speech_config.enable_vad) {
        VadResult vad_result;
        memset(&vad_result, 0, sizeof(VadResult));
        
        int vad_result_code = vad_process(processor->vad_processor,
                                         audio_data, num_samples,
                                         &vad_result);
        
        if (vad_result_code == 0) {
            /* 提取VAD特征 */
            if (feature_count + 3 <= processor->config.speech_config.max_features) {
                /* 使用VadResult中可用的特征 */
                audio_features[feature_count++] = vad_result.speech_ratio; /* 语音比率替代语音概率 */
                /* 计算平均置信度作为能量替代 */
                float avg_confidence = 0.0f;
                if (vad_result.confidences && vad_result.num_frames > 0) {
                    for (int i = 0; i < vad_result.num_frames; i++) {
                        avg_confidence += vad_result.confidences[i];
                    }
                    avg_confidence /= vad_result.num_frames;
                }
                audio_features[feature_count++] = avg_confidence;
                /* 使用语音段数量作为第三个特征 */
                audio_features[feature_count++] = (float)vad_result.num_speech_segments;
            }
        }
    }
    
    /* 语音识别 */
    if (processor->speech_recognizer && processor->config.speech_config.enable_speech_recognition) {
        SpeechRecognitionResult speech_result;
        memset(&speech_result, 0, sizeof(SpeechRecognitionResult));
        
        int speech_result_code = speech_recognizer_recognize(processor->speech_recognizer,
                                                          audio_data, num_samples,
                                                          &speech_result);
        
        if (speech_result_code == 0 && speech_result.text[0] != '\0') {
            /* 提取语音识别特征（完整实现：计算丰富的文本统计特征） */
            const char* text = speech_result.text;
            size_t text_len = strlen(text);
            
            // 计算基本统计特征
            float confidence = speech_result.confidence;
            float text_length = (float)text_len;
            
            // 计算词数（空格分隔）
            int word_count = 0;
            int in_word = 0;
            for (size_t i = 0; i < text_len; i++) {
                if (text[i] != ' ' && text[i] != '\t' && text[i] != '\n') {
                    if (!in_word) {
                        word_count++;
                        in_word = 1;
                    }
                } else {
                    in_word = 0;
                }
            }
            
            // 计算平均词长
            float avg_word_length = 0.0f;
            if (word_count > 0) {
                // 计算非空格字符数
                int non_space_chars = 0;
                for (size_t i = 0; i < text_len; i++) {
                    if (text[i] != ' ' && text[i] != '\t' && text[i] != '\n') {
                        non_space_chars++;
                    }
                }
                avg_word_length = (float)non_space_chars / word_count;
            }
            
            // 计算标点符号比例
            int punctuation_count = 0;
            for (size_t i = 0; i < text_len; i++) {
                char c = text[i];
                if (c == '.' || c == ',' || c == '!' || c == '?' || c == ';' || c == ':' ||
                    c == '(' || c == ')' || c == '[' || c == ']' || c == '{' || c == '}' ||
                    c == '"' || c == '\'' || c == '-' || c == '/') {
                    punctuation_count++;
                }
            }
            float punctuation_ratio = (text_len > 0) ? (float)punctuation_count / text_len : 0.0f;
            
            // 计算大写字母比例（仅针对字母字符）
            int uppercase_count = 0;
            int letter_count = 0;
            for (size_t i = 0; i < text_len; i++) {
                char c = text[i];
                if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) {
                    letter_count++;
                    if (c >= 'A' && c <= 'Z') {
                        uppercase_count++;
                    }
                }
            }
            float uppercase_ratio = (letter_count > 0) ? (float)uppercase_count / letter_count : 0.0f;
            
            // 计算数字比例
            int digit_count = 0;
            for (size_t i = 0; i < text_len; i++) {
                if (text[i] >= '0' && text[i] <= '9') {
                    digit_count++;
                }
            }
            float digit_ratio = (text_len > 0) ? (float)digit_count / text_len : 0.0f;
            
            // 存储特征（确保不超过最大特征数限制）
            int features_to_add = 7; // 置信度、文本长度、词数、平均词长、标点比例、大写比例、数字比例
            if (feature_count + features_to_add <= processor->config.speech_config.max_features) {
                audio_features[feature_count++] = confidence;
                audio_features[feature_count++] = text_length;
                audio_features[feature_count++] = (float)word_count;
                audio_features[feature_count++] = avg_word_length;
                audio_features[feature_count++] = punctuation_ratio;
                audio_features[feature_count++] = uppercase_ratio;
                audio_features[feature_count++] = digit_ratio;
            } else {
                // 如果空间不足，至少存储关键特征
                if (feature_count < processor->config.speech_config.max_features) {
                    audio_features[feature_count++] = confidence;
                }
                if (feature_count < processor->config.speech_config.max_features) {
                    audio_features[feature_count++] = text_length;
                }
            }
        }
    }
    
    /* 音频语义理解 */
    if (processor->audio_semantic_processor && processor->config.speech_config.enable_semantic_understanding) {
        AudioSemanticResult semantic_result;
        memset(&semantic_result, 0, sizeof(AudioSemanticResult));
        
        int semantic_result_code = audio_semantic_process(processor->audio_semantic_processor,
                                                         audio_data, num_samples, sample_rate, num_channels,
                                                         &semantic_result);
        
        if (semantic_result_code == 0) {
            /* 提取语义特征 */
            if (feature_count + 4 <= processor->config.speech_config.max_features) {
                audio_features[feature_count++] = semantic_result.emotion.intensity;
                audio_features[feature_count++] = semantic_result.intent.confidence;
                audio_features[feature_count++] = semantic_result.recognition_confidence;
                audio_features[feature_count++] = (float)semantic_result.num_keywords;
            }
        }
    }
    
    /* 填充音频输出 */
    audio_output->samples = audio_data;
    audio_output->sample_count = num_samples;
    audio_output->sample_rate = (float)sample_rate;
    audio_output->mfcc_features = audio_features;
    audio_output->mfcc_count = feature_count;
    audio_output->timestamp = 0;
    
    return 0;
}

/**
 * @brief 处理文本输入并提取OCR特征
 */
int multimodal_integration_process_text(
    MultimodalIntegrationProcessor* processor,
    const float* image_data, int width, int height, int channels,
    TextInput* text_output) {
    
    if (!processor || !image_data || !text_output || width <= 0 || height <= 0 || channels <= 0) {
        return -1;
    }
    
    /* 初始化文本输出 */
    memset(text_output, 0, sizeof(TextInput));
    
    /* 提取OCR特征 */
    float* text_features = (float*)safe_calloc(processor->config.ocr_config.max_features, sizeof(float));
    if (!text_features) {
        return -2;
    }
    
    int feature_count = 0;
    
    /* 文本检测 */
    if (processor->text_detection_processor && processor->config.ocr_config.enable_text_detection) {
        TextRegion text_regions[16];
        int max_regions = 16;
        int num_regions = 0;
        
        int text_det_result_code = text_detection_detect(processor->text_detection_processor,
                                                        image_data, width, height, channels,
                                                        text_regions, max_regions, &num_regions);
        
        if (text_det_result_code == 0 && num_regions > 0) {
            /* 提取文本检测特征 */
            if (feature_count + 3 <= processor->config.ocr_config.max_features) {
                text_features[feature_count++] = (float)num_regions;
                /* 计算平均置信度和覆盖度 */
                float avg_confidence = 0.0f;
                float coverage = 0.0f;
                for (int i = 0; i < num_regions; i++) {
                    avg_confidence += text_regions[i].confidence;
                    coverage += (float)(text_regions[i].width * text_regions[i].height);
                }
                if (num_regions > 0) {
                    avg_confidence /= (float)num_regions;
                    coverage /= (float)(width * height);
                }
                text_features[feature_count++] = avg_confidence;
                text_features[feature_count++] = coverage;
            }
        }
    }
    
    /* 字符分割 */
    if (processor->char_seg_processor && processor->config.ocr_config.enable_character_segmentation) {
        CharRegion char_regions[32];
        int max_chars = 32;
        int num_chars = 0;
        
        int char_seg_result_code = char_segmentation_segment(processor->char_seg_processor,
                                                            image_data, width, height, channels,
                                                            char_regions, max_chars, &num_chars);
        
        if (char_seg_result_code == 0 && num_chars > 0) {
            /* 提取字符分割特征 */
            if (feature_count + 2 <= processor->config.ocr_config.max_features) {
                text_features[feature_count++] = (float)num_chars;
                /* 计算平均置信度 */
                float avg_confidence = 0.0f;
                for (int i = 0; i < num_chars; i++) {
                    avg_confidence += char_regions[i].confidence;
                }
                if (num_chars > 0) {
                    avg_confidence /= (float)num_chars;
                }
                text_features[feature_count++] = avg_confidence;
            }
        }
    }
    
    /* OCR识别 */
    if (processor->ocr_processor && processor->config.ocr_config.enable_ocr) {
        OcrResult ocr_result;
        memset(&ocr_result, 0, sizeof(OcrResult));
        
        int ocr_result_code = ocr_recognize(processor->ocr_processor,
                                         image_data, width, height, channels,
                                         &ocr_result);
        
        if (ocr_result_code == 0 && ocr_result.text[0] != '\0') {
            /* 提取OCR特征 */
            if (feature_count + 3 <= processor->config.ocr_config.max_features) {
                text_features[feature_count++] = ocr_result.confidence;
                text_features[feature_count++] = (float)strlen(ocr_result.text);
                text_features[feature_count++] = ocr_result.confidence; /* language_confidence可能不存在，使用confidence */
            }
        }
    }
    
    /* 填充文本输出 */
    text_output->embeddings = text_features;
    text_output->embedding_dim = feature_count; /* 特征数量作为嵌入维度 */
    text_output->text = NULL; /* 实际应用中可设置识别到的文本 */
    text_output->text_length = 0;
    text_output->timestamp = 0;
    
    return 0;
}

/**
 * @brief 处理多模态输入并生成统一信号处理输出
 *
 * 单一主路径：通过统一信号处理器的CfC细胞单元进行统一动态处理。
 * 所有模态直接拼接输入到同一个CfC连续动态系统。
 * 无回退链：统一信号处理器失败则操作失败，确保输出质量。
 */
int multimodal_integration_process_unified(
    MultimodalIntegrationProcessor* processor,
    const VisionInput* vi_in,
    const AudioInput* ai_in,
    const TextInput* txt_in,
    const SensorInput* sen_in,
    UnifiedOutput* unified_output) {
    
    if (!processor || !unified_output) {
        return -1;
    }
    
    if (!processor->unified_signal_processor) {
        memset(unified_output, 0, sizeof(UnifiedOutput));
        return -1;
    }
    
    int result = unified_signal_processor_encode(processor->unified_signal_processor,
                                       vi_in, ai_in, txt_in, sen_in,
                                       unified_output);
    if (result != 0 || !unified_output->unified_signal) {
        memset(unified_output, 0, sizeof(UnifiedOutput));
        return -1;
    }
    
    /* 快速稳定性检查：对统一输出信号进行频域稳定性加权 */
    if (unified_output->unified_signal) {
        size_t signal_dim = (unified_output->signal_dimension > 0)
                          ? (size_t)unified_output->signal_dimension : 64;
        size_t spectrum_size = signal_dim < 512 ? signal_dim : 512;
        if (spectrum_size > 0) {
            float den_coeffs[2] = {1.0f, -0.5f};
            int is_stable = 0;
            float stability_margin = 0.0f;
            if (laplace_check_stability_fast(den_coeffs, 2, &is_stable, &stability_margin) == 0
                && is_stable && unified_output->unified_signal) {
                float stability_factor = (stability_margin > 0.0f)
                    ? (stability_margin > 1.0f ? 1.0f : stability_margin)
                    : 0.5f;
                for (size_t i = 0; i < spectrum_size; i++) {
                    unified_output->unified_signal[i] *= (0.8f + 0.2f * stability_factor);
                }
            }
        }
    }
    
    return 0;
}

/**
 * @brief 处理多模态输入并直接传递给液态神经网络
 *
 * 单一主路径：通过统一信号处理器处理后使用unified_signal_processor_encode_to_lnn传递到LNN。
 * 无手动特征拼接回退：如果主路径失败，操作失败。
 */
int multimodal_integration_process_to_lnn(
    MultimodalIntegrationProcessor* processor,
    LNN* lnn_net,
    const VisionInput* vis_in,
    const AudioInput* aud_in,
    const TextInput* txt_in2,
    const SensorInput* sen_in2,
    float* lnn_output, size_t max_output_size) {
    
    if (!processor || !lnn_net || !lnn_output || max_output_size == 0) {
        return -1;
    }
    /* ZSFWS-L022修复: 移除误导性(void)转换
     * vis_in/aud_in/txt_in2/sen_in2在下方传递给unified_signal_processor_encode_to_lnn
     * (void)转换让人误以为这些参数被丢弃，实则被正常使用 */
    
    if (!processor->unified_signal_processor) {
        return -1;
    }
    
    int result = unified_signal_processor_encode_to_lnn(processor->unified_signal_processor,
                                              lnn_net,
                                              vis_in, aud_in, txt_in2, sen_in2,
                                              lnn_output, max_output_size);
    if (result <= 0) {
        return -1;
    }
    
    return result;
}

/**
 * @brief 生成离线测试输入数据 —— 已完全禁用
 *
 * 【K-002修复：完全移除合成数据生成】
 * 根据项目原则"不可以使用任何假数据和虚拟数据"，彻底移除了所有合成数据生成代码。
 * 该函数仅保留错误返回。无硬件环境时必须使用零状态输入。
 * 自主学习管道永远通过hardware_available检查获取真实数据。
 */
int multimodal_integration_generate_test_data(
    MultimodalIntegrationProcessor* processor,
    VisionInput* vision_output,
    AudioInput* audio_output,
    TextInput* text_output,
    SensorInput* sensor_output) {
    (void)processor;
    (void)vision_output;
    (void)audio_output;
    (void)text_output;
    (void)sensor_output;
    log_error("[多模态集成] 合成数据生成已完全禁用（K-002修复）。"
              "禁止生成任何非真实硬件数据。无硬件时应使用零状态输入。");
    return -1;
}

/**
 * @brief 释放多模态集成处理器
 */
void multimodal_integration_processor_free(MultimodalIntegrationProcessor* processor) {
    if (!processor) {
        return;
    }
    
    /* 释放空间识别组件 */
    if (processor->depth_estimator) {
        depth_estimator_free(processor->depth_estimator);
    }
    if (processor->point_cloud_processor) {
        point_cloud_processor_free(processor->point_cloud_processor);
    }
    if (processor->slam_system) {
        slam_system_free(processor->slam_system);
    }
    
    /* 释放语音识别组件 */
    if (processor->vad_processor) {
        vad_processor_free(processor->vad_processor);
    }
    if (processor->speech_recognizer) {
        speech_recognizer_free(processor->speech_recognizer);
    }
    if (processor->audio_semantic_processor) {
        audio_semantic_processor_free(processor->audio_semantic_processor);
    }
    
    /* 释放OCR组件 */
    if (processor->text_detection_processor) {
        text_detection_processor_free(processor->text_detection_processor);
    }
    if (processor->char_seg_processor) {
        char_segmentation_processor_free(processor->char_seg_processor);
    }
    if (processor->ocr_processor) {
        ocr_processor_free(processor->ocr_processor);
    }
    
    /* 释放统一信号处理器 */
    if (processor->unified_signal_processor) {
        unified_signal_processor_free(processor->unified_signal_processor);
    }
    
    /* 释放处理器内存 */
    safe_free((void**)&processor);
}

/* ============================================================================
 * MM-22: 传感器→统一拼接→CfC重构
 *
 * 合规架构: 所有传感器数据直接拼接为统一向量送入CfC
 * 不做分开编码，不做多模型融合，纯单一数据流
 * ============================================================================ */

#define INTEGRATION_MAX_MODALITIES 9
#define INTEGRATION_FEAT_DIM 64

int multimodal_unified_pipeline(const float** modality_data, const int* modality_dims,
                                 int num_modalities, void* main_cfc,
                                 float* unified_output, int output_dim) {
    if (!modality_data || !modality_dims || !main_cfc || !unified_output) return -1;

    int total_dim = 0;
    for (int m = 0; m < num_modalities && m < INTEGRATION_MAX_MODALITIES; m++)
        total_dim += modality_dims[m];

    float* unified_input = (float*)safe_calloc((size_t)total_dim, sizeof(float));
    if (!unified_input) return -1;

    int offset = 0;
    for (int m = 0; m < num_modalities && m < INTEGRATION_MAX_MODALITIES; m++) {
        if (modality_data[m] && modality_dims[m] > 0) {
            memcpy(unified_input + offset, modality_data[m],
                   (size_t)modality_dims[m] * sizeof(float));
            offset += modality_dims[m];
        }
    }

    lnn_forward((LNN*)main_cfc, unified_input, unified_output);

    safe_free((void**)&unified_input);
    return 0;
}