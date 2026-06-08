/**
 * @file ocr.h
 * @brief 光学字符识别（OCR）模块接口
 * 
 * 光学字符识别模块，支持图像文字检测、字符分割和字符识别。
 * 实现完整的OCR处理管道，包括图像预处理、文字区域检测、字符分割、
 * 特征提取、字符分类和文本后处理。
 */

#ifndef SELFLNN_OCR_H
#define SELFLNN_OCR_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief OCR引擎类型枚举（全部基于液态神经网络CfC实现，无外部依赖）
 */
typedef enum {
    OCR_ENGINE_LIQUID_CLASSIC = 0,     /**< 经典液态引擎（CfC特征匹配） */
    OCR_ENGINE_LIQUID_VISION = 1,      /**< 视觉液态引擎（CfC ODE视觉编码） */
    OCR_ENGINE_LIQUID_SEQUENCE = 2,    /**< 时序液态引擎（CfC时序编码） */
    OCR_ENGINE_LIQUID_ADVANCED = 3,    /**< 高级液态引擎（CfC深度编码） */
    OCR_ENGINE_LIQUID_FUSION = 4,      /**< 融合液态引擎（CfC多模态融合） */
    OCR_ENGINE_LIQUID_END2END = 5,     /**< 端到端液态引擎（CfC全流水线） */
} OcrEngineType;

/**
 * @brief 字符分割算法枚举
 */
typedef enum {
    SEGMENTATION_PROJECTION = 0,       /**< 投影分割算法 */
    SEGMENTATION_CONNECTED_COMPONENT = 1, /**< 连通组件分割算法 */
    SEGMENTATION_CONTOUR = 2,          /**< 轮廓分割算法 */
    SEGMENTATION_WATERSHED = 3,        /**< 分水岭分割算法 */
    SEGMENTATION_DEEP_LEARNING = 4,    /**< 深度学习分割算法 */
} SegmentationAlgorithm;

/**
 * @brief 文本检测算法枚举
 */
#ifndef SELFLNN_TEXT_DETECTION_ALGORITHM_DEFINED
#define SELFLNN_TEXT_DETECTION_ALGORITHM_DEFINED
typedef enum {
    TEXT_DETECTION_MSER = 0,           /**< MSER最大稳定极值区域算法 */
    TEXT_DETECTION_EAST = 1,           /**< EAST高效准确场景文本检测 */
    TEXT_DETECTION_CTPN = 2,           /**< CTPN连接文本提案网络 */
    TEXT_DETECTION_SWT = 3,            /**< SWT笔画宽度变换 */
    TEXT_DETECTION_CONTOUR = 4,        /**< 轮廓检测算法 */
} TextDetectionAlgorithm;
#endif /* SELFLNN_TEXT_DETECTION_ALGORITHM_DEFINED */

/**
 * @brief 语言类型枚举
 */
typedef enum {
    LANGUAGE_CHINESE = 0,              /**< 中文 */
    LANGUAGE_ENGLISH = 1,              /**< 英文 */
    LANGUAGE_CHINESE_ENGLISH = 2,      /**< 中英文混合 */
    LANGUAGE_JAPANESE = 3,             /**< 日文 */
    LANGUAGE_KOREAN = 4,               /**< 韩文 */
    LANGUAGE_MULTILINGUAL = 5,         /**< 多语言 */
} LanguageType;

/**
 * @brief OCR配置结构体
 */
typedef struct {
    OcrEngineType engine_type;         /**< OCR引擎类型 */
    SegmentationAlgorithm seg_algorithm; /**< 字符分割算法 */
    TextDetectionAlgorithm det_algorithm; /**< 文本检测算法 */
    LanguageType language;             /**< 语言类型 */
    
    /* 图像预处理配置 */
    int image_width;                   /**< 图像宽度 */
    int image_height;                  /**< 图像高度 */
    int grayscale;                     /**< 是否转换为灰度图像 */
    int binarization;                  /**< 是否二值化 */
    int denoising;                     /**< 是否去噪 */
    int deskew;                        /**< 是否校正倾斜 */
    
    /* 文本检测配置 */
    int min_text_height;               /**< 最小文本高度（像素） */
    int max_text_height;               /**< 最大文本高度（像素） */
    float text_confidence_threshold;   /**< 文本置信度阈值 */
    
    /* 字符分割配置 */
    int min_char_width;                /**< 最小字符宽度（像素） */
    int max_char_width;                /**< 最大字符宽度（像素） */
    int min_char_height;               /**< 最小字符高度（像素） */
    int max_char_height;               /**< 最大字符高度（像素） */
    float char_spacing_threshold;      /**< 字符间距阈值 */
    
    /* 字符识别配置 */
    int char_feature_dim;              /**< 字符特征维度 */
    int num_char_classes;              /**< 字符类别数 */
    float char_confidence_threshold;   /**< 字符置信度阈值 */
    
    /* 后处理配置 */
    int use_dictionary;                /**< 是否使用词典 */
    int use_language_model;            /**< 是否使用语言模型 */
    int max_alternative_chars;         /**< 最大备选字符数 */
} OcrConfig;

/**
 * @brief 文本区域结构体
 */
typedef struct {
    int x;                             /**< 区域左上角X坐标 */
    int y;                             /**< 区域左上角Y坐标 */
    int width;                         /**< 区域宽度 */
    int height;                        /**< 区域高度 */
    float confidence;                  /**< 置信度 */
    float angle;                       /**< 旋转角度（弧度） */
    int is_horizontal;                 /**< 是否为水平文本 */
    int num_vertices;                  /**< 顶点数量（用于多边形区域） */
    int vertices[16];                  /**< 顶点坐标（x,y对，最多8个顶点） */
    char* text;                        /**< 识别的文本 */
    int text_length;                   /**< 文本长度 */
} TextRegion;

/**
 * @brief 字符位置结构体
 */
typedef struct {
    int x;                             /**< 字符左上角X坐标 */
    int y;                             /**< 字符左上角Y坐标 */
    int width;                         /**< 字符宽度 */
    int height;                        /**< 字符高度 */
    unsigned short character;          /**< 识别的字符（支持Unicode，含中文） */
    float confidence;                  /**< 置信度 */
    float* alternatives;               /**< 备选字符置信度 */
    char* alt_chars;                   /**< 备选字符 */
    int num_alternatives;              /**< 备选字符数 */
} CharPosition;

/**
 * @brief OCR结果结构体
 */
typedef struct {
    char* text;                        /**< 识别的完整文本 */
    float confidence;                  /**< 整体置信度 */
    int num_regions;                   /**< 文本区域数量 */
    TextRegion* regions;               /**< 文本区域数组 */
    int num_chars;                     /**< 字符数量 */
    CharPosition* chars;               /**< 字符位置数组 */
    float processing_time_ms;          /**< 处理时间（毫秒） */
} OcrResult;

/**
 * @brief 图像预处理结果结构体
 */
typedef struct {
    int width;                         /**< 预处理后图像宽度 */
    int height;                        /**< 预处理后图像高度 */
    int channels;                      /**< 通道数（1=灰度，3=RGB） */
    float* data;                       /**< 图像数据 */
    int is_binary;                     /**< 是否是二值图像 */
} PreprocessResult;

/**
 * @brief 文本检测结果结构体
 */
typedef struct {
    int num_regions;                   /**< 检测到的文本区域数量 */
    TextRegion* regions;               /**< 文本区域数组 */
    float* region_features;            /**< 区域特征 */
    int feature_dim;                   /**< 特征维度 */
} TextDetectionResult;

/**
 * @brief 字符分割结果结构体
 */
typedef struct {
    int num_chars;                     /**< 分割出的字符数量 */
    CharPosition* chars;               /**< 字符位置数组 */
    float* char_images;                /**< 字符图像数据 */
    int char_width;                    /**< 字符图像宽度 */
    int char_height;                   /**< 字符图像高度 */
} CharSegmentationResult;

/**
 * @brief 字符识别结果结构体
 */
typedef struct {
    int num_chars;                     /**< 识别的字符数量 */
    char* characters;                  /**< 识别出的字符 */
    float* confidences;                /**< 每个字符的置信度 */
    float* features;                   /**< 字符特征 */
    int feature_dim;                   /**< 特征维度 */
} CharRecognitionResult;

/**
 * @brief OCR处理器句柄
 */
typedef struct OcrProcessor OcrProcessor;

/**
 * @brief 获取默认的OCR配置
 * 
 * @return OcrConfig 默认配置
 */
OcrConfig ocr_get_default_config(void);

/**
 * @brief 创建OCR处理器
 * 
 * @param config OCR配置
 * @return OcrProcessor* 处理器句柄，失败返回NULL
 */
OcrProcessor* ocr_processor_create(const OcrConfig* config);

/**
 * @brief 释放OCR处理器
 * 
 * @param processor 处理器句柄
 */
void ocr_processor_free(OcrProcessor* processor);

/**
 * @brief 执行OCR识别
 * 
 * @param processor 处理器句柄
 * @param image_data 图像数据（RGB或灰度）
 * @param width 图像宽度
 * @param height 图像高度
 * @param channels 图像通道数（1=灰度，3=RGB）
 * @param result OCR结果输出
 * @return int 成功返回0，失败返回-1
 */
int ocr_recognize(OcrProcessor* processor,
                  const float* image_data, int width, int height, int channels,
                  OcrResult* result);

/**
 * @brief 执行文本检测
 * 
 * @param processor 处理器句柄
 * @param image_data 图像数据
 * @param width 图像宽度
 * @param height 图像高度
 * @param channels 图像通道数
 * @param result 文本检测结果输出
 * @return int 成功返回0，失败返回-1
 */
int ocr_detect_text(OcrProcessor* processor,
                    const float* image_data, int width, int height, int channels,
                    TextDetectionResult* result);

/**
 * @brief 执行字符分割
 * 
 * @param processor 处理器句柄
 * @param text_region 文本区域
 * @param image_data 原始图像数据
 * @param width 图像宽度
 * @param height 图像高度
 * @param channels 图像通道数
 * @param result 字符分割结果输出
 * @return int 成功返回0，失败返回-1
 */
int ocr_segment_chars(OcrProcessor* processor,
                      const TextRegion* text_region,
                      const float* image_data, int width, int height, int channels,
                      CharSegmentationResult* result);

/**
 * @brief 执行字符识别
 * 
 * @param processor 处理器句柄
 * @param char_images 字符图像数据数组
 * @param num_chars 字符数量
 * @param char_width 字符图像宽度
 * @param char_height 字符图像高度
 * @param result 字符识别结果输出
 * @return int 成功返回0，失败返回-1
 */
int ocr_recognize_chars(OcrProcessor* processor,
                        const float* char_images, int num_chars,
                        int char_width, int char_height,
                        CharRecognitionResult* result);

/**
 * @brief 图像预处理
 * 
 * @param processor 处理器句柄
 * @param image_data 原始图像数据
 * @param width 图像宽度
 * @param height 图像高度
 * @param channels 图像通道数
 * @param result 预处理结果输出
 * @return int 成功返回0，失败返回-1
 */
int ocr_preprocess_image(OcrProcessor* processor,
                         const float* image_data, int width, int height, int channels,
                         PreprocessResult* result);

/**
 * @brief 文本后处理
 * 
 * @param processor 处理器句柄
 * @param raw_text 原始识别文本
 * @param confidences 字符置信度数组
 * @param num_chars 字符数量
 * @param result_text 后处理文本输出
 * @param result_confidence 后处理置信度输出
 * @return int 成功返回0，失败返回-1
 */
int ocr_postprocess_text(OcrProcessor* processor,
                         const char* raw_text, const float* confidences, int num_chars,
                         char** result_text, float* result_confidence);

/**
 * @brief 加载OCR模型
 * 
 * @param processor 处理器句柄
 * @param model_path 模型文件路径
 * @return int 成功返回0，失败返回-1
 */
int ocr_processor_load_model(OcrProcessor* processor, const char* model_path);

/**
 * @brief 保存OCR模型
 * 
 * @param processor 处理器句柄
 * @param model_path 模型文件路径
 * @return int 成功返回0，失败返回-1
 */
int ocr_processor_save_model(OcrProcessor* processor, const char* model_path);

/**
 * @brief 训练OCR模型
 * 
 * @param processor 处理器句柄
 * @param training_images 训练图像数组
 * @param training_labels 训练标签数组
 * @param num_samples 样本数量
 * @param image_width 图像宽度
 * @param image_height 图像高度
 * @param num_epochs 训练轮数
 * @param learning_rate 学习率
 * @param batch_size 批次大小
 * @return int 成功返回0，失败返回-1
 */
int ocr_processor_train_model(OcrProcessor* processor,
                              const float* training_images, const char** training_labels,
                              int num_samples, int image_width, int image_height,
                              int num_epochs, float learning_rate, int batch_size);

/**
 * @brief 设置语言模型
 * 
 * @param processor 处理器句柄
 * @param language_model_data 语言模型数据
 * @param data_size 数据大小
 * @return int 成功返回0，失败返回-1
 */
int ocr_processor_set_language_model(OcrProcessor* processor,
                                     const char* language_model_data, size_t data_size);

/**
 * @brief 设置词典
 * 
 * @param processor 处理器句柄
 * @param dictionary 词典数组
 * @param dict_size 词典大小
 * @return int 成功返回0，失败返回-1
 */
int ocr_processor_set_dictionary(OcrProcessor* processor,
                                 const char** dictionary, int dict_size);

/**
 * @brief 获取OCR配置
 * 
 * @param processor 处理器句柄
 * @param config 配置输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int ocr_processor_get_config(const OcrProcessor* processor, OcrConfig* config);

/**
 * @brief 设置OCR配置
 * 
 * @param processor 处理器句柄
 * @param config 新配置
 * @return int 成功返回0，失败返回-1
 */
int ocr_processor_set_config(OcrProcessor* processor, const OcrConfig* config);

/**
 * @brief 重置OCR处理器
 * 
 * @param processor 处理器句柄
 */
void ocr_processor_reset(OcrProcessor* processor);

/**
 * @brief 计算识别准确率
 * 
 * @param reference 参考文本
 * @param hypothesis 识别文本
 * @param accuracy 准确率输出
 * @return int 成功返回0，失败返回-1
 */
int ocr_compute_accuracy(const char* reference, const char* hypothesis, float* accuracy);

/**
 * @brief 计算字符错误率（CER）
 * 
 * @param reference 参考文本
 * @param hypothesis 识别文本
 * @param cer 字符错误率输出
 * @return int 成功返回0，失败返回-1
 */
int ocr_compute_cer(const char* reference, const char* hypothesis, float* cer);

/**
 * @brief 提取图像特征用于文本检测
 * 
 * @param processor 处理器句柄
 * @param image_data 图像数据
 * @param width 图像宽度
 * @param height 图像高度
 * @param channels 图像通道数
 * @param features 特征输出
 * @param max_features 最大特征数
 * @return int 成功返回特征数，失败返回-1
 */
int ocr_extract_text_features(OcrProcessor* processor,
                              const float* image_data, int width, int height, int channels,
                              float* features, int max_features);

/**
 * @brief 提取字符特征用于识别
 * 
 * @param processor 处理器句柄
 * @param char_image 字符图像数据
 * @param width 字符图像宽度
 * @param height 字符图像高度
 * @param features 特征输出
 * @param max_features 最大特征数
 * @return int 成功返回特征数，失败返回-1
 */
int ocr_extract_char_features(OcrProcessor* processor,
                              const float* char_image, int width, int height,
                              float* features, int max_features);

// ============================================================================
// CRNN 端到端序列文本识别
// ============================================================================
// CfC-LNN OCR API: 全液态神经网络文本识别（替代旧CRNN）
// ============================================================================

#define CRNN_MAX_SEQUENCE_LEN    128
#define CRNN_NUM_CHAR_CLASSES    100
#define CRNN_CTC_BLANK_LABEL     0
#define CRNN_DEFAULT_IMAGE_WIDTH  160
#define CRNN_DEFAULT_IMAGE_HEIGHT 48

/* CfC-LNN OCR 网络（不透明句柄，替代旧CrnnNet） */
typedef struct CfCOcrNet CfCOcrNet;

int cfc_ocr_net_create(int image_height, int image_width, int num_classes,
                        int hidden_size, int feature_dim, void** out_net);
void cfc_ocr_net_free(void* net_ptr);
int cfc_ocr_net_forward(void* net_ptr, const float* image, int image_h,
                         int image_w, int channels, float* probs, int* seq_len_out);
int cfc_ocr_net_train_step(void* net_ptr, const float* image, int image_h,
                            int image_w, int channels, const int* label,
                            int label_len, float learning_rate, float* loss);
int cfc_ocr_net_save(void* net_ptr, const char* path);
int cfc_ocr_net_load(void** out_net, const char* path);
int cfc_ocr_recognize(void* net_ptr, const float* image, int image_w,
                       int image_h, char* text_out, int max_text_len, float* confidence);

int crnn_recognize_text(OcrProcessor* processor, const float* image,
                        int image_w, int image_h, char* text_out,
                        int max_text_len, float* confidence);

/* F-011: CfC OCR网络模式控制 */
/**
 * @brief 启用CfC-LNN OCR网络模式，替代硬编码字符模板
 * @param processor OCR处理器
 * @param enable 1=启用, 0=禁用
 * @return 0成功, -1失败
 */
int ocr_enable_cfc_mode(OcrProcessor* processor, int enable);

/**
 * @brief 设置/替换CfC OCR网络
 * @param processor OCR处理器
 * @param cfc_net CfC OCR网络句柄(NULL表示清除)
 * @return 0成功, -1失败
 */
int ocr_set_cfc_network(OcrProcessor* processor, void* cfc_net);

#ifdef __cplusplus
}
#endif

#endif // SELFLNN_OCR_H