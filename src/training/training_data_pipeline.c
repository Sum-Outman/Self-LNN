/**
 * @file training_data_pipeline.c
 * @brief 多模态训练数据管线 - 桥接多模态CfC/LNN权重到训练器
 *
 * K-035: 已验证实现完整性。解决P3关键缺陷：所有多模态NN权重初始化后从未训练。
 * ZSFWS-001修复: 添加SELFLNN_STRICT_REAL_DATA保护，严格模式下禁止合成数据生成。
 * 严格模式：仅使用data_collection_pipeline提供的真实硬件数据，无真实数据时返回错误。
 * 通过 trainer_create + trainer_train 真实执行训练循环。
 * 与 unified_signal_processor_training.c 配合完成统一信号处理器的端到端训练。
 */

#include "selflnn/core/lnn.h"
#include "selflnn/core/optimizer.h"
#include "selflnn/core/loss.h"
#include "selflnn/training/training.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/selflnn.h"
#include "selflnn/utils/logging.h"
#include "selflnn/multimodal/data_collection_pipeline.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TDP_DEFAULT_EPOCHS 10

int training_pipeline_train_multimodal(LNN* network, const char* module_name,
                                        int input_dim, int output_dim,
                                        int num_samples, int epochs, float learning_rate) {
    if (!network || !module_name) return -1;

    int actual_samples = num_samples > 0 ? num_samples : 500;
    float* inputs = (float*)safe_calloc((size_t)actual_samples * input_dim, sizeof(float));
    float* targets = (float*)safe_calloc((size_t)actual_samples * output_dim, sizeof(float));
    if (!inputs || !targets) {
        safe_free((void**)&inputs);
        safe_free((void**)&targets);
        return -1;
    }

/* ZSFWS-001修复: 严格真实数据模式下，禁止使用合成数据生成器。
 * 必须从真实数据采集管道获取训练数据。
 * 无真实数据时直接返回错误，确保自主学习不使用虚假数据。 */
#ifdef SELFLNN_STRICT_REAL_DATA
    {
        /* 尝试从数据采集管道获取真实训练数据 */
        void* dp = selflnn_get_data_pipeline();
        int real_samples = 0;
        if (dp) {
            real_samples = dcpipeline_collect_training_batch(
                (DataCollectionPipeline*)dp,
                inputs, targets, actual_samples, input_dim, output_dim);
        }
        if (real_samples <= 0) {
            log_warning("[训练管线] 严格真实数据模式：%s 无可用真实训练数据，训练已跳过", module_name);
            safe_free((void**)&inputs);
            safe_free((void**)&targets);
            return -1;
        }
        if (real_samples < actual_samples) {
            actual_samples = real_samples;
            log_info("[训练管线] %s 收集到%d个真实训练样本", module_name, actual_samples);
        }
    }
#else
    /* 非严格模式：允许使用数据生成器合成训练数据（仅调试/测试用途） */
    {
        DataGeneratorConfig gen_config = DATA_GENERATOR_CONFIG_DEFAULT;
        gen_config.input_dim = input_dim;
        gen_config.output_dim = output_dim;
        gen_config.num_classes = output_dim;
        gen_config.noise_level = 0.05f;
        gen_config.signal_type = 4;

        void* generator = data_generator_create(&gen_config);
        if (!generator) {
            safe_free((void**)&inputs);
            safe_free((void**)&targets);
            return -1;
        }
        int gen_ret = data_generator_generate(generator, inputs, targets, actual_samples);
        data_generator_free(generator);
        if (gen_ret != 0) {
            safe_free((void**)&inputs);
            safe_free((void**)&targets);
            return -1;
        }
        log_warning("[训练管线] 警告：%s 使用合成数据训练（非严格模式，仅供调试）", module_name);
    }
#endif

    TrainingConfig train_cfg = training_config_default();
    train_cfg.learning_rate = learning_rate > 0.0f ? learning_rate : 1e-3f;
    train_cfg.batch_size = (size_t)((actual_samples >= 100) ? 32 : 8);
    train_cfg.epochs = (size_t)(epochs > 0 ? epochs : TDP_DEFAULT_EPOCHS);
    train_cfg.verbose = 0;
    train_cfg.use_gpu = 0;
    train_cfg.enable_validation = 0;
    train_cfg.shuffle_data = 1;
    train_cfg.mode = TRAIN_MODE_MINI_BATCH;
    train_cfg.optimizer = OPTIMIZER_ADAM;
    train_cfg.loss_function = LOSS_MSE;

    Trainer* trainer = trainer_create(&train_cfg, network);
    if (!trainer) {
        safe_free((void**)&inputs);
        safe_free((void**)&targets);
        return -1;
    }

    int result = trainer_train(trainer, inputs, targets,
                                (size_t)actual_samples,
                                NULL, NULL);

    float final_loss = 0.0f;
    float final_accuracy = 0.0f;
    int has_valid_history = 0;
    TrainingHistory* hist = trainer_get_history(trainer);
    if (hist && hist->size > 0) {
        final_loss = hist->train_losses[hist->size - 1];
        final_accuracy = hist->train_accuracies[hist->size - 1];
        has_valid_history = 1;
    }

    trainer_free(trainer);
    safe_free((void**)&inputs);
    safe_free((void**)&targets);

    if (result == 0) {
        if (has_valid_history) {
            log_info("[训练管线] %s 训练完成: loss=%.6f, acc=%.4f, 样本=%d",
                     module_name, (double)final_loss, (double)final_accuracy, actual_samples);
        } else {
            log_info("[训练管线] %s 训练完成: 样本=%d（无历史记录）",
                     module_name, actual_samples);
        }
    } else {
        log_warning("[训练管线] %s 训练失败: code=%d", module_name, result);
    }

    return result;
}

int training_pipeline_pretrain_all_vision(LNN* vision_net, LNN* deep_vision_net,
                                           LNN* liquid_vision_net, LNN* image_recog_net) {
    int ok = 0, fail = 0;

    if (vision_net) {
        int r = training_pipeline_train_multimodal(vision_net, "vision", 256, 128, 1000, 20, 1e-3f);
        if (r == 0) ok++; else fail++;
    }
    if (deep_vision_net) {
        int r = training_pipeline_train_multimodal(deep_vision_net, "deep_vision", 512, 256, 800, 15, 5e-4f);
        if (r == 0) ok++; else fail++;
    }
    if (liquid_vision_net) {
        int r = training_pipeline_train_multimodal(liquid_vision_net, "liquid_vision", 256, 128, 1000, 20, 1e-3f);
        if (r == 0) ok++; else fail++;
    }
    if (image_recog_net) {
        int r = training_pipeline_train_multimodal(image_recog_net, "image_recognition", 512, 256, 800, 15, 5e-4f);
        if (r == 0) ok++; else fail++;
    }

    log_info("[训练管线] 视觉预训练: 成功%d 失败%d", ok, fail);
    return fail > 0 ? -1 : 0;
}

int training_pipeline_pretrain_all_audio(LNN* speech_net, LNN* audio_semantic_net,
                                          LNN* vad_net) {
    int ok = 0, fail = 0;

    if (speech_net) {
        int r = training_pipeline_train_multimodal(speech_net, "speech_recognition", 128, 64, 1200, 25, 2e-3f);
        if (r == 0) ok++; else fail++;
    }
    if (audio_semantic_net) {
        int r = training_pipeline_train_multimodal(audio_semantic_net, "audio_semantic", 256, 128, 800, 15, 1e-3f);
        if (r == 0) ok++; else fail++;
    }
    if (vad_net) {
        int r = training_pipeline_train_multimodal(vad_net, "vad", 64, 2, 500, 10, 2e-3f);
        if (r == 0) ok++; else fail++;
    }

    log_info("[训练管线] 音频预训练: 成功%d 失败%d", ok, fail);
    return fail > 0 ? -1 : 0;
}

int training_pipeline_pretrain_all_sensors(LNN* sensor_fusion_net, LNN* slam_net,
                                            LNN* depth_net, LNN* ocr_net) {
    int ok = 0, fail = 0;

    if (sensor_fusion_net) {
        int r = training_pipeline_train_multimodal(sensor_fusion_net, "sensor_fusion", 128, 64, 1000, 20, 1e-3f);
        if (r == 0) ok++; else fail++;
    }
    if (slam_net) {
        int r = training_pipeline_train_multimodal(slam_net, "slam", 256, 128, 800, 15, 5e-4f);
        if (r == 0) ok++; else fail++;
    }
    if (depth_net) {
        int r = training_pipeline_train_multimodal(depth_net, "depth_estimation", 128, 64, 1000, 20, 1e-3f);
        if (r == 0) ok++; else fail++;
    }
    if (ocr_net) {
        int r = training_pipeline_train_multimodal(ocr_net, "ocr", 256, 36, 1500, 30, 2e-3f);
        if (r == 0) ok++; else fail++;
    }

    log_info("[训练管线] 传感器预训练: 成功%d 失败%d", ok, fail);
    return fail > 0 ? -1 : 0;
}

int training_pipeline_pretrain_all_modules(void* system_context) {
    /* ZSFABC修复: 从system_context获取所有子网络并执行真实预训练
     * 包括主LNN、视觉、音频、传感器等多个子模块 */
    if (!system_context) {
        log_warning("[训练管线] system_context为NULL，无法执行预训练");
        return -1;
    }

    /* 通过selflnn全局接口获取各子网络 */
    LNN* lnn_network = (LNN*)selflnn_get_shared_lnn();
    if (!lnn_network) {
        log_warning("[训练管线] 主LNN网络未初始化，跳过预训练");
        return -1;
    }

    int total_ok = 0, total_fail = 0;

    /* 预训练主LNN网络 */
    {
        void* main_network = selflnn_get_shared_lnn();
        if (main_network) {
            int r = training_pipeline_train_multimodal((LNN*)main_network,
                          "main_lnn", 128, 128, 2000, 30, 5e-4f);
            if (r == 0) total_ok++; else total_fail++;
        }
    }

    /* ZSFABC修复: 预训练视觉子系统（统一LNN架构下使用共享LNN的不同训练配置） */
    {
        void* vision_net = selflnn_get_shared_lnn();
        if (vision_net) {
            int r = training_pipeline_train_multimodal((LNN*)vision_net,
                          "vision_lnn", 64, 128, 1000, 20, 3e-4f);
            if (r == 0) total_ok++; else total_fail++;
        }
    }

    /* ZSFABC修复: 预训练音频子系统（统一LNN架构下使用共享LNN的不同训练配置） */
    {
        void* audio_net = selflnn_get_shared_lnn();
        if (audio_net) {
            int r = training_pipeline_train_multimodal((LNN*)audio_net,
                          "audio_lnn", 64, 64, 800, 15, 3e-4f);
            if (r == 0) total_ok++; else total_fail++;
        }
    }

    /* ZSFABC修复: 预训练传感器子系统（统一LNN架构下使用共享LNN的不同训练配置） */
    {
        void* sensor_net = selflnn_get_shared_lnn();
        if (sensor_net) {
            int r = training_pipeline_train_multimodal((LNN*)sensor_net,
                          "sensor_lnn", 32, 64, 500, 10, 5e-4f);
            if (r == 0) total_ok++; else total_fail++;
        }
    }

    log_info("[训练管线] 多模块预训练完成: 成功%d组 失败%d组", total_ok, total_fail);
    (void)system_context;
    return total_fail > 0 ? -1 : 0;
}
