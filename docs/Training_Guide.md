# 训练指南 / Training Guide

---

## 目录 / Table of Contents
- [简介 / Introduction](#简介--introduction)
- [训练模式 / Training Modes](#训练模式--training-modes)
- [优化器 / Optimizers](#优化器--optimizers)
- [损失函数 / Loss Functions](#损失函数--loss-functions)
- [训练配置 / Training Configuration](#训练配置--training-configuration)
- [训练流程 / Training Workflow](#训练流程--training-workflow)
- [学习率调度 / Learning Rate Scheduling](#学习率调度--learning-rate-scheduling)
- [模型保存与加载 / Model Saving and Loading](#模型保存与加载--model-saving-and-loading)

---

## 简介 / Introduction

### 中文
SELF-LNN提供完整的训练框架，支持多种训练模式、优化器和学习率调度策略。系统同时支持CPU训练和各种品牌GPU训练。

### English
SELF-LNN provides a complete training framework, supporting multiple training modes, optimizers, and learning rate scheduling strategies. The system supports both CPU training and various brands of GPU training.

---

## 训练模式 / Training Modes

### 中文
| 模式 / Mode | 中文名称 / Chinese Name | 描述 / Description |
|------------|-----------------------|------------------|
| TRAIN_MODE_BATCH | 批量训练 | 使用全部数据进行一次参数更新 |
| TRAIN_MODE_MINI_BATCH | 小批量训练 | 将数据分成小批次进行训练 |
| TRAIN_MODE_ONLINE | 在线训练 | 每个样本进行一次参数更新 |
| TRAIN_MODE_ADAPTIVE | 自适应训练 | 根据数据特性动态调整训练策略 |

### English
| Mode | Name | Description |
|------|------|-------------|
| TRAIN_MODE_BATCH | Batch Training | Update parameters once using all data |
| TRAIN_MODE_MINI_BATCH | Mini-Batch Training | Split data into small batches for training |
| TRAIN_MODE_ONLINE | Online Training | Update parameters once per sample |
| TRAIN_MODE_ADAPTIVE | Adaptive Training | Dynamically adjust training strategy based on data characteristics |

---

## 优化器 / Optimizers

### 中文
| 优化器 / Optimizer | 描述 / Description |
|------------------|-------------------|
| OPTIMIZER_SGD | 随机梯度下降 |
| OPTIMIZER_MOMENTUM | 带动量的SGD |
| OPTIMIZER_ADAM | Adam优化器 |
| OPTIMIZER_RMSPROP | RMSprop优化器 |

### English
| Optimizer | Description |
|-----------|-------------|
| OPTIMIZER_SGD | Stochastic Gradient Descent |
| OPTIMIZER_MOMENTUM | SGD with Momentum |
| OPTIMIZER_ADAM | Adam Optimizer |
| OPTIMIZER_RMSPROP | RMSprop Optimizer |

---

## 损失函数 / Loss Functions

### 中文
| 损失函数 / Loss Function | 中文名称 / Chinese Name | 适用场景 / Use Case |
|------------------------|-----------------------|-------------------|
| LOSS_MSE | 均方误差 / Mean Squared Error | 回归问题 / Regression |
| LOSS_MAE | 平均绝对误差 / Mean Absolute Error | 回归问题 / Regression |
| LOSS_CROSS_ENTROPY | 交叉熵 / Cross Entropy | 分类问题 / Classification |
| LOSS_BINARY_CROSS_ENTROPY | 二元交叉熵 / Binary Cross Entropy | 二分类问题 / Binary Classification |

### English
| Loss Function | Name | Use Case |
|----------------|------|----------|
| LOSS_MSE | Mean Squared Error | Regression |
| LOSS_MAE | Mean Absolute Error | Regression |
| LOSS_CROSS_ENTROPY | Cross Entropy | Classification |
| LOSS_BINARY_CROSS_ENTROPY | Binary Cross Entropy | Binary Classification |

---

## 训练配置 / Training Configuration

### 中文
系统提供默认训练配置，包含训练模式、优化器、损失函数、学习率、批次大小、训练轮数等关键参数配置。

### English
The system provides default training configurations, including key parameters like training mode, optimizer, loss function, learning rate, batch size, and training epochs.

---

## 训练流程 / Training Workflow

### 中文
1. **数据准备**：加载和预处理训练数据
2. **网络创建**：初始化液态神经网络
3. **训练器创建**：配置训练参数
4. **训练循环**：前向传播、损失计算、反向传播、参数更新
5. **验证**：在验证集上评估性能
6. **模型保存**：保存最佳模型
7. **测试**：在测试集上评估最终性能

### English
1. **Data Preparation**: Load and preprocess training data
2. **Network Creation**: Initialize liquid neural network
3. **Trainer Creation**: Configure training parameters
4. **Training Loop**: Forward pass, loss calculation, backward pass, parameter update
5. **Validation**: Evaluate performance on validation set
6. **Model Saving**: Save best model
7. **Testing**: Evaluate final performance on test set

---

## 学习率调度 / Learning Rate Scheduling

### 中文
| 调度器 / Scheduler | 中文名称 / Chinese Name | 描述 / Description |
|-------------------|-----------------------|------------------|
| SCHEDULER_CONSTANT | 恒定学习率 | 学习率保持不变 |
| SCHEDULER_STEP | 阶梯衰减 | 每隔固定步数衰减 |
| SCHEDULER_EXPONENTIAL | 指数衰减 | 每步指数衰减 |
| SCHEDULER_COSINE | 余弦衰减 | 余弦函数衰减 |
| SCHEDULER_CYCLIC | 循环学习率 | 在范围内循环变化 |

### English
| Scheduler | Name | Description |
|-----------|------|-------------|
| SCHEDULER_CONSTANT | Constant Learning Rate | Learning rate remains unchanged |
| SCHEDULER_STEP | Step Decay | Decay at fixed intervals |
| SCHEDULER_EXPONENTIAL | Exponential Decay | Exponential decay each step |
| SCHEDULER_COSINE | Cosine Decay | Cosine function decay |
| SCHEDULER_CYCLIC | Cyclic Learning Rate | Cyclically varies within range |

---

## 模型保存与加载 / Model Saving and Loading

### 中文
系统支持模型的保存、加载、检查点保存和恢复功能。

### English
The system supports model saving, loading, checkpoint saving and recovery functions.

---

> **文档更新 / Updated**: 2026-05-11
> **相关文档 / Related Docs**: [架构图](./Architecture_Diagram.md) | [用户手册](./USER_GUIDE.md) | [开发指南](./DEVELOPMENT_GUIDE_ZH.md)
