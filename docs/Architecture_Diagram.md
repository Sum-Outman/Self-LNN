# SELF-LNN 全液态神经网络 AGI 系统架构图
# SELF-LNN Full Liquid Neural Network AGI System Architecture Diagram

> **版本 / Version:** 1.6.0 | **语言 / Language:** 100% Pure C (C11) | **构建 / Build:** CMake 3.10+
> **核心模型 / Core Model:** CfC LNN — Token-Free 连续信号架构 / Continuous Signal Architecture
> **ODE求解器 / Solvers:** 8种/8 types | **GPU后端 / Backends:** 10种 | **API Slots:** 290
> **前端/Frontend:** SPA + 19 JS | **项目:** [GitHub](https://github.com/Sum-Outman/Self-LNN)

---

## 一、系统总体架构 / I. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│          SELF-LNN AGI 渐进分层架构 / Progressive Layering Architecture      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─ 层1: 前端 SPA + 19 JS ─── HTTP REST 8080 + WebSocket 9090 ──────────┐  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│  ┌─ 层2: 后端服务层 ── 290 handler槽位 | 路由分发 | 认证 | 安全 ────────┐  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│  ┌─ 层3: 统一多模态输入层 ── 固定Xavier投影矩阵(projection_locked=1) ───┐  │
│  │  视觉/语音/文本/传感器/触觉/本体感/热感/雷达/电机 → 统一256D → 共享LNN │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│  ┌─ 层4: 核心引擎层 ── 渐进分层隔离 ────────────────────────────────────┐  │
│  │                                                                      │  │
│  │  ╔═══════════════════════════════════════╗                            │  │
│  │  ║ 共享LNN (全局唯一) 128→256→128       ║                            │  │
│  │  ╚═══════════════════════════════════════╝                            │  │
│  │           │                                                          │  │
│  │  ┌────────┴────────┐                                                  │  │
│  │  │ 感知馈入(写)     │   │ 生成隔离(只读)     │                       │  │
│  │  │ lnn_forward     │   │ lnn_get_output    │                       │  │
│  │  │ 修改hidden_state│   │ 不修改hidden_state │                       │  │
│  │  └────────┬────────┘   └────────┬──────────┘                        │  │
│  │  ┌──视觉CfC(独立)    ┌──对话私有ODE(gen_private_hidden)             │  │
│  │  │──语音(共享LNN)    │──TTS自包含CfC(embedding_table)               │  │
│  │  │──传感器(独立)     │──后端回退(lnn_get_output)                    │  │
│  │  │──统一信号(共享LNN) └──────────────────                            │  │
│  │  └──────────────                                                     │  │
│  │                                                                      │  │
│  │  子系统: 认知|知识库|推理|记忆|学习|训练|机器人|GPU|安全|演化        │  │
│  │                                                                      │  │
│  │  知识图谱桥接: knowledge_graph_to_lnn_bridge() → 概念嵌入聚合 → LNN扰动      │  │
│  │  KG→LNN Injection: bridge(kg,lnn,0.15f) at dialogue entry + 0.1f at AGI loop │  │
│  │                                                                      │  │
│  │  ┌──────────────────────────────────────────────────────────────────┐│  │
│  │  │ 动态架构控制器 / Dynamic Architecture Controller                ││  │
│  │  │ 运行时扩展/收缩/增删层·知识迁移·安全审批·原子交换·KG API(7端点)/KG API(7 endpoints)││  │
│  │  └──────────────────────────────────────────────────────────────────┘│  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 二、渐进分层架构原则 / Progressive Layering Principles

### 中文
1. **感知模态**（视觉/语音/传感器）：通过 `lnn_forward` 馈入共享LNN，修改 `hidden_state` — 正确行为
2. **生成模态**（对话/TTS）：通过 `lnn_get_output` 只读查询，使用私有ODE状态自回归 — 零污染
3. **投影矩阵**：Xavier初始化后锁定（`projection_locked=1`），固定线性投影
4. **VH融合**：废弃独立CfC ODE (`enable_cfc_fusion=0`)，改用投影拼接+EMA
5. **对话隔离**：512次/token共享LNN状态重写 → 0次

### English
1. **Perception Modalities** (vision/speech/sensors): Feed into shared LNN via `lnn_forward`, modify `hidden_state` — correct behavior
2. **Generation Modalities** (dialogue/TTS): Query via `lnn_get_output` (read-only), use private ODE states — zero pollution
3. **Projection Matrix**: Locked after Xavier init (`projection_locked=1`), fixed linear projection
4. **VH Fusion**: Deprecated independent CfC ODE (`enable_cfc_fusion=0`), replaced with projection concat + EMA
5. **Dialogue Isolation**: 512 shared LNN state rewrites per token → 0

---

## 三、数据流架构 / Data Flow Architecture

```
传感器数据流 (不污染):
  原始传感器 → EKF/UKF/PF本地滤波(CfC默认禁用) → z-score+tanh归一化
  → 固定Xavier投影拼接 → lnn_forward_with_memory_context(共享LNN)
  ✅ 传感器CfC权重全部本地独立，0次LNN函数调用

对话生成流 (已隔离):
  用户输入 → lnn_forward(共享LNN) ← 感知馈入(正确)
  生成回复 → lnn_get_output(共享LNN) ← 只读查询
  → gen_private_hidden(私有CfC ODE) → gen_projection_lnn(独立LNN) → 输出
  ✅ 0次共享LNN hidden_state修改

TTS合成流 (已隔离):
  文本tokens → 嵌入表编码 → 自包含CfC(embedding_table权重)
  → TTS编码器3层CfC ODE(独立) → 解码器3层CfC ODE(独立) → 波形投影
  ✅ 0次共享LNN调用
```

---

## 四、独立动态系统清单 / Independent Dynamic Systems

| 系统 | 数量 | 耦合共享LNN |
|------|------|------------|
| 共享LNN (唯一核心) | 1 | 自身 |
| CfcVisionProcessor | 6 | 独立 |
| TTS编码器CfC层 | 3 | 独立 |
| TTS解码器CfC层 | 3 | 独立 |
| HapticCfcProcessor | 1 | 独立 |
| DialogueGenerator CfC | 1 | 独立 |
| QuaternionLNN | 1 | 独立 |
| ArchitectureController | 1 | 与共享LNN耦合（运行时监控+架构变更） |
| **总计** | **17** | |
