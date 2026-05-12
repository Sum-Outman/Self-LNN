# SELF-LNN 全液态神经网络 AGI 系统架构图
# SELF-LNN Full Liquid Neural Network AGI System Architecture Diagram

> **版本 / Version:** 0.3.0 | **语言 / Language:** 100% Pure C (C11) | **构建 / Build:** CMake 3.10+
> **核心模型 / Core Model:** CfC (Closed-form Continuous-time) LNN — Token-Free 连续信号架构 / Continuous Signal Architecture
> **ODE求解器 / Solvers:** 7种/7 types (闭式解/Closed-Form, RK4, RK45, DP54, Rosenbrock, Symplectic, CTBP)
> **GPU后端 / Backends:** 10种/10 types | **API端点 / Endpoints:** ~180 | **前端/Frontend:** 15 HTML + 16 JS
> **项目信息 / Info:** [GitHub](https://github.com/Sum-Outman/Self-LNN) | [Email](mailto:silencecrowtom@qq.com)

---

## 一、系统总体架构 / I. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              SELF-LNN AGI 系统总体架构 / System Architecture                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  层1 / Layer 1: 前端交互层 / Frontend Layer — 15 HTML + 16 JS        │  │
│  │  仪表盘/Dashboard | LNN控制台/Console | 训练中心/Training             │  │
│  │  机器人控制/Robot Ctrl | 仿真控制/Simulation | 多模态学习/Multimodal  │  │
│  │  语音控制/Voice Ctrl | 对话界面/Dialogue | 安全面板/Safety Panel      │  │
│  │  知识图谱/Knowledge Graph | API文档/API Docs | 使用记录/Usage Logs   │  │
│  │  通信/Comm: WebSocket (实时推送/Real-time) + HTTP REST, 端口/Port 8080│  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                    │  ↕ HTTP/WebSocket                     │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  层2 / Layer 2: 后端服务层 / Backend Layer — main.c + backend.c      │  │
│  │  HTTP路由分发/Routing | WebSocket实时通信 | API密钥认证/Auth          │  │
│  │  熔断器/CircuitBreaker | 日志系统/Logging | 线程池调度/ThreadPool     │  │
│  │  配置管理/Config | 信号处理/Signal | 安全头注入/SecurityHeaders      │  │
│  │  ~180个API端点/Endpoints                                              │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  层3 / Layer 3: 统一多模态输入层 / Unified Multimodal Input          │  │
│  │                                                                      │  │
│  │  视觉/Vision[1024] ───→ W·x+b ──┐                                    │  │
│  │  音频/Audio[256]   ───→ W·x+b ──┤                                    │  │
│  │  文本/Text[512]    ───→ W·x+b ──┤                                    │  │
│  │  (Unicode码点/Codepoint→哈希投影/Hash→float[], Token-Free) ──┤       │  │
│  │  传感器/Sensor[128]───→ W·x+b ──┤                                    │  │
│  │  触觉/Haptic[128]  ───→ W·x+b ──┤                                    │  │
│  │  本体感/Proprio[128]──→ W·x+b ──┤                                    │  │
│  │  热感/Thermal[64]  ───→ W·x+b ──┤                                    │  │
│  │  雷达/Radar[128]   ───→ W·x+b ──┤                                    │  │
│  │  电机/Motor[64]    ───→ W·x+b ──┼──→ combined_input[256]            │  │
│  │      线性投影/Linear Projection → 直接求和/Element-wise Sum           │  │
│  │      (无融合权重/No Fusion Weights, 无注意力/No Attention)            │  │
│  │                                  ↓                                    │  │
│  │            单一 CfCCell ODE 连续动态系统 / Continuous Dynamic System │  │
│  │            τ·dh/dt = -h + σ⊙tanh(Wx+Uh+b)                             │  │
│  │            封闭形式解/Closed-Form:                                     │  │
│  │            h(t+Δt)=h(t)·e^(-Δt/τ)+(1-e^(-Δt/τ))·driver                │  │
│  │            7种ODE求解器自动选择 / 7 ODE Solvers Auto-Select           │  │
│  │                                  ↓                                    │  │
│  │            统一输出/Unified Output → 共享主LNN/Shared Main LNN       │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  层4 / Layer 4: 核心引擎层 / Core Engine — 唯一共享LNN/Single Shared │  │
│  │                                                                      │  │
│  │  ╔════════════════════════════════════════════════════════════════╗  │  │
│  │  ║  单一CfC液态神经网络 / Single CfC LNN — 128→256→128           ║  │  │
│  │  ║  全局1个实例/1 Global Instance | 20+子系统注册表共享           ║  │  │
│  │  ║  lnn_create()→cfc_create() 内部封装/Internal Wrapping          ║  │  │
│  │  ║  cfc_forward→lnn_forward | 禁止独立LNN创建/No Standalone LNN   ║  │  │
│  │  ║  全部子系统LNN化完成/All Subsystems LNNized (C-01~C-08 8/8)    ║  │  │
│  │  ╚══════════════════════════════════════════════════════════════╝  │  │
│  │                                                                      │  │
│  │  ┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐     │  │
│  │  │自我认知  │ 知识库  │ 推理引擎 │ 记忆系统 │ 学习系统 │ 训练系统 │     │  │
│  │  │Cognit.  │Knowledg.│Reasoning│ Memory  │Learning │Training │     │  │
│  │  │8403行    │三元组   │6种推理  │5层记忆  │PPO/SAC  │6阶段    │     │  │
│  │  │完整保留  │知识图谱 │因果推理 │遗忘曲线 │模仿学习 │分布式   │     │  │
│  │  │不拆分    │多跳推理 │数学推理 │记忆巩固 │元学习   │混合精度 │     │  │
│  │  │(Kept)   │KG+Infer │Causal   │Ebbingh. │Imitation│MixedPrec│     │  │
│  │  ├─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤     │  │
│  │  │机器人   │GPU加速  │安全系统 │演化引擎 │AGI核心  │分布式   │     │  │
│  │  │ Robot   │GPU Accel│ Safety  │Evolution│AGI Core │Distrib. │     │  │
│  │  │DH运动学 │10后端   │紧急停止 │CMA-ES   │12开关   │PBFT     │     │  │
│  │  │A*/RRT*  │SIMD加速 │熔断器   │NAS搜索  │全真实   │负载均衡 │     │  │
│  │  │Kinematics│10Bkends │EmergStop│NAS Arch │12Switch │LdBalance│     │  │
│  │  └─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘     │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 二、多模态统一输入架构 / II. Unified Multimodal Input Architecture

```
                    ┌──────────────────────────────────────────────────┐
                    │   统一LNN状态处理器 / Unified LNN State Processor │
                    │   src/core/unified_lnn_state.c                    │
                    │                                                   │
                    │  架构原则 / Architecture Principles:              │
                    │  ✓ 零独立编码器 / No separate encoders           │
                    │  ✓ 零跨模态注意力 / No cross-modal attention      │
                    │  ✓ 零多模型融合 / No multi-model fusion           │
                    │  ✓ 仅线性投影+求和注入同一CfC ODE                │
                    │    Linear projection + sum → single CfC ODE      │
                    │                                                   │
   ┌──────────┐     │  ┌─ 视觉/Vision[1024]       ─┐    ┌──────────┐   │
   │  摄像头   │─────┼─▶│  音频/Audio[256]         │     │  机器人   │   │
   │  Camera  │     │  │  文本/Text[512]          │     │  Robot    │   │
   ├──────────┤     │  │  (Token-Free: Unicode→Hash)│    ├──────────┤   │
   │  麦克风   │─────┼─▶│  传感器/Sensor[128]      │     │  语音输出 │   │
   │  Mic     │     │  │  触觉/Haptic[128]         │     │  Voice    │   │
   ├──────────┤     │  │  本体感/Proprio[128]      │     ├──────────┤   │
   │  传感器   │─────┼─▶│  热感/Thermal[64]        │     │  控制信号 │   │
   │  Sensors │     │  │  雷达/Radar[128]          │     │  Control  │   │
   ├──────────┤     │  │  电机/Motor[64]           │     ├──────────┤   │
   │  文本输入 │─────┼─▶│                            │     │  决策输出 │   │
   │  Text    │     │  │  ↓ 各自线性投影(256维)     │     │  Decision │   │
   └──────────┘     │  │  ↓ Linear Projection 256d  │     └──────────┘   │
                    │  │  ↓ 直接求和 / Elt-wise Sum │                   │
                    │  │  ↓ 单一CfC ODE演化         │                   │
                    │  │  ↓ Single CfC ODE Evolution│                   │
                    │  │  ↓ hidden_state[256]        │                   │
                    │  │  ↓ output_weight·h+bias     │                   │
                    │  │  ↓ 统一输出 / Unified Out  │                   │
                    │  └────────────────────────────┘                     │
                    └──────────────────────────────────────────────────┘
```

---

## 三、训练流水线架构 / III. Training Pipeline Architecture (6 Stages)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  训练流水线 / Training Pipeline — src/training/                              │
│                                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──┐│
│  │ 阶段1    │  │ 阶段2    │  │ 阶段3    │  │ 阶段4    │  │ 阶段5    │  │  ││
│  │ Stage 1  │  │ Stage 2  │  │ Stage 3  │  │ Stage 4  │  │ Stage 5  │  │  ││
│  │ 预训练   │─▶│ 深度训练 │─▶│ 多模态   │─▶│ 微调     │─▶│ 本地     │─▶│评估││
│  │ Pretrain │  │ Deep     │  │ 联合训练 │  │ Fine-tune│  │ 适配     │  │Eval││
│  ├──────────┤  ├──────────┤  ├──────────┤  ├──────────┤  ├──────────┤  ├──┤│
│  │基础CfC   │  │多层CfC   │  │多模态    │  │任务      │  │本地数据  │  │测试││
│  │BaseCfC   │  │MultiLayer│  │对齐联合  │  │特定微调  │  │个性化适配│  │验证││
│  │权重初始化│  │全部层训练│  │Multimodal│  │Task Spec │  │LocalAdapt│  │Test││
│  │Init Wgts │  │AllLayers │  │AlignJoint│  │Fine-Tune │  │Personaliz│  │Val ││
│  ├──────────┤  ├──────────┤  ├──────────┤  ├──────────┤  ├──────────┤  │  ││
│  │lr=1e-3   │  │lr=5e-4   │  │lr=1e-4   │  │lr=1e-5   │  │lr=5e-6   │  │  ││
│  │bs=64     │  │bs=32     │  │bs=16     │  │bs=8      │  │bs=4      │  │  ││
│  │ep=100    │  │ep=200    │  │ep=300    │  │ep=50     │  │ep=30     │  │  ││
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──┘│
│                                                                              │
│  优化策略/Optimization: 混合精度/MixedPrec(FP16/BF16) | 梯度裁剪/GradClip   │
│             自适应学习率调度/Adaptive LR Schedule                             │
│  分布式/Distributed: Top-k梯度压缩/Top-k Compress | Ring AllReduce | 故障恢复│
│  扩展/Extension: 外部API训练/Ext API Training | 模型版本管理/Model Version   │
│              增量检查点/Increm Checkpoint                                    │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 四、学习与演化系统 / IV. Learning & Evolution System

```
┌──────────────────────────────────────────────────────────────────────┐
│  学习子系统 / Learning Subsystem — src/learning/ (全LNN化 ✅)         │
│                                                                      │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌──────────────────┐  │
│  │ 在线学习   │ │ 强化学习   │ │ 模仿学习   │ │ 元学习           │  │
│  │ Online     │ │ RL         │ │ Imitation  │ │ Meta-Learning    │  │
│  │ ADWIN      │ │ PPO/SAC    │ │ BC/DAgger  │ │ MAML/Reptile     │  │
│  │ PageHinkley│ │ GAE        │ │ IRL/GAIL   │ │ 快速适应/FastAdpt│  │
│  │ KSWIN      │ │ 经验回放   │ │ 专家演示   │ │                  │  │
│  │            │ │ ExpReplay  │ │ ExpertDemo │ │                  │  │
│  ├────────────┤ ├────────────┤ ├────────────┤ ├──────────────────┤  │
│  │ 人工教学   │ │ 演示教学   │ │ 知识整合   │ │ API训练          │  │
│  │ HumanTeach │ │ DemoTeach  │ │ Knowledge  │ │ API Training     │  │
│  │ 人类反馈   │ │ 示教闭环   │ │ 知识融合   │ │ 远程数据/Remote  │  │
│  │ HumanFB    │ │ TeachLoop  │ │ Integrat.  │ │                  │  │
│  └────────────┘ └────────────┘ └────────────┘ └──────────────────┘  │
│                                                                      │
│  演化子系统 / Evolution Subsystem — src/evolution/ (全LNN化 ✅)       │
│                                                                      │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐             │
│  │ CMA-ES       │ │ 帕累托优化   │ │ NAS 架构搜索      │             │
│  │ SBX交叉      │ │ Pareto Opt   │ │ Neural Arch      │             │
│  │ 多项式变异   │ │ 非支配排序   │ │ Search            │             │
│  │ PloyMutate   │ │ NonDomSort   │ │ 网络拓扑自动设计  │             │
│  ├──────────────┤ ├──────────────┤ │ AutoTopology      │             │
│  │ 5种选择策略  │ │ 4种交叉策略  │ ├──────────────────┤             │
│  │ 5 SelectStrat│ │ 4 CrossStrat │ │ 5种变异策略       │             │
│  │ 岛模型迁移   │ │ 多样性增强   │ │ IPOP重启          │             │
│  │ IslandMigrat │ │ Diversity    │ │ IPOP Restart      │             │
│  └──────────────┘ └──────────────┘ └──────────────────┘             │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 五、自我认知与元认知 / V. Self-Cognition & Metacognition

```
┌────────────────────────────────────────────────────────────────────┐
│  自我认知系统 / Self-Cognition System — src/cognition/              │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  自我认知 / Self-Cognition — self_cognition.c                │   │
│  │  (8403行/8403 Lines, 整体保留不拆分/Kept As-Is)              │   │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌────────────┐   │   │
│  │  │ 状态监控  │ │ 能力评估  │ │ 知识元认知│ │ 目标自省   │   │   │
│  │  │StateMonit │ │CapEval    │ │Metacognit │ │GoalIntrosp │   │   │
│  │  │ CPU/内存  │ │ 推理/学习 │ │ 缺陷识别  │ │ 进展跟踪   │   │   │
│  │  │ 误差率    │ │ 适应/创意 │ │ 知识缺口  │ │ 优先级调整 │   │   │
│  │  └───────────┘ └───────────┘ └───────────┘ └────────────┘   │   │
│  │  ┌──────────────────────────────────────────────────────┐    │   │
│  │  │ 深度反思 / Deep Reflection — 5维迭代分析/5-Dim Iter │    │   │
│  │  │ ①信念一致性/BeliefConsistent                        │    │   │
│  │  │ ②假设检验/HypothesisTest(趋势/Trend)                 │    │   │
│  │  │ ③矛盾检测/ContradictionDetect(认知冲突/CogConflict)  │    │   │
│  │  │ ④风险评估/RiskAssessment(错误率/修正强度)            │    │   │
│  │  │ ⑤体验质量评估/ExperienceQualityEvaluation            │    │   │
│  │  └──────────────────────────────────────────────────────┘    │   │
│  │  ┌──────────────────────────────────────────────────────┐    │   │
│  │  │ 认知记忆 / Cognitive Memory                           │    │   │
│  │  │ Ebbinghaus遗忘曲线/ForgettingCurve | Hebbian巩固      │    │   │
│  │  │ 内容哈希检索/ContentHashRetrieval                     │    │   │
│  │  │ 短期→长期记忆整合/STM→LTM Consolidation               │    │   │
│  │  │ 256片段容量/256 Segments | 自动过期清除/AutoExpiry    │    │   │
│  │  └──────────────────────────────────────────────────────┘    │   │
│  │  ┌──────────────────────────────────────────────────────┐    │   │
│  │  │ 迭代元认知循环 / Iterative Metacognition              │    │   │
│  │  │ 反思→修正→再反思 递归循环(最多5轮/Max5Rounds)        │    │   │
│  │  │ Reflect→Correct→Re-reflect recursive cycle            │    │   │
│  └─────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────┘
```

---

## 六、GPU 加速架构 / VI. GPU Acceleration Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  GPU 抽象层 / GPU Abstraction Layer — src/gpu/gpu.c                  │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  统一接口/Unified: gpu_forward_dense / gpu_matmul_train / ... │   │
│  │  运行时选择/Runtime: 检测硬件→动态加载驱动→创建上下文         │   │
│  │  Detect HW → Dynamic Load Driver → Create Context             │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                    │                                  │
│  ┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐      │
│  │ CUDA    │ OpenCL  │ Vulkan  │ Metal   │ ROCm    │ Intel   │      │
│  │ Driver  │ 1.2+    │ Compute │ macOS   │ AMD GPU │ Level   │      │
│  │ API     │         │ Shader  │         │         │ Zero    │      │
│  ├─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤      │
│  │ Ascend  │Cambricon│ TPU     │ CPU-SIMD│         │         │      │
│  │ NPU     │ MLU     │ Google  │ SSE/AVX │         │         │      │
│  │AscendCL │ CNRT    │tpuExec  │ NEON    │         │         │      │
│  └─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘      │
│                                                                      │
│  CPU-SIMD 详情/Details: ~260行/Lines SSE/NEON向量化/Vectorized代码   │
│  • 向量点积/Dot Product    • 向量加法/Add                            │
│  • Sigmoid批量近似(泰勒展开/TaylorExpansion)                          │
│  • Tanh批量近似             • CfC门控计算/gate*sigmoid+(1-gate)*tanh │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 七、编译架构 / VII. Build Architecture

```
CMakeLists.txt (根/Root)
├── src/core/CMakeLists.txt       → selflnn_core.lib (32源文件/32 Files)
├── src/multimodal/CMakeLists.txt → selflnn_multimodal.lib
├── src/cognition/CMakeLists.txt  → selflnn_cognition.lib
├── src/reasoning/CMakeLists.txt  → selflnn_reasoning.lib
├── src/knowledge/CMakeLists.txt  → selflnn_knowledge.lib
├── src/memory/CMakeLists.txt     → selflnn_memory.lib
├── src/learning/CMakeLists.txt   → selflnn_learning.lib (全LNN化 ✅)
├── src/training/CMakeLists.txt   → selflnn_training.lib (全LNN化 ✅)
├── src/robot/CMakeLists.txt      → selflnn_robot.lib
├── src/gpu/CMakeLists.txt        → selflnn_gpu.lib (10后端/10 Backends)
├── src/backend/CMakeLists.txt    → selflnn_backend.lib
├── src/evolution/CMakeLists.txt  → selflnn_nas.lib (全LNN化 ✅)
├── src/safety/CMakeLists.txt     → selflnn_safety.lib
├── src/concurrency/CMakeLists.txt→ selflnn_concurrency.lib
├── src/distributed/CMakeLists.txt→ selflnn_distributed.lib
├── src/agi/CMakeLists.txt        → selflnn_agi.lib
├── src/programming/CMakeLists.txt→ selflnn_programming.lib
├── src/product_design/CMakeLists.txt → selflnn_product_design.lib
├── src/multisystem/CMakeLists.txt→ selflnn_multisystem.lib
├── src/utils/CMakeLists.txt      → utils.lib
├── tests/                        → 15+ 测试可执行文件/Test Exes
└── examples/                     → 5 示例可执行文件/Example Exes
```

---

## 八、Token-Free 连续信号架构 / VIII. Token-Free Continuous Signal Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  SELF-LNN 是 Token-Free 架构 — 不使用离散 token id                    │
│  SELF-LNN is a Token-Free Architecture — No Discrete Token IDs       │
│                                                                      │
│  文本输入流程 / Text Input Flow:                                      │
│  原始文本/RawText → UTF-8解码/Decode → Unicode码点/Codepoint(uint32)  │
│     → 哈希投影/Hash Projection(h=cp*2654435761u)                     │
│     → float[input_dim] 连续向量/Continuous Vector                    │
│     → LNN 逐字符序列前向传播/Char-Seq Forward → float[hidden_dim]     │
│                                                                      │
│  vs GPT/BERT Token 体系 / vs GPT/BERT Token System:                   │
│  ┌────────────────┬───────────────────┬──────────────────┐          │
│  │ 特性/Feature    │ GPT/BERT           │ SELF-LNN          │          │
│  ├────────────────┼───────────────────┼──────────────────┤          │
│  │ 输入类型/InType │ 离散token ID/Discrete │ 连续float向量/Cont │       │
│  │ 分词器/Tokenizer│ BPE/WordPiece      │ 无/None(哈希投影) │          │
│  │ 词汇表/Vocab    │ 固定50000+/Fixed    │ 零词汇表/ZeroVocab │         │
│  │ [UNK]问题       │ 有/Yes              │ 无/No(全Unicode)  │          │
│  │ 词嵌入/WordEmb  │ 查表embedding[]     │ 连续哈希/ContHash  │          │
│  │ 数学基础/Math   │ 概率链式法则/Chain  │ CfC ODE微分方程/ODE│         │
│  │ 输出词汇表/Out  │ N/A                 │ 仅decode用/Decode  │          │
│  └────────────────┴───────────────────┴──────────────────┘          │
│                                                                      │
│  输出流程 / Output Flow:                                              │
│  LNN float[] 连续输出/Continuous Out                                 │
│     → gen_projection_lnn (LNN投影/Projection)                        │
│     → float[] logits → softmax → Unicode码点/Codepoint               │
│     → UTF-8字符串/String                                              │
│                                                                      │
│  输出端词汇表(仅127字符，非NLP token) / Output Char Set:               │
│  BOS/EOS + 95 ASCII + 30标点/Punctuation + 3969中文字/Chinese         │
│  = ~4096字符/Characters                                              │
│  仅用于LNN连续logits→人类可读文本, 非输入分词                          │
│  For LNN logits→human-readable text decode only, NOT input tokenizer │
└──────────────────────────────────────────────────────────────────────┘
```

---

> **文档更新 / Updated**: 2026-05-11
> **相关文档 / Related Docs**: [README](../README.md) | [AGI机器人指南](./AGI_Robot_Guide.md) | [开发指南](./DEVELOPMENT_GUIDE_ZH.md)
