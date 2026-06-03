# SELF-LNN AGI 系统 API 参考文档 / API Reference

---

## 概述 / Overview

SELF-LNN AGI 基于**单一 CfC 液态神经网络 + 渐进分层架构**。感知模态通过 `lnn_forward` 馈入共享LNN，生成模态通过 `lnn_get_output` 只读查询后使用私有ODE自回归，确保零全局状态污染。
默认服务端口：`http://localhost:8080`（可通过 `--port` 参数修改）

SELF-LNN AGI is based on a **single CfC Liquid Neural Network** architecture. All modality data is concatenated into a unified input vector and evolved through the same CfC continuous dynamic system.
Default port: `http://localhost:8080` (configurable via `--port`)

---

## 系统状态 / System Status

| 方法/Method | 路径/Path | 描述/Description |
|------------|-----------|-----------------|
| GET | /api/status | 系统运行状态 / System status |
| GET | /api/health | 健康检查（含GPU/CPU/组件状态）/ Health check (GPU/CPU/components) |
| GET | /api/stats | 服务器统计信息 / Server statistics |
| GET | /api/memory | 内存状态 / Memory status |
| GET | /api/system/diagnostic | 系统诊断 / System diagnostic |
| GET | /api/system/export_diagnostic | 导出诊断数据 / Export diagnostic data |
| POST | /api/reset | 重置系统 / Reset system |
| POST | /api/shutdown | 关闭系统 / Shutdown system |
| POST | /api/backup | 系统备份 / System backup |
| POST | /api/model/load | 模型加载 / Load model |

---

## AGI 功能 / AGI Features

| 方法/Method | 路径/Path | 描述/Description |
|------------|-----------|-----------------|
| GET | /api/agi/features | AGI功能状态列表 / Feature status list |
| GET | /api/agi/feature_list | 所有AGI功能状态 / All feature statuses |
| GET | /api/agi/cognition/state | 自我认知详细状态 / Self-cognition state |
| POST | /api/agi/feature/toggle | 切换AGI功能 / Toggle AGI feature |
| POST | /api/agi/execute | AGI任务执行 / Execute AGI task |
| POST | /api/agi/task/status | 任务状态查询 / Query task status |
| POST | /api/agi/self_correction | 触发自我修正 / Trigger self-correction |
| POST | /api/agi/think | 全模态思考（思维链+反思+认知）/ Full-modal thinking |
| POST | /api/agi/decide | 自主决策 / Autonomous decision |
| POST | /api/agi/learn | 在线学习 / Online learning |
| POST | /api/agi/evolve | 进化演化 / Evolution |
| POST | /api/agi/memory | 记忆系统读写 / Memory read/write |
| POST | /api/agi/plan | 自主规划 / Autonomous planning |

---

## 多模态 / Multimodal

| 方法/Method | 路径/Path | 描述/Description |
|------------|-----------|-----------------|
| POST | /api/vision | 视觉输入处理 / Vision input |
| POST | /api/audio | 音频输入处理 / Audio input |
| POST | /api/text | 文本输入处理 / Text input |
| POST | /api/sensor | 传感器输入处理 / Sensor input |
| POST | /api/multimodal/learn | 多模态学习触发 / Trigger multimodal learning |
| GET | /api/multimodal/status | 多模态处理状态 / Multimodal status |
| POST | /api/multimodal/process | 多模态输入处理 / Process multimodal input |
| POST | /api/multimodal/config | 配置多模态参数 / Configure multimodal |
| POST | /api/multimodal/reset | 重置多模态配置 / Reset multimodal config |
| POST | /api/multimodal/stop | 停止多模态处理 / Stop multimodal |
| POST | /api/multimodal/teach | 多模态教学 / Multimodal teaching |
| POST | /api/multimodal/teach/test | 多模态教学测试 / Test teaching |
| GET | /api/sensor/pipeline/status | 传感器管道状态 / Sensor pipeline status |

---

## 对话 / Dialogue

| 方法/Method | 路径/Path | 描述/Description |
|------------|-----------|-----------------|
| POST | /api/dialogue | 对话处理 / Dialogue processing |
| GET | /api/dialogue/history | 获取对话历史 / Get dialogue history |
| POST | /api/dialogue/clear | 清除对话历史 / Clear dialogue history |
| POST | /api/dialogue/multimodal | 多模态对话 / Multimodal dialogue |

---

## 知识库 / Knowledge

| 方法/Method | 路径/Path | 描述/Description |
|------------|-----------|-----------------|
| GET/POST | /api/knowledge | 知识库查询/添加 / Query/add knowledge |
| POST | /api/learning/from-dialogue | 从对话中学习 / Learn from dialogue |
| POST | /api/learning/from-manual | 从说明书学习 / Learn from manual |

---

## 训练 / Training

| 方法/Method | 路径/Path | 描述/Description |
|------------|-----------|-----------------|
| POST | /api/training | 训练请求 / Training request |
| POST | /api/training/start | 开始训练 / Start training |
| POST | /api/training/stop | 停止训练 / Stop training |
| POST | /api/training/pause | 暂停训练 / Pause training |
| GET | /api/training/status | 训练状态 / Training status |
| POST | /api/training/pretrain | 预训练 / Pretraining |
| POST | /api/training/fine-tune | 微调 / Fine-tuning |
| POST | /api/training/transfer | 迁移学习 / Transfer learning |
| POST | /api/training/continual | 持续学习 / Continual learning |
| POST | /api/training/from-scratch | 从零开始训练 / Train from scratch |
| POST | /api/training/external-api | 外部API训练 / External API training |
| GET | /api/training/history | 训练历史 / Training history |
| POST | /api/training/export | 导出训练结果 / Export results |
| POST | /api/training/log/clear | 清除训练日志 / Clear training log |

---

## 推理 / Reasoning

| 方法/Method | 路径/Path | 描述/Description |
|------------|-----------|-----------------|
| GET | /api/reasoning | 执行推理 / Execute reasoning |

---

## 机器人控制 / Robot Control

| 方法/Method | 路径/Path | 描述/Description |
|------------|-----------|-----------------|
| GET | /api/robot/status | 机器人状态 / Robot status |
| POST | /api/robot/command | 发送控制命令 / Send command |
| GET | /api/robot/sensor | 传感器数据 / Sensor data |
| POST | /api/robot/trajectory | 执行轨迹 / Execute trajectory |
| POST | /api/robot/emergency_stop | 紧急停止 / Emergency stop |
| POST | /api/robot/connect | 连接机器人 / Connect robot |
| POST | /api/robot/disconnect | 断开连接 / Disconnect robot |
| GET | /api/robot/list | 机器人列表 / Robot list |
| POST | /api/robot/parameters | 设置参数 / Set parameters |
| POST | /api/robot/coordinate | 坐标控制 / Coordinate control |
| POST | /api/robot/training | 训练控制 / Training control |
| POST | /api/robot/calibrate | 标定 / Calibrate |
| POST | /api/robot/execute_task | 执行任务 / Execute task |
| POST | /api/robot/execute_action | 执行动作 / Execute action |
| POST | /api/robot/stop_task | 停止任务 / Stop task |
| POST | /api/robot/learn_from_demo | 从演示中学习 / Learn from demo |
| POST | /api/robot/reboot | 重启机器人 / Reboot robot |
| POST | /api/robot/firmware | 固件升级 / Firmware update |
| POST | /api/robot/analyze_screen | 分析机器人屏幕 / Analyze robot screen |
| POST | /api/robot/config/reset | 重置配置 / Reset config |
| POST | /api/multi_robot/sync | 多机器人同步 / Multi-robot sync |

---

## ROS 接口 / ROS Interface

| 方法/Method | 路径/Path | 描述/Description |
|------------|-----------|-----------------|
| GET | /api/ros/status | ROS Master状态 / ROS Master status |
| POST | /api/ros/configure | 配置ROS连接 / Configure ROS connection |
| GET | /api/ros/nodes | ROS节点列表 / ROS nodes list |
| GET | /api/ros/topics | ROS主题列表 / ROS topics list |
| POST | /api/ros/publish | 话题发布 / Publish topic |
| POST | /api/ros/subscribe | 话题订阅 / Subscribe topic |
| POST | /api/ros/service | 服务调用 / Service call |
| POST | /api/gazebo/control | Gazebo仿真控制 / Gazebo control |

---

## 计算机操作 / Computer Operations

| 方法/Method | 路径/Path | 描述/Description |
|------------|-----------|-----------------|
| POST | /api/computer/launch | 启动应用 / Launch application |
| POST | /api/computer/close | 关闭应用 / Close application |
| POST | /api/computer/type | 键盘输入 / Keyboard input |
| POST | /api/computer/screenshot | 截取屏幕 / Screenshot |
| POST | /api/computer/execute | 执行命令 / Execute command |
| POST | /api/computer/volume | 音量控制 / Volume control |

---

## 仿真控制 / Simulation Control

| 方法/Method | 路径/Path | 描述/Description |
|------------|-----------|-----------------|
| POST | /api/simulation/start | 启动仿真 / Start simulation |
| POST | /api/simulation/stop | 停止仿真 / Stop simulation |
| GET | /api/simulation/status | 仿真状态 / Simulation status |
| POST | /api/simulation/reset | 重置仿真 / Reset simulation |
| POST | /api/simulation/plan_path | 路径规划 / Path planning |
| POST | /api/simulation/robot_control | 仿真机器人控制 / Robot control in sim |
| POST | /api/simulation/reconstruct | 3D场景重建 / 3D scene reconstruction |

---

## 教学 API (Say-Look-Touch-Count) / Teaching API

| 方法/Method | 路径/Path | 描述/Description |
|------------|-----------|-----------------|
| GET | /api/teach/get_concepts | 获取已教概念 / Get taught concepts |
| POST | /api/teach/test_concept | 测试概念 / Test concept |
| POST | /api/teach/say_and_associate | 说与关联 / Say & associate |
| POST | /api/teach/look_and_learn | 看与学习 / Look & learn |
| POST | /api/teach/touch_and_understand | 触与理解 / Touch & understand |
| POST | /api/teach/count_and_generalize | 计数与泛化 / Count & generalize |
| POST | /api/teach/clear_concept | 清除单个概念 / Clear single concept |
| POST | /api/teach/clear_all_concepts | 清除所有概念 / Clear all concepts |

---

## 语音 / Voice

| 方法/Method | 路径/Path | 描述/Description |
|------------|-----------|-----------------|
| POST | /api/audio/recognize | 语音识别 / Speech recognition |
| POST | /api/tts/synthesize | 语音合成 / TTS synthesis |
| POST | /api/audio/stream | 实时语音流 / Real-time audio stream |
| POST | /api/audio/command | 语音指令 / Voice command |
| GET | /api/voice/history | 语音历史 / Voice history |

---

## 设备与硬件 / Devices & Hardware

| 方法/Method | 路径/Path | 描述/Description |
|------------|-----------|-----------------|
| POST | /api/devices/list | 列出设备 / List devices |
| POST | /api/devices/register | 注册设备 / Register device |
| POST | /api/devices/unregister | 注销设备 / Unregister device |
| POST | /api/devices/status | 设备状态 / Device status |
| POST | /api/devices/mode | 设置模式 / Set mode |
| POST | /api/devices/emergency_stop | 紧急停止 / Emergency stop |
| POST | /api/hardware/scan | 扫描硬件 / Scan hardware |
| GET | /api/hardware/info | 硬件信息 / Hardware info |
| POST | /api/hardware/config | 配置硬件 / Configure hardware |

---

## GPU / 加速器 / GPU / Accelerator

| 方法/Method | 路径/Path | 描述/Description |
|------------|-----------|-----------------|
| GET | /api/gpu/status | GPU加速状态 / GPU acceleration status |
| GET | /api/gpu/diagnostic | GPU完整诊断 / GPU diagnostic |
| POST | /api/gpu/benchmark | GPU基准测试 / GPU benchmark |

---

## API 密钥 / API Keys

| 方法/Method | 路径/Path | 描述/Description |
|------------|-----------|-----------------|
| GET | /api/key/list | 密钥列表 / Key list |
| POST | /api/key/create | 创建密钥 / Create key |
| POST | /api/key/delete | 删除密钥 / Delete key |
| POST | /api/key/update | 更新密钥 / Update key |
| POST | /api/key/toggle | 启用/禁用 / Enable/disable key |
| POST | /api/key/set | 设置密钥 / Set key |
| GET | /api/key/status | 密钥状态 / Key status |
| GET | /api/key/stats | 调用统计 / Usage statistics |
| GET | /api/key/rate-limit | 限流状态 / Rate limit status |

---

## 技能库 / Skills

| 方法/Method | 路径/Path | 描述/Description |
|------------|-----------|-----------------|
| GET | /api/skills | 技能库列表 / Skills list |
| POST | /api/skills/search | 搜索技能 / Search skills |
| POST | /api/skills/execute | 执行技能 / Execute skill |
| POST | /api/skills/compose | 组合技能 / Compose skills |
| GET | /api/skills/stats | 技能统计 / Skills stats |

---

## 自主学习 / Auto Learning

| 方法/Method | 路径/Path | 描述/Description |
|------------|-----------|-----------------|
| POST | /api/auto-learn/scan | 扫描学习 / Scan & learn |
| GET | /api/auto-learn/stats | 自主学习统计 / Auto-learn stats |
| POST | /api/auto-learn/export | 导出学习结果 / Export learning |
| POST | /api/auto-learn/toggle | 开关自主学习 / Toggle auto-learn |

---

## 演化 / Evolution

| 方法/Method | 路径/Path | 描述/Description |
|------------|-----------|-----------------|
| POST | /api/evolution | 演化请求 / Evolution request |
| POST | /api/evolution/pareto | 帕累托前沿 / Pareto frontier |

---

## 模仿学习 / Imitation Learning

| 方法/Method | 路径/Path | 描述/Description |
|------------|-----------|-----------------|
| POST | /api/imitation/demonstration | 提交示范数据 / Submit demonstration |
| POST | /api/imitation/train | 触发训练 / Trigger training |
| POST | /api/imitation/predict | 策略预测 / Policy prediction |
| GET | /api/imitation/status | 学习状态 / Learning status |
| POST | /api/imitation/algorithm | 切换算法 / Switch algorithm |

---

## 安全 / Safety

| 方法/Method | 路径/Path | 描述/Description |
|------------|-----------|-----------------|
| GET | /api/safety/status | 安全监控状态 / Safety monitor status |
| GET | /api/safety/events | 安全事件列表 / Safety events |
| POST | /api/safety/emergency_stop | 紧急停止 / Emergency stop |
| POST | /api/safety/soft_stop | 软停止 / Soft stop |
| POST | /api/safety/reset | 重置安全状态 / Reset safety |

---

## 文件与串口 / Files & Serial

| 方法/Method | 路径/Path | 描述/Description |
|------------|-----------|-----------------|
| POST | /api/files/read | 读取文件 / Read file |
| POST | /api/files/write | 写入文件 / Write file |
| POST | /api/files/delete | 删除文件 / Delete file |
| GET | /api/files/list | 列出目录 / List directory |
| GET | /api/serial/list | 串口列表 / Serial port list |
| POST | /api/serial/open | 打开串口 / Open serial port |
| POST | /api/serial/close | 关闭串口 / Close serial port |
| POST | /api/serial/send | 发送数据 / Send serial data |

---

## HTTP 状态码 / HTTP Status Codes

| 状态码/Code | 含义/Meaning |
|-------------|-------------|
| 200 | 成功 / Success |
| 400 | 请求参数错误 / Bad request |
| 403 | 认证失败 / Auth failure |
| 413 | 请求体超过10MB限制 / Payload exceeds 10MB limit |
| 500 | 内部服务器错误 / Internal server error |
| 503 | 子模块/硬件不可用（不降级返回虚假数据）/ Module/hardware unavailable |

---

## 安全头 / Security Headers

所有 API 响应包含 / All API responses include:
- `Access-Control-Allow-Origin: *`
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Strict-Transport-Security: max-age=31536000`
- `Cache-Control: no-store`

---

## 动态架构控制器 API / Dynamic Architecture Controller API

### 架构统计查询 / Architecture Stats Query

**GET** `/api/lnn/architecture/status`

返回当前 LNN 的神经元数量、参数数量和隐藏层配置。

Returns current LNN neuron count, parameter count, and hidden layer configuration.

**响应 / Response:**
```json
{
  "neuron_count": 768,
  "param_count": 953472,
  "hidden_size": 256,
  "num_layers": 2,
  "input_size": 128,
  "output_size": 128
}
```

**错误码 / Error Codes:**
| 码 / Code | 含义 / Meaning |
|-----------|---------------|
| 200 | 成功 / Success |
| 404 | LNN 未初始化 / LNN not initialized |

---

### 架构变更历史 / Architecture Change History

**GET** `/api/lnn/architecture/history`

返回最近的架构变更历史记录（最多 128 条）。

Returns recent architecture change history (up to 128 entries).

**响应 / Response:**
```json
{
  "total_changes": 3,
  "entries": [
    {
      "timestamp": "2026-06-03 12:00:00",
      "type": "EXPAND_HIDDEN",
      "old_neurons": 768,
      "new_neurons": 1280,
      "reason": "Self-cognition detected capacity shortage"
    }
  ]
}
```

---

*文档版本 / Doc Version: 2026-06-03 | 基于 backend.c 真实端点注册表 / Based on backend.c endpoint registry*
