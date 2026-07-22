$BASE = "http://127.0.0.1:8080"
$pass = 0; $fail = 0; $results = @()

function Test-GET($path) {
    try {
        $r = Invoke-WebRequest -Uri "$BASE$path" -Method GET -UseBasicParsing -TimeoutSec 5 -ErrorAction Stop
        if ($r.StatusCode -eq 200) { $script:pass++ } 
        else { $script:fail++; $script:results += "$path -> $($r.StatusCode)" }
    } catch {
        $sc = if ($_.Exception.Response) { $_.Exception.Response.StatusCode.value__ } else { "ERR" }
        $script:fail++; $script:results += "$path -> $sc"
    }
}

function Test-POST($path, $body) {
    try {
        $r = Invoke-WebRequest -Uri "$BASE$path" -Method POST -Body $body -ContentType "application/json" -UseBasicParsing -TimeoutSec 5 -ErrorAction Stop
        if ($r.StatusCode -eq 200) { $script:pass++ }
        else { $script:fail++; $script:results += "$path -> $($r.StatusCode)" }
    } catch {
        $sc = if ($_.Exception.Response) { $_.Exception.Response.StatusCode.value__ } else { "ERR" }
        $script:fail++; $script:results += "$path -> $sc"
    }
}

# 系统状态
Test-GET "/api/status"
Test-GET "/api/health"
Test-GET "/api/stats"
Test-GET "/api/version"
Test-GET "/api/system/diagnostic"
Test-GET "/api/system/logs"
Test-GET "/api/system/info"
Test-GET "/api/metrics"
Test-GET "/api/config"
Test-GET "/api/config/list"
Test-GET "/api/system/full_status"
Test-GET "/api/usage-logs"

# 多模态
Test-POST "/api/vision" '{"image":"test"}'
Test-POST "/api/audio" '{"audio":"test"}'
Test-POST "/api/text" '{"text":"test"}'
Test-POST "/api/sensor" '{"sensor":"test"}'
Test-GET "/api/sensor/list"
Test-GET "/api/sensor/pipeline/status"
Test-GET "/api/multimodal/status"
Test-GET "/api/multimodal/modalities"
Test-POST "/api/multimodal/config" '{}'
Test-POST "/api/multimodal/test" '{}'
Test-GET "/api/audio/devices"
Test-GET "/api/camera/devices"

# 对话
Test-GET "/api/dialogue"
Test-POST "/api/dialogue" '{"message":"你好"}'
Test-GET "/api/dialogue/history"
Test-POST "/api/dialogue/send" '{"message":"测试"}'
Test-POST "/api/dialogue/chat" '{"message":"测试"}'
Test-GET "/api/voice/history"

# 知识库
Test-GET "/api/knowledge"
Test-GET "/api/knowledge/stats"
Test-GET "/api/knowledge/categories"
Test-GET "/api/knowledge/status"
Test-GET "/api/knowledge/list"
Test-GET "/api/knowledge/count"
Test-POST "/api/knowledge/search" '{"query":"test"}'
Test-GET "/api/knowledge/version"
Test-POST "/api/knowledge/add" '{"subject":"test","predicate":"is","object":"value"}'
Test-GET "/api/knowledge/entry?id=1"

# 知识图谱
Test-GET "/api/kg/stats"
Test-GET "/api/kg/pagerank"
Test-GET "/api/kg/communities"
Test-POST "/api/kg/path" '{"source":1,"target":2}'
Test-POST "/api/kg/search" '{"query":"test"}'
Test-GET "/api/kg/visualize"
Test-GET "/api/kg/betweenness"
Test-GET "/api/kg/closeness"
Test-GET "/api/kg/diameter"
Test-GET "/api/kg/clustering"

# 记忆
Test-GET "/api/memory"
Test-GET "/api/memory/stats"
Test-GET "/api/memory/usage"
Test-POST "/api/memory/search" '{"query":"test"}'
Test-POST "/api/memory/add" '{"content":"test"}'
Test-GET "/api/memory/entry?id=1"
Test-POST "/api/memory/export" '{}'
Test-POST "/api/memory/save" '{}'

# LNN
Test-GET "/api/lnn/status"
Test-GET "/api/lnn/info"
Test-POST "/api/lnn/parameters" '{"params":[]}'
Test-GET "/api/lnn/params"
Test-POST "/api/lnn/config" '{}'
Test-GET "/api/lnn/activation/heatmap"
Test-GET "/api/lnn/prediction/scatter"

# 训练
Test-GET "/api/training/status"
Test-GET "/api/training/history"
Test-POST "/api/training/meta" '{}'
Test-POST "/api/training/start" '{}'
Test-POST "/api/training/pipeline" '{}'
Test-POST "/api/training/schedule" '{}'
Test-POST "/api/training/from-scratch" '{}'
Test-POST "/api/training/pretrain" '{}'
Test-POST "/api/training/fine-tune" '{}'

# 演化
Test-POST "/api/evolution" '{}'
Test-GET "/api/evolution/status"
Test-GET "/api/evolution/history"
Test-POST "/api/evolution/step" '{}'
Test-POST "/api/evolution/trigger" '{}'
Test-POST "/api/evolution/pareto" '{}'

# AGI
Test-GET "/api/agi/status"
Test-GET "/api/agi/features"
Test-GET "/api/agi/feature_list"
Test-GET "/api/agi/feature-list"
Test-GET "/api/agi/cognition/state"
Test-GET "/api/agi/tasks"
Test-GET "/api/agi/diagnostic"
Test-POST "/api/agi/think" '{"query":"分析系统状态"}'
Test-POST "/api/agi/decide" '{"context":"优化"}'
Test-POST "/api/agi/plan" '{"goal":"测试"}'
Test-POST "/api/agi/analyze" '{"query":"test"}'
Test-POST "/api/agi/self_reflect" '{"query":"自我反思"}'
Test-POST "/api/agi/execute" '{"task":"测试任务","type":"general"}'
Test-POST "/api/agi/learn" '{"data":"1,2,3","target":"4,5,6"}'

# 机器人
Test-GET "/api/robot/status"
Test-GET "/api/robot/list"
Test-POST "/api/robot/command" '{"cmd":"test"}'
Test-GET "/api/robot/sensor"
Test-POST "/api/robot/control" '{"cmd":"test"}'
Test-POST "/api/robot/plan" '{}'
Test-GET "/api/fleet/status"
Test-POST "/api/robot/connect" '{"id":1}'
Test-POST "/api/robot/register" '{"name":"test"}'
Test-POST "/api/robot/params" '{}'
Test-POST "/api/robot/calibrate" '{}'

# 安全
Test-GET "/api/safety/status"
Test-GET "/api/safety/events"
Test-GET "/api/safety/bounds"
Test-GET "/api/safety/policies"
Test-GET "/api/safety/config"
Test-POST "/api/safety/check" '{}'
Test-POST "/api/safety/scan" '{}'
Test-POST "/api/safety/emergency_stop" '{}'

# 自我认知
Test-GET "/api/self/status"
Test-GET "/api/self/cognition"
Test-GET "/api/self/identity"
Test-POST "/api/self/reflect" '{"query":"自我评估"}'
Test-GET "/api/cognition/state"
Test-POST "/api/cognition/reflect" '{"query":"认知反思"}'
Test-GET "/api/cognition/tom"
Test-GET "/api/cognition/health"
Test-POST "/api/cognition/self" '{"query":"test"}'

# 元认知
Test-GET "/api/metacognition/state"
Test-POST "/api/metacognition/reflect" '{"query":"test"}'
Test-POST "/api/metacognition/calibrate" '{}'

# 学习
Test-GET "/api/learning"
Test-GET "/api/learning/status"
Test-GET "/api/learning/progress"
Test-GET "/api/learning/experiences"
Test-POST "/api/learning/from-dialogue" '{}'
Test-POST "/api/learning/consistency" '{}'

# 推理
Test-GET "/api/reasoning"
Test-GET "/api/reasoning/status"
Test-GET "/api/reasoning/test"
Test-POST "/api/reasoning/start" '{}'
Test-POST "/api/reasoning/symbolic" '{}'

# 因果推理
Test-POST "/api/causal/build" '{}'
Test-POST "/api/causal/infer" '{}'
Test-POST "/api/causal/discover" '{}'

# 技能
Test-GET "/api/skills"
Test-GET "/api/skills/status"
Test-POST "/api/skills/search" '{"query":"test"}'
Test-POST "/api/skills/execute" '{"skill":"test"}'
Test-GET "/api/skills/stats"

# 设备
Test-GET "/api/devices/list"
Test-GET "/api/device/list"
Test-GET "/api/devices/discover"
Test-POST "/api/device/command" '{}'
Test-POST "/api/device/control" '{}'
Test-POST "/api/device/status" '{}'

# 硬件/GPU
Test-GET "/api/hardware/info"
Test-GET "/api/hardware/resources"
Test-POST "/api/hardware/scan" '{}'
Test-GET "/api/gpu/status"
Test-GET "/api/gpu/info"
Test-GET "/api/gpu/diagnostic"

# 模型
Test-GET "/api/model/info"
Test-POST "/api/model/load" '{}'
Test-POST "/api/model/export" '{}'
Test-POST "/api/model/start" '{}'
Test-POST "/api/model/stop" '{}'
Test-GET "/api/model/parallel"
Test-POST "/api/checkpoint/save" '{}'
Test-GET "/api/checkpoint/list"

# 仿真
Test-GET "/api/simulation/status"
Test-POST "/api/simulation/start" '{}'
Test-POST "/api/simulation/command" '{}'
Test-POST "/api/simulation/plan_path" '{}'
Test-POST "/api/simulation/robot_control" '{}'
Test-GET "/api/slam/status"
Test-GET "/api/stereo/status"

# 规划
Test-POST "/api/plan/generate" '{"goal":"test"}'
Test-POST "/api/plan/execute" '{}'
Test-POST "/api/plan/multi-objective" '{}'
Test-POST "/api/plan/mcts" '{}'

# 编程
Test-GET "/api/programming/status"
Test-GET "/api/programming/sample"
Test-GET "/api/programming/projects"
Test-POST "/api/programming/analyze" '{}'
Test-POST "/api/programming/generate" '{}'

# 数据集
Test-GET "/api/dataset/list"
Test-GET "/api/dataset/stats"
Test-POST "/api/dataset/create" '{}'
Test-GET "/api/datasets"

# 产品设计
Test-GET "/api/product/status"
Test-POST "/api/product/design" '{}'
Test-GET "/api/product/spec"
Test-POST "/api/product/generate" '{}'

# 自动学习
Test-POST "/api/auto-learn/scan" '{}'
Test-GET "/api/auto-learn/stats"
Test-POST "/api/auto-learn/start" '{}'
Test-GET "/api/auto-learn/export"

# 超参数
Test-GET "/api/hyperparameter/status"
Test-POST "/api/hyperparameter/start" '{}'
Test-POST "/api/hyperparameter/tune" '{}'

# 决策
Test-GET "/api/decision/log"
Test-POST "/api/decision/make" '{"context":"test"}'

# 模仿学习
Test-GET "/api/imitation/status"
Test-POST "/api/imitation/demonstration" '{}'
Test-POST "/api/imitation/predict" '{"state":[1,2],"state_dim":2,"action_dim":2}'

# 教学
Test-POST "/api/teach/look_and_learn" '{}'
Test-GET "/api/teach/get_concepts"

# 串口
Test-GET "/api/serial/list"

# 文件
Test-GET "/api/files/list"

# ROS
Test-GET "/api/ros/status"
Test-GET "/api/ros/nodes"
Test-GET "/api/ros/topics"

# 多智能体
Test-GET "/api/multi-agent/status"

# 多系统
Test-GET "/api/multi-system/topology"

# 语义网络
Test-GET "/api/semantic/importance"

# 能力
Test-GET "/api/capability/diagnose"
Test-GET "/api/capability/health"
Test-GET "/api/capability/list"

# 密钥
Test-GET "/api/key/list"
Test-GET "/api/auth/status"
Test-GET "/api/auth/set_key"
Test-GET "/api/key/status"
Test-GET "/api/key/stats"

# 任务
Test-GET "/api/task/queue"
Test-POST "/api/task/create" '{"name":"test"}'

# 计算机操作
Test-POST "/api/computer/launch" '{}'
Test-POST "/api/computer/close" '{}'

# 其他
Test-GET "/api/command/prefixes"
Test-GET "/api/api/docs"
Test-GET "/api/log/recent"
Test-GET "/api/resources/usage"
Test-POST "/api/command/send" '{}'
Test-GET "/api/swarm/status"
Test-POST "/api/sensor/calibrate" '{}'
Test-POST "/api/stereo/perception" '{}'
Test-POST "/api/laplace/spectrum" '{}'
Test-POST "/api/laplace/adaptive-lr" '{}'

Write-Host "====== 测试结果 ======"
Write-Host "通过: $pass, 失败: $fail"
if ($results.Count -gt 0) {
    Write-Host "失败列表:"
    $results | ForEach-Object { Write-Host "  $_" }
}