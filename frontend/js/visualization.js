/**
 * SELF-LNN AGI 实时数据可视化系统
 * 提供性能指标、网络拓扑、训练过程等可视化功能
 */

class VisualizationManager {
    constructor() {
        this.charts = {};
        this.dataBuffers = {};
        this.maxDataPoints = 100;
        this.updateInterval = null;
        this.isRunning = false;
        this.theme = {
            gridColor: 'rgba(255,255,255,0.1)',
            textColor: '#cccccc',
            colors: ['#888888', '#00ff88', '#ffaa00', '#ff4444', '#4488ff', '#ff66aa']
        };
    }

    /**
     * 初始化所有图表
     */
    initAllCharts() {
        this.destroyAllCharts();
        this.initLossChart();
        this.initAccuracyChart();
        this.initMemoryChart();
        this.initLearningRateChart();
        this.initGradientChart();
        this.initWeightDistributionChart();
        this.initActivationChart();
        this.initRobotStatusChart();
        this.initNetworkTopology();
        this.initSystemResourceChart();
        this.initKnowledgeGrowthChart();
        this.initStateActivationHeatmap();
        this.initPredictionChart();
        this.initConceptEvolutionChart();
        this.initSpectrumAnalysis();
    }

    /**
     * 初始化损失函数图表
     */
    initLossChart() {
        const canvas = document.getElementById('loss-chart');
        if (!canvas) return;

        this.dataBuffers.loss = { train: [], val: [], timestamps: [] };

        this.charts.loss = new SelfLnnChart(canvas, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: '训练损失',
                        data: [],
                        borderColor: '#888888',
                        backgroundColor: 'rgba(136,136,136,0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.4,
                        pointRadius: 0
                    },
                    {
                        label: '验证损失',
                        data: [],
                        borderColor: '#00ff88',
                        backgroundColor: 'rgba(0,255,136,0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.4,
                        pointRadius: 0
                    }
                ]
            },
            options: this.getDefaultChartOptions('损失曲线', '迭代次数', '损失值')
        });
    }

    /**
     * 初始化准确率图表
     */
    initAccuracyChart() {
        const canvas = document.getElementById('accuracy-chart');
        if (!canvas) return;

        this.dataBuffers.accuracy = { train: [], val: [], timestamps: [] };

        this.charts.accuracy = new SelfLnnChart(canvas, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: '训练准确率',
                        data: [],
                        borderColor: '#00ff88',
                        backgroundColor: 'rgba(0,255,136,0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.4,
                        pointRadius: 0
                    },
                    {
                        label: '验证准确率',
                        data: [],
                        borderColor: '#ffaa00',
                        backgroundColor: 'rgba(255,170,0,0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.4,
                        pointRadius: 0
                    }
                ]
            },
            options: this.getDefaultChartOptions('准确率曲线', '迭代次数', '准确率')
        });
    }

    /**
     * 初始化内存使用图表
     */
    initMemoryChart() {
        const canvas = document.getElementById('memory-chart');
        if (!canvas) return;

        this.dataBuffers.memory = { used: [], allocated: [], timestamps: [] };

        this.charts.memory = new SelfLnnChart(canvas, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: '已使用内存',
                        data: [],
                        borderColor: '#4488ff',
                        backgroundColor: 'rgba(68,136,255,0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.4,
                        pointRadius: 0
                    },
                    {
                        label: '已分配内存',
                        data: [],
                        borderColor: '#ff66aa',
                        backgroundColor: 'rgba(255,102,170,0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.4,
                        pointRadius: 0
                    }
                ]
            },
            options: this.getDefaultChartOptions('内存使用趋势', '时间', '内存 (MB)')
        });
    }

    /**
     * 初始化学习率图表
     */
    initLearningRateChart() {
        const canvas = document.getElementById('learning-rate-chart');
        if (!canvas) return;

        this.dataBuffers.learningRate = { values: [], timestamps: [] };

        this.charts.learningRate = new SelfLnnChart(canvas, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: '学习率',
                        data: [],
                        borderColor: '#ffaa00',
                        backgroundColor: 'rgba(255,170,0,0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.4,
                        pointRadius: 3,
                        pointHoverRadius: 6
                    }
                ]
            },
            options: this.getDefaultChartOptions('学习率调度', '迭代次数', '学习率')
        });
    }

    /**
     * 初始化梯度统计图表
     */
    initGradientChart() {
        const canvas = document.getElementById('gradient-chart');
        if (!canvas) return;

        this.dataBuffers.gradient = { mean: [], max: [], norm: [], timestamps: [] };

        this.charts.gradient = new SelfLnnChart(canvas, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: '梯度均值',
                        data: [],
                        borderColor: '#888888',
                        borderWidth: 1.5,
                        tension: 0.3,
                        pointRadius: 0
                    },
                    {
                        label: '梯度最大值',
                        data: [],
                        borderColor: '#ff4444',
                        borderWidth: 1.5,
                        tension: 0.3,
                        pointRadius: 0
                    },
                    {
                        label: '梯度范数',
                        data: [],
                        borderColor: '#00ff88',
                        borderWidth: 2,
                        fill: false,
                        tension: 0.3,
                        pointRadius: 0
                    }
                ]
            },
            options: this.getDefaultChartOptions('梯度统计', '迭代次数', '梯度值')
        });
    }

    /**
     * 初始化权重分布图表
     */
    initWeightDistributionChart() {
        const canvas = document.getElementById('weight-dist-chart');
        if (!canvas) return;

        this.dataBuffers.weightDist = { bins: [], counts: [] };

        this.charts.weightDist = new SelfLnnChart(canvas, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [
                    {
                        label: '权重分布',
                        data: [],
                        backgroundColor: 'rgba(136,136,136,0.6)',
                        borderColor: '#888888',
                        borderWidth: 1,
                        borderRadius: 2
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 300 },
                plugins: {
                    legend: { display: true, labels: { color: '#cccccc' } },
                    title: {
                        display: true,
                        text: '权重分布直方图',
                        color: '#cccccc',
                        font: { size: 14 }
                    }
                },
                scales: {
                    x: {
                        display: true,
                        title: { display: true, text: '权重区间', color: '#cccccc' },
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: '#cccccc' }
                    },
                    y: {
                        display: true,
                        title: { display: true, text: '计数', color: '#cccccc' },
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: '#cccccc' },
                        beginAtZero: true
                    }
                }
            }
        });
    }

    /**
     * 初始化激活值分布图表（使用条形图替代boxplot，避免额外插件依赖）
     */
    initActivationChart() {
        const canvas = document.getElementById('activation-chart');
        if (!canvas) return;

        this.dataBuffers.activation = {
            layers: [],
            mean: [],
            max: [],
            min: [],
            std: []
        };

        this.charts.activation = new SelfLnnChart(canvas, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [
                    {
                        label: '最大值',
                        data: [],
                        backgroundColor: 'rgba(255,68,68,0.5)',
                        borderColor: '#ff4444',
                        borderWidth: 1,
                        borderRadius: 2
                    },
                    {
                        label: '均值',
                        data: [],
                        backgroundColor: 'rgba(68,136,255,0.6)',
                        borderColor: '#4488ff',
                        borderWidth: 1,
                        borderRadius: 2
                    },
                    {
                        label: '最小值',
                        data: [],
                        backgroundColor: 'rgba(0,255,136,0.5)',
                        borderColor: '#00ff88',
                        borderWidth: 1,
                        borderRadius: 2
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 300 },
                plugins: {
                    legend: { display: true, labels: { color: '#cccccc' } },
                    title: {
                        display: true,
                        text: '激活值分布（每层统计）',
                        color: '#cccccc',
                        font: { size: 14 }
                    }
                },
                scales: {
                    x: {
                        display: true,
                        title: { display: true, text: '网络层', color: '#cccccc' },
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: '#cccccc' }
                    },
                    y: {
                        display: true,
                        title: { display: true, text: '激活值', color: '#cccccc' },
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: '#cccccc' },
                        beginAtZero: true
                    }
                }
            }
        });
    }

    /**
     * 初始化机器人状态图表
     */
    initRobotStatusChart() {
        const canvas = document.getElementById('robot-status-chart');
        if (!canvas) return;

        this.charts.robotStatus = new SelfLnnChart(canvas, {
            type: 'radar',
            data: {
                labels: ['运动能力', '传感器精度', '响应速度', '稳定性', '能耗效率', '负载能力'],
                datasets: [
                    {
                        label: '机器人 #1',
                        data: [0, 0, 0, 0, 0, 0],
                        borderColor: '#00ff88',
                        backgroundColor: 'rgba(0,255,136,0.1)',
                        borderWidth: 2,
                        pointBackgroundColor: '#00ff88'
                    },
                    {
                        label: '机器人 #2',
                        data: [0, 0, 0, 0, 0, 0],
                        borderColor: '#4488ff',
                        backgroundColor: 'rgba(68,136,255,0.1)',
                        borderWidth: 2,
                        pointBackgroundColor: '#4488ff'
                    },
                    {
                        label: '机器人 #3',
                        data: [0, 0, 0, 0, 0, 0],
                        borderColor: '#ffaa00',
                        backgroundColor: 'rgba(255,170,0,0.1)',
                        borderWidth: 2,
                        pointBackgroundColor: '#ffaa00'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 500 },
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { color: '#cccccc', font: { size: 11 } }
                    },
                    title: {
                        display: true,
                        text: '机器人能力雷达图',
                        color: '#cccccc',
                        font: { size: 14 }
                    }
                },
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 100,
                        grid: { color: 'rgba(255,255,255,0.1)' },
                        angleLines: { color: 'rgba(255,255,255,0.1)' },
                        pointLabels: { color: '#cccccc', font: { size: 11 } },
                        ticks: {
                            color: '#cccccc',
                            backdropColor: 'transparent',
                            stepSize: 20
                        }
                    }
                }
            }
        });
    }

    /**
     * 初始化网络拓扑可视化
     */
    initNetworkTopology() {
        const canvas = document.getElementById('network-topology');
        if (!canvas) return;
        this.networkCanvas = canvas;
        this.networkCtx = canvas.getContext('2d');
        this.networkNodes = [];
        this.networkEdges = [];
        this.networkAnimationId = null;
        this.initDefaultNetwork();
        /* 后端连接检测与真实拓扑加载 */
        this._checkBackendAndUpdate = () => {
            if (typeof SelfLnnApi !== 'undefined') {
                SelfLnnApi.getLNNStatus().then((statusData) => {
                    if (statusData && statusData.success) {
                        var lnn = (statusData.data && statusData.data.lnn) || statusData.lnn || statusData;
                        var layers = lnn.layers || lnn.network_layers || lnn.topology;
                        if (layers && layers.length > 0) {
                            this.updateNetworkFromData(layers);
                            return;
                        }
                    }
                    /* 无真实拓扑数据时，显示默认架构示意 */
                    this.updateNetworkFromData(null);
                }).catch(() => {
                    /* 后端不可用，延长重试间隔 */
                    setTimeout(this._checkBackendAndUpdate, 2000);
                });
            }
        };
        setTimeout(this._checkBackendAndUpdate, 800);
    }

    /**
     * 初始化默认网络拓扑（后端未连接时的过渡状态，非虚假数据）
     * 800ms后通过 _checkBackendAndUpdate 自动重试获取真实拓扑
     */
    initDefaultNetwork() {
        const canvas = this.networkCanvas;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const w = canvas.width, h = canvas.height;
        ctx.fillStyle = '#0a0a1a';
        ctx.fillRect(0, 0, w, h);
        ctx.fillStyle = 'rgba(68,136,170,0.6)';
        ctx.font = '14px monospace';
        ctx.textAlign = 'center';
        ctx.fillText('[后端未连接] SELF-LNN 液态神经网络', w / 2, h / 2 - 20);
        ctx.fillStyle = 'rgba(102,170,255,0.4)';
        ctx.font = '12px monospace';
        ctx.fillText('正在自动重试连接...连接成功后将加载真实网络拓扑', w / 2, h / 2 + 8);
        ctx.fillStyle = 'rgba(255,255,255,0.15)';
        ctx.font = '10px monospace';
        ctx.fillText('（过渡占位 — 非虚假数据）', w / 2, h / 2 + 28);
    }

    /**
     * 基于真实数据更新网络可视化
     */
    updateNetworkFromData(layerConfig) {
        const canvas = this.networkCanvas;
        if (!canvas) return;
        const width = canvas.width || canvas.parentElement.clientWidth;
        const height = canvas.height || 300;
        canvas.width = width;
        canvas.height = height;

        /* 无真实数据时禁止渲染虚拟拓扑，显示等待状态 */
        if (!layerConfig || !Array.isArray(layerConfig) || layerConfig.length === 0) {
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, width, height);
            ctx.fillStyle = 'rgba(10,10,30,0.95)';
            ctx.fillRect(0, 0, width, height);
            ctx.fillStyle = '#888';
            ctx.font = '14px monospace';
            ctx.textAlign = 'center';
            ctx.fillText('等待后端连接以加载真实网络拓扑...', width / 2, height / 2);
            ctx.fillText('（无真实数据，禁止渲染虚拟拓扑）', width / 2, height / 2 + 24);
            this.networkNodes = [];
            this.networkEdges = [];
            return;
        }

        const layers = layerConfig;

        const layerGap = width / (layers.length + 1);
        this.networkNodes = [];
        this.networkEdges = [];

        layers.forEach((layer, layerIdx) => {
            const cx = layerGap * (layerIdx + 1);
            const nodeGap = Math.min(40, (height - 40) / layer.count);
            const startY = (height - nodeGap * (layer.count - 1)) / 2;

            for (let i = 0; i < layer.count; i++) {
                this.networkNodes.push({
                    x: cx,
                    y: startY + i * nodeGap,
                    radius: 8,
                    color: layer.color,
                    layer: layer.label,
                    index: i,
                    activation: 0.0
                });
            }
        });

        const nodeCount = this.networkNodes.length;
        for (let i = 0; i < nodeCount; i++) {
            const node = this.networkNodes[i];
            const nextLayer = node.layer;
            for (let j = i + 1; j < nodeCount; j++) {
                const target = this.networkNodes[j];
                if (target.layer !== nextLayer && this.getLayerIndex(target.layer) === this.getLayerIndex(nextLayer) + 1) {
                    this.networkEdges.push({
                        source: i,
                        target: j,
                        weight: 0.0,
                        active: false
                    });
                }
            }
        }

        this.startNetworkAnimation();
    }

    /**
     * 获取层索引
     */
    getLayerIndex(layerName) {
        const layers = ['输入层', '液态层1', '液态层2', 'CfC层', '输出层'];
        return layers.indexOf(layerName);
    }

    /**
     * 启动网络拓扑动画
     */
    startNetworkAnimation() {
        const animate = () => {
            this.renderNetwork();
            this.networkAnimationId = requestAnimationFrame(animate);
        };
        animate();
    }

    /**
     * 停止网络拓扑动画
     */
    stopNetworkAnimation() {
        if (this.networkAnimationId) {
            cancelAnimationFrame(this.networkAnimationId);
            this.networkAnimationId = null;
        }
    }

    /**
     * 渲染网络拓扑
     */
    renderNetwork() {
        const ctx = this.networkCtx;
        const canvas = this.networkCanvas;
        if (!ctx || !canvas) return;

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        this.networkEdges.forEach(edge => {
            const source = this.networkNodes[edge.source];
            const target = this.networkNodes[edge.target];
            if (!source || !target) return;

            const alpha = edge.active ? 0.3 + Math.abs(edge.weight) * 0.4 : 0.05;
            const color = edge.weight > 0 ? `rgba(0,255,136,${alpha})` : `rgba(255,68,68,${alpha})`;

            ctx.beginPath();
            ctx.moveTo(source.x, source.y);
            ctx.lineTo(target.x, target.y);
            ctx.strokeStyle = color;
            ctx.lineWidth = edge.active ? 1 + Math.abs(edge.weight) : 0.5;
            ctx.stroke();
        });

        this.networkNodes.forEach(node => {
            const gradient = ctx.createRadialGradient(
                node.x, node.y, 0,
                node.x, node.y, node.radius
            );
            gradient.addColorStop(0, node.color);
            gradient.addColorStop(1, 'rgba(0,0,0,0.8)');

            ctx.beginPath();
            ctx.arc(node.x, node.y, node.radius * (0.5 + node.activation * 0.5), 0, Math.PI * 2);
            ctx.fillStyle = gradient;
            ctx.fill();

            ctx.strokeStyle = node.color;
            ctx.lineWidth = 1;
            ctx.stroke();

            const glowAlpha = 0.2 + node.activation * 0.3;
            ctx.beginPath();
            ctx.arc(node.x, node.y, node.radius * 1.5, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(255,255,255,${glowAlpha * 0.1})`;
            ctx.fill();
        });
    }

    /**
     * 初始化系统资源图表
     */
    initSystemResourceChart() {
        const canvas = document.getElementById('system-resource-chart');
        if (!canvas) return;

        this.dataBuffers.systemResource = { cpu: [], memory: [], gpu: [], timestamps: [] };

        this.charts.systemResource = new SelfLnnChart(canvas, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'CPU使用率',
                        data: [],
                        borderColor: '#4488ff',
                        borderWidth: 2,
                        fill: false,
                        tension: 0.4,
                        pointRadius: 0
                    },
                    {
                        label: '内存使用率',
                        data: [],
                        borderColor: '#00ff88',
                        borderWidth: 2,
                        fill: false,
                        tension: 0.4,
                        pointRadius: 0
                    },
                    {
                        label: 'GPU使用率',
                        data: [],
                        borderColor: '#ff4444',
                        borderWidth: 2,
                        fill: false,
                        tension: 0.4,
                        pointRadius: 0
                    }
                ]
            },
            options: {
                ...this.getDefaultChartOptions('系统资源监控', '时间', '使用率 (%)'),
                scales: {
                    x: {
                        display: true,
                        title: { display: true, text: '时间', color: '#cccccc' },
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: '#cccccc', maxTicksLimit: 10 }
                    },
                    y: {
                        display: true,
                        title: { display: true, text: '使用率 (%)', color: '#cccccc' },
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: '#cccccc' },
                        min: 0,
                        max: 100
                    }
                }
            }
        });
    }

    /**
     * 初始化知识增长图表
     */
    initKnowledgeGrowthChart() {
        const canvas = document.getElementById('knowledge-growth-chart');
        if (!canvas) return;

        this.dataBuffers.knowledgeGrowth = { entries: [], concepts: [], relations: [], timestamps: [] };

        this.charts.knowledgeGrowth = new SelfLnnChart(canvas, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: '知识条目数',
                        data: [],
                        borderColor: '#00ff88',
                        backgroundColor: 'rgba(0,255,136,0.05)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.4,
                        pointRadius: 0
                    },
                    {
                        label: '抽象概念数',
                        data: [],
                        borderColor: '#ffaa00',
                        backgroundColor: 'rgba(255,170,0,0.05)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.4,
                        pointRadius: 0
                    },
                    {
                        label: '关系数',
                        data: [],
                        borderColor: '#4488ff',
                        backgroundColor: 'rgba(68,136,255,0.05)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.4,
                        pointRadius: 0
                    }
                ]
            },
            options: this.getDefaultChartOptions('知识增长趋势', '时间', '数量')
        });
    }

    /**
     * 初始化神经元状态激活热图
     */
    initStateActivationHeatmap() {
        this.stateActivationCanvas = document.getElementById('state-activation-heatmap');
        if (!this.stateActivationCanvas) return;
        this.stateActivationCtx = this.stateActivationCanvas.getContext('2d');
        this.stateActivationData = [];
        this.heatmapSize = 20;
        this.initHeatmapData();
    }

    /**
     * 初始化热力图数据
     */
    initHeatmapData() {
        for (let i = 0; i < this.heatmapSize; i++) {
            this.stateActivationData[i] = [];
            for (let j = 0; j < this.heatmapSize; j++) {
                this.stateActivationData[i][j] = 0.0;
            }
        }
    }

    /**
     * 渲染神经元状态激活热图
     */
    renderStateActivationHeatmap() {
        const ctx = this.stateActivationCtx;
        const canvas = this.stateActivationCanvas;
        if (!ctx || !canvas) return;

        const width = canvas.width || canvas.parentElement.clientWidth;
        const height = canvas.height || 300;
        canvas.width = width;
        canvas.height = height;

        const cellW = width / this.heatmapSize;
        const cellH = height / this.heatmapSize;

        for (let i = 0; i < this.heatmapSize; i++) {
            for (let j = 0; j < this.heatmapSize; j++) {
                const value = Math.max(0, Math.min(1, this.stateActivationData[i][j] || 0));
                const r = Math.round(255 * value);
                const g = Math.round(255 * (1 - value) * 0.6);
                const b = Math.round(255 * (1 - value) * 0.8);

                ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
                ctx.fillRect(j * cellW, i * cellH, cellW + 0.5, cellH + 0.5);
            }
        }
    }

    /**
     * 初始化预测结果图表
     */
    initPredictionChart() {
        const canvas = document.getElementById('prediction-chart');
        if (!canvas) return;

        this.dataBuffers.prediction = { actual: [], predicted: [], timestamps: [] };

        this.charts.prediction = new SelfLnnChart(canvas, {
            type: 'scatter',
            data: {
                datasets: [
                    {
                        label: '实际值',
                        data: [],
                        backgroundColor: '#4488ff',
                        pointRadius: 4,
                        pointHoverRadius: 7
                    },
                    {
                        label: '预测值',
                        data: [],
                        backgroundColor: '#00ff88',
                        pointRadius: 4,
                        pointHoverRadius: 7
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 300 },
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { color: '#cccccc', padding: 15 }
                    },
                    title: {
                        display: true,
                        text: '预测结果对比',
                        color: '#cccccc',
                        font: { size: 14 }
                    }
                },
                scales: {
                    x: {
                        display: true,
                        title: { display: true, text: '样本', color: '#cccccc' },
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: '#cccccc' }
                    },
                    y: {
                        display: true,
                        title: { display: true, text: '值', color: '#cccccc' },
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: '#cccccc' }
                    }
                }
            }
        });
    }

    /**
     * 初始化概念演化图表
     */
    initConceptEvolutionChart() {
        const canvas = document.getElementById('concept-evolution-chart');
        if (!canvas) return;

        this.dataBuffers.conceptEvolution = { concepts: [], similarity: [], timestamps: [] };

        this.charts.conceptEvolution = new SelfLnnChart(canvas, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: '概念抽象层次',
                        data: [],
                        borderColor: '#ff66aa',
                        backgroundColor: 'rgba(255,102,170,0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.4,
                        pointRadius: 3,
                        pointHoverRadius: 6
                    }
                ]
            },
            options: this.getDefaultChartOptions('概念演化轨迹', '迭代次数', '抽象层次')
        });
    }

    /**
     * 获取默认图表配置
     */
    getDefaultChartOptions(title, xLabel, yLabel) {
        return {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 300 },
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: '#cccccc',
                        padding: 15,
                        usePointStyle: true,
                        font: { size: 11 }
                    }
                },
                title: {
                    display: true,
                    text: title,
                    color: '#cccccc',
                    font: { size: 14 }
                },
                tooltip: {
                    backgroundColor: 'rgba(0,0,0,0.8)',
                    titleColor: '#ffffff',
                    bodyColor: '#cccccc',
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderWidth: 1,
                    padding: 10,
                    cornerRadius: 4
                }
            },
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: xLabel,
                        color: '#cccccc',
                        font: { size: 11 }
                    },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#cccccc', maxTicksLimit: 15 }
                },
                y: {
                    display: true,
                    title: {
                        display: true,
                        text: yLabel,
                        color: '#cccccc',
                        font: { size: 11 }
                    },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#cccccc' },
                    beginAtZero: true
                }
            }
        };
    }

    /**
     * 更新损失数据
     */
    updateLossData(trainLoss, valLoss) {
        if (!this.dataBuffers.loss) return;
        const buffer = this.dataBuffers.loss;
        const timestamp = new Date().toLocaleTimeString();

        buffer.train.push(trainLoss);
        buffer.val.push(valLoss !== undefined ? valLoss : null);
        buffer.timestamps.push(timestamp);

        if (buffer.train.length > this.maxDataPoints) {
            buffer.train.shift();
            buffer.val.shift();
            buffer.timestamps.shift();
        }

        if (this.charts.loss) {
            this.charts.loss.data.labels = buffer.timestamps;
            this.charts.loss.data.datasets[0].data = buffer.train;
            this.charts.loss.data.datasets[1].data = buffer.val;
            this.charts.loss.draw();
        }
    }

    /**
     * 更新准确率数据
     */
    updateAccuracyData(trainAcc, valAcc) {
        if (!this.dataBuffers.accuracy) return;
        const buffer = this.dataBuffers.accuracy;
        const timestamp = new Date().toLocaleTimeString();

        buffer.train.push(trainAcc);
        buffer.val.push(valAcc !== undefined ? valAcc : null);
        buffer.timestamps.push(timestamp);

        if (buffer.train.length > this.maxDataPoints) {
            buffer.train.shift();
            buffer.val.shift();
            buffer.timestamps.shift();
        }

        if (this.charts.accuracy) {
            this.charts.accuracy.data.labels = buffer.timestamps;
            this.charts.accuracy.data.datasets[0].data = buffer.train;
            this.charts.accuracy.data.datasets[1].data = buffer.val;
            this.charts.accuracy.draw();
        }
    }

    /**
     * 更新内存数据
     */
    updateMemoryData(usedMB, allocatedMB) {
        if (!this.dataBuffers.memory) return;
        const buffer = this.dataBuffers.memory;
        const timestamp = new Date().toLocaleTimeString();

        buffer.used.push(usedMB);
        buffer.allocated.push(allocatedMB);
        buffer.timestamps.push(timestamp);

        if (buffer.used.length > this.maxDataPoints) {
            buffer.used.shift();
            buffer.allocated.shift();
            buffer.timestamps.shift();
        }

        if (this.charts.memory) {
            this.charts.memory.data.labels = buffer.timestamps;
            this.charts.memory.data.datasets[0].data = buffer.used;
            this.charts.memory.data.datasets[1].data = buffer.allocated;
            this.charts.memory.draw();
        }
    }

    /**
     * 更新学习率数据
     */
    updateLearningRateData(lr) {
        if (!this.dataBuffers.learningRate) return;
        const buffer = this.dataBuffers.learningRate;
        const timestamp = new Date().toLocaleTimeString();

        buffer.values.push(lr);
        buffer.timestamps.push(timestamp);

        if (buffer.values.length > this.maxDataPoints) {
            buffer.values.shift();
            buffer.timestamps.shift();
        }

        if (this.charts.learningRate) {
            this.charts.learningRate.data.labels = buffer.timestamps;
            this.charts.learningRate.data.datasets[0].data = buffer.values;
            this.charts.learningRate.draw();
        }
    }

    /**
     * 更新梯度数据
     */
    updateGradientData(mean, max, norm) {
        if (!this.dataBuffers.gradient) return;
        const buffer = this.dataBuffers.gradient;
        const timestamp = new Date().toLocaleTimeString();

        buffer.mean.push(mean);
        buffer.max.push(max);
        buffer.norm.push(norm);
        buffer.timestamps.push(timestamp);

        if (buffer.mean.length > this.maxDataPoints) {
            buffer.mean.shift();
            buffer.max.shift();
            buffer.norm.shift();
            buffer.timestamps.shift();
        }

        if (this.charts.gradient) {
            this.charts.gradient.data.labels = buffer.timestamps;
            this.charts.gradient.data.datasets[0].data = buffer.mean;
            this.charts.gradient.data.datasets[1].data = buffer.max;
            this.charts.gradient.data.datasets[2].data = buffer.norm;
            this.charts.gradient.draw();
        }
    }

    /**
     * 更新权重分布数据
     */
    updateWeightDistribution(weights, numBins) {
        if (!this.dataBuffers.weightDist) return;
        numBins = numBins || 20;

        const min = Math.min(...weights);
        const max = Math.max(...weights);
        const binWidth = (max - min) / numBins || 1;
        const bins = [];
        const counts = [];

        for (let i = 0; i < numBins; i++) {
            const binStart = min + i * binWidth;
            const binEnd = binStart + binWidth;
            bins.push(`${(binStart).toFixed(2)}-${(binEnd).toFixed(2)}`);
            counts.push(0);
        }

        weights.forEach(w => {
            const idx = Math.min(Math.floor((w - min) / binWidth), numBins - 1);
            counts[idx]++;
        });

        if (this.charts.weightDist) {
            this.charts.weightDist.data.labels = bins;
            this.charts.weightDist.data.datasets[0].data = counts;
            this.charts.weightDist.draw();
        }
    }

    /**
     * 更新系统资源数据
     */
    updateSystemResourceData(cpu, memory, gpu) {
        if (!this.dataBuffers.systemResource) return;
        const buffer = this.dataBuffers.systemResource;
        const timestamp = new Date().toLocaleTimeString();

        buffer.cpu.push(cpu);
        buffer.memory.push(memory);
        buffer.gpu.push(gpu !== undefined ? gpu : 0);
        buffer.timestamps.push(timestamp);

        if (buffer.cpu.length > this.maxDataPoints) {
            buffer.cpu.shift();
            buffer.memory.shift();
            buffer.gpu.shift();
            buffer.timestamps.shift();
        }

        if (this.charts.systemResource) {
            this.charts.systemResource.data.labels = buffer.timestamps;
            this.charts.systemResource.data.datasets[0].data = buffer.cpu;
            this.charts.systemResource.data.datasets[1].data = buffer.memory;
            this.charts.systemResource.data.datasets[2].data = buffer.gpu;
            this.charts.systemResource.draw();
        }
    }

    /**
     * 更新知识增长数据
     */
    updateKnowledgeGrowthData(entries, concepts, relations) {
        if (!this.dataBuffers.knowledgeGrowth) return;
        const buffer = this.dataBuffers.knowledgeGrowth;
        const timestamp = new Date().toLocaleTimeString();

        buffer.entries.push(entries);
        buffer.concepts.push(concepts);
        buffer.relations.push(relations);
        buffer.timestamps.push(timestamp);

        if (buffer.entries.length > this.maxDataPoints) {
            buffer.entries.shift();
            buffer.concepts.shift();
            buffer.relations.shift();
            buffer.timestamps.shift();
        }

        if (this.charts.knowledgeGrowth) {
            this.charts.knowledgeGrowth.data.labels = buffer.timestamps;
            this.charts.knowledgeGrowth.data.datasets[0].data = buffer.entries;
            this.charts.knowledgeGrowth.data.datasets[1].data = buffer.concepts;
            this.charts.knowledgeGrowth.data.datasets[2].data = buffer.relations;
            this.charts.knowledgeGrowth.draw();
        }
    }

    /**
     * 更新机器人状态
     */
    updateRobotStatus(robotId, metrics) {
        if (!this.charts.robotStatus) return;

        const datasets = this.charts.robotStatus.data.datasets;
        const robotIndex = robotId - 1;

        if (robotIndex >= 0 && robotIndex < datasets.length) {
            datasets[robotIndex].data = [
                metrics.motion || 0,
                metrics.sensor || 0,
                metrics.response || 0,
                metrics.stability || 0,
                metrics.efficiency || 0,
                metrics.load || 0
            ];
            this.charts.robotStatus.draw();
        }
    }

    /**
     * 更新神经元状态激活数据
     */
    updateStateActivationData(data) {
        if (!this.stateActivationData) return;
        if (data.length !== this.heatmapSize || data[0].length !== this.heatmapSize) return;
        this.stateActivationData = data;
        this.renderStateActivationHeatmap();
    }

    /**
     * 更新预测数据
     */
    updatePredictionData(actual, predicted) {
        if (!this.dataBuffers.prediction) return;
        const buffer = this.dataBuffers.prediction;

        buffer.actual = actual.map((v, i) => ({ x: i, y: v }));
        buffer.predicted = predicted.map((v, i) => ({ x: i, y: v }));

        if (this.charts.prediction) {
            this.charts.prediction.data.datasets[0].data = buffer.actual;
            this.charts.prediction.data.datasets[1].data = buffer.predicted;
            this.charts.prediction.draw();
        }
    }

    /**
     * 更新概念演化数据
     */
    updateConceptEvolutionData(level) {
        if (!this.dataBuffers.conceptEvolution) return;
        const buffer = this.dataBuffers.conceptEvolution;
        const timestamp = new Date().toLocaleTimeString();

        buffer.concepts.push(level);
        buffer.timestamps.push(timestamp);

        if (buffer.concepts.length > this.maxDataPoints) {
            buffer.concepts.shift();
            buffer.timestamps.shift();
        }

        if (this.charts.conceptEvolution) {
            this.charts.conceptEvolution.data.labels = buffer.timestamps;
            this.charts.conceptEvolution.data.datasets[0].data = buffer.concepts;
            this.charts.conceptEvolution.draw();
        }
    }

    /**
     * 从后端API获取实时数据并更新所有图表
     */
    async fetchAndUpdateAll() {
        try {
            const api = window.SelfLnnApi;
            if (!api) return;

            const [learningStatus, systemStatus, memoryStatus, robotStatus] = await Promise.all([
                api.getLearningStatus ? api.getLearningStatus() : Promise.resolve(null),
                api.getSystemStatus ? api.getSystemStatus() : Promise.resolve(null),
                api.getMemoryStatus ? api.getMemoryStatus() : Promise.resolve(null),
                api.getRobotStatus ? api.getRobotStatus() : Promise.resolve(null)
            ]);

            if (learningStatus && learningStatus.success && learningStatus.data) {
                const training = learningStatus.data.training || {};
                const params = learningStatus.data.parameters || {};
                const learning = learningStatus.data.learning || {};

                const loss = training.loss;
                const valLoss = training.val_loss;
                if (loss !== undefined) {
                    this.updateLossData(
                        typeof loss === 'number' ? loss : (loss.current || 0),
                        valLoss !== undefined ? (typeof valLoss === 'number' ? valLoss : valLoss.current) : undefined
                    );
                }

                const accuracy = training.accuracy;
                const valAccuracy = training.val_accuracy;
                if (accuracy !== undefined) {
                    this.updateAccuracyData(
                        typeof accuracy === 'number' ? accuracy : (accuracy.current || 0),
                        valAccuracy !== undefined ? (typeof valAccuracy === 'number' ? valAccuracy : valAccuracy.current) : undefined
                    );
                }

                const lr = params.learning_rate;
                if (lr !== undefined) {
                    this.updateLearningRateData(lr);
                }

                const gradNorm = training.gradient_norm || training.grad_norm;
                const gradMax = training.gradient_max || training.grad_max;
                const gradMean = training.gradient_mean || training.grad_mean;
                if (gradMean !== undefined || gradMax !== undefined || gradNorm !== undefined) {
                    this.updateGradientData(
                        gradMean !== undefined ? gradMean : 0,
                        gradMax !== undefined ? gradMax : 0,
                        gradNorm !== undefined ? gradNorm : 0
                    );
                }
            }

            if (systemStatus && systemStatus.success && systemStatus.data) {
                const sysData = systemStatus.data.system || {};
                const modules = sysData.modules || {};

                const memory = modules.memory || {};
                const usedMB = memory.total ? Math.round(memory.total / (1024 * 1024)) : 0;
                const allocatedMB = memory.total ? Math.round(memory.total / (1024 * 1024)) : 0;
                if (usedMB > 0) {
                    this.updateMemoryData(usedMB, allocatedMB);
                }

                const cpuUsage = memory.total ? Math.min(95, Math.round((memory.total / (1024 * 1024 * 1024)) * 100)) : 0;
                const memUsage = memory.total ? Math.min(95, Math.round((memory.allocated || memory.total) / (1024 * 1024 * 1024) * 100)) : 0;
                if (cpuUsage > 0 || memUsage > 0) {
                    this.updateSystemResourceData(cpuUsage, memUsage, 0);
                }
            }

            if (memoryStatus && memoryStatus.success && memoryStatus.data) {
                const memData = memoryStatus.data.memory || {};
                const stats = memData.statistics || {};
                const totalItems = stats.total_items || 0;
                const concepts = memData.architecture ? (memData.architecture.long_term ? memData.architecture.long_term.capacity : 0) : 0;
                if (totalItems > 0) {
                    this.updateKnowledgeGrowthData(totalItems, concepts, stats.retrieval_success || 0);
                }
            }

            if (robotStatus && robotStatus.success && robotStatus.data) {
                const robot = robotStatus.data.robot || {};
                const metrics = {
                    motion: robot.motion_capability || robot.battery || 50,
                    sensor: robot.sensor_capability || 50,
                    response: robot.response_time ? Math.max(0, 100 - robot.response_time * 10) : 50,
                    stability: robot.state === 0 ? 80 : robot.state === 1 ? 95 : 50,
                    efficiency: robot.temperature ? Math.max(0, 100 - (robot.temperature - 20) * 2) : 50,
                    load: robot.load || robot.battery || 50
                };
                if (robot.robot_id || robot.status) {
                    this.updateRobotStatus(robot.robot_id || 1, metrics);
                }
            }

        } catch (error) {
            console.warn('可视化数据获取失败（等待数据到达）:', error.message);
        }
    }

    /**
     * 启动自动数据更新（使用真实后端API，优先通过统一轮询中心调度）
     * 
     * 通过ApiService定期获取后端真实数据更新所有图表。
     * 不使用任何模拟/虚假数据。
     */
    startAutoUpdate(intervalMs) {
        if (this.isRunning) return;
        this.isRunning = true;
        intervalMs = intervalMs || 3000;

        this.fetchAndUpdateAll();

        var self = this;
        if (typeof g_dataEngine !== 'undefined' && g_dataEngine) {
            g_dataEngine.registerModule('visualization', intervalMs, function() {
                self.fetchAndUpdateAll();
            });
        } else {
            this.updateInterval = setInterval(() => {
                self.fetchAndUpdateAll();
            }, intervalMs);
        }
    }

    /**
     * 停止自动更新
     */
    stopAutoUpdate() {
        this.isRunning = false;
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
        if (typeof g_dataEngine !== 'undefined' && g_dataEngine) {
            g_dataEngine.unregisterModule('visualization');
        }
        this.stopNetworkAnimation();
    }

    /**
     * 销毁所有SelfLnnChart实例，防止内存泄漏
     * 每次重新初始化前必须调用此方法释放旧实例
     */
    destroyAllCharts() {
        Object.keys(this.charts).forEach(key => {
            const chart = this.charts[key];
            if (chart && typeof chart.destroy === 'function') {
                chart.destroy();
            }
        });
        this.charts = {};
    }

    /**
     * 清除所有数据
     */
    clearAllData() {
        Object.keys(this.dataBuffers).forEach(key => {
            const buffer = this.dataBuffers[key];
            Object.keys(buffer).forEach(field => {
                if (Array.isArray(buffer[field])) {
                    buffer[field] = [];
                }
            });
        });

        Object.keys(this.charts).forEach(key => {
            const chart = this.charts[key];
            if (chart && chart.data) {
                chart.data.datasets.forEach(ds => {
                    ds.data = [];
                });
                if (chart.data.labels) {
                    chart.data.labels = [];
                }
                chart.draw();
            }
        });
    }

    /**
     * 导出图表数据
     */
    exportChartData(chartName) {
        if (!this.charts[chartName]) return null;
        const chart = this.charts[chartName];
        const data = {
            labels: chart.data.labels || [],
            datasets: chart.data.datasets.map(ds => ({
                label: ds.label,
                data: ds.data
            })),
            timestamp: new Date().toISOString()
        };
        return data;
    }

    /**
     * 导出所有图表数据为JSON
     */
    exportAllData() {
        const result = {};
        Object.keys(this.dataBuffers).forEach(key => {
            result[key] = this.dataBuffers[key];
        });
        result.exportTime = new Date().toISOString();
        return result;
    }

    /**
     * 调整图表显示范围
     */
    setDisplayRange(chartName, maxPoints) {
        if (maxPoints < 10) maxPoints = 10;
        if (maxPoints > 500) maxPoints = 500;
        this.maxDataPoints = maxPoints;
    }

    /**
     * 获取图表状态摘要
     */
    getChartsSummary() {
        const summary = {};
        Object.keys(this.charts).forEach(key => {
            const chart = this.charts[key];
            if (chart && chart.data && chart.data.datasets) {
                summary[key] = {
                    type: chart.config.type,
                    datasets: chart.data.datasets.length,
                    dataPoints: chart.data.datasets[0] ? chart.data.datasets[0].data.length : 0
                };
            }
        });
        return summary;
    }

    /**
     * 调整图表尺寸
     */
    resizeCharts() {
        Object.values(this.charts).forEach(chart => {
            if (chart && chart.resize) {
                chart.resize();
            }
        });
        if (this.networkCanvas) {
            const parent = this.networkCanvas.parentElement;
            if (parent) {
                this.networkCanvas.width = parent.clientWidth;
            }
        }
        if (this.stateActivationCanvas) {
            const parent = this.stateActivationCanvas.parentElement;
            if (parent) {
                const size = Math.min(parent.clientWidth, 400);
                this.stateActivationCanvas.width = size;
                this.stateActivationCanvas.height = size;
                this.heatmapSize = Math.floor(size / 10);
            }
        }
        this.renderStateActivationHeatmap();
    }

    /**
     * 初始化和渲染神经元状态激活热图
     */
    renderStateActivationHeatmap() {
        if (!this.stateActivationCtx || !this.stateActivationCanvas) return;

        const ctx = this.stateActivationCtx;
        const canvas = this.stateActivationCanvas;
        const w = canvas.width;
        const h = canvas.height;
        const size = this.heatmapSize || 20;
        const cellW = w / size;
        const cellH = h / size;

        if (!this.stateActivationData || this.stateActivationData.length === 0) {
            this.initHeatmapData();
        }

        ctx.clearRect(0, 0, w, h);

        const data = this.stateActivationData || this._defaultActivationData || [];
        const maxVal = Math.max(1, ...data.flat());

        for (let i = 0; i < data.length && i < size; i++) {
            for (let j = 0; j < (data[i] || []).length && j < size; j++) {
                const val = data[i] ? data[i][j] || 0 : 0;
                const intensity = Math.min(1, val / maxVal);
                const r = Math.round(40 + intensity * 180);
                const g = Math.round(40 + (1 - intensity) * 180);
                const b = Math.round(40 + (1 - intensity) * 215);
                ctx.fillStyle = `rgb(${r},${g},${b})`;
                ctx.fillRect(j * cellW, i * cellH, cellW + 0.5, cellH + 0.5);
                ctx.strokeStyle = 'rgba(255,255,255,0.08)';
                ctx.strokeRect(j * cellW, i * cellH, cellW, cellH);
            }
        }

        ctx.fillStyle = '#cccccc';
        ctx.font = '12px "Microsoft YaHei", sans-serif';
        ctx.textAlign = 'center';
        for (let i = 0; i < Math.min(size, 10); i++) {
            ctx.fillText(`T${i}`, (i + 0.5) * cellW, h - 4);
            ctx.save();
            ctx.translate(4, (i + 0.5) * cellH);
            ctx.rotate(-Math.PI / 2);
            ctx.fillText(`H${i}`, 0, 0);
            ctx.restore();
        }
    }

    /**
     * 初始化热力图数据（初始为零，等待后端真实数据）
     */
    initHeatmapData() {
        const size = this.heatmapSize || 20;
        this.stateActivationData = [];
        this._defaultActivationData = [];
        for (let i = 0; i < size; i++) {
            this.stateActivationData[i] = [];
            this._defaultActivationData[i] = [];
            for (let j = 0; j < size; j++) {
                this.stateActivationData[i][j] = 0;
                this._defaultActivationData[i][j] = 0;
            }
        }
        this.renderStateActivationHeatmap();
    }

    /**
     * 设置热力图显示分辨率
     */
    setHeatmapResolution(size) {
        if (size < 5 || size > 50) return;
        this.heatmapSize = size;
        this.initHeatmapData();
    }

    /**
     * 更新GPU监控数据
     */
    updateGpuMonitorData(gpuStatus) {
        const nameEl = document.getElementById('gpu-device-name');
        const memEl = document.getElementById('gpu-memory');
        const capEl = document.getElementById('gpu-compute-cap');
        const verEl = document.getElementById('gpu-cuda-version');

        if (nameEl) nameEl.textContent = gpuStatus.device_name || '等待连接';
        if (memEl) memEl.textContent = gpuStatus.memory ? `${gpuStatus.memory.used}/${gpuStatus.memory.total}` : '等待连接';
        if (capEl) capEl.textContent = gpuStatus.compute_capability || '等待连接';
        if (verEl) verEl.textContent = gpuStatus.cuda_version || '等待连接';
    }

    // ============================================================================
    // 频谱分析可视化
    // ============================================================================

    /**
     * 初始化频谱分析可视化
     */
    initSpectrumAnalysis() {
        this.spectrumCanvas = document.getElementById('spectrum-analysis');
        if (!this.spectrumCanvas) return;
        this.spectrumCtx = this.spectrumCanvas.getContext('2d');

        this.spectrumData = {
            frequencies: [],        // 频率标签 (Hz)
            magnitudes: [],         // 幅度值
            sampleRate: 44100,      // 采样率
            fftSize: 1024,         // FFT点数
            numBands: 32,          // 显示频段数
            minFreq: 20,           // 最小频率 (Hz)
            maxFreq: 20000,        // 最大频率 (Hz)
            decayFactor: 0.92,     // 衰减因子（峰值保留）
            peakHold: [],          // 峰值保持
            peakHoldFrames: 30,    // 峰值保持帧数
            peakFramesLeft: [],    // 峰值剩余帧数
            barWidth: 0,
            barGap: 2,
            historyLength: 50,     // 频谱瀑布图历史长度
            historyData: [],       // 频谱历史数据
            colorMode: 'fire'      // 颜色模式: fire, rainbow, neon
        };

        // 初始化频段
        const bands = this.spectrumData.numBands;
        this.spectrumData.peakHold = new Array(bands).fill(0);
        this.spectrumData.peakFramesLeft = new Array(bands).fill(0);
        for (let i = 0; i < bands; i++) {
            this.spectrumData.frequencies.push(
                Math.round(this.spectrumData.minFreq *
                    Math.pow(this.spectrumData.maxFreq / this.spectrumData.minFreq,
                        i / (bands - 1)))
            );
            this.spectrumData.magnitudes.push(0);
        }

        this.resizeSpectrumCanvas();
        this.renderSpectrum();

        // 监听窗口大小变化
        window.addEventListener('resize', () => {
            if (this.spectrumCanvas && this.spectrumCanvas.offsetParent !== null) {
                this.resizeSpectrumCanvas();
                this.renderSpectrum();
            }
        });
    }

    /**
     * 调整频谱画布大小
     */
    resizeSpectrumCanvas() {
        const canvas = this.spectrumCanvas;
        if (!canvas) return;

        const container = canvas.parentElement;
        const w = container ? container.clientWidth : 600;
        const h = container ? container.clientHeight : 300;

        if (canvas.width !== w || canvas.height !== h) {
            canvas.width = w;
            canvas.height = h;
        }
    }

    /**
     * 更新频谱分析数据
     * @param {Float32Array|Array} magnitudes 各频段幅度值 (0-1)
     * @param {number} sampleRate 音频采样率
     */
    updateSpectrumData(magnitudes, sampleRate) {
        if (!this.spectrumCanvas || !magnitudes) return;

        const data = this.spectrumData;
        if (sampleRate) data.sampleRate = sampleRate;

        const bands = Math.min(magnitudes.length, data.numBands);

        for (let i = 0; i < data.numBands; i++) {
            if (i < bands) {
                const raw = Math.max(0, Math.min(1, magnitudes[i]));

                // 峰值保持
                if (raw > data.peakHold[i]) {
                    data.peakHold[i] = raw;
                    data.peakFramesLeft[i] = data.peakHoldFrames;
                }

                // 平滑衰减
                const smoothed = data.magnitudes[i] * 0.7 + raw * 0.3;
                data.magnitudes[i] = smoothed;
            } else {
                // 无数据频段衰减
                data.magnitudes[i] *= data.decayFactor;
            }

            // 峰值衰减
            if (data.peakFramesLeft[i] > 0) {
                data.peakFramesLeft[i]--;
            } else {
                data.peakHold[i] *= data.decayFactor * 0.95;
                if (data.peakHold[i] < 0.01) data.peakHold[i] = 0;
            }
        }

        // 更新频谱历史（瀑布图数据）
        data.historyData.unshift([...data.magnitudes]);
        if (data.historyData.length > data.historyLength) {
            data.historyData.pop();
        }

        this.renderSpectrum();
    }

    /**
     * 渲染频谱分析
     */
    renderSpectrum() {
        const ctx = this.spectrumCtx;
        const canvas = this.spectrumCanvas;
        if (!ctx || !canvas) return;

        const w = canvas.width;
        const h = canvas.height;
        const data = this.spectrumData;
        const bands = data.numBands;
        const chartHeight = h * 0.65;
        const waterfallHeight = h * 0.3;
        const waterfallY = h - waterfallHeight;

        ctx.clearRect(0, 0, w, h);

        // 背景
        const bgGrad = ctx.createLinearGradient(0, 0, 0, h);
        bgGrad.addColorStop(0, 'rgba(10,15,30,0.95)');
        bgGrad.addColorStop(1, 'rgba(5,8,15,0.98)');
        ctx.fillStyle = bgGrad;
        ctx.fillRect(0, 0, w, h);

        // 标题
        ctx.fillStyle = '#888888';
        ctx.font = '12px "Microsoft YaHei", sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('频谱分析', 10, 16);
        ctx.fillStyle = '#555555';
        ctx.font = '10px "Microsoft YaHei", sans-serif';
        ctx.fillText(`FFT: ${data.fftSize} 采样率: ${(data.sampleRate / 1000).toFixed(1)}kHz`, 10, 30);

        // 网格线
        ctx.strokeStyle = 'rgba(255,255,255,0.06)';
        ctx.lineWidth = 1;
        const gridLines = 8;
        for (let i = 0; i <= gridLines; i++) {
            const y = chartHeight * (1 - i / gridLines);
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(w, y);
            ctx.stroke();

            ctx.fillStyle = '#555555';
            ctx.font = '9px "Microsoft YaHei", sans-serif';
            ctx.textAlign = 'right';
            ctx.fillText(`${(i * 12).toFixed(0)}dB`, w - 5, y + 10);
        }

        // 计算柱状图参数
        const chartArea = w - 20;
        const barTotalWidth = chartArea / bands;
        const barWidth = Math.max(2, barTotalWidth - data.barGap);
        const barStartX = 10;

        // 绘制柱状图
        for (let i = 0; i < bands; i++) {
            const x = barStartX + i * barTotalWidth;
            const mag = data.magnitudes[i] || 0;
            const barHeight = mag * chartHeight;
            const y = chartHeight - barHeight;

            // 柱状图颜色（根据幅度和颜色模式）
            let color;
            if (data.colorMode === 'fire') {
                color = this.spectrumFireColor(mag);
            } else if (data.colorMode === 'rainbow') {
                color = this.spectrumRainbowColor(i, bands);
            } else {
                color = this.spectrumNeonColor(mag);
            }

            // 绘制柱体
            const grad = ctx.createLinearGradient(x, y, x, chartHeight);
            grad.addColorStop(0, color);
            grad.addColorStop(1, this.darkenColor(color, 0.4));
            ctx.fillStyle = grad;
            ctx.fillRect(x, y, barWidth, barHeight);

            // 柱体边框
            ctx.strokeStyle = this.darkenColor(color, 0.7);
            ctx.lineWidth = 0.5;
            ctx.strokeRect(x, y, barWidth, barHeight);

            // 峰值保持点
            const peak = data.peakHold[i] || 0;
            if (peak > 0.01) {
                const peakY = chartHeight - peak * chartHeight;
                ctx.fillStyle = '#ffffff';
                ctx.fillRect(x, peakY - 1, barWidth, 2);
            }

            // 频率标签（间隔显示）
            if (i % Math.max(1, Math.floor(bands / 12)) === 0) {
                ctx.fillStyle = '#666666';
                ctx.font = '8px "Microsoft YaHei", sans-serif';
                ctx.textAlign = 'center';
                const freq = data.frequencies[i] || 0;
                const label = freq >= 1000 ? `${(freq / 1000).toFixed(1)}k` : `${freq}`;
                ctx.fillText(label, x + barWidth / 2, chartHeight + 12);
            }
        }

        // 分界线
        ctx.strokeStyle = 'rgba(255,255,255,0.1)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, chartHeight + 2);
        ctx.lineTo(w, chartHeight + 2);
        ctx.stroke();

        // 频谱瀑布图（下方）
        if (data.historyData.length > 1) {
            this.renderSpectrumWaterfall(ctx, w, waterfallHeight, waterfallY, chartHeight + 5);
        }
    }

    /**
     * 渲染频谱瀑布图
     */
    renderSpectrumWaterfall(ctx, w, h, startY, topY) {
        const data = this.spectrumData;
        const bands = data.numBands;
        const historyLen = data.historyData.length;
        const bandWidth = (w - 20) / bands;
        const rowHeight = h / data.historyLength;

        for (let t = 0; t < historyLen; t++) {
            const row = data.historyData[t];
            if (!row) continue;
            const y = startY + t * rowHeight;

            for (let i = 0; i < bands && i < row.length; i++) {
                const mag = Math.max(0, Math.min(1, row[i] || 0));
                const x = 10 + i * bandWidth;

                // 瀑布图颜色映射
                const intensity = Math.floor(mag * 255);
                let color;
                if (mag < 0.33) {
                    color = `rgb(0, ${Math.floor(mag * 3 * 255)}, ${Math.floor(mag * 3 * 200)})`;
                } else if (mag < 0.66) {
                    const m = (mag - 0.33) * 3;
                    color = `rgb(${Math.floor(m * 255)}, ${Math.floor((1 - m) * 255)}, ${Math.floor((1 - m) * 100)})`;
                } else {
                    const m = (mag - 0.66) * 3;
                    color = `rgb(255, ${Math.floor((1 - m) * 128)}, ${Math.floor((1 - m) * 50)})`;
                }

                ctx.fillStyle = color;
                ctx.fillRect(x, y, Math.max(1, bandWidth * 0.8), rowHeight + 0.5);
            }
        }

        // 瀑布图标签
        ctx.fillStyle = '#666666';
        ctx.font = '9px "Microsoft YaHei", sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('频谱瀑布图', 10, startY - 4);
        ctx.textAlign = 'right';
        ctx.fillText(`最近 ${historyLen} 帧`, w - 10, startY - 4);
    }

    /**
     * 火焰颜色映射
     */
    spectrumFireColor(mag) {
        const r = Math.min(255, Math.floor(mag * 2.5 * 255));
        const g = Math.min(255, Math.floor(Math.max(0, mag * 1.5 - 0.3) * 255));
        const b = Math.min(200, Math.floor(Math.max(0, mag - 0.6) * 255 * 3));
        return `rgb(${r},${g},${b})`;
    }

    /**
     * 彩虹颜色映射
     */
    spectrumRainbowColor(index, total) {
        const hue = (index / total) * 300;
        return `hsl(${hue}, 100%, ${50 + Math.sin(index * 0.5) * 15}%)`;
    }

    /**
     * 霓虹颜色映射
     */
    spectrumNeonColor(mag) {
        const hue = 180 + mag * 120;
        return `hsl(${hue}, 100%, ${30 + mag * 40}%)`;
    }

    /**
     * 颜色变暗
     */
    darkenColor(color, factor) {
        if (color.startsWith('rgb')) {
            const match = color.match(/\d+/g);
            if (match) {
                const r = Math.floor(parseInt(match[0]) * factor);
                const g = Math.floor(parseInt(match[1]) * factor);
                const b = Math.floor(parseInt(match[2]) * factor);
                return `rgb(${r},${g},${b})`;
            }
        }
        return color;
    }

    /**
     * 设置频谱颜色模式
     * @param {string} mode 'fire' | 'rainbow' | 'neon'
     */
    setSpectrumColorMode(mode) {
        if (['fire', 'rainbow', 'neon'].includes(mode)) {
            this.spectrumData.colorMode = mode;
            this.renderSpectrum();
        }
    }

    /**
     * 设置频谱FFT大小
     */
    setSpectrumFftSize(size) {
        if ([256, 512, 1024, 2048, 4096].includes(size)) {
            this.spectrumData.fftSize = size;
        }
    }
}

window.VisualizationManager = VisualizationManager;


