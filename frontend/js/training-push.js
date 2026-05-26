/**
 * SELF-LNN AGI 训练可视化面板 - WebSocket实时推送集成
 * 将后端推送的训练数据实时映射到前端可视化图表和监控面板
 */

class TrainingPushManager {
    constructor() {
        this.bufferSize = 1000;
        this.dataBuffers = {
            loss: { train: [], val: [], timestamps: [] },
            accuracy: { train: [], val: [], timestamps: [] },
            learningRate: { values: [], timestamps: [] },
            gradient: { mean: [], max: [], norm: [], timestamps: [] },
            speed: { samplesPerSec: [], timestamps: [] },
            gpu: { util: [], memory: [], temp: [], timestamps: [] }
        };
        this.trainingState = {
            running: false,
            currentEpoch: 0,
            totalEpochs: 0,
            startTime: null,
            elapsedSec: 0,
            etaSec: 0,
            samplesPerSec: 0,
            totalSamples: 0,
            currentTask: '等待连接',
            modelName: '',
            trainingType: ''
        };
        this.logEntries = [];
        this.maxLogEntries = 500;
        this.logFilter = 'all';
        this.initialized = false;
        this._pollTimer = null;
        this._speedSamples = [];
    }

    init() {
        if (this.initialized) return;
        this.initialized = true;
        this._registerWebSocketHandlers();
        this._startGPUPolling();
    }

    _registerWebSocketHandlers(retryCount) {
        retryCount = retryCount || 0;
        if (!window.SelfLnnWebSocket) {
            if (retryCount < 20) {
                setTimeout(() => this._registerWebSocketHandlers(retryCount + 1), 500);
            } else {
                console.error('[TrainingPush] WebSocket模块加载超时(10s)，训练推送功能不可用');
                if (typeof window.showNotification === 'function') {
                    window.showNotification('⚠️ WebSocket模块加载超时，训练推送功能不可用', 'warning');
                }
            }
            return;
        }

        if (typeof window.SelfLnnWebSocket.on !== 'function') {
            console.error('[TrainingPush] WebSocket模块不支持on()方法，训练推送功能不可用');
            return;
        }

        /* 存储命名函数引用，便于后续通过off()移除事件监听器 */
        this._wsTrainingProgressHandler = function(data) {
            this._handleTrainingProgress(data);
        }.bind(this);

        this._wsSystemStatusHandler = function(data) {
            this._handleSystemStatus(data);
        }.bind(this);

        this._wsTrainingLogHandler = function(data) {
            this._handleTrainingLog(data);
        }.bind(this);

        this._wsTrainingMetricsHandler = function(data) {
            this._handleTrainingMetrics(data);
        }.bind(this);

        this._wsGpuStatusHandler = function(data) {
            this._handleGPUStatus(data);
        }.bind(this);

        window.SelfLnnWebSocket.on('training_progress', this._wsTrainingProgressHandler);
        window.SelfLnnWebSocket.on('system_status', this._wsSystemStatusHandler);
        window.SelfLnnWebSocket.on('training_log', this._wsTrainingLogHandler);
        window.SelfLnnWebSocket.on('training_metrics', this._wsTrainingMetricsHandler);
        window.SelfLnnWebSocket.on('gpu_status', this._wsGpuStatusHandler);
    }

    _handleTrainingProgress(data) {
        if (data.current_epoch !== undefined) {
            this.trainingState.currentEpoch = data.current_epoch;
        }
        if (data.total_epochs !== undefined) {
            this.trainingState.totalEpochs = data.total_epochs;
        }
        if (data.task !== undefined) {
            this.trainingState.currentTask = data.task;
        }
        if (data.elapsed_sec !== undefined) {
            this.trainingState.elapsedSec = data.elapsed_sec;
        }
        if (data.eta_sec !== undefined) {
            this.trainingState.etaSec = data.eta_sec;
        }
        if (data.samples_per_sec !== undefined) {
            this.trainingState.samplesPerSec = data.samples_per_sec;
            this._appendSpeedData(data.samples_per_sec);
        }
        if (data.running !== undefined) {
            this.trainingState.running = data.running;
            if (data.running && !this.trainingState.startTime) {
                this.trainingState.startTime = Date.now();
            }
            if (!data.running) {
                this.trainingState.startTime = null;
            }
        }
        if (data.progress !== undefined) {
            this._updateProgressBar(data.progress);
        }

        if (data.loss !== undefined) {
            this._appendLossData(data.loss, data.val_loss);
        }
        if (data.accuracy !== undefined) {
            this._appendAccuracyData(data.accuracy, data.val_accuracy);
        }
        if (data.learning_rate !== undefined) {
            this._appendLearningRateData(data.learning_rate);
        }

        this._updateTrainingUI();

        if (window.visualizationManager) {
            if (data.loss !== undefined) {
                window.visualizationManager.updateLossData(data.loss, data.val_loss);
            }
            if (data.accuracy !== undefined) {
                window.visualizationManager.updateAccuracyData(data.accuracy, data.val_accuracy);
            }
            if (data.learning_rate !== undefined) {
                window.visualizationManager.updateLearningRateData(data.learning_rate);
            }
            if (data.gradient_mean !== undefined) {
                window.visualizationManager.updateGradientData(
                    data.gradient_mean || 0,
                    data.gradient_max || 0,
                    data.gradient_norm || 0
                );
            }
        }
    }

    _handleSystemStatus(data) {
        if (window.visualizationManager) {
            window.visualizationManager.updateSystemResourceData(
                data.cpu_usage || 0,
                data.memory_usage || 0,
                data.gpu_usage || 0
            );
        }
        this._updateGPUDisplay(data);
    }

    _handleTrainingLog(data) {
        if (!data.message) return;
        const entry = {
            time: data.time || Date.now(),
            level: data.level || 'info',
            message: data.message,
            source: data.source || '训练系统'
        };
        this.logEntries.push(entry);
        if (this.logEntries.length > this.maxLogEntries) {
            this.logEntries.splice(0, this.logEntries.length - this.maxLogEntries);
        }
        this._appendLogToUI(entry);
    }

    _handleTrainingMetrics(data) {
        if (data.gradient_mean !== undefined && window.visualizationManager) {
            window.visualizationManager.updateGradientData(
                data.gradient_mean,
                data.gradient_max || 0,
                data.gradient_norm || 0
            );
        }
        if (data.weight_distribution && window.visualizationManager) {
            window.visualizationManager.updateWeightDistributionData(data.weight_distribution);
        }
        if (data.activation_stats && window.visualizationManager) {
            window.visualizationManager.updateActivationData(
                data.activation_stats.layers,
                data.activation_stats.mean,
                data.activation_stats.max,
                data.activation_stats.min,
                data.activation_stats.std
            );
        }
    }

    _handleGPUStatus(data) {
        const util = data.utilization !== undefined ? data.utilization : 0;
        const mem = data.memory_used !== undefined ? data.memory_used : 0;
        const temp = data.temperature !== undefined ? data.temperature : 0;
        const now = Date.now();

        this.dataBuffers.gpu.util.push({ x: now, y: util });
        this.dataBuffers.gpu.memory.push({ x: now, y: mem });
        this.dataBuffers.gpu.temp.push({ x: now, y: temp });

        const maxLen = 200;
        if (this.dataBuffers.gpu.util.length > maxLen) {
            this.dataBuffers.gpu.util.shift();
            this.dataBuffers.gpu.memory.shift();
            this.dataBuffers.gpu.temp.shift();
        }

        this._updateGPUDisplay(data);
    }

    _appendLossData(train, val) {
        const now = Date.now();
        this.dataBuffers.loss.train.push({ x: now, y: train });
        if (val !== undefined) {
            this.dataBuffers.loss.val.push({ x: now, y: val });
        }
        this.dataBuffers.loss.timestamps.push(now);
        if (this.dataBuffers.loss.train.length > this.bufferSize) {
            this.dataBuffers.loss.train.shift();
            this.dataBuffers.loss.val.shift();
            this.dataBuffers.loss.timestamps.shift();
        }
    }

    _appendAccuracyData(train, val) {
        const now = Date.now();
        this.dataBuffers.accuracy.train.push({ x: now, y: train });
        if (val !== undefined) {
            this.dataBuffers.accuracy.val.push({ x: now, y: val });
        }
        this.dataBuffers.accuracy.timestamps.push(now);
        if (this.dataBuffers.accuracy.train.length > this.bufferSize) {
            this.dataBuffers.accuracy.train.shift();
            this.dataBuffers.accuracy.val.shift();
            this.dataBuffers.accuracy.timestamps.shift();
        }
    }

    _appendLearningRateData(value) {
        const now = Date.now();
        this.dataBuffers.learningRate.values.push({ x: now, y: value });
        this.dataBuffers.learningRate.timestamps.push(now);
        if (this.dataBuffers.learningRate.values.length > this.bufferSize) {
            this.dataBuffers.learningRate.values.shift();
            this.dataBuffers.learningRate.timestamps.shift();
        }
    }

    _appendSpeedData(sps) {
        const now = Date.now();
        this._speedSamples.push({ x: now, y: sps });
        if (this._speedSamples.length > 100) this._speedSamples.shift();
        this.dataBuffers.speed.samplesPerSec.push({ x: now, y: sps });
        if (this.dataBuffers.speed.samplesPerSec.length > this.bufferSize) {
            this.dataBuffers.speed.samplesPerSec.shift();
        }
    }

    _updateProgressBar(progress) {
        const fill = document.getElementById('training-progress-fill');
        const text = document.getElementById('training-progress-text');
        if (fill) fill.style.width = Math.min(progress, 100) + '%';
        if (text) text.textContent = Math.round(progress) + '%';
    }

    _updateTrainingUI() {
        const s = this.trainingState;
        const setText = (id, value) => {
            const el = document.getElementById(id);
            if (el) el.textContent = value !== null && value !== undefined ? value : '等待连接';
        };

        setText('training-current-task', s.currentTask);
        setText('training-current-epoch', s.totalEpochs > 0 ? `${s.currentEpoch}/${s.totalEpochs}` : '等待连接');

        if (s.elapsedSec > 0) {
            const h = Math.floor(s.elapsedSec / 3600);
            const m = Math.floor((s.elapsedSec % 3600) / 60);
            const sec = Math.floor(s.elapsedSec % 60);
            setText('training-elapsed-time', `${h}h ${m}m ${sec}s`);
        }

        if (s.etaSec > 0 && s.etaSec < 86400) {
            const h = Math.floor(s.etaSec / 3600);
            const m = Math.floor((s.etaSec % 3600) / 60);
            setText('training-remaining-time', `${h}h ${m}m`);
        } else {
            setText('training-remaining-time', '计算中...');
        }

        const speedEl = document.getElementById('training-speed');
        if (speedEl) {
            speedEl.textContent = s.samplesPerSec > 0 ? `${s.samplesPerSec.toFixed(1)} 样本/秒` : '等待连接';
        }
    }

    _updateGPUDisplay(data) {
        const gpuUtil = document.getElementById('gpu-util-bar');
        const gpuMem = document.getElementById('gpu-mem-bar');
        const gpuTemp = document.getElementById('gpu-temp-value');

        if (gpuUtil) {
            const u = data.gpu_usage !== undefined ? data.gpu_usage : (data.utilization !== undefined ? data.utilization : 0);
            gpuUtil.style.width = Math.max(0, Math.min(u, 100)) + '%';
            gpuUtil.textContent = Math.round(u) + '%';
        }
        if (gpuMem) {
            const m = data.gpu_memory_used !== undefined ? data.gpu_memory_used : (data.memory_used !== undefined ? data.memory_used : 0);
            gpuMem.style.width = Math.max(0, Math.min(m, 100)) + '%';
            gpuMem.textContent = Math.round(m) + '%';
        }
        if (gpuTemp) {
            const t = data.gpu_temperature !== undefined ? data.gpu_temperature : (data.temperature !== undefined ? data.temperature : 0);
            gpuTemp.textContent = t > 0 ? t.toFixed(1) + '°C' : '--°C';
        }
    }

    _appendLogToUI(entry) {
        const logOutput = document.getElementById('training-log-output');
        if (!logOutput) return;

        if (this.logFilter !== 'all' && entry.level !== this.logFilter) return;

        const timeStr = new Date(entry.time).toLocaleTimeString('zh-CN', { hour12: false });
        const div = document.createElement('div');
        div.className = `log-entry ${entry.level}`;
        div.innerHTML = `<span class="log-time">[${timeStr}]</span> <span class="log-source">[${entry.source}]</span> <span class="log-msg">${this._escapeHtml(entry.message)}</span>`;
        logOutput.appendChild(div);
        logOutput.scrollTop = logOutput.scrollHeight;

        const maxVisible = 200;
        while (logOutput.children.length > maxVisible) {
            logOutput.removeChild(logOutput.firstChild);
        }
    }

    _escapeHtml(str) {
        const map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' };
        return String(str).replace(/[&<>"']/g, (c) => map[c]);
    }

    _startGPUPolling() {
        if (this._pollTimer) return;
        this._pollTimer = setInterval(() => {
            if (!this.trainingState.running) return;
            const gpuEl = document.getElementById('gpu-util-bar');
            const memEl = document.getElementById('gpu-mem-bar');
            const tempEl = document.getElementById('gpu-temp-value');
            if (gpuEl && gpuEl.textContent === '' && gpuEl.style.width === '0%') {
                gpuEl.style.width = '0%';
                gpuEl.textContent = '等待数据...';
            }
            if (memEl && memEl.textContent === '') {
                memEl.style.width = '0%';
                memEl.textContent = '等待数据...';
            }
            if (tempEl && tempEl.textContent === '') {
                tempEl.textContent = '--°C';
            }
        }, 5000);
    }

    startTraining(model, trainingType, config) {
        this.trainingState.running = true;
        this.trainingState.startTime = Date.now();
        this.trainingState.modelName = model || '';
        this.trainingState.trainingType = trainingType || '';
        this.trainingState.currentTask = '初始化训练...';
        this._updateTrainingUI();
        this._addLogEntry('info', `训练任务开始 - 模型: ${model}, 类型: ${trainingType}`);
    }

    stopTraining() {
        this.trainingState.running = false;
        this.trainingState.startTime = null;
        this._addLogEntry('info', '训练任务已停止');
        this._updateTrainingUI();
    }

    pauseTraining() {
        this.trainingState.running = false;
        this._addLogEntry('warning', '训练任务已暂停');
    }

    resumeTraining() {
        this.trainingState.running = true;
        this.trainingState.startTime = Date.now() - this.trainingState.elapsedSec * 1000;
        this._addLogEntry('info', '训练任务已恢复');
    }

    _addLogEntry(level, message) {
        const entry = { time: Date.now(), level, message, source: '训练面板' };
        this.logEntries.push(entry);
        if (this.logEntries.length > this.maxLogEntries) {
            this.logEntries.splice(0, this.logEntries.length - this.maxLogEntries);
        }
        this._appendLogToUI(entry);
    }

    getBufferedData(dataType) {
        return this.dataBuffers[dataType] || null;
    }

    getTrainingState() {
        return { ...this.trainingState };
    }

    exportTrainingData() {
        const exportData = {
            exportTime: new Date().toISOString(),
            trainingState: this.trainingState,
            buffers: {
                loss: { train: this.dataBuffers.loss.train, val: this.dataBuffers.loss.val },
                accuracy: { train: this.dataBuffers.accuracy.train, val: this.dataBuffers.accuracy.val },
                learningRate: this.dataBuffers.learningRate.values,
                gradient: this.dataBuffers.gradient,
                speed: this.dataBuffers.speed,
                gpu: this.dataBuffers.gpu
            },
            logs: this.logEntries.slice(-100)
        };
        const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `training-data-${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.json`;
        a.click();
        URL.revokeObjectURL(url);
    }

    setLogFilter(filter) {
        this.logFilter = filter;
        const logOutput = document.getElementById('training-log-output');
        if (!logOutput) return;
        logOutput.innerHTML = '';
        const filtered = this.logFilter === 'all'
            ? this.logEntries
            : this.logEntries.filter(e => e.level === this.logFilter);
        filtered.forEach(e => this._appendLogToUI(e));
        if (filtered.length === 0) {
            const div = document.createElement('div');
            div.className = 'log-entry info';
            div.textContent = '没有匹配的日志条目';
            logOutput.appendChild(div);
        }
    }

    destroy() {
        if (this._pollTimer) {
            clearInterval(this._pollTimer);
            this._pollTimer = null;
        }
        /* 使用命名函数引用移除WebSocket事件监听器，防止内存泄漏 */
        if (window.SelfLnnWebSocket && typeof window.SelfLnnWebSocket.off === 'function') {
            if (this._wsTrainingProgressHandler) window.SelfLnnWebSocket.off('training_progress', this._wsTrainingProgressHandler);
            if (this._wsSystemStatusHandler) window.SelfLnnWebSocket.off('system_status', this._wsSystemStatusHandler);
            if (this._wsTrainingLogHandler) window.SelfLnnWebSocket.off('training_log', this._wsTrainingLogHandler);
            if (this._wsTrainingMetricsHandler) window.SelfLnnWebSocket.off('training_metrics', this._wsTrainingMetricsHandler);
            if (this._wsGpuStatusHandler) window.SelfLnnWebSocket.off('gpu_status', this._wsGpuStatusHandler);
        }
        this._wsTrainingProgressHandler = null;
        this._wsSystemStatusHandler = null;
        this._wsTrainingLogHandler = null;
        this._wsTrainingMetricsHandler = null;
        this._wsGpuStatusHandler = null;
        this.initialized = false;
    }
}

let trainingPushManager = null;

function initTrainingPushManager() {
    if (!trainingPushManager) {
        trainingPushManager = new TrainingPushManager();
        window.trainingPushManager = trainingPushManager;
    }
    trainingPushManager.init();
    return trainingPushManager;
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initTrainingPushManager);
} else {
    initTrainingPushManager();
}
