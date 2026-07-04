/**
 * SELF-LNN AGI 训练可视化面板 - WebSocket实时推送集成
 * 将后端推送的训练数据实时映射到前端可视化图表和监控面板
 *
 * L-2修复注意: WebSocket消息类型在main.js和training-push.js双注册。
 * system_status/memory_status/robot_status等在两个文件中都有注册。
 * 确保回调不冲突: 两个文件共享同一个window.SelfLnnWebSocket实例，
 * 使用.on()逐一注册回调，WebSocket事件为广播模式，所有回调都会执行。
 */

'use strict';

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
        this.lastUpdateTime = 0; /* WebSocket数据时效性追踪 */
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
        /* M03修复: system_status/robot_status/memory_status/knowledge_status等
         * 仪表盘级事件处理器已统一迁移到 main.js，避免重复注册 */
        window.SelfLnnWebSocket.on('log', this._wsTrainingLogHandler);
        window.SelfLnnWebSocket.on('training_metrics', this._wsTrainingMetricsHandler);
        window.SelfLnnWebSocket.on('gpu_status', this._wsGpuStatusHandler);

        /* P1-01: 为所有未被消费的WS推送类型添加监听处理器 */
        this._wsKnowledgeUpdateHandler = (function(data) {
            console.log('[TrainingPush] knowledge_update:', data);
            document.dispatchEvent(new CustomEvent('ws-knowledge-update', { detail: data }));
        }).bind(this);
        window.SelfLnnWebSocket.on('knowledge_update', this._wsKnowledgeUpdateHandler);

        this._wsModelOutputHandler = (function(data) {
            console.log('[TrainingPush] model_output:', data);
            document.dispatchEvent(new CustomEvent('ws-model-output', { detail: data }));
        }).bind(this);
        window.SelfLnnWebSocket.on('model_output', this._wsModelOutputHandler);

        this._wsErrorHandler = (function(data) {
            console.log('[TrainingPush] error:', data);
            document.dispatchEvent(new CustomEvent('ws-error', { detail: data }));
        }).bind(this);
        window.SelfLnnWebSocket.on('error', this._wsErrorHandler);

        this._wsCustomHandler = (function(data) {
            console.log('[TrainingPush] custom:', data);
            document.dispatchEvent(new CustomEvent('ws-custom', { detail: data }));
        }).bind(this);
        window.SelfLnnWebSocket.on('custom', this._wsCustomHandler);

        this._wsDialogueResponseHandler = (function(data) {
            console.log('[TrainingPush] dialogue_response:', data);
            document.dispatchEvent(new CustomEvent('ws-dialogue-response', { detail: data }));
        }).bind(this);
        window.SelfLnnWebSocket.on('dialogue_response', this._wsDialogueResponseHandler);

        this._wsDialogueTokenHandler = (function(data) {
            console.log('[TrainingPush] dialogue_token:', data);
            document.dispatchEvent(new CustomEvent('ws-dialogue-token', { detail: data }));
        }).bind(this);
        window.SelfLnnWebSocket.on('dialogue_token', this._wsDialogueTokenHandler);

        this._wsEvolutionEventHandler = (function(data) {
            console.log('[TrainingPush] evolution_event:', data);
            document.dispatchEvent(new CustomEvent('ws-evolution-event', { detail: data }));
        }).bind(this);
        window.SelfLnnWebSocket.on('evolution_event', this._wsEvolutionEventHandler);

        this._wsCognitionEventHandler = (function(data) {
            console.log('[TrainingPush] cognition_event:', data);
            document.dispatchEvent(new CustomEvent('ws-cognition-event', { detail: data }));
        }).bind(this);
        window.SelfLnnWebSocket.on('cognition_event', this._wsCognitionEventHandler);

        this._wsDiagnosticHandler = (function(data) {
            console.log('[TrainingPush] diagnostic:', data);
            document.dispatchEvent(new CustomEvent('ws-diagnostic', { detail: data }));
        }).bind(this);
        window.SelfLnnWebSocket.on('diagnostic', this._wsDiagnosticHandler);

        this._wsMultimodalDataHandler = (function(data) {
            console.log('[TrainingPush] multimodal_data:', data);
            document.dispatchEvent(new CustomEvent('ws-multimodal-data', { detail: data }));
        }).bind(this);
        window.SelfLnnWebSocket.on('multimodal_data', this._wsMultimodalDataHandler);

/* 补全11个后端广播但前端未监听的WS事件类型 */
        this._wsSafetyAlertHandler = (function(data) {
            console.warn('[TrainingPush] 安全告警:', data);
            if (data && data.message && typeof window.showNotification === 'function') {
                window.showNotification('⚠️ 安全告警: ' + data.message, 'danger');
            }
            document.dispatchEvent(new CustomEvent('ws-safety-alert', { detail: data }));
        }).bind(this);
        window.SelfLnnWebSocket.on('safety_alert', this._wsSafetyAlertHandler);

        this._wsRobotStatusHandler = (function(data) {
            console.log('[TrainingPush] 机器人状态:', data);
            /* P-FIX-002: DataEngine是类构造函数, 实际实例为g_dataEngine */
            if (data && window.g_dataEngine && typeof window.g_dataEngine.updateRobotStatus === 'function') {
                window.g_dataEngine.updateRobotStatus(data);
            }
            document.dispatchEvent(new CustomEvent('ws-robot-status', { detail: data }));
        }).bind(this);
        window.SelfLnnWebSocket.on('robot_status', this._wsRobotStatusHandler);

        this._wsMemoryStatusHandler = (function(data) {
            console.log('[TrainingPush] 记忆状态:', data);
            if (data && window.g_dataEngine && typeof window.g_dataEngine.updateMemoryStatus === 'function') {
                window.g_dataEngine.updateMemoryStatus(data);
            }
            document.dispatchEvent(new CustomEvent('ws-memory-status', { detail: data }));
        }).bind(this);
        window.SelfLnnWebSocket.on('memory_status', this._wsMemoryStatusHandler);

        this._wsKnowledgeStatusHandler = (function(data) {
            console.log('[TrainingPush] 知识状态:', data);
            if (data && window.g_dataEngine && typeof window.g_dataEngine.updateKnowledgeStatus === 'function') {
                window.g_dataEngine.updateKnowledgeStatus(data);
            }
            document.dispatchEvent(new CustomEvent('ws-knowledge-status', { detail: data }));
        }).bind(this);
        window.SelfLnnWebSocket.on('knowledge_status', this._wsKnowledgeStatusHandler);

        this._wsPredictionResultHandler = (function(data) {
            console.log('[TrainingPush] 预测结果:', data);
            document.dispatchEvent(new CustomEvent('ws-prediction-result', { detail: data }));
        }).bind(this);
        window.SelfLnnWebSocket.on('prediction_result', this._wsPredictionResultHandler);

        this._wsConceptEvolutionHandler = (function(data) {
            console.log('[TrainingPush] 概念演化:', data);
            document.dispatchEvent(new CustomEvent('ws-concept-evolution', { detail: data }));
        }).bind(this);
        window.SelfLnnWebSocket.on('concept_evolution', this._wsConceptEvolutionHandler);

        this._wsStateActivationDataHandler = (function(data) {
            console.log('[TrainingPush] 状态激活数据:', data);
            if (data && window.visualizationManager) {
                window.visualizationManager.updateStateActivationData(data);
            }
            if (data && window.g_dataEngine && typeof window.g_dataEngine.updateStateActivation === 'function') {
                window.g_dataEngine.updateStateActivation(data);
            }
            document.dispatchEvent(new CustomEvent('ws-state-activation-data', { detail: data }));
        }).bind(this);
        window.SelfLnnWebSocket.on('state_activation_data', this._wsStateActivationDataHandler);

        this._wsWeightDistributionHandler = (function(data) {
            console.log('[TrainingPush] 权重分布:', data);
            if (data && window.visualizationManager) {
                window.visualizationManager.updateWeightDistributionData(data);
            }
            document.dispatchEvent(new CustomEvent('ws-weight-distribution', { detail: data }));
        }).bind(this);
        window.SelfLnnWebSocket.on('weight_distribution', this._wsWeightDistributionHandler);

        this._wsActivationStatsHandler = (function(data) {
            console.log('[TrainingPush] 激活统计:', data);
            if (data && window.visualizationManager) {
                window.visualizationManager.updateActivationData(
                    data.layers || [],
                    data.mean || 0,
                    data.max || 0,
                    data.min || 0,
                    data.std || 0
                );
            }
            document.dispatchEvent(new CustomEvent('ws-activation-stats', { detail: data }));
        }).bind(this);
        window.SelfLnnWebSocket.on('activation_stats', this._wsActivationStatsHandler);

        this._wsLnnStateHandler = (function(data) {
            console.log('[TrainingPush] LNN状态:', data);
            if (data && window.visualizationManager) {
                window.visualizationManager.updateLnnState(data);
            }
            if (data && window.g_dataEngine && typeof window.g_dataEngine.updateLnnState === 'function') {
                window.g_dataEngine.updateLnnState(data);
            }
            document.dispatchEvent(new CustomEvent('ws-lnn-state', { detail: data }));
        }).bind(this);
        window.SelfLnnWebSocket.on('lnn_state', this._wsLnnStateHandler);

        this._wsMetacognitionStatusHandler = (function(data) {
            console.log('[TrainingPush] 元认知状态:', data);
            if (data && window.visualizationManager) {
                window.visualizationManager.updateMetacognitionStatus(data);
            }
            if (data && window.g_dataEngine && typeof window.g_dataEngine.updateMetacognitionStatus === 'function') {
                window.g_dataEngine.updateMetacognitionStatus(data);
            }
            document.dispatchEvent(new CustomEvent('ws-metacognition-status', { detail: data }));
        }).bind(this);
        window.SelfLnnWebSocket.on('metacognition_status', this._wsMetacognitionStatusHandler);
    }

    _handleTrainingProgress(data) {
/* 记录数据更新时间戳 */
        this.lastUpdateTime = Date.now();
        /* FIX-F2-CRIT-6: 后端发送epoch/loss/progress/progress_pct,映射到前端字段 */
        if (data.epoch !== undefined) data.current_epoch = data.epoch;
        if (data.progress_pct !== undefined && data.progress === undefined) data.progress = data.progress_pct;
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
        /* FIX-F2-CRIT-6: 后端发送active_tasks/total_memories/reflection_count,无cpu/memory/gpu */
        if (window.visualizationManager) {
            window.visualizationManager.updateSystemResourceData(
                data.cpu_usage || data.cpu_percent || 0,
                data.memory_usage || data.memory_percent || 0,
                data.gpu_usage || data.gpu_percent || 0
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
/* 记录数据更新时间戳 */
        this.lastUpdateTime = Date.now();
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

        /* BUG-22修复：ETA超过24小时也显示具体时间，不再显示"计算中..." */
        if (s.etaSec > 0) {
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
            if (this._wsTrainingLogHandler) window.SelfLnnWebSocket.off('log', this._wsTrainingLogHandler);
            if (this._wsTrainingMetricsHandler) window.SelfLnnWebSocket.off('training_metrics', this._wsTrainingMetricsHandler);
            if (this._wsGpuStatusHandler) window.SelfLnnWebSocket.off('gpu_status', this._wsGpuStatusHandler);
            /* P1-01: 清理新增的21个WS监听器 */
            if (this._wsKnowledgeUpdateHandler) window.SelfLnnWebSocket.off('knowledge_update', this._wsKnowledgeUpdateHandler);
            if (this._wsModelOutputHandler) window.SelfLnnWebSocket.off('model_output', this._wsModelOutputHandler);
            if (this._wsErrorHandler) window.SelfLnnWebSocket.off('error', this._wsErrorHandler);
            if (this._wsCustomHandler) window.SelfLnnWebSocket.off('custom', this._wsCustomHandler);
            if (this._wsDialogueResponseHandler) window.SelfLnnWebSocket.off('dialogue_response', this._wsDialogueResponseHandler);
            if (this._wsDialogueTokenHandler) window.SelfLnnWebSocket.off('dialogue_token', this._wsDialogueTokenHandler);
            if (this._wsEvolutionEventHandler) window.SelfLnnWebSocket.off('evolution_event', this._wsEvolutionEventHandler);
            if (this._wsSafetyAlertHandler) window.SelfLnnWebSocket.off('safety_alert', this._wsSafetyAlertHandler);
            if (this._wsRobotStatusHandler) window.SelfLnnWebSocket.off('robot_status', this._wsRobotStatusHandler);
            if (this._wsCognitionEventHandler) window.SelfLnnWebSocket.off('cognition_event', this._wsCognitionEventHandler);
            if (this._wsDiagnosticHandler) window.SelfLnnWebSocket.off('diagnostic', this._wsDiagnosticHandler);
            if (this._wsMultimodalDataHandler) window.SelfLnnWebSocket.off('multimodal_data', this._wsMultimodalDataHandler);
            if (this._wsMemoryStatusHandler) window.SelfLnnWebSocket.off('memory_status', this._wsMemoryStatusHandler);
            if (this._wsKnowledgeStatusHandler) window.SelfLnnWebSocket.off('knowledge_status', this._wsKnowledgeStatusHandler);
            if (this._wsPredictionResultHandler) window.SelfLnnWebSocket.off('prediction_result', this._wsPredictionResultHandler);
            if (this._wsConceptEvolutionHandler) window.SelfLnnWebSocket.off('concept_evolution', this._wsConceptEvolutionHandler);
            if (this._wsStateActivationDataHandler) window.SelfLnnWebSocket.off('state_activation_data', this._wsStateActivationDataHandler);
            if (this._wsWeightDistributionHandler) window.SelfLnnWebSocket.off('weight_distribution', this._wsWeightDistributionHandler);
            if (this._wsActivationStatsHandler) window.SelfLnnWebSocket.off('activation_stats', this._wsActivationStatsHandler);
            if (this._wsLnnStateHandler) window.SelfLnnWebSocket.off('lnn_state', this._wsLnnStateHandler);
            if (this._wsMetacognitionStatusHandler) window.SelfLnnWebSocket.off('metacognition_status', this._wsMetacognitionStatusHandler);
        }
        this._wsTrainingProgressHandler = null;
        this._wsSystemStatusHandler = null;
        this._wsTrainingLogHandler = null;
        this._wsTrainingMetricsHandler = null;
        this._wsGpuStatusHandler = null;
        /* P1-01: 清理新增的21个WS处理器引用 */
        this._wsKnowledgeUpdateHandler = null;
        this._wsModelOutputHandler = null;
        this._wsErrorHandler = null;
        this._wsCustomHandler = null;
        this._wsDialogueResponseHandler = null;
        this._wsDialogueTokenHandler = null;
        this._wsEvolutionEventHandler = null;
        this._wsSafetyAlertHandler = null;
        this._wsRobotStatusHandler = null;
        this._wsCognitionEventHandler = null;
        this._wsDiagnosticHandler = null;
        this._wsMultimodalDataHandler = null;
        this._wsMemoryStatusHandler = null;
        this._wsKnowledgeStatusHandler = null;
        this._wsPredictionResultHandler = null;
        this._wsConceptEvolutionHandler = null;
        this._wsStateActivationDataHandler = null;
        this._wsWeightDistributionHandler = null;
        this._wsActivationStatsHandler = null;
        this._wsLnnStateHandler = null;
        this._wsMetacognitionStatusHandler = null;
        this.initialized = false;
    }
}

/* L-1修复: 显式挂载到window，确保跨模块可访问 */
window.TrainingPushManager = TrainingPushManager;

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
