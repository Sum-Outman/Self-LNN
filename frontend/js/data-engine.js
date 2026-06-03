/**
 * @file data-engine.js
 * @brief 数据引擎 - 真实后端数据传递层 + 统一轮询中心
 *
 * 该引擎从SELF-LNN后端API获取真实系统运行数据，分发给前端UI组件。
 * 同时作为全局唯一的轮询调度中心，所有定时数据获取统一经由此处调度，
 * 避免多个独立 setInterval 造成的性能浪费。
 *
 * 当后端不可用时，显示清晰的"后端未连接"状态，绝不生成虚假数据。
 * 严格遵守需求："不可以使用任何假数据和虚拟数据"
 */

class DataEngine {
    constructor() {
        this.initialized = false;
        this.listeners = [];
        this.timerId = null;
        this._backendConnected = false;
        this._lastError = null;
        this._fetchCount = 0;

        this.data = {
            system: this._createEmptyState('系统状态'),
            lnn: this._createEmptyState('LNN状态'),
            memory: this._createEmptyState('LNN记忆'),
            learning: this._createEmptyState('学习状态'),
            multimodal: this._createEmptyState('多模态状态'),
            robot: this._createEmptyState('机器人状态'),
            knowledge: this._createEmptyState('知识库状态'),
            reasoning: this._createEmptyState('LNN推理状态'),
            training: this._createEmptyState('训练状态'),
            resources: this._createEmptyState('资源状态'),
            cognition: this._createEmptyState('认知状态'),
            evolution: this._createEmptyState('演化状态')
        };

        this._history = {};
        this._pollModules = new Map();
        this._baseInterval = 2000;
        this._consecutiveErrors = 0; /* 初始化连续错误计数器 */
    }

    registerModule(name, intervalMs, callback) {
        if (typeof callback !== 'function') {
            console.error('轮询模块注册失败: ' + name + '，回调必须为函数');
            return;
        }
        this._pollModules.set(name, {
            interval: Math.max(intervalMs, this._baseInterval),
            lastRun: 0,
            callback: callback
        });
    }

    unregisterModule(name) {
        if (this._pollModules.delete(name)) {
        }
    }

    unregisterModuleByPrefix(prefix) {
        for (const key of this._pollModules.keys()) {
            if (key.startsWith(prefix)) {
                this._pollModules.delete(key);
            }
        }
    }

    /**
     * 创建空状态对象（所有字段显示"等待后端连接"）
     */
    _createEmptyState(moduleName) {
        return {
            _module: moduleName,
            _connected: false,
            _error: null,
            _timestamp: null
        };
    }

    /**
     * 检查后端连接状态
     */
    async checkConnection() {
        try {
            const result = await window.SelfLnnApi.checkConnection();
            this._backendConnected = result.connected;
            if (!result.connected) {
                this._lastError = result.message || '后端服务器未连接';
            }
            return result;
        } catch (error) {
            this._backendConnected = false;
            this._lastError = error.message || '连接检查失败';
            return { connected: false, message: this._lastError };
        }
    }

    /**
     * 从后端获取所有数据（单一/status调用，避免11路并行超时）
     */
    async _fetchAllData() {
        if (!this._backendConnected) {
            return;
        }

        try {
            var result = await this._fetchWithTimeout(window.SelfLnnApi.getSystemStatus(), 'system');
            if (result && result.success && result.data) {
                var data = result.data;
                /* S-020修复: 后端/status返回{"system":{...}}，应提取data.system传入
                 * 否则this.data.system会变成{system:{...}}的嵌套结构，UI层无法直接读取字段 */
                this._updateFromApi('system', data.system || data);
                if (data.system && data.system.modules) {
                    var mods = data.system.modules;
                    this._updateFromApi('lnn', mods.lnn || {});
                    this._updateFromApi('memory', mods.memory || {});
                    this._updateFromApi('reasoning', mods.reasoning || {});
                    this._updateFromApi('learning', mods.learning || {});
                    this._updateFromApi('multimodal', { available: mods.multimodal === true || mods.multimodal === 'true' });
                    this._updateFromApi('robot', mods.robotics || {});
                    this._updateFromApi('cognition', mods.cognition || {});
                }
                if (data.system) {
                    this._updateFromApi('knowledge', data.system.knowledge ? { count: data.system.knowledge.count } : {});
                    this._updateFromApi('training', data.system.training || {});
                    this._updateFromApi('resources', { cpu_usage: data.system.cpu_usage, requests: data.system.requests });
                }
                if (data.system && data.system.modules && data.system.modules.lnn_state) {
                    this._updateFromApi('evolution', { lnn_state: data.system.modules.lnn_state });
                }
            } else {
                this.data.system._connected = false;
                this.data.system._error = result ? result.error : 'status API 返回空';
            }
        } catch (error) {
            console.error('数据获取失败:', error);
/* 不在此直接断开，由_tick统一管理连接状态（连续3次失败才断开） */
            this._lastError = error.message;
        }
    }

    /**
     * 带超时的fetch包装
     */
    async _fetchWithTimeout(promise, label) {
        const timeout = new Promise((_, reject) =>
            setTimeout(() => reject(new Error(label + '请求超时')), 20000)
        );
        return Promise.race([promise, timeout]);
    }

    /**
     * L-023修复: 通用骨架屏移除 — 在所有数据成功更新后清除页面骨架屏加载动画
     */
    removeAllSkeletonClasses() {
        try {
            var skeletonEls = document.querySelectorAll('.skeleton-loading');
            for (var i = 0; i < skeletonEls.length; i++) {
                skeletonEls[i].classList.remove('skeleton-loading');
            }
        } catch (e) {
            /* DOM不可用时静默忽略 */
        }
    }

    /**
     * 从API响应更新数据
     */
    _updateFromApi(key, apiData) {
        if (!apiData) return;

        const target = this.data[key];
        target._connected = true;
        target._error = null;
        target._timestamp = Date.now();

        if (typeof apiData === 'object' && apiData !== null) {
            Object.keys(apiData).forEach(field => {
                if (!field.startsWith('_')) {
                    target[field] = apiData[field];
                }
            });
        }
    }

    /**
     * 运行数据更新循环（统一心跳，所有轮询模块共享此单一 setInterval）
     */
    async _tick() {
        this._fetchCount++;

/* 连接检查每6秒执行一次，基于_baseInterval计算tick数量 */
        var connectionCheckIntervalMs = 6000;
        var internalTick = Math.max(1, Math.round(connectionCheckIntervalMs / this._baseInterval));

        if (this._fetchCount % (internalTick * 3) === 0) {
            await this.checkConnection();
        }

        if (this._backendConnected) {
            try {
                await this._fetchAllData();
                if (this._lastError) {
                    var errMsg = this._lastError;
                    this._lastError = null;
                    throw new Error(errMsg);
                }
                this._saveCachedData();
                this._consecutiveErrors = 0;
                /* L-023修复: 数据更新成功后清除所有骨架屏加载动画 */
                this.removeAllSkeletonClasses();
            } catch (e) {
                this._consecutiveErrors = (this._consecutiveErrors || 0) + 1;
                if (this._consecutiveErrors > 3) {
                    this._backendConnected = false;
                }
            }
        } else {
            // 指数退避：重连间隔逐渐增加
            var backoffCount = this._consecutiveErrors || 0;
            var skipInterval = Math.min(internalTick * Math.pow(2, Math.min(backoffCount, 6)), 120);
            if (this._fetchCount % skipInterval !== 0) {
                return;
            }
        }

        this._notifyListeners();

        for (const [name, mod] of this._pollModules) {
            var tickRatio = Math.round(mod.interval / this._baseInterval);
            if (this._fetchCount - mod.lastRun >= tickRatio) {
                mod.lastRun = this._fetchCount;
                try {
                    mod.callback();
                } catch (e) {
                    console.error('轮询模块 [' + name + '] 执行失败:', e);
                }
            }
        }
    }

    /**
     * 启动数据引擎（使用统一1s基础心跳，所有注册模块共享此单一定时器）
     */
    start(intervalMs) {
        if (this.timerId) {
            console.warn('数据引擎已在运行');
            return;
        }
        this.initialized = true;
        /* FIX-F2-7: 立即执行一次连接检查,消除18秒冷启动延迟 */
        this.checkConnection().then(function() {
            /* 连接成功后立即拉取第一批数据 */
            if (this._backendConnected) {
                this._fetchAllData().then(function() {
                    /* L-023修复: 首批数据成功后清除骨架屏 */
                    this.removeAllSkeletonClasses();
                }.bind(this)).catch(function(){});
            }
        }.bind(this)).catch(function() {});
        var self = this;
        /* BUG修复: 使用递归setTimeout替代setInterval，防止_tick()并发重叠执行 */
        var scheduleNext = function() {
            if (!self.initialized || !self.timerId) return;
            self.timerId = setTimeout(function() {
                self._tick().finally(function() {
                    scheduleNext();
                });
            }, self._baseInterval);
        };
        this.timerId = setTimeout(function() {
            self._tick().finally(function() {
                scheduleNext();
            });
        }, self._baseInterval);
    }

    /**
     * 缓存上次成功获取的真实数据到localStorage
     */
    _saveCachedData() {
        try {
            const cache = {};
            for (const key of Object.keys(this.data)) {
                if (this.data[key]._connected) {
                    cache[key] = JSON.parse(JSON.stringify(this.data[key]));
                }
            }
            cache._timestamp = Date.now();
            localStorage.setItem('selflnn_data_cache', JSON.stringify(cache));
        } catch (e) {
            // localStorage不可用，静默忽略
        }
    }

    /**
     * 从localStorage加载上次缓存的真实数据
     */
    _loadCachedData() {
        try {
            const raw = localStorage.getItem('selflnn_data_cache');
            if (raw) {
                const cache = JSON.parse(raw);
                if (cache._timestamp && (Date.now() - cache._timestamp < 3600000)) {
                    for (const key of Object.keys(cache)) {
                        if (key !== '_timestamp' && cache[key] && cache[key]._connected) {
                            cache[key]._cached = true;
                            cache[key]._cache_age = Math.round((Date.now() - cache._timestamp) / 1000);
                            this.data[key] = cache[key];
                        }
                    }
                }
            }
        } catch (e) {
            // 缓存读取失败，使用空状态
        }
    }

    /**
     * 停止数据引擎
     */
    stop() {
        if (this.timerId) {
            clearTimeout(this.timerId);
            this.timerId = null;
        }
        this.initialized = false;
        this._backendConnected = false;
    }

    /**
     * 获取当前数据
     */
    getData() {
        return this.data;
    }

    /**
     * 获取后端连接状态
     */
    isConnected() {
        return this._backendConnected;
    }

    /**
     * 获取最后错误信息
     */
    getLastError() {
        return this._lastError;
    }

    /**
     * 添加数据监听器
     */
    addListener(callback) {
        if (typeof callback === 'function') {
            this.listeners.push(callback);
        }
    }

    /**
     * 移除数据监听器
     */
    removeListener(callback) {
        this.listeners = this.listeners.filter(l => l !== callback);
    }

    /**
     * 通知所有监听器
     */
    _notifyListeners() {
        const snapshot = JSON.parse(JSON.stringify(this.data));
        snapshot._backendConnected = this._backendConnected;
        snapshot._lastError = this._lastError;
        this.listeners.forEach(cb => {
            try { cb(snapshot); } catch (e) { console.error('[DataEngine] 监听器回调异常:', e && e.message ? e.message : e); }
        });
    }
}


window.DataEngine = DataEngine;
