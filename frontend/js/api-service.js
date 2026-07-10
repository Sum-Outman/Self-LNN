/**
 * SELF-LNN AGI 后端API服务
 * 提供与SELF-LNN后端服务器的通信接口
 *
 * IIFE封装：防止全局变量污染，仅暴露 window.SelfLnnApi 一个入口
 */

(function() {
    'use strict';

/* BUG-23修复：使用Object.assign合并默认值和用户配置，确保所有默认字段都被保留 */
var SELFLNN_CONFIG = Object.assign({ port: 8080, host: 'localhost' }, window.SELFLNN_CONFIG || {});

/* 端点→子系统映射表（配置驱动，替代字符串匹配） */
var SUBSYSTEM_ENDPOINT_MAP = {
    '/status':        'system',
    '/lnn':           'lnn',
    '/memory':        'memory',
    '/reasoning':     'reasoning',
    '/learning':      'learning',
    '/training':      'training',
    '/knowledge':     'knowledge',
    '/robot':         'robot',
    '/robots':        'robot',
    '/ros':           'robot',
    '/gazebo':        'robot',
    '/simulation':    'simulation',
    '/dialogue':      'dialogue',
    '/multimodal':    'multimodal',
    '/voice':         'voice',
    '/tts':           'voice',
    '/gpu':           'gpu',
    '/evolution':     'evolution',
    '/programming':   'programming',
    '/safety':        'safety',
    '/files':         'files',
    '/device':        'device',
    '/devices':       'device',
    '/computer':      'computer',
    '/serial':        'serial',
    '/hyperparam':    'training',
    '/checkpoint':    'training',
    '/product':       'product_design',
    '/skills':        'knowledge',
    '/logs':          'system',
    '/api':           'system',
    '/agents':        'agent',
    '/keys':          'key',
    '/key':           'key',
    '/datasets':      'dataset',
    '/stereo':        'stereo',
    '/agi':           'agi',
    '/hardware':      'hardware',
    '/imitation':     'imitation',
    '/teach':         'teach',
    '/audio':         'audio',
    '/camera':        'camera',
    '/laplace':       'laplace',
    '/auto-learn':    'auto_learn',
    '/capability':    'capability',
    /* 【M-001修复】补全缺失的子系统映射，避免熔断器误判为unknown */
    '/kg':            'knowledge',    /* 知识图谱子系统 */
    '/task':          'agi',          /* 任务调度→AGI */
    '/model':         'system',       /* 模型管理→系统 */
    '/system':        'system',       /* 系统管理 */
    '/decision':      'reasoning',    /* 决策日志→推理 */
    '/sensor':        'sensor',       /* 传感器管道 */
    '/fleet':         'robot',        /* 机器人集群 */
    '/multi-agent':   'agent',        /* 多智能体 */
    '/usage-logs':    'system'        /* 使用日志→系统 */
};

class ApiService {
    constructor() {
        this.baseURL = `http://${SELFLNN_CONFIG.host}:${SELFLNN_CONFIG.port}/api`;
        this.connected = false;  /* F-003修复: 初始状态必须为未连接，由checkConnection异步确认后更新 */
        this._connectionVerified = false;  /* 首次连接确认标记 */
        this._coldStartUntil = Date.now()+ 30000; /* 冷启动期30秒，避免启动时误报熔断 */
        this.connectionCheckInterval = null;
        
        /* P1-FIX-04: 安全警告 - API密钥不应存储在localStorage中
         * localStorage数据在浏览器关闭后仍持久化，且XSS攻击可轻易窃取。
         * 改用sessionStorage（浏览器会话结束后自动清除），降低持久化泄露风险。
         * 最佳实践：仅在内存中存储，使用后立即清除，敏感操作通过后端token交换机制。
         */
        this._apiKey = sessionStorage.getItem('selflnn_api_key') || null;

        this.requestConfig = {
            retryCount: 1,
            retryDelay: 3000,
            timeout: 15000,
            useExponentialBackoff: true,
            maxRetryDelay: 60000
        };

        /* 客户端熔断器：监控后端子系统健康状态，熔断时直接报错不做降级 */
        this.circuitBreakers = {
            general:  { state: 'CLOSED', failures: 0, lastFailureTime: 0, threshold: 15, resetTimeoutMs: 5000 }, /* 阈值从50降低到15 */
            reasoning: { state: 'CLOSED', failures: 0, lastFailureTime: 0, threshold: 25, resetTimeoutMs: 15000 },
            learning: { state: 'CLOSED', failures: 0, lastFailureTime: 0, threshold: 5, resetTimeoutMs: 30000 },
            knowledge: { state: 'CLOSED', failures: 0, lastFailureTime: 0, threshold: 5, resetTimeoutMs: 30000 },
            multimodal: { state: 'CLOSED', failures: 0, lastFailureTime: 0, threshold: 9999, resetTimeoutMs: 1000 },
            dialogue: { state: 'CLOSED', failures: 0, lastFailureTime: 0, threshold: 5, resetTimeoutMs: 30000 },
            training: { state: 'CLOSED', failures: 0, lastFailureTime: 0, threshold: 5, resetTimeoutMs: 30000 },
            robot: { state: 'CLOSED', failures: 0, lastFailureTime: 0, threshold: 9999, resetTimeoutMs: 1000 },
            memory: { state: 'CLOSED', failures: 0, lastFailureTime: 0, threshold: 5, resetTimeoutMs: 30000 },
            lnn: { state: 'CLOSED', failures: 0, lastFailureTime: 0, threshold: 5, resetTimeoutMs: 30000 },
            gpu: { state: 'CLOSED', failures: 0, lastFailureTime: 0, threshold: 5, resetTimeoutMs: 30000 },
            evolution: { state: 'CLOSED', failures: 0, lastFailureTime: 0, threshold: 5, resetTimeoutMs: 30000 }
        };

        /* 请求队列：管理高并发请求，避免同时发出过多请求 */
        this.requestQueue = [];
        this.maxConcurrentRequests = 3;
        this.activeRequestCount = 0;
        this.requestQueueProcessing = false;
        this._drainInterval = 0;

        /* ====================================================================
         * 7.2 修复: 真实降级策略
         * 
         * 后端连接失败时的降级策略（不产生虚假数据）:
         *   1. OFFLINE:    无后端连接 → 所有API返回 {status:'offline',error:'未连接'}
         *   2. DEGRADED:   后端部分子系统熔断 → 仅返回可用子系统数据
         *   2. DEGRADED: 后端连接但部分子系统不可用 → 仅使用缓存数据
         *   3. OFFLINE: 完全不可用 → 返回错误码
         *   4. DISABLED:   严格模式 → 不返回任何数据, 纯错误提示
         * 
         * 传递到UI层: 每个API响应带有 downgrade_level 字段,
         * UI根据该字段决定是否显示模拟数据警告横幅。
         * ==================================================================== */
        this.downgradeLevel = 'OFFLINE';
        this.isDevMode = false;

        this.downgradeStrategies = {
            CONNECTED: { level: 2, allowSimulated: false, allowCached: true,  toastMsg: null },
            OFFLINE:   { level: 0, allowSimulated: false, allowCached: false, toastMsg: '未连接到后端服务器' },
            DEGRADED:  { level: 1, allowSimulated: false, allowCached: true,  toastMsg: '后端部分服务不可用' },
            DISABLED:   { level: -1, allowSimulated: false, allowCached: false, toastMsg: '功能已禁用' }
        };
    }

    /**
     * 获取指数退避延迟（含完整抖动）
     * @param {number} attempt - 当前重试次数（从0开始）
     * @param {number} baseDelay - 基础延迟（毫秒）
     * @param {number} maxDelay - 最大延迟（毫秒）
     * @returns {number} 带抖动的延迟毫秒数
     */
    getBackoffDelay(attempt, baseDelay, maxDelay) {
        /* 指数退避：baseDelay * 2^attempt + 随机抖动 */
        const exponentialDelay = Math.min(baseDelay * Math.pow(2, attempt), maxDelay);
        /* 完整抖动（Full Jitter）：0 ~ exponentialDelay 之间的随机值 */
        const jitteredDelay = Math.random()* exponentialDelay;
        return Math.min(jitteredDelay, maxDelay);
    }

    /**
     * 检查客户端熔断器是否允许请求发送
     * @param {string} subsystem - 子系统名称
     * @returns {boolean} true=允许发送，false=熔断拒绝
     */
    isCircuitBreakerAvailable(subsystem) {
        const cb = this.circuitBreakers[subsystem];
        if (!cb) return true;

        const now = Date.now();

        if (cb.state === 'OPEN') {
            if (now - cb.lastFailureTime >= cb.resetTimeoutMs) {
                cb.state = 'HALF_OPEN';
                return true;
            }
            return false;
        }

        if (cb.state === 'HALF_OPEN') {
            return true;
        }

        return true;
    }

    /**
     * 记录熔断器成功
     * @param {string} subsystem - 子系统名称
     */
    recordCircuitBreakerSuccess(subsystem) {
        const cb = this.circuitBreakers[subsystem];
        if (!cb) return;

        if (cb.state === 'HALF_OPEN') {
            cb.state = 'CLOSED';
            cb.failures = 0;
        } else if (cb.state === 'CLOSED' && cb.failures > 0) {
            cb.failures = 0;
        }
    }

    /**
     * 记录熔断器失败
     * @param {string} subsystem - 子系统名称
     */
    recordCircuitBreakerFailure(subsystem) {
        if (Date.now()< this._coldStartUntil) return;
        const cb = this.circuitBreakers[subsystem];
        if (!cb) return;

        cb.failures++;
        cb.lastFailureTime = Date.now();

        if (cb.state === 'CLOSED' && cb.failures >= cb.threshold) {
            cb.state = 'OPEN';
        } else if (cb.state === 'HALF_OPEN') {
            cb.state = 'OPEN';
        }
    }

    /**
     * 从响应头或响应体中提取熔断器子系统信息
     * @param {Response} response - fetch响应对象
     * @returns {string|null} 子系统名称或null
     */
    extractSubsystemFromResponse(response) {
        /* 尝试从响应头的X-Subsystem获取 */
        const subsystem = response.headers.get('X-Subsystem');
        if (subsystem && this.circuitBreakers[subsystem]) {
            return subsystem;
        }
        return null;
    }

    /**
     * 从错误中提取子系统（根据请求端点推测）
     * @param {string} endpoint - API端点
     * @returns {string} 子系统名称
     */
    guessSubsystemFromEndpoint(endpoint) {
        if (!endpoint) return 'unknown';
/* 配置驱动查找 */
        var path = endpoint.replace(/^https?:\/\/[^\/]+/, '');
        /* BUG-15修复：使用Object.keys避免for...in遍历原型链上的属性 */
        /* P2-FIX-10修复：按前缀长度降序排序，确保/api-docs优先于/api匹配 */
        var keys = Object.keys(SUBSYSTEM_ENDPOINT_MAP).sort(function(a, b) { return b.length - a.length; });
        for (var i = 0; i < keys.length; i++) {
            var prefix = keys[i];
            if (path.indexOf(prefix) === 0) {
                return SUBSYSTEM_ENDPOINT_MAP[prefix];
            }
        }
        return 'unknown';
    }

    /**
     * 处理请求队列
     * 按优先级处理排队中的请求，控制并发数不超过上限
     */
    async processRequestQueue() {
        if (this.requestQueueProcessing) return;
        this.requestQueueProcessing = true;

        while (this.requestQueue.length > 0 && this.activeRequestCount < this.maxConcurrentRequests) {
            const queueItem = this.requestQueue.shift();
            if (!queueItem) continue;

            this.activeRequestCount++;
            this.executeRequestWithRetry(queueItem.endpoint, queueItem.options, queueItem.resolve, queueItem.reject, queueItem.subsystem);
        }

        this.requestQueueProcessing = false;
    }

    /**
     * 执行带重试的请求（内部方法）
     */
    async executeRequestWithRetry(endpoint, options, resolve, reject, subsystem) {
        try {
            const result = await this._requestInternal(endpoint, options, subsystem);
            resolve(result);
        } catch (error) {
            reject(error);
        } finally {
            this.activeRequestCount--;
            if (this.requestQueue.length > 0) {
                this.processRequestQueue();
            }
        }
    }

    /**
     * 内部请求方法（不含队列调度）
     */
    async _requestInternal(endpoint, options = {}, subsystem) {
        const maxRetries = this.requestConfig.retryCount;
        const timeout = this.requestConfig.timeout;
        const baseDelay = this.requestConfig.retryDelay;
        const maxDelay = this.requestConfig.maxRetryDelay;
        const useBackoff = this.requestConfig.useExponentialBackoff;

        let lastError = null;

        for (let attempt = 0; attempt <= maxRetries; attempt++) {
            try {
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), timeout);

                /* BUG-16修复：URL拼接增加斜杠保护，避免baseURL无尾斜杠且endpoint无前导斜杠导致404 */
                /* L03修复：自动剥离/api/前缀，防止双重/api/api/xxx导致404 */
                var safeEndpoint = endpoint;
                if (safeEndpoint.indexOf('/api/') === 0) {
                    safeEndpoint = safeEndpoint.substring(4);
                }
                const response = await fetch(this.baseURL + (safeEndpoint.startsWith('/') ? safeEndpoint : '/' + safeEndpoint), {
                    ...options,
                    signal: controller.signal
                });

                clearTimeout(timeoutId);

                /* 熔断器：根据响应状态码记录成功/失败 */
                if (response.ok) {
                    if (subsystem) this.recordCircuitBreakerSuccess(subsystem);
                    return response;
                }

                if (response.status === 503) {
                    /* 503 Service Unavailable - 子系统未启用（如--no-robotics），不重试不熔断 */
                    return response;
                }

                if (response.status >= 500 && attempt < maxRetries) {
                    if (subsystem) this.recordCircuitBreakerFailure(subsystem);
                    const delayMs = useBackoff
                        ? this.getBackoffDelay(attempt, baseDelay, maxDelay)
                        : baseDelay;
                    console.warn('API请求失败(' + response.status + ')，子系统[' + subsystem + ']，第' + (attempt + 1) + '次重试，等待' + Math.round(delayMs) + 'ms...');
                    await this.delay(delayMs);
                    continue;
                }

                /* >=500且已达最大重试 → 记录为失败（非4xx穿透） */
                if (response.status >= 500) {
                    if (subsystem) this.recordCircuitBreakerFailure(subsystem);
                    return response;
                }

                /* 4xx客户端错误不重试，不记录熔断 */
                return response;

            } catch (error) {
                lastError = error;

                if (error.name === 'AbortError') {
                    if (subsystem) this.recordCircuitBreakerFailure(subsystem);
                    if (attempt < maxRetries) {
                        const delayMs = useBackoff
                            ? this.getBackoffDelay(attempt, baseDelay, maxDelay)
                            : baseDelay;
                        console.warn(`API请求超时[${subsystem}]，第${attempt + 1}次重试，等待${Math.round(delayMs)}ms...`);
                        await this.delay(delayMs);
                        continue;
                    }
                    throw new Error(`API请求超时：${endpoint}，已达到最大重试次数`);
                }

                if (error instanceof TypeError && error.message.includes('fetch')) {
                    /* 网络错误（如连接被拒绝） */
                    if (subsystem) this.recordCircuitBreakerFailure(subsystem);
                    if (attempt < maxRetries) {
                        const delayMs = useBackoff
                            ? this.getBackoffDelay(attempt, baseDelay, maxDelay)
                            : baseDelay;
                        console.warn(`网络连接失败[${subsystem}]，第${attempt + 1}次重试，等待${Math.round(delayMs)}ms...`);
                        await this.delay(delayMs);
                        continue;
                    }
                    throw new Error(`无法连接到后端服务器：${SELFLNN_CONFIG.host}:${SELFLNN_CONFIG.port}`);
                }

                if (attempt < maxRetries) {
                    const delayMs = useBackoff
                        ? this.getBackoffDelay(attempt, baseDelay, maxDelay)
                        : baseDelay;
                    console.warn(`API请求异常[${subsystem}]: ${error.message}，第${attempt + 1}次重试`);
                    await this.delay(delayMs);
                    continue;
                }
            }
        }

        throw lastError || new Error(`API请求失败[${subsystem}]，已达到最大重试次数`);
    }

    /**
     * 通用API请求方法（对外接口）
     * 支持指数退避重试、客户端熔断器、请求队列并发控制
     * @param {string} endpoint - API端点
     * @param {Object} options - fetch选项
     * @param {number} retryCount - 重试次数（可选，默认使用配置）
     * @returns {Promise<Response>}
     */
    async request(endpoint, options = {}, retryCount = null) {
        /* 页面加载期间排队请求，就绪后逐次释放（间隔200ms防止洪流） */
/* 添加3秒超时机制，防止__PAGE_READY未置true导致无限等待 */
        if (window.__PAGE_READY !== true) {
            var self = this;
            var _checkStartTime = Date.now();
            var _CHECK_TIMEOUT_MS = 3000;
            return new Promise(function(resolve) {
/* 每次排空时重置间隔为初始值，防止累积延迟 */
                self._drainInterval = 0;
                var check = function() {
                    if (window.__PAGE_READY === true) {
                        setTimeout(function() {
                            resolve(self.request(endpoint, options, retryCount));
                        }, self._drainInterval);
                        self._drainInterval += 200;
                    } else if (Date.now()- _checkStartTime >= _CHECK_TIMEOUT_MS) {
                        console.warn('[ApiService] __PAGE_READY等待超时(%dms)，自动释放所有挂起请求，强制继续', _CHECK_TIMEOUT_MS);
                        window.__PAGE_READY = true;
                        setTimeout(function() {
                            resolve(self.request(endpoint, options, retryCount));
                        }, self._drainInterval);
                        self._drainInterval += 200;
                    } else {
                        setTimeout(check, 500);
                    }
                };
                setTimeout(check, 500);
            });
        }
        /* F-014: 自动附加X-Api-Key认证头 */
        if (this._apiKey) {
            if (!options.headers) options.headers = {};
            if (!options.headers['X-Api-Key'] && !options.headers['x-api-key']) {
                options.headers['X-Api-Key'] = this._apiKey;
            }
        }
        /* 确定子系统 */
        const subsystem = options.subsystem || this.guessSubsystemFromEndpoint(endpoint);

        /* 覆盖重试次数 */
        if (retryCount !== null) {
            const originalRetryCount = this.requestConfig.retryCount;
            this.requestConfig.retryCount = retryCount;
            try {
                return await this._requestWithBreaker(endpoint, options, subsystem);
            } finally {
                this.requestConfig.retryCount = originalRetryCount;
            }
        }

        return await this._requestWithBreaker(endpoint, options, subsystem);
    }

    /**
     * 含熔断器检查和请求队列的请求方法
     */
    async _requestWithBreaker(endpoint, options = {}, subsystem) {
        /* 熔断器检查：如果熔断打开，直接抛出错误，不做任何降级处理 */
        if (subsystem && !this.isCircuitBreakerAvailable(subsystem)) {
            throw new Error(`子系统[${subsystem}]熔断保护：连续失败已达阈值，请在30秒后重试`);
        }

        /* 如果并发数已满，加入请求队列 */
        if (this.activeRequestCount >= this.maxConcurrentRequests) {
            return new Promise((resolve, reject) => {
                this.requestQueue.push({
                    endpoint,
                    options,
                    subsystem,
                    resolve,
                    reject
                });
                if (!this.requestQueueProcessing) {
                    this.processRequestQueue();
                }
            });
        }

        /* 直接执行 */
        this.activeRequestCount++;
        try {
            const result = await this._requestInternal(endpoint, options, subsystem);
            return result;
        } finally {
            this.activeRequestCount--;
            if (this.requestQueue.length > 0) {
                this.processRequestQueue();
            }
        }
    }

    /**
     * 延迟函数
     * @param {number} ms - 延迟毫秒数
     * @returns {Promise<void>}
     */
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
    
    /**
     * 检查后端连接状态
     */
    async checkConnection() {
        try {
            const response = await this.request('/status', {}, 1); // 快速检查，只重试1次
            
            if (response.ok) {
                this.connected = true;
                this._connectionVerified = true;  /* F-003修复: 只有服务器确认后才标记已验证 */
                return { connected: true, message: '后端服务器已连接' };
            } else {
                this.connected = false;
                return { connected: false, message: '后端服务器响应异常' };
            }
        } catch (error) {
            this.connected = false;
            return { connected: false, message: `连接失败: ${error.message}` };
        }
    }
    
    /**
     * 启动连接监控
     */
    startConnectionMonitor(interval = 5000) {
        /* BUG-9修复：先停止已有监控防止重复，保存setTimeout定时器ID支持取消 */
        this.stopConnectionMonitor();
        this._connectionMonitorTimeout = setTimeout(async () => {
            if (!this._connectionMonitorTimeout) return;
            const status = await this.checkConnection();
            this.notifyConnectionStatus(status);
            this.connectionCheckInterval = setInterval(async () => {
                const status = await this.checkConnection();
                this.notifyConnectionStatus(status);
            }, interval);
        }, 5000);
    }
    
    /**
     * 停止连接监控
     */
    stopConnectionMonitor() {
        /* BUG-9修复：清除初始延迟定时器，防止use-after-free */
        if (this._connectionMonitorTimeout) {
            clearTimeout(this._connectionMonitorTimeout);
            this._connectionMonitorTimeout = null;
        }
        if (this.connectionCheckInterval) {
            clearInterval(this.connectionCheckInterval);
            this.connectionCheckInterval = null;
        }
    }
    
    /**
     * 通知连接状态
     */
    notifyConnectionStatus(status) {
        // 可以在这里触发自定义事件或更新UI
        const event = new CustomEvent('backend-connection-status', {
            detail: status
        });
        document.dispatchEvent(event);
    }

    getDowngradeStrategy() {
        if (!this.connected) {
            this.downgradeLevel = 'OFFLINE';
        } else {
            var degraded = false;
            for (var k in this.circuitBreakers) {
                if (this.circuitBreakers[k].state === 'OPEN') { degraded = true; break; }
            }
            this.downgradeLevel = degraded ? 'DEGRADED' : 'CONNECTED';
        }
        return this.downgradeStrategies[this.downgradeLevel] || this.downgradeStrategies.OFFLINE;
    }
    
    /**
     * 获取系统状态
     */
    async getSystemStatus() {
        try {
            const response = await this.request('/status');
            if (!response.ok) {
                throw new Error(`HTTP错误: ${response.status}`);
            }
            const data = await response.json();
            return {
                success: true,
                data: data
            };
        } catch (error) {
            console.error('获取系统状态失败:', error);
            return {
                success: false,
                error: error.message,
                data: null
            };
        }
    }
    
    /**
     * F-致命修复: trainingStart — training-center.js调用的训练启动方法
     * 将位置参数映射为后端期望的配置对象格式
     * @param {string} mode - 训练模式: 'pretrain'|'train'|'finetune'|'transfer'|'continual'|'from-scratch'
     * @param {number} lr - 学习率
     * @param {number} batch - 批量大小
     * @param {number} epochs - 训练轮数
     * @param {string} datasetPath - 数据集路径
     * @returns {Promise<Object>} {success, data, error}
     */
    async trainingStart(mode, lr, batch, epochs, datasetPath) {
        var config = {
            mode: mode || 'train',
            learning_rate: lr || 0.001,
            batch_size: batch || 64,
            epochs: epochs || 100,
            dataset_path: datasetPath || '/data/training'
        };
        return await this.startTraining(config);
    }
    
    /**
     * 获取音频频谱分析数据
     */
    async getAudioSpectrum() {
        try {
            const response = await this.request('/audio/spectrum');
            if (!response.ok) throw new Error(`HTTP错误: ${response.status}`);
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error('[API] 获取音频频谱失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * 获取LNN神经元激活热力图数据
     */
    async getLnnActivationHeatmap() {
        try {
            const response = await this.request('/lnn/activation/heatmap');
            if (!response.ok) throw new Error(`HTTP错误: ${response.status}`);
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error('[API] 获取LNN激活热力图失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * 获取LNN预测结果散点图数据
     */
    async getLnnPredictionScatter() {
        try {
            const response = await this.request('/lnn/prediction/scatter');
            if (!response.ok) throw new Error(`HTTP错误: ${response.status}`);
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error('[API] 获取LNN预测散点图失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }
    
    /**
     * 获取内存状态
     */
    async getMemoryStatus() {
        try {
            const response = await this.request('/memory');
            if (!response.ok) {
                throw new Error(`HTTP错误: ${response.status}`);
            }
            const data = await response.json();
            return {
                success: true,
                data: data
            };
        } catch (error) {
            console.error('获取内存状态失败:', error);
            return {
                success: false,
                error: error.message,
                data: null
            };
        }
    }
    
    /**
     * 获取LNN推理状态
     */
    async getReasoningStatus() {
        try {
            const response = await this.request('/reasoning');
            if (!response.ok) {
                throw new Error(`HTTP错误: ${response.status}`);
            }
            const data = await response.json();
            return {
                success: true,
                data: data
            };
        } catch (error) {
            console.error('获取LNN推理状态失败:', error);
            return {
                success: false,
                error: error.message,
                data: null
            };
        }
    }
    
    /**
     * 获取学习引擎状态
     */
    async getLearningStatus() {
        try {
            const response = await this.request('/learning');
            if (!response.ok) {
                throw new Error(`HTTP错误: ${response.status}`);
            }
            const data = await response.json();
            return {
                success: true,
                data: data
            };
        } catch (error) {
            console.error('获取学习引擎状态失败:', error);
            return {
                success: false,
                error: error.message,
                data: null
            };
        }
    }
    
    /**
     * 处理视觉输入
     */
    async processVisionInput(imageData) {
        try {
            const response = await this.request('/vision', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ image: imageData })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP错误: ${response.status}`);
            }
            
            const data = await response.json();
            return {
                success: true,
                data: data
            };
        } catch (error) {
            console.error('处理视觉输入失败:', error);
            return {
                success: false,
                error: error.message,
                data: null
            };
        }
    }
    
    /**
     * 处理音频输入
     */
    async processAudioInput(audioData) {
        try {
            const response = await this.request('/audio', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ audio: audioData })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP错误: ${response.status}`);
            }
            
            const data = await response.json();
            return {
                success: true,
                data: data
            };
        } catch (error) {
            console.error('处理音频输入失败:', error);
            return {
                success: false,
                error: error.message,
                data: null
            };
        }
    }
    
    /**
     * 处理文本输入
     */
    async processTextInput(textData) {
        try {
            const response = await this.request('/text', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text: textData })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP错误: ${response.status}`);
            }
            
            const data = await response.json();
            return {
                success: true,
                data: data
            };
        } catch (error) {
            console.error('处理文本输入失败:', error);
            return {
                success: false,
                error: error.message,
                data: null
            };
        }
    }
    
    /**
     * 处理传感器输入
     */
    async processSensorInput(sensorData) {
        try {
            const response = await this.request('/sensor', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ sensor: sensorData })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP错误: ${response.status}`);
            }
            
            const data = await response.json();
            return {
                success: true,
                data: data
            };
        } catch (error) {
            console.error('处理传感器输入失败:', error);
            return {
                success: false,
                error: error.message,
                data: null
            };
        }
    }
    
    /**
     * 开始训练
     */
    async startTraining(trainingConfig) {
        try {
            const response = await this.request('/training/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(trainingConfig || {})
            });
            
            if (!response.ok) {
                throw new Error(`HTTP错误: ${response.status}`);
            }
            
            const data = await response.json();
            return {
                success: true,
                data: data
            };
        } catch (error) {
            console.error('开始训练失败:', error);
            return {
                success: false,
                error: error.message,
                data: null
            };
        }
    }
    
    /**
     * 开始演化
 *修复: 后端期望扁平字段(population_size/generations/mutation_rate)，非嵌套config对象
     */
    async startEvolution(evolutionConfig) {
        try {
            var cfg = evolutionConfig || {};
            const response = await this.request('/agi/evolve', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    generations: cfg.generations || cfg.max_generations || 100,
                    population_size: cfg.population_size || 50,
                    mutation_rate: cfg.mutation_rate || 0.01,
                    crossover_rate: cfg.crossover_rate || 0.7
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP错误: ${response.status}`);
            }
            
            const data = await response.json();
            return {
                success: true,
                data: data
            };
        } catch (error) {
            console.error('开始演化失败:', error);
            return {
                success: false,
                error: error.message,
                data: null
            };
        }
    }
    
    /**
     * 开始自我学习
     */
    async startSelfLearning(learningConfig) {
        try {
            const response = await this.request('/agi/learn', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ config: learningConfig || {} })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP错误: ${response.status}`);
            }
            
            const data = await response.json();
            return {
                success: true,
                data: data
            };
        } catch (error) {
            console.error('开始自我学习失败:', error);
            return {
                success: false,
                error: error.message,
                data: null
            };
        }
    }
    
    /**
     * 触发自我修正
     */
    async triggerSelfCorrection(correctionConfig) {
        try {
            const response = await this.request('/agi/self_correction', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ config: correctionConfig || {} })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP错误: ${response.status}`);
            }
            
            const data = await response.json();
            return {
                success: true,
                data: data
            };
        } catch (error) {
            console.error('触发自我修正失败:', error);
            return {
                success: false,
                error: error.message,
                data: null
            };
        }
    }
    
    /**
     * 重置系统
     */
    async resetSystem() {
        try {
            const response = await this.request('/reset', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });

            if (!response.ok) {
                throw new Error(`HTTP错误: ${response.status}`);
            }

            const data = await response.json();
            return {
                success: true,
                data: data
            };
        } catch (error) {
            console.error('重置系统失败:', error);
            return {
                success: false,
                error: error.message,
                data: null
            };
        }
    }
    
    /**
     * 关闭系统
     */
    async shutdownSystem() {
        try {
            const response = await this.request('/shutdown', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ action: 'shutdown' })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP错误: ${response.status}`);
            }
            
            const data = await response.json();
            return {
                success: true,
                data: data
            };
        } catch (error) {
            console.error('关闭系统失败:', error);
            return {
                success: false,
                error: error.message,
                data: null
            };
        }
    }
    
    /**
     * 系统备份
     */
    async backupSystem() {
        try {
            const response = await this.request('/backup', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });

            if (!response.ok) {
                throw new Error(`HTTP错误: ${response.status}`);
            }

            const data = await response.json();
            return {
                success: true,
                data: data
            };
        } catch (error) {
            console.error('系统备份失败:', error);
            return {
                success: false,
                error: error.message || '系统备份后端连接失败，请检查服务器状态',
                data: null
            };
        }
    }
    
    /**
     * 获取机器人状态
     */
    async getRobotStatus() {
        try {
            const response = await this.request('/robot/status');
            if (!response.ok) {
                if (response.status === 503) {
                    return { success: true, status: 'disabled', data: { status: 'disabled', message: '机器人控制已禁用' } };
                }
                throw new Error(`HTTP错误: ${response.status}`);
            }
            const data = await response.json();
/* 规范化后端robot_status.old_interface格式为前端期望的robot.battery平铺格式 */
            var rs = (data && data.robot_status) ? data.robot_status : data;
            var oi = (rs && rs.old_interface) ? rs.old_interface : {};
            var robot = {
                available: rs.hasOwnProperty('hardware_mode') ? (rs.hardware_mode !== 'none') : (oi.battery > 0),
                hardware_mode: rs.hardware_mode || 'unknown',
                battery: (typeof oi.battery === 'number') ? oi.battery : 0,
                temperature: (typeof oi.temperature === 'number') ? oi.temperature : 0,
                position: Array.isArray(oi.position) ? oi.position : [0, 0, 0],
                state: (oi.state !== undefined) ? oi.state : '--',
                is_simulation: rs.is_simulation || false,
                ros_controller: rs.ros_controller || { robot_count: 0, connected: 0 }
            };
            return {
                success: true,
                data: { robot: robot, robot_status: rs }
            };
        } catch (error) {
            console.warn('获取机器人状态:', error.message);
            return {
                success: false,
                error: error.message || '机器人状态后端连接失败，请检查服务器状态',
                data: null
            };
        }
    }
    
    /**
     * 发送机器人控制命令
     */
    async sendRobotCommand(command) {
        try {
            const response = await this.request('/robot/command', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(command)
            });
            
            if (!response.ok) {
                throw new Error(`HTTP错误: ${response.status}`);
            }
            
            const data = await response.json();
            return {
                success: true,
                data: data
            };
        } catch (error) {
            console.error('发送机器人命令失败:', error);
            return {
                success: false,
                error: error.message || '机器人控制后端连接失败，请检查服务器状态',
                data: null
            };
        }
    }
    
    /**
     * 获取机器人传感器数据
     */
    async getRobotSensorData() {
        try {
            const response = await this.request('/robot/sensor');
            if (!response.ok) {
                throw new Error(`HTTP错误: ${response.status}`);
            }
            const data = await response.json();
            return {
                success: true,
                data: data
            };
        } catch (error) {
            console.error('获取机器人传感器数据失败:', error);
            return {
                success: false,
                error: error.message || '机器人传感器后端连接失败，请检查服务器状态',
                data: null
            };
        }
    }
    
    /**
     * 执行机器人轨迹
     */
    async sendRobotTrajectory(trajectory) {
        try {
            const response = await this.request('/robot/trajectory', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(trajectory)
            });
            
            if (!response.ok) {
                throw new Error(`HTTP错误: ${response.status}`);
            }
            
            const data = await response.json();
            return {
                success: true,
                data: data
            };
        } catch (error) {
            console.error('发送机器人轨迹失败:', error);
            return {
                success: false,
                error: error.message || '机器人轨迹后端连接失败，请检查服务器状态',
                data: null
            };
        }
    }
    
    /**
     * 机器人紧急停止
     */
    async robotEmergencyStop() {
        try {
            const response = await this.request('/robot/emergency_stop', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            if (!response.ok) {
                throw new Error(`HTTP错误: ${response.status}`);
            }
            
            const data = await response.json();
            return {
                success: true,
                data: data
            };
        } catch (error) {
            console.error('机器人紧急停止失败:', error);
            return {
                success: false,
                error: error.message || '机器人紧急停止后端连接失败，请检查服务器状态',
                data: null
            };
        }
    }
    
    /**
     * 连接机器人
     */
    async connectRobot() {
        try {
            // 尝试调用后端API连接机器人
            const response = await this.request('/robot/connect', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ action: 'connect' })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP错误: ${response.status}`);
            }
            
            const data = await response.json();
            return {
                success: true,
                data: data
            };
        } catch (error) {
            console.error('连接机器人失败:', error);
            return {
                success: false,
                error: error.message || '机器人连接后端连接失败，请检查服务器状态',
                data: null
            };
        }
    }
    
    /**
     * 断开机器人连接
     */
    async disconnectRobot() {
        try {
            // 尝试调用后端API断开机器人连接
            const response = await this.request('/robot/disconnect', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ action: 'disconnect' })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP错误: ${response.status}`);
            }
            
            const data = await response.json();
            return {
                success: true,
                data: data
            };
        } catch (error) {
            console.error('断开机器人连接失败:', error);
            return {
                success: false,
                error: error.message || '机器人断开连接后端连接失败，请检查服务器状态',
                data: null
            };
        }
    }
    
    /**
     * 更新机器人固件
     */
    async updateFirmware(data) {
        try {
            var options = {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            };
            if (data !== undefined) {
                options.body = JSON.stringify(data);
            }
            const response = await this.request('/robot/firmware', options);

            if (!response.ok) {
                throw new Error(`HTTP错误: ${response.status}`);
            }

            const result = await response.json();
            return {
                success: true,
                data: result
            };
        } catch (error) {
            console.error('更新固件失败:', error);
            return {
                success: false,
                error: error.message || '固件更新后端连接失败，请检查服务器状态',
                data: null
            };
        }
    }
    
    /**
     * 设置机器人参数
     */
    async setRobotParameters(params) {
        try {
            // 尝试调用后端API设置机器人参数
            const response = await this.request('/robot/parameters', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(params)
            });
            
            if (!response.ok) {
                throw new Error(`HTTP错误: ${response.status}`);
            }
            
            const data = await response.json();
            return {
                success: true,
                data: data
            };
        } catch (error) {
            console.error('设置机器人参数失败:', error);
            return {
                success: false,
                error: error.message || '机器人参数设置后端连接失败，请检查服务器状态',
                data: null
            };
        }
    }
    
    /**
     * 重启机器人
     */
    async rebootRobot() {
        try {
            // 尝试调用后端API重启机器人
            const response = await this.request('/robot/reboot', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ action: 'reboot' })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP错误: ${response.status}`);
            }
            
            const data = await response.json();
            return {
                success: true,
                data: data
            };
        } catch (error) {
            console.error('重启机器人失败:', error);
            return {
                success: false,
                error: error.message || '机器人重启后端连接失败，请检查服务器状态',
                data: null
            };
        }
    }
    
    /**
     * 校准传感器
     */
    async calibrateSensors() {
        try {
            // 尝试调用后端API校准传感器
            const response = await this.request('/robot/calibrate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ action: 'calibrate_sensors' })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP错误: ${response.status}`);
            }
            
            const data = await response.json();
            return {
                success: true,
                data: data
            };
        } catch (error) {
            console.error('校准传感器失败:', error);
            return {
                success: false,
                error: error.message || '传感器校准后端连接失败，请检查服务器状态',
                data: null
            };
        }
    }
    
    /**
     * 运行系统自检
     */
    async runSelfDiagnostic() {
        try {
            const response = await this.request('/system/diagnostic', {
                method: 'GET'
            });

            if (!response.ok) {
                throw new Error(`HTTP错误: ${response.status}`);
            }

            const data = await response.json();
            return {
                success: true,
                data: data
            };
        } catch (error) {
            console.error('运行系统自检失败:', error);
            return {
                success: false,
                error: error.message || '系统自检后端连接失败，请检查服务器状态',
                data: null
            };
        }
    }
    
    /**
     * 导出诊断数据
     */
    async exportDiagnosticData() {
        try {
            const response = await this.request('/system/export_diagnostic', {
                method: 'GET'
            });

            if (!response.ok) {
                throw new Error(`HTTP错误: ${response.status}`);
            }

            const data = await response.json();
            return {
                success: true,
                data: data
            };
        } catch (error) {
            console.error('导出诊断数据失败:', error);
            return {
                success: false,
                error: error.message || '诊断数据导出后端连接失败，请检查服务器状态',
                data: null
            };
        }
    }
    
    /**
     * 重置机器人配置
     */
    async resetRobotConfig() {
        try {
            // 尝试调用后端API重置机器人配置
            const response = await this.request('/robot/config/reset', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ action: 'reset_config' })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP错误: ${response.status}`);
            }
            
            const data = await response.json();
            return {
                success: true,
                data: data
            };
        } catch (error) {
            console.error('重置机器人配置失败:', error);
            return {
                success: false,
                error: error.message || '机器人配置重置后端连接失败，请检查服务器状态',
                data: null
            };
        }
    }
    
    /**
     * 获取LNN状态
     */
    async getLNNStatus() {
        try {
            // 尝试调用后端API获取LNN状态
            const response = await this.request('/lnn/status');
            
            if (!response.ok) {
                throw new Error(`HTTP错误: ${response.status}`);
            }
            
            const data = await response.json();
            return {
                success: true,
                data: data
            };
        } catch (error) {
            console.error('获取LNN状态失败:', error);
            return {
                success: false,
                error: error.message || 'LNN状态后端连接失败，请检查服务器状态',
                data: null
            };
        }
    }
    
    /**
     * 保存LNN参数
     */
    async saveLNNParameters(parameters) {
        try {
            // 尝试调用后端API保存LNN参数
            const response = await this.request('/lnn/parameters', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(parameters)
            });
            
            if (!response.ok) {
                throw new Error(`HTTP错误: ${response.status}`);
            }
            
            const data = await response.json();
            return {
                success: true,
                data: data
            };
        } catch (error) {
            console.error('保存LNN参数失败:', error);
            return {
                success: false,
                error: error.message || 'LNN参数后端连接失败，请检查服务器状态',
                data: null
            };
        }
    }
    
    /**
     * 开始训练任务 - C-006修复: 改为startTraining别名，消除重复代码
     */
    async startTrainingJob(config) {
        return await this.startTraining(config);
    }
    
    /**
     * 【FE-003修复】暂停训练 - 已弃用，请使用 trainingPause()
     * @deprecated 自2026-07起弃用，转发到 trainingPause() 新方法
     *              保留此方法以保证 agi-controller.js / voice-command.js 等旧调用方兼容
     */
    async pauseTraining() {
        console.warn('[ApiService] pauseTraining() 已弃用，请迁移到 trainingPause()');
        return await this.trainingPause();
    }
    
    /**
     * 【FE-003修复】停止训练任务 - 已弃用，请使用 trainingStop()
     * @deprecated 自2026-07起弃用，转发到 trainingStop() 新方法
     *              C-005修复: 保留以兼容 agi-controller.js 和 voice-command.js
     */
    async stopTraining() {
        console.warn('[ApiService] stopTraining() 已弃用，请迁移到 trainingStop()');
        return await this.trainingStop();
    }

    /**
     * 【FE-003修复】停止训练任务 - 已弃用，请使用 trainingStop()
     * @deprecated 自2026-07起弃用，转发到 trainingStop() 新方法
     */
    async stopTrainingJob() {
        console.warn('[ApiService] stopTrainingJob() 已弃用，请迁移到 trainingStop()');
        return await this.trainingStop();
    }
    
    /**
     * 【FE-003修复】恢复训练任务 - 已弃用，请使用 trainingResume()
     * @deprecated 自2026-07起弃用，转发到 trainingResume() 新方法
     */
    async resumeTrainingJob() {
        console.warn('[ApiService] resumeTrainingJob() 已弃用，请迁移到 trainingResume()');
        return await this.trainingResume();
    }
    
    /**
     * 导出训练数据
     */
    async exportTrainingData() {
        try {
            // 尝试调用后端API导出训练数据
            const response = await this.request('/training/export', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            if (!response.ok) {
                throw new Error(`HTTP错误: ${response.status}`);
            }
            
            const data = await response.json();
            return {
                success: true,
                data: data
            };
        } catch (error) {
            console.error('导出训练数据失败:', error);
            return {
                success: false,
                error: error.message || '训练数据导出后端连接失败，请检查服务器状态',
                data: null
            };
        }
    }
    
    /**
     * 清空训练日志
     */
    async clearTrainingLog() {
        try {
            // 尝试调用后端API清空训练日志
            const response = await this.request('/training/log/clear', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            if (!response.ok) {
                throw new Error(`HTTP错误: ${response.status}`);
            }
            
            const data = await response.json();
            return {
                success: true,
                data: data
            };
        } catch (error) {
            console.error('清空训练日志失败:', error);
            return {
                success: false,
                error: error.message || '训练日志后端连接失败，请检查服务器状态',
                data: null
            };
        }
    }
    
    /**
     * 获取服务器统计信息
     */
    async getServerStats() {
        try {
            const response = await this.request('/stats');
            if (!response.ok) {
                throw new Error(`HTTP错误: ${response.status}`);
            }
            const data = await response.json();
            return {
                success: true,
                data: data
            };
        } catch (error) {
            console.error('获取服务器统计信息失败:', error);
            return {
                success: false,
                error: error.message,
                data: null
            };
        }
    }

    /**
     * 获取模型信息 - C-012修复: 使用正确的/model/info端点
     */
    async getModelInfo() {
        try {
            const response = await this.request('/model/info');
            if (!response.ok) throw new Error('HTTP错误: ' + response.status);
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * 获取模型状态 - C-011修复: 使用/model/info端点而非/status
     */
    async getModelStatus() {
        try {
            const response = await this.request('/model/info');
            if (!response.ok) throw new Error('HTTP错误: ' + response.status);
            const data = await response.json();
            return {
                success: true,
                data: {
                    model: data.lnn || data.model || data,
                    status: data.status || 'unknown',
                    uptime: data.uptime || 0
                }
            };
        } catch (error) {
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * 获取GPU状态
     */
    async getGpuStatus() {
        try {
            const response = await this.request('/gpu/status');
            if (!response.ok) {
                throw new Error(`HTTP错误: ${response.status}`);
            }
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error('获取GPU状态失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * 查询知识库
     * API路径标准化 (M-039): /knowledge
     * 注: 后端知识库主查询端点为 /knowledge (非 /knowledge/query)
     */
    async queryKnowledge(query) {
        try {
            const response = await this.request('/knowledge' + (query ? '?q=' + encodeURIComponent(query) : ''));
            if (!response.ok) {
                throw new Error('HTTP错误: ' + response.status);
            }
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error('查询知识库失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * 添加知识到知识库
     */
    async addKnowledge(entry) {
        try {
            const response = await this.request('/knowledge/add', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    subject: entry.subject || entry.content || '',
                    predicate: entry.predicate || '',
                    object: entry.object || '',
                    type: entry.type || 'fact',
                    category: entry.category || 'general',
                    tags: entry.tags || [],
                    confidence: entry.confidence !== undefined ? entry.confidence : -1
                })
            });
            if (!response.ok) {
                throw new Error('HTTP错误: ' + response.status);
            }
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error('添加知识失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * 知识图谱统计信息 (P3修复)
     * GET /kg/stats → 返回节点数、边数、密度、聚类系数等统计指标
     */
    async kgStats() {
        try {
            const response = await this.request('/kg/stats', { method: 'GET' });
            if (!response.ok) throw new Error('HTTP错误: ' + response.status);
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error('知识图谱统计失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * 知识图谱可视化数据 (P3修复)
     * GET /kg/visualize → 返回完整图谱可视化数据(节点坐标+边关系+布局信息)
     */
    async kgVisualize() {
        try {
            const response = await this.request('/kg/visualize', { method: 'GET' });
            if (!response.ok) throw new Error('HTTP错误: ' + response.status);
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error('知识图谱可视化数据获取失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * 知识图谱语义搜索 (P3修复)
     * POST /kg/search → body: { query: "...", top_k: N }
     */
    async kgSearch(query, topK) {
        try {
            const response = await this.request('/kg/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: query || '', top_k: topK || 20 })
            });
            if (!response.ok) throw new Error('HTTP错误: ' + response.status);
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error('知识图谱语义搜索失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /* ==================== I-S01: 知识图谱扩展API ==================== */

    /**
     * 添加知识图谱节点/边
     * POST /api/kg/add → body: { nodes: [...], edges: [...] }
     */
    async addKnowledgeGraph(data) {
        try {
            const response = await this.request('/kg/add', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data || {})
            });
            if (!response.ok) throw new Error('HTTP错误: ' + response.status);
            const result = await response.json();
            return { success: true, data: result };
        } catch (error) {
            console.error('添加知识图谱失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * 查询知识图谱（使用后端 kg/search 端点）
     * POST /api/kg/search → 返回匹配的节点和边
     */
    async queryKnowledgeGraph(query) {
        try {
            const response = await this.request('/kg/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: query || '' })
            });
            if (!response.ok) throw new Error('HTTP错误: ' + response.status);
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error('查询知识图谱失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * 删除知识图谱节点/边
     * DELETE /api/kg/delete → body: { id: "..." }
     */
    async deleteKnowledgeGraph(id) {
        try {
            const response = await this.request('/kg/delete', {
                method: 'DELETE',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ id: id })
            });
            if (!response.ok) throw new Error('HTTP错误: ' + response.status);
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error('删除知识图谱失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * 扩展知识图谱（使用后端 kg/path 端点实现多跳扩展）
     * POST /api/kg/path → body: { source: "...", target: "..." }
     */
    async expandKnowledgeGraph(data) {
        try {
            const response = await this.request('/kg/path', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data || {})
            });
            if (!response.ok) throw new Error('HTTP错误: ' + response.status);
            const result = await response.json();
            return { success: true, data: result };
        } catch (error) {
            console.error('扩展知识图谱失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * 合并知识图谱（使用后端 kg/search 端点查询后前端合并）
     * POST /api/kg/search → body: { query: "..." }
     */
    async mergeKnowledgeGraph(data) {
        try {
            const response = await this.request('/kg/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data || {})
            });
            if (!response.ok) throw new Error('HTTP错误: ' + response.status);
            const result = await response.json();
            return { success: true, data: result };
        } catch (error) {
            console.error('合并知识图谱失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * 知识图谱可视化数据
     * GET /api/kg/visualize → 返回完整图谱可视化数据
     */
    async visualizeKnowledgeGraph() {
        try {
            const response = await this.request('/kg/visualize', { method: 'GET' });
            if (!response.ok) throw new Error('HTTP错误: ' + response.status);
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error('知识图谱可视化失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * 知识图谱搜索
     * POST /api/kg/search → 返回匹配的节点和边
     */
    async searchKnowledgeGraph(query) {
        try {
            const params = (query && query.trim()) ? { q: query.trim() } : {};
            const response = await this.request('/kg/search', { method: 'POST', body: JSON.stringify(params) });
            if (!response.ok) throw new Error('HTTP错误: ' + response.status);
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error('知识图谱搜索失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /* L-013修复: 重命名方法，明确语义为机器人传感器而非通用传感器状态 */
    async getRobotSensorStatus() {
        try {
            const response = await this.request('/robot/sensor');
            if (!response.ok) {
                throw new Error('HTTP\u9519\u8BEF: ' + response.status);
            }
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.warn('传感器状态:', error.message);
            return { success: false, error: error.message, data: null };
        }
    }

    /* 向后兼容别名 */
    async getSensorStatus() { return await this.getRobotSensorStatus(); }

    async getSecurityStatus() {
/* 原调用/status通用端点改为/safety/status安全专用端点 */
        try {
            const response = await this.request('/safety/status');
            if (!response.ok) {
                throw new Error('HTTP错误: ' + response.status);
            }
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error('获取安全状态失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    async softStop(target) {
        try {
            var payload = JSON.stringify({ target: target || 'all' });
            var resp = await this.request('/safety/soft_stop', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: payload
            });
            var d = await resp.json();
            return { success: resp.ok, data: d };
        } catch (e) {
            return { success: false, error: e.message };
        }
    }

    async resetSafetyState() {
        try {
            var resp = await this.request('/safety/reset', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({})
            });
            var d = await resp.json();
            return { success: resp.ok, data: d };
        } catch (e) {
            return { success: false, error: e.message };
        }
    }

    async getSafetyEvents(limit) {
        try {
            var q = (limit && limit > 0) ? ('?limit=' + limit) : '';
            var resp = await this.request('/safety/events' + q, { method: 'GET' });
            var d = await resp.json();
            return { success: resp.ok, data: d };
        } catch (e) {
            return { success: false, error: e.message };
        }
    }

    /**
 * 设置安全边界配置
     * 向 /api/safety/bounds 发送POST请求更新机器人安全边界参数
     *
     * @param {Object} bounds - 安全边界配置对象，包含以下可选字段：
     *   @param {number} [bounds.maxSpeed]       - 最大速度限制 (m/s)，默认2.5
     *   @param {number} [bounds.maxAccel]       - 最大加速度限制 (m/s²)，默认5.0
     *   @param {number} [bounds.maxTorque]      - 最大扭矩限制 (Nm)，默认150.0
     *   @param {number} [bounds.safetyZoneRadius] - 安全区域半径 (m)，默认1.2
     *   @param {number} [bounds.collisionDistance] - 最小碰撞距离 (m)，默认0.3
     * @returns {Promise<{success:boolean, data:Object, message:string}>}
     *   成功时 data.bounds 包含更新后的边界值
     */
    async setSafetyBounds(bounds) {
        try {
            var resp = await this.request('/safety/bounds', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(bounds)
            });
            var data = await resp.json();
            return { success: resp.ok, data: data, message: data && data.message };
        } catch (e) {
            return { success: false, error: e.message };
        }
    }

    /**
     * I-S04: 获取安全边界配置（使用后端 safety/bounds 端点）
     * GET /api/safety/bounds → 返回当前安全边界参数
     */
    async getSafetyBoundaryConfig() {
        try {
            var resp = await this.request('/safety/bounds', { method: 'GET' });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) {
            return { success: false, error: e.message };
        }
    }

    /**
     * 获取认知状态（含心智理论数据）
     */
    async getCognitionStatus() {
        try {
            const response = await this.request('/cognition/status');
            if (!response.ok) {
                throw new Error('HTTP错误: ' + response.status);
            }
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error('获取认知状态失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * AGI自主决策
     */
    async agiDecide(goal, context) {
        try {
            const response = await this.request('/agi/decide', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                /* FIX-1: 添加 goal_priority, risk_tolerance 匹配后端parse_json */
                body: JSON.stringify({ goal: goal || '系统优化', goal_priority: 0.8, risk_tolerance: 0.5, context: context || {} })
            });
            if (!response.ok) throw new Error('HTTP错误: ' + response.status);
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error('AGI决策失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * AGI在线学习
     */
    async agiLearn(data, mode) {
        try {
            const response = await this.request('/agi/learn', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                /* FIX-1: mode→target, 添加store_knowledge匹配后端 */
                body: JSON.stringify({ data: data || '', target: mode || 'auto', store_knowledge: 1 })
            });
            if (!response.ok) throw new Error('HTTP错误: ' + response.status);
            const dataJson = await response.json();
            return { success: true, data: dataJson };
        } catch (error) {
            console.error('AGI学习失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * AGI进化演化
     */
    async agiEvolve(config) {
        try {
            const response = await this.request('/agi/evolve', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                /* FIX-1: mode→target 匹配后端parse_json_string("target",...) */
                body: JSON.stringify({ generations: (config && config.generations) || 10, target: (config && config.mode) || 'auto' })
            });
            if (!response.ok) throw new Error('HTTP错误: ' + response.status);
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error('AGI进化失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * LNN记忆读写（LNN内部状态存取）
     */
    async agiMemory(action, key, data) {
        try {
            const response = await this.request('/agi/memory', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                /* FIX-1: action→op, data→value, 添加priority/strength */
                body: JSON.stringify({ op: action || 'read', key: key || 'latest', value: data || '', priority: 5, strength: 0.8 })
            });
            if (!response.ok) throw new Error('HTTP错误: ' + response.status);
            const dataJson = await response.json();
            return { success: true, data: dataJson };
        } catch (error) {
            console.error('AGI记忆失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * AGI自主规划
     */
    async agiPlan(goal, horizon) {
        try {
            const response = await this.request('/agi/plan', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                /* FIX-1: horizon→steps, 添加long_term */
                body: JSON.stringify({ goal: goal || '系统优化', steps: horizon || 10, long_term: 0 })
            });
            if (!response.ok) throw new Error('HTTP错误: ' + response.status);
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error('AGI规划失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * 发送对话消息
     * API路径标准化 (M-037): /dialogue/send - 后端已注册此别名路由
     */
    async sendDialogueMessage(message, config) {
        try {
            const response = await this.request('/dialogue/send', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    message: message,
                    temperature: Math.round((config.temperature || 0.8) * 10),
                    max_length: config.maxLength || 128,
                    top_k: config.topK || 10,
                    memory: config.memoryRounds || 0
                })
            });
            if (!response.ok) {
                throw new Error(`HTTP错误: ${response.status}`);
            }
            const data = await response.json();
            return {
                success: true,
                data: data
            };
        } catch (error) {
            console.error('发送对话消息失败:', error);
            return {
                success: false,
                error: error.message || '对话后端连接失败，请检查服务器状态',
                data: null
            };
        }
    }

    /**
     * 获取对话历史
     * API路径标准化 (M-038): /dialogue/history
     */
    async getDialogueHistory() {
        try {
            const response = await this.request('/dialogue/history');
            if (!response.ok) {
                throw new Error(`HTTP错误: ${response.status}`);
            }
            const data = await response.json();
            return {
                success: true,
                data: data
            };
        } catch (error) {
            console.error('获取对话历史失败:', error);
            return {
                success: false,
                error: error.message || '对话历史后端连接失败，请检查服务器状态',
                data: null
            };
        }
    }

    /**
     * 清空对话历史
     * API路径标准化 (M-038): /dialogue/clear
     */
    async clearDialogueHistory() {
        try {
            const response = await this.request('/dialogue/clear', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            if (!response.ok) {
                throw new Error(`HTTP错误: ${response.status}`);
            }
            const data = await response.json();
            return {
                success: true,
                data: data
            };
        } catch (error) {
            console.error('清空对话历史失败:', error);
            return {
                success: false,
                error: error.message || '对话清空后端连接失败，请检查服务器状态',
                data: null
            };
        }
    }

    /**
     * 获取AGI功能状态
     */
    async getAgiFeatureStatus() {
        try {
            const response = await this.request('/agi/features');
            if (!response.ok) {
                throw new Error(`HTTP错误: ${response.status}`);
            }
            const data = await response.json();
            return {
                success: true,
                data: data
            };
        } catch (error) {
            console.error('获取AGI功能状态失败:', error);
            return {
                success: false,
                error: error.message || 'AGI功能后端连接失败，请检查服务器状态',
                data: null
            };
        }
    }

    /**
     * 切换AGI功能
     */
    async toggleAgiFeature(feature, enabled) {
        try {
            const response = await this.request('/agi/feature/toggle', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    feature: feature,
                    enabled: enabled
                })
            });
            if (!response.ok) {
                throw new Error(`HTTP错误: ${response.status}`);
            }
            const data = await response.json();
            return {
                success: true,
                data: data
            };
        } catch (error) {
            console.error('切换AGI功能失败:', error);
            return {
                success: false,
                error: error.message || 'AGI功能切换后端连接失败，请检查服务器状态',
                data: null
            };
        }
    }

    // ================================================================
    // 音频与语音识别API
    // ================================================================

    /**
     * 语音识别：发送音频数据，返回识别文本
     */
    async recognizeAudio(audioBlob) {
        try {
            const formData = new FormData;
            formData.append('audio', audioBlob, 'recording.wav');
            const response = await this.request('/audio/recognize', {
                method: 'POST',
                body: formData
            });
            if (!response.ok) throw new Error(`HTTP错误: ${response.status}`);
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error('语音识别失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * 语音合成：发送文本，返回音频数据
     */
    async synthesizeSpeech(text, speed) {
        try {
            const response = await this.request('/tts/synthesize', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text, speed: speed || 1.0 })
            });
            if (!response.ok) throw new Error(`HTTP错误: ${response.status}`);
            const blob = await response.blob;
            return { success: true, data: blob, url: URL.createObjectURL(blob) };
        } catch (error) {
            console.error('语音合成失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * I-L02: 获取音频设备列表
     * GET /api/audio/devices → 返回可用音频输入/输出设备列表
     */
    async getAudioDevices() {
        try {
            var resp = await this.request('/audio/devices', { method: 'GET' });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    // ================================================================
    // 多模态对话API
    // ================================================================

    /**
     * 发送多模态对话消息（文本+图像+音频）
     */
    async sendMultimodalDialogue(message, options) {
        try {
            const payload = {
                message: message,
                temperature: options.temperature || 0.8,
                max_length: options.maxLength || 128,
                top_k: options.topK || 10,
                memory_rounds: options.memoryRounds || 0
            };
            if (options.images && options.images.length > 0) {
                payload.images = options.images;
            }
            if (options.audio && options.audio.length > 0) {
                payload.audio = options.audio;
            }
            /* API路径标准化 (M-038): /dialogue/multimodal */
            const response = await this.request('/dialogue/multimodal', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            if (!response.ok) throw new Error(`HTTP错误: ${response.status}`);
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error('多模态对话失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * AGI控制任务执行（机器人/计算机/设备控制）
     */
    async executeAGITask(task) {
        try {
            const response = await this.request('/agi/execute', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(task)
            });
            if (!response.ok) throw new Error(`HTTP错误: ${response.status}`);
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error('AGI任务执行失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * AGI任务执行状态查询
     */
    async getAGITaskStatus(taskId) {
        try {
            const response = await this.request('/agi/task/status', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ task_id: taskId })
            });
            if (!response.ok) throw new Error(`HTTP错误: ${response.status}`);
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error('AGI任务状态查询失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * 发送设备控制命令（经AGI系统处理）
     */
    async sendDeviceCommand(deviceType, command, params) {
        try {
            const response = await this.request('/device/command', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    device_type: deviceType,
                    command: command,
                    params: params || {}
                })
            });
            if (!response.ok) throw new Error(`HTTP错误: ${response.status}`);
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error('设备命令发送失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    // ================================================================
    // ROS/Gazebo 集成API
    // ================================================================

    /**
     * 添加仿真机器人到3D视图中
 *修复: 原调/simulation/robot_control(不存在的路由)→改为/simulation/robot/add
     * @param {object} params - { x, y, z, urdf, model_name }
     */
    async addSimulationRobot(params) {
        try {
            var pos = params || { x: 0, y: 0, z: 0 };
            const response = await this.request('/simulation/robot/add', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: params.model_name || 'custom_robot', urdf: params.urdf || null })
            });
            if (!response.ok) throw new Error('HTTP ' + response.status);
            const data = await response.json();
            return { success: true, robot: data.robot || { x: pos.x, y: pos.y, z: pos.z, name: params.model_name || 'robot' } };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /**
     * 从仿真中移除机器人
     */
    async removeSimulationRobot(robotId) {
        try {
            const response = await this.request('/simulation/robot_control', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action: 'remove_robot', robot_id: robotId })
            });
            return { success: response.ok };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /**
     * 获取机器人列表（增强版，包含ROS机器人）
     */
    async getRobotList() {
        try {
            const response = await this.request('/robot/list');
            if (!response.ok) throw new Error(`HTTP错误: ${response.status}`);
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error('获取机器人列表失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * 获取ROS系统状态
     */
    async getRosStatus() {
        try {
            const response = await this.request('/ros/status');
            if (!response.ok) throw new Error(`HTTP错误: ${response.status}`);
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error('获取ROS状态失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * 配置ROS Master连接
     */
    async configureRos(config) {
        try {
            const response = await this.request('/ros/configure', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config)
            });
            if (!response.ok) throw new Error(`HTTP错误: ${response.status}`);
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error('配置ROS失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * 获取ROS节点列表
     */
    async getRosNodes() {
        try {
            const response = await this.request('/ros/nodes');
            if (!response.ok) throw new Error(`HTTP错误: ${response.status}`);
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error('获取ROS节点失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * 获取ROS主题列表
     */
    async getRosTopics() {
        try {
            const response = await this.request('/ros/topics');
            if (!response.ok) throw new Error(`HTTP错误: ${response.status}`);
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error('获取ROS主题失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * 获取传感器管道状态
     */
    async getSensorPipelineStatus() {
        try {
            const response = await this.request('/sensor/pipeline/status');
            if (!response.ok) throw new Error(`HTTP错误: ${response.status}`);
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error('获取传感器管道状态失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * Gazebo仿真控制
     */
    async controlGazebo(params) {
        try {
            const response = await this.request('/gazebo/control', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(params)
            });
            if (!response.ok) throw new Error(`HTTP错误: ${response.status}`);
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error('Gazebo控制失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * 机器人训练控制（ROS+Gazebo/PyBullet）
     */
    async controlRobotTraining(params) {
        try {
            const response = await this.request('/robot/training', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(params)
            });
            if (!response.ok) throw new Error(`HTTP错误: ${response.status}`);
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error('机器人训练控制失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * 增强版连接机器人（支持ROS/Gazebo）
     */
    async connectRobotEnhanced(robotId, connectGazebo, robotName) {
        try {
            const response = await this.request('/robot/connect', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    robot_id: robotId,
                    connect_gazebo: connectGazebo ? 1 : 0,
                    robot_name: robotName || ''
                })
            });
            if (!response.ok) throw new Error(`HTTP错误: ${response.status}`);
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error('连接机器人失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * 增强版断开机器人（支持ROS/Gazebo模型删除）
     */
    async disconnectRobotEnhanced(robotId, disconnectGazebo, modelName) {
        try {
            const response = await this.request('/robot/disconnect', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    robot_id: robotId,
                    disconnect_gazebo: disconnectGazebo ? 1 : 0,
                    model_name: modelName || ''
                })
            });
            if (!response.ok) throw new Error(`HTTP错误: ${response.status}`);
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error('断开机器人失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    // ================================================================
    // B1.1 设备管理API
    // ================================================================

    /**
     * 列出所有可用设备
     * API路径标准化 (M-040): /devices/list
     */
    async listDevices() {
        try {
            const response = await this.request('/devices/list', { method: 'GET' });
            if (!response.ok) throw new Error(`HTTP错误: ${response.status}`);
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.warn('列出设备:', error.message);
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * 注册设备到后端
     */
    async registerDevice(deviceId, type, name) {
        try {
            const response = await this.request('/devices/register', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ deviceId: deviceId, type: type, name: name })
            });
            if (!response.ok) throw new Error(`HTTP错误: ${response.status}`);
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error('注册设备失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * 从后端注销设备
     */
    async unregisterDevice(deviceId) {
        try {
            const response = await this.request('/devices/unregister', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ deviceId: deviceId })
            });
            if (!response.ok) throw new Error(`HTTP错误: ${response.status}`);
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error('注销设备失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * 获取设备状态
     */
    async getDeviceStatus(deviceId) {
        try {
            const response = await this.request('/devices/status', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ deviceId: deviceId })
            });
            if (!response.ok) throw new Error(`HTTP错误: ${response.status}`);
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error('获取设备状态失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    // ================================================================
    // B1.2 音频流处理API
    // ================================================================

    /**
     * 发送实时音频流数据
     */
    async sendAudioStream(audioData) {
        try {
            const response = await this.request('/audio/stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/octet-stream' },
                body: audioData
            });
            if (!response.ok) throw new Error(`HTTP错误: ${response.status}`);
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error('发送音频流失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * 发送语音指令识别与执行
     */
    async sendAudioCommand(text, confidence) {
        try {
            const response = await this.request('/audio/command', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text, confidence: confidence || 0.0 })
            });
            if (!response.ok) throw new Error(`HTTP错误: ${response.status}`);
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error('发送语音指令失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    // ================================================================
    // B1.3 视频流处理API
    // ================================================================

    /**
     * 发送实时视频流数据
     */
    async sendVideoStream(frameData) {
        try {
            const response = await this.request('/video/stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/octet-stream' },
                body: frameData
            });
            if (!response.ok) throw new Error(`HTTP错误: ${response.status}`);
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error('发送视频流失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * 视频帧捕获与处理
     */
    async captureVideoFrame(cameraId) {
        try {
            const response = await this.request('/video/capture', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ cameraId: cameraId || 'default' })
            });
            if (!response.ok) throw new Error(`HTTP错误: ${response.status}`);
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error('视频帧捕获失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    // ================================================================
    // B1.4 媒体设备集成方法
    // ================================================================

    /**
     * 请求用户媒体设备权限并获取流
     * @param {Object} constraints - MediaStreamConstraints
     * @returns {Promise<Object>} { success, stream, error }
     */
    async getUserMedia(constraints) {
        try {
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                throw new Error('浏览器不支持媒体设备API');
            }
            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            return { success: true, stream: stream };
        } catch (error) {
            console.error('获取媒体设备失败:', error.message);
            if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
                return { success: false, error: '用户拒绝授权媒体设备', stream: null };
            }
            if (error.name === 'NotFoundError') {
                return { success: false, error: '未找到指定媒体设备', stream: null };
            }
            if (error.name === 'NotReadableError') {
                return { success: false, error: '媒体设备被其他应用占用', stream: null };
            }
            return { success: false, error: error.message, stream: null };
        }
    }

    /**
     * 开始发送媒体流到后端处理
     * @param {string} type - 'audio' | 'video' | 'audiovideo'
     * @param {MediaStream} stream - 媒体流
     * @param {Object} options - 配置项
     * @returns {Object} 控制句柄
     */
    startMediaStream(type, stream, options) {
        if (!stream) {
            return { success: false, error: '未提供媒体流' };
        }
        options = options || {};
        const frameInterval = options.frameInterval || 200;
        const audioInterval = options.audioInterval || 100;
        const streamId = 'stream_' + Date.now()+ '_' + Math.random().toString(36).substr(2, 8);
        const handle = {
            id: streamId,
            type: type,
            stream: stream,
            running: true,
            videoTimer: null,
            audioTimer: null,
            canvas: document.createElement('canvas'),
            audioContext: null,
            mediaRecorder: null,
            stop: function() {
                this.running = false;
                if (this.videoTimer) {
                    clearInterval(this.videoTimer);
                    this.videoTimer = null;
                }
                if (this.audioTimer) {
                    clearInterval(this.audioTimer);
                    this.audioTimer = null;
                }
                if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
                    this.mediaRecorder.stop();
                }
            }
        };
        var self = this;
        if (type === 'video' || type === 'audiovideo') {
            var videoTrack = stream.getVideoTracks()[0];
            if (videoTrack) {
                handle.canvas.width = options.width || 640;
                handle.canvas.height = options.height || 480;
                var ctx = handle.canvas.getContext('2d');
                var tempVideo = document.createElement('video');
                tempVideo.srcObject = new MediaStream([videoTrack]);
                tempVideo.play();
                handle.videoTimer = setInterval(function() {
                    if (!handle.running) return;
                    try {
                        ctx.drawImage(tempVideo, 0, 0, handle.canvas.width, handle.canvas.height);
                        handle.canvas.toBlob(function(blob) {
                            if (!handle.running) return;
                            self.sendVideoStream(blob);
                        }, 'image/jpeg', 0.7);
                    } catch (e) {
                        console.warn('视频帧捕获错误:', e.message);
                    }
                }, frameInterval);
            }
        }
        if (type === 'audio' || type === 'audiovideo') {
            var audioTrack = stream.getAudioTracks()[0];
            if (audioTrack) {
                try {
                    var audioStream = new MediaStream([audioTrack]);
                    var mimeType = 'audio/webm';
                    if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) {
                        mimeType = 'audio/webm;codecs=opus';
                    }
                    handle.mediaRecorder = new MediaRecorder(audioStream, {
                        mimeType: mimeType,
                        audioBitsPerSecond: 16000
                    });
                    handle.mediaRecorder.ondataavailable = function(event) {
                        if (!handle.running || !event.data || event.data.size === 0) return;
                        self.sendAudioStream(event.data);
                    };
                    handle.mediaRecorder.start(audioInterval);
                } catch (e) {
                    console.warn('音频流录制启动失败:', e.message);
                }
            }
        }
        return { success: true, handle: handle };
    }

    /**
     * 停止媒体流发送
     * @param {Object} handle - 从 startMediaStream 返回的控制句柄
     */
    stopMediaStream(handle) {
        if (handle && typeof handle.stop === 'function') {
            handle.stop();
            return { success: true };
        }
        return { success: false, error: '无效的控制句柄' };
    }

    /**
     * 从视频元素捕获单帧图像
     * @param {HTMLVideoElement} videoElement - 视频元素
     * @param {Object} options - { width, height, quality }
     * @returns {Object} { success, dataUrl, blob, width, height }
     */
    captureImage(videoElement, options) {
        if (!videoElement || !videoElement.videoWidth) {
            return { success: false, error: '视频元素无效或未就绪' };
        }
        options = options || {};
        var width = options.width || videoElement.videoWidth || 640;
        var height = options.height || videoElement.videoHeight || 480;
        var quality = options.quality || 0.85;
        var canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        var ctx = canvas.getContext('2d');
        if (options.mirrored) {
            ctx.translate(width, 0);
            ctx.scale(-1, 1);
        }
        ctx.drawImage(videoElement, 0, 0, width, height);
        var dataUrl = canvas.toDataURL('image/jpeg', quality);
        return {
            success: true,
            dataUrl: dataUrl,
            width: width,
            height: height,
            size: Math.round(dataUrl.length * 0.75)
        };
    }

    /**
     * 播放音频数据（通过扬声器输出）
     * @param {ArrayBuffer|Blob|string} audioData - 音频数据（ArrayBuffer/Blob/Base64）
     * @param {Object} options - { format, volume }
     * @returns {Object} { success, audioElement, error }
     */
    playAudio(audioData, options) {
        try {
            options = options || {};
            var blob;
            if (audioData instanceof ArrayBuffer) {
                blob = new Blob([audioData], { type: options.format || 'audio/wav' });
            } else if (audioData instanceof Blob) {
                blob = audioData;
            } else if (typeof audioData === 'string') {
                var byteStr = atob(audioData.split(',')[1] || audioData);
                var byteArr = new Uint8Array(byteStr.length);
                for (var i = 0; i < byteStr.length; i++) {
                    byteArr[i] = byteStr.charCodeAt(i);
                }
                blob = new Blob([byteArr], { type: options.format || 'audio/wav' });
            } else {
                throw new Error('不支持的音频数据格式');
            }
            var url = URL.createObjectURL(blob);
            var audio = new Audio(url);
            if (options.volume !== undefined) {
                audio.volume = Math.max(0, Math.min(1, options.volume));
            }
            audio.onended = function() {
                URL.revokeObjectURL(url);
            };
            audio.play.catch(function(err) {
                console.error('音频播放失败:', err.message);
            });
            return { success: true, audioElement: audio };
        } catch (error) {
            console.error('播放音频失败:', error.message);
            return { success: false, error: error.message, audioElement: null };
        }
    }

    /**
     * 扬声器测试 - 播放测试音调
     * @param {number} duration - 持续时间(ms)
     * @param {number} frequency - 频率(Hz)
     * @returns {Object} { success, error }
     */
    testSpeaker(duration, frequency) {
        try {
            if (!window.AudioContext && !window.webkitAudioContext) {
                return { success: false, error: '浏览器不支持AudioContext' };
            }
            var AudioCtx = window.AudioContext || window.webkitAudioContext;
            var ctx = new AudioCtx;
            duration = duration || 500;
            frequency = frequency || 440;
            var oscillator = ctx.createOscillator();
            var gainNode = ctx.createGain();
            oscillator.type = 'sine';
            oscillator.frequency.value = frequency;
            gainNode.gain.setValueAtTime(0.3, ctx.currentTime);
            gainNode.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + duration / 1000);
            oscillator.connect(gainNode);
            gainNode.connect(ctx.destination);
            oscillator.start(ctx.currentTime);
            oscillator.stop(ctx.currentTime + duration / 1000);
            oscillator.onended = function() {
                ctx.close;
            };
            return { success: true, message: '测试音调已播放' };
        } catch (error) {
            console.error('扬声器测试失败:', error.message);
            return { success: false, error: error.message };
        }
    }

    /**
     * 枚举所有可用媒体设备
     * @returns {Promise<Object>} { success, devices: { audioinput, audiooutput, videoinput }, error }
     */
    async enumerateMediaDevices() {
        try {
            if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
                throw new Error('浏览器不支持枚举媒体设备');
            }
            var devices = await navigator.mediaDevices.enumerateDevices();
            var result = {
                audioinput: [],
                audiooutput: [],
                videoinput: []
            };
            for (var i = 0; i < devices.length; i++) {
                var d = devices[i];
                if (result[d.kind]) {
                    result[d.kind].push({
                        deviceId: d.deviceId,
                        groupId: d.groupId,
                        label: d.label || d.kind + '#' + result[d.kind].length
                    });
                }
            }
            return { success: true, devices: result };
        } catch (error) {
            console.error('枚举媒体设备失败:', error.message);
            return { success: false, error: error.message, devices: null };
        }
    }

    /**
     * 请求屏幕共享流
     * @param {Object} options - { video: true, audio: true }
     * @returns {Promise<Object>} { success, stream, error }
     */
    async getDisplayMedia(options) {
        try {
            if (!navigator.mediaDevices || !navigator.mediaDevices.getDisplayMedia) {
                throw new Error('浏览器不支持屏幕共享');
            }
            var stream = await navigator.mediaDevices.getDisplayMedia(options || { video: true, audio: false });
            return { success: true, stream: stream };
        } catch (error) {
            if (error.name === 'NotAllowedError') {
                return { success: false, error: '用户取消屏幕共享', stream: null };
            }
            return { success: false, error: error.message, stream: null };
        }
    }

    /* ==================== 实物教学系统API ==================== */

    /**
     * 看一眼就学习 - 视觉单样本学习
     * @param {string} name - 物品名称
     * @param {number} category - 分类 (0=物体,1=动作,2=属性,3=数量,4=关系)
     * @param {string} image - base64图像数据
     * @returns {Promise<Object>}
     */
    async teachLookAndLearn(name, category, image) {
        try {
            var resp = await this.request('/teach/look_and_learn', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: name, category: category, image: image })
            });
            var data = await resp.json();
            return { success: resp.ok, data: data, error: resp.ok ? null : (data.error || '请求失败') };
        } catch (e) {
            return { success: false, error: e.message, data: null };
        }
    }

    /**
     * 说出来就关联 - 语音+视觉关联学习
     * @param {string} name - 物品名称
     * @param {number} category - 分类
     * @param {string} image - base64图像
     * @param {string|null} audio - base64音频
     * @returns {Promise<Object>}
     */
    async teachSayAndAssociate(name, category, image, audio) {
        try {
            var resp = await this.request('/teach/say_and_associate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: name, category: category, image: image, audio: audio })
            });
            var data = await resp.json();
            return { success: resp.ok, data: data, error: resp.ok ? null : (data.error || '请求失败') };
        } catch (e) {
            return { success: false, error: e.message, data: null };
        }
    }

    /**
     * 摸一下知属性 - 触觉传感器学习
     * @param {string} name - 物品名称
     * @param {number} category - 分类
     * @param {Object} sensor - { hardness, temperature, texture, weight }
     * @returns {Promise<Object>}
     */
    async teachTouchAndUnderstand(name, category, sensor) {
        try {
            var resp = await this.request('/teach/touch_and_understand', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: name, category: category, sensor: sensor })
            });
            var data = await resp.json();
            return { success: resp.ok, data: data, error: resp.ok ? null : (data.error || '请求失败') };
        } catch (e) {
            return { success: false, error: e.message, data: null };
        }
    }

    /**
     * 数数字概括 - 视觉数量概念学习
     * @param {string} name - 数量概念名称
     * @param {string} image - base64图像
     * @returns {Promise<Object>}
     */
    async teachCountAndGeneralize(name, image) {
        try {
            var resp = await this.request('/teach/count_and_generalize', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: name, category: 3, image: image })
            });
            var data = await resp.json();
            return { success: resp.ok, data: data, error: resp.ok ? null : (data.error || '请求失败') };
        } catch (e) {
            return { success: false, error: e.message, data: null };
        }
    }

    /**
     * 测试概念识别 - 多模态匹配
     * @param {string} image - base64图像
     * @returns {Promise<Object>}
     */
    async teachTestConcept(image) {
        try {
            var resp = await this.request('/teach/test_concept', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: image })
            });
            var data = await resp.json();
            return { success: resp.ok, data: data, error: resp.ok ? null : (data.error || '请求失败') };
        } catch (e) {
            return { success: false, error: e.message, data: null };
        }
    }

    /**
     * 获取已学概念列表
     * @returns {Promise<Object>}
     */
    async teachGetConcepts() {
        try {
            var resp = await this.request('/teach/get_concepts', { method: 'GET' });
            var data = await resp.json();
            return { success: resp.ok, data: data, error: resp.ok ? null : (data.error || '请求失败') };
        } catch (e) {
            return { success: false, error: e.message, data: null };
        }
    }

    /**
     * 清空所有已学概念
     * @returns {Promise<Object>}
     */
    async teachClearAllConcepts() {
        try {
            var resp = await this.request('/teach/clear_all_concepts', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({})
            });
            var data = await resp.json();
            return { success: resp.ok, data: data, error: resp.ok ? null : (data.error || '请求失败') };
        } catch (e) {
            return { success: false, error: e.message, data: null };
        }
    }

    /**
     * 删除指定概念
     * @param {string} name - 概念名称
     * @returns {Promise<Object>}
     */
    async teachClearConcept(name) {
        try {
            var resp = await this.request('/teach/clear_concept', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: name })
            });
            var data = await resp.json();
            return { success: resp.ok, data: data, error: resp.ok ? null : (data.error || '请求失败') };
        } catch (e) {
            return { success: false, error: e.message, data: null };
        }
    }

    /* ==================== 机器人操作电脑系统API ==================== */

    /**
     * 分析屏幕UI元素
     */
    async robotAnalyzeScreen(image) {
        try {
            var resp = await this.request('/robot/analyze_screen', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: image })
            });
            var data = await resp.json();
            return { success: resp.ok, data: data, error: resp.ok ? null : (data.error || '分析失败') };
        } catch (e) {
            return { success: false, error: e.message, data: null };
        }
    }

    /**
     * 执行单个动作
     * @param {Object} action - { action_type, x, y, key_code, text, ... }
     */
    async robotExecuteAction(action) {
        try {
            var resp = await this.request('/robot/execute_action', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(action)
            });
            var data = await resp.json();
            return { success: resp.ok, data: data, error: resp.ok ? null : (data.error || '动作执行失败') };
        } catch (e) {
            return { success: false, error: e.message, data: null };
        }
    }

    /**
     * 执行任务计划
     * @param {string} task - 任务描述
     * @param {string} screen - base64屏幕截图
     * @param {Object} config - 配置 { speed, vision_guide }
     */
    async robotExecuteTask(task, screen, config) {
        try {
            var resp = await this.request('/robot/execute_task', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ task: task, screen: screen, config: config || {} })
            });
            var data = await resp.json();
            return { success: resp.ok, data: data, error: resp.ok ? null : (data.error || '任务执行失败') };
        } catch (e) {
            return { success: false, error: e.message, data: null };
        }
    }

    /**
     * 停止当前任务
     */
    async robotStopTask() {
        try {
            var resp = await this.request('/robot/stop_task', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: '{}'
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) {
            return { success: false, error: e.message };
        }
    }

    /**
     * 从演示中学习
     * @param {Array} screen_sequence - 屏幕帧序列
     * @param {Array} action_sequence - 动作序列
     * @param {string} task_label - 任务标签
     */
    async robotLearnFromDemo(screen_sequence, action_sequence, task_label) {
        try {
            var resp = await this.request('/robot/learn_from_demo', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    screen_sequence: screen_sequence,
                    action_sequence: action_sequence,
                    task_label: task_label
                })
            });
            var data = await resp.json();
            return { success: resp.ok, data: data, error: resp.ok ? null : (data.error || '学习失败') };
        } catch (e) {
            return { success: false, error: e.message, data: null };
        }
    }

    /**
     * 获取系统状态
     */
    async robotGetStatus() {
        try {
            var resp = await this.request('/robot/status', { method: 'GET' });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) {
            return { success: false, error: e.message };
        }
    }

    /**
     * 验证动作执行结果
     */
    async robotVerifyActionResult(before, after, action) {
        try {
            var resp = await this.request('/robot/verify_action', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ before: before, after: after, action: action })
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) {
            return { success: false, error: e.message };
        }
    }


    /**
     * 【FE-003修复】获取训练状态 - 已弃用，请使用 getTrainingStatus()
     * @deprecated 自2026-07起弃用，转发到 getTrainingStatus() 新方法
     */
    async trainingStatus() {
        console.warn('[ApiService] trainingStatus() 已弃用，请迁移到 getTrainingStatus()');
        return await this.getTrainingStatus();
    }

    async trainingPause() {
        try {
            var resp = await this.request('/training/pause', {method: 'POST'});
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /**
     * 【FE-003/FE-004修复】恢复训练任务 - 支持可选checkpointId参数
     * @param {string} [checkpointId] - 可选，检查点ID，用于加载指定检查点后恢复训练
     */
    async trainingResume(checkpointId) {
        try {
            var options = { method: 'POST' };
            /* FE-004修复: 支持传入checkpoint_id以加载检查点 */
            if (checkpointId) {
                options.headers = { 'Content-Type': 'application/json' };
                options.body = JSON.stringify({ checkpoint_id: checkpointId });
            }
            var resp = await this.request('/training/resume', options);
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async trainingStop() {
        try {
            var resp = await this.request('/training/stop', {method: 'POST'});
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async trainingHistory() {
        try {
            var resp = await this.request('/training/history', {method: 'GET'});
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    // ==================== API密钥管理 API ====================

    /* F-014: 设置API密钥用于后续请求认证
     * P1-FIX-04: 安全修复 - 迁移至sessionStorage，会话结束自动清除
     * 若用户明确要求"记住我"功能，应由用户主动选择是否持久化存储
     */
    setApiKey(key) {
        this._apiKey = key;
        if (key) {
            sessionStorage.setItem('selflnn_api_key', key);
        } else {
            sessionStorage.removeItem('selflnn_api_key');
        }
    }

    /* F-014: 获取当前API密钥 */
    getApiKey() {
        return this._apiKey;
    }

    /* F-014: 检查API密钥是否已配置 */
    hasApiKey() {
        return this._apiKey && this._apiKey.length >= 8;
    }

    async apiKeysList() {
        try {
            var resp = await this.request('/key/list', {method: 'GET'});
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async apiKeyCreate(name, value, scope) {
        try {
            var perm = (scope === 'admin') ? 2 : ((scope === 'readonly') ? 0 : 1);
            var resp = await this.request('/key/create', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({name: name, api_key: value, permission: perm, expiration_days: 0})
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async apiKeyToggle(name) {
        try {
            var resp = await this.request('/key/toggle', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({key_prefix: name, enable: 1})
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async apiKeyDelete(name) {
        try {
            var resp = await this.request('/key/delete', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({key_prefix: name})
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    // ==================== 硬件设置 API ====================

    async hardwareScan() {
        try {
            var resp = await this.request('/hardware/scan', {method: 'POST'});
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async hardwareInfo() {
        try {
            var resp = await this.request('/hardware/info', {method: 'GET'});
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /**
     * 统一硬件扫描接口
     * 同时执行硬件扫描和设备信息查询
     * @param {boolean} includeDevices - 是否同时查询设备列表
     * @returns {Promise<object>} { success, scan, devices }
     */
    async scanHardware(includeDevices) {
        try {
            var scanResp = await this.request('/hardware/scan', {method: 'POST'}, 1);
            var scanData = scanResp.ok ? await scanResp.json : {};
            var result = { success: scanResp.ok, scan: scanData };
            if (includeDevices) {
                try {
                    var devResp = await this.request('/hardware/info', {method: 'GET'}, 1);
                    var devData = devResp.ok ? await devResp.json() : {};
                    result.devices = devData.devices || [];
                } catch (e) { result.devices = []; }
            }
            return result;
        } catch (e) { return { success: false, error: e.message }; }
    }

    async hardwareConfig(backend, deviceId) {
        try {
            var resp = await this.request('/hardware/config', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({backend, device_id: deviceId})
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async hardwareResources() {
        try {
            var resp = await this.request('/hardware/resources', {method: 'GET'});
            var data = await resp.json();
            return { success: resp.ok, ...data };
        } catch (e) { return { success: false, error: e.message }; }
    }

/* 编程工作台示例代码API */
    async programmingSample() {
        try {
            var resp = await this.request('/programming/sample', {method: 'GET'});
            var data = await resp.json();
            return { success: resp.ok, code: data.code || '', language: data.language || 'c' };
        } catch (e) { return { success: false, error: e.message }; }
    }

/* 从后端动态获取命令前缀列表 */
    async getCommandPrefixes() {
        try {
            var resp = await this.request('/command/prefixes', {method: 'GET'});
            var data = await resp.json();
            return { success: resp.ok, prefixes: data.prefixes || [] };
        } catch (e) { return { success: false, error: e.message, prefixes: [] }; }
    }

    async hardwareResourcesAllocate() {
        try {
            var resp = await this.request('/hardware/resources/allocate', {method: 'POST'}, 1);
            var data = await resp.json();
            return { success: resp.ok, ...data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async runInference(query) {
        try {
            var resp = await this.request('/knowledge', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({action: 'infer', query: query || ''})
            });
            var data = await resp.json();
            return { success: resp.ok, conclusions: data.conclusions || 0, ...data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async validateKnowledge() {
        try {
            var resp = await this.request('/knowledge', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({action: 'validate'})
            });
            var data = await resp.json();
            return { success: resp.ok, verified: data.verified || 0, total: data.total || 0, ...data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async consistencyCheck() {
        try {
            var resp = await this.request('/knowledge', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({action: 'consistency'})
            });
            var data = await resp.json();
            return { success: resp.ok, score: data.score || 'N/A', ...data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    // ==================== 仿真控制 API ====================

    async simulationStart(engine, scene, dt) {
        try {
            var resp = await this.request('/simulation/start', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({engine, scene, dt})
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async simulationStop() {
        try {
            var resp = await this.request('/simulation/stop', {method: 'POST'});
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async simulationStatus() {
        try {
            var resp = await this.request('/simulation/status', {method: 'GET'});
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async simulationReset() {
        try {
            var resp = await this.request('/simulation/reset', {method: 'POST'});
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async simulationPlanPath(goalX, goalY) {
        try {
            var resp = await this.request('/simulation/plan_path', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({goal_x: goalX, goal_y: goalY})
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* ==================== I-S02: 仿真高级控制API ==================== */

    /**
     * 3D重建（使用后端 simulation/reconstruct3d 端点）
     * POST /api/simulation/reconstruct3d → body: { point_cloud: [...] }
     */
    async simulation3dReconstruct(data) {
        try {
            var resp = await this.request('/simulation/reconstruct3d', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(data || {})
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /**
     * 路径规划
     * POST /api/robot/path/plan → body: { start: [...], goal: [...], constraints: {} }
     */
    async simulationPathPlan(data) {
        try {
            var resp = await this.request('/robot/path/plan', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(data || {})
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /**
     * 重置仿真视图
     * POST /api/simulation/view/reset
     */
    async simulationViewReset() {
        try {
            var resp = await this.request('/simulation/view/reset', { method: 'POST' });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /**
     * 切换网格显示
     * POST /api/simulation/view/toggle_grid
     */
    async simulationGridToggle() {
        try {
            var resp = await this.request('/simulation/view/toggle_grid', { method: 'POST' });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /**
     * 清除场景
     * POST /api/simulation/clear
     */
    async simulationClearScene() {
        try {
            var resp = await this.request('/simulation/clear', { method: 'POST' });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /**
     * 导出仿真场景（通过后端 simulation/command 端点配合导出请求）
     * POST /api/simulation/command → body: { cmd: "export" }
     * 【C-002修复】原使用GET方法，但后端声明为POST，已修正
     */
    async simulationExport() {
        try {
            var resp = await this.request('/simulation/command', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ cmd: 'export' })
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    // ==================== 多模态学习 API ====================

    async multimodalLearn(learnMode, architecture) {
        try {
            var resp = await this.request('/multimodal/learn', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({learn_mode: learnMode, architecture: architecture || 'single-cfc-lnn'})
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async multimodalStatus() {
        try {
            var resp = await this.request('/multimodal/status', {method: 'GET'});
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

/* 多模态教学 — 前端调用的语义对应/multimodal/teach */
    async multimodalTeach(params) {
        try {
            var resp = await this.request('/multimodal/teach', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(params || {})
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

/* 多模态测试 — 运行教学后的验证测试 */
    async multimodalTest() {
        try {
            var resp = await this.request('/multimodal/teach/test', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({action: 'test'})
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    // ==================== 自主学习 API ====================

    async autoLearnToggle(enabled) {
        try {
            var resp = await this.request('/auto-learn/toggle', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({enabled: enabled ? true : false})
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async autoLearnScan() {
        try {
            var resp = await this.request('/auto-learn/scan', {method: 'POST'});
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async autoLearnStats() {
        try {
            var resp = await this.request('/auto-learn/stats', {method: 'GET'});
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async systemCommand(command, params) {
        try {
            var resp = await this.request('/system/command', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ command: command, params: params || {} })
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async sendCommand(target, command, payload) {
        try {
            var body = { target: target, command: command };
            if (payload) { body.payload = payload; }
            var resp = await this.request('/command/send', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(body)
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    // ==================== 语音控制 API ====================

/* 使用统一request替代裸fetch，获得认证/重试/熔断保护 */
    async voiceRecognize(audioBlob, lang) {
        try {
            var formData = new FormData;
            formData.append('audio', audioBlob);
            formData.append('lang', lang);
            var resp = await this.request('/voice/recognize', {method: 'POST', body: formData});
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async voiceSynthesize(text, lang) {
        try {
            var resp = await this.request('/voice/synthesize', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({text, lang})
            });
            var data = await resp.json();
            return { success: resp.ok, data: data, blob: resp.ok ? await resp.blob : null };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async voiceHistory() {
        try {
            var resp = await this.request('/voice/history', {method: 'GET'});
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    // ==================== 设备控制 API ====================

    /* API路径标准化 (M-040): /devices/list */
    async devicesList() {
        try {
            var resp = await this.request('/devices/list', {method: 'GET'});
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async deviceCommand(deviceId, command) {
        try {
            var resp = await this.request('/device/command', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                /* FIX-1: device_id → device_type 匹配后端 parse_json_string("device_type",...) */
                body: JSON.stringify({device_type: deviceId, command})
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async deviceSetMode(mode) {
        try {
            var resp = await this.request('/devices/mode', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({mode})
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async deviceEmergencyStop() {
        try {
            var resp = await this.request('/devices/emergency_stop', {method: 'POST'});
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /**
     * 统一紧急停止接口
     * @param {string} target - 停止目标: 'system' | 'safety' | 'device' | 'robot' | 'all'
     * @param {object} options - 附加选项 { robot_id, reason }
     * @returns {Promise<object>}
     */
    async emergencyStop(target, options) {
        var endpointMap = {
            system: '/safety/emergency_stop',
            safety: '/safety/emergency_stop',
            device: '/devices/emergency_stop',
            robot: '/robot/emergency_stop',
            all: '/safety/emergency_stop'
        };
        var endpoint = endpointMap[target] || '/safety/emergency_stop';
        try {
            var payload = { method: 'POST' };
            if (options) {
                payload.headers = { 'Content-Type': 'application/json' };
                payload.body = JSON.stringify(options);
            }
            var resp = await this.request(endpoint, payload, 1);
            var data = await resp.json();
            return { success: resp.ok, data: data, target: target };
        } catch (e) { return { success: false, error: e.message, target: target }; }
    }

    /* ==================== 技能管理 API ==================== */

    /**
     * 执行指定技能
     * @param {string} skillId - 技能ID
     * @returns {Promise<object>}
     */
    async executeSkill(skillId) {
        try {
            const response = await this.request('/skills/execute', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ skill_id: skillId })
            });
            const data = await response.json();
            return { success: response.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* ==================== 高级训练模式 API ==================== */

    async trainingFromScratch(config) {
        try {
            var resp = await this.request('/training/from-scratch', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(config || {})
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async trainingPretrain(config) {
        try {
            var resp = await this.request('/training/pretrain', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(config || {})
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async trainingFineTune(config) {
        try {
            var resp = await this.request('/training/fine-tune', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(config || {})
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async trainingTransfer(config) {
        try {
            var resp = await this.request('/training/transfer', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(config || {})
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async trainingContinual(config) {
        try {
            var resp = await this.request('/training/continual', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(config || {})
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async trainingExternalApi(config) {
        try {
            var resp = await this.request('/training/external-api', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(config || {})
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* ==================== 模仿学习 API ==================== */

    async imitationDemonstration(demoData) {
        try {
            var resp = await this.request('/imitation/demonstration', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(demoData)
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async imitationTrain(config) {
        try {
            var resp = await this.request('/imitation/train', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(config || {})
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async imitationPredict(input) {
        try {
            var resp = await this.request('/imitation/predict', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ input: input })
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async getImitationStatus() {
        try {
            var resp = await this.request('/imitation/status', {method: 'GET'});
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async setImitationAlgorithm(algorithm) {
        try {
            var resp = await this.request('/imitation/algorithm', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ algorithm: algorithm })
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async learningFromManual(text, category) {
        try {
            var resp = await this.request('/learning/from-manual', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ text: text, category: category || 'general' })
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* ==================== 系统健康与认知 API ==================== */

    async getHealth() {
        try {
            var resp = await this.request('/health', {method: 'GET'});
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* ==================== 拉普拉斯频域分析 API ==================== */
    /* H007: 拉普拉斯分析端点前端调用方 */
    async laplaceSpectrum(params) {
        try {
            var p = params || {};
            var resp = await this.request('/laplace/spectrum', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(p)
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async laplaceAdaptiveLR(params) {
        try {
            var p = params || {};
            var resp = await this.request('/laplace/adaptive-lr', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(p)
            });
            var data = await resp.json();
            return { success: resp.ok, data: data, adaptive_lr: data.adaptive_lr };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* ==================== 密钥管理增强 API ==================== */
/* 删除与第一组重复的getKeyList/keyCreate/keyDelete/keySet */

    /* 密钥更新（唯一功能，保留） */
    async keyUpdate(keyPrefix, updates) {
        try {
            var up = updates || {};
            var body = { key_prefix: keyPrefix };
            if (up.name) body.name = up.name;
            if (up.permission !== undefined) body.permission = up.permission;
            if (up.enabled !== undefined) body.enabled = up.enabled;
            var resp = await this.request('/key/update', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(body)
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async getApiStats() {
        try {
            var resp = await this.request('/stats', {method: 'GET'});
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async getRateLimitStatus() {
        try {
            var resp = await this.request('/rate-limit/status', {method: 'GET'});
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* P2-4: 设置API密钥 — 封装/key/set端点
 *修复: 删除重复的keySet，保留setKey为唯一入口 */
    async setKey(apiKey) {
        try {
            var resp = await this.request('/key/set', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({api_key: apiKey})
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* 获取API密钥状态（被main.js多处调用） */
    async getKeyStatus() {
        try {
            var resp = await this.request('/key/status', {method: 'GET'});
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* ==================== ROS 增强 API ==================== */

    async rosPublish(topic, type, data) {
        try {
            var resp = await this.request('/ros/publish', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ topic: topic, type: type, data: data })
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async rosSubscribe(topic, type) {
        try {
            var resp = await this.request('/ros/subscribe', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ topic: topic, type: type })
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async rosCallService(service, args) {
        try {
            var resp = await this.request('/ros/service', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ service: service, args: args || {} })
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* ==================== 串口通信 API ==================== */

    async getSerialList() {
        try {
            var resp = await this.request('/serial/list', {method: 'GET'});
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async serialOpen(port, baud) {
        try {
            var resp = await this.request('/serial/open', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ port: port, baud: baud || 115200 })
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async serialClose(port) {
        try {
            var resp = await this.request('/serial/close', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ port: port })
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async serialSend(port, data) {
        try {
            var resp = await this.request('/serial/send', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ port: port, data: data })
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* ==================== 多机器人同步 API ==================== */

    async multiRobotSync(syncConfig) {
        try {
            var resp = await this.request('/multi_robot/sync', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(syncConfig || {})
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /**
     * I-M06: 多机器人同步（别名方法）
     * POST /api/multi_robot/sync → body: { targets: [...], config: {} }
     */
    async syncMultiRobot() {
        try {
            var resp = await this.request('/multi_robot/sync', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({})
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* ==================== AGI 功能列表 API ==================== */

    async getAgiFeatureList() {
        try {
            var resp = await this.request('/agi/feature_list', {method: 'GET'});
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* ==================== 从对话学习 API ==================== */

    async learningFromDialogue(messages) {
        try {
            var resp = await this.request('/learning/from-dialogue', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ messages: messages || [] })
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* ==================== API 文档 API ==================== */

    async getApiDocs() {
        try {
            var resp = await this.request('/docs', {method: 'GET'});
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async getApiKeyDocs() {
        try {
            var resp = await this.request('/key/docs', {method: 'GET'});
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* ==================== 计算机控制 API ==================== */

    async computerLaunch(appPath) {
        try {
            var resp = await this.request('/computer/launch', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ app_path: appPath })
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async computerClose(appName) {
        try {
            var resp = await this.request('/computer/close', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ app_name: appName })
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async computerType(text) {
        try {
            var resp = await this.request('/computer/type', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ text: text })
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async computerScreenshot() {
        try {
            var resp = await this.request('/computer/screenshot', {method: 'POST'});
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async computerExecute(command) {
        try {
            var resp = await this.request('/computer/execute', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ command: command })
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async computerVolume(level) {
        try {
            var resp = await this.request('/computer/volume', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ level: level })
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* ==================== 文件系统操作 API ==================== */

    async fileRead(path) {
        try {
            var resp = await this.request('/files/read', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ path: path })
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async fileWrite(path, content) {
        try {
            var resp = await this.request('/files/write', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ path: path, content: content })
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async fileDelete(path) {
        try {
            var resp = await this.request('/files/delete', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ path: path })
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* 【H-001修复】GET请求不应携带body（违反HTTP规范），改为查询参数 */
    async fileList(dir) {
        try {
            var queryDir = encodeURIComponent(dir || '.');
            var resp = await this.request('/files/list?dir=' + queryDir, { method: 'GET' });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* ==================== 演化帕累托前沿 API ==================== */

    async getEvolutionPareto() {
        try {
            var resp = await this.request('/evolution/pareto', {method: 'POST'}); /* P1-001修复: 后端声明为POST，非GET */
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* 【C-001修复】移除getParetoFront()重复方法，统一使用getEvolutionPareto()。
     * 原getParetoFront用GET调用/evolution/pareto，但后端声明为POST，会导致405错误。
     * getEvolutionPareto()已正确使用POST方法，无需重复。 */

    /* ==================== 机器人坐标控制 API ==================== */

    async robotCoordinateControl(coords) {
        try {
            var resp = await this.request('/robot/coordinate', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(coords || {})
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* ==================== 设备开关控制 API ==================== */

    async deviceControl(deviceId, action, params) {
        try {
            var resp = await this.request('/device/control', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                /* FIX-1: device_id→device 匹配后端 parse_json_string("device",...) */
                body: JSON.stringify({ device: deviceId, action: action, params: params || {} })
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* ==================== 知识库 API ==================== */

    async addKnowledgeEntry(params) {
        try {
            var resp = await this.request('/knowledge/add', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(params || {})
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async getKnowledgeEntry(id) {
        try {
            var resp = await this.request('/knowledge/entry/' + id, { method: 'GET' });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async deleteKnowledgeEntry(id) {
        try {
            var resp = await this.request('/knowledge/entry/' + id, { method: 'DELETE' });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async getKnowledgeStats() {
        try {
            var resp = await this.request('/knowledge/stats', { method: 'GET' });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* 【C-003修复】原使用GET方法，但后端声明为POST，已修正 */
    async exportKnowledge() {
        try {
            var resp = await this.request('/knowledge/export', { method: 'POST' });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* ==================== 记忆管理 API ==================== */

    async addMemory(params) {
        try {
            var resp = await this.request('/memory/add', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(params || {})
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async getMemoryEntry(id) {
        try {
            var resp = await this.request('/memory/entry/' + id, { method: 'GET' });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async updateMemoryEntry(id, params) {
        try {
            var resp = await this.request('/memory/entry/' + id, {
                method: 'PUT', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(params || {})
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async deleteMemoryEntry(id) {
        try {
            var resp = await this.request('/memory/entry/' + id, { method: 'DELETE' });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async exportMemory() {
        try {
            var resp = await this.request('/memory/export', { method: 'GET' });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async clearOldMemories(params) {
        try {
            var resp = await this.request('/memory/clear', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(params || {})
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async searchMemories(params) {
        try {
            var resp = await this.request('/memory/search', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(params || {})
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async sleepConsolidation(params) {
        try {
            var resp = await this.request('/memory/sleep_consolidation', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(params || {})
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* ==================== 任务管理 API ==================== */

    async getTaskQueue() {
        try {
            var resp = await this.request('/task/queue', { method: 'GET' });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async assignTask(params) {
        try {
            var resp = await this.request('/task/assign', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(params || {})
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* ==================== 推理系统 API ==================== */

    async startReasoning(params) {
        try {
            var resp = await this.request('/reasoning/start', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(params || {})
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async stopAllReasoning() {
        try {
            var resp = await this.request('/reasoning/stop_all', { method: 'POST' });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async pauseReasoning() {
        try {
            var resp = await this.request('/reasoning/pause', { method: 'POST' });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async saveReasoningConfig(config) {
        try {
            var resp = await this.request('/reasoning/config/save', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(config || {})
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async getDecisionLog(params) {
        try {
            /* P-AUDIT修复: 后端 /decision/log 为 GET 方法,原用 POST 导致 405 */
            var qs = params ? '?' + new URLSearchParams(params).toString() : '';
            var resp = await this.request('/decision/log' + qs, {
                method: 'GET'
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* ==================== LNN 参数控制 API ==================== */

    async resetLNNParameters() {
        try {
            var resp = await this.request('/lnn/parameters/reset', { method: 'POST' });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async calibrateLNN() {
        try {
            var resp = await this.request('/lnn/calibrate', { method: 'POST' });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async exportLNNConfig() {
        try {
            var resp = await this.request('/lnn/config/export', { method: 'GET' });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* ==================== 模型管理 API ==================== */

    async loadModel(modelId) {
        try {
            var resp = await this.request('/model/load', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ model_id: modelId })
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async saveModelConfig(config) {
        try {
            var resp = await this.request('/model/config/save', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(config || {})
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async startModel(modelId) {
        try {
            var resp = await this.request('/model/start', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ model_id: modelId })
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async stopModel(modelId) {
        try {
            var resp = await this.request('/model/stop', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ model_id: modelId })
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async unloadModel(modelId) {
        try {
            var resp = await this.request('/model/unload', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ model_id: modelId })
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* ==================== 超参数搜索 API ==================== */

    async startHyperparameterSearch(config) {
        try {
            var resp = await this.request('/hyperparameter/start', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(config || {})
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /**
     * I-L05: 超参数搜索（别名方法，使用后端 hyperparameter/start 端点）
     * POST /api/hyperparameter/start → body: { search_space: {}, trials: N }
     */
    async hyperparameterSearch() {
        try {
            var resp = await this.request('/hyperparameter/start', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({})
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async getHyperparameterStatus() {
        try {
            var resp = await this.request('/hyperparameter/status', { method: 'GET' });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* ==================== 系统管理 API ==================== */

    async getStatus() {
        try {
            var resp = await this.request('/system/status', { method: 'GET' });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async updateConfig(key, value) {
        try {
            var resp = await this.request('/system/config/update', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ key: key, value: value })
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async saveSettings(settings) {
        try {
            var resp = await this.request('/system/settings', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(settings || {})
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async changePassword(passwords) {
        try {
            var resp = await this.request('/system/change_password', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(passwords || {})
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async restart() {
        try {
            var resp = await this.request('/system/restart', { method: 'POST' });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async shutdown() {
        try {
            var resp = await this.request('/system/shutdown', { method: 'POST' });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async exportLogs() {
        try {
            var resp = await this.request('/system/logs/export', { method: 'GET' });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* ==================== 集群/舰队管理 API ==================== */

    async getFleetStatus() {
        try {
            var resp = await this.request('/fleet/status', { method: 'GET' });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* ==================== 自我编程 API ==================== */

    async programmingAnalyze(code) {
        try {
            var resp = await this.request('/programming/analyze', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({code: code || ''})
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async programmingGenerate(functionName, description, paramCount) {
        try {
            var resp = await this.request('/programming/generate', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({function_name: functionName, description: description, param_count: paramCount || 0})
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async programmingOptimize(code, iterations) {
        try {
            var resp = await this.request('/programming/optimize', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({code: code || '', iterations: iterations || 1})
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async programmingCompile(code) {
        try {
            var resp = await this.request('/programming/compile', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({code: code || ''})
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async programmingExecute(code, input) {
        try {
            var resp = await this.request('/programming/execute', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({code: code || '', input: input || ''})
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    async programmingStatus() {
        try {
            var resp = await this.request('/programming/status', { method: 'GET' });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    // ================================================================
    // AGI认知系统API
    // ================================================================

    /**
     * 获取认知系统状态（自我认知、元认知、深度反思）
     * GET /api/agi/cognition/state
     */
    async getCognitionState() {
        try {
            const response = await this.request('/agi/cognition/state');
            if (!response.ok) {
                throw new Error(`HTTP错误: ${response.status}`);
            }
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error('获取认知状态失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * AGI思考推理（带目标导向的深度推理）
     * @param {string} prompt - 推理提示词
     */
    async agiThink(prompt) {
        try {
            const response = await this.request('/agi/think', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt: prompt })
            });
            if (!response.ok) throw new Error(`HTTP错误: ${response.status}`);
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error('AGI思考失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * 获取多模态处理状态
     */
    async getMultimodalStatus() {
        try {
            const response = await this.request('/multimodal/status');
            if (!response.ok) throw new Error(`HTTP错误: ${response.status}`);
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error('获取多模态状态失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * 保存多模态配置
     */
    async saveMultimodalConfig(config) {
        try {
            const response = await this.request('/multimodal/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config || {})
            });
            if (!response.ok) throw new Error(`HTTP错误: ${response.status}`);
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error('保存多模态配置失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * 测试多模态处理
     */
    async testMultimodalProcessing(testData) {
        try {
            const response = await this.request('/multimodal/test', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(testData || {})
            });
            if (!response.ok) throw new Error(`HTTP错误: ${response.status}`);
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error('测试多模态处理失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * 重置多模态配置
     */
    async resetMultimodalConfig() {
        try {
            const response = await this.request('/multimodal/config/reset', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action: 'reset' })
            });
            if (!response.ok) throw new Error(`HTTP错误: ${response.status}`);
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error('重置多模态配置失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * 停止多模态处理
     */
    async stopMultimodalProcessing() {
        try {
            const response = await this.request('/multimodal/stop', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action: 'stop' })
            });
            if (!response.ok) throw new Error(`HTTP错误: ${response.status}`);
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error('停止多模态处理失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * I-L04: 停止多模态处理（别名方法）
     * POST /api/multimodal/stop → 停止多模态处理流程
     */
    async stopMultimodal() {
        try {
            var resp = await this.request('/multimodal/stop', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action: 'stop' })
            });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /**
     * 重置机器人配置
     */
    async resetRobotConfig() {
        try {
            const response = await this.request('/robot/config/reset', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action: 'reset' })
            });
            if (!response.ok) throw new Error(`HTTP错误: ${response.status}`);
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error('重置机器人配置失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * 暂停AGI任务
     * @param {string|number} taskId - 任务ID
     */
    async pauseTask(taskId) {
        try {
            const response = await this.request('/task/pause', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ task_id: taskId })
            });
            const data = await response.json();
            return { success: response.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /**
     * 取消AGI任务
     * @param {string|number} taskId - 任务ID
     */
    async cancelTask(taskId) {
        try {
            const response = await this.request('/task/cancel', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ task_id: taskId })
            });
            const data = await response.json();
            return { success: response.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* ================================================================
 *修复: 补充10个前端缺失的API方法 + 恢复getCognitionHealth
     * 以下方法在main.js中被安全校验调用（typeof === 'function'），
     * 但因api-service.js中缺失定义而静默失败。
     * ================================================================ */

    /**
     * 认知系统健康检查
     */
    async getCognitionHealth() {
        try {
            var resp = await this.request('/cognition/health', {method: 'GET'});
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /**
     * 知识搜索
     * @param {string} query - 搜索查询
     */
    async searchKnowledge(query) {
        try {
            const response = await this.request('/knowledge/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: query || '' })
            });
            if (!response.ok) throw new Error(`HTTP错误: ${response.status}`);
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error('知识搜索失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

/* getKnowledgeBase — 前端fallback分支调用的知识库查询(语义等同/knowledge) */
    async getKnowledgeBase(params) {
        try {
            var queryStr = '';
            if (params && params.search) queryStr = '?search=' + encodeURIComponent(params.search);
            if (params && params.type) queryStr += (queryStr ? '&' : '?') + 'type=' + encodeURIComponent(params.type);
            var resp = await this.request('/knowledge' + queryStr, { method: 'GET' });
            var data = await resp.json();
            return { success: resp.ok, data: data };
        } catch (e) {
            console.error('getKnowledgeBase失败:', e);
            return { success: false, error: e.message };
        }
    }

    /**
     * 推理测试
     * @param {object} testParams - 测试参数
     */
    async testInference(testParams) {
        try {
            /* P-AUDIT修复: 后端 /reasoning/test 为 GET 方法,原用 POST 导致 405 */
            const qs = testParams ? '?' + new URLSearchParams(testParams).toString() : '';
            const response = await this.request('/reasoning/test' + qs, {
                method: 'GET'
            });
            if (!response.ok) throw new Error(`HTTP错误: ${response.status}`);
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error('推理测试失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * 知识保存
     * @param {object} entry - 知识条目
     */
    async knowledgeSave(entry) {
        try {
            const response = await this.request('/knowledge/save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(entry || {})
            });
            if (!response.ok) throw new Error(`HTTP错误: ${response.status}`);
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error('知识保存失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * 知识加载
     * @param {string} knowledgeId - 知识ID
     */
    async knowledgeLoad(knowledgeId) {
        try {
            /* P-AUDIT修复: 后端 /knowledge/load 为 GET 方法,原用 POST 导致 405 */
            const response = await this.request('/knowledge/load?id=' + encodeURIComponent(knowledgeId || ''), {
                method: 'GET'
            });
            if (!response.ok) throw new Error(`HTTP错误: ${response.status}`);
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error('知识加载失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * 知识导出JSON
     */
    async knowledgeExportJSON() {
        try {
            const response = await this.request('/knowledge/export', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ format: 'json' })
            });
            if (!response.ok) throw new Error(`HTTP错误: ${response.status}`);
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error('知识导出失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * 知识导入JSON
     * @param {object} importData - 导入数据
     */
    async knowledgeImportJSON(importData) {
        try {
            const response = await this.request('/knowledge/import', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(importData || {})
            });
            if (!response.ok) throw new Error(`HTTP错误: ${response.status}`);
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error('知识导入失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * 保存机器人配置
     * @param {object} config - 机器人配置
     */
    async saveRobotConfig(config) {
        try {
            const response = await this.request('/robot/config/save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config || {})
            });
            if (!response.ok) throw new Error(`HTTP错误: ${response.status}`);
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error('保存机器人配置失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * 创建AGI任务
     * @param {object} taskData - 任务数据
     */
    async createTask(taskData) {
        try {
            const response = await this.request('/task/create', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(taskData || {})
            });
            if (!response.ok) throw new Error(`HTTP错误: ${response.status}`);
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error('创建任务失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * 切换摄像头源
     * @param {object} source - 摄像头源配置 {camera: ...}
     */
    async switchCameraSource(source) {
        try {
            const response = await this.request('/camera/switch', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(source || {})
            });
            if (!response.ok) throw new Error(`HTTP错误: ${response.status}`);
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error('切换摄像头失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * 设置视频质量
     * @param {object} quality - 质量参数 (width, height, fps等)
     */
    async setVideoQuality(quality) {
        try {
            const response = await this.request('/video/quality', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(quality || {})
            });
            if (!response.ok) throw new Error(`HTTP错误: ${response.status}`);
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error('设置视频质量失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * 获取训练状态（/5修复: 添加缺失的API方法）
     * API路径标准化 (M-039): GET /training/status
     */
    async getTrainingStatus() {
        try {
            const response = await this.request('/training/status');
            if (!response.ok) throw new Error(`HTTP错误: ${response.status}`);
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error('获取训练状态失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * 恢复/继续AGI任务
     * API路径标准化: POST /task/resume
     * @param {string|number} taskId - 任务ID
     */
    async resumeTask(taskId) {
        try {
            const response = await this.request('/task/resume', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ task_id: taskId })
            });
            if (!response.ok) throw new Error(`HTTP错误: ${response.status}`);
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error('恢复任务失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * 导入知识库数据（添加缺失的API方法）
     * API路径标准化 (M-039): POST /knowledge/import
     * @param {string} fileContent - 文件内容
     * @param {string} fileName - 文件名
     */
    async importKnowledge(fileContent, fileName) {
        try {
            const response = await this.request('/knowledge/import', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ content: fileContent, file_name: fileName || 'import.json' })
            });
            if (!response.ok) throw new Error(`HTTP错误: ${response.status}`);
            const data = await response.json();
            return { success: true, data: data, importedCount: data.imported_count || data.count || 0 };
        } catch (error) {
            console.error('知识导入失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /* ==================== 补充缺失的API端点封装 ==================== */

    /* --- 知识库高级操作 --- */
    async knowledgeStats() {
        try { var r = await this.request('/knowledge/stats'); var d = await r.json(); return { success: true, data: d }; }
        catch(e) { return { success: false, error: e.message }; }
    }
    /* P-AUDIT修复: knowledgeExport 原用 GET (无方法指定),后端 /knowledge/export 为 POST,导致 405 */
    async knowledgeExport() {
        try { var r = await this.request('/knowledge/export', { method: 'POST' }); var d = await r.json(); return { success: true, data: d }; }
        catch(e) { return { success: false, error: e.message }; }
    }
    async knowledgeDelete(entryId) {
        try { var r = await this.request('/knowledge/delete', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({id:entryId}) }); var d = await r.json(); return { success: true, data: d }; }
        catch(e) { return { success: false, error: e.message }; }
    }
    async knowledgeEntry(entryId) {
        try { var r = await this.request('/knowledge/entry/' + entryId); var d = await r.json(); return { success: true, data: d }; }
        catch(e) { return { success: false, error: e.message }; }
    }

    /* --- 记忆系统高级操作 --- */
    async memoryAdd(key, data, strength) {
        try {
            var r = await this.request('/memory/add', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({key:key, data:data, strength:strength||0.8}) });
            var d = await r.json(); return { success: true, data: d };
        } catch(e) { return { success: false, error: e.message }; }
    }
    async memoryEntry(entryId) {
        try { var r = await this.request('/memory/entry/' + entryId); var d = await r.json(); return { success: true, data: d }; }
        catch(e) { return { success: false, error: e.message }; }
    }
    async memoryExport() {
        try { var r = await this.request('/memory/export'); var d = await r.json(); return { success: true, data: d }; }
        catch(e) { return { success: false, error: e.message }; }
    }
    async memoryClear() {
        try { var r = await this.request('/memory/clear', { method:'POST' }); var d = await r.json(); return { success: true, data: d }; }
        catch(e) { return { success: false, error: e.message }; }
    }
    async memorySearch(query) {
        try { var r = await this.request('/memory/search', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({query:query}) }); var d = await r.json(); return { success: true, data: d }; }
        catch(e) { return { success: false, error: e.message }; }
    }
    async memorySleepConsolidation() {
        try { var r = await this.request('/memory/sleep_consolidation', { method:'POST' }); var d = await r.json(); return { success: true, data: d }; }
        catch(e) { return { success: false, error: e.message }; }
    }

    /* --- 技能库 --- */
    async skillsList() {
        try { var r = await this.request('/skills'); var d = await r.json(); return { success: true, data: d }; }
        catch(e) { return { success: false, error: e.message }; }
    }
    async skillsSearch(query) {
        try { var r = await this.request('/skills/search', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({query:query}) }); var d = await r.json(); return { success: true, data: d }; }
        catch(e) { return { success: false, error: e.message }; }
    }
    async skillsCompose(skillIds) {
        try { var r = await this.request('/skills/compose', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({skills:skillIds}) }); var d = await r.json(); return { success: true, data: d }; }
        catch(e) { return { success: false, error: e.message }; }
    }
    async skillsStats() {
        try { var r = await this.request('/skills/stats'); var d = await r.json(); return { success: true, data: d }; }
        catch(e) { return { success: false, error: e.message }; }
    }

    /* --- 多智能体 --- */
    async multiAgentStatus() {
        try { var r = await this.request('/multi-agent/status'); var d = await r.json(); return { success: true, data: d }; }
        catch(e) { return { success: false, error: e.message }; }
    }
    async multiAgentNegotiate(params) {
        try { var r = await this.request('/multi-agent/negotiate', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(params||{}) }); var d = await r.json(); return { success: true, data: d }; }
        catch(e) { return { success: false, error: e.message }; }
    }
    async multiAgentConsensus(params) {
        try { var r = await this.request('/multi-agent/consensus', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(params||{}) }); var d = await r.json(); return { success: true, data: d }; }
        catch(e) { return { success: false, error: e.message }; }
    }
    async multiAgentMessage(msg) {
        try { var r = await this.request('/multi-agent/message', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(msg||{}) }); var d = await r.json(); return { success: true, data: d }; }
        catch(e) { return { success: false, error: e.message }; }
    }
    async multiAgentTask(task) {
        try { var r = await this.request('/multi-agent/task', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(task||{}) }); var d = await r.json(); return { success: true, data: d }; } catch(e) { return { success: false, error: e.message }; }
    }

    /* ==================== I-S03: 多智能体协商API ==================== */

    /**
     * 智能体协商（使用后端 multi-agent/negotiate 端点）
     * POST /api/multi-agent/negotiate → body: { agents: [...], proposal: ... }
     */
    async negotiateAgents(data) {
        try {
            var r = await this.request('/multi-agent/negotiate', {
                method:'POST', headers:{'Content-Type':'application/json'},
                body:JSON.stringify(data || {})
            });
            var d = await r.json();
            return { success: true, data: d };
        }
        catch(e) { return { success: false, error: e.message }; }
    }

    /**
     * 智能体达成共识（使用后端 multi-agent/consensus 端点）
     * POST /api/multi-agent/consensus → body: { votes: [...], proposals: ... }
     */
    async consensusAgents(data) {
        try {
            var r = await this.request('/multi-agent/consensus', {
                method:'POST', headers:{'Content-Type':'application/json'},
                body:JSON.stringify(data || {})
            });
            var d = await r.json();
            return { success: true, data: d };
        }
        catch(e) { return { success: false, error: e.message }; }
    }

    /* --- 系统管理 --- */
    async systemLogs(lines) {
        try { var r = await this.request('/system/logs?lines=' + (lines||100)); var d = await r.json(); return { success: true, data: d }; }
        catch(e) { return { success: false, error: e.message }; }
    }
    async systemConfigUpdate(config) {
        try { var r = await this.request('/system/config/update', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(config||{}) }); var d = await r.json(); return { success: true, data: d }; }
        catch(e) { return { success: false, error: e.message }; }
    }
    async systemSettings(settings) {
        try { var r = await this.request('/system/settings', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(settings||{}) }); var d = await r.json(); return { success: true, data: d }; }
        catch(e) { return { success: false, error: e.message }; }
    }
    async systemRestart() {
        try { var r = await this.request('/system/restart', { method:'POST' }); var d = await r.json(); return { success: true, data: d }; }
        catch(e) { return { success: false, error: e.message }; }
    }
    async systemDiagnostic() {
        try { var r = await this.request('/system/diagnostic'); var d = await r.json(); return { success: true, data: d }; }
        catch(e) { return { success: false, error: e.message }; }
    }

    /* --- 数据集管理 --- */
    async datasetList() {
        try { var r = await this.request('/dataset/list'); var d = await r.json(); return { success: true, data: d }; }
        catch(e) { return { success: false, error: e.message }; }
    }
    async datasetCreate(name, config) {
        try { var r = await this.request('/dataset/create', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({name:name, config:config||{}}) }); var d = await r.json(); return { success: true, data: d }; }
        catch(e) { return { success: false, error: e.message }; }
    }
    async datasetStats() {
        try { var r = await this.request('/dataset/stats'); var d = await r.json(); return { success: true, data: d }; }
        catch(e) { return { success: false, error: e.message }; }
    }

    /* --- 推理引擎高级操作 --- */
    async reasoningStart(params) {
        try { var r = await this.request('/reasoning/start', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(params||{}) }); var d = await r.json(); return { success: true, data: d }; }
        catch(e) { return { success: false, error: e.message }; }
    }
    async reasoningStopAll() {
        try { var r = await this.request('/reasoning/stop_all', { method:'POST' }); var d = await r.json(); return { success: true, data: d }; }
        catch(e) { return { success: false, error: e.message }; }
    }

    /* --- 能力诊断 --- */
    async capabilityDiagnose() {
        /* 集成修复: 后端声明为POST方法(backend.c:593)，前端需指定method */
        try { var r = await this.request('/capability/diagnose', { method: 'POST', headers: {'Content-Type': 'application/json'} }); var d = await r.json(); return { success: true, data: d }; }
        catch(e) { return { success: false, error: e.message }; }
    }

    /* ==================== P2-002: 补充的API封装方法 ==================== */

    async emergencyStopRobots() {
        try { var r = await this.request('/robot/emergency_stop', { method:'POST' }); var d = await r.json(); return { success: true, data: d }; }
        catch(e) { return { success: false, error: e.message }; }
    }
    async serialList() {
        try { var r = await this.request('/serial/list', { method:'GET' }); var d = await r.json(); return { success: true, data: d }; }
        catch(e) { return { success: false, error: e.message }; }
    }
    async computerLaunch(appName) {
        try { var r = await this.request('/computer/launch', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({name:appName}) }); var d = await r.json(); return { success: true, data: d }; }
        catch(e) { return { success: false, error: e.message }; }
    }
    async computerScreenshot() {
        try { var r = await this.request('/computer/screenshot', { method:'POST' }); var d = await r.json(); return { success: true, data: d }; }
        catch(e) { return { success: false, error: e.message }; }
    }
    async computerType(text) {
        try { var r = await this.request('/computer/type', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({text:text}) }); var d = await r.json(); return { success: true, data: d }; }
        catch(e) { return { success: false, error: e.message }; }
    }
    async simulationPlanPath(start, goal) {
        try { var r = await this.request('/simulation/plan_path', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({start:start, goal:goal}) }); var d = await r.json(); return { success: true, data: d }; }
        catch(e) { return { success: false, error: e.message }; }
    }
    /* P1-2修复: teachGetConcepts为重复定义,已删除(保留先定义于第3396行) */
    async teachTestConcept(concept) {
        try { var r = await this.request('/teach/test_concept', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({concept:concept}) }); var d = await r.json(); return { success: true, data: d }; }
        catch(e) { return { success: false, error: e.message }; }
    }
/* 删除3个重复/错误API KEY函数
     * apiKeyCreate 缺少api_key字段 | apiKeyList第三次重复 | apiKeyDelete发送key_id而非key_prefix */
    async devicesEmergencyStop() {
        try { var r = await this.request('/devices/emergency_stop', { method:'POST' }); var d = await r.json(); return { success: true, data: d }; }
        catch(e) { return { success: false, error: e.message }; }
    }
    async getHardwareInfo() {
        try { var r = await this.request('/hardware/info', { method:'GET' }); var d = await r.json(); return { success: true, data: d }; }
        catch(e) { return { success: false, error: e.message }; }
    }

/* 补充前端缺失的关键API调用 */
    /**
     * 获取元认知系统状态
     * 对应后端 /metacognition/state (GET)
     * API路径标准化: /metacognition/state
     */
    async getMetacognitionState() {
        try {
            const response = await this.request('/metacognition/state', { method: 'GET' });
            if (!response.ok) throw new Error('HTTP错误: ' + response.status);
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error(' 获取元认知状态失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * 启动训练管线
     * 对应后端 /training/pipeline (POST)
     * API路径标准化: /training/pipeline
     */
    async postTrainingPipeline(config) {
        try {
            const response = await this.request('/training/pipeline', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config || {})
            });
            if (!response.ok) throw new Error('HTTP错误: ' + response.status);
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error(' 训练管线启动失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * 获取检查点列表
     * 对应后端 /checkpoint/list (GET)
     * API路径标准化: /checkpoint/list
     */
    async getCheckpointList() {
        try {
            const response = await this.request('/checkpoint/list', { method: 'GET' });
            if (!response.ok) throw new Error('HTTP错误: ' + response.status);
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error(' 获取检查点列表失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /**
     * 加载检查点
     * 对应后端 /checkpoint/load (POST)
     * API路径标准化: /checkpoint/load
     */
    async postCheckpointLoad(checkpointId) {
        try {
            const response = await this.request('/checkpoint/load', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ checkpoint_id: checkpointId || '' })
            });
            if (!response.ok) throw new Error('HTTP错误: ' + response.status);
            const data = await response.json();
            return { success: true, data: data };
        } catch (error) {
            console.error(' 加载检查点失败:', error);
            return { success: false, error: error.message, data: null };
        }
    }

    /* ==================== I-03~I-18 集成缺陷修复: 补充缺失的前端API方法 ==================== */

    /* --- I-03: 记忆系统补充方法 --- */
    /* clearMemory: 清空所有记忆 */
    async clearMemory() {
        try {
            var r = await this.request('/memory/clear', { method: 'POST' });
            var d = await r.json();
            return { success: true, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* searchMemory: 按查询字符串搜索记忆 (POST方式, 后端期望JSON body) */
    async searchMemory(query) {
        try {
            var r = await this.request('/memory/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: query || '' })
            });
            var d = await r.json();
            return { success: true, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* --- I-04: 推理引擎补充方法 --- */
    /* P-AUDIT修复: testReasoning 后端 /reasoning/test 为 GET,原用 POST 导致 405 */
    async testReasoning(data) {
        try {
            var qs = data ? '?' + new URLSearchParams(data).toString() : '';
            var r = await this.request('/reasoning/test' + qs, {
                method: 'GET'
            });
            var d = await r.json();
            return { success: true, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* --- I-05: 学习状态补充方法 --- */
    /* learnFromDialogue: 从对话中学习 */
    async learnFromDialogue(data) {
        try {
            var r = await this.request('/learning/from-dialogue', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data || {})
            });
            var d = await r.json();
            return { success: true, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* learnFromManual: 从说明书/教学文本学习 */
    async learnFromManual(data) {
        try {
            var r = await this.request('/learning/from-manual', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data || {})
            });
            var d = await r.json();
            return { success: true, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* checkConsistency: 知识一致性检查 */
    async checkConsistency() {
        try {
            var r = await this.request('/learning/consistency', { method: 'POST' });
            var d = await r.json();
            return { success: true, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* --- I-06: 编程工作台 - 所有方法已存在(programmingAnalyze/Generate/Execute/Optimize/Compile/Status) --- */

    /* --- I-07: 产品设计补充方法 --- */
    /* P-AUDIT修复: getProductSpec 后端 /product/spec 为 POST,原用 GET(通过query param)导致 405 */
    async getProductSpec(id) {
        try {
            var r = await this.request('/product/spec', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ id: id || '' })
            });
            var d = await r.json();
            return { success: true, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* getProductStatus: 获取产品设计引擎状态 */
    async getProductStatus() {
        try {
            var r = await this.request('/product/status');
            var d = await r.json();
            return { success: true, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* designProduct: 启动产品设计 */
    async designProduct(data) {
        try {
            var r = await this.request('/product/design', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data || {})
            });
            var d = await r.json();
            return { success: true, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* --- I-08: 技能库补充方法 --- */
    /* getSkills: 获取所有技能 */
    async getSkills() {
        try {
            var r = await this.request('/skills');
            var d = await r.json();
            return { success: true, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* searchSkills: 按查询搜索技能 */
    /* P1-1修复: 后端期望POST方法,原GET导致方法不匹配 */
    async searchSkills(query) {
        try {
            var r = await this.request('/skills/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: query || '' })
            });
            var d = await r.json();
            return { success: true, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* composeSkills: 组合技能 */
    async composeSkills(data) {
        try {
            var r = await this.request('/skills/compose', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data || {})
            });
            var d = await r.json();
            return { success: true, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* getSkillStats: 获取技能统计 */
    async getSkillStats() {
        try {
            var r = await this.request('/skills/stats');
            var d = await r.json();
            return { success: true, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* --- I-09: LNN控制面板补充方法 --- */
    /* P-AUDIT修复: getLnnStatus 原调用 /lnn (断链),改为后端实际路由 /lnn/status */
    async getLnnStatus() {
        try {
            var r = await this.request('/lnn/status');
            var d = await r.json();
            return { success: true, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* P-AUDIT修复: updateLnnConfig 原调用 PUT /lnn (断链),改为 POST /lnn/config */
    async updateLnnConfig(data) {
        try {
            var r = await this.request('/lnn/config', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data || {})
            });
            var d = await r.json();
            return { success: true, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* calibrateLnn: 校准LNN */
    async calibrateLnn() {
        try {
            var r = await this.request('/lnn/calibrate', { method: 'POST' });
            var d = await r.json();
            return { success: true, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* --- I-10: 数据集管理补充方法 --- */
    /* getDatasets: 获取所有数据集 */
    async getDatasets() {
        try {
            var r = await this.request('/datasets');
            var d = await r.json();
            return { success: true, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* createDataset: 创建数据集 */
    async createDataset(data) {
        try {
            var r = await this.request('/datasets', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data || {})
            });
            var d = await r.json();
            return { success: true, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* P-AUDIT修复: deleteDataset 原调用 /datasets/{id} (断链),改为后端实际路由 /dataset/delete */
    async deleteDataset(id) {
        try {
            var r = await this.request('/dataset/delete', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ id: id })
            });
            var d = await r.json();
            return { success: true, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* --- I-11: 多机器人补充方法 --- */
    /* P-AUDIT修复: getRobots 原调用 /robots (断链),改为后端实际路由 /robot/list */
    async getRobots() {
        try {
            var r = await this.request('/robot/list');
            var d = await r.json();
            return { success: true, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* P-AUDIT修复: createRobot 原调用 POST /robots (断链),改为后端实际路由 /robot/register */
    async createRobot(data) {
        try {
            var r = await this.request('/robot/register', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data || {})
            });
            var d = await r.json();
            return { success: true, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* P-AUDIT修复: controlRobotGroup 原调用 /robots/group (断链),改为后端实际路由 /multi_robot/sync */
    async controlRobotGroup(data) {
        try {
            var r = await this.request('/multi_robot/sync', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data || {})
            });
            var d = await r.json();
            return { success: true, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* --- I-12: 知识库 - 所有方法已存在(getKnowledgeStats/exportKnowledge/importKnowledge) --- */

    /* --- I-13: 仿真控制补充方法 --- */
    /* P-AUDIT修复: getSimulationView 原调用 /simulation/view (断链),改为 /simulation/status */
    async getSimulationView() {
        try {
            var r = await this.request('/simulation/status');
            var d = await r.json();
            return { success: true, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* --- I-14: 设备管理补充方法 --- */
    /* P-AUDIT修复: getDevices 原调用 /devices (断链),改为后端实际路由 /devices/list */
    async getDevices() {
        try {
            var r = await this.request('/devices/list', { method: 'POST' });
            var d = await r.json();
            return { success: true, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* P-AUDIT修复: discoverDevices 原用 POST (方法不匹配),后端为 GET /devices/discover */
    async discoverDevices() {
        try {
            var r = await this.request('/devices/discover');
            var d = await r.json();
            return { success: true, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* --- I-15: API密钥补充方法 --- */
    /* P-AUDIT修复: getApiKeys 原调用 /keys (断链),改为后端实际路由 /key/list */
    async getApiKeys() {
        try {
            var r = await this.request('/key/list');
            var d = await r.json();
            return { success: true, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* P-AUDIT修复: createApiKey 原调用 /keys (断链),改为后端实际路由 /key/create */
    async createApiKey(data) {
        try {
            var r = await this.request('/key/create', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data || {})
            });
            var d = await r.json();
            return { success: true, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* P-AUDIT修复: getKeyStats 原调用 /keys/stats (断链),改为后端实际路由 /key/stats */
    async getKeyStats() {
        try {
            var r = await this.request('/key/stats');
            var d = await r.json();
            return { success: true, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* --- I-16: 多智能体补充方法 --- */
    /* P-AUDIT修复: getAgents 原调用 /agents (断链),改为后端实际路由 /multi-agent/status */
    async getAgents() {
        try {
            var r = await this.request('/multi-agent/status');
            var d = await r.json();
            return { success: true, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* P-AUDIT修复: createAgent 原调用 /agents (断链),改为后端实际路由 /multi-agent/coalition */
    async createAgent(data) {
        try {
            var r = await this.request('/multi-agent/coalition', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data || {})
            });
            var d = await r.json();
            return { success: true, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* P-AUDIT修复: agentMessage 原调用 /agents/message (断链),改为后端实际路由 /multi-agent/message */
    async agentMessage(data) {
        try {
            var r = await this.request('/multi-agent/message', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data || {})
            });
            var d = await r.json();
            return { success: true, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* --- I-17: AGI诊断补充方法 --- */
    /* getAgiDiagnostic: 获取AGI系统诊断 */
    async getAgiDiagnostic() {
        try {
            var r = await this.request('/agi/diagnostic');
            var d = await r.json();
            return { success: true, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* P-AUDIT修复: runAgiTest 原调用 /agi/test (断链),改为后端实际路由 /agi/execute */
    async runAgiTest(data) {
        try {
            var r = await this.request('/agi/execute', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data || {})
            });
            var d = await r.json();
            return { success: true, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* I-M05: exportAgiDiagnostic: 导出AGI诊断报告 */
    /* GET /api/agi/diagnostic/export → 返回完整的AGI诊断数据 */
    async exportAgiDiagnostic() {
        try {
            var r = await this.request('/agi/diagnostic/export', { method: 'GET' });
            var d = await r.json();
            return { success: true, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* --- I-18: 双目感知补充方法 --- */
    /* P-AUDIT修复: getStereoStatus 原调用 /stereo (断链),改为后端实际路由 /stereo/perception */
    /* P1-1修复: 后端期望POST方法,原GET导致方法不匹配 */
    async getStereoStatus() {
        try {
            var r = await this.request('/stereo/perception', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: '{}'
            });
            var d = await r.json();
            return { success: true, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* P-AUDIT修复: calibrateStereo 原调用 /stereo/calibrate (断链),改为后端实际路由 /stereo/perception */
    async calibrateStereo() {
        try {
            var r = await this.request('/stereo/perception', { method: 'POST' });
            var d = await r.json();
            return { success: true, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* ==================== v4修复: 补全缺失端点 ==================== */

    /* --- AGI模块补充 --- */
    /* getCognitionTom: 获取认知TOM(心理理论)状态 */
    /* GET /api/cognition/tom */
    async getCognitionTom() {
        try {
            var r = await this.request('/cognition/tom', { method: 'GET' });
            var d = await r.json();
            return { success: r.ok, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* getAgiTasks: 获取AGI任务列表 */
    /* GET /api/agi/tasks */
    async getAgiTasks() {
        try {
            var r = await this.request('/agi/tasks', { method: 'GET' });
            var d = await r.json();
            return { success: r.ok, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* createAgentCoalition: 创建多智能体联盟 */
    /* POST /api/multi-agent/coalition */
    async createAgentCoalition(data) {
        try {
            var r = await this.request('/multi-agent/coalition', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data || {})
            });
            var d = await r.json();
            return { success: r.ok, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* --- 多模态补充 --- */
    /* stereoPerception: 双目空间感知处理 */
    /* POST /api/stereo/perception */
    async stereoPerception(data) {
        try {
            var r = await this.request('/stereo/perception', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data || {})
            });
            var d = await r.json();
            return { success: r.ok, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* processMultimodal: 多模态数据处理 */
    /* POST /api/multimodal/process */
    async processMultimodal(data) {
        try {
            var r = await this.request('/multimodal/process', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data || {})
            });
            var d = await r.json();
            return { success: r.ok, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* resetMultimodal: 重置多模态系统 */
    /* POST /api/multimodal/reset */
    async resetMultimodal() {
        try {
            var r = await this.request('/multimodal/reset', { method: 'POST' });
            var d = await r.json();
            return { success: r.ok, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* --- 训练模块补充 --- */
    /* scheduleTraining: 调度训练任务 */
    /* POST /api/training/schedule */
    async scheduleTraining(data) {
        try {
            var r = await this.request('/training/schedule', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data || {})
            });
            var d = await r.json();
            return { success: r.ok, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* --- 仿真模块补充 --- */
    /* simulationReconstruct: 仿真场景重建 */
    /* POST /api/simulation/reconstruct */
    async simulationReconstruct(data) {
        try {
            var r = await this.request('/simulation/reconstruct', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data || {})
            });
            var d = await r.json();
            return { success: r.ok, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* simulationCommand: 仿真控制命令 */
    /* POST /api/simulation/command */
    async simulationCommand(data) {
        try {
            var r = await this.request('/simulation/command', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data || {})
            });
            var d = await r.json();
            return { success: r.ok, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* simulationReconstruct3d: 3D仿真重建 */
    /* POST /api/simulation/reconstruct3d */
    async simulationReconstruct3d(data) {
        try {
            var r = await this.request('/simulation/reconstruct3d', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data || {})
            });
            var d = await r.json();
            return { success: r.ok, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* --- 机器人模块补充 --- */
    /* robotPathPlan: 机器人路径规划 */
    /* POST /api/robot/path/plan */
    async robotPathPlan(data) {
        try {
            var r = await this.request('/robot/path/plan', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data || {})
            });
            var d = await r.json();
            return { success: r.ok, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* getRobotParams: 获取机器人参数 */
    /* P-AUDIT修复: 后端 /robot/params 为 POST,原用 GET 导致 405 */
    async getRobotParams() {
        try {
            var r = await this.request('/robot/params', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
            var d = await r.json();
            return { success: r.ok, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* --- 知识图谱补充 --- */
    /* getKgPagerank: 获取知识图谱PageRank */
    /* GET /api/kg/pagerank */
    async getKgPagerank() {
        try {
            var r = await this.request('/kg/pagerank', { method: 'GET' });
            var d = await r.json();
            return { success: r.ok, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* getKgCommunities: 获取知识图谱社区结构 */
    /* GET /api/kg/communities */
    async getKgCommunities() {
        try {
            var r = await this.request('/kg/communities', { method: 'GET' });
            var d = await r.json();
            return { success: r.ok, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* getKgPath: 知识图谱路径查询 */
    /* POST /api/kg/path */
    async getKgPath(data) {
        try {
            var r = await this.request('/kg/path', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data || {})
            });
            var d = await r.json();
            return { success: r.ok, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* queryKgSparql: SPARQL知识图谱查询 */
    /* POST /api/kg/sparql */
    async queryKgSparql(query) {
        try {
            var r = await this.request('/kg/sparql', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: query })
            });
            var d = await r.json();
            return { success: r.ok, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* --- 音频/摄像头模块补充 --- */
    /* startAudioCapture: 启动音频采集 */
    /* POST /api/audio/capture/start */
    async startAudioCapture() {
        try {
            var r = await this.request('/audio/capture/start', { method: 'POST' });
            var d = await r.json();
            return { success: r.ok, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* stopAudioCapture: 停止音频采集 */
    /* POST /api/audio/capture/stop */
    async stopAudioCapture() {
        try {
            var r = await this.request('/audio/capture/stop', { method: 'POST' });
            var d = await r.json();
            return { success: r.ok, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* getCameraDevices: 获取摄像头设备列表 */
    /* GET /api/camera/devices */
    async getCameraDevices() {
        try {
            var r = await this.request('/camera/devices', { method: 'GET' });
            var d = await r.json();
            return { success: r.ok, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* startCameraCapture: 启动摄像头采集 */
    /* POST /api/camera/capture/start */
    async startCameraCapture() {
        try {
            var r = await this.request('/camera/capture/start', { method: 'POST' });
            var d = await r.json();
            return { success: r.ok, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* stopCameraCapture: 停止摄像头采集 */
    /* POST /api/camera/capture/stop */
    async stopCameraCapture() {
        try {
            var r = await this.request('/camera/capture/stop', { method: 'POST' });
            var d = await r.json();
            return { success: r.ok, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* --- GPU模块补充 --- */
    /* getGpuDiagnostic: 获取GPU诊断信息 */
    /* GET /api/gpu/diagnostic */
    async getGpuDiagnostic() {
        try {
            var r = await this.request('/gpu/diagnostic', { method: 'GET' });
            var d = await r.json();
            return { success: r.ok, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* runGpuBenchmark: 运行GPU基准测试 */
    /* POST /api/gpu/benchmark */
    async runGpuBenchmark() {
        try {
            var r = await this.request('/gpu/benchmark', { method: 'POST' });
            var d = await r.json();
            return { success: r.ok, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* --- 数据集模块补充 --- */
    /* importDataset: 导入数据集 */
    /* POST /api/dataset/import */
    async importDataset(data) {
        try {
            var r = await this.request('/dataset/import', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data || {})
            });
            var d = await r.json();
            return { success: r.ok, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* augmentDataset: 数据增强 */
    /* POST /api/dataset/augment */
    async augmentDataset(data) {
        try {
            var r = await this.request('/dataset/augment', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data || {})
            });
            var d = await r.json();
            return { success: r.ok, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* --- 系统模块补充 --- */
    /* getSystemLogs: 获取系统日志 */
    /* GET /api/system/logs */
    async getSystemLogs() {
        try {
            var r = await this.request('/system/logs', { method: 'GET' });
            var d = await r.json();
            return { success: r.ok, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* getKeyRateLimit: 获取API密钥速率限制 */
    /* GET /api/key/rate-limit */
    async getKeyRateLimit() {
        try {
            var r = await this.request('/key/rate-limit', { method: 'GET' });
            var d = await r.json();
            return { success: r.ok, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* --- 串口模块补充 --- */
    /* receiveSerial: 接收串口数据 */
    /* GET /api/serial/receive */
    async receiveSerial() {
        try {
            var r = await this.request('/serial/receive', { method: 'GET' });
            var d = await r.json();
            return { success: r.ok, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* --- 自学习模块补充 --- */
    /* exportAutoLearn: 导出自动学习结果 */
    /* POST /api/auto-learn/export */
    async exportAutoLearn() {
        try {
            var r = await this.request('/auto-learn/export', { method: 'POST' });
            var d = await r.json();
            return { success: r.ok, data: d };
        } catch (e) { return { success: false, error: e.message }; }
    }

    /* === H-007修复: 后端已实现但前端缺失的API封装补充(20个) === */

    /* --- GPU诊断与基准测试 --- */
    /* P1-2修复: getGpuDiagnostic为重复定义,已删除(保留先定义于GPU模块补充区) */

    /* postGpuBenchmark: 运行GPU基准测试 */
    /* POST /api/gpu/benchmark */
    async postGpuBenchmark() {
        try { var r = await this.request('/gpu/benchmark', { method: 'POST' }); var d = await r.json(); return { success: r.ok, data: d }; }
        catch (e) { return { success: false, error: e.message }; }
    }

    /* --- 数据集管理 --- */
    /* P1-2修复: getDatasets、createDataset为重复定义,已删除(保留先定义于I-10数据集管理区,调用/datasets更标准) */

    /* getDatasetStats: 获取数据集统计信息 */
    /* GET /api/dataset/stats */
    async getDatasetStats(datasetId) {
        try { var r = await this.request('/dataset/stats?dataset_id=' + encodeURIComponent(datasetId || '')); var d = await r.json(); return { success: r.ok, data: d }; }
        catch (e) { return { success: false, error: e.message }; }
    }

    /* --- 演化系统 --- */
    /* getEvolutionStatus: 获取演化系统状态 */
    /* GET /api/evolution/status */
    async getEvolutionStatus() {
        try { var r = await this.request('/evolution/status'); var d = await r.json(); return { success: r.ok, data: d }; }
        catch (e) { return { success: false, error: e.message }; }
    }

    /* --- 知识库版本与导入导出 --- */
    /* getKnowledgeVersion: 获取知识库版本信息 */
    /* GET /api/knowledge/version */
    async getKnowledgeVersion() {
        try { var r = await this.request('/knowledge/version'); var d = await r.json(); return { success: r.ok, data: d }; }
        catch (e) { return { success: false, error: e.message }; }
    }

    /* P1-2修复: exportKnowledge、importKnowledge为重复定义,已删除(保留先定义) */

    /* --- 系统诊断与日志 --- */
    /* getSystemDiagnostic: 运行系统诊断 */
    /* GET /api/system/diagnostic */
    async getSystemDiagnostic() {
        try { var r = await this.request('/system/diagnostic'); var d = await r.json(); return { success: r.ok, data: d }; }
        catch (e) { return { success: false, error: e.message }; }
    }

    /* P1-2修复: exportDiagnosticData、getSystemLogs为重复定义,已删除(保留先定义) */

    /* restartSystem: 重启系统 */
    /* POST /api/system/restart */
    async restartSystem() {
        try { var r = await this.request('/system/restart', { method: 'POST' }); var d = await r.json(); return { success: r.ok, data: d }; }
        catch (e) { return { success: false, error: e.message }; }
    }

    /* --- 摄像头控制 --- */
    /* P1-2修复: getCameraDevices、startCameraCapture、stopCameraCapture为重复定义,已删除(保留先定义) */

    /* --- 音频采集 --- */
    /* P1-2修复: startAudioCapture、stopAudioCapture为重复定义,已删除(保留先定义) */

    /* --- 知识图谱SPARQL查询 --- */
    /* kgSparql: 执行SPARQL语义查询 */
    /* POST /api/kg/sparql */
    async kgSparql(query) {
        try { var r = await this.request('/kg/sparql', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query: query }) }); var d = await r.json(); return { success: r.ok, data: d }; }
        catch (e) { return { success: false, error: e.message }; }
    }

}

/**
 * SELF-LNN WebSocket 实时通信管理器
 * 支持断线自动重连（指数退避+随机抖动）、消息队列、心跳检测（Ping/Pong超时）、网络状态检测
 */
class WebSocketManager {
    constructor(url) {
        /* ZSFOOO-001修复: WebSocket使用独立端口8081，与port_config.h中SELFLNN_WEBSOCKET_PORT一致 */
        var cfg = window.SELFLNN_CONFIG || { host: 'localhost', port: 8080, wsPort: 8081 };
        /* 后端WebSocket服务器监听独立端口8081，HTTP服务器8080未实现WebSocket Upgrade */
        var wsPort = cfg.wsPort || cfg.port + 1 || 8081;
        var wsProtocol = (cfg.host === 'localhost' || cfg.host === '127.0.0.1') ? 'ws://' : 'wss://';
        this.url = url || (wsProtocol + cfg.host + ':' + wsPort + '/ws');
        this.ws = null;
        this.isConnected = false;
        this.isManualDisconnect = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 50;
        this.reconnectDelay = 1000;
        this.maxReconnectDelay = 60000;
        this.heartbeatInterval = null;
        this.heartbeatTimeout = 30000;
        this.pongTimeout = 10000;
        this.pongTimer = null;
        this.lastPingTimestamp = 0;
        this.pongReceived = true;
        this.messageHandlers = {};
        this.pendingMessages = [];
        this.pendingMessageTimestamps = [];
        this.statusCallback = null;
        this.reconnectCallback = null;
        this.reconnectTimer = null;
        this.lastOnlineCheck = true;
        this.idleReconnectInterval = null;
    }

    /**
     * 建立WebSocket连接
     */
    connect() {
        if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
            return;
        }
        if (typeof navigator !== 'undefined' && navigator.onLine === false) {
            console.warn('设备处于离线状态，等待网络恢复后将自动重连');
            this._scheduleReconnect();
            return;
        }
        this.isManualDisconnect = false;
        try {
            this.ws = new WebSocket(this.url);
            this.ws.onopen = () => this._onOpen();
            this.ws.onclose = (event) => this._onClose(event);
            this.ws.onerror = (error) => this._onError(error);
            this.ws.onmessage = (message) => this._onMessage(message);
        } catch (error) {
            console.error('WebSocket连接创建失败:', error);
            /* L-12修复: WebSocket初始化失败时向用户提示 */
            if (typeof showNotification === 'function') {
                showNotification('WebSocket连接失败，将自动重连。离线功能仍可用。', 'warning');
            }
            this._scheduleReconnect();
        }
    }

    /**
     * 断开WebSocket连接
     */
    disconnect() {
        this.isManualDisconnect = true;
        this._cleanup();
        if (this.ws) {
            this.ws.close(1000, '正常关闭');
            this.ws = null;
        }
        this.isConnected = false;
        this._notifyStatus({ connected: false, reason: 'manual' });
    }

    /**
     * 发送数据
     */
    send(data) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            const message = typeof data === 'string' ? data : JSON.stringify(data);
            this.ws.send(message);
            return true;
        }
        const timestamp = Date.now();
        this.pendingMessages.push(data);
        this.pendingMessageTimestamps.push(timestamp);
        /* 清理超过30秒的过期待发送消息 */
        this._cleanExpiredPending();
        return false;
    }

    /**
     * 设置连接状态回调
     */
    onStatusChange(callback) {
        this.statusCallback = callback;
    }

    /**
     * 设置重连状态回调（每次重连尝试时通知详细信息）
     */
    onReconnect(callback) {
        this.reconnectCallback = callback;
    }

    /** WebSocket打开事件 */
    _onOpen() {
        this.isConnected = true;
        this.reconnectAttempts = 0;
        this.reconnectDelay = 1000;
        this.pongReceived = true;
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }
        this._startHeartbeat();
        this._flushPendingMessages();
        this._notifyStatus({ connected: true });
    }

    /** WebSocket关闭事件 */
    _onClose(event) {
        this.isConnected = false;
        this._stopHeartbeat();
        this._clearPongTimer();
        this._notifyStatus({ connected: false, code: event.code, reason: event.reason });
        /* 正常关闭(code 1000) 或 手动断开 不自动重连 */
        if (!this.isManualDisconnect && event.code !== 1000) {
            this._scheduleReconnect();
        }
    }

    /** WebSocket错误事件 */
    _onError(error) {
        console.error('WebSocket连接错误:', error);
        /* L-12修复: WebSocket错误时向用户提示 */
        if (typeof showNotification === 'function') {
            showNotification('WebSocket通信错误，正在重连...', 'warning');
        }
        this._notifyStatus({ connected: false, reason: 'error' });
    }

    /** 注册对话流式消息回调 */
    onDialogueToken(callback) {
        this.on('dialogue_token', callback);
        return this;
    }

    onDialogueResponse(callback) {
        this.on('dialogue_response', callback);
        return this;
    }

    on(typeOrTypes, callback) {
        const types = Array.isArray(typeOrTypes) ? typeOrTypes : [typeOrTypes];
        types.forEach(type => {
            if (!this.messageHandlers[type]) {
                this.messageHandlers[type] = [];
            }
            this.messageHandlers[type].push(callback);
        });
        return this;
    }

    off(type, callback) {
        const handlers = this.messageHandlers[type];
        if (handlers) {
            const idx = handlers.indexOf(callback);
            if (idx !== -1) handlers.splice(idx, 1);
        }
        return this;
    }

    /** WebSocket消息事件 */
    _onMessage(message) {
        try {
            const data = JSON.parse(message.data);
            /* 处理Pong响应 */
            if (data.type === 'pong') {
                this.pongReceived = true;
                this._clearPongTimer();
                return;
            }
            /* 处理对话token(type=8)和对话响应(type=7)的数值类型 */
            if (typeof data.type === 'number') {
                if (data.type === 8) {
                    data.type_str = 'dialogue_token';
                    const handlers = this.messageHandlers['dialogue_token'];
                    if (handlers) handlers.forEach(handler => handler(data));
                } else if (data.type === 7) {
                    data.type_str = 'dialogue_response';
                    const handlers = this.messageHandlers['dialogue_response'];
                    if (handlers) handlers.forEach(handler => handler(data));
                }
                return;
            }
            const type = data.type || data.event || data.action || 'message';
            const handlers = this.messageHandlers[type];
            if (handlers) {
                handlers.forEach(handler => {
                    try {
                        handler(data);
                    } catch (err) {
                        console.error(`WebSocket消息处理器异常 (type=${type}):`, err);
                    }
                });
            }
            const generalHandlers = this.messageHandlers['*'];
            if (generalHandlers) {
                generalHandlers.forEach(handler => {
                    try {
                        handler(data);
                    } catch (err) {
                        console.error('WebSocket通用消息处理器异常:', err);
                    }
                });
            }
        } catch (error) {
            console.warn('WebSocket消息解析失败(JSON格式错误)');
        }
    }

    /** 启动心跳检测 */
    _startHeartbeat() {
        this._stopHeartbeat();
        this.pongReceived = true;
        this.heartbeatInterval = setInterval(() => {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.lastPingTimestamp = Date.now();
                this.pongReceived = false;
                this.ws.send(JSON.stringify({ type: 'ping', timestamp: this.lastPingTimestamp }));
                /* 设置Pong超时检测 */
                this._clearPongTimer();
                this.pongTimer = setTimeout(() => {
                    if (!this.pongReceived) {
                        console.warn('Pong响应超时, 判定连接断开, 触发重连');
                        this._notifyStatus({ connected: false, reason: 'pong_timeout' });
                        if (this.ws) {
                            this.ws.close(4001, 'Pong超时');
                            this.ws = null;
                        }
                        this.isConnected = false;
                        if (!this.isManualDisconnect) {
                            this._scheduleReconnect();
                        }
                    }
                }, this.pongTimeout);
            }
        }, this.heartbeatTimeout);
    }

    /** 停止心跳检测 */
    _stopHeartbeat() {
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
            this.heartbeatInterval = null;
        }
        this._clearPongTimer();
    }

    /** 清理Pong超时定时器 */
    _clearPongTimer() {
        if (this.pongTimer) {
            clearTimeout(this.pongTimer);
            this.pongTimer = null;
        }
    }

    /** 清理过期待发送消息（超过30秒的丢弃） */
    _cleanExpiredPending() {
        const now = Date.now();
        const cutoff = 30000;
        while (this.pendingMessageTimestamps.length > 0 &&
               (now - this.pendingMessageTimestamps[0]) > cutoff) {
            this.pendingMessageTimestamps.shift();
            this.pendingMessages.shift();
        }
    }

    /** 发送待发送消息队列 */
    _flushPendingMessages() {
        while (this.pendingMessages.length > 0) {
            const msg = this.pendingMessages.shift();
            this.pendingMessageTimestamps.shift();
            this.send(msg);
        }
    }

    /** 安排重连（指数退避 + 随机抖动） */
    _scheduleReconnect() {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.error('WebSocket重连次数已达上限:', this.maxReconnectAttempts);
            this._notifyStatus({ connected: false, reason: 'max_attempts' });
            return;
        }
        /* 检测网络状态 */
        if (typeof navigator !== 'undefined' && navigator.onLine === false) {
            if (this.lastOnlineCheck !== false) {
                console.warn('设备离线，暂停重连，等待网络恢复');
                this.lastOnlineCheck = false;
                this._notifyStatus({ connected: false, reason: 'offline' });
            }
            /* 网络恢复后立即重连 */
            if (!this.idleReconnectInterval) {
                this.idleReconnectInterval = setInterval(() => {
                    if (navigator.onLine) {
                        clearInterval(this.idleReconnectInterval);
                        this.idleReconnectInterval = null;
                        this.lastOnlineCheck = true;
                        this.connect();
                    }
                }, 2000);
            }
            return;
        }
        this.lastOnlineCheck = true;

        /* 渐进式退避：长时间重连失败后延长最大间隔 */
        let effectiveMaxDelay = this.maxReconnectDelay;
        if (this.reconnectAttempts > 20) {
            effectiveMaxDelay = Math.min(this.maxReconnectDelay * 2, 120000);
        }

        /* 指数退避 + 随机抖动 ±30% */
        const baseDelay = Math.min(this.reconnectDelay, effectiveMaxDelay);
        const jitter = baseDelay * (0.7 + Math.random()* 0.6);
        const delay = Math.round(jitter);

        /* 通知重连状态 */
        this._notifyReconnect(this.reconnectAttempts + 1, delay);

        this.reconnectTimer = setTimeout(() => {
            this.reconnectTimer = null;
            this.reconnectAttempts++;
            this.reconnectDelay = Math.min(this.reconnectDelay * 2, effectiveMaxDelay);
            this.connect();  /* P0-2修复: 添加()实际调用connect方法，原代码this.connect缺少括号导致重连定时器触发后不执行重连 */
        }, delay);
    }

    /** 通知重连进度 */
    _notifyReconnect(attempt, delay) {
        if (this.reconnectCallback) {
            this.reconnectCallback({
                attempt: attempt,
                maxAttempts: this.maxReconnectAttempts,
                delay: delay,
                nextAttempt: delay
            });
        }
        /* 通过自定义事件通知 */
        const event = new CustomEvent('websocket-reconnect-status', {
            detail: {
                attempt: attempt,
                maxAttempts: this.maxReconnectAttempts,
                delay: delay,
                url: this.url
            }
        });
        document.dispatchEvent(event);
    }

    /** 清理资源 */
    _cleanup() {
        this._stopHeartbeat();
        this.pendingMessages = [];
        this.pendingMessageTimestamps = [];
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }
        if (this.idleReconnectInterval) {
            clearInterval(this.idleReconnectInterval);
            this.idleReconnectInterval = null;
        }
    }

    /** 通知连接状态变化 */
    _notifyStatus(status) {
        if (this.statusCallback) {
            this.statusCallback(status);
        }
        const event = new CustomEvent('websocket-connection-status', {
            detail: Object.assign({ url: this.url }, status)
        });
        document.dispatchEvent(event);
    }
}

// 创建全局API服务实例（IIFE内暴露）
window.SelfLnnApi = new ApiService;
/* WebSocket通过HTTP Upgrade共用HTTP端口 — 构造函数动态读取window.SELFLNN_CONFIG */
window.SelfLnnWebSocket = new WebSocketManager;

}); /* IIFE结束 */

/* WebSocket自动连接已启用（延迟连接, 确保服务器就绪） */

/* 全局连接横幅管理器 — 所有页面共享 */
(function() {
    var banner = null;
    function ensureBanner() {
        if (!banner) {
            banner = document.getElementById('global-connection-banner');
            if (!banner) {
                banner = document.createElement('div');
                banner.id = 'global-connection-banner';
                banner.className = 'connection-banner';
                banner.style.cursor = 'pointer';
                banner.onclick = function() {
                    if (window.SelfLnnWebSocket && !window.SelfLnnWebSocket.isConnected) {
                        window.SelfLnnWebSocket.connect();
                    }
                };
                /* 插入到body最前面，避免影响布局（position:fixed不影响文档流，但放在前面更规范） */
                if (document.body) document.body.insertBefore(banner, document.body.firstChild);
            }
        }
        return banner;
    }
    document.addEventListener('websocket-connection-status', function(e) {
        var b = ensureBanner();
        if (e.detail && e.detail.connected) {
            b.className = 'connection-banner js-ready connected';
            b.innerHTML = '<span class="connection-dot connected"></span> WebSocket已连接';
        } else {
            b.className = 'connection-banner js-ready disconnected';
            b.innerHTML = '<span class="connection-dot disconnected"></span> 点击重连WebSocket';
        }
    });
});

/* 注册对话流式消息全局分发（使用WebSocketManager.on接口） */
(function() {
    var ws = window.SelfLnnWebSocket;
    if (!ws) return;

    /* 分发dialogue_token事件到全局和DialogueEnhanced */
    ws.on('dialogue_token', function(data) {
        var event = new CustomEvent('selflnn:dialogue-token', {
            detail: {
                token: data.token || data.text || '',
                token_id: data.token_id || -1,
                progress: data.progress || 0,
                is_final: data.is_final || 0
            }
        });
        document.dispatchEvent(event);

        if (window.g_dialogueEnhanced && window.g_dialogueEnhanced.onDialogueToken) {
            window.g_dialogueEnhanced.onDialogueToken(
                data.token || data.text || '',
                data.progress || 0,
                data.is_final || 0
            );
        }
    });

    /* 分发dialogue_response事件 */
    ws.on('dialogue_response', function(data) {
        var event = new CustomEvent('selflnn:dialogue-response', {
            detail: {
                response: data.text || data.response || '',
                confidence: data.confidence || 0,
                tokens_used: data.tokens_used || 0
            }
        });
        document.dispatchEvent(event);

        if (window.g_dialogueEnhanced && window.g_dialogueEnhanced.onDialogueResponse) {
            window.g_dialogueEnhanced.onDialogueResponse(
                data.text || data.response || '',
                data.confidence || 0
            );
        }
    });

/* 延迟启动WebSocket连接（确保服务器先完成初始化） */
    setTimeout(function() {
        ws.connect();
    }, 3000);
});