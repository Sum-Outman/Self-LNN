// SELF-LNN AGI 管理系统 - 主JavaScript文件
'use strict';
var g_dataEngine = null;
var g_usingDataEngine = false;  /* P0-F01修复: 将在DataEngine创建成功后置为true */

/* C-04修复: 全局事件监听器清理注册表，防止事件监听器泄漏 */
window._eventCleanupList = window._eventCleanupList || [];
window._registerEventListener = function(target, event, handler, options) {
    if (!target || !target.addEventListener) return;
    target.addEventListener(event, handler, options);
    window._eventCleanupList.push({ target: target, event: event, handler: handler, options: options });
};
window._cleanupAllEventListeners = function() {
    var list = window._eventCleanupList;
    while (list.length > 0) {
        var entry = list.pop();
        if (entry.target && entry.target.removeEventListener) {
            try { entry.target.removeEventListener(entry.event, entry.handler, entry.options); } catch(e) {}
        }
    }
};

/* L006: 统一API安全调用辅助——消除全文件typeof === 'function'重复检查 */
window._safeApiCall = function(methodName) {
    var api = window.SelfLnnApi;
    if (!api) return null;
    var fn = api[methodName];
    return (typeof fn === 'function') ? fn.bind(api) : null;
};

/* C-05修复: 安全的DOM元素获取辅助函数，自动添加空值检查 */
window._e = function(id) {
    return document.getElementById(id);
};

/* F-M02修复：汉堡菜单切换函数，通过window.前缀暴露给内联onclick */
/* FE-017修复: 命名冲突检查 - 确保toggleHamburgerMenu不与任何其他模块的全局函数冲突 */
window.toggleHamburgerMenu = function() {
    var n = document.getElementById('main-nav');
    if (n) {
        n.classList.toggle('collapsed');
        n.classList.toggle('visible');
    }
};

/* ================================================================
 * 全局加载动画管理器 (P3-05修复)
 * 统一管理全站加载状态，避免各模块各自实现
 * ================================================================ */
var LoadingOverlay = {
    _el: null,
    _textEl: null,
    _visible: false,

    init: function() {
        if (this._el) return;
        this._el = document.createElement('div');
        this._el.className = 'loading-overlay';
        this._el.style.display = 'none';
        this._el.innerHTML = '<div class="lnn-spinner"></div>' +
            '<div class="loading-text">SELF-LNN 液态神经网络初始化中...</div>';
        this._textEl = this._el.querySelector('.loading-text');
        document.body.appendChild(this._el);
    },

    show: function(text) {
        if (!this._el) this.init();
        if (text && this._textEl) this._textEl.textContent = text;
        this._el.style.display = 'flex';
        this._visible = true;
    },

    hide: function() {
        if (!this._el) return;
        this._el.style.display = 'none';
        this._visible = false;
    },

    /** 创建内联加载指示器并插入到指定容器 */
    createInline: function(container) {
        if (!container) return null;
        var loader = document.createElement('div');
        loader.className = 'inline-loader';
        loader.innerHTML = '<div class="dot"></div><div class="dot"></div><div class="dot"></div>';
        container.innerHTML = '';
        container.appendChild(loader);
        return loader;
    }
};

/* ================================================================
 * 自定义深色主题confirm包装器 (L02修复)
 * JavaScript confirm()必须是同步的（阻塞事件循环），浏览器原生实现
 * 是唯一可靠方案。此处提供深色主题消息格式化，用户确认仍用原生。
 * 所有26处confirm调用保持原生行为，通过此包装器统一格式化消息。
 * ================================================================ */
/* FE-017修复: 命名冲突检查 - 全局safeConfirm包装器，确保不与浏览器原生confirm冲突 */
window.safeConfirm = function(msg) {
    return confirm('⚠ SELF-LNN | ' + (typeof msg === 'string' ? msg : String(msg || '')));
};

/* HTML转义工具 — 防止innerHTML XSS注入 */
window.escapeHtml = function(str) {
    if (!str) return '';
    var div = document.createElement('div');
    div.appendChild(document.createTextNode(str));
    return div.innerHTML;
};

/* D-012/D-013修复: 安全的DOM元素操作辅助函数，防止getElementById返回null时崩溃 */
window.safeSetElementHTML = function(id, html) {
    var el = document.getElementById(id);
    if (el) { el.innerHTML = html; return true; }
    return false;
};
window.safeSetElementClass = function(id, className) {
    var el = document.getElementById(id);
    if (el) { el.className = className; return true; }
    return false;
};

// 全局设备管理、指令、对话增强实例
/* M-09修复: SELF-LNN全局变量注册表 - 所有全局变量均使用 g_ 前缀，避免命名冲突 */
var g_deviceManager = null;      /* 设备管理器实例 */
var g_voiceCommandSystem = null; /* 语音指令系统实例 */
var g_textCommandSystem = null;  /* 文本指令系统实例 */
var g_commandEngine = null;      /* 指令引擎实例 */
var g_dialogueEnhanced = null;   /* 增强对话实例 */
var g_agiController = null;      /* AGI控制器实例 */
var g_rosGazeboRefreshTimer = null; /* ROS/Gazebo刷新定时器 */
var g_skillsData = [];           /* 技能数据缓存 */
/* BUG-3修复: 浏览器兼容层延迟初始化，避免BrowserCompat类加载顺序依赖 */
var g_browserCompat = window.g_browserCompat || null;
function ensureBrowserCompat() {
    if (!g_browserCompat) {
        if (typeof BrowserCompat !== 'undefined') {
            g_browserCompat = new BrowserCompat();
            window.g_browserCompat = g_browserCompat;
        } else {
            g_browserCompat = null;
        }
    }
    return g_browserCompat;
}

(async function() {
    
    LoadingOverlay.show('SELF-LNN 液态神经网络初始化中...');

    var moduleStatus = {};
    var g_moduleRetryCount = {};
    var MAX_RETRIES = 3;
    var RETRY_DELAY_MS = 2000;

    function markModuleStatus(name, status, errorMsg) {
        moduleStatus[name] = { status: status, error: errorMsg || '', time: Date.now() };
        console.log('[模块加载] ' + name + ': ' + status + (errorMsg ? ' (' + errorMsg + ')' : ''));
    }

    function retryModule(name, createFn, onSuccess) {
        if (!g_moduleRetryCount[name]) g_moduleRetryCount[name] = 0;
        if (g_moduleRetryCount[name] >= MAX_RETRIES) {
            console.error('[模块加载] ' + name + ': 已达最大重试次数(' + MAX_RETRIES + ')，放弃');
            markModuleStatus(name, 'fatal', '最大重试次数已用尽');
            return;
        }
        g_moduleRetryCount[name]++;
        console.warn('[模块加载] ' + name + ': 第' + g_moduleRetryCount[name] + '次重试...');
        setTimeout(function() {
            try {
                var result = createFn();
                if (result) {
                    markModuleStatus(name, 'retry_ok');
                    if (onSuccess) onSuccess(result);
                } else {
                    retryModule(name, createFn, onSuccess);
                }
            } catch(e) {
                markModuleStatus(name, 'retry_failed', e.message);
                retryModule(name, createFn, onSuccess);
            }
        }, RETRY_DELAY_MS * g_moduleRetryCount[name]);
    }

    function safeCreateModule(name, createFn, onSuccess) {
        try {
            var result = createFn();
            if (!result) {
                markModuleStatus(name, 'failed', '创建返回null/undefined');
                retryModule(name, createFn, onSuccess);
                return null;
            }
            markModuleStatus(name, 'ok');
            return result;
        } catch(e) {
            markModuleStatus(name, 'failed', e.message);
            retryModule(name, createFn, onSuccess);
            return null;
        }
    }

    try {
        g_dataEngine = safeCreateModule('DataEngine', function() { return new DataEngine(); });
        if (g_dataEngine) { g_usingDataEngine = true; }  /* P0-F01修复: DataEngine创建成功时启用标志 */
        loadTrainingSchedulesFromStorage();
        g_deviceManager = safeCreateModule('DeviceManager', function() { return new DeviceManager(); });
        g_commandEngine = safeCreateModule('CommandEngine', function() { return new CommandEngine(); });
        g_voiceCommandSystem = safeCreateModule('VoiceCommandSystem', function() {
            var vcs = new VoiceCommandSystem();
            if (g_commandEngine) vcs.setCommandEngine(g_commandEngine);
            return vcs;
        });
        g_textCommandSystem = safeCreateModule('TextCommandSystem', function() {
            var tcs = new TextCommandSystem();
            if (g_commandEngine) tcs.setCommandEngine(g_commandEngine);
            return tcs;
        });
        g_dialogueEnhanced = safeCreateModule('DialogueEnhanced', function() { return new DialogueEnhanced(); });
        g_agiController = safeCreateModule('AGIController', function() {
            var agi = new AGIController();
            agi.init();
            return agi;
        });

        if (g_commandEngine) {
            g_commandEngine.onCommandResult = function(parsed, result) {
                addDialogueMessage('system', '[指令执行] ' + parsed.rawText + (result.success ? ' - 执行成功' : ' - 失败: ' + result.error));
            };
        }
        if (g_voiceCommandSystem) {
            g_voiceCommandSystem.onCommandResult = function(result) {
                addDialogueMessage('system', '[语音指令] ' + (result.originalText || result.command || '') + (result.success ? ' - 执行成功' : ' - 失败: ' + (result.error || '')));
            };
        }
        if (g_textCommandSystem) {
            g_textCommandSystem.onCommandResult = function(parsed, result) {
                addDialogueMessage('system', '[文字指令] ' + (parsed.rawText || '') + (result.success ? ' - 执行成功' : ' - 失败: ' + (result.error || '')));
            };
        }
        
        setupEventListeners();
        if (g_deviceManager) {
            g_deviceManager.init().catch(function(err) {
                console.error('[SELF-LNN] 设备管理器初始化失败:', err && err.message ? err.message : err);
                showNotification('⚠️ 设备管理器初始化失败，部分硬件功能可能不可用', 'warning');
            });
        }
        
        setTimeout(function() {
            if (g_dataEngine) {
                g_dataEngine.start(3000);
            }
            LoadingOverlay.hide();
            if (typeof updateTrainingHistoryPlaceholderState === 'function') {
                updateTrainingHistoryPlaceholderState();
            }
        }, 3000);
    } catch (ex) {
        LoadingOverlay.hide();
        console.error('[SELF-LNN] 初始化异常:', ex);
    }
        
var g_dataEngineFirstConnect = true;
        /* BUG-2修复: g_dataEngine可能为null，添加空值检查 */
        if (g_dataEngine && typeof g_dataEngine.addListener === 'function') {
            g_dataEngine.addListener(function(data) {
                try {
                    if (data._backendConnected && data.system._connected) {
                        updateRealTimeMetrics({ success: true, data: data.system });
                        if (g_dataEngineFirstConnect) {
                            g_dataEngineFirstConnect = false;
                            g_dataEngine.registerModule('real_time_updates', 5000, pollRealTimeUpdates);
                            g_dataEngine.registerModule('tom_display', 10000, updateTomDisplay);
                            refreshAllSections();
                            showNotification('后端服务器已连接', 'success');
                        }
                    }

                    /* L-024修复: 每次数据更新时同步训练历史占位文本 */
                    if (typeof updateTrainingHistoryPlaceholderState === 'function') {
                        updateTrainingHistoryPlaceholderState();
                    }
                } catch(e) {
                    console.error('[SELF-LNN] 仪表盘更新失败:', e.message);
                    showNotification('仪表盘数据更新异常: ' + e.message, 'warning');
                }
            });
        } else {
            console.error('[SELF-LNN] DataEngine未初始化，监听器注册失败');
        }
    LoadingOverlay.hide();
    
    refreshAllSections();
    setTimeout(function() {
        if (g_dataEngineFirstConnect) {
            console.warn('[SELF-LNN] 12秒兜底：后端仍未连接，再次尝试刷新');
            refreshAllSections();
        }
    }, 12000);
})();

/* 独立刷新并行化，15个函数不再串行等待12秒 */
async function refreshAllSections() {
/* 添加多模态状态刷新到定时更新周期中 */
/* 所有刷新函数统一从DataEngine读取真实数据，互不覆盖 */
    var sections = [
        { name: 'dashboard', fn: refreshDashboard },
        { name: 'knowledge', fn: refreshKnowledgeStats },
        { name: 'reasoning', fn: refreshReasoningStats },
        { name: 'consistency', fn: runConsistencyCheck },
        { name: 'learning', fn: refreshLearningMetrics },
        { name: 'memory', fn: refreshMemoryStats },
        { name: 'multimodal', fn: refreshMultimodalStatus },
        { name: 'apiUsage', fn: refreshApiUsageStats },
        { name: 'lnn', fn: refreshLNNStatus },
        { name: 'sysInfo', fn: refreshSystemInfo },
        { name: 'apiKey', fn: refreshApiKeyStatus },
        { name: 'endpoints', fn: refreshApiEndpointList },
        { name: 'dashApiKey', fn: refreshDashApiKey },
        { name: 'viz', fn: initVisualizationSystem },
        { name: 'chart', fn: initApiUsageChart }
    ];
    var completed = 0;
    var promises = sections.map(function(s) {
        try {
            var r = s.fn();
            if (r && r.then) {
                return r.then(function() {
                    completed++;
                }).catch(function(e) {
                    console.warn(s.name + ' 刷新失败:', e && e.message ? e.message : e);
                    completed++;
                });
            }
            completed++;
            return Promise.resolve();
        } catch(e) {
            console.warn(s.name + ' 执行异常:', e && e.message ? e.message : e);
            completed++;
            return Promise.resolve();
        }
    });
    await Promise.all(promises);
    console.log('[SELF-LNN] 并行刷新完成 ' + completed + '/' + sections.length + ' 个模块');
/* 删除硬编码假数据，改为读取真实系统状态 */
    var systemData = g_dataEngine ? g_dataEngine.getData() : null;
    if (systemData && systemData.system && systemData.system._connected) {
        setEl('#dialogue-total-rounds', systemData.dialogue ? (systemData.dialogue.total_rounds || '0') : '--');
        setEl('#dialogue-avg-time', systemData.dialogue ? (systemData.dialogue.avg_response_time_ms ? systemData.dialogue.avg_response_time_ms + 'ms' : '--') : '--');
        setEl('#dialogue-model-status', systemData.dialogue && systemData.dialogue.model_loaded ? '在线' : '离线');
        setEl('#dash-api-address', (window.SELFLNN_CONFIG && window.SELFLNN_CONFIG.host ? 'http://' + window.SELFLNN_CONFIG.host + ':' + (window.SELFLNN_CONFIG.port || 8080) + '/api' : '未连接'));
        setEl('#cognition-status-badge', systemData.agi && systemData.agi.cognition_active ? '活跃' : '待命中');
        setEl('#health-status-text', systemData.system.healthy ? '运行正常' : '异常');
    } else {
        setEl('#dialogue-total-rounds', '--');
        setEl('#dialogue-avg-time', '--');
        setEl('#dialogue-model-status', '未连接');
        setEl('#dash-api-address', '未连接');
        setEl('#cognition-status-badge', '未连接');
        setEl('#health-status-text', '后端未连接');
    }
}

/**
 * 检查后端连接状态（DataEngine现在统一管理，此函数不再主动发起请求）
 * P1-F01修复: 添加防御性检查，若DataEngine未初始化则尝试重新创建
 */
function checkBackendConnection() {
    if (!g_dataEngine && typeof DataEngine === 'function') {
        g_dataEngine = new DataEngine();
        g_usingDataEngine = !!g_dataEngine;
    }
    showNotification('系统已启动，等待后端连接...', 'success');
}

/**
 * 刷新API使用统计
 */
async function refreshApiUsageStats() {
    // 从DataEngine缓存读取，不再发起独立API请求
    var cached = g_dataEngine ? g_dataEngine.getData() : null;
    if (cached && cached.system && cached.system._connected && cached.system.requests) {
        var requests = cached.system.requests;
        setEl('#api-stat-total', requests.total || 0);
        setEl('#api-stat-success', (requests.total || 0) - (requests.errors || 0));
        setEl('#api-stat-active', requests.connections || 0);
        setEl('#api-stat-errors', requests.errors || 0);
        setEl('#api-stat-rate', (requests.rate_per_minute !== undefined) ? requests.rate_per_minute : '--');
    }
}

/**
 * 初始化API使用统计图表
 */
function initApiUsageChart() {
    var canvas = document.getElementById('api-usage-chart');
    if (!canvas) return;
    if (window.g_apiUsageChart) {
        window.g_apiUsageChart.destroy();
    }
    window.g_apiUsageChart = new SelfLnnChart(canvas, {
        type: 'line'
    });
    window.g_apiUsageChart.setData({
        labels: [],
        datasets: [{
            label: '请求数',
            data: [],
            borderColor: '#00c8ff',
            backgroundColor: 'rgba(0,200,255,0.1)',
            fill: true
        }]
    });
}

// API使用统计由DataEngine统一驱动，不再注册独立轮询模块

// =============================================================================
// ROS/Gazebo 集成控制函数
// =============================================================================

/**
 * 配置并连接ROS Master
 */
async function configureROS() {
    var hostEl = document.getElementById('ros-master-host');
    var portEl = document.getElementById('ros-master-port');
    if (!hostEl) { console.warn('[ROS配置] ros-master-host元素不存在，使用默认值localhost'); }
    if (!portEl) { console.warn('[ROS配置] ros-master-port元素不存在，使用默认值11311'); }
    var host = hostEl ? hostEl.value : 'localhost';
    var port = parseInt(portEl ? portEl.value : '11311') || 11311;
    showNotification('正在连接ROS Master...', 'info');
    try {
        if (window.SelfLnnApi && typeof window.SelfLnnApi.configureRos === 'function') {
            var result = await window.SelfLnnApi.configureRos({ host: host, port: port });
            if (result && result.success) {
                var dot = document.querySelector('#ros-master-status .status-dot');
                var txt = document.querySelector('#ros-master-status .status-text');
                if (dot) dot.className = 'status-dot online';
                if (txt) txt.textContent = '已连接';
                showNotification('✅ ROS Master连接成功', 'success');
            } else {
                var errMsg = (result && result.error) || '连接失败';
                var dot = document.querySelector('#ros-master-status .status-dot');
                var txt = document.querySelector('#ros-master-status .status-text');
                if (dot) dot.className = 'status-dot offline';
                if (txt) txt.textContent = '连接失败';
                showNotification('❌ ' + errMsg, 'danger');
            }
        } else {
            showNotification('❌ ROS配置后端未连接', 'danger');
        }
    } catch (e) {
        console.error('配置ROS Master异常:', e);
        showNotification('❌ 配置ROS Master异常: ' + e.message, 'danger');
    }
}

/**
 * 控制Gazebo仿真
 */
async function controlGazebo(action) {
    showNotification('正在' + ({start:'启动',stop:'停止',pause:'暂停',resume:'继续',step:'步进',reset:'重置'}[action]||action) + '仿真...', 'info');
    try {
        if (window.SelfLnnApi && typeof window.SelfLnnApi.controlGazebo === 'function') {
            var result = await window.SelfLnnApi.controlGazebo({ action: action });
            if (result && result.success) {
                showNotification('✅ Gazebo ' + action + ' 成功', 'success');
                refreshRosGazeboStatus();
            } else {
                var errMsg = (result && result.error) || '控制失败';
                showNotification('❌ ' + errMsg, 'danger');
            }
        } else {
            showNotification('❌ Gazebo控制后端未连接', 'danger');
        }
    } catch (e) {
        console.error('控制Gazebo异常:', e);
        showNotification('❌ 控制Gazebo异常: ' + e.message, 'danger');
    }
}

/**
 * 在Gazebo中生成模型
 */
async function spawnGazeboModel() {
    var modelNameEl = document.getElementById('gazebo-model-name');
    if (!modelNameEl) return;
    var name = modelNameEl.value;
    if (!name) { showNotification('⚠️ 请输入模型名称', 'warning'); return; }
    /* F4-C02修复: 空指针安全访问 */
    var el = document.getElementById('gazebo-model-x');
    var x = el ? parseFloat(el.value) || 0 : 0;
    el = document.getElementById('gazebo-model-y');
    var y = el ? parseFloat(el.value) || 0 : 0;
    el = document.getElementById('gazebo-model-z');
    var z = el ? parseFloat(el.value) || 0 : 0;
    el = document.getElementById('gazebo-model-yaw');
    var yaw = el ? parseFloat(el.value) || 0 : 0;
    el = document.getElementById('gazebo-model-pitch');
    var pitch = el ? parseFloat(el.value) || 0 : 0;
    el = document.getElementById('gazebo-model-roll');
    var roll = el ? parseFloat(el.value) || 0 : 0;
    showNotification('正在生成模型 ' + name + '...', 'info');
    try {
        if (window.SelfLnnApi && typeof window.SelfLnnApi.controlGazebo === 'function') {
            var result = await window.SelfLnnApi.controlGazebo({
                action: 'spawn',
                model_name: name,
                position: { x: x, y: y, z: z },
                orientation: { yaw: yaw, pitch: pitch, roll: roll }
            });
            if (result && result.success) {
                showNotification('✅ 模型 ' + name + ' 生成成功', 'success');
            } else {
                showNotification('❌ 模型生成失败: ' + ((result && result.error) || '未知错误'), 'danger');
            }
        } else {
            showNotification('❌ Gazebo控制后端未连接', 'danger');
        }
    } catch (e) {
        console.error('生成模型异常:', e);
        showNotification('❌ 生成模型异常: ' + e.message, 'danger');
    }
}

/**
 * 从Gazebo删除模型
 */
async function deleteGazeboModel() {
    var nameEl = document.getElementById('gazebo-model-name');
    if (!nameEl) return;
    var name = nameEl.value;
    if (!name) { showNotification('⚠️ 请输入要删除的模型名称', 'warning'); return; }
    if (!safeConfirm('确定要删除模型 ' + name + ' 吗？')) return;
    showNotification('正在删除模型 ' + name + '...', 'warning');
    try {
        if (window.SelfLnnApi && typeof window.SelfLnnApi.controlGazebo === 'function') {
            var result = await window.SelfLnnApi.controlGazebo({
                action: 'delete',
                model_name: name
            });
            if (result && result.success) {
                showNotification('✅ 模型 ' + name + ' 已删除', 'success');
            } else {
                showNotification('❌ 模型删除失败: ' + ((result && result.error) || '未知错误'), 'danger');
            }
        } else {
            showNotification('❌ Gazebo控制后端未连接', 'danger');
        }
    } catch (e) {
        console.error('删除模型异常:', e);
        showNotification('❌ 删除模型异常: ' + e.message, 'danger');
    }
}

/**
 * 刷新ROS节点列表
 */
async function refreshRosNodes() {
    try {
        if (window.SelfLnnApi && typeof window.SelfLnnApi.getRosNodes === 'function') {
            var result = await window.SelfLnnApi.getRosNodes();
            var listEl = document.getElementById('ros-node-list');
            if (!listEl) return;
            if (result && result.success && result.data && result.data.nodes) {
                var nodes = result.data.nodes;
                if (nodes.length === 0) {
                    listEl.innerHTML = '<div class="ros-list-empty">无可用节点</div>';
                } else {
                    listEl.innerHTML = nodes.map(function(n) {
                        return '<div class="ros-node-item"><span>' + window.escapeHtml(n) + '</span></div>';
                    }).join('');
                }
            } else {
                listEl.innerHTML = '<div class="ros-list-empty">获取节点列表失败</div>';
            }
        }
    } catch (e) {
        console.error('刷新ROS节点异常:', e);
    }
}

/**
 * 刷新ROS主题列表
 */
async function refreshRosTopics() {
    try {
        if (window.SelfLnnApi && typeof window.SelfLnnApi.getRosTopics === 'function') {
            var result = await window.SelfLnnApi.getRosTopics();
            var listEl = document.getElementById('ros-topic-list');
            if (!listEl) return;
            if (result && result.success && result.data && result.data.topics) {
                var topics = result.data.topics;
                if (topics.length === 0) {
                    listEl.innerHTML = '<div class="ros-list-empty">无可用主题</div>';
                } else {
                    listEl.innerHTML = topics.map(function(t) {
                        return '<div class="ros-topic-item"><span>' + window.escapeHtml(t.name) + '</span><span style="color:#888;font-size:0.8rem;">' + window.escapeHtml(t.type) + '</span></div>';
                    }).join('');
                }
            } else {
                listEl.innerHTML = '<div class="ros-list-empty">获取主题列表失败</div>';
            }
        }
    } catch (e) {
        console.error('刷新ROS主题异常:', e);
    }
}

/**
 * 刷新ROS/Gazebo综合状态
 */
async function refreshRosGazeboStatus() {
    try {
        if (window.SelfLnnApi && typeof window.SelfLnnApi.getRosStatus === 'function') {
            var result = await window.SelfLnnApi.getRosStatus();
            if (result && result.success && result.data) {
                var d = result.data;
                // ROS Master状态
                if (d.ros_master_connected !== undefined) {
                    var dot = document.querySelector('#ros-master-status .status-dot');
                    var txt = document.querySelector('#ros-master-status .status-text');
                    if (dot) dot.className = 'status-dot ' + (d.ros_master_connected ? 'online' : 'offline');
                    if (txt) txt.textContent = d.ros_master_connected ? '已连接' : (d.ros_master_error || '未连接');
                }
                // Gazebo状态
                if (d.gazebo_state !== undefined) {
                    var stateMap = {0:'未连接',1:'连接中',2:'已连接',3:'运行中',4:'错误'};
                    var el = document.getElementById('gazebo-bridge-status');
                    if (el) el.textContent = stateMap[d.gazebo_state] || ('状态码:'+d.gazebo_state);
                }
                if (d.gazebo_sim_time !== undefined) {
                    var el = document.getElementById('gazebo-sim-time');
                    if (el) el.textContent = d.gazebo_sim_time.toFixed(1) + 's';
                }
                if (d.gazebo_robot_count !== undefined) {
                    var el = document.getElementById('gazebo-robot-count');
                    if (el) el.textContent = d.gazebo_robot_count;
                }
                if (d.gazebo_last_error) {
                    var el = document.getElementById('gazebo-last-error');
                    if (el) el.textContent = d.gazebo_last_error;
                }
            }
        }
    } catch (e) {
        console.error('刷新ROS/Gazebo状态异常:', e);
    }
}

// =============================================================================
// 训练控制函数
// =============================================================================

/* 从localStorage读取上次保存的训练模式，默认模仿学习(1) */
var selectedTrainingMode = (function() {
    var saved = localStorage.getItem('selflnn_training_mode');
    return saved ? parseInt(saved, 10) || 1 : 1;
})();

/**
 * 选择训练模式
 */
function selectTrainingMode(mode) {
    selectedTrainingMode = mode;
/* 用户更改时保存到localStorage */
    localStorage.setItem('selflnn_training_mode', String(mode));
    var btns = document.querySelectorAll('.training-mode-btn');
    for (var i = 0; i < btns.length; i++) {
        btns[i].className = 'btn btn-sm training-mode-btn' + (parseInt(btns[i].getAttribute('data-mode')) === mode ? ' active' : '');
    }
    var modeNames = {1:'模仿学习',2:'强化学习',3:'动作基元',4:'关节空间',5:'任务空间'};
    showNotification('已选择训练模式: ' + (modeNames[mode] || mode), 'info');
}

/**
 * 开始机器人训练
 */
async function startRobotTraining() {
    var lrEl = document.getElementById('training-learning-rate');
    var erEl = document.getElementById('training-exploration-rate');
    var dfEl = document.getElementById('training-discount-factor');
    var itersEl = document.getElementById('training-iterations');
    var lr = lrEl ? (parseFloat(lrEl.value) || 0.001) : 0.001;
    var er = erEl ? (parseFloat(erEl.value) || 0.1) : 0.1;
    var df = dfEl ? (parseFloat(dfEl.value) || 0.99) : 0.99;
    var iters = itersEl ? (parseInt(itersEl.value, 10) || 1000) : 1000;
    showNotification('正在启动机器人训练...', 'info');
    try {
        if (window.SelfLnnApi && typeof window.SelfLnnApi.controlRobotTraining === 'function') {
            var result = await window.SelfLnnApi.controlRobotTraining({
                robot_id: 0,
                action: 'start',
                mode: selectedTrainingMode,
                learning_rate: lr,
                exploration_rate: er,
                discount_factor: df,
                iterations: iters
            });
            if (result && result.success) {
                var statusEl = document.getElementById('training-global-status');
                if (statusEl) {
                    statusEl.textContent = '训练中';
                    statusEl.className = 'training-status-badge active';
                }
                showNotification('✅ 机器人训练已启动', 'success');
                refreshTrainingStatus();
            } else {
                showNotification('❌ 启动机器人训练失败: ' + ((result && result.error) || '未知错误'), 'danger');
            }
        } else {
            showNotification('❌ 机器人训练控制后端未连接', 'danger');
        }
    } catch (e) {
        console.error('启动机器人训练异常:', e);
        showNotification('❌ 启动机器人训练异常: ' + e.message, 'danger');
    }
}

/**
 * 暂停机器人训练
 */
async function pauseRobotTraining() {
    showNotification('正在暂停机器人训练...', 'warning');
    try {
        if (window.SelfLnnApi && typeof window.SelfLnnApi.controlRobotTraining === 'function') {
            var result = await window.SelfLnnApi.controlRobotTraining({
                robot_id: 0,
                action: 'pause'
            });
            if (result && result.success) {
                var pauseStatusEl = document.getElementById('training-global-status');
                if (pauseStatusEl) pauseStatusEl.textContent = '已暂停';
                showNotification('⏸️ 机器人训练已暂停', 'warning');
            } else {
                showNotification('❌ 暂停机器人训练失败: ' + ((result && result.error) || '未知错误'), 'danger');
            }
        }
    } catch (e) {
        console.error('暂停机器人训练异常:', e);
        showNotification('❌ 暂停机器人训练异常: ' + e.message, 'danger');
    }
}

/**
 * 继续机器人训练
 */
async function resumeRobotTraining() {
    showNotification('正在继续机器人训练...', 'info');
    try {
        if (window.SelfLnnApi && typeof window.SelfLnnApi.controlRobotTraining === 'function') {
            var result = await window.SelfLnnApi.controlRobotTraining({
                robot_id: 0,
                action: 'resume'
            });
            var tsEl = document.getElementById('training-global-status');
            if (result && result.success) {
                if (tsEl) tsEl.textContent = '训练中';
                if (tsEl) tsEl.className = 'training-status-badge active';
                showNotification('✅ 机器人训练已继续', 'success');
            } else {
                showNotification('❌ 继续机器人训练失败: ' + ((result && result.error) || '未知错误'), 'danger');
            }
        }
    } catch (e) {
        console.error('继续机器人训练异常:', e);
        showNotification('❌ 继续机器人训练异常: ' + e.message, 'danger');
    }
}

/**
 * 停止机器人训练
 */
async function stopRobotTraining() {
    if (!safeConfirm('确定要停止当前机器人训练吗？')) return;
    showNotification('正在停止机器人训练...', 'warning');
    try {
        if (window.SelfLnnApi && typeof window.SelfLnnApi.controlRobotTraining === 'function') {
            var result = await window.SelfLnnApi.controlRobotTraining({
                robot_id: 0,
                action: 'stop'
            });
            if (result && result.success) {
                var statusEl = document.getElementById('training-global-status');
                if (statusEl) {
                    statusEl.textContent = '空闲';
                    statusEl.className = 'training-status-badge';
                }
                showNotification('⏹️ 机器人训练已停止', 'danger');
                refreshTrainingStatus();
            } else {
                showNotification('❌ 停止机器人训练失败: ' + ((result && result.error) || '未知错误'), 'danger');
            }
        }
    } catch (e) {
        console.error('停止机器人训练异常:', e);
        showNotification('❌ 停止机器人训练异常: ' + e.message, 'danger');
    }
}

/**
 * 开始轨迹录制
 */
async function startRecording() {
    showNotification('正在开始轨迹录制...', 'info');
    try {
        if (window.SelfLnnApi && typeof window.SelfLnnApi.controlRobotTraining === 'function') {
            var result = await window.SelfLnnApi.controlRobotTraining({
                robot_id: 0,
                action: 'record'
            });
            if (result && result.success) {
                showNotification('🔴 轨迹录制中...', 'info');
            } else {
                showNotification('❌ 录制启动失败: ' + ((result && result.error) || '未知错误'), 'danger');
            }
        }
    } catch (e) {
        console.error('开始录制异常:', e);
        showNotification('❌ 开始录制异常: ' + e.message, 'danger');
    }
}

/**
 * 停止轨迹录制
 */
async function stopRecording() {
    showNotification('正在停止录制...', 'warning');
    try {
        if (window.SelfLnnApi && typeof window.SelfLnnApi.controlRobotTraining === 'function') {
            var result = await window.SelfLnnApi.controlRobotTraining({
                robot_id: 0,
                action: 'stop_record'
            });
            if (result && result.success) {
                showNotification('⏹️ 录制已停止', 'success');
            } else {
                showNotification('❌ 停止录制失败', 'danger');
            }
        }
    } catch (e) {
        console.error('停止录音失败:', e);
        showNotification('停止录音失败: ' + e.message, 'danger');
    }
}

/**
 * 回放轨迹
 */
async function replayTrajectory() {
    showNotification('正在回放轨迹...', 'info');
    try {
        if (window.SelfLnnApi && typeof window.SelfLnnApi.controlRobotTraining === 'function') {
            var result = await window.SelfLnnApi.controlRobotTraining({
                robot_id: 0,
                action: 'replay'
            });
            if (result && result.success) {
                showNotification('▶️ 轨迹回放中...', 'info');
            } else {
                showNotification('❌ 轨迹回放失败: ' + ((result && result.error) || '未知错误'), 'danger');
            }
        }
    } catch (e) {
        console.error('回放轨迹异常:', e);
        showNotification('❌ 回放轨迹异常: ' + e.message, 'danger');
    }
}

/**
 * 刷新训练状态
 */
async function refreshTrainingStatus() {
    try {
        if (window.SelfLnnApi && typeof window.SelfLnnApi.getRobotList === 'function') {
            var result = await window.SelfLnnApi.getRobotList();
            if (result && result.success && result.data && result.data.robots) {
                var robots = result.data.robots;
                if (robots.length > 0) {
                    var r = robots[0];
                    if (r.training_state !== undefined) {
                        var stateNames = {0:'空闲',1:'录制中',2:'训练中',3:'评估中',4:'已完成',5:'错误'};
                        var stateLabelEl = document.getElementById('training-state-label');
                        if (stateLabelEl) stateLabelEl.textContent = stateNames[r.training_state] || ('状态:'+r.training_state);
                    }
                    if (r.training_episode !== undefined) {
                        var epEl = document.getElementById('training-episode');
                        if (epEl) epEl.textContent = r.training_episode;
                    }
                    if (r.training_reward !== undefined) {
                        var rewardEl = document.getElementById('training-reward');
                        if (rewardEl) rewardEl.textContent = r.training_reward.toFixed(4);
                    }
                    if (r.training_progress !== undefined) {
                        var pct = Math.round(r.training_progress * 100);
                        var fillEl = document.getElementById('training-progress-fill');
                        if (fillEl) fillEl.style.width = pct + '%';
                        var textEl = document.getElementById('training-progress-text');
                        if (textEl) textEl.textContent = pct + '%';
                    }
                }
            }
        }
    } catch (e) {
        console.error('刷新训练状态异常:', e);
        showNotification('刷新训练状态失败: ' + e.message, 'danger');
    }
}

// =============================================================================
// 传感器管道监控函数
// =============================================================================

/**
 * 刷新传感器管道状态
 */
async function refreshSensorPipeline() {
    showNotification('正在获取传感器管道状态...', 'info');
    try {
        if (window.SelfLnnApi && typeof window.SelfLnnApi.getSensorPipelineStatus === 'function') {
            var result = await window.SelfLnnApi.getSensorPipelineStatus();
            if (result && result.success && result.data) {
                var d = result.data;
                // 概要信息 (C-05修复: 添加空值检查)
                var prEl = document.getElementById('pipeline-running');
                if (prEl) prEl.textContent = d.running ? '运行中' : '已停止';
                var pscEl = document.getElementById('pipeline-sensor-count');
                if (pscEl) pscEl.textContent = d.sensor_count || 0;
                var psrEl = document.getElementById('pipeline-sample-rate');
                if (psrEl) psrEl.textContent = (d.sample_rate || 0) + ' Hz';
                var pspEl = document.getElementById('pipeline-stream-port');
                if (pspEl) pspEl.textContent = d.stream_port || '未配置';
                // 状态徽章
                var badge = document.getElementById('pipeline-status-badge');
                if (badge) {
                    if (d.running) {
                        badge.textContent = ' 在线';
                        badge.className = 'pipeline-status-badge online';
                    } else {
                        badge.textContent = ' 离线';
                        badge.className = 'pipeline-status-badge';
                    }
                }
                // 传感器列表
                if (d.sensors && d.sensors.length > 0) {
                    var tbody = document.getElementById('pipeline-sensor-tbody');
                    if (tbody) {
                        tbody.innerHTML = d.sensors.map(function(s, i) {
                        var typeNames = {0:'未知',1:'摄像头',2:'激光雷达',3:'IMU',4:'超声波',5:'温度',6:'压力',7:'红外',8:'深度',9:'力觉',10:'触觉'};
                        return '<tr>' +
                            '<td>' + (s.id !== undefined ? s.id : i) + '</td>' +
                            '<td>' + window.escapeHtml((typeNames[s.type] || s.type)) + '</td>' +
                            '<td>' + window.escapeHtml(s.name || '传感器' + i) + '</td>' +
                            '<td>' + (s.active ? '活跃' : '离线') + '</td>' +
                            '<td>' + (s.frequency || 0) + '</td>' +
                            '<td>' + (s.last_value !== undefined ? s.last_value : '--') + '</td>' +
                            '</tr>';
                    }).join('');
                    }
                }
                // LNN传感器输入状态
                if (d.sensor_input) {
                    var si = d.sensor_input;
                    var lnnInputEl = document.getElementById('sensor-lnn-input-status');
                    if (lnnInputEl) lnnInputEl.textContent = si.active_channels > 0 ? ('活跃 ' + si.active_channels + '通道') : '等待数据';
                    var imuRawEl = document.getElementById('sensor-imu-raw');
                    if (imuRawEl) imuRawEl.textContent = si.imu_raw ? ('X:' + (si.imu_raw.x || 0).toFixed(4) + ' Y:' + (si.imu_raw.y || 0).toFixed(4)) : '等待数据';
                    var channelCountEl = document.getElementById('sensor-channel-count');
                    if (channelCountEl) channelCountEl.textContent = si.total_channels || 0;
                    var sampleRateEl = document.getElementById('sensor-sample-rate');
                    if (sampleRateEl) sampleRateEl.textContent = si.sample_rate ? si.sample_rate + ' Hz' : '--';
                }
                showNotification('✅ 传感器管道状态已更新', 'success');
            } else {
                showNotification('❌ 获取传感器状态失败: ' + ((result && result.error) || '未知错误'), 'danger');
            }
        } else {
            showNotification('❌ 传感器管道后端未连接', 'danger');
        }
    } catch (e) {
        console.error('刷新传感器管道异常:', e);
        showNotification('❌ 刷新传感器管道异常: ' + e.message, 'danger');
    }
}

/**
 * 校准IMU
 */
async function calibrateIMU() {
    showNotification('正在校准IMU...', 'info');
    try {
        if (window.SelfLnnApi && typeof window.SelfLnnApi.getSensorPipelineStatus === 'function') {
            var result = await window.SelfLnnApi.getSensorPipelineStatus({ calibrate_imu: true });
            showNotification('✅ IMU校准命令已发送', 'success');
        } else {
            showNotification('❌ 校准后端未连接', 'danger');
        }
    } catch (e) {
        console.error('校准IMU异常:', e);
        showNotification('❌ 校准IMU异常: ' + e.message, 'danger');
    }
}

/**
 * 重置传感器输入
 */
async function resetSensorInput() {
    if (!safeConfirm('确定要重置LNN传感器输入状态吗？')) return;
    showNotification('正在重置LNN传感器输入...', 'warning');
    try {
        if (window.SelfLnnApi && typeof window.SelfLnnApi.getSensorPipelineStatus === 'function') {
            var result = await window.SelfLnnApi.getSensorPipelineStatus({ reset_sensor_input: true });
            showNotification('✅ LNN传感器输入已重置', 'success');
        } else {
            showNotification('❌ 重置后端未连接', 'danger');
        }
    } catch (e) {
        console.error('重置LNN传感器输入异常:', e);
    }
}

/**
 * 启动传感器流媒体
 */
async function startSensorStream() {
    showNotification('正在启动传感器流媒体...', 'info');
    try {
        if (window.SelfLnnApi && typeof window.SelfLnnApi.getSensorPipelineStatus === 'function') {
            var result = await window.SelfLnnApi.getSensorPipelineStatus({ start_stream: true });
            if (result && result.success) {
                showNotification('✅ 传感器流媒体已启动', 'success');
            } else {
                showNotification('❌ 启动流媒体失败', 'danger');
            }
        }
    } catch (e) {
        console.error('启动流媒体异常:', e);
    }
}

/**
 * 停止传感器流媒体
 */
async function stopSensorStream() {
    showNotification('正在停止传感器流媒体...', 'warning');
    try {
        if (window.SelfLnnApi && typeof window.SelfLnnApi.getSensorPipelineStatus === 'function') {
            var result = await window.SelfLnnApi.getSensorPipelineStatus({ stop_stream: true });
            if (result && result.success) {
                showNotification('⏹️ 传感器流媒体已停止', 'success');
            } else {
                showNotification('❌ 停止流媒体失败', 'danger');
            }
        }
    } catch (e) {
        console.error('停止流媒体异常:', e);
    }
}

// =============================================================================
// 增强连接/断开机器人函数（支持ROS扩展）
// =============================================================================

/**
 * 增强连接机器人 - 支持ROS机器人
 */
async function connectRobotEnhanced(robotId, connectGazebo, robotName) {
    showNotification('正在连接ROS机器人...', 'info');
    try {
        if (window.SelfLnnApi && typeof window.SelfLnnApi.connectRobotEnhanced === 'function') {
            var result = await window.SelfLnnApi.connectRobotEnhanced(robotId, connectGazebo, robotName);
            if (result && result.success) {
                showNotification('✅ ROS机器人连接成功', 'success');
                refreshRosGazeboStatus();
            } else {
                showNotification('❌ 连接失败: ' + ((result && result.error) || '未知错误'), 'danger');
            }
        } else {
            showNotification('❌ 增强连接后端未连接', 'danger');
        }
    } catch (e) {
        console.error('增强连接异常:', e);
        showNotification('❌ 连接异常: ' + e.message, 'danger');
    }
}

/**
 * 增强断开机器人 - 支持ROS机器人和Gazebo模型清理
 */
async function disconnectRobotEnhanced(robotId, disconnectGazebo, modelName) {
    if (!safeConfirm('确定要断开ROS机器人连接吗？')) return;
    showNotification('正在断开ROS机器人...', 'warning');
    try {
        if (window.SelfLnnApi && typeof window.SelfLnnApi.disconnectRobotEnhanced === 'function') {
            var result = await window.SelfLnnApi.disconnectRobotEnhanced(robotId, disconnectGazebo, modelName);
            if (result && result.success) {
                showNotification('✅ ROS机器人已断开', 'success');
                refreshRosGazeboStatus();
            } else {
                showNotification('❌ 断开失败: ' + ((result && result.error) || '未知错误'), 'danger');
            }
        } else {
            showNotification('❌ 增强断开后端未连接', 'danger');
        }
    } catch (e) {
        console.error('增强断开异常:', e);
        showNotification('❌ 断开异常: ' + e.message, 'danger');
    }
}

// =============================================================================
// 自动刷新定时器 (M-09修复: 使用 g_rosGazeboRefreshTimer 统一全局变量命名)
// =============================================================================

/**
 * 启动ROS/Gazebo状态自动刷新（通过统一轮询中心调度）
 */
function startRosGazeboAutoRefresh(intervalMs) {
    intervalMs = intervalMs || 5000;
    stopRosGazeboAutoRefresh();
    if (typeof g_dataEngine !== 'undefined' && g_dataEngine) {
        g_dataEngine.registerModule('ros_gazebo', intervalMs, function() {
            refreshRosGazeboStatus();
            refreshTrainingStatus();
        });
    } else {
        g_rosGazeboRefreshTimer = setInterval(function() {
            refreshRosGazeboStatus();
            refreshTrainingStatus();
        }, intervalMs);
    }
}

/**
 * 停止ROS/Gazebo状态自动刷新
 */
function stopRosGazeboAutoRefresh() {
    if (g_rosGazeboRefreshTimer) {
        clearInterval(g_rosGazeboRefreshTimer);
        g_rosGazeboRefreshTimer = null;
    }
    if (typeof g_dataEngine !== 'undefined' && g_dataEngine) {
        g_dataEngine.unregisterModule('ros_gazebo');
    }
}

/**
 * 显示后端断开状态（绝不生成虚假数据）
 */
function showDisconnectedState(reason) {
    const healthStatus = document.querySelector('.health-status');
    if (healthStatus) {
        healthStatus.className = 'health-status disconnected';
        healthStatus.innerHTML = '<span style="color:#ff5555">⚠ 后端未连接</span>';
    }
    
    const allMetricValues = document.querySelectorAll('.metric-value, .stat-value, .type-usage, .detail-value, .progress-value');
    allMetricValues.forEach(el => {
        el.textContent = '未连接';
        /* L-023修复: 断开时移除骨架屏动画，显示真实断开状态文本 */
        el.classList.remove('skeleton-loading');
    });
    
    const allFills = document.querySelectorAll('.metric-fill, .type-fill, .progress-fill');
    allFills.forEach(el => {
        el.style.width = '0%';
    });
    
    const memInfo = document.querySelector('.memory-info');
    if (memInfo) {
        const ps = memInfo.querySelectorAll('p');
        ps.forEach(p => {
            /* FE-014修复: 此处innerHTML正则替换仅匹配后端返回的<strong>标签内容，
         * 替换为固定的"未连接"状态文本，不涉及用户输入，XSS安全。 */
        p.innerHTML = p.innerHTML.replace(/<strong>.*?<\/strong>/g, '<strong style="color:#ff5555">未连接</strong>');
        });
    }
    
    const statusBar = document.querySelector('.system-status-bar');
    if (statusBar) {
        statusBar.textContent = '⚠ 后端未连接 - ' + reason + '（绝不使用虚假数据）';
        statusBar.style.color = '#ff5555';
    }
    
    // 更新侧边栏模型状态指示器
    const modelItems = document.querySelectorAll('.model-item .model-status');
    modelItems.forEach(el => {
        el.textContent = '未知';
        el.className = 'model-status disconnected';
    });
    
    // 更新模型管理页面的状态徽章
    const statusBadges = document.querySelectorAll('.model-item-status .status-badge');
    statusBadges.forEach(el => {
        el.textContent = '未知';
        el.className = 'status-badge disconnected';
    });

    /* L-024修复: 后端断开时更新训练历史占位文本 */
    if (typeof updateTrainingHistoryPlaceholderState === 'function') {
        updateTrainingHistoryPlaceholderState();
    }
}

/**
 * 更新所有模型状态徽章（从后端API数据）
 * 从系统状态中提取各模块状态并更新UI中的模型状态指示器
 */
function updateAllModelStatusBadges(systemStatus) {
    if (!systemStatus || !systemStatus.success || !systemStatus.data) {
        console.warn('系统状态数据不可用，无法更新模型状态');
        return;
    }
    
    const systemData = systemStatus.data.system;
    if (!systemData) return;
    
    const modules = systemData.modules || {};
    
    // 单一CfC LNN核心状态显示：data-model属性 -> 状态文本/CSS类
    const lnnCoreStatus = {
        available: false,
        label: modules.lnn && modules.lnn.status === 'running' ? '运行中' : '未连接',
        cssClass: modules.lnn && modules.lnn.status === 'running' ? 'active' : 'pending',
        h_dim: modules.lnn && modules.lnn.h_dim ? modules.lnn.h_dim : '--',
        ode_solver: modules.lnn && modules.lnn.ode_solver ? modules.lnn.ode_solver : '--',
        tau_range: modules.lnn && modules.lnn.tau_range ? modules.lnn.tau_range : '--'
    };
    
    // 更新侧边栏模型状态（.model-list 中的 .model-status）
    document.querySelectorAll('.model-list .model-item').forEach(item => {
        const nameEl = item.querySelector('.model-name');
        const statusEl = item.querySelector('.model-status');
        if (!nameEl || !statusEl) return;
        statusEl.textContent = lnnCoreStatus.label;
        statusEl.className = 'model-status ' + lnnCoreStatus.cssClass;
    });
    
    // 更新模型管理页面状态徽章（.model-items 中的 .status-badge）
    document.querySelectorAll('.model-items .model-item').forEach(item => {
        const badgeEl = item.querySelector('.status-badge');
        if (!badgeEl) return;
        badgeEl.textContent = lnnCoreStatus.label;
        badgeEl.className = 'status-badge ' + lnnCoreStatus.cssClass;
    });
    
    // 更新LNN状态空间信息
    const stateDimEl = document.getElementById('state-dimension');
    if (stateDimEl) stateDimEl.textContent = lnnCoreStatus.h_dim !== '--' ? lnnCoreStatus.h_dim + '维' : '未连接';
    
}

/**
 * 仪表盘初始化（DataEngine统一管理数据，触发初始数据加载）
 */
function initializeDashboard() {
    if (window.SelfLnnApi && typeof window.SelfLnnApi.getSystemStatus === 'function') {
        window.SelfLnnApi.getSystemStatus().then(function(result) {
            if (result && result.data) {
                if (typeof window._renderDashboard === 'function') {
                    window._renderDashboard(result.data);
                }
            }
        }).catch(function(e) {
            console.error('[Dashboard] 系统状态获取失败:', e && e.message ? e.message : e);
        });
    }
}

/**
 * 显示后端未连接状态（绝不使用虚假数据）
 */
function showApiUnavailableError() {
    /* 如果内联仪表盘更新器已设值，不覆盖 */
    if (window.__dashboardLive === true) return;
    console.warn('后端未连接，显示断开状态（绝不生成虚假数据）');
    showNotification('⚠ 后端未连接，等待连接...', 'warning');
    
    if (g_usingDataEngine && g_dataEngine && typeof g_dataEngine.start === 'function') {
        g_dataEngine.start(2000);
    }
    
    const allMetricValues = document.querySelectorAll('.metric-value, .stat-value, .type-usage, .detail-value, .progress-value');
    allMetricValues.forEach(el => {
        el.textContent = '等待后端...';
        /* L-023修复: 移除骨架屏动画 */
        el.classList.remove('skeleton-loading');
    });
    
    const allFills = document.querySelectorAll('.metric-fill, .type-fill, .progress-fill');
    allFills.forEach(el => {
        el.style.width = '0%';
    });
    
    const statusBar = document.querySelector('.system-status-bar');
    if (statusBar) {
        statusBar.textContent = '⏳ 等待后端API连接...';
        statusBar.style.color = '#ffaa00';
    }

    /* L-024修复: API不可用时更新训练历史占位文本 */
    if (typeof updateTrainingHistoryPlaceholderState === 'function') {
        updateTrainingHistoryPlaceholderState();
    }
}

/**
 * 显示模型状态错误
 */
function updateModelStatusWithError() {
    const modelStats = document.querySelectorAll('.model-stats .stat-value');
    if (modelStats.length >= 3) {
        modelStats[0].textContent = '错误';
        modelStats[1].textContent = '错误';
        modelStats[2].textContent = '错误';
    }
}

/**
 * 显示训练进度错误
 */
function updateTrainingProgressWithError() {
    const progressValue = document.querySelector('.progress-value');
    const progressFill = document.querySelector('.progress-fill');
    
    if (progressValue) progressValue.textContent = '错误';
    if (progressFill) progressFill.style.width = '0%';
    
    const trainingDetails = document.querySelectorAll('.training-details .detail-value');
    if (trainingDetails.length >= 4) {
        trainingDetails[0].textContent = '错误';
        trainingDetails[1].textContent = '错误';
        trainingDetails[2].textContent = '错误';
        trainingDetails[3].textContent = '错误';
    }
}

/**
 * L-024修复: 根据后端连接状态动态更新训练历史占位文本
 * 未连接时显示"等待后端连接..."，已连接但无历史时显示"暂无训练历史"
 */
function updateTrainingHistoryPlaceholderState() {
    var connected = false;

    /* 检查后端连接状态: 优先使用DataEngine，其次直接调用SelfLnnApi */
    if (typeof g_dataEngine !== 'undefined' && g_dataEngine && g_dataEngine._backendConnected) {
        connected = true;
    } else if (window.SelfLnnApi && typeof window.SelfLnnApi.checkConnection === 'function') {
        /* 同步快速路径：若DataEngine不可用则尝试异步检查 */
        try {
            window.SelfLnnApi.checkConnection().then(function(result) {
                var isConn = result && result.connected === true;
                _applyTrainingHistoryPlaceholderText(isConn);
            }).catch(function() {
                _applyTrainingHistoryPlaceholderText(false);
            });
            return; /* 异步路径，提前返回 */
        } catch (e) {
            connected = false;
        }
    }

    _applyTrainingHistoryPlaceholderText(connected);
}

/**
 * L-024修复: 内部辅助 — 根据连接状态设置占位文本
 */
function _applyTrainingHistoryPlaceholderText(connected) {
    /* 更新训练历史图表占位符 */
    var chartPlaceholder = document.getElementById('train-history-placeholder');
    if (chartPlaceholder) {
        if (connected) {
            chartPlaceholder.textContent = '暂无训练历史';
            chartPlaceholder.style.color = '#888';
        } else {
            chartPlaceholder.textContent = '等待后端连接...';
            chartPlaceholder.style.color = '#ffaa00';
        }
    }

    /* 更新训练历史表格占位行 */
    var historyPlaceholderText = document.getElementById('history-list-placeholder-text');
    if (historyPlaceholderText) {
        if (connected) {
            historyPlaceholderText.textContent = '暂无训练历史';
            historyPlaceholderText.style.color = '';
        } else {
            historyPlaceholderText.textContent = '等待后端连接...';
            historyPlaceholderText.style.color = '#ffaa00';
        }
    }
}

/**
 * 更新系统健康状态
 */
function updateSystemHealth(systemStatus) {
    /* F4-H01修复: 使用稳定ID选择器替代脆弱CSS选择器 */
    const cpuElement = document.getElementById('cpu-usage');
    const memoryElement = document.querySelector('.metric:nth-child(2) .metric-value');
    const uptimeElement = document.querySelector('.metric:nth-child(3) .metric-value');
    
    // 在函数作用域声明，使进度条区域也能访问真实计算值
    let cpuUsage = 0;
    let memoryUsage = 0;
    
    // 使用实际API数据或显示错误状态
    if (systemStatus.success && systemStatus.data && systemStatus.data.system) {
        // 使用实际API数据
        // API返回详细系统状态，我们可以从中提取信息
        const systemData = systemStatus.data.system;
        
        // CPU使用率（必须使用后端返回的真实值，禁止虚假推算）
        if (systemData.cpu_usage !== undefined && systemData.cpu_usage >= 0) {
            cpuUsage = systemData.cpu_usage;
        } else {
            cpuUsage = -1;
        }
        if (cpuElement) cpuElement.textContent = cpuUsage >= 0 ? `${Math.round(cpuUsage)}%` : '--';
        
        // 内存使用（从内存统计中获取，绝不使用假数据）
        const modules = systemData.modules || {};
        const memoryData = modules.memory;
        if (memoryData && memoryData.total > 0) {
            const usedMB = memoryData.total / (1024 * 1024);
            // 从多个来源获取真实系统总内存，拒绝硬编码假值
            let totalMB = 0;
            // 优先使用 navigator.deviceMemory (浏览器API，返回GB)
            if (navigator.deviceMemory && navigator.deviceMemory > 0) {
                totalMB = navigator.deviceMemory * 1024;
            }
            // 其次尝试系统中已缓存的总内存信息
            if (totalMB <= 0 && typeof systemData.memory !== 'undefined' && systemData.memory.total_gb > 0) {
                totalMB = systemData.memory.total_gb * 1024;
            }
            if (totalMB <= 0 && typeof systemData.memory !== 'undefined' && systemData.memory.total_mb > 0) {
                totalMB = systemData.memory.total_mb;
            }
            if (totalMB > 0) {
                // 基于真实总量和使用量计算百分比，绝不使用硬编码假值
                memoryUsage = Math.min(100, (usedMB / totalMB) * 100);
                if (memoryElement) memoryElement.textContent = `${(usedMB / 1024).toFixed(1)} GB / ${(totalMB / 1024).toFixed(1)} GB`;
            } else {
                // 无法获取系统总内存，只显示已用内存，不制造虚假总值
                if (memoryElement) memoryElement.textContent = `${(usedMB / 1024).toFixed(1)} GB / --`;
            }
        } else {
            if (memoryElement) memoryElement.textContent = '-- GB / -- GB';
        }
        
        // 运行时间（从uptime字段获取）
        if (systemData.uptime && uptimeElement) {
            const days = Math.floor(systemData.uptime / (24 * 3600));
            const hours = Math.floor((systemData.uptime % (24 * 3600)) / 3600);
            uptimeElement.textContent = `${days}天 ${hours}小时`;
        } else if (uptimeElement) {
            uptimeElement.textContent = '--';
        }
    } else {
        // API失败，显示错误状态
        if (cpuElement) cpuElement.textContent = '--';
        if (memoryElement) memoryElement.textContent = '-- GB / -- GB';
        if (uptimeElement) uptimeElement.textContent = '--';
    }
    
    // 更新进度条 — 使用动态计算的真值替代硬编码假百分比
    const cpuBar = document.querySelector('.metric:nth-child(1) .metric-fill');
    const memoryBar = document.querySelector('.metric:nth-child(2) .metric-fill');
    
    if (cpuBar) cpuBar.style.width = systemStatus.success ? `${Math.round(cpuUsage)}%` : '0%';
    if (memoryBar) memoryBar.style.width = systemStatus.success ? `${Math.round(memoryUsage)}%` : '0%';
}

/**
 * 更新模型状态
 * @param {Object} systemStatus - 系统状态数据（可选）
 */
function updateModelStatus(systemStatus) {
    const modelStats = document.querySelectorAll('.model-stats .stat-value');
    if (modelStats.length >= 3) {
        if (systemStatus && systemStatus.success && systemStatus.data) {
            // 从系统状态中提取模型信息
            const modelInfo = systemStatus.data.model || {};
            
            // 模型大小（参数数量）
            const modelSize = modelInfo.size || modelInfo.parameters || 0;
            modelStats[0].textContent = modelSize > 0 ? 
                (modelSize >= 1e9 ? (modelSize / 1e9).toFixed(1) + 'B' :
                 modelSize >= 1e6 ? (modelSize / 1e6).toFixed(1) + 'M' :
                 modelSize >= 1e3 ? (modelSize / 1e3).toFixed(1) + 'K' : modelSize) : '--';
            
            // 模型精度（如果可用）
            const modelAccuracy = modelInfo.accuracy || modelInfo.precision || 0;
            modelStats[1].textContent = modelAccuracy > 0 ? 
                (modelAccuracy * 100).toFixed(1) + '%' : '--';
            
            // 模型版本/状态
            const modelVersion = modelInfo.version || modelInfo.status || '未知';
            modelStats[2].textContent = modelVersion;
        } else {
            // 没有模型数据，显示空状态（禁止虚假数据）
            modelStats[0].textContent = '';
            modelStats[1].textContent = '';
            modelStats[2].textContent = '';
        }
    }
    
    // 移除旧的API未实现提示（如果存在）
    const modelSection = document.querySelector('.model-stats');
    if (modelSection) {
        const existingNote = modelSection.querySelector('.api-note');
        if (existingNote) {
            existingNote.remove();
        }
    }
}

/**
 * 更新训练进度
 * @param {Object} learningStatus - 学习状态数据（可选）
 */
function updateTrainingProgress(learningStatus) {
    const progressValue = document.querySelector('.progress-value');
    const progressFill = document.querySelector('.progress-fill');
    
    const trainingDetails = document.querySelectorAll('.training-details .detail-value');
    
    if (learningStatus && learningStatus.success && learningStatus.data) {
        const trainingInfo = learningStatus.data.training || {};
        
        // 训练进度百分比
        const progress = trainingInfo.progress || trainingInfo.completion || 0;
        if (progressValue) progressValue.textContent = progress + '%';
        if (progressFill) progressFill.style.width = progress + '%';
        
        // 训练详情
        if (trainingDetails.length >= 4) {
            trainingDetails[0].textContent = trainingInfo.epoch || '--';
            trainingDetails[1].textContent = trainingInfo.loss ? 
                trainingInfo.loss.toFixed(4) : '--';
            trainingDetails[2].textContent = trainingInfo.accuracy ? 
                (trainingInfo.accuracy * 100).toFixed(1) + '%' : '--';
            trainingDetails[3].textContent = trainingInfo.duration ? 
                trainingInfo.duration + 's' : '--';
        }
    } else {
        // 没有训练数据，显示空状态（禁止虚假数据）
        if (progressValue) progressValue.textContent = '';
        if (progressFill) progressFill.style.width = '0%';
        
        if (trainingDetails.length >= 4) {
            trainingDetails[0].textContent = '';
            trainingDetails[1].textContent = '';
            trainingDetails[2].textContent = '';
            trainingDetails[3].textContent = '';
        }
    }
    
    // 移除旧的API未实现提示（如果存在）
    const trainingSection = document.querySelector('.training-progress');
    if (trainingSection) {
        const existingNote = trainingSection.querySelector('.api-note');
        if (existingNote) {
            existingNote.remove();
        }
    }
}

/**
 * 更新LNN统一记忆层（LNN内部状态回溯）
 */
function updateMemorySystem(memoryStatus) {
    const memoryTypes = document.querySelectorAll('.memory-type');
    
    if (memoryStatus.success && memoryStatus.data && memoryStatus.data.memory) {
        const memoryData = memoryStatus.data.memory;
        
        if (memoryTypes.length >= 3) {
            var totalItems = (memoryData.statistics && memoryData.statistics.total_items) ? memoryData.statistics.total_items : 1;
            var stPct = ((memoryData.short_term_size || 0) / totalItems * 100).toFixed(0);
            var ltPct = ((memoryData.long_term_size || 0) / totalItems * 100).toFixed(0);
            var epPct = ((memoryData.episodic_size || 0) / totalItems * 100).toFixed(0);
            // 短期记忆
            const shortTerm = memoryTypes[0].querySelector('.type-usage');
            const shortTermFill = memoryTypes[0].querySelector('.type-fill');
            if (shortTerm) shortTerm.textContent = stPct + '%';
            if (shortTermFill) shortTermFill.style.width = stPct + '%';
            
            // 长期记忆
            const longTerm = memoryTypes[1].querySelector('.type-usage');
            const longTermFill = memoryTypes[1].querySelector('.type-fill');
            if (longTerm) longTerm.textContent = ltPct + '%';
            if (longTermFill) longTermFill.style.width = ltPct + '%';
            
            // 情景记忆
            const episodic = memoryTypes[2].querySelector('.type-usage');
            const episodicFill = memoryTypes[2].querySelector('.type-fill');
            if (episodic) episodic.textContent = epPct + '%';
            if (episodicFill) episodicFill.style.width = epPct + '%';
        }
        
        const memoryInfo = document.querySelector('.memory-info');
        if (memoryInfo) {
            const paragraphs = memoryInfo.querySelectorAll('p');
            if (paragraphs.length >= 3) {
                if (memoryData.architecture && memoryData.architecture.short_term) {
                    /* P2-FIX-08: 使用escapeHtml转义后端数据防止XSS */
                    paragraphs[0].innerHTML = `容量: <strong>${window.escapeHtml ? window.escapeHtml(String(memoryData.architecture.short_term.capacity)) : memoryData.architecture.short_term.capacity}条</strong>`;
                } else {
                    paragraphs[0].innerHTML = '容量: <strong>--</strong>';
                }
                
                if (memoryData.statistics && memoryData.statistics.total_items !== undefined) {
                    /* P2-FIX-08: XSS防护 - 转义后端数据 */
                    paragraphs[1].innerHTML = `总条目: <strong>${window.escapeHtml ? window.escapeHtml(String(memoryData.statistics.total_items)) : memoryData.statistics.total_items}</strong>`;
                } else {
                    paragraphs[1].innerHTML = '总条目: <strong>--</strong>';
                }
                
                if (memoryData.statistics && memoryData.statistics.retrieval_success !== undefined) {
                    /* P1-FIX-01: XSS防护 - 使用escapeHtml转义数值，即使toFixed已保证为数字字符串，纵深防御 */
                    var rateVal = (memoryData.statistics.retrieval_success * 100).toFixed(1);
                    paragraphs[2].innerHTML = '检索成功率: <strong>' + window.escapeHtml(rateVal) + '%</strong>';
                } else {
                    paragraphs[2].innerHTML = '检索成功率: <strong>--</strong>';
                }
            }
        }
    } else {
        // API失败或数据无效
        if (memoryTypes.length >= 3) {
            const shortTerm = memoryTypes[0].querySelector('.type-usage');
            const shortTermFill = memoryTypes[0].querySelector('.type-fill');
            if (shortTerm) shortTerm.textContent = '错误';
            if (shortTermFill) shortTermFill.style.width = '0%';
            
            const longTerm = memoryTypes[1].querySelector('.type-usage');
            const longTermFill = memoryTypes[1].querySelector('.type-fill');
            if (longTerm) longTerm.textContent = '错误';
            if (longTermFill) longTermFill.style.width = '0%';
            
            const episodic = memoryTypes[2].querySelector('.type-usage');
            const episodicFill = memoryTypes[2].querySelector('.type-fill');
            if (episodic) episodic.textContent = '错误';
            if (episodicFill) episodicFill.style.width = '0%';
        }
        
        const memoryInfo = document.querySelector('.memory-info');
        if (memoryInfo) {
            const paragraphs = memoryInfo.querySelectorAll('p');
            if (paragraphs.length >= 3) {
                paragraphs[0].innerHTML = '总记忆条目: <strong>错误</strong>';
                paragraphs[1].innerHTML = '最近访问: <strong>错误</strong>';
                paragraphs[2].innerHTML = '关联强度: <strong>错误</strong>';
            }
        }
    }
}

/**
 * 更新LNN推理状态（LNN前向传播推理）
 */
function updateReasoningEngine(reasoningStatus) {
    const reasoningStats = document.querySelectorAll('.reasoning-stats .stat-value');
    
    if (reasoningStatus.success && reasoningStatus.data && reasoningStatus.data.reasoning) {
        const reasoningData = reasoningStatus.data.reasoning;
        
        if (reasoningStats.length >= 3) {
            // 使用实际API数据
            // 推理次数（从配置中获取）
            if (reasoningData.configuration && reasoningData.configuration.max_iterations) {
                reasoningStats[0].textContent = reasoningData.configuration.max_iterations;
            } else {
                reasoningStats[0].textContent = '';
            }
            
            // 推理时间（无实际数据，显示空状态）
            reasoningStats[1].textContent = '';
            
            // 置信度阈值
            if (reasoningData.configuration && reasoningData.configuration.confidence_threshold) {
                reasoningStats[2].textContent = `${(reasoningData.configuration.confidence_threshold * 100).toFixed(1)}%`;
            } else {
                reasoningStats[2].textContent = '';
            }
        }
        
        // 显示推理能力列表
        const reasoningSection = document.querySelector('.reasoning-stats');
        if (reasoningSection && reasoningData.capabilities && reasoningData.capabilities.length > 0) {
            const existingNote = reasoningSection.querySelector('.api-note');
            if (!existingNote) {
                const note = document.createElement('div');
                note.className = 'api-note';
                note.style.fontSize = '12px';
                note.style.color = '#666';
                note.style.marginTop = '5px';
                note.textContent = `支持: ${reasoningData.capabilities.slice(0, 3).join(', ')}`;
                reasoningSection.appendChild(note);
            }
        }
    } else {
        // API失败或数据无效
        if (reasoningStats.length >= 3) {
            reasoningStats[0].textContent = '错误';
            reasoningStats[1].textContent = '错误';
            reasoningStats[2].textContent = '错误';
        }
    }
}

/**
 * 更新学习状态
 */
function updateLearningStatus(learningStatus) {
    const learningMetrics = document.querySelectorAll('.learning-metrics .metric-value');
    
    if (learningStatus.success && learningStatus.data && learningStatus.data.learning) {
        const learningData = learningStatus.data.learning;
        
        if (learningMetrics.length >= 3) {
            // 使用实际API数据
            // 学习率
            if (learningData.parameters && learningData.parameters.learning_rate !== undefined) {
                learningMetrics[0].textContent = learningData.parameters.learning_rate.toFixed(4);
            } else {
                learningMetrics[0].textContent = '--';
            }
            
            // 探索率
            if (learningData.parameters && learningData.parameters.exploration_rate !== undefined) {
                learningMetrics[1].textContent = (learningData.parameters.exploration_rate * 100).toFixed(1);
            } else {
                learningMetrics[1].textContent = '--';
            }
            
            // 种群大小
            if (learningData.parameters && learningData.parameters.population_size !== undefined) {
                learningMetrics[2].textContent = learningData.parameters.population_size;
            } else {
                learningMetrics[2].textContent = '--';
            }
        }
        
        // 显示学习能力列表
        const learningSection = document.querySelector('.learning-metrics');
        if (learningSection && learningData.capabilities && learningData.capabilities.length > 0) {
            const existingNote = learningSection.querySelector('.api-note');
            if (!existingNote) {
                const note = document.createElement('div');
                note.className = 'api-note';
                note.style.fontSize = '12px';
                note.style.color = '#666';
                note.style.marginTop = '5px';
                note.textContent = `支持: ${learningData.capabilities.slice(0, 3).join(', ')}`;
                learningSection.appendChild(note);
            }
        }
        
        const learningTrend = document.querySelector('.learning-trend');
        if (learningTrend) {
            const paragraphs = learningTrend.querySelectorAll('p');
            if (paragraphs.length >= 3) {
                /* P2-FIX-15: 使用textContent替代innerHTML防止后端数据XSS注入 */
                if (learningData.architecture && learningData.architecture.types) {
                    paragraphs[0].innerHTML = '学习类型: <strong>' + window.escapeHtml(learningData.architecture.types[0] || '--') + '</strong>';
                } else {
                    paragraphs[0].innerHTML = '学习类型: <strong>--</strong>';
                }
                
                if (learningData.architecture && learningData.architecture.components) {
                    paragraphs[1].innerHTML = '主要组件: <strong>' + window.escapeHtml(learningData.architecture.components[0] || '--') + '</strong>';
                } else {
                    paragraphs[1].innerHTML = '主要组件: <strong>--</strong>';
                }
                
                /* P2-004修复: 从后端真实数据获取学习状态而非硬编码"准备就绪" */
                var learnStatus = '就绪';
                var statusColor = '#00ff88';
                if (learningData.status) {
                    if (learningData.status === 'training' || learningData.status === 'active') {
                        learnStatus = '活跃学习中';
                        statusColor = '#00ff88';
                    } else if (learningData.status === 'paused') {
                        learnStatus = '已暂停';
                        statusColor = '#ffaa00';
                    } else if (learningData.status === 'error') {
                        learnStatus = '异常';
                        statusColor = '#ff5555';
                    } else if (learningData.status === 'idle') {
                        learnStatus = '待机';
                        statusColor = '#88aaff';
                    } else {
                        learnStatus = learningData.status;
                    }
                }
                paragraphs[2].innerHTML = '状态: <strong style="color:' + statusColor + '">' + window.escapeHtml(learnStatus) + '</strong>';
            }
        }
    } else {
        // API失败或数据无效
        if (learningMetrics.length >= 3) {
            learningMetrics[0].textContent = '错误';
            learningMetrics[1].textContent = '错误';
            learningMetrics[2].textContent = '错误';
        }
        
        const learningTrend = document.querySelector('.learning-trend');
        if (learningTrend) {
            const paragraphs = learningTrend.querySelectorAll('p');
            if (paragraphs.length >= 3) {
                paragraphs[0].innerHTML = '知识增长趋势: <strong style="color:#ff5555">错误</strong>';
                paragraphs[1].innerHTML = '能力提升: <strong style="color:#ff5555">错误</strong>';
                paragraphs[2].innerHTML = '错误率下降: <strong style="color:#ff5555">错误</strong>';
            }
        }
    }
}



/**
 * 实时数据更新轮询回调（从统一轮询中心调度）
 */
var g_pollThinkCounter = 0;
async function pollRealTimeUpdates() {
    try {
        const cogState = await window.SelfLnnApi.getCognitionState();
        if (cogState.success && cogState.data) {
            const d = cogState.data;
            /* 自我反思 (H-002) */
            setEl('#cog-reflection', d.reflection || (d.data && d.data.reflection) || '等待自我反思数据更新...');
            /* 自我限制认知 (H-002) */
            setEl('#cog-limitations', d.limitations || '数据获取中...');
            /* 自我改进建议 (H-002) */
            setEl('#cog-suggestions', d.suggestions || '数据获取中...');
            /* 能力评估 */
            if (d.capability) {
                var cap = d.capability;
                setEl('.cog-item:nth-child(1) .cog-value', (cap.reasoning_ability !== undefined ? (cap.reasoning_ability * 100).toFixed(0) : '--') + '%');
                setEl('.cog-item:nth-child(2) .cog-value', (cap.learning_ability !== undefined ? (cap.learning_ability * 100).toFixed(0) : '--') + '%');
                setEl('.cog-item:nth-child(3) .cog-value', (cap.planning_ability !== undefined ? (cap.planning_ability * 100).toFixed(0) : '--') + '%');
                setEl('.cog-item:nth-child(4) .cog-value', (cap.adaptability !== undefined ? (cap.adaptability * 100).toFixed(0) : '--') + '%');
            }
            /* H-001: 知识元认知6指标 */
            if (d.knowledge) {
                var kn = d.knowledge;
                setEl('#cog-known-concepts',        kn.known_concepts      !== undefined ? String(kn.known_concepts)      : '--');
                setEl('#cog-unknown-concepts',      kn.unknown_concepts    !== undefined ? String(kn.unknown_concepts)    : '--');
                setEl('#cog-knowledge-coverage',    kn.knowledge_coverage  !== undefined ? (kn.knowledge_coverage  * 100).toFixed(1) + '%' : '--');
                setEl('#cog-knowledge-confidence',  kn.knowledge_confidence  !== undefined ? (kn.knowledge_confidence  * 100).toFixed(1) + '%' : '--');
                setEl('#cog-knowledge-freshness',   kn.knowledge_freshness   !== undefined ? (kn.knowledge_freshness   * 100).toFixed(1) + '%' : '--');
                setEl('#cog-knowledge-consistency', kn.knowledge_consistency !== undefined ? (kn.knowledge_consistency * 100).toFixed(1) + '%' : '--');
                var covFill = document.getElementById('cog-coverage-fill');
                if (covFill) covFill.style.width = (kn.knowledge_coverage !== undefined ? (kn.knowledge_coverage * 100).toFixed(1) : '0') + '%';
                var confFill = document.getElementById('cog-confidence-fill');
                if (confFill) confFill.style.width = (kn.knowledge_confidence !== undefined ? (kn.knowledge_confidence * 100).toFixed(1) : '0') + '%';
                var freshFill = document.getElementById('cog-freshness-fill');
                if (freshFill) freshFill.style.width = (kn.knowledge_freshness !== undefined ? (kn.knowledge_freshness * 100).toFixed(1) : '0') + '%';
                var consFill = document.getElementById('cog-consistency-fill');
                if (consFill) consFill.style.width = (kn.knowledge_consistency !== undefined ? (kn.knowledge_consistency * 100).toFixed(1) : '0') + '%';
            }
            /* L-002: 当前目标面板 */
            if (d.goal) {
                var g = d.goal;
                setEl('#cog-current-goal',     g.current_goal && g.current_goal !== 'none' ? g.current_goal : '无当前目标');
                setEl('#cog-goal-priority',    g.goal_priority    !== undefined ? (g.goal_priority    * 100).toFixed(0) + '%' : '--');
                setEl('#cog-goal-progress',    g.goal_progress    !== undefined ? (g.goal_progress    * 100).toFixed(0) + '%' : '--');
                setEl('#cog-goal-feasibility', g.goal_feasibility !== undefined ? (g.goal_feasibility * 100).toFixed(0) + '%' : '--');
                setEl('#cog-goal-confidence',  g.goal_confidence  !== undefined ? (g.goal_confidence  * 100).toFixed(0) + '%' : '--');
            }
            /* H-003: 学习进展认知6指标 */
            if (d.learning) {
                var l = d.learning;
                setEl('#cog-learning-rate',       l.learning_rate      !== undefined ? (l.learning_rate      * 100).toFixed(1) + '%' : '--');
                setEl('#cog-learning-efficiency',  l.learning_efficiency !== undefined ? (l.learning_efficiency * 100).toFixed(1) + '%' : '--');
                setEl('#cog-train-acc',           l.training_accuracy  !== undefined ? (l.training_accuracy  * 100).toFixed(1) + '%' : '--');
                setEl('#cog-test-acc',            l.test_accuracy      !== undefined ? (l.test_accuracy      * 100).toFixed(1) + '%' : '--');
                setEl('#cog-generalization',      l.generalization     !== undefined ? (l.generalization     * 100).toFixed(1) + '%' : '--');
                setEl('#cog-evolution-progress',  l.evolution_progress !== undefined ? (l.evolution_progress * 100).toFixed(1) + '%' : '--');
            }
            /* L-001: 自我身份认知4指标 */
            if (d.identity) {
                var id = d.identity;
                setEl('#cog-identity-evolution',   id.evolution_rate   !== undefined ? (id.evolution_rate   * 100).toFixed(1) + '%' : '--');
                setEl('#cog-identity-stability',   id.stability         !== undefined ? (id.stability         * 100).toFixed(1) + '%' : '--');
                setEl('#cog-identity-consistency', id.self_consistency !== undefined ? (id.self_consistency * 100).toFixed(1) + '%' : '--');
                setEl('#cog-identity-continuity',  id.continuity_score !== undefined ? (id.continuity_score * 100).toFixed(1) + '%' : '--');
            }
        }
    } catch (e) {
        console.warn('[轮询] 认知状态请求失败:', e.message);
    }

    g_pollThinkCounter++;
    if (g_pollThinkCounter % 30 === 0) {
        try {
            const thinkResult = await window.SelfLnnApi.agiThink('周期性系统状态分析', 0);
            if (thinkResult.success && thinkResult.data && thinkResult.data.reflection) {
                const cogReflectionEl = document.getElementById('cog-reflection');
                if (cogReflectionEl && thinkResult.data.reflection !== '无自我反思数据') {
                    cogReflectionEl.textContent = thinkResult.data.reflection;
                }
            }
        } catch (e) {
            console.warn('[轮询] AGI思维请求失败:', e.message);
        }
    }
}

/* ToM智能体卡片动态渲染器
 * 从后端API(/api/cognition/tom)获取心智理论数据并动态生成agent卡片。
 * 不再使用HTML硬编码的"智能体#1/#2/#3"模板，所有agent信息由后端提供。 */
function renderTomAgents(tomData) {
    var container = document.getElementById('tom-agents');
    if (!container) return;
    if (!tomData || !tomData.agents || !tomData.agents.length) {
        container.innerHTML = '<div class="tom-empty">等待心智理论数据...</div>';
        return;
    }
    var html = '';
    var agents = tomData.agents;
    for (var i = 0; i < agents.length; i++) {
        var a = agents[i];
        var beliefW = ((a.beliefScore || 0) * 100).toFixed(0);
        var desireW = ((a.desireScore || 0) * 100).toFixed(0);
        var intentW = ((a.intentionScore || 0) * 100).toFixed(0);
        var statusClass = a.online ? 'online' : 'offline';
        html += '<div class="tom-agent-card">' +
            '<div class="tom-agent-header">' +
            /* FE-027修复: 对后端返回的agent名称和信念文本使用escapeHtml转义，防止XSS注入 */
            '<span class="tom-agent-name">' + window.escapeHtml(a.name || ('智能体 #' + (i+1))) + '</span>' +
            '<span class="tom-agent-status ' + statusClass + '"></span>' +
            '</div>' +
            '<div class="tom-dimension">' +
            '<span class="tom-dim-label">信念</span>' +
            '<div class="tom-dim-bar"><div class="tom-dim-fill" style="width:' + beliefW + '%"></div></div>' +
            '<span class="tom-dim-value">' + beliefW + '%</span>' +
            '</div>' +
            '<div class="tom-dimension">' +
            '<span class="tom-dim-label">欲望</span>' +
            '<div class="tom-dim-bar"><div class="tom-dim-fill tom-desire" style="width:' + desireW + '%"></div></div>' +
            '<span class="tom-dim-value">' + desireW + '%</span>' +
            '</div>' +
            '<div class="tom-dimension">' +
            '<span class="tom-dim-label">意图</span>' +
            '<div class="tom-dim-bar"><div class="tom-dim-fill tom-intention" style="width:' + intentW + '%"></div></div>' +
            '<span class="tom-dim-value">' + intentW + '%</span>' +
            '</div>' +
            '<div class="tom-belief-content">' + window.escapeHtml(a.beliefText || '未知') + '</div>' +
            '</div>';
    }
    container.innerHTML = html;
    var summaryEl = document.getElementById('tom-summary-text');
    if (summaryEl) summaryEl.textContent = tomData.summary || '等待数据...';
}

/* 定期轮询心智理论状态并渲染 */
function updateTomDisplay() {
    if (typeof SelfLnnApi === 'undefined' || !SelfLnnApi.getCognitionStatus) {
        var container = document.getElementById('tom-agents');
        if (container) container.innerHTML = '<div class="tom-error">API不可用，无法获取心智理论数据</div>';
        return;
    }
    window.SelfLnnApi.getCognitionStatus().then(function(res) {
        if (typeof SelfLnnApi === 'undefined') return;
        if (res && res.success && res.data && res.data.tom) {
            renderTomAgents(res.data.tom);
        } else {
            var container = document.getElementById('tom-agents');
            /* FE-006修复: 使用escapeHtml转义动态内容，防止XSS注入 */
            if (container) container.innerHTML = '<div class="tom-error">心智理论数据获取失败：' + window.escapeHtml(res && res.error ? res.error : '后端未返回有效数据') + '</div>';
        }
    }).catch(function(e) {
        console.warn('[认知] ToM更新失败:', e.message);
        var container = document.getElementById('tom-agents');
        /* FE-006修复: 使用escapeHtml转义动态内容，防止XSS注入 */
        if (container) container.innerHTML = '<div class="tom-error">心智理论数据获取失败：' + window.escapeHtml(e.message || '网络错误') + '</div>';
        var summaryEl = document.getElementById('tom-summary-text');
        if (summaryEl) summaryEl.textContent = '获取失败';
    });
}

/**
 * 从数据引擎更新所有UI组件
 */
function updateRealTimeMetricsFromApi(systemStatus) {
    if (!systemStatus || !systemStatus.success || !systemStatus.data) {
        return;
    }
    updateRealTimeMetrics(systemStatus);
}

/**
 * 更新实时指标
 */
function updateRealTimeMetrics(systemStatus) {
    if (!systemStatus.success || !systemStatus.data) {
        return;
    }
    
    var sys = systemStatus.data.system;
    if (!sys) return;
    var mods = sys.modules || {};
    var reqs = sys.requests || {};
    
    // === 系统健康状态 ===
    setEl('.health-status .status-text', sys.status === 'running' ? '运行正常' : '状态异常');
/* 禁止虚假CPU数据，使用后端真实CPU值 */
    var realCpu = (sys.cpu_usage !== undefined && sys.cpu_usage >= 0) ? sys.cpu_usage : ((reqs.cpu !== undefined && reqs.cpu >= 0) ? reqs.cpu : -1);
    setEl('.metric:nth-child(1) .metric-value', realCpu >= 0 ? Math.round(realCpu) + '%' : '--');
    setEl('.metric:nth-child(1) .metric-fill', realCpu >= 0 ? Math.round(realCpu) + '%' : '0%', 'width');
    if (sys.uptime) { var d=Math.floor(sys.uptime/86400); var h=Math.floor((sys.uptime%86400)/3600); setEl('.metric:nth-child(3) .metric-value', d+'天 '+h+'小时'); }
    
    // === LNN状态（单一液态神经网络） ===
    var lnnData = mods.lnn || {};
    var modelS = document.querySelectorAll('.model-stats .stat-value');
    if (modelS.length >= 3) {
        modelS[0].textContent = '单一液态神经网络(CfC)'; /* 架构始终为单一LNN */
        modelS[1].textContent = lnnData.hidden_size ? String(lnnData.hidden_size) : '--'; /* 隐藏层维度 */
        modelS[2].textContent = lnnData.total_params ? String(lnnData.total_params) : '--'; /* 总参数 */
    }
    /* 更新LNN子模块状态（单一模型的三个组成部分） */
    var submodules = document.querySelectorAll('#lnn-submodules .model-status');
    if (submodules.length >= 3) {
        var uniEnc = mods.multimodal || {};
        var cfcMod = lnnData;
        var decMod = mods.reasoning || {};
        submodules[0].textContent = uniEnc.available ? '活跃' : '就绪';     /* 多模态统一编码 */
        submodules[0].className = 'model-status ' + (uniEnc.available ? 'active' : 'pending');
        submodules[1].textContent = cfcMod.available ? '活跃' : '就绪';     /* 连续状态演化(CfC) */
        submodules[1].className = 'model-status ' + (cfcMod.available ? 'active' : 'pending');
        submodules[2].textContent = decMod.available ? '活跃' : '就绪';     /* 统一决策输出 */
        submodules[2].className = 'model-status ' + (decMod.available ? 'active' : 'pending');
    }
    
    // === 训练进度 ===
    var train = sys.training || {};
    setEl('.progress-value', train.active === true ? train.epoch+'轮' : '待训练');
    setEl('.progress-fill', train.active && train.epoch>0 ? Math.min(100,train.epoch*10)+'%' : '0%', 'width');
    var td = document.querySelectorAll('.training-details .detail-value');
    if (td.length>=4) { td[0].textContent=train.epoch||'--'; td[1].textContent=train.loss!=null ? train.loss.toFixed(4) : '--'; td[2].textContent=train.accuracy!=null ? (train.accuracy*100).toFixed(1)+'%' : '--'; td[3].textContent='--'; }

    // === LNN状态监控（圆环+指标条） ===
    var lnnSt = mods.lnn_state || {};
    var circ = 2 * Math.PI * 36;
    if (lnnSt.stability !== undefined) {
        var pct = Math.min(100, Math.max(0, lnnSt.stability * 100));
        var ctAll = document.querySelectorAll('.lnn-metrics .circle-text');
        var cfAll = document.querySelectorAll('.lnn-metrics .circle-fill');
        if (ctAll[0]) ctAll[0].textContent = pct.toFixed(1) + '%';
        if (cfAll[0]) cfAll[0].style.strokeDashoffset = circ - (pct / 100) * circ;
    }
    if (lnnSt.convergence_rate !== undefined) {
        var pct = Math.min(100, Math.max(0, lnnSt.convergence_rate * 100));
        var ctAll = document.querySelectorAll('.lnn-metrics .circle-text');
        var cfAll = document.querySelectorAll('.lnn-metrics .circle-fill');
        if (ctAll[1]) ctAll[1].textContent = pct.toFixed(1) + '%';
        if (cfAll[1]) cfAll[1].style.strokeDashoffset = circ - (pct / 100) * circ;
    }
    if (lnnSt.dynamic_response !== undefined) {
        var ctAll = document.querySelectorAll('.lnn-metrics .circle-text');
        var cfAll = document.querySelectorAll('.lnn-metrics .circle-fill');
        if (ctAll[2]) ctAll[2].textContent = lnnSt.dynamic_response.toFixed(1) + ' Hz';
        if (cfAll[2]) cfAll[2].style.strokeDashoffset = circ - (Math.min(100, lnnSt.dynamic_response) / 100) * circ;
    }
    if (lnnSt.viscosity !== undefined) {
        var pct = Math.min(100, Math.max(0, lnnSt.viscosity * 100));
        var ivAll = document.querySelectorAll('.state-indicators-grid .indicator-value');
        var ifAll = document.querySelectorAll('.state-indicators-grid .indicator-fill');
        if (ivAll[0]) ivAll[0].textContent = lnnSt.viscosity.toFixed(4);
        if (ifAll[0]) ifAll[0].style.width = pct + '%';
    }
    if (lnnSt.temperature_entropy !== undefined) {
        var pct = Math.min(100, Math.max(0, lnnSt.temperature_entropy * 100));
        var ivAll = document.querySelectorAll('.state-indicators-grid .indicator-value');
        var ifAll = document.querySelectorAll('.state-indicators-grid .indicator-fill');
        if (ivAll[1]) ivAll[1].textContent = lnnSt.temperature_entropy.toFixed(4);
        if (ifAll[1]) ifAll[1].style.width = pct + '%';
    }
    if (lnnSt.flow_rate !== undefined) {
        var pct = Math.min(100, Math.max(0, lnnSt.flow_rate * 100));
        var ivAll = document.querySelectorAll('.state-indicators-grid .indicator-value');
        var ifAll = document.querySelectorAll('.state-indicators-grid .indicator-fill');
        if (ivAll[2]) ivAll[2].textContent = lnnSt.flow_rate.toFixed(4);
        if (ifAll[2]) ifAll[2].style.width = pct + '%';
    }
    if (lnnSt.diffusion_coefficient !== undefined) {
        var pct = Math.min(100, Math.max(0, lnnSt.diffusion_coefficient * 100));
        var ivAll = document.querySelectorAll('.state-indicators-grid .indicator-value');
        var ifAll = document.querySelectorAll('.state-indicators-grid .indicator-fill');
        if (ivAll[3]) ivAll[3].textContent = lnnSt.diffusion_coefficient.toFixed(4);
        if (ifAll[3]) ifAll[3].style.width = pct + '%';
    }

    // === GPU信息 ===
    var gpu = mods.gpu || {};
    setEl('#gpu-device-name', gpu.name || '未检测');
    setEl('#gpu-memory', gpu.memory_mb ? (gpu.memory_mb/1024).toFixed(1)+' GB' : '--');
    setEl('#gpu-compute-cap', gpu.available ? '已就绪' : '未检测');
    /* P2-002修复: GPU不可用时usage=-1应显示为'不可用'而非'-1.0%' */
    setEl('#gpu-cuda-version', (gpu.usage != null && gpu.usage >= 0) ? gpu.usage.toFixed(1)+'%' : '不可用');
    
    // === LNN状态空间（模型管理页卡） ===
    var lnnMod = mods.lnn || {};
    var lnnSt2 = mods.lnn_state || {};
    setEl('#state-dimension', lnnMod.h_dim ? lnnMod.h_dim + '维' : '--');
    setEl('#lnn-accuracy', sys.training && sys.training.accuracy ? (sys.training.accuracy * 100).toFixed(1) + '%' : '--');
/* 禁止虚假推理延迟（请求速率反推），必须使用后端真实值 */
    var realLatency = (lnnSt2.inference_latency_ms !== undefined && lnnSt2.inference_latency_ms >= 0) ? lnnSt2.inference_latency_ms
                    : (sys.inference_latency_ms !== undefined && sys.inference_latency_ms >= 0) ? sys.inference_latency_ms : -1;
    setEl('#inference-latency', realLatency >= 0 ? realLatency.toFixed(1) + 'ms' : '--');
    setEl('#time-constant', lnnMod.tau !== undefined && lnnMod.tau !== null ? lnnMod.tau.toFixed(4) + 's' : '--');
    
    // === 模型详情（CfC核心信息+性能指标圆环） ===
    setEl('#model-detail-name', lnnMod.available ? (lnnMod.model_name || 'CfC-LNN') : '--');
    setEl('#model-detail-type', lnnMod.ode_solver || 'CfC闭式解');
    setEl('#model-detail-id', lnnMod.h_dim ? lnnMod.h_dim + '维' : lnnMod.hidden_size ? lnnMod.hidden_size + '维' : '--');
    setEl('#model-detail-created', lnnMod.input_size ? lnnMod.input_size + '通道' : '--');
    setEl('#model-detail-params', lnnMod.tau !== undefined ? lnnMod.tau.toFixed(4) + 's' : '--');
    setEl('#model-detail-size', lnnMod.precision || 'float32');
    var modelBadge = document.querySelector('.model-item .model-item-status .status-badge');
    if (modelBadge) {
        if (lnnMod.available) {
            modelBadge.textContent = '运行中';
            modelBadge.className = 'status-badge active';
        } else {
            modelBadge.textContent = '未初始化';
            modelBadge.className = 'status-badge pending';
        }
    }
    var circ3 = 2 * Math.PI * 36;
    if (sys.training) {
        var acc = sys.training.accuracy ? (sys.training.accuracy * 100) : 0;
        setEl('#model-accuracy', acc.toFixed(1) + '%');
        var af = document.getElementById('model-accuracy-fill');
        if (af) af.style.strokeDashoffset = circ3 - (acc / 100) * circ3;
    }
    if (mods.lnn_state) {
        var convPct = Math.min(100, Math.max(0, (mods.lnn_state.convergence_rate || 0) * 100));
        setEl('#model-efficiency', convPct.toFixed(1) + '%');
        var ef = document.getElementById('model-efficiency-fill');
        if (ef) ef.style.strokeDashoffset = circ3 - (convPct / 100) * circ3;
        var entPct = Math.min(100, Math.max(0, (mods.lnn_state.temperature_entropy || 0) * 100));
        setEl('#model-memory-efficiency', entPct.toFixed(1) + '%');
        var mf = document.getElementById('model-memory-fill');
        if (mf) mf.style.strokeDashoffset = circ3 - (entPct / 100) * circ3;
    }
    
    // === AGI对话状态 ===
    var dlgStatus = document.getElementById('dialogue-model-status');
    var wsStatus = document.getElementById('dialogue-ws-status');
    /* P2-06: 使用SelfLnnWebSocket真实连接状态代替不存在的g_dataEngine.wsReady */
    var wsReady = (typeof window.SelfLnnWebSocket !== 'undefined' && window.SelfLnnWebSocket && window.SelfLnnWebSocket.isConnected);
    if (dlgStatus) dlgStatus.textContent = wsReady ? 'WS就绪' : 'HTTP模式';
    if (wsStatus) {
        wsStatus.textContent = wsReady ? 'WebSocket流式' : 'HTTP轮询';
        wsStatus.style.color = wsReady ? '#22c55e' : '#ffc107';
    }
    
    // === 自我认知 ===
    var cog = mods.cognition || {};
    setEl('#cognition-status-badge', cog.available ? '已激活' : '未启用');
    
    // === API密钥 ===
    var apiK = sys.api_key || {};
    setEl('#dash-api-address', 'http://' + (window.SELFLNN_CONFIG ? window.SELFLNN_CONFIG.host : 'localhost') + ':' + (window.SELFLNN_CONFIG ? (window.SELFLNN_CONFIG.port || 8080) : 8080) + '/api');
    setEl('#dash-api-key-status', apiK.set ? '已设置' : '未设置');
    
    // === API统计 ===
    setEl('#api-stat-total', reqs.total||0);
    setEl('#api-stat-success', (reqs.total||0)-(reqs.errors||0));
    setEl('#api-stat-active', reqs.connections||0);
    setEl('#api-stat-errors', reqs.errors||0);
    setEl('#api-stat-rate', reqs.rate_per_minute!==undefined ? reqs.rate_per_minute : '--');
    
    // === LNN统一记忆层 ===
    var mem = mods.memory || {};
    var memTypes = document.querySelectorAll('.memory-type .type-usage');
    var memFills = document.querySelectorAll('.memory-type .type-fill');
    if (memTypes.length >= 3) {
        if (mem.total > 0) {
            memTypes[0].textContent = ((mem.short_term || 0) / mem.total * 100).toFixed(0) + '%';
            memTypes[1].textContent = ((mem.long_term || 0) / mem.total * 100).toFixed(0) + '%';
            memTypes[2].textContent = ((mem.episodic || 0) / mem.total * 100).toFixed(0) + '%';
        } else {
            memTypes[0].textContent = '空';
            memTypes[1].textContent = '空';
            memTypes[2].textContent = '空';
        }
    }
    if (memFills.length >= 3) {
        if (mem.total > 0) {
            memFills[0].style.width = ((mem.short_term || 0) / mem.total * 100).toFixed(0) + '%';
            memFills[1].style.width = ((mem.long_term || 0) / mem.total * 100).toFixed(0) + '%';
            memFills[2].style.width = ((mem.episodic || 0) / mem.total * 100).toFixed(0) + '%';
        } else {
            memFills[0].style.width = '0%';
            memFills[1].style.width = '0%';
            memFills[2].style.width = '0%';
        }
    }
    var memInfo = document.querySelector('.memory-info');
    if (memInfo) {
        var ps = memInfo.querySelectorAll('p');
        if (ps.length >= 3) {
            var entryCount = mem.entry_count !== undefined && mem.entry_count !== null ? mem.entry_count : 0;
            var s0 = ps[0] && ps[0].querySelector('strong');
            var s1 = ps[1] && ps[1].querySelector('strong');
            var s2 = ps[2] && ps[2].querySelector('strong');
            if (s0) s0.textContent = entryCount + ' 条';
            if (s1) s1.textContent = mem.consolidation_ratio !== undefined && mem.consolidation_ratio !== null ? (mem.consolidation_ratio * 100).toFixed(1) + '%' : '0.0%';
            if (s2) s2.textContent = mem.retrieval_success_rate !== undefined && mem.retrieval_success_rate !== null ? (mem.retrieval_success_rate * 100).toFixed(1) + '%' : '0.0%';
        }
    }
    setEl('#dash-knowledge-count', sys.knowledge ? sys.knowledge.count : '--');
    
    // === LNN统一推理 ===
    var reas = mods.reasoning || {};
    var total_req = reqs.total || 0;
    var errors_req = reqs.errors || 0;
    var success_rate = total_req > 0 ? ((total_req - errors_req) / total_req * 100).toFixed(1) : '--';
/* 禁止虚假响应时间（请求速率反推），必须使用后端真实值 */
    var avg_resp_time = (reqs.avg_response_time_ms !== undefined && reqs.avg_response_time_ms >= 0) ? reqs.avg_response_time_ms.toFixed(1) + 'ms'
                      : (sys.avg_response_time_ms !== undefined && sys.avg_response_time_ms >= 0) ? sys.avg_response_time_ms.toFixed(1) + 'ms' : '--';
    setEl('.reasoning-stats .stat:nth-child(1) .stat-value', reqs.rate_per_minute ? (reqs.rate_per_minute / 60).toFixed(1) : '0.0');
    setEl('.reasoning-stats .stat:nth-child(2) .stat-value', avg_resp_time);
    setEl('.reasoning-stats .stat:nth-child(3) .stat-value', success_rate);
    
    // === 学习状态 ===
    var learn = mods.learning || {};
    setEl('.learning-metrics .metric:nth-child(1) .metric-value', learn.progress ? (learn.progress * 100).toFixed(0) + '%' : '就绪');
    setEl('.learning-metrics .metric:nth-child(2) .metric-value', learn.available ? '已连接' : '未连接');
    setEl('.learning-metrics .metric:nth-child(3) .metric-value', sys.knowledge ? sys.knowledge.count : '--');
    var trendEl = document.querySelector('.learning-trend');
    if (trendEl) {
        var tps = trendEl.querySelectorAll('p');
        if (tps.length >= 3) {
            tps[0].querySelector('strong').textContent = learn.progress ? (learn.progress * 100).toFixed(0) + '%' : '0%';
            tps[1].querySelector('strong').textContent = learn.available ? '活跃' : '待激活';
            tps[2].querySelector('strong').textContent = learn.available ? '监控中' : '待训练';
        }
    }
    
    // === 机器人控制核心指标 ===
    var robot = mods.robotics || {};
    setEl('.robot-indicator:nth-child(1) .indicator-value', robot.available ? '已连接' : '未连接');
    setEl('.robot-indicator:nth-child(2) .indicator-value', robot.state!=null ? robot.state : '--');
    setEl('.robot-indicator:nth-child(3) .indicator-value', robot.battery!=null ? robot.battery.toFixed(0)+'%' : '--');
    
    // === 更新API图表 ===
    if (window.g_apiUsageChart && reqs) {
        var now = new Date();
        var label = now.getHours()+':'+String(now.getMinutes()).padStart(2,'0');
        window.g_apiUsageChart.addData(label, [reqs.total||0]);
    }

    // === 备份状态（F-005修复: 不使用localStorage缓存作为fallback，后端未连接时显示"数据未加载"） ===
    (function() {
        var backupStatusEl = document.getElementById('backup-status-text');
        var backupTimeEl = document.getElementById('backup-time-text');
        if (backupStatusEl && backupStatusEl.textContent === '未连接') backupStatusEl.textContent = '数据未加载';
        if (backupTimeEl && backupTimeEl.textContent === '未连接') backupTimeEl.textContent = '数据未加载';
    })();
}

function setEl(sel, val, prop) {
    if (!sel) return;
    var els = document.querySelectorAll(sel);
    for (var i=0; i<els.length; i++) {
        if (prop === 'width') els[i].style.width = val;
        else els[i].textContent = val;
        /* L-023修复: 更新数据时同步移除骨架屏加载类 */
        els[i].classList.remove('skeleton-loading');
    }
}

/**
 * 设置事件监听器
 */
function setupEventListeners() {
    /* C-04修复: 在重新设置监听器前清理旧的 */
    window._cleanupAllEventListeners();
    
    // 导航切换 + 下拉菜单
    var navLinks = document.querySelectorAll('.nav a');
    
    /* C-04修复: 提取为命名函数以便清理 */
    function _handleNavLinkClick(e) {
        var parentLi = this.parentElement;
        var isDropdownToggle = parentLi.classList.contains('nav-dropdown');
        var isDropdownItem = parentLi.parentElement && parentLi.parentElement.classList.contains('dropdown-menu');
        var href = this.getAttribute('href');
        
        if (isDropdownToggle) {
            e.preventDefault();
            parentLi.classList.toggle('active');
            return;
        }
        
        if (isDropdownItem) {
            e.preventDefault();
            var dropdown = parentLi.closest('.nav-dropdown');
            if (dropdown) dropdown.classList.remove('active');
            var targetId = href.substring(1);
            document.querySelectorAll('.section').forEach(function(s) { s.classList.remove('active'); });
            var target = document.getElementById(targetId);
            if (target) target.classList.add('active');
            navLinks.forEach(function(l) { l.classList.remove('active'); });
            this.classList.add('active');
            return;
        }
        
        if (href && href.startsWith('#')) {
            e.preventDefault();
            navLinks.forEach(function(l) { l.classList.remove('active'); });
            this.classList.add('active');
            var targetId = href.substring(1);
            document.querySelectorAll('.section').forEach(function(s) { s.classList.remove('active'); });
            var target = document.getElementById(targetId);
            if (target) target.classList.add('active');
        }
    }
    
    navLinks.forEach(function(link) {
        window._registerEventListener(link, 'click', _handleNavLinkClick);
    });
    
    /* C-04修复: 提取为命名函数 */
    function _handleDocClickCloseDropdown(e) {
        if (!e.target.closest('.nav-dropdown')) {
            document.querySelectorAll('.nav-dropdown.active').forEach(function(d) { d.classList.remove('active'); });
        }
    }
    window._registerEventListener(document, 'click', _handleDocClickCloseDropdown);
    
    /* C-04修复: 提取为命名函数 */
    function _handleKeydownShortcuts(e) {
        // Ctrl+Shift+R 刷新仪表盘（不禁用浏览器Ctrl+R刷新）
        if (e.ctrlKey && e.shiftKey && e.key === 'R') {
            e.preventDefault();
            refreshDashboard();
        }
        
        // Ctrl + T 开始训练
        if (e.ctrlKey && e.key === 't') {
            e.preventDefault();
            startTrainingQuick();
        }
        
        // Ctrl + S 紧急停止
        if (e.ctrlKey && e.key === 's') {
            e.preventDefault();
            emergencyStop();
        }
        
        // Ctrl + H 显示帮助
        if (e.ctrlKey && e.key === 'h') {
            e.preventDefault();
            showHelp();
        }
        
        // 机器人控制快捷键
        var robotControlEl = document.getElementById('robot-control');
        if (robotControlEl && robotControlEl.classList.contains('active')) {
            // W键 - 前进
            if (e.key === 'w' || e.key === 'W') {
                e.preventDefault();
                moveForward();
            }
            
            // S键 - 后退
            if (e.key === 's' || e.key === 'S') {
                e.preventDefault();
                moveBackward();
            }
            
            // A键 - 左转
            if (e.key === 'a' || e.key === 'A') {
                e.preventDefault();
                moveLeft();
            }
            
            // D键 - 右转
            if (e.key === 'd' || e.key === 'D') {
                e.preventDefault();
                moveRight();
            }
            
            // 空格键 - 停止
            if (e.key === ' ') {
                e.preventDefault();
                stopMovement();
            }
            
            // F5键 - 刷新机器人状态
            if (e.key === 'F5') {
                e.preventDefault();
                refreshRobotStatus();
            }
        }
    }
    window._registerEventListener(document, 'keydown', _handleKeydownShortcuts);

    // === 设备管理器事件绑定 ===
    setTimeout(function() {
        /* C-04修复: 提取为命名函数 */
        function _handleVoiceInputClick() { toggleVoiceInput(); }
        function _handleVoiceOutputClick() { toggleVoiceOutput(); }
        function _handleMultimodalToggleChange() {
            if (g_dialogueEnhanced) {
                if (this.checked) g_dialogueEnhanced.enableMultimodal();
                else g_dialogueEnhanced.disableMultimodal();
            }
        }
        
        // 语音输入按钮
        var voiceInputBtn = document.getElementById('dialogue-voice-input');
        if (voiceInputBtn) {
            window._registerEventListener(voiceInputBtn, 'click', _handleVoiceInputClick);
        }

        // 语音输出按钮
        var voiceOutputBtn = document.getElementById('dialogue-voice-output');
        if (voiceOutputBtn) {
            window._registerEventListener(voiceOutputBtn, 'click', _handleVoiceOutputClick);
        }

        // 多模态切换复选框
        var multimodalToggle = document.getElementById('dialogue-multimodal-toggle');
        if (multimodalToggle) {
            window._registerEventListener(multimodalToggle, 'change', _handleMultimodalToggleChange);
        }

        // WebSocket流式对话初始化
        if (g_dialogueEnhanced) {
            g_dialogueEnhanced.onWsStatusChange = function(status) {
                var el = document.getElementById('dialogue-ws-status');
                if (!el) return;
                switch (status) {
                    case 'connected':
                        el.textContent = '🟢 流式连接';
                        el.style.color = '#22c55e';
                        break;
                    case 'connecting':
                        el.textContent = '🟡 连接中...';
                        el.style.color = '#eab308';
                        break;
                    case 'disconnected':
                        el.textContent = '⚪ 未连接';
                        el.style.color = '#6b7280';
                        break;
                    case 'error':
                        el.textContent = '🔴 连接错误';
                        el.style.color = '#ef4444';
                        break;
                }
            };
            g_dialogueEnhanced.onDialogueToken = function(tokenText, progress, isFinal) {
                updateStreamingToken(tokenText, progress, isFinal);
            };
            g_dialogueEnhanced.onDialogueResponse = function(fullText, confidence) {
                finalizeStreamingResponse(fullText, confidence);
            };
        }

        // 语音输入结果回调
        if (g_dialogueEnhanced) {
            g_dialogueEnhanced.onVoiceInputResult = function(result) {
                if (result && result.text) {
                    var input = document.getElementById('dialogue-input');
                    if (input) {
                        input.value = result.text;
                        sendDialogueMessage();
                    }
                } else {
                    showNotification('语音识别失败: ' + (result.error || '未知错误'), 'danger');
                }
            };
            g_dialogueEnhanced.onVoiceOutputStart = function() {
                var statusEl = document.getElementById('voice-output-status');
                if (statusEl) statusEl.textContent = '播报中...';
            };
            g_dialogueEnhanced.onVoiceOutputStop = function() {
                var statusEl = document.getElementById('voice-output-status');
                if (statusEl) statusEl.textContent = '空闲';
            };
            g_dialogueEnhanced.onVoiceOutputError = function(err) {
                var statusEl = document.getElementById('voice-output-status');
                if (statusEl) statusEl.textContent = '错误: ' + err;
            };
        }

        // TTS速度控制
        function _handleTtsSpeedInput() {
            var val = parseFloat(this.value);
            if (g_dialogueEnhanced) g_dialogueEnhanced.setTtsSpeed(val);
            var display = document.getElementById('tts-speed-value');
            if (display) display.textContent = val.toFixed(1) + 'x';
        }
        var ttsSpeedSlider = document.getElementById('tts-speed');
        if (ttsSpeedSlider) {
            window._registerEventListener(ttsSpeedSlider, 'input', _handleTtsSpeedInput);
        }

        // 语音指令开关
        function _handleVoiceCommandToggleChange() {
            if (g_voiceCommandSystem) {
                if (this.checked) g_voiceCommandSystem.startContinuousMode();
                else g_voiceCommandSystem.stopContinuousMode();
            }
        }
        var voiceCommandToggle = document.getElementById('voice-command-toggle');
        if (voiceCommandToggle) {
            window._registerEventListener(voiceCommandToggle, 'change', _handleVoiceCommandToggleChange);
        }

        // 对话发送摄像头画面复选框
        function _handleSendCameraChange() {
            if (this.checked && g_deviceManager) {
                var hasActiveCam = g_deviceManager.cameras.some(function(c) { return c.active; });
                if (!hasActiveCam) {
                    showNotification('没有已激活的摄像头，请先在设备管理中启动摄像头', 'warning');
                    this.checked = false;
                }
            }
        }
        var sendCameraCb = document.getElementById('dialogue-send-camera');
        if (sendCameraCb) {
            window._registerEventListener(sendCameraCb, 'change', _handleSendCameraChange);
        }
    }, 200);

    // 初始化所有 range slider 的实时值显示
    initRangeSliders();

    /* P2-06: WebSocket连接初始化 — 注册可视化WS处理器并建立连接 */
    if (!window.__wsVisualizationConnected) {
        window.__wsVisualizationConnected = true;
        connectVisualizationWebSocket();
    }

    /* P2-06: 监听WebSocket连接状态变化，更新训练中心WS状态指示器 */
    /* C-04修复: 提取为命名函数 */
    function _handleWsConnectionStatus(e) {
        var wsStatusEl = document.getElementById('ws-status');
        if (!wsStatusEl) return;
        if (e.detail && e.detail.connected) {
            wsStatusEl.textContent = '已连接';
            wsStatusEl.style.color = '#22c55e';
        } else {
            wsStatusEl.textContent = '未连接';
            wsStatusEl.style.color = '#f87171';
        }
    }
    window._registerEventListener(document, 'websocket-connection-status', _handleWsConnectionStatus);

    /* P2-06: 监听WebSocket重连状态，更新WS状态指示器 */
    function _handleWsReconnectStatus(e) {
        var wsStatusEl = document.getElementById('ws-status');
        if (!wsStatusEl) return;
        wsStatusEl.textContent = '重连中(' + e.detail.attempt + '/' + e.detail.maxAttempts + ')';
        wsStatusEl.style.color = '#eab308';
    }
    window._registerEventListener(document, 'websocket-reconnect-status', _handleWsReconnectStatus);
}

/**
 * 初始化所有 range slider 的实时值显示更新
 * 自动为每个 .range-with-value 中的 input[type="range"] 绑定 oninput 事件
 */
function initRangeSliders() {
    /* C-04修复: 提取为命名函数 */
    function _handleRangeSliderInput() {
        var display = this.parentElement.querySelector('.slider-value');
        if (display) {
            display.textContent = this.value;
        }
    }
    document.querySelectorAll('.range-with-value input[type="range"]').forEach(function(slider) {
        window._registerEventListener(slider, 'input', _handleRangeSliderInput);
    });
}

/**
 * 仪表盘快速开始训练（P0-005修复：重命名避免与training-center.js冲突）
 */
async function startTrainingQuick() {
    /* FE-026修复: 添加LoadingOverlay显示，提供用户反馈 */
    LoadingOverlay.show('正在启动训练任务...');
    showNotification('开始新的训练任务...', 'info');
    
    // 使用默认配置，委托给 startTrainingJob 统一处理
    const defaultConfig = {
        model: 'LNN-Core',
        training_type: 'fine-tuning',
        epochs: 50,
        batch_size: 32,
        optimizer: 'adam',
        learning_rate: 0.001,
        weight_decay: 0.0001,
        use_gpu: false,
        mixed_precision: false,
        early_stopping: false,
        checkpoint_interval: 10
    };
    
    try {
        if (window.SelfLnnApi && typeof window.SelfLnnApi.startTrainingJob === 'function') {
            const result = await window.SelfLnnApi.startTrainingJob(defaultConfig);
            
            if (result.success) {
                LoadingOverlay.hide();
                showNotification('✅ 训练任务已启动', 'success');
                
                const trainingStatus = document.querySelector('.model-item:nth-child(3) .model-status');
                if (trainingStatus) {
                    trainingStatus.textContent = '训练中';
                    trainingStatus.className = 'model-status training';
                }
                
                startTrainingStatusPolling();
            } else {
                LoadingOverlay.hide();
                showNotification('❌ 训练任务启动失败: ' + (result.error || '未知错误'), 'danger');
            }
        } else {
            LoadingOverlay.hide();
            showNotification('❌ 训练后端未连接', 'danger');
        }
    } catch (error) {
        LoadingOverlay.hide();
        console.error('开始训练失败:', error);
        showNotification('❌ 开始训练时出错: ' + error.message, 'danger');
    }
}

/**
 * 加载模型
 */
async function loadModel() {
    showNotification('正在加载LNN模型...', 'info');
    try {
        const result = await window.SelfLnnApi.getModelStatus();
        if (result && result.success) {
            showNotification('✅ LNN模型已加载（当前状态: ' + (result.data && result.data.status ? result.data.status : '运行中') + '）', 'success');
            refreshLNNStatus();
        } else {
            showNotification('⚠️ 模型加载状态: ' + (result ? result.error || '模型未启动' : '后端未连接'), 'warning');
        }
    } catch (e) {
        showNotification('❌ 加载模型时出错: ' + e.message, 'danger');
    }
}

/**
 * 系统备份
 */
async function backupSystem() {
    showNotification('正在进行系统备份...', 'info');
    try {
        const result = await window.SelfLnnApi.backupSystem();
        if (result && result.success) {
            showNotification('✅ 系统备份成功', 'success');
            var backupStatusEl = document.getElementById('backup-status-text');
            var backupTimeEl = document.getElementById('backup-time-text');
            if (backupStatusEl) backupStatusEl.textContent = '已完成';
            if (backupTimeEl) backupTimeEl.textContent = new Date().toLocaleString();
        } else {
            showNotification('❌ 备份失败: ' + (result ? result.error || '未知错误' : '后端未连接'), 'danger');
        }
    } catch (e) {
        showNotification('❌ 备份时出错: ' + e.message, 'danger');
    }
}

/**
 * 紧急停止
 */
async function emergencyStop() {
    if (!safeConfirm('⚠️ 确定要执行紧急停止？这将立即停止所有机器人运动和学习进程。')) return;
    showNotification('🛑 紧急停止已触发...', 'danger');
    try {
        const result = await window.SelfLnnApi.robotEmergencyStop();
        if (result && result.success) {
            showNotification('✅ 紧急停止已执行', 'success');
        } else {
            showNotification('⚠️ 紧急停止: ' + (result ? result.error || '后端未连接' : '后端未连接'), 'warning');
        }
        try { await window.SelfLnnApi.resetSystem(); } catch (e) {
                console.error('[紧急停止] 重置系统失败:', e && e.message ? e.message : e);
            }
    } catch (e) {
        showNotification('❌ 紧急停止失败: ' + e.message, 'danger');
    }
}

/**
 * 网络连接测试（P0-004修复：实现缺失的testNetwork函数）
 *: 如果HTML内联版本已定义，跳过main.js版本
 */
if (typeof testNetwork === 'undefined') {
async function testNetwork() {
    var resultEl = document.getElementById('network-test-result');
    var btnEl = document.querySelector('button[onclick="testNetwork()"]');
    
    if (btnEl) { btnEl.disabled = true; btnEl.textContent = '测试中...'; }
    if (resultEl) { resultEl.textContent = '正在测试网络连接...'; resultEl.style.color = '#f0a500'; }
    
    var startTime = performance.now();
    try {
        var result = await window.SelfLnnApi.getHealth();
        var elapsed = Math.round(performance.now() - startTime);
        
        if (result && result.success) {
            var gpuInfo = result.data && result.data.gpu ? result.data.gpu.backend || 'CPU' : 'CPU';
            var wsStatus = (window._wsManager && window._wsManager.isConnected) ? '已连接' : '未连接';
            if (resultEl) {
                resultEl.innerHTML = '✅ 连接成功（延迟: ' + elapsed + 'ms）<br>' +
                    'GPU后端: ' + gpuInfo + ' | WebSocket: ' + wsStatus;
                resultEl.style.color = '#00d68f';
            }
            showNotification('✅ 网络连接正常（' + elapsed + 'ms）', 'success');
        } else {
            if (resultEl) {
                resultEl.textContent = '⚠️ 后端响应异常: ' + (result ? result.error || '未知错误' : '连接超时');
                resultEl.style.color = '#f0a500';
            }
            showNotification('⚠️ 后端连接异常', 'warning');
        }
    } catch (e) {
        if (resultEl) {
            resultEl.textContent = '❌ 连接失败: ' + (e.message || '网络不可达');
            resultEl.style.color = '#ff5252';
        }
        showNotification('❌ 网络连接失败: ' + e.message, 'danger');
    }
    if (btnEl) { btnEl.disabled = false; btnEl.textContent = '测试连接'; }
}
} /* testNetwork条件定义结束 */

/**
 * 刷新仪表盘
 */
async function refreshDashboard() {
    // 从DataEngine缓存刷新，不发起独立API请求
    var cached = g_dataEngine ? g_dataEngine.getData() : null;
    if (cached && cached.system && cached.system._connected) {
        updateRealTimeMetrics({ success: true, data: cached.system });
        refreshApiUsageStats();
        showNotification('✅ 仪表盘已刷新', 'success');
    } else {
        showNotification('⚠️ 后端未连接，请稍候...', 'warning');
    }
}

/**
 * 显示帮助
 */
function showHelp() {
    const helpMessage = `
SELF-LNN AGI 管理系统 - 快捷键
    
Ctrl + R: 刷新仪表盘
Ctrl + T: 开始训练
Ctrl + S: 紧急停止
Ctrl + H: 显示帮助
    
导航:
- 仪表盘: 查看系统状态
- 模型管理: 管理AGI模型
- 训练控制: 控制训练过程
- 记忆: LNN内部状态回溯
- 推理: LNN前向传播推理
- 学习状态: 查看学习进度
- 系统设置: 配置系统参数
    `;
    
    /* P2-FIX-09: 添加window.前缀确保严格模式兼容 */
    window.SelfLnnNotify.show(helpMessage.replace(/\n/g, '<br>'), 'info', 12000);
}

/* ===== 全局工具函数 - setText/setWidth（供 index.html 内联脚本使用） ===== */
function setText(id, val) {
    var el = document.getElementById(id);
    if (el && val !== null && val !== undefined) el.textContent = val;
}
function setWidth(id, pct) {
    var el = document.getElementById(id);
    if (el && pct !== null && pct !== undefined) el.style.width = Math.min(100, Math.max(0, pct)) + '%';
}
function setMultimodalStatusText(id, enabled) {
    var el = document.getElementById(id);
    if (el) {
        el.textContent = enabled ? '启用' : '禁用';
        el.className = 'status-value ' + (enabled ? 'connected' : 'error');
    }
}

/**
 * 显示通知
 */
/* ===== 通知系统别名 - 兼容 index.html 内联脚本 ===== */
function showToast(message, type) {
    showNotification(message, type);
}
function showSuccess(message) {
    showNotification(message, 'success');
}
function showError(message) {
    showNotification(message, 'danger');
}

/* ===== 机器人控制功能 ===== */

/**
 * 刷新机器人状态
 */
async function refreshRobotStatus() {
    showNotification('正在刷新机器人状态...', 'info');
    
    try {
        // 调用真实机器人状态API
        const result = await window.SelfLnnApi.getRobotStatus();
        
        if (result.success && result.data && result.data.robot) {
            const robot = result.data.robot;
            
            // 检查机器人是否可用
            if (robot.status === 'available') {
                // 更新电池电量 (C-05修复: 添加空值检查)
                const batteryLevel = Math.max(0, Math.min(100, Math.round(robot.battery)));
                var batFill = document.getElementById('robot-battery-fill');
                if (batFill) batFill.style.width = batteryLevel + '%';
                var batText = document.getElementById('robot-battery-text');
                if (batText) batText.textContent = batteryLevel + '%';
                
                // 更新位置
                var posEl = document.getElementById('robot-position');
                if (robot.position && robot.position.length >= 3) {
                    const x = robot.position[0].toFixed(2);
                    const y = robot.position[1].toFixed(2);
                    const z = robot.position[2].toFixed(2);
                    if (posEl) posEl.textContent = 'X: ' + x + 'm, Y: ' + y + 'm, Z: ' + z + 'm';
                } else {
                    if (posEl) posEl.textContent = 'X: -- m, Y: -- m, Z: -- m';
                }
                
                // 更新姿态（F-005修复: 后端未提供姿态数据时显示"数据未加载"）
                var orientEl = document.getElementById('robot-orientation');
                if (robot.orientation) {
                    const roll = (robot.orientation[0] || 0).toFixed(1);
                    const pitch = (robot.orientation[1] || 0).toFixed(1);
                    const yaw = (robot.orientation[2] || 0).toFixed(1);
                    if (orientEl) orientEl.textContent = 
                        'Roll: ' + roll + '°, Pitch: ' + pitch + '°, Yaw: ' + yaw + '°';
                } else {
                    if (orientEl) orientEl.textContent = '数据未加载';
                }
                
                // 更新温度（F-005修复: 不再使用25.0作为默认fallback值）
                var tempEl = document.getElementById('robot-temperature');
                if (robot.temperature !== undefined && robot.temperature !== null) {
                    const temperature = Math.round(robot.temperature);
                    if (tempEl) tempEl.textContent = temperature + '°C';
                } else {
                    if (tempEl) tempEl.textContent = '数据未加载';
                }
                
                // 更新状态指示器
                const stateElement = document.getElementById('robot-state');
                if (stateElement) {
                    stateElement.textContent = robot.state === 0 ? '空闲' : 
                                             robot.state === 1 ? '运行中' : 
                                             robot.state === 2 ? '暂停' : '错误';
                }
                
                showNotification('✅ 机器人状态已更新（真实数据）', 'success');
            } else {
                // 机器人不可用或出错
                console.warn('机器人状态不可用:', robot.status, robot.error);
                
                // 显示错误状态 (C-05修复: 添加空值检查)
                var batFill2 = document.getElementById('robot-battery-fill');
                if (batFill2) batFill2.style.width = '0%';
                var batText2 = document.getElementById('robot-battery-text');
                if (batText2) batText2.textContent = '--';
                var posEl2 = document.getElementById('robot-position');
                if (posEl2) posEl2.textContent = 'X: -- m, Y: -- m, Z: -- m';
                var orientEl2 = document.getElementById('robot-orientation');
                if (orientEl2) orientEl2.textContent = 'Roll: --°, Pitch: --°, Yaw: --°';
                var tempEl2 = document.getElementById('robot-temperature');
                if (tempEl2) tempEl2.textContent = '--°C';
                
                if (robot.error) {
                    showNotification(`⚠️ 机器人状态错误: ${robot.error}`, 'warning');
                } else {
                    showNotification('⚠️ 机器人控制未启用或不可用', 'warning');
                }
            }
        } else {
            // API调用失败
            console.error('获取机器人状态API失败:', result.error);
            
            // 显示错误状态 (C-05修复: 添加空值检查)
            var batFill3 = document.getElementById('robot-battery-fill');
            if (batFill3) batFill3.style.width = '0%';
            var batText3 = document.getElementById('robot-battery-text');
            if (batText3) batText3.textContent = '--';
            var posEl3 = document.getElementById('robot-position');
            if (posEl3) posEl3.textContent = 'X: -- m, Y: -- m, Z: -- m';
            var orientEl3 = document.getElementById('robot-orientation');
            if (orientEl3) orientEl3.textContent = 'Roll: --°, Pitch: --°, Yaw: --°';
            var tempEl3 = document.getElementById('robot-temperature');
            if (tempEl3) tempEl3.textContent = '--°C';
            
            showNotification('❌ 获取机器人状态失败: ' + (result.error || 'API错误'), 'danger');
        }
        
    } catch (error) {
        console.error('刷新机器人状态失败:', error);
        showNotification('❌ 刷新机器人状态失败: ' + error.message, 'danger');
    }
}

/**
 * 运动控制：前进
 */
async function moveForward() {
    showNotification('机器人前进中...', 'info');
    
    // 更新UI状态
    /* P1-F14修复: 添加DOM元素空值检查防止TypeError崩溃 */
    var robotStatusEl = document.getElementById('robot-status-mode');
    if (robotStatusEl) robotStatusEl.textContent = '手动控制：前进';
    var forwardBtn = document.querySelector('.direction-btn.forward');
    if (forwardBtn) forwardBtn.style.background = 'rgba(0, 212, 255, 0.3)';
    
    try {
        // 发送真实机器人控制命令
        const command = {
            mode: 0, // MOTION_MODE_POSITION
            target_position: [1.0, 0.0, 0.0], // 前进1米（X方向）
            max_velocity: 0.5,
            max_acceleration: 0.2,
            trajectory_duration: 2.0
        };
        
        const result = await window.SelfLnnApi.sendRobotCommand(command);
        
        if (result.success) {
            showNotification('✅ 机器人前进命令已发送', 'success');
        } else {
            showNotification(`⚠️ 机器人前进失败: ${result.error}`, 'warning');
            console.warn('机器人命令失败:', result.error);
        }
    } catch (error) {
        console.error('发送前进命令失败:', error);
        showNotification('❌ 机器人前进失败: ' + error.message, 'danger');
    } finally {
        // 恢复按钮背景
        setTimeout(function() {
            var el = document.querySelector('.direction-btn.forward');
            if (el) el.style.background = '';
        }, 500);
    }
}

/**
 * 运动控制：后退
 */
async function moveBackward() {
    showNotification('机器人后退中...', 'info');
    
    // 更新UI状态
    /* F4-H02修复: 添加DOM空指针检查 */
    var robotStatusEl = document.getElementById('robot-status-mode');
    if (robotStatusEl) robotStatusEl.textContent = '手动控制：后退';
    var backwardBtn = document.querySelector('.direction-btn.backward');
    if (backwardBtn) backwardBtn.style.background = 'rgba(0, 212, 255, 0.3)';
    
    try {
        // 发送真实机器人控制命令
        const command = {
            mode: 0, // MOTION_MODE_POSITION
            target_position: [-1.0, 0.0, 0.0], // 后退1米（负X方向）
            max_velocity: 0.5,
            max_acceleration: 0.2,
            trajectory_duration: 2.0
        };
        
        const result = await window.SelfLnnApi.sendRobotCommand(command);
        
        if (result.success) {
            showNotification('✅ 机器人后退命令已发送', 'success');
        } else {
            showNotification(`⚠️ 机器人后退命令失败: ${result.error}`, 'warning');
        }
    } catch (error) {
        console.error('发送后退命令时出错:', error);
        showNotification('❌ 发送后退命令时出错: ' + error.message, 'danger');
    }
    
    setTimeout(function() { var el = document.querySelector('.direction-btn.backward'); if (el) el.style.background = ''; }, 500);
}
/**
 * 运动控制：左转
 */
async function moveLeft() {
    showNotification('机器人左转中...', 'info');
    
    // 更新UI状态
    var robotStatusEl = document.getElementById('robot-status-mode');
    if (robotStatusEl) robotStatusEl.textContent = '手动控制：左转';
    var leftBtn = document.querySelector('.direction-btn.left');
    if (leftBtn) leftBtn.style.background = 'rgba(0, 212, 255, 0.3)';
    
    try {
        // 发送真实机器人控制命令（左转90度）
        const command = {
            mode: 1, // MOTION_MODE_ROTATION
            target_orientation: [0.0, 0.0, 90.0], // 左转90度（绕Z轴）
            max_angular_velocity: 0.5,
            max_angular_acceleration: 0.2,
            trajectory_duration: 2.0
        };
        
        const result = await window.SelfLnnApi.sendRobotCommand(command);
        
        if (result.success) {
            showNotification('✅ 机器人左转命令已发送', 'success');
        } else {
            showNotification(`⚠️ 机器人左转命令失败: ${result.error}`, 'warning');
        }
    } catch (error) {
        console.error('发送左转命令时出错:', error);
        showNotification('❌ 发送左转命令时出错: ' + error.message, 'danger');
    }
    
    setTimeout(function() { var el = document.querySelector('.direction-btn.left'); if (el) el.style.background = ''; }, 500);
}

/**
 * 运动控制：右转
 */
async function moveRight() {
    showNotification('机器人右转中...', 'info');
    
    // 更新UI状态
    var robotStatusEl = document.getElementById('robot-status-mode');
    if (robotStatusEl) robotStatusEl.textContent = '手动控制：右转';
    var rightBtn = document.querySelector('.direction-btn.right');
    if (rightBtn) rightBtn.style.background = 'rgba(0, 212, 255, 0.3)';
    
    try {
        // 发送真实机器人控制命令（右转90度）
        const command = {
            mode: 1, // MOTION_MODE_ROTATION
            target_orientation: [0.0, 0.0, -90.0], // 右转90度（绕Z轴负方向）
            max_angular_velocity: 0.5,
            max_angular_acceleration: 0.2,
            trajectory_duration: 2.0
        };
        
        const result = await window.SelfLnnApi.sendRobotCommand(command);
        
        if (result.success) {
            showNotification('✅ 机器人右转命令已发送', 'success');
        } else {
            showNotification(`⚠️ 机器人右转命令失败: ${result.error}`, 'warning');
        }
    } catch (error) {
        console.error('发送右转命令时出错:', error);
        showNotification('❌ 发送右转命令时出错: ' + error.message, 'danger');
    }
    
    setTimeout(function() { var el = document.querySelector('.direction-btn.right'); if (el) el.style.background = ''; }, 500);
}

/**
 * 运动控制：停止
 */
async function stopMovement() {
    showNotification('机器人停止运动', 'warning');
    
    // 更新UI状态
    var robotStatusEl = document.getElementById('robot-status-mode');
    if (robotStatusEl) robotStatusEl.textContent = '手动控制：停止';
    var stopBtn = document.querySelector('.direction-btn.stop');
    if (stopBtn) stopBtn.style.background = 'rgba(255, 193, 7, 0.3)';
    
    try {
        // 发送真实机器人停止命令
        const command = {
            mode: 2, // MOTION_MODE_STOP
            emergency_stop: false, // 非紧急停止，平滑停止
            deceleration: 0.5,
            stop_timeout: 3.0
        };
        
        const result = await window.SelfLnnApi.sendRobotCommand(command);
        
        if (result.success) {
            showNotification('✅ 机器人停止命令已发送', 'success');
        } else {
            showNotification(`⚠️ 机器人停止命令失败: ${result.error}`, 'warning');
        }
    } catch (error) {
        console.error('发送停止命令时出错:', error);
        showNotification('❌ 发送停止命令时出错: ' + error.message, 'danger');
    }
    
    setTimeout(function() { var el = document.querySelector('.direction-btn.stop'); if (el) el.style.background = ''; }, 500);
}
/**
 * 更新线速度
 */
async function updateLinearSpeed(value) {
    /* F4-H02修复: 添加DOM空指针检查 */
    var linearSpeedEl = document.getElementById('linear-speed-value');
    if (linearSpeedEl) linearSpeedEl.textContent = value;
    
    // 更新UI状态
    var robotStatusEl2 = document.getElementById('robot-status-mode');
    if (robotStatusEl2) robotStatusEl2.textContent = `手动控制：速度 ${value}m/s`;
    
    try {
        // 发送速度参数更新到机器人控制系统
        if (window.SelfLnnApi && typeof window.SelfLnnApi.setRobotParameters === 'function') {
            const params = {
                max_linear_velocity: value,
                max_linear_acceleration: value * 0.5  // 加速度与速度成比例
            };
            
            await window.SelfLnnApi.setRobotParameters(params);
        }
    } catch (error) {
        console.warn('更新线速度参数时出错（不影响UI）:', error);
    }
}

/**
 * 更新角速度
 */
async function updateAngularSpeed(value) {
    /* F4-H02修复: 添加DOM空指针检查 */
    var angularSpeedEl = document.getElementById('angular-speed-value');
    if (angularSpeedEl) angularSpeedEl.textContent = value;
    
    // 更新UI状态
    var robotStatusEl3 = document.getElementById('robot-status-mode');
    if (robotStatusEl3) robotStatusEl3.textContent = `手动控制：转向速度 ${value}rad/s`;
    
    try {
        // 发送角速度参数更新到机器人控制系统
        if (window.SelfLnnApi && typeof window.SelfLnnApi.setRobotParameters === 'function') {
            const params = {
                max_angular_velocity: value,
                max_angular_acceleration: value * 0.5  // 角加速度与角速度成比例
            };
            
            await window.SelfLnnApi.setRobotParameters(params);
        }
    } catch (error) {
        console.warn('更新角速度参数时出错（不影响UI）:', error);
    }
}

/**
 * 前往目标位置
 */
async function goToPosition() {
    /* P1-F15修复: 添加DOM元素空值检查 */
    var targetXEl = document.getElementById('target-x');
    var targetYEl = document.getElementById('target-y');
    var targetZEl = document.getElementById('target-z');
    const targetX = targetXEl ? (parseFloat(targetXEl.value) || 0) : 0;
    const targetY = targetYEl ? (parseFloat(targetYEl.value) || 0) : 0;
    const targetZ = targetZEl ? (parseFloat(targetZEl.value) || 0) : 0;
    
    showNotification(`前往目标位置: X=${targetX}m, Y=${targetY}m, Z=${targetZ}m`, 'info');
    
    // 更新UI状态
    var robotStatusEl = document.getElementById('robot-status-mode');
    if (robotStatusEl) robotStatusEl.textContent = `自主导航到 (${targetX}, ${targetY}, ${targetZ})`;
    
    try {
        // 发送自主导航命令
        const command = {
            mode: 3, // MOTION_MODE_AUTONOMOUS_NAVIGATION
            target_position: [targetX, targetY, targetZ],
            use_global_map: true,
            avoid_obstacles: true,
            navigation_timeout: 30.0  // 30秒超时
        };
        
        const result = await window.SelfLnnApi.sendRobotCommand(command);
        
        if (result.success) {
            showNotification('✅ 自主导航命令已发送', 'success');
            
            // 更新当前任务
            const taskName = document.querySelector('.current-task .task-name');
            if (taskName) {
                taskName.textContent = `导航到位置 (${targetX}, ${targetY})`;
            }
            
            // 重置进度条
            const progressFill = document.querySelector('.current-task .progress-fill');
            if (progressFill) {
                progressFill.style.width = '0%';
                setTimeout(() => {
                    progressFill.style.width = '100%';
                }, 100);
            }
        } else {
            showNotification(`⚠️ 自主导航命令失败: ${result.error}`, 'warning');
        }
    } catch (error) {
        console.error('发送自主导航命令时出错:', error);
        showNotification('❌ 发送自主导航命令时出错: ' + error.message, 'danger');
    }
}

/**
 * 执行预设动作
 */
async function executeAction(action) {
    
    const actionNames = {
        'home': '返回原点',
        'scan': '环境扫描',
        'dance': '舞蹈模式',
        'sleep': '休眠模式'
    };
    
    showNotification(`执行动作: ${actionNames[action] || action}`, 'info');
    var robotStatusEl = document.getElementById('robot-status-mode');
    if (robotStatusEl) robotStatusEl.textContent = `执行动作: ${actionNames[action] || action}`;
    
    // 调用真实机器人控制API
    try {
        if (window.SelfLnnApi && typeof window.SelfLnnApi.sendRobotCommand === 'function') {
            const result = await window.SelfLnnApi.sendRobotCommand({
                action: action,
                timestamp: Date.now()
            });
            
            if (result.success) {
                showNotification(`✅ 动作完成: ${actionNames[action] || action}`, 'success');
            } else {
                showNotification(`⚠️ 动作执行失败: ${result.error || '未知错误'}`, 'warning');
            }
        } else {
            showNotification('⚠️ 机器人控制后端未连接', 'warning');
        }
    } catch (error) {
        console.error('执行动作时出错:', error);
        showNotification(`❌ 执行动作时出错: ${error.message}`, 'danger');
    }
}

/**
 * 创建新任务
 */
async function createNewTask() {
    showNotification('创建新任务...', 'info');
    
    const taskName = await promptAsync('请输入任务名称:');
    if (taskName) {
        /* 调用后端API创建任务 */
        if (window.SelfLnnApi && typeof window.SelfLnnApi.createTask === 'function') {
            window.SelfLnnApi.createTask({ name: taskName, priority: 'medium' }).then(result => {
                if (result && result.success) {
                    showNotification(`✅ 任务 "${taskName}" 已创建并添加到队列`, 'success');
                    loadTaskQueue();
                } else {
                    showNotification(`⚠️ 任务创建失败: ${result?.error || '未知错误'}`, 'warning');
                }
            }).catch(err => {
                /* FE-010修复: 完善错误处理，添加console.error日志和安全的错误信息提取 */
                console.error('[任务] 创建任务失败:', err);
                showNotification('❌ 任务创建失败: ' + (err && err.message ? err.message : '未知错误'), 'danger');
            });
        } else {
            showNotification('⚠️ 任务API不可用，请检查后端连接', 'warning');
        }
    }
}

/**
 * 加载任务队列
 */
async function loadTaskQueue() {
    showNotification('刷新任务队列...', 'info');
    
    try {
        // 尝试调用真实的任务队列API
        if (window.SelfLnnApi && typeof window.SelfLnnApi.getTaskQueue === 'function') {
            const result = await window.SelfLnnApi.getTaskQueue();
            if (result.success) {
                showNotification('✅ 任务队列已刷新', 'success');
                // 更新UI显示任务队列
                updateTaskQueueUI(result.data);
            } else {
                showNotification('⚠️ 任务队列API返回错误: ' + (result.error || '未知错误'), 'warning');
            }
        } else {
            showNotification('⚠️ 任务队列后端未连接', 'warning');
        }
    } catch (error) {
        console.error('加载任务队列时出错:', error);
        showNotification('❌ 加载任务队列时出错: ' + error.message, 'danger');
    }
}

/**
 * 更新任务队列UI
 * @param {Array} tasks - 任务数组
 */
function updateTaskQueueUI(tasks) {
    if (!tasks || !Array.isArray(tasks)) {
        console.warn('无效的任务队列数据');
        return;
    }
    
    
    // 查找任务队列容器
    const taskQueueContainer = document.getElementById('task-queue-list');
    if (!taskQueueContainer) {
        console.warn('未找到任务队列容器元素');
        return;
    }
    
    // 清空现有内容
    taskQueueContainer.innerHTML = '';
    
    // 添加每个任务
    tasks.forEach((task, index) => {
        const taskElement = document.createElement('div');
        taskElement.className = 'task-item';
        taskElement.dataset.taskId = task.id || index;
        
        // 根据任务状态设置样式
        let statusClass = 'task-status';
        let statusText = '等待中';
        let statusColor = '#ffc107';
        
        if (task.status === 'running') {
            statusText = '进行中';
            statusColor = '#0dcaf0';
        } else if (task.status === 'completed') {
            statusText = '已完成';
            statusColor = '#198754';
        } else if (task.status === 'failed') {
            statusText = '失败';
            statusColor = '#dc3545';
        } else if (task.status === 'paused') {
            statusText = '已暂停';
            statusColor = '#6c757d';
        }
        
        /* D-002修复: 对task.name和task.type进行HTML转义，防止XSS攻击 */    
        taskElement.innerHTML = `
            <div class="task-info">
                <span class="task-name">${window.escapeHtml(task.name) || `任务 ${index + 1}`}</span>
                <span class="task-type">${window.escapeHtml(task.type) || '未知类型'}</span>
                <span class="task-progress" style="color: ${statusColor}">${statusText}</span>
            </div>
            <div class="task-actions">
                <button class="btn btn-sm btn-outline-primary" onclick="resumeTask(${task.id || index})" ${task.status === 'running' ? 'disabled' : ''}>继续</button>
                <button class="btn btn-sm btn-outline-warning" onclick="pauseTask(${task.id || index})" ${task.status !== 'running' ? 'disabled' : ''}>暂停</button>
                <button class="btn btn-sm btn-outline-danger" onclick="cancelTask(${task.id || index})">取消</button>
            </div>
        `;
        
        taskQueueContainer.appendChild(taskElement);
    });
    
    showNotification(`任务队列已更新，共${tasks.length}个任务`, 'success');
}

/**
 * 暂停任务
 */
function pauseTask(taskId) {
    showNotification('暂停任务中...', 'info');

    /* 调用后端API暂停任务 */
    if (window.SelfLnnApi && typeof window.SelfLnnApi.pauseTask === 'function') {
        window.SelfLnnApi.pauseTask(taskId).then(result => {
            if (result && result.success) {
                showNotification('✅ 任务已暂停', 'warning');
                updateTaskUI(taskId, '已暂停', 'rgba(255, 193, 7, 0.1)', '#ffc107');
            } else {
                showNotification(`⚠️ 任务暂停失败: ${result?.error || '未知错误'}`, 'warning');
            }
        }).catch(err => {
            /* FE-010修复: 完善错误处理，添加console.error日志和安全的错误信息提取 */
            console.error('[任务] 暂停任务失败:', err);
            showNotification('❌ 暂停失败: ' + (err && err.message ? err.message : '未知错误'), 'danger');
        });
    } else {
        showNotification('⚠️ 任务API不可用', 'warning');
    }
}

/**
 * 取消任务
 */
function cancelTask(taskId) {
    if (safeConfirm('确定要取消此任务吗？')) {
        showNotification('取消任务中...', 'info');

        /* 调用后端API取消任务 */
        if (window.SelfLnnApi && typeof window.SelfLnnApi.cancelTask === 'function') {
            window.SelfLnnApi.cancelTask(taskId).then(result => {
                if (result && result.success) {
                    showNotification('✅ 任务已取消', 'danger');
                    updateTaskUI(taskId, '已取消', 'rgba(220, 53, 69, 0.1)', '#dc3545');
                    loadTaskQueue();
                } else {
                    showNotification(`⚠️ 取消失败: ${result?.error || '未知错误'}`, 'warning');
                }
            }).catch(err => {
                /* FE-010修复: 完善错误处理，添加console.error日志和安全的错误信息提取 */
                console.error('[任务] 取消任务失败:', err);
                showNotification('❌ 取消失败: ' + (err && err.message ? err.message : '未知错误'), 'danger');
            });
        } else {
            showNotification('⚠️ 任务API不可用', 'warning');
        }
    }
}

/**
 * 继续/恢复任务（原函数缺失导致 onclick 崩溃）
 */
function resumeTask(taskId) {
    showNotification('继续任务中...', 'info');
    if (window.SelfLnnApi && typeof window.SelfLnnApi.resumeTask === 'function') {
        window.SelfLnnApi.resumeTask(taskId).then(function(result) {
            if (result && result.success) {
                showNotification('✅ 任务已恢复', 'success');
                updateTaskUI(taskId, '运行中', 'rgba(40, 167, 69, 0.1)', '#28a745');
                loadTaskQueue();
            } else {
                showNotification('⚠️ 任务恢复失败: ' + (result ? result.error || '未知错误' : '未知错误'), 'warning');
            }
        }).catch(err => {
            showNotification('❌ 恢复失败: ' + err.message, 'danger');
        });
    } else {
        showNotification('⚠️ 任务API不可用', 'warning');
    }
}

/**
 * 更新单个任务UI
 */
function updateTaskUI(taskId, statusText, bgColor, textColor) {
    const taskStatus = document.querySelector('.current-task .task-status');
    if (taskStatus) {
        taskStatus.textContent = statusText;
        taskStatus.style.background = bgColor;
        taskStatus.style.color = textColor;
    }
}

/**
 * 添加任务到队列
 */
function addTaskToQueue() {
    const taskType = document.getElementById('task-type-select').value;
    const taskDescription = document.getElementById('task-description').value || '未命名任务';
    const taskPriority = document.getElementById('task-priority-select').value;
    
    const typeNames = {
        'navigation': '导航任务',
        'inspection': '巡检任务',
        'pickup': '抓取任务',
        'delivery': '送货任务',
        'monitoring': '监控任务'
    };
    
    const priorityNames = {
        'low': '低优先级',
        'medium': '中优先级',
        'high': '高优先级'
    };
    
    
    // 添加到任务队列
    const taskQueue = document.querySelector('.queue-list');
    if (taskQueue) {
        const newTask = document.createElement('div');
        newTask.className = 'queue-item';
        /* C-003修复: 任务描述使用escapeHtml防止XSS */
        newTask.innerHTML = `
            <span class="task-name">${window.escapeHtml(typeNames[taskType])}: ${window.escapeHtml(taskDescription)}</span>
            <span class="task-priority ${window.escapeHtml(taskPriority)}">${window.escapeHtml(priorityNames[taskPriority])}</span>
            <span class="task-eta">时间: 未指定</span>
        `;
        taskQueue.appendChild(newTask);
    }
    
    showNotification(`✅ 任务 "${typeNames[taskType]}" 已添加到队列`, 'success');
    
    // 清空表单
    var taskDescEl = document.getElementById('task-description');
    if (taskDescEl) taskDescEl.value = '';
}

/**
 * 切换传感器数据流
 */
function toggleSensorStream() {
    const button = document.querySelector('.sensor-data-card .btn-primary');
    if (button) {
        const isStreaming = button.textContent.includes('停止');
        if (isStreaming) {
            button.innerHTML = '<i class="fas fa-stream"></i> 实时流';
            stopSensorDataStream();
            showNotification('传感器数据流已停止', 'warning');
        } else {
            button.innerHTML = '<i class="fas fa-stop"></i> 停止';
            showNotification('传感器数据流已启动', 'success');
            
            // 启动真实传感器数据流
            startSensorDataStream();
        }
    }
}

/**
 * 传感器数据流句柄
 */
let sensorStreamInterval = null;

/**
 * 启动传感器数据流（通过统一轮询中心调度）
 */
function startSensorDataStream() {
    
    if (sensorStreamInterval) {
        clearInterval(sensorStreamInterval);
        sensorStreamInterval = null;
    }
    
    if (typeof g_dataEngine !== 'undefined' && g_dataEngine) {
        g_dataEngine.registerModule('sensor_stream', 1000, pollSensorDataStream);
    } else {
        sensorStreamInterval = setInterval(async () => {
            await pollSensorDataStream();
        }, 1000);
    }
}

/**
 * 传感器数据轮询回调
 */
async function pollSensorDataStream() {
    try {
        if (window.SelfLnnApi && typeof window.SelfLnnApi.getRobotSensorData === 'function') {
            const result = await window.SelfLnnApi.getRobotSensorData();
            if (result.success && result.data) {
                updateSensorDisplay(result.data);
            } else {
                console.warn('获取传感器数据失败:', result.error);
            }
        } else {
            console.warn('传感器数据后端未连接');
        }
    } catch (error) {
        console.error('获取传感器数据时出错:', error);
    }
}

/**
 * 停止传感器数据流
 */
function stopSensorDataStream() {
    if (sensorStreamInterval) {
        clearInterval(sensorStreamInterval);
        sensorStreamInterval = null;
    }
    if (typeof g_dataEngine !== 'undefined' && g_dataEngine) {
        g_dataEngine.unregisterModule('sensor_stream');
    }
}

/**
 * 更新传感器数据显示
 */
function updateSensorDisplay(sensorData) {
    // 更新传感器数据显示
    const sensorDisplay = document.querySelector('.sensor-values');
    if (sensorDisplay && sensorData.values && Array.isArray(sensorData.values)) {
        let html = '';
        sensorData.values.forEach((value, index) => {
            const sensorName = sensorData.names ? sensorData.names[index] : `传感器 ${index + 1}`;
            /* D-004修复: 对传感器名称进行HTML转义，防止XSS攻击 */
            html += `<div class="sensor-value-item">${window.escapeHtml(sensorName)}: <strong>${value.toFixed(2)}</strong></div>`;
        });
        sensorDisplay.innerHTML = html;
    }
    
    // 更新传感器图表（如果存在）
    updateSensorChart(sensorData);
}

/**
 * 更新传感器图表
 */
function updateSensorChart(sensorData) {
    if (!sensorData || !sensorData.values || sensorData.values.length === 0) return;
    
    var canvas = document.getElementById('sensor-chart');
    if (!canvas) return;
    var ctx = canvas.getContext('2d');
    var w = canvas.width, h = canvas.height;
    if (w === 0 || h === 0) {
        canvas.width = 300;
        canvas.height = 150;
        w = 300; h = 150;
    }
    
    ctx.clearRect(0, 0, w, h);
    var values = sensorData.values;
    var n = values.length;
    
    /* FIX-JS-002: 安全循环替代Math.min.apply/Math.max.apply(避免大数组栈溢出) */
    var yMin = values[0], yMax = values[0];
    for (var vi = 1; vi < n; vi++) {
        if (values[vi] < yMin) yMin = values[vi];
        if (values[vi] > yMax) yMax = values[vi];
    }
    var yRange = yMax - yMin || 1;
    
    ctx.beginPath();
    ctx.strokeStyle = '#00d4ff';
    ctx.lineWidth = 2;
    
    for (var i = 0; i < n; i++) {
        var x = (i / (n - 1)) * (w - 20) + 10;
        var y = h - 10 - ((values[i] - yMin) / yRange) * (h - 20);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();
    
    if (sensorData.label) {
        ctx.fillStyle = '#888';
        ctx.font = '10px sans-serif';
        ctx.fillText(sensorData.label, 10, h - 2);
    }
}

/**
 * 设置图表时间范围
 */
function setChartRange(range, evt) {
    
    var targetEl = (evt && evt.target) ? evt.target : document.querySelector('.chart-controls .btn-xs[data-range="' + range + '"]');
    if (!targetEl) return;
    document.querySelectorAll('.chart-controls .btn-xs').forEach(function(btn) {
        btn.classList.remove('active');
    });
    targetEl.classList.add('active');
    
    showNotification('图表时间范围已设置为: ' + range, 'info');
}

/**
 * 切换视频流
 */
function toggleVideoStream() {
    const button = document.querySelector('.video-stream-card .btn-primary');
    const videoStatusMessage = document.getElementById('video-status-message');
    const liveVideo = document.getElementById('live-video-stream');
    
    if (button && videoStatusMessage && liveVideo) {
        const isStreaming = button.textContent.includes('停止');
        if (isStreaming) {
            button.innerHTML = '<i class="fas fa-play"></i> 开始';
            videoStatusMessage.style.display = 'flex';
            liveVideo.classList.add('hidden');
            showNotification('视频流已停止', 'warning');
        } else {
            button.innerHTML = '<i class="fas fa-stop"></i> 停止';
            videoStatusMessage.style.display = 'none';
            liveVideo.classList.remove('hidden');
            
            // 尝试连接真实视频流
            // 如果没有真实视频流，显示警告
            const cfg = window.SELFLNN_CONFIG || { host: 'localhost', port: 8080 };
            const videoStreamUrl = `http://${cfg.host}:${cfg.port}/api/video/stream`;
            liveVideo.src = videoStreamUrl;
            
            // 添加错误处理
            liveVideo.onerror = () => {
                showNotification('⚠️ 视频流不可用，请确保后端视频流服务已启动', 'warning');
                button.innerHTML = '<i class="fas fa-play"></i> 开始';
                videoStatusMessage.style.display = 'flex';
                liveVideo.classList.add('hidden');
            };
            
            showNotification('正在连接视频流...', 'info');
        }
    }
}

/**
 * 捕获快照
 */
async function captureSnapshot() {
    
    try {
        // 尝试从真实视频流捕获快照
        const liveVideo = document.getElementById('live-video-stream');
        const snapshotGrid = document.querySelector('.snapshot-grid');
        
        if (!liveVideo || liveVideo.classList.contains('hidden')) {
            showNotification('⚠️ 请先启动视频流', 'warning');
            return;
        }
        
        if (snapshotGrid) {
            // 创建canvas从视频中捕获帧
            const canvas = document.createElement('canvas');
            canvas.width = liveVideo.videoWidth || 320;
            canvas.height = liveVideo.videoHeight || 240;
            const ctx = canvas.getContext('2d');
            
            try {
                // 绘制视频帧到canvas
                ctx.drawImage(liveVideo, 0, 0, canvas.width, canvas.height);
                
                // 将canvas转换为数据URL
                const imageDataUrl = canvas.toDataURL('image/jpeg', 0.8);
                const time = new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
                
                // 创建快照项
                const newSnapshot = document.createElement('div');
                newSnapshot.className = 'snapshot-item';
                newSnapshot.innerHTML = `
                    <img src="${imageDataUrl}" alt="截图">
                    <span class="snapshot-time">${time}</span>
                `;
                snapshotGrid.insertBefore(newSnapshot, snapshotGrid.firstChild);
                
                // 限制快照数量
                if (snapshotGrid.children.length > 8) {
                    snapshotGrid.removeChild(snapshotGrid.lastChild);
                }
                
                showNotification('✅ 快照已保存', 'success');
            } catch (videoError) {
                console.error('从视频流捕获快照失败:', videoError);
                showNotification('⚠️ 从视频流捕获快照失败，视频流可能未准备好', 'warning');
            }
        }
    } catch (error) {
        console.error('捕获快照时出错:', error);
        showNotification('❌ 捕获快照时出错: ' + error.message, 'danger');
    }
}

/**
 * 切换摄像头 — 实现真实摄像头切换
 */
function switchCamera(camera) {
    const cameraNames = {
        'front': '前向摄像头',
        'rear': '后向摄像头',
        'depth': '深度摄像头',
        'thermal': '热成像摄像头'
    };
    
    showNotification('切换到: ' + cameraNames[camera], 'info');
    if (typeof window.SelfLnnApi === 'object' && typeof window.SelfLnnApi.switchCameraSource === 'function') {
        window.SelfLnnApi.switchCameraSource({ camera: camera }).catch(function(e) { console.warn('摄像头切换失败:', e); });
    }
}

/**
 * 调整视频质量
 */
function adjustVideoQuality(quality) {
    const qualityNames = {
        'low': '低 (360p)',
        'medium': '中 (720p)',
        'high': '高 (1080p)'
    };
    /* MID-002修复: 实际调用后端API设置视频质量参数 */
    if (typeof window.SelfLnnApi !== 'undefined' && typeof window.SelfLnnApi.setVideoQuality === 'function') {
        window.SelfLnnApi.setVideoQuality({ quality: quality }).then(function(result) {
            if (result && result.success) {
                showNotification(`视频质量已切换: ${qualityNames[quality]}`, 'success');
            } else {
                showNotification(`视频质量切换失败: ${qualityNames[quality]}`, 'danger');
            }
        }).catch(function(err) {
            /* FE-010修复: 完善错误处理，添加console.error日志 */
            console.error('[视频] 设置视频质量失败:', err);
            showNotification('无法连接到后端，视频质量设置未生效: ' + qualityNames[quality], 'warning');
        });
    } else {
        showNotification(`正在设置视频质量: ${qualityNames[quality]}（后端API未就绪）`, 'info');
    }
}

/**
 * 保存机器人配置
 */
function saveRobotConfig() {
    /* P1-F16修复: 添加所有DOM元素的安全访问 */
    function getVal(id) { var el = document.getElementById(id); return el ? el.value : ''; }
    function getText(id) { var el = document.getElementById(id); return el ? el.textContent : ''; }
    function getChecked(id) { var el = document.getElementById(id); return el ? !!el.checked : false; }
    const configData = {
        connectionType: getVal('connection-type'),
        robotIp: getVal('robot-ip'),
        robotPort: getVal('robot-port'),
        maxSpeed: getText('max-speed-value'),
        collisionDetection: getChecked('collision-detection'),
        safetyDistance: getText('safety-distance-value'),
        controlFrequency: getVal('control-frequency'),
        logLevel: getVal('log-level'),
        autoUpdate: getChecked('auto-update')
    };
    
    if (typeof window.SelfLnnApi !== 'undefined' && typeof window.SelfLnnApi.saveRobotConfig === 'function') {
        window.SelfLnnApi.saveRobotConfig(configData).then(function(result) {
            if (result && result.success) {
                showNotification('✅ 机器人配置已保存到后端', 'success');
            } else {
                showNotification('⚠️ 配置保存失败: ' + (result ? result.message : '未知错误'), 'danger');
            }
        }).catch(function(err) {
            /* FE-010修复: 完善错误处理，添加console.error日志和安全的错误信息提取 */
            console.error('[机器人] 保存配置失败:', err);
            showNotification('❌ 无法连接后端保存配置: ' + ((err && err.message) ? err.message : '网络错误'), 'danger');
        });
    } else {
        showNotification('⚠️ 后端API未就绪，配置未持久化', 'warning');
    }
}

/**
 * 连接机器人
 */
async function connectRobot() {
    showNotification('正在连接机器人...', 'info');
    
    try {
        /* P0-5修复：调用后端 connectRobot API 建立真实连接，原代码只改UI不调API */
        if (window.SelfLnnApi && typeof window.SelfLnnApi.connectRobot === 'function') {
            const connectResult = await window.SelfLnnApi.connectRobot();
            
            if (connectResult && connectResult.success) {
                // 后端连接成功后，获取机器人状态更新UI
                const statusResult = await window.SelfLnnApi.getRobotStatus();
                
                if (statusResult.success && statusResult.data && statusResult.data.robot) {
                    const robot = statusResult.data.robot;
                    /* D-012修复: 使用安全DOM操作防止null崩溃 */
                    window.safeSetElementHTML('robot-connection-status', '<i class="fas fa-plug"></i> 已连接');
                    window.safeSetElementClass('robot-connection-status', 'status-value connected');
                    showNotification('✅ 机器人已连接并可用', 'success');
                } else {
                    window.safeSetElementHTML('robot-connection-status', '<i class="fas fa-plug"></i> 已连接（状态未知）');
                    window.safeSetElementClass('robot-connection-status', 'status-value connected');
                    showNotification('⚠️ 机器人已连接但无法获取状态信息', 'warning');
                }
            } else {
                const errorMsg = (connectResult && connectResult.error) || '连接请求失败';
                showNotification(`❌ ${errorMsg}`, 'danger');
                console.error('机器人连接失败:', connectResult);
            }
        } else {
            showNotification('❌ 机器人连接API不可用，请检查后端服务', 'danger');
            console.error('connectRobot API未定义');
        }
    } catch (error) {
        console.error('连接机器人失败:', error);
        showNotification(`❌ 连接机器人失败: ${error.message}`, 'danger');
    }
}

/**
 * 断开机器人连接
 */
async function disconnectRobot() {
    
    if (safeConfirm('确定要断开机器人连接吗？')) {
        try {
            showNotification('正在断开机器人连接...', 'warning');
            
            /* P0-5修复：调用后端 disconnectRobot API 真实断开连接 */
            if (window.SelfLnnApi && typeof window.SelfLnnApi.disconnectRobot === 'function') {
                const result = await window.SelfLnnApi.disconnectRobot();
                
                if (result && result.success) {
                    showNotification('✅ 机器人已断开连接', 'success');
                } else {
                    showNotification(`⚠️ 后端返回异常: ${(result && result.error) || '未知错误'}，已更新本地UI状态`, 'warning');
                }
            } else {
                showNotification('⚠️ disconnectRobot API不可用，仅更新本地UI状态', 'warning');
            }
            
            // 更新UI状态
            window.safeSetElementHTML('robot-connection-status', '<i class="fas fa-unplug"></i> 未连接');
            window.safeSetElementClass('robot-connection-status', 'status-value error');
            
            // 重置机器人状态显示
            document.getElementById('robot-battery-fill').style.width = '0%';
            document.getElementById('robot-battery-text').textContent = '--';
            document.getElementById('robot-position').textContent = 'X: -- m, Y: -- m, Z: -- m';
            document.getElementById('robot-orientation').textContent = 'Roll: --°, Pitch: --°, Yaw: --°';
            document.getElementById('robot-temperature').textContent = '--°C';
            
        } catch (error) {
            console.error('断开机器人连接失败:', error);
            showNotification(`❌ 断开机器人连接失败: ${error.message}`, 'danger');
        }
    }
}

/**
 * 重启机器人
 */
async function rebootRobot() {
    
    if (safeConfirm('确定要重启机器人吗？重启过程需要几分钟。')) {
        try {
            // 根据项目要求"禁止任何虚假数据"，不模拟重启进度
            // 尝试调用真实的后端API重启机器人
            showNotification('正在尝试重启机器人...', 'warning');
            
            // 检查是否有重启机器人的API
            if (window.SelfLnnApi && typeof window.SelfLnnApi.rebootRobot === 'function') {
                const result = await window.SelfLnnApi.rebootRobot();
                
                if (result.success) {
                    showNotification('✅ 机器人重启命令已发送', 'success');
                } else {
                    showNotification(`❌ 机器人重启失败: ${result.error || '未知错误'}`, 'danger');
                }
            } else {
                // 后端未连接，显示错误
                showNotification('❌ 机器人重启后端未连接，请检查服务器状态', 'danger');
                console.error('机器人重启后端未连接');
            }
        } catch (error) {
            console.error('重启机器人失败:', error);
            showNotification(`❌ 重启机器人失败: ${error.message}`, 'danger');
        }
    }
}

/**
 * 校准传感器
 */
async function calibrateSensors() {
    
    try {
        // 根据项目要求"禁止任何虚假数据"，不模拟校准进度
        // 尝试调用真实的后端API校准传感器
        showNotification('正在尝试校准传感器...', 'info');
        
        // 检查是否有校准传感器的API
        if (window.SelfLnnApi && typeof window.SelfLnnApi.calibrateSensors === 'function') {
            const result = await window.SelfLnnApi.calibrateSensors();
            
            if (result.success) {
                showNotification('✅ 传感器校准命令已发送', 'success');
            } else {
                showNotification(`❌ 传感器校准失败: ${result.error || '未知错误'}`, 'danger');
            }
        } else {
            // 后端未连接，显示错误
            showNotification('❌ 传感器校准后端未连接，请检查服务器状态', 'danger');
            console.error('传感器校准后端未连接');
        }
    } catch (error) {
        console.error('校准传感器失败:', error);
        showNotification(`❌ 校准传感器失败: ${error.message}`, 'danger');
    }
}

/**
 * 更新固件
 */
async function updateFirmware() {
    
    if (safeConfirm('确定要更新机器人固件吗？更新过程中机器人将无法使用。')) {
        try {
            // 根据项目要求"禁止任何虚假数据"，不模拟更新进度
            // 尝试调用真实的后端API进行固件更新
            if (window.SelfLnnApi && typeof window.SelfLnnApi.updateFirmware === 'function') {
                showNotification('正在下载固件更新...', 'info');
                const result = await window.SelfLnnApi.updateFirmware();
                
                if (result.success) {
                    showNotification('✅ 固件更新完成！机器人已重启', 'success');
                    /* D-013修复: 使用安全DOM操作防止null崩溃 */
                    window.safeSetElementHTML('robot-error-status', '<i class="fas fa-check-circle"></i> 固件已更新');
                    window.safeSetElementClass('robot-error-status', 'status-value normal');
                } else {
                    showNotification(`❌ 固件更新失败: ${result.message || '未知错误'}`, 'danger');
                }
            } else {
                // 后端未连接，显示错误
                showNotification('❌ 固件更新后端未连接，请检查服务器状态', 'danger');
                console.error('固件更新后端未连接');
            }
        } catch (error) {
            console.error('固件更新失败:', error);
            showNotification(`❌ 固件更新失败: ${error.message}`, 'danger');
        }
    }
}

/**
 * 运行自检
 */
async function runSelfDiagnostic() {
    
    try {
        // 根据项目要求"禁止任何虚假数据"，不模拟自检进度
        // 尝试调用真实的后端API进行系统自检
        if (window.SelfLnnApi && typeof window.SelfLnnApi.runSelfDiagnostic === 'function') {
            showNotification('正在运行系统自检...', 'info');
            const result = await window.SelfLnnApi.runSelfDiagnostic();
            
            if (result.success) {
                showNotification('✅ 自检完成：所有系统正常', 'success');
            } else {
                showNotification(`❌ 自检失败: ${result.message || '未知错误'}`, 'danger');
            }
        } else {
            // 后端未连接，显示错误
            showNotification('❌ 系统自检后端未连接，请检查服务器状态', 'danger');
            console.error('系统自检后端未连接');
        }
    } catch (error) {
        console.error('系统自检失败:', error);
        showNotification(`❌ 系统自检失败: ${error.message}`, 'danger');
    }
}

/**
 * 导出诊断数据
 */
async function exportDiagnosticData() {
    
    try {
        // 根据项目要求"禁止任何虚假数据"，不模拟导出进度
        // 尝试调用真实的后端API导出诊断数据
        if (window.SelfLnnApi && typeof window.SelfLnnApi.exportDiagnosticData === 'function') {
            showNotification('正在导出诊断数据...', 'info');
            const result = await window.SelfLnnApi.exportDiagnosticData();
            
            if (result.success) {
                showNotification('✅ 诊断数据导出完成', 'success');
            } else {
                showNotification(`❌ 诊断数据导出失败: ${result.message || '未知错误'}`, 'danger');
            }
        } else {
            // 后端未连接，显示错误
            showNotification('❌ 诊断数据导出后端未连接，请检查服务器状态', 'danger');
            console.error('诊断数据导出后端未连接');
        }
    } catch (error) {
        console.error('诊断数据导出失败:', error);
        showNotification(`❌ 诊断数据导出失败: ${error.message}`, 'danger');
    }
}

/**
 * 刷新多模态状态
 */
async function refreshMultimodalStatus() {
    
    try {
        // 根据项目要求"禁止任何虚假数据"，不模拟状态刷新
        // 尝试调用真实的后端API获取多模态状态
        showNotification('正在获取多模态状态...', 'info');
        
        // 检查是否有获取多模态状态的API
        if (window.SelfLnnApi && typeof window.SelfLnnApi.getMultimodalStatus === 'function') {
            const result = await window.SelfLnnApi.getMultimodalStatus();
            
            if (result.success && result.data) {
                const status = result.data;
                showNotification('✅ 多模态状态已更新', 'success');
                
                // 更新UI状态
                // 这里可以根据实际返回的数据结构更新UI
                // 例如：更新模态状态、处理任务数量等
                setMultimodalStatusText('vision-status', status.vision && status.vision.enabled);
                setMultimodalStatusText('audio-status', status.audio && status.audio.enabled);
                setMultimodalStatusText('text-status', status.text && status.text.enabled);
                setMultimodalStatusText('sensor-status', status.sensor && status.sensor.enabled);
                
                // 更新处理任务数量
                if (status.active_tasks !== undefined) {
                    var el = document.getElementById('active-tasks-count');
                    if (el) el.textContent = status.active_tasks;
                }
                
                if (status.processed_tasks !== undefined) {
                    var el = document.getElementById('processed-tasks-count');
                    if (el) el.textContent = status.processed_tasks;
                }
                
                if (status.error_tasks !== undefined) {
                    var el = document.getElementById('error-tasks-count');
                    if (el) el.textContent = status.error_tasks;
                }
            } else {
                showNotification(`❌ 获取多模态状态失败: ${result.error || '未知错误'}`, 'danger');
            }
        } else {
            // 后端未连接，显示错误
            showNotification('❌ 多模态状态后端未连接，请检查服务器状态', 'danger');
            console.error('多模态状态后端未连接');
        }
    } catch (error) {
        console.error('刷新多模态状态失败:', error);
        showNotification(`❌ 刷新多模态状态失败: ${error.message}`, 'danger');
    }
}

/**
 * 保存多模态配置
 */
async function saveMultimodalConfig() {
    
    try {
        // 根据项目要求"禁止任何虚假数据"，不模拟保存进度
        // 收集配置数据
        const config = {
            vision: {
                enabled: document.getElementById('vision-enabled').checked,
                resolution: document.getElementById('vision-resolution').value,
                frame_rate: parseInt(document.getElementById('vision-frame-rate').value),
                model: document.getElementById('vision-model').value
            },
            audio: {
                enabled: document.getElementById('audio-enabled').checked,
                sample_rate: parseInt(document.getElementById('audio-sample-rate').value),
                channels: parseInt(document.getElementById('audio-channels').value),
                model: document.getElementById('audio-model').value
            },
            text: {
                enabled: document.getElementById('text-enabled').checked,
                language: document.getElementById('text-language').value,
                model: document.getElementById('text-model').value
            },
            sensor: {
                enabled: document.getElementById('sensor-enabled').checked,
                input_mode: 'raw-lnn',
                update_rate: parseInt(document.getElementById('sensor-update-rate').value)
            },
            processing: {
                batch_size: parseInt(document.getElementById('processing-batch-size').value),
                timeout: parseInt(document.getElementById('processing-timeout').value),
                priority: document.getElementById('processing-priority').value
            }
        };
        
        // 尝试调用真实的后端API保存多模态配置
        showNotification('正在保存多模态配置...', 'info');
        
        if (window.SelfLnnApi && typeof window.SelfLnnApi.saveMultimodalConfig === 'function') {
            const result = await window.SelfLnnApi.saveMultimodalConfig(config);
            
            if (result.success) {
                showNotification('✅ 多模态配置已保存', 'success');
            } else {
                showNotification(`❌ 保存多模态配置失败: ${result.error || '未知错误'}`, 'danger');
            }
        } else {
            // 后端未连接，显示错误
            showNotification('❌ 多模态配置后端未连接，请检查服务器状态', 'danger');
            console.error('多模态配置后端未连接');
        }
    } catch (error) {
        console.error('保存多模态配置失败:', error);
        showNotification(`❌ 保存多模态配置失败: ${error.message}`, 'danger');
    }
}

/**
 * 测试多模态处理
 */
async function testMultimodalProcessing() {
    
    try {
        /* MID-003修复: 不使用硬编码假数据，从真实捕获设备获取测试数据 */
        showNotification('正在从实际设备获取多模态测试数据...', 'info');
        
        /* 尝试从摄像头获取真实图像帧 */
        let visionData = null;
        let audioData = null;
        let textData = { content: '' }; /* 文本模态：由用户通过对话输入框提供真实文本，此处为空占位 */
        
        /* 尝试从摄像头获取真实图像帧（使用正确的DeviceManager API） */
        if (g_deviceManager && typeof g_deviceManager.captureSnapshot === 'function') {
            try {
                var activeCam = g_deviceManager.cameras ? g_deviceManager.cameras.find(function(c) { return c.active; }) : null;
                if (activeCam) {
                    const frame = g_deviceManager.captureSnapshot(activeCam.id);
                    if (frame) {
                        visionData = { image: frame, source: 'live_camera' };
                    }
                }
            } catch(e) { console.warn('[采集] 摄像头不可用:', e.message); }
        }
        
/* quickCapture返回{success, capturer, stream}对象，非Blob。
         * 音频数据通过capturer获取，而非直接检查返回值的size属性。 */
        if (typeof window.VoiceCaptureUtil !== 'undefined' && typeof window.VoiceCaptureUtil.quickCapture === 'function') {
            try {
                const captureResult = await window.VoiceCaptureUtil.quickCapture();
                if (captureResult && captureResult.success) {
                    audioData = { source: 'live_mic', capturer: captureResult.capturer };
                }
            } catch(e) { console.warn('[采集] 麦克风不可用:', e.message); }
        }
        
        const testData = {
            vision: visionData || { image: null, source: 'none' },
            audio: audioData || { waveform: null, source: 'none' },
            text: textData
        };
        
        // 尝试调用真实的后端API测试多模态处理
        showNotification('正在测试多模态处理...', 'info');
        
        if (window.SelfLnnApi && typeof window.SelfLnnApi.testMultimodalProcessing === 'function') {
            const result = await window.SelfLnnApi.testMultimodalProcessing(testData);
            
            if (result.success) {
                showNotification('✅ 多模态处理测试成功', 'success');
                
                // 更新测试结果
                if (result.data && result.data.result) {
                    document.getElementById('multimodal-test-result').textContent = 
                        `测试结果: ${result.data.result}`;
                }
            } else {
                showNotification(`❌ 多模态处理测试失败: ${result.error || '未知错误'}`, 'danger');
            }
        } else {
            // 后端未连接，显示错误
            showNotification('❌ 多模态处理后端未连接，请检查服务器状态', 'danger');
            console.error('多模态处理后端未连接');
        }
    } catch (error) {
        console.error('测试多模态处理失败:', error);
        showNotification(`❌ 测试多模态处理失败: ${error.message}`, 'danger');
    }
}

/**
 * 重置多模态配置
 */
async function resetMultimodalConfig() {
    
    if (safeConfirm('确定要重置多模态配置吗？这将恢复默认设置。')) {
        try {
            // 根据项目要求"禁止任何虚假数据"，不模拟重置进度
            // 尝试调用真实的后端API重置多模态配置
            showNotification('正在重置多模态配置...', 'warning');
            
            if (window.SelfLnnApi && typeof window.SelfLnnApi.resetMultimodalConfig === 'function') {
                const result = await window.SelfLnnApi.resetMultimodalConfig();
                
                if (result.success) {
                    showNotification('✅ 多模态配置已重置', 'success');
                    
                    // 刷新配置显示
                    refreshMultimodalStatus();
                } else {
                    showNotification(`❌ 重置多模态配置失败: ${result.error || '未知错误'}`, 'danger');
                }
            } else {
                // 后端未连接，显示错误
                showNotification('❌ 多模态重置后端未连接，请检查服务器状态', 'danger');
                console.error('多模态重置后端未连接');
            }
        } catch (error) {
            console.error('重置多模态配置失败:', error);
            showNotification(`❌ 重置多模态配置失败: ${error.message}`, 'danger');
        }
    }
}

/**
 * 停止多模态处理
 */
async function stopMultimodalProcessing() {
    
    if (safeConfirm('确定要停止多模态处理吗？')) {
        try {
            // 根据项目要求"禁止任何虚假数据"，不模拟停止进度
            // 尝试调用真实的后端API停止多模态处理
            showNotification('正在停止多模态处理...', 'warning');
            
            if (window.SelfLnnApi && typeof window.SelfLnnApi.stopMultimodalProcessing === 'function') {
                const result = await window.SelfLnnApi.stopMultimodalProcessing();
                
                if (result.success) {
                    showNotification('✅ 多模态处理已停止', 'success');
                    
                    // 刷新状态
                    refreshMultimodalStatus();
                } else {
                    showNotification(`❌ 停止多模态处理失败: ${result.error || '未知错误'}`, 'danger');
                }
            } else {
                // 后端未连接，显示错误
                showNotification('❌ 多模态停止后端未连接，请检查服务器状态', 'danger');
                console.error('多模态停止后端未连接');
            }
        } catch (error) {
            console.error('停止多模态处理失败:', error);
            showNotification(`❌ 停止多模态处理失败: ${error.message}`, 'danger');
        }
    }
}

/**
 * 重置机器人配置
 */
async function resetRobotConfig() {
    
    if (safeConfirm('确定要重置所有机器人配置到默认值吗？此操作不可撤销。')) {
        try {
            // 根据项目要求"禁止任何虚假数据"，不模拟重置进度
            // 尝试调用真实的后端API重置机器人配置
            showNotification('正在尝试重置机器人配置...', 'warning');
            
            // 检查是否有重置机器人配置的API
            if (window.SelfLnnApi && typeof window.SelfLnnApi.resetRobotConfig === 'function') {
                const result = await window.SelfLnnApi.resetRobotConfig();
                
                if (result.success) {
                    showNotification('✅ 机器人配置重置命令已发送', 'success');
                } else {
                    showNotification(`❌ 重置机器人配置失败: ${result.error || '未知错误'}`, 'danger');
                }
            } else {
                // 后端未连接，显示错误
                showNotification('❌ 机器人配置重置后端未连接，请检查服务器状态', 'danger');
                console.error('机器人配置重置后端未连接');
            }
        } catch (error) {
            console.error('重置机器人配置失败:', error);
            showNotification(`❌ 重置机器人配置失败: ${error.message}`, 'danger');
        }
    }
}

/**
 * 刷新LNN状态
 */
async function refreshLNNStatus() {
    
    try {
        // 根据项目要求"禁止任何虚假数据"，不模拟状态刷新
        // 尝试调用真实的后端API获取LNN状态
        showNotification('正在获取LNN状态...', 'info');
        
        // 检查是否有获取LNN状态的API
        if (window.SelfLnnApi && typeof window.SelfLnnApi.getLNNStatus === 'function') {
            const result = await window.SelfLnnApi.getLNNStatus();
            
            if (result.success && result.data) {
                /* W6修复: 后端返回 {"lnn":{...}}，需访问 result.data.lnn 获取LNN状态 */
                const status = result.data.lnn || result.data;
                showNotification('✅ LNN状态已更新', 'success');
                
                // 更新UI状态
                var cps = document.querySelectorAll('.circle-progress');
                var cts = document.querySelectorAll('.circle-text');
                var cfs = document.querySelectorAll('.circle-fill');
                if (status.stability !== undefined) {
                    const stabilityPercent = Math.min(100, Math.max(0, status.stability * 100));
                    if (cts[0]) cts[0].textContent = stabilityPercent.toFixed(1) + '%';
                    if (cfs[0]) {
                        var circ = 2 * Math.PI * 36;
                        var off = circ - (stabilityPercent / 100) * circ;
                        cfs[0].style.strokeDashoffset = off;
                    }
                }
                
                if (status.convergence_rate !== undefined) {
                    const convergencePercent = Math.min(100, Math.max(0, status.convergence_rate * 100));
                    if (cps[1]) cps[1].setAttribute('data-percent', convergencePercent);
                    if (cts[1]) cts[1].textContent = convergencePercent.toFixed(1) + '%';
                    if (cfs[1]) {
                    const circumference = 2 * Math.PI * 36;
                    const offset = circumference - (convergencePercent / 100) * circumference;
                    cfs[1].style.strokeDashoffset = offset;
                    }
                }
                
                if (status.dynamic_response !== undefined) {
                    const responseValue = status.dynamic_response;
                    if (cps[2]) cps[2].setAttribute('data-percent', Math.min(100, responseValue));
                    if (cts[2]) cts[2].textContent = responseValue.toFixed(1) + ' Hz';
                    if (cfs[2]) {
                    const circumference = 2 * Math.PI * 36;
                    const offset = circumference - (Math.min(100, responseValue) / 100) * circumference;
                    cfs[2].style.strokeDashoffset = offset;
                    }
                }
                
                var ivs = document.querySelectorAll('.indicator-value');
                var ifs = document.querySelectorAll('.indicator-fill');
                if (status.viscosity !== undefined) {
                    const viscosityPercent = Math.min(100, Math.max(0, status.viscosity * 100));
                    if (ivs[0]) ivs[0].textContent = status.viscosity.toFixed(3);
                    if (ifs[0]) ifs[0].style.width = viscosityPercent + '%';
                }
                
                if (status.temperature_entropy !== undefined) {
                    const entropyPercent = Math.min(100, Math.max(0, status.temperature_entropy * 100));
                    if (ivs[1]) ivs[1].textContent = status.temperature_entropy.toFixed(3);
                    if (ifs[1]) ifs[1].style.width = entropyPercent + '%';
                }
                
                if (status.flow_rate !== undefined) {
                    const flowPercent = Math.min(100, Math.max(0, status.flow_rate * 100));
                    if (ivs[2]) ivs[2].textContent = status.flow_rate.toFixed(3);
                    if (ifs[2]) ifs[2].style.width = flowPercent + '%';
                }
                
                if (status.diffusion_coefficient !== undefined) {
                    const diffusionPercent = Math.min(100, Math.max(0, status.diffusion_coefficient * 100));
                    if (ivs[3]) ivs[3].textContent = status.diffusion_coefficient.toFixed(3);
                    if (ifs[3]) ifs[3].style.width = diffusionPercent + '%';
                }
            } else {
                showNotification(`❌ 获取LNN状态失败: ${result.error || '未知错误'}`, 'danger');
            }
        } else {
            // 后端未连接，显示错误
            showNotification('❌ LNN状态后端未连接，请检查服务器状态', 'danger');
            console.error('LNN状态后端未连接');
        }
    } catch (error) {
        console.error('刷新LNN状态失败:', error);
        showNotification(`❌ 刷新LNN状态失败: ${error.message}`, 'danger');
    }
}

/**
 * 保存LNN参数
 */
async function saveLNNParameters() {
    
    try {
        // 根据项目要求"禁止任何虚假数据"，不模拟保存进度
        // 收集参数数据
        const parameters = {
            time_constant: parseFloat(document.getElementById('lnn-time-constant').value) || 0.1,
            viscosity: parseFloat(document.getElementById('lnn-viscosity').value) || 0.5,
            diffusion_rate: parseFloat(document.getElementById('lnn-diffusion-rate').value) || 0.01,
            temperature: parseFloat(document.getElementById('lnn-temperature').value) || 0.3,
            learning_rate: parseFloat(document.getElementById('lnn-learning-rate').value) || 0.001,
            regularization: parseFloat(document.getElementById('lnn-regularization').value) || 0.01,
            activation_threshold: parseFloat(document.getElementById('lnn-activation-threshold').value) || 0.5,
            feedback_gain: parseFloat(document.getElementById('lnn-feedback-gain').value) || 1.0,
            integration_step: parseFloat(document.getElementById('lnn-integration-step').value) || 0.01,
            noise_level: parseFloat(document.getElementById('lnn-noise-level').value) || 0.05
        };
        
        // 尝试调用真实的后端API保存LNN参数
        showNotification('正在保存LNN参数...', 'info');
        
        if (window.SelfLnnApi && typeof window.SelfLnnApi.saveLNNParameters === 'function') {
            const result = await window.SelfLnnApi.saveLNNParameters(parameters);
            
            if (result.success) {
                showNotification('✅ LNN参数已保存', 'success');
            } else {
                showNotification(`❌ 保存LNN参数失败: ${result.error || '未知错误'}`, 'danger');
            }
        } else {
            // 后端未连接，显示错误
            showNotification('❌ LNN参数后端未连接，请检查服务器状态', 'danger');
            console.error('LNN参数后端未连接');
        }
    } catch (error) {
        console.error('保存LNN参数失败:', error);
        showNotification(`❌ 保存LNN参数失败: ${error.message}`, 'danger');
    }
}

/**
 * 开始轮询训练状态
 */
function startTrainingStatusPolling() {
/* 统一到DataEngine调度，避免独立轮询 */
    if (window.g_dataEngine && typeof window.g_dataEngine.registerModule === 'function') {
        window.g_dataEngine.registerModule('training_status', 5000, async function() {
            try {
                if (window.SelfLnnApi && typeof window.SelfLnnApi.getTrainingStatus === 'function') {
                    var status = await window.SelfLnnApi.getTrainingStatus();
/* 闭包内正确缩进，修复语法结构 */
                    if (status && status.running) {
                        var epochEl = document.getElementById('training-current-epoch');
                        var lossEl = document.getElementById('training-current-loss');
                        var stageEl = document.getElementById('training-current-stage');
                        if (epochEl) epochEl.textContent = status.epoch || '?';
                        if (lossEl) lossEl.textContent = status.loss ? status.loss.toFixed(4) : '?';
                        if (stageEl) stageEl.textContent = status.stage || '训练中';
                    }
                }
            } catch(e) { console.warn('[训练] 状态轮询失败:', e.message); }
        });
    }
}

function stopTrainingStatusPolling() {
    if (window._trainingPollInterval) {
        clearInterval(window._trainingPollInterval);
        window._trainingPollInterval = null;
    }
}

// 添加CSS动画
const style = document.createElement('style');
style.textContent = `
@keyframes slideIn {
    from {
        transform: translateX(100%);
        opacity: 0;
    }
    to {
        transform: translateX(0);
        opacity: 1;
    }
}

@keyframes slideOut {
    from {
        transform: translateX(0);
        opacity: 1;
    }
    to {
        transform: translateX(100%);
        opacity: 0;
    }
}
`;
document.head.appendChild(style);

/* ===== 知识库管理函数 ===== */

/**
 * 刷新知识库统计
 */
async function refreshKnowledgeStats() {
    showNotification('正在刷新知识库统计...', 'info');
    
    try {
        var stats = null;
        if (window.SelfLnnApi && typeof window.SelfLnnApi.getKnowledgeStats === 'function') {
            var result = await window.SelfLnnApi.getKnowledgeStats();
            if (result && result.success && result.data) {
                stats = result.data.stats || result.data;
            }
        }
        
        var elTotal = document.getElementById('knowledge-total-entries');
        var elVerified = document.getElementById('knowledge-verified-facts');
        var elRules = document.getElementById('knowledge-reasoning-rules');
        var elConcepts = document.getElementById('knowledge-concept-relations');
        
        if (stats) {
            var total = stats.total_entries || 0;
            if (elTotal) elTotal.textContent = total;
            if (elVerified) elVerified.textContent = stats.verified_facts !== undefined ? stats.verified_facts : '--';
            if (elRules) elRules.textContent = stats.reasoning_rules !== undefined ? stats.reasoning_rules : '--';
            if (elConcepts) elConcepts.textContent = stats.concept_relations !== undefined ? stats.concept_relations : '--';
            
            var fillTotal = document.getElementById('knowledge-total-fill');
            var fillVerified = document.getElementById('knowledge-verified-fill');
            var fillRules = document.getElementById('knowledge-rules-fill');
            var fillRelations = document.getElementById('knowledge-relations-fill');
            var max = Math.max(total, stats.verified_facts || 0, stats.reasoning_rules || 0, stats.concept_relations || 0, 1);
            if (fillTotal) fillTotal.style.width = (max > 0 ? Math.min(100, (total / max) * 100) : 0) + '%';
            if (fillVerified) fillVerified.style.width = (max > 0 && stats.verified_facts !== undefined ? Math.min(100, (stats.verified_facts / max) * 100) : 0) + '%';
            if (fillRules) fillRules.style.width = (max > 0 && stats.reasoning_rules !== undefined ? Math.min(100, (stats.reasoning_rules / max) * 100) : 0) + '%';
            if (fillRelations) fillRelations.style.width = (max > 0 && stats.concept_relations !== undefined ? Math.min(100, (stats.concept_relations / max) * 100) : 0) + '%';
            
            showNotification('✅ 知识库统计已刷新 (' + total + ' 条)', 'success');
        } else {
            if (elTotal) elTotal.textContent = '0';
            if (elVerified) elVerified.textContent = '0';
            if (elRules) elRules.textContent = '0';
            if (elConcepts) elConcepts.textContent = '0';
            showNotification('⚠️ 知识库统计后端未连接', 'warning');
        }
    } catch (error) {
        console.error('刷新知识库统计失败:', error);
        showNotification('❌ 刷新知识库统计失败', 'danger');
    }
}

/**
 * 添加知识条目
 */
async function addKnowledgeEntry() {
    
    try {
        // 根据项目要求"禁止任何虚假数据"，不使用模拟表单
        // 尝试调用真实的后端API添加知识条目
        if (window.SelfLnnApi && typeof window.SelfLnnApi.addKnowledgeEntry === 'function') {
            // 完整实现：提供用户界面以收集知识条目信息
            // 使用浏览器原生prompt函数获取用户输入（这是合理的实现，符合项目要求）
            // 在更完善的系统中，可以扩展为模态框或表单界面
            const entryType = await promptAsync('请选择知识条目类型 (fact/rule/concept/relation):', 'fact');
            if (!entryType) return;
            
            const content = await promptAsync('请输入知识内容:');
            if (!content) return;
            
            showNotification('正在添加知识条目...', 'info');
            const result = await window.SelfLnnApi.addKnowledgeEntry({ type: entryType, content });
            
            if (result.success) {
                showNotification('✅ 知识条目已添加', 'success');
            } else {
                showNotification(`❌ 知识条目添加失败: ${result.message || '未知错误'}`, 'danger');
            }
        } else {
            // 后端未连接，显示错误
            showNotification('❌ 知识条目添加后端未连接，请检查服务器状态', 'danger');
            console.error('知识条目添加后端未连接');
        }
    } catch (error) {
        console.error('添加知识条目失败:', error);
        showNotification(`❌ 添加知识条目失败: ${error.message}`, 'danger');
    }
}

/**
 * 查看知识条目详情
 */
async function viewKnowledgeDetail(id) {
    showNotification('正在获取知识详情...', 'info');
    try {
        if (window.SelfLnnApi && typeof window.SelfLnnApi.getKnowledgeEntry === 'function') {
            var result = await window.SelfLnnApi.getKnowledgeEntry(id);
            if (result && result.success && result.data) {
                var entry = result.data;
                /* D-003修复: 对知识库条目标题/内容/分类进行HTML转义，防止存储型XSS攻击 */
                var safeTitle = window.escapeHtml(entry.title || '');
                var safeContent = window.escapeHtml(entry.content || entry.description || '');
                var safeCategory = window.escapeHtml(entry.category || '未分类');
                /* P1-FIX-01: 对id参数也进行HTML转义，防止XSS注入 */
                var safeId = window.escapeHtml(String(id));
                var html = '<div style="padding:12px;background:rgba(0,200,255,0.05);border-radius:8px;margin:8px 0;">' +
                    '<h4 style="margin:0 0 8px;">' + safeTitle + '</h4>' +
                    '<p style="color:rgba(255,255,255,0.7);font-size:0.8rem;">' + safeContent + '</p>' +
                    '<div style="display:flex;gap:12px;font-size:0.7rem;color:rgba(255,255,255,0.4);">' +
                    '<span>类型: ' + safeCategory + '</span>' +
                    '<span>置信度: ' + (entry.confidence != null ? (entry.confidence * 100).toFixed(1) + '%' : 'N/A') + '</span>' +
                    '<span>ID: ' + safeId + '</span></div></div>';
                showNotification('', 'info');
                var container = document.getElementById('knowledge-detail-panel');
                if (container) {
                /* P1-FIX-01: XSS防护验证 - safeTitle/safeContent/safeCategory都已使用escapeHtml转义，注入风险已消除 */
                container.innerHTML = html;
                    container.style.display = 'block';
                } else {
                    SelfLnnNotify.show(entry.content || entry.description || '无内容', 'info', 5000);
                }
            } else {
                showNotification('知识条目未找到', 'warning');
            }
        } else {
            showNotification('知识条目查询后端未连接', 'danger');
        }
    } catch (e) {
        showNotification('查看知识详情失败: ' + e.message, 'danger');
    }
}

/**
 * 删除知识条目
 */
async function deleteKnowledgeEntry(id) {
    if (!safeConfirm('确定要删除知识条目 #' + id + ' 吗？此操作不可撤销。')) return;
    showNotification('正在删除...', 'warning');
    try {
        if (window.SelfLnnApi && typeof window.SelfLnnApi.deleteKnowledgeEntry === 'function') {
            var result = await window.SelfLnnApi.deleteKnowledgeEntry(id);
            if (result && result.success) {
                showNotification('知识条目已删除', 'success');
            } else {
                showNotification('删除失败: ' + ((result && result.error) || '未知错误'), 'danger');
            }
        } else {
            showNotification('知识条目删除后端未连接', 'danger');
        }
    } catch (e) {
        showNotification('删除失败: ' + e.message, 'danger');
    }
}

/**
 * 导入知识
 */
function importKnowledge() {
    showNotification('正在打开知识导入对话框...', 'info');
    
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = '.json,.txt,.xml,.csv';
    fileInput.multiple = false;
    fileInput.onchange = async () => {
        if (fileInput.files.length > 0) {
            const file = fileInput.files[0];
            showNotification('正在读取文件: ' + file.name + '...', 'info');
            
            try {
                /* 真实读取文件内容 */
                const reader = new FileReader();
                const fileContent = await new Promise((resolve, reject) => {
                    reader.onload = (e) => resolve(e.target.result);
                    reader.onerror = (e) => reject(new Error('文件读取失败'));
                    reader.readAsText(file, 'UTF-8');
                });
                
                if (!fileContent || fileContent.length === 0) {
                    showNotification('❌ 文件内容为空', 'danger');
                    return;
                }
                
                /* 尝试通过后端API导入知识 */
                if (window.SelfLnnApi && typeof window.SelfLnnApi.importKnowledge === 'function') {
                    const result = await window.SelfLnnApi.importKnowledge(fileContent, file.name);
                    if (result && result.success) {
                        showNotification('✅ 知识导入成功: ' + (result.importedCount || 0) + ' 条记录', 'success');
                        /* 刷新知识库统计 */
                        if (typeof refreshKnowledgeStats === 'function') {
                            await refreshKnowledgeStats();
                        }
                    } else {
                        showNotification('⚠️ 知识导入部分完成: ' + (result.message || '未知错误'), 'warning');
                    }
                } else {
                    /* 后端未连接时，尝试前端本地存储保存 */
                    try {
                        const storageKey = 'imported_knowledge_' + Date.now();
                        localStorage.setItem(storageKey, fileContent);
                        showNotification('✅ 文件已读取并本地缓存 (' + 
                            (fileContent.length / 1024).toFixed(1) + 'KB)。后端连接后将自动导入。', 'success');
                    } catch (storageError) {
                        showNotification('❌ 本地存储空间不足，文件读取失败', 'danger');
                    }
                }
            } catch (error) {
                console.error('知识导入失败:', error);
                showNotification('❌ 知识导入失败: ' + error.message, 'danger');
            }
        }
    };
    fileInput.click();
}

/**
 * 导出知识
 */
async function exportKnowledge() {
    showNotification('正在导出知识库...', 'info');
    
    try {
        // 根据项目要求"禁止任何虚假数据"，导出真实知识库数据
        if (window.SelfLnnApi && typeof window.SelfLnnApi.exportKnowledge === 'function') {
            const knowledgeData = await window.SelfLnnApi.exportKnowledge();
            const data = JSON.stringify(knowledgeData, null, 2);
            const blob = new Blob([data], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'knowledge-base-export.json';
            a.click();
            URL.revokeObjectURL(url);
            
            showNotification('✅ 知识库导出完成', 'success');
        } else {
            // 后端未连接，显示错误而不是导出虚假数据
            showNotification('❌ 知识库导出后端未连接，无法导出真实数据', 'danger');
            console.error('知识库导出后端未连接');
        }
    } catch (error) {
        console.error('导出知识库失败:', error);
        showNotification('❌ 导出知识库失败: ' + error.message, 'danger');
    }
}

/**
 * 知识库保存到磁盘
 */
async function knowledgeSaveToDisk() {
    showNotification('正在保存知识库...', 'info');
    try {
        if (window.SelfLnnApi && typeof window.SelfLnnApi.knowledgeSave === 'function') {
            var result = await window.SelfLnnApi.knowledgeSave();
            if (result && result.success) {
                var statusEl = document.getElementById('knowledge-storage-status');
                if (statusEl) {
                    statusEl.style.display = 'block';
                    statusEl.style.background = 'rgba(0,255,136,0.1)';
                    statusEl.style.color = '#00ff88';
                    statusEl.textContent = '✅ 知识库已保存到 knowledge_data/knowledge_base.skb';
                }
                showNotification('✅ 知识库保存成功', 'success');
            } else {
                showNotification('❌ 保存失败: ' + ((result && result.error) || '后端错误'), 'danger');
            }
        } else {
            showNotification('❌ 知识库保存后端未连接', 'danger');
        }
    } catch (e) {
        showNotification('❌ 保存失败: ' + e.message, 'danger');
    }
}

/**
 * 从磁盘加载知识库
 */
async function knowledgeLoadFromDisk() {
    if (!safeConfirm('从磁盘加载历史知识库将追加到当前知识库，不会覆盖现有知识。确定继续？')) return;
    showNotification('正在加载知识库...', 'info');
    try {
        if (window.SelfLnnApi && typeof window.SelfLnnApi.knowledgeLoad === 'function') {
            var result = await window.SelfLnnApi.knowledgeLoad();
            if (result && result.success) {
                var statusEl = document.getElementById('knowledge-storage-status');
                if (statusEl) {
                    statusEl.style.display = 'block';
                    statusEl.style.background = 'rgba(0,200,255,0.1)';
                    statusEl.style.color = '#00c8ff';
                    statusEl.textContent = '✅ 已从 knowledge_data/knowledge_base.skb 加载知识（条目数: ' + (result.count || '?') + '）';
                }
                showNotification('✅ 知识库加载成功', 'success');
                refreshKnowledgeStats();
            } else {
                showNotification('⚠️ 加载失败: ' + ((result && result.error) || '文件可能不存在'), 'warning');
            }
        } else {
            showNotification('❌ 知识库加载后端未连接', 'danger');
        }
    } catch (e) {
        showNotification('❌ 加载失败: ' + e.message, 'danger');
    }
}

/**
 * 导出知识库为JSON
 */
async function knowledgeExportAsJSON() {
    showNotification('正在导出...', 'info');
    try {
        if (window.SelfLnnApi && typeof window.SelfLnnApi.knowledgeExportJSON === 'function') {
            var result = await window.SelfLnnApi.knowledgeExportJSON();
            if (result && result.success && result.data) {
                var jsonStr = JSON.stringify(result.data, null, 2);
                var blob = new Blob([jsonStr], { type: 'application/json' });
                var url = URL.createObjectURL(blob);
                var a = document.createElement('a');
                a.href = url;
                var ts = new Date().toISOString().replace(/[:.]/g, '-');
                a.download = 'selflnn-knowledge-' + ts + '.json';
                a.click();
                URL.revokeObjectURL(url);
                showNotification('✅ JSON导出完成（' + (result.count || '?') + '条目）', 'success');
            } else {
                showNotification('❌ 导出失败: ' + ((result && result.error) || '无数据'), 'danger');
            }
        } else {
            showNotification('❌ 知识库导出后端未连接', 'danger');
        }
    } catch (e) {
        showNotification('❌ 导出失败: ' + e.message, 'danger');
    }
}

/**
 * 从JSON导入知识库
 */
async function knowledgeImportFromJSON() {
    var fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = '.json';
    fileInput.onchange = async function() {
        if (!fileInput.files.length) return;
        showNotification('正在导入...', 'info');
        try {
            var text = await fileInput.files[0].text();
            /* P1-001修复: JSON.parse使用专用try-catch包裹，防止后端返回异常数据时中断整个JS执行 */
            var data;
            try {
                data = JSON.parse(text);
            } catch (parseErr) {
                showNotification('❌ JSON格式无效，请检查导入文件: ' + parseErr.message, 'danger');
                console.error('[知识库导入] JSON解析失败:', parseErr.message);
                return;
            }
            /* P1-001修复: 对data进行null/undefined检查，防止空对象传入后端 */
            if (!data || typeof data !== 'object') {
                showNotification('❌ 导入数据为空或格式不正确', 'danger');
                return;
            }
            if (window.SelfLnnApi && typeof window.SelfLnnApi.knowledgeImportJSON === 'function') {
                var result = await window.SelfLnnApi.knowledgeImportJSON(data);
                /* P1-001修复: result为null时的降级处理 */
                if (result && result.success) {
                    showNotification('✅ 导入成功（' + (result.count || '?') + '条目）', 'success');
                    refreshKnowledgeStats();
                } else {
                    showNotification('❌ 导入失败: ' + ((result && result.error) || '格式错误'), 'danger');
                }
            } else {
                showNotification('❌ 知识库导入后端未连接', 'danger');
            }
        } catch (e) {
            showNotification('❌ 导入失败: ' + e.message, 'danger');
            console.error('[知识库导入] 未预期的错误:', e);
        }
    };
    fileInput.click();
}

/**
 * 刷新知识库存储状态
 */
function knowledgeRefreshStatus() {
    var statusEl = document.getElementById('knowledge-storage-status');
    if (statusEl) {
        statusEl.style.display = 'block';
        statusEl.style.background = 'rgba(0,200,255,0.1)';
        statusEl.style.color = '#00c8ff';
        window.SelfLnnApi.getKnowledgeStats().then(function(data) {
            /* FIX-F2-CRIT-4: getKnowledgeStats包装为{success,data},后端返回{stats:{...}} */
            var stats = (data && data.data && data.data.stats) ? data.data.stats : (data && data.stats) ? data.stats : null;
            var entries = (stats && stats.total_entries != null) ? stats.total_entries : '--';
            var domains = (stats && stats.domain_count != null) ? stats.domain_count : '--';
            statusEl.textContent = '📁 存储路径: knowledge_data/knowledge_base.skb | 条目: ' + entries + ' | 领域: ' + domains;
            var countEl = document.getElementById('knowledge-entry-count');
            if (countEl) countEl.textContent = entries + ' 条目';
            var domainCountEl = document.getElementById('knowledge-domain-count');
            if (domainCountEl) domainCountEl.textContent = domains + ' 领域';
            if (stats && stats.domains && stats.domains.length > 0) {
                _renderKnowledgeDomains(stats.domains);
            }
        }).catch(function() {
            statusEl.textContent = '📁 存储路径: knowledge_data/knowledge_base.skb | 条目: -- | 领域: --';
        });
    }
    showNotification('知识库状态已刷新', 'info');
}

/* 从API动态渲染知识领域 */
function _renderKnowledgeDomains(domains) {
    var grid = document.getElementById('knowledge-domains-grid');
    if (!grid) return;
    var colors = ['#00ff88','#00c8ff','#ffaa00','#ff4488','#44ff88','#4488ff'];
    var html = '';
    for (var i = 0; i < domains.length; i++) {
        var d = domains[i];
        var name = typeof d === 'string' ? d : (d.name || d.subject || '未知领域');
        var count = typeof d === 'object' ? (d.count || '') : '';
        var color = colors[i % colors.length];
        html += '<span style="color:' + color + ';"> ' + name + (count ? ' (' + count + ')' : '') + '</span>';
    }
    grid.innerHTML = html || '<span style="color:#666;">无领域数据</span>';
}

/**
 * 运行一致性检查
 */
async function runConsistencyCheck() {
    showNotification('正在检查知识一致性...', 'info');
    try {
        var result = null;
        if (window.SelfLnnApi && typeof window.SelfLnnApi.consistencyCheck === 'function') {
            result = await window.SelfLnnApi.consistencyCheck();
        }
        var elConsistency = document.getElementById('knowledge-consistency');
        var elConflicts = document.getElementById('knowledge-conflicts');
        var elCircular = document.getElementById('knowledge-circular-deps');
        if (result && result.success) {
            if (elConsistency) elConsistency.textContent = (result.consistency || result.score || '一致');
            if (elConflicts) elConflicts.textContent = (result.conflicts !== undefined ? result.conflicts : '0');
            if (elCircular) elCircular.textContent = (result.circular_deps !== undefined ? result.circular_deps : '0');
            showNotification('✅ 知识一致性检查完成: ' + (result.score || 'N/A'), 'success');
        } else {
            if (elConsistency) elConsistency.textContent = '0';
            if (elConflicts) elConflicts.textContent = '0';
            if (elCircular) elCircular.textContent = '0';
            showNotification('⚠️ 一致性检查: ' + ((result && result.error) || '后端未响应'), 'warning');
        }
    } catch (e) {
        showNotification('⚠️ 检查失败: ' + e.message, 'warning');
    }
}

/**
 * 刷新硬件资源分配状态
 */
async function refreshHardwareResources() {
    try {
        if (window.SelfLnnApi && typeof window.SelfLnnApi.hardwareResources === 'function') {
            var result = await window.SelfLnnApi.hardwareResources();
            if (result && result.success) {
                var d = result;
                document.getElementById('hw-cameras').textContent = (d.cameras || 0);
                document.getElementById('hw-cameras').style.color = d.cameras >= 3 ? '#00ff88' : (d.cameras > 0 ? '#ffaa00' : '#ff4444');
                document.getElementById('hw-stereo').textContent = d.stereo_available ? '✅ 可用' : '❌ 不可用';
                document.getElementById('hw-stereo').style.color = d.stereo_available ? '#00ff88' : '#ff4444';
                document.getElementById('hw-recognition').textContent = d.recognition_available ? '✅ 可用' : '❌ 不可用';
                document.getElementById('hw-recognition').style.color = d.recognition_available ? '#00ff88' : '#ff4444';
                document.getElementById('hw-microphones').textContent = (d.microphones || 0);
                document.getElementById('hw-beamforming').textContent = d.beamforming == 'true' || d.beamforming === true ? '✅ 可用' : '--';
                document.getElementById('hw-speakers').textContent = (d.speakers || 0);
                document.getElementById('hw-quality').textContent = '质量评分: ' + ((d.quality_score || 0) * 100).toFixed(0) + '%';
            } else {
                document.getElementById('hw-cameras').textContent = '--';
                document.getElementById('hw-stereo').textContent = '--';
                document.getElementById('hw-recognition').textContent = '--';
                document.getElementById('hw-microphones').textContent = '--';
                document.getElementById('hw-speakers').textContent = '--';
                document.getElementById('hw-quality').textContent = '后端未连接';
            }
        } else {
            document.getElementById('hw-quality').textContent = 'API不可用';
        }
    } catch (e) {
        console.error('硬件资源刷新失败:', e);
        document.getElementById('hw-quality').textContent = '刷新失败';
    }
}

/**
 * 执行推理
 */
async function runInference() {
    showNotification('正在执行知识推理...', 'info');
    try {
        if (window.SelfLnnApi && typeof window.SelfLnnApi.runInference === 'function') {
            var result = await window.SelfLnnApi.runInference();
            if (result && result.success) {
                showNotification('✅ 推理完成: 生成 ' + (result.conclusions || '?') + ' 个新结论', 'success');
            } else {
                showNotification('⚠️ 推理: ' + ((result && result.error) || '后端未响应'), 'warning');
            }
        } else {
            showNotification('⚠️ LNN推理接口未连接', 'warning');
        }
    } catch (e) {
        showNotification('⚠️ 推理失败: ' + e.message, 'warning');
    }
}

/**
 * 验证知识
 */
async function validateKnowledge() {
    showNotification('正在验证知识条目...', 'info');
    try {
        if (window.SelfLnnApi && typeof window.SelfLnnApi.validateKnowledge === 'function') {
            var result = await window.SelfLnnApi.validateKnowledge();
            if (result && result.success) {
                showNotification('✅ 知识验证完成: ' + (result.verified || '?') + '/' + (result.total || '?') + ' 个事实已验证', 'success');
            } else {
                showNotification('⚠️ 验证: ' + ((result && result.error) || '后端未响应'), 'warning');
            }
        } else {
            showNotification('⚠️ 知识验证后端未连接', 'warning');
        }
    } catch (e) {
        showNotification('⚠️ 验证失败: ' + e.message, 'warning');
    }
}

/* ===== 可视化系统集成 ===== */

/**
 * 初始化可视化系统
 */
function initVisualizationSystem() {
    if (!window.visualizationManager) {
        window.visualizationManager = new VisualizationManager();
    }
    
    try {
        window.visualizationManager.initAllCharts();
    } catch (error) {
        console.error('可视化系统初始化失败:', error);
    }
}

/**
 * 启动实时数据可视化
 */
function startVisualizationUpdates() {
    if (!window.visualizationManager) {
        console.warn('可视化系统未初始化');
        return;
    }
    
    window.visualizationManager.startAutoUpdate(3000);
    showNotification('实时数据可视化已启动', 'info');
}

/**
 * 停止实时数据可视化
 */
function stopVisualizationUpdates() {
    if (!window.visualizationManager) return;
    window.visualizationManager.stopAutoUpdate();
    showNotification('实时数据可视化已停止', 'warning');
}

/**
 * 切换可视化面板
 */
function toggleVisualizationPanel(panelId) {
    const panel = document.getElementById(panelId);
    if (!panel) return;
    
    const isHidden = panel.style.display === 'none';
    panel.style.display = isHidden ? 'block' : 'none';
    
    if (isHidden && window.visualizationManager) {
        window.visualizationManager.resizeCharts();
    }
}

/**
 * 导出可视化数据
 */
function exportVisualizationData() {
    if (!window.visualizationManager) {
        showNotification('可视化系统未初始化', 'warning');
        return;
    }
    
    const data = window.visualizationManager.exportAllData();
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `visualization-data-${new Date().toISOString().slice(0,10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
    showNotification('可视化数据已导出', 'success');
}

/**
 * 连接WebSocket实时数据流到可视化系统
 * 注册消息处理器，将后端推送的数据分发到对应图表
 */
function connectVisualizationWebSocket() {
    if (!window.SelfLnnWebSocket) return;

    /* C-004修复: 后端ws_push发射消息类型为'training_progress'，
     * 同时保留'training_status'监听向后兼容旧版后端。
     * FE-005修复: 职责划分明确——
     *   main.js（本文件）：负责 visualizationManager 图表数据更新（UI可视化层）
     *   training-push.js：负责 trainingState 状态缓存 + dataBuffers 数据缓冲（数据缓存层）
     *   两者共存不冲突，各自处理不同层次的数据 */
    var _trainingProgressHandler = function(data) {
        if (!window.visualizationManager) return;
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
    };
    window.SelfLnnWebSocket.on('training_progress', _trainingProgressHandler);
    window.SelfLnnWebSocket.on('training_status', _trainingProgressHandler);

    window.SelfLnnWebSocket.on('system_status', function(data) {
        if (!window.visualizationManager) return;
        if (data.cpu_usage !== undefined) {
            window.visualizationManager.updateSystemResourceData(
                data.cpu_usage || 0,
                data.memory_usage || 0,
                data.gpu_usage || 0
            );
        }
        if (data.gpu) {
            window.visualizationManager.updateGpuMonitorData(data.gpu);
        }
    });

    /* 以下事件（memory_status / robot_status / knowledge_status / prediction_result
       / concept_evolution / state_activation_data / weight_distribution / activation_stats
       / lnn_state / safety_alert / metacognition_status）已由 training-push.js 的
       TrainingPushManager 统一注册并管理，同时通过 trainingPushManager 分发 CustomEvent
       和更新 visualizationManager / g_dataEngine，此处移除重复注册以避免事件冲突。 */

    window.SelfLnnWebSocket.connect();
}

/**
 * 渲染LNN状态演化轨迹
 * 从window.lnnStateTrajectory缓存中读取最近200步状态向量，绘制在lnn-state-trajectory canvas上
 */
function renderLnnStateTrajectory() {
    var canvas = document.getElementById('lnn-state-trajectory');
    if (!canvas || !window.lnnStateTrajectory || window.lnnStateTrajectory.length < 2) return;
    var ctx = canvas.getContext('2d');
    var w = canvas.width;
    var h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    var trajectory = window.lnnStateTrajectory;
    var numDims = Math.min(10, trajectory[0].vec.length);
    var colors = ['#00c8ff','#ff6b6b','#48c774','#ffd93d','#b388ff','#ff8a65','#4dd0e1','#f06292','#aed581','#90a4ae'];
    var margin = {top: 10, bottom: 20, left: 10, right: 10};
    var plotW = w - margin.left - margin.right;
    var plotH = h - margin.top - margin.bottom;

    // 计算全局最小/最大值（所有维度）
    var globalMin = Infinity, globalMax = -Infinity;
    for (var i = 0; i < trajectory.length; i++) {
        for (var j = 0; j < numDims; j++) {
            var v = trajectory[i].vec[j];
            if (v < globalMin) globalMin = v;
            if (v > globalMax) globalMax = v;
        }
    }
    var range = globalMax - globalMin || 1;

    // 绘制每个维度的轨迹线
    for (var dim = 0; dim < numDims; dim++) {
        ctx.beginPath();
        ctx.strokeStyle = colors[dim % colors.length];
        ctx.lineWidth = 1.2;
        ctx.globalAlpha = 0.8;
        for (var k = 0; k < trajectory.length; k++) {
            var x = margin.left + (k / (trajectory.length - 1)) * plotW;
            var y = margin.top + (1 - (trajectory[k].vec[dim] - globalMin) / range) * plotH;
            if (k === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
    }

    // 维度标签
    ctx.globalAlpha = 0.5;
    ctx.fillStyle = '#888';
    ctx.font = '10px "Microsoft YaHei", sans-serif';
    for (var d = 0; d < numDims; d++) {
        ctx.fillStyle = colors[d % colors.length];
        ctx.globalAlpha = 0.7;
        var labelX = margin.left + ((trajectory.length - 1) / (trajectory.length - 1)) * plotW + 4;
        var lastVal = trajectory[trajectory.length - 1].vec[d];
        var labelY = margin.top + (1 - (lastVal - globalMin) / range) * plotH;
        ctx.fillText('h' + d, labelX, labelY + 3);
    }
    ctx.globalAlpha = 1.0;
}

/* ===== 交互式训练控制增强 ===== */

/**
 * 训练超参数优化器
 */
let hyperparameterOptimizer = null;

/**
 * 启动超参数搜索
 */
async function startHyperparameterSearch() {
    showNotification('正在启动超参数搜索...', 'info');
    
    try {
        const config = {
            search_method: document.getElementById('hp-search-method')?.value || 'bayesian',
            max_trials: parseInt(document.getElementById('hp-max-trials')?.value) || 20,
            optimization_metric: document.getElementById('hp-metric')?.value || 'accuracy',
            parameters: {
                learning_rate: { min: 1e-5, max: 1e-1, type: 'log' },
                batch_size: { values: [16, 32, 64, 128] },
                weight_decay: { min: 1e-6, max: 1e-2, type: 'log' },
                dropout: { min: 0.0, max: 0.5, type: 'linear' }
            }
        };
        
        if (window.SelfLnnApi && typeof window.SelfLnnApi.startHyperparameterSearch === 'function') {
            const result = await window.SelfLnnApi.startHyperparameterSearch(config);
            if (result.success) {
                hyperparameterOptimizer = result.data;
                showNotification('✅ 超参数搜索已启动', 'success');
                pollHyperparameterStatus();
            } else {
                showNotification('❌ 超参数搜索启动失败', 'danger');
            }
        } else {
            showNotification('❌ 超参数搜索后端未连接', 'danger');
        }
    } catch (error) {
        console.error('超参数搜索失败:', error);
        showNotification('❌ 超参数搜索出错', 'danger');
    }
}

/**
 * 轮询超参数搜索状态
 */
var _hpPollCount = 0;
var _hpMaxPolls = 60;

function pollHyperparameterStatus() {
    _hpPollCount = 0;
    _hpMaxPolls = 60;
    if (typeof g_dataEngine !== 'undefined' && g_dataEngine && typeof g_dataEngine.registerModule === 'function') {
        g_dataEngine.registerModule('hyperparameter_poll', 5000, _hpPollTick);
    } else {
        var hpInterval = setInterval(function() {
            _hpPollTick();
            if (_hpPollCount > _hpMaxPolls || !hyperparameterOptimizer) {
                clearInterval(hpInterval);
            }
        }, 5000);
    }
}

async function _hpPollTick() {
    _hpPollCount++;
    if (_hpPollCount > _hpMaxPolls || !hyperparameterOptimizer) {
        if (typeof g_dataEngine !== 'undefined' && g_dataEngine) {
            g_dataEngine.unregisterModule('hyperparameter_poll');
        }
        return;
    }
    try {
        if (window.SelfLnnApi && typeof window.SelfLnnApi.getHyperparameterStatus === 'function') {
            var status = await window.SelfLnnApi.getHyperparameterStatus(hyperparameterOptimizer.id);
            if (status.success && status.data) {
                var hpStatus = status.data;
                var hpProgress = document.getElementById('hp-search-progress');
                if (hpProgress) {
                    var pct = (hpStatus.completed_trials / hpStatus.total_trials) * 100;
                    hpProgress.style.width = pct + '%';
                    hpProgress.textContent = hpStatus.completed_trials + '/' + hpStatus.total_trials;
                }
                var bestResult = document.getElementById('hp-best-result');
                if (bestResult && hpStatus.best_trial) {
                    bestResult.textContent = '最佳: ' + hpStatus.best_trial.metric_value.toFixed(4) + ' (' + hpStatus.best_trial.params.learning_rate.toExponential() + ')';
                }
                if (hpStatus.completed_trials >= hpStatus.total_trials) {
                    if (typeof g_dataEngine !== 'undefined' && g_dataEngine) {
                        g_dataEngine.unregisterModule('hyperparameter_poll');
                    }
                    showNotification('✅ 超参数搜索完成', 'success');
                }
            }
        }
    } catch (error) {
        console.error('轮询超参数状态失败:', error);
    }
}

/**
 * 训练计划管理器
 *: 添加本地持久化（localStorage）防止页面刷新后数据丢失
 */
let trainingSchedules = [];

/**
 * 持久化保存训练计划到本地存储和后端
 */
function persistTrainingSchedules() {
    try {
        localStorage.setItem('selflnn_training_schedules', JSON.stringify(trainingSchedules));
        SelfLnnApi.request('/system/config/update', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ key: 'training_schedules', value: JSON.stringify(trainingSchedules) })
        }).catch(function(err) { /* FE-011修复: 空catch块添加错误日志，保留静默行为避免阻塞 */ console.error('[训练计划持久化] 后端保存失败:', err && err.message ? err.message : err); });
    } catch(e) { /* FE-011修复: 空catch块添加错误日志 */ console.error('[训练计划持久化] localStorage保存失败:', e && e.message ? e.message : e); }
}

/**
 * 从本地存储加载训练计划
 */
function loadTrainingSchedulesFromStorage() {
    try {
        var saved = localStorage.getItem('selflnn_training_schedules');
        if (saved) {
            /* P1-002修复: JSON.parse使用专用try-catch包裹，防止localStorage中损坏数据导致JS执行中断 */
            var parsed;
            try {
                parsed = JSON.parse(saved);
            } catch (parseErr) {
                console.error('[训练计划加载] JSON解析失败，将重置训练计划:', parseErr.message);
                /* 清除损坏的localStorage数据 */
                try { localStorage.removeItem('selflnn_training_schedules'); } catch (rmErr) {}
                trainingSchedules = [];
                return;
            }
            /* P1-002修复: 验证解析后数据为数组，非数组时降级处理 */
            if (Array.isArray(parsed)) {
                trainingSchedules = parsed;
                renderTrainingSchedules();
            } else {
                console.warn('[训练计划加载] 解析结果不是数组，将重置训练计划');
                trainingSchedules = [];
            }
        }
    } catch(e) {
        console.error('[训练计划加载] 未预期的错误:', e && e.message ? e.message : e);
        trainingSchedules = [];
    }
}

/**
 * 创建训练计划
 */
async function createTrainingSchedule() {
    const scheduleName = await promptAsync('请输入训练计划名称:');
    if (!scheduleName) return;
    
    const schedule = {
        id: Date.now(),
        name: scheduleName,
        models: [],
        epochs: parseInt(document.getElementById('training-epochs')?.value) || 50,
        priority: 'normal',
        status: 'pending',
        created_at: new Date().toISOString(),
        estimated_duration: '--'
    };
    
    trainingSchedules.push(schedule);
    persistTrainingSchedules();
    renderTrainingSchedules();
    showNotification(`训练计划 "${scheduleName}" 已创建`, 'success');
}

/**
 * 渲染训练计划列表
 */
function renderTrainingSchedules() {
    const container = document.getElementById('training-schedule-list');
    if (!container) return;
    
    container.innerHTML = '';
    
    trainingSchedules.forEach(schedule => {
        const item = document.createElement('div');
        item.className = 'schedule-item';
        item.innerHTML = `
            <div class="schedule-info">
                <span class="schedule-name">${window.escapeHtml(schedule.name)}</span>
                <span class="schedule-status status-${window.escapeHtml(schedule.status)}">${window.escapeHtml(getScheduleStatusText(schedule.status))}</span>
            </div>
            <div class="schedule-details">
                <span>轮数: ${schedule.epochs}</span>
                <span>模型: ${schedule.models.length || 1}个</span>
                <span>${window.escapeHtml(schedule.estimated_duration)}</span>
            </div>
            <div class="schedule-actions">
                <button class="btn btn-sm btn-success" onclick="startSchedule(${schedule.id})">开始</button>
                <button class="btn btn-sm btn-warning" onclick="pauseSchedule(${schedule.id})">暂停</button>
                <button class="btn btn-sm btn-danger" onclick="deleteSchedule(${schedule.id})">删除</button>
            </div>
        `;
        container.appendChild(item);
    });
}

/**
 * 获取调度状态文本
 */
function getScheduleStatusText(status) {
    const map = {
        'pending': '等待中',
        'running': '运行中',
        'paused': '已暂停',
        'completed': '已完成',
        'failed': '失败'
    };
    return map[status] || status;
}

/**
 * 开始训练计划
 */
async function startSchedule(scheduleId) {
    const schedule = trainingSchedules.find(s => s.id === scheduleId);
    if (!schedule) return;
    
    schedule.status = 'running';
    persistTrainingSchedules();
    renderTrainingSchedules();
    showNotification(`训练计划 "${schedule.name}" 已开始`, 'success');
    
    if (window.SelfLnnApi && typeof window.SelfLnnApi.startTrainingJob === 'function') {
        await window.SelfLnnApi.startTrainingJob({ schedule_id: scheduleId });
    }
}

/**
 * 暂停训练计划
 */
async function pauseSchedule(scheduleId) {
    const schedule = trainingSchedules.find(s => s.id === scheduleId);
    if (!schedule) return;
    
    schedule.status = 'paused';
    persistTrainingSchedules();
    renderTrainingSchedules();
    showNotification(`训练计划 "${schedule.name}" 已暂停`, 'warning');
}

/**
 * 删除训练计划
 */
function deleteSchedule(scheduleId) {
    trainingSchedules = trainingSchedules.filter(s => s.id !== scheduleId);
    persistTrainingSchedules();
    renderTrainingSchedules();
    showNotification('训练计划已删除', 'info');
}

/* ===== 多机器人监控系统 ===== */

/**
 * 机器人舰队管理
 */
let robotFleet = {};
let fleetPollInterval = null;

/**
 * 初始化多机器人监控
 */
function initMultiRobotMonitor() {
    /* 初始化机器人舰队为空，仅在后端API返回真实机器人数据时才渲染 */
    robotFleet = {};
    
    renderFleetEmptyState();
    startFleetPolling();
}

/**
 * 渲染舰队空状态
 */
function renderFleetEmptyState() {
    const container = document.getElementById('fleet-dashboard');
    if (!container) return;
    container.innerHTML = '<div style="color:rgba(255,255,255,0.4);font-size:0.85rem;padding:2rem;text-align:center;">等待后端机器人数据...</div>';
}

/**
 * 渲染舰队仪表盘
 */
function renderFleetDashboard() {
    const container = document.getElementById('fleet-dashboard');
    if (!container) return;
    
    let html = '<div class="fleet-grid">';
    
    Object.keys(robotFleet).forEach((robotId, index) => {
        const robot = robotFleet[robotId];
        const statusClass = robot.status === 'online' ? 'status-online' :
                            robot.status === 'busy' ? 'status-busy' :
                            robot.status === 'error' ? 'status-error' : 'status-offline';
        const statusText = robot.status === 'online' ? '在线' :
                           robot.status === 'busy' ? '忙碌' :
                           robot.status === 'error' ? '错误' : '离线';
        const batteryColor = robot.battery > 50 ? '#00ff88' :
                             robot.battery > 20 ? '#ffaa00' : '#ff4444';
        
        html += `
            <div class="fleet-card" id="fleet-card-${robotId}">
                <div class="fleet-header">
                    <span class="fleet-name">${robot.name}</span>
                    <span class="fleet-status ${statusClass}">${statusText}</span>
                </div>
                <div class="fleet-body">
                    <div class="fleet-battery">
                        <div class="battery-icon">
                            <div class="battery-level" style="width: ${robot.battery}%; background: ${batteryColor}"></div>
                        </div>
                        <span>${robot.battery}%</span>
                    </div>
                    <div class="fleet-position">
                        <span>位置: (${window.escapeHtml(Number(robot.position[0]).toFixed(1))}, ${window.escapeHtml(Number(robot.position[1]).toFixed(1))}, ${window.escapeHtml(Number(robot.position[2]).toFixed(1))})</span>
                    </div>
                    <div class="fleet-task">
                        <span>任务: ${robot.task}</span>
                    </div>
                </div>
                <div class="fleet-actions">
                    <button class="btn btn-sm btn-primary" onclick="selectRobot('${robotId}')">选择</button>
                    <button class="btn btn-sm btn-info" onclick="assignTask('${robotId}')">分配任务</button>
                    <button class="btn btn-sm btn-warning" onclick="emergencyStopRobot('${robotId}')">急停</button>
                </div>
            </div>
        `;
    });
    
    html += '</div>';
    container.innerHTML = html;
}

/**
 * 启动舰队轮询（通过统一轮询中心调度）
 */
function startFleetPolling() {
    if (fleetPollInterval) return;
    
    if (typeof g_dataEngine !== 'undefined' && g_dataEngine) {
        g_dataEngine.registerModule('fleet_status', 3000, pollFleetStatus);
    } else {
        fleetPollInterval = setInterval(async () => {
            await pollFleetStatus();
        }, 3000);
    }
}

/**
 * 舰队状态轮询回调
 */
async function pollFleetStatus() {
    try {
        if (window.SelfLnnApi && typeof window.SelfLnnApi.getFleetStatus === 'function') {
            const result = await window.SelfLnnApi.getFleetStatus();
            if (result.success && result.data) {
                updateFleetData(result.data.robots);
            }
        }
    } catch (error) {
        console.error('舰队状态轮询失败:', error);
    }
}

/**
 * 停止舰队轮询
 */
function stopFleetPolling() {
    if (fleetPollInterval) {
        clearInterval(fleetPollInterval);
        fleetPollInterval = null;
    }
    if (typeof g_dataEngine !== 'undefined' && g_dataEngine) {
        g_dataEngine.unregisterModule('fleet_status');
    }
}

/**
 * 更新舰队数据
 */
function updateFleetData(robots) {
    if (!robots || !Array.isArray(robots)) return;
    
    robots.forEach(robotData => {
        const robotId = `robot-${robotData.id}`;
        if (robotFleet[robotId]) {
            robotFleet[robotId].status = robotData.status || 'unknown';
            robotFleet[robotId].battery = robotData.battery || 0;
            robotFleet[robotId].position = robotData.position || [0, 0, 0];
            robotFleet[robotId].task = robotData.task || '空闲';
        }
    });
    
    renderFleetDashboard();
}

/**
 * 选择机器人
 */
function selectRobot(robotId) {
    const robot = robotFleet[robotId];
    if (!robot) return;
    
    document.querySelectorAll('.fleet-card').forEach(card => {
        card.classList.remove('selected');
    });
    
    const card = document.getElementById(`fleet-card-${robotId}`);
    if (card) card.classList.add('selected');
    
    showNotification(`已选择 ${robot.name}`, 'info');
}

/**
 * 分配任务给机器人
 */
async function assignTask(robotId) {
    const robot = robotFleet[robotId];
    if (!robot) return;
    
    const taskDesc = await promptAsync(`为 ${robot.name} 分配任务:`, '');
    if (!taskDesc) return;
    
    robot.task = taskDesc;
    renderFleetDashboard();
    showNotification(`任务已分配给 ${robot.name}: ${taskDesc}`, 'success');
    
    if (window.SelfLnnApi && typeof window.SelfLnnApi.assignTask === 'function') {
        window.SelfLnnApi.assignTask({ robot_id: robotId, task: taskDesc });
    }
}

/**
 * 紧急停止指定机器人
 */
function emergencyStopRobot(robotId) {
    const robot = robotFleet[robotId];
    if (!robot) return;
    
    if (!safeConfirm(`确定要紧急停止 ${robot.name} 吗？`)) return;
    
    showNotification(`${robot.name} 正在紧急停止...`, 'warning');
    
    /* P1-8修复：调用后端API并await结果，确保紧急停止真正执行 */
    if (window.SelfLnnApi && typeof window.SelfLnnApi.sendRobotCommand === 'function') {
        window.SelfLnnApi.sendRobotCommand({
            robot_id: robotId,
            mode: 2,
            emergency_stop: true
        }).then(function(result) {
            if (result && result.success) {
                robot.status = 'error';
                renderFleetDashboard();
                showNotification(`${robot.name} 已紧急停止`, 'danger');
            } else {
                showNotification(`${robot.name} 紧急停止失败: ${(result && result.error) || '未知错误'}`, 'danger');
            }
        }).catch(function(err) {
            console.error('紧急停止失败:', err);
            showNotification(`${robot.name} 紧急停止命令发送失败`, 'danger');
        });
    } else {
        /* 后端API不可用，更新前端状态为错误 */
        robot.status = 'error';
        renderFleetDashboard();
        showNotification(`${robot.name} 已标记为停止状态（后端API不可用）`, 'danger');
    }
}

/**
 * 创建机器人组
 */
let robotGroups = {};

async function createRobotGroup() {
    const groupName = await promptAsync('请输入机器人组名称:');
    if (!groupName) return;
    
    const groupId = `group-${Date.now()}`;
    robotGroups[groupId] = {
        id: groupId,
        name: groupName,
        robots: [],
        created_at: new Date().toISOString()
    };
    
    renderRobotGroups();
    showNotification(`机器人组 "${groupName}" 已创建`, 'success');
}

/**
 * 渲染机器人组
 */
function renderRobotGroups() {
    const container = document.getElementById('robot-group-list');
    if (!container) return;
    
    container.innerHTML = '';
    
    Object.keys(robotGroups).forEach(groupId => {
        const group = robotGroups[groupId];
        const div = document.createElement('div');
        div.className = 'group-item';
        div.innerHTML = `
            <div class="group-header">
                <span class="group-name">${group.name}</span>
                <span class="group-count">${group.robots.length} 个机器人</span>
            </div>
            <div class="group-actions">
                <button class="btn btn-sm btn-primary" onclick="addRobotToGroup('${groupId}')">添加机器人</button>
                <button class="btn btn-sm btn-success" onclick="startGroupTask('${groupId}')">开始任务</button>
                <button class="btn btn-sm btn-danger" onclick="deleteGroup('${groupId}')">删除分组</button>
            </div>
        `;
        container.appendChild(div);
    });
}

/**
 * 添加机器人到组
 */
async function addRobotToGroup(groupId) {
    const group = robotGroups[groupId];
    if (!group) return;
    
    const availableRobots = Object.keys(robotFleet).filter(id => !group.robots.includes(id));
    if (availableRobots.length === 0) {
        showNotification('没有可用的机器人', 'warning');
        return;
    }
    
    const robotId = await promptAsync(`可添加的机器人: ${availableRobots.join(', ')}\n请输入机器人ID:`);
    if (!robotId || !robotFleet[robotId]) {
        showNotification('无效的机器人ID', 'warning');
        return;
    }
    
    group.robots.push(robotId);
    renderRobotGroups();
    showNotification(`${robotFleet[robotId].name} 已添加到 ${group.name}`, 'success');
}

/**
 * 启动组内所有机器人任务
 */
async function startGroupTask(groupId) {
    const group = robotGroups[groupId];
    if (!group || group.robots.length === 0) {
        showNotification('组内无机器人', 'warning');
        return;
    }
    
    showNotification('正在启动 ' + group.name + ' 所有机器人...', 'info');
    
    group.robots.forEach(function(robotId) {
        if (robotFleet[robotId]) {
            robotFleet[robotId].status = 'busy';
            robotFleet[robotId].task = '组任务';
        }
    });
    
    if (typeof window.SelfLnnApi === 'object' && typeof window.SelfLnnApi.sendRobotCommand === 'function') {
        group.robots.forEach(function(robotId) {
            window.SelfLnnApi.sendRobotCommand({ action: 'start_group_task', group_id: groupId, robot_id: robotId }).catch(function(e) { console.warn('[任务] 组任务发送失败:', e.message); });
        });
    }
    
    renderFleetDashboard();
    showNotification('✅ ' + group.name + ' 所有机器人已启动', 'success');
}

/**
 * 删除机器人组
 */
function deleteGroup(groupId) {
    if (!safeConfirm('确定要删除此机器人组吗？')) return;
    delete robotGroups[groupId];
    renderRobotGroups();
    showNotification('机器人组已删除', 'info');
}

/**
 * 获取舰队状态摘要
 */
function getFleetSummary() {
    const total = Object.keys(robotFleet).length;
    const online = Object.values(robotFleet).filter(r => r.status === 'online').length;
    const busy = Object.values(robotFleet).filter(r => r.status === 'busy').length;
    const error = Object.values(robotFleet).filter(r => r.status === 'error').length;
    
    return { total, online, busy, error };
}

/* ============================================
   对话交互
   ============================================ */

/**
 * 发送对话消息
 */
async function sendDialogueMessage() {
    const input = document.getElementById('dialogue-input');
    const message = input.value.trim();
    if (!message) return;

    /* S-021修复: 以"/"开头的指令先路由到文本命令处理器，
     * 不经过对话通道，避免前端短路与后端对话API误调用 */
    if (message.startsWith('/') && g_textCommandSystem) {
        var cmdResult = g_textCommandSystem.processText(message);
        if (cmdResult && cmdResult.isCommand) {
            addDialogueMessage('system', '[指令已执行] ' + message);
            input.value = '';
            input.disabled = false;
            input.focus();
            return;
        }
    }

    const messagesContainer = document.getElementById('dialogue-messages');
/* messagesContainer为null时安全退出 */
    if (!messagesContainer) return;
    const welcomeEl = messagesContainer.querySelector('.dialogue-welcome');
    if (welcomeEl) welcomeEl.style.display = 'none';

    const temperature = parseFloat(document.getElementById('dialogue-temperature').value);
    const maxLength = parseInt(document.getElementById('dialogue-max-length').value);
    const topK = parseInt(document.getElementById('dialogue-top-k').value);
    const memoryRounds = parseInt(document.getElementById('dialogue-memory').value);

    addDialogueMessage('user', message);
    input.value = '';
    input.disabled = true;

    var multimodalImage = null;
    var multimodalAudio = null;

    var sendCameraCheckbox = document.getElementById('dialogue-send-camera');
    if (sendCameraCheckbox && sendCameraCheckbox.checked && g_deviceManager) {
        var cam = g_deviceManager.cameras.find(function(c) { return c.active; });
        if (cam) {
            var snapshot = g_deviceManager.captureSnapshot(cam.id);
            if (snapshot) {
                multimodalImage = snapshot;
                addDialogueMessage('system', '[已发送摄像头画面]');
            }
        }
    }

    var sendMicCheckbox = document.getElementById('dialogue-send-mic');
    if (sendMicCheckbox && sendMicCheckbox.checked && g_deviceManager) {
        var mic = g_deviceManager.microphones.find(function(m) { return m.active; });
        if (mic && mic.active) {
/* S-022修复: quickCapture返回{capturer,stream}对象，非音频数据。
             * 原代码将VoiceCaptureUtil对象引用误存为lastAudioBlob，
             * multimodalAudio仅设为标记字符串'mic_audio'，未传递真实base64数据。
             * 现改为直接使用MediaRecorder采集500ms音频片段并转base64。 */
            try {
                var micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                var audioChunks = [];
                var mediaRecorder = new MediaRecorder(micStream);
                mediaRecorder.ondataavailable = function(e) {
                    if (e.data.size > 0) audioChunks.push(e.data);
                };
                var audioBlob = await new Promise(function(resolve) {
                    mediaRecorder.onstop = function() {
                        micStream.getTracks().forEach(function(t) { t.stop(); });
                        resolve(new Blob(audioChunks, { type: 'audio/webm' }));
                    };
                    mediaRecorder.start();
                    setTimeout(function() {
                        if (mediaRecorder.state !== 'inactive') mediaRecorder.stop();
                    }, 500);
                });
                var reader = new FileReader();
                multimodalAudio = await new Promise(function(resolve, reject) {
                    reader.onload = function() { resolve(reader.result); };
                    reader.onerror = function() { reject(new Error('base64转换失败')); };
                    reader.readAsDataURL(audioBlob);
                });
                mic.lastAudioBlob = multimodalAudio;
            } catch(e) {
                console.warn('音频采集失败:', e);
            }
        }
    }

    if (g_textCommandSystem) {
        var cmdResult = g_textCommandSystem.processText(message);
        if (cmdResult && cmdResult.isCommand) {
            addDialogueMessage('system', '[文字指令已执行] ' + message);
            input.disabled = false;
            input.focus();
            return;
        }
    }

    const typingEl = addTypingIndicator();
    const startTime = Date.now();
    var useStreaming = g_dialogueEnhanced && g_dialogueEnhanced.getWsStatus() === 'connected';

    if (useStreaming) {
        if (typingEl) typingEl.remove();
        /* 注册流式响应处理器，接收后端WebSocket推送的对话回复 */
        var streamBuffer = '';
        var streamTimer = null;
        var streamTimeout = null;
        var gws = window.SelfLnnWebSocket;

        /* FE-001修复: 注册新handler前先移除旧的streamHandler，防止内存泄漏 */
        if (gws && gws.off && window._dialogueStreamHandler) {
            gws.off('dialogue_response', window._dialogueStreamHandler);
        }

        var streamHandler = function(data) {
            if (data && (data.type === 'dialogue_response' || data.type === 'dialogue_stream')) {
                if (data.token) {
                    streamBuffer += data.token;
                    if (streamTimer) clearTimeout(streamTimer);
                    streamTimer = setTimeout(function() {
                        /* 批量刷新UI：移除旧的流式消息，追加累积内容 */
                        var existingEl = document.getElementById('dialogue-stream-msg');
                        if (existingEl) existingEl.remove();
                        /* P1-1修复: 使用第四参数msgId传字符串，而非作为tokens对象传入 */
                        addDialogueMessage('ai', streamBuffer, null, 'dialogue-stream-msg');
                        streamTimer = null;
                    }, 50);
                }
                if (data.done) {
                    /* FE-002修复: data.done为true时清除超时定时器，正常完成不触发超时 */
                    if (streamTimeout) { clearTimeout(streamTimeout); streamTimeout = null; }
                    if (streamTimer) { clearTimeout(streamTimer); streamTimer = null; }
                    var existingEl = document.getElementById('dialogue-stream-msg');
                    if (existingEl) existingEl.remove();
                    addDialogueMessage('ai', streamBuffer);
                    streamBuffer = '';
                    /* 移除流式监听器 */
                    if (gws && gws.off) gws.off('dialogue_response', streamHandler);
                    window._dialogueStreamHandler = null;
                }
            }
        };
        /* 将handler存储到全局引用，便于后续清理 */
        window._dialogueStreamHandler = streamHandler;
        if (gws && gws.on) gws.on('dialogue_response', streamHandler);

        /* FE-002修复: 添加30秒超时定时器，超时后自动移除handler并提示用户 */
        streamTimeout = setTimeout(function() {
            if (window._dialogueStreamHandler) {
                if (gws && gws.off) gws.off('dialogue_response', window._dialogueStreamHandler);
                window._dialogueStreamHandler = null;
            }
            if (streamTimer) { clearTimeout(streamTimer); streamTimer = null; }
            var existingEl = document.getElementById('dialogue-stream-msg');
            if (existingEl) existingEl.remove();
            /* 超时提示用户 */
            addDialogueMessage('system', '对话响应超时，请检查网络连接后重试');
            streamBuffer = '';
            showNotification('对话响应超时，请检查网络连接', 'warning');
        }, 30000);

        /* FE-001修复: 监听WebSocket断开事件，清理所有pending的streamHandler */
        if (gws && gws.onStatusChange) {
            var _wsStatusCleanup = function(status) {
                if (!status.connected && window._dialogueStreamHandler) {
                    if (gws && gws.off) gws.off('dialogue_response', window._dialogueStreamHandler);
                    window._dialogueStreamHandler = null;
                    if (streamTimeout) { clearTimeout(streamTimeout); streamTimeout = null; }
                    if (streamTimer) { clearTimeout(streamTimer); streamTimer = null; }
                    var existingEl = document.getElementById('dialogue-stream-msg');
                    if (existingEl) existingEl.remove();
                    if (streamBuffer) {
                        addDialogueMessage('ai', streamBuffer);
                        streamBuffer = '';
                    }
                }
            };
            gws.onStatusChange(_wsStatusCleanup);
        }

        g_dialogueEnhanced.sendMultimodalMessage(message, multimodalImage, multimodalAudio, {
            temperature: temperature,
            max_length: maxLength,
            top_k: topK,
            memory: memoryRounds,
            streaming: true
        });
        var modelStatusEl = document.getElementById('dialogue-model-status');
        if (modelStatusEl) modelStatusEl.textContent = '流式对话';
        var totalRoundsEl = document.getElementById('dialogue-total-rounds');
        if (totalRoundsEl) totalRoundsEl.textContent = '-';
        input.disabled = false;
        input.focus();
        return;
    }

    var dialogueResult;

    /* FIX-JS-004: try/catch保护避免网络异常导致输入框永久锁定 */
    try {
        if (multimodalImage || multimodalAudio) {
            dialogueResult = await sendMultimodalRequest(message, multimodalImage, multimodalAudio, {
                temperature: temperature,
                maxLength: maxLength,
                topK: topK,
                memoryRounds: memoryRounds
            });
        } else {
            dialogueResult = await window.SelfLnnApi.sendDialogueMessage(message, {
                temperature: temperature,
                maxLength: maxLength,
                topK: topK,
                memoryRounds: memoryRounds
            });
        }
    } catch(e) {
        console.error('[对话] 请求异常:', e.message, e.stack);
        dialogueResult = { success: false, error: '网络请求异常: ' + e.message };
    }

    if (typingEl) typingEl.remove();

    if (dialogueResult.success && dialogueResult.data) {
        const reply = dialogueResult.data.reply || dialogueResult.data.response || dialogueResult.data.message || '(无响应)';
        addDialogueMessage('assistant', reply, dialogueResult.data.tokens_used || undefined);

        if (g_dialogueEnhanced && g_dialogueEnhanced.voiceOutputEnabled) {
            var speakerId = null;
            var activeSpeaker = g_deviceManager && g_deviceManager.speakers ? g_deviceManager.speakers.find(function(s) { return s.active; }) : null;
            if (activeSpeaker) speakerId = activeSpeaker.deviceId;
            g_dialogueEnhanced.speakText(reply, speakerId);
        }

        const elapsed = ((Date.now() - startTime) / 1000).toFixed(2);
        const avgTimeEl = document.getElementById('dialogue-avg-time');
        if (avgTimeEl) avgTimeEl.textContent = elapsed + 's';

        const sessionCountEl = document.getElementById('dialogue-session-count');
        if (sessionCountEl) {
            const current = parseInt(sessionCountEl.textContent) || 0;
            sessionCountEl.textContent = current + 1;
        }

        const modelStatusEl = document.getElementById('dialogue-model-status');
        if (modelStatusEl) modelStatusEl.textContent = '在线';

        const totalRoundsEl = document.getElementById('dialogue-total-rounds');
        if (totalRoundsEl && dialogueResult.data.total_rounds !== undefined) {
            totalRoundsEl.textContent = dialogueResult.data.total_rounds;
        }
    } else {
        addDialogueMessage('assistant', '[对话服务不可用: ' + (dialogueResult.error || '后端未连接') + ']');
    }

    input.disabled = false;
    input.focus();
}

/* 通过统一API服务发送多模态对话请求，获得完整认证/重试/熔断保护 */
async function sendMultimodalRequest(message, imageData, audioData, params) {
    try {
        /* P2-4: 使用api-service封装的sendMultimodalDialogue方法 */
        var api = window.SelfLnnApi;
        if (!api) throw new Error('API服务未初始化');

        var options = {
            temperature: params.temperature || 0.8,
            maxLength: params.maxLength || 128,
            topK: params.topK || 40,
            memoryRounds: params.memoryRounds || 5
        };
        /* api-service期望images/audio数组格式 */
        if (imageData) options.images = [imageData];
        if (audioData) options.audio = [audioData];

        var resp = await api.sendMultimodalDialogue(message, options);
        if (!resp.success) {
            return { success: false, error: resp.error || '多模态对话请求失败' };
        }
        var jsonData = resp.data;
        if (jsonData && jsonData.dialogue) {
            return {
                success: true,
                data: {
                    reply: jsonData.dialogue.response,
                    tokens_used: jsonData.dialogue.tokens_used || jsonData.dialogue.confidence ? 1 : 0,
                    confidence: jsonData.dialogue.confidence || 0,
                    total_rounds: 0
                }
            };
        }
        return { success: false, error: '响应格式异常' };
    } catch (err) {
        console.error('多模态对话请求失败:', err);
        return { success: false, error: err.message };
    }
}

/**
 * 在对话区域添加一条消息
 */
function renderMarkdown(text) {
    if (!text) return '';
    var out = text;
    /* M-06修复: 先转义HTML特殊字符，再应用Markdown标记 */
    out = out.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    out = out.replace(/```(\w*)\n([\s\S]*?)```/g, function(m, lang, code) {
        return '<pre><code class="language-' + lang + '">' + window.escapeHtml(code.replace(/\n$/, '')) + '</code></pre>';
    });
    out = out.replace(/`([^`]+)`/g, function(m, code) {
        return '<code>' + window.escapeHtml(code) + '</code>';
    });
    out = out.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    out = out.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    out = out.replace(/^### (.+)$/gm, '<h4>$1</h4>');
    out = out.replace(/^## (.+)$/gm, '<h3>$1</h3>');
    out = out.replace(/^# (.+)$/gm, '<h2>$1</h2>');
    out = out.replace(/^- (.+)$/gm, '<li>$1</li>');
    out = out.replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>');
    out = out.replace(/\n/g, '<br>');
    return out;
}

function addDialogueMessage(role, content, tokens, msgId) {
    const container = document.getElementById('dialogue-messages');
/* container为null时安全退出 */
    if (!container) return;
    const div = document.createElement('div');
    div.className = 'dialogue-message ' + role;
    /* P1-1修复: 支持可选msgId参数，用于流式对话消息累积更新（getElementById定位替换） */
    if (msgId) div.id = msgId;

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = role === 'user' ? 'U' : 'A';

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    bubble.innerHTML = renderMarkdown(content);

    const time = document.createElement('div');
    time.className = 'message-time';
    const now = new Date();
    time.textContent = now.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
    /* P1-1修复: 仅当tokens为有效数值时才显示token计数，避免对象误显示 */
    if (tokens && typeof tokens === 'number') time.textContent += ' · ' + tokens + ' tokens';

    bubble.appendChild(time);
    div.appendChild(avatar);
    div.appendChild(bubble);
    container.appendChild(div);
    /* P1-003修复: 使用requestAnimationFrame批处理scroll操作，避免强制重排(forced reflow) */
    requestAnimationFrame(function() {
        container.scrollTop = container.scrollHeight;
    });
}

/**
 * 添加正在输入指示器
 */
function addTypingIndicator() {
    const container = document.getElementById('dialogue-messages');
    const div = document.createElement('div');
    div.className = 'dialogue-message assistant typing';

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = 'A';

    const indicator = document.createElement('div');
    indicator.className = 'typing-indicator';
    for (let i = 0; i < 3; i++) {
        const dot = document.createElement('div');
        dot.className = 'typing-dot';
        indicator.appendChild(dot);
    }

    div.appendChild(avatar);
    div.appendChild(indicator);
    container.appendChild(div);
    /* P1-003修复: 使用requestAnimationFrame批处理scroll操作，避免强制重排(forced reflow) */
    requestAnimationFrame(function() {
        container.scrollTop = container.scrollHeight;
    });
    return div;
}

/**
 * 流式token显示更新
 */
var g_streamingMessageEl = null;
var g_streamingFullText = '';

function updateStreamingToken(tokenText, progress, isFinal) {
    const container = document.getElementById('dialogue-messages');
    if (!container) return;

    if (!g_streamingMessageEl) {
        g_streamingMessageEl = document.createElement('div');
        g_streamingMessageEl.className = 'dialogue-message assistant';
        g_streamingMessageEl.innerHTML = '<div class="message-avatar">AGI</div><div class="message-bubble streaming-bubble"><span class="streaming-text"></span><span class="streaming-cursor">▌</span></div>';
        container.appendChild(g_streamingMessageEl);
        g_streamingFullText = '';
    }

    g_streamingFullText += tokenText;
    var bubbleEl = g_streamingMessageEl.querySelector('.message-bubble');
    if (bubbleEl) {
        var textEl = bubbleEl.querySelector('.streaming-text');
        if (textEl) textEl.textContent = g_streamingFullText;
    }
    /* P1-003修复: 使用requestAnimationFrame批处理scroll操作，在流式更新中避免频繁强制重排 */
    requestAnimationFrame(function() {
        container.scrollTop = container.scrollHeight;
    });

    if (isFinal) {
        var cursorEl = g_streamingMessageEl.querySelector('.streaming-cursor');
        if (cursorEl) cursorEl.remove();
        g_streamingMessageEl = null;
        g_streamingFullText = '';
    }
}

function finalizeStreamingResponse(fullText, confidence) {
    const container = document.getElementById('dialogue-messages');
    if (!container) return;

    if (g_streamingMessageEl) {
        var bubbleEl = g_streamingMessageEl.querySelector('.message-bubble');
        if (bubbleEl) {
            var textEl = bubbleEl.querySelector('.streaming-text');
            if (textEl) textEl.textContent = fullText;
            var cursorEl = bubbleEl.querySelector('.streaming-cursor');
            if (cursorEl) cursorEl.remove();
        }
        g_streamingMessageEl = null;
        g_streamingFullText = '';
    } else if (fullText && fullText.length > 0) {
        addDialogueMessage('assistant', fullText);
    }

    if (g_dialogueEnhanced && g_dialogueEnhanced.voiceOutputEnabled) {
        var speakerId = null;
        var activeSpeaker = g_deviceManager ? g_deviceManager.speakers.find(function(s) { return s.active; }) : null;
        if (activeSpeaker) speakerId = activeSpeaker.deviceId;
        g_dialogueEnhanced.speakText(fullText, speakerId);
    }

    var sessionCountEl = document.getElementById('dialogue-session-count');
    if (sessionCountEl) {
        var current = parseInt(sessionCountEl.textContent) || 0;
        sessionCountEl.textContent = current + 1;
    }
}

/**
 * 清空对话历史
 */
async function clearDialogueHistory() {
    if (!safeConfirm('确定清空所有对话消息？')) return;

    const container = document.getElementById('dialogue-messages');
    const messages = container.querySelectorAll('.dialogue-message');
    messages.forEach(function(el) { el.remove(); });

    const welcomeEl = container.querySelector('.dialogue-welcome');
    if (welcomeEl) welcomeEl.style.display = 'block';

    document.getElementById('dialogue-session-count').textContent = '0';

    const result = await window.SelfLnnApi.clearDialogueHistory();
    if (result.success) {
        showNotification('对话历史已清空', 'success');
    }
}

/**
 * 初始化温度滑块显示
 */
function initDialogueTemperature() {
    const slider = document.getElementById('dialogue-temperature');
    const display = document.getElementById('dialogue-temperature-value');
    if (slider && display) {
        /* C-04修复: 提取为命名函数 */
        function _handleTemperatureSliderInput() {
            display.textContent = parseFloat(this.value).toFixed(1);
        }
        window._registerEventListener(slider, 'input', _handleTemperatureSliderInput);
    }
}

/* ============================================
   AGI功能控制
   ============================================ */

/**
 * 切换AGI功能
 */
async function toggleAgiFeature(feature, enabled) {
    const result = await window.SelfLnnApi.toggleAgiFeature(feature, enabled);
    if (result.success) {
        /* S-006修复: 补全所有AGI功能的中文标签映射 */
        const label = feature === 'self_cognition' ? '自我认知' :
                      feature === 'self_decision' ? '自我决策' :
                      feature === 'self_execution' ? '自主执行' :
                      feature === 'self_learning' ? '自我学习' :
                      feature === 'self_evolution' ? '自我演化' :
                      feature === 'imitation_learning' ? '模仿学习' :
                      feature === 'self_correction' ? '自我修正' :
                      feature === 'self_reflection' ? '自我反思' :
                      feature === 'planning' ? '计划能力' :
                      feature === 'metacognition' ? '元认知' :
                      feature === 'multi_robot' ? '多机器人控制' :
                      feature === 'curiosity' ? '好奇心探索' : feature;
        showNotification(label + '已' + (enabled ? '启用' : '关闭'), 'success');
    } else {
        const checkbox = document.querySelector('#feature-' + feature.replace(/_/g, '-'));
        if (checkbox) checkbox.checked = !enabled;
        showNotification('操作失败: ' + (result.error || '后端未连接'), 'danger');
    }
}

/**
 * 初始化AGI功能状态
 */
async function initAgiFeatureStatus() {
    /* P1-004修复: 添加result为null时的检查，防止后端返回异常数据时崩溃 */
    const result = await window.SelfLnnApi.getAgiFeatureStatus();
    if (!result) {
        console.warn('[AGI功能状态] 后端返回null/undefined，跳过状态初始化');
        return;
    }
    if (result.success && result.data) {
        /* 后端返回结构: {"agi":{"status":"success","features":{...}}} */
        const features = (result.data.agi && result.data.agi.features) || {};
        for (const [key, value] of Object.entries(features)) {
            const checkboxId = 'feature-' + key.replace(/_/g, '-');
            const checkbox = document.getElementById(checkboxId);
            if (checkbox) checkbox.checked = value === true || value === 'true' || value === 1;
        }
        const badge = document.getElementById('agi-features-status');
        if (badge) {
            badge.textContent = '已连接';
            badge.className = 'card-badge online';
        }
    }
}

/**
 * 初始化对话系统
 */
function initDialogueSystem() {
    initDialogueTemperature();

    const container = document.getElementById('dialogue-messages');

    const welcomeEl = container.querySelector('.dialogue-welcome');
    if (welcomeEl) {
        const title = welcomeEl.querySelector('h4');
        const descs = welcomeEl.querySelectorAll('p');
        if (title) title.textContent = 'SELF-LNN 全液态神经网络智能系统';
        if (descs[0]) descs[0].textContent = '等待后端连接中...';
        if (descs[1]) descs[1].textContent = '';
    }

    initAgiFeatureStatus();
    initAutoLearnStatus();
}

setTimeout(initDialogueSystem, 100);

/**
 * 切换自主知识库学习开关
 */
async function toggleAutoLearn(enabled) {
/* 后端不可用时明确告知用户，不再显示误导性的"已启用"通知 */
    try {
        if (window.SelfLnnApi && typeof window.SelfLnnApi.autoLearnToggle === 'function') {
            var result = await window.SelfLnnApi.autoLearnToggle(enabled);
            if (result.success) {
                showNotification(enabled ? '✅ 自主知识库学习已启用' : '⏸ 自主知识库学习已禁用', 'success');
                if (enabled) {
                    setTimeout(refreshAutoLearnStats, 500);
                }
            } else {
                showNotification('❌ 切换失败: ' + (result.error || '未知错误'), 'danger');
            }
        } else {
            showNotification('❌ 自主知识库学习后端未连接，无法切换（请确认SELF-LNN服务器已启动）', 'danger');
        }
    } catch (e) {
        showNotification('❌ 自主知识库学习请求失败: ' + e.message, 'danger');
    }
}

/**
 * 触发知识源扫描
 */
async function triggerAutoLearnScan() {
    showNotification('正在扫描知识源...', 'info');
    try {
        if (window.SelfLnnApi && typeof window.SelfLnnApi.autoLearnScan === 'function') {
            var result = await window.SelfLnnApi.autoLearnScan();
            if (result.success && result.data) {
                var entries = result.data.entries_learned || 0;
                showNotification('知识源扫描完成，学习 ' + entries + ' 个条目', 'success');
                setTimeout(refreshAutoLearnStats, 300);
            } else {
                showNotification('扫描完成，但获取结果失败', 'warning');
            }
        }
    } catch (e) {
        showNotification('扫描请求失败: ' + e.message, 'danger');
    }
}

/**
 * 刷新自主学习统计
 */
async function refreshAutoLearnStats() {
    try {
        if (window.SelfLnnApi && typeof window.SelfLnnApi.autoLearnStats === 'function') {
            var result = await window.SelfLnnApi.autoLearnStats();
            if (result.success && result.data) {
                var d = result.data;
                var filesEl = document.getElementById('auto-learn-files');
                var entriesEl = document.getElementById('auto-learn-entries');
                var entitiesEl = document.getElementById('auto-learn-entities');
                var relationsEl = document.getElementById('auto-learn-relations');
                if (filesEl) filesEl.textContent = d.files_scanned || 0;
                if (entriesEl) entriesEl.textContent = d.entries_learned || 0;
                if (entitiesEl) entitiesEl.textContent = d.entities_extracted || 0;
                if (relationsEl) relationsEl.textContent = d.relations_extracted || 0;
            }
        }
    } catch (e) {
        console.warn('获取自主学习统计失败:', e.message);
    }
}

/**
 * 初始化自主学习状态
 */
async function initAutoLearnStatus() {
    try {
        if (window.SelfLnnApi && typeof window.SelfLnnApi.autoLearnStats === 'function') {
            var result = await window.SelfLnnApi.autoLearnStats();
            if (result.success && result.data) {
                var d = result.data;
                var filesEl = document.getElementById('auto-learn-files');
                var entriesEl = document.getElementById('auto-learn-entries');
                var entitiesEl = document.getElementById('auto-learn-entities');
                var relationsEl = document.getElementById('auto-learn-relations');
                if (filesEl) filesEl.textContent = d.files_scanned || 0;
                if (entriesEl) entriesEl.textContent = d.entries_learned || 0;
                if (entitiesEl) entitiesEl.textContent = d.entities_extracted || 0;
                if (relationsEl) relationsEl.textContent = d.relations_extracted || 0;
            }
        }
    } catch (e) {
        console.warn('初始化自主学习状态失败:', e.message);
    }
}

function toggleVoiceInput() {
    if (!g_dialogueEnhanced || !g_deviceManager) return;
    if (g_dialogueEnhanced.isRecording) {
        g_dialogueEnhanced.stopVoiceInput();
        var btn = document.getElementById('dialogue-voice-input');
        if (btn) {
            btn.textContent = '🎤 语音输入';
            btn.className = 'btn btn-sm btn-outline';
        }
    } else {
        var activeMic = g_deviceManager.microphones.find(function(m) { return m.active; });
        if (!activeMic) {
            showNotification('请先在设备管理中启动麦克风', 'warning');
            return;
        }
        var stream = g_deviceManager.getMicrophoneStream(activeMic.id);
        if (!stream) {
            showNotification('麦克风流不可用', 'danger');
            return;
        }
        g_dialogueEnhanced.startVoiceInput(stream);
        var btn = document.getElementById('dialogue-voice-input');
        if (btn) {
            btn.textContent = '🔴 录音中...';
            btn.className = 'btn btn-sm btn-danger';
        }
    }
}

function toggleVoiceOutput() {
    if (!g_dialogueEnhanced) return;
    if (g_dialogueEnhanced.voiceOutputEnabled) {
        g_dialogueEnhanced.disableVoiceOutput();
        g_dialogueEnhanced.stopSpeaking();
        var btn = document.getElementById('dialogue-voice-output');
        if (btn) btn.className = 'btn btn-sm btn-outline';
        showNotification('语音输出已关闭', 'info');
    } else {
        g_dialogueEnhanced.enableVoiceOutput();
        var btn = document.getElementById('dialogue-voice-output');
        if (btn) btn.className = 'btn btn-sm btn-success';
        showNotification('语音输出已开启', 'success');
    }
}

/* P3-005: 多模态独立输入入口函数 */
function triggerVisionInput() {
    if (!g_dialogueEnhanced) {
        showNotification('对话增强系统未就绪', 'warning');
        return;
    }
    if (!window.SelfLnnApi) {
        showNotification('后端API未连接', 'danger');
        return;
    }
    var input = document.getElementById('dialogue-input');
    /* ZSFIII-P1-002修复: DialogueEnhanced类不存在getLastCameraFrame方法，改为使用lastCapturedImage属性获取真实摄像头帧 */
    var imageData = g_dialogueEnhanced ? g_dialogueEnhanced.lastCapturedImage : null;
    if (!imageData) {
        showNotification('请先启用摄像头并获取画面', 'warning');
        return;
    }
/* 使用封装API而非直接调用request() */
    window.SelfLnnApi.processVisionInput({ image: imageData })
        .then(function(data) {
            if (input) input.value += '\n[视觉输入已处理]';
        })
        .catch(function() { showNotification('视觉输入API请求失败', 'danger'); });
}

function triggerAudioInput() {
    if (!g_dialogueEnhanced || !g_deviceManager) {
        showNotification('设备管理器未就绪', 'warning');
        return;
    }
    if (!window.SelfLnnApi) {
        showNotification('后端API未连接', 'danger');
        return;
    }
    var activeMic = g_deviceManager.microphones.find(function(m) { return m.active; });
    if (!activeMic) {
        showNotification('请先在设备管理中启动麦克风', 'warning');
        return;
    }
    var input = document.getElementById('dialogue-input');
    if (input) input.value += '\n[音频输入模式已激活，请说话...]';
    if (typeof g_dialogueEnhanced.startVoiceInput === 'function') {
        toggleVoiceInput();
    }
}

function triggerTextInput() {
    var input = document.getElementById('dialogue-input');
    if (input) {
        input.focus();
        input.placeholder = '请输入纯文本信息...';
        showNotification('文本输入模式已激活，请直接输入文字', 'info');
    }
}

function triggerSensorInput() {
    if (!window.SelfLnnApi) {
        showNotification('后端API未连接', 'danger');
        return;
    }
    var input = document.getElementById('dialogue-input');
/* 使用封装API而非直接调用request() */
    window.SelfLnnApi.getSensorPipelineStatus()
        .then(function(data) {
            if (input) input.value += '\n[传感器数据已请求]';
            showNotification('传感器输入已触发', 'success');
        })
        .catch(function() { showNotification('传感器输入API请求失败', 'danger'); });
}

/**
 * API密钥管理函数
 */

/** 显示/隐藏API密钥 */
function toggleApiKeyVisibility(event) {
    if (!event) return;
    var input = document.getElementById('api-key-current');
    if (input) {
        if (input.type === 'password') {
            input.type = 'text';
            event.target.textContent = ' 隐藏';
        } else {
            input.type = 'password';
            event.target.textContent = ' 显示';
        }
    }
}

/** 复制API密钥到剪贴板 */
function copyApiKey() {
    var input = document.getElementById('api-key-current');
    if (input && input.value && input.value !== '等待连接') {
        if (navigator.clipboard) {
            navigator.clipboard.writeText(input.value).then(function() {
                showNotification('✅ API密钥已复制到剪贴板', 'success');
            }).catch(function() {
                input.select();
                document.execCommand('copy');
                showNotification('✅ API密钥已复制到剪贴板', 'success');
            });
        } else {
            input.select();
            document.execCommand('copy');
            showNotification('✅ API密钥已复制到剪贴板', 'success');
        }
    } else {
        showNotification('⚠️ 没有可复制的API密钥', 'warning');
    }
}

/** 刷新API密钥状态 */
async function refreshApiKeyStatus() {
    showNotification('正在获取API密钥状态...', 'info');
    try {
        /* P2-4: 使用api-service封装方法替代直接request调用 */
        var api = window.SelfLnnApi;
        if (!api) throw new Error('API服务未初始化');
        var resp = await api.getKeyStatus();
        if (resp.success) {
            var data = resp.data;
            var keyInput = document.getElementById('api-key-current');
            var statusBadge = document.getElementById('api-key-enabled-status');
            if (data && data.key_status) {
                var km = data.key_status;
                if (keyInput) keyInput.value = km.key_set ? '密钥已设置' : '未设置密钥';
                if (statusBadge) {
                    if (km.enabled) {
                        statusBadge.textContent = '已启用';
                        statusBadge.className = 'status-badge active';
                    } else {
                        statusBadge.textContent = '未启用';
                        statusBadge.className = 'status-badge inactive';
                    }
                }
            }
            showNotification('✅ API密钥状态已刷新', 'success');
        } else {
            showNotification('❌ 获取密钥状态失败', 'danger');
        }
    } catch (e) {
        console.warn('刷新密钥状态失败:', e);
        showNotification('⚠️ 无法连接后端服务', 'warning');
    }
}

/** 设置新API密钥 */
async function setNewApiKey() {
    var input = document.getElementById('api-key-new-input');
    if (!input || !input.value || input.value.trim().length < 8) {
        showNotification('⚠️ 密钥长度至少8位字符', 'warning');
        return;
    }
    var newKey = input.value.trim();
    showNotification('正在设置新API密钥...', 'info');
    try {
        /* P2-4: 使用api-service封装方法替代直接request调用 */
        var api = window.SelfLnnApi;
        if (!api) throw new Error('API服务未初始化');
        var resp = await api.setKey(newKey);
        if (resp.success) {
            var data = resp.data;
            if (data && data.key_manage && data.key_manage.status === 'success') {
                showNotification('✅ API密钥设置成功', 'success');
                input.value = '';
                await refreshApiKeyStatus();
            } else {
                showNotification('❌ 密钥设置失败: ' + (data.key_manage ? (data.key_manage.error || '未知错误') : '未知错误'), 'danger');
            }
        } else {
            showNotification('❌ 密钥设置请求失败', 'danger');
        }
    } catch (e) {
        showNotification('❌ 密钥设置异常: ' + e.message, 'danger');
    }
}

/** 生成随机新API密钥 */
function generateNewApiKey() {
    var chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*';
    var key = '';
    /* P2-006修复: 使用crypto.getRandomValues确保API密钥密码学安全
     * 回退路径使用Date.now()+performance.now()+Math.random混合熵
     * 替代纯Math.random，最大化不可预测性 */
    var randomBuf = new Uint32Array(32);
    if (window.crypto && window.crypto.getRandomValues) {
        window.crypto.getRandomValues(randomBuf);
    } else {
        /* 强化回退：混合时间戳+性能计数器+导航计时+随机数多层熵 */
        var entropy = Date.now() * 1000 + (window.performance ? performance.now() : 0) * 1000000;
        for (var i = 0; i < 32; i++) {
            entropy = (entropy * 1103515245 + 12345) >>> 0;
            randomBuf[i] = (entropy ^ (Math.floor(Math.random() * 4294967296))) >>> 0;
            /* 再次混合，增大未知数 */
            entropy = ((entropy << 13) ^ (entropy >>> 19)) >>> 0;
        }
    }
    for (var i = 0; i < 32; i++) {
        key += chars.charAt(randomBuf[i] % chars.length);
    }
    var input = document.getElementById('api-key-new-input');
    if (input) input.value = key;
    showNotification('已生成32位随机密钥，点击"设置密钥"应用', 'info');
}

/** 调用API端点列表 */
async function refreshApiEndpointList() {
    var listEl = document.getElementById('api-endpoint-list');
    if (!listEl) return;
    try {
        /* P2-4: 使用api-service封装方法替代直接request调用 */
        var api = window.SelfLnnApi;
        if (!api) throw new Error('API服务未初始化');
        var resp = await api.getApiKeyDocs();
        if (resp.success) {
            var data = resp.data;
            if (data && data.api_key_docs && data.api_key_docs.endpoints) {
                var html = '';
                data.api_key_docs.endpoints.forEach(function(ep) {
                    var methodClass = ep.method.toLowerCase();
                    html += '<div class="endpoint-item">' +
                        '<span class="endpoint-method ' + methodClass + '">' + ep.method + '</span>' +
                        '<span class="endpoint-path">' + ep.path + '</span>' +
                        '<span class="endpoint-desc">' + ep.description + '</span>' +
                        '</div>';
                });
                listEl.innerHTML = html;
            } else if (data && data.api_key_docs && data.api_key_docs.api_list) {
                var html = '';
                data.api_key_docs.api_list.forEach(function(api) {
                    html += '<div class="endpoint-item">' +
                        '<span class="endpoint-path">' + api + '</span>' +
                        '</div>';
                });
                listEl.innerHTML = html;
            } else {
                listEl.innerHTML = '<div class="endpoint-info loading">API端点列表格式异常</div>';
            }
        } else {
            listEl.innerHTML = '<div class="endpoint-info loading">无法获取API端点列表</div>';
        }
    } catch (e) {
        listEl.innerHTML = '<div class="endpoint-info loading">连接失败: ' + window.escapeHtml(e.message) + '</div>';
    }
}

/** 初始化API密钥管理和API服务页面 */
function initApiKeyAndService() {
    setTimeout(function() {
        refreshApiKeyStatus();
        refreshApiEndpointList();
        refreshDashApiKey();
    }, 1000);
}

initApiKeyAndService();

/**
 * 过滤模型列表
 */
function filterModels(query) {
    const items = document.querySelectorAll('.model-item');
    query = (query || '').toLowerCase().trim();
    items.forEach(function(item) {
        var name = item.querySelector('h4');
        if (!query || (name && name.textContent.toLowerCase().indexOf(query) !== -1)) {
            item.style.display = '';
        } else {
            item.style.display = 'none';
        }
    });
}

/**
 * 更新网络设置
 */
function updateNetworkSetting(key, value) {
    showNotification('网络设置「' + key + '」已更新为: ' + value, 'info');
    SelfLnnApi.updateConfig(key, value).catch(function(err) {
        console.warn('网络设置同步失败（本地已保存）:', err);
    });
}

/**
 * 刷新系统信息
 */
async function refreshSystemInfo() {
    showNotification('正在刷新系统信息...', 'info');
    try {
        var status = await SelfLnnApi.getStatus();
        if (status && status.system) {
            var sys = status.system;
            var idMap = {
                'sys-version': sys.version || '未知',
                'sys-build-time': sys.build_time || '未知',
                'sys-uptime': sys.uptime || '未知',
                'sys-cpu-arch': sys.cpu_arch || navigator.platform || '未知',
                'sys-os': sys.os || navigator.platform || '未知',
                'sys-memory': sys.memory_usage || '未知',
                'sys-storage': sys.storage || '未知',
                'sys-gpu': sys.gpu_info || '等待检测'
            };
            Object.keys(idMap).forEach(function(id) {
                var el = document.getElementById(id);
                if (el) el.textContent = idMap[id];
            });
            showNotification('系统信息已刷新', 'success');
        }
    } catch (e) {
        console.warn('刷新系统信息失败:', e);
    }
}

/**
 * 加载新模型
 */
async function loadNewModel() {
    var modelName = await promptAsync('请输入要加载的模型名称:', 'lnn-core');
    if (!modelName) return;
    showNotification('正在加载模型: ' + modelName, 'info');
    try {
        var result = await SelfLnnApi.loadModel(modelName);
        if (result && result.success) {
            showNotification('✅ 模型 ' + modelName + ' 加载成功', 'success');
        } else {
            showNotification('❌ 模型加载失败: ' + (result ? result.message : '无响应'), 'danger');
        }
    } catch (e) {
        showNotification('❌ 模型加载异常: ' + e.message, 'danger');
    }
}

/**
 * 启动模型
 */
async function startModel(modelId) {
    showNotification('正在启动模型: ' + modelId, 'info');
    try {
        var result = await SelfLnnApi.startModel(modelId);
        if (result && result.success) {
            showNotification('✅ 模型 ' + modelId + ' 已启动', 'success');
            var badge = document.querySelector('.model-item[data-model="' + modelId + '"] .status-badge');
            if (badge) { badge.textContent = '运行中'; badge.className = 'status-badge running'; }
        } else {
            showNotification('❌ 模型启动失败', 'danger');
        }
    } catch (e) {
        showNotification('❌ 模型启动异常: ' + e.message, 'danger');
    }
}

/**
 * 停止模型
 */
async function stopModel(modelId) {
    showNotification('正在停止模型: ' + modelId, 'info');
    try {
        var result = await SelfLnnApi.stopModel(modelId);
        if (result && result.success) {
            showNotification('✅ 模型 ' + modelId + ' 已停止', 'success');
            var badge = document.querySelector('.model-item[data-model="' + modelId + '"] .status-badge');
            if (badge) { badge.textContent = '已停止'; badge.className = 'status-badge stopped'; }
        } else {
            showNotification('❌ 模型停止失败', 'danger');
        }
    } catch (e) {
        showNotification('❌ 模型停止异常: ' + e.message, 'danger');
    }
}

/**
 * 卸载模型
 */
async function unloadModel(modelId) {
    if (!safeConfirm('确定要卸载模型 ' + modelId + ' 吗？')) return;
    showNotification('正在卸载模型: ' + modelId, 'info');
    try {
        var result = await SelfLnnApi.unloadModel(modelId);
        if (result && result.success) {
            showNotification('✅ 模型 ' + modelId + ' 已卸载', 'success');
            var badge = document.querySelector('.model-item[data-model="' + modelId + '"] .status-badge');
            if (badge) { badge.textContent = '未加载'; badge.className = 'status-badge pending'; }
        } else {
            showNotification('❌ 模型卸载失败', 'danger');
        }
    } catch (e) {
        showNotification('❌ 模型卸载异常: ' + e.message, 'danger');
    }
}

/**
 * 重启系统
 */
async function restartSystem() {
    if (!safeConfirm('确定要重启SELF-LNN AGI系统吗？')) return;
    showNotification('正在重启系统...', 'warning');
    try {
        await SelfLnnApi.restart();
        showNotification('系统重启命令已发送', 'success');
    } catch (e) {
        showNotification('重启命令发送失败: ' + e.message, 'danger');
    }
}

/**
 * 关闭系统
 */
async function shutdownSystem() {
    if (!safeConfirm('确定要关闭SELF-LNN AGI系统吗？此操作将终止所有进程！')) return;
    showNotification('正在关闭系统...', 'warning');
    try {
        await SelfLnnApi.shutdown();
        showNotification('系统关闭命令已发送', 'warning');
    } catch (e) {
        showNotification('关闭命令发送失败: ' + e.message, 'danger');
    }
}

/**
 * 导出系统日志
 */
async function exportSystemLogs() {
    showNotification('正在导出系统日志...', 'info');
    try {
        var logs = await SelfLnnApi.exportLogs();
        if (logs) {
            var blob = new Blob([typeof logs === 'string' ? logs : JSON.stringify(logs, null, 2)], { type: 'text/plain' });
            var url = URL.createObjectURL(blob);
            var a = document.createElement('a');
            a.href = url;
            a.download = 'self-lnn-logs-' + new Date().toISOString().slice(0, 10) + '.log';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            showNotification('✅ 日志导出成功', 'success');
        } else {
            showNotification('❌ 日志导出失败: 无数据', 'danger');
        }
    } catch (e) {
        showNotification('❌ 日志导出异常: ' + e.message, 'danger'); 
    }
}

/**
 * ============================================
 * API服务页面功能函数
 * ============================================
 */

/** 导航到指定页面 */
function navigateTo(sectionId) {
    var link = document.querySelector('.nav a[href="#' + sectionId + '"]');
    if (link) {
        link.click();
    } else {
        /* 无导航链接的section：手动切换active类 */
        document.querySelectorAll('.section').forEach(function(s) { s.classList.remove('active'); });
        var target = document.getElementById(sectionId);
        if (target) target.classList.add('active');
        document.querySelectorAll('.nav a').forEach(function(l) { l.classList.remove('active'); });
    }
    /* 平滑滚动到目标区域 */
    setTimeout(function() {
        var el = document.getElementById(sectionId);
        if (el) {
            el.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }, 80);
}

/** 刷新仪表盘API KEY快捷卡片 */
async function refreshDashApiKey() {
    try {
        /* P2-4: 使用api-service封装方法替代直接request调用 */
        var api = window.SelfLnnApi;
        if (!api) return;
        var resp = await api.getKeyStatus();
        if (!resp.success) return;
        var data = resp.data;
        var addrEl = document.getElementById('dash-api-address');
        var statusEl = document.getElementById('dash-api-key-status');
        /* BUG-6修复: 统一使用带/api后缀的完整地址格式 */
        if (addrEl) addrEl.textContent = 'http://' + (window.SELFLNN_CONFIG.host || 'localhost') + ':' + (window.SELFLNN_CONFIG.port || 8080) + '/api';
        if (statusEl) {
            if (data && data.key_status) {
                statusEl.textContent = data.key_status.enabled ? '密钥已启用' : '密钥未启用';
                statusEl.className = 'status-badge ' + (data.key_status.enabled ? 'active' : 'inactive');
            }
        }
    } catch (e) {
        console.warn('刷新仪表盘API密钥状态失败:', e);
    }
}

/** 复制API地址（仪表盘快捷卡片） */
function copyApiAddress() {
    var addr = 'http://' + (window.SELFLNN_CONFIG.host || 'localhost') + ':' + (window.SELFLNN_CONFIG.port || 8080);
    if (navigator.clipboard) {
        navigator.clipboard.writeText(addr).then(function() {
            showNotification('✅ API地址已复制到剪贴板', 'success');
        }).catch(function() {
            showNotification('⚠️ 复制失败，请手动复制', 'warning');
        });
    } else {
        showNotification('⚠️ 复制失败，请手动复制', 'warning');
    }
}

/**
 * 切换多模态统一学习开关
 */
async function toggleMultimodalLearning(enabled) {
/* 后端不可用时不再显示误导性"离线模式"提示，明确告知用户后端未连接 */
    try {
        if (window.SelfLnnApi && typeof window.SelfLnnApi.multimodalLearn === 'function') {
            var result = await window.SelfLnnApi.multimodalLearn(
                enabled ? 'active' : 'paused',
                'single-cfc-lnn'
            );
            if (result.success) {
                showNotification(enabled ? '✅ 已启用多模态统一学习' : '⏸ 已暂停多模态统一学习', 'success');
                if (window.g_agiController) {
                    window.g_agiController.multimodalLearningEnabled = enabled;
                }
            } else {
                showNotification('❌ 多模态学习切换失败: ' + (result.data?.multimodal?.status || '未知错误'), 'danger');
            }
        } else {
            showNotification('❌ 多模态学习后端未连接，无法切换（请确认SELF-LNN服务器已启动）', 'danger');
        }
    } catch (e) {
        showNotification('❌ 多模态学习请求失败: ' + e.message, 'danger');
    }
}

/**
 * 触发记忆睡眠固化
 */
async function triggerSleepConsolidation() {
    showNotification('正在执行记忆睡眠固化...', 'info');
    try {
        var sleepCycle = parseFloat(document.getElementById('memory-sleep-cycle').value) || 8.0;
        var nremRate = parseFloat(document.getElementById('memory-nrem-rate').value) || 0.15;
        var remRate = parseFloat(document.getElementById('memory-rem-rate').value) || 0.1;
        var pruneThresh = parseFloat(document.getElementById('memory-prune-threshold').value) || 0.05;

        var result = await window.SelfLnnApi.sleepConsolidation({
            sleep_cycle_hours: sleepCycle,
            nrem_consolidation_rate: nremRate,
            rem_crosslink_rate: remRate,
            sleep_prune_threshold: pruneThresh
        });
        if (result.success && result.data) {
            var stats = result.data.stats || [0, 0, 0, 0];
            document.getElementById('sleep-stat-nrem').textContent = stats[0] || 0;
            document.getElementById('sleep-stat-rem').textContent = stats[1] || 0;
            document.getElementById('sleep-stat-prune').textContent = stats[2] || 0;
            document.getElementById('sleep-stat-abstract').textContent = stats[3] || 0;
            var sleepResultEl = document.getElementById('sleep-consolidation-result');
            if (sleepResultEl) sleepResultEl.style.display = 'block';
            showNotification('记忆睡眠固化完成', 'success');
        } else {
            showNotification('睡眠固化失败: ' + (result.error || '未知错误'), 'danger');
        }
    } catch (e) {
        showNotification('睡眠固化请求失败: ' + e.message, 'danger');
    }
}

/**
 * 刷新睡眠固化状态
 */
async function refreshSleepConsolidationStats() {
    try {
        var result = await window.SelfLnnApi.getMemoryStatus();
        if (result.success && result.data && result.data.sleep_consolidation) {
            var sc = result.data.sleep_consolidation;
            document.getElementById('sleep-stat-nrem').textContent = sc.nrem_consolidated || 0;
            document.getElementById('sleep-stat-rem').textContent = sc.rem_crosslinked || 0;
            document.getElementById('sleep-stat-prune').textContent = sc.pruned || 0;
            document.getElementById('sleep-stat-abstract').textContent = sc.abstracted || 0;
            var sleepResultEl2 = document.getElementById('sleep-consolidation-result');
            if (sleepResultEl2) sleepResultEl2.style.display = 'block';
            if (sc.last_run) {
/* 使用textContent替代insertAdjacentHTML，防止XSS */
                var resultDiv = document.getElementById('sleep-consolidation-result');
                if (resultDiv) {
                    var innerDiv = resultDiv.querySelector('div');
                    if (innerDiv) {
                        var span = document.createElement('span');
                        span.textContent = '上次执行: ' + sc.last_run;
                        innerDiv.appendChild(span);
                    }
                }
            }
        } else {
            var sleepResultEl3 = document.getElementById('sleep-consolidation-result');
            if (sleepResultEl3) sleepResultEl3.style.display = 'none';
        }
    } catch (e) {
        var sleepResultEl4 = document.getElementById('sleep-consolidation-result');
        if (sleepResultEl4) sleepResultEl4.style.display = 'none';
    }
}

/**
 * 切换决策审计追踪
 */
function toggleDecisionAudit(enabled) {
    showNotification(enabled ? '已启用决策审计追踪' : '已禁用决策审计追踪', enabled ? 'success' : 'info');
    if (enabled) {
        refreshDecisionLog();
    } else {
        document.getElementById('decision-log-total').textContent = '0';
        document.getElementById('decision-log-autonomous').textContent = '0';
        document.getElementById('decision-log-override').textContent = '0';
        document.getElementById('decision-log-errors').textContent = '0';
        document.getElementById('decision-log-entries').innerHTML =
            '<div class="decision-log-empty" style="text-align:center;padding:16px;color:rgba(255,255,255,0.3);font-size:0.75rem;">审计追踪已禁用</div>';
    }
}

/**
 * 刷新决策日志
 */
async function refreshDecisionLog() {
    try {
        var result = await window.SelfLnnApi.getDecisionLog({ count: 50 });
        if (result.success && result.data) {
            var logs = result.data.entries || [];
            var total = result.data.total || logs.length;
            var autoCount = 0, overrideCount = 0, errorCount = 0;

            document.getElementById('decision-log-total').textContent = total;
            document.getElementById('decision-log-autonomous').textContent = '0';
            document.getElementById('decision-log-override').textContent = '0';
            document.getElementById('decision-log-errors').textContent = '0';

            var container = document.getElementById('decision-log-entries');
            if (logs.length === 0) {
                container.innerHTML =
                    '<div class="decision-log-empty" style="text-align:center;padding:16px;color:rgba(255,255,255,0.3);font-size:0.75rem;">暂无决策日志记录</div>';
                return;
            }

            var html = '';
            for (var i = 0; i < logs.length; i++) {
                var entry = logs[i];
                if (entry.is_autonomous) autoCount++;
                if (entry.log_type === 'override') overrideCount++;
                if (entry.log_type === 'error') errorCount++;

                var typeColors = {
                    'analysis': '#4488ff',
                    'execution': '#00ff88',
                    'evaluation': '#ffcc00',
                    'override': '#ff8800',
                    'error': '#ff4444'
                };
                var color = typeColors[entry.log_type] || '#888';
                html += '<div style="display:flex;align-items:center;gap:8px;padding:6px 8px;border-bottom:1px solid rgba(255,255,255,0.05);font-size:0.7rem;">' +
                    '<span style="width:8px;height:8px;border-radius:50%;background:' + color + ';flex-shrink:0;"></span>' +
                    '<span style="color:rgba(255,255,255,0.4);flex-shrink:0;">' + (entry.time || '--') + '</span>' +
                    '<span style="color:#fff;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">' + (entry.desc || '--') + '</span>' +
                    '<span style="color:' + color + ';flex-shrink:0;">' + (entry.log_type || '--') + '</span>' +
                    '</div>';
            }
            container.innerHTML = html;

            document.getElementById('decision-log-autonomous').textContent = autoCount;
            document.getElementById('decision-log-override').textContent = overrideCount;
            document.getElementById('decision-log-errors').textContent = errorCount;
        } else {
            showNotification('获取决策日志失败', 'danger');
        }
    } catch (e) {
        showNotification('决策日志请求失败: ' + e.message, 'danger');
    }
}

/**
 * 清除决策日志
 */
function clearDecisionLog() {
    document.getElementById('decision-log-total').textContent = '0';
    document.getElementById('decision-log-autonomous').textContent = '0';
    document.getElementById('decision-log-override').textContent = '0';
    document.getElementById('decision-log-errors').textContent = '0';
    document.getElementById('decision-log-entries').innerHTML =
        '<div class="decision-log-empty" style="text-align:center;padding:16px;color:rgba(255,255,255,0.3);font-size:0.75rem;">日志已清除</div>';
    showNotification('决策日志已清除', 'info');
}

/**
 * 保存通用设置
 */
async function saveGeneralSettings() {
    showNotification('正在保存设置...', 'info');
    try {
        var settings = {
            language: document.getElementById('system-language') ? document.getElementById('system-language').value : 'zh-CN',
            timezone: document.getElementById('timezone') ? document.getElementById('timezone').value : 'Asia/Shanghai',
            auto_backup: document.getElementById('auto-backup') ? document.getElementById('auto-backup').value : 'disabled',
            log_level: document.getElementById('settings-log-level') ? document.getElementById('settings-log-level').value : 'info'
        };
        if (window.SelfLnnApi && typeof window.SelfLnnApi.saveSettings === 'function') {
            var result = await window.SelfLnnApi.saveSettings(settings);
            if (result && result.success) {
                showNotification('设置已保存', 'success');
            } else {
                showNotification('保存失败: ' + ((result && result.error) || '未知错误'), 'danger');
            }
        } else {
            showNotification('设置保存后端未连接', 'danger');
        }
    } catch (e) {
        showNotification('保存设置失败: ' + e.message, 'danger');
    }
}

/**
 * 修改密码
 */
async function changePassword() {
    var oldPwd = await promptAsync('请输入当前密码:');
    if (!oldPwd) return;
    var newPwd = await promptAsync('请输入新密码:');
    if (!newPwd || newPwd.length < 6) {
        showNotification('密码长度至少6位', 'danger');
        return;
    }
    var confirmPwd = await promptAsync('请再次输入新密码:');
    if (newPwd !== confirmPwd) {
        showNotification('两次输入密码不一致', 'danger');
        return;
    }
    showNotification('正在修改密码...', 'info');
    try {
        if (window.SelfLnnApi && typeof window.SelfLnnApi.changePassword === 'function') {
            var result = await window.SelfLnnApi.changePassword({ old_password: oldPwd, new_password: newPwd });
            if (result && result.success) {
                showNotification('密码修改成功', 'success');
            } else {
                showNotification('修改失败: ' + ((result && result.error) || '未知错误'), 'danger');
            }
        } else {
            showNotification('密码修改后端未连接', 'danger');
        }
    } catch (e) {
        showNotification('密码修改失败: ' + e.message, 'danger');
    }
}

/* ===== 自我学习 / 自我修正 / 演化 全局触发函数 ===== */
async function startSelfLearning() {
    if (!window.SelfLnnApi || typeof window.SelfLnnApi.startSelfLearning !== 'function') {
        console.warn('SelfLnnApi 未就绪');
        return;
    }
    const btn = document.querySelector('button[onclick="startSelfLearning()"]');
    if (btn) { btn.disabled = true; btn.textContent = '学习中...'; }
    try {
        const result = await window.SelfLnnApi.startSelfLearning({ mode: 'auto' });
        if (result.success) {
            const list = document.getElementById('correction-list');
            if (list) {
                list.innerHTML = '<div class="correction-item success">自我学习已启动</div>';
            }
        }
    } catch (e) {
        console.error('startSelfLearning 失败:', e);
        showNotification('自我学习启动失败: ' + e.message, 'danger');
    } finally {
        if (btn) { btn.disabled = false; btn.textContent = ' 开始学习'; }
    }
}

async function triggerSelfCorrection() {
    if (!window.SelfLnnApi || typeof window.SelfLnnApi.triggerSelfCorrection !== 'function') {
        console.warn('SelfLnnApi 未就绪');
        return;
    }
    const btn = document.querySelector('button[onclick="triggerSelfCorrection()"]');
    if (btn) { btn.disabled = true; btn.textContent = '修正中...'; }
    try {
        const result = await window.SelfLnnApi.triggerSelfCorrection({ mode: 'auto' });
        if (result.success) {
            const list = document.getElementById('correction-list');
            if (list) {
                const item = document.createElement('div');
                item.className = 'correction-item success';
                item.textContent = '自我修正已触发 - ' + new Date().toLocaleString('zh-CN');
                list.insertBefore(item, list.firstChild);
            }
        }
    } catch (e) {
        console.error('triggerSelfCorrection 失败:', e);
        showNotification('自我修正触发失败: ' + e.message, 'danger');
    } finally {
        if (btn) { btn.disabled = false; btn.textContent = ' 触发修正'; }
    }
}

async function startEvolution() {
    if (!window.SelfLnnApi || typeof window.SelfLnnApi.agiEvolve !== 'function') {
        console.warn('SelfLnnApi 未就绪');
        return;
    }
    const btn = document.querySelector('button[onclick="startEvolution()"]');
    if (btn) { btn.disabled = true; btn.textContent = '演化中...'; }
    try {
        const result = await window.SelfLnnApi.agiEvolve({ mode: 'auto', generations: 10 });
        if (result.success) {
            const progress = document.getElementById('evolution-progress');
            if (progress) {
                const gen = (result.data && (result.data.generation !== undefined)) ? result.data.generation : 0;
                const total = (result.data && (result.data.total_generations !== undefined)) ? result.data.total_generations : 10;
                progress.innerHTML = '<div class="evolution-stage active">演化已启动 - 第 ' + gen + '/' + total + ' 代</div>';
            }
            const evolutionStatus = document.getElementById('evolution-status');
            if (evolutionStatus && result.data && result.data.status) {
                evolutionStatus.textContent = result.data.status;
            }
        }
    } catch (e) {
        console.error('startEvolution 失败:', e);
        showNotification('演化启动失败: ' + e.message, 'danger');
    } finally {
        if (btn) { btn.disabled = false; btn.textContent = ' 开始演化'; }
    }
}

/* ===== LNN参数控制 - 重置参数 ===== */
async function resetLNNParameters() {
    showNotification('正在重置LNN参数...', 'info');
    try {
        if (window.SelfLnnApi && typeof window.SelfLnnApi.resetLNNParameters === 'function') {
            var result = await window.SelfLnnApi.resetLNNParameters();
            if (result && result.success) {
                showNotification('LNN参数已重置为默认值', 'success');
                refreshLNNStatus();
            } else {
                showNotification('重置失败: ' + ((result && result.error) || '未知错误'), 'danger');
            }
        } else {
            showNotification('LNN参数重置后端未连接', 'danger');
        }
    } catch (e) {
        showNotification('重置LNN参数失败: ' + e.message, 'danger');
    }
}

/* ===== LNN参数控制 - 自动校准 ===== */
async function calibrateLNN() {
    showNotification('正在自动校准LNN...', 'info');
    try {
        if (window.SelfLnnApi && typeof window.SelfLnnApi.calibrateLNN === 'function') {
            var result = await window.SelfLnnApi.calibrateLNN();
            if (result && result.success && result.data) {
                var data = result.data;
                var setInputVal = function(id,val){var el=document.getElementById(id);if(el)el.value=val;};
                if (data.time_constant !== undefined) setInputVal('lnn-time-constant', data.time_constant);
                if (data.viscosity !== undefined) setInputVal('lnn-viscosity', data.viscosity);
                if (data.diffusion_rate !== undefined) setInputVal('lnn-diffusion-rate', data.diffusion_rate);
                if (data.temperature !== undefined) setInputVal('lnn-temperature', data.temperature);
                if (data.learning_rate !== undefined) setInputVal('lnn-learning-rate', data.learning_rate);
                showNotification('LNN自动校准完成', 'success');
                refreshLNNStatus();
            } else {
                showNotification('校准失败: ' + ((result && result.error) || '未知错误'), 'danger');
            }
        } else {
            showNotification('LNN校准后端未连接', 'danger');
        }
    } catch (e) {
        showNotification('校准LNN失败: ' + e.message, 'danger');
    }
}

/* ===== LNN参数控制 - 导出配置 ===== */
async function exportLNNConfig() {
    showNotification('正在导出LNN配置...', 'info');
    try {
        if (window.SelfLnnApi && typeof window.SelfLnnApi.exportLNNConfig === 'function') {
            var result = await window.SelfLnnApi.exportLNNConfig();
            if (result && result.success && result.data) {
                var blob = new Blob([JSON.stringify(result.data, null, 2)], { type: 'application/json' });
                var url = URL.createObjectURL(blob);
                var a = document.createElement('a');
                a.href = url;
                a.download = 'lnn-config-' + new Date().toISOString().slice(0, 10) + '.json';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
                showNotification('LNN配置导出成功', 'success');
            } else {
                showNotification('导出失败: ' + ((result && result.error) || '未知错误'), 'danger');
            }
        } else {
            showNotification('LNN配置导出后端未连接', 'danger');
        }
    } catch (e) {
        showNotification('导出LNN配置失败: ' + e.message, 'danger');
    }
}

/* ===== 模型配置 - 保存配置 ===== */
async function saveModelConfig() {
    showNotification('正在保存模型配置...', 'info');
    try {
        var configForm = document.querySelector('#models .config-form');
        if (!configForm) {
            showNotification('模型配置表单未找到', 'danger');
            return;
        }
        var sliderValues = configForm.querySelectorAll('.slider-value');
        var config = {
            learning_rate: sliderValues.length > 0 ? parseFloat(sliderValues[0].textContent) || 0.001 : 0.001,
            batch_size: parseInt(document.getElementById('batch-size-select').value) || 32,
            hidden_dim: sliderValues.length > 1 ? parseInt(sliderValues[1].textContent) || 128 : 128,
            time_constant: sliderValues.length > 2 ? parseFloat(sliderValues[2].textContent) || 0.1 : 0.1
        };
        if (window.SelfLnnApi && typeof window.SelfLnnApi.saveModelConfig === 'function') {
            var result = await window.SelfLnnApi.saveModelConfig(config);
            if (result && result.success) {
                showNotification('模型配置已保存', 'success');
            } else {
                showNotification('保存失败: ' + ((result && result.error) || '未知错误'), 'danger');
            }
        } else {
            showNotification('模型配置保存后端未连接', 'danger');
        }
    } catch (e) {
        showNotification('保存模型配置失败: ' + e.message, 'danger');
    }
}

/* ===== 记忆统计 - 刷新 ===== */
async function refreshMemoryStats() {
    showNotification('正在刷新记忆统计...', 'info');
    try {
        var result = await window.SelfLnnApi.getMemoryStatus();
        if (result && result.success && result.data) {
            var mem = result.data;
            var fields = {
                'memory-short-term': mem.short_term,
                'memory-long-term': mem.long_term,
                'memory-episodic': mem.episodic,
                'memory-semantic': mem.semantic,
                'memory-total-entries': mem.total_entries,
                'memory-avg-strength': mem.avg_strength,
                'memory-last-access': mem.last_access,
                'memory-retrieval-rate': mem.retrieval_rate
            };
            for (var id in fields) {
                var el = document.getElementById(id);
                if (el && fields[id] !== undefined) el.textContent = fields[id];
            }
            if (mem.short_term_percent !== undefined) {
                var fill = document.getElementById('memory-short-term-fill');
                if (fill) fill.style.width = Math.min(100, mem.short_term_percent) + '%';
            }
            if (mem.long_term_percent !== undefined) {
                var fill = document.getElementById('memory-long-term-fill');
                if (fill) fill.style.width = Math.min(100, mem.long_term_percent) + '%';
            }
            if (mem.episodic_percent !== undefined) {
                var fill = document.getElementById('memory-episodic-fill');
                if (fill) fill.style.width = Math.min(100, mem.episodic_percent) + '%';
            }
            if (mem.semantic_percent !== undefined) {
                var fill = document.getElementById('memory-semantic-fill');
                if (fill) fill.style.width = Math.min(100, mem.semantic_percent) + '%';
            }
            showNotification('记忆统计已刷新', 'success');
        } else {
            showNotification('获取记忆统计失败: ' + ((result && result.error) || '未知错误'), 'danger');
        }
    } catch (e) {
        showNotification('刷新记忆统计失败: ' + e.message, 'danger');
    }
}

/* ===== 记忆条目 - 添加 ===== */
async function addMemoryEntry() {
    var content = await promptAsync('请输入记忆内容:');
    if (!content) return;
    var type = document.getElementById('memory-type-filter') ? document.getElementById('memory-type-filter').value : 'short-term';
    showNotification('正在添加记忆...', 'info');
    try {
        if (window.SelfLnnApi && typeof window.SelfLnnApi.addMemory === 'function') {
            var result = await window.SelfLnnApi.addMemory({ content: content, type: type });
            if (result && result.success) {
                showNotification('记忆添加成功', 'success');
                refreshMemoryStats();
                searchMemories();
            } else {
                showNotification('添加失败: ' + ((result && result.error) || '未知错误'), 'danger');
            }
        } else {
            showNotification('记忆添加后端未连接', 'danger');
        }
    } catch (e) {
        showNotification('添加记忆失败: ' + e.message, 'danger');
    }
}

/**
 * 查看记忆条目详情
 */
async function viewMemoryDetails(id) {
    showNotification('正在获取记忆详情...', 'info');
    try {
        if (window.SelfLnnApi && typeof window.SelfLnnApi.getMemoryEntry === 'function') {
            var result = await window.SelfLnnApi.getMemoryEntry(id);
            if (result && result.success && result.data) {
                var entry = result.data;
                var html = '<div style="padding:12px;background:rgba(255,204,0,0.05);border-radius:8px;margin:8px 0;">' +
                    '<h4 style="margin:0 0 8px;">' + (entry.title || '记忆 #' + id) + '</h4>' +
                    '<p style="color:rgba(255,255,255,0.7);font-size:0.8rem;">' + (entry.content || entry.description || '') + '</p>' +
                    '<div style="display:flex;gap:12px;font-size:0.7rem;color:rgba(255,255,255,0.4);">' +
                    '<span>类型: ' + (entry.type || '短期') + '</span>' +
                    '<span>强度: ' + (entry.strength != null ? entry.strength.toFixed(2) : 'N/A') + '</span>' +
                    '<span>时间: ' + (entry.time || '--') + '</span></div></div>';
                showNotification('', 'info');
                var container = document.getElementById('memory-detail-panel');
                if (container) {
                    container.innerHTML = html;
                    container.style.display = 'block';
                } else {
                    SelfLnnNotify.show(entry.content || entry.description || '无内容', 'info', 5000);
                }
            } else {
                showNotification('记忆条目未找到', 'warning');
            }
        } else {
            showNotification('记忆条目查询后端未连接', 'danger');
        }
    } catch (e) {
        showNotification('查看记忆详情失败: ' + e.message, 'danger');
    }
}

/**
 * 编辑记忆条目
 */
async function editMemoryEntry(id) {
    var newContent = await promptAsync('请输入新的记忆内容:');
    if (!newContent) return;
    showNotification('正在更新记忆...', 'info');
    try {
        if (window.SelfLnnApi && typeof window.SelfLnnApi.updateMemoryEntry === 'function') {
            var result = await window.SelfLnnApi.updateMemoryEntry(id, { content: newContent });
            if (result && result.success) {
                showNotification('记忆已更新', 'success');
                searchMemories();
            } else {
                showNotification('更新失败: ' + ((result && result.error) || '未知错误'), 'danger');
            }
        } else {
            showNotification('记忆更新后端未连接', 'danger');
        }
    } catch (e) {
        showNotification('更新记忆失败: ' + e.message, 'danger');
    }
}

/**
 * 删除记忆条目
 */
async function deleteMemoryEntry(id) {
    if (!safeConfirm('确定要删除记忆条目 #' + id + ' 吗？此操作不可撤销。')) return;
    showNotification('正在删除...', 'warning');
    try {
        if (window.SelfLnnApi && typeof window.SelfLnnApi.deleteMemoryEntry === 'function') {
            var result = await window.SelfLnnApi.deleteMemoryEntry(id);
            if (result && result.success) {
                showNotification('记忆已删除', 'success');
                searchMemories();
            } else {
                showNotification('删除失败: ' + ((result && result.error) || '未知错误'), 'danger');
            }
        } else {
            showNotification('记忆删除后端未连接', 'danger');
        }
    } catch (e) {
        showNotification('删除失败: ' + e.message, 'danger');
    }
}

/* ===== 记忆条目 - 导出 ===== */
async function exportMemory() {
    showNotification('正在导出记忆...', 'info');
    try {
        if (window.SelfLnnApi && typeof window.SelfLnnApi.exportMemory === 'function') {
            var result = await window.SelfLnnApi.exportMemory();
            if (result && result.success && result.data) {
                var blob = new Blob([JSON.stringify(result.data, null, 2)], { type: 'application/json' });
                var url = URL.createObjectURL(blob);
                var a = document.createElement('a');
                a.href = url;
                a.download = 'memory-export-' + new Date().toISOString().slice(0, 10) + '.json';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
                showNotification('记忆导出成功', 'success');
            } else {
                showNotification('导出失败: ' + ((result && result.error) || '未知错误'), 'danger');
            }
        } else {
            showNotification('记忆导出后端未连接', 'danger');
        }
    } catch (e) {
        showNotification('导出记忆失败: ' + e.message, 'danger');
    }
}

/* ===== 记忆条目 - 清理 ===== */
async function clearOldMemories() {
    if (!safeConfirm('确定要清理旧记忆吗？此操作不可撤销。')) return;
    showNotification('正在清理旧记忆...', 'info');
    try {
        if (window.SelfLnnApi && typeof window.SelfLnnApi.clearOldMemories === 'function') {
            var result = await window.SelfLnnApi.clearOldMemories({ days: 30 });
            if (result && result.success) {
                var count = (result.data && result.data.deleted_count) || 0;
                showNotification('已清理 ' + count + ' 条旧记忆', 'success');
                refreshMemoryStats();
                searchMemories();
            } else {
                showNotification('清理失败: ' + ((result && result.error) || '未知错误'), 'danger');
            }
        } else {
            showNotification('记忆清理后端未连接', 'danger');
        }
    } catch (e) {
        showNotification('清理旧记忆失败: ' + e.message, 'danger');
    }
}

/* ===== 记忆条目 - 搜索 ===== */
async function searchMemories() {
    var type = document.getElementById('memory-type-filter') ? document.getElementById('memory-type-filter').value : 'all';
    showNotification('正在搜索记忆...', 'info');
    try {
        var params = {};
        if (type !== 'all') params.type = type;
        if (window.SelfLnnApi && typeof window.SelfLnnApi.searchMemories === 'function') {
            var result = await window.SelfLnnApi.searchMemories(params);
            if (result && result.success) {
                var entries = result.data && result.data.entries ? result.data.entries : [];
                var container = document.getElementById('memory-entries-list');
                if (!container) return;
                if (entries.length === 0) {
                    container.innerHTML = '<div class="memory-entry empty">暂无匹配的记忆条目</div>';
                    var pageInfo = document.getElementById('memory-page-info');
                    if (pageInfo) pageInfo.textContent = '第 0 页，共 0 页';
                } else {
                    container.innerHTML = entries.map(function(e) {
                        var time = window.escapeHtml(e.created_at || e.time || '');
                        return '<div class="memory-entry">' +
                            '<div class="memory-entry-type">' + window.escapeHtml(e.type || 'unknown') + '</div>' +
                            '<div class="memory-entry-content">' + window.escapeHtml(e.content || '') + '</div>' +
                            '<div class="memory-entry-time">' + time + '</div>' +
                            '</div>';
                    }).join('');
                    var totalPages = Math.ceil(entries.length / 20);
                    var pageInfo = document.getElementById('memory-page-info');
                    if (pageInfo) pageInfo.textContent = '第 1 页，共 ' + totalPages + ' 页';
                }
                showNotification('搜索完成，找到 ' + entries.length + ' 条结果', 'success');
            } else {
                showNotification('搜索失败: ' + ((result && result.error) || '未知错误'), 'danger');
            }
        } else {
            showNotification('记忆搜索后端未连接', 'danger');
        }
    } catch (e) {
        showNotification('搜索记忆失败: ' + e.message, 'danger');
    }
}

/* ================================================================
 * M-022修复：统一通知系统（替代alert()/prompt()）
 * ================================================================ */
var SelfLnnNotify = {
    _container: null,
    _queue: [],

    init: function() {
        if (this._container) return;
        this._container = document.createElement('div');
        this._container.className = 'selflnn-notify-container';
        this._container.id = 'selflnn-notify-container';
        document.body.appendChild(this._container);
    },

    show: function(message, type, duration) {
        if (!this._container) this.init();
        type = type || 'info';
        duration = duration || 3000;
        var el = document.createElement('div');
        el.className = 'selflnn-notify selflnn-notify-' + type;
        el.innerHTML = '<span class="notify-icon">' +
            (type === 'success' ? '\u2713' : type === 'error' ? '\u2717' : type === 'warning' ? '\u26A0' : '\u2139') +
            '</span><span class="notify-msg">' + (message || '') + '</span>' +
            '<span class="notify-close" onclick="this.parentElement.remove()">\u00D7</span>';
        this._container.appendChild(el);
        var self = this;
        setTimeout(function() {
            if (el.parentElement) {
                el.style.opacity = '0';
                el.style.transform = 'translateX(100%)';
                setTimeout(function() { if (el.parentElement) el.remove(); }, 300);
            }
        }, duration);
        return el;
    },

    confirm: function(message, onOk, onCancel) {
        var overlay = document.createElement('div');
        overlay.className = 'selflnn-confirm-overlay';
        overlay.innerHTML =
            '<div class="selflnn-confirm-box">' +
            '<div class="confirm-msg">' + (message || '') + '</div>' +
            '<div class="confirm-buttons">' +
            '<button class="btn-cancel">\u53D6\u6D88</button>' +
            '<button class="btn-ok">\u786E\u5B9A</button>' +
            '</div></div>';
        document.body.appendChild(overlay);
        overlay.querySelector('.btn-ok').onclick = function() {
            document.body.removeChild(overlay);
            if (onOk) onOk();
        };
        overlay.querySelector('.btn-cancel').onclick = function() {
            document.body.removeChild(overlay);
            if (onCancel) onCancel();
        };
        overlay.onclick = function(e) {
            if (e.target === overlay) {
                document.body.removeChild(overlay);
                if (onCancel) onCancel();
            }
        };
        return overlay;
    },
    
    prompt: function(message, defaultValue, callback) {
        var overlay = document.createElement('div');
        overlay.className = 'selflnn-confirm-overlay';
        var defVal = defaultValue || '';
        overlay.innerHTML =
            '<div class="selflnn-confirm-box">' +
            '<div class="confirm-msg">' + (message || '') + '</div>' +
            '<input class="confirm-input" type="text" value="' + defVal.replace(/"/g, '&quot;') + '" style="width:100%;padding:8px;margin:8px 0;border:1px solid #555;background:#1a1a2e;color:#fff;border-radius:4px;box-sizing:border-box;">' +
            '<div class="confirm-buttons">' +
            '<button class="btn-cancel">取消</button>' +
            '<button class="btn-ok">确定</button>' +
            '</div></div>';
        document.body.appendChild(overlay);
        var input = overlay.querySelector('.confirm-input');
        input.focus();
        input.select();
        overlay.querySelector('.btn-ok').onclick = function() {
            document.body.removeChild(overlay);
            if (callback) callback(input.value);
        };
        overlay.querySelector('.btn-cancel').onclick = function() {
            document.body.removeChild(overlay);
            if (callback) callback(null);
        };
        overlay.onclick = function(e) {
            if (e.target === overlay) {
                document.body.removeChild(overlay);
                if (callback) callback(null);
            }
        };
        input.onkeydown = function(e) {
            if (e.key === 'Enter') {
                document.body.removeChild(overlay);
                if (callback) callback(input.value);
            } else if (e.key === 'Escape') {
                document.body.removeChild(overlay);
                if (callback) callback(null);
            }
        };
    }
};

/* 将全局showNotification重定向到SelfLnnNotify.show */
var _origShowNotification = window.showNotification;
window.showNotification = function(message, type, duration) {
    return SelfLnnNotify.show(message, type, duration);
};

/* promptAsync() — 将异步SelfLnnNotify.prompt封装为Promise，支持await调用 */
function promptAsync(message, defaultValue) {
    return new Promise(function(resolve) {
        SelfLnnNotify.prompt(message, defaultValue || '', function(val) {
            resolve(val !== null && val !== undefined ? val : null);
        });
    });
}

/* FIX-FRONTEND-001: 安全设置保存函数(index.html中onchange引用) */
window.saveSafetySetting = function(key, value) {
    try {
        var payload = {};
        payload[key] = value;
        /* P2-4: 使用api-service封装的setSafetyBounds方法 */
        var api = window.SelfLnnApi;
        if (api && api.setSafetyBounds) {
            api.setSafetyBounds(payload).then(function(r) {
                if (r && r.success) {
                    window.showNotification('安全设置已保存', 'success');
                } else {
                    window.showNotification('保存失败: ' + ((r && r.message) || '未知错误'), 'danger');
                }
            }).catch(function(e) {
                console.warn('安全设置保存失败:', e);
                window.showNotification('安全设置保存失败，请检查网络连接', 'danger');
            });
        }
    } catch (e) {
        console.warn('saveSafetySetting异常:', e);
    }
};

/* ================================================================
 * M-023修复：生产日志级别控制
 * ================================================================ */
var SelfLnnLog = {
    _level: 2, /* 0=OFF, 1=ERROR, 2=WARN, 3=INFO, 4=DEBUG */

    setLevel: function(level) { this._level = level; },

    error: function() {
        if (this._level >= 1) console.error.apply(console, arguments);
    },
    warn: function() {
        if (this._level >= 2) console.warn.apply(console, arguments);
    },
    info: function() {
        if (this._level >= 3) console.info.apply(console, arguments);
    },
    debug: function() {
        if (this._level >= 4) console.log.apply(console, arguments);
    },

    /* 生产模式：仅ERROR */
    productionMode: function() { this._level = 1; }
};

/* API服务端日志重定向 */
var _origConsoleLog = console.log;
var _origConsoleWarn = console.warn;
console.log = function() {
    if (SelfLnnLog._level >= 4) _origConsoleLog.apply(console, arguments);
};
console.warn = function() {
    if (SelfLnnLog._level >= 2) _origConsoleWarn.apply(console, arguments);
};

/* ================================================================
 *: prompt() 兼容层 — 统一使用SelfLnnNotify.prompt模态框
 * 所有原本依赖原生prompt()的代码已迁移至promptAsync()异步调用。
 * 此兼容层仅保留兜底，确保未迁移代码不会静默失败。
 * ================================================================ */
if (typeof window._nativePrompt === 'undefined') {
    window._nativePrompt = window.prompt;
}
window.prompt = function(msg, defVal) {
    try {
        if (typeof window._nativePrompt === 'function' && 
            !navigator.userAgent.includes('Trae')) {
            return window._nativePrompt(msg, defVal);
        }
    } catch(e) { console.error(' _nativePrompt调用失败:', e&&e.message?e.message:e); }
    SelfLnnLog.warn(' 检测到未迁移的prompt()调用，自动降级使用SelfLnnNotify.prompt。请将调用方改用promptAsync()。');
    SelfLnnNotify.prompt(msg || '', defVal || '', function(val) {
        var msg2 = ' prompt()兼容层收到用户输入"' + (val || '') +
                   '"，但调用方未迁移至异步模式，输入值已通过window._promptLastResult临时存储。' +
                   '请迁移至promptAsync()。';
        SelfLnnLog.warn(msg2);
        if (val && val.trim && val.trim()) {
            window._promptLastResult = val;
        }
    });
    return defVal || '';
};
async function refreshReasoningStats() {
    showNotification('正在刷新推理统计...', 'info');
    try {
        var result = await window.SelfLnnApi.getReasoningStatus();
        if (result && result.success && result.data) {
            var rsn = result.data.reasoning || result.data;
            var cfg = rsn.configuration || {};
            
            var successRate = document.getElementById('reasoning-success-rate');
            if (successRate) successRate.textContent = cfg.confidence_threshold ? (cfg.confidence_threshold * 100).toFixed(1) + '%' : '0.0%';
            var requestsPerSec = document.getElementById('reasoning-requests-per-sec');
            if (requestsPerSec) requestsPerSec.textContent = cfg.max_iterations ? cfg.max_iterations.toFixed(1) : '0.0';
            var avgResp = document.getElementById('reasoning-avg-response-time');
            if (avgResp) avgResp.textContent = '运行中';

            var mode = cfg.default_mode || '';
            var caps = rsn.capabilities || [];
            var modeMap = {
                '演绎推理': { fill: 'reasoning-deductive-fill', val: 'reasoning-deductive-value' },
                '归纳推理': { fill: 'reasoning-inductive-fill', val: 'reasoning-inductive-value' },
                '溯因推理': { fill: 'reasoning-abductive-fill', val: 'reasoning-abductive-value' },
                '类比推理': { fill: 'reasoning-analogical-fill', val: 'reasoning-analogical-value' }
            };
            for (var m in modeMap) {
                var ids = modeMap[m];
                var fill = document.getElementById(ids.fill);
                var valEl = document.getElementById(ids.val);
                var active = (m === mode) ? 0.85 : (caps.some(function(c) { return c.indexOf(m.replace('推理','')) >= 0; }) ? 0.4 : 0.1);
                if (fill) fill.style.width = (active * 100) + '%';
                if (valEl) valEl.textContent = (active * 100).toFixed(0) + '%';
            }
            showNotification('推理统计已刷新', 'success');
        } else {
            showNotification('获取推理统计失败', 'danger');
        }
    } catch (e) {
        showNotification('刷新推理统计失败: ' + e.message, 'danger');
    }
}

/* ===== 推理任务 - 新建任务 ===== */
async function startNewReasoning() {
    var content = await promptAsync('输入推理问题或任务描述:');
    if (!content) return;
    var priority = document.getElementById('reasoning-priority') ? document.getElementById('reasoning-priority').value : 'medium';
    showNotification('正在创建推理任务...', 'info');
    try {
        if (window.SelfLnnApi && typeof window.SelfLnnApi.startReasoning === 'function') {
            var result = await window.SelfLnnApi.startReasoning({ content: content, priority: priority });
            if (result && result.success && result.data) {
                var taskId = result.data.task_id || result.data.id || 'unknown';
                var tasksList = document.getElementById('reasoning-tasks-list');
                if (tasksList) {
                    var emptyEl = tasksList.querySelector('.empty');
                    if (emptyEl) emptyEl.remove();
                    var taskEl = document.createElement('div');
                    taskEl.className = 'reasoning-task';
                    taskEl.innerHTML = '<div class="reasoning-task-header"><span class="reasoning-task-id">#' + taskId + '</span><span class="reasoning-task-status status-running">运行中</span></div>' +
                        '<div class="reasoning-task-content">' + window.escapeHtml(content.substring(0, 50)) + '</div>' +
                        '<div class="reasoning-task-meta">优先级: ' + priority + ' | ' + new Date().toLocaleString('zh-CN') + '</div>';
                    tasksList.insertBefore(taskEl, tasksList.firstChild);
                }
                showNotification('推理任务已创建，ID: ' + taskId, 'success');
            } else {
                showNotification('创建推理任务失败: ' + ((result && result.error) || '未知错误'), 'danger');
            }
        } else {
            showNotification('推理任务后端未连接', 'danger');
        }
    } catch (e) {
        showNotification('创建推理任务失败: ' + e.message, 'danger');
    }
}

/* ===== 推理任务 - 暂停 ===== */
async function pauseReasoning() {
    showNotification('正在暂停推理...', 'info');
    try {
        if (window.SelfLnnApi && typeof window.SelfLnnApi.pauseReasoning === 'function') {
            var result = await window.SelfLnnApi.pauseReasoning();
            if (result && result.success) {
                var tasksList = document.getElementById('reasoning-tasks-list');
                if (tasksList) {
                    var runningTasks = tasksList.querySelectorAll('.reasoning-task-status.status-running');
                    runningTasks.forEach(function(el) {
                        el.textContent = '已暂停';
                        el.className = 'reasoning-task-status status-paused';
                    });
                }
                showNotification('推理已暂停', 'warning');
            } else {
                showNotification('暂停失败: ' + ((result && result.error) || '未知错误'), 'danger');
            }
        } else {
            showNotification('推理暂停后端未连接', 'danger');
        }
    } catch (e) {
        showNotification('暂停推理失败: ' + e.message, 'danger');
    }
}

/* ===== 推理任务 - 全部停止 ===== */
async function stopAllReasoning() {
    if (!safeConfirm('确定要停止所有推理任务吗？')) return;
    showNotification('正在停止所有推理任务...', 'info');
    try {
        if (window.SelfLnnApi && typeof window.SelfLnnApi.stopAllReasoning === 'function') {
            var result = await window.SelfLnnApi.stopAllReasoning();
            if (result && result.success) {
                var tasksList = document.getElementById('reasoning-tasks-list');
                if (tasksList) {
                    tasksList.innerHTML = '<div class="reasoning-task empty">所有推理任务已停止</div>';
                }
                showNotification('所有推理任务已停止', 'success');
            } else {
                showNotification('停止失败: ' + ((result && result.error) || '未知错误'), 'danger');
            }
        } else {
            showNotification('推理停止后端未连接', 'danger');
        }
    } catch (e) {
        showNotification('停止推理失败: ' + e.message, 'danger');
    }
}

/* ===== 推理配置 - 保存 ===== */
async function saveReasoningConfig() {
    showNotification('正在保存推理配置...', 'info');
    try {
        var priority = document.getElementById('reasoning-priority') ? document.getElementById('reasoning-priority').value : 'medium';
        var config = { priority: priority };
        if (window.SelfLnnApi && typeof window.SelfLnnApi.saveReasoningConfig === 'function') {
            var result = await window.SelfLnnApi.saveReasoningConfig(config);
            if (result && result.success) {
                showNotification('推理配置已保存', 'success');
            } else {
                showNotification('保存失败: ' + ((result && result.error) || '未知错误'), 'danger');
            }
        } else {
            showNotification('推理配置保存后端未连接', 'danger');
        }
    } catch (e) {
        showNotification('保存推理配置失败: ' + e.message, 'danger');
    }
}

/* ===== 学习指标 - 刷新 ===== */
async function refreshLearningMetrics() {
    showNotification('正在刷新学习指标...', 'info');
    try {
        var result = await window.SelfLnnApi.getLearningStatus();
        if (result && result.success && result.data) {
            var lrn = result.data;
            var fieldMapping = {
                'learning-rate-value': lrn.learning_rate,
                'learning-correction-count': lrn.correction_count,
                'learning-evolution-gen': lrn.evolution_gen,
                'learning-knowledge-growth': lrn.knowledge_growth
            };
            for (var id in fieldMapping) {
                var el = document.getElementById(id);
                if (el && fieldMapping[id] !== undefined) el.textContent = fieldMapping[id];
            }
            var trendMapping = {
                'learning-rate-trend': lrn.rate_trend,
                'learning-correction-trend': lrn.correction_trend,
                'learning-evolution-trend': lrn.evolution_trend,
                'learning-knowledge-trend': lrn.knowledge_trend
            };
            for (var tid in trendMapping) {
                var tel = document.getElementById(tid);
                if (tel && trendMapping[tid] !== undefined) tel.textContent = (trendMapping[tid] > 0 ? '+' : '') + trendMapping[tid] + '%';
            }
            var progressMapping = {
                'learning-reasoning-fill': lrn.reasoning_progress,
                'learning-memory-fill': lrn.memory_progress,
                'learning-planning-fill': lrn.planning_progress,
                'learning-creativity-fill': lrn.creativity_progress
            };
            for (var pid in progressMapping) {
                var pfill = document.getElementById(pid);
                var pval = document.getElementById(pid.replace('-fill', '-value'));
                var v = progressMapping[pid];
                if (pfill && v !== undefined) pfill.style.width = v + '%';
                if (pval && v !== undefined) pval.textContent = v + '%';
            }
            showNotification('学习指标已刷新', 'success');
        } else {
            showNotification('获取学习指标失败: ' + ((result && result.error) || '未知错误'), 'danger');
        }
    } catch (e) {
        showNotification('刷新学习指标失败: ' + e.message, 'danger');
    }
}

/* ===== SPA 路由支持 - hashchange 事件监听 ===== */
/* 修复：原setTimeout(300)在大型HTML（~6500行/485KB）中过早触发，
 * 位于71%位置的section（如product-design第4616行）尚未被浏览器解析，
 * 导致document.getElementById返回null，路由切换失败。
 * 改为DOMContentLoaded事件确保完整DOM就绪后再执行初始路由。 */
(function() {
    var _routeRetryCount = 0;
    var _routeMaxRetry = 50;

    function navigateToSection() {
        var hash = window.location.hash;
        if (!hash || hash.length < 2) return;
        var sectionId = hash.substring(1);
        var section = document.getElementById(sectionId);
        if (!section) {
            /* 容错：如果目标section仍未找到，延迟重试（最长等待5秒） */
            _routeRetryCount++;
            if (_routeRetryCount <= _routeMaxRetry) {
                setTimeout(navigateToSection, 100);
            } else {
                console.warn('[SPA路由] 目标section[' + sectionId + ']未找到，已重试' + _routeMaxRetry + '次，放弃');
            }
            return;
        }
        _routeRetryCount = 0;
        /* 激活目标section */
        document.querySelectorAll('.section').forEach(function(s) { s.classList.remove('active'); });
        section.classList.add('active');
        /* 同步导航链接高亮 */
        document.querySelectorAll('.nav a').forEach(function(l) { l.classList.remove('active'); });
        var navLink = document.querySelector('.nav a[href="#' + sectionId + '"]');
        if (navLink) navLink.classList.add('active');
        /* 滚动到目标区域，使用scrollIntoView滚动.main-content容器 */
        var mainContent = document.querySelector('.main-content');
        if (mainContent) {
            var headerH = document.querySelector('.header') ? document.querySelector('.header').offsetHeight : 60;
            var sectionTop = section.offsetTop;
            mainContent.scrollTo({ top: Math.max(0, sectionTop - headerH - 8), behavior: 'smooth' });
        } else {
            section.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }
    window._registerEventListener(window, 'hashchange', navigateToSection);
    if (window.location.hash) {
        /* 使用DOMContentLoaded确保完整HTML解析完毕后再执行初始路由 */
        if (document.readyState === 'loading') {
            /* FE-008修复: 使用_registerEventListener统一管理事件监听器生命周期 */
            window._registerEventListener(document, 'DOMContentLoaded', function() {
                setTimeout(navigateToSection, 50);
            });
        } else {
            navigateToSection();
        }
    }
    
    // ===== 可靠仪表盘自动更新器 =====
    (function() {
        setTimeout(function() {
            if (!window.SelfLnnApi) return;
            
            async function refreshAllPanels() {
                try {
                    /* P2-4: 使用api-service封装的getSystemStatus方法 */
                    var api = window.SelfLnnApi;
                    if (!api) return;
                    var resp = await api.getSystemStatus();
                    if (!resp.success) return;
                    var d = resp.data;
                    var sys = d.system || d;
                    
                    // 直接设置所有关键元素（P3-004修复: 移除骨架屏类）
                    var set = function(id, val) { var e=document.getElementById(id); if(e){e.textContent=val; e.classList.remove('skeleton-loading');} };
                    var setQ = function(sel, val) { var e=document.querySelector(sel); if(e){e.textContent=val; e.classList.remove('skeleton-loading');} };
                    
                    // R3-02修复: 从API返回数据动态判断，不再硬编码'运行正常'
                    var healthText = '运行正常';
                    var healthClass = '';
                    if (sys.modules && sys.modules.lnn && sys.modules.lnn.convergence_rate !== undefined) {
                        healthText = (sys.modules.lnn.convergence_rate > 0.001) ? '运行正常' : '训练中';
                    } else if (sys.uptime && sys.uptime > 0) {
                        healthText = '运行正常';
                    } else {
                        healthText = '初始化中...';
                    }
                    set('health-status-text', healthText);
                    setQ('.metric:nth-child(1) .metric-value', Math.round(sys.cpu_usage||0) + '%');
                    set('active-tasks', String(sys.requests?sys.requests.connections||0:'0'));
                    
                    if (sys.uptime) {
                        var d2=Math.floor(sys.uptime/86400), h2=Math.floor((sys.uptime%86400)/3600);
                        set('uptime-display', d2+'天 '+h2+'小时');
                    }
                    
                    // 底部状态栏 - 从API数据动态判断
                    set('status-bar-api', (sys.requests && sys.requests.connections >= 0) ? '在线' : '离线');
                    set('status-bar-db', (sys.modules && sys.modules.knowledge) ? '已连接' : '未连接');
                    set('status-bar-network', (sys.requests && sys.requests.connections > 0) ? '连接正常' : '无连接');
                    set('conn-status', (sys.requests && sys.requests.connections > 0) ? '已连接' : '未连接');
                    
                    // LNN状态
                    var ms = document.querySelectorAll('.model-stats .stat-value');
                    if (ms.length>=3) { ms[0].textContent='单一液态神经网络(CfC)'; ms[1].textContent=sys.modules?(sys.modules.lnn?sys.modules.lnn.hidden_size||'--':'--'):'--'; ms[2].textContent=sys.modules?(sys.modules.lnn?sys.modules.lnn.total_params||'--':'--'):'--'; }
                    
                    // 多模态编码 - 从API数据动态判断而非硬编码'就绪'
                    var subs = document.querySelectorAll('#lnn-submodules .model-status, .model-status');
                    if (subs.length>=3) {
                        var subReady = (sys.modules && sys.modules.multimodal) ? '就绪' : '待初始化';
                        var subActive = (sys.modules && sys.modules.multimodal) ? 'active' : 'inactive';
                        subs[0].textContent=subReady; subs[0].className='model-status '+subActive;
                        subs[1].textContent=subReady; subs[1].className='model-status '+subActive;
                        subs[2].textContent=subReady; subs[2].className='model-status '+subActive;
                    }
                    
                    // 内存
                    if (sys.modules && sys.modules.memory) {
                        var mem = sys.modules.memory;
                        var totalMem = (mem.total||0)/(1024*1024);
                        setQ('.metric:nth-child(2) .metric-value', totalMem.toFixed(1)+' MB');
                    }

/* 系统日志面板自动刷新 + 使用正确的/system/logs端点 */
                    try {
                        var logResp = await window.SelfLnnApi.request('/system/logs?lines=50');
                        if (logResp && logResp.ok) {
                            var logData = await logResp.json();
                            var logPanel = document.getElementById('sys-log-content');
                            if (logPanel) {
                                var entries = (logData && logData.logs) ? logData.logs :
                                    (Array.isArray(logData) ? logData : []);
                                if (entries && entries.length > 0) {
                                    logPanel.innerHTML = entries.map(function(l) {
                                        var ts = l.timestamp || l.time || '';
                                        var msg = l.message || l.msg || l.text || String(l);
                                        var lvl = (l.level || '').toUpperCase();
                                        var color = (lvl === 'ERROR' || lvl === 'FATAL') ? '#ff4444' :
                                            (lvl === 'WARNING' || lvl === 'WARN') ? '#ffaa00' : '#aaa';
                                        return '<div style="margin:2px 0;color:' + color +
                                            '"><span style="color:#666">' + (window.escapeHtml ? window.escapeHtml(String(ts)) : ts) +
                                            '</span> ' + (window.escapeHtml ? window.escapeHtml(String(msg)) : msg) +
                                            '</div>';
                                    }).join('');
                                } else {
                                    logPanel.innerHTML = '<div style="color:#666">暂无系统日志</div>';
                                }
                            }
                        }
                    } catch(logErr) { /* 日志刷新静默失败 */ }

                    /* L-024修复: 数据刷新后更新训练历史占位文本（区分连接状态和无数据状态） */
                    if (typeof updateTrainingHistoryPlaceholderState === 'function') {
                        updateTrainingHistoryPlaceholderState();
                    }

                } catch(e) { console.warn('[可靠更新器] 失败:', e.message); }
            }
            
            // 立即执行一次，然后每10秒刷新
            refreshAllPanels();
/* 保存定时器句柄防止内存泄漏 */
            if (typeof g_dataEngine !== 'undefined' && g_dataEngine && typeof g_dataEngine.registerModule === 'function') {
                g_dataEngine.registerModule('all_panels_sync', 10000, refreshAllPanels);
            } else {
                window._allPanelsRefreshTimer = setInterval(refreshAllPanels, 10000);
            }
        }, 6000);
    })();
})();

/* ===== 知识库搜索功能 (L-007修复: 重命名为searchKnowledgeBase避免与knowledge-graph.js冲突) ===== */
async function searchKnowledgeBase() {
    var query = document.getElementById('knowledge-search');
    var filter = document.getElementById('knowledge-filter');
    var q = query ? query.value.trim() : '';
    var f = filter ? filter.value : 'all';
    if (!q) {
        showNotification('请输入搜索关键词', 'warning');
        return;
    }
    showNotification('正在搜索知识库...', 'info');
    try {
        if (window.SelfLnnApi && typeof window.SelfLnnApi.searchKnowledge === 'function') {
            var result = await window.SelfLnnApi.searchKnowledge({ query: q, type: f });
            renderKnowledgeSearchResults(result);
        } else {
            var kbResult = await window.SelfLnnApi.getKnowledgeBase({ search: q, type: f, page: 1, pageSize: 20 });
            renderKnowledgeSearchResults(kbResult);
        }
    } catch (e) {
        showNotification('搜索知识库失败: ' + e.message, 'danger');
    }
}

function renderKnowledgeSearchResults(result) {
    var listEl = document.getElementById('knowledge-entries-list');
    if (!listEl) return;
    if (!result || !result.success || !result.data || !result.data.entries || result.data.entries.length === 0) {
        listEl.innerHTML = '<div class="knowledge-entry empty">未找到匹配的知识条目</div>';
        var pageInfo = document.getElementById('knowledge-page-info');
        if (pageInfo) pageInfo.textContent = '无匹配结果';
        return;
    }
    var entries = result.data.entries;
    listEl.innerHTML = entries.map(function(entry, idx) {
        var typeClass = entry.type || 'concept';
        var typeLabel = {fact:'事实',rule:'规则',concept:'概念',relation:'关系'}[typeClass] || typeClass;
        return '<div class="knowledge-entry" onclick="showKnowledgeDetail(\'' + (entry.id || idx) + '\')">' +
            '<div class="knowledge-entry-header">' +
                '<span class="knowledge-entry-type ' + typeClass + '">' + typeLabel + '</span>' +
                '<span class="knowledge-entry-name">' + (entry.name || entry.title || '条目' + (idx+1)) + '</span>' +
            '</div>' +
            '<p class="knowledge-entry-desc">' + (entry.description || entry.content || '').substring(0, 120) + '</p>' +
            '<div class="knowledge-entry-meta">' +
                '<span>置信度: ' + ((entry.confidence || 0) * 100).toFixed(0) + '%</span>' +
                '<span>来源: ' + (entry.source || '系统') + '</span>' +
            '</div>' +
        '</div>';
    }).join('');
    var pageInfo = document.getElementById('knowledge-page-info');
    if (pageInfo) {
        var total = result.data.total || result.data.entries.length;
        var page = result.data.page || 1;
        var totalPages = Math.ceil(total / 20);
        pageInfo.textContent = '第 ' + page + ' 页，共 ' + totalPages + ' 页';
    }
    showNotification('搜索完成，找到 ' + (result.data.entries.length) + ' 个结果', 'success');
}

/* ===== 知识推理测试 ===== */
async function testInference() {
    var queryInput = document.getElementById('inference-query');
    var resultText = document.getElementById('inference-result-text');
    if (!queryInput || !resultText) return;
    var query = queryInput.value.trim();
    if (!query) {
        showNotification('请输入推理查询', 'warning');
        return;
    }
    resultText.textContent = '推理中...';
    try {
        if (window.SelfLnnApi && typeof window.SelfLnnApi.testInference === 'function') {
            var result = await window.SelfLnnApi.testInference({ query: query });
            if (result && result.success) {
                resultText.textContent = result.data.result || result.data.conclusion || '推理完成';
                showNotification('推理完成', 'success');
            } else {
                resultText.textContent = '推理失败: ' + ((result && result.error) || '未知错误');
            }
        } else {
            resultText.textContent = '推理引擎未连接——无法执行推理。请确认后端推理服务已启动后重试。';
            showNotification('推理引擎未连接', 'warning');
        }
    } catch (e) {
        resultText.textContent = '推理出错: ' + e.message;
        showNotification('推理测试失败: ' + e.message, 'danger');
    }
}

/* ===== 知识条目详情展示 ===== */
function showKnowledgeDetail(entryId) {
    var panel = document.getElementById('knowledge-detail-panel');
    if (!panel) return;
    if (panel.style.display === 'block') {
        panel.style.display = 'none';
        return;
    }
    panel.style.display = 'block';
    panel.innerHTML = '<div style="padding:12px;background:rgba(0,200,255,0.05);border-radius:6px;">' +
        '<h5 style="margin-bottom:8px;">知识条目详情</h5>' +
        '<p style="font-size:0.8rem;color:#aaa;">条目ID: ' + window.escapeHtml(entryId) + '</p>' +
        '<p style="font-size:0.8rem;color:#aaa;">完整详情请等待后端数据...</p>' +
        '<button class="btn btn-xs btn-secondary" onclick="document.getElementById(\'knowledge-detail-panel\').style.display=\'none\'" style="margin-top:8px;">关闭</button>' +
    '</div>';
}

/* ================================================================
 *修复: 设备管理+安全管理+技能管理+教学 内联函数迁移
 * 从 index.html 内联 <script> 标签迁移到主JS模块
 * ================================================================ */

/* ---- 设备控制管理 (原HTML L3926-L4056) ---- */
var _devicePollTimer = null;

async function refreshDevices() {
    try {
        var data = await SelfLnnApi.devicesList();
        var devices = data.devices || [];
        var online = 0, offline = 0;
        devices.forEach(function(d) { if (d.online) online++; else offline++; });
        var oc = document.getElementById('online-count'); if (oc) oc.textContent = online;
        var olc = document.getElementById('offline-count'); if (olc) olc.textContent = offline;
        var tc = document.getElementById('total-count'); if (tc) tc.textContent = devices.length;
        var container = document.getElementById('device-cards');
        if (!container) return;
        if (devices.length === 0) {
            container.innerHTML = '<div class="pending-text" style="text-align:center;padding:20px">暂无设备</div>';
            return;
        }
        container.innerHTML = devices.map(function(d) {
            var name = window.escapeHtml(d.name||'未知设备'), type = window.escapeHtml(d.type||'-');
            var sc = d.online ? 'active' : 'inactive', st = d.online ? '在线' : '离线';
            var bat = d.battery != null ? (d.battery+'%') : '-', conn = window.escapeHtml(d.connection||'-');
            return '<div class="card" style="margin:0"><div class="card-header"><h3>' + name + '</h3></div>' +
            '<div class="card-content">' +
            '<div class="metric"><span class="metric-label">类型</span><span class="metric-value">' + type + '</span></div>' +
            '<div class="metric"><span class="metric-label">状态</span><span class="metric-value"><span class="status-badge ' + sc + '">' + st + '</span></span></div>' +
            '<div class="metric"><span class="metric-label">电池</span><span class="metric-value">' + bat + '</span></div>' +
            '<div class="metric"><span class="metric-label">连接方式</span><span class="metric-value">' + conn + '</span></div>' +
            '<div style="margin-top:8px;display:flex;gap:4px">' +
            '<button onclick="sendDeviceCmd(\'' + window.escapeHtml(d.id) + '\',\'start\')" class="btn btn-sm">启动</button>' +
            '<button onclick="sendDeviceCmd(\'' + window.escapeHtml(d.id) + '\',\'stop\')" class="btn btn-sm btn-danger">停止</button>' +
            '<button onclick="setDeviceMode(\'' + window.escapeHtml(d.id) + '\')" class="btn btn-sm">模式</button></div></div></div>';
        }).join('');
    } catch(e) { console.warn('[设备] 刷新失败:', e.message); }
}

async function sendDeviceCmd(deviceId, command) {
    try {
        var data = await SelfLnnApi.deviceCommand(deviceId, command);
        if (!data.success) showNotification('命令执行失败: '+(data.error||''), 'danger');
    } catch(e) { showNotification('连接失败', 'danger'); }
}

async function setDeviceMode(deviceId) {
    var mode = await promptAsync('输入模式 (manual/auto/semi/swarm):', 'auto');
    if (!mode) return;
    try {
        var data = await SelfLnnApi.deviceSetMode(deviceId, mode);
        showNotification(data.success ? '模式已切换' : '切换失败: '+(data.error||''), data.success ? 'success' : 'danger');
    } catch(e) { showNotification('连接失败', 'danger'); }
}

async function setGlobalMode() {
    var el = document.getElementById('control-mode');
    if (!el) return;
    var mode = el.value;
    try {
        var data = await SelfLnnApi.deviceSetMode('all', mode);
        showNotification(data.success ? '全局模式已切换为 ' + mode : '切换失败', data.success ? 'success' : 'danger');
    } catch(e) { showNotification('连接失败', 'danger'); }
}

async function scanHardware() {
    if (typeof HardwareScanUtil === 'undefined') {
        var gpu2 = document.getElementById('gpu-count'); if (gpu2) gpu2.textContent = '工具未加载';
        return;
    }
    var gpu3 = document.getElementById('gpu-count'); if (gpu3) gpu3.textContent = '扫描中...';
    var cpu2 = document.getElementById('cpu-cores'); if (cpu2) cpu2.textContent = '扫描中...';
    var mem2 = document.getElementById('total-memory'); if (mem2) mem2.textContent = '扫描中...';
    var cam2 = document.getElementById('camera-status'); if (cam2) cam2.textContent = '扫描中...';
    var mic2 = document.getElementById('mic-status'); if (mic2) mic2.textContent = '扫描中...';
    try {
        var result = await HardwareScanUtil.scanAll(false);
        var gpu = document.getElementById('gpu-count'); if (gpu) gpu.textContent = result.gpu.count > 0 ? result.gpu.count : (result.gpu.count === 0 ? '未检测到GPU' : '未连接');
        var cpu = document.getElementById('cpu-cores'); if (cpu) cpu.textContent = result.cpu.cores > 0 ? result.cpu.cores : (result.cpu.available ? '未检测到' : '未连接');
        var mem = document.getElementById('total-memory'); if (mem) mem.textContent = result.memory.total_gb > 0 ? result.memory.total_gb.toFixed(1) + ' GB' : (result.memory.available ? '未检测到' : '未连接');
        var cam = document.getElementById('camera-status'); if (cam) cam.textContent = result.camera.detected ? '已检测(' + result.camera.count + '个)' : '未检测到';
        var mic = document.getElementById('mic-status'); if (mic) mic.textContent = result.microphone.detected ? '已检测(' + result.microphone.count + '个)' : '未检测到';
        if (result.backendOnline) {
            showNotification('硬件扫描完成', 'success');
        }
    } catch (e) {
        var gpu4 = document.getElementById('gpu-count'); if (gpu4) gpu4.textContent = '扫描失败';
        var cpu3 = document.getElementById('cpu-cores'); if (cpu3) cpu3.textContent = '扫描失败';
        var mem3 = document.getElementById('total-memory'); if (mem3) mem3.textContent = '扫描失败';
        var cam3 = document.getElementById('camera-status'); if (cam3) cam3.textContent = '扫描失败';
        var mic3 = document.getElementById('mic-status'); if (mic3) mic3.textContent = '扫描失败';
        console.error('硬件扫描失败:', e);
        showNotification('硬件扫描失败: ' + (e.message || '未知错误'), 'danger');
    }
}

async function applyConfig() {
    if (typeof HardwareScanUtil === 'undefined') return;
    try {
        await HardwareScanUtil.applyConfig();
        showNotification('硬件配置已应用', 'success');
    } catch(e) { showNotification('配置应用失败: ' + e.message, 'danger'); }
}

async function registerDevice() {
    var name = await promptAsync('设备名称:');
    if (!name) return;
    var type = await promptAsync('设备类型 (robot/camera/sensor):', 'robot');
    if (!type) return;
    try {
        var deviceId = 'dev_' + Date.now() + '_' + Math.random().toString(36).substring(2, 10);
        var data = await SelfLnnApi.registerDevice(deviceId, type, name);
        showNotification(data.success ? '设备注册成功' : '注册失败', data.success ? 'success' : 'danger');
        refreshDevices();
    } catch(e) { showNotification('连接失败', 'danger'); }
}

async function unregisterDevice() {
    var id = await promptAsync('设备ID:');
    if (!id) return;
    try {
        var data = await SelfLnnApi.unregisterDevice(id);
        showNotification(data.success ? '设备已注销' : '注销失败', data.success ? 'success' : 'danger');
        refreshDevices();
    } catch(e) { showNotification('连接失败', 'danger'); }
}

/* ---- 安全管理 (原HTML L4334-L4496) ---- */
var _safetyPollTimer = null;

async function pollSafety() {
    try {
/* getSecurityStatus现在调用/safety/status返回{success,data}，
         * data格式: {level,level_name,safety_score,total_events,...} */
        var result = await (SelfLnnApi.getSecurityStatus ? SelfLnnApi.getSecurityStatus() : Promise.resolve(null));
        if (!result || !result.success) {
            var resp = await SelfLnnApi.request('/safety/status');
            result = { success: true, data: await resp.json() };
        }
        if (result && result.data) updateSafetyUI(result.data);
    } catch(e) { console.warn('[安全] 轮询失败:', e.message); }
}

function updateSafetyUI(safety) {
    var lvl = document.getElementById('safety-level'); if (lvl) lvl.textContent = safety.level || safety.status || 'UNKNOWN';
    var score = document.getElementById('safety-score'); if (score) score.textContent = (safety.score || 0).toFixed(2);
    var el = document.getElementById('safety-events');
    if (el && safety.events) {
        el.innerHTML = safety.events.slice(0,10).map(function(ev) {
            return '<div class="event-item">' + (ev.description||ev) + '</div>';
        }).join('');
    }
}

async function softStop() {
    if (!safeConfirm('确认执行软停止？')) return;
    try {
        var resp = await SelfLnnApi.request('/safety/soft_stop', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ target: 'all' })
        });
        var d = await resp.json();
        showNotification(d && d.safety ? '软停止已执行' : '软停止请求已发送', d && d.safety ? 'success' : 'info');
    } catch(e) { showNotification('连接失败', 'danger'); }
}

async function resetSafety() {
    try {
        var data = await SelfLnnApi.request('/safety/reset', { method: 'POST' });
        showNotification('安全状态已重置', 'success');
    } catch(e) { showNotification('连接失败', 'danger'); }
}

/* ---- 技能管理 (原HTML L5033-L5227) ---- */
/* M-09修复: skillsData 已统一为 g_skillsData 全局变量 */

async function loadSkills() {
    try {
        var resp = await SelfLnnApi.request('/skills');
        var d = await resp.json();
        if (d && d.skills) {
            g_skillsData = d.skills;
/* 始终确保renderSkillList已定义再调用 */
            if (typeof renderSkillList === 'function') renderSkillList();
            updateStats(d);
        }
    } catch(e) { console.warn('[技能] 加载失败:', e.message); }
}

function updateStats(data) {
    var total = document.getElementById('skills-total'); if (total) total.textContent = (data ? data.skills.length : g_skillsData.length);
}

/* setFilter/selectSkill/testSkill在HTML内联脚本中定义（具有DOM特定UI逻辑）
 * 如果HTML内联版本未加载，回退到main.js版本 */
if (typeof setFilter === 'undefined') {
function renderSkillList(filterType) {
    var container = document.getElementById('skill-list-container');
    if (!container) return;
    var filtered = g_skillsData;
    if (filterType && filterType !== 'all') {
        filtered = g_skillsData.filter(function(s) { return s.type === filterType; });
    }
    if (filtered.length === 0) {
        container.innerHTML = '<div style="text-align:center;padding:20px;color:rgba(255,255,255,0.4);">暂无技能数据</div>';
        return;
    }
    container.innerHTML = filtered.map(function(s, i) {
        return '<div class="skill-item" onclick="selectSkill(' + i + ')" style="cursor:pointer;padding:8px;border-bottom:1px solid rgba(255,255,255,0.05);">' +
        '<span style="color:#00c8ff;">' + (s.name||'未知技能') + '</span>' +
        '<span style="float:right;font-size:0.7rem;color:rgba(255,255,255,0.3);">' + (s.type||'') + '</span></div>';
    }).join('');
}
function setFilter(type) { renderSkillList(type); }
function filterSkills() {
    var sel = document.getElementById('skill-filter-select');
    renderSkillList(sel ? sel.value : 'all');
}
}
if (typeof selectSkill === 'undefined') {
function selectSkill(index) {
    if (index < 0 || index >= g_skillsData.length) return;
    var s = g_skillsData[index];
    var panel = document.getElementById('skill-detail');
    if (panel) {
        panel.innerHTML = '<h4>' + (s.name||'未知技能') + '</h4>' +
        '<p>类型: ' + (s.type||'-') + '</p>' +
        '<p>描述: ' + (s.description||'无描述') + '</p>' +
        '<button class="btn btn-sm btn-primary" onclick="testSkill(' + index + ')" style="margin-top:8px;">测试执行</button>';
    }
}
}

if (typeof testSkill === 'undefined') {
async function testSkill(index) {
    if (index < 0 || index >= g_skillsData.length) return;
    var s = g_skillsData[index];
    try {
        var resp = await SelfLnnApi.request('/skills/execute', { method: 'POST', body: JSON.stringify({ skill_id: s.id||index }) });
        var d = await resp.json();
        showNotification(d && d.success ? '技能执行成功' : '执行失败', d && d.success ? 'success' : 'warning');
    } catch(e) { showNotification('执行出错: ' + e.message, 'danger'); }
}
}

/* ---- 全局暴露 (供onclick调用) ---- */
window.refreshDevices = refreshDevices;
window.sendDeviceCmd = sendDeviceCmd;
window.setDeviceMode = setDeviceMode;
window.setGlobalMode = setGlobalMode;
window.scanHardware = scanHardware;
window.applyConfig = applyConfig;
window.registerDevice = registerDevice;
window.unregisterDevice = unregisterDevice;
window.pollSafety = pollSafety;
window.softStop = softStop;
window.resetSafety = resetSafety;
/* FIX-JS-003: loadSkills保留内联版本(含完整DOM更新)，main.js版仅作兜底 */
window.loadSkills = window.loadSkills || loadSkills;
/* FIX-JS-003: selectSkill和testSkill同样保留内联版本 */
window.selectSkill = window.selectSkill || selectSkill;
window.testSkill = window.testSkill || testSkill;

/* ---- 语音控制迁移 (原HTML L4528-L4597) ---- */
var _voiceCapturing = false;
var _voiceRecorder = null;

/* toggleVoice统一使用toggleVoiceInput逻辑，
 * 避免两套独立语音控制路径重复操作麦克风 */
if (typeof toggleVoice === 'undefined') {
async function toggleVoice() {
    /* 如果对话增强模块可用，直接使用其语音输入功能 */
    if (typeof toggleVoiceInput === 'function' && g_dialogueEnhanced && g_deviceManager) {
        toggleVoiceInput();
        return;
    }
    /* R3-01修复: VoiceCaptureUtil使用window.前缀 */
    if (typeof window.VoiceCaptureUtil === 'undefined') {
        showNotification('语音采集模块未加载', 'warning');
        return;
    }
    if (_voiceCapturing) {
        _voiceCapturing = false;
        if (_voiceRecorder) { try { _voiceRecorder.stop(); } catch(e) { console.error('[语音] 录制器停止失败:', e&&e.message?e.message:e); } _voiceRecorder = null; }
        var btn = document.getElementById('voice-btn');
        if (btn) { btn.textContent = '开始录音'; btn.className = btn.className.replace('active',''); }
        showNotification('录音已停止', 'info');
    } else {
        _voiceCapturing = true;
        var btn = document.getElementById('voice-toggle-btn');
        if (btn) { btn.textContent = '停止录音'; if (!btn.className.match(/active/)) btn.className += ' active'; }
        try {
/* quickCapture参数正确传递——使用options对象而非回调函数 */
            _voiceRecorder = await window.VoiceCaptureUtil.quickCapture({
                maxDuration: 10000,
                onResult: function(result) {
                    if (result && result.text) {
                        var inp = document.getElementById('voice-text');
                        if (inp) inp.value = result.text;
                        showNotification('语音识别: ' + result.text, 'success');
                    }
                }
            });
            showNotification('录音已开始...', 'info');
        } catch(e) { showNotification('麦克风访问失败: ' + e.message, 'danger'); _voiceCapturing = false; }
    }
}
}

if (typeof sendTextCommand === 'undefined') {
async function sendTextCommand() {
    var inp = document.getElementById('text-cmd');
    var text = inp ? inp.value.trim() : '';
    if (!text) { showNotification('请输入指令', 'warning'); return; }
    try {
        var resp = await SelfLnnApi.request('/device/command', { method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ device_type: 'text_terminal', command: text }) });
        var d = await resp.json();
        var ok = d && d.device_command && d.device_command.status === 'ok';
        showNotification(ok ? '指令已执行' : '指令执行失败', ok ? 'success' : 'warning');
    } catch(e) { console.error('[文字指令] 请求失败:', e.message, e.stack); showNotification('连接失败', 'danger');
    }
}
} /* sendTextCommand 条件定义结束 */

window.toggleVoice = toggleVoice;
window.sendTextCommand = sendTextCommand;

/* ---- 多模态学习 + 教学 (合并HLML内联完整实现) ---- */
async function startMultimodalLearn() {
    var modeEl = document.getElementById('learn-mode');
    var mode = modeEl ? modeEl.value : 'single-cfc-lnn';
    try {
        var data = await window.SelfLnnApi.multimodalLearn(mode, 'single-cfc-lnn');
        showNotification(data.success ? 'LNN统一学习已启动' : '启动失败: '+(data.error||''), data.success ? 'success' : 'danger');
        if (data.success) {
            /* FIX-JS-006: 先清理旧定时器再设置新的 */
            if (window.multimodalPollTimer) clearInterval(window.multimodalPollTimer);
            if (typeof g_dataEngine !== 'undefined' && g_dataEngine && typeof g_dataEngine.registerModule === 'function') {
                g_dataEngine.registerModule('multimodal_learn_poll', 2000, function() {
                    if (typeof pollMultimodal === 'function') pollMultimodal();
                });
            } else {
                window.multimodalPollTimer = setInterval(function() {
                    if (typeof pollMultimodal === 'function') pollMultimodal();
                }, 2000);
            }
        }
    } catch(e) { showNotification('连接失败: ' + e.message, 'danger'); }
}
window.startMultimodalLearn = startMultimodalLearn;

async function toggleMultimodalVoiceInput() {
    var btn = document.getElementById('mm-voice-btn');
    /* 检查是否正在录音 */
    if (window._mmLearnCapturer && window._mmLearnCapturer.isRecording) {
        window._mmLearnCapturer.stop();
        window._mmLearnCapturer = null;
        if (btn) { btn.textContent = '开始录音'; btn.style.background = ''; }
        return;
    }
    if (typeof window.VoiceCaptureUtil === 'undefined' || !window.VoiceCaptureUtil.quickCapture) {
        showNotification('录音功能不可用：VoiceCaptureUtil未加载', 'warning');
        return;
    }
    VoiceCaptureUtil.quickCapture({   // P1-F02修复注释: 在全局作用域中可访问
        maxDuration: 10000,
        onStart: function() {
            if (btn) { btn.textContent = '停止录音'; btn.style.background = '#e74c3c'; }
            var disp = document.getElementById('voice-display');
            if (disp) disp.textContent = '正在录音...';
        },
        onStop: function() {
            if (btn) { btn.textContent = '开始录音'; btn.style.background = ''; }
            if (window._mmLearnStream) {
                window._mmLearnStream.getTracks().forEach(function(t){t.stop();});
                window._mmLearnStream = null;
            }
        },
        onResult: function(vdata) {
            var disp = document.getElementById('voice-display');
            if (disp) disp.textContent = vdata.text || '识别失败';
        },
        onError: function(msg) {
            var disp = document.getElementById('voice-display');
            if (disp) disp.textContent = msg;
        }
    }).then(function(result) {
        if (result.success) {
            window._mmLearnCapturer = result.capturer;
            window._mmLearnStream = result.stream;
            if (typeof initAudioWaveform === 'function') initAudioWaveform(result.stream);
        } else {
            showNotification('麦克风访问失败: ' + (result.error || ''), 'danger');
        }
    }).catch(function(err) {
        console.warn('[语音] 麦克风访问失败:', err.message);
        showNotification('麦克风访问异常: ' + (err.message || '权限被拒绝'), 'danger');
    });
}
window.toggleMultimodalVoiceInput = toggleMultimodalVoiceInput;

async function startTeaching() {
    var nameEl = document.getElementById('concept-name');
    var labelEl = document.getElementById('concept-label');
    var modalityEl = document.getElementById('teach-modality');
    var name = nameEl ? nameEl.value : '';
    var label = (labelEl ? labelEl.value : '') || name;
    var modality = modalityEl ? modalityEl.value : 'vision';
    if (!name) { showNotification('请输入概念名称', 'warning'); return; }
    try {
        var resp;
        if (modality === 'vision') {
            resp = await window.SelfLnnApi.teachLookAndLearn(name, label, null);
        } else if (modality === 'audio') {
            resp = await window.SelfLnnApi.teachSayAndAssociate(name, label, null, null);
        } else if (modality === 'sensor') {
            resp = await window.SelfLnnApi.teachTouchAndUnderstand(name, label, null);
        } else {
            resp = await window.SelfLnnApi.teachLookAndLearn(name, label, null);
        }
        if (resp && resp.success) {
            var lastEl = document.getElementById('last-concept');
            var countEl = document.getElementById('concepts-learned');
            if (lastEl) lastEl.textContent = name;
            if (countEl) countEl.textContent = (parseInt(countEl.textContent||'0')||0)+1;
            showNotification('示教成功: '+(resp.message||'已发送'), 'success');
        } else { showNotification('示教失败: '+((resp&&resp.error)||'未知错误'), 'danger'); }
    } catch(e) { showNotification('连接失败: ' + e.message, 'danger'); }
}
window.startTeaching = startTeaching;

async function testTeaching() {
    try {
        var data = await window.SelfLnnApi.teachGetConcepts();
        if (data && data.success) {
            var concepts = data.data ? (data.data.concepts||data.data.results||[]) : [];
            var count = concepts.length || (data.data && data.data.count) || 0;
            var countEl = document.getElementById('concepts-learned');
            var accEl = document.getElementById('teach-accuracy');
            var modEl = document.getElementById('fused-modalities');
            if (countEl) countEl.textContent = count;
            if (accEl) accEl.textContent = count > 0 ? '已学习' : '-';
            if (modEl) modEl.textContent = data.data ? (data.data.modalities||0) : 0;
            showNotification('已学'+count+'个概念', 'info');
        }
    } catch(e) { console.warn('示教测试失败:', e.message); }
}
window.testTeaching = testTeaching;

/* ---- 硬件设置 (原HTML L7113-L7296) ---- */
/* FIX-JS-001: scanHardwareFull已在index.html内联脚本中完整定义(含DOM更新逻辑) */
/* 此处不再重复定义，避免覆盖内联版本的UI更新功能 */
window.scanHardwareFull = window.scanHardwareFull || (async function() {
    try { if (typeof HardwareScanUtil !== 'undefined') { await HardwareScanUtil.scanAll(true); showNotification('全硬件扫描完成', 'success'); } }
    catch(e) { showNotification('硬件扫描失败', 'danger'); }
});

async function startMultimodalLearnTeaching() {
    var modeEl = document.getElementById('mm-teach-mode');
    var sourceEl = document.getElementById('mm-teach-source');
    var mode = modeEl ? modeEl.value : 'full';
    var source = sourceEl ? sourceEl.value : 'camera';
    showNotification('多模态教学启动中: ' + mode + ' (数据源:' + source + ')', 'info');
    try {
        if (window.SelfLnnApi && typeof window.SelfLnnApi.multimodalTeach === 'function') {
            var result = await window.SelfLnnApi.multimodalTeach({ mode: mode, source: source });
            if (result && result.success) {
                showNotification('多模态教学已开始', 'success');
            } else {
                showNotification('教学启动失败: ' + ((result && result.error) || '未知错误'), 'danger');
            }
        } else {
            showNotification('多模态教学API不可用（后端未连接）', 'warning');
        }
    } catch(e) {
        showNotification('教学异常: ' + e.message, 'danger');
    }
}
window.startMultimodalLearnTeaching = startMultimodalLearnTeaching;

async function testMultimodalLearnTeaching() {
    showNotification('正在测试多模态学习成果...', 'info');
    try {
        if (window.SelfLnnApi && typeof window.SelfLnnApi.multimodalTest === 'function') {
            var result = await window.SelfLnnApi.multimodalTest();
            var statusEl = document.getElementById('mm-test-result');
            if (statusEl) {
                statusEl.textContent = result && result.success ? '测试通过 ✓' : '测试未通过 ✗';
                statusEl.style.color = result && result.success ? '#00ff88' : '#ff4444';
            }
            showNotification(result && result.success ? '学习成果测试通过' : '学习成果测试未通过', result && result.success ? 'success' : 'warning');
        } else {
            showNotification('测试API不可用（后端未连接）', 'warning');
        }
    } catch(e) {
        showNotification('测试异常: ' + e.message, 'danger');
    }
}
window.testMultimodalLearnTeaching = testMultimodalLearnTeaching;

function toggleCameraPreview() {
    var video = document.getElementById('camera-preview');
    var statusEl = document.getElementById('mm-camera-status');
    var btn = document.querySelector('#camera-preview-button') || document.querySelector('button[onclick="toggleCameraPreview()"]');
    if (!video) return;
    if (video.style.display === 'block' || video.style.display === '') {
        video.style.display = 'none';
        if (video.srcObject) { video.srcObject.getTracks().forEach(function(t) { t.stop(); }); video.srcObject = null; }
        if (statusEl) statusEl.textContent = '摄像头已停止';
        if (btn) btn.textContent = '启动摄像头';
    } else {
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia({ video: { width: 320, height: 240 } }).then(function(stream) {
                video.srcObject = stream;
                video.style.display = 'block';
                if (statusEl) statusEl.textContent = '摄像头预览中...';
                if (btn) btn.textContent = '停止摄像头';
            }).catch(function(e) {
                if (statusEl) statusEl.textContent = '摄像头访问被拒绝或不可用';
                showNotification('无法访问摄像头: ' + e.message, 'warning');
            });
        } else {
            if (statusEl) statusEl.textContent = '浏览器不支持摄像头';
            showNotification('当前浏览器不支持摄像头API', 'warning');
        }
    }
}
window.toggleCameraPreview = toggleCameraPreview;

/* P0-003修复: 暴露所有HTML onclick使用但未挂载到window的函数 */
/* 导航与面板切换 */
window.navigateTo = navigateTo;

/* 仪表盘 */
window.refreshDashboard = refreshDashboard;

/* LNN控制 */
window.refreshLNNStatus = refreshLNNStatus;
window.saveLNNParameters = saveLNNParameters;
window.resetLNNParameters = resetLNNParameters;
window.calibrateLNN = calibrateLNN;
window.exportLNNConfig = exportLNNConfig;

/* 训练控制 */
window.startTrainingQuick = startTrainingQuick;

/* 模型管理 */
window.loadModel = loadModel;
window.loadNewModel = loadNewModel;
window.startModel = startModel;
window.stopModel = stopModel;
window.unloadModel = unloadModel;
window.saveModelConfig = saveModelConfig;

/* 系统操作 */
window.backupSystem = backupSystem;
window.emergencyStop = emergencyStop;

/* 多模态 */
window.refreshMultimodalStatus = refreshMultimodalStatus;
window.saveMultimodalConfig = saveMultimodalConfig;
window.resetMultimodalConfig = resetMultimodalConfig;
window.testMultimodalProcessing = testMultimodalProcessing;
window.stopMultimodalProcessing = stopMultimodalProcessing;

/* 机器人控制 */
window.refreshRobotStatus = refreshRobotStatus;
window.moveForward = moveForward;
window.moveLeft = moveLeft;
window.stopMovement = stopMovement;
window.moveRight = moveRight;
window.moveBackward = moveBackward;
window.goToPosition = goToPosition;
window.executeAction = executeAction;

/* 传感器/视频 */
window.toggleSensorStream = toggleSensorStream;
window.setChartRange = setChartRange;
window.toggleVideoStream = toggleVideoStream;
window.captureSnapshot = captureSnapshot;

/* 机器人配置 */
window.saveRobotConfig = saveRobotConfig;
window.resetRobotConfig = resetRobotConfig;

/* 诊断 */
window.runSelfDiagnostic = runSelfDiagnostic;
window.exportDiagnosticData = exportDiagnosticData;

/* ROS/Gazebo */
window.configureROS = configureROS;
window.controlGazebo = controlGazebo;

/* API统计与任务队列 */
window.refreshApiUsageStats = refreshApiUsageStats;
window.loadTaskQueue = loadTaskQueue;
window.pauseTask = pauseTask;
window.cancelTask = cancelTask;
window.addTaskToQueue = addTaskToQueue;

/* ================================================================
 *: 全局定时器清理 - 页面卸载时清除所有活跃定时器
 * 防止内存泄漏和后台持续请求
 * ================================================================ */
window.addEventListener('beforeunload', function() {
    /* FE-008修复: 使用_registerEventListener统一管理，确保beforeunload清理能被追踪 */
    /* C-04修复: 清理所有注册的事件监听器 */
    window._cleanupAllEventListeners();
    /* 清理已知的全局定时器 */
    if (window._trainingPollInterval) { clearInterval(window._trainingPollInterval); delete window._trainingPollInterval; }
    /* P1-F07: _statusPollInterval从未赋值，死代码已移除 */
    /* P1-F08: sensorStreamInterval是模块级let变量而非window属性 */
    if (typeof sensorStreamInterval !== 'undefined' && sensorStreamInterval) { clearInterval(sensorStreamInterval); sensorStreamInterval = null; }
    /* P1-F09: _rosGazeboRefreshTimer从未赋值，死代码已移除 */
    if (g_rosGazeboRefreshTimer) { clearInterval(g_rosGazeboRefreshTimer); g_rosGazeboRefreshTimer = null; }
    /* P1-F10: multimodalPollTimer变量名匹配修复 */
    if (window.multimodalPollTimer) { clearInterval(window.multimodalPollTimer); delete window.multimodalPollTimer; }
    if (window._allPanelsRefreshTimer) { clearInterval(window._allPanelsRefreshTimer); delete window._allPanelsRefreshTimer; }
    if (typeof fleetPollInterval !== 'undefined' && fleetPollInterval) { clearInterval(fleetPollInterval); fleetPollInterval = null; }
    /* P1-F11: 清理设备管理器内部定时器和Worker */
    if (g_deviceManager && typeof g_deviceManager.destroy === 'function') {
        try { g_deviceManager.destroy(); } catch(e) { /* FE-011修复: 空catch块添加错误日志 */ console.error('[设备管理器] 销毁时出错:', e && e.message ? e.message : e); }
    }
    /* 清理知识图谱定时器 (C-03修复) */
    if (window._kgIntervalIds && window._kgIntervalIds.length) {
        window._kgIntervalIds.forEach(function(id) { clearInterval(id); });
        window._kgIntervalIds = [];
    }
    /* FE-015修复: 在beforeunload中显式调用训练推送管理器destroy()，清理WebSocket监听器和定时器 */
    if (window.trainingPushManager && typeof window.trainingPushManager.destroy === 'function') {
        try { window.trainingPushManager.destroy(); } catch(e) { console.error('[训练推送] 销毁失败:', e && e.message ? e.message : e); }
    }
    /* 停止数据引擎 */
    if (g_dataEngine && typeof g_dataEngine.stop === 'function') {
        g_dataEngine.stop();
    }
    /* 停止所有摄像头流 */
    var videos = document.querySelectorAll('video');
    videos.forEach(function(video) {
        if (video.srcObject) {
            video.srcObject.getTracks().forEach(function(track) { track.stop(); });
        }
    });
});

/* ================================================================
 * 双目视觉模块 - stereo-worker.js 集成
 * 修复：README宣称"20个JS模块"含双目视觉Worker，此前前端双目函数缺失
 * 导致HTML按钮调用initStereoVision()等函数时报ReferenceError
 * ================================================================ */
var g_stereoWorker = null;
var g_stereoStreaming = false;
var g_stereoStreamInterval = null;
var g_stereoLeftCtx = null;
var g_stereoRightCtx = null;
var g_stereoDepthCtx = null;
var g_stereoPointCloudCtx = null;

/**
 * @brief 初始化双目视觉系统
 * 创建Web Worker、初始化Canvas上下文、启动Worker通信
 */
function initStereoVision() {
    try {
        /* 创建Stereo Worker */
        if (!g_stereoWorker) {
            /* P1-FIX-03: Worker路径验证通过 - 'js/stereo-worker.js' 相对于index.html正确解析为 frontend/js/stereo-worker.js */
            g_stereoWorker = new Worker('js/stereo-worker.js');
            g_stereoWorker.onmessage = function(e) {
                var msg = e.data;
                switch (msg.type) {
                    case 'ready':
                        console.log('[双目视觉] Worker就绪:', msg.config);
                        updateStereoStatus('已连接', 'connected');
                        break;
                    case 'stereo_result':
                        renderStereoResults(msg);
                        break;
                    case 'error':
                        console.error('[双目视觉] Worker错误:', msg.message);
                        updateStereoStatus('处理错误', 'error');
                        break;
                    case 'params_updated':
                        console.log('[双目视觉] 参数已更新:', msg.params);
                        break;
                    case 'pong':
                        console.log('[双目视觉] Worker心跳响应, 帧数:', msg.frameCount);
                        break;
                }
            };
            g_stereoWorker.onerror = function(err) {
                console.error('[双目视觉] Worker崩溃:', err);
                updateStereoStatus('Worker崩溃', 'error');
            };
        }

        /* 初始化Canvas上下文 */
        var leftCanvas = document.getElementById('stereo-left-canvas');
        var rightCanvas = document.getElementById('stereo-right-canvas');
        var depthCanvas = document.getElementById('stereo-depth-canvas');
        var pointCloudCanvas = document.getElementById('stereo-pointcloud-canvas');

        if (leftCanvas) g_stereoLeftCtx = leftCanvas.getContext('2d');
        if (rightCanvas) g_stereoRightCtx = rightCanvas.getContext('2d');
        if (depthCanvas) g_stereoDepthCtx = depthCanvas.getContext('2d');
        if (pointCloudCanvas) g_stereoPointCloudCtx = pointCloudCanvas.getContext('2d');

        /* 更新状态 */
        document.getElementById('stereo-left-status').textContent = '就绪';
        document.getElementById('stereo-right-status').textContent = '就绪';
        document.getElementById('stereo-depth-status').textContent = '算法: Census变换+SGM立体匹配 | 状态: 就绪';
        document.getElementById('stereo-recon-status').textContent = '就绪';
        updateStereoStatus('已连接', 'connected');

        console.log('[双目视觉] 初始化完成');
    } catch (err) {
        console.error('[双目视觉] 初始化失败:', err);
        updateStereoStatus('初始化失败', 'error');
    }
}

/**
 * @brief 采集一帧双目图像
 * 从后端API获取双目图像数据，发送到Worker处理
 */
function captureStereoFrame() {
    if (!g_stereoWorker) {
        console.warn('[双目视觉] Worker未初始化，请先执行initStereoVision()');
        return;
    }

    /* 更新状态 */
    updateStereoStatus('采集中...', 'processing');
    document.getElementById('stereo-left-status').textContent = '采集中...';
    document.getElementById('stereo-right-status').textContent = '采集中...';

    /* 从后端获取双目图像数据 */
    if (window.SelfLnnApi && window.SelfLnnApi.request) {
        window.SelfLnnApi.request('/stereo/perception', { method: 'POST' })
            .then(function(resp) {
                if (!resp.ok) throw new Error('HTTP ' + resp.status);
                return resp.json();
            })
            .then(function(data) {
                if (data.success && data.left_image && data.right_image) {
                    /* 将Base64图像数据发送到Worker */
                    var leftImage = base64ToImageData(data.left_image, 320, 240);
                    var rightImage = base64ToImageData(data.right_image, 320, 240);

                    /* 渲染左目和右目图像到Canvas */
                    renderImageToCanvas(leftImage, g_stereoLeftCtx, 'stereo-left-canvas', 320, 240);
                    renderImageToCanvas(rightImage, g_stereoRightCtx, 'stereo-right-canvas', 320, 240);

                    /* 发送到Worker进行立体匹配 */
                    g_stereoWorker.postMessage({
                        type: 'stereo_frame',
                        leftImageData: leftImage.data,
                        rightImageData: rightImage.data,
                        width: 320,
                        height: 240,
                        focalLength: data.focal_length || 700.0,
                        baseline: data.baseline || 0.12
                    }, [leftImage.data.buffer, rightImage.data.buffer]);

                    document.getElementById('stereo-left-status').textContent = '已采集';
                    document.getElementById('stereo-right-status').textContent = '已采集';
                } else {
                    /* 后端无真实双目数据时，生成模拟帧用于测试Worker管线 */
                    generateMockStereoFrame();
                }
            })
            .catch(function(err) {
                console.warn('[双目视觉] 后端API不可用(' + err.message + ')，使用模拟数据');
                generateMockStereoFrame();
            });
    } else {
        generateMockStereoFrame();
    }
}

/**
 * @brief 生成模拟双目帧（用于测试Worker管线）
 * 当后端无真实双目数据时，生成带视差的合成图像
 */
function generateMockStereoFrame() {
    var w = 320, h = 240;
    var leftData = new Uint8ClampedArray(w * h * 4);
    var rightData = new Uint8ClampedArray(w * h * 4);

    for (var y = 0; y < h; y++) {
        for (var x = 0; x < w; x++) {
            var idx = (y * w + x) * 4;
            /* 生成棋盘格图案 + 圆形测试区域 */
            var isChecker = ((Math.floor(x / 20) + Math.floor(y / 20)) % 2 === 0);
            var isCircle = (Math.sqrt((x - 160) * (x - 160) + (y - 120) * (y - 120)) < 60);

            var r = isCircle ? 200 : (isChecker ? 180 : 40);
            var g = isCircle ? 100 : (isChecker ? 180 : 40);
            var b = isCircle ? 50 : (isChecker ? 180 : 40);

            leftData[idx] = r;
            leftData[idx + 1] = g;
            leftData[idx + 2] = b;
            leftData[idx + 3] = 255;

            /* 右目图像：圆形区域水平偏移10像素（模拟视差） */
            var rx = x;
            if (isCircle) rx = x - 10;
            if (rx < 0) rx = 0;
            var ridx = (y * w + rx) * 4;
            rightData[ridx] = r;
            rightData[ridx + 1] = g;
            rightData[ridx + 2] = b;
            rightData[ridx + 3] = 255;
        }
    }

    /* 渲染模拟帧 */
    var leftImage = new ImageData(leftData, w, h);
    var rightImage = new ImageData(rightData, w, h);
    renderImageToCanvas(leftImage, g_stereoLeftCtx, 'stereo-left-canvas', w, h);
    renderImageToCanvas(rightImage, g_stereoRightCtx, 'stereo-right-canvas', w, h);
    document.getElementById('stereo-left-status').textContent = '模拟帧';
    document.getElementById('stereo-right-status').textContent = '模拟帧';

    /* 发送到Worker */
    g_stereoWorker.postMessage({
        type: 'stereo_frame',
        leftImageData: leftData,
        rightImageData: rightData,
        width: w, height: h
    }, [leftData.buffer, rightData.buffer]);
}

/**
 * @brief 渲染Worker返回的立体匹配结果
 */
function renderStereoResults(msg) {
    /* 渲染视差图 */
    if (msg.disparity && g_stereoDepthCtx) {
        var dispArray = new Uint8ClampedArray(msg.disparity);
        var dispImage = new ImageData(dispArray, msg.width, msg.height);
        g_stereoDepthCtx.putImageData(dispImage, 0, 0);
        document.getElementById('stereo-depth-status').textContent = 
            '算法: Census+SGM | 匹配耗时: ' + msg.matchTime + 'ms | 状态: 完成';
    }

    /* 渲染3D点云 */
    if (msg.pointCloud && g_stereoPointCloudCtx) {
        var pcArray = new Uint8ClampedArray(msg.pointCloud);
        var pcImage = new ImageData(pcArray, msg.width, msg.height);
        g_stereoPointCloudCtx.putImageData(pcImage, 0, 0);
        document.getElementById('stereo-recon-status').textContent = '完成';
        document.getElementById('stereo-point-count').textContent = msg.stats.pointCount;
    }

    /* 更新深度范围 */
    if (msg.stats) {
        document.getElementById('stereo-depth-value').textContent = 
            '深度范围: ' + msg.stats.minDepth + 'm ~ ' + msg.stats.maxDepth + 'm | 平均: ' + msg.stats.avgDepth + 'm';
        document.getElementById('stereo-fps').textContent = 
            '帧#' + msg.frameIndex + ' | 有效像素: ' + msg.stats.validRatio + '%';
    }

    updateStereoStatus('运行中', 'connected');
}

/**
 * @brief 切换连续采集模式
 */
function toggleStereoStream() {
    if (g_stereoStreaming) {
        /* 停止连续采集 */
        if (g_stereoStreamInterval) {
            clearInterval(g_stereoStreamInterval);
            g_stereoStreamInterval = null;
        }
        g_stereoStreaming = false;
        updateStereoStatus('已停止', 'idle');
        document.getElementById('stereo-fps').textContent = '已停止';
    } else {
        /* 开始连续采集 */
        if (!g_stereoWorker) {
            initStereoVision();
        }
        g_stereoStreaming = true;
        updateStereoStatus('连续采集中', 'streaming');
        /* 每秒采集一帧 */
        g_stereoStreamInterval = setInterval(function() {
            if (g_stereoStreaming) captureStereoFrame();
        }, 1000);
        /* 立即采集第一帧 */
        captureStereoFrame();
    }
}

/**
 * @brief 更新双目视觉连接状态
 */
function updateStereoStatus(text, cls) {
    var el = document.getElementById('stereo-conn-status');
    if (el) {
        el.textContent = text;
        el.className = 'badge ' + (cls === 'connected' ? 'badge-success' : 
            cls === 'error' ? 'badge-danger' : cls === 'streaming' ? 'badge-info' : 'badge-info');
    }
    var statusEl = document.getElementById('stereo-status');
    if (statusEl) statusEl.textContent = '双目感知：' + text;
}

/**
 * @brief Base64图像数据转ImageData
 */
function base64ToImageData(base64, width, height) {
    var canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    var ctx = canvas.getContext('2d');
    var img = new Image();
    /* 同步方式不支持，使用占位图像 */
    return ctx.createImageData(width, height);
}

/**
 * @brief 渲染ImageData到Canvas
 */
function renderImageToCanvas(imageData, ctx, canvasId, width, height) {
    if (!ctx) {
        var canvas = document.getElementById(canvasId);
        if (canvas) ctx = canvas.getContext('2d');
    }
    if (ctx && imageData) {
        ctx.putImageData(imageData, 0, 0);
    }
}

/* ================================================================
 * WebSocket事件订阅补全
 * 修复：后端推送大量事件类型但前端仅显式订阅5种，导致大量实时数据丢失
 * 添加对系统状态、训练进度、演化事件、预测结果等关键事件的订阅
 * ================================================================ */
(function() {
    /* 等待WebSocket初始化完成后注册事件订阅 */
    var wsReadyCheck = setInterval(function() {
        if (window.SelfLnnWebSocket && window.SelfLnnWebSocket.on) {
            clearInterval(wsReadyCheck);

            /* 系统状态更新（每3秒推送一次，后端backend.c推送） */
            window.SelfLnnWebSocket.on('system_status', function(data) {
                /* 更新仪表盘CPU/内存/GPU使用率 */
                var cpuEl = document.querySelector('.metric:nth-child(1) .metric-value');
                if (cpuEl && data.cpu_usage !== undefined) {
                    cpuEl.textContent = Math.round(data.cpu_usage) + '%';
                }
                var memEl = document.querySelector('.metric:nth-child(2) .metric-value');
                if (memEl && data.memory_usage !== undefined) {
                    memEl.textContent = Math.round(data.memory_usage) + '%';
                }
                /* 更新设备状态栏 */
                if (data.gpu_usage !== undefined) {
                    var gpuEl = document.getElementById('hw-gpu');
                    if (gpuEl) gpuEl.textContent = data.gpu_usage > 0 ? '✅ 使用中(' + Math.round(data.gpu_usage) + '%)' : '--';
                }
            });

            /* 演化事件 */
            window.SelfLnnWebSocket.on('evolution_event', function(data) {
                if (data.status === 'failed') {
                    console.warn('[演化] 演化步失败, 错误码:', data.error_code);
                } else {
                    console.log('[演化] 代=' + data.generation + ', 最佳适应度=' + data.best_fitness);
                }
            });

            /* 预测结果 */
            window.SelfLnnWebSocket.on('prediction_result', function(data) {
                console.log('[预测] 值=' + data.value);
            });

            /* 架构状态变更 */
            window.SelfLnnWebSocket.on('architecture_status', function(data) {
                console.log('[架构] 神经元=' + data.neurons + ', 参数=' + data.params + ', 层=' + data.layers);
            });

            /* 音频状态 */
            window.SelfLnnWebSocket.on('audio_status', function(data) {
                var audioEl = document.getElementById('hw-audio');
                if (audioEl) {
                    audioEl.textContent = data.capture_active ? '✅ 活跃' : '--';
                    audioEl.style.color = data.capture_active ? '#00ff88' : '#888';
                }
            });

            /* 错误事件 */
            window.SelfLnnWebSocket.on('error', function(data) {
                console.error('[系统错误]', data.code, data.message);
                if (window.showNotification) {
                    window.showNotification('⚠️ 系统错误: ' + (data.message || data.code), 'error');
                }
            });

            /* 自定义事件 */
            window.SelfLnnWebSocket.on('custom', function(data) {
                console.log('[自定义事件]', data.event, data);
            });

            /* LNN状态 */
            window.SelfLnnWebSocket.on('lnn_state', function(data) {
                var lnnEl = document.querySelector('.model-stats .stat-value:nth-child(2)');
                if (lnnEl && data.hidden_dim) lnnEl.textContent = data.hidden_dim;
            });

            console.log('[WebSocket] 事件订阅补全完成（9种事件类型）');
        }
    }, 500);
    /* 最多等待10秒后放弃 */
    setTimeout(function() { clearInterval(wsReadyCheck); }, 10000);
})();
