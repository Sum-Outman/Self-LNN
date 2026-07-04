/**
 * SELF-LNN AGI 语音指令控制系统
 * 使用 MediaRecorder API 录制语音，通过后端语音识别解析指令
 * 支持运动控制、设备控制、系统控制、计算机控制
 */

/* P2-005修复: IIFE封装，防止VOICE_FEATURE_MAP/VOICE_COMMAND_ROUTES等全局变量污染 */
(function() {
'use strict';

/* 语音指令中文功能名→后端英文字段名映射
 * 修复语音命令"启用自我学习能力"将中文"自我学习能力"直接发送给后端的Bug */
var VOICE_FEATURE_MAP = {
    '自我学习能力': 'self_learning',
    '自我学习': 'self_learning',
    '学习能力': 'self_learning',
    '自我决策能力': 'self_decision',
    '自我决策': 'self_decision',
    '决策能力': 'self_decision',
    '自主执行能力': 'self_execution',
    '自主执行': 'self_execution',
    '执行能力': 'self_execution',
    '模仿学习能力': 'imitation_learning',
    '模仿学习': 'imitation_learning',
    '模仿能力': 'imitation_learning',
    '自我修正能力': 'self_correction',
    '自我修正': 'self_correction',
    '修正能力': 'self_correction',
    '自我反思能力': 'reflection',
    '自我反思': 'reflection',
    '反思能力': 'reflection',
    '规划能力': 'planning',
    '规划': 'planning',
    '自我演化能力': 'self_evolution',
    '自我演化': 'self_evolution',
    '演化能力': 'self_evolution',
    /* P2-003修复: 扩展语音命令覆盖需求中所有功能 */
    '对话功能': 'dialogue',
    '对话': 'dialogue',
    '聊天': 'dialogue',
    '训练功能': 'training',
    '训练': 'training',
    '模型训练': 'training',
    '视觉功能': 'vision',
    '视觉': 'vision',
    '图像识别': 'vision',
    '知识库': 'knowledge',
    '知识查询': 'knowledge',
    '搜索知识': 'knowledge',
    '记忆功能': 'memory',
    '记忆': 'memory',
    '回忆': 'memory',
    '推理功能': 'reasoning',
    '推理': 'reasoning',
    '深度思考': 'deep_thinking',
    '思考': 'deep_thinking',
    '自我认知': 'self_cognition',
    '机器人控制': 'robot_control',
    '控制机器人': 'robot_control',
    '系统状态': 'system_status',
    '系统信息': 'system_status',
    '多模态': 'multimodal',
    '多模态输入': 'multimodal',
    '紧急停止': 'emergency_stop',
    '停止': 'emergency_stop',
    '关机': 'shutdown',
    '重启': 'restart',
    '重新启动': 'restart'
};
/* M024: 映射表自检——启动时验证映射表完整性 */
var voiceCommandCheckFeatureMap = function() {
    var expected = ['self_learning','self_decision','self_execution','imitation_learning',
                    'self_correction','reflection','planning','self_evolution',
                    /* P2-003修复: 验证扩展后的映射表 */
                    'dialogue','training','vision','knowledge','memory','reasoning',
                    'deep_thinking','self_cognition','robot_control','system_status',
                    'multimodal','emergency_stop','shutdown','restart'];
    var found = {};
    var keys = Object.keys(VOICE_FEATURE_MAP);
    for (var i = 0; i < keys.length; i++) found[VOICE_FEATURE_MAP[keys[i]]] = true;
    var missing = [];
    for (var j = 0; j < expected.length; j++) {
        if (!found[expected[j]]) missing.push(expected[j]);
    }
    if (missing.length > 0) {
        console.warn('[语音控制] 功能映射表缺少以下后端字段的映射: ' + missing.join(', ') +
                     '。请更新VOICE_FEATURE_MAP。');
    }
};
function voiceCommandTranslateFeature(chineseInput) {
    if (!chineseInput) return '';
    var trimmed = chineseInput.replace(/\s+/g, '');
    if (VOICE_FEATURE_MAP[trimmed]) return VOICE_FEATURE_MAP[trimmed];
    /* 未知中文输入时返回空字符串,避免传递无效参数给后端 */
    return '';
}

class VoiceCommandSystem {
    constructor() {
        /* P2-005修复: 检查VoiceCaptureUtil依赖 */
        if (typeof window.VoiceCaptureUtil === 'undefined') {
            console.error('[VoiceCommandSystem] VoiceCaptureUtil模块未加载，语音指令系统不可用');
            this._capturer = null;
        } else {
            this._capturer = new window.VoiceCaptureUtil({ maxDuration: 15000 });
        }
        if (this._capturer) {
        this._capturer.onStart = function() {
            this.isProcessing = false;
            if (this.onRecordingStart) this.onRecordingStart();
        }.bind(this);
        this._capturer.onStop = function() {
            if (this.onRecordingStop) this.onRecordingStop();
        }.bind(this);
        this._capturer.onProgress = function(duration) {
            if (this.onRecordingProgress) this.onRecordingProgress(duration);
        }.bind(this);
        this._capturer.onBlobReady = function(blob) {
            this._processAudioBlob(blob);
        }.bind(this);
        this._capturer.onError = function(msg) {
            if (this.onError) this.onError(msg);
        }.bind(this);

        this.isProcessing = false;

        this.onRecordingStart = null;
        this.onRecordingStop = null;
        this.onRecordingProgress = null;
        this.onCommandResult = null;
        this.onError = null;

        /* P2-006修复：不自行创建CommandEngine，改为检查外部注入
         * main.js通过setCommandEngine()注入共享的CommandEngine实例 */
        this.commandEngine = null;
        this.continuousMode = false;
        this.continuousInterval = null;
        } /* P2-005: 关闭 if (this._capturer) 块 */
    }

    /* P2-006修复：添加setCommandEngine方法支持外部注入共享引擎 */
    setCommandEngine(engine) {
        this.commandEngine = engine;
    }

    /* P2-005修复: 检查VoiceCaptureUtil依赖时也要处理安全回退 */
    get isRecording() { return this._capturer ? this._capturer.isRecording : false; }

    async startRecording(micStream) {
        if (this.isRecording) return { success: false, error: '录音已在进行中' };
        this.isProcessing = false;
        return this._capturer.start(micStream);
    }

    stopRecording() {
        this._capturer.stop();
    }

    async _processAudioBlob(audioBlob) {
        this.isProcessing = true;
        /* P2-006修复：检查commandEngine是否已被外部注入 */
        if (!this.commandEngine) {
            console.warn('语音指令：commandEngine未注入，跳过指令解析');
            if (this.onCommandResult) {
                this.onCommandResult({ success: false, error: '指令引擎未初始化', command: null });
            }
            this.isProcessing = false;
            return;
        }
        try {
            var uploadResult = await window.VoiceCaptureUtil.uploadBlob(audioBlob);
            /* F-004修复: 严格区分三层返回状态：API失败 / 识别无内容 / 识别成功 */
            if (!uploadResult.success) {
                if (this.onCommandResult) {
                    this.onCommandResult({
                        success: false,
                        error: uploadResult.error || '语音上传/识别失败，服务器未响应',
                        command: null,
                        originalText: null,
                        confidence: -1
                    });
                }
                this.isProcessing = false;
                return;
            }
            /* API成功但未识别到任何文本内容 */
            if (!uploadResult.text || (typeof uploadResult.text === 'string' && uploadResult.text.trim().length === 0)) {
                if (this.onCommandResult) {
                    this.onCommandResult({
                        success: false,
                        error: '语音未识别到有效文本内容',
                        command: null,
                        originalText: null,
                        confidence: -1
                    });
                }
                this.isProcessing = false;
                return;
            }
            /* 识别成功，解析指令 */
            var commandResult = this.commandEngine.parseCommand(uploadResult.text);
            commandResult.originalText = uploadResult.text;
            /* F-019修复：不使用假默认置信度0.8，未提供时设为-1表示未知 */
            commandResult.confidence = (uploadResult.confidence !== undefined && uploadResult.confidence !== null && uploadResult.confidence >= 0)
                ? uploadResult.confidence : -1;
            /* 验证parseCommand返回格式 */
            if (commandResult.success && commandResult.command) {
                var execResult = await this.commandEngine.executeCommand(commandResult);
                /* 附加执行结果到commandResult */
                commandResult.execSuccess = execResult ? execResult.success : false;
                commandResult.execMessage = execResult ? execResult.message : null;
                commandResult.execError = execResult ? execResult.error : null;
            } else if (commandResult.success && !commandResult.command) {
                /* 识别到文本但未匹配到指令 */
                commandResult._noCommand = true;
            } else {
                /* parseCommand返回了失败状态 */
                commandResult.success = false;
                commandResult.error = commandResult.error || '指令解析异常';
            }
            if (this.onCommandResult) this.onCommandResult(commandResult);
        } catch (err) {
            console.error('语音处理失败:', err);
            if (this.onCommandResult) {
                this.onCommandResult({
                    success: false,
                    error: '语音处理异常: ' + (err.message || '未知错误'),
                    command: null,
                    originalText: null,
                    confidence: -1
                });
            }
        } finally {
            this.isProcessing = false;
        }
    }

    startContinuousMode(micStream) {
        if (this.continuousInterval) {
            clearInterval(this.continuousInterval);
            this.continuousInterval = null;
        }
        this.continuousMode = true;
        this.continuousInterval = setInterval(() => {
            if (!this.isRecording && !this.isProcessing && this.continuousMode) {
                this.startRecording(micStream).catch(function(e) {
                    console.warn('连续模式录音启动失败:', e && e.message);
                });
                setTimeout(() => {
                    if (this.isRecording) this.stopRecording();
                }, 3000);
            }
        }, 4000);
    }

    stopContinuousMode() {
        this.continuousMode = false;
        if (this.continuousInterval) {
            clearInterval(this.continuousInterval);
            this.continuousInterval = null;
        }
        if (this.isRecording) {
            this.stopRecording();
        }
    }

    destroy() {
        this.stopContinuousMode();
        this._capturer.destroy();
        this.commandEngine = null;
    }
}

/**
 * 指令解析引擎（语音和文字共享）
 */

/* 指令路由常量表（数据驱动，新增指令只需添加条目）
 * 以 parsed.params 为输入参数源，支持机器人/系统/设备三类API路由。
 * 优先级：VOICE_COMMAND_ROUTES(数据驱动) > handlerMap(回退) */
var VOICE_COMMAND_ROUTES = {
    /* 机器人运动控制 */
    'robotMoveForward':      { type: 'robot', action: 'move_forward' },
    'robotMoveBackward':     { type: 'robot', action: 'move_backward' },
    'robotTurnLeft':         { type: 'robot', action: 'turn_left' },
    'robotTurnRight':        { type: 'robot', action: 'turn_right' },
    'robotStop':             { type: 'robot', action: 'stop' },
    'robotSetSpeed':         { type: 'robot', action: 'set_speed', paramMap: { speed: 'arg1' }, defaults: { speed: 0.5 } },
    'robotStandUp':          { type: 'robot', action: 'stand_up' },
    'robotSitDown':          { type: 'robot', action: 'sit_down' },
    'robotGoHome':           { type: 'robot', action: 'go_home' },
    'robotConnect':          { type: 'robot', action: 'connect' },
    'robotDisconnect':       { type: 'robot', action: 'disconnect' },
    'robotEmergencyStop':    { type: 'robot', action: 'emergency_stop' },
    /* 计算机控制 */
    'computerLaunchApp':     { type: 'system', action: 'launch_app', paramMap: { name: 'arg1' } },
    'computerCloseApp':      { type: 'system', action: 'close_app', paramMap: { name: 'arg1' } },
    'computerTypeText':      { type: 'system', action: 'type_text', paramMap: { text: 'arg1' } },
    'computerScreenshot':    { type: 'system', action: 'screenshot' },
    'computerRestart':       { type: 'system', action: 'restart' },
    'computerShutdown':      { type: 'system', action: 'shutdown' },
    'computerSleep':         { type: 'system', action: 'sleep' },
    'computerLock':          { type: 'system', action: 'lock' },
    'computerVolumeUp':      { type: 'system', action: 'volume_up', paramMap: { value: 'arg1' }, defaults: { value: 10 } },
    'computerVolumeDown':    { type: 'system', action: 'volume_down', paramMap: { value: 'arg1' }, defaults: { value: 10 } },
    'computerMuteToggle':    { type: 'system', action: 'mute_toggle' },
    /* 系统控制 */
    'systemStartTraining':   { type: 'system', action: 'start_training' },
    'systemStopTraining':    { type: 'system', action: 'stop_training' },
    'systemPauseTraining':   { type: 'system', action: 'pause_training' },
    'systemStartEvolution':  { type: 'system', action: 'start_evolution' },
    'systemStopEvolution':   { type: 'system', action: 'stop_evolution' },
    'systemSaveModel':       { type: 'system', action: 'save_model' },
    'systemLoadModel':       { type: 'system', action: 'load_model' },
    'systemStatus':          { type: 'system', action: 'get_status' },
    'systemEnableFeature':   { type: 'system', action: 'enable_feature', paramMap: { feature: 'arg1' }, transform: { feature: function(v) { return voiceCommandTranslateFeature(v || ''); } } },
    'systemDisableFeature':  { type: 'system', action: 'disable_feature', paramMap: { feature: 'arg1' }, transform: { feature: function(v) { return voiceCommandTranslateFeature(v || ''); } } },
    /* 摄像头控制 */
    'cameraTurnOn':          { type: 'device', device: 'camera', action: 'on' },
    'cameraTurnOff':         { type: 'device', device: 'camera', action: 'off' },
    'cameraCapture':         { type: 'device', device: 'camera', action: 'capture' },
    'cameraSwitch':          { type: 'device', device: 'camera', action: 'switch', paramMap: { target: 'arg1' } },
    /* 麦克风控制 */
    'microphoneTurnOn':      { type: 'device', device: 'microphone', action: 'on' },
    'microphoneTurnOff':     { type: 'device', device: 'microphone', action: 'off' },
    'microphoneMuteToggle':  { type: 'device', device: 'microphone', action: 'mute_toggle' },
    /* 扬声器控制 */
    'speakerTurnOn':         { type: 'device', device: 'speaker', action: 'on' },
    'speakerTurnOff':        { type: 'device', device: 'speaker', action: 'off' },
    'speakerVolumeUp':       { type: 'device', device: 'speaker', action: 'volume_up', paramMap: { value: 'arg1' }, defaults: { value: 10 } },
    'speakerVolumeDown':     { type: 'device', device: 'speaker', action: 'volume_down', paramMap: { value: 'arg1' }, defaults: { value: 10 } },
    'speakerMuteToggle':     { type: 'device', device: 'speaker', action: 'mute_toggle' },
    /* ZSF-100修复：补全7个设备控制指令路由（灯光/空调/风扇），
     * 确保语音正则匹配器(_initCommands device类别)产生的handler名称有对应路由条目 */
    'deviceTurnOnLight':     { type: 'device', device: 'light', action: 'on' },
    'deviceTurnOffLight':    { type: 'device', device: 'light', action: 'off' },
    'deviceTurnOnAC':        { type: 'device', device: 'ac', action: 'on' },
    'deviceTurnOffAC':       { type: 'device', device: 'ac', action: 'off' },
    'deviceSetTemperature':  { type: 'device', device: 'ac', action: 'set_temperature', paramMap: { temperature: 'arg1' }, defaults: { temperature: 26 } },
    'deviceTurnOnFan':       { type: 'device', device: 'fan', action: 'on' },
    'deviceTurnOffFan':      { type: 'device', device: 'fan', action: 'off' }
};

/* 数据驱动路由分发函数
 * 从 VOICE_COMMAND_ROUTES 表读取配置，动态调用对应的API方法 */
function voiceCommandDataDrivenRoute(routeDef, parsed) {
    var params = parsed.params || {};
    var resolved = {};
    /* 解析参数映射：将 parsed.params 中的参数按 paramMap 映射到目标参数 */
    if (routeDef.paramMap) {
        var defaults = routeDef.defaults || {};
        var keys = Object.keys(routeDef.paramMap);
        for (var k = 0; k < keys.length; k++) {
            var targetKey = keys[k];
            var sourceKey = routeDef.paramMap[targetKey];
            var raw = params[sourceKey];
            if (routeDef.transform && routeDef.transform[targetKey]) {
                resolved[targetKey] = routeDef.transform[targetKey](raw);
            } else {
                resolved[targetKey] = raw;
            }
            if (resolved[targetKey] === undefined && defaults[targetKey] !== undefined) {
                resolved[targetKey] = defaults[targetKey];
            }
        }
    }
    return resolved;
}

class CommandEngine {
    constructor() {
        this.commands = this._initCommands();
        this.executionHistory = [];
        this.maxHistorySize = 100;
        this.safetyEnabled = true;
        this.allowedCommands = ['robot', 'computer', 'device', 'system', 'camera', 'microphone', 'speaker'];
        /* BUG-18修复：初始化_pendingConfirm，防止高危操作确认时访问未定义属性 */
        this._pendingConfirm = null;
    }

    _initCommands() {
        return {
            robot: {
                patterns: [
                    { regex: /(?:控制)?机器人(前进|向前)(?:\s+(\d+(?:\.\d+)?))?\s*(?:米|步|速度)?/, handler: 'robotMoveForward' },
                    { regex: /(?:控制)?机器人(后退|向后)(?:\s+(\d+(?:\.\d+)?))?\s*(?:米|步|速度)?/, handler: 'robotMoveBackward' },
                    { regex: /(?:控制)?机器人左转(?:\s+(\d+(?:\.\d+)?))?\s*(?:度|角度)?/, handler: 'robotTurnLeft' },
                    { regex: /(?:控制)?机器人右转(?:\s+(\d+(?:\.\d+)?))?\s*(?:度|角度)?/, handler: 'robotTurnRight' },
                    { regex: /(?:控制)?机器人停止/, handler: 'robotStop' },
                    { regex: /(?:控制)?机器人(?:速度|加速)(?:\s+(\d+(?:\.\d+)?))?/, handler: 'robotSetSpeed' },
                    { regex: /(?:控制)?机器人(?:站立|站起)/, handler: 'robotStandUp' },
                    { regex: /(?:控制)?机器人(?:坐下|蹲下)/, handler: 'robotSitDown' },
                    { regex: /(?:控制)?机器人(?:回家|归位|复位)/, handler: 'robotGoHome' },
                    { regex: /(?:控制)?机器人(?:连接|上线)/, handler: 'robotConnect' },
                    { regex: /(?:控制)?机器人(?:断开|下线|断开连接)/, handler: 'robotDisconnect' },
                    { regex: /(?:控制)?机器人紧急停止/, handler: 'robotEmergencyStop' }
                ],
                category: '运动控制'
            },
            computer: {
                patterns: [
                    { regex: /(?:控制)?(?:电脑|计算机)(?:打开|启动|运行)\s*(.+)/, handler: 'computerLaunchApp' },
                    { regex: /(?:控制)?(?:电脑|计算机)(?:关闭|停止|退出)\s*(.+)/, handler: 'computerCloseApp' },
                    { regex: /(?:控制)?(?:电脑|计算机)(?:输入|键入|打字)\s*(.+)/, handler: 'computerTypeText' },
                    { regex: /(?:控制)?(?:电脑|计算机)(?:截图|截屏|屏幕捕获)/, handler: 'computerScreenshot' },
                    { regex: /(?:控制)?(?:电脑|计算机)(?:重启|重新启动)/, handler: 'computerRestart' },
                    { regex: /(?:控制)?(?:电脑|计算机)(?:关机|关闭系统)/, handler: 'computerShutdown' },
                    { regex: /(?:控制)?(?:电脑|计算机)(?:休眠|睡眠)/, handler: 'computerSleep' },
                    { regex: /(?:控制)?(?:电脑|计算机)(?:锁定|锁屏)/, handler: 'computerLock' },
                    { regex: /(?:控制)?(?:电脑|计算机)(?:音量)(?:增加|调高|加大|提高)(?:\s+(\d+))?/, handler: 'computerVolumeUp' },
                    { regex: /(?:控制)?(?:电脑|计算机)(?:音量)(?:减小|调低|降低|减少)(?:\s+(\d+))?/, handler: 'computerVolumeDown' },
                    { regex: /(?:控制)?(?:电脑|计算机)(?:静音|静音开关)/, handler: 'computerMuteToggle' }
                ],
                category: '计算机控制'
            },
            device: {
                patterns: [
                    { regex: /(?:控制)?(?:打开|开启)(?:灯|灯光|照明)/, handler: 'deviceTurnOnLight' },
                    { regex: /(?:控制)?(?:关闭|关掉)(?:灯|灯光|照明)/, handler: 'deviceTurnOffLight' },
                    { regex: /(?:控制)?(?:打开|开启)(?:空调|制冷)/, handler: 'deviceTurnOnAC' },
                    { regex: /(?:控制)?(?:关闭|关掉)(?:空调|制冷)/, handler: 'deviceTurnOffAC' },
                    { regex: /(?:控制)?(?:空调)?设置温度(?:\s+(\d+))?\s*度/, handler: 'deviceSetTemperature' },
                    { regex: /(?:控制)?(?:打开|开启)(?:风扇|排气扇)/, handler: 'deviceTurnOnFan' },
                    { regex: /(?:控制)?(?:关闭|关掉)(?:风扇|排气扇)/, handler: 'deviceTurnOffFan' }
                ],
                category: '设备控制'
            },
            system: {
                patterns: [
                    { regex: /开始训练/, handler: 'systemStartTraining' },
                    { regex: /停止训练/, handler: 'systemStopTraining' },
                    { regex: /暂停训练/, handler: 'systemPauseTraining' },
                    { regex: /开始演化/, handler: 'systemStartEvolution' },
                    { regex: /停止演化/, handler: 'systemStopEvolution' },
                    { regex: /保存模型/, handler: 'systemSaveModel' },
                    { regex: /加载模型/, handler: 'systemLoadModel' },
                    { regex: /(?:系统)?(?:状态|状态报告)/, handler: 'systemStatus' },
                    { regex: /(?:AGI)?(?:启用|开启|激活)\s*(.+能力|.+功能)?/, handler: 'systemEnableFeature' },
                    { regex: /(?:AGI)?(?:禁用|关闭|停用)\s*(.+能力|.+功能)?/, handler: 'systemDisableFeature' }
                ],
                category: '系统控制'
            },
            camera: {
                patterns: [
                    { regex: /(?:控制)?(?:摄像头|相机)(?:打开|开启|启动)/, handler: 'cameraTurnOn' },
                    { regex: /(?:控制)?(?:摄像头|相机)(?:关闭|关掉|停止)/, handler: 'cameraTurnOff' },
                    { regex: /(?:控制)?(?:摄像头|相机)(?:拍照|截图|拍摄)/, handler: 'cameraCapture' },
                    { regex: /(?:控制)?(?:摄像头|相机)切换(?:\s*(.+))?/, handler: 'cameraSwitch' }
                ],
                category: '摄像头控制'
            },
            microphone: {
                patterns: [
                    { regex: /(?:控制)?(?:麦克风|话筒)(?:打开|开启|启动)/, handler: 'microphoneTurnOn' },
                    { regex: /(?:控制)?(?:麦克风|话筒)(?:关闭|关掉|停止)/, handler: 'microphoneTurnOff' },
                    { regex: /(?:控制)?(?:麦克风|话筒)(?:静音|静音开关)/, handler: 'microphoneMuteToggle' }
                ],
                category: '麦克风控制'
            },
            speaker: {
                patterns: [
                    { regex: /(?:控制)?(?:扬声器|喇叭|音箱)(?:打开|开启|启动)/, handler: 'speakerTurnOn' },
                    { regex: /(?:控制)?(?:扬声器|喇叭|音箱)(?:关闭|关掉|停止)/, handler: 'speakerTurnOff' },
                    { regex: /(?:控制)?(?:扬声器|喇叭|音箱)(?:音量|声音)(?:增加|调高)(?:\s+(\d+))?/, handler: 'speakerVolumeUp' },
                    { regex: /(?:控制)?(?:扬声器|喇叭|音箱)(?:音量|声音)(?:减小|调低)(?:\s+(\d+))?/, handler: 'speakerVolumeDown' },
                    { regex: /(?:控制)?(?:扬声器|喇叭|音箱)(?:静音|静音开关)/, handler: 'speakerMuteToggle' }
                ],
                category: '扬声器控制'
            }
        };
    }

    parseCommand(text) {
        if (!text || text.trim().length === 0) {
            return { success: false, error: '指令文本为空', command: null, params: null };
        }
        const trimmed = text.trim();
        for (const [category, config] of Object.entries(this.commands)) {
            for (const pattern of config.patterns) {
                const match = trimmed.match(pattern.regex);
                if (match) {
                    const params = {};
                    if (match[1] !== undefined) params.arg1 = match[1];
                    if (match[2] !== undefined) params.arg2 = parseFloat(match[2]);
                    return {
                        success: true,
                        command: pattern.handler,
                        category: config.category,
                        params: params,
                        rawText: trimmed,
                        matchedPattern: pattern.regex.source
                    };
                }
            }
        }
/* command为null时应标记failure */
        return {
            success: false,
            command: null,
            category: '未识别',
            params: {},
            rawText: trimmed,
            matchedPattern: null,
            error: '无法解析语音指令: "' + trimmed + '"'
        };
    }

    async executeCommand(parsed) {
        if (!parsed || !parsed.command) {
            return { success: false, error: '未识别到有效指令' };
        }
        if (this.safetyEnabled) {
            var safetyCheck = this._safetyCheck(parsed);
            if (!safetyCheck.allowed) {
                /* ZSF-036修复：高危操作需要用户确认，不再静默拒绝 */
                if (safetyCheck.pending_confirm) {
                    var that = this;
                    try {
                        await new Promise(function(resolve, reject) {
                            that._pendingConfirm = {
                                command: parsed.command,
                                reason: safetyCheck.reason,
                                timestamp: Date.now(),
                                resolve: resolve,
                                reject: reject,
                                processed: false,
                                timeoutHandle: setTimeout(function() {
                                    if (that._pendingConfirm) {
                                        that._pendingConfirm.processed = true;
                                        reject(new Error('高危操作确认超时'));
                                    }
                                }, safetyCheck.timeout_ms || 15000)
                            };
                            /* 触发待确认事件供UI层响应 */
                            document.dispatchEvent(new CustomEvent('safety-confirm-required', {
                                detail: {
                                    command: parsed.command,
                                    reason: safetyCheck.reason
                                }
                            }));
                        });
                        /* 用户确认后继续执行 */
                        console.log('[安全] 高危操作已确认，继续执行:', parsed.command);
                    } catch (err) {
                        this._addHistory(parsed, false, '高危操作被拒绝或超时: ' + err.message);
                        return { success: false, error: '高危操作被拒绝或超时' };
                    }
                } else {
                    this._addHistory(parsed, false, safetyCheck.reason);
                    return { success: false, error: safetyCheck.reason };
                }
            }
        }
        try {
            const result = await this._routeCommand(parsed);
            this._addHistory(parsed, result.success, result.message || '');
            document.dispatchEvent(new CustomEvent('command-executed', {
                detail: { command: parsed, result: result }
            }));
            return result;
        } catch (err) {
            this._addHistory(parsed, false, err.message);
            return { success: false, error: err.message };
        }
    }

    _safetyCheck(parsed) {
        var dangerousHandlers = ['computerShutdown', 'computerRestart', 'robotEmergencyStop'];
        if (dangerousHandlers.indexOf(parsed.command) >= 0) {
            /* ZSF-036修复：不再静默拒绝，返回待确认状态触发用户确认对话框 */
            return { 
                allowed: false, 
                pending_confirm: true, 
                command: parsed.command,
                reason: '高危操作，需要用户确认: ' + parsed.command + ' - ' + 
                       (parsed.params && parsed.params.text ? parsed.params.text : ''),
                timeout_ms: 15000
            };
        }
        return { allowed: true, reason: '' };
    }

    /**
     * 确认待处理的高危操作
     * @param {boolean} confirm - 用户是否确认
     */
    confirmPending(confirm) {
        if (!this._pendingConfirm) return;
        if (confirm && this._pendingConfirm.command) {
            console.log('[安全] 高危操作已确认:', this._pendingConfirm.command);
            this._pendingConfirm.approved = true;
            this._pendingConfirm.resolve();
        } else {
            this._pendingConfirm.rejected = true;
            this._pendingConfirm.reject();
        }
        clearTimeout(this._pendingConfirm.timeoutHandle);
        this._pendingConfirm = null;
    }

    /**
     * 获取当前待确认操作信息
     * @returns {object|null}
     */
    getPendingConfirm() {
        if (!this._pendingConfirm || this._pendingConfirm.processed) return null;
        return {
            command: this._pendingConfirm.command,
            reason: this._pendingConfirm.reason,
            timestamp: this._pendingConfirm.timestamp
        };
    }

/* 指令路由常量表（数据驱动，新增指令只需添加条目）
     * 以 parsed.params 为输入参数源，支持机器人/系统/设备三类API路由。
     * 优先级：VOICE_COMMAND_ROUTES(数据驱动) > handlerMap(回退) */
    async _routeCommand(parsed) {
        var route = VOICE_COMMAND_ROUTES[parsed.command];
        if (route) {
            var params = voiceCommandDataDrivenRoute(route, parsed);
            if (route.type === 'robot') return await this._callRobotApi(route.action, params);
            if (route.type === 'system') return await this._callSystemApi(route.action, params);
            if (route.type === 'device') return await this._callDeviceApi(route.device, route.action, params);
        }
        return { success: false, error: '未知指令处理器: ' + parsed.command };
    }

    async _callRobotApi(action, params) {
        if (window.SelfLnnApi && typeof window.SelfLnnApi.sendRobotCommand === 'function') {
            var cmd = { action: action };
            if (params) Object.keys(params).forEach(function(k) { cmd[k] = params[k]; });
            try {
                return await window.SelfLnnApi.sendRobotCommand(cmd);
            } catch (e) {
                return { success: false, error: '机器人指令发送失败: ' + e.message };
            }
        }
        return { success: false, error: '机器人API不可用' };
    }

    async _callSystemApi(action, params) {
        if (!window.SelfLnnApi) return { success: false, error: '系统API服务未加载' };
        const apiMap = {
            'launch_app': async () => window.SelfLnnApi.systemCommand ? await window.SelfLnnApi.systemCommand('launch_app', {name:params.name}) : { success: false, error: 'API不可用' },
            'close_app': async () => window.SelfLnnApi.systemCommand ? await window.SelfLnnApi.systemCommand('close_app', {name:params.name}) : { success: false, error: 'API不可用' },
            'type_text': async () => window.SelfLnnApi.systemCommand ? await window.SelfLnnApi.systemCommand('type_text', {text:params.text}) : { success: false, error: 'API不可用' },
            'screenshot': async () => window.SelfLnnApi.sendCommand ? await window.SelfLnnApi.sendCommand('system','screenshot',{}) : { success: false, error: 'API不可用' },
            'restart': async () => window.SelfLnnApi.sendCommand ? await window.SelfLnnApi.sendCommand('system','restart',{}) : { success: false, error: 'API不可用' },
            'shutdown': async () => window.SelfLnnApi.sendCommand ? await window.SelfLnnApi.sendCommand('system','shutdown',{}) : { success: false, error: 'API不可用' },
            'sleep': async () => window.SelfLnnApi.sendCommand ? await window.SelfLnnApi.sendCommand('system','sleep',{}) : { success: false, error: 'API不可用' },
            'lock': async () => window.SelfLnnApi.sendCommand ? await window.SelfLnnApi.sendCommand('system','lock',{}) : { success: false, error: 'API不可用' },
            'volume_up': async () => window.SelfLnnApi.sendCommand ? await window.SelfLnnApi.sendCommand('system','volume_up',{value:params.value}) : { success: false, error: 'API不可用' },
            'volume_down': async () => window.SelfLnnApi.sendCommand ? await window.SelfLnnApi.sendCommand('system','volume_down',{value:params.value}) : { success: false, error: 'API不可用' },
            'mute_toggle': async () => window.SelfLnnApi.sendCommand ? await window.SelfLnnApi.sendCommand('system','mute_toggle',{}) : { success: false, error: 'API不可用' },
            'start_training': async () => window.SelfLnnApi.startTraining ? await window.SelfLnnApi.startTraining(params) : { success: false, error: 'API不可用' },
            /* ZSF-088修复：统一API命名，stopTraining替代stopTrainingJob */
            'stop_training': async () => {
                if (window.SelfLnnApi.stopTraining) return await window.SelfLnnApi.stopTraining();
                if (window.SelfLnnApi.stopTrainingJob) return await window.SelfLnnApi.stopTrainingJob();
                return { success: false, error: 'API不可用' };
            },
            'pause_training': async () => window.SelfLnnApi.pauseTraining ? await window.SelfLnnApi.pauseTraining() : { success: false, error: 'API不可用' },
            'start_evolution': async () => {
                if (window.SelfLnnApi && typeof window.SelfLnnApi.toggleAgiFeature === 'function') {
                    return await window.SelfLnnApi.toggleAgiFeature('self_evolution', true);
                }
                return { success: false, error: 'API不可用' };
            },
            'stop_evolution': async () => {
                if (window.SelfLnnApi && typeof window.SelfLnnApi.toggleAgiFeature === 'function') {
                    return await window.SelfLnnApi.toggleAgiFeature('self_evolution', false);
                }
                return { success: false, error: 'API不可用' };
            },
            'save_model': async () => {
                if (window.SelfLnnApi && typeof window.SelfLnnApi.saveModelConfig === 'function') {
                    return await window.SelfLnnApi.saveModelConfig({ action: 'voice_command_save' });
                }
                return { success: false, error: 'API不可用' };
            },
            'load_model': async () => {
                if (window.SelfLnnApi && typeof window.SelfLnnApi.loadModel === 'function') {
                    var loadParams = params || {};
                    if (!loadParams.modelId && !loadParams.model_id) {
                        return { success: false, error: '缺少模型ID参数 (modelId)' };
                    }
                    return await window.SelfLnnApi.loadModel(loadParams.modelId || loadParams.model_id);
                }
                return { success: false, error: 'API不可用' };
            },
            'get_status': async () => window.SelfLnnApi.getSystemStatus ? await window.SelfLnnApi.getSystemStatus() : { success: false, error: 'API不可用' },
            'enable_feature': async () => {
                if (window.SelfLnnApi && typeof window.SelfLnnApi.toggleAgiFeature === 'function') {
                    return await window.SelfLnnApi.toggleAgiFeature(params.feature, true);
                }
                return { success: false, error: 'API不可用' };
            },
            'disable_feature': async () => {
                if (window.SelfLnnApi && typeof window.SelfLnnApi.toggleAgiFeature === 'function') {
                    return await window.SelfLnnApi.toggleAgiFeature(params.feature, false);
                }
                return { success: false, error: 'API不可用' };
            }
        };
        const handler = apiMap[action];
        if (handler) {
            const result = await handler();
            if (result) return result;
        }
        return { success: false, error: '系统API不可用: ' + action };
    }

    async _callDeviceApi(deviceType, action, params) {
        document.dispatchEvent(new CustomEvent('device-command', {
            detail: { deviceType: deviceType, action: action, params: params }
        }));
/* 使用全局单例g_deviceManager（main.js创建），
         * 而非创建重复的DeviceManager实例，避免设备列表不同步 */
        var dm = window.g_deviceManager || window.SelfLnnDeviceManager;
        if (!dm) {
            return { success: false, message: '设备管理器不可用，无法执行设备指令: ' + deviceType + '/' + action };
        }
        try {
            var result;
            switch (deviceType) {
                case 'camera':
                    switch (action) {
                        case 'on':
                            var cams = dm.getAvailableCameras();
                            if (cams.length === 0) return { success: false, error: '未找到可用摄像头' };
                            var addResult = await dm.addCamera(cams[0].deviceId);
                            if (addResult.success) {
                                result = await dm.startCamera(addResult.data.id);
                            } else {
                                var existingCam = dm.cameras.find(function(c) { return c.active === false; });
                                if (existingCam) result = await dm.startCamera(existingCam.id);
                                else result = { success: false, error: '无法添加摄像头' };
                            }
                            return result;
                        case 'off':
                            var activeCams = dm.cameras.filter(function(c) { return c.active; });
                            if (activeCams.length > 0) {
                                for (var i = 0; i < activeCams.length; i++) {
                                    dm.stopCamera(activeCams[i].id);
                                }
                            }
                            return { success: true, message: '摄像头已关闭' };
                        case 'capture':
                            var activeCam = dm.cameras.find(function(c) { return c.active; });
                            if (!activeCam) return { success: false, error: '没有活动的摄像头' };
                            var snapshot = dm.captureSnapshot(activeCam.id);
                            if (snapshot && window.SelfLnnApi) {
                                window.SelfLnnApi.captureVideoFrame(activeCam.deviceId);
                            }
                            return { success: true, data: snapshot, message: '拍照完成' };
                        case 'switch':
                            if (params && params.target) {
                                var allCams = dm.getAvailableCameras();
                                var target = allCams.find(function(c) {
                                    return c.label.indexOf(params.target) !== -1 || c.deviceId.indexOf(params.target) !== -1;
                                });
                                if (target) {
                                    var activeCams2 = dm.cameras.filter(function(c) { return c.active; });
                                    for (var j = 0; j < activeCams2.length; j++) {
                                        dm.stopCamera(activeCams2[j].id);
                                    }
                                    var addResult2 = await dm.addCamera(target.deviceId);
                                    if (addResult2.success) {
                                        return await dm.startCamera(addResult2.data.id);
                                    }
                                }
                            }
                            return { success: false, error: '未指定目标摄像头' };
                    }
                    break;
                case 'microphone':
                    switch (action) {
                        case 'on':
                            var mics = dm.getAvailableMicrophones();
                            if (mics.length === 0) return { success: false, error: '未找到可用麦克风' };
                            var addResult = await dm.addMicrophone(mics[0].deviceId);
                            if (addResult.success) {
                                result = await dm.startMicrophone(addResult.data.id);
                            } else {
                                var existingMic = dm.microphones.find(function(m) { return m.active === false; });
                                if (existingMic) result = await dm.startMicrophone(existingMic.id);
                                else result = { success: false, error: '无法添加麦克风' };
                            }
                            return result;
                        case 'off':
                            var activeMics = dm.microphones.filter(function(m) { return m.active; });
                            for (var i = 0; i < activeMics.length; i++) {
                                dm.stopMicrophone(activeMics[i].id);
                            }
                            return { success: true, message: '麦克风已关闭' };
                        case 'mute_toggle':
                            var targetMic = dm.microphones.find(function(m) { return m.active; }) || dm.microphones[0];
                            if (targetMic) {
                                targetMic.muted = !targetMic.muted;
                                return { success: true, muted: targetMic.muted, message: targetMic.muted ? '麦克风已静音' : '麦克风已取消静音' };
                            }
                            return { success: false, error: '没有可用的麦克风' };
                    }
                    break;
                case 'speaker':
                    switch (action) {
                        case 'on':
                        var spks = dm.getAvailableSpeakers();
                        if (spks.length === 0) {
                            /* L-009修复: 无硬件扬声器时返回真实状态，不使用客户端testSpeaker模拟确认
                             * testSpeaker是纯浏览器端Web Audio API，无法验证后端音频通道 */
                            return { success: false, error: '未找到可用扬声器硬件，无法播放音频。请连接物理扬声器设备后重试。' };
                        }
                            var addResult = await dm.addSpeaker(spks[0].deviceId);
                            if (addResult.success) {
                                result = await dm.startSpeaker(addResult.data.id);
                                /* L-009修复: 移除testSpeaker客户端模拟确认 */
                            } else {
                                var existingSpk = dm.speakers.find(function(s) { return s.active === false; });
                                if (existingSpk) result = await dm.startSpeaker(existingSpk.id);
                                else result = { success: false, error: '无法添加扬声器' };
                            }
                            return result && typeof result.success !== 'undefined' ? result : { success: true, message: '扬声器已启动' };
                        case 'off':
                            var activeSpks = dm.speakers.filter(function(s) { return s.active; });
                            for (var i = 0; i < activeSpks.length; i++) {
                                dm.stopSpeaker(activeSpks[i].id);
                            }
                            dm.stopAllAudio();
                            return { success: true, message: '扬声器已关闭' };
                        case 'volume_up':
                            if (dm.speakers.length > 0) {
                                var volSpk = dm.speakers[0];
                                var newVol = Math.min(100, (volSpk.volume || 80) + (parseInt(params.value) || 10));
                                dm.setSpeakerVolume(volSpk.id, newVol);
                            }
                            return { success: true, message: '音量已调高' };
                        case 'volume_down':
                            if (dm.speakers.length > 0) {
                                var volSpk2 = dm.speakers[0];
                                var newVol2 = Math.max(0, (volSpk2.volume || 80) - (parseInt(params.value) || 10));
                                dm.setSpeakerVolume(volSpk2.id, newVol2);
                            }
                            return { success: true, message: '音量已调低' };
                        case 'mute_toggle':
                            if (dm.speakers.length > 0) {
                                var muteSpk = dm.speakers[0];
                                if (muteSpk.muted) {
                                    dm.unmuteSpeaker(muteSpk.id);
                                } else {
                                    dm.muteSpeaker(muteSpk.id);
                                }
                                return { success: true, muted: muteSpk.muted, message: muteSpk.muted ? '扬声器已静音' : '扬声器已取消静音' };
                            }
                            return { success: false, error: '没有可用的扬声器' };
                    }
                    break;
                /* ZSF-100修复：新增灯光、空调、风扇设备控制case，
                 * 通过后端 sendDeviceCommand API 统一处理物联网设备指令 */
                case 'light':
                    if (window.SelfLnnApi && typeof window.SelfLnnApi.sendDeviceCommand === 'function') {
                        return await window.SelfLnnApi.sendDeviceCommand('light', action, params || {});
                    }
                    return { success: false, error: '设备API不可用，无法控制灯光' };
                case 'ac':
                    if (window.SelfLnnApi && typeof window.SelfLnnApi.sendDeviceCommand === 'function') {
                        return await window.SelfLnnApi.sendDeviceCommand('ac', action, params || {});
                    }
                    return { success: false, error: '设备API不可用，无法控制空调' };
                case 'fan':
                    if (window.SelfLnnApi && typeof window.SelfLnnApi.sendDeviceCommand === 'function') {
                        return await window.SelfLnnApi.sendDeviceCommand('fan', action, params || {});
                    }
                    return { success: false, error: '设备API不可用，无法控制风扇' };
            }
            return { success: true, message: '设备指令已发送: ' + deviceType + '/' + action };
        } catch (err) {
            console.error('设备控制失败:', err);
            return { success: false, error: '设备控制失败: ' + err.message };
        }
    }

    _addHistory(parsed, success, message) {
        this.executionHistory.unshift({
            timestamp: Date.now(),
            command: parsed.command,
            category: parsed.category,
            rawText: parsed.rawText,
            success: success,
            message: message
        });
        if (this.executionHistory.length > this.maxHistorySize) {
            this.executionHistory.pop();
        }
    }

    getHistory(count) {
        return this.executionHistory.slice(0, count || 20);
    }

    clearHistory() {
        this.executionHistory = [];
    }
}

window.VoiceCommandSystem = VoiceCommandSystem;
window.CommandEngine = CommandEngine;

/* P2-006修复: 激活映射表自检，死代码复活 */
voiceCommandCheckFeatureMap();

})(); /* P2-005修复: IIFE封装结束 */
