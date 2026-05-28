/**
 * SELF-LNN AGI 语音指令控制系统
 * 使用 MediaRecorder API 录制语音，通过后端语音识别解析指令
 * 支持运动控制、设备控制、系统控制、计算机控制
 */

/* ZSFX-DEEP-R4-001: 语音指令中文功能名→后端英文字段名映射
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
    '演化能力': 'self_evolution'
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
        this._capturer = new window.VoiceCaptureUtil({ maxDuration: 15000 });
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
    }

    /* P2-006修复：添加setCommandEngine方法支持外部注入共享引擎 */
    setCommandEngine(engine) {
        this.commandEngine = engine;
    }

    get isRecording() { return this._capturer.isRecording; }

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
class CommandEngine {
    constructor() {
        this.commands = this._initCommands();
        this.executionHistory = [];
        this.maxHistorySize = 100;
        this.safetyEnabled = true;
        this.allowedCommands = ['robot', 'computer', 'device', 'system', 'camera', 'microphone', 'speaker'];
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
        return {
            success: true,
            command: null,
            category: '未识别',
            params: {},
            rawText: trimmed,
            matchedPattern: null
        };
    }

    async executeCommand(parsed) {
        if (!parsed || !parsed.command) {
            return { success: false, error: '未识别到有效指令' };
        }
        if (this.safetyEnabled) {
            const safetyCheck = this._safetyCheck(parsed);
            if (!safetyCheck.allowed) {
                this._addHistory(parsed, false, safetyCheck.reason);
                return { success: false, error: safetyCheck.reason };
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
            return { allowed: false, reason: '高危操作，需用户确认' };
        }
        return { allowed: true, reason: '' };
    }

    async _routeCommand(parsed) {
        const handlerMap = {
            robotMoveForward: () => this._callRobotApi('move_forward', parsed.params),
            robotMoveBackward: () => this._callRobotApi('move_backward', parsed.params),
            robotTurnLeft: () => this._callRobotApi('turn_left', parsed.params),
            robotTurnRight: () => this._callRobotApi('turn_right', parsed.params),
            robotStop: () => this._callRobotApi('stop', {}),
            robotSetSpeed: () => this._callRobotApi('set_speed', { speed: parsed.params.arg1 || 0.5 }),
            robotStandUp: () => this._callRobotApi('stand_up', {}),
            robotSitDown: () => this._callRobotApi('sit_down', {}),
            robotGoHome: () => this._callRobotApi('go_home', {}),
            robotConnect: () => this._callRobotApi('connect', {}),
            robotDisconnect: () => this._callRobotApi('disconnect', {}),
            robotEmergencyStop: () => this._callRobotApi('emergency_stop', {}),
            computerLaunchApp: () => this._callSystemApi('launch_app', { name: parsed.params.arg1 || '' }),
            computerCloseApp: () => this._callSystemApi('close_app', { name: parsed.params.arg1 || '' }),
            computerTypeText: () => this._callSystemApi('type_text', { text: parsed.params.arg1 || '' }),
            computerScreenshot: () => this._callSystemApi('screenshot', {}),
            computerRestart: () => this._callSystemApi('restart', {}),
            computerShutdown: () => this._callSystemApi('shutdown', {}),
            computerSleep: () => this._callSystemApi('sleep', {}),
            computerLock: () => this._callSystemApi('lock', {}),
            computerVolumeUp: () => this._callSystemApi('volume_up', { value: parsed.params.arg1 || 10 }),
            computerVolumeDown: () => this._callSystemApi('volume_down', { value: parsed.params.arg1 || 10 }),
            computerMuteToggle: () => this._callSystemApi('mute_toggle', {}),
            systemStartTraining: () => this._callSystemApi('start_training', {}),
            systemStopTraining: () => this._callSystemApi('stop_training', {}),
            systemPauseTraining: () => this._callSystemApi('pause_training', {}),
            systemStartEvolution: () => this._callSystemApi('start_evolution', {}),
            systemStopEvolution: () => this._callSystemApi('stop_evolution', {}),
            systemSaveModel: () => this._callSystemApi('save_model', {}),
            systemLoadModel: () => this._callSystemApi('load_model', {}),
            systemStatus: () => this._callSystemApi('get_status', {}),
            systemEnableFeature: () => this._callSystemApi('enable_feature', { feature: voiceCommandTranslateFeature(parsed.params.arg1 || '') }),
            systemDisableFeature: () => this._callSystemApi('disable_feature', { feature: voiceCommandTranslateFeature(parsed.params.arg1 || '') }),
            cameraTurnOn: () => this._callDeviceApi('camera', 'on', {}),
            cameraTurnOff: () => this._callDeviceApi('camera', 'off', {}),
            cameraCapture: () => this._callDeviceApi('camera', 'capture', {}),
            cameraSwitch: () => this._callDeviceApi('camera', 'switch', { target: parsed.params.arg1 || '' }),
            microphoneTurnOn: () => this._callDeviceApi('microphone', 'on', {}),
            microphoneTurnOff: () => this._callDeviceApi('microphone', 'off', {}),
            microphoneMuteToggle: () => this._callDeviceApi('microphone', 'mute_toggle', {}),
            speakerTurnOn: () => this._callDeviceApi('speaker', 'on', {}),
            speakerTurnOff: () => this._callDeviceApi('speaker', 'off', {}),
            speakerVolumeUp: () => this._callDeviceApi('speaker', 'volume_up', { value: parsed.params.arg1 || 10 }),
            speakerVolumeDown: () => this._callDeviceApi('speaker', 'volume_down', { value: parsed.params.arg1 || 10 }),
            speakerMuteToggle: () => this._callDeviceApi('speaker', 'mute_toggle', {})
        };
        const handler = handlerMap[parsed.command];
        if (!handler) {
            return { success: false, error: '未知指令处理器: ' + parsed.command };
        }
        return await handler();
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
            'launch_app': async () => window.SelfLnnApi.systemCommand ? await window.SelfLnnApi.systemCommand('launch_app', {name:params.name}) : null,
            'close_app': async () => window.SelfLnnApi.systemCommand ? await window.SelfLnnApi.systemCommand('close_app', {name:params.name}) : null,
            'type_text': async () => window.SelfLnnApi.systemCommand ? await window.SelfLnnApi.systemCommand('type_text', {text:params.text}) : null,
            'screenshot': async () => window.SelfLnnApi.sendCommand ? await window.SelfLnnApi.sendCommand('system','screenshot',{}) : null,
            'restart': async () => window.SelfLnnApi.sendCommand ? await window.SelfLnnApi.sendCommand('system','restart',{}) : null,
            'shutdown': async () => window.SelfLnnApi.sendCommand ? await window.SelfLnnApi.sendCommand('system','shutdown',{}) : null,
            'sleep': async () => window.SelfLnnApi.sendCommand ? await window.SelfLnnApi.sendCommand('system','sleep',{}) : null,
            'lock': async () => window.SelfLnnApi.sendCommand ? await window.SelfLnnApi.sendCommand('system','lock',{}) : null,
            'volume_up': async () => window.SelfLnnApi.sendCommand ? await window.SelfLnnApi.sendCommand('system','volume_up',{value:params.value}) : null,
            'volume_down': async () => window.SelfLnnApi.sendCommand ? await window.SelfLnnApi.sendCommand('system','volume_down',{value:params.value}) : null,
            'mute_toggle': async () => window.SelfLnnApi.sendCommand ? await window.SelfLnnApi.sendCommand('system','mute_toggle',{}) : null,
            'start_training': async () => window.SelfLnnApi.startTraining ? await window.SelfLnnApi.startTraining(params) : null,
            'stop_training': async () => window.SelfLnnApi.stopTrainingJob ? await window.SelfLnnApi.stopTrainingJob() : null,
            'pause_training': async () => window.SelfLnnApi.pauseTraining ? await window.SelfLnnApi.pauseTraining() : null,
            'start_evolution': async () => window.SelfLnnApi.startEvolution ? await window.SelfLnnApi.startEvolution(params) : null,
            'stop_evolution': async () => {
                // 使用 toggleAgiFeature 正确关闭自我演化功能，而非错误调用 startEvolution
                if (window.SelfLnnApi && typeof window.SelfLnnApi.toggleAgiFeature === 'function') {
                    return await window.SelfLnnApi.toggleAgiFeature('self_evolution', false);
                }
                return null;
            },
            'save_model': async () => window.SelfLnnApi.backupSystem ? await window.SelfLnnApi.backupSystem() : null,
            'load_model': async () => {
                if (window.SelfLnnApi && typeof window.SelfLnnApi.loadModel === 'function') {
                    var loadParams = params || {};
                    if (!loadParams.modelId && !loadParams.model_id) {
                        return { success: false, error: '缺少模型ID参数 (modelId)' };
                    }
                    return await window.SelfLnnApi.loadModel(loadParams.modelId || loadParams.model_id);
                }
                return null;
            },
            'get_status': async () => window.SelfLnnApi.getSystemStatus ? await window.SelfLnnApi.getSystemStatus() : null,
            'enable_feature': async () => {
                if (window.SelfLnnApi && typeof window.SelfLnnApi.toggleAgiFeature === 'function') {
                    return await window.SelfLnnApi.toggleAgiFeature(params.feature, true);
                }
                return null;
            },
            'disable_feature': async () => {
                if (window.SelfLnnApi && typeof window.SelfLnnApi.toggleAgiFeature === 'function') {
                    return await window.SelfLnnApi.toggleAgiFeature(params.feature, false);
                }
                return null;
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
        /* ZSFWS-S009修复: 使用全局单例g_deviceManager（main.js创建），
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
