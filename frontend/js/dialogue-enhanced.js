/**
 * SELF-LNN AGI 增强对话系统
 * 支持语音输入/输出、多模态对话（视觉+语音+文字）、摄像头画面发送
 * 集成TTS语音播报和语音识别录制
 */

'use strict';

class DialogueEnhanced {
    constructor() {
        this.voiceInputEnabled = false;
        this.voiceOutputEnabled = false;
        this.isSpeaking = false;
        this._capturer = null;
        this.audioPlayer = null;
        this.speechQueue = [];
        this.isPlayingQueue = false;
        this.ttsSpeed = 1.0;
        this.currentCameraId = null;
        this.multimodalEnabled = false;
        this.lastCapturedImage = null;

        /* S-008修复: 显式声明流式对话回调属性，避免隐式外部赋值风险 */
        this.onDialogueToken = null;       /* 流式token回调(P0-001修复) */
        this.onDialogueResponse = null;    /* 完整响应回调 */

        this.onVoiceInputStart = null;
        this.onVoiceInputStop = null;
        this.onVoiceInputResult = null;
        this.onVoiceOutputStart = null;
        this.onVoiceOutputStop = null;
        this.onVoiceOutputError = null;
        this.onVoiceInputError = null;   /* BUG-3修复: 初始化语音输入错误回调 */
    }

    get isRecording() { return this._capturer ? this._capturer.isRecording : false; }

    async startVoiceInput(micStream) {
        /* P1-F17修复: 防御性this上下文检查 */
        if (!(this instanceof DialogueEnhanced)) {
            if (window.g_dialogueEnhanced) return window.g_dialogueEnhanced.startVoiceInput(micStream);
            return { success: false, error: 'DialogueEnhanced实例不可用' };
        }
        if (this.isRecording) return;
        if (this._capturer) { this._capturer.destroy(); this._capturer = null; }
        try {
            if (!micStream) throw new Error('麦克风流不可用');
            var result = await window.VoiceCaptureUtil.quickCapture({
                stream: micStream,
                onStart: function() {
                    if (this.onVoiceInputStart) this.onVoiceInputStart();
                }.bind(this),
                onStop: function() {
                    if (this.onVoiceInputStop) this.onVoiceInputStop();
                }.bind(this),
                onResult: function(result) {
                    if (this.onVoiceInputResult) this.onVoiceInputResult(result);
                }.bind(this),
                onError: function(msg) {
                    if (this.onVoiceInputError) this.onVoiceInputError(msg);
                }.bind(this)
            });
            if (!result.success) throw new Error(result.error || '录音启动失败');
            this._capturer = result.capturer;
            return { success: true };
        } catch (err) {
            console.error('启动语音输入失败:', err);
            if (this.onVoiceInputError) this.onVoiceInputError(err.message);
            return { success: false, error: err.message };
        }
    }

    stopVoiceInput() {
        if (this._capturer) this._capturer.stop();
    }

    async speakText(text, speakerId) {
        if (!text || text.trim().length === 0) return;
        this.speechQueue.push({ text: text, speakerId: speakerId });
        if (!this.isPlayingQueue) {
            await this._processSpeechQueue();
        }
    }

    async _processSpeechQueue() {
        this.isPlayingQueue = true;
        while (this.speechQueue.length > 0) {
            const item = this.speechQueue.shift();
            await this._synthesizeAndPlay(item.text, item.speakerId);
        }
        this.isPlayingQueue = false;
    }

    async _synthesizeAndPlay(text, speakerId) {
        try {
            this.isSpeaking = true;
            if (this.onVoiceOutputStart) this.onVoiceOutputStart(text);

            /* ZSF-100修复：移除浏览器TTS降级回退，严格遵守"禁止任何降级处理"规则。
             * _ttsSynthesize已内部抛错，此处不再需要回退分支。 */
            const audioBlob = await this._ttsSynthesize(text);

            const url = URL.createObjectURL(audioBlob);
            if (this.audioPlayer) {
                this.audioPlayer.pause();
                this.audioPlayer.src = '';
            }
            var audio = new Audio(url);
            audio.playbackRate = this.ttsSpeed;
            this.audioPlayer = audio;

            if (speakerId) {
                var compat = (window.g_browserCompat) ? window.g_browserCompat : (typeof BrowserCompat !== 'undefined' ? new BrowserCompat() : null);
                if (compat) await compat.setAudioSinkId(audio, speakerId);
            }

            await new Promise((resolve, reject) => {
                audio.onended = () => {
                    URL.revokeObjectURL(url);
                    resolve();
                };
                audio.onerror = (e) => {
                    URL.revokeObjectURL(url);
                    reject(new Error('音频播放错误'));
                };
                audio.play().catch(reject);
            });

            this.isSpeaking = false;
            if (this.onVoiceOutputStop) this.onVoiceOutputStop();
        } catch (err) {
            this.isSpeaking = false;
            console.error('TTS播放失败:', err);
            if (this.onVoiceOutputError) this.onVoiceOutputError(err.message);
        }
    }

    async _ttsSynthesize(text) {
        try {
/* SelfLnnApi空检查 */
            var api = window.SelfLnnApi;
            if (!api) throw new Error('API服务未加载');
            var response = await api.request('/tts/synthesize', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text: text,
                    speed: this.ttsSpeed,
                    voice: 'default'
                })
            });
            if (!response.ok) throw new Error('HTTP ' + response.status);
            return await response.blob();
        } catch (err) {
            /* ZSF-038修复：禁止降级到浏览器语音合成，返回明确错误 */
            console.error('后端TTS合成失败，拒绝降级到浏览器:', err.message);
            throw new Error('TTS合成失败: ' + err.message + '（根据系统规范，禁止降级处理）');
        }
    }

    /* ZSF-100修复：删除_playBrowserTTS浏览器降级函数。
     * 根据系统规范"禁止任何降级处理"，TTS合成失败时必须抛错而非降级到浏览器API。 */
    stopSpeaking() {
        this.speechQueue = [];
        if (this.audioPlayer) {
            this.audioPlayer.pause();
            this.audioPlayer.src = '';
            this.audioPlayer = null;
        }
        /* P1-F03修复: 根据"禁止任何降级处理"规范，移除speechSynthesis浏览器TTS残留调用 */
        this.isSpeaking = false;
        this.isPlayingQueue = false;
    }

    pauseSpeaking() {
        if (this.audioPlayer && !this.audioPlayer.paused) {
            this.audioPlayer.pause();
        }
        /* P1-F03修复: 移除speechSynthesis.pause残留 */
    }

    resumeSpeaking() {
        if (this.audioPlayer && this.audioPlayer.paused) {
            this.audioPlayer.play();
        }
        /* P2-15修复: 移除speechSynthesis.resume残留调用（遵循禁止浏览器TTS降级规范） */
    }

    setTtsSpeed(speed) {
        this.ttsSpeed = Math.max(0.5, Math.min(2.0, speed));
    }

    async captureCameraImage(cameraId) {
        /* FIX-FRONTEND-004: camera-preview ID映射 */
        var videoEl = document.getElementById('camera-preview-' + cameraId);
        if (!videoEl) videoEl = document.getElementById('camera-preview-fallback');
        if (!videoEl) { var allVids = document.querySelectorAll('[id^="camera-preview-"]'); if (allVids.length) videoEl = allVids[0]; }
        if (!videoEl) return null;
        const canvas = document.createElement('canvas');
        canvas.width = videoEl.videoWidth || 640;
        canvas.height = videoEl.videoHeight || 480;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(videoEl, 0, 0);
        const dataUrl = canvas.toDataURL('image/jpeg', 0.8);
        canvas.remove();
        this.lastCapturedImage = dataUrl;
        return dataUrl;
    }

    async captureDualCameraImages(leftCameraId, rightCameraId) {
        const leftImage = await this.captureCameraImage(leftCameraId);
        const rightImage = await this.captureCameraImage(rightCameraId);
        if (leftImage && rightImage) {
            return { left: leftImage, right: rightImage };
        }
        return null;
    }

    enableVoiceInput() { this.voiceInputEnabled = true; }
    disableVoiceInput() { this.voiceInputEnabled = false; }
    enableVoiceOutput() { this.voiceOutputEnabled = true; }
    disableVoiceOutput() { this.voiceOutputEnabled = false; }
    enableMultimodal() { this.multimodalEnabled = true; }
    disableMultimodal() { this.multimodalEnabled = false; }

    // ================================================================
    // A4.4 对话历史管理
    // ================================================================

    /**
     * 添加对话历史条目
     * @param {string} role - 'user', 'assistant', 'system'
     * @param {string} text - 对话文本
     * @param {object} options - 附加选项
     * @param {string} options.audioUrl - 语音文件URL（如果有）
     * @param {string} options.imageData - 图像数据（如果有）
     * @param {string} options.metadata - 额外元数据
     */
    addHistoryEntry(role, text, options) {
        if (!this.dialogueHistory) {
            this.dialogueHistory = [];
        }
        const entry = {
            role: role,
            text: text,
            timestamp: Date.now(),
            audioUrl: options && options.audioUrl ? options.audioUrl : null,
            imageData: options && options.imageData ? options.imageData : null,
            metadata: options && options.metadata ? options.metadata : null
        };
        this.dialogueHistory.push(entry);

        /* F-严重修复: 对话历史持久化到后端
         * 使用 /dialogue/history 端点进行历史记录的持久化存储。
         * P2-18修复: 统一使用不带/api/前缀的路径，由api-service.js自动补全。
         * 检查是否有专门的历史追加API，若无则仍使用主dialogue端点。 */
        if (window.SelfLnnApi && window.SelfLnnApi.connected) {
            try {
                /* 优先使用专用的历史记录API端点 */
                window.SelfLnnApi.request('/dialogue/history/save', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        role: role,
                        text: text,
                        timestamp: entry.timestamp
                    })
                }).catch(function(err) {
                    console.warn('对话历史保存失败:', err);
                    if (typeof window.showNotification === 'function') {
                        window.showNotification('对话历史保存失败', 'warning');
                    }
                });
            } catch(e) {} /* 忽略持久化异常 */
        }
        return entry;
    }

    /**
     * 获取对话历史
     */
    getHistory() {
        return this.dialogueHistory || [];
    }

    /**
     * 获取对话历史中所有音频链接列表
     */
    getAudioLinks() {
        if (!this.dialogueHistory) return [];
        return this.dialogueHistory
            .filter(function(e) { return e.audioUrl !== null; })
            .map(function(e) {
                return {
                    role: e.role,
                    text: e.text,
                    audioUrl: e.audioUrl,
                    timestamp: e.timestamp
                };
            });
    }

    /**
     * 导出对话历史为JSON格式
     */
    exportHistoryAsJson() {
        if (!this.dialogueHistory || this.dialogueHistory.length === 0) {
            return JSON.stringify({ entries: [], exportTime: Date.now(), totalEntries: 0 }, null, 2);
        }
        const exportData = {
            exportTime: Date.now(),
            totalEntries: this.dialogueHistory.length,
            entries: this.dialogueHistory.map(function(e) {
                var entry = {
                    role: e.role,
                    text: e.text,
                    timestamp: e.timestamp
                };
                if (e.audioUrl) entry.hasAudio = true;
                if (e.imageData) entry.hasImage = true;
                if (e.metadata) entry.metadata = e.metadata;
                return entry;
            })
        };
        return JSON.stringify(exportData, null, 2);
    }

    /**
     * 导出对话历史为纯文本格式
     */
    exportHistoryAsText() {
        if (!this.dialogueHistory || this.dialogueHistory.length === 0) {
            return '暂无对话记录\n';
        }
        var lines = [];
        lines.push('SELF-LNN AGI 对话记录');
        lines.push('导出时间: ' + new Date().toLocaleString('zh-CN'));
        lines.push('对话条目数: ' + this.dialogueHistory.length);
        lines.push('========================================');
        lines.push('');
        this.dialogueHistory.forEach(function(e) {
            var roleLabel = '';
            if (e.role === 'user') roleLabel = '用户';
            else if (e.role === 'assistant') roleLabel = 'AGI';
            else roleLabel = '系统';
            var timeStr = new Date(e.timestamp).toLocaleTimeString('zh-CN');
            var line = '[' + timeStr + '] ' + roleLabel + ': ' + e.text;
            if (e.audioUrl) line += ' [含语音]';
            if (e.imageData) line += ' [含图像]';
            lines.push(line);
        });
        lines.push('');
        lines.push('========================================');
        return lines.join('\n');
    }

    /**
     * 下载对话历史
     */
    downloadHistory(format) {
        var content = '';
        var mimeType = '';
        var extension = '';
        if (format === 'json') {
            content = this.exportHistoryAsJson();
            mimeType = 'application/json';
            extension = 'json';
        } else {
            content = this.exportHistoryAsText();
            mimeType = 'text/plain;charset=utf-8';
            extension = 'txt';
        }
        var blob = new Blob([content], { type: mimeType });
        var url = URL.createObjectURL(blob);
        var a = document.createElement('a');
        a.href = url;
        a.download = '对话历史_' + new Date().toISOString().slice(0, 19).replace(/[:-]/g, '') + '.' + extension;
        document.body.appendChild(a);
        a.click();
        setTimeout(function() {
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }, 100);
    }

    /**
     * 清空对话历史
     */
    clearHistory() {
        var count = this.dialogueHistory ? this.dialogueHistory.length : 0;
        this.dialogueHistory = [];
        /* F-严重修复: 清空后端对话历史 */
        /* P2-18修复: 统一使用不带/api/前缀的路径，由api-service.js自动补全 */
        if (window.SelfLnnApi && window.SelfLnnApi.connected) {
            try {
                window.SelfLnnApi.request('/dialogue/clear', { method: 'POST' })
                    .catch(function(err) {
                        console.warn('对话历史清空失败:', err);
                        if (typeof window.showNotification === 'function') {
                            window.showNotification('对话历史清空失败', 'warning');
                        }
                    });
            } catch(e) {}
        }
        return count;
    }

    // ================================================================
    // A5 实时流式通信（WebSocket 对话推送）
    // ================================================================

    /**
     * 连接WebSocket流式推送服务器（使用全局SelfLnnWebSocket避免重复连接）
     * @param {string} wsUrl - WebSocket服务器地址，默认 ws://host:port/ws
     * @returns {boolean} 是否成功启动连接
     */
    connectWebSocket(wsUrl) {
        var gws = window.SelfLnnWebSocket;
        if (!gws) {
            console.error('DialogueEnhanced: 全局SelfLnnWebSocket不可用');
            return false;
        }
        if (gws.isConnected) {
            return true;
        }
        if (!wsUrl) {
            /* ZSFOOO-001修复: WebSocket使用独立端口8081，与port_config.h中SELFLNN_WEBSOCKET_PORT一致 */
            var cfg2 = window.SELFLNN_CONFIG || { host: 'localhost', wsPort: 8081 };
            var host = cfg2.host || 'localhost';
            var wport = cfg2.wsPort || (cfg2.port ? (cfg2.port + 1) : 8081);
            wsUrl = 'ws://' + host + ':' + wport + '/ws';
        }
        this.wsUrl = wsUrl;
        this.wsReconnectAttempts = 0;
        this.wsMaxReconnect = 5;
/* 使用addEventListener避免覆盖其他模块的close处理器 */
        var self = this;
        if (gws.ws) {
            var wsElement = gws.ws;
/* 将onCloseHandler存储在实例上，确保removeEventListener能匹配到同一个函数引用 */
            if (this._wsCloseHandler) {
                try { wsElement.removeEventListener('close', this._wsCloseHandler); } catch(e) {}
            }
/* WebSocket断开时自动重连（之前仅计数未重连） */
            this._wsCloseHandler = function(evt) {
                self.wsReconnectAttempts++;
                if (self.wsReconnectAttempts > self.wsMaxReconnect) {
                    console.error('[SELF-LNN] WebSocket重连已达上限(' + self.wsMaxReconnect + '次)，停止重连。');
                    if (self.onWsStatusChange) self.onWsStatusChange('disconnected');
                    return;
                }
                var delay = Math.min(1000 * Math.pow(2, self.wsReconnectAttempts - 1), 30000);
                console.warn('[SELF-LNN] WebSocket断开，第' + self.wsReconnectAttempts + '/' + self.wsMaxReconnect + '次重连，延迟' + delay + 'ms...');
                if (self.onWsStatusChange) self.onWsStatusChange('reconnecting');
                /* FE-009修复: 重连时传递this.url，确保WS使用正确的URL而非默认值 */
                setTimeout(function() {
                    try { gws.connect(self.wsUrl || self.url); } catch(er) { console.warn('[SELF-LNN] WS重连失败:', er); }
                }, delay);
            };
            wsElement.addEventListener('close', this._wsCloseHandler);
        }
        /* 使用全局SelfLnnWebSocket的connect方法，设置URL后连接 */
        /* ZSFOOO-001修复: 动态获取WebSocket端口，与port_config.h中SELFLNN_WEBSOCKET_PORT=8081一致 */
        var cfg = window.SELFLNN_CONFIG || { host: 'localhost', wsPort: 8081 };
        var wport2 = cfg.wsPort || (cfg.port ? (cfg.port + 1) : 8081);
        var defaultUrl = 'ws://' + cfg.host + ':' + wport2 + '/ws';
        if (wsUrl && gws.url !== wsUrl) {
            gws.url = wsUrl;
        } else if (!gws.url) {
            gws.url = defaultUrl;
        }
        gws.connect();
        return true;
    }

    /**
     * 断开WebSocket连接（使用全局SelfLnnWebSocket）
     */
    disconnectWebSocket() {
        var gws = window.SelfLnnWebSocket;
        if (gws && gws.ws && gws.ws.readyState === WebSocket.OPEN) {
            gws.ws.close();
        }
        if (this.onWsStatusChange) this.onWsStatusChange('disconnected');
    }

    /**
     * WebSocket连接状态查询
     * @returns {string} 'connected' | 'connecting' | 'disconnected' | 'error'
     */
    getWsStatus() {
        var gws = window.SelfLnnWebSocket;
        if (!gws || !gws.ws) return 'disconnected';
        switch (gws.ws.readyState) {
            case WebSocket.OPEN: return 'connected';
            case WebSocket.CONNECTING: return 'connecting';
            default: return 'disconnected';
        }
    }

    /**
     * 增强多模态消息发送（支持参数和流式回调）
     * @param {string} text - 文本消息
     * @param {string} imageData - 图像数据（base64）
     * @param {string} audioData - 音频数据标记
     * @param {object} params - 附加参数 { temperature, max_length, top_k, memory }
     * @returns {Promise<object>} 响应结果
     */
    async sendMultimodalMessage(text, imageData, audioData, params) {
        if (!window.SelfLnnApi || !window.SelfLnnApi.connected) {
            return { success: false, error: '会话系统未就绪，请先完成模型训练或检查后端连接。' };
        }
        var payload = {
            message: text || '',
            mode: 'multimodal'
        };
        if (imageData) payload.image = imageData;
        if (audioData) payload.audio = audioData;
        if (params) {
            /* P1-14修复: temperature后端期望整数(0-20)，需乘10取整转换 */
            if (params.temperature !== undefined) payload.temperature = Math.round(params.temperature * 10);
            if (params.max_length !== undefined) payload.max_length = params.max_length;
            if (params.top_k !== undefined) payload.top_k = params.top_k;
            if (params.memory !== undefined) payload.memory = params.memory;
        }

        var isStreaming = params && params.streaming === true;

        if (isStreaming && this.getWsStatus() === 'connected') {
            try {
                var wsPayload = JSON.stringify({
                    type: 'dialogue_request',
                    data: payload
                });
                var gws = window.SelfLnnWebSocket;
                if (gws && gws.send) gws.send(wsPayload);
                return { success: true, streaming: true };
            } catch (err) {
                console.warn('WebSocket发送失败，回退到HTTP:', err);
            }
        }

        try {
            var response = await window.SelfLnnApi.request('/dialogue/multimodal', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            if (!response.ok) throw new Error('HTTP ' + response.status);
            return await response.json();
        } catch (err) {
            console.error('多模态对话失败:', err);
            return { success: false, error: '会话系统未就绪，请先完成模型训练或检查后端连接。' };
        }
    }

    destroy() {
        this.stopSpeaking();
        if (this._capturer) {
            this._capturer.destroy();
            this._capturer = null;
        }
        this.speechQueue = [];
        this.audioPlayer = null;
        this.lastCapturedImage = null;
    }
}

window.DialogueEnhanced = DialogueEnhanced;
