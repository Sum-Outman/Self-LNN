/**
 * SELF-LNN AGI 增强对话系统
 * 支持语音输入/输出、多模态对话（视觉+语音+文字）、摄像头画面发送
 * 集成TTS语音播报和语音识别录制
 */

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
        if (this.isRecording) return;
        if (this._capturer) { this._capturer.destroy(); this._capturer = null; }
        try {
            if (!micStream) throw new Error('麦克风流不可用');
            this._capturer = new VoiceCaptureUtil();
            this._capturer.onStart = function() {
                if (this.onVoiceInputStart) this.onVoiceInputStart();
            }.bind(this);
            this._capturer.onStop = function() {
                if (this.onVoiceInputStop) this.onVoiceInputStop();
            }.bind(this);
            this._capturer.onBlobReady = async function(blob) {
                var result = await VoiceCaptureUtil.uploadBlob(blob);
                if (this.onVoiceInputResult) this.onVoiceInputResult(result);
            }.bind(this);
            return this._capturer.start(micStream);
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

            const audioBlob = await this._ttsSynthesize(text);
            if (!audioBlob) {
                await this._playBrowserTTS(text);
                this.isSpeaking = false;
                if (this.onVoiceOutputStop) this.onVoiceOutputStop();
                return;
            }

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
            var response = await SelfLnnApi.request('/tts/synthesize', {
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
            console.warn('后端TTS不可用，使用浏览器语音合成:', err.message);
            return null;
        }
    }

    async _playBrowserTTS(text) {
        if (!window.speechSynthesis) {
            throw new Error('浏览器语音合成不可用');
        }
        var self = this;
        return new Promise(function(resolve, reject) {
            var utterance = new SpeechSynthesisUtterance(text);
            var compat = (window.g_browserCompat) ? window.g_browserCompat : (typeof BrowserCompat !== 'undefined' ? new BrowserCompat() : null);
            var voice = compat ? compat.getSpeechSynthesisVoice('zh-CN') : null;
            if (voice) utterance.voice = voice;
            utterance.lang = 'zh-CN';
            utterance.rate = self.ttsSpeed;
            utterance.volume = 1.0;
            utterance.onend = function() { resolve(); };
            utterance.onerror = function(e) { reject(new Error('语音合成错误: ' + e.error)); };
            window.speechSynthesis.speak(utterance);
        });
    }

    stopSpeaking() {
        this.speechQueue = [];
        if (this.audioPlayer) {
            this.audioPlayer.pause();
            this.audioPlayer.src = '';
            this.audioPlayer = null;
        }
        if (window.speechSynthesis) {
            window.speechSynthesis.cancel();
        }
        this.isSpeaking = false;
        this.isPlayingQueue = false;
    }

    pauseSpeaking() {
        if (this.audioPlayer && !this.audioPlayer.paused) {
            this.audioPlayer.pause();
        }
        if (window.speechSynthesis && window.speechSynthesis.speaking) {
            window.speechSynthesis.pause();
        }
    }

    resumeSpeaking() {
        if (this.audioPlayer && this.audioPlayer.paused) {
            this.audioPlayer.play();
        }
        if (window.speechSynthesis && window.speechSynthesis.paused) {
            window.speechSynthesis.resume();
        }
    }

    setTtsSpeed(speed) {
        this.ttsSpeed = Math.max(0.5, Math.min(2.0, speed));
    }

    async captureCameraImage(cameraId) {
        const videoEl = document.getElementById('camera-preview-' + cameraId);
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
            var host = (window.SELFLNN_CONFIG && window.SELFLNN_CONFIG.host) || 'localhost';
            var port = (window.SELFLNN_CONFIG && window.SELFLNN_CONFIG.port) || 8080;
            wsUrl = 'ws://' + host + ':' + port + '/ws';
        }
        this.wsUrl = wsUrl;
        this.wsReconnectAttempts = 0;
        this.wsMaxReconnect = 5;
        /* ZSFWS-M025修复: 监听WebSocket事件，实现真实重连计数和上限 */
        var self = this;
        if (gws.ws) {
            var existingClose = gws.ws.onclose;
            gws.ws.onclose = function(evt) {
                self.wsReconnectAttempts++;
                if (self.wsReconnectAttempts > self.wsMaxReconnect) {
                    console.error('[SELF-LNN] WebSocket重连已达上限(' + self.wsMaxReconnect + '次)，停止重连。请检查后端服务。');
                    if (self.onWsStatusChange) self.onWsStatusChange('disconnected');
                    return;
                }
                console.warn('[SELF-LNN] WebSocket断开，第' + self.wsReconnectAttempts + '/' + self.wsMaxReconnect + '次重连...');
                if (typeof existingClose === 'function') existingClose.call(gws.ws, evt);
            };
        }
        /* 使用全局SelfLnnWebSocket的connect方法 */
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
        var payload = {
            message: text || '',
            mode: 'multimodal'
        };
        if (imageData) payload.image = imageData;
        if (audioData) payload.audio = audioData;
        if (params) {
            if (params.temperature !== undefined) payload.temperature = params.temperature;
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
            var response = await SelfLnnApi.request('/dialogue/multimodal', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            if (!response.ok) throw new Error('HTTP ' + response.status);
            return await response.json();
        } catch (err) {
            console.error('多模态对话失败:', err);
            return { success: false, error: err.message };
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
