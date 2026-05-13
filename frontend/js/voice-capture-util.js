/**
 * SELF-LNN AGI 语音采集共享工具
 * 统一管理 MediaRecorder 生命周期、音频采集、Blob组装
 * 消除 voice-command.js / dialogue-enhanced.js / voice-control.html 中的三重重复实现
 */
class VoiceCaptureUtil {
    constructor(options) {
        options = options || {};
        this.maxDuration = options.maxDuration || 15000;
        this.sampleRate = options.sampleRate || 16000;
        this.onBlobReady = null;
        this.onStart = null;
        this.onStop = null;
        this.onProgress = null;
        this.onError = null;
        this._recorder = null;
        this._chunks = [];
        this._stream = null;
        this._timer = null;
        this._startTime = 0;
        this._recording = false;
        this._aborted = false;
    }

    get isRecording() { return this._recording; }

    async start(stream) {
        if (this._recording) return;
        try {
            if (!stream) throw new Error('未提供麦克风流');
            var compat = window.g_browserCompat || new BrowserCompat();
            var mimeType = compat.getSupportedMediaRecorderMimeType();
            this._chunks = [];
            this._stream = stream;
            this._recorder = new MediaRecorder(stream, {
                mimeType: mimeType || 'audio/webm',
                audioBitsPerSecond: this.sampleRate
            });
            this._recorder.ondataavailable = function(e) {
                if (e.data.size > 0) this._chunks.push(e.data);
            }.bind(this);
            this._recorder.onstop = function() {
                this._onRecordingComplete();
            }.bind(this);
            this._recorder.onerror = function(err) {
                if (this.onError) this.onError('录音错误: ' + (err.message || '未知'));
            }.bind(this);
            this._recorder.start(100);
            this._recording = true;
            this._startTime = Date.now();
            this._timer = setInterval(function() {
                var elapsed = Date.now() - this._startTime;
                if (this.onProgress) this.onProgress(elapsed);
                if (elapsed >= this.maxDuration) this.stop();
            }.bind(this), 100);
            if (this.onStart) this.onStart();
            return { success: true };
        } catch (err) {
            if (this.onError) this.onError('启动录音失败: ' + err.message);
            return { success: false, error: err.message };
        }
    }

    stop() {
        if (!this._recording || !this._recorder) return;
        try {
            if (this._recorder.state !== 'inactive') this._recorder.stop();
        } catch (e) { /* ignore */ }
        this._cleanup();
        if (this.onStop) this.onStop();
    }

    abort() {
        if (!this._recording || !this._recorder) return;
        this._aborted = true;
        try {
            if (this._recorder.state !== 'inactive') this._recorder.stop();
        } catch (e) { /* ignore */ }
        this._chunks = [];
        this._cleanup();
    }

    _cleanup() {
        this._recording = false;
        if (this._timer) { clearInterval(this._timer); this._timer = null; }
    }

    _onRecordingComplete() {
        this._cleanup();
        if (this._aborted) { this._aborted = false; return; }
        if (this._chunks.length === 0) {
            if (this.onError) this.onError('未采集到音频数据');
            return;
        }
        var compat = window.g_browserCompat || new BrowserCompat();
        var mimeType = compat.getSupportedMediaRecorderMimeType();
        var blob = new Blob(this._chunks, { type: mimeType || 'audio/webm' });
        this._chunks = [];
        if (this.onBlobReady) this.onBlobReady(blob);
    }

    destroy() {
        this.abort();
        this._aborted = false;
        this.onBlobReady = null;
        this.onStart = null;
        this.onStop = null;
        this.onProgress = null;
        this.onError = null;
        if (this._stream) {
            this._stream.getTracks().forEach(function(t) { t.stop(); });
            this._stream = null;
        }
    }

    static async uploadBlob(blob) {
        if (!window.SelfLnnApi) return { success: false, error: 'API服务不可用' };
        try {
            var result = await window.SelfLnnApi.voiceRecognize(blob);
            var data = result.data || result;
            return {
                success: true,
                text: data.text || data.result || '',
                /* F-018修复：不使用假默认置信度，未提供时返回-1表示未知 */
                confidence: (data.confidence !== undefined && data.confidence !== null) ? data.confidence : -1
            };
        } catch (err) {
            return { success: false, error: '语音识别失败: ' + err.message, text: '' };
        }
    }

    /**
     * 统一快捷语音采集方法 — 封装 getUserMedia + VoiceCaptureUtil 创建 + 回调绑定
     * 消除 voice-control.html / multimodal-learn.html / dialogue-enhanced.js 中的重复模板代码
     * @param {Object} options - { maxDuration, onStart, onStop, onResult, onError }
     * @returns {Promise<{success:boolean, capturer:VoiceCaptureUtil|null, error:string|null}>}
     */
    static async quickCapture(options) {
        options = options || {};
        var maxDuration = options.maxDuration || 15000;
        try {
            var stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            var capturer = new VoiceCaptureUtil({ maxDuration: maxDuration });

            capturer.onStart = function() {
                if (options.onStart) options.onStart();
            };
            capturer.onStop = function() {
                if (options.onStop) {
                    options.onStop();
                } else {
                    stream.getTracks().forEach(function(t) { t.stop(); });
                }
            };
            capturer.onBlobReady = async function(blob) {
                try {
                    var result = await VoiceCaptureUtil.uploadBlob(blob);
                    if (options.onResult) options.onResult(result);
                } catch (e) {
                    if (options.onError) options.onError('识别失败: ' + e.message);
                }
            };
            capturer.onError = function(msg) {
                if (options.onError) options.onError(msg);
            };

            capturer.start(stream);
            return { success: true, capturer: capturer, stream: stream, error: null };
        } catch (err) {
            return { success: false, capturer: null, stream: null, error: err.message };
        }
    }
}
