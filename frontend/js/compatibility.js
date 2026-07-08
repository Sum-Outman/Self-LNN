/**
 * SELF-LNN AGI 多系统兼容性层
 * 处理Chrome/Edge/Firefox/Safari的WebRTC API差异
 * 移动端兼容性 + 后端平台差异适配
 */

'use strict';

class BrowserCompat {
    constructor() {
        this.ua = navigator.userAgent;
        this.browser = this._detectBrowser();
        this.os = this._detectOS();
        this.isMobile = /Mobi|Android|iPhone|iPad|iPod/i.test(this.ua);
        this.isTouchDevice = 'ontouchstart' in window || navigator.maxTouchPoints > 0;
        this.features = this._checkFeatures();
    }

    _detectBrowser() {
        const ua = this.ua.toLowerCase();
        if (ua.indexOf('edge') > -1 || ua.indexOf('edg/') > -1) return 'edge';
        if (ua.indexOf('chrome') > -1 && ua.indexOf('chromium') === -1) return 'chrome';
        if (ua.indexOf('firefox') > -1) return 'firefox';
        if (ua.indexOf('safari') > -1) return 'safari';
        if (ua.indexOf('opera') > -1 || ua.indexOf('opr') > -1) return 'opera';
        return 'unknown';
    }

    _detectOS() {
        const ua = this.ua.toLowerCase();
        if (ua.indexOf('win') > -1) return 'windows';
        if (ua.indexOf('mac') > -1) return 'macos';
        if (ua.indexOf('linux') > -1) return 'linux';
        if (ua.indexOf('android') > -1) return 'android';
        if (ua.indexOf('iphone') > -1 || ua.indexOf('ipad') > -1) return 'ios';
        return 'unknown';
    }

    _checkFeatures() {
        return {
            getUserMedia: !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia),
            enumerateDevices: !!(navigator.mediaDevices && navigator.mediaDevices.enumerateDevices),
            setSinkId: !!(HTMLAudioElement.prototype.setSinkId),
            mediaRecorder: !!window.MediaRecorder,
            webAudio: !!(window.AudioContext || window.webkitAudioContext),
            speechSynthesis: !!window.speechSynthesis,
            webRTC: !!(window.RTCPeerConnection || window.webkitRTCPeerConnection || window.mozRTCPeerConnection),
            webSocket: !!window.WebSocket,
            canvas: !!document.createElement('canvas').getContext,
            webGL: this._checkWebGL(),
            sharedWorker: !!window.SharedWorker,
            serviceWorker: 'serviceWorker' in navigator,
            /* FE-022修复: 添加ResizeObserver、IntersectionObserver、AbortController检测 */
            resizeObserver: !!window.ResizeObserver,
            intersectionObserver: !!window.IntersectionObserver,
            abortController: !!window.AbortController
        };
    }

    _checkWebGL() {
        try {
            var canvas = document.createElement('canvas');
            var result = !!(canvas.getContext('webgl') || canvas.getContext('experimental-webgl'));
            /* L-009修复: 显式释放临时Canvas引用，提示GC回收 */
            canvas = null;
            return result;
        } catch (e) {
            return false;
        }
    }

    getAudioContext() {
        var AC = window.AudioContext || window.webkitAudioContext;
        if (!AC) return null;
        return new AC();
    }

    getSupportedMediaRecorderMimeType() {
        /* P2-003修复: 检查MediaRecorder是否存在，避免在不支持的浏览器崩溃 */
        if (typeof MediaRecorder === 'undefined' || !MediaRecorder.isTypeSupported) return '';
        var types = [
            'audio/webm;codecs=opus',
            'audio/webm',
            'audio/ogg;codecs=opus',
            'audio/wav',
            'audio/mp4',
            'audio/mpeg'
        ];
        for (var i = 0; i < types.length; i++) {
            if (MediaRecorder.isTypeSupported(types[i])) {
                return types[i];
            }
        }
        return '';
    }

    getSupportedVideoMimeType() {
        var types = [
            'video/webm;codecs=vp9,opus',
            'video/webm;codecs=vp8,opus',
            'video/webm',
            'video/mp4'
        ];
        for (var i = 0; i < types.length; i++) {
            if (MediaRecorder.isTypeSupported(types[i])) {
                return types[i];
            }
        }
        return 'video/webm';
    }

    async getDisplayMedia(constraints) {
        if (navigator.mediaDevices && navigator.mediaDevices.getDisplayMedia) {
            return await navigator.mediaDevices.getDisplayMedia(constraints || { video: true });
        }
        throw new Error('getDisplayMedia API不可用，浏览器不支持屏幕共享');
    }

    async setAudioSinkId(audioElement, sinkId) {
        if (this.features.setSinkId && this.browser !== 'firefox') {
            try {
                await audioElement.setSinkId(sinkId);
                return true;
            } catch (e) {
                console.warn('setSinkId失败:', e.message);
                return false;
            }
        }
        if (this.browser === 'firefox') {
            console.warn('Firefox不支持setSinkId，使用默认扬声器');
        }
        return false;
    }

    async enumerateMediaDevices() {
        if (!this.features.enumerateDevices) {
            console.warn('enumerateDevices不可用，尝试降级处理（仅音频）');
            try {
                var stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                var devices = [];
                if (stream.getAudioTracks().length > 0) {
                    devices.push({ kind: 'audioinput', label: '默认麦克风', deviceId: 'default' });
                }
                if (stream.getVideoTracks().length > 0) {
                    devices.push({ kind: 'videoinput', label: '默认摄像头', deviceId: 'default' });
                }
                stream.getTracks().forEach(function(t) { t.stop(); });
                return devices;
            } catch (e) {
                return [];
            }
        }
        try {
            var allDevices = await navigator.mediaDevices.enumerateDevices();
            return allDevices;
        } catch (e) {
            console.warn('enumerateDevices失败:', e.message);
            return [];
        }
    }

    async getUserMediaWithFallback(constraints) {
        if (!this.features.getUserMedia) {
            throw new Error('浏览器不支持getUserMedia API');
        }
        try {
            return await navigator.mediaDevices.getUserMedia(constraints);
        } catch (err) {
            if (err.name === 'NotAllowedError') {
                throw new Error('设备访问被拒绝，请在浏览器设置中允许摄像头/麦克风权限');
            }
            if (err.name === 'NotFoundError') {
                throw new Error('未找到请求的设备');
            }
            if (err.name === 'NotReadableError' && this.browser === 'firefox') {
                console.warn('Firefox设备忙，尝试重新获取...');
                await new Promise(function(resolve) { setTimeout(resolve, 500); });
                return await navigator.mediaDevices.getUserMedia(constraints);
            }
            if (err.name === 'OverconstrainedError') {
                var relaxed = JSON.parse(JSON.stringify(constraints));
                if (relaxed.video && relaxed.video.width) {
                    delete relaxed.video.width;
                    delete relaxed.video.height;
                }
                if (relaxed.audio && relaxed.audio.sampleRate) {
                    delete relaxed.audio.sampleRate;
                }
                return await navigator.mediaDevices.getUserMedia(relaxed);
            }
            throw err;
        }
    }

    getSpeechSynthesisVoice(lang) {
        if (!this.features.speechSynthesis) return null;
        var voices = window.speechSynthesis.getVoices();
        if (voices.length === 0) {
            voices = window.speechSynthesis.getVoices();
        }
        /* 如果voices仍为空（某些浏览器异步加载），注册voiceschanged事件重试 */
        if (voices.length === 0) {
            var self = this;
            var retryHandler = function() {
                window.speechSynthesis.removeEventListener('voiceschanged', retryHandler);
                var retryVoices = window.speechSynthesis.getVoices();
                for (var i = 0; i < retryVoices.length; i++) {
                    if (retryVoices[i].lang.indexOf(lang) >= 0) {
                        self._cachedVoice = retryVoices[i];
                        return;
                    }
                }
            };
            window.speechSynthesis.addEventListener('voiceschanged', retryHandler);
            setTimeout(function() {
                window.speechSynthesis.removeEventListener('voiceschanged', retryHandler);
            }, 5000);
            return null;
        }
        lang = lang || 'zh-CN';
        for (var i = 0; i < voices.length; i++) {
            if (voices[i].lang.indexOf(lang) >= 0) return voices[i];
        }
        for (var i = 0; i < voices.length; i++) {
            if (voices[i].lang.indexOf('zh') >= 0) return voices[i];
        }
        return voices.length > 0 ? voices[0] : null;
    }

    getPlatformInfo() {
        return {
            browser: this.browser,
            os: this.os,
            isMobile: this.isMobile,
            isTouchDevice: this.isTouchDevice,
            features: this.features,
            userAgent: this.ua
        };
    }

    getVideoConstraintsForResolution(resolution) {
        var resolutions = {
            '480p': { width: { exact: 640 }, height: { exact: 480 } },
            '720p': { width: { exact: 1280 }, height: { exact: 720 } },
            '1080p': { width: { exact: 1920 }, height: { exact: 1080 } },
            '4k': { width: { exact: 3840 }, height: { exact: 2160 } }
        };
        var res = resolutions[resolution];
        if (!res) return { width: { ideal: 1280 }, height: { ideal: 720 } };
        if (this.isMobile) {
            return { width: { ideal: res.width.exact / 2 }, height: { ideal: res.height.exact / 2 } };
        }
        return res;
    }

    isFeatureSupported(featureName) {
        return this.features[featureName] === true;
    }

    getUnsupportedFeatures() {
        var unsupported = [];
        var requiredFeatures = ['getUserMedia', 'enumerateDevices', 'mediaRecorder', 'webAudio', 'webSocket', 'canvas'];
        for (var i = 0; i < requiredFeatures.length; i++) {
            if (!this.features[requiredFeatures[i]]) {
                unsupported.push(requiredFeatures[i]);
            }
        }
        return unsupported;
    }

    getDefaultAudioConstraints() {
        var constraints = { audio: true };
        if (this.browser === 'safari') {
            constraints.audio = {
                sampleRate: 44100,
                echoCancellation: true,
                noiseSuppression: true
            };
        } else if (this.browser === 'firefox') {
            constraints.audio = {
                sampleRate: 16000,
                echoCancellation: true,
                noiseSuppression: true,
                channelCount: 1
            };
        } else {
            constraints.audio = {
                sampleRate: { ideal: 16000 },
                echoCancellation: true,
                noiseSuppression: true,
                channelCount: 1
            };
        }
        return constraints;
    }
}

window.BrowserCompat = BrowserCompat;
window.g_browserCompat = new BrowserCompat();
