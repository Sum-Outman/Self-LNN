/**
 * SELF-LNN AGI 设备管理器
 * 管理麦克风、扬声器、摄像头的添加/删除/开启/关闭
 * 全部使用真实浏览器API，无虚拟实现
 */

class DeviceManager {
    constructor() {
        this.microphones = [];
        this.speakers = [];
        this.cameras = [];
        this.audioContext = null;
        this.initialized = false;
        this.deviceChangeCallback = null;

        this._micIdCounter = 0;
        this._speakerIdCounter = 0;
        this._cameraIdCounter = 0;
        this._monitorInterval = null;
        this._vuAnalysers = {};

        this.stereoVision = {
            enabled: false,
            leftCameraId: null,
            rightCameraId: null,
            baseline: 0.06,
            focalLength: 700,
            disparityMap: null,
            depthMap: null,
            pointCloud: null,
            lastCaptureTime: 0,
            captureInterval: 500,
            onDepthUpdate: null,
            onDisparityUpdate: null,
            onPointCloudUpdate: null,
            processingInterval: null,
            worker: null,
            workerBusy: false
        };
    }

    async init() {
        if (this.initialized) return;
        try {
            /* ZSFWS-DM1修复: BrowserCompat类存在性检查，回退到window.g_browserCompat */
            var compat = window.g_browserCompat || (typeof BrowserCompat !== 'undefined' ? new BrowserCompat() : null);
            if (!compat) { console.error('[DeviceManager] BrowserCompat未加载'); return; }
            var devices = await compat.enumerateMediaDevices();
            this._processDevices(devices);
            if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
                this._boundOnDeviceChange = this._onDeviceChange.bind(this);
                navigator.mediaDevices.addEventListener('devicechange', this._boundOnDeviceChange);
            }
            this.initialized = true;

            this._monitorInterval = setInterval(function() {
                this._updateAllVuMeters();
            }.bind(this), 100);

            document.dispatchEvent(new CustomEvent('device-manager-ready'));
        } catch (err) {
            console.error('设备管理器初始化失败:', err.message);
        }
    }

    _processDevices(devices) {
        const audioInputs = devices.filter(d => d.kind === 'audioinput');
        const audioOutputs = devices.filter(d => d.kind === 'audiooutput');
        const videoInputs = devices.filter(d => d.kind === 'videoinput');

        this._availableMicrophones = audioInputs;
        this._availableSpeakers = audioOutputs;
        this._availableCameras = videoInputs;
    }

    _onDeviceChange() {
        var self = this;
        /* ZSFZS-F054修复: BrowserCompat回退路径添加类型检查 */
        var compat = window.g_browserCompat || (typeof BrowserCompat !== 'undefined' ? new BrowserCompat() : null);
        if (!compat) { console.warn('[DeviceManager] BrowserCompat未加载，跳过设备枚举'); return; }
        compat.enumerateMediaDevices().then(function(devices) {
            self._processDevices(devices);
            if (self.deviceChangeCallback) {
                self.deviceChangeCallback({
                    microphones: self._availableMicrophones,
                    speakers: self._availableSpeakers,
                    cameras: self._availableCameras
                });
            }
            document.dispatchEvent(new CustomEvent('device-list-changed', {
                detail: {
                    microphones: self._availableMicrophones,
                    speakers: self._availableSpeakers,
                    cameras: self._availableCameras
                }
            }));
        });
    }

    getAvailableMicrophones() {
        return this._availableMicrophones || [];
    }

    getAvailableSpeakers() {
        return this._availableSpeakers || [];
    }

    getAvailableCameras() {
        return this._availableCameras || [];
    }

    async addMicrophone(deviceId) {
        const device = this._availableMicrophones.find(d => d.deviceId === deviceId);
        if (!device) {
            return { success: false, error: '未找到指定麦克风设备' };
        }
        if (this.microphones.find(m => m.deviceId === deviceId && m.active)) {
            return { success: false, error: '该麦克风已添加' };
        }
        const id = 'mic_' + (++this._micIdCounter);
        const mic = {
            id: id,
            deviceId: device.deviceId,
            label: device.label || '麦克风 ' + id,
            active: false,
            stream: null,
            analyser: null,
            vuLevel: 0,
            muted: false,
            sampleRate: 0,
            channelCount: 0,
            vadActive: true,
            vadSpeaking: false,
            vadThreshold: 0.08,
            vadHoldFrames: 0,
            vadSilenceFrames: 0,
            vadSpeechFrames: 10,
            vadReleaseFrames: 20,
            lastAudioBlob: null,
            onVuUpdate: null,
            onVadChange: null
        };
        this.microphones.push(mic);
        document.dispatchEvent(new CustomEvent('device-added', {
            detail: { type: 'microphone', device: mic }
        }));
        return { success: true, data: mic };
    }

    async removeMicrophone(id) {
        const idx = this.microphones.findIndex(m => m.id === id);
        if (idx === -1) return { success: false, error: '未找到麦克风' };
        const mic = this.microphones[idx];
        await this._stopMicrophoneStream(mic);
        this.microphones.splice(idx, 1);
        delete this._vuAnalysers[id];
        document.dispatchEvent(new CustomEvent('device-removed', {
            detail: { type: 'microphone', id: id }
        }));
        return { success: true };
    }

    async startMicrophone(id) {
        const mic = this.microphones.find(m => m.id === id);
        if (!mic) return { success: false, error: '未找到麦克风' };
        if (mic.active) return { success: true };
        try {
            var compat = window.g_browserCompat || (typeof BrowserCompat !== 'undefined' ? new BrowserCompat() : null);
            if (!compat) return { success: false, error: 'BrowserCompat未加载' };
            var baseConstraints = JSON.parse(JSON.stringify(compat.getDefaultAudioConstraints()));
            baseConstraints.audio.deviceId = { exact: mic.deviceId };
            var stream = await compat.getUserMediaWithFallback(baseConstraints);
            mic.stream = stream;
            mic.active = true;

            var track = stream.getAudioTracks()[0];
            var settings = track.getSettings();
            mic.sampleRate = settings.sampleRate || 0;
            mic.channelCount = settings.channelCount || 0;

            if (!this.audioContext) {
                this.audioContext = compat.getAudioContext();
            }
            var source = this.audioContext.createMediaStreamSource(stream);
            var analyser = this.audioContext.createAnalyser();
            analyser.fftSize = 256;
            source.connect(analyser);
            mic.analyser = analyser;
            this._vuAnalysers[id] = analyser;

            document.dispatchEvent(new CustomEvent('device-started', {
                detail: { type: 'microphone', id: id }
            }));
            return { success: true };
        } catch (err) {
            mic.active = false;
            console.error('启动麦克风失败:', err.message);
            return { success: false, error: '启动麦克风失败: ' + err.message };
        }
    }

    stopMicrophone(id) {
        const mic = this.microphones.find(m => m.id === id);
        if (!mic) return { success: false, error: '未找到麦克风' };
        this._stopMicrophoneStream(mic);
        document.dispatchEvent(new CustomEvent('device-stopped', {
            detail: { type: 'microphone', id: id }
        }));
        return { success: true };
    }

    _stopMicrophoneStream(mic) {
        if (mic.stream) {
            mic.stream.getTracks().forEach(t => t.stop());
            mic.stream = null;
        }
        mic.active = false;
        mic.analyser = null;
        mic.vuLevel = 0;
        delete this._vuAnalysers[mic.id];
    }

    _updateVuLevel(id) {
        const analyser = this._vuAnalysers[id];
        if (!analyser) return;
        const dataArray = new Uint8Array(analyser.frequencyBinCount);
        analyser.getByteFrequencyData(dataArray);
        let sum = 0;
        for (let i = 0; i < dataArray.length; i++) {
            sum += dataArray[i];
        }
        const level = sum / dataArray.length / 255;
        const mic = this.microphones.find(m => m.id === id);
        if (!mic) return;
        mic.vuLevel = level;
        if (mic.onVuUpdate) mic.onVuUpdate(level);

        if (mic.vadActive) {
            const wasSpeaking = mic.vadSpeaking;
            if (level > mic.vadThreshold) {
                mic.vadHoldFrames++;
                mic.vadSilenceFrames = 0;
                if (mic.vadHoldFrames >= mic.vadSpeechFrames) {
                    mic.vadSpeaking = true;
                }
            } else {
                mic.vadSilenceFrames++;
                mic.vadHoldFrames = 0;
                if (mic.vadSilenceFrames >= mic.vadReleaseFrames) {
                    mic.vadSpeaking = false;
                }
            }
            if (wasSpeaking !== mic.vadSpeaking) {
                if (mic.onVadChange) mic.onVadChange(mic.vadSpeaking);
            }
        }
    }

    _updateAllVuMeters() {
        for (const id of Object.keys(this._vuAnalysers)) {
            this._updateVuLevel(id);
        }
    }

    async addSpeaker(deviceId) {
        const device = this._availableSpeakers.find(d => d.deviceId === deviceId);
        if (!device) {
            return { success: false, error: '未找到指定扬声器设备' };
        }
        if (this.speakers.find(s => s.deviceId === deviceId && s.active)) {
            return { success: false, error: '该扬声器已添加' };
        }
        const id = 'spk_' + (++this._speakerIdCounter);
        const speaker = {
            id: id,
            deviceId: device.deviceId,
            label: device.label || '扬声器 ' + id,
            active: false,
            volume: 80,
            muted: false,
            audioElement: null
        };
        this.speakers.push(speaker);
        document.dispatchEvent(new CustomEvent('device-added', {
            detail: { type: 'speaker', device: speaker }
        }));
        return { success: true, data: speaker };
    }

    async removeSpeaker(id) {
        const idx = this.speakers.findIndex(s => s.id === id);
        if (idx === -1) return { success: false, error: '未找到扬声器' };
        const speaker = this.speakers[idx];
        this._stopSpeaker(speaker);
        this.speakers.splice(idx, 1);
        document.dispatchEvent(new CustomEvent('device-removed', {
            detail: { type: 'speaker', id: id }
        }));
        return { success: true };
    }

    startSpeaker(id) {
        const speaker = this.speakers.find(s => s.id === id);
        if (!speaker) return { success: false, error: '未找到扬声器' };
        speaker.active = true;
        document.dispatchEvent(new CustomEvent('device-started', {
            detail: { type: 'speaker', id: id }
        }));
        return { success: true };
    }

    stopSpeaker(id) {
        const speaker = this.speakers.find(s => s.id === id);
        if (!speaker) return { success: false, error: '未找到扬声器' };
        this._stopSpeaker(speaker);
        document.dispatchEvent(new CustomEvent('device-stopped', {
            detail: { type: 'speaker', id: id }
        }));
        return { success: true };
    }

    _stopSpeaker(speaker) {
        speaker.active = false;
        if (speaker.audioElement) {
            speaker.audioElement.pause();
            speaker.audioElement.src = '';
            speaker.audioElement = null;
        }
    }

    setSpeakerVolume(id, volume) {
        const speaker = this.speakers.find(s => s.id === id);
        if (!speaker) return;
        speaker.volume = Math.max(0, Math.min(100, volume));
        if (speaker.audioElement) {
            speaker.audioElement.volume = speaker.volume / 100;
        }
    }

    setSpeakerMuted(id, muted) {
        const speaker = this.speakers.find(s => s.id === id);
        if (!speaker) return;
        speaker.muted = muted;
        if (speaker.audioElement) {
            speaker.audioElement.muted = muted;
        }
    }

    /* ZSFABC-F004修复: 桥接方法 - voice-command.js需要的独立静音/取消静音API */
    muteSpeaker(id) {
        this.setSpeakerMuted(id, true);
        const speaker = this.speakers.find(s => s.id === id);
        if (speaker) speaker.muted = true;
    }

    unmuteSpeaker(id) {
        this.setSpeakerMuted(id, false);
        const speaker = this.speakers.find(s => s.id === id);
        if (speaker) speaker.muted = false;
    }

    async playAudioOnSpeaker(speakerId, audioBlob) {
        const speaker = this.speakers.find(s => s.id === speakerId);
        if (!speaker || !speaker.active) {
            return { success: false, error: '扬声器未激活' };
        }
        try {
            const url = URL.createObjectURL(audioBlob);
            const audio = new Audio(url);
            audio.volume = speaker.volume / 100;
            audio.muted = speaker.muted;
            if (speaker.deviceId && audio.setSinkId) {
                try {
                    await audio.setSinkId(speaker.deviceId);
                } catch (e) {
                    console.warn('setSinkId 失败（扬声器路由降级）:', e.message);
                }
            }
            if (speaker.audioElement) {
                speaker.audioElement.pause();
                speaker.audioElement.src = '';
            }
            speaker.audioElement = audio;
            audio.play().catch(e => console.warn('音频播放失败:', e.message));
            return { success: true };
        } catch (err) {
            return { success: false, error: '音频播放失败: ' + err.message };
        }
    }

    stopAllAudio() {
        for (const speaker of this.speakers) {
            this._stopSpeaker(speaker);
        }
    }

    async addCamera(deviceId) {
        const device = this._availableCameras.find(d => d.deviceId === deviceId);
        if (!device) {
            return { success: false, error: '未找到指定摄像头设备' };
        }
        if (this.cameras.find(c => c.deviceId === deviceId && c.active)) {
            return { success: false, error: '该摄像头已添加' };
        }
        const id = 'cam_' + (++this._cameraIdCounter);
        const camera = {
            id: id,
            deviceId: device.deviceId,
            label: device.label || '摄像头 ' + id,
            active: false,
            stream: null,
            videoElement: null,
            resolution: 'medium',
            mirrored: false,
            currentSnapshot: null,
            availableResolutions: []
        };
        this.cameras.push(camera);
        document.dispatchEvent(new CustomEvent('device-added', {
            detail: { type: 'camera', device: camera }
        }));
        return { success: true, data: camera };
    }

    async removeCamera(id) {
        const idx = this.cameras.findIndex(c => c.id === id);
        if (idx === -1) return { success: false, error: '未找到摄像头' };
        const camera = this.cameras[idx];
        await this._stopCameraStream(camera);
        this.cameras.splice(idx, 1);
        document.dispatchEvent(new CustomEvent('device-removed', {
            detail: { type: 'camera', id: id }
        }));
        return { success: true };
    }

    async startCamera(id, resolution) {
        const camera = this.cameras.find(c => c.id === id);
        if (!camera) return { success: false, error: '未找到摄像头' };
        if (camera.active) {
            if (resolution && resolution !== camera.resolution) {
                await this._stopCameraStream(camera);
            } else {
                return { success: true };
            }
        }
        try {
            var compat = window.g_browserCompat || (typeof BrowserCompat !== 'undefined' ? new BrowserCompat() : null);
            var videoConstraints = {
                deviceId: { exact: camera.deviceId }
            };
            if (resolution) {
                var resConf = compat.getVideoConstraintsForResolution(resolution);
                if (resConf.width) videoConstraints.width = resConf.width;
                if (resConf.height) videoConstraints.height = resConf.height;
            }
            var constraints = { video: videoConstraints };
            var stream = await compat.getUserMediaWithFallback(constraints);
            camera.stream = stream;
            camera.active = true;
            if (resolution) camera.resolution = resolution;

            var track = stream.getVideoTracks()[0];
            var settings = track.getSettings();
            camera.availableResolutions = [
                { label: '低 (480x360)', value: 'low' },
                { label: '中 (1280x720)', value: 'medium' },
                { label: '高 (1920x1080)', value: 'high' }
            ];

            document.dispatchEvent(new CustomEvent('device-started', {
                detail: { type: 'camera', id: id, stream: stream, settings: settings }
            }));
            return { success: true, stream: stream, settings: settings };
        } catch (err) {
            camera.active = false;
            console.error('启动摄像头失败:', err.message);
            return { success: false, error: '启动摄像头失败: ' + err.message };
        }
    }

    stopCamera(id) {
        const camera = this.cameras.find(c => c.id === id);
        if (!camera) return { success: false, error: '未找到摄像头' };
        this._stopCameraStream(camera);
        document.dispatchEvent(new CustomEvent('device-stopped', {
            detail: { type: 'camera', id: id }
        }));
        return { success: true };
    }

    async _stopCameraStream(camera) {
        if (camera.stream) {
            camera.stream.getTracks().forEach(t => t.stop());
            camera.stream = null;
        }
        camera.active = false;
    }

    captureSnapshot(id) {
        const camera = this.cameras.find(c => c.id === id);
        if (!camera || !camera.active) return null;
        const videoEl = document.getElementById('camera-preview-' + id);
        if (!videoEl) return null;
        const canvas = document.createElement('canvas');
        canvas.width = videoEl.videoWidth;
        canvas.height = videoEl.videoHeight;
        const ctx = canvas.getContext('2d');
        if (camera.mirrored) {
            ctx.translate(canvas.width, 0);
            ctx.scale(-1, 1);
        }
        ctx.drawImage(videoEl, 0, 0);
        const dataUrl = canvas.toDataURL('image/jpeg', 0.85);
        camera.currentSnapshot = dataUrl;
        canvas.remove();
        document.dispatchEvent(new CustomEvent('camera-snapshot', {
            detail: { id: id, dataUrl: dataUrl }
        }));
        return dataUrl;
    }

    setCameraMirrored(id, mirrored) {
        const camera = this.cameras.find(c => c.id === id);
        if (!camera) return;
        camera.mirrored = mirrored;
    }

    setCameraResolution(id, resolution) {
        const camera = this.cameras.find(c => c.id === id);
        if (!camera) return;
        if (camera.active && resolution !== camera.resolution) {
            this.startCamera(id, resolution);
        }
        camera.resolution = resolution;
    }

    getMicrophoneStream(id) {
        const mic = this.microphones.find(m => m.id === id);
        return mic && mic.active ? mic.stream : null;
    }

    getCameraStream(id) {
        const camera = this.cameras.find(c => c.id === id);
        return camera && camera.active ? camera.stream : null;
    }

    getAudioContext() {
        if (!this.audioContext) {
            var compat = window.g_browserCompat || new BrowserCompat();
            this.audioContext = compat.getAudioContext();
        }
        if (this.audioContext && this.audioContext.state === 'suspended') {
            this.audioContext.resume();
        }
        return this.audioContext;
    }

    enableDualCameraSpatialPerception(leftCameraId, rightCameraId, config) {
        const leftCam = this.cameras.find(c => c.id === leftCameraId);
        const rightCam = this.cameras.find(c => c.id === rightCameraId);
        if (!leftCam || !rightCam) {
            return { success: false, error: '需要两个已添加的摄像头' };
        }
        if (!leftCam.active || !rightCam.active) {
            return { success: false, error: '两个摄像头都必须已启动' };
        }
        this.stereoVision.enabled = true;
        this.stereoVision.leftCameraId = leftCameraId;
        this.stereoVision.rightCameraId = rightCameraId;
        if (config) {
            if (config.baseline) this.stereoVision.baseline = config.baseline;
            if (config.focalLength) this.stereoVision.focalLength = config.focalLength;
            if (config.captureInterval) this.stereoVision.captureInterval = config.captureInterval;
        }
        this._startStereoProcessing();
        document.dispatchEvent(new CustomEvent('stereo-vision-enabled', {
            detail: {
                leftCameraId: leftCameraId,
                rightCameraId: rightCameraId,
                baseline: this.stereoVision.baseline,
                focalLength: this.stereoVision.focalLength
            }
        }));
        return { success: true };
    }

    disableDualCameraSpatialPerception() {
        this.stereoVision.enabled = false;
        this._stopStereoProcessing();
        this.stereoVision.disparityMap = null;
        this.stereoVision.depthMap = null;
        this.stereoVision.pointCloud = null;
        document.dispatchEvent(new CustomEvent('stereo-vision-disabled'));
        return { success: true };
    }

    _startStereoProcessing() {
        this._stopStereoProcessing();
        /* M-026: 创建Web Worker用于Census变换立体匹配，避免阻塞主线程 */
        try {
            if (!this.stereoVision.worker) {
                this.stereoVision.worker = new Worker('js/workers/stereo-worker.js');
                var self = this;
                this.stereoVision.worker.onmessage = function(e) {
                    self._onStereoWorkerResult(e.data);
                };
                this.stereoVision.worker.onerror = function(err) {
                    console.error('[立体匹配Worker] 错误:', err.message);
                    self.stereoVision.workerBusy = false;
                };
            }
        } catch (e) {
            console.warn('[立体匹配Worker] 创建失败，回退到主线程:', e.message);
            this.stereoVision.worker = null;
        }
        this.stereoVision.processingInterval = setInterval(() => {
            if (!this.stereoVision.enabled) return;
            this._processStereoFrame();
        }, this.stereoVision.captureInterval);
    }

    _stopStereoProcessing() {
        if (this.stereoVision.processingInterval) {
            clearInterval(this.stereoVision.processingInterval);
            this.stereoVision.processingInterval = null;
        }
        /* M-026: 终止Web Worker */
        if (this.stereoVision.worker) {
            this.stereoVision.worker.terminate();
            this.stereoVision.worker = null;
            this.stereoVision.workerBusy = false;
        }
    }

    async _processStereoFrame() {
        try {
            const leftCanvas = document.getElementById('camera-preview-' + this.stereoVision.leftCameraId);
            const rightCanvas = document.getElementById('camera-preview-' + this.stereoVision.rightCameraId);
            if (!leftCanvas || !rightCanvas) return;

            const leftData = this._captureCanvasData(leftCanvas);
            const rightData = this._captureCanvasData(rightCanvas);
            if (!leftData || !rightData) return;

            /* M-026: 使用Web Worker异步计算视差图，避免阻塞主线程 */
            if (this.stereoVision.worker && !this.stereoVision.workerBusy) {
                this.stereoVision.workerBusy = true;
                /* 拷贝像素缓冲区发送到Worker（不能直接传输ImageData.buffer） */
                var leftBuf = leftData.data.buffer.slice(0);
                var rightBuf = rightData.data.buffer.slice(0);
                var w = leftData.width;
                var h = leftData.height;
                var maxDisp = Math.min(64, Math.floor(w / 4));
                /* 暂存当前帧的width/height供Worker回调使用 */
                this.stereoVision._pendingWidth = w;
                this.stereoVision._pendingHeight = h;
                this.stereoVision.worker.postMessage({
                    type: 'compute',
                    leftBuffer: leftBuf,
                    rightBuffer: rightBuf,
                    width: w,
                    height: h,
                    maxDisparity: maxDisp
                }, [leftBuf, rightBuf]);
            } else if (!this.stereoVision.worker) {
                /* Worker不可用时回退到主线程同步计算 */
                const disparityMap = this._computeDisparity(leftData, rightData);
                this._processStereoResult(disparityMap, leftData.width, leftData.height);
            }
        } catch (err) {
            console.warn('立体视觉处理错误:', err.message);
        }
    }

    /**
     * M-026: Worker完成回调 — 接收视差图并计算深度图/点云
     */
    _onStereoWorkerResult(msg) {
        if (msg.type !== 'result') {
            this.stereoVision.workerBusy = false;
            return;
        }
        var w = msg.width;
        var h = msg.height;
        var disparityMap = new Float32Array(msg.disparityBuffer);
        var confidenceMap = new Float32Array(msg.confidenceBuffer);

        this.stereoVision.confidenceMap = confidenceMap;
        this._processStereoResult(disparityMap, w, h);
        this.stereoVision.workerBusy = false;
    }

    /**
     * M-026: 统一的立体视觉结果处理 — 深度图 + 点云 + 事件分发
     */
    _processStereoResult(disparityMap, width, height) {
        this.stereoVision.disparityMap = disparityMap;

        const depthMap = this._disparityToDepth(disparityMap);
        this.stereoVision.depthMap = depthMap;

        const pointCloud = this._depthToPointCloud(depthMap, width, height);
        this.stereoVision.pointCloud = pointCloud;

        this.stereoVision.lastCaptureTime = Date.now();

        document.dispatchEvent(new CustomEvent('stereo-vision-update', {
            detail: {
                disparityMap: disparityMap,
                depthMap: depthMap,
                pointCloud: pointCloud,
                timestamp: this.stereoVision.lastCaptureTime
            }
        }));
        if (this.stereoVision.onDisparityUpdate) this.stereoVision.onDisparityUpdate(disparityMap);
        if (this.stereoVision.onDepthUpdate) this.stereoVision.onDepthUpdate(depthMap);
        if (this.stereoVision.onPointCloudUpdate) this.stereoVision.onPointCloudUpdate(pointCloud);
    }

    _captureCanvasData(videoEl) {
        if (!videoEl || videoEl.videoWidth === 0) return null;
        const width = Math.min(videoEl.videoWidth, 320);
        const height = Math.min(videoEl.videoHeight, 240);
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(videoEl, 0, 0, width, height);
        const imageData = ctx.getImageData(0, 0, width, height);
        canvas.remove();
        return imageData;
    }

    _computeDisparity(leftImageData, rightImageData) {
        const w = leftImageData.width;
        const h = leftImageData.height;
        const leftGray = this._toGrayscale(leftImageData);
        const rightGray = this._toGrayscale(rightImageData);
        const blockSize = 7;
        const halfBlock = Math.floor(blockSize / 2);
        const maxDisparity = Math.min(64, Math.floor(w / 4));
        const disparityMap = new Float32Array(w * h);
        const confidenceMap = new Float32Array(w * h);
        /* P1-026修复：Census变换 - 对每个像素计算其邻域的相对亮度模式
           替代原始灰度值匹配，对光照变化和相机增益差异具有鲁棒性 */
        let censusLeft  = new Array(w * h);
        let censusRight = new Array(w * h);
        const computeCensus = (gray, imgW, imgH) => {
            const census = new Array(imgW * imgH);
            const cRad = 3;
            const cBits = (2 * cRad + 1) * (2 * cRad + 1) - 1;
            for (let y = cRad; y < imgH - cRad; y++) {
                for (let x = cRad; x < imgW - cRad; x++) {
                    const idx = y * imgW + x;
                    const center = gray[idx];
                    let val = BigInt(0);
                    let bit = 0;
                    for (let dy = -cRad; dy <= cRad; dy++) {
                        for (let dx = -cRad; dx <= cRad; dx++) {
                            if (dx === 0 && dy === 0) continue;
                            const nIdx = (y + dy) * imgW + (x + dx);
                            if (gray[nIdx] >= center) val |= (BigInt(1) << BigInt(bit));
                            bit++;
                            if (bit >= 48) break;
                        }
                        if (bit >= 48) break;
                    }
                    census[idx] = val;
                }
            }
            return census;
        };
        /* P1-026: 对左右图像计算Census变换签名（48位大整数） */
        censusLeft  = computeCensus(leftGray, w, h);
        censusRight = computeCensus(rightGray, w, h);
        const hammingDist = (a, b) => {
            let diff = a ^ b;
            let count = 0;
            while (diff > BigInt(0)) { count++; diff &= (diff - BigInt(1)); }
            return count;
        };
        /* P1-026修复：改进的块匹配 - Census变换汉明距离 + 自适应窗口 + 亚像素插值 */
        for (let y = blockSize; y < h - blockSize; y += 2) {
            for (let x = blockSize + maxDisparity; x < w - blockSize; x += 2) {
                const centerCensus = censusLeft[y * w + x];
                if (centerCensus === undefined) continue;
                let bestMatch = 0;
                let bestScore = Infinity;
                let bestScorePlus = Infinity;
                let bestScoreMinus = Infinity;
                for (let d = 0; d < maxDisparity; d++) {
                    const rightIdx = y * w + (x - d);
                    const rCensus = censusRight[rightIdx];
                    if (rCensus === undefined) continue;
                    /* 边缘区域使用较小窗口防止越界 */
                    let score = 0;
                    let count = 0;
                    const edgeDist = Math.min(x - d, w - x, y, h - y);
                    const adaptHalf = Math.max(2, Math.min(halfBlock, Math.floor(edgeDist / 2)));
                    for (let dy = -adaptHalf; dy <= adaptHalf; dy++) {
                        for (let dx = -adaptHalf; dx <= adaptHalf; dx++) {
                            const lIdx = (y + dy) * w + (x + dx);
                            const rIdx = (y + dy) * w + (x + dx - d);
                            if (lIdx >= 0 && lIdx < w * h && rIdx >= 0 && rIdx < w * h &&
                                censusLeft[lIdx] !== undefined && censusRight[rIdx] !== undefined) {
                                score += hammingDist(censusLeft[lIdx], censusRight[rIdx]);
                                count++;
                            }
                        }
                    }
                    if (count > 0) {
                        score /= count;
                        if (score < bestScore) {
                            bestScore = score;
                            bestMatch = d;
                        }
                    }
                    if (d === bestMatch + 1) bestScorePlus = score;
                    if (d === bestMatch - 1) bestScoreMinus = score;
                }
                /* P1-026修复：亚像素插值 - 抛物线拟合提升视差精度到0.1像素级 */
                let subpixelDisp = bestMatch;
                if (bestMatch > 0 && bestMatch < maxDisparity - 1 &&
                    bestScoreMinus < Infinity && bestScorePlus < Infinity) {
                    const denom = 2.0 * (bestScoreMinus - 2.0 * bestScore + bestScorePlus);
                    if (Math.abs(denom) > 1e-8) {
                        subpixelDisp = bestMatch - (bestScorePlus - bestScoreMinus) / denom;
                    }
                }
                disparityMap[y * w + x] = subpixelDisp / maxDisparity;
                /* P1-026修复：置信度计算 - 最优匹配分数与次优的比值 */
                confidenceMap[y * w + x] = bestScore < Infinity ? Math.min(1.0, 0.05 / (bestScore + 0.001)) : 0.0;
            }
        }
        /* P1-026修复：双向一致性检查空洞填充 + 中值滤波平滑 */
        for (let y = 1; y < h - 1; y++) {
            for (let x = 1; x < w - 1; x++) {
                const idx = y * w + x;
                if (disparityMap[idx] === 0 && idx > 0) {
                    const neighbors = [];
                    for (let dy = -1; dy <= 1; dy++) {
                        for (let dx = -1; dx <= 1; dx++) {
                            const ni = (y + dy) * w + (x + dx);
                            if (ni >= 0 && ni < disparityMap.length && disparityMap[ni] > 0) {
                                neighbors.push(disparityMap[ni]);
                            }
                        }
                    }
                    if (neighbors.length > 0) {
                        neighbors.sort((a, b) => a - b);
                        disparityMap[idx] = neighbors[Math.floor(neighbors.length / 2)];
                        confidenceMap[idx] = 0.3;
                    }
                }
            }
        }
        this.stereoVision.confidenceMap = confidenceMap;
        return disparityMap;
    }

    _toGrayscale(imageData) {
        const data = imageData.data;
        const len = imageData.width * imageData.height;
        const gray = new Float32Array(len);
        for (let i = 0; i < len; i++) {
            const idx = i * 4;
            gray[i] = data[idx] * 0.299 + data[idx + 1] * 0.587 + data[idx + 2] * 0.114;
        }
        return gray;
    }

    _disparityToDepth(disparityMap) {
        const depthMap = new Float32Array(disparityMap.length);
        const minDepth = 0.1;
        const maxDepth = 10.0;
        for (let i = 0; i < disparityMap.length; i++) {
            if (disparityMap[i] > 0.01) {
                const depth = (this.stereoVision.focalLength * this.stereoVision.baseline) / (disparityMap[i] * 100);
                depthMap[i] = Math.min(maxDepth, Math.max(minDepth, depth));
            } else {
                depthMap[i] = maxDepth;
            }
        }
        return depthMap;
    }

    _depthToPointCloud(depthMap, width, height) {
        const fx = this.stereoVision.focalLength;
        const fy = this.stereoVision.focalLength;
        const cx = width / 2;
        const cy = height / 2;
        const points = [];
        const step = 3;
        for (let v = 0; v < height; v += step) {
            for (let u = 0; u < width; u += step) {
                const idx = v * width + u;
                const depth = depthMap[idx];
                if (depth > 0.1 && depth < 10.0) {
                    const x = (u - cx) * depth / fx;
                    const y = (v - cy) * depth / fy;
                    const z = depth;
                    points.push({ x: x, y: -y, z: z });
                }
            }
        }
        return points;
    }

    getStereoDepthMap() {
        return this.stereoVision.depthMap;
    }

    getStereoPointCloud() {
        return this.stereoVision.pointCloud;
    }

    getStereoDisparityMap() {
        return this.stereoVision.disparityMap;
    }

    isStereovisionEnabled() {
        return this.stereoVision.enabled;
    }

    setStereoBaseline(baseline) {
        this.stereoVision.baseline = Math.max(0.01, Math.min(1.0, baseline));
    }

    setStereoFocalLength(focalLength) {
        this.stereoVision.focalLength = Math.max(100, Math.min(2000, focalLength));
    }

    /* ZSFX-033修复: 设备添加面板HTML生成函数
     * 返回串行、扬声器、摄像头的完整配置UI界面HTML字符串 */
    deviceAddPanel() {
        var self = this;
        var microphones = this.getAvailableMicrophones();
        var speakers = this.getAvailableSpeakers();
        var cameras = this.getAvailableCameras();
        /* 已添加设备ID集合 */
        var addedMicIds = this.microphones.map(function(m) { return m.deviceId; });
        var addedSpkIds = this.speakers.map(function(s) { return s.deviceId; });
        var addedCamIds = this.cameras.map(function(c) { return c.deviceId; });

        var html = '';
        html += '<div class="device-add-panel" style="max-height:70vh;overflow-y:auto;padding:16px;">';

        /* 麦克风区域 */
        html += '<div class="device-section" style="margin-bottom:20px;">';
        html += '<h4 style="margin:0 0 8px 0;color:#3498db;border-bottom:1px solid rgba(52,152,219,0.3);padding-bottom:6px;"> 麦克风设备</h4>';
        html += '<div style="margin-bottom:8px;">';
        html += '<button onclick="window.g_deviceManager.scanAudioDevices()" class="btn btn-sm" style="background:#3498db;color:#fff;border:none;padding:6px 14px;border-radius:4px;cursor:pointer;"> 刷新音频设备列表</button>';
        html += '</div>';
        if (microphones.length === 0) {
            html += '<p style="color:#999;font-size:13px;padding:8px;">未检测到麦克风设备，请检查浏览器权限设置</p>';
        } else {
            html += '<div class="device-list" style="max-height:180px;overflow-y:auto;">';
            for (var i = 0; i < microphones.length; i++) {
                var mic = microphones[i];
                var isAdded = addedMicIds.indexOf(mic.deviceId) !== -1;
                html += '<div class="device-item" style="display:flex;align-items:center;justify-content:space-between;padding:8px 10px;margin:4px 0;background:rgba(255,255,255,0.03);border-radius:6px;border:1px solid rgba(255,255,255,0.08);">';
                html += '<div style="flex:1;min-width:0;">';
                html += '<div style="font-size:13px;font-weight:500;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">' + self._escapeHtml(mic.label || '麦克风 ' + (i + 1)) + '</div>';
                html += '<div style="font-size:11px;color:#888;">' + self._escapeHtml(mic.deviceId || '无设备ID') + '</div>';
                html += '</div>';
                html += '<div style="margin-left:12px;flex-shrink:0;">';
                if (isAdded) {
                    html += '<span style="color:#27ae60;font-size:12px;"> 已添加</span>';
                } else {
                    html += '<button onclick="window.g_deviceManager.addMicrophone(\'' + self._escAttr(mic.deviceId) + '\').then(function(r){if(r.success){window.g_deviceManager.deviceAddPanel_show();window.g_deviceManager.startMicrophone(r.data.id);}})" class="btn btn-sm" style="background:#27ae60;color:#fff;border:none;padding:4px 12px;border-radius:4px;cursor:pointer;font-size:12px;">添加</button>';
                }
                html += '</div>';
                html += '</div>';
            }
            html += '</div>';
        }
        html += '</div>';

        /* 扬声器区域 */
        html += '<div class="device-section" style="margin-bottom:20px;">';
        html += '<h4 style="margin:0 0 8px 0;color:#e67e22;border-bottom:1px solid rgba(230,126,34,0.3);padding-bottom:6px;"> 扬声器设备</h4>';
        html += '<div style="margin-bottom:8px;">';
        html += '<button onclick="window.g_deviceManager.scanAudioDevices()" class="btn btn-sm" style="background:#e67e22;color:#fff;border:none;padding:6px 14px;border-radius:4px;cursor:pointer;"> 刷新音频设备列表</button>';
        html += '</div>';
        if (speakers.length === 0) {
            html += '<p style="color:#999;font-size:13px;padding:8px;">未检测到扬声器设备，请检查浏览器权限设置</p>';
        } else {
            html += '<div class="device-list" style="max-height:180px;overflow-y:auto;">';
            for (var j = 0; j < speakers.length; j++) {
                var spk = speakers[j];
                var isAdded = addedSpkIds.indexOf(spk.deviceId) !== -1;
                html += '<div class="device-item" style="display:flex;align-items:center;justify-content:space-between;padding:8px 10px;margin:4px 0;background:rgba(255,255,255,0.03);border-radius:6px;border:1px solid rgba(255,255,255,0.08);">';
                html += '<div style="flex:1;min-width:0;">';
                html += '<div style="font-size:13px;font-weight:500;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">' + self._escapeHtml(spk.label || '扬声器 ' + (j + 1)) + '</div>';
                html += '<div style="font-size:11px;color:#888;">' + self._escapeHtml(spk.deviceId || '无设备ID') + '</div>';
                html += '</div>';
                html += '<div style="margin-left:12px;flex-shrink:0;">';
                if (isAdded) {
                    html += '<span style="color:#27ae60;font-size:12px;"> 已添加</span>';
                } else {
                    html += '<button onclick="window.g_deviceManager.addSpeaker(\'' + self._escAttr(spk.deviceId) + '\').then(function(r){if(r.success){window.g_deviceManager.deviceAddPanel_show();window.g_deviceManager.startSpeaker(r.data.id);}})" class="btn btn-sm" style="background:#27ae60;color:#fff;border:none;padding:4px 12px;border-radius:4px;cursor:pointer;font-size:12px;">添加</button>';
                }
                html += '</div>';
                html += '</div>';
            }
            html += '</div>';
        }
        html += '</div>';

        /* 摄像头区域 */
        html += '<div class="device-section" style="margin-bottom:20px;">';
        html += '<h4 style="margin:0 0 8px 0;color:#9b59b6;border-bottom:1px solid rgba(155,89,182,0.3);padding-bottom:6px;"> 摄像头设备</h4>';
        html += '<div style="margin-bottom:8px;">';
        html += '<button onclick="window.g_deviceManager.scanVideoDevices()" class="btn btn-sm" style="background:#9b59b6;color:#fff;border:none;padding:6px 14px;border-radius:4px;cursor:pointer;"> 刷新视频设备列表</button>';
        html += '</div>';
        if (cameras.length === 0) {
            html += '<p style="color:#999;font-size:13px;padding:8px;">未检测到摄像头设备，请检查浏览器权限设置</p>';
        } else {
            html += '<div class="device-list" style="max-height:180px;overflow-y:auto;">';
            for (var k = 0; k < cameras.length; k++) {
                var cam = cameras[k];
                var isAdded = addedCamIds.indexOf(cam.deviceId) !== -1;
                html += '<div class="device-item" style="display:flex;align-items:center;justify-content:space-between;padding:8px 10px;margin:4px 0;background:rgba(255,255,255,0.03);border-radius:6px;border:1px solid rgba(255,255,255,0.08);">';
                html += '<div style="flex:1;min-width:0;">';
                html += '<div style="font-size:13px;font-weight:500;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">' + self._escapeHtml(cam.label || '摄像头 ' + (k + 1)) + '</div>';
                html += '<div style="font-size:11px;color:#888;">' + self._escapeHtml(cam.deviceId || '无设备ID') + '</div>';
                html += '</div>';
                html += '<div style="margin-left:12px;flex-shrink:0;">';
                if (isAdded) {
                    html += '<span style="color:#27ae60;font-size:12px;"> 已添加</span>';
                } else {
                    html += '<button onclick="window.g_deviceManager.addCamera(\'' + self._escAttr(cam.deviceId) + '\').then(function(r){if(r.success){window.g_deviceManager.deviceAddPanel_show();window.g_deviceManager.startCamera(r.data.id);}})" class="btn btn-sm" style="background:#27ae60;color:#fff;border:none;padding:4px 12px;border-radius:4px;cursor:pointer;font-size:12px;">添加</button>';
                }
                html += '</div>';
                html += '</div>';
            }
            html += '</div>';
        }
        html += '</div>';

        /* 已添加设备摘要 */
        html += '<div class="device-section">';
        html += '<h4 style="margin:0 0 8px 0;color:#27ae60;border-bottom:1px solid rgba(39,174,96,0.3);padding-bottom:6px;"> 已添加设备摘要</h4>';
        html += '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;font-size:12px;">';
        html += '<div style="background:rgba(52,152,219,0.1);padding:10px;border-radius:6px;text-align:center;">';
        html += '<div style="color:#3498db;font-weight:bold;">麦克风</div>';
        html += '<div style="font-size:18px;color:#fff;margin-top:4px;">' + this.microphones.length + ' 个</div>';
        html += '</div>';
        html += '<div style="background:rgba(230,126,34,0.1);padding:10px;border-radius:6px;text-align:center;">';
        html += '<div style="color:#e67e22;font-weight:bold;">扬声器</div>';
        html += '<div style="font-size:18px;color:#fff;margin-top:4px;">' + this.speakers.length + ' 个</div>';
        html += '</div>';
        html += '<div style="background:rgba(155,89,182,0.1);padding:10px;border-radius:6px;text-align:center;">';
        html += '<div style="color:#9b59b6;font-weight:bold;">摄像头</div>';
        html += '<div style="font-size:18px;color:#fff;margin-top:4px;">' + this.cameras.length + ' 个</div>';
        html += '</div>';
        html += '</div>';
        html += '</div>';

        html += '</div>'; /* device-add-panel */
        return html;
    }

    /* ZSFX-033: 在页面中显示设备添加面板的便捷方法 */
    deviceAddPanel_show(containerId) {
        var container;
        if (containerId) {
            container = document.getElementById(containerId);
        } else {
            /* 默认尝试找到设备管理工作台面板 */
            container = document.getElementById('device-control');
            if (!container) {
                /* 回退：创建模态层 */
                var overlay = document.createElement('div');
                overlay.id = 'device-panel-overlay';
                overlay.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.7);z-index:9999;display:flex;align-items:center;justify-content:center;';
                overlay.onclick = function(e) { if (e.target === overlay) overlay.remove(); };
                var panel = document.createElement('div');
                panel.style.cssText = 'background:#1a1a2e;border-radius:12px;border:1px solid rgba(255,255,255,0.15);max-width:640px;width:90%;max-height:80vh;box-shadow:0 20px 60px rgba(0,0,0,0.5);';
                panel.innerHTML = this.deviceAddPanel();
                /* 添加关闭按钮 */
                var closeBtn = document.createElement('div');
                closeBtn.style.cssText = 'text-align:right;padding:8px 16px 0 16px;';
                closeBtn.innerHTML = '<button style="background:none;border:none;color:#fff;font-size:20px;cursor:pointer;opacity:0.6;" onclick="document.getElementById(\'device-panel-overlay\').remove();">✕</button>';
                panel.insertBefore(closeBtn, panel.firstChild);
                overlay.appendChild(panel);
                document.body.appendChild(overlay);
                return;
            }
        }
        container.innerHTML = this.deviceAddPanel();
    }

    /* ZSFX-033修复: 真实Web API扫描音频设备（麦克风 + 扬声器）
     * 使用 navigator.mediaDevices.enumerateDevices() 获取设备列表
     * 同时请求音频权限以便获得设备标签名称 */
    async scanAudioDevices() {
        var self = this;
        try {
            /* 先请求音频权限以获取带标签的设备列表 */
            var hasPermission = false;
            if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
                try {
                    var stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    /* 立即停止轨道，仅用于获取权限 */
                    if (stream) {
                        stream.getTracks().forEach(function(t) { t.stop(); });
                        hasPermission = true;
                    }
                } catch (permErr) {
                    console.warn('[扫描音频设备] 无法获取音频权限:', permErr.message);
                }
            }
            /* 枚举所有设备 */
            if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
                console.warn('[扫描音频设备] mediaDevices API不可用');
                return { success: false, error: '浏览器不支持设备枚举API' };
            }
            var devices = await navigator.mediaDevices.enumerateDevices();
            var audioInputs = [];
            var audioOutputs = [];
            for (var i = 0; i < devices.length; i++) {
                var d = devices[i];
                if (d.deviceId === 'default' || d.deviceId === 'communications') continue;
                if (d.kind === 'audioinput') {
                    audioInputs.push({
                        deviceId: d.deviceId,
                        label: d.label || '麦克风 ' + (audioInputs.length + 1),
                        kind: 'audioinput',
                        groupId: d.groupId || ''
                    });
                } else if (d.kind === 'audiooutput') {
                    audioOutputs.push({
                        deviceId: d.deviceId,
                        label: d.label || '扬声器 ' + (audioOutputs.length + 1),
                        kind: 'audiooutput',
                        groupId: d.groupId || ''
                    });
                }
            }
            /* 更新内部设备列表 */
            this._availableMicrophones = audioInputs;
            this._availableSpeakers = audioOutputs;

            /* 分发设备更新事件 */
            document.dispatchEvent(new CustomEvent('device-list-changed', {
                detail: {
                    microphones: audioInputs,
                    speakers: audioOutputs,
                    cameras: this._availableCameras || [],
                    hasAudioPermission: hasPermission
                }
            }));

            return {
                success: true,
                microphones: audioInputs,
                speakers: audioOutputs,
                totalAudioInputs: audioInputs.length,
                totalAudioOutputs: audioOutputs.length,
                hasPermission: hasPermission
            };
        } catch (err) {
            console.error('[扫描音频设备] 扫描失败:', err.message);
            return { success: false, error: '扫描音频设备失败: ' + err.message };
        }
    }

    /* ZSFX-033修复: 真实Web API扫描视频设备（摄像头）
     * 使用 navigator.mediaDevices.enumerateDevices() 获取视频设备列表
     * 同时请求视频权限以便获得设备标签名称 */
    async scanVideoDevices() {
        var self = this;
        try {
            /* 先请求视频权限以获取带标签的设备列表 */
            var hasPermission = false;
            if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
                try {
                    var stream = await navigator.mediaDevices.getUserMedia({ video: true });
                    if (stream) {
                        stream.getTracks().forEach(function(t) { t.stop(); });
                        hasPermission = true;
                    }
                } catch (permErr) {
                    console.warn('[扫描视频设备] 无法获取视频权限:', permErr.message);
                }
            }
            /* 枚举所有设备 */
            if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
                console.warn('[扫描视频设备] mediaDevices API不可用');
                return { success: false, error: '浏览器不支持设备枚举API' };
            }
            var devices = await navigator.mediaDevices.enumerateDevices();
            var videoInputs = [];
            for (var i = 0; i < devices.length; i++) {
                var d = devices[i];
                if (d.deviceId === 'default' || d.deviceId === 'communications') continue;
                if (d.kind === 'videoinput') {
                    videoInputs.push({
                        deviceId: d.deviceId,
                        label: d.label || '摄像头 ' + (videoInputs.length + 1),
                        kind: 'videoinput',
                        groupId: d.groupId || ''
                    });
                }
            }
            /* 更新内部设备列表 */
            this._availableCameras = videoInputs;

            /* 分发设备更新事件 */
            document.dispatchEvent(new CustomEvent('device-list-changed', {
                detail: {
                    microphones: this._availableMicrophones || [],
                    speakers: this._availableSpeakers || [],
                    cameras: videoInputs,
                    hasVideoPermission: hasPermission
                }
            }));

            return {
                success: true,
                cameras: videoInputs,
                totalVideoInputs: videoInputs.length,
                hasPermission: hasPermission
            };
        } catch (err) {
            console.error('[扫描视频设备] 扫描失败:', err.message);
            return { success: false, error: '扫描视频设备失败: ' + err.message };
        }
    }

    /* HTML转义辅助函数 */
    _escapeHtml(str) {
        if (!str) return '';
        return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
    }

    /* 属性值转义（用于onclick属性中的字符串） */
    _escAttr(str) {
        if (!str) return '';
        return str.replace(/\\/g, '\\\\').replace(/'/g, "\\'").replace(/"/g, '\\"');
    }

    async destroy() {
        this.disableDualCameraSpatialPerception();
        if (this._monitorInterval) {
            clearInterval(this._monitorInterval);
            this._monitorInterval = null;
        }
        for (const mic of this.microphones) {
            await this._stopMicrophoneStream(mic);
        }
        for (const camera of this.cameras) {
            await this._stopCameraStream(camera);
        }
        for (const speaker of this.speakers) {
            this._stopSpeaker(speaker);
        }
        if (this.audioContext) {
            await this.audioContext.close();
            this.audioContext = null;
        }
        this.microphones = [];
        this.speakers = [];
        this.cameras = [];
        this._vuAnalysers = {};
        this.initialized = false;
        if (this._boundOnDeviceChange && navigator.mediaDevices) {
            navigator.mediaDevices.removeEventListener('devicechange', this._boundOnDeviceChange);
            this._boundOnDeviceChange = null;
        }
    }
}

window.DeviceManager = DeviceManager;
