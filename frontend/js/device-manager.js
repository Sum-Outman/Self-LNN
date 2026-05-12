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
            processingInterval: null
        };
    }

    async init() {
        if (this.initialized) return;
        try {
            var compat = window.g_browserCompat || new BrowserCompat();
            var devices = await compat.enumerateMediaDevices();
            this._processDevices(devices);
            if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
                navigator.mediaDevices.addEventListener('devicechange', function() {
                    this._onDeviceChange();
                }.bind(this));
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
        var compat = window.g_browserCompat || new BrowserCompat();
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
            var compat = window.g_browserCompat || new BrowserCompat();
            var baseConstraints = compat.getDefaultAudioConstraints();
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
            var compat = window.g_browserCompat || new BrowserCompat();
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
    }

    async _processStereoFrame() {
        try {
            const leftCanvas = document.getElementById('camera-preview-' + this.stereoVision.leftCameraId);
            const rightCanvas = document.getElementById('camera-preview-' + this.stereoVision.rightCameraId);
            if (!leftCanvas || !rightCanvas) return;

            const leftData = this._captureCanvasData(leftCanvas);
            const rightData = this._captureCanvasData(rightCanvas);
            if (!leftData || !rightData) return;

            const disparityMap = this._computeDisparity(leftData, rightData);
            this.stereoVision.disparityMap = disparityMap;

            const depthMap = this._disparityToDepth(disparityMap);
            this.stereoVision.depthMap = depthMap;

            const pointCloud = this._depthToPointCloud(depthMap, leftData.width, leftData.height);
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
        } catch (err) {
            console.warn('立体视觉处理错误:', err.message);
        }
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
        const blockSize = 5;
        const maxDisparity = Math.min(60, w / 4);
        const disparityMap = new Float32Array(w * h);

        for (let y = blockSize; y < h - blockSize; y += 2) {
            for (let x = blockSize + maxDisparity; x < w - blockSize; x += 2) {
                let bestMatch = 0;
                let bestScore = Infinity;
                for (let d = 0; d < maxDisparity; d++) {
                    let score = 0;
                    let count = 0;
                    for (let dy = -blockSize; dy <= blockSize; dy++) {
                        for (let dx = -blockSize; dx <= blockSize; dx++) {
                            const leftIdx = (y + dy) * w + (x + dx);
                            const rightIdx = (y + dy) * w + (x + dx - d);
                            if (leftIdx >= 0 && leftIdx < leftGray.length && rightIdx >= 0 && rightIdx < rightGray.length) {
                                const diff = leftGray[leftIdx] - rightGray[rightIdx];
                                score += diff * diff;
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
                }
                disparityMap[y * w + x] = (bestMatch / maxDisparity);
            }
        }
        for (let i = 0; i < disparityMap.length; i++) {
            if (disparityMap[i] === 0 && i > 0) {
                disparityMap[i] = disparityMap[i - 1];
            }
        }
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
        navigator.mediaDevices.removeEventListener('devicechange', this._onDeviceChange);
    }
}

window.DeviceManager = DeviceManager;
