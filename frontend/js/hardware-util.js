/**
 * SELF-LNN AGI 硬件扫描共享工具
 * 统一调用后端扫描API，返回规范化数据结构
 * 消除 device-control.html / hardware-setup.html / index.html 中的重复硬件扫描逻辑
 */

var HardwareScanUtil = {
    _lastScanResult: null,
    _lastScanTime: 0,
    _cacheMaxAgeMs: 5000,

    /**
     * 执行完整硬件扫描（含后端API + 浏览器设备检测）
     * @param {boolean} verbose - 是否获取详细诊断信息
     * @returns {Promise<Object>} 规范化的硬件扫描结果
     */
    async scanAll(verbose) {
        var now = Date.now();
        if (this._lastScanResult && (now - this._lastScanTime) < this._cacheMaxAgeMs) {
            return this._lastScanResult;
        }

        var result = {
            success: false,
            cpu: { cores: 'N/A', usage: 0, name: '' },
            gpu: { count: 0, devices: [], usage: 0, memory_mb: 0 },
            memory: { total_gb: 0, used_gb: 0, usage_pct: 0 },
            camera: { detected: false, count: 0, devices: [] },
            microphone: { detected: false, count: 0, devices: [] },
            speaker: { detected: false, count: 0, devices: [] },
            backendOnline: false,
            diagnosticLines: [],
            raw: null
        };

        /* 第一步：尝试后端API */
        try {
            if (window.SelfLnnApi && typeof window.SelfLnnApi.scanHardware === 'function') {
                var apiResult = await window.SelfLnnApi.scanHardware(!!verbose);
                result.raw = apiResult;
                result.backendOnline = true;

                if (apiResult.success) {
                    var scan = apiResult.scan || {};
                    if (scan.cpu) {
                        if (typeof scan.cpu === 'object') {
                            result.cpu.cores = scan.cpu.cores || scan.cpu.core_count || 'N/A';
                            result.cpu.usage = scan.cpu.usage || 0;
                            result.cpu.name = scan.cpu.name || scan.cpu.model || '';
                        } else {
                            result.cpu.name = String(scan.cpu);
                        }
                    }
                    if (scan.gpu_count !== undefined) {
                        result.gpu.count = parseInt(scan.gpu_count) || 0;
                    }
                    if (scan.gpu && typeof scan.gpu === 'object') {
                        result.gpu.count = scan.gpu.count || scan.gpu.device_count || result.gpu.count;
                        result.gpu.devices = scan.gpu.devices || [];
                        result.gpu.usage = scan.gpu.usage || scan.gpu.utilization || 0;
                        result.gpu.memory_mb = scan.gpu.memory_mb || scan.gpu.vram_mb || 0;
                    }
                    if (scan.memory_gb !== undefined) {
                        result.memory.total_gb = parseFloat(scan.memory_gb) || 0;
                    }
                    if (scan.memory && typeof scan.memory === 'object') {
                        result.memory.total_gb = scan.memory.total_gb || (scan.memory.total ? scan.memory.total / 1024 / 1024 / 1024 : 0);
                        result.memory.used_gb = scan.memory.used_gb || (scan.memory.used ? scan.memory.used / 1024 / 1024 / 1024 : 0);
                        result.memory.usage_pct = scan.memory.usage_pct || scan.memory.utilization || 0;
                    }

                    var hasCam = scan.has_camera;
                    result.camera.detected = hasCam === true || hasCam === 'true';
                    var hasMic = scan.has_mic;
                    result.microphone.detected = hasMic === true || hasMic === 'true';

                    if (apiResult.devices && apiResult.devices.length > 0) {
                        apiResult.devices.forEach(function(d) {
                            if (d.type === 'camera' || d.type === 'videoinput') {
                                result.camera.count++;
                                result.camera.devices.push(d);
                            }
                            if (d.type === 'microphone' || d.type === 'audioinput') {
                                result.microphone.count++;
                                result.microphone.devices.push(d);
                            }
                            if (d.type === 'speaker' || d.type === 'audiooutput') {
                                result.speaker.count++;
                                result.speaker.devices.push(d);
                            }
                        });
                    }

                    /* 生成诊断文本行 */
                    result.diagnosticLines.push('✅ 硬件扫描完成');
                    if (result.cpu.name) result.diagnosticLines.push('CPU: ' + result.cpu.name);
                    if (result.gpu.count > 0) result.diagnosticLines.push('GPU: ' + result.gpu.count + ' 个设备');
                    if (result.memory.total_gb > 0) result.diagnosticLines.push('内存: ' + result.memory.total_gb.toFixed(1) + ' GB');
                    result.diagnosticLines.push('摄像头: ' + (result.camera.detected ? '已检测' : '未检测'));
                    result.diagnosticLines.push('麦克风: ' + (result.microphone.detected ? '已检测' : '未检测'));

                    if (apiResult.devices && apiResult.devices.length > 0) {
                        result.diagnosticLines.push('--- 设备列表 ---');
                        apiResult.devices.slice(0, 20).forEach(function(d, i) {
                            result.diagnosticLines.push('  [' + d.type + '] ' + d.name + ' (' + (d.available ? '可用' : '不可用') + ')');
                        });
                    }

                    result.success = true;
                } else {
                    result.diagnosticLines.push('⚠️ 扫描结果异常');
                }
            }
        } catch (e) {
            result.diagnosticLines.push('后端扫描失败: ' + e.message);
        }

        /* 第二步：浏览器端检测作为补充 */
        try {
            if (navigator.hardwareConcurrency) {
                if (result.cpu.cores === 'N/A') result.cpu.cores = navigator.hardwareConcurrency;
            }
            if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
                var browserDevices = await navigator.mediaDevices.enumerateDevices();
                if (!result.camera.detected) {
                    var cams = browserDevices.filter(function(d) { return d.kind === 'videoinput'; });
                    result.camera.count = cams.length;
                    result.camera.detected = cams.length > 0;
                    result.camera.devices = cams.map(function(d) {
                        return { name: d.label || '摄像头 #' + cams.indexOf(d), id: d.deviceId, available: true };
                    });
                }
                if (!result.microphone.detected) {
                    var mics = browserDevices.filter(function(d) { return d.kind === 'audioinput'; });
                    result.microphone.count = mics.length;
                    result.microphone.detected = mics.length > 0;
                    result.microphone.devices = mics.map(function(d) {
                        return { name: d.label || '麦克风 #' + mics.indexOf(d), id: d.deviceId, available: true };
                    });
                }
                var spks = browserDevices.filter(function(d) { return d.kind === 'audiooutput'; });
                result.speaker.count = spks.length;
                result.speaker.detected = spks.length > 0;
                result.speaker.devices = spks.map(function(d) {
                    return { name: d.label || '扬声器 #' + spks.indexOf(d), id: d.deviceId, available: true };
                });
            }
        } catch (e) {
            /* 浏览器设备检测失败不影响已有数据 */
        }

        result.success = result.backendOnline || result.camera.detected || result.microphone.detected;
        this._lastScanResult = result;
        this._lastScanTime = now;
        return result;
    },

    /**
     * 格式化字节数为可读字符串
     */
    formatBytes(bytes) {
        if (!bytes || bytes <= 0) return 'N/A';
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
        if (bytes < 1073741824) return (bytes / 1048576).toFixed(1) + ' MB';
        return (bytes / 1073741824).toFixed(1) + ' GB';
    },

    /**
     * GPU完整诊断
     */
    async gpuDiagnostic() {
        var result = {
            success: false,
            backend: '未知',
            available: false,
            devices: 0,
            totalMemoryMb: 0,
            supportsCompute: false,
            supportsFp16: false,
            supportsFp64: false,
            cpuFallback: false,
            health: '未知',
            lines: []
        };

        try {
            var resp = await window.SelfLnnApi.request('/gpu/diagnostic', { method: 'GET' }, 1);
            var data = await resp.json();
            if (data.success && data.gpu_diagnostic) {
                var d = data.gpu_diagnostic;
                result.success = true;
                result.backend = d.backend;
                result.available = d.available === true || d.available === 'true';
                result.devices = parseInt(d.devices) || 0;
                result.totalMemoryMb = parseInt(d.total_memory_mb) || 0;
                result.supportsCompute = d.supports_compute === true || d.supports_compute === 'true';
                result.supportsFp16 = d.supports_fp16 === true || d.supports_fp16 === 'true';
                result.supportsFp64 = d.supports_fp64 === true || d.supports_fp64 === 'true';
                result.cpuFallback = d.cpu_backend_fallback === true || d.cpu_backend_fallback === 'true';
                result.health = d.health || '未知';

                result.lines.push('✅ GPU诊断完成');
                result.lines.push('后端: ' + result.backend);
                result.lines.push('可用: ' + (result.available ? '是' : '否'));
                result.lines.push('设备数: ' + result.devices);
                result.lines.push('显存: ' + result.totalMemoryMb + ' MB');
                result.lines.push('计算支持: ' + (result.supportsCompute ? '是' : '否'));
                result.lines.push('FP16: ' + (result.supportsFp16 ? '是' : '否'));
                result.lines.push('FP64: ' + (result.supportsFp64 ? '是' : '否'));
                result.lines.push('CPU回退: ' + (result.cpuFallback ? '已启用' : '未启用'));
                result.lines.push('健康: ' + result.health);
            }
        } catch (e) {
            result.lines.push('GPU诊断失败: ' + e.message);
        }
        return result;
    },

    /**
     * 清除缓存，强制下次重新扫描
     */
    clearCache() {
        this._lastScanResult = null;
        this._lastScanTime = 0;
    }
};
