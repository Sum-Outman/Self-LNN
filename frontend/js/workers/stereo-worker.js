/* ZSF-001修复: 立体视觉Census变换Web Worker
 * 在独立线程中执行Census变换和Hamming距离立体匹配，避免阻塞主线程UI渲染。
 * ZSF-001: 统一消息协议，与device-manager.js发送格式对齐
 * 消息协议:
 *   输入: { type: 'compute', leftBuffer: ArrayBuffer, rightBuffer: ArrayBuffer, width: number, height: number, maxDisparity: number }
 *   输出: { type: 'result', disparityBuffer: ArrayBuffer, confidenceBuffer: ArrayBuffer, width: number, height: number }
 *   错误: { type: 'error', message: string }
 */

'use strict';

/* Census变换: 将像素邻域编码为二进制描述符 */
function censusTransform(pixels, width, height, x, y, windowRadius) {
    var centerValue = pixels[(y * width + x) * 4];
    var censusBits = 0;
    var bitIndex = 0;

    for (var dy = -windowRadius; dy <= windowRadius; dy++) {
        for (var dx = -windowRadius; dx <= windowRadius; dx++) {
            if (dx === 0 && dy === 0) continue;
            var nx = x + dx;
            var ny = y + dy;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                var neighborValue = pixels[(ny * width + nx) * 4];
                if (neighborValue > centerValue) {
                    censusBits |= (1 << bitIndex);
                }
            }
            bitIndex++;
            if (bitIndex >= 31) break;
        }
        if (bitIndex >= 31) break;
    }
    return censusBits;
}

/* Hamming距离 */
function hammingDistance(a, b) {
    var diff = (a ^ b) >>> 0;
    var dist = 0;
    while (diff) {
        dist += (diff & 1);
        diff >>>= 1;
    }
    return dist;
}

/* 立体匹配主函数: Census变换 + Hamming距离 */
function stereoMatch(leftPixels, rightPixels, width, height, maxDisparity, windowRadius) {
    var disparityMap = new Uint8ClampedArray(width * height);

    /* 为左图像每个窗口构建Census描述符 */
    var leftCensus = new Uint32Array(width * height);
    for (var y = windowRadius; y < height - windowRadius; y++) {
        for (var x = windowRadius; x < width - windowRadius; x++) {
            leftCensus[y * width + x] = censusTransform(leftPixels, width, height, x, y, windowRadius);
        }
    }

    /* 对右图像行扫描匹配 */
    for (var y = windowRadius; y < height - windowRadius; y++) {
        for (var x = maxDisparity + windowRadius; x < width - windowRadius; x++) {
            var leftBits = leftCensus[y * width + x];
            var bestDisparity = 0;
            var minDist = 999999;

            for (var d = 0; d <= maxDisparity; d++) {
                var rx = x - d;
                if (rx >= windowRadius) {
                    var rightBits = censusTransform(rightPixels, width, height, rx, y, windowRadius);
                    var dist = hammingDistance(leftBits, rightBits);
                    if (dist < minDist) {
                        minDist = dist;
                        bestDisparity = d;
                    }
                }
            }

            disparityMap[y * width + x] = (minDist < 999999) ? bestDisparity : 0;
        }
    }

    return disparityMap;
}

/* 置信度图计算: 基于匹配代价计算每个像素的匹配置信度 */
function computeConfidenceMap(disparityMap, width, height, leftPixels, rightPixels, maxDisparity) {
    var confidenceMap = new Float32Array(width * height);
    for (var y = 1; y < height - 1; y++) {
        for (var x = 1; x < width - 1; x++) {
            var d = disparityMap[y * width + x];
            if (d <= 0 || d >= maxDisparity) {
                confidenceMap[y * width + x] = 0.0;
                continue;
            }
            var leftVal = leftPixels[(y * width + x) * 4];
            var rx = x - d;
            if (rx < 0) { confidenceMap[y * width + x] = 0.0; continue; }
            var rightVal = rightPixels[(y * width + rx) * 4];
            var diff = Math.abs(leftVal - rightVal);
            var localConsistency = 0;
            var count = 0;
            for (var dy = -1; dy <= 1; dy++) {
                for (var dx = -1; dx <= 1; dx++) {
                    var nd = disparityMap[(y + dy) * width + (x + dx)];
                    if (Math.abs(nd - d) <= 1) { count++; }
                    localConsistency++;
                }
            }
            var consistency = count / Math.max(localConsistency, 1);
            confidenceMap[y * width + x] = consistency * Math.max(0, 1.0 - diff / 255.0);
        }
    }
    return confidenceMap;
}

/* 消息处理 */
self.onmessage = function(e) {
    var data = e.data;

    try {
        /* ZSF-001: 统一使用'compute'类型，与device-manager.js发送格式对齐 */
        if (data.type === 'compute') {
            if (!data.leftBuffer || !data.rightBuffer) {
                self.postMessage({ type: 'error', message: '缺少左右图像缓冲区数据' });
                return;
            }

            var width = data.width || 320;
            var height = data.height || 240;
            var maxDisparity = data.maxDisparity || 64;
            var windowRadius = 2;

            /* 将ArrayBuffer转为Uint8ClampedArray用于像素处理 */
            var leftPixels = new Uint8ClampedArray(data.leftBuffer);
            var rightPixels = new Uint8ClampedArray(data.rightBuffer);

            var disparityMap = stereoMatch(leftPixels, rightPixels, width, height, maxDisparity, windowRadius);

            /* 计算置信度图 */
            var confidenceMap = computeConfidenceMap(disparityMap, width, height, leftPixels, rightPixels, maxDisparity);

            /* ZSF-001: 统一输出格式为'result'，与device-manager.js接收格式对齐 */
            var disparityFloat = new Float32Array(width * height);
            for (var i = 0; i < disparityMap.length; i++) {
                disparityFloat[i] = disparityMap[i];
            }

            self.postMessage({
                type: 'result',
                disparityBuffer: disparityFloat.buffer,
                confidenceBuffer: confidenceMap.buffer,
                width: width,
                height: height
            }, [disparityFloat.buffer, confidenceMap.buffer]);
        } else if (data.type === 'ping') {
            self.postMessage({ type: 'pong' });
        } else {
            self.postMessage({ type: 'error', message: '未知消息类型: ' + data.type });
        }
    } catch (err) {
        self.postMessage({ type: 'error', message: err.message || 'Worker内部错误' });
    }
};
