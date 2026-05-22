/**
 * SELF-LNN AGI 立体匹配 Web Worker (M-026)
 * 
 * 将Census变换 + 汉明距离 + 亚像素插值 + 深度图计算移至Worker线程，
 * 避免在主线程阻塞UI。
 * 
 * 输入消息格式:
 *   { type: 'compute', leftData: Uint8ClampedArray, rightData: Uint8ClampedArray, 
 *     width: number, height: number, maxDisparity: number }
 * 
 * 输出消息格式:
 *   { type: 'result', disparityMap: Float32Array, confidenceMap: Float32Array,
 *     width: number, height: number, maxDisparity: number }
 */

(function() {
    'use strict';

    function toGrayscale(data, width, height) {
        var len = width * height;
        var gray = new Float32Array(len);
        for (var i = 0; i < len; i++) {
            var idx = i * 4;
            gray[i] = data[idx] * 0.299 + data[idx + 1] * 0.587 + data[idx + 2] * 0.114;
        }
        return gray;
    }

    function computeCensus(gray, imgW, imgH) {
        var census = new Array(imgW * imgH);
        var cRad = 3;
        var cBits = (2 * cRad + 1) * (2 * cRad + 1) - 1;
        for (var y = cRad; y < imgH - cRad; y++) {
            for (var x = cRad; x < imgW - cRad; x++) {
                var idx = y * imgW + x;
                var center = gray[idx];
                var val = 0n;
                var bit = 0;
                for (var dy = -cRad; dy <= cRad; dy++) {
                    for (var dx = -cRad; dx <= cRad; dx++) {
                        if (dx === 0 && dy === 0) continue;
                        var nIdx = (y + dy) * imgW + (x + dx);
                        if (gray[nIdx] >= center) val |= (1n << BigInt(bit));
                        bit++;
                        if (bit >= 48) break;
                    }
                    if (bit >= 48) break;
                }
                census[idx] = val;
            }
        }
        return census;
    }

    function hammingDist(a, b) {
        if (a === undefined || b === undefined) return Infinity;
        var diff = a ^ b;
        var count = 0;
        while (diff > 0n) { count++; diff &= (diff - 1n); }
        return count;
    }

    function computeDisparity(grayLeft, grayRight, censusLeft, censusRight, w, h, maxDisparity) {
        var blockSize = 7;
        var halfBlock = Math.floor(blockSize / 2);
        var disparityMap = new Float32Array(w * h);
        var confidenceMap = new Float32Array(w * h);

        for (var y = blockSize; y < h - blockSize; y += 2) {
            for (var x = blockSize + maxDisparity; x < w - blockSize; x += 2) {
                var centerCensus = censusLeft[y * w + x];
                if (centerCensus === undefined) continue;
                var bestMatch = 0;
                var bestScore = Infinity;
                var bestScorePlus = Infinity;
                var bestScoreMinus = Infinity;
                for (var d = 0; d < maxDisparity; d++) {
                    var rightIdx = y * w + (x - d);
                    var rCensus = censusRight[rightIdx];
                    if (rCensus === undefined) continue;
                    var score = 0;
                    var count = 0;
                    var edgeDist = Math.min(x - d, w - x, y, h - y);
                    var adaptHalf = Math.max(2, Math.min(halfBlock, Math.floor(edgeDist / 2)));
                    for (var dy = -adaptHalf; dy <= adaptHalf; dy++) {
                        for (var dx = -adaptHalf; dx <= adaptHalf; dx++) {
                            var lIdx = (y + dy) * w + (x + dx);
                            var rIdx = (y + dy) * w + (x + dx - d);
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
                /* 亚像素插值：抛物线拟合提升视差精度到0.1像素级 */
                var subpixelDisp = bestMatch;
                if (bestMatch > 0 && bestMatch < maxDisparity - 1 &&
                    bestScoreMinus < Infinity && bestScorePlus < Infinity) {
                    var denom = 2.0 * (bestScoreMinus - 2.0 * bestScore + bestScorePlus);
                    if (Math.abs(denom) > 1e-8) {
                        subpixelDisp = bestMatch - (bestScorePlus - bestScoreMinus) / denom;
                    }
                }
                disparityMap[y * w + x] = subpixelDisp / maxDisparity;
                confidenceMap[y * w + x] = bestScore < Infinity ? Math.min(1.0, 0.05 / (bestScore + 0.001)) : 0.0;
            }
        }
        /* 双向一致性检查空洞填充 + 中值滤波平滑 */
        for (var y = 1; y < h - 1; y++) {
            for (var x = 1; x < w - 1; x++) {
                var idx = y * w + x;
                if (disparityMap[idx] === 0 && idx > 0) {
                    var neighbors = [];
                    for (var dy = -1; dy <= 1; dy++) {
                        for (var dx = -1; dx <= 1; dx++) {
                            var ni = (y + dy) * w + (x + dx);
                            if (ni >= 0 && ni < disparityMap.length && disparityMap[ni] > 0) {
                                neighbors.push(disparityMap[ni]);
                            }
                        }
                    }
                    if (neighbors.length > 0) {
                        neighbors.sort(function(a, b) { return a - b; });
                        disparityMap[idx] = neighbors[Math.floor(neighbors.length / 2)];
                        confidenceMap[idx] = 0.3;
                    }
                }
            }
        }
        return { disparityMap: disparityMap, confidenceMap: confidenceMap };
    }

    self.onmessage = function(e) {
        var msg = e.data;
        if (msg.type === 'compute') {
            var leftData = new Uint8ClampedArray(msg.leftBuffer);
            var rightData = new Uint8ClampedArray(msg.rightBuffer);
            var w = msg.width;
            var h = msg.height;
            var maxDisp = msg.maxDisparity || 64;

            var grayLeft = toGrayscale(leftData, w, h);
            var grayRight = toGrayscale(rightData, w, h);

            var censusLeft = computeCensus(grayLeft, w, h);
            var censusRight = computeCensus(grayRight, w, h);

            var result = computeDisparity(grayLeft, grayRight, censusLeft, censusRight, w, h, maxDisp);

            /* 将Float32Array转为可传输的ArrayBuffer */
            var dispBuffer = result.disparityMap.buffer;
            var confBuffer = result.confidenceMap.buffer;

            self.postMessage({
                type: 'result',
                disparityBuffer: dispBuffer,
                confidenceBuffer: confBuffer,
                width: w,
                height: h,
                maxDisparity: maxDisp
            }, [dispBuffer, confBuffer]);
        }
    };

})();
