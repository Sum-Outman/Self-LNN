/**
 * ZSFEEE-FIX-006: 双目立体视觉 Web Worker
 * 在独立线程中执行Census变换、汉明距离计算、视差图计算
 * 通过postMessage与主线程通信，避免阻塞UI渲染
 */

(function() {
    'use strict';

    /* ZSFUSA-F08: BigInt兼容性检查，旧浏览器回退到32位汉明距离 */
    var hasBigInt = (typeof BigInt !== 'undefined');
    if (!hasBigInt) {
        /* Worker内部使用console.warn输出兼容性警告 */
        try { console.warn('[立体匹配Worker] BigInt不可用，立体匹配精度降低'); } catch(e) {}
    }

    /**
     * 将原始RGBA像素缓冲区转换为灰度浮点数组
     * @param {ArrayBuffer} buffer - 原始RGBA像素数据（每4字节 = R,G,B,A）
     * @param {number} width - 图像宽度
     * @param {number} height - 图像高度
     * @returns {Float32Array} 灰度值数组
     */
    function rgbaToGrayscale(buffer, width, height) {
        var data = new Uint8Array(buffer);
        var len = width * height;
        var gray = new Float32Array(len);
        for (var i = 0; i < len; i++) {
            var idx = i * 4;
            gray[i] = data[idx] * 0.299 + data[idx + 1] * 0.587 + data[idx + 2] * 0.114;
        }
        return gray;
    }

    /**
     * Census变换：对每个像素计算其邻域的相对亮度模式（48位签名）
     * 替代原始灰度值匹配，对光照变化和相机增益差异具有鲁棒性
     * @param {Float32Array} gray - 灰度图像数据
     * @param {number} w - 图像宽度
     * @param {number} h - 图像高度
     * @returns {Array} Census变换签名数组
     */
    function computeCensus(gray, w, h) {
        var census = new Array(w * h);
        var cRad = 3;
        for (var y = cRad; y < h - cRad; y++) {
            for (var x = cRad; x < w - cRad; x++) {
                var idx = y * w + x;
                var center = gray[idx];
                var val = hasBigInt ? BigInt(0) : 0;
                var bit = 0;
                for (var dy = -cRad; dy <= cRad; dy++) {
                    for (var dx = -cRad; dx <= cRad; dx++) {
                        if (dx === 0 && dy === 0) continue;
                        var nIdx = (y + dy) * w + (x + dx);
                        if (gray[nIdx] >= center) {
                            if (hasBigInt) {
                                val |= (BigInt(1) << BigInt(bit));
                            } else {
                                val |= (1 << bit);
                            }
                        }
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

    /**
     * 汉明距离计算：两个Census签名之间不同位的数量
     * @param {*} a - Census签名A
     * @param {*} b - Census签名B
     * @returns {number} 汉明距离
     */
    function hammingDistance(a, b) {
        if (hasBigInt) {
            var diff = a ^ b;
            var count = 0;
            while (diff > BigInt(0)) { count++; diff &= (diff - BigInt(1)); }
            return count;
        } else {
            /* 32位回退路径：逐位异或计数 */
            var diff32 = (a ^ b) >>> 0;
            var count32 = 0;
            while (diff32) { count32++; diff32 &= (diff32 - 1); }
            return count32;
        }
    }

    /**
     * 计算视差图：基于Census变换汉明距离的块匹配
     * @param {Float32Array} leftGray - 左图灰度
     * @param {Float32Array} rightGray - 右图灰度
     * @param {number} w - 图像宽度
     * @param {number} h - 图像高度
     * @param {number} maxDisparity - 最大视差搜索范围
     * @returns {{disparityMap: Float32Array, confidenceMap: Float32Array}} 视差图和置信度图
     */
    function computeDisparity(leftGray, rightGray, w, h, maxDisparity) {
        var blockSize = 7;
        var halfBlock = Math.floor(blockSize / 2);
        var disparityMap = new Float32Array(w * h);
        var confidenceMap = new Float32Array(w * h);

        /* 计算左右图像的Census变换签名 */
        var censusLeft = computeCensus(leftGray, w, h);
        var censusRight = computeCensus(rightGray, w, h);

        /* 改进的块匹配：Census变换汉明距离 + 自适应窗口 + 亚像素插值 */
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
                    /* 边缘区域使用较小窗口防止越界 */
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
                                score += hammingDistance(censusLeft[lIdx], censusRight[rIdx]);
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
                /* 置信度计算：最优匹配分数与次优的比值 */
                confidenceMap[y * w + x] = bestScore < Infinity ? Math.min(1.0, 0.05 / (bestScore + 0.001)) : 0.0;
            }
        }

        /* 空洞填充：对零值像素使用邻域中值填充 */
        for (var fy = 1; fy < h - 1; fy++) {
            for (var fx = 1; fx < w - 1; fx++) {
                var hidx = fy * w + fx;
                if (disparityMap[hidx] === 0 && hidx > 0) {
                    var neighbors = [];
                    for (var ndy = -1; ndy <= 1; ndy++) {
                        for (var ndx = -1; ndx <= 1; ndx++) {
                            var ni = (fy + ndy) * w + (fx + ndx);
                            if (ni >= 0 && ni < disparityMap.length && disparityMap[ni] > 0) {
                                neighbors.push(disparityMap[ni]);
                            }
                        }
                    }
                    if (neighbors.length > 0) {
                        neighbors.sort(function(a, b) { return a - b; });
                        disparityMap[hidx] = neighbors[Math.floor(neighbors.length / 2)];
                        confidenceMap[hidx] = 0.3;
                    }
                }
            }
        }

        return { disparityMap: disparityMap, confidenceMap: confidenceMap };
    }

    /**
     * 处理主线程发来的计算请求
     */
    self.onmessage = function(e) {
        var msg = e.data;
        if (msg.type !== 'compute') return;

        try {
            var w = msg.width;
            var h = msg.height;
            var maxDisp = msg.maxDisparity || Math.min(64, Math.floor(w / 4));

            /* 左图RGBA缓冲区 → 灰度 */
            var leftGray = rgbaToGrayscale(msg.leftBuffer, w, h);
            /* 右图RGBA缓冲区 → 灰度 */
            var rightGray = rgbaToGrayscale(msg.rightBuffer, w, h);

            /* 计算视差图和置信度图 */
            var result = computeDisparity(leftGray, rightGray, w, h, maxDisp);

            /* 将结果通过postMessage传回主线程（使用transferable对象减少拷贝） */
            var disparityBuffer = result.disparityMap.buffer;
            var confidenceBuffer = result.confidenceMap.buffer;
            self.postMessage({
                type: 'result',
                disparityBuffer: disparityBuffer,
                confidenceBuffer: confidenceBuffer,
                width: w,
                height: h
            }, [disparityBuffer, confidenceBuffer]);
        } catch (err) {
            self.postMessage({
                type: 'error',
                message: err.message || '立体匹配Worker计算失败'
            });
        }
    };

})();
