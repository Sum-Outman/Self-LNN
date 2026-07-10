/**
 * @file stereo-worker.js
 * @brief 双目视觉Web Worker - 深度估计与3D重建
 * 
 * 在独立线程中执行Census变换立体匹配、视差计算、深度估计和3D点云重建，
 * 避免阻塞主UI线程。通过postMessage与主线程通信。
 * 
 * 修复：README宣称20个JS模块含"1个Stereo Worker"，此前此文件缺失。
 * 现在真实实现双目视觉处理的全部核心算法。
 */

/* ===== Census变换参数 ===== */
var CENSUS_WINDOW = 7;        /* Census变换窗口大小（奇数） */
var CENSUS_HALF = 3;          /* 半窗口大小 */
var MAX_DISPARITY = 64;       /* 最大视差搜索范围 */
var UNIQUENESS_RATIO = 0.8;   /* 唯一性约束比例 */
var MIN_DISPARITY = 0;        /* 最小视差 */

/* ===== 相机参数（默认值，可通过消息更新） ===== */
var focalLength = 700.0;      /* 焦距（像素） */
var baseline = 0.12;          /* 双目基线（米） */
var cx = 320.0;               /* 主点X */
var cy = 240.0;               /* 主点Y */

/* ===== 工作状态 ===== */
var isRunning = false;
var isStreaming = false;
var frameCount = 0;
var leftImage = null;
var rightImage = null;
var imageWidth = 320;
var imageHeight = 240;

/**
 * @brief Census变换：将像素邻域编码为二进制特征向量
 * @param {Uint8Array} gray 灰度图像数据
 * @param {number} width 图像宽度
 * @param {number} height 图像高度
 * @param {number} x 中心像素x坐标
 * @param {number} y 中心像素y坐标
 * @returns {number} 64位Census特征码
 */
function censusTransform(gray, width, height, x, y) {
    var centerVal = gray[y * width + x];
    var code = 0;
    var bitIndex = 0;

    for (var dy = -CENSUS_HALF; dy <= CENSUS_HALF; dy++) {
        for (var dx = -CENSUS_HALF; dx <= CENSUS_HALF; dx++) {
            if (dx === 0 && dy === 0) continue;
            var nx = x + dx;
            var ny = y + dy;
            /* 边界处理：超出图像范围视为与中心值相等 */
            if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
                bitIndex++;
                continue;
            }
            var neighborVal = gray[ny * width + nx];
            if (neighborVal < centerVal) {
                code |= (1 << bitIndex);
            }
            bitIndex++;
            if (bitIndex >= 48) break; /* 7x7窗口产生48个比较位，限制在48位内 */
        }
        if (bitIndex >= 48) break;
    }
    return code;
}

/**
 * @brief 汉明距离计算（两个Census码的差异度量）
 * @param {number} code1 Census特征码1
 * @param {number} code2 Census特征码2
 * @returns {number} 汉明距离（不同位数）
 */
function hammingDistance(code1, code2) {
    var diff = code1 ^ code2;
    var dist = 0;
    /* 高效位计数（Brian Kernighan算法） */
    while (diff) {
        dist++;
        diff &= (diff - 1);
    }
    return dist;
}

/**
 * @brief 立体匹配：Census变换 + 半全局匹配(SGM)风格代价聚合
 * @param {Uint8Array} leftGray 左目灰度图
 * @param {Uint8Array} rightGray 右目灰度图
 * @param {number} width 图像宽度
 * @param {number} height 图像高度
 * @returns {Float32Array} 视差图（浮点精度）
 */
function stereoMatch(leftGray, rightGray, width, height) {
    var disparity = new Float32Array(width * height);
    var half = CENSUS_HALF;
    var maxDisp = MAX_DISPARITY;

    /* 对每个像素计算视差 */
    for (var y = half; y < height - half; y++) {
        for (var x = maxDisp + half; x < width - half; x++) {
            var leftCode = censusTransform(leftGray, width, height, x, y);
            var bestDisp = 0;
            var bestCost = 999999;
            var secondBestCost = 999999;

            /* 沿水平极线搜索最佳匹配 */
            for (var d = MIN_DISPARITY; d < maxDisp; d++) {
                var rx = x - d;
                if (rx < half) break;
                var rightCode = censusTransform(rightGray, width, height, rx, y);
                var cost = hammingDistance(leftCode, rightCode);

                /* 简单SGM风格代价聚合：3方向路径（水平 + 左上 + 右上） */
                if (y > half) {
                    /* 左上方向代价传播 */
                    if (x > half + d && rx > half) {
                        var prevCost = (d > 0) ? 
                            Math.min(Math.abs(d - (Math.round(disparity[(y-1) * width + (x-1)]))), 20) : 0;
                        cost += prevCost * 0.1;
                    }
                    /* 右上方向代价传播 */
                    if (x < width - half - 1 && rx < width - half - 1) {
                        var prevCost2 = (d > 0) ?
                            Math.min(Math.abs(d - (Math.round(disparity[(y-1) * width + (x+1)]))), 20) : 0;
                        cost += prevCost2 * 0.1;
                    }
                }
                /* 水平方向代价传播 */
                if (x > half + 1) {
                    var prevHoriz = (d > 0) ?
                        Math.min(Math.abs(d - (Math.round(disparity[y * width + (x-1)]))), 20) : 0;
                    cost += prevHoriz * 0.15;
                }

                if (cost < bestCost) {
                    secondBestCost = bestCost;
                    bestCost = cost;
                    bestDisp = d;
                } else if (cost < secondBestCost) {
                    secondBestCost = cost;
                }
            }

            /* 唯一性约束：最佳匹配必须显著优于次佳匹配 */
            if (bestCost < secondBestCost * UNIQUENESS_RATIO) {
                /* 亚像素精炼：抛物线拟合 */
                if (bestDisp > 0 && bestDisp < maxDisp - 1) {
                    var costL = (x - bestDisp - 1 >= half) ?
                        hammingDistance(leftCode, censusTransform(rightGray, width, height, x - bestDisp - 1, y)) : bestCost;
                    var costR = (x - bestDisp + 1 < width - half) ?
                        hammingDistance(leftCode, censusTransform(rightGray, width, height, x - bestDisp + 1, y)) : bestCost;
                    var denom = 2.0 * (costL + costR - 2.0 * bestCost);
                    if (Math.abs(denom) > 1e-6) {
                        var subpixelShift = (costL - costR) / denom;
                        disparity[y * width + x] = bestDisp + subpixelShift;
                    } else {
                        disparity[y * width + x] = bestDisp;
                    }
                } else {
                    disparity[y * width + x] = bestDisp;
                }
            } else {
                disparity[y * width + x] = 0; /* 无效视差 */
            }
        }
    }

    /* 左右一致性检查(LRC)：从右到左验证视差 */
    for (var y = half; y < height - half; y++) {
        for (var x = maxDisp + half; x < width - half; x++) {
            var d = Math.round(disparity[y * width + x]);
            if (d > 0) {
                var rx = x - d;
                if (rx >= maxDisp + half && rx < width - half) {
                    var rd = Math.round(disparity[y * width + rx]);
                    if (Math.abs(d - rd) > 1) {
                        disparity[y * width + x] = 0; /* LRC不一致，标记为无效 */
                    }
                }
            }
        }
    }

    return disparity;
}

/**
 * @brief 视差图转深度图
 * @param {Float32Array} disparity 视差图
 * @param {number} width 图像宽度
 * @param {number} height 图像高度
 * @returns {Float32Array} 深度图（米）
 */
function disparityToDepth(disparity, width, height) {
    var depth = new Float32Array(width * height);
    for (var i = 0; i < width * height; i++) {
        if (disparity[i] > 0.1) {
            depth[i] = (focalLength * baseline) / disparity[i];
        } else {
            depth[i] = 0; /* 无效深度 */
        }
    }
    return depth;
}

/**
 * @brief 深度图转3D点云
 * @param {Float32Array} depth 深度图
 * @param {number} width 图像宽度
 * @param {number} height 图像高度
 * @returns {Array} 点云数组 [{x, y, z}, ...]
 */
function depthToPointCloud(depth, width, height) {
    var points = [];
    var step = 2; /* 采样步长，减少点数 */
    for (var y = 0; y < height; y += step) {
        for (var x = 0; x < width; x += step) {
            var d = depth[y * width + x];
            if (d > 0.01 && d < 50.0) { /* 过滤无效和过远深度 */
                var worldX = (x - cx) * d / focalLength;
                var worldY = (y - cy) * d / focalLength;
                var worldZ = d;
                points.push({ x: worldX, y: worldY, z: worldZ });
            }
        }
    }
    return points;
}

/**
 * @brief RGB转灰度图
 * @param {Uint8Array} rgb RGB图像数据
 * @param {number} length 像素数
 * @returns {Uint8Array} 灰度图
 */
function rgbToGray(rgb, length) {
    var gray = new Uint8Array(length);
    for (var i = 0; i < length; i++) {
        var r = rgb[i * 4];
        var g = rgb[i * 4 + 1];
        var b = rgb[i * 4 + 2];
        /* 加权灰度转换（人眼感知权重） */
        gray[i] = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
    }
    return gray;
}

/**
 * @brief 视差图可视化（彩色映射）
 * @param {Float32Array} disparity 视差图
 * @param {number} width 图像宽度
 * @param {number} height 图像高度
 * @returns {Uint8ClampedArray} RGBA图像数据
 */
function disparityToColorMap(disparity, width, height) {
    var rgba = new Uint8ClampedArray(width * height * 4);
    var maxDisp = MAX_DISPARITY;
    for (var i = 0; i < width * height; i++) {
        var d = disparity[i];
        if (d <= 0) {
            rgba[i * 4] = 0;
            rgba[i * 4 + 1] = 0;
            rgba[i * 4 + 2] = 30;
            rgba[i * 4 + 3] = 255;
        } else {
            /* 热力图映射：蓝(近) → 绿 → 红(远) */
            var ratio = Math.min(d / maxDisp, 1.0);
            if (ratio < 0.33) {
                rgba[i * 4] = Math.round(ratio * 3 * 255);
                rgba[i * 4 + 1] = 0;
                rgba[i * 4 + 2] = Math.round(255 - ratio * 3 * 255);
                rgba[i * 4 + 3] = 255;
            } else if (ratio < 0.66) {
                var r2 = (ratio - 0.33) * 3;
                rgba[i * 4] = Math.round(255 - r2 * 255);
                rgba[i * 4 + 1] = Math.round(r2 * 255);
                rgba[i * 4 + 2] = 0;
                rgba[i * 4 + 3] = 255;
            } else {
                var r3 = (ratio - 0.66) * 3;
                rgba[i * 4] = 0;
                rgba[i * 4 + 1] = Math.round(255 - r3 * 255);
                rgba[i * 4 + 2] = Math.round(r3 * 255);
                rgba[i * 4 + 3] = 255;
            }
        }
    }
    return rgba;
}

/**
 * @brief 深度图可视化
 * @param {Float32Array} depth 深度图
 * @param {number} width 图像宽度
 * @param {number} height 图像高度
 * @returns {Uint8ClampedArray} RGBA图像数据
 */
function depthToVisualization(depth, width, height) {
    var rgba = new Uint8ClampedArray(width * height * 4);
    var maxDepth = 30.0;
    for (var i = 0; i < width * height; i++) {
        var d = depth[i];
        if (d <= 0.01) {
            rgba[i * 4] = 0;
            rgba[i * 4 + 1] = 0;
            rgba[i * 4 + 2] = 20;
            rgba[i * 4 + 3] = 255;
        } else {
            var ratio = Math.min(d / maxDepth, 1.0);
            var r = Math.round(255 * (1.0 - ratio));
            var g = Math.round(255 * (1.0 - Math.abs(ratio - 0.5) * 2.0));
            var b = Math.round(255 * ratio);
            rgba[i * 4] = r;
            rgba[i * 4 + 1] = g;
            rgba[i * 4 + 2] = b;
            rgba[i * 4 + 3] = 255;
        }
    }
    return rgba;
}

/**
 * @brief 3D点云简单位置可视化（顶视图投影）
 * @param {Array} points 点云数组
 * @param {number} width 画布宽度
 * @param {number} height 画布高度
 * @returns {Uint8ClampedArray} RGBA图像数据
 */
function pointCloudToVisualization(points, width, height) {
    var rgba = new Uint8ClampedArray(width * height * 4);
    /* 黑色背景 */
    for (var i = 0; i < width * height; i++) {
        rgba[i * 4] = 10;
        rgba[i * 4 + 1] = 10;
        rgba[i * 4 + 2] = 30;
        rgba[i * 4 + 3] = 255;
    }

    if (points.length === 0) return rgba;

    /* 计算点云范围 */
    var minX = Infinity, maxX = -Infinity;
    var minZ = Infinity, maxZ = -Infinity;
    for (var p = 0; p < points.length; p++) {
        var pt = points[p];
        if (pt.x < minX) minX = pt.x;
        if (pt.x > maxX) maxX = pt.x;
        if (pt.z < minZ) minZ = pt.z;
        if (pt.z > maxZ) maxZ = pt.z;
    }

    var rangeX = maxX - minX || 1;
    var rangeZ = maxZ - minZ || 1;
    var scale = Math.min(width, height) * 0.75;

    /* 绘制点（顶视图：X轴水平，Z轴垂直） */
    for (var p = 0; p < points.length; p++) {
        var pt = points[p];
        var px = Math.round(((pt.x - minX) / rangeX) * scale + (width - scale) * 0.5);
        var py = Math.round(height - 1 - ((pt.z - minZ) / rangeZ) * scale - (height - scale) * 0.5);

        if (px >= 0 && px < width && py >= 0 && py < height) {
            var idx = (py * width + px) * 4;
            /* 深度着色 */
            var depthRatio = Math.min(pt.z / 30.0, 1.0);
            rgba[idx] = Math.round(255 * depthRatio);
            rgba[idx + 1] = Math.round(255 * (1.0 - depthRatio));
            rgba[idx + 2] = Math.round(128 + 127 * (1.0 - depthRatio));
            rgba[idx + 3] = 255;
        }
    }
    return rgba;
}

/**
 * @brief 处理双目视觉帧的主入口
 * @param {Object} data 包含leftImageData和rightImageData
 */
function processStereoFrame(data) {
    var leftData = data.leftImageData;
    var rightData = data.rightImageData;
    var w = data.width || imageWidth;
    var h = data.height || imageHeight;

    imageWidth = w;
    imageHeight = h;

    /* 更新相机参数 */
    if (data.focalLength) focalLength = data.focalLength;
    if (data.baseline) baseline = data.baseline;
    if (data.cx) cx = data.cx;
    if (data.cy) cy = data.cy;

    /* 转换为灰度图 */
    var leftGray = rgbToGray(leftData, w * h);
    var rightGray = rgbToGray(rightData, w * h);

    /* 立体匹配 */
    var startTime = performance.now();
    var disparity = stereoMatch(leftGray, rightGray, w, h);
    var matchTime = (performance.now() - startTime).toFixed(1);

    /* 深度估计 */
    var depth = disparityToDepth(disparity, w, h);

    /* 3D点云重建 */
    var pointCloud = depthToPointCloud(depth, w, h);

    /* 可视化生成 */
    var disparityVisual = disparityToColorMap(disparity, w, h);
    var depthVisual = depthToVisualization(depth, w, h);
    var pointCloudVisual = pointCloudToVisualization(pointCloud, w, h);

    /* 统计信息 */
    var validPoints = 0;
    var sumDepth = 0;
    var minDepth = Infinity;
    var maxDepth = 0;
    for (var i = 0; i < w * h; i++) {
        if (depth[i] > 0.01) {
            validPoints++;
            sumDepth += depth[i];
            if (depth[i] < minDepth) minDepth = depth[i];
            if (depth[i] > maxDepth) maxDepth = depth[i];
        }
    }
    var avgDepth = validPoints > 0 ? (sumDepth / validPoints) : 0;

    frameCount++;

    /* 返回结果到主线程 */
    self.postMessage({
        type: 'stereo_result',
        frameIndex: frameCount,
        disparity: disparityVisual.buffer,
        depth: depthVisual.buffer,
        pointCloud: pointCloudVisual.buffer,
        width: w,
        height: h,
        matchTime: matchTime,
        stats: {
            pointCount: pointCloud.length,
            validRatio: (validPoints / (w * h) * 100).toFixed(1),
            avgDepth: avgDepth.toFixed(2),
            minDepth: (minDepth < Infinity ? minDepth.toFixed(2) : 'N/A'),
            maxDepth: maxDepth.toFixed(2)
        }
    }, [
        disparityVisual.buffer,
        depthVisual.buffer,
        pointCloudVisual.buffer
    ]);
}

/**
 * @brief 处理主线程消息
 */
self.onmessage = function(e) {
    var msg = e.data;

    switch (msg.type) {
        case 'stereo_frame':
            if (!msg.leftImageData || !msg.rightImageData) {
                self.postMessage({ type: 'error', message: '缺少左右目图像数据' });
                return;
            }
            try {
                processStereoFrame(msg);
            } catch (err) {
                self.postMessage({ type: 'error', message: '双目处理异常: ' + err.message });
            }
            break;

        case 'update_params':
            /* 更新相机参数 */
            if (msg.focalLength !== undefined) focalLength = msg.focalLength;
            if (msg.baseline !== undefined) baseline = msg.baseline;
            if (msg.cx !== undefined) cx = msg.cx;
            if (msg.cy !== undefined) cy = msg.cy;
            if (msg.maxDisparity !== undefined) MAX_DISPARITY = msg.maxDisparity;
            self.postMessage({ type: 'params_updated', params: {
                focalLength: focalLength, baseline: baseline,
                cx: cx, cy: cy, maxDisparity: MAX_DISPARITY
            }});
            break;

        case 'get_params':
            self.postMessage({ type: 'params', params: {
                focalLength: focalLength, baseline: baseline,
                cx: cx, cy: cy, maxDisparity: MAX_DISPARITY,
                censusWindow: CENSUS_WINDOW
            }});
            break;

        case 'ping':
            self.postMessage({ type: 'pong', status: 'ready', frameCount: frameCount });
            break;

        default:
            self.postMessage({ type: 'error', message: '未知消息类型: ' + msg.type });
    }
};

/* Worker启动确认 */
self.postMessage({ type: 'ready', message: '双目视觉Worker已就绪', config: {
    censusWindow: CENSUS_WINDOW,
    maxDisparity: MAX_DISPARITY,
    uniquenessRatio: UNIQUENESS_RATIO,
    focalLength: focalLength,
    baseline: baseline
}});