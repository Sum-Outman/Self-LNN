
/**
 * SELF-LNN 纯Canvas图表工具
 * 100%自实现，不依赖任何第三方库
 * 严格遵循项目要求：纯前端自实现
 */

class SelfLnnChart {
    constructor(canvas, options) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.options = options || {};
        this.data = { labels: [], datasets: [] };

        this._animFrame = null;
        this._animProgress = 0;
        this._animTarget = 1;
        this._animStart = null;
        this._resizeObserver = null;
        this._dpr = window.devicePixelRatio || 1;

        this._initDpr();
        this._bindEvents();
    }

    _initDpr() {
        const rect = this.canvas.getBoundingClientRect();
        const w = rect.width;
        const h = rect.height;
        if (w <= 0 || h <= 0) return;
        this.canvas.width = w * this._dpr;
        this.canvas.height = h * this._dpr;
        this.canvas.style.width = w + 'px';
        this.canvas.style.height = h + 'px';
        this.ctx.setTransform(this._dpr, 0, 0, this._dpr, 0, 0);
        this._w = w;
        this._h = h;
    }

    _bindEvents() {
        if (typeof ResizeObserver !== 'undefined') {
            this._resizeObserver = new ResizeObserver(() => {
                this._initDpr();
                this.draw();
            });
            this._resizeObserver.observe(this.canvas.parentElement || this.canvas);
        }
    }

    setData(data) {
        this.data = data;
        this.draw();
    }

    updateDataset(index, newData) {
        if (this.data.datasets[index]) {
            this.data.datasets[index].data = newData;
            this.draw();
        }
    }

    addData(label, values) {
        this.data.labels.push(label);
        if (this.data.labels.length > 60) this.data.labels.shift();
        for (let i = 0; i < this.data.datasets.length; i++) {
            if (i < values.length) {
                this.data.datasets[i].data.push(values[i]);
                if (this.data.datasets[i].data.length > 60) {
                    this.data.datasets[i].data.shift();
                }
            }
        }
        this.draw();
    }

    _clear() {
        this._w = this.canvas.width / this._dpr;
        this._h = this.canvas.height / this._dpr;
        this.ctx.clearRect(0, 0, this._w, this._h);
    }

    _getDrawArea() {
        const pad = { top: 15, right: 15, bottom: 25, left: 45 };
        if (this.options.scales && this.options.scales.x && this.options.scales.x.display === false) {
            pad.bottom = 5;
        }
        if (this.options.scales && this.options.scales.y && this.options.scales.y.display === false) {
            pad.left = 5;
        }
        return {
            x: pad.left,
            y: pad.top,
            w: this._w - pad.left - pad.right,
            h: this._h - pad.top - pad.bottom
        };
    }

    draw() {
        if (!this.ctx) return;
        this._clear();
        const type = this.options.type || 'line';
        if (type === 'line') this._drawLine();
        else if (type === 'bar') this._drawBar();
        else if (type === 'radar') this._drawRadar();
        /* P1-025修复：添加散点图(scatter)、面积图(area)、饼图(pie)、热力图(heatmap)、仪表盘(gauge)渲染支持 */
        else if (type === 'scatter') this._drawScatter();
        else if (type === 'area') this._drawArea();
        else if (type === 'pie') this._drawPie();
        else if (type === 'heatmap') this._drawHeatmap();
        else if (type === 'gauge') this._drawGauge();
    }

    _drawLine() {
        const area = this._getDrawArea();
        const ctx = this.ctx;
        const datasets = this.data.datasets;
        if (!datasets || datasets.length === 0) return;

        const allValues = [];
        for (const ds of datasets) {
            for (const v of (ds.data || [])) {
                if (typeof v === 'number' && !isNaN(v)) allValues.push(v);
            }
        }
        if (allValues.length === 0) return;
        let yMin = Math.min(...allValues);
        let yMax = Math.max(...allValues);
        const range = yMax - yMin;
        if (range < 0.001) { yMin -= 0.5; yMax += 0.5; }
        const yPad = (yMax - yMin) * 0.1 || 0.1;
        yMin -= yPad;
        yMax += yPad;

        this._drawGrid(area, yMin, yMax);

        const n = datasets[0].data ? datasets[0].data.length : 0;
        if (n === 0) return;

        for (let di = 0; di < datasets.length; di++) {
            const ds = datasets[di];
            const pts = ds.data || [];
            ctx.beginPath();
            ctx.strokeStyle = ds.borderColor || '#00c8ff';
            ctx.lineWidth = 1.5;
            const fillColor = ds.backgroundColor || 'rgba(0,200,255,0.1)';

            for (let i = 0; i < pts.length; i++) {
                const x = area.x + (i / Math.max(pts.length - 1, 1)) * area.w;
                const y = area.y + area.h - ((pts[i] - yMin) / (yMax - yMin)) * area.h;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }

            ctx.stroke();

            if (ds.fill !== false) {
                const lastX = area.x + area.w;
                ctx.lineTo(lastX, area.y + area.h);
                ctx.lineTo(area.x, area.y + area.h);
                ctx.closePath();
                ctx.fillStyle = fillColor;
                ctx.fill();
            }

            if (pts.length <= 30) {
                for (let i = 0; i < pts.length; i++) {
                    const x = area.x + (i / Math.max(pts.length - 1, 1)) * area.w;
                    const y = area.y + area.h - ((pts[i] - yMin) / (yMax - yMin)) * area.h;
                    ctx.beginPath();
                    ctx.arc(x, y, 2.5, 0, Math.PI * 2);
                    ctx.fillStyle = ds.borderColor || '#00c8ff';
                    ctx.fill();
                }
            }
        }

        this._drawLabels(area, yMin, yMax);
    }

    /* P1-025修复：散点图渲染方法 - 仅绘制数据点不连线 */
    _drawScatter() {
        const area = this._getDrawArea();
        const ctx = this.ctx;
        const datasets = this.data.datasets;
        if (!datasets || datasets.length === 0) return;

        /* P1-025: 收集所有数据值计算Y轴范围 */
        const allValues = [];
        for (const ds of datasets) {
            for (const v of (ds.data || [])) {
                if (typeof v === 'number' && !isNaN(v)) allValues.push(v);
            }
        }
        if (allValues.length === 0) return;
        let yMin = Math.min(...allValues);
        let yMax = Math.max(...allValues);
        const range = yMax - yMin;
        if (range < 0.001) { yMin -= 0.5; yMax += 0.5; }
        const yPad = (yMax - yMin) * 0.1 || 0.1;
        yMin -= yPad;
        yMax += yPad;

        this._drawGrid(area, yMin, yMax);

        const n = datasets[0].data ? datasets[0].data.length : 0;
        if (n === 0) return;

        /* P1-025: 支持点半径和透明度配置 */
        for (let di = 0; di < datasets.length; di++) {
            const ds = datasets[di];
            const pts = ds.data || [];
            const pointRadius = ds.pointRadius || (this.options.pointRadius || 4);
            const pointColor = ds.borderColor || ds.backgroundColor || '#00c8ff';
            const pointAlpha = ds.pointAlpha !== undefined ? ds.pointAlpha : 0.8;

            ctx.save();
            ctx.globalAlpha = pointAlpha;

            for (let i = 0; i < pts.length; i++) {
                const x = area.x + (i / Math.max(pts.length - 1, 1)) * area.w;
                const y = area.y + area.h - ((pts[i] - yMin) / (yMax - yMin)) * area.h;

                /* P1-025: 外圈光晕效果 */
                ctx.beginPath();
                ctx.arc(x, y, pointRadius + 1.5, 0, Math.PI * 2);
                ctx.fillStyle = 'rgba(0,200,255,0.15)';
                ctx.fill();

                /* P1-025: 实心数据点 */
                ctx.beginPath();
                ctx.arc(x, y, pointRadius, 0, Math.PI * 2);
                ctx.fillStyle = pointColor;
                ctx.fill();

                /* P1-025: 点描边增强可见性 */
                ctx.strokeStyle = 'rgba(255,255,255,0.3)';
                ctx.lineWidth = 1;
                ctx.stroke();
            }

            ctx.restore();
        }

        this._drawLabels(area, yMin, yMax);
    }

    /* P1-025修复：面积图渲染方法 - 填充面积区域并绘制渐变 */
    _drawArea() {
        const area = this._getDrawArea();
        const ctx = this.ctx;
        const datasets = this.data.datasets;
        if (!datasets || datasets.length === 0) return;

        const allValues = [];
        for (const ds of datasets) {
            for (const v of (ds.data || [])) {
                if (typeof v === 'number' && !isNaN(v)) allValues.push(v);
            }
        }
        if (allValues.length === 0) return;
        let yMin = Math.min(...allValues);
        let yMax = Math.max(...allValues);
        const range = yMax - yMin;
        if (range < 0.001) { yMin -= 0.5; yMax += 0.5; }
        const yPad = (yMax - yMin) * 0.1 || 0.1;
        yMin -= yPad;
        yMax += yPad;

        this._drawGrid(area, yMin, yMax);

        const n = datasets[0].data ? datasets[0].data.length : 0;
        if (n === 0) return;

        /* P1-025: 预计算第一组数据集的渐变高度用于渐变创建 */
        const firstPts = datasets[0].data || [];
        const baselineY = area.y + area.h;

        for (let di = 0; di < datasets.length; di++) {
            const ds = datasets[di];
            const pts = ds.data || [];
            const borderColor = ds.borderColor || '#00c8ff';

            /* P1-025: 面积图始终填充，创建垂直渐变使颜色从顶部到底部逐渐淡化 */
            ctx.beginPath();

            for (let i = 0; i < pts.length; i++) {
                const x = area.x + (i / Math.max(pts.length - 1, 1)) * area.w;
                const y = area.y + area.h - ((pts[i] - yMin) / (yMax - yMin)) * area.h;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }

            /* P1-025: 闭合到基线形成面积区域 */
            const lastX = area.x + area.w;
            ctx.lineTo(lastX, baselineY);
            ctx.lineTo(area.x, baselineY);
            ctx.closePath();

            /* P1-025: 使用线性渐变填充面积区域 */
            const gradient = ctx.createLinearGradient(0, area.y, 0, baselineY);
            const baseColor = ds.backgroundColor || borderColor;
            gradient.addColorStop(0, baseColor);
            gradient.addColorStop(1, 'rgba(0,0,0,0)');
            ctx.fillStyle = gradient;
            ctx.fill();

            /* P1-025: 在上层重新绘制轮廓线 */
            ctx.beginPath();
            for (let i = 0; i < pts.length; i++) {
                const x = area.x + (i / Math.max(pts.length - 1, 1)) * area.w;
                const y = area.y + area.h - ((pts[i] - yMin) / (yMax - yMin)) * area.h;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.strokeStyle = borderColor;
            ctx.lineWidth = 1.5;
            ctx.stroke();
        }

        this._drawLabels(area, yMin, yMax);
    }

    _drawGrid(area, yMin, yMax) {
        const ctx = this.ctx;
        const ticks = 5;
        ctx.strokeStyle = 'rgba(255,255,255,0.06)';
        ctx.lineWidth = 1;
        for (let i = 0; i <= ticks; i++) {
            const y = area.y + (i / ticks) * area.h;
            ctx.beginPath();
            ctx.moveTo(area.x, y);
            ctx.lineTo(area.x + area.w, y);
            ctx.stroke();
        }
    }

    _drawLabels(area, yMin, yMax) {
        const ctx = this.ctx;
        ctx.fillStyle = 'rgba(255,255,255,0.35)';
        ctx.font = '9px monospace';
        ctx.textAlign = 'right';
        const ticks = 5;
        for (let i = 0; i <= ticks; i++) {
            const y = area.y + (i / ticks) * area.h;
            const val = yMax - (i / ticks) * (yMax - yMin);
            const label = val >= 1000 ? (val / 1000).toFixed(1) + 'k' :
                          val >= 1 ? val.toFixed(1) : val.toFixed(3);
            ctx.fillText(label, area.x - 5, y + 3);
        }
    }

    _drawBar() {
        const area = this._getDrawArea();
        const ctx = this.ctx;
        const datasets = this.data.datasets;
        if (!datasets || datasets.length === 0) return;

        const allValues = [];
        for (const ds of datasets) {
            for (const v of (ds.data || [])) {
                if (typeof v === 'number' && !isNaN(v)) allValues.push(v);
            }
        }
        if (allValues.length === 0) return;
        const yMax = Math.max(...allValues) * 1.15 || 1;
        const yMin = 0;

        this._drawGrid(area, yMin, yMax);

        const n = datasets[0].data ? datasets[0].data.length : 0;
        const barWidth = (area.w / n) * 0.7;
        const gap = (area.w / n) * 0.3;

        for (let di = 0; di < datasets.length; di++) {
            const ds = datasets[di];
            const pts = ds.data || [];
            const color = ds.backgroundColor || '#00c8ff';

            for (let i = 0; i < pts.length; i++) {
                const x = area.x + i * (barWidth + gap) + gap / 2;
                const barH = (pts[i] / yMax) * area.h;
                const y = area.y + area.h - barH;

                ctx.fillStyle = color;
                ctx.fillRect(x, y, barWidth, barH);

                ctx.strokeStyle = 'rgba(255,255,255,0.1)';
                ctx.lineWidth = 1;
                ctx.strokeRect(x, y, barWidth, barH);
            }
        }

        this._drawLabels(area, yMin, yMax);
    }

    _drawRadar() {
        const ctx = this.ctx;
        const cx = this._w / 2;
        const cy = this._h / 2;
        const radius = Math.min(cx, cy) - 20;
        const datasets = this.data.datasets;
        const labels = this.data.labels || [];
        const n = labels.length;
        if (n < 3 || !datasets || datasets.length === 0) return;

        ctx.strokeStyle = 'rgba(255,255,255,0.1)';
        ctx.lineWidth = 1;
        for (let i = 1; i <= 5; i++) {
            ctx.beginPath();
            for (let j = 0; j < n; j++) {
                const angle = (Math.PI * 2 * j) / n - Math.PI / 2;
                const r = (radius * i) / 5;
                const x = cx + Math.cos(angle) * r;
                const y = cy + Math.sin(angle) * r;
                if (j === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.closePath();
            ctx.stroke();
        }

        for (let j = 0; j < n; j++) {
            const angle = (Math.PI * 2 * j) / n - Math.PI / 2;
            ctx.beginPath();
            ctx.moveTo(cx, cy);
            ctx.lineTo(cx + Math.cos(angle) * radius, cy + Math.sin(angle) * radius);
            ctx.stroke();
        }

        for (let di = 0; di < datasets.length; di++) {
            const ds = datasets[di];
            const values = ds.data || [];
            const color = ds.borderColor || '#00c8ff';
            const fillColor = ds.backgroundColor || 'rgba(0,200,255,0.15)';

            ctx.beginPath();
            for (let j = 0; j < n; j++) {
                const angle = (Math.PI * 2 * j) / n - Math.PI / 2;
                const val = Math.min(Math.max(values[j] || 0, 0), 1);
                const r = radius * val;
                const x = cx + Math.cos(angle) * r;
                const y = cy + Math.sin(angle) * r;
                if (j === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.closePath();
            ctx.fillStyle = fillColor;
            ctx.fill();
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.stroke();

            for (let j = 0; j < n; j++) {
                const angle = (Math.PI * 2 * j) / n - Math.PI / 2;
                const val = Math.min(Math.max(values[j] || 0, 0), 1);
                const r = radius * val;
                const x = cx + Math.cos(angle) * r;
                const y = cy + Math.sin(angle) * r;
                ctx.beginPath();
                ctx.arc(x, y, 3, 0, Math.PI * 2);
                ctx.fillStyle = color;
                ctx.fill();
            }
        }

        ctx.fillStyle = 'rgba(255,255,255,0.5)';
        ctx.font = '10px monospace';
        ctx.textAlign = 'center';
        for (let j = 0; j < n; j++) {
            const angle = (Math.PI * 2 * j) / n - Math.PI / 2;
            const x = cx + Math.cos(angle) * (radius + 14);
            const y = cy + Math.sin(angle) * (radius + 14);
            ctx.fillText(labels[j] || '', x, y + 4);
        }
    }

    /* P1-025修复：饼图渲染方法 - 扇形绘制带标签和百分比 */
    _drawPie() {
        const ctx = this.ctx;
        const cx = this._w / 2;
        const cy = this._h / 2;
        const radius = Math.min(cx, cy) - 30;
        const datasets = this.data.datasets;
        const labels = this.data.labels || [];
        if (!datasets || datasets.length === 0) return;
        const values = datasets[0].data || [];
        if (values.length === 0) return;
        const total = values.reduce((a, b) => a + b, 0);
        if (total <= 0) return;
        const colors = [
            '#00c8ff', '#00ff88', '#ff6600', '#ffcc00', '#cc44ff',
            '#ff4488', '#44ccff', '#88ff44', '#ff8844', '#ffdd44'
        ];
        let startAngle = -Math.PI / 2;
        const slices = [];
        for (let i = 0; i < values.length; i++) {
            const sweep = (values[i] / total) * Math.PI * 2;
            const midAngle = startAngle + sweep / 2;
            slices.push({ startAngle, sweep, midAngle, value: values[i], pct: (values[i] / total * 100), label: labels[i] || '', color: datasets[0].backgroundColor ? (Array.isArray(datasets[0].backgroundColor) ? datasets[0].backgroundColor[i] : datasets[0].backgroundColor) : colors[i % colors.length] });
            startAngle += sweep;
        }
        /* 阴影底座 */
        for (const s of slices) {
            ctx.beginPath();
            ctx.moveTo(cx, cy);
            ctx.arc(cx, cy + 3, radius, s.startAngle, s.startAngle + s.sweep);
            ctx.closePath();
            ctx.fillStyle = 'rgba(0,0,0,0.3)';
            ctx.fill();
        }
        /* 扇形 */
        for (const s of slices) {
            ctx.beginPath();
            ctx.moveTo(cx, cy);
            ctx.arc(cx, cy, radius, s.startAngle, s.startAngle + s.sweep);
            ctx.closePath();
            ctx.fillStyle = s.color;
            ctx.fill();
            ctx.strokeStyle = 'rgba(0,0,0,0.3)';
            ctx.lineWidth = 1;
            ctx.stroke();
        }
        /* 标签和百分比 */
        ctx.fillStyle = 'rgba(255,255,255,0.85)';
        ctx.font = '11px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        for (const s of slices) {
            if (s.pct < 3) continue;
            const lx = cx + Math.cos(s.midAngle) * radius * 0.7;
            const ly = cy + Math.sin(s.midAngle) * radius * 0.7;
            ctx.fillText(s.pct.toFixed(1) + '%', lx, ly - 8);
            if (s.label && s.pct >= 8) {
                ctx.fillText(s.label, lx, ly + 10);
            }
        }
    }

    /* P1-025修复：热力图渲染方法 - 网格着色带色标 */
    _drawHeatmap() {
        const ctx = this.ctx;
        const area = this._getDrawArea();
        const datasets = this.data.datasets;
        const labels = this.data.labels || [];
        if (!datasets || datasets.length === 0) return;
        const rows = datasets.length;
        const cols = (datasets[0].data && datasets[0].data.length) || 0;
        if (rows === 0 || cols === 0) return;
        let minVal = Infinity, maxVal = -Infinity;
        for (const ds of datasets) {
            for (const v of (ds.data || [])) {
                if (typeof v === 'number' && !isNaN(v)) {
                    if (v < minVal) minVal = v;
                    if (v > maxVal) maxVal = v;
                }
            }
        }
        if (!isFinite(minVal)) { minVal = 0; maxVal = 1; }
        if (maxVal - minVal < 0.001) maxVal = minVal + 1;
        const cellW = area.w / cols;
        const cellH = area.h / rows;
        const heatColors = [
            [0, 0, 0.5], [0, 0, 1], [0, 0.5, 1], [0, 1, 1],
            [0, 1, 0.5], [0, 1, 0], [0.5, 1, 0], [1, 1, 0],
            [1, 0.5, 0], [1, 0, 0]
        ];
        const interpHeat = (t) => {
            const idx = t * (heatColors.length - 1);
            const lo = Math.floor(Math.min(idx, heatColors.length - 2));
            const frac = idx - lo;
            const a = heatColors[lo], b = heatColors[lo + 1];
            return [a[0] + (b[0] - a[0]) * frac, a[1] + (b[1] - a[1]) * frac, a[2] + (b[2] - a[2]) * frac];
        };
        for (let ri = 0; ri < rows; ri++) {
            for (let ci = 0; ci < cols; ci++) {
                if (!datasets[ri] || !datasets[ri].data) continue;
                const v = datasets[ri].data[ci];
                if (typeof v !== 'number' || isNaN(v)) continue;
                const t = (v - minVal) / (maxVal - minVal);
                const [r, g, b] = interpHeat(Math.max(0, Math.min(1, t)));
                ctx.fillStyle = 'rgb(' + Math.round(r * 255) + ',' + Math.round(g * 255) + ',' + Math.round(b * 255) + ')';
                ctx.fillRect(area.x + ci * cellW, area.y + ri * cellH, cellW, cellH);
                ctx.strokeStyle = 'rgba(255,255,255,0.05)';
                ctx.lineWidth = 0.5;
                ctx.strokeRect(area.x + ci * cellW, area.y + ri * cellH, cellW, cellH);
            }
        }
        /* 色标 */
        const barW = 12, barH = area.h;
        const barX = area.x + area.w + 10;
        const barY = area.y;
        const steps = 20;
        for (let i = 0; i < steps; i++) {
            const t = i / (steps - 1);
            const [r, g, b] = interpHeat(1 - t);
            ctx.fillStyle = 'rgb(' + Math.round(r * 255) + ',' + Math.round(g * 255) + ',' + Math.round(b * 255) + ')';
            ctx.fillRect(barX, barY + i * (barH / steps), barW, barH / steps);
        }
        ctx.fillStyle = 'rgba(255,255,255,0.5)';
        ctx.font = '9px monospace';
        ctx.textAlign = 'left';
        ctx.fillText(maxVal.toFixed(2), barX + barW + 4, barY + 8);
        ctx.fillText(minVal.toFixed(2), barX + barW + 4, barY + barH);
    }

    /* P1-025修复：仪表盘渲染方法 - 半圆弧带指针和刻度 */
    _drawGauge() {
        const ctx = this.ctx;
        const cx = this._w / 2;
        const cy = this._h * 0.75;
        const radius = Math.min(cx, cy) - 10;
        const datasets = this.data.datasets;
        if (!datasets || datasets.length === 0) return;
        const value = datasets[0].data ? datasets[0].data[0] : 0;
        const minVal = this.options.gaugeMin !== undefined ? this.options.gaugeMin : 0;
        const maxVal = this.options.gaugeMax !== undefined ? this.options.gaugeMax : 100;
        const range = maxVal - minVal;
        if (range <= 0) return;
        const t = Math.max(0, Math.min(1, (value - minVal) / range));
        /* 刻度弧背景 */
        ctx.beginPath();
        ctx.arc(cx, cy, radius, Math.PI, 0);
        ctx.strokeStyle = 'rgba(255,255,255,0.08)';
        ctx.lineWidth = radius * 0.25;
        ctx.stroke();
        /* 值弧 */
        const g2 = ctx.createLinearGradient(cx - radius, 0, cx + radius, 0);
        g2.addColorStop(0, '#00ff88');
        g2.addColorStop(0.5, '#ffcc00');
        g2.addColorStop(1, '#ff4400');
        ctx.beginPath();
        ctx.arc(cx, cy, radius, Math.PI, Math.PI + t * Math.PI);
        ctx.strokeStyle = g2;
        ctx.lineWidth = radius * 0.22;
        ctx.stroke();
        /* 指针 */
        const a = Math.PI + t * Math.PI;
        const px = cx + Math.cos(a) * radius * 0.85;
        const py = cy + Math.sin(a) * radius * 0.85;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(px, py);
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2.5;
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(cx, cy, 5, 0, Math.PI * 2);
        ctx.fillStyle = '#ffffff';
        ctx.fill();
        /* 刻度标签 */
        ctx.fillStyle = 'rgba(255,255,255,0.6)';
        ctx.font = '10px monospace';
        ctx.textAlign = 'center';
        const ticks = 5;
        for (let i = 0; i <= ticks; i++) {
            const tickT = i / ticks;
            const ta = Math.PI + tickT * Math.PI;
            const tx = cx + Math.cos(ta) * radius * 0.65;
            const ty = cy + Math.sin(ta) * radius * 0.65;
            ctx.fillText((minVal + tickT * range).toFixed(0), tx, ty);
        }
        /* 中心值 */
        ctx.fillStyle = '#ffffff';
        ctx.font = (radius * 0.18) + 'px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(value.toFixed(1), cx, cy + radius * 0.3);
        const title = this.options.gaugeTitle || '';
        if (title) {
            ctx.font = '12px monospace';
            ctx.fillText(title, cx, cy + radius * 0.55);
        }
    }

    destroy() {
        if (this._resizeObserver) {
            this._resizeObserver.disconnect();
            this._resizeObserver = null;
        }
        if (this._animFrame) {
            cancelAnimationFrame(this._animFrame);
            this._animFrame = null;
        }
        this.ctx = null;
        this.canvas = null;
    }
}
