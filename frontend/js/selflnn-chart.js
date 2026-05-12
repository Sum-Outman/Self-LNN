
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
