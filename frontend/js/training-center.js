(function() {
    'use strict';
    var trainingPollInterval = null;
    var trainingHistoryData = [];
    var hyperparamPoll = null;
    var selectedCheckpoint = null;
    var gpuList = [];
    var datasetIndex = [];
    var convergenceHistory = [];

    function stopPolling() { if (trainingPollInterval) { clearInterval(trainingPollInterval); trainingPollInterval = null; } }
    function stopHyperparamPoll() { if (hyperparamPoll) { clearInterval(hyperparamPoll); hyperparamPoll = null; } }

    async function startTraining() {
        try {
            var modeMap = {1:'imitation',2:'rl',3:'primitive',4:'joint',5:'task'};
            var selMode = (window.selectedTrainingMode != null) ? window.selectedTrainingMode : 1;
            var mode = modeMap[selMode] || 'pretrain';
            var lr = 0.001;
            var batch = 64;
            var epochs = 100;
            var datasetPath = '/data/training';
            var lrEl = document.getElementById('train-lr');
            if (lrEl) lr = parseFloat(lrEl.value) || 0.001;
            var batchEl = document.getElementById('train-batch');
            if (batchEl) batch = parseInt(batchEl.value, 10) || 64;
            var epochEl = document.getElementById('train-epochs');
            if (epochEl) epochs = parseInt(epochEl.value, 10) || 100;

            var data = await window.SelfLnnApi.trainingStart(mode, lr, batch, epochs, datasetPath);
            if (data.success) {
                window.showNotification('训练已启动(' + mode + ')', 'success');
                trainingHistoryData = [];
                convergenceHistory = [];
                if (trainingPollInterval) clearInterval(trainingPollInterval);
                trainingPollInterval = setInterval(pollTraining, 2000);
                pollTraining();
            } else {
                window.showNotification('启动失败: ' + (data.error || ''), 'danger');
            }
        } catch(e) { window.showNotification('启动失败: ' + e.message, 'danger'); }
    }
    window.startTraining = startTraining;

    async function pollTraining() {
        try {
            var data = await window.SelfLnnApi.getTrainingStatus();
            if (data) {
                var taskEl = document.getElementById('train-current-task');
                if (taskEl) taskEl.textContent = data.current_stage || data.stage || '--';
                var epochEl = document.getElementById('current-epoch');
                if (epochEl) epochEl.textContent = data.current_epoch || data.epoch || '--';
                var lossEl = document.getElementById('train-loss');
                if (lossEl) {
                    var lossVal = data.current_loss;
                    var accInfo = (data.accuracy !== undefined && data.accuracy !== null) ? (' | 准确率:' + (data.accuracy * 100).toFixed(1) + '%') : '';
                    /* ZSFWS-002: 显示收敛速率 */
                    var convInfo = (data.convergence_rate !== undefined && data.convergence_rate !== null) ? (' | 收敛速率:' + (data.convergence_rate * 100).toFixed(2) + '%/epoch') : '';
                    var timeInfo = data.estimated_time ? (' | 预计:' + data.estimated_time) : '';
                    lossEl.textContent = (typeof lossVal === 'number' ? lossVal.toFixed(4) : String(lossVal || '--')) + accInfo + convInfo + timeInfo;
                }
                var progressEl = document.getElementById('train-progress-fill');
                if (progressEl) progressEl.style.width = data.progress ? data.progress + '%' : '0%';
                var progressText = document.getElementById('train-progress-text');
                if (progressText) progressText.textContent = data.progress ? data.progress.toFixed(1) + '%' : '--';

                /* ZSFWS-002: 记录训练历史数据 */
                if (data.current_epoch !== undefined && data.current_loss !== undefined) {
                    trainingHistoryData.push({
                        epoch: data.current_epoch,
                        loss: data.current_loss,
                        accuracy: data.accuracy || 0,
                        lr: data.learning_rate || 0,
                        convergence_rate: data.convergence_rate || 0
                    });
                    if (trainingHistoryData.length > 500) trainingHistoryData.shift();
                    renderTrainingChart();
                }

                if (data.running === false) stopPolling();
            }
        } catch(e) { console.warn('训练轮询失败:', e.message); }
    }

    /* ZSFWS-002: 训练历史图表渲染 */
    function renderTrainingChart() {
        var canvas = document.getElementById('train-history-chart');
        if (!canvas || trainingHistoryData.length < 2) {
            var placeholder = document.getElementById('train-history-placeholder');
            if (placeholder) placeholder.style.display = 'block';
            return;
        }
        var placeholder = document.getElementById('train-history-placeholder');
        if (placeholder) placeholder.style.display = 'none';
        var ctx = canvas.getContext('2d');
        var w = canvas.width || 400, h = canvas.height || 200;
        canvas.width = w; canvas.height = h;
        ctx.clearRect(0, 0, w, h);

        /* 背景 */
        ctx.fillStyle = '#0a0a14';
        ctx.fillRect(0, 0, w, h);

        /* 计算边界 */
        var margin = { top: 10, right: 60, bottom: 25, left: 45 };
        var pw = w - margin.left - margin.right, ph = h - margin.top - margin.bottom;
        var losses = trainingHistoryData.map(function(d) { return d.loss; });
        var minLoss = Math.min.apply(null, losses), maxLoss = Math.max.apply(null, losses);
        var range = maxLoss - minLoss || 1;
        var yScale = ph / range;
        var xScale = pw / Math.max(trainingHistoryData.length - 1, 1);

        /* 网格 */
        ctx.strokeStyle = '#1a1a2e';
        ctx.lineWidth = 0.5;
        for (var i = 0; i <= 4; i++) {
            var gy = margin.top + ph * i / 4;
            ctx.beginPath(); ctx.moveTo(margin.left, gy); ctx.lineTo(w - margin.right, gy); ctx.stroke();
        }

        /* 损失曲线 */
        ctx.strokeStyle = '#00d4ff';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        for (var j = 0; j < trainingHistoryData.length; j++) {
            var x = margin.left + j * xScale;
            var y = margin.top + (maxLoss - trainingHistoryData[j].loss) * yScale;
            if (j === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();

        /* 标签 */
        ctx.fillStyle = '#888';
        ctx.font = '10px monospace';
        ctx.fillText((minLoss).toFixed(3), margin.left + 2, margin.top + ph);
        ctx.fillText((maxLoss).toFixed(3), margin.left + 2, margin.top + 10);
        ctx.fillText('Loss', 2, margin.top + ph / 2 + 4);
        ctx.fillText('Epoch ' + (trainingHistoryData.length - 1), w - margin.right - 30, h - 3);
    }

    /* ZSFWS-002: 超参数搜索 */
    async function startHyperparameterSearch() {
        try {
            var gridSize = parseInt((document.getElementById('hyper-grid-size') || {}).value || '10', 10);
            /* FIX-7: search_type→method, grid_size→max_trials 匹配后端 */
            var params = { method: 'grid', max_trials: gridSize };
            var data = await window.SelfLnnApi.hyperparameterStart(params);
            if (data.success) {
                window.showNotification('超参数搜索已启动(网格搜索)', 'success');
                if (hyperparamPoll) clearInterval(hyperparamPoll);
                hyperparamPoll = setInterval(pollHyperparameterSearch, 3000);
            } else {
                window.showNotification('超参数搜索启动失败', 'danger');
            }
        } catch(e) { window.showNotification('启动失败: ' + e.message, 'danger'); }
    }
    window.startHyperparameterSearch = startHyperparameterSearch;

    async function pollHyperparameterSearch() {
        try {
            var data = await window.SelfLnnApi.hyperparameterStatus();
            if (data.success && data.data) {
                var d = data.data;
                var statusEl = document.getElementById('hyper-status');
                if (statusEl) statusEl.textContent = d.status || '搜索中...';
                var bestEl = document.getElementById('hyper-best');
                if (bestEl && d.best_score !== undefined) bestEl.textContent = '最佳: ' + d.best_score.toFixed(4);
                if (d.status === 'completed') stopHyperparamPoll();
            }
        } catch(e) { console.warn('超参数轮询失败:', e.message); }
    }
    window.pollHyperparameterSearch = pollHyperparameterSearch;

    /* ZSFWS-002: GPU资源选择 */
    async function refreshGpuList() {
        try {
            var data = await window.SelfLnnApi.getGpuStatus();
            if (data && data.backends) {
                gpuList = data.backends;
                renderGpuSelector();
            }
        } catch(e) { console.warn('GPU列表获取失败:', e.message); }
    }
    window.refreshGpuList = refreshGpuList;

    function renderGpuSelector() {
        var container = document.getElementById('gpu-selector');
        if (!container) return;
        var html = '<option value="auto">自动选择</option>';
        for (var i = 0; i < gpuList.length; i++) {
            var g = gpuList[i];
            html += '<option value="' + (g.name || g.backend || '') + '">' +
                (g.label || g.name || g.backend || ('GPU ' + i)) +
                (g.available ? '' : ' (不可用)') + '</option>';
        }
        container.innerHTML = html;
    }

    /* ZSFWS-002: 检查点列表管理 */
    async function loadCheckpointList() {
        try {
            var data = await window.SelfLnnApi.request('/checkpoint/list');
            if (data.ok) {
                var d = await data.json();
                renderCheckpointList(d.checkpoints || d.list || []);
            }
        } catch(e) { console.warn('检查点列表加载失败:', e.message); }
    }
    window.loadCheckpointList = loadCheckpointList;

    function renderCheckpointList(checkpoints) {
        var container = document.getElementById('checkpoint-list');
        if (!container) return;
        if (checkpoints.length === 0) {
            container.innerHTML = '<div class="empty-state">暂无检查点</div>';
            return;
        }
        var html = '<table class="checkpoint-table"><tr><th>名称</th><th>损失</th><th>准确率</th><th>时间</th><th>操作</th></tr>';
        for (var i = 0; i < checkpoints.length; i++) {
            var cp = checkpoints[i];
            html += '<tr onclick="window.selectCheckpoint(\'' + (cp.id || cp.filename || '') + '\')">' +
                '<td>' + (cp.name || cp.filename || '检查点' + i) + '</td>' +
                '<td>' + (typeof cp.loss === 'number' ? cp.loss.toFixed(4) : '--') + '</td>' +
                '<td>' + (typeof cp.accuracy === 'number' ? (cp.accuracy * 100).toFixed(1) + '%' : '--') + '</td>' +
                '<td>' + (cp.timestamp || '--') + '</td>' +
                '<td><button onclick="event.stopPropagation();window.loadCheckpoint(\'' + (cp.id || cp.filename || '') + '\')">加载</button></td>' +
                '</tr>';
        }
        html += '</table>';
        container.innerHTML = html;
    }

    window.selectCheckpoint = function(id) {
        selectedCheckpoint = id;
        var rows = document.querySelectorAll('#checkpoint-list tr');
        for (var i = 0; i < rows.length; i++) rows[i].classList.remove('selected');
        var el = document.querySelector('#checkpoint-list tr[data-id="' + id + '"]');
        if (el) el.classList.add('selected');
    };

    window.loadCheckpoint = async function(id) {
        try {
            var data = await window.SelfLnnApi.request('/checkpoint/load', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({checkpoint_id: id})
            });
            if (data.ok) {
                window.showNotification('检查点已加载', 'success');
            } else {
                window.showNotification('加载失败', 'danger');
            }
        } catch(e) { window.showNotification('操作失败: ' + e.message, 'danger'); }
    };

    /* ZSFWS-002: 数据集管理 */
    async function loadDatasetList() {
        try {
            var data = await window.SelfLnnApi.datasetList();
            if (data.success) {
                datasetIndex = data.data.datasets || data.data || [];
                renderDatasetList();
            }
        } catch(e) { console.warn('数据集列表加载失败:', e.message); }
    }
    window.loadDatasetList = loadDatasetList;

    function renderDatasetList() {
        var container = document.getElementById('dataset-list');
        if (!container) return;
        if (datasetIndex.length === 0) {
            container.innerHTML = '<div class="empty-state">暂无数据集</div>';
            return;
        }
        var html = '';
        for (var i = 0; i < datasetIndex.length; i++) {
            var ds = datasetIndex[i];
            html += '<div class="dataset-item"><span>' + (ds.name || '数据集' + i) + '</span>' +
                '<span class="ds-info">' + (ds.samples || ds.count || '?') + '样本</span>' +
                '<button onclick="window.useDataset(\'' + (ds.name || '') + '\')">使用</button></div>';
        }
        container.innerHTML = html;
    }

    window.useDataset = function(name) {
        var el = document.getElementById('train-dataset');
        if (el) el.value = name || '';
        window.showNotification('数据集 ' + name + ' 已选择', 'info');
    };

    /* 训练历史导出 */
    window.exportTrainingHistory = function() {
        if (trainingHistoryData.length === 0) {
            window.showNotification('无训练历史数据', 'warning');
            return;
        }
        var csv = 'epoch,loss,accuracy,learning_rate,convergence_rate\n';
        for (var i = 0; i < trainingHistoryData.length; i++) {
            var d = trainingHistoryData[i];
            csv += d.epoch + ',' + d.loss.toFixed(6) + ',' + d.accuracy.toFixed(6) + ',' + d.lr.toFixed(8) + ',' + d.convergence_rate.toFixed(6) + '\n';
        }
        var blob = new Blob([csv], {type:'text/csv'});
        var url = URL.createObjectURL(blob);
        var a = document.createElement('a');
        a.href = url; a.download = 'training_history_' + Date.now() + '.csv';
        document.body.appendChild(a); a.click(); document.body.removeChild(a);
        URL.revokeObjectURL(url);
        window.showNotification('训练历史已导出', 'success');
    };

    /* ZSFWS-002: 训练配置重置 */
    window.resetTrainingConfig = function() {
        var defaults = { lr: '0.001', batch: '64', epochs: '100', dataset: '/data/training' };
        var lrEl = document.getElementById('train-lr');
        if (lrEl) lrEl.value = defaults.lr;
        var batchEl = document.getElementById('train-batch');
        if (batchEl) batchEl.value = defaults.batch;
        var epochEl = document.getElementById('train-epochs');
        if (epochEl) epochEl.value = defaults.epochs;
        window.showNotification('训练配置已重置', 'info');
    };

    window.pauseTraining = async function() {
        try {
            var data = await window.SelfLnnApi.trainingPause();
            if (data && data.success) {
                window.showNotification('训练已暂停', 'success');
                stopPolling();
            } else {
                window.showNotification(data && data.error ? '暂停失败: ' + data.error : '暂停失败', 'danger');
            }
        } catch(e) { window.showNotification('操作失败', 'danger'); console.error('[训练中心] pauseTraining失败:', e&&e.message?e.message:e); }
    };
    window.resumeTraining = async function() {
        try {
            if (typeof window.SelfLnnApi === 'undefined') { window.showNotification('API服务未就绪', 'danger'); return; }
            var data = await window.SelfLnnApi.trainingResume();
            var ok = data && data.success;
            window.showNotification(ok ? '训练已恢复' : '恢复失败', ok ? 'success' : 'danger');
            if (trainingPollInterval) clearInterval(trainingPollInterval);
            trainingPollInterval = setInterval(pollTraining, 2000);
            pollTraining();
        } catch(e) { console.error('[训练中心] resumeTraining失败:', e.message); window.showNotification('操作失败', 'danger'); }
    };
    window.stopTraining = async function() {
        try {
            var data = await window.SelfLnnApi.trainingStop();
            stopPolling();
            stopHyperparamPoll();
            var ok = data && data.success;
            window.showNotification(ok ? '训练已停止' : '停止失败', ok ? 'success' : 'danger');
        } catch(e) { console.error('[训练中心] stopTraining失败:', e.message); window.showNotification('操作失败', 'danger'); }
    };

    window.pollTraining = pollTraining;

    /* 页面卸载时清理 */
    window.addEventListener('beforeunload', function() {
        stopPolling();
        stopHyperparamPoll();
    });

    /* 初始化：加载GPU列表和检查点 */
    if (document.readyState === 'complete' || document.readyState === 'interactive') {
        setTimeout(function() {
            refreshGpuList();
            loadCheckpointList();
        }, 1000);
    } else {
        document.addEventListener('DOMContentLoaded', function() {
            setTimeout(function() {
                refreshGpuList();
                loadCheckpointList();
            }, 1000);
        });
    }
})();
