(function() {
    var trainingPolling = null;
    var trainingWs = null;
    var trainingChart = null;
    var chartData = { losses: [], accs: [], maxPoints: 100 };
    var wsReconnectAttempt = 0;
    var wsReconnectMax = 10;
    var wsReconnectBaseMs = 1000;
    var wsReconnectMaxMs = 30000;
    var wsHeartbeatTimer = null;
    var wsHeartbeatIntervalMs = 15000;

    function calcBackoff(attempt) {
        var delay = Math.min(wsReconnectBaseMs * Math.pow(2, attempt), wsReconnectMaxMs);
        var jitter = Math.random() * 1000;
        return Math.floor(delay + jitter);
    }

    function startHeartbeat() {
        stopHeartbeat();
        wsHeartbeatTimer = setInterval(function() {
            if (trainingWs && trainingWs.readyState === WebSocket.OPEN) {
                trainingWs.send(JSON.stringify({ type: 'ping', ts: Date.now() }));
            }
        }, wsHeartbeatIntervalMs);
    }

    function stopHeartbeat() {
        if (wsHeartbeatTimer) { clearInterval(wsHeartbeatTimer); wsHeartbeatTimer = null; }
    }

    function startPolling() {
        stopPolling();
        if (window.g_dataEngine) {
            window.g_dataEngine.registerModule('training_center', 2000, pollTraining);
        } else {
            trainingPolling = setInterval(pollTraining, 2000);
        }
    }

    function stopPolling() {
        if (trainingPolling) {
            clearInterval(trainingPolling);
            trainingPolling = null;
        }
        if (window.g_dataEngine) {
            window.g_dataEngine.unregisterModule('training_center');
        }
    }

    function initChart() {
        var canvas = document.getElementById('training-chart');
        if (!canvas || !window.SelfLnnChart) return;
        trainingChart = new SelfLnnChart(canvas, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    { label: '损失', borderColor: '#e74c3c', backgroundColor: 'rgba(231,76,60,0.1)', data: chartData.losses, fill: true },
                    { label: '准确率', borderColor: '#2ecc71', backgroundColor: 'rgba(46,204,113,0.1)', data: chartData.accs, fill: true }
                ]
            }
        });
    }

    function updateTrainingUI(tr) {
        var el;
        el = document.getElementById('current-epoch');
        if (el) el.textContent = tr.current_epoch != null ? tr.current_epoch : '-';
        el = document.getElementById('train-loss');
        if (el) el.textContent = tr.train_loss != null ? tr.train_loss.toFixed(6) : '-';
        el = document.getElementById('val-loss');
        if (el) el.textContent = tr.val_loss != null ? tr.val_loss.toFixed(6) : '-';
        el = document.getElementById('train-accuracy');
        if (el) el.textContent = tr.accuracy != null ? (tr.accuracy * 100).toFixed(2) + '%' : '-';
        el = document.getElementById('train-time');
        if (el) el.textContent = tr.elapsed_time ? tr.elapsed_time + 's' : '-';

        if (tr.train_loss != null && tr.accuracy != null) {
            chartData.losses.push(tr.train_loss);
            chartData.accs.push(tr.accuracy);
            if (chartData.losses.length > chartData.maxPoints) { chartData.losses.shift(); chartData.accs.shift(); }
            if (trainingChart) trainingChart.draw();
        }
    }

    async function startTraining() {
        var mode = document.getElementById('train-mode').value;
        var lr = parseFloat(document.getElementById('train-lr').value);
        var batch = parseInt(document.getElementById('train-batch').value);
        var epochs = parseInt(document.getElementById('train-epochs').value);
        var dataPath = document.getElementById('train-data').value;
        var config = { mode: mode, learning_rate: lr, batch_size: batch, num_epochs: epochs, dataset_path: dataPath };
        var modeMap = { 'pretrain': 'trainingPretrain', 'finetune': 'trainingFineTune', 'rl': 'trainingContinual', 'contrastive': 'trainingTransfer', 'curriculum': 'trainingFromScratch' };
        var methodName = modeMap[mode] || 'trainingPretrain';
        try {
            var data = await SelfLnnApi[methodName](config);
            if (data.training && data.training.status !== 'error') {
                window.showNotification('训练任务已启动', 'success');
                startPolling();
                pollTraining();
            } else { window.showNotification('启动失败: ' + (data.error || '未知错误'), 'danger'); }
        } catch (e) { window.showNotification('连接失败: ' + e.message, 'danger'); }
    }

    async function pollTraining() {
        try {
            var data = await SelfLnnApi.trainingStatus();
            var tr = data.training || data;
            if (tr.status === 'error' || tr.status === 'stopped') {
                stopPolling();
            }
            updateTrainingUI(tr);
            /* 后端连接成功，清除离线提示 */
            var offlineEl = document.getElementById('tc-offline-notice');
            if (offlineEl) offlineEl.style.display = 'none';
        } catch (e) {
            console.warn('训练轮询失败:', e.message);
            /* W-006: 后端未连接时显示明确离线提示（禁止虚假数据） */
            showTrainingOffline(e.message);
        }
    }

    /* W-006: 后端未连接离线状态提示 */
    function showTrainingOffline(reason) {
        var el = document.getElementById('tc-offline-notice');
        if (!el) {
            el = document.createElement('div');
            el.id = 'tc-offline-notice';
            el.style.cssText = 'background:#fff3cd;color:#856404;padding:12px 16px;' +
                'margin:12px 0;border-radius:6px;border:1px solid #ffc107;text-align:center;';
            var container = document.querySelector('.training-container') ||
                           document.getElementById('training-center') ||
                           document.body;
            if (container && container.firstChild) {
                container.insertBefore(el, container.firstChild);
            }
        }
        el.style.display = 'block';
        el.innerHTML = '⚠️ <strong>后端服务未连接</strong> — ' +
            (reason || '无法获取训练数据') +
            '（符合\"禁止任何虚假数据\"原则，不显示模拟数据）';
    }

    function connectTrainingWs() {
        if (trainingWs && (trainingWs.readyState === WebSocket.CONNECTING || trainingWs.readyState === WebSocket.OPEN)) {
            return;
        }
        /* ZSFABC-011修复: WebSocket端口统一使用SELFLNN_CONFIG.port，而非浏览器页面端口 */
        var scheme = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        var port = (window.SELFLNN_CONFIG && window.SELFLNN_CONFIG.port) || 8080;
        trainingWs = new WebSocket(scheme + '//' + (window.SELFLNN_CONFIG && window.SELFLNN_CONFIG.host || window.location.hostname) + ':' + port + '/ws');
        trainingWs.onopen = function () {
            wsReconnectAttempt = 0;
            var el = document.getElementById('ws-status');
            if (el) { el.textContent = 'WS已连接'; el.style.color = '#00ff88'; }
            stopPolling();
            startHeartbeat();
        };
        trainingWs.onmessage = function (evt) {
            try {
                var msg = JSON.parse(evt.data);
                if (msg.type === 'pong') return;
                if (msg.type === 'training_update') {
                    updateTrainingUI(msg);
                } else if (msg.type === 'system_status' && msg.data && msg.data.training) {
                    var t = msg.data.training;
                    updateTrainingUI({ current_epoch: t.epoch, train_loss: t.train_loss != null ? t.train_loss : null, val_loss: t.val_loss, accuracy: t.accuracy, elapsed_time: t.elapsed_time });
                }
            } catch (e) { /* WebSocket消息非JSON，忽略 */ }
        };
        trainingWs.onerror = function () {
            stopHeartbeat();
            var el = document.getElementById('ws-status');
            if (el) { el.textContent = 'WS错误(轮询模式)'; el.style.color = '#ffaa00'; }
            if (!trainingPolling && !window.g_dataEngine) { trainingPolling = setInterval(pollTraining, 2000); }
            if (window.g_dataEngine) { startPolling(); }
        };
        trainingWs.onclose = function () {
            stopHeartbeat();
            var el = document.getElementById('ws-status');
            if (el) { el.textContent = 'WS关闭(自动重连)'; el.style.color = '#888'; }
            if (wsReconnectAttempt < wsReconnectMax) {
                var delay = calcBackoff(wsReconnectAttempt);
                wsReconnectAttempt++;
                var el2 = document.getElementById('ws-status');
                if (el2) el2.textContent = 'WS重连中(' + wsReconnectAttempt + '/' + wsReconnectMax + ')...';
                setTimeout(connectTrainingWs, delay);
            } else {
                if (el) el.textContent = 'WS重连失败(已达最大次数)';
                if (!trainingPolling) { trainingPolling = setInterval(pollTraining, 3000); }
            }
        };
    }

    window.pauseTraining = async function() {
        try { var data = await SelfLnnApi.trainingPause(); alert(data.success ? '训练已暂停' : '暂停失败'); }
        catch (e) { alert('操作失败'); }
    };
    window.resumeTraining = async function() {
        try { var data = await SelfLnnApi.trainingResume(); alert(data.success ? '训练已恢复' : '恢复失败'); }
        catch (e) { alert('操作失败'); }
    };
    window.stopTraining = async function() {
        try {
            var data = await SelfLnnApi.trainingStop();
            stopPolling();
            alert(data.success ? '训练已停止' : '停止失败');
        } catch (e) { alert('操作失败'); }
    };

    async function loadHistory() {
        try {
            var data = await SelfLnnApi.trainingHistory();
            var tbody = document.getElementById('history-list');
            var tasks = data.tasks || data.history || [];
            if (tasks.length === 0) { tbody.innerHTML = '<tr><td colspan="7" class="pending-text">暂无记录</td></tr>'; return; }
            tbody.innerHTML = tasks.map(function (t) {
                return '<tr><td>' + (t.id || '-') + '</td><td>' + (t.mode || '-') + '</td><td><span class="status-badge ' + (t.status === 'running' ? 'active' : t.status === 'done' ? 'completed' : 'inactive') + '">' + (t.status || '-') + '</span></td><td>' + (t.loss != null ? t.loss.toFixed(6) : '-') + '</td><td>' + (t.accuracy != null ? (t.accuracy * 100).toFixed(2) + '%' : '-') + '</td><td>' + (t.epochs || '-') + '</td><td>' + (t.date || '-') + '</td></tr>';
            }).join('');
        } catch (e) { console.warn('训练历史加载失败:', e.message); }
    }

    window.startTraining = startTraining;

    document.addEventListener('DOMContentLoaded', function () {
        initChart();
        connectTrainingWs();
        loadHistory();
    });
})();
