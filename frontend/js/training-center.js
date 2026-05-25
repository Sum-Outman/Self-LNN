(function() {
    'use strict';
    var trainingPollInterval = null;

    function stopPolling() { if (trainingPollInterval) { clearInterval(trainingPollInterval); trainingPollInterval = null; } }

    async function startTraining() {
        try {
            /* FIX-JS-005: 使用selectedTrainingMode(全局变量)替代不存在的training-mode DOM元素 */
            var modeMap = {1:'imitation',2:'rl',3:'primitive',4:'joint',5:'task'};
            var mode = modeMap[window.selectedTrainingMode || 1] || 'pretrain';
            var lr = 0.001;
            var batch = 64;
            var epochs = 100;
            var datasetPath = '/data/training';
            var lrEl = document.getElementById('training-learning-rate');
            if (lrEl) lr = parseFloat(lrEl.value) || 0.001;
            var batchEl = document.getElementById('training-batch-size');
            if (batchEl) batch = parseInt(batchEl.value, 10) || 64;
            var epochEl = document.getElementById('training-epochs');
            if (epochEl) epochs = parseInt(epochEl.value, 10) || 100;

            /* ZSFZS-F024修复: 参数传递修正——trainingStart期望5个独立参数而非对象 */
            var data = await SelfLnnApi.trainingStart(mode, lr, batch, epochs, datasetPath);
            if (data.success) {
                showNotification('训练已启动(' + mode + ')', 'success');
                if (trainingPollInterval) clearInterval(trainingPollInterval);
                trainingPollInterval = setInterval(pollTraining, 2000);
                pollTraining();
            } else {
                showNotification('启动失败: ' + (data.error || ''), 'danger');
            }
        } catch(e) { showNotification('启动失败: ' + e.message, 'danger'); }
    }
    window.startTraining = startTraining;

    async function pollTraining() {
        try {
            var data = await SelfLnnApi.getTrainingStatus();
            if (data) {
                /* FIX-JS-005: accuracy和estimated_time写入loss值所在区域的边距span */
                var taskEl = document.getElementById('training-current-task');
                if (taskEl) taskEl.textContent = data.current_stage || data.stage || '--';
                var epochEl = document.getElementById('training-current-epoch');
                if (epochEl) epochEl.textContent = data.current_epoch || data.epoch || '--';
                var lossEl = document.getElementById('training-current-loss');
                if (lossEl) {
                    var lossVal = data.current_loss;
                    var accInfo = (data.accuracy !== undefined && data.accuracy !== null) ? (' | 准确率:' + (data.accuracy * 100).toFixed(1) + '%') : '';
                    var timeInfo = data.estimated_time ? (' | 预计:' + data.estimated_time) : '';
                    lossEl.textContent = (typeof lossVal === 'number' ? lossVal.toFixed(4) : String(lossVal || '--')) + accInfo + timeInfo;
                }
                var progressEl = document.getElementById('training-progress-fill');
                if (progressEl) progressEl.style.width = data.progress ? data.progress + '%' : '0%';
                var progressText = document.getElementById('training-progress-text');
                if (progressText) progressText.textContent = data.progress ? data.progress.toFixed(1) + '%' : '--';

                if (data.running === false) stopPolling();
            }
        } catch(e) { console.warn('训练轮询失败:', e.message); }
    }

    window.pauseTraining = async function() {
        try {
            var data = await SelfLnnApi.trainingPause();
            if (data && data.success) {
                showNotification('训练已暂停', 'success');
                stopPolling();
            } else {
                showNotification(data && data.error ? '暂停失败: ' + data.error : '暂停失败', 'danger');
            }
        } catch(e) { showNotification('操作失败', 'danger'); console.error('[训练中心] pauseTraining失败:', e&&e.message?e.message:e); }
    };
    window.resumeTraining = async function() {
        try {
            var data = await SelfLnnApi.trainingResume();
            /* ZSFZS-F052修复: 防止data为undefined/null时访问.success崩溃 */
            var ok = data && data.success;
            showNotification(ok ? '训练已恢复' : '恢复失败', ok ? 'success' : 'danger');
            if (trainingPollInterval) clearInterval(trainingPollInterval);
            trainingPollInterval = setInterval(pollTraining, 2000);
            pollTraining();
        } catch(e) { console.error('[训练中心] resumeTraining失败:', e.message); showNotification('操作失败', 'danger'); }
    };
    window.stopTraining = async function() {
        try {
            var data = await SelfLnnApi.trainingStop();
            stopPolling();
            var ok = data && data.success;
            showNotification(ok ? '训练已停止' : '停止失败', ok ? 'success' : 'danger');
        } catch(e) { console.error('[训练中心] stopTraining失败:', e.message); showNotification('操作失败', 'danger'); }
    };

    window.pollTraining = pollTraining;
})();
