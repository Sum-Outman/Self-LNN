(function() {
    'use strict';
    var trainingPollInterval = null;

    function stopPolling() { if (trainingPollInterval) { clearInterval(trainingPollInterval); trainingPollInterval = null; } }

    async function startTraining() {
        try {
            var mode = document.getElementById('training-mode') ? document.getElementById('training-mode').value : 'pretrain';
            var lr = 0.001;
            var batch = 64;
            var epochs = 100;
            var datasetPath = document.getElementById('training-dataset-path') ?
                document.getElementById('training-dataset-path').value : null;
            if (!datasetPath) datasetPath = '/data/training';
            var lrEl = document.getElementById('training-learning-rate');
            if (lrEl) lr = parseFloat(lrEl.value) || 0.001;
            var batchEl = document.getElementById('training-batch-size');
            if (batchEl) batch = parseInt(batchEl.value, 10) || 64;
            var epochEl = document.getElementById('training-epochs');
            if (epochEl) epochs = parseInt(epochEl.value, 10) || 100;

            var data = await SelfLnnApi.trainingStart({ mode: mode, learning_rate: lr, batch_size: batch, epochs: epochs, dataset_path: datasetPath });
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
                var stageEl = document.getElementById('training-current-stage');
                if (stageEl) stageEl.textContent = data.current_stage || data.stage || '--';
                var epochEl = document.getElementById('training-current-epoch');
                if (epochEl) epochEl.textContent = data.current_epoch || data.epoch || '--';
                var lossEl = document.getElementById('training-current-loss');
                if (lossEl) lossEl.textContent = data.current_loss ? data.current_loss.toFixed(4) : '--';
                var accEl = document.getElementById('training-current-accuracy');
                if (accEl) accEl.textContent = data.accuracy ? (data.accuracy * 100).toFixed(1) + '%' : '--';
                var timeEl = document.getElementById('training-estimated-time');
                if (timeEl) timeEl.textContent = data.estimated_time || '--';
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
            showNotification(data.success ? '训练已恢复' : '恢复失败', data.success ? 'success' : 'danger');
            if (trainingPollInterval) clearInterval(trainingPollInterval);
            trainingPollInterval = setInterval(pollTraining, 2000);
            pollTraining();
        } catch(e) { showNotification('操作失败', 'danger'); }
    };
    window.stopTraining = async function() {
        try {
            var data = await SelfLnnApi.trainingStop();
            stopPolling();
            showNotification(data.success ? '训练已停止' : '停止失败', data.success ? 'success' : 'danger');
        } catch(e) { showNotification('操作失败', 'danger'); }
    };

    window.pollTraining = pollTraining;
})();
