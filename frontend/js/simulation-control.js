(function() {
    'use strict';
    var simPolling = null;

    async function startSimulation() {
        var engine = document.getElementById('sim-engine') ? document.getElementById('sim-engine').value : 'internal';
        var scene = document.getElementById('sim-scene') ? document.getElementById('sim-scene').value : '';
        var dt = 0.01;
        try {
            var data = await SelfLnnApi.simulationStart({
                engine: engine,
                scene: scene || undefined,
                dt: dt
            });
            if (data.success) {
                showNotification('仿真已启动', 'success');
                if (simPolling) { clearInterval(simPolling); simPolling = null; }
                if (window.g_dataEngine && typeof window.g_dataEngine.registerModule === 'function') {
                    window.g_dataEngine.registerModule('simulation_status', 1000, pollSimulation);
                } else {
                    simPolling = setInterval(pollSimulation, 1000);
                }
                pollSimulation();
            } else {
                showNotification('启动失败: ' + (data.error || ''), 'danger');
            }
        } catch(e) { showNotification('连接失败: ' + e.message, 'danger'); }
    }

    async function stopSimulation() {
        try {
            await SelfLnnApi.simulationStop();
            if (simPolling) { clearInterval(simPolling); simPolling = null; }
            if (window.g_dataEngine && typeof window.g_dataEngine.unregisterModule === 'function') {
                window.g_dataEngine.unregisterModule('simulation_status');
            }
            var el = document.getElementById('sim-status');
            if (el) el.textContent = '已停止';
            showNotification('仿真已停止', 'info');
        } catch(e) { showNotification('操作失败', 'danger'); }
    }

    async function resetSimulation() {
        try {
            await SelfLnnApi.simulationReset();
            showNotification('仿真已重置', 'info');
        } catch(e) { showNotification('操作失败', 'danger'); }
    }

    async function pollSimulation() {
        try {
            var data = await SelfLnnApi.simulationStatus();
            if (data) {
                var statusEl = document.getElementById('sim-status');
                if (statusEl) statusEl.textContent = data.running ? '运行中' : '已停止';
                var stepEl = document.getElementById('sim-step');
                if (stepEl) stepEl.textContent = data.step || data.steps || '--';
                var timeEl = document.getElementById('sim-time');
                if (timeEl) timeEl.textContent = (data.sim_time !== undefined ? data.sim_time.toFixed(2) + 's' : '--');
                var fpsEl = document.getElementById('sim-fps');
                if (fpsEl) fpsEl.textContent = data.fps || '--';
                if (data.robots && data.robots.length > 0) {
                    var robotList = document.getElementById('sim-robot-list');
                    if (robotList) {
                        robotList.innerHTML = data.robots.map(function(r, i) {
                            return '<tr><td>' + (r.name || ('机器人' + (i + 1))) + '</td>' +
                                '<td>' + (r.status || '活跃') + '</td>' +
                                '<td>' + (r.position ? 'x=' + r.position.x.toFixed(1) + ',y=' + r.position.y.toFixed(1) + ',z=' + r.position.z.toFixed(1) : '--') + '</td></tr>';
                        }).join('');
                    }
                }
            }
        } catch(e) { console.warn('仿真轮询失败:', e.message); }
    }

    window.startSimulation = startSimulation;
    window.stopSimulation = stopSimulation;
    window.resetSimulation = resetSimulation;
    window.pollSimulation = pollSimulation;

    async function sim3dResetView() {
        try { await SelfLnnApi.request('/simulation/view/reset', { method: 'POST' }); showNotification('视图已重置', 'info'); }
        catch(e) { showNotification('操作失败', 'danger'); }
    }
    window.sim3dResetView = sim3dResetView;

    async function sim3dToggleGrid() {
        try { await SelfLnnApi.request('/simulation/view/toggle_grid', { method: 'POST' }); showNotification('网格已切换', 'info'); }
        catch(e) { showNotification('操作失败', 'danger'); }
    }
    window.sim3dToggleGrid = sim3dToggleGrid;

    async function sim3dAddRobot() {
        try { var d = await SelfLnnApi.request('/simulation/robot/add', { method: 'POST', body: JSON.stringify({}) }); showNotification('机器人已添加', 'success'); }
        catch(e) { showNotification('添加失败', 'danger'); }
    }
    window.sim3dAddRobot = sim3dAddRobot;

    async function sim3dClearAll() {
        try { await SelfLnnApi.request('/simulation/clear', { method: 'POST' }); showNotification('场景已清空', 'info'); }
        catch(e) { showNotification('操作失败', 'danger'); }
    }
    window.sim3dClearAll = sim3dClearAll;

    async function start3DReconstruction() {
        try { var d = await SelfLnnApi.request('/simulation/reconstruct3d', { method: 'POST', body: JSON.stringify({}) }); showNotification('三维重建已启动', 'success'); }
        catch(e) { showNotification('启动失败', 'danger'); }
    }
    window.start3DReconstruction = start3DReconstruction;

    async function executeCommand() {
        var cmd = document.getElementById('sim-cmd') ? document.getElementById('sim-cmd').value : '';
        if (!cmd) { showNotification('请输入命令', 'warning'); return; }
        try {
            var d = await SelfLnnApi.request('/simulation/command', { method: 'POST', body: JSON.stringify({ command: cmd }) });
            showNotification(d.success ? '命令已执行' : '执行失败', d.success ? 'success' : 'danger');
        } catch(e) { showNotification('连接失败', 'danger'); }
    }
    window.executeCommand = executeCommand;

    async function planPath() {
        try {
            var data = await SelfLnnApi.request('/robot/path/plan', { method: 'POST', body: JSON.stringify({}) });
            showNotification('路径规划完成，步数: ' + (data.steps || 0), 'success');
        } catch(e) { showNotification('连接失败', 'danger'); }
    }
    window.planPath = planPath;

})();
