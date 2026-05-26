(function() {
    'use strict';
    var simPolling = null;
    var _simStarting = false;
    var sim3dGl = null;
    var sim3dInitialized = false;

    function escapeHtml(str) { if (!str) return ''; var d = document.createElement('div'); d.textContent = str; return d.innerHTML; }
    function safeCoord(v) { return (typeof v === 'number' && !isNaN(v) && isFinite(v)) ? v.toFixed(1) : '?'; }

    /* ZSF-ZNB修复S-010: 初始化仿真3D WebGL画布 */
    function initSim3D() {
        var canvas = document.getElementById('sim3d-canvas');
        if (!canvas) return false;
        try {
            sim3dGl = canvas.getContext('webgl', { 
                alpha: true, 
                antialias: true,
                preserveDrawingBuffer: true 
            });
            if (!sim3dGl) {
                sim3dGl = canvas.getContext('experimental-webgl', { 
                    alpha: true, antialias: true 
                });
            }
            if (sim3dGl) {
                sim3dGl.clearColor(0.05, 0.05, 0.1, 1.0);
                sim3dGl.clearDepth(1.0);
                sim3dGl.enable(sim3dGl.DEPTH_TEST);
                sim3dGl.depthFunc(sim3dGl.LEQUAL);
                sim3dGl.viewport(0, 0, canvas.width || 640, canvas.height || 480);
                sim3dInitialized = true;
                return true;
            }
        } catch(e) {
            console.warn('WebGL初始化失败:', e.message);
        }
        return false;
    }

    function renderSim3DFrame() {
        if (!sim3dGl || !sim3dInitialized) return;
        var canvas = document.getElementById('sim3d-canvas');
        if (!canvas) return;
        sim3dGl.clear(sim3dGl.COLOR_BUFFER_BIT | sim3dGl.DEPTH_BUFFER_BIT);
        sim3dGl.viewport(0, 0, canvas.clientWidth || 640, canvas.clientHeight || 480);
        sim3dGl.flush();
    }

    initSim3D();

    async function startSimulation() {
        if (_simStarting) { window.showNotification('仿真正在启动中，请稍候', 'warning'); return; }
        _simStarting = true;
        var engine = document.getElementById('sim-engine') ? document.getElementById('sim-engine').value : 'internal';
        var scene = document.getElementById('scene-file') ? document.getElementById('scene-file').value : '';
        var dt = 0.01;
        try {
            var data = await window.SelfLnnApi.simulationStart(engine, scene || undefined, dt);
            if (data.success) {
                window.showNotification('仿真已启动', 'success');
                if (simPolling) { clearInterval(simPolling); simPolling = null; }
                if (window.g_dataEngine && typeof window.g_dataEngine.registerModule === 'function') {
                    window.g_dataEngine.registerModule('simulation_status', 1000, pollSimulation);
                } else {
                    simPolling = setInterval(pollSimulation, 1000);
                }
                pollSimulation();
            } else {
                window.showNotification('启动失败: ' + (data.error || ''), 'danger');
            }
        } catch(e) { console.error('[仿真] 启动失败:', e.message); window.showNotification('连接失败: ' + e.message, 'danger'); }
        _simStarting = false;
    }

    async function stopSimulation() {
        try {
            await window.SelfLnnApi.simulationStop();
            if (simPolling) { clearInterval(simPolling); simPolling = null; }
            if (window.g_dataEngine && typeof window.g_dataEngine.unregisterModule === 'function') {
                window.g_dataEngine.unregisterModule('simulation_status');
            }
            var el = document.getElementById('sim-status');
            if (el) el.textContent = '已停止';
            window.showNotification('仿真已停止', 'info');
        } catch(e) { console.error('[仿真] 停止失败:', e.message); window.showNotification('操作失败', 'danger'); }
    }

    async function resetSimulation() {
        try {
            await window.SelfLnnApi.simulationReset();
            window.showNotification('仿真已重置', 'info');
        } catch(e) { console.error('[仿真] 重置失败:', e.message); window.showNotification('操作失败', 'danger'); }
    }

    async function pollSimulation() {
        try {
            var data = await window.SelfLnnApi.simulationStatus();
            if (data) {
                /* ZSFABC-Fix: 仿真停止时自动清除轮询，防止资源泄漏 */
                if (data.running === false) {
                    if (simPolling) { clearInterval(simPolling); simPolling = null; }
                    if (window.g_dataEngine && typeof window.g_dataEngine.unregisterModule === 'function') {
                        window.g_dataEngine.unregisterModule('simulation_status');
                    }
                }
                var statusEl = document.getElementById('sim-status');
                if (statusEl) statusEl.textContent = data.running ? '运行中' : '已停止';
                var stepEl = document.getElementById('sim-steps');
                if (stepEl) stepEl.textContent = data.step || data.steps || '--';
                var timeEl = document.getElementById('sim-time');
                if (timeEl) timeEl.textContent = (typeof data.sim_time === 'number' && !isNaN(data.sim_time) && isFinite(data.sim_time) ? data.sim_time.toFixed(2) + 's' : '--');
                var fpsEl = document.getElementById('sim-fps');
                if (fpsEl) fpsEl.textContent = data.fps || '--';
                if (data.robots && data.robots.length > 0) {
                    var robotList = document.getElementById('sim-robot-list');
                    if (robotList) {
                        robotList.innerHTML = data.robots.map(function(r, i) {
                            return '<tr><td>' + escapeHtml(r.name || ('机器人' + (i + 1))) + '</td>' +
                                '<td>' + escapeHtml(r.status || '活跃') + '</td>' +
                                '<td>' + (r.position ? 'x=' + safeCoord(r.position.x) + ',y=' + safeCoord(r.position.y) + ',z=' + safeCoord(r.position.z) : '--') + '</td></tr>';
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
        try { await window.SelfLnnApi.request('/simulation/view/reset', { method: 'POST' }); window.showNotification('视图已重置', 'info'); }
        catch(e) { console.error('[仿真] 视图重置失败:', e.message); window.showNotification('操作失败', 'danger'); }
    }
    window.sim3dResetView = sim3dResetView;

    async function sim3dToggleGrid() {
        try { await window.SelfLnnApi.request('/simulation/view/toggle_grid', { method: 'POST' }); window.showNotification('网格已切换', 'info'); }
        catch(e) { console.error('[仿真] 网格切换失败:', e.message); window.showNotification('操作失败', 'danger'); }
    }
    window.sim3dToggleGrid = sim3dToggleGrid;

    async function sim3dAddRobot() {
        try { var d = await window.SelfLnnApi.request('/simulation/robot/add', { method: 'POST', body: JSON.stringify({}) }); window.showNotification(d && d.success ? '机器人已添加' : '添加失败: ' + (d && d.error || '未知错误'), d && d.success ? 'success' : 'danger'); }
        catch(e) { window.showNotification('添加失败: ' + (e && e.message || '网络错误'), 'danger'); }
    }
    window.sim3dAddRobot = sim3dAddRobot;

    async function sim3dClearAll() {
        try { await window.SelfLnnApi.request('/simulation/clear', { method: 'POST' }); window.showNotification('场景已清空', 'info'); }
        catch(e) { console.error('[仿真] 场景清空失败:', e.message); window.showNotification('操作失败', 'danger'); }
    }
    window.sim3dClearAll = sim3dClearAll;

    async function start3DReconstruction() {
        try { var d = await window.SelfLnnApi.request('/simulation/reconstruct3d', { method: 'POST', body: JSON.stringify({}) }); window.showNotification(d && d.success ? '三维重建已启动' : '启动失败: ' + (d && d.error || '未知错误'), d && d.success ? 'success' : 'danger'); }
        catch(e) { window.showNotification('启动失败: ' + (e && e.message || '网络错误'), 'danger'); }
    }
    window.start3DReconstruction = start3DReconstruction;

    async function executeCommand() {
        var cmd = document.getElementById('cmd-input') ? document.getElementById('cmd-input').value : '';
        if (!cmd) { window.showNotification('请输入命令', 'warning'); return; }
        try {
            var d = await window.SelfLnnApi.request('/simulation/command', { method: 'POST', body: JSON.stringify({ command: cmd }) });
            window.showNotification(d.success ? '命令已执行' : '执行失败', d.success ? 'success' : 'danger');
        } catch(e) { window.showNotification('连接失败', 'danger'); }
    }
    window.executeCommand = executeCommand;

    async function planPath() {
        try {
            var data = await window.SelfLnnApi.request('/robot/path/plan', { method: 'POST', body: JSON.stringify({}) });
            window.showNotification('路径规划完成，步数: ' + (data && data.steps || 0), 'success');
        } catch(e) { window.showNotification('连接失败', 'danger'); }
    }
    window.planPath = planPath;

    /* 页面卸载时清理定时器 */
    window.addEventListener('beforeunload', function() {
        if (simPolling) { clearInterval(simPolling); simPolling = null; }
    });

})();
