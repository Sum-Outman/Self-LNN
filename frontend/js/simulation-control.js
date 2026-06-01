(function() {
    'use strict';
    var simPolling = null;
    var _simStarting = false;
    var sim3dGl = null;
    var sim3dInitialized = false;
    var sim3dProgram = null;
    var sim3dUniforms = {};
    var sim3dBuffers = {};
    var sim3dLastFrame = 0;
    var sim3dAnimId = null;

    function escapeHtml(str) { if (!str) return ''; var d = document.createElement('div'); d.textContent = str; return d.innerHTML; }
    function safeCoord(v) { return (typeof v === 'number' && !isNaN(v) && isFinite(v)) ? v.toFixed(1) : '?'; }

    /* 顶点着色器：3D变换 + 光照 */
    var SIM3D_VERT_SHADER = [
        'attribute vec3 aPosition;',
        'attribute vec3 aNormal;',
        'attribute vec3 aColor;',
        'uniform mat4 uModelView;',
        'uniform mat4 uProjection;',
        'uniform mat3 uNormalMatrix;',
        'uniform vec3 uLightDir;',
        'varying vec3 vColor;',
        'void main(void) {',
        '  vec3 normal = normalize(uNormalMatrix * aNormal);',
        '  float diffuse = max(dot(normal, normalize(uLightDir)), 0.1);',
        '  vColor = aColor * (diffuse * 0.8 + 0.2);',
        '  gl_Position = uProjection * uModelView * vec4(aPosition, 1.0);',
        '}'
    ].join('\n');

    /* 片元着色器 */
    var SIM3D_FRAG_SHADER = [
        'precision mediump float;',
        'varying vec3 vColor;',
        'void main(void) {',
        '  gl_FragColor = vec4(vColor, 1.0);',
        '}'
    ].join('\n');

    /* 编译着色器 */
    function compileShader(gl, type, source) {
        var shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            console.warn('着色器编译失败:', gl.getShaderInfoLog(shader));
            gl.deleteShader(shader);
            return null;
        }
        return shader;
    }

    /* 创建着色器程序 */
    function createShaderProgram(gl) {
        var vs = compileShader(gl, gl.VERTEX_SHADER, SIM3D_VERT_SHADER);
        var fs = compileShader(gl, gl.FRAGMENT_SHADER, SIM3D_FRAG_SHADER);
        if (!vs || !fs) return null;
        var prog = gl.createProgram();
        gl.attachShader(prog, vs);
        gl.attachShader(prog, fs);
        gl.linkProgram(prog);
        if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
            console.warn('着色器链接失败:', gl.getProgramInfoLog(prog));
            return null;
        }
        gl.deleteShader(vs);
        gl.deleteShader(fs);
        return prog;
    }

    /* 创建立方体几何（代理机器人模型） */
    function createCubeGeometry(gl, size, color) {
        var s = size * 0.5;
        /* 36个顶点（6面×2三角形×3顶点），每面独立法线 */
        var vertices = new Float32Array([
            /* 前面 */ -s,-s, s,  s,-s, s,  s, s, s,   -s,-s, s,  s, s, s, -s, s, s,
            /* 后面 */  s,-s,-s, -s,-s,-s, -s, s,-s,    s,-s,-s, -s, s,-s,  s, s,-s,
            /* 顶面 */ -s, s, s,  s, s, s,  s, s,-s,   -s, s, s,  s, s,-s, -s, s,-s,
            /* 底面 */ -s,-s,-s,  s,-s,-s,  s,-s, s,   -s,-s,-s,  s,-s, s, -s,-s, s,
            /* 右面 */  s,-s, s,  s,-s,-s,  s, s,-s,    s,-s, s,  s, s,-s,  s, s, s,
            /* 左面 */ -s,-s,-s, -s,-s, s, -s, s, s,   -s,-s,-s, -s, s, s, -s, s,-s
        ]);
        /* 每个面的法线 */
        var fx = 0, fy = 0, fz = 1, bx = 0, by = 0, bz = -1;
        var tx = 0, ty = 1, tz = 0, dx = 0, dy = -1, dz = 0;
        var rx = 1, ry = 0, rz = 0, lx = -1, ly = 0, lz = 0;
        var normals = new Float32Array([
            fx,fy,fz,fx,fy,fz,fx,fy,fz,fx,fy,fz,fx,fy,fz,fx,fy,fz,
            bx,by,bz,bx,by,bz,bx,by,bz,bx,by,bz,bx,by,bz,bx,by,bz,
            tx,ty,tz,tx,ty,tz,tx,ty,tz,tx,ty,tz,tx,ty,tz,tx,ty,tz,
            dx,dy,dz,dx,dy,dz,dx,dy,dz,dx,dy,dz,dx,dy,dz,dx,dy,dz,
            rx,ry,rz,rx,ry,rz,rx,ry,rz,rx,ry,rz,rx,ry,rz,rx,ry,rz,
            lx,ly,lz,lx,ly,lz,lx,ly,lz,lx,ly,lz,lx,ly,lz,lx,ly,lz
        ]);
        var r = color[0], g = color[1], b = color[2];
        var colors = new Float32Array(36 * 3);
        for (var i = 0; i < 36; i++) {
            colors[i * 3] = r; colors[i * 3 + 1] = g; colors[i * 3 + 2] = b;
        }
        return {
            vertices: vertices,
            normals: normals,
            colors: colors,
            vertexCount: 36
        };
    }

    /* 创建地面平面几何 */
    function createGroundGeometry(gl, size, color) {
        var s = size * 0.5;
        var vertices = new Float32Array([
            -s, 0, -s,  s, 0, -s,  s, 0, s,
            -s, 0, -s,  s, 0,  s, -s, 0, s
        ]);
        var normals = new Float32Array([
            0,1,0, 0,1,0, 0,1,0, 0,1,0, 0,1,0, 0,1,0
        ]);
        var r = color[0], g = color[1], b = color[2];
        var colors = new Float32Array(18);
        for (var i = 0; i < 6; i++) {
            colors[i * 3] = r; colors[i * 3 + 1] = g; colors[i * 3 + 2] = b;
        }
        return {
            vertices: vertices, normals: normals, colors: colors, vertexCount: 6
        };
    }

    /* 初始化WebGL资源和着色器 */
    function initSim3D() {
        var canvas = document.getElementById('sim3d-canvas');
        if (!canvas) return false;
        try {
            sim3dGl = canvas.getContext('webgl', { 
                alpha: true, antialias: true, preserveDrawingBuffer: true 
            });
            if (!sim3dGl) {
                sim3dGl = canvas.getContext('experimental-webgl', { 
                    alpha: true, antialias: true 
                });
            }
            if (!sim3dGl) return false;

            sim3dGl.clearColor(0.06, 0.06, 0.12, 1.0);
            sim3dGl.clearDepth(1.0);
            sim3dGl.enable(sim3dGl.DEPTH_TEST);
            sim3dGl.depthFunc(sim3dGl.LEQUAL);

            /* 创建着色器程序 */
            sim3dProgram = createShaderProgram(sim3dGl);
            if (!sim3dProgram) { sim3dGl = null; return false; }

            /* 获取Uniform位置 */
            sim3dUniforms.modelView = sim3dGl.getUniformLocation(sim3dProgram, 'uModelView');
            sim3dUniforms.projection = sim3dGl.getUniformLocation(sim3dProgram, 'uProjection');
            sim3dUniforms.normalMatrix = sim3dGl.getUniformLocation(sim3dProgram, 'uNormalMatrix');
            sim3dUniforms.lightDir = sim3dGl.getUniformLocation(sim3dProgram, 'uLightDir');

            /* 获取Attribute位置 */
            sim3dBuffers.posLoc = sim3dGl.getAttribLocation(sim3dProgram, 'aPosition');
            sim3dBuffers.normLoc = sim3dGl.getAttribLocation(sim3dProgram, 'aNormal');
            sim3dBuffers.colLoc = sim3dGl.getAttribLocation(sim3dProgram, 'aColor');

            /* 创建立方体几何（机器人代理） */
            var cube = createCubeGeometry(sim3dGl, 0.6, [0.3, 0.7, 1.0]);
            sim3dBuffers.robotVBuf = createGLBuffer(sim3dGl, cube.vertices);
            sim3dBuffers.robotNBuf = createGLBuffer(sim3dGl, cube.normals);
            sim3dBuffers.robotCBuf = createGLBuffer(sim3dGl, cube.colors);
            sim3dBuffers.robotCount = cube.vertexCount;

            /* 创建地面几何 */
            var ground = createGroundGeometry(sim3dGl, 10.0, [0.15, 0.3, 0.15]);
            sim3dBuffers.groundVBuf = createGLBuffer(sim3dGl, ground.vertices);
            sim3dBuffers.groundNBuf = createGLBuffer(sim3dGl, ground.normals);
            sim3dBuffers.groundCBuf = createGLBuffer(sim3dGl, ground.colors);
            sim3dBuffers.groundCount = ground.vertexCount;

            /* 创建轴指示器（红色X，蓝色Z，绿色Y） */
            var axisVerts = new Float32Array([
                0,0,0, 1.5,0,0, 0,0,0, 0,1.5,0, 0,0,0, 0,0,1.5
            ]);
            var axisColors = new Float32Array([
                1,0.2,0.2, 1,0.2,0.2, 0.2,1,0.2, 0.2,1,0.2, 0.2,0.4,1, 0.2,0.4,1
            ]);
            var axisNorms = new Float32Array(18);
            for (var i = 0; i < 6; i++) { axisNorms[i * 3] = 0; axisNorms[i * 3 + 1] = 1; axisNorms[i * 3 + 2] = 0; }
            sim3dBuffers.axisVBuf = createGLBuffer(sim3dGl, axisVerts);
            sim3dBuffers.axisNBuf = createGLBuffer(sim3dGl, axisNorms);
            sim3dBuffers.axisCBuf = createGLBuffer(sim3dGl, axisColors);
            sim3dBuffers.axisCount = 6;

            sim3dInitialized = true;
            return true;
        } catch(e) {
            console.warn('WebGL初始化失败:', e.message);
        }
        return false;
    }

    /* 创建并上传GL缓冲区 */
    function createGLBuffer(gl, data) {
        var buf = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buf);
        gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
        return buf;
    }

    /* 矩阵运算辅助函数 */
    function mat4Identity() {
        return [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1];
    }
    function mat4Perspective(fov, aspect, near, far) {
        var f = 1.0 / Math.tan(fov * Math.PI / 360.0);
        var nf = 1.0 / (near - far);
        return [f/aspect,0,0,0, 0,f,0,0, 0,0,(far+near)*nf,-1, 0,0,2*far*near*nf,0];
    }
    function mat4LookAt(eyex, eyey, eyez, cx, cy, cz, ux, uy, uz) {
        var zx = eyex - cx, zy = eyey - cy, zz = eyez - cz;
        var zl = Math.sqrt(zx*zx + zy*zy + zz*zz);
        zx /= zl; zy /= zl; zz /= zl;
        var xx = uy * zz - uz * zy, xy = uz * zx - ux * zz, xz = ux * zy - uy * zx;
        var xl = Math.sqrt(xx*xx + xy*xy + xz*xz);
        xx /= xl; xy /= xl; xz /= xl;
        var yx = zy * xz - zz * xy, yy = zz * xx - zx * xz, yz = zx * xy - zy * xx;
        return [
            xx, yx, zx, 0,
            xy, yy, zy, 0,
            xz, yz, zz, 0,
            -(xx*eyex + xy*eyey + xz*eyez),
            -(yx*eyex + yy*eyey + yz*eyez),
            -(zx*eyex + zy*eyey + zz*eyez), 1
        ];
    }
    function mat4Translate(tx, ty, tz) {
        var m = mat4Identity();
        m[12] = tx; m[13] = ty; m[14] = tz;
        return m;
    }
    function mat4Multiply(a, b) {
        var r = new Array(16);
        for (var i = 0; i < 4; i++) {
            for (var j = 0; j < 4; j++) {
                r[i * 4 + j] = a[j] * b[i * 4] + a[4 + j] * b[i * 4 + 1] + a[8 + j] * b[i * 4 + 2] + a[12 + j] * b[i * 4 + 3];
            }
        }
        return r;
    }
    function mat3FromMat4(m) {
        return [m[0],m[1],m[2], m[4],m[5],m[6], m[8],m[9],m[10]];
    }

    /* 绑定几何并绘制 */
    function drawGeometry(gl, vBuf, nBuf, cBuf, count) {
        gl.bindBuffer(gl.ARRAY_BUFFER, vBuf);
        gl.enableVertexAttribArray(sim3dBuffers.posLoc);
        gl.vertexAttribPointer(sim3dBuffers.posLoc, 3, gl.FLOAT, false, 0, 0);

        gl.bindBuffer(gl.ARRAY_BUFFER, nBuf);
        gl.enableVertexAttribArray(sim3dBuffers.normLoc);
        gl.vertexAttribPointer(sim3dBuffers.normLoc, 3, gl.FLOAT, false, 0, 0);

        gl.bindBuffer(gl.ARRAY_BUFFER, cBuf);
        gl.enableVertexAttribArray(sim3dBuffers.colLoc);
        gl.vertexAttribPointer(sim3dBuffers.colLoc, 3, gl.FLOAT, false, 0, 0);

        gl.drawArrays(gl.TRIANGLES, 0, count);
    }

    /* ZSFWS-001修复: 完整3D渲染帧 */
    function renderSim3DFrame() {
        if (!sim3dGl || !sim3dInitialized || !sim3dProgram) return;
        var gl = sim3dGl;
        var canvas = document.getElementById('sim3d-canvas');
        if (!canvas) return;

        var w = canvas.clientWidth || 640;
        var h = canvas.clientHeight || 480;
        if (canvas.width !== w || canvas.height !== h) {
            canvas.width = w; canvas.height = h;
        }
        gl.viewport(0, 0, w, h);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        gl.useProgram(sim3dProgram);

        /* 透视投影 */
        var proj = mat4Perspective(55, w / Math.max(h, 1), 0.1, 50.0);
        gl.uniformMatrix4fv(sim3dUniforms.projection, false, proj);

        /* 动态摄像机（绕场景旋转） */
        var time = Date.now() * 0.001;
        var camDist = 5.0;
        var camX = Math.sin(time * 0.3) * camDist;
        var camZ = Math.cos(time * 0.3) * camDist;
        var camY = 2.5 + Math.sin(time * 0.2) * 0.5;
        var view = mat4LookAt(camX, camY, camZ, 0, 0.3, 0, 0, 1, 0);

        /* 光照方向 */
        gl.uniform3f(sim3dUniforms.lightDir, 0.6, 0.8, 0.4);

        /* 绘制地面 */
        var groundMv = mat4Translate(0, -0.6, 0);
        var groundMvp = mat4Multiply(view, groundMv);
        gl.uniformMatrix4fv(sim3dUniforms.modelView, false, groundMvp);
        gl.uniformMatrix3fv(sim3dUniforms.normalMatrix, false, mat3FromMat4(groundMvp));
        drawGeometry(gl, sim3dBuffers.groundVBuf, sim3dBuffers.groundNBuf, sim3dBuffers.groundCBuf, sim3dBuffers.groundCount);

        /* 绘制机器人代理立方体 */
        var robotMv = mat4Multiply(view, mat4Translate(0, 0, 0));
        gl.uniformMatrix4fv(sim3dUniforms.modelView, false, robotMv);
        gl.uniformMatrix3fv(sim3dUniforms.normalMatrix, false, mat3FromMat4(robotMv));
        drawGeometry(gl, sim3dBuffers.robotVBuf, sim3dBuffers.robotNBuf, sim3dBuffers.robotCBuf, sim3dBuffers.robotCount);

        /* 绘制坐标轴 */
        var axisMv = mat4Multiply(view, mat4Translate(0, 0.01, 0));
        gl.uniformMatrix4fv(sim3dUniforms.modelView, false, axisMv);
        gl.uniformMatrix3fv(sim3dUniforms.normalMatrix, false, mat3FromMat4(axisMv));
        drawGeometry(gl, sim3dBuffers.axisVBuf, sim3dBuffers.axisNBuf, sim3dBuffers.axisCBuf, sim3dBuffers.axisCount);

        gl.flush();
        sim3dLastFrame = time;
    }

    /* 启动渲染循环 */
    function startRenderLoop() {
        if (sim3dAnimId) return;
        function loop() {
            renderSim3DFrame();
            sim3dAnimId = requestAnimationFrame(loop);
        }
        loop();
    }
    function stopRenderLoop() {
        if (sim3dAnimId) { cancelAnimationFrame(sim3dAnimId); sim3dAnimId = null; }
    }

    /* FIX-F2-CRIT-2: 延迟到DOMContentLoaded后初始化WebGL,此时canvas已存在 */
    var _simInitDone = false;
    function _ensureSimInit() {
        if (_simInitDone) return;
        _simInitDone = true;
        initSim3D();
        startRenderLoop();
    }
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', _ensureSimInit);
    } else {
        _ensureSimInit();
    }

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
            var resp = await window.SelfLnnApi.simulationStatus();
            /* FIX-F2-CRIT-1: SelfLnnApi包装响应为{success,data},需取.data层 */
            var data = (resp && resp.data) ? resp.data : resp;
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
        try { var resp = await window.SelfLnnApi.request('/simulation/robot/add', { method: 'POST', body: JSON.stringify({}) }); var d = await resp.json(); window.showNotification(d && d.success ? '机器人已添加' : '添加失败: ' + (d && d.error || '未知错误'), d && d.success ? 'success' : 'danger'); }
        catch(e) { window.showNotification('添加失败: ' + (e && e.message || '网络错误'), 'danger'); }
    }
    window.sim3dAddRobot = sim3dAddRobot;

    async function sim3dClearAll() {
        try { await window.SelfLnnApi.request('/simulation/clear', { method: 'POST' }); window.showNotification('场景已清空', 'info'); }
        catch(e) { console.error('[仿真] 场景清空失败:', e.message); window.showNotification('操作失败', 'danger'); }
    }
    window.sim3dClearAll = sim3dClearAll;

    async function start3DReconstruction() {
        try { var resp = await window.SelfLnnApi.request('/simulation/reconstruct3d', { method: 'POST', body: JSON.stringify({}) }); var d = await resp.json(); window.showNotification(d && d.success ? '三维重建已启动' : '启动失败: ' + (d && d.error || '未知错误'), d && d.success ? 'success' : 'danger'); }
        catch(e) { window.showNotification('启动失败: ' + (e && e.message || '网络错误'), 'danger'); }
    }
    window.start3DReconstruction = start3DReconstruction;

    async function executeCommand() {
        var cmd = document.getElementById('cmd-input') ? document.getElementById('cmd-input').value : '';
        if (!cmd) { window.showNotification('请输入命令', 'warning'); return; }
        try {
            var resp = await window.SelfLnnApi.request('/simulation/command', { method: 'POST', body: JSON.stringify({ command: cmd }) });
            var d = await resp.json();
            var execOk = d && d.simulation && d.simulation.executed;
            window.showNotification(execOk ? '命令已执行' : '执行失败', execOk ? 'success' : 'danger');
        } catch(e) { window.showNotification('连接失败', 'danger'); }
    }
    window.executeCommand = executeCommand;

    async function planPath() {
        try {
            var resp = await window.SelfLnnApi.request('/robot/path/plan', { method: 'POST', body: JSON.stringify({}) });
            var d = await resp.json();
            window.showNotification('路径规划完成，步数: ' + (d && d.steps || 0), 'success');
        } catch(e) { window.showNotification('连接失败', 'danger'); }
    }
    window.planPath = planPath;

    /* 页面卸载时清理定时器和WebGL资源 */
    window.addEventListener('beforeunload', function() {
        if (simPolling) { clearInterval(simPolling); simPolling = null; }
        /* ZSFXXXQ-P2-006: 完整清理WebGL资源 */
        stopRenderLoop();
        var gl = sim3dGl;
        if (gl) {
            if (sim3dProgram) { gl.deleteProgram(sim3dProgram); sim3dProgram = null; }
            var bufs = ['robotVBuf','robotNBuf','robotCBuf','groundVBuf','groundNBuf','groundCBuf','axisVBuf','axisNBuf','axisCBuf'];
            for (var i = 0; i < bufs.length; i++) {
                if (sim3dBuffers[bufs[i]]) { gl.deleteBuffer(sim3dBuffers[bufs[i]]); sim3dBuffers[bufs[i]] = null; }
            }
            if (gl.getExtension('WEBGL_lose_context')) { gl.getExtension('WEBGL_lose_context').loseContext(); }
            sim3dGl = null;
            sim3dInitialized = false;
        }
    });

})();
