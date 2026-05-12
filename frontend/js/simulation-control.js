/* ===== WebGL 3D 仿真渲染器（声明必须在 IIFE 之前，确保 IIFE 内可安全引用） ===== */
var g_sim3d = {
    canvas: null, gl: null,
    robots: [], objects: [],
    cameraAngleX: 0.5, cameraAngleY: -0.3, cameraDist: 15,
    mouseDown: false, mouseX: 0, mouseY: 0, mouseButton: 0,
    showGrid: true, gridSize: 10,
    animId: null
};

(function() {
    let simPolling = null;

    async function startSimulation() {
        const engine = document.getElementById('sim-engine').value;
        const scene = document.getElementById('scene-file').value;
        const dt = parseInt(document.getElementById('sim-dt').value);
        try {
            const data = await SelfLnnApi.simulationStart({
                engine,
                scene: scene || undefined,
                dt
            });
            if (data.success) {
                alert('仿真已启动');
                if (simPolling) clearInterval(simPolling);
                simPolling = setInterval(pollSimulation, 1000);
                pollSimulation();
            } else {
                alert('启动失败: '+(data.error||''));
            }
        } catch(e) { alert('连接失败: ' + e.message); }
    }

    async function stopSimulation() {
        try {
            await SelfLnnApi.simulationStop();
            if (simPolling) { clearInterval(simPolling); simPolling = null; }
            document.getElementById('sim-status').textContent = '已停止';
            alert('仿真已停止');
        } catch(e) { alert('操作失败'); }
    }

    async function resetSimulation() {
        try {
            await SelfLnnApi.simulationReset();
            alert('仿真已重置');
        } catch(e) { alert('操作失败'); }
    }

    async function pollSimulation() {
        try {
            const data = await SelfLnnApi.simulationStatus();
            if (!data || typeof data.running === 'undefined') {
                document.getElementById('sim-status').textContent = '数据不可用';
                document.getElementById('sim-time').textContent = '-';
                document.getElementById('sim-steps').textContent = '-';
                document.getElementById('sim-realtime').textContent = '-';
                document.getElementById('sim-robots').textContent = '-';
                return;
            }
            document.getElementById('sim-status').textContent = data.running ? '运行中' : '已停止';
            document.getElementById('sim-time').textContent = (typeof data.elapsed_time === 'number') ? data.elapsed_time.toFixed(1)+'s' : '-';
            document.getElementById('sim-steps').textContent = (typeof data.steps === 'number') ? data.steps : '-';
            document.getElementById('sim-realtime').textContent = (typeof data.realtime_factor === 'number') ? data.realtime_factor.toFixed(2)+'x' : '-';
            document.getElementById('sim-robots').textContent = (typeof data.robot_count === 'number') ? data.robot_count : '-';

            const tbody = document.getElementById('sim-robot-list');
            if (data.robots && data.robots.length > 0) {
                tbody.innerHTML = data.robots.map(r => `
                    <tr>
                        <td>${r.id||'-'}</td>
                        <td>${r.name||'-'}</td>
                        <td>${r.type||'-'}</td>
                        <td>${(r.position && typeof r.position.x==='number') ? '('+r.position.x.toFixed(2)+', '+r.position.y.toFixed(2)+')' : '-'}</td>
                        <td>${(typeof r.velocity==='number') ? r.velocity.toFixed(2) : '-'}</td>
                        <td><button onclick="controlRobot('${r.id}')" class="btn btn-sm">控制</button></td>
                    </tr>
                `).join('');
                /* F-049/F-050: 仅当存在真实位置数据时才更新3D渲染器 */
                var hasPosData = data.robots.some(function(r){return r.position && typeof r.position.x==='number';});
                if (hasPosData) {
                    g_sim3d.robots = data.robots.map(function(r) {
                        if (r.position && typeof r.position.x==='number') {
                            return {id: r.id, x: r.position.x, y: r.position.y + 0.5, z: r.position.z, v: r.velocity||0};
                        }
                        return null;
                    }).filter(function(v){return v!==null;});
                }
                if (data.objects && data.objects.length > 0) {
                    g_sim3d.objects = data.objects.map(function(o) {
                        return {x: o.x||0, y: o.y||0, z: o.z||0, sx: o.sx||0.3, sy: o.sy||0.3, sz: o.sz||0.3, color: o.color||[0.2,1,0.2]};
                    });
                }
            } else {
                tbody.innerHTML = '<tr><td colspan="6" class="pending-text">暂无机器人</td></tr>';
                /* F-050: 无数据时清除渲染器 */
                g_sim3d.robots = [];
                g_sim3d.objects = [];
            }

            document.getElementById('lidar-status').textContent = data.lidar_ready ? '就绪' : '未连接';
            document.getElementById('depth-status').textContent = data.depth_cam_ready ? '就绪' : '未连接';
            document.getElementById('imu-status').textContent = data.imu_ready ? '就绪' : '未连接';
            document.getElementById('odom-status').textContent = data.odom_ready ? '就绪' : '未连接';
        } catch(e) {
            if (simPolling) { clearInterval(simPolling); simPolling = null; }
        }
    }

    async function planPath() {
        const gx = parseFloat(document.getElementById('goal-x').value);
        const gy = parseFloat(document.getElementById('goal-y').value);
        try {
            const data = await SelfLnnApi.simulationPlanPath(gx, gy);
            if (data.success) {
                alert('路径规划完成，步数: '+(data.steps||0));
            } else {
                alert('规划失败: '+(data.error||''));
            }
        } catch(e) { alert('连接失败'); }
    }

    window.controlRobot = function(robotId) {
        const cmd = prompt(`输入对机器人 ${robotId} 的命令 (move/stop/turn):`, 'move');
        if (!cmd) return;
        SelfLnnApi.request('/simulation/robot_control', {
            method: 'POST',
            body: JSON.stringify({robot_id: robotId, command: cmd})
        }, 1).catch(e => alert('操作失败'));
    };

    window.startSimulation = startSimulation;
    window.stopSimulation = stopSimulation;
    window.resetSimulation = resetSimulation;
    window.planPath = planPath;

    document.addEventListener('DOMContentLoaded', () => {
        SelfLnnApi.simulationStatus().then(data => {
            if (data.running) {
                simPolling = setInterval(pollSimulation, 1000);
                pollSimulation();
            }
        }).catch(() => {});
        sim3dInit();
    });
})();

/* ===== WebGL 3D 仿真渲染器函数 ===== */

function sim3dInit() {
    var c = document.getElementById('sim3d-canvas');
    if (!c) return;
    g_sim3d.canvas = c;
    var gl = c.getContext('webgl') || c.getContext('experimental-webgl');
    if (!gl) { c.style.background = '#111'; return; }
    g_sim3d.gl = gl;
    var vs = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vs, 'attribute vec3 aPos; uniform mat4 uMVP; void main(){gl_Position=uMVP*vec4(aPos,1.0);gl_PointSize=4.0;}');
    gl.compileShader(vs);
    var fs = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fs, 'precision mediump float; uniform vec3 uColor; void main(){gl_FragColor=vec4(uColor,1.0);}');
    gl.compileShader(fs);
    g_sim3d.program = gl.createProgram();
    gl.attachShader(g_sim3d.program, vs);
    gl.attachShader(g_sim3d.program, fs);
    gl.linkProgram(g_sim3d.program);
    gl.useProgram(g_sim3d.program);
    g_sim3d.uMVP = gl.getUniformLocation(g_sim3d.program, 'uMVP');
    g_sim3d.uColor = gl.getUniformLocation(g_sim3d.program, 'uColor');
    gl.enable(gl.DEPTH_TEST);
    c.addEventListener('mousedown', function(e){ g_sim3d.mouseDown=true; g_sim3d.mouseX=e.clientX; g_sim3d.mouseY=e.clientY; g_sim3d.mouseButton=e.button; });
    c.addEventListener('mousemove', function(e){
        if(!g_sim3d.mouseDown) return;
        var dx=e.clientX-g_sim3d.mouseX, dy=e.clientY-g_sim3d.mouseY;
        if(g_sim3d.mouseButton===0){g_sim3d.cameraAngleY+=dx*0.005;g_sim3d.cameraAngleX-=dy*0.005;}
        else if(g_sim3d.mouseButton===2){g_sim3d.cameraDist+=dy*0.1;}
        g_sim3d.mouseX=e.clientX;g_sim3d.mouseY=e.clientY;
    });
    c.addEventListener('mouseup', function(){g_sim3d.mouseDown=false;});
    c.addEventListener('wheel', function(e){e.preventDefault();g_sim3d.cameraDist+=e.deltaY*0.02;if(g_sim3d.cameraDist<2)g_sim3d.cameraDist=2;if(g_sim3d.cameraDist>50)g_sim3d.cameraDist=50;});
    c.addEventListener('contextmenu', function(e){e.preventDefault();});
    sim3dRenderLoop();
}

function sim3dBuildMVP() {
    var gl = g_sim3d.gl;
    var w = g_sim3d.canvas.width, h = g_sim3d.canvas.height;
    var aspect = w / Math.max(h, 1);
    var proj = new Float32Array(16);
    var fov = 45 * Math.PI / 180;
    var near = 0.1, far = 100;
    var f = 1.0 / Math.tan(fov / 2);
    proj[0]=f/aspect;proj[1]=0;proj[2]=0;proj[3]=0;
    proj[4]=0;proj[5]=f;proj[6]=0;proj[7]=0;
    proj[8]=0;proj[9]=0;proj[10]=(far+near)/(near-far);proj[11]=-1;
    proj[12]=0;proj[13]=0;proj[14]=(2*far*near)/(near-far);proj[15]=0;
    var view = new Float32Array(16);
    var eyeX = g_sim3d.cameraDist * Math.cos(g_sim3d.cameraAngleX) * Math.sin(g_sim3d.cameraAngleY);
    var eyeY = g_sim3d.cameraDist * Math.sin(g_sim3d.cameraAngleX);
    var eyeZ = g_sim3d.cameraDist * Math.cos(g_sim3d.cameraAngleX) * Math.cos(g_sim3d.cameraAngleY);
    var upX=0,upY=1,upZ=0;
    var zAxis=[eyeX,eyeY,eyeZ]; var zLen=Math.sqrt(zAxis[0]*zAxis[0]+zAxis[1]*zAxis[1]+zAxis[2]*zAxis[2]);
    zAxis[0]/=zLen;zAxis[1]/=zLen;zAxis[2]/=zLen;
    var xAxis=[upY*zAxis[2]-upZ*zAxis[1],upZ*zAxis[0]-upX*zAxis[2],upX*zAxis[1]-upY*zAxis[0]];
    var xLen=Math.sqrt(xAxis[0]*xAxis[0]+xAxis[1]*xAxis[1]+xAxis[2]*xAxis[2]);
    xAxis[0]/=xLen;xAxis[1]/=xLen;xAxis[2]/=xLen;
    var yAxis=[zAxis[1]*xAxis[2]-zAxis[2]*xAxis[1],zAxis[2]*xAxis[0]-zAxis[0]*xAxis[2],zAxis[0]*xAxis[1]-zAxis[1]*xAxis[0]];
    view[0]=xAxis[0];view[1]=yAxis[0];view[2]=-zAxis[0];view[3]=0;
    view[4]=xAxis[1];view[5]=yAxis[1];view[6]=-zAxis[1];view[7]=0;
    view[8]=xAxis[2];view[9]=yAxis[2];view[10]=-zAxis[2];view[11]=0;
    view[12]=-(xAxis[0]*eyeX+xAxis[1]*eyeY+xAxis[2]*eyeZ);
    view[13]=-(yAxis[0]*eyeX+yAxis[1]*eyeY+yAxis[2]*eyeZ);
    view[14]=(zAxis[0]*eyeX+zAxis[1]*eyeY+zAxis[2]*eyeZ);
    view[15]=1;
    var mvp = new Float32Array(16);
    for(var i=0;i<4;i++) for(var j=0;j<4;j++) { var sum=0; for(var k=0;k<4;k++) sum+=proj[i+4*k]*view[k+4*j]; mvp[i+4*j]=sum; }
    return mvp;
}

function sim3dDrawGrid(mvp) {
    var gl = g_sim3d.gl, gs = g_sim3d.gridSize;
    var verts = [];
    for(var i=-gs;i<=gs;i+=1) { verts.push(i,0,-gs,i,0,gs,-gs,0,i,gs,0,i); }
    var buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(verts), gl.STATIC_DRAW);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(0);
    gl.uniformMatrix4fv(g_sim3d.uMVP, false, mvp);
    gl.uniform3f(g_sim3d.uColor, 0.2, 0.25, 0.3);
    gl.drawArrays(gl.LINES, 0, verts.length/3);
    gl.deleteBuffer(buf);
}

function sim3dDrawCube(mvp, x, y, z, sx, sy, sz, color) {
    var gl = g_sim3d.gl;
    var cx=x,cy=y,cz=z,hx=sx/2,hy=sy/2,hz=sz/2;
    var verts = [
        cx-hx,cy-hy,cz-hz, cx+hx,cy-hy,cz-hz, cx+hx,cy-hy,cz-hz, cx+hx,cy-hy,cz+hz,
        cx+hx,cy-hy,cz+hz, cx-hx,cy-hy,cz+hz, cx-hx,cy-hy,cz+hz, cx-hx,cy-hy,cz-hz,
        cx-hx,cy+hy,cz-hz, cx+hx,cy+hy,cz-hz, cx+hx,cy+hy,cz-hz, cx+hx,cy+hy,cz+hz,
        cx+hx,cy+hy,cz+hz, cx-hx,cy+hy,cz+hz, cx-hx,cy+hy,cz+hz, cx-hx,cy+hy,cz-hz,
        cx-hx,cy-hy,cz-hz, cx-hx,cy+hy,cz-hz, cx+hx,cy-hy,cz-hz, cx+hx,cy+hy,cz-hz,
        cx+hx,cy-hy,cz+hz, cx+hx,cy+hy,cz+hz, cx-hx,cy-hy,cz+hz, cx-hx,cy+hy,cz+hz
    ];
    var buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(verts), gl.STATIC_DRAW);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(0);
    gl.uniformMatrix4fv(g_sim3d.uMVP, false, mvp);
    gl.uniform3f(g_sim3d.uColor, color[0], color[1], color[2]);
    gl.drawArrays(gl.LINES, 0, verts.length/3);
    gl.deleteBuffer(buf);
}

function sim3dRenderLoop() {
    var gl = g_sim3d.gl;
    if (!gl) return;
    gl.clearColor(0.04, 0.04, 0.1, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.viewport(0, 0, g_sim3d.canvas.width, g_sim3d.canvas.height);
    var mvp = sim3dBuildMVP();
    if (g_sim3d.showGrid) sim3dDrawGrid(mvp);
    var hasData = g_sim3d.robots.length > 0 || g_sim3d.objects.length > 0;
    for (var i = 0; i < g_sim3d.robots.length; i++) {
        var r = g_sim3d.robots[i];
        sim3dDrawCube(mvp, r.x||0, r.y||0.5, r.z||0, 0.6, 1.0, 0.4, [0, 0.8, 1]);
        sim3dDrawCube(mvp, r.x||0, r.y||1.1, r.z||0, 0.3, 0.3, 0.3, [1, 0.6, 0]);
    }
    for (var j = 0; j < g_sim3d.objects.length; j++) {
        var o = g_sim3d.objects[j];
        sim3dDrawCube(mvp, o.x, o.y, o.z, o.sx, o.sy, o.sz, o.color||[0.2,1,0.2]);
    }
    g_sim3d.animId = requestAnimationFrame(sim3dRenderLoop);
}

function sim3dResetView() { g_sim3d.cameraAngleX=0.5; g_sim3d.cameraAngleY=-0.3; g_sim3d.cameraDist=15; }
function sim3dToggleGrid() { g_sim3d.showGrid=!g_sim3d.showGrid; }
function sim3dAddRobot() {
    /* 通过后端API在仿真环境中创建真实机器人 */
    if (window.SelfLnnApi && typeof window.SelfLnnApi.addSimulationRobot === 'function') {
        SelfLnnApi.addSimulationRobot({ x: 0, y: 0, z: 0 }).then(result => {
            if (result && result.success) {
                if (result.robot) {
                    g_sim3d.robots.push(result.robot);
                }
                document.getElementById('sim-robots').textContent = g_sim3d.robots.length;
            } else {
                console.warn('仿真机器人创建失败:', result?.error);
            }
        }).catch(err => {
            console.warn('仿真机器人创建失败:', err.message);
        });
    } else {
        console.warn('仿真API不可用，无法创建机器人');
    }
}
function sim3dClearAll() { g_sim3d.robots=[]; g_sim3d.objects=[]; document.getElementById('sim-robots').textContent='0'; }

/* ===== 额外仿真控制 ===== */
async function start3DReconstruction() { const m=document.getElementById('recon-mode').value; try{var r=await SelfLnnApi.request('/simulation/reconstruct',{method:'POST',body:JSON.stringify({mode:m})},1);var d=await r.json();if(d.success){document.getElementById('recon-points').textContent=d.point_count||0;document.getElementById('recon-objects').textContent=d.object_count||0;}}catch(e){console.warn('3D重建失败:',e.message);} }
async function executeCommand() { var c=document.getElementById('cmd-input').value;if(!c)return;var o=document.getElementById('cmd-output');o.textContent='执行中...';try{var r=await SelfLnnApi.request('/computer/execute',{method:'POST',body:JSON.stringify({command:c})},1);var d=await r.json();o.textContent=d.output||d.error||'命令已执行';}catch(e){o.textContent=e.message;} }
