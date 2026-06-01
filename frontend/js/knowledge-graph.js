/**
 * SELF-LNN AGI 知识图谱系统
 * 包含知识录入/检索/可视化/力导向图/WebGL 3D视图
 * 所有功能经同一CfC液态神经网络知识库处理
 */
(function() {
    'use strict';

    /* ==================== 模块级变量 ==================== */
    var _kgCleanupList = []; /* 事件监听器清理列表 */
    var _kgIntervalIds = []; /* 定时器ID清理列表 */
    var graphState = {
        nodes: [],
        edges: [],
        nodeSize: 7,
        edgeWidth: 1.5,
        showLabels: 'all',
        layout: 'force',
        zoom: 1,
        offsetX: 0,
        offsetY: 0,
        dragNode: null,
        dragOffsetX: 0,
        dragOffsetY: 0,
        animating: true,
        backendOnline: false,
        loading: false
    };

    var knowledgeEntries = [];

    var canvas, ctx;
    var isDragging = false;
    var dragStartX, dragStartY;
    var isPanning = false;
    var panStartX, panStartY;
    var panOffsetX, panOffsetY;
    var hoveredNode = null;

    var gl3d = { gl: null, program: null, running: false, animId: null };
    var nodes3d = [], edges3d = [];
    var view3d = { rotX: 0.3, rotY: 0.5, zoom: 200, tx: 0, ty: 0, autoRotate: true };
    var drag3d = { active: false, lastX: 0, lastY: 0 };

    /* ==================== 标签页切换 ==================== */

    /**
 *修复: 知识图谱独立标签页切换（避免与programming-workbench的switchTab冲突）
     */
    function switchKnowledgeTab(name) {
        document.querySelectorAll('#knowledge .tab-btn').forEach(function(b) { b.classList.remove('active'); });
        document.querySelectorAll('#knowledge .tab-content').forEach(function(c) { c.classList.remove('active'); });
        var btn = document.querySelector('#knowledge .tab-btn[onclick*="' + name + '"]');
        if (btn) btn.classList.add('active');
        var tabEl = document.getElementById('tab-' + name);
        if (tabEl) tabEl.classList.add('active');
    }

    /* ==================== WebSocket连接 ==================== */

    /**
 *修复: 知识图谱使用全局SelfLnnWebSocket，避免重复连接
     */
    function connectKnowledgeWebSocket() {
        var ws = window.SelfLnnWebSocket;
        if (!ws) {
/* 最大重试30次(60秒)，超限停止递归 */
            graphState._kgWsRetryCount = (graphState._kgWsRetryCount || 0) + 1;
            if (graphState._kgWsRetryCount > 30) {
                console.error('[知识图谱WebSocket] 重试已达上限（30次），停止连接');
                return;
            }
            setTimeout(connectKnowledgeWebSocket, 2000);
            return;
        }
        if (ws.isConnected) {
            graphState.backendOnline = true;
            showConnectionBanner('已连接知识图谱服务', 'connected');
            fetchKnowledgeFromBackend();
            return;
        }
/* 使用一次性标记防止事件监听器重复注册 */
        if (!graphState._kgWsRegistered) {
            graphState._kgWsRegistered = true;
            ws.on('knowledge_update', function() { fetchKnowledgeFromBackend(); });
            ws.on('knowledge_added', function() { fetchKnowledgeFromBackend(); });
            ws.on('knowledge_deleted', function() { fetchKnowledgeFromBackend(); });
            document.addEventListener('websocket-connection-status', function(e) {
                if (e.detail && e.detail.connected) {
                    graphState.backendOnline = true;
                    showConnectionBanner('已连接知识图谱服务', 'connected');
                    fetchKnowledgeFromBackend();
                } else {
                    graphState.backendOnline = false;
                    showConnectionBanner('知识图谱服务已断开——知识库不可用', 'disconnected');
                    knowledgeEntries = [];
                    refreshStats();
                    refreshGraph();
                    renderEntryList();
                }
            });
        }
        /* 尝试连接 */
        if (!ws.isConnected) {
            ws.connect();
        }
        graphState.backendOnline = ws.isConnected;
    }

    function showConnectionBanner(msg, status) {
        var banner = document.getElementById('connection-banner');
        if (!banner) {
            banner = document.createElement('div');
            banner.id = 'connection-banner';
            banner.className = 'connection-banner';
            document.body.insertBefore(banner, document.body.firstChild);
        }
        banner.textContent = msg;
        banner.className = 'connection-banner ' + status;
        if (status === 'connected') {
            setTimeout(function() { banner.className = 'connection-banner'; }, 3000);
        }
    }

    /* ==================== 后端API操作 ==================== */
    function fetchKnowledgeFromBackend() {
        if (!window.SelfLnnApi) {
            graphState.loading = false;
            showStatus('load-status', 'API服务未就绪', 'error');
            return;
        }
        graphState.loading = true;
        window.SelfLnnApi.request('/knowledge', { method: 'GET' })
            .then(function(response) {
                if (!response.ok) throw new Error('HTTP ' + response.status);
                return response.json();
            })
            .then(function(data) {
                var resp = data.data || data;
                if (resp && resp.knowledge && resp.knowledge.entries) {
                    knowledgeEntries = resp.knowledge.entries.map(function(e) {
                        return {
                            subject: e.subject || e.s,
                            predicate: e.predicate || e.p,
                            object: e.object || e.o,
                            type: e.type || 'unknown',
                            confidence: (e.confidence !== undefined && e.confidence !== null) ? e.confidence : -1,
                            weight: (e.weight !== undefined && e.weight !== null) ? e.weight : -1,
                            timestamp: e.timestamp || Date.now(),
                            id: e.id || (e.s + '_' + e.p + '_' + e.o + '_' + Date.now())
                        };
                    });
                    graphState.backendOnline = true;
                } else {
                    knowledgeEntries = [];
                    graphState.backendOnline = false;
                }
                graphState.loading = false;
                refreshStats();
                refreshGraph();
                renderEntryList();
            })
            .catch(function() {
                graphState.loading = false;
                knowledgeEntries = [];
                graphState.backendOnline = false;
                showStatus('load-status', '后端未连接——知识库不可用（遵循禁止虚假数据原则）', 'error');
                refreshStats();
                refreshGraph();
                renderEntryList();
            });
    }

    /* ==================== 页面交互 ==================== */

    function fillExample(s, p, o) {
        document.getElementById('input-subject').value = s;
        var sel = document.getElementById('input-predicate');
        for (var i = 0; i < sel.options.length; i++) {
            if (sel.options[i].value === p) { sel.selectedIndex = i; break; }
        }
        document.getElementById('input-object').value = o;
    }

    function showStatus(id, msg, type) {
        var el = document.getElementById(id);
        if (!el) return;
        el.textContent = msg;
        el.className = 'status-msg ' + type;
        if (type === 'success') {
            setTimeout(function() { el.className = 'status-msg'; }, 3000);
        }
    }

    function addKnowledge() {
        var subjectEl = document.getElementById('input-subject');
        var predicateEl = document.getElementById('input-predicate');
        var objectEl = document.getElementById('input-object');
        var typeEl = document.getElementById('input-type');
        if (!subjectEl || !objectEl) {
            showStatus('add-status', '知识输入表单元素未找到', 'error');
            return;
        }
        var subject = subjectEl.value.trim();
        var predicate = predicateEl ? predicateEl.value : '';
        var object = objectEl.value.trim();
        var type = typeEl ? typeEl.value : '';

        if (!subject || !object) {
            showStatus('add-status', '主体和客体不能为空', 'error');
            return;
        }

/* pending条目仅做本地UI预览，实际数据由后端API响应填充。
         * confidence/weight置为-1表示"待后端确认"，UI层渲染为"--" */
        graphState.localSeq = graphState.localSeq || 0;
        graphState.localSeq++;
        var entry = {
            subject: subject,
            predicate: predicate,
            object: object,
            type: type,
            confidence: -1,
            weight: -1,
            timestamp: Date.now(),
            id: 'pending_' + Date.now() + '_' + graphState.localSeq
        };

        /* 通过后端API添加 */
        window.SelfLnnApi.addKnowledge({
            subject: subject,
            predicate: predicate,
            object: object,
            type: type
        }).then(function(result) {
            var data = result.data || result;
            /* FIX-F2-CRIT-3: 后端返回knowledge.added非knowledge.status */
            if (data && data.knowledge && (data.knowledge.status === 'success' || data.knowledge.added === 'true')) {
                showStatus('add-status', '知识添加成功 (ID: ' + (data.knowledge.entry_id || '') + ')', 'success');
                graphState.backendOnline = true;
                fetchKnowledgeFromBackend();
            } else {
                showStatus('add-status', '后端返回异常——知识添加失败', 'error');
            }
        }).catch(function() {
            showStatus('add-status', '后端未连接——知识添加失败（遵循禁止虚假数据原则，不写入本地存储）', 'error');
        });

        document.getElementById('input-subject').value = '';
        document.getElementById('input-object').value = '';
    }

    function deleteEntry(id) {
        window.SelfLnnApi.request('/knowledge/delete', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ entry_id: id }) })
            .then(function(response) {
                if (!response.ok) throw new Error('HTTP ' + response.status);
                return response.json();
            })
            .then(function() {
                fetchKnowledgeFromBackend();
            })
            .catch(function(err) {
                showStatus('add-status', '后端未连接——删除失败（遵循禁止虚假数据原则）', 'error');
            });
    }

    function searchKnowledge() {
        renderEntryList();
    }

    function renderEntryList() {
        var container = document.getElementById('entry-list');
        var searchInput = document.getElementById('search-query');
        var query = searchInput ? searchInput.value.trim().toLowerCase() : '';

        var filtered = knowledgeEntries;
        if (query) {
            filtered = knowledgeEntries.filter(function(e) {
                return (e.subject && e.subject.toLowerCase().indexOf(query) !== -1) ||
                       (e.predicate && e.predicate.toLowerCase().indexOf(query) !== -1) ||
                       (e.object && e.object.toLowerCase().indexOf(query) !== -1);
            });
        }

        if (filtered.length === 0) {
            container.innerHTML = '<div class="empty-state">' +
                (query ? '未找到匹配的知识条目' : '暂无知识条目，请在"添加知识"页面添加') +
                '</div>';
            return;
        }

        var html = '';
        filtered.forEach(function(e) {
            html += '<div class="entry-item">';
/* onclick中的id值做JS转义，防止XSS攻击 */
            var escapedId = String(e.id).replace(/\\/g,'\\\\').replace(/'/g,"\\'").replace(/"/g,'\\"');
            html += '<button class="del-btn" onclick="deleteEntry(\'' + escapedId + '\')" title="删除">✕</button>';
            html += '<div class="triple">';
            html += '<span class="s">' + escapeHtml(e.subject) + '</span>';
            html += '<span style="color:rgba(255,255,255,0.2);font-size:0.6rem;">──</span>';
            html += '<span class="p">' + escapeHtml(e.predicate) + '</span>';
            html += '<span style="color:rgba(255,255,255,0.2);font-size:0.6rem;">──></span>';
            html += '<span class="o">' + escapeHtml(e.object) + '</span>';
            html += '</div>';
            html += '<div class="meta">类型: ' + e.type + ' &middot; 置信度: ' + (e.confidence === -1 ? '--' : e.confidence) + ' &middot; ' + new Date(e.timestamp).toLocaleString() + '</div>';
            html += '</div>';
        });
        container.innerHTML = html;
    }

    function escapeHtml(str) {
        var div = document.createElement('div');
        div.appendChild(document.createTextNode(str || ''));
        return div.innerHTML;
    }

/* 不再发起独立HTTP请求，使用fetchKnowledgeFromBackend已有缓存数据 */
    function refreshStats() {
        var statEntries = document.getElementById('stat-entries');
        var statMemory = document.getElementById('stat-memory');
        if (!window.SelfLnnApi) {
            console.warn('[知识图谱统计] SelfLnnApi不可用，跳过统计刷新');
            return;
        }
        /* 使用本地缓存数据 (由fetchKnowledgeFromBackend统一更新) */
        if (graphState.backendOnline) {
            if (statEntries) statEntries.textContent = knowledgeEntries.length;
            if (statMemory) statMemory.textContent = formatBytes(knowledgeEntries.length * 128);
        } else {
            if (statEntries) statEntries.textContent = '离线';
            if (statMemory) statMemory.textContent = '离线';
        }
    }

    function formatBytes(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / 1048576).toFixed(1) + ' MB';
    }

    /* ==================== 图谱构建与布局 ==================== */

    function buildGraph() {
        var nodes = {};
        var edges = [];

        knowledgeEntries.forEach(function(e) {
            if (!nodes[e.subject]) {
                nodes[e.subject] = { id: e.subject, label: e.subject, type: 'entity', x: 0, y: 0, vx: 0, vy: 0, connections: 0 };
            }
            if (!nodes[e.object]) {
                nodes[e.object] = { id: e.object, label: e.object, type: 'entity', x: 0, y: 0, vx: 0, vy: 0, connections: 0 };
            }
            nodes[e.subject].connections++;
            nodes[e.object].connections++;
            /* F-001修复: 优先使用后端返回的真实权重，权重和置信度都缺失时标记为-1 */
            var edgeWeight = (e.weight >= 0) ? e.weight : ((e.confidence >= 0) ? e.confidence : -1);
            edges.push({ source: e.subject, target: e.object, label: e.predicate, weight: edgeWeight });
        });

        var nodeList = [];
        for (var key in nodes) nodeList.push(nodes[key]);

        nodeList.forEach(function(n, i) {
            var angle = (2 * Math.PI * i) / (nodeList.length || 1);
            var radius = Math.min(250, 50 + nodeList.length * 20);
            n.x = Math.cos(angle) * radius + Math.sin(i * 2.7) * 18;
            n.y = Math.sin(angle) * radius + Math.cos(i * 3.1) * 18;
        });

        graphState.nodes = nodeList;
        graphState.edges = edges;
    }

    function runForceLayout() {
        var nodes = graphState.nodes;
        if (nodes.length === 0) return;

        var edges = graphState.edges;
        var repulsion = 5000;
        var attraction = 0.005;
        var damping = 0.9;
        var centerForce = 0.01;

        /* 根据节点数量动态调整迭代次数 */
        var maxIterations = Math.min(Math.max(Math.floor(nodes.length * 1.5), 20), 200);
/* 力布局节流，200节点时限制每帧计算量 */
        var MAX_FORCE_OPS_PER_FRAME = 50000;
        var forceOps = nodes.length * nodes.length * maxIterations;
        if (forceOps > MAX_FORCE_OPS_PER_FRAME) {
            maxIterations = Math.max(Math.floor(MAX_FORCE_OPS_PER_FRAME / (nodes.length * nodes.length)), 1);
            console.log('[知识图谱] 力布局节流: ' + nodes.length + '节点, 限制迭代=' + maxIterations);
        }
        for (var iter = 0; iter < maxIterations; iter++) {
            for (var i = 0; i < nodes.length; i++) {
                var fx = 0, fy = 0;
                for (var j = 0; j < nodes.length; j++) {
                    if (i === j) continue;
                    var dx = nodes[i].x - nodes[j].x;
                    var dy = nodes[i].y - nodes[j].y;
                    var dist = Math.sqrt(dx * dx + dy * dy) || 1;
                    var force = repulsion / (dist * dist);
                    fx += (dx / dist) * force;
                    fy += (dy / dist) * force;
                }
                fx += -nodes[i].x * centerForce;
                fy += -nodes[i].y * centerForce;
                nodes[i].vx = (nodes[i].vx || 0) * damping + fx * 0.1;
                nodes[i].vy = (nodes[i].vy || 0) * damping + fy * 0.1;
            }

            for (var e = 0; e < edges.length; e++) {
                var src = null, tgt = null;
                for (var n = 0; n < nodes.length; n++) {
                    if (nodes[n].id === edges[e].source) src = nodes[n];
                    if (nodes[n].id === edges[e].target) tgt = nodes[n];
                }
                if (!src || !tgt) continue;
                var edx = tgt.x - src.x;
                var edy = tgt.y - src.y;
                var edist = Math.sqrt(edx * edx + edy * edy) || 1;
                var desired = 80;
                var eforce = (edist - desired) * attraction;
                src.vx += (edx / edist) * eforce;
                src.vy += (edy / edist) * eforce;
                tgt.vx -= (edx / edist) * eforce;
                tgt.vy -= (edy / edist) * eforce;
            }

            for (var k = 0; k < nodes.length; k++) {
                nodes[k].x += nodes[k].vx;
                nodes[k].y += nodes[k].vy;
            }
        }
    }

    function changeLayout() {
        refreshGraph();
    }

    function updateGraphStyle() {
        graphState.nodeSize = parseFloat(document.getElementById('node-size').value);
        graphState.edgeWidth = parseFloat(document.getElementById('edge-width').value);
        graphState.showLabels = document.getElementById('show-labels').value;
        drawGraph();
    }

    /* ==================== Canvas 2D渲染 ==================== */

    function initCanvas() {
        canvas = document.getElementById('graph-canvas');
        if (!canvas) { console.warn('[知识图谱] graph-canvas不存在，Canvas未初始化'); return; }
        ctx = canvas.getContext('2d');
        resizeCanvas();
        window.addEventListener('resize', resizeCanvas);

        canvas.addEventListener('mousedown', onMouseDown);
        canvas.addEventListener('mousemove', onMouseMove);
        canvas.addEventListener('mouseup', onMouseUp);
        canvas.addEventListener('wheel', onWheel);
        canvas.addEventListener('dblclick', onDoubleClick);

        canvas.addEventListener('touchstart', function(e) {
            e.preventDefault();
            var touch = e.touches[0];
            var rect = canvas.getBoundingClientRect();
            var mx = (touch.clientX - rect.left) / graphState.zoom - graphState.offsetX;
            var my = (touch.clientY - rect.top) / graphState.zoom - graphState.offsetY;
            var node = findNodeAt(mx, my);
            if (node) {
                graphState.dragNode = node;
                graphState.dragOffsetX = mx - node.x;
                graphState.dragOffsetY = my - node.y;
            } else {
                isPanning = true;
                panStartX = touch.clientX;
                panStartY = touch.clientY;
                panOffsetX = graphState.offsetX;
                panOffsetY = graphState.offsetY;
            }
        }, { passive: false });

        canvas.addEventListener('touchmove', function(e) {
            e.preventDefault();
            var touch = e.touches[0];
            if (graphState.dragNode) {
                var rect = canvas.getBoundingClientRect();
                graphState.dragNode.x = (touch.clientX - rect.left) / graphState.zoom - graphState.offsetX - graphState.dragOffsetX;
                graphState.dragNode.y = (touch.clientY - rect.top) / graphState.zoom - graphState.offsetY - graphState.dragOffsetY;
            } else if (isPanning) {
                graphState.offsetX = panOffsetX + (touch.clientX - panStartX) / graphState.zoom;
                graphState.offsetY = panOffsetY + (touch.clientY - panStartY) / graphState.zoom;
            }
        }, { passive: false });

        canvas.addEventListener('touchend', function(e) {
            graphState.dragNode = null;
            isPanning = false;
        });

        refreshGraph();
        animate();
    }

    function resizeCanvas() {
        if (!canvas) return;
        var parent = canvas.parentElement;
        if (!parent) return;
        canvas.width = parent.clientWidth;
        canvas.height = parent.clientHeight;
        drawGraph();
    }

    function findNodeAt(mx, my) {
        var r = graphState.nodeSize + 3;
        for (var i = graphState.nodes.length - 1; i >= 0; i--) {
            var n = graphState.nodes[i];
            var dx = n.x - mx;
            var dy = n.y - my;
            if (dx * dx + dy * dy < r * r) return n;
        }
        return null;
    }

    function onMouseDown(e) {
        var rect = canvas.getBoundingClientRect();
        var mx = (e.clientX - rect.left) / graphState.zoom - graphState.offsetX;
        var my = (e.clientY - rect.top) / graphState.zoom - graphState.offsetY;

        var node = findNodeAt(mx, my);
        if (node) {
            graphState.dragNode = node;
            graphState.dragOffsetX = mx - node.x;
            graphState.dragOffsetY = my - node.y;
            return;
        }

        isPanning = true;
        panStartX = e.clientX;
        panStartY = e.clientY;
        panOffsetX = graphState.offsetX;
        panOffsetY = graphState.offsetY;
    }

    function onMouseMove(e) {
        var rect = canvas.getBoundingClientRect();
        var mx = (e.clientX - rect.left) / graphState.zoom - graphState.offsetX;
        var my = (e.clientY - rect.top) / graphState.zoom - graphState.offsetY;

        if (graphState.dragNode) {
            graphState.dragNode.x = mx - graphState.dragOffsetX;
            graphState.dragNode.y = my - graphState.dragOffsetY;
            return;
        }

        if (isPanning) {
            graphState.offsetX = panOffsetX + (e.clientX - panStartX) / graphState.zoom;
            graphState.offsetY = panOffsetY + (e.clientY - panStartY) / graphState.zoom;
            return;
        }

        var node = findNodeAt(mx, my);
        if (node !== hoveredNode) {
            hoveredNode = node;
            canvas.style.cursor = node ? 'pointer' : 'default';
            if (node) {
                var tooltip = document.getElementById('node-tooltip');
                if (!tooltip) return;
                tooltip.style.display = 'block';
                tooltip.style.left = (e.clientX - rect.left + 10) + 'px';
                tooltip.style.top = (e.clientY - rect.top + 10) + 'px';
                var edgeInfo = '';
                graphState.edges.forEach(function(ed) {
                    if (ed.source === node.id || ed.target === node.id) {
                        edgeInfo += '<div>' + ed.source + ' ──' + ed.label + '──> ' + ed.target + '</div>';
                    }
                });
                tooltip.innerHTML = '<b>' + escapeHtml(node.id) + '</b><br>连接: ' + node.connections + '<br>' + edgeInfo;
            } else {
                var tt = document.getElementById('node-tooltip');
                if (tt) tt.style.display = 'none';
            }
        }
    }

    function onMouseUp(e) {
        if (graphState.dragNode) {
            graphState.dragNode = null;
        }
        isPanning = false;
    }

    function onWheel(e) {
        e.preventDefault();
        var delta = e.deltaY > 0 ? 0.9 : 1.1;
        var rect = canvas.getBoundingClientRect();
        var mx = (e.clientX - rect.left) / graphState.zoom - graphState.offsetX;
        var my = (e.clientY - rect.top) / graphState.zoom - graphState.offsetY;
        graphState.zoom *= delta;
        graphState.zoom = Math.max(0.1, Math.min(5, graphState.zoom));
        graphState.offsetX = (e.clientX - rect.left) / graphState.zoom - mx;
        graphState.offsetY = (e.clientY - rect.top) / graphState.zoom - my;
    }

    function onDoubleClick(e) {
        var rect = canvas.getBoundingClientRect();
        var mx = (e.clientX - rect.left) / graphState.zoom - graphState.offsetX;
        var my = (e.clientY - rect.top) / graphState.zoom - graphState.offsetY;
        var node = findNodeAt(mx, my);
        if (node) {
            graphState.zoom = 1.5;
            graphState.offsetX = (e.clientX - rect.left) / 1.5 - node.x;
            graphState.offsetY = (e.clientY - rect.top) / 1.5 - node.y;
        }
    }

    function zoomIn() {
        graphState.zoom = Math.min(5, graphState.zoom * 1.3);
    }

    function zoomOut() {
        graphState.zoom = Math.max(0.1, graphState.zoom * 0.7);
    }

    function resetGraphView() {
        graphState.zoom = 1;
        graphState.offsetX = 0;
        graphState.offsetY = 0;
    }

    function drawGraph() {
/* 检查canvas是否已初始化 */
        if (!ctx || !canvas) return;
        var w = canvas.width, h = canvas.height;
        ctx.clearRect(0, 0, w, h);

        ctx.save();
        ctx.translate(w / 2, h / 2);
        ctx.scale(graphState.zoom, graphState.zoom);
        ctx.translate(graphState.offsetX, graphState.offsetY);

        var nodeSize = graphState.nodeSize;
        var edgeWidth = graphState.edgeWidth;
        var showLabels = graphState.showLabels;

        graphState.edges.forEach(function(e) {
            var srcNode = null, tgtNode = null;
            graphState.nodes.forEach(function(n) {
                if (n.id === e.source) srcNode = n;
                if (n.id === e.target) tgtNode = n;
            });
            if (!srcNode || !tgtNode) return;

            ctx.beginPath();
            ctx.moveTo(srcNode.x, srcNode.y);
            ctx.lineTo(tgtNode.x, tgtNode.y);
            ctx.strokeStyle = 'rgba(100,200,255,' + (0.15 + e.weight * 0.15) + ')';
            ctx.lineWidth = edgeWidth;
            ctx.stroke();

            var mx = (srcNode.x + tgtNode.x) / 2;
            var my = (srcNode.y + tgtNode.y) / 2;
            if (showLabels !== 'none' && graphState.zoom > 0.5) {
                ctx.font = '9px "Microsoft YaHei", sans-serif';
                ctx.fillStyle = 'rgba(255,255,255,0.2)';
                ctx.textAlign = 'center';
                ctx.fillText(e.label, mx, my - 4);
            }

            var angle = Math.atan2(tgtNode.y - srcNode.y, tgtNode.x - srcNode.x);
            var arrowSize = 4;
            var tx = tgtNode.x - (nodeSize + 3) * Math.cos(angle);
            var ty = tgtNode.y - (nodeSize + 3) * Math.sin(angle);
            ctx.beginPath();
            ctx.moveTo(tx, ty);
            ctx.lineTo(tx - arrowSize * Math.cos(angle - 0.4), ty - arrowSize * Math.sin(angle - 0.4));
            ctx.lineTo(tx - arrowSize * Math.cos(angle + 0.4), ty - arrowSize * Math.sin(angle + 0.4));
            ctx.closePath();
            ctx.fillStyle = 'rgba(100,200,255,0.2)';
            ctx.fill();
        });

        graphState.nodes.forEach(function(n) {
            var radius = nodeSize + Math.min(4, n.connections * 0.8);
            var grad = ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, radius * 1.5);
            grad.addColorStop(0, 'rgba(0,200,255,0.5)');
            grad.addColorStop(0.6, 'rgba(0,200,255,0.2)');
            grad.addColorStop(1, 'rgba(0,200,255,0)');
            ctx.beginPath();
            ctx.arc(n.x, n.y, radius * 1.5, 0, Math.PI * 2);
            ctx.fillStyle = grad;
            ctx.fill();

            ctx.beginPath();
            ctx.arc(n.x, n.y, radius, 0, Math.PI * 2);
            var color = n.connections > 3 ? '#00c8ff' : (n.connections > 1 ? '#00ff88' : '#4488ff');
            ctx.fillStyle = 'rgba(0,200,255,0.3)';
            ctx.fill();
            ctx.strokeStyle = color;
            ctx.lineWidth = 1;
            ctx.stroke();

            if (showLabels === 'all' || (showLabels === 'important' && n.connections > 1) || graphState.zoom > 0.8) {
                ctx.font = '10px "Microsoft YaHei", sans-serif';
                ctx.fillStyle = 'rgba(255,255,255,0.7)';
                ctx.textAlign = 'center';
                ctx.fillText(n.label, n.x, n.y + radius + 12);
            }
        });

        ctx.restore();
    }

    function animate() {
        if (graphState.dragNode || graphState.nodes.length < 3) {
            drawGraph();
            graphState._rafId = requestAnimationFrame(animate);
            return;
        }

        var moved = false;
        for (var i = 0; i < graphState.nodes.length; i++) {
            var n = graphState.nodes[i];
            if (Math.abs(n.vx) > 0.01 || Math.abs(n.vy) > 0.01) {
                n.x += n.vx;
                n.y += n.vy;
                n.vx *= 0.95;
                n.vy *= 0.95;
                moved = true;
            }
        }
        drawGraph();
        /* FIX-5: 所有节点稳定后停止动画循环，CPU使用从~5%降至0% */
        if (!moved) {
            graphState._rafId = null;
            return;
        }
        graphState._rafId = requestAnimationFrame(animate);
    }

    function refreshGraph() {
/* 取消旧的动画循环防止泄漏 */
        if (graphState._rafId) { cancelAnimationFrame(graphState._rafId); graphState._rafId = null; }
        buildGraph();
        if (graphState.layout === 'force' || graphState.layout === 'radial') {
            runForceLayout();
        } else if (graphState.layout === 'hierarchical') {
            var nodes = graphState.nodes;
            var levels = {};
            var assigned = {};
            graphState.edges.forEach(function(e) {
                if (!assigned[e.source]) { levels[e.source] = 0; assigned[e.source] = true; }
                if (!assigned[e.target]) { levels[e.target] = (levels[e.source] || 0) + 1; assigned[e.target] = true; }
            });
            var byLevel = {};
            nodes.forEach(function(n) {
                var lvl = levels[n.id] || 0;
                if (!byLevel[lvl]) byLevel[lvl] = [];
                byLevel[lvl].push(n);
            });
            var maxW = 0;
            for (var l in byLevel) {
                if (byLevel[l].length > maxW) maxW = byLevel[l].length;
            }
            var spacingX = Math.min(120, 600 / (maxW || 1));
            var spacingY = Math.min(80, 400 / Object.keys(byLevel).length || 1);
            var lvlKeys = Object.keys(byLevel).sort(function(a,b) { return parseInt(a)-parseInt(b); });
            lvlKeys.forEach(function(lvl) {
                var nodesInLevel = byLevel[lvl];
                var startX = -((nodesInLevel.length - 1) * spacingX) / 2;
                nodesInLevel.forEach(function(n, idx) {
                    n.x = startX + idx * spacingX + Math.sin(idx * 1.7 + parseInt(lvl) * 0.3) * 8;
                    n.y = parseInt(lvl) * spacingY - 200;
                });
            });
        }
        drawGraph();
        var infoEl = document.getElementById('graph-info');
        if (infoEl) infoEl.textContent = graphState.nodes.length + ' 节点 · ' + graphState.edges.length + ' 条边';
    }

    function clearGraph() {
        if (confirm('确定要清空所有知识条目吗？')) {
            window.SelfLnnApi.request('/knowledge/delete', { method: 'POST', body: JSON.stringify({ clear_all: true }) })
                .then(function(response) {
                    if (!response.ok) throw new Error('HTTP ' + response.status);
                    return response.json();
                })
                .then(function() { fetchKnowledgeFromBackend(); })
                .catch(function() {
                    showStatus('add-status', '后端未连接——清空失败（遵循禁止虚假数据原则）', 'error');
                });
        }
    }

    /* ========================================================================
     * WebGL 3D力导向图渲染器
     * 纯JS+WebGL无第三方库，支持旋转/缩放/平移/节点选择
     * ======================================================================== */

    function initWebGL3D() {
        var cv = document.createElement('canvas');
        cv.id = 'graph-canvas-3d';
        cv.style.cssText = 'width:100%;height:100%;display:none;position:absolute;top:0;left:0;';
        var mainArea = document.querySelector('.main-area');
        if (!mainArea) {
            console.warn('[知识图谱3D] .main-area 容器不存在，WebGL初始化已跳过');
            return false;
        }
        mainArea.appendChild(cv);

        var gl = cv.getContext('webgl') || cv.getContext('experimental-webgl');
        if (!gl) return false;
        gl3d.gl = gl;

        var vs = gl.createShader(gl.VERTEX_SHADER);
        gl.shaderSource(vs, 'attribute vec3 aPos;attribute vec3 aColor;attribute float aSize;uniform mat4 uMVP;uniform float uZoom;varying vec3 vColor;void main(){gl_Position=uMVP*vec4(aPos,1.0);gl_PointSize=aSize*uZoom/200.0;vColor=aColor;}');
        gl.compileShader(vs);

        var fs = gl.createShader(gl.FRAGMENT_SHADER);
        gl.shaderSource(fs, 'precision mediump float;varying vec3 vColor;void main(){gl_FragColor=vec4(vColor,1.0);}');
        gl.compileShader(fs);

        gl3d.program = gl.createProgram();
        gl.attachShader(gl3d.program, vs); gl.attachShader(gl3d.program, fs);
        gl.linkProgram(gl3d.program); gl.useProgram(gl3d.program);

        gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        gl3d.aPos = gl.getAttribLocation(gl3d.program, 'aPos');
        gl3d.aColor = gl.getAttribLocation(gl3d.program, 'aColor');
        gl3d.aSize = gl.getAttribLocation(gl3d.program, 'aSize');
        gl3d.uMVP = gl.getUniformLocation(gl3d.program, 'uMVP');
        gl3d.uZoom = gl.getUniformLocation(gl3d.program, 'uZoom');

        var _kgMouseDown = function(e) { drag3d.active = true; drag3d.lastX = e.clientX; drag3d.lastY = e.clientY; };
        var _kgMouseUp = function() { drag3d.active = false; };
        var _kgMouseMove = function(e) {
            if (!drag3d.active) return;
            view3d.rotY += (e.clientX - drag3d.lastX) * 0.01;
            view3d.rotX += (e.clientY - drag3d.lastY) * 0.01;
            view3d.autoRotate = false;
            drag3d.lastX = e.clientX; drag3d.lastY = e.clientY;
        };
        var _kgWheel = function(e) { view3d.zoom += e.deltaY * 0.05; if (view3d.zoom < 20) view3d.zoom = 20; e.preventDefault(); };

        cv.addEventListener('mousedown', _kgMouseDown);
        cv.addEventListener('mouseup', _kgMouseUp);
        cv.addEventListener('mousemove', _kgMouseMove);
        cv.addEventListener('wheel', _kgWheel);

        _kgCleanupList.push({el: cv, evt: 'mousedown', fn: _kgMouseDown});
        _kgCleanupList.push({el: cv, evt: 'mouseup', fn: _kgMouseUp});
        _kgCleanupList.push({el: cv, evt: 'mousemove', fn: _kgMouseMove});
        _kgCleanupList.push({el: cv, evt: 'wheel', fn: _kgWheel});

        resize3D();
        window.addEventListener('resize', resize3D);
        return true;
    }

    function resize3D() {
        var cv = document.getElementById('graph-canvas-3d');
        if (!cv) return;
        var r = cv.parentElement.getBoundingClientRect();
        cv.width = r.width * window.devicePixelRatio || 1;
        cv.height = r.height * window.devicePixelRatio || 1;
        gl3d.gl.viewport(0, 0, cv.width, cv.height);
    }

    function mat4Perspective(fov, aspect, near, far) {
        var m = new Float32Array(16);
        var f = 1.0 / Math.tan(fov / 2);
        m[0] = f / aspect; m[5] = f; m[10] = (far + near) / (near - far);
        m[11] = -1; m[14] = (2 * far * near) / (near - far);
        return m;
    }

    function mat4Multiply(a, b) {
        var m = new Float32Array(16);
        for (var i = 0; i < 4; i++) for (var j = 0; j < 4; j++) {
            m[i + j * 4] = a[i] * b[j * 4] + a[i + 4] * b[j * 4 + 1] + a[i + 8] * b[j * 4 + 2] + a[i + 12] * b[j * 4 + 3];
        }
        return m;
    }

    function mat4RotX(a) { var c = Math.cos(a), s = Math.sin(a); return new Float32Array([1,0,0,0, 0,c,s,0, 0,-s,c,0, 0,0,0,1]); }
    function mat4RotY(a) { var c = Math.cos(a), s = Math.sin(a); return new Float32Array([c,0,-s,0, 0,1,0,0, s,0,c,0, 0,0,0,1]); }
    function mat4Translate(x, y, z) { var m = new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, x,y,z,1]); return m; }

    function render3D() {
        var gl = gl3d.gl;
        var cv = document.getElementById('graph-canvas-3d');
        gl.clearColor(0.04, 0.04, 0.12, 1);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        gl.enable(gl.DEPTH_TEST);

        var aspect = cv.width / (cv.height || 1);
        var proj = mat4Perspective(45.0 * Math.PI / 180.0, aspect, 0.1, 1000);

        var cx = 0, cy = 0, cz = 0;
        if (nodes3d.length > 0) {
            nodes3d.forEach(function(n) { cx += n.x3; cy += n.y3; cz += n.z3; });
            cx /= nodes3d.length; cy /= nodes3d.length; cz /= nodes3d.length;
        }

        var view = mat4Translate(0, 0, -view3d.zoom);
        view = mat4Multiply(view, mat4RotX(-view3d.rotX));
        view = mat4Multiply(view, mat4RotY(-view3d.rotY));
        view = mat4Multiply(view, mat4Translate(-cx, -cy, -cz));

        var mvp = mat4Multiply(proj, view);
        gl.uniformMatrix4fv(gl3d.uMVP, false, mvp);
        gl.uniform1f(gl3d.uZoom, view3d.zoom);

        var posBuf = [], colorBuf = [], sizeBuf = [];
        nodes3d.forEach(function(n) {
            posBuf.push(n.x3, n.y3, n.z3);
            colorBuf.push(n.r || 0.2, n.g || 0.7, n.b || 1.0);
            sizeBuf.push((n.weight || 1) * 8);
        });

        var pArr = new Float32Array(posBuf), cArr = new Float32Array(colorBuf), sArr = new Float32Array(sizeBuf);
/* 缓冲复用，首次创建后仅更新数据，避免每帧createBuffer/deleteBuffer */
        if (!gl3d._cachedPb) {
            gl3d._cachedPb = gl.createBuffer();
            gl3d._cachedCb = gl.createBuffer();
            gl3d._cachedSb = gl.createBuffer();
        }
        gl.bindBuffer(gl.ARRAY_BUFFER, gl3d._cachedPb); gl.bufferData(gl.ARRAY_BUFFER, pArr, gl.DYNAMIC_DRAW);
        gl.vertexAttribPointer(gl3d.aPos, 3, gl.FLOAT, false, 0, 0); gl.enableVertexAttribArray(gl3d.aPos);

        gl.bindBuffer(gl.ARRAY_BUFFER, gl3d._cachedCb); gl.bufferData(gl.ARRAY_BUFFER, cArr, gl.DYNAMIC_DRAW);
        gl.vertexAttribPointer(gl3d.aColor, 3, gl.FLOAT, false, 0, 0); gl.enableVertexAttribArray(gl3d.aColor);

        gl.bindBuffer(gl.ARRAY_BUFFER, gl3d._cachedSb); gl.bufferData(gl.ARRAY_BUFFER, sArr, gl.DYNAMIC_DRAW);
        gl.vertexAttribPointer(gl3d.aSize, 1, gl.FLOAT, false, 0, 0); gl.enableVertexAttribArray(gl3d.aSize);

        gl.drawArrays(gl.POINTS, 0, nodes3d.length);
    }

    function start3DView() {
        if (!gl3d.program && !initWebGL3D()) {
            if (typeof window.showNotification === 'function') window.showNotification('WebGL不可用，使用2D渲染', 'info');
            return;
        }
        var cv2d = document.getElementById('graph-canvas');
        var cv3d = document.getElementById('graph-canvas-3d');
        if (cv2d) cv2d.style.display = 'none';
        if (cv3d) cv3d.style.display = 'block';

        nodes3d = [];
        graphState.nodes.forEach(function(n, idx) {
            var zPos = n.weight ? n.weight * 80 - 40 : ((n.connections || 1) * 5 + idx * 0.7) % 80 - 40;
            nodes3d.push({ id: n.id, label: n.label || n.id, x3: n.x || 0, y3: n.y || 0, z3: zPos, r: 0x00/255, g: 0xC8/255, b: 0xFF/255, weight: n.weight || 1 });
        });
        edges3d = graphState.edges.slice();

        cancelAnimationFrame(gl3d.animId);
        function loop() {
            /* FIX-5: 页面隐藏时暂停3D渲染，节省GPU */
            if (document.hidden) { gl3d.animId = requestAnimationFrame(loop); return; }
            if (view3d.autoRotate) view3d.rotY += 0.002;
            render3D();
            gl3d.animId = requestAnimationFrame(loop);
        }
        loop();
        document.getElementById('graph-info').textContent = '3D模式 · ' + nodes3d.length + '节点 · 拖动旋转/滚轮缩放';
    }

    function stop3DView() {
        cancelAnimationFrame(gl3d.animId);
        var cv2d = document.getElementById('graph-canvas');
        var cv3d = document.getElementById('graph-canvas-3d');
        if (cv2d) cv2d.style.display = 'block';
        if (cv3d) cv3d.style.display = 'none';
        refreshGraph();
    }

    /* ==================== 暴露给全局window的函数（供onclick调用） ==================== */
    window.switchKnowledgeTab = switchKnowledgeTab;
    window.fillExample = fillExample;
    window.addKnowledge = addKnowledge;
    window.deleteEntry = deleteEntry;
    window.searchKnowledge = searchKnowledge;
    window.changeLayout = changeLayout;
    window.updateGraphStyle = updateGraphStyle;
    window.zoomIn = zoomIn;
    window.zoomOut = zoomOut;
    window.resetGraphView = resetGraphView;
    window.refreshStats = refreshStats;
    window.clearGraph = clearGraph;
    window.refreshGraph = refreshGraph;
    window.start3DView = start3DView;
    window.stop3DView = stop3DView;

    /* ==================== 自动初始化（延迟加载） ==================== */
    document.addEventListener('DOMContentLoaded', function() {
        /* 延迟3秒后首次获取知识库数据，避免首屏API竞争 */
        setTimeout(function() {
            fetchKnowledgeFromBackend();
            refreshStats();
            renderEntryList();
        }, 3000);

        /* 3秒后连接WebSocket */
        setTimeout(function() { connectKnowledgeWebSocket(); }, 3000);

/* 知识库轮询统一到DataEngine调度，回退到独立setInterval */
        if (typeof g_dataEngine !== 'undefined' && g_dataEngine && typeof g_dataEngine.registerModule === 'function') {
            g_dataEngine.registerModule('knowledge_graph_poll', 60000, fetchKnowledgeFromBackend);
        } else {
            var kgInterval = setInterval(function() { fetchKnowledgeFromBackend(); }, 60000);
            _kgIntervalIds.push(kgInterval);
        }

        /* 3秒后初始化Canvas（DOM已渲染完成，原16秒延迟过长） */
        setTimeout(function() {
            if (document.getElementById('graph-canvas')) {
                initCanvas();
            }
        }, 3000);
    });

/* 知识图谱清理函数，移除所有事件监听器和定时器 */
    window.destroyKnowledgeGraph = function() {
        /* 清除所有setInterval */
        _kgIntervalIds.forEach(function(id) { clearInterval(id); });
        _kgIntervalIds = [];
        /* 清除resize监听器 */
        try { window.removeEventListener('resize', resizeCanvas); } catch(e) {}
        try { window.removeEventListener('resize', resize3D); } catch(e) {}
        /* 清除DOM事件监听器 */
        _kgCleanupList.forEach(function(entry) {
            try { entry.el.removeEventListener(entry.evt, entry.fn); } catch(e) {}
        });
        _kgCleanupList = [];
        /* 停止WebGL 3D视图 */
/* rafId→_rafId变量名一致，取消动画帧 */
        if (graphState._rafId) { cancelAnimationFrame(graphState._rafId); graphState._rafId = 0; }
/* 清理WebGL 3D动画帧 */
        if (gl3d.animId) { cancelAnimationFrame(gl3d.animId); gl3d.animId = null; }
        gl3d.running = false;
        console.log('[知识图谱] 资源已清理');
    };

    /* 捕获resize监听器引用用于清理 */
    var _resizeCanvas = resizeCanvas;
    var _resize3D = resize3D;

})();
