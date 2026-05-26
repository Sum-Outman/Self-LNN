/**
 * @file product-design.js
 * @brief 产品设计模块前端控制
 *
 * 连接后端产品设计引擎API，实现：
 * - 需求文本输入 → 自动生成产品规格
 * - 引擎状态查询
 * - 设计结果可视化展示
 *
 * 严格遵循禁止虚假数据原则：所有数据均通过真实后端API获取。
 */

(function() {
    'use strict';

    var PD_REQUIREMENT_INPUT = null;
    var PD_GENERATE_BTN = null;
    var PD_STATUS_BTN = null;
    var PD_STATUS_CARD = null;
    var PD_STATUS_CONTENT = null;
    var PD_RESULT_CARD = null;
    var PD_RESULT_CONTENT = null;
    var PD_INITIALIZED = false;

    function pdInit() {
        if (PD_INITIALIZED) return;
        PD_REQUIREMENT_INPUT = document.getElementById('pd-requirement');
        PD_GENERATE_BTN = document.getElementById('pd-generate-btn');
        PD_STATUS_BTN = document.getElementById('pd-status-btn');
        PD_STATUS_CARD = document.getElementById('pd-status-card');
        PD_STATUS_CONTENT = document.getElementById('pd-status-content');
        PD_RESULT_CARD = document.getElementById('pd-result-card');
        PD_RESULT_CONTENT = document.getElementById('pd-result-content');

        if (!PD_GENERATE_BTN) return;
        PD_GENERATE_BTN.addEventListener('click', pdGenerate);
        if (PD_STATUS_BTN) PD_STATUS_BTN.addEventListener('click', pdCheckStatus);
        PD_INITIALIZED = true;
    }

    /* 后端API基础URL — 使用统一api-service.exe进行请求 */
    function pdGetBaseUrl() {
        var host = window.location.hostname || '127.0.0.1';
        var port = 8080;
        return 'http://' + host + ':' + port;
    }

    function pdSetStatus(html, show) {
        if (!PD_STATUS_CARD || !PD_STATUS_CONTENT) return;
        PD_STATUS_CONTENT.innerHTML = html;
        PD_STATUS_CARD.style.display = show ? 'block' : 'none';
    }

    function pdSetResult(html, show) {
        if (!PD_RESULT_CARD || !PD_RESULT_CONTENT) return;
        PD_RESULT_CONTENT.innerHTML = html;
        PD_RESULT_CARD.style.display = show ? 'block' : 'none';
    }

    function pdShowLoading() {
        pdSetResult('<div style="text-align:center;padding:30px"><span style="color:#4fc3f7">正在调用产品设计引擎...</span><div class="spinner" style="margin:15px auto"></div></div>', true);
    }

    /* 生成设计方案 */
    function pdGenerate() {
        pdInit();
        if (!PD_REQUIREMENT_INPUT) return;
        var reqText = PD_REQUIREMENT_INPUT.value.trim();
        if (!reqText) {
            pdSetResult('<div style="color:#ff8a65;padding:15px">请输入产品需求描述</div>', true);
            return;
        }
        pdShowLoading();
        var url = '/product/spec';
        var postData = JSON.stringify({ requirement: reqText });

        window.SelfLnnApi.request(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: postData
        })
        .then(function(resp) { return resp.json(); })
        .then(function(data) {
            if (data.error) {
                pdSetResult('<div style="color:#ef5350;padding:15px">错误：' + pdEscapeHtml(data.error) + '</div>', true);
                return;
            }
            var spec = data.product_spec;
            if (!spec) {
                pdSetResult('<div style="color:#ef5350;padding:15px">未收到设计结果</div>', true);
                return;
            }
            var typeNames = ['硬件产品', '软件产品', '系统产品', '自定义产品'];
            var typeName = typeNames[spec.type] || '未知类型';
            var evalHtml = '';
            if (spec.evaluation) {
                evalHtml = [
                    '<div style="background:#1a1a2e;border-radius:8px;padding:15px;margin-top:15px">',
                    '<h4 style="color:#81c784;margin:0 0 10px 0">设计评估</h4>',
                    '<table style="width:100%;color:#ccc;font-size:13px">',
                    '<tr><td>可行性</td><td style="text-align:right">' + (spec.evaluation.feasibility * 100).toFixed(1) + '%</td></tr>',
                    '<tr><td>成本效益</td><td style="text-align:right">' + (spec.evaluation.cost_effectiveness * 100).toFixed(1) + '%</td></tr>',
                    '<tr><td>创新程度</td><td style="text-align:right">' + (spec.evaluation.innovation * 100).toFixed(1) + '%</td></tr>',
                    '<tr><td>市场潜力</td><td style="text-align:right">' + (spec.evaluation.market_potential * 100).toFixed(1) + '%</td></tr>',
                    '</table>',
                    spec.evaluation.recommendations && spec.evaluation.recommendations.length > 0 ?
                    '<h4 style="color:#ffcc80;margin:10px 0 5px 0">改进建议</h4><ul style="color:#aaa;font-size:13px;padding-left:20px">' +
                    spec.evaluation.recommendations.map(function(r) { return '<li>' + pdEscapeHtml(r) + '</li>'; }).join('') +
                    '</ul>' : '',
                    '</div>'
                ].join('');
            }
            var featuresHtml = '';
            if (spec.features && spec.features.length > 0) {
                featuresHtml = '<h4 style="color:#64b5f6;margin:10px 0 5px 0">功能特性</h4><ul style="color:#aaa;font-size:13px;padding-left:20px">' +
                    spec.features.map(function(f) { return '<li>' + pdEscapeHtml(f) + '</li>'; }).join('') + '</ul>';
            }
            var resultHtml = [
                '<div style="background:#12122a;border-radius:8px;padding:20px">',
                '<h3 style="color:#e0e0e0;margin:0 0 8px 0;font-size:18px">' + pdEscapeHtml(spec.name || '未命名产品') + '</h3>',
                '<span style="background:#1565c0;padding:2px 10px;border-radius:10px;font-size:12px">' + typeName + '</span>',
                '<p style="color:#b0b0b0;margin:12px 0;font-size:14px">' + pdEscapeHtml(spec.description || '') + '</p>',
                '<div style="display:flex;gap:20px;flex-wrap:wrap;margin:10px 0">',
                '<div><span style="color:#888;font-size:11px">预估成本</span><br><span style="color:#4fc3f7;font-size:16px">¥' + (spec.estimated_cost || 0).toFixed(0) + '</span></div>',
                '<div><span style="color:#888;font-size:11px">开发时间</span><br><span style="color:#81c784;font-size:16px">' + (spec.development_time || 0).toFixed(1) + '月</span></div>',
                '<div><span style="color:#888;font-size:11px">复杂度</span><br><span style="color:#ffcc80;font-size:16px">' + ((spec.complexity || 0) * 100).toFixed(0) + '%</span></div>',
                '<div><span style="color:#888;font-size:11px">可行性</span><br><span style="color:#ce93d8;font-size:16px">' + ((spec.feasibility || 0) * 100).toFixed(0) + '%</span></div>',
                '</div>',
                featuresHtml,
                evalHtml,
                '</div>'
            ].join('');
            pdSetResult(resultHtml, true);
        })
        .catch(function(err) {
            pdSetResult('<div style="color:#ef5350;padding:15px">请求失败：' + pdEscapeHtml(err.message) + '（请确认后端服务已启动）</div>', true);
        });
    }

    /* 检查引擎状态 */
    function pdCheckStatus() {
        pdInit();
        pdSetStatus('<div style="text-align:center;padding:20px"><span style="color:#4fc3f7">查询引擎状态...</span></div>', true);

        window.SelfLnnApi.request('/product/design', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: '{}'
        })
        .then(function(resp) { return resp.json(); })
        .then(function(data) {
            var pd = data.product_design;
            if (!pd) {
                pdSetStatus('<div style="color:#ef5350;padding:15px">未收到引擎状态</div>', true);
                return;
            }
            var statusHtml = [
                '<table style="width:100%;color:#ccc;font-size:13px">',
                '<tr><td style="padding:4px 0">引擎状态</td><td style="text-align:right;color:' + (pd.status === 'ready' ? '#81c784' : '#ef5350') + '">' + pdEscapeHtml(pd.status || '未知') + '</td></tr>',
                '<tr><td style="padding:4px 0">引擎就绪</td><td style="text-align:right;color:' + (pd.engine_ready === true ? '#81c784' : '#ef5350') + '">' + (pd.engine_ready ? '是' : '否') + '</td></tr>',
                '<tr><td style="padding:4px 0">认知系统可用</td><td style="text-align:right">' + (pd.cognition_available === true ? '✅' : '❌') + '</td></tr>',
                '<tr><td style="padding:4px 0">元认知系统可用</td><td style="text-align:right">' + (pd.metacognition_available === true ? '✅' : '❌') + '</td></tr>',
                '<tr><td style="padding:4px 0">知识库可用</td><td style="text-align:right">' + (pd.knowledge_available === true ? '✅' : '❌') + '</td></tr>',
                '<tr><td style="padding:4px 0">知识库条目</td><td style="text-align:right;color:#64b5f6">' + (pd.knowledge_entries || 0) + '</td></tr>',
                '</table>'
            ].join('');
            pdSetStatus(statusHtml, true);
        })
        .catch(function(err) {
            pdSetStatus('<div style="color:#ef5350;padding:15px">请求失败：' + pdEscapeHtml(err.message) + '</div>', true);
        });
    }

    function pdEscapeHtml(text) {
        if (!text) return '';
        var div = document.createElement('div');
        div.appendChild(document.createTextNode(text));
        return div.innerHTML;
    }

    /* 初始化：绑定导航切换事件 */
    document.addEventListener('DOMContentLoaded', function() {
        pdInit();
        /* 监听导航切换，当产品设计面板可见时自动检查引擎状态 */
        var pdSection = document.getElementById('product-design');
        if (pdSection) {
            var observer = new MutationObserver(function(mutations) {
                mutations.forEach(function(m) {
                    if (pdSection.classList.contains('active')) {
                        pdCheckStatus();
                    }
                });
            });
            observer.observe(pdSection, { attributes: true, attributeFilter: ['class'] });
        }
    });

    /* 暴露全局接口 */
    window.ProductDesign = {
        generate: pdGenerate,
        checkStatus: pdCheckStatus
    };
})();
