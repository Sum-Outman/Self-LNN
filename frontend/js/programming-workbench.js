(function() {
    'use strict';

    var currentTab = 'editor';
    var editorCode = '';

    function switchTab(tabName) {
        currentTab = tabName;
        document.querySelectorAll('.pw-tab-content').forEach(function(el) {
            el.style.display = 'none';
        });
        document.querySelectorAll('.pw-tab-btn').forEach(function(el) {
            el.classList.remove('active');
        });
        var tabContent = document.getElementById('pw-tab-' + tabName);
        var tabBtn = document.querySelector('.pw-tab-btn[data-tab="' + tabName + '"]');
        if (tabContent) tabContent.style.display = 'block';
        if (tabBtn) tabBtn.classList.add('active');
    }

    function getEditorCode() {
        var el = document.getElementById('pw-code-editor');
        return el ? el.value : '';
    }

    function setEditorCode(code) {
        var el = document.getElementById('pw-code-editor');
        if (el) el.value = code || '';
        editorCode = el ? el.value : '';
    }

    var _pwMaxOutputLines = 2000; /* 输出行数上限，防止内存泄漏 */
    function appendOutput(text) {
        var el = document.getElementById('pw-output');
        if (el) {
            el.value += text + '\n';
            /* 超过上限时截断前一半 */
            var lines = el.value.split('\n');
            if (lines.length > _pwMaxOutputLines) {
                el.value = lines.slice(lines.length - _pwMaxOutputLines / 2).join('\n');
            }
            el.scrollTop = el.scrollHeight;
        }
    }

    function clearOutput() {
        var el = document.getElementById('pw-output');
        if (el) el.value = '';
    }

    function showStatus(msg, isError) {
        var el = document.getElementById('pw-status');
        if (el) {
            el.textContent = msg;
            el.style.color = isError ? '#e74c3c' : '#2ecc71';
            el.style.display = 'block';
            setTimeout(function() { el.style.display = 'none'; }, 5000);
        }
    }

    function updateAnalysisResult(data) {
        var container = document.getElementById('pw-analysis-result');
        if (!container) return;
        if (!data || !data.success || !data.analysis) {
            container.innerHTML = '<div class="pw-error">分析失败</div>';
            return;
        }
        var a = data.analysis;
        var suggestions = '';
        if (data.optimization_suggestions && data.optimization_suggestions.length > 0) {
            suggestions = '<div class="pw-section-title">优化建议</div><ul class="pw-suggestion-list">';
            for (var i = 0; i < data.optimization_suggestions.length; i++) {
                suggestions += '<li>' + data.optimization_suggestions[i] + '</li>';
            }
            suggestions += '</ul>';
        }
        container.innerHTML =
            '<div class="pw-section-title">代码分析结果</div>' +
            '<div class="pw-metrics-grid">' +
                '<div class="pw-metric-card"><div class="pw-metric-value">' + a.cyclomatic_complexity + '</div><div class="pw-metric-label">圈复杂度</div></div>' +
                '<div class="pw-metric-card"><div class="pw-metric-value">' + a.line_count + '</div><div class="pw-metric-label">代码行数</div></div>' +
                '<div class="pw-metric-card"><div class="pw-metric-value">' + a.function_count + '</div><div class="pw-metric-label">函数数量</div></div>' +
                '<div class="pw-metric-card"><div class="pw-metric-value">' + a.variable_count + '</div><div class="pw-metric-label">变量数量</div></div>' +
                '<div class="pw-metric-card"><div class="pw-metric-value">' + a.max_nesting_depth + '</div><div class="pw-metric-label">最大嵌套</div></div>' +
                '<div class="pw-metric-card"><div class="pw-metric-value">' + (a.maintainability_index != null ? a.maintainability_index.toFixed(2) : 'N/A') + '</div><div class="pw-metric-label">可维护性</div></div>' +
                '<div class="pw-metric-card pw-metric-error"><div class="pw-metric-value">' + a.error_count + '</div><div class="pw-metric-label">错误数</div></div>' +
                '<div class="pw-metric-card pw-metric-warning"><div class="pw-metric-value">' + a.warning_count + '</div><div class="pw-metric-label">警告数</div></div>' +
            '</div>' +
            suggestions;
    }

    window.runCodeAnalysis = async function() {
        var code = getEditorCode();
        if (!code.trim()) { showStatus('请先输入代码', true); return; }
        clearOutput();
        appendOutput('正在分析代码...');
        try {
            var result = await window.SelfLnnApi.programmingAnalyze(code);
            if (result.success && result.data) {
                var data = result.data;
                updateAnalysisResult(data);
                appendOutput('分析完成');
                showStatus('分析完成');
            } else {
                appendOutput('分析失败: ' + (result.error || '未知错误'));
                showStatus('分析失败', true);
            }
        } catch (e) {
            appendOutput('请求异常: ' + e.message);
            showStatus('请求异常', true);
        }
    };

    window.runCodeGeneration = async function() {
        var funcName = document.getElementById('pw-gen-funcname') ? document.getElementById('pw-gen-funcname').value : 'my_function';
        var desc = document.getElementById('pw-gen-desc') ? document.getElementById('pw-gen-desc').value : '';
        var paramCount = parseInt(document.getElementById('pw-gen-paramcount') ? document.getElementById('pw-gen-paramcount').value : '0', 10);
        if (isNaN(paramCount)) paramCount = 0;
        if (!funcName.trim()) { showStatus('请输入函数名称', true); return; }
        clearOutput();
        appendOutput('正在生成代码...');
        try {
            var result = await window.SelfLnnApi.programmingGenerate(funcName, desc, paramCount);
            if (result.success && result.data) {
                var data = result.data;
                if (data.success && data.code) {
                    setEditorCode(data.code);
                    appendOutput('代码已生成并填入编辑器');
                    showStatus('代码生成成功');
                } else {
                    appendOutput('生成失败: ' + (data.error || '未知错误'));
                    showStatus('生成失败', true);
                }
            } else {
                appendOutput('请求失败: ' + (result.error || '未知错误'));
                showStatus('请求失败', true);
            }
        } catch (e) {
            appendOutput('请求异常: ' + e.message);
            showStatus('请求异常', true);
        }
    };

    window.runCodeOptimization = async function() {
        var code = getEditorCode();
        if (!code.trim()) { showStatus('请先输入代码', true); return; }
        var iterations = parseInt(document.getElementById('pw-opt-iterations') ? document.getElementById('pw-opt-iterations').value : '1', 10);
        if (isNaN(iterations)) iterations = 1;
        clearOutput();
        appendOutput('正在优化代码 (迭代' + iterations + '次)...');
        try {
            var result = await window.SelfLnnApi.programmingOptimize(code, iterations);
            if (result.success && result.data) {
                var data = result.data;
                if (data.success && data.code) {
                    setEditorCode(data.code);
                    appendOutput('优化完成');
                    showStatus('代码优化成功');
                } else {
                    appendOutput('优化失败: ' + (data.error || '未知错误'));
                    showStatus('优化失败', true);
                }
            } else {
                appendOutput('请求失败: ' + (result.error || '未知错误'));
                showStatus('请求失败', true);
            }
        } catch (e) {
            appendOutput('请求异常: ' + e.message);
            showStatus('请求异常', true);
        }
    };

    window.runCodeCompilation = async function() {
        var code = getEditorCode();
        if (!code.trim()) { showStatus('请先输入代码', true); return; }
        clearOutput();
        appendOutput('正在编译验证...');
        try {
            var result = await window.SelfLnnApi.programmingCompile(code);
            if (result.success && result.data) {
                var data = result.data;
                if (data.compilation) {
                    var c = data.compilation;
                    appendOutput('编译状态: ' + (c.success ? '成功' : '失败'));
                    appendOutput('错误数: ' + c.error_count + ', 警告数: ' + c.warning_count);
                    if (c.error_message) appendOutput('编译信息: ' + c.error_message);
                    showStatus(c.success ? '编译通过' : '编译失败', !c.success);
                } else {
                    appendOutput('编译结果异常');
                }
            } else {
                appendOutput('请求失败: ' + (result.error || '未知错误'));
                showStatus('请求失败', true);
            }
        } catch (e) {
            appendOutput('请求异常: ' + e.message);
            showStatus('请求异常', true);
        }
    };

    window.runCodeExecution = async function() {
        var code = getEditorCode();
        if (!code.trim()) { showStatus('请先输入代码', true); return; }
        var input = document.getElementById('pw-exec-input') ? document.getElementById('pw-exec-input').value : '';
        clearOutput();
        appendOutput('正在沙箱中执行代码...');
        try {
            var result = await window.SelfLnnApi.programmingExecute(code, input);
            if (result.success && result.data) {
                var data = result.data;
                if (data.success) {
                    appendOutput('执行输出:');
                    appendOutput(data.output || '(无输出)');
                    showStatus('执行成功');
                } else {
                    appendOutput('执行异常: ' + (data.error || '未知错误'));
                    if (data.output) appendOutput('部分输出: ' + data.output);
                    showStatus('执行异常', true);
                }
            } else {
                appendOutput('请求失败: ' + (result.error || '未知错误'));
                showStatus('请求失败', true);
            }
        } catch (e) {
            appendOutput('请求异常: ' + e.message);
            showStatus('请求异常', true);
        }
    };

    window.checkProgrammingStatus = async function() {
        clearOutput();
        appendOutput('正在检查编程引擎状态...');
        try {
            var result = await window.SelfLnnApi.programmingStatus();
            if (result.success && result.data) {
                var data = result.data;
                if (data.status) {
                    appendOutput('引擎状态: ' + (data.status.engine_ready ? '就绪' : '不可用'));
                    appendOutput('语言: ' + (data.status.language || 'N/A'));
                    showStatus('引擎就绪');
                } else {
                    appendOutput('状态信息获取失败');
                    showStatus('状态异常', true);
                }
            } else {
                appendOutput('请求失败: ' + (result.error || '未知错误'));
                showStatus('请求失败', true);
            }
        } catch (e) {
            appendOutput('请求异常: ' + e.message);
            showStatus('请求异常', true);
        }
    };

    window.loadSampleCode = async function() {
        /* L-004修复：移除硬编码回退代码，API失败时显示提示而非假数据 */
        try {
            var result = await window.SelfLnnApi.programmingSample();
            if (result.success && result.code) {
                setEditorCode(result.code);
                showStatus('示例代码已加载（来自后端）');
                return;
            }
        } catch(e) { console.warn('从API获取示例代码失败:', e.message); }
        showStatus('⚠ 后端未连接——无法加载示例代码（遵循禁止虚假数据原则）', true);
    };

    window.clearEditor = function() {
        setEditorCode('');
        clearOutput();
        showStatus('已清空');
    };

/* 处理DOMContentLoaded竞态条件 */
    function _pwInit() {
        switchTab('editor');
        checkProgrammingStatus();
    }
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', _pwInit);
    } else {
        _pwInit();
    }

/* 重命名全局暴露避免与skills/knowledge的switchTab冲突 */
    window.switchProgrammingTab = switchTab;
    window.switchTab = undefined;
    window.clearOutput = clearOutput;

})();
