/**
 * SELF-LNN AGI 文字指令控制系统
 * 在对话中输入文字指令，直接控制机器人和设备
 * 支持自然语言解析和结构化命令
 */

'use strict';

class TextCommandSystem {
    constructor() {
        this.commandEngine = null;
        this.enabled = true;
        this.onCommandResult = null;
        this.commandPrefix = '';
        /* M-034修复：默认前缀列表可从后端动态更新 */
        this._dynamicPrefixes = null;
        this._staticPrefixes = ['控制', '机器人', '电脑', '计算机', '打开', '关闭', '开始', '停止', '系统'];
    }

    setCommandEngine(engine) {
        this.commandEngine = engine;
    }

    /* M-034修复：从后端动态获取命令前缀列表 */
    async fetchCommandPrefixes() {
        try {
            if (window.SelfLnnApi && window.SelfLnnApi.getCommandPrefixes) {
                var result = await window.SelfLnnApi.getCommandPrefixes();
                if (result && result.prefixes && result.prefixes.length > 0) {
                    this._dynamicPrefixes = result.prefixes;
                }
            }
        } catch (e) {
            this._dynamicPrefixes = null;
        }
    }

    setCommandPrefixes(prefixes) {
        if (Array.isArray(prefixes)) {
            this._dynamicPrefixes = prefixes;
        }
    }

    processText(text) {
        if (!this.enabled || !this.commandEngine) {
            return { success: false, error: '文字指令系统未就绪', isCommand: false };
        }
        if (!text || text.trim().length === 0) {
            return { success: false, error: '输入为空', isCommand: false };
        }
        const trimmed = text.trim();
        const parsed = this.commandEngine.parseCommand(trimmed);
        if (parsed.command) {
            parsed.isCommand = true;
/* 添加await确保错误能正确传递到上层 */
            this.commandEngine.executeCommand(parsed).then(result => {
                if (this.onCommandResult) {
                    this.onCommandResult(parsed, result);
                }
            }).catch(err => {
                console.error('命令执行异常:', err);
                if (this.onCommandResult) {
                    this.onCommandResult(parsed, { success: false, error: err && err.message ? err.message : '命令执行异常' });
                }
            });
            return parsed;
        }
        return { success: true, command: null, isCommand: false, rawText: trimmed };
    }

    isCommandText(text) {
        if (!text) return false;
        const trimmed = text.trim().toLowerCase();
        const prefixes = this._dynamicPrefixes || this._staticPrefixes;
        return prefixes.some(p => trimmed.startsWith(p));
    }

    enable() {
        this.enabled = true;
    }

    disable() {
        this.enabled = false;
    }

    setPrefix(prefix) {
        this.commandPrefix = prefix || '';
    }
}

window.TextCommandSystem = TextCommandSystem;
