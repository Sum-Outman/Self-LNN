/**
 * SELF-LNN AGI 控制器
 * 通过AGI对话控制计算机和设备
 * 提供系统命令执行、文件操作、进程管理、机器设备控制接口
 */

class AGIController {
    constructor() {
        this.initialized = false;
        this.controlEnabled = true;
        this.safetyLock = true;
        this.executionHistory = [];
        this.maxHistorySize = 200;
        this.pendingTasks = new Map();
        this.taskIdCounter = 0;
        this.activeRobots = new Map();
        this.computerControlEnabled = false;
        this.deviceControlEnabled = false;
    }

    init() {
        this.initialized = true;
        document.dispatchEvent(new CustomEvent('agi-controller-ready'));
    }

    async executeTask(task) {
        if (!this.initialized) {
            return { success: false, error: 'AGI控制器未初始化' };
        }
        if (!this.controlEnabled) {
            return { success: false, error: 'AGI控制已禁用' };
        }
        if (this.safetyLock && this._isHighRiskTask(task)) {
            return { success: false, error: '安全锁定: 高风险操作需手动确认', requiresConfirmation: true };
        }
        const taskId = 'task_' + (++this.taskIdCounter);
        const taskRecord = {
            id: taskId,
            type: task.type,
            action: task.action,
            params: task.params,
            status: 'pending',
            createdAt: Date.now(),
            completedAt: null,
            result: null,
            error: null
        };
        this.pendingTasks.set(taskId, taskRecord);
        try {
            const result = await this._routeTask(task);
            taskRecord.status = 'completed';
            taskRecord.completedAt = Date.now();
            taskRecord.result = result;
            this._addHistory(taskRecord);
            return { success: true, taskId: taskId, result: result };
        } catch (err) {
            taskRecord.status = 'failed';
            taskRecord.completedAt = Date.now();
            taskRecord.error = err.message;
            this._addHistory(taskRecord);
            return { success: false, taskId: taskId, error: err.message };
        } finally {
            this.pendingTasks.delete(taskId);
        }
    }

    async _routeTask(task) {
        switch (task.type) {
            case 'computer':
                return await this._executeComputerTask(task);
            case 'robot':
                return await this._executeRobotTask(task);
            case 'device':
                return await this._executeDeviceTask(task);
            case 'system':
                return await this._executeSystemTask(task);
            case 'file':
                return await this._executeFileTask(task);
            default:
                throw new Error('未知任务类型: ' + task.type);
        }
    }

    async _executeComputerTask(task) {
        if (!this.computerControlEnabled) {
            throw new Error('计算机控制未启用');
        }
        switch (task.action) {
            case 'launch_app':
                return await this._apiPost('/api/computer/launch', { name: task.params.name });
            case 'close_app':
                return await this._apiPost('/api/computer/close', { name: task.params.name });
            case 'type_text':
                return await this._apiPost('/api/computer/type', { text: task.params.text });
            case 'screenshot':
                return await this._apiPost('/api/computer/screenshot', {});
            case 'execute_command':
                return await this._apiPost('/api/computer/execute', { command: task.params.command });
            case 'volume_up':
                return await this._apiPost('/api/computer/volume', { direction: 'up', value: task.params.value || 10 });
            case 'volume_down':
                return await this._apiPost('/api/computer/volume', { direction: 'down', value: task.params.value || 10 });
            case 'mute':
                return await this._apiPost('/api/computer/volume', { action: 'mute_toggle' });
            default:
                throw new Error('未知计算机操作: ' + task.action);
        }
    }

    async _executeRobotTask(task) {
        switch (task.action) {
            case 'connect':
                return await this._apiPost('/api/robot/connect', task.params);
            case 'disconnect':
                return await this._apiPost('/api/robot/disconnect', task.params);
            case 'move':
                return await this._apiPost('/api/robot/command', {
                    action: 'move',
                    direction: task.params.direction,
                    distance: task.params.distance || 0.5,
                    speed: task.params.speed || 0.3
                });
            case 'rotate':
                return await this._apiPost('/api/robot/command', {
                    action: 'rotate',
                    angle: task.params.angle || 90,
                    speed: task.params.speed || 0.5
                });
            case 'stop':
                return await this._apiPost('/api/robot/command', { action: 'stop' });
            case 'emergency_stop':
                return await this._apiPost('/api/robot/emergency_stop', {});
            case 'set_speed':
                return await this._apiPost('/api/robot/parameters', { linear_speed: task.params.speed || 0.5 });
            case 'get_status':
                return await this._apiGet('/api/robot/status');
            case 'coordinate':
                return await this._apiPost('/api/robot/coordinate', {
                    robot_ids: task.params.robotIds || [],
                    action: task.params.action,
                    params: task.params.params || {}
                });
            // A7.2 ROS指令转发
            case 'ros_publish':
                return await this._apiPost('/api/ros/publish', {
                    topic: task.params.topic,
                    message_type: task.params.messageType || 'std_msgs/String',
                    data: task.params.data || {}
                });
            case 'ros_subscribe':
                return await this._apiPost('/api/ros/subscribe', {
                    topic: task.params.topic,
                    message_type: task.params.messageType || 'std_msgs/String',
                    callback: task.params.callback || ''
                });
            case 'ros_service_call':
                return await this._apiPost('/api/ros/service', {
                    service: task.params.service,
                    request: task.params.request || {}
                });
            case 'ros_list_topics':
                return await this._apiGet('/api/ros/topics');
            case 'ros_list_nodes':
                return await this._apiGet('/api/ros/nodes');
            // A7.2 串口设备控制
            case 'serial_list':
                return await this._apiGet('/api/serial/list');
            case 'serial_open':
                return await this._apiPost('/api/serial/open', {
                    port: task.params.port,
                    baudrate: task.params.baudrate || 115200,
                    data_bits: task.params.dataBits || 8,
                    stop_bits: task.params.stopBits || 1,
                    parity: task.params.parity || 'none'
                });
            case 'serial_close':
                return await this._apiPost('/api/serial/close', {
                    port: task.params.port
                });
            case 'serial_send':
                return await this._apiPost('/api/serial/send', {
                    port: task.params.port,
                    data: task.params.data,
                    encoding: task.params.encoding || 'utf-8'
                });
            case 'serial_receive':
                return await this._apiPost('/api/serial/send', {
                    port: task.params.port,
                    command: 'receive',
                    timeout_ms: task.params.timeoutMs || 1000
                });
            // A7.2 多机器人协调
            case 'multi_robot_sync':
                return await this._apiPost('/api/multi_robot/sync', {
                    robot_ids: task.params.robotIds || [],
                    action: task.params.action,
                    params: task.params.params || {},
                    sync_mode: task.params.syncMode || 'parallel'
                });
            case 'robot_group_command':
                return await this._apiPost('/api/multi_robot/sync', {
                    robot_ids: task.params.robotIds || [],
                    action: 'group_command',
                    params: {
                        command: task.params.command,
                        value: task.params.value
                    },
                    sync_mode: task.params.syncMode || 'parallel'
                });
            default:
                throw new Error('未知机器人操作: ' + task.action);
        }
    }

    async _executeDeviceTask(task) {
        if (!this.deviceControlEnabled) {
            throw new Error('设备控制未启用');
        }
        switch (task.action) {
            case 'turn_on':
                return await this._apiPost('/api/device/control', { device: task.params.device, action: 'on' });
            case 'turn_off':
                return await this._apiPost('/api/device/control', { device: task.params.device, action: 'off' });
            case 'set_temperature':
                return await this._apiPost('/api/device/control', {
                    device: 'air_conditioner',
                    action: 'set_temperature',
                    value: task.params.temperature || 26
                });
            default:
                throw new Error('未知设备操作: ' + task.action);
        }
    }

    async _executeSystemTask(task) {
        switch (task.action) {
            case 'get_status':
                if (window.SelfLnnApi && typeof window.SelfLnnApi.getSystemStatus === 'function') {
                    return await window.SelfLnnApi.getSystemStatus();
                }
                throw new Error('系统API不可用');
            case 'start_training':
                if (window.SelfLnnApi && typeof window.SelfLnnApi.startTraining === 'function') {
                    return await window.SelfLnnApi.startTraining(task.params || {});
                }
                throw new Error('训练API不可用');
            case 'stop_training':
                if (window.SelfLnnApi && typeof window.SelfLnnApi.stopTrainingJob === 'function') {
                    return await window.SelfLnnApi.stopTrainingJob();
                }
                throw new Error('训练API不可用');
            case 'enable_feature':
                if (window.SelfLnnApi && typeof window.SelfLnnApi.toggleAgiFeature === 'function') {
                    return await window.SelfLnnApi.toggleAgiFeature(task.params.feature, true);
                }
                throw new Error('功能API不可用');
            case 'disable_feature':
                if (window.SelfLnnApi && typeof window.SelfLnnApi.toggleAgiFeature === 'function') {
                    return await window.SelfLnnApi.toggleAgiFeature(task.params.feature, false);
                }
                throw new Error('功能API不可用');
            // A7.3 AGI内部系统功能调用
            case 'get_feature_list':
                return await this._apiGet('/api/agi/features');
            case 'start_evolution':
                return await this._apiPost('/api/agi/feature/toggle', { feature: 'self_evolution', enabled: true });
            case 'stop_evolution':
                return await this._apiPost('/api/agi/feature/toggle', { feature: 'self_evolution', enabled: false });
            case 'toggle_self_learning':
                return await this._apiPost('/api/agi/feature/toggle', { feature: 'self_learning', enabled: task.params.enabled });
            case 'toggle_self_decision':
                return await this._apiPost('/api/agi/feature/toggle', { feature: 'self_decision', enabled: task.params.enabled });
            case 'toggle_self_execution':
                return await this._apiPost('/api/agi/feature/toggle', { feature: 'self_execution', enabled: task.params.enabled });
            case 'toggle_imitation_learning':
                return await this._apiPost('/api/agi/feature/toggle', { feature: 'imitation_learning', enabled: task.params.enabled });
            case 'toggle_self_correction':
                return await this._apiPost('/api/agi/feature/toggle', { feature: 'self_correction', enabled: task.params.enabled });
            case 'trigger_self_correction':
                return await this._apiPost('/api/agi/self_correction', { trigger: task.params.trigger || 'manual', context: task.params.context || {} });
            default:
                throw new Error('未知系统操作: ' + task.action);
        }
    }

    async _executeFileTask(task) {
        switch (task.action) {
            case 'read':
                return await this._apiPost('/api/files/read', { path: task.params.path });
            case 'write':
                return await this._apiPost('/api/files/write', {
                    path: task.params.path,
                    content: task.params.content
                });
            case 'delete':
                return await this._apiPost('/api/files/delete', { path: task.params.path });
            case 'list':
                return await this._apiPost('/api/files/list', { path: task.params.path || '/' });
            default:
                throw new Error('未知文件操作: ' + task.action);
        }
    }

    /* ZSFABC-001修复: api-service.js.request()已自动添加baseURL(/api)前缀，
     * 故此处传入endpoint不应再含/api前缀，否则产生双重/api/api/xxx导致404 */
    async _apiPost(endpoint, data) {
        try {
            if (!window.SelfLnnApi) {
                throw new Error('API服务不可用');
            }
            var fixedEndpoint = endpoint;
            if (fixedEndpoint.indexOf('/api/') === 0) {
                fixedEndpoint = fixedEndpoint.substring(4);
            }
            const response = await window.SelfLnnApi.request(fixedEndpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data || {})
            });
            if (!response.ok) {
                throw new Error('请求失败: HTTP ' + response.status);
            }
            return await response.json();
        } catch (err) {
            console.warn('API请求失败:', endpoint, err.message);
            document.dispatchEvent(new CustomEvent('agi-control-error', {
                detail: { endpoint: endpoint, error: err.message }
            }));
            throw err;
        }
    }

    async _apiGet(endpoint) {
        try {
            if (!window.SelfLnnApi) {
                throw new Error('API服务不可用');
            }
            var fixedEndpoint = endpoint;
            if (fixedEndpoint.indexOf('/api/') === 0) {
                fixedEndpoint = fixedEndpoint.substring(4);
            }
            const response = await window.SelfLnnApi.request(fixedEndpoint, { method: 'GET' });
            if (!response.ok) {
                throw new Error('请求失败: HTTP ' + response.status);
            }
            return await response.json();
        } catch (err) {
            console.warn('API请求失败:', endpoint, err.message);
            throw err;
        }
    }

    _isHighRiskTask(task) {
        const highRisk = ['shutdown', 'restart', 'reboot', 'delete', 'emergency_stop'];
        return highRisk.includes(task.action);
    }

    registerRobot(robotId, info) {
        this.activeRobots.set(robotId, {
            id: robotId,
            info: info || {},
            registeredAt: Date.now(),
            lastHeartbeat: Date.now()
        });
    }

    unregisterRobot(robotId) {
        this.activeRobots.delete(robotId);
    }

    getActiveRobots() {
        return Array.from(this.activeRobots.values());
    }

    enableComputerControl() {
        this.computerControlEnabled = true;
    }

    disableComputerControl() {
        this.computerControlEnabled = false;
    }

    enableDeviceControl() {
        this.deviceControlEnabled = true;
    }

    disableDeviceControl() {
        this.deviceControlEnabled = false;
    }

    setSafetyLock(locked) {
        this.safetyLock = locked;
    }

    setControlEnabled(enabled) {
        this.controlEnabled = enabled;
    }

    _addHistory(record) {
        this.executionHistory.unshift(record);
        if (this.executionHistory.length > this.maxHistorySize) {
            this.executionHistory.pop();
        }
    }

    getHistory(count) {
        return this.executionHistory.slice(0, count || 50);
    }

    getPendingTasks() {
        return Array.from(this.pendingTasks.values());
    }

    getTaskStatus(taskId) {
        const task = this.pendingTasks.get(taskId);
        if (task) return task;
        return this.executionHistory.find(t => t.id === taskId) || null;
    }

    destroy() {
        this.pendingTasks.clear();
        this.activeRobots.clear();
        this.executionHistory = [];
        this.initialized = false;
    }
}

window.AGIController = AGIController;
