#!/usr/bin/env python3
"""
SELF-LNN PyBullet IPC桥接脚本 (F-023修复: 从内嵌单行代码提取为独立文件)
通过stdin/stdout JSON协议与C主程序通信。

协议格式（每行一个JSON对象）:
请求: {"action":"step|disconnect|load_urdf|get_joint|set_joints|get_base|get_contacts|camera","param":value,...}
响应: {"status":"ok|error",...}
"""

import sys
import json
import struct
import math

try:
    import pybullet as p
    import numpy as np
except ImportError as e:
    print(json.dumps({"status":"error","msg":f"PyBullet依赖缺失: {e}"}))
    sys.stdout.flush()

def main():
    gui_mode = "--gui" in sys.argv
    time_step = 1.0 / 240.0
    max_steps = 10
    
    # 解析参数
    for arg in sys.argv[1:]:
        if arg.startswith("--ts="):
            time_step = float(arg[5:])
        elif arg.startswith("--steps="):
            max_steps = int(arg[8:])
    
    # 连接PyBullet
    try:
        if gui_mode:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setTimeStep(time_step)
    except Exception as e:
        print(json.dumps({"status":"error","msg":f"连接失败: {e}"}))
        sys.stdout.flush()
        return 1
    
    print(json.dumps({"status":"connected","ts":time_step,"steps":max_steps}))
    sys.stdout.flush()
    
    robot_id = None
    
    # 主IPC循环
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        
        try:
            cmd = json.loads(line.strip())
        except json.JSONDecodeError:
            continue
        
        resp = {"status":"ok"}
        action = cmd.get("action","")
        
        try:
            if action == "step":
                for _ in range(max_steps):
                    p.stepSimulation()
                resp["steps"] = max_steps
                
            elif action == "disconnect":
                p.disconnect()
                resp["status"] = "disconnected"
                print(json.dumps(resp))
                sys.stdout.flush()
                break
                
            elif action == "load_urdf":
                path = cmd.get("path","")
                pos = cmd.get("pos", [0,0,0])
                orn = cmd.get("orn", [0,0,0,1])
                fixed = cmd.get("fixed", False)
                robot_id = p.loadURDF(path, pos, orn, useFixedBase=fixed)
                resp["robot_id"] = robot_id
                
            elif action == "get_joint":
                r = cmd.get("robot", robot_id)
                j = cmd.get("joint", 0)
                s = p.getJointState(r, j)
                pos, vel, forces, torque = s[0], s[1], s[2], s[3]
                print(json.dumps({"pos":pos,"vel":vel,"torque":torque,"forces":list(forces) if hasattr(forces,'__iter__') else [forces]}))
                sys.stdout.flush()
                continue
                
            elif action == "set_joints":
                r = cmd.get("robot", robot_id)
                for jt in cmd.get("joints", []):
                    p.setJointMotorControl2(r, jt["idx"], p.POSITION_CONTROL,
                                           targetPosition=jt["pos"])
                resp["count"] = len(cmd.get("joints", []))
                
            elif action == "get_base":
                r = cmd.get("robot", robot_id)
                pos, orn = p.getBasePositionAndOrientation(r)
                lv, av = p.getBaseVelocity(r)
                print(json.dumps({
                    "pos":list(pos), "orn":list(orn),
                    "lv":list(lv), "av":list(av)
                }))
                sys.stdout.flush()
                continue
                
            elif action == "get_contacts":
                r = cmd.get("robot", robot_id)
                max_c = cmd.get("max", 10)
                pts = p.getContactPoints(bodyA=r)
                items = []
                for c in pts[:max_c]:
                    items.append({
                        "a":c[1],"b":c[2],
                        "p":[c[5][0],c[5][1],c[5][2]],
                        "n":[c[7][0],c[7][1],c[7][2]],
                        "f":c[9],"d":c[8]
                    })
                print(json.dumps(items))
                sys.stdout.flush()
                continue
                
            elif action == "camera":
                w = cmd.get("w", 640)
                h = cmd.get("h", 480)
                eye = cmd.get("eye", [0,0,1])
                target = cmd.get("target", [0,0,0])
                up = cmd.get("up", [0,1,0])
                vm = p.computeViewMatrix(eye, target, up)
                pm = p.computeProjectionMatrixFOV(60, float(w)/float(h), 0.01, 100)
                img = p.getCameraImage(w, h, vm, pm)
                rgb = img[2]
                depth = img[3]
                
                if isinstance(rgb, list):
                    rgb_bytes = np.array(rgb, dtype=np.uint8).tobytes()
                else:
                    rgb_bytes = bytes(rgb)
                
                depth_flat = [float(d) for d in depth] if depth else []
                depth_bytes = struct.pack("f"*len(depth_flat), *depth_flat) if depth_flat else b""
                
                resp["w"] = w
                resp["h"] = h
                resp["rgb_size"] = len(rgb_bytes)
                resp["depth_size"] = len(depth_bytes) // 4
                print(json.dumps(resp))
                sys.stdout.flush()
                sys.stdout.buffer.write(rgb_bytes)
                if depth_bytes:
                    sys.stdout.buffer.write(depth_bytes)
                sys.stdout.buffer.flush()
                continue
                
            elif action == "ping":
                resp["pong"] = True
                
            else:
                resp = {"status":"error","msg":f"未知动作: {action}"}
                
        except Exception as e:
            resp = {"status":"error","msg":str(e)}
        
        print(json.dumps(resp))
        sys.stdout.flush()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
