"""
MiniMind 训练管理后端
运行在 minimind-npu Docker 容器内，提供 REST + SSE 接口控制训练任务。

启动方式（宿主机执行）：
  docker run -d --rm --name minimind-manager --network=host \
    --device /dev/davinci0 ... --device /dev/davinci7 \
    --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro \
    -v /data/code/minimind:/workspace/minimind \
    minimind-npu python /workspace/minimind/scripts/serve_train_manager.py
"""

import os
import sys
import json
import time
import uuid
import signal
import subprocess
import threading
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone, timedelta

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="MiniMind 训练管理")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 项目根目录（容器内 /workspace/minimind）
BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "out"
DATASET_DIR = BASE_DIR / "dataset"
TRAINER_DIR = BASE_DIR / "trainer"

# 训练状态
class TrainState:
    def __init__(self):
        self.status = "idle"  # idle / running / finished / error
        self.process: Optional[subprocess.Popen] = None
        self.log_buffer: list[str] = []
        self.lock = threading.Lock()
        self.subscribers: list[threading.Event] = []
        self.stage = ""
        self.error_msg = ""
        self.start_time: Optional[float] = None

    def reset(self):
        self.status = "idle"
        self.process = None
        self.log_buffer = []
        self.stage = ""
        self.error_msg = ""
        self.start_time = None

state = TrainState()


class TrainRequest(BaseModel):
    stage: str  # pretrain / full_sft / lora / dpo / ppo / grpo / spo / distillation / reason
    hidden_size: int = 512
    num_hidden_layers: int = 8
    use_moe: bool = False
    epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 1e-4
    data_path: str = ""
    from_weight: str = ""
    save_weight: str = ""
    extra_args: str = ""


def append_log(line: str):
    """添加日志行并通知所有 SSE 订阅者"""
    with state.lock:
        state.log_buffer.append(line)
        for evt in state.subscribers:
            evt.set()


def run_training(cmd: list[str]):
    """在子线程中运行训练进程"""
    try:
        append_log(f"[Manager] 启动命令: {' '.join(cmd)}")
        append_log(f"[Manager] 工作目录: {BASE_DIR}")

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(BASE_DIR),
            bufsize=1,
            universal_newlines=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        state.process = proc

        for line in proc.stdout:
            append_log(line.rstrip("\n"))

        proc.wait()

        if proc.returncode == 0:
            state.status = "finished"
            append_log(f"[Manager] 训练完成，退出码: 0")
        elif proc.returncode == -signal.SIGTERM or proc.returncode == -signal.SIGKILL:
            state.status = "finished"
            append_log(f"[Manager] 训练已被手动终止")
        else:
            state.status = "error"
            state.error_msg = f"进程退出码: {proc.returncode}"
            append_log(f"[Manager] 训练异常退出，退出码: {proc.returncode}")

    except Exception as e:
        state.status = "error"
        state.error_msg = str(e)
        append_log(f"[Manager] 异常: {e}")


@app.get("/api/status")
async def get_status():
    elapsed = None
    if state.start_time and state.status == "running":
        elapsed = int(time.time() - state.start_time)
    return {
        "status": state.status,
        "stage": state.stage,
        "error": state.error_msg,
        "log_lines": len(state.log_buffer),
        "elapsed_seconds": elapsed,
    }


@app.get("/api/weights")
async def list_weights():
    files = []
    if OUT_DIR.exists():
        for f in sorted(OUT_DIR.glob("*.pth")):
            files.append({
                "name": f.name,
                "size_mb": round(f.stat().st_size / 1024 / 1024, 1),
            })
    return {"weights": files}


@app.get("/api/datasets")
async def list_datasets():
    files = []
    if DATASET_DIR.exists():
        for f in sorted(DATASET_DIR.glob("*.jsonl")):
            files.append({
                "name": f.name,
                "size_mb": round(f.stat().st_size / 1024 / 1024, 1),
            })
    return {"datasets": files}


@app.post("/api/train/start")
async def start_training(req: TrainRequest):
    if state.status == "running":
        return {"ok": False, "error": "已有训练任务正在运行"}

    # 验证 stage
    valid_stages = ["pretrain", "full_sft", "lora", "dpo", "ppo", "grpo", "spo", "distillation", "reason"]
    if req.stage not in valid_stages:
        return {"ok": False, "error": f"无效的训练阶段: {req.stage}，可选: {valid_stages}"}

    script = TRAINER_DIR / f"train_{req.stage}.py"
    if not script.exists():
        return {"ok": False, "error": f"训练脚本不存在: {script}"}

    # 构建命令
    # 检测可用 NPU 数量
    npu_count = 8
    try:
        result = subprocess.run(["npu-smi", "info", "-l"], capture_output=True, text=True, timeout=5)
        for line in result.stdout.splitlines():
            if "Total Count" in line:
                npu_count = int(line.split(":")[-1].strip())
                break
    except Exception:
        pass

    cmd = [
        "torchrun",
        f"--nproc_per_node={npu_count}",
        str(script),
        "--epochs", str(req.epochs),
        "--batch_size", str(req.batch_size),
        "--learning_rate", str(req.learning_rate),
        "--hidden_size", str(req.hidden_size),
        "--num_hidden_layers", str(req.num_hidden_layers),
    ]

    if req.use_moe:
        cmd.extend(["--use_moe", "1"])
    if req.data_path:
        cmd.extend(["--data_path", req.data_path])
    if req.from_weight:
        cmd.extend(["--from_weight", req.from_weight])
    if req.save_weight:
        cmd.extend(["--save_weight", req.save_weight])
    if req.extra_args:
        cmd.extend(req.extra_args.split())

    # 重置状态
    state.reset()
    state.status = "running"
    state.stage = req.stage
    state.start_time = time.time()

    # 启动训练线程
    t = threading.Thread(target=run_training, args=(cmd,), daemon=True)
    t.start()

    return {"ok": True, "command": " ".join(cmd)}


@app.post("/api/train/stop")
async def stop_training():
    if state.status != "running" or state.process is None:
        return {"ok": False, "error": "没有正在运行的训练任务"}

    try:
        # 先发 SIGTERM，给进程优雅退出的机会
        os.killpg(os.getpgid(state.process.pid), signal.SIGTERM)
        append_log("[Manager] 已发送终止信号 (SIGTERM)")
    except ProcessLookupError:
        pass
    except Exception:
        try:
            state.process.kill()
            append_log("[Manager] 已强制终止进程 (SIGKILL)")
        except Exception:
            pass

    return {"ok": True}


@app.get("/api/train/logs")
async def stream_logs():
    """SSE 日志流"""
    def generate():
        idx = 0
        evt = threading.Event()
        with state.lock:
            state.subscribers.append(evt)
        try:
            while True:
                # 发送所有新日志
                with state.lock:
                    new_lines = state.log_buffer[idx:]
                    idx = len(state.log_buffer)
                    current_status = state.status

                for line in new_lines:
                    yield f"data: {json.dumps({'type': 'log', 'data': line}, ensure_ascii=False)}\n\n"

                # 训练结束后发送状态并退出
                if current_status in ("finished", "error", "idle") and not new_lines:
                    yield f"data: {json.dumps({'type': 'status', 'data': current_status}, ensure_ascii=False)}\n\n"
                    if current_status != "idle":
                        break

                # 等待新日志
                evt.wait(timeout=2.0)
                evt.clear()
        finally:
            with state.lock:
                if evt in state.subscribers:
                    state.subscribers.remove(evt)

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/npu")
async def npu_status():
    """执行 npu-smi 返回设备状态"""
    try:
        result = subprocess.run(
            ["npu-smi", "info"],
            capture_output=True, text=True, timeout=10,
        )
        # 解析 npu-smi 输出
        lines = result.stdout.strip()
        devices = []
        for line in lines.splitlines():
            # 简单提取设备信息行
            parts = line.split()
            if len(parts) >= 8 and parts[0].isdigit():
                try:
                    devices.append({
                        "id": int(parts[0]),
                        "name": parts[1] if not parts[1].isdigit() else f"NPU {parts[0]}",
                        "health": parts[2] if len(parts) > 2 else "OK",
                        "power": parts[3] if len(parts) > 3 else "N/A",
                        "temp": parts[4] if len(parts) > 4 else "N/A",
                    })
                except (ValueError, IndexError):
                    pass
        return {"raw": lines, "devices": devices}
    except FileNotFoundError:
        return {"raw": "npu-smi 不可用", "devices": []}
    except Exception as e:
        return {"raw": str(e), "devices": []}


# ==================== MCP 代理接口 ====================

# MCP 工具定义
MCP_TOOLS = [
    {
        "name": "get_weather",
        "description": "获取指定城市的天气信息",
        "inputSchema": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "城市名称"}},
            "required": ["city"],
        },
    },
    {
        "name": "calculate",
        "description": "计算数学表达式的结果",
        "inputSchema": {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "数学表达式"}},
            "required": ["expression"],
        },
    },
    {
        "name": "search",
        "description": "搜索互联网获取相关信息",
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "搜索关键词"}},
            "required": ["query"],
        },
    },
    {
        "name": "get_time",
        "description": "获取当前时间或指定时区的时间",
        "inputSchema": {
            "type": "object",
            "properties": {"timezone": {"type": "string", "description": "时区，如 Asia/Shanghai", "default": "Asia/Shanghai"}},
            "required": [],
        },
    },
]


def mcp_execute_tool(name: str, arguments: dict) -> tuple[str, bool]:
    """执行 MCP 工具，返回 (结果文本, 是否出错)"""
    try:
        if name == "get_weather":
            city = arguments.get("city", "未知")
            # 模拟天气数据（结构化 JSON）
            import random
            weather_data = {
                "city": city,
                "weather": random.choice(["晴", "多云", "阴", "小雨"]),
                "temperature": random.randint(15, 35),
                "humidity": random.randint(30, 80),
                "wind": random.choice(["东风", "南风", "西风", "北风"]) + str(random.randint(1, 5)) + "级",
                "aqi": random.randint(20, 150),
            }
            return json.dumps(weather_data, ensure_ascii=False), False
        elif name == "calculate":
            expr = str(arguments.get("expression", "0"))
            import re
            if not re.match(r'^[\d\s+\-*/().%]+$', expr):
                return "不支持的表达式（仅允许数字和基本运算符）", True
            result = eval(expr)  # noqa: S307 — 受限表达式
            return str(result), False
        elif name == "search":
            query = arguments.get("query", "")
            results = [
                {"title": f"关于「{query}」的最新资讯", "snippet": f"这是一条与「{query}」相关的搜索结果摘要。"},
                {"title": f"「{query}」- 百科介绍", "snippet": f"「{query}」的详细百科介绍内容。"},
                {"title": f"「{query}」相关讨论", "snippet": f"社区中关于「{query}」的热门讨论。"},
            ]
            return json.dumps(results, ensure_ascii=False), False
        elif name == "get_time":
            tz = arguments.get("timezone", "Asia/Shanghai")
            # 常用时区偏移映射（避免依赖 pytz）
            tz_offsets = {
                "Asia/Shanghai": 8, "Asia/Tokyo": 9, "Asia/Seoul": 9,
                "Asia/Singapore": 8, "Asia/Hong_Kong": 8,
                "US/Eastern": -5, "US/Pacific": -8, "Europe/London": 0,
                "Europe/Berlin": 1, "Europe/Paris": 1, "UTC": 0,
            }
            offset_h = tz_offsets.get(tz, 8)  # 默认东八区
            now = datetime.now(timezone(timedelta(hours=offset_h)))
            return f"当前时间（{tz}）：{now.strftime('%Y-%m-%d %H:%M:%S')}", False
        else:
            return f"未知工具: {name}", True
    except Exception as e:
        return f"工具执行异常: {e}", True


class MCPToolCallRequest(BaseModel):
    name: str
    arguments: dict = {}


@app.get("/api/mcp/info")
async def mcp_info():
    """返回 MCP Server 基本信息"""
    return {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "result": {
            "name": "minimind-mcp-server",
            "version": "1.0.0",
            "description": "MiniMind MCP 工具代理服务（教学演示）",
            "capabilities": {"tools": True},
        },
    }


@app.get("/api/mcp/tools/list")
async def mcp_tools_list():
    """返回 MCP 格式的工具列表"""
    return {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "result": {"tools": MCP_TOOLS},
    }


@app.post("/api/mcp/tools/call")
async def mcp_tools_call(req: MCPToolCallRequest):
    """执行工具调用，返回 MCP 格式结果"""
    call_id = f"call_{uuid.uuid4().hex[:12]}"
    text, is_error = mcp_execute_tool(req.name, req.arguments)
    return {
        "jsonrpc": "2.0",
        "id": call_id,
        "result": {
            "content": [{"type": "text", "text": text}],
            "isError": is_error,
        },
    }


if __name__ == "__main__":
    print(f"[TrainManager] 项目根目录: {BASE_DIR}")
    print(f"[TrainManager] 权重目录: {OUT_DIR}")
    print(f"[TrainManager] 数据集目录: {DATASET_DIR}")
    uvicorn.run(app, host="0.0.0.0", port=8999)
