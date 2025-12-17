import sys
from pathlib import Path
import os
import threading
from queue import Queue
import time
import random

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT / "src"))

from DFTAgent import DFTAgent

app = FastAPI()

# ========== 初始化 ==========
agent = DFTAgent(
    model="gpt-4o",
    dft_tool="quantum espresso",
    verbose=True,
    backend="openai",
    work_dir="",
    max_new_tokens=4096,
    temperature=0.0,
    top_p=0.9,
    need_query_info=True,
    parallel_exec=True,
    parallel_np=12
)

print("✅ DFT Agent Loaded")


class ChatRequest(BaseModel):
    messages: list   # 跟 OpenAI 保持一致格式


def extract_user_message(messages):
    """取最后一条用户消息"""
    for msg in reversed(messages):
        if msg["role"] == "user":
            return msg["content"]
    return messages[-1]["content"]


def stream_generator(query: str):
    """
    把所有 stdout / stderr 全部劫持
    不再使用 agent.stream
    只使用 agent.run
    """

    q = Queue()

    # 保存原有输出方式
    old_stdout = sys.stdout
    old_stderr = sys.stderr

    # 自定义捕获器
    class StreamCatcher:
        def write(self, text):
            if text:
                q.put(text)
        def flush(self):
            pass

    sys.stdout = StreamCatcher()
    sys.stderr = StreamCatcher()

    def run_agent():
        try:
            agent.run(query)   # ✅ 直接运行，不用 stream
        except Exception as e:
            q.put(f"\n[ERROR] {str(e)}\n")
        finally:
            q.put(None)        # 🔚 结束标识

    t = threading.Thread(target=run_agent)
    t.start()

    try:
        while True:
            msg = q.get()

            if msg is None:
                break

            # yield f"{msg}\n\n"
            # 逐字流式输出（ChatGPT风格）
            for ch in str(msg):
                yield ch
                time.sleep(random.uniform(0.001, 0.01))  # 人类抖动
            yield "\n\n"

    finally:
        # 恢复 stdout
        sys.stdout = old_stdout
        sys.stderr = old_stderr


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])

    query = extract_user_message(messages)

    print(f"\n📩 Incoming query:\n{query}\n")

    return StreamingResponse(
        stream_generator(query),
        media_type="text/plain"
    )
