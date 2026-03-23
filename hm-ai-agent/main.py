import uvicorn
import json
import asyncio
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from graph.workflow import create_agent_app
from langchain_core.messages import HumanMessage
from core.logger import get_logger
import os

# 在启动前手动清空昨天的旧日志
if os.path.exists("logs/backend.log"):
    open("logs/backend.log", 'w').close()

app = FastAPI(title="黑马点评 AI Agent 后端")
agent_app = create_agent_app()
backend_log = get_logger("API_SERVER")


# 定义请求模型
class ChatRequest(BaseModel):
    message: str


# 2. 【核心】Agent 流式转换器 (已适配 Token 注入)
async def agent_stream_generator(user_message: str, token: str):
    """
    将 LangGraph 的 astream 转换为前端 Vue 可识别的 SSE 格式
    :param user_message: 用户输入的文本
    :param token: 从 FastAPI Header 中提取的 authorization
    """
    # 1. 在 config 中注入 authorization
    # 这里的 key "authorization" 必须与你工具函数中 config.get() 的 key 保持一致
    config = {
        "configurable": {
            "thread_id": "common_user_1",
            "authorization": token
        }
    }

    # 记录最后一次发送的文本长度，实现“增量发送”
    last_sent_len = 0

    # astream 执行时，工具节点 (ToolNode) 会自动读取上面的 config
    async for event in agent_app.astream(
            {"messages": [HumanMessage(content=user_message)]},
            config=config,
            stream_mode="values"
    ):
        if "messages" in event:
            # 获取当前对话历史中的最后一条消息
            last_msg = event["messages"][-1]

            # 情况 A: AI 正在输出文本内容 (不是在调用工具)
            if last_msg.type == "ai" and last_msg.content and not last_msg.tool_calls:
                full_content = last_msg.content
                new_content = full_content[last_sent_len:]

                if new_content:
                    # 包装成前端 Vue 约定的 JSON 格式
                    yield f"data: {json.dumps({'text': new_content}, ensure_ascii=False)}\n\n"
                    last_sent_len = len(full_content)

            # 情况 B: 可以在这里增加对工具调用的日志监控 (可选)
            elif last_msg.type == "ai" and last_msg.tool_calls:
                # 如果你想让前端看到“正在抢券...”的提示，可以在这里 yield
                pass

    # 传输结束标识
    yield "data: [DONE]\n\n"


# 3. 定义接口
@app.post("/chat")
async def chat_endpoint(request: Request, chat_request: ChatRequest):
    # 1. 从 Header 中获取 authorization
    token = request.headers.get("authorization")

    # 如果前端没传 token，可以根据业务需求报错或允许匿名
    if not token:
        backend_log.warning("未检测到 authorization header")

    backend_log.info(f"收到前端请求: {chat_request.message} | Token: {token[:30] if token else 'None'}...")

    # 2. 将 token 传递给生成器
    return StreamingResponse(
        agent_stream_generator(chat_request.message, token),
        media_type="text/event-stream"
    )


if __name__ == "__main__":
    # 打印华丽的启动提示
    print("\n" + "=" * 70)
    print("点评 AI Agent 后端服务已准备就绪！")
    print("强烈建议：请打开一个新的 PowerShell 终端，执行以下命令实时监控 AI 大脑思考过程：")
    print("-" * 70)
    print("chcp 65001")
    print('Get-Content -Path "logs/backend.log" -Wait -Tail 10 -Encoding utf8')
    print("=" * 70 + "\n")

    # 正常启动 Web 服务
    uvicorn.run(app, host="127.0.0.1", port=8000)

