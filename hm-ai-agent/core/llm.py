import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# 加载 .env 环境变量
load_dotenv()


def get_llm():
    """
    获取连接到 AutoDL vLLM 服务的语言模型实例
    """
    base_url = os.getenv("AUTODL_BASE_URL")
    api_key = os.getenv("AUTODL_API_KEY", "sk-1234")

    if not base_url:
        raise ValueError("请在 .env 文件中配置 AUTODL_BASE_URL")

    # 使用 ChatOpenAI 包装器，但指向我们自己的服务器
    llm = ChatOpenAI(
        model="qwen",  # 必须与 vLLM 启动时的 --served-model-name 保持一致
        api_key=api_key,  # 假口令，应付 LangChain 的格式校验
        base_url=base_url,  # 指向 AutoDL 的自定义服务地址
        temperature=0.7,  # 创造力参数 (0-1，0 严谨，1 发散)
        max_tokens=1024,  # 单次最大输出字数
        stop=["<|im_end|>", "<|endoftext|>"]
    )
    return llm


# ================= 阶段测试 =================
if __name__ == "__main__":
    print("正在连接 AutoDL 云端大模型...")
    try:
        llm = get_llm()

        # 构造对话消息
        messages = [
            SystemMessage(content="你是一个幽默的美食探店博主。"),
            HumanMessage(content="你好！请用一句话证明你已经上线了。不少于500字")
        ]

        print("发送请求中，等待云端响应...\n")

        # 调用大模型 (invoke)
        response = llm.invoke(messages)

        print("收到远端响应！")
        print(f"AI 回复: {response.content}")

    except Exception as e:
        print(f"连接失败，请检查 AutoDL 服务是否启动，以及 BASE_URL 是否正确: {e}")
