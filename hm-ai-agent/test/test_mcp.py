# test/test_mcp.py
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import os


async def run_test():
    # 1. 配置服务器启动参数
    # 这里指向你刚才写的 shop_server.py
    server_params = StdioServerParameters(
        command="python",
        args=[r"D:\Document\coding\hm-ai-agent\mcp\mcp_shop_server.py"],
        env=os.environ.copy()  # 确保环境变量传递，以便连接 Redis
    )

    print("正在连接 MCP Server...")

    # 2. 建立 stdio 通信隧道
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 初始化会话
            await session.initialize()
            print("会话初始化成功！")

            # 列出可用工具
            tools = await session.list_tools()
            print(f"发现可用工具: {[t.name for t in tools.tools]}")

            # 调用 search_shops_mcp 工具进行测试
            test_query = "找一家好吃的火锅店"
            print(f"模拟调用: search_shops_mcp(query='{test_query}')")

            result = await session.call_tool(
                "search_shops_mcp",
                arguments={"query": test_query}
            )

            print("\n" + "=" * 40)
            print("MCP Server 返回结果：")
            print("=" * 40)
            # 提取返回的文本内容
            for content in result.content:
                print(content.text)
            print("=" * 40)


if __name__ == "__main__":
    asyncio.run(run_test())
