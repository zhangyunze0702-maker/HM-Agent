# hm-ai-agent/mcp/mcp_shop_server.py
from mcp.server.fastmcp import FastMCP
import sys
import os

# 💡 确保能找到根目录下的 tools 包
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.tools import search_shops  # 复用你原来的核心函数

mcp = FastMCP("hmdp")

@mcp.tool()
def search_shops_mcp(query: str) -> str:
    """搜索黑马点评中的餐厅和笔记。"""
    # 🚀 直接调用核心层逻辑
    try:
        # 如果你的工具只有一个参数，通常 invoke(query) 也可以
        # 但标准做法是传入参数字典
        return search_shops.invoke({"query": query})
    except Exception as e:
        return f"执行工具出错: {str(e)}"

if __name__ == "__main__":
    mcp.run()