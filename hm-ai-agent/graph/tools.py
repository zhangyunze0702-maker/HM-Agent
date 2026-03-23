from typing import List
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from rag.reranker import rerank_docs
from tools.java_api import claim_voucher
from rag.vectorstore import get_shop_retriever
from langchain_core.tools import tool
from core.logger import get_logger
from rag.reranker import rerank_docs

# ================= 制造工具 (Tools) =================
# 初始化专属于工具层的 logger，这样日志中会显示 [SHOP_TOOL]
tool_log = get_logger("SHOP_TOOL")


import re
from langchain_core.tools import tool
# 假设 tool_log, rerank_docs 等在你的文件顶部已经导入或定义

@tool
def search_shops(query: str) -> str:
    """
    高级检索工具：采用混合检索+重排技术。
    当用户寻找餐厅、寻找推荐、询问附近有什么好吃的时使用。
    """
    tool_log.info(f"收到检索请求: '{query}'")

    try:
        # 1. 召回阶段 (Retrieval)
        from rag.vectorstore import get_shop_retriever
        # 召回 25 条，给重排留出筛选空间
        retriever = get_shop_retriever(k=25)
        initial_docs = retriever.invoke(query)

        if not initial_docs:
            tool_log.warning(f"未命中任何结果: '{query}'")
            return "【系统状态】: 数据库中未找到相关信息，请引导用户更换搜索词。"

        # 2. 重排阶段 (Rerank)
        tool_log.info(f"正在进行 Rerank (召回数: {len(initial_docs)})")
        # 使用 FlashRank 进行精排
        reranked_docs = rerank_docs(query, initial_docs, top_n=3)
        tool_log.info(f"Rerank 完成，选取 Top-{len(reranked_docs)}")

        # 3. 格式化输出 =
        res = [
            "【系统数据返回】(以下为结构化业务数据，请严格提取 SHOP_ID 并保持你的 JSON 输出规范)："
        ]
        valid_count = 0

        for doc in reranked_docs:
            m = doc.metadata
            doc_type = "SHOP" if m.get('type') == 'shop' else "BLOG"
            name = m.get('name') or m.get('title') or "UNKNOWN"

            # --- 核心对齐逻辑：智能提取正确的业务 ID ---
            raw_id = m.get('shop_id') or m.get('id')

            # 长乱码或空值，拦截
            if not raw_id or (isinstance(raw_id, str) and len(raw_id) > 20):
                tool_log.warning(f"拦截脏数据 (名称: {name}, ID: {raw_id})")
                continue

            business_id = str(raw_id)
            score = m.get('relevance_score', 0)
            valid_count += 1

            # JSON 清洗危险字符
            # 将双引号替换为单引号，将换行符替换为空格
            raw_content = str(doc.page_content[:150])
            safe_content = re.sub(r'[\n\r]+', ' ', raw_content) # 剥离换行
            safe_content = safe_content.replace('"', "'")       # 剥离双引号

            info = (
                f"--- ITEM {valid_count} ---\n"
                f"TYPE: {doc_type}\n"
                f"NAME: {name}\n"
                f"SHOP_ID: {business_id}\n"
                f"SCORE: {score:.4f}\n"
                f"SUMMARY: {safe_content}..."
            )
            res.append(info)

        # 兜底：如果过滤完发现全是脏数据
        if valid_count == 0:
            tool_log.error("严重警告：检索到的 Top 数据全部没有合法的业务 ID！")
            return "【系统状态】: 数据异常（缺失合法店铺ID），暂无法提供详情。请更换搜索词。"

        final_result = "\n".join(res)
        tool_log.info(f"[DEBUG] 发给大模型的安全文本:\n{final_result[:400]}\n" + "-" * 50)
        return final_result

    except Exception as e:
        tool_log.error(f"检索系统发生崩溃: {str(e)}", exc_info=True)
        return f"【系统状态】: 检索服务当前异常 (Error: {str(e)})"


@tool
def get_shop_detail_tool(shop_id: str) -> str:
    """
    获取餐厅的详细地址、评分、人均消费等深度信息。
    注意：参数 shop_id 必须通过 search_shops 工具检索获得。
    禁止直接传入餐厅名称！必须是一个数字形式的字符串。
    """
    from tools.java_api import get_shop_detail
    return get_shop_detail(shop_id)


@tool
def get_shop_vouchers_tool(shop_id: str) -> str:
    """
    通过店铺的数据库 ID（通常是数字字符串，如 '1'）查询优惠券。
    注意：严禁传入餐厅名称（如 '103茶餐厅'），必须传入 search_shops 结果中的 'ID' 字段。
    """
    from tools.java_api import get_shop_vouchers
    return get_shop_vouchers(shop_id)


@tool
def claim_voucher_tool(voucher_id: str, config: RunnableConfig) -> str:
    """
    为用户抢购指定 ID 的优惠券。
    注意：必须传入数字形式的 voucher_id，严禁传入优惠券名称。
    """
    # 实现透传
    auth_token = config.get("configurable", {}).get("authorization")

    # 调用解耦后的 API 方法
    return claim_voucher(voucher_id, auth_token)
