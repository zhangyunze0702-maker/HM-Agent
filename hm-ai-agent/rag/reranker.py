import time
from typing import List
from flashrank import Ranker, RerankRequest
from langchain_core.documents import Document
# 接入你的日志系统
from core.logger import get_logger

# 初始化专属于重排模块的 logger
log = get_logger("RERANKER")

# 1. 在模块级别初始化，确保只在项目启动时加载一次模型
_ranker_instance = None

try:
    log.info("正在初始化本地 Reranker (FlashRank - ms-marco-MiniLM-L-12-v2)...")
    _ranker_instance = Ranker(
        model_name="ms-marco-MiniLM-L-12-v2",
        cache_dir="models/flashrank"
    )
    log.info("Reranker 模型加载成功，随时待命。")
except Exception as e:
    log.error(f"Reranker 加载失败: {e}", exc_info=True)
    _ranker_instance = None


def rerank_docs(query: str, docs: List[Document], top_n: int = 5) -> List[Document]:
    """
    使用 FlashRank 对文档进行重排，提升检索精度。
    """
    if not _ranker_instance or not docs:
        log.warning("Reranker 未就绪或输入文档为空，跳过重排步骤。")
        return docs[:top_n]

    start_time = time.time()

    try:
        # 将 LangChain Document 转换为 FlashRank 格式
        passages = [
            {
                "id": i,
                "text": doc.page_content,
                "meta": doc.metadata.copy()  # 建议 copy 一份原数据
            }
            for i, doc in enumerate(docs)
        ]

        # 执行重排计算
        rerank_request = RerankRequest(query=query, passages=passages)
        results = _ranker_instance.rerank(rerank_request)

        # 转换为最终格式
        reranked_docs = []
        for res in results[:top_n]:
            new_doc = Document(
                page_content=res["text"],
                metadata={
                    **res["meta"],
                    "relevance_score": round(float(res["score"]), 4)  # 格式化分数
                }
            )
            reranked_docs.append(new_doc)

        duration = time.time() - start_time
        log.info(f"重排完成: 召回 {len(docs)} 条 -> 精选 {len(reranked_docs)} 条 | 耗时: {duration:.2f}s")

        return reranked_docs

    except Exception as e:
        log.error(f"重排计算过程出错: {e}", exc_info=True)
        return docs[:top_n]  # 出错时降级处理：返回原始召回的前 N 条
