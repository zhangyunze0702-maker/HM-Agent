from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Redis
from langchain_huggingface import HuggingFaceEmbeddings
from core.config import REDIS_HOST, REDIS_PORT, EMBEDDING_MODEL_NAME
import os
from core.logger import get_logger

backend_log = get_logger("AGENT_BACKEND")

os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["ACCELERATE_USE_CPU"] = "True"
_embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={'device': 'cpu'},  
    encode_kwargs={'normalize_embeddings': True}
)


def get_vector_store():
    redis_url = f"redis://{REDIS_HOST}:{REDIS_PORT}/0"
    target_index = "idx:hmdp_v1"

    # 100% 完美匹配的 Schema 适配混合了 shop 和 blog 的新数据结构
    schema = {
        "text": [
            {"name": "content"},
            {"name": "name"},
            {"name": "title"}
        ],
        "tag": [
            {"name": "type"},
            {"name": "shop_id"}
        ],
        "vector": [
            {
                "name": "content_vector",
                "dims": 384,
                "algorithm": "FLAT",
                "datatype": "FLOAT32",
                "distance_metric": "COSINE"
            }
        ]
    }

    try:
        vector_store = Redis.from_existing_index(
            embedding=_embeddings,
            redis_url=redis_url,
            index_name=target_index,
            schema=schema
        )
        return vector_store
    except Exception as e:
        import traceback
        print(f"检索器加载失败，详细错误如下：")
        traceback.print_exc()
        return None



def get_shop_retriever(k=5):
    """
    重构后的进阶检索器：实现双路召回 (BM25 + Vector)
    """
    # 1. 加载 Redis 向量存储
    vector_store = get_vector_store()
    if vector_store is None:
        backend_log.error("向量数据库初始化失败，请检查配置！")
        raise ValueError("向量数据库初始化失败")

    # 向量检索 (Semantic Search)
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": k})

    # 关键词检索 (BM25 / Keyword Search)
    try:
        # 获取基础文档用于构建 BM25 索引
        all_docs = vector_store.similarity_search("", k=500)

        if not all_docs:
            # 替换 print 为 warning
            backend_log.warning("Redis 中未获取到文档，BM25 索引构建跳过。")
            return vector_retriever  # 退化为纯向量模式

        bm25_retriever = BM25Retriever.from_documents(all_docs)
        bm25_retriever.k = k
    except Exception as e:
        # 替换 print 为 error
        backend_log.error(f"BM25 初始化失败: {str(e)}，系统退化至仅向量检索模式。")
        return vector_retriever

    # 合并结果
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )

    # 替换 print 为 info，确保日志流向 stderr 而非 stdout
    backend_log.info("混合检索器 (Hybrid Retriever) 初始化完成：BM25 + Vector 双路模式已就绪。")
    return ensemble_retriever


def get_embeddings():
    return _embeddings


# ================= 阶段测试 =================
if __name__ == "__main__":
    print("正在初始化检索器...")
    retriever = get_shop_retriever(k=10)

    query = "辣椒"
    print(f"\n模拟用户提问: '{query}'")
    print("正在进行向量语义检索...\n")

    results = retriever.invoke(query)

    print("检索结果如下：")
    for i, doc in enumerate(results, 1):
        print(f"[{i}] 店铺名: {doc.metadata.get('name')}")
        print(f"    相关度匹配文本: {doc.page_content}")
        # 这里同步修改为获取 'id' 而不是 'shop_id'
        print(f"    内部关联 ID: {doc.metadata.get('id')}\n")
