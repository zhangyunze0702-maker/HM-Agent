import pymysql
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Redis
from core.config import MYSQL_CONFIG, REDIS_HOST, REDIS_PORT, EMBEDDING_MODEL_NAME

# 店铺类型映射表
SHOP_TYPE_MAP = {
    1: "美食", 2: "KTV", 3: "丽人·美发", 4: "健身运动", 5: "按摩·足疗",
    6: "美容SPA", 7: "亲子游乐", 8: "酒吧", 9: "轰趴馆", 10: "美睫·美甲"
}


def ingest_data_to_redis():
    documents = []
    db = pymysql.connect(**MYSQL_CONFIG)

    try:
        with db.cursor(pymysql.cursors.DictCursor) as cursor:
            # 处理店铺数据 (tb_shop)
            sql_shop = "SELECT id, name, type_id, area, address, avg_price, score FROM tb_shop"
            cursor.execute(sql_shop)
            for shop in cursor.fetchall():
                type_name = SHOP_TYPE_MAP.get(shop['type_id'], "其他")
                # 评分除以 10 还原为 1~5 分
                display_score = shop['score'] / 10.0

                # 拼接增强型文本：包含分类和商圈
                text = (f"店铺名：{shop['name']}。类别：{type_name}。位于{shop['area']}，"
                        f"地址：{shop['address']}。人均消费：{shop['avg_price']}元，评分：{display_score}分。")

                documents.append(Document(
                    page_content=text,
                    metadata={
                        "type": "shop",
                        "id": str(shop['id']),
                        "shop_id": str(shop['id']),
                        "name": shop['name']
                    }
                ))

            # 处理探店笔记数据 (tb_blog)
            # 关联 shop 表拿到店铺名，增加 AI 检索时的上下文
            sql_blog = """
                SELECT b.id, b.shop_id, b.title, b.content, s.name as shop_name 
                FROM tb_blog b 
                LEFT JOIN tb_shop s ON b.shop_id = s.id
            """
            cursor.execute(sql_blog)
            for blog in cursor.fetchall():
                # 拼接笔记内容：标题 + 正文
                text = f"关于【{blog['shop_name']}】的探店笔记：{blog['title']}。内容描述：{blog['content']}"

                documents.append(Document(
                    page_content=text,
                    metadata={
                        "type": "blog",
                        "id": str(blog['id']),
                        "shop_id": str(blog['shop_id']),
                        "title": blog['title'],
                        "name": blog['title']
                    }
                ))
        print(f"成功构造 {len(documents)} 条向量文档！")

    except Exception as e:
        print(f"数据提取失败: {e}")
        return
    finally:
        db.close()

    # 向量化并写入
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    redis_url = f"redis://{REDIS_HOST}:{REDIS_PORT}/0"

    Redis.from_documents(
        documents=documents,
        embedding=embeddings,
        redis_url=redis_url,
        index_name="idx:hmdp_v1"  # 使用统一的混合索引
    )


if __name__ == "__main__":
    ingest_data_to_redis()
