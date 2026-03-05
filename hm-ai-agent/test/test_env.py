from sentence_transformers import SentenceTransformer
from redis.commands.search.index_definition import IndexDefinition, IndexType
import numpy as np

# 1. 测试模型库是否正常
print("正在检查模型库...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("✅ SentenceTransformers 加载正常")

# 2. 测试 Redis 向量组件是否正常
print("正在检查 Redis 组件...")
# 如果下面这行不报错，说明你的 redis 库版本是正确的
test_def = IndexDefinition(prefix=["test:"])
print("✅ Redis 向量组件导入正常")