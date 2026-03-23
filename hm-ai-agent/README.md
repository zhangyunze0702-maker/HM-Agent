# 点评 AI Agent 助手
该项目名为 **点评 AI Agent 助手**，是一个基于 LangGraph 构建的 Multi-Agent 系统，旨在为点评平台提供智能化的店铺搜索、详情查询及优惠券抢购功能。

### 项目核心内容总结

1.  **Supervisor Architecture**：
    * 系统采用Supervisor模式进行任务路由。Supervisor节点负责分析用户意图，并将任务分发给 Guide Agent 或 Transaction Agent。
    * **Guide Agent**：专注于通过混合检索寻找店铺并查看详情。
    * **Transaction Agent**：负责查询店铺商品、优惠券及执行抢购操作。

2.  **RAG 检索**：
    * **混合检索**：结合了 BM25 关键词检索和基于 Redis 的向量检索 (`MiniLM-L-12-v2`)。
    * **重排（Rerank）**：使用 FlashRank (`MiniLM-L-12-v2`) 对初筛结果进行精排，提升回复准确度。

3.  **工程化与集成**：
    * **后端集成**：通过 REST API 与现有的 Java 业务后端通信，获取实时店铺数据和执行交易指令。
    * **流式响应**：基于 FastAPI 实现 SSE (Server-Sent Events) 流式输出，提供更好的用户交互体验。
    * **MCP 支持**：实现了 MCP 服务端，允许其他支持该协议的客户端调用其店铺搜索工具。

4.  **量化评估系统**：
    * 项目内置了基于 Ragas 的评估框架，通过“LLM-as-a-Judge”技术对 AI 的回答进行Faithfulness、Answer Relevancy 等维度的自动化评分。


## 项目结构

```text
hm-ai-agent/
├── core/               # 核心配置、LLM 实例化与日志管理
├── graph/              # LangGraph 工作流定义、节点逻辑与 Schema
├── rag/                # RAG 模块（向量存储、数据入库、重排器）
├── tools/              # 外部业务接口（Java API 调用）
├── evaluation/         # Ragas 自动化评估数据集构建与运行
├── mcp/                # MCP 服务端实现
└── main.py             # FastAPI 流式后端服务
```

