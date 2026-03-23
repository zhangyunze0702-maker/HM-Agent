import os
import sys
import json
import uuid
import re

from langchain_core.messages import HumanMessage, ToolMessage, AIMessage

# 环境补丁：处理子目录导入与路径问题
# 获取当前脚本所在目录 (evaluation/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (hm-ai-agent/)
project_root = os.path.dirname(current_dir)

# 将项目根目录加入系统路径，确保能 import 到 graph 模块
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 尝试导入工厂函数
try:
    from graph.workflow import create_agent_app

    # 调用工厂函数生成真正的 LangGraph 实例
    agent_graph = create_agent_app()
    print("[系统] 成功通过工厂函数实例化 LangGraph 引擎")
except ImportError as e:
    print(f"[错误] 导入失败，请检查 graph/workflow.py 路径。错误: {e}")
    sys.exit(1)
except Exception as e:
    print(f"[错误] 实例化 Agent 失败，请检查配置或数据库连接。错误: {e}")
    sys.exit(1)

# 测试用例库：问题与人类标准答案 (Ground Truth)
TEST_CASES = [
    {
        "question": "帮我查找103茶餐厅。",
        "ground_truth": "我找到了一家名为“103茶餐厅”的店铺供您参考：\n\n- **103茶餐厅**\n  - 地址：金华路锦昌文华苑29号\n  - 人均消费：80元\n  - 评分：3.7分\n\n这家餐厅位于大关地区，您可以根据自己的需求前往品尝。如果有其他问题或需要更多帮助，请随时告诉我！您想看哪家？"
    },
    {
        "question": "推荐几家肉蟹煲",
        "ground_truth": "看来您对肉蟹煲挺感兴趣的呢！根据我的搜索，我找到了几家不错的店铺供您参考：\n\n1. **胖哥俩·肉蟹煲旗舰店**（店铺业务ID: 36）：这家店的肉蟹煲得到了不少好评，有顾客提到螃蟹很新鲜，鸡爪炖得特别软糯，汤底也很美味。不过也有些反馈提到排队时间较长。"
    },
]


def build_dataset():
    """
    自动化执行任务并收集 Ragas 所需的 contexts 和 answer
    """
    eval_dataset_dict = {
        "question": [],
        "contexts": [],
        "answer": [],
        "ground_truth": []
    }

    print(f"[启动] 开始收集数据，共 {len(TEST_CASES)} 组测试用例...")

    for index, case in enumerate(TEST_CASES):
        q = case["question"]
        gt = case["ground_truth"]
        print(f"\n[{index + 1}/{len(TEST_CASES)}] 正在运行: {q}")

        # 构造输入状态与配置
        state = {"messages": [HumanMessage(content=q)]}
        config = {"configurable": {"thread_id": f"eval_{uuid.uuid4().hex[:6]}"}}

        try:
            # 核心调用：执行 Agent 逻辑
            result = agent_graph.invoke(state, config=config)

            contexts = []
            final_answer = ""

            # 遍历消息历史，精准拦截 search_shops 的输出作为检索上下文
            for msg in result["messages"]:
                # 拦截检索工具的返回内容
                if isinstance(msg, ToolMessage) and msg.name == "search_shops":
                    # 按照之前 search_shops 的拼接格式进行反切分
                    docs = msg.content.split("\n\n")
                    contexts.extend([doc.strip() for doc in docs if doc.strip()])

                # 拦截最后一条有内容的 AIMessage 作为最终回答
                if isinstance(msg, AIMessage) and msg.content.strip():
                    final_answer = msg.content

            # 如果没有触发检索，填充一个占位符防止 Ragas 评分报错
            if not contexts:
                contexts = ["系统内部逻辑判断无需调用外部检索工具。"]

            # 记录数据
            eval_dataset_dict["question"].append(q)
            eval_dataset_dict["contexts"].append(contexts)
            eval_dataset_dict["answer"].append(final_answer)
            eval_dataset_dict["ground_truth"].append(gt)

            print(f"成功提取：召回了 {len(contexts)} 条知识片段")

        except Exception as e:
            print(f"运行失败: {e}")

    # 保存为 JSON 文件
    output_path = os.path.join(current_dir, "eval_dataset.json")

    # 确保文件夹存在
    if not os.path.exists(current_dir):
        os.makedirs(current_dir)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(eval_dataset_dict, f, ensure_ascii=False, indent=2)

    print("-" * 50)
    print(f"任务完成！量化评估数据集已生成：\n {output_path}")


if __name__ == "__main__":
    build_dataset()
