import json
import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate

# 从类定义导入
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall
)

from core.llm import get_llm
from rag.vectorstore import get_embeddings


def run_ragas_evaluation():
    # --- 步骤 1：路径与数据加载 ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "eval_dataset.json")

    if not os.path.exists(json_path):
        print(f"错误：找不到数据集文件 {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    eval_dataset = Dataset.from_dict(data)
    print(f"成功加载测试集，包含 {len(eval_dataset)} 条样本")

    # --- 步骤 2：初始化阅卷官 ---
    eval_llm = get_llm()
    eval_embeddings = get_embeddings()

    # --- 步骤 3：核心修复 2 - 实例化指标对象 ---
    # 必须使用 () 进行实例化，否则会报 TypeError
    metrics = [
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision(),
        ContextRecall()
    ]

    print("Ragas 正在阅卷（LLM-as-a-Judge），请稍候...")

    try:
        # 运行评估
        result = evaluate(
            dataset=eval_dataset,
            metrics=metrics,
            llm=eval_llm,
            embeddings=eval_embeddings
        )
    except Exception as e:
        print(f"评估运行阶段崩溃: {e}")
        return

    # --- 步骤 4：处理结果并修复 Pandas 结构 ---
    print("\n" + "=" * 40)
    print("最终量化评估报告")
    print("=" * 40)
    print(result)

    df = result.to_pandas()
    # 确保列名一致性
    df = df.reset_index() if 'question' not in df.columns else df

    output_csv = os.path.join(current_dir, "rag_evaluation_report.csv")
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n详细评分已保存至：{output_csv}")

    # --- 步骤 5：诊断低分案例 ---
    # 兼容处理大小写列名
    f_col = 'faithfulness' if 'faithfulness' in df.columns else 'Faithfulness'
    q_col = 'question' if 'question' in df.columns else 'Question'

    if f_col in df.columns:
        print(f"\n低分案例诊断 ({f_col} < 0.8):")
        low_scores = df[df[f_col] < 0.8]
        for i, row in low_scores.iterrows():
            print(f"题号 {i} | 问题: {str(row.get(q_col))[:20]}... | 得分: {row.get(f_col)}")


if __name__ == "__main__":
    run_ragas_evaluation()
