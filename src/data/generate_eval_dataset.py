import json
import random
from pathlib import Path


# ===============================
# 路径配置
# ===============================

ROOT = Path(__file__).resolve().parents[2]

BUSINESS_DATA = ROOT / "data/raw/business_data.json"
OUTPUT_PATH = ROOT / "data/processed/eval.json"


# ===============================
# Evaluation 配置
# ===============================

EVAL_SIZE = 1000

random.seed(42)


# ===============================
# 生成评测问题
# ===============================

QUESTIONS = [
    "根据该企业经营数据，分析收入变化趋势以及潜在经营风险。",
    
    "请结合图表信息，对企业当前经营状况进行分析。",
    
    "分析该企业收入、成本和利润之间的关系。",
    
    "根据经营指标变化，判断企业未来可能面临的问题。",
    
    "请从经营管理角度分析该企业的发展情况。"
]


# ===============================
# 生成标准答案
# ===============================

def generate_answer(company):

    revenue = company["revenue"]
    profit = company["profit"]
    cost = company["cost"]

    if profit > 0 and revenue > cost:
        return (
            "企业整体经营状况较好，收入能够覆盖成本并保持盈利。"
            "但是仍需持续关注成本控制和市场竞争压力。"
        )

    elif revenue > cost:
        return (
            "企业收入规模较大，但是盈利能力不足。"
            "需要进一步优化成本结构，提高利润率。"
        )

    else:
        return (
            "企业当前经营压力较大，收入无法有效覆盖成本。"
            "建议重点关注运营效率和盈利模式优化。"
        )


# ===============================
# 主函数
# ===============================

def generate_eval_dataset():

    with open(
        BUSINESS_DATA,
        "r",
        encoding="utf-8"
    ) as f:
        companies = json.load(f)


    # 随机选择测试企业
    eval_companies = random.sample(
        companies,
        EVAL_SIZE
    )


    eval_data = []


    for company in eval_companies:

        company_id = company["company_id"]

        item = {
            "id": company_id,

            "image":
            f"data/processed/charts/{company_id}.png",

            "question":
            random.choice(QUESTIONS),

            "answer":
            generate_answer(company)
        }


        eval_data.append(item)


    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            eval_data,
            f,
            ensure_ascii=False,
            indent=2
        )


    print(
        f"Generated {len(eval_data)} evaluation samples"
    )



if __name__ == "__main__":
    generate_eval_dataset()