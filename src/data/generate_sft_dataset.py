import json
from pathlib import Path


def calculate_metrics(company):

    revenue = company["revenue"]
    cost = company["cost"]
    users = company["users"]
    profit = company["profit"]


    revenue_change = revenue[-1] - revenue[0]

    cost_change = cost[-1] - cost[0]

    users_change = users[-1] - users[0]

    profit_change = profit[-1] - profit[0]


    revenue_rate = (
        revenue_change / revenue[0]
    )

    users_rate = (
        users_change / users[0]
    )


    return {

        "revenue_change": revenue_change,
        "cost_change": cost_change,
        "users_change": users_change,
        "profit_change": profit_change,

        "revenue_rate": revenue_rate,
        "users_rate": users_rate

    }



def generate_analysis(company):


    metrics = calculate_metrics(company)


    reasoning = (

        f"收入从{company['revenue'][0]}变化到"
        f"{company['revenue'][-1]}，"
        f"变化率约为"
        f"{metrics['revenue_rate']:.1%}。"

        f"用户数量从"
        f"{company['users'][0]}"
        f"变化到"
        f"{company['users'][-1]}。"

        f"利润变化为"
        f"{metrics['profit_change']}。"

    )


    if (
        metrics["revenue_change"] < 0
        and
        metrics["users_change"] < 0
    ):

        answer = (

            "该企业经营趋势整体下降。"
            "收入和用户规模均出现减少，"
            "说明企业市场竞争力下降，"
            "需要重点关注用户增长和产品优化。"

        )


        difficulty = "hard"


    elif metrics["revenue_change"] > 0:


        answer = (

            "该企业收入规模持续扩大。"
            "但仍需要关注成本增长速度，"
            "避免利润空间被压缩。"

        )


        difficulty = "medium"


    else:


        answer = (

            "该企业经营状态较为稳定，"
            "建议持续关注收入、用户和利润变化。"

        )


        difficulty = "easy"



    return reasoning, answer, difficulty




def generate_sft_dataset():


    input_file = (
        "data/raw/business_data.json"
    )


    output_file = (
        "data/processed/sft_train.json"
    )


    with open(
        input_file,
        "r",
        encoding="utf-8"
    ) as f:

        companies = json.load(f)



    tasks = [

        "请分析该企业近五年的经营趋势",

        "请分析该企业目前存在的经营风险",

        "请分析该企业收入、成本和利润关系",

        "请提出该企业未来经营优化建议",

        "请生成一份企业经营分析报告"

    ]



    dataset = []



    for company in companies:


        image = (
            f"data/processed/charts/"
            f"{company['company_id']}.png"
        )


        reasoning, answer, difficulty = (
            generate_analysis(company)
        )


        for task in tasks:


            sample = {


                "image": image,


                "instruction": task,


                "reasoning": reasoning,


                "answer": answer,


                "industry":
                    company["industry"],


                "business_type":
                    company["business_type"],


                "difficulty":
                    difficulty

            }


            dataset.append(sample)




    Path(
        "data/processed"
    ).mkdir(
        exist_ok=True
    )



    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:


        json.dump(
            dataset,
            f,
            ensure_ascii=False,
            indent=4
        )



    print(
        f"Generated {len(dataset)} SFT samples"
    )



if __name__ == "__main__":

    generate_sft_dataset()