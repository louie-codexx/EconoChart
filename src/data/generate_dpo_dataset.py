import json


def generate_rejected_answer(item):
    """
    生成较弱答案
    """

    company = item["instruction"]

    rejected = f"""
{company}

该企业经营情况总体稳定。

收入有所变化，
用户数量有所变化，
企业未来需要继续关注市场发展。
"""

    return rejected.strip()



def generate_dpo_dataset():

    input_file = (
        "data/processed/sft_train.json"
    )

    output_file = (
        "data/processed/dpo_train.json"
    )


    # 读取SFT数据

    with open(
        input_file,
        "r",
        encoding="utf-8"
    ) as f:

        sft_data = json.load(f)


    dpo_data = []


    for item in sft_data:


        prompt = {

            "image":
            item["image"],


            "instruction":
            item["instruction"]

        }


        # 好答案

        chosen = item["answer"]


        # 差答案

        rejected = generate_rejected_answer(
            item
        )


        sample = {

            "prompt": prompt,

            "chosen": chosen,

            "rejected": rejected

        }


        dpo_data.append(sample)



    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            dpo_data,
            f,
            ensure_ascii=False,
            indent=4
        )


    print(
        f"Generated {len(dpo_data)} DPO samples"
    )



if __name__ == "__main__":

    generate_dpo_dataset()