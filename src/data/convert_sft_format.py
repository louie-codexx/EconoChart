import json


def convert_to_sft():

    # 输入文件
    input_file = "data/processed/instruction_data.json"

    # 输出文件
    output_file = "data/processed/train_sft.json"


    # 读取原始instruction数据
    with open(
        input_file,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)


    sft_data = []


    for item in data:


        sample = {

            "messages": [

                {
                    "role": "user",

                    "content": [

                        {
                            "type": "image",
                            "image": item["image"]
                        },

                        {
                            "type": "text",
                            "text": item["instruction"]
                        }

                    ]

                },


                {
                    "role": "assistant",

                    "content": item["output"]

                }

            ]

        }


        sft_data.append(sample)



    # 保存
    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            sft_data,
            f,
            ensure_ascii=False,
            indent=4
        )


    print(
        f"Converted {len(sft_data)} samples successfully!"
    )



if __name__ == "__main__":

    convert_to_sft()