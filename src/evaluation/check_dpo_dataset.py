import json


def check_dpo_dataset():

    file_path = (
        "data/processed/dpo_train.json"
    )


    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)


    print(
        "样本数量:",
        len(data)
    )


    print("\n第一个样本:")
    
    print(
        json.dumps(
            data[0],
            ensure_ascii=False,
            indent=4
        )
    )


    print("\n字段检查:")

    print(
        data[0].keys()
    )


if __name__ == "__main__":

    check_dpo_dataset()