import json


DATA_PATH = "data/processed/sft_train.json"


def check_dataset():

    with open(
        DATA_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)


    print(f"样本数量: {len(data)}")


    print("\n第一个样本:")

    print(
        json.dumps(
            data[0],
            ensure_ascii=False,
            indent=2
        )
    )


    print("\n字段检查:")

    print(
        data[0].keys()
    )


if __name__ == "__main__":
    check_dataset()