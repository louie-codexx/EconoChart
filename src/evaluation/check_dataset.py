import json
from pathlib import Path


def check_dataset():

    file = Path(
        "data/processed/train_sft.json"
    )


    with open(
        file,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)



    print(
        "样本数量:",
        len(data)
    )


    missing = []


    for item in data:

        image_path = Path(
            item["messages"][0]["content"][0]["image"]
        )


        if not image_path.exists():

            missing.append(
                str(image_path)
            )



    if missing:

        print(
            "缺失图片:"
        )

        for m in missing:
            print(m)

    else:

        print(
            "所有图片路径正常 ✅"
        )



    print("\n随机样本:")
    print(data[0])



if __name__ == "__main__":

    check_dataset()