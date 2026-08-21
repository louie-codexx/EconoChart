import json
from pathlib import Path
import matplotlib.pyplot as plt


def generate_charts():

    # 读取企业数据
    with open(
        "data/raw/business_data.json",
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)


    # 图片保存路径
    save_dir = Path(
        "data/processed/charts"
    )

    save_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # 遍历所有公司

    for company_data in data:


        company_id = company_data["company_id"]

        industry = company_data["industry"]

        business_type = company_data["business_type"]


        years = company_data["year"]

        revenue = company_data["revenue"]

        cost = company_data["cost"]

        users = company_data["users"]

        profit = company_data["profit"]



        # 创建画布

        plt.figure(
            figsize=(10,6)
        )


        # 收入

        plt.plot(
            years,
            revenue,
            marker="o",
            label="Revenue"
        )


        # 成本

        plt.plot(
            years,
            cost,
            marker="o",
            label="Cost"
        )


        # 利润

        plt.plot(
            years,
            profit,
            marker="o",
            label="Profit"
        )


        # 用户

        plt.plot(
            years,
            users,
            marker="o",
            label="Users"
        )


        # 标题

        plt.title(
            f"{company_id}\nIndustry:{industry}\nType:{business_type}"
        )


        plt.xlabel(
            "Year"
        )

        plt.ylabel(
            "Value"
        )


        plt.legend()


        plt.grid(
            True
        )


        # 保存

        save_path = (
            save_dir /
            f"{company_id}.png"
        )


        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )


        plt.close()


        print(
            f"Generated {company_id}"
        )



    print(
        "All charts generated successfully!"
    )



if __name__ == "__main__":

    generate_charts()