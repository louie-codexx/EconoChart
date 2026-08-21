import json
import random
from pathlib import Path


def generate_company_data(company_id):
    """
    生成单个企业经营数据
    """

    years = [
        2021,
        2022,
        2023,
        2024,
        2025
    ]


    industries = [
        "ecommerce",
        "software",
        "manufacturing",
        "finance"
    ]


    business_types = [
        "high_growth",
        "stable",
        "cost_pressure",
        "declining",
        "user_loss"
    ]


    industry = random.choice(industries)

    business_type = random.choice(business_types)


    # 初始值

    revenue = random.randint(80, 200)

    cost = random.randint(
        int(revenue * 0.5),
        int(revenue * 0.8)
    )

    users = random.randint(500, 3000)



    revenues = []
    costs = []
    users_list = []
    profits = []



    for year in years:


        if business_type == "high_growth":

            revenue_growth = random.randint(20, 40)

            cost_growth = random.randint(10, 25)

            user_growth = random.randint(20, 50)



        elif business_type == "stable":

            revenue_growth = random.randint(5, 15)

            cost_growth = random.randint(5, 12)

            user_growth = random.randint(3, 10)



        elif business_type == "cost_pressure":

            revenue_growth = random.randint(10, 25)

            cost_growth = random.randint(25, 40)

            user_growth = random.randint(5, 15)



        elif business_type == "declining":

            revenue_growth = random.randint(-20, -5)

            cost_growth = random.randint(-5, 10)

            user_growth = random.randint(-20, -5)



        else:   # user_loss

            revenue_growth = random.randint(-5, 15)

            cost_growth = random.randint(5, 20)

            user_growth = random.randint(-30, -10)



        revenue = int(
            revenue * (1 + revenue_growth / 100)
        )


        cost = int(
            cost * (1 + cost_growth / 100)
        )


        users = int(
            users * (1 + user_growth / 100)
        )


        profit = revenue - cost



        revenues.append(revenue)

        costs.append(cost)

        users_list.append(users)

        profits.append(profit)



    data = {

        "company_id":
            f"Company_{company_id:05d}",


        "industry":
            industry,


        "business_type":
            business_type,


        "year":
            years,


        "revenue":
            revenues,


        "cost":
            costs,


        "users":
            users_list,


        "profit":
            profits

    }


    return data



def generate_dataset(
        num_companies=5000
):

    dataset = []


    for i in range(1, num_companies + 1):

        company_data = generate_company_data(i)

        dataset.append(company_data)


    return dataset



if __name__ == "__main__":


    output_path = Path(
        "data/raw/business_data.json"
    )


    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    data = generate_dataset(
        num_companies=5000
    )


    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:


        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=4
        )


    print(
        f"Generated {len(data)} companies"
    )