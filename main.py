"""Точка входа: пайплайн обработки данных проекта «Афиша»."""

import numpy as np
import pandas as pd


def convert_revenue_to_rub(df: pd.DataFrame) -> pd.DataFrame:
    """Переводит выручку в рубли: kzt по курсу за 100 тенге, rub — как есть."""
    rate = pd.read_csv("data/final_tickets_tenge_df.csv")
    rate = rate.rename(columns={"data": "date", "curs": "rate_per_100"})
    rate["date"] = pd.to_datetime(rate["date"]).dt.normalize()

    # защита от повторного запуска: убираем остатки прошлых прогонов
    leftover = [c for c in ("order_date", "date", "rate_per_100", "revenue_rub") if c in df.columns]
    df = df.drop(columns=leftover)

    df["order_date"] = df["order_dt"].dt.normalize()
    df = df.merge(rate[["date", "rate_per_100"]],
                  left_on="order_date", right_on="date", how="left")
    df = df.drop(columns=["order_date", "date"])

    no_rate = df["rate_per_100"].isna().sum()
    print(f"Заказов без курса: {no_rate}")
    if no_rate:
        print(df.loc[df["rate_per_100"].isna(), "order_dt"].dt.date.unique())

    df["revenue_rub"] = np.where(df["currency_code"] == "rub",
                                 df["revenue"],
                                 df["revenue"] * df["rate_per_100"] / 100)

    print(df["currency_code"].value_counts())
    return df


def main() -> None:
    df = pd.read_csv("data/afisha_orders_eda.csv", parse_dates=["order_dt"])
    print(f"Загружено строк: {len(df)} | Колонок: {df.shape[1]}")

    df = convert_revenue_to_rub(df)

    print(df.head())
    print(df.dtypes)


if __name__ == "__main__":
    main()
