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


def drop_implicit_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Удаляет неявные дубликаты: одинаковый заказ под разными order_id."""
    key = ["user_id", "order_ts", "device_type_canonical",
           "service_name", "event_id", "revenue_rub"]
    mask = df.duplicated(subset=key, keep=False)
    print(f"Строк в неявных дублях: {mask.sum()} ({mask.mean() * 100:.2f}%)")

    before = len(df)
    df = df.drop_duplicates(subset=key, keep="first").copy()
    print(f"Удалено повторов: {before - len(df)} ({(before - len(df)) / before * 100:.2f}%)")
    return df


def recompute_days_since_prev(df: pd.DataFrame) -> pd.DataFrame:
    """Пересчитывает интервал между заказами после удаления дублей."""
    df = df.sort_values(["user_id", "order_dt", "order_id"]).reset_index(drop=True)
    df["days_since_prev"] = df.groupby("user_id")["order_dt"].diff().dt.days
    print("NaN в days_since_prev:", df["days_since_prev"].isna().sum(),
          "— это первые заказы пользователей (норма)")
    return df


def clean_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """Убирает неположительную выручку и выбросы выше 99-го перцентиля."""
    q99 = df["revenue_rub"].quantile(0.99)
    n_nonpos = (df["revenue_rub"] <= 0).sum()
    n_tail = (df["revenue_rub"] > q99).sum()
    n_before = len(df)

    df = df[(df["revenue_rub"] > 0) & (df["revenue_rub"] <= q99)].copy()

    print(f"99-й перцентиль revenue_rub: {q99:.2f} руб")
    print(f"Удалено: неположительных {n_nonpos}, выбросов {n_tail} "
          f"— {(n_nonpos + n_tail) / n_before * 100:.2f}% из {n_before} строк")
    print("Осталось строк:", len(df))
    return df


def drop_anomalous_users(df: pd.DataFrame) -> pd.DataFrame:
    """Исключает пользователей с экстремальным числом заказов (выше 99-го перцентиля)."""
    orders_per_user = df.groupby("user_id")["order_id"].count()
    q99_orders = orders_per_user.quantile(0.99)
    anomaly_mask = orders_per_user > q99_orders
    print(f"99-й перцентиль числа заказов: {q99_orders:.0f}")
    print(f"Пользователей выше порога: {anomaly_mask.sum()}")
    print(f"({anomaly_mask.mean() * 100:.2f}%)")
    print("Топ-5 значений orders_n у аномальных:",
          sorted(orders_per_user[anomaly_mask].values, reverse=True)[:5])

    n_users_before = df["user_id"].nunique()
    df = df[~df["user_id"].isin(orders_per_user[anomaly_mask].index)].copy()
    print(f"Исключено пользователей: {n_users_before - df['user_id'].nunique()}")
    print(f"({(n_users_before - df['user_id'].nunique()) / n_users_before * 100:.2f}%)")
    print("Осталось строк:", len(df))
    return df


def recompute_intervals_final(df: pd.DataFrame) -> pd.DataFrame:
    """Финальный пересчёт интервалов после всех фильтров."""
    df = df.sort_values(["user_id", "order_dt"]).reset_index(drop=True)
    df["days_since_prev"] = df.groupby("user_id")["order_dt"].diff().dt.days

    n_nan = df["days_since_prev"].isna().sum()
    n_users = df["user_id"].nunique()
    n_multi = (df.groupby("user_id").size() >= 2).sum()
    print(f"NaN в days_since_prev: {n_nan} — равно числу пользователей: {n_users}")
    print(f"Пользователей с 2+ оставшимися заказами: {n_multi} — "
          f"столько же непустых интервалов")
    return df


def main() -> None:
    df = pd.read_csv("data/afisha_orders_eda.csv", parse_dates=["order_dt"])
    print(f"Загружено строк: {len(df)} | Колонок: {df.shape[1]}")

    df = convert_revenue_to_rub(df)
    df = drop_implicit_duplicates(df)
    df = recompute_days_since_prev(df)
    df = clean_revenue(df)
    df = drop_anomalous_users(df)
    df = recompute_intervals_final(df)

    print(df.head())
    print(df.dtypes)


if __name__ == "__main__":
    main()

