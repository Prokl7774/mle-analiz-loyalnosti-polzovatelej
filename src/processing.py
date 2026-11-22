"""Выгрузка данных из учебной базы Яндекс.Облака в data/."""

import os

import psycopg2
import pandas as pd
from dotenv import load_dotenv

# Загружаем доступы из .env — пароли в коде больше не живут
load_dotenv()

SQL_QUERY = """
WITH base AS (
    SELECT
        p.user_id,
        p.device_type_canonical,
        p.order_id,
        p.created_dt_msk::date AS order_dt,
        p.created_ts_msk       AS order_ts,
        p.currency_code,
        p.revenue::numeric     AS revenue,
        p.tickets_count,
        e.event_id,
        e.event_name_code      AS event_name,
        p.service_name,
        e.event_type_main,
        r.region_name,
        c.city_name
    FROM afisha.purchases p
    JOIN afisha.events  e ON e.event_id  = p.event_id
    JOIN afisha.city    c ON c.city_id   = e.city_id
    JOIN afisha.regions r ON r.region_id = c.region_id
    WHERE p.device_type_canonical IN ('mobile', 'desktop')
      AND e.event_type_main <> 'Фильм'
)
SELECT
    user_id, device_type_canonical, order_id, order_dt, order_ts,
    currency_code, revenue, tickets_count,
    order_dt - LAG(order_dt) OVER (PARTITION BY user_id ORDER BY order_ts) AS days_since_prev,
    event_id, event_name, service_name, event_type_main, region_name, city_name
FROM base
"""


def fetch_data() -> pd.DataFrame:
    """Подключается к базе и выполняет SQL-запрос."""
    conn = psycopg2.connect(
        dbname=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PWD'),
        host=os.getenv('DB_HOST'),
        port=os.getenv('DB_PORT'),
        sslmode='require',  # база требует защищённое соединение
    )
    try:
        with conn.cursor() as cur:
            cur.execute(SQL_QUERY)
            cols = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
        return pd.DataFrame(rows, columns=cols)
    finally:
        conn.close()


if __name__ == '__main__':
    df = fetch_data()
    print(f'Загружено строк: {len(df)} | Колонок: {df.shape[1]}')
    print(df.head())
    df.to_csv('data/afisha_orders_eda.csv', index=False)
    print('Сохранено: data/afisha_orders_eda.csv')
