#!/usr/bin/env python3
import polars as pl
import json


def extract_plots():
    # Читаем Parquet
    df = pl.read_parquet("user_sessions.parquet")

    # 1. Временной ряд - количество сессий по дням
    df_daily = df.group_by(pl.col("timestamp").dt.date()).len().sort("timestamp")
    df_daily.write_csv("model_data/daily_sessions.csv")

    # 2. Распределение просмотров страниц (для гистограммы)
    pages_dist = df.group_by("pages_viewed").len().sort("pages_viewed")
    pages_dist.write_csv("model_data/pages_distribution.csv")

    # 4. Соотношение ботов/нормальных пользователей
    bot_ratio = df.group_by("is_bot").len()
    bot_ratio.write_csv("model_data/bot_ratio.csv")

    # 5. Метрики для metrics diff
    metrics = {
        "avg_pages": float(df["pages_viewed"].mean()),
        "bot_ratio": float(df["is_bot"].mean()),
        "total_sessions": len(df),
        "unique_users": df["user_id"].n_unique(),
        "avg_session_duration": float(df["session_duration"].mean())
    }

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Данные для визуализации сохранены")
    print(df)

if __name__ == "__main__":
    extract_plots()