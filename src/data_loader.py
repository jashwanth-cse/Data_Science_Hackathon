from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "order_id",
    "user_id",
    "order_date",
    "time",
    "order_hour_of_day",
    "product_name",
    "quantity",
    "price",
    "category",
    "product_id",
}


def normalize_column_name(name: str) -> str:
    return "_".join(str(name).strip().lower().replace("-", " ").split())


def load_transactions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    frame = pd.read_csv(path)
    frame.columns = [normalize_column_name(c) for c in frame.columns]
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return frame


def convert_types(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    for column in ("order_id", "user_id", "order_hour_of_day", "product_id"):
        df[column] = pd.to_numeric(df[column], errors="raise").astype("int64")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="raise").astype("int64")
    df["price"] = pd.to_numeric(df["price"], errors="raise").astype(float)
    df["order_date"] = pd.to_datetime(df["order_date"], errors="raise")
    parsed_time = pd.to_datetime(df["time"], format="%H:%M:%S", errors="raise")
    df["time"] = parsed_time.dt.strftime("%H:%M:%S")
    df["product_name"] = df["product_name"].astype("string").str.strip()
    df["category"] = df["category"].astype("string").str.strip()
    return df

