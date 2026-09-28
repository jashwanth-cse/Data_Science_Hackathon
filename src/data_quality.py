from typing import Any

import pandas as pd


def _order_inconsistency_counts(df: pd.DataFrame) -> dict[str, int]:
    columns = ["user_id", "order_date", "time", "order_hour_of_day"]
    return {
        column: int((df.groupby("order_id")[column].nunique(dropna=False) > 1).sum())
        for column in columns
    }


def build_quality_report(raw: pd.DataFrame, typed: pd.DataFrame) -> dict[str, Any]:
    duplicate_sizes = typed.groupby(["order_id", "product_name"], observed=True).size()
    id_to_names = typed.groupby("product_id")["product_name"].nunique()
    name_to_ids = typed.groupby("product_name")["product_id"].nunique()
    id_to_categories = typed.groupby("product_id")["category"].nunique()
    name_to_categories = typed.groupby("product_name")["category"].nunique()
    parsed_hours = pd.to_datetime(typed["time"], format="%H:%M:%S").dt.hour

    report: dict[str, Any] = {
        "raw_rows": int(len(raw)),
        "raw_columns": int(raw.shape[1]),
        "missing_values_total": int(raw.isna().sum().sum()),
        "missing_values_by_column": {k: int(v) for k, v in raw.isna().sum().items()},
        "exact_duplicate_rows": int(raw.duplicated().sum()),
        "duplicate_order_product_groups": int((duplicate_sizes > 1).sum()),
        "rows_in_duplicate_order_product_groups": int(duplicate_sizes[duplicate_sizes > 1].sum()),
        "orders_with_duplicate_product_rows": int(
            typed.loc[
                typed.duplicated(["order_id", "product_name"], keep=False), "order_id"
            ].nunique()
        ),
        "invalid_quantity_count": int((typed["quantity"] <= 0).sum()),
        "negative_price_count": int((typed["price"] < 0).sum()),
        "invalid_hour_count": int((~typed["order_hour_of_day"].between(0, 23)).sum()),
        "blank_product_name_count": int(typed["product_name"].fillna("").eq("").sum()),
        "blank_category_count": int(typed["category"].fillna("").eq("").sum()),
        "time_hour_mismatch_count": int((parsed_hours != typed["order_hour_of_day"]).sum()),
        "order_level_inconsistent_orders": _order_inconsistency_counts(typed),
        "unique_orders": int(typed["order_id"].nunique()),
        "unique_users": int(typed["user_id"].nunique()),
        "unique_product_names": int(typed["product_name"].nunique()),
        "unique_product_ids": int(typed["product_id"].nunique()),
        "unique_categories": int(typed["category"].nunique()),
        "product_id_integrity": {
            "ids_mapping_to_multiple_product_names": int((id_to_names > 1).sum()),
            "product_names_mapping_to_multiple_ids": int((name_to_ids > 1).sum()),
            "ids_mapping_to_multiple_categories": int((id_to_categories > 1).sum()),
            "max_product_names_per_id": int(id_to_names.max()),
            "max_ids_per_product_name": int(name_to_ids.max()),
            "max_categories_per_id": int(id_to_categories.max()),
            "reliable_as_item_key": bool(
                id_to_names.max() == 1 and name_to_ids.max() == 1 and id_to_categories.max() == 1
            ),
        },
        "product_name_category_max_categories": int(name_to_categories.max()),
        "product_name_is_valid_item_key": bool(name_to_categories.max() == 1),
    }
    return report


def assert_data_quality(df: pd.DataFrame, report: dict[str, Any]) -> None:
    assert df["order_id"].notna().all()
    assert df["product_name"].notna().all()
    assert (df["quantity"] > 0).all()
    assert (df["price"] >= 0).all()
    assert df["order_hour_of_day"].between(0, 23).all()
    assert report["blank_product_name_count"] == 0
    assert report["blank_category_count"] == 0
    assert report["time_hour_mismatch_count"] == 0
    assert sum(report["order_level_inconsistent_orders"].values()) == 0
    product_name_to_category = df.groupby("product_name")["category"].nunique()
    assert product_name_to_category.max() == 1

