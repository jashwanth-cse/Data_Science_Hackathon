import numpy as np
import pandas as pd


def summarize_dataset(cleaned: pd.DataFrame, basket_matrix: pd.DataFrame) -> pd.DataFrame:
    basket_sizes = basket_matrix.sum(axis=1)
    orders_per_user = cleaned[["order_id", "user_id"]].drop_duplicates().groupby("user_id").size()
    metrics = {
        "baskets": int(cleaned["order_id"].nunique()),
        "customers": int(cleaned["user_id"].nunique()),
        "products": int(cleaned["product_name"].nunique()),
        "categories": int(cleaned["category"].nunique()),
        "date_min": cleaned["order_date"].min().date().isoformat(),
        "date_max": cleaned["order_date"].max().date().isoformat(),
        "basket_size_mean": float(basket_sizes.mean()),
        "basket_size_median": float(basket_sizes.median()),
        "basket_size_std": float(basket_sizes.std()),
        "basket_size_min": int(basket_sizes.min()),
        "basket_size_p25": float(basket_sizes.quantile(0.25)),
        "basket_size_p75": float(basket_sizes.quantile(0.75)),
        "basket_size_p90": float(basket_sizes.quantile(0.90)),
        "basket_size_max": int(basket_sizes.max()),
        "quantity_per_basket_mean": float(cleaned.groupby("order_id")["quantity"].sum().mean()),
        "repeat_customer_pct": float((orders_per_user > 1).mean()),
        "orders_per_customer_mean": float(orders_per_user.mean()),
        "max_orders_per_customer": int(orders_per_user.max()),
        "busiest_order_hour": int(cleaned[["order_id", "order_hour_of_day"]].drop_duplicates()["order_hour_of_day"].mode().iloc[0]),
        "busiest_weekday": str(
            cleaned[["order_id", "order_date"]]
            .drop_duplicates()
            .assign(weekday=lambda x: x["order_date"].dt.day_name())["weekday"]
            .mode()
            .iloc[0]
        ),
    }
    return pd.DataFrame({"metric": list(metrics), "value": list(metrics.values())})


def product_support(cleaned: pd.DataFrame, basket_matrix: pd.DataFrame) -> pd.DataFrame:
    n_baskets = len(basket_matrix)
    basket_count = basket_matrix.sum(axis=0)
    quantities = cleaned.groupby("product_name")["quantity"].sum()
    sales = cleaned.groupby("product_name")["line_value_proxy"].sum()
    categories = cleaned.groupby("product_name")["category"].first()
    out = pd.DataFrame(
        {
            "product_name": basket_count.index,
            "category": categories.reindex(basket_count.index).values,
            "basket_count": basket_count.values.astype(int),
            "support": basket_count.values / n_baskets,
            "total_quantity": quantities.reindex(basket_count.index).values.astype(int),
            "total_sales_proxy": sales.reindex(basket_count.index).values,
        }
    )
    return out.sort_values(["basket_count", "product_name"], ascending=[False, True]).reset_index(drop=True)


def category_support(cleaned: pd.DataFrame, category_matrix: pd.DataFrame) -> pd.DataFrame:
    n_baskets = len(category_matrix)
    basket_count = category_matrix.sum(axis=0)
    quantities = cleaned.groupby("category")["quantity"].sum()
    sales = cleaned.groupby("category")["line_value_proxy"].sum()
    out = pd.DataFrame(
        {
            "category": basket_count.index,
            "basket_count": basket_count.values.astype(int),
            "support": basket_count.values / n_baskets,
            "total_quantity": quantities.reindex(basket_count.index).values.astype(int),
            "total_sales_proxy": sales.reindex(basket_count.index).values,
        }
    )
    return out.sort_values(["basket_count", "category"], ascending=[False, True]).reset_index(drop=True)


def time_customer_descriptives(cleaned: pd.DataFrame) -> dict[str, list[dict]]:
    orders = cleaned[["order_id", "user_id", "order_date", "order_hour_of_day"]].drop_duplicates()
    by_hour = orders.groupby("order_hour_of_day").size().rename("orders").reset_index()
    by_weekday = (
        orders.assign(weekday=orders["order_date"].dt.day_name())
        .groupby("weekday")
        .size()
        .rename("orders")
        .reset_index()
    )
    by_month = (
        orders.assign(month=orders["order_date"].dt.to_period("M").astype(str))
        .groupby("month")
        .size()
        .rename("orders")
        .reset_index()
    )
    return {
        "orders_by_hour": by_hour.to_dict("records"),
        "orders_by_weekday": by_weekday.to_dict("records"),
        "orders_by_month": by_month.to_dict("records"),
    }

