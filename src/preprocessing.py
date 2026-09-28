import pandas as pd


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["line_value_proxy"] = work["quantity"] * work["price"]
    keys = ["order_id", "product_name"]
    grouped = work.groupby(keys, as_index=False, observed=True).agg(
        user_id=("user_id", "first"),
        order_date=("order_date", "first"),
        time=("time", "first"),
        order_hour_of_day=("order_hour_of_day", "first"),
        category=("category", "first"),
        quantity=("quantity", "sum"),
        line_value_proxy=("line_value_proxy", "sum"),
        source_line_count=("product_id", "size"),
        distinct_product_ids=("product_id", "nunique"),
    )
    assert not grouped.duplicated(keys).any()
    return grouped.sort_values(keys).reset_index(drop=True)


def build_basket_data(cleaned: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    basket_items = cleaned[["order_id", "product_name"]].drop_duplicates().sort_values(
        ["order_id", "product_name"]
    )
    basket_matrix = pd.crosstab(basket_items["order_id"], basket_items["product_name"]).clip(upper=1)
    basket_matrix = basket_matrix.astype("int8").sort_index().sort_index(axis=1)
    order_categories = cleaned[["order_id", "category"]].drop_duplicates()
    category_matrix = pd.crosstab(order_categories["order_id"], order_categories["category"]).clip(upper=1)
    category_matrix = category_matrix.astype("int8").sort_index().sort_index(axis=1)
    assert basket_matrix.to_numpy().max() == 1
    return basket_items.reset_index(drop=True), basket_matrix, category_matrix

