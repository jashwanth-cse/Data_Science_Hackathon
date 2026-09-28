from pathlib import Path
from typing import Any

import pandas as pd


def _metric(summary: pd.DataFrame, name: str, cast=float):
    value = summary.loc[summary["metric"] == name, "value"].iloc[0]
    return cast(value)


def _markdown_table(df: pd.DataFrame, columns: list[str], formats: dict[str, str] | None = None) -> str:
    view = df[columns].copy()
    formats = formats or {}
    for column, spec in formats.items():
        view[column] = view[column].map(lambda x: spec.format(x) if pd.notna(x) else "n/a")
    headers = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = ["| " + " | ".join(str(v).replace("|", "\\|") for v in row) + " |" for row in view.itertuples(index=False, name=None)]
    return "\n".join([headers, separator, *rows])


def generate_report(
    path: Path,
    quality: dict[str, Any],
    summary: pd.DataFrame,
    products: pd.DataFrame,
    categories: pd.DataFrame,
    pairs: pd.DataFrame,
    bundles: pd.DataFrame,
    actions: pd.DataFrame,
    knn: dict[str, Any],
    min_pair_count: int,
) -> None:
    top_product = products.iloc[0]
    top_category = categories.iloc[0]
    top_pair = bundles.iloc[0]
    significant = int(pairs["is_fdr_significant"].sum())
    robust = int((pairs["evidence_level"] == "Robust association").sum())
    high_support_independent = pairs.assign(distance=(pairs["lift"] - 1).abs()).sort_values(
        ["cooccurrence_count", "distance"], ascending=[False, True]
    ).iloc[0]
    basket_mean = _metric(summary, "basket_size_mean")
    basket_median = _metric(summary, "basket_size_median")

    bundle_table = bundles.rename(columns={
        "bundle": "Bundle", "products": "Products", "support": "Support",
        "max_confidence": "Confidence", "lift": "Lift", "evidence_level": "Evidence level",
        "promotion_idea": "Promotion idea",
    }).copy()
    bundle_table["Confidence"] = bundles[["confidence_a_to_b", "confidence_b_to_a"]].max(axis=1)
    bundle_md = _markdown_table(
        bundle_table,
        ["Bundle", "Products", "Support", "Confidence", "Lift", "Evidence level", "Promotion idea"],
        {"Support": "{:.2%}", "Confidence": "{:.1%}", "Lift": "{:.2f}"},
    )
    action_md = _markdown_table(
        actions,
        ["Action", "Evidence", "Placement idea", "Expected purpose", "How to test", "Success metric", "Risk/caveat"],
    )
    plan_md = _markdown_table(
        actions,
        ["Priority", "Action", "Evidence", "Implementation", "Measurement", "Decision rule"],
    )
    insights = [
        ("Moderate baskets create room for cross-selling",
         f"Baskets contain {basket_mean:.2f} unique products on average (median {basket_median:.0f}).",
         "Customers already combine several items per trip, so adjacent discovery can be tested.",
         "Use small, measurable cross-merchandising pilots rather than broad resets."),
        (f"{top_product['product_name']} is the most penetrated product",
         f"It appears in {int(top_product['basket_count'])} baskets ({top_product['support']:.2%}).",
         "It is a traffic/popularity anchor, not automatically the best bundle partner.",
         "Protect its availability and test complements independently."),
        (f"{top_category['category']} leads category penetration",
         f"It appears in {int(top_category['basket_count'])} baskets ({top_category['support']:.2%}).",
         "Category penetration identifies broad shopping missions but not causal affinities.",
         "Use it to prioritize where tests receive exposure."),
        (f"The leading candidate is {top_pair['products']}",
         f"It co-occurs {int(top_pair['cooccurrence_count'])} times, with {top_pair['support']:.2%} support and {top_pair['lift']:.2f} lift.",
         f"Its evidence level is {top_pair['evidence_level'].lower()}, so it is a hypothesis rather than a promise.",
         "Pilot the pairing and measure incremental attach rate."),
        ("Popularity and association diverge",
         f"{high_support_independent['product_a']} + {high_support_independent['product_b']} co-occurs {int(high_support_independent['cooccurrence_count'])} times but has lift {high_support_independent['lift']:.2f}.",
         "Raw co-purchase frequency can be explained by individual popularity.",
         "Require lift and uncertainty evidence before allocating bundle space."),
        ("Multiplicity leaves limited validated evidence",
         f"Across {len(pairs):,} observed pairs, {significant} pass q<0.05 and {robust} meet the full robust rule.",
         "Many apparent high-lift pairs can occur by chance when thousands are screened.",
         "Treat non-robust pairs as experiments and accumulate more data."),
        ("KNN generalization is benchmarked honestly",
         f"Group-aware test accuracy is {knn['accuracy']:.2%}, macro F1 {knn['macro_f1']:.3f}, top-3 accuracy {knn['top_3_accuracy']:.2%}, versus a {knn['majority_class_baseline_accuracy']:.2%} majority baseline.",
         "This measures whether the remaining basket predicts the held-out item's category without order leakage.",
         "Use the model as an affinity diagnostic, not an autonomous merchandising system."),
    ]
    insight_md = "\n\n".join(
        f"{i}. **Finding:** {finding}  \n**Evidence:** {evidence}  \n**Business interpretation:** {interpretation}  \n**Action implication:** {action}"
        for i, (finding, evidence, interpretation, action) in enumerate(insights, 1)
    )

    report = f"""# Retail Basket Analysis

## 1. Executive Summary

This analysis covers {_metric(summary, 'baskets', int):,} baskets, {_metric(summary, 'products', int)} products, and {_metric(summary, 'categories', int)} categories. The strongest observed pairs are merchandising hypotheses, not proven preferences: {significant} of {len(pairs):,} observed pairs survive false-discovery-rate correction and {robust} satisfy the complete robust-association rule. The leakage-safe KNN reaches {knn['accuracy']:.2%} accuracy versus a {knn['majority_class_baseline_accuracy']:.2%} majority baseline. The practical recommendation is to test exactly three candidate bundles with controls and scale only measurable incremental gains.

## 2. Business Problem

The supermarket needs to distinguish popular products from products bought together more often than independence would predict, assess the reliability of those relationships, and turn the evidence into controlled shelf and promotion tests.

## 3. Dataset Overview

- Date range: {_metric(summary, 'date_min', str)} to {_metric(summary, 'date_max', str)}.
- Customers: {_metric(summary, 'customers', int):,}; repeat-customer rate: {_metric(summary, 'repeat_customer_pct'):.1%}.
- Unique basket size: mean {basket_mean:.2f}, median {basket_median:.0f}, range {_metric(summary, 'basket_size_min', int)}–{_metric(summary, 'basket_size_max', int)}.
- Busiest observed hour: {_metric(summary, 'busiest_order_hour', int):02d}:00; busiest weekday: {_metric(summary, 'busiest_weekday', str)}.

## 4. Data Quality and Cleaning

The source contains {quality['raw_rows']:,} rows, {quality['missing_values_total']} missing values, and {quality['exact_duplicate_rows']} exact duplicates. It includes {quality['duplicate_order_product_groups']} repeated order/product groups covering {quality['rows_in_duplicate_order_product_groups']} rows; these were aggregated to one item occurrence per basket while summing quantity and the observed transaction-value proxy. Product name maps consistently to one category. The supplied `product_id` is not a reliable item key: {quality['product_id_integrity']['ids_mapping_to_multiple_product_names']} IDs map to multiple names and {quality['product_id_integrity']['product_names_mapping_to_multiple_ids']} names map to multiple IDs. It was retained for auditing but excluded from basket identity and modeling.

## 5. Exploratory Data Analysis

The average basket contains {basket_mean:.2f} distinct products and {_metric(summary, 'quantity_per_basket_mean'):.2f} total units. Customers place {_metric(summary, 'orders_per_customer_mean'):.2f} orders on average, with a maximum of {_metric(summary, 'max_orders_per_customer', int)}. Hour, weekday, and month counts are captured reproducibly in `outputs/time_customer_descriptives.json`.

## 6. Product Popularity

{top_product['product_name']} leads basket penetration at {top_product['support']:.2%} ({int(top_product['basket_count'])} baskets). Popularity is measured by basket presence; total quantity and the transaction-value proxy are reported separately.

![Top products by basket support](../figures/01_product_support.png)

![Category penetration](../figures/02_category_penetration.png)

## 7. Basket Construction

Repeated product-name rows were collapsed before a binary {_metric(summary, 'baskets', int):,} × {_metric(summary, 'products', int)} basket matrix was created. Each product can contribute at most once to basket support, confidence, and lift.

## 8. Association Analysis

### 8.1 Support

Item support is the share of baskets containing an item; pair support is the share containing both items. All observed pairs are retained, while business candidates require at least {min_pair_count} co-occurrences.

### 8.2 Confidence

Both A→B and B→A confidence are calculated because attachment is directional. Candidate ranking reports the stronger direction without implying causality.

### 8.3 Lift

Lift compares observed co-occurrence with independent expectation. Lift above 1 indicates positive association, but count and uncertainty remain essential.

![Highest-lift eligible pairs](../figures/03_top_pair_lift.png)

### 8.4 Statistical Significance

Every observed pair receives a one-sided Fisher exact test and Benjamini–Hochberg q-value across all {len(pairs):,} tests. {significant} pairs pass q<0.05.

### 8.5 Stability

The top eligible pairs were resampled at the basket level 1,000 times. A robust label requires positive lift, sufficient count, q<0.05, and a bootstrap 95% lower bound above 1. The final analysis finds {robust} robust pairs.

## 9. Popularity vs Genuine Association

### Popular does not mean associated

{top_product['product_name']} has high individual support ({top_product['support']:.2%}). Meanwhile, {high_support_independent['product_a']} + {high_support_independent['product_b']} has {int(high_support_independent['cooccurrence_count'])} co-occurrences but lift {high_support_independent['lift']:.2f}, close to independent expectation. By contrast, {top_pair['products']} has lower pair support ({top_pair['support']:.2%}) but lift {top_pair['lift']:.2f}. This is why raw frequency alone should not determine shelf placement; count, lift, multiplicity correction, and bootstrap stability must be considered together.

![Support versus lift](../figures/04_support_lift_scatter.png)

## 10. KNN Model

### 10.1 Problem Formulation

KNN predicts the category of one held-out product from the remaining products in that basket.

### 10.2 Feature Construction

The model uses {len(knn['feature_columns'])} binary product-name indicators. The target product's indicator is always masked to zero; `product_id`, price, and target-category-derived features are excluded.

### 10.3 Leakage Prevention

Orders are split with grouped holdout, and five-fold `StratifiedGroupKFold` selects hyperparameters on training baskets only. Train/test order overlap is {knn['group_overlap_count']}.

### 10.4 Hyperparameter Selection

The search compares k ∈ {{3, 5, 7, 9, 11, 15, 21}} and uniform/distance weighting under cosine distance, maximizing grouped-CV macro F1. Selected: k={knn['selected_k']}, weights={knn['weights']}; CV macro F1={knn['cv_best_macro_f1']:.3f}.

### 10.5 Evaluation

| Metric | Result |
|---|---:|
| Accuracy | {knn['accuracy']:.3f} |
| Balanced accuracy | {knn['balanced_accuracy']:.3f} |
| Macro F1 | {knn['macro_f1']:.3f} |
| Weighted F1 | {knn['weighted_f1']:.3f} |
| Top-3 accuracy | {knn['top_3_accuracy']:.3f} |
| Majority baseline accuracy | {knn['majority_class_baseline_accuracy']:.3f} |

The result should be interpreted as evidence about generalizable basket affinity, not as guaranteed category prediction.

## 11. Three Product Bundles

{bundle_md}

These are exactly three testable hypotheses selected from computed evidence; none should be described as causal.

## 12. Shelf Placement Recommendations

{action_md}

## 13. Key Insights

{insight_md}

## 14. Practical Action Plan

{plan_md}

## 15. Limitations

1. The supplied `product_id` is not a reliable unique item identifier.
2. Some orders contain repeated product-name rows; these were aggregated before basket analysis.
3. Association metrics are observational and do not prove causality.
4. High lift from low counts can be unstable.
5. Multiple testing matters because thousands of pairs are evaluated.
6. KNN uses group-aware splitting to prevent order-level leakage.
7. The data covers one calendar year and may not represent all seasons, stores, or customers.
8. Price varies substantially within product names and is only used as an observed value proxy, not a stable catalog price.
9. Strong statistically validated associations did {'not ' if robust == 0 else ''}emerge under the complete evidence rule; weak evidence is stated directly.

## 16. Conclusion

Frequency and association answer different questions. Maintain availability for popular products, but allocate bundle and shelf-placement investment only through controlled tests of the three computed candidates. Re-run the pipeline as new baskets accumulate and require repeatable attach-rate improvement before scaling.
"""
    path.write_text(report, encoding="utf-8")

