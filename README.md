# Retail Basket Analysis + KNN

An end-to-end, reproducible supermarket basket analysis that separates popularity from association, quantifies statistical uncertainty, trains a leakage-safe KNN affinity model, and produces three testable bundle recommendations.

## Dataset schema

The input CSV must contain `order_id`, `user_id`, `order_date`, `time`, `order_hour_of_day`, `product_name`, `quantity`, `price`, `category`, and `product_id`. Product name is the analytical item key because the supplied product IDs have many-to-many integrity problems.

## Installation

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Or specify paths explicitly:

```bash
python main.py --input data/raw/transactions.csv --output outputs --figures figures --model-dir models --random-state 42 --n-jobs 2
```

The pipeline is intentionally fixed to two CPU workers for parallel bootstrap/model-selection work.

## Outputs

- `data/processed/`: aggregated transactions, basket items, and binary basket matrix.
- `outputs/`: quality audit, EDA summaries, product/category support, all pair statistics, screened candidates, exactly three bundles, model metrics, class report, CV results, and business actions.
- `models/knn_category_affinity.joblib`: fitted KNN plus metadata.
- `figures/`: exactly four business-facing PNG charts.
- `reports/final_report.md`: management-readable, technically defensible report.

## Association metrics

- Support: fraction of baskets containing an item or pair.
- Confidence A→B: pair count divided by baskets containing A.
- Lift: pair support divided by the product of individual supports; 1 indicates independence.
- Significance: one-sided Fisher exact p-values, corrected across all observed pairs with Benjamini–Hochberg q-values.
- Stability: 1,000 basket-level bootstrap resamples for leading eligible candidates.

Candidate screening requires positive lift and at least `max(5, ceil(0.5% × basket count))` co-occurrences. Robust evidence additionally requires q < 0.05 and a bootstrap 95% lower lift bound above 1.

## KNN task and evaluation

For every product occurrence, the product is masked from its basket and KNN predicts its category using only the other binary product indicators. Orders—not generated examples—are split into train/test groups. Hyperparameters are selected using five-fold `StratifiedGroupKFold` and macro F1; final evaluation reports accuracy, balanced accuracy, macro/weighted F1, top-3 accuracy, and majority baseline.

## Limitations

Associations are observational, sparse high-lift pairs can be unstable, and multiple testing is material. Product IDs are unreliable, repeated item lines require aggregation, and price is not a validated catalog price. One calendar year may not generalize across stores, customers, or future seasons. KNN performance measures affinity signal and does not establish purchase causality.

