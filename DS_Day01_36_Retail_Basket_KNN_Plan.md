# DS_Day01_36 — Retail Basket Analysis + KNN
## End-to-End Implementation Plan for Codex

> **Objective:** Build a complete, reproducible retail basket-analysis project from the provided CSV. The implementation must clean and validate the data, perform EDA, identify frequently purchased combinations, calculate support/confidence/lift, distinguish popularity from genuine association, train and evaluate one KNN model, design three data-driven bundle candidates, propose shelf-placement actions, and generate a final business-focused report.

---

## 1. Implementation mandate

Treat this file as an **implementation specification**, not a high-level suggestion.

Codex must:

1. Inspect the actual CSV before writing assumptions into code.
2. Preserve the raw input unchanged.
3. Build a cleaned/validated analytical dataset.
4. Use `product_name` as the basket item identity for association analysis because the supplied `product_id` field fails uniqueness/integrity checks.
5. Use basket-level presence for support/confidence/lift; do **not** count repeated line rows as separate basket occurrences.
6. Treat the KNN requirement as a supervised recommendation/affinity task: predict the **category of one held-out product from the other products in the same basket**.
7. Prevent data leakage by keeping all rows derived from the same `order_id` in the same train/test fold.
8. Quantify statistical uncertainty so high lift with tiny co-occurrence counts is not presented as a proven association.
9. Never fabricate a strong business relationship when the evidence is weak.
10. Produce exactly **4 business-facing visualizations**, **5–7 key insights**, **one final trained KNN model**, evaluation metrics, **3 bundle candidates**, shelf-placement recommendations, and a practical action plan.
11. Make the entire project executable end-to-end from a clean environment with one documented command.
12. Save intermediate outputs so every important result is auditable.

---

# 2. Dataset already inspected

The provided CSV was inspected before writing this plan.

### Observed structure

| Item | Observed value |
|---|---:|
| Rows | 10,000 |
| Unique orders/baskets | 1,991 |
| Unique users | 488 |
| Unique product names | 68 |
| Unique product IDs | 900 |
| Unique categories | 17 |
| Date range | 2024-01-01 to 2024-12-31 |
| Exact duplicate rows | 0 |
| Missing values | 0 |
| Average unique products/basket | 4.87 |
| Average transaction rows/basket | 5.02 |
| Average quantity/basket | 10.09 |
| Min/max unique products/basket | 2 / 7 |
| Quantity values observed | 1, 2, 3 |
| Order hours observed | 06:00–22:00 |

### Important integrity finding

The `product_id` column cannot safely be treated as the unique product identifier:

- all 68 product names appear with multiple `product_id` values;
- 899 of the 900 `product_id` values map to multiple product names;
- 899 of 900 IDs also map to multiple categories.

By contrast:

- every `product_name` maps to exactly one `category`;
- every order has exactly one `user_id`, `order_date`, `time`, and `order_hour_of_day`;
- `order_hour_of_day` agrees with the parsed `time` in every observed row.

### Important duplicate-line finding

There are repeated `(order_id, product_name)` rows:

- 595 rows are part of repeated product-name/order combinations;
- 283 orders are affected;
- this must be aggregated before basket construction.

Therefore:

**Analytical item key = `product_name`**

`product_id` should be retained only for data-quality auditing and traceability; it must not be used as the item identity for association mining or as an ordinary ML feature.

### Product popularity is fairly even

Basket occurrence counts range from approximately 117 to 163 baskets per product, corresponding to roughly **5.88%–8.19% support**.

This means the dataset does not have a classic small set of extremely dominant supermarket products. The implementation should therefore distinguish:

- products that are merely frequent/popular;
- pairs whose co-occurrence is materially higher than expected from their individual frequencies.

### Pair sparsity

There are 2,277 unique unordered product pairs among the 68 products.

The strongest observed pairs are still small in absolute count. Using a minimum co-occurrence count of 10 gives 966 eligible pairs.

Several pairs have unadjusted Fisher-test p-values below 0.05, but after Benjamini–Hochberg false-discovery-rate correction across all 2,277 tested pairs, **no pair remains statistically significant at q < 0.05** in the initial inspection.

This is a critical reporting constraint:

> The final report may show high-lift pairs as **candidate associations/hypotheses**, but must not call them statistically validated genuine associations unless the final pipeline confirms the relevant evidence.

---

# 3. Recommended project structure

Create the project in this structure:

```text
retail-basket-analysis/
│
├── data/
│   ├── raw/
│   │   └── <provided_csv>.csv
│   └── processed/
│       ├── cleaned_transactions.csv
│       ├── basket_items.csv
│       └── basket_matrix.csv
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── data_quality.py
│   ├── preprocessing.py
│   ├── eda.py
│   ├── association_analysis.py
│   ├── bundle_analysis.py
│   ├── knn_model.py
│   ├── evaluation.py
│   ├── visualization.py
│   └── reporting.py
│
├── models/
│   └── knn_category_affinity.joblib
│
├── outputs/
│   ├── data_quality_report.json
│   ├── dataset_summary.csv
│   ├── product_support.csv
│   ├── category_support.csv
│   ├── pair_associations.csv
│   ├── candidate_associations.csv
│   ├── bundle_candidates.csv
│   ├── knn_metrics.json
│   ├── knn_classification_report.csv
│   └── business_actions.csv
│
├── figures/
│   ├── 01_product_support.png
│   ├── 02_category_penetration.png
│   ├── 03_top_pair_lift.png
│   └── 04_support_lift_scatter.png
│
├── reports/
│   └── final_report.md
│
├── notebooks/
│   └── exploration.ipynb
│
├── requirements.txt
├── README.md
└── main.py
```

The implementation may adjust filenames slightly, but must preserve the same logical separation.

---

# 4. Environment and reproducibility

Use Python.

Recommended libraries:

```text
pandas
numpy
scipy
scikit-learn
statsmodels
matplotlib
seaborn
joblib
```

Optional:

```text
mlxtend
```

Do not make `mlxtend` mandatory if exhaustive pair counting is cleaner. There are only 68 products, so direct pair counting is simple and auditable.

Pin versions to the environment that Codex actually installs rather than inventing old versions. After installation, record the installed package versions in the report or a generated `environment.txt`.

Set:

```python
RANDOM_STATE = 42
```

All stochastic operations must use this seed.

---

# 5. Data loading and schema handling

## 5.1 Load the CSV

The actual supplied schema contains:

```text
order_id
user_id
order_date
time
order_hour_of_day
product_name
quantity
price
category
product_id
```

The problem statement describes:

```text
Transaction_ID, Product_ID, Product_Category, Quantity
```

Do not force the smaller schema onto the actual file. Use the real columns and explain the mapping in the report:

- `order_id` → transaction/basket ID
- `product_id` → supplied product identifier, but invalid as a unique item key
- `category` → product category
- `quantity` → transaction quantity
- `product_name` → reliable analytical product label

Normalize column names to predictable snake_case.

---

# 6. Data cleaning and validation

Create a dedicated quality pipeline.

## 6.1 Type conversion

Convert:

```python
order_id              -> integer
user_id               -> integer
order_date            -> datetime
time                  -> time/datetime
order_hour_of_day     -> integer
product_name          -> string
quantity              -> numeric/integer
price                 -> float
category              -> string
product_id            -> integer
```

## 6.2 Required validation tests

Check:

- missing values;
- exact duplicate rows;
- duplicate order/product-name lines;
- invalid dates;
- invalid times;
- invalid hours;
- quantities <= 0;
- prices < 0;
- blank product names;
- blank categories;
- order-level inconsistencies.

For each order, verify that:

```text
user_id is constant
order_date is constant
time is constant
order_hour_of_day is constant
```

Verify:

```python
parsed_time.hour == order_hour_of_day
```

Verify the mapping:

```text
product_name -> exactly one category
```

Report any failure with counts.

## 6.3 Product ID integrity audit

Explicitly calculate:

```text
unique product IDs
unique product names
product names per ID
IDs per product name
categories per ID
```

Save this as part of the quality report.

Do not silently discard `product_id`.

The report must explain that `product_id` is not suitable as the item key in this dataset because it is many-to-many with product names/categories.

---

# 7. Handling repeated product rows inside one order

Create a cleaned order-item table at:

```text
order_id + product_name
```

For each combination:

```python
quantity = sum(quantity)
```

Preserve category from the validated one-to-one `product_name -> category` mapping.

Do not treat repeated rows as multiple purchases for support.

Example:

```text
Order 1008:
Cookies qty 3
Cookies qty 1
```

must become:

```text
Order 1008:
Cookies qty 4
```

for quantity analysis, but it is still counted as only **one basket occurrence** of Cookies for support/lift.

Create:

```text
cleaned_transactions.csv
```

with one row per:

```text
order_id + product_name
```

plus:

```text
user_id
order_date
time
order_hour_of_day
category
quantity
```

For price/value:

- do not use raw `price` as a product-level stable price because the same product name has large price variation in this dataset;
- optionally create `line_value = quantity * price` only as an observed monetary proxy;
- clearly label it as a derived transaction-value proxy rather than a validated product price model.

Avoid using `price` for the association model.

---

# 8. Basket-level dataset

Construct a basket dictionary:

```python
order_id -> set(product_name)
```

Every product appears at most once in a basket for support calculations.

Create:

```text
basket_items.csv
```

with:

```text
order_id
product_name
```

Create a binary basket-product matrix:

```text
rows = orders
columns = 68 product names
values = 0/1
```

Shape should be approximately:

```text
1991 x 68
```

Save it as:

```text
basket_matrix.csv
```

Also construct a category-level basket matrix:

```text
1991 x 17
```

for category-level descriptive analysis.

---

# 9. Exploratory Data Analysis

The EDA must answer business questions, not merely print tables.

## 9.1 Dataset overview

Report:

- number of baskets;
- number of customers;
- number of products;
- number of categories;
- date range;
- basket size distribution;
- quantity distribution;
- repeat-customer distribution;
- order-hour distribution.

## 9.2 Basket size

Calculate:

- mean;
- median;
- min;
- max;
- standard deviation;
- percentiles.

Expected observed shape:

- minimum unique products = 2;
- maximum = 7;
- mean ≈ 4.87.

Explain what this means for the available cross-selling opportunity.

## 9.3 Product popularity

Calculate for each product:

```text
basket_count
support
total_quantity
total_sales_proxy
```

Do not use raw row count as the product popularity measure because repeated product rows exist.

Rank by:

1. basket_count/support for popularity;
2. total quantity separately;
3. sales proxy separately, if included.

## 9.4 Category popularity

Calculate:

```text
category
basket_count
support
total_quantity
sales_proxy
```

Remember category support is the percentage of baskets containing the category. Multiple categories can occur in the same basket, so category supports do not have to sum to 100%.

## 9.5 Time patterns

Analyze:

- orders by hour;
- orders by weekday;
- optional month-level order count.

Do not create additional business-facing figures unless needed; these can be tables in the report.

## 9.6 Repeat customer pattern

Calculate:

- orders per user;
- percentage of users with multiple orders;
- maximum orders per user.

Use this only as contextual EDA, not as a predictive feature unless there is a strong reason.

---

# 10. Association analysis

This is the core business analysis.

## 10.1 Item support

For item A:

```text
support(A) =
number of baskets containing A
--------------------------------
total number of baskets
```

Report as both proportion and percentage.

## 10.2 Pair support

For A and B:

```text
support(A,B) =
number of baskets containing A and B
------------------------------------
total number of baskets
```

## 10.3 Confidence

For rule:

```text
A -> B
```

calculate:

```text
confidence(A -> B) =
support(A,B)
-----------------
support(A)
```

Also calculate the reverse:

```text
confidence(B -> A)
```

Do not assume confidence is symmetric.

## 10.4 Lift

Calculate:

```text
lift(A -> B) =
confidence(A -> B)
------------------
support(B)
```

Equivalent:

```text
lift(A,B) =
support(A,B)
-----------------------------
support(A) * support(B)
```

Interpretation:

- lift ≈ 1: co-occurrence close to independent expectation;
- lift > 1: positive association relative to independence;
- lift < 1: negative association.

Important:

> High lift does not automatically mean a useful business relationship. A tiny pair count can generate large lift.

---

# 11. Pair-generation strategy

Because there are only 68 product names, use an exhaustive unordered-pair approach.

For every basket:

```python
for a, b in combinations(sorted(items), 2):
    pair_count[(a, b)] += 1
```

This is transparent and avoids unnecessary algorithmic complexity.

Total possible unordered pairs:

```text
C(68, 2) = 2,278
```

Only observed pairs should be included in the output. The initial inspection found 2,277 observed pairs.

---

# 12. Minimum support/count policy

Do not use an arbitrary 5% pair-support threshold because the data's pair frequencies are sparse.

Use two stages.

## Stage A: full calculation

Calculate statistics for every observed pair.

## Stage B: candidate screening

Use:

```python
MIN_PAIR_COUNT = max(5, ceil(0.005 * number_of_baskets))
```

For 1,991 baskets this is approximately:

```text
10 co-occurring baskets
```

Store all pairs in `pair_associations.csv`, but mark candidate pairs using:

```text
cooccurrence_count >= MIN_PAIR_COUNT
```

This avoids hiding low-count information while keeping the business ranking stable.

---

# 13. Statistical significance and "genuine association"

This section is essential.

For every pair, build a 2x2 contingency table:

```text
                    B present   B absent
A present              n11        n10
A absent               n01        n00
```

Use a one-sided Fisher exact test for positive association because pair counts are relatively small.

Calculate:

```text
p_value
```

Then perform Benjamini–Hochberg correction across **all tested product pairs**:

```text
q_value
```

Use:

```text
q_value < 0.05
```

as the statistical significance threshold.

Also require:

```text
lift > 1
cooccurrence_count >= MIN_PAIR_COUNT
```

for a candidate to be called a statistically supported positive association.

## Business interpretation policy

Use these labels:

### `POPULAR_ONLY`
High individual support, but no meaningful positive pair evidence.

### `POSITIVE_ASSOCIATION_CANDIDATE`
Lift > 1 and adequate co-occurrence, but statistical evidence is not strong enough after multiple-testing correction.

### `ROBUST_ASSOCIATION`
Lift > 1, sufficient co-occurrence, q < 0.05, and stable bootstrap evidence.

Never use labels like:

```text
best bundle
worst bundle
guaranteed cross-sell
proven customer preference
```

unless explicitly supported by the analysis.

---

# 14. Stability analysis for top associations

For the top 10–20 candidate pairs after initial screening, perform bootstrap resampling at the **basket level**.

Use approximately:

```text
1,000 bootstrap resamples
random_state = 42
```

For each resample, recalculate pair lift.

Produce:

```text
lift_mean
lift_median
lift_ci_lower
lift_ci_upper
```

A stable positive relationship should preferably satisfy:

```text
lift_ci_lower > 1
```

along with q < 0.05.

Because the initial dataset inspection showed no pair surviving FDR correction, Codex should expect that the final result may contain **candidate pairs but zero robust statistically validated pairs**.

That is an acceptable and important finding.

---

# 15. Candidate bundle selection

The assignment requires three possible product bundles.

Do not invent them manually.

Automatically derive three candidate bundles from the analysis.

## Recommended priority

First look for 2-product combinations with:

```text
cooccurrence_count >= MIN_PAIR_COUNT
lift > 1
high confidence
stable lift
```

Then optionally evaluate 3-product combinations.

For a 3-item bundle:

```text
A + B + C
```

calculate:

```text
support(A,B,C)
```

and relevant directional confidence/rule metrics.

To keep computation manageable, only examine triples where all constituent pairs pass the candidate count threshold.

## Bundle scoring

Use a transparent ranking rather than a mysterious arbitrary score.

Recommended ranking order:

1. robust statistical evidence;
2. lift;
3. support;
4. confidence;
5. co-occurrence count.

If no pair is statistically robust, select the top three **candidate bundles** by:

```text
eligible co-occurrence
positive lift
confidence
bootstrap stability
```

and clearly label them as testable business hypotheses.

The report must contain exactly three bundle recommendations.

For each:

```text
Bundle name
Products
Support
Co-occurrence count
Confidence A -> B
Confidence B -> A
Lift
p-value
q-value
Bootstrap CI
Business rationale
Recommended promotion test
```

---

# 16. Required KNN model

The problem explicitly specifies:

```text
KNN
```

The KNN implementation must be meaningful for the business problem.

## Chosen task

Train KNN to predict the **category of a held-out product using the other products in that same basket**.

This turns co-purchase structure into a supervised affinity task.

Example:

Original basket:

```text
Milk
Butter
Bread
Soap
```

Create a training example by masking `Milk`:

```text
Observed features:
Butter = 1
Bread = 1
Soap = 1
Milk = 0

Target:
Dairy & Eggs
```

Do the same for every unique product occurrence in each basket.

---

# 17. KNN dataset construction

For each basket containing `m` unique products:

```text
create m supervised examples
```

For each target product:

```text
X = all other product-presence indicators
y = category of held-out target product
```

The initial inspection suggests approximately:

```text
9,701 supervised examples
```

because the dataset contains roughly 9,701 order-product-name combinations after collapsing duplicate product names within an order.

Features:

```text
68 binary product-presence columns
```

The target product feature must always be removed/masked from its own row.

Do not use `product_id`.

Do not use target-category-derived information in the features.

---

# 18. Leakage prevention for KNN

This is mandatory.

Multiple supervised examples originate from the same basket.

Therefore:

> Never randomly split the 9,701 generated rows independently.

That would leak the other masked versions of the same basket into both train and test.

Instead:

1. Split original baskets/orders into train and test groups.
2. Generate masked-item examples separately within each group, or generate first and use `order_id` as the group column.
3. Keep every `order_id` entirely inside one split.

Recommended:

```python
GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)
```

Then perform hyperparameter selection on the training portion using:

```python
StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

with `order_id` as the group.

---

# 19. KNN preprocessing

Use binary item features.

Recommended KNN configuration search:

```text
n_neighbors:
3, 5, 7, 9, 11, 15, 21

weights:
uniform
distance

metric:
cosine

algorithm:
brute
```

Cosine distance is appropriate for binary basket-overlap patterns.

Do not add `price` unless a separate experiment demonstrates that it improves group-aware validation without leakage.

Do not add raw `product_id`.

Do not add the target product itself.

Select the final hyperparameters using:

```text
macro F1
```

on group-aware cross-validation.

---

# 20. KNN evaluation

Report at minimum:

```text
Accuracy
Balanced Accuracy
Macro F1
Weighted F1
Top-3 Accuracy
```

Also report:

```text
Majority-class baseline accuracy
```

so the model is not judged without context.

Save:

```text
outputs/knn_metrics.json
outputs/knn_classification_report.csv
```

Save final model:

```text
models/knn_category_affinity.joblib
```

Also save model metadata:

```text
selected_k
metric
weights
feature_columns
class_names
random_state
train_baskets
test_baskets
```

---

# 21. Expected interpretation of KNN

The initial exploratory experiment suggests exact product/category prediction is difficult because basket composition is weakly structured and products are fairly evenly sampled.

Therefore:

> Do not optimize the model endlessly merely to increase the metric.

If the final KNN performs only slightly better than the majority baseline, report that honestly.

A weak KNN result is still a valid ML result:

```text
"The basket features provide limited predictive information for the held-out product category under group-aware evaluation."
```

This can actually strengthen the business discussion:

- association metrics can reveal local co-occurrence candidates;
- KNN can test whether those patterns generalize enough to predict withheld category membership;
- if generalization is weak, shelf changes should be treated as experiments rather than assumed customer preferences.

---

# 22. Exactly four required visualizations

Create **exactly four main business-facing figures**.

## Figure 1 — Product popularity

File:

```text
figures/01_product_support.png
```

Type:

Horizontal bar chart.

Show:

```text
Top 15 products by basket support
```

Annotate:

```text
basket count
support %
```

Purpose:

Distinguish popular products from association strength.

---

## Figure 2 — Category penetration

File:

```text
figures/02_category_penetration.png
```

Type:

Horizontal bar chart.

Show all 17 categories ranked by:

```text
basket penetration %
```

Annotate counts where readable.

Purpose:

Show which product categories appear across the greatest number of baskets.

---

## Figure 3 — Highest-lift candidate pairs

File:

```text
figures/03_top_pair_lift.png
```

Type:

Horizontal bar chart.

Show top 10 eligible candidate pairs ranked by lift.

For every bar, include compact metadata such as:

```text
co-occurrence count
support %
confidence
```

Do not hide small counts.

Purpose:

Show products that are associated more strongly than their raw popularity would suggest.

---

## Figure 4 — Support vs lift

File:

```text
figures/04_support_lift_scatter.png
```

Type:

Scatter plot.

X-axis:

```text
pair support %
```

Y-axis:

```text
lift
```

Point size:

```text
co-occurrence count
```

Visually identify:

```text
candidate pairs
```

Annotate the top candidate pairs only to prevent label clutter.

Draw a horizontal reference line:

```text
lift = 1
```

Purpose:

Make the central business distinction visible:

- high support + lift near 1 = popular but not especially associated;
- lower support + lift well above 1 = potentially strong cross-sell candidate;
- tiny count + high lift = unstable candidate and must be treated cautiously.

Do not add a fifth primary visualization.

---

# 23. Visual design requirements

Every chart must:

- have a meaningful title;
- have labeled axes;
- use readable font sizes;
- use percentage formatting where relevant;
- include units;
- avoid 3D charts;
- avoid unreadable legends;
- use consistent naming;
- be saved at high resolution;
- use `tight_layout()` or equivalent;
- be reproducible.

Use one figure per plot.

---

# 24. Key insights generation

The final report must contain **5–7 key insights**, generated from the actual computed outputs rather than hardcoded statements.

Each insight should contain:

```text
Finding
Evidence
Business interpretation
Action implication
```

Potential insight types:

1. Basket size / cross-selling opportunity.
2. Most popular products.
3. Most penetrated categories.
4. Top positive-lift candidate pair(s).
5. Difference between popularity and association.
6. Statistical-significance/stability result.
7. KNN generalization result.

Do not force an insight category if the data does not support it.

---

# 25. Required "popular vs genuinely associated" comparison

Create a clear section in the report:

## Popular does not mean associated

Explain with actual data:

- products with high individual support;
- pairs with high support but lift close to 1;
- pairs with lower support but higher lift;
- whether statistically significant associations exist;
- why raw frequency alone should not drive shelf placement.

A good example should use actual products generated by the final analysis.

Do not hardcode product names in the report.

---

# 26. Shelf-placement strategy

Create a practical action table.

Required columns:

```text
Action
Evidence
Placement idea
Expected purpose
How to test
Success metric
Risk/caveat
```

Recommended strategy framework:

### A. Anchor + complementary placement

For a strong candidate pair:

```text
popular anchor product
+
associated complementary product
```

Place them in nearby shelf zones, an end-cap section, or cross-merchandising display.

### B. Do not move products together only because both are popular

If:

```text
high support
+
lift ≈ 1
```

they may simply be independently popular.

### C. Prefer cross-category experiments

When a pair spans categories, it may provide a clearer cross-sell opportunity than grouping two products that are already in the same department.

### D. Use test-and-measure

For each shelf placement experiment:

```text
baseline period
test period
same store/store area when possible
measure unit sales
measure basket attach rate
measure revenue
```

Do not claim causality from observational transaction data.

---

# 27. Bundle design

The final report must provide three bundles in a compact table.

Required fields:

| Bundle | Products | Support | Confidence | Lift | Evidence level | Promotion idea |
|---|---|---:|---:|---:|---|---|

Evidence level must be one of:

```text
Robust association
Candidate association
Weak/experimental candidate
```

The labels must be assigned automatically from the statistical/stability rules.

If no robust pair exists, it is perfectly valid for all three bundles to be labeled:

```text
Candidate association
```

or:

```text
Experimental candidate
```

provided the report explains why.

---

# 28. Business action plan

Produce a final action plan with approximately 5 actions.

Suggested columns:

```text
Priority
Action
Data evidence
Implementation
Measurement
Decision rule
```

Example structure:

```text
1. Test the top candidate pair as a bundle.
2. Place the complementary item adjacent to the anchor.
3. Keep highly popular but independent products in high-traffic zones without assuming cross-sell synergy.
4. Run an A/B or before/after shelf test.
5. Recompute support/lift monthly as more transactions arrive.
```

These are implementation ideas, not guaranteed outcomes.

---

# 29. Final report structure

Generate:

```text
reports/final_report.md
```

with the following sections:

```text
# Retail Basket Analysis

## 1. Executive Summary

## 2. Business Problem

## 3. Dataset Overview

## 4. Data Quality and Cleaning

## 5. Exploratory Data Analysis

## 6. Product Popularity

## 7. Basket Construction

## 8. Association Analysis

### 8.1 Support
### 8.2 Confidence
### 8.3 Lift
### 8.4 Statistical Significance
### 8.5 Stability

## 9. Popularity vs Genuine Association

## 10. KNN Model

### 10.1 Problem Formulation
### 10.2 Feature Construction
### 10.3 Leakage Prevention
### 10.4 Hyperparameter Selection
### 10.5 Evaluation

## 11. Three Product Bundles

## 12. Shelf Placement Recommendations

## 13. Key Insights

## 14. Practical Action Plan

## 15. Limitations

## 16. Conclusion
```

Embed or reference the four figures.

---

# 30. Limitations that must be disclosed

The report must explicitly mention:

1. The supplied `product_id` is not a reliable unique item identifier.
2. The dataset contains repeated product-name rows within some orders.
3. Association metrics are observational and do not prove causality.
4. High lift from low counts can be unstable.
5. Multiple testing matters because thousands of product pairs are evaluated.
6. KNN is evaluated with group-aware splitting to avoid order-level leakage.
7. The supplied dataset covers one calendar year and may not represent all retail seasons/stores/customers.
8. The `price` field varies substantially within product names, so it should not be interpreted as a stable catalog unit price without external validation.
9. If strong statistically validated associations do not emerge, the report must say so directly.

---

# 31. Acceptance criteria

Codex implementation is complete only when all of the following are true:

## Data

- [ ] CSV loads successfully.
- [ ] Raw file is preserved.
- [ ] Cleaning pipeline runs without manual edits.
- [ ] Data-quality report is generated.
- [ ] Product-name/category consistency is checked.
- [ ] Product-ID integrity problem is documented.
- [ ] Duplicate order/product-name lines are aggregated.
- [ ] Basket matrix is generated.

## EDA

- [ ] Product support ranking is generated.
- [ ] Category support ranking is generated.
- [ ] Basket-size statistics are generated.
- [ ] Time/customer descriptive analysis is generated.

## Association analysis

- [ ] Support is calculated.
- [ ] Confidence is calculated in both directions.
- [ ] Lift is calculated.
- [ ] Pair co-occurrence count is retained.
- [ ] Fisher exact test is calculated.
- [ ] BH q-values are calculated.
- [ ] Candidate screening is applied.
- [ ] Bootstrap stability is evaluated for top candidates.
- [ ] 3 bundle candidates are produced.

## KNN

- [ ] Supervised masked-product dataset is generated.
- [ ] Product identity leakage is prevented.
- [ ] Order-level grouping is enforced.
- [ ] KNN hyperparameters are tuned using group-aware CV.
- [ ] One final model is trained.
- [ ] Accuracy is reported.
- [ ] Balanced accuracy is reported.
- [ ] Macro F1 is reported.
- [ ] Weighted F1 is reported.
- [ ] Top-3 accuracy is reported.
- [ ] Majority baseline is reported.
- [ ] Model is saved as `.joblib`.

## Visualizations

Exactly four primary figures exist:

- [ ] `01_product_support.png`
- [ ] `02_category_penetration.png`
- [ ] `03_top_pair_lift.png`
- [ ] `04_support_lift_scatter.png`

## Business outputs

- [ ] 5–7 key insights.
- [ ] Popular vs associated comparison.
- [ ] 3 bundle recommendations.
- [ ] Shelf-placement recommendations.
- [ ] Practical action plan.
- [ ] Limitations.
- [ ] Final report.

---

# 32. Main execution flow

Create a single entry point:

```bash
python main.py
```

The execution should perform:

```text
1. Load raw CSV
2. Validate schema
3. Run data-quality checks
4. Clean/aggregate transactions
5. Build basket data
6. Run EDA
7. Run product/category popularity analysis
8. Run pair association analysis
9. Run statistical significance testing
10. Run bootstrap stability testing
11. Select 3 bundle candidates
12. Build masked-product KNN dataset
13. Split by order_id
14. Tune KNN with grouped CV
15. Train final KNN
16. Evaluate model
17. Generate exactly 4 figures
18. Generate business action table
19. Generate final_report.md
20. Save model and metrics
```

No manual notebook-only step should be required.

---

# 33. Recommended command-line interface

Support:

```bash
python main.py --input data/raw/transactions.csv
```

Optional:

```bash
python main.py \
  --input data/raw/transactions.csv \
  --output outputs \
  --figures figures \
  --model-dir models \
  --random-state 42
```

Defaults should work without requiring every argument.

---

# 34. Logging

Use Python logging.

Log:

```text
dataset shape
number of orders
number of products
number of categories
rows removed
duplicate groups aggregated
basket-matrix shape
number of pairs tested
number of candidate pairs
number passing q < 0.05
selected KNN parameters
final metrics
number of bundle candidates
output locations
```

Do not flood the console.

---

# 35. Quality safeguards

Codex must add assertions such as:

```python
assert df["order_id"].notna().all()
assert df["product_name"].notna().all()
assert (df["quantity"] > 0).all()
assert (df["price"] >= 0).all()
assert df["order_hour_of_day"].between(0, 23).all()
```

Also assert:

```python
product_name_to_category.nunique().max() == 1
```

and verify no target product exists in the KNN input features for each masked example.

Verify:

```text
train_order_ids ∩ test_order_ids == empty
```

before model training.

---

# 36. Code quality requirements

Use functions, not one huge script.

Every major module should have unit-testable functions.

Avoid:

- hardcoded product names;
- hardcoded result rankings;
- arbitrary conclusions;
- data leakage;
- hidden filtering;
- using product IDs as item identity;
- treating repeated rows as separate basket occurrences.

The code must derive all reported numbers dynamically from the input data.

---

# 37. Final business narrative

The final result should answer five practical questions:

### Question 1
What products are popular?

Answer with basket support and quantity.

### Question 2
What products are bought together more than expected?

Answer with support, confidence, lift, and significance/stability.

### Question 3
Which associations are genuine enough to trust?

Answer using co-occurrence count, Fisher test, BH q-value, and bootstrap stability.

### Question 4
Can basket composition generalize to predict a held-out product category?

Answer using the KNN model and group-aware evaluation.

### Question 5
What should the supermarket actually do?

Answer with exactly three bundle candidates and a shelf-placement/test plan.

---

# 38. Important interpretation rule

The most important rule for the final report is:

> **Do not confuse frequency with association.**

A product can have high support because it is widely purchased. That does not mean it causes or strongly predicts purchases of another product.

Likewise:

> **Do not confuse high lift with reliable business opportunity.**

A pair appearing in only a small number of baskets can have a large lift by chance.

Therefore the final ranking should combine:

```text
frequency
+
co-occurrence
+
confidence
+
lift
+
statistical significance
+
stability
```

rather than relying on a single metric.

---

# 39. Expected current-data interpretation to verify after implementation

The initial inspection suggests the following themes, but Codex must recompute and verify them rather than hardcode them:

- Data completeness is very good: no missing values and no exact duplicate rows were detected.
- Product ID integrity is problematic and product name is the safer basket item key.
- Baskets are moderate-sized, with about five distinct products on average.
- Product popularity is fairly evenly distributed.
- Product-pair co-occurrences are sparse.
- Several pairs can show lift above 1, but significance after multiple-testing correction may disappear.
- KNN product/category prediction is likely to have limited predictive strength, so the evaluation must be compared against a baseline.
- Bundle recommendations should therefore be framed as **testable merchandising hypotheses**, not guaranteed customer preferences.

These observations are starting points only. The final report must use the outputs produced by the implementation.

---

# 40. Deliverables checklist for Codex

At completion, return or make available:

```text
data/processed/cleaned_transactions.csv
data/processed/basket_items.csv
data/processed/basket_matrix.csv

outputs/data_quality_report.json
outputs/dataset_summary.csv
outputs/product_support.csv
outputs/category_support.csv
outputs/pair_associations.csv
outputs/candidate_associations.csv
outputs/bundle_candidates.csv
outputs/knn_metrics.json
outputs/knn_classification_report.csv
outputs/business_actions.csv

models/knn_category_affinity.joblib

figures/01_product_support.png
figures/02_category_penetration.png
figures/03_top_pair_lift.png
figures/04_support_lift_scatter.png

reports/final_report.md

README.md
requirements.txt
main.py
```

The README must include:

```text
project overview
dataset schema
installation
run command
output descriptions
association metric definitions
KNN task definition
evaluation methodology
limitations
```

---

# 41. Final instruction to Codex

**Implement the entire pipeline end-to-end now.**

Do not stop after EDA.

Do not stop after association mining.

Do not skip the KNN requirement because the dataset is unsuited to a traditional classification label; use the specified masked-product-category formulation.

Do not leak rows from the same `order_id` across train and test.

Do not use `product_id` as the product identity.

Do not report a high-lift pair as a proven business relationship without checking count, significance, multiple-testing correction, and stability.

Do not fabricate findings when the dataset has weak signal.

Generate all outputs, save the model, save the four figures, and produce the final `reports/final_report.md`.

The final report must be understandable to a supermarket manager while still being technically defensible to an evaluator.

