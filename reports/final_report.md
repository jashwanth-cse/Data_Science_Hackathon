# Retail Basket Analysis

## 1. Executive Summary

This analysis covers 1,991 baskets, 68 products, and 17 categories. The strongest observed pairs are merchandising hypotheses, not proven preferences: 0 of 2,277 observed pairs survive false-discovery-rate correction and 0 satisfy the complete robust-association rule. The leakage-safe KNN reaches 7.95% accuracy versus a 12.68% majority baseline. The practical recommendation is to test exactly three candidate bundles with controls and scale only measurable incremental gains.

## 2. Business Problem

The supermarket needs to distinguish popular products from products bought together more often than independence would predict, assess the reliability of those relationships, and turn the evidence into controlled shelf and promotion tests.

## 3. Dataset Overview

- Date range: 2024-01-01 to 2024-12-31.
- Customers: 488; repeat-customer rate: 92.4%.
- Unique basket size: mean 4.87, median 5, range 2–7.
- Busiest observed hour: 11:00; busiest weekday: Tuesday.

## 4. Data Quality and Cleaning

The source contains 10,000 rows, 0 missing values, and 0 exact duplicates. It includes 296 repeated order/product groups covering 595 rows; these were aggregated to one item occurrence per basket while summing quantity and the observed transaction-value proxy. Product name maps consistently to one category. The supplied `product_id` is not a reliable item key: 900 IDs map to multiple names and 68 names map to multiple IDs. It was retained for auditing but excluded from basket identity and modeling.

## 5. Exploratory Data Analysis

The average basket contains 4.87 distinct products and 10.09 total units. Customers place 4.08 orders on average, with a maximum of 10. Hour, weekday, and month counts are captured reproducibly in `outputs/time_customer_descriptives.json`.

## 6. Product Popularity

Dishwashing Liquid leads basket penetration at 8.19% (163 baskets). Popularity is measured by basket presence; total quantity and the transaction-value proxy are reported separately.

![Top products by basket support](../figures/01_product_support.png)

![Category penetration](../figures/02_category_penetration.png)

## 7. Basket Construction

Repeated product-name rows were collapsed before a binary 1,991 × 68 basket matrix was created. Each product can contribute at most once to basket support, confidence, and lift.

## 8. Association Analysis

### 8.1 Support

Item support is the share of baskets containing an item; pair support is the share containing both items. All observed pairs are retained, while business candidates require at least 10 co-occurrences.

### 8.2 Confidence

Both A→B and B→A confidence are calculated because attachment is directional. Candidate ranking reports the stronger direction without implying causality.

### 8.3 Lift

Lift compares observed co-occurrence with independent expectation. Lift above 1 indicates positive association, but count and uncertainty remain essential.

![Highest-lift eligible pairs](../figures/03_top_pair_lift.png)

### 8.4 Statistical Significance

Every observed pair receives a one-sided Fisher exact test and Benjamini–Hochberg q-value across all 2,277 tests. 0 pairs pass q<0.05.

### 8.5 Stability

The top eligible pairs were resampled at the basket level 1,000 times. A robust label requires positive lift, sufficient count, q<0.05, and a bootstrap 95% lower bound above 1. The final analysis finds 0 robust pairs.

## 9. Popularity vs Genuine Association

### Popular does not mean associated

Dishwashing Liquid has high individual support (8.19%). Meanwhile, Butter + Milk has 19 co-occurrences but lift 1.69, close to independent expectation. By contrast, Juice + Shirt has lower pair support (0.75%) but lift 1.93. This is why raw frequency alone should not determine shelf placement; count, lift, multiplicity correction, and bootstrap stability must be considered together.

![Support versus lift](../figures/04_support_lift_scatter.png)

## 10. KNN Model

### 10.1 Problem Formulation

KNN predicts the category of one held-out product from the remaining products in that basket.

### 10.2 Feature Construction

The model uses 68 binary product-name indicators. The target product's indicator is always masked to zero; `product_id`, price, and target-category-derived features are excluded.

### 10.3 Leakage Prevention

Orders are split with grouped holdout, and five-fold `StratifiedGroupKFold` selects hyperparameters on training baskets only. Train/test order overlap is 0.

### 10.4 Hyperparameter Selection

The search compares k ∈ {3, 5, 7, 9, 11, 15, 21} and uniform/distance weighting under cosine distance, maximizing grouped-CV macro F1. Selected: k=11, weights=distance; CV macro F1=0.060.

### 10.5 Evaluation

| Metric | Result |
|---|---:|
| Accuracy | 0.079 |
| Balanced accuracy | 0.055 |
| Macro F1 | 0.052 |
| Weighted F1 | 0.072 |
| Top-3 accuracy | 0.225 |
| Majority baseline accuracy | 0.127 |

The result should be interpreted as evidence about generalizable basket affinity, not as guaranteed category prediction.

## 11. Three Product Bundles

| Bundle | Products | Support | Confidence | Lift | Evidence level | Promotion idea |
|---|---|---|---|---|---|---|
| Bundle 1 | Juice + Shirt | 0.75% | 12.6% | 1.93 | Candidate association | Run a time-limited adjacent-display or joint-offer test and compare attach rate with baseline. |
| Bundle 2 | Croissant + Educational Book | 0.80% | 13.7% | 1.89 | Candidate association | Run a time-limited adjacent-display or joint-offer test and compare attach rate with baseline. |
| Bundle 3 | Detergent + Earphones | 0.95% | 13.4% | 1.85 | Candidate association | Run a time-limited adjacent-display or joint-offer test and compare attach rate with baseline. |

These are exactly three testable hypotheses selected from computed evidence; none should be described as causal.

## 12. Shelf Placement Recommendations

| Action | Evidence | Placement idea | Expected purpose | How to test | Success metric | Risk/caveat |
|---|---|---|---|---|---|---|
| Test Juice + Shirt as a bundle | count=15, lift=1.93, q=1.000 | Adjacent shelf positions or a small end-cap display. | Test whether observed affinity converts into incremental attachment. | Use matched baseline and test periods in the same store area where possible. | Incremental basket attach rate, unit sales, and revenue proxy. | Observational association is not causal; stop if control-adjusted lift is absent. |
| Place the complementary item near the stronger anchor | Directional confidences are 11.5% and 12.6%. | Keep the higher-support item in its normal zone; cross-merchandise the complement nearby. | Reduce discovery friction while preserving the anchor's traffic role. | Use matched baseline and test periods in the same store area where possible. | Incremental basket attach rate, unit sales, and revenue proxy. | Direction reflects conditional frequency, not customer intent. |
| Keep Dishwashing Liquid visible as a popularity anchor | Highest basket support: 8.2% (163 baskets). | Maintain a high-traffic location; do not infer a specific pairing from popularity alone. | Protect availability and shopper convenience. | Use matched baseline and test periods in the same store area where possible. | Incremental basket attach rate, unit sales, and revenue proxy. | High support alone does not establish cross-sell synergy. |
| Avoid grouping products solely because both are popular | Butter + Milk has lift 1.69 despite 19 co-occurrences. | Use independent high-traffic placement unless an experiment proves synergy. | Avoid wasting scarce display space on coincidental frequency. | Use matched baseline and test periods in the same store area where possible. | Incremental basket attach rate, unit sales, and revenue proxy. | A lift near 1 suggests independence in this dataset, not universal independence. |
| Refresh association evidence monthly | Thousands of pairs are tested and low counts make lift estimates unstable. | No permanent move; maintain an experiment backlog. | Accumulate evidence across seasons and detect changing affinities. | Use matched baseline and test periods in the same store area where possible. | Incremental basket attach rate, unit sales, and revenue proxy. | Seasonality, store mix, and repeated testing can create false positives. |

## 13. Key Insights

1. **Finding:** Moderate baskets create room for cross-selling  
**Evidence:** Baskets contain 4.87 unique products on average (median 5).  
**Business interpretation:** Customers already combine several items per trip, so adjacent discovery can be tested.  
**Action implication:** Use small, measurable cross-merchandising pilots rather than broad resets.

2. **Finding:** Dishwashing Liquid is the most penetrated product  
**Evidence:** It appears in 163 baskets (8.19%).  
**Business interpretation:** It is a traffic/popularity anchor, not automatically the best bundle partner.  
**Action implication:** Protect its availability and test complements independently.

3. **Finding:** Grains & Staples leads category penetration  
**Evidence:** It appears in 937 baskets (47.06%).  
**Business interpretation:** Category penetration identifies broad shopping missions but not causal affinities.  
**Action implication:** Use it to prioritize where tests receive exposure.

4. **Finding:** The leading candidate is Juice + Shirt  
**Evidence:** It co-occurs 15 times, with 0.75% support and 1.93 lift.  
**Business interpretation:** Its evidence level is candidate association, so it is a hypothesis rather than a promise.  
**Action implication:** Pilot the pairing and measure incremental attach rate.

5. **Finding:** Popularity and association diverge  
**Evidence:** Butter + Milk co-occurs 19 times but has lift 1.69.  
**Business interpretation:** Raw co-purchase frequency can be explained by individual popularity.  
**Action implication:** Require lift and uncertainty evidence before allocating bundle space.

6. **Finding:** Multiplicity leaves limited validated evidence  
**Evidence:** Across 2,277 observed pairs, 0 pass q<0.05 and 0 meet the full robust rule.  
**Business interpretation:** Many apparent high-lift pairs can occur by chance when thousands are screened.  
**Action implication:** Treat non-robust pairs as experiments and accumulate more data.

7. **Finding:** KNN generalization is benchmarked honestly  
**Evidence:** Group-aware test accuracy is 7.95%, macro F1 0.052, top-3 accuracy 22.49%, versus a 12.68% majority baseline.  
**Business interpretation:** This measures whether the remaining basket predicts the held-out item's category without order leakage.  
**Action implication:** Use the model as an affinity diagnostic, not an autonomous merchandising system.

## 14. Practical Action Plan

| Priority | Action | Evidence | Implementation | Measurement | Decision rule |
|---|---|---|---|---|---|
| 1 | Test Juice + Shirt as a bundle | count=15, lift=1.93, q=1.000 | Four-week pilot after a comparable baseline period. | Attach rate, units, sales proxy. | Scale only if attach rate improves without material category cannibalization. |
| 2 | Place the complementary item near the stronger anchor | Directional confidences are 11.5% and 12.6%. | Use a tagged secondary facing rather than relocating the anchor. | Directional attach rate and anchor sales. | Retain only if complement gains do not reduce anchor availability or sales. |
| 3 | Keep Dishwashing Liquid visible as a popularity anchor | Highest basket support: 8.2% (163 baskets). | Monitor availability and test complements separately. | Availability, units, basket penetration. | Keep anchor placement based on penetration; evaluate complements on incremental lift. |
| 4 | Avoid grouping products solely because both are popular | Butter + Milk has lift 1.69 despite 19 co-occurrences. | Reserve bundle displays for pre-registered tests. | Control-adjusted attach rate. | Reject bundle placement when interval estimates include no improvement. |
| 5 | Refresh association evidence monthly | Thousands of pairs are tested and low counts make lift estimates unstable. | Rerun the reproducible pipeline and compare confidence intervals. | Count, lift, q-value, bootstrap interval over time. | Promote only associations that repeat and survive multiplicity/stability checks. |

## 15. Limitations

1. The supplied `product_id` is not a reliable unique item identifier.
2. Some orders contain repeated product-name rows; these were aggregated before basket analysis.
3. Association metrics are observational and do not prove causality.
4. High lift from low counts can be unstable.
5. Multiple testing matters because thousands of pairs are evaluated.
6. KNN uses group-aware splitting to prevent order-level leakage.
7. The data covers one calendar year and may not represent all seasons, stores, or customers.
8. Price varies substantially within product names and is only used as an observed value proxy, not a stable catalog price.
9. Strong statistically validated associations did not emerge under the complete evidence rule; weak evidence is stated directly.

## 16. Conclusion

Frequency and association answer different questions. Maintain availability for popular products, but allocate bundle and shelf-placement investment only through controlled tests of the three computed candidates. Re-run the pipeline as new baskets accumulate and require repeatable attach-rate improvement before scaling.
