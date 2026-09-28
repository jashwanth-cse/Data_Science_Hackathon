from itertools import combinations
from math import ceil

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import fisher_exact


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    result = np.empty(n, dtype=float)
    result[order] = adjusted
    return result


def calculate_pair_associations(basket_matrix: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    n_baskets = len(basket_matrix)
    min_pair_count = max(5, ceil(0.005 * n_baskets))
    product_counts = basket_matrix.sum(axis=0).astype(int)
    pair_counts: dict[tuple[str, str], int] = {}
    for row in basket_matrix.itertuples(index=False, name=None):
        present = [basket_matrix.columns[i] for i, value in enumerate(row) if value]
        for pair in combinations(present, 2):
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

    records = []
    for (a, b), n11 in pair_counts.items():
        count_a, count_b = int(product_counts[a]), int(product_counts[b])
        n10, n01 = count_a - n11, count_b - n11
        n00 = n_baskets - n11 - n10 - n01
        support = n11 / n_baskets
        confidence_ab = n11 / count_a
        confidence_ba = n11 / count_b
        lift = support / ((count_a / n_baskets) * (count_b / n_baskets))
        _, p_value = fisher_exact([[n11, n10], [n01, n00]], alternative="greater")
        records.append(
            {
                "product_a": a,
                "product_b": b,
                "count_a": count_a,
                "count_b": count_b,
                "cooccurrence_count": n11,
                "pair_support": support,
                "confidence_a_to_b": confidence_ab,
                "confidence_b_to_a": confidence_ba,
                "max_confidence": max(confidence_ab, confidence_ba),
                "lift": lift,
                "p_value": p_value,
            }
        )
    result = pd.DataFrame(records)
    result["q_value"] = benjamini_hochberg(result["p_value"].to_numpy())
    result["is_candidate"] = (result["cooccurrence_count"] >= min_pair_count) & (result["lift"] > 1)
    result["is_fdr_significant"] = result["q_value"] < 0.05
    result = result.sort_values(["lift", "cooccurrence_count"], ascending=[False, False]).reset_index(drop=True)
    return result, min_pair_count


def _bootstrap_pair(
    a_idx: int,
    b_idx: int,
    values: np.ndarray,
    resample_indices: np.ndarray,
) -> tuple[float, float, float, float]:
    a = values[:, a_idx]
    b = values[:, b_idx]
    sampled_a = a[resample_indices]
    sampled_b = b[resample_indices]
    support_a = sampled_a.mean(axis=1)
    support_b = sampled_b.mean(axis=1)
    support_ab = (sampled_a * sampled_b).mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        lift = support_ab / (support_a * support_b)
    lift = lift[np.isfinite(lift)]
    return (
        float(np.mean(lift)),
        float(np.median(lift)),
        float(np.quantile(lift, 0.025)),
        float(np.quantile(lift, 0.975)),
    )


def add_bootstrap_stability(
    pairs: pd.DataFrame,
    basket_matrix: pd.DataFrame,
    random_state: int = 42,
    n_resamples: int = 1000,
    top_n: int = 20,
    n_jobs: int = 2,
) -> pd.DataFrame:
    out = pairs.copy()
    for column in ("lift_mean", "lift_median", "lift_ci_lower", "lift_ci_upper"):
        out[column] = np.nan
    eligible = out[out["is_candidate"]].sort_values(
        ["lift", "pair_support", "max_confidence", "cooccurrence_count"], ascending=False
    ).head(top_n)
    if eligible.empty:
        return out
    rng = np.random.default_rng(random_state)
    resample_indices = rng.integers(0, len(basket_matrix), size=(n_resamples, len(basket_matrix)))
    col_index = {name: i for i, name in enumerate(basket_matrix.columns)}
    values = basket_matrix.to_numpy(dtype=np.uint8)
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_bootstrap_pair)(
            col_index[row.product_a], col_index[row.product_b], values, resample_indices
        )
        for row in eligible.itertuples()
    )
    for idx, stats in zip(eligible.index, results):
        out.loc[idx, ["lift_mean", "lift_median", "lift_ci_lower", "lift_ci_upper"]] = stats
    out["is_bootstrap_stable"] = out["lift_ci_lower"].gt(1).fillna(False)
    out["evidence_level"] = np.select(
        [
            out["is_candidate"] & out["is_fdr_significant"] & out["is_bootstrap_stable"],
            out["is_candidate"] & out["is_bootstrap_stable"],
            out["is_candidate"],
        ],
        ["Robust association", "Candidate association", "Weak/experimental candidate"],
        default="Insufficient evidence",
    )
    return out

