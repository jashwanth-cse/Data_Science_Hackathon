import numpy as np
import pandas as pd

from src.association_analysis import benjamini_hochberg, calculate_pair_associations
from src.knn_model import build_masked_product_dataset


def test_benjamini_hochberg_is_bounded_and_monotonic_by_rank():
    p = np.array([0.01, 0.04, 0.03, 0.002])
    q = benjamini_hochberg(p)
    assert np.all((q >= 0) & (q <= 1))
    ranked = q[np.argsort(p)]
    assert np.all(np.diff(ranked) >= -1e-12)


def test_pair_metrics_and_candidate_threshold():
    baskets = pd.DataFrame(
        {"A": [1, 1, 1, 0, 0], "B": [1, 1, 0, 0, 0], "C": [0, 0, 1, 1, 1]},
        index=[1, 2, 3, 4, 5],
    )
    pairs, minimum = calculate_pair_associations(baskets)
    ab = pairs[(pairs.product_a == "A") & (pairs.product_b == "B")].iloc[0]
    assert minimum == 5
    assert ab.cooccurrence_count == 2
    assert np.isclose(ab.confidence_a_to_b, 2 / 3)
    assert np.isclose(ab.lift, (2 / 5) / ((3 / 5) * (2 / 5)))


def test_masked_target_feature_is_zero_and_groups_repeat_by_order():
    baskets = pd.DataFrame({"A": [1, 0], "B": [1, 1]}, index=[10, 11])
    categories = pd.Series({"A": "x", "B": "y"})
    X, y, groups, targets, features = build_masked_product_dataset(baskets, categories)
    indices = {name: i for i, name in enumerate(features)}
    assert all(X[i, indices[name]] == 0 for i, name in enumerate(targets))
    assert list(groups).count(10) == 2
    assert len(y) == 3

