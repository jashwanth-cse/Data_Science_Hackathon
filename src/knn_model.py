from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier


def build_masked_product_dataset(
    basket_matrix: pd.DataFrame, product_categories: pd.Series
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    features = basket_matrix.columns.tolist()
    rows: list[np.ndarray] = []
    targets: list[str] = []
    groups: list[int] = []
    target_products: list[str] = []
    for order_id, basket in basket_matrix.iterrows():
        present_indices = np.flatnonzero(basket.to_numpy())
        for target_idx in present_indices:
            masked = basket.to_numpy(dtype=np.uint8).copy()
            masked[target_idx] = 0
            assert masked[target_idx] == 0
            target_product = features[target_idx]
            rows.append(masked)
            targets.append(str(product_categories[target_product]))
            groups.append(int(order_id))
            target_products.append(target_product)
    return (
        np.vstack(rows),
        np.asarray(targets),
        np.asarray(groups),
        np.asarray(target_products),
        features,
    )


def train_evaluate_knn(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_columns: list[str],
    random_state: int,
    n_jobs: int,
    model_path: str,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    train_groups, test_groups = groups[train_idx], groups[test_idx]
    assert set(train_groups).isdisjoint(set(test_groups))

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    estimator = KNeighborsClassifier(metric="cosine", algorithm="brute", n_jobs=1)
    search = GridSearchCV(
        estimator,
        param_grid={"n_neighbors": [3, 5, 7, 9, 11, 15, 21], "weights": ["uniform", "distance"]},
        scoring="f1_macro",
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
        return_train_score=False,
    )
    search.fit(X[train_idx], y[train_idx], groups=train_groups)
    best = search.best_estimator_
    best.set_params(n_jobs=n_jobs)
    predictions = best.predict(X[test_idx])
    probabilities = best.predict_proba(X[test_idx])
    majority_class = pd.Series(y[train_idx]).mode().iloc[0]
    baseline = np.full(len(test_idx), majority_class, dtype=object)
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y[test_idx], predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y[test_idx], predictions)),
        "macro_f1": float(f1_score(y[test_idx], predictions, average="macro")),
        "weighted_f1": float(f1_score(y[test_idx], predictions, average="weighted")),
        "top_3_accuracy": float(
            top_k_accuracy_score(y[test_idx], probabilities, k=3, labels=best.classes_)
        ),
        "majority_class_baseline_accuracy": float(accuracy_score(y[test_idx], baseline)),
        "majority_class": str(majority_class),
        "cv_best_macro_f1": float(search.best_score_),
        "selected_k": int(search.best_params_["n_neighbors"]),
        "weights": str(search.best_params_["weights"]),
        "metric": "cosine",
        "algorithm": "brute",
        "random_state": random_state,
        "train_examples": int(len(train_idx)),
        "test_examples": int(len(test_idx)),
        "train_baskets": int(len(set(train_groups))),
        "test_baskets": int(len(set(test_groups))),
        "group_overlap_count": int(len(set(train_groups).intersection(set(test_groups)))),
        "class_names": best.classes_.tolist(),
        "feature_columns": feature_columns,
    }
    report = pd.DataFrame(classification_report(y[test_idx], predictions, output_dict=True, zero_division=0)).T
    report.index.name = "class"
    cv_results = pd.DataFrame(search.cv_results_)[
        ["param_n_neighbors", "param_weights", "mean_test_score", "std_test_score", "rank_test_score"]
    ].sort_values("rank_test_score")
    artifact = {
        "model": best,
        "metadata": metrics,
    }
    joblib.dump(artifact, model_path)
    return metrics, report.reset_index(), cv_results

