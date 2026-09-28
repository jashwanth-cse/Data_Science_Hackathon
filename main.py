import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src.association_analysis import add_bootstrap_stability, calculate_pair_associations
from src.bundle_analysis import build_business_actions, select_bundle_candidates
from src.config import PipelineConfig
from src.data_loader import convert_types, load_transactions
from src.data_quality import assert_data_quality, build_quality_report
from src.eda import category_support, product_support, summarize_dataset, time_customer_descriptives
from src.knn_model import build_masked_product_dataset, train_evaluate_knn
from src.preprocessing import build_basket_data, clean_transactions
from src.reporting import generate_report
from src.visualization import create_all_figures


def json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"Not JSON serializable: {type(value)}")


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, default=json_default), encoding="utf-8")


def run_pipeline(config: PipelineConfig) -> None:
    config.create_directories()
    log = logging.getLogger("retail_basket")
    raw = load_transactions(config.input_path)
    typed = convert_types(raw)
    log.info("Dataset shape: %s", typed.shape)
    quality = build_quality_report(raw, typed)
    assert_data_quality(typed, quality)
    write_json(config.output_dir / "data_quality_report.json", quality)
    log.info(
        "Orders=%d products=%d categories=%d duplicate groups aggregated=%d",
        quality["unique_orders"], quality["unique_product_names"], quality["unique_categories"],
        quality["duplicate_order_product_groups"],
    )

    cleaned = clean_transactions(typed)
    basket_items, basket_matrix, category_matrix = build_basket_data(cleaned)
    cleaned.to_csv(config.processed_dir / "cleaned_transactions.csv", index=False)
    basket_items.to_csv(config.processed_dir / "basket_items.csv", index=False)
    basket_matrix.to_csv(config.processed_dir / "basket_matrix.csv", index_label="order_id")
    log.info("Rows removed by aggregation: %d; basket-matrix shape: %s", len(typed) - len(cleaned), basket_matrix.shape)

    summary = summarize_dataset(cleaned, basket_matrix)
    products = product_support(cleaned, basket_matrix)
    categories = category_support(cleaned, category_matrix)
    summary.to_csv(config.output_dir / "dataset_summary.csv", index=False)
    products.to_csv(config.output_dir / "product_support.csv", index=False)
    categories.to_csv(config.output_dir / "category_support.csv", index=False)
    write_json(config.output_dir / "time_customer_descriptives.json", time_customer_descriptives(cleaned))

    pairs, min_count = calculate_pair_associations(basket_matrix)
    pairs = add_bootstrap_stability(
        pairs, basket_matrix, config.random_state, config.bootstrap_resamples,
        config.bootstrap_pairs, config.n_jobs,
    )
    pairs.to_csv(config.output_dir / "pair_associations.csv", index=False)
    candidates = pairs[pairs["is_candidate"]].copy()
    candidates.to_csv(config.output_dir / "candidate_associations.csv", index=False)
    log.info(
        "Pairs tested=%d candidates=%d q<0.05=%d",
        len(pairs), len(candidates), int(pairs["is_fdr_significant"].sum()),
    )

    bundles = select_bundle_candidates(pairs, count=3)
    bundles.to_csv(config.output_dir / "bundle_candidates.csv", index=False)
    actions = build_business_actions(bundles, products, pairs)
    actions.to_csv(config.output_dir / "business_actions.csv", index=False)

    product_categories = cleaned.groupby("product_name")["category"].first()
    X, y, groups, target_products, feature_columns = build_masked_product_dataset(
        basket_matrix, product_categories
    )
    column_index = {name: i for i, name in enumerate(feature_columns)}
    assert all(X[i, column_index[name]] == 0 for i, name in enumerate(target_products))
    metrics, class_report, cv_results = train_evaluate_knn(
        X, y, groups, feature_columns, config.random_state, config.n_jobs,
        str(config.model_dir / "knn_category_affinity.joblib"),
    )
    write_json(config.output_dir / "knn_metrics.json", metrics)
    class_report.to_csv(config.output_dir / "knn_classification_report.csv", index=False)
    cv_results.to_csv(config.output_dir / "knn_cv_results.csv", index=False)
    log.info("Selected KNN k=%d weights=%s metrics=%s", metrics["selected_k"], metrics["weights"],
             {k: round(metrics[k], 4) for k in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "top_3_accuracy"]})

    create_all_figures(products, categories, pairs, config.figures_dir)
    generate_report(
        config.report_dir / "final_report.md", quality, summary, products, categories,
        pairs, bundles, actions, metrics, min_count,
    )
    log.info("Bundle candidates=%d; outputs=%s figures=%s report=%s", len(bundles), config.output_dir,
             config.figures_dir, config.report_dir / "final_report.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end retail basket analysis and grouped KNN")
    parser.add_argument("--input", type=Path, default=Path("data/raw/transactions.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--figures", type=Path, default=Path("figures"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=2, choices=[2], help="Fixed at two CPU workers")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    config = PipelineConfig(
        input_path=args.input, output_dir=args.output, figures_dir=args.figures,
        model_dir=args.model_dir, processed_dir=args.processed_dir,
        report_dir=args.report_dir, random_state=args.random_state, n_jobs=args.n_jobs,
    )
    run_pipeline(config)


if __name__ == "__main__":
    main()

