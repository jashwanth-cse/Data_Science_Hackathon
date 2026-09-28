from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


COLOR = "#2563EB"
ACCENT = "#EA580C"


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_product_support(products: pd.DataFrame, path: Path) -> None:
    data = products.head(15).sort_values("support")
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(data["product_name"], data["support"], color=COLOR)
    for bar, (_, row) in zip(bars, data.iterrows()):
        ax.text(bar.get_width() + 0.0008, bar.get_y() + bar.get_height() / 2,
                f"{int(row['basket_count'])} | {row['support']:.1%}", va="center", fontsize=8)
    ax.set_title("Top 15 Products by Basket Support")
    ax.set_xlabel("Share of baskets")
    ax.set_ylabel("Product")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlim(0, data["support"].max() * 1.27)
    ax.grid(axis="x", alpha=0.2)
    _save(fig, path)


def plot_category_support(categories: pd.DataFrame, path: Path) -> None:
    data = categories.sort_values("support")
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(data["category"], data["support"], color="#0F766E")
    for bar, count in zip(bars, data["basket_count"]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                str(int(count)), va="center", fontsize=8)
    ax.set_title("Category Penetration Across Baskets")
    ax.set_xlabel("Share of baskets (labels show basket count)")
    ax.set_ylabel("Category")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlim(0, data["support"].max() * 1.16)
    ax.grid(axis="x", alpha=0.2)
    _save(fig, path)


def plot_top_pair_lift(pairs: pd.DataFrame, path: Path) -> None:
    data = pairs[pairs["is_candidate"]].sort_values("lift", ascending=False).head(10).copy()
    data["pair"] = data["product_a"] + " + " + data["product_b"]
    data = data.sort_values("lift")
    fig, ax = plt.subplots(figsize=(11, 7.5))
    bars = ax.barh(data["pair"], data["lift"], color=ACCENT)
    for bar, (_, row) in zip(bars, data.iterrows()):
        ax.text(bar.get_width() + 0.015, bar.get_y() + bar.get_height() / 2,
                f"n={int(row['cooccurrence_count'])}, supp={row['pair_support']:.2%}, conf={row['max_confidence']:.1%}",
                va="center", fontsize=7.5)
    ax.axvline(1, color="#374151", linestyle="--", linewidth=1)
    ax.set_title("Highest-Lift Eligible Product Pairs")
    ax.set_xlabel("Lift (1 = independence)")
    ax.set_ylabel("Product pair")
    ax.set_xlim(0, data["lift"].max() * 1.42)
    ax.grid(axis="x", alpha=0.2)
    _save(fig, path)


def plot_support_lift(pairs: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    normal = pairs[~pairs["is_candidate"]]
    candidates = pairs[pairs["is_candidate"]]
    ax.scatter(normal["pair_support"], normal["lift"],
               s=np.maximum(normal["cooccurrence_count"], 2) * 2, alpha=0.25, color="#64748B", label="Other observed pairs")
    ax.scatter(candidates["pair_support"], candidates["lift"],
               s=candidates["cooccurrence_count"] * 4, alpha=0.75, color=ACCENT, label="Eligible candidates")
    for row in candidates.sort_values("lift", ascending=False).head(5).itertuples():
        ax.annotate(f"{row.product_a} + {row.product_b}", (row.pair_support, row.lift),
                    xytext=(4, 5), textcoords="offset points", fontsize=7)
    ax.axhline(1, color="#111827", linestyle="--", linewidth=1, label="Lift = 1")
    ax.set_title("Pair Support vs Lift: Popularity Is Not Association")
    ax.set_xlabel("Pair support (% of baskets)")
    ax.set_ylabel("Lift")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)
    _save(fig, path)


def create_all_figures(
    products: pd.DataFrame, categories: pd.DataFrame, pairs: pd.DataFrame, directory: Path
) -> None:
    plot_product_support(products, directory / "01_product_support.png")
    plot_category_support(categories, directory / "02_category_penetration.png")
    plot_top_pair_lift(pairs, directory / "03_top_pair_lift.png")
    plot_support_lift(pairs, directory / "04_support_lift_scatter.png")

