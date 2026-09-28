import pandas as pd


def select_bundle_candidates(pairs: pd.DataFrame, count: int = 3) -> pd.DataFrame:
    eligible = pairs[pairs["is_candidate"]].copy()
    eligible["robust_rank"] = (eligible["evidence_level"] == "Robust association").astype(int)
    eligible = eligible.sort_values(
        ["robust_rank", "lift", "pair_support", "max_confidence", "cooccurrence_count"],
        ascending=[False, False, False, False, False],
    ).head(count)
    if len(eligible) < count:
        fallback = pairs.loc[~pairs.index.isin(eligible.index)].sort_values(
            ["cooccurrence_count", "lift"], ascending=False
        ).head(count - len(eligible))
        fallback = fallback.copy()
        fallback["evidence_level"] = "Weak/experimental candidate"
        eligible = pd.concat([eligible, fallback])
    rows = []
    for number, row in enumerate(eligible.itertuples(), 1):
        direction = (
            f"{row.product_a} → {row.product_b}"
            if row.confidence_a_to_b >= row.confidence_b_to_a
            else f"{row.product_b} → {row.product_a}"
        )
        rows.append(
            {
                "bundle": f"Bundle {number}",
                "products": f"{row.product_a} + {row.product_b}",
                "product_a": row.product_a,
                "product_b": row.product_b,
                "support": row.pair_support,
                "cooccurrence_count": int(row.cooccurrence_count),
                "confidence_a_to_b": row.confidence_a_to_b,
                "confidence_b_to_a": row.confidence_b_to_a,
                "lift": row.lift,
                "p_value": row.p_value,
                "q_value": row.q_value,
                "lift_ci_lower": row.lift_ci_lower,
                "lift_ci_upper": row.lift_ci_upper,
                "evidence_level": row.evidence_level,
                "business_rationale": f"Test the stronger observed attachment direction ({direction}) without assuming causality.",
                "promotion_idea": "Run a time-limited adjacent-display or joint-offer test and compare attach rate with baseline.",
            }
        )
    result = pd.DataFrame(rows)
    assert len(result) == 3
    return result


def build_business_actions(
    bundles: pd.DataFrame, product_support: pd.DataFrame, pairs: pd.DataFrame
) -> pd.DataFrame:
    top = bundles.iloc[0]
    popular = product_support.iloc[0]
    near_independent = pairs.assign(distance=(pairs["lift"] - 1).abs()).sort_values(
        ["cooccurrence_count", "distance"], ascending=[False, True]
    ).iloc[0]
    common = {
        "How to test": "Use matched baseline and test periods in the same store area where possible.",
        "Success metric": "Incremental basket attach rate, unit sales, and revenue proxy.",
    }
    rows = [
        {
            "Priority": 1,
            "Action": f"Test {top['products']} as a bundle",
            "Evidence": f"count={int(top['cooccurrence_count'])}, lift={top['lift']:.2f}, q={top['q_value']:.3f}",
            "Placement idea": "Adjacent shelf positions or a small end-cap display.",
            "Expected purpose": "Test whether observed affinity converts into incremental attachment.",
            **common,
            "Risk/caveat": "Observational association is not causal; stop if control-adjusted lift is absent.",
            "Implementation": "Four-week pilot after a comparable baseline period.",
            "Measurement": "Attach rate, units, sales proxy.",
            "Decision rule": "Scale only if attach rate improves without material category cannibalization.",
        },
        {
            "Priority": 2,
            "Action": "Place the complementary item near the stronger anchor",
            "Evidence": f"Directional confidences are {top['confidence_a_to_b']:.1%} and {top['confidence_b_to_a']:.1%}.",
            "Placement idea": "Keep the higher-support item in its normal zone; cross-merchandise the complement nearby.",
            "Expected purpose": "Reduce discovery friction while preserving the anchor's traffic role.",
            **common,
            "Risk/caveat": "Direction reflects conditional frequency, not customer intent.",
            "Implementation": "Use a tagged secondary facing rather than relocating the anchor.",
            "Measurement": "Directional attach rate and anchor sales.",
            "Decision rule": "Retain only if complement gains do not reduce anchor availability or sales.",
        },
        {
            "Priority": 3,
            "Action": f"Keep {popular['product_name']} visible as a popularity anchor",
            "Evidence": f"Highest basket support: {popular['support']:.1%} ({int(popular['basket_count'])} baskets).",
            "Placement idea": "Maintain a high-traffic location; do not infer a specific pairing from popularity alone.",
            "Expected purpose": "Protect availability and shopper convenience.",
            **common,
            "Risk/caveat": "High support alone does not establish cross-sell synergy.",
            "Implementation": "Monitor availability and test complements separately.",
            "Measurement": "Availability, units, basket penetration.",
            "Decision rule": "Keep anchor placement based on penetration; evaluate complements on incremental lift.",
        },
        {
            "Priority": 4,
            "Action": "Avoid grouping products solely because both are popular",
            "Evidence": f"{near_independent['product_a']} + {near_independent['product_b']} has lift {near_independent['lift']:.2f} despite {int(near_independent['cooccurrence_count'])} co-occurrences.",
            "Placement idea": "Use independent high-traffic placement unless an experiment proves synergy.",
            "Expected purpose": "Avoid wasting scarce display space on coincidental frequency.",
            **common,
            "Risk/caveat": "A lift near 1 suggests independence in this dataset, not universal independence.",
            "Implementation": "Reserve bundle displays for pre-registered tests.",
            "Measurement": "Control-adjusted attach rate.",
            "Decision rule": "Reject bundle placement when interval estimates include no improvement.",
        },
        {
            "Priority": 5,
            "Action": "Refresh association evidence monthly",
            "Evidence": "Thousands of pairs are tested and low counts make lift estimates unstable.",
            "Placement idea": "No permanent move; maintain an experiment backlog.",
            "Expected purpose": "Accumulate evidence across seasons and detect changing affinities.",
            **common,
            "Risk/caveat": "Seasonality, store mix, and repeated testing can create false positives.",
            "Implementation": "Rerun the reproducible pipeline and compare confidence intervals.",
            "Measurement": "Count, lift, q-value, bootstrap interval over time.",
            "Decision rule": "Promote only associations that repeat and survive multiplicity/stability checks.",
        },
    ]
    return pd.DataFrame(rows)

