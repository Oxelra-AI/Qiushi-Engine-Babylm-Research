#!/usr/bin/env python3
"""Independent integrity checks for the research proxy-ceiling artifacts."""

from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import math
from pathlib import Path

import pandas as pd


HERE = _public_path('experiments/archive/relation_learning/analysis/proxy_ceiling')


def close(x: float, y: float, tol: float = 1e-12) -> None:
    assert math.isclose(float(x), float(y), rel_tol=0.0, abs_tol=tol), (x, y)


def main() -> None:
    summary = json.loads((_public_path('experiments/archive/relation_learning/analysis/proxy_ceiling/summary.json')).read_text())
    features = pd.read_csv(_public_path('experiments/archive/relation_learning/analysis/proxy_ceiling/proxy_features.csv'))
    targets = pd.read_csv(_public_path('experiments/archive/relation_learning/analysis/proxy_ceiling/targets_evaluation_only.csv'))
    cv = pd.read_csv(_public_path('experiments/archive/relation_learning/analysis/proxy_ceiling/cv_results.csv'))
    oof = pd.read_csv(_public_path('experiments/archive/relation_learning/analysis/proxy_ceiling/oof_predictions.csv'))

    assert summary["status"] == "PROXY_CEILING_DONE"
    assert summary["samples"] == {
        "cdi_rows_total": 504,
        "official_v4_fit_subset": 225,
        "valid_child_aoa": 406,
    }
    assert summary["scan_audit"]["rows"] == 647_400
    assert summary["scan_audit"]["words_from_metadata"] == 100_000_000
    assert len(summary["scan_audit"]["stream_sha256"]) == 64

    assert len(features) == 504 and len(targets) == 406
    assert features["word"].is_unique and targets["word"].is_unique
    assert set(targets["word"]).issubset(features["word"])
    assert "child_aoa_month" not in features.columns
    assert "official_v4_fit_ok" not in features.columns
    assert targets["child_aoa_month"].notna().sum() == 406
    assert targets["official_v4_fit_ok"].sum() == 225

    assert len(summary["feature_sets"]["all_legal"]) == 60
    assert len(summary["feature_sets"]["compact_legal"]) == 30
    assert set(summary["feature_sets"]["all_legal"]).issubset(features.columns)
    assert len(cv) == 24  # 11 estimators plus one ensemble on two datasets
    assert set(cv["n"]) == {225, 406}
    assert cv.select_dtypes(include="number").apply(lambda s: s.map(math.isfinite).all()).all()
    assert len(oof) == 631 and set(oof["dataset"]) == {
        "all_child_valid",
        "official_v4_fit_subset",
    }
    assert oof.drop(columns=["dataset", "word"]).apply(lambda s: s.map(math.isfinite).all()).all()

    count_audit = summary["count_audit"]
    assert count_audit["exact_match_words"] == 501
    assert count_audit["speaker_tag_collision_words"] == ["ant", "mad", "man"]
    assert count_audit["unexpected_mismatch_words"] == []
    timing = summary["actual_order_timing_audit"]
    assert timing["valid_words_seen"] == 406
    close(timing["early_10m_fraction_unique_seen"][0], 0.1)
    close(timing["early_20m_fraction_unique_seen"][0], 0.2)
    close(timing["early_50m_fraction_unique_seen"][0], 0.5)
    close(timing["last_10m_fraction_unique_seen"][0], 0.1)

    primary = summary["primary_ceiling"]
    exploratory = summary["exploratory_best"]
    close(primary["oof_pearson_r"], 0.4011123553787616)
    close(primary["oof_spearman_r"], 0.4050725335757854)
    close(exploratory["oof_pearson_r"], 0.5222310277663941)
    close(summary["official_subset_exploratory_best"]["oof_pearson_r"], 0.5370430914911914)

    translation = summary["translation_to_stage3"]
    close(
        translation["positive_r_boundary_n225_p0p1"] - translation["v4_raw_r"],
        translation["required_raw_movement"],
    )
    close(
        translation["full_mask_family_spread"]
        * translation["primary_proxy_r_div_childes_enrichment_r"],
        translation["heuristic_scaled_movement_from_full_family_spread"],
    )
    assert translation["heuristic_fraction_of_required"] < 0.32
    assert translation["exploratory_heuristic_fraction_of_required"] < 0.42

    # Artifact paths should remain relative to the repository root.
    summary_md = (_public_path('research/notes/relation_learning/analysis/proxy_ceiling/summary.md')).read_text()
    assert "/Sessions/" not in summary_md
    assert "| model | feature set | features |" in summary_md
    print(json.dumps({"status": "OUTPUTS_VERIFIED", "checks": 35}, indent=2))


if __name__ == "__main__":
    main()
