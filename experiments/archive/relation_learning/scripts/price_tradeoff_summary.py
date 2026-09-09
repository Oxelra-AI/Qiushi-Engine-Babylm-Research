#!/usr/bin/env python3
"""research: price-and-purchase decomposition for in-window relation practice.

Uses the seed43022 integrated VIEW/REPEAT/VIEW_SPLIT/REPEAT_SPLIT terms to express
local co-occurrence as a fixed-budget trade-off: broad unrelated-source fit cost
versus source-specific competence gained/lost by having the companion in the same
window.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import pathlib
import time
from typing import Any

import pandas as pd

ROOT = _public_path('experiments/archive/relation_learning/scripts/price_tradeoff_summary.py')
ROOT = _PUBLIC_ROOT

WS = _public_path('experiments/archive/relation_learning')
IN_DIR = _public_path('experiments/archive/relation_learning/data/view_split_integration')
OUT_DIR = _public_path('experiments/archive/relation_learning/data/price_tradeoff')
NOTE = _public_path('research/notes/relation_learning/price_tradeoff.md')


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_terms() -> pd.DataFrame:
    p = _public_path('experiments/archive/relation_learning/data/view_split_integration/rewrite_late_all_roles_seed43022.csv')
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    return df


def rowmap(df: pd.DataFrame) -> dict[tuple[str, str], dict[str, float]]:
    out: dict[tuple[str, str], dict[str, float]] = {}
    for _, r in df.iterrows():
        out[(str(r["role"]), str(r["token_class"]))] = {
            "gain": float(r["mean_gain"]),
            "T": float(r["true_source_nll"]),
            "U": float(r["unrelated_source_nll"]),
            "n": float(r["n_min"]),
        }
    return out


def compute_for(token_class: str, terms: dict[tuple[str, str], dict[str, float]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pairs = [
        ("exact_recurrence", "R", "RS"),
        ("restatement", "V", "VS"),
    ]
    for label, local, split in pairs:
        a = terms[(local, token_class)]
        b = terms[(split, token_class)]
        # local - split terms: negative U means local in-window relation spent broad U fit;
        # gain delta tells whether that spending bought source-specific use.
        dT = a["T"] - b["T"]
        dU = a["U"] - b["U"]
        dG = a["gain"] - b["gain"]
        rows.append({
            "token_class": token_class,
            "relation": label,
            "local_role": local,
            "split_role": split,
            "n_min": int(a["n"]),
            "local_gain_U_minus_T": a["gain"],
            "split_gain_U_minus_T": b["gain"],
            "local_minus_split_gain_delta": dG,
            "local_minus_split_true_source_nll_delta": dT,
            "local_minus_split_unrelated_source_nll_delta": dU,
            "broad_U_fit_price_local_minus_split": dU,
            "source_specific_purchase_gain_delta": dG,
            "extra_true_source_effect_beyond_U": dT - dU,
        })
    # mixture predictions for a hash 50/50 local arm, using original local R/V endpoints vs CLEAN.
    c = terms[("C", token_class)]
    r = terms[("R", token_class)]
    v = terms[("V", token_class)]
    avg = {k: 0.5 * (r[k] + v[k]) for k in ["gain", "T", "U"]}
    rows.append({
        "token_class": token_class,
        "relation": "hash_half_exact_half_restatement_pred_vs_clean",
        "local_role": "0.5R+0.5V",
        "split_role": "C",
        "n_min": int(c["n"]),
        "local_gain_U_minus_T": avg["gain"],
        "split_gain_U_minus_T": c["gain"],
        "local_minus_split_gain_delta": avg["gain"] - c["gain"],
        "local_minus_split_true_source_nll_delta": avg["T"] - c["T"],
        "local_minus_split_unrelated_source_nll_delta": avg["U"] - c["U"],
        "broad_U_fit_price_local_minus_split": avg["U"] - c["U"],
        "source_specific_purchase_gain_delta": avg["gain"] - c["gain"],
        "extra_true_source_effect_beyond_U": (avg["T"] - c["T"]) - (avg["U"] - c["U"]),
    })
    return pd.DataFrame(rows)


def fmt(x: float) -> str:
    return f"{x:+.4f}"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_terms()
    terms = rowmap(df)
    all_rows = pd.concat([compute_for("nonoverlap", terms), compute_for("overlap", terms)], ignore_index=True)
    out_csv = _public_path('experiments/archive/relation_learning/data/price_tradeoff/price_tradeoff_seed43022.csv')
    all_rows.to_csv(out_csv, index=False)

    # Key seed43022 decomposition focused on token-nonoverlap targets.
    key = all_rows[(all_rows["token_class"] == "nonoverlap") & (all_rows["relation"].isin(["exact_recurrence", "restatement"]))].copy()
    pred = all_rows[(all_rows["token_class"] == "nonoverlap") & (all_rows["relation"].str.startswith("hash_half"))].iloc[0]

    lines: list[str] = []
    lines.append("# research price-and-purchase decomposition")
    lines.append("")
    lines.append("This note rewrites the seed43022 relation-practice 2×2 in the fixed-budget language inherited from frontier_consolidation. The split arms preserve the same selected text and 100M budget while removing paired same-window co-occurrence. Therefore local-minus-split isolates what same-window pairing spends and buys on the held-out compact-rewrite probe.")
    lines.append("")
    lines.append("For token-nonoverlap targets, the terms are:")
    lines.append("")
    lines.append("| local relation | local vs split gain Δ | true-source NLL Δ | unrelated-source NLL Δ | reading |")
    lines.append("|---|---:|---:|---:|---|")
    for _, r in key.iterrows():
        if r["relation"] == "exact_recurrence":
            reading = "local exact recurrence spends broad fit and buys an identity readout that is harmful for nonidentical targets"
        else:
            reading = "local restatement spends broad fit and buys content-conditioned use of the related source"
        lines.append(
            f"| {r['relation']} ({r['local_role']}−{r['split_role']}) | {fmt(r['local_minus_split_gain_delta'])} | {fmt(r['local_minus_split_true_source_nll_delta'])} | {fmt(r['local_minus_split_unrelated_source_nll_delta'])} | {reading} |"
        )
    lines.append("")
    lines.append("The unrelated-source term is the broad target-fit term under a source that should not help the target. It is lower for the split arms than the local arms by about a third of a nat in both branches: original REPEAT is worse than REPEAT_SPLIT by +0.2745 nats on U, and original VIEW is worse than VIEW_SPLIT by +0.3269 nats on U. The symmetry matters: same-window pairing is not simply 'bad repetition' or 'good restatement'. It spends residual prediction work in both cases because part of the target-relevant answer is locally visible during training.")
    lines.append("")
    lines.append("What differs is the computation bought by that spending. Exact recurrence changes gain by −0.7204 nats relative to its split form; restatement changes gain by +0.6539 nats relative to its split form. Thus the local relation determines the sign of the source-specific competence purchased with roughly the same broad-fit price.")
    lines.append("")
    lines.append("## Hash-mixture arm prediction")
    lines.append("")
    lines.append("A 50/50 hash mixture of exact local recurrence and local restatement, with every source retained and companion type assigned per pair, distinguishes three laws:")
    lines.append("")
    lines.append(f"- Proportional mixture: token-nonoverlap gain versus CLEAN should be near the average of original R−C and V−C, {fmt(pred['local_minus_split_gain_delta'])} nats, i.e. close to zero compared with the original endpoints.")
    lines.append("- Identity-dominant capture: a half admixture of exact local copies pulls the arm near original REPEAT, implying that near-duplicates inside a context window can disproportionately set the readout.")
    lines.append("- Restatement-robust content use: the arm stays near original VIEW, implying that content-conditioned reading survives substantial identity admixture.")
    lines.append("")
    lines.append("This is a cheap graded test of the price-and-purchase form because it keeps the source set, budget, and row-local availability fixed while changing the per-pair relation composition by hash.")
    lines.append("")
    lines.append(f"Data table: `{rel(out_csv)}`")
    _public_path('research/notes/relation_learning').mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        "status": "PRICE_TRADEOFF_DONE",
        "created_utc": now(),
        "note": rel(NOTE),
        "table": rel(out_csv),
        "nonoverlap_exact_RminusRS_gain_delta": float(key[key["relation"] == "exact_recurrence"]["local_minus_split_gain_delta"].iloc[0]),
        "nonoverlap_exact_RminusRS_U_delta": float(key[key["relation"] == "exact_recurrence"]["local_minus_split_unrelated_source_nll_delta"].iloc[0]),
        "nonoverlap_restate_VminusVS_gain_delta": float(key[key["relation"] == "restatement"]["local_minus_split_gain_delta"].iloc[0]),
        "nonoverlap_restate_VminusVS_U_delta": float(key[key["relation"] == "restatement"]["local_minus_split_unrelated_source_nll_delta"].iloc[0]),
        "hash_half_pred_gain_vs_clean_nonoverlap": float(pred["local_minus_split_gain_delta"]),
    }
    (_public_path('experiments/archive/relation_learning/data/price_tradeoff/price_tradeoff_summary.json')).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
