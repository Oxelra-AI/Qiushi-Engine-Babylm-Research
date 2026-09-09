#!/usr/bin/env python3
"""research: recompute research context-isolation contrasts after removing spans where
row-context scoring truncates the target span or masks the whole target.
"""
from __future__ import annotations

import json
import math
import pathlib
import time
from typing import Any

import pandas as pd

ROOT = pathlib.Path(".").resolve()
DATA = ROOT / "experiments/archive/relation_learning/data/context_isolation_screen_all100M"
OUT = ROOT / "experiments/archive/relation_learning/data/context_isolation_filtered"
SCORES = DATA / "context_isolation_scores.csv"
SELECTED = [
    ("compact_experience", 43022, "chck_100M", "ALN", "OFF"),
    ("compact_experience", 43122, "chck_100M", "ALN", "OFF"),
    ("compact_experience", 43022, "chck_100M", "SHUF", "OFF"),
    ("compact_experience", 43122, "chck_100M", "SHUF", "OFF"),
    ("compact_experience", 43022, "chck_100M", "ALN", "SHUF"),
    ("compact_experience", 43122, "chck_100M", "ALN", "SHUF"),
    ("compact_experience", 43022, "chck_100M", "DUP", "OFF"),
    ("compact_experience", 43022, "chck_100M", "SEP", "OFF"),
    ("dose", 43022, "chck_100M", "dose21", "base0"),
    ("dose", 43122, "chck_100M", "dose21", "base0"),
    ("dose", 43022, "chck_100M", "dose25", "base0"),
    ("dose", 43122, "chck_100M", "dose25", "base0"),
    ("crv", 43022, "chck_100M", "REPEAT", "CLEAN"),
    ("crv", 43122, "chck_100M", "REPEAT", "CLEAN"),
    ("crv", 43022, "chck_100M", "VIEW", "CLEAN"),
    ("crv", 43122, "chck_100M", "VIEW", "CLEAN"),
]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path | str) -> str:
    pp = pathlib.Path(p)
    try:
        return str(pp.resolve().relative_to(ROOT))
    except Exception:
        return str(pp)


def se(xs) -> float:
    s = pd.Series(xs).dropna()
    if len(s) <= 1:
        return float("nan")
    return float(s.std(ddof=1) / math.sqrt(len(s)))


def contrast(piv: pd.DataFrame, family: str, seed: int, ck: str, a: str, b: str) -> dict[str, Any] | None:
    sub = piv[(piv.family == family) & (piv.seed == seed) & (piv.checkpoint == ck) & (piv.relation.isin([a, b]))]
    if sub.empty:
        return None
    wide = sub.pivot_table(index="sentence_uid", columns="relation", values=["row_context", "isolation", "context_gain"], aggfunc="mean")
    need = [("row_context", a), ("row_context", b), ("isolation", a), ("isolation", b), ("context_gain", a), ("context_gain", b)]
    for n in need:
        if n not in wide.columns:
            return None
    wide = wide.dropna(subset=need)
    if wide.empty:
        return None
    d_gain = wide[("context_gain", a)] - wide[("context_gain", b)]
    d_row = wide[("row_context", a)] - wide[("row_context", b)]
    d_iso = wide[("isolation", a)] - wide[("isolation", b)]
    return {
        "family": family,
        "seed": seed,
        "checkpoint": ck,
        "contrast": f"{a}-minus-{b}",
        "n_sentences_filtered": int(len(wide)),
        "delta_row_context_loss": float(d_row.mean()),
        "delta_isolation_loss": float(d_iso.mean()),
        "delta_context_gain": float(d_gain.mean()),
        "se_delta_context_gain": se(d_gain),
        "fraction_a_higher_context_gain": float((d_gain > 0).mean()),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SCORES)
    for col in ["target_truncated_right"]:
        df[col] = df[col].astype(str).str.lower().isin(["true", "1"])
    # Remove only rows whose scored target span is incomplete under row_context and
    # the single degenerate case where every target token is masked.  Isolation rows
    # are kept for a sentence only if the matching row_context row remains.
    row_bad = df[(df["mode"] == "row_context") & (df["target_truncated_right"] | (df["n_masked"] >= df["n_target_tokens"]))]
    bad_keys = set(zip(row_bad["family"], row_bad["seed"], row_bad["relation"], row_bad["arm"], row_bad["checkpoint"], row_bad["sentence_uid"]))
    def good(row):
        return (row.family, row.seed, row.relation, row.arm, row.checkpoint, row.sentence_uid) not in bad_keys
    filt = df[df.apply(good, axis=1)].copy()
    # Require row_context and isolation for a sentence/arm after filtering.
    piv = filt.pivot_table(index=["family", "seed", "relation", "arm", "checkpoint", "sentence_uid"], columns="mode", values="loss", aggfunc="mean").reset_index()
    piv = piv.dropna(subset=["row_context", "isolation"])
    piv["context_gain"] = piv["isolation"] - piv["row_context"]
    rows = [r for args in SELECTED for r in [contrast(piv, *args)] if r is not None]
    pd.DataFrame(rows).to_csv(OUT / "filtered_selected_contrasts.csv", index=False)
    summary = {
        "status": "CONTEXT_ISOLATION_FILTERED_CONTRASTS",
        "created_utc": now(),
        "source_scores": rel(SCORES),
        "removed_row_context_records": int(len(row_bad)),
        "removed_unique_arm_sentence_keys": int(len(bad_keys)),
        "remaining_mode_rows": int(len(filt)),
        "remaining_paired_sentence_arm_rows": int(len(piv)),
        "selected_contrasts": rows,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research filtered context-isolation contrasts", "", f"Removed row-context scored records with target truncation or whole-target masking: {len(row_bad)}.", "", "| family | seed | contrast | n | d_row | d_iso | d_gain | se |", "|---|---:|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['family']} | {r['seed']} | {r['contrast']} | {r['n_sentences_filtered']} | {r['delta_row_context_loss']:+.4f} | {r['delta_isolation_loss']:+.4f} | {r['delta_context_gain']:+.4f} | {r['se_delta_context_gain']:.4f} |")
    ((ROOT / 'research/documents/relation_learning/data/context_isolation_filtered/summary.md')).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "summary": rel(OUT / "summary.json"), "removed": len(row_bad), "selected_contrasts": rows[:8]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
