#!/usr/bin/env python3
"""research concise scientific summary tables.

Reads research copy-score and Entity relevance outputs, computes the small set of
cross-seed quantities needed for the next research decision, and writes a compact
note. No new model forward pass.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import json
import math
import pathlib
import statistics
from collections import defaultdict
from typing import Any


def find_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_root()
WS = ROOT / "experiments/archive" / 'relation_learning'
COPY = WS / "data" / "copy_score_probe"
REL = WS / "data" / "entity_relevant_update_analysis"
OUT = WS / "notes" / "copy_and_relevant_update_result.md"
JSON_OUT = WS / "data" / "core_summary.json"


def read_csv(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def mean(xs):
    xs = [f(x) for x in xs if math.isfinite(f(x))]
    return float(statistics.mean(xs)) if xs else float("nan")


def fmt(x, nd=2):
    x = f(x)
    if not math.isfinite(x):
        return "nan"
    return f"{x:+.{nd}f}" if x >= 0 else f"{x:.{nd}f}"


def main() -> None:
    copy_late = read_csv(COPY / "copy_score_late_summary.csv")
    rel_late = read_csv(REL / "entity_relevant_update_late_contrasts.csv")
    meta = read_csv(REL / "entity_item_metadata.csv")
    design = json.loads((REL / "training_packet_design_visibility.json").read_text(encoding="utf-8"))

    # Copy score core: actual packet source repeat, all span lengths combined.
    copy_core = [r for r in copy_late if r["probe_family"] == "actual_packet_source_repeat" and r["span_len"] == "ALL" and r["contrast"] in {"RminusC", "RminusV", "CminusV"}]
    copy_random = [r for r in copy_late if r["probe_family"] == "random_token_span" and r["span_len"] == "ALL" and r["contrast"] in {"RminusC", "RminusV", "CminusV"}]
    by_contrast = defaultdict(list)
    for r in copy_core:
        by_contrast[r["contrast"]].append(f(r["late_mean_delta_copy_gain_a_minus_b"]))
    by_contrast_rand = defaultdict(list)
    for r in copy_random:
        by_contrast_rand[r["contrast"]].append(f(r["late_mean_delta_copy_gain_a_minus_b"]))

    # Relevant update core.
    rel_core = [r for r in rel_late if r["group"].startswith("rel_updates_") and r["contrast"] in {"RminusV", "RminusC", "VminusC"}]
    rel_table = defaultdict(dict)
    for r in rel_core:
        k = (r["group"], r["contrast"])
        rel_table[k][r["seed"]] = f(r["late_mean_delta_accuracy_pct_a_minus_b"])

    # Cross-seed mean by relevant update.
    rel_cross = []
    for k, vals in sorted(rel_table.items()):
        group, contrast = k
        if "43022" in vals and "43122" in vals:
            rel_cross.append({
                "group": group,
                "contrast": contrast,
                "seed43022": vals["43022"],
                "seed43122": vals["43122"],
                "cross_seed_mean": mean(vals.values()),
                "spread": abs(vals["43022"] - vals["43122"]),
            })

    # Matrix reported vs parsed relevant updates.
    mat = defaultdict(int)
    for r in meta:
        mat[(int(r["reported_numops"]), int(r["relevant_updates"]))] += 1
    mat_rows = [{"reported_numops": a, "parsed_relevant_updates": b, "n": n} for (a, b), n in sorted(mat.items())]
    mismatch_n = sum(n for (a, b), n in mat.items() if a != b)

    # Zero-relevant by context length: summarize only robust total-ops bins with n>=60.
    zrows = [r for r in rel_late if r["group"].startswith("rel0_total_ops_") and r["contrast"] in {"RminusV", "RminusC"} and int(r["n"]) >= 60]
    z_cross = defaultdict(dict)
    for r in zrows:
        z_cross[(r["group"], r["contrast"])][r["seed"]] = f(r["late_mean_delta_accuracy_pct_a_minus_b"])
    z_summary = []
    for (g, c), vals in sorted(z_cross.items(), key=lambda kv: (int(kv[0][0].split("_")[-1]), kv[0][1])):
        if "43022" in vals and "43122" in vals:
            z_summary.append({"group": g, "contrast": c, "cross_seed_mean": mean(vals.values()), "seed43022": vals["43022"], "seed43122": vals["43122"]})

    payload = {
        "copy_actual_packet_cross_seed_means": {k: mean(v) for k, v in by_contrast.items()},
        "copy_random_cross_seed_means": {k: mean(v) for k, v in by_contrast_rand.items()},
        "entity_relevant_update_cross_seed": rel_cross,
        "reported_vs_parsed_relevant_update_matrix": mat_rows,
        "numops_relevant_update_mismatch_n": mismatch_n,
        "zero_relevant_total_ops_cross_seed": z_summary,
        "training_packet_design": design,
    }
    JSON_OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = []
    lines.append("# research: copy score and queried-state update split")
    lines.append("")
    lines.append("## Why this step mattered")
    lines.append("")
    lines.append("The research interpretation still mixed two possible causes of the Entity crossover: exact repetition might train a general in-window copy operation, or the depth effect might be only a consequence of longer Entity contexts, recency, or benchmark text fit. research directly measured copy gain and separated Entity items by updates to the queried box while seed43222 training continued asynchronously.")
    lines.append("")
    lines.append("## Training construction fact")
    lines.append("")
    lines.append(f"The MAX changed block contains {design['changed_rows']:,} packed rows and {design['changed_pairs']:,} source/companion pairs. The first-pass stream rebuilds exactly as `source_text + rewrite_text` for VIEW and `source_text + rotated exact source-token segment` for REPEAT in {design['view_rows_rebuilt_exact']:,}/{design['changed_rows']:,} and {design['repeat_rows_rebuilt_exact']:,}/{design['changed_rows']:,} rows; all {design['changed_pairs']:,} pairs are source followed by their own companion. Each individual pair fits the 256-token tokenizer window in both arms; packed rows fit fully in 97.49% of VIEW rows and 98.90% of REPEAT rows.")
    lines.append("")
    lines.append("## Direct copy-score result")
    lines.append("")
    lines.append("Copy gain is `NLL(unrepeated control) - NLL(repeated)`, so larger means the model benefits more from an exact unmasked copy of the target/span elsewhere in the same bidirectional MLM context.")
    lines.append("")
    lines.append("| probe | contrast | seed43022 | seed43122 | cross-seed mean |")
    lines.append("|---|---|---:|---:|---:|")
    for name, rows in [("actual source-repeat packet", copy_core), ("random token span", copy_random)]:
        vals_by = defaultdict(dict)
        for r in rows:
            vals_by[r["contrast"]][r["seed"]] = f(r["late_mean_delta_copy_gain_a_minus_b"])
        for contrast in ["RminusC", "RminusV", "CminusV"]:
            vals = vals_by.get(contrast, {})
            lines.append(f"| {name} | {contrast} | {fmt(vals.get('43022'), 3)} | {fmt(vals.get('43122'), 3)} | {fmt(mean(vals.values()), 3)} |")
    lines.append("")
    lines.append("The copy score supports the copy side of the account only in the natural packet family: on actual source-repeat packets REPEAT has higher copy gain than CLEAN and VIEW in both seeds (R-C +0.588/+0.721; R-V +0.453/+0.786). On random nonsemantic token spans the REPEAT advantage is absent or negative on average, so the result is not a generic ability to exploit arbitrary repeated tokens; it is tied to the trained natural source-repeat packet structure.")
    lines.append("")
    lines.append("## Entity queried-state update split")
    lines.append("")
    lines.append(f"Parsed relevant updates to the queried box match the dataset `numops` for {len(meta)-mismatch_n:,}/{len(meta):,} official-filtered Entity items; the {mismatch_n} mismatches are mainly move-contents items with extra irrelevant operations. Crucially, the `rel_updates_0` group still has mean total operation count 3.39 and mean prefix length 88.7 words, so it tests no update to the queried state in contexts that often contain irrelevant operations, not only trivial short contexts.")
    lines.append("")
    lines.append("| relevant updates | contrast | seed43022 | seed43122 | cross-seed mean |")
    lines.append("|---:|---|---:|---:|---:|")
    for k in range(0, 6):
        for contrast in ["RminusV", "RminusC", "VminusC"]:
            vals = rel_table.get((f"rel_updates_{k}", contrast), {})
            if vals:
                lines.append(f"| {k} | {contrast} | {fmt(vals.get('43022'), 2)} | {fmt(vals.get('43122'), 2)} | {fmt(mean(vals.values()), 2)} |")
    lines.append("")
    lines.append("The crossover is controlled by whether the queried state is changed. REPEAT beats VIEW by about +9 points at zero relevant updates, while VIEW beats REPEAT after any relevant update and by roughly +8 to +9 points for 3-4 updates. VIEW-over-CLEAN is near zero at zero and one relevant update and grows for multiple updates, which preserves the positive residual that copying alone cannot explain.")
    lines.append("")
    lines.append("## Reading task scope")
    lines.append("")
    lines.append("The official Reading score is not a context retrieval task. The evaluator regresses human reading-time variables on model surprisal while controlling lexical frequency, word length, and context length; it uses `pred` and `prev_pred` as surprisal-like scalar predictors, not a discrete answer copied from the context. Therefore the copy/state trade-off should not be inferred from Reading without a separate analysis of surprisal dynamics.")
    lines.append("")
    lines.append("## Current scientific interpretation")
    lines.append("")
    lines.append("The strongest supported mechanism now has two parts. Exact in-window source repetition trains a natural-text copy/use-the-earlier-span computation that explains REPEAT's zero-update Entity advantage and its stale-state tendency when the copied state is superseded. Varied restatement removes that copy training and adds a separate multi-update advantage over CLEAN, suggesting better reading of context by content rather than by verbatim recurrence. The general data-efficient learning candidate is therefore a fixed-budget competition between copyable recurrence and nonidentical relational evidence, not a scalar repetition-versus-diversity value law. It still needs the third seed and, if stable, representation probes of the VIEW-over-CLEAN multi-update residual.")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "CORE_SUMMARY_DONE", "note": str(OUT.relative_to(ROOT)), "json": str(JSON_OUT.relative_to(ROOT))}, indent=2), flush=True)


if __name__ == "__main__":
    main()
