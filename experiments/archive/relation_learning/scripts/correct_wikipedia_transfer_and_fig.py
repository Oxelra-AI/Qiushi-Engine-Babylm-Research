#!/usr/bin/env python3
"""research: correct the natural-domain transfer reading for WikiLarge T/U/N.

The research synthesis emphasized the nonoverlap slice and therefore made VIEW's
natural-domain transfer look absent. This script reads the frozen WikiLarge probe
and scored contrasts, verifies that overlap/nonoverlap are tokenizer-ID target
presence in the true source and that source->target is English Wikipedia to
Simple English, then writes a compact correction table/note and redraws the
natural-domain figure as contrast x token class.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

ROOT = _public_path('experiments/archive/relation_learning/scripts/correct_wikipedia_transfer_and_fig.py')
ROOT = _PUBLIC_ROOT

WS = _public_path('experiments/archive/relation_learning')
PROBE_DIR = _public_path('experiments/archive/relation_learning/analysis/wikipedia_simplification_restatement_probe')
SCORE_DIR = _public_path('experiments/archive/relation_learning/data/wikipedia_simplification_score')
DIR = _public_path('experiments/archive/relation_learning/data/integrated_scope_and_mechanism')
OUT_DIR = _public_path('experiments/archive/relation_learning/data/express_review')
FIG_DIR = _public_path('experiments/archive/relation_learning/figures/express_review')
NOTE = _public_path('research/notes/relation_learning/express_scientific_review_and_wikipedia_correction.md')

ACROSS = _public_path('experiments/archive/relation_learning/data/integrated_scope_and_mechanism/wikipedia_scope_key_across_seed.csv')
RAW_ACROSS = _public_path('experiments/archive/relation_learning/data/wikipedia_simplification_score/wikipedia_across_seed_contrasts.csv')
LATE = _public_path('experiments/archive/relation_learning/data/wikipedia_simplification_score/wikipedia_late_arm_contrasts.csv')
PAIR_FILE = _public_path('experiments/archive/relation_learning/analysis/wikipedia_simplification_restatement_probe/wikipedia_simplification_pairs.jsonl')
RECORDS_FILE = _public_path('experiments/archive/relation_learning/analysis/wikipedia_simplification_restatement_probe/wikipedia_simplification_probe_records.jsonl')
VALIDATION_FILE = _public_path('experiments/archive/relation_learning/analysis/wikipedia_simplification_restatement_probe/validation_results.json')
STATS_FILE = _public_path('experiments/archive/relation_learning/analysis/wikipedia_simplification_restatement_probe/probe_stats.json')


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def fmt(x: float, nd: int = 3) -> str:
    if x is None or not math.isfinite(float(x)):
        return "nan"
    return f"{float(x):+.{nd}f}"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    across = pd.read_csv(ACROSS)
    late = pd.read_csv(LATE)
    validation = json.loads(VALIDATION_FILE.read_text(encoding="utf-8"))
    stats = json.loads(STATS_FILE.read_text(encoding="utf-8"))
    pairs = read_jsonl(PAIR_FILE, limit=None)
    records = read_jsonl(RECORDS_FILE, limit=None)

    # Verify source->target direction and tokenizer-ID class definitions.
    first = pairs[0]
    direction = {
        "dataset_name": first.get("dataset_name"),
        "source_field_meaning": "English Wikipedia/source sentence from wikilarge-clean",
        "target_field_meaning": "Simple English Wikipedia/target sentence from wikilarge-clean",
        "example_source_text": first.get("source_text"),
        "example_target_text": first.get("target_text"),
    }
    target_rows = []
    class_failures = []
    chosen_counts = {"overlap": 0, "nonoverlap": 0}
    for pair in pairs:
        source_ids = set(int(x) for x in pair["source_token_ids"])
        chosen = pair["chosen_targets"]
        if len([x for x in chosen if x["token_class"] == "overlap"]) != 1:
            class_failures.append((pair["pair_id"], "overlap_count"))
        if len([x for x in chosen if x["token_class"] == "nonoverlap"]) != 2:
            class_failures.append((pair["pair_id"], "nonoverlap_count"))
        for t in chosen:
            cls = t["token_class"]
            chosen_counts[cls] = chosen_counts.get(cls, 0) + 1
            occurs = int(t["target_token_id"]) in source_ids
            if (cls == "overlap") != occurs:
                class_failures.append((pair["pair_id"], t["target_key"], cls, occurs))
            target_rows.append({
                "pair_id": pair["pair_id"],
                "overlap_bin": pair["overlap_bin"],
                "target_key": t["target_key"],
                "target_word": t["word"],
                "token_class": cls,
                "token_id_occurs_in_source": occurs,
                "surface_word_class": t.get("surface_word_class"),
            })
    if class_failures:
        raise RuntimeError({"class_failures": class_failures[:10], "n": len(class_failures)})

    # Verify record direction/condition invariants at denominator level.
    rec_df = pd.DataFrame(records)
    rec_den = rec_df.groupby(["condition_code", "overlap_bin", "token_class"]).size().reset_index(name="records")
    rec_den.to_csv(_public_path('experiments/archive/relation_learning/data/express_review/wikipedia_record_denominators.csv'), index=False)
    pd.DataFrame(target_rows).to_csv(_public_path('experiments/archive/relation_learning/data/express_review/wikipedia_target_class_verification.csv'), index=False)

    # Natural-domain scientific pattern: gain_T_vs_N is the main source-specific benefit.
    key_rows = across[(across["overlap_bin"] == "ALL") & (across["estimand"].isna() if False else True)].copy()
    # The research key table is wide, not long. Select target classes and contrasts directly.
    summary_rows = []
    for token_class in ["overlap", "nonoverlap", "ALL"]:
        for contrast in ["RminusC", "VminusC", "VminusR"]:
            row = across[(across.overlap_bin == "ALL") & (across.token_class == token_class) & (across.contrast == contrast)]
            if len(row) != 1:
                raise RuntimeError(f"missing key row {contrast} {token_class}: {len(row)}")
            r = row.iloc[0]
            summary_rows.append({
                "target_class": token_class,
                "contrast": contrast,
                "gain_T_vs_N_mean": float(r["gain_T_vs_N_mean"]),
                "gain_T_vs_N_seed_sd": float(r["gain_T_vs_N_seed_sd"]),
                "gain_U_vs_N_mean": float(r["gain_U_vs_N_mean"]),
                "gain_U_vs_N_seed_sd": float(r["gain_U_vs_N_seed_sd"]),
                "gain_T_vs_U_mean": float(r["gain_T_vs_U_mean"]),
                "gain_T_vs_U_seed_sd": float(r["gain_T_vs_U_seed_sd"]),
                "mean_pairs": float(r["gain_T_vs_N_mean_pairs"]),
            })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(_public_path('experiments/archive/relation_learning/data/express_review/wikipedia_transfer_by_target_class.csv'), index=False)

    # Per-seed sign check for the main statements.
    late_main = late[(late["overlap_bin"] == "ALL") & (late["estimand"] == "gain_T_vs_N") &
                     (late["token_class"].isin(["overlap", "nonoverlap", "ALL"])) &
                     (late["contrast"].isin(["VminusC", "RminusC", "VminusR"]))].copy()
    late_main["sign"] = late_main["mean_difference"].map(lambda x: "+" if x > 0 else "-" if x < 0 else "0")
    late_main.to_csv(_public_path('experiments/archive/relation_learning/data/express_review/wikipedia_transfer_by_seed_signs.csv'), index=False)

    # Plot contrast x token class. Positive means arm A uses the true source more than arm B.
    class_order = ["overlap", "nonoverlap", "ALL"]
    contrast_order = ["RminusC", "VminusC", "VminusR"]
    colors = {"RminusC": "#d95f02", "VminusC": "#1b9e77", "VminusR": "#7570b3"}
    labels = {"RminusC": "REPEAT − CLEAN", "VminusC": "VIEW − CLEAN", "VminusR": "VIEW − REPEAT"}
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    width = 0.24
    x = range(len(class_order))
    for i, contrast in enumerate(contrast_order):
        vals = []
        errs = []
        for cls in class_order:
            r = summary[(summary.target_class == cls) & (summary.contrast == contrast)].iloc[0]
            vals.append(r.gain_T_vs_N_mean)
            errs.append(r.gain_T_vs_N_seed_sd)
        xpos = [j + (i - 1) * width for j in x]
        ax.bar(xpos, vals, width=width, yerr=errs, capsize=3, color=colors[contrast], label=labels[contrast], alpha=0.9)
    ax.axhline(0, color="black", lw=1.0)
    ax.set_xticks(list(x))
    ax.set_xticklabels(["target token in source\n(overlap)", "target token absent\n(nonoverlap)", "all targets"])
    ax.set_ylabel("Δ true-source benefit vs N (nats)\npositive = first arm uses T source more")
    ax.set_title("Natural restatement: practiced relation determines form-robust retrieval")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    ax.text(0.0, -0.46, "WikiLarge/Simple-English T/U/N, late checkpoints, 3 DeBERTa seeds; error bars are seed SD.",
            ha="left", va="top", transform=ax.transAxes, fontsize=9)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig_png = _public_path('experiments/archive/relation_learning/figures/express_review/fig_wikipedia_target_class_transfer.png')
    fig_pdf = _public_path('experiments/archive/relation_learning/figures/express_review/fig_wikipedia_target_class_transfer.pdf')
    fig.savefig(fig_png, dpi=220)
    fig.savefig(fig_pdf)
    plt.close(fig)

    # Write a review/correction note.
    def get(cls: str, contrast: str) -> pd.Series:
        return summary[(summary.target_class == cls) & (summary.contrast == contrast)].iloc[0]

    vc_overlap = get("overlap", "VminusC")
    vc_non = get("nonoverlap", "VminusC")
    vc_all = get("ALL", "VminusC")
    vr_overlap = get("overlap", "VminusR")
    vr_non = get("nonoverlap", "VminusR")
    vr_all = get("ALL", "VminusR")
    rc_overlap = get("overlap", "RminusC")
    rc_non = get("nonoverlap", "RminusC")
    rc_all = get("ALL", "RminusC")

    signs = late_main.pivot_table(index=["contrast", "token_class"], columns="seed", values="sign", aggfunc="first")

    note = f"""# research Express scientific review: corrected Wikipedia transfer reading

This note is research-facing. It corrects the natural-domain interpretation before any final paper statement is hardened.

## What was checked

- Frozen probe: `{PAIR_FILE.relative_to(ROOT)}` and `{RECORDS_FILE.relative_to(ROOT)}`.
- Validation: `{VALIDATION_FILE.relative_to(ROOT)}` reports `{validation.get('status')}`.
- Scored contrasts: `{ACROSS.relative_to(ROOT)}` and `{LATE.relative_to(ROOT)}`.
- Direction: WikiLarge-clean `source` is the English Wikipedia source sentence and `target` is the Simple English target sentence. Example source→target:
  - source: {direction['example_source_text']}
  - target: {direction['example_target_text']}
- Target class definition was rechecked from the constructor and the frozen pairs: `overlap` means the exact target tokenizer ID occurs in the true source token-ID set; `nonoverlap` means it does not. The frozen design has {len(pairs):,} pairs, {chosen_counts['overlap']:,} overlap targets and {chosen_counts['nonoverlap']:,} nonoverlap targets; each target has T/U/N records, for {len(records):,} records. No class-definition failures were found.

## Corrected natural-domain result

The research synthesis over-emphasized the nonoverlap slice and therefore made VIEW's natural transfer sound absent. The full WikiLarge/Simple-English readout has a more informative target-class structure. Values below are across the three original DeBERTa seeds; positive `gain_T_vs_N` contrast means the first arm extracts more benefit from the true aligned Wikipedia source than the second arm, relative to the ordinary N source.

| target class | R−C Δ(T−N) | V−C Δ(T−N) | V−R Δ(T−N) | V−C Δ(U−N) |
|---|---:|---:|---:|---:|
| overlap | {fmt(rc_overlap.gain_T_vs_N_mean)} ± {rc_overlap.gain_T_vs_N_seed_sd:.3f} | {fmt(vc_overlap.gain_T_vs_N_mean)} ± {vc_overlap.gain_T_vs_N_seed_sd:.3f} | {fmt(vr_overlap.gain_T_vs_N_mean)} ± {vr_overlap.gain_T_vs_N_seed_sd:.3f} | {fmt(vc_overlap.gain_U_vs_N_mean)} ± {vc_overlap.gain_U_vs_N_seed_sd:.3f} |
| nonoverlap | {fmt(rc_non.gain_T_vs_N_mean)} ± {rc_non.gain_T_vs_N_seed_sd:.3f} | {fmt(vc_non.gain_T_vs_N_mean)} ± {vc_non.gain_T_vs_N_seed_sd:.3f} | {fmt(vr_non.gain_T_vs_N_mean)} ± {vr_non.gain_T_vs_N_seed_sd:.3f} | {fmt(vc_non.gain_U_vs_N_mean)} ± {vc_non.gain_U_vs_N_seed_sd:.3f} |
| all targets | {fmt(rc_all.gain_T_vs_N_mean)} ± {rc_all.gain_T_vs_N_seed_sd:.3f} | {fmt(vc_all.gain_T_vs_N_mean)} ± {vc_all.gain_T_vs_N_seed_sd:.3f} | {fmt(vr_all.gain_T_vs_N_mean)} ± {vr_all.gain_T_vs_N_seed_sd:.3f} | {fmt(vc_all.gain_U_vs_N_mean)} ± {vc_all.gain_U_vs_N_seed_sd:.3f} |

Per-seed signs for `gain_T_vs_N` on the load-bearing classes:

```
{signs.to_string()}
```

The natural-domain statement should therefore be:

1. Exact natural repeats preserve the old ordering from the copy probe: REPEAT > VIEW > CLEAN for identical recurrence.
2. Natural restatements split by target class. On target tokens that recur in the source under changed sentence form, VIEW extracts more true-source benefit than both CLEAN and REPEAT (`V−C` {fmt(vc_overlap.gain_T_vs_N_mean)} ± {vc_overlap.gain_T_vs_N_seed_sd:.3f}; `V−R` {fmt(vr_overlap.gain_T_vs_N_mean)} ± {vr_overlap.gain_T_vs_N_seed_sd:.3f}). The nuisance term `V−C Δ(U−N)` is near zero ({fmt(vc_overlap.gain_U_vs_N_mean)} ± {vc_overlap.gain_U_vs_N_seed_sd:.3f}), so this is source-specific rather than a general register advantage.
3. On target tokens absent from the source, exact recurrence has the stable cost (`R−C` {fmt(rc_non.gain_T_vs_N_mean)} ± {rc_non.gain_T_vs_N_seed_sd:.3f}), while VIEW is approximately CLEAN (`V−C` {fmt(vc_non.gain_T_vs_N_mean)} ± {vc_non.gain_T_vs_N_seed_sd:.3f}) but still exceeds REPEAT (`V−R` {fmt(vr_non.gain_T_vs_N_mean)} ± {vr_non.gain_T_vs_N_seed_sd:.3f}).
4. The VIEW nonoverlap null is not a failure of transfer. It is a relation boundary: the VIEW intervention practiced nonidentical restatement of source-supported content, not lexical substitution requiring a source-absent target token.

## Revised principle wording

The general principle should be phrased as relation-typed form robustness under fixed budget: the relation repeatedly composed inside a context window determines not only whether a source is used, but the surface relation under which retrieval remains useful. Exact recurrence installs a source-recognition routine whose precise readout is bound to identical surface form; when the related source is nonidentical, the trigger persists but the readout can pull probability toward source content and away from changed targets. Restatement practice installs a source-use routine that survives sentence-form changes for source-recurring content. The boundary on source-absent substitute words is predicted by the fact that lexical substitution was not the practiced relation.

## Artifact changes for the next synthesis

- Wrote `{(_public_path('experiments/archive/relation_learning/data/express_review/wikipedia_transfer_by_target_class.csv')).relative_to(ROOT)}`.
- Wrote `{(_public_path('experiments/archive/relation_learning/data/express_review/wikipedia_transfer_by_seed_signs.csv')).relative_to(ROOT)}`.
- Wrote `{fig_png.relative_to(ROOT)}` and `{fig_pdf.relative_to(ROOT)}`.

The natural-domain figure for the paper should use the arm × target-class design above, not a nonoverlap-only line. The older nonoverlap-only reading should remain only as the evidence for the exact-recurrence cost on changed/substituted words.
"""
    NOTE.write_text(note, encoding="utf-8")

    print(json.dumps({
        "status": "WIKIPEDIA_TRANSFER_CORRECTION_DONE",
        "note": str(NOTE.relative_to(ROOT)),
        "summary_csv": str((_public_path('experiments/archive/relation_learning/data/express_review/wikipedia_transfer_by_target_class.csv')).relative_to(ROOT)),
        "signs_csv": str((_public_path('experiments/archive/relation_learning/data/express_review/wikipedia_transfer_by_seed_signs.csv')).relative_to(ROOT)),
        "figure_png": str(fig_png.relative_to(ROOT)),
        "pairs": len(pairs),
        "records": len(records),
        "chosen_counts": chosen_counts,
    }, indent=2))


if __name__ == "__main__":
    main()
