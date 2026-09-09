#!/usr/bin/env python3
"""Analyze research/131 sparse20 dual-view panel once cheap7 summaries exist.

Consumes available summaries for:
  - separated sparse20 aligned (research trainer)
  - separated sparse20 shuffled (research trainer)
  - coupled sparse20 aligned (research trainer)
and compares them with exact 20M references from research/research.

The script is safe to run before all summaries exist; it writes PENDING with the
available subset and computes decisions only for observed comparisons.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import time
from pathlib import Path
from statistics import mean

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/frontier_consolidation')
OUT_DIR = _public_path('experiments/archive/frontier_consolidation/data/sparse20_panel_analysis')
OUT_JSON = _public_path('experiments/archive/frontier_consolidation/data/sparse20_panel_analysis/sparse20_panel_analysis.json')
OUT_MD = _public_path('research/documents/frontier_consolidation/data/sparse20_panel_analysis/sparse20_panel_analysis.md')

CHEAP_COLS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]

RUNS = {
    "sep_aligned": {
        "summary": _public_path('experiments/archive/frontier_consolidation/data/sep_sparse20_aligned_20m_summary/sep_sparse20_aligned_20M_summary.json'),
        "run_dir": _public_path('experiments/archive/frontier_consolidation/training/runs/sep_sparse20_aligned_20M_seed43022'),
        "pathway": "separated",
        "mode": "aligned",
    },
    "sep_shuffled": {
        "summary": _public_path('experiments/archive/frontier_consolidation/data/sep_sparse20_shuffled_20m_summary/sep_sparse20_shuffled_20M_summary.json'),
        "run_dir": _public_path('experiments/archive/frontier_consolidation/training/runs/sep_sparse20_shuffled_20M_seed43022'),
        "pathway": "separated",
        "mode": "shuffled",
    },
    "coupled_aligned": {
        "summary": _public_path('experiments/archive/frontier_consolidation/data/coupled_sparse20_aligned_20m_summary/coupled_sparse20_aligned_20M_summary.json'),
        "run_dir": _public_path('experiments/archive/frontier_consolidation/training/runs/coupled_sparse20_aligned_20M_seed43022'),
        "pathway": "coupled",
        "mode": "aligned",
    },
}

# Exact or full-precision references established in prior steps.
REFERENCES = {
    "reference_20M": {
        "scores": {"BLiMP": 59.69, "Supplement": 55.45, "EWoK": 50.73, "Entity": 18.65,
                   "COMPS": 50.26, "GlobalPIQA": 34.195, "Reading": 8.67},
    },
    "mlm_only_20M": {
        "scores": {"BLiMP": 62.05, "Supplement": 58.43, "EWoK": 50.10, "Entity": 18.36,
                   "COMPS": 50.70, "GlobalPIQA": 32.67, "Reading": 6.20},
    },
    "broad_coupled_aligned_20M": {
        "scores": {"BLiMP": 56.72, "Supplement": 51.76, "EWoK": 51.11, "Entity": 17.91,
                   "COMPS": 50.17, "GlobalPIQA": 36.10, "Reading": 7.57},
    },
    "broad_coupled_shuffled_20M": {
        "scores": {"BLiMP": 56.54, "Supplement": 52.82, "EWoK": 51.31, "Entity": 17.35,
                   "COMPS": 49.80, "GlobalPIQA": 34.62, "Reading": 7.35},
    },
}
for ref in REFERENCES.values():
    ref["cheap7"] = float(mean(ref["scores"][c] for c in CHEAP_COLS))


def rel(p: Path | str) -> str:
    p = Path(p)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def load_summary(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_metrics(run_dir: Path):
    p = run_dir / "scientific_metrics.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def score_record(spec):
    summ = load_summary(spec["summary"])
    metrics = load_metrics(spec["run_dir"])
    rec = {
        "summary_path": rel(spec["summary"]),
        "summary_exists": summ is not None,
        "run_dir": rel(spec["run_dir"]),
        "pathway": spec["pathway"],
        "mode": spec["mode"],
        "training_metrics": None,
        "scores": None,
        "cheap7": None,
        "deltas_vs_refs": None,
    }
    if metrics is not None:
        rec["training_metrics"] = {
            "status": metrics.get("status"),
            "mode": metrics.get("mode"),
            "pathway": metrics.get("pathway"),
            "updates": metrics.get("updates"),
            "total_main_word_exposure": metrics.get("total_main_word_exposure"),
            "total_aux_word_exposure": metrics.get("total_aux_word_exposure"),
            "total_charged_words": metrics.get("total_charged_words"),
            "first_loss": metrics.get("first_loss"),
            "final_loss": metrics.get("final_loss"),
            "mean_loss": metrics.get("mean_loss"),
            "mean_aux_loss": metrics.get("mean_aux_loss"),
            "mean_neutral_loss": metrics.get("mean_neutral_loss"),
            "aux_loss_batches": metrics.get("aux_loss_batches"),
        }
    if summ is not None:
        scores = summ.get("scores", {})
        rec["scores"] = {c: scores.get(c) for c in CHEAP_COLS}
        rec["cheap7"] = summ.get("cheap7")
        drefs = {}
        for name, ref in REFERENCES.items():
            drefs[name] = {
                "cheap7_delta": None if rec["cheap7"] is None else float(rec["cheap7"] - ref["cheap7"]),
                "column_deltas": {c: (None if rec["scores"].get(c) is None else float(rec["scores"][c] - ref["scores"][c])) for c in CHEAP_COLS},
            }
        rec["deltas_vs_refs"] = drefs
    return rec


def pair_delta(a, b, records):
    ra, rb = records.get(a), records.get(b)
    if not ra or not rb or ra.get("cheap7") is None or rb.get("cheap7") is None:
        return None
    return {
        "cheap7_delta": float(ra["cheap7"] - rb["cheap7"]),
        "column_deltas": {c: (None if ra["scores"].get(c) is None or rb["scores"].get(c) is None else float(ra["scores"][c] - rb["scores"][c])) for c in CHEAP_COLS},
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = {name: score_record(spec) for name, spec in RUNS.items()}
    available = [k for k, v in records.items() if v.get("summary_exists")]
    pending = [k for k, v in records.items() if not v.get("summary_exists")]

    comparisons = {
        "sep_aligned_minus_sep_shuffled": pair_delta("sep_aligned", "sep_shuffled", records),
        "sep_aligned_minus_coupled_aligned": pair_delta("sep_aligned", "coupled_aligned", records),
        "coupled_aligned_minus_broad_aligned": None,
        "sep_aligned_minus_mlm_only_ref": None,
    }
    if records["coupled_aligned"].get("cheap7") is not None:
        comparisons["coupled_aligned_minus_broad_aligned"] = {
            "cheap7_delta": float(records["coupled_aligned"]["cheap7"] - REFERENCES["broad_coupled_aligned_20M"]["cheap7"]),
            "column_deltas": {c: float(records["coupled_aligned"]["scores"][c] - REFERENCES["broad_coupled_aligned_20M"]["scores"][c]) for c in CHEAP_COLS},
        }
    if records["sep_aligned"].get("cheap7") is not None:
        comparisons["sep_aligned_minus_mlm_only_ref"] = {
            "cheap7_delta": float(records["sep_aligned"]["cheap7"] - REFERENCES["mlm_only_20M"]["cheap7"]),
            "column_deltas": {c: float(records["sep_aligned"]["scores"][c] - REFERENCES["mlm_only_20M"]["scores"][c]) for c in CHEAP_COLS},
        }

    # Interpret only observed facts. The 20M gate is intentionally conservative:
    # it is a cheap discriminator for whether to use the verified 82M slow function;
    # it is not an endpoint score by itself.
    decisions = {}
    sa = records["sep_aligned"]
    ss = records["sep_shuffled"]
    ca = records["coupled_aligned"]
    if sa.get("cheap7") is not None:
        d_mlm = comparisons["sep_aligned_minus_mlm_only_ref"]["cheap7_delta"]
        sent_dmg = comparisons["sep_aligned_minus_mlm_only_ref"]["column_deltas"]
        decisions["sep_aligned_beats_mlm_only"] = d_mlm > 0
        decisions["sep_aligned_no_major_fragile_family_damage_vs_mlm_only"] = all(sent_dmg[c] > -1.0 for c in ["EWoK", "Reading", "Supplement"])
    if comparisons["sep_aligned_minus_coupled_aligned"] is not None:
        decisions["separation_beats_coupled_under_same_sparse_data"] = comparisons["sep_aligned_minus_coupled_aligned"]["cheap7_delta"] > 0
    if comparisons["sep_aligned_minus_sep_shuffled"] is not None:
        decisions["true_correspondence_matters_under_separation"] = comparisons["sep_aligned_minus_sep_shuffled"]["cheap7_delta"] > 0
    if ca.get("cheap7") is not None:
        decisions["coupled_sparse_beats_mlm_only"] = (ca["cheap7"] - REFERENCES["mlm_only_20M"]["cheap7"]) > 0
        decisions["sparsity_alone_improves_broad_coupled"] = comparisons["coupled_aligned_minus_broad_aligned"]["cheap7_delta"] > 0

    if pending:
        route_read = "pending_sparse20_panel"
    elif decisions.get("sep_aligned_beats_mlm_only") and decisions.get("true_correspondence_matters_under_separation") and decisions.get("sep_aligned_no_major_fragile_family_damage_vs_mlm_only"):
        route_read = "supports_82M_frozen_slow_private_tail_test"
    elif decisions.get("separation_beats_coupled_under_same_sparse_data") and not decisions.get("sep_aligned_beats_mlm_only", False):
        route_read = "separation_reduces_damage_but_does_not_add_broad_value"
    else:
        route_read = "does_not_support_more_training_in_this_family"

    result = {
        "status": "PENDING" if pending else "COMPLETE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "available": available,
        "pending": pending,
        "records": records,
        "references": REFERENCES,
        "comparisons": comparisons,
        "decisions": decisions,
        "route_read": route_read,
        "scientific_reading": "This panel factors sparsity from pathway separation. A positive separated aligned-vs-shuffled effect matters only if separated aligned also exceeds the exact mlm_only adapter scaffold and avoids the known EWoK/Reading/Supplement damage; otherwise it is local correspondence learning inside an unhelpful score tradeoff.",
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["# research sparse20 panel analysis", "", f"Status: **{result['status']}**", f"Route read: `{route_read}`", ""]
    lines.append(f"Available: `{available}`; pending: `{pending}`")
    lines.append("")
    lines.append("## Scores")
    lines.append("| arm | cheap7 | BLiMP | Supp | EWoK | Entity | COMPS | GP | Reading | main words | aux words |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, rec in records.items():
        s = rec.get("scores") or {}
        tm = rec.get("training_metrics") or {}
        lines.append("| {name} | {cheap7} | {BLiMP} | {Supplement} | {EWoK} | {Entity} | {COMPS} | {GlobalPIQA} | {Reading} | {main} | {aux} |".format(
            name=name, cheap7=rec.get("cheap7"), BLiMP=s.get("BLiMP"), Supplement=s.get("Supplement"), EWoK=s.get("EWoK"),
            Entity=s.get("Entity"), COMPS=s.get("COMPS"), GlobalPIQA=s.get("GlobalPIQA"), Reading=s.get("Reading"),
            main=tm.get("total_main_word_exposure"), aux=tm.get("total_aux_word_exposure")))
    lines.append("")
    lines.append("## Observed decisions")
    for k, v in decisions.items():
        lines.append(f"- `{k}`: `{v}`")
    lines.append("")
    lines.append("## Comparisons")
    for k, v in comparisons.items():
        lines.append(f"- `{k}`: `{v}`")
    lines.append("")
    lines.append(result["scientific_reading"])
    lines.append(f"\nJSON: `{rel(OUT_JSON)}`")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "route_read": route_read, "available": available, "pending": pending, "out_json": rel(OUT_JSON), "out_md": rel(OUT_MD)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
