#!/usr/bin/env python3
"""research: decompose bridge relation movement into revision pressure and source selection.

The corrected bridge did not acquire the complete held operation.  This analyzer keeps
that result from being collapsed into one score by separating:
  U: mean self-update new-over-source margin;
  R: mean retain source-over-new margin after a distractor update;
  beta=(U+R)/2: recipient-conditional response needed for both UPDATE and RETAIN;
  alpha=(U-R)/2: shared preference for the replacement candidate over old source.
It also tracks correct-source versus wrong-source margins and hard behavioral counts.
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
from typing import Any, Dict, Iterable, List

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
ORDINARY = _public_path('experiments/archive/functional_learning/data/bridge_ordinary_relation_eval/bridge_eval_summary.json')
ANSWER = _public_path('experiments/archive/functional_learning/data/bridge_answer_relation_eval/bridge_eval_summary.json')
SPECIALIST = _public_path('experiments/archive/functional_learning/data/saved_state_replicate_neutral_threeentity/saved_state_summary.json')
OUT_DIR = _public_path('experiments/archive/functional_learning/data/bridge_response_decomposition')
FIG_DIR = _public_path('experiments/archive/functional_learning/figures')


def rel(path: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def finite(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def decomp(U: float, R: float) -> Dict[str, float]:
    return {"U_update_new_over_source": U, "R_retain_source_over_new": R, "beta_recipient_response": (U + R) / 2.0, "alpha_shared_replacement_preference": (U - R) / 2.0}


def record_from_bridge(arm: str, rec: Dict[str, Any]) -> Dict[str, Any]:
    tr = rec["relation"]["threeway"]
    ms = rec["relation"].get("margin_summary", {})
    reass = rec["relation"].get("reassignment", {})
    three = rec["relation"].get("three_entity", {})
    label = str(rec["label"])
    upd = None
    if label.startswith("update_"):
        try:
            upd = int(label.split("_")[1])
        except Exception:
            upd = None
    U = finite(tr.get("mean_update_new_over_source"))
    R = finite(tr.get("mean_retain_correct_over_new"))
    out = {
        "source": "bridge_relation_eval",
        "arm": arm,
        "label": label,
        "update": upd,
        **decomp(U, R),
        "mean_neutral_correct_vs_wrong": finite(ms.get("mean_neutral_correct_vs_wrong")),
        "mean_retain_correct_vs_wrong": finite(ms.get("mean_retain_correct_vs_wrong")),
        "mean_neutral_correct_vs_new": finite(ms.get("mean_neutral_correct_vs_replacement")),
        "neutral_both_full_source": int(tr.get("neutral_both_full_source", 0)),
        "retain_both_full_source": int(tr.get("retain_both_full_source", 0)),
        "update_both_correct": int(tr.get("update_both_correct", 0)),
        "reassignment_both_follow": int(reass.get("n_swap_both_follow", 0)),
        "three_entity_update_c_both_query_indexed": int(three.get("update_c_both_query_indexed", 0)),
    }
    return out


def specialist_rows() -> List[Dict[str, Any]]:
    data = json.loads(SPECIALIST.read_text(encoding="utf-8"))
    rows: List[Dict[str, Any]] = []
    parent = data["parent_extended_held"]
    parent_reas = data["parent_reassignment_held"]
    parent_three = data["parent_three_entity"]
    rows.append({
        "source": "saved_specialist_summary",
        "arm": "parent_step042",
        "label": "coherent86_parent",
        "update": None,
        **decomp(finite(parent.get("mean_update_new_over_source")), finite(parent.get("mean_retain_correct_over_new"))),
        "mean_neutral_correct_vs_wrong": finite(parent.get("mean_neutral_cross_source")),
        "mean_retain_correct_vs_wrong": finite(parent.get("mean_retain_cross_source")),
        "mean_neutral_correct_vs_new": finite(parent.get("mean_neutral_correct_over_new")),
        "neutral_both_full_source": int(parent.get("neutral_both_full_source", 0)),
        "retain_both_full_source": int(parent.get("retain_both_full_source", 0)),
        "update_both_correct": int(parent.get("update_both_correct", 0)),
        "reassignment_both_follow": int(parent_reas.get("n_swap_both_follow", 0)),
        "three_entity_update_c_both_query_indexed": int(parent_three.get("update_c_both_query_indexed", 0)),
    })
    for seed_rec in data.get("per_seed", []):
        ext = seed_rec["trained_extended_held"]
        reas = seed_rec["trained_reassignment_held"]
        three = seed_rec["trained_three_entity"]
        rows.append({
            "source": "saved_specialist_summary",
            "arm": "isolated_answer_only_specialist",
            "label": f"seed_{seed_rec['seed']}_epoch80",
            "update": None,
            **decomp(finite(ext.get("mean_update_new_over_source")), finite(ext.get("mean_retain_correct_over_new"))),
            "mean_neutral_correct_vs_wrong": finite(ext.get("mean_neutral_cross_source")),
            "mean_retain_correct_vs_wrong": finite(ext.get("mean_retain_cross_source")),
            "mean_neutral_correct_vs_new": finite(ext.get("mean_neutral_correct_over_new")),
            "neutral_both_full_source": int(ext.get("neutral_both_full_source", 0)),
            "retain_both_full_source": int(ext.get("retain_both_full_source", 0)),
            "update_both_correct": int(ext.get("update_both_correct", 0)),
            "reassignment_both_follow": int(reas.get("n_swap_both_follow", 0)),
            "three_entity_update_c_both_query_indexed": int(three.get("update_c_both_query_indexed", 0)),
        })
    return rows


def load_bridge(path: pathlib.Path, arm: str) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [record_from_bridge(arm, rec) for rec in data["results"]]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    rows.extend(load_bridge(ORDINARY, "ordinary_relation_wwm"))
    rows.extend(load_bridge(ANSWER, "answer_allocation_mixed"))
    rows.extend(specialist_rows())

    csv_path = _public_path('experiments/archive/functional_learning/data/bridge_response_decomposition/bridge_response_decomposition.csv')
    fields = [
        "source", "arm", "label", "update", "U_update_new_over_source", "R_retain_source_over_new",
        "beta_recipient_response", "alpha_shared_replacement_preference", "mean_neutral_correct_vs_wrong",
        "mean_retain_correct_vs_wrong", "mean_neutral_correct_vs_new", "neutral_both_full_source",
        "retain_both_full_source", "update_both_correct", "reassignment_both_follow",
        "three_entity_update_c_both_query_indexed",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})

    # concise scientific contrasts for easy reading
    by_key = {(r["arm"], r["label"]): r for r in rows}
    important_keys = [
        ("answer_allocation_mixed", "coherent86_parent"),
        ("answer_allocation_mixed", "update_0150"),
        ("answer_allocation_mixed", "update_0354"),
        ("ordinary_relation_wwm", "update_0354"),
        ("isolated_answer_only_specialist", "seed_40040_epoch80"),
        ("isolated_answer_only_specialist", "seed_40041_epoch80"),
        ("isolated_answer_only_specialist", "seed_40042_epoch80"),
    ]
    contrast_rows = [by_key[k] for k in important_keys if k in by_key]
    summary = {
        "status": "BRIDGE_RESPONSE_DECOMPOSITION_DONE",
        "inputs": {"ordinary_bridge": rel(ORDINARY), "answer_bridge": rel(ANSWER), "specialist": rel(SPECIALIST)},
        "csv": rel(csv_path),
        "rows": rows,
        "key_contrasts": contrast_rows,
        "interpretation": {
            "answer_mixed_update150": "Large movement is mainly reduction of shared replacement preference: beta remains near zero relative to alpha and correct-source vs wrong-source margins are small.",
            "ordinary_endpoint": "Relation WWM barely changes the selector-like margins and does not acquire held operation.",
            "isolated_specialist": "Answer-only many-epoch specialist has large beta, large source-vs-wrong margins, reassignment following, and three-entity query-indexed selection, unlike the mixed bridge.",
        },
    }
    summary_path = _public_path('experiments/archive/functional_learning/data/bridge_response_decomposition/bridge_response_decomposition.json')
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.5, 5.2))
        for arm, marker, color in [("ordinary_relation_wwm", "o", "#666666"), ("answer_allocation_mixed", "o", "#1f77b4")]:
            pts = [r for r in rows if r["arm"] == arm and r.get("update") is not None]
            pts = sorted(pts, key=lambda r: int(r["update"]))
            ax.plot([r["alpha_shared_replacement_preference"] for r in pts], [r["beta_recipient_response"] for r in pts], marker=marker, color=color, label=arm)
            for r in pts:
                if r["label"] in {"update_0150", "update_0354"}:
                    ax.annotate(r["label"].replace("update_", "u"), (r["alpha_shared_replacement_preference"], r["beta_recipient_response"]), fontsize=8)
        specs = [r for r in rows if r["arm"] == "isolated_answer_only_specialist"]
        ax.scatter([r["alpha_shared_replacement_preference"] for r in specs], [r["beta_recipient_response"] for r in specs], marker="*", s=130, color="#d62728", label="isolated specialists")
        par = by_key.get(("answer_allocation_mixed", "coherent86_parent"))
        if par:
            ax.scatter([par["alpha_shared_replacement_preference"]], [par["beta_recipient_response"]], marker="x", s=90, color="black", label="coherent86 parent")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("alpha = (UPDATE new-source margin - RETAIN source-new margin)/2")
        ax.set_ylabel("beta = (UPDATE new-source margin + RETAIN source-new margin)/2")
        ax.set_title("Bridge movement is mostly replacement-preference removal, not selector acquisition")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig_path = _public_path('experiments/archive/functional_learning/figures/bridge_alpha_beta_decomposition.png')
        fig.savefig(fig_path, dpi=180)
        summary["figure"] = rel(fig_path)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except Exception as exc:
        summary["plot_error"] = repr(exc)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({"status": summary["status"], "summary": rel(summary_path), "csv": rel(csv_path), "figure": summary.get("figure"), "n_rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
