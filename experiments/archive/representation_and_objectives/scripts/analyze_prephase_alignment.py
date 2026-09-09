#!/usr/bin/env python3
"""research analysis: prephase alignment results with micro-EEBF decomposition.

Loads alignment_summary.json for each arm and computes:
  1. Standard contrastive accuracy per eval block
  2. Micro-EEBF: |focal_before_acc - focal_after_acc| on changed-focal worlds
     - Symmetric update→correct for both before+after → genuine coordinate use
     - focal_after >> focal_before → operation-prediction bias (after-state prior)
  3. Direction breakdown if available (A->B vs B->A)
  4. Cross-arm summary table

Usage:
  python3 analyze_prephase_alignment.py
  python3 analyze_prephase_alignment.py --arms aligned permuted disjoint scratch
  python3 analyze_prephase_alignment.py --seed 27100

Output: data/prephase_analysis/prephase_analysis_summary.md
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

STUDY = Path("experiments/archive/representation_and_objectives")
DATA = STUDY / "data"
OUT = DATA / "prephase_analysis"

ARMS = ["aligned", "permuted", "disjoint", "scratch"]
EVAL_BLOCKS_INTEREST = [
    "after_prephase_inline",
    "after_prephase_tag",
    "after_continuation_inline",
    "after_continuation_tag",
]


def load_summary(arm: str, seed: int) -> dict | None:
    p = DATA / f"step271_{arm}_seed{seed}" / "alignment_summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def query_acc(ev: dict, name: str) -> dict[str, float]:
    """Extract by_q from an eval block entry."""
    return ev.get("by_q", {})


def contrastive_acc(ev: dict) -> float | None:
    return ev.get("con_acc")


def micro_eebf(bq: dict[str, float]) -> float:
    """focal_after_acc - focal_before_acc for changed focal worlds.
    Positive = bias toward after-state.
    Near zero = symmetric, genuine before/after distinction.
    """
    fb = bq.get("focal_before", float("nan"))
    fa = bq.get("focal_after", float("nan"))
    try:
        return float(fa) - float(fb)
    except (TypeError, ValueError):
        return float("nan")


def summarize_arm(summary: dict) -> dict:
    arm = summary["arm"]
    seed = summary["seed"]
    results = {}

    for block in EVAL_BLOCKS_INTEREST:
        evdict = summary.get("evals", {}).get(block, {})
        if not evdict:
            results[block] = None
            continue
        entries = {}
        for en, ev in sorted(evdict.items()):
            bq = ev.get("by_q", {})
            entries[en] = {
                "con": contrastive_acc(ev),
                "fb": bq.get("focal_before"),
                "fa": bq.get("focal_after"),
                "sb": bq.get("secondary_before"),
                "sa": bq.get("secondary_after"),
                "micro_eebf": micro_eebf(bq),
                "fa_m": ev.get("by_m", {}).get("focal_after"),
                "sa_m": ev.get("by_m", {}).get("secondary_after"),
            }
        results[block] = entries

    # Fit summaries
    fits = {}
    for phase in ["prephase", "continuation"]:
        r = summary.get("results", {}).get(phase)
        if r:
            fits[phase] = {
                "best_acc": r.get("best_train_acc"),
                "by_kind": r.get("fit", {}).get("by_train_kind", {}),
            }

    return {"arm": arm, "seed": seed, "eval_blocks": results, "fits": fits}


def write_report(arms_data: list[dict], out: Path):
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# research Prephase Alignment Analysis\n\n"]
    lines.append("## Decision table\n\n")
    lines.append("| arm | pre fit changed | cont fit changed | cont_inline hC_hS con | cont inline fa | cont inline fb | micro_eebf | secondary | tag retention |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")

    for d in arms_data:
        arm = d["arm"]
        pre_fit = d["fits"].get("prephase", {}).get("by_kind", {})
        cont_fit = d["fits"].get("continuation", {}).get("by_kind", {})
        pre_changed = pre_fit.get("sparse_changed_focal|ctx=tag_only|q=direct_tag",
                      pre_fit.get("sparse_changed_focal|ctx=inline_role|q=direct_tag", float("nan")))
        cont_changed = cont_fit.get("sparse_changed_focal|ctx=inline_role|q=direct_tag", float("nan"))

        # Main continuation inline eval - heldChanged heldStable
        cont_inline = d["eval_blocks"].get("after_continuation_inline", {}) or {}
        hc_hs = next((v for k, v in cont_inline.items() if "heldChanged" in k or "hC_hS" in k), None)
        con_acc = hc_hs.get("con") if hc_hs else float("nan")
        fa = hc_hs.get("fa") if hc_hs else float("nan")
        fb = hc_hs.get("fb") if hc_hs else float("nan")
        eebf = hc_hs.get("micro_eebf") if hc_hs else float("nan")
        sa = hc_hs.get("sa") if hc_hs else float("nan")

        # Tag retention
        cont_tag = d["eval_blocks"].get("after_continuation_tag", {}) or {}
        tag_hc = next((v for k, v in cont_tag.items() if "heldChanged" in k or "hC_hS" in k), None)
        tag_con = tag_hc.get("con") if tag_hc else float("nan")

        def fmt(v):
            if v is None or (isinstance(v, float) and v != v):
                return "–"
            return f"{float(v):.3f}"

        lines.append(f"| {arm} | {fmt(pre_changed)} | {fmt(cont_changed)} | {fmt(con_acc)} | {fmt(fa)} | {fmt(fb)} | {fmt(eebf)} | {fmt(sa)} | {fmt(tag_con)} |\n")

    lines.append("\n")
    lines.append("Micro-EEBF = focal_after_acc − focal_before_acc on held changed/stable queries.\n")
    lines.append("Near zero → genuine before/after discrimination; large positive → after-state bias.\n\n")

    # Per-arm detailed tables
    for d in arms_data:
        arm = d["arm"]
        lines.append(f"## {arm} (seed {d['seed']})\n\n")

        # Fits
        for phase in ["prephase", "continuation"]:
            f = d["fits"].get(phase)
            if f:
                lines.append(f"**{phase}** best_acc={f.get('best_acc'):.6f}\n")
                for k, v in sorted(f.get("by_kind", {}).items()):
                    lines.append(f"  - {k}: {v:.4f}\n")
                lines.append("\n")

        for block in EVAL_BLOCKS_INTEREST:
            evdict = d["eval_blocks"].get(block)
            if not evdict:
                lines.append(f"### {block}: not available\n\n")
                continue
            lines.append(f"### {block}\n\n")
            lines.append("| eval | con | fb | fa | sb | sa | micro_eebf | fa_m | sa_m |\n")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            for en, ev in sorted(evdict.items()):
                def gv(k):
                    v = ev.get(k)
                    return f"{float(v):.3f}" if v is not None else "–"
                def gf(k):
                    v = ev.get(k)
                    return f"{float(v):.2f}" if v is not None else "–"
                lines.append(f"| {en} | {gv('con')} | {gv('fb')} | {gv('fa')} "
                             f"| {gv('sb')} | {gv('sa')} | {gv('micro_eebf')} "
                             f"| {gf('fa_m')} | {gf('sa_m')} |\n")
            lines.append("\n")

    (out / "prephase_analysis_summary.md").write_text("".join(lines), "utf-8")
    print(f"Analysis written: {out / 'prephase_analysis_summary.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--seed", type=int, default=27100)
    args = ap.parse_args()

    loaded = []
    for arm in args.arms:
        s = load_summary(arm, args.seed)
        if s is None:
            print(f"  [skip] {arm} seed{args.seed} — no summary found")
            continue
        loaded.append(summarize_arm(s))
        print(f"  [loaded] {arm} seed{args.seed}")

    if not loaded:
        print("No arm results found yet.")
        return

    write_report(loaded, OUT)


if __name__ == "__main__":
    main()
