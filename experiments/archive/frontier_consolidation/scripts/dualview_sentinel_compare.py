#!/usr/bin/env python3
"""research sentinel comparison for dual-view 20M panels.

This is a CPU-only consumer of saved official-compatible prediction payloads.  It
reuses the validated research item reconstruction code and focuses on the fragile
families that decided the mature endpoint failures: Supplement QA/subject-aux,
EWoK material/spatial/quantitative domains, high-operation Entity groups, COMPS
subsets, GlobalPIQA halves, and Reading.  It is generic: run it first on existing
payloads to validate, then on aligned/shuffled/mlm_only as soon as their cheap
payloads exist.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from statistics import mean
from typing import Any

USER_ROOT = pathlib.Path(".").resolve()
SCRIPTS = USER_ROOT / "experiments/archive/frontier_consolidation/scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from pairwise_item_flip_analysis import PayloadLoader, compare_column  # noqa: E402

CHEAP_COLS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
DISCRETE_COLS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA"]

SENTINEL_GROUPS = {
    "Supplement": ["subject_aux_inversion", "qa_congruence_tricky", "qa_congruence_easy"],
    "EWoK": ["material-dynamics", "material-properties", "spatial-relations", "quantitative-properties", "physical-dynamics", "physical-interactions", "social-relations"],
    "Entity": ["regular_5_ops", "regular_4_ops", "regular_3_ops", "ambiref_5_ops", "ambiref_4_ops", "move_contents_5_ops", "move_contents_4_ops"],
    "COMPS": ["base", "wugs", "wugs_dist_before", "wugs_dist_in_between"],
    "GlobalPIQA": ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"],
}

reference_20M_REF = {
    "BLiMP": 59.69,
    "Supplement": 55.45,
    "EWoK": 50.73,
    "Entity": 18.65,
    "COMPS": 50.26,
    "GlobalPIQA": 34.195,
    "Reading": 8.67,
}
reference_20M_CHEAP7 = float(mean(reference_20M_REF[c] for c in CHEAP_COLS))


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: pathlib.Path | str) -> str:
    p = pathlib.Path(path)
    try:
        return str(p.resolve().relative_to(USER_ROOT))
    except Exception:
        return str(path)


def parse_candidate(spec: str) -> tuple[str, pathlib.Path]:
    if "=" in spec:
        label, path = spec.split("=", 1)
    elif ":" in spec and not spec.startswith("Sessions/"):
        label, path = spec.split(":", 1)
    else:
        p = pathlib.Path(spec)
        label, path = p.stem, spec
    return label.strip(), pathlib.Path(path)


def scores_from_payload(payload: dict[str, Any]) -> dict[str, float | None]:
    official = payload.get("official_overall", {}).get("scores", {})
    tasks = payload.get("tasks", {})
    out = {c: None for c in CHEAP_COLS}
    for c in CHEAP_COLS:
        if official.get(c) is not None:
            out[c] = float(official[c])
    # Fallback for partial payloads.
    for c in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]:
        if out[c] is None:
            r = tasks.get(c, {})
            if isinstance(r, dict) and r.get("score") is not None:
                out[c] = float(r["score"])
    if out["GlobalPIQA"] is None:
        vals = []
        for sub in ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]:
            r = tasks.get(sub, {})
            if isinstance(r, dict) and r.get("score") is not None:
                vals.append(float(r["score"]))
        if len(vals) == 2:
            out["GlobalPIQA"] = float(mean(vals))
    if out["Reading"] is None:
        r = tasks.get("Reading", {})
        if isinstance(r, dict) and isinstance(r.get("scores"), dict) and r["scores"].get("Reading") is not None:
            out["Reading"] = float(r["scores"]["Reading"])
        elif isinstance(r, dict) and r.get("score") is not None:
            out["Reading"] = float(r["score"])
    return out


def cheap7(scores: dict[str, float | None]) -> float | None:
    vals = [scores.get(c) for c in CHEAP_COLS]
    if any(v is None for v in vals):
        return None
    return float(mean(float(v) for v in vals))


def maybe_training_metrics(payload_path: pathlib.Path, payload: dict[str, Any]) -> dict[str, Any]:
    run_dir = payload.get("run_dir")
    if not run_dir:
        return {}
    sm = pathlib.Path(run_dir) / "scientific_metrics.json"
    if not sm.exists() and not sm.is_absolute():
        sm = USER_ROOT / sm
    if not sm.exists():
        return {"run_dir": run_dir, "scientific_metrics_exists": False}
    m = read_json(sm)
    return {
        "run_dir": rel(run_dir),
        "scientific_metrics": rel(sm),
        "mode": m.get("mode"),
        "updates": m.get("updates", m.get("actual_training_steps")),
        "total_charged_words": m.get("total_charged_words", m.get("word_exposure")),
        "total_main_word_exposure": m.get("total_main_word_exposure"),
        "total_aux_word_exposure": m.get("total_aux_word_exposure"),
        "schedule_total": m.get("schedule_total"),
        "final_loss": m.get("final_loss", m.get("loss_last")),
        "mean_aux_loss": m.get("mean_aux_loss"),
        "aux_loss_batches": m.get("aux_loss_batches"),
    }


def compare(base_label: str, base_path: pathlib.Path, cand_label: str, cand_path: pathlib.Path) -> dict[str, Any]:
    base_loader = PayloadLoader(base_path)
    cand_loader = PayloadLoader(cand_path)
    base_scores = scores_from_payload(base_loader.payload)
    cand_scores = scores_from_payload(cand_loader.payload)
    b7 = cheap7(base_scores); c7 = cheap7(cand_scores)
    cols: dict[str, Any] = {}
    sentinels: dict[str, Any] = {}
    for col in DISCRETE_COLS:
        c = compare_column(base_loader, cand_loader, col)
        # Keep full group tables in JSON, but present selected fragile groups first.
        cols[col] = {
            "delta_score_payload": c["delta_score_payload"],
            "delta_score_reconstructed_common": c["delta_score_reconstructed_common"],
            "n_common": c["n_common"],
            "flip_counts": c["flip_counts"],
            "gain_minus_loss_items": c["gain_minus_loss_items"],
            "best_groups_by_item_net": c["best_groups_by_item_net"][:12],
            "worst_groups_by_item_net": c["worst_groups_by_item_net"][:12],
            "payload_reconstruction_errors": c["payload_reconstruction_errors"],
        }
        if col in SENTINEL_GROUPS:
            by_name = c["groups_by_name"]
            sentinels[col] = {g: by_name.get(g) for g in SENTINEL_GROUPS[col]}
    return {
        "base_label": base_label,
        "candidate_label": cand_label,
        "base_payload": rel(base_path),
        "candidate_payload": rel(cand_path),
        "base_training": maybe_training_metrics(base_path, base_loader.payload),
        "candidate_training": maybe_training_metrics(cand_path, cand_loader.payload),
        "scores": {"base": base_scores, "candidate": cand_scores},
        "cheap7": {"base": b7, "candidate": c7, "delta": None if b7 is None or c7 is None else c7 - b7},
        "deltas": {k: (None if base_scores.get(k) is None or cand_scores.get(k) is None else cand_scores[k] - base_scores[k]) for k in CHEAP_COLS},
        "columns": cols,
        "sentinel_groups": sentinels,
    }


def build_markdown(out: dict[str, Any]) -> str:
    lines = [f"# research dual-view sentinel comparison — {out['label']}", ""]
    if out.get("reference_20M_reference"):
        lines += [f"research 20M reference cheap7: `{reference_20M_CHEAP7:.6f}`", ""]
    for cmp in out["comparisons"]:
        lines += [f"## {cmp['candidate_label']} vs {cmp['base_label']}", ""]
        lines += ["| Column | base | candidate | delta |", "|---|---:|---:|---:|"]
        bs = cmp["scores"]["base"]; cs = cmp["scores"]["candidate"]
        for c in CHEAP_COLS:
            b = bs.get(c); v = cs.get(c); d = cmp["deltas"].get(c)
            lines.append(f"| {c} | {b:.4f} | {v:.4f} | {d:+.4f} |" if b is not None and v is not None and d is not None else f"| {c} | {b} | {v} | {d} |")
        b7 = cmp["cheap7"]["base"]; c7 = cmp["cheap7"]["candidate"]; d7 = cmp["cheap7"]["delta"]
        if b7 is not None and c7 is not None and d7 is not None:
            lines.append(f"| **cheap7** | **{b7:.6f}** | **{c7:.6f}** | **{d7:+.6f}** |")
        lines.append("")
        lines.append("### Training/accounting")
        lines.append(f"- base: `{cmp.get('base_training')}`")
        lines.append(f"- candidate: `{cmp.get('candidate_training')}`")
        lines.append("")
        lines.append("### Fragile groups")
        for col, gs in cmp["sentinel_groups"].items():
            lines.append(f"#### {col}")
            lines.append("| group | n | base item % | cand item % | net gain-loss | gain | loss |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|")
            for g, r in gs.items():
                if r is None:
                    lines.append(f"| {g} | — | — | — | — | — | — |")
                else:
                    lines.append(f"| {g} | {r['n']} | {r['base_item_pct']:.2f} | {r['cand_item_pct']:.2f} | {r['net_gain_minus_loss']:+d} | {r['gain']} | {r['loss']} |")
            lines.append("")
        lines.append("### Best/worst item-net groups")
        for col, c in cmp["columns"].items():
            best = ", ".join(f"{g['group']} {g['net_gain_minus_loss']:+d}/{g['n']}" for g in c["best_groups_by_item_net"][:4])
            worst = ", ".join(f"{g['group']} {g['net_gain_minus_loss']:+d}/{g['n']}" for g in c["worst_groups_by_item_net"][:4])
            lines.append(f"- {col}: best {best}; worst {worst}")
        lines.append("")
    lines.append(f"JSON: `{out['out_json']}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Baseline payload path")
    ap.add_argument("--base-label", default="base")
    ap.add_argument("--candidate", action="append", required=True, help="label=payload path; may repeat")
    ap.add_argument("--pairwise-candidates", action="store_true", help="Also compare candidate[0] vs candidate[1] when at least two are given")
    ap.add_argument("--label", default="dualview_panel")
    ap.add_argument("--out-dir", default="experiments/archive/frontier_consolidation/data/dualview_sentinel_compare")
    args = ap.parse_args()

    base_path = pathlib.Path(args.base)
    candidates = [parse_candidate(x) for x in args.candidate]
    comparisons = [compare(args.base_label, base_path, lab, p) for lab, p in candidates]
    if args.pairwise_candidates and len(candidates) >= 2:
        lab0, p0 = candidates[0]
        for lab, p in candidates[1:]:
            comparisons.append(compare(lab0, p0, lab, p))
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{args.label}.json"
    out = {
        "status": "DUALVIEW_SENTINEL_COMPARE",
        "label": args.label,
        "base": {"label": args.base_label, "payload": rel(base_path)},
        "candidates": [{"label": lab, "payload": rel(p)} for lab, p in candidates],
        "reference_20M_reference": {"scores": reference_20M_REF, "cheap7": reference_20M_CHEAP7},
        "comparisons": comparisons,
        "out_json": rel(out_json),
    }
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_md = out_dir / f"{args.label}.md"
    out["out_md"] = rel(out_md)
    # update JSON with md path too
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_md.write_text(build_markdown(out), encoding="utf-8")
    summary = []
    for cmp in comparisons:
        summary.append({
            "comparison": f"{cmp['candidate_label']} vs {cmp['base_label']}",
            "cheap7_delta": cmp["cheap7"]["delta"],
            "deltas": cmp["deltas"],
        })
    print(json.dumps({"status": out["status"], "label": args.label, "summary": summary,
                      "out_json": rel(out_json), "out_md": rel(out_md)}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
