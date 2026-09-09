#!/usr/bin/env python3
"""research: format-matched T/U/N state-margin readout for designed relation arms.

This reruns the research state-margin scorer on CLEAN/REPEAT/VIEW and the matched
split arms for both available seeds, then evaluates the format-matched contrast
T(new-source) - U(new-source).  T and U both include source, an update sentence,
and the same query frame; only whether the update belongs to the target entity is
changed.  This avoids letting the neutral N anchor carry the interpretation when
split arms have a source-alone format shift.
"""
from __future__ import annotations

import csv
import json
import math
import pathlib
import statistics
import sys
import time
from collections import defaultdict

ROOT = pathlib.Path.cwd()
OUT_DIR = ROOT / "experiments/archive/relation_learning/data/relation_arm_state_margin_tu"

ARM_MODELS = {
    # seed 43022: clean/repeat/view from frontier_consolidation, splits from relation_learning
    "seed43022_clean": {
        "seed": "43022", "arm": "clean",
        "root": ROOT / "experiments/archive/frontier_consolidation/training/runs/full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43022/hf_model",
    },
    "seed43022_repeat": {
        "seed": "43022", "arm": "repeat",
        "root": ROOT / "experiments/archive/frontier_consolidation/training/runs/full_p2c_c2p_abs_repeat_dose2p64x_matched_rowholdout_deberta100M_seed43022/hf_model",
    },
    "seed43022_view": {
        "seed": "43022", "arm": "view",
        "root": ROOT / "experiments/archive/frontier_consolidation/training/runs/full_p2c_c2p_abs_view_dose2p64x_matched_rowholdout_deberta100M_seed43022/hf_model",
    },
    "seed43022_repeat_split": {
        "seed": "43022", "arm": "repeat_split",
        "root": ROOT / "experiments/archive/relation_learning/training/runs/full_p2c_c2p_abs_repeat_split_dose2p64x_rowholdout_deberta100M_seed43022/hf_model",
    },
    "seed43022_view_split": {
        "seed": "43022", "arm": "view_split",
        "root": ROOT / "experiments/archive/relation_learning/training/runs/full_p2c_c2p_abs_view_split_dose2p64x_rowholdout_deberta100M_seed43022/hf_model",
    },
    # seed 43122
    "seed43122_clean": {
        "seed": "43122", "arm": "clean",
        "root": ROOT / "experiments/archive/frontier_consolidation/training/runs/full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43122/hf_model",
    },
    "seed43122_repeat": {
        "seed": "43122", "arm": "repeat",
        "root": ROOT / "experiments/archive/frontier_consolidation/training/runs/full_p2c_c2p_abs_repeat_dose2p64x_matched_rowholdout_deberta100M_seed43122/hf_model",
    },
    "seed43122_view": {
        "seed": "43122", "arm": "view",
        "root": ROOT / "experiments/archive/frontier_consolidation/training/runs/full_p2c_c2p_abs_view_dose2p64x_matched_rowholdout_deberta100M_seed43122/hf_model",
    },
    "seed43122_repeat_split": {
        "seed": "43122", "arm": "repeat_split",
        "root": ROOT / "experiments/archive/relation_learning/training/runs/full_p2c_c2p_abs_repeat_split_dose2p64x_rowholdout_deberta100M_seed43122/hf_model",
    },
    "seed43122_view_split": {
        "seed": "43122", "arm": "view_split",
        "root": ROOT / "experiments/archive/relation_learning/training/runs/full_p2c_c2p_abs_view_split_dose2p64x_rowholdout_deberta100M_seed43122/hf_model",
    },
}

PRESCORED_PREDICTION = {
    "step": 56,
    "purpose": "Test whether research state-margin relation-arm result survives a format-matched T-U contrast and seed43122 replication.",
    "contrast": "UPDATED_USE: T(full_m_orig_new_minus_source) - U(full_m_orig_new_minus_source)",
    "prediction": "If this is a true relation-locality readout rather than neutral-format anchoring, VIEW should exceed CLEAN and split arms should be near CLEAN in T-U at both seeds. REPEAT may share a positive second-sentence-use component, but it should not exceed VIEW on changed-form state use.",
    "interpretation_boundary": "Report T, U, N, T-U, and T-N side by side. A T-N-only pattern is not sufficient because split arms alter N format scores.",
}


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def mean(xs: list[float]) -> float:
    ys = [x for x in xs if math.isfinite(x)]
    return float(statistics.mean(ys)) if ys else float("nan")


def se(xs: list[float]) -> float:
    ys = [x for x in xs if math.isfinite(x)]
    if len(ys) <= 1:
        return float("nan")
    return float(statistics.stdev(ys) / math.sqrt(len(ys)))


def ffloat(x: object) -> float:
    try:
        return float(x)  # type: ignore[arg-type]
    except Exception:
        return float("nan")


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def run_scorer(device: str, batch_size: int) -> dict:
    script_dir = str(ROOT / "experiments/archive/relation_learning/scripts")
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    import score_state_margin_probe as scorer

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = OUT_DIR / "pre_score_prediction_revision_056.json"
    pred_path.write_text(json.dumps(PRESCORED_PREDICTION, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    scorer.MODEL_ROOTS.update(ARM_MODELS)
    model_names = list(ARM_MODELS.keys())
    raw_dir = OUT_DIR / "raw_scores"
    sys.argv = [
        "relation_arm_tu",
        "--models", *model_names,
        "--checkpoints", "final",
        "--out-dir", str(raw_dir),
        "--device", device,
        "--batch-size", str(batch_size),
        "--skip-missing",
    ]
    print(f"[RUN] scoring {len(model_names)} relation-arm models at final on {device}", flush=True)
    scorer.main()
    return {"prediction": rel(pred_path), "raw_dir": rel(raw_dir)}


def compute_contrasts() -> dict:
    raw_rows_path = OUT_DIR / "raw_scores/raw_TUN_margin_rows.csv"
    if not raw_rows_path.exists():
        raise FileNotFoundError(raw_rows_path)
    with raw_rows_path.open(encoding="utf-8") as f:
        raw_rows = list(csv.DictReader(f))

    # one row per condition; join T/U/N for each packet/model
    by_key: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for r in raw_rows:
        key = (r["seed"], r["arm"], r["checkpoint"], r["probe_set"], r["slot_mode"], r["packet_type"], r["packet_id"])
        by_key[key][r["condition"]] = r

    contrast_rows: list[dict] = []
    for key, conds in by_key.items():
        if not all(c in conds for c in ("T", "U", "N")):
            continue
        seed, arm, ckpt, pset, slot, ptype, pid = key
        T, U, N = conds["T"], conds["U"], conds["N"]
        rec = {
            "seed": seed, "arm": arm, "checkpoint": ckpt, "probe_set": pset,
            "slot_mode": slot, "packet_type": ptype, "packet_id": pid,
            "gold_state_type": T.get("gold_state_type", ""),
            "updated_conflict_heuristic": T.get("updated_conflict_heuristic", ""),
        }
        for prefix, field in [
            ("full_new_minus_source", "full_m_orig_new_minus_source"),
            ("content_new_minus_source", "content_m_orig_new_minus_source"),
            ("full_source_minus_new", "full_m_source_minus_orig_new"),
            ("content_source_minus_new", "content_m_source_minus_orig_new"),
            ("full_source_minus_swapped", "full_m_source_minus_swapped_new"),
            ("content_source_minus_swapped", "content_m_source_minus_swapped_new"),
        ]:
            tv, uv, nv = ffloat(T.get(field)), ffloat(U.get(field)), ffloat(N.get(field))
            rec[f"{prefix}_T"] = tv
            rec[f"{prefix}_U"] = uv
            rec[f"{prefix}_N"] = nv
            rec[f"{prefix}_T_minus_U"] = tv - uv
            rec[f"{prefix}_T_minus_N"] = tv - nv
            rec[f"{prefix}_U_minus_N"] = uv - nv
        contrast_rows.append(rec)

    contrast_path = OUT_DIR / "format_matched_contrast_rows.csv"
    write_csv(contrast_path, contrast_rows)

    # Per-arm T/U/N and contrast summary.
    group_rows: dict[tuple, list[dict]] = defaultdict(list)
    for r in contrast_rows:
        group_rows[(r["seed"], r["arm"], r["checkpoint"], r["probe_set"], r["packet_type"], r["slot_mode"])].append(r)
    summary_rows: list[dict] = []
    fields = [
        "full_new_minus_source_T", "full_new_minus_source_U", "full_new_minus_source_N",
        "full_new_minus_source_T_minus_U", "full_new_minus_source_T_minus_N",
        "content_new_minus_source_T", "content_new_minus_source_U", "content_new_minus_source_N",
        "content_new_minus_source_T_minus_U", "content_new_minus_source_T_minus_N",
        "full_source_minus_new_T", "full_source_minus_new_U", "full_source_minus_new_N",
        "full_source_minus_new_T_minus_U", "full_source_minus_new_T_minus_N",
        "content_source_minus_new_T", "content_source_minus_new_U", "content_source_minus_new_N",
        "content_source_minus_new_T_minus_U", "content_source_minus_new_T_minus_N",
    ]
    for (seed, arm, ckpt, pset, ptype, slot), rows in sorted(group_rows.items()):
        rec = {"seed": seed, "arm": arm, "checkpoint": ckpt, "probe_set": pset, "packet_type": ptype, "slot_mode": slot, "n_packets": len(rows)}
        for field in fields:
            xs = [ffloat(r.get(field)) for r in rows]
            rec[f"mean_{field}"] = mean(xs)
            rec[f"se_{field}"] = se(xs)
            rec[f"frac_pos_{field}"] = sum(1 for x in xs if math.isfinite(x) and x > 0) / max(1, sum(1 for x in xs if math.isfinite(x)))
        summary_rows.append(rec)
    summary_path = OUT_DIR / "format_matched_contrast_summary.csv"
    write_csv(summary_path, summary_rows)

    # Paired arm-minus-clean summaries for the central UPDATED_USE T-U quantity.
    by_pair_arm: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for r in contrast_rows:
        key = (r["seed"], r["checkpoint"], r["probe_set"], r["packet_type"], r["slot_mode"], r["packet_id"])
        by_pair_arm[key][r["arm"]] = r
    delta_rows: list[dict] = []
    for key, arms in by_pair_arm.items():
        if "clean" not in arms:
            continue
        seed, ckpt, pset, ptype, slot, pid = key
        c = arms["clean"]
        for arm in ["repeat", "view", "repeat_split", "view_split"]:
            if arm not in arms:
                continue
            a = arms[arm]
            rec = {"seed": seed, "comparison": f"{arm}_minus_clean", "arm": arm, "checkpoint": ckpt,
                   "probe_set": pset, "packet_type": ptype, "slot_mode": slot, "packet_id": pid}
            for field in fields:
                rec[f"delta_{field}"] = ffloat(a.get(field)) - ffloat(c.get(field))
            delta_rows.append(rec)
    delta_path = OUT_DIR / "arm_minus_clean_format_matched_rows.csv"
    write_csv(delta_path, delta_rows)

    dg: dict[tuple, list[dict]] = defaultdict(list)
    for r in delta_rows:
        dg[(r["seed"], r["comparison"], r["checkpoint"], r["probe_set"], r["packet_type"], r["slot_mode"])].append(r)
    delta_summary: list[dict] = []
    for (seed, comp, ckpt, pset, ptype, slot), rows in sorted(dg.items()):
        rec = {"seed": seed, "comparison": comp, "checkpoint": ckpt, "probe_set": pset, "packet_type": ptype, "slot_mode": slot, "n_packets": len(rows)}
        for field in fields:
            dfield = f"delta_{field}"
            xs = [ffloat(r.get(dfield)) for r in rows]
            rec[f"mean_{dfield}"] = mean(xs)
            rec[f"se_{dfield}"] = se(xs)
            rec[f"frac_pos_{dfield}"] = sum(1 for x in xs if math.isfinite(x) and x > 0) / max(1, sum(1 for x in xs if math.isfinite(x)))
        delta_summary.append(rec)
    delta_summary_path = OUT_DIR / "arm_minus_clean_format_matched_summary.csv"
    write_csv(delta_summary_path, delta_summary)

    note_path = OUT_DIR / "relation_arm_tu_note.md"
    note_path.write_text(make_note(summary_rows, delta_summary), encoding="utf-8")

    # Print a concise machine-readable central table.
    central = []
    for r in delta_summary:
        if r["packet_type"] == "UPDATED_USE" and r["probe_set"] == "extended_nontrain":
            central.append({
                "seed": r["seed"],
                "comparison": r["comparison"],
                "n": r["n_packets"],
                "delta_T_minus_U_full": round(ffloat(r["mean_delta_full_new_minus_source_T_minus_U"]), 6),
                "se": round(ffloat(r["se_delta_full_new_minus_source_T_minus_U"]), 6),
                "delta_T_minus_N_full": round(ffloat(r["mean_delta_full_new_minus_source_T_minus_N"]), 6),
                "delta_T_raw_full": round(ffloat(r["mean_delta_full_new_minus_source_T"]), 6),
                "delta_U_raw_full": round(ffloat(r["mean_delta_full_new_minus_source_U"]), 6),
            })
    return {
        "raw_rows": len(raw_rows),
        "contrast_rows": len(contrast_rows),
        "summary_rows": len(summary_rows),
        "delta_rows": len(delta_rows),
        "delta_summary_rows": len(delta_summary),
        "paths": {
            "contrast_rows": rel(contrast_path),
            "summary": rel(summary_path),
            "arm_minus_clean_rows": rel(delta_path),
            "arm_minus_clean_summary": rel(delta_summary_path),
            "note": rel(note_path),
        },
        "central_updated_extended": central,
    }


def make_note(summary_rows: list[dict], delta_summary: list[dict]) -> str:
    def fmt(x: object) -> str:
        v = ffloat(x)
        return "NA" if not math.isfinite(v) else f"{v:+.4f}"

    lines: list[str] = []
    lines.append("# research relation-arm state-margin: T/U/N and format-matched T-U\n\n")
    lines.append(f"Created UTC: {now_utc()}\n\n")
    lines.append("## Why this rerun was necessary\n\n")
    lines.append("research used T-N to subtract neutral phrase preference, but the split arms were themselves trained in a source-alone-plus-query-like format. Their higher N therefore could be a construction-specific neutral-format shift. This note treats the format-matched quantity as primary: on UPDATED_USE packets, T and U both contain source, update sentence, and query frame; the contrast T(new-source)-U(new-source) asks whether the true target update adds more new-state preference than a foreign update does. T, U, and N remain shown side by side so the neutral-format shift is visible rather than hidden.\n\n")
    lines.append("## UPDATED_USE extended_nontrain: arm-minus-CLEAN central quantities\n\n")
    lines.append("| seed | arm minus CLEAN | n | raw T | raw U | T-U | T-N | content T-U |\n")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|\n")
    for r in delta_summary:
        if r.get("packet_type") == "UPDATED_USE" and r.get("probe_set") == "extended_nontrain":
            lines.append(
                f"| {r['seed']} | {r['comparison']} | {r['n_packets']} | "
                f"{fmt(r.get('mean_delta_full_new_minus_source_T'))} | "
                f"{fmt(r.get('mean_delta_full_new_minus_source_U'))} | "
                f"{fmt(r.get('mean_delta_full_new_minus_source_T_minus_U'))} | "
                f"{fmt(r.get('mean_delta_full_new_minus_source_T_minus_N'))} | "
                f"{fmt(r.get('mean_delta_content_new_minus_source_T_minus_U'))} |\n"
            )
    lines.append("\n## Per-arm T/U/N means for UPDATED_USE extended_nontrain\n\n")
    lines.append("| seed | arm | n | T | U | N | T-U | T-N |\n")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|\n")
    arm_order = {"clean": 0, "repeat": 1, "view": 2, "repeat_split": 3, "view_split": 4}
    for r in sorted(summary_rows, key=lambda x: (x.get("seed", ""), arm_order.get(x.get("arm", ""), 99))):
        if r.get("packet_type") == "UPDATED_USE" and r.get("probe_set") == "extended_nontrain":
            lines.append(
                f"| {r['seed']} | {r['arm']} | {r['n_packets']} | "
                f"{fmt(r.get('mean_full_new_minus_source_T'))} | "
                f"{fmt(r.get('mean_full_new_minus_source_U'))} | "
                f"{fmt(r.get('mean_full_new_minus_source_N'))} | "
                f"{fmt(r.get('mean_full_new_minus_source_T_minus_U'))} | "
                f"{fmt(r.get('mean_full_new_minus_source_T_minus_N'))} |\n"
            )
    lines.append("\n## Interpretation boundary\n\n")
    lines.append("This table decides whether the state-margin readout can be used as causal-locality support. If VIEW remains above CLEAN while split arms stay near CLEAN on T-U in both seeds, the result is relation-locality evidence. If T-U collapses or reverses while T-N shows the old pattern, the result is instead a phrase-prior/source-format shift in split training. REPEAT sharing VIEW's sign on T-U would point to a shared second-sentence-use component rather than the exact-copy liability seen on compact changed-form probes.\n")
    return "".join(lines)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--reuse-existing", action="store_true", help="Skip scorer if raw rows already exist")
    args = ap.parse_args()

    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    score_info: dict = {}
    if args.reuse_existing and (OUT_DIR / "raw_scores/raw_TUN_margin_rows.csv").exists():
        score_info = {"raw_dir": rel(OUT_DIR / "raw_scores"), "reused": True}
    else:
        score_info = run_scorer(args.device, args.batch_size)
    contrast_info = compute_contrasts()
    meta = {
        "status": "RELATION_ARM_TU_DONE",
        "created_utc": now_utc(),
        "elapsed_sec": round(time.time() - started, 1),
        "models_requested": list(ARM_MODELS.keys()),
        "score_info": score_info,
        **contrast_info,
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
