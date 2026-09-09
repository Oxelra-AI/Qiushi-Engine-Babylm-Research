#!/usr/bin/env python3
"""Summarize research balanced paired binding run.

Reads per-arm summaries from binding_factorial_balanced_pairbatch and
writes a compact cross-arm table.  It maps the saved field `gating_count` to
`pairwise_excess_count`: joint_correct minus the independent product of the two
marginal half successes.
"""
from __future__ import annotations
import csv, json, pathlib, time
from typing import Any

ROOT = pathlib.Path.cwd()
OUT = ROOT / "experiments/archive/relation_learning/data/binding_factorial_balanced_pairbatch"
ARMS = ["answer_clean", "uniform_wwm", "answer_corrupt_update_state"]


def rel(p: pathlib.Path) -> str:
    try: return str(p.relative_to(ROOT))
    except Exception: return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8"); return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def main() -> None:
    rows=[]; arms={}
    for arm in ARMS:
        p = OUT / arm / "summary.json"
        if not p.exists():
            arms[arm] = {"status": "missing", "summary": rel(p)}
            continue
        s = read_json(p)
        arms[arm] = {
            "status": s.get("status"),
            "summary": rel(p),
            "created_utc": s.get("created_utc"),
            "updates_seen": s.get("updates_seen"),
            "n_train_tokenized": s.get("n_train_tokenized"),
            "n_heldout_tokenized": s.get("n_heldout_tokenized"),
            "n_train_pair_groups": s.get("n_train_pair_groups"),
            "n_binding_pairs": s.get("n_binding_pairs"),
            "loaded_class": s.get("model_identity", {}).get("loaded_class"),
            "total_params_loaded": s.get("model_identity", {}).get("total_params_loaded"),
            "adapter_params_loaded": s.get("model_identity", {}).get("adapter_params_loaded"),
            "adapter_trainable_params": s.get("freeze_stats", {}).get("adapter_trainable_params"),
            "train_counts": s.get("plan", {}).get("train_counts"),
        }
        for tr in s.get("trajectory", []):
            epoch = int(tr.get("epoch", -1))
            for subset, v in tr.get("eval", {}).get("gating", {}).items():
                rows.append({
                    "arm": arm, "epoch": epoch, "subset": subset,
                    "n": v.get("n"), "joint_correct": v.get("joint_correct"),
                    "a_correct": v.get("a_correct"), "b_correct": v.get("b_correct"),
                    "expected_joint_independent_count": v.get("expected_joint_independent_count"),
                    "pairwise_excess_count": v.get("gating_count"),
                    "pairwise_excess_frac": v.get("gating_frac"),
                    "mean_a_margin": v.get("mean_a_margin"),
                    "mean_b_margin": v.get("mean_b_margin"),
                    "mean_joint_min_margin": v.get("mean_joint_min_margin"),
                })
    write_csv(OUT / "trajectory_pairwise_excess.csv", rows)
    summary={"status":"BALANCED_PAIRBATCH_SUMMARY","created_utc":now(),"out_root":rel(OUT),"arms":arms,"trajectory_csv":rel(OUT/"trajectory_pairwise_excess.csv")}
    (OUT/"summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    lines=["# research balanced paired binding run\n\n", "`pairwise_excess_count = joint_correct - a_correct*b_correct/n`.\n\n"]
    for subset in ["all_pairs", "updated_entity_in_source_119", "both_entities_in_source_115"]:
        lines += [f"## Epoch 20: {subset}\n\n", "| arm | joint | A source-retain | B update-select | expected joint | pairwise excess | mean A margin | mean B margin |\n", "|---|---:|---:|---:|---:|---:|---:|---:|\n"]
        for arm in ARMS:
            match=[r for r in rows if r["arm"]==arm and r["epoch"]==20 and r["subset"]==subset]
            if not match: continue
            r=match[0]
            lines.append(f"| {arm} | {int(r['joint_correct'])}/{int(r['n'])} | {int(r['a_correct'])}/{int(r['n'])} | {int(r['b_correct'])}/{int(r['n'])} | {float(r['expected_joint_independent_count']):.2f} | {float(r['pairwise_excess_count']):.2f} | {float(r['mean_a_margin']):.3f} | {float(r['mean_b_margin']):.3f} |\n")
        lines.append("\n")
    (OUT/"summary.md").write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "summary": rel(OUT/"summary.json"), "md": rel(OUT/"summary.md"), "csv": rel(OUT/"trajectory_pairwise_excess.csv")}, indent=2), flush=True)

if __name__ == "__main__":
    main()
