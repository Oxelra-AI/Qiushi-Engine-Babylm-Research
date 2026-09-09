#!/usr/bin/env python3
"""research: synthesize the mature (100M) GlobalPIQA margin readout for the two
completed FW arms (compact same-proposition recurrence vs row-block whole-
sentence source-breadth), and reconcile with the 70M readouts and research prior
endpoints.

Scientific purpose
-------------------
The scientific question is whether the mature FW arms show a *combination* of
preserved broad capability and improved deep GlobalPIQA hard-row preference. The
70M readout showed row-block breadth trades broad capability (Entity, GlobalPIQA
nonparallel) for a small hard-parallel-margin softening. The analysis tests whether
that tradeoff persists, worsens, or resolves at the final 100M checkpoint, which
is the checkpoint any submission would use.

CPU-only, reads existing JSON only. No training, no submission-scorer tuning.
"""
from __future__ import annotations

import json
from pathlib import Path

WS = Path("experiments/archive/representation_and_objectives")
MARGIN_JSON = WS / "data/fw_globalpiqa_margin_reader/globalpiqa_margin_reader_results.json"
OUT_DIR = WS / "data/fw_100m_globalpiqa_mature_synthesis"
NOTE = (WS.parents[2] / 'research/notes/representation_and_objectives/fw_100m_globalpiqa_mature_synthesis.md')

# GlobalPIQA official aggregate is the mean of the two subtask accuracies.
def globalpiqa(parallel_acc: float, nonparallel_acc: float) -> float:
    return (parallel_acc + nonparallel_acc) / 2.0


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    d = json.loads(MARGIN_JSON.read_text())
    targets = d["targets"]

    rows = []
    for tkey, res in targets.items():
        par = res["modes"]["parallel"]["summary"]
        non = res["modes"]["nonparallel"]["summary"]
        aw = par.get("always_wrong_subset") or {}
        rc = par["correct_rank_counts"]
        rows.append({
            "arm": tkey,
            "model_path": res["model_root"],
            "revision": res["revision"],
            "parallel_acc": par["accuracy"],
            "nonparallel_acc": non["accuracy"],
            "globalpiqa_agg": globalpiqa(par["accuracy"], non["accuracy"]),
            "parallel_rank1": rc.get("1", 0),
            "parallel_rank2": rc.get("2", 0),
            "parallel_rank3": rc.get("3", 0),
            "parallel_rank4": rc.get("4", 0),
            "hard52_acc": aw.get("accuracy"),
            "hard52_rank1": (aw.get("correct_rank_counts") or {}).get("1", 0),
            "hard52_mean_top_minus_correct": aw.get("mean_top_minus_correct"),
            "hard52_small_le_0p50": aw.get("small_wrong_margin_le_0p50_nats"),
            "nonparallel_small_le_0p50": non["all_rows_margin_summary"].get("small_wrong_margin_le_0p50_nats"),
        })

    # Order compact then breadth for deltas.
    def get(arm_sub):
        for r in rows:
            if arm_sub in r["arm"]:
                return r
        return None

    compact = get("compact")
    breadth = get("breadth")

    deltas = {}
    if compact and breadth:
        for k in ["parallel_acc", "nonparallel_acc", "globalpiqa_agg", "parallel_rank1",
                  "hard52_acc", "hard52_rank1", "hard52_mean_top_minus_correct", "hard52_small_le_0p50"]:
            cv, bv = compact.get(k), breadth.get(k)
            if isinstance(cv, (int, float)) and isinstance(bv, (int, float)):
                deltas[f"breadth_minus_compact_{k}"] = bv - cv

    # 70M reference (from margin synthesis / cheap scores) for the same two arms.
    ref_70m = {
        "compact": {"parallel_acc": 26.21359223300971, "nonparallel_acc": 51.0,
                    "hard52_rank1": 2, "hard52_mean_top_minus_correct": 1.7031002964455157,
                    "globalpiqa_agg": globalpiqa(26.21359223300971, 51.0)},
        "breadth_rowblock": {"parallel_acc": 26.21359223300971, "nonparallel_acc": 47.0,
                             "hard52_rank1": 1, "hard52_mean_top_minus_correct": 1.4763328583714115,
                             "globalpiqa_agg": globalpiqa(26.21359223300971, 47.0)},
    }
    within_arm_70m_to_100m = {}
    if compact:
        within_arm_70m_to_100m["compact"] = {
            "parallel_70to100": compact["parallel_acc"] - ref_70m["compact"]["parallel_acc"],
            "nonparallel_70to100": compact["nonparallel_acc"] - ref_70m["compact"]["nonparallel_acc"],
            "globalpiqa_70to100": compact["globalpiqa_agg"] - ref_70m["compact"]["globalpiqa_agg"],
            "hard52_mean_margin_70to100": compact["hard52_mean_top_minus_correct"] - ref_70m["compact"]["hard52_mean_top_minus_correct"],
        }
    if breadth:
        within_arm_70m_to_100m["breadth_rowblock"] = {
            "parallel_70to100": breadth["parallel_acc"] - ref_70m["breadth_rowblock"]["parallel_acc"],
            "nonparallel_70to100": breadth["nonparallel_acc"] - ref_70m["breadth_rowblock"]["nonparallel_acc"],
            "globalpiqa_70to100": breadth["globalpiqa_agg"] - ref_70m["breadth_rowblock"]["globalpiqa_agg"],
            "hard52_mean_margin_70to100": breadth["hard52_mean_top_minus_correct"] - ref_70m["breadth_rowblock"]["hard52_mean_top_minus_correct"],
        }

    out = {
        "status": "FW_100M_GLOBALPIQA_MATURE_SYNTHESIS_DONE",
        "source_margin_json": str(MARGIN_JSON),
        "rows_100m": rows,
        "breadth_minus_compact_100m": deltas,
        "ref_70m": ref_70m,
        "within_arm_70m_to_100m": within_arm_70m_to_100m,
        "interpretation": [
            "Mature 100M GlobalPIQA is a within-checkpoint tradeoff, not a joint improvement.",
            "Row-block breadth raises hard parallel accuracy and softens hard52 margins but lowers nonparallel accuracy by a larger margin, so its GlobalPIQA aggregate is lower than compact.",
            "Compact preserves broad GlobalPIQA (higher nonparallel) but keeps the deep-rank parallel weakness.",
            "This reproduces the 70M direction at the final checkpoint: independent coverage shifts probability mass toward hard parallel structure while degrading broad capability; the proposed 'combination' of preserved breadth plus improved relational ranks does not appear.",
        ],
    }
    (OUT_DIR / "fw_100m_globalpiqa_mature_synthesis.json").write_text(json.dumps(out, indent=2) + "\n")

    lines = ["# research — mature (100M) FW GlobalPIQA margin synthesis", "",
             "Reads the research wrapper 100M readout for the two completed A02 arms.", ""]
    lines.append("## 100M GlobalPIQA split and hard-row ranks")
    lines.append("")
    lines.append("| arm | parallel | nonparallel | GlobalPIQA agg | par rank1 | par rank4 | hard52 acc | hard52 rank1 | hard52 mean margin (nats) |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(f"| {r['arm']} | {r['parallel_acc']:.2f} | {r['nonparallel_acc']:.2f} | {r['globalpiqa_agg']:.2f} | {r['parallel_rank1']} | {r['parallel_rank4']} | {r['hard52_acc']:.2f} | {r['hard52_rank1']} | {r['hard52_mean_top_minus_correct']:.3f} |")
    lines.append("")
    lines.append("## Row-block breadth minus compact (100M)")
    lines.append("")
    for k, v in deltas.items():
        lines.append(f"- {k}: {v:+.4f}")
    lines.append("")
    lines.append("## Within-arm 70M→100M movement")
    lines.append("")
    for arm, m in within_arm_70m_to_100m.items():
        lines.append(f"- {arm}: parallel {m['parallel_70to100']:+.2f}, nonparallel {m['nonparallel_70to100']:+.2f}, GlobalPIQA agg {m['globalpiqa_70to100']:+.2f}, hard52 mean margin {m['hard52_mean_margin_70to100']:+.3f}")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    for s in out["interpretation"]:
        lines.append(f"- {s}")
    lines.append("")
    lines.append(f"JSON: `{OUT_DIR / 'fw_100m_globalpiqa_mature_synthesis.json'}`")
    NOTE.write_text("\n".join(lines) + "\n")

    print(json.dumps({"status": out["status"],
                      "compact_globalpiqa": compact["globalpiqa_agg"] if compact else None,
                      "breadth_globalpiqa": breadth["globalpiqa_agg"] if breadth else None,
                      "breadth_minus_compact_parallel": deltas.get("breadth_minus_compact_parallel_acc"),
                      "breadth_minus_compact_nonparallel": deltas.get("breadth_minus_compact_nonparallel_acc"),
                      "breadth_minus_compact_globalpiqa": deltas.get("breadth_minus_compact_globalpiqa_agg"),
                      "json": str(OUT_DIR / "fw_100m_globalpiqa_mature_synthesis.json"),
                      "note": str(NOTE)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
