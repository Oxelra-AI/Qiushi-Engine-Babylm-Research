#!/usr/bin/env python3
"""Synthesize GlobalPIQA 70M margin movement for FW arms."""
from __future__ import annotations

import csv
import json
import statistics as stats
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path("experiments/archive/representation_and_objectives")
MARGIN_DIR = ROOT / "data/fw_70m_globalpiqa_margin_reader"
ROW_OVERLAP = ROOT / "data/fw_70m_globalpiqa_row_overlap/fw_70m_globalpiqa_row_overlap.json"
SYN = ROOT / "data/globalpiqa_margin_synthesis/globalpiqa_margin_summary.csv"
OUT = ROOT / "data/fw_70m_globalpiqa_margin_synthesis"
NOTE = (ROOT.parents[2] / 'research/notes/representation_and_objectives/fw_70m_globalpiqa_margin_synthesis.md')

TARGETS = {
    "compact_view_70M": MARGIN_DIR / "a02_fw_compact_70M_seed43022_margins.json",
    "source_breadth_rowblock_70M": MARGIN_DIR / "a02_fw_breadth_rowblock_70M_seed43022_margins.json",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = []
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def read_historical_parallel() -> list[dict[str, str]]:
    if not SYN.exists():
        return []
    with SYN.open(newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r.get("mode") == "parallel"]


def row_map(d: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = d["modes"]["parallel"]["rows"]
    return {r["example_id"]: r for r in rows}


def summary_from_margin(key: str, d: dict[str, Any]) -> dict[str, Any]:
    s = d["modes"]["parallel"]["summary"]
    aw = s["always_wrong_subset"]
    allm = s["all_rows_margin_summary"]
    rank = s["correct_rank_counts"]
    awrank = aw["correct_rank_counts"]
    return {
        "arm": key,
        "official_reproduced_parallel_accuracy": s["accuracy"],
        "all_rank1": rank.get("1", 0),
        "all_rank2": rank.get("2", 0),
        "all_rank3": rank.get("3", 0),
        "all_rank4": rank.get("4", 0),
        "all_mean_top_minus_correct": allm["mean_top_minus_correct"],
        "all_median_top_minus_correct": allm["median_top_minus_correct"],
        "all_small_wrong_le_0p25": allm["small_wrong_margin_le_0p25_nats"],
        "all_small_wrong_le_0p50": allm["small_wrong_margin_le_0p50_nats"],
        "hard52_accuracy": aw["accuracy"],
        "hard52_rank1": awrank.get("1", 0),
        "hard52_rank2": awrank.get("2", 0),
        "hard52_rank3": awrank.get("3", 0),
        "hard52_rank4": awrank.get("4", 0),
        "hard52_mean_top_minus_correct": aw["mean_top_minus_correct"],
        "hard52_median_top_minus_correct": aw["median_top_minus_correct"],
        "hard52_small_wrong_le_0p25": aw["small_wrong_margin_le_0p25_nats"],
        "hard52_small_wrong_le_0p50": aw["small_wrong_margin_le_0p50_nats"],
        "model_root": d["model_root"],
        "revision": d["revision"],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = {k: load_json(p) for k, p in TARGETS.items()}
    rowoverlap = load_json(ROW_OVERLAP)
    summaries = [summary_from_margin(k, d) for k, d in data.items()]
    write_csv(OUT / "fw_70m_globalpiqa_margin_summary.csv", summaries)

    compact = row_map(data["compact_view_70M"])
    breadth = row_map(data["source_breadth_rowblock_70M"])
    hard_ids = set(rowoverlap["subset_stats"]["parallel_step107_always_wrong_all10"]["compact_view_70M_choice_counts"] and [])
    # Use the 52-row set from the row-overlap movement table to avoid relying on choice-count keys.
    hard_set = set()
    with (ROOT / "data/globalpiqa_parallel_anatomy/globalpiqa_parallel_cross_endpoint_rows.csv").open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if int(r["n_ok"]) == 0:
                hard_set.add(r["example_id"])
    row_deltas = []
    for eid in sorted(set(compact) & set(breadth)):
        a = compact[eid]; b = breadth[eid]
        subset = "hard52" if eid in hard_set else "other51"
        rank_delta = int(b["correct_rank"]) - int(a["correct_rank"])
        margin_delta = float(b["top_minus_correct"]) - float(a["top_minus_correct"])
        row_deltas.append({
            "example_id": eid,
            "subset": subset,
            "label": a["label"],
            "compact_correct": a["correct"],
            "breadth_correct": b["correct"],
            "compact_rank": a["correct_rank"],
            "breadth_rank": b["correct_rank"],
            "rank_delta_breadth_minus_compact": rank_delta,
            "rank_movement": "breadth_better" if rank_delta < 0 else "breadth_worse" if rank_delta > 0 else "same_rank",
            "compact_top_minus_correct": a["top_minus_correct"],
            "breadth_top_minus_correct": b["top_minus_correct"],
            "margin_delta_breadth_minus_compact": margin_delta,
            "margin_movement": "breadth_lower" if margin_delta < -1e-9 else "breadth_higher" if margin_delta > 1e-9 else "same_margin",
            "prompt": a["prompt"],
            "compact_choice": a["choice"],
            "breadth_choice": b["choice"],
            "completion_label": a["completions"][int(a["label"])] if a.get("completions") else "",
            "compact_top_completion": a["completions"][int(a["choice"])] if a.get("completions") else "",
            "breadth_top_completion": b["completions"][int(b["choice"])] if b.get("completions") else "",
        })
    write_csv(OUT / "fw_70m_globalpiqa_margin_row_deltas.csv", row_deltas)

    subset_rows = []
    for subset in ["all103", "hard52", "other51"]:
        rows = row_deltas if subset == "all103" else [r for r in row_deltas if r["subset"] == subset]
        n = len(rows)
        margins = [float(r["margin_delta_breadth_minus_compact"]) for r in rows]
        rank_ctr = Counter(r["rank_movement"] for r in rows)
        margin_ctr = Counter(r["margin_movement"] for r in rows)
        comp_ok = sum(bool(r["compact_correct"]) for r in rows)
        br_ok = sum(bool(r["breadth_correct"]) for r in rows)
        subset_rows.append({
            "subset": subset,
            "n": n,
            "compact_correct": comp_ok,
            "breadth_correct": br_ok,
            "breadth_minus_compact_correct": br_ok - comp_ok,
            "breadth_better_rank_rows": rank_ctr.get("breadth_better", 0),
            "same_rank_rows": rank_ctr.get("same_rank", 0),
            "breadth_worse_rank_rows": rank_ctr.get("breadth_worse", 0),
            "breadth_lower_margin_rows": margin_ctr.get("breadth_lower", 0),
            "breadth_higher_margin_rows": margin_ctr.get("breadth_higher", 0),
            "mean_margin_delta_breadth_minus_compact": sum(margins) / n if n else None,
            "median_margin_delta_breadth_minus_compact": stats.median(margins) if margins else None,
        })
    write_csv(OUT / "fw_70m_globalpiqa_margin_subset_deltas.csv", subset_rows)

    historical = read_historical_parallel()
    hist_rows = []
    for r in historical:
        hist_rows.append({
            "source": "previous_endpoint",
            "target": r["target"],
            "parallel_accuracy": r["accuracy"],
            "hard52_mean_top_minus_correct": r["always_wrong_mean_top_minus_correct"],
            "hard52_rank2": r["always_wrong_rank2"],
            "hard52_rank3": r["always_wrong_rank3"],
            "hard52_rank4": r["always_wrong_rank4"],
        })
    for s in summaries:
        hist_rows.append({
            "source": "A02_70M",
            "target": s["arm"],
            "parallel_accuracy": s["official_reproduced_parallel_accuracy"],
            "hard52_mean_top_minus_correct": s["hard52_mean_top_minus_correct"],
            "hard52_rank2": s["hard52_rank2"],
            "hard52_rank3": s["hard52_rank3"],
            "hard52_rank4": s["hard52_rank4"],
        })
    write_csv(OUT / "fw_70m_vs_hard_margin_reference.csv", hist_rows)

    hard_summary = next(r for r in subset_rows if r["subset"] == "hard52")
    all_summary = next(r for r in subset_rows if r["subset"] == "all103")
    payload = {
        "status": "FW_70M_MARGIN_SYNTHESIS_DONE",
        "input_margin_jsons": {k: str(p) for k, p in TARGETS.items()},
        "row_overlap_json": str(ROW_OVERLAP),
        "implementation_repair": "The first margin wrapper used the local hf_model parent plus revision=chck_70M, but that parent also contains final weights. The repaired wrapper points model_root directly to hf_model/chck_70M with revision=main; the final margin summaries reproduce the official 70M parallel score of 26.2136 for both arms.",
        "arm_summaries": summaries,
        "subset_deltas": subset_rows,
        "historical_hard_margin_reference": hist_rows,
        "scientific_reading": {
            "accuracy": "At 70M, compact and row-block breadth tie on official GlobalPIQA_parallel accuracy at 26.21; on the research 52-row prior hard set compact gets 2/52 and breadth gets 1/52.",
            "rank_and_margin": "Despite no accuracy gain, breadth shifts some hard-row probability mass toward the correct option: hard52 mean top-minus-correct drops from compact 1.703 to breadth 1.476 nats, hard52 rank4 count drops 21 to 11, and hard52 <=0.50-nat near rows rise 6 to 9. This is still far from solving the parallel weakness because only one hard row is correct.",
            "overall_vector": "Row-block breadth's softer hard-row margins coexist with worse 70M cheap7, especially Entity and nonparallel GlobalPIQA. The signal is mechanistically interesting but too weak to support an expensive GlobalPIQA-specialized training route by itself.",
        },
    }
    (OUT / "fw_70m_globalpiqa_margin_synthesis.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = []
    lines.append("# research — A02 FW 70M GlobalPIQA margin synthesis")
    lines.append("")
    lines.append("## Implementation repair")
    lines.append("")
    lines.append("The first wrapper invocation passed the local A02 `hf_model` parent plus `revision=chck_70M`. That parent also contains final weights, so the first run did not faithfully read the 70M checkpoint. The repaired wrapper points directly at `hf_model/chck_70M` with `revision=main`. The final all-option scores reproduce the official 70M GlobalPIQA_parallel value 26.2136 for both arms.")
    lines.append("")
    lines.append("## Arm-level parallel margins")
    lines.append("")
    lines.append("| arm | parallel acc | rank1/2/3/4 | all mean top-correct | hard52 acc | hard52 rank1/2/3/4 | hard52 mean top-correct | hard52 <=0.50 nats |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for s in summaries:
        lines.append(f"| `{s['arm']}` | {s['official_reproduced_parallel_accuracy']:.2f} | {s['all_rank1']}/{s['all_rank2']}/{s['all_rank3']}/{s['all_rank4']} | {s['all_mean_top_minus_correct']:.3f} | {s['hard52_accuracy']:.2f} | {s['hard52_rank1']}/{s['hard52_rank2']}/{s['hard52_rank3']}/{s['hard52_rank4']} | {s['hard52_mean_top_minus_correct']:.3f} | {s['hard52_small_wrong_le_0p50']} |")
    lines.append("")
    lines.append("## Compact versus row-block breadth movement")
    lines.append("")
    lines.append("| subset | n | compact correct | breadth correct | breadth-compact correct | breadth better/same/worse rank | breadth lower/higher margin | mean margin delta | median margin delta |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in subset_rows:
        lines.append(f"| `{r['subset']}` | {r['n']} | {r['compact_correct']} | {r['breadth_correct']} | {r['breadth_minus_compact_correct']} | {r['breadth_better_rank_rows']}/{r['same_rank_rows']}/{r['breadth_worse_rank_rows']} | {r['breadth_lower_margin_rows']}/{r['breadth_higher_margin_rows']} | {r['mean_margin_delta_breadth_minus_compact']:.3f} | {r['median_margin_delta_breadth_minus_compact']:.3f} |")
    lines.append("")
    lines.append("## Scientific reading")
    lines.append("")
    lines.append("At 70M, row-block breadth does not improve official GlobalPIQA_parallel accuracy: both arms are 26.21, and on the research hard52 set compact is 2/52 while breadth is 1/52. This blocks any interpretation that the row-block breadth EWoK gain has already solved the load-bearing GlobalPIQA weakness.")
    lines.append("")
    lines.append("However, the all-option margins reveal a softer movement than exact accuracy: breadth lowers the hard52 mean top-minus-correct margin from 1.703 to 1.476 nats, reduces hard52 rank4 from 21 to 11, and increases hard52 rows within 0.50 nats from 6 to 9. This is mechanistically meaningful as a weak probability-mass shift toward conditional physical/spatial/temporal answers, but it is far too small for SOTA and coexists with worse 70M cheap7 through Entity and nonparallel GlobalPIQA losses.")
    lines.append("")
    lines.append("The next evidence should therefore remain the complete 100M cheap/full vector when available. If 100M cheap7 remains far below the leader, the FW row-block breadth route should not consume full official evaluation just because it softened margins. If a 100M arm is near range, the same direct-checkpoint margin wrapper should be run on 100M before interpreting GlobalPIQA movement.")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append(f"- JSON: `{OUT/'fw_70m_globalpiqa_margin_synthesis.json'}`")
    lines.append(f"- arm summary CSV: `{OUT/'fw_70m_globalpiqa_margin_summary.csv'}`")
    lines.append(f"- subset delta CSV: `{OUT/'fw_70m_globalpiqa_margin_subset_deltas.csv'}`")
    lines.append(f"- row delta CSV: `{OUT/'fw_70m_globalpiqa_margin_row_deltas.csv'}`")
    lines.append(f"- historical hard-row reference CSV: `{OUT/'fw_70m_vs_hard_margin_reference.csv'}`")
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": payload["status"],
        "hard52": hard_summary,
        "all103": all_summary,
        "json": str(OUT / "fw_70m_globalpiqa_margin_synthesis.json"),
        "note": str(NOTE),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
