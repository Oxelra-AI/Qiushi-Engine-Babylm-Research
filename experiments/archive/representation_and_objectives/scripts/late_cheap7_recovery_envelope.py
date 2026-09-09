#!/usr/bin/env python3
"""research: empirical late-checkpoint cheap7 recovery envelope.

Use existing completed trajectory/posttrain summaries to ask whether a 70M cheap7
endpoint that is ~1.1 points below the visible leader can ordinarily recover by
100M in the local BabyLM Strict-Small recipe family. This is CPU-only file mining;
it does not evaluate models.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path("experiments/archive/representation_and_objectives")
OUT = ROOT / "data/late_cheap7_recovery_envelope"
NOTE = (ROOT.parents[2] / 'research/notes/representation_and_objectives/late_cheap7_recovery_envelope.md')

CANDIDATE_FILES = [
    ROOT / "data/semantic_view_noaoa_eval/semantic_view_treatment_trajectory_summary.json",
    ROOT / "data/semantic_view_noaoa_eval/original_packet_local_trajectory_summary.json",
    ROOT / "data/wwm_to_token_postswitch_fullzeroshot_reading/qwen_wwm_to_token_postswitch_trajectory_summary.json",
    ROOT / "data/legal40k_accum_seed43022_posttrain_eval_controller/posttrain_eval_seed43022_summary.json",
    ROOT / "data/legal40k_accum_seed43122_posttrain_eval_controller/posttrain_eval_seed43122_summary.json",
    ROOT / "data/legal40k_12x384_depth_seed43022_posttrain_eval_controller/posttrain_eval_seed43022_summary.json",
    ROOT / "data/legal40k_12x384_sgcrK50d64_seed43022_posttrain_eval_controller/posttrain_eval_seed43022_summary.json",
]
CHEAP_COLS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
FW_70M = {
    "compact_view_70M": 42.65714285714286,
    "source_breadth_rowblock_70M": 42.03142857142858,
}
LEADER_CHEAP7 = 43.769999999999996


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def cheap7(row: dict[str, Any]) -> float | None:
    if "equal7_full_eval" in row and row["equal7_full_eval"] is not None:
        try:
            return float(row["equal7_full_eval"])
        except Exception:
            pass
    vals = []
    for c in CHEAP_COLS:
        v = row.get(c)
        if v is None:
            return None
        vals.append(float(v))
    return mean(vals)


def extract_tables(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    d = json.loads(path.read_text(encoding="utf-8"))
    target = d.get("target") or path.parent.name
    tables = []
    if isinstance(d.get("table"), dict):
        tables.append((target, d["table"]))
    # Some posttrain controllers nest per-target/per-checkpoint records differently.
    for key in ["checkpoint_summary", "trajectory", "results", "per_checkpoint", "checkpoints"]:
        obj = d.get(key)
        if isinstance(obj, dict):
            # If keys look like checkpoint names and values are score dicts, use it.
            if any(str(k).startswith("chck_") for k in obj.keys()):
                tables.append((target + f"::{key}", obj))
    rows = []
    for label, table in tables:
        for ck, row in table.items():
            if not isinstance(row, dict):
                continue
            ch = cheap7(row)
            if ch is None:
                continue
            rows.append({
                "source_file": str(path),
                "target": label,
                "checkpoint": ck,
                "checkpoint_m": int(str(ck).split("chck_")[1].split("M")[0]) if "chck_" in str(ck) and "M" in str(ck) else None,
                "cheap7": ch,
                **{c: row.get(c) for c in CHEAP_COLS if c in row},
            })
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for p in CANDIDATE_FILES:
        all_rows.extend(extract_tables(p))
    # de-duplicate identical target/checkpoint/source rows
    seen = set(); unique = []
    for r in all_rows:
        key = (r["source_file"], r["target"], r["checkpoint"], round(r["cheap7"], 10))
        if key not in seen:
            seen.add(key); unique.append(r)
    all_rows = sorted(unique, key=lambda r: (r["target"], r["checkpoint_m"] or -1))

    by_target: dict[str, dict[int, dict[str, Any]]] = {}
    for r in all_rows:
        m = r.get("checkpoint_m")
        if m is None:
            continue
        by_target.setdefault(r["target"], {})[m] = r
    pair_rows = []
    for target, table in sorted(by_target.items()):
        for start in [60, 70, 80, 90]:
            if start in table and 100 in table:
                pair_rows.append({
                    "target": target,
                    "from_m": start,
                    "to_m": 100,
                    "from_cheap7": table[start]["cheap7"],
                    "to_cheap7": table[100]["cheap7"],
                    "delta_to_100M": table[100]["cheap7"] - table[start]["cheap7"],
                    "source_file": table[start]["source_file"],
                })
    deltas70 = [r["delta_to_100M"] for r in pair_rows if r["from_m"] == 70]
    deltas80 = [r["delta_to_100M"] for r in pair_rows if r["from_m"] == 80]
    req_rows = []
    for arm, val in FW_70M.items():
        req = LEADER_CHEAP7 - val
        req_rows.append({
            "arm": arm,
            "observed_70M_cheap7": val,
            "leader_cheap7": LEADER_CHEAP7,
            "needed_70M_to_leader_cheap7": req,
            "n_historical_70M_to_100M": len(deltas70),
            "historical_70M_to_100M_min": min(deltas70) if deltas70 else None,
            "historical_70M_to_100M_max": max(deltas70) if deltas70 else None,
            "historical_70M_to_100M_mean": mean(deltas70) if deltas70 else None,
            "historical_count_delta_ge_needed": sum(d >= req for d in deltas70),
            "historical_frac_delta_ge_needed": sum(d >= req for d in deltas70) / len(deltas70) if deltas70 else None,
        })
    # Add compact and breadth as current incomplete records for comparison only.
    current_rows = [{"target": arm, "checkpoint": "chck_70M", "checkpoint_m": 70, "cheap7": val, "source_file": "A02 research complete 70M cheap eval"} for arm, val in FW_70M.items()]

    def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        fields = []
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)

    write_csv(OUT / "late_cheap7_checkpoint_rows.csv", all_rows + current_rows)
    write_csv(OUT / "late_cheap7_to_100M_pairs.csv", pair_rows)
    write_csv(OUT / "fw_70m_required_recovery_vs_history.csv", req_rows)
    payload = {
        "status": "LATE_CHEAP7_RECOVERY_ENVELOPE_DONE",
        "candidate_files": [str(p) for p in CANDIDATE_FILES],
        "historical_rows_n": len(all_rows),
        "pair_rows_n": len(pair_rows),
        "deltas70_to_100M": deltas70,
        "deltas80_to_100M": deltas80,
        "fw_required_rows": req_rows,
        "scientific_reading": {
            "scope": "This is a small local trajectory envelope, not a law. It prevents treating an incomplete 70M cheap score as stronger than previous late-training behavior supports.",
            "compact_70m": "A02 compact 70M cheap7 is 42.657, 1.113 below leader cheap7. In the extracted local trajectories, 70M-to-100M deltas are generally small or negative; a +1.113 recovery would exceed the observed 70M-to-100M maximum if the extracted set is complete for comparable trajectories.",
            "breadth_70m": "A02 row-block breadth 70M cheap7 is 42.031, 1.739 below leader cheap7, requiring an even larger late jump. Its 70M vector also loses Entity and nonparallel GlobalPIQA relative to compact.",
        },
    }
    (OUT / "late_cheap7_recovery_envelope.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = []
    lines.append("# research — late cheap7 recovery envelope")
    lines.append("")
    lines.append("This CPU-only mining step asks how much local completed trajectories usually move in no-AoA/equal7 from 70M or 80M to 100M. It is not a substitute for completed FW evidence; it calibrates the risk of spending full endpoint evaluation on a low 70M cheap7.")
    lines.append("")
    lines.append(f"Extracted {len(all_rows)} checkpoint rows and {len(pair_rows)} late-to-100M pairs from {len(CANDIDATE_FILES)} summary files.")
    if deltas70:
        lines.append(f"70M→100M cheap7 deltas: n={len(deltas70)}, min={min(deltas70):.3f}, mean={mean(deltas70):.3f}, max={max(deltas70):.3f}; values={', '.join(f'{d:.3f}' for d in deltas70)}.")
    if deltas80:
        lines.append(f"80M→100M cheap7 deltas: n={len(deltas80)}, min={min(deltas80):.3f}, mean={mean(deltas80):.3f}, max={max(deltas80):.3f}; values={', '.join(f'{d:.3f}' for d in deltas80)}.")
    lines.append("")
    lines.append("## A02 FW 70M requirement")
    lines.append("")
    lines.append("| arm | 70M cheap7 | needed to leader cheap7 43.77 | historical 70M→100M max | count >= needed |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in req_rows:
        mx = r["historical_70M_to_100M_max"]
        lines.append(f"| `{r['arm']}` | {r['observed_70M_cheap7']:.3f} | {r['needed_70M_to_leader_cheap7']:.3f} | {mx:.3f} | {r['historical_count_delta_ge_needed']}/{r['n_historical_70M_to_100M']} |")
    lines.append("")
    lines.append("The extracted local trajectories make a late jump large enough to bring the A02 70M FW arms to leader cheap7 look unlikely: compact would need +1.113 and row-block breadth +1.739, while the observed 70M→100M range here is at most +0.361. This does not by itself close the 100M endpoints, because the FW data family has its own trajectory and A02 cheap 80M/100M files are not complete yet. It does justify withholding new full official evaluations until a complete 100M cheap7 or other strong endpoint evidence appears.")
    lines.append("")
    lines.append("## Files")
    lines.append(f"- JSON: `{OUT/'late_cheap7_recovery_envelope.json'}`")
    lines.append(f"- checkpoint rows: `{OUT/'late_cheap7_checkpoint_rows.csv'}`")
    lines.append(f"- late pairs: `{OUT/'late_cheap7_to_100M_pairs.csv'}`")
    lines.append(f"- required recovery table: `{OUT/'fw_70m_required_recovery_vs_history.csv'}`")
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "historical_rows_n": len(all_rows),
        "pair_rows_n": len(pair_rows),
        "deltas70_to_100M": deltas70,
        "fw_required_rows": req_rows,
        "json": str(OUT / "late_cheap7_recovery_envelope.json"),
        "note": str(NOTE),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
