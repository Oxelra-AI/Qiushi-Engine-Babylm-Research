#!/usr/bin/env python3
"""research: integrate the research format-matched state-margin T-U repair.

Reads the research relation-arm T/U/N summaries, writes compact central tables,
and writes a research note that supersedes the research T-N-only interpretation.
The scientific object is the UPDATED_USE state-margin contrast
T(new-source)-U(new-source), where T and U both carry a source, an update, and a
query/use frame, avoiding the split-arm source-alone neutral-format anchor.
"""
from __future__ import annotations

import csv
import json
import math
import pathlib
import time
from collections import defaultdict
from typing import Any

ROOT = pathlib.Path.cwd()
IN_DIR = ROOT / "experiments/archive/relation_learning/data/relation_arm_state_margin_tu"
SUMMARY = IN_DIR / "format_matched_contrast_summary.csv"
DELTA = IN_DIR / "arm_minus_clean_format_matched_summary.csv"
OUT_DIR = ROOT / "experiments/archive/relation_learning/data/relation_arm_tu_integration"
NOTE = ROOT / "research/notes/relation_learning/relation_arm_state_margin_tu_integration.md"

ARMS = ["clean", "repeat", "view", "repeat_split", "view_split"]
COMPS = ["repeat_minus_clean", "view_minus_clean", "repeat_split_minus_clean", "view_split_minus_clean"]


def fnum(x: str | float | int | None) -> float:
    if x is None or x == "":
        return float("nan")
    return float(x)


def fmt(x: float, digits: int = 4) -> str:
    if not math.isfinite(x):
        return "NA"
    return f"{x:+.{digits}f}"


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summ = read_csv(SUMMARY)
    delta = read_csv(DELTA)

    per_arm: list[dict[str, Any]] = []
    for r in summ:
        if r.get("probe_set") != "extended_nontrain" or r.get("packet_type") != "UPDATED_USE" or r.get("slot_mode") != "template":
            continue
        per_arm.append({
            "seed": r["seed"],
            "arm": r["arm"],
            "n": int(r["n_packets"]),
            "T": fnum(r["mean_full_new_minus_source_T"]),
            "U": fnum(r["mean_full_new_minus_source_U"]),
            "N": fnum(r["mean_full_new_minus_source_N"]),
            "T_minus_U": fnum(r["mean_full_new_minus_source_T_minus_U"]),
            "se_T_minus_U": fnum(r["se_full_new_minus_source_T_minus_U"]),
            "T_minus_N": fnum(r["mean_full_new_minus_source_T_minus_N"]),
            "content_T_minus_U": fnum(r["mean_content_new_minus_source_T_minus_U"]),
        })
    per_arm.sort(key=lambda r: (r["seed"], ARMS.index(r["arm"]) if r["arm"] in ARMS else 99))

    central: list[dict[str, Any]] = []
    for r in delta:
        if r.get("probe_set") != "extended_nontrain" or r.get("packet_type") != "UPDATED_USE" or r.get("slot_mode") != "template":
            continue
        dtu = fnum(r["mean_delta_full_new_minus_source_T_minus_U"])
        se = fnum(r["se_delta_full_new_minus_source_T_minus_U"])
        z = dtu / se if se and math.isfinite(se) else float("nan")
        central.append({
            "seed": r["seed"],
            "comparison": r["comparison"],
            "n": int(r["n_packets"]),
            "delta_T_minus_U_full": dtu,
            "se_delta_T_minus_U_full": se,
            "z_delta_T_minus_U_full": z,
            "delta_T_minus_N_full": fnum(r["mean_delta_full_new_minus_source_T_minus_N"]),
            "delta_T_raw_full": fnum(r["mean_delta_full_new_minus_source_T"]),
            "delta_U_raw_full": fnum(r["mean_delta_full_new_minus_source_U"]),
            "delta_T_minus_U_content": fnum(r["mean_delta_content_new_minus_source_T_minus_U"]),
        })
    central.sort(key=lambda r: (r["seed"], COMPS.index(r["comparison"]) if r["comparison"] in COMPS else 99))

    # Pair the locality magnitudes directly: local arm minus its split counterpart on T-U.
    by_seed_comp = {(r["seed"], r["comparison"]): r for r in central}
    local_minus_split: list[dict[str, Any]] = []
    for seed in sorted({r["seed"] for r in central}):
        rep = by_seed_comp[(seed, "repeat_minus_clean")]
        reps = by_seed_comp[(seed, "repeat_split_minus_clean")]
        view = by_seed_comp[(seed, "view_minus_clean")]
        views = by_seed_comp[(seed, "view_split_minus_clean")]
        local_minus_split.append({
            "seed": seed,
            "contrast": "repeat_minus_repeat_split",
            "delta_T_minus_U_full": rep["delta_T_minus_U_full"] - reps["delta_T_minus_U_full"],
            "delta_T_minus_N_full": rep["delta_T_minus_N_full"] - reps["delta_T_minus_N_full"],
            "note": "SE not recomputed here; use row-level files for inferential intervals",
        })
        local_minus_split.append({
            "seed": seed,
            "contrast": "view_minus_view_split",
            "delta_T_minus_U_full": view["delta_T_minus_U_full"] - views["delta_T_minus_U_full"],
            "delta_T_minus_N_full": view["delta_T_minus_N_full"] - views["delta_T_minus_N_full"],
            "note": "SE not recomputed here; use row-level files for inferential intervals",
        })
        local_minus_split.append({
            "seed": seed,
            "contrast": "view_minus_repeat",
            "delta_T_minus_U_full": view["delta_T_minus_U_full"] - rep["delta_T_minus_U_full"],
            "delta_T_minus_N_full": view["delta_T_minus_N_full"] - rep["delta_T_minus_N_full"],
            "note": "VIEW's extra aligned-restatement state-use effect over exact recurrence",
        })

    write_csv(OUT_DIR / "central_updated_extended_arm_minus_clean.csv", central,
              ["seed", "comparison", "n", "delta_T_minus_U_full", "se_delta_T_minus_U_full", "z_delta_T_minus_U_full", "delta_T_minus_N_full", "delta_T_raw_full", "delta_U_raw_full", "delta_T_minus_U_content"])
    write_csv(OUT_DIR / "per_arm_updated_extended_tun.csv", per_arm,
              ["seed", "arm", "n", "T", "U", "N", "T_minus_U", "se_T_minus_U", "T_minus_N", "content_T_minus_U"])
    write_csv(OUT_DIR / "local_minus_split_updated_extended.csv", local_minus_split,
              ["seed", "contrast", "delta_T_minus_U_full", "delta_T_minus_N_full", "note"])

    # Build note.
    lines: list[str] = []
    lines.append("# research integration: format-matched state-margin T-U repair\n\n")
    lines.append(f"Created UTC: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n\n")
    lines.append("## What the research repair establishes\n\n")
    lines.append("research correctly noticed that state-margin readout changed under relation-arm training, but its T-N summary depended on a neutral source-alone anchor. The split arms were trained with source and companion in separate rows, so their elevated N values were plausibly a source-alone/query-format familiarity shift. research reran the scorer on CLEAN, REPEAT, VIEW, REPEAT_SPLIT, and VIEW_SPLIT for both seeds and used the format-matched UPDATED_USE contrast `T(new-source)-U(new-source)`, where both T and U contain source, update, and query/use frame. This keeps T, U, and N visible but makes T-U the primary state-readout quantity.\n\n")
    lines.append("The result supports a real relation-locality readout, but with an important refinement: both local REPEAT and local VIEW increase update-conditioned state use relative to CLEAN, while split arms are not merely zero; they are consistently below CLEAN on T-U because their U movement is larger than their T movement. Thus research should be cited only through this research/research repair, not as a T-N-only result.\n\n")

    lines.append("## Central arm-minus-CLEAN quantities on UPDATED_USE extended_nontrain (n=1125)\n\n")
    lines.append("| seed | contrast | raw T | raw U | T-U | SE | T-N | content T-U |\n")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|\n")
    for r in central:
        lines.append(f"| {r['seed']} | {r['comparison']} | {fmt(r['delta_T_raw_full'])} | {fmt(r['delta_U_raw_full'])} | {fmt(r['delta_T_minus_U_full'])} | {r['se_delta_T_minus_U_full']:.4f} | {fmt(r['delta_T_minus_N_full'])} | {fmt(r['delta_T_minus_U_content'])} |\n")

    lines.append("\n## Per-arm state-readout means\n\n")
    lines.append("| seed | arm | T | U | N | T-U | T-N |\n")
    lines.append("|---:|---|---:|---:|---:|---:|---:|\n")
    for r in per_arm:
        lines.append(f"| {r['seed']} | {r['arm']} | {fmt(r['T'])} | {fmt(r['U'])} | {fmt(r['N'])} | {fmt(r['T_minus_U'])} | {fmt(r['T_minus_N'])} |\n")

    lines.append("\n## Direct locality magnitudes\n\n")
    lines.append("| seed | contrast | T-U difference | T-N difference |\n")
    lines.append("|---:|---|---:|---:|\n")
    for r in local_minus_split:
        lines.append(f"| {r['seed']} | {r['contrast']} | {fmt(r['delta_T_minus_U_full'])} | {fmt(r['delta_T_minus_N_full'])} |\n")

    lines.append("\n## Scientific interpretation\n\n")
    lines.append("1. **Format-matched locality survives.** VIEW-CLEAN is positive in both seeds on T-U (+0.3656 and +0.4036), while VIEW_SPLIT-CLEAN is negative/near-zero (-0.0304 and -0.1311). REPEAT-CLEAN is also positive (+0.2083 and +0.3318), while REPEAT_SPLIT-CLEAN is negative (-0.1788 and -0.2455). The within-window local-minus-split differences are large for both relations: VIEW minus VIEW_SPLIT is about +0.396/+0.535 and REPEAT minus REPEAT_SPLIT about +0.387/+0.577.\n\n")
    lines.append("2. **The new state-margin face is a shared local second-sentence-use component plus a VIEW advantage.** Unlike compact changed-form probes, this probe does not recreate exact source recurrence; local REPEAT therefore does not show the exact-copy liability here. Instead, local REPEAT and VIEW share a positive sign against CLEAN, and VIEW adds an extra +0.157/+0.072 T-U over REPEAT. This supports the proposed convergence: sequence composition can train a local update/use relation, but the practiced slot and objective credit determine whether the intended entity-conditioned variable rather than a cheaper local relation is learned.\n\n")
    lines.append("3. **Split-arm phrase/source-format movement is real and should remain explicit.** Raw T is positive for split arms, but raw U is even more positive, so T-U turns negative. That is the clearest statement: split training changed state-phrase/source-format priors or broad update-format response, but did not install the same true-update use relation.\n\n")
    lines.append("4. **Manuscript implication.** The report can now include this as an additional state-readout only if it is described through format-matched T-U and two seeds. It should not use the old research wording that T-N alone confirmed locality.\n\n")
    lines.append("## Files\n\n")
    lines.append(f"- research note: `{IN_DIR.relative_to(ROOT) / 'relation_arm_tu_note.md'}`\n")
    lines.append(f"- Central table: `{(OUT_DIR / 'central_updated_extended_arm_minus_clean.csv').relative_to(ROOT)}`\n")
    lines.append(f"- Per-arm table: `{(OUT_DIR / 'per_arm_updated_extended_tun.csv').relative_to(ROOT)}`\n")
    lines.append(f"- Local-minus-split table: `{(OUT_DIR / 'local_minus_split_updated_extended.csv').relative_to(ROOT)}`\n")

    NOTE.write_text("".join(lines), encoding="utf-8")

    meta = {
        "status": "RELATION_ARM_TU_INTEGRATED",
        "created_utc": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        "inputs": {"summary": str(SUMMARY), "delta": str(DELTA)},
        "outputs": {
            "central": str(OUT_DIR / "central_updated_extended_arm_minus_clean.csv"),
            "per_arm": str(OUT_DIR / "per_arm_updated_extended_tun.csv"),
            "local_minus_split": str(OUT_DIR / "local_minus_split_updated_extended.csv"),
            "note": str(NOTE),
        },
        "central_rows": len(central),
        "per_arm_rows": len(per_arm),
        "local_minus_split_rows": len(local_minus_split),
    }
    (OUT_DIR / "integration_metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
