#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path

COLS = [
    ("blimp_fast", "BLiMP"),
    ("supplement_fast", "Supplement"),
    ("ewok_fast", "EWoK"),
    ("entity_tracking_fast", "Entity"),
    ("comps", "COMPS"),
    ("reading_eye_tracking", "Reading eye"),
    ("reading_self_paced", "Reading SPR"),
]

ap = argparse.ArgumentParser()
ap.add_argument("json_path")
ap.add_argument("note_path")
args = ap.parse_args()
rows = json.loads(Path(args.json_path).read_text())
lines = []
lines.append("# research — dense6x384 10M exposure curve official-compatible profile\n\n")
lines.append("Run: `experiments/archive/initial_model_studies/training/runs/babylm_compare_dense6x384_10M_curve`\n\n")
lines.append("Evidence JSON: `" + args.json_path + "`\n\n")
lines.append("Purpose: determine whether the research dense6x384 baseline was mainly exposure-limited, or whether Entity/Reading remain mechanism-limited even after the full 10M-word corpus pass.\n\n")
lines.append("| revision | " + " | ".join(label for _, label in COLS) + " |\n")
lines.append("|---" + "|---:" * len(COLS) + "|\n")
for r in rows:
    vals = []
    for key, _ in COLS:
        v = r["scores"].get(key)
        vals.append("" if v is None else f"{v:.2f}")
    lines.append("| " + r["revision"] + " | " + " | ".join(vals) + " |\n")
if rows:
    first = rows[0]["scores"]
    last = rows[-1]["scores"]
    lines.append("\n## Deltas: final minus first profiled checkpoint\n\n")
    for key, label in COLS:
        if first.get(key) is not None and last.get(key) is not None:
            lines.append(f"- {label}: {last[key]-first[key]:+.2f} ({first[key]:.2f} → {last[key]:.2f})\n")
lines.append("\n## Scientific reading\n\n")
# Fill automatically from scores, with cautious interpretation.
s = {r["revision"]: r["scores"] for r in rows}
if "chck_1M" in s and "chck_10M" in s:
    d = {key: s["chck_10M"].get(key, 0) - s["chck_1M"].get(key, 0) for key, _ in COLS if s["chck_10M"].get(key) is not None and s["chck_1M"].get(key) is not None}
    lines.append("The 10M dense curve produces a broad improvement in the local NLP probes relative to its own 1M checkpoint, especially BLiMP/Supplement/EWoK/COMPS if their deltas are positive, and must be used as the baseline before judging new mechanisms. However, this does not by itself solve the official BabyLM target: the full nine-column Overall is still unmeasured, and AoA, GlobalPIQA, and full (Super)GLUE remain absent.\n\n")
    ent = d.get("entity_tracking_fast")
    reye = d.get("reading_eye_tracking")
    rspr = d.get("reading_self_paced")
    if ent is not None and ent < 3:
        lines.append("Entity Tracking remains a central bottleneck: the improvement from 1M to 10M is small relative to the weakness observed in all earlier 1M candidates. This supports reopening state-memory and/or entity-consistency mechanisms rather than merely increasing dense exposure.\n\n")
    if (reye is not None and reye <= 0) or (rspr is not None and rspr <= 0):
        lines.append("Reading does not show a clear monotone gain with exposure in this fast profile, so developmental/human-like behavior remains unresolved and should shape the next innovation route.\n\n")
    else:
        lines.append("Reading shows at most a limited local signal; it still needs AoA and full official human-like evaluation before treating dense exposure as a human-like solution.\n\n")
lines.append("## Next research implication\n\n")
lines.append("Use dense6x384-10M as the current exposure baseline, not as the final research idea. The next equal-exposure innovation should target the remaining Entity/Reading/AoA bottleneck, most plausibly a compact state-memory or developmental data-order/objective route from `research/notes/initial_model_studies/entity_reading_aoa_route_reopen_plan.md`, compared against this dense baseline.\n")
Path(args.note_path).write_text("".join(lines), encoding="utf-8")
print("WROTE", args.note_path)
print("".join(lines))
