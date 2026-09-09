#!/usr/bin/env python3
"""research: per-target anatomy of the non-official bridge selector.

Uses only research bridge_layer_rows.csv outputs. This does not inspect official
EWoK/GlobalPIQA rows and is meant to check whether the pooled bridge-fixed rule
hides a target-specific adjacent-depth recovery.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path.cwd()
WS = ROOT / "experiments/archive/representation_and_objectives"
INPUTS = {
    "matched_base_80M": WS / "data/decoder_alignment_base80/matched_base_80M/bridge_layer_rows.csv",
    "scale1p75_live_80M": WS / "data/decoder_alignment_scale80/scale1p75_live_80M/bridge_layer_rows.csv",
    "scale1p75_disabled_80M": WS / "data/decoder_alignment_disabled80/scale1p75_disabled_80M/bridge_layer_rows.csv",
}
OUT = WS / "data/decoder_alignment_synthesis/bridge_target_selector_anatomy.json"

def b(x: Any) -> bool:
    return str(x).strip().lower() in {"true", "1", "yes"}

def f(x: Any) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except Exception:
        return float("nan")

def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))

def bandify(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by = defaultdict(dict)
    for r in rows:
        by[(r["decoder_mode"], r["pair_id"])][int(r["layer_index"])] = r
    out = []
    for (mode, pair_id), byli in by.items():
        for lo in sorted(byli):
            hi = lo + 1
            if hi not in byli:
                continue
            r0, r1 = byli[lo], byli[hi]
            # Both/swap persistence is the strict signal used by the main selector.
            out.append({
                "decoder_mode": mode,
                "band_start": lo,
                "band_end": hi,
                "band": f"L{lo}-L{hi}",
                "pair_id": pair_id,
                "family": r0.get("family"),
                "persistent_both": b(r0.get("both_correct")) and b(r1.get("both_correct")),
                "persistent_swap": b(r0.get("swap_both_correct")) and b(r1.get("swap_both_correct")),
                "band_both": (f(r0.get("margin_AB")) + f(r1.get("margin_AB"))) * 0.5 > 0 and (f(r0.get("margin_BA")) + f(r1.get("margin_BA"))) * 0.5 > 0,
                "band_swap": (f(r0.get("margin_AB")) + f(r1.get("margin_AB"))) * 0.5 < 0 and (f(r0.get("margin_BA")) + f(r1.get("margin_BA"))) * 0.5 < 0,
                "M_mean_pair": (f(r0.get("four_cell_M")) + f(r1.get("four_cell_M"))) * 0.5,
            })
    return out

def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["decoder_mode"], r["band_start"], r["band_end"])].append(r)
    table = []
    for (mode, lo, hi), rs in sorted(groups.items()):
        n = len(rs)
        pb = sum(r["persistent_both"] for r in rs)
        ps = sum(r["persistent_swap"] for r in rs)
        bb = sum(r["band_both"] for r in rs)
        bs = sum(r["band_swap"] for r in rs)
        table.append({
            "decoder_mode": mode,
            "band_start": lo,
            "band_end": hi,
            "band": f"L{lo}-L{hi}",
            "n": n,
            "persistent_both": pb,
            "persistent_swap": ps,
            "persistent_both_frac": pb / n if n else None,
            "persistent_swap_frac": ps / n if n else None,
            "band_both": bb,
            "band_swap": bs,
            "band_both_frac": bb / n if n else None,
            "band_swap_frac": bs / n if n else None,
            "actual_minus_swap_persistent": pb - ps,
            "actual_minus_swap_band": bb - bs,
            "selector_score": (pb - ps) + 0.25 * (bb - bs),
            "M_mean": statistics.fmean([r["M_mean_pair"] for r in rs]) if rs else None,
        })
    return sorted(table, key=lambda r: (r["selector_score"], r["actual_minus_swap_persistent"], r["band_both_frac"] or -1, -r["band_start"]), reverse=True)

def main() -> None:
    out = {}
    for target, path in INPUTS.items():
        rows = read_csv(path)
        table = summarize(bandify(rows))
        out[target] = {"input": str(path.relative_to(ROOT)), "top10": table[:10], "bottom5": table[-5:]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "BRIDGE_TARGET_SELECTOR_ANATOMY_DONE", "summary": str(OUT.relative_to(ROOT))}, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
