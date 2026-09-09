#!/usr/bin/env python3
"""research: reconcile fast seed-variance evidence with official full seed43122 vector.

Uses research fast analysis and research official-coordinate comparison/localization to
identify what the fast screen did and did not predict once the full official coordinate
became available. CPU-only; no new evaluation.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import math
from pathlib import Path
import time
from typing import Any


def find_user_root() -> Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
WORKSPACE = ROOT / "experiments/archive/representation_and_objectives"
FAST = WORKSPACE / "data/seed_variance_fast_and_dynamics_repaired/seed_variance_fast_and_dynamics.json"
OFFICIAL_COMPARE = WORKSPACE / "data/seed43122_official_comparison_after_delivery/seed43122_official_comparison_after_delivery.json"
LOCALIZATION = WORKSPACE / "data/seed43122_official_failure_localization/seed43122_official_failure_localization.json"
OUT_DIR = WORKSPACE / "data/fast_full_seed43122_reconciliation"
NOTE = WORKSPACE / "notes/fast_full_seed43122_reconciliation.md"


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_get(obj: Any, dotted: str, default: Any = None) -> Any:
    cur = obj
    for p in dotted.split("."):
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default
    return cur


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else None}


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx, my = sum(xs)/len(xs), sum(ys)/len(ys)
    vx = sum((x-mx)**2 for x in xs)
    vy = sum((y-my)**2 for y in ys)
    if vx == 0 or vy == 0:
        return None
    return sum((x-mx)*(y-my) for x, y in zip(xs, ys)) / math.sqrt(vx*vy)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    fast = load(FAST)
    official = load(OFFICIAL_COMPARE)
    loc = load(LOCALIZATION)

    fast_cols = dict(safe_get(fast, "fast_endpoint_delta_431_minus_430.columns", {}))
    # Map full official names onto fast names.
    full_cols = dict(safe_get(official, "comparison.seed43122_minus_seed43022_scores", {}))
    mapped = []
    for full_name, fast_name in [("BLiMP", "BLiMP"), ("Supplement", "Supplement"), ("EWoK", "EWoK"), ("Entity", "Entity"), ("COMPS", "COMPS"), ("GlobalPIQA", "GlobalPIQA_mean"), ("Reading", "Reading")]:
        if full_name in full_cols and fast_name in fast_cols:
            mapped.append({
                "column": full_name,
                "fast_delta_431_minus_430": fast_cols[fast_name],
                "official_full_delta_431_minus_430": full_cols[full_name],
                "full_minus_fast_delta": full_cols[full_name] - fast_cols[fast_name],
                "same_sign": (fast_cols[fast_name] == 0 and full_cols[full_name] == 0) or (fast_cols[fast_name] > 0 and full_cols[full_name] > 0) or (fast_cols[fast_name] < 0 and full_cols[full_name] < 0),
            })
    xs = [float(r["fast_delta_431_minus_430"]) for r in mapped]
    ys = [float(r["official_full_delta_431_minus_430"]) for r in mapped]

    full_overall = safe_get(official, "comparison.seed43122_official_overall")
    full_margin = safe_get(official, "comparison.seed43122_margin_over_visible_leader_41p8")
    full_delta = safe_get(official, "comparison.seed43122_minus_seed43022_overall")
    fast_equal7_delta = safe_get(fast, "fast_endpoint_delta_431_minus_430.equal7_delta")
    # Official no-AoA seven NLP column mean delta for the same seven fast columns.
    official_equal7_like_delta = sum(float(r["official_full_delta_431_minus_430"]) for r in mapped) / len(mapped)

    worst_official = safe_get(loc, "worst_by_delta", [])[:20]
    family_summary = safe_get(loc, "family_summary", [])
    family_by_mean = sorted(family_summary, key=lambda x: x.get("mean_delta", 0.0))

    payload = {
        "status": "FAST_FULL_SEED43122_RECONCILIATION",
        "created_utc": now_utc(),
        "scientific_purpose": "Use the newly delivered official full seed43122 coordinate to decide how much the prior fast screen predicted, and what kind of repair is scientifically implied before any new expensive branch.",
        "inputs": {"fast": file_record(FAST), "official_compare": file_record(OFFICIAL_COMPARE), "localization": file_record(LOCALIZATION)},
        "official_seed43122": {"overall": full_overall, "margin_over_41p8": full_margin, "minus_seed43022_overall": full_delta},
        "fast_vs_full_column_deltas": mapped,
        "fast_equal7_delta": fast_equal7_delta,
        "official_equal7_like_delta_over_same_columns": official_equal7_like_delta,
        "fast_full_delta_pearson_over_7_columns": pearson(xs, ys),
        "family_summary_sorted_by_mean_delta": family_by_mean,
        "largest_official_subtask_losses": worst_official,
        "interpretation": {
            "fast_screen_was_directionally_right": all(bool(r["same_sign"]) for r in mapped),
            "fast_overstated_some_losses": "Fast screen overstated Supplement and EWoK loss magnitude; full official still shows broad negative movement and adds a real SuperGLUE loss that fast could not measure.",
            "not_coordinate_or_aoa_failure": "Staging passed unmodified collator with null_keys=[], EWoK total 7618, AoA 8005 rows/checkpoint and AoA 0.0, so the below-leader result is not the known stale-coordinate artifact.",
            "scientific_consequence": "Compact-view reinvestment produced a real seed43022 endpoint above the visible leader but not a reproducible two-seed SOTA coordinate. The next execution should separate pretraining RNG from fine-tuning/checkpoint choice using minimal existing-checkpoint/finetune evidence, or have Lead/Reviewer decide whether the single strong seed is adequate for submission while research continues on stability.",
        },
    }
    out_json = OUT_DIR / "fast_full_seed43122_reconciliation.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# research — fast/full reconciliation for seed43122",
        "",
        f"Official seed43122 Overall is `{full_overall}` with margin `{full_margin}` over visible 41.8; it is `{full_delta}` below seed43022.",
        f"Fast equal7 delta was `{fast_equal7_delta}`; official full delta averaged over the same seven columns is `{official_equal7_like_delta}`.",
        f"Pearson correlation over seven matched columns: `{payload['fast_full_delta_pearson_over_7_columns']}`.",
        "",
        "Column reconciliation:",
    ]
    for r in mapped:
        lines.append(f"- {r['column']}: fast `{r['fast_delta_431_minus_430']}`, official `{r['official_full_delta_431_minus_430']}`, full-minus-fast `{r['full_minus_fast_delta']}`, same_sign `{r['same_sign']}`")
    lines += ["", "Most negative official families:"]
    for fs in family_by_mean[:6]:
        lines.append(f"- {fs['family']}: mean_delta `{fs['mean_delta']}`, worst {fs['worst_subtasks'][:3]}")
    lines += [
        "",
        "Interpretation: the below-leader seed43122 result is not an AoA/EWoK coordinate artifact. Fast screening was directionally informative, but the official full vector shows a broad seed-dependent representation/optimization difference plus a SuperGLUE loss that fast evaluation did not observe.",
        f"JSON: `{out_json}`",
    ]
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "out_json": str(out_json),
        "note": str(NOTE),
        "official_overall431": full_overall,
        "margin431": full_margin,
        "fast_equal7_delta": fast_equal7_delta,
        "official_equal7_like_delta": official_equal7_like_delta,
        "pearson": payload["fast_full_delta_pearson_over_7_columns"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
