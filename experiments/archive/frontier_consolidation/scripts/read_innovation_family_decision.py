#!/usr/bin/env python3
"""research: combine broad research scores with strict-target prediction readout.

This is a post-delivery reader. It should be run only after:
  1) research managed cheap official-compatible evaluation has written
     strict_innovation_vs_trajectory.json, and
  2) strict_innovation_target_probe.py has been rerun with
     --checkpoints tokenmean_70M,tokenmean_80M,reference_70M,reference_80M
     (or at least reference_80M and tokenmean_80M are present in its JSON).

The decision criterion is: research's score vector alone does not
settle whether exact-swap merits another long run. The innovation family is
kept scientifically live only if the same repaired row-unique target prediction
object improves under research, or if broad transfer is already strong enough to
continue research itself. If broad transfer weakens and the target object does not
improve, do not continue the innovation family just because exact-swap is cleaner.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import math
import pathlib
import time
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
BROAD_TRAJ = WS / "data/strict_innovation_trajectory/strict_innovation_vs_trajectory.json"
TARGET_PROBE = WS / "data/strict_innovation_target_probe/strict_innovation_target_probe.json"
OUT_DIR = WS / "data/innovation_family_decision_reading"
OUT_JSON = OUT_DIR / "innovation_family_decision_reading.json"
OUT_MD = OUT_DIR / "innovation_family_decision_reading.md"
CORE = ["BLiMP", "Supplement", "EWoK"]
ALL7 = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_json(path: pathlib.Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def f(x: Any) -> float | None:
    try:
        y = float(x)
    except Exception:
        return None
    return y if math.isfinite(y) else None


def fmt(x: Any) -> str:
    y = f(x)
    return "NA" if y is None else f"{y:.4f}"


def get_broad80(traj: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(traj, dict):
        return None
    rows = traj.get("rows") or []
    if not rows:
        return None
    # Prefer 80M complete row, else largest complete row.
    complete = [r for r in rows if r.get("complete")]
    if not complete:
        return None
    for r in complete:
        if int(r.get("exposure_m", -1)) == 80:
            return r
    return sorted(complete, key=lambda r: int(r.get("exposure_m", -1)))[-1]


def summarize_broad(row: dict[str, Any] | None, traj: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    d = {c: f(row.get(f"delta_strict_minus_step35_{c}")) for c in ALL7}
    core_vals = [d[c] for c in CORE]
    raw_rows = (traj or {}).get("raw_rows", []) if isinstance(traj, dict) else []
    raw80 = next((x for x in raw_rows if int(x.get("exposure_m", -1)) == int(row.get("exposure_m", -1))), {})
    return {
        "exposure_m": int(row.get("exposure_m", -1)),
        "delta_mean7": f(row.get("delta_strict_minus_step35_mean7")),
        "delta_core_mean": sum(v for v in core_vals if v is not None) / len(core_vals) if all(v is not None for v in core_vals) else None,
        "positive_all7": sum(1 for v in d.values() if v is not None and v > 0),
        "positive_core3": sum(1 for c in CORE if d[c] is not None and d[c] > 0),
        "deltas": d,
        "delta_globalpiqa_parallel": f(raw80.get("delta_strict_minus_step35_GlobalPIQA_parallel")),
        "delta_globalpiqa_nonparallel": f(raw80.get("delta_strict_minus_step35_GlobalPIQA_nonparallel")),
    }


def checkpoint_summary(target: dict[str, Any] | None, label: str) -> dict[str, Any] | None:
    if not isinstance(target, dict):
        return None
    for r in target.get("checkpoint_results", []):
        if r.get("label") == label and r.get("status") == "ok":
            s = r.get("summary", {})
            return {
                "label": label,
                "true_loss": f((s.get("true_loss") or {}).get("mean")),
                "source_help": f((s.get("source_help") or {}).get("mean")),
                "same_decoy_advantage": f((s.get("same_decoy_advantage") or {}).get("mean")),
                "cross_decoy_advantage": f((s.get("cross_decoy_advantage") or {}).get("mean")),
                "masked_minus_same_decoy": f((s.get("masked_minus_same_decoy") or {}).get("mean")),
                "n": int((s.get("true_loss") or {}).get("n", 0)),
            }
    return None


def target_delta(a: dict[str, Any] | None, b: dict[str, Any] | None) -> dict[str, Any] | None:
    if a is None or b is None:
        return None
    out = {}
    for metric in ["true_loss", "source_help", "same_decoy_advantage", "cross_decoy_advantage", "masked_minus_same_decoy"]:
        va = f(a.get(metric)); vb = f(b.get(metric))
        out[f"delta_{metric}"] = None if va is None or vb is None else va - vb
    # Directional convenience: positive means research improves target prediction.
    dl = out.get("delta_true_loss")
    ds = out.get("delta_source_help")
    dd = out.get("delta_same_decoy_advantage")
    out["improved_true_loss"] = dl is not None and dl < -0.05
    out["improved_source_help"] = ds is not None and ds > 0.05
    out["improved_same_decoy_advantage"] = dd is not None and dd > 0.05
    out["target_improvement_count"] = int(bool(out["improved_true_loss"])) + int(bool(out["improved_source_help"])) + int(bool(out["improved_same_decoy_advantage"]))
    return out


def classify(broad: dict[str, Any] | None, d80: dict[str, Any] | None, d70: dict[str, Any] | None) -> dict[str, Any]:
    if broad is None:
        return {
            "state": "waiting_for_broad_scores",
            "reading": "The official-compatible research 70M/80M trajectory is not available, so no continuation judgment is made.",
            "next_action": "After managed evaluation delivers the trajectory, rerun target probe with research checkpoints and rerun this reader.",
        }
    if d80 is None:
        return {
            "state": "waiting_for_target_probe",
            "reading": "The broad research trajectory exists, but the strict target readout has not yet evaluated reference_80M against tokenmean_80M.",
            "next_action": "Run strict_innovation_target_probe.py with tokenmean_70M,tokenmean_80M,reference_70M,reference_80M before deciding whether exact-swap merits H100 time.",
            "broad80": broad,
        }
    dm = f(broad.get("delta_mean7")); dc = f(broad.get("delta_core_mean"))
    pos_all = int(broad.get("positive_all7", 0)); pos_core = int(broad.get("positive_core3", 0))
    gp_non = f(broad.get("delta_globalpiqa_nonparallel")); gpiqa = f((broad.get("deltas") or {}).get("GlobalPIQA"))
    target_count = int(d80.get("target_improvement_count", 0))
    true_loss_delta = f(d80.get("delta_true_loss"))
    source_help_delta = f(d80.get("delta_source_help"))

    broad_strong = dm is not None and dc is not None and dm >= 0.25 and dc > 0 and pos_core >= 2 and pos_all >= 5
    broad_weak_or_negative = dm is not None and (dm <= 0.0 or pos_core == 0)
    redistribution = (gp_non is not None and gp_non > 4.0 and (dm is None or dm <= 0.15)) or (gpiqa is not None and gpiqa > 2.0 and pos_core <= 1)
    target_improves = target_count >= 2 or (true_loss_delta is not None and true_loss_delta < -0.10) or (source_help_delta is not None and source_help_delta > 0.10)

    if broad_strong:
        return {
            "state": "continue_step075_probability_route_first",
            "reading": "research broad cheap surface is strong enough that the launched probability route, not a new exact-swap variant, should be continued toward 100M/full evaluation after ordinary metric inspection.",
            "next_action": "Inspect per-target official columns and training metrics, then continue research rather than spending a new run on exact-swap.",
            "broad80": broad,
            "target_delta80": d80,
            "target_delta70": d70,
        }
    if target_improves and (broad_weak_or_negative or redistribution or (dm is not None and dm < 0.25)):
        return {
            "state": "close_step075_keep_exact_swap_as_distinct",
            "reading": "research does not provide broad transfer, but it improves the repaired strict target-prediction object. The failure may come from excessive pressure or copyable suppression, so the smaller copyable-preserving exact-swap remains a distinct hypothesis if H100 budget allows.",
            "next_action": "Do not continue research to 100M. If route judgment favors one more innovation test, port and smoke exact-swap as a lower-perturbation single-variable run; otherwise return to route search.",
            "broad80": broad,
            "target_delta80": d80,
            "target_delta70": d70,
        }
    if (broad_weak_or_negative or redistribution) and not target_improves:
        return {
            "state": "close_innovation_family_for_now",
            "reading": "research weakens or redistributes broad transfer and does not improve the exact repaired target-prediction object. The innovation-masking family should not be carried forward merely because exact-swap is mechanically cleaner.",
            "next_action": "Close research without 100M, second seed, or retune, and do not launch exact-swap unless a new independent mechanism appears. Return to route selection around another unsaturated quantity.",
            "broad80": broad,
            "target_delta80": d80,
            "target_delta70": d70,
        }
    return {
        "state": "ambiguous_requires_anatomy",
        "reading": "Broad transfer and target-prediction deltas do not give a clean continuation or closure signal. Do not spend another long run before reading per-subtask anatomy and verifying the target probe on more rows or a second fixed sample.",
        "next_action": "Inspect broad per-target outputs and, if necessary, rerun the CPU target probe with a larger sample before choosing research continuation, exact-swap, or route closure.",
        "broad80": broad,
        "target_delta80": d80,
        "target_delta70": d70,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    traj = load_json(BROAD_TRAJ)
    target = load_json(TARGET_PROBE)
    missing = [str(p) for p, obj in [(BROAD_TRAJ, traj), (TARGET_PROBE, target)] if obj is None]
    broad = summarize_broad(get_broad80(traj if isinstance(traj, dict) else None), traj if isinstance(traj, dict) else None)
    t80 = checkpoint_summary(target if isinstance(target, dict) else None, "reference_80M")
    b80 = checkpoint_summary(target if isinstance(target, dict) else None, "tokenmean_80M")
    t70 = checkpoint_summary(target if isinstance(target, dict) else None, "reference_70M")
    b70 = checkpoint_summary(target if isinstance(target, dict) else None, "tokenmean_70M")
    d80 = target_delta(t80, b80)
    d70 = target_delta(t70, b70)
    interpretation = classify(broad, d80, d70)
    result = {
        "status": "INNOVATION_FAMILY_DECISION_READING",
        "created_utc": now(),
        "purpose": "Combine official-compatible research transfer with the repaired strict-target prediction readout so exact-swap is judged by mechanism evidence, not score vector alone.",
        "inputs": {"broad_trajectory": str(BROAD_TRAJ), "target_probe": str(TARGET_PROBE)},
        "missing_inputs": missing,
        "target_checkpoint_summaries": {"tokenmean_70M": b70, "reference_70M": t70, "tokenmean_80M": b80, "reference_80M": t80},
        "interpretation": interpretation,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research innovation-family decision reading", "", result["purpose"], "", f"State: `{interpretation['state']}`", "", interpretation["reading"], "", f"Next action: {interpretation['next_action']}", "", "## Missing inputs"]
    lines.extend([f"- `{m}`" for m in missing] if missing else ["- none"])
    if broad:
        d = broad.get("deltas", {})
        lines += ["", "## Broad 80M research - research", "", f"- Δmean7: `{fmt(broad.get('delta_mean7'))}`", f"- Δcore BLiMP/Supp/EWoK mean: `{fmt(broad.get('delta_core_mean'))}`", f"- positive all7/core3: `{broad.get('positive_all7')}` / `{broad.get('positive_core3')}`", f"- BLiMP `{fmt(d.get('BLiMP'))}`, Supplement `{fmt(d.get('Supplement'))}`, EWoK `{fmt(d.get('EWoK'))}`, GlobalPIQA `{fmt(d.get('GlobalPIQA'))}`, COMPS `{fmt(d.get('COMPS'))}`", f"- GlobalPIQA parallel/nonparallel deltas: `{fmt(broad.get('delta_globalpiqa_parallel'))}` / `{fmt(broad.get('delta_globalpiqa_nonparallel'))}`"]
    if d80:
        lines += ["", "## Strict-target reference_80M - tokenmean_80M", ""]
        for k, v in d80.items():
            lines.append(f"- `{k}`: `{fmt(v) if not isinstance(v, bool) else v}`")
    lines += ["", f"Full JSON: `{OUT_JSON}`"]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "state": interpretation["state"], "missing_inputs": missing, "out_json": str(OUT_JSON), "out_md": str(OUT_MD)}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
