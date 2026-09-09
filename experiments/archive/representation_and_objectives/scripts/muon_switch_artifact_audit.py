#!/usr/bin/env python3
"""Audit Muon/AdamW switch artifacts.

This is a CPU/read-only analysis.  It records what checkpoints actually exist,
how far the custom runs trained, where the optimizer switch happened, and how
large the immediate loss shock was.  The purpose is to interpret later 40M
readouts without treating incomplete 80M run directory names as mature endpoints.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import math
import re
import time
from pathlib import Path
from statistics import fmean
from typing import Any

ROOT = _public_path('.')
A01_WS = _public_path('experiments/archive/representation_and_objectives')
A02_WS = _public_path('experiments/archive/frontier_consolidation')
OUT = _public_path('experiments/archive/representation_and_objectives/data/muon_switch_artifact_audit')
NOTE = _public_path('research/notes/representation_and_objectives/muon_switch_artifact_audit.md')

RUNS = {
    "adamw_reference": {
        "label": "continuous AdamW research/36 r2 reference",
        "run_dir": _public_path('experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2'),
        "switch_after_steps": None,
    },
    "continuous_muon": {
        "label": "continuous matched-decay Muon reference",
        "run_dir": _public_path('experiments/archive/frontier_consolidation/training/runs/muon_lr008_wd00125_seed43022_80M'),
        "switch_after_steps": None,
    },
    "muon20toadamw": {
        "label": "Muon 20M then AdamW",
        "run_dir": _public_path('experiments/archive/frontier_consolidation/training/runs/muon20toadamw_seed43022_80M'),
        "switch_after_steps": 506,
    },
    "muon40toadamw": {
        "label": "Muon 40M then AdamW",
        "run_dir": _public_path('experiments/archive/frontier_consolidation/training/runs/muon40toadamw_seed43022_80M'),
        "switch_after_steps": 1012,
    },
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def checkpoint_words(name: str) -> int | None:
    m = re.fullmatch(r"chck_(\d+)M", name)
    return int(m.group(1)) if m else None


def q(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    idx = p * (len(ys) - 1)
    lo = math.floor(idx); hi = math.ceil(idx)
    if lo == hi:
        return ys[lo]
    return ys[lo] * (hi - idx) + ys[hi] * (idx - lo)


def stats(xs: list[float]) -> dict[str, Any]:
    xs = [float(x) for x in xs if math.isfinite(float(x))]
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "mean": fmean(xs), "median": q(xs, 0.5), "min": min(xs), "max": max(xs), "p05": q(xs, 0.05), "p95": q(xs, 0.95)}


def window(rows: list[dict[str, Any]], start: int, end: int) -> list[dict[str, Any]]:
    return [r for r in rows if start <= int(r.get("step", -1)) <= end]


def audit_run(key: str, spec: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(spec["run_dir"])
    hf = run_dir / "hf_model"
    ckpts = []
    if hf.exists():
        for p in hf.iterdir():
            if p.is_dir():
                w = checkpoint_words(p.name)
                if w is not None:
                    ckpts.append((w, p.name, (p / "model.safetensors").exists(), (p / "config.json").exists()))
    ckpts.sort()
    logs = read_jsonl(run_dir / "training_log.jsonl")
    final = logs[-1] if logs else None
    rec: dict[str, Any] = {
        "key": key,
        "label": spec["label"],
        "run_dir": str(run_dir),
        "exists": run_dir.exists(),
        "checkpoint_names": [name for _, name, _, _ in ckpts],
        "max_checkpoint_M": max([w for w, _, _, _ in ckpts], default=None),
        "checkpoint_count": len(ckpts),
        "bad_checkpoints": [name for _, name, has_w, has_c in ckpts if not (has_w and has_c)],
        "training_log_exists": (run_dir / "training_log.jsonl").exists(),
        "training_log_rows": len(logs),
        "final_logged_step": int(final.get("step")) if final else None,
        "final_logged_word_exposure": int(final.get("cumulative_word_exposure")) if final and final.get("cumulative_word_exposure") is not None else None,
        "final_logged_loss": float(final.get("loss")) if final and final.get("loss") is not None else None,
        "has_scientific_metrics": (run_dir / "scientific_metrics.json").exists(),
        "switch_after_steps": spec.get("switch_after_steps"),
    }
    # milestone losses near checkpoints
    milestones = [20_000_000, 40_000_000, 50_000_000, 70_000_000, 80_000_000]
    rec["nearest_milestones"] = {}
    for mw in milestones:
        if logs:
            best = min(logs, key=lambda r: abs(int(r.get("cumulative_word_exposure", 0)) - mw))
            rec["nearest_milestones"][str(mw)] = {"step": best.get("step"), "word_exposure": best.get("cumulative_word_exposure"), "loss": best.get("loss"), "lr": best.get("lr")}
    sw = spec.get("switch_after_steps")
    if sw is not None and logs:
        pre = window(logs, max(1, sw - 20), sw)
        post1 = window(logs, sw + 1, sw + 5)
        post2 = window(logs, sw + 6, sw + 50)
        later = window(logs, sw + 200, sw + 300)
        swrow = [r for r in logs if int(r.get("step", -1)) == sw]
        first_after = [r for r in logs if int(r.get("step", -1)) == sw + 1]
        rec["switch_dynamics"] = {
            "pre20_loss": stats([r["loss"] for r in pre if "loss" in r]),
            "post1to5_loss": stats([r["loss"] for r in post1 if "loss" in r]),
            "post6to50_loss": stats([r["loss"] for r in post2 if "loss" in r]),
            "post200to300_loss": stats([r["loss"] for r in later if "loss" in r]),
            "switch_row": swrow[0] if swrow else None,
            "first_after_switch": first_after[0] if first_after else None,
        }
        if pre and post1:
            rec["switch_dynamics"]["mean_loss_jump_post1to5_minus_pre20"] = stats([r["loss"] for r in post1])["mean"] - stats([r["loss"] for r in pre])["mean"]
    return rec


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = {k: audit_run(k, v) for k, v in RUNS.items()}
    summary = {
        "status": "MUON_SWITCH_ARTIFACT_AUDIT",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Read-only audit of A02 Muon switch artifacts before route decisions or further GPU spending.",
        "rows": rows,
    }
    out_json = _public_path('experiments/archive/representation_and_objectives/data/muon_switch_artifact_audit/muon_switch_artifact_audit.json')
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = ["# research — A02 Muon switch artifact audit", "", "This is a read-only audit of what A02's optimizer-switch artifacts actually contain before using them for route decisions.", ""]
    lines.append("| run | max checkpoint | final logged exposure | final step | final loss | metrics file | switch step | immediate switch loss jump |")
    lines.append("|---|---:|---:|---:|---:|---|---:|---:|")
    for k, r in rows.items():
        swdyn = r.get("switch_dynamics") or {}
        jump = swdyn.get("mean_loss_jump_post1to5_minus_pre20")
        lines.append(f"| {k} | {r.get('max_checkpoint_M')}M | {r.get('final_logged_word_exposure')} | {r.get('final_logged_step')} | {r.get('final_logged_loss')} | {r.get('has_scientific_metrics')} | {r.get('switch_after_steps')} | {jump if jump is not None else ''} |")
    lines += ["", "Notes:"]
    for k, r in rows.items():
        if k.startswith("muon") and r.get("switch_dynamics"):
            sd = r["switch_dynamics"]
            lines.append(f"- `{k}`: pre-switch 20-step mean loss {sd['pre20_loss'].get('mean'):.4f}; first 5 post-switch mean {sd['post1to5_loss'].get('mean'):.4f}; post+6..50 mean {sd['post6to50_loss'].get('mean'):.4f}; post+200..300 mean {sd['post200to300_loss'].get('mean') if sd['post200to300_loss'].get('n') else 'NA'}. This reflects zero-initialized AdamW moments for hidden matrices after Muon, not just a smooth optimizer schedule change.")
    lines += ["", f"JSON: `{out_json}`"]
    _public_path('research/notes/representation_and_objectives').mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "summary": str(out_json), "note": str(NOTE)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
