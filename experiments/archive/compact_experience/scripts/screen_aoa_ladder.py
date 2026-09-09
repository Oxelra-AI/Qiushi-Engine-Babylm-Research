#!/usr/bin/env python3
"""research: AoA-ladder screening for a trained developmental-order arm.

Runs ONLY the official AoA curve-fitness computation over a run's chck_1M..chck_100M
ladder using the repaired local helper, then reports the leaderboard-unit AoA and the
per-checkpoint mean surprisal.  This is a fast screen of the developmental-order effect
on the AoA column without the full nine-column evaluation, and it is measurement-only:
it does not use AoA outputs to select or tune any training decision.

Usage:
  python screen_aoa_ladder.py --run <run_dir> --gpu 0
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import pathlib
import subprocess
import sys

WORKSPACE = _public_path('experiments/archive/compact_experience')
HELPER = _public_path('experiments/archive/compact_experience/scripts/aoa_local_ckpts_for_model.py')
OUT_ROOT = _public_path('experiments/archive/compact_experience/data/devcurr_aoa_screen')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir containing hf_model/chck_*M")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    run = pathlib.Path(args.run)
    model_root = run / "hf_model"
    if not model_root.exists():
        raise SystemExit(f"missing model root: {model_root}")
    required = [f"chck_{i}M" for i in range(1, 10)] + [f"chck_{i*10}M" for i in range(1, 11)]
    missing = [c for c in required if not (model_root / c).exists()]
    if missing:
        raise SystemExit(f"missing AoA ladder checkpoints: {missing[:5]} ... ({len(missing)} total)")

    label = args.label or run.name
    out_dir = OUT_ROOT / label
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "aoa_local_ckpts.json"
    out_note = out_dir / "aoa_local_ckpts.md"
    log = out_dir / "aoa.log"
    cmd = [
        sys.executable, str(_public_path('experiments/archive/compact_experience/scripts/aoa_local_ckpts_for_model.py')),
        "--model_root", str(model_root.resolve()),
        "--out_dir", str(out_dir.resolve()),
        "--out_json", str(out_json.resolve()),
        "--out_note", str(out_note.resolve()),
        "--log", str(log.resolve()),
        "--gpu", str(args.gpu),
    ]
    print(json.dumps({"event": "aoa_screen_start", "run": str(run), "label": label, "gpu": args.gpu}), flush=True)
    rc = subprocess.run(cmd).returncode
    payload = {"event": "aoa_screen_done", "run": str(run), "label": label, "returncode": rc}
    if out_json.exists():
        d = json.loads(out_json.read_text(encoding="utf-8"))
        raw = float(d.get("aoa", 0.0))
        payload.update({
            "aoa_raw_correlation": raw,
            "aoa_leaderboard_score": 100.0 * raw,
            "num_rows": d.get("num_rows"),
            "num_steps": d.get("num_steps"),
            "step_mean_surprisal": d.get("step_mean_surprisal"),
        })
    print(json.dumps(payload, indent=2), flush=True)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
