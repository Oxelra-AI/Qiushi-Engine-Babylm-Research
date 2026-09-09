#!/usr/bin/env python3
"""research: wait for both RoBERTa 100M transfer arms, then score selected cheap
official-compatible metrics across the full 10M checkpoint grid, in two-GPU parallel
waves, and integrate compact-minus-repeat late-band deltas.

No SuperGLUE/AoA/upload/leaderboard submission. Selected cheap columns only.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def find_user_root() -> Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
EVAL_WRAPPER = WS / "scripts/eval_custom_checkpoint.py"
COMPACT_RUN = WS / "training/runs/roberta_compact_reinvest_100M_seed43022"
REPEAT_RUN = WS / "training/runs/roberta_repeat_compact_reinvest_100M_seed43022"
EVAL_DIR = WS / "data/roberta_full100m_selected_eval"
INTEG_DIR = WS / "data/roberta_full100m_integrated"

CHECKPOINTS = ["chck_10M", "chck_20M", "chck_30M", "chck_40M", "chck_50M", "chck_60M", "chck_70M", "chck_80M", "chck_90M", "chck_100M"]
LATE = ["chck_60M", "chck_70M", "chck_80M", "chck_90M", "chck_100M"]
POLL_SEC = 60
MAX_WAIT_SEC = 16000


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def arm_done(run: Path) -> bool:
    return (run / "scientific_metrics.json").exists()


def wait_both() -> None:
    t0 = time.time()
    while True:
        c = arm_done(COMPACT_RUN)
        r = arm_done(REPEAT_RUN)
        print(json.dumps({"event": "wait_training", "utc": now(), "compact_done": c, "repeat_done": r, "elapsed_sec": round(time.time() - t0, 1)}), flush=True)
        if c and r:
            return
        if time.time() - t0 > MAX_WAIT_SEC:
            print(json.dumps({"event": "wait_timeout", "compact_done": c, "repeat_done": r}), flush=True)
            sys.exit(3)
        time.sleep(POLL_SEC)


def run_eval(run: Path, arm: str, ck: str, gpu: int) -> subprocess.Popen:
    target = f"roberta_{arm}_{ck}"
    out_base = EVAL_DIR / arm / ck
    out_base.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-B", str(EVAL_WRAPPER),
        "--run-dir", str(run), "--endpoint", ck,
        "--target", target, "--out-base", str(out_base),
        "--gpu", str(gpu), "--force",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    logf = (out_base / "supervisor_stdout.log").open("w")
    return subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)


def read_scores(arm: str, ck: str) -> dict[str, Any] | None:
    summary = EVAL_DIR / arm / ck / f"roberta_{arm}_{ck}_summary.json"
    if not summary.exists():
        return None
    data = json.loads(summary.read_text(encoding="utf-8"))
    rec = data.get("record", {})
    scores = rec.get("scores", {})
    cheap7 = rec.get("cheap7")
    return {"scores": scores, "cheap7": cheap7}


def derived(scores: dict[str, float]) -> dict[str, float]:
    b = scores.get("BLiMP"); s = scores.get("Supplement"); e = scores.get("EWoK")
    en = scores.get("Entity"); c = scores.get("COMPS"); g = scores.get("GlobalPIQA"); r = scores.get("Reading")
    cheap6 = None
    if all(v is not None for v in [b, s, e, en, c, r]):
        cheap6 = (b + s + e + en + c + r) / 6.0
    cheap5 = None
    if all(v is not None for v in [b, s, e, en, c]):
        cheap5 = (b + s + e + en + c) / 5.0
    ewok_entity = None
    if e is not None and en is not None:
        ewok_entity = e + en
    return {"cheap6_no_GlobalPIQA": cheap6, "cheap5_no_GlobalPIQA_Reading": cheap5, "EWoK_plus_Entity": ewok_entity}


def main() -> None:
    print(json.dumps({"event": "start", "utc": now(), "compact_run": str(COMPACT_RUN), "repeat_run": str(REPEAT_RUN)}), flush=True)
    wait_both()
    # Score in waves: for each checkpoint, compact on GPU0 and repeat on GPU1 in parallel.
    for ck in CHECKPOINTS:
        p0 = run_eval(COMPACT_RUN, "compact_reinvest", ck, 0)
        p1 = run_eval(REPEAT_RUN, "repeat_compact", ck, 1)
        rc0 = p0.wait(); rc1 = p1.wait()
        print(json.dumps({"event": "wave_done", "utc": now(), "ck": ck, "compact_rc": rc0, "repeat_rc": rc1}), flush=True)

    # Integrate
    rows = []
    for ck in CHECKPOINTS:
        cs = read_scores("compact_reinvest", ck)
        rs = read_scores("repeat_compact", ck)
        if not cs or not rs:
            rows.append({"ck": ck, "missing": True})
            continue
        cd = derived(cs["scores"]); rd = derived(rs["scores"])
        delta = {"cheap7": (cs["cheap7"] - rs["cheap7"]) if (cs["cheap7"] is not None and rs["cheap7"] is not None) else None}
        for k in ["cheap6_no_GlobalPIQA", "cheap5_no_GlobalPIQA_Reading", "EWoK_plus_Entity"]:
            if cd[k] is not None and rd[k] is not None:
                delta[k] = cd[k] - rd[k]
        for col in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]:
            if col in cs["scores"] and col in rs["scores"]:
                delta[col] = cs["scores"][col] - rs["scores"][col]
        rows.append({"ck": ck, "compact": {**cs, **cd}, "repeat": {**rs, **rd}, "compact_minus_repeat": delta})

    late_rows = [r for r in rows if r.get("ck") in LATE and not r.get("missing")]
    def mean_late(key: str) -> float | None:
        vals = [r["compact_minus_repeat"].get(key) for r in late_rows if r["compact_minus_repeat"].get(key) is not None]
        return sum(vals) / len(vals) if vals else None
    late_summary = {k: mean_late(k) for k in ["cheap7", "cheap6_no_GlobalPIQA", "cheap5_no_GlobalPIQA_Reading", "EWoK_plus_Entity", "Supplement", "Entity", "COMPS"]}

    payload = {
        "status": "ROBERTA_TRANSFER_INTEGRATED",
        "meaning": "Compact-vs-repeat RoBERTa MLM architecture-transfer readout across full 100M trajectory; positive late deltas on stable families would show the compact data marginal transfers beyond DeBERTa.",
        "created_utc": now(),
        "checkpoints": CHECKPOINTS,
        "late_band": LATE,
        "per_checkpoint": rows,
        "late_band_mean_compact_minus_repeat": late_summary,
        "readout_note": "Judge on cheap6_no_GlobalPIQA, cheap5_no_GlobalPIQA_Reading, EWoK+Entity, Supplement, Entity, COMPS in the late band; do not promote on GlobalPIQA/Reading alone.",
    }
    INTEG_DIR.mkdir(parents=True, exist_ok=True)
    (INTEG_DIR / "roberta_transfer_integrated.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "late_band_mean_compact_minus_repeat": late_summary, "out": str(INTEG_DIR / "roberta_transfer_integrated.json")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
