#!/usr/bin/env python3
"""research: nonconflicting minimal selected evaluation for RoBERTa compact-vs-repeat.

Runs selected official-compatible cheap-column scoring for already trained RoBERTa
checkpoints into a separate research output tree, so it does not collide with the
long-running research managed evaluator. No training, upload, SuperGLUE, AoA, or
leaderboard submission.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any


def find_user_root() -> Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
WRAPPER = WS / "scripts/eval_custom_checkpoint.py"
COMPACT_RUN = WS / "training/runs/roberta_compact_reinvest_100M_seed43022"
REPEAT_RUN = WS / "training/runs/roberta_repeat_compact_reinvest_100M_seed43022"
DEFAULT_OUT = WS / "data/roberta_minimal_selected_eval"
CHEAP_COLS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
STABLE_KEYS = ["cheap6_no_GlobalPIQA", "cheap5_no_GlobalPIQA_Reading", "EWoK_plus_Entity", "Supplement", "Entity", "COMPS"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def score_path(out_dir: Path, arm: str, ck: str) -> Path:
    return out_dir / arm / ck / f"roberta_{arm}_{ck}_summary.json"


def run_one(run_dir: Path, out_dir: Path, arm: str, ck: str, gpu: int, force: bool) -> subprocess.Popen:
    target = f"roberta_{arm}_{ck}"
    out_base = out_dir / arm / ck
    out_base.mkdir(parents=True, exist_ok=True)
    if score_path(out_dir, arm, ck).exists() and not force:
        # Use a tiny Python no-op process so caller can handle uniformly.
        return subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(0)"])
    cmd = [
        sys.executable, "-B", str(WRAPPER),
        "--run-dir", str(run_dir),
        "--endpoint", ck,
        "--target", target,
        "--out-base", str(out_base),
        "--gpu", str(gpu),
    ]
    if force:
        cmd.append("--force")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    logf = (out_base / "pair_driver_stdout.log").open("w", encoding="utf-8")
    return subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=logf, stderr=subprocess.STDOUT)


def read_score(path: Path) -> dict[str, Any]:
    d = json.loads(path.read_text(encoding="utf-8"))
    rec = d.get("record", {})
    return {"returncode": rec.get("returncode"), "cheap7": rec.get("cheap7"), "scores": rec.get("scores", {}), "per_target": rec.get("per_target"), "summary_path": str(path)}


def derived(scores: dict[str, Any]) -> dict[str, float | None]:
    vals = {k: scores.get(k) for k in CHEAP_COLS}
    b, s, e, en, c, g, r = [vals.get(k) for k in CHEAP_COLS]
    out: dict[str, float | None] = {}
    if all(v is not None for v in [b, s, e, en, c, r]):
        out["cheap6_no_GlobalPIQA"] = float(mean([b, s, e, en, c, r]))
    else:
        out["cheap6_no_GlobalPIQA"] = None
    if all(v is not None for v in [b, s, e, en, c]):
        out["cheap5_no_GlobalPIQA_Reading"] = float(mean([b, s, e, en, c]))
    else:
        out["cheap5_no_GlobalPIQA_Reading"] = None
    if e is not None and en is not None:
        out["EWoK_plus_Entity"] = float(e + en)
    else:
        out["EWoK_plus_Entity"] = None
    return out


def summarize(out_dir: Path, checkpoints: list[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    problems: list[str] = []
    for ck in checkpoints:
        cp = score_path(out_dir, "compact", ck)
        rp = score_path(out_dir, "repeat", ck)
        if not cp.exists() or not rp.exists():
            rows.append({"ck": ck, "missing": True, "compact_summary_exists": cp.exists(), "repeat_summary_exists": rp.exists()})
            problems.append(f"{ck}: missing compact or repeat summary")
            continue
        cs, rs = read_score(cp), read_score(rp)
        cd, rd = derived(cs["scores"]), derived(rs["scores"])
        delta: dict[str, float | None] = {}
        if cs.get("cheap7") is not None and rs.get("cheap7") is not None:
            delta["cheap7"] = float(cs["cheap7"] - rs["cheap7"])
        else:
            delta["cheap7"] = None
        for k in STABLE_KEYS:
            if k in cd:
                delta[k] = None if cd[k] is None or rd[k] is None else float(cd[k] - rd[k])
            elif k in cs["scores"]:
                delta[k] = None if cs["scores"].get(k) is None or rs["scores"].get(k) is None else float(cs["scores"][k] - rs["scores"][k])
        for col in CHEAP_COLS:
            if cs["scores"].get(col) is not None and rs["scores"].get(col) is not None:
                delta[col] = float(cs["scores"][col] - rs["scores"][col])
        rows.append({"ck": ck, "compact": {**cs, **cd}, "repeat": {**rs, **rd}, "compact_minus_repeat": delta})

    ready_rows = [r for r in rows if not r.get("missing")]
    mean_delta: dict[str, float | None] = {}
    for k in ["cheap7", *STABLE_KEYS, *CHEAP_COLS]:
        vals = [r["compact_minus_repeat"].get(k) for r in ready_rows if r["compact_minus_repeat"].get(k) is not None]
        mean_delta[k] = float(mean(vals)) if vals else None

    one_endpoint = len(ready_rows) == 1
    stable_positive_keys = [k for k in STABLE_KEYS if mean_delta.get(k) is not None and mean_delta[k] > 0]
    stable_negative_keys = [k for k in STABLE_KEYS if mean_delta.get(k) is not None and mean_delta[k] <= 0]
    cheap6 = mean_delta.get("cheap6_no_GlobalPIQA")
    cheap5 = mean_delta.get("cheap5_no_GlobalPIQA_Reading")
    ewok_ent = mean_delta.get("EWoK_plus_Entity")
    stable_positive = bool(cheap6 is not None and cheap6 > 0 and cheap5 is not None and cheap5 > 0 and ewok_ent is not None and ewok_ent >= 0 and len(stable_positive_keys) >= 4)
    stable_negative = bool(cheap6 is not None and cheap6 <= 0 and cheap5 is not None and cheap5 <= 0 and (ewok_ent is None or ewok_ent < 0 or len(stable_negative_keys) >= 4))
    if stable_positive:
        reading = "compact is positive on the selected stable-family readout in this minimal RoBERTa panel; use late-band confirmation before independent-seed replication unless the endpoint movement is already large and coherent"
    elif stable_negative:
        reading = "compact is neutral/negative on the selected stable-family readout in this minimal RoBERTa panel despite positive local pseudolikelihood; this supports a local-learning/downstream split unless additional late checkpoints overturn it"
    else:
        reading = "minimal endpoint panel is mixed or borderline; evaluate the smallest additional late checkpoints needed to decide stable-family direction"
    payload = {
        "status": "ROBERTA_MINIMAL_SELECTED_EVAL",
        "created_utc": now(),
        "meaning": "Nonconflicting selected official-compatible cheap-column readout for completed RoBERTa compact-vs-repeat checkpoints, run because the managed research full evaluator did not deliver an integrated trajectory.",
        "checkpoints": checkpoints,
        "per_checkpoint": rows,
        "mean_compact_minus_repeat": mean_delta,
        "stable_positive_keys": stable_positive_keys,
        "stable_nonpositive_keys": stable_negative_keys,
        "stable_selected_positive": stable_positive,
        "stable_selected_negative": stable_negative,
        "single_endpoint_panel": one_endpoint,
        "scientific_reading": reading,
        "problems": problems,
        "no_training_upload_superglue_aoa_or_leaderboard": True,
        "does_not_write_step211_managed_output_tree": True,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "minimal_selected_eval_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md = [
        "# research RoBERTa minimal selected evaluation",
        "",
        f"Created UTC: `{payload['created_utc']}`",
        "",
        f"Checkpoints: `{checkpoints}`",
        f"Stable selected positive: `{stable_positive}`; stable selected negative: `{stable_negative}`",
        f"Mean compact-minus-repeat: `{json.dumps(mean_delta, sort_keys=True)}`",
        "",
        payload["scientific_reading"],
        "",
        "No training, upload, SuperGLUE, AoA, or leaderboard submission was performed.",
    ]
    (out_dir / "minimal_selected_eval_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", default=["chck_100M"])
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    print(json.dumps({"event": "start", "utc": now(), "checkpoints": args.checkpoints, "out_dir": str(args.out_dir), "output_tree_touched": False}), flush=True)
    for ck in args.checkpoints:
        print(json.dumps({"event": "wave_start", "utc": now(), "ck": ck, "compact_gpu": 0, "repeat_gpu": 1}), flush=True)
        p0 = run_one(COMPACT_RUN, args.out_dir, "compact", ck, 0, args.force)
        p1 = run_one(REPEAT_RUN, args.out_dir, "repeat", ck, 1, args.force)
        rc0 = p0.wait(); rc1 = p1.wait()
        print(json.dumps({"event": "wave_done", "utc": now(), "ck": ck, "compact_rc": rc0, "repeat_rc": rc1}), flush=True)
        if rc0 != 0 or rc1 != 0:
            # Summaries usually carry exact stderr; still stop to inspect before expanding.
            summarize(args.out_dir, args.checkpoints)
            sys.exit(max(rc0, rc1))
    payload = summarize(args.out_dir, args.checkpoints)
    print(json.dumps({"status": payload["status"], "out": str(args.out_dir / "minimal_selected_eval_summary.json"), "mean_compact_minus_repeat": payload["mean_compact_minus_repeat"], "stable_selected_positive": payload["stable_selected_positive"], "stable_selected_negative": payload["stable_selected_negative"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
