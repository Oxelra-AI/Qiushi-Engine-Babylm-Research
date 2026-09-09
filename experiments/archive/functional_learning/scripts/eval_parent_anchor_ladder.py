#!/usr/bin/env python3
"""Evaluate selected research parent-anchored continuation checkpoints on the common cheap7 screen."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
EVAL_SCRIPT = _public_path('experiments/archive/functional_learning/scripts/eval_common_screen.py')
RUN = _public_path('experiments/archive/functional_learning/training/runs/coherent86_continue_parent_anchor_seed43023')
DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/parent_anchor_ladder_eval')
DEFAULT_MODELS = [
    ("pa_87M", _public_path('experiments/archive/functional_learning/training/runs/coherent86_continue_parent_anchor_seed43023/hf_model/chck_total_87005295w')),
    ("pa_90M", _public_path('experiments/archive/functional_learning/training/runs/coherent86_continue_parent_anchor_seed43023/hf_model/chck_total_90005295w')),
    ("pa_94M", _public_path('experiments/archive/functional_learning/training/runs/coherent86_continue_parent_anchor_seed43023/hf_model/chck_total_94005295w')),
    ("pa_98M", _public_path('experiments/archive/functional_learning/training/runs/coherent86_continue_parent_anchor_seed43023/hf_model/chck_total_98005295w')),
    ("pa_100M_final", _public_path('experiments/archive/functional_learning/training/runs/coherent86_continue_parent_anchor_seed43023/hf_model/final')),
]


def rel(p: Path | str) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def load_scores(out_root: Path, tag: str) -> dict | None:
    p = out_root / f"{tag}_eval.json"
    if not p.exists():
        return None
    return json.loads(p.read_text()).get("scores", {})


def parse_model_specs(specs: list[str] | None):
    if not specs:
        return DEFAULT_MODELS
    out = []
    for spec in specs:
        tag, path = spec.split("=", 1)
        out.append((tag, Path(path)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default=str(DEFAULT_OUT))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--models", nargs="*", default=None, help="Optional tag=path list replacing default parent-anchor ladder")
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    records = []
    for tag, model_path in parse_model_specs(args.models):
        if not model_path.exists():
            records.append({"tag": tag, "model_path": rel(model_path), "status": "missing"})
            print(json.dumps(records[-1]), flush=True)
            continue
        if args.skip_existing and load_scores(out_root, tag):
            scores = load_scores(out_root, tag)
            records.append({"tag": tag, "model_path": rel(model_path), "status": "cached", "scores": scores})
            print(json.dumps({"event": "cached", "tag": tag, "eq": scores.get("equal_valid_mean")}), flush=True)
            continue
        cmd = [sys.executable, str(EVAL_SCRIPT), "--model_path", str(model_path), "--tag", tag, "--gpu", str(args.gpu), "--out_root", str(out_root)]
        print(json.dumps({"event": "eval_start", "tag": tag, "model_path": rel(model_path), "cmd": cmd}), flush=True)
        p = subprocess.run(cmd, cwd=str(ROOT), text=True)
        status = "ok" if p.returncode == 0 else "failed"
        scores = load_scores(out_root, tag)
        records.append({"tag": tag, "model_path": rel(model_path), "status": status, "returncode": p.returncode, "scores": scores})
        print(json.dumps({"event": "eval_done", "tag": tag, "status": status, "eq": None if scores is None else scores.get("equal_valid_mean")}), flush=True)
        if p.returncode != 0:
            break
    summary = {"status": "PARENT_ANCHOR_LADDER_EVAL_DONE", "records": records}
    out_json = out_root / "selected_ladder_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# research parent-anchor selected ladder common-screen evaluation", "", "| tag | status | equal7 | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | Reading |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in records:
        s = r.get("scores") or {}
        lines.append("| {tag} | {status} | {eq} | {BLiMP} | {Supplement} | {EWoK} | {Entity} | {COMPS} | {GPIQA} | {Reading} |".format(
            tag=r.get("tag", ""), status=r.get("status", ""),
            eq="" if s.get("equal_valid_mean") is None else f"{s['equal_valid_mean']:.4f}",
            BLiMP="" if s.get("BLiMP") is None else f"{s['BLiMP']:.2f}",
            Supplement="" if s.get("Supplement") is None else f"{s['Supplement']:.2f}",
            EWoK="" if s.get("EWoK") is None else f"{s['EWoK']:.2f}",
            Entity="" if s.get("Entity") is None else f"{s['Entity']:.2f}",
            COMPS="" if s.get("COMPS") is None else f"{s['COMPS']:.2f}",
            GPIQA="" if s.get("GlobalPIQA_mean") is None else f"{s['GlobalPIQA_mean']:.3f}",
            Reading="" if s.get("Reading") is None else f"{s['Reading']:.3f}",
        ))
    out_md = out_root / "selected_ladder_summary.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "out_json": rel(out_json), "out_md": rel(out_md)}), flush=True)


if __name__ == "__main__":
    main()
