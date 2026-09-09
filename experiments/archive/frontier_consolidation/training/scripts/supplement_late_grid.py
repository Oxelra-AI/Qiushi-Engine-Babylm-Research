#!/usr/bin/env python3
"""research: fast Supplement-only late checkpoint grid for existing ladders.

Purpose: decide whether earlier stopping can repair the compact_view_reinvest
late Supplement reversal without starting any new pretraining or full evaluation.
The script evaluates the same fast BLiMP Supplement slices for clean-Qwen and
compact_view_reinvest seeds at selected existing checkpoints.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import gc
import json
import os
import pathlib
import shutil
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from transformers import AutoModelForMaskedLM

ROOT = pathlib.Path.cwd()
if not (ROOT / "experiments").exists():
    p = _public_path('experiments/archive/frontier_consolidation/training/scripts/supplement_late_grid.py')
    for parent in [p] + list(p.parents):
        if (parent / "experiments").exists():
            ROOT = parent
            break
STUDY = ROOT / "experiments/archive" / 'frontier_consolidation'
STRICT = ROOT / "experiments/archive" / 'initial_model_studies' / "repos" / "babylm-eval" / "strict"
if str(STRICT) not in sys.path:
    sys.path.insert(0, str(STRICT))

from evaluation_pipeline.sentence_zero_shot.dataset import get_dataloader  # noqa: E402
from evaluation_pipeline.sentence_zero_shot.compute_results import compute_results  # noqa: E402
from evaluation_pipeline.sentence_zero_shot.run import process_results  # noqa: E402


@dataclass(frozen=True)
class Condition:
    name: str
    family: str
    seed: str
    run_dir: pathlib.Path
    description: str


CONDITIONS: Dict[str, Condition] = {
    "reinvest_43022": Condition(
        name="reinvest_43022",
        family="compact_view_reinvest",
        seed="43022",
        run_dir=ROOT / "experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022",
        description="compact_view_reinvest seed43022, current official endpoint seed",
    ),
    "reinvest_43122": Condition(
        name="reinvest_43122",
        family="compact_view_reinvest",
        seed="43122",
        run_dir=ROOT / "experiments/archive/representation_and_objectives/training/runs/repl_compact_view_reinvest_seed43122",
        description="compact_view_reinvest seed43122 replication seed",
    ),
    "clean_43022": Condition(
        name="clean_43022",
        family="clean_qwen",
        seed="43022",
        run_dir=ROOT / "experiments/archive/compact_experience/training/runs/qwen_clean_aligned_16k_seed43022",
        description="inherited clean-Qwen seed43022 control",
    ),
    "clean_43122": Condition(
        name="clean_43122",
        family="clean_qwen",
        seed="43122",
        run_dir=ROOT / "experiments/archive/compact_experience/training/runs/qwen_clean_aligned_16k_seed43122",
        description="inherited clean-Qwen seed43122 control",
    ),
}

SUPPLEMENT_STEMS = [
    "qa_congruence_easy",
    "qa_congruence_tricky",
    "hypernym",
    "subject_aux_inversion",
    "turn_taking",
]


class EvalArgs:
    pass


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=sorted(CONDITIONS))
    ap.add_argument("--exposures", nargs="+", type=int, default=list(range(40, 101, 5)))
    ap.add_argument("--gpu", default="1", help="CUDA_VISIBLE_DEVICES value, or cpu")
    ap.add_argument("--out-dir", default=str(STUDY / "data/supplement_late_grid"))
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def checkpoint_path(cond: Condition, exposure_m: int) -> pathlib.Path:
    return cond.run_dir / "hf_model" / f"chck_{exposure_m}M"


def prepare_data_slice(out_dir: pathlib.Path) -> pathlib.Path:
    src_dir = STRICT / "evaluation_data" / "fast_eval" / "supplement_fast"
    dst_dir = out_dir / "data_slice" / "supplement_fast_selected"
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for stem in SUPPLEMENT_STEMS:
        src = src_dir / f"{stem}.jsonl"
        if not src.exists():
            raise FileNotFoundError(src)
        dst = dst_dir / src.name
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copyfile(src, dst)
        copied.append(src.name)
    (dst_dir / "slice_manifest.json").write_text(json.dumps({
        "task": "blimp",
        "source_dir": str(src_dir),
        "files": copied,
        "purpose": "research fast Supplement-only late checkpoint grid.",
    }, indent=2) + "\n", encoding="utf-8")
    return dst_dir


def make_eval_args(model_path: pathlib.Path, data_dir: pathlib.Path, batch_size: int) -> EvalArgs:
    a = EvalArgs()
    a.data_path = data_dir
    a.task = "blimp"
    a.model_path_or_name = str(model_path)
    a.backend = "mlm"
    a.output_dir = pathlib.Path("unused_step031_supplement_late_grid")
    a.images_path = None
    a.image_split = None
    a.image_template = None
    a.revision_name = None
    a.min_temperature = 1.0
    a.max_temperature = None
    a.temperature_interval = 0.05
    a.batch_size = batch_size
    a.non_causal_batch_size = batch_size
    a.full_sentence_scores = False
    a.save_predictions = False
    return a


def eval_supplement(model_path: pathlib.Path, data_dir: pathlib.Path, batch_size: int) -> dict[str, Any]:
    a = make_eval_args(model_path, data_dir, batch_size)
    t0 = time.time()
    model = AutoModelForMaskedLM.from_pretrained(str(model_path), trust_remote_code=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    load_sec = time.time() - t0
    try:
        dataloader = get_dataloader(a)
        t_eval = time.time()
        with torch.no_grad():
            results, _preds = compute_results(a, model, dataloader, [1.0])
        accuracies, averages = process_results(a, results)
        temp = sorted(averages.keys(), key=float)[0]
        avg = float(averages[temp])
        subtasks = {str(k): float(v) for k, v in accuracies[temp].get("UID", {}).items()}
        return {
            "average_score": avg,
            "subtask_scores": subtasks,
            "n_subtasks": len(subtasks),
            "load_sec": round(load_sec, 3),
            "eval_sec": round(time.time() - t_eval, 3),
        }
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def nested(rows: list[dict[str, Any]]) -> dict[tuple[str, str, int], dict[str, Any]]:
    return {(r["family"], r["seed"], int(r["exposure_m"])): r for r in rows}


def build_comparisons(rows: list[dict[str, Any]], exposures: list[int]) -> dict[str, Any]:
    ix = nested(rows)
    by_exp: dict[str, Any] = {}
    for exp in exposures:
        rec: dict[str, Any] = {}
        for family in ["compact_view_reinvest", "clean_qwen"]:
            r0 = ix.get((family, "43022", exp))
            r1 = ix.get((family, "43122", exp))
            if r0 and r1:
                rec[f"{family}_seed_gap_43122_minus_43022"] = r1["average_score"] - r0["average_score"]
                rec[f"{family}_seed_mean"] = (r1["average_score"] + r0["average_score"]) / 2
                rec[f"{family}_seed_min"] = min(r1["average_score"], r0["average_score"])
        for seed in ["43022", "43122"]:
            rr = ix.get(("compact_view_reinvest", seed, exp))
            cr = ix.get(("clean_qwen", seed, exp))
            if rr and cr:
                rec[f"TE_{seed}_reinvest_minus_clean"] = rr["average_score"] - cr["average_score"]
        if "TE_43022_reinvest_minus_clean" in rec and "TE_43122_reinvest_minus_clean" in rec:
            rec["DiD_TE43122_minus_TE43022"] = rec["TE_43122_reinvest_minus_clean"] - rec["TE_43022_reinvest_minus_clean"]
        if rec:
            by_exp[str(exp)] = rec

    # Candidate summaries: best reinvest exposure for each seed and for the seed minimum.
    candidates = []
    for exp in exposures:
        r0 = ix.get(("compact_view_reinvest", "43022", exp))
        r1 = ix.get(("compact_view_reinvest", "43122", exp))
        if not (r0 and r1):
            continue
        candidates.append({
            "exposure_m": exp,
            "reinvest_43022": r0["average_score"],
            "reinvest_43122": r1["average_score"],
            "reinvest_mean": (r0["average_score"] + r1["average_score"]) / 2,
            "reinvest_min_seed": min(r0["average_score"], r1["average_score"]),
            "seed_gap_43122_minus_43022": r1["average_score"] - r0["average_score"],
            "TE_43022": by_exp.get(str(exp), {}).get("TE_43022_reinvest_minus_clean"),
            "TE_43122": by_exp.get(str(exp), {}).get("TE_43122_reinvest_minus_clean"),
            "DiD": by_exp.get(str(exp), {}).get("DiD_TE43122_minus_TE43022"),
        })
    best = {
        "max_reinvest_43022": max(candidates, key=lambda r: r["reinvest_43022"]) if candidates else None,
        "max_reinvest_43122": max(candidates, key=lambda r: r["reinvest_43122"]) if candidates else None,
        "max_reinvest_mean": max(candidates, key=lambda r: r["reinvest_mean"]) if candidates else None,
        "max_reinvest_min_seed": max(candidates, key=lambda r: r["reinvest_min_seed"]) if candidates else None,
        "min_abs_seed_gap": min(candidates, key=lambda r: abs(r["seed_gap_43122_minus_43022"])) if candidates else None,
    }
    return {"by_exposure": by_exp, "candidate_rows": candidates, "best": best}


def write_note(result: dict[str, Any], out_md: pathlib.Path) -> None:
    lines = ["# research — fast Supplement late checkpoint grid\n\n"]
    lines.append("Existing checkpoint ladders only; fast Supplement slices only; no pretraining, corpus modification, SuperGLUE, AoA, or full official evaluation.\n\n")
    lines.append("## Reinvest candidate rows\n")
    lines.append("| exposure M | reinvest 43022 | reinvest 43122 | mean | min seed | gap 43122-43022 | TE43022 | TE43122 | DiD |\n")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in result["comparisons"]["candidate_rows"]:
        def fmt(x: Any) -> str:
            return "" if x is None else f"{float(x):.3f}"
        lines.append(
            f"| {r['exposure_m']} | {r['reinvest_43022']:.3f} | {r['reinvest_43122']:.3f} | "
            f"{r['reinvest_mean']:.3f} | {r['reinvest_min_seed']:.3f} | {r['seed_gap_43122_minus_43022']:+.3f} | "
            f"{fmt(r.get('TE_43022'))} | {fmt(r.get('TE_43122'))} | {fmt(r.get('DiD'))} |\n"
        )
    lines.append("\n## Best fast-Supplement exposures\n")
    for k, v in result["comparisons"]["best"].items():
        lines.append(f"- {k}: {v}\n")
    lines.append("\n## Scientific read\n")
    read = result.get("scientific_read", {})
    for v in read.values():
        lines.append(f"- {v}\n")
    lines.append(f"\nMachine-readable output: `{result['out_json']}`\n")
    out_md.write_text("".join(lines), encoding="utf-8")


def scientific_read(comparisons: dict[str, Any]) -> dict[str, str]:
    candidates = comparisons["candidate_rows"]
    if not candidates:
        return {"empty": "No complete reinvest candidate rows were available."}
    final = next((r for r in candidates if r["exposure_m"] == 100), candidates[-1])
    best431 = comparisons["best"]["max_reinvest_43122"]
    bestmin = comparisons["best"]["max_reinvest_min_seed"]
    # Avoid over-deciding: this is fast Supplement only.
    return {
        "earlier_stopping_seed43122": (
            f"Seed43122 fast Supplement best is {best431['reinvest_43122']:.3f} at {best431['exposure_m']}M "
            f"versus {final['reinvest_43122']:.3f} at 100M; this says whether earlier stopping can actually raise the weak seed's Supplement before broad task evaluation."
        ),
        "across_seed_minimum": (
            f"Best across-seed minimum is {bestmin['reinvest_min_seed']:.3f} at {bestmin['exposure_m']}M; "
            "if this occurs far before 100M, broad balance must be checked because Entity/EWoK/Reading may move differently."
        ),
        "scope": "This grid only selects or rejects candidate stopping exposures for further narrow broad-surface checks; it cannot by itself establish Overall or official full-Supplement improvement.",
    }


def main() -> None:
    ns = parse_args()
    out_dir = pathlib.Path(ns.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if ns.gpu.lower() != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ns.gpu
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    data_dir = prepare_data_slice(out_dir)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started = time.time()
    existing = out_dir / "supplement_late_grid.json"
    if existing.exists() and not ns.force:
        old = json.loads(existing.read_text(encoding="utf-8"))
        rows = old.get("rows", [])
        failures = old.get("failures", [])
    done = {(r["condition"], int(r["exposure_m"])) for r in rows}

    manifest = {
        "status": "SUPPLEMENT_LATE_GRID_RUNNING",
        "purpose": "Fast Supplement-only grid over existing clean/reinvest ladders to test earlier stopping before any new training or full evaluation.",
        "conditions": {k: {"family": v.family, "seed": v.seed, "run_dir": str(v.run_dir), "description": v.description} for k, v in CONDITIONS.items() if k in ns.conditions},
        "exposures_m": ns.exposures,
        "data_dir": str(data_dir),
        "gpu": ns.gpu,
        "batch_size": ns.batch_size,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    for cname in ns.conditions:
        cond = CONDITIONS[cname]
        for exp in ns.exposures:
            if (cname, exp) in done:
                continue
            ckpt = checkpoint_path(cond, exp)
            if not (ckpt / "model.safetensors").exists():
                failures.append({"condition": cname, "exposure_m": exp, "checkpoint_path": str(ckpt), "error": "missing model.safetensors"})
                continue
            print(json.dumps({"event": "eval_start", "condition": cname, "exposure_m": exp, "checkpoint": str(ckpt)}, ensure_ascii=False), flush=True)
            try:
                rec = eval_supplement(ckpt, data_dir, ns.batch_size)
                row = {
                    "condition": cname,
                    "family": cond.family,
                    "seed": cond.seed,
                    "exposure_m": exp,
                    "checkpoint_path": str(ckpt),
                    **rec,
                }
                rows.append(row)
                done.add((cname, exp))
                print(json.dumps({"event": "eval_done", "condition": cname, "exposure_m": exp, "score": rec["average_score"], "subtasks": rec["subtask_scores"]}, ensure_ascii=False), flush=True)
            except Exception as e:
                err = {"condition": cname, "exposure_m": exp, "checkpoint_path": str(ckpt), "error": repr(e)}
                failures.append(err)
                print(json.dumps({"event": "eval_error", **err}, ensure_ascii=False), flush=True)
            partial = {
                "status": "SUPPLEMENT_LATE_GRID_PARTIAL",
                "rows": rows,
                "failures": failures,
                "elapsed_sec": round(time.time() - started, 3),
            }
            (out_dir / "partial_supplement_late_grid.json").write_text(json.dumps(partial, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    rows.sort(key=lambda r: (r["family"], r["seed"], int(r["exposure_m"])))
    comparisons = build_comparisons(rows, ns.exposures)
    result = {
        "status": "SUPPLEMENT_LATE_GRID_COMPLETE",
        "purpose": manifest["purpose"],
        "inputs": manifest,
        "rows": rows,
        "failures": failures,
        "comparisons": comparisons,
        "scientific_read": scientific_read(comparisons),
        "elapsed_sec": round(time.time() - started, 3),
    }
    out_json = out_dir / "supplement_late_grid.json"
    result["out_json"] = str(out_json)
    out_md = out_dir / "supplement_late_grid.md"
    result["out_md"] = str(out_md)
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_note(result, out_md)
    print(json.dumps({
        "status": result["status"],
        "rows": len(rows),
        "failures": len(failures),
        "best": comparisons["best"],
        "out_json": str(out_json),
        "out_md": str(out_md),
        "elapsed_sec": result["elapsed_sec"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
