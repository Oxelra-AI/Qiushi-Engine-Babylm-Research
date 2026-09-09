#!/usr/bin/env python3
"""research corrected 30M AoA calibration extraction and arm-minus-control fit.

This script turns the completed research adapter-faithful calibration arms into a
single measured AoA table.  It intentionally does not reconstruct exact credited
exposure; the first decision readout is simpler and directly preregistered by the
research question: do schedule or corpus-internal enrichment arms move fitted model
AoA in the useful direction relative to the same-seed v4-order control?

For each arm it extracts official-style AoA surprisals at selected checkpoints
(default 1M..10M,20M,30M) using the repaired research batched MLM extractor, fits
raw model AoA with the research official-semantics refit code, and regresses
arm-minus-control fitted model AoA against the token-level CHILDES-minus-whole
enrichment z-score averaged over each CDI word's subword pieces.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import importlib.util
import json
import math
import pathlib
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from transformers import AutoTokenizer

ROOT = _public_path('.')
SCRIPTS_A01 = _public_path('experiments/archive/functional_learning/scripts')
SCRIPTS_A02 = _public_path('experiments/archive/relation_learning/scripts')
DEFAULT_OUT = _public_path('experiments/archive/relation_learning/data/aoa_calibration_extract_fit')
TOKENIZER_PATH = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model')
ENRICHMENT_WEIGHTS = _public_path('experiments/archive/relation_learning/data/aoa_calibration_prep/token_enrichment_weights.json')

ARM_HF_MODELS = {
    "control": _public_path('experiments/archive/relation_learning/data/aoa_control/hf_model'),
    "schedule": _public_path('experiments/archive/relation_learning/data/aoa_schedule_arm/hf_model'),
    "enrichment": _public_path('experiments/archive/relation_learning/data/aoa_enrichment_arm/hf_model'),
}
ARM_METRICS = {
    name: path.parent / "scientific_metrics.json" for name, path in ARM_HF_MODELS.items()
}
DEFAULT_CHECKPOINTS = [*(f"chck_{i}M" for i in range(1, 11)), "chck_20M", "chck_30M"]


def rel(path: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def import_module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


EXT = import_module(_public_path('experiments/archive/functional_learning/scripts/batched_aoa_extractor.py'), "batched_aoa_extractor_for_step121")
FIT = import_module(_public_path('experiments/archive/relation_learning/scripts/refit_raw_aoa_endpoints_official.py'), "raw_aoa_fit_for_step121")


def checkpoint_word_count(name: str) -> int:
    import re
    m = re.search(r"chck_(\d+(?:\.\d+)?)M", name)
    if not m:
        raise ValueError(f"unsupported checkpoint name {name}")
    return int(round(float(m.group(1)) * 1_000_000))


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def complete_manifest(path: pathlib.Path, checkpoints: list[str], max_words: int) -> bool:
    man = path / "manifest.json"
    surp = path / "surprisal.json"
    if not man.is_file() or not surp.is_file():
        return False
    try:
        obj = read_json(man)
    except Exception:
        return False
    summary = obj.get("summary") or {}
    if obj.get("status") != "BATCHED_AOA_EXTRACTION_COMPLETE":
        return False
    if obj.get("steps") != checkpoints:
        return False
    if max_words and int((obj.get("eval_info") or {}).get("words_evaluated", -1)) != int(max_words):
        return False
    return summary.get("missing_steps") == [] and summary.get("steps_with_wrong_counts") == {} and summary.get("n_nan_or_nonfinite") == 0


def verify_arm_configs(arms: list[str], checkpoints: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm in arms:
        hf = ARM_HF_MODELS[arm]
        metrics_path = ARM_METRICS[arm]
        metrics = read_json(metrics_path) if metrics_path.is_file() else {}
        cfg_path = hf / "chck_30M/config.json"
        cfg = read_json(cfg_path) if cfg_path.is_file() else {}
        existing = [c for c in checkpoints if (hf / c / "config.json").is_file()]
        rows.append({
            "arm": arm,
            "hf_model": rel(hf),
            "metrics_path": rel(metrics_path),
            "checkpoint_configs_present": len(existing),
            "missing_checkpoint_configs": [c for c in checkpoints if c not in existing],
            "parameter_count": metrics.get("parameter_count"),
            "word_exposure": metrics.get("word_exposure"),
            "actual_training_steps": metrics.get("actual_training_steps"),
            "loss_first": metrics.get("loss_first"),
            "loss_last": metrics.get("loss_last"),
            "architecture": (cfg.get("architectures") or [None])[0],
            "adapter_enabled": cfg.get("adapter_enabled"),
            "adapter_bottleneck": cfg.get("adapter_bottleneck"),
            "adapter_scale": cfg.get("adapter_scale"),
            "position_biased_input": cfg.get("position_biased_input"),
            "max_position_embeddings": cfg.get("max_position_embeddings"),
            "max_relative_positions": cfg.get("max_relative_positions"),
            "pad_token_id": cfg.get("pad_token_id"),
            "bos_token_id": cfg.get("bos_token_id"),
            "eos_token_id": cfg.get("eos_token_id"),
        })
    return rows


def run_extraction_for_arm(arm: str, out_dir: pathlib.Path, checkpoints: list[str], args: argparse.Namespace) -> dict[str, Any]:
    arm_out = out_dir / "raw" / arm
    arm_out.mkdir(parents=True, exist_ok=True)
    if complete_manifest(arm_out, checkpoints, args.max_words) and not args.force_extract:
        print(json.dumps({"event": "reuse_complete_raw", "arm": arm, "out_dir": rel(arm_out)}), flush=True)
        return read_json(arm_out / "manifest.json")
    # Monkey-patch the repaired extractor's shared-ladder surface to the selected arm.
    EXT.LADDER = ARM_HF_MODELS[arm]
    EXT.EARLY_STOP_NAMES = list(checkpoints)
    EXT.EARLY_STOP_WORDS = [checkpoint_word_count(c) for c in checkpoints]
    ns = argparse.Namespace(
        mode="shared",
        target="coherent86",
        out_dir=arm_out,
        word_start=args.word_start,
        word_end=args.word_end,
        max_words=args.max_words,
        ancestral_limit=len(checkpoints),
        batch_size=args.batch_size,
        gpu=args.gpu,
        device=args.device,
        use_bos_only=False,
        progress_every=args.progress_every,
    )
    print(json.dumps({"event": "extract_arm_start", "arm": arm, "hf_model": rel(ARM_HF_MODELS[arm]), "checkpoints": checkpoints, "device": args.device or f"gpu{args.gpu}", "utc": now()}), flush=True)
    manifest = EXT.run_extract(ns)
    print(json.dumps({"event": "extract_arm_done", "arm": arm, "status": manifest.get("status"), "out_dir": rel(arm_out), "utc": now()}), flush=True)
    return manifest


def load_word_z(tokenizer) -> dict[str, float]:
    z_obj = read_json(ENRICHMENT_WEIGHTS)
    z = np.array(z_obj["z_enrichment"], dtype=float)
    target_words, _contexts = EXT.load_eval(EXT.CDI_WORDS_PATH, min_context=0, debug=False)
    prefix = tokenizer("The", add_special_tokens=False)["input_ids"]
    out: dict[str, float] = {}
    rows = []
    for word in target_words:
        ids_all = tokenizer("The " + word, add_special_tokens=False)["input_ids"]
        ids = ids_all[len(prefix):]
        if not ids:
            ids = tokenizer(word, add_special_tokens=False)["input_ids"]
        vals = [float(z[int(i)]) for i in ids if 0 <= int(i) < len(z)]
        out[word] = float(np.mean(vals)) if vals else 0.0
        rows.append({"word": word, "token_ids": " ".join(map(str, ids)), "z_mean_subword": out[word], "n_subword_pieces": len(ids)})
    return out, rows, {k: v for k, v in z_obj.items() if k != "z_enrichment"}


def fit_arm(arm: str, raw_path: pathlib.Path, out_dir: pathlib.Path, child_aoas: dict[str, float], tokenizer) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary, rows = FIT.fit_endpoint(arm, raw_path, child_aoas, tokenizer)
    pd.DataFrame(rows).to_csv(out_dir / f"{arm}_word_fits.csv", index=False)
    return summary, rows


def safe_pearson(x: list[float], y: list[float]) -> dict[str, Any]:
    pairs = [(float(a), float(b)) for a, b in zip(x, y, strict=False) if math.isfinite(float(a)) and math.isfinite(float(b))]
    if len(pairs) < 3:
        return {"pearson_r": None, "pearson_p": None, "n": len(pairs)}
    xs, ys = zip(*pairs)
    r, p = pearsonr(xs, ys)
    return {"pearson_r": float(r), "pearson_p": float(p), "n": len(pairs)}


def slope(x: list[float], y: list[float]) -> dict[str, Any]:
    pairs = [(float(a), float(b)) for a, b in zip(x, y, strict=False) if math.isfinite(float(a)) and math.isfinite(float(b))]
    if len(pairs) < 3:
        return {"slope": None, "intercept": None, "n": len(pairs)}
    xs = np.array([p[0] for p in pairs], dtype=float)
    ys = np.array([p[1] for p in pairs], dtype=float)
    X = np.vstack([np.ones_like(xs), xs]).T
    beta = np.linalg.lstsq(X, ys, rcond=None)[0]
    return {"intercept": float(beta[0]), "slope": float(beta[1]), "n": len(pairs)}


def delta_analysis(control_rows: list[dict[str, Any]], arm_rows: list[dict[str, Any]], word_z: dict[str, float], arm: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    c = {str(r["word"]): r for r in control_rows}
    a = {str(r["word"]): r for r in arm_rows}
    common = sorted(set(c) & set(a))
    rows: list[dict[str, Any]] = []
    for w in common:
        delta = float(a[w]["model_aoa_log10_step"]) - float(c[w]["model_aoa_log10_step"])
        rows.append({
            "arm": arm,
            "word": w,
            "z_mean_subword": float(word_z.get(w, 0.0)),
            "child_aoa": float(c[w]["child_aoa"]),
            "control_model_aoa_log10_step": float(c[w]["model_aoa_log10_step"]),
            "arm_model_aoa_log10_step": float(a[w]["model_aoa_log10_step"]),
            "delta_model_aoa_log10_step": delta,
            "beneficial_for_childes_enriched_words_if_negative": delta,
            "control_n_points": int(c[w].get("n_points", 0)),
            "arm_n_points": int(a[w].get("n_points", 0)),
        })
    zvals = [r["z_mean_subword"] for r in rows]
    deltas = [r["delta_model_aoa_log10_step"] for r in rows]
    child = [r["child_aoa"] for r in rows]
    arm_model = [r["arm_model_aoa_log10_step"] for r in rows]
    control_model = [r["control_model_aoa_log10_step"] for r in rows]
    summary = {
        "arm": arm,
        "n_common_fitted_words": len(rows),
        "mean_delta_model_aoa_log10_step": float(np.mean(deltas)) if deltas else None,
        "median_delta_model_aoa_log10_step": float(np.median(deltas)) if deltas else None,
        "delta_vs_z": safe_pearson(deltas, zvals),
        "delta_regressed_on_z": slope(zvals, deltas),
        "delta_vs_child_aoa": safe_pearson(deltas, child),
        "control_model_vs_child_aoa_common": safe_pearson(control_model, child),
        "arm_model_vs_child_aoa_common": safe_pearson(arm_model, child),
        "interpretation_of_z_slope": "negative slope means CHILDES-enriched words reach fitted threshold earlier than the same-seed control; positive slope means the intervention delays them relative to control.",
    }
    return summary, rows


def write_md(out_dir: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = ["# research AoA calibration extraction and fitted-direction readout", ""]
    lines.append("This is a 30M same-seed calibration screen using the corrected research adapter-faithful arms. It measures arm-minus-control fitted model AoA against corpus-internal CHILDES-minus-whole enrichment z; exact credited-exposure replay is intentionally not required for this first direction readout.")
    lines.append("")
    lines.append("## Arm integrity")
    lines.append("")
    lines.append("| arm | params | words | steps | loss first | loss last | architecture | missing ckpts |")
    lines.append("|---|---:|---:|---:|---:|---:|---|---|")
    for r in payload["arm_config_verification"]:
        lines.append(f"| {r['arm']} | {r.get('parameter_count')} | {r.get('word_exposure')} | {r.get('actual_training_steps')} | {r.get('loss_first')} | {r.get('loss_last')} | {r.get('architecture')} | {r.get('missing_checkpoint_configs')} |")
    lines.append("")
    lines.append("## Raw AoA summaries")
    lines.append("")
    lines.append("| arm | raw r | p | fitted words | clipped score |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in payload["fit_summaries"]:
        lines.append(f"| {r['endpoint']} | {r.get('raw_pearson_r')} | {r.get('raw_pearson_p')} | {r.get('fitted_word_count')} | {r.get('official_clipped_score')} |")
    lines.append("")
    lines.append("## Arm-minus-control fitted AoA versus enrichment z")
    lines.append("")
    lines.append("| arm | n | mean delta | r(delta,z) | p | slope delta~z | direction |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for r in payload["delta_summaries"]:
        corr = r["delta_vs_z"]
        sl = r["delta_regressed_on_z"]
        direction = "useful" if (sl.get("slope") is not None and float(sl["slope"]) < 0) else "opposite_or_flat"
        lines.append(f"| {r['arm']} | {r.get('n_common_fitted_words')} | {r.get('mean_delta_model_aoa_log10_step')} | {corr.get('pearson_r')} | {corr.get('pearson_p')} | {sl.get('slope')} | {direction} |")
    lines.append("")
    lines.append(f"Full JSON: `{rel(out_dir / 'summary.json')}`")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--arms", nargs="*", default=["control", "schedule", "enrichment"], choices=sorted(ARM_HF_MODELS))
    ap.add_argument("--checkpoint-names", nargs="*", default=DEFAULT_CHECKPOINTS)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--device", default="")
    ap.add_argument("--word-start", type=int, default=0)
    ap.add_argument("--word-end", type=int, default=0)
    ap.add_argument("--max-words", type=int, default=0)
    ap.add_argument("--progress-every", type=int, default=50)
    ap.add_argument("--force-extract", action="store_true")
    args = ap.parse_args()

    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = list(args.checkpoint_names)
    arms = list(args.arms)
    verification = verify_arm_configs(arms, checkpoints)
    (out_dir / "arm_config_verification.json").write_text(json.dumps(verification, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    raw_manifests = {}
    for arm in arms:
        raw_manifests[arm] = run_extraction_for_arm(arm, out_dir, checkpoints, args)

    child_aoas = FIT.load_cdi_child_aoas(FIT.CDI_PATH)
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH), use_fast=True, local_files_only=True)
    word_z, word_z_rows, z_meta = load_word_z(tokenizer)
    pd.DataFrame(word_z_rows).to_csv(out_dir / "cdi_word_enrichment_z.csv", index=False)

    fit_summaries = []
    fit_rows_by_arm: dict[str, list[dict[str, Any]]] = {}
    for arm in arms:
        raw_path = out_dir / "raw" / arm / "surprisal.json"
        summary, rows = fit_arm(arm, raw_path, out_dir, child_aoas, tokenizer)
        fit_summaries.append(summary)
        fit_rows_by_arm[arm] = rows
        print(json.dumps({"event": "fit_arm_done", "arm": arm, "raw_r": summary.get("raw_pearson_r"), "p": summary.get("raw_pearson_p"), "n": summary.get("fitted_word_count")}), flush=True)

    delta_summaries = []
    all_delta_rows: list[dict[str, Any]] = []
    if "control" in fit_rows_by_arm:
        for arm in arms:
            if arm == "control":
                continue
            ds, rows = delta_analysis(fit_rows_by_arm["control"], fit_rows_by_arm[arm], word_z, arm)
            delta_summaries.append(ds)
            all_delta_rows.extend(rows)
    pd.DataFrame(all_delta_rows).to_csv(out_dir / "arm_minus_control_word_deltas.csv", index=False)

    payload = {
        "status": "AOA_CALIBRATION_EXTRACT_FIT_DONE",
        "created_utc": now(),
        "script": rel(_public_path('experiments/archive/relation_learning/scripts/aoa_calibration_extract_fit.py')),
        "out_dir": rel(out_dir),
        "arms": arms,
        "checkpoints": checkpoints,
        "checkpoint_word_counts_nominal": [checkpoint_word_count(c) for c in checkpoints],
        "semantics": "Batched official-style MLM AoA surprisal extraction; research official-semantics raw AoA refit; first decision readout is arm-minus-control fitted model AoA against corpus-internal CHILDES-minus-whole token enrichment z averaged over each CDI word's subwords.",
        "arm_config_verification": verification,
        "raw_manifests": {k: {"out_dir": v.get("out_dir"), "status": v.get("status"), "summary": v.get("summary"), "eval_info": v.get("eval_info"), "steps": v.get("steps")} for k, v in raw_manifests.items()},
        "z_metadata": z_meta,
        "fit_summaries": fit_summaries,
        "delta_summaries": delta_summaries,
        "outputs": {
            "summary_md": rel(out_dir / "summary.md"),
            "word_z_csv": rel(out_dir / "cdi_word_enrichment_z.csv"),
            "delta_csv": rel(out_dir / "arm_minus_control_word_deltas.csv"),
            "arm_config_verification": rel(out_dir / "arm_config_verification.json"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    write_md(out_dir, payload)
    print(json.dumps({"status": payload["status"], "summary_json": rel(out_dir / "summary.json"), "summary_md": rel(out_dir / "summary.md"), "delta_summaries": delta_summaries}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
