#!/usr/bin/env python3
"""research shrinkage-null mechanism probe.

Materializing dense seed62064 at lower private_adapter_scale gives evaluation-only
models with the same private weights but smaller execution scale. This script scores
the chck82 anchor, dense64 alpha0.75, dense64 lower-scale variants, and clean-pres64 on
(1) the research/087 source/rank probes and (2) the research CDI endpoint rank probe.

The purpose is to test whether clean preservation is merely an effective smaller step.
If a dense lower-scale variant with KL close to clean-pres64 also matches clean-pres64's
CDI drift but loses much more source-responsive movement, then preservation is selective
rather than simple shrinkage. This is explanatory evidence, not official scoring.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import json
import math
import os
import pathlib
import statistics
import sys
import time
from typing import Any, Iterable

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = _public_path('.')
A01 = _public_path('experiments/archive/functional_learning')
A01_SCRIPTS = _public_path('experiments/archive/functional_learning/scripts')
if str(A01_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(A01_SCRIPTS))

import temperature_source_readout as src  # noqa: E402
import cdi_endpoint_temperature_rank_profile_five_models as cdi  # noqa: E402

VARIANT_ROOT = _public_path('experiments/archive/relation_learning/data/shrinkage_scale_variants')
DEFAULT_OUT = _public_path('experiments/archive/relation_learning/data/shrinkage_source_cdi_probe')
ENDPOINTS = {
    "chck82": _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M'),
    "dense64_scale0p75": _public_path('experiments/archive/functional_learning/data/automodel_repair/repaired_dense_seed62064_u0080'),
    "dense64_scale0p55": _public_path('experiments/archive/relation_learning/data/shrinkage_scale_variants/dense64_private_scale_0p55'),
    "dense64_scale0p57": _public_path('experiments/archive/relation_learning/data/shrinkage_scale_variants/dense64_private_scale_0p57'),
    "dense64_scale0p58": _public_path('experiments/archive/relation_learning/data/shrinkage_scale_variants/dense64_private_scale_0p58'),
    "dense64_scale0p60": _public_path('experiments/archive/relation_learning/data/shrinkage_scale_variants/dense64_private_scale_0p60'),
    "clean_pres64": _public_path('experiments/archive/functional_learning/data/automodel_repair_clean_preservation/repaired_clean_pres_lambda1_eval_seed62064_u0080'),
}
ANCHOR = "chck82"


def rel(p: pathlib.Path | str | None) -> str | None:
    if p is None:
        return None
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def finite(xs: Iterable[Any]) -> list[float]:
    out: list[float] = []
    for x in xs:
        try:
            y = float(x)
            if math.isfinite(y): out.append(y)
        except Exception:
            pass
    return out


def mean(xs: Iterable[Any]) -> float | None:
    v = finite(xs)
    return sum(v) / len(v) if v else None


def median(xs: Iterable[Any]) -> float | None:
    v = finite(xs)
    return statistics.median(v) if v else None


def sub(a: Any, b: Any) -> float | None:
    try: return float(a) - float(b)
    except Exception: return None


def prepare_cache(out_dir: pathlib.Path) -> None:
    for key, suffix in [
        ("HF_HOME", "hf_home"), ("HF_HUB_CACHE", "hub"), ("HF_DATASETS_CACHE", "datasets"),
        ("TRANSFORMERS_CACHE", "transformers"), ("HF_MODULES_CACHE", "modules")]:
        p = out_dir / "hf_cache" / suffix
        p.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(p.resolve())
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        import transformers.utils.hub as hub
        import transformers.dynamic_module_utils as dyn
        hub.HF_MODULES_CACHE = os.environ["HF_MODULES_CACHE"]
        dyn.HF_MODULES_CACHE = os.environ["HF_MODULES_CACHE"]
    except Exception:
        pass


def load_model(endpoint: pathlib.Path, device: torch.device):
    model = AutoModelForMaskedLM.from_pretrained(str(endpoint), trust_remote_code=True, local_files_only=True)
    model.to(device)
    model.eval()
    named = list(model.named_parameters())
    private = [(n, p) for n, p in named if ".private_adapter." in n]
    scales = []
    try:
        scales = [float(layer.private_adapter.scale) for layer in model.deberta.encoder.layer]
    except Exception:
        pass
    ident = {
        "endpoint": rel(endpoint),
        "class": type(model).__name__,
        "module": type(model).__module__,
        "total_params": int(sum(p.numel() for _, p in named)),
        "private_adapter_params": int(sum(p.numel() for _, p in private)),
        "private_adapter_tensor_count": len(private),
        "executed_private_scales": scales,
        "config_private_adapter_scale": float(getattr(model.config, "private_adapter_scale", -1.0)) if hasattr(model, "config") else None,
    }
    return model, ident


def source_flat(name: str, summ: dict[str, Any], temp_fit: dict[str, Any]) -> dict[str, Any]:
    q = summ["qwen_source"]["source_specificity"]
    c = summ["common_source_reversal"]["source_follow"]
    cond = summ["qwen_source"]["by_condition"]
    return {
        "model": name,
        "scale": None,
        "temperature": summ.get("temperature"),
        "calib_nll_T1": temp_fit.get("mean_nll_at_T1"),
        "qwen_delta_spec_Tfit": q.get("mean_delta_specific_advantage_Tfit_vs_parent"),
        "qwen_delta_spec_rank": q.get("mean_delta_specific_rank_advantage_vs_parent"),
        "qwen_delta_total_Tfit": q.get("mean_delta_total_advantage_Tfit_vs_parent"),
        "qwen_delta_total_rank": q.get("mean_delta_total_rank_advantage_vs_parent"),
        "common_delta_swing_Tfit": c.get("mean_delta_source_follow_swing_Tfit_vs_parent"),
        "common_delta_rank_swing": c.get("mean_delta_source_follow_rank_swing_vs_parent"),
        "common_both_Tfit": c.get("both_correct_Tfit"),
        "correct_source_delta_nll_Tfit": cond.get("correct_source", {}).get("mean_delta_nll_Tfit_vs_parent"),
        "wrong_source_delta_nll_Tfit": cond.get("wrong_source", {}).get("mean_delta_nll_Tfit_vs_parent"),
        "view_only_delta_nll_Tfit": cond.get("view_only", {}).get("mean_delta_nll_Tfit_vs_parent"),
        "correct_source_delta_rank": cond.get("correct_source", {}).get("mean_delta_rank_vs_parent"),
        "wrong_source_delta_rank": cond.get("wrong_source", {}).get("mean_delta_rank_vs_parent"),
        "view_only_delta_rank": cond.get("view_only", {}).get("mean_delta_rank_vs_parent"),
    }


def cdi_flat(name: str, s: dict[str, Any], temp: float) -> dict[str, Any]:
    return {
        "model": name,
        "temperature": temp,
        "cdi_mean_nll_Tfit": s.get("mean_nll_Tfit"),
        "cdi_mean_rank": s.get("mean_rank"),
        "cdi_delta_nll_Tfit": s.get("mean_delta_nll_Tfit_vs_parent"),
        "cdi_delta_rank": s.get("mean_delta_rank_vs_parent"),
        "cdi_improved_fraction_rank": s.get("improved_fraction_rank"),
        "cdi_word_mean_delta_nll_Tfit": s.get("word_mean_delta_nll_Tfit"),
        "cdi_word_mean_delta_rank": s.get("word_mean_delta_rank"),
    }


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8"); return
    fields=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen:
                fields.append(k); seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_md(path: pathlib.Path, result: dict[str, Any]) -> None:
    lines=[]
    lines.append("# research shrinkage-null source/CDI probe\n\n")
    lines.append("Dense seed62064 was evaluated at lower private adapter scales without changing weights. Scale 0.60 was chosen because the research KL screen makes its coherent-row KL nearly equal to clean-pres64.\n\n")
    lines.append("## Source-responsive and CDI drift table\n\n")
    lines.append("| model | scale | Qwen Δspec Tfit | Qwen Δspec rank | common Δswing Tfit | common Δrank | view-only ΔNLL | CDI ΔNLL | CDI Δrank |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in result["combined_rows"]:
        lines.append(f"| {r['model']} | {r.get('scale')} | {r.get('qwen_delta_spec_Tfit')} | {r.get('qwen_delta_spec_rank')} | {r.get('common_delta_swing_Tfit')} | {r.get('common_delta_rank_swing')} | {r.get('view_only_delta_nll_Tfit')} | {r.get('cdi_delta_nll_Tfit')} | {r.get('cdi_delta_rank')} |\n")
    lines.append("\n## Matched-KL contrast: clean_pres64 minus dense64_scale0p60\n\n")
    lines.append(json.dumps(result.get("matched_kl_clean_minus_scale0p60", {}), indent=2, ensure_ascii=False) + "\n\n")
    lines.append("## Interpretation\n\n")
    for item in result["interpretation"]:
        lines.append(f"- {item}\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--models", nargs="+", default=list(ENDPOINTS), choices=list(ENDPOINTS))
    ap.add_argument("--device", default="cpu", choices=["cpu","cuda","auto"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch-size-source", type=int, default=32)
    ap.add_argument("--batch-size-cdi", type=int, default=64)
    ap.add_argument("--torch-threads", type=int, default=8)
    ap.add_argument("--calib-records", type=int, default=192)
    ap.add_argument("--calib-seed", type=int, default=86031)
    ap.add_argument("--source-max-segments", type=int, default=120)
    ap.add_argument("--source-seed", type=int, default=67067)
    ap.add_argument("--max-targets-per-segment", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=320)
    ap.add_argument("--cdi-max-words", type=int, default=96)
    ap.add_argument("--cdi-sample-seed", type=int, default=86032)
    ap.add_argument("--temperature-grid", default="0.70,0.80,0.90,1.00,1.10,1.25,1.40,1.60,1.90")
    args=ap.parse_args()
    out=args.out_dir if args.out_dir.is_absolute() else ROOT/args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    prepare_cache(out)
    if args.torch_threads>0: torch.set_num_threads(int(args.torch_threads))
    if args.device=="auto": device=torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    elif args.device=="cuda": device=torch.device(f"cuda:{args.gpu}")
    else: device=torch.device("cpu")

    tok = AutoTokenizer.from_pretrained(str(src.bridge.PARENT_PATH), local_files_only=True, use_fast=True)
    calib_records, calib_summary = src.build_calibration_records(src.DEFAULT_TAIL, tok, n_records=args.calib_records, seed=args.calib_seed, max_length=args.max_length)
    source_records, source_segment_stats = src.source_probe.build_segment_records(argparse.Namespace(tail_jsonl=src.DEFAULT_TAIL, max_updates=src.PREFIX_UPDATES, words_per_update=src.WORDS_PER_UPDATE, max_segments=args.source_max_segments, sample_seed=args.source_seed))
    qwen_tasks, qwen_task_stats = src.source_probe.build_tasks(source_records, tok, argparse.Namespace(sample_seed=args.source_seed, max_targets_per_segment=args.max_targets_per_segment, max_length=args.max_length))
    common_bank, common_bank_summary = src.common_probe.materialize_bank(src.DEFAULT_LABELS)
    common_records, common_rec_summary = src.common_probe.build_scoring_records(tok, common_bank, args.max_length)
    cdi_words, cdi_contexts, cdi_subset = cdi.make_subset(args.cdi_max_words, args.cdi_sample_seed)
    temp_grid=[float(x) for x in args.temperature_grid.split(',') if x.strip()]
    if 1.0 not in temp_grid:
        temp_grid.append(1.0); temp_grid.sort()
    plan={
        "status":"SHRINKAGE_SOURCE_CDI_PROBE_PLAN",
        "created_utc":now(),
        "anchor":ANCHOR,
        "models":{m:rel(ENDPOINTS[m]) for m in args.models},
        "calibration":calib_summary,
        "qwen_source_task_stats":{"segment_stats":source_segment_stats,"task_stats":qwen_task_stats},
        "common_source_reversal_stats":{"bank":common_bank_summary,"records":common_rec_summary},
        "cdi_subset":cdi_subset,
        "device":str(device),
        "boundary":"Evaluation-only private-scale variants; no official score changed.",
    }
    (out/"plan.json").write_text(json.dumps(plan,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps(plan,indent=2,ensure_ascii=False),flush=True)

    source_parent_q=None; source_parent_c=None; cdi_parent=None
    rows_source=[]; rows_cdi=[]; identities={}; temperature_fits={}
    for name in args.models:
        endpoint=ENDPOINTS[name]
        print(json.dumps({"event":"load_source_model","model":name,"endpoint":rel(endpoint)},ensure_ascii=False),flush=True)
        model, ident=load_model(endpoint, device)
        identities[name]=ident
        logits, targets, _ = src.gather_position_logits(model, tok, calib_records, device, args.batch_size_source)
        tf=src.fit_temperature_from_position_logits(logits, targets, temp_grid)
        temperature_fits[name]=tf
        temp=float(tf.get("best_temperature",1.0) or 1.0)
        q_scores=src.score_tasks_with_temperature(model, tok, qwen_tasks, device, args.batch_size_source, temp)
        cm_scores=src.score_tasks_with_temperature(model, tok, common_records, device, args.batch_size_source, temp)
        src.write_jsonl(out/f"qwen_scores_{name}.jsonl", q_scores)
        src.write_jsonl(out/f"common_scores_{name}.jsonl", cm_scores)
        if name==ANCHOR:
            source_parent_q=q_scores; source_parent_c=cm_scores
        q_s=src.summarize_qwen_source(q_scores, source_parent_q)
        cm_s=src.summarize_common(cm_scores, source_parent_c)
        ssum={"temperature":temp,"calibration":tf,"qwen_source":q_s,"common_source_reversal":cm_s}
        sf=source_flat(name, ssum, tf)
        sf["scale"]=ident.get("config_private_adapter_scale")
        rows_source.append(sf)
        del model, logits
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print(json.dumps({"event":"source_done", **sf},ensure_ascii=False),flush=True)

        print(json.dumps({"event":"score_cdi","model":name,"endpoint":rel(endpoint),"temperature":temp},ensure_ascii=False),flush=True)
        crows, cinfo = cdi.score_model(name, endpoint, temp, cdi_words, cdi_contexts, str(device), args.batch_size_cdi)
        write_jsonl(out/f"cdi_scores_{name}.jsonl", crows)
        if name==ANCHOR:
            cdi_parent=crows
        cs=cdi.summarize(crows, cdi_parent)
        cf=cdi_flat(name, cs, temp)
        cf["scale"]=ident.get("config_private_adapter_scale")
        rows_cdi.append(cf)
        print(json.dumps({"event":"cdi_done", **cf},ensure_ascii=False),flush=True)

    by_model={r["model"]:{**r} for r in rows_source}
    for r in rows_cdi:
        by_model.setdefault(r["model"],{}).update(r)
    combined=[by_model[m] for m in args.models if m in by_model]
    write_csv(out/"source_compact.csv", rows_source)
    write_csv(out/"cdi_compact.csv", rows_cdi)
    write_csv(out/"combined_compact.csv", combined)
    matched={}
    if "clean_pres64" in by_model and "dense64_scale0p60" in by_model:
        a=by_model["clean_pres64"]; b=by_model["dense64_scale0p60"]
        matched={k:sub(a.get(k),b.get(k)) for k in set(a)|set(b) if k not in {"model"}}
    interpretation=[
        "The matched-KL shrinkage probe is strongest at dense64_scale0p60 because its coherent-row KL nearly matches clean_pres64 in the research KL screen.",
        "If clean_pres64 has clearly larger source-responsive Qwen/common movement than dense64_scale0p60 while not having worse CDI drift, preservation cannot be reduced to a smaller effective private-adapter scale.",
        "If dense64_scale0p60 matches clean_pres64 on both source movement and CDI drift, the shrinkage null remains plausible and the full cheap7 GPU control becomes more important.",
    ]
    result={
        "status":"SHRINKAGE_SOURCE_CDI_PROBE_DONE",
        "created_utc":now(),
        "anchor":ANCHOR,
        "plan":rel(out/"plan.json"),
        "identities":identities,
        "temperature_fits":temperature_fits,
        "source_rows":rows_source,
        "cdi_rows":rows_cdi,
        "combined_rows":combined,
        "matched_kl_clean_minus_scale0p60":matched,
        "outputs":{"source_csv":rel(out/"source_compact.csv"),"cdi_csv":rel(out/"cdi_compact.csv"),"combined_csv":rel(out/"combined_compact.csv")},
        "interpretation":interpretation,
    }
    (out/"shrinkage_source_cdi_probe.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    write_md(out/"shrinkage_source_cdi_probe.md", result)
    print(json.dumps({"status":result["status"],"out_json":rel(out/"shrinkage_source_cdi_probe.json"),"out_md":rel(out/"shrinkage_source_cdi_probe.md")},indent=2),flush=True)

if __name__=="__main__": main()
