#!/usr/bin/env python3
"""research: temperature-adjusted source/rank readout for the preservation candidate family.

This is the research/087 source-use instrument rerun on the actual Stage-III candidate
family with the correct parent anchor: chck_82M slow-adapter parent, coherent86, dense
64/65, and clean-preservation 64/65. It tests the mechanism proposed after research:

dense private training should increase evidence-responsive source/rank movement but also
introduce evidence-absent lexical drift; eval-mode preservation should retain the source-
visible movement while reducing the view-only / ordinary drift. This is an explanatory
frozen-checkpoint readout, not BabyLM official scoring.
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

ROOT = _public_path('.')
A01 = _public_path('experiments/archive/functional_learning')
A01_SCRIPTS = _public_path('experiments/archive/functional_learning/scripts')
if str(A01_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(A01_SCRIPTS))

import temperature_source_readout as src  # noqa: E402

DEFAULT_OUT = _public_path('experiments/archive/relation_learning/data/source_margin_candidate_family')
CHCK82 = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M')
COHERENT86 = _public_path('models/frontier')
MODEL_SPECS: dict[str, pathlib.Path] = {
    "chck82_slow_scale1p75": CHCK82,
    "coherent86_alpha075": COHERENT86,
    "dense64_u0080": _public_path('experiments/archive/functional_learning/data/automodel_repair/repaired_dense_seed62064_u0080'),
    "dense65_u0080": _public_path('experiments/archive/functional_learning/data/automodel_repair/repaired_dense_seed62065_u0080'),
    "clean_pres64_u0080": _public_path('experiments/archive/functional_learning/data/automodel_repair_clean_preservation/repaired_clean_pres_lambda1_eval_seed62064_u0080'),
    "clean_pres65_u0080": _public_path('experiments/archive/functional_learning/data/automodel_repair_candidates/repaired_clean_pres_lambda1_eval_seed62065_u0080'),
}
ANCHOR = "chck82_slow_scale1p75"


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
            if math.isfinite(y):
                out.append(y)
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
    try:
        return float(a) - float(b)
    except Exception:
        return None


def flatten_summary(name: str, summary: dict[str, Any]) -> dict[str, Any]:
    q = summary["qwen_source"]["source_specificity"]
    c = summary["common_source_reversal"]["source_follow"]
    cond = summary["qwen_source"]["by_condition"]
    row = {
        "model": name,
        "temperature": summary.get("temperature"),
        "calib_nll_T1": summary.get("calibration", {}).get("mean_nll_at_T1"),
        "qwen_delta_spec_T1_vs_chck82": q.get("mean_delta_specific_advantage_T1_vs_parent"),
        "qwen_delta_spec_Tfit_vs_chck82": q.get("mean_delta_specific_advantage_Tfit_vs_parent"),
        "qwen_delta_spec_rank_vs_chck82": q.get("mean_delta_specific_rank_advantage_vs_parent"),
        "qwen_delta_total_Tfit_vs_chck82": q.get("mean_delta_total_advantage_Tfit_vs_parent"),
        "qwen_delta_total_rank_vs_chck82": q.get("mean_delta_total_rank_advantage_vs_parent"),
        "common_delta_swing_T1_vs_chck82": c.get("mean_delta_source_follow_swing_T1_vs_parent"),
        "common_delta_swing_Tfit_vs_chck82": c.get("mean_delta_source_follow_swing_Tfit_vs_parent"),
        "common_delta_rank_swing_vs_chck82": c.get("mean_delta_source_follow_rank_swing_vs_parent"),
        "common_both_T1": c.get("both_correct_T1"),
        "common_both_Tfit": c.get("both_correct_Tfit"),
        "common_both_rank": c.get("both_rank_better"),
    }
    for key in ["correct_source", "wrong_source", "view_only"]:
        ck = cond.get(key, {})
        row[f"{key}_delta_nll_Tfit_vs_chck82"] = ck.get("mean_delta_nll_Tfit_vs_parent")
        row[f"{key}_delta_rank_vs_chck82"] = ck.get("mean_delta_rank_vs_parent")
    return row


def build_pairwise(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    pairs = {}
    for seed in ["64", "65"]:
        d = rows[f"dense{seed}_u0080"]
        p = rows[f"clean_pres{seed}_u0080"]
        pairs[f"clean_minus_dense_seed{seed}"] = {
            k: sub(p.get(k), d.get(k)) for k in d.keys() if k != "model"
        }
    pairs["dense65_minus_dense64"] = {k: sub(rows["dense65_u0080"].get(k), rows["dense64_u0080"].get(k)) for k in rows["dense64_u0080"] if k != "model"}
    pairs["clean65_minus_clean64"] = {k: sub(rows["clean_pres65_u0080"].get(k), rows["clean_pres64_u0080"].get(k)) for k in rows["clean_pres64_u0080"] if k != "model"}
    return pairs


def write_markdown(path: pathlib.Path, result: dict[str, Any]) -> None:
    rows = result["compact_rows"]
    lines: list[str] = []
    lines.append("# research source/rank decomposition for candidate family\n\n")
    lines.append("Anchor is `chck82_slow_scale1p75`, the protected parent before private dense-credit training. Temperatures are fitted only on ordinary legal-tail text, then source-visible probes are scored on Qwen-pair and common source-reversal banks. This readout is explanatory, not official BabyLM scoring.\n\n")
    lines.append("## Compact source-use table\n\n")
    lines.append("| model | T | Qwen Δspecific Tfit | Qwen Δspecific rank | view-only ΔNLL Tfit | view-only Δrank | common Δswing Tfit | common Δrank swing | common both Tfit/rank |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|\n")
    for r in rows:
        lines.append(
            f"| {r['model']} | {r.get('temperature')} | {r.get('qwen_delta_spec_Tfit_vs_chck82')} | {r.get('qwen_delta_spec_rank_vs_chck82')} | {r.get('view_only_delta_nll_Tfit_vs_chck82')} | {r.get('view_only_delta_rank_vs_chck82')} | {r.get('common_delta_swing_Tfit_vs_chck82')} | {r.get('common_delta_rank_swing_vs_chck82')} | {r.get('common_both_Tfit')}/{r.get('common_both_rank')} |\n"
        )
    lines.append("\n## Seed-paired clean-preservation minus dense\n\n")
    lines.append("Positive Qwen/common deltas mean preservation kept or increased source-responsive movement; negative view-only / ordinary drift deltas mean preservation reduced evidence-absent drift relative to dense.\n\n")
    for k, v in result["pairwise"].items():
        lines.append(f"### {k}\n\n")
        keep = {kk: vv for kk, vv in v.items() if kk in [
            "qwen_delta_spec_Tfit_vs_chck82", "qwen_delta_spec_rank_vs_chck82", "qwen_delta_total_Tfit_vs_chck82",
            "common_delta_swing_Tfit_vs_chck82", "common_delta_rank_swing_vs_chck82",
            "correct_source_delta_nll_Tfit_vs_chck82", "wrong_source_delta_nll_Tfit_vs_chck82", "view_only_delta_nll_Tfit_vs_chck82",
            "correct_source_delta_rank_vs_chck82", "wrong_source_delta_rank_vs_chck82", "view_only_delta_rank_vs_chck82"
        ]}
        lines.append(json.dumps(keep, indent=2, ensure_ascii=False) + "\n\n")
    lines.append("## Initial interpretation\n\n")
    for item in result["interpretation"]:
        lines.append(f"- {item}\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--models", nargs="+", default=list(MODEL_SPECS), choices=list(MODEL_SPECS))
    ap.add_argument("--calib-records", type=int, default=192)
    ap.add_argument("--calib-seed", type=int, default=86031)
    ap.add_argument("--source-max-segments", type=int, default=120)
    ap.add_argument("--source-seed", type=int, default=67067)
    ap.add_argument("--max-targets-per-segment", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=320)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--private-scale", type=float, default=0.75)
    ap.add_argument("--temperature-grid", default="0.70,0.80,0.90,1.00,1.10,1.25,1.40,1.60,1.90")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--torch-threads", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.torch_threads > 0:
        torch.set_num_threads(int(args.torch_threads))

    tokenizer = src.bridge.AutoTokenizer.from_pretrained(str(src.bridge.PARENT_PATH), local_files_only=True, use_fast=True)
    calib_records, calib_summary = src.build_calibration_records(
        src.DEFAULT_TAIL, tokenizer, n_records=int(args.calib_records), seed=int(args.calib_seed), max_length=int(args.max_length)
    )
    source_records, source_segment_stats = src.source_probe.build_segment_records(
        argparse.Namespace(tail_jsonl=src.DEFAULT_TAIL, max_updates=src.PREFIX_UPDATES, words_per_update=src.WORDS_PER_UPDATE,
                           max_segments=int(args.source_max_segments), sample_seed=int(args.source_seed))
    )
    qwen_tasks, qwen_task_stats = src.source_probe.build_tasks(
        source_records, tokenizer,
        argparse.Namespace(sample_seed=int(args.source_seed), max_targets_per_segment=int(args.max_targets_per_segment), max_length=int(args.max_length))
    )
    common_bank, common_bank_summary = src.common_probe.materialize_bank(src.DEFAULT_LABELS)
    common_records, common_rec_summary = src.common_probe.build_scoring_records(tokenizer, common_bank, int(args.max_length))

    src.write_jsonl(out_dir / "calibration_records.jsonl", (src.strip_task_for_json(t) for t in calib_records))
    src.write_jsonl(out_dir / "qwen_source_tasks.jsonl", (src.strip_task_for_json(t) for t in qwen_tasks))
    src.write_jsonl(out_dir / "common_source_reversal_records.jsonl", (src.strip_task_for_json(t) for t in common_records))

    plan = {
        "status": "TEMPERATURE_SOURCE_READOUT_CANDIDATE_FAMILY_PLAN",
        "created_utc": now(),
        "out_dir": rel(out_dir),
        "anchor": ANCHOR,
        "models": {m: rel(MODEL_SPECS[m]) for m in args.models},
        "calibration": calib_summary,
        "qwen_source_task_stats": {"segment_stats": source_segment_stats, "task_stats": qwen_task_stats},
        "common_source_reversal_stats": {"bank": common_bank_summary, "records": common_rec_summary},
        "temperature_grid": [float(x) for x in args.temperature_grid.split(",") if x.strip()],
        "private_scale_for_private_branch": float(args.private_scale),
        "boundary": "Explanatory frozen-checkpoint readout; official BabyLM metrics are not modified.",
    }
    (out_dir / "plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if args.dry_run:
        return

    if args.device == "auto":
        device = torch.device(f"cuda:{int(args.gpu)}" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")
    temp_grid = [float(x) for x in args.temperature_grid.split(",") if x.strip()]
    if 1.0 not in temp_grid:
        temp_grid.append(1.0)
        temp_grid.sort()

    temperature_fits: dict[str, Any] = {}
    model_summaries: dict[str, Any] = {}
    load_infos: dict[str, Any] = {}
    qwen_parent_scores = None
    common_parent_scores = None
    compact_rows: list[dict[str, Any]] = []

    for model_name in args.models:
        t0 = time.time()
        model_path = MODEL_SPECS[model_name]
        print(json.dumps({"event": "load_model", "model": model_name, "path": rel(model_path), "device": str(device)}), flush=True)
        model, load_info = src.qview.load_scoring_model(model_path, device, float(args.private_scale))
        load_infos[model_name] = load_info

        calib_logits, calib_targets, _calib_metas = src.gather_position_logits(model, tokenizer, calib_records, device, int(args.batch_size))
        temp_fit = src.fit_temperature_from_position_logits(calib_logits, calib_targets, temp_grid)
        temperature_fits[model_name] = temp_fit
        best_t = float(temp_fit.get("best_temperature", 1.0) or 1.0)
        del calib_logits

        q_scores = src.score_tasks_with_temperature(model, tokenizer, qwen_tasks, device, int(args.batch_size), best_t)
        c_scores = src.score_tasks_with_temperature(model, tokenizer, common_records, device, int(args.batch_size), best_t)
        src.write_jsonl(out_dir / f"qwen_scores_{model_name}.jsonl", q_scores)
        src.write_jsonl(out_dir / f"common_scores_{model_name}.jsonl", c_scores)
        if model_name == ANCHOR:
            qwen_parent_scores = q_scores
            common_parent_scores = c_scores
        q_summ = src.summarize_qwen_source(q_scores, qwen_parent_scores)
        c_summ = src.summarize_common(c_scores, common_parent_scores)
        model_summaries[model_name] = {
            "temperature": best_t,
            "elapsed_sec": round(time.time() - t0, 2),
            "calibration": temp_fit,
            "qwen_source": q_summ,
            "common_source_reversal": c_summ,
        }
        slim = {
            "temperature": best_t,
            "elapsed_sec": round(time.time() - t0, 2),
            "calibration": temp_fit,
            "qwen_source": src.compact_summary(q_summ),
            "common_source_reversal": src.compact_summary(c_summ),
        }
        (out_dir / f"summary_{model_name}.json").write_text(json.dumps(slim, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        flat = flatten_summary(model_name, model_summaries[model_name])
        compact_rows.append(flat)
        print(json.dumps({"event": "model_done", **flat, "elapsed_sec": round(time.time() - t0, 2)}, ensure_ascii=False), flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with (out_dir / "compact_model_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(compact_rows[0].keys()) if compact_rows else []
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in compact_rows:
            w.writerow(r)

    compact_by_model = {r["model"]: r for r in compact_rows}
    pairwise = build_pairwise(compact_by_model)
    interpretation = [
        "The source-responsive quantities are Qwen correct-vs-wrong specific advantage and common source-follow swing, especially rank movement because it cannot be explained by global temperature.",
        "The evidence-absent drift quantities are view-only NLL/rank and CDI endpoint NLL/rank in the companion research CDI profile; source probes alone cannot settle broad lexical preservation.",
        "If clean-preservation has dense-like source-responsive deltas while view-only and CDI drifts shrink relative to dense, the recipe is not simply less private movement: it selectively preserves evidence-supported acquisition while damping unsupported distributional movement.",
    ]
    result = {
        "status": "TEMPERATURE_SOURCE_READOUT_CANDIDATE_FAMILY_DONE",
        "created_utc": now(),
        "anchor": ANCHOR,
        "plan": rel(out_dir / "plan.json"),
        "load_infos": load_infos,
        "temperature_fits": temperature_fits,
        "model_summaries": model_summaries,
        "compact_rows": compact_rows,
        "pairwise": pairwise,
        "compact_model_comparison": rel(out_dir / "compact_model_comparison.csv"),
        "interpretation": interpretation,
    }
    out_json = out_dir / "temperature_source_readout_candidate_family.json"
    out_md = out_dir / "temperature_source_readout_candidate_family.md"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(out_md, result)
    print(json.dumps({"status": result["status"], "out_json": rel(out_json), "out_md": rel(out_md)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
