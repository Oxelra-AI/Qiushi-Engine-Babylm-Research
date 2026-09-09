#!/usr/bin/env python3
"""research: endpoint CDI NLL/rank profile for the preservation candidate family.

This reruns the research CDI endpoint temperature/rank diagnostic on the actual Stage-III
candidate family with the correct parent anchor, chck_82M. It measures whether eval-mode
preservation reduces evidence-absent lexical drift on CDI masked-token contexts while the
source/rank readout tests source-visible evidence-responsive movement.

This is not official AoA scoring: it is an endpoint-only masked-token NLL/rank readout on
a deterministic subset of CDI words and contexts. Official AoA trajectory refits are kept
separately in research.
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

import cdi_endpoint_temperature_rank_profile_five_models as cdi  # noqa: E402

DEFAULT_TEMP = _public_path('experiments/archive/relation_learning/data/source_margin_candidate_family/temperature_source_readout_candidate_family.json')
DEFAULT_OUT = _public_path('experiments/archive/relation_learning/data/cdi_endpoint_rank_candidate_family')
MODEL_PATHS: dict[str, pathlib.Path] = {
    "chck82_slow_scale1p75": _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M'),
    "coherent86_alpha075": _public_path('models/frontier'),
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


def load_temperatures(path: pathlib.Path) -> dict[str, float]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    temps: dict[str, float] = {}
    # Preferred input is the research candidate source-readout JSON.
    for name, fit in obj.get("temperature_fits", {}).items():
        try:
            temps[name] = float(fit.get("best_temperature", 1.0))
        except Exception:
            pass
    # Back-compatible with research/087 temperature JSONs.
    for name, fit in obj.get("temperature_fits", {}).items():
        if isinstance(fit, dict) and name not in temps:
            try:
                temps[name] = float(fit.get("best_temperature", 1.0))
            except Exception:
                pass
    for name in MODEL_PATHS:
        temps.setdefault(name, 1.0)
    return temps


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def compact_row(name: str, s: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": name,
        "mean_nll_T1": s.get("mean_nll_T1"),
        "mean_nll_Tfit": s.get("mean_nll_Tfit"),
        "median_nll_T1": s.get("median_nll_T1"),
        "mean_rank": s.get("mean_rank"),
        "median_rank": s.get("median_rank"),
        "delta_nll_T1_vs_chck82": s.get("mean_delta_nll_T1_vs_parent"),
        "delta_nll_Tfit_vs_chck82": s.get("mean_delta_nll_Tfit_vs_parent"),
        "delta_rank_vs_chck82": s.get("mean_delta_rank_vs_parent"),
        "median_delta_rank_vs_chck82": s.get("median_delta_rank_vs_parent"),
        "improved_fraction_nll_T1": s.get("improved_fraction_nll_T1"),
        "improved_fraction_nll_Tfit": s.get("improved_fraction_nll_Tfit"),
        "improved_fraction_rank": s.get("improved_fraction_rank"),
        "word_mean_delta_nll_T1": s.get("word_mean_delta_nll_T1"),
        "word_mean_delta_nll_Tfit": s.get("word_mean_delta_nll_Tfit"),
        "word_mean_delta_rank": s.get("word_mean_delta_rank"),
        "word_improved_fraction_rank": s.get("word_improved_fraction_rank"),
    }


def pairwise(rows_by_model: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for seed in ["64", "65"]:
        d = rows_by_model[f"dense{seed}_u0080"]
        p = rows_by_model[f"clean_pres{seed}_u0080"]
        out[f"clean_minus_dense_seed{seed}"] = {k: sub(p.get(k), d.get(k)) for k in d if k != "model"}
    out["dense65_minus_dense64"] = {k: sub(rows_by_model["dense65_u0080"].get(k), rows_by_model["dense64_u0080"].get(k)) for k in rows_by_model["dense64_u0080"] if k != "model"}
    out["clean65_minus_clean64"] = {k: sub(rows_by_model["clean_pres65_u0080"].get(k), rows_by_model["clean_pres64_u0080"].get(k)) for k in rows_by_model["clean_pres64_u0080"] if k != "model"}
    return out


def write_md(path: pathlib.Path, result: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# research CDI endpoint rank profile for candidate family\n\n")
    lines.append("Anchor is `chck82_slow_scale1p75`. This endpoint-only CDI readout measures evidence-absent lexical/word-context drift. It is separate from official AoA trajectory scoring.\n\n")
    lines.append("## Compact table\n\n")
    lines.append("| model | T | mean NLL Tfit | mean rank | ΔNLL Tfit vs chck82 | Δrank vs chck82 | rank improved fraction |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
    temps = result.get("temperatures", {})
    for r in result["compact_rows"]:
        lines.append(f"| {r['model']} | {temps.get(r['model'])} | {r.get('mean_nll_Tfit')} | {r.get('mean_rank')} | {r.get('delta_nll_Tfit_vs_chck82')} | {r.get('delta_rank_vs_chck82')} | {r.get('improved_fraction_rank')} |\n")
    lines.append("\n## Seed-paired clean-preservation minus dense\n\n")
    lines.append("Negative ΔNLL/Δrank here means preservation reduced dense endpoint CDI drift.\n\n")
    for k, v in result["pairwise"].items():
        keep = {kk: vv for kk, vv in v.items() if kk in [
            "delta_nll_T1_vs_chck82", "delta_nll_Tfit_vs_chck82", "delta_rank_vs_chck82",
            "median_delta_rank_vs_chck82", "improved_fraction_nll_Tfit", "improved_fraction_rank",
            "word_mean_delta_nll_Tfit", "word_mean_delta_rank", "word_improved_fraction_rank"
        ]}
        lines.append(f"### {k}\n\n")
        lines.append(json.dumps(keep, indent=2, ensure_ascii=False) + "\n\n")
    lines.append("## Interpretation\n\n")
    for item in result["interpretation"]:
        lines.append(f"- {item}\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--temperature-json", type=pathlib.Path, default=DEFAULT_TEMP)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--models", nargs="+", default=list(MODEL_PATHS), choices=list(MODEL_PATHS))
    ap.add_argument("--max-words", type=int, default=96)
    ap.add_argument("--sample-seed", type=int, default=86032)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--torch-threads", type=int, default=8)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.torch_threads > 0:
        torch.set_num_threads(int(args.torch_threads))
    cdi.ensure_cache(args.out_dir)
    temps = load_temperatures(args.temperature_json)
    words, contexts, subset = cdi.make_subset(int(args.max_words), int(args.sample_seed))
    if args.device == "auto":
        device = f"cuda:{int(args.gpu)}" if torch.cuda.is_available() else "cpu"
    elif args.device == "cuda":
        device = f"cuda:{int(args.gpu)}"
    else:
        device = "cpu"
    plan = {
        "status": "CDI_ENDPOINT_RANK_CANDIDATE_FAMILY_PLAN",
        "created_utc": now(),
        "anchor": ANCHOR,
        "temperature_json": rel(args.temperature_json),
        "temperatures": temps,
        "models": {m: rel(MODEL_PATHS[m]) for m in args.models},
        "subset": subset,
        "device": device,
        "boundary": "Endpoint-only CDI readout; official AoA trajectory scores are not changed.",
    }
    (args.out_dir / "plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)

    parent_rows = None
    summaries: dict[str, Any] = {}
    infos: dict[str, Any] = {}
    compact_rows: list[dict[str, Any]] = []
    for name in args.models:
        print(json.dumps({"event": "score_model", "model": name, "temperature": temps.get(name, 1.0), "device": device}, ensure_ascii=False), flush=True)
        rows, info = cdi.score_model(name, MODEL_PATHS[name], float(temps.get(name, 1.0)), words, contexts, device, int(args.batch_size))
        write_jsonl(args.out_dir / f"cdi_scores_{name}.jsonl", rows)
        if name == ANCHOR:
            parent_rows = rows
        summaries[name] = cdi.summarize(rows, parent_rows)
        infos[name] = info
        cr = compact_row(name, summaries[name])
        compact_rows.append(cr)
        print(json.dumps({"event": "model_done", **cr}, ensure_ascii=False), flush=True)
    with (args.out_dir / "compact_cdi_endpoint_rank.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(compact_rows[0].keys()) if compact_rows else []
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in compact_rows:
            w.writerow(r)
    by_model = {r["model"]: r for r in compact_rows}
    result = {
        "status": "CDI_ENDPOINT_RANK_CANDIDATE_FAMILY_DONE",
        "created_utc": now(),
        "anchor": ANCHOR,
        "plan": rel(args.out_dir / "plan.json"),
        "model_infos": infos,
        "summaries": summaries,
        "compact_rows": compact_rows,
        "pairwise": pairwise(by_model),
        "outputs": {
            "compact_csv": rel(args.out_dir / "compact_cdi_endpoint_rank.csv"),
        },
        "interpretation": [
            "Dense endpoints increase CDI NLL/rank relative to the chck82 parent; this is evidence-absent lexical drift rather than official AoA movement.",
            "If clean-preservation reduces the dense CDI drift while research source probes show it retains a substantial source-responsive shift, the preservation objective is not equivalent to uniform step-size shrinkage on the private adapter.",
            "The readout uses a deterministic 96-word CDI subset for quick mechanism pressure; official endpoint columns and research raw AoA refits remain the score evidence.",
        ],
    }
    out_json = args.out_dir / "cdi_endpoint_rank_candidate_family.json"
    out_md = args.out_dir / "cdi_endpoint_rank_candidate_family.md"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_md(out_md, result)
    print(json.dumps({"status": result["status"], "out_json": rel(out_json), "out_md": rel(out_md)}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
