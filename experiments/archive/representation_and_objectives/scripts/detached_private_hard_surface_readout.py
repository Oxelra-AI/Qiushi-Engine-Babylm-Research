#!/usr/bin/env python3
"""Hard-surface readout for the detached-private sparse dual-view route.

Scientific purpose
------------------
The separated sparse20 aligned 20M arm improved cheap7 early-training scores, but
aggregate columns do not tell whether the route improves the central unsolved
mechanism: context-conditioned alternative binding.  This script performs a
posthoc readout on existing checkpoints only, with no training and no custom
submission scoring:

  * GlobalPIQA: reuse the research official-compatible all-option, length-normalized
    completion reader, especially the fixed 52-row cross-endpoint hard subset.
  * EWoK: reuse the research four-cell reader on the fixed research shared stable-
    failure rows, so we can see whether the route actually reduces stable
    conditional-reversal failures.

The comparison is mechanism-facing, not submission-facing.  It compares the
pathway-separated aligned private route against exact MLM-only and shuffled-control
models, plus the coupled sparse20 aligned model when available.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import gc
import importlib.util
import json
import math
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

# Keep this readout off H100 unless a caller deliberately changes the script.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

USER_ROOT = _public_path('.')
os.chdir(USER_ROOT)
A01_WS = _public_path('experiments/archive/representation_and_objectives')
A02_WS = _public_path('experiments/archive/frontier_consolidation')
SCRIPT_DIR = _public_path('experiments/archive/representation_and_objectives/scripts')
DEFAULT_OUT_ROOT = _public_path('experiments/archive/representation_and_objectives/data/detached_private_hard_surface_readout')
STABLE_EWOK_CSV = _public_path('experiments/archive/representation_and_objectives/data/ewok_interaction_synthesis/both_models_stable_failure_rows.csv')

TARGETS: dict[str, dict[str, Any]] = {
    "mlm_only_20M": {
        "label": "A02 exact MLM-only 20M payload baseline for dual-view panel",
        "model_path": _public_path('experiments/archive/frontier_consolidation/training/runs/dualview_mlm_only_20M_seed43022/hf_model/final'),
        "metrics_path": _public_path('experiments/archive/frontier_consolidation/training/runs/dualview_mlm_only_20M_seed43022/scientific_metrics.json'),
        "eval_payload_path": _public_path('experiments/archive/frontier_consolidation/data/dualview_mlm_only_20m_eval/per_target/dualview_mlm_only_20M.json'),
        "family": "mlm_only",
    },
    "sep_sparse20_aligned_20M": {
        "label": "A02 pathway-separated sparse20 true-alignment private route at 20M",
        "model_path": _public_path('experiments/archive/frontier_consolidation/training/runs/sep_sparse20_aligned_20M_seed43022/hf_model/final'),
        "metrics_path": _public_path('experiments/archive/frontier_consolidation/training/runs/sep_sparse20_aligned_20M_seed43022/scientific_metrics.json'),
        "eval_payload_path": _public_path('experiments/archive/frontier_consolidation/data/sep_sparse20_aligned_20m_eval/per_target/sep_sparse20_aligned_20M.json'),
        "summary_path": _public_path('experiments/archive/frontier_consolidation/data/sep_sparse20_aligned_20m_summary/sep_sparse20_aligned_20M_summary.json'),
        "family": "separated_true_alignment",
    },
    "sep_sparse20_shuffled_20M": {
        "label": "A02 pathway-separated sparse20 shuffled-alignment private control at 20M",
        "model_path": _public_path('experiments/archive/frontier_consolidation/training/runs/sep_sparse20_shuffled_20M_seed43022/hf_model/final'),
        "metrics_path": _public_path('experiments/archive/frontier_consolidation/training/runs/sep_sparse20_shuffled_20M_seed43022/scientific_metrics.json'),
        "eval_payload_path": _public_path('experiments/archive/frontier_consolidation/data/sep_sparse20_shuffled_20m_eval/per_target/sep_sparse20_shuffled_20M.json'),
        "family": "separated_shuffled_alignment",
    },
    "coupled_sparse20_aligned_20M": {
        "label": "A02 coupled sparse20 true-alignment route at 20M",
        "model_path": _public_path('experiments/archive/frontier_consolidation/training/runs/coupled_sparse20_aligned_20M_seed43022/hf_model/final'),
        "metrics_path": _public_path('experiments/archive/frontier_consolidation/training/runs/coupled_sparse20_aligned_20M_seed43022/scientific_metrics.json'),
        "eval_payload_path": _public_path('experiments/archive/frontier_consolidation/data/coupled_sparse20_aligned_20m_eval/per_target/coupled_sparse20_aligned_20M.json'),
        "family": "coupled_true_alignment",
    },
}


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: Path | str | None) -> str | None:
    if p is None:
        return None
    pp = Path(p)
    try:
        return str(pp.resolve().relative_to(USER_ROOT))
    except Exception:
        return str(pp)


def import_from(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def qstats(vals) -> dict[str, Any]:
    xs = sorted(float(v) for v in vals if finite(v))
    if not xs:
        return {"n": 0}
    def q(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        idx = p * (len(xs) - 1)
        lo = math.floor(idx); hi = math.ceil(idx)
        if lo == hi:
            return xs[lo]
        return xs[lo] * (hi - idx) + xs[hi] * (idx - lo)
    return {
        "n": len(xs),
        "min": xs[0],
        "p05": q(0.05),
        "mean": statistics.fmean(xs),
        "median": statistics.median(xs),
        "p95": q(0.95),
        "max": xs[-1],
    }


def read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"_read_error": repr(e)}


def load_eval_scores(path: Path) -> dict[str, Any]:
    d = read_json(path)
    if not isinstance(d, dict):
        return {"path": rel(path), "exists": path.exists(), "scores": None, "tasks_complete": None}
    scores = d.get("scores") if isinstance(d.get("scores"), dict) else {}
    tasks = d.get("tasks") if isinstance(d.get("tasks"), dict) else {}
    task_scores: dict[str, Any] = {}
    task_status: dict[str, Any] = {}
    for name, payload in tasks.items():
        if isinstance(payload, dict):
            if isinstance(payload.get("score"), (int, float)):
                task_scores[name] = payload.get("score")
            task_status[name] = {
                "returncode": payload.get("returncode"),
                "score": payload.get("score"),
                "predictions_exists": Path(payload.get("predictions", "")).exists() if payload.get("predictions") else None,
            }
    if not scores and task_scores:
        scores = task_scores
    cheap_cols = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
    cheap7 = None
    if all(isinstance(scores.get(c), (int, float)) for c in cheap_cols):
        cheap7 = sum(float(scores[c]) for c in cheap_cols) / len(cheap_cols)
    return {
        "path": rel(path),
        "exists": path.exists(),
        "target": d.get("target"),
        "endpoint": d.get("endpoint"),
        "scores": scores,
        "cheap7": d.get("cheap7", cheap7),
        "task_status": task_status,
    }


def load_stable_indices(max_rows: int | None = None) -> tuple[list[int], dict[str, Any]]:
    if not STABLE_EWOK_CSV.exists():
        raise FileNotFoundError(STABLE_EWOK_CSV)
    rows = []
    with STABLE_EWOK_CSV.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    indices = [int(r["global_index"]) for r in rows]
    if max_rows is not None and max_rows > 0:
        indices = indices[:max_rows]
    meta = {
        "source": rel(STABLE_EWOK_CSV),
        "definition": "research rows that were stable conditional-reversal failures for both compared earlier legal models; fixed before this route.",
        "available_rows": len(rows),
        "used_rows": len(indices),
        "domain_counts_available": dict(Counter(r.get("domain", "") for r in rows)),
        "domain_counts_used": dict(Counter(rows[i].get("domain", "") for i in range(min(len(indices), len(rows))))),
        "first_indices": indices[:10],
    }
    return indices, meta


def flatten_for_csv(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in row.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, list):
            out[k] = json.dumps(v, ensure_ascii=False)
        elif isinstance(v, dict):
            out[k] = json.dumps(v, ensure_ascii=False, sort_keys=True)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = [flatten_for_csv(r) for r in rows]
    if not flat:
        path.write_text("", encoding="utf-8")
        return
    keys = []
    seen = set()
    for r in flat:
        for k in r:
            if k not in seen:
                seen.add(k); keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(flat)


def compact_gp(res: dict[str, Any]) -> dict[str, Any]:
    return {
        "target": res.get("target"),
        "label": res.get("label"),
        "model_root": rel(res.get("model_root")),
        "revision": res.get("revision"),
        "device": res.get("device"),
        "model_load_sec": res.get("model_load_sec"),
        "modes": {mode: payload.get("summary", {}) for mode, payload in res.get("modes", {}).items() if isinstance(payload, dict)},
    }


def gp_delta(base: dict[str, Any], cand: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for mode in ["parallel", "nonparallel"]:
        b = base.get("modes", {}).get(mode, {})
        c = cand.get("modes", {}).get(mode, {})
        md: dict[str, Any] = {}
        for key in ["accuracy", "chance_adjusted_accuracy"]:
            if isinstance(b.get(key), (int, float)) and isinstance(c.get(key), (int, float)):
                md[key] = c[key] - b[key]
        for key in ["rank1", "rank2", "rank3", "rank4"]:
            # research names rank counts as strings in correct_rank_counts.
            br = b.get("correct_rank_counts", {}).get(key[-1]) if isinstance(b.get("correct_rank_counts"), dict) else None
            cr = c.get("correct_rank_counts", {}).get(key[-1]) if isinstance(c.get("correct_rank_counts"), dict) else None
            if isinstance(br, (int, float)) and isinstance(cr, (int, float)):
                md[f"all_{key}"] = cr - br
        bh = b.get("always_wrong_subset")
        ch = c.get("always_wrong_subset")
        if isinstance(bh, dict) and isinstance(ch, dict):
            for key in ["accuracy", "mean_top_minus_correct", "median_top_minus_correct", "small_wrong_margin_le_0p25_nats", "small_wrong_margin_le_0p50_nats"]:
                if isinstance(bh.get(key), (int, float)) and isinstance(ch.get(key), (int, float)):
                    md[f"hard52_{key}"] = ch[key] - bh[key]
            brr = bh.get("correct_rank_counts", {}) if isinstance(bh.get("correct_rank_counts"), dict) else {}
            crr = ch.get("correct_rank_counts", {}) if isinstance(ch.get("correct_rank_counts"), dict) else {}
            for r in ["1", "2", "3", "4"]:
                if isinstance(brr.get(r), (int, float)) or isinstance(crr.get(r), (int, float)):
                    md[f"hard52_rank{r}"] = int(crr.get(r, 0)) - int(brr.get(r, 0))
        bm = b.get("all_rows_margin_summary")
        cm = c.get("all_rows_margin_summary")
        if isinstance(bm, dict) and isinstance(cm, dict):
            for key in ["mean_top_minus_correct", "median_top_minus_correct"]:
                if isinstance(bm.get(key), (int, float)) and isinstance(cm.get(key), (int, float)):
                    md[f"all_{key}"] = cm[key] - bm[key]
        out[mode] = md
    return out


def run_globalpiqa(targets: dict[str, dict[str, Any]], out_root: Path, gp_max_items: int | None, threads: int) -> tuple[dict[str, Any], dict[str, Any]]:
    gp = import_from(_public_path('experiments/archive/representation_and_objectives/scripts/globalpiqa_margin_reader.py'), "globalpiqa_margin_reader_step175")
    gp.OUT_ROOT = out_root / "raw"
    gp.NOTE = out_root / "globalpiqa_note.md"
    gp.TARGETS = {
        name: {"label": meta["label"], "model_root": Path(meta["model_path"]), "revision": None, "family": meta.get("family")}
        for name, meta in targets.items()
    }
    hard_ids = gp.load_always_wrong_ids()
    hard_meta = {
        "source": rel(gp.ANATOMY_JSON),
        "definition": "research fixed cross-endpoint GlobalPIQA_parallel rows with n_ok == 0.",
        "n_ids": len(hard_ids),
        "ids_sample": sorted(hard_ids)[:10],
        "reader": rel(_public_path('experiments/archive/representation_and_objectives/scripts/globalpiqa_margin_reader.py')),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    raw_dir = out_root / "rows"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in targets:
        print(json.dumps({"event": "globalpiqa_start", "target": name, "utc": now_utc()}), flush=True)
        res = gp.run_target(name, ["parallel", "nonparallel"], batch_size=8, non_causal_batch_size=32, max_items=gp_max_items, threads=threads)
        compact = compact_gp(res)
        results[name] = compact
        (out_root / f"{name}_globalpiqa_margins.json").write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        for mode, payload in res.get("modes", {}).items():
            write_csv(raw_dir / f"{name}_{mode}_rows.csv", payload.get("rows", []))
        par = compact.get("modes", {}).get("parallel", {})
        hard = par.get("always_wrong_subset") if isinstance(par, dict) else None
        print(json.dumps({"event": "globalpiqa_done", "target": name, "parallel_accuracy": par.get("accuracy") if isinstance(par, dict) else None, "hard52_accuracy": hard.get("accuracy") if isinstance(hard, dict) else None, "hard52_mean_top_minus_correct": hard.get("mean_top_minus_correct") if isinstance(hard, dict) else None}, ensure_ascii=False), flush=True)
        del res
        gc.collect()
    return results, hard_meta


def run_ewok_subset(targets: dict[str, dict[str, Any]], out_root: Path, max_rows: int | None, threads: int, row_batch_size: int, masked_batch_size: int) -> tuple[dict[str, Any], dict[str, Any]]:
    ew = import_from(_public_path('experiments/archive/representation_and_objectives/scripts/fw_ewok_interaction_reader.py'), "fw_ewok_interaction_reader_step175")
    if threads > 0:
        import torch
        torch.set_num_threads(threads)
    else:
        import torch
    device = torch.device("cpu")
    all_rows = ew.load_rows()
    indices, subset_meta = load_stable_indices(max_rows=max_rows)
    out_root.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for name, meta in targets.items():
        model_path = Path(meta["model_path"])
        print(json.dumps({"event": "ewok_stable_subset_start", "target": name, "n_rows": len(indices), "utc": now_utc()}), flush=True)
        t0 = time.time()
        tokenizer = ew.AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        model = ew.AutoModelForMaskedLM.from_pretrained(str(model_path), trust_remote_code=True)
        model.eval().to(device)
        scorer = ew.BatchedPseudoScorer(model, tokenizer, device, masked_batch_size)
        out_records: list[dict[str, Any]] = []
        for b0 in range(0, len(indices), row_batch_size):
            idx_batch = indices[b0:b0 + row_batch_size]
            recs, tasks = ew.build_batch_records(all_rows, idx_batch)
            scores = scorer.score(tasks)
            ew.attach_scores(recs, scores)
            for rec in recs:
                rec["target"] = name
                rec["subset_source"] = "both_models_stable_failure_rows"
            out_records.extend(recs)
            if b0 == 0 or ((b0 // row_batch_size) % 10 == 0):
                print(json.dumps({"event": "ewok_stable_rows_done", "target": name, "rows_done": len(out_records), "rows_total": len(indices), "utc": now_utc()}), flush=True)
        summary = ew.summarize(out_records)
        by_domain = ew.summarize_by(out_records, "domain")
        by_context_diff = ew.summarize_by(out_records, "ContextDiff")
        tdir = out_root / name
        tdir.mkdir(parents=True, exist_ok=True)
        ew.write_csv(tdir / "ewok_stable_subset_records.csv", out_records)
        ew.write_csv(tdir / "ewok_stable_subset_by_domain.csv", by_domain)
        ew.write_csv(tdir / "ewok_stable_subset_by_context_diff.csv", by_context_diff)
        result = {
            "target": name,
            "label": meta["label"],
            "model_path": rel(model_path),
            "device": str(device),
            "subset": subset_meta,
            "threads": threads,
            "row_batch_size": row_batch_size,
            "masked_batch_size": masked_batch_size,
            "elapsed_sec": round(time.time() - t0, 2),
            "summary": summary,
            "by_domain": by_domain,
            "by_context_diff": by_context_diff,
        }
        (tdir / "ewok_stable_subset_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        results[name] = {k: v for k, v in result.items() if k not in {"by_domain", "by_context_diff"}}
        print(json.dumps({"event": "ewok_stable_subset_done", "target": name, "accuracy": summary.get("accuracy"), "stable_failure": summary.get("stable_failure"), "stable_failure_frac_all": summary.get("stable_failure_frac_all"), "interaction_mean_wrong": summary.get("interaction_sum_wrong", {}).get("mean")}, ensure_ascii=False), flush=True)
        del model, tokenizer, scorer, out_records
        gc.collect()
    return results, subset_meta


def ewok_delta(base: dict[str, Any], cand: dict[str, Any]) -> dict[str, Any]:
    b = base.get("summary", {})
    c = cand.get("summary", {})
    out: dict[str, Any] = {}
    for key in ["accuracy", "saved_wrong", "stable_failure", "stable_failure_frac_all", "stable_failure_frac_wrong", "within_both_positive_wrong_frac", "local_both_actual_over_swapped_positive_wrong_frac"]:
        if isinstance(b.get(key), (int, float)) and isinstance(c.get(key), (int, float)):
            out[key] = c[key] - b[key]
    for block in ["interaction_sum_all", "interaction_sum_wrong", "interaction_mean_wrong", "deletion_interaction_sum_wrong", "interaction_minus_deletion_sum_wrong"]:
        bb = b.get(block); cc = c.get(block)
        if isinstance(bb, dict) and isinstance(cc, dict):
            for stat in ["mean", "median", "p05", "p95"]:
                if isinstance(bb.get(stat), (int, float)) and isinstance(cc.get(stat), (int, float)):
                    out[f"{block}_{stat}"] = cc[stat] - bb[stat]
    return out


def build_deltas(globalpiqa: dict[str, Any], ewok: dict[str, Any]) -> dict[str, Any]:
    pairs = [
        ("sep_aligned_minus_mlm_only", "mlm_only_20M", "sep_sparse20_aligned_20M"),
        ("sep_aligned_minus_sep_shuffled", "sep_sparse20_shuffled_20M", "sep_sparse20_aligned_20M"),
        ("sep_aligned_minus_coupled_aligned", "coupled_sparse20_aligned_20M", "sep_sparse20_aligned_20M"),
        ("sep_shuffled_minus_mlm_only", "mlm_only_20M", "sep_sparse20_shuffled_20M"),
        ("coupled_aligned_minus_mlm_only", "mlm_only_20M", "coupled_sparse20_aligned_20M"),
    ]
    out: dict[str, Any] = {"globalpiqa": {}, "ewok_stable_subset": {}}
    for label, base, cand in pairs:
        if base in globalpiqa and cand in globalpiqa:
            out["globalpiqa"][label] = gp_delta(globalpiqa[base], globalpiqa[cand])
        if base in ewok and cand in ewok:
            out["ewok_stable_subset"][label] = ewok_delta(ewok[base], ewok[cand])
    return out


def make_note(summary: dict[str, Any], out_md: Path) -> None:
    lines: list[str] = []
    lines.append("# research detached-private hard-surface readout")
    lines.append("")
    lines.append(f"Status: **{summary['status']}**")
    lines.append("")
    lines.append("This is a posthoc mechanism readout on existing A02 checkpoints. It does not train a model and does not change official submission scoring.")
    lines.append("")
    lines.append("## Targets")
    for name, meta in summary["targets"].items():
        lines.append(f"- `{name}` ({meta.get('family')}): ready={meta.get('ready')} model=`{meta.get('model_path')}`")
    lines.append("")
    if summary.get("globalpiqa"):
        lines.append("## GlobalPIQA fixed hard subset")
        for name, res in summary["globalpiqa"].items():
            par = res.get("modes", {}).get("parallel", {})
            hard = par.get("always_wrong_subset", {}) if isinstance(par, dict) else {}
            lines.append(f"- `{name}`: parallel={par.get('accuracy')} hard52_acc={hard.get('accuracy')} hard52_mean_top_minus_correct={hard.get('mean_top_minus_correct')} hard52_rank_counts={hard.get('correct_rank_counts')}")
        for label, d in summary.get("deltas", {}).get("globalpiqa", {}).items():
            hp = d.get("parallel", {})
            lines.append(f"  - delta `{label}` parallel: acc={hp.get('accuracy')} hard52_acc={hp.get('hard52_accuracy')} hard52_mean_margin={hp.get('hard52_mean_top_minus_correct')} hard52_rank1={hp.get('hard52_rank1')}")
        lines.append("")
    if summary.get("ewok_stable_subset"):
        lines.append("## EWoK research stable-failure subset")
        for name, res in summary["ewok_stable_subset"].items():
            s = res.get("summary", {})
            lines.append(f"- `{name}`: n={s.get('n')} acc={s.get('accuracy')} saved_wrong={s.get('saved_wrong')} stable_failure={s.get('stable_failure')} stable_frac_all={s.get('stable_failure_frac_all')} within_both_positive_wrong_frac={s.get('within_both_positive_wrong_frac')} interaction_wrong_mean={s.get('interaction_sum_wrong', {}).get('mean') if isinstance(s.get('interaction_sum_wrong'), dict) else None}")
        for label, d in summary.get("deltas", {}).get("ewok_stable_subset", {}).items():
            lines.append(f"  - delta `{label}`: acc={d.get('accuracy')} stable_failure={d.get('stable_failure')} stable_frac_all={d.get('stable_failure_frac_all')} interaction_wrong_mean={d.get('interaction_sum_wrong_mean')}")
        lines.append("")
    lines.append("## Interpretation boundary")
    for item in summary.get("interpretation", []):
        lines.append(f"- {item}")
    lines.append("")
    lines.append(f"JSON: `{rel(summary['summary_json'])}`")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def interpret(summary: dict[str, Any]) -> list[str]:
    items: list[str] = []
    dgp = summary.get("deltas", {}).get("globalpiqa", {})
    dew = summary.get("deltas", {}).get("ewok_stable_subset", {})
    # Conservative automatic reading: separate true-alignment effect from broad/pathway effect.
    sa_mlm_gp = dgp.get("sep_aligned_minus_mlm_only", {}).get("parallel", {})
    sa_shuf_gp = dgp.get("sep_aligned_minus_sep_shuffled", {}).get("parallel", {})
    sa_mlm_ew = dew.get("sep_aligned_minus_mlm_only", {})
    sa_shuf_ew = dew.get("sep_aligned_minus_sep_shuffled", {})
    if sa_mlm_gp:
        items.append(f"Separated aligned versus MLM-only GlobalPIQA_parallel delta is {sa_mlm_gp.get('accuracy')} points; hard52 accuracy delta is {sa_mlm_gp.get('hard52_accuracy')} points and hard52 mean top-minus-correct delta is {sa_mlm_gp.get('hard52_mean_top_minus_correct')} nats.")
    if sa_shuf_gp:
        items.append(f"True alignment versus shuffled control on GlobalPIQA_parallel has delta {sa_shuf_gp.get('accuracy')} points; hard52 accuracy delta {sa_shuf_gp.get('hard52_accuracy')} and hard52 mean-margin delta {sa_shuf_gp.get('hard52_mean_top_minus_correct')} nats.")
    if sa_mlm_ew:
        items.append(f"Separated aligned versus MLM-only on the fixed research EWoK stable-failure subset has accuracy delta {sa_mlm_ew.get('accuracy')} and stable-failure count delta {sa_mlm_ew.get('stable_failure')}.")
    if sa_shuf_ew:
        items.append(f"True alignment versus shuffled control on the EWoK stable-failure subset has accuracy delta {sa_shuf_ew.get('accuracy')} and stable-failure count delta {sa_shuf_ew.get('stable_failure')}.")
    items.append("A positive aligned-vs-MLM delta alone is not enough: the route only supports a correspondence-learning mechanism if aligned beats shuffled and coupled controls on fixed natural hard surfaces, not merely on aggregate cheap7 or broad early-training columns.")
    items.append("This readout intentionally uses fixed research/research hard surfaces and existing checkpoints; it must not be used to tune official examples or define a submission-time scoring rule.")
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description="Detached-private dual-view hard-surface readout")
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--targets", nargs="+", default=list(TARGETS), choices=sorted(TARGETS))
    ap.add_argument("--ready-only", action="store_true")
    ap.add_argument("--skip-globalpiqa", action="store_true")
    ap.add_argument("--skip-ewok", action="store_true")
    ap.add_argument("--gp-max-items", type=int, default=0, help="smoke-test only; 0 means full")
    ap.add_argument("--ewok-max-rows", type=int, default=0, help="smoke-test only; 0 means all fixed stable rows")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--ewok-row-batch-size", type=int, default=64)
    ap.add_argument("--ewok-masked-batch-size", type=int, default=160)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    cache_root = out_root / "hf_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str((out_root / "home").resolve())
    os.environ["XDG_CACHE_HOME"] = str((out_root / "xdg_cache").resolve())
    os.environ["HF_HOME"] = str(cache_root.resolve())
    os.environ["TRANSFORMERS_CACHE"] = str(cache_root.resolve())
    os.environ["HF_MODULES_CACHE"] = str((cache_root / "modules").resolve())
    os.environ["TORCH_HOME"] = str((out_root / "torch_cache").resolve())
    os.environ["TMPDIR"] = str((out_root / "tmp").resolve())
    for d in [Path(os.environ["HOME"]), Path(os.environ["XDG_CACHE_HOME"]), cache_root, cache_root / "modules", Path(os.environ["TORCH_HOME"]), Path(os.environ["TMPDIR"] )]:
        d.mkdir(parents=True, exist_ok=True)

    selected = {name: TARGETS[name] for name in args.targets}
    target_status: dict[str, Any] = {}
    missing = []
    for name, meta in selected.items():
        mp = Path(meta["model_path"])
        ready = mp.exists() and (mp / "model.safetensors").exists() and (mp / "config.json").exists()
        if not ready:
            missing.append(name)
        target_status[name] = {
            "label": meta["label"],
            "family": meta.get("family"),
            "model_path": rel(mp),
            "ready": ready,
            "model_safetensors": (mp / "model.safetensors").exists(),
            "config_json": (mp / "config.json").exists(),
            "training_metrics": read_json(Path(meta["metrics_path"])),
            "eval_scores": load_eval_scores(Path(meta["eval_payload_path"])),
            "summary_path": rel(meta.get("summary_path")),
        }
    preflight = {
        "status": "READY" if not missing else "NOT_READY",
        "created_utc": now_utc(),
        "boundary": "posthoc hard-surface readout on existing A02 checkpoints; no training; CPU only",
        "targets": target_status,
        "missing_targets": missing,
        "requested_targets": args.targets,
        "stable_ewok_rows": rel(STABLE_EWOK_CSV),
        "out_root": rel(out_root),
    }
    (out_root / "preflight.json").write_text(json.dumps(preflight, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(preflight, ensure_ascii=False), flush=True)
    if args.ready_only:
        return
    if missing:
        raise FileNotFoundError(json.dumps(preflight, ensure_ascii=False))

    summary: dict[str, Any] = {
        "status": "DETACHED_PRIVATE_HARD_SURFACE_READOUT_DONE",
        "created_utc": now_utc(),
        "boundary": preflight["boundary"],
        "out_root": rel(out_root),
        "summary_json": str(out_root / "detached_private_hard_surface_summary.json"),
        "targets": target_status,
        "globalpiqa_fixed_hard_set": None,
        "ewok_stable_subset_definition": None,
        "globalpiqa": {},
        "ewok_stable_subset": {},
        "deltas": {},
        "interpretation": [],
    }

    gp_max = args.gp_max_items if args.gp_max_items and args.gp_max_items > 0 else None
    ewok_max = args.ewok_max_rows if args.ewok_max_rows and args.ewok_max_rows > 0 else None
    if not args.skip_globalpiqa:
        gp_results, gp_meta = run_globalpiqa(selected, out_root / "globalpiqa", gp_max, args.threads)
        summary["globalpiqa"] = gp_results
        summary["globalpiqa_fixed_hard_set"] = gp_meta
    if not args.skip_ewok:
        ew_results, ew_meta = run_ewok_subset(selected, out_root / "ewok_stable_subset", ewok_max, args.threads, args.ewok_row_batch_size, args.ewok_masked_batch_size)
        summary["ewok_stable_subset"] = ew_results
        summary["ewok_stable_subset_definition"] = ew_meta
    summary["deltas"] = build_deltas(summary["globalpiqa"], summary["ewok_stable_subset"])
    summary["interpretation"] = interpret(summary)

    out_json = out_root / "detached_private_hard_surface_summary.json"
    summary["summary_json"] = str(out_json)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    make_note(summary, out_root / "detached_private_hard_surface_summary.md")
    print(json.dumps({"status": summary["status"], "summary_json": rel(out_json), "summary_md": rel(out_root / "detached_private_hard_surface_summary.md")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
