#!/usr/bin/env python3
"""research: score mirrored four-row binding rows for any checkpoint, with quad diagnostics.

The research scorer was sufficient for the initial chck82 baseline, but post-pilot
checkpoints may not be safely reloadable through AutoModel.  This script first tries
trusted AutoModel loading, then falls back to the research FrozenSlowPrivate class and
safetensors.  It scores all present literal values at the answer slot by masked-token
pseudo-log-likelihood and reports row, shared-context-pair, full-quad, k, assignment-
position, relation and frame slices for both raw logprob and per-token normalized
logprob.
"""

from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import math
import os
import pathlib
import statistics
import sys
import time
from collections import Counter, defaultdict
from typing import Any

ROOT = _public_path('.')
CACHE_BASE = _public_path('experiments/archive/relation_learning/data/score_mirrored_explicit/hf_cache')
for sub in ["hf_home", "hf_home/hub", "transformers", "modules", "datasets", "tmp"]:
    (CACHE_BASE / sub).mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(_public_path('experiments/archive/relation_learning/data/score_mirrored_explicit/hf_cache/hf_home'))
os.environ["HF_HUB_CACHE"] = str(_public_path('experiments/archive/relation_learning/data/score_mirrored_explicit/hf_cache/hf_home/hub'))
os.environ["HUGGINGFACE_HUB_CACHE"] = str(_public_path('experiments/archive/relation_learning/data/score_mirrored_explicit/hf_cache/hf_home/hub'))
os.environ["TRANSFORMERS_CACHE"] = str(_public_path('experiments/archive/relation_learning/data/score_mirrored_explicit/hf_cache/transformers'))
os.environ["HF_MODULES_CACHE"] = str(_public_path('experiments/archive/relation_learning/data/score_mirrored_explicit/hf_cache/modules'))
os.environ["HF_DATASETS_CACHE"] = str(_public_path('experiments/archive/relation_learning/data/score_mirrored_explicit/hf_cache/datasets'))
os.environ["TMPDIR"] = str(_public_path('experiments/archive/relation_learning/data/score_mirrored_explicit/hf_cache/tmp'))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F

CHCK82 = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M')
ROWS_DEFAULT = _public_path('experiments/archive/relation_learning/data/mirrored_four_row_family_position_var/mirrored_heldout_frame_all.jsonl')
OUT_DEFAULT = _public_path('experiments/archive/relation_learning/data/score_mirrored_explicit')


def rel(p: str | pathlib.Path) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def norm_path(p: str | pathlib.Path) -> pathlib.Path:
    q = pathlib.Path(p)
    return q if q.is_absolute() else ROOT / q


def load_model(model_dir: pathlib.Path, device: torch.device):
    from transformers import AutoModelForMaskedLM, AutoTokenizer, DebertaV2Config
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True, trust_remote_code=True, local_files_only=True)
    try:
        model = AutoModelForMaskedLM.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True, torch_dtype=torch.float32)
        loader = "AutoModelForMaskedLM_trust_remote_code"
    except Exception as e:
        sys.path.insert(0, str(_public_path('experiments/archive/frontier_consolidation/scripts')))
        from frozen82_private_modeling import FrozenSlowPrivateDebertaV2ForMaskedLM  # noqa: E402
        from safetensors.torch import load_file  # noqa: E402
        config = DebertaV2Config.from_pretrained(str(model_dir), local_files_only=True)
        model = FrozenSlowPrivateDebertaV2ForMaskedLM(config)
        sf = model_dir / "model.safetensors"
        state = load_file(str(sf), device="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        loader = f"FrozenSlowPrivate_fallback missing={len(missing)} unexpected={len(unexpected)} auto_error={type(e).__name__}: {e}"
    model.to(device)
    model.eval()
    identity = {
        "model_dir": rel(model_dir),
        "loader": loader,
        "loaded_class": type(model).__module__ + "." + type(model).__qualname__,
        "total_params": int(sum(p.numel() for p in model.parameters())),
        "private_params": int(sum(p.numel() for n, p in model.named_parameters() if "private_adapter" in n)),
        "adapter_like_params": int(sum(p.numel() for n, p in model.named_parameters() if "adapter" in n.lower())),
        "private_adapter_scale": getattr(model.config, "private_adapter_scale", None),
        "adapter_scale": getattr(model.config, "adapter_scale", None),
        "private_adapter_enabled": getattr(model.config, "private_adapter_enabled", None),
    }
    print(json.dumps({"event": "model_loaded", **identity}), flush=True)
    return model, tokenizer, identity


def candidate_scores(model, tokenizer, prefix: str, candidates: list[str], device: torch.device, max_length: int) -> dict[str, dict[str, float]]:
    mask_id = int(tokenizer.mask_token_id)
    cls_id = int(tokenizer.cls_token_id if tokenizer.cls_token_id is not None else 0)
    sep_id = int(tokenizer.sep_token_id if tokenizer.sep_token_id is not None else 0)
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    out: dict[str, dict[str, float]] = {}
    for cand in candidates:
        cand_ids = tokenizer.encode(cand, add_special_tokens=False)
        if not cand_ids:
            continue
        avail = max_length - len(cand_ids) - 2
        if avail < 1:
            continue
        trunc = prefix_ids[-avail:] if len(prefix_ids) > avail else prefix_ids
        base = [cls_id] + trunc + cand_ids + [sep_id]
        cand_start = len(base) - len(cand_ids) - 1
        batch = []
        for j in range(len(cand_ids)):
            m = list(base)
            m[cand_start + j] = mask_id
            batch.append(m)
        ids = torch.tensor(batch, dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(input_ids=ids).logits.float()
        total = 0.0
        for j, tid in enumerate(cand_ids):
            total += float(F.log_softmax(logits[j, cand_start + j], dim=-1)[int(tid)].detach().cpu())
        out[cand] = {"logprob": total, "n_tokens": len(cand_ids), "logprob_per_token": total / max(1, len(cand_ids))}
    return out


def score_one(model, tokenizer, row: dict[str, Any], device: torch.device, max_length: int) -> dict[str, Any] | None:
    candidates = list(dict.fromkeys([str(v) for v in row.get("all_present_values", []) if str(v)]))
    if str(row["answer"]) not in candidates:
        candidates.append(str(row["answer"]))
    prefix = row["context_text"] + " " + row["query_prefix"]
    scores = candidate_scores(model, tokenizer, prefix, candidates, device, max_length)
    if not scores or str(row["answer"]) not in scores:
        return None
    correct = str(row["answer"])
    top_raw = max(scores, key=lambda c: scores[c]["logprob"])
    top_norm = max(scores, key=lambda c: scores[c]["logprob_per_token"])
    comps = [c for c in scores if c != correct]
    comp_raw = max(comps, key=lambda c: scores[c]["logprob"]) if comps else None
    comp_norm = max(comps, key=lambda c: scores[c]["logprob_per_token"]) if comps else None
    return {
        "row_id": row["row_id"], "quad_id": row["quad_id"], "map_id": row["map_id"],
        "relation": row.get("relation"), "k": int(row.get("k", -1)),
        "assignment_position": int(row.get("assignment_position", -1)),
        "frame_id": row.get("frame_id"), "frame_split": row.get("frame_split"),
        "context_variant": row.get("context_variant"), "answer_role": row.get("answer_role"),
        "entity_queried": row.get("entity_queried"), "answer": correct,
        "n_candidates": len(scores),
        "answer_n_tokens": scores[correct]["n_tokens"],
        "top_raw": top_raw, "top_norm": top_norm,
        "correct_raw": int(top_raw == correct), "correct_norm": int(top_norm == correct),
        "correct_lp_raw": scores[correct]["logprob"], "correct_lp_norm": scores[correct]["logprob_per_token"],
        "top_comp_raw": comp_raw,
        "margin_raw": (scores[correct]["logprob"] - scores[comp_raw]["logprob"]) if comp_raw else float("inf"),
        "top_comp_norm": comp_norm,
        "margin_norm": (scores[correct]["logprob_per_token"] - scores[comp_norm]["logprob_per_token"]) if comp_norm else float("inf"),
        "top_raw_origin": origin_label(row, top_raw),
        "top_norm_origin": origin_label(row, top_norm),
    }


def origin_label(row: dict[str, Any], value: str | None) -> str:
    if value is None:
        return "none"
    if value == row.get("shared_new_value"):
        return "shared_new"
    if value == row.get("source_value_queried"):
        return "queried_source"
    if value == row.get("source_value_partner"):
        return "partner_source"
    if value in set(row.get("stranger_values", [])):
        return "stranger_update"
    return "other"


def safe_mean(xs):
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return statistics.mean(vals) if vals else float("nan")


def aggregate(results: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    ck = "correct_" + mode
    mk = "margin_" + mode
    top_origin = "top_" + mode + "_origin"
    n = len(results)
    by_role = defaultdict(list)
    by_k = defaultdict(list)
    by_pos = defaultdict(list)
    by_rel = defaultdict(list)
    by_frame_split = defaultdict(list)
    by_k_pos = defaultdict(list)
    for r in results:
        by_role[r["answer_role"]].append(r)
        by_k[r["k"]].append(r)
        by_pos[(r["k"], r["assignment_position"])].append(r)
        by_rel[r["relation"]].append(r)
        by_frame_split[r["frame_split"]].append(r)
        by_k_pos[(r["k"], r["assignment_position"])].append(r)

    pair_g = defaultdict(dict)
    for r in results:
        pair_g[(r["quad_id"], r["context_variant"])][r["answer_role"]] = bool(r[ck])
    pairs = [v for v in pair_g.values() if "updated" in v and "retained" in v]
    pair_joint = sum(1 for p in pairs if p["updated"] and p["retained"])
    pair_both_wrong = sum(1 for p in pairs if (not p["updated"]) and (not p["retained"]))

    quad_g = defaultdict(list)
    for r in results:
        quad_g[r["quad_id"]].append(bool(r[ck]))
    quads = [v for v in quad_g.values() if len(v) == 4]
    quad_all = sum(1 for v in quads if all(v))
    quad_none = sum(1 for v in quads if not any(v))

    def slice_summary(rs):
        return {
            "n": len(rs),
            "acc": sum(int(x[ck]) for x in rs) / len(rs) if rs else float("nan"),
            "updated_acc": sum(int(x[ck]) for x in rs if x["answer_role"] == "updated") / max(1, sum(1 for x in rs if x["answer_role"] == "updated")),
            "retained_acc": sum(int(x[ck]) for x in rs if x["answer_role"] == "retained") / max(1, sum(1 for x in rs if x["answer_role"] == "retained")),
            "margin_mean": safe_mean(x[mk] for x in rs),
            "top_origins": dict(Counter(x[top_origin] for x in rs)),
        }

    return {
        "mode": mode,
        "rows": n,
        "accuracy": sum(int(r[ck]) for r in results) / n if n else float("nan"),
        "margin_mean": safe_mean(r[mk] for r in results),
        "by_role": {str(k): slice_summary(v) for k, v in sorted(by_role.items(), key=lambda kv: str(kv[0]))},
        "by_k": {str(k): slice_summary(v) for k, v in sorted(by_k.items())},
        "by_assignment_position": {f"k{k}_pos{p}": slice_summary(v) for (k, p), v in sorted(by_pos.items())},
        "by_relation": {str(k): slice_summary(v) for k, v in sorted(by_rel.items(), key=lambda kv: str(kv[0]))},
        "by_frame_split": {str(k): slice_summary(v) for k, v in sorted(by_frame_split.items(), key=lambda kv: str(kv[0]))},
        "shared_context_pairs": {
            "total": len(pairs),
            "both_correct": pair_joint,
            "both_wrong": pair_both_wrong,
            "joint_rate": pair_joint / len(pairs) if pairs else float("nan"),
            "both_wrong_rate": pair_both_wrong / len(pairs) if pairs else float("nan"),
        },
        "full_quads": {
            "total": len(quads),
            "all_correct": quad_all,
            "none_correct": quad_none,
            "rate": quad_all / len(quads) if quads else float("nan"),
            "none_rate": quad_none / len(quads) if quads else float("nan"),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(CHCK82))
    ap.add_argument("--rows-file", default=str(ROWS_DEFAULT))
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--frame-filter", default="")
    ap.add_argument("--max-rows", type=int, default=0)
    args = ap.parse_args()
    out = norm_path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model, tokenizer, identity = load_model(norm_path(args.model_dir), device)

    rows = []
    with norm_path(args.rows_file).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if args.frame_filter and args.frame_filter not in str(r.get("frame_id", "")):
                continue
            rows.append(r)
    if int(args.max_rows) > 0:
        rows = rows[:int(args.max_rows)]
    print(json.dumps({"event": "score_start", "rows": len(rows), "rows_file": rel(args.rows_file), "frame_filter": args.frame_filter}), flush=True)
    t0 = time.time()
    results = []
    for i, row in enumerate(rows, 1):
        res = score_one(model, tokenizer, row, device, int(args.max_length))
        if res is not None:
            results.append(res)
        if i % 100 == 0 or i == len(rows):
            print(json.dumps({"event": "progress", "i": i, "n": len(rows), "elapsed_sec": round(time.time() - t0, 1)}), flush=True)
    elapsed = time.time() - t0
    with (out / "raw_scores.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {
        "status": "MIRRORED_SCORE_DONE",
        "model_identity": identity,
        "rows_file": rel(args.rows_file),
        "frame_filter": args.frame_filter,
        "rows_requested": len(rows),
        "rows_scored": len(results),
        "elapsed_sec": round(elapsed, 2),
        "raw": aggregate(results, "raw"),
        "norm": aggregate(results, "norm"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "rows_scored": len(results),
        "elapsed_sec": summary["elapsed_sec"],
        "raw_acc": summary["raw"]["accuracy"],
        "raw_pair_joint": summary["raw"]["shared_context_pairs"],
        "raw_quads": summary["raw"]["full_quads"],
        "norm_acc": summary["norm"]["accuracy"],
        "norm_pair_joint": summary["norm"]["shared_context_pairs"],
        "norm_quads": summary["norm"]["full_quads"],
        "out": rel(out),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
