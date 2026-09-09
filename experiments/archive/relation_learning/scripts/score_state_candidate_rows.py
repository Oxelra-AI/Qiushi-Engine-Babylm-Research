#!/usr/bin/env python3
"""research: score local binding row families over every state candidate present.

This is a pre-training admission readout.  It masks each candidate phrase in the final
query slot and scores the candidate by pseudo-log-likelihood per token.  For the
research identical-context twins, a row is correct only when the highest-scoring value
among all state values present in the context is the query source value (A/retention)
or the shared new update value (B/updated).  Source-vs-new margins are also recorded
for continuity with research/079, but the all-candidate top choice is the important
readout because official Entity offers multiple operation-derived alternatives.
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
from collections import Counter, defaultdict
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = _public_path('.')
DEFAULT_ROWS = _public_path('experiments/archive/relation_learning/data/identical_twin_margin_rows/identical_twin_heldout_frame_all.jsonl')
DEFAULT_PAIRS = _public_path('experiments/archive/relation_learning/data/identical_twin_margin_rows/binding_pairs_heldout_frame_all.jsonl')
DEFAULT_MODEL = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M')
DEFAULT_OUT = _public_path('experiments/archive/relation_learning/data/candidate_state_readout_chck82')


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                keys.append(k); seen.add(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def norm(s: Any) -> str:
    return " ".join(str(s or "").strip().split())


def locate_token_positions(offsets: list[tuple[int, int]], a0: int, b0: int) -> list[int]:
    pos = []
    for i, (a, b) in enumerate(offsets):
        if b <= a0 or a >= b0:
            continue
        if b > a:
            pos.append(i)
    return pos


def candidate_context(row: dict[str, Any], candidate: str) -> tuple[str, int, int]:
    a = int(row["answer_char_start"]); b = int(row["answer_char_end"])
    text0 = str(row["context_text"])
    cand = norm(candidate)
    text = text0[:a] + cand + text0[b:]
    return text, a, a + len(cand)


def candidate_records(tokenizer: Any, rows: list[dict[str, Any]], max_length: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    out: list[dict[str, Any]] = []
    skips: Counter[str] = Counter()
    for row in rows:
        cands = row.get("state_candidates") or [
            {"candidate_id": "query_source", "value": row.get("source_state", ""), "origin": "source_context_query_entity"},
            {"candidate_id": "shared_new", "value": row.get("new_state", ""), "origin": "assignment_update_shared_new"},
        ]
        seen_vals = set()
        for c in cands:
            cid = str(c.get("candidate_id") or c.get("origin") or c.get("value"))
            val = norm(c.get("value", ""))
            if not val or val in seen_vals:
                continue
            seen_vals.add(val)
            text, a0, b0 = candidate_context(row, val)
            enc = tokenizer(text, add_special_tokens=True, truncation=True, max_length=max_length, return_offsets_mapping=True)
            ids = [int(x) for x in enc["input_ids"]]
            offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
            pos = locate_token_positions(offsets, a0, b0)
            if not pos:
                skips["no_candidate_tokens"] += 1
                continue
            for j, p in enumerate(pos):
                masked = list(ids)
                masked[p] = int(tokenizer.mask_token_id)
                out.append({
                    "row_id": row["row_id"], "pair_id": row.get("pair_id", ""),
                    "pair_half": row.get("pair_half", ""), "answer_kind": row.get("answer_kind", ""),
                    "role": row.get("role", ""), "query_entity": row.get("query_entity", ""),
                    "query_side": row.get("query_side", ""), "frame_id": row.get("frame_id", ""),
                    "frame_split": row.get("frame_split", ""), "relation": row.get("relation", ""),
                    "operation_count": row.get("operation_count", ""),
                    "distractor_update_count": row.get("distractor_update_count", ""),
                    "assignment_update_position_0based": row.get("assignment_update_position_0based", ""),
                    "correct_candidate_id": row.get("correct_candidate_id", ""),
                    "candidate_id": cid, "candidate_value": val, "candidate_origin": c.get("origin", ""),
                    "candidate_token_index": j, "candidate_n_tokens": len(pos),
                    "target_token_id": int(ids[p]), "input_ids": masked, "attention_mask": [1] * len(masked),
                })
    return out, dict(skips)


@torch.no_grad()
def score_records(model: Any, tokenizer: Any, records: list[dict[str, Any]], device: torch.device, batch_size: int) -> list[dict[str, Any]]:
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    token_rows: list[dict[str, Any]] = []
    model.eval()
    for start in range(0, len(records), batch_size):
        batch = records[start:start+batch_size]
        max_len = max(len(r["input_ids"]) for r in batch)
        ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long, device=device)
        att = torch.zeros((len(batch), max_len), dtype=torch.long, device=device)
        for i, r in enumerate(batch):
            L = len(r["input_ids"])
            ids[i, :L] = torch.tensor(r["input_ids"], dtype=torch.long, device=device)
            att[i, :L] = 1
        logits = model(input_ids=ids, attention_mask=att).logits.float()
        logp = F.log_softmax(logits, dim=-1)
        for i, r in enumerate(batch):
            mpos = [k for k, tid in enumerate(r["input_ids"]) if int(tid) == int(tokenizer.mask_token_id)]
            if len(mpos) != 1:
                continue
            lp = float(logp[i, mpos[0], int(r["target_token_id"])].detach().cpu())
            keep = {k: r[k] for k in ["row_id", "pair_id", "pair_half", "answer_kind", "role", "query_entity", "query_side", "frame_id", "frame_split", "relation", "operation_count", "distractor_update_count", "assignment_update_position_0based", "correct_candidate_id", "candidate_id", "candidate_value", "candidate_origin", "candidate_n_tokens"]}
            keep.update({"candidate_token_index": r["candidate_token_index"], "logprob": lp})
            token_rows.append(keep)
    return token_rows


def avg(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return float(statistics.mean(vals)) if vals else float("nan")


def aggregate(token_rows: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for tr in token_rows:
        groups[(tr["row_id"], tr["candidate_id"])].append(tr)
    cand_rows: list[dict[str, Any]] = []
    for (rid, cid), rs in groups.items():
        base = rs[0]
        s = sum(float(r["logprob"]) for r in rs)
        n = max(1, int(base["candidate_n_tokens"]))
        rec = {k: base[k] for k in ["row_id", "pair_id", "pair_half", "answer_kind", "role", "query_entity", "query_side", "frame_id", "frame_split", "relation", "operation_count", "distractor_update_count", "assignment_update_position_0based", "correct_candidate_id", "candidate_id", "candidate_value", "candidate_origin", "candidate_n_tokens"]}
        rec.update({"logprob_sum": s, "logprob_per_token": s / n})
        cand_rows.append(rec)
    by_row: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cr in cand_rows:
        by_row[cr["row_id"]].append(cr)
    row_rows: list[dict[str, Any]] = []
    for rid, rs in by_row.items():
        top = max(rs, key=lambda x: float(x["logprob_per_token"]))
        correct_id = str(top.get("correct_candidate_id", ""))
        by_cid = {r["candidate_id"]: r for r in rs}
        src = by_cid.get("query_source")
        new = by_cid.get("shared_new")
        margin_new_minus_source = None
        correct_margin = None
        src_new_correct = None
        if src is not None and new is not None:
            margin_new_minus_source = float(new["logprob_per_token"]) - float(src["logprob_per_token"])
            if correct_id == "query_source":
                correct_margin = -margin_new_minus_source
                src_new_correct = int(correct_margin > 0)
            elif correct_id == "shared_new":
                correct_margin = margin_new_minus_source
                src_new_correct = int(correct_margin > 0)
        rec = {k: top.get(k) for k in ["row_id", "pair_id", "pair_half", "answer_kind", "role", "query_entity", "query_side", "frame_id", "frame_split", "relation", "operation_count", "distractor_update_count", "assignment_update_position_0based", "correct_candidate_id"]}
        rec.update({
            "n_candidates": len(rs),
            "top_candidate_id": top["candidate_id"],
            "top_candidate_origin": top.get("candidate_origin", ""),
            "top_candidate_value": top["candidate_value"],
            "top_logprob_per_token": float(top["logprob_per_token"]),
            "all_candidate_correct": int(top["candidate_id"] == correct_id),
            "margin_new_minus_source": margin_new_minus_source,
            "source_new_correct_margin": correct_margin,
            "source_new_correct": src_new_correct,
            "top2_gap": None,
        })
        sorted_rs = sorted(rs, key=lambda x: float(x["logprob_per_token"]), reverse=True)
        if len(sorted_rs) >= 2:
            rec["top2_gap"] = float(sorted_rs[0]["logprob_per_token"]) - float(sorted_rs[1]["logprob_per_token"])
        row_rows.append(rec)
    by_id = {r["row_id"]: r for r in row_rows}
    pair_rows: list[dict[str, Any]] = []
    for p in pairs:
        a = by_id.get(p["row_a_id"]); b = by_id.get(p["row_b_id"])
        if not a or not b:
            continue
        pair_rows.append({
            "pair_id": p["pair_id"], "row_a_id": p["row_a_id"], "row_b_id": p["row_b_id"],
            "base_pair_id": p.get("base_pair_id", ""), "query_side": p.get("query_side", ""),
            "frame_id": p.get("frame_id", ""), "frame_split": p.get("frame_split", ""),
            "operation_count": p.get("operation_count", ""), "distractor_update_count": p.get("distractor_update_count", ""),
            "assignment_update_position_0based": p.get("assignment_update_position_0based", ""),
            "a_all_candidate_correct": int(a["all_candidate_correct"]),
            "b_all_candidate_correct": int(b["all_candidate_correct"]),
            "joint_all_candidate_correct": int(a["all_candidate_correct"] and b["all_candidate_correct"]),
            "a_source_new_correct": int(a["source_new_correct"] or 0),
            "b_source_new_correct": int(b["source_new_correct"] or 0),
            "joint_source_new_correct": int((a["source_new_correct"] or 0) and (b["source_new_correct"] or 0)),
            "a_top_candidate_id": a["top_candidate_id"], "b_top_candidate_id": b["top_candidate_id"],
            "a_top_origin": a["top_candidate_origin"], "b_top_origin": b["top_candidate_origin"],
            "a_margin": a["source_new_correct_margin"], "b_margin": b["source_new_correct_margin"],
            "joint_min_source_new_margin": None if a["source_new_correct_margin"] is None or b["source_new_correct_margin"] is None else min(float(a["source_new_correct_margin"]), float(b["source_new_correct_margin"])),
        })
    summary: dict[str, Any] = {}
    def summarize_rows(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n": len(rows),
            "all_candidate_correct": sum(int(r["all_candidate_correct"]) for r in rows),
            "all_candidate_accuracy": sum(int(r["all_candidate_correct"]) for r in rows) / len(rows) if rows else float("nan"),
            "source_new_correct": sum(int(r["source_new_correct"] or 0) for r in rows),
            "source_new_accuracy": sum(int(r["source_new_correct"] or 0) for r in rows) / len(rows) if rows else float("nan"),
            "mean_source_new_correct_margin": avg(r["source_new_correct_margin"] for r in rows if r["source_new_correct_margin"] is not None),
            "top_candidate_counts": dict(Counter(r["top_candidate_id"] for r in rows)),
            "top_origin_counts": dict(Counter(r["top_candidate_origin"] for r in rows)),
        }
    summary["rows_all"] = summarize_rows("all", row_rows)
    summary["rows_A_retention"] = summarize_rows("A", [r for r in row_rows if r.get("pair_half") == "A"])
    summary["rows_B_updated"] = summarize_rows("B", [r for r in row_rows if r.get("pair_half") == "B"])
    summary["pairs"] = {
        "n": len(pair_rows),
        "joint_all_candidate_correct": sum(p["joint_all_candidate_correct"] for p in pair_rows),
        "joint_all_candidate_accuracy": sum(p["joint_all_candidate_correct"] for p in pair_rows) / len(pair_rows) if pair_rows else float("nan"),
        "joint_source_new_correct": sum(p["joint_source_new_correct"] for p in pair_rows),
        "joint_source_new_accuracy": sum(p["joint_source_new_correct"] for p in pair_rows) / len(pair_rows) if pair_rows else float("nan"),
        "a_all_candidate_correct": sum(p["a_all_candidate_correct"] for p in pair_rows),
        "b_all_candidate_correct": sum(p["b_all_candidate_correct"] for p in pair_rows),
        "both_wrong_all_candidate": sum(1 for p in pair_rows if not p["a_all_candidate_correct"] and not p["b_all_candidate_correct"]),
        "mean_joint_min_source_new_margin": avg(p["joint_min_source_new_margin"] for p in pair_rows if p["joint_min_source_new_margin"] is not None),
    }
    for field in ["distractor_update_count", "operation_count", "assignment_update_position_0based", "frame_split", "frame_id", "query_side", "relation"]:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in row_rows:
            buckets[str(r.get(field, ""))].append(r)
        summary[f"rows_by_{field}"] = {k: summarize_rows(k, v) for k, v in sorted(buckets.items(), key=lambda kv: kv[0])}
        pb: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for p in pair_rows:
            pb[str(p.get(field, ""))].append(p)
        summary[f"pairs_by_{field}"] = {
            k: {
                "n": len(v),
                "joint_all_candidate_correct": sum(x["joint_all_candidate_correct"] for x in v),
                "joint_all_candidate_accuracy": sum(x["joint_all_candidate_correct"] for x in v) / len(v) if v else float("nan"),
                "a_all_candidate_accuracy": sum(x["a_all_candidate_correct"] for x in v) / len(v) if v else float("nan"),
                "b_all_candidate_accuracy": sum(x["b_all_candidate_correct"] for x in v) / len(v) if v else float("nan"),
                "both_wrong_all_candidate": sum(1 for x in v if not x["a_all_candidate_correct"] and not x["b_all_candidate_correct"]),
            } for k, v in sorted(pb.items(), key=lambda kv: kv[0])
        }
    return row_rows, pair_rows, summary


def load_model(model_path: pathlib.Path, device: torch.device) -> tuple[Any, Any, dict[str, Any]]:
    cache = pathlib.Path(os.environ.get("SCORE_CACHE", str(_public_path('experiments/archive/relation_learning/data/candidate_state_readout_chck82/hf_cache'))))
    for k, p in {
        "HF_HOME": cache / "hf_home",
        "HF_HUB_CACHE": cache / "hf_home" / "hub",
        "HUGGINGFACE_HUB_CACHE": cache / "hf_home" / "hub",
        "TRANSFORMERS_CACHE": cache / "transformers",
        "HF_MODULES_CACHE": cache / "modules",
        "HF_DATASETS_CACHE": cache / "datasets",
        "TMPDIR": cache / "tmp",
    }.items():
        p.mkdir(parents=True, exist_ok=True)
        os.environ[k] = str(p.resolve())
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    tok = AutoTokenizer.from_pretrained(str(model_path), use_fast=True, local_files_only=True, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True).to(device)
    ident = {
        "model_class": type(model).__name__,
        "model_path": rel(model_path),
        "total_params": sum(p.numel() for p in model.parameters()),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "private_adapter_scale": getattr(model.config, "private_adapter_scale", None),
        "private_adapter_enabled": getattr(model.config, "private_adapter_enabled", None),
    }
    return model, tok, ident


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=pathlib.Path, default=DEFAULT_MODEL)
    ap.add_argument("--rows", type=pathlib.Path, default=DEFAULT_ROWS)
    ap.add_argument("--pairs", type=pathlib.Path, default=DEFAULT_PAIRS)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--max-pairs", type=int, default=0)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.rows)
    pairs = read_jsonl(args.pairs)
    if args.max_pairs:
        keep = {p["pair_id"] for p in pairs[:args.max_pairs]}
        pairs = [p for p in pairs if p["pair_id"] in keep]
        keep_rows = {p["row_a_id"] for p in pairs} | {p["row_b_id"] for p in pairs}
        rows = [r for r in rows if r["row_id"] in keep_rows]
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    t0 = time.time()
    model, tok, ident = load_model(args.model, device)
    records, skips = candidate_records(tok, rows, args.max_length)
    token_rows = score_records(model, tok, records, device, args.batch_size)
    row_rows, pair_rows, summary = aggregate(token_rows, pairs)
    write_csv(args.out_dir / "candidate_token_logprobs.csv", token_rows)
    write_csv(args.out_dir / "row_candidate_readout.csv", row_rows)
    write_csv(args.out_dir / "pair_candidate_readout.csv", pair_rows)
    obj = {
        "status": "STATE_CANDIDATE_READOUT",
        "created_utc": now(),
        "model_identity": ident,
        "rows_path": rel(args.rows),
        "pairs_path": rel(args.pairs),
        "rows_input": len(rows),
        "pairs_input": len(pairs),
        "candidate_mask_records": len(records),
        "token_logprob_rows": len(token_rows),
        "skips": skips,
        "summary": summary,
        "files": {
            "candidate_token_logprobs": rel(args.out_dir / "candidate_token_logprobs.csv"),
            "row_candidate_readout": rel(args.out_dir / "row_candidate_readout.csv"),
            "pair_candidate_readout": rel(args.out_dir / "pair_candidate_readout.csv"),
        },
        "elapsed_sec": round(time.time() - t0, 2),
        "scientific_reading": "Admission readout before training: a useful binding family should not already have A/retention at ceiling against present alternatives; source retention should not collapse with distractor count; all-candidate top choice is stricter than source-vs-new margin.",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research state-candidate readout",
        "",
        f"Model: `{ident['model_path']}` ({ident['model_class']}).",
        f"Rows {len(rows)}, pairs {len(pairs)}, candidate mask records {len(records)}, elapsed {obj['elapsed_sec']} sec.",
        "",
        "## Aggregate",
        f"A/retention all-candidate accuracy: {summary['rows_A_retention']['all_candidate_accuracy']:.3f} ({summary['rows_A_retention']['all_candidate_correct']}/{summary['rows_A_retention']['n']})",
        f"B/updated all-candidate accuracy: {summary['rows_B_updated']['all_candidate_accuracy']:.3f} ({summary['rows_B_updated']['all_candidate_correct']}/{summary['rows_B_updated']['n']})",
        f"Pair joint all-candidate accuracy: {summary['pairs']['joint_all_candidate_accuracy']:.3f} ({summary['pairs']['joint_all_candidate_correct']}/{summary['pairs']['n']})",
        f"Both-wrong all-candidate pairs: {summary['pairs']['both_wrong_all_candidate']}/{summary['pairs']['n']}",
        "",
        "## Pair accuracy by distractor count",
    ]
    for k, rec in summary["pairs_by_distractor_update_count"].items():
        lines.append(f"- d={k}: joint={rec['joint_all_candidate_accuracy']:.3f}, A={rec['a_all_candidate_accuracy']:.3f}, B={rec['b_all_candidate_accuracy']:.3f}, both_wrong={rec['both_wrong_all_candidate']}/{rec['n']}")
    lines += ["", "## Top candidates", f"A top ids: {summary['rows_A_retention']['top_candidate_counts']}", f"B top ids: {summary['rows_B_updated']['top_candidate_counts']}"]
    (args.out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": obj["status"], "out_dir": rel(args.out_dir), "elapsed_sec": obj["elapsed_sec"], "summary": {"A": summary["rows_A_retention"], "B": summary["rows_B_updated"], "pairs": summary["pairs"]}}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
