#!/usr/bin/env python3
"""research: trusted baseline scorer for plain-use answer-slot rows.

Scores the answer vs foil state phrase in the final {STATE} slot for the converted
research natural packets.  This establishes the low-noise readout before any private-
adapter answer-only phase is trained.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import collections
import csv
import json
import math
import os
import pathlib
import statistics
import time
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT0 = _public_path('experiments/archive/relation_learning/scripts/baseline_plainuse_answer_slot_scorer.py')
ROOT = _PUBLIC_ROOT

ROWS_DEFAULT = ROOT / "experiments/archive/relation_learning/data/plainuse_answer_slot_rows/answer_slot_rows_heldout.jsonl"
MODEL_DEFAULT = ROOT / "experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_100M"
OUT_DEFAULT = ROOT / "experiments/archive/relation_learning/data/plainuse_answer_slot_baseline_base43022"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8"); return
    fields=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen:
                fields.append(k); seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def configure_cache(path: pathlib.Path | None) -> dict[str, str]:
    if path is None:
        return {}
    mapping = {
        "HF_HOME": path / "hf_home",
        "HF_HUB_CACHE": path / "hf_home" / "hub",
        "HUGGINGFACE_HUB_CACHE": path / "hf_home" / "hub",
        "TRANSFORMERS_CACHE": path / "transformers",
        "HF_MODULES_CACHE": path / "modules",
        "HF_DATASETS_CACHE": path / "datasets",
        "TMPDIR": path / "tmp",
    }
    out={}
    for k,v in mapping.items():
        v.mkdir(parents=True, exist_ok=True); os.environ[k]=str(v.resolve()); out[k]=str(v.resolve())
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return out


def model_identity(mp: pathlib.Path, model: Any, trust_remote_code: bool, local_files_only: bool) -> dict[str, Any]:
    try:
        cfg=json.loads((mp / "config.json").read_text(encoding="utf-8"))
    except Exception:
        cfg={}
    named=list(model.named_parameters())
    return {
        "checkpoint_path": rel(mp),
        "loaded_class": model.__class__.__module__ + "." + model.__class__.__name__,
        "total_params_loaded": int(sum(p.numel() for _,p in named)),
        "adapter_params_loaded": int(sum(p.numel() for n,p in named if "adapter" in n.lower())),
        "architectures": cfg.get("architectures"),
        "model_type": cfg.get("model_type"),
        "auto_map_present": bool(cfg.get("auto_map")),
        "adapter_enabled_config": cfg.get("adapter_enabled"),
        "adapter_scale_config": cfg.get("adapter_scale"),
        "adapter_bottleneck_config": cfg.get("adapter_bottleneck"),
        "dynamic_modeling_files": sorted(p.name for p in mp.glob("*modeling*.py")),
        "trust_remote_code": bool(trust_remote_code),
        "local_files_only": bool(local_files_only),
    }


def locate_span_positions(offsets: Sequence[tuple[int,int]], start: int, end: int) -> list[int]:
    return [i for i,(s,e) in enumerate(offsets) if int(e)>int(s) and int(s)<end and int(e)>start]


def full_text_and_span(row: dict[str, Any], candidate: str) -> tuple[str,int,int]:
    frame = str(row["use_sentence_frame"])
    if frame.count("{STATE}") != 1:
        raise ValueError(f"bad frame {row.get('row_id')}: {frame!r}")
    before, after = frame.split("{STATE}")
    prefix = str(row["source_sentence"]).rstrip() + " " + str(row["update_sentence"]).strip() + " "
    full = prefix + before + candidate + after
    start = len(prefix) + len(before)
    end = start + len(candidate)
    if full[start:end] != candidate:
        raise RuntimeError("span construction error")
    return full,start,end


@torch.no_grad()
def score_candidate(model, tok, row: dict[str, Any], candidate: str, device: torch.device, seq_length: int) -> dict[str, Any]:
    full,start,end = full_text_and_span(row, candidate)
    enc = tok(full, add_special_tokens=True, return_offsets_mapping=True, return_tensors="pt", max_length=seq_length, truncation=True)
    offsets=[(int(a),int(b)) for a,b in enc["offset_mapping"].squeeze(0).tolist()]
    pos=locate_span_positions(offsets,start,end)
    if not pos:
        raise ValueError(f"span missing/truncated row={row.get('row_id')} cand={candidate!r} full_len={len(full)}")
    if min(offsets[p][0] for p in pos)>start or max(offsets[p][1] for p in pos)<end:
        raise ValueError(f"span partial row={row.get('row_id')} cand={candidate!r} span={(start,end)} covered={(min(offsets[p][0] for p in pos),max(offsets[p][1] for p in pos))}")
    input_ids=enc["input_ids"].to(device)
    att=enc["attention_mask"].to(device)
    target_ids=input_ids[0,pos].detach().clone()
    masked=input_ids.clone(); masked[0,pos]=int(tok.mask_token_id)
    logits=model(input_ids=masked, attention_mask=att).logits[0,pos]
    lp=F.log_softmax(logits, dim=-1)
    token_logps=lp[torch.arange(len(pos),device=device),target_ids].detach().cpu().tolist()
    top_ids=logits.argmax(dim=-1).detach().cpu().tolist()
    toks=[tok.convert_ids_to_tokens(int(x)) for x in target_ids.detach().cpu().tolist()]
    return {"n_tokens":len(pos),"sum_logp":float(sum(token_logps)),"mean_logp":float(sum(token_logps)/len(token_logps)),"token_logps":[float(x) for x in token_logps],"tokens":toks,"top_tokens":[tok.convert_ids_to_tokens(int(x)) for x in top_ids],"seq_len":int(input_ids.shape[1])}


def score_rows(model, tok, rows, device, seq_length):
    scored=[]; errors=[]
    for i,row in enumerate(rows):
        try:
            ans=score_candidate(model,tok,row,str(row["answer_text"]),device,seq_length)
            foi=score_candidate(model,tok,row,str(row["foil_text"]),device,seq_length)
            margin_mean=ans["mean_logp"]-foi["mean_logp"]
            margin_sum=ans["sum_logp"]-foi["sum_logp"]
            scored.append({
                "row_id": row.get("row_id"), "pair_id": row.get("pair_id"), "split": row.get("split"),
                "packet_type": row.get("packet_type"), "answer_state_kind": row.get("answer_state_kind"),
                "foil_state_kind": row.get("foil_state_kind"), "target_entity": row.get("target_entity"),
                "updated_entity": row.get("updated_entity"), "answer_text": row.get("answer_text"), "foil_text": row.get("foil_text"),
                "answer_mean_logp": ans["mean_logp"], "foil_mean_logp": foi["mean_logp"],
                "margin_mean_answer_minus_foil": margin_mean, "margin_sum_answer_minus_foil": margin_sum,
                "correct_mean": margin_mean > 0, "correct_sum": margin_sum > 0,
                "answer_n_tokens": ans["n_tokens"], "foil_n_tokens": foi["n_tokens"],
                "same_token_length": ans["n_tokens"] == foi["n_tokens"],
                "answer_tokens": " ".join(ans["tokens"]), "foil_tokens": " ".join(foi["tokens"]),
                "answer_top_tokens": " ".join(ans["top_tokens"][:8]), "foil_top_tokens": " ".join(foi["top_tokens"][:8]),
            })
        except Exception as e:
            errors.append({"i":i,"row_id":row.get("row_id"),"pair_id":row.get("pair_id"),"packet_type":row.get("packet_type"),"error":repr(e)})
    return scored, errors


def summarize(scored: list[dict[str, Any]]) -> dict[str, Any]:
    out={"n_rows":len(scored)}
    for key, sub in [("all", scored)] + [(pt, [r for r in scored if r["packet_type"]==pt]) for pt in ["UPDATED_USE","UNCHANGED_DISTRACTOR_USE"]]:
        vals=[float(r["margin_mean_answer_minus_foil"]) for r in sub if math.isfinite(float(r["margin_mean_answer_minus_foil"]))]
        out[key]={
            "n": len(sub),
            "correct_mean": sum(1 for r in sub if r["correct_mean"]),
            "accuracy_mean": sum(1 for r in sub if r["correct_mean"])/len(sub) if sub else float("nan"),
            "mean_margin_mean": statistics.mean(vals) if vals else float("nan"),
            "se_margin_mean": statistics.stdev(vals)/math.sqrt(len(vals)) if len(vals)>1 else float("nan"),
            "median_margin_mean": statistics.median(vals) if vals else float("nan"),
            "same_token_length_n": sum(1 for r in sub if r["same_token_length"]),
        }
    # Paired by original pair_id is sparse because research train/heldout are single-packet IDs; preserve singleton fact.
    by=collections.defaultdict(list)
    for r in scored:
        by[r["pair_id"]].append(r["packet_type"])
    out["pair_id_structure"]={"pair_ids":len(by),"ids_with_both_packet_types":sum(set(v)=={"UPDATED_USE","UNCHANGED_DISTRACTOR_USE"} for v in by.values())}
    return out


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", default=str(ROWS_DEFAULT))
    ap.add_argument("--model-path", default=str(MODEL_DEFAULT))
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seq-length", type=int, default=256)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--hf-cache-dir", default="")
    ap.add_argument("--max-rows", type=int, default=0)
    args=ap.parse_args()
    out=pathlib.Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    cache_env=configure_cache(pathlib.Path(args.hf_cache_dir) if args.hf_cache_dir else None)
    rows=read_jsonl(pathlib.Path(args.rows))
    if args.max_rows:
        rows=rows[:args.max_rows]
    device=torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    tok=AutoTokenizer.from_pretrained(str(args.model_path), use_fast=True, local_files_only=args.local_files_only)
    kwargs={"torch_dtype":torch.float32}
    if args.trust_remote_code:
        kwargs["trust_remote_code"]=True
    if args.local_files_only:
        kwargs["local_files_only"]=True
    model=AutoModelForMaskedLM.from_pretrained(str(args.model_path), **kwargs).eval().to(device)
    ident=model_identity(pathlib.Path(args.model_path), model, args.trust_remote_code, args.local_files_only)
    (out/"model_identity_preamble.jsonl").write_text(json.dumps(ident, ensure_ascii=False)+"\n", encoding="utf-8")
    scored, errors=score_rows(model,tok,rows,device,args.seq_length)
    write_csv(out/"row_scores.csv", scored)
    (out/"errors.jsonl").write_text("".join(json.dumps(e,ensure_ascii=False)+"\n" for e in errors), encoding="utf-8")
    summary={
        "status":"PLAINUSE_ANSWER_SLOT_BASELINE_DONE",
        "created_utc":now(),
        "rows_path":rel(pathlib.Path(args.rows)),
        "model_path":rel(pathlib.Path(args.model_path)),
        "out_dir":rel(out),
        "cache_env":cache_env,
        "model_identity":ident,
        "n_input_rows":len(rows),
        "n_scored_rows":len(scored),
        "n_errors":len(errors),
        "errors_first20":errors[:20],
        "summary":summarize(scored),
    }
    (out/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2,ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
