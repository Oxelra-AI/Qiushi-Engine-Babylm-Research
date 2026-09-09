#!/usr/bin/env python3
"""research: frozen chck82 scoring of matched graph-transfer probe examples.

This is a forward-only measurement.  It tests whether an equalized structured
context changes the model's masked target-vs-opposite preference beyond a matched
neutral context and a minimally reversed context.  It does not train, upload, or
submit anything.
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
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/frontier_consolidation')
WORKSPACE = _public_path('experiments/archive/frontier_consolidation')
BASE = _public_path('experiments/archive/frontier_consolidation/data/matched_graph_transfer_probe')
DEFAULT_ACCEPTED = _public_path('experiments/archive/frontier_consolidation/data/matched_graph_transfer_probe/matched_probe_accepted.jsonl')
DEFAULT_OUT = _public_path('experiments/archive/frontier_consolidation/data/matched_graph_transfer_probe/frozen_chck82_scores')
DEFAULT_CKPT = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M')


def rel(p: Path | str) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def set_cache_env(out_dir: Path, gpu: int | None) -> None:
    cache = out_dir / "runtime_cache"
    for key, sub in {
        "HF_HOME": "home",
        "HF_HUB_CACHE": "hub",
        "HUGGINGFACE_HUB_CACHE": "hub",
        "HF_DATASETS_CACHE": "datasets",
        "TRANSFORMERS_CACHE": "transformers",
        "HF_MODULES_CACHE": "modules",
        "XDG_CACHE_HOME": "xdg",
    }.items():
        p = cache / sub
        p.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(p.resolve())
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows=[]
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def mean(xs: list[float]) -> float | None:
    xs=[float(x) for x in xs if math.isfinite(float(x))]
    return statistics.fmean(xs) if xs else None


def quantile(xs: list[float], q: float) -> float | None:
    xs=sorted(float(x) for x in xs if math.isfinite(float(x)))
    if not xs:
        return None
    if len(xs)==1:
        return xs[0]
    idx=(len(xs)-1)*q
    lo=math.floor(idx); hi=math.ceil(idx)
    if lo==hi:
        return xs[lo]
    return xs[lo]*(hi-idx)+xs[hi]*(idx-lo)


def summarize(xs: list[float]) -> dict[str, Any]:
    xs=[float(x) for x in xs if math.isfinite(float(x))]
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "mean": mean(xs), "median": quantile(xs,0.5), "p10": quantile(xs,0.1), "p90": quantile(xs,0.9), "min": min(xs), "max": max(xs)}


def bootstrap_mean_ci(xs: list[float], *, seed: int=166, reps: int=5000) -> dict[str, Any]:
    xs=[float(x) for x in xs if math.isfinite(float(x))]
    if not xs:
        return {"n": 0}
    rng=random.Random(seed)
    n=len(xs)
    means=[]
    for _ in range(reps):
        means.append(sum(xs[rng.randrange(n)] for _ in range(n))/n)
    means.sort()
    return {"n": n, "mean": sum(xs)/n, "ci2p5": means[int(0.025*(reps-1))], "ci97p5": means[int(0.975*(reps-1))], "p_mean_le_0": sum(1 for m in means if m <= 0)/reps}


def find_span_token(tokenizer, text: str, start: int, end: int) -> tuple[list[int], int, dict[str, Any]]:
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
    ids=[int(x) for x in enc["input_ids"]]
    offsets=list(enc["offset_mapping"])
    positions=[]
    for i,(s,e) in enumerate(offsets):
        if e > s and e > start and s < end:
            positions.append(i)
    info={"positions": positions, "offsets": [offsets[i] for i in positions], "tokens": tokenizer.convert_ids_to_tokens([ids[i] for i in positions])}
    if len(positions) != 1:
        raise ValueError(f"span did not map to exactly one token: {info}")
    return ids, int(positions[0]), info


def build_masked(tokenizer, context: str, query_prefix: str, target: str, distractor: str, query_suffix: str, max_seq_len: int) -> tuple[dict[str, Any], dict[str, Any]]:
    context = (context or "").strip()
    qp = (query_prefix or "").strip()
    qs = (query_suffix or "").strip()
    context_part = (context + "\n") if context else ""
    pre = context_part + qp + " "
    # Put a space before suffix unless suffix begins with punctuation.
    sep_after = "" if qs[:1] in {".", ",", ";", ":", "?", "!"} else " "
    text = pre + target + sep_after + qs
    t_start = len(pre); t_end = t_start + len(target)
    ids, pos, target_info = find_span_token(tokenizer, text, t_start, t_end)
    text_d = pre + distractor + sep_after + qs
    d_start = len(pre); d_end = d_start + len(distractor)
    ids_d, pos_d, dist_info = find_span_token(tokenizer, text_d, d_start, d_end)
    # Same tokenizer context length is useful, but target and distractor may have different token ids at one position.
    target_id = int(ids[pos])
    distractor_id = int(ids_d[pos_d])
    masked=list(ids)
    masked[pos] = int(tokenizer.mask_token_id)
    if len(masked) > max_seq_len:
        raise ValueError(f"sequence too long {len(masked)}>{max_seq_len}")
    ctx={"input_ids": masked, "mask_index": pos, "target_id": target_id, "distractor_id": distractor_id}
    return ctx, {"text": text, "masked_len": len(masked), "target_token_info": target_info, "distractor_token_info": dist_info, "target_id": target_id, "distractor_id": distractor_id}


def batch_score(model, tokenizer, contexts: list[dict[str, Any]], *, device: str, batch_size: int) -> list[dict[str, float]]:
    import torch
    import torch.nn.functional as F
    pad=tokenizer.pad_token_id
    out=[]
    for start in range(0, len(contexts), batch_size):
        batch=contexts[start:start+batch_size]
        max_len=max(len(c["input_ids"]) for c in batch)
        x=[]; am=[]; mi=[]; ti=[]; di=[]
        for c in batch:
            ids=list(c["input_ids"])
            padn=max_len-len(ids)
            x.append(ids+[pad]*padn)
            am.append([1]*len(ids)+[0]*padn)
            mi.append(int(c["mask_index"]))
            ti.append(int(c["target_id"]))
            di.append(int(c["distractor_id"]))
        with torch.no_grad():
            xt=torch.tensor(x,dtype=torch.long,device=device)
            mt=torch.tensor(am,dtype=torch.long,device=device)
            logits=model(input_ids=xt, attention_mask=mt).logits
            row=torch.arange(len(batch),device=device)
            pos=torch.tensor(mi,dtype=torch.long,device=device)
            logp=F.log_softmax(logits[row,pos,:], dim=-1)
            target=torch.tensor(ti,dtype=torch.long,device=device)
            dist=torch.tensor(di,dtype=torch.long,device=device)
            lp_t=logp[row,target]
            lp_d=logp[row,dist]
            vals=torch.stack([lp_t,lp_d,lp_t-lp_d],dim=1).detach().cpu().tolist()
            for a,b,c in vals:
                out.append({"logp_target": float(a), "logp_distractor": float(b), "delta_target_minus_distractor": float(c), "target_nll": float(-a), "distractor_nll": float(-b)})
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir=Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_cache_env(out_dir, args.gpu)
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    device=args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    ckpt=Path(args.checkpoint)
    tokenizer=AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True, local_files_only=True)
    model=AutoModelForMaskedLM.from_pretrained(ckpt, trust_remote_code=True, local_files_only=True)
    model.to(device); model.eval()

    accepted=load_jsonl(Path(args.accepted))
    if args.max_items and args.max_items > 0:
        accepted=accepted[:args.max_items]
    contexts=[]; row_context_keys=[]; build_failures=[]
    for rec in accepted:
        gen=rec.get("generated") or {}
        target=str(gen.get("target","")).strip().lower()
        distractor=str(gen.get("distractor","")).strip().lower()
        qp=str(gen.get("query_prefix","")).strip()
        qs=str(gen.get("query_suffix","")).strip()
        context_map={
            "query_only": "",
            "neutral": str(gen.get("view_neutral","")).strip(),
            "structured": str(gen.get("view_structured","")).strip(),
            "reversed": str(gen.get("view_reversed","")).strip(),
        }
        for kind, view in context_map.items():
            try:
                ctx, info = build_masked(tokenizer, view, qp, target, distractor, qs, args.max_seq_len)
                ctx["item_id"] = rec.get("prompt_id")
                ctx["context_kind"] = kind
                contexts.append(ctx)
                row_context_keys.append((rec, kind, info))
            except Exception as e:
                build_failures.append({"prompt_id": rec.get("prompt_id"), "context_kind": kind, "error": repr(e)})
    if not contexts:
        raise RuntimeError("no contexts built")
    t0=time.time()
    scores=batch_score(model, tokenizer, contexts, device=device, batch_size=args.batch_size)
    elapsed=time.time()-t0
    rows=[]
    for (rec, kind, info), score in zip(row_context_keys, scores):
        gen=rec.get("generated") or {}
        row={
            "prompt_id": rec.get("prompt_id"),
            "row_index": rec.get("row_index"),
            "family": rec.get("family"),
            "source_bucket": rec.get("source_bucket"),
            "context_kind": kind,
            "target": gen.get("target"),
            "distractor": gen.get("distractor"),
            "masked_len": info.get("masked_len"),
            "target_id": info.get("target_id"),
            "distractor_id": info.get("distractor_id"),
            **score,
        }
        rows.append(row)
    csv_path=out_dir/"matched_probe_frozen_scores.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w=csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader(); w.writerows(rows)
    jsonl_path=out_dir/"matched_probe_frozen_scores.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False)+"\n")

    by_item=defaultdict(dict)
    for r in rows:
        by_item[str(r["prompt_id"])][str(r["context_kind"])] = r
    paired=[]
    for pid, mm in by_item.items():
        if {"query_only","neutral","structured","reversed"} <= set(mm):
            st=mm["structured"]; ne=mm["neutral"]; rev=mm["reversed"]; qo=mm["query_only"]
            paired.append({
                "prompt_id": pid,
                "family": st.get("family"),
                "structured_delta": st["delta_target_minus_distractor"],
                "neutral_delta": ne["delta_target_minus_distractor"],
                "reversed_delta": rev["delta_target_minus_distractor"],
                "query_only_delta": qo["delta_target_minus_distractor"],
                "structured_target_nll": st["target_nll"],
                "neutral_target_nll": ne["target_nll"],
                "reversed_target_nll": rev["target_nll"],
                "query_only_target_nll": qo["target_nll"],
                "lift_delta_structured_minus_neutral": st["delta_target_minus_distractor"] - ne["delta_target_minus_distractor"],
                "sensitivity_delta_structured_minus_reversed": st["delta_target_minus_distractor"] - rev["delta_target_minus_distractor"],
                "target_nll_improvement_neutral_minus_structured": ne["target_nll"] - st["target_nll"],
                "target_nll_improvement_reversed_minus_structured": rev["target_nll"] - st["target_nll"],
            })
    pair_csv=out_dir/"matched_probe_paired_effects.csv"
    if paired:
        with pair_csv.open("w", encoding="utf-8", newline="") as f:
            w=csv.DictWriter(f, fieldnames=sorted({k for r in paired for k in r}))
            w.writeheader(); w.writerows(paired)

    summary={
        "status": "MATCHED_GRAPH_TRANSFER_FROZEN_SCORE",
        "accepted_file": rel(args.accepted),
        "checkpoint": rel(ckpt),
        "device": device,
        "gpu_env": args.gpu,
        "n_accepted_input": len(accepted),
        "n_contexts_scored": len(rows),
        "n_complete_paired_items": len(paired),
        "build_failures": build_failures,
        "elapsed_scoring_sec": elapsed,
        "score_csv": rel(csv_path),
        "paired_effects_csv": rel(pair_csv),
        "score_jsonl": rel(jsonl_path),
        "by_context": {},
        "paired_effects": {},
        "by_family_paired_effects": {},
        "interpretation_rule": "Route remains live only if the matched packet is large enough and structured contexts improve target-vs-distractor margins over both neutral and reversed contexts without target/distractor lexical availability. Positive numbers here are not sufficient for 4M training without independent quality review of examples.",
    }
    for kind in ["query_only","neutral","structured","reversed"]:
        rs=[r for r in rows if r.get("context_kind")==kind]
        summary["by_context"][kind]={
            "n": len(rs),
            "target_preferred_fraction": mean([1.0 if r["delta_target_minus_distractor"]>0 else 0.0 for r in rs]),
            "delta": summarize([r["delta_target_minus_distractor"] for r in rs]),
            "target_nll": summarize([r["target_nll"] for r in rs]),
        }
    for key in ["lift_delta_structured_minus_neutral", "sensitivity_delta_structured_minus_reversed", "target_nll_improvement_neutral_minus_structured", "target_nll_improvement_reversed_minus_structured"]:
        vals=[float(r[key]) for r in paired]
        summary["paired_effects"][key]={"summary": summarize(vals), "bootstrap": bootstrap_mean_ci(vals, seed=args.seed)}
    fams=defaultdict(list)
    for r in paired:
        fams[str(r.get("family"))].append(r)
    for fam,rs in sorted(fams.items()):
        summary["by_family_paired_effects"][fam]={
            "n": len(rs),
            "mean_lift_structured_minus_neutral": mean([r["lift_delta_structured_minus_neutral"] for r in rs]),
            "mean_sensitivity_structured_minus_reversed": mean([r["sensitivity_delta_structured_minus_reversed"] for r in rs]),
            "structured_preferred_fraction": mean([1.0 if r["structured_delta"]>0 else 0.0 for r in rs]),
        }
    out_json=out_dir/"matched_probe_frozen_score_summary.json"
    out_md=out_dir/"matched_probe_frozen_score_summary.md"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    lines=[
        "# research frozen chck82 matched graph-transfer scores",
        "",
        "Forward-only measurement on lexically matched generated probes. No training, upload, or leaderboard submission.",
        f"Input accepted examples: {len(accepted)}; complete paired items: {len(paired)}; contexts scored: {len(rows)}",
        f"Checkpoint: `{rel(ckpt)}`; device `{device}`; elapsed score sec `{elapsed:.1f}`",
        "",
        "## Context-level target-vs-distractor margins",
    ]
    for kind, s in summary["by_context"].items():
        lines.append(f"- {kind}: n={s['n']}, target-preferred={s['target_preferred_fraction']}, mean delta={s['delta'].get('mean')}, mean target NLL={s['target_nll'].get('mean')}")
    lines.extend(["", "## Paired effects"])
    for key, obj in summary["paired_effects"].items():
        b=obj["bootstrap"]
        lines.append(f"- {key}: mean {b.get('mean')}, CI [{b.get('ci2p5')}, {b.get('ci97p5')}], P(mean<=0)={b.get('p_mean_le_0')}")
    lines.extend(["", "## Family means"])
    for fam, s in summary["by_family_paired_effects"].items():
        lines.append(f"- {fam}: n={s['n']}, lift={s['mean_lift_structured_minus_neutral']}, sensitivity={s['mean_sensitivity_structured_minus_reversed']}, structured_pref_frac={s['structured_preferred_fraction']}")
    if build_failures:
        lines.extend(["", "## Build failures"])
        for bf in build_failures[:20]:
            lines.append(f"- {bf}")
    lines.extend(["", f"CSV: `{rel(csv_path)}`", f"Paired effects: `{rel(pair_csv)}`", f"JSON: `{rel(out_json)}`"])
    out_md.write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "paired_items": len(paired), "out_json": rel(out_json), "out_md": rel(out_md)}, indent=2), flush=True)
    return summary


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--accepted", default=str(DEFAULT_ACCEPTED))
    ap.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-seq-len", type=int, default=256)
    ap.add_argument("--max-items", type=int, default=0)
    ap.add_argument("--seed", type=int, default=16603)
    args=ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
