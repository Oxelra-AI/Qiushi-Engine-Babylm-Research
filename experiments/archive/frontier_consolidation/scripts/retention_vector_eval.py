#!/usr/bin/env python3
"""research: evaluate a label-free corpus-internal retention vector across checkpoints.

Input: `retention_probe/retention_probe.json`, built from the legal 10M corpus.
For each checkpoint, mask the exact same word group(s) and compute masked-piece NLL.
Aggregate by source, by structure (function/content), and source×structure.

Scientific use: test whether a benchmark-independent vector-retention signal (not a
scalar MLM loss or weight norm) identifies the scale1.75 late competence peak.
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path
from collections import defaultdict

# Custom adapter checkpoints need HuggingFace dynamic modules. The shared default
# cache is read-only in the execution environment, so force a local writable
# cache before importing transformers.
SESSION_HF_CACHE = Path("experiments/archive/frontier_consolidation/.hf_cache_retention")
os.environ["HF_HOME"] = str(SESSION_HF_CACHE / "home")
os.environ["HF_MODULES_CACHE"] = str(SESSION_HF_CACHE / "modules")
os.environ["TRANSFORMERS_CACHE"] = str(SESSION_HF_CACHE / "transformers")
(SESSION_HF_CACHE / "home").mkdir(parents=True, exist_ok=True)
(SESSION_HF_CACHE / "modules").mkdir(parents=True, exist_ok=True)
(SESSION_HF_CACHE / "transformers").mkdir(parents=True, exist_ok=True)

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM

ROOT = Path("experiments/archive/frontier_consolidation")
DEFAULT_RUN_HF = ROOT / "training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model"
DEFAULT_PROBE_JSON = ROOT / "data/retention_probe/retention_probe.json"
OUT_DIR = ROOT / "data/retention_vector"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Cheap7 from the saved official-compatible checkpoint sweep and hardened endpoint.
# 100M is the repaired full endpoint score; 82M hardened cheap7 is used when available.
CHEAP7 = {
    "chck_77M": 43.28214285714286,
    "chck_78M": 43.70214285714286,
    "chck_79M": 43.57857142857143,
    "chck_80M": 43.81214285714286,
    "chck_81M": 43.64928571428572,
    "chck_82M": 43.95944987645173,
    "chck_83M": 43.80785714285714,
    "chck_100M": 43.543159919261925,
}


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(Path.cwd()))
    except Exception:
        return str(p)


def checkpoint_key(name: str) -> int:
    # chck_82M -> 82
    return int(name.replace("chck_", "").replace("M", ""))


def prepare_examples(tokenizer, rows, max_length=256, limit=None):
    exs = []
    skipped = defaultdict(int)
    if limit:
        rows = rows[:limit]
    for r in rows:
        words = r["words"]
        masked_idx = int(r["masked_idx"])
        enc = tokenizer(
            words,
            is_split_into_words=True,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
        )
        word_ids = enc.word_ids()
        pos = [i for i, w in enumerate(word_ids) if w == masked_idx]
        if not pos:
            skipped["target_truncated_or_missing"] += 1
            continue
        input_ids = list(enc["input_ids"])
        labels = [-100] * len(input_ids)
        for p in pos:
            labels[p] = input_ids[p]
            input_ids[p] = tokenizer.mask_token_id
        exs.append({
            "input_ids": input_ids,
            "attention_mask": list(enc["attention_mask"]),
            "labels": labels,
            "source": r["source"],
            # Random probe rows only have is_function_word; targeted probe rows
            # carry a richer label-free corpus category such as negation,
            # wh, preposition_relation, rare_content, etc.
            "structure": r.get("structure") or ("function" if r["is_function_word"] else "content"),
            "target_word": r["target_word"],
            "n_masked_pieces": len(pos),
        })
    return exs, dict(skipped)


def make_batches(exs, pad_id, batch_size):
    for i in range(0, len(exs), batch_size):
        chunk = exs[i:i+batch_size]
        maxlen = max(len(e["input_ids"]) for e in chunk)
        input_ids=[]; attn=[]; labels=[]
        metas=[]
        for e in chunk:
            n=len(e["input_ids"])
            pad=maxlen-n
            input_ids.append(e["input_ids"] + [pad_id]*pad)
            attn.append(e["attention_mask"] + [0]*pad)
            labels.append(e["labels"] + [-100]*pad)
            metas.append(e)
        yield metas, torch.tensor(input_ids, dtype=torch.long), torch.tensor(attn, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def accum_init():
    return {"n_targets": 0, "n_pieces": 0, "nll_sum": 0.0, "nll_word_sum": 0.0}


def accum_add(acc, piece_sum, n_pieces):
    acc["n_targets"] += 1
    acc["n_pieces"] += int(n_pieces)
    acc["nll_sum"] += float(piece_sum)
    acc["nll_word_sum"] += float(piece_sum) / max(1, int(n_pieces))


def accum_finish(acc):
    out = dict(acc)
    out["piece_nll"] = (acc["nll_sum"] / acc["n_pieces"]) if acc["n_pieces"] else None
    out["word_mean_piece_nll"] = (acc["nll_word_sum"] / acc["n_targets"]) if acc["n_targets"] else None
    return out


def eval_one_checkpoint(ckpt_dir: Path, tokenizer, exs, device="cuda", batch_size=32, dtype="bf16"):
    t0=time.time()
    print(json.dumps({"event":"load_model","checkpoint":ckpt_dir.name,"path":rel(ckpt_dir),"utc":time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}), flush=True)
    model = AutoModelForMaskedLM.from_pretrained(str(ckpt_dir), trust_remote_code=True)
    model.eval().to(device)
    use_amp = (device.startswith("cuda") and dtype in {"bf16","fp16"})
    amp_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16

    overall = accum_init()
    by_source = defaultdict(accum_init)
    by_structure = defaultdict(accum_init)
    by_source_structure = defaultdict(accum_init)
    n_batches=0
    with torch.no_grad():
        for metas, input_ids, attn, labels in make_batches(exs, tokenizer.pad_token_id or 0, batch_size):
            input_ids=input_ids.to(device); attn=attn.to(device); labels=labels.to(device)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = model(input_ids=input_ids, attention_mask=attn).logits
            else:
                logits = model(input_ids=input_ids, attention_mask=attn).logits
            # Per-position CE without reduction.
            vocab = logits.shape[-1]
            ce = F.cross_entropy(logits.float().view(-1, vocab), labels.view(-1), ignore_index=-100, reduction="none").view(labels.shape)
            mask = labels.ne(-100)
            piece_sums = (ce * mask).sum(dim=1).detach().cpu().tolist()
            piece_counts = mask.sum(dim=1).detach().cpu().tolist()
            for meta, ps, pc in zip(metas, piece_sums, piece_counts):
                if pc <= 0:
                    continue
                s=meta["source"]; st=meta["structure"]; ss=f"{s}::{st}"
                accum_add(overall, ps, pc)
                accum_add(by_source[s], ps, pc)
                accum_add(by_structure[st], ps, pc)
                accum_add(by_source_structure[ss], ps, pc)
            n_batches += 1
            if n_batches % 50 == 0:
                print(json.dumps({"event":"progress","checkpoint":ckpt_dir.name,"batches":n_batches,"targets_done":min(n_batches*batch_size,len(exs))}), flush=True)
    # clean memory
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    rec = {
        "checkpoint": ckpt_dir.name,
        "checkpoint_M": checkpoint_key(ckpt_dir.name),
        "cheap7": CHEAP7.get(ckpt_dir.name),
        "elapsed_sec": round(time.time()-t0, 3),
        "overall": accum_finish(overall),
        "by_source": {k: accum_finish(v) for k,v in sorted(by_source.items())},
        "by_structure": {k: accum_finish(v) for k,v in sorted(by_structure.items())},
        "by_source_structure": {k: accum_finish(v) for k,v in sorted(by_source_structure.items())},
    }
    print(json.dumps({"event":"checkpoint_done","checkpoint":ckpt_dir.name,"elapsed_sec":rec["elapsed_sec"],"piece_nll":rec["overall"]["piece_nll"],"cheap7":rec["cheap7"]}), flush=True)
    return rec


def pearson(xs, ys):
    pairs=[(x,y) for x,y in zip(xs,ys) if x is not None and y is not None and math.isfinite(x) and math.isfinite(y)]
    if len(pairs)<3: return None
    xs=[p[0] for p in pairs]; ys=[p[1] for p in pairs]
    mx=sum(xs)/len(xs); my=sum(ys)/len(ys)
    vx=sum((x-mx)**2 for x in xs); vy=sum((y-my)**2 for y in ys)
    if vx<=0 or vy<=0: return None
    return sum((x-mx)*(y-my) for x,y in pairs)/math.sqrt(vx*vy)


def analyze(records):
    records=sorted(records, key=lambda r:r["checkpoint_M"])
    stratum_keys=sorted(records[0]["by_source_structure"].keys())
    # Label-free summaries: scalar MLM-like, balanced source/structure, cumulative forgetting burden.
    running_best = {k: float("inf") for k in stratum_keys}
    analyses=[]
    for rec in records:
        vals=[]; forget=[]; improvements=[]
        for k in stratum_keys:
            v=rec["by_source_structure"][k]["word_mean_piece_nll"]
            vals.append(v)
            forget.append(max(0.0, v-running_best[k]))
            improvements.append(running_best[k]-v if running_best[k] < float("inf") else 0.0)
        # update running_best AFTER computing forgetting vs past for this checkpoint
        for k,v in zip(stratum_keys, vals):
            if v < running_best[k]: running_best[k]=v
        mean=sum(vals)/len(vals)
        std=math.sqrt(sum((v-mean)**2 for v in vals)/len(vals))
        q90=sorted(vals)[int(0.9*(len(vals)-1))]
        forget_mean=sum(forget)/len(forget)
        forget_max=max(forget)
        # fixed vector-retention summary: prefer low mean NLL but penalize stratum forgetting and imbalance.
        # Coefficients are dimensionless/simple, not fitted to official labels.
        retention_score = mean + 1.0*forget_mean + 0.25*std
        analyses.append({
            "checkpoint": rec["checkpoint"],
            "checkpoint_M": rec["checkpoint_M"],
            "cheap7": rec.get("cheap7"),
            "macro_source_structure_nll": mean,
            "source_structure_std": std,
            "source_structure_q90_nll": q90,
            "forgetting_mean_vs_past_best": forget_mean,
            "forgetting_max_vs_past_best": forget_max,
            "retention_score_mean_plus_forget_plus_0p25std": retention_score,
            "overall_piece_nll": rec["overall"]["piece_nll"],
            "overall_word_nll": rec["overall"]["word_mean_piece_nll"],
        })
    # Selectors over checkpoints with known cheap7 and over all records separately.
    selector_fields=[
        "overall_piece_nll",
        "macro_source_structure_nll",
        "source_structure_q90_nll",
        "forgetting_mean_vs_past_best",
        "retention_score_mean_plus_forget_plus_0p25std",
    ]
    selectors={}
    for fld in selector_fields:
        selectors[fld] = min(analyses, key=lambda x:x[fld])["checkpoint"]
    # Correlations with cheap7 over available official cheap7 rows. Because NLL lower should mean better, use -metric.
    correlations={}
    for fld in selector_fields:
        xs=[-a[fld] for a in analyses]
        ys=[a.get("cheap7") for a in analyses]
        correlations[fld] = pearson(xs, ys)
    # Stratum correlations: which source×structure NLLs track cheap7?
    stratum_corr={}
    for k in stratum_keys:
        xs=[]; ys=[]
        for rec in records:
            xs.append(-rec["by_source_structure"][k]["word_mean_piece_nll"])
            ys.append(rec.get("cheap7"))
        stratum_corr[k]=pearson(xs, ys)
    top_pos=sorted([(v,k) for k,v in stratum_corr.items() if v is not None], reverse=True)[:8]
    top_neg=sorted([(v,k) for k,v in stratum_corr.items() if v is not None])[:8]
    return {"analysis_rows": analyses, "selectors_minimum": selectors, "correlations_with_cheap7_using_negative_nll": correlations, "top_positive_stratum_correlations": top_pos, "top_negative_stratum_correlations": top_neg}


def write_md(out_json, out_md, result):
    lines=[]
    lines.append("# research corpus-internal retention vector")
    lines.append("")
    lines.append("Inference-only fixed masked-word probe drawn from the legal 10M corpus. No official labels or benchmark data are used to compute the retention vector.")
    lines.append("")
    lines.append("## Probe summary")
    for k,v in result["probe_summary"].items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("## Checkpoint summary")
    lines.append("| checkpoint | cheap7 | overall piece NLL | macro source×structure NLL | q90 NLL | forgetting mean | vector retention score |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for a in result["analysis"]["analysis_rows"]:
        lines.append(f"| {a['checkpoint']} | {a.get('cheap7') if a.get('cheap7') is not None else ''} | {a['overall_piece_nll']:.6f} | {a['macro_source_structure_nll']:.6f} | {a['source_structure_q90_nll']:.6f} | {a['forgetting_mean_vs_past_best']:.6f} | {a['retention_score_mean_plus_forget_plus_0p25std']:.6f} |")
    lines.append("")
    lines.append("## Label-free selectors (lower is better)")
    for k,v in result["analysis"]["selectors_minimum"].items():
        lines.append(f"- `{k}` selects **{v}**")
    lines.append("")
    lines.append("## Correlation with official cheap7 (post-hoc reading, not used to build the probe)")
    lines.append("Correlations are Pearson r between official cheap7 and negative NLL/score over checkpoints with cheap7 available.")
    for k,v in result["analysis"]["correlations_with_cheap7_using_negative_nll"].items():
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("### Source×structure strata most positively correlated with cheap7")
    for v,k in result["analysis"]["top_positive_stratum_correlations"]:
        lines.append(f"- {k}: r={v:.4f}")
    lines.append("")
    lines.append("### Source×structure strata most negatively correlated with cheap7")
    for v,k in result["analysis"]["top_negative_stratum_correlations"]:
        lines.append(f"- {k}: r={v:.4f}")
    lines.append("")
    lines.append("## Scientific interpretation")
    sels=set(result["analysis"]["selectors_minimum"].values())
    if "chck_82M" in sels:
        lines.append("At least one fixed label-free retention summary selects the reproduced 82M peak. This is only an in-trajectory signal until tested on another seed or route.")
    else:
        lines.append("The fixed label-free retention summaries do not select the reproduced 82M peak in this first in-trajectory test. The 82M endpoint remains score-bearing, but this probe does not yet provide a general consolidation principle.")
    lines.append("")
    lines.append(f"JSON: `{rel(out_json)}`")
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", default=["chck_77M","chck_78M","chck_79M","chck_80M","chck_81M","chck_82M","chck_83M","chck_100M"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--dtype", choices=["fp32","bf16","fp16"], default="bf16")
    ap.add_argument("--limit-targets", type=int, default=0)
    ap.add_argument("--label", default="scale1p75_late77_83_100")
    ap.add_argument("--probe-json", default=str(DEFAULT_PROBE_JSON))
    ap.add_argument("--run-hf", default=str(DEFAULT_RUN_HF))
    args=ap.parse_args()
    if args.device.startswith("cuda"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device="cuda"
    else:
        device="cpu"
    probe_path = Path(args.probe_json)
    data=json.loads(probe_path.read_text())
    run_hf = Path(args.run_hf)
    tok=AutoTokenizer.from_pretrained(str(run_hf), use_fast=True, trust_remote_code=True)
    exs, skipped = prepare_examples(tok, data["rows"], limit=(args.limit_targets or None))
    print(json.dumps({"event":"probe_prepared","targets":len(exs),"skipped":skipped,"device":device,"checkpoints":args.checkpoints}), flush=True)
    records=[]
    for ck in args.checkpoints:
        ckpt_dir=run_hf/ck
        if not ckpt_dir.exists():
            raise FileNotFoundError(ckpt_dir)
        records.append(eval_one_checkpoint(ckpt_dir, tok, exs, device=device, batch_size=args.batch_size, dtype=args.dtype))
    result={
        "status":"RETENTION_VECTOR_DONE",
        "created_utc": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        "label": args.label,
        "run_hf": rel(run_hf),
        "probe_json": rel(probe_path),
        "probe_summary": data["summary"] | {"prepared_targets_after_tokenization": len(exs), "skipped": skipped},
        "checkpoints": args.checkpoints,
        "records": records,
        "analysis": analyze(records),
    }
    out_json=OUT_DIR/f"{args.label}.json"
    out_md=OUT_DIR/f"{args.label}.md"
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_md(out_json, out_md, result)
    print(json.dumps({"status":result["status"],"out_json":rel(out_json),"out_md":rel(out_md),"selectors":result["analysis"]["selectors_minimum"]}, indent=2), flush=True)

if __name__ == "__main__":
    main()
