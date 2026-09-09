#!/usr/bin/env python3
"""research: CPU-only GlobalPIQA option-margin reader for existing MLM checkpoints.

Scientific purpose
------------------
Existing official prediction artifacts expose only the chosen option.  The current
research needs to know whether GlobalPIQA_parallel failures are no-signal failures
(correct option rank 3-4, large wrong margin) or near-miss failures (correct option
rank 2, small margin) before spending more 100M-word H100 runs.  This script mirrors
the official MLM completion-scoring path for GlobalPIQA and saves all option scores.

Important boundary
------------------
This is a diagnostic on existing endpoints.  It must not be used to tune a custom
scorer on official examples for submission.  If prior-subtraction or margin shaping
looks promising here, the calibration/training rule must be developed on a corpus-
derived held-out contrast set, not on these official items.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

# Force CPU unless caller deliberately overrides before import; this script is intended to
# avoid competing with H100 training.  It still respects --threads for CPU parallelism.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM

USER_ROOT = Path.cwd()
WORKSPACE = USER_ROOT / "experiments/archive" / 'representation_and_objectives'
STRICT = USER_ROOT / "experiments/archive" / 'initial_model_studies' / "repos" / "babylm-eval" / "strict"
if str(STRICT) not in sys.path:
    sys.path.insert(0, str(STRICT))

from evaluation_pipeline.sentence_zero_shot.dataset import get_dataloader  # noqa: E402

DATA_ROOT = WORKSPACE / "data" / "globalpiqa_official_lineage" / "official_dl_scratch" / "generated_by_current_official_dl" / "evaluation_data" / "full_eval"
OUT_ROOT = WORKSPACE / "data" / "globalpiqa_margin_reader"
NOTE = WORKSPACE / "notes" / "globalpiqa_margin_reader.md"
ANATOMY_JSON = WORKSPACE / "data" / "globalpiqa_parallel_anatomy" / "globalpiqa_parallel_anatomy.json"

TARGETS: Dict[str, Dict[str, Any]] = {
    "legal40k8x480_43022": {
        "label": "legal40k 8x480 seed43022, best compliant endpoint",
        "model_root": WORKSPACE / "training/runs/legal40k_accum_compact_view_reinvest_seed43022/hf_model",
        "revision": "chck_100M",
        "official_parallel": 22.33,
        "official_nonparallel": 47.0,
    },
    "legal16k8x480_43022": {
        "label": "legal16k 8x480 seed43022",
        "model_root": WORKSPACE / "training/runs/strictsmalltok_compact_view_reinvest_seed43022/hf_model",
        "revision": "chck_100M",
        "official_parallel": 26.21,
        "official_nonparallel": 46.0,
    },
    "depth12x384_43022": {
        "label": "legal40k 12x384 depth seed43022",
        "model_root": WORKSPACE / "training/runs/legal40k_12x384_depth_compact_view_reinvest_seed43022/hf_model",
        "revision": "chck_100M",
        "official_parallel": 24.27,
        "official_nonparallel": 47.0,
    },
    "inherited16k_noncompliant_43022": {
        "label": "inherited-tokenizer compact reinvest seed43022 (non-submission mechanism evidence)",
        "model_root": USER_ROOT / "experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/hf_model",
        "revision": "chck_100M",
        "official_parallel": 25.24,
        "official_nonparallel": 46.0,
    },
}

TASKS = {
    "parallel": {
        "task": "global_piqa_parallel",
        "data_path": DATA_ROOT / "global_piqa_parallel",
        "num_choices": 4,
        "chance": 0.25,
    },
    "nonparallel": {
        "task": "global_piqa_nonparallel",
        "data_path": DATA_ROOT / "global_piqa_nonparallel",
        "num_choices": 2,
        "chance": 0.50,
    },
}


def load_always_wrong_ids() -> set[str]:
    if not ANATOMY_JSON.exists():
        return set()
    d = json.loads(ANATOMY_JSON.read_text())
    rows = d.get("agreement", {}).get("parallel", {}).get("rows", [])
    return {r["example_id"] for r in rows if r.get("n_ok") == 0}


def load_categories() -> Dict[str, List[str]]:
    if not ANATOMY_JSON.exists():
        return {}
    d = json.loads(ANATOMY_JSON.read_text())
    rows = d.get("agreement", {}).get("parallel", {}).get("rows", [])
    return {r["example_id"]: r.get("categories") or [] for r in rows}


def dataloader_args(model_root: Path, revision: str, mode: str, batch_size: int, non_causal_batch_size: int) -> argparse.Namespace:
    spec = TASKS[mode]
    return argparse.Namespace(
        data_path=spec["data_path"].resolve(),
        task=spec["task"],
        model_path_or_name=str(model_root.resolve()),
        backend="mlm",
        output_dir=OUT_ROOT,
        images_path=None,
        image_split=None,
        image_template=None,
        revision_name=revision,
        min_temperature=1.0,
        max_temperature=None,
        temperature_interval=0.05,
        batch_size=batch_size,
        non_causal_batch_size=non_causal_batch_size,
        full_sentence_scores=False,
        save_predictions=False,
    )


def score_mode(model: torch.nn.Module, args: argparse.Namespace, mode: str, max_items: Optional[int] = None) -> List[Dict[str, Any]]:
    dataloader = get_dataloader(args)
    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    processed = 0
    with torch.no_grad():
        for raw_sentences, sentence_dict, labels, metadatas, uids, images in dataloader:
            num_sentences = len([key for key in sentence_dict.keys() if key.endswith("attn_mask")])
            prefixes = [f"sentence_{sentence_idx}" for sentence_idx in range(num_sentences)]
            batch_candidate_scores: List[List[float]] = []
            batch_candidate_lengths: List[List[int]] = []
            for prefix in prefixes:
                num_examples = sentence_dict[f"{prefix}_tokens"].shape[0]
                bsz = args.non_causal_batch_size
                num_batches = math.ceil(num_examples / bsz)
                individual_log_probs: List[torch.Tensor] = []
                for batch_idx in range(num_batches):
                    tokens = sentence_dict[f"{prefix}_tokens"][batch_idx * bsz:(batch_idx + 1) * bsz]
                    attn_mask = sentence_dict[f"{prefix}_attn_mask"][batch_idx * bsz:(batch_idx + 1) * bsz]
                    indices = sentence_dict[f"{prefix}_indices"][batch_idx * bsz:(batch_idx + 1) * bsz]
                    targets = sentence_dict[f"{prefix}_targets"][batch_idx * bsz:(batch_idx + 1) * bsz]
                    logits = model(input_ids=tokens, attention_mask=attn_mask)
                    logits = logits[0] if isinstance(logits, tuple) else logits["logits"]
                    if logits.size(1) != sentence_dict[f"{prefix}_tokens"].size(1):
                        logits = logits[:, -sentence_dict[f"{prefix}_tokens"].size(1):]
                    minibatch_indices = torch.arange(logits.shape[0])
                    masked_logits = logits[minibatch_indices, indices]
                    log_probs = F.log_softmax(masked_logits, dim=-1)
                    target_log_probs = torch.gather(log_probs, -1, targets.unsqueeze(-1)).squeeze(-1)
                    individual_log_probs.append(target_log_probs.cpu())
                concat = torch.cat(individual_log_probs, dim=0) if individual_log_probs else torch.empty(0)
                summed_scores=[]
                lengths=[]
                curr=0
                for examples_per_batch in sentence_dict[f"{prefix}_examples_per_batch"]:
                    start=curr; end=curr+examples_per_batch
                    raw_sum = float(torch.sum(concat[start:end]).item()) if end > start else float("nan")
                    # GlobalPIQA official path length-normalizes by number of completion tokens.
                    score = raw_sum / max(examples_per_batch, 1)
                    summed_scores.append(score)
                    lengths.append(int(examples_per_batch))
                    curr=end
                batch_candidate_scores.append(summed_scores)
                batch_candidate_lengths.append(lengths)
            # Transpose candidates x batch -> batch x candidates.
            for bi, (raw_sentence_dict, label, metadata, uid) in enumerate(zip(raw_sentences, labels, metadatas, uids)):
                scores = [batch_candidate_scores[c][bi] for c in range(num_sentences)]
                lengths = [batch_candidate_lengths[c][bi] for c in range(num_sentences)]
                order = sorted(range(num_sentences), key=lambda j: scores[j], reverse=True)
                choice = order[0]
                lab = int(label)
                rank = order.index(lab) + 1
                top_score = scores[choice]
                correct_score = scores[lab]
                second_score = scores[order[1]] if len(order) > 1 else float("nan")
                row = {
                    "mode": mode,
                    "example_id": uid,
                    "label": lab,
                    "choice": choice,
                    "correct": choice == lab,
                    "correct_rank": rank,
                    "top_minus_correct": top_score - correct_score,
                    "correct_minus_second_best": correct_score - (second_score if choice == lab else top_score),
                    "top_score": top_score,
                    "correct_score": correct_score,
                    "scores": scores,
                    "completion_token_lengths": lengths,
                    "prompt": raw_sentence_dict.get("prefixes", [""])[0],
                    "completions": raw_sentence_dict.get("completions", []),
                    "metadata": dict(metadata),
                    "elapsed_so_far_sec": round(time.time() - t0, 3),
                }
                rows.append(row)
                processed += 1
                if max_items is not None and processed >= max_items:
                    return rows
    return rows


def summarize_rows(rows: List[Dict[str, Any]], mode: str, always_wrong_ids: set[str], categories: Dict[str, List[str]]) -> Dict[str, Any]:
    n=len(rows)
    correct=sum(1 for r in rows if r["correct"])
    choice=Counter(str(r["choice"]) for r in rows)
    label=Counter(str(r["label"]) for r in rows)
    rank=Counter(str(r["correct_rank"]) for r in rows)
    aw=[r for r in rows if r["example_id"] in always_wrong_ids]
    def sub_summary(sub: List[Dict[str,Any]]) -> Dict[str,Any]:
        if not sub:
            return {"n":0}
        return {
            "n": len(sub),
            "accuracy": 100*sum(r["correct"] for r in sub)/len(sub),
            "correct_rank_counts": dict(Counter(str(r["correct_rank"]) for r in sub)),
            "median_top_minus_correct": sorted([r["top_minus_correct"] for r in sub])[len(sub)//2],
            "mean_top_minus_correct": sum(r["top_minus_correct"] for r in sub)/len(sub),
            "small_wrong_margin_le_0p25_nats": sum((not r["correct"]) and r["top_minus_correct"] <= 0.25 for r in sub),
            "small_wrong_margin_le_0p50_nats": sum((not r["correct"]) and r["top_minus_correct"] <= 0.50 for r in sub),
        }
    # Category summaries for parallel rows only.
    cat_tot=Counter(); cat_ok=Counter(); cat_rank=defaultdict(Counter); cat_margin=defaultdict(list)
    for r in rows:
        cats = categories.get(r["example_id"], []) if mode == "parallel" else []
        if not cats:
            cats = ["all"]
        for cat in cats:
            cat_tot[cat]+=1
            cat_ok[cat]+=int(r["correct"])
            cat_rank[cat][str(r["correct_rank"])] += 1
            cat_margin[cat].append(r["top_minus_correct"])
    cat_table=[]
    for cat, tot in cat_tot.most_common():
        margins=cat_margin[cat]
        cat_table.append({
            "category": cat,
            "n": tot,
            "accuracy": 100*cat_ok[cat]/tot,
            "correct_rank_counts": dict(cat_rank[cat]),
            "mean_top_minus_correct": sum(margins)/len(margins),
        })
    spec=TASKS[mode]
    chance=spec["chance"]
    chance_adj=(correct/n - chance)/(1-chance) if n and chance < 1 else None
    return {
        "mode": mode,
        "n": n,
        "accuracy": 100*correct/n if n else None,
        "chance_adjusted_accuracy": chance_adj,
        "choice_counts": dict(choice),
        "label_counts": dict(label),
        "correct_rank_counts": dict(rank),
        "always_wrong_subset": sub_summary(aw) if mode == "parallel" else None,
        "all_rows_margin_summary": sub_summary(rows),
        "category_summary": cat_table,
    }


def run_target(target_key: str, modes: List[str], batch_size: int, non_causal_batch_size: int, max_items: Optional[int], threads: int) -> Dict[str, Any]:
    meta=TARGETS[target_key]
    model_root=Path(meta["model_root"])
    if not model_root.exists():
        raise FileNotFoundError(f"model_root missing for {target_key}: {model_root}")
    torch.set_num_threads(max(1, int(threads)))
    t0=time.time()
    model=AutoModelForMaskedLM.from_pretrained(str(model_root.resolve()), trust_remote_code=True, revision=meta["revision"])
    model.eval()
    model.to("cpu")
    load_sec=time.time()-t0
    always_wrong_ids=load_always_wrong_ids()
    categories=load_categories()
    result={
        "target": target_key,
        "label": meta["label"],
        "model_root": str(model_root),
        "revision": meta["revision"],
        "device": "cpu",
        "threads": threads,
        "max_items": max_items,
        "model_load_sec": round(load_sec,3),
        "modes": {},
    }
    for mode in modes:
        args=dataloader_args(model_root, meta["revision"], mode, batch_size, non_causal_batch_size)
        rows=score_mode(model, args, mode, max_items=max_items)
        summary=summarize_rows(rows, mode, always_wrong_ids, categories)
        result["modes"][mode]={"summary": summary, "rows": rows}
    return result


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=["legal40k8x480_43022"], choices=sorted(TARGETS))
    ap.add_argument("--modes", nargs="+", default=["parallel"], choices=sorted(TASKS))
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--non_causal_batch_size", type=int, default=32)
    ap.add_argument("--max_items", type=int, default=None)
    ap.add_argument("--threads", type=int, default=8)
    args=ap.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_results={
        "status": "GLOBALPIQA_MARGIN_READER_DONE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "boundary": "Diagnostic only; no official-example tuning for submission.",
        "targets": {},
    }
    for target in args.targets:
        print(json.dumps({"event":"target_start", "target":target, "modes":args.modes}), flush=True)
        res=run_target(target, args.modes, args.batch_size, args.non_causal_batch_size, args.max_items, args.threads)
        all_results["targets"][target]=res
        target_json=OUT_ROOT / f"{target}_margins.json"
        target_json.write_text(json.dumps(res, indent=2, ensure_ascii=False))
        # rows csv per mode
        for mode, md in res["modes"].items():
            rows=md["rows"]
            if rows:
                out_csv=OUT_ROOT / f"{target}_{mode}_rows.csv"
                flat=[]
                for r in rows:
                    rr={k:v for k,v in r.items() if k not in ("scores","completions","metadata")}
                    rr.update({f"score_{i}": s for i,s in enumerate(r["scores"])})
                    rr.update({f"completion_{i}": c for i,c in enumerate(r["completions"])})
                    flat.append(rr)
                with out_csv.open("w", newline="") as f:
                    writer=csv.DictWriter(f, fieldnames=sorted({k for rr in flat for k in rr.keys()}))
                    writer.writeheader(); writer.writerows(flat)
        print(json.dumps({"event":"target_done", "target":target, "summaries": {m:res['modes'][m]['summary'] for m in res['modes']}}, indent=2), flush=True)
    combined_json=OUT_ROOT / "globalpiqa_margin_reader_results.json"
    combined_json.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    # Note summarizing if full run (not max_items pilot).
    lines=[]
    lines.append("# research — GlobalPIQA option-margin reader")
    lines.append("")
    lines.append("This CPU-only diagnostic mirrors the official MLM length-normalized completion scoring for GlobalPIQA and records all candidate scores. It is not a custom submission scorer and must not be tuned on official rows.")
    lines.append("")
    for target,res in all_results["targets"].items():
        lines.append(f"## `{target}`")
        lines.append("")
        lines.append(f"Model: `{res['model_root']}` revision `{res['revision']}`; load {res['model_load_sec']} sec on CPU.")
        for mode,md in res["modes"].items():
            s=md["summary"]
            lines.append(f"- {mode}: n={s['n']}, accuracy={s['accuracy']:.2f}, chance-adjusted={s['chance_adjusted_accuracy']:.3f}, rank counts={s['correct_rank_counts']}, choice counts={s['choice_counts']}")
            if mode == "parallel" and s.get("always_wrong_subset"):
                aw=s["always_wrong_subset"]
                lines.append(f"  - 52-row cross-endpoint always-wrong subset in this target: accuracy={aw.get('accuracy'):.2f}, rank counts={aw.get('correct_rank_counts')}, mean top-minus-correct={aw.get('mean_top_minus_correct'):.3f}, small wrong margins ≤0.25/≤0.50 nats={aw.get('small_wrong_margin_le_0p25_nats')}/{aw.get('small_wrong_margin_le_0p50_nats')}")
        lines.append("")
    lines.append("Files:")
    lines.append(f"- combined JSON: `{combined_json}`")
    lines.append(f"- per-target JSON/CSV under `{OUT_ROOT}`")
    NOTE.write_text("\n".join(lines)+"\n")
    print(json.dumps({"status": all_results["status"], "combined_json": str(combined_json), "note": str(NOTE)}, indent=2), flush=True)

if __name__ == "__main__":
    main()
