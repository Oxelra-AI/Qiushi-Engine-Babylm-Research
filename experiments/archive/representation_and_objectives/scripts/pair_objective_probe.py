#!/usr/bin/env python3
"""research: internal pair-objective probe for forked causal compact screens.

Official BabyLM endpoints are noisy and may mature late. This probe reads the
trained fork models directly on the exact compact pair multiset and measures the
causal second-segment prediction objective in both orientations:

  forward: source -> side (rewrite/extract/adjbreak target)
  reverse: side   -> source (source target)

For each available model arm it reports side-target NLL split into copied vs
noncopied targets, where "copied" means the target token type occurred in the
context segment. It also computes, when all four arms exist, the reciprocal
mixed-vs-oneway interaction

  I_loss = 0.5*(FR + RF) - 0.5*(FF + RR)

for each eval orientation and target class. Negative I_loss means mixed
bidirectional exposure lowers pair-target loss relative to one-direction
baselines. Direction-matched continuation effects are also reported:

  reverse target after forward prefix: FR - RR
  forward target after reverse prefix: RF - FF

These losses are internal mechanistic evidence only. They do not replace official
endpoint scoring or the copy-matched semantic/random controls.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import time
from typing import Any, Dict, Iterable, List

import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

V3_PATH = pathlib.Path("experiments/archive/representation_and_objectives/scripts/pair_aware_causal_trainer_v3.py")
spec = importlib.util.spec_from_file_location("v3", V3_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not import {V3_PATH}")
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)

ARMS = {
    "prefix_forward": "forward_branch/prefix_forward/hf_model/prefix_epoch1",
    "prefix_reverse": "reverse_branch/prefix_reverse/hf_model/prefix_epoch1",
    "ff": "forward_branch/ff/hf_model/final",
    "fr": "forward_branch/fr/hf_model/final",
    "rr": "reverse_branch/rr/hf_model/final",
    "rf": "reverse_branch/rf/hf_model/final",
}
FINAL_ARMS = ["ff", "fr", "rr", "rf"]
EVAL_SCHEDULE = {"forward": "ff", "reverse": "rr"}


def load_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def available_models(run_root: pathlib.Path) -> Dict[str, pathlib.Path]:
    out = {}
    for arm, rel in ARMS.items():
        p = run_root / rel
        if (p / "config.json").exists():
            out[arm] = p
    return out


def build_pair_eval_sequences(pair_jsonl: pathlib.Path, tokenizer, seq_len: int, pad_id: int) -> Dict[str, List[dict]]:
    pairs, charged_words, pair_token_stats = v3.pre_tokenize_pairs(pair_jsonl, tokenizer, max_pairs=0)
    seqs = {}
    for orientation, schedule in EVAL_SCHEDULE.items():
        cur = []
        for p in pairs:
            cur.extend(v3.make_pair_seqs(p, 0, schedule, seq_len, pad_id))
        seqs[orientation] = cur
    meta = {
        "pairs": len(pairs),
        "charged_pair_words": charged_words,
        "pair_token_stats": pair_token_stats,
        "seq_counts": {k: len(v) for k, v in seqs.items()},
    }
    return seqs, meta


def merge_stats(acc: Dict[str, float], add: Dict[str, float]) -> None:
    for k, v in add.items():
        acc[k] = acc.get(k, 0.0) + float(v)


def summarise_side(acc: Dict[str, float]) -> Dict[str, Any]:
    copied_n = acc.get("copied_targets", 0.0)
    noncopied_n = acc.get("noncopied_targets", 0.0)
    side_n = copied_n + noncopied_n
    copied_nll = acc.get("copied_nll", 0.0)
    noncopied_nll = acc.get("noncopied_nll", 0.0)
    return {
        "side_loss": (copied_nll + noncopied_nll) / side_n if side_n > 0 else None,
        "copied_loss": copied_nll / copied_n if copied_n > 0 else None,
        "noncopied_loss": noncopied_nll / noncopied_n if noncopied_n > 0 else None,
        "side_targets": side_n,
        "copied_targets": copied_n,
        "noncopied_targets": noncopied_n,
        "copied_fraction": copied_n / side_n if side_n > 0 else None,
    }


def eval_model(model_dir: pathlib.Path, seqs_by_orientation: Dict[str, List[dict]], batch_size: int, device: str) -> Dict[str, Any]:
    model = GPT2LMHeadModel.from_pretrained(model_dir).to(device)
    model.eval()
    out: Dict[str, Any] = {"model_dir": str(model_dir), "orientations": {}}
    with torch.no_grad():
        for orientation, seqs in seqs_by_orientation.items():
            acc: Dict[str, float] = {}
            t0 = time.time()
            for i in range(0, len(seqs), batch_size):
                batch = seqs[i : i + batch_size]
                tok_t = torch.tensor([s["tokens"] for s in batch], dtype=torch.long, device=device)
                reals = [s["real"] for s in batch]
                copieds = [s["copied"] for s in batch]
                side_masks = [s["side_mask"] for s in batch]
                logits = model(tok_t).logits
                _, stats = v3.loss_with_stats(logits, tok_t, reals, copieds, side_masks)
                merge_stats(acc, stats)
            out["orientations"][orientation] = {**summarise_side(acc), "elapsed_sec": time.time() - t0}
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return out


def get_metric(results: Dict[str, Any], arm: str, orientation: str, metric: str):
    return results.get(arm, {}).get("orientations", {}).get(orientation, {}).get(metric)


def diff(a, b):
    return None if a is None or b is None else a - b


def compute_interactions(results: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"available_final_arms": {a: a in results for a in FINAL_ARMS}}
    if not all(a in results for a in FINAL_ARMS):
        out["status"] = "partial_waiting_for_all_four_final_arms"
        return out
    out["status"] = "complete"
    metrics = ["side_loss", "copied_loss", "noncopied_loss"]
    for orientation in ["forward", "reverse"]:
        out[orientation] = {}
        for metric in metrics:
            vals = {a: get_metric(results, a, orientation, metric) for a in FINAL_ARMS}
            if all(v is not None for v in vals.values()):
                one = 0.5 * (vals["ff"] + vals["rr"])
                mix = 0.5 * (vals["fr"] + vals["rf"])
                out[orientation][metric] = {
                    "values": vals,
                    "oneway_avg_ff_rr": one,
                    "mixed_avg_fr_rf": mix,
                    "mixed_minus_oneway_loss": mix - one,
                    "lower_is_better": True,
                    "direction_matched_second_epoch_effects": {
                        "RF_minus_FF_for_forward_second_epoch": diff(vals["rf"], vals["ff"]),
                        "FR_minus_RR_for_reverse_second_epoch": diff(vals["fr"], vals["rr"]),
                    },
                    "retention_or_cross_exposure_effects": {
                        "FR_minus_RR": diff(vals["fr"], vals["rr"]),
                        "RF_minus_FF": diff(vals["rf"], vals["ff"]),
                    },
                }
    # Copied-vs-noncopied differential of the reciprocal interaction. Negative
    # means the mixed exposure improves noncopied loss more than copied loss.
    out["copied_vs_noncopied_interaction"] = {}
    for orientation in ["forward", "reverse"]:
        c = out.get(orientation, {}).get("copied_loss", {}).get("mixed_minus_oneway_loss")
        n = out.get(orientation, {}).get("noncopied_loss", {}).get("mixed_minus_oneway_loss")
        out["copied_vs_noncopied_interaction"][orientation] = None if c is None or n is None else n - c
    return out


def compute_learning_changes(results: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    comparisons = [
        ("prefix_forward", "ff", "forward", "second forward epoch after forward prefix"),
        ("prefix_forward", "fr", "reverse", "reverse epoch after forward prefix"),
        ("prefix_reverse", "rr", "reverse", "second reverse epoch after reverse prefix"),
        ("prefix_reverse", "rf", "forward", "forward epoch after reverse prefix"),
    ]
    for base, final, orientation, label in comparisons:
        if base not in results or final not in results:
            continue
        cur = {}
        for metric in ["side_loss", "copied_loss", "noncopied_loss"]:
            b = get_metric(results, base, orientation, metric)
            f = get_metric(results, final, orientation, metric)
            cur[metric] = {"prefix": b, "final": f, "final_minus_prefix_loss": diff(f, b), "lower_is_better": True}
        out[label] = cur
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root", required=True)
    ap.add_argument("--pair_jsonl", required=True)
    ap.add_argument("--tokenizer_dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--seq_len", type=int, default=256)
    args = ap.parse_args()

    run_root = pathlib.Path(args.run_root)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    seqs_by_orientation, data_meta = build_pair_eval_sequences(pathlib.Path(args.pair_jsonl), tokenizer, args.seq_len, pad_id)
    models = available_models(run_root)
    print(f"Available models: {sorted(models)}", flush=True)
    print(json.dumps(data_meta, indent=2), flush=True)
    results = {}
    t0 = time.time()
    for arm, model_dir in models.items():
        print(f"Evaluating {arm}: {model_dir}", flush=True)
        results[arm] = eval_model(model_dir, seqs_by_orientation, args.batch_size, args.device)
        print(json.dumps({"arm": arm, "orientations": results[arm]["orientations"]}, indent=2), flush=True)

    summary = {
        "status": "PAIR_OBJECTIVE_PROBE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_root": str(run_root),
        "pair_jsonl": args.pair_jsonl,
        "tokenizer_dir": args.tokenizer_dir,
        "device": args.device,
        "data_meta": data_meta,
        "available_models": {k: str(v) for k, v in models.items()},
        "results": results,
        "learning_changes": compute_learning_changes(results),
        "interactions": compute_interactions(results),
        "elapsed_sec": time.time() - t0,
        "interpretation_note": "Internal pair losses are mechanistic evidence for whether the source/side objective is learned and whether mixed directions affect copied vs noncopied targets. They do not replace official endpoint evaluation or copy-matched semantic/random controls.",
    }
    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    md = ["# research pair-objective probe", "", f"Run root: `{run_root}`", f"Models: {', '.join(sorted(models))}", "", "## Learning changes"]
    for label, rec in summary["learning_changes"].items():
        md.append(f"- {label}: {rec}")
    md += ["", "## Interactions", "", json.dumps(summary["interactions"], indent=2), "", f"JSON: `{out}`"]
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "output": str(out), "models": sorted(models), "interaction_status": summary["interactions"].get("status"), "elapsed_sec": summary["elapsed_sec"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
