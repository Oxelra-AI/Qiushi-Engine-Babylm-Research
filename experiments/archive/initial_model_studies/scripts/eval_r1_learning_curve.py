#!/usr/bin/env python3
"""research R1 learning-curve evaluator.

Scores the same held-out equal-answer-token-length counterfactual pair set for:
  - protected chck_10M reference (chance baseline)
  - ordered-dynamic R1 checkpoints chck_1M..chck_5M
  - static-indirect-control checkpoints chck_1M..chck_5M

Readout: frozen MLM-head log-likelihood margin true answer minus paired
counterfactual answer. No trained probe/readout.
"""
from __future__ import annotations
import argparse, importlib.util, json, pathlib, sys, time
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
EVAL = ROOT / "scripts/eval_r1_mlm_counterfactual.py"
spec = importlib.util.spec_from_file_location("r1eval", EVAL)
r1eval = importlib.util.module_from_spec(spec); sys.modules[spec.name] = r1eval; spec.loader.exec_module(r1eval)

PROTECTED = ROOT / "training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/hf_model/chck_10M"
ORDERED_ROOT = ROOT / "training/runs/r1_ordered_dynamic_5M/hf_model"
STATIC_ROOT = ROOT / "training/runs/r1_static_indirect_control_5M/hf_model"
OUT = ROOT / "data/r1_mlm_counterfactual_learning_curve.json"


def score_model(model_path: pathlib.Path, tokenizer, records: list[dict], device: torch.device) -> dict:
    model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True).to(device).eval()
    rows = []
    margins = []
    correct = 0
    t0 = time.time()
    for i, r in enumerate(records):
        mt_a, _ = r1eval.make_masked(r["passage_a"], r["answer_a"], tokenizer)
        mt_b, _ = r1eval.make_masked(r["passage_b"], r["answer_b"], tokenizer)
        a_true = r1eval.answer_logprob(model, tokenizer, mt_a, r["answer_a"], device)
        a_cf = r1eval.answer_logprob(model, tokenizer, mt_a, r["answer_b"], device)
        b_true = r1eval.answer_logprob(model, tokenizer, mt_b, r["answer_b"], device)
        b_cf = r1eval.answer_logprob(model, tokenizer, mt_b, r["answer_a"], device)
        ma = a_true - a_cf
        mb = b_true - b_cf
        margins.extend([ma, mb])
        correct += int(ma > 0) + int(mb > 0)
        if i < 12:
            rows.append({
                "pair_index": i,
                "answer_a": r["answer_a"], "answer_b": r["answer_b"],
                "answer_token_len": r["answer_token_len"],
                "margin_a": ma, "margin_b": mb,
                "correct_a": ma > 0, "correct_b": mb > 0,
                "a_true": a_true, "a_cf": a_cf, "b_true": b_true, "b_cf": b_cf,
            })
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    margins_sorted = sorted(margins)
    n_cases = len(margins)
    return {
        "model_path": str(model_path),
        "n_pairs": len(records),
        "n_cases": n_cases,
        "paired_accuracy": correct / max(1, n_cases),
        "mean_margin_true_minus_counterfactual": sum(margins) / max(1, n_cases),
        "median_margin": margins_sorted[n_cases // 2] if margins_sorted else None,
        "positive_margin_fraction": sum(1 for m in margins if m > 0) / max(1, n_cases),
        "elapsed_sec": round(time.time() - t0, 1),
        "sample_rows": rows,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_pairs", type=int, default=200)
    p.add_argument("--seed", type=int, default=2000)
    p.add_argument("--output_json", default=str(OUT))
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(PROTECTED, use_fast=True)
    if tokenizer.mask_token is None:
        tokenizer.mask_token = "<mask>"
    records = r1eval.generate_records(args.n_pairs, args.seed, tokenizer)
    payload = {
        "status": "R1_MLM_COUNTERFACTUAL_LEARNING_CURVE",
        "readout": "Frozen MLM-head true-answer log-likelihood minus paired counterfactual answer on same held-out equal-token-length counterfactual pairs; no trained probe.",
        "device": str(device),
        "n_pairs_requested": args.n_pairs,
        "seed": args.seed,
        "heldout_locations": r1eval.HELDOUT_LOCS,
        "heldout_items": r1eval.HELDOUT_ITEMS,
        "records_sample": records[:5],
        "protected_reference": None,
        "ordered_dynamic": {},
        "static_indirect_control": {},
        "deltas_ordered_minus_static": {},
    }
    out = pathlib.Path(args.output_json); out.parent.mkdir(parents=True, exist_ok=True)

    payload["protected_reference"] = score_model(PROTECTED, tokenizer, records, device)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    checkpoints = [f"chck_{i}M" for i in range(1, 6)]
    for ck in checkpoints:
        op = ORDERED_ROOT / ck
        sp = STATIC_ROOT / ck
        if not op.exists() or not sp.exists():
            raise FileNotFoundError(f"missing checkpoint {ck}: {op.exists()} {sp.exists()}")
        payload["ordered_dynamic"][ck] = score_model(op, tokenizer, records, device)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        payload["static_indirect_control"][ck] = score_model(sp, tokenizer, records, device)
        od = payload["ordered_dynamic"][ck]
        st = payload["static_indirect_control"][ck]
        payload["deltas_ordered_minus_static"][ck] = {
            "paired_accuracy": od["paired_accuracy"] - st["paired_accuracy"],
            "mean_margin_true_minus_counterfactual": od["mean_margin_true_minus_counterfactual"] - st["mean_margin_true_minus_counterfactual"],
            "positive_margin_fraction": od["positive_margin_fraction"] - st["positive_margin_fraction"],
        }
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"checkpoint": ck, "ordered": od["mean_margin_true_minus_counterfactual"], "static": st["mean_margin_true_minus_counterfactual"], "delta": payload["deltas_ordered_minus_static"][ck]}, indent=2), flush=True)
    payload["final_interpretation_fields"] = {
        "best_ordered_margin": max(v["mean_margin_true_minus_counterfactual"] for v in payload["ordered_dynamic"].values()),
        "best_static_margin": max(v["mean_margin_true_minus_counterfactual"] for v in payload["static_indirect_control"].values()),
        "best_delta_margin": max(v["mean_margin_true_minus_counterfactual"] for v in payload["deltas_ordered_minus_static"].values()),
        "final_delta_margin": payload["deltas_ordered_minus_static"]["chck_5M"]["mean_margin_true_minus_counterfactual"],
        "final_delta_accuracy": payload["deltas_ordered_minus_static"]["chck_5M"]["paired_accuracy"],
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "summary": payload["final_interpretation_fields"]}, indent=2))

if __name__ == "__main__":
    main()
