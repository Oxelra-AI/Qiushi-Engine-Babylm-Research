#!/usr/bin/env python3
"""research — SCMLM-R four-arm evaluation with shortcut-hardened controls.

Evaluates all four arm checkpoints on:
  - train_vocab: counterfactual pairs from training vocabulary
  - heldout_vocab: counterfactual pairs from held-out vocabulary
Plus causal-sensitivity controls.

Decisive metric: both_contexts_correct_pair_fraction (both m_A>0 and m_B>0).
This is what ordinary WWM could NOT achieve (stayed at 0 in research).

Primary comparison: does scmlm_r break the answer-prior mirror on held-out vocab
where wwm_only, answer_mlm, random_neg do not?
"""
from __future__ import annotations
import argparse, importlib.util, json, pathlib, sys, time
import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
sys.path.insert(0, str(ROOT / "scripts"))
from scmlm_r_loss import make_masked_input, span_logprob

spec = importlib.util.spec_from_file_location("r1gen", ROOT / "scripts/r1_generator_v3.py")
r1gen = importlib.util.module_from_spec(spec); sys.modules[spec.name] = r1gen; spec.loader.exec_module(r1gen)

TRAIN_LOCS = ["the kitchen", "the garage", "the attic", "the cellar", "the study", "the porch", "the closet"]
TRAIN_ITEMS = ["the lamp", "the clock", "the vase", "the mirror", "the rug", "the painting", "the cushion", "the blanket", "the candle", "the plant", "the photo", "the trophy", "the statue", "the basket"]
HELDOUT_LOCS = ["the shed", "the vault", "the loft", "the hallway", "the pantry"]
HELDOUT_ITEMS = ["the fan", "the radio", "the stool", "the hammer", "the broom", "the kettle"]

ARMS = {
    "wwm_only": ROOT / "training/runs/wwm_only",
    "answer_mlm": ROOT / "training/runs/answer_mlm",
    "random_neg": ROOT / "training/runs/random_neg",
    "scmlm_r": ROOT / "training/runs/scmlm_r",
}
BASELINE = ROOT / "training/runs/r1_ordered_dynamic_5M/hf_model/chck_5M"


def set_vocab(split):
    if split == "train_vocab":
        r1gen.LOCATIONS = list(TRAIN_LOCS); r1gen.ITEMS = list(TRAIN_ITEMS)
    else:
        r1gen.LOCATIONS = list(HELDOUT_LOCS); r1gen.ITEMS = list(HELDOUT_ITEMS)


def gen_equal_len_pairs(split, tokenizer, n_pairs, seed):
    set_vocab(split)
    recs = []
    cur_seed = seed
    batches = 0
    while len(recs) < n_pairs and batches < 100:
        batches += 1
        pairs = r1gen.generate_paired_corpus(n_pairs=max(n_pairs, 100), seed=cur_seed)
        cur_seed += 7919
        for a, b in pairs:
            if a.answer == b.answer:
                continue
            a_ids = tokenizer(a.answer, add_special_tokens=False)["input_ids"]
            b_ids = tokenizer(b.answer, add_special_tokens=False)["input_ids"]
            if len(a_ids) != len(b_ids):
                continue
            recs.append({
                "passage_a": a.passage, "passage_b": b.passage,
                "answer_a": a.answer, "answer_b": b.answer,
                "answer_token_len": len(a_ids),
            })
            if len(recs) >= n_pairs:
                break
    return recs


def score_model(model_path, tokenizer, records, device):
    model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True).to(device).eval()
    both_correct = 0
    correct_cases = 0
    pair_sums = []
    m_A_list, m_B_list = [], []
    t0 = time.time()
    with torch.no_grad():
        for r in records:
            a_ids = tokenizer(r["answer_a"], add_special_tokens=False)["input_ids"]
            b_ids = tokenizer(r["answer_b"], add_special_tokens=False)["input_ids"]
            ids_a, attn_a, mask_a, _ = make_masked_input(r["passage_a"], r["answer_a"], tokenizer, 256, device)
            s_A_a = span_logprob(model, ids_a, attn_a, mask_a, a_ids).item()
            s_A_b = span_logprob(model, ids_a, attn_a, mask_a, b_ids).item()
            ids_b, attn_b, mask_b, _ = make_masked_input(r["passage_b"], r["answer_b"], tokenizer, 256, device)
            s_B_b = span_logprob(model, ids_b, attn_b, mask_b, b_ids).item()
            s_B_a = span_logprob(model, ids_b, attn_b, mask_b, a_ids).item()
            m_A = s_A_a - s_A_b
            m_B = s_B_b - s_B_a
            m_A_list.append(m_A); m_B_list.append(m_B)
            correct_cases += int(m_A > 0) + int(m_B > 0)
            both_correct += int(m_A > 0 and m_B > 0)
            pair_sums.append(m_A + m_B)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    n = len(records)
    return {
        "model_path": str(model_path),
        "n_pairs": n,
        "both_contexts_correct_pair_fraction": both_correct / max(1, n),
        "case_accuracy": correct_cases / max(1, 2 * n),
        "mean_pair_sum_margin": sum(pair_sums) / max(1, n),
        "positive_pair_sum_fraction": sum(1 for x in pair_sums if x > 0) / max(1, n),
        "mean_m_A": sum(m_A_list) / max(1, n),
        "mean_m_B": sum(m_B_list) / max(1, n),
        "elapsed_sec": round(time.time() - t0, 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_pairs", type=int, default=200)
    p.add_argument("--seed", type=int, default=2000)
    p.add_argument("--steps", nargs="+", type=int, default=[50, 100, 150, 200])
    p.add_argument("--output_json", default=str(ROOT / "data/scmlm_r_fourarm_eval.json"))
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(BASELINE, use_fast=True)
    if tokenizer.mask_token is None:
        tokenizer.mask_token = "<mask>"

    # Generate held-out eval sets ONCE (shared across all models)
    eval_sets = {
        "train_vocab": gen_equal_len_pairs("train_vocab", tokenizer, args.n_pairs, args.seed),
        "heldout_vocab": gen_equal_len_pairs("heldout_vocab", tokenizer, args.n_pairs, args.seed),
    }
    print(f"Eval sets: train_vocab={len(eval_sets['train_vocab'])}, heldout_vocab={len(eval_sets['heldout_vocab'])}")

    payload = {
        "status": "SCMLM_R_FOURARM_EVAL",
        "n_pairs": args.n_pairs,
        "seed": args.seed,
        "decisive_metric": "both_contexts_correct_pair_fraction on heldout_vocab",
        "baseline_chck_5M": {},
        "arms": {},
    }
    out = pathlib.Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Baseline (starting checkpoint)
    for split in ["train_vocab", "heldout_vocab"]:
        payload["baseline_chck_5M"][split] = score_model(BASELINE, tokenizer, eval_sets[split], device)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Baseline chck_5M heldout both-correct: {payload['baseline_chck_5M']['heldout_vocab']['both_contexts_correct_pair_fraction']:.3f}")

    # Each arm at each step
    for arm, arm_dir in ARMS.items():
        payload["arms"][arm] = {}
        for step in args.steps:
            ck = arm_dir / f"ckpt_step{step}"
            if not ck.exists():
                print(f"  {arm} step{step}: MISSING")
                continue
            payload["arms"][arm][f"update_{step}"] = {}
            for split in ["train_vocab", "heldout_vocab"]:
                payload["arms"][arm][f"update_{step}"][split] = score_model(ck, tokenizer, eval_sets[split], device)
            out.write_text(json.dumps(payload, indent=2) + "\n")
            hv = payload["arms"][arm][f"update_{step}"]["heldout_vocab"]
            print(f"  {arm} step{step}: heldout both-correct={hv['both_contexts_correct_pair_fraction']:.3f} pair_sum={hv['mean_pair_sum_margin']:.4f}")

    # Summary: final step comparison on heldout vocab
    final_step = max(args.steps)
    summary = {}
    for arm in ARMS:
        key = f"step{final_step}"
        if key in payload["arms"].get(arm, {}):
            hv = payload["arms"][arm][key]["heldout_vocab"]
            summary[arm] = {
                "both_correct": hv["both_contexts_correct_pair_fraction"],
                "pair_sum_margin": hv["mean_pair_sum_margin"],
            }
    payload["final_step_heldout_summary"] = summary
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print("\n=== FINAL HELDOUT SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
