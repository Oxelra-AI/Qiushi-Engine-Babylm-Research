#!/usr/bin/env python3
"""research R1 frozen MLM-head counterfactual evaluator.

For each held-out pair (A,B) with the same/similar word bag but different correct
answers, score the model's MLM head on:
  passage A with answer masked: log p(answer_A) vs log p(answer_B)
  passage B with answer masked: log p(answer_B) vs log p(answer_A)
No trained probe/readout is used.
"""
from __future__ import annotations
import argparse, importlib.util, json, pathlib, re, sys
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
GEN = ROOT / "scripts/r1_generator_v3.py"
spec = importlib.util.spec_from_file_location("r1gen", GEN)
r1gen = importlib.util.module_from_spec(spec); sys.modules[spec.name] = r1gen; spec.loader.exec_module(r1gen)

# Force held-out vocabulary for evaluation, disjoint from research training vocab.
HELDOUT_LOCS = r1gen.LOCATIONS[7:]
HELDOUT_ITEMS = r1gen.ITEMS[14:]


def set_heldout_vocab():
    r1gen.LOCATIONS = list(HELDOUT_LOCS)
    r1gen.ITEMS = list(HELDOUT_ITEMS)


def make_masked(text: str, answer: str, tokenizer) -> tuple[str, int]:
    """Replace the answer phrase in the query sentence with the same number of mask tokens."""
    answer_toks = tokenizer(answer, add_special_tokens=False)["input_ids"]
    n = len(answer_toks)
    mask_phrase = " ".join([tokenizer.mask_token] * n)
    # Replace final occurrence to avoid masking initial/operation mention.
    idx = text.rfind(answer)
    if idx < 0:
        raise RuntimeError(f"answer phrase not found: {answer!r} in {text}")
    return text[:idx] + mask_phrase + text[idx + len(answer):], n


def answer_logprob(model, tokenizer, masked_text: str, answer: str, device: torch.device) -> float:
    enc = tokenizer(masked_text, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=256)
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    mask_positions = (input_ids[0] == tokenizer.mask_token_id).nonzero(as_tuple=False).view(-1)
    ans_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    if len(mask_positions) != len(ans_ids):
        raise RuntimeError(f"mask/answer length mismatch {len(mask_positions)} vs {len(ans_ids)} for {answer}")
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attn).logits[0]
        logp = torch.log_softmax(logits[mask_positions], dim=-1)
        val = 0.0
        for i, tid in enumerate(ans_ids):
            val += float(logp[i, tid].detach().cpu())
    return val


def generate_records(n_pairs: int, seed: int, tokenizer, max_attempt_batches: int = 50):
    """Generate held-out records and keep only answer pairs with equal token length.

    Equal answer-token length lets true and counterfactual candidates be scored in
    the same masked context, avoiding a length/readout confound.
    """
    set_heldout_vocab()
    recs = []
    cur_seed = seed
    batches = 0
    while len(recs) < n_pairs and batches < max_attempt_batches:
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
                "query_obj": a.query_obj,
                "word_count_a": a.word_count, "word_count_b": b.word_count,
            })
            if len(recs) >= n_pairs:
                break
    if len(recs) < n_pairs:
        raise RuntimeError(f"only generated {len(recs)} equal-token-length pairs out of requested {n_pairs}")
    return recs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--output_json", required=True)
    p.add_argument("--n_pairs", type=int, default=200)
    p.add_argument("--seed", type=int, default=2000)
    p.add_argument("--batch_note", default="")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tok.mask_token is None:
        tok.mask_token = "<mask>"
    model = AutoModelForMaskedLM.from_pretrained(args.model_path, trust_remote_code=True).to(device).eval()
    records = generate_records(args.n_pairs, args.seed, tok)
    rows = []
    correct = 0
    margins = []
    for i, r in enumerate(records):
        mt_a, _ = make_masked(r["passage_a"], r["answer_a"], tok)
        mt_b, _ = make_masked(r["passage_b"], r["answer_b"], tok)
        a_true = answer_logprob(model, tok, mt_a, r["answer_a"], device)
        a_cf = answer_logprob(model, tok, mt_a, r["answer_b"], device)
        b_true = answer_logprob(model, tok, mt_b, r["answer_b"], device)
        b_cf = answer_logprob(model, tok, mt_b, r["answer_a"], device)
        ma = a_true - a_cf
        mb = b_true - b_cf
        ca = ma > 0
        cb = mb > 0
        correct += int(ca) + int(cb)
        margins.extend([ma, mb])
        if i < 20:
            rows.append({**r, "a_true": a_true, "a_cf": a_cf, "b_true": b_true, "b_cf": b_cf, "margin_a": ma, "margin_b": mb, "correct_a": ca, "correct_b": cb})
    n_cases = 2 * len(records)
    payload = {
        "status": "R1_MLM_COUNTERFACTUAL",
        "model_path": args.model_path,
        "batch_note": args.batch_note,
        "heldout_locations": HELDOUT_LOCS,
        "heldout_items": HELDOUT_ITEMS,
        "n_pairs": len(records),
        "n_cases": n_cases,
        "paired_accuracy": correct / max(1, n_cases),
        "mean_margin_true_minus_counterfactual": sum(margins) / max(1, len(margins)),
        "median_margin": sorted(margins)[len(margins)//2] if margins else None,
        "positive_margin_fraction": sum(1 for m in margins if m > 0) / max(1, len(margins)),
        "sample_rows": rows,
    }
    out = pathlib.Path(args.output_json); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ["model_path", "n_pairs", "paired_accuracy", "mean_margin_true_minus_counterfactual", "positive_margin_fraction"]}, indent=2))

if __name__ == "__main__":
    main()
