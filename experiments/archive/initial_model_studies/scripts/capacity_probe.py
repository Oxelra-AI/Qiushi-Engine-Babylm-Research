#!/usr/bin/env python3
"""research — Capacity probe: can the ordinary MLM head break the mirror AT ALL?

Decisive disambiguation before choosing architecture-pivot vs SOTA-consolidation.

The research finding: baseline held-out mean_m_A = -0.4999, mean_m_B = +0.4999 —
a near-perfect mirror. The model has a FIXED candidate preference insensitive to
context. Every training arm preserved this mirror (both-contexts-correct = 0).

Question: Is this an ARCHITECTURAL limit (the DeBERTa MLM head structurally cannot
make the query-position logits depend on operation order), or a
LEARNING/GENERALIZATION limit (it could but doesn't from this signal)?

Test: Directly overfit SCMLM-R with strong settings on a SMALL fixed set of pairs
(memorization). Evaluate on the SAME pairs (train) and on FRESH pairs (generalize).
  - If it cannot even push both-contexts-correct up on the TRAINED pairs ->
    architectural limit (the head cannot represent the contrast) -> branch A hard.
  - If it CAN memorize trained pairs but fails fresh -> generalization/data limit;
    the mechanism exists but the signal/scale is insufficient.
  - If it memorizes AND partially generalizes with strong settings -> the earlier
    200-step run was under-optimized; worth a stronger/longer SCMLM-R run.
"""
import sys, pathlib, json, random, torch
sys.path.insert(0, str(pathlib.Path("experiments/archive/initial_model_studies/scripts")))
from scmlm_r_loss import scmlm_r_loss, make_masked_input, span_logprob
import importlib.util
ROOT = pathlib.Path("experiments/archive/initial_model_studies")
spec = importlib.util.spec_from_file_location("r1gen", ROOT / "scripts/r1_generator_v3.py")
r1gen = importlib.util.module_from_spec(spec); sys.modules[spec.name] = r1gen; spec.loader.exec_module(r1gen)
from transformers import AutoModelForMaskedLM, AutoTokenizer

CKPT = str(ROOT / "training/runs/r1_ordered_dynamic_5M/hf_model/chck_5M")
TRAIN_LOCS = r1gen.LOCATIONS[:7]; TRAIN_ITEMS = r1gen.ITEMS[:14]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gen_pairs(n, seed, tokenizer):
    r1gen.LOCATIONS = list(TRAIN_LOCS); r1gen.ITEMS = list(TRAIN_ITEMS)
    recs = []; cur = seed
    while len(recs) < n:
        cur += 7919
        for a, b in r1gen.generate_paired_corpus(n_pairs=max(n, 100), seed=cur):
            if a.answer == b.answer: continue
            ai = tokenizer(a.answer, add_special_tokens=False)["input_ids"]
            bi = tokenizer(b.answer, add_special_tokens=False)["input_ids"]
            if len(ai) != len(bi): continue
            recs.append({"passage_a": a.passage, "passage_b": b.passage,
                         "answer_a": a.answer, "answer_b": b.answer})
            if len(recs) >= n: break
    return recs


def eval_pairs(model, tokenizer, recs):
    model.eval(); both = 0; ps = []
    with torch.no_grad():
        for r in recs:
            ai = tokenizer(r["answer_a"], add_special_tokens=False)["input_ids"]
            bi = tokenizer(r["answer_b"], add_special_tokens=False)["input_ids"]
            ida, aa, ma, _ = make_masked_input(r["passage_a"], r["answer_a"], tokenizer, 256, DEVICE)
            sAa = span_logprob(model, ida, aa, ma, ai).item(); sAb = span_logprob(model, ida, aa, ma, bi).item()
            idb, ab, mb, _ = make_masked_input(r["passage_b"], r["answer_b"], tokenizer, 256, DEVICE)
            sBb = span_logprob(model, idb, ab, mb, bi).item(); sBa = span_logprob(model, idb, ab, mb, ai).item()
            mA = sAa - sAb; mB = sBb - sBa
            both += int(mA > 0 and mB > 0); ps.append(mA + mB)
    model.train()
    return {"both_correct": both / max(1, len(recs)),
            "mean_pair_sum": sum(ps) / max(1, len(ps)),
            "positive_frac": sum(1 for x in ps if x > 0) / max(1, len(ps))}


def main():
    tok = AutoTokenizer.from_pretrained(CKPT, use_fast=True)
    if tok.mask_token is None: tok.mask_token = "<mask>"
    model = AutoModelForMaskedLM.from_pretrained(CKPT, trust_remote_code=True).to(DEVICE)

    train_recs = gen_pairs(32, 111, tok)   # small fixed set to memorize
    fresh_recs = gen_pairs(100, 222, tok)  # fresh pairs, same train vocab

    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    # Strong settings: high interaction weight, low tau for sharp margin
    cfg = dict(tau=0.5, gamma=1.0, lambda_dir=1.0, lambda_inter=2.0, lambda_multi=0.0)

    pre_train = eval_pairs(model, tok, train_recs)
    pre_fresh = eval_pairs(model, tok, fresh_recs)
    print(f"PRE  train both={pre_train['both_correct']:.3f} fresh both={pre_fresh['both_correct']:.3f}")

    curve = []
    for epoch in range(60):  # 60 passes over 32 pairs = 1920 updates
        random.Random(epoch).shuffle(train_recs)
        for r in train_recs:
            opt.zero_grad()
            res = scmlm_r_loss(model, tok, r["passage_a"], r["passage_b"],
                               r["answer_a"], r["answer_b"], other_candidates=None,
                               device=DEVICE, **cfg)
            res["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if (epoch + 1) % 10 == 0:
            tr = eval_pairs(model, tok, train_recs)
            fr = eval_pairs(model, tok, fresh_recs)
            curve.append({"epoch": epoch + 1, "train": tr, "fresh": fr})
            print(f"epoch {epoch+1}: TRAIN both={tr['both_correct']:.3f} pair_sum={tr['mean_pair_sum']:.3f} | "
                  f"FRESH both={fr['both_correct']:.3f} pair_sum={fr['mean_pair_sum']:.3f}")

    final_train = eval_pairs(model, tok, train_recs)
    final_fresh = eval_pairs(model, tok, fresh_recs)

    if final_train["both_correct"] < 0.2:
        verdict = "ARCHITECTURAL_LIMIT: MLM head cannot even memorize the context-order contrast. Branch A (architecture) is required; reshaping loss on MLM head is futile."
    elif final_fresh["both_correct"] < 0.15:
        verdict = "GENERALIZATION_LIMIT: head CAN memorize but does not generalize the state variable. Mechanism exists locally but signal/scale insufficient; representation still the bottleneck for transfer."
    else:
        verdict = "UNDER_OPTIMIZED_EARLIER: strong settings break the mirror and partially generalize. A stronger/longer SCMLM-R run is worth testing before pivoting."

    out = {
        "status": "CAPACITY_PROBE",
        "checkpoint": CKPT, "config": cfg,
        "n_train_pairs": len(train_recs), "n_fresh_pairs": len(fresh_recs),
        "pre": {"train": pre_train, "fresh": pre_fresh},
        "curve": curve,
        "final": {"train": final_train, "fresh": final_fresh},
        "verdict": verdict,
    }
    p = ROOT / "data/capacity_probe.json"
    p.write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(f"\nVERDICT: {verdict}\nSaved: {p}")


if __name__ == "__main__":
    main()
