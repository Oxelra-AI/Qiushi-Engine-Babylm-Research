#!/usr/bin/env python3
"""research — Diagnose WHY the mirror is structural.

Hypothesis: in same-word-bag counterfactual pairs, passages A and B are
near-identical token sequences differing only in operation ORDER. When the answer
span is masked, the query-position hidden state may be nearly IDENTICAL in A and B,
because the distinguishing operations are distant and the bidirectional encoder
does not localize order at the query position. If so:
  s_A(x) ~= s_B(x) for every candidate x
  => m_A = s_A(a)-s_A(b) ~= -(s_B(b)-s_B(a)) = -m_B  (structural mirror)
This forces both_correct=0 REGARDLESS of head capacity.

Tests:
1. Token-level diff between passage A and B (how different are they at all?).
2. Query-position hidden-state cosine similarity between A and B (from the frozen
   base checkpoint). If ~1.0, the contexts are collapsed at the query position.
3. Whether s_A(a)-s_B(a) (SAME candidate, different context) is ~0. If ~0, the
   model literally cannot tell the two contexts apart at the scoring position.

This decides: is state unreachable because of HEAD capacity (would justify pivot),
or because the ENCODER does not represent order at the query position (a different,
possibly addressable problem, or a firm reason the ordinary bidirectional MLM
scoring interface cannot carry this signal)?
"""
import sys, pathlib, json, torch, torch.nn.functional as F
sys.path.insert(0, str(pathlib.Path("experiments/archive/initial_model_studies/scripts")))
from scmlm_r_loss import make_masked_input
import importlib.util
ROOT = pathlib.Path("experiments/archive/initial_model_studies")
spec = importlib.util.spec_from_file_location("r1gen", ROOT / "scripts/r1_generator_v3.py")
r1gen = importlib.util.module_from_spec(spec); sys.modules[spec.name] = r1gen; spec.loader.exec_module(r1gen)
from transformers import AutoModelForMaskedLM, AutoTokenizer

CKPT = str(ROOT / "training/runs/r1_ordered_dynamic_5M/hf_model/chck_5M")
TRAIN_LOCS = r1gen.LOCATIONS[:7]; TRAIN_ITEMS = r1gen.ITEMS[:14]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gen_pairs(n, seed, tok):
    r1gen.LOCATIONS = list(TRAIN_LOCS); r1gen.ITEMS = list(TRAIN_ITEMS)
    recs = []; cur = seed
    while len(recs) < n:
        cur += 7919
        for a, b in r1gen.generate_paired_corpus(n_pairs=max(n, 100), seed=cur):
            if a.answer == b.answer: continue
            ai = tok(a.answer, add_special_tokens=False)["input_ids"]
            bi = tok(b.answer, add_special_tokens=False)["input_ids"]
            if len(ai) != len(bi): continue
            recs.append({"passage_a": a.passage, "passage_b": b.passage,
                         "answer_a": a.answer, "answer_b": b.answer})
            if len(recs) >= n: break
    return recs


def query_hidden(model, tok, passage, answer):
    ids, attn, mask_pos, _ = make_masked_input(passage, answer, tok, 256, DEVICE)
    out = model(input_ids=ids, attention_mask=attn, output_hidden_states=True)
    h = out.hidden_states[-1][0]  # (seq, hid)
    return h[mask_pos].mean(0)  # mean over mask positions


def main():
    tok = AutoTokenizer.from_pretrained(CKPT, use_fast=True)
    if tok.mask_token is None: tok.mask_token = "<mask>"
    model = AutoModelForMaskedLM.from_pretrained(CKPT, trust_remote_code=True).to(DEVICE).eval()

    recs = gen_pairs(60, 333, tok)
    token_diffs = []
    cos_sims = []
    same_cand_score_diffs = []  # s_A(a) - s_B(a): does context change the score for SAME candidate?
    from scmlm_r_loss import span_logprob

    with torch.no_grad():
        for r in recs:
            # 1. token diff (both masked at their own answer)
            ida = tok(r["passage_a"], add_special_tokens=False)["input_ids"]
            idb = tok(r["passage_b"], add_special_tokens=False)["input_ids"]
            L = min(len(ida), len(idb))
            ndiff = sum(1 for i in range(L) if ida[i] != idb[i]) + abs(len(ida) - len(idb))
            token_diffs.append(ndiff)

            # 2. query-position hidden cosine between A and B (mask each at its OWN answer)
            ha = query_hidden(model, tok, r["passage_a"], r["answer_a"])
            hb = query_hidden(model, tok, r["passage_b"], r["answer_b"])
            cos_sims.append(F.cosine_similarity(ha, hb, dim=0).item())

            # 3. same candidate 'a' scored in context A vs context B
            ai = tok(r["answer_a"], add_special_tokens=False)["input_ids"]
            idA, atA, mA, _ = make_masked_input(r["passage_a"], r["answer_a"], tok, 256, DEVICE)
            # in B, mask B's answer span (equal length), score candidate a there
            idB, atB, mB, _ = make_masked_input(r["passage_b"], r["answer_b"], tok, 256, DEVICE)
            sAa = span_logprob(model, idA, atA, mA, ai).item()
            sBa = span_logprob(model, idB, atB, mB, ai).item()
            same_cand_score_diffs.append(sAa - sBa)

    n = len(recs)
    out = {
        "status": "MIRROR_DIAGNOSIS",
        "checkpoint": CKPT,
        "n_pairs": n,
        "token_diff": {
            "mean": sum(token_diffs)/n, "min": min(token_diffs), "max": max(token_diffs),
            "note": "How many token positions differ between counterfactual A and B (differ only by operation order)."
        },
        "query_hidden_cosine_A_vs_B": {
            "mean": sum(cos_sims)/n, "min": min(cos_sims), "max": max(cos_sims),
            "note": "Cosine of final-layer hidden at the masked query position, context A vs B. ~1.0 => contexts collapsed at query position."
        },
        "same_candidate_score_diff_sAa_minus_sBa": {
            "mean": sum(same_cand_score_diffs)/n,
            "mean_abs": sum(abs(x) for x in same_cand_score_diffs)/n,
            "max_abs": max(abs(x) for x in same_cand_score_diffs),
            "note": "Log-prob of the SAME candidate answer_a scored in context A vs context B. ~0 => the scoring position cannot distinguish the two operation orders."
        },
    }
    # Interpretation
    cos_mean = out["query_hidden_cosine_A_vs_B"]["mean"]
    scd = out["same_candidate_score_diff_sAa_minus_sBa"]["mean_abs"]
    if cos_mean > 0.98 and scd < 0.2:
        out["diagnosis"] = ("QUERY_POSITION_COLLAPSE: the encoder produces nearly identical query-position "
                            "representations for both operation orders, so the MLM scoring head receives no order "
                            "signal. The mirror is structural, not a head-capacity issue. Final-answer MLM scoring "
                            "cannot carry this state distinction; only a representation/objective that forces the "
                            "query position to encode order could work. Supports pivoting away from final-answer ranking.")
    elif scd >= 0.2:
        out["diagnosis"] = ("QUERY_POSITION_DOES_ENCODE_ORDER: the scoring position DOES change with context, so the "
                            "prior SCMLM-R failure was optimization/loss-design, not representational. A trajectory- or "
                            "prefix-supervised objective might still succeed; the pivot should not fully close Route A.")
    else:
        out["diagnosis"] = "PARTIAL: contexts partly distinguished; ambiguous; needs a stronger probe."
    p = ROOT / "data/mirror_diagnosis.json"
    p.write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(json.dumps(out, indent=2, default=str))
    print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()
