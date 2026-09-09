#!/usr/bin/env python3
"""research — Corrected intermediate-layer / paired-gradient order probe.

Repairs the diagnostic's paired-objective and scoring confounds:
  1. Computes BOTH context A and context B in a joint paired objective.
  2. Uses position-aligned span scoring (token i at mask position i), not matrix indexing.
  3. Measures answer-direction alignment: does the paired objective gradient reinforce
     the A-vs-B hidden difference at each layer?
  4. Separates two claims:
       - hidden A/B masked-position differences decay with depth (representation fact)
       - the final-answer objective has/has not a local gradient lever to train order
         information (gradient/alignment fact)

No training is performed. Uses the same R1 counterfactual family as Steps 216/222.
"""
from __future__ import annotations
import json, pathlib, sys, importlib.util
from statistics import mean
import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
sys.path.insert(0, str(ROOT / "scripts"))
from scmlm_r_loss import make_masked_input

spec = importlib.util.spec_from_file_location("r1gen", ROOT / "scripts/r1_generator_v3.py")
r1gen = importlib.util.module_from_spec(spec); sys.modules[spec.name] = r1gen; spec.loader.exec_module(r1gen)

CKPT = str(ROOT / "training/runs/r1_ordered_dynamic_5M/hf_model/chck_5M")
TRAIN_LOCS = r1gen.LOCATIONS[:7]
TRAIN_ITEMS = r1gen.ITEMS[:14]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gen_pairs(n: int, seed: int, tok):
    r1gen.LOCATIONS = list(TRAIN_LOCS); r1gen.ITEMS = list(TRAIN_ITEMS)
    recs = []; cur = seed
    while len(recs) < n:
        cur += 7919
        for a, b in r1gen.generate_paired_corpus(n_pairs=max(n, 100), seed=cur):
            if a.answer == b.answer:
                continue
            ai = tok(a.answer, add_special_tokens=False)["input_ids"]
            bi = tok(b.answer, add_special_tokens=False)["input_ids"]
            if len(ai) != len(bi):
                continue
            recs.append({
                "passage_a": a.passage,
                "passage_b": b.passage,
                "answer_a": a.answer,
                "answer_b": b.answer,
                "n_answer_tokens": len(ai),
            })
            if len(recs) >= n:
                break
    return recs


def span_score_from_logits(logits_seq: torch.Tensor, mask_pos: torch.Tensor, answer_ids: list[int]) -> torch.Tensor:
    """Position-aligned span log-prob.

    logits_seq: (seq, vocab); mask_pos: (m,); answer_ids: length m.
    Returns sum_i log p(answer_ids[i] at mask_pos[i]).
    """
    if len(mask_pos) != len(answer_ids):
        raise ValueError(f"mask/answer length mismatch: {len(mask_pos)} vs {len(answer_ids)}")
    lp = F.log_softmax(logits_seq[mask_pos], dim=-1)  # (m, vocab)
    ans = torch.tensor(answer_ids, device=logits_seq.device, dtype=torch.long)
    rows = torch.arange(len(answer_ids), device=logits_seq.device)
    return lp[rows, ans].sum()


def mlm_head_logits(model, hidden: torch.Tensor) -> torch.Tensor:
    """Apply the model's MLM prediction head to arbitrary hidden states."""
    if hasattr(model, "cls"):
        return model.cls(hidden)
    if hasattr(model, "lm_predictions"):
        return model.lm_predictions(hidden)
    if hasattr(model, "lm_head"):
        return model.lm_head(hidden)
    raise AttributeError("Could not locate MLM head")


def layer_hidden_metrics(out_a, out_b, mask_a, mask_b, layer_idx: int) -> dict:
    h_a_pos = out_a.hidden_states[layer_idx][0, mask_a]  # (m, h)
    h_b_pos = out_b.hidden_states[layer_idx][0, mask_b]
    mean_a = h_a_pos.mean(0)
    mean_b = h_b_pos.mean(0)
    pos_cos = F.cosine_similarity(h_a_pos, h_b_pos, dim=-1)  # (m,)
    pos_l2 = (h_a_pos - h_b_pos).norm(dim=-1)
    return {
        "mean_cosine": F.cosine_similarity(mean_a.unsqueeze(0), mean_b.unsqueeze(0)).item(),
        "mean_l2": (mean_a - mean_b).norm().item(),
        "positionwise_cosine_mean": pos_cos.mean().item(),
        "positionwise_l2_mean": pos_l2.mean().item(),
    }


def direct_head_layer_scores(model, out_a, out_b, mask_a, mask_b, a_ids, b_ids, layer_idx: int) -> dict:
    """Score candidates by applying the trained MLM head directly to layer hidden states.

    This is a diagnostic, not a trained intermediate readout.
    """
    logits_a = mlm_head_logits(model, out_a.hidden_states[layer_idx])[0]
    logits_b = mlm_head_logits(model, out_b.hidden_states[layer_idx])[0]
    s_A_a = span_score_from_logits(logits_a, mask_a, a_ids)
    s_A_b = span_score_from_logits(logits_a, mask_a, b_ids)
    s_B_b = span_score_from_logits(logits_b, mask_b, b_ids)
    s_B_a = span_score_from_logits(logits_b, mask_b, a_ids)
    m_A = s_A_a - s_A_b
    m_B = s_B_b - s_B_a
    return {
        "m_A": m_A.item(),
        "m_B": m_B.item(),
        "I_pair_sum": (m_A + m_B).item(),
        "both_correct": float((m_A.item() > 0) and (m_B.item() > 0)),
    }


def safe_cos(x: torch.Tensor, y: torch.Tensor) -> float:
    nx = x.norm().item(); ny = y.norm().item()
    if nx == 0 or ny == 0:
        return 0.0
    return F.cosine_similarity(x.unsqueeze(0), y.unsqueeze(0)).item()


def main():
    tok = AutoTokenizer.from_pretrained(CKPT, use_fast=True)
    if tok.mask_token is None:
        tok.mask_token = "<mask>"
    model = AutoModelForMaskedLM.from_pretrained(CKPT, trust_remote_code=True).to(DEVICE)
    model.eval()

    recs = gen_pairs(60, 333, tok)
    print(f"Generated {len(recs)} equal-token counterfactual pairs")

    # Part 1: representation decay and direct intermediate-head scores.
    n_layers = None
    layer_names = None
    reps = None
    head_scores = None
    final_exact = []

    with torch.no_grad():
        for r in recs:
            a_ids = tok(r["answer_a"], add_special_tokens=False)["input_ids"]
            b_ids = tok(r["answer_b"], add_special_tokens=False)["input_ids"]
            ids_a, attn_a, mask_a, _ = make_masked_input(r["passage_a"], r["answer_a"], tok, 256, DEVICE)
            ids_b, attn_b, mask_b, _ = make_masked_input(r["passage_b"], r["answer_b"], tok, 256, DEVICE)
            out_a = model(input_ids=ids_a, attention_mask=attn_a, output_hidden_states=True)
            out_b = model(input_ids=ids_b, attention_mask=attn_b, output_hidden_states=True)
            if n_layers is None:
                n_layers = len(out_a.hidden_states)
                layer_names = ["embedding"] + [f"layer_{i}" for i in range(n_layers - 1)]
                reps = {name: [] for name in layer_names}
                head_scores = {name: [] for name in layer_names}
            for li, lname in enumerate(layer_names):
                reps[lname].append(layer_hidden_metrics(out_a, out_b, mask_a, mask_b, li))
                head_scores[lname].append(direct_head_layer_scores(model, out_a, out_b, mask_a, mask_b, a_ids, b_ids, li))
            # final exact scores from model logits (same as direct head final, but explicit)
            s_A_a = span_score_from_logits(out_a.logits[0], mask_a, a_ids)
            s_A_b = span_score_from_logits(out_a.logits[0], mask_a, b_ids)
            s_B_b = span_score_from_logits(out_b.logits[0], mask_b, b_ids)
            s_B_a = span_score_from_logits(out_b.logits[0], mask_b, a_ids)
            final_exact.append({
                "m_A": (s_A_a - s_A_b).item(),
                "m_B": (s_B_b - s_B_a).item(),
                "I_pair_sum": (s_A_a - s_A_b + s_B_b - s_B_a).item(),
                "both_correct": float(((s_A_a - s_A_b).item() > 0) and ((s_B_b - s_B_a).item() > 0)),
            })

    # Part 2: corrected paired gradients for joint objective I = m_A + m_B.
    grad_records = {name: [] for name in layer_names}
    for r in recs[:20]:
        a_ids = tok(r["answer_a"], add_special_tokens=False)["input_ids"]
        b_ids = tok(r["answer_b"], add_special_tokens=False)["input_ids"]
        ids_a, attn_a, mask_a, _ = make_masked_input(r["passage_a"], r["answer_a"], tok, 256, DEVICE)
        ids_b, attn_b, mask_b, _ = make_masked_input(r["passage_b"], r["answer_b"], tok, 256, DEVICE)
        model.zero_grad(set_to_none=True)
        out_a = model(input_ids=ids_a, attention_mask=attn_a, output_hidden_states=True)
        out_b = model(input_ids=ids_b, attention_mask=attn_b, output_hidden_states=True)
        for h in list(out_a.hidden_states) + list(out_b.hidden_states):
            h.retain_grad()
        s_A_a = span_score_from_logits(out_a.logits[0], mask_a, a_ids)
        s_A_b = span_score_from_logits(out_a.logits[0], mask_a, b_ids)
        s_B_b = span_score_from_logits(out_b.logits[0], mask_b, b_ids)
        s_B_a = span_score_from_logits(out_b.logits[0], mask_b, a_ids)
        I = (s_A_a - s_A_b) + (s_B_b - s_B_a)
        I.backward()
        for li, lname in enumerate(layer_names):
            hA = out_a.hidden_states[li][0, mask_a].mean(0).detach()
            hB = out_b.hidden_states[li][0, mask_b].mean(0).detach()
            delta = hA - hB
            gA = out_a.hidden_states[li].grad[0, mask_a].mean(0).detach()
            gB = out_b.hidden_states[li].grad[0, mask_b].mean(0).detach()
            sep_grad = gA - gB  # local ascent direction for increasing A/B hidden separation under I
            grad_records[lname].append({
                "grad_A_norm": gA.norm().item(),
                "grad_B_norm": gB.norm().item(),
                "sep_grad_norm": sep_grad.norm().item(),
                "grad_A_B_cosine": safe_cos(gA, gB),
                "delta_sepgrad_cosine": safe_cos(delta, sep_grad),
                "delta_dot_sepgrad": torch.dot(delta, sep_grad).item(),
                "delta_norm": delta.norm().item(),
                "I_pair_sum": I.item(),
            })
        model.zero_grad(set_to_none=True)

    def summarize_list_dict(rows: list[dict], keys: list[str]) -> dict:
        out = {}
        for k in keys:
            vals = [float(r[k]) for r in rows]
            out[k + "_mean"] = mean(vals) if vals else 0.0
            out[k + "_min"] = min(vals) if vals else 0.0
            out[k + "_max"] = max(vals) if vals else 0.0
        return out

    per_layer = {}
    for lname in layer_names:
        per_layer[lname] = {
            "representation": summarize_list_dict(
                reps[lname], ["mean_cosine", "mean_l2", "positionwise_cosine_mean", "positionwise_l2_mean"]
            ),
            "direct_intermediate_head": summarize_list_dict(
                head_scores[lname], ["m_A", "m_B", "I_pair_sum", "both_correct"]
            ),
            "paired_gradient_I": summarize_list_dict(
                grad_records[lname], [
                    "grad_A_norm", "grad_B_norm", "sep_grad_norm", "grad_A_B_cosine",
                    "delta_sepgrad_cosine", "delta_dot_sepgrad", "delta_norm", "I_pair_sum"
                ]
            ),
        }
        print(f"{lname:10s} rep_L2={per_layer[lname]['representation']['mean_l2_mean']:.6f} "
              f"rep_cos={per_layer[lname]['representation']['mean_cosine_mean']:.9f} "
              f"I_head={per_layer[lname]['direct_intermediate_head']['I_pair_sum_mean']:.6f} "
              f"both={per_layer[lname]['direct_intermediate_head']['both_correct_mean']:.3f} "
              f"sepgrad={per_layer[lname]['paired_gradient_I']['sep_grad_norm_mean']:.3e} "
              f"align={per_layer[lname]['paired_gradient_I']['delta_sepgrad_cosine_mean']:.3f}")

    final_summary = summarize_list_dict(final_exact, ["m_A", "m_B", "I_pair_sum", "both_correct"])

    # Conservative interpretation: separate representation fact from training-lever fact.
    rep_decay = per_layer["layer_0"]["representation"]["mean_l2_mean"] > 1000 * per_layer[layer_names[-1]]["representation"]["mean_l2_mean"]
    final_both = final_summary["both_correct_mean"]
    early_sep_grad = max(per_layer[l]["paired_gradient_I"]["sep_grad_norm_mean"] for l in layer_names[:4])
    final_sep_grad = per_layer[layer_names[-1]]["paired_gradient_I"]["sep_grad_norm_mean"]
    if rep_decay and final_both == 0 and early_sep_grad < 1e-5:
        interpretation = (
            "CORRECTED_FINDING: masked-query A/B hidden differences decay strongly with depth, "
            "final position-aligned paired scores remain mirrored, and the joint paired objective's "
            "local gradient at early signal-bearing layers is extremely small. This supports, but does "
            "not by itself prove, the hypothesis that final-answer losses have weak leverage for training "
            "order use in this checkpoint. It does not justify a universal DeBERTa/WWM impossibility claim."
        )
    else:
        interpretation = (
            "MIXED_FINDING: corrected gradients or intermediate-head scores show nontrivial leverage. "
            "A bidirectional repair may remain viable and should be studied before choosing a causal route."
        )

    payload = {
        "status": "CORRECTED_ORDER_GRADIENT_PROBE",
        "checkpoint": CKPT,
        "device": str(DEVICE),
        "n_pairs_representation": len(recs),
        "n_pairs_gradient": min(20, len(recs)),
        "layer_names": layer_names,
        "scoring_correction": "Position-aligned span log-prob: sum_i log p(answer_token_i at mask_position_i). No matrix indexing across all mask positions.",
        "paired_gradient_objective": "I=(s_A(a)-s_A(b))+(s_B(b)-s_B(a)); gradients computed jointly for A and B.",
        "final_exact_scores": final_summary,
        "per_layer": per_layer,
        "interpretation": interpretation,
        "caveat": "This is a local no-training diagnostic on the R1 chck_5M checkpoint. It distinguishes representation decay and local gradient leverage; it does not prove WWM can never learn order under other data/objectives/checkpoints.",
    }
    out = ROOT / "data/corrected_order_gradient_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print("\nFinal exact:", json.dumps(final_summary, indent=2))
    print("Interpretation:", interpretation)
    print("Saved:", out)


if __name__ == "__main__":
    main()
