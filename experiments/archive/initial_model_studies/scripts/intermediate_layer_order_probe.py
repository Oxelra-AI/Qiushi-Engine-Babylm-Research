#!/usr/bin/env python3
"""research — Intermediate-layer order probe.

For the R1 ordered-dynamic chck_5M checkpoint, measures whether operation-order
information exists at intermediate layers but collapses before the final MLM score.

For each layer l ∈ {embedding, 0, 1, ..., 7}:
  - cosine(h_A^l, h_B^l) at masked query position
  - ||h_A^l - h_B^l||_2
  - projection of (h_A^l - h_B^l) onto the answer-discrimination direction

Also computes paired gradients: d(score_true - score_wrong)/d(h^l) for each layer.

Uses the same counterfactual pair generation (seed 333) as research mirror diagnosis.
"""
from __future__ import annotations
import sys, pathlib, json, torch
import torch.nn.functional as F
sys.path.insert(0, str(pathlib.Path("experiments/archive/initial_model_studies/scripts")))
from scmlm_r_loss import make_masked_input, span_logprob
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


def get_all_hidden_at_mask(model, tok, passage, answer, device):
    """Return list of hidden states at mask positions for each layer (embedding + 8 layers)."""
    ids, attn, mask_pos, _ = make_masked_input(passage, answer, tok, 256, device)
    out = model(input_ids=ids, attention_mask=attn, output_hidden_states=True)
    # out.hidden_states: tuple of (embedding, layer0, ..., layer7) each (1, seq, hidden)
    hiddens = []
    for h in out.hidden_states:
        # Mean over mask positions for this layer
        hiddens.append(h[0, mask_pos].mean(0))  # (hidden,)
    return hiddens  # list of 9 tensors (embedding + 8 layers)


def main():
    tok = AutoTokenizer.from_pretrained(CKPT, use_fast=True)
    if tok.mask_token is None: tok.mask_token = "<mask>"
    model = AutoModelForMaskedLM.from_pretrained(CKPT, trust_remote_code=True).to(DEVICE)
    model.eval()

    # Same seed as research mirror diagnosis
    recs = gen_pairs(60, 333, tok)
    n_layers = None  # will be set on first pair

    # Per-layer metrics accumulators
    cosines_by_layer = []
    l2_by_layer = []
    # Also track final-layer answer discrimination
    answer_score_diffs = []

    print(f"Running intermediate-layer probe on {len(recs)} pairs...")

    with torch.no_grad():
        for i, r in enumerate(recs):
            h_a = get_all_hidden_at_mask(model, tok, r["passage_a"], r["answer_a"], DEVICE)
            h_b = get_all_hidden_at_mask(model, tok, r["passage_b"], r["answer_b"], DEVICE)

            if n_layers is None:
                n_layers = len(h_a)
                cosines_by_layer = [[] for _ in range(n_layers)]
                l2_by_layer = [[] for _ in range(n_layers)]

            for l in range(n_layers):
                cos = F.cosine_similarity(h_a[l].unsqueeze(0), h_b[l].unsqueeze(0)).item()
                l2 = (h_a[l] - h_b[l]).norm().item()
                cosines_by_layer[l].append(cos)
                l2_by_layer[l].append(l2)

    # Paired gradient analysis: for a subset, compute grad of (score_true - score_wrong) w.r.t. each layer
    print("Running paired gradient analysis on 20 pairs...")
    grad_norms_by_layer = [[] for _ in range(n_layers)] if n_layers else []
    model.eval()  # keep eval mode but enable grad
    for r in recs[:20]:
        a_ids_tok = tok(r["answer_a"], add_special_tokens=False)["input_ids"]
        b_ids_tok = tok(r["answer_b"], add_special_tokens=False)["input_ids"]

        # Context A: score true (answer_a) minus wrong (answer_b)
        ids_a, attn_a, mask_a, _ = make_masked_input(r["passage_a"], r["answer_a"], tok, 256, DEVICE)

        # Forward with hooks to capture intermediate hidden states
        hidden_states = []
        hooks = []
        # Register hooks on each layer output
        # DeBERTa-v2: model.deberta.encoder.layer[i] for layers, model.deberta.embeddings for embedding
        base = model.deberta if hasattr(model, 'deberta') else model.base_model
        
        # Use output_hidden_states and retain_grad
        model.zero_grad()
        out = model(input_ids=ids_a, attention_mask=attn_a, output_hidden_states=True)
        
        # Compute score difference
        logits = out.logits[0]  # (seq, vocab)
        log_probs = F.log_softmax(logits[mask_a], dim=-1)
        score_true = log_probs[:, a_ids_tok].sum()
        score_wrong = log_probs[:, b_ids_tok].sum()
        diff = score_true - score_wrong
        
        # Retain grad on hidden states and backward
        for h in out.hidden_states:
            h.retain_grad()
        diff.backward()
        
        for l, h in enumerate(out.hidden_states):
            if h.grad is not None:
                # Grad at mask positions
                g = h.grad[0, mask_a].mean(0)
                grad_norms_by_layer[l].append(g.norm().item())
            else:
                grad_norms_by_layer[l].append(0.0)
        
        model.zero_grad()

    # Summarize
    layer_names = ["embedding"] + [f"layer_{i}" for i in range(n_layers - 1)]
    results = {
        "status": "INTERMEDIATE_LAYER_ORDER_PROBE",
        "checkpoint": CKPT,
        "n_pairs": len(recs),
        "n_layers": n_layers,
        "layer_names": layer_names,
        "per_layer": {}
    }
    
    for l in range(n_layers):
        cos_vals = cosines_by_layer[l]
        l2_vals = l2_by_layer[l]
        grad_vals = grad_norms_by_layer[l] if l < len(grad_norms_by_layer) else []
        results["per_layer"][layer_names[l]] = {
            "cosine_mean": sum(cos_vals) / len(cos_vals),
            "cosine_min": min(cos_vals),
            "cosine_max": max(cos_vals),
            "l2_diff_mean": sum(l2_vals) / len(l2_vals),
            "l2_diff_min": min(l2_vals),
            "l2_diff_max": max(l2_vals),
            "grad_norm_mean": sum(grad_vals) / max(1, len(grad_vals)),
            "grad_norm_max": max(grad_vals) if grad_vals else 0,
        }
        print(f"  {layer_names[l]:12s}: cos={sum(cos_vals)/len(cos_vals):.8f}  "
              f"L2={sum(l2_vals)/len(l2_vals):.4f}  "
              f"grad={sum(grad_vals)/max(1,len(grad_vals)):.6f}")

    # Interpretation
    cos_embed = results["per_layer"]["embedding"]["cosine_mean"]
    cos_final = results["per_layer"][layer_names[-1]]["cosine_mean"]
    cos_mid = results["per_layer"][layer_names[n_layers // 2]]["cosine_mean"] if n_layers > 2 else cos_final
    
    if cos_embed > 0.999 and cos_final > 0.999:
        interpretation = ("ORDER_NEVER_LEARNED: all layers including embeddings have near-identical "
                         "representations for both operation orders. The bidirectional encoder never "
                         "separated the two contexts at any depth. Supports Route C (causal interface).")
    elif cos_mid < 0.99 and cos_final > 0.999:
        interpretation = ("ORDER_LEARNED_THEN_COLLAPSED: intermediate layers separate the operation "
                         "orders but the final layer collapses them. A bidirectional repair may exist: "
                         "intermediate residual readout, layer mixing, or auxiliary supervision at the "
                         "order-sensitive layer.")
    elif cos_final < 0.99:
        interpretation = ("ORDER_PRESENT_AT_FINAL: the final layer does encode order differences. "
                         "The research mirror may have been a property of the specific evaluation "
                         "method or pair type, not a representational collapse.")
    else:
        interpretation = ("PARTIAL: order signal exists at some layers but interpretation is ambiguous. "
                         "Check L2 norms and gradient analysis for additional signal.")
    
    results["interpretation"] = interpretation
    
    out_path = ROOT / "data/intermediate_layer_order_probe.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nInterpretation: {interpretation}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
