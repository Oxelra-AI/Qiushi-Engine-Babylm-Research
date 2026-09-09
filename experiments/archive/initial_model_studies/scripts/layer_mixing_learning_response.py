#!/usr/bin/env python3
"""research — Layer-mixing learning-response: can a trained readout use the order signal?

Arms:
  1. final_head: frozen DeBERTa, final-layer MLM head scores candidates (known failure)
  2. layer_mix_frozen: frozen DeBERTa, trained layer-mixing readout scores candidates
  3. layer_mix_unfrozen: unfrozen DeBERTa + trained layer-mixing readout

Train/test separation: different entity words and location words for train vs held-out,
so success requires generalizing the ORDER mechanism, not memorizing specific answer tokens.

All arms share the same generated R1 counterfactual pairs and evaluation.
"""
from __future__ import annotations
import argparse, json, pathlib, sys, time, random
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
sys.path.insert(0, str(ROOT / "scripts"))
from scmlm_r_loss import make_masked_input
import importlib.util

spec = importlib.util.spec_from_file_location("r1gen", ROOT / "scripts/r1_generator_v3.py")
r1gen = importlib.util.module_from_spec(spec); sys.modules[spec.name] = r1gen; spec.loader.exec_module(r1gen)

CKPT = str(ROOT / "training/runs/r1_ordered_dynamic_5M/hf_model/chck_5M")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train/test vocabulary split (held-out entities & locations never seen in training)
TRAIN_LOCS = r1gen.LOCATIONS[:7]   # kitchen, garage, attic, cellar, study, porch, closet
TRAIN_ITEMS = r1gen.ITEMS[:14]     # lamp..basket
TEST_LOCS = r1gen.LOCATIONS[7:]    # shed, vault, loft, hallway, pantry
TEST_ITEMS = r1gen.ITEMS[14:]      # fan, radio, stool, hammer, broom, kettle


class LayerMixingReadout(nn.Module):
    """Trained layer-mixing + MLP answer scorer."""

    def __init__(self, hidden_size: int, n_layers: int, vocab_size: int):
        super().__init__()
        self.n_layers = n_layers
        self.layer_weights = nn.Parameter(torch.zeros(n_layers))
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(n_layers)])
        # Project mixed hidden to vocab logits (same interface as MLM head)
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, vocab_size),
        )

    def forward(self, hidden_states: list[torch.Tensor], mask_positions: torch.Tensor) -> torch.Tensor:
        """hidden_states: list of n_layers tensors each (1, seq, hidden).
        Returns logits at mask positions: (n_mask, vocab)."""
        alpha = F.softmax(self.layer_weights, dim=0)
        mixed = torch.zeros_like(hidden_states[0][0, mask_positions])  # (n_mask, hidden)
        for l in range(self.n_layers):
            h_l = hidden_states[l][0, mask_positions]  # (n_mask, hidden)
            mixed = mixed + alpha[l] * self.layer_norms[l](h_l)
        return self.proj(mixed)  # (n_mask, vocab)


def gen_pairs(split: str, n_pairs: int, seed: int, tok) -> list[dict]:
    """Generate equal-token-length counterfactual pairs for train or test split."""
    if split == "train":
        r1gen.LOCATIONS = list(TRAIN_LOCS); r1gen.ITEMS = list(TRAIN_ITEMS)
    else:
        r1gen.LOCATIONS = list(TEST_LOCS); r1gen.ITEMS = list(TEST_ITEMS)
    recs = []; cur = seed
    attempts = 0
    while len(recs) < n_pairs and attempts < 200:
        cur += 7919; attempts += 1
        for a, b in r1gen.generate_paired_corpus(n_pairs=max(n_pairs, 100), seed=cur):
            if a.answer == b.answer: continue
            ai = tok(a.answer, add_special_tokens=False)["input_ids"]
            bi = tok(b.answer, add_special_tokens=False)["input_ids"]
            if len(ai) != len(bi): continue
            recs.append({"passage_a": a.passage, "passage_b": b.passage,
                         "answer_a": a.answer, "answer_b": b.answer,
                         "a_ids": ai, "b_ids": bi})
            if len(recs) >= n_pairs: break
    return recs


def score_pair_final_head(model, tok, rec) -> dict:
    """Score using existing final MLM head (Arm 1 control)."""
    ids_a, attn_a, mask_a, a_ids = make_masked_input(rec["passage_a"], rec["answer_a"], tok, 256, DEVICE)
    ids_b, attn_b, mask_b, b_ids = make_masked_input(rec["passage_b"], rec["answer_b"], tok, 256, DEVICE)
    with torch.no_grad():
        logits_a = model(input_ids=ids_a, attention_mask=attn_a).logits[0]
        logits_b = model(input_ids=ids_b, attention_mask=attn_b).logits[0]
    lp_a = F.log_softmax(logits_a[mask_a], dim=-1)
    lp_b = F.log_softmax(logits_b[mask_b], dim=-1)
    a_t = torch.tensor(a_ids, device=DEVICE); b_t = torch.tensor(b_ids, device=DEVICE)
    rows = torch.arange(len(a_ids), device=DEVICE)
    s_Aa = lp_a[rows, a_t].sum().item(); s_Ab = lp_a[rows, b_t].sum().item()
    s_Bb = lp_b[rows, b_t].sum().item(); s_Ba = lp_b[rows, a_t].sum().item()
    m_A = s_Aa - s_Ab; m_B = s_Bb - s_Ba
    return {"m_A": m_A, "m_B": m_B, "I": m_A + m_B, "both_correct": float(m_A > 0 and m_B > 0)}


def train_layer_mixing(model, tok, readout: LayerMixingReadout, train_recs: list[dict],
                        test_recs: list[dict], n_epochs: int, lr: float,
                        unfreeze_encoder: bool, arm_name: str) -> dict:
    """Train layer-mixing readout on R1 counterfactual pairs."""
    if unfreeze_encoder:
        model.train()
        opt = torch.optim.AdamW(
            [{"params": readout.parameters(), "lr": lr},
             {"params": model.parameters(), "lr": lr * 0.1}],
            weight_decay=0.01
        )
    else:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        opt = torch.optim.AdamW(readout.parameters(), lr=lr, weight_decay=0.01)

    readout.train()
    curve = []
    t0 = time.time()

    for epoch in range(n_epochs):
        random.Random(epoch).shuffle(train_recs)
        epoch_losses = []
        for rec in train_recs:
            a_ids = rec["a_ids"]; b_ids = rec["b_ids"]
            ids_a, attn_a, mask_a, _ = make_masked_input(rec["passage_a"], rec["answer_a"], tok, 256, DEVICE)
            ids_b, attn_b, mask_b, _ = make_masked_input(rec["passage_b"], rec["answer_b"], tok, 256, DEVICE)
            out_a = model(input_ids=ids_a, attention_mask=attn_a, output_hidden_states=True)
            out_b = model(input_ids=ids_b, attention_mask=attn_b, output_hidden_states=True)
            logits_a = readout(list(out_a.hidden_states), mask_a)
            logits_b = readout(list(out_b.hidden_states), mask_b)
            lp_a = F.log_softmax(logits_a, dim=-1)
            lp_b = F.log_softmax(logits_b, dim=-1)
            a_t = torch.tensor(a_ids, device=DEVICE); b_t = torch.tensor(b_ids, device=DEVICE)
            rows = torch.arange(len(a_ids), device=DEVICE)
            s_Aa = lp_a[rows, a_t].sum(); s_Ab = lp_a[rows, b_t].sum()
            s_Bb = lp_b[rows, b_t].sum(); s_Ba = lp_b[rows, a_t].sum()
            m_A = s_Aa - s_Ab; m_B = s_Bb - s_Ba
            # Maximize I = m_A + m_B via hinge-style loss
            loss = F.softplus(-m_A) + F.softplus(-m_B) + F.softplus(0.5 - (m_A + m_B))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(readout.parameters()) +
                                           (list(model.parameters()) if unfreeze_encoder else []), 1.0)
            opt.step()
            epoch_losses.append(loss.item())

        if (epoch + 1) % 5 == 0 or epoch == 0:
            train_eval = evaluate_readout(model, tok, readout, train_recs[:50])
            test_eval = evaluate_readout(model, tok, readout, test_recs)
            entry = {"epoch": epoch + 1, "loss": sum(epoch_losses) / len(epoch_losses),
                     "train": train_eval, "test": test_eval}
            curve.append(entry)
            print(f"  {arm_name} ep{epoch+1}: loss={entry['loss']:.4f} "
                  f"train_bc={train_eval['both_correct']:.3f} test_bc={test_eval['both_correct']:.3f} "
                  f"train_I={train_eval['I_mean']:.4f} test_I={test_eval['I_mean']:.4f}")

    final_train = evaluate_readout(model, tok, readout, train_recs[:50])
    final_test = evaluate_readout(model, tok, readout, test_recs)
    elapsed = time.time() - t0

    # Learned layer weights
    with torch.no_grad():
        alpha = F.softmax(readout.layer_weights, dim=0).cpu().tolist()

    return {
        "arm": arm_name, "n_epochs": n_epochs, "lr": lr, "unfreeze_encoder": unfreeze_encoder,
        "n_train": len(train_recs), "n_test": len(test_recs),
        "final_train": final_train, "final_test": final_test,
        "curve": curve, "layer_weights": alpha, "elapsed_sec": round(elapsed, 1),
    }


def evaluate_readout(model, tok, readout: LayerMixingReadout, recs: list[dict]) -> dict:
    """Evaluate layer-mixing readout on pairs."""
    readout.eval(); model.eval()
    results = []
    with torch.no_grad():
        for rec in recs:
            a_ids = rec["a_ids"]; b_ids = rec["b_ids"]
            ids_a, attn_a, mask_a, _ = make_masked_input(rec["passage_a"], rec["answer_a"], tok, 256, DEVICE)
            ids_b, attn_b, mask_b, _ = make_masked_input(rec["passage_b"], rec["answer_b"], tok, 256, DEVICE)
            out_a = model(input_ids=ids_a, attention_mask=attn_a, output_hidden_states=True)
            out_b = model(input_ids=ids_b, attention_mask=attn_b, output_hidden_states=True)
            logits_a = readout(list(out_a.hidden_states), mask_a)
            logits_b = readout(list(out_b.hidden_states), mask_b)
            lp_a = F.log_softmax(logits_a, dim=-1)
            lp_b = F.log_softmax(logits_b, dim=-1)
            a_t = torch.tensor(a_ids, device=DEVICE); b_t = torch.tensor(b_ids, device=DEVICE)
            rows = torch.arange(len(a_ids), device=DEVICE)
            s_Aa = lp_a[rows, a_t].sum().item(); s_Ab = lp_a[rows, b_t].sum().item()
            s_Bb = lp_b[rows, b_t].sum().item(); s_Ba = lp_b[rows, a_t].sum().item()
            m_A = s_Aa - s_Ab; m_B = s_Bb - s_Ba
            results.append({"m_A": m_A, "m_B": m_B, "I": m_A + m_B,
                           "both_correct": float(m_A > 0 and m_B > 0)})
    readout.train()
    n = len(results)
    return {
        "both_correct": sum(r["both_correct"] for r in results) / n,
        "I_mean": sum(r["I"] for r in results) / n,
        "m_A_mean": sum(r["m_A"] for r in results) / n,
        "m_B_mean": sum(r["m_B"] for r in results) / n,
        "positive_I_frac": sum(1 for r in results if r["I"] > 0) / n,
        "n": n,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_train", type=int, default=200)
    p.add_argument("--n_test", type=int, default=100)
    p.add_argument("--n_epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=224)
    p.add_argument("--output", default=str(ROOT / "data/layer_mixing_learning_response.json"))
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(CKPT, use_fast=True)
    if tok.mask_token is None: tok.mask_token = "<mask>"
    model = AutoModelForMaskedLM.from_pretrained(CKPT, trust_remote_code=True).to(DEVICE)

    print("Generating R1 pairs...")
    train_recs = gen_pairs("train", args.n_train, args.seed, tok)
    test_recs = gen_pairs("test", args.n_test, args.seed + 1000, tok)
    print(f"  Train: {len(train_recs)} pairs, Test: {len(test_recs)} pairs")

    hidden_size = model.config.hidden_size
    n_layers = model.config.num_hidden_layers + 1  # embedding + layers
    vocab_size = model.config.vocab_size

    payload = {"status": "LAYER_MIXING_LEARNING_RESPONSE",
               "checkpoint": CKPT, "n_train": len(train_recs), "n_test": len(test_recs),
               "train_vocab": {"locations": TRAIN_LOCS, "items": TRAIN_ITEMS},
               "test_vocab": {"locations": TEST_LOCS, "items": TEST_ITEMS},
               "arms": {}}
    out_path = pathlib.Path(args.output); out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Arm 1: Final-head control ──
    print("\n=== Arm 1: Final MLM head (frozen, control) ===")
    model.eval()
    arm1_train = [score_pair_final_head(model, tok, r) for r in train_recs[:50]]
    arm1_test = [score_pair_final_head(model, tok, r) for r in test_recs]
    payload["arms"]["final_head"] = {
        "train": {"both_correct": sum(r["both_correct"] for r in arm1_train) / len(arm1_train),
                  "I_mean": sum(r["I"] for r in arm1_train) / len(arm1_train),
                  "m_A_mean": sum(r["m_A"] for r in arm1_train) / len(arm1_train),
                  "m_B_mean": sum(r["m_B"] for r in arm1_train) / len(arm1_train)},
        "test": {"both_correct": sum(r["both_correct"] for r in arm1_test) / len(arm1_test),
                 "I_mean": sum(r["I"] for r in arm1_test) / len(arm1_test),
                 "m_A_mean": sum(r["m_A"] for r in arm1_test) / len(arm1_test),
                 "m_B_mean": sum(r["m_B"] for r in arm1_test) / len(arm1_test)},
    }
    print(f"  Arm1 train bc={payload['arms']['final_head']['train']['both_correct']:.3f} "
          f"I={payload['arms']['final_head']['train']['I_mean']:.6f}")
    print(f"  Arm1 test  bc={payload['arms']['final_head']['test']['both_correct']:.3f} "
          f"I={payload['arms']['final_head']['test']['I_mean']:.6f}")

    # ── Arm 2: Layer-mixing, encoder frozen ──
    print("\n=== Arm 2: Layer-mixing readout (encoder frozen) ===")
    readout2 = LayerMixingReadout(hidden_size, n_layers, vocab_size).to(DEVICE)
    arm2 = train_layer_mixing(model, tok, readout2, train_recs, test_recs,
                               args.n_epochs, args.lr, unfreeze_encoder=False, arm_name="layer_mix_frozen")
    payload["arms"]["layer_mix_frozen"] = arm2
    out_path.write_text(json.dumps(payload, indent=2) + "\n")

    # ── Arm 3: Layer-mixing, encoder unfrozen ──
    print("\n=== Arm 3: Layer-mixing readout (encoder unfrozen) ===")
    # Reload model fresh so Arm 3 starts from the same checkpoint
    del model; torch.cuda.empty_cache()
    model = AutoModelForMaskedLM.from_pretrained(CKPT, trust_remote_code=True).to(DEVICE)
    readout3 = LayerMixingReadout(hidden_size, n_layers, vocab_size).to(DEVICE)
    arm3 = train_layer_mixing(model, tok, readout3, train_recs, test_recs,
                               args.n_epochs, args.lr, unfreeze_encoder=True, arm_name="layer_mix_unfrozen")
    payload["arms"]["layer_mix_unfrozen"] = arm3
    out_path.write_text(json.dumps(payload, indent=2) + "\n")

    # ── Summary ──
    print("\n=== SUMMARY ===")
    for arm_name, arm_data in payload["arms"].items():
        if "final_test" in arm_data:
            t = arm_data["final_test"]
        else:
            t = arm_data["test"]
        print(f"  {arm_name:25s}: test both_correct={t['both_correct']:.3f} I={t.get('I_mean', t.get('I_mean', 0)):.4f}")

    # Interpretation
    test_bc_2 = payload["arms"]["layer_mix_frozen"]["final_test"]["both_correct"]
    test_bc_3 = payload["arms"]["layer_mix_unfrozen"]["final_test"]["both_correct"]
    if test_bc_2 > 0.15:
        interp = ("BIDIRECTIONAL_REPAIR_VIABLE: frozen encoder + trained layer-mixing readout "
                  "achieves held-out both_correct > 0.15. The order signal in shallow layers IS "
                  "usable by a trained readout. Bidirectional intermediate repair should be developed.")
    elif test_bc_3 > 0.15:
        interp = ("REPRESENTATION_SHAPING_NEEDED: trained layer-mixing with unfrozen encoder "
                  "achieves held-out both_correct > 0.15, but frozen encoder fails. The order "
                  "signal must be actively shaped by training. Intermediate supervision route viable.")
    else:
        interp = ("CAUSAL_INTERFACE_NEEDED: neither frozen nor unfrozen layer-mixing achieves "
                  "held-out both_correct > 0.15. The bidirectional encoder does not provide "
                  "usable order information even with trained readout. Causal/recursive interface "
                  "becomes the primary route.")
    payload["interpretation"] = interp
    payload["decisive_metric"] = "test both_correct (held-out vocab)"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nInterpretation: {interp}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
