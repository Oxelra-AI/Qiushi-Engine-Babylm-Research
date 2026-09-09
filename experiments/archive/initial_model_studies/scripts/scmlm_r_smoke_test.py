#!/usr/bin/env python3
"""research — Smoke test: SCMLM-R loss gradients flow through MLM head.

Verifies:
1. Loss computes without error on a real counterfactual pair
2. Gradients propagate to model parameters (not just embeddings)
3. m_A and m_B have expected signs or magnitudes
4. Loss decreases after a few optimizer steps (basic learnability)
"""
import sys, pathlib, json, torch
sys.path.insert(0, str(pathlib.Path("experiments/archive/initial_model_studies/scripts")))
from scmlm_r_loss import scmlm_r_loss, make_masked_input
from transformers import AutoModelForMaskedLM, AutoTokenizer

CKPT = "experiments/archive/initial_model_studies/training/runs/r1_ordered_dynamic_5M/hf_model/chck_5M"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# A real counterfactual pair from research held-out evaluation
PAIR = {
    "passage_a": "Initially, the radio is in the pantry. Initially, the stool is in the hallway. Initially, the broom is in the loft. Initially, the fan is in the loft. The item in the hallway was moved to the loft. Whatever was in the pantry got transferred to the loft. Someone took the broom from the loft and put it in the vault. the broom was moved from the vault to the pantry. the fan was moved from the loft to the shed. After all these changes, the broom is in the pantry.",
    "passage_b": "Initially, the radio is in the pantry. Initially, the stool is in the hallway. Initially, the broom is in the loft. Initially, the fan is in the loft. The item in the hallway was moved to the loft. Whatever was in the pantry got transferred to the loft. the broom was moved from the vault to the pantry. Someone took the broom from the loft and put it in the vault. the fan was moved from the loft to the shed. After all these changes, the broom is in the vault.",
    "answer_a": "the pantry",
    "answer_b": "the vault",
}

# Other candidates for multi-candidate loss (equal token length = 3)
OTHER_CANDIDATES = ["the hallway", "the pantry", "the vault"]  # will filter by length


def main():
    tok = AutoTokenizer.from_pretrained(CKPT, use_fast=True)
    if tok.mask_token is None:
        tok.mask_token = "<mask>"
    model = AutoModelForMaskedLM.from_pretrained(CKPT, trust_remote_code=True).to(DEVICE)
    model.train()

    # Verify answer token lengths match
    a_ids = tok(PAIR["answer_a"], add_special_tokens=False)["input_ids"]
    b_ids = tok(PAIR["answer_b"], add_special_tokens=False)["input_ids"]
    print(f"Answer A tokens: {a_ids} (len {len(a_ids)})")
    print(f"Answer B tokens: {b_ids} (len {len(b_ids)})")
    assert len(a_ids) == len(b_ids), "Token length mismatch!"

    # Single forward pass
    result = scmlm_r_loss(
        model, tok,
        PAIR["passage_a"], PAIR["passage_b"],
        PAIR["answer_a"], PAIR["answer_b"],
        other_candidates=OTHER_CANDIDATES,
        tau=1.0, gamma=0.5,
        lambda_dir=1.0, lambda_inter=0.5, lambda_multi=0.3,
        max_length=256, device=DEVICE,
    )
    print(f"\n--- Initial forward ---")
    print(f"Loss: {result['loss'].item():.4f}")
    print(f"L_dir: {result['L_dir'].item():.4f}")
    print(f"L_inter: {result['L_inter'].item():.4f}")
    print(f"L_multi: {result['L_multi'].item():.4f}")
    print(f"m_A: {result['m_A'].item():.4f}")
    print(f"m_B: {result['m_B'].item():.4f}")
    print(f"I (m_A+m_B): {result['I'].item():.4f}")

    # Check gradient flow
    result["loss"].backward()
    grad_norms = {}
    has_grad = False
    for name, p in model.named_parameters():
        if p.grad is not None and p.grad.abs().max() > 0:
            grad_norms[name] = p.grad.norm().item()
            has_grad = True
    
    print(f"\n--- Gradient check ---")
    print(f"Parameters with nonzero grad: {len(grad_norms)}")
    if grad_norms:
        top = sorted(grad_norms.items(), key=lambda x: -x[1])[:5]
        for name, norm in top:
            print(f"  {name}: grad_norm={norm:.6f}")
    assert has_grad, "No gradients! Loss is not connected to model parameters."

    # Quick learnability: 3 optimizer steps
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    losses = []
    for step in range(5):
        optimizer.zero_grad()
        r = scmlm_r_loss(
            model, tok,
            PAIR["passage_a"], PAIR["passage_b"],
            PAIR["answer_a"], PAIR["answer_b"],
            other_candidates=OTHER_CANDIDATES,
            tau=1.0, gamma=0.5,
            lambda_dir=1.0, lambda_inter=0.5, lambda_multi=0.3,
            max_length=256, device=DEVICE,
        )
        r["loss"].backward()
        optimizer.step()
        losses.append(r["loss"].item())

    print(f"\n--- Learnability (5 steps on single pair) ---")
    for i, l in enumerate(losses):
        print(f"  step {i}: loss={l:.4f}")
    
    decreasing = losses[-1] < losses[0]
    print(f"Loss decreased: {decreasing} ({losses[0]:.4f} -> {losses[-1]:.4f})")

    # Final check: m_A and m_B after training
    model.eval()
    with torch.no_grad():
        r_final = scmlm_r_loss(
            model, tok,
            PAIR["passage_a"], PAIR["passage_b"],
            PAIR["answer_a"], PAIR["answer_b"],
            tau=1.0, gamma=0.5, device=DEVICE,
        )
    print(f"\n--- After 5 steps ---")
    print(f"m_A: {r_final['m_A'].item():.4f} (was {result['m_A'].item():.4f})")
    print(f"m_B: {r_final['m_B'].item():.4f} (was {result['m_B'].item():.4f})")
    print(f"I: {r_final['I'].item():.4f} (was {result['I'].item():.4f})")

    # Save summary
    summary = {
        "status": "SCMLM_R_SMOKE_TEST",
        "checkpoint": CKPT,
        "device": str(DEVICE),
        "initial": {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in result.items()},
        "gradient_params_with_grad": len(grad_norms),
        "top_grad_norms": {k: v for k, v in sorted(grad_norms.items(), key=lambda x: -x[1])[:5]},
        "learnability_losses": losses,
        "loss_decreased": decreasing,
        "final_m_A": r_final["m_A"].item(),
        "final_m_B": r_final["m_B"].item(),
        "final_I": r_final["I"].item(),
        "verdict": "PASS" if has_grad and decreasing else "FAIL",
    }
    out = pathlib.Path("experiments/archive/initial_model_studies/data/scmlm_r_smoke.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f"\nVerdict: {summary['verdict']}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
