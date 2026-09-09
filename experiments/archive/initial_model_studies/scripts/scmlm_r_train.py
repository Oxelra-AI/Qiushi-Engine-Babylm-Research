#!/usr/bin/env python3
"""research — SCMLM-R four-arm matched continuation training.

Forks from the research ordered-dynamic 5M checkpoint and runs short equal-budget
continuations to test whether the state-counterfactual ranking loss breaks the
answer-prior mirror failure that ordinary WWM could not.

Arms (selected by --arm):
  wwm_only:    ordinary WWM continuation on mixed official+generated text
  answer_mlm:  mask+predict true answer span (no counterfactual negative)
  random_neg:  rank true answer above a random equal-length location
  scmlm_r:    full SCMLM-R: direction + interaction + multi-candidate

All arms share: same checkpoint init, same official WWM exposure, same optimizer,
same total steps, same seeds. Only the additional state-pair loss differs.
"""
from __future__ import annotations
import argparse, importlib.util, json, os, pathlib, random, sys, time
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForMaskedLM, AutoTokenizer, get_cosine_schedule_with_warmup

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
sys.path.insert(0, str(ROOT / "scripts"))
from scmlm_r_loss import scmlm_r_loss, make_masked_input, span_logprob

# Import generator for on-the-fly pair generation
spec = importlib.util.spec_from_file_location("r1gen", ROOT / "scripts/r1_generator_v3.py")
r1gen = importlib.util.module_from_spec(spec); sys.modules[spec.name] = r1gen; spec.loader.exec_module(r1gen)

TRAIN_LOCS = r1gen.LOCATIONS[:7]
TRAIN_ITEMS = r1gen.ITEMS[:14]

# WWM utilities from existing trainer
spec_tr = importlib.util.spec_from_file_location("trainer", ROOT / "training/scripts/babylm_masked_train_leadershape.py")
trainer_mod = importlib.util.module_from_spec(spec_tr); sys.modules[spec_tr.name] = trainer_mod; spec_tr.loader.exec_module(trainer_mod)


def generate_training_pairs(n_pairs: int, seed: int, tokenizer) -> list[dict]:
    """Generate equal-token-length counterfactual pairs from training vocabulary."""
    r1gen.LOCATIONS = list(TRAIN_LOCS)
    r1gen.ITEMS = list(TRAIN_ITEMS)
    rng = random.Random(seed)
    pairs = []
    cur_seed = seed
    while len(pairs) < n_pairs:
        cur_seed += 7919
        batch = r1gen.generate_paired_corpus(n_pairs=max(n_pairs * 2, 200), seed=cur_seed)
        for a, b in batch:
            if a.answer == b.answer:
                continue
            a_ids = tokenizer(a.answer, add_special_tokens=False)["input_ids"]
            b_ids = tokenizer(b.answer, add_special_tokens=False)["input_ids"]
            if len(a_ids) != len(b_ids):
                continue
            # Collect other candidates (locations in init that aren't the answers)
            other_locs = [loc for loc in set(a.init_loc.values())
                          if loc != a.answer and loc != b.answer]
            # Filter to equal token length
            others_equal = [loc for loc in other_locs
                           if len(tokenizer(loc, add_special_tokens=False)["input_ids"]) == len(a_ids)]
            pairs.append({
                "passage_a": a.passage, "passage_b": b.passage,
                "answer_a": a.answer, "answer_b": b.answer,
                "other_candidates": others_equal[:3],
                "answer_token_len": len(a_ids),
            })
            if len(pairs) >= n_pairs:
                break
    return pairs


def wwm_forward(model, tokenizer, text: str, mask_prob: float, device, rng: random.Random):
    """Single-example WWM forward pass. Returns loss tensor."""
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=256)
    input_ids = enc["input_ids"][0].to(device)
    # Whole-word masking: find word boundaries
    tokens = tokenizer.convert_ids_to_tokens(input_ids.cpu().tolist())
    word_starts = [i for i, t in enumerate(tokens) if not t.startswith("Ġ") or i == 0]
    # Actually use word groups: a word group starts with a space-prefixed token
    word_groups = []
    current_group = [0]
    for i in range(1, len(tokens)):
        if tokens[i].startswith("Ġ"):
            word_groups.append(current_group)
            current_group = [i]
        else:
            current_group.append(i)
    word_groups.append(current_group)
    
    # Select ~mask_prob of word groups
    n_select = max(1, int(len(word_groups) * mask_prob))
    selected = rng.sample(range(len(word_groups)), min(n_select, len(word_groups)))
    mask_positions = []
    for idx in selected:
        mask_positions.extend(word_groups[idx])
    
    if not mask_positions:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    labels = torch.full_like(input_ids, -100)
    masked_input = input_ids.clone()
    for pos in mask_positions:
        labels[pos] = input_ids[pos]
        masked_input[pos] = tokenizer.mask_token_id
    
    outputs = model(input_ids=masked_input.unsqueeze(0), labels=labels.unsqueeze(0))
    return outputs.loss


def answer_mlm_forward(model, tokenizer, pair: dict, device):
    """Mask and predict true answer span — no counterfactual negative."""
    ids_a, attn_a, mask_a, a_ids_tok = make_masked_input(
        pair["passage_a"], pair["answer_a"], tokenizer, 256, device)
    logits = model(input_ids=ids_a, attention_mask=attn_a).logits[0]
    log_probs = F.log_softmax(logits[mask_a], dim=-1)
    ans_tensor = torch.tensor(a_ids_tok, device=device)
    loss_a = -log_probs.gather(1, ans_tensor.unsqueeze(1)).mean()
    
    ids_b, attn_b, mask_b, b_ids_tok = make_masked_input(
        pair["passage_b"], pair["answer_b"], tokenizer, 256, device)
    logits_b = model(input_ids=ids_b, attention_mask=attn_b).logits[0]
    log_probs_b = F.log_softmax(logits_b[mask_b], dim=-1)
    ans_tensor_b = torch.tensor(b_ids_tok, device=device)
    loss_b = -log_probs_b.gather(1, ans_tensor_b.unsqueeze(1)).mean()
    
    return (loss_a + loss_b) / 2


def random_neg_forward(model, tokenizer, pair: dict, device, rng: random.Random):
    """Rank true answer above a random equal-length location."""
    a_ids = tokenizer(pair["answer_a"], add_special_tokens=False)["input_ids"]
    target_len = len(a_ids)
    # Find a random negative from TRAIN_LOCS with same token length
    candidates = [loc for loc in TRAIN_LOCS
                  if loc != pair["answer_a"] and loc != pair["answer_b"]
                  and len(tokenizer(loc, add_special_tokens=False)["input_ids"]) == target_len]
    if not candidates:
        candidates = [pair["answer_b"]]  # fallback
    neg = rng.choice(candidates)
    neg_ids = tokenizer(neg, add_special_tokens=False)["input_ids"]
    
    ids_a, attn_a, mask_a, _ = make_masked_input(pair["passage_a"], pair["answer_a"], tokenizer, 256, device)
    s_true = span_logprob(model, ids_a, attn_a, mask_a, a_ids)
    s_neg = span_logprob(model, ids_a, attn_a, mask_a, neg_ids)
    loss = F.softplus(-(s_true - s_neg))
    return loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["wwm_only", "answer_mlm", "random_neg", "scmlm_r"], required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--n_steps", type=int, default=200)
    p.add_argument("--n_pairs", type=int, default=500)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--wwm_every", type=int, default=2, help="WWM step frequency (every N steps)")
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--lambda_dir", type=float, default=1.0)
    p.add_argument("--lambda_inter", type=float, default=0.5)
    p.add_argument("--lambda_multi", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=213)
    p.add_argument("--checkpoint_every", type=int, default=50)
    p.add_argument("--wwm_text_jsonl", default=str(ROOT / "data/r1_static_control/r1_ordered_dynamic_5M.jsonl"))
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = str(ROOT / "training/runs/r1_ordered_dynamic_5M/hf_model/chck_5M")
    out = pathlib.Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(ckpt, use_fast=True)
    if tokenizer.mask_token is None:
        tokenizer.mask_token = "<mask>"
    model = AutoModelForMaskedLM.from_pretrained(ckpt, trust_remote_code=True).to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(args.n_steps * 0.05), args.n_steps)

    # Generate training pairs
    rng = random.Random(args.seed)
    pairs = generate_training_pairs(args.n_pairs, args.seed, tokenizer)
    print(f"Generated {len(pairs)} training pairs")

    # Load WWM text pool
    wwm_texts = []
    with open(args.wwm_text_jsonl) as f:
        for i, line in enumerate(f):
            if i >= 5000:
                break
            row = json.loads(line)
            wwm_texts.append(row["text"])
    rng.shuffle(wwm_texts)
    print(f"Loaded {len(wwm_texts)} WWM texts")

    # Training loop
    log = []
    pair_idx = 0
    wwm_idx = 0
    for step in range(args.n_steps):
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=device)
        step_info = {"step": step}

        # WWM component (every step or every N steps)
        if step % args.wwm_every == 0:
            text = wwm_texts[wwm_idx % len(wwm_texts)]
            wwm_idx += 1
            wwm_loss = wwm_forward(model, tokenizer, text, 0.15, device, rng)
            total_loss = total_loss + wwm_loss
            step_info["wwm_loss"] = wwm_loss.item()

        # Arm-specific component
        pair = pairs[pair_idx % len(pairs)]
        pair_idx += 1

        if args.arm == "wwm_only":
            # Extra WWM instead of pair loss
            text2 = wwm_texts[wwm_idx % len(wwm_texts)]; wwm_idx += 1
            extra = wwm_forward(model, tokenizer, text2, 0.15, device, rng)
            total_loss = total_loss + extra
            step_info["arm_loss"] = extra.item()

        elif args.arm == "answer_mlm":
            arm_loss = answer_mlm_forward(model, tokenizer, pair, device)
            total_loss = total_loss + arm_loss
            step_info["arm_loss"] = arm_loss.item()

        elif args.arm == "random_neg":
            arm_loss = random_neg_forward(model, tokenizer, pair, device, rng)
            total_loss = total_loss + arm_loss
            step_info["arm_loss"] = arm_loss.item()

        elif args.arm == "scmlm_r":
            result = scmlm_r_loss(
                model, tokenizer,
                pair["passage_a"], pair["passage_b"],
                pair["answer_a"], pair["answer_b"],
                other_candidates=pair.get("other_candidates"),
                tau=args.tau, gamma=args.gamma,
                lambda_dir=args.lambda_dir, lambda_inter=args.lambda_inter,
                lambda_multi=args.lambda_multi,
                max_length=256, device=device,
            )
            total_loss = total_loss + result["loss"]
            step_info["arm_loss"] = result["loss"].item()
            step_info["m_A"] = result["m_A"].item()
            step_info["m_B"] = result["m_B"].item()
            step_info["I"] = result["I"].item()

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        step_info["total_loss"] = total_loss.item()
        step_info["lr"] = scheduler.get_last_lr()[0]
        log.append(step_info)

        if step % 20 == 0:
            print(f"[{args.arm}] step={step} total_loss={total_loss.item():.4f} lr={step_info['lr']:.2e}" +
                  (f" m_A={step_info.get('m_A',0):.3f} m_B={step_info.get('m_B',0):.3f} I={step_info.get('I',0):.3f}" if 'I' in step_info else ""))

        # Checkpoint
        if (step + 1) % args.checkpoint_every == 0 or step == args.n_steps - 1:
            ck_dir = out / f"ckpt_step{step+1}"
            model.save_pretrained(ck_dir)
            tokenizer.save_pretrained(ck_dir)

    # Save log
    meta = {
        "arm": args.arm, "n_steps": args.n_steps, "n_pairs": len(pairs),
        "lr": args.lr, "seed": args.seed, "checkpoint": ckpt,
        "tau": args.tau, "gamma": args.gamma, "device": str(device),
        "log": log,
    }
    (out / "training_log.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Done. Output: {out}")


if __name__ == "__main__":
    main()
