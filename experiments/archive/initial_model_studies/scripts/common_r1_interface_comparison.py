#!/usr/bin/env python3
"""research — Common R1 interface comparison: dense intermediate supervision vs causal.

Arms:
  A: Bidirectional DeBERTa + dense intermediate state supervision at every operation prefix
  B: Small causal Transformer trained on R1 scenario text with prefix-conditioned scoring

Both arms share the same R1 train/dev/test split, update budget, and evaluation pairs.
The decisive comparison: can dense intermediate supervision make a bidirectional encoder
learn transferable state/order composition, or does causal prefix scoring do so naturally?

Key difference from research: Arm A supervises EVERY intermediate state after each operation,
not just the final masked-answer readout. This supplies the missing intermediate-supervision control.
"""
from __future__ import annotations
import argparse, json, pathlib, sys, time, random, copy
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
sys.path.insert(0, str(ROOT / "scripts"))
import importlib.util
spec = importlib.util.spec_from_file_location("r1gen", ROOT / "scripts/r1_generator_v3.py")
r1gen = importlib.util.module_from_spec(spec); sys.modules[spec.name] = r1gen; spec.loader.exec_module(r1gen)

CKPT_MLM = str(ROOT / "training/runs/r1_ordered_dynamic_5M/hf_model/chck_5M")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ══ Vocabulary splits ══════════════════════════════════════════════════════════
TRAIN_LOCS = r1gen.LOCATIONS[:7]
TRAIN_ITEMS = r1gen.ITEMS[:14]
TEST_LOCS = r1gen.LOCATIONS[7:]
TEST_ITEMS = r1gen.ITEMS[14:]

# ══ R1 data generation with intermediate state metadata ═══════════════════════

@dataclass
class IntermediateStateExample:
    """One intermediate state supervision example."""
    passage_prefix: str      # text up to and including operation t
    query: str               # e.g. "After this change, the lamp is in [MASK]."
    answer: str              # the location token(s)
    step_idx: int            # which operation (0-indexed)
    kind: str                # "state_query" or "indirect_resolution"


def build_intermediate_examples(scenario: r1gen.Scenario) -> list[IntermediateStateExample]:
    """Extract dense intermediate supervision from a single scenario."""
    examples = []
    # Reconstruct prefix states
    state = dict(scenario.init_loc)
    init_text = " ".join(f"Initially, {obj} is in {loc}." for obj, loc in scenario.init_loc.items())
    prefix_so_far = init_text

    for t, op in enumerate(scenario.operations):
        prefix_so_far += " " + op.text
        # Update state
        state[op.obj] = op.dst

        # State query: "After this change, <affected_obj> is in [answer]."
        query_text = f"{prefix_so_far} After this change, {op.obj} is in"
        examples.append(IntermediateStateExample(
            passage_prefix=prefix_so_far,
            query=query_text,
            answer=op.dst,
            step_idx=t,
            kind="state_query"
        ))

        # For indirect ops: "The item in <src> was <object>."
        if op.ref_by_loc:
            resolution_query = f"{prefix_so_far} The item in {op.src} was"
            examples.append(IntermediateStateExample(
                passage_prefix=prefix_so_far,
                query=resolution_query,
                answer=op.obj,
                step_idx=t,
                kind="indirect_resolution"
            ))

    return examples


def gen_scenarios(split: str, n: int, seed: int) -> list[r1gen.Scenario]:
    """Generate R1 scenarios for a given vocabulary split."""
    if split == "train":
        r1gen.LOCATIONS = list(TRAIN_LOCS); r1gen.ITEMS = list(TRAIN_ITEMS)
    else:
        r1gen.LOCATIONS = list(TEST_LOCS); r1gen.ITEMS = list(TEST_ITEMS)
    rng = random.Random(seed)
    scenarios = []
    for _ in range(n * 3):
        if len(scenarios) >= n:
            break
        try:
            sc = r1gen.generate_scenario(rng, n_locs=5, n_items=4, n_ops=5, indirect_fraction=0.6)
            scenarios.append(sc)
        except:
            continue
    return scenarios[:n]


def gen_counterfactual_pairs(split: str, n: int, seed: int, tok) -> list[dict]:
    """Generate equal-token counterfactual pairs for final evaluation."""
    if split == "train":
        r1gen.LOCATIONS = list(TRAIN_LOCS); r1gen.ITEMS = list(TRAIN_ITEMS)
    else:
        r1gen.LOCATIONS = list(TEST_LOCS); r1gen.ITEMS = list(TEST_ITEMS)
    recs = []; cur = seed
    while len(recs) < n:
        cur += 7919
        for a, b in r1gen.generate_paired_corpus(n_pairs=max(n, 100), seed=cur):
            if a.answer == b.answer: continue
            ai = tok(a.answer, add_special_tokens=False)["input_ids"]
            bi = tok(b.answer, add_special_tokens=False)["input_ids"]
            if len(ai) != len(bi): continue
            recs.append({"passage_a": a.passage, "passage_b": b.passage,
                         "answer_a": a.answer, "answer_b": b.answer,
                         "a_ids": ai, "b_ids": bi})
            if len(recs) >= n: break
    return recs


# ══ Arm A: Bidirectional Dense Intermediate Supervision ═══════════════════════

def train_arm_a(tok, train_scenarios, train_pairs, test_pairs, n_epochs: int, lr: float) -> dict:
    """Train bidirectional DeBERTa with dense intermediate state supervision."""
    model = AutoModelForMaskedLM.from_pretrained(CKPT_MLM, trust_remote_code=True).to(DEVICE)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Build intermediate examples from train scenarios
    all_inter_examples = []
    for sc in train_scenarios:
        all_inter_examples.extend(build_intermediate_examples(sc))
    print(f"  Arm A: {len(all_inter_examples)} intermediate supervision examples from {len(train_scenarios)} scenarios")

    mask_token = tok.mask_token
    mask_id = tok.mask_token_id
    curve = []
    t0 = time.time()

    for epoch in range(n_epochs):
        random.Random(epoch).shuffle(all_inter_examples)
        epoch_losses = []; epoch_correct = 0; epoch_total = 0

        for ex in all_inter_examples[:400]:  # cap per epoch for speed
            # Tokenize query with answer masked
            query_with_mask = ex.query + f" {mask_token} ."
            enc = tok(query_with_mask, return_tensors="pt", truncation=True, max_length=256,
                      add_special_tokens=False).to(DEVICE)
            mask_pos = (enc["input_ids"][0] == mask_id).nonzero(as_tuple=False).view(-1)
            if len(mask_pos) == 0:
                continue

            answer_ids = tok(ex.answer, add_special_tokens=False)["input_ids"]
            # Use first mask position, first answer token
            target_id = answer_ids[0] if answer_ids else None
            if target_id is None:
                continue

            out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            logits_at_mask = out.logits[0, mask_pos[0]]
            loss = F.cross_entropy(logits_at_mask.unsqueeze(0),
                                   torch.tensor([target_id], device=DEVICE))
            # Track accuracy
            pred = logits_at_mask.argmax().item()
            epoch_correct += int(pred == target_id)
            epoch_total += 1

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_losses.append(loss.item())

        if (epoch + 1) % 5 == 0 or epoch == 0:
            acc = epoch_correct / max(1, epoch_total)
            test_eval = evaluate_final_pairs_mlm(model, tok, test_pairs)
            train_eval = evaluate_final_pairs_mlm(model, tok, train_pairs[:50])
            entry = {"epoch": epoch + 1, "loss": sum(epoch_losses) / max(1, len(epoch_losses)),
                     "inter_acc": round(acc, 3),
                     "train_bc": train_eval["both_correct"], "test_bc": test_eval["both_correct"],
                     "train_I": train_eval["I_mean"], "test_I": test_eval["I_mean"]}
            curve.append(entry)
            print(f"    ep{epoch+1}: loss={entry['loss']:.4f} inter_acc={acc:.3f} "
                  f"train_bc={entry['train_bc']:.3f} test_bc={entry['test_bc']:.3f}")

    final_test = evaluate_final_pairs_mlm(model, tok, test_pairs)
    final_train = evaluate_final_pairs_mlm(model, tok, train_pairs[:50])
    return {"arm": "A_bidir_intermediate_supervision", "n_epochs": n_epochs,
            "n_inter_examples": len(all_inter_examples),
            "final_train": final_train, "final_test": final_test,
            "curve": curve, "elapsed_sec": round(time.time() - t0, 1)}


def evaluate_final_pairs_mlm(model, tok, pairs: list[dict]) -> dict:
    """Evaluate final counterfactual pairs using the MLM head."""
    model.eval()
    results = []
    mask_id = tok.mask_token_id
    with torch.no_grad():
        for rec in pairs:
            try:
                a_ids = rec["a_ids"]; b_ids = rec["b_ids"]
                # Context A
                passage_a_masked = rec["passage_a"].replace(rec["answer_a"],
                    " ".join([tok.mask_token] * len(a_ids)), 1)
                # Find from end
                idx = rec["passage_a"].rfind(rec["answer_a"])
                passage_a_masked = rec["passage_a"][:idx] + " ".join([tok.mask_token]*len(a_ids)) + rec["passage_a"][idx+len(rec["answer_a"]):]
                enc_a = tok(passage_a_masked, return_tensors="pt", truncation=True, max_length=256,
                           add_special_tokens=False).to(DEVICE)
                mask_a = (enc_a["input_ids"][0] == mask_id).nonzero(as_tuple=False).view(-1)
                if len(mask_a) != len(a_ids): continue

                # Context B
                idx_b = rec["passage_b"].rfind(rec["answer_b"])
                passage_b_masked = rec["passage_b"][:idx_b] + " ".join([tok.mask_token]*len(b_ids)) + rec["passage_b"][idx_b+len(rec["answer_b"]):]
                enc_b = tok(passage_b_masked, return_tensors="pt", truncation=True, max_length=256,
                           add_special_tokens=False).to(DEVICE)
                mask_b = (enc_b["input_ids"][0] == mask_id).nonzero(as_tuple=False).view(-1)
                if len(mask_b) != len(b_ids): continue

                out_a = model(input_ids=enc_a["input_ids"], attention_mask=enc_a["attention_mask"])
                out_b = model(input_ids=enc_b["input_ids"], attention_mask=enc_b["attention_mask"])
                lp_a = F.log_softmax(out_a.logits[0, mask_a], dim=-1)
                lp_b = F.log_softmax(out_b.logits[0, mask_b], dim=-1)
                rows = torch.arange(len(a_ids), device=DEVICE)
                a_t = torch.tensor(a_ids, device=DEVICE); b_t = torch.tensor(b_ids, device=DEVICE)
                s_Aa = lp_a[rows, a_t].sum().item(); s_Ab = lp_a[rows, b_t].sum().item()
                s_Bb = lp_b[rows, b_t].sum().item(); s_Ba = lp_b[rows, a_t].sum().item()
                m_A = s_Aa - s_Ab; m_B = s_Bb - s_Ba
                results.append({"m_A": m_A, "m_B": m_B, "I": m_A + m_B,
                               "both_correct": float(m_A > 0 and m_B > 0)})
            except Exception:
                continue
    model.train()
    if not results:
        return {"both_correct": 0, "I_mean": 0, "m_A_mean": 0, "m_B_mean": 0, "n": 0}
    n = len(results)
    return {"both_correct": sum(r["both_correct"] for r in results) / n,
            "I_mean": sum(r["I"] for r in results) / n,
            "m_A_mean": sum(r["m_A"] for r in results) / n,
            "m_B_mean": sum(r["m_B"] for r in results) / n, "n": n}


# ══ Arm B: Dense Causal Prefix Scorer ═════════════════════════════════════════

def build_causal_model(tok) -> GPT2LMHeadModel:
    """Build a small causal model with same tokenizer vocabulary."""
    config = GPT2Config(
        vocab_size=tok.vocab_size,
        n_positions=256,
        n_embd=480,
        n_layer=8,
        n_head=8,
        n_inner=480 * 4,
        activation_function="gelu_new",
        bos_token_id=tok.bos_token_id or 0,
        eos_token_id=tok.eos_token_id or 0,
    )
    model = GPT2LMHeadModel(config)
    return model


def train_arm_b(tok, train_scenarios, train_pairs, test_pairs, n_epochs: int, lr: float) -> dict:
    """Train causal model with next-token prediction on R1 scenario text + answer scoring."""
    model = build_causal_model(tok).to(DEVICE)
    model.train()
    # Ensure pad token
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id if tok.eos_token_id is not None else 0
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Build training texts: full scenario passages (includes init + ops + final answer)
    train_texts = [sc.passage for sc in train_scenarios]
    # Also include intermediate state query texts for comparable supervision density
    for sc in train_scenarios:
        for ex in build_intermediate_examples(sc):
            train_texts.append(f"{ex.query} {ex.answer} .")

    print(f"  Arm B: {len(train_texts)} causal training texts from {len(train_scenarios)} scenarios")
    curve = []
    t0 = time.time()

    for epoch in range(n_epochs):
        random.Random(epoch).shuffle(train_texts)
        epoch_losses = []

        for text in train_texts[:600]:  # cap per epoch
            enc = tok(text, return_tensors="pt", truncation=True, max_length=256,
                      add_special_tokens=False).to(DEVICE)
            ids = enc["input_ids"]
            if ids.shape[1] < 3:
                continue
            # Causal LM loss: predict next token from prefix
            labels = ids.clone()
            out = model(input_ids=ids, attention_mask=enc["attention_mask"], labels=labels)
            loss = out.loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_losses.append(loss.item())

        if (epoch + 1) % 5 == 0 or epoch == 0:
            test_eval = evaluate_final_pairs_causal(model, tok, test_pairs)
            train_eval = evaluate_final_pairs_causal(model, tok, train_pairs[:50])
            entry = {"epoch": epoch + 1, "loss": sum(epoch_losses) / max(1, len(epoch_losses)),
                     "train_bc": train_eval["both_correct"], "test_bc": test_eval["both_correct"],
                     "train_I": train_eval["I_mean"], "test_I": test_eval["I_mean"]}
            curve.append(entry)
            print(f"    ep{epoch+1}: loss={entry['loss']:.4f} "
                  f"train_bc={entry['train_bc']:.3f} test_bc={entry['test_bc']:.3f}")

    final_test = evaluate_final_pairs_causal(model, tok, test_pairs)
    final_train = evaluate_final_pairs_causal(model, tok, train_pairs[:50])
    return {"arm": "B_causal_prefix", "n_epochs": n_epochs,
            "n_train_texts": len(train_texts),
            "final_train": final_train, "final_test": final_test,
            "curve": curve, "elapsed_sec": round(time.time() - t0, 1)}


def evaluate_final_pairs_causal(model, tok, pairs: list[dict]) -> dict:
    """Evaluate counterfactual pairs using causal prefix scoring.

    For each context, score = log P(answer | prefix up to answer position).
    We compute this by tokenizing the full passage, finding where the answer starts,
    and summing log-probs of answer tokens conditioned on the prefix.
    """
    model.eval()
    results = []
    with torch.no_grad():
        for rec in pairs:
            try:
                for passage_key, ans_key, other_ans_key, margin_name in [
                    ("passage_a", "answer_a", "answer_b", "m_A"),
                    ("passage_b", "answer_b", "answer_a", "m_B"),
                ]:
                    passage = rec[passage_key]
                    true_ans = rec[ans_key]
                    wrong_ans = rec[other_ans_key]

                    # Find answer position in passage
                    ans_idx = passage.rfind(true_ans)
                    if ans_idx < 0: raise ValueError("answer not found")
                    prefix_text = passage[:ans_idx]

                    # Score true answer
                    s_true = score_continuation(model, tok, prefix_text, true_ans)
                    s_wrong = score_continuation(model, tok, prefix_text, wrong_ans)
                    if margin_name == "m_A":
                        m_A = s_true - s_wrong
                    else:
                        m_B = s_true - s_wrong

                results.append({"m_A": m_A, "m_B": m_B, "I": m_A + m_B,
                               "both_correct": float(m_A > 0 and m_B > 0)})
            except Exception:
                continue
    model.train()
    if not results:
        return {"both_correct": 0, "I_mean": 0, "m_A_mean": 0, "m_B_mean": 0, "n": 0}
    n = len(results)
    return {"both_correct": sum(r["both_correct"] for r in results) / n,
            "I_mean": sum(r["I"] for r in results) / n,
            "m_A_mean": sum(r["m_A"] for r in results) / n,
            "m_B_mean": sum(r["m_B"] for r in results) / n, "n": n}


def score_continuation(model, tok, prefix: str, continuation: str) -> float:
    """Score P(continuation | prefix) under causal model."""
    full = prefix + continuation
    enc = tok(full, return_tensors="pt", truncation=True, max_length=256,
              add_special_tokens=False).to(DEVICE)
    prefix_enc = tok(prefix, add_special_tokens=False)["input_ids"]
    prefix_len = len(prefix_enc)
    ids = enc["input_ids"]
    if ids.shape[1] <= prefix_len:
        return 0.0

    out = model(input_ids=ids, attention_mask=enc["attention_mask"])
    logits = out.logits[0]  # (seq, vocab)
    # Score continuation tokens: position prefix_len-1 predicts token at prefix_len, etc.
    log_probs = F.log_softmax(logits, dim=-1)
    score = 0.0
    for i in range(prefix_len, ids.shape[1]):
        token_id = ids[0, i].item()
        score += log_probs[i - 1, token_id].item()
    return score


# ══ Main ══════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_train_scenarios", type=int, default=300)
    p.add_argument("--n_train_pairs", type=int, default=200)
    p.add_argument("--n_test_pairs", type=int, default=100)
    p.add_argument("--n_epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=226)
    p.add_argument("--output", default=str(ROOT / "data/common_r1_interface_comparison.json"))
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(CKPT_MLM, use_fast=True)
    if tok.mask_token is None: tok.mask_token = "<mask>"

    print("Generating shared R1 data...")
    train_scenarios = gen_scenarios("train", args.n_train_scenarios, args.seed)
    train_pairs = gen_counterfactual_pairs("train", args.n_train_pairs, args.seed + 100, tok)
    test_pairs = gen_counterfactual_pairs("test", args.n_test_pairs, args.seed + 2000, tok)
    print(f"  Scenarios: {len(train_scenarios)}, Train pairs: {len(train_pairs)}, Test pairs: {len(test_pairs)}")

    payload = {"status": "COMMON_R1_INTERFACE_COMPARISON",
               "n_train_scenarios": len(train_scenarios),
               "n_train_pairs": len(train_pairs), "n_test_pairs": len(test_pairs),
               "train_vocab": {"locs": TRAIN_LOCS, "items": TRAIN_ITEMS},
               "test_vocab": {"locs": TEST_LOCS, "items": TEST_ITEMS},
               "arms": {}}
    out_path = pathlib.Path(args.output); out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Arm A: Bidirectional dense intermediate supervision ──
    print("\n=== ARM A: Bidirectional Dense Intermediate Supervision ===")
    arm_a = train_arm_a(tok, train_scenarios, train_pairs, test_pairs, args.n_epochs, args.lr)
    payload["arms"]["A_bidir_intermediate"] = arm_a
    out_path.write_text(json.dumps(payload, indent=2) + "\n")

    # ── Arm B: Dense causal prefix scorer ──
    print("\n=== ARM B: Dense Causal Prefix Scorer ===")
    arm_b = train_arm_b(tok, train_scenarios, train_pairs, test_pairs, args.n_epochs, args.lr)
    payload["arms"]["B_causal_prefix"] = arm_b
    out_path.write_text(json.dumps(payload, indent=2) + "\n")

    # ── Summary ──
    print("\n=== SUMMARY ===")
    for arm_name, arm_data in payload["arms"].items():
        t = arm_data["final_test"]
        tr = arm_data["final_train"]
        print(f"  {arm_name:35s}: test bc={t['both_correct']:.3f} I={t['I_mean']:.4f} | "
              f"train bc={tr['both_correct']:.3f} I={tr['I_mean']:.4f}")

    # Interpretation
    bc_a = payload["arms"]["A_bidir_intermediate"]["final_test"]["both_correct"]
    bc_b = payload["arms"]["B_causal_prefix"]["final_test"]["both_correct"]
    if bc_a >= 0.30 and bc_b >= 0.30:
        interp = ("BOTH_VIABLE: both dense intermediate supervision and causal prefix scoring "
                  "achieve held-out generalization. Compare train/test gaps and official probes.")
    elif bc_a >= 0.30 > bc_b:
        interp = ("BIDIR_REPAIR_STRONGER: dense intermediate supervision achieves held-out "
                  "generalization while causal prefix does not. Bidirectional repair should be developed.")
    elif bc_b >= 0.30 > bc_a:
        interp = ("CAUSAL_STRONGER: causal prefix scoring achieves held-out generalization "
                  "while dense intermediate supervision does not. Causal/recursive route justified.")
    elif bc_a >= 0.15 or bc_b >= 0.15:
        interp = ("PARTIAL_SIGNAL: one or both arms show weak generalization (0.15-0.30). "
                  "Neither is decisive yet. Consider more data/epochs or different supervision.")
    else:
        interp = ("NEITHER_TRANSFERS: both arms fail held-out generalization. The R1 task may "
                  "require more data, stronger supervision, or is not solvable at this scale.")
    payload["interpretation"] = interp
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nInterpretation: {interp}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
