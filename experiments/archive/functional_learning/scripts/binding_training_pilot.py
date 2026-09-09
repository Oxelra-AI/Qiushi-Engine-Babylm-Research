#!/usr/bin/env python3
"""research: Minimal entity-binding training pilot.

Scientific purpose
------------------
Test whether ordinary WWM MLM training on paired contrastive packets can close the
asymmetric binding gap identified in the counterfactual diagnostic:
  - The model has strong UPDATE detection but RETAIN margins are below NEUTRAL
  - Distractor updates weaken source-state predictions regardless of entity identity
  - Success = RETAIN margin approaches NEUTRAL margin while UPDATE margin stays positive

This pilot trains the inherited model on a tiny set of paired packets (mixed with
neutral padding text) for a short schedule and re-measures the binding diagnostic.

The experiment connects the two open questions:
  1. Does the paired structure provide enough binding supervision under standard MLM?
  2. If not, does a focused masking intervention (preferentially masking answer positions)
     improve acquisition?

Design:
  - Arm A: standard WWM (15%) on paired packets  
  - Arm B: focused masking (always mask answer positions + 10% random)
  - Both arms use same packets, same model initialization, same number of gradient steps
  - Measured by: RETAIN margin improvement over neutral baseline

Usage
-----
  python binding_training_pilot.py \\
    --model-path <coherent86_endpoint> \\
    --packet-jsonl <paired_packets.jsonl> \\
    --out-dir <output_directory> \\
    --gpu 0 --epochs 200
"""
import argparse, json, pathlib, sys, math, copy, time, hashlib
from typing import Dict, List, Any, Tuple, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def is_word_start(token_str: str) -> bool:
    return token_str.startswith("Ġ") or token_str.startswith("▁")


class PairedPacketDataset(Dataset):
    """Dataset that tokenizes paired packets for MLM training."""
    
    def __init__(self, packets, tokenizer, seq_length=256):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.special_ids = set(tokenizer.all_special_ids)
        self._word_start_cache = {}
        
        # Tokenize all packets
        self.items = []
        for pkt in packets:
            text = pkt["full_text"]
            enc = tokenizer(text, add_special_tokens=True, max_length=seq_length,
                            truncation=True, padding="max_length", return_tensors="pt")
            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)
            
            # Build word groups
            word_group = torch.full((seq_length,), -1, dtype=torch.long)
            gid = -1
            for i in range(seq_length):
                if attention_mask[i] == 0:
                    continue
                tid = int(input_ids[i])
                if tid in self.special_ids:
                    continue
                if gid < 0 or self._word_start_flag(tid) or i == 0:
                    gid += 1
                word_group[i] = gid
            
            # Find answer token positions (for focused masking)
            answer_text = pkt["answer_text"]
            answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]
            answer_positions = []
            ids_list = input_ids.tolist()
            for start in range(len(ids_list) - len(answer_ids), -1, -1):
                if ids_list[start:start+len(answer_ids)] == answer_ids:
                    answer_positions = list(range(start, start + len(answer_ids)))
                    break
            
            self.items.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "word_group": word_group,
                "answer_positions": answer_positions,
                "packet_type": pkt["packet_type"],
                "pair_id": pkt["pair_id"],
            })
    
    def _word_start_flag(self, tid):
        v = self._word_start_cache.get(tid)
        if v is None:
            s = self.tokenizer.convert_ids_to_tokens(int(tid))
            v = bool(s is not None and is_word_start(str(s)))
            self._word_start_cache[tid] = v
        return v
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        return self.items[idx]


def collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "word_group": torch.stack([b["word_group"] for b in batch]),
        "answer_positions": [b["answer_positions"] for b in batch],
    }


def apply_wwm(input_ids, attention_mask, word_group, mask_token_id, mask_prob, gen):
    """Standard whole-word masking matching the training code."""
    device = input_ids.device
    bsz, seq = input_ids.shape
    labels = input_ids.clone()
    
    for b in range(bsz):
        max_gid = int(word_group[b].max().item())
        if max_gid < 0:
            labels[b, :] = -100
            continue
        n_groups = max_gid + 1
        group_mask = torch.rand(n_groups, generator=gen, device=device) < mask_prob
        token_mask = torch.zeros(seq, dtype=torch.bool, device=device)
        for g in range(n_groups):
            if group_mask[g]:
                token_mask |= (word_group[b] == g)
        labels[b, ~token_mask] = -100
    
    masked_indices = (labels != -100)
    masked_inputs = input_ids.clone()
    replace_mask = torch.bernoulli(
        torch.full(input_ids.shape, 0.8, device=device), generator=gen).bool() & masked_indices
    masked_inputs[replace_mask] = mask_token_id
    rand_mask = torch.bernoulli(
        torch.full(input_ids.shape, 0.5, device=device), generator=gen).bool() & masked_indices & ~replace_mask
    vocab_size = 16384  # known
    rand_ids = torch.randint(0, vocab_size, (int(rand_mask.sum()),), generator=gen, device=device)
    masked_inputs[rand_mask] = rand_ids
    return masked_inputs, labels


def apply_focused_masking(input_ids, attention_mask, word_group, answer_positions_batch,
                          mask_token_id, background_mask_prob, gen):
    """Focused masking: always mask the answer + background random masking."""
    device = input_ids.device
    bsz, seq = input_ids.shape
    labels = input_ids.clone()
    
    for b in range(bsz):
        max_gid = int(word_group[b].max().item())
        if max_gid < 0:
            labels[b, :] = -100
            continue
        n_groups = max_gid + 1
        
        # Background masking at reduced rate
        group_mask = torch.rand(n_groups, generator=gen, device=device) < background_mask_prob
        token_mask = torch.zeros(seq, dtype=torch.bool, device=device)
        for g in range(n_groups):
            if group_mask[g]:
                token_mask |= (word_group[b] == g)
        
        # Force-mask answer positions
        ans_pos = answer_positions_batch[b]
        for pos in ans_pos:
            if 0 <= pos < seq:
                token_mask[pos] = True
        
        labels[b, ~token_mask] = -100
    
    masked_indices = (labels != -100)
    masked_inputs = input_ids.clone()
    replace_mask = torch.bernoulli(
        torch.full(input_ids.shape, 0.8, device=device), generator=gen).bool() & masked_indices
    masked_inputs[replace_mask] = mask_token_id
    rand_mask = torch.bernoulli(
        torch.full(input_ids.shape, 0.5, device=device), generator=gen).bool() & masked_indices & ~replace_mask
    vocab_size = 16384
    rand_ids = torch.randint(0, vocab_size, (int(rand_mask.sum()),), generator=gen, device=device)
    masked_inputs[rand_mask] = rand_ids
    return masked_inputs, labels


def evaluate_binding(model, tokenizer, packets, device):
    """Quick binding evaluation: measure UPDATE, RETAIN, and NEUTRAL margins."""
    from collections import defaultdict
    
    by_pair = defaultdict(dict)
    for p in packets:
        by_pair[p["pair_id"]][p["packet_type"]] = p
    
    results = {}
    model.eval()
    with torch.no_grad():
        for pair_id, pair in sorted(by_pair.items()):
            if "UPDATE" not in pair or "RETAIN" not in pair:
                continue
            u, r = pair["UPDATE"], pair["RETAIN"]
            source = u["source_sentence"]
            new_state = u["answer_text"]
            source_state = r["answer_text"]
            use_frame = u["use_sentence_frame"]
            
            def score(text, answer, foil):
                enc = tokenizer(text, return_tensors="pt", add_special_tokens=True,
                                max_length=256, truncation=True).to(device)
                ans_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
                foil_ids = tokenizer(foil, add_special_tokens=False)["input_ids"]
                ids = enc["input_ids"].squeeze(0).tolist()
                # Find last occurrence
                start = None
                for s in range(len(ids) - len(ans_ids), -1, -1):
                    if ids[s:s+len(ans_ids)] == ans_ids:
                        start = s
                        break
                if start is None:
                    return 0.0
                masked = list(ids)
                for i in range(start, start + len(ans_ids)):
                    masked[i] = tokenizer.mask_token_id
                logits = model(input_ids=torch.tensor([masked], device=device),
                               attention_mask=enc["attention_mask"]).logits
                alp = sum(float(torch.log_softmax(logits[0, start+i], -1)[ans_ids[i]])
                          for i in range(len(ans_ids)))
                n = min(len(foil_ids), len(ans_ids))
                flp = sum(float(torch.log_softmax(logits[0, start+i], -1)[foil_ids[i]])
                          for i in range(n))
                return alp - flp
            
            update_text = f"{source} {u['update_sentence']} {use_frame.replace('{STATE}', new_state)}"
            retain_text = f"{source} {r['update_sentence']} {use_frame.replace('{STATE}', source_state)}"
            neutral_text = f"{source} {use_frame.replace('{STATE}', source_state)}"
            
            um = score(update_text, new_state, source_state)
            rm = score(retain_text, source_state, new_state)
            nm = score(neutral_text, source_state, new_state)
            
            results[pair_id] = {
                "update_margin": um,
                "retain_margin": rm,
                "neutral_margin": nm,
                "retain_minus_neutral": rm - nm,
                "update_correct": um > 0,
                "retain_correct": rm > 0,
            }
    
    model.train()
    
    # Aggregate
    vals = list(results.values())
    n = len(vals)
    return {
        "per_pair": results,
        "mean_update": sum(v["update_margin"] for v in vals) / max(1, n),
        "mean_retain": sum(v["retain_margin"] for v in vals) / max(1, n),
        "mean_neutral": sum(v["neutral_margin"] for v in vals) / max(1, n),
        "mean_retain_minus_neutral": sum(v["retain_minus_neutral"] for v in vals) / max(1, n),
        "n_both_correct": sum(1 for v in vals if v["update_correct"] and v["retain_correct"]),
        "n_pairs": n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=str, required=True)
    ap.add_argument("--packet-jsonl", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--private-scale", type=float, default=0.75)
    ap.add_argument("--private-bottleneck", type=int, default=128)
    ap.add_argument("--eval-every", type=int, default=50)
    args = ap.parse_args()
    
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # Load model
    from transformers import AutoTokenizer, DebertaV2Config
    from safetensors.torch import load_file
    
    model_path = pathlib.Path(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    
    def load_model():
        sys.path.insert(0, str(pathlib.Path("experiments/archive/functional_learning/scripts")))
        from context_credit_trainer import FrozenSlowPrivateDebertaV2ForMaskedLM
        cfg = DebertaV2Config.from_pretrained(str(model_path), local_files_only=True)
        cfg.private_adapter_bottleneck = args.private_bottleneck
        cfg.private_adapter_scale = args.private_scale
        cfg.private_adapter_enabled = True
        m = FrozenSlowPrivateDebertaV2ForMaskedLM(cfg)
        sd = load_file(str(model_path / "model.safetensors"), device="cpu")
        m.load_state_dict(sd, strict=False)
        for mod in m.modules():
            if hasattr(mod, "private_adapter_enabled"):
                mod.private_adapter_enabled = True
        return m.to(device)
    
    # Load packets
    packets = []
    with open(args.packet_jsonl) as f:
        for line in f:
            if line.strip():
                packets.append(json.loads(line))
    
    print(f"Loaded {len(packets)} packets, device={device}", flush=True)
    
    ds = PairedPacketDataset(packets, tokenizer, seq_length=256)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    # ── ARM A: Standard WWM ──
    print("\n=== ARM A: Standard WWM ===", flush=True)
    model_a = load_model()
    model_a.train()
    # Only train private adapter parameters (freeze carrier)
    trainable_params = [p for n, p in model_a.named_parameters() if "private_adapter" in n]
    if not trainable_params:
        # Fallback: train all
        trainable_params = list(model_a.parameters())
        print("  Training ALL parameters (no private adapter found)", flush=True)
    else:
        for p in model_a.parameters():
            p.requires_grad_(False)
        for p in trainable_params:
            p.requires_grad_(True)
        print(f"  Training {len(trainable_params)} private adapter parameters", flush=True)
    
    optimizer_a = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    gen_a = torch.Generator(device=device)
    gen_a.manual_seed(43032)
    
    log_a = []
    eval_a = [evaluate_binding(model_a, tokenizer, packets, device)]
    eval_a[0]["epoch"] = 0
    print(f"  Epoch 0: UPDATE={eval_a[0]['mean_update']:+.2f} RETAIN={eval_a[0]['mean_retain']:+.2f} "
          f"NEUTRAL={eval_a[0]['mean_neutral']:+.2f} R-N={eval_a[0]['mean_retain_minus_neutral']:+.2f}", flush=True)
    
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        epoch_targets = 0
        epoch_answer_targets = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            word_group = batch["word_group"].to(device)
            
            masked_inputs, labels = apply_wwm(
                input_ids, attention_mask, word_group,
                tokenizer.mask_token_id, 0.15, gen_a
            )
            n_targets = int((labels != -100).sum().item())
            if n_targets == 0:
                continue
            
            out = model_a(input_ids=masked_inputs, attention_mask=attention_mask, labels=labels)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer_a.step()
            optimizer_a.zero_grad(set_to_none=True)
            
            epoch_loss += float(loss) * n_targets
            epoch_targets += n_targets
        
        if epoch_targets > 0:
            log_a.append({"epoch": epoch, "loss": epoch_loss / epoch_targets, "targets": epoch_targets})
        
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            ev = evaluate_binding(model_a, tokenizer, packets, device)
            ev["epoch"] = epoch
            eval_a.append(ev)
            print(f"  Epoch {epoch}: loss={log_a[-1]['loss']:.4f} UPDATE={ev['mean_update']:+.2f} "
                  f"RETAIN={ev['mean_retain']:+.2f} NEUTRAL={ev['mean_neutral']:+.2f} "
                  f"R-N={ev['mean_retain_minus_neutral']:+.2f} both={ev['n_both_correct']}/{ev['n_pairs']}", flush=True)
    
    del model_a, optimizer_a
    torch.cuda.empty_cache()
    
    # ── ARM B: Focused masking ──
    print("\n=== ARM B: Focused Masking ===", flush=True)
    model_b = load_model()
    model_b.train()
    trainable_params_b = [p for n, p in model_b.named_parameters() if "private_adapter" in n]
    if not trainable_params_b:
        trainable_params_b = list(model_b.parameters())
    else:
        for p in model_b.parameters():
            p.requires_grad_(False)
        for p in trainable_params_b:
            p.requires_grad_(True)
    
    optimizer_b = torch.optim.AdamW(trainable_params_b, lr=args.lr, weight_decay=0.01)
    gen_b = torch.Generator(device=device)
    gen_b.manual_seed(43032)
    
    log_b = []
    eval_b = [evaluate_binding(model_b, tokenizer, packets, device)]
    eval_b[0]["epoch"] = 0
    print(f"  Epoch 0: UPDATE={eval_b[0]['mean_update']:+.2f} RETAIN={eval_b[0]['mean_retain']:+.2f} "
          f"NEUTRAL={eval_b[0]['mean_neutral']:+.2f} R-N={eval_b[0]['mean_retain_minus_neutral']:+.2f}", flush=True)
    
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        epoch_targets = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            word_group = batch["word_group"].to(device)
            answer_pos = batch["answer_positions"]
            
            masked_inputs, labels = apply_focused_masking(
                input_ids, attention_mask, word_group, answer_pos,
                tokenizer.mask_token_id, 0.10, gen_b
            )
            n_targets = int((labels != -100).sum().item())
            if n_targets == 0:
                continue
            
            out = model_b(input_ids=masked_inputs, attention_mask=attention_mask, labels=labels)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params_b, 1.0)
            optimizer_b.step()
            optimizer_b.zero_grad(set_to_none=True)
            
            epoch_loss += float(loss) * n_targets
            epoch_targets += n_targets
        
        if epoch_targets > 0:
            log_b.append({"epoch": epoch, "loss": epoch_loss / epoch_targets, "targets": epoch_targets})
        
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            ev = evaluate_binding(model_b, tokenizer, packets, device)
            ev["epoch"] = epoch
            eval_b.append(ev)
            print(f"  Epoch {epoch}: loss={log_b[-1]['loss']:.4f} UPDATE={ev['mean_update']:+.2f} "
                  f"RETAIN={ev['mean_retain']:+.2f} NEUTRAL={ev['mean_neutral']:+.2f} "
                  f"R-N={ev['mean_retain_minus_neutral']:+.2f} both={ev['n_both_correct']}/{ev['n_pairs']}", flush=True)
    
    # ── Save results ──
    summary = {
        "status": "BINDING_TRAINING_PILOT",
        "model_path": str(args.model_path),
        "n_packets": len(packets),
        "epochs": args.epochs,
        "lr": args.lr,
        "arm_a_standard_wwm": {
            "final_eval": eval_a[-1],
            "trajectory": eval_a,
        },
        "arm_b_focused_masking": {
            "final_eval": eval_b[-1],
            "trajectory": eval_b,
        },
    }
    
    with open(out_dir / "binding_training_pilot.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Compact summary
    print(json.dumps({
        "status": "BINDING_TRAINING_PILOT",
        "arm_a_final_R_minus_N": eval_a[-1]["mean_retain_minus_neutral"],
        "arm_b_final_R_minus_N": eval_b[-1]["mean_retain_minus_neutral"],
        "arm_a_both_correct": eval_a[-1]["n_both_correct"],
        "arm_b_both_correct": eval_b[-1]["n_both_correct"],
        "baseline_R_minus_N": eval_a[0]["mean_retain_minus_neutral"],
        "out": str(out_dir / "binding_training_pilot.json"),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
