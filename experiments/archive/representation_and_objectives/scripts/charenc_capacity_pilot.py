#!/usr/bin/env python3
"""Quick CPU pilot: does larger CharGRU capacity fix hard-assignment generalization?

Test char_hidden in {16, 32, 64, 128} with 50 epochs equality pretraining.
Measures argmax hard-assignment accuracy on train and eval names.
"""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import sys, json, time, copy, random
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, str(_public_path('experiments/archive/representation_and_objectives/training/scripts')))
import raw_span_discovery_probe as base

data_root = Path("experiments/archive/representation_and_objectives/data/information_budget_substrate/replace_k16_spread")
arm = "aligned_state_bridge"
out_dir = Path("experiments/archive/representation_and_objectives/data/charenc_capacity_pilot")
out_dir.mkdir(parents=True, exist_ok=True)


def unique_train_events(states, comps):
    seen = set(); out = []
    for q in states:
        if q.is_changed:
            k = (q.event, tuple(q.names))
            if k not in seen: seen.add(k); out.append(k)
    for c in comps:
        for ev in [c.event1, c.event2]:
            k = (ev, tuple(c.names))
            if k not in seen: seen.add(k); out.append(k)
    return out


def equality_loss_for_text(trunk, text, names, device, null_weight=1.0):
    _, char_forms = base.raw_tokenize(text)
    losses = []
    for ci, cand in enumerate(names):
        other = names[1 - ci]
        tok_chars = [base.name_to_char_ids(f) if f else [0] for f in char_forms]
        tcb = base.pad_char_batch(tok_chars, device)
        cc = base.pad_char_batch([base.name_to_char_ids(cand)], device)
        oc = base.pad_char_batch([base.name_to_char_ids(other)], device)
        tok_vecs = trunk.char_enc(tcb)
        cand_vec = trunk.char_enc(cc)
        other_vec = trunk.char_enc(oc)
        sc = (tok_vecs * cand_vec).sum(-1)
        so = (tok_vecs * other_vec).sum(-1)
        sn = trunk.neither_bias.expand(len(char_forms))
        logits = torch.stack([sc, so, sn], dim=-1)
        for i, f in enumerate(char_forms):
            if not f or f == "<hyp>": logits[i, 0] = -1e6; logits[i, 1] = -1e6
        cand_lc, other_lc = cand.lower(), other.lower()
        targets = torch.tensor([0 if f == cand_lc else (1 if f == other_lc else 2) for f in char_forms],
                               dtype=torch.long, device=device)
        ce = F.cross_entropy(logits, targets, reduction="none")
        name_mask = targets != 2
        null_mask = (targets == 2) & torch.tensor([bool(f) and f != "<hyp>" for f in char_forms], device=device)
        parts = []
        if name_mask.any(): parts.append(ce[name_mask].mean())
        if null_mask.any(): parts.append(null_weight * ce[null_mask].mean())
        if parts: losses.append(torch.stack(parts).mean())
    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=device)


def eval_hard_assignment(model, vocab, states, comps, device):
    model.eval()
    correct_both, total = 0, 0
    correct_cand, correct_other = 0, 0
    # Relative discrimination: at the name position, is the correct query higher?
    rel_correct, rel_total = 0, 0

    with torch.no_grad():
        items = []
        for q in states:
            if q.is_changed: items.append(("state", q.event, q.names))
        for c in comps:
            items.append(("cmp", c.event1, c.names))
            items.append(("cmp", c.event2, c.names))

        for kind, text, names in items:
            scorer = model.event_state if kind == "state" else model.event_cmp
            trunk = scorer.trunk
            _, char_forms = base.raw_tokenize(text)
            token_ids = vocab.encode(_) if False else None  # not needed for matching
            
            # Get char forms
            tok_ids_str, char_forms = base.raw_tokenize(text)
            
            for ci, cand in enumerate(names):
                other = names[1 - ci]
                tok_chars = [base.name_to_char_ids(f) if f else [0] for f in char_forms]
                tcb = base.pad_char_batch(tok_chars, device)
                cc = base.pad_char_batch([base.name_to_char_ids(cand)], device)
                oc = base.pad_char_batch([base.name_to_char_ids(other)], device)
                tok_vecs = trunk.char_enc(tcb)
                cand_vec = trunk.char_enc(cc)
                other_vec = trunk.char_enc(oc)
                sc = (tok_vecs * cand_vec).sum(-1).cpu().numpy()
                so = (tok_vecs * other_vec).sum(-1).cpu().numpy()
                
                cand_lc, other_lc = cand.lower(), other.lower()
                cand_pos = next((i for i, f in enumerate(char_forms) if f == cand_lc), -1)
                other_pos = next((i for i, f in enumerate(char_forms) if f == other_lc), -1)
                
                if cand_pos < 0 or other_pos < 0: continue
                
                # Hard assignment: argmax over non-special tokens
                for i, f in enumerate(char_forms):
                    if not f or f == "<hyp>": sc[i] = -1e6; so[i] = -1e6
                
                pred_cand = int(sc.argmax())
                pred_other = int(so.argmax())
                if pred_cand == pred_other:
                    if sc[pred_cand] >= so[pred_other]:
                        so_copy = so.copy(); so_copy[pred_cand] = -1e9
                        pred_other = int(so_copy.argmax())
                    else:
                        sc_copy = sc.copy(); sc_copy[pred_other] = -1e9
                        pred_cand = int(sc_copy.argmax())
                
                c_ok = pred_cand == cand_pos
                o_ok = pred_other == other_pos
                correct_cand += c_ok; correct_other += o_ok
                correct_both += (c_ok and o_ok); total += 1
                
                # Relative: at cand_pos, is sc > so?
                rel_correct += int(sc[cand_pos] > so[cand_pos])
                rel_total += 1
    
    return {
        "n": total,
        "cand_correct": correct_cand / total if total else 0,
        "other_correct": correct_other / total if total else 0,
        "both_correct": correct_both / total if total else 0,
        "relative_correct": rel_correct / rel_total if rel_total else 0,
    }


# Load data
ts, tc, es, ec, pe, counts = base.load_dataset(data_root, arm)
train_names = set()
for q in ts:
    for n in q.names: train_names.add(n)
for c in tc:
    for n in c.names: train_names.add(n)
vocab = base.RawVocab()
base.collect_raw_vocab(vocab, ts, tc, exclude_names=train_names)
print(f"Vocab: {len(vocab.itos)}", flush=True)

examples = unique_train_events(ts, tc)
device = torch.device("cpu")
checkpoints = [0, 10, 25, 50]
all_results = []

for char_hidden in [16, 32, 64, 128]:
    print(f"\n=== char_hidden={char_hidden} ===", flush=True)
    torch.manual_seed(29800); random.seed(29800); np.random.seed(29800)
    
    # Build model with this char_hidden
    paired = base.create_paired_span_models(len(vocab.itos), ["shared_trunk"], 29800,
                                             48, 64, 16, char_hidden)
    model = copy.deepcopy(paired["shared_trunk"])
    model.to(device)
    
    # Get trunks
    trunks = []
    seen_ids = set()
    for scorer in [model.event_state, model.event_cmp]:
        if id(scorer.trunk) not in seen_ids:
            trunks.append(scorer.trunk); seen_ids.add(id(scorer.trunk))
    
    params = []
    seen_p = set()
    for tr in trunks:
        for p in list(tr.char_enc.parameters()) + [tr.neither_bias]:
            if id(p) not in seen_p: params.append(p); seen_p.add(id(p))
    
    for p in params: p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=5e-3, weight_decay=0.0)
    rng = random.Random(42)
    t0 = time.time()
    
    for ep in range(0, 51):
        if ep in checkpoints:
            train_acc = eval_hard_assignment(model, vocab, ts, tc, device)
            eval_acc = eval_hard_assignment(model, vocab, es, ec, device)
            rec = {"char_hidden": char_hidden, "epoch": ep,
                   "elapsed": round(time.time() - t0, 1),
                   "train_both": round(train_acc["both_correct"], 4),
                   "train_rel": round(train_acc["relative_correct"], 4),
                   "eval_both": round(eval_acc["both_correct"], 4),
                   "eval_rel": round(eval_acc["relative_correct"], 4)}
            all_results.append(rec)
            print(json.dumps(rec), flush=True)
        
        if ep < 50:
            model.train()
            exs = list(examples); rng.shuffle(exs)
            opt.zero_grad(set_to_none=True)
            losses = []
            for tr in trunks:
                for text, names in exs:
                    losses.append(equality_loss_for_text(tr, text, names, device))
            if losses:
                loss = torch.stack(losses).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                opt.step()

# Save results
(out_dir / "capacity_pilot.json").write_text(json.dumps({"results": all_results}, indent=2))
lines = ["# CharGRU capacity pilot", "",
         "| char_hidden | epoch | train_both | train_rel | eval_both | eval_rel |",
         "|---:|---:|---:|---:|---:|---:|"]
for r in all_results:
    lines.append(f"| {r['char_hidden']} | {r['epoch']} | {r['train_both']:.4f} | {r['train_rel']:.4f} | {r['eval_both']:.4f} | {r['eval_rel']:.4f} |")
(out_dir / "capacity_pilot.md").write_text("\n".join(lines) + "\n")
print(json.dumps({"status": "COMPLETE", "md": str(out_dir / "capacity_pilot.md")}, indent=2))
