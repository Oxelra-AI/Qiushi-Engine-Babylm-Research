#!/usr/bin/env python3
"""research: MLM+RTD gradient-separated mechanism probe.

Tests whether adding dense RTD supervision to the existing DeBERTa-v2 8×480
MLM model produces a useful, non-shortcut, gradient-compatible learning signal.

Design from research route comparison + independent review: 
  H-A (GDES-wired dense supervision) was judged strongest new mechanism.
  Must verify BEFORE any H100 training run:
    1. Gradient geometry: per-layer cosine between MLM and RTD gradients
    2. Shortcut resistance: model-sampled vs random corruption detection gap
    3. GDES necessity: how much RTD pushes embeddings vs trunk
    4. Calibrated discrimination: frozen-encoder RTD head training

Procedure:
  Phase A: Gradient geometry with fresh RTD head (no training, raw gradients)
  Phase B: Train RTD head with frozen encoder (~100 steps), measure calibrated
           discrimination on held-out batches for both hard and random corruptions

Uses research legal 80M checkpoint (mature, well-trained) on the frozen reinvest
100M stream with the same WWM recipe as training.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, DebertaV2ForMaskedLM

# ─── Paths ──────────────────────────────────────────────────────────
CHECKPOINT = pathlib.Path(
    "experiments/archive/frontier_consolidation/training/runs"
    "complianttok_reinvest_seed43022_r2/hf_model/chck_80M"
)
STREAM = pathlib.Path(
    "experiments/archive/frontier_consolidation/data"
    "density_cleanqwen_overlay_medium_riskhard/"
    "cleanqwen_fineweb_compact_view_reinvest_100M.jsonl"
)
OUT_DIR = pathlib.Path(
    "experiments/archive/frontier_consolidation/data/mlm_rtd_mechanism_probe"
)

# ─── Config ─────────────────────────────────────────────────────────
NUM_PROBE_WORDS = 800_000  # ~5100 rows → ~20 batches
BATCH_SIZE = 256
SEQ_LENGTH = 256
MASK_PROB = 0.15
RTD_TEMPERATURE = 1.0
SEED = 43093
RTD_TRAIN_STEPS = 100  # frozen-encoder RTD head training
RTD_HEAD_LR = 3e-3


# ─── Data loading ───────────────────────────────────────────────────
def load_data(stream_path, max_words):
    examples = []
    total_words = 0
    with open(stream_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = obj["text"]
            words = int(obj.get("words", len(text.split())))
            if total_words + words > max_words and total_words > 0:
                break
            examples.append({"text": text, "words": words})
            total_words += words
    return examples, total_words


def is_word_start(token_str: str) -> bool:
    return token_str.startswith("Ġ") or token_str.startswith("▁")


class ProbeDataset(Dataset):
    def __init__(self, examples, tokenizer, seq_length):
        self.examples = examples
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.special_ids = set(tokenizer.all_special_ids)
        self._ws = {}

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex["text"], add_special_tokens=False, truncation=True,
            max_length=self.seq_length, padding="max_length", return_tensors="pt",
        )
        ids = enc["input_ids"].squeeze(0)
        mask = enc["attention_mask"].squeeze(0)
        grp = torch.full_like(ids, -1)
        gid = -1
        for i in range(ids.shape[0]):
            if mask[i] == 0:
                continue
            tid = int(ids[i])
            if tid in self.special_ids:
                continue
            if tid not in self._ws:
                s = self.tokenizer.convert_ids_to_tokens(tid)
                self._ws[tid] = bool(s and is_word_start(str(s)))
            if gid < 0 or self._ws[tid] or i == 0:
                gid += 1
            grp[i] = gid
        return ids, mask, grp


def collate_probe(batch):
    ids, masks, grps = zip(*batch)
    return torch.stack(ids), torch.stack(masks), torch.stack(grps)


# ─── WWM masking ────────────────────────────────────────────────────
def apply_wwm(input_ids, attention_mask, word_group, tokenizer, mask_prob, gen):
    device = input_ids.device
    bsz, seq = input_ids.shape
    mask_id = tokenizer.mask_token_id
    sp = torch.tensor(sorted(tokenizer.all_special_ids), device=device)
    cand = attention_mask.bool() & ~torch.isin(input_ids, sp)
    sel = torch.zeros_like(cand)

    for b in range(bsz):
        g = word_group[b]
        vg = torch.unique(g[g >= 0])
        if vg.numel() == 0:
            continue
        chosen = vg[torch.rand(vg.numel(), generator=gen, device=device) < mask_prob]
        if chosen.numel() > 0:
            sel[b] = torch.isin(g, chosen) & cand[b]

    if sel.sum() == 0:
        flat = cand.view(-1).nonzero(as_tuple=False)
        if flat.numel() > 0:
            sel.view(-1)[flat[0, 0]] = True

    labels = input_ids.clone()
    labels[~sel] = -100
    masked = input_ids.clone()
    r = torch.rand(bsz, seq, generator=gen, device=device)
    masked[sel & (r < 0.8)] = mask_id
    rt = sel & (r >= 0.8) & (r < 0.9)
    if rt.any():
        masked[rt] = torch.randint(0, len(tokenizer), (int(rt.sum()),),
                                   generator=gen, device=device)
    return masked, labels, sel


# ─── Corruption generators ──────────────────────────────────────────
def generate_hard_corruptions(model, input_ids, masked_inputs, attention_mask,
                              select, temperature):
    """Model-sampled corruptions (contextually plausible)."""
    device = input_ids.device
    bsz, seq = input_ids.shape
    with torch.no_grad():
        logits = model(input_ids=masked_inputs,
                       attention_mask=attention_mask).logits
    corrupted = input_ids.clone()
    rtd_labels = torch.zeros(bsz, seq, dtype=torch.long, device=device)
    if select.any():
        probs = F.softmax(logits[select] / temperature, dim=-1)
        sampled = torch.multinomial(probs, 1).squeeze(-1)
        corrupted[select] = sampled
        rtd_labels[select] = (sampled != input_ids[select]).long()
    n_m = select.sum().item()
    n_r = rtd_labels.sum().item()
    return corrupted, rtd_labels, {
        "n_masked": n_m, "n_replaced": n_r,
        "replacement_rate": n_r / max(1, n_m),
        "generator_accuracy": 1.0 - n_r / max(1, n_m),
    }


def generate_random_corruptions(input_ids, select, vocab_size, gen):
    """Random token replacement (shortcut baseline)."""
    device = input_ids.device
    bsz, seq = input_ids.shape
    corrupted = input_ids.clone()
    rtd_labels = torch.zeros(bsz, seq, dtype=torch.long, device=device)
    if select.any():
        rand_tok = torch.randint(0, vocab_size, (int(select.sum()),),
                                 generator=gen, device=device)
        corrupted[select] = rand_tok
        rtd_labels[select] = (rand_tok != input_ids[select]).long()
    return corrupted, rtd_labels, {
        "n_masked": select.sum().item(),
        "n_replaced": rtd_labels.sum().item(),
    }


# ─── RTD head ───────────────────────────────────────────────────────
class RTDHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()
        self.out = nn.Linear(hidden_size, 2)
        nn.init.normal_(self.dense.weight, std=0.02)
        nn.init.zeros_(self.dense.bias)
        nn.init.normal_(self.out.weight, std=0.02)
        nn.init.zeros_(self.out.bias)

    def forward(self, h):
        return self.out(self.act(self.dense(h)))


# ─── Gradient collection ────────────────────────────────────────────
def get_grad_vectors(model):
    """Per-layer flattened gradient vectors, keyed by named parameter groups."""
    groups = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        # Classify parameter into groups
        if "word_embeddings" in name:
            key = "embedding"
        elif "encoder.layer." in name:
            # Extract layer index from e.g. 'deberta.encoder.layer.3.attention...'
            parts = name.split(".")
            li = parts.index("layer") + 1
            key = f"layer_{parts[li]}"
        elif "cls." in name:
            key = "mlm_head"
        elif "encoder.rel_embeddings" in name:
            key = "rel_embeddings"
        elif "encoder." in name:
            key = "encoder_other"
        elif "embeddings." in name:
            key = "embeddings_other"
        else:
            key = "other"
        if key not in groups:
            groups[key] = []
        groups[key].append((name, param.grad.detach().flatten().clone()))
    # Concatenate within each group, sorted by name for consistency
    vecs = {}
    for key, parts in groups.items():
        parts.sort(key=lambda x: x[0])
        vecs[key] = torch.cat([p for _, p in parts])
    return vecs


def compute_rtd_metrics(logits, labels, attn_mask):
    v = attn_mask.bool()
    pred = logits[v].argmax(-1)
    true = labels[v]
    n = v.sum().item()
    n0 = (true == 0).sum().item()
    n1 = (true == 1).sum().item()
    acc = (pred == true).sum().item() / max(1, n)
    acc0 = ((pred == 0) & (true == 0)).sum().item() / max(1, n0)
    acc1 = ((pred == 1) & (true == 1)).sum().item() / max(1, n1)
    # AUROC
    auroc = None
    if n1 > 0 and n0 > 0:
        try:
            p1 = F.softmax(logits[v], dim=-1)[:, 1].detach().cpu().numpy()
            y = true.detach().cpu().numpy()
            # manual AUROC (avoid sklearn dependency)
            pos = p1[y == 1]
            neg = p1[y == 0]
            if len(pos) > 0 and len(neg) > 0:
                # sample-based approximation
                n_sample = min(5000, len(pos) * len(neg))
                if len(pos) * len(neg) <= n_sample:
                    auroc = float(np.mean(pos[:, None] > neg[None, :]))
                else:
                    pi = np.random.choice(len(pos), n_sample)
                    ni = np.random.choice(len(neg), n_sample)
                    auroc = float(np.mean(pos[pi] > neg[ni]))
        except Exception:
            pass
    return {"accuracy": acc, "acc_original": acc0, "acc_replaced": acc1,
            "n_original": n0, "n_replaced": n1, "auroc": auroc}


# ─── Main ───────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # Load data
    print("Loading data...", flush=True)
    examples, total_words = load_data(STREAM, NUM_PROBE_WORDS)
    print(f"  {len(examples)} examples, {total_words} words", flush=True)

    # Load model
    print("Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT), use_fast=True)
    model = DebertaV2ForMaskedLM.from_pretrained(str(CHECKPOINT))
    model.to(device)
    hs = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    vs = model.config.vocab_size
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  hidden={hs}, layers={n_layers}, vocab={vs}, params={n_params:,}", flush=True)

    # RTD head
    rtd_head = RTDHead(hs).to(device)
    n_rtd_params = sum(p.numel() for p in rtd_head.parameters())
    print(f"  RTD head params: {n_rtd_params:,}", flush=True)

    # Dataset
    ds = ProbeDataset(examples, tokenizer, SEQ_LENGTH)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=collate_probe, num_workers=0)
    batches = list(loader)
    n_batches = len(batches)
    # Split: first 75% for gradient/training, last 25% for eval
    n_train = max(1, int(n_batches * 0.75))
    n_eval = n_batches - n_train
    train_batches = batches[:n_train]
    eval_batches = batches[n_train:]
    print(f"  {n_batches} batches: {n_train} train, {n_eval} eval", flush=True)

    gen_m = torch.Generator(device=device)
    gen_m.manual_seed(SEED)
    gen_r = torch.Generator(device=device)
    gen_r.manual_seed(SEED + 1)

    # ================================================================
    # PHASE A: Gradient geometry with fresh RTD head
    # ================================================================
    print("\n=== Phase A: Gradient Geometry ===", flush=True)
    model.train()
    rtd_head.train()

    grad_cosines_hard = defaultdict(list)
    grad_cosines_rand = defaultdict(list)
    grad_norms_mlm = defaultdict(list)
    grad_norms_rtd_hard = defaultdict(list)
    grad_norms_rtd_rand = defaultdict(list)
    emb_rtd_norms = []
    emb_mlm_norms = []
    mlm_losses = []
    rtd_hard_losses = []
    rtd_rand_losses = []
    hard_stats_all = []
    rand_stats_all = []
    raw_hard_metrics = []
    raw_rand_metrics = []

    for bi, (ids, amask, wgrp) in enumerate(train_batches):
        ids, amask, wgrp = ids.to(device), amask.to(device), wgrp.to(device)

        # Apply WWM
        masked, labels, sel = apply_wwm(ids, amask, wgrp, tokenizer, MASK_PROB, gen_m)

        # ── MLM gradients ──
        model.zero_grad()
        rtd_head.zero_grad()
        out_mlm = model(input_ids=masked, attention_mask=amask, labels=labels)
        out_mlm.loss.backward()
        mlm_vecs = get_grad_vectors(model)
        mlm_losses.append(out_mlm.loss.item())
        for k, v in mlm_vecs.items():
            grad_norms_mlm[k].append(v.norm().item())
        if "embedding" in mlm_vecs:
            emb_mlm_norms.append(mlm_vecs["embedding"].norm().item())

        # ── Hard corruption + RTD gradients ──
        hard_corr, hard_lbl, hard_st = generate_hard_corruptions(
            model, ids, masked, amask, sel, RTD_TEMPERATURE
        )
        hard_stats_all.append(hard_st)

        model.zero_grad()
        rtd_head.zero_grad()
        enc_hard = model.deberta(input_ids=hard_corr, attention_mask=amask)
        rtd_logits_h = rtd_head(enc_hard.last_hidden_state)
        valid = amask.bool()
        loss_h = F.cross_entropy(rtd_logits_h[valid], hard_lbl[valid])
        loss_h.backward()
        rtd_hard_vecs = get_grad_vectors(model)
        rtd_hard_losses.append(loss_h.item())
        for k, v in rtd_hard_vecs.items():
            grad_norms_rtd_hard[k].append(v.norm().item())
        emb_w = model.deberta.embeddings.word_embeddings.weight
        if emb_w.grad is not None:
            emb_rtd_norms.append(emb_w.grad.norm().item())

        # Cosines MLM vs RTD-hard
        for k in mlm_vecs:
            if k in rtd_hard_vecs:
                c = F.cosine_similarity(
                    mlm_vecs[k].unsqueeze(0), rtd_hard_vecs[k].unsqueeze(0)
                ).item()
                grad_cosines_hard[k].append(c)

        # RTD metrics (untrained head)
        with torch.no_grad():
            raw_hard_metrics.append(
                compute_rtd_metrics(rtd_logits_h, hard_lbl, amask)
            )

        # ── Random corruption + RTD gradients ──
        rand_corr, rand_lbl, rand_st = generate_random_corruptions(
            ids, sel, vs, gen_r
        )
        rand_stats_all.append(rand_st)

        model.zero_grad()
        rtd_head.zero_grad()
        enc_rand = model.deberta(input_ids=rand_corr, attention_mask=amask)
        rtd_logits_r = rtd_head(enc_rand.last_hidden_state)
        loss_r = F.cross_entropy(rtd_logits_r[valid], rand_lbl[valid])
        loss_r.backward()
        rtd_rand_vecs = get_grad_vectors(model)
        rtd_rand_losses.append(loss_r.item())
        for k, v in rtd_rand_vecs.items():
            grad_norms_rtd_rand[k].append(v.norm().item())
        for k in mlm_vecs:
            if k in rtd_rand_vecs:
                c = F.cosine_similarity(
                    mlm_vecs[k].unsqueeze(0), rtd_rand_vecs[k].unsqueeze(0)
                ).item()
                grad_cosines_rand[k].append(c)

        with torch.no_grad():
            raw_rand_metrics.append(
                compute_rtd_metrics(rtd_logits_r, rand_lbl, amask)
            )

        if (bi + 1) % 5 == 0 or bi == 0:
            print(f"  batch {bi+1}/{n_train}: MLM loss={mlm_losses[-1]:.4f}  "
                  f"RTD-hard loss={rtd_hard_losses[-1]:.4f}  "
                  f"RTD-rand loss={rtd_rand_losses[-1]:.4f}", flush=True)

    # ================================================================
    # PHASE B: Calibrated RTD discrimination (frozen encoder)
    # ================================================================
    print("\n=== Phase B: Calibrated RTD Head Training ===", flush=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Fresh RTD head for calibration
    rtd_cal = RTDHead(hs).to(device)
    rtd_opt = torch.optim.Adam(rtd_cal.parameters(), lr=RTD_HEAD_LR)
    rtd_cal.train()

    gen_cal = torch.Generator(device=device)
    gen_cal.manual_seed(SEED + 100)
    cal_losses = []
    step = 0
    for epoch in range(10):  # repeat train batches if needed
        for ids, amask, wgrp in train_batches:
            if step >= RTD_TRAIN_STEPS:
                break
            ids, amask, wgrp = ids.to(device), amask.to(device), wgrp.to(device)
            masked, labels, sel = apply_wwm(ids, amask, wgrp, tokenizer,
                                            MASK_PROB, gen_cal)
            hard_corr, hard_lbl, _ = generate_hard_corruptions(
                model, ids, masked, amask, sel, RTD_TEMPERATURE
            )
            with torch.no_grad():
                h = model.deberta(input_ids=hard_corr, attention_mask=amask
                                  ).last_hidden_state
            logits = rtd_cal(h)
            v = amask.bool()
            loss = F.cross_entropy(logits[v], hard_lbl[v])
            rtd_opt.zero_grad()
            loss.backward()
            rtd_opt.step()
            cal_losses.append(loss.item())
            step += 1
            if step % 20 == 0:
                print(f"  RTD head step {step}/{RTD_TRAIN_STEPS}: "
                      f"loss={loss.item():.4f}", flush=True)
        if step >= RTD_TRAIN_STEPS:
            break

    # Evaluate calibrated RTD head on held-out batches
    print("\n=== Phase B Eval: Calibrated discrimination ===", flush=True)
    rtd_cal.eval()
    gen_eval = torch.Generator(device=device)
    gen_eval.manual_seed(SEED + 200)
    gen_eval_r = torch.Generator(device=device)
    gen_eval_r.manual_seed(SEED + 201)

    cal_hard_metrics = []
    cal_rand_metrics = []

    for ids, amask, wgrp in eval_batches:
        ids, amask, wgrp = ids.to(device), amask.to(device), wgrp.to(device)
        masked, labels, sel = apply_wwm(ids, amask, wgrp, tokenizer,
                                        MASK_PROB, gen_eval)
        hard_corr, hard_lbl, _ = generate_hard_corruptions(
            model, ids, masked, amask, sel, RTD_TEMPERATURE
        )
        rand_corr, rand_lbl, _ = generate_random_corruptions(ids, sel, vs, gen_eval_r)

        with torch.no_grad():
            h_hard = model.deberta(input_ids=hard_corr, attention_mask=amask
                                   ).last_hidden_state
            h_rand = model.deberta(input_ids=rand_corr, attention_mask=amask
                                   ).last_hidden_state
            logits_h = rtd_cal(h_hard)
            logits_r = rtd_cal(h_rand)
        cal_hard_metrics.append(compute_rtd_metrics(logits_h, hard_lbl, amask))
        cal_rand_metrics.append(compute_rtd_metrics(logits_r, rand_lbl, amask))

    # Restore model grad state
    for p in model.parameters():
        p.requires_grad_(True)

    # ================================================================
    # Aggregate results
    # ================================================================
    def mn(lst):
        return sum(lst) / len(lst) if lst else 0.0

    def sd(lst):
        if len(lst) < 2:
            return 0.0
        m = mn(lst)
        return (sum((x - m) ** 2 for x in lst) / (len(lst) - 1)) ** 0.5

    # Gradient summary per layer
    all_layers = sorted(set(list(grad_norms_mlm) + list(grad_norms_rtd_hard)),
                        key=lambda k: (0 if k == "embedding" else
                                       1 if k.startswith("layer") else 2,
                                       k))
    grad_table = {}
    for k in all_layers:
        e = {}
        if k in grad_norms_mlm:
            e["mlm_norm"] = mn(grad_norms_mlm[k])
        if k in grad_norms_rtd_hard:
            e["rtd_hard_norm"] = mn(grad_norms_rtd_hard[k])
        if k in grad_norms_rtd_rand:
            e["rtd_rand_norm"] = mn(grad_norms_rtd_rand[k])
        if k in grad_cosines_hard:
            e["cos_mlm_rtd_hard"] = mn(grad_cosines_hard[k])
            e["cos_mlm_rtd_hard_std"] = sd(grad_cosines_hard[k])
        if k in grad_cosines_rand:
            e["cos_mlm_rtd_rand"] = mn(grad_cosines_rand[k])
        if "mlm_norm" in e and "rtd_hard_norm" in e and e["mlm_norm"] > 0:
            e["ratio_hard"] = e["rtd_hard_norm"] / e["mlm_norm"]
        if "mlm_norm" in e and "rtd_rand_norm" in e and e["mlm_norm"] > 0:
            e["ratio_rand"] = e["rtd_rand_norm"] / e["mlm_norm"]
        grad_table[k] = e

    # RTD accuracy summaries
    def agg_metrics(mlist):
        return {
            "accuracy": mn([m["accuracy"] for m in mlist]),
            "acc_replaced": mn([m["acc_replaced"] for m in mlist]),
            "acc_original": mn([m["acc_original"] for m in mlist]),
            "auroc": mn([m["auroc"] for m in mlist if m["auroc"] is not None]),
            "n": len(mlist),
        }

    # Corruption stats
    hard_repl_rate = mn([s["replacement_rate"] for s in hard_stats_all])
    hard_gen_acc = mn([s["generator_accuracy"] for s in hard_stats_all])

    result = {
        "status": "PROBE_COMPLETE",
        "config": {
            "checkpoint": str(CHECKPOINT),
            "n_batches_total": n_batches,
            "n_train": n_train, "n_eval": n_eval,
            "batch_size": BATCH_SIZE, "seq_length": SEQ_LENGTH,
            "mask_prob": MASK_PROB, "rtd_temperature": RTD_TEMPERATURE,
            "rtd_train_steps": RTD_TRAIN_STEPS,
            "num_examples": len(examples), "total_words": total_words,
        },
        "phase_a_gradient_geometry": {
            "per_layer": grad_table,
            "embedding_rtd_hard_norm_mean": mn(emb_rtd_norms),
            "embedding_rtd_hard_norm_std": sd(emb_rtd_norms),
            "embedding_mlm_norm_mean": mn(emb_mlm_norms),
            "embedding_rtd_mlm_ratio": (
                mn(emb_rtd_norms) / mn(emb_mlm_norms)
                if mn(emb_mlm_norms) > 0 else None
            ),
            "trunk_cosine_hard_mean": mn([
                mn(grad_cosines_hard[f"layer_{i}"])
                for i in range(n_layers) if f"layer_{i}" in grad_cosines_hard
            ]),
            "trunk_cosine_rand_mean": mn([
                mn(grad_cosines_rand[f"layer_{i}"])
                for i in range(n_layers) if f"layer_{i}" in grad_cosines_rand
            ]),
            "mlm_loss_mean": mn(mlm_losses),
            "rtd_hard_loss_mean": mn(rtd_hard_losses),
            "rtd_rand_loss_mean": mn(rtd_rand_losses),
        },
        "phase_a_untrained_rtd": {
            "hard": agg_metrics(raw_hard_metrics),
            "random": agg_metrics(raw_rand_metrics),
        },
        "corruption_stats": {
            "hard_replacement_rate": hard_repl_rate,
            "hard_generator_accuracy": hard_gen_acc,
        },
        "phase_b_calibrated_rtd": {
            "training_steps": step,
            "loss_first": cal_losses[0] if cal_losses else None,
            "loss_last": cal_losses[-1] if cal_losses else None,
            "hard": agg_metrics(cal_hard_metrics) if cal_hard_metrics else {},
            "random": agg_metrics(cal_rand_metrics) if cal_rand_metrics else {},
            "shortcut_gap_accuracy": (
                (agg_metrics(cal_rand_metrics)["accuracy"]
                 - agg_metrics(cal_hard_metrics)["accuracy"])
                if cal_hard_metrics and cal_rand_metrics else None
            ),
            "shortcut_gap_auroc": (
                (agg_metrics(cal_rand_metrics)["auroc"]
                 - agg_metrics(cal_hard_metrics)["auroc"])
                if cal_hard_metrics and cal_rand_metrics and
                agg_metrics(cal_hard_metrics)["auroc"] and
                agg_metrics(cal_rand_metrics)["auroc"] else None
            ),
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }

    # ── Interpretation ──
    interp = {}
    # Shortcut resistance
    if cal_hard_metrics and cal_rand_metrics:
        h_acc = agg_metrics(cal_hard_metrics)["accuracy"]
        r_acc = agg_metrics(cal_rand_metrics)["accuracy"]
        gap = r_acc - h_acc
        if gap > 0.10:
            interp["shortcut"] = (
                f"GOOD: Calibrated shortcut gap = {gap:.3f} (random {r_acc:.3f} vs "
                f"hard {h_acc:.3f}). Model-sampled corruptions require context."
            )
        elif gap > 0.03:
            interp["shortcut"] = (
                f"MODERATE: Shortcut gap = {gap:.3f}. Partial context requirement."
            )
        else:
            interp["shortcut"] = (
                f"WEAK: Shortcut gap = {gap:.3f}. Model-sampled corruptions may "
                f"be nearly as detectable as random ones."
            )
    # Gradient compatibility
    trunk_cos = result["phase_a_gradient_geometry"]["trunk_cosine_hard_mean"]
    neg_layers = sum(
        1 for i in range(n_layers)
        if f"layer_{i}" in grad_cosines_hard and mn(grad_cosines_hard[f"layer_{i}"]) < 0
    )
    if trunk_cos > 0.05:
        interp["gradient_compatibility"] = (
            f"POSITIVE: Trunk gradient cosine = {trunk_cos:.4f} "
            f"({neg_layers}/{n_layers} negative layers). "
            f"RTD and MLM gradients are broadly aligned in the encoder."
        )
    elif trunk_cos > -0.05:
        interp["gradient_compatibility"] = (
            f"ORTHOGONAL: Trunk gradient cosine = {trunk_cos:.4f} "
            f"({neg_layers}/{n_layers} negative). "
            f"RTD provides independent signal, no strong conflict."
        )
    else:
        interp["gradient_compatibility"] = (
            f"NEGATIVE: Trunk gradient cosine = {trunk_cos:.4f} "
            f"({neg_layers}/{n_layers} negative). "
            f"RTD fights MLM in the trunk. Careful GDES + layer routing needed."
        )
    # GDES
    emb_ratio = result["phase_a_gradient_geometry"]["embedding_rtd_mlm_ratio"]
    if emb_ratio is not None:
        if emb_ratio > 0.3:
            interp["gdes_necessity"] = (
                f"NEEDED: RTD embedding gradient = {emb_ratio:.3f}x MLM. "
                f"GDES blocking is critical to prevent embedding interference."
            )
        elif emb_ratio > 0.1:
            interp["gdes_necessity"] = (
                f"RECOMMENDED: RTD embedding gradient = {emb_ratio:.3f}x MLM. "
                f"GDES blocking is advisable."
            )
        else:
            interp["gdes_necessity"] = (
                f"OPTIONAL: RTD embedding gradient = {emb_ratio:.3f}x MLM. "
                f"Small relative to MLM embedding gradient."
            )
    # Generator quality
    interp["generator_quality"] = (
        f"At temperature {RTD_TEMPERATURE}, the 80M model correctly predicts "
        f"{hard_gen_acc:.1%} of masked tokens, leaving {hard_repl_rate:.1%} as "
        f"genuine replacements for RTD."
    )
    # Overall
    cal_hard_acc = (agg_metrics(cal_hard_metrics)["accuracy"]
                    if cal_hard_metrics else None)
    if cal_hard_acc is not None:
        if cal_hard_acc > 0.95:
            interp["overall"] = (
                "CONCERN: Calibrated hard RTD accuracy > 95%. "
                "Corruptions too easy; lower temperature or use frozen-earlier generator."
            )
        elif cal_hard_acc > 0.85:
            interp["overall"] = (
                f"GOOD: Calibrated hard RTD accuracy = {cal_hard_acc:.3f}. "
                f"The task is learnable but not trivial."
            )
        else:
            interp["overall"] = (
                f"PROMISING: Calibrated hard RTD accuracy = {cal_hard_acc:.3f}. "
                f"Substantial room for learning; gradient probe should drive the decision."
            )

    result["interpretation"] = interp

    # Save JSON
    out_json = OUT_DIR / "mlm_rtd_mechanism_probe.json"
    out_json.write_text(json.dumps(result, indent=2, default=str) + "\n")

    # Save readable summary
    lines = [
        "# research: MLM+RTD Gradient-Separated Mechanism Probe",
        "",
        f"Checkpoint: {CHECKPOINT.name} ({n_params:,} params)",
        f"Probe data: {len(examples)} examples, {total_words} words",
        f"Batches: {n_train} train + {n_eval} eval, batch {BATCH_SIZE}×seq{SEQ_LENGTH}",
        "",
        "## Phase A: Gradient Geometry (untrained RTD head)",
        f"MLM loss mean: {mn(mlm_losses):.4f}",
        f"RTD hard loss mean: {mn(rtd_hard_losses):.4f}",
        f"RTD random loss mean: {mn(rtd_rand_losses):.4f}",
        "",
        "### Per-layer gradient cosine (MLM vs RTD-hard) and norm ratio",
    ]
    for k in all_layers:
        e = grad_table[k]
        c = e.get("cos_mlm_rtd_hard", "—")
        r = e.get("ratio_hard", "—")
        if isinstance(c, float):
            c = f"{c:+.4f}"
        if isinstance(r, float):
            r = f"{r:.4f}"
        lines.append(f"  {k:12s}  cos={c}  norm_ratio={r}")

    lines.extend([
        "",
        f"Trunk cosine mean (hard): {trunk_cos:.4f}",
        f"Trunk cosine mean (rand): {result['phase_a_gradient_geometry']['trunk_cosine_rand_mean']:.4f}",
        f"Embedding RTD/MLM norm ratio: {emb_ratio if emb_ratio else '—'}",
        "",
        "## Corruption Statistics",
        f"Generator accuracy (T={RTD_TEMPERATURE}): {hard_gen_acc:.1%}",
        f"Hard replacement rate: {hard_repl_rate:.1%}",
        "",
        "## Phase B: Calibrated RTD Head ({step} steps frozen-encoder training)",
        f"RTD head loss: {cal_losses[0]:.4f} → {cal_losses[-1]:.4f}"
        if cal_losses else "  No training",
    ])
    if cal_hard_metrics and cal_rand_metrics:
        ch = agg_metrics(cal_hard_metrics)
        cr = agg_metrics(cal_rand_metrics)
        lines.extend([
            f"Hard:   acc={ch['accuracy']:.4f}  repl_acc={ch['acc_replaced']:.4f}  "
            f"auroc={ch['auroc']:.4f}" if ch['auroc'] else
            f"Hard:   acc={ch['accuracy']:.4f}  repl_acc={ch['acc_replaced']:.4f}",
            f"Random: acc={cr['accuracy']:.4f}  repl_acc={cr['acc_replaced']:.4f}  "
            f"auroc={cr['auroc']:.4f}" if cr['auroc'] else
            f"Random: acc={cr['accuracy']:.4f}  repl_acc={cr['acc_replaced']:.4f}",
            f"Shortcut gap (random - hard): "
            f"{result['phase_b_calibrated_rtd']['shortcut_gap_accuracy']:.4f}",
        ])

    lines.extend(["", "## Interpretation"])
    for k, v in interp.items():
        lines.append(f"  {k}: {v}")
    lines.append(f"\nElapsed: {result['elapsed_sec']:.1f}s")

    out_md = (OUT_DIR.parents[4] / 'research/documents/frontier_consolidation/data/mlm_rtd_mechanism_probe/mlm_rtd_mechanism_probe.md')
    out_md.write_text("\n".join(lines) + "\n")

    print(f"\nResults: {out_json}")
    print(f"Summary: {out_md}")
    print(json.dumps({k: result[k] for k in ("status", "elapsed_sec")}, indent=2))


if __name__ == "__main__":
    main()
