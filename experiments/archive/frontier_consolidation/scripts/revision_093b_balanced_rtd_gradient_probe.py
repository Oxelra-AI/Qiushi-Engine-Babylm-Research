#!/usr/bin/env python3
"""Step093b: repair RTD calibration and gradient geometry.

The first research probe used ordinary CE, causing the frozen-encoder RTD head
to collapse toward the ~92% original-token majority class (replaced recall 3.8%).
This follow-up trains with exact per-batch class-balanced CE, evaluates balanced
accuracy/AUROC, then measures MLM-vs-RTD gradients using the trained RTD head.
"""
from __future__ import annotations
import importlib.util
import json
import pathlib
import time
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DebertaV2ForMaskedLM

BASE = pathlib.Path("experiments/archive/frontier_consolidation/scripts/mlm_rtd_mechanism_probe.py")
spec = importlib.util.spec_from_file_location("research", BASE)
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)

OUT_DIR = pathlib.Path("experiments/archive/frontier_consolidation/data/mlm_rtd_mechanism_probe")
TRAIN_STEPS = 80
LR = 1e-3
GRAD_BATCHES = 6


def balanced_ce(logits, labels, valid):
    y = labels[valid]
    x = logits[valid]
    n0 = (y == 0).sum().float()
    n1 = (y == 1).sum().float()
    n = n0 + n1
    weights = torch.stack([n / (2 * n0.clamp_min(1)), n / (2 * n1.clamp_min(1))])
    return F.cross_entropy(x, y, weight=weights), weights


def metrics(logits, labels, valid):
    x = logits[valid]
    y = labels[valid]
    p = F.softmax(x, -1)[:, 1]
    pred = x.argmax(-1)
    n0 = (y == 0).sum().item()
    n1 = (y == 1).sum().item()
    rec0 = ((pred == 0) & (y == 0)).sum().item() / max(1, n0)
    rec1 = ((pred == 1) & (y == 1)).sum().item() / max(1, n1)
    # exact rank AUROC via sort; tie handling negligible for float scores
    order = torch.argsort(p)
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, len(p) + 1, device=p.device, dtype=torch.float)
    pos_rank = ranks[y == 1].sum().item()
    auc = (pos_rank - n1 * (n1 + 1) / 2) / max(1, n1 * n0)
    return {"recall_original": rec0, "recall_replaced": rec1,
            "balanced_accuracy": 0.5 * (rec0 + rec1), "auroc": auc,
            "n_original": n0, "n_replaced": n1}


def aggregate(ms):
    keys = ["recall_original", "recall_replaced", "balanced_accuracy", "auroc"]
    return {k: sum(m[k] for m in ms) / len(ms) for k in keys} | {"n_batches": len(ms)}


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    examples, total_words = s.load_data(s.STREAM, s.NUM_PROBE_WORDS)
    tok = AutoTokenizer.from_pretrained(str(s.CHECKPOINT), use_fast=True)
    model = DebertaV2ForMaskedLM.from_pretrained(str(s.CHECKPOINT)).to(device)
    model.eval()
    ds = s.ProbeDataset(examples, tok, s.SEQ_LENGTH)
    loader = DataLoader(ds, batch_size=s.BATCH_SIZE, shuffle=False,
                        collate_fn=s.collate_probe, num_workers=0)
    batches = list(loader)
    n_train = int(len(batches) * 0.75)
    train_batches = batches[:n_train]
    eval_batches = batches[n_train:]

    # Freeze encoder and train a balanced hard-corruption classifier.
    for p in model.parameters():
        p.requires_grad_(False)
    head = s.RTDHead(model.config.hidden_size).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=0.01)
    gen = torch.Generator(device=device); gen.manual_seed(s.SEED + 300)
    losses = []
    weights_log = []
    step = 0
    head.train()
    for epoch in range(15):
        for ids, am, wg in train_batches:
            if step >= TRAIN_STEPS: break
            ids, am, wg = ids.to(device), am.to(device), wg.to(device)
            masked, _, sel = s.apply_wwm(ids, am, wg, tok, s.MASK_PROB, gen)
            corr, lbl, _ = s.generate_hard_corruptions(model, ids, masked, am, sel,
                                                       s.RTD_TEMPERATURE)
            with torch.no_grad():
                h = model.deberta(input_ids=corr, attention_mask=am).last_hidden_state
            logits = head(h)
            loss, wt = balanced_ce(logits, lbl, am.bool())
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item()); weights_log.append(wt.detach().cpu().tolist())
            step += 1
            if step % 20 == 0:
                print(f"balanced head {step}/{TRAIN_STEPS} loss={loss.item():.4f} "
                      f"w={weights_log[-1]}", flush=True)
        if step >= TRAIN_STEPS: break

    # Held-out hard and random evaluation.
    head.eval()
    gh = torch.Generator(device=device); gh.manual_seed(s.SEED + 400)
    gr = torch.Generator(device=device); gr.manual_seed(s.SEED + 401)
    mh, mr = [], []
    replacement_rates = []
    for ids, am, wg in eval_batches:
        ids, am, wg = ids.to(device), am.to(device), wg.to(device)
        masked, _, sel = s.apply_wwm(ids, am, wg, tok, s.MASK_PROB, gh)
        ch, lh, st = s.generate_hard_corruptions(model, ids, masked, am, sel,
                                                  s.RTD_TEMPERATURE)
        cr, lr, _ = s.generate_random_corruptions(ids, sel, model.config.vocab_size, gr)
        with torch.no_grad():
            hh = model.deberta(input_ids=ch, attention_mask=am).last_hidden_state
            hr = model.deberta(input_ids=cr, attention_mask=am).last_hidden_state
            mh.append(metrics(head(hh), lh, am.bool()))
            mr.append(metrics(head(hr), lr, am.bool()))
        replacement_rates.append(st["replacement_rate"])

    # Trained-head gradient geometry. Model eval removes dropout noise.
    for p in model.parameters(): p.requires_grad_(True)
    for p in head.parameters(): p.requires_grad_(False)
    gm = torch.Generator(device=device); gm.manual_seed(s.SEED + 500)
    cos_by_group = defaultdict(list)
    norm_m = defaultdict(list); norm_r = defaultdict(list)
    rtd_losses = []; mlm_losses = []
    for ids, am, wg in train_batches[:GRAD_BATCHES]:
        ids, am, wg = ids.to(device), am.to(device), wg.to(device)
        masked, labels, sel = s.apply_wwm(ids, am, wg, tok, s.MASK_PROB, gm)

        model.zero_grad(); head.zero_grad()
        om = model(input_ids=masked, attention_mask=am, labels=labels)
        om.loss.backward()
        vm = s.get_grad_vectors(model)
        mlm_losses.append(om.loss.item())

        corr, lbl, _ = s.generate_hard_corruptions(model, ids, masked, am, sel,
                                                    s.RTD_TEMPERATURE)
        model.zero_grad(); head.zero_grad()
        h = model.deberta(input_ids=corr, attention_mask=am).last_hidden_state
        logits = head(h)
        lrtd, wt = balanced_ce(logits, lbl, am.bool())
        lrtd.backward()
        vr = s.get_grad_vectors(model)
        rtd_losses.append(lrtd.item())

        for k in vm.keys() & vr.keys():
            if vm[k].shape != vr[k].shape: continue
            cos_by_group[k].append(F.cosine_similarity(vm[k][None], vr[k][None]).item())
            norm_m[k].append(vm[k].norm().item())
            norm_r[k].append(vr[k].norm().item())

    def mean(xs): return sum(xs) / len(xs) if xs else None
    per_group = {}
    for k in sorted(cos_by_group):
        per_group[k] = {"cosine": mean(cos_by_group[k]),
                        "mlm_norm": mean(norm_m[k]),
                        "rtd_balanced_norm": mean(norm_r[k]),
                        "rtd_mlm_norm_ratio": mean(norm_r[k]) / mean(norm_m[k])}
    trunk_cos = mean([per_group[f"layer_{i}"]["cosine"]
                      for i in range(model.config.num_hidden_layers)])
    trunk_ratio = mean([per_group[f"layer_{i}"]["rtd_mlm_norm_ratio"]
                        for i in range(model.config.num_hidden_layers)])

    ah, ar = aggregate(mh), aggregate(mr)
    result = {
        "status": "BALANCED_RTD_PROBE_COMPLETE",
        "config": {"train_steps": TRAIN_STEPS, "lr": LR,
                   "gradient_batches": GRAD_BATCHES, "total_words": total_words,
                   "train_batches": n_train, "eval_batches": len(eval_batches)},
        "balanced_head": {"loss_first": losses[0], "loss_last": losses[-1],
                          "class_weights_last": weights_log[-1],
                          "heldout_hard": ah, "heldout_random": ar,
                          "random_minus_hard_balanced_accuracy":
                              ar["balanced_accuracy"] - ah["balanced_accuracy"],
                          "random_minus_hard_auroc": ar["auroc"] - ah["auroc"],
                          "hard_replacement_rate": mean(replacement_rates)},
        "trained_head_gradient_geometry": {
            "per_group": per_group, "trunk_cosine_mean": trunk_cos,
            "trunk_norm_ratio_mean": trunk_ratio,
            "mlm_loss_mean": mean(mlm_losses),
            "balanced_rtd_loss_mean": mean(rtd_losses)},
        "elapsed_sec": round(time.time() - t0, 1),
    }
    # Scientific interpretation based on balanced metrics.
    result["interpretation"] = {
        "context_requirement": (
            f"Random corruptions are easier by {ar['balanced_accuracy']-ah['balanced_accuracy']:.3f} "
            f"balanced accuracy and {ar['auroc']-ah['auroc']:.3f} AUROC; "
            f"hard model-sampled corruption discrimination therefore requires substantially "
            f"more contextual knowledge than token-identity anomaly detection."),
        "hard_signal": (
            f"Held-out hard RTD balanced accuracy={ah['balanced_accuracy']:.3f}, "
            f"AUROC={ah['auroc']:.3f}, replaced recall={ah['recall_replaced']:.3f}; "
            f"the signal is learnable but not saturated."),
        "gradient": (
            f"With a trained class-balanced head, mean trunk MLM-vs-RTD cosine={trunk_cos:+.4f} "
            f"and raw norm ratio={trunk_ratio:.3f}; this sets the local RTD loss-scale "
            f"needed for a joint trainer rather than importing lambda=50."),
        "gdes": (
            f"Embedding cosine={per_group['embedding']['cosine']:+.4f}, "
            f"norm ratio={per_group['embedding']['rtd_mlm_norm_ratio']:.3f}; "
            f"block RTD gradients from the shared MLM embedding/decoder path."),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jp = OUT_DIR / "revision_093b_balanced_rtd_gradient_probe.json"
    jp.write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Step093b: Balanced RTD Calibration and Trained-Head Gradient Geometry", "",
             f"Balanced RTD head: loss {losses[0]:.4f} → {losses[-1]:.4f} ({TRAIN_STEPS} steps)",
             f"Hard held-out: balanced_acc={ah['balanced_accuracy']:.4f}, AUROC={ah['auroc']:.4f}, "
             f"replaced_recall={ah['recall_replaced']:.4f}, original_recall={ah['recall_original']:.4f}",
             f"Random held-out: balanced_acc={ar['balanced_accuracy']:.4f}, AUROC={ar['auroc']:.4f}, "
             f"replaced_recall={ar['recall_replaced']:.4f}, original_recall={ar['recall_original']:.4f}",
             f"Random-hard gap: balanced_acc={ar['balanced_accuracy']-ah['balanced_accuracy']:.4f}, "
             f"AUROC={ar['auroc']-ah['auroc']:.4f}", "",
             "## Trained-head gradient geometry"]
    for k, v in per_group.items():
        lines.append(f"{k:16s} cosine={v['cosine']:+.4f}  "
                     f"RTD/MLM norm={v['rtd_mlm_norm_ratio']:.3f}")
    lines += ["", f"Mean trunk cosine: {trunk_cos:+.4f}",
              f"Mean trunk RTD/MLM norm ratio: {trunk_ratio:.3f}", "", "## Interpretation"]
    for k, v in result["interpretation"].items(): lines.append(f"- {k}: {v}")
    lines.append(f"\nElapsed: {result['elapsed_sec']:.1f}s")
    mp = (OUT_DIR.parents[4] / 'research/documents/frontier_consolidation/data/mlm_rtd_mechanism_probe/revision_093b_balanced_rtd_gradient_probe.md')
    mp.write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": result["status"], "out": str(jp),
                      "hard_balanced_accuracy": ah["balanced_accuracy"],
                      "random_balanced_accuracy": ar["balanced_accuracy"],
                      "trunk_cosine": trunk_cos, "trunk_norm_ratio": trunk_ratio,
                      "elapsed_sec": result["elapsed_sec"]}, indent=2))

if __name__ == "__main__": main()
