#!/usr/bin/env python3
"""research: compare RTD-MLM interaction across existing model states.

This is a low-cost probe requested before any second RTD training run.  It
reuses existing checkpoints and applies the same balanced hard-RTD head-fitting,
shortcut-resistance, and MLM-vs-RTD gradient geometry measurement to:
  - research legal MLM 20M
  - research legal MLM 80M
  - research from-scratch MLM+RTD-GDES 20M encoder

The purpose is to test whether model maturity changes the RTD interaction in a
way that can explain why the mature 80M probe looked favorable while the
from-scratch 20M score screen rotated competence.
"""
from __future__ import annotations

import importlib.util
import json
import math
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

OUT_DIR = pathlib.Path("experiments/archive/frontier_consolidation/data/state_dependent_rtd_probe")

CHECKPOINTS = {
    "mlm20_step35": pathlib.Path(
        "experiments/archive/frontier_consolidation/training/runs"
        "complianttok_reinvest_seed43022_r2/hf_model/chck_20M"
    ),
    "mlm80_step35": pathlib.Path(
        "experiments/archive/frontier_consolidation/training/runs"
        "complianttok_reinvest_seed43022_r2/hf_model/chck_80M"
    ),
    "rtd20_step094": pathlib.Path(
        "experiments/archive/frontier_consolidation/training/runs"
        "mlm_rtd_lambda1_seed43022_20M/hf_model/chck_20M"
    ),
}

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
    order = torch.argsort(p)
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, len(p) + 1, device=p.device, dtype=torch.float)
    pos_rank = ranks[y == 1].sum().item()
    auc = (pos_rank - n1 * (n1 + 1) / 2) / max(1, n1 * n0)
    return {
        "recall_original": rec0,
        "recall_replaced": rec1,
        "balanced_accuracy": 0.5 * (rec0 + rec1),
        "auroc": auc,
        "n_original": n0,
        "n_replaced": n1,
    }


def aggregate(items):
    keys = ["recall_original", "recall_replaced", "balanced_accuracy", "auroc"]
    return {k: sum(m[k] for m in items) / len(items) for k in keys} | {"n_batches": len(items)}


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def train_and_probe_one(label, checkpoint, batches, n_train, device):
    torch.manual_seed(s.SEED + 290)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(s.SEED + 290)
    tok = AutoTokenizer.from_pretrained(str(checkpoint), use_fast=True)
    model = DebertaV2ForMaskedLM.from_pretrained(str(checkpoint)).to(device)
    model.eval()
    train_batches = batches[:n_train]
    eval_batches = batches[n_train:]

    # Freeze encoder and fit a fresh balanced RTD head for this state.
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
            if step >= TRAIN_STEPS:
                break
            ids, am, wg = ids.to(device), am.to(device), wg.to(device)
            masked, _, sel = s.apply_wwm(ids, am, wg, tok, s.MASK_PROB, gen)
            corr, lbl, _ = s.generate_hard_corruptions(model, ids, masked, am, sel, s.RTD_TEMPERATURE)
            with torch.no_grad():
                h = model.deberta(input_ids=corr, attention_mask=am).last_hidden_state
            logits = head(h)
            loss, wt = balanced_ce(logits, lbl, am.bool())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
            weights_log.append([float(x) for x in wt.detach().cpu().tolist()])
            step += 1
            if step % 20 == 0:
                print(json.dumps({"event": "head_fit", "checkpoint": label, "step": step,
                                  "loss": round(losses[-1], 5), "weights": weights_log[-1]}), flush=True)
        if step >= TRAIN_STEPS:
            break

    # Held-out hard and random corruption discrimination.
    head.eval()
    gh = torch.Generator(device=device); gh.manual_seed(s.SEED + 400)
    gr = torch.Generator(device=device); gr.manual_seed(s.SEED + 401)
    hard_ms, rand_ms = [], []
    replacement_rates = []
    generator_accs = []
    for ids, am, wg in eval_batches:
        ids, am, wg = ids.to(device), am.to(device), wg.to(device)
        masked, _, sel = s.apply_wwm(ids, am, wg, tok, s.MASK_PROB, gh)
        ch, lh, st = s.generate_hard_corruptions(model, ids, masked, am, sel, s.RTD_TEMPERATURE)
        cr, lr, _ = s.generate_random_corruptions(ids, sel, model.config.vocab_size, gr)
        with torch.no_grad():
            hh = model.deberta(input_ids=ch, attention_mask=am).last_hidden_state
            hr = model.deberta(input_ids=cr, attention_mask=am).last_hidden_state
            hard_ms.append(metrics(head(hh), lh, am.bool()))
            rand_ms.append(metrics(head(hr), lr, am.bool()))
        replacement_rates.append(st["replacement_rate"])
        generator_accs.append(st["generator_accuracy"])

    # Gradient geometry using the trained head. Keep model in eval mode to remove dropout noise.
    for p in model.parameters():
        p.requires_grad_(True)
    for p in head.parameters():
        p.requires_grad_(False)
    gm = torch.Generator(device=device); gm.manual_seed(s.SEED + 500)
    cos_by_group = defaultdict(list)
    norm_m = defaultdict(list)
    norm_r = defaultdict(list)
    rtd_losses = []
    mlm_losses = []
    for ids, am, wg in train_batches[:GRAD_BATCHES]:
        ids, am, wg = ids.to(device), am.to(device), wg.to(device)
        masked, labels, sel = s.apply_wwm(ids, am, wg, tok, s.MASK_PROB, gm)

        model.zero_grad(set_to_none=True); head.zero_grad(set_to_none=True)
        om = model(input_ids=masked, attention_mask=am, labels=labels)
        om.loss.backward()
        vm = s.get_grad_vectors(model)
        mlm_losses.append(float(om.loss.detach().cpu()))

        corr, lbl, _ = s.generate_hard_corruptions(model, ids, masked, am, sel, s.RTD_TEMPERATURE)
        model.zero_grad(set_to_none=True); head.zero_grad(set_to_none=True)
        h = model.deberta(input_ids=corr, attention_mask=am).last_hidden_state
        logits = head(h)
        lrtd, _ = balanced_ce(logits, lbl, am.bool())
        lrtd.backward()
        vr = s.get_grad_vectors(model)
        rtd_losses.append(float(lrtd.detach().cpu()))

        for k in vm.keys() & vr.keys():
            if vm[k].shape != vr[k].shape:
                continue
            cos_by_group[k].append(F.cosine_similarity(vm[k][None], vr[k][None]).item())
            norm_m[k].append(vm[k].norm().item())
            norm_r[k].append(vr[k].norm().item())

    per_group = {}
    for k in sorted(cos_by_group):
        mm = mean(norm_m[k])
        rr = mean(norm_r[k])
        per_group[k] = {
            "cosine": mean(cos_by_group[k]),
            "mlm_norm": mm,
            "rtd_balanced_norm": rr,
            "rtd_mlm_norm_ratio": rr / mm if mm and mm > 0 else None,
        }
    trunk_layers = [f"layer_{i}" for i in range(model.config.num_hidden_layers)]
    trunk_cos = mean([per_group[k]["cosine"] for k in trunk_layers if k in per_group])
    trunk_ratio = mean([per_group[k]["rtd_mlm_norm_ratio"] for k in trunk_layers if k in per_group])
    ah, ar = aggregate(hard_ms), aggregate(rand_ms)
    out = {
        "label": label,
        "checkpoint": str(checkpoint),
        "balanced_head": {
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "class_weights_last": weights_log[-1],
            "heldout_hard": ah,
            "heldout_random": ar,
            "random_minus_hard_balanced_accuracy": ar["balanced_accuracy"] - ah["balanced_accuracy"],
            "random_minus_hard_auroc": ar["auroc"] - ah["auroc"],
            "hard_replacement_rate": mean(replacement_rates),
            "hard_generator_accuracy": mean(generator_accs),
        },
        "trained_head_gradient_geometry": {
            "per_group": per_group,
            "trunk_cosine_mean": trunk_cos,
            "trunk_norm_ratio_mean": trunk_ratio,
            "mlm_loss_mean": mean(mlm_losses),
            "balanced_rtd_loss_mean": mean(rtd_losses),
        },
    }
    # Directional quantities useful for deciding if the mature-tail hypothesis gains support.
    r = trunk_ratio or 0.0
    c = trunk_cos or 0.0
    out["joint_update_at_lambda1"] = {
        "norm_multiplier": math.sqrt(max(0.0, 1 + r * r + 2 * r * c)),
        "rotation_degrees": math.degrees(math.atan2(r * math.sqrt(max(0.0, 1 - c * c)), 1 + r * c)) if r else 0.0,
    }
    del model, head, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    examples, total_words = s.load_data(s.STREAM, s.NUM_PROBE_WORDS)
    # Use one tokenizer only to form the fixed probe batches; all three checkpoints use the same tokenizer.
    base_tok = AutoTokenizer.from_pretrained(str(CHECKPOINTS["mlm80_step35"]), use_fast=True)
    ds = s.ProbeDataset(examples, base_tok, s.SEQ_LENGTH)
    loader = DataLoader(ds, batch_size=s.BATCH_SIZE, shuffle=False, collate_fn=s.collate_probe, num_workers=0)
    batches = list(loader)
    n_train = int(len(batches) * 0.75)
    result = {
        "status": "STATE_DEPENDENT_RTD_PROBE_COMPLETE",
        "purpose": "Compare identical balanced RTD shortcut-resistance and gradient-geometry probes across existing 20M/80M MLM states and the RTD-trained 20M state before admitting any delayed RTD tail training.",
        "config": {
            "probe_words": total_words,
            "n_examples": len(examples),
            "n_batches": len(batches),
            "train_batches": n_train,
            "eval_batches": len(batches) - n_train,
            "head_train_steps": TRAIN_STEPS,
            "gradient_batches": GRAD_BATCHES,
            "mask_prob": s.MASK_PROB,
            "rtd_temperature": s.RTD_TEMPERATURE,
            "seed_base": s.SEED,
        },
        "checkpoints": {},
    }
    for label, cp in CHECKPOINTS.items():
        if not cp.exists():
            raise FileNotFoundError(f"checkpoint missing: {label}: {cp}")
        print(json.dumps({"event": "start_checkpoint", "label": label, "path": str(cp)}), flush=True)
        result["checkpoints"][label] = train_and_probe_one(label, cp, batches, n_train, device)
        jp_part = OUT_DIR / f"{label}.json"
        jp_part.write_text(json.dumps(result["checkpoints"][label], indent=2) + "\n")
        print(json.dumps({"event": "finish_checkpoint", "label": label,
                          "hard_auroc": result["checkpoints"][label]["balanced_head"]["heldout_hard"]["auroc"],
                          "trunk_cosine": result["checkpoints"][label]["trained_head_gradient_geometry"]["trunk_cosine_mean"],
                          "trunk_ratio": result["checkpoints"][label]["trained_head_gradient_geometry"]["trunk_norm_ratio_mean"]}), flush=True)

    # Comparisons that directly answer the state-dependence question.
    def grab(label):
        x = result["checkpoints"][label]
        return {
            "hard_bal_acc": x["balanced_head"]["heldout_hard"]["balanced_accuracy"],
            "hard_auroc": x["balanced_head"]["heldout_hard"]["auroc"],
            "random_minus_hard_auroc": x["balanced_head"]["random_minus_hard_auroc"],
            "replacement_rate": x["balanced_head"]["hard_replacement_rate"],
            "trunk_cosine": x["trained_head_gradient_geometry"]["trunk_cosine_mean"],
            "trunk_ratio": x["trained_head_gradient_geometry"]["trunk_norm_ratio_mean"],
            "rel_cosine": x["trained_head_gradient_geometry"]["per_group"].get("rel_embeddings", {}).get("cosine"),
            "embedding_cosine": x["trained_head_gradient_geometry"]["per_group"].get("embedding", {}).get("cosine"),
            "rotation_degrees": x["joint_update_at_lambda1"]["rotation_degrees"],
        }
    summary = {k: grab(k) for k in CHECKPOINTS}
    result["summary_table"] = summary
    result["comparisons"] = {
        "mlm80_minus_mlm20": {k: summary["mlm80_step35"][k] - summary["mlm20_step35"][k]
                               for k in summary["mlm20_step35"] if summary["mlm20_step35"][k] is not None},
        "rtd20_minus_mlm20": {k: summary["rtd20_step094"][k] - summary["mlm20_step35"][k]
                               for k in summary["mlm20_step35"] if summary["mlm20_step35"][k] is not None},
    }
    result["elapsed_sec"] = round(time.time() - t0, 1)

    # Compact scientific reading without pretending this probe alone scores BabyLM.
    cmp80 = result["comparisons"]["mlm80_minus_mlm20"]
    cmprtd = result["comparisons"]["rtd20_minus_mlm20"]
    result["interpretation"] = {
        "maturity_shift": (
            f"80M vs 20M changes trunk cosine by {cmp80['trunk_cosine']:+.4f}, "
            f"RTD/MLM trunk norm ratio by {cmp80['trunk_ratio']:+.4f}, hard AUROC by {cmp80['hard_auroc']:+.4f}, "
            f"and random-hard AUROC gap by {cmp80['random_minus_hard_auroc']:+.4f}."),
        "rtd20_state_shift": (
            f"The RTD-trained 20M encoder differs from matched MLM20 by trunk cosine {cmprtd['trunk_cosine']:+.4f}, "
            f"hard AUROC {cmprtd['hard_auroc']:+.4f}, random-hard AUROC gap {cmprtd['random_minus_hard_auroc']:+.4f}, "
            f"and lambda1 update rotation {cmprtd['rotation_degrees']:+.2f} degrees."),
    }

    jp = OUT_DIR / "state_dependent_rtd_probe.json"
    jp.write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# research state-dependent RTD/MLM probe", "",
             "Purpose: compare the same balanced hard-RTD head-fitting, shortcut-resistance, and MLM-vs-RTD gradient geometry across existing checkpoints before any delayed RTD tail training.", "",
             f"Probe data: {len(examples)} examples, {total_words} words, {len(batches)} batches ({n_train} head-fit batches, {len(batches)-n_train} held-out batches).", "",
             "## Summary", "",
             "| checkpoint | hard bal acc | hard AUROC | random-hard AUROC gap | repl rate | trunk cosine | trunk RTD/MLM | rel cos | lambda1 rotation |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label in CHECKPOINTS:
        row = summary[label]
        lines.append(
            f"| {label} | {row['hard_bal_acc']:.4f} | {row['hard_auroc']:.4f} | "
            f"{row['random_minus_hard_auroc']:.4f} | {row['replacement_rate']:.4f} | "
            f"{row['trunk_cosine']:+.4f} | {row['trunk_ratio']:.4f} | "
            f"{row['rel_cosine']:+.4f} | {row['rotation_degrees']:.2f}° |"
        )
    lines += ["", "## Direct comparisons", "",
              "| comparison | Δhard AUROC | Δrandom-hard AUROC gap | Δtrunk cosine | Δtrunk RTD/MLM | Δrotation |",
              "|---|---:|---:|---:|---:|---:|"]
    for cname, comp in result["comparisons"].items():
        lines.append(
            f"| {cname} | {comp['hard_auroc']:+.4f} | {comp['random_minus_hard_auroc']:+.4f} | "
            f"{comp['trunk_cosine']:+.4f} | {comp['trunk_ratio']:+.4f} | {comp['rotation_degrees']:+.2f}° |"
        )
    lines += ["", "## Interpretation", ""]
    for v in result["interpretation"].values():
        lines.append(f"- {v}")
    lines += ["", f"Elapsed: {result['elapsed_sec']:.1f}s", ""]
    mp = (OUT_DIR.parents[4] / 'research/documents/frontier_consolidation/data/state_dependent_rtd_probe/state_dependent_rtd_probe.md')
    mp.write_text("\n".join(lines))
    print(json.dumps({"status": result["status"], "out_json": str(jp), "out_md": str(mp),
                      "elapsed_sec": result["elapsed_sec"], "summary_table": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
