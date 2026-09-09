#!/usr/bin/env python3
"""CPU pilot for research equality-router pretraining.

Runs only the explicit token/query equality objective on the current CharGRU +
neither gate and evaluates train-name and held-name routing. This determines
whether the equality-aux relational experiment is worth a GPU launch.
"""
from __future__ import annotations
import json, random, time
from pathlib import Path

import torch

from experiments.archive.representation_and_objectives.ai_lab.scripts import raw_span_discovery_probe as base
from experiments.archive.representation_and_objectives.ai_lab.scripts import equality_aux_span_probe as eqaux

PROJECT = Path("experiments/archive/representation_and_objectives")
OUT = PROJECT / "data/eq_pretrain_pilot"


def unique_event_examples(states, comps):
    seen = set(); out = []
    for q in states:
        if q.is_changed:
            k = ("state", q.event, tuple(q.names))
            if k not in seen:
                seen.add(k); out.append(k)
    for c in comps:
        for tag, ev in [("cmp1", c.event1), ("cmp2", c.event2)]:
            k = (tag, ev, tuple(c.names))
            if k not in seen:
                seen.add(k); out.append(k)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data_root = base.DEFAULT_DATA_ROOT
    arm = base.DEFAULT_ARM
    seed = 29600
    torch.manual_seed(seed); random.seed(seed)
    ts, tc, es, ec, _, _ = base.load_dataset(data_root, arm)
    train_examples = unique_event_examples(ts, tc)
    eval_examples = unique_event_examples(es, ec)
    vocab = eqaux.build_vocab(ts, tc, no_name_vocab=True)
    model = base.create_paired_span_models(len(vocab.itos), ["shared_trunk"], seed, 48, 64, 16, 16)["shared_trunk"]
    device = torch.device("cpu")
    model.to(device)
    checkpoints = [0, 1, 5, 10, 25, 50, 100]
    rows = []
    rng = random.Random(12345)
    params = eqaux.matcher_parameters(model, include_static=False)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=5e-3, weight_decay=0.0)
    t0 = time.time()
    for ep in range(0, max(checkpoints)+1):
        if ep in checkpoints:
            tr = eqaux.equality_eval(model, train_examples, device, include_static=False)
            ev = eqaux.equality_eval(model, eval_examples, device, include_static=False)
            rec = {"epoch": ep, "elapsed_seconds": time.time()-t0,
                   "train": tr, "eval": ev}
            rows.append(rec)
            print(json.dumps(rec, sort_keys=True), flush=True)
        if ep == max(checkpoints):
            break
        exs = list(train_examples); rng.shuffle(exs)
        model.train(); opt.zero_grad(set_to_none=True)
        losses = []
        for trk in eqaux.unique_matcher_trunks(model, include_static=False):
            for _, text, names in exs:
                losses.append(eqaux.equality_loss_for_text(trk, text, names, device, null_weight=1.0))
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
    (OUT / "eq_pretrain_pilot.json").write_text(json.dumps({"rows": rows, "n_train_examples": len(train_examples), "n_eval_examples": len(eval_examples), "vocab_size": len(vocab.itos)}, indent=2, sort_keys=True)+"\n")
    lines = ["# research equality pretraining pilot", "", f"Train examples: {len(train_examples)}; eval examples: {len(eval_examples)}; vocab size: {len(vocab.itos)}", "",
             "| epoch | train both gate | eval both gate | train cand>other | eval cand>other | train neither(cand) | eval neither(cand) | elapsed s |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        tr = r["train"]; ev = r["eval"]
        lines.append("| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.1f} |".format(
            r["epoch"], tr.get("both_wins_gate",0), ev.get("both_wins_gate",0), tr.get("cand_gt_other",0), ev.get("cand_gt_other",0), tr.get("mean_cand_p_neither",0), ev.get("mean_cand_p_neither",0), r["elapsed_seconds"]))
    ((OUT.parents[4] / 'research/documents/representation_and_objectives/data/eq_pretrain_pilot/eq_pretrain_pilot.md')).write_text("\n".join(lines)+"\n")
    print(json.dumps({"status":"EQ_PRETRAIN_PILOT_COMPLETE", "md": str((OUT.parents[4] / 'research/documents/representation_and_objectives/data/eq_pretrain_pilot/eq_pretrain_pilot.md'))}, indent=2), flush=True)

if __name__ == "__main__":
    main()
