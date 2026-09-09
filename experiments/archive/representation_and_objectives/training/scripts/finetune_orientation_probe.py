#!/usr/bin/env python3
"""research: full fine-tuning orientation probe on repaired research substrate.

Scientific purpose
------------------
Test whether fine-tuning a pretrained DeBERTa encoder on sparse bridge state 
evidence produces opposite role orientations on mixed held-seen comparisons.

The frozen-encoder probe showed that pretrained [CLS] representations
do NOT carry transferable role-slot information: a linear head on frozen reps
overfits training names (0.96) but anti-transfers to eval names (hh ≈ 0.33).

This script tests whether fine-tuning enables the encoder to learn abstract 
role representations that generalize across name sets AND show bridge-controlled 
orientation. The repaired research surface eliminates the research 0.750 order 
shortcut via active/passive counterbalancing.

Design
------
For each arm × seed:
  1. Load pretrained DeBERTa + fresh linear head (same init per seed)
  2. Fine-tune ALL parameters on arm's supervised data
  3. Evaluate on all 5 eval suites

Critical prediction:
  - aligned mixed > heldheld_only mixed (true orientation from bridge)
  - inverted mixed < heldheld_only mixed (opposite orientation from bridge)  
  - aligned - inverted > 0 (bridge controls orientation direction)
  - neutral ≈ heldheld_only (irrelevant evidence doesn't help)
  - pair_both: changed + unchanged both correct (conservation)

Controls:
  - heldheld_only: comparison-only (no bridge, calibration baseline)
  - exposure_only: random head (prior baseline)
  - neutral_decoupled: irrelevant bridge evidence
"""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import os, json, sys, random, copy, time, gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from collections import defaultdict

ARMS = [
    "exposure_only", "heldheld_only", "aligned_state_bridge",
    "inverted_state_bridge", "neutral_decoupled", "mixed_event_bridge",
]
SEEDS = [27900, 27901, 27902, 27903, 27904]
EPOCHS = 50
LR_ENCODER = 2e-5
LR_HEAD = 1e-3
BATCH = 16
MAX_LEN = 196
WARMUP_FRAC = 0.1

EVAL_SUITES = [
    "heldheld_unseen_edge_closure",
    "mixed_held_seen_orientation",
    "paired_state_conservation",
    "cross_template_state_readout",
    "name_permutation_counterfactual",
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class ClassificationModel(nn.Module):
    def __init__(self, base_model, hidden_size, dropout=0.1):
        super().__init__()
        self.base = base_model
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.base(
            input_ids=input_ids, attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls_rep = out.last_hidden_state[:, 0]
        return self.head(self.dropout(cls_rep))


def tokenize_row(row, tokenizer, max_len):
    if row["task"] == "relation_comparison":
        text_a, text_b = row["event1"], row["event2"]
    elif row["task"] == "state_query":
        text_a, text_b = row["premise"], row["hypothesis"]
    else:
        raise ValueError(row["task"])
    enc = tokenizer(text_a, text_b, max_length=max_len, truncation=True, padding=False)
    enc["label"] = 1 if row["label"] else 0
    for k in ("query_kind", "pair_id", "suite"):
        if k in row:
            enc[k] = row[k]
    return enc


def make_batches(encoded_rows, batch_size, pad_id, device, shuffle=False):
    idxs = list(range(len(encoded_rows)))
    if shuffle:
        random.shuffle(idxs)
    for start in range(0, len(idxs), batch_size):
        batch = [encoded_rows[idxs[i]] for i in range(start, min(start + batch_size, len(idxs)))]
        max_len = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long, device=device)
        mask = torch.zeros(len(batch), max_len, dtype=torch.long, device=device)
        tids = torch.zeros(len(batch), max_len, dtype=torch.long, device=device)
        labels = torch.zeros(len(batch), dtype=torch.long, device=device)
        for i, b in enumerate(batch):
            L = len(b["input_ids"])
            ids[i, :L] = torch.tensor(b["input_ids"], device=device)
            mask[i, :L] = torch.tensor(b["attention_mask"], device=device)
            if "token_type_ids" in b:
                tids[i, :L] = torch.tensor(b["token_type_ids"], device=device)
            labels[i] = b["label"]
        yield ids, mask, tids, labels


def train_one_arm(model, train_rows, epochs, pad_id, device):
    """Full fine-tuning with differential learning rates."""
    if len(train_rows) == 0:
        return float("nan"), float("nan")

    # Separate encoder and head parameters
    encoder_params = [p for n, p in model.named_parameters() if not n.startswith("head")]
    head_params = [p for n, p in model.named_parameters() if n.startswith("head")]

    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": LR_ENCODER, "weight_decay": 0.01},
        {"params": head_params, "lr": LR_HEAD, "weight_decay": 0.0},
    ])

    total_steps = epochs * max(1, (len(train_rows) + BATCH - 1) // BATCH)
    warmup_steps = int(total_steps * WARMUP_FRAC)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        return max(0.0, 1.0 - (step - warmup_steps) / max(total_steps - warmup_steps, 1))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model.train()
    step = 0
    for ep in range(epochs):
        total_loss = 0.0
        total_correct = 0
        total_n = 0
        for ids, mask, tids, labels in make_batches(train_rows, BATCH, pad_id, device, shuffle=True):
            logits = model(ids, mask, tids)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            total_loss += loss.item() * len(labels)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_n += len(labels)

    train_acc = total_correct / max(total_n, 1)
    train_loss = total_loss / max(total_n, 1)
    return train_acc, train_loss


@torch.no_grad()
def evaluate(model, eval_encoded, eval_meta, pad_id, device):
    model.eval()
    all_preds = []
    all_labels = []
    for ids, mask, tids, labels in make_batches(eval_encoded, BATCH * 2, pad_id, device):
        logits = model(ids, mask, tids)
        all_preds.append(logits.argmax(1).cpu())
        all_labels.append(labels.cpu())

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    correct = (preds == labels).float()

    acc = correct.mean().item()

    # Group by query_kind
    kind_accs = defaultdict(list)
    pair_results = defaultdict(dict)
    for i in range(len(labels)):
        meta = eval_meta[i]
        qk = meta.get("query_kind", "comparison")
        kind_accs[qk].append(correct[i].item())
        pid = meta.get("pair_id")
        if pid is not None:
            pair_results[pid][qk] = correct[i].item()

    kind_means = {k: np.mean(v) for k, v in kind_accs.items()}

    both_correct = []
    for pid, kinds in pair_results.items():
        if "changed" in kinds and "unchanged" in kinds:
            both_correct.append(
                1.0 if kinds["changed"] >= 0.99 and kinds["unchanged"] >= 0.99 else 0.0
            )

    return {
        "accuracy": round(acc, 4),
        "kind_means": {k: round(v, 4) for k, v in kind_means.items()},
        "pair_both_correct": round(np.mean(both_correct), 4) if both_correct else None,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--arms", type=str, nargs="+", default=ARMS)
    parser.add_argument("--outdir", type=str, default=None)
    args = parser.parse_args()

    ws = _public_path('experiments/archive/representation_and_objectives')
    ckpt = ws / "training" / "runs" / "qwen_8x480_16k_wwm_to_token_100M_seed43022" / "hf_model" / "chck_80M"
    substrate = ws / "data" / "equivariant_symmetry_substrate"
    outdir = Path(args.outdir) if args.outdir else ws / "data" / "finetune_orientation_probe"
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    from transformers import AutoTokenizer, DebertaV2Model
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
    pad_id = tokenizer.pad_token_id or 0

    # Pre-tokenize all data
    arm_train_data = {}
    for arm in args.arms:
        sup_path = substrate / "arms" / arm / "train_supervised.jsonl"
        if sup_path.exists():
            rows = load_jsonl(str(sup_path))
            arm_train_data[arm] = [tokenize_row(r, tokenizer, MAX_LEN) for r in rows]
        else:
            arm_train_data[arm] = []
        print(f"  {arm}: {len(arm_train_data[arm])} train rows", flush=True)

    eval_suites = {}
    for suite in EVAL_SUITES:
        path = substrate / "eval" / f"{suite}.jsonl"
        rows = load_jsonl(str(path))
        encoded = [tokenize_row(r, tokenizer, MAX_LEN) for r in rows]
        meta = [{"query_kind": r.get("query_kind", "comparison"), "pair_id": r.get("pair_id")} for r in rows]
        eval_suites[suite] = {"encoded": encoded, "meta": meta}
        print(f"  eval {suite}: {len(encoded)} rows", flush=True)

    # Get hidden size from config
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(str(ckpt))
    hidden_size = config.hidden_size
    print(f"Hidden size: {hidden_size}", flush=True)

    all_results = []
    for seed in args.seeds:
        # Save initial head state for this seed
        set_seed(seed)
        ref_head = nn.Linear(hidden_size, 2)
        head_init_state = copy.deepcopy(ref_head.state_dict())
        del ref_head

        for arm in args.arms:
            t0 = time.time()
            set_seed(seed)

            # Load fresh pretrained model
            base_model = DebertaV2Model.from_pretrained(str(ckpt))
            model = ClassificationModel(base_model, hidden_size)
            # Set head to the seed-specific initialization
            model.head.load_state_dict(head_init_state)
            model.to(device)

            # Train
            train_acc, train_loss = train_one_arm(
                model, arm_train_data[arm], args.epochs, pad_id, device,
            )

            # Evaluate on all suites
            result = {
                "arm": arm, "seed": seed,
                "train_acc": round(train_acc, 4),
                "elapsed": round(time.time() - t0, 1),
            }
            for suite in EVAL_SUITES:
                ev = evaluate(model, eval_suites[suite]["encoded"],
                              eval_suites[suite]["meta"], pad_id, device)
                sk = (suite
                      .replace("heldheld_unseen_edge_closure", "hh")
                      .replace("mixed_held_seen_orientation", "mixed")
                      .replace("paired_state_conservation", "state")
                      .replace("cross_template_state_readout", "xtempl")
                      .replace("name_permutation_counterfactual", "nperm"))
                result[f"{sk}_acc"] = ev["accuracy"]
                for kind, val in ev["kind_means"].items():
                    result[f"{sk}_{kind}"] = val
                if ev["pair_both_correct"] is not None:
                    result[f"{sk}_pair_both"] = ev["pair_both_correct"]

            all_results.append(result)
            print(f"arm={arm} seed={seed} train={result['train_acc']} "
                  f"hh={result.get('hh_acc','?')} "
                  f"mixed={result.get('mixed_acc','?')} "
                  f"state_chg={result.get('state_changed','?')} "
                  f"state_unchg={result.get('state_unchanged','?')} "
                  f"pair_both={result.get('state_pair_both','?')} "
                  f"[{result['elapsed']}s]", flush=True)

            # Free GPU
            model.cpu()
            del model, base_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Aggregate per arm
    arm_agg = {}
    for arm in args.arms:
        arm_results = [r for r in all_results if r["arm"] == arm]
        agg = {"arm": arm, "n_seeds": len(arm_results)}
        for key in ["train_acc", "hh_acc", "mixed_acc", "state_acc",
                     "state_changed", "state_unchanged", "state_pair_both",
                     "xtempl_acc", "xtempl_changed", "xtempl_unchanged",
                     "nperm_acc", "nperm_changed", "nperm_unchanged"]:
            vals = [r.get(key) for r in arm_results
                    if r.get(key) is not None and not np.isnan(float(r.get(key, "nan")))]
            if vals:
                agg[f"{key}_mean"] = round(np.mean(vals), 4)
                agg[f"{key}_std"] = round(np.std(vals), 4)
        arm_agg[arm] = agg

    # Write results
    rp = outdir / "finetune_orientation_results.json"
    sp = outdir / "finetune_orientation_summary.md"
    with open(rp, "w") as f:
        json.dump({"all_results": all_results, "arm_aggregates": arm_agg}, f, indent=2)

    lines = ["# research fine-tuning orientation probe\n"]
    lines.append(f"Seeds: {args.seeds}, epochs: {args.epochs}\n")
    lines.append("| arm | train | hh | mixed | state_chg | state_unchg | pair_both |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for arm in args.arms:
        a = arm_agg.get(arm, {})
        def fmt(k):
            m = a.get(f"{k}_mean")
            s = a.get(f"{k}_std")
            if m is None: return "nan"
            return f"{m:.3f}±{s:.3f}" if s is not None else f"{m:.3f}"
        lines.append(f"| {arm} | {fmt('train_acc')} | {fmt('hh_acc')} | {fmt('mixed_acc')} "
                      f"| {fmt('state_changed')} | {fmt('state_unchanged')} | {fmt('state_pair_both')} |")

    lines.append("\n## Critical comparison")
    al = arm_agg.get("aligned_state_bridge", {})
    inv = arm_agg.get("inverted_state_bridge", {})
    hh = arm_agg.get("heldheld_only", {})
    lines.append(f"- heldheld_only mixed: {hh.get('mixed_acc_mean', '?')}")
    lines.append(f"- aligned mixed: {al.get('mixed_acc_mean', '?')}")
    lines.append(f"- inverted mixed: {inv.get('mixed_acc_mean', '?')}")
    am = al.get('mixed_acc_mean')
    im = inv.get('mixed_acc_mean')
    if am is not None and im is not None:
        lines.append(f"- aligned - inverted: {am - im:.4f}")
    lines.append(f"\n- results: `{rp}`")

    with open(sp, "w") as f:
        f.write("\n".join(lines))

    # Print critical comparison
    print(f"\n{'='*60}")
    print("CRITICAL COMPARISON:")
    print(f"  heldheld_only mixed: {hh.get('mixed_acc_mean', '?')}")
    print(f"  aligned mixed:      {al.get('mixed_acc_mean', '?')}")
    print(f"  inverted mixed:     {inv.get('mixed_acc_mean', '?')}")
    if am is not None and im is not None:
        print(f"  aligned - inverted: {am - im:.4f}")
    print(f"{'='*60}")

    print(json.dumps({
        "status": "FINETUNE_PROBE_COMPLETE",
        "summary": str(sp),
        "results": str(rp),
        "critical": {
            "heldheld_mixed": hh.get("mixed_acc_mean"),
            "aligned_mixed": al.get("mixed_acc_mean"),
            "inverted_mixed": inv.get("mixed_acc_mean"),
        },
        "no_official_evaluation_upload_or_leaderboard": True,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
