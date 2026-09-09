#!/usr/bin/env python3
"""research: frozen-encoder orientation probe.

Scientific purpose
------------------
Test whether sparse bridge state evidence propagates role orientation through
FROZEN pretrained DeBERTa representations to mixed held-seen relation comparisons.

Design
------
Freeze the entire DeBERTa encoder. Train only a linear classification head.
All arms share identical head initialization per seed. Only bridge evidence differs.

Critical prediction: aligned and inverted arms should produce OPPOSITE mixed
held-seen orientations if the frozen encoder carries a role-slot interface that
bridge evidence can orient. If both arms give similar orientations → the head
absorbs bridge polarity (calibration). If neither moves from 0.5 → frozen
representations don't carry role info.

Controls:
- exposure_only: no supervised data (random head baseline)
- heldheld_only: comparison-only (no bridge evidence, calibration baseline)
- neutral_decoupled: irrelevant bridge evidence (should ≈ heldheld_only)
- random_init: frozen random encoder (architecture-only baseline)
"""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import os, json, sys, random, copy, time
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from collections import defaultdict

# ---- configuration ----
ARMS = [
    "exposure_only", "heldheld_only", "aligned_state_bridge",
    "inverted_state_bridge", "neutral_decoupled", "mixed_event_bridge",
]
SEEDS = [27900, 27901, 27902, 27903, 27904]
EPOCHS = 80
LR = 1e-3
BATCH = 64
MAX_LEN = 196

EVAL_SUITES = [
    "heldheld_unseen_edge_closure",
    "mixed_held_seen_orientation",
    "paired_state_conservation",
    "cross_template_state_readout",
    "name_permutation_counterfactual",
]

# ---- helpers ----

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def tokenize_row(row, tokenizer):
    if row["task"] == "relation_comparison":
        text_a, text_b = row["event1"], row["event2"]
    elif row["task"] == "state_query":
        text_a, text_b = row["premise"], row["hypothesis"]
    else:
        raise ValueError(f"Unknown task: {row['task']}")
    enc = tokenizer(
        text_a, text_b, max_length=MAX_LEN,
        truncation=True, padding=False, return_tensors=None,
    )
    enc["label"] = 1 if row["label"] else 0
    # copy metadata for eval grouping
    for k in ("query_kind", "pair_id", "suite"):
        if k in row:
            enc[k] = row[k]
    return enc


def make_batches(encoded_rows, batch_size, pad_id, device, shuffle=False):
    """Yield (ids, mask, tids, labels) batches."""
    idxs = list(range(len(encoded_rows)))
    if shuffle:
        random.shuffle(idxs)
    for start in range(0, len(idxs), batch_size):
        batch_idxs = idxs[start : start + batch_size]
        batch = [encoded_rows[i] for i in batch_idxs]
        max_len = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        mask = torch.zeros(len(batch), max_len, dtype=torch.long)
        tids = torch.zeros(len(batch), max_len, dtype=torch.long)
        labels = torch.zeros(len(batch), dtype=torch.long)
        for i, b in enumerate(batch):
            L = len(b["input_ids"])
            ids[i, :L] = torch.tensor(b["input_ids"])
            mask[i, :L] = torch.tensor(b["attention_mask"])
            if "token_type_ids" in b:
                tids[i, :L] = torch.tensor(b["token_type_ids"])
            labels[i] = b["label"]
        yield ids.to(device), mask.to(device), tids.to(device), labels.to(device)


@torch.no_grad()
def cache_representations(model, encoded_rows, pad_id, device, batch_size=64):
    """Cache frozen [CLS] representations for all rows."""
    reps = []
    for ids, mask, tids, _ in make_batches(encoded_rows, batch_size, pad_id, device):
        out = model(input_ids=ids, attention_mask=mask, token_type_ids=tids)
        cls_rep = out.last_hidden_state[:, 0]  # [CLS] token
        reps.append(cls_rep.cpu())
    return torch.cat(reps, dim=0)


def train_head(head, train_reps, train_labels, epochs, lr, device):
    """Train a linear head on cached representations."""
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    head.train()
    n = len(train_labels)
    for ep in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        for start in range(0, n, BATCH):
            idx = perm[start : start + BATCH]
            x = train_reps[idx].to(device)
            y = train_labels[idx].to(device)
            logits = head(x)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
    return total_loss / max(n, 1)


def eval_head(head, eval_reps, eval_labels, eval_meta, device):
    """Evaluate head on cached eval representations."""
    head.eval()
    with torch.no_grad():
        logits = head(eval_reps.to(device))
        preds = logits.argmax(dim=1).cpu()
    
    labels = eval_labels
    correct = (preds == labels).float()
    
    # Overall accuracy
    acc = correct.mean().item()
    
    # Group by query_kind for state suites
    kind_accs = {}
    pair_results = {}
    
    for i in range(len(labels)):
        meta = eval_meta[i]
        qk = meta.get("query_kind", "comparison")
        if qk not in kind_accs:
            kind_accs[qk] = []
        kind_accs[qk].append(correct[i].item())
        
        pid = meta.get("pair_id")
        if pid is not None:
            if pid not in pair_results:
                pair_results[pid] = {}
            pair_results[pid][qk] = correct[i].item()
    
    # Mean per kind
    kind_means = {k: np.mean(v) for k, v in kind_accs.items()}
    
    # Pair both-correct
    both_correct = []
    for pid, kinds in pair_results.items():
        if "changed" in kinds and "unchanged" in kinds:
            both_correct.append(1.0 if kinds["changed"] >= 0.99 and kinds["unchanged"] >= 0.99 else 0.0)
    
    return {
        "accuracy": acc,
        "kind_means": kind_means,
        "pair_both_correct": np.mean(both_correct) if both_correct else None,
        "n_pairs": len(both_correct),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", choices=["pretrained", "random"], default="pretrained")
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--gpu", type=int, default=None)
    args = parser.parse_args()

    # Paths
    ws = _public_path('experiments/archive/representation_and_objectives')  # 
    ckpt = ws / "training" / "runs" / "qwen_8x480_16k_wwm_to_token_100M_seed43022" / "hf_model" / "chck_80M"
    substrate = ws / "data" / "equivariant_symmetry_substrate"
    
    if args.outdir:
        outdir = Path(args.outdir)
    else:
        outdir = ws / "data" / f"frozen_{args.init}_probe"
    outdir.mkdir(parents=True, exist_ok=True)

    # Device
    if args.gpu is not None:
        device = torch.device(f"cuda:{args.gpu}")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}, init: {args.init}", flush=True)

    # Load tokenizer
    from transformers import AutoTokenizer, DebertaV2Model, DebertaV2Config
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
    pad_id = tokenizer.pad_token_id or 0
    hidden_size = None

    # Load model
    if args.init == "pretrained":
        model = DebertaV2Model.from_pretrained(str(ckpt))
        hidden_size = model.config.hidden_size
    else:
        config = DebertaV2Config.from_pretrained(str(ckpt))
        set_seed(42)  # fixed random init seed
        model = DebertaV2Model(config)
        hidden_size = config.hidden_size
    
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model.to(device)
    print(f"Model loaded, hidden_size={hidden_size}, frozen params={sum(p.numel() for p in model.parameters())}", flush=True)

    # Load and tokenize all data
    # Training data per arm
    arm_train = {}
    for arm in ARMS:
        sup_path = substrate / "arms" / arm / "train_supervised.jsonl"
        if sup_path.exists():
            rows = load_jsonl(str(sup_path))
            arm_train[arm] = [tokenize_row(r, tokenizer) for r in rows]
        else:
            arm_train[arm] = []
        print(f"  {arm}: {len(arm_train[arm])} train rows", flush=True)

    # Eval data (shared across arms)
    eval_data = {}
    for suite in EVAL_SUITES:
        path = substrate / "eval" / f"{suite}.jsonl"
        rows = load_jsonl(str(path))
        encoded = [tokenize_row(r, tokenizer) for r in rows]
        eval_data[suite] = {
            "encoded": encoded,
            "meta": [{"query_kind": r.get("query_kind", "comparison"),
                       "pair_id": r.get("pair_id")} for r in rows],
        }
        print(f"  eval {suite}: {len(encoded)} rows", flush=True)

    # Cache ALL representations (train + eval)
    print("\nCaching encoder representations...", flush=True)
    t0 = time.time()
    
    # Collect all unique encoded rows
    all_encoded = []
    arm_train_indices = {}
    offset = 0
    for arm in ARMS:
        arm_train_indices[arm] = list(range(offset, offset + len(arm_train[arm])))
        all_encoded.extend(arm_train[arm])
        offset += len(arm_train[arm])
    
    eval_indices = {}
    for suite in EVAL_SUITES:
        eval_indices[suite] = list(range(offset, offset + len(eval_data[suite]["encoded"])))
        all_encoded.extend(eval_data[suite]["encoded"])
        offset += len(eval_data[suite]["encoded"])
    
    all_reps = cache_representations(model, all_encoded, pad_id, device, batch_size=64)
    print(f"Cached {all_reps.shape[0]} representations in {time.time()-t0:.1f}s, shape={all_reps.shape}", flush=True)
    
    # Free GPU model memory
    model.cpu()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Extract per-arm train reps/labels and eval reps/labels/meta
    arm_train_reps = {}
    arm_train_labels = {}
    for arm in ARMS:
        if arm_train_indices[arm]:
            arm_train_reps[arm] = all_reps[arm_train_indices[arm]]
            arm_train_labels[arm] = torch.tensor(
                [all_encoded[i]["label"] for i in arm_train_indices[arm]], dtype=torch.long
            )
        else:
            arm_train_reps[arm] = torch.zeros(0, hidden_size)
            arm_train_labels[arm] = torch.zeros(0, dtype=torch.long)

    eval_reps_dict = {}
    eval_labels_dict = {}
    eval_meta_dict = {}
    for suite in EVAL_SUITES:
        idx = eval_indices[suite]
        eval_reps_dict[suite] = all_reps[idx]
        eval_labels_dict[suite] = torch.tensor(
            [all_encoded[i]["label"] for i in idx], dtype=torch.long
        )
        eval_meta_dict[suite] = eval_data[suite]["meta"]

    # Run probe for each seed × arm
    all_results = []
    for seed in args.seeds:
        for arm in ARMS:
            set_seed(seed)
            head = torch.nn.Linear(hidden_size, 2).to(device)
            
            if len(arm_train_reps[arm]) == 0:
                # exposure_only: no training, just evaluate random head
                final_loss = float("nan")
            else:
                # Train head
                final_loss = train_head(
                    head, arm_train_reps[arm], arm_train_labels[arm],
                    args.epochs, LR, device,
                )
            
            # Train accuracy
            if len(arm_train_reps[arm]) > 0:
                head.eval()
                with torch.no_grad():
                    train_logits = head(arm_train_reps[arm].to(device))
                    train_preds = train_logits.argmax(1).cpu()
                    train_acc = (train_preds == arm_train_labels[arm]).float().mean().item()
            else:
                train_acc = float("nan")
            
            # Evaluate on all suites
            result = {
                "init": args.init, "arm": arm, "seed": seed,
                "train_acc": round(train_acc, 4),
                "train_loss": round(final_loss, 4) if not np.isnan(final_loss) else None,
            }
            for suite in EVAL_SUITES:
                ev = eval_head(
                    head, eval_reps_dict[suite], eval_labels_dict[suite],
                    eval_meta_dict[suite], device,
                )
                # Short keys
                sk = suite.replace("heldheld_unseen_edge_closure", "hh") \
                          .replace("mixed_held_seen_orientation", "mixed") \
                          .replace("paired_state_conservation", "state") \
                          .replace("cross_template_state_readout", "xtempl") \
                          .replace("name_permutation_counterfactual", "nperm")
                result[f"{sk}_acc"] = round(ev["accuracy"], 4)
                for kind, val in ev["kind_means"].items():
                    result[f"{sk}_{kind}"] = round(val, 4)
                if ev["pair_both_correct"] is not None:
                    result[f"{sk}_pair_both"] = round(ev["pair_both_correct"], 4)

            all_results.append(result)
            
            # Print compact summary
            print(f"init={args.init} arm={arm} seed={seed} "
                  f"train={result['train_acc']} "
                  f"hh={result.get('hh_acc','?')} "
                  f"mixed={result.get('mixed_acc','?')} "
                  f"state_chg={result.get('state_changed','?')} "
                  f"state_unchg={result.get('state_unchanged','?')} "
                  f"pair_both={result.get('state_pair_both','?')}",
                  flush=True)
            
            del head
    
    # Aggregate per arm
    arm_agg = {}
    for arm in ARMS:
        arm_results = [r for r in all_results if r["arm"] == arm]
        agg = {"arm": arm, "n_seeds": len(arm_results)}
        for key in ["train_acc", "hh_acc", "mixed_acc", "state_acc",
                     "state_changed", "state_unchanged", "state_pair_both",
                     "xtempl_acc", "xtempl_changed", "xtempl_unchanged",
                     "nperm_acc", "nperm_changed", "nperm_unchanged"]:
            vals = [r.get(key) for r in arm_results if r.get(key) is not None and not np.isnan(r.get(key, float("nan")))]
            if vals:
                agg[f"{key}_mean"] = round(np.mean(vals), 4)
                agg[f"{key}_std"] = round(np.std(vals), 4)
        arm_agg[arm] = agg

    # Save results
    rp = outdir / f"frozen_{args.init}_results.json"
    sp = outdir / f"frozen_{args.init}_summary.md"
    
    with open(rp, "w") as f:
        json.dump({"all_results": all_results, "arm_aggregates": arm_agg}, f, indent=2)
    
    # Write summary
    lines = [f"# research frozen-encoder probe ({args.init})\n"]
    lines.append(f"Init: {args.init}, seeds: {args.seeds}, epochs: {args.epochs}\n")
    lines.append("| arm | train | hh | mixed | state_chg | state_unchg | pair_both |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for arm in ARMS:
        a = arm_agg.get(arm, {})
        def fmt(k):
            m = a.get(f"{k}_mean")
            s = a.get(f"{k}_std")
            if m is None:
                return "nan"
            return f"{m:.3f}±{s:.3f}" if s is not None else f"{m:.3f}"
        lines.append(f"| {arm} | {fmt('train_acc')} | {fmt('hh_acc')} | {fmt('mixed_acc')} "
                      f"| {fmt('state_changed')} | {fmt('state_unchanged')} | {fmt('state_pair_both')} |")
    
    lines.append("\n## Interpretation guide")
    lines.append("- mixed: aligned > 0.5 AND inverted < 0.5 → frozen encoder carries orientable role interface")
    lines.append("- mixed: aligned ≈ inverted ≈ heldheld → head calibration absorbs bridge polarity")
    lines.append("- mixed: all ≈ 0.5 → frozen representations don't carry role info")
    lines.append("- pair_both: conservation of changed+unchanged states")
    lines.append(f"\n- results: `{rp}`")
    
    with open(sp, "w") as f:
        f.write("\n".join(lines))
    
    # Critical comparison
    aligned_mixed = arm_agg.get("aligned_state_bridge", {}).get("mixed_acc_mean")
    inverted_mixed = arm_agg.get("inverted_state_bridge", {}).get("mixed_acc_mean")
    hh_mixed = arm_agg.get("heldheld_only", {}).get("mixed_acc_mean")
    
    print(f"\n{'='*60}")
    print(f"CRITICAL COMPARISON (init={args.init}):")
    print(f"  heldheld_only mixed: {hh_mixed}")
    print(f"  aligned mixed:      {aligned_mixed}")
    print(f"  inverted mixed:     {inverted_mixed}")
    if aligned_mixed is not None and inverted_mixed is not None:
        print(f"  aligned - inverted: {aligned_mixed - inverted_mixed:.4f}")
    print(f"{'='*60}")
    
    print(json.dumps({
        "status": f"FROZEN_{args.init.upper()}_PROBE_COMPLETE",
        "summary": str(sp),
        "results": str(rp),
        "init": args.init,
        "critical": {
            "heldheld_mixed": hh_mixed,
            "aligned_mixed": aligned_mixed,
            "inverted_mixed": inverted_mixed,
        },
        "no_official_evaluation_upload_or_leaderboard": True,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
