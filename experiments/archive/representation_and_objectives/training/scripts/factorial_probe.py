#!/usr/bin/env python3
"""research: Factorial initial-ownership learned probe.

Compare underdetermined vs disambiguated training on the repaired role-orbit surface.
Fine-tune pretrained DeBERTa on each condition's arm data, evaluate on shared eval.

Scientific prediction:
- Underdetermined → anti-copy → fail on initial=same eval → no mixed orientation
- Disambiguated → event-role → succeed on initial=same eval → mixed orientation separates

The transition from no-orientation to orientation (if observed) would demonstrate
that disambiguation of observationally equivalent rules forces compositional
structure acquisition — connecting research identifiability to language learning.

Uses GPU. No official BabyLM evaluation, upload, or leaderboard.
"""
import json, os, sys, random, math, gc, time
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

PROJECT_ROOT = Path("experiments/archive/representation_and_objectives")
CHECKPOINT = PROJECT_ROOT / "training/runs/qwen_8x480_16k_wwm_to_token_100M_seed43022/hf_model/chck_80M"
DATA_ROOT = PROJECT_ROOT / "data/factorial_initial_ownership"
OUT_DEFAULT = PROJECT_ROOT / "data/factorial_probe"

ARMS = ["aligned_state_bridge", "inverted_state_bridge", "heldheld_only"]
CONDITIONS = ["underdetermined", "disambiguated"]
SEEDS = [28200, 28201, 28202]
EPOCHS = 40
BATCH_SIZE = 8
LR = 2e-5
MAX_LEN = 128


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


class NLIDataset(Dataset):
    def __init__(self, rows, tokenizer, max_len=128):
        self.encodings = []
        self.labels = []
        for r in rows:
            if "premise" in r and "hypothesis" in r:
                enc = tokenizer(r["premise"], r["hypothesis"],
                                max_length=max_len, padding="max_length",
                                truncation=True, return_tensors="pt")
            elif "text" in r:
                enc = tokenizer(r["text"], max_length=max_len,
                                padding="max_length", truncation=True,
                                return_tensors="pt")
            else:
                continue
            self.encodings.append({k: v.squeeze(0) for k, v in enc.items()})
            self.labels.append(int(r["label"]))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v for k, v in self.encodings[idx].items()}
        item["labels"] = self.labels[idx]
        return item


def collate(batch):
    out = {}
    for k in batch[0]:
        if isinstance(batch[0][k], torch.Tensor):
            out[k] = torch.stack([b[k] for b in batch])
        else:
            out[k] = torch.tensor([b[k] for b in batch])
    return out


def evaluate_suite(model, rows, tokenizer, device, max_len=128) -> Dict[str, Any]:
    """Evaluate model on a suite of rows. Returns metrics by task/pattern."""
    ds = NLIDataset(rows, tokenizer, max_len)
    dl = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate)

    all_preds = []
    model.eval()
    with torch.no_grad():
        for batch in dl:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=mask)
            preds = outputs.logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)

    # Annotate rows with predictions
    assert len(all_preds) == len(rows), f"pred count {len(all_preds)} != row count {len(rows)}"
    for i, r in enumerate(rows):
        r["_pred"] = all_preds[i]
        r["_correct"] = all_preds[i] == int(r["label"])

    metrics: Dict[str, Any] = {
        "n": len(rows),
        "overall_acc": sum(r["_correct"] for r in rows) / len(rows),
    }

    # --- State rows ---
    state = [r for r in rows if r.get("task") == "state_query"]
    if state:
        for pat in ["opposite", "same", None]:
            for qk in ["changed", "unchanged"]:
                true_label = [r for r in state
                              if r["query_kind"] == qk and r["label"] is True
                              and (pat is None or r.get("initial_pattern") == pat)]
                if true_label:
                    acc = sum(r["_correct"] for r in true_label) / len(true_label)
                    prefix = "all" if pat is None else pat
                    metrics[f"{prefix}_{qk}_acc"] = acc
                    metrics[f"{prefix}_{qk}_n"] = len(true_label)

        # Pair-both by initial_pattern
        by_pair: Dict[str, Dict] = {}
        for r in state:
            if r["label"] is True:
                by_pair.setdefault(r["pair_id"], {})[r["query_kind"]] = r

        for pat in ["opposite", "same", None]:
            both_ok = []
            for pid, d in by_pair.items():
                if "changed" not in d or "unchanged" not in d:
                    continue
                if pat is not None and d["changed"].get("initial_pattern") != pat:
                    continue
                both_ok.append(int(d["changed"]["_correct"] and d["unchanged"]["_correct"]))
            if both_ok:
                prefix = "all" if pat is None else pat
                metrics[f"{prefix}_pair_both"] = sum(both_ok) / len(both_ok)
                metrics[f"{prefix}_pair_both_n"] = len(both_ok)

    # --- Comparison rows: mixed orientation ---
    comp = [r for r in rows if r.get("task") == "relation_comparison"]
    if comp:
        true_stmt = [r for r in comp if r["label"] is True]
        if true_stmt:
            metrics["mixed_true_stmt"] = sum(r["_correct"] for r in true_stmt) / len(true_stmt)
            metrics["mixed_true_stmt_n"] = len(true_stmt)

    # Clean temporary fields
    for r in rows:
        r.pop("_pred", None)
        r.pop("_correct", None)

    return metrics


def train_one(condition: str, arm: str, seed: int, tokenizer, device) -> Dict:
    """Train on one condition+arm, evaluate on shared eval, return all metrics."""
    t0 = time.time()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cond_dir = DATA_ROOT / condition
    train_rows = load_jsonl(cond_dir / "common_seen_train.jsonl")
    arm_sup = load_jsonl(cond_dir / "arms" / arm / "train_supervised.jsonl")
    train_rows.extend(arm_sup)

    # Load model
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(
        str(CHECKPOINT), num_labels=2, ignore_mismatched_sizes=True
    )
    model.to(device)

    # Training
    train_ds = NLIDataset(train_rows, tokenizer, MAX_LEN)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    model.train()
    for epoch in range(EPOCHS):
        for batch in train_dl:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=mask, labels=labels)
            outputs.loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    # Training accuracy
    model.eval()
    train_correct = 0
    train_total = 0
    with torch.no_grad():
        for batch in train_dl:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            preds = model(input_ids=input_ids, attention_mask=mask).logits.argmax(-1)
            train_correct += (preds == batch["labels"].to(device)).sum().item()
            train_total += len(preds)
    train_acc = train_correct / train_total if train_total else 0

    # Evaluate on all suites
    eval_results = {}
    for ef in sorted((cond_dir / "eval").iterdir()):
        if ef.suffix == ".jsonl":
            eval_rows = load_jsonl(ef)
            eval_results[ef.stem] = evaluate_suite(model, eval_rows, tokenizer, device, MAX_LEN)

    elapsed = time.time() - t0

    # Cleanup
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "condition": condition,
        "arm": arm,
        "seed": seed,
        "train_acc": train_acc,
        "train_rows": len(train_rows),
        "elapsed": round(elapsed, 1),
        "eval": eval_results,
    }


def main():
    out = Path(OUT_DEFAULT)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT))

    all_results = []
    for condition in CONDITIONS:
        for arm in ARMS:
            for seed in SEEDS:
                tag = f"{condition}/{arm}/seed={seed}"
                print(f"\n=== {tag} ===", flush=True)
                try:
                    result = train_one(condition, arm, seed, tokenizer, device)
                    all_results.append(result)

                    psc = result["eval"].get("paired_state_conservation", {})
                    mhs = result["eval"].get("mixed_held_seen_orientation", {})
                    print(f"  train_acc: {result['train_acc']:.3f}", flush=True)
                    print(f"  mixed_true_stmt: {mhs.get('mixed_true_stmt', 'n/a')}", flush=True)
                    print(f"  opp_changed: {psc.get('opposite_changed_acc', 'n/a')}", flush=True)
                    print(f"  same_changed: {psc.get('same_changed_acc', 'n/a')}", flush=True)
                    print(f"  opp_pair_both: {psc.get('opposite_pair_both', 'n/a')}", flush=True)
                    print(f"  same_pair_both: {psc.get('same_pair_both', 'n/a')}", flush=True)
                    print(f"  elapsed: {result['elapsed']}s", flush=True)
                except Exception as e:
                    print(f"  ERROR: {e}", flush=True)
                    all_results.append({"condition": condition, "arm": arm, "seed": seed, "error": str(e)})

    # Save all results
    (out / "all_results.json").write_text(json.dumps(all_results, indent=2) + "\n")

    # Aggregate by condition × arm
    agg: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for entry in all_results:
        if "error" in entry:
            continue
        key = f"{entry['condition']}_{entry['arm']}"
        for suite, metrics in entry.get("eval", {}).items():
            for mk, mv in metrics.items():
                if isinstance(mv, (int, float)):
                    agg[key][f"{suite}__{mk}"].append(mv)
        agg[key]["train_acc"].append(entry["train_acc"])

    def fmt_agg(vals):
        if not vals:
            return "n/a"
        m = np.mean(vals)
        s = np.std(vals) if len(vals) > 1 else 0
        return f"{m:.3f}±{s:.3f}" if s > 0.001 else f"{m:.3f}"

    # Summary table
    lines = ["# research factorial probe results", ""]
    lines.append("## Key transition prediction metrics")
    lines.append("")
    header = "| condition | arm | train_acc | mixed_true | opp_chg | same_chg | opp_pb | same_pb |"
    lines.append(header)
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")

    for cond in CONDITIONS:
        for arm in ARMS:
            key = f"{cond}_{arm}"
            d = agg.get(key, {})
            ta = fmt_agg(d.get("train_acc", []))
            mt = fmt_agg(d.get("mixed_held_seen_orientation__mixed_true_stmt", []))
            oc = fmt_agg(d.get("paired_state_conservation__opposite_changed_acc", []))
            sc = fmt_agg(d.get("paired_state_conservation__same_changed_acc", []))
            opb = fmt_agg(d.get("paired_state_conservation__opposite_pair_both", []))
            spb = fmt_agg(d.get("paired_state_conservation__same_pair_both", []))
            lines.append(f"| {cond} | {arm} | {ta} | {mt} | {oc} | {sc} | {opb} | {spb} |")

    lines.append("")
    lines.append("## Transition test")
    lines.append("")
    lines.append("For aligned_state_bridge:")
    for metric_name, metric_key in [
        ("same_changed", "paired_state_conservation__same_changed_acc"),
        ("same_pair_both", "paired_state_conservation__same_pair_both"),
        ("mixed_true_stmt", "mixed_held_seen_orientation__mixed_true_stmt"),
    ]:
        u_vals = agg.get("underdetermined_aligned_state_bridge", {}).get(metric_key, [])
        d_vals = agg.get("disambiguated_aligned_state_bridge", {}).get(metric_key, [])
        u_mean = np.mean(u_vals) if u_vals else float("nan")
        d_mean = np.mean(d_vals) if d_vals else float("nan")
        delta = d_mean - u_mean
        lines.append(f"- {metric_name}: underdetermined {u_mean:.3f} → disambiguated {d_mean:.3f} (Δ={delta:+.3f})")

    lines.append("")
    lines.append("## Cross-template and name-permutation state readouts")
    lines.append("")
    for suite in ["cross_template_state_readout", "name_permutation_counterfactual"]:
        lines.append(f"**{suite}**")
        lines.append("")
        lines.append("| condition | arm | opp_changed | same_changed | opp_pair_both | same_pair_both |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for cond in CONDITIONS:
            for arm in ARMS:
                key = f"{cond}_{arm}"
                d = agg.get(key, {})
                oc = fmt_agg(d.get(f"{suite}__opposite_changed_acc", []))
                sc = fmt_agg(d.get(f"{suite}__same_changed_acc", []))
                opb = fmt_agg(d.get(f"{suite}__opposite_pair_both", []))
                spb = fmt_agg(d.get(f"{suite}__same_pair_both", []))
                lines.append(f"| {cond} | {arm} | {oc} | {sc} | {opb} | {spb} |")
        lines.append("")

    lines.append("## Files")
    lines.append(f"- All results: `{out / 'all_results.json'}`")
    lines.append(f"- Substrate: `{DATA_ROOT}`")
    lines.append(f"- Formal derivation: `notes/formal_derivation_factorial_disambiguation.md`")
    lines.append("")

    (out / "factorial_probe_summary.md").write_text("\n".join(lines))

    # Print compact status
    status = {
        "status": "FACTORIAL_PROBE_COMPLETE",
        "out": str(out.relative_to(PROJECT_ROOT)),
        "summary": str((out / "factorial_probe_summary.md").relative_to(PROJECT_ROOT)),
        "n_results": len([r for r in all_results if "error" not in r]),
        "n_errors": len([r for r in all_results if "error" in r]),
    }
    # Add key transition metrics
    for arm in ["aligned_state_bridge", "inverted_state_bridge"]:
        for mk in ["same_changed_acc", "same_pair_both", "mixed_true_stmt"]:
            suite = "paired_state_conservation" if "changed" in mk or "pair" in mk else "mixed_held_seen_orientation"
            full_key = f"{suite}__{mk}" if suite != "mixed_held_seen_orientation" else f"{suite}__{mk}"
            u_vals = agg.get(f"underdetermined_{arm}", {}).get(full_key, [])
            d_vals = agg.get(f"disambiguated_{arm}", {}).get(full_key, [])
            if u_vals and d_vals:
                status[f"{arm}_{mk}_delta"] = round(np.mean(d_vals) - np.mean(u_vals), 4)

    status["no_official_evaluation_upload_or_leaderboard"] = True
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
