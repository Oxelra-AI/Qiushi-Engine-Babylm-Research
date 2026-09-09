#!/usr/bin/env python3
"""research: symmetry-identification learned pilot.

Scientific purpose
------------------
Test whether a small pretrained model, fine-tuned on the repaired v2 substrate,
shows the predicted orientation pattern:
  - exposure_only:   chance on all evals
  - heldheld_only:   high heldheld_closure, ~0.500 mixed (Z2 ambiguity)
  - aligned_state:   high heldheld_closure, high mixed (true assignment)
  - inverted_state:  high heldheld_closure, LOW mixed (inverted assignment)
  - neutral:         high heldheld_closure, ~0.500 mixed (Z2 preserved)
  - mixed_event:     high heldheld_closure, high mixed (true assignment)

The decisive test: aligned and inverted should produce OPPOSITE mixed_held_seen
orientation while both have high heldheld consistency.

Uses already-trained BabyLM DeBERTa-v2 8x480 from chck_80M as encoder.
Classification head randomly initialized per arm. Tiny substrate → fast GPU run.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import copy
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    DebertaV2Config,
    DebertaV2ForMaskedLM,
    DebertaV2ForSequenceClassification,
)

AI_LAB_DIR = _public_path('experiments/archive/representation_and_objectives/training')  # training/
WORKSPACE = _public_path('experiments/archive/representation_and_objectives')                     # 
SUBSTRATE_DIR = _public_path('experiments/archive/representation_and_objectives/data/symmetry_identification_substrate_v2')
MODEL_PATH = _public_path('experiments/archive/representation_and_objectives/training/runs/qwen_8x480_16k_wwm_to_token_100M_seed43022/hf_model/chck_80M')
OUT_DEFAULT = _public_path('experiments/archive/representation_and_objectives/data/symmetry_pilot')

ARMS = [
    "exposure_only",
    "heldheld_only",
    "aligned_state_bridge",
    "inverted_state_bridge",
    "neutral_decoupled",
    "mixed_event_bridge",
]
EVAL_SUITES = [
    "heldheld_unseen_edge_closure",
    "mixed_held_seen_orientation",
    "paired_state_conservation",
    "cross_template_state_readout",
    "name_permutation_counterfactual",
]


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def format_input(row: Dict) -> Tuple[str, str | None]:
    """Return (text_a, text_b) for tokenizer pair encoding."""
    if row.get("task") == "state_query":
        return row["premise"], row["hypothesis"]
    # comparison: use the text field which contains the full question
    return row.get("text", ""), None


class SubstrateDataset(Dataset):
    def __init__(self, rows: List[Dict], tokenizer, max_len: int = 192):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        text_a, text_b = format_input(row)
        enc = self.tokenizer(
            text_a,
            text_b,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        label = int(bool(row["label"])) if "label" in row else -1
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc.get("token_type_ids", torch.zeros_like(enc["input_ids"])).squeeze(0),
            "label": label,
        }


def build_model(model_path: Path, device: torch.device, seed: int) -> DebertaV2ForSequenceClassification:
    """Load pretrained encoder, attach fresh classification head."""
    config = DebertaV2Config.from_pretrained(str(model_path))
    config.num_labels = 2
    # Load MLM model to extract encoder
    mlm = DebertaV2ForMaskedLM.from_pretrained(str(model_path))
    # Create classification model
    torch.manual_seed(seed)
    cls_model = DebertaV2ForSequenceClassification(config)
    # Copy encoder weights
    cls_model.deberta.load_state_dict(mlm.deberta.state_dict())
    del mlm
    return cls_model.to(device)


def train_and_eval(
    arm_name: str,
    train_rows: List[Dict],
    eval_suites: Dict[str, List[Dict]],
    tokenizer,
    model_path: Path,
    device: torch.device,
    seed: int,
    epochs: int = 30,
    lr: float = 2e-5,
    batch_size: int = 16,
    max_len: int = 192,
) -> Dict[str, Any]:
    """Train on arm's supervised data + common_seen, evaluate on all suites."""
    t0 = time.time()

    # Seed everything
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Separate labeled rows
    labeled = [r for r in train_rows if "label" in r]
    if not labeled:
        # exposure_only: no supervised signal, evaluate with random model
        model = build_model(model_path, device, seed)
        model.eval()
        results = {"arm": arm_name, "seed": seed, "train_rows": 0, "train_labeled": 0}
        results["train_loss_first"] = None
        results["train_loss_last"] = None
        results["train_acc_last"] = None
        for suite_name, suite_rows in eval_suites.items():
            results[suite_name] = eval_suite(model, suite_rows, tokenizer, device, max_len, batch_size)
        results["elapsed_sec"] = time.time() - t0
        del model
        return results

    model = build_model(model_path, device, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    train_ds = SubstrateDataset(labeled, tokenizer, max_len)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    first_loss = None
    last_loss = None
    last_acc = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )
            loss = outputs.loss
            total_loss += loss.item() * input_ids.size(0)
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / max(total, 1)
        acc = correct / max(total, 1)
        if epoch == 0:
            first_loss = avg_loss
        last_loss = avg_loss
        last_acc = acc

    # Evaluate
    model.eval()
    results = {
        "arm": arm_name,
        "seed": seed,
        "train_rows": len(train_rows),
        "train_labeled": len(labeled),
        "train_loss_first": round(first_loss, 4) if first_loss is not None else None,
        "train_loss_last": round(last_loss, 4) if last_loss is not None else None,
        "train_acc_last": round(last_acc, 4) if last_acc is not None else None,
    }
    for suite_name, suite_rows in eval_suites.items():
        results[suite_name] = eval_suite(model, suite_rows, tokenizer, device, max_len, batch_size)
    results["elapsed_sec"] = round(time.time() - t0, 1)

    del model, optimizer
    torch.cuda.empty_cache()
    return results


def eval_suite(
    model: DebertaV2ForSequenceClassification,
    rows: List[Dict],
    tokenizer,
    device: torch.device,
    max_len: int,
    batch_size: int,
) -> Dict[str, Any]:
    """Evaluate on a suite's rows, return accuracy by task type and diagnostic details."""
    if not rows:
        return {"n": 0, "accuracy": None}

    labeled = [r for r in rows if "label" in r]
    if not labeled:
        return {"n": 0, "accuracy": None}

    ds = SubstrateDataset(labeled, tokenizer, max_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            preds = outputs.logits.argmax(dim=-1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
            all_logits.extend(outputs.logits.cpu().tolist())

    n = len(all_labels)
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    accuracy = correct / n if n > 0 else None

    # Detailed breakdowns
    result: Dict[str, Any] = {"n": n, "accuracy": round(accuracy, 4) if accuracy is not None else None}

    # By task type
    for task_type in set(r.get("task") for r in labeled):
        mask = [i for i, r in enumerate(labeled) if r.get("task") == task_type]
        if mask:
            task_correct = sum(all_preds[i] == all_labels[i] for i in mask)
            result[f"acc_{task_type}"] = round(task_correct / len(mask), 4)

    # By orientation_dependency
    for dep in set(r.get("orientation_dependency") for r in labeled):
        if dep is None:
            continue
        mask = [i for i, r in enumerate(labeled) if r.get("orientation_dependency") == dep]
        if mask:
            dep_correct = sum(all_preds[i] == all_labels[i] for i in mask)
            result[f"acc_dep_{dep}"] = round(dep_correct / len(mask), 4)

    # For state_query suites: changed vs unchanged accuracy
    changed_mask = [i for i, r in enumerate(labeled) if r.get("query_kind") == "changed"]
    unchanged_mask = [i for i, r in enumerate(labeled) if r.get("query_kind") == "unchanged"]
    if changed_mask:
        result["acc_changed"] = round(sum(all_preds[i] == all_labels[i] for i in changed_mask) / len(changed_mask), 4)
    if unchanged_mask:
        result["acc_unchanged"] = round(sum(all_preds[i] == all_labels[i] for i in unchanged_mask) / len(unchanged_mask), 4)

    # Prediction bias
    result["pred_true_frac"] = round(sum(all_preds) / n, 4) if n > 0 else None
    result["label_true_frac"] = round(sum(all_labels) / n, 4) if n > 0 else None

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrate-dir", type=Path, default=SUBSTRATE_DIR)
    ap.add_argument("--model-path", type=Path, default=MODEL_PATH)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--seed", type=int, default=27700)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--arms", nargs="*", default=None, help="Subset of arms to run")
    args = ap.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path))
    print(f"Tokenizer loaded, vocab={tokenizer.vocab_size}", flush=True)

    # Load common_seen_train
    common_seen = load_jsonl(args.substrate_dir / "common_seen_train.jsonl")
    print(f"Common seen train: {len(common_seen)} rows", flush=True)

    # Load eval suites
    eval_suites: Dict[str, List[Dict]] = {}
    for suite in EVAL_SUITES:
        p = args.substrate_dir / "eval" / f"{suite}.jsonl"
        if p.exists():
            eval_suites[suite] = load_jsonl(p)
            print(f"Eval {suite}: {len(eval_suites[suite])} rows", flush=True)

    # Run each arm
    arms_to_run = args.arms or ARMS
    all_results = []

    for arm in arms_to_run:
        print(f"\n{'='*60}\nArm: {arm}\n{'='*60}", flush=True)

        # Load arm data
        sup_path = args.substrate_dir / "arms" / arm / "train_supervised.jsonl"
        sup_rows = load_jsonl(sup_path) if sup_path.exists() else []
        print(f"  Supervised: {len(sup_rows)} rows", flush=True)

        # Combine supervised + common_seen for training
        train_rows = sup_rows + common_seen

        result = train_and_eval(
            arm_name=arm,
            train_rows=train_rows,
            eval_suites=eval_suites,
            tokenizer=tokenizer,
            model_path=args.model_path,
            device=device,
            seed=args.seed,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            max_len=args.max_len,
        )
        all_results.append(result)

        # Print key metrics
        print(f"\n  Train: loss_first={result.get('train_loss_first')}, "
              f"loss_last={result.get('train_loss_last')}, "
              f"acc_last={result.get('train_acc_last')}", flush=True)
        for suite in EVAL_SUITES:
            if suite in result:
                sr = result[suite]
                print(f"  {suite}: acc={sr.get('accuracy')}", flush=True)
                if "acc_changed" in sr:
                    print(f"    changed={sr.get('acc_changed')}, unchanged={sr.get('acc_unchanged')}", flush=True)
        print(f"  Elapsed: {result.get('elapsed_sec')}s", flush=True)

    # Save results
    project_root = _public_path('experiments/archive')
    results_path = out / "symmetry_pilot_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Write summary table
    summary_path = out / "symmetry_pilot_summary.md"
    write_summary(summary_path, all_results)

    def relpath(p):
        try:
            return str(p.relative_to(project_root))
        except ValueError:
            return str(p)

    print(json.dumps({
        "status": "SYMMETRY_PILOT_COMPLETE",
        "results_json": relpath(results_path),
        "summary_md": relpath(summary_path),
        "n_arms": len(all_results),
        "device": str(device),
        "no_official_evaluation_upload_or_leaderboard": True,
    }, indent=2, ensure_ascii=False), flush=True)


def write_summary(path: Path, results: List[Dict]):
    lines = []
    lines.append("# research symmetry-identification learned pilot")
    lines.append("")
    lines.append("## Predicted pattern")
    lines.append("")
    lines.append("| arm | heldheld_closure | mixed_orientation | state_changed | state_unchanged |")
    lines.append("|---|---|---|---|---|")
    lines.append("| exposure_only | chance | chance | chance | chance |")
    lines.append("| heldheld_only | high | ~0.500 | varies | varies |")
    lines.append("| aligned_state | high | high | high | high |")
    lines.append("| inverted_state | high | LOW (~0.0) | inverted | varies |")
    lines.append("| neutral | high | ~0.500 | varies | varies |")
    lines.append("| mixed_event | high | high | varies | varies |")
    lines.append("")
    lines.append("## Observed results")
    lines.append("")
    header = "| arm | train_acc | hh_closure | mixed_orient | state_conserv | state_conserv_chg | state_conserv_unchg | cross_tmpl | name_perm |"
    lines.append(header)
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        def g(suite, key="accuracy"):
            v = r.get(suite, {}).get(key)
            return f"{v:.3f}" if v is not None else "nan"
        def gc(suite):
            v = r.get(suite, {}).get("acc_changed")
            return f"{v:.3f}" if v is not None else "nan"
        def gu(suite):
            v = r.get(suite, {}).get("acc_unchanged")
            return f"{v:.3f}" if v is not None else "nan"
        ta = r.get("train_acc_last")
        ta_s = f"{ta:.3f}" if ta is not None else "nan"
        lines.append(f"| {r['arm']} | {ta_s} | {g('heldheld_unseen_edge_closure')} | {g('mixed_held_seen_orientation')} | {g('paired_state_conservation')} | {gc('paired_state_conservation')} | {gu('paired_state_conservation')} | {g('cross_template_state_readout')} | {g('name_permutation_counterfactual')} |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("If aligned >> inverted on mixed_orientation while both high on hh_closure, sparse bridges identify absolute role orientation from nonce lexical components. If heldheld_only and neutral are near 0.500 on mixed, the Z2 ambiguity is real and cannot be resolved without bridge evidence.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
