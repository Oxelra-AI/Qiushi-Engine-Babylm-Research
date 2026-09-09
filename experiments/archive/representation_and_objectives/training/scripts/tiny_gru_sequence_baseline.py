#!/usr/bin/env python3
"""research tiny GRU sequence baseline for paired-world shortcut stress.

This is a small learned shortcut/stress probe, not BabyLM pretraining. It trains
simple sequence classifiers on the research paired-world NLI rows to test whether
the current sports-outcome corpus can be solved by finite template/order parsing,
and whether such parsing transfers to held templates and hypothesis predicates.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import collections
import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

STUDY = Path("experiments/archive/representation_and_objectives")
DATA_DIR = STUDY / "data/paired_world_sequence_teacher"
ORIG = DATA_DIR / "sequence_examples_original_ablated.jsonl"
PERT = DATA_DIR / "sequence_examples_perturbations.jsonl"
SEED = 252411


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def toks(s: str) -> list[str]:
    return re.findall(r"[A-Za-z_]+|\d+|[^\sA-Za-z_\d]", str(s).lower())


def text(row: dict[str, Any], mode: str) -> str:
    if mode == "canonical":
        return row["canonical_text"] + " [SEP] " + row["canonical_hyp"]
    if mode == "raw":
        return row["text"] + " [SEP] " + row["hyp"]
    raise ValueError(mode)


def build_vocab(rows: list[dict[str, Any]], mode: str, min_freq: int = 1) -> dict[str, int]:
    cnt = collections.Counter()
    for r in rows:
        cnt.update(toks(text(r, mode)))
    vocab = {"<pad>": 0, "<unk>": 1}
    for w, c in cnt.most_common():
        if c >= min_freq and w not in vocab:
            vocab[w] = len(vocab)
    return vocab


def encode(rows: list[dict[str, Any]], mode: str, vocab: dict[str, int], max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    for r in rows:
        ids = [vocab.get(t, 1) for t in toks(text(r, mode))][:max_len]
        ids += [0] * (max_len - len(ids))
        xs.append(ids)
        ys.append(float(r["y"]))
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.float32)


class GRUClassifier(nn.Module):
    def __init__(self, vocab_size: int, emb: int = 64, hid: int = 64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.gru = nn.GRU(emb, hid, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(2 * hid, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.emb(x)
        out, h = self.gru(e)
        hcat = torch.cat([h[-2], h[-1]], dim=-1)
        return self.head(hcat).squeeze(-1)


def accuracy(model: nn.Module, rows: list[dict[str, Any]], mode: str, vocab: dict[str, int], max_len: int, batch: int = 256) -> float:
    if not rows:
        return float("nan")
    model.eval()
    x, y = encode(rows, mode, vocab, max_len)
    ok, n = 0, 0
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            logits = model(x[i:i+batch])
            pred = (torch.sigmoid(logits) >= 0.5).float()
            ok += int((pred == y[i:i+batch]).sum().item())
            n += len(pred)
    return ok / max(1, n)


def train_one(train_rows: list[dict[str, Any]], eval_sets: dict[str, list[dict[str, Any]]], mode: str, epochs: int, lr: float, seed_offset: int) -> dict[str, Any]:
    random.seed(SEED + seed_offset)
    torch.manual_seed(SEED + seed_offset)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    vocab = build_vocab(train_rows, mode)
    max_len = min(160, max(len(toks(text(r, mode))) for r in train_rows + [x for rows in eval_sets.values() for x in rows]))
    x, y = encode(train_rows, mode, vocab, max_len)
    ds = TensorDataset(x, y)
    gen = torch.Generator().manual_seed(SEED + seed_offset)
    loader = DataLoader(ds, batch_size=64, shuffle=True, generator=gen)
    model = GRUClassifier(len(vocab))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    history = []
    best = None
    best_state = None
    for ep in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * len(xb)
            n += len(xb)
        rec = {"epoch": ep, "train_loss": total / max(1, n), "train_acc": accuracy(model, train_rows, mode, vocab, max_len)}
        # Track held original as the selector only for reporting a stable trained model.
        if "held_original_all" in eval_sets:
            rec["held_original_all"] = accuracy(model, eval_sets["held_original_all"], mode, vocab, max_len)
        history.append(rec)
        score = rec.get("held_original_all", rec["train_acc"])
        if best is None or score > best:
            best = score
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    eval_out = {name: {"n": len(rows), "accuracy": accuracy(model, rows, mode, vocab, max_len)} for name, rows in eval_sets.items()}
    return {"mode": mode, "epochs": epochs, "vocab_size": len(vocab), "max_len": max_len, "best_selector_score": best, "eval": eval_out, "history_tail": history[-5:]}


def subset(rows: list[dict[str, Any]], **conds: str) -> list[dict[str, Any]]:
    out = rows
    for k, v in conds.items():
        out = [r for r in out if str(r.get(k)) == str(v)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()
    run_dir = Path(os.environ.get("QIUSHI_AI_LAB_RUN_DIR", STUDY / "training/runs/tiny_gru_sequence_baseline_direct"))
    run_dir.mkdir(parents=True, exist_ok=True)
    orig = read_jsonl(ORIG)
    pert = read_jsonl(PERT)
    train_orig = subset(orig, family_split="train")
    held_orig = subset(orig, family_split="held")
    held_orig_train_templates = [r for r in held_orig if r.get("context_template_split") == "train"]
    held_orig_held_templates = [r for r in held_orig if r.get("context_template_split") == "held"]
    eval_sets = {
        "held_original_all": held_orig,
        "held_original_train_templates": held_orig_train_templates,
        "held_original_held_templates": held_orig_held_templates,
        "held_name_swapped_context": [r for r in pert if r.get("family_split") == "held" and r.get("variant") == "name_swapped_context"],
        "held_hyp_lost_to": [r for r in pert if r.get("family_split") == "held" and r.get("variant") == "hyp_lost_to"],
        "held_hyp_passive": [r for r in pert if r.get("family_split") == "held" and r.get("variant") == "hyp_passive"],
        "held_retemplate_opposite_order": [r for r in pert if r.get("family_split") == "held" and r.get("variant") == "retemplate_opposite_order"],
    }
    # Original-only training probes whether finite train-template role parsing is learnable.
    results = {"orig_only": {}}
    for mode in ["canonical", "raw"]:
        results["orig_only"][mode] = train_one(train_orig, eval_sets, mode, args.epochs, args.lr, seed_offset=0 if mode == "canonical" else 10)
    # Augmented training probes whether the same small learner can master the finite perturbation set when shown it.
    aug_train = list(train_orig) + [r for r in pert if r.get("family_split") == "train" and r.get("variant") in {"name_swapped_context", "hyp_lost_to", "hyp_passive", "retemplate_opposite_order"}]
    aug_eval_sets = dict(eval_sets)
    results["augmented_variants"] = {}
    for mode in ["canonical", "raw"]:
        results["augmented_variants"][mode] = train_one(aug_train, aug_eval_sets, mode, args.epochs, args.lr, seed_offset=100 if mode == "canonical" else 110)
    summary = {
        "status": "TINY_GRU_SEQUENCE_BASELINE_COMPLETED",
        "created_utc": now(),
        "script": str(_public_path('experiments/archive/representation_and_objectives/training/scripts/tiny_gru_sequence_baseline.py')),
        "input_original": str(ORIG),
        "input_perturbations": str(PERT),
        "train_orig_rows": len(train_orig),
        "train_aug_rows": len(aug_train),
        "eval_set_sizes": {k: len(v) for k, v in eval_sets.items()},
        "results": results,
        "interpretation": "A high orig-only held score would mean the current sports corpus is learnable by a small sequence parser from finite train templates, not that a transferable data-efficient principle has been found. Cross-template/predicate failures require broader sources before any BabyLM-scale training.",
        "no_babylm_training_eval_upload_submission": True,
    }
    write_json(run_dir / "tiny_gru_sequence_baseline_summary.json", summary)
    # Compact markdown
    md = ["# research tiny GRU sequence baseline", "", "## Purpose", "", "Test a learned sequence parser against the paired-world shortcut concern. This is not BabyLM training.", "", "## Key accuracies", ""]
    for regime, modes in results.items():
        md += [f"### {regime}", "", "| mode | held original | held held-template | name swap | lost-to hyp | passive hyp | retemplate |", "|---|---:|---:|---:|---:|---:|---:|"]
        for mode, res in modes.items():
            ev = res["eval"]
            md.append(f"| {mode} | {ev['held_original_all']['accuracy']:.3f} | {ev['held_original_held_templates']['accuracy']:.3f} | {ev['held_name_swapped_context']['accuracy']:.3f} | {ev['held_hyp_lost_to']['accuracy']:.3f} | {ev['held_hyp_passive']['accuracy']:.3f} | {ev['held_retemplate_opposite_order']['accuracy']:.3f} |")
        md.append("")
    md += ["## Files", "", f"- summary: `{run_dir / 'tiny_gru_sequence_baseline_summary.json'}`"]
    (run_dir / "tiny_gru_sequence_baseline_summary.md").write_text("\n".join(md)+"\n", encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "run_dir": str(run_dir),
        "orig_canonical_held": results["orig_only"]["canonical"]["eval"]["held_original_all"]["accuracy"],
        "orig_canonical_held_template": results["orig_only"]["canonical"]["eval"]["held_original_held_templates"]["accuracy"],
        "orig_canonical_lost_to": results["orig_only"]["canonical"]["eval"]["held_hyp_lost_to"]["accuracy"],
        "aug_canonical_held": results["augmented_variants"]["canonical"]["eval"]["held_original_all"]["accuracy"],
        "aug_canonical_lost_to": results["augmented_variants"]["canonical"]["eval"]["held_hyp_lost_to"]["accuracy"],
    }, indent=2), flush=True)

if __name__ == "__main__":
    main()
