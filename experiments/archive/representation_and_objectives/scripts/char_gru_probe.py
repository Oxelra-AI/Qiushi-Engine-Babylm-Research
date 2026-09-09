#!/usr/bin/env python3
"""research char-GRU learned probe on the repaired equivariant substrate.

This is a low-cost replacement for full random-init DeBERTa fine-tuning after
that run proved too expensive and underfit.  It uses raw text only (no row
metadata in the input), train/eval-disjoint names and objects, and a small
character-level BiGRU.  The aim is not to claim BabyLM-scale behavior, but to
ask whether the repaired surface is learnable by a generic sequence model and
whether sparse aligned/inverted bridges produce opposite orientations once the
research 0.750 name/order shortcut has been removed.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

WORKSPACE = _public_path('experiments/archive/representation_and_objectives')
PROJECT_ROOT = _public_path('.')
SUBSTRATE_DEFAULT = _public_path('experiments/archive/representation_and_objectives/data/equivariant_symmetry_substrate')
OUT_DEFAULT = _public_path('experiments/archive/representation_and_objectives/data/char_gru_probe')
ARMS_DEFAULT = ["heldheld_only", "aligned_state_bridge", "inverted_state_bridge", "neutral_decoupled", "mixed_event_bridge"]
EVAL_SUITES = ["heldheld_unseen_edge_closure", "mixed_held_seen_orientation", "paired_state_conservation", "cross_template_state_readout", "name_permutation_counterfactual"]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def row_text(row: Dict[str, Any]) -> str:
    if row.get("task") == "state_query":
        return row.get("premise", "") + " [SEP] " + row.get("hypothesis", "")
    return row.get("text", "")


def build_vocab(all_rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    chars = sorted(set("".join(row_text(r) for r in all_rows)))
    # 0 pad, 1 unknown
    return {ch: i + 2 for i, ch in enumerate(chars)}


class CharDataset(Dataset):
    def __init__(self, rows: Sequence[Dict[str, Any]], vocab: Dict[str, int], max_len: int):
        self.rows = [r for r in rows if "label" in r]
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        r = self.rows[idx]
        ids = [self.vocab.get(ch, 1) for ch in row_text(r)[: self.max_len]]
        length = max(1, len(ids))
        if len(ids) < self.max_len:
            ids += [0] * (self.max_len - len(ids))
        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "length": torch.tensor(length, dtype=torch.long),
            "label": torch.tensor(int(bool(r["label"])), dtype=torch.long),
        }


class CharGRU(nn.Module):
    def __init__(self, vocab_size: int, emb: int = 48, hidden: int = 96, dropout: float = 0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.gru = nn.GRU(emb, hidden, batch_first=True, bidirectional=True)
        self.norm = nn.LayerNorm(hidden * 4)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden * 4, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 2)
        )

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.emb(ids)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, h = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=ids.size(1))
        # Mean pool over nonpad plus max pool.
        mask = (ids != 0).unsqueeze(-1)
        out_masked = out.masked_fill(~mask, 0.0)
        mean = out_masked.sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        maxp = out.masked_fill(~mask, -1e9).max(dim=1).values
        feat = self.norm(torch.cat([mean, maxp], dim=-1))
        return self.fc(self.dropout(feat))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_eval_one(arm: str, seed: int, train_rows: Sequence[Dict[str, Any]], evals: Dict[str, List[Dict[str, Any]]], vocab: Dict[str, int], args) -> Dict[str, Any]:
    set_seed(seed)
    device = torch.device(args.device)
    model = CharGRU(len(vocab) + 2, emb=args.emb, hidden=args.hidden, dropout=args.dropout).to(device)
    ds = CharDataset(train_rows, vocab, args.max_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history: List[Dict[str, float]] = []
    t0 = time.time()
    first_loss = None
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        correct = 0
        total = 0
        for batch in loader:
            ids = batch["ids"].to(device)
            lengths = batch["length"].to(device)
            labs = batch["label"].to(device)
            logits = model(ids, lengths)
            loss = nn.functional.cross_entropy(logits, labs)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += float(loss.item()) * labs.numel()
            pred = logits.argmax(-1)
            correct += int((pred == labs).sum().item())
            total += int(labs.numel())
        avg = loss_sum / max(total, 1)
        acc = correct / max(total, 1)
        if first_loss is None:
            first_loss = avg
        if epoch == args.epochs - 1 or epoch in {0, 1, 2, 4, 9, 19, 39, 79, 119}:
            history.append({"epoch": epoch + 1, "loss": round(avg, 5), "acc": round(acc, 5)})
    res: Dict[str, Any] = {
        "arm": arm,
        "seed": seed,
        "train_labeled": len(ds),
        "train_loss_first": round(float(first_loss), 5) if first_loss is not None else None,
        "train_loss_last": history[-1]["loss"] if history else None,
        "train_acc_last": history[-1]["acc"] if history else None,
        "history": history,
    }
    model.eval()
    for suite, rows in evals.items():
        res[suite] = eval_suite(model, rows, vocab, args)
    res["elapsed_sec"] = round(time.time() - t0, 1)
    return res


def eval_suite(model: nn.Module, rows: Sequence[Dict[str, Any]], vocab: Dict[str, int], args) -> Dict[str, Any]:
    ds = CharDataset(rows, vocab, args.max_len)
    if len(ds) == 0:
        return {"n": 0, "accuracy": None}
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    device = torch.device(args.device)
    preds: List[int] = []
    labels: List[int] = []
    margins: List[float] = []
    with torch.no_grad():
        for batch in loader:
            ids = batch["ids"].to(device)
            lengths = batch["length"].to(device)
            labs = batch["label"].numpy().astype(int).tolist()
            logits = model(ids, lengths).cpu()
            pred = logits.argmax(-1).numpy().astype(int).tolist()
            preds.extend(pred)
            labels.extend(labs)
            margins.extend((logits[:, 1] - logits[:, 0]).numpy().astype(float).tolist())
    n = len(labels)
    out: Dict[str, Any] = {
        "n": n,
        "accuracy": round(sum(p == y for p, y in zip(preds, labels)) / n, 5),
        "pred_true_frac": round(sum(preds) / n, 5),
        "label_true_frac": round(sum(labels) / n, 5),
        "mean_margin": round(float(np.mean(margins)), 5),
        "mean_abs_margin": round(float(np.mean(np.abs(margins))), 5),
    }
    for q in ["changed", "unchanged"]:
        idx = [i for i, r in enumerate(ds.rows) if r.get("query_kind") == q]
        if idx:
            out[f"acc_{q}"] = round(sum(preds[i] == labels[i] for i in idx) / len(idx), 5)
            out[f"pred_true_frac_{q}"] = round(sum(preds[i] for i in idx) / len(idx), 5)
    for dep in sorted(set(r.get("orientation_dependency") for r in ds.rows)):
        if dep is None:
            continue
        idx = [i for i, r in enumerate(ds.rows) if r.get("orientation_dependency") == dep]
        if idx:
            out[f"acc_dep_{dep}"] = round(sum(preds[i] == labels[i] for i in idx) / len(idx), 5)
    if any(r.get("task") == "state_query" for r in ds.rows):
        by_pair: Dict[str, List[int]] = defaultdict(list)
        for i, r in enumerate(ds.rows):
            if r.get("task") == "state_query":
                by_pair[str(r.get("pair_id"))].append(i)
        c = Counter()
        for idxs in by_pair.values():
            ch = [i for i in idxs if ds.rows[i].get("query_kind") == "changed"]
            un = [i for i in idxs if ds.rows[i].get("query_kind") == "unchanged"]
            cok = bool(ch) and all(preds[i] == labels[i] for i in ch)
            uok = bool(un) and all(preds[i] == labels[i] for i in un)
            c["both_correct" if cok and uok else "changed_only" if cok else "unchanged_only" if uok else "neither"] += 1
        denom = max(sum(c.values()), 1)
        for k in ["both_correct", "changed_only", "unchanged_only", "neither"]:
            out[f"pair_{k}"] = round(c[k] / denom, 5)
    return out


def fmt(x: Any) -> str:
    if x is None:
        return "nan"
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def write_summary(path: Path, results: Sequence[Dict[str, Any]]) -> None:
    lines: List[str] = []
    lines.append("# research char-GRU repaired-surface probe")
    lines.append("")
    lines.append("Small character-level BiGRU trained from scratch on raw text. Scores are against the true assignment; an inverted bridge that successfully installs the opposite orientation should have low mixed/changed accuracy against these true labels while retaining held-held consistency.")
    lines.append("")
    lines.append("## Per-run metrics")
    lines.append("")
    lines.append("| arm | seed | train acc | hh | mixed | state chg | state unchg | pair both | pair chg-only | pair unchg-only |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        def g(s: str, k: str = "accuracy") -> str:
            v = r.get(s, {}).get(k)
            return fmt(float(v)) if v is not None else "nan"
        ta = r.get("train_acc_last")
        lines.append(f"| {r['arm']} | {r['seed']} | {fmt(float(ta)) if ta is not None else 'nan'} | {g('heldheld_unseen_edge_closure')} | {g('mixed_held_seen_orientation')} | {g('paired_state_conservation', 'acc_changed')} | {g('paired_state_conservation', 'acc_unchanged')} | {g('paired_state_conservation', 'pair_both_correct')} | {g('paired_state_conservation', 'pair_changed_only')} | {g('paired_state_conservation', 'pair_unchanged_only')} |")
    lines.append("")
    lines.append("## Cross-seed means")
    lines.append("")
    lines.append("| arm | n | train | hh | mixed | std mixed | changed | unchanged | pair both |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for arm in sorted(set(r["arm"] for r in results)):
        rs = [r for r in results if r["arm"] == arm]
        def vals(s: str, k: str = "accuracy") -> List[float]:
            return [float(r.get(s, {}).get(k)) for r in rs if r.get(s, {}).get(k) is not None]
        def mean(xs: List[float]) -> str:
            return fmt(float(np.mean(xs))) if xs else "nan"
        def std(xs: List[float]) -> str:
            return fmt(float(np.std(xs))) if xs else "nan"
        tas = [float(r["train_acc_last"]) for r in rs if r.get("train_acc_last") is not None]
        mxs = vals("mixed_held_seen_orientation")
        lines.append(f"| {arm} | {len(rs)} | {mean(tas)} | {mean(vals('heldheld_unseen_edge_closure'))} | {mean(mxs)} | {std(mxs)} | {mean(vals('paired_state_conservation', 'acc_changed'))} | {mean(vals('paired_state_conservation', 'acc_unchanged'))} | {mean(vals('paired_state_conservation', 'pair_both_correct'))} |")
    lines.append("")
    lines.append("## Scientific reading")
    lines.append("")
    lines.append("A useful bridge-identification signal requires fitted held-held consistency, aligned and inverted state bridges moving changed/mixed orientation in opposite directions, neutral remaining ambiguous, and unaffected facts not being traded against changed-state accuracy.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrate", type=Path, default=SUBSTRATE_DEFAULT)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--arms", nargs="*", default=ARMS_DEFAULT)
    ap.add_argument("--seeds", nargs="*", type=int, default=[27800, 27801, 27802])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--emb", type=int, default=48)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--no-common-seen", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(min(8, max(1, torch.get_num_threads())))
    args.out.mkdir(parents=True, exist_ok=True)
    common = [] if args.no_common_seen else load_jsonl(args.substrate / "common_seen_train.jsonl")
    arms = {a: load_jsonl(args.substrate / "arms" / a / "train_supervised.jsonl") for a in args.arms}
    evals = {s: load_jsonl(args.substrate / "eval" / f"{s}.jsonl") for s in EVAL_SUITES}
    vocab = build_vocab(common + [r for rows in arms.values() for r in rows] + [r for rows in evals.values() for r in rows])

    all_results: List[Dict[str, Any]] = []
    print(f"device={args.device} arms={args.arms} seeds={args.seeds} epochs={args.epochs} vocab={len(vocab)+2} common={len(common)}", flush=True)
    for seed in args.seeds:
        for arm in args.arms:
            train = arms[arm] + common
            print(f"\n=== char-gru arm={arm} seed={seed} train={len(train)} ===", flush=True)
            res = train_eval_one(arm, seed, train, evals, vocab, args)
            all_results.append(res)
            print(
                f"train={res.get('train_acc_last')} hh={res['heldheld_unseen_edge_closure']['accuracy']} "
                f"mixed={res['mixed_held_seen_orientation']['accuracy']} changed={res['paired_state_conservation'].get('acc_changed')} "
                f"unchanged={res['paired_state_conservation'].get('acc_unchanged')} pair_both={res['paired_state_conservation'].get('pair_both_correct')} elapsed={res['elapsed_sec']}s",
                flush=True,
            )

    tag = "nocommon" if args.no_common_seen else "withcommon"
    jp = args.out / f"char_gru_probe_{tag}.json"
    sp = args.out / f"char_gru_probe_{tag}_summary.md"
    jp.write_text(json.dumps(all_results, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    write_summary(sp, all_results)
    print(json.dumps({
        "status": "CHAR_GRU_PROBE_COMPLETE",
        "with_common_seen": not args.no_common_seen,
        "summary_md": str(sp.relative_to(PROJECT_ROOT)),
        "results_json": str(jp.relative_to(PROJECT_ROOT)),
        "n_runs": len(all_results),
        "device": args.device,
        "no_official_evaluation_upload_or_leaderboard": True,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
