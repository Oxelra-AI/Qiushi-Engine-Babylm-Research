#!/usr/bin/env python3
"""research learned pilot on the repaired equivariant symmetry substrate.

Scientific purpose
------------------
After research exposed a deterministic 0.750 name/order solution, research first
repaired the interface and verified transparent order-only solvers are at chance
on the critical readouts.  This script now runs the minimum learned comparison:
pretrained BabyLM DeBERTa vs same-configuration random initialization under
heldheld-only, aligned, inverted, neutral, and mixed bridges.

Decisive signatures on the repaired surface:
  * heldheld_only / neutral_decoupled: held-held consistency may fit, but mixed
    held-seen orientation should remain ambiguous (near 0.5 or seed-variable).
  * aligned_state_bridge: changed-state and mixed readouts should move toward
    TRUE assignment while unaffected facts are preserved.
  * inverted_state_bridge: changed-state and mixed readouts should move toward
    INVERTED assignment (low accuracy against true labels) while unaffected facts
    are preserved.
  * mixed_event_bridge: mixed comparison evidence should identify TRUE on mixed
    readout, but state conservation must still be separately checked.

The run is a small mechanism pilot, not BabyLM training or official evaluation.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import copy
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, DebertaV2Config, DebertaV2ForMaskedLM, DebertaV2ForSequenceClassification

AI_LAB_DIR = _public_path('experiments/archive/representation_and_objectives/training')
WORKSPACE = _public_path('experiments/archive/representation_and_objectives')
PROJECT_ROOT = _public_path('experiments/archive')
SUBSTRATE_DIR = _public_path('experiments/archive/representation_and_objectives/data/equivariant_symmetry_substrate')
MODEL_PATH = _public_path('experiments/archive/representation_and_objectives/training/runs/qwen_8x480_16k_wwm_to_token_100M_seed43022/hf_model/chck_80M')
OUT_DEFAULT = _public_path('experiments/archive/representation_and_objectives/data/equivariant_model_pilot')

ARMS_ALL = [
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


def format_input(row: Dict[str, Any]) -> Tuple[str, str | None]:
    if row.get("task") == "state_query":
        return row["premise"], row["hypothesis"]
    return row.get("text", ""), None


class SubstrateDataset(Dataset):
    def __init__(self, rows: Sequence[Dict[str, Any]], tokenizer, max_len: int):
        self.rows = list(rows)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        a, b = format_input(row)
        enc = self.tokenizer(a, b, max_length=self.max_len, truncation=True, padding="max_length", return_tensors="pt")
        token_type_ids = enc.get("token_type_ids")
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(enc["input_ids"])
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": token_type_ids.squeeze(0),
            "label": torch.tensor(int(bool(row["label"])), dtype=torch.long),
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(init_mode: str, model_path: Path, device: torch.device, seed: int) -> DebertaV2ForSequenceClassification:
    config = DebertaV2Config.from_pretrained(str(model_path))
    config.num_labels = 2
    set_seed(seed)
    model = DebertaV2ForSequenceClassification(config)
    if init_mode == "pretrained":
        mlm = DebertaV2ForMaskedLM.from_pretrained(str(model_path))
        model.deberta.load_state_dict(mlm.deberta.state_dict())
        del mlm
    elif init_mode == "random":
        pass
    else:
        raise ValueError(f"unknown init_mode {init_mode}")
    return model.to(device)


def token_length_stats(rows: Sequence[Dict[str, Any]], tokenizer, max_len: int) -> Dict[str, Any]:
    lens: List[int] = []
    for r in rows:
        a, b = format_input(r)
        enc = tokenizer(a, b, truncation=False)
        lens.append(len(enc["input_ids"]))
    if not lens:
        return {"n": 0}
    return {
        "n": len(lens),
        "max": max(lens),
        "p95": float(np.percentile(lens, 95)),
        "over_max_len": sum(x > max_len for x in lens),
    }


def train_one(
    init_mode: str,
    arm: str,
    train_rows: Sequence[Dict[str, Any]],
    evals: Dict[str, List[Dict[str, Any]]],
    tokenizer,
    model_path: Path,
    device: torch.device,
    seed: int,
    epochs: int,
    lr: float,
    batch_size: int,
    max_len: int,
) -> Dict[str, Any]:
    t0 = time.time()
    set_seed(seed)
    labeled = [r for r in train_rows if "label" in r]
    model = build_model(init_mode, model_path, device, seed)

    out: Dict[str, Any] = {
        "init_mode": init_mode,
        "arm": arm,
        "seed": seed,
        "train_labeled": len(labeled),
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "max_len": max_len,
        "train_token_lengths": token_length_stats(labeled, tokenizer, max_len),
    }

    if labeled:
        ds = SubstrateDataset(labeled, tokenizer, max_len)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        first_loss = None
        history = []
        for epoch in range(epochs):
            model.train()
            loss_sum = 0.0
            correct = 0
            total = 0
            for batch in loader:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                tids = batch["token_type_ids"].to(device)
                labs = batch["label"].to(device)
                res = model(input_ids=ids, attention_mask=mask, token_type_ids=tids, labels=labs)
                loss = res.loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                loss_sum += float(loss.item()) * ids.size(0)
                pred = res.logits.argmax(-1)
                correct += int((pred == labs).sum().item())
                total += int(labs.numel())
            avg_loss = loss_sum / max(total, 1)
            acc = correct / max(total, 1)
            if first_loss is None:
                first_loss = avg_loss
            if epoch == epochs - 1 or epoch in {0, 1, 2, 4, 9, 19, 39, 79}:
                history.append({"epoch": epoch + 1, "loss": round(avg_loss, 5), "acc": round(acc, 5)})
        out.update({
            "train_loss_first": round(first_loss, 5),
            "train_loss_last": history[-1]["loss"],
            "train_acc_last": history[-1]["acc"],
            "train_history_sparse": history,
        })
        del opt
    else:
        out.update({"train_loss_first": None, "train_loss_last": None, "train_acc_last": None, "train_history_sparse": []})

    model.eval()
    for suite, rows in evals.items():
        out[suite] = eval_suite(model, rows, tokenizer, device, max_len, batch_size)
    out["elapsed_sec"] = round(time.time() - t0, 1)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def eval_suite(model, rows: Sequence[Dict[str, Any]], tokenizer, device: torch.device, max_len: int, batch_size: int) -> Dict[str, Any]:
    labeled = [r for r in rows if "label" in r]
    if not labeled:
        return {"n": 0, "accuracy": None}
    ds = SubstrateDataset(labeled, tokenizer, max_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)
    preds: List[int] = []
    labels: List[int] = []
    margins: List[float] = []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            tids = batch["token_type_ids"].to(device)
            res = model(input_ids=ids, attention_mask=mask, token_type_ids=tids)
            logits = res.logits.detach().cpu()
            pred = logits.argmax(-1).tolist()
            lab = batch["label"].tolist()
            preds.extend(int(x) for x in pred)
            labels.extend(int(x) for x in lab)
            margins.extend((logits[:, 1] - logits[:, 0]).tolist())
    n = len(labels)
    acc = sum(p == y for p, y in zip(preds, labels)) / n
    out: Dict[str, Any] = {
        "n": n,
        "accuracy": round(acc, 5),
        "pred_true_frac": round(sum(preds) / n, 5),
        "label_true_frac": round(sum(labels) / n, 5),
        "mean_margin": round(float(np.mean(margins)), 5),
        "mean_abs_margin": round(float(np.mean(np.abs(margins))), 5),
    }
    for task in sorted(set(r.get("task") for r in labeled)):
        idx = [i for i, r in enumerate(labeled) if r.get("task") == task]
        out[f"acc_task_{task}"] = round(sum(preds[i] == labels[i] for i in idx) / len(idx), 5)
    for q in ["changed", "unchanged"]:
        idx = [i for i, r in enumerate(labeled) if r.get("query_kind") == q]
        if idx:
            out[f"acc_{q}"] = round(sum(preds[i] == labels[i] for i in idx) / len(idx), 5)
            out[f"pred_true_frac_{q}"] = round(sum(preds[i] for i in idx) / len(idx), 5)
            out[f"mean_margin_{q}"] = round(float(np.mean([margins[i] for i in idx])), 5)
    for dep in sorted(set(r.get("orientation_dependency") for r in labeled)):
        if dep is None:
            continue
        idx = [i for i, r in enumerate(labeled) if r.get("orientation_dependency") == dep]
        out[f"acc_dep_{dep}"] = round(sum(preds[i] == labels[i] for i in idx) / len(idx), 5)

    # Paired conservation states: every pair_id has changed true/false rows and unchanged true/false rows.
    if any(r.get("task") == "state_query" for r in labeled):
        by_pair: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: {"idxs": []})
        for i, r in enumerate(labeled):
            if r.get("task") == "state_query":
                by_pair[r["pair_id"]]["idxs"].append(i)
        state_counts = Counter()
        for pid, d in by_pair.items():
            idxs = d["idxs"]
            ch = [i for i in idxs if labeled[i].get("query_kind") == "changed"]
            un = [i for i in idxs if labeled[i].get("query_kind") == "unchanged"]
            changed_ok = bool(ch) and all(preds[i] == labels[i] for i in ch)
            unchanged_ok = bool(un) and all(preds[i] == labels[i] for i in un)
            if changed_ok and unchanged_ok:
                state_counts["both_correct"] += 1
            elif changed_ok:
                state_counts["changed_only"] += 1
            elif unchanged_ok:
                state_counts["unchanged_only"] += 1
            else:
                state_counts["neither"] += 1
        denom = max(sum(state_counts.values()), 1)
        out["pair_state_n"] = denom
        for k in ["both_correct", "changed_only", "unchanged_only", "neither"]:
            out[f"pair_{k}"] = round(state_counts[k] / denom, 5)
    return out


def write_summary(path: Path, results: Sequence[Dict[str, Any]]) -> None:
    lines: List[str] = []
    lines.append("# research equivariant learned pilot")
    lines.append("")
    lines.append("Small mechanism pilot on the repaired active/passive symmetry surface. Scores are against the true assignment; inverted orientation should therefore appear as low changed/mixed accuracy rather than high accuracy.")
    lines.append("")
    lines.append("## Per-run critical metrics")
    lines.append("")
    lines.append("| init | arm | seed | train acc | hh | mixed | state chg | state unchg | pair both | pair chg-only | pair unchg-only |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        def g(s: str, k: str = "accuracy") -> str:
            v = r.get(s, {}).get(k)
            if v is None:
                return "nan"
            return f"{float(v):.3f}"
        ta = r.get("train_acc_last")
        lines.append(
            f"| {r['init_mode']} | {r['arm']} | {r['seed']} | {f'{float(ta):.3f}' if ta is not None else 'nan'} | "
            f"{g('heldheld_unseen_edge_closure')} | {g('mixed_held_seen_orientation')} | "
            f"{g('paired_state_conservation', 'acc_changed')} | {g('paired_state_conservation', 'acc_unchanged')} | "
            f"{g('paired_state_conservation', 'pair_both_correct')} | {g('paired_state_conservation', 'pair_changed_only')} | {g('paired_state_conservation', 'pair_unchanged_only')} |"
        )
    lines.append("")
    lines.append("## Cross-seed means")
    lines.append("")
    lines.append("| init | arm | n | mean train | mean hh | mean mixed | std mixed | mean chg | mean unchg | mean pair both |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    keys = sorted(set((r["init_mode"], r["arm"]) for r in results))
    for init, arm in keys:
        rs = [r for r in results if r["init_mode"] == init and r["arm"] == arm]
        def vals(s: str, k: str = "accuracy") -> List[float]:
            return [float(r.get(s, {}).get(k)) for r in rs if r.get(s, {}).get(k) is not None]
        def mean(xs: List[float]) -> str:
            return f"{float(np.mean(xs)):.3f}" if xs else "nan"
        def std(xs: List[float]) -> str:
            return f"{float(np.std(xs)):.3f}" if xs else "nan"
        tas = [float(r["train_acc_last"]) for r in rs if r.get("train_acc_last") is not None]
        mxs = vals("mixed_held_seen_orientation")
        lines.append(
            f"| {init} | {arm} | {len(rs)} | {mean(tas)} | {mean(vals('heldheld_unseen_edge_closure'))} | "
            f"{mean(mxs)} | {std(mxs)} | {mean(vals('paired_state_conservation', 'acc_changed'))} | "
            f"{mean(vals('paired_state_conservation', 'acc_unchanged'))} | {mean(vals('paired_state_conservation', 'pair_both_correct'))} |"
        )
    lines.append("")
    lines.append("## Reading")
    lines.append("")
    lines.append("Aligned-vs-inverted evidence is mechanism-relevant only if both arms fit training and preserve unaffected facts, while changing mixed held-seen and changed-state orientation in opposite directions. If all arms collapse to the same mixed score, the run is task calibration or residual surface/pretraining bias rather than sparse coordinate identification.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def relpath(p: Path) -> str:
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrate-dir", type=Path, default=SUBSTRATE_DIR)
    ap.add_argument("--model-path", type=Path, default=MODEL_PATH)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--init-mode", choices=["pretrained", "random"], required=True)
    ap.add_argument("--arms", nargs="*", default=["heldheld_only", "aligned_state_bridge", "inverted_state_bridge", "neutral_decoupled", "mixed_event_bridge"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[27800, 27801, 27802])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--no-common-seen", action="store_true")
    args = ap.parse_args()

    if args.epochs is None:
        args.epochs = 40 if args.init_mode == "pretrained" else 80
    if args.lr is None:
        args.lr = 2e-5 if args.init_mode == "pretrained" else 3e-4

    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path))
    print(f"Device={device}; init={args.init_mode}; arms={args.arms}; seeds={args.seeds}; epochs={args.epochs}; lr={args.lr}", flush=True)

    common = [] if args.no_common_seen else load_jsonl(args.substrate_dir / "common_seen_train.jsonl")
    evals: Dict[str, List[Dict[str, Any]]] = {}
    for suite in EVAL_SUITES:
        p = args.substrate_dir / "eval" / f"{suite}.jsonl"
        evals[suite] = load_jsonl(p)
        print(f"Eval {suite}: {len(evals[suite])} rows", flush=True)
    all_results: List[Dict[str, Any]] = []
    for seed in args.seeds:
        for arm in args.arms:
            sup = load_jsonl(args.substrate_dir / "arms" / arm / "train_supervised.jsonl")
            train_rows = sup + common
            print(f"\n=== init={args.init_mode} arm={arm} seed={seed} train_rows={len(train_rows)} ===", flush=True)
            res = train_one(args.init_mode, arm, train_rows, evals, tokenizer, args.model_path, device,
                            seed, args.epochs, args.lr, args.batch_size, args.max_len)
            all_results.append(res)
            print(
                f"train={res.get('train_acc_last')} hh={res['heldheld_unseen_edge_closure']['accuracy']} "
                f"mixed={res['mixed_held_seen_orientation']['accuracy']} "
                f"state_chg={res['paired_state_conservation'].get('acc_changed')} "
                f"state_unchg={res['paired_state_conservation'].get('acc_unchanged')} "
                f"pair_both={res['paired_state_conservation'].get('pair_both_correct')} elapsed={res['elapsed_sec']}s",
                flush=True,
            )

    tag = f"{args.init_mode}_{'nocommon' if args.no_common_seen else 'withcommon'}"
    results_path = args.out / f"equivariant_model_pilot_{tag}.json"
    summary_path = args.out / f"equivariant_model_pilot_{tag}_summary.md"
    results_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    write_summary(summary_path, all_results)
    print(json.dumps({
        "status": "EQUIVARIANT_MODEL_PILOT_COMPLETE",
        "init_mode": args.init_mode,
        "with_common_seen": not args.no_common_seen,
        "results_json": relpath(results_path),
        "summary_md": relpath(summary_path),
        "n_runs": len(all_results),
        "device": str(device),
        "no_official_evaluation_upload_or_leaderboard": True,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
