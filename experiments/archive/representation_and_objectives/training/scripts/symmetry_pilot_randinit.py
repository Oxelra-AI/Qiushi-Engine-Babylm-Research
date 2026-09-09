#!/usr/bin/env python3
"""Step277c: random-init pilot — does Z2 ambiguity manifest without pretraining?

Scientific question: with no pretrained weights, does heldheld_only show
seed-dependent mixed orientation (the Z2 ambiguity), and do aligned/inverted
bridges then produce opposite effects?

If yes: confirms frame inheritance resolves Z2 in pretrained models
If no: the architecture/tokenizer itself carries enough structure
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse, json, random, time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DebertaV2Config, DebertaV2ForSequenceClassification

import sys
sys.path.insert(0, str(_public_path('experiments/archive/representation_and_objectives/training/scripts')))
from symmetry_pilot import (
    AI_LAB_DIR, WORKSPACE, SUBSTRATE_DIR, MODEL_PATH,
    ARMS, EVAL_SUITES,
    load_jsonl, SubstrateDataset, eval_suite, format_input,
)

OUT_DEFAULT = WORKSPACE / "data" / "symmetry_pilot_randinit"


def build_random_model(config_path: Path, device: torch.device, seed: int):
    """Create a randomly initialized model (no pretrained weights)."""
    config = DebertaV2Config.from_pretrained(str(config_path))
    config.num_labels = 2
    torch.manual_seed(seed)
    model = DebertaV2ForSequenceClassification(config)
    # All weights are random — no pretrained encoder loaded
    return model.to(device)


def train_and_eval_randinit(
    arm_name: str,
    train_rows: List[Dict],
    eval_suites_dict: Dict[str, List[Dict]],
    tokenizer,
    config_path: Path,
    device: torch.device,
    seed: int,
    epochs: int = 100,
    lr: float = 5e-4,
    batch_size: int = 16,
    max_len: int = 128,
) -> Dict[str, Any]:
    t0 = time.time()
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    labeled = [r for r in train_rows if "label" in r]
    model = build_random_model(config_path, device, seed)

    if not labeled:
        model.eval()
        results = {"arm": arm_name, "seed": seed, "train_rows": 0, "train_labeled": 0,
                   "train_loss_first": None, "train_loss_last": None, "train_acc_last": None}
        for sn, sr in eval_suites_dict.items():
            results[sn] = eval_suite(model, sr, tokenizer, device, max_len, batch_size)
        results["elapsed_sec"] = round(time.time() - t0, 1)
        del model; return results

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    ds = SubstrateDataset(labeled, tokenizer, max_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    first_loss = None; last_loss = None; last_acc = None
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0; correct = 0; total = 0
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            tids = batch["token_type_ids"].to(device)
            labs = batch["label"].to(device)
            out = model(input_ids=ids, attention_mask=mask, token_type_ids=tids, labels=labs)
            total_loss += out.loss.item() * ids.size(0)
            preds = out.logits.argmax(-1)
            correct += (preds == labs).sum().item()
            total += labs.size(0)
            optimizer.zero_grad(); out.loss.backward(); optimizer.step()
        avg = total_loss / max(total, 1); acc = correct / max(total, 1)
        if epoch == 0: first_loss = avg
        last_loss = avg; last_acc = acc

    model.eval()
    results = {
        "arm": arm_name, "seed": seed,
        "train_rows": len(train_rows), "train_labeled": len(labeled),
        "train_loss_first": round(first_loss, 4) if first_loss else None,
        "train_loss_last": round(last_loss, 4) if last_loss else None,
        "train_acc_last": round(last_acc, 4) if last_acc else None,
    }
    for sn, sr in eval_suites_dict.items():
        results[sn] = eval_suite(model, sr, tokenizer, device, max_len, batch_size)
    results["elapsed_sec"] = round(time.time() - t0, 1)
    del model, optimizer; torch.cuda.empty_cache()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrate-dir", type=Path, default=SUBSTRATE_DIR)
    ap.add_argument("--model-path", type=Path, default=MODEL_PATH)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--seeds", nargs="*", type=int, default=[27700, 27701, 27702, 27703, 27704])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    out = args.out; out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path))
    print(f"Device: {device}, random-init model", flush=True)

    eval_suites_dict: Dict[str, List[Dict]] = {}
    for suite in EVAL_SUITES:
        p = args.substrate_dir / "eval" / f"{suite}.jsonl"
        if p.exists():
            eval_suites_dict[suite] = load_jsonl(p)

    all_results = []
    # Focus on the key arms: exposure_only, heldheld_only, aligned, inverted
    focus_arms = ["exposure_only", "heldheld_only", "aligned_state_bridge", "inverted_state_bridge"]

    for seed in args.seeds:
        for arm in focus_arms:
            print(f"\n{'='*60}\nArm: {arm}, seed: {seed}\n{'='*60}", flush=True)
            sup_path = args.substrate_dir / "arms" / arm / "train_supervised.jsonl"
            sup_rows = load_jsonl(sup_path) if sup_path.exists() else []

            result = train_and_eval_randinit(
                arm_name=arm, train_rows=sup_rows,
                eval_suites_dict=eval_suites_dict, tokenizer=tokenizer,
                config_path=args.model_path, device=device, seed=seed,
                epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                max_len=args.max_len,
            )
            all_results.append(result)
            hh = result.get("heldheld_unseen_edge_closure", {}).get("accuracy")
            mx = result.get("mixed_held_seen_orientation", {}).get("accuracy")
            sc = result.get("paired_state_conservation", {}).get("acc_changed")
            print(f"  train_acc={result.get('train_acc_last')}, hh={hh}, mixed={mx}, state_chg={sc}", flush=True)

    rp = out / "randinit_pilot_results.json"
    with open(rp, "w") as f: json.dump(all_results, f, indent=2, ensure_ascii=False)

    sp = out / "randinit_pilot_summary.md"
    lines = ["# Step277c: random-init symmetry pilot", ""]
    lines.append("| arm | seed | train_acc | hh_closure | mixed_orient | state_chg |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in all_results:
        def g(s, k="accuracy"):
            v = r.get(s, {}).get(k); return f"{v:.3f}" if v is not None else "nan"
        ta = r.get("train_acc_last")
        lines.append(f"| {r['arm']} | {r['seed']} | {f'{ta:.3f}' if ta else 'nan'} | {g('heldheld_unseen_edge_closure')} | {g('mixed_held_seen_orientation')} | {g('paired_state_conservation', 'acc_changed')} |")

    lines.extend(["", "## Cross-seed", ""])
    lines.append("| arm | mean_hh | mean_mixed | std_mixed | mean_state_chg |")
    lines.append("|---|---:|---:|---:|---:|")
    for arm in focus_arms:
        ar = [r for r in all_results if r["arm"] == arm]
        hhs = [r.get("heldheld_unseen_edge_closure", {}).get("accuracy", 0.5) for r in ar]
        mxs = [r.get("mixed_held_seen_orientation", {}).get("accuracy", 0.5) for r in ar]
        scs = [r.get("paired_state_conservation", {}).get("acc_changed", 0.5) for r in ar]
        lines.append(f"| {arm} | {np.mean(hhs):.3f} | {np.mean(mxs):.3f} | {np.std(mxs):.3f} | {np.mean(scs):.3f} |")
    lines.append("")
    with open(sp, "w") as f: f.write("\n".join(lines))

    print(json.dumps({"status": "STEP277C_RANDINIT_PILOT_COMPLETE", "results": str(rp), "summary": str(sp)}, indent=2), flush=True)

if __name__ == "__main__":
    main()
