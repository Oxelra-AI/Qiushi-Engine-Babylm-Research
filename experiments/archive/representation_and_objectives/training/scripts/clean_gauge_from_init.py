#!/usr/bin/env python3
"""research: clean re-establishment of the supplied-harness gauge effect.

This is a minimal contamination check after research exposed sequential reuse in the
raw-name script.  It imports the research harness but does not use research's main
loop.  For each condition/sign cell it explicitly deep-copies an untouched
condition-specific initialization, verifies the init hash before training, and
checks that the stored initialization remains unchanged after all cells.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

PROJECT = Path("experiments/archive/representation_and_objectives")
PATH = PROJECT / "training/scripts/causal_gauge_probe.py"
DEFAULT_OUT = PROJECT / "data/clean_gauge_from_init"
DEFAULT_DATA_ROOT = PROJECT / "data/information_budget_substrate/replace_k16_spread"


def load_step290():
    spec = importlib.util.spec_from_file_location("causal_gauge_probe", PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {PATH}")
    mod = importlib.util.module_from_spec(spec)
    # dataclasses resolves cls.__module__ through sys.modules at decoration time.
    # Register under the exact spec name before executing the imported file.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def tensor_hash(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode("utf-8")); h.update(b"\0")
        arr = tensor.detach().cpu().contiguous().numpy()
        h.update(str(arr.shape).encode("ascii")); h.update(str(arr.dtype).encode("ascii")); h.update(arr.tobytes())
    return h.hexdigest()


def project_rel(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except Exception:
        return str(path)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--models", nargs="+", default=["shared_trunk", "untied"])
    ap.add_argument("--arm", default="aligned_state_bridge")
    ap.add_argument("--bridge-signs", nargs="+", type=int, default=[1, -1])
    ap.add_argument("--seeds", nargs="+", type=int, default=[29000])
    ap.add_argument("--epochs", type=int, default=220)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--emb-dim", type=int, default=48)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--cmp-weight", type=float, default=1.0)
    ap.add_argument("--state-weight", type=float, default=1.0)
    ap.add_argument("--static-weight", type=float, default=1.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--print-every", type=int, default=55)
    args = ap.parse_args()

    mod = load_step290()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

    results: List[Dict[str, Any]] = []
    init_audits: List[Dict[str, Any]] = []

    for seed in args.seeds:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        train_states, train_comps, eval_states, eval_comps, parse_errors, counts = mod.load_dataset(
            args.data_root, args.arm, False, False)
        vocab = mod.Vocab(); mod.collect_vocab(vocab, train_states, train_comps)
        paired = mod.create_paired_models(len(vocab.itos), args.models, seed, args.emb_dim, args.hidden)
        init_hash = {cond: tensor_hash(model) for cond, model in paired.items()}

        for cond in args.models:
            for bs in args.bridge_signs:
                clone = copy.deepcopy(paired[cond])
                clone_hash = tensor_hash(clone)
                if clone_hash != init_hash[cond]:
                    raise RuntimeError(f"Init clone mismatch for {cond} seed {seed} bs {bs}")
                print(f"=== clean {cond}/bs={bs:+d}/seed={seed} init={clone_hash[:12]} ===", flush=True)
                train_info = mod.train_one(clone, vocab, train_states, train_comps, device, bs,
                                           args.epochs, args.lr, args.weight_decay, seed,
                                           args.cmp_weight, args.state_weight, args.static_weight,
                                           print_every=args.print_every)
                with torch.no_grad():
                    final_train = mod.quick_train_metrics(clone, vocab, train_states, train_comps, device, bs)
                train_state_rows, train_comp_rows = mod.eval_predictions(clone, vocab, train_states, train_comps, device, cond, args.arm, seed, bs)
                eval_state_rows, eval_comp_rows = mod.eval_predictions(clone, vocab, eval_states, eval_comps, device, cond, args.arm, seed, bs)
                for r in train_state_rows + train_comp_rows:
                    r["prediction_split"] = "train"
                for r in eval_state_rows + eval_comp_rows:
                    r["prediction_split"] = "eval"
                choices = mod.state_choice_records(eval_state_rows)
                boths = mod.pair_both_records(choices)
                central = mod.compute_central(choices, boths, eval_comp_rows, cond, args.arm, seed, bs)
                cell_tag = f"bs{'+' if bs == 1 else '-'}1_cleaninit"
                rec = {
                    "condition": cond, "arm": args.arm, "seed": seed, "bridge_sign": bs,
                    "cell_tag": cell_tag, "epochs": args.epochs, "lr": args.lr,
                    "weight_decay": args.weight_decay, "emb_dim": args.emb_dim,
                    "hidden": args.hidden, "dropout": 0.0,
                    "counts": counts, "parse_errors_sample": parse_errors[:10],
                    "vocab_size": len(vocab.itos), "init_hash": clone_hash,
                    "train_info": train_info, "final_train_metrics": final_train,
                    "central_eval": central,
                    "no_official_evaluation_upload_or_leaderboard": True,
                }
                results.append(rec)
                run_dir = args.out / f"{cond}_{cell_tag}_seed{seed}"
                write_json(run_dir / "result.json", rec)
                write_jsonl(run_dir / "eval_state_predictions.jsonl", eval_state_rows)
                write_jsonl(run_dir / "eval_comparison_predictions.jsonl", eval_comp_rows)
                write_jsonl(run_dir / "train_state_predictions.jsonl", train_state_rows)
                write_jsonl(run_dir / "train_comparison_predictions.jsonl", train_comp_rows)
                ce = central
                print(json.dumps({
                    "condition": cond, "bridge_sign": bs, "seed": seed,
                    "train_state_acc": final_train.get("train_state_acc"),
                    "train_cmp_acc": final_train.get("train_cmp_acc"),
                    "direct_same": ce.get("direct_same"),
                    "graph_same": ce.get("graph_same"),
                    "pair_both_graph_same": ce.get("pair_both_graph_same"),
                    "unchanged": ce.get("unchanged"),
                    "mixed_acc": ce.get("mixed_held_seen_orientation_acc"),
                    "mixed_margin": ce.get("mixed_held_seen_orientation_signed_margin"),
                    "hh_closure": ce.get("heldheld_unseen_edge_closure_acc"),
                    "graph_same_margin": ce.get("graph_same_margin"),
                    "graph_mean_de": ce.get("graph_mean_de"),
                    "init_hash_prefix": clone_hash[:16],
                }, sort_keys=True), flush=True)

        for cond, model in paired.items():
            after = tensor_hash(model)
            init_audits.append({
                "seed": seed, "condition": cond, "init_hash": init_hash[cond],
                "after_all_runs_hash": after, "untouched_after_all_runs": after == init_hash[cond],
            })
            if after != init_hash[cond]:
                raise RuntimeError(f"Stored initialization mutated for {cond} seed {seed}")

    summary = mod.write_summary_and_json(args.out, results, args)
    audit = {"init_audits": init_audits, "summary": summary, "n_results": len(results)}
    write_json(args.out / "clean_init_audit.json", audit)

    md = ["# research clean-init supplied-harness gauge recheck\n\n",
          "Each bridge-sign cell is trained from a fresh deep-copy of an untouched condition-specific initialization.\n\n",
          "## Initialization audit\n\n",
          "| seed | condition | init hash prefix | untouched after all runs |\n",
          "|---:|---|---|---:|\n"]
    for a in init_audits:
        md.append(f"| {a['seed']} | {a['condition']} | {a['init_hash'][:16]} | {a['untouched_after_all_runs']} |\n")
    md.append("\n## Central readout\n\n")
    cols = ["condition", "bridge_sign", "seed", "train_state_acc", "train_cmp_acc", "direct_same", "graph_same", "pair_both_graph_same", "unchanged", "mixed_acc", "mixed_margin", "hh_closure", "graph_same_margin", "graph_mean_de"]
    md.append("| " + " | ".join(cols) + " |\n")
    md.append("|" + "|".join(["---"] * len(cols)) + "|\n")
    for row in summary["central_by_run"]:
        vals = []
        for c in cols:
            v = row.get(c)
            vals.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        md.append("| " + " | ".join(vals) + " |\n")
    md.append("\n## Bridge-sign pairs\n\n")
    for k, v in sorted(summary.get("bridge_sign_paired", {}).items()):
        md.append(f"### {k}\n")
        for kk, vv in sorted(v.items()):
            md.append(f"- {kk}: {vv}\n")
        md.append("\n")
    note_path = args.out / "clean_gauge_from_init_summary.md"
    note_path.write_text("".join(md), encoding="utf-8")

    print(json.dumps({
        "status": "CLEAN_GAUGE_FROM_INIT_COMPLETE",
        "device": str(device), "n_results": len(results),
        "summary": project_rel(note_path),
        "audit": project_rel(args.out / "clean_init_audit.json"),
        "no_official_evaluation_upload_or_leaderboard": True,
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
