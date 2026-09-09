#!/usr/bin/env python3
"""research: balanced paired rerun of the natural recombination binding factorial.

The research/073 factorial used 4,998 training rows: 1,666 paired
UNCHANGED_DISTRACTOR_USE A rows (source_state answer), 1,666 paired B rows
(new_state answer), and 1,666 UPDATED_USE single rows (new_state answer).  That
2:1 new-state answer mix makes a global update preference a cheap solution.

This rerun removes the single rows and trains only on complete A/B pairs from
UNCHANGED_DISTRACTOR_USE.  Each optimizer update contains both halves of every
selected pair, so within the update the source+update evidence is matched and
query identity is the intended varying coordinate.  The same three arms are run
against the same trusted base, adapter-only trainability, LR, and step budget:

  answer_clean: answer-only credit, visible support;
  uniform_wwm: ordinary WWM over the same paired rows;
  answer_corrupt_update_state: answer-only credit with update-state evidence
      masked in the input.

This is still a specialist adapter acquisition test, not a legal 100M endpoint.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import pathlib
import random
import statistics
import time
from collections import Counter, defaultdict
from typing import Any, Iterable

ROOT = pathlib.Path.cwd()
# Make the imported research loader use a research writable cache before it imports transformers.
os.environ.setdefault("CACHE_BASE", str(ROOT / "experiments/archive/relation_learning/data/binding_factorial_balanced_pairbatch/hf_cache"))
MOD_PATH = ROOT / "experiments/archive/relation_learning/scripts/binding_factorial_train.py"
spec = importlib.util.spec_from_file_location("binding_factorial_train", MOD_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import {MOD_PATH}")
S73 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S73)  # type: ignore[union-attr]

TRAIN_ROWS = ROOT / "experiments/archive/relation_learning/data/recombination_rows/recombination_train.jsonl"
HELDOUT_ROWS = ROOT / "experiments/archive/relation_learning/data/recombination_rows/recombination_heldout.jsonl"
BINDING_PAIRS = ROOT / "experiments/archive/relation_learning/data/recombination_rows/binding_pairs_heldout.jsonl"
STRATA = ROOT / "experiments/archive/relation_learning/data/binding_pair_strata_and_export/heldout_binding_pair_source_presence.csv"
BASE_CHCK = ROOT / "experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_100M"
DEFAULT_OUT = ROOT / "experiments/archive/relation_learning/data/binding_factorial_balanced_pairbatch"
ARMS = ["answer_clean", "uniform_wwm", "answer_corrupt_update_state"]


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: pathlib.Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def sha256_file(path: pathlib.Path) -> str:
    return S73.sha256_file(path)


def parse_bool(x: str) -> bool:
    return str(x).strip().lower() in {"true", "1", "yes"}


def load_strata() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with STRATA.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rr: dict[str, Any] = dict(row)
            for k in ["target_entity_in_source", "updated_entity_in_source", "both_entities_in_source", "same_source_update_context"]:
                rr[k] = parse_bool(str(rr.get(k, "")))
            out[str(rr["pair_id"])] = rr
    return out


def filter_complete_pair_rows(rows: list[dict[str, Any]], max_pairs: int = 0) -> tuple[list[dict[str, Any]], list[str]]:
    by_pair: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        if r.get("packet_type") != "UNCHANGED_DISTRACTOR_USE":
            continue
        if r.get("pair_half") not in {"A", "B"}:
            continue
        by_pair[str(r["pair_id"])][str(r["pair_half"])] = r
    pair_ids = [pid for pid, halves in by_pair.items() if "A" in halves and "B" in halves]
    pair_ids.sort()
    if max_pairs and max_pairs > 0:
        pair_ids = pair_ids[:max_pairs]
    out: list[dict[str, Any]] = []
    for pid in pair_ids:
        # fixed within-pair order; shuffling operates on pair_ids, not rows.
        out.append(by_pair[pid]["A"])
        out.append(by_pair[pid]["B"])
    return out, pair_ids


def pair_token_groups(train_toks: list[dict[str, Any]], pair_ids: list[str]) -> list[list[dict[str, Any]]]:
    by_pair: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for t in train_toks:
        by_pair[str(t["pair_id"])][str(t.get("pair_half", ""))] = t
    groups: list[list[dict[str, Any]]] = []
    for pid in pair_ids:
        d = by_pair.get(pid, {})
        if "A" in d and "B" in d:
            groups.append([d["A"], d["B"]])
    return groups


def dataset_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_rows": len(rows),
        "answer_kind": dict(Counter(r.get("answer_kind", "") for r in rows)),
        "role": dict(Counter(r.get("role", "") for r in rows)),
        "pair_half": dict(Counter(r.get("pair_half", "") for r in rows)),
        "packet_type": dict(Counter(r.get("packet_type", "") for r in rows)),
    }


def subset_pair_rows(pair_rows: list[dict[str, Any]], strata: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    for p in pair_rows:
        p["_strata"] = strata.get(str(p.get("pair_id")), {})
    return {
        "all_pairs": pair_rows,
        "updated_entity_in_source_119": [p for p in pair_rows if p["_strata"].get("updated_entity_in_source") is True],
        "updated_entity_not_in_source_81": [p for p in pair_rows if p["_strata"].get("updated_entity_in_source") is False],
        "both_entities_in_source_115": [p for p in pair_rows if p["_strata"].get("both_entities_in_source") is True],
        "not_both_entities_in_source_85": [p for p in pair_rows if p["_strata"].get("both_entities_in_source") is False],
    }


def avg(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return float(statistics.mean(vals)) if vals else float("nan")


def summarize_pair_subset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if not rows:
        return {"n": 0}
    a = int(sum(int(p.get("a_correct", 0)) for p in rows))
    b = int(sum(int(p.get("b_correct", 0)) for p in rows))
    j = int(sum(int(p.get("joint_correct", 0)) for p in rows))
    expected = a * b / n
    return {
        "n": n,
        "a_correct": a,
        "b_correct": b,
        "joint_correct": j,
        "a_acc": a / n,
        "b_acc": b / n,
        "joint_acc": j / n,
        "expected_joint_independent_count": expected,
        "expected_joint_independent_acc": expected / n,
        "gating_count": j - expected,
        "gating_frac": j / n - (a / n) * (b / n),
        "mean_a_margin": avg(p.get("a_correct_margin", float("nan")) for p in rows),
        "mean_b_margin": avg(p.get("b_correct_margin", float("nan")) for p in rows),
        "mean_joint_min_margin": avg(p.get("joint_min_margin", float("nan")) for p in rows),
    }


def add_gating(ev: dict[str, Any], strata: dict[str, dict[str, Any]]) -> dict[str, Any]:
    subsets = subset_pair_rows([dict(p) for p in ev.get("pair_margins", [])], strata)
    return {name: summarize_pair_subset(rows) for name, rows in subsets.items()}


def strip_eval(ev: dict[str, Any], strata: dict[str, dict[str, Any]]) -> dict[str, Any]:
    d = {k: v for k, v in ev.items() if k not in {"row_margins", "pair_margins"}}
    d["gating"] = add_gating(ev, strata)
    return d


def run_one_arm(args: argparse.Namespace, arm: str) -> dict[str, Any]:
    out_dir = pathlib.Path(args.out_root) / arm
    out_dir.mkdir(parents=True, exist_ok=True)
    all_train = read_jsonl(TRAIN_ROWS)
    all_held = read_jsonl(HELDOUT_ROWS)
    all_pairs = read_jsonl(BINDING_PAIRS)
    train_rows, train_pair_ids = filter_complete_pair_rows(all_train, args.max_train_pairs)
    held_rows, held_pair_ids = filter_complete_pair_rows(all_held, args.max_heldout_pairs)
    keep = set(held_pair_ids)
    binding_pairs = [bp for bp in all_pairs if bp.get("pair_id") in keep]
    strata = load_strata()
    plan = {
        "status": "BALANCED_PAIRBATCH_BINDING_FACTORIAL_PLAN",
        "created_utc": now(),
        "arm": arm,
        "base_model": rel(BASE_CHCK),
        "train_rows_source": rel(TRAIN_ROWS),
        "heldout_rows_source": rel(HELDOUT_ROWS),
        "binding_pairs_source": rel(BINDING_PAIRS),
        "source_sha256": {"train": sha256_file(TRAIN_ROWS), "heldout": sha256_file(HELDOUT_ROWS), "binding_pairs": sha256_file(BINDING_PAIRS)},
        "selection": "only complete UNCHANGED_DISTRACTOR_USE A/B rows; UPDATED_USE single rows dropped",
        "n_train_rows_selected": len(train_rows),
        "n_train_pairs_selected": len(train_pair_ids),
        "n_heldout_rows_selected": len(held_rows),
        "n_heldout_pairs_selected": len(held_pair_ids),
        "train_counts": dataset_counts(train_rows),
        "heldout_counts": dataset_counts(held_rows),
        "epochs": args.epochs,
        "pair_batch_size": args.pair_batch_size,
        "row_batch_size": args.pair_batch_size * 2,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "eval_every": args.eval_every,
        "seed": args.seed,
        "device_requested": args.device,
        "max_length": args.max_length,
        "wwm_prob": args.wwm_prob,
        "out_dir": rel(out_dir),
        "paired_update_form": "each optimizer update is built from complete A/B pair groups so every selected pair contributes both source-retain and update-select examples in the same batch",
    }
    write_json(out_dir / "plan.json", plan)
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if args.plan_only:
        return {"status": "PLAN_ONLY", "arm": arm, "out_dir": rel(out_dir), "plan": plan}

    import torch
    device = torch.device(args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model, tokenizer, identity = S73.trusted_load_model(BASE_CHCK, device)
    if int(identity.get("adapter_params_loaded", 0)) <= 0:
        raise RuntimeError(f"Adapter trusted load failed: {identity}")
    write_json(out_dir / "model_identity.json", identity)
    freeze_stats = S73.freeze_base_train_adapter(model)
    write_json(out_dir / "freeze_stats.json", freeze_stats)

    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    train_toks = [x for x in (S73.tokenize_row(tokenizer, r, args.max_length) for r in train_rows) if x is not None]
    held_toks = [x for x in (S73.tokenize_row(tokenizer, r, args.max_length) for r in held_rows) if x is not None]
    tok_audit = S73.tokenization_audit(train_toks, held_toks)
    tok_audit["train_pairs_after_tokenization"] = len(pair_token_groups(train_toks, train_pair_ids))
    tok_audit["heldout_rows_after_tokenization"] = len(held_toks)
    write_json(out_dir / "tokenization_audit.json", tok_audit)

    pair_groups = pair_token_groups(train_toks, train_pair_ids)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed + 1777)

    base_eval = S73.score_margin_eval(model, tokenizer, held_rows, binding_pairs, device, args.max_length, args.eval_batch_size, 0)
    write_json(out_dir / "eval_epoch_000.json", base_eval)
    trajectory: list[dict[str, Any]] = [{"epoch": 0, "updates_seen": 0, "eval": strip_eval(base_eval, strata)}]
    print(json.dumps({"arm": arm, "epoch": 0, "eval": trajectory[-1]["eval"]}, indent=2, ensure_ascii=False), flush=True)
    if args.eval_only or args.epochs == 0:
        summary = {
            "status": "BALANCED_PAIRBATCH_BINDING_EVAL_ONLY_DONE",
            "created_utc": now(),
            "arm": arm,
            "model_identity": identity,
            "freeze_stats": freeze_stats,
            "tokenization_audit": tok_audit,
            "plan": plan,
            "trajectory": trajectory,
        }
        write_json(out_dir / "summary.json", summary)
        return summary

    updates_seen = 0
    epoch_summaries: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(pair_groups)
        losses: list[float] = []
        labeled_positions = support_positions = answer_positions = wwm_positions = 0
        t0 = time.time()
        for start in range(0, len(pair_groups), args.pair_batch_size):
            groups = pair_groups[start:start + args.pair_batch_size]
            batch = [row for pair in groups for row in pair]
            batch_t = S73.collate(batch, pad_id)
            batch_t = {k: v.to(device) for k, v in batch_t.items()}
            st = S73.batch_train_step(model, opt, batch_t, tokenizer, arm, gen, args.wwm_prob)
            if math.isfinite(float(st.get("loss", float("nan")))):
                losses.append(float(st["loss"]))
            labeled_positions += int(st.get("labeled_positions", 0))
            support_positions += int(st.get("support_mask_positions", 0))
            answer_positions += int(st.get("answer_label_positions", 0))
            wwm_positions += int(st.get("masked_label_positions", 0))
            updates_seen += 1
        ep = {
            "epoch": epoch,
            "updates_seen": updates_seen,
            "mean_train_loss": float(statistics.mean(losses)) if losses else float("nan"),
            "n_train_batches": len(losses),
            "labeled_positions": labeled_positions,
            "answer_label_positions": answer_positions,
            "support_mask_positions": support_positions,
            "wwm_label_positions": wwm_positions,
            "elapsed_sec": round(time.time() - t0, 2),
        }
        epoch_summaries.append(ep)
        print(json.dumps({"arm": arm, "epoch": epoch, "train": ep}, indent=2), flush=True)
        if (epoch % args.eval_every == 0) or (epoch == args.epochs):
            ev = S73.score_margin_eval(model, tokenizer, held_rows, binding_pairs, device, args.max_length, args.eval_batch_size, 0)
            write_json(out_dir / f"eval_epoch_{epoch:03d}.json", ev)
            trajectory.append({"epoch": epoch, "updates_seen": updates_seen, "train": ep, "eval": strip_eval(ev, strata)})
            print(json.dumps({"arm": arm, "epoch": epoch, "eval": trajectory[-1]["eval"]}, indent=2, ensure_ascii=False), flush=True)
    ckpt = out_dir / "checkpoint"
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt), safe_serialization=True)
    tokenizer.save_pretrained(str(ckpt))
    summary = {
        "status": "BALANCED_PAIRBATCH_BINDING_TRAIN_DONE",
        "created_utc": now(),
        "arm": arm,
        "model_identity": identity,
        "freeze_stats": freeze_stats,
        "tokenization_audit": tok_audit,
        "plan": plan,
        "n_train_tokenized": len(train_toks),
        "n_heldout_tokenized": len(held_toks),
        "n_train_pair_groups": len(pair_groups),
        "n_binding_pairs": len(binding_pairs),
        "updates_seen": updates_seen,
        "epoch_summaries": epoch_summaries,
        "trajectory": trajectory,
        "checkpoint_dir": rel(ckpt),
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def summarize_all(out_root: pathlib.Path, arms: list[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    contrasts: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for arm in arms:
        p = out_root / arm / "summary.json"
        if not p.exists():
            continue
        s = json.loads(p.read_text(encoding="utf-8"))
        summaries[arm] = {k: s.get(k) for k in ["status", "created_utc", "updates_seen", "n_train_tokenized", "n_heldout_tokenized", "n_train_pair_groups", "n_binding_pairs"]}
        for tr in s.get("trajectory", []):
            epoch = int(tr["epoch"])
            g = tr["eval"].get("gating", {})
            for subset, vals in g.items():
                row = {"arm": arm, "epoch": epoch, "subset": subset}
                row.update({k: vals.get(k) for k in ["n", "a_correct", "b_correct", "joint_correct", "a_acc", "b_acc", "joint_acc", "expected_joint_independent_count", "gating_count", "gating_frac", "mean_a_margin", "mean_b_margin", "mean_joint_min_margin"]})
                rows.append(row)
    # epoch-20 contrasts for key subset.
    def get(arm: str, subset: str) -> dict[str, Any]:
        for r in rows:
            if r["arm"] == arm and r["epoch"] == 20 and r["subset"] == subset:
                return r
        return {}
    for subset in ["all_pairs", "updated_entity_in_source_119", "both_entities_in_source_115"]:
        vals = {arm: get(arm, subset) for arm in arms}
        for a1, a2 in [("answer_clean", "uniform_wwm"), ("answer_clean", "answer_corrupt_update_state"), ("answer_corrupt_update_state", "uniform_wwm")]:
            if not vals.get(a1) or not vals.get(a2):
                continue
            contrasts.append({
                "epoch": 20,
                "subset": subset,
                "contrast": f"{a1}-minus-{a2}",
                "joint_correct_delta": vals[a1].get("joint_correct", 0) - vals[a2].get("joint_correct", 0),
                "a_correct_delta": vals[a1].get("a_correct", 0) - vals[a2].get("a_correct", 0),
                "b_correct_delta": vals[a1].get("b_correct", 0) - vals[a2].get("b_correct", 0),
                "gating_count_delta": vals[a1].get("gating_count", float("nan")) - vals[a2].get("gating_count", float("nan")),
                "gating_frac_delta": vals[a1].get("gating_frac", float("nan")) - vals[a2].get("gating_frac", float("nan")),
            })
    write_csv(out_root / "trajectory_gating_flat.csv", rows)
    write_csv(out_root / "epoch20_contrasts.csv", contrasts)
    result = {
        "status": "BALANCED_PAIRBATCH_BINDING_SUMMARY",
        "created_utc": now(),
        "out_root": rel(out_root),
        "arms": summaries,
        "flat_csv": rel(out_root / "trajectory_gating_flat.csv"),
        "contrast_csv": rel(out_root / "epoch20_contrasts.csv"),
    }
    write_json(out_root / "summary.json", result)
    # Markdown table if all arms finished.
    lines = ["# research balanced paired binding factorial\n\n"]
    lines.append("This rerun drops UPDATED_USE single rows and trains only complete UNCHANGED_DISTRACTOR_USE A/B pairs, with both halves of each pair in the same optimizer update.\n\n")
    lines.append("## Epoch 20 all-pair gating\n\n")
    lines.append("| arm | joint | A source-retain | B update-select | expected joint | gating count | gating frac | mean A margin | mean B margin |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for arm in arms:
        v = get(arm, "all_pairs")
        if v:
            lines.append(f"| {arm} | {int(v['joint_correct'])}/{int(v['n'])} | {int(v['a_correct'])}/{int(v['n'])} | {int(v['b_correct'])}/{int(v['n'])} | {float(v['expected_joint_independent_count']):.2f} | {float(v['gating_count']):.2f} | {float(v['gating_frac']):.3f} | {float(v['mean_a_margin']):.3f} | {float(v['mean_b_margin']):.3f} |\n")
    lines.append("\n## Epoch 20 updated-entity-in-source subset\n\n")
    lines.append("| arm | joint | A source-retain | B update-select | expected joint | gating count | gating frac | mean A margin | mean B margin |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for arm in arms:
        v = get(arm, "updated_entity_in_source_119")
        if v:
            lines.append(f"| {arm} | {int(v['joint_correct'])}/{int(v['n'])} | {int(v['a_correct'])}/{int(v['n'])} | {int(v['b_correct'])}/{int(v['n'])} | {float(v['expected_joint_independent_count']):.2f} | {float(v['gating_count']):.2f} | {float(v['gating_frac']):.3f} | {float(v['mean_a_margin']):.3f} | {float(v['mean_b_margin']):.3f} |\n")
    (out_root / "summary.md").write_text("".join(lines), encoding="utf-8")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=ARMS + ["all"], required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--pair-batch-size", type=int, default=8)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--seed", type=int, default=77077)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--wwm-prob", type=float, default=0.15)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--max-train-pairs", type=int, default=0)
    ap.add_argument("--max-heldout-pairs", type=int, default=0)
    ap.add_argument("--out-root", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    arms = ARMS if args.arm == "all" else [args.arm]
    done = []
    for i, arm in enumerate(arms):
        # Same model init and same nominal seed per arm; offset only the stochastic masks/order by arm index
        # through args.seed inside the imported training generator would not be fair, so keep seed fixed.
        summary = run_one_arm(args, arm)
        done.append({"arm": arm, "status": summary.get("status"), "out": rel(args.out_root / arm)})
    final = summarize_all(args.out_root, arms)
    final["completed"] = done
    write_json(args.out_root / "summary.json", final)
    print(json.dumps({"status": final["status"], "out_root": rel(args.out_root), "summary": rel(args.out_root / "summary.json"), "md": rel(args.out_root / "summary.md"), "completed": done}, indent=2), flush=True)


if __name__ == "__main__":
    main()
