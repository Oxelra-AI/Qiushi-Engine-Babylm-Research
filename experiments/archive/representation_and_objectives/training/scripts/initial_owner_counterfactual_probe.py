#!/usr/bin/env python3
"""research: initial-owner counterfactual probe for the matched state effect.

Scientific purpose
------------------
The matched state-format probe showed that aligned and inverted held-state
training both improve true-labeled state readouts while mixed held-seen relation
orientation remains chance. A possible explanation is not role-coordinate
learning at all: in the original research state rows, the changed object always
starts with the non-final participant, so a model can answer by a weaker rule:
"the event moves the changed object to the other participant."

This script trains the same small matched arms, then evaluates counterfactual
state rows where the changed object initially belongs either to the true final
slot or to the opposite slot. True labels always ask for the true final owner
under the research relation assignment. Collapse on the true-final-initial rows
is evidence for a transfer-away-from-initial-owner state heuristic, not a reusable
role coordinate.

No official BabyLM evaluation, upload, or leaderboard interaction occurs.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import copy
import gc
import importlib.util
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the tested research runner's model/training/evaluation utilities.
def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def state_rows_fixed_initial(mod: Any, row_id: str, rel_key: str, a: str, b: str,
                             changed_obj: str, static_obj: str, initial_slot: int,
                             split: str, suite: str, voice: int, static_slot: int,
                             assignment_for_labels: Dict[str, int]) -> List[Dict[str, Any]]:
    """State query rows where initial changed-object owner is controlled directly."""
    final_owner = mod.final_owner(rel_key, a, b, assignment_for_labels)
    initial_owner = mod.owner_from_slot(initial_slot, a, b)
    static = (a, b)[static_slot % 2]
    ev, surface_order, tmpl, vv = mod.event(rel_key, a, b, changed_obj, voice)
    premise = f"At first, {initial_owner} had the {changed_obj}, and {static} had the {static_obj}. The event was this: {ev}"
    rows: List[Dict[str, Any]] = []
    for query_kind, obj, true_owner, dep in [
        ("changed", changed_obj, final_owner, "held_seen_breaks_symmetry" if rel_key in mod.HELD else "seen_seen_fixed"),
        ("unchanged", static_obj, static, "unaffected_fact_should_be_preserved"),
    ]:
        for cand in [a, b]:
            rows.append({
                "row_id": f"{row_id}_{query_kind}_{0 if cand == a else 1}",
                "pair_id": row_id,
                "task": "state_query",
                "split": split,
                "suite": suite,
                "relation": rel_key,
                "component": mod.REL[rel_key].component,
                "arg_order": [a, b],
                "surface_order": surface_order,
                "template": tmpl,
                "voice": vv,
                "static_slot": static_slot,
                "initial_owner_slot": int(initial_slot),
                "initial_owner": initial_owner,
                "final_owner_slot_true": int(mod.TRUE_ASSIGNMENT[rel_key]),
                "final_owner_slot_inverted": int(mod.INVERTED_ASSIGNMENT[rel_key]),
                "initial_equals_true_final": bool(initial_slot == mod.TRUE_ASSIGNMENT[rel_key]),
                "changed_object": changed_obj,
                "static_object": static_obj,
                "query_kind": query_kind,
                "candidate": cand,
                "candidate_slot": 0 if cand == a else 1,
                "correct_slot": 0 if true_owner == a else 1,
                "premise": premise,
                "hypothesis": f"After the event, {cand} had the {obj}.",
                "label": bool(cand == true_owner),
                "orientation_dependency": dep,
                "global_swap_changes_label": dep == "held_seen_breaks_symmetry",
                "cause_relation": rel_key,
                "cause_event": ev,
                "static_owner": static,
                "supervised_changed_owner": final_owner,
                "inverted_bridge_label": assignment_for_labels != mod.TRUE_ASSIGNMENT,
            })
    return rows


def build_counterfactual_suites(mod: Any, n_per_rel: int = 8) -> Dict[str, List[Dict[str, Any]]]:
    suites: Dict[str, List[Dict[str, Any]]] = {
        "cf_initial_true_final": [],
        "cf_initial_opposite_true_final": [],
        "cf_initial_slot0": [],
        "cf_initial_slot1": [],
    }
    k = 0
    for rel in mod.HELD_KEYS:
        true_slot = int(mod.TRUE_ASSIGNMENT[rel])
        for j in range(n_per_rel):
            a, b = mod.choose_pair(mod.EVAL_NAMES, k + 7000)
            changed = mod.EVAL_CHANGED[(k + 2) % len(mod.EVAL_CHANGED)]
            static = mod.EVAL_STATIC[(k * 3 + 4) % len(mod.EVAL_STATIC)]
            for voice in [0, 1]:
                for static_slot in [0, 1]:
                    base = f"cf_{rel}_{j:04d}_v{voice}_st{static_slot}"
                    for suite, init_slot in [
                        ("cf_initial_true_final", true_slot),
                        ("cf_initial_opposite_true_final", 1 - true_slot),
                        ("cf_initial_slot0", 0),
                        ("cf_initial_slot1", 1),
                    ]:
                        rows = state_rows_fixed_initial(
                            mod, f"{suite}_{base}", rel, a, b, changed, static,
                            init_slot, "eval", suite, voice, static_slot, mod.TRUE_ASSIGNMENT,
                        )
                        suites[suite].extend(rows)
            k += 1
    return suites


def fmt(x: Any, digits: int = 3) -> str:
    if x is None:
        return "nan"
    try:
        xf = float(x)
    except Exception:
        return str(x)
    if math.isnan(xf):
        return "nan"
    return f"{xf:.{digits}f}"


def gp(obj: Dict[str, Any], path: Sequence[str]) -> Any:
    cur: Any = obj
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def aggregate(all_seed_summaries: Dict[str, Dict[int, Dict[str, Any]]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for arm, seed_map in all_seed_summaries.items():
        paths: Dict[Tuple[str, ...], List[float]] = defaultdict(list)
        def collect(prefix: Tuple[str, ...], obj: Any) -> None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    collect(prefix + (str(k),), v)
            elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
                if not math.isnan(float(obj)):
                    paths[prefix].append(float(obj))
        for s in seed_map.values():
            collect((), s)
        arm_out: Dict[str, Any] = {}
        for path, vals in paths.items():
            cur = arm_out
            for p in path[:-1]:
                cur = cur.setdefault(p, {})
            cur[path[-1] + "_mean"] = float(np.mean(vals))
            cur[path[-1] + "_std"] = float(np.std(vals))
            cur[path[-1] + "_n"] = len(vals)
        out[arm] = arm_out
    return out


def write_summary(path: Path, args: argparse.Namespace, agg: Dict[str, Any], construction: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# research initial-owner counterfactual probe")
    lines.append("")
    lines.append("The original research state rows always initialize the changed object with the non-final participant, so coherent state supervision can be solved by a weaker transfer-away rule. This probe evaluates the same trained matched arms on rows where the initial owner is controlled independently of the true final role.")
    lines.append("")
    lines.append(f"- seeds: `{args.seeds}`")
    lines.append(f"- epochs: `{args.epochs}`")
    lines.append(f"- arms: `{args.arms}`")
    lines.append("- no official evaluation, upload, or leaderboard interaction")
    lines.append("")
    lines.append("## Counterfactual readouts")
    lines.append("")
    lines.append("`cf_initial_true_final` means the changed object initially belongs to the same role slot that is the true final owner. A transfer-away-from-initial-owner rule should fail on the changed-object half of this suite. `cf_initial_opposite_true_final` is the original natural-transfer configuration: the object begins with the opposite participant and should move to the true final owner.")
    lines.append("")
    lines.append("| arm | train row-label | paired ordinary true | ordinary changed | ordinary unchanged | cf true-final choice | cf true-final changed | cf true-final unchanged | cf opposite choice | cf opposite changed | cf opposite unchanged | mixed true stmt |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for arm in args.arms:
        a = agg[arm]
        vals = [
            gp(a, ["train_all", "statement_row", "acc_mean"]),
            gp(a, ["paired_state_conservation", "state_choice_true", "choice_acc_mean"]),
            gp(a, ["paired_state_conservation", "state_choice_true", "by_query_kind", "changed_mean"]),
            gp(a, ["paired_state_conservation", "state_choice_true", "by_query_kind", "unchanged_mean"]),
            gp(a, ["cf_initial_true_final", "state_choice_true", "choice_acc_mean"]),
            gp(a, ["cf_initial_true_final", "state_choice_true", "by_query_kind", "changed_mean"]),
            gp(a, ["cf_initial_true_final", "state_choice_true", "by_query_kind", "unchanged_mean"]),
            gp(a, ["cf_initial_opposite_true_final", "state_choice_true", "choice_acc_mean"]),
            gp(a, ["cf_initial_opposite_true_final", "state_choice_true", "by_query_kind", "changed_mean"]),
            gp(a, ["cf_initial_opposite_true_final", "state_choice_true", "by_query_kind", "unchanged_mean"]),
            gp(a, ["mixed_held_seen_orientation", "statement_true", "acc_mean"]),
        ]
        lines.append(f"| {arm} | " + " | ".join(fmt(v) for v in vals) + " |")
    lines.append("")
    lines.append("## Slot-controlled counterfactuals")
    lines.append("")
    lines.append("| arm | initial slot0 changed | initial slot1 changed | initial true-final changed | initial opposite changed |")
    lines.append("|---|---:|---:|---:|---:|")
    for arm in args.arms:
        a = agg[arm]
        vals = [
            gp(a, ["cf_initial_slot0", "state_choice_true", "by_query_kind", "changed_mean"]),
            gp(a, ["cf_initial_slot1", "state_choice_true", "by_query_kind", "changed_mean"]),
            gp(a, ["cf_initial_true_final", "state_choice_true", "by_query_kind", "changed_mean"]),
            gp(a, ["cf_initial_opposite_true_final", "state_choice_true", "by_query_kind", "changed_mean"]),
        ]
        lines.append(f"| {arm} | " + " | ".join(fmt(v) for v in vals) + " |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("If aligned/inverted state-trained arms score high on the ordinary/opposite-initial rows but near zero on true-final-initial changed rows, the state transfer is a non-initial-owner transition heuristic rather than a role-coordinate representation. If they remain high regardless of initial owner, the evidence supports a stronger final-role coordinate. Mixed relation orientation is reported to keep the state and relation conclusions separate.")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("- aggregate JSON: `counterfactual_aggregate.json`")
    lines.append("- seed summaries: `counterfactual_seed_summaries.json`")
    lines.append("- per-row predictions: `counterfactual_per_row_predictions.jsonl`")
    lines.append("- construction: `counterfactual_construction.json`")
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[28110, 28111, 28112])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--arms", nargs="+", default=["heldheld_only", "aligned_matched", "inverted_matched", "neutral_matched"])
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dry-run-rows", action="store_true")
    args = parser.parse_args()

    workspace = _public_path('experiments/archive/representation_and_objectives')
    outdir = Path(args.outdir) if args.outdir else workspace / "data" / "initial_owner_counterfactual_probe"
    outdir.mkdir(parents=True, exist_ok=True)
    runner = load_module("matched_state_probe_reuse", workspace / "training" / "scripts" / "matched_state_probe.py")
    mod = runner.load_step278_module(workspace)
    substrate = workspace / "data" / "equivariant_symmetry_substrate"
    ckpt = workspace / "training" / "runs" / "qwen_8x480_16k_wwm_to_token_100M_seed43022" / "hf_model" / "chck_80M"

    train_sets_all = runner.build_matched_train_sets(substrate)
    train_sets = {arm: train_sets_all[arm] for arm in args.arms}
    cf_suites = build_counterfactual_suites(mod, n_per_rel=8)
    base_eval_raw = {
        "paired_state_conservation": load_jsonl(substrate / "eval" / "paired_state_conservation.jsonl"),
        "mixed_held_seen_orientation": load_jsonl(substrate / "eval" / "mixed_held_seen_orientation.jsonl"),
    }
    eval_raw = {**base_eval_raw, **cf_suites}
    construction = {
        "train": {arm: {"rows": len(rows), "relation": sum(r.get("task") == "relation_comparison" for r in rows), "state": sum(r.get("task") == "state_query" for r in rows)} for arm, rows in train_sets.items()},
        "eval": {suite: {"rows": len(rows), "state_groups": len({str(r.get('pair_id')) for r in rows if r.get('task') == 'state_query'}), "query_counts": dict(Counter(str(r.get('query_kind')) for r in rows if r.get('task') == 'state_query'))} for suite, rows in eval_raw.items()},
    }
    (outdir / "counterfactual_construction.json").write_text(json.dumps(construction, indent=2, sort_keys=True))
    for suite, rows in cf_suites.items():
        write_jsonl(outdir / f"eval_{suite}.jsonl", rows)
    if args.dry_run_rows:
        print(json.dumps({"status": "ROWS_ONLY", "outdir": str(outdir), "construction": construction}, indent=2), flush=True)
        return

    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}", flush=True)
    print(f"Checkpoint: {ckpt}", flush=True)
    print(f"Output: {outdir}", flush=True)

    from transformers import AutoConfig, AutoTokenizer, DebertaV2Model
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt), local_files_only=True)
    pad_id = tokenizer.pad_token_id or 0
    hidden_size = int(AutoConfig.from_pretrained(str(ckpt), local_files_only=True).hidden_size)
    print(f"Hidden size: {hidden_size}", flush=True)

    train_encoded = {arm: [runner.tokenize_row(r, tokenizer, runner.MAX_LEN) for r in rows] for arm, rows in train_sets.items()}
    eval_encoded = {suite: [runner.tokenize_row(r, tokenizer, runner.MAX_LEN) for r in rows] for suite, rows in eval_raw.items()}
    per_row_path = outdir / "counterfactual_per_row_predictions.jsonl"
    if per_row_path.exists():
        per_row_path.unlink()
    all_seed_summaries: Dict[str, Dict[int, Dict[str, Any]]] = {arm: {} for arm in args.arms}

    for seed in args.seeds:
        set_seed(seed)
        ref_head = nn.Linear(hidden_size, 2)
        head_state = copy.deepcopy(ref_head.state_dict())
        del ref_head
        for arm in args.arms:
            t0 = time.time()
            set_seed(seed)
            base_model = DebertaV2Model.from_pretrained(str(ckpt), local_files_only=True)
            model = runner.ClassificationModel(base_model, hidden_size)
            model.head.load_state_dict(head_state)
            model.to(device)
            train_stats = runner.train_one_arm(model, train_encoded[arm], args.epochs, pad_id, device)
            suite_predictions: Dict[str, Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]] = {}
            pred_train = runner.predict(model, train_encoded[arm], train_sets[arm], "train_all", arm, seed, pad_id, device)
            suite_predictions["train_all"] = (pred_train, train_sets[arm])
            for suite, rows in eval_raw.items():
                preds = runner.predict(model, eval_encoded[suite], rows, suite, arm, seed, pad_id, device)
                suite_predictions[suite] = (preds, rows)
            seed_summary = runner.summarize_by_seed(suite_predictions, mod)
            seed_summary["train_stats_last_epoch"] = train_stats
            seed_summary["elapsed_sec"] = time.time() - t0
            all_seed_summaries[arm][seed] = seed_summary
            with per_row_path.open("a") as f:
                for suite, (preds, raws) in suite_predictions.items():
                    for pr, rr in zip(preds, raws):
                        pr2 = dict(pr)
                        pr2["target_true_assignment"] = runner.label_for_assignment(rr, mod, "true")
                        pr2["target_inverted_assignment"] = runner.label_for_assignment(rr, mod, "inverted")
                        pr2["initial_equals_true_final"] = rr.get("initial_equals_true_final", None)
                        pr2["initial_owner_slot"] = rr.get("initial_owner_slot", None)
                        f.write(json.dumps(pr2, ensure_ascii=False, sort_keys=True) + "\n")
            ordinary = gp(seed_summary, ["paired_state_conservation", "state_choice_true", "by_query_kind", "changed"])
            cf_same = gp(seed_summary, ["cf_initial_true_final", "state_choice_true", "by_query_kind", "changed"])
            cf_opp = gp(seed_summary, ["cf_initial_opposite_true_final", "state_choice_true", "by_query_kind", "changed"])
            mixed = gp(seed_summary, ["mixed_held_seen_orientation", "statement_true", "acc"])
            train_row = gp(seed_summary, ["train_all", "statement_row", "acc"])
            print(f"arm={arm} seed={seed} train={fmt(train_row,4)} mixed={fmt(mixed,4)} ordinary_changed={fmt(ordinary,4)} cf_truefinal_changed={fmt(cf_same,4)} cf_opp_changed={fmt(cf_opp,4)} [{seed_summary['elapsed_sec']:.1f}s]", flush=True)
            model.cpu(); del model, base_model; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        (outdir / "counterfactual_seed_summaries.json").write_text(json.dumps(all_seed_summaries, indent=2, sort_keys=True))

    agg = aggregate(all_seed_summaries)
    (outdir / "counterfactual_seed_summaries.json").write_text(json.dumps(all_seed_summaries, indent=2, sort_keys=True))
    (outdir / "counterfactual_aggregate.json").write_text(json.dumps(agg, indent=2, sort_keys=True))
    summary_md = outdir / "initial_owner_counterfactual_summary.md"
    write_summary(summary_md, args, agg, construction)
    compact = {
        "status": "INITIAL_OWNER_COUNTERFACTUAL_COMPLETE",
        "summary": str(summary_md),
        "aggregate": str(outdir / "counterfactual_aggregate.json"),
        "per_row_predictions": str(per_row_path),
        "core": {},
        "no_official_evaluation_upload_or_leaderboard": True,
    }
    for arm in args.arms:
        a = agg[arm]
        compact["core"][arm] = {
            "mixed_true_stmt": gp(a, ["mixed_held_seen_orientation", "statement_true", "acc_mean"]),
            "ordinary_changed_true": gp(a, ["paired_state_conservation", "state_choice_true", "by_query_kind", "changed_mean"]),
            "ordinary_unchanged_true": gp(a, ["paired_state_conservation", "state_choice_true", "by_query_kind", "unchanged_mean"]),
            "cf_initial_true_final_changed": gp(a, ["cf_initial_true_final", "state_choice_true", "by_query_kind", "changed_mean"]),
            "cf_initial_true_final_unchanged": gp(a, ["cf_initial_true_final", "state_choice_true", "by_query_kind", "unchanged_mean"]),
            "cf_initial_opposite_changed": gp(a, ["cf_initial_opposite_true_final", "state_choice_true", "by_query_kind", "changed_mean"]),
            "cf_initial_opposite_unchanged": gp(a, ["cf_initial_opposite_true_final", "state_choice_true", "by_query_kind", "unchanged_mean"]),
        }
    print(json.dumps(compact, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
