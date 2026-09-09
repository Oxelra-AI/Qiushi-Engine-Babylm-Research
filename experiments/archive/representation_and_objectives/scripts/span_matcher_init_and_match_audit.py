#!/usr/bin/env python3
"""research audit of research SpanMatcher initialization and matcher diagnostic.

The research summaries record an aggregate hash over all conditions in the run. The
learned bs+ run used conditions [shared_trunk, untied] while the learned bs- rerun
used only [shared_trunk], so aggregate hashes are not comparable. This script
reconstructs condition-specific initial hashes.

It also tests whether a fresh untrained CharGRUMatcher already identifies a query
name among event names by self-similarity. If so, research's perfect matching should
be interpreted as a context-invariant equality/matching interface, not as evidence
that sparse relational supervision taught a general name matcher.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import random
import sys
from pathlib import Path

import torch

ROOT = Path("experiments/archive/representation_and_objectives")
SCRIPT = ROOT / "training/scripts/span_matcher_gauge_probe.py"
OUT = ROOT / "data/span_matcher_init_match_audit"

TRAIN_NAMES = ['Ava', 'Felix', 'Iris', 'Jonas', 'Keira', 'Lena', 'Milo', 'Mira', 'Nia', 'Noel', 'Omar', 'Pavel', 'Rina', 'Sara', 'Theo', 'Tomas']
EVAL_NAMES = ['Arun', 'Ben', 'Caleb', 'Eli', 'Hugo', 'June', 'Leah', 'Luca', 'Mateo', 'Maya', 'Nora', 'Simon', 'Tara', 'Vera', 'Yara', 'Zara']
FRESH_ALPHA = ['Alba', 'Borin', 'Cato', 'Darin', 'Elora', 'Farin', 'Galen', 'Hedra', 'Isla', 'Jorin', 'Kato', 'Liora', 'Maren', 'Nolan', 'Orin', 'Pira']
NEAR_NAMES = ['Mira', 'Mara', 'Mila', 'Mina', 'Maya', 'Naya', 'Nora', 'Nola', 'Luca', 'Luna', 'Leah', 'Leia', 'Tara', 'Sara', 'Yara', 'Zara']
COLLIDING_DIGIT_NAMES = ['Xname0', 'Xname1', 'Xname2', 'Xname3']


def import_step294():
    spec = importlib.util.spec_from_file_location("span_matcher_gauge_probe", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def tensor_hash_model(model):
    h = hashlib.sha256()
    for name, p in sorted(model.named_parameters()):
        h.update(name.encode())
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def aggregate_hash(models):
    h = hashlib.sha256()
    for k in sorted(models.keys()):
        for n, p in sorted(models[k].named_parameters()):
            h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def char_ids(mod, name, device='cpu'):
    return torch.tensor(mod.name_to_char_ids(name), dtype=torch.long, device=device)


def eval_matcher(mod, matcher, names, pair_mode='all_pairs'):
    rows = []
    if pair_mode == 'all_pairs':
        pairs = [(a, b) for i, a in enumerate(names) for j, b in enumerate(names) if i != j]
    elif pair_mode == 'adjacent':
        pairs = list(zip(names[0::2], names[1::2]))
    else:
        raise ValueError(pair_mode)
    correct = total = 0
    probs_true = []
    margins = []
    worst = []
    with torch.no_grad():
        for a, b in pairs:
            e_chars = [char_ids(mod, a), char_ids(mod, b)]
            for true_idx, qn in enumerate([a, b]):
                probs = matcher(char_ids(mod, qn), e_chars).detach().cpu().tolist()
                pred = int(max(range(len(probs)), key=lambda i: probs[i])) if probs else -1
                ok = pred == true_idx
                correct += int(ok); total += 1
                pt = probs[true_idx]
                po = probs[1 - true_idx]
                probs_true.append(pt)
                margins.append(pt - po)
                if not ok or pt < 0.55:
                    worst.append({"pair": [a, b], "query": qn, "true_idx": true_idx, "probs": probs, "pred": pred})
    return {
        "n_pairs": len(pairs),
        "n_queries": total,
        "acc": correct / total if total else math.nan,
        "min_true_prob": min(probs_true) if probs_true else math.nan,
        "mean_true_prob": sum(probs_true) / len(probs_true) if probs_true else math.nan,
        "min_margin": min(margins) if margins else math.nan,
        "mean_margin": sum(margins) / len(margins) if margins else math.nan,
        "low_or_wrong_examples": worst[:20],
    }


def char_collision_report(mod, names):
    enc = {}
    for n in names:
        ids = tuple(mod.name_to_char_ids(n))
        enc.setdefault(ids, []).append(n)
    return {"n_names": len(names), "n_unique_char_sequences": len(enc), "collisions": [v for v in enc.values() if len(v) > 1]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mod = import_step294()
    torch.set_grad_enabled(False)
    init_audit = {}
    for oracle in [False, True]:
        for conds in [["shared_trunk"], ["shared_trunk", "untied"], ["untied"], ["tied", "shared_trunk", "untied"]]:
            torch.manual_seed(29400)
            models = mod.create_paired_models(48, conds, 29400, oracle, 48, 64)
            tag = f"{'oracle' if oracle else 'learned'}__{'_'.join(conds)}"
            init_audit[tag] = {
                "aggregate_hash": aggregate_hash(models),
                "condition_hashes": {k: tensor_hash_model(v) for k, v in models.items()},
            }
    shared_hash_comparisons = {
        "learned_shared_hash_same_when_untied_present": init_audit["learned__shared_trunk"]["condition_hashes"]["shared_trunk"] == init_audit["learned__shared_trunk_untied"]["condition_hashes"]["shared_trunk"],
        "oracle_shared_hash_same_when_untied_present": init_audit["oracle__shared_trunk"]["condition_hashes"]["shared_trunk"] == init_audit["oracle__shared_trunk_untied"]["condition_hashes"]["shared_trunk"],
        "aggregate_hash_differs_when_condition_list_differs": init_audit["learned__shared_trunk"]["aggregate_hash"] != init_audit["learned__shared_trunk_untied"]["aggregate_hash"],
    }

    matcher_audit = {}
    for seed in [29400, 29401, 29402, 29500, 1, 2, 3, 4, 5]:
        torch.manual_seed(seed)
        # MatchGatedEncoder initialization matches actual matcher placement better
        # than a bare CharGRUMatcher because embedding/GRU params are drawn first.
        enc = mod.MatchGatedEncoder(48, 48, 64, oracle=False)
        m = enc.matcher
        matcher_audit[str(seed)] = {
            "train_all_pairs": eval_matcher(mod, m, TRAIN_NAMES, "all_pairs"),
            "eval_all_pairs": eval_matcher(mod, m, EVAL_NAMES, "all_pairs"),
            "fresh_alpha_all_pairs": eval_matcher(mod, m, FRESH_ALPHA, "all_pairs"),
            "near_names_all_pairs": eval_matcher(mod, m, NEAR_NAMES, "all_pairs"),
            "colliding_digit_adjacent": eval_matcher(mod, m, COLLIDING_DIGIT_NAMES, "adjacent"),
        }
    collision = {
        "train": char_collision_report(mod, TRAIN_NAMES),
        "eval": char_collision_report(mod, EVAL_NAMES),
        "fresh_alpha": char_collision_report(mod, FRESH_ALPHA),
        "near_names": char_collision_report(mod, NEAR_NAMES),
        "colliding_digit": char_collision_report(mod, COLLIDING_DIGIT_NAMES),
        "fresh_rename_bug_example": "fresh_rename_state_queries uses Xname0... but name_to_char_ids ignores digits, collapsing all to the same 'xname' character sequence if --fresh-rename is used.",
    }
    # Aggregate compact view over seeds
    aggregate = {}
    for dataset in ["train_all_pairs", "eval_all_pairs", "fresh_alpha_all_pairs", "near_names_all_pairs", "colliding_digit_adjacent"]:
        vals = [matcher_audit[str(seed)][dataset]["acc"] for seed in [29400,29401,29402,29500,1,2,3,4,5]]
        mins = [matcher_audit[str(seed)][dataset]["min_true_prob"] for seed in [29400,29401,29402,29500,1,2,3,4,5]]
        aggregate[dataset] = {"acc_values": vals, "mean_acc": sum(vals)/len(vals), "min_acc": min(vals), "min_true_prob_over_seeds": min(mins)}
    report = {
        "init_audit": init_audit,
        "shared_hash_comparisons": shared_hash_comparisons,
        "matcher_untrained_audit": matcher_audit,
        "matcher_untrained_aggregate": aggregate,
        "char_collision_report": collision,
    }
    json_path = OUT / "init_and_match_audit.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    md = []
    md.append("# research SpanMatcher initialization and matching audit\n\n")
    md.append("## Condition-specific initialization\n\n")
    md.append(f"- learned shared_trunk hash same when untied is also in the run: {shared_hash_comparisons['learned_shared_hash_same_when_untied_present']}\n")
    md.append(f"- oracle shared_trunk hash same when untied is also in the run: {shared_hash_comparisons['oracle_shared_hash_same_when_untied_present']}\n")
    md.append(f"- aggregate hash differs when condition list differs: {shared_hash_comparisons['aggregate_hash_differs_when_condition_list_differs']}\n\n")
    md.append("Interpretation: research's displayed aggregate init hashes should not be used as a sign-pair equality check, but the reconstructed shared_trunk initial parameters are identical across the relevant condition-list variants.\n\n")
    md.append("## Untrained matcher identity behavior\n\n")
    md.append("| dataset | mean acc across seeds | min acc | min true prob over seeds |\n")
    md.append("|---|---:|---:|---:|\n")
    for dataset, s in aggregate.items():
        md.append(f"| {dataset} | {s['mean_acc']:.3f} | {s['min_acc']:.3f} | {s['min_true_prob_over_seeds']:.3f} |\n")
    md.append("\nInterpretation: if train/eval/fresh-alpha matching is already perfect before relational training, research should be described as using a context-invariant equality-style matcher that supplies stable candidate coordinates from character strings. The relational experiment verifies that this interface feeds gauge transport, not that sparse state/comparison labels taught a broad name-matching algorithm.\n\n")
    md.append("## Character collision note\n\n")
    for name, c in collision.items():
        if name.startswith('step'):
            continue
        md.append(f"- {name}: {c['n_unique_char_sequences']}/{c['n_names']} unique char sequences; collisions={c['collisions']}\n")
    md.append(f"\n{collision['fresh_rename_bug_example']}\n")
    md_path = (OUT.parents[4] / 'research/documents/representation_and_objectives/data/span_matcher_init_match_audit/init_and_match_audit.md')
    md_path.write_text("".join(md))
    print(json.dumps({"status": "INIT_MATCH_AUDIT_COMPLETE", "json": str(json_path), "md": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
