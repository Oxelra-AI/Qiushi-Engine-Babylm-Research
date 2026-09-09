#!/usr/bin/env python3
"""Audit raw-string inputs used by research shared-coordinate probe.

Checks vocabulary scope and candidate/name replacement. No model loading.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

PROJECT = Path("experiments/archive/representation_and_objectives")
SCRIPT = PROJECT / "training/scripts/shared_relation_coordinate_probe.py"
DEFAULT_DATA = PROJECT / "data/information_budget_substrate/replace_k16_spread"
DEFAULT_OUT = PROJECT / "data/factorized_probe_input_audit"


def load_probe():
    spec = importlib.util.spec_from_file_location("probe", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    import sys
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def audit_arm(mod: Any, data_root: Path, arm: str) -> Dict[str, Any]:
    train_states, train_comps, eval_states, eval_comps, parse_errors, counts = mod.load_dataset(data_root, arm)
    train_vocab = mod.Vocab(); mod.collect_vocab(train_vocab, train_states, train_comps)
    full_vocab = mod.Vocab(); mod.collect_vocab(full_vocab, train_states + eval_states, train_comps + eval_comps)
    def seqs_for(states, comps):
        out = []
        raw_name_hits = Counter()
        for q in states:
            for cand in q.names:
                if q.is_changed:
                    toks = mod.normalize_event_for_candidate(q.event, cand, q.names)
                else:
                    toks = mod.normalize_static_for_candidate(q.prefix, q.hypothesis_text_by_name[cand], cand, q.names)
                out.append((q.suite, "state_changed" if q.is_changed else "state_unchanged", toks))
                for n in q.names:
                    raw_name_hits[n] += int(n.lower() in toks)
            
        for c in comps:
            for cand in c.names:
                out.append((c.suite, "comparison", mod.normalize_event_for_candidate(c.event1, cand, c.names)))
                out.append((c.suite, "comparison", mod.normalize_event_for_candidate(c.event2, cand, c.names)))
                for n in c.names:
                    raw_name_hits[n] += int(n.lower() in out[-1][2])
        return out, raw_name_hits
    eval_seqs, raw_name_hits = seqs_for(eval_states, eval_comps)
    train_seqs, train_raw_name_hits = seqs_for(train_states, train_comps)
    unk_by = defaultdict(lambda: [0, 0])
    unk_tokens = Counter()
    for suite, kind, toks in eval_seqs:
        k = f"{suite}|{kind}"
        for t in toks:
            unk_by[k][1] += 1
            if t not in train_vocab.stoi:
                unk_by[k][0] += 1
                unk_tokens[t] += 1
    train_unk = sum(1 for _, _, toks in train_seqs for t in toks if t not in train_vocab.stoi)
    return {
        "arm": arm,
        "counts": counts,
        "parse_errors_sample": parse_errors[:20],
        "train_vocab_size": len(train_vocab.itos),
        "full_train_eval_vocab_size": len(full_vocab.itos),
        "eval_only_vocab_items": sorted([t for t in full_vocab.stoi if t not in train_vocab.stoi])[:200],
        "n_eval_only_vocab_items": len([t for t in full_vocab.stoi if t not in train_vocab.stoi]),
        "train_unknown_tokens_under_train_vocab": train_unk,
        "eval_unknown_by_suite_kind": {k: {"unk": v[0], "total": v[1], "frac": v[0] / v[1] if v[1] else None} for k, v in sorted(unk_by.items())},
        "top_eval_unknown_tokens": unk_tokens.most_common(50),
        "raw_eval_name_tokens_remaining_after_candidate_other_replacement": sum(raw_name_hits.values()),
        "raw_train_name_tokens_remaining_after_candidate_other_replacement": sum(train_raw_name_hits.values()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--arms", nargs="+", default=["aligned_state_bridge", "inverted_state_bridge"])
    args = ap.parse_args()
    mod = load_probe()
    report = {
        "status": "FACTORIZED_PROBE_INPUT_AUDIT_COMPLETE",
        "script": str(SCRIPT),
        "data_root": str(args.data_root),
        "arms": {arm: audit_arm(mod, args.data_root, arm) for arm in args.arms},
        "no_model_loading_training_evaluation_upload_or_leaderboard": True,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "input_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# research factorized probe input audit", ""]
    for arm, ar in report["arms"].items():
        lines += [f"## {arm}", "", f"- train vocab size: {ar['train_vocab_size']}", f"- full train+eval vocab size: {ar['full_train_eval_vocab_size']}", f"- eval-only vocab items: {ar['n_eval_only_vocab_items']}", f"- raw eval/train name tokens remaining after replacement: {ar['raw_eval_name_tokens_remaining_after_candidate_other_replacement']} / {ar['raw_train_name_tokens_remaining_after_candidate_other_replacement']}", "", "| suite/kind | unk | total | frac |", "|---|---:|---:|---:|"]
        for k, v in ar["eval_unknown_by_suite_kind"].items():
            lines.append(f"| {k} | {v['unk']} | {v['total']} | {v['frac']:.3f} |")
        lines += ["", f"Top eval unknown tokens: `{ar['top_eval_unknown_tokens'][:20]}`", ""]
    lines += ["## Scientific reading", "", "Under `--vocab-scope train`, eval names are not accessible as individual learned embeddings because the normalizer replaces the candidate with `<cand>` and the other participant with `<other>`. Eval object words become `<unk>` when absent from training vocabulary. Therefore a surviving tied-vs-untied separation in the train-only replication cannot be attributed to eval-token vocabulary access.", ""]
    (args.out / "input_audit_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": report["status"], "summary": str(args.out / "input_audit_summary.md"), "json": str(args.out / "input_audit.json")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
