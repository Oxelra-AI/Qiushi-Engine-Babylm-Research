#!/usr/bin/env python3
"""research CPU audit for Step292b query-attention raw-name script."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch

PATH = Path("experiments/archive/representation_and_objectives/training/scripts/revision_292b_raw_name_attention_probe.py")
OUT = Path("experiments/archive/representation_and_objectives/data/attention_smoke_audit")


def load_mod():
    spec = importlib.util.spec_from_file_location("step292b_raw_name_attention_probe_step293", PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    mod = load_mod()
    data_root = mod.DEFAULT_DATA_ROOT
    arm_dir = data_root / "arms" / mod.DEFAULT_ARM
    eval_dir = data_root / "eval"
    parse_errors = []
    arm_rows = mod.load_jsonl(arm_dir / "train_supervised.jsonl")
    common_rows = mod.load_jsonl(data_root / "common_seen_train.jsonl")
    train_states_arm = mod.build_state_queries(arm_rows, mod.DEFAULT_ARM, parse_errors)
    train_comps_arm = mod.build_comparisons(arm_rows, mod.DEFAULT_ARM, parse_errors)
    train_states_common = mod.build_state_queries(common_rows, "common", parse_errors)
    train_states = train_states_arm + train_states_common
    train_comps = train_comps_arm
    vocab = mod.Vocab(); mod.collect_vocab(vocab, train_states, train_comps)
    qry_id = vocab.stoi["<QRY>"]

    sample = train_states_arm[0]
    candidate_records = []
    seqs = []
    for cand in sample.names:
        toks = mod.encode_event_with_query(sample.event, cand, sample.names)
        ids = vocab.encode(toks)
        qry_positions = [i for i, x in enumerate(ids) if x == qry_id]
        candidate_records.append({
            "candidate": cand,
            "tokens_prefix": toks[:40],
            "n_tokens": len(toks),
            "unk_count": sum(1 for x in ids if x == vocab.stoi["<unk>"]),
            "qry_positions": qry_positions,
            "event_len_before_qry": qry_positions[0] if qry_positions else None,
            "query_len_after_qry": len(toks) - qry_positions[0] - 1 if qry_positions else None,
        })
        seqs.append(ids)

    model = mod.create_single_model(len(vocab.itos), "shared_trunk", qry_id, seed=29300)
    device = torch.device("cpu")
    model.to(device); model.eval()
    with torch.no_grad():
        scores = mod.score_sequences(model.event_state, seqs, device).detach().cpu().tolist()
        cmp_p, cmp_s1, cmp_s2 = mod.comparison_prob_same(model, vocab, train_comps_arm[0], device)
    score_delta = float(scores[0] - scores[1])

    result = {
        "status": "ATTENTION_SMOKE_AUDIT_COMPLETE",
        "vocab_size": len(vocab.itos),
        "parse_error_count": len(parse_errors),
        "sample_event": sample.event,
        "candidate_records": candidate_records,
        "untrained_shared_trunk_event_state_scores": scores,
        "untrained_score_delta_candidate0_minus_candidate1": score_delta,
        "query_conditioned_scores_differ": abs(score_delta) > 1e-9,
        "comparison_sample_prob_same": float(cmp_p.detach().cpu()),
        "comparison_sample_d_e1": float(cmp_s1[0] - cmp_s1[1]),
        "comparison_sample_d_e2": float(cmp_s2[0] - cmp_s2[1]),
        "all_sequences_have_exactly_one_qry": all(len(r["qry_positions"]) == 1 for r in candidate_records),
        "all_sequences_have_nonempty_event_and_query": all((r["event_len_before_qry"] or 0) > 0 and (r["query_len_after_qry"] or 0) > 0 for r in candidate_records),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "attention_smoke_audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# research attention smoke audit\n\n",
             f"Status: {result['status']}\n\n",
             f"Vocab size: {result['vocab_size']}\n\n",
             f"Sample event: {result['sample_event']}\n\n",
             f"Exactly one <QRY>: {result['all_sequences_have_exactly_one_qry']}\n\n",
             f"Nonempty event/query masks: {result['all_sequences_have_nonempty_event_and_query']}\n\n",
             f"Untrained state scores: {scores}\n\n",
             f"Candidate score delta: {score_delta}\n\n",
             f"Query-conditioned scores differ: {result['query_conditioned_scores_differ']}\n\n",
             f"Comparison p_same: {result['comparison_sample_prob_same']}\n"]
    ((OUT.parents[4] / 'research/documents/representation_and_objectives/data/attention_smoke_audit/attention_smoke_audit.md')).write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": result["status"], "json": str(OUT / "attention_smoke_audit.json"), "md": str((OUT.parents[4] / 'research/documents/representation_and_objectives/data/attention_smoke_audit/attention_smoke_audit.md'))}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
