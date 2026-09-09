#!/usr/bin/env python3
"""research: verify joint visibility in actual pairfit stage files.

This checks the materialized `qwen10_stagewise_pairfit` corpus directly:
all Qwen pair-atomic examples should be present exactly once, fit inside their
stage sequence length under the same 40k tokenizer used for training, and thus
preserve complete original+rewrite joint visibility. It also records the filler
budget and stage file contract.
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

COMPACT_EXPERIENCE = Path("experiments/archive/compact_experience")
ROOT = Path("experiments/archive/frontier_consolidation")
sys.path.insert(0, str((COMPACT_EXPERIENCE / "scripts").resolve()))
import phase2_sota_trainer as tr  # type: ignore  # noqa: E402

SELECTED_PAIRS = COMPACT_EXPERIENCE / "data/qwen_clean_aligned/selected_pairs.jsonl"
PAIRFIT_DIR = ROOT / "data/qwen10_stagewise_pairfit"
SUMMARY = PAIRFIT_DIR / "stagewise_pairfit_summary.json"
OUT = ROOT / "data/pairfit_contract"
JSON_OUT = OUT / "pairfit_joint_visibility_contract.json"
NOTE_OUT = (ROOT.parents[2] / 'research/notes/frontier_consolidation/pairfit_joint_visibility_contract.md')


def load_selected_pair_ids() -> set[str]:
    ids = set()
    with SELECTED_PAIRS.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ids.add(str(json.loads(line)["pair_id"]))
    return ids


def stage_seq(stage_meta: dict[str, Any]) -> int:
    return int(stage_meta["seq_len"])


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    tokenizer = tr.make_portable_tokenizer(summary["tokenizer_path"])
    selected = load_selected_pair_ids()

    pair_seen: Counter[str] = Counter()
    stage_out: dict[str, Any] = {}
    source_words_total = Counter()
    source_rows_total = Counter()
    pair_token_lengths_all = []
    trunc_examples = []
    word_mismatch = []

    for st, meta in summary["stages"].items():
        path = Path(meta["path"])
        seq = stage_seq(meta)
        rows = words = qrows = qwords = fillers = filler_words = 0
        q_toks = []
        max_tokens = 0
        with path.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                r = json.loads(line)
                rows += 1
                w = len(str(r["text"]).split())
                words += w
                if w != int(r["words"]):
                    word_mismatch.append({"stage": st, "line": i, "field_words": r["words"], "actual_words": w})
                src = str(r.get("orig_source", r.get("source", "")))
                source_words_total[src] += w
                source_rows_total[src] += 1
                if str(r.get("source", "")).startswith("qwen_pair_atomic"):
                    qrows += 1
                    qwords += w
                    pid = str(r["pair_id"])
                    pair_seen[pid] += 1
                    nt = len(tokenizer(str(r["text"]), add_special_tokens=False, truncation=False)["input_ids"])
                    q_toks.append(nt)
                    pair_token_lengths_all.append(nt)
                    max_tokens = max(max_tokens, nt)
                    if nt > seq:
                        trunc_examples.append({"stage": st, "line": i, "pair_id": pid, "tokens": nt, "seq": seq, "words": w})
                else:
                    fillers += 1
                    filler_words += w
        def q(vals, frac):
            if not vals:
                return None
            ss = sorted(vals)
            return ss[min(len(ss)-1, int(round(frac*(len(ss)-1))))]
        stage_out[st] = {
            "path": str(path),
            "seq_len": seq,
            "rows": rows,
            "words": words,
            "qwen_pair_rows": qrows,
            "qwen_pair_words": qwords,
            "filler_rows": fillers,
            "filler_words": filler_words,
            "max_qwen_pair_tokens": max_tokens,
            "qwen_pair_token_quantiles": {"p0": q(q_toks,0), "p50": q(q_toks,.5), "p90": q(q_toks,.9), "p99": q(q_toks,.99), "p100": q(q_toks,1)},
            "qwen_pair_complete_joint_rate": 1.0 if qrows and max_tokens <= seq else (0.0 if qrows else None),
        }

    duplicates = sorted([pid for pid, c in pair_seen.items() if c != 1])
    missing = sorted(selected - set(pair_seen))
    extras = sorted(set(pair_seen) - selected)
    status_ok = (not duplicates and not missing and not extras and not trunc_examples and not word_mismatch and sum(v["words"] for v in stage_out.values()) == 10_000_000)
    payload = {
        "status": "PAIRFIT_JOINT_VISIBILITY_CONTRACT_DONE",
        "ok": bool(status_ok),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": str(SUMMARY),
        "tokenizer_path": summary["tokenizer_path"],
        "tokenizer_vocab_size": len(tokenizer),
        "selected_pairs": str(SELECTED_PAIRS),
        "selected_pair_count": len(selected),
        "seen_pair_count": len(pair_seen),
        "duplicate_or_missing_pair_ids": {"duplicates": duplicates[:20], "missing": missing[:20], "extras": extras[:20], "n_duplicates": len(duplicates), "n_missing": len(missing), "n_extras": len(extras)},
        "truncating_pair_examples": trunc_examples[:20],
        "n_truncating_pair_examples": len(trunc_examples),
        "word_mismatch_sample": word_mismatch[:20],
        "n_word_mismatches": len(word_mismatch),
        "stage_contract": stage_out,
        "total_words": sum(v["words"] for v in stage_out.values()),
        "total_qwen_pair_words": sum(v["qwen_pair_words"] for v in stage_out.values()),
        "total_qwen_pair_rows": sum(v["qwen_pair_rows"] for v in stage_out.values()),
        "complete_pair_joint_visibility_rate": 0.0 if selected and trunc_examples else 1.0,
        "source_words_total": dict(sorted(source_words_total.items())),
        "source_rows_total": dict(sorted(source_rows_total.items())),
        "elapsed_sec": round(time.time() - t0, 3),
    }
    JSON_OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research — pairfit joint-visibility contract",
        "",
        f"JSON: `{JSON_OUT}`",
        "",
        f"Status ok: `{payload['ok']}`",
        "",
        "The actual materialized pairfit corpus keeps every Qwen original+rewrite pair as one training example and assigns it to the shortest containing 40k-token stage. This supersedes the earlier conservative simulated `candidate_pair_atomic` number in `pair_joint_visibility_audit.md` for the actual pairfit arm.",
        "",
        "| stage | seq | rows | words | Qwen pair rows | Qwen pair words | max pair tokens | complete joint rate |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for st, v in stage_out.items():
        lines.append(f"| {st} | {v['seq_len']} | {v['rows']} | {v['words']} | {v['qwen_pair_rows']} | {v['qwen_pair_words']} | {v['max_qwen_pair_tokens']} | {v['qwen_pair_complete_joint_rate']} |")
    lines += [
        "",
        f"All selected Qwen pairs seen exactly once: `{len(pair_seen) == len(selected) and not duplicates and not missing and not extras}`.",
        f"Qwen pair words: `{payload['total_qwen_pair_words']}`; total words: `{payload['total_words']}`.",
        f"Truncating Qwen pair examples under assigned stage windows: `{payload['n_truncating_pair_examples']}`.",
        "",
        "Scientific reading: row-chunked research is a useful low-hidden-word training-dynamics control but breaks complete pair visibility for many early pairs. Pairfit is the correct 10M comparator for asking whether leader-style sequence/masking curriculum can coexist with the clean-Qwen paired-view mechanism.",
    ]
    NOTE_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "ok": payload["ok"],
        "json": str(JSON_OUT),
        "note": str(NOTE_OUT),
        "seen_pair_count": payload["seen_pair_count"],
        "total_qwen_pair_words": payload["total_qwen_pair_words"],
        "complete_pair_joint_visibility_rate": payload["complete_pair_joint_visibility_rate"],
        "n_truncating_pair_examples": payload["n_truncating_pair_examples"],
        "elapsed_sec": payload["elapsed_sec"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
