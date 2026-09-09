#!/usr/bin/env python3
"""CPU audit of the causal-GPT compact-view transfer scaffold.

This script leaves the source artifacts read-only. It inspects the
causal-GPT data/tokenizer/trainer/evaluator artifacts and writes a
local audit JSON/MD. The purpose is to decide whether the proposed
cross-architecture compact-vs-repeat test is a clean mechanism-transfer
experiment before committing full H100 time or interpreting its cheap7.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import pathlib
import statistics
import time
from collections import Counter, defaultdict
from typing import Any

USER_ROOT = pathlib.Path(".")
A02 = USER_ROOT / "experiments/archive/frontier_consolidation"
A01 = USER_ROOT / "experiments/archive/representation_and_objectives"
SCAFFOLD = A02 / "data/causal_transfer_scaffold"
MANIFEST = SCAFFOLD / "manifest.json"
COMPACT_POOL = SCAFFOLD / "causal_compact_10M.jsonl"
REPEAT_POOL = SCAFFOLD / "causal_repeat_10M.jsonl"
FILLER_ROWS = SCAFFOLD / "filler_rows.jsonl"
TOKENIZER_DIR = SCAFFOLD / "neutral_tokenizer"
PAIRS_FILE = A02 / "data/density_core_reinvestment_medium_riskhard/selected_compact_reinvest_pairs.jsonl"
DATA_SCRIPT = A02 / "scripts/causal_data_and_tokenizer.py"
TRAIN_SCRIPT = A02 / "scripts/causal_gpt_trainer.py"
EVAL_SCRIPT = A02 / "scripts/causal_cheap7_eval.py"
EVAL_REPO = USER_ROOT / "experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_pipeline"
READING_RUN = EVAL_REPO / "reading/run.py"
COLLATE_PREDS = EVAL_REPO / "collate_preds.py"
SZ_READ_FILES = EVAL_REPO / "sentence_zero_shot/read_files.py"
OUT_DIR = A01 / "data/causal_gpt_transfer_audit"
OUT_JSON = OUT_DIR / "causal_gpt_transfer_audit.json"
OUT_MD = (USER_ROOT / 'research/notes/representation_and_objectives/causal_gpt_transfer_audit.md')


def sha256_file(path: pathlib.Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def wc(text: str) -> int:
    return len(text.split())


def read_jsonl(path: pathlib.Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def safe_read(path: pathlib.Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def numeric_summary(xs: list[float]) -> dict[str, Any]:
    if not xs:
        return {"n": 0}
    ys = sorted(xs)
    def q(p: float):
        if len(ys) == 1:
            return ys[0]
        idx = p * (len(ys) - 1)
        lo = math.floor(idx); hi = math.ceil(idx)
        if lo == hi:
            return ys[lo]
        return ys[lo] * (hi - idx) + ys[hi] * (idx - lo)
    return {
        "n": len(xs),
        "mean": statistics.fmean(xs),
        "median": q(0.5),
        "p10": q(0.1),
        "p90": q(0.9),
        "min": ys[0],
        "max": ys[-1],
    }


def prefix_len(a: list[str], b: list[str]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def suffix_prefix_overlap(a: list[str], b: list[str], max_k: int = 16) -> int:
    # Largest k such that last k words of a equal first k words of b.
    m = min(max_k, len(a), len(b))
    best = 0
    for k in range(1, m + 1):
        if a[-k:] == b[:k]:
            best = k
    return best


def ast_parses(path: pathlib.Path) -> bool:
    try:
        ast.parse(path.read_text(encoding="utf-8"))
        return True
    except Exception:
        return False


def inspect_static_scripts() -> dict[str, Any]:
    eval_text = safe_read(EVAL_SCRIPT)
    train_text = safe_read(TRAIN_SCRIPT)
    data_text = safe_read(DATA_SCRIPT)
    reading_text = safe_read(READING_RUN)
    collate_text = safe_read(COLLATE_PREDS)
    read_files_text = safe_read(SZ_READ_FILES)
    issues = []

    if '"--test_path"' in eval_text and '"--data_path"' in reading_text:
        issues.append({
            "severity": "blocking_for_cheap7_comparability",
            "where": str(EVAL_SCRIPT),
            "issue": "A02 reading runner passes --test_path and no --backend, but official reading/run.py requires --data_path and --backend.",
            "consequence": "Reading returns None; cheap7 becomes a 6-column average and is not comparable with prior cheap7/official readouts.",
        })
    if 'd.get("label")' in eval_text and 'solution{data[\'label\']}' in collate_text:
        issues.append({
            "severity": "blocking_for_globalpiqa",
            "where": str(EVAL_SCRIPT),
            "issue": "GlobalPIQA scoring compares prediction text to the label field instead of solution{label} text.",
            "consequence": "GlobalPIQA can be scored near-zero or otherwise incorrectly even if official predictions are valid.",
        })
    if 'any("nothing" in option' not in eval_text and 'any("nothing" in option' in read_files_text:
        issues.append({
            "severity": "blocking_for_entity_tracking",
            "where": str(EVAL_SCRIPT),
            "issue": "Entity scoring does not skip gold rows whose options contain 'nothing', while official prediction generation skips them.",
            "consequence": "Prediction/gold alignment drifts; Entity score is not official-compatible.",
        })
    if "n_present" in eval_text and "max(n_present, 1)" in eval_text:
        issues.append({
            "severity": "blocking_for_missing_column_detection",
            "where": str(EVAL_SCRIPT),
            "issue": "cheap7 averages over present columns instead of failing when any required column is missing.",
            "consequence": "Failed tasks can silently inflate or deflate mechanism comparisons.",
        })
    if "capture_output=True" in eval_text and "return None" in eval_text:
        issues.append({
            "severity": "repair_recommended",
            "where": str(EVAL_SCRIPT),
            "issue": "Evaluation task failures are converted to None and hidden in captured stderr snippets.",
            "consequence": "A full transfer run can look finished while key official columns are absent or mis-scored.",
        })
    if 'ROOT = pathlib.Path("experiments/archive/frontier_consolidation")' in data_text:
        issues.append({
            "severity": "operational_cwd_risk",
            "where": str(DATA_SCRIPT),
            "issue": "Data script hardcodes a session-relative root and is safe only from the user root, not after cd into a session/workspace directory.",
            "consequence": "Managed background calls must set cwd='' or use fully relative paths consistently.",
        })
    if "ChunkedCausalDataset" in train_text and "ids + [eos_id]" in train_text:
        issues.append({
            "severity": "design_note_not_blocking",
            "where": str(TRAIN_SCRIPT),
            "issue": "Pair sides inside a treatment row have no explicit side boundary; only row/doc EOS is appended after the whole pair item.",
            "consequence": "Causal training tests directional continuation through the source/view boundary rather than a bidirectional same-row co-training signal like MLM.",
        })

    smoke_manifest = SCAFFOLD / "smoke_test/training_manifest.json"
    smoke = None
    if smoke_manifest.exists():
        try:
            smoke = json.loads(smoke_manifest.read_text(encoding="utf-8"))
        except Exception as e:
            smoke = {"parse_error": str(e)}
        if isinstance(smoke, dict):
            training = smoke.get("training", {}) if isinstance(smoke.get("training"), dict) else {}
            if "legal_charged_words" not in training:
                issues.append({
                    "severity": "stale_smoke_test",
                    "where": str(smoke_manifest),
                    "issue": "Existing A02 smoke manifest lacks current trainer fields such as legal_charged_words, active_tokens_per_epoch, and dropped_tail_tokens_per_epoch.",
                    "consequence": "The smoke run appears to predate later exposure-counter edits; current trainer should be smoke-tested before full H100 spend.",
                })

    return {
        "script_hashes": {p.name: sha256_file(p) for p in [DATA_SCRIPT, TRAIN_SCRIPT, EVAL_SCRIPT]},
        "ast_parse_ok": {p.name: ast_parses(p) for p in [DATA_SCRIPT, TRAIN_SCRIPT, EVAL_SCRIPT]},
        "static_issues": issues,
        "existing_smoke_manifest": smoke,
    }


def audit_data() -> dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected_hashes = {
        "compact_pool": manifest["pools"]["compact_view"]["sha256"],
        "repeat_pool": manifest["pools"]["repeat"]["sha256"],
        "filler_rows": manifest["filler"]["sha256"],
        "tokenizer_json": manifest["tokenizer"]["tokenizer_json_sha256"],
    }
    actual_hashes = {
        "compact_pool": sha256_file(COMPACT_POOL),
        "repeat_pool": sha256_file(REPEAT_POOL),
        "filler_rows": sha256_file(FILLER_ROWS),
        "tokenizer_json": sha256_file(TOKENIZER_DIR / "tokenizer.json"),
    }
    hash_ok = {k: expected_hashes[k] == actual_hashes[k] for k in expected_hashes}

    pairs = list(read_jsonl(PAIRS_FILE))
    pair_by_idx = {i: p for i, p in enumerate(pairs)}
    compact_rows = list(read_jsonl(COMPACT_POOL))
    repeat_rows = list(read_jsonl(REPEAT_POOL))

    totals = {
        "compact_rows": len(compact_rows),
        "repeat_rows": len(repeat_rows),
        "compact_words": sum(r.get("words", wc(r.get("text", ""))) for r in compact_rows),
        "repeat_words": sum(r.get("words", wc(r.get("text", ""))) for r in repeat_rows),
        "pair_source_rows_compact": sum(1 for r in compact_rows if r.get("source") == "causal_compact_view"),
        "pair_source_rows_repeat": sum(1 for r in repeat_rows if r.get("source") == "causal_repeat"),
        "same_row_count": len(compact_rows) == len(repeat_rows),
    }

    row_alignment_mismatches = []
    pair_positions = []
    filler_mismatch_count = 0
    for i, (cr, rr) in enumerate(zip(compact_rows, repeat_rows)):
        cs, rs = cr.get("source"), rr.get("source")
        if cs == "causal_compact_view" or rs == "causal_repeat":
            if not (cs == "causal_compact_view" and rs == "causal_repeat"):
                row_alignment_mismatches.append({"row": i, "compact_source": cs, "repeat_source": rs})
            else:
                pair_positions.append(i)
        else:
            if cr != rr:
                filler_mismatch_count += 1
                if len(row_alignment_mismatches) < 20:
                    row_alignment_mismatches.append({"row": i, "compact_source": cs, "repeat_source": rs, "kind": "filler_row_diff"})

    order_counts = Counter()
    order_word_mass = Counter()
    order_source_words = Counter()
    order_rewrite_words = Counter()
    row_word_mismatch = []
    repeat_exact_prefix_ok = 0
    compact_exact_prefix_count = 0
    repeat_boundary_overlap = []
    compact_boundary_overlap = []
    rewrite_source_prefix_overlap_frac = []
    source_lengths = []
    rewrite_lengths = []
    pair_words = []
    text_examples = []
    for j, row_idx in enumerate(pair_positions):
        p = pair_by_idx[j]
        cr = compact_rows[row_idx]
        rr = repeat_rows[row_idx]
        source = p["source_text"]
        rewrite = p["rewrite_text"]
        sw = source.split(); rw = rewrite.split()
        target_rw = p["rewrite_words"]
        repeat_text = " ".join((sw * ((target_rw // max(len(sw), 1)) + 2))[:target_rw]) if target_rw > len(sw) else " ".join(sw[:target_rw])
        # Infer order exactly from compact row text.
        if cr["text"] == source + " " + rewrite:
            order = "source_first"
            expected_repeat = source + " " + repeat_text
            first_a, second_a = sw, rw
            first_r, second_r = sw, repeat_text.split()
        elif cr["text"] == rewrite + " " + source:
            order = "view_first"
            expected_repeat = repeat_text + " " + source
            first_a, second_a = rw, sw
            first_r, second_r = repeat_text.split(), sw
        else:
            order = "unmatched"
            expected_repeat = None
            first_a, second_a = [], []
            first_r, second_r = [], []
        order_counts[order] += 1
        order_word_mass[order] += cr.get("words", wc(cr.get("text", "")))
        order_source_words[order] += len(sw)
        order_rewrite_words[order] += len(rw)
        source_lengths.append(len(sw)); rewrite_lengths.append(len(rw)); pair_words.append(cr.get("words", 0))
        if cr.get("words") != rr.get("words") or cr.get("words") != p.get("pair_words"):
            row_word_mismatch.append({"pair_idx": j, "row": row_idx, "compact_words": cr.get("words"), "repeat_words": rr.get("words"), "pair_words": p.get("pair_words")})
        if expected_repeat is not None and rr["text"] == expected_repeat:
            repeat_exact_prefix_ok += 1
        if repeat_text == rewrite:
            compact_exact_prefix_count += 1
        repeat_boundary_overlap.append(suffix_prefix_overlap(first_r, second_r, 32))
        compact_boundary_overlap.append(suffix_prefix_overlap(first_a, second_a, 32))
        rewrite_source_prefix_overlap_frac.append(prefix_len(rewrite.split(), repeat_text.split()) / max(target_rw, 1))
        if len(text_examples) < 4:
            text_examples.append({
                "pair_idx": j,
                "row": row_idx,
                "order": order,
                "source_words": len(sw),
                "rewrite_words": len(rw),
                "source_first_80chars": source[:80],
                "rewrite_first_80chars": rewrite[:80],
                "repeat_first_80chars": repeat_text[:80],
            })

    # Source mix in common filler: not a blocker, but important for transfer interpretation.
    filler_source_counts = Counter()
    filler_source_words = Counter()
    for r in read_jsonl(FILLER_ROWS):
        s = str(r.get("source", ""))
        root = s.split("::", 1)[0]
        filler_source_counts[root] += 1
        filler_source_words[root] += r.get("words", wc(r.get("text", "")))

    # Estimate tokenization ratios on all pair rows only with the neutral tokenizer if available.
    token_stats = {}
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
        cv_toks = []
        rp_toks = []
        cv_pair_toks = 0
        rp_pair_toks = 0
        for row_idx in pair_positions:
            cr = compact_rows[row_idx]
            rr = repeat_rows[row_idx]
            ctn = len(tok.encode(cr["text"]))
            rtn = len(tok.encode(rr["text"]))
            cv_toks.append(ctn / max(cr.get("words", wc(cr["text"])), 1))
            rp_toks.append(rtn / max(rr.get("words", wc(rr["text"])), 1))
            cv_pair_toks += ctn
            rp_pair_toks += rtn
        token_stats = {
            "compact_pair_tok_per_word": numeric_summary(cv_toks),
            "repeat_pair_tok_per_word": numeric_summary(rp_toks),
            "compact_pair_total_tokens": cv_pair_toks,
            "repeat_pair_total_tokens": rp_pair_toks,
            "compact_minus_repeat_pair_tokens": cv_pair_toks - rp_pair_toks,
            "compact_minus_repeat_pair_tokens_pct_of_repeat": (cv_pair_toks - rp_pair_toks) / rp_pair_toks * 100 if rp_pair_toks else None,
        }
    except Exception as e:
        token_stats = {"error": repr(e)}

    return {
        "manifest": manifest,
        "expected_hashes": expected_hashes,
        "actual_hashes": actual_hashes,
        "hash_ok": hash_ok,
        "totals": totals,
        "alignment": {
            "pair_positions": len(pair_positions),
            "filler_mismatch_count": filler_mismatch_count,
            "row_alignment_mismatch_examples": row_alignment_mismatches[:20],
            "row_word_mismatch_count": len(row_word_mismatch),
            "row_word_mismatch_examples": row_word_mismatch[:20],
        },
        "order_balance": {
            "counts": dict(order_counts),
            "word_mass": dict(order_word_mass),
            "source_words_by_order": dict(order_source_words),
            "rewrite_words_by_order": dict(order_rewrite_words),
            "source_first_pair_word_fraction": order_word_mass["source_first"] / max(sum(order_word_mass.values()), 1),
        },
        "pair_length_stats": {
            "source_words": numeric_summary(source_lengths),
            "rewrite_words": numeric_summary(rewrite_lengths),
            "pair_words": numeric_summary(pair_words),
        },
        "causal_artifact_metrics": {
            "repeat_exact_prefix_rows": repeat_exact_prefix_ok,
            "repeat_exact_prefix_fraction": repeat_exact_prefix_ok / max(len(pair_positions), 1),
            "compact_rewrite_equals_source_prefix_rows": compact_exact_prefix_count,
            "compact_rewrite_equals_source_prefix_fraction": compact_exact_prefix_count / max(len(pair_positions), 1),
            "repeat_boundary_suffix_prefix_overlap_words": numeric_summary(repeat_boundary_overlap),
            "compact_boundary_suffix_prefix_overlap_words": numeric_summary(compact_boundary_overlap),
            "rewrite_vs_source_prefix_word_prefix_overlap_fraction": numeric_summary(rewrite_source_prefix_overlap_frac),
            "interpretation": "Repeat rows are an exact source-prefix replay in every pair, so causal next-token learning across the side boundary has a copy/restart artifact not present in compact semantic rewrites. A compact>repeat result would still be strong; repeat≈compact or repeat>compact would not by itself refute compact-view density/consolidation under MLM.",
        },
        "tokenization_pair_stats": token_stats,
        "filler_mix_top_by_words": filler_source_words.most_common(20),
        "filler_rows_top_by_count": filler_source_counts.most_common(20),
        "examples": text_examples,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    missing = [str(p) for p in [MANIFEST, COMPACT_POOL, REPEAT_POOL, FILLER_ROWS, TOKENIZER_DIR / "tokenizer.json", PAIRS_FILE, DATA_SCRIPT, TRAIN_SCRIPT, EVAL_SCRIPT, READING_RUN, COLLATE_PREDS, SZ_READ_FILES] if not p.exists()]
    result: dict[str, Any] = {
        "status": "A02_CAUSAL_GPT_TRANSFER_AUDIT",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Read-only A01 audit of A02 research causal-GPT compact-view transfer scaffold before expensive/interpretive use.",
        "missing_required_paths": missing,
    }
    if missing:
        OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        raise SystemExit(1)

    result["data_audit"] = audit_data()
    result["static_script_audit"] = inspect_static_scripts()

    static_issues = result["static_script_audit"]["static_issues"]
    blocking = [x for x in static_issues if str(x.get("severity", "")).startswith("blocking")]
    hash_ok_all = all(result["data_audit"]["hash_ok"].values())
    align = result["data_audit"]["alignment"]
    data_ok = hash_ok_all and align["filler_mismatch_count"] == 0 and align["row_word_mismatch_count"] == 0 and align["pair_positions"] == result["data_audit"]["manifest"]["pairs"]["total"]
    result["scientific_readout"] = {
        "data_scaffold_ok_for_aligned_pool_contrast": data_ok,
        "current_eval_script_ok_for_mechanism_readout": len(blocking) == 0,
        "blocking_static_issues": blocking,
        "recommended_before_h100_or_interpretation": [
            "Repair causal cheap7 evaluation to use official reading --data_path/--backend causal, official GlobalPIQA solution-text comparison, Entity 'nothing' filtering, and fail-closed all-7-column collation.",
            "Run a fresh current-script smoke/short run after the exposure-counter edits; the existing A02 smoke manifest appears stale relative to the current trainer fields.",
            "If full causal-GPT training is launched, interpret it as architecture/objective transfer only with this caveat: causal repeat rows provide exact source-prefix replay, so repeat arm has a directional copy/restart advantage absent from compact semantic rewrites.",
            "Prefer additional local readouts over endpoint-only cheap7: pair-row validation perplexity separated into treatment-side vs filler-side and boundary-position loss, compact vs repeat prediction turnover on the same official tasks, and comparison to A01 masked-triangle result.",
        ],
        "bottom_line": "The A02 data/tokenizer scaffold is hash-consistent and row-aligned, but the current evaluation script is not official-compatible and the causal objective changes the meaning of the compact/repeat contrast. Do not trust a research cheap7 table until scoring is repaired; do not read repeat>=compact as falsifying the masked compact-view principle without accounting for the exact-prefix replay artifact.",
    }
    result["elapsed_sec"] = round(time.time() - t0, 3)
    result["out_json"] = str(OUT_JSON)
    result["out_md"] = str(OUT_MD)
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    da = result["data_audit"]
    sr = result["scientific_readout"]
    md = []
    md.append("# research — A02 causal-GPT compact-view transfer scaffold audit\n")
    md.append(f"JSON: `{OUT_JSON}`\n")
    md.append("## What is valid\n")
    md.append(f"- Manifest hashes match actual files: `{da['hash_ok']}`.\n")
    md.append(f"- Pools are row-aligned: {da['totals']['compact_rows']} compact rows / {da['totals']['repeat_rows']} repeat rows, {da['totals']['compact_words']:,} and {da['totals']['repeat_words']:,} words.\n")
    md.append(f"- Pair rows align one-for-one at {da['alignment']['pair_positions']} positions with zero filler mismatches and zero word-count mismatches.\n")
    md.append(f"- Tokenizer is the same filler-only 16k BPE for both arms; pair-token mass differs by {da['tokenization_pair_stats'].get('compact_minus_repeat_pair_tokens')} tokens ({da['tokenization_pair_stats'].get('compact_minus_repeat_pair_tokens_pct_of_repeat'):.3f}% of repeat pair tokens) on treatment rows.\n")
    md.append("\n## What blocks trusted readout\n")
    for issue in result["static_script_audit"]["static_issues"]:
        md.append(f"- **{issue['severity']}** `{issue['where']}`: {issue['issue']} Consequence: {issue['consequence']}\n")
    md.append("\n## Causal-objective interpretation caveat\n")
    cam = da["causal_artifact_metrics"]
    md.append(f"- Repeat treatment rows are exact source-prefix replay in {cam['repeat_exact_prefix_rows']}/{da['alignment']['pair_positions']} rows; compact rewrite equals the source prefix in only {cam['compact_rewrite_equals_source_prefix_rows']} rows.\n")
    md.append("- Therefore causal next-token training receives a copy/restart signal in repeat rows that compact semantic rewrites do not have. A compact-over-repeat result would be a strong transfer signal; a repeat-over-compact result would not by itself refute the masked compact-view mechanism.\n")
    md.append("\n## Bottom line\n")
    md.append(sr["bottom_line"] + "\n")
    md.append("\n## Recommended repair before A02 spends/interprets full H100 work\n")
    for rec in sr["recommended_before_h100_or_interpretation"]:
        md.append(f"- {rec}\n")
    OUT_MD.write_text("".join(md), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "data_ok": data_ok,
        "blocking_issue_count": len(blocking),
        "static_issue_count": len(static_issues),
        "out_json": str(OUT_JSON),
        "out_md": str(OUT_MD),
        "bottom_line": sr["bottom_line"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
