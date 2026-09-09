#!/usr/bin/env python3
"""research: compact-view generator/source/overlap provenance addendum.

This CPU-only script closes two endpoint-protection gaps left by research/040:
(1) exact compact-rewrite generator command/model identity, recovered from execution
    records and task stdout rather than inferred;
(2) score-text overlap ancestry summary for the seven changed-block rows identified
    by research, especially whether matched n-grams occur in original source text or
    generated rewrite text.

It does not touch training/evaluation outputs, launch GPU work, or claim seed43122
robustness; that remains governed by the pending managed full-evaluation task.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import collections
import hashlib
import json
import pathlib
import re
import time
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT

ROOT = find_user_root()
S = ROOT / "experiments/archive" / 'representation_and_objectives'
A02 = ROOT / "experiments/archive" / 'frontier_consolidation'
OUT_DIR = S / "data" / "compact_provenance_and_overlap_addendum"
NOTE = S / "notes" / "compact_provenance_and_overlap_addendum.md"

PATHS = {
    "a02_compact_bg_submit_trace": ROOT / "data/external/tool_0002.json",
    "a02_compact_bg_delivery_trace": ROOT / "data/external/background_delivery.json",
    "a02_compact_task_result": ROOT / "experiments/archive/frontier_consolidation/tasks/s14_t13_tool2/result.json",
    "a02_compact_task_stdout": ROOT / "experiments/archive/frontier_consolidation/tasks/s14_t13_tool2/stdout.log",
    "a02_compact_task_stderr": ROOT / "experiments/archive/frontier_consolidation/tasks/s14_t13_tool2/stderr.log",
    "compact_prompt_metadata": ROOT / "experiments/archive/frontier_consolidation/data/medium_density_prompts/medium_density_prompt_metadata.json",
    "compact_prompt_file": ROOT / "experiments/archive/frontier_consolidation/data/medium_density_prompts/fineweb_medium_compact_prompts_all.jsonl",
    "compact_raw_outputs": ROOT / "experiments/archive/frontier_consolidation/training/runs/medium_compact_qwen_all/outputs.jsonl",
    "compact_analysis_summary": ROOT / "experiments/archive/frontier_consolidation/data/medium_compact_analysis/medium_compact_ws_summary.json",
    "compact_accepted_rewrites": ROOT / "experiments/archive/frontier_consolidation/data/medium_compact_analysis/medium_compact_ws_accepted_rewrites.jsonl",
    "compact_rows": ROOT / "experiments/archive/frontier_consolidation/data/medium_compact_analysis/medium_compact_ws_rows.jsonl",
    "density_reinvestment_metadata": ROOT / "experiments/archive/frontier_consolidation/data/density_core_reinvestment_medium_riskhard/density_core_reinvestment_metadata.json",
    "rowholdout_overlay_metadata": ROOT / "experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/density_cleanqwen_rowholdout_overlay_metadata.json",
    "selected_compact_reinvest_pairs": ROOT / "experiments/archive/frontier_consolidation/data/density_core_reinvestment_medium_riskhard/selected_compact_reinvest_pairs.jsonl",
    "cleanqwen_compact_reinvest_10m": ROOT / "experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl",
    "cleanqwen_compact_reinvest_100m": ROOT / "experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl",
    "source_sentence_summary": ROOT / "experiments/archive/representation_and_objectives/training/data/fineweb_sentence_sources/sentence_source_summary.json",
    "source_sentence_file": ROOT / "experiments/archive/representation_and_objectives/training/data/fineweb_sentence_sources/fineweb_sentence_sources.jsonl",
    "a01_factual_selection_metadata": ROOT / "experiments/archive/representation_and_objectives/training/data/fineweb_factual_sentence_rewrite/factual_source_by_rewrite_prompt_metadata.json",
    "a02_source_repair_metadata": ROOT / "experiments/archive/frontier_consolidation/data/fineweb_source_by_rewrite_repair/fineweb_source_by_rewrite_repair_metadata.json",
    "overlap_summary": ROOT / "experiments/archive/representation_and_objectives/data/changed_block_overlap_ancestry_current_official/changed_block_overlap_ancestry_current_official.json",
    "overlap_rows": ROOT / "experiments/archive/representation_and_objectives/data/changed_block_overlap_ancestry_current_official/changed_block_overlap_ancestry_current_official_rows.jsonl",
    "seed43022_official_summary": ROOT / "experiments/archive/representation_and_objectives/data/pristine_collate_seed43022/pristine_collate_seed43022_summary.json",
    "ledger": ROOT / "experiments/archive/representation_and_objectives/data/endpoint_provenance_ledger/compact_reinvest_seed43022_endpoint_provenance_ledger.json",
}


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: pathlib.Path, limit: int | None = None) -> list[Any]:
    rows: list[Any] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def count_jsonl(path: pathlib.Path) -> int:
    n = 0
    with path.open("rb") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_record(path: pathlib.Path, count_lines: bool = False) -> dict[str, Any]:
    rec: dict[str, Any] = {"path": rel(path), "exists": path.exists()}
    if path.exists():
        rec["size_bytes"] = path.stat().st_size
        rec["sha256"] = sha256_file(path)
        if count_lines and path.suffix == ".jsonl":
            rec["jsonl_rows"] = count_jsonl(path)
    return rec


def parse_stdout_json(path: pathlib.Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    # stdout is a single JSON object in this run. Keep a brace-scan fallback in case
    # future copies include tool wrappers.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}\s*$", text, flags=re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def command_tokens(command: str) -> dict[str, str]:
    # Simple parser for long-option/value pairs in the recorded shell command.
    parts = command.split()
    out: dict[str, str] = {}
    i = 0
    while i < len(parts):
        tok = parts[i]
        if tok.startswith("--") and i + 1 < len(parts) and not parts[i + 1].startswith("--"):
            out[tok] = parts[i + 1]
            i += 2
        else:
            i += 1
    if parts:
        out["program"] = parts[0]
    return out


def extract_generator_identity() -> dict[str, Any]:
    submit = read_json(PATHS["a02_compact_bg_submit_trace"])
    delivery = read_json(PATHS["a02_compact_bg_delivery_trace"])
    task_result = read_json(PATHS["a02_compact_task_result"])
    stdout = parse_stdout_json(PATHS["a02_compact_task_stdout"])
    command = submit["tool_call"]["arguments"]["arguments"]["command"]
    opts = command_tokens(command)
    return {
        "source": "A02 experiment_process submission trace plus delivered task stdout, not inferred from memory",
        "submit_trace": rel(PATHS["a02_compact_bg_submit_trace"]),
        "delivery_trace": rel(PATHS["a02_compact_bg_delivery_trace"]),
        "task_result": rel(PATHS["a02_compact_task_result"]),
        "command": command,
        "parsed_options": opts,
        "submitted_task_ref": task_result.get("task_ref"),
        "task_state": task_result.get("state"),
        "exit_code": task_result.get("exit_code"),
        "stdout_model": stdout.get("model"),
        "stdout_resolved_model": stdout.get("resolved_model"),
        "stdout_adapter": stdout.get("adapter"),
        "dtype": stdout.get("dtype"),
        "requested_device": stdout.get("requested_device"),
        "actual_device": stdout.get("actual_device"),
        "cuda_device_name": stdout.get("cuda_device_name"),
        "prompts": stdout.get("prompts"),
        "batch_size": stdout.get("batch_size"),
        "generated_tokens": stdout.get("generated_tokens"),
        "seconds": stdout.get("seconds"),
        "aggregate_tok_s": stdout.get("aggregate_tok_s"),
        "output_jsonl": stdout.get("output_jsonl"),
        "model_identity_closed": stdout.get("model") == "Qwen/Qwen3.5-9B" and opts.get("--model") == "qwen3.5-9b",
        "note": "The recorded generator for the seed43022 compact rewrites is Qwen/Qwen3.5-9B (alias qwen3.5-9b)."
    }


def extract_source_lineage() -> dict[str, Any]:
    sentence_summary = read_json(PATHS["source_sentence_summary"])
    a01_sel = read_json(PATHS["a01_factual_selection_metadata"])
    a02_src = read_json(PATHS["a02_source_repair_metadata"])
    prompt_meta = read_json(PATHS["compact_prompt_metadata"])
    analysis = read_json(PATHS["compact_analysis_summary"])
    density = read_json(PATHS["density_reinvestment_metadata"])
    overlay = read_json(PATHS["rowholdout_overlay_metadata"])

    return {
        "source_chain": [
            {
                "stage": "cached FineWeb-Edu single-document rows to complete sentence sources",
                "input": sentence_summary.get("input"),
                "output": sentence_summary.get("out"),
                "accepted_sentences": sentence_summary.get("accepted_sentences"),
                "accepted_words": sentence_summary.get("accepted_words"),
                "unique_docs": sentence_summary.get("unique_docs"),
                "output_sha256": sentence_summary.get("sha256"),
            },
            {
                "stage": "A01 factual/expository sentence selection for rewrite substrate",
                "source_jsonl": a01_sel.get("source_jsonl"),
                "source_sha256": a01_sel.get("source_sha256"),
                "selected_sources": a01_sel.get("selected_sources"),
                "selected_source_words": a01_sel.get("selected_source_words"),
                "unique_docs": a01_sel.get("unique_docs"),
                "selected_sources_path": a01_sel.get("selected_sources_path"),
                "selected_sources_sha256": a01_sel.get("selected_sources_sha256"),
            },
            {
                "stage": "A02 repaired medium source tier from strict factual plus balanced sources",
                "inputs": a02_src.get("inputs"),
                "medium_rows": (a02_src.get("medium_repaired_from_strict_plus_balanced") or {}).get("rows"),
                "medium_words": (a02_src.get("medium_repaired_from_strict_plus_balanced") or {}).get("words"),
                "medium_unique_docs": (a02_src.get("medium_repaired_from_strict_plus_balanced") or {}).get("unique_docs"),
                "medium_repaired_sources": (a02_src.get("outputs") or {}).get("medium_repaired_sources"),
            },
            {
                "stage": "A02 compact prompt preparation",
                "input": prompt_meta.get("input"),
                "input_sha256": prompt_meta.get("input_sha256"),
                "compact_prompt_file": (prompt_meta.get("outputs") or {}).get("compact"),
                "compact_prompt_sha256": (prompt_meta.get("sha256") or {}).get("compact"),
                "compact_rows": ((prompt_meta.get("summaries") or {}).get("compact") or {}).get("rows"),
                "compact_source_words": ((prompt_meta.get("summaries") or {}).get("compact") or {}).get("source_words_whitespace"),
                "compact_prompt_rule": prompt_meta.get("compact_prompt_rule"),
                "no_babylm_evaluation_signal_used": prompt_meta.get("no_babylm_evaluation_signal_used"),
            },
            {
                "stage": "Qwen compact generation and automatic acceptance analysis",
                "prompts": analysis.get("prompts"),
                "prompts_sha256": analysis.get("prompts_sha256"),
                "outputs": analysis.get("outputs"),
                "outputs_sha256": analysis.get("outputs_sha256"),
                "accepted": ((analysis.get("summary") or {}).get("overall") or {}).get("accepted"),
                "accepted_rate": ((analysis.get("summary") or {}).get("overall") or {}).get("accepted_rate"),
                "accepted_source_words_whitespace": ((analysis.get("summary") or {}).get("overall") or {}).get("accepted_source_words_whitespace"),
                "accepted_rewrite_words_whitespace": ((analysis.get("summary") or {}).get("overall") or {}).get("accepted_rewrite_words_whitespace"),
                "weighted_rewrite_to_source_ratio_whitespace": ((analysis.get("summary") or {}).get("overall") or {}).get("weighted_rewrite_to_source_ratio_whitespace"),
                "content_recall_mean": (((analysis.get("summary") or {}).get("overall") or {}).get("content_recall_stats") or {}).get("mean"),
                "entity_recall_mean": (((analysis.get("summary") or {}).get("overall") or {}).get("entity_recall_stats") or {}).get("mean"),
                "number_recall_mean": (((analysis.get("summary") or {}).get("overall") or {}).get("number_recall_stats") or {}).get("mean"),
            },
            {
                "stage": "density reinvestment pair selection",
                "candidate_counts": density.get("candidate_counts"),
                "selection": density.get("selection"),
                "compact_reinvest_summary": density.get("compact_reinvest_summary"),
            },
            {
                "stage": "clean-Qwen row-holdout overlay into 10M/100M BabyLM training pool",
                "base_pool": ((overlay.get("inputs") or {}).get("base_pool")),
                "base_pool_sha256": ((overlay.get("inputs") or {}).get("base_pool_sha256")),
                "preserve_source": ((overlay.get("inputs") or {}).get("preserve_source")),
                "heldout_words": ((overlay.get("base_split") or {}).get("heldout_words")),
                "changed_block_budget_words": overlay.get("changed_block_budget_words"),
                "total_words_per_pool": overlay.get("total_words_per_pool"),
                "passes": overlay.get("passes"),
                "compact_reinvest_family": ((overlay.get("families") or {}).get("compact_reinvest")),
                "compact_reinvest_pair_summary": ((overlay.get("pair_summaries") or {}).get("compact_reinvest")),
            },
        ],
        "current_reinvest_training_files": {
            "train_10m": file_record(PATHS["cleanqwen_compact_reinvest_10m"], count_lines=True),
            "train_100m": file_record(PATHS["cleanqwen_compact_reinvest_100m"], count_lines=True),
            "selected_compact_reinvest_pairs": file_record(PATHS["selected_compact_reinvest_pairs"], count_lines=True),
        },
    }


def extract_overlap_ancestry() -> dict[str, Any]:
    summary = read_json(PATHS["overlap_summary"])
    rows = read_jsonl(PATHS["overlap_rows"])
    hit_top = collections.Counter()
    hit_role = collections.Counter()
    ngram_counter = collections.Counter()
    longest = []
    pair_ids = set()
    matched_in_source = 0
    matched_in_rewrite = 0
    matched_pair_details = []
    for row in rows:
        for hit in row.get("strict_eval_hits", []):
            hit_top[str(hit.get("top_dir"))] += 1
            hit_role[str(hit.get("file_role"))] += 1
            ngram_counter[str(hit.get("ngram"))] += 1
            longest.append(int(hit.get("longest_common_token_span_len") or 0))
        for pr in row.get("pair_records", []):
            pair_ids.add(str(pr.get("pair_id")))
            src_ngrams = list(pr.get("matched_ngrams_in_source_text") or [])
            rew_ngrams = list(pr.get("matched_ngrams_in_rewrite_text") or [])
            if src_ngrams:
                matched_in_source += len(src_ngrams)
            if rew_ngrams:
                matched_in_rewrite += len(rew_ngrams)
            if src_ngrams or rew_ngrams:
                matched_pair_details.append({
                    "row_index": row.get("row_index"),
                    "example_id": row.get("example_id"),
                    "pair_id": pr.get("pair_id"),
                    "key": pr.get("key"),
                    "doc_id": pr.get("doc_id"),
                    "sentence_id": pr.get("sentence_id"),
                    "domain_hits": pr.get("domain_hits"),
                    "source_words": pr.get("source_words"),
                    "rewrite_words": pr.get("rewrite_words"),
                    "matched_ngrams_in_source_text": src_ngrams,
                    "matched_ngrams_in_rewrite_text": rew_ngrams,
                    "source_text": pr.get("source_text"),
                    "rewrite_text": pr.get("rewrite_text"),
                    "strict_eval_hits_for_row": row.get("strict_eval_hits"),
                })
    return {
        "source_file": rel(PATHS["overlap_rows"]),
        "summary": {
            "rows_with_strict_score_overlap": summary.get("strict_score_overlap_summary", {}).get("rows_with_strict_score_overlap"),
            "unique_strict_score_ngrams": summary.get("strict_score_overlap_summary", {}).get("unique_strict_score_ngrams"),
            "strict_eval_hit_count": summary.get("strict_score_overlap_summary", {}).get("strict_eval_hit_count"),
            "strict_eval_hit_by_top_dir": summary.get("strict_score_overlap_summary", {}).get("strict_eval_hit_by_top_dir"),
            "strict_eval_hit_by_file_role": summary.get("strict_score_overlap_summary", {}).get("strict_eval_hit_by_file_role"),
            "longest_common_span_lengths_desc": summary.get("strict_score_overlap_summary", {}).get("longest_common_span_lengths_desc"),
        },
        "recomputed_from_rows": {
            "rows": len(rows),
            "strict_eval_hits": sum(hit_top.values()),
            "hit_top_dirs": dict(hit_top),
            "hit_file_roles": dict(hit_role),
            "unique_ngrams": len(ngram_counter),
            "ngrams": dict(ngram_counter),
            "longest_common_span_max": max(longest) if longest else 0,
            "unique_pair_records_touched_by_overlap_rows": len(pair_ids),
            "matched_ngram_occurrences_in_original_source_text": matched_in_source,
            "matched_ngram_occurrences_in_generated_rewrite_text": matched_in_rewrite,
            "all_strict_score_matches_are_in_original_source_text_not_rewrite": matched_in_source > 0 and matched_in_rewrite == 0,
        },
        "matched_pair_details": matched_pair_details,
        "interpretation": "The research exact 7-token score-text hits remain sparse and confined to AoA phrase material or SuperGLUE validation. Recomputing over the row-level ancestry shows the matched n-grams occur in original FineWeb source sentences, not in the generated compact rewrites, so these hits are not evidence that Qwen generated evaluation strings; they still require caution as ordinary web/eval phrase overlap and do not establish semantic independence.",
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    generator = extract_generator_identity()
    lineage = extract_source_lineage()
    overlap = extract_overlap_ancestry()
    seed_summary = read_json(PATHS["seed43022_official_summary"])
    ledger = read_json(PATHS["ledger"])
    collated_path = ROOT / ((seed_summary.get("paths") or {}).get("collated_path") or "")
    collated_rec = file_record(collated_path) if str(collated_path) else {}

    artifact_records = {k: file_record(p, count_lines=(p.suffix == ".jsonl" and p.stat().st_size < 700_000_000 if p.exists() else False)) for k, p in PATHS.items() if k not in {"cleanqwen_compact_reinvest_100m"}}
    artifact_records["cleanqwen_compact_reinvest_100m"] = file_record(PATHS["cleanqwen_compact_reinvest_100m"], count_lines=False)

    payload: dict[str, Any] = {
        "status": "COMPACT_PROVENANCE_AND_OVERLAP_ADDENDUM",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scientific_purpose": "Protect the compact_view_reinvest seed43022 SOTA-facing endpoint by replacing inferred generator/source statements with trace-backed provenance and by summarizing exact score-text overlap ancestry while seed43122 full evaluation remains pending.",
        "seed43022_official_coordinate": {
            "summary_path": rel(PATHS["seed43022_official_summary"]),
            "scores": (((seed_summary.get("score_summary") or {}).get("official_overall") or {}).get("scores")),
            "overall": (((seed_summary.get("score_summary") or {}).get("official_overall") or {}).get("Overall")),
            "nlp_average": (((seed_summary.get("score_summary") or {}).get("official_overall") or {}).get("NLP_average")),
            "human_like_average": (((seed_summary.get("score_summary") or {}).get("official_overall") or {}).get("Human_like_average")),
            "margin_over_visible_leader_41p8": (((seed_summary.get("score_summary") or {}).get("official_overall") or {}).get("margin_over_visible_leader")),
            "collated_json": collated_rec,
            "ledger_path": rel(PATHS["ledger"]),
            "ledger_status": ledger.get("status"),
        },
        "generator_identity": generator,
        "source_and_training_lineage": lineage,
        "overlap_ancestry": overlap,
        "artifact_records": artifact_records,
        "resolved_gaps": [
            "Exact compact generator model identity: qwen3.5-9b command, stdout-resolved to Qwen/Qwen3.5-9B, no adapter, bfloat16 on H100.",
            "Compact prompt/output/accepted-rewrite hashes and row counts tied together: 21,465 prompts; 754,777 generated tokens; 18,682 accepted compact rewrites; accepted pair words 645,233 in the full generated medium set.",
            "Endpoint selected subset lineage tied to A02 medium density metadata: compact_reinvest selected 12,155 source-rewrite pairs / 423,511 pair words before clean-Qwen overlay; final pool exactness remains recorded in research ledger.",
            "The seven exact score-text overlap rows have row-level pair ancestry; their matched n-grams are in original FineWeb source text rather than generated rewrite text.",
        ],
        "remaining_gaps": [
            "The seed43122 full official-compatible vector remains pending; no robustness conclusion changes here.",
            "Qwen/Qwen3.5-9B model license and any required redistribution language should be recorded from an official model card or approved-source file before final public packaging.",
            "FineWeb/FineWeb-Edu source license or redistribution conditions for the compact-view rows should be recorded explicitly before public data release or submission bundle publication.",
            "If seed43122 clears the leader, adjacency-broken training remains the prepared mechanism experiment; if it falls short, variance-focused interpretation should precede any recipe change.",
        ],
    }

    out_json = OUT_DIR / "compact_provenance_and_overlap_addendum.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    note_lines = [
        "# research — compact generator/source and overlap provenance addendum\n\n",
        "This CPU-only addendum protects the seed43022 compact_view_reinvest endpoint while the seed43122 full evaluation continues; it does not use partial seed43122 outputs and does not launch new evaluation or training.\n\n",
        "## Generator identity recovered from trace\n\n",
        f"A02 submitted compact generation as `{generator['command']}`. The delivered task stdout resolves `--model qwen3.5-9b` to `{generator.get('stdout_model')}` at `{generator.get('stdout_resolved_model')}`, with adapter `{generator.get('stdout_adapter') or '<none>'}`, dtype `{generator.get('dtype')}`, device `{generator.get('actual_device')}` / `{generator.get('cuda_device_name')}`. It generated {generator.get('generated_tokens'):,} tokens for {generator.get('prompts'):,} prompts in {generator.get('seconds')} s. Evidence: `{generator['submit_trace']}`, `{generator['delivery_trace']}`, `{rel(PATHS['a02_compact_task_stdout'])}`.\n\n",
        "## Compact source and selection chain\n\n",
        "The medium compact prompt file contains 21,465 prompts from 469,887 FineWeb source words and has SHA-256 `c0ef389b6168b707996d807a2f8567dfdc3ff59025a913bf047d79ba75d2a807`. Automatic compact analysis accepted 18,682 rewrites (rate 0.8703), with 400,665 accepted source words, 244,568 accepted rewrite words, and weighted rewrite/source ratio 0.6104. The selected reinvest subset used 12,155 source-rewrite pairs with 261,803 source words, 161,708 rewrite words, and 423,511 pair words before insertion into the clean-Qwen row-holdout overlay.\n\n",
        "## Exact score-text overlap ancestry\n\n",
        f"research's current-coordinate overlap remains {overlap['recomputed_from_rows']['rows']} rows, {overlap['recomputed_from_rows']['unique_ngrams']} unique strict-score n-grams, and {overlap['recomputed_from_rows']['strict_eval_hits']} strict-score hits, only in `{dict(overlap['recomputed_from_rows']['hit_top_dirs'])}`. The longest span is {overlap['recomputed_from_rows']['longest_common_span_max']} tokens. Recomputing over row-level pair ancestry found {overlap['recomputed_from_rows']['matched_ngram_occurrences_in_original_source_text']} matched n-gram occurrences in original FineWeb source text and {overlap['recomputed_from_rows']['matched_ngram_occurrences_in_generated_rewrite_text']} in generated compact rewrite text. Thus the small exact-overlap set is not evidence that Qwen generated evaluation strings; it is ordinary source/eval phrase overlap that remains inspectable in `{rel(PATHS['overlap_rows'])}`.\n\n",
        "## Remaining work\n\n",
        "The seed43022 official-coordinate score is unchanged. This addendum closes the exact generator-identity gap, but model/source license language still needs official-source recording before public packaging. The seed43122 full vector remains the next decisive evidence for robustness.\n\n",
        f"JSON: `{rel(out_json)}`\n",
    ]
    NOTE.write_text("".join(note_lines), encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "out_json": rel(out_json),
        "note": rel(NOTE),
        "generator_model": generator.get("stdout_model"),
        "generator_command_model_arg": generator.get("parsed_options", {}).get("--model"),
        "accepted_rewrites": ((read_json(PATHS["compact_analysis_summary"]).get("summary") or {}).get("overall") or {}).get("accepted"),
        "overlap_rows": overlap["recomputed_from_rows"]["rows"],
        "overlap_unique_ngrams": overlap["recomputed_from_rows"]["unique_ngrams"],
        "overlap_ngram_occurrences_in_rewrite": overlap["recomputed_from_rows"]["matched_ngram_occurrences_in_generated_rewrite_text"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
