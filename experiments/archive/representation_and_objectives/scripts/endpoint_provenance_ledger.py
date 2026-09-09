#!/usr/bin/env python3
"""research provenance ledger for compact_view_reinvest seed43022 official-coordinate endpoint.

Builds a machine-readable, hash-rich ledger from existing training/evaluation artifacts.
This is endpoint protection, not final packaging. It records data, model, tokenizer,
official evaluation coordinate, GlobalPIQA lineage, and known remaining ledger gaps.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import hashlib
import json
import pathlib
import time
from collections import Counter
from typing import Any

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/representation_and_objectives')
OUT_DIR = _public_path('experiments/archive/representation_and_objectives/data/endpoint_provenance_ledger')
OUT_JSON = _public_path('experiments/archive/representation_and_objectives/data/endpoint_provenance_ledger/compact_reinvest_seed43022_endpoint_provenance_ledger.json')
NOTE = _public_path('research/notes/representation_and_objectives/endpoint_provenance_ledger.md')

A02_DATA = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard')
TRAIN_10M = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl')
TRAIN_100M = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl')
ROW_META = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_changed_block_rows_meta.jsonl')
OVERLAY_META = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/density_cleanqwen_rowholdout_overlay_metadata.json')
PAIR_META = _public_path('experiments/archive/frontier_consolidation/data/density_core_reinvestment_medium_riskhard/selected_compact_reinvest_pairs.jsonl')
PROMPTS = _public_path('experiments/archive/frontier_consolidation/data/medium_density_prompts/fineweb_medium_compact_prompts_all.jsonl')
ACCEPTED = _public_path('experiments/archive/frontier_consolidation/data/medium_compact_analysis/medium_compact_ws_accepted_rewrites.jsonl')
RAW_ROWS = _public_path('experiments/archive/frontier_consolidation/data/medium_compact_analysis/medium_compact_ws_rows.jsonl')
RUN = _public_path('experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022')
HF_MODEL = _public_path('experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/hf_model')
CHECKPOINT_100M = _public_path('experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/hf_model/chck_100M')
TRAIN_COMMAND = _public_path('experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/train_command.json')
ORDER_MANIFEST = _public_path('experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/example_order_manifest.json')
SCI_METRICS = _public_path('experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/scientific_metrics.json')
CONFIG = _public_path('experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/hf_model/config.json')
TOKENIZER = _public_path('experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/hf_model/tokenizer.json')
TOKENIZER_CONFIG = _public_path('experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/hf_model/tokenizer_config.json')
SPECIAL_TOKENS = _public_path('experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/hf_model/special_tokens_map.json')
SUMMARY = _public_path('experiments/archive/representation_and_objectives/data/pristine_collate_seed43022/pristine_collate_seed43022_summary.json')
COLL = _public_path('experiments/archive/representation_and_objectives/data/pristine_collate_seed43022/results/hf_model/all_full_preds_and_fast_scores_mlm.json')
GPIQA = _public_path('experiments/archive/representation_and_objectives/data/globalpiqa_official_lineage/globalpiqa_official_lineage.json')
OVERLAP = _public_path('experiments/archive/representation_and_objectives/data/changed_block_overlap_ancestry_current_official/changed_block_overlap_ancestry_current_official.json')
PRISTINE_STRICT = _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict')

HASH_FILES = {
    "train_10m_pool": TRAIN_10M,
    "train_100m_stream": TRAIN_100M,
    "changed_block_row_meta": ROW_META,
    "overlay_metadata": OVERLAY_META,
    "selected_compact_pairs": PAIR_META,
    "generation_prompts_all": PROMPTS,
    "accepted_rewrites": ACCEPTED,
    "raw_generation_rows": RAW_ROWS,
    "train_command": TRAIN_COMMAND,
    "example_order_manifest": ORDER_MANIFEST,
    "scientific_metrics": SCI_METRICS,
    "model_config": CONFIG,
    "tokenizer_json": TOKENIZER,
    "tokenizer_config": TOKENIZER_CONFIG,
    "special_tokens_map": SPECIAL_TOKENS,
    "checkpoint_100m_safetensors": _public_path('experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/hf_model/chck_100M/model.safetensors'),
    "official_coordinate_summary": SUMMARY,
    "collated_submission_json": COLL,
    "globalpiqa_lineage": GPIQA,
    "overlap_ancestry": OVERLAP,
    "pristine_collate_preds_py": _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_pipeline/collate_preds.py'),
    "pristine_globalpiqa_dl_py": _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_pipeline/global_piqa/dl.py'),
    "pristine_print_results_table_py": _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/scripts/print_results_table.py'),
}


def sha256_file(path: pathlib.Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_entry(path: pathlib.Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
        "sha256": sha256_file(path),
    }


def load_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def count_jsonl_sources(path: pathlib.Path, max_rows: int | None = None) -> dict[str, Any]:
    source_words = Counter()
    source_rows = Counter()
    total_words = 0
    total_rows = 0
    first = None
    last = None
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if first is None:
                first = obj
            last = obj
            total_rows += 1
            words = int(obj.get("words") or len(str(obj.get("text", "")).split()))
            total_words += words
            src = str(obj.get("source", "<missing>"))
            source_words[src] += words
            source_rows[src] += 1
            if max_rows and total_rows >= max_rows:
                break
    return {
        "path": str(path),
        "rows": total_rows,
        "words": total_words,
        "source_words": dict(source_words),
        "source_rows": dict(source_rows),
        "first_row": first,
        "last_row": last,
    }


def count_jsonl(path: pathlib.Path) -> int | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def checkpoint_ladder(root: pathlib.Path) -> dict[str, Any]:
    need = [f"chck_{i}M" for i in range(1, 10)] + [f"chck_{i}M" for i in range(10, 101, 10)]
    present = [x for x in need if (root / x / "model.safetensors").exists()]
    missing = [x for x in need if x not in present]
    all_ckpts = sorted([p.name for p in root.glob("chck_*M") if p.is_dir()], key=lambda s: int(s.split("_")[1][:-1]))
    return {"required_for_strict_small_aoa": need, "present_required": present, "missing_required": missing, "num_all_checkpoint_dirs": len(all_ckpts), "first_10_all": all_ckpts[:10], "last_10_all": all_ckpts[-10:]}


def compact_pair_stats(path: pathlib.Path) -> dict[str, Any]:
    n = 0
    source_words = 0
    rewrite_words = 0
    pair_words = 0
    docs = set()
    sentence_ids = set()
    domains = Counter()
    recalls = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            n += 1
            source_words += int(r.get("source_words") or 0)
            rewrite_words += int(r.get("rewrite_words") or 0)
            pair_words += int(r.get("pair_words") or 0)
            docs.add(str(r.get("doc_id")))
            sentence_ids.add(str(r.get("sentence_id")))
            for d in r.get("domain_hits") or []:
                domains[d] += 1
            if r.get("content_recall") is not None:
                recalls.append(float(r["content_recall"]))
    return {
        "pairs": n,
        "source_words": source_words,
        "rewrite_words": rewrite_words,
        "pair_words": pair_words,
        "unique_doc_ids": len(docs),
        "unique_sentence_ids": len(sentence_ids),
        "domain_hit_counts": dict(domains),
        "mean_content_recall": sum(recalls) / len(recalls) if recalls else None,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_command = load_json(TRAIN_COMMAND)
    order = load_json(ORDER_MANIFEST)
    metrics = load_json(SCI_METRICS)
    model_config = load_json(CONFIG)
    research = load_json(SUMMARY)
    gpiqa = load_json(GPIQA)
    overlap = load_json(OVERLAP)
    overlay = load_json(OVERLAY_META) if OVERLAY_META.exists() else {}

    hashes = {name: file_entry(path) for name, path in HASH_FILES.items()}
    train_10m_counts = count_jsonl_sources(TRAIN_10M)
    # For 100M, exact source accounting is already in order_manifest; count rows/words once as a check.
    train_100m_counts = count_jsonl_sources(TRAIN_100M)

    ledger = {
        "status": "ENDPOINT_PROVENANCE_LEDGER",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Submission-facing provenance ledger for compact_view_reinvest seed43022 official-coordinate endpoint; preserves exact evidence carriers for later review/packaging without modifying results.",
        "endpoint_result": {
            "model_name": "compact_view_reinvest seed43022",
            "official_coordinate_summary": str(SUMMARY),
            "official_path_scores": research.get("score_summary", {}).get("official_overall", {}),
            "collated_json": str(COLL),
            "collated_json_sha256": hashes["collated_submission_json"]["sha256"],
            "visible_leader_reference_overall": 41.8,
            "margin_over_visible_leader": research.get("score_summary", {}).get("official_overall", {}).get("margin_over_visible_leader"),
        },
        "training_recipe": {
            "train_command_json": str(TRAIN_COMMAND),
            "recipe": train_command.get("recipe"),
            "command": train_command.get("command"),
            "cuda_visible_devices_recorded": train_command.get("cuda_visible_devices"),
            "scientific_metrics_summary": {k: metrics.get(k) for k in ["variant", "backend", "model_family", "parameter_count", "vocab_size", "tokenizer_label", "word_exposure", "loss_first", "loss_last", "actual_training_steps", "masking_curriculum", "mask_prob_start", "mask_prob_end", "hidden_size", "n_layer", "n_head", "ffn_mult", "seed", "seq_length", "max_seq_length"]},
            "checkpoint_ladder": checkpoint_ladder(HF_MODEL),
        },
        "model_and_tokenizer": {
            "hf_model_root": str(HF_MODEL),
            "checkpoint_100m": str(CHECKPOINT_100M),
            "model_config": model_config,
            "tokenizer_source_path_recorded": train_command.get("recipe", {}).get("tokenizer_path"),
            "tokenizer_label": train_command.get("recipe", {}).get("tokenizer_label"),
            "tokenizer_vocab_size_config": model_config.get("vocab_size"),
            "hashes": {k: hashes[k] for k in ["model_config", "tokenizer_json", "tokenizer_config", "special_tokens_map", "checkpoint_100m_safetensors"]},
        },
        "data_accounting": {
            "train_10m_pool_counts": train_10m_counts,
            "train_100m_stream_counts": train_100m_counts,
            "example_order_manifest": order,
            "train_command_hash_ok": train_command.get("hash_ok"),
            "train_command_expected_sha256": train_command.get("expected_sha256"),
            "train_command_actual_sha256": train_command.get("actual_sha256"),
            "all_exact_10M_recorded": train_command.get("all_exact_10M"),
            "metadata_status": train_command.get("metadata_status"),
            "compact_pair_stats": compact_pair_stats(PAIR_META),
            "changed_block_row_meta_count": count_jsonl(ROW_META),
            "selected_pair_count": count_jsonl(PAIR_META),
            "prompt_count": count_jsonl(PROMPTS),
            "accepted_rewrite_count": count_jsonl(ACCEPTED),
            "raw_generation_row_count": count_jsonl(RAW_ROWS),
            "overlay_metadata_keys": sorted(overlay.keys()) if isinstance(overlay, dict) else None,
        },
        "evaluation_coordinate": {
            "pristine_strict_root": str(PRISTINE_STRICT),
            "official_eval_commit_recorded_step036": "6f825c291e2c4c78ad33b1935fd64d45f52642dc",
            "strict_eval_dataset_revision_recorded_step036": "8d52da9424a9ff30b9e8266c4f751aba9c504233",
            "ewok_source_revision_recorded_step036": "34d912a608066c92e2990a0328ffc3bd9a716042",
            "globalpiqa_lineage": {
                "summary_path": str(GPIQA),
                "checks": gpiqa.get("checks"),
                "upstream_dataset_revisions": gpiqa.get("upstream_dataset_revisions"),
                "generated_parallel_sha256": gpiqa.get("generated_files", {}).get("parallel", {}).get("full_eval", {}).get("sha256"),
                "generated_nonparallel_sha256": gpiqa.get("generated_files", {}).get("nonparallel", {}).get("full_eval", {}).get("sha256"),
            },
            "code_hashes": {k: hashes[k] for k in ["pristine_collate_preds_py", "pristine_globalpiqa_dl_py", "pristine_print_results_table_py"]},
            "collated_shape": research.get("collated_summary"),
        },
        "teacher_and_generation_trace": {
            "compact_rewrite_prompt_file": str(PROMPTS),
            "accepted_rewrite_file": str(ACCEPTED),
            "raw_generation_file": str(RAW_ROWS),
            "prompt_count": count_jsonl(PROMPTS),
            "accepted_rewrite_count": count_jsonl(ACCEPTED),
            "raw_generation_row_count": count_jsonl(RAW_ROWS),
            "hashes": {k: hashes[k] for k in ["generation_prompts_all", "accepted_rewrites", "raw_generation_rows"]},
            "teacher_identity_status": "Generator identity/model must be read from A02 generation logs or reconstructed from group/source notes before final submission; this ledger records prompt/output artifacts but has not yet located a definitive model-name line.",
            "known_context": "Session memory indicates compact rewrites were generated by a Qwen 3.5/9B-family approved teacher and filtered for faithful compactness, but final ledger still needs exact generator model identifier, runtime, and any license terms from source files/logs.",
        },
        "overlap_and_contamination_protection": {
            "current_official_overlap_ancestry": str(OVERLAP),
            "rows_with_strict_score_overlap": overlap.get("strict_score_overlap_summary", {}).get("rows_with_strict_score_overlap"),
            "unique_strict_score_ngrams": overlap.get("strict_score_overlap_summary", {}).get("unique_strict_score_ngrams"),
            "strict_eval_hit_by_top_dir": overlap.get("strict_score_overlap_summary", {}).get("strict_eval_hit_by_top_dir"),
            "interpretation": overlap.get("interpretation"),
        },
        "file_hashes": hashes,
        "remaining_for_submission_package": [
            "Locate exact compact-rewrite generator model/tool log and source license lineage, not only prompt/output hashes.",
            "Compare the completed seed43122 results on the same official coordinate; if its full vector is weak, analyze seed variance before packaging as robust principle.",
            "A portable path-independent model package remains conditional on the completed evidence.",
            "Coordinate submission ownership with frontier_consolidation to avoid divergent bundles.",
        ],
    }
    OUT_JSON.write_text(json.dumps(ledger, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    scores = ledger["endpoint_result"]["official_path_scores"].get("scores", {}) if isinstance(ledger["endpoint_result"]["official_path_scores"], dict) else {}
    lines = [
        "# research — Endpoint provenance ledger for compact_view_reinvest seed43022",
        "",
        f"Machine-readable ledger: `{OUT_JSON}`",
        "",
        "## Endpoint",
        f"- Official-coordinate Overall: **{ledger['endpoint_result']['official_path_scores'].get('Overall')}**",
        f"- Margin over visible 41.8 leader: **{ledger['endpoint_result']['margin_over_visible_leader']}**",
        f"- Collated JSON sha256: `{ledger['endpoint_result']['collated_json_sha256']}`",
        f"- Columns: `{scores}`",
        "",
        "## Training/data",
        f"- 10M pool: {train_10m_counts['rows']} rows / {train_10m_counts['words']} words, sha `{hashes['train_10m_pool']['sha256']}`",
        f"- 100M stream: {train_100m_counts['rows']} rows / {train_100m_counts['words']} words, sha `{hashes['train_100m_stream']['sha256']}`",
        f"- Source words in 10M pool: `{train_10m_counts['source_words']}`",
        f"- Compact pairs selected: {ledger['data_accounting']['compact_pair_stats']['pairs']} pairs, {ledger['data_accounting']['compact_pair_stats']['pair_words']} pair words, mean content recall {ledger['data_accounting']['compact_pair_stats']['mean_content_recall']:.4f}",
        "",
        "## Model/tokenizer",
        f"- Architecture: DeBERTa-v2 MLM 8x480, {ledger['training_recipe']['scientific_metrics_summary']['parameter_count']} parameters, vocab {ledger['training_recipe']['scientific_metrics_summary']['vocab_size']}",
        f"- Tokenizer hash: `{hashes['tokenizer_json']['sha256']}`",
        f"- chck_100M model hash: `{hashes['checkpoint_100m_safetensors']['sha256']}`",
        f"- Required AoA checkpoint ladder missing: `{ledger['training_recipe']['checkpoint_ladder']['missing_required']}`",
        "",
        "## Evaluation coordinate",
        f"- Pristine strict root: `{PRISTINE_STRICT}`",
        f"- GlobalPIQA lineage checks: `{gpiqa.get('checks')}`",
        f"- Collated shape has null keys: `{research.get('collated_summary', {}).get('null_keys')}`; EWoK total {sum(research.get('collated_summary', {}).get('ewok_lengths', {}).values()) if isinstance(research.get('collated_summary', {}).get('ewok_lengths'), dict) else 'NA'}; AoA row counts {research.get('collated_summary', {}).get('aoa_surprisal_row_count_values')}",
        "",
        "## Remaining",
        "- Exact compact-rewrite generator identity/log and source license lineage still need a focused pass.",
        "- seed43122 full official-coordinate result is still pending.",
    ]
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": ledger["status"],
        "out_json": str(OUT_JSON),
        "note": str(NOTE),
        "overall": ledger["endpoint_result"]["official_path_scores"].get("Overall"),
        "margin": ledger["endpoint_result"].get("margin_over_visible_leader"),
        "train_10m_words": train_10m_counts["words"],
        "train_100m_words": train_100m_counts["words"],
        "required_ladder_missing": ledger["training_recipe"]["checkpoint_ladder"]["missing_required"],
        "globalpiqa_lineage_ok": gpiqa.get("checks", {}).get("all_generated_match_inherited_bytes"),
        "overlap_rows": overlap.get("strict_score_overlap_summary", {}).get("rows_with_strict_score_overlap"),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
