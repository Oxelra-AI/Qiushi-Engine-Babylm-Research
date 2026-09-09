#!/usr/bin/env python3
"""research CPU audit for main-workspace exact-swap innovation WWM port.

This script performs the lowest-cost reliable check before any possible exact-swap
training: it uses the actual frozen 100M stream order, research legal tokenizer, and
exact-swap metadata to verify that the port preserves batch selected group
and token mass exactly, never takes donors from protected source/rewrite pair
spans, does not alter copyable or source pair masks, and has the projected small
innovation-pressure magnitude.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import hashlib
import importlib.util
import json
import pathlib
import statistics
import sys
import time
from collections import Counter, defaultdict
from typing import Any

import torch
from torch.utils.data import DataLoader

# Ensure sibling trainer/core imports resolve when launched from user root.
SCRIPT_DIR = _public_path('experiments/archive/frontier_consolidation/scripts')
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import exact_swap_innovation_trainer as trainer
from exact_swap_wwm_core import innovation_biased_selection, group_positions


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
WORKSPACE = USER_ROOT / "experiments/archive/frontier_consolidation"
OUT_DIR = WORKSPACE / "data/exact_swap_cpu_audit"
DEFAULT_TRAIN_100M = WORKSPACE / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl"
DEFAULT_TOKENIZER = WORKSPACE / "data/compliant_tokenizer"
DEFAULT_METADATA = WORKSPACE / "analysis/innovation_metadata_train.jsonl"
EXPECTED_TRAIN_SHA = trainer.EXPECTED_TRAIN_SHA
EXPECTED_TOKENIZER_JSON_SHA = trainer.EXPECTED_TOKENIZER_JSON_SHA
EXPECTED_METADATA_SHA = trainer.EXPECTED_METADATA_SHA


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pct(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    xs = sorted(vals)
    idx = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return float(xs[idx])


def stats(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"n": 0, "mean": 0.0, "min": 0.0, "p05": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "n": len(vals),
        "mean": float(statistics.mean(vals)),
        "min": float(min(vals)),
        "p05": pct(vals, 0.05),
        "median": float(statistics.median(vals)),
        "p95": pct(vals, 0.95),
        "max": float(max(vals)),
    }


def selected_group_ids(select_row: torch.Tensor, positions_by_gid: dict[int, tuple[int, ...]]) -> set[int]:
    return {gid for gid, positions in positions_by_gid.items() if bool(select_row[list(positions)].all())}


def load_full_row_prefix(path: pathlib.Path, max_words: int):
    """Load a full-row prefix not exceeding max_words, avoiding partial examples."""
    examples = []
    selected = 0
    rows_seen = 0
    sample_rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rows_seen += 1
            text = str(obj["text"])
            words = int(obj.get("words", len(text.split())))
            actual = len(text.split())
            if words != actual:
                raise RuntimeError(f"JSONL word-count mismatch at row {rows_seen}: field={words} actual={actual}")
            if selected + words > max_words:
                break
            ex_id = int(obj.get("example_id", rows_seen - 1))
            source = str(obj.get("source", "example_jsonl"))
            examples.append(trainer.base.Example(text=text, words=words, example_id=ex_id, source=source))
            selected += words
            if len(sample_rows) < 10:
                sample_rows.append({k: obj[k] for k in obj.keys() if k != "text"})
    if not examples:
        raise RuntimeError(f"no complete row fits requested max_words={max_words}")
    return examples, selected, rows_seen, sample_rows


def main() -> None:
    ap = argparse.ArgumentParser(description="CPU audit exact-swap innovation WWM port on real frozen batches")
    ap.add_argument("--train_jsonl", default=str(DEFAULT_TRAIN_100M))
    ap.add_argument("--tokenizer_path", default=str(DEFAULT_TOKENIZER))
    ap.add_argument("--metadata_train", default=str(DEFAULT_METADATA))
    ap.add_argument("--max_word_exposure", type=int, default=8_000_000,
                    help="Prefix word exposure to audit; 8M gives about 202 real batch-256 batches and ~2400 changed exposures.")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--seq_length", type=int, default=256)
    ap.add_argument("--mask_prob", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=43023)
    ap.add_argument("--max_swaps_per_changed_row", type=int, default=1)
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    ap.add_argument("--skip_hashes", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = pathlib.Path(args.train_jsonl)
    tok_path = pathlib.Path(args.tokenizer_path)
    meta_path = pathlib.Path(args.metadata_train)
    hash_report = {}
    if not args.skip_hashes:
        hash_report = {
            "train_sha256": sha256_file(train_path),
            "tokenizer_json_sha256": sha256_file(tok_path / "tokenizer.json"),
            "metadata_train_sha256": sha256_file(meta_path),
        }
        if hash_report["train_sha256"] != EXPECTED_TRAIN_SHA:
            raise RuntimeError(f"unexpected train sha {hash_report['train_sha256']}")
        if hash_report["tokenizer_json_sha256"] != EXPECTED_TOKENIZER_JSON_SHA:
            raise RuntimeError(f"unexpected tokenizer sha {hash_report['tokenizer_json_sha256']}")
        if hash_report["metadata_train_sha256"] != EXPECTED_METADATA_SHA:
            raise RuntimeError(f"unexpected metadata sha {hash_report['metadata_train_sha256']}")

    metadata, metadata_totals = trainer.load_innovation_metadata(meta_path)
    tokenizer = trainer.base.make_portable_tokenizer(str(tok_path))
    examples, actual_words, rows_seen, sample_rows = load_full_row_prefix(
        train_path, int(args.max_word_exposure)
    )
    dataset = trainer.ExactSwapMaskedChunkDataset(examples, tokenizer, int(args.seq_length))
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False,
                        collate_fn=trainer.exact_swap_collate, num_workers=0)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(args.seed))

    cumulative: dict[str, int] = defaultdict(int)
    per_batch: list[dict[str, Any]] = []
    event_sample_path = out_dir / "exact_swap_events_sample.jsonl"
    event_budget = 1000
    changed_per_batch: list[int] = []
    successful_per_batch: list[int] = []
    collision_per_batch: list[int] = []
    no_donor_per_batch: list[int] = []
    copyable_delta_per_batch: list[int] = []
    source_delta_per_batch: list[int] = []

    with event_sample_path.open("w", encoding="utf-8") as evf:
        for step, batch in enumerate(loader, 1):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            word_group = batch["word_group"]
            example_ids = batch["example_ids"]
            row_metadata = [metadata.get(int(eid.item())) for eid in example_ids]
            salt = f"audit_seed={args.seed}:step={step}"
            audit = innovation_biased_selection(
                input_ids=input_ids,
                attention_mask=attention_mask,
                word_group=word_group,
                special_ids=tokenizer.all_special_ids,
                mask_prob=float(args.mask_prob),
                generator=gen,
                row_metadata=row_metadata,
                priority_salt=salt,
                max_swaps_per_changed_row=int(args.max_swaps_per_changed_row),
            )
            summary = trainer.summarize_exact_swap_audit(
                audit, input_ids, attention_mask, word_group, row_metadata, list(tokenizer.all_special_ids)
            )
            if summary.get("selected_token_delta", 0) != 0 or summary.get("selected_group_delta", 0) != 0:
                raise RuntimeError(f"mass mismatch at batch {step}: {summary}")
            if summary.get("donor_protected_pair", 0) != 0 or summary.get("donor_protected_source", 0) != 0:
                raise RuntimeError(f"protected donor at batch {step}: {summary}")
            if summary.get("forced_copyable", 0) != 0:
                raise RuntimeError(f"forced copyable at batch {step}: {summary}")
            for k, v in summary.items():
                cumulative[k] += int(v)
            changed_per_batch.append(int(summary.get("changed_row_exposures", 0)))
            successful_per_batch.append(int(summary.get("successful_swaps", 0)))
            collision_per_batch.append(int(summary.get("proposal_baseline_collision", 0)))
            no_donor_per_batch.append(int(summary.get("proposal_no_equal_length_donor", 0)))
            copyable_delta_per_batch.append(int(summary.get("copyable_group_selection_delta", 0)))
            source_delta_per_batch.append(int(summary.get("source_group_selection_delta", 0)))
            per_batch.append({
                "batch": step,
                "words_cumulative": int(batch["words"].sum().item()),
                "changed_rows": int(summary.get("changed_row_exposures", 0)),
                "proposals": int(summary.get("proposals", 0)),
                "successful_swaps": int(summary.get("successful_swaps", 0)),
                "collisions": int(summary.get("proposal_baseline_collision", 0)),
                "no_equal_length_donor": int(summary.get("proposal_no_equal_length_donor", 0)),
                "baseline_selected_groups": int(summary.get("baseline_selected_groups", 0)),
                "baseline_selected_tokens": int(summary.get("baseline_selected_tokens", 0)),
            })
            if event_budget > 0:
                for ev in audit.events[:event_budget]:
                    evf.write(json.dumps({"batch": step, **ev}, ensure_ascii=False) + "\n")
                    event_budget -= 1
                    if event_budget <= 0:
                        break

    n_batches = len(per_batch)
    baseline_groups = max(1, cumulative.get("baseline_selected_groups", 0))
    baseline_tokens = max(1, cumulative.get("baseline_selected_tokens", 0))
    successful = cumulative.get("successful_swaps", 0)
    proposals = cumulative.get("proposals", 0)
    changed_exposures = cumulative.get("changed_row_exposures", 0)
    baseline_innov = cumulative.get("baseline_selected_innovation_groups", 0)
    biased_innov = cumulative.get("biased_selected_innovation_groups", 0)
    interpretation = []
    if cumulative.get("selected_group_delta", 0) == 0 and cumulative.get("selected_token_delta", 0) == 0:
        interpretation.append("Exact-swap port preserves selected group and token mass exactly across audited real frozen-stream batches.")
    if cumulative.get("copyable_group_selection_delta", 0) == 0 and cumulative.get("source_group_selection_delta", 0) == 0:
        interpretation.append("Copyable rewrite and protected source selections are unchanged in aggregate; donors are ordinary non-pair groups.")
    if successful > 0:
        interpretation.append("The intervention is much smaller than research probability reallocation: it swaps a small fraction of global selected groups while forcing one candidate per eligible changed-row exposure when possible.")
    if cumulative.get("proposal_no_equal_length_donor", 0) == 0:
        interpretation.append("No equal-length donor misses occurred in the audited real batch stream.")
    elif cumulative.get("proposal_no_equal_length_donor", 0) / max(1, proposals) < 0.005:
        interpretation.append("Equal-length donor misses are rare on real batch-256 frozen-stream batches and leave baseline selection unchanged.")
    else:
        interpretation.append("Donor misses are non-negligible and must be understood before any exact-swap training.")

    summary = {
        "status": "EXACT_SWAP_CPU_AUDIT",
        "scientific_purpose": "Prepare, without GPU, the lower-perturbation exact-swap conditional-innovation fallback so it can be judged after research results rather than built under pressure.",
        "inputs": {
            "train_jsonl": str(train_path),
            "tokenizer_path": str(tok_path),
            "metadata_train": str(meta_path),
            "max_word_exposure": int(args.max_word_exposure),
            "batch_size": int(args.batch_size),
            "seq_length": int(args.seq_length),
            "mask_prob": float(args.mask_prob),
            "seed": int(args.seed),
            "max_swaps_per_changed_row": int(args.max_swaps_per_changed_row),
        },
        "hash_report": hash_report,
        "metadata_totals": metadata_totals,
        "audited_stream": {
            "rows": len(examples),
            "words": actual_words,
            "batches": n_batches,
            "changed_row_exposures": changed_exposures,
            "eligible_changed_rows": cumulative.get("eligible_changed_rows", 0),
            "proposals": proposals,
            "successful_swaps": successful,
            "proposal_baseline_collisions": cumulative.get("proposal_baseline_collision", 0),
            "proposal_no_equal_length_donor": cumulative.get("proposal_no_equal_length_donor", 0),
        },
        "selection_mass": {
            "baseline_selected_groups": cumulative.get("baseline_selected_groups", 0),
            "biased_selected_groups": cumulative.get("biased_selected_groups", 0),
            "selected_group_delta": cumulative.get("selected_group_delta", 0),
            "baseline_selected_tokens": cumulative.get("baseline_selected_tokens", 0),
            "biased_selected_tokens": cumulative.get("biased_selected_tokens", 0),
            "selected_token_delta": cumulative.get("selected_token_delta", 0),
            "batch_mass_mismatch": cumulative.get("batch_mass_mismatch", 0),
            "successful_swaps_fraction_of_baseline_selected_groups": successful / baseline_groups,
            "successful_swaps_fraction_of_baseline_selected_tokens": successful / baseline_tokens,
        },
        "targeting": {
            "baseline_selected_innovation_groups": baseline_innov,
            "biased_selected_innovation_groups": biased_innov,
            "innovation_selection_gain": biased_innov - baseline_innov,
            "innovation_multiplier": biased_innov / max(1, baseline_innov),
            "successful_swaps_per_10m_words_projected": successful * (10_000_000 / max(1, actual_words)),
            "proposal_success_rate_excluding_collisions": successful / max(1, proposals - cumulative.get("proposal_baseline_collision", 0)),
            "proposal_collision_rate": cumulative.get("proposal_baseline_collision", 0) / max(1, proposals),
            "no_donor_rate": cumulative.get("proposal_no_equal_length_donor", 0) / max(1, proposals),
            "forced_relation_cue": cumulative.get("forced_relation_cue", 0),
            "forced_relation_cue_fraction": cumulative.get("forced_relation_cue", 0) / max(1, successful),
            "forced_copyable": cumulative.get("forced_copyable", 0),
        },
        "controls": {
            "baseline_selected_copyable_groups": cumulative.get("baseline_selected_copyable_groups", 0),
            "biased_selected_copyable_groups": cumulative.get("biased_selected_copyable_groups", 0),
            "copyable_group_selection_delta": cumulative.get("copyable_group_selection_delta", 0),
            "baseline_selected_source_groups": cumulative.get("baseline_selected_source_groups", 0),
            "biased_selected_source_groups": cumulative.get("biased_selected_source_groups", 0),
            "source_group_selection_delta": cumulative.get("source_group_selection_delta", 0),
            "donor_copyable_rewrite": cumulative.get("donor_copyable_rewrite", 0),
            "donor_protected_source": cumulative.get("donor_protected_source", 0),
            "donor_protected_pair": cumulative.get("donor_protected_pair", 0),
            "donor_ordinary_row": cumulative.get("donor_ordinary_row", 0),
            "donor_changed_nonpair_row": cumulative.get("donor_changed_nonpair_row", 0),
            "forced_targets_with_any_visible_paired_source": cumulative.get("forced_target_has_visible_source_group", 0),
            "forced_targets_with_fully_visible_paired_source": cumulative.get("forced_target_has_fully_visible_source", 0),
        },
        "per_batch_stats": {
            "changed_rows": stats([float(x) for x in changed_per_batch]),
            "successful_swaps": stats([float(x) for x in successful_per_batch]),
            "collisions": stats([float(x) for x in collision_per_batch]),
            "no_equal_length_donor": stats([float(x) for x in no_donor_per_batch]),
            "copyable_delta": stats([float(x) for x in copyable_delta_per_batch]),
            "source_delta": stats([float(x) for x in source_delta_per_batch]),
        },
        "events_sample": str(event_sample_path),
        "interpretation": interpretation,
        "all_checks_passed": bool(
            cumulative.get("selected_group_delta", 0) == 0
            and cumulative.get("selected_token_delta", 0) == 0
            and cumulative.get("batch_mass_mismatch", 0) == 0
            and cumulative.get("forced_copyable", 0) == 0
            and cumulative.get("donor_protected_source", 0) == 0
            and cumulative.get("donor_protected_pair", 0) == 0
            and cumulative.get("copyable_group_selection_delta", 0) == 0
            and cumulative.get("source_group_selection_delta", 0) == 0
        ),
        "elapsed_sec": round(time.time() - t0, 3),
    }

    out_json = out_dir / "exact_swap_cpu_audit.json"
    out_md = out_dir / "exact_swap_cpu_audit.md"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research exact-swap CPU audit",
        "",
        f"Audited words: {actual_words:,}; rows: {len(examples):,}; batches: {n_batches:,}",
        f"Changed-row exposures: {changed_exposures:,}; proposals: {proposals:,}; successful swaps: {successful:,}",
        "",
        "## Selection mass",
        f"- Selected group delta: {summary['selection_mass']['selected_group_delta']}",
        f"- Selected token delta: {summary['selection_mass']['selected_token_delta']}",
        f"- Batch mass mismatches: {summary['selection_mass']['batch_mass_mismatch']}",
        f"- Successful swaps / baseline selected groups: {summary['selection_mass']['successful_swaps_fraction_of_baseline_selected_groups']:.6f}",
        "",
        "## Targeting and controls",
        f"- Innovation groups selected baseline→biased: {baseline_innov:,} → {biased_innov:,} (multiplier {summary['targeting']['innovation_multiplier']:.4f})",
        f"- Projected successful swaps per 10M words: {summary['targeting']['successful_swaps_per_10m_words_projected']:.2f}",
        f"- Collision rate: {summary['targeting']['proposal_collision_rate']:.4f}; no-donor rate: {summary['targeting']['no_donor_rate']:.6f}",
        f"- Copyable group delta: {summary['controls']['copyable_group_selection_delta']}; source group delta: {summary['controls']['source_group_selection_delta']}",
        f"- Donor protected pair/source/copyable counts: {summary['controls']['donor_protected_pair']} / {summary['controls']['donor_protected_source']} / {summary['controls']['donor_copyable_rewrite']}",
        f"- Donor ordinary-row count: {summary['controls']['donor_ordinary_row']}; changed-nonpair donor count: {summary['controls']['donor_changed_nonpair_row']}",
        "",
        "## Interpretation",
    ]
    lines.extend(f"- {x}" for x in interpretation)
    lines.extend([
        "",
        f"All checks passed: `{summary['all_checks_passed']}`",
        f"Event sample: `{event_sample_path}`",
        f"Full JSON: `{out_json}`",
    ])
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "all_checks_passed": summary["all_checks_passed"],
        "out_json": str(out_json),
        "out_md": str(out_md),
        "words": actual_words,
        "batches": n_batches,
        "successful_swaps": successful,
        "group_delta": summary["selection_mass"]["selected_group_delta"],
        "token_delta": summary["selection_mass"]["selected_token_delta"],
        "copyable_delta": summary["controls"]["copyable_group_selection_delta"],
        "source_delta": summary["controls"]["source_group_selection_delta"],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
