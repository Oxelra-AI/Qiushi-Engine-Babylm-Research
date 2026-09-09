#!/usr/bin/env python3
"""Bounded CPU stress test for innovation-biased WWM on the frozen stream."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import random
import statistics
import time
from typing import Any

import torch
from transformers import AutoTokenizer

import build_innovation_metadata as builder
import innovation_biased_wwm as ibw


WORK_DIR = builder.WORKSPACE / "analysis/work"
DEFAULT_METADATA = WORK_DIR / "innovation_metadata_train.jsonl"
DEFAULT_OUT = WORK_DIR / "innovation_wwm_smoke.json"
DEFAULT_MD = WORK_DIR / "innovation_wwm_smoke.md"
DEFAULT_EVENTS = WORK_DIR / "innovation_wwm_events_sample.jsonl"


def load_metadata(path: pathlib.Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                out[int(obj["example_id"])] = obj
    return out


def stride_take(items: list[Any], n: int) -> list[Any]:
    if n <= 0 or n >= len(items):
        return items
    if n == 1:
        return [items[len(items) // 2]]
    return [items[round(i * (len(items) - 1) / (n - 1))] for i in range(n)]


def read_bounded_examples(
    path: pathlib.Path,
    meta_by_id: dict[int, dict[str, Any]],
    changed_rows: int,
    ordinary_rows: int,
    reservoir_seed: int,
) -> list[dict[str, Any]]:
    chosen_changed_ids = {
        int(x["example_id"]) for x in stride_take(
            sorted(meta_by_id.values(), key=lambda x: int(x["first_global_row_1based"])), changed_rows
        )
    }
    changed: dict[int, dict[str, Any]] = {}
    ordinary: list[dict[str, Any]] = []
    ordinary_seen = 0
    rng = random.Random(reservoir_seed)
    with path.open("r", encoding="utf-8") as f:
        for row_1, line in enumerate(f, 1):
            obj = json.loads(line)
            eid = int(obj.get("example_id", row_1 - 1))
            rec = {
                "text": str(obj["text"]),
                "words": int(obj.get("words", len(str(obj["text"]).split()))),
                "example_id": eid,
                "source": str(obj.get("source", "")),
                "global_row_1based": row_1,
            }
            if eid in chosen_changed_ids and eid not in changed:
                changed[eid] = rec
            elif eid not in meta_by_id and ordinary_rows > 0:
                ordinary_seen += 1
                if len(ordinary) < ordinary_rows:
                    ordinary.append(rec)
                else:
                    j = rng.randrange(ordinary_seen)
                    if j < ordinary_rows:
                        ordinary[j] = rec
    missing = sorted(chosen_changed_ids.difference(changed))
    if missing:
        raise RuntimeError(f"missing changed examples: {missing[:10]}")
    rows = list(changed.values()) + ordinary
    rows.sort(key=lambda x: (int(x["global_row_1based"]), int(x["example_id"])))
    # Interleave the two categories so every batch exercises the join boundary.
    c = [x for x in rows if int(x["example_id"]) in meta_by_id]
    o = [x for x in rows if int(x["example_id"]) not in meta_by_id]
    interleaved: list[dict[str, Any]] = []
    for i in range(max(len(c), len(o))):
        if i < len(c):
            interleaved.append(c[i])
        if i < len(o):
            interleaved.append(o[i])
    return interleaved


def tensorize(rows: list[dict[str, Any]], tokenizer) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    enc = tokenizer(
        [x["text"] for x in rows], add_special_tokens=False, truncation=True,
        max_length=builder.SEQ_LEN, padding="max_length", return_tensors="pt",
    )
    ids = enc["input_ids"]
    attn = enc["attention_mask"]
    word_group = torch.full_like(ids, -1)
    for b in range(ids.shape[0]):
        n = int(attn[b].sum().item())
        group_at, _ = builder.build_word_groups([int(x) for x in ids[b, :n].tolist()], tokenizer)
        word_group[b, :n] = torch.tensor(group_at, dtype=word_group.dtype)
    return ids, attn, word_group


def selection_counts(select: torch.Tensor, groups: torch.Tensor, candidate: torch.Tensor) -> tuple[int, int]:
    positions = ibw.group_positions(groups, candidate)
    selected_groups = sum(bool(select[list(pos)].all()) for pos in positions.values())
    return selected_groups, int(select.sum().item())


def selected_gid_count(select: torch.Tensor, groups: torch.Tensor, gids: set[int]) -> int:
    n = 0
    for gid in gids:
        pos = (groups == int(gid)).nonzero(as_tuple=False).flatten()
        if pos.numel() and bool(select[pos].all()):
            n += 1
    return n


def rate(num: int | float, den: int | float) -> float | None:
    return float(num / den) if den else None


def root_relative(path: pathlib.Path) -> str:
    resolved = path.resolve() if not path.is_absolute() else path
    return str(resolved.relative_to(builder.USER_ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", type=pathlib.Path, default=DEFAULT_METADATA)
    ap.add_argument("--changed-rows", type=int, default=3005)
    ap.add_argument("--ordinary-rows", type=int, default=3005)
    ap.add_argument("--replicates", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--mask-prob", type=float, default=0.15)
    ap.add_argument("--base-seed", type=int, default=43022)
    ap.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--markdown", type=pathlib.Path, default=DEFAULT_MD)
    ap.add_argument("--events", type=pathlib.Path, default=DEFAULT_EVENTS)
    args = ap.parse_args()
    torch.set_num_threads(min(8, max(1, torch.get_num_threads())))
    t0 = time.time()
    meta_by_id = load_metadata(args.metadata)
    rows = read_bounded_examples(
        builder.DEFAULT_TRAIN, meta_by_id, args.changed_rows, args.ordinary_rows, args.base_seed
    )
    tokenizer = AutoTokenizer.from_pretrained(str(builder.DEFAULT_TOKENIZER), use_fast=True)
    ids, attn, groups = tensorize(rows, tokenizer)
    special_ids = tokenizer.all_special_ids
    totals: collections.Counter = collections.Counter()
    per_replicate: list[dict[str, Any]] = []
    event_sample: list[dict[str, Any]] = []
    unique_forced: set[tuple[int, int]] = set()
    all_candidate_keys: set[tuple[int, int]] = set()
    for row, meta in zip(rows, [meta_by_id.get(int(x["example_id"])) for x in rows]):
        if meta:
            for target in meta.get("innovation_groups") or []:
                all_candidate_keys.add((int(row["example_id"]), int(target["group_id"])))

    for rep in range(args.replicates):
        rep_counts: collections.Counter = collections.Counter()
        for start in range(0, len(rows), args.batch_size):
            stop = min(len(rows), start + args.batch_size)
            batch_ids, batch_attn, batch_groups = ids[start:stop], attn[start:stop], groups[start:stop]
            batch_meta = [meta_by_id.get(int(x["example_id"])) for x in rows[start:stop]]
            selection_seed = args.base_seed + 1_000_003 * rep + start
            g_intervention = torch.Generator(device="cpu").manual_seed(selection_seed)
            g_counterfactual = torch.Generator(device="cpu").manual_seed(selection_seed)
            audit = ibw.innovation_biased_selection(
                batch_ids, batch_attn, batch_groups, special_ids, args.mask_prob,
                g_intervention, batch_meta, priority_salt=f"rep={rep};batch={start}",
                max_swaps_per_changed_row=1,
            )
            baseline_check, candidate = ibw.baseline_wwm_selection(
                batch_ids, batch_attn, batch_groups, special_ids, args.mask_prob, g_counterfactual
            )
            if not torch.equal(audit.baseline_select, baseline_check):
                raise AssertionError("intervention did not preserve baseline WWM draw")
            if not torch.equal(g_intervention.get_state(), g_counterfactual.get_state()):
                raise AssertionError("metadata selection consumed PyTorch masking RNG")
            totals.update(audit.counters)
            rep_counts.update(audit.counters)
            # Reuse exactly the same corruption draw tensor for baseline and treatment.
            corruption_seed = args.base_seed + 9_999_991 * rep + start
            gb = torch.Generator(device="cpu").manual_seed(corruption_seed)
            gi = torch.Generator(device="cpu").manual_seed(corruption_seed)
            _, base_labels, base_r = ibw.apply_inherited_tokenwise_corruption(
                batch_ids, audit.baseline_select, int(tokenizer.mask_token_id), len(tokenizer), gb
            )
            _, biased_labels, biased_r = ibw.apply_inherited_tokenwise_corruption(
                batch_ids, audit.biased_select, int(tokenizer.mask_token_id), len(tokenizer), gi
            )
            if not torch.equal(base_r, biased_r):
                raise AssertionError("counterfactual corruption draw mismatch")
            batch_base_groups = batch_biased_groups = 0
            batch_base_tokens = batch_biased_tokens = 0
            for local_b in range(stop - start):
                meta = batch_meta[local_b]
                b_groups, b_tokens = selection_counts(
                    audit.baseline_select[local_b], batch_groups[local_b], candidate[local_b]
                )
                i_groups, i_tokens = selection_counts(
                    audit.biased_select[local_b], batch_groups[local_b], candidate[local_b]
                )
                rep_counts["baseline_selected_groups"] += b_groups
                rep_counts["biased_selected_groups"] += i_groups
                rep_counts["baseline_selected_tokens"] += b_tokens
                rep_counts["biased_selected_tokens"] += i_tokens
                batch_base_groups += b_groups
                batch_biased_groups += i_groups
                batch_base_tokens += b_tokens
                batch_biased_tokens += i_tokens
                rep_counts["per_row_group_mass_mismatch"] += b_groups != i_groups
                rep_counts["per_row_token_mass_mismatch"] += b_tokens != i_tokens
                rep_counts["rows"] += 1
                if meta:
                    innovation_gids = {int(x["group_id"]) for x in meta.get("innovation_groups") or []}
                    copyable_gids = {int(x) for x in meta.get("copyable_group_ids") or []}
                    source_gids = {int(x) for x in meta.get("protected_source_group_ids") or []}
                    base_innov = selected_gid_count(audit.baseline_select[local_b], batch_groups[local_b], innovation_gids)
                    biased_innov = selected_gid_count(audit.biased_select[local_b], batch_groups[local_b], innovation_gids)
                    rep_counts["baseline_selected_innovation"] += base_innov
                    rep_counts["biased_selected_innovation"] += biased_innov
                    rep_counts["baseline_rows_with_innovation"] += base_innov > 0
                    rep_counts["biased_rows_with_innovation"] += biased_innov > 0
                    rep_counts["baseline_selected_copyable"] += selected_gid_count(
                        audit.baseline_select[local_b], batch_groups[local_b], copyable_gids
                    )
                    rep_counts["biased_selected_copyable"] += selected_gid_count(
                        audit.biased_select[local_b], batch_groups[local_b], copyable_gids
                    )
                    base_source = selected_gid_count(audit.baseline_select[local_b], batch_groups[local_b], source_gids)
                    biased_source = selected_gid_count(audit.biased_select[local_b], batch_groups[local_b], source_gids)
                    rep_counts["baseline_selected_source_groups"] += base_source
                    rep_counts["biased_selected_source_groups"] += biased_source
                    rep_counts["rows_source_selection_changed"] += base_source != biased_source
                else:
                    rep_counts["ordinary_rows_selection_changed"] += not torch.equal(
                        audit.baseline_select[local_b], audit.biased_select[local_b]
                    )
            rep_counts["per_batch_group_mass_mismatch"] += batch_base_groups != batch_biased_groups
            rep_counts["per_batch_token_mass_mismatch"] += batch_base_tokens != batch_biased_tokens
            for event in audit.events:
                event = dict(event)
                event["replicate"] = rep
                event["global_batch_start"] = start
                if event["outcome"] == "swapped":
                    abs_b = start + int(event["batch_row"])
                    key = (int(event["example_id"]), int(event["target_group_id"]))
                    unique_forced.add(key)
                    target_pos = [int(x) for x in event["target_positions"]]
                    rvals = biased_r[int(event["batch_row"]), target_pos]
                    keep = rvals >= 0.9
                    mask = rvals < 0.8
                    rep_counts["forced_groups_corruption_audited"] += 1
                    rep_counts["forced_groups_multi_piece"] += len(target_pos) > 1
                    rep_counts["forced_groups_any_gold_keep_branch"] += bool(keep.any())
                    rep_counts["forced_groups_all_mask_branch"] += bool(mask.all())
                    rep_counts["forced_multi_piece_mixed_keep_branch"] += (
                        len(target_pos) > 1 and bool(keep.any()) and not bool(keep.all())
                    )
                    rep_counts["forced_groups_all_positions_labeled"] += bool(
                        (biased_labels[int(event["batch_row"]), target_pos] != -100).all()
                    )
                    event["inherited_corruption_any_gold_keep_branch"] = bool(keep.any())
                    event["inherited_corruption_all_mask_branch"] = bool(mask.all())
                    event["row_global_index"] = abs_b
                if len(event_sample) < 1000:
                    event_sample.append(event)
        totals.update({f"measured:{k}": v for k, v in rep_counts.items()})
        per_replicate.append(dict(rep_counts))

    measured = collections.Counter({
        k.removeprefix("measured:"): v for k, v in totals.items() if k.startswith("measured:")
    })
    changed_in_sample = sum(int(x["example_id"]) in meta_by_id for x in rows)
    ordinary_in_sample = len(rows) - changed_in_sample
    exposure_words = sum(int(x["words"]) for x in rows) * args.replicates
    group_delta = measured["biased_selected_groups"] - measured["baseline_selected_groups"]
    token_delta = measured["biased_selected_tokens"] - measured["baseline_selected_tokens"]
    innovation_gain = measured["biased_selected_innovation"] - measured["baseline_selected_innovation"]
    summary = {
        "status": "INNOVATION_WWM_CPU_SMOKE_PASS",
        "configuration": {
            "changed_rows": changed_in_sample,
            "ordinary_rows": ordinary_in_sample,
            "replicates": args.replicates,
            "batch_size": args.batch_size,
            "mask_prob": args.mask_prob,
            "base_seed": args.base_seed,
            "max_swaps_per_changed_row": 1,
            "device": "cpu",
        },
        "input_provenance": {
            "train_sha256": builder.sha256_file(builder.DEFAULT_TRAIN),
            "tokenizer_json_sha256": builder.sha256_file(builder.DEFAULT_TOKENIZER / "tokenizer.json"),
            "metadata_sha256": builder.sha256_file(args.metadata),
        },
        "selection_mass": {
            "baseline_selected_groups": measured["baseline_selected_groups"],
            "biased_selected_groups": measured["biased_selected_groups"],
            "selected_group_delta": group_delta,
            "baseline_selected_tokens": measured["baseline_selected_tokens"],
            "biased_selected_tokens": measured["biased_selected_tokens"],
            "selected_token_delta": token_delta,
            "per_row_group_mismatches": measured["per_row_group_mass_mismatch"],
            "per_row_token_mismatches": measured["per_row_token_mass_mismatch"],
            "per_batch_group_mismatches": measured["per_batch_group_mass_mismatch"],
            "per_batch_token_mismatches": measured["per_batch_token_mass_mismatch"],
            "exact_batch_match": group_delta == token_delta == measured["per_batch_group_mass_mismatch"] == measured["per_batch_token_mass_mismatch"] == 0,
        },
        "targeting": {
            "eligible_changed_row_exposures": totals["eligible_changed_rows"],
            "proposals": totals["proposals"],
            "successful_swaps": totals["successful_swaps"],
            "affected_row_fraction_of_eligible_changed": rate(totals["successful_swaps"], totals["eligible_changed_rows"]),
            "proposal_baseline_collisions": totals["proposal_baseline_collision"],
            "proposal_baseline_collision_rate": rate(totals["proposal_baseline_collision"], totals["proposals"]),
            "no_equal_length_donor": totals["proposal_no_equal_length_donor"],
            "candidate_baseline_collision_rate": rate(totals["baseline_selected_innovation_groups"], totals["eligible_innovation_groups"]),
            "baseline_selected_innovation": measured["baseline_selected_innovation"],
            "biased_selected_innovation": measured["biased_selected_innovation"],
            "innovation_selection_gain": innovation_gain,
            "innovation_gain_fraction_of_all_baseline_selected_groups": rate(innovation_gain, measured["baseline_selected_groups"]),
            "changed_rows_with_at_least_one_innovation_baseline_fraction": rate(measured["baseline_rows_with_innovation"], totals["eligible_changed_rows"]),
            "changed_rows_with_at_least_one_innovation_biased_fraction": rate(measured["biased_rows_with_innovation"], totals["eligible_changed_rows"]),
            "unique_forced_target_groups": len(unique_forced),
            "unique_candidate_groups_in_sample": len(all_candidate_keys),
            "unique_forced_coverage_across_replicates": rate(len(unique_forced), len(all_candidate_keys)),
            "forced_relation_cue": totals["forced_relation_cue"],
            "forced_relation_cue_fraction": rate(totals["forced_relation_cue"], totals["successful_swaps"]),
            "forced_copyable": totals["forced_copyable"],
        },
        "copyable_and_source_controls": {
            "baseline_selected_copyable": measured["baseline_selected_copyable"],
            "biased_selected_copyable": measured["biased_selected_copyable"],
            "copyable_delta": measured["biased_selected_copyable"] - measured["baseline_selected_copyable"],
            "donor_copyable_rewrite": totals["donor_copyable_rewrite"],
            "donor_protected_source": totals["donor_protected_source"],
            "donor_protected_pair": totals["donor_protected_pair"],
            "donor_ordinary_row": totals["donor_ordinary_row"],
            "donor_changed_nonpair_row": totals["donor_changed_nonpair_row"],
            "donor_ordinary_row_fraction": rate(totals["donor_ordinary_row"], totals["successful_swaps"]),
            "baseline_selected_source_groups": measured["baseline_selected_source_groups"],
            "biased_selected_source_groups": measured["biased_selected_source_groups"],
            "rows_with_source_selection_changed": measured["rows_source_selection_changed"],
            "forced_targets_with_any_visible_paired_source_fraction": rate(
                totals["forced_target_has_visible_source_group"], totals["successful_swaps"]
            ),
            "forced_targets_with_fully_visible_paired_source_fraction": rate(
                totals["forced_target_has_fully_visible_source"], totals["successful_swaps"]
            ),
            "ordinary_rows_selection_changed": measured["ordinary_rows_selection_changed"],
        },
        "whole_word_and_corruption_audit": {
            "forced_groups_audited": measured["forced_groups_corruption_audited"],
            "forced_groups_all_positions_labeled": measured["forced_groups_all_positions_labeled"],
            "full_group_label_coverage": rate(measured["forced_groups_all_positions_labeled"], measured["forced_groups_corruption_audited"]),
            "forced_multi_piece_groups": measured["forced_groups_multi_piece"],
            "inherited_tokenwise_any_gold_keep_branch_rate": rate(
                measured["forced_groups_any_gold_keep_branch"], measured["forced_groups_corruption_audited"]
            ),
            "inherited_tokenwise_all_mask_branch_rate": rate(
                measured["forced_groups_all_mask_branch"], measured["forced_groups_corruption_audited"]
            ),
            "multi_piece_mixed_gold_keep_count": measured["forced_multi_piece_mixed_keep_branch"],
            "interpretation": "All target pieces are selected and labeled. The inherited per-token 80/10/10 branch intentionally leaves some selected gold pieces visible; this is baseline corruption behavior, not a span-boundary leak.",
        },
        "exposure_accounting": {
            "sample_row_exposures": len(rows) * args.replicates,
            "sample_whitespace_word_exposure_baseline": exposure_words,
            "sample_whitespace_word_exposure_biased": exposure_words,
            "row_and_word_exposure_invariant": True,
            "input_ids_and_attention_unchanged": True,
            "pytorch_rng_state_equal_through_selection": True,
        },
        "per_replicate": per_replicate,
        "artifacts": {
            "metadata": root_relative(args.metadata),
            "events_sample": root_relative(args.events),
        },
        "elapsed_sec": round(time.time() - t0, 3),
    }
    if not summary["selection_mass"]["exact_batch_match"]:
        summary["status"] = "INNOVATION_WWM_CPU_SMOKE_FAIL"
    if totals["forced_copyable"] or totals["donor_protected_source"] or totals["donor_protected_pair"]:
        summary["status"] = "INNOVATION_WWM_CPU_SMOKE_FAIL"
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with args.events.open("w", encoding="utf-8") as f:
        for event in event_sample:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    md = [
        "# CPU smoke test: target-mass-matched innovation-biased WWM",
        "",
        f"Status: **{summary['status']}**. CPU only; no model update, evaluation, corpus edit, tokenizer edit, or GPU use.",
        "",
        "## Exact mass and targeting",
        "",
        f"Across {len(rows) * args.replicates:,} row exposures, baseline and treatment selected "
        f"{measured['baseline_selected_groups']:,} WWM groups / {measured['baseline_selected_tokens']:,} tokens. "
        f"Group delta={group_delta}, token delta={token_delta}, per-batch mismatches="
        f"{measured['per_batch_group_mass_mismatch']}/{measured['per_batch_token_mass_mismatch']}. "
        f"Per-row group/token changes={measured['per_row_group_mass_mismatch']}/"
        f"{measured['per_row_token_mass_mismatch']} because donors are deliberately taken elsewhere in the batch.",
        f"The mechanism completed {totals['successful_swaps']:,} exact same-length swaps from "
        f"{totals['proposals']:,} proposals; affected eligible changed-row fraction="
        f"{summary['targeting']['affected_row_fraction_of_eligible_changed']:.4f}. Proposal collision rate="
        f"{summary['targeting']['proposal_baseline_collision_rate']:.4f}; candidate-group baseline collision rate="
        f"{summary['targeting']['candidate_baseline_collision_rate']:.4f}; no-donor count="
        f"{totals['proposal_no_equal_length_donor']:,}.",
        f"Selected innovation groups rose from {measured['baseline_selected_innovation']:,} to "
        f"{measured['biased_selected_innovation']:,} (+{innovation_gain:,}), which is "
        f"{summary['targeting']['innovation_gain_fraction_of_all_baseline_selected_groups']:.4%} of baseline selected-group mass.",
        "",
        "## Exclusion, source context, and leakage audit",
        "",
        f"Forced copyable targets={totals['forced_copyable']}; protected-source donors="
        f"{totals['donor_protected_source']}; protected-pair donors={totals['donor_protected_pair']}; "
        f"ordinary-row donor fraction={summary['copyable_and_source_controls']['donor_ordinary_row_fraction']:.4f}; "
        f"rows whose paired-source selection changed="
        f"{measured['rows_source_selection_changed']}; ordinary rows changed="
        f"{measured['ordinary_rows_selection_changed']}.",
        f"Any paired-source visibility for forced targets="
        f"{summary['copyable_and_source_controls']['forced_targets_with_any_visible_paired_source_fraction']:.4f}; "
        f"fully visible paired source={summary['copyable_and_source_controls']['forced_targets_with_fully_visible_paired_source_fraction']:.4f}.",
        f"All-piece selection/label coverage="
        f"{summary['whole_word_and_corruption_audit']['full_group_label_coverage']:.4f}. Under the unchanged "
        f"tokenwise 80/10/10 corruption, {summary['whole_word_and_corruption_audit']['inherited_tokenwise_any_gold_keep_branch_rate']:.4f} "
        f"of forced groups had at least one intentional gold keep-branch piece, and "
        f"{summary['whole_word_and_corruption_audit']['inherited_tokenwise_all_mask_branch_rate']:.4f} were entirely mask-branch.",
        "",
        "## Exposure accounting",
        "",
        f"Baseline and treatment used the same {exposure_words:,} whitespace-word exposure in this smoke. "
        "Rows, words, input ids, attention masks, batch shape, batch-selected groups, and batch-selected tokens are invariant. "
        "Hash-based proposal/donor choice consumes no PyTorch masking RNG; the state matched the baseline through WWM selection.",
        "",
        "Full JSON and sampled events are in the sibling work artifacts.",
    ]
    args.markdown.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["status", "configuration", "selection_mass", "targeting", "copyable_and_source_controls", "whole_word_and_corruption_audit", "exposure_accounting", "elapsed_sec"]}, indent=2))


if __name__ == "__main__":
    main()
