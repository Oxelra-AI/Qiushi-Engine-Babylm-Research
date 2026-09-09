#!/usr/bin/env python3
"""Build strict, corpus-derived metadata for innovation-biased WWM.

The output is keyed by the frozen stream's ``example_id``.  It is metadata for
mask selection only: normalized target strings and categories must never be
passed to the model.  A rewrite group is eligible only when *every* token piece
in its tokenizer WWM group is contained in one pair's visible rewrite ranges.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import collections
import hashlib
import json
import pathlib
import re
import time
from typing import Any, Iterable

from transformers import AutoTokenizer


EXPECTED_TRAIN_SHA = "3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691"
EXPECTED_TOKENIZER_SHA = "91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9"
CHANGED_SOURCE = "cleanqwen_fineweb_compact_view_reinvest"
SEQ_LEN = 256

RELATION_WORDS = {
    "in", "on", "under", "over", "above", "below", "behind", "beside", "near", "inside", "outside",
    "into", "onto", "between", "through", "across", "around", "from", "to", "with", "without", "before",
    "after", "during", "while", "when", "because", "therefore", "then", "if", "unless", "although",
    "but", "not", "no", "never", "only", "all", "some", "every", "any", "more", "less", "same", "different",
    "cause", "causes", "caused", "make", "makes", "made", "move", "moves", "moved", "put", "puts", "placed",
    "go", "goes", "went", "fall", "falls", "fell", "open", "opens", "closed", "break", "breaks", "broke",
}


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
WORKSPACE = USER_ROOT / "experiments/archive/frontier_consolidation"
DEFAULT_TRAIN = WORKSPACE / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl"
DEFAULT_SPANS = WORKSPACE / "data/pair_span_map/pair_span_map.jsonl"
DEFAULT_TOKENIZER = WORKSPACE / "data/compliant_tokenizer"
DEFAULT_OUT = WORKSPACE / "analysis/innovation_metadata.jsonl"
DEFAULT_TRAINER_OUT = WORKSPACE / "analysis/innovation_metadata_train.jsonl"
DEFAULT_SUMMARY = WORKSPACE / "analysis/innovation_metadata_summary.json"


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_word_start(token: str) -> bool:
    return token.startswith("Ġ") or token.startswith("▁")


def build_word_groups(input_ids: list[int], tokenizer) -> tuple[list[int], dict[int, tuple[int, ...]]]:
    """Reproduce the baseline trainer's tokenizer-derived WWM grouping."""
    special = set(int(x) for x in tokenizer.all_special_ids)
    group_at = [-1] * len(input_ids)
    groups: dict[int, list[int]] = collections.defaultdict(list)
    gid = -1
    for pos, tid in enumerate(input_ids):
        if int(tid) in special:
            continue
        token = str(tokenizer.convert_ids_to_tokens(int(tid)))
        if gid < 0 or is_word_start(token) or pos == 0:
            gid += 1
        group_at[pos] = gid
        groups[gid].append(pos)
    return group_at, {g: tuple(v) for g, v in groups.items()}


def normalize_decoded(text: str) -> str:
    pieces = re.findall(r"[a-z0-9]+", text.lower().replace("Ġ", " ").replace("▁", " "))
    return " ".join(pieces)


def normalized_group(input_ids: list[int], positions: Iterable[int], tokenizer) -> str:
    ids = [int(input_ids[p]) for p in positions]
    return normalize_decoded(tokenizer.decode(ids, clean_up_tokenization_spaces=False))


def range_union(ranges: list[list[int]], limit: int) -> set[int]:
    out: set[int] = set()
    for raw_a, raw_b in ranges:
        a, b = int(raw_a), int(raw_b)
        if a < 0 or b <= a or b > limit:
            raise ValueError(f"invalid range [{a}, {b}) for token length {limit}")
        out.update(range(a, b))
    return out


def strict_groups_in_positions(
    group_at: list[int], groups: dict[int, tuple[int, ...]], allowed: set[int]
) -> tuple[list[int], list[int]]:
    """Return wholly-contained group ids and groups touching a boundary/gap."""
    touching = {group_at[p] for p in allowed if 0 <= p < len(group_at) and group_at[p] >= 0}
    strict = sorted(g for g in touching if set(groups[g]).issubset(allowed))
    partial = sorted(touching.difference(strict))
    return strict, partial


def load_spans(path: pathlib.Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                eid = int(obj["example_id"])
                if eid in out:
                    raise RuntimeError(f"duplicate span example_id {eid}")
                out[eid] = obj
    return out


def load_frozen_examples(
    path: pathlib.Path, wanted: set[int]
) -> tuple[dict[int, dict[str, Any]], collections.Counter, dict[int, set[str]]]:
    """Scan all 100M rows to audit repeated-id identity and exposure count."""
    first: dict[int, dict[str, Any]] = {}
    counts: collections.Counter = collections.Counter()
    text_shas: dict[int, set[str]] = collections.defaultdict(set)
    with path.open("r", encoding="utf-8") as f:
        for row_1, line in enumerate(f, 1):
            obj = json.loads(line)
            eid = int(obj.get("example_id", row_1 - 1))
            if eid not in wanted:
                continue
            counts[eid] += 1
            text = str(obj["text"])
            text_shas[eid].add(sha256_text(text))
            if eid not in first:
                first[eid] = {
                    "text": text,
                    "words": int(obj.get("words", len(text.split()))),
                    "source": str(obj.get("source", "")),
                    "first_global_row_1based": row_1,
                }
    return first, counts, text_shas


def target_record(gid: int, positions: tuple[int, ...], norm: str, cue: bool, source_occurrences: int) -> dict[str, Any]:
    return {
        "group_id": int(gid),
        "positions": list(positions),
        "token_count": len(positions),
        "normalized_text": norm,
        "normalized_sha256": sha256_text(norm),
        "cue_class": "relation_cue" if cue else "content_or_other",
        "source_normalized_occurrences": int(source_occurrences),
        "strict_full_group": True,
    }


def build_row_metadata(example_id: int, row: dict[str, Any], span: dict[str, Any], tokenizer) -> tuple[dict[str, Any], collections.Counter]:
    stats: collections.Counter = collections.Counter()
    enc = tokenizer(
        row["text"], add_special_tokens=False, truncation=True, max_length=SEQ_LEN,
        padding=False, return_attention_mask=False,
    )
    input_ids = [int(x) for x in enc["input_ids"]]
    if len(input_ids) != int(span["token_len_truncated"]):
        raise RuntimeError(
            f"token length mismatch for example_id {example_id}: rebuilt={len(input_ids)} span={span['token_len_truncated']}"
        )
    group_at, groups = build_word_groups(input_ids, tokenizer)
    pair_records: list[dict[str, Any]] = []
    all_innovation: dict[int, dict[str, Any]] = {}
    all_copyable: dict[int, dict[str, Any]] = {}
    all_source_gids: set[int] = set()
    all_pair_gids: set[int] = set()
    rewrite_membership: dict[int, list[str]] = collections.defaultdict(list)

    for pair in span.get("pairs") or []:
        visibility = str(pair.get("visibility", ""))
        stats[f"pair_visibility:{visibility}"] += 1
        if visibility != "both_visible":
            continue
        src_pos = range_union(pair.get("source_token_ranges") or [], len(input_ids))
        rew_pos = range_union(pair.get("rewrite_token_ranges") or [], len(input_ids))
        if src_pos.intersection(rew_pos):
            raise RuntimeError(f"source/rewrite overlap in {pair.get('pair_id')} example_id={example_id}")
        source_gids, partial_source = strict_groups_in_positions(group_at, groups, src_pos)
        rewrite_gids, partial_rewrite = strict_groups_in_positions(group_at, groups, rew_pos)
        stats["partial_source_groups_excluded"] += len(partial_source)
        stats["partial_rewrite_groups_excluded"] += len(partial_rewrite)
        stats["discontiguous_source_pairs"] += len(pair.get("source_token_ranges") or []) > 1
        stats["discontiguous_rewrite_pairs"] += len(pair.get("rewrite_token_ranges") or []) > 1
        source_norm_counter: collections.Counter = collections.Counter()
        for gid in source_gids:
            norm = normalized_group(input_ids, groups[gid], tokenizer)
            if norm:
                source_norm_counter[norm] += 1
        innovations: list[dict[str, Any]] = []
        copyables: list[dict[str, Any]] = []
        pair_id = str(pair.get("pair_id", ""))
        for gid in rewrite_gids:
            norm = normalized_group(input_ids, groups[gid], tokenizer)
            if not norm or len(norm) <= 1:
                stats["rewrite_skip_empty_or_single_character"] += 1
                continue
            source_count = int(source_norm_counter[norm])
            cue = (norm.split()[0] if norm.split() else norm) in RELATION_WORDS
            rec = target_record(gid, groups[gid], norm, cue, source_count)
            rec["pair_id"] = pair_id
            rewrite_membership[gid].append(pair_id)
            if source_count:
                rec["category"] = "copyable"
                copyables.append(rec)
                all_copyable.setdefault(gid, rec)
                stats["copyable_groups"] += 1
                stats["copyable_tokens"] += len(groups[gid])
                stats["copyable_repeated_source_match"] += source_count > 1
            else:
                rec["category"] = "innovation"
                innovations.append(rec)
                all_innovation.setdefault(gid, rec)
                stats["innovation_groups"] += 1
                stats["innovation_tokens"] += len(groups[gid])
                stats[f"innovation_cue:{rec['cue_class']}"] += 1
        all_source_gids.update(source_gids)
        # Donor protection is deliberately broader than target eligibility:
        # no selected group touching either side of any pair may be removed.
        all_pair_gids.update(source_gids)
        all_pair_gids.update(rewrite_gids)
        all_pair_gids.update(partial_source)
        all_pair_gids.update(partial_rewrite)
        pair_records.append({
            "pair_id": pair_id,
            "source_ranges": pair.get("source_token_ranges") or [],
            "rewrite_ranges": pair.get("rewrite_token_ranges") or [],
            "source_group_ids": source_gids,
            "partial_source_group_ids_excluded": partial_source,
            "partial_rewrite_group_ids_excluded": partial_rewrite,
            "copyable_groups": copyables,
            "innovation_groups": innovations,
        })

    conflicting = sorted(set(all_innovation).intersection(all_copyable))
    multiple_rewrite_membership = sorted(g for g, memberships in rewrite_membership.items() if len(set(memberships)) > 1)
    if conflicting:
        raise RuntimeError(f"groups classified both copyable and innovation for example_id {example_id}: {conflicting}")
    stats["multiple_pair_rewrite_group_membership"] += len(multiple_rewrite_membership)
    stats["rows_with_innovation"] += bool(all_innovation)
    stats["rows_truncated"] += bool(span.get("truncated_by_seq256"))
    stats["rows_multiple_pairs"] += int(span.get("pair_count", 0)) > 1
    return {
        "schema": "frontier_consolidation_innovation_wwm_v1",
        "example_id": int(example_id),
        "source": row["source"],
        "words": int(row["words"]),
        "first_global_row_1based": int(row["first_global_row_1based"]),
        "token_len_truncated": len(input_ids),
        "token_len_full": int(span["token_len_full"]),
        "truncated_by_seq256": bool(span["truncated_by_seq256"]),
        "pair_count_declared": int(span.get("pair_count", 0)),
        "both_visible_pair_count": len(pair_records),
        "pairs": pair_records,
        "innovation_groups": [all_innovation[g] for g in sorted(all_innovation)],
        "copyable_group_ids": sorted(all_copyable),
        "protected_source_group_ids": sorted(all_source_gids),
        "protected_pair_group_ids": sorted(all_pair_gids),
        "multiple_pair_rewrite_group_ids": multiple_rewrite_membership,
        "audit": {
            "classification": "exact normalized full-WWM-group membership in the true paired source only",
            "decoy_used_for_classification": False,
            "model_visible_fields": ["input_ids", "attention_mask"],
            "metadata_model_visible": False,
        },
    }, stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=pathlib.Path, default=DEFAULT_TRAIN)
    p.add_argument("--span-map", type=pathlib.Path, default=DEFAULT_SPANS)
    p.add_argument("--tokenizer", type=pathlib.Path, default=DEFAULT_TOKENIZER)
    p.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    p.add_argument("--trainer-out", type=pathlib.Path, default=DEFAULT_TRAINER_OUT)
    p.add_argument("--summary", type=pathlib.Path, default=DEFAULT_SUMMARY)
    return p.parse_args()


def trainer_projection(rec: dict[str, Any]) -> dict[str, Any]:
    """Remove decoded lexical audit material from the trainer payload."""
    keep_target = (
        "group_id", "positions", "token_count", "cue_class", "category",
        "strict_full_group", "pair_id",
    )
    return {
        "schema": "frontier_consolidation_innovation_wwm_train_v1",
        "example_id": rec["example_id"],
        "frozen_stream_occurrences": rec["frozen_stream_occurrences"],
        "first_global_row_1based": rec["first_global_row_1based"],
        "token_len_truncated": rec["token_len_truncated"],
        "innovation_groups": [
            {k: target[k] for k in keep_target if k in target}
            for target in rec["innovation_groups"]
        ],
        "copyable_group_ids": rec["copyable_group_ids"],
        "protected_source_group_ids": rec["protected_source_group_ids"],
        "protected_pair_group_ids": rec["protected_pair_group_ids"],
        "pairs": [
            {"pair_id": pair["pair_id"], "source_group_ids": pair["source_group_ids"]}
            for pair in rec["pairs"]
        ],
    }


def main() -> None:
    args = parse_args()
    t0 = time.time()
    train_sha = sha256_file(args.train)
    tokenizer_sha = sha256_file(args.tokenizer / "tokenizer.json")
    if train_sha != EXPECTED_TRAIN_SHA:
        raise RuntimeError(f"frozen stream SHA mismatch: {train_sha}")
    if tokenizer_sha != EXPECTED_TOKENIZER_SHA:
        raise RuntimeError(f"research tokenizer SHA mismatch: {tokenizer_sha}")
    spans = load_spans(args.span_map)
    frozen, exposure_counts, text_shas = load_frozen_examples(args.train, set(spans))
    missing = sorted(set(spans).difference(frozen))
    bad_sources = sorted(eid for eid, row in frozen.items() if row["source"] != CHANGED_SOURCE)
    nonidentical_repeats = sorted(eid for eid, shas in text_shas.items() if len(shas) != 1)
    if missing or bad_sources or nonidentical_repeats:
        raise RuntimeError(
            f"join audit failed missing={missing[:5]} bad_sources={bad_sources[:5]} nonidentical={nonidentical_repeats[:5]}"
        )
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), use_fast=True)
    totals: collections.Counter = collections.Counter()
    records: list[dict[str, Any]] = []
    for eid in sorted(spans):
        rec, row_stats = build_row_metadata(eid, frozen[eid], spans[eid], tokenizer)
        rec["frozen_stream_occurrences"] = int(exposure_counts[eid])
        rec["frozen_text_sha256"] = next(iter(text_shas[eid]))
        records.append(rec)
        totals.update(row_stats)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
    with args.trainer_out.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(trainer_projection(rec), ensure_ascii=False, separators=(",", ":")) + "\n")
    occurrence_hist = collections.Counter(int(v) for v in exposure_counts.values())
    summary = {
        "status": "INNOVATION_METADATA_BUILT",
        "schema": "frontier_consolidation_innovation_wwm_v1",
        "inputs": {
            "train": str(args.train.relative_to(USER_ROOT)),
            "train_sha256": train_sha,
            "tokenizer": str(args.tokenizer.relative_to(USER_ROOT)),
            "tokenizer_json_sha256": tokenizer_sha,
            "span_map": str(args.span_map.relative_to(USER_ROOT)),
            "span_map_sha256": sha256_file(args.span_map),
            "seq_len": SEQ_LEN,
        },
        "records": len(records),
        "exposure_count_histogram": dict(sorted(occurrence_hist.items())),
        "totals_unique_10m_pool": dict(sorted(totals.items())),
        "totals_frozen_100m_exposure": {
            "innovation_groups": sum(
                len(rec["innovation_groups"]) * rec["frozen_stream_occurrences"] for rec in records
            ),
            "innovation_tokens": sum(
                sum(x["token_count"] for x in rec["innovation_groups"]) * rec["frozen_stream_occurrences"]
                for rec in records
            ),
            "copyable_groups": sum(
                len(rec["copyable_group_ids"]) * rec["frozen_stream_occurrences"] for rec in records
            ),
        },
        "safety": {
            "copyable_definition": "rewrite normalized full group occurs in its own true source span",
            "innovation_definition": "eligible rewrite normalized full group absent from its own true source span",
            "same_row_decoy_used": False,
            "strict_whole_group_containment": True,
            "partial_boundary_groups_excluded": True,
            "metadata_is_model_input": False,
            "trainer_payload_contains_normalized_text": False,
        },
        "audit_output": str(args.out.relative_to(USER_ROOT)),
        "audit_output_sha256": sha256_file(args.out),
        "trainer_output": str(args.trainer_out.relative_to(USER_ROOT)),
        "trainer_output_sha256": sha256_file(args.trainer_out),
        "elapsed_sec": round(time.time() - t0, 3),
    }
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
