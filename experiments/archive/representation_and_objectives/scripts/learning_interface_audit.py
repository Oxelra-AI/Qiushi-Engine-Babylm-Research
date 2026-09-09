#!/usr/bin/env python3
"""research: learning-interface audit for S/G/C marginal factorization.

This script does not train. It tests the following concern before H100 training:
  * realized WWM target burden under the historical trainer geometry;
  * legal16k tokenizer support/tail distributions for side-view tokens and predicted targets;
  * fragmentation/pathology differences and high-mismatch examples;
  * a matched-support subset suggestion when S/G generic interfaces remain imbalanced.

It mirrors the relevant historical trainer mechanics:
  * examples are taken in stream order from a 100M stream for max_word_exposure words;
  * seq_length=256 truncation, padding ignored;
  * whole-word groups are inferred from tokenizer token strings using the same word-start rule;
  * wwm_fixed masks groups with probability 0.15 using torch.Generator seed 43023.

For C, the same 10M compact_view_reinvest pool is audited alongside S/G pools. For S/G,
side segments are mapped by pair_id inside the first 3,005 pair-block rows, so target
statistics can be separated into source and side portions.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import pathlib
import re
import statistics
from dataclasses import dataclass
from typing import Any, Iterable

import torch
from transformers import AutoTokenizer

ROOT = pathlib.Path("experiments/archive/representation_and_objectives")
A02 = pathlib.Path("experiments/archive/frontier_consolidation")
TOKENIZER = ROOT / "training/runs/gc_compact_repeat_reinvest_16k_seed43022_r2/hf_model/chck_1M"
CSG = ROOT / "data/marginal_corpora_v2/marginal_corpora_csg_v2.jsonl"
STREAMS = {
    "C": A02 / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl",
    "S": ROOT / "data/training_streams/arm_s_100M_stream.jsonl",
    "G": ROOT / "data/training_streams/arm_g_100M_stream.jsonl",
}
OUT_DIR = ROOT / "data/learning_interface_audit"
SEQ_LEN = 256
MASK_PROB = 0.15
TRAIN_SEED = 43023
MAX_WORD_EXPOSURE = 20_000_000
PAIR_SOURCES = {
    "S": "fineweb_s_extract_reinvest",
    "G": "fineweb_g_extract_reinvest",
}
# Historical compact streams are not row-aligned with the research S/G streams; C is audited
# statically from the CSG rows unless/until an aligned C stream is rebuilt.
REALIZED_ARMS = ["S", "G"]

# Same proxy vocabulary class as research, kept local for reproducibility.
FUNC_WORDS = frozenset("a an the this that these those my your his her its our their "
    "is am are was were be been being have has had do does did will would shall should "
    "can could may might must need ought to of in on at by for with from into through "
    "during before after above below between under over about against along across "
    "around behind beside beyond down near off since toward upon within without "
    "and but or nor so yet both either neither not no nor if then else than as "
    "which who whom whose what when where how while until because although though "
    "even also just only still already very much more most less least too quite "
    "really rather than such same other another each every all some any few many "
    "no more several enough another each own same different many little much few "
    "i me we us you he him she her it they them myself yourself himself herself "
    "itself ourselves themselves one ones there here up out off away again back "
    "now then so however therefore thus hence moreover furthermore nevertheless "
    "meanwhile otherwise instead indeed certainly perhaps maybe probably "
    "been being getting going doing having making taking".split())


def norm(w: str) -> str:
    return re.sub(r"[^a-z0-9]", "", w.lower())


def is_content(w: str) -> bool:
    n = norm(w)
    return n not in FUNC_WORDS and len(n) > 1


def is_word_start(tok: str) -> bool:
    # Copied from historical trainer. ByteLevel BPE starts non-initial words with Ġ.
    if tok is None:
        return False
    return tok.startswith("Ġ") or tok.startswith("▁") or tok.startswith(" ")


def stats(xs: list[float | int]) -> dict[str, Any]:
    if not xs:
        return {"n": 0}
    s = sorted(xs)
    n = len(s)
    def q(p: float):
        if n == 1:
            return s[0]
        return s[min(n-1, max(0, int(round(p*(n-1)))))]
    return {
        "n": n,
        "mean": round(float(statistics.mean(s)), 6),
        "median": round(float(statistics.median(s)), 6),
        "p05": round(float(q(0.05)), 6),
        "p10": round(float(q(0.10)), 6),
        "p25": round(float(q(0.25)), 6),
        "p75": round(float(q(0.75)), 6),
        "p90": round(float(q(0.90)), 6),
        "p95": round(float(q(0.95)), 6),
        "min": round(float(s[0]), 6),
        "max": round(float(s[-1]), 6),
    }


def read_jsonl(path: pathlib.Path) -> Iterable[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class PairRow:
    pair_id: str
    source_text: str
    C_text: str
    S_text: str
    G_text: str
    S_positions: list[int]
    G_positions: list[int]


def load_csg() -> dict[str, PairRow]:
    d: dict[str, PairRow] = {}
    for r in read_jsonl(CSG):
        d[r["pair_id"]] = PairRow(
            pair_id=r["pair_id"], source_text=r["source_text"], C_text=r["C_text"],
            S_text=r["S_text"], G_text=r["G_text"],
            S_positions=list(r.get("S_positions") or []), G_positions=list(r.get("G_positions") or []),
        )
    return d


def encode_ids(tok, text: str) -> list[int]:
    return tok(text, add_special_tokens=False)["input_ids"]


def encode_trunc_ids(tok, text: str) -> list[int]:
    return tok(text, add_special_tokens=False, truncation=True, max_length=SEQ_LEN)["input_ids"]


def word_groups_for_ids(tok, ids: list[int], special_ids: set[int]) -> list[int]:
    groups = []
    gid = -1
    for i, tid in enumerate(ids):
        if tid in special_ids:
            groups.append(-1)
            continue
        ts = tok.convert_ids_to_tokens(int(tid))
        if gid < 0 or is_word_start(str(ts)) or i == 0:
            gid += 1
        groups.append(gid)
    return groups


def support_counts_from_streams(tok) -> tuple[collections.Counter[int], collections.Counter[int]]:
    """Counts over one 10M epoch for C/S/G union: static support baseline.

    Return both full-token counts and truncated-to-seq token counts. This avoids relying on
    the trainer's first-5000 sample rank when evaluating the experimental interface.
    """
    full = collections.Counter()
    trunc = collections.Counter()
    for arm in ["S", "G"]:
        path = STREAMS[arm]
        words = 0
        for row in read_jsonl(path):
            # Only first 10M words: stream is 10 epochs, first epoch is the pool.
            if words >= 10_000_000:
                break
            text = row["text"]
            ids = encode_ids(tok, text)
            full.update(ids)
            tids = ids[:SEQ_LEN]
            trunc.update(tids)
            words += int(row["words"])
    # Add compact side C directly from the CSG rows so C static support is not missing.
    for r in read_jsonl(CSG):
        ids = encode_ids(tok, r["C_text"])
        full.update(ids)
        trunc.update(ids[:SEQ_LEN])
    return full, trunc


def bin_support(c: int) -> str:
    if c <= 0:
        return "0"
    if c <= 1:
        return "1"
    if c <= 2:
        return "2"
    if c <= 5:
        return "3-5"
    if c <= 10:
        return "6-10"
    if c <= 50:
        return "11-50"
    if c <= 200:
        return "51-200"
    if c <= 1000:
        return "201-1000"
    return ">1000"


def normalize_hist(counter: collections.Counter[str]) -> dict[str, Any]:
    total = sum(counter.values())
    order = ["0", "1", "2", "3-5", "6-10", "11-50", "51-200", "201-1000", ">1000"]
    return {k: {"count": int(counter.get(k, 0)), "frac": round(counter.get(k, 0)/total, 6) if total else 0.0} for k in order}


def support_hist(ids: Iterable[int], support: collections.Counter[int]) -> dict[str, Any]:
    h = collections.Counter(bin_support(int(support.get(int(t), 0))) for t in ids)
    return normalize_hist(h)


def target_hist_to_payload(counter: collections.Counter[int], support: collections.Counter[int]) -> dict[str, Any]:
    total = sum(counter.values())
    bins = collections.Counter()
    token_mass_by_bin = collections.Counter()
    for tid, c in counter.items():
        b = bin_support(int(support.get(int(tid), 0)))
        bins[b] += c
        token_mass_by_bin[b] += c
    return {
        "total": int(total),
        "support_bins": normalize_hist(bins),
        "unique_token_types": len(counter),
        "rare_le_5_count": int(sum(c for tid, c in counter.items() if support.get(int(tid), 0) <= 5)),
        "rare_le_5_frac": round(sum(c for tid, c in counter.items() if support.get(int(tid), 0) <= 5) / total, 6) if total else 0.0,
        "rare_le_10_count": int(sum(c for tid, c in counter.items() if support.get(int(tid), 0) <= 10)),
        "rare_le_10_frac": round(sum(c for tid, c in counter.items() if support.get(int(tid), 0) <= 10) / total, 6) if total else 0.0,
    }


def token_side_static(tok, pairs: dict[str, PairRow], support: collections.Counter[int]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in ["C", "S", "G"]:
        all_ids = []
        bpe_per_word = []
        words = []
        for pr in pairs.values():
            text = getattr(pr, f"{arm}_text")
            ids = encode_ids(tok, text)
            n_words = len(text.split())
            all_ids.extend(ids)
            bpe_per_word.append(len(ids)/n_words if n_words else 0.0)
            words.append(n_words)
        cnt = collections.Counter(all_ids)
        out[arm] = {
            "side_words": int(sum(words)),
            "side_bpe": int(len(all_ids)),
            "bpe_per_word": stats(bpe_per_word),
            "support_bins_full_10M_union": target_hist_to_payload(cnt, support),
        }
    return out


def fragmentation_static(pairs: dict[str, PairRow]) -> dict[str, Any]:
    rows = []
    for pr in pairs.values():
        src = pr.source_text.split()
        rw_norms = set(norm(w) for w in pr.C_text.split() if norm(w))
        row = {"pair_id": pr.pair_id, "source_text": pr.source_text, "C_text": pr.C_text, "S_text": pr.S_text, "G_text": pr.G_text}
        for arm, pos in [("S", pr.S_positions), ("G", pr.G_positions)]:
            gaps = [pos[i]-pos[i-1]-1 for i in range(1, len(pos))]
            spans = 0
            if pos:
                spans = 1
                for i in range(1, len(pos)):
                    if pos[i] > pos[i-1] + 1:
                        spans += 1
            selected_norms = set(norm(src[j]) for j in pos if 0 <= j < len(src) and norm(src[j]))
            compact_overlap = len(selected_norms & rw_norms) / len(rw_norms) if rw_norms else 0.0
            row[f"{arm}_span_count"] = spans
            row[f"{arm}_max_gap"] = max(gaps) if gaps else 0
            row[f"{arm}_mean_gap"] = statistics.mean(gaps) if gaps else 0.0
            row[f"{arm}_compact_overlap"] = compact_overlap
            row[f"{arm}_content_frac"] = sum(1 for j in pos if is_content(src[j])) / len(pos) if pos else 0.0
        row["G_minus_S_spans"] = row["G_span_count"] - row["S_span_count"]
        row["G_minus_S_max_gap"] = row["G_max_gap"] - row["S_max_gap"]
        row["S_minus_G_compact_overlap"] = row["S_compact_overlap"] - row["G_compact_overlap"]
        rows.append(row)
    return {
        "S_span_count": stats([r["S_span_count"] for r in rows]),
        "G_span_count": stats([r["G_span_count"] for r in rows]),
        "G_minus_S_spans": stats([r["G_minus_S_spans"] for r in rows]),
        "S_max_gap": stats([r["S_max_gap"] for r in rows]),
        "G_max_gap": stats([r["G_max_gap"] for r in rows]),
        "S_mean_gap": stats([r["S_mean_gap"] for r in rows]),
        "G_mean_gap": stats([r["G_mean_gap"] for r in rows]),
        "high_mismatch_examples": sorted(rows, key=lambda r: (r["G_minus_S_spans"], r["S_minus_G_compact_overlap"], r["G_max_gap"]), reverse=True)[:20],
        "low_semantic_separation_examples": sorted(rows, key=lambda r: r["S_minus_G_compact_overlap"])[:10],
    }


def detect_side_start_token_index(tok, text: str, source_text: str, side_text: str) -> tuple[int | None, int, int]:
    src_ids = encode_ids(tok, source_text)
    side_ids = encode_ids(tok, side_text)
    full_ids = encode_ids(tok, text)
    # In all constructed streams, text is source + space + side; tokenization should concatenate this way.
    # Byte-level BPE can merge through spaces only by word-start markers, so use source token count as first approximation.
    start = len(src_ids)
    if full_ids[start:start+min(5, len(side_ids))] == side_ids[:min(5, len(side_ids))]:
        return start, len(src_ids), len(side_ids)
    # Robust fallback: find side token sequence.
    max_probe = min(len(side_ids), 16)
    probe = side_ids[:max_probe]
    found = None
    for i in range(max(0, start-4), min(len(full_ids), start+8)):
        if full_ids[i:i+max_probe] == probe:
            found = i
            break
    if found is None:
        for i in range(len(full_ids)-max_probe+1):
            if full_ids[i:i+max_probe] == probe:
                found = i
                break
    return found, len(src_ids), len(side_ids)


def load_pair_lookup_for_stream(arm: str, pairs: dict[str, PairRow]) -> dict[str, tuple[str, str]]:
    # block rows contain source+side concatenations in original pair order. To map row source text to side text,
    # reconstruct per-pair fragments and perform a deterministic split by source prefix sequence.
    # The training-stream builder used original pair rows in CSG order, accumulating blocks, so pair ids are not in stream rows.
    # For row-level WWM audit, instead recreate the first 3005 pair-block row boundaries from the original C stream,
    # then align S/G by row index. This function is not needed; kept only for clarity.
    return {}


def build_pair_block_side_boundaries(tok, csg_rows: list[PairRow], block_word_counts: list[int], arm: str) -> list[list[tuple[int, int, str]]]:
    """Recreate per-block token side spans for source+side concatenated pair rows.

    Returns list for each pair-block row: [(side_token_start, side_token_end, pair_id), ...] in untruncated token coordinates.
    """
    out: list[list[tuple[int, int, str]]] = []
    idx = 0
    for bw in block_word_counts:
        used_words = 0
        text_parts = []
        spans = []
        token_cursor_text = ""
        while idx < len(csg_rows):
            pr = csg_rows[idx]
            side_text = getattr(pr, f"{arm}_text")
            pair_text = pr.source_text + " " + side_text
            pair_words = len(pair_text.split())
            if used_words + pair_words > bw:
                break
            # Build prefix token count before side by actually tokenizing concatenated text prefix.
            before_text = (token_cursor_text + " " + pr.source_text).strip() if token_cursor_text else pr.source_text
            full_after_pair = (token_cursor_text + " " + pair_text).strip() if token_cursor_text else pair_text
            side_start = len(encode_ids(tok, before_text))
            side_end = len(encode_ids(tok, full_after_pair))
            spans.append((side_start, side_end, pr.pair_id))
            token_cursor_text = full_after_pair
            used_words += pair_words
            idx += 1
        if used_words != bw:
            raise RuntimeError(f"block reconstruction word mismatch: got {used_words}, expected {bw}, idx={idx}")
        out.append(spans)
    if idx != len(csg_rows):
        raise RuntimeError(f"did not consume all csg rows: {idx} vs {len(csg_rows)}")
    return out


def first_epoch_pair_block_word_counts() -> list[int]:
    counts = []
    for row in read_jsonl(STREAMS["S"]):
        if row.get("source") == PAIR_SOURCES["S"]:
            counts.append(int(row["words"]))
        else:
            break
    if len(counts) != 3005:
        raise RuntimeError(f"expected 3005 pair block rows at start of S stream, got {len(counts)}")
    return counts


def simulate_wwm(tok, csg_rows: list[PairRow], support: collections.Counter[int], max_words: int = MAX_WORD_EXPOSURE) -> dict[str, Any]:
    special_ids = set(tok.all_special_ids)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(TRAIN_SEED)
    block_word_counts = first_epoch_pair_block_word_counts()
    side_spans_by_arm = {arm: build_pair_block_side_boundaries(tok, csg_rows, block_word_counts, arm) for arm in REALIZED_ARMS}
    result: dict[str, Any] = {}
    for arm in REALIZED_ARMS:
        path = STREAMS[arm]
        words_seen = 0
        row_index = 0
        pair_row_index_within_epoch = 0
        side_target_counter = collections.Counter()
        source_target_counter = collections.Counter()
        filler_target_counter = collections.Counter()
        side_piece_targets_per_row = []
        side_word_targets_per_row = []
        source_piece_targets_per_row = []
        source_word_targets_per_row = []
        all_piece_targets_per_row = []
        effective_mask_rates = []
        side_trunc_loss_rows = 0
        side_rows_seen = 0
        side_token_candidates_total = 0
        side_token_visible_total = 0
        side_word_group_candidates_total = 0
        side_word_group_masked_total = 0
        examples_seen = 0
        for row in read_jsonl(path):
            if words_seen >= max_words:
                break
            take_words = int(row["words"])
            # The selected streams are exact 10M pools repeated; max_words=20M should stop at epoch boundary.
            if words_seen + take_words > max_words:
                # The trainer would truncate by whitespace only if exposure cut through an example; not needed for 20M.
                text = " ".join(row["text"].split()[:max_words - words_seen])
                take_words = max_words - words_seen
            else:
                text = row["text"]
            ids = encode_trunc_ids(tok, text)
            groups = word_groups_for_ids(tok, ids, special_ids)
            valid_groups = sorted(set(g for g in groups if g >= 0))
            chosen_groups = set()
            if valid_groups:
                rand = torch.rand(len(valid_groups), generator=gen)
                for g, rv in zip(valid_groups, rand.tolist()):
                    if rv < MASK_PROB:
                        chosen_groups.add(g)
            # historical fallback if no token selected in whole batch, not per example; negligible for seq256 and omitted here.
            selected_positions = [i for i, g in enumerate(groups) if g in chosen_groups and ids[i] not in special_ids]
            all_piece_targets_per_row.append(len(selected_positions))
            effective_mask_rates.append(len(selected_positions) / max(1, len([t for t in ids if t not in special_ids])))
            # Map side/source/filler.
            is_pair = row.get("source") == PAIR_SOURCES[arm]
            side_mask = [False] * len(ids)
            if is_pair:
                side_rows_seen += 1
                # Index within epoch repeats every 62859 rows; pair blocks at start of each 10M epoch.
                bidx = pair_row_index_within_epoch % len(block_word_counts)
                spans = side_spans_by_arm[arm][bidx]
                for st, en, _pid in spans:
                    if st < SEQ_LEN:
                        for i in range(st, min(en, SEQ_LEN)):
                            side_mask[i] = True
                    if en > SEQ_LEN:
                        side_trunc_loss_rows += 1
                side_visible = sum(1 for i, m in enumerate(side_mask) if m and i < len(ids) and ids[i] not in special_ids)
                side_full = sum(max(0, en-st) for st, en, _ in spans)
                side_token_visible_total += side_visible
                side_token_candidates_total += side_full
                # word groups in side
                side_groups = set(groups[i] for i, m in enumerate(side_mask) if m and groups[i] >= 0)
                side_word_group_candidates_total += len(side_groups)
                pair_row_index_within_epoch += 1
            side_piece = 0
            source_piece = 0
            filler_piece = 0
            masked_side_groups = set()
            masked_source_groups = set()
            for i in selected_positions:
                tid = ids[i]
                if is_pair and side_mask[i]:
                    side_target_counter[tid] += 1
                    side_piece += 1
                    masked_side_groups.add(groups[i])
                elif is_pair:
                    source_target_counter[tid] += 1
                    source_piece += 1
                    masked_source_groups.add(groups[i])
                else:
                    filler_target_counter[tid] += 1
                    filler_piece += 1
            if is_pair:
                side_piece_targets_per_row.append(side_piece)
                source_piece_targets_per_row.append(source_piece)
                side_word_targets_per_row.append(len([g for g in masked_side_groups if g >= 0]))
                source_word_targets_per_row.append(len([g for g in masked_source_groups if g >= 0]))
                side_word_group_masked_total += len([g for g in masked_side_groups if g >= 0])
            words_seen += take_words
            examples_seen += 1
            row_index += 1
        side_target_total = sum(side_target_counter.values())
        source_target_total = sum(source_target_counter.values())
        filler_target_total = sum(filler_target_counter.values())
        result[arm] = {
            "examples_seen": examples_seen,
            "words_seen": words_seen,
            "pair_rows_seen": side_rows_seen,
            "side_truncated_span_touches": side_trunc_loss_rows,
            "side_visible_token_fraction": round(side_token_visible_total / side_token_candidates_total, 6) if side_token_candidates_total else None,
            "side_word_masked_rate_realized": round(side_word_group_masked_total / side_word_group_candidates_total, 6) if side_word_group_candidates_total else None,
            "effective_mask_rate_all_tokens": stats(effective_mask_rates),
            "target_counts": {
                "side_pieces": int(side_target_total),
                "source_pieces": int(source_target_total),
                "filler_pieces": int(filler_target_total),
                "all_pieces": int(side_target_total + source_target_total + filler_target_total),
                "side_piece_fraction_of_pair_targets": round(side_target_total / max(1, side_target_total + source_target_total), 6),
            },
            "side_piece_targets_per_pair_row": stats(side_piece_targets_per_row),
            "side_word_targets_per_pair_row": stats(side_word_targets_per_row),
            "source_piece_targets_per_pair_row": stats(source_piece_targets_per_row),
            "source_word_targets_per_pair_row": stats(source_word_targets_per_row),
            "support_bins_realized_side_targets_full_10M_union": target_hist_to_payload(side_target_counter, support),
            "support_bins_realized_source_targets_full_10M_union": target_hist_to_payload(source_target_counter, support),
            "support_bins_realized_filler_targets_full_10M_union": target_hist_to_payload(filler_target_counter, support),
        }
    return result


def compare_arm_metrics(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    return {
        "side_pieces_diff_a_minus_b": a["target_counts"]["side_pieces"] - b["target_counts"]["side_pieces"],
        "side_pieces_ratio_a_over_b": round(a["target_counts"]["side_pieces"] / max(1, b["target_counts"]["side_pieces"]), 6),
        "side_words_mask_rate_diff": round((a["side_word_masked_rate_realized"] or 0) - (b["side_word_masked_rate_realized"] or 0), 6),
        "side_visible_token_fraction_diff": round((a["side_visible_token_fraction"] or 0) - (b["side_visible_token_fraction"] or 0), 6),
        "side_target_rare_le5_frac_diff": round(a["support_bins_realized_side_targets_full_10M_union"]["rare_le_5_frac"] - b["support_bins_realized_side_targets_full_10M_union"]["rare_le_5_frac"], 6),
        "side_target_rare_le10_frac_diff": round(a["support_bins_realized_side_targets_full_10M_union"]["rare_le_10_frac"] - b["support_bins_realized_side_targets_full_10M_union"]["rare_le_10_frac"], 6),
    }


def matched_subset_suggestion(tok, pairs: dict[str, PairRow], support: collections.Counter[int]) -> dict[str, Any]:
    """Find a common-support subset with closer S/G BPE and fragmentation while preserving semantic separation.

    This is a CPU planning output, not a new corpus.
    """
    rows = []
    for pr in pairs.values():
        s_ids = encode_ids(tok, pr.S_text)
        g_ids = encode_ids(tok, pr.G_text)
        n_words = max(1, len(pr.S_text.split()))
        s_bpw = len(s_ids) / n_words
        g_bpw = len(g_ids) / n_words
        src = pr.source_text.split()
        def spans(pos):
            if not pos: return 0
            x = 1
            for i in range(1, len(pos)):
                if pos[i] > pos[i-1] + 1: x += 1
            return x
        rw_norms = set(norm(w) for w in pr.C_text.split() if norm(w))
        s_norms = set(norm(src[j]) for j in pr.S_positions if j < len(src) and norm(src[j]))
        g_norms = set(norm(src[j]) for j in pr.G_positions if j < len(src) and norm(src[j]))
        sep = (len(s_norms & rw_norms) - len(g_norms & rw_norms)) / max(1, len(rw_norms))
        s_rare = sum(1 for t in s_ids if support.get(t, 0) <= 10) / max(1, len(s_ids))
        g_rare = sum(1 for t in g_ids if support.get(t, 0) <= 10) / max(1, len(g_ids))
        rows.append({
            "pair_id": pr.pair_id, "s_bpw": s_bpw, "g_bpw": g_bpw, "bpw_absdiff": abs(s_bpw-g_bpw),
            "s_spans": spans(pr.S_positions), "g_spans": spans(pr.G_positions), "span_absdiff": abs(spans(pr.S_positions)-spans(pr.G_positions)),
            "sep": sep, "s_rare10": s_rare, "g_rare10": g_rare, "rare_absdiff": abs(s_rare-g_rare),
            "source_text": pr.source_text, "C_text": pr.C_text, "S_text": pr.S_text, "G_text": pr.G_text,
        })
    suggestions = {}
    configs = [
        ("strict_bpe_fragment_sep", lambda r: r["bpw_absdiff"] <= 0.20 and r["span_absdiff"] <= 2 and r["sep"] >= 0.10),
        ("bpe_only_sep", lambda r: r["bpw_absdiff"] <= 0.15 and r["sep"] >= 0.10),
        ("rare_bpe_fragment_sep", lambda r: r["bpw_absdiff"] <= 0.20 and r["rare_absdiff"] <= 0.10 and r["span_absdiff"] <= 2 and r["sep"] >= 0.10),
        ("large_semantic_sep", lambda r: r["sep"] >= 0.25),
    ]
    for name, pred in configs:
        keep = [r for r in rows if pred(r)]
        suggestions[name] = {
            "n_pairs": len(keep),
            "pair_fraction": round(len(keep)/len(rows), 6),
            "S_bpe_per_word": stats([r["s_bpw"] for r in keep]),
            "G_bpe_per_word": stats([r["g_bpw"] for r in keep]),
            "S_minus_G_bpw": stats([r["s_bpw"]-r["g_bpw"] for r in keep]),
            "S_spans": stats([r["s_spans"] for r in keep]),
            "G_spans": stats([r["g_spans"] for r in keep]),
            "semantic_sep": stats([r["sep"] for r in keep]),
            "rare10_absdiff": stats([r["rare_absdiff"] for r in keep]),
            "examples": keep[:5],
        }
    return suggestions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_word_exposure", type=int, default=MAX_WORD_EXPOSURE)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / "learning_interface_audit.json"
    if out_json.exists() and not args.force:
        print(out_json)
        return
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True, use_fast=True)
    pairs = load_csg()
    csg_rows = list(pairs.values())
    full_support, trunc_support = support_counts_from_streams(tok)
    static = token_side_static(tok, pairs, full_support)
    frag = fragmentation_static(pairs)
    realized = simulate_wwm(tok, csg_rows, full_support, max_words=args.max_word_exposure)
    comparisons = {
        "S_minus_G_realized": compare_arm_metrics(realized["S"], realized["G"]),
        "C_minus_S_realized": "not_computed_without_row_aligned_C_stream",
        "C_minus_G_realized": "not_computed_without_row_aligned_C_stream",
    }
    suggestions = matched_subset_suggestion(tok, pairs, full_support)
    payload = {
        "status": "LEARNING_INTERFACE_AUDIT",
        "meaning": "CPU-only pre-H100 audit of realized WWM burden, tokenizer support/tail, and fragmentation for C/S/G marginal factorization. This supersedes a loss-only 20M launch decision.",
        "inputs": {
            "tokenizer": str(TOKENIZER),
            "tokenizer_len": len(tok),
            "csg": str(CSG),
            "csg_sha256": sha256_file(CSG),
            "streams": {k: str(v) for k, v in STREAMS.items()},
            "stream_sha256": {k: sha256_file(v) for k, v in STREAMS.items()},
            "seq_len": SEQ_LEN,
            "mask_prob": MASK_PROB,
            "train_seed": TRAIN_SEED,
            "max_word_exposure_simulated": args.max_word_exposure,
        },
        "static_side_token_ecology": static,
        "fragmentation_and_examples": frag,
        "realized_wwm_first_exposure": realized,
        "arm_comparisons": comparisons,
        "matched_subset_suggestions": suggestions,
        "scientific_reading": {
            "stage1_launch_in_step223_form": "paused; interpret after reading measured interface differences, not by 20M MLM loss alone",
            "loss_decision_warning": "A short-run loss gap would conflate reconstruction ease with useful semantic denoising; harder local MLM data has produced better competence in earlier triangle evidence.",
        },
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    # Short console summary.
    print(json.dumps({
        "status": payload["status"],
        "out": str(out_json),
        "S_static_bpe": static["S"]["side_bpe"],
        "G_static_bpe": static["G"]["side_bpe"],
        "S_G_realized": comparisons["S_minus_G_realized"],
        "fragment_G_minus_S_spans_mean": frag["G_minus_S_spans"]["mean"],
        "strict_subset_pairs": suggestions["strict_bpe_fragment_sep"]["n_pairs"],
    }, indent=2))


if __name__ == "__main__":
    main()
