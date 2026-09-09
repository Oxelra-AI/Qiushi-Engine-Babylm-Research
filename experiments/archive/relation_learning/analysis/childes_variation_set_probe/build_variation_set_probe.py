#!/usr/bin/env python3
"""Build a surface-held-out natural CHILDES variation-set MLM probe.

The current experimental CLEAN arm contains the nominal row-holdout slice, so
row provenance alone is insufficient.  This builder first identifies natural
candidate pairs and then runs a token-sequence Aho--Corasick scan over the
actual CLEAN/VIEW/REPEAT 10M pools.  A pair is eligible only when neither
cleaned utterance occurs verbatim in any arm.
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
import math
import random
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from transformers import AutoTokenizer


ROOT = _public_path('.')
OUT_DIR = _public_path('experiments/archive/relation_learning/analysis/childes_variation_set_probe')
RAW_CHILDES = _public_path('experiments/archive/initial_model_studies/data/reconstruct_tmp/raw_dataset/childes.train.txt')
TOKENIZER_PATH = _public_path('experiments/archive/frontier_consolidation/data/compliant_tokenizer')
POOL_DIR = _public_path('experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools')
ARM_POOLS = {
    "clean": _public_path('experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/cleanqwen_lengthmatched_dose2p64x_10M.jsonl'),
    "view": _public_path('experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/compact_view_dose2p64x_10M.jsonl'),
    "repeat": _public_path('experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/compact_repeat_dose2p64x_10M.jsonl'),
}

SEED = 9061701
MAX_PER_BIN = 500
MAX_PAIRS_PER_TRANSCRIPT_PER_BIN = 5
MAX_SEQ_LEN = 256
SPEAKER_RE = re.compile(r"^\*([A-Za-z0-9]+):\t(.*)$")
TRANSCRIPT_RE = re.compile(r"^= = = (childes/.+?\.cha) = = =$", re.I)
WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*")
BRACKET_RE = re.compile(r"\[[^\]]*\]")
CHAT_CODE_RE = re.compile(r"(?<!\w)(?:xxx|yyy|www|zzz|0)(?!\w)", re.I)
CHAT_EVENT_RE = re.compile(r"(?<!\w)&(?:=|\+)[^\s]+")

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "than", "that", "this", "these", "those",
    "of", "in", "on", "at", "to", "for", "from", "with", "without", "by", "as", "is", "are", "was",
    "were", "be", "been", "being", "it", "its", "they", "them", "their", "he", "she", "his", "her",
    "we", "you", "i", "not", "no", "do", "does", "did", "can", "could", "would", "should", "will",
    "have", "has", "had", "just", "so", "very", "also", "about", "more", "some", "any", "all", "each",
    "every", "both", "few", "many", "much", "such", "own", "other", "up", "out",
}


def stable_hex(*parts: object, n: int = 16) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:n]


def normalize_ws(text: str) -> str:
    return " ".join(text.replace("\u00a0", " ").split())


def clean_utterance(text: str) -> str:
    text = BRACKET_RE.sub(" ", text)
    text = CHAT_EVENT_RE.sub(" ", text)
    text = CHAT_CODE_RE.sub(" ", text)
    # CHAT scope/retrace glyphs are not lexical input. Keep ordinary punctuation.
    text = text.replace("<", " ").replace(">", " ")
    text = re.sub(r"(?<!\w)\+(?:/|//|\.{3}|[!?]+)(?!\w)", " ", text)
    return normalize_ws(text).strip()


def content_occurrences(text: str) -> list[dict]:
    out = []
    for match in WORD_RE.finditer(text):
        word = match.group(0).lower().replace("’", "'")
        if len(word) >= 2 and word not in STOPWORDS and not word.isdigit():
            out.append({"word": word, "start": match.start(), "end": match.end()})
    return out


def bin_overlap(jaccard: float) -> str:
    if jaccard >= 0.5:
        return "high"
    if jaccard >= 0.2:
        return "partial"
    return "low"


@dataclass
class Utterance:
    transcript_id: str
    transcript_position: int
    physical_line: int
    speaker: str
    text: str


class TokenAho:
    """Small token-sequence Aho--Corasick matcher, avoiding external deps."""

    def __init__(self, patterns: dict[tuple[object, ...], set[str]]):
        self.next: list[dict[object, int]] = [{}]
        self.fail: list[int] = [0]
        self.out: list[set[str]] = [set()]
        for toks, labels in patterns.items():
            state = 0
            for tok in toks:
                if tok not in self.next[state]:
                    self.next[state][tok] = self._new_state()
                state = self.next[state][tok]
            self.out[state].update(labels)
        queue = collections.deque()
        for child in self.next[0].values():
            self.fail[child] = 0
            queue.append(child)
        while queue:
            state = queue.popleft()
            for tok, child in self.next[state].items():
                queue.append(child)
                fallback = self.fail[state]
                while fallback and tok not in self.next[fallback]:
                    fallback = self.fail[fallback]
                self.fail[child] = self.next[fallback].get(tok, 0)
                self.out[child].update(self.out[self.fail[child]])

    def _new_state(self) -> int:
        self.next.append({})
        self.fail.append(0)
        self.out.append(set())
        return len(self.next) - 1

    def scan(self, tokens: Iterable[object]) -> set[str]:
        found: set[str] = set()
        state = 0
        for tok in tokens:
            while state and tok not in self.next[state]:
                state = self.fail[state]
            state = self.next[state].get(tok, 0)
            if self.out[state]:
                found.update(self.out[state])
        return found


def parse_candidates(tokenizer) -> tuple[list[dict], dict]:
    candidates: list[dict] = []
    parse = collections.Counter()
    current_transcript: str | None = None
    transcript_position = 0
    previous_adult: Utterance | None = None

    with RAW_CHILDES.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, raw in enumerate(handle, 1):
            raw = raw.rstrip("\n\r")
            boundary = TRANSCRIPT_RE.match(raw)
            if boundary:
                current_transcript = boundary.group(1)
                transcript_position = 0
                previous_adult = None
                parse["transcript_boundaries"] += 1
                continue
            match = SPEAKER_RE.match(raw)
            if not match:
                parse["non_speaker_lines"] += 1
                continue
            parse["speaker_lines"] += 1
            transcript_position += 1
            speaker, source_text = match.groups()
            text = clean_utterance(source_text)
            utt = Utterance(current_transcript or "", transcript_position, line_no, speaker, text)
            if speaker == "CHI":
                parse["chi_utterances"] += 1
                if not text:
                    parse["empty_after_cleaning"] += 1
                continue
            parse["adult_by_not_chi"] += 1
            if not text:
                # An empty/noise-only adult turn still interrupts adult-turn
                # adjacency even though it cannot itself qualify.
                parse["empty_after_cleaning"] += 1
            if not current_transcript:
                parse["adult_before_first_boundary"] += 1
                previous_adult = utt
                continue
            if previous_adult is None:
                previous_adult = utt
                continue

            parse["consecutive_adult_pairs"] += 1
            gap = line_no - previous_adult.physical_line
            if gap > 3:
                parse["reject_line_gap"] += 1
                previous_adult = utt
                continue
            c1_occ = content_occurrences(previous_adult.text)
            c2_occ = content_occurrences(utt.text)
            if len(c1_occ) < 3 or len(c2_occ) < 3:
                parse["reject_lt3_content_occurrences"] += 1
                previous_adult = utt
                continue
            c1 = {x["word"] for x in c1_occ}
            c2 = {x["word"] for x in c2_occ}
            overlap = c1 & c2
            nonoverlap = c2 - c1
            if len(nonoverlap) < 2:
                parse["reject_lt2_unique_nonoverlap"] += 1
                previous_adult = utt
                continue
            # The objective says natural variation sets / partially overlapping;
            # require a genuine lexical anchor even in the low-overlap stratum.
            if not overlap:
                parse["reject_zero_overlap_not_variation_set"] += 1
                previous_adult = utt
                continue
            ids1 = tokenizer(previous_adult.text, add_special_tokens=False)["input_ids"]
            ids2 = tokenizer(utt.text, add_special_tokens=False)["input_ids"]
            if 2 + len(ids1) + len(ids2) > MAX_SEQ_LEN:
                parse["reject_gt256_model_tokens"] += 1
                previous_adult = utt
                continue
            union = c1 | c2
            jac = len(overlap) / len(union)
            pair_id = "childes_vs_" + stable_hex(
                current_transcript, previous_adult.physical_line, line_no
            )
            candidates.append({
                "pair_id": pair_id,
                "transcript_id": current_transcript,
                "utt1_text": previous_adult.text,
                "utt2_text": utt.text,
                "speaker1": previous_adult.speaker,
                "speaker2": utt.speaker,
                "same_speaker": previous_adult.speaker == utt.speaker,
                "physical_line_1": previous_adult.physical_line,
                "physical_line_2": line_no,
                "physical_line_gap": gap,
                "transcript_position_1": previous_adult.transcript_position,
                "transcript_position_2": utt.transcript_position,
                "utt1_content_words": sorted(c1),
                "utt2_content_words": sorted(c2),
                "overlap_words": sorted(overlap),
                "nonoverlap_words": sorted(nonoverlap),
                "n_overlap": len(overlap),
                "n_nonoverlap": len(nonoverlap),
                "utt1_content_occurrences": len(c1_occ),
                "utt2_content_occurrences": len(c2_occ),
                "jaccard": jac,
                "overlap_bin": bin_overlap(jac),
                "utt1_token_len": len(ids1),
                "utt2_token_len": len(ids2),
            })
            parse["structurally_qualified"] += 1
            previous_adult = utt
    return candidates, dict(parse)


def surface_exposure_scan(candidates: list[dict], tokenizer) -> tuple[dict[str, set[str]], dict]:
    """Find exact tokenizer-ID occurrences of either cleaned probe utterance."""
    pattern_map: dict[tuple[int, ...], set[str]] = collections.defaultdict(set)
    for row in candidates:
        pattern_map[tuple(tokenizer(row["utt1_text"], add_special_tokens=False)["input_ids"])].add(row["pair_id"])
        pattern_map[tuple(tokenizer(row["utt2_text"], add_special_tokens=False)["input_ids"])].add(row["pair_id"])
    matcher = TokenAho(pattern_map)
    hits_by_arm: dict[str, set[str]] = {}
    rows_by_arm = {}
    words_by_arm = {}
    for arm, path in ARM_POOLS.items():
        hits: set[str] = set()
        rows = words = 0
        with path.open("r", encoding="utf-8") as handle:
            batch: list[str] = []
            for line in handle:
                if line.strip():
                    text = str(json.loads(line)["text"])
                    batch.append(text)
                    rows += 1
                    words += len(text.split())
                if len(batch) == 256:
                    encoded = tokenizer(batch, add_special_tokens=False)["input_ids"]
                    for ids in encoded:
                        hits.update(matcher.scan(ids))
                    batch = []
            if batch:
                encoded = tokenizer(batch, add_special_tokens=False)["input_ids"]
                for ids in encoded:
                    hits.update(matcher.scan(ids))
        hits_by_arm[arm] = hits
        rows_by_arm[arm] = rows
        words_by_arm[arm] = words
    exposure_arms: dict[str, set[str]] = collections.defaultdict(set)
    for arm, hits in hits_by_arm.items():
        for pair_id in hits:
            exposure_arms[pair_id].add(arm)
    return dict(exposure_arms), {
        "patterns": len(pattern_map),
        "candidate_pairs": len(candidates),
        "hits_by_arm": {k: len(v) for k, v in hits_by_arm.items()},
        "hit_union_pairs": len(exposure_arms),
        "rows_scanned": rows_by_arm,
        "whitespace_words_scanned": words_by_arm,
        "pool_paths": {k: str(v.relative_to(ROOT)) for k, v in ARM_POOLS.items()},
        "criterion": "pair rejected if either cleaned utterance's tokenizer-ID sequence is an exact contiguous subsequence of any actual 10M arm row",
    }


def token_targets(tokenizer, row: dict) -> list[dict]:
    encoded = tokenizer(row["utt2_text"], add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded["input_ids"]
    offsets = [tuple(x) for x in encoded["offset_mapping"]]
    overlap = set(row["overlap_words"])
    eligible = []
    for occurrence_index, occ in enumerate(content_occurrences(row["utt2_text"])):
        covering = [i for i, (start, end) in enumerate(offsets) if start < occ["end"] and end > occ["start"]]
        if len(covering) != 1:
            continue
        tok_i = covering[0]
        token_start, token_end = offsets[tok_i]
        # Byte-level BPE offsets commonly include the preceding space in the
        # token span (e.g. "Ġdays" -> the substring " days").
        if token_end != occ["end"] or token_start > occ["start"]:
            continue
        if row["utt2_text"][token_start:occ["start"]].strip():
            continue
        eligible.append({
            "occurrence_index": occurrence_index,
            "word": occ["word"],
            "char_start": occ["start"],
            "char_end": occ["end"],
            "utt2_token_index": tok_i,
            "target_token_id": int(ids[tok_i]),
            "overlap_class": "overlap" if occ["word"] in overlap else "nonoverlap",
        })
    by_class = {
        cls: [x for x in eligible if x["overlap_class"] == cls]
        for cls in ("overlap", "nonoverlap")
    }
    for cls in by_class:
        by_class[cls].sort(key=lambda x: stable_hex(row["pair_id"], cls, x["occurrence_index"]))
    selected: list[dict] = []
    if by_class["overlap"] and by_class["nonoverlap"]:
        selected.extend(by_class["overlap"][:1])
        selected.extend(by_class["nonoverlap"][:2])
        if len(selected) < 3:
            already = {x["occurrence_index"] for x in selected}
            extras = [x for x in eligible if x["occurrence_index"] not in already]
            extras.sort(key=lambda x: stable_hex(row["pair_id"], "fill", x["occurrence_index"]))
            selected.extend(extras[: 3 - len(selected)])
    else:
        only = by_class["overlap"] or by_class["nonoverlap"]
        selected = only[:3]
    selected.sort(key=lambda x: x["utt2_token_index"])
    return selected


def stratified_select(rows: list[dict]) -> tuple[list[dict], dict]:
    by_bin: dict[str, list[dict]] = {b: [] for b in ("low", "partial", "high")}
    for row in rows:
        by_bin[row["overlap_bin"]].append(row)
    selected = []
    audit = {}
    for overlap_bin, items in by_bin.items():
        items.sort(key=lambda r: stable_hex(SEED, r["pair_id"]))
        chosen = []
        per_transcript = collections.Counter()
        for row in items:
            tid = row["transcript_id"]
            if per_transcript[tid] >= MAX_PAIRS_PER_TRANSCRIPT_PER_BIN:
                continue
            chosen.append(row)
            per_transcript[tid] += 1
            if len(chosen) == MAX_PER_BIN:
                break
        # Relax the transcript cap only if needed to reach the target.
        if len(chosen) < min(MAX_PER_BIN, len(items)):
            have = {r["pair_id"] for r in chosen}
            for row in items:
                if row["pair_id"] in have:
                    continue
                chosen.append(row)
                if len(chosen) == min(MAX_PER_BIN, len(items)):
                    break
        selected.extend(chosen)
        audit[overlap_bin] = {
            "available": len(items),
            "selected": len(chosen),
            "unique_transcripts": len({r["transcript_id"] for r in chosen}),
            "max_pairs_per_transcript": max(collections.Counter(r["transcript_id"] for r in chosen).values(), default=0),
        }
    selected.sort(key=lambda r: r["pair_id"])
    return selected, audit


def assign_controls(rows: list[dict]) -> None:
    for row in rows:
        target_len = row["utt1_token_len"]
        lo, hi = math.ceil(0.7 * target_len), math.floor(1.3 * target_len)
        target2 = set(row["utt2_content_words"])
        choices = []
        for control in rows:
            if control["pair_id"] == row["pair_id"] or control["transcript_id"] == row["transcript_id"]:
                continue
            clen = control["utt1_token_len"]
            if not (lo <= clen <= hi):
                continue
            if 2 + clen + row["utt2_token_len"] > MAX_SEQ_LEN:
                continue
            overlap = len(set(control["utt1_content_words"]) & target2)
            choices.append((overlap, abs(clen - target_len), stable_hex(row["pair_id"], control["pair_id"]), control))
        if not choices:
            raise RuntimeError(f"No valid different-transcript control for {row['pair_id']}")
        _, _, _, control = min(choices, key=lambda x: x[:3])
        row["control_pair_id"] = control["pair_id"]
        row["control_transcript_id"] = control["transcript_id"]
        row["control_utt_text"] = control["utt1_text"]
        row["control_speaker"] = control["speaker1"]
        row["control_token_len"] = control["utt1_token_len"]
        row["control_length_ratio"] = control["utt1_token_len"] / target_len
        row["control_target_content_overlap"] = len(set(control["utt1_content_words"]) & target2)


def make_probe_records(tokenizer, rows: list[dict]) -> tuple[list[dict], dict]:
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    mask = tokenizer.mask_token_id
    if bos is None or eos is None or mask is None:
        raise RuntimeError("Tokenizer must define BOS, EOS, and MASK IDs")
    records = []
    target_audit = collections.Counter()
    for row in rows:
        targets = token_targets(tokenizer, row)
        if not targets:
            target_audit["pairs_without_single_token_targets"] += 1
            continue
        target_audit[f"pairs_with_{len(targets)}_targets"] += 1
        utt2_ids = tokenizer(row["utt2_text"], add_special_tokens=False)["input_ids"]
        for condition, context_text in (("intact", row["utt1_text"]), ("replaced", row["control_utt_text"])):
            context_ids = tokenizer(context_text, add_special_tokens=False)["input_ids"]
            canonical = [bos] + context_ids + utt2_ids + [eos]
            for target in targets:
                mask_position = 1 + len(context_ids) + target["utt2_token_index"]
                target_id = int(canonical[mask_position])
                if target_id != target["target_token_id"]:
                    raise AssertionError("Target offset/id mapping drift")
                masked = list(canonical)
                masked[mask_position] = mask
                record_id = "probe_" + stable_hex(row["pair_id"], condition, target["occurrence_index"], n=20)
                records.append({
                    "record_id": record_id,
                    "pair_id": row["pair_id"],
                    "condition": condition,
                    "overlap_bin": row["overlap_bin"],
                    "overlap_class": target["overlap_class"],
                    "input_ids": masked,
                    "attention_mask": [1] * len(masked),
                    "mask_position": mask_position,
                    "target_token_id": target_id,
                    "target_word": target["word"],
                    "target_occurrence_index": target["occurrence_index"],
                    "target_char_start": target["char_start"],
                    "target_char_end": target["char_end"],
                    "utt2_token_index": target["utt2_token_index"],
                    "utt1_token_len": row["utt1_token_len"],
                    "condition_context_token_len": len(context_ids),
                    "utt2_token_len": len(utt2_ids),
                    "seq_len": len(masked),
                    "transcript_id": row["transcript_id"],
                    "control_transcript_id": row["control_transcript_id"] if condition == "replaced" else None,
                })
                target_audit[f"records_{condition}_{target['overlap_class']}"] += 1
    return records, dict(target_audit)


def stats_summary(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(values),
        "min": min(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
        "max": max(values),
    }


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-surface-scan", action="store_true", help="debug only: do not claim held-out")
    args = parser.parse_args()
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True, use_fast=True)
    candidates, parse_audit = parse_candidates(tokenizer)

    if args.skip_surface_scan:
        exposure_arms, exposure_audit = {}, {"skipped": True}
    else:
        exposure_arms, exposure_audit = surface_exposure_scan(candidates, tokenizer)
    for row in candidates:
        row["surface_exposure_arms"] = sorted(exposure_arms.get(row["pair_id"], set()))
    structural_by_bin = collections.Counter(r["overlap_bin"] for r in candidates)
    exposed_by_bin = collections.Counter(r["overlap_bin"] for r in candidates if r["surface_exposure_arms"])
    surface_safe_by_bin = collections.Counter(r["overlap_bin"] for r in candidates if not r["surface_exposure_arms"])
    exposure_audit["candidate_pairs_by_bin"] = dict(structural_by_bin)
    exposure_audit["exposed_pairs_by_bin"] = dict(exposed_by_bin)
    exposure_audit["surface_safe_pairs_by_bin"] = dict(surface_safe_by_bin)
    exposure_audit["surface_safe_fraction"] = (
        sum(surface_safe_by_bin.values()) / len(candidates) if candidates else 0.0
    )
    # Retain only pairs that can yield at least one unambiguous single-token MLM target.
    for row in candidates:
        row["n_single_token_targets_available"] = len(token_targets(tokenizer, row))
    surface_safe = [r for r in candidates if not r["surface_exposure_arms"]]
    safe = [r for r in surface_safe if r["n_single_token_targets_available"] > 0]
    selected, selection_audit = stratified_select(safe)
    minimum_bin_target_met = all(v["selected"] >= 100 for v in selection_audit.values())
    assign_controls(selected)
    records, target_audit = make_probe_records(tokenizer, selected)

    counts = collections.Counter((r["overlap_bin"], r["overlap_class"], r["condition"]) for r in records)
    pair_bins = collections.Counter(r["overlap_bin"] for r in selected)
    speakers = collections.Counter(r["speaker1"] for r in selected)
    stats = {
        "status": "SURFACE_HELDOUT_PROBE_BUILT_BALANCE_TARGET_INFEASIBLE" if not minimum_bin_target_met else "SURFACE_HELDOUT_CHILDES_VARIATION_SET_PROBE_BUILT",
        "created_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "seed": SEED,
        "source": str(RAW_CHILDES.relative_to(ROOT)),
        "tokenizer": str(TOKENIZER_PATH.relative_to(ROOT)),
        "tokenizer_vocab_size": len(tokenizer),
        "special_token_ids": {"bos_as_cls": tokenizer.bos_token_id, "eos_as_sep": tokenizer.eos_token_id, "mask": tokenizer.mask_token_id, "pad": tokenizer.pad_token_id, "unk": tokenizer.unk_token_id},
        "adult_operationalization": "speaker code is any value other than the literal CHI, exactly following the proposal",
        "variation_set_operationalization": "at least one unique content word overlaps; low bin therefore means 0 < Jaccard < 0.2",
        "parse_audit": parse_audit,
        "surface_exposure_audit": exposure_audit,
        "surface_safe_pairs_before_target_filter": len(surface_safe),
        "safe_pairs_before_balance": len(safe),
        "selection_audit": selection_audit,
        "minimum_100_pairs_per_bin_met": minimum_bin_target_met,
        "selected_pairs": len(selected),
        "selected_pair_bins": dict(pair_bins),
        "selected_unique_transcripts": len({r["transcript_id"] for r in selected}),
        "same_speaker_pairs": sum(r["same_speaker"] for r in selected),
        "speaker1_top": dict(speakers.most_common(20)),
        "pair_metrics": {
            "jaccard": stats_summary([r["jaccard"] for r in selected]),
            "utt1_content_occurrences": stats_summary([r["utt1_content_occurrences"] for r in selected]),
            "utt2_content_occurrences": stats_summary([r["utt2_content_occurrences"] for r in selected]),
            "n_overlap_unique": stats_summary([r["n_overlap"] for r in selected]),
            "n_nonoverlap_unique": stats_summary([r["n_nonoverlap"] for r in selected]),
            "intact_seq_len": stats_summary([2 + r["utt1_token_len"] + r["utt2_token_len"] for r in selected]),
            "replaced_seq_len": stats_summary([2 + r["control_token_len"] + r["utt2_token_len"] for r in selected]),
            "control_length_ratio": stats_summary([r["control_length_ratio"] for r in selected]),
        },
        "control_audit": {
            "all_different_pair": all(r["control_pair_id"] != r["pair_id"] for r in selected),
            "all_different_transcript": all(r["control_transcript_id"] != r["transcript_id"] for r in selected),
            "all_within_30pct_token_length": all(0.7 <= r["control_length_ratio"] <= 1.3 for r in selected),
            "zero_target_content_overlap": sum(r["control_target_content_overlap"] == 0 for r in selected),
        },
        "target_audit": target_audit,
        "probe_records": len(records),
        "counts_by_overlap_bin_class_condition": {
            f"{b}|{c}|{d}": counts[(b, c, d)]
            for b in ("low", "partial", "high")
            for c in ("overlap", "nonoverlap")
            for d in ("intact", "replaced")
        },
        "single_token_target_policy": "only content-word occurrences mapping to exactly one tokenizer token are masked; this makes target_token_id unambiguous and prevents within-word subtoken leakage",
        "elapsed_seconds": time.time() - started,
    }

    pairs_path = _public_path('experiments/archive/relation_learning/analysis/childes_variation_set_probe/variation_set_pairs.jsonl')
    records_path = _public_path('experiments/archive/relation_learning/analysis/childes_variation_set_probe/probe_records.jsonl')
    stats_path = _public_path('experiments/archive/relation_learning/analysis/childes_variation_set_probe/probe_stats.json')
    write_jsonl(pairs_path, selected)
    write_jsonl(records_path, records)
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Also save the largest proposal-balanced diagnostic, explicitly marked as
    # exposure-confounded. It is useful for sensitivity analyses but cannot be
    # interpreted as held out.
    diagnostic_pool = [r for r in candidates if r["n_single_token_targets_available"] > 0]
    diagnostic, diagnostic_selection = stratified_select(diagnostic_pool)
    assign_controls(diagnostic)
    diagnostic_records, diagnostic_target_audit = make_probe_records(tokenizer, diagnostic)
    write_jsonl(_public_path('experiments/archive/relation_learning/analysis/childes_variation_set_probe/exposure_confounded_pairs.jsonl'), diagnostic)
    write_jsonl(_public_path('experiments/archive/relation_learning/analysis/childes_variation_set_probe/exposure_confounded_probe_records.jsonl'), diagnostic_records)
    diagnostic_bins = collections.Counter(r["overlap_bin"] for r in diagnostic)
    diagnostic_exposure = collections.Counter(
        arm for r in diagnostic for arm in r["surface_exposure_arms"]
    )
    diagnostic_stats = {
        "status": "EXPOSURE_CONFOUNDED_BALANCED_SENSITIVITY_INSTRUMENT_NOT_HELD_OUT",
        "pairs": len(diagnostic),
        "records": len(diagnostic_records),
        "pair_bins": dict(diagnostic_bins),
        "selection": diagnostic_selection,
        "pairs_surface_exposed_in_any_arm": sum(bool(r["surface_exposure_arms"]) for r in diagnostic),
        "exposure_arm_memberships": dict(diagnostic_exposure),
        "target_audit": diagnostic_target_audit,
    }
    (_public_path('experiments/archive/relation_learning/analysis/childes_variation_set_probe/exposure_confounded_stats.json')).write_text(
        json.dumps(diagnostic_stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = [
        "# Natural CHILDES variation-set probe\n\n",
        f"Built **{len(selected):,}** surface-held-out adult-utterance pairs from **{len({r['transcript_id'] for r in selected}):,}** transcripts, yielding **{len(records):,}** masked-token records. The available pair counts are "
        + ", ".join(f"{b}={pair_bins[b]:,}" for b in ("low", "partial", "high")) + ".\n",
        "\n## What `held out` means\n\n",
        "Neither cleaned utterance in any retained pair occurs as an exact contiguous tokenizer-ID sequence in any actual CLEAN, VIEW, or REPEAT 10M training-pool row. This direct screen is necessary because the file called `heldout_cleanqwen_rows.jsonl` is repacked into the CLEAN control and is therefore a row-holdout for arm construction, not an evaluation holdout. Exact surface absence does not prove semantic independence or absence from tokenizer fitting.\n\n",
        "## Construction\n\n",
        "Transcript boundaries are the `= = = childes/...cha = = =` markers. A candidate joins consecutive non-`CHI` speaker lines within three physical lines; both sides have at least three content-word occurrences, utterance 2 has at least two unique novel content words, at least one unique word overlaps, and the intact sequence is at most 256 model tokens. Content words and Jaccard bins follow the proposal. The literal non-`CHI` rule can include sibling codes such as BRO/SIS; use the stored speaker fields for a canonical-caregiver sensitivity analysis.\n\n",
        "The tokenizer has no named CLS/SEP IDs, so BOS `<s>` is used as conceptual CLS and EOS `</s>` as conceptual SEP. Only whole content-word occurrences represented by one tokenizer token are targets. Each retained target has paired intact/replaced records. Controls are selected from another retained pair and always from a different transcript, within 30% of original context token length, preferring zero lexical overlap with the target utterance.\n\n",
        "## Counts\n\n",
        "| bin | pairs | overlap records/condition | non-overlap records/condition |\n|---|---:|---:|---:|\n",
    ]
    for b in ("low", "partial", "high"):
        lines.append(f"| {b} | {pair_bins[b]} | {counts[(b, 'overlap', 'intact')]} | {counts[(b, 'nonoverlap', 'intact')]} |\n")
    lines.extend([
        "\n## Files\n\n",
        "- `variation_set_pairs.jsonl`: pair text, transcript/speaker provenance, overlap metrics, control assignment, and token lengths.\n",
        "- `probe_records.jsonl`: ready-to-score masked inputs and target metadata.\n",
        "- `probe_stats.json`: full construction, exposure, balance, target, and control audit.\n",
        "- `validation_results.json`: independent invariant checks produced by the validator.\n",
        "- `manual_sample_audit.md`: qualitative review and interpretation cautions from a 30-pair sample.\n",
        "- `exposure_confounded_*`: the largest ≥100/bin sensitivity instrument possible under the structural rules; it is explicitly not held out and must not support the primary causal claim.\n",
        "\n## Feasibility result\n\n",
        f"The requested ≥100 pairs per bin is **{'met' if minimum_bin_target_met else 'not feasible'}** under direct surface-held-out screening. The high-overlap bin has only {pair_bins['high']} retained pairs. The exposure-confounded sensitivity set has "
        + ", ".join(f"{b}={diagnostic_bins[b]}" for b in ("low", "partial", "high"))
        + "; use it only to measure how strongly training exposure changes the readout.\n",
        "\nScore `target_token_id` at `mask_position`, then aggregate paired `intact - replaced` log-probability by model arm, overlap class, and overlap bin. Use pair-clustered or transcript-clustered uncertainty; do not treat the two conditions as independent observations.\n",
    ])
    (_public_path('research/notes/relation_learning/analysis/childes_variation_set_probe/variation_set_summary.md')).write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"pairs": len(selected), "records": len(records), "bins": dict(pair_bins), "exposed": len(exposure_arms), "minimum_bin_target_met": minimum_bin_target_met, "diagnostic_pairs": len(diagnostic), "output": str(OUT_DIR.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
