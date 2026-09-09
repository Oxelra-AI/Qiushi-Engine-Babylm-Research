#!/usr/bin/env python3
"""Independent structural and tokenizer validation for the CHILDES probe."""

from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import collections
import hashlib
import json
import math
import re
import time
from pathlib import Path

from transformers import AutoTokenizer


ROOT = _public_path('.')
HERE = _public_path('experiments/archive/relation_learning/analysis/childes_variation_set_probe')
RAW = _public_path('experiments/archive/initial_model_studies/data/reconstruct_tmp/raw_dataset/childes.train.txt')
TOKENIZER_DIR = _public_path('experiments/archive/frontier_consolidation/data/compliant_tokenizer')
POOL_DIR = _public_path('experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools')
POOLS = {
    "clean": _public_path('experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/cleanqwen_lengthmatched_dose2p64x_10M.jsonl'),
    "view": _public_path('experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/compact_view_dose2p64x_10M.jsonl'),
    "repeat": _public_path('experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/compact_repeat_dose2p64x_10M.jsonl'),
}
WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*")
SPEAKER_RE = re.compile(r"^\*([A-Za-z0-9]+):\t(.*)$")
TRANSCRIPT_RE = re.compile(r"^= = = (childes/.+?\.cha) = = =$", re.I)
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


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def content_set(text: str) -> set[str]:
    return {
        m.group(0).lower().replace("’", "'")
        for m in WORD_RE.finditer(text)
        if len(m.group(0)) >= 2
        and m.group(0).lower().replace("’", "'") not in STOPWORDS
        and not m.group(0).isdigit()
    }


def clean_utterance(text: str) -> str:
    text = BRACKET_RE.sub(" ", text)
    text = CHAT_EVENT_RE.sub(" ", text)
    text = CHAT_CODE_RE.sub(" ", text)
    text = text.replace("<", " ").replace(">", " ")
    text = re.sub(r"(?<!\w)\+(?:/|//|\.{3}|[!?]+)(?!\w)", " ", text)
    return " ".join(text.replace("\u00a0", " ").split()).strip()


def verify_raw_provenance(pairs: list[dict]) -> list[str]:
    wanted = {row["physical_line_1"] for row in pairs} | {row["physical_line_2"] for row in pairs}
    captured: dict[int, dict] = {}
    raw_lines: dict[int, str] = {}
    transcript = ""
    transcript_position = 0
    with RAW.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, raw in enumerate(handle, 1):
            raw = raw.rstrip("\r\n")
            if line_no in wanted:
                raw_lines[line_no] = raw
            boundary = TRANSCRIPT_RE.match(raw)
            if boundary:
                transcript = boundary.group(1)
                transcript_position = 0
                continue
            speaker = SPEAKER_RE.match(raw)
            if speaker:
                transcript_position += 1
                if line_no in wanted:
                    code, text = speaker.groups()
                    captured[line_no] = {
                        "transcript_id": transcript,
                        "transcript_position": transcript_position,
                        "speaker": code,
                        "text": clean_utterance(text),
                    }
    failures = []
    # A second tiny read supplies intervening lines only for selected pairs.
    selected_lines = set()
    for row in pairs:
        selected_lines.update(range(row["physical_line_1"], row["physical_line_2"] + 1))
    with RAW.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, raw in enumerate(handle, 1):
            if line_no in selected_lines:
                raw_lines[line_no] = raw.rstrip("\r\n")
    for row in pairs:
        for suffix in ("1", "2"):
            line_no = row[f"physical_line_{suffix}"]
            got = captured.get(line_no)
            expected = {
                "transcript_id": row["transcript_id"],
                "transcript_position": row[f"transcript_position_{suffix}"],
                "speaker": row[f"speaker{suffix}"],
                "text": row[f"utt{suffix}_text"],
            }
            if got != expected:
                failures.append(f"raw provenance mismatch {row['pair_id']} side {suffix}")
        for line_no in range(row["physical_line_1"] + 1, row["physical_line_2"]):
            speaker = SPEAKER_RE.match(raw_lines.get(line_no, ""))
            if speaker and speaker.group(1) != "CHI":
                failures.append(f"intervening adult line {row['pair_id']} at {line_no}")
    return failures


def expected_bin(value: float) -> str:
    return "high" if value >= 0.5 else "partial" if value >= 0.2 else "low"


class TokenAho:
    def __init__(self, patterns: dict[tuple[object, ...], set[str]]):
        self.go = [{}]
        self.fail = [0]
        self.out = [set()]
        for toks, labels in patterns.items():
            node = 0
            for tok in toks:
                if tok not in self.go[node]:
                    self.go[node][tok] = len(self.go)
                    self.go.append({})
                    self.fail.append(0)
                    self.out.append(set())
                node = self.go[node][tok]
            self.out[node].update(labels)
        queue = collections.deque(self.go[0].values())
        while queue:
            node = queue.popleft()
            for tok, child in self.go[node].items():
                queue.append(child)
                fallback = self.fail[node]
                while fallback and tok not in self.go[fallback]:
                    fallback = self.fail[fallback]
                self.fail[child] = self.go[fallback].get(tok, 0)
                self.out[child].update(self.out[self.fail[child]])

    def scan(self, tokens: list[object]) -> set[str]:
        node = 0
        found = set()
        for tok in tokens:
            while node and tok not in self.go[node]:
                node = self.fail[node]
            node = self.go[node].get(tok, 0)
            found.update(self.out[node])
        return found


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_instrument(pair_path: Path, record_path: Path, tokenizer, require_unexposed: bool) -> dict:
    pairs = load_jsonl(pair_path)
    records = load_jsonl(record_path)
    failures: list[str] = []

    pair_ids = [row["pair_id"] for row in pairs]
    record_ids = [row["record_id"] for row in records]
    if len(pair_ids) != len(set(pair_ids)):
        failures.append("duplicate pair_id")
    if len(record_ids) != len(set(record_ids)):
        failures.append("duplicate record_id")
    pair_map = {row["pair_id"]: row for row in pairs}
    provenance_keys = [
        (row["transcript_id"], row["physical_line_1"], row["physical_line_2"])
        for row in pairs
    ]
    if len(provenance_keys) != len(set(provenance_keys)):
        failures.append("duplicate transcript/line pair")
    failures.extend(verify_raw_provenance(pairs))

    pair_checks = collections.Counter()
    for row in pairs:
        c1, c2 = content_set(row["utt1_text"]), content_set(row["utt2_text"])
        overlap, nonoverlap = c1 & c2, c2 - c1
        jac = len(overlap) / len(c1 | c2)
        if sorted(c1) != row["utt1_content_words"] or sorted(c2) != row["utt2_content_words"]:
            failures.append(f"content set mismatch {row['pair_id']}")
        if sorted(overlap) != row["overlap_words"] or sorted(nonoverlap) != row["nonoverlap_words"]:
            failures.append(f"overlap mismatch {row['pair_id']}")
        if abs(jac - row["jaccard"]) > 1e-12 or expected_bin(jac) != row["overlap_bin"]:
            failures.append(f"Jaccard/bin mismatch {row['pair_id']}")
        if row["speaker1"] == "CHI" or row["speaker2"] == "CHI":
            failures.append(f"CHI speaker in adult pair {row['pair_id']}")
        if not (1 <= row["physical_line_gap"] <= 3):
            failures.append(f"physical gap invalid {row['pair_id']}")
        if not overlap or len(nonoverlap) < 2:
            failures.append(f"variation-set lexical constraints fail {row['pair_id']}")
        if row["control_pair_id"] == row["pair_id"] or row["control_transcript_id"] == row["transcript_id"]:
            failures.append(f"control provenance invalid {row['pair_id']}")
        if not (0.7 - 1e-12 <= row["control_length_ratio"] <= 1.3 + 1e-12):
            failures.append(f"control length invalid {row['pair_id']}")
        if require_unexposed and row.get("surface_exposure_arms"):
            failures.append(f"primary pair marked exposed {row['pair_id']}")
        pair_checks["checked"] += 1

    by_target: dict[tuple, set[str]] = collections.defaultdict(set)
    per_pair_condition = collections.Counter()
    vocab = len(tokenizer)
    special_ids = set(tokenizer.all_special_ids)
    for row in records:
        pair = pair_map.get(row["pair_id"])
        if pair is None:
            failures.append(f"orphan record {row['record_id']}")
            continue
        ids = row["input_ids"]
        if row["seq_len"] != len(ids) or len(ids) != len(row["attention_mask"]):
            failures.append(f"length mismatch {row['record_id']}")
        if len(ids) > 256 or any(x != 1 for x in row["attention_mask"]):
            failures.append(f"attention/max-length mismatch {row['record_id']}")
        if any(not isinstance(x, int) or x < 0 or x >= vocab for x in ids):
            failures.append(f"out-of-vocab input id {row['record_id']}")
        pos = row["mask_position"]
        if not (0 <= pos < len(ids)) or ids[pos] != tokenizer.mask_token_id:
            failures.append(f"mask position mismatch {row['record_id']}")
            continue
        if ids.count(tokenizer.mask_token_id) != 1:
            failures.append(f"not exactly one mask {row['record_id']}")
        target_id = row["target_token_id"]
        if not isinstance(target_id, int) or not (0 <= target_id < vocab) or target_id in special_ids:
            failures.append(f"invalid target ID {row['record_id']}")
        context = pair["utt1_text"] if row["condition"] == "intact" else pair["control_utt_text"]
        context_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
        utt2_ids = tokenizer(pair["utt2_text"], add_special_tokens=False)["input_ids"]
        canonical = [tokenizer.bos_token_id] + context_ids + utt2_ids + [tokenizer.eos_token_id]
        if row["condition_context_token_len"] != len(context_ids) or row["utt2_token_len"] != len(utt2_ids):
            failures.append(f"component length mismatch {row['record_id']}")
        restored = list(ids)
        restored[pos] = target_id
        if restored != canonical:
            failures.append(f"masked sequence cannot reconstruct canonical input {row['record_id']}")
        if pos != 1 + len(context_ids) + row["utt2_token_index"]:
            failures.append(f"target index shift mismatch {row['record_id']}")
        raw_target = pair["utt2_text"][row["target_char_start"]:row["target_char_end"]].lower().replace("’", "'")
        if raw_target != row["target_word"] or raw_target not in content_set(pair["utt2_text"]):
            failures.append(f"target char/word mismatch {row['record_id']}")
        expected_class = "overlap" if raw_target in set(pair["overlap_words"]) else "nonoverlap"
        if row["overlap_class"] != expected_class:
            failures.append(f"target overlap class mismatch {row['record_id']}")
        by_target[(row["pair_id"], row["target_occurrence_index"], row["target_token_id"], row["overlap_class"])].add(row["condition"])
        per_pair_condition[(row["pair_id"], row["condition"])] += 1

    if any(conditions != {"intact", "replaced"} for conditions in by_target.values()):
        failures.append("one or more targets lack paired intact/replaced records")
    if any(count > 3 for count in per_pair_condition.values()):
        failures.append("more than three targets in a pair-condition")
    if any((pair_id, condition) not in per_pair_condition for pair_id in pair_ids for condition in ("intact", "replaced")):
        failures.append("one or more pairs lack a condition")

    exposure_hits = {arm: set() for arm in POOLS}
    if require_unexposed:
        patterns: dict[tuple[int, ...], set[str]] = collections.defaultdict(set)
        for row in pairs:
            patterns[tuple(tokenizer(row["utt1_text"], add_special_tokens=False)["input_ids"])].add(row["pair_id"])
            patterns[tuple(tokenizer(row["utt2_text"], add_special_tokens=False)["input_ids"])].add(row["pair_id"])
        matcher = TokenAho(patterns)
        for arm, path in POOLS.items():
            with path.open("r", encoding="utf-8") as handle:
                batch = []
                for line in handle:
                    if line.strip():
                        batch.append(str(json.loads(line)["text"]))
                    if len(batch) == 256:
                        for ids in tokenizer(batch, add_special_tokens=False)["input_ids"]:
                            exposure_hits[arm].update(matcher.scan(ids))
                        batch = []
                if batch:
                    for ids in tokenizer(batch, add_special_tokens=False)["input_ids"]:
                        exposure_hits[arm].update(matcher.scan(ids))
        if any(exposure_hits.values()):
            failures.append("primary surface-held-out scan found exposure")

    bins = collections.Counter(row["overlap_bin"] for row in pairs)
    classes = collections.Counter((row["overlap_bin"], row["overlap_class"], row["condition"]) for row in records)
    return {
        "status": "PASS" if not failures else "FAIL",
        "pairs": len(pairs),
        "records": len(records),
        "pair_bins": dict(bins),
        "unique_transcripts": len({row["transcript_id"] for row in pairs}),
        "record_counts": {"|".join(key): value for key, value in sorted(classes.items())},
        "paired_targets": len(by_target),
        "surface_exposure_hits": {arm: len(hits) for arm, hits in exposure_hits.items()},
        "checks": {
            "unique_pair_ids": len(pair_ids) == len(set(pair_ids)),
            "unique_record_ids": len(record_ids) == len(set(record_ids)),
            "unique_provenance_keys": len(provenance_keys) == len(set(provenance_keys)),
            "all_targets_condition_paired": all(v == {"intact", "replaced"} for v in by_target.values()),
            "max_three_masks_per_pair_condition": all(v <= 3 for v in per_pair_condition.values()),
            "surface_unexposed_all_arms": not any(exposure_hits.values()) if require_unexposed else None,
            "raw_line_provenance_and_adjacency": not any("provenance" in x or "intervening adult" in x for x in failures),
        },
        "failures": failures[:100],
        "failure_count": len(failures),
    }


def main() -> None:
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True, use_fast=True)
    primary = validate_instrument(_public_path('experiments/archive/relation_learning/analysis/childes_variation_set_probe/variation_set_pairs.jsonl'), _public_path('experiments/archive/relation_learning/analysis/childes_variation_set_probe/probe_records.jsonl'), tokenizer, True)
    diagnostic = validate_instrument(_public_path('experiments/archive/relation_learning/analysis/childes_variation_set_probe/exposure_confounded_pairs.jsonl'), _public_path('experiments/archive/relation_learning/analysis/childes_variation_set_probe/exposure_confounded_probe_records.jsonl'), tokenizer, False)
    result = {
        "status": "PASS" if primary["status"] == diagnostic["status"] == "PASS" else "FAIL",
        "validated_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "primary": primary,
        "exposure_confounded_sensitivity": diagnostic,
        "file_sha256": {
            path.name: sha256(path)
            for path in [
                _public_path('experiments/archive/relation_learning/analysis/childes_variation_set_probe/variation_set_pairs.jsonl'),
                _public_path('experiments/archive/relation_learning/analysis/childes_variation_set_probe/probe_records.jsonl'),
                _public_path('experiments/archive/relation_learning/analysis/childes_variation_set_probe/probe_stats.json'),
                _public_path('experiments/archive/relation_learning/analysis/childes_variation_set_probe/exposure_confounded_pairs.jsonl'),
                _public_path('experiments/archive/relation_learning/analysis/childes_variation_set_probe/exposure_confounded_probe_records.jsonl'),
                _public_path('experiments/archive/relation_learning/analysis/childes_variation_set_probe/exposure_confounded_stats.json'),
            ]
        },
        "elapsed_seconds": time.time() - started,
    }
    (_public_path('experiments/archive/relation_learning/analysis/childes_variation_set_probe/validation_results.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
