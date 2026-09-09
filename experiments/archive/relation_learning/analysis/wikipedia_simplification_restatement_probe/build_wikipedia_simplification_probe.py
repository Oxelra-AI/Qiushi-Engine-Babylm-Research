#!/usr/bin/env python3
"""Construct a CPU-only WikiLarge-clean natural restatement T/U/N probe."""

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
import os
import random
import re
import statistics
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


HERE = _public_path('experiments/archive/relation_learning/analysis/wikipedia_simplification_restatement_probe')
ROOT = _public_path('.')
HF_CACHE = _public_path('experiments/archive/relation_learning/analysis/wikipedia_simplification_restatement_probe/_hf_cache')
MODEL_CACHE = _public_path('experiments/archive/relation_learning/analysis/wikipedia_simplification_restatement_probe/_model_cache/sentence_transformers')
os.environ.setdefault("HF_HOME", str(HF_CACHE))
os.environ.setdefault("HF_DATASETS_CACHE", str(_public_path('experiments/archive/relation_learning/analysis/wikipedia_simplification_restatement_probe/_hf_cache/datasets')))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(MODEL_CACHE))

from datasets import load_dataset  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402


DATASET_NAME = "eilamc14/wikilarge-clean"
DATASET_REVISION = "216fedb399e10141b390c8b89039737c934d43be"
DATASET_URL = "https://huggingface.co/datasets/eilamc14/wikilarge-clean"
SEMANTIC_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SEMANTIC_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
NLI_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"
NLI_MODEL_REVISION = "b95119ce93d3e065de6214e38cd4a97b0f2f2c6d"
NLI_ENTAILMENT_MIN = 0.80
TOKENIZER_PATH = _public_path('experiments/archive/frontier_consolidation/data/compliant_tokenizer')
POOL_DIR = _public_path('experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools')
ARM_POOLS = {
    "clean": _public_path('experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/cleanqwen_lengthmatched_dose2p64x_10M.jsonl'),
    "view": _public_path('experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/compact_view_dose2p64x_10M.jsonl'),
    "repeat": _public_path('experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/compact_repeat_dose2p64x_10M.jsonl'),
}
NEUTRAL_ROWS = _public_path('experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/heldout_cleanqwen_rows.jsonl')
CHILDES_STATS = _public_path('experiments/archive/relation_learning/analysis/childes_variation_set_probe/probe_stats.json')

SEED = 90621021
MAX_SEQ_LEN = 256
SEMANTIC_SIMILARITY_MIN = 0.65
SHORTLIST_PER_BIN = 1800
TARGET_PER_BIN = 400
DATASET_BINS = ("low", "medium", "high")
WORD_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.UNICODE)
NUMBER_RE = re.compile(r"\d+(?:[.,:/-]\d+)*")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
BAD_TEXT_MARKERS = ("Ã", "Â", "â", "Ë", "Å", "�", "\\", "& ndash", "& #")

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "than", "that", "this", "these", "those",
    "of", "in", "on", "at", "to", "for", "from", "with", "without", "by", "as", "is", "are", "was",
    "were", "be", "been", "being", "it", "its", "they", "them", "their", "he", "she", "his", "her",
    "we", "you", "i", "not", "no", "do", "does", "did", "can", "could", "would", "should", "will",
    "have", "has", "had", "just", "so", "very", "also", "about", "more", "some", "any", "all", "each",
    "every", "both", "few", "many", "much", "such", "own", "other", "up", "out",
}


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def normalize(text: str) -> str:
    return " ".join(str(text).replace("\u00a0", " ").split()).strip()


def stable_hex(*parts: object, n: int = 20) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:n]


def content_occurrences(text: str) -> list[dict]:
    out = []
    for match in WORD_RE.finditer(text):
        word = match.group(0).lower().replace("’", "'")
        if len(word) >= 2 and word not in STOPWORDS:
            out.append({"word": word, "start": match.start(), "end": match.end()})
    return out


def content_set(text: str) -> set[str]:
    return {x["word"] for x in content_occurrences(text)}


def numbers(text: str) -> set[str]:
    return set(NUMBER_RE.findall(text))


def overlap_bin(jaccard: float) -> str:
    # The low/medium boundary is 0.25 so that "low" still denotes no more than
    # one shared type per four union types while supporting a useful denominator.
    return "high" if jaccard >= 0.5 else "medium" if jaccard >= 0.25 else "low"


def noninitial_capitalized_content(text: str) -> set[str]:
    """Conservative proxy for named entities introduced only by the target.

    Sentence-initial capitalization is ignored. This deliberately sacrifices
    recall: an aligned target that introduces a name not stated in the source is
    unsuitable for a source-conditioned restatement test even if it is true in
    the wider article.
    """
    out = set()
    for match in WORD_RE.finditer(text):
        surface = match.group(0)
        prefix = text[:match.start()].rstrip(" \t\r\n\"'“”‘’([{—–-")
        sentence_initial = not prefix or prefix[-1:] in ".?!"
        word = surface.lower().replace("’", "'")
        if not sentence_initial and word not in STOPWORDS and (surface[0].isupper() or surface.isupper()):
            out.add(word)
    return out


def stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(values), "min": min(values), "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))], "max": max(values),
    }


def single_token_occurrences(tokenizer, text: str) -> tuple[list[int], list[dict]]:
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = [int(x) for x in enc["input_ids"]]
    offsets = [tuple(x) for x in enc["offset_mapping"]]
    out = []
    for occurrence_index, occ in enumerate(content_occurrences(text)):
        covering = [i for i, (a, b) in enumerate(offsets) if a < occ["end"] and b > occ["start"]]
        if len(covering) != 1:
            continue
        token_index = covering[0]
        a, b = offsets[token_index]
        if b != occ["end"] or a > occ["start"] or text[a:occ["start"]].strip():
            continue
        out.append({
            **occ, "occurrence_index": occurrence_index, "target_token_index": token_index,
            "target_token_id": ids[token_index],
        })
    return ids, out


def lexical_candidates(dataset, tokenizer) -> tuple[list[dict], dict]:
    rows = []
    audit = collections.Counter()
    split_inputs = {}
    for split in dataset:
        split_inputs[split] = len(dataset[split])
        for row_index, obj in enumerate(dataset[split]):
            audit["input_rows"] += 1
            source = normalize(obj["source"])
            target = normalize(obj["target"])
            if not source or not target or source.casefold() == target.casefold():
                audit["reject_empty_or_identical"] += 1
                continue
            if any(marker in source or marker in target for marker in BAD_TEXT_MARKERS):
                audit["reject_encoding_or_markup_noise"] += 1
                continue
            if source[-1:] not in ".?!" or target[-1:] not in ".?!":
                audit["reject_sentence_boundary"] += 1
                continue
            source_occ = content_occurrences(source)
            target_occ = content_occurrences(target)
            source_content = {x["word"] for x in source_occ}
            target_content = {x["word"] for x in target_occ}
            shared = source_content & target_content
            novel = target_content - source_content
            if len(source_content) < 5 or len(target_content) < 4:
                audit["reject_short_content"] += 1
                continue
            if len(shared) < 2 or len(novel) < 2:
                audit["reject_insufficient_shared_or_novel_content"] += 1
                continue
            novel_capitalized = noninitial_capitalized_content(target) - source_content
            if novel_capitalized:
                audit["reject_target_introduced_capitalized_content"] += 1
                continue
            union = source_content | target_content
            jaccard = len(shared) / len(union)
            target_recall = len(shared) / len(target_content)
            source_recall = len(shared) / len(source_content)
            if jaccard >= 0.9 or target_recall < 0.4 or source_recall < 0.15:
                audit["reject_lexical_relation"] += 1
                continue
            if not numbers(target).issubset(numbers(source)):
                audit["reject_novel_number"] += 1
                continue
            source_ids = [int(x) for x in tokenizer(source, add_special_tokens=False)["input_ids"]]
            target_ids, single_occ = single_token_occurrences(tokenizer, target)
            if len(source_ids) < 6 or len(target_ids) < 5 or len(source_ids) + len(target_ids) + 2 > MAX_SEQ_LEN:
                audit["reject_model_token_length"] += 1
                continue
            length_ratio = len(target_ids) / len(source_ids)
            if not 0.4 <= length_ratio <= 1.1:
                audit["reject_length_ratio"] += 1
                continue
            source_id_set = set(source_ids)
            overlap_targets = [x for x in single_occ if x["target_token_id"] in source_id_set]
            nonoverlap_targets = [x for x in single_occ if x["target_token_id"] not in source_id_set]
            if not overlap_targets or len({x["word"] for x in nonoverlap_targets}) < 2:
                audit["reject_single_token_target_denominator"] += 1
                continue
            pair_id = "wikilarge_" + stable_hex(DATASET_REVISION, split, row_index)
            rows.append({
                "pair_id": pair_id, "dataset_name": DATASET_NAME, "dataset_revision": DATASET_REVISION,
                "dataset_split": split, "dataset_row_index": row_index,
                "article_id": None, "article_title": None,
                "source_text": source, "target_text": target,
                "source_token_ids": source_ids, "target_token_ids": target_ids,
                "source_token_len": len(source_ids), "target_token_len": len(target_ids),
                "source_content_words": sorted(source_content), "target_content_words": sorted(target_content),
                "overlap_words": sorted(shared), "nonoverlap_words": sorted(novel),
                "n_overlap": len(shared), "n_nonoverlap": len(novel),
                "content_jaccard": jaccard, "target_content_recall": target_recall,
                "source_content_recall": source_recall, "target_source_token_length_ratio": length_ratio,
                "target_introduced_capitalized_content": sorted(novel_capitalized),
                "overlap_bin": overlap_bin(jaccard), "_single_occ": single_occ,
            })
            audit["lexically_eligible"] += 1
    return rows, {"split_input_rows": split_inputs, "first_fail_counts": dict(audit)}


def semantic_filter(rows: list[dict], skip: bool) -> tuple[list[dict], dict]:
    if skip:
        for row in rows:
            row["semantic_similarity"] = None
        return rows, {"skipped": True, "retained": len(rows)}
    model = SentenceTransformer(
        SEMANTIC_MODEL, revision=SEMANTIC_MODEL_REVISION, cache_folder=str(MODEL_CACHE), device="cpu"
    )
    source_embeddings = model.encode(
        [r["source_text"] for r in rows], batch_size=256, normalize_embeddings=True,
        show_progress_bar=True, convert_to_numpy=True,
    )
    target_embeddings = model.encode(
        [r["target_text"] for r in rows], batch_size=256, normalize_embeddings=True,
        show_progress_bar=True, convert_to_numpy=True,
    )
    similarities = np.sum(source_embeddings * target_embeddings, axis=1)
    retained = []
    before = collections.Counter(r["overlap_bin"] for r in rows)
    rejected = collections.Counter()
    for row, similarity in zip(rows, similarities):
        row["semantic_similarity"] = float(similarity)
        if similarity < SEMANTIC_SIMILARITY_MIN:
            rejected[row["overlap_bin"]] += 1
        else:
            retained.append(row)
    return retained, {
        "model": SEMANTIC_MODEL, "model_revision": SEMANTIC_MODEL_REVISION,
        "minimum_cosine_similarity": SEMANTIC_SIMILARITY_MIN,
        "before_by_bin": dict(before), "rejected_by_bin": dict(rejected),
        "retained_by_bin": dict(collections.Counter(r["overlap_bin"] for r in retained)),
        "similarity_all_lexical_candidates": stats([float(x) for x in similarities]),
    }


def entailment_filter(rows: list[dict], skip: bool) -> tuple[list[dict], dict]:
    """Require the complex source to entail the simplified target.

    MiniLM cosine catches gross misalignment but not polarity, role, or location
    substitutions. This directional NLI gate is therefore applied after the
    embedding gate. Scores are an automated safeguard, not gold annotations.
    """
    if skip:
        for row in rows:
            row.update({"nli_contradiction_probability": None, "nli_entailment_probability": None,
                        "nli_neutral_probability": None, "nli_predicted_label": None})
        return rows, {"skipped": True, "retained": len(rows)}
    cache = _public_path('experiments/archive/relation_learning/analysis/wikipedia_simplification_restatement_probe/_model_cache/nli')
    nli_tokenizer = AutoTokenizer.from_pretrained(
        NLI_MODEL, revision=NLI_MODEL_REVISION, cache_dir=str(cache), use_fast=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        NLI_MODEL, revision=NLI_MODEL_REVISION, cache_dir=str(cache),
    ).to("cpu").eval()
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(rows), 64):
            batch = rows[start:start + 64]
            features = nli_tokenizer(
                [row["source_text"] for row in batch],
                [row["target_text"] for row in batch],
                padding=True, truncation=True, max_length=320, return_tensors="pt",
            )
            probabilities.extend(model(**features).logits.softmax(dim=-1).cpu().tolist())
    labels = ("contradiction", "entailment", "neutral")
    before = collections.Counter(row["overlap_bin"] for row in rows)
    rejected = collections.Counter(); retained = []
    for row, probability in zip(rows, probabilities):
        row["nli_contradiction_probability"] = float(probability[0])
        row["nli_entailment_probability"] = float(probability[1])
        row["nli_neutral_probability"] = float(probability[2])
        row["nli_predicted_label"] = labels[int(np.argmax(probability))]
        if probability[1] < NLI_ENTAILMENT_MIN:
            rejected[row["overlap_bin"]] += 1
        else:
            retained.append(row)
    return retained, {
        "model": NLI_MODEL, "model_revision": NLI_MODEL_REVISION,
        "label_order": list(labels), "direction": "source premise -> target hypothesis",
        "minimum_entailment_probability": NLI_ENTAILMENT_MIN,
        "before_by_bin": dict(before), "rejected_by_bin": dict(rejected),
        "retained_by_bin": dict(collections.Counter(row["overlap_bin"] for row in retained)),
        "entailment_probability_all_semantic_candidates": stats([float(p[1]) for p in probabilities]),
        "caveat": "MNLI/SNLI NLI probability is an automated filter, not a gold semantic judgment",
    }


def shortlist(rows: list[dict]) -> tuple[list[dict], dict]:
    out = []
    audit = {}
    for bin_name in DATASET_BINS:
        items = [r for r in rows if r["overlap_bin"] == bin_name]
        external = [r for r in items if r["dataset_split"] != "train"]
        train = [r for r in items if r["dataset_split"] == "train"]
        external.sort(key=lambda r: stable_hex(SEED, "external", r["pair_id"]))
        train.sort(key=lambda r: stable_hex(SEED, "train", r["pair_id"]))
        chosen = (external + train)[:SHORTLIST_PER_BIN]
        out.extend(chosen)
        audit[bin_name] = {
            "available": len(items), "external_available": len(external), "shortlisted": len(chosen),
            "shortlisted_splits": dict(collections.Counter(r["dataset_split"] for r in chosen)),
        }
    return out, audit


class TokenAho:
    def __init__(self, patterns: dict[tuple[int, ...], set[str]]):
        self.go: list[dict[int, int]] = [{}]
        self.fail = [0]
        self.out = [set()]
        for pattern, labels in patterns.items():
            state = 0
            for token in pattern:
                if token not in self.go[state]:
                    self.go[state][token] = len(self.go)
                    self.go.append({}); self.fail.append(0); self.out.append(set())
                state = self.go[state][token]
            self.out[state].update(labels)
        queue = collections.deque(self.go[0].values())
        while queue:
            state = queue.popleft()
            for token, child in self.go[state].items():
                queue.append(child)
                fallback = self.fail[state]
                while fallback and token not in self.go[fallback]:
                    fallback = self.fail[fallback]
                self.fail[child] = self.go[fallback].get(token, 0)
                self.out[child].update(self.out[self.fail[child]])

    def scan(self, tokens: Iterable[int]) -> set[str]:
        state = 0; found = set()
        for token in tokens:
            while state and token not in self.go[state]:
                state = self.fail[state]
            state = self.go[state].get(token, 0)
            found.update(self.out[state])
        return found


def exposure_scan(rows: list[dict], tokenizer) -> dict:
    patterns: dict[tuple[int, ...], set[str]] = collections.defaultdict(set)
    for row in rows:
        pid = row["pair_id"]
        patterns[tuple(row["target_token_ids"])].add(pid + "|target")
        patterns[tuple(row["source_token_ids"] + row["target_token_ids"])].add(pid + "|pair")
        patterns[tuple(row["source_token_ids"])].add(pid + "|source")
    matcher = TokenAho(patterns)
    hit_arms: dict[str, dict[str, set[str]]] = {
        row["pair_id"]: {"target": set(), "pair": set(), "source": set()} for row in rows
    }
    scanned = {}
    for arm, path in ARM_POOLS.items():
        nrows = nwords = 0; batch = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    text = str(json.loads(line)["text"])
                    batch.append(text); nrows += 1; nwords += len(text.split())
                if len(batch) == 256:
                    for ids in tokenizer(batch, add_special_tokens=False)["input_ids"]:
                        for label in matcher.scan(ids):
                            pid, kind = label.rsplit("|", 1); hit_arms[pid][kind].add(arm)
                    batch = []
            if batch:
                for ids in tokenizer(batch, add_special_tokens=False)["input_ids"]:
                    for label in matcher.scan(ids):
                        pid, kind = label.rsplit("|", 1); hit_arms[pid][kind].add(arm)
        scanned[arm] = {"rows": nrows, "whitespace_words": nwords}
    for row in rows:
        hits = hit_arms[row["pair_id"]]
        row["target_surface_exposure_arms"] = sorted(hits["target"])
        row["full_pair_surface_exposure_arms"] = sorted(hits["pair"])
        row["source_surface_exposure_arms"] = sorted(hits["source"])
    return {
        "criterion": "reject target or concatenated T-body exact tokenizer-ID subsequence exposure; source-only exposure is retained but reported",
        "patterns": len(patterns), "automaton_states": len(matcher.go), "scanned": scanned,
        "target_hits_by_arm": {a: sum(a in r["target_surface_exposure_arms"] for r in rows) for a in ARM_POOLS},
        "pair_hits_by_arm": {a: sum(a in r["full_pair_surface_exposure_arms"] for r in rows) for a in ARM_POOLS},
        "source_hits_by_arm": {a: sum(a in r["source_surface_exposure_arms"] for r in rows) for a in ARM_POOLS},
    }


def choose_balanced(rows: list[dict]) -> tuple[list[dict], dict]:
    safe_by_bin = {
        b: [r for r in rows if r["overlap_bin"] == b and not r["target_surface_exposure_arms"] and not r["full_pair_surface_exposure_arms"]]
        for b in DATASET_BINS
    }
    n_per_bin = min(TARGET_PER_BIN, *(len(v) for v in safe_by_bin.values()))
    if n_per_bin < 100:
        raise RuntimeError(f"fewer than 100 surface-safe rows in a bin: { {b:len(v) for b,v in safe_by_bin.items()} }")
    selected = []
    audit = {"n_per_bin": n_per_bin, "bins": {}}
    for bin_name, items in safe_by_bin.items():
        external = [r for r in items if r["dataset_split"] != "train"]
        train = [r for r in items if r["dataset_split"] == "train"]
        external.sort(key=lambda r: stable_hex(SEED, "select_external", r["pair_id"]))
        train.sort(key=lambda r: stable_hex(SEED, "select_train", r["pair_id"]))
        chosen = (external + train)[:n_per_bin]
        selected.extend(chosen)
        audit["bins"][bin_name] = {
            "surface_safe_available": len(items), "selected": len(chosen),
            "selected_splits": dict(collections.Counter(r["dataset_split"] for r in chosen)),
        }
    selected.sort(key=lambda r: r["pair_id"])
    return selected, audit


def choose_targets(row: dict) -> list[dict]:
    source_content = set(row["source_content_words"])
    source_ids = set(row["source_token_ids"])
    overlap = [x for x in row["_single_occ"] if x["target_token_id"] in source_ids]
    nonoverlap = [x for x in row["_single_occ"] if x["target_token_id"] not in source_ids]
    overlap.sort(key=lambda x: stable_hex(SEED, row["pair_id"], "overlap", x["occurrence_index"]))
    nonoverlap.sort(key=lambda x: stable_hex(SEED, row["pair_id"], "nonoverlap", x["occurrence_index"]))
    chosen_non = []; seen_words = set()
    for target in nonoverlap:
        if target["word"] not in seen_words:
            chosen_non.append(target); seen_words.add(target["word"])
        if len(chosen_non) == 2:
            break
    if not overlap or len(chosen_non) != 2:
        raise AssertionError("target denominator drift")
    chosen = [dict(overlap[0], token_class="overlap")] + [dict(x, token_class="nonoverlap") for x in chosen_non]
    chosen.sort(key=lambda x: x["target_token_index"])
    for target in chosen:
        target["target_key"] = f"{row['pair_id']}:{target['occurrence_index']}"
        target["token_id_occurs_in_source"] = target["target_token_id"] in source_ids
        target["surface_word_class"] = "overlap" if target["word"] in source_content else "nonoverlap"
    return chosen


def assign_unrelated_controls(selected: list[dict], source_pool: list[dict]) -> dict:
    buckets: dict[int, list[dict]] = collections.defaultdict(list)
    for row in source_pool:
        buckets[row["source_token_len"]].append(row)
    overlap_counts = []
    for row in selected:
        target_content = set(row["target_content_words"])
        choices = []
        for control in buckets[row["source_token_len"]]:
            if control["pair_id"] == row["pair_id"] or control["source_text"] == row["source_text"]:
                continue
            n_overlap = len(target_content & set(control["source_content_words"]))
            choices.append((n_overlap, stable_hex(SEED, "U", row["pair_id"], control["pair_id"]), control))
        if not choices:
            raise RuntimeError(f"no exact-length U control for {row['pair_id']}")
        n_overlap, _, control = min(choices, key=lambda x: x[:2])
        row["unrelated_source_id"] = control["pair_id"]
        row["unrelated_source_dataset_split"] = control["dataset_split"]
        row["unrelated_source_dataset_row_index"] = control["dataset_row_index"]
        row["unrelated_source_text"] = control["source_text"]
        row["unrelated_source_token_ids"] = control["source_token_ids"]
        row["unrelated_target_content_overlap_words"] = sorted(target_content & set(control["source_content_words"]))
        overlap_counts.append(n_overlap)
    return {
        "all_different_pair": all(r["unrelated_source_id"] != r["pair_id"] for r in selected),
        "all_exact_token_length": all(len(r["unrelated_source_token_ids"]) == r["source_token_len"] for r in selected),
        "zero_target_content_overlap": sum(not r["unrelated_target_content_overlap_words"] for r in selected),
        "overlap_word_count": stats(overlap_counts),
    }


def neutral_sentence_pool(tokenizer) -> tuple[list[dict], dict]:
    candidates = []; input_rows = 0; rejected_fragment = 0
    with NEUTRAL_ROWS.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line); input_rows += 1
            text = normalize(obj["text"])
            sentences = [normalize(s) for s in SENTENCE_SPLIT_RE.split(text) if normalize(s)]
            for sentence_index, sentence in enumerate(sentences):
                first_word = WORD_RE.search(sentence)
                # The BabyLM files are fixed-size packed rows, so the first and
                # last pieces can be cross-row fragments. Retain only segments
                # with a sentence terminator and an uppercase lexical start (or
                # a CHILDES-style speaker marker).
                if (sentence[-1:] not in ".?!" or first_word is None or
                        not (first_word.group(0)[0].isupper() or sentence.startswith("*"))):
                    rejected_fragment += 1
                    continue
                ids = [int(x) for x in tokenizer(sentence, add_special_tokens=False)["input_ids"]]
                if 6 <= len(ids) <= 180:
                    candidates.append({
                        "neutral_source_id": f"babyrow:{obj.get('example_id')}:sent:{sentence_index}",
                        "neutral_row_example_id": obj.get("example_id"), "neutral_row_source": obj.get("source"),
                        "neutral_source_text": sentence, "neutral_source_token_ids": ids,
                        "content": content_set(sentence), "neutral_kind": "punctuation_delimited_sentence_from_rowholdout_slice",
                    })
    return candidates, {
        "input_path": rel(NEUTRAL_ROWS), "input_rows": input_rows, "sentence_candidates": len(candidates),
        "rejected_possible_packed_row_fragments": rejected_fragment,
        "note": "rowholdout slice is used only as an ordinary N-register source pool; it is not claimed evaluation-held-out",
    }


def assign_neutral_controls(selected: list[dict], neutral_pool: list[dict]) -> dict:
    buckets: dict[int, list[dict]] = collections.defaultdict(list)
    for item in neutral_pool:
        buckets[len(item["neutral_source_token_ids"])].append(item)
    used = collections.Counter(); overlap_counts = []
    for row in selected:
        target_content = set(row["target_content_words"])
        choices = []
        for control in buckets[row["source_token_len"]]:
            n_overlap = len(target_content & control["content"])
            reuse = used[control["neutral_source_id"]]
            choices.append((n_overlap, reuse, stable_hex(SEED, "N", row["pair_id"], control["neutral_source_id"]), control))
        if not choices:
            raise RuntimeError(f"no exact-length neutral sentence for {row['pair_id']} length={row['source_token_len']}")
        n_overlap, _, _, control = min(choices, key=lambda x: x[:3])
        used[control["neutral_source_id"]] += 1
        for key in ("neutral_source_id", "neutral_row_example_id", "neutral_row_source", "neutral_source_text", "neutral_source_token_ids", "neutral_kind"):
            row[key] = control[key]
        row["neutral_target_content_overlap_words"] = sorted(target_content & control["content"])
        overlap_counts.append(n_overlap)
    return {
        "all_exact_token_length": all(len(r["neutral_source_token_ids"]) == r["source_token_len"] for r in selected),
        "zero_target_content_overlap": sum(not r["neutral_target_content_overlap_words"] for r in selected),
        "unique_neutral_sources": len(used), "max_reuse": max(used.values(), default=0),
        "overlap_word_count": stats(overlap_counts),
    }


def build_records(selected: list[dict], tokenizer) -> list[dict]:
    bos = tokenizer.bos_token_id; eos = tokenizer.eos_token_id; mask = tokenizer.mask_token_id
    if bos is None or eos is None or mask is None:
        raise RuntimeError("tokenizer lacks BOS/EOS/MASK")
    records = []
    for row in selected:
        row["chosen_targets"] = choose_targets(row)
        slots = {
            "T": ("true_source", row["pair_id"], row["source_token_ids"]),
            "U": ("unrelated_source", row["unrelated_source_id"], row["unrelated_source_token_ids"]),
            "N": ("neutral_ordinary", row["neutral_source_id"], row["neutral_source_token_ids"]),
        }
        for target in row["chosen_targets"]:
            for condition_code, (condition, source_slot_id, source_ids) in slots.items():
                canonical = [int(bos)] + list(source_ids) + row["target_token_ids"] + [int(eos)]
                mask_position = 1 + len(source_ids) + target["target_token_index"]
                if canonical[mask_position] != target["target_token_id"]:
                    raise AssertionError("target alignment drift")
                input_ids = list(canonical); input_ids[mask_position] = int(mask)
                records.append({
                    "record_id": "wikirec_" + stable_hex(target["target_key"], condition_code),
                    "pair_id": row["pair_id"], "target_key": target["target_key"],
                    "condition_code": condition_code, "condition": condition,
                    "source_slot_id": source_slot_id, "dataset_split": row["dataset_split"],
                    "overlap_bin": row["overlap_bin"], "token_class": target["token_class"],
                    "input_ids": input_ids, "attention_mask": [1] * len(input_ids),
                    "mask_position": mask_position, "target_token_id": target["target_token_id"],
                    "target_word": target["word"], "target_occurrence_index": target["occurrence_index"],
                    "surface_word_class": target["surface_word_class"],
                    "target_token_index": target["target_token_index"],
                    "target_char_start": target["start"], "target_char_end": target["end"],
                    "source_slot_token_len": len(source_ids), "target_token_len": row["target_token_len"],
                    "seq_len": len(input_ids),
                })
    return records


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            cleaned = {k: v for k, v in row.items() if not k.startswith("_")}
            handle.write(json.dumps(cleaned, ensure_ascii=False, separators=(",", ":")) + "\n")


def childes_comparison() -> dict:
    if not CHILDES_STATS.exists():
        return {"available": False}
    old = json.loads(CHILDES_STATS.read_text(encoding="utf-8"))
    counts = old["counts_by_overlap_bin_class_condition"]
    return {
        "available": True, "path": rel(CHILDES_STATS), "pairs": old["selected_pairs"],
        "records": old["probe_records"],
        "targets_per_condition": {
            "overlap": sum(v for k, v in counts.items() if "|overlap|intact" in k),
            "nonoverlap": sum(v for k, v in counts.items() if "|nonoverlap|intact" in k),
        },
        "pair_bins": old["selected_pair_bins"],
    }


def make_manual_sample(selected: list[dict]) -> None:
    lines = [
        "# WikiLarge simplification probe manual sample\n\n",
        "This deterministic 27-pair sample contains nine items from each lexical-overlap bin. Reviewers should assess whether the target is supported by and restates/simplifies the true source; the embedding and directional-NLI scores are filters, not gold semantic labels. U and N are shown to expose accidental topical matches.\n\n",
    ]
    def esc(text: str) -> str:
        return text.replace("|", "\\|").replace("\n", " ")
    for bin_name in DATASET_BINS:
        items = sorted([r for r in selected if r["overlap_bin"] == bin_name], key=lambda r: stable_hex(SEED, "manual", r["pair_id"]))[:9]
        lines.extend([
            f"## {bin_name.capitalize()} overlap\n\n",
            "| pair | split:row | Jaccard / cosine / entailment | T source → target | targets (class) | U source | N source |\n",
            "|---|---|---:|---|---|---|---|\n",
        ])
        for row in items:
            targets = ", ".join(f"{t['word']} ({t['token_class']})" for t in row["chosen_targets"])
            lines.append(
                f"| {row['pair_id']} | {row['dataset_split']}:{row['dataset_row_index']} | "
                f"{row['content_jaccard']:.3f} / {row['semantic_similarity']:.3f} / {row['nli_entailment_probability']:.3f} | "
                f"{esc(row['source_text'])} **→** {esc(row['target_text'])} | {esc(targets)} | "
                f"{esc(row['unrelated_source_text'])} | {esc(row['neutral_source_text'])} |\n"
            )
        lines.append("\n")
    lines.extend([
        "## Automated safeguards represented in this sample\n\n",
        "All targets and complete T source+target bodies are tokenizer-surface absent from the three original 10M pools. Each pair has one overlap and two distinct-word non-overlap targets. U and N match the T source-slot token length exactly; their target-content overlap is recorded in the pair file. WikiLarge-clean exposes no article/title identifier, so article-level holdout cannot be audited.\n",
        "\n## Qualitative screen (not human annotation)\n\n",
        "All 27 displayed T pairs were read after construction. I found 22 cleanly supported restatements, five conservative/borderline cases, and no obvious contradiction or wholesale alignment mismatch. The borderline cases are `wikilarge_2cebae642e5cc7e3912e` (a definition is partly implicit), `wikilarge_649093e63ed44a228090` (\"title character\" is implicit), `wikilarge_954674065c1150e71ffe` (the previous university name is implicit), `wikilarge_c1b124fc7a9972dde39c` (\"champion\"→\"played\" inference), and `wikilarge_d5241c1337e3968d4c31` (evaluative wording). This screen is deliberately reported rather than converted into a gold-quality claim. The N texts are punctuation-delimited segments of packed BabyLM rows; abbreviations can occasionally create sentence-like fragments, but N carries no relational interpretation.\n",
    ])
    (_public_path('experiments/archive/relation_learning/analysis/wikipedia_simplification_restatement_probe/manual_sample.md')).write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-semantic-filter", action="store_true", help="debug only")
    args = parser.parse_args()
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True, use_fast=True)
    dataset = load_dataset(DATASET_NAME, revision=DATASET_REVISION, cache_dir=str(_public_path('experiments/archive/relation_learning/analysis/wikipedia_simplification_restatement_probe/_hf_cache/datasets')))
    lexical, lexical_audit = lexical_candidates(dataset, tokenizer)
    semantic, semantic_audit = semantic_filter(lexical, args.skip_semantic_filter)
    entailed, entailment_audit = entailment_filter(semantic, args.skip_semantic_filter)
    shortlisted, shortlist_audit = shortlist(entailed)
    exposure_audit = exposure_scan(shortlisted, tokenizer)
    selected, selection_audit = choose_balanced(shortlisted)
    # U need only be a well-formed WikiLarge source, not itself an entailed
    # source/target pair. The larger cosine-filtered pool makes exact-length,
    # target-content-disjoint matching feasible even at rare long lengths.
    unrelated_audit = assign_unrelated_controls(selected, semantic)
    neutral_pool, neutral_pool_audit = neutral_sentence_pool(tokenizer)
    neutral_audit = assign_neutral_controls(selected, neutral_pool)
    records = build_records(selected, tokenizer)

    pairs_path = _public_path('experiments/archive/relation_learning/analysis/wikipedia_simplification_restatement_probe/wikipedia_simplification_pairs.jsonl')
    records_path = _public_path('experiments/archive/relation_learning/analysis/wikipedia_simplification_restatement_probe/wikipedia_simplification_probe_records.jsonl')
    write_jsonl(pairs_path, selected); write_jsonl(records_path, records)
    make_manual_sample(selected)

    record_counts = collections.Counter((r["overlap_bin"], r["token_class"], r["condition_code"]) for r in records)
    pair_bins = collections.Counter(r["overlap_bin"] for r in selected)
    pair_splits = collections.Counter(r["dataset_split"] for r in selected)
    source_exposure = {a: sum(a in r["source_surface_exposure_arms"] for r in selected) for a in ARM_POOLS}
    probe_stats = {
        "status": "WIKILARGE_CLEAN_TUN_RESTATEMENT_PROBE_BUILT",
        "created_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seed": SEED,
        "dataset": {
            "name": DATASET_NAME, "revision": DATASET_REVISION, "url": DATASET_URL,
            "license": "CC-BY-SA-4.0", "fields": ["source", "target"],
            "article_or_title_fields_available": False,
            "split_rows": {split: len(dataset[split]) for split in dataset},
        },
        "tokenizer": rel(TOKENIZER_PATH), "tokenizer_vocab_size": len(tokenizer),
        "special_token_ids": {"bos_as_cls": tokenizer.bos_token_id, "eos_as_sep": tokenizer.eos_token_id, "mask": tokenizer.mask_token_id, "pad": tokenizer.pad_token_id, "unk": tokenizer.unk_token_id},
        "overlap_bin_boundaries": {"low": "[0, 0.25)", "medium": "[0.25, 0.50)", "high": "[0.50, 0.90) after near-copy rejection"},
        "lexical_filter": lexical_audit, "semantic_filter": semantic_audit, "entailment_filter": entailment_audit,
        "shortlist": shortlist_audit, "exposure_scan": exposure_audit, "selection": selection_audit,
        "selected_pairs": len(selected), "selected_pair_bins": dict(pair_bins), "selected_splits": dict(pair_splits),
        "surface_holdout": {
            "all_target_sentences_unexposed": all(not r["target_surface_exposure_arms"] for r in selected),
            "all_full_T_bodies_unexposed": all(not r["full_pair_surface_exposure_arms"] for r in selected),
            "selected_source_only_hits_by_arm": source_exposure,
            "article_level_holdout_possible": False,
        },
        "unrelated_control_audit": unrelated_audit,
        "neutral_pool_audit": neutral_pool_audit, "neutral_control_audit": neutral_audit,
        "records": len(records), "paired_targets": len(records) // 3,
        "record_counts_by_bin_class_condition": {
            f"{b}|{c}|{d}": record_counts[(b, c, d)]
            for b in DATASET_BINS for c in ("overlap", "nonoverlap") for d in ("T", "U", "N")
        },
        "pair_metrics": {
            "content_jaccard": stats([r["content_jaccard"] for r in selected]),
            "semantic_similarity": stats([r["semantic_similarity"] for r in selected]),
            "nli_entailment_probability": stats([r["nli_entailment_probability"] for r in selected]),
            "source_token_len": stats([r["source_token_len"] for r in selected]),
            "target_token_len": stats([r["target_token_len"] for r in selected]),
            "target_source_token_length_ratio": stats([r["target_source_token_length_ratio"] for r in selected]),
        },
        "target_policy": "exactly one whole-word single-token occurrence whose tokenizer ID occurs in the true source and two distinct-word whole-word single-token occurrences whose tokenizer IDs are absent from the true source per pair",
        "condition_policy": "T/U/N source slots have exactly equal tokenizer length; U is another WikiLarge source; N is a punctuation-delimited ordinary sentence from the BabyLM rowholdout construction slice",
        "childes_denominator_comparison": childes_comparison(),
        "elapsed_seconds": time.time() - started,
    }
    (_public_path('experiments/archive/relation_learning/analysis/wikipedia_simplification_restatement_probe/probe_stats.json')).write_text(json.dumps(probe_stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    provenance = {
        "dataset_name": DATASET_NAME, "dataset_url": DATASET_URL, "revision": DATASET_REVISION,
        "license": "CC BY-SA 4.0", "attribution_note": "Derivative of English Wikipedia and Simple English Wikipedia; preserve attribution and ShareAlike requirements.",
        "split_rows": {split: len(dataset[split]) for split in dataset},
        "source_fields": ["source", "target"],
        "parquet_sha256_from_repository_metadata": {
            "train": "0df1b0d546075648f3ef01913e0ef2b8bacdb1cee7d935a75a39eb130f5f0329",
            "validation": "11e5d219a76dd09bbb323f6001ca6ac5eaf2a97dfaf1be98679c7320d9dd41ee",
            "test": "2792264f67b18b7b9eb6c729d2605d10138b13aaddce2c559b7b1222381d81e5",
        },
        "semantic_filter_model": SEMANTIC_MODEL, "semantic_filter_revision": SEMANTIC_MODEL_REVISION,
        "entailment_filter_model": NLI_MODEL, "entailment_filter_revision": NLI_MODEL_REVISION,
    }
    (_public_path('experiments/archive/relation_learning/analysis/wikipedia_simplification_restatement_probe/dataset_provenance.json')).write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    child = probe_stats["childes_denominator_comparison"]
    lines = [
        "# Wikipedia simplification natural-restatement probe\n\n",
        f"The primary instrument contains **{len(selected):,}** WikiLarge-clean source→simplification pairs "
        f"({', '.join(f'{b}={pair_bins[b]}' for b in DATASET_BINS)}) and **{len(records):,}** T/U/N records. "
        "Each pair supplies one source-token-ID-present and two source-token-ID-absent single-token targets, so denominator differences do not drive bin comparisons.\n\n",
        "## Scientific role\n\n",
        "T presents the true complex Wikipedia source, U an exact-token-length unrelated WikiLarge source, and N an exact-token-length ordinary BabyLM source-slot segment. The same simplified target and mask position are used in all three conditions. T−U isolates related-source use within Wikipedia register; T−N anchors that effect against ordinary source-slot text; U−N measures register or unrelated-neighbor effects.\n\n",
        "## Quality and holdout\n\n",
        f"Pairs pass lexical, introduced-name, numeric-consistency, encoding-noise, length, semantic-similarity (cosine ≥{SEMANTIC_SIMILARITY_MIN}), and directional entailment (source→target probability ≥{NLI_ENTAILMENT_MIN}) filters. Exact tokenizer-ID screening found no selected target sentence or full T body in CLEAN, VIEW, or REPEAT. Source-only exposure is reported rather than excluded. The dataset has no article/title/id field, so only surface holdout—not article-level holdout—is possible.\n\n",
        "## Target denominators\n\n",
        "| bin | pairs | overlap targets per condition | non-overlap targets per condition |\n|---|---:|---:|---:|\n",
    ]
    for b in DATASET_BINS:
        lines.append(f"| {b} | {pair_bins[b]} | {record_counts[(b, 'overlap', 'T')]} | {record_counts[(b, 'nonoverlap', 'T')]} |\n")
    if child.get("available"):
        lines.extend([
            f"\nThe WikiLarge probe has {sum(record_counts[(b, 'overlap', 'T')] for b in DATASET_BINS):,} overlap and {sum(record_counts[(b, 'nonoverlap', 'T')] for b in DATASET_BINS):,} non-overlap targets per condition, versus "
            f"{child['targets_per_condition']['overlap']} overlap and {child['targets_per_condition']['nonoverlap']} non-overlap targets in the primary CHILDES probe. More importantly, WikiLarge’s source-token-absent targets occur inside an aligned simplification relation rather than merely following an adjacent utterance.\n",
        ])
    lines.extend([
        "\n## Files and scoring\n\n",
        "Use `wikipedia_simplification_pairs.jsonl`, `wikipedia_simplification_probe_records.jsonl`, `probe_stats.json`, and `validation_results.json` as the primary bundle. Inspect `manual_sample.md` before scoring. `dataset_provenance.json` records the pinned public revision and license.\n\n",
        "Run `score_wikipedia_simplification_probe.py --plan-only` first, then invoke it on the desired DeBERTa arms. The full command is recorded in `Research_Report.md`.\n",
    ])
    (_public_path('research/notes/relation_learning/analysis/wikipedia_simplification_restatement_probe/variation_or_restatement_summary.md')).write_text("".join(lines), encoding="utf-8")
    print(json.dumps({
        "pairs": len(selected), "records": len(records), "bins": dict(pair_bins),
        "splits": dict(pair_splits), "output": rel(HERE),
    }, indent=2))


if __name__ == "__main__":
    main()
