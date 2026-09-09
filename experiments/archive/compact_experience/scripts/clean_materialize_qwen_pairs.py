#!/usr/bin/env python3
"""research: clean Qwen rewrites, preserve complete pair boundaries, and materialize matched corpora.

This replaces the unsafe research materializer.  The old materializer could truncate
original--rewrite pairs and pad an aligned row with unrelated word fragments.  Here
we first reject noisy rewrites with transparent lexical/structural checks, quantify
the cleaned set, then build exact-10M corpora without splitting any original--rewrite
pair.

Scientific design of the first clean causal test:
  * treatment corpus = complete packed Qwen pairs + complete official 160-word rows
  * control corpus   = official text chunks with the exact same example-length
                       sequence as treatment, so batch word counts and row-length
                       effects are matched
  * both 10M pools are repeated for 10 shuffled passes with the same per-pass order
    to create exact 100M training JSONL files

The paired fraction is whatever the cleaned generation supports, capped at 25% of
corpus words.  If the cleaned pair budget is much lower than 25%, the metadata says
so explicitly; this is still a clean, matched test, not an inflated 25% corpus made
from fragments.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import collections
import hashlib
import json
import math
import pathlib
import random
import re
import statistics
import time
from dataclasses import dataclass, asdict
from typing import Iterable

ROOT = _public_path('experiments/archive/compact_experience')  # 
DATA_DIR = _public_path('experiments/archive/compact_experience/data/qwen_aligned')
POOL_PATH = _public_path('experiments/archive/compact_experience/data/mixture/official_pool.jsonl')
OUTPUTS_PATH = _public_path('experiments/archive/compact_experience/training/runs/qwen_rewrites_full/outputs.jsonl')
SOURCES_PATH = _public_path('experiments/archive/compact_experience/data/qwen_aligned/source_sentences.jsonl')
OUT_DIR = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned')
OUT_DIR.mkdir(parents=True, exist_ok=True)
CORPUS_DIR = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora')
CORPUS_DIR.mkdir(parents=True, exist_ok=True)

TOTAL_WORDS = 10_000_000
PASSES = 10
WORDS_PER_OFFICIAL_ROW = 160
TARGET_PAIR_WORD_FRACTION = 0.25
TARGET_PAIR_WORDS = int(TOTAL_WORDS * TARGET_PAIR_WORD_FRACTION)
MAX_PACKED_EXAMPLE_WORDS = 160
MIN_PACKED_EXAMPLE_WORDS_SOFT = 80  # not a rejection; used in report only
RNG_SEED = 28043

# Conservative but not semantic-model-dependent checks.  These are meant to remove
# obvious noise and quantify remaining risk; they do not prove semantic equivalence.
MIN_REWRITE_WORDS = 6
MAX_REWRITE_WORDS = 70
MIN_LEN_RATIO = 0.45
MAX_LEN_RATIO = 2.10
MIN_CONTENT_OVERLAP = 0.14
MAX_CONTENT_OVERLAP = 0.94
MIN_ENTITY_RECALL_SMALL = 1.0
MIN_ENTITY_RECALL_LARGE = 0.75
MAX_DUPLICATE_OUTPUTS_PER_FINGERPRINT = 1

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by", "can", "could",
    "did", "do", "does", "for", "from", "had", "has", "have", "he", "her", "hers", "him", "his",
    "i", "if", "in", "into", "is", "it", "its", "just", "may", "might", "must", "not", "of", "on",
    "or", "our", "she", "should", "so", "some", "such", "than", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "to", "too", "under", "up", "very", "was", "we", "were",
    "what", "when", "where", "which", "who", "will", "with", "would", "you", "your", "the", "one",
    "only", "more", "most", "all", "any", "both", "each", "every", "own", "same", "other", "through",
}
FIRST_WORD_NOT_ENTITY = {
    "A", "An", "And", "As", "At", "But", "By", "For", "From", "He", "Her", "His", "I", "If", "In",
    "It", "On", "She", "Some", "That", "The", "Then", "There", "They", "This", "To", "Under", "We",
    "When", "Where", "While", "With", "You", "Your",
}
BAD_START_LOWER = {
    # Starts that are usually continuation fragments rather than self-contained sentences.
    # Conditional/subordinate openings such as If/When/Although can be complete sentences,
    # so they are not rejected here; many good rewrites begin with such clauses.
    "and", "but", "or", "which", "that", "whose", "whom", "soon", "of",
}
BAD_START_ANYCASE = {
    "as soon", "which", "of",
}
META_PREFIXES = (
    "here is", "here's", "sure", "certainly", "of course", "the rewritten", "rewritten sentence",
    "sentence:", "rewrite:", "output:", "translation:", "i'm sorry", "i cannot", "i can",
)

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[\u2019'][A-Za-z0-9]+)?|\*[A-Za-z]+:?|[£$€]\s*\d+(?:\.\d+)?|\d+(?:[.,:/-]\d+)*(?:%|st|nd|rd|th)?")
NUM_RE = re.compile(r"(?:[£$€]\s*)?\d+(?:[.,:/-]\d+)*(?:%|st|nd|rd|th)?", re.I)


@dataclass
class SourceSentence:
    pair_id: str
    text: str
    words: int
    source: str
    example_id: int
    cohort: str


@dataclass
class CleanPair:
    pair_id: str
    original: str
    rewrite: str
    original_words: int
    rewrite_words: int
    pair_words: int
    source: str
    example_id: int
    cohort: str
    len_ratio: float
    content_overlap: float
    entity_recall: float
    num_source: list[str]
    num_rewrite: list[str]
    entity_source: list[str]
    entity_rewrite: list[str]


@dataclass
class PackedRow:
    text: str
    words: int
    source: str
    example_id: int
    pair_ids: list[str] | None = None
    n_pairs: int | None = None
    component_sources: dict[str, int] | None = None


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def words(text: str) -> list[str]:
    return text.split()


def normalize_ws(text: str) -> str:
    return " ".join(text.replace("\u00a0", " ").split())


def strip_outer_quotes(text: str) -> str:
    text = normalize_ws(text.strip())
    changed = True
    while changed and len(text) >= 2:
        changed = False
        pairs = [("\"", "\""), ("'", "'"), ("“", "”"), ("‘", "’")]
        for a, b in pairs:
            if text.startswith(a) and text.endswith(b):
                text = normalize_ws(text[1:-1].strip())
                changed = True
    return text


def clean_output(raw: str) -> str:
    text = strip_outer_quotes(raw or "")
    # Remove rare prompt echoes conservatively.
    low = text.lower()
    for marker in ("rewritten sentence:", "rewrite:", "output:", "sentence:"):
        if low.startswith(marker):
            text = normalize_ws(text[len(marker):].strip())
            low = text.lower()
    return strip_outer_quotes(text)


def norm_tokens(text: str) -> list[str]:
    toks = []
    for m in WORD_RE.finditer(text):
        t = m.group(0).strip("'\"“”‘’.,!?;:()[]{}")
        if t:
            toks.append(t.lower())
    return toks


def content_tokens(text: str) -> list[str]:
    toks = []
    for t in norm_tokens(text):
        if len(t) >= 3 and t not in STOPWORDS:
            toks.append(t)
    return toks


def output_fingerprint(text: str) -> str:
    return " ".join(norm_tokens(text))


def number_tokens(text: str) -> list[str]:
    return [re.sub(r"\s+", "", m.group(0).lower()) for m in NUM_RE.finditer(text)]


def entity_tokens(text: str) -> list[str]:
    ents: list[str] = []
    raw = words(text)
    for i, w in enumerate(raw):
        stripped = w.strip("\"“”‘’.,!?;:()[]{}")
        if not stripped:
            continue
        if stripped.startswith("*") and len(stripped) > 2:
            ents.append(stripped.lower().rstrip(":"))
            continue
        # Words containing internal capitals or all caps are strong anchors.
        letters = re.sub(r"[^A-Za-z]", "", stripped)
        if not letters:
            continue
        is_cap = letters[0].isupper() and (len(letters) > 1)
        has_internal_cap = any(c.isupper() for c in letters[1:])
        all_caps = len(letters) >= 2 and letters.isupper()
        if is_cap or has_internal_cap or all_caps:
            # A single title-cased first word is usually sentence capitalization, not a reliable
            # entity anchor.  Keep first-position anchors only when they are strong markers
            # such as *CHI, ALLCAPS, or internal capitals (e.g. McDonald).  This avoids
            # rejecting good rewrites because the original began with ordinary words such
            # as Here/Now/Yes that were capitalized only by position.
            if i == 0 and not has_internal_cap and not all_caps and not stripped.startswith("*"):
                continue
            if stripped.lower() in STOPWORDS:
                continue
            ents.append(stripped.lower())
    # Deduplicate while preserving order.
    seen = set()
    out = []
    for e in ents:
        if e not in seen:
            out.append(e)
            seen.add(e)
    return out


def has_repetition(text: str) -> bool:
    toks = norm_tokens(text)
    if len(toks) < 8:
        return False
    # Four identical tokens in a row, or repeated 4-gram.
    run = 1
    for a, b in zip(toks, toks[1:]):
        run = run + 1 if a == b else 1
        if run >= 4:
            return True
    grams = [tuple(toks[i:i+4]) for i in range(len(toks) - 3)]
    c = collections.Counter(grams)
    return any(v >= 2 for v in c.values())


def starts_bad_fragment(text: str) -> bool:
    s = text.lstrip(" \t\"'“”‘’(")
    if not s:
        return True
    first = s.split()[0].strip(".,!?;:()[]{}\"'“”‘’") if s.split() else ""
    low = s.lower()
    if any(low.startswith(p + " ") or low == p for p in BAD_START_ANYCASE):
        # Allow explicit quotations or speaker turns; otherwise these are often subordinate fragments.
        if not text.lstrip().startswith(("\"", "'", "“", "*")):
            return True
    if first and first[0].islower() and first.lower() in BAD_START_LOWER:
        return True
    return False


TRAILING_FRAGMENT_WORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "from", "with", "without", "by", "as",
    "than", "that", "which", "who", "whose", "whom", "when", "where", "while", "if", "because", "but",
    "and", "or", "nor", "so", "then", "he", "she", "it", "they", "we", "you", "i", "his", "her", "their",
    "our", "your", "my", "not", "no", "had", "has", "have", "was", "were", "is", "are", "be", "been",
    "being", "do", "does", "did", "will", "would", "could", "should", "may", "might", "must",
}
TRAILING_ABBREVIATIONS = {"mr.", "mrs.", "ms.", "dr.", "prof.", "st.", "jr.", "sr.", "no.", "fig."}


def is_complete_pair_text(text: str, is_source: bool) -> bool:
    """Heuristic completeness gate for both source and rewrite sides.

    This is intentionally stricter than the original research source extraction, because
    incomplete sources cause Qwen to hallucinate continuations, which would mix semantic
    alignment with generation noise.  It is still a surface filter, not a proof of meaning.
    """
    s = normalize_ws(text).strip()
    if not s:
        return False
    low = s.lower()
    if "..." in s or "…" in s:
        return False
    if "= =" in s or "http" in low or ".cha" in low or "/childes" in low:
        return False
    if starts_bad_fragment(s):
        return False
    stripped = s.rstrip(' \t\"\'”’)]}')
    if not stripped:
        return False
    last = stripped.split()[-1].strip(' \t\"\'“”‘’()[]{}')
    last_low = last.lower()
    if last_low in TRAILING_FRAGMENT_WORDS or last_low in TRAILING_ABBREVIATIONS:
        return False
    # Require sentence-like closure.  Speaker labels and bracket-heavy CHILDES lines are kept only
    # when they also close with punctuation.
    if not re.search(r"[.!?][\"'”’\)\]]*$", s):
        return False
    # Simple balance checks.  They avoid many quote/bracket fragments while not requiring perfect
    # typographic normalization.
    if s.count("[") != s.count("]"):
        return False
    if s.count("(") != s.count(")"):
        return False
    if s.count('"') % 2 == 1:
        return False
    if s.count("“") != s.count("”"):
        return False
    if s.count("‘") != s.count("’"):
        return False
    return True

def validate_pair(src: SourceSentence, raw_output: str, duplicate_seen: collections.Counter[str]) -> tuple[CleanPair | None, list[str]]:
    reasons: list[str] = []
    out = clean_output(raw_output)
    if not out:
        return None, ["empty_output"]
    if "\n" in out or "\r" in out:
        reasons.append("newline")
    low = out.lower().strip()
    if low.startswith(META_PREFIXES):
        reasons.append("meta_prefix")
    if out[-1:] in {",", ";", ":", "-", "—"}:
        reasons.append("bad_terminal_fragment")
    if not is_complete_pair_text(out, is_source=False):
        reasons.append("rewrite_incomplete_or_fragment")
    if not is_complete_pair_text(src.text, is_source=True):
        reasons.append("source_incomplete_or_fragment")
    if has_repetition(out):
        reasons.append("repetition")

    ow = src.words
    rw = len(words(out))
    if rw < MIN_REWRITE_WORDS:
        reasons.append(f"too_short_{rw}")
    if rw > MAX_REWRITE_WORDS:
        reasons.append(f"too_long_{rw}")
    ratio = rw / max(1, ow)
    if ratio < MIN_LEN_RATIO or ratio > MAX_LEN_RATIO:
        reasons.append(f"len_ratio_{ratio:.2f}")

    src_nums = number_tokens(src.text)
    out_nums = number_tokens(out)
    if collections.Counter(src_nums) != collections.Counter(out_nums):
        reasons.append("number_mismatch")

    src_ents = entity_tokens(src.text)
    out_ents = entity_tokens(out)
    if src_ents:
        out_norm = set(out_ents)
        matched = 0
        for e in src_ents:
            if e in out_norm or e.replace("'s", "") in out_norm or e.rstrip("s") in out_norm:
                matched += 1
        recall = matched / len(src_ents)
        needed = MIN_ENTITY_RECALL_SMALL if len(src_ents) <= 2 else MIN_ENTITY_RECALL_LARGE
        if recall < needed:
            reasons.append(f"entity_recall_{recall:.2f}")
    else:
        recall = 1.0

    src_content = set(content_tokens(src.text))
    out_content = set(content_tokens(out))
    if src_content or out_content:
        overlap = len(src_content & out_content) / max(1, min(len(src_content), len(out_content)))
    else:
        overlap = 0.0
    # Low lexical content overlap is not proof of wrong meaning, but many observed failures have near-zero anchors.
    if overlap < MIN_CONTENT_OVERLAP and src.words >= 10:
        reasons.append(f"low_content_overlap_{overlap:.2f}")
    if overlap > MAX_CONTENT_OVERLAP and rw >= ow * 0.85:
        reasons.append(f"copy_overlap_{overlap:.2f}")

    fp = output_fingerprint(out)
    if not fp:
        reasons.append("empty_fingerprint")
    # Duplicate filtering is applied only after all other checks pass.  Otherwise an invalid
    # earlier output could cause a later valid output with the same fingerprint to be rejected
    # solely because of source order.
    if not reasons:
        duplicate_seen[fp] += 1
        if duplicate_seen[fp] > MAX_DUPLICATE_OUTPUTS_PER_FINGERPRINT:
            reasons.append("duplicate_output")

    if reasons:
        return None, reasons
    return CleanPair(
        pair_id=src.pair_id,
        original=src.text,
        rewrite=out,
        original_words=ow,
        rewrite_words=rw,
        pair_words=ow + rw,
        source=src.source,
        example_id=src.example_id,
        cohort=src.cohort,
        len_ratio=ratio,
        content_overlap=overlap,
        entity_recall=recall,
        num_source=src_nums,
        num_rewrite=out_nums,
        entity_source=src_ents,
        entity_rewrite=out_ents,
    ), []


def pack_pairs(pairs: list[CleanPair], max_words: int = MAX_PACKED_EXAMPLE_WORDS) -> list[PackedRow]:
    rows: list[PackedRow] = []
    cur_segments: list[str] = []
    cur_pair_ids: list[str] = []
    cur_words = 0
    cur_sources: collections.Counter[str] = collections.Counter()
    for p in pairs:
        if p.pair_words > max_words:
            # Should be very rare due validation, but keep pair intact as its own row if it happens.
            if cur_segments:
                rows.append(PackedRow(
                    text=" ".join(cur_segments), words=cur_words, source="qwen_pair_packed",
                    example_id=300000 + len(rows), pair_ids=list(cur_pair_ids), n_pairs=len(cur_pair_ids),
                    component_sources=dict(cur_sources),
                ))
                cur_segments, cur_pair_ids, cur_words, cur_sources = [], [], 0, collections.Counter()
            rows.append(PackedRow(
                text=f"{p.original} {p.rewrite}", words=p.pair_words, source="qwen_pair_packed",
                example_id=300000 + len(rows), pair_ids=[p.pair_id], n_pairs=1,
                component_sources={p.source: p.pair_words},
            ))
            continue
        if cur_words and cur_words + p.pair_words > max_words:
            rows.append(PackedRow(
                text=" ".join(cur_segments), words=cur_words, source="qwen_pair_packed",
                example_id=300000 + len(rows), pair_ids=list(cur_pair_ids), n_pairs=len(cur_pair_ids),
                component_sources=dict(cur_sources),
            ))
            cur_segments, cur_pair_ids, cur_words, cur_sources = [], [], 0, collections.Counter()
        cur_segments.append(p.original)
        cur_segments.append(p.rewrite)
        cur_pair_ids.append(p.pair_id)
        cur_words += p.pair_words
        cur_sources[p.source] += p.pair_words
    if cur_segments:
        rows.append(PackedRow(
            text=" ".join(cur_segments), words=cur_words, source="qwen_pair_packed",
            example_id=300000 + len(rows), pair_ids=list(cur_pair_ids), n_pairs=len(cur_pair_ids),
            component_sources=dict(cur_sources),
        ))
    # Sanity: no truncation, no padding.
    assert sum(r.words for r in rows) == sum(p.pair_words for p in pairs)
    for r in rows:
        assert r.words == len(r.text.split())
        assert r.words <= max_words or (r.n_pairs == 1)
    return rows


def read_official_pool() -> list[dict]:
    rows: list[dict] = []
    with POOL_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            obj["words"] = int(obj.get("words", len(obj["text"].split())))
            assert obj["words"] == len(obj["text"].split())
            rows.append(obj)
    assert sum(r["words"] for r in rows) == TOTAL_WORDS
    return rows


def choose_pairs(clean_pairs: list[CleanPair]) -> list[CleanPair]:
    # Shuffle deterministically, then greedily accept pairs up to target, with total pair words divisible by 160
    # so qwen treatment can be completed with whole official rows and no word-fragment padding.  Prefer the
    # research extra cohorts produced with stricter prompts and complete-sentence extraction; use research base
    # pairs only if the extra cohorts cannot fill the target budget.
    rng = random.Random(RNG_SEED)
    pairs = list(clean_pairs)
    rng.shuffle(pairs)
    priority = {"extra_shard0": 0, "extra_shard1": 0, "base": 1}
    pairs.sort(key=lambda p: priority.get(p.cohort, 2))
    selected: list[CleanPair] = []
    total = 0
    for p in pairs:
        if total + p.pair_words <= TARGET_PAIR_WORDS:
            selected.append(p)
            total += p.pair_words
    # Drop a small suffix until official filler can be exact whole 160-word rows.
    while selected and total % WORDS_PER_OFFICIAL_ROW != 0:
        p = selected.pop()
        total -= p.pair_words
    return selected


def official_rows_for_filler(official_pool: list[dict], excluded_example_ids: set[int], needed_words: int) -> list[PackedRow]:
    assert needed_words % WORDS_PER_OFFICIAL_ROW == 0
    needed_rows = needed_words // WORDS_PER_OFFICIAL_ROW
    rows_non_source = [r for r in official_pool if int(r["example_id"]) not in excluded_example_ids]
    rows_source = [r for r in official_pool if int(r["example_id"]) in excluded_example_ids]
    rng = random.Random(RNG_SEED + 1)
    rng.shuffle(rows_non_source)
    rng.shuffle(rows_source)
    chosen = (rows_non_source + rows_source)[:needed_rows]
    if len(chosen) != needed_rows:
        raise RuntimeError(f"Need {needed_rows} official rows, got {len(chosen)}")
    return [PackedRow(text=r["text"], words=int(r["words"]), source=str(r["source"]), example_id=int(r["example_id"])) for r in chosen]


def official_length_matched_control(official_pool: list[dict], lengths: list[int]) -> list[PackedRow]:
    # Continuous official words are chunked to the treatment length sequence.  This deliberately matches
    # example count, example lengths, and batch word totals; text remains strictly official training text.
    all_words: list[str] = []
    source_marks: list[str] = []
    example_marks: list[int] = []
    for r in official_pool:
        ws = r["text"].split()
        all_words.extend(ws)
        source_marks.extend([str(r["source"])] * len(ws))
        example_marks.extend([int(r["example_id"])] * len(ws))
    if len(all_words) != TOTAL_WORDS:
        raise RuntimeError(f"official word stream {len(all_words)} != {TOTAL_WORDS}")
    # Deterministic rotation avoids always beginning with bnc_spoken while preserving official-only text.
    offset = 137 * WORDS_PER_OFFICIAL_ROW
    all_words = all_words[offset:] + all_words[:offset]
    source_marks = source_marks[offset:] + source_marks[:offset]
    example_marks = example_marks[offset:] + example_marks[:offset]
    rows: list[PackedRow] = []
    pos = 0
    for i, L in enumerate(lengths):
        seg_words = all_words[pos:pos+L]
        seg_sources = source_marks[pos:pos+L]
        seg_examples = example_marks[pos:pos+L]
        if len(seg_words) != L:
            raise RuntimeError("length sequence exceeded official stream")
        common_source = collections.Counter(seg_sources).most_common(1)[0][0]
        common_example = collections.Counter(seg_examples).most_common(1)[0][0]
        rows.append(PackedRow(
            text=" ".join(seg_words), words=L, source=f"official_lengthmatched::{common_source}",
            example_id=500000 + i,
        ))
        pos += L
    assert pos == TOTAL_WORDS
    return rows


def write_pool(path: pathlib.Path, rows: list[PackedRow], include_pair_meta_path: pathlib.Path | None = None) -> None:
    meta_f = include_pair_meta_path.open("w", encoding="utf-8") if include_pair_meta_path else None
    try:
        with path.open("w", encoding="utf-8") as f:
            for i, r in enumerate(rows):
                if r.words != len(r.text.split()):
                    raise RuntimeError(f"row {i} mismatch words={r.words} actual={len(r.text.split())}")
                rec = {"text": r.text, "words": r.words, "example_id": r.example_id, "source": r.source}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if meta_f and r.pair_ids:
                    meta_f.write(json.dumps({
                        "row_index": i,
                        "example_id": r.example_id,
                        "words": r.words,
                        "pair_ids": r.pair_ids,
                        "n_pairs": r.n_pairs,
                        "component_sources": r.component_sources,
                    }, ensure_ascii=False) + "\n")
    finally:
        if meta_f:
            meta_f.close()


def write_training_file(path: pathlib.Path, pool_rows: list[PackedRow], pass_orders: list[list[int]]) -> None:
    total = 0
    with path.open("w", encoding="utf-8") as f:
        for pass_i, order in enumerate(pass_orders):
            for idx in order:
                r = pool_rows[idx]
                rec = {"text": r.text, "words": r.words, "example_id": r.example_id, "source": r.source}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total += r.words
    assert total == TOTAL_WORDS * PASSES, f"{path} total {total}"


def stats(vals: Iterable[float]) -> dict:
    vals = list(vals)
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "min": round(min(vals), 4),
        "p05": round(statistics.quantiles(vals, n=20)[0], 4) if len(vals) >= 20 else round(min(vals), 4),
        "mean": round(statistics.mean(vals), 4),
        "median": round(statistics.median(vals), 4),
        "p95": round(statistics.quantiles(vals, n=20)[-1], 4) if len(vals) >= 20 else round(max(vals), 4),
        "max": round(max(vals), 4),
    }


def main() -> None:
    t0 = time.time()
    print("research clean materialization", flush=True)
    generation_specs = [
        {
            "name": "base",
            "source_path": SOURCES_PATH,
            "output_path": OUTPUTS_PATH,
            "id_prefix": "rw",
        },
        {
            "name": "extra_shard0",
            "source_path": _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/extra_source_sentences_shard0.jsonl'),
            "output_path": _public_path('experiments/archive/compact_experience/training/runs/extra_qwen2_shard0/outputs.jsonl'),
            "id_prefix": "rw2s0",
        },
        {
            "name": "extra_shard1",
            "source_path": _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/extra_source_sentences_shard1.jsonl'),
            "output_path": _public_path('experiments/archive/compact_experience/training/runs/extra_qwen2_shard1/outputs.jsonl'),
            "id_prefix": "rw2s1",
        },
    ]

    sources: dict[str, SourceSentence] = {}
    source_counts_by_spec: dict[str, int] = {}
    for spec in generation_specs:
        sp = pathlib.Path(spec["source_path"])
        if not sp.exists():
            print(f"  source missing, skipped: {spec['name']} {sp}")
            continue
        count = 0
        with sp.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                o = json.loads(line)
                pid = str(o["id"])
                sources[pid] = SourceSentence(
                    pair_id=pid, text=normalize_ws(o["text"]), words=int(o["words"]),
                    source=str(o["source"]), example_id=int(o["example_id"]), cohort=str(spec["name"]),
                )
                count += 1
        source_counts_by_spec[spec["name"]] = count
    print(f"  source sentences: {len(sources)} from {source_counts_by_spec}")

    outputs_by_pair: dict[str, str] = {}
    output_records = 0
    output_counts_by_spec: dict[str, int] = {}
    for spec in generation_specs:
        op = pathlib.Path(spec["output_path"])
        if not op.exists():
            print(f"  output missing, skipped: {spec['name']} {op}")
            continue
        count = 0
        with op.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                idx = int(rec["index"])
                pair_id = f"{spec['id_prefix']}_{idx:06d}"
                outputs_by_pair[pair_id] = str(rec.get("output", ""))
                output_records += 1
                count += 1
        output_counts_by_spec[spec["name"]] = count
    print(f"  output records: {output_records} from {output_counts_by_spec}")

    duplicate_seen: collections.Counter[str] = collections.Counter()
    clean_pairs: list[CleanPair] = []
    rejection_reasons: collections.Counter[str] = collections.Counter()
    rejected_examples: list[dict] = []
    accepted_examples: list[dict] = []

    for pair_id in sorted(sources):
        src = sources[pair_id]
        raw = outputs_by_pair.get(pair_id, "")
        pair, reasons = validate_pair(src, raw, duplicate_seen)
        if pair is None:
            for r in reasons:
                # Bucket numeric suffix reasons so report is readable.
                b = re.sub(r"_(?:0\.\d+|\d+\.\d+|\d+)$", "", r)
                rejection_reasons[b] += 1
            if len(rejected_examples) < 120:
                rejected_examples.append({
                    "pair_id": pair_id, "cohort": src.cohort, "source": src.source, "original": src.text,
                    "raw_output": raw, "clean_output": clean_output(raw), "reasons": reasons,
                })
        else:
            clean_pairs.append(pair)
            if len(accepted_examples) < 120:
                accepted_examples.append({
                    "pair_id": pair_id, "cohort": pair.cohort, "source": src.source, "original": pair.original,
                    "rewrite": pair.rewrite, "words": [pair.original_words, pair.rewrite_words],
                    "len_ratio": round(pair.len_ratio, 3), "content_overlap": round(pair.content_overlap, 3),
                    "entities": [pair.entity_source, pair.entity_rewrite], "numbers": [pair.num_source, pair.num_rewrite],
                })

    print(f"  clean pairs: {len(clean_pairs)}; rejected: {len(sources) - len(clean_pairs)}")
    print(f"  rejection top: {rejection_reasons.most_common(12)}")
    # Write early spotcheck material even if later corpus-size checks stop the run.
    (_public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/accepted_spotcheck_preselection.json')).write_text(json.dumps(accepted_examples, indent=2, ensure_ascii=False), encoding="utf-8")
    (_public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/rejected_spotcheck_preselection.json')).write_text(json.dumps(rejected_examples, indent=2, ensure_ascii=False), encoding="utf-8")

    if len(clean_pairs) < 1000:
        raise RuntimeError("Too few clean pairs; do not train")

    selected_pairs = choose_pairs(clean_pairs)
    selected_pair_words = sum(p.pair_words for p in selected_pairs)
    if selected_pair_words <= 0 or selected_pair_words % WORDS_PER_OFFICIAL_ROW != 0:
        raise RuntimeError(f"selected pair words invalid: {selected_pair_words}")
    if selected_pair_words < 0.10 * TOTAL_WORDS:
        raise RuntimeError(f"Clean aligned words only {selected_pair_words}; below 10% corpus, generate more before training")

    packed_pair_rows = pack_pairs(selected_pairs)
    pair_row_words = sum(r.words for r in packed_pair_rows)
    assert pair_row_words == selected_pair_words

    official_pool = read_official_pool()
    filler_words = TOTAL_WORDS - selected_pair_words
    excluded = {p.example_id for p in selected_pairs}
    qwen_filler_rows = official_rows_for_filler(official_pool, excluded, filler_words)
    qwen_pool = packed_pair_rows + qwen_filler_rows

    # Shuffle the 10M pool once for training pool files; the 100M files then use matched pass orders.
    rng = random.Random(RNG_SEED + 2)
    rng.shuffle(qwen_pool)
    qwen_lengths = [r.words for r in qwen_pool]
    official_control_pool = official_length_matched_control(official_pool, qwen_lengths)

    assert sum(r.words for r in qwen_pool) == TOTAL_WORDS
    assert sum(r.words for r in official_control_pool) == TOTAL_WORDS
    assert [r.words for r in qwen_pool] == [r.words for r in official_control_pool]

    # Save clean-pair table and examples for inspection.
    clean_pairs_path = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/clean_pairs.jsonl')
    with clean_pairs_path.open("w", encoding="utf-8") as f:
        for p in clean_pairs:
            f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")
    selected_pairs_path = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/selected_pairs.jsonl')
    with selected_pairs_path.open("w", encoding="utf-8") as f:
        for p in selected_pairs:
            f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")
    (_public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/accepted_spotcheck.json')).write_text(json.dumps(accepted_examples, indent=2, ensure_ascii=False), encoding="utf-8")
    (_public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/rejected_spotcheck.json')).write_text(json.dumps(rejected_examples, indent=2, ensure_ascii=False), encoding="utf-8")

    official_pool_path = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/official_only_10M.jsonl')
    qwen_pool_path = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl')
    pair_meta_path = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/qwen_pair_packed_rows_meta.jsonl')
    write_pool(official_pool_path, official_control_pool)
    write_pool(qwen_pool_path, qwen_pool, include_pair_meta_path=pair_meta_path)

    pass_orders: list[list[int]] = []
    n_rows = len(qwen_pool)
    for pass_i in range(PASSES):
        order = list(range(n_rows))
        random.Random(RNG_SEED + 100 + pass_i).shuffle(order)
        pass_orders.append(order)

    official_train_path = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/official_only_100M.jsonl')
    qwen_train_path = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_100M.jsonl')
    write_training_file(official_train_path, official_control_pool, pass_orders)
    write_training_file(qwen_train_path, qwen_pool, pass_orders)

    length_counts = collections.Counter(qwen_lengths)
    qwen_source_words = collections.Counter()
    for r in qwen_pool:
        qwen_source_words[r.source] += r.words
    selected_source_words = collections.Counter()
    selected_source_pairs = collections.Counter()
    selected_cohort_words = collections.Counter()
    selected_cohort_pairs = collections.Counter()
    clean_cohort_pairs = collections.Counter()
    clean_cohort_words = collections.Counter()
    for p in clean_pairs:
        clean_cohort_pairs[p.cohort] += 1
        clean_cohort_words[p.cohort] += p.pair_words
    for p in selected_pairs:
        selected_source_words[p.source] += p.pair_words
        selected_source_pairs[p.source] += 1
        selected_cohort_words[p.cohort] += p.pair_words
        selected_cohort_pairs[p.cohort] += 1

    # Compare length/source changes in the selected clean pairs.
    metadata = {
        "status": "CLEAN_QWEN_CORPORA_MATERIALIZED",
        "replaces_unsafe_step027_materializer": True,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generation_output_records": output_records,
        "source_sentences": len(sources),
        "clean_pairs_available": len(clean_pairs),
        "rejected_pairs": len(sources) - len(clean_pairs),
        "rejection_reasons": dict(rejection_reasons.most_common()),
        "validation_thresholds": {
            "min_rewrite_words": MIN_REWRITE_WORDS,
            "max_rewrite_words": MAX_REWRITE_WORDS,
            "len_ratio": [MIN_LEN_RATIO, MAX_LEN_RATIO],
            "content_overlap": [MIN_CONTENT_OVERLAP, MAX_CONTENT_OVERLAP],
            "entity_recall_small": MIN_ENTITY_RECALL_SMALL,
            "entity_recall_large": MIN_ENTITY_RECALL_LARGE,
            "target_pair_word_fraction_cap": TARGET_PAIR_WORD_FRACTION,
        },
        "selected_pairs": len(selected_pairs),
        "selected_pair_words": selected_pair_words,
        "selected_pair_word_fraction": round(selected_pair_words / TOTAL_WORDS, 6),
        "target_pair_words": TARGET_PAIR_WORDS,
        "target_pair_word_fraction_reached": round(selected_pair_words / TARGET_PAIR_WORDS, 6),
        "official_filler_words_in_treatment": filler_words,
        "official_filler_rows_in_treatment": len(qwen_filler_rows),
        "treatment_pool_rows": len(qwen_pool),
        "control_pool_rows": len(official_control_pool),
        "row_length_sequence_matched": [r.words for r in qwen_pool] == [r.words for r in official_control_pool],
        "pair_boundary_preserved": True,
        "pair_truncation": False,
        "word_fragment_padding_in_qwen_pair_rows": False,
        "qwen_pair_rows": len(packed_pair_rows),
        "qwen_pair_row_word_stats": stats([r.words for r in packed_pair_rows]),
        "qwen_pair_rows_below_soft_80w": sum(1 for r in packed_pair_rows if r.words < MIN_PACKED_EXAMPLE_WORDS_SOFT),
        "all_pool_word_totals": {
            "official_only_10M": sum(r.words for r in official_control_pool),
            "qwen_aligned_10M": sum(r.words for r in qwen_pool),
            "official_only_100M": TOTAL_WORDS * PASSES,
            "qwen_aligned_100M": TOTAL_WORDS * PASSES,
        },
          "clean_cohort_pairs": dict(clean_cohort_pairs),
          "clean_cohort_words": dict(clean_cohort_words),
          "selected_cohort_pairs": dict(selected_cohort_pairs),
          "selected_cohort_words": dict(selected_cohort_words),
          "selected_pair_source_pairs": dict(selected_source_pairs),
          "selected_pair_source_words": dict(selected_source_words),
          "treatment_pool_source_words": dict(qwen_source_words),
        "source_and_length_change": {
            "source_words": stats([p.original_words for p in selected_pairs]),
            "rewrite_words": stats([p.rewrite_words for p in selected_pairs]),
            "pair_words": stats([p.pair_words for p in selected_pairs]),
            "len_ratio_rewrite_over_source": stats([p.len_ratio for p in selected_pairs]),
            "content_overlap": stats([p.content_overlap for p in selected_pairs]),
            "entity_recall": stats([p.entity_recall for p in selected_pairs if p.entity_source]),
            "pairs_with_source_entities": sum(1 for p in selected_pairs if p.entity_source),
            "pairs_with_numbers": sum(1 for p in selected_pairs if p.num_source),
        },
        "duplicate_output_fingerprints_seen": sum(1 for _, c in duplicate_seen.items() if c > 1),
        "files": {
            "clean_pairs": str(clean_pairs_path),
            "selected_pairs": str(selected_pairs_path),
            "accepted_spotcheck": str(_public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/accepted_spotcheck.json')),
            "rejected_spotcheck": str(_public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/rejected_spotcheck.json')),
            "packed_pair_rows_meta": str(pair_meta_path),
            "official_only_pool": str(official_pool_path),
            "qwen_aligned_pool": str(qwen_pool_path),
            "official_only_training": str(official_train_path),
            "qwen_aligned_training": str(qwen_train_path),
        },
        "sha256": {
            "official_only_10M.jsonl": sha256_file(official_pool_path),
            "qwen_aligned_10M.jsonl": sha256_file(qwen_pool_path),
            "official_only_100M.jsonl": sha256_file(official_train_path),
            "qwen_aligned_100M.jsonl": sha256_file(qwen_train_path),
            "clean_pairs.jsonl": sha256_file(clean_pairs_path),
            "selected_pairs.jsonl": sha256_file(selected_pairs_path),
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }
    meta_path = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/clean_materialization_metadata.json')
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({
        "status": metadata["status"],
        "clean_pairs_available": metadata["clean_pairs_available"],
        "selected_pairs": metadata["selected_pairs"],
        "selected_pair_words": metadata["selected_pair_words"],
        "selected_pair_word_fraction": metadata["selected_pair_word_fraction"],
        "target_fraction_reached": metadata["target_pair_word_fraction_reached"],
        "selected_cohort_words": metadata["selected_cohort_words"],
        "treatment_pool_rows": metadata["treatment_pool_rows"],
        "row_length_sequence_matched": metadata["row_length_sequence_matched"],
        "rejection_top": rejection_reasons.most_common(10),
        "metadata": str(meta_path),
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
