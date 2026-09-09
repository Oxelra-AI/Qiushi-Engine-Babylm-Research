#!/usr/bin/env python3
"""Materialize a matched semantic-view contrast corpus for BabyLM Strict-Small.

Scientific purpose
------------------
research showed that Qwen simplification/paraphrase outputs are much safer than
free factual expansions, but they do not add
broad new facts.  They are multiple linguistic views of the same SimpleWiki
source rows.  To test that mechanism without mixing it with repeated source
allocation, this script builds two 10M-word pools with identical row-length
sequences and identical official filler:

  1. semantic_view_treatment: for each accepted source row, concatenate the
     original SimpleWiki source with accepted simplification/paraphrase views.
  2. original_packet_local: for each treatment row, use only that row's own
     original source words, starting with the exact source and then using
     balanced cyclic offsets for any extra repetitions to exactly the same row
     length.  This preserves row identity, topic boundary, length,
     and repetition while removing generated wording.
  3. original_stream_matched: use the same selected SimpleWiki sources as a
     shuffled/cycled source stream chunked to the treatment row-length sequence.
     This remains useful for measuring packing effects but can splice unrelated
     articles within a row.
  4. original_packet_exact_matched: naive prefix repetition of each row's source,
     kept only as a forensic artifact because it over-emphasizes source starts.

The remaining budget is filled by identical official rows.  The resulting 100M
training JSONL files repeat the 10M pool for 10 shuffled passes using identical
per-pass orders across all arms.  All word counts are exact and recorded.

The script intentionally avoids the unsafe expansion route.  It is a corpus
construction/checking tool, not an evaluator.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import random
import re
import statistics
import time
from dataclasses import dataclass, asdict
from typing import Iterable

ROOT = pathlib.Path("experiments/archive/representation_and_objectives")
DEFAULT_PROMPTS = ROOT / "training/data/factual_prompts_shard1_simpara.jsonl"
DEFAULT_OUTPUTS = ROOT / "training/runs/gen_simpara_full_qwen/outputs.jsonl"
DEFAULT_OUT = ROOT / "training/data/semantic_view/full_contrast"
OFFICIAL_POOL = pathlib.Path("experiments/archive/compact_experience/data/mixture/official_pool.jsonl")
SLICE_SUMMARY = ROOT / "training/data/generation_slice/slice_quality_summary.json"

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[’'][A-Za-z0-9]+)?|\*[A-Za-z]+:?|[£$€]\s*\d+(?:\.\d+)?|\d+(?:[.,:/-]\d+)*(?:%|st|nd|rd|th)?")
NUM_RE = re.compile(r"(?:[£$€]\s*)?\d+(?:[.,:/-]\d+)*(?:%|st|nd|rd|th)?", re.I)
BAD_PATTERNS = [
    re.compile(p, re.I) for p in [
        r"^\s*(sure|here('| i)s|certainly|of course)\b",
        r"as an ai", r"i (cannot|can't)", r"please provide", r"output only",
        r"simplified text\s*:", r"rewritten sentence\s*:", r"rewrite\s*:", r"output\s*:",
        r"the original text", r"not named in the original", r"potentially a confusion",
        r"while the specific .* is not named",
    ]
]
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by", "can", "could",
    "did", "do", "does", "for", "from", "had", "has", "have", "he", "her", "hers", "him", "his",
    "i", "if", "in", "into", "is", "it", "its", "just", "may", "might", "must", "not", "of", "on",
    "or", "our", "she", "should", "so", "some", "such", "than", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "to", "too", "under", "up", "very", "was", "we", "were",
    "what", "when", "where", "which", "who", "will", "with", "would", "you", "your", "one", "only",
    "more", "most", "all", "any", "both", "each", "every", "own", "same", "other", "through",
}
TRAILING_FRAGMENT_WORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "from", "with", "without", "by", "as",
    "than", "that", "which", "who", "whose", "whom", "when", "where", "while", "if", "because", "but",
    "and", "or", "nor", "so", "then", "he", "she", "it", "they", "we", "you", "i", "his", "her", "their",
    "our", "your", "my", "not", "no", "had", "has", "have", "was", "were", "is", "are", "be", "been",
    "being", "do", "does", "did", "will", "would", "could", "should", "may", "might", "must",
}
TRAILING_ABBREVIATIONS = {"mr.", "mrs.", "ms.", "dr.", "prof.", "st.", "jr.", "sr.", "no.", "fig."}
BAD_START_LOWER = {"and", "but", "or", "which", "whose", "whom", "soon", "of"}
BAD_START_ANYCASE = {"as soon", "which", "of"}

DOMAIN_TERMS = {
    "science_physical": {"volcanic","lava","glacier","molecular","cloud","dark","matter","galaxy","energy","electric","chemical","silicon","virus","respiratory","species","freshwater","cyclone","weather","planet","star","comet","biology","engineering"},
    "geography_places": {"city","province","county","district","municipality","canton","region","capital","river","mountain","island","country","state","switzerland","italy","iceland","australia","canada","hong","kong"},
    "people_history": {"born","died","served","mayor","governor","senate","war","emperor","lawyer","professor","leader","olympics","champion","president","secretary","minister","actress","writer"},
    "institutions_society": {"company","university","club","government","council","education","law","act","organization","competition","factory","brand","team","league","prize","school","bank"},
    "media_culture": {"film","album","band","song","movie","game","book","published","released","television","magazine","novel","music","dragon","streetlight"},
    "quant_numeric": {"january","february","march","april","may","june","july","august","september","october","november","december","km","metres","feet","century","year","million","percent"},
    "causal_relational": {"because","therefore","caused","cause","effect","result","led","leading","formed","became","created","due","reason","replaced","allows","requires","helps","influenced"},
}


@dataclass
class PromptRow:
    row_index: int
    prompt_id: str
    typ: str
    source_text: str
    source_words: int
    source_article: str
    source_key: str


@dataclass
class AcceptedView:
    prompt_id: str
    typ: str
    source_key: str
    output: str
    output_words: int
    len_ratio: float
    content_overlap: float
    entity_recall: float
    new_entities: int
    number_recall: float
    new_numbers: int


@dataclass
class CorpusRow:
    text: str
    words: int
    source: str
    example_id: int
    source_key: str | None = None
    source_article: str | None = None
    component: str | None = None
    view_types: list[str] | None = None
    prompt_ids: list[str] | None = None


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def normalize_ws(text: str) -> str:
    return " ".join(str(text).replace("\u00a0", " ").split())


def strip_outer_quotes(text: str) -> str:
    text = normalize_ws(text.strip())
    changed = True
    while changed and len(text) >= 2:
        changed = False
        for a, b in [("\"", "\""), ("'", "'"), ("“", "”"), ("‘", "’")]:
            if text.startswith(a) and text.endswith(b):
                text = normalize_ws(text[1:-1])
                changed = True
    return text


def clean_output(raw: str) -> str:
    text = strip_outer_quotes(raw or "")
    low = text.lower()
    for marker in ("simplified text:", "rewritten sentence:", "rewrite:", "output:", "sentence:"):
        if low.startswith(marker):
            text = strip_outer_quotes(text[len(marker):])
            low = text.lower()
    return text


def norm_tokens(text: str) -> list[str]:
    toks: list[str] = []
    for m in WORD_RE.finditer(text):
        t = m.group(0).strip("'\"“”‘’.,!?;:()[]{}")
        if t:
            toks.append(t.lower())
    return toks


def content_tokens(text: str) -> list[str]:
    out: list[str] = []
    for t in norm_tokens(text):
        if len(t) >= 3 and t not in STOPWORDS:
            out.append(t)
    return out


def number_tokens(text: str) -> list[str]:
    return [re.sub(r"\s+", "", m.group(0).lower()) for m in NUM_RE.finditer(text)]


def entity_tokens(text: str) -> list[str]:
    raw = text.split()
    ents: list[str] = []
    for i, w in enumerate(raw):
        stripped = w.strip("\"“”‘’.,!?;:()[]{}")
        if not stripped:
            continue
        if stripped.startswith("*") and len(stripped) > 2:
            ents.append(stripped.lower().rstrip(":"))
            continue
        letters = re.sub(r"[^A-Za-z]", "", stripped)
        if not letters:
            continue
        is_cap = letters[0].isupper() and len(letters) > 1
        has_internal_cap = any(c.isupper() for c in letters[1:])
        all_caps = len(letters) >= 2 and letters.isupper()
        if is_cap or has_internal_cap or all_caps:
            if i == 0 and not has_internal_cap and not all_caps and not stripped.startswith("*"):
                continue
            if stripped.lower() in STOPWORDS:
                continue
            ents.append(stripped.lower())
    seen = set()
    dedup: list[str] = []
    for e in ents:
        if e not in seen:
            dedup.append(e)
            seen.add(e)
    return dedup


def starts_bad_fragment(text: str) -> bool:
    s = text.lstrip(" \t\"'“”‘’(")
    if not s:
        return True
    first = s.split()[0].strip(".,!?;:()[]{}\"'“”‘’") if s.split() else ""
    low = s.lower()
    if any(low.startswith(p + " ") or low == p for p in BAD_START_ANYCASE):
        if not text.lstrip().startswith(("\"", "'", "“", "*")):
            return True
    if first and first[0].islower() and first.lower() in BAD_START_LOWER:
        return True
    return False


def has_repetition(text: str) -> bool:
    toks = norm_tokens(text)
    if len(toks) < 8:
        return False
    run = 1
    for a, b in zip(toks, toks[1:]):
        run = run + 1 if a == b else 1
        if run >= 4:
            return True
    grams = [tuple(toks[i:i+4]) for i in range(len(toks)-3)]
    return any(v >= 2 for v in collections.Counter(grams).values())


def complete_enough(text: str) -> bool:
    s = normalize_ws(text).strip()
    if not s:
        return False
    low = s.lower()
    if "..." in s or "…" in s or "http" in low:
        return False
    if starts_bad_fragment(s):
        return False
    stripped = s.rstrip(' \t\"\'”’)]}')
    if not stripped:
        return False
    last = stripped.split()[-1].strip(' \t\"\'“”‘’()[]{}')
    if last.lower() in TRAILING_FRAGMENT_WORDS or last.lower() in TRAILING_ABBREVIATIONS:
        return False
    if not re.search(r"[.!?][\"'”’\)\]]*$", s):
        return False
    if s.count("[") != s.count("]") or s.count("(") != s.count(")"):
        return False
    if s.count('"') % 2 == 1 or s.count("“") != s.count("”") or s.count("‘") != s.count("’"):
        return False
    return True


def overlap_score(source: str, output: str) -> float:
    a = set(content_tokens(source))
    b = set(content_tokens(output))
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, min(len(a), len(b)))


def domain_hits(text: str) -> list[str]:
    toks = set(content_tokens(text))
    return [k for k, v in DOMAIN_TERMS.items() if toks & v]


def validate_view(pr: PromptRow, raw_output: str, strict_numbers: bool) -> tuple[AcceptedView | None, list[str]]:
    reasons: list[str] = []
    out = clean_output(raw_output)
    if not out:
        return None, ["empty_output"]
    if "\n" in out or "\r" in out:
        reasons.append("newline")
    if any(rx.search(out) for rx in BAD_PATTERNS):
        reasons.append("bad_pattern")
    if not complete_enough(out):
        reasons.append("incomplete_or_fragment")
    if has_repetition(out):
        reasons.append("repetition")
    ow = pr.source_words
    rw = len(out.split())
    if rw < 5:
        reasons.append(f"too_short_{rw}")
    if rw > 90:
        reasons.append(f"too_long_{rw}")
    ratio = rw / max(1, ow)
    low_ratio = 0.35 if pr.typ == "simplification" else 0.45
    if ratio < low_ratio or ratio > 2.25:
        reasons.append(f"len_ratio_{ratio:.2f}")

    src_nums = number_tokens(pr.source_text)
    out_nums = number_tokens(out)
    src_num_counter = collections.Counter(src_nums)
    out_num_counter = collections.Counter(out_nums)
    if src_nums:
        matched_nums = sum(min(src_num_counter[k], out_num_counter[k]) for k in src_num_counter)
        nrec = matched_nums / max(1, len(src_nums))
    else:
        nrec = 1.0
    new_numbers = sum((out_num_counter - src_num_counter).values())
    if strict_numbers and src_num_counter != out_num_counter:
        reasons.append("number_mismatch")
    elif nrec < 0.67 or new_numbers > 1:
        reasons.append(f"number_risk_rec{nrec:.2f}_new{new_numbers}")

    src_ents = entity_tokens(pr.source_text)
    out_ents = entity_tokens(out)
    out_norm = set(out_ents)
    if src_ents:
        matched = 0
        for e in src_ents:
            if e in out_norm or e.replace("'s", "") in out_norm or e.rstrip("s") in out_norm:
                matched += 1
        erec = matched / len(src_ents)
    else:
        erec = 1.0
    new_entities = len(set(out_ents) - set(src_ents))
    needed = 1.0 if 0 < len(src_ents) <= 2 else 0.75
    if erec < needed:
        reasons.append(f"entity_recall_{erec:.2f}")
    if new_entities > 3:
        reasons.append(f"new_entities_{new_entities}")

    ov = overlap_score(pr.source_text, out)
    if ov < 0.10 and pr.source_words >= 10:
        reasons.append(f"low_content_overlap_{ov:.2f}")
    if ov > 0.96 and ratio > 0.85:
        reasons.append(f"copy_overlap_{ov:.2f}")

    if reasons:
        return None, reasons
    return AcceptedView(
        prompt_id=pr.prompt_id, typ=pr.typ, source_key=pr.source_key,
        output=out, output_words=rw, len_ratio=ratio, content_overlap=ov,
        entity_recall=erec, new_entities=new_entities, number_recall=nrec, new_numbers=new_numbers,
    ), []


def load_prompts(path: pathlib.Path) -> list[PromptRow]:
    rows: list[PromptRow] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            o = json.loads(line)
            typ = str(o.get("type", ""))
            if typ not in {"simplification", "paraphrase"}:
                continue
            source = normalize_ws(str(o["source_text"]))
            sw = int(o.get("source_words", len(source.split())))
            if sw != len(source.split()):
                raise RuntimeError(f"prompt row {i} word mismatch field={sw} actual={len(source.split())}")
            rows.append(PromptRow(
                row_index=int(o.get("slice_index", i)),
                prompt_id=str(o.get("id", f"row_{i}")),
                typ=typ,
                source_text=source,
                source_words=sw,
                source_article=str(o.get("source_article", "")),
                source_key=sha256_text(source),
            ))
    return rows


def load_outputs_by_index(path: pathlib.Path) -> dict[int, str]:
    out: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for fallback_i, line in enumerate(f):
            if not line.strip():
                continue
            o = json.loads(line)
            idx = int(o.get("index", fallback_i))
            text = str(o.get("output", o.get("generated_text", o.get("text", ""))))
            out[idx] = text
    return out


def select_accepted_views_for_use(accepted: list[AcceptedView], max_per_type_per_source: int) -> tuple[list[AcceptedView], dict]:
    """Keep a bounded number of accepted views of each type for each source text.

    The prompt shard can contain duplicate exact source texts.  If all accepted
    duplicate outputs are concatenated, a few packets become many-view rows rather
    than the intended original+simplification+paraphrase semantic packet.  Capping
    preserves a clean mechanism test while keeping validation rows available for
    forensic inspection.
    """
    before_by_key: dict[str, int] = collections.Counter(v.source_key for v in accepted)
    if max_per_type_per_source <= 0:
        return accepted, {
            "max_per_type_per_source": max_per_type_per_source,
            "cap_active": False,
            "available_views": len(accepted),
            "used_views": len(accepted),
            "dropped_views": 0,
            "dropped_by_type": {},
            "source_keys_with_drops": 0,
            "max_views_per_source_before": max(before_by_key.values()) if before_by_key else 0,
            "max_views_per_source_after": max(before_by_key.values()) if before_by_key else 0,
        }
    counts: dict[tuple[str, str], int] = collections.Counter()
    used: list[AcceptedView] = []
    dropped: list[AcceptedView] = []
    for v in accepted:
        k = (v.source_key, v.typ)
        if counts[k] < max_per_type_per_source:
            used.append(v)
            counts[k] += 1
        else:
            dropped.append(v)
    after_by_key: dict[str, int] = collections.Counter(v.source_key for v in used)
    drop_keys = {v.source_key for v in dropped}
    return used, {
        "max_per_type_per_source": max_per_type_per_source,
        "cap_active": True,
        "available_views": len(accepted),
        "used_views": len(used),
        "dropped_views": len(dropped),
        "dropped_by_type": dict(collections.Counter(v.typ for v in dropped)),
        "source_keys_with_drops": len(drop_keys),
        "max_views_per_source_before": max(before_by_key.values()) if before_by_key else 0,
        "max_views_per_source_after": max(after_by_key.values()) if after_by_key else 0,
    }


def stats(vals: Iterable[int | float]) -> dict:
    xs = list(vals)
    if not xs:
        return {"n": 0}
    ys = sorted(xs)
    def q(p: float):
        return ys[min(len(ys)-1, max(0, round((len(ys)-1)*p)))]
    return {
        "n": len(xs), "min": min(xs), "p05": q(0.05),
        "mean": round(statistics.mean(xs), 4), "median": statistics.median(xs),
        "p95": q(0.95), "max": max(xs), "sum": sum(xs),
    }


def read_official_pool(path: pathlib.Path, total_words: int) -> list[CorpusRow]:
    rows: list[CorpusRow] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            o = json.loads(line)
            text = normalize_ws(o["text"])
            w = int(o.get("words", len(text.split())))
            if w != len(text.split()):
                raise RuntimeError(f"official row {i} word mismatch")
            rows.append(CorpusRow(text=text, words=w, source=str(o.get("source", "official")), example_id=int(o.get("example_id", i)), component="official_filler"))
    if sum(r.words for r in rows) < total_words:
        raise RuntimeError(f"official pool has only {sum(r.words for r in rows)} words < requested {total_words}")
    return rows


def choose_official_filler(official_rows: list[CorpusRow], needed_words: int, seed: int) -> list[CorpusRow]:
    """Choose official filler exactly, using full rows and one final word chunk if needed."""
    rng = random.Random(seed)
    rows = list(official_rows)
    rng.shuffle(rows)
    chosen: list[CorpusRow] = []
    total = 0
    i = 0
    while total < needed_words:
        r = rows[i % len(rows)]
        remain = needed_words - total
        if r.words <= remain:
            chosen.append(CorpusRow(text=r.text, words=r.words, source=r.source, example_id=r.example_id, component="official_filler"))
            total += r.words
        else:
            ws = r.text.split()[:remain]
            chosen.append(CorpusRow(text=" ".join(ws), words=remain, source=f"{r.source}::partial_filler", example_id=900000 + len(chosen), component="official_filler_partial"))
            total += remain
        i += 1
    assert total == needed_words
    return chosen


def repeated_original_to_length(source_text: str, target_words: int) -> str:
    """Packet-exact original-only text, kept as a secondary forensic contrast.

    This is not used as the primary control because it can create row-final
    fragments and over-represent source prefixes.  The primary control below is
    a selected-source stream chunked to the treatment row-length sequence.
    """
    base = source_text.split()
    if not base:
        raise ValueError("empty source")
    out: list[str] = []
    while len(out) < target_words:
        out.extend(base)
    return " ".join(out[:target_words])


def sentence_word_units(text: str) -> list[list[str]]:
    """Split text into approximate sentence word units while preserving words."""
    words = text.split()
    if not words:
        return []
    units: list[list[str]] = []
    cur: list[str] = []
    for w in words:
        cur.append(w)
        stripped = w.rstrip('"\'”’)]}')
        if stripped.endswith((".", "!", "?")):
            units.append(cur)
            cur = []
    if cur:
        units.append(cur)
    return units or [words]


def balanced_packet_local_original_to_length(source_text: str, target_words: int, offset: int) -> str:
    """Packet-local original-only text with sentence-boundary repetition.

    Uses ONLY this row's own source words.  The row starts with the exact source
    text, matching the treatment prefix (which also begins with the source), so
    the control is not artificially hurt by an initial mid-sentence fragment.
    If more words are needed, later repetitions rotate complete sentence units
    rather than arbitrary word offsets.  This preserves row identity, topical
    boundary (single source article per row), exact row length, source-only
    repetition, and more natural local boundaries while removing generated
    wording.  The last appended sentence may still be truncated only when exact
    word accounting requires it.

    This is the preferred packet-local comparison for distinguishing generated
    paraphrastic variation from coherent same-source repetition: the control row
    is equally coherent and equally local to the same source packet, but contains
    no Qwen simplification/paraphrase text.
    """
    base = source_text.split()
    n = len(base)
    if n == 0:
        raise ValueError("empty source")
    if target_words <= n:
        return " ".join(base[:target_words])
    units = sentence_word_units(source_text)
    m = len(units)
    out: list[str] = list(base)
    copy_i = 0
    while len(out) < target_words:
        start = (offset + copy_i) % max(1, m)
        ordered_units = units[start:] + units[:start]
        for unit in ordered_units:
            if len(out) >= target_words:
                break
            remain = target_words - len(out)
            out.extend(unit[:remain])
        copy_i += 1
        if copy_i > 10000:
            raise RuntimeError("unexpectedly many packet-local repetitions")
    return " ".join(out[:target_words])


def build_source_stream_matched_control(
    treatment_semantic_rows: list[CorpusRow],
    source_key_to_prompt: dict[str, PromptRow],
    seed: int,
) -> tuple[list[CorpusRow], list[dict]]:
    """Build the primary original-only control for the semantic-view packet prefix.

    It uses only the exact SimpleWiki source texts that appear in the treatment
    semantic packets.  The selected originals are shuffled, concatenated, cycled,
    and chunked to the treatment row-length sequence.  This matches source
    article set, original-source pool, total semantic-prefix word exposure, and
    row lengths without inserting Qwen transformations.  Compared with rowwise
    source-prefix repetition it avoids repeatedly beginning every control row at
    word 0 of one source sentence.
    """
    selected: list[PromptRow] = []
    for r in treatment_semantic_rows:
        if not r.source_key or r.source_key not in source_key_to_prompt:
            raise RuntimeError(f"missing source prompt for treatment row {r.example_id}: {r.source_key}")
        selected.append(source_key_to_prompt[r.source_key])
    lengths = [r.words for r in treatment_semantic_rows]
    total_needed = sum(lengths)
    rng = random.Random(seed)
    stream_words: list[str] = []
    stream_articles: list[str] = []
    stream_keys: list[str] = []
    cycle = 0
    base_indices = list(range(len(selected)))
    while len(stream_words) < total_needed:
        order = list(base_indices)
        rng.shuffle(order)
        for idx in order:
            p = selected[idx]
            ws = p.source_text.split()
            stream_words.extend(ws)
            stream_articles.extend([p.source_article] * len(ws))
            stream_keys.extend([p.source_key] * len(ws))
            if len(stream_words) >= total_needed:
                break
        cycle += 1
        if cycle > 100000:
            raise RuntimeError("unexpectedly many cycles while building source stream control")
    rows: list[CorpusRow] = []
    meta: list[dict] = []
    pos = 0
    for i, L in enumerate(lengths):
        ws = stream_words[pos:pos+L]
        arts = stream_articles[pos:pos+L]
        keys = stream_keys[pos:pos+L]
        if len(ws) != L:
            raise RuntimeError("source stream ended early")
        art_counts = collections.Counter(arts)
        key_counts = collections.Counter(keys)
        common_article = art_counts.most_common(1)[0][0] if art_counts else ""
        common_key = key_counts.most_common(1)[0][0] if key_counts else ""
        rows.append(CorpusRow(
            text=" ".join(ws), words=L, source="simplewiki_original_stream_matched",
            example_id=400000 + i, source_key=common_key, source_article=common_article,
            component="original_stream_matched_packet",
        ))
        meta.append({
            "row_index": i,
            "example_id": 400000 + i,
            "words": L,
            "common_source_article": common_article,
            "source_article_counts": dict(art_counts),
            "source_key_counts": dict(key_counts),
            "starts_at_stream_word": pos,
        })
        pos += L
    assert pos == total_needed
    assert sum(r.words for r in rows) == total_needed
    return rows, meta


def build_semantic_rows(prompts: list[PromptRow], accepted: list[AcceptedView], seed: int) -> tuple[list[CorpusRow], list[CorpusRow], list[CorpusRow], list[dict]]:
    prompt_by_key: dict[str, PromptRow] = {}
    for p in prompts:
        prompt_by_key.setdefault(p.source_key, p)
    views_by_key: dict[str, list[AcceptedView]] = collections.defaultdict(list)
    for v in accepted:
        views_by_key[v.source_key].append(v)
    groups = []
    for key, views in views_by_key.items():
        p = prompt_by_key[key]
        views_sorted = sorted(views, key=lambda v: {"simplification": 0, "paraphrase": 1}.get(v.typ, 9))
        text_parts = [p.source_text] + [v.output for v in views_sorted]
        text = normalize_ws(" ".join(text_parts))
        w = len(text.split())
        if w <= 0:
            continue
        groups.append((key, p, views_sorted, text, w))
    rng = random.Random(seed)
    rng.shuffle(groups)
    # Prefer rows with both generated views, then deterministic shuffled order within groups.
    groups.sort(key=lambda item: (-len(item[2]), item[1].source_article, item[1].prompt_id))

    treat: list[CorpusRow] = []
    ctrl: list[CorpusRow] = []
    ctrl_local: list[CorpusRow] = []
    meta_rows: list[dict] = []
    for idx, (key, p, views, text, w) in enumerate(groups):
        view_types = [v.typ for v in views]
        prompt_ids = [v.prompt_id for v in views]
        example_id = 400000 + idx
        treat.append(CorpusRow(
            text=text, words=w, source="simplewiki_semantic_view", example_id=example_id,
            source_key=key, source_article=p.source_article, component="semantic_view_packet",
            view_types=view_types, prompt_ids=prompt_ids,
        ))
        ctrl_text = repeated_original_to_length(p.source_text, w)
        ctrl.append(CorpusRow(
            text=ctrl_text, words=w, source="simplewiki_original_only_matched", example_id=example_id,
            source_key=key, source_article=p.source_article, component="original_only_matched_packet",
            view_types=view_types, prompt_ids=prompt_ids,
        ))
        # Packet-local cyclic control: same single source, same length, offset repetition.
        ctrl_local_text = balanced_packet_local_original_to_length(p.source_text, w, offset=idx * 3 + 1)
        ctrl_local.append(CorpusRow(
            text=ctrl_local_text, words=w, source="simplewiki_original_packet_local", example_id=example_id,
            source_key=key, source_article=p.source_article, component="original_packet_local_packet",
            view_types=view_types, prompt_ids=prompt_ids,
        ))
        meta_rows.append({
            "row_index": idx,
            "example_id": example_id,
            "source_key": key,
            "source_article": p.source_article,
            "source_words": p.source_words,
            "row_words": w,
            "view_types": view_types,
            "prompt_ids": prompt_ids,
            "view_words": {v.typ: v.output_words for v in views},
            "view_quality": {v.typ: {"len_ratio": round(v.len_ratio, 4), "content_overlap": round(v.content_overlap, 4), "entity_recall": round(v.entity_recall, 4), "new_entities": v.new_entities, "number_recall": round(v.number_recall, 4), "new_numbers": v.new_numbers} for v in views},
            "source_text": p.source_text,
        })
    return treat, ctrl, ctrl_local, meta_rows


def truncate_semantic_budget(treat: list[CorpusRow], ctrl: list[CorpusRow], ctrl_local: list[CorpusRow], meta_rows: list[dict], max_words: int) -> tuple[list[CorpusRow], list[CorpusRow], list[CorpusRow], list[dict]]:
    total = 0
    n = 0
    for r in treat:
        if total + r.words > max_words:
            break
        total += r.words
        n += 1
    return treat[:n], ctrl[:n], ctrl_local[:n], meta_rows[:n]


def write_pool(path: pathlib.Path, rows: list[CorpusRow], meta_path: pathlib.Path | None = None) -> None:
    total = 0
    with path.open("w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            if r.words != len(r.text.split()):
                raise RuntimeError(f"{path} row {i} words mismatch {r.words} vs {len(r.text.split())}")
            f.write(json.dumps({"text": r.text, "words": r.words, "example_id": r.example_id, "source": r.source}, ensure_ascii=False) + "\n")
            total += r.words
    if meta_path:
        with meta_path.open("w", encoding="utf-8") as f:
            for i, r in enumerate(rows):
                if r.component and r.component != "official_filler" and r.component != "official_filler_partial":
                    f.write(json.dumps({
                        "row_index": i, "example_id": r.example_id, "words": r.words,
                        "source": r.source, "source_key": r.source_key, "source_article": r.source_article,
                        "component": r.component, "view_types": r.view_types, "prompt_ids": r.prompt_ids,
                    }, ensure_ascii=False) + "\n")


def write_training(path: pathlib.Path, rows: list[CorpusRow], pass_orders: list[list[int]]) -> None:
    total = 0
    with path.open("w", encoding="utf-8") as f:
        for order in pass_orders:
            for idx in order:
                r = rows[idx]
                f.write(json.dumps({"text": r.text, "words": r.words, "example_id": r.example_id, "source": r.source}, ensure_ascii=False) + "\n")
                total += r.words
    expected = sum(r.words for r in rows) * len(pass_orders)
    assert total == expected


def row_source_words(rows: list[CorpusRow]) -> dict[str, int]:
    c = collections.Counter()
    for r in rows:
        c[r.source] += r.words
    return dict(c)


def make_report(args: argparse.Namespace) -> None:
    t0 = time.time()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(pathlib.Path(args.prompts))
    outputs = load_outputs_by_index(pathlib.Path(args.outputs))
    if not prompts:
        raise RuntimeError("no simplification/paraphrase prompts loaded")
    if not outputs:
        raise RuntimeError("no generation outputs loaded")

    accepted: list[AcceptedView] = []
    rejection_reasons = collections.Counter()
    missing_outputs = 0
    raw_rows = []
    for p in prompts:
        if p.row_index not in outputs:
            missing_outputs += 1
            continue
        view, reasons = validate_view(p, outputs[p.row_index], strict_numbers=args.strict_numbers)
        if view is None:
            for r in reasons:
                rejection_reasons[r.split("_")[0] if r.startswith(("too_", "len_", "entity_", "number_", "new_", "low_", "copy_")) else r] += 1
            raw_rows.append({"prompt_id": p.prompt_id, "type": p.typ, "source_key": p.source_key, "accepted": False, "reasons": reasons, "source_article": p.source_article, "source_words": p.source_words, "output": outputs.get(p.row_index, "")[:300]})
        else:
            accepted.append(view)
            raw_rows.append({"prompt_id": p.prompt_id, "type": p.typ, "source_key": p.source_key, "accepted": True, **asdict(view)})

    used_accepted, accepted_use_policy = select_accepted_views_for_use(accepted, args.max_per_type_per_source)
    treat_sem, ctrl_packet_exact_sem, ctrl_packet_local_sem, semantic_meta = build_semantic_rows(prompts, used_accepted, args.seed)
    treat_sem, ctrl_packet_exact_sem, ctrl_packet_local_sem, semantic_meta = truncate_semantic_budget(treat_sem, ctrl_packet_exact_sem, ctrl_packet_local_sem, semantic_meta, args.max_semantic_words)
    source_key_to_prompt = {p.source_key: p for p in prompts}
    ctrl_stream_sem, ctrl_stream_meta = build_source_stream_matched_control(treat_sem, source_key_to_prompt, args.seed + 7)
    semantic_words = sum(r.words for r in treat_sem)
    if semantic_words == 0:
        raise RuntimeError("no accepted semantic rows after filtering")
    if semantic_words > args.total_words:
        raise RuntimeError(f"semantic words {semantic_words} exceed total_words {args.total_words}")
    official_rows = read_official_pool(pathlib.Path(args.official_pool), args.total_words)
    filler_needed = args.total_words - semantic_words
    filler = choose_official_filler(official_rows, filler_needed, args.seed + 1)
    treatment_pool = treat_sem + filler
    control_pool = ctrl_stream_sem + filler
    packet_exact_control_pool = ctrl_packet_exact_sem + filler
    packet_local_control_pool = ctrl_packet_local_sem + filler
    assert sum(r.words for r in treatment_pool) == args.total_words
    assert sum(r.words for r in control_pool) == args.total_words
    assert sum(r.words for r in packet_exact_control_pool) == args.total_words
    assert sum(r.words for r in packet_local_control_pool) == args.total_words
    assert [r.words for r in treatment_pool] == [r.words for r in control_pool]
    assert [r.words for r in treatment_pool] == [r.words for r in packet_exact_control_pool]
    assert [r.words for r in treatment_pool] == [r.words for r in packet_local_control_pool]

    rng = random.Random(args.seed + 2)
    pass_orders: list[list[int]] = []
    base = list(range(len(treatment_pool)))
    for pass_i in range(args.passes):
        order = list(base)
        rng.shuffle(order)
        pass_orders.append(order)

    paths = {
        "treatment_pool": out_dir / "semantic_view_treatment_10M.jsonl",
        "control_pool": out_dir / "original_stream_matched_10M.jsonl",
        "packet_local_control_pool": out_dir / "original_packet_local_10M.jsonl",
        "packet_exact_control_pool": out_dir / "original_packet_exact_matched_10M.jsonl",
        "treatment_training": out_dir / "semantic_view_treatment_100M.jsonl",
        "control_training": out_dir / "original_stream_matched_100M.jsonl",
        "packet_local_control_training": out_dir / "original_packet_local_100M.jsonl",
        "packet_exact_control_training": out_dir / "original_packet_exact_matched_100M.jsonl",
        "semantic_packet_meta": out_dir / "semantic_packet_rows_meta.jsonl",
        "matched_packet_meta": out_dir / "matched_packet_rows_meta.jsonl",
        "packet_local_meta": out_dir / "packet_local_rows_meta.jsonl",
        "stream_packet_meta": out_dir / "stream_matched_packet_rows_meta.jsonl",
        "accepted_views": out_dir / "accepted_views.jsonl",
        "view_validation_rows": out_dir / "view_validation_rows.jsonl",
        "materialization_metadata": out_dir / "semantic_view_materialization_metadata.json",
        "report_note": pathlib.Path(args.note),
    }

    write_pool(paths["treatment_pool"], treatment_pool, paths["semantic_packet_meta"])
    write_pool(paths["control_pool"], control_pool, paths["matched_packet_meta"])
    write_pool(paths["packet_local_control_pool"], packet_local_control_pool, paths["packet_local_meta"])
    write_pool(paths["packet_exact_control_pool"], packet_exact_control_pool, None)
    with paths["stream_packet_meta"].open("w", encoding="utf-8") as f:
        for r in ctrl_stream_meta:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if args.write_training:
        write_training(paths["treatment_training"], treatment_pool, pass_orders)
        write_training(paths["control_training"], control_pool, pass_orders)
        write_training(paths["packet_local_control_training"], packet_local_control_pool, pass_orders)
        write_training(paths["packet_exact_control_training"], packet_exact_control_pool, pass_orders)
    with paths["accepted_views"].open("w", encoding="utf-8") as f:
        for v in used_accepted:
            f.write(json.dumps(asdict(v), ensure_ascii=False) + "\n")
    with paths["view_validation_rows"].open("w", encoding="utf-8") as f:
        for r in raw_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    semantic_source_articles = collections.Counter()
    semantic_source_words_by_article = collections.Counter()
    selected_keys = {r.source_key for r in treat_sem if r.source_key}
    for key in selected_keys:
        p = source_key_to_prompt[key]
        semantic_source_articles[p.source_article] += 1
        # Count only original source words here; row_words includes generated views.
        semantic_source_words_by_article[p.source_article] += p.source_words
    domain_counter = collections.Counter()
    for key in selected_keys:
        for d in domain_hits(source_key_to_prompt[key].source_text):
            domain_counter[d] += 1

    metadata = {
        "status": "SEMANTIC_VIEW_CONTRAST_MATERIALIZED",
        "created_utc_unix": int(time.time()),
        "purpose": "matched semantic-view vs original-only source contrast under BabyLM strict-small word accounting",
        "prompts": str(args.prompts),
        "outputs": str(args.outputs),
        "prompt_sha256": sha256_file(pathlib.Path(args.prompts)),
        "output_sha256": sha256_file(pathlib.Path(args.outputs)),
        "official_pool": str(args.official_pool),
        "official_pool_sha256": sha256_file(pathlib.Path(args.official_pool)),
        "total_words_per_pool": args.total_words,
        "passes": args.passes,
        "word_exposure_per_training_file": args.total_words * args.passes,
        "strict_numbers": args.strict_numbers,
        "prompt_rows_loaded": len(prompts),
        "output_rows_loaded": len(outputs),
        "missing_outputs_for_prompts": missing_outputs,
        "accepted_views_available_before_use_cap": len(accepted),
        "accepted_views_available_by_type": dict(collections.Counter(v.typ for v in accepted)),
        "accepted_views_used": len(used_accepted),
        "accepted_views_used_by_type": dict(collections.Counter(v.typ for v in used_accepted)),
        "accepted_view_use_policy": accepted_use_policy,
        "accepted_views": len(used_accepted),
        "accepted_views_by_type": dict(collections.Counter(v.typ for v in used_accepted)),
        "rejected_prompt_outputs": len(prompts) - len(accepted) - missing_outputs,
        "rejection_reasons_prefix_counts": dict(rejection_reasons),
        "semantic_packet_rows": len(treat_sem),
        "semantic_packet_words": semantic_words,
        "semantic_packet_fraction_of_pool": round(semantic_words / args.total_words, 6),
        "official_filler_words_identical": filler_needed,
        "official_filler_rows_identical": len(filler),
        "row_length_sequence_identical": [r.words for r in treatment_pool] == [r.words for r in control_pool],
        "official_filler_identical_after_semantic_prefix": True,
        "control_construction": "three source-only controls are written: original_stream_matched chunks a shuffled/cycled stream of the same selected SimpleWiki sources to the treatment row-length sequence; original_packet_local is the packet-local balanced-repetition control preserving each row's own source identity, boundary, length, and repetition without generated wording (first copy starts with the exact source; extra repetitions use cyclic offsets); original_packet_exact_matched is a naive rowwise prefix-repeated original kept only as a forensic artifact because it over-emphasizes source prefixes",
        "pool_rows": {"treatment": len(treatment_pool), "stream_control": len(control_pool), "packet_local_control": len(packet_local_control_pool), "packet_exact_control": len(packet_exact_control_pool)},
        "semantic_row_word_stats": stats([r.words for r in treat_sem]),
        "semantic_original_source_word_stats": stats([source_key_to_prompt[str(r.source_key)].source_words for r in treat_sem if r.source_key]),
        "accepted_view_word_stats_by_type": {typ: stats([v.output_words for v in used_accepted if v.typ == typ]) for typ in ["simplification", "paraphrase"]},
        "accepted_view_quality_by_type": {
            typ: {
                "len_ratio": stats([v.len_ratio for v in used_accepted if v.typ == typ]),
                "content_overlap": stats([v.content_overlap for v in used_accepted if v.typ == typ]),
                "entity_recall": stats([v.entity_recall for v in used_accepted if v.typ == typ]),
                "number_recall": stats([v.number_recall for v in used_accepted if v.typ == typ]),
                "new_entities": stats([v.new_entities for v in used_accepted if v.typ == typ]),
                "new_numbers": stats([v.new_numbers for v in used_accepted if v.typ == typ]),
            } for typ in ["simplification", "paraphrase"]
        },
        "semantic_unique_source_keys": len(selected_keys),
        "semantic_source_articles": len(semantic_source_articles),
        "top_semantic_source_articles_by_rows": semantic_source_articles.most_common(30),
        "top_semantic_source_articles_by_original_words": semantic_source_words_by_article.most_common(30),
        "domain_row_hits_selected_sources": dict(domain_counter),
        "source_word_counts_treatment": row_source_words(treatment_pool),
        "source_word_counts_stream_control": row_source_words(control_pool),
        "source_word_counts_packet_local_control": row_source_words(packet_local_control_pool),
        "source_word_counts_packet_exact_control": row_source_words(packet_exact_control_pool),
        "files": {k: str(v) for k, v in paths.items()},
        "sha256": {k: sha256_file(v) for k, v in paths.items() if v.exists() and v.is_file() and k != "report_note"},
        "elapsed_sec": round(time.time() - t0, 3),
    }
    paths["materialization_metadata"].write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    note = []
    note.append("# research — matched semantic-view contrast materialization\n\n")
    note.append("## Purpose\n")
    note.append("This constructs the contrast required by the current research state: Qwen simplifications/paraphrases are treated as semantic views of existing SimpleWiki facts, not as a broad new factual source. It writes three source-only comparison arms. The packet-local balanced-repetition control is the cleanest transformation test: each row uses only its own original source words, starts with the exact source to match the treatment prefix, then uses cyclic offsets for any extra repetitions to reach the exact treatment row length; it preserves source identity, topical boundaries, row length, and repetition while removing generated wording. The stream control chunks a shuffled/cycled stream of the same selected SimpleWiki source texts to the same row lengths; it remains useful for detecting packing effects but can splice unrelated articles within a row. A naive prefix-repeated packet-exact file is kept only for forensic inspection because it over-emphasizes source prefixes. Official filler is identical in all arms.\n\n")
    note.append("## Corpus evidence\n")
    note.append(f"- Accepted generated views used: {len(used_accepted)} / {len(prompts)} prompts; available before per-source/type cap: {len(accepted)}; by type {metadata['accepted_views_by_type']}; use policy {json.dumps(accepted_use_policy, ensure_ascii=False)}.\n")
    note.append(f"- Semantic packet rows: {len(treat_sem)}; words {semantic_words:,} ({semantic_words/args.total_words:.2%} of the {args.total_words:,}-word pool).\n")
    note.append(f"- Identical official filler: {filler_needed:,} words in {len(filler):,} rows.\n")
    note.append(f"- Treatment/control row-length sequence identical: {metadata['row_length_sequence_identical']}.\n")
    note.append(f"- Unique selected source keys: {len(selected_keys)} across {len(semantic_source_articles)} SimpleWiki article labels.\n")
    note.append("- Domain row hits among selected source rows: " + json.dumps(dict(domain_counter), ensure_ascii=False) + "\n\n")
    note.append("## Interpretation for the next training run\n")
    note.append("A score difference between these two arms will be interpretable as evidence about linguistic-view transformation of the same source facts under matched word exposure and row lengths, not as broad factual/entity/causal coverage. If this contrast is weak or negative, the data route should reopen a genuinely broader verifiable factual source rather than adding more paraphrases of the same SimpleWiki rows.\n\n")
    note.append(f"Metadata JSON: `{paths['materialization_metadata']}`\n")
    paths["report_note"].parent.mkdir(parents=True, exist_ok=True)
    paths["report_note"].write_text("".join(note), encoding="utf-8")
    print(json.dumps({
        "status": metadata["status"],
        "accepted_views_available_before_cap": len(accepted),
        "accepted_views": len(used_accepted),
        "accepted_by_type": metadata["accepted_views_by_type"],
        "semantic_rows": len(treat_sem),
        "semantic_words": semantic_words,
        "filler_words": filler_needed,
        "row_length_sequence_identical": metadata["row_length_sequence_identical"],
        "metadata": str(paths["materialization_metadata"]),
        "note": str(paths["report_note"]),
    }, indent=2, ensure_ascii=False))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    p.add_argument("--outputs", default=str(DEFAULT_OUTPUTS))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--note", default=str((ROOT.parents[2] / 'research/notes/representation_and_objectives/semantic_view_contrast_materialization.md')))
    p.add_argument("--official-pool", default=str(OFFICIAL_POOL))
    p.add_argument("--total-words", type=int, default=10_000_000)
    p.add_argument("--passes", type=int, default=10)
    p.add_argument("--max-semantic-words", type=int, default=2_500_000)
    p.add_argument("--seed", type=int, default=82904)
    p.add_argument("--strict-numbers", action="store_true", help="require exact source/output numeric multiset equality")
    p.add_argument("--max-per-type-per-source", type=int, default=1, help="cap accepted generated views used per exact source text and type; default 1 yields at most one simplification and one paraphrase per packet")
    p.add_argument("--write-training", action="store_true", help="also write repeated exposure training JSONL files")
    args = p.parse_args()
    make_report(args)


if __name__ == "__main__":
    main()
