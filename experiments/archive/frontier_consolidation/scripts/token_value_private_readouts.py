#!/usr/bin/env python3
"""research: zero-training token-value and detached private-readout discriminators.

Scientific purpose
------------------
After the legal research, scale1.75, and U256 endpoints, the open question is not
whether a frozen model can exploit extra context when handed it.  If frozen research
predicts masked tokens better with the true source/neighbors, that means the
existing function already uses that structure when it is present.  The stronger
cheap question is whether true legal structure supplies a signal that a detached,
capacity-matched private readout can use to explain research residual token errors
better than matched false structure, while the research reference logits remain
unchanged.

This script implements two modes:
  * edit: legal source -> compact-rewrite directed changed-span targets.
  * discourse: natural intra-row triples, excluding synthetic paired rows.

For each mode it:
  1. builds single-mask examples from legal corpus artifacts only;
  2. compares frozen research NLL under base-only, true-structure, and false-structure contexts;
  3. trains equal-capacity residual private probes on detached [MASK] hidden states using
     the base-only frozen logits as an unchanged offset: CE(base_logits + probe(h), target);
  4. evaluates held-out and high-baseline-error subsets.

No pretraining, official evaluation, endpoint scoring, corpus modification, or model
parameter update is performed.
"""
from __future__ import annotations

import argparse
import collections
import csv
import difflib
import hashlib
import json
import math
import os
import random
import re
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# Keep Transformers dynamic/cache writes local if any model code path needs them.
STUDY = Path("experiments/archive/frontier_consolidation")
WORKSPACE = STUDY
DEFAULT_CACHE = WORKSPACE / "data/hf_cache"
os.environ.setdefault("HF_HOME", str(DEFAULT_CACHE / "hf_home"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(DEFAULT_CACHE / "transformers"))
os.environ.setdefault("HF_MODULES_CACHE", str(DEFAULT_CACHE / "modules"))
for _p in [os.environ["HF_HOME"], os.environ["TRANSFORMERS_CACHE"], os.environ["HF_MODULES_CACHE"]]:
    Path(_p).mkdir(parents=True, exist_ok=True)

import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

EXPECTED_POOL_SHA = "215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23"
EXPECTED_TOKENIZER_SHA = "91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9"

POOL_10M = WORKSPACE / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"
PAIR_JSONL = WORKSPACE / "data/density_core_reinvestment_medium_riskhard/selected_compact_reinvest_pairs.jsonl"
ROW_META_CANDIDATES = [
    WORKSPACE / "data/density_core_reinvestment_medium_riskhard/fineweb_compact_view_reinvest_changed_block_rows_meta.jsonl",
    WORKSPACE / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_changed_block_rows_meta.jsonl",
]
TOKENIZER_DIR = WORKSPACE / "data/compliant_tokenizer"
CKPT = WORKSPACE / "training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_100M"

NATURAL_SOURCES = {"childes", "gutenberg", "open_subtitles", "simple_wiki", "bnc_spoken", "switchboard"}
SYNTHETIC_SOURCES = {"qwen_pair_packed", "cleanqwen_fineweb_compact_view_reinvest"}

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*|[^\w\s]", re.UNICODE)
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=(?:[A-Z0-9\*\[]|[\"'“‘]))")
CHILDES_MARK_RE = re.compile(r"(?=\*[A-Z]{2,5}:)")
SWB_MARK_RE = re.compile(r"(?=\b[AB]:\s)")
SUBTITLE_DASH_RE = re.compile(r"\s+-\s+(?=[A-Z\[\-])")

FUNCTION_WORDS = {
    "a", "an", "the", "this", "that", "these", "those", "some", "any", "each", "every", "no", "not",
    "and", "or", "but", "if", "then", "because", "so", "while", "when", "before", "after", "although",
    "as", "than", "of", "in", "on", "at", "to", "for", "from", "by", "with", "without", "about", "into",
    "over", "under", "between", "through", "during", "is", "are", "was", "were", "be", "been", "being",
    "am", "do", "does", "did", "have", "has", "had", "can", "could", "will", "would", "shall", "should",
    "may", "might", "must", "it", "there", "here", "who", "what", "where", "why", "how", "which", "whose",
}
PRONOUNS = {
    "i", "me", "my", "mine", "you", "your", "yours", "he", "him", "his", "she", "her", "hers", "it", "its",
    "we", "us", "our", "ours", "they", "them", "their", "theirs", "myself", "yourself", "himself", "herself",
    "itself", "ourselves", "themselves", "who", "whom", "whose", "which", "that",
}
CONNECTIVES = {"and", "or", "but", "because", "so", "if", "then", "while", "when", "before", "after", "although", "though", "since", "until"}


@dataclass
class TokSpan:
    text: str
    norm: str
    start: int
    end: int
    is_word: bool


@dataclass
class PairRec:
    pair_id: str
    row_index: int
    example_id: int
    source_text: str
    rewrite_text: str
    source_words: int
    rewrite_words: int
    doc_id: str
    content_overlap: float
    content_recall: float


@dataclass
class TripleRec:
    row_index: int
    source: str
    example_id: int
    prev: str
    middle: str
    next: str
    prev_words: int
    mid_words: int
    next_words: int
    overlap_neighbors_mid: float


@dataclass
class MaskItem:
    item_id: str
    split_key: str
    source: str
    target_kind: str
    target_text: str
    target_id: int
    context_ids: dict[str, list[int]]
    meta: dict[str, Any] = field(default_factory=dict)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def token_spans(text: str) -> list[TokSpan]:
    return [TokSpan(m.group(0), m.group(0).lower(), m.start(), m.end(), any(ch.isalnum() for ch in m.group(0))) for m in WORD_RE.finditer(text)]


def tokenish(text: str) -> list[str]:
    return [t.norm for t in token_spans(text) if t.text.strip()]


def word_tokens(text: str) -> list[str]:
    return [t.norm for t in token_spans(text) if t.is_word]


def replace_span(text: str, span: TokSpan, repl: str) -> str:
    return text[:span.start] + repl + text[span.end:]


def words_count(text: str) -> int:
    return len(word_tokens(text))


def overlap(a: str, b: str) -> float:
    ca = collections.Counter(word_tokens(a)); cb = collections.Counter(word_tokens(b))
    inter = sum((ca & cb).values()); union = sum((ca | cb).values())
    return inter / union if union else 0.0


def mean(xs: Iterable[float]) -> float | None:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return statistics.mean(vals) if vals else None


def quantile(vals: list[float], q: float) -> float | None:
    vals = sorted(float(v) for v in vals if math.isfinite(float(v)))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = q * (len(vals) - 1)
    lo = int(math.floor(pos)); hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def summarize(vals: Iterable[float]) -> dict[str, Any]:
    vals = [float(v) for v in vals if math.isfinite(float(v))]
    if not vals:
        return {"n": 0, "mean": None, "median": None, "p05": None, "p25": None, "p75": None, "p95": None, "std": None, "min": None, "max": None}
    return {
        "n": len(vals),
        "mean": statistics.mean(vals),
        "median": statistics.median(vals),
        "p05": quantile(vals, 0.05),
        "p25": quantile(vals, 0.25),
        "p75": quantile(vals, 0.75),
        "p95": quantile(vals, 0.95),
        "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
    }


def bootstrap_mean_diff(a: list[float], b: list[float], seed: int, n_boot: int = 400) -> dict[str, Any]:
    # Positive diff means b - a positive if caller passes desired arrays accordingly.
    vals = [float(x) - float(y) for x, y in zip(a, b) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if not vals:
        return {"n": 0, "mean": None, "ci05": None, "ci95": None}
    rng = random.Random(seed)
    means = []
    n = len(vals)
    for _ in range(n_boot):
        means.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return {"n": n, "mean": statistics.mean(vals), "ci05": means[int(0.05 * (n_boot - 1))], "ci95": means[int(0.95 * (n_boot - 1))]}


def bin_float(x: float, cuts: list[float], labels: list[str]) -> str:
    for c, lab in zip(cuts, labels):
        if x < c:
            return lab
    return labels[-1]


def norm_pair_id(pid: str) -> str:
    return str(pid).split(":", 1)[-1] if str(pid).startswith("compact:") else str(pid)


def find_row_meta_path() -> Path:
    for p in ROW_META_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError("No changed-block row metadata found")


def load_pair_rows() -> dict[str, tuple[int, int]]:
    path = find_row_meta_path()
    out: dict[str, tuple[int, int]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            row_index = int(obj.get("row_index", obj.get("row_index_in_pool_0based", 0)))
            example_id = int(obj.get("example_id", -1))
            for pid in obj.get("pair_ids") or []:
                out[norm_pair_id(pid)] = (row_index, example_id)
    return out


def load_pairs() -> list[PairRec]:
    row_of = load_pair_rows()
    recs: list[PairRec] = []
    with PAIR_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            pid = norm_pair_id(obj.get("pair_id"))
            if pid not in row_of:
                continue
            src = str(obj.get("source_text") or obj.get("source") or "")
            rew = str(obj.get("rewrite_text") or obj.get("rewrite") or obj.get("compact") or "")
            if not src or not rew:
                continue
            row_index, example_id = row_of[pid]
            recs.append(PairRec(
                pair_id=pid,
                row_index=row_index,
                example_id=example_id,
                source_text=src,
                rewrite_text=rew,
                source_words=int(obj.get("source_words", words_count(src))),
                rewrite_words=int(obj.get("rewrite_words", words_count(rew))),
                doc_id=str(obj.get("doc_id", "")),
                content_overlap=float(obj.get("content_overlap", overlap(src, rew))),
                content_recall=float(obj.get("content_recall", math.nan)),
            ))
    recs.sort(key=lambda r: (r.row_index, r.pair_id))
    return recs


def pair_edit_changed_spans(pair: PairRec) -> list[TokSpan]:
    s = [t.norm for t in token_spans(pair.source_text)]
    r_spans = token_spans(pair.rewrite_text)
    r = [t.norm for t in r_spans]
    sm = difflib.SequenceMatcher(a=s, b=r, autojunk=False)
    changed: list[TokSpan] = []
    equal_anchor_ge3 = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal" and (i2 - i1) >= 3:
            equal_anchor_ge3 += 1
        if tag in {"insert", "replace"}:
            changed.extend(r_spans[j1:j2])
    if len([t for t in changed if t.is_word]) < 3 or equal_anchor_ge3 < 1:
        return []
    return changed


def target_kind(tok: str) -> str:
    t = tok.lower()
    if t in PRONOUNS:
        return "pronoun"
    if t in CONNECTIVES:
        return "connective"
    if t in FUNCTION_WORDS:
        return "function"
    if any(ch.isalpha() for ch in t):
        return "content"
    if any(ch.isdigit() for ch in t):
        return "number"
    return "punct_or_other"


def join_ctx(parts: list[str], sep: str) -> str:
    good = [p.strip() for p in parts if p and p.strip()]
    return f" {sep} ".join(good)


def single_piece_word_candidates(tokenizer, text: str, allowed_kinds: set[str] | None = None, drop_edges: bool = True) -> list[tuple[int, int, int, str]]:
    """Return (char_start, char_end, token_id, kind) for individual token pieces inside
    alphanumeric word spans in `text`. Whole-word single-token pieces are included, but split-word
    pieces are also allowed because the legal MLM operates on tokenizer pieces; the surrounding
    word determines kind. Masking exactly one existing token id preserves sequence length."""
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    offs = enc["offset_mapping"]
    spans = [s for s in token_spans(text) if s.is_word]
    if drop_edges and len(spans) >= 3:
        spans = spans[1:-1]
    out: list[tuple[int, int, int, str]] = []
    for sp in spans:
        k = target_kind(sp.text)
        if allowed_kinds is not None and k not in allowed_kinds:
            continue
        for i, (a, b) in enumerate(offs):
            if b <= a:
                continue
            if a < sp.start or b > sp.end:
                continue
            piece_text = text[a:b]
            if not any(ch.isalnum() for ch in piece_text):
                continue
            out.append((int(a), int(b), int(ids[i]), k))
    return out


def build_idlevel_contexts(tokenizer, base_text: str, base_char_start: int, base_char_end: int,
                           context_variants: dict[str, tuple[str, str]], max_length: int,
                           mask_id: int) -> tuple[dict[str, list[int]], int | None, str | None]:
    """Build token-id sequences for each context, masking exactly the single target token piece
    in the base text. context_variants maps key -> (prefix_text, suffix_text) placed around the
    base text with a separator; 'base' uses empty prefix/suffix.
    Returns (masked_id_lists, target_id, reason)."""
    base_enc = tokenizer(base_text, add_special_tokens=False, return_offsets_mapping=True)
    base_ids = list(base_enc["input_ids"])
    base_offs = base_enc["offset_mapping"]
    tgt_positions = [i for i, (a, b) in enumerate(base_offs) if a == base_char_start and b == base_char_end]
    if len(tgt_positions) != 1:
        return {}, None, "target_not_single_piece"
    tpos = tgt_positions[0]
    target_id = int(base_ids[tpos])
    if target_id == mask_id:
        return {}, None, "target_is_mask"
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    if cls_id is None or sep_id is None:
        # fall back to whatever special tokens build_inputs uses
        cls_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
        sep_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id
    masked_base = list(base_ids)
    masked_base[tpos] = mask_id
    out: dict[str, list[int]] = {}
    for key, (prefix, suffix) in context_variants.items():
        pre_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"] if prefix else []
        suf_ids = tokenizer(suffix, add_special_tokens=False)["input_ids"] if suffix else []
        seq = [cls_id]
        if pre_ids:
            seq += pre_ids + [sep_id]
        seq += masked_base
        if suf_ids:
            seq += [sep_id] + suf_ids
        seq += [sep_id]
        if len(seq) > max_length:
            return {}, None, f"too_long_{key}"
        if seq.count(mask_id) != 1:
            return {}, None, f"mask_count_{key}_{seq.count(mask_id)}"
        out[key] = seq
    return out, target_id, None


def validate_single_mask(tokenizer, contexts_unmasked: dict[str, str], contexts_masked: dict[str, str], max_length: int) -> tuple[int | None, dict[str, int], str | None]:
    mask_id = int(tokenizer.mask_token_id)
    special = set(int(x) for x in tokenizer.all_special_ids)
    target_ids = []
    mask_pos: dict[str, int] = {}
    for key in contexts_masked:
        ids_m = tokenizer(contexts_masked[key], add_special_tokens=True, truncation=False)["input_ids"]
        ids_u = tokenizer(contexts_unmasked[key], add_special_tokens=True, truncation=False)["input_ids"]
        if len(ids_m) > max_length or len(ids_u) > max_length:
            return None, {}, "too_long"
        mpos = [i for i, tid in enumerate(ids_m) if int(tid) == mask_id]
        if len(mpos) != 1:
            return None, {}, f"mask_count_{key}_{len(mpos)}"
        if len(ids_m) != len(ids_u):
            return None, {}, f"tokenization_length_change_{key}"
        tid = int(ids_u[mpos[0]])
        if tid in special or tid == mask_id:
            return None, {}, f"special_target_{key}"
        target_ids.append(tid)
        mask_pos[key] = int(mpos[0])
    if len(set(target_ids)) != 1:
        return None, {}, "target_id_mismatch"
    return target_ids[0], mask_pos, None


def build_edit_items(tokenizer, args) -> tuple[list[MaskItem], dict[str, Any]]:
    rng = random.Random(args.seed)
    sep = tokenizer.sep_token or "</s>"
    pairs = load_pairs()
    # Build matched decoy buckets at pair level.
    def pkey(p: PairRec) -> tuple[str, str, str]:
        ob = bin_float(p.content_overlap, [0.4, 0.6, 0.8], ["lt0p4", "0p4_0p6", "0p6_0p8", "ge0p8"])
        lb = "lt16" if p.rewrite_words < 16 else "16_25" if p.rewrite_words <= 25 else "gt25"
        sb = "lt30" if p.source_words < 30 else "30_50" if p.source_words <= 50 else "gt50"
        return ob, lb, sb
    buckets: dict[tuple[str, str, str], list[PairRec]] = collections.defaultdict(list)
    for p in pairs:
        buckets[pkey(p)].append(p)
    all_by_len = pairs[:]

    # Randomize pair order but keep deterministic.
    candidate_pairs = pairs[:]
    if args.low_overlap_focus:
        candidate_pairs.sort(key=lambda p: (p.content_overlap, p.row_index, p.pair_id))
    else:
        rng.shuffle(candidate_pairs)

    items: list[MaskItem] = []
    skip = collections.Counter()
    used_per_pair = collections.Counter()
    mask_id = int(tokenizer.mask_token_id)
    allowed = {"content", "number"} if args.content_targets_only else {"content", "number", "function", "pronoun", "connective"}
    for p in candidate_pairs:
        if len(items) >= args.max_items:
            break
        changed_spans = pair_edit_changed_spans(p)
        if not changed_spans:
            skip["no_changed_span"] += 1
            continue
        changed_char = set()
        for s in changed_spans:
            for c in range(s.start, s.end):
                changed_char.add(c)
        # candidate single-piece target words that lie within changed rewrite spans
        cands = [c for c in single_piece_word_candidates(tokenizer, p.rewrite_text, allowed, drop_edges=False)
                 if c[3] in allowed and (c[0] in changed_char or (c[1] - 1) in changed_char)]
        if not cands:
            skip["no_single_piece_changed_target"] += 1
            continue
        rng.shuffle(cands)
        decoy_pool = [q for q in buckets.get(pkey(p), []) if q.row_index != p.row_index and q.doc_id != p.doc_id]
        if len(decoy_pool) < 3:
            decoy_pool = [q for q in all_by_len if q.row_index != p.row_index and q.doc_id != p.doc_id and abs(q.source_words - p.source_words) <= 20]
        if not decoy_pool:
            skip["no_decoy"] += 1
            continue
        decoy = decoy_pool[(p.row_index + len(items) + args.seed) % len(decoy_pool)]
        n_changed_words = len([t for t in changed_spans if t.is_word])
        n_rew_words = max(1, len([x for x in token_spans(p.rewrite_text) if x.is_word]))
        chg_frac = n_changed_words / n_rew_words
        for (cs, ce, _cid, kind) in cands:
            if used_per_pair[p.pair_id] >= args.max_targets_per_pair or len(items) >= args.max_items:
                break
            context_variants = {
                "base": ("", ""),
                "true": (p.source_text, ""),
                "decoy": (decoy.source_text, ""),
            }
            ctx_ids, tid, reason = build_idlevel_contexts(tokenizer, p.rewrite_text, cs, ce, context_variants, args.max_length, mask_id)
            if tid is None:
                skip[reason or "invalid"] += 1
                continue
            source_token_ids = set(int(x) for x in tokenizer(p.source_text, add_special_tokens=False)["input_ids"])
            decoy_token_ids = set(int(x) for x in tokenizer(decoy.source_text, add_special_tokens=False)["input_ids"])
            target_id_in_source = int(tid) in source_token_ids
            target_id_in_decoy_source = int(tid) in decoy_token_ids
            if args.edit_require_target_id_absent_source and target_id_in_source:
                skip["target_id_present_in_source"] += 1
                continue
            item = MaskItem(
                item_id=f"edit:{p.pair_id}:{used_per_pair[p.pair_id]}:{cs}-{ce}",
                split_key=f"row:{p.row_index}",
                source="compact_edit",
                target_kind=kind,
                target_text=p.rewrite_text[cs:ce],
                target_id=int(tid),
                context_ids=ctx_ids,
                meta={
                    "pair_id": p.pair_id,
                    "row_index": p.row_index,
                    "example_id": p.example_id,
                    "doc_id": p.doc_id,
                    "source_words": p.source_words,
                    "rewrite_words": p.rewrite_words,
                    "decoy_pair_id": decoy.pair_id,
                    "decoy_row_index": decoy.row_index,
                    "content_overlap": p.content_overlap,
                    "overlap_bin": bin_float(p.content_overlap, [0.4, 0.6, 0.8], ["lt0p4", "0p4_0p6", "0p6_0p8", "ge0p8"]),
                    "changed_target_frac_proxy": chg_frac,
                    "changed_frac_bin": bin_float(chg_frac, [0.2, 0.4, 0.6], ["lt0p2", "0p2_0p4", "0p4_0p6", "ge0p6"]),
                    "target_char_start": cs,
                    "target_char_end": ce,
                    "target_id_in_source": target_id_in_source,
                    "target_id_in_decoy_source": target_id_in_decoy_source,
                },
            )
            items.append(item)
            used_per_pair[p.pair_id] += 1
    info = {
        "all_pairs": len(pairs),
        "items_built": len(items),
        "skip_counts": dict(skip),
        "unique_pairs": len({it.meta.get("pair_id") for it in items}),
        "unique_rows": len({it.split_key for it in items}),
        "contexts": ["base", "true", "decoy"],
    }
    return items, info


def split_segments(text: str, source: str) -> list[str]:
    txt = " ".join(str(text).split())
    if not txt:
        return []
    if source == "childes":
        chunks = [c.strip() for c in CHILDES_MARK_RE.split(txt) if c.strip()]
    elif source == "switchboard":
        chunks = [c.strip() for c in SWB_MARK_RE.split(txt) if c.strip()]
    elif source == "open_subtitles":
        rough: list[str] = []
        for part in SUBTITLE_DASH_RE.split(txt):
            rough.extend(SENT_SPLIT_RE.split(part))
        chunks = [c.strip(" -") for c in rough if c.strip(" -")]
    else:
        chunks = [c.strip() for c in SENT_SPLIT_RE.split(txt) if c.strip()]
    return [c for c in chunks if words_count(c) >= 3]


def load_triples(args) -> list[TripleRec]:
    triples: list[TripleRec] = []
    with POOL_10M.open("r", encoding="utf-8") as f:
        for row_idx, line in enumerate(f, 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            src = rec.get("source")
            if src not in NATURAL_SOURCES or src in SYNTHETIC_SOURCES or str(src).startswith("qwen") or str(src).startswith("cleanqwen"):
                continue
            segs = split_segments(rec.get("text", ""), src)
            if len(segs) < 3:
                continue
            for i in range(1, len(segs) - 1):
                prev, mid, nxt = segs[i - 1], segs[i], segs[i + 1]
                mw = words_count(mid)
                if mw < args.min_middle_words or mw > args.max_middle_words:
                    continue
                triples.append(TripleRec(
                    row_index=row_idx,
                    source=str(src),
                    example_id=int(rec.get("example_id", -1)),
                    prev=prev,
                    middle=mid,
                    next=nxt,
                    prev_words=words_count(prev),
                    mid_words=mw,
                    next_words=words_count(nxt),
                    overlap_neighbors_mid=0.5 * (overlap(prev, mid) + overlap(nxt, mid)),
                ))
    return triples


def select_target_span_for_middle(mid: str, rng: random.Random, target_mix: str) -> TokSpan | None:
    spans = [s for s in token_spans(mid) if s.is_word]
    # Avoid sentence-edge words because their tokenization can change most across base vs contextual strings.
    if len(spans) >= 3:
        spans = spans[1:-1]
    if not spans:
        return None
    buckets: dict[str, list[TokSpan]] = collections.defaultdict(list)
    for s in spans:
        buckets[target_kind(s.text)].append(s)
    if target_mix == "content":
        prefs = ["content", "number"]
    elif target_mix == "function":
        prefs = ["pronoun", "connective", "function"]
    else:
        prefs = ["pronoun", "connective", "function", "content", "number"]
    candidates: list[TokSpan] = []
    for k in prefs:
        candidates.extend(buckets.get(k, []))
    if not candidates:
        candidates = spans
    return rng.choice(candidates)


def build_discourse_items(tokenizer, args) -> tuple[list[MaskItem], dict[str, Any]]:
    rng = random.Random(args.seed)
    sep = tokenizer.sep_token or "</s>"
    triples = load_triples(args)
    by_source: dict[str, list[TripleRec]] = collections.defaultdict(list)
    for tr in triples:
        by_source[tr.source].append(tr)
    # Source-balanced selection so CHILDES does not absorb the whole test.
    per_source_target = max(1, math.ceil(args.max_items / max(1, len(by_source))))
    selected_triples: list[TripleRec] = []
    for src, rows in sorted(by_source.items()):
        rows2 = rows[:]
        rng.shuffle(rows2)
        selected_triples.extend(rows2[:per_source_target * 3])  # oversample because some targets fail validation
    rng.shuffle(selected_triples)

    def length_bin(x: int) -> str:
        return "lt8" if x < 8 else "8_15" if x <= 15 else "16_30" if x <= 30 else "gt30"
    index_by_source: dict[str, list[TripleRec]] = by_source

    def choose_shuffle(tr: TripleRec) -> TripleRec | None:
        pool = index_by_source.get(tr.source, [])
        if not pool:
            return None
        # Deterministic small candidate search: same source, different row, similar middle/neighbor lengths and overlap.
        best: tuple[float, TripleRec] | None = None
        start = rng.randrange(len(pool)) if pool else 0
        for off in range(min(80, len(pool))):
            cand = pool[(start + off * 9973) % len(pool)]
            if cand.row_index == tr.row_index:
                continue
            score = (
                abs(cand.prev_words - tr.prev_words) + abs(cand.next_words - tr.next_words)
                + 0.5 * abs(cand.mid_words - tr.mid_words)
                + 12.0 * abs(cand.overlap_neighbors_mid - tr.overlap_neighbors_mid)
            )
            if best is None or score < best[0]:
                best = (score, cand)
        return best[1] if best else None

    items: list[MaskItem] = []
    skip = collections.Counter()
    seen_rows = collections.Counter()
    mask_id = int(tokenizer.mask_token_id)
    if args.target_mix == "content":
        allowed = {"content", "number"}
    elif args.target_mix == "function":
        allowed = {"pronoun", "connective", "function"}
    else:
        allowed = {"content", "number", "pronoun", "connective", "function"}
    for tr in selected_triples:
        if len(items) >= args.max_items:
            break
        if args.max_per_row and seen_rows[tr.row_index] >= args.max_per_row:
            continue
        cands = single_piece_word_candidates(tokenizer, tr.middle, allowed, drop_edges=True)
        if not cands:
            skip["no_single_piece_target"] += 1
            continue
        cs, ce, _cid, kind = rng.choice(cands)
        sh = choose_shuffle(tr)
        if sh is None:
            skip["no_shuffle"] += 1
            continue
        context_variants = {
            "base": ("", ""),
            "true": (tr.prev, tr.next),
            "reversed": (tr.next, tr.prev),
            "shuffled": (sh.prev, sh.next),
        }
        ctx_ids, tid, reason = build_idlevel_contexts(tokenizer, tr.middle, cs, ce, context_variants, args.max_length, mask_id)
        if tid is None:
            skip[reason or "invalid"] += 1
            continue
        item = MaskItem(
            item_id=f"disc:{tr.row_index}:{len(items)}:{cs}-{ce}",
            split_key=f"row:{tr.row_index}",
            source=tr.source,
            target_kind=kind,
            target_text=tr.middle[cs:ce],
            target_id=int(tid),
            context_ids=ctx_ids,
            meta={
                "row_index": tr.row_index,
                "example_id": tr.example_id,
                "source": tr.source,
                "prev_words": tr.prev_words,
                "mid_words": tr.mid_words,
                "next_words": tr.next_words,
                "shuffle_row_index": sh.row_index,
                "shuffle_prev_words": sh.prev_words,
                "shuffle_next_words": sh.next_words,
                "mid_len_bin": length_bin(tr.mid_words),
                "neighbor_mid_overlap": tr.overlap_neighbors_mid,
                "neighbor_overlap_bin": bin_float(tr.overlap_neighbors_mid, [0.05, 0.15, 0.30], ["lt0p05", "0p05_0p15", "0p15_0p30", "ge0p30"]),
                "target_char_start": cs,
                "target_char_end": ce,
            },
        )
        items.append(item)
        seen_rows[tr.row_index] += 1
    info = {
        "all_triples": len(triples),
        "items_built": len(items),
        "skip_counts": dict(skip),
        "source_counts_all_triples": {src: len(v) for src, v in sorted(by_source.items())},
        "source_counts_items": dict(collections.Counter(it.source for it in items)),
        "unique_rows": len({it.split_key for it in items}),
        "contexts": ["base", "true", "reversed", "shuffled"],
    }
    return items, info


def split_items(items: list[MaskItem], train_frac: float, seed: int) -> tuple[list[int], list[int]]:
    keys = sorted(set(it.split_key for it in items))
    rng = random.Random(seed)
    rng.shuffle(keys)
    n_train = max(1, min(len(keys) - 1, int(round(len(keys) * train_frac)))) if len(keys) > 1 else 1
    train_keys = set(keys[:n_train])
    train, test = [], []
    for i, it in enumerate(items):
        (train if it.split_key in train_keys else test).append(i)
    return train, test


def load_model_tokenizer(device: str):
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), use_fast=True)
    if tok.mask_token is None or tok.mask_token_id is None:
        raise RuntimeError("Tokenizer has no mask token")
    model = AutoModelForMaskedLM.from_pretrained(str(CKPT))
    model.eval()
    model.to(device)
    return model, tok


def pad_id_batch(id_lists: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(x) for x in id_lists)
    input_ids = torch.full((len(id_lists), max_len), int(pad_id), dtype=torch.long)
    attention = torch.zeros((len(id_lists), max_len), dtype=torch.long)
    for i, ids in enumerate(id_lists):
        input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention[i, :len(ids)] = 1
    return input_ids, attention


def run_context_forward(model, tokenizer, items: list[MaskItem], context_key: str, batch_size: int, device: str) -> dict[str, Any]:
    seqs = [it.context_ids[context_key] for it in items]
    targets = torch.tensor([it.target_id for it in items], dtype=torch.long)
    mask_id = int(tokenizer.mask_token_id)
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    all_logits: list[torch.Tensor] = []
    all_hidden: list[torch.Tensor] = []
    all_nll: list[torch.Tensor] = []
    all_top1: list[torch.Tensor] = []
    all_rank: list[int] = []
    with torch.no_grad():
        for start in range(0, len(seqs), batch_size):
            end = min(len(seqs), start + batch_size)
            input_ids_cpu, attn_cpu = pad_id_batch(seqs[start:end], pad_id)
            input_ids = input_ids_cpu.to(device)
            attn = attn_cpu.to(device)
            mask_positions = (input_ids == mask_id).nonzero(as_tuple=False)
            if mask_positions.shape[0] != (end - start):
                raise RuntimeError(f"expected one mask per row for {context_key}, got {mask_positions.shape[0]} over {end-start}")
            if not torch.equal(mask_positions[:, 0].cpu(), torch.arange(end - start)):
                raise RuntimeError(f"mask row order mismatch for {context_key}")
            out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True, return_dict=True)
            logits = out.logits[torch.arange(end - start, device=device), mask_positions[:, 1]].float()
            hidden = out.hidden_states[-1][torch.arange(end - start, device=device), mask_positions[:, 1]].float()
            tgt = targets[start:end].to(device)
            nll = F.cross_entropy(logits, tgt, reduction="none")
            top1 = logits.argmax(dim=-1).eq(tgt)
            tgt_logits = logits.gather(1, tgt.view(-1, 1))
            ranks = (logits > tgt_logits).sum(dim=1) + 1
            all_logits.append(logits.cpu())
            all_hidden.append(hidden.cpu())
            all_nll.append(nll.cpu())
            all_top1.append(top1.cpu())
            all_rank.extend(int(x) for x in ranks.cpu().tolist())
    return {
        "logits": torch.cat(all_logits, dim=0),
        "hidden": torch.cat(all_hidden, dim=0),
        "nll": torch.cat(all_nll, dim=0),
        "top1": torch.cat(all_top1, dim=0),
        "rank": all_rank,
    }


class ResidualProbe(torch.nn.Module):
    def __init__(self, in_dim: int, bottleneck: int, vocab: int, dropout: float = 0.0):
        super().__init__()
        self.down = torch.nn.Linear(in_dim, bottleneck)
        self.act = torch.nn.GELU()
        self.drop = torch.nn.Dropout(dropout)
        self.up = torch.nn.Linear(bottleneck, vocab)
        # Zero-init keeps the initial function exactly at the frozen base-offset logits.
        torch.nn.init.zeros_(self.up.weight)
        torch.nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.drop(self.act(self.down(x))))


def eval_probe(model: ResidualProbe, H: torch.Tensor, base_logits: torch.Tensor, targets: torch.Tensor, idx: list[int], device: str, batch_size: int) -> dict[str, Any]:
    if not idx:
        return {"n": 0, "nll_mean": None, "top1": None, "nll": []}
    model.eval()
    nlls: list[torch.Tensor] = []
    tops: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(idx), batch_size):
            ids = idx[start:start + batch_size]
            h = H[ids].to(device)
            off = base_logits[ids].to(device)
            tgt = targets[ids].to(device)
            logits = off + model(h)
            nll = F.cross_entropy(logits, tgt, reduction="none")
            top = logits.argmax(dim=-1).eq(tgt)
            nlls.append(nll.cpu()); tops.append(top.cpu())
    nll_cat = torch.cat(nlls, dim=0)
    top_cat = torch.cat(tops, dim=0)
    return {"n": len(idx), "nll_mean": float(nll_cat.mean().item()), "top1": float(top_cat.float().mean().item()), "nll": [float(x) for x in nll_cat.tolist()]}


def train_probe_for_context(label: str, hidden: torch.Tensor, base_logits: torch.Tensor, targets: torch.Tensor, train_idx: list[int], test_idx: list[int], high_test_idx: list[int], args, device: str) -> dict[str, Any]:
    # Standardize using train only; all tensors are detached from the frozen LM.
    mu = hidden[train_idx].mean(dim=0, keepdim=True)
    sd = hidden[train_idx].std(dim=0, keepdim=True).clamp_min(1e-4)
    H = ((hidden - mu) / sd).float()
    vocab = int(base_logits.shape[1])
    probe = ResidualProbe(H.shape[1], args.probe_dim, vocab, dropout=args.probe_dropout).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=args.probe_lr, weight_decay=args.probe_wd)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(args.seed + 101)
    history = []
    train_tensor = torch.tensor(train_idx, dtype=torch.long)
    for epoch in range(args.probe_epochs):
        perm = train_tensor[torch.randperm(len(train_tensor), generator=gen)].tolist()
        total = 0.0; nb = 0
        probe.train()
        for start in range(0, len(perm), args.probe_batch):
            ids = perm[start:start + args.probe_batch]
            h = H[ids].to(device)
            off = base_logits[ids].to(device)
            tgt = targets[ids].to(device)
            logits = off + probe(h)
            loss = F.cross_entropy(logits, tgt)
            if args.delta_l2 > 0:
                delta = logits - off
                loss = loss + args.delta_l2 * delta.square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), args.grad_clip)
            opt.step()
            total += float(loss.detach().cpu())
            nb += 1
        if epoch in {0, 1, 2, 4, args.probe_epochs - 1} or (epoch + 1) % 10 == 0:
            ev = eval_probe(probe, H, base_logits, targets, test_idx, device, args.probe_eval_batch)
            history.append({"epoch": epoch + 1, "train_loss_mean": total / max(1, nb), "test_nll_mean": ev["nll_mean"], "test_top1": ev["top1"]})
    train_ev = eval_probe(probe, H, base_logits, targets, train_idx, device, args.probe_eval_batch)
    test_ev = eval_probe(probe, H, base_logits, targets, test_idx, device, args.probe_eval_batch)
    high_ev = eval_probe(probe, H, base_logits, targets, high_test_idx, device, args.probe_eval_batch)
    n_params = sum(p.numel() for p in probe.parameters())
    return {
        "label": label,
        "probe": "zero_init_bottleneck_residual_private_readout",
        "probe_dim": args.probe_dim,
        "n_params": int(n_params),
        "base_offset_logits_unchanged": True,
        "lm_parameters_updated": False,
        "hidden_detached": True,
        "train": {k: v for k, v in train_ev.items() if k != "nll"},
        "test": {k: v for k, v in test_ev.items() if k != "nll"},
        "high_baseline_error_test": {k: v for k, v in high_ev.items() if k != "nll"},
        "test_nll_vector": test_ev["nll"],
        "high_test_nll_vector": high_ev["nll"],
        "history": history,
    }


def summarize_raw(raw: dict[str, Any], contexts: list[str], base_key: str = "base") -> dict[str, Any]:
    out = {}
    base = raw[base_key]["nll"]
    for c in contexts:
        nll = raw[c]["nll"]
        vals = [float(x) for x in nll.tolist()]
        delta = [float(b - x) for b, x in zip(base.tolist(), nll.tolist())]  # positive = c improves over base
        out[c] = {
            "n": len(vals),
            "nll": summarize(vals),
            "top1": float(raw[c]["top1"].float().mean().item()),
            "rank": summarize(raw[c]["rank"]),
            "nll_reduction_vs_base_positive_good": summarize(delta),
        }
    # Pairwise true-vs-false: positive means true has lower NLL than false.
    if "true" in raw:
        for false_key in [k for k in contexts if k not in {"base", "true"}]:
            out[f"true_vs_{false_key}"] = bootstrap_mean_diff(raw[false_key]["nll"].tolist(), raw["true"]["nll"].tolist(), seed=1729)
    return out


def group_raw_metrics(items: list[MaskItem], raw: dict[str, Any], contexts: list[str], group_fields: list[str], min_n: int = 20) -> list[dict[str, Any]]:
    rows = []
    for gf in group_fields:
        buckets: dict[str, list[int]] = collections.defaultdict(list)
        for i, it in enumerate(items):
            if gf == "source":
                key = it.source
            elif gf == "target_kind":
                key = it.target_kind
            else:
                key = str(it.meta.get(gf, "NA"))
            buckets[key].append(i)
        for key, idxs in sorted(buckets.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            if len(idxs) < min_n:
                continue
            rec = {"group_field": gf, "group_value": key, "n": len(idxs)}
            base = raw["base"]["nll"]
            if "true" in raw:
                for false_key in [k for k in contexts if k not in {"base", "true"}]:
                    diff = [float(raw[false_key]["nll"][i] - raw["true"]["nll"][i]) for i in idxs]
                    rec[f"true_nll_advantage_vs_{false_key}"] = summarize(diff)
                rec["true_reduction_vs_base"] = summarize([float(base[i] - raw["true"]["nll"][i]) for i in idxs])
            rows.append(rec)
    return rows


def private_readout_analysis(raw: dict[str, Any], contexts: list[str], train_idx: list[int], test_idx: list[int], args, device: str) -> dict[str, Any]:
    targets = torch.tensor([int(x) for x in raw["targets"]], dtype=torch.long)
    base_logits = raw["base"]["logits"].float()
    base_nll = raw["base"]["nll"]
    # High residual-error subset = held-out examples where unchanged base logits assign above-median NLL.
    test_base_vals = [float(base_nll[i]) for i in test_idx]
    thresh = quantile(test_base_vals, args.high_error_quantile) if test_base_vals else None
    high_test_idx = [i for i in test_idx if thresh is not None and float(base_nll[i]) >= float(thresh)]
    probes = {}
    for c in contexts:
        probes[c] = train_probe_for_context(c, raw[c]["hidden"].float(), base_logits, targets, train_idx, test_idx, high_test_idx, args, device)
    base_test_nll = [float(base_nll[i]) for i in test_idx]
    base_high_nll = [float(base_nll[i]) for i in high_test_idx]
    comparison: dict[str, Any] = {
        "base_offset": {
            "test": {"n": len(test_idx), "nll_mean": mean(base_test_nll), "top1": float(raw["base"]["top1"][test_idx].float().mean().item()) if test_idx else None},
            "high_baseline_error_test": {"n": len(high_test_idx), "threshold_quantile": args.high_error_quantile, "nll_threshold": thresh, "nll_mean": mean(base_high_nll)},
        },
        "probe_summaries": {k: {kk: vv for kk, vv in v.items() if kk not in {"test_nll_vector", "high_test_nll_vector"}} for k, v in probes.items()},
    }
    # Positive values below mean the first named condition explains residual errors better.
    for false_key in [k for k in contexts if k not in {"base", "true"}]:
        true_nll = probes["true"]["test_nll_vector"]
        false_nll = probes[false_key]["test_nll_vector"]
        comparison[f"private_true_advantage_vs_{false_key}_test_positive_good"] = bootstrap_mean_diff(false_nll, true_nll, seed=2219)
        true_high = probes["true"]["high_test_nll_vector"]
        false_high = probes[false_key]["high_test_nll_vector"]
        comparison[f"private_true_advantage_vs_{false_key}_high_error_positive_good"] = bootstrap_mean_diff(false_high, true_high, seed=2229)
    # Improvement against unchanged offset.
    for c in contexts:
        comparison[f"private_{c}_nll_reduction_vs_base_test_positive_good"] = summarize([b - p for b, p in zip(base_test_nll, probes[c]["test_nll_vector"])])
        comparison[f"private_{c}_nll_reduction_vs_base_high_error_positive_good"] = summarize([b - p for b, p in zip(base_high_nll, probes[c]["high_test_nll_vector"])])
    return comparison


def write_examples_csv(path: Path, items: list[MaskItem], raw: dict[str, Any], contexts: list[str], probe_comp: dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["i", "item_id", "split_key", "source", "target_kind", "target_text", "target_id"]
    for c in contexts:
        fields += [f"nll_{c}", f"top1_{c}", f"rank_{c}"]
    meta_keys = sorted({k for it in items for k in it.meta.keys()})
    fields += meta_keys
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, it in enumerate(items):
            row = {"i": i, "item_id": it.item_id, "split_key": it.split_key, "source": it.source, "target_kind": it.target_kind, "target_text": it.target_text, "target_id": it.target_id}
            for c in contexts:
                row[f"nll_{c}"] = float(raw[c]["nll"][i])
                row[f"top1_{c}"] = int(bool(raw[c]["top1"][i]))
                row[f"rank_{c}"] = int(raw[c]["rank"][i])
            for k in meta_keys:
                row[k] = it.meta.get(k)
            w.writerow(row)


def write_group_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    # Flatten selected nested summaries.
    flat_rows = []
    for r in rows:
        fr = {"group_field": r["group_field"], "group_value": r["group_value"], "n": r["n"]}
        for k, v in r.items():
            if isinstance(v, dict) and "mean" in v:
                fr[k + "_mean"] = v.get("mean")
                fr[k + "_p05"] = v.get("p05")
                fr[k + "_p95"] = v.get("p95")
        flat_rows.append(fr)
    fields = sorted({k for r in flat_rows for k in r.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(flat_rows)


def run(args) -> dict[str, Any]:
    t0 = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.torch_threads)
    pool_sha = sha256_file(POOL_10M)
    tok_sha = sha256_file(TOKENIZER_DIR / "tokenizer.json")
    if pool_sha != EXPECTED_POOL_SHA:
        raise RuntimeError(f"legal pool SHA mismatch: {pool_sha}")
    if tok_sha != EXPECTED_TOKENIZER_SHA:
        raise RuntimeError(f"tokenizer SHA mismatch: {tok_sha}")

    dev = "cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), use_fast=True)
    if args.mode == "edit":
        items, build_info = build_edit_items(tokenizer, args)
        contexts = ["base", "true", "decoy"]
        group_fields = ["target_kind", "overlap_bin", "changed_frac_bin"]
    elif args.mode == "discourse":
        items, build_info = build_discourse_items(tokenizer, args)
        contexts = ["base", "true", "reversed", "shuffled"]
        group_fields = ["source", "target_kind", "neighbor_overlap_bin", "mid_len_bin"]
    else:
        raise ValueError(args.mode)
    if len(items) < args.min_items:
        raise RuntimeError(f"too few valid items for {args.mode}: {len(items)} < {args.min_items}; build_info={build_info}")
    train_idx, test_idx = split_items(items, args.train_frac, args.seed + 7)
    model = AutoModelForMaskedLM.from_pretrained(str(CKPT))
    model.to(dev); model.eval()
    raw: dict[str, Any] = {"targets": [it.target_id for it in items]}
    for c in contexts:
        print(json.dumps({"event": "forward", "mode": args.mode, "context": c, "n_items": len(items), "device": dev}), flush=True)
        raw[c] = run_context_forward(model, tokenizer, items, c, args.forward_batch, dev)
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()

    raw_summary = summarize_raw(raw, contexts)
    group_rows = group_raw_metrics(items, raw, contexts, group_fields, min_n=args.group_min_n)
    probe_dev = dev if args.probe_device == "same" else args.probe_device
    if probe_dev == "cuda" and not torch.cuda.is_available():
        probe_dev = "cpu"
    private = private_readout_analysis(raw, contexts, train_idx, test_idx, args, probe_dev)

    # Compact per-example output; tensors are not saved.
    examples_csv = out_dir / f"{args.mode}_token_value_examples.csv"
    groups_csv = out_dir / f"{args.mode}_token_value_group_summary.csv"
    write_examples_csv(examples_csv, items, raw, contexts)
    write_group_csv(groups_csv, group_rows)

    # Main interpretive flags: these are not endpoint decisions.
    route_readout = {
        "raw_true_structure_has_token_value": None,
        "private_true_structure_explains_residual_errors": None,
        "scientific_interpretation": "Raw frozen-NLL gains locate structure already usable by research. The detached residual private-readout true-vs-false advantage is the cheap evidence for remaining prediction-connected signal without changing reference logits.",
    }
    false_keys = [k for k in contexts if k not in {"base", "true"}]
    raw_adv = []
    priv_adv = []
    high_adv = []
    for fk in false_keys:
        rv = raw_summary.get(f"true_vs_{fk}", {}).get("mean")
        pv = private.get(f"private_true_advantage_vs_{fk}_test_positive_good", {}).get("mean")
        hv = private.get(f"private_true_advantage_vs_{fk}_high_error_positive_good", {}).get("mean")
        if rv is not None: raw_adv.append(rv)
        if pv is not None: priv_adv.append(pv)
        if hv is not None: high_adv.append(hv)
    route_readout["raw_true_structure_has_token_value"] = bool(raw_adv and min(raw_adv) > args.raw_advantage_threshold)
    route_readout["private_true_structure_explains_residual_errors"] = bool(priv_adv and min(priv_adv) > args.private_advantage_threshold and (not high_adv or min(high_adv) > args.private_high_error_threshold))
    route_readout["raw_true_advantage_min_nll"] = min(raw_adv) if raw_adv else None
    route_readout["private_true_advantage_min_nll"] = min(priv_adv) if priv_adv else None
    route_readout["private_true_high_error_advantage_min_nll"] = min(high_adv) if high_adv else None

    result = {
        "status": "TOKEN_VALUE_PRIVATE_READOUTS",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": args.mode,
        "purpose": "Locate legal true-structure token value and test whether a detached equal-capacity private residual readout explains research residual masked-token errors better than matched false structure while reference logits remain unchanged.",
        "legal_and_runtime": {
            "uses_eval_labels": False,
            "uses_external_text": False,
            "updates_lm_parameters": False,
            "pretraining_or_endpoint_eval": False,
            "pool_10m": str(POOL_10M),
            "pool_sha256": pool_sha,
            "tokenizer": str(TOKENIZER_DIR),
            "tokenizer_sha256": tok_sha,
            "checkpoint": str(CKPT),
            "device_forward": dev,
            "device_probe": probe_dev,
            "hf_home": os.environ.get("HF_HOME"),
            "hf_modules_cache": os.environ.get("HF_MODULES_CACHE"),
        },
        "args": vars(args),
        "build_info": build_info,
        "sample": {
            "n_items": len(items),
            "train_items": len(train_idx),
            "test_items": len(test_idx),
            "unique_split_keys": len({it.split_key for it in items}),
            "target_kind_counts": dict(collections.Counter(it.target_kind for it in items)),
            "target_id_unique": len(set(it.target_id for it in items)),
            "source_counts": dict(collections.Counter(it.source for it in items)),
        },
        "raw_frozen_token_value": raw_summary,
        "group_raw_token_value": group_rows,
        "private_residual_readout": private,
        "route_readout": route_readout,
        "outputs": {
            "examples_csv": str(examples_csv),
            "group_summary_csv": str(groups_csv),
        },
        "elapsed_sec": round(time.time() - t0, 3),
    }
    out_json = out_dir / f"step122_{args.mode}_token_value_private_readouts.json"
    out_md = out_dir / f"step122_{args.mode}_token_value_private_readouts.md"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        f"# research — {args.mode} token-value and private residual readout",
        "",
        "This is a zero-training discriminator using the frozen legal research checkpoint. It does not change LM parameters or run official endpoint evaluation.",
        "",
        "## Sample",
        f"- items: `{len(items)}`; train/test: `{len(train_idx)}`/`{len(test_idx)}`; split keys: `{result['sample']['unique_split_keys']}`.",
        f"- target kinds: `{result['sample']['target_kind_counts']}`.",
        f"- legal pool SHA: `{pool_sha}`; tokenizer SHA: `{tok_sha}`.",
        "",
        "## Frozen research raw token value",
        "Positive true-vs-false numbers mean lower NLL with true structure. This locates structure already usable by the frozen model, not unsaturation by itself.",
        "",
        "| comparison | mean NLL advantage | 5% boot | 95% boot | n |",
        "|---|---:|---:|---:|---:|",
    ]
    for fk in false_keys:
        comp = raw_summary.get(f"true_vs_{fk}", {})
        lines.append(f"| true vs {fk} | {comp.get('mean')} | {comp.get('ci05')} | {comp.get('ci95')} | {comp.get('n')} |")
    lines += [
        "",
        "| context | mean NLL | top1 | mean rank | NLL reduction vs base |",
        "|---|---:|---:|---:|---:|",
    ]
    for c in contexts:
        rs = raw_summary[c]
        lines.append(f"| {c} | {rs['nll']['mean']} | {rs['top1']} | {rs['rank']['mean']} | {rs['nll_reduction_vs_base_positive_good']['mean']} |")
    lines += [
        "",
        "## Detached residual private readout",
        "Each private readout has the same bottleneck size and uses the unchanged base-only frozen logits as an offset. Positive true-vs-false numbers mean the true-structure hidden state explains held-out residual token errors better than matched false structure.",
        "",
        "| comparison | held-out NLL advantage | 5% boot | 95% boot | high-error advantage | high 5% | high 95% |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for fk in false_keys:
        comp = private.get(f"private_true_advantage_vs_{fk}_test_positive_good", {})
        high = private.get(f"private_true_advantage_vs_{fk}_high_error_positive_good", {})
        lines.append(f"| true private vs {fk} private | {comp.get('mean')} | {comp.get('ci05')} | {comp.get('ci95')} | {high.get('mean')} | {high.get('ci05')} | {high.get('ci95')} |")
    lines += [
        "",
        "| private context | held-out NLL reduction vs base offset | high-error reduction vs base offset | held-out top1 |",
        "|---|---:|---:|---:|",
    ]
    for c in contexts:
        red = private.get(f"private_{c}_nll_reduction_vs_base_test_positive_good", {})
        hred = private.get(f"private_{c}_nll_reduction_vs_base_high_error_positive_good", {})
        top1 = private["probe_summaries"][c]["test"]["top1"]
        lines.append(f"| {c} | {red.get('mean')} | {hred.get('mean')} | {top1} |")
    lines += [
        "",
        "## Route readout",
        f"- raw_true_structure_has_token_value: `{route_readout['raw_true_structure_has_token_value']}`; min raw true advantage `{route_readout['raw_true_advantage_min_nll']}`.",
        f"- private_true_structure_explains_residual_errors: `{route_readout['private_true_structure_explains_residual_errors']}`; min held-out private advantage `{route_readout['private_true_advantage_min_nll']}`; min high-error advantage `{route_readout['private_true_high_error_advantage_min_nll']}`.",
        "- Raw token-value gains alone mean research can use supplied structure. The private true-vs-false residual readout is the relevant cheap signal for a future protected/private train-time pathway.",
        "",
        f"Examples CSV: `{examples_csv}`",
        f"Group CSV: `{groups_csv}`",
        f"JSON: `{out_json}`",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "mode": args.mode,
        "out_json": str(out_json),
        "out_md": str(out_md),
        "n_items": len(items),
        "raw_true_advantage_min_nll": route_readout["raw_true_advantage_min_nll"],
        "private_true_advantage_min_nll": route_readout["private_true_advantage_min_nll"],
        "private_true_high_error_advantage_min_nll": route_readout["private_true_high_error_advantage_min_nll"],
        "private_true_structure_explains_residual_errors": route_readout["private_true_structure_explains_residual_errors"],
        "elapsed_sec": result["elapsed_sec"],
    }, indent=2), flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["edit", "discourse"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-items", type=int, default=4096)
    ap.add_argument("--min-items", type=int, default=128)
    ap.add_argument("--seed", type=int, default=8122)
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--max-length", type=int, default=224)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--probe-device", default="same", choices=["same", "cpu", "cuda"])
    ap.add_argument("--torch-threads", type=int, default=8)
    ap.add_argument("--forward-batch", type=int, default=64)
    ap.add_argument("--probe-dim", type=int, default=64)
    ap.add_argument("--probe-epochs", type=int, default=24)
    ap.add_argument("--probe-batch", type=int, default=256)
    ap.add_argument("--probe-eval-batch", type=int, default=512)
    ap.add_argument("--probe-lr", type=float, default=2e-3)
    ap.add_argument("--probe-wd", type=float, default=1e-4)
    ap.add_argument("--probe-dropout", type=float, default=0.05)
    ap.add_argument("--delta-l2", type=float, default=1e-5)
    ap.add_argument("--grad-clip", type=float, default=2.0)
    ap.add_argument("--high-error-quantile", type=float, default=0.50)
    ap.add_argument("--group-min-n", type=int, default=20)
    ap.add_argument("--raw-advantage-threshold", type=float, default=0.02)
    ap.add_argument("--private-advantage-threshold", type=float, default=0.02)
    ap.add_argument("--private-high-error-threshold", type=float, default=0.02)
    # Edit-specific.
    ap.add_argument("--max-targets-per-pair", type=int, default=1)
    ap.add_argument("--low-overlap-focus", action="store_true")
    ap.add_argument("--content-targets-only", action="store_true")
    ap.add_argument("--edit-require-target-id-absent-source", action="store_true")
    # Discourse-specific.
    ap.add_argument("--min-middle-words", type=int, default=4)
    ap.add_argument("--max-middle-words", type=int, default=38)
    ap.add_argument("--max-per-row", type=int, default=3)
    ap.add_argument("--target-mix", default="balanced", choices=["balanced", "content", "function"])
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
