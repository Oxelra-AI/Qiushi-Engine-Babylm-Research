#!/usr/bin/env python3
"""research CPU assets for post-legal40k route decisions.

Purpose
-------
The legal-40k compact_view_reinvest trainings are running asynchronously.  This
script does not poll, train, or evaluate them.  It prepares interpretation and
next-route evidence that uses only the allowed 10M training pool and existing
legal tokenizers:

1. relation-cue density of the 10M pool under the research coarse relation-frame
   lexicon (training-pool text only, not benchmark-driven selection);
2. representation-interface summaries for legal16k, plain legal40k, and
   support-floored legal40k tokenizers, including raw/visible seq256 lengths and
   relation-cue target burden under WWM;
3. normalized relation-boost masking budgets that preserve the same expected
   number of visible selected word groups while reallocating prediction pressure
   toward relation cues;
4. DeBERTa-v2 parameter-count table for plausible next architectures/tokenizers.

The output is a research-facing asset for deciding what to do after the two
legal-40k official vectors arrive.  It is not downstream evidence and must not be
used to stop the active 40k route.
"""
from __future__ import annotations

import bisect
import csv
import hashlib
import importlib.util
import json
import math
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, DebertaV2Config, DebertaV2ForMaskedLM

ROOT = Path(".").resolve()
WS = ROOT / "experiments/archive/representation_and_objectives"
POOL10 = ROOT / "experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"
research = WS / "scripts/relation_frame_retention_audit.py"
OUT_DIR = WS / "data/post40k_route_assets"
NOTE = (ROOT / 'research/notes/representation_and_objectives/post40k_route_assets.md')

EXPECTED_POOL10_SHA = "215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23"
SPECIAL_IDS = {"bos_token": 1, "eos_token": 2, "unk_token": 0, "pad_token": 3, "mask_token": 4}

TOKENIZERS = {
    "legal16k": WS / "data/strictsmall_tokenizer_retrain/strictsmall_compact_reinvest_16k_tokenizer",
    "legal40k": WS / "data/legal_representation_route_map/tokenizers/legal_byte_bpe_40k",
    "legal40k_minfreq25": WS / "data/tokenizer_support_spectrum/tokenizers/legal_byte_bpe_40k_minfreq25",
    "legal40k_minfreq50": WS / "data/tokenizer_support_spectrum/tokenizers/legal_byte_bpe_40k_minfreq50",
}

ARCHITECTURES = [
    # name, tokenizer_key, vocab_size inferred at runtime, hidden, layers, heads, intermediate, max_position_embeddings
    ("8x480_current_legal16k", "legal16k", 480, 8, 8, 1920, 512),
    ("8x480_current_legal40k", "legal40k", 480, 8, 8, 1920, 512),
    ("8x480_minfreq50", "legal40k_minfreq50", 480, 8, 8, 1920, 512),
    ("12x384_leader_shape_legal16k", "legal16k", 384, 12, 12, 1280, 512),
    ("12x384_leader_shape_legal40k", "legal40k", 384, 12, 12, 1280, 512),
    ("12x384_leader_shape_minfreq50", "legal40k_minfreq50", 384, 12, 12, 1280, 512),
    ("12x384_ffn4_legal40k_existing_trainer", "legal40k", 384, 12, 12, 1536, 512),
    ("12x384_leader_pos1024_legal40k", "legal40k", 384, 12, 12, 1280, 1024),
]

WORD_SPAN_RE = re.compile(r"\S+")
WORD_TOKEN_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)?|\d+(?:\.\d+)?")


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_jsonable(obj: Any) -> str:
    text = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_step54_lexicon() -> tuple[dict[str, list[str]], dict[str, list[tuple[str, ...]]]]:
    spec = importlib.util.spec_from_file_location("relation_frame_retention_audit", research)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {research}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.CUE_LEXICON, mod.MULTI_CUES


def norm_token(t: str) -> str:
    t = t.lower().strip()
    if len(t) > 5 and t.endswith("ing"):
        stem = t[:-3]
        if stem:
            return stem
    if len(t) > 4 and t.endswith("ed"):
        stem = t[:-2]
        if stem:
            return stem
    if len(t) > 4 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


def whitespace_word_norms(word: str) -> list[str]:
    return [norm_token(m.group(0)) for m in WORD_TOKEN_RE.finditer(word)]


def build_cue_maps(cue_lexicon: dict[str, list[str]]) -> dict[str, set[str]]:
    single: dict[str, set[str]] = defaultdict(set)
    for cat, words in cue_lexicon.items():
        for w in words:
            single[norm_token(w)].add(cat)
    return dict(single)


def relation_flags_for_text(text: str, cue_single: dict[str, set[str]], multi_cues: dict[str, list[tuple[str, ...]]]) -> tuple[list[tuple[int, int, str]], list[set[str]]]:
    spans = [(m.start(), m.end(), m.group(0)) for m in WORD_SPAN_RE.finditer(text)]
    cats_by_word: list[set[str]] = []
    primary_norm: list[str] = []
    for _s, _e, word in spans:
        cats: set[str] = set()
        norms = whitespace_word_norms(word)
        if norms:
            primary_norm.append(norms[0])
        else:
            primary_norm.append("")
        for n in norms:
            cats.update(cue_single.get(n, set()))
        cats_by_word.append(cats)
    # Mark multi-word relation phrases by assigning the category to all words in the matched phrase.
    for cat, phrases in multi_cues.items():
        for raw_phrase in phrases:
            phrase = tuple(norm_token(x) for x in raw_phrase)
            L = len(phrase)
            if L == 0 or L > len(primary_norm):
                continue
            for i in range(len(primary_norm) - L + 1):
                if tuple(primary_norm[i:i+L]) == phrase:
                    for j in range(i, i + L):
                        cats_by_word[j].add(cat)
    return spans, cats_by_word


def word_index_for_offset(starts: list[int], spans: list[tuple[int, int, str]], start: int, end: int) -> int | None:
    if not spans:
        return None
    # Tokenizer offsets for byte-level BPE can point at whitespace.  Use the token
    # midpoint when the start falls outside a word span.
    idx = bisect.bisect_right(starts, start) - 1
    if idx >= 0 and start < spans[idx][1] and end > spans[idx][0]:
        return idx
    mid = (start + end) // 2
    idx = bisect.bisect_right(starts, mid) - 1
    if idx >= 0 and mid < spans[idx][1]:
        return idx
    # If the token begins on whitespace just before the next word, map to that word.
    nxt = idx + 1
    if 0 <= nxt < len(spans) and end > spans[nxt][0]:
        return nxt
    return None


def load_tokenizers() -> dict[str, Any]:
    toks: dict[str, Any] = {}
    for name, path in TOKENIZERS.items():
        tok = AutoTokenizer.from_pretrained(str(path), use_fast=True)
        backend = getattr(tok, "backend_tokenizer", None) or getattr(tok, "_tokenizer", None)
        if backend is not None:
            try:
                backend.no_padding()
                backend.no_truncation()
            except Exception:
                pass
        toks[name] = tok
    return toks


def init_tok_stats(tok, path: Path) -> dict[str, Any]:
    return {
        "path": rel(path),
        "tokenizer_json_sha256": sha256_file(path / "tokenizer.json") if (path / "tokenizer.json").exists() else None,
        "vocab_size_property": tok.vocab_size,
        "len_tokenizer": len(tok),
        "is_fast": bool(tok.is_fast),
        "special_token_ids": {name: getattr(tok, name + "_id") for name in SPECIAL_IDS},
        "vocab_hash": hash_jsonable(tok.get_vocab()),
        "rows": 0,
        "words": 0,
        "raw_tokens": 0,
        "visible_tokens": 0,
        "raw_over256_rows": 0,
        "visible_word_groups": 0,
        "visible_relation_word_groups": 0,
        "visible_relation_tokens": 0,
        "visible_nonrelation_tokens": 0,
        "unk_tokens_raw": 0,
        "max_raw_tokens": 0,
        "raw_tokens_per_word_rows": [],
        "visible_tokens_per_word_rows": [],
    }


def update_tokenizer_stats(stats: dict[str, Any], tok, text: str, spans: list[tuple[int, int, str]], cats_by_word: list[set[str]]) -> None:
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    n_raw = len(ids)
    n_vis = min(n_raw, 256)
    stats["rows"] += 1
    n_words = len(spans)
    stats["words"] += n_words
    stats["raw_tokens"] += n_raw
    stats["visible_tokens"] += n_vis
    stats["max_raw_tokens"] = max(stats["max_raw_tokens"], n_raw)
    if n_raw > 256:
        stats["raw_over256_rows"] += 1
    unk_id = tok.unk_token_id
    if unk_id is not None:
        stats["unk_tokens_raw"] += sum(1 for tid in ids if int(tid) == int(unk_id))
    starts = [s for s, _e, _w in spans]
    visible_words: set[int] = set()
    relation_words: set[int] = set()
    rel_tokens = 0
    nonrel_tokens = 0
    for start, end in offsets[:256]:
        wi = word_index_for_offset(starts, spans, int(start), int(end))
        if wi is None or not (0 <= wi < len(cats_by_word)):
            nonrel_tokens += 1
            continue
        visible_words.add(wi)
        if cats_by_word[wi]:
            relation_words.add(wi)
            rel_tokens += 1
        else:
            nonrel_tokens += 1
    stats["visible_word_groups"] += len(visible_words)
    stats["visible_relation_word_groups"] += len(relation_words)
    stats["visible_relation_tokens"] += rel_tokens
    stats["visible_nonrelation_tokens"] += nonrel_tokens
    if n_words > 0:
        stats["raw_tokens_per_word_rows"].append(n_raw / n_words)
        stats["visible_tokens_per_word_rows"].append(n_vis / n_words)


def row_source(obj: dict[str, Any]) -> str:
    return str(obj.get("source", "unknown_source"))


def q(xs: list[float], p: float) -> float | None:
    xs = sorted(float(x) for x in xs if math.isfinite(float(x)))
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    pos = p * (len(xs) - 1)
    lo = int(math.floor(pos)); hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def finalize_tok_stats(stats: dict[str, Any]) -> dict[str, Any]:
    rows = stats["rows"]
    words = stats["words"]
    visible_groups = stats["visible_word_groups"]
    rel_groups = stats["visible_relation_word_groups"]
    vis_tokens = stats["visible_tokens"]
    rel_tokens = stats["visible_relation_tokens"]
    out = {k: v for k, v in stats.items() if k not in ("raw_tokens_per_word_rows", "visible_tokens_per_word_rows")}
    out.update({
        "raw_tokens_per_word_weighted": stats["raw_tokens"] / words if words else None,
        "visible_tokens_per_word_weighted": vis_tokens / words if words else None,
        "visible_groups_per_word_weighted": visible_groups / words if words else None,
        "raw_over256_row_frac": stats["raw_over256_rows"] / rows if rows else None,
        "visible_relation_group_frac": rel_groups / visible_groups if visible_groups else None,
        "visible_relation_token_frac": rel_tokens / vis_tokens if vis_tokens else None,
        "visible_tokens_per_visible_group": vis_tokens / visible_groups if visible_groups else None,
        "visible_relation_tokens_per_relation_group": rel_tokens / rel_groups if rel_groups else None,
        "row_raw_tpw_median": q(stats["raw_tokens_per_word_rows"], 0.5),
        "row_raw_tpw_p90": q(stats["raw_tokens_per_word_rows"], 0.9),
        "row_visible_tpw_median": q(stats["visible_tokens_per_word_rows"], 0.5),
        "row_visible_tpw_p90": q(stats["visible_tokens_per_word_rows"], 0.9),
    })
    return out


def relation_boost_table(tok_summaries: dict[str, dict[str, Any]], boosts: list[float] = [1.5, 2.0, 3.0, 4.0]) -> list[dict[str, Any]]:
    rows = []
    base_p = 0.15
    for name, s in tok_summaries.items():
        G = float(s["visible_word_groups"])
        R = float(s["visible_relation_word_groups"])
        T = float(s["visible_tokens"])
        TR = float(s["visible_relation_tokens"])
        TN = T - TR
        N = G - R
        for b in boosts:
            p_rel = min(0.6, base_p * b)
            p_non = (base_p * G - p_rel * R) / N if N > 0 else 0.0
            valid = p_non >= 0 and p_non <= 1
            exp_groups = p_rel * R + max(p_non, 0) * N
            exp_rel_groups = p_rel * R
            exp_tokens = p_rel * TR + max(p_non, 0) * TN
            exp_rel_tokens = p_rel * TR
            rows.append({
                "tokenizer": name,
                "boost": b,
                "base_uniform_group_prob": base_p,
                "relation_group_prob": p_rel,
                "nonrelation_group_prob_normalized": p_non,
                "valid_probability_budget": valid,
                "expected_selected_groups_per_epoch": exp_groups,
                "expected_relation_selected_groups_per_epoch": exp_rel_groups,
                "relation_selected_group_fraction": exp_rel_groups / exp_groups if exp_groups else None,
                "expected_target_tokens_per_epoch": exp_tokens,
                "expected_relation_target_tokens_per_epoch": exp_rel_tokens,
                "relation_target_token_fraction": exp_rel_tokens / exp_tokens if exp_tokens else None,
                "target_token_multiplier_vs_uniform_visible_groups": exp_tokens / (base_p * T) if T else None,
            })
    return rows


def param_count_row(name: str, tok_key: str, vocab_size: int, hidden: int, layers: int, heads: int, intermediate: int, max_pos: int) -> dict[str, Any]:
    cfg = DebertaV2Config(
        vocab_size=vocab_size,
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        intermediate_size=intermediate,
        max_position_embeddings=max(max_pos, 512),
        max_relative_positions=256,
        position_buckets=256,
        relative_attention=True,
        pos_att_type=["p2c", "c2p"],
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        pad_token_id=3,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = DebertaV2ForMaskedLM(cfg)
    total = sum(p.numel() for p in model.parameters())
    emb = model.get_input_embeddings().weight.numel()
    out = {
        "architecture": name,
        "tokenizer": tok_key,
        "vocab_size": vocab_size,
        "hidden_size": hidden,
        "num_hidden_layers": layers,
        "num_attention_heads": heads,
        "intermediate_size": intermediate,
        "max_position_embeddings": cfg.max_position_embeddings,
        "parameter_count": total,
        "embedding_parameter_count": emb,
        "non_embedding_parameter_count": total - emb,
    }
    del model
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            rec = {}
            for k in fields:
                v = r.get(k)
                if isinstance(v, (dict, list, tuple, set)):
                    v = json.dumps(v, ensure_ascii=False, sort_keys=True)
                rec[k] = v
            w.writerow(rec)


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not POOL10.exists():
        raise FileNotFoundError(POOL10)
    pool_sha = sha256_file(POOL10)
    if pool_sha != EXPECTED_POOL10_SHA:
        raise RuntimeError({"pool10_sha_mismatch": pool_sha, "expected": EXPECTED_POOL10_SHA})
    cue_lexicon, multi_cues = load_step54_lexicon()
    cue_single = build_cue_maps(cue_lexicon)
    toks = load_tokenizers()
    tok_stats = {name: init_tok_stats(tok, TOKENIZERS[name]) for name, tok in toks.items()}

    total_rows = 0
    total_words = 0
    relation_words = 0
    category_word_counts: Counter[str] = Counter()
    source_stats: dict[str, Counter[str]] = defaultdict(Counter)
    row_relation_fracs: list[float] = []
    word_cache: dict[str, list[str]] = {}

    # Monkey-patch the word normalizer cache locally for speed.
    def cached_word_norms(word: str) -> list[str]:
        got = word_cache.get(word)
        if got is None:
            got = old_whitespace_word_norms(word)
            if len(word_cache) < 300000:
                word_cache[word] = got
        return got

    global whitespace_word_norms  # use cache in relation_flags_for_text
    old_whitespace_word_norms = whitespace_word_norms
    whitespace_word_norms = cached_word_norms  # type: ignore[assignment]
    try:
        with POOL10.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                text = str(obj["text"])
                src = row_source(obj)
                spans, cats_by_word = relation_flags_for_text(text, cue_single, multi_cues)
                words = int(obj.get("words", len(spans)))
                if words != len(spans):
                    # Keep the script strict: all downstream word accounting uses exact whitespace words.
                    raise RuntimeError({"row_word_count_mismatch": total_rows, "field": words, "spans": len(spans)})
                total_rows += 1
                total_words += words
                row_rel = sum(1 for c in cats_by_word if c)
                relation_words += row_rel
                row_relation_fracs.append(row_rel / words if words else 0.0)
                source_stats[src]["rows"] += 1
                source_stats[src]["words"] += words
                source_stats[src]["relation_words"] += row_rel
                for cats in cats_by_word:
                    for cat in cats:
                        category_word_counts[cat] += 1
                        source_stats[src][f"cat::{cat}"] += 1
                for name, tok in toks.items():
                    update_tokenizer_stats(tok_stats[name], tok, text, spans, cats_by_word)
    finally:
        whitespace_word_norms = old_whitespace_word_norms  # type: ignore[assignment]

    if total_words != 10_000_000:
        raise RuntimeError({"total_words": total_words, "expected": 10_000_000})

    tok_summaries = {name: finalize_tok_stats(s) for name, s in tok_stats.items()}
    boost_rows = relation_boost_table(tok_summaries)
    param_rows = []
    for arch_name, tok_key, h, L, heads, inter, max_pos in ARCHITECTURES:
        param_rows.append(param_count_row(arch_name, tok_key, len(toks[tok_key]), h, L, heads, inter, max_pos))

    source_rows = []
    for src, c in sorted(source_stats.items()):
        rec: dict[str, Any] = {
            "source_class": src,
            "rows": c["rows"],
            "words": c["words"],
            "relation_words": c["relation_words"],
            "relation_word_frac": c["relation_words"] / c["words"] if c["words"] else None,
        }
        for cat in sorted(cue_lexicon):
            rec[f"cat_{cat}_words"] = c[f"cat::{cat}"]
        source_rows.append(rec)

    tok_rows = []
    for name, s in tok_summaries.items():
        tok_rows.append({
            "tokenizer": name,
            "vocab_size": s["len_tokenizer"],
            "raw_tokens_per_word": s["raw_tokens_per_word_weighted"],
            "visible_tokens_per_word": s["visible_tokens_per_word_weighted"],
            "visible_groups_per_word": s["visible_groups_per_word_weighted"],
            "raw_over256_rows": s["raw_over256_rows"],
            "raw_over256_row_frac": s["raw_over256_row_frac"],
            "visible_relation_group_frac": s["visible_relation_group_frac"],
            "visible_relation_token_frac": s["visible_relation_token_frac"],
            "visible_tokens_per_visible_group": s["visible_tokens_per_visible_group"],
            "visible_relation_tokens_per_relation_group": s["visible_relation_tokens_per_relation_group"],
            "unk_tokens_raw": s["unk_tokens_raw"],
            "tokenizer_json_sha256": s["tokenizer_json_sha256"],
        })

    write_csv(OUT_DIR / "tokenizer_interface_summary.csv", tok_rows)
    write_csv(OUT_DIR / "relation_source_class_summary.csv", source_rows)
    write_csv(OUT_DIR / "relation_boost_mask_budget.csv", boost_rows)
    write_csv(OUT_DIR / "architecture_parameter_counts.csv", param_rows)

    payload = {
        "status": "POST40K_ROUTE_ASSETS",
        "created_utc": now_utc(),
        "elapsed_sec": round(time.time() - t0, 1),
        "purpose": "CPU-only preparation for interpreting legal40k and selecting a genuinely distinct post-40k route if needed; no model training or evaluation performed.",
        "inputs": {
            "pool10": rel(POOL10),
            "pool10_sha256": pool_sha,
            "relation_lexicon_script": rel(research),
            "script_sha256": sha256_file(research),
            "tokenizers": {k: rel(v) for k, v in TOKENIZERS.items()},
        },
        "relation_pool_summary": {
            "rows": total_rows,
            "words": total_words,
            "relation_cue_words": relation_words,
            "relation_cue_word_frac": relation_words / total_words,
            "row_relation_frac_mean": statistics.mean(row_relation_fracs),
            "row_relation_frac_median": q(row_relation_fracs, 0.5),
            "row_relation_frac_p90": q(row_relation_fracs, 0.9),
            "category_word_counts": dict(category_word_counts.most_common()),
            "lexicon_method": "coarse research relation-frame cue lexicon; multi-word cues mark all words in matched phrase; used only on training-pool text.",
        },
        "tokenizer_interface_summaries": tok_summaries,
        "relation_boost_mask_budget": boost_rows,
        "architecture_parameter_counts": param_rows,
        "files": {
            "json": rel(OUT_DIR / "post40k_route_assets.json"),
            "tokenizer_interface_summary": rel(OUT_DIR / "tokenizer_interface_summary.csv"),
            "relation_source_class_summary": rel(OUT_DIR / "relation_source_class_summary.csv"),
            "relation_boost_mask_budget": rel(OUT_DIR / "relation_boost_mask_budget.csv"),
            "architecture_parameter_counts": rel(OUT_DIR / "architecture_parameter_counts.csv"),
            "note": rel(NOTE),
        },
        "interpretation_limits": [
            "These are pretraining-pool and tokenizer-surface measurements, not BabyLM downstream scores.",
            "The relation lexicon is broad and coarse; it should guide mechanism tests, not replace official EWoK/Supplement evaluation.",
            "Evaluation text is not used to train tokenizers or select training rows; candidate route choice must wait for the full legal40k official vectors.",
        ],
    }
    (OUT_DIR / "post40k_route_assets.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# research post-legal40k route assets",
        "",
        "This CPU-only asset was built while the two legal-40k trainings were still managed asynchronously. It does not poll them, train a model, or evaluate a model.",
        "",
        "## Relation-cue density in the allowed 10M pool",
        f"- Pool SHA: `{pool_sha}`; rows `{total_rows}`, words `{total_words}`.",
        f"- Relation-cue words under the research coarse lexicon: `{relation_words}` ({relation_words/total_words:.4%} of whitespace words).",
        f"- Mean/median/p90 row relation-cue fraction: `{statistics.mean(row_relation_fracs):.4f}` / `{q(row_relation_fracs,0.5):.4f}` / `{q(row_relation_fracs,0.9):.4f}`.",
        f"- Top relation categories by word count: `{dict(category_word_counts.most_common(10))}`.",
        "",
        "## Tokenizer interface",
        "| tokenizer | vocab | raw tok/word | visible tok/word | over256 rows | visible relation-group frac | visible relation-token frac | unk |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in tok_rows:
        lines.append(
            f"| {r['tokenizer']} | {r['vocab_size']} | {r['raw_tokens_per_word']:.4f} | {r['visible_tokens_per_word']:.4f} | {r['raw_over256_rows']} | {r['visible_relation_group_frac']:.4f} | {r['visible_relation_token_frac']:.4f} | {r['unk_tokens_raw']} |"
        )
    lines.extend([
        "",
        "## Relation-boost masking budget",
        "A relation-focused WWM route can increase relation-cue group probability while lowering non-relation probability so the expected visible selected word-group count stays at 0.15 per group. This keeps total group-level target exposure comparable while testing whether Supplement/EWoK need denser relation/predicate prediction pressure.",
        "",
        "Selected budget rows for legal40k:",
        "| boost | p_relation | p_nonrelation | relation selected group frac | target token multiplier |",
        "|---:|---:|---:|---:|---:|",
    ])
    for r in boost_rows:
        if r["tokenizer"] == "legal40k" and r["boost"] in (2.0, 3.0):
            lines.append(f"| {r['boost']:.1f} | {r['relation_group_prob']:.3f} | {r['nonrelation_group_prob_normalized']:.3f} | {r['relation_selected_group_fraction']:.4f} | {r['target_token_multiplier_vs_uniform_visible_groups']:.4f} |")
    lines.extend([
        "",
        "## Architecture parameter counts",
        "| architecture | tokenizer | params | embedding | non-embedding | hidden/layers/heads/intermediate |",
        "|---|---|---:|---:|---:|---|",
    ])
    for r in param_rows:
        lines.append(f"| {r['architecture']} | {r['tokenizer']} | {r['parameter_count']} | {r['embedding_parameter_count']} | {r['non_embedding_parameter_count']} | {r['hidden_size']}/{r['num_hidden_layers']}/{r['num_attention_heads']}/{r['intermediate_size']} |")
    lines.extend([
        "",
        "## Scientific use",
        "- If legal40k improves Supplement/EWoK while preserving GlobalPIQA/Entity/COMPS, this asset helps characterize how much of the gain comes with low-support representation breadth and reduced target-token burden.",
        "- If legal40k fails or trades away compact-view gains, the relation-boost budget and leader-shape parameter table are ready for a distinct objective or depth route; support-floored tokenizers remain interpretation assets, not automatic next GPU work.",
        "- The full legal40k official vectors remain decisive; no route is selected by this CPU asset alone.",
        "",
        f"JSON: `{rel(OUT_DIR / 'post40k_route_assets.json')}`",
    ])
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": payload["status"],
        "elapsed_sec": payload["elapsed_sec"],
        "relation_word_frac": payload["relation_pool_summary"]["relation_cue_word_frac"],
        "tokenizers": {k: {"vocab": v["len_tokenizer"], "raw_tpw": v["raw_tokens_per_word_weighted"], "over256": v["raw_over256_rows"]} for k, v in tok_summaries.items()},
        "param_counts": {r["architecture"]: r["parameter_count"] for r in param_rows},
        "out_json": rel(OUT_DIR / "post40k_route_assets.json"),
        "note": rel(NOTE),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
