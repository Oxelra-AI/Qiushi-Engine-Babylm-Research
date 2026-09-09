#!/usr/bin/env python3
"""research: training-only representation-allocation frontier for legal tokenizers.

This CPU-only script prepares an interpretation asset for the pending legal-tokenizer
BabyLM Strict-Small route decisions.  It uses only the frozen allowed 10M
compact-view-reinvest training pool and already-trained legal tokenizers; it does not
use official evaluation text and does not train or evaluate any language model.

Scientific question: if legal40k helps or fails, is the effect plausibly explained by
compression/visibility, sparse lexical estimation, source-concentrated tail units, or
whole-word target burden?  The script compares legal 16k/24k/32k/40k and support-floored
candidate tokenizers on the actual pool, preserving source-class attribution.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import hashlib
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer

SESSION_ROOT = _public_path('experiments/archive/frontier_consolidation')
SESSIONS_ROOT = _public_path('experiments/archive')
A01_ROOT = _public_path('experiments/archive/representation_and_objectives')
A02_ROOT = SESSION_ROOT
POOL = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl')
OUT_DIR = _public_path('experiments/archive/frontier_consolidation/data/representation_allocation_frontier')
HIDDEN = 480
MAX_LEN = 256
EXPECTED_POOL_SHA = "215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23"

TOKENIZERS = {
    # Best legal endpoint tokenizer. Included because the pending clean-vs-reinvest
    # vector uses this coordinate.
    "a02_step35_16k": _public_path('experiments/archive/frontier_consolidation/data/compliant_tokenizer/tokenizer.json'),
    # Full-byte legal16k baseline used for the 16k-vs-40k representation comparison.
    "a01_legal16k": _public_path('experiments/archive/representation_and_objectives/data/strictsmall_tokenizer_retrain/strictsmall_compact_reinvest_16k_tokenizer/tokenizer.json'),
    "a01_24k": _public_path('experiments/archive/representation_and_objectives/data/legal_representation_route_map/tokenizers/legal_byte_bpe_24k/tokenizer.json'),
    "a01_32k": _public_path('experiments/archive/representation_and_objectives/data/legal_representation_route_map/tokenizers/legal_byte_bpe_32k/tokenizer.json'),
    "a01_40k": _public_path('experiments/archive/representation_and_objectives/data/legal_representation_route_map/tokenizers/legal_byte_bpe_40k/tokenizer.json'),
    "a01_40k_minfreq25": _public_path('experiments/archive/representation_and_objectives/data/tokenizer_support_spectrum/tokenizers/legal_byte_bpe_40k_minfreq25/tokenizer.json'),
    "a01_40k_minfreq50": _public_path('experiments/archive/representation_and_objectives/data/tokenizer_support_spectrum/tokenizers/legal_byte_bpe_40k_minfreq50/tokenizer.json'),
}
EXPECTED_TOKENIZER_SHA = {
    "a02_step35_16k": "91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9",
    "a01_40k": "94b44bcf5901d097e20294e8220740758eac29af3ec20d58d8da867c85197758",
}


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def source_class(src: str) -> str:
    s = str(src)
    if s == "qwen_pair_packed":
        return "inherited_qwen_pair_packed"
    if s == "cleanqwen_fineweb_compact_view_reinvest":
        return "fineweb_source_compact_view_pair"
    if s.startswith("neutral_cleanqwen_topup"):
        return "neutral_topup"
    if "::" in s:
        head, tail = s.split("::", 1)
        if tail:
            return tail
    return s


def is_word_start(tok: str) -> bool:
    return tok.startswith("Ġ") or tok.startswith("▁")


def encode_geometry(tok: Tokenizer, text: str) -> dict[str, Any]:
    enc = tok.encode(text, add_special_tokens=False)
    tokens = enc.tokens
    ids = enc.ids
    n_raw = len(ids)
    visible = min(n_raw, MAX_LEN)
    group_hist: Counter[int] = Counter()
    group_len = 0
    groups_visible = 0
    for i, ts in enumerate(tokens[:visible]):
        if groups_visible == 0 or i == 0 or is_word_start(ts):
            if group_len:
                group_hist[group_len] += 1
            groups_visible += 1
            group_len = 1
        else:
            group_len += 1
    if group_len:
        group_hist[group_len] += 1
    return {
        "ids": ids,
        "tokens": tokens,
        "n_raw": n_raw,
        "visible_tokens": visible,
        "over256": int(n_raw > MAX_LEN),
        "truncated_tokens": max(0, n_raw - MAX_LEN),
        "visible_groups": groups_visible,
        "group_hist": group_hist,
    }


def q(vals: list[float]) -> dict[str, float | None]:
    if not vals:
        return {k: None for k in ["mean", "median", "p10", "p90", "p99"]}
    a = np.asarray(vals, dtype=float)
    return {
        "mean": float(a.mean()),
        "median": float(np.quantile(a, 0.5)),
        "p10": float(np.quantile(a, 0.1)),
        "p90": float(np.quantile(a, 0.9)),
        "p99": float(np.quantile(a, 0.99)),
    }


def safe_entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            h -= p * math.log(p)
    return h


def norm_entropy(counts: Counter[str], n_sources: int) -> float:
    if n_sources <= 1:
        return 0.0
    return safe_entropy(counts) / math.log(n_sources)


def pct(n: float, d: float) -> float:
    return float(n / d) if d else 0.0


def count_support_bins(token_counts: Counter[str], source_counts: dict[str, Counter[str]], row_counts: Counter[str], n_sources: int) -> dict[str, Any]:
    bins = [
        ("1", 1, 1),
        ("2_9", 2, 9),
        ("10_24", 10, 24),
        ("25_49", 25, 49),
        ("50_99", 50, 99),
        ("100_199", 100, 199),
        ("200_499", 200, 499),
        ("500_999", 500, 999),
        ("1000_plus", 1000, 10**18),
    ]
    total_mass = sum(token_counts.values())
    out: dict[str, Any] = {}
    for name, lo, hi in bins:
        toks = [t for t, c in token_counts.items() if lo <= c <= hi]
        mass = sum(token_counts[t] for t in toks)
        ent_weight_num = 0.0
        maxsrc_weight_num = 0.0
        rows = []
        for t in toks:
            c = token_counts[t]
            rows.append(row_counts[t])
            sc = source_counts[t]
            ent_weight_num += c * norm_entropy(sc, n_sources)
            maxsrc_weight_num += c * (max(sc.values()) / sum(sc.values()) if sc else 0.0)
        out[name] = {
            "vocab_count": len(toks),
            "occurrence_mass": mass,
            "mass_frac": pct(mass, total_mass),
            "row_count_median": float(statistics.median(rows)) if rows else None,
            "source_entropy_occ_weighted": pct(ent_weight_num, mass),
            "max_source_frac_occ_weighted": pct(maxsrc_weight_num, mass),
        }
    return out


def summarize_tokenizer(label: str, path: Path, tok: Tokenizer, stats: dict[str, Any], declared_words: int, n_rows: int, source_labels: list[str]) -> dict[str, Any]:
    token_counts: Counter[str] = stats["token_counts"]
    row_counts: Counter[str] = stats["row_counts"]
    source_counts: dict[str, Counter[str]] = stats["source_counts"]
    total_token_mass = sum(token_counts.values())
    counts = list(token_counts.values())
    non_special_vocab = [s for s in tok.get_vocab().keys() if not (s.startswith("<") and s.endswith(">"))]
    observed = len(token_counts)
    vocab_size = len(tok.get_vocab())
    low50_vocab = sum(1 for t in non_special_vocab if token_counts.get(t, 0) < 50)
    low100_vocab = sum(1 for t in non_special_vocab if token_counts.get(t, 0) < 100)
    low50_obs = sum(1 for t, c in token_counts.items() if c < 50)
    low100_obs = sum(1 for t, c in token_counts.items() if c < 100)
    low50_mass = sum(c for c in counts if c < 50)
    low100_mass = sum(c for c in counts if c < 100)
    ent_weight_num = 0.0
    maxsrc_weight_num = 0.0
    source_concentrated_low_tail_mass = 0
    for t, c in token_counts.items():
        sc = source_counts[t]
        total = sum(sc.values())
        maxfrac = max(sc.values()) / total if total else 0.0
        ent = norm_entropy(sc, len(source_labels))
        ent_weight_num += c * ent
        maxsrc_weight_num += c * maxfrac
        if c < 100 and maxfrac >= 0.90:
            source_concentrated_low_tail_mass += c
    return {
        "label": label,
        "path": str(path.relative_to(Path.cwd())) if path.is_absolute() and str(path).startswith(str(Path.cwd())) else str(path),
        "sha256": sha256_file(path),
        "expected_sha_match": (sha256_file(path) == EXPECTED_TOKENIZER_SHA[label]) if label in EXPECTED_TOKENIZER_SHA else None,
        "vocab_size": vocab_size,
        "declared_words": declared_words,
        "rows": n_rows,
        "raw_tokens": stats["raw_tokens"],
        "visible_tokens": stats["visible_tokens"],
        "visible_groups": stats["visible_groups"],
        "over256_rows": stats["over256_rows"],
        "truncated_tokens": stats["truncated_tokens"],
        "raw_tokens_per_word": pct(stats["raw_tokens"], declared_words),
        "visible_tokens_per_word": pct(stats["visible_tokens"], declared_words),
        "visible_groups_per_word": pct(stats["visible_groups"], declared_words),
        "tokens_per_visible_group": pct(stats["visible_tokens"], stats["visible_groups"]),
        "observed_token_strings": observed,
        "observed_count_quantiles": q([float(c) for c in counts]),
        "vocab_lt50_frac_including_unobserved_non_special": pct(low50_vocab, len(non_special_vocab)),
        "vocab_lt100_frac_including_unobserved_non_special": pct(low100_vocab, len(non_special_vocab)),
        "observed_lt50_frac": pct(low50_obs, observed),
        "observed_lt100_frac": pct(low100_obs, observed),
        "mass_lt50_frac": pct(low50_mass, total_token_mass),
        "mass_lt100_frac": pct(low100_mass, total_token_mass),
        "source_entropy_occ_weighted": pct(ent_weight_num, total_token_mass),
        "max_source_frac_occ_weighted": pct(maxsrc_weight_num, total_token_mass),
        "source_concentrated_low_tail_mass_frac": pct(source_concentrated_low_tail_mass, total_token_mass),
        "group_hist_visible": dict(sorted(stats["group_hist_visible"].items(), key=lambda kv: int(kv[0]))),
        "source_raw_tokens": dict(stats["source_raw_tokens"]),
        "source_declared_words": dict(stats["source_declared_words"]),
        "support_bins": count_support_bins(token_counts, source_counts, row_counts, len(source_labels)),
    }


def incremental_summary(label: str, baseline_label: str, tok_vocab: set[str], base_vocab: set[str], stats: dict[str, Any], base_stats: dict[str, Any], declared_words: int, source_labels: list[str]) -> dict[str, Any]:
    token_counts: Counter[str] = stats["token_counts"]
    row_counts: Counter[str] = stats["row_counts"]
    source_counts: dict[str, Counter[str]] = stats["source_counts"]
    new_vocab = tok_vocab - base_vocab
    new_observed = [t for t in new_vocab if token_counts.get(t, 0) > 0]
    new_mass = sum(token_counts[t] for t in new_observed)
    low50_new = [t for t in new_observed if token_counts[t] < 50]
    low100_new = [t for t in new_observed if token_counts[t] < 100]
    ent_weight_num = 0.0
    maxsrc_weight_num = 0.0
    source_conc_mass = 0
    counts = []
    rowc = []
    for t in new_observed:
        c = token_counts[t]
        counts.append(float(c))
        rowc.append(float(row_counts[t]))
        sc = source_counts[t]
        total = sum(sc.values())
        maxfrac = max(sc.values()) / total if total else 0.0
        ent = norm_entropy(sc, len(source_labels))
        ent_weight_num += c * ent
        maxsrc_weight_num += c * maxfrac
        if c < 100 and maxfrac >= 0.90:
            source_conc_mass += c
    saving = base_stats["raw_tokens"] - stats["raw_tokens"]
    visible_saving = base_stats["visible_tokens"] - stats["visible_tokens"]
    group_gain = stats["visible_groups"] - base_stats["visible_groups"]
    by_source = {}
    all_sources = set(base_stats["source_raw_tokens"]) | set(stats["source_raw_tokens"])
    for s in sorted(all_sources):
        base_tok = base_stats["source_raw_tokens"].get(s, 0)
        tok_tok = stats["source_raw_tokens"].get(s, 0)
        words = stats["source_declared_words"].get(s, base_stats["source_declared_words"].get(s, 0))
        by_source[s] = {
            "raw_token_saving_vs_baseline": base_tok - tok_tok,
            "saving_per_declared_word": pct(base_tok - tok_tok, words),
            "baseline_raw_tokens": base_tok,
            "tokenizer_raw_tokens": tok_tok,
            "declared_words": words,
        }
    # top new tokens by mass and by source concentration, with token strings only from training tokenizer vocabulary
    top_new_by_count = sorted(new_observed, key=lambda t: token_counts[t], reverse=True)[:40]
    top_source_conc = sorted(
        new_observed,
        key=lambda t: ((max(source_counts[t].values()) / sum(source_counts[t].values())) if source_counts[t] else 0.0, token_counts[t]),
        reverse=True,
    )[:40]
    def token_row(t: str) -> dict[str, Any]:
        sc = source_counts[t]
        total = sum(sc.values())
        top_src, top_c = (None, 0)
        if sc:
            top_src, top_c = max(sc.items(), key=lambda kv: kv[1])
        return {
            "token": t,
            "count": token_counts[t],
            "row_count": row_counts[t],
            "top_source": top_src,
            "top_source_frac": pct(top_c, total),
            "norm_source_entropy": norm_entropy(sc, len(source_labels)),
        }
    return {
        "label": label,
        "baseline_label": baseline_label,
        "new_vocab_count": len(new_vocab),
        "new_observed_count": len(new_observed),
        "new_observed_mass": new_mass,
        "new_observed_mass_frac": pct(new_mass, stats["raw_tokens"]),
        "new_count_quantiles": q(counts),
        "new_row_count_quantiles": q(rowc),
        "new_vocab_lt50_frac_observed": pct(len(low50_new), len(new_observed)),
        "new_vocab_lt100_frac_observed": pct(len(low100_new), len(new_observed)),
        "new_mass_lt50_frac_total": pct(sum(token_counts[t] for t in low50_new), stats["raw_tokens"]),
        "new_mass_lt100_frac_total": pct(sum(token_counts[t] for t in low100_new), stats["raw_tokens"]),
        "new_source_entropy_occ_weighted": pct(ent_weight_num, new_mass),
        "new_max_source_frac_occ_weighted": pct(maxsrc_weight_num, new_mass),
        "new_source_concentrated_low_tail_mass_frac_total": pct(source_conc_mass, stats["raw_tokens"]),
        "raw_token_saving_vs_baseline": saving,
        "visible_token_saving_vs_baseline": visible_saving,
        "visible_group_gain_vs_baseline": group_gain,
        "raw_token_saving_per_added_vocab": pct(saving, len(new_vocab)),
        "visible_token_saving_per_added_vocab": pct(visible_saving, len(new_vocab)),
        "extra_embedding_rows_vs_baseline": len(tok_vocab) - len(base_vocab),
        "extra_embedding_parameters_at_hidden480": (len(tok_vocab) - len(base_vocab)) * HIDDEN,
        "raw_token_saving_per_extra_embedding_parameter": pct(saving, max(1, (len(tok_vocab) - len(base_vocab)) * HIDDEN)),
        "by_source": by_source,
        "top_new_tokens_by_count": [token_row(t) for t in top_new_by_count],
        "top_new_tokens_by_source_concentration": [token_row(t) for t in top_source_conc],
    }


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pool_sha = sha256_file(POOL)
    if pool_sha != EXPECTED_POOL_SHA:
        raise SystemExit(f"pool sha mismatch: {pool_sha} != {EXPECTED_POOL_SHA}")

    tokenizers: dict[str, Tokenizer] = {}
    vocab_sets: dict[str, set[str]] = {}
    tokenizer_shas: dict[str, str] = {}
    for label, path in TOKENIZERS.items():
        if not path.exists():
            raise SystemExit(f"missing tokenizer {label}: {path}")
        tokenizer_shas[label] = sha256_file(path)
        tokenizers[label] = Tokenizer.from_file(str(path))
        # Some saved tokenizer.json files carry runtime truncation metadata from HF
        # save_pretrained/evaluation use.  Disable it explicitly: this script must
        # measure raw corpus segmentation before applying the trainer's seq256 crop.
        tokenizers[label].no_truncation()
        tokenizers[label].no_padding()
        vocab_sets[label] = set(tokenizers[label].get_vocab().keys())

    stats: dict[str, dict[str, Any]] = {}
    for label in tokenizers:
        stats[label] = {
            "raw_tokens": 0,
            "visible_tokens": 0,
            "visible_groups": 0,
            "over256_rows": 0,
            "truncated_tokens": 0,
            "group_hist_visible": Counter(),
            "token_counts": Counter(),
            "row_counts": Counter(),
            "source_counts": defaultdict(Counter),
            "source_raw_tokens": Counter(),
            "source_declared_words": Counter(),
        }

    declared_words = 0
    n_rows = 0
    source_labels_set: set[str] = set()
    with POOL.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = obj.get("text", "")
            words = int(obj.get("words", len(text.split())))
            src = source_class(obj.get("source", ""))
            source_labels_set.add(src)
            declared_words += words
            n_rows += 1
            for label, tok in tokenizers.items():
                g = encode_geometry(tok, text)
                st = stats[label]
                st["raw_tokens"] += g["n_raw"]
                st["visible_tokens"] += g["visible_tokens"]
                st["visible_groups"] += g["visible_groups"]
                st["over256_rows"] += g["over256"]
                st["truncated_tokens"] += g["truncated_tokens"]
                st["group_hist_visible"].update(g["group_hist"])
                st["token_counts"].update(g["tokens"])
                st["source_raw_tokens"][src] += g["n_raw"]
                st["source_declared_words"][src] += words
                unique = set(g["tokens"])
                for ts in unique:
                    st["row_counts"][ts] += 1
                # occurrence-weighted source distribution
                local_counts = Counter(g["tokens"])
                for ts, c in local_counts.items():
                    st["source_counts"][ts][src] += c

    source_labels = sorted(source_labels_set)
    if declared_words != 10_000_000 or n_rows != 64_740:
        raise SystemExit(f"unexpected pool rows/words: rows={n_rows} words={declared_words}")

    summaries = {
        label: summarize_tokenizer(label, path, tokenizers[label], stats[label], declared_words, n_rows, source_labels)
        for label, path in TOKENIZERS.items()
    }

    incrementals: dict[str, dict[str, dict[str, Any]]] = {"vs_a01_legal16k": {}, "vs_a02_step35_16k": {}}
    for base_label in ["a01_legal16k", "a02_step35_16k"]:
        for label in tokenizers:
            if label == base_label:
                continue
            incrementals[f"vs_{base_label}"][label] = incremental_summary(
                label,
                base_label,
                vocab_sets[label],
                vocab_sets[base_label],
                stats[label],
                stats[base_label],
                declared_words,
                source_labels,
            )

    frontier_rows = []
    for label in tokenizers:
        s = summaries[label]
        inc = incrementals["vs_a01_legal16k"].get(label)
        frontier_rows.append({
            "label": label,
            "vocab_size": s["vocab_size"],
            "raw_tokens_per_word": s["raw_tokens_per_word"],
            "visible_tokens_per_word": s["visible_tokens_per_word"],
            "visible_groups_per_word": s["visible_groups_per_word"],
            "over256_rows": s["over256_rows"],
            "truncated_tokens": s["truncated_tokens"],
            "observed_lt50_frac": s["observed_lt50_frac"],
            "mass_lt50_frac": s["mass_lt50_frac"],
            "source_concentrated_low_tail_mass_frac": s["source_concentrated_low_tail_mass_frac"],
            "raw_token_saving_vs_a01_legal16k": inc["raw_token_saving_vs_baseline"] if inc else 0,
            "new_observed_mass_frac_vs_a01_legal16k": inc["new_observed_mass_frac"] if inc else 0,
            "new_vocab_lt50_frac_observed_vs_a01_legal16k": inc["new_vocab_lt50_frac_observed"] if inc else 0,
            "extra_embedding_parameters_vs_a01_legal16k": inc["extra_embedding_parameters_at_hidden480"] if inc else 0,
        })

    result = {
        "status": "REPRESENTATION_ALLOCATION_FRONTIER",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": round(time.time() - t0, 3),
        "purpose": "training-only legal-tokenizer representation allocation map for interpreting pending 40k and clean-vs-reinvest results; no eval text, no model training",
        "pool": {
            "path": str(POOL.relative_to(Path.cwd())) if POOL.is_absolute() and str(POOL).startswith(str(Path.cwd())) else str(POOL),
            "sha256": pool_sha,
            "expected_sha_match": pool_sha == EXPECTED_POOL_SHA,
            "rows": n_rows,
            "declared_words": declared_words,
            "source_labels": source_labels,
        },
        "tokenizer_shas": tokenizer_shas,
        "summaries": summaries,
        "incremental": incrementals,
        "frontier_rows": frontier_rows,
        "interpretation": {
            "scope": "tokenizer representation/support geometry on allowed training pool only",
            "does_not_establish": [
                "official-score effect of any tokenizer",
                "pure vocabulary-size causality",
                "benchmark-specific weakness or repair target",
            ],
            "main_questions_for_pending_results": [
                "If legal40k improves, do gains come despite a large low-support new-token tail, or only where compression/visibility is high?",
                "If legal40k fails, is the failure plausibly linked to source-concentrated rare units or reduced subword-sharing regularization?",
                "Do support-floored inventories retain much of the compression with far fewer low-support rows, making a later compositional/support-aware representation scientifically distinct from another size sweep?",
            ],
        },
    }

    out_json = _public_path('experiments/archive/frontier_consolidation/data/representation_allocation_frontier/representation_allocation_frontier.json')
    out_md = _public_path('research/documents/frontier_consolidation/data/representation_allocation_frontier/representation_allocation_frontier.md')
    out_csv = _public_path('experiments/archive/frontier_consolidation/data/representation_allocation_frontier/frontier_summary.csv')
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        cols = list(frontier_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in frontier_rows:
            w.writerow(r)

    # concise markdown with numbers a future agent can read quickly
    def fmt(x: Any, nd: int = 6) -> str:
        if isinstance(x, float):
            return f"{x:.{nd}f}"
        return str(x)

    lines = []
    lines.append("# research representation-allocation frontier (training-only)\n")
    lines.append("Uses only the frozen legal compact-view-reinvest 10M training pool and already trained legal tokenizers. No official evaluation text, no model training, no route selection by itself.\n")
    lines.append("## Integrity\n")
    lines.append(f"- Pool rows/words: `{n_rows}` / `{declared_words}`")
    lines.append(f"- Pool SHA matched expected: `{pool_sha == EXPECTED_POOL_SHA}`")
    for label in tokenizers:
        exp = EXPECTED_TOKENIZER_SHA.get(label)
        if exp:
            lines.append(f"- {label} tokenizer SHA matched expected: `{tokenizer_shas[label] == exp}`")
        else:
            lines.append(f"- {label} tokenizer SHA: `{tokenizer_shas[label]}`")
    lines.append("\n## Frontier vs A01 legal16k baseline\n")
    lines.append("| tokenizer | vocab | tok/word | visible tok/word | visible groups/word | over256 | trunc toks | raw saving | new mass frac | observed new <50 | mass<50 | source-conc low-tail mass | extra emb params |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in frontier_rows:
        lines.append(
            f"| {r['label']} | {r['vocab_size']} | {fmt(r['raw_tokens_per_word'])} | {fmt(r['visible_tokens_per_word'])} | {fmt(r['visible_groups_per_word'])} | {r['over256_rows']} | {r['truncated_tokens']} | {r['raw_token_saving_vs_a01_legal16k']} | {fmt(r['new_observed_mass_frac_vs_a01_legal16k'])} | {fmt(r['new_vocab_lt50_frac_observed_vs_a01_legal16k'])} | {fmt(r['mass_lt50_frac'])} | {fmt(r['source_concentrated_low_tail_mass_frac'])} | {r['extra_embedding_parameters_vs_a01_legal16k']} |"
        )
    lines.append("\n## Source distribution of 40k raw-token savings vs A01 legal16k\n")
    inc40 = incrementals["vs_a01_legal16k"].get("a01_40k")
    if inc40:
        lines.append("| source | raw token saving | saving/word | declared words |")
        lines.append("|---|---:|---:|---:|")
        for src, vals in sorted(inc40["by_source"].items(), key=lambda kv: kv[1]["raw_token_saving_vs_baseline"], reverse=True):
            lines.append(f"| {src} | {vals['raw_token_saving_vs_baseline']} | {fmt(vals['saving_per_declared_word'])} | {vals['declared_words']} |")
        lines.append("\nTop 40k new-token counts vs A01 legal16k (first 20):")
        for tr in inc40["top_new_tokens_by_count"][:20]:
            lines.append(f"- `{tr['token']}` count={tr['count']} rows={tr['row_count']} top_source={tr['top_source']} top_frac={fmt(tr['top_source_frac'],3)} Hsrc={fmt(tr['norm_source_entropy'],3)}")
    lines.append("\n## Interpretation\n")
    lines.append("This asset separates compression/visibility from sparse lexical allocation. The pending legal40k official result remains a package-level result: tokenizer inventory, target-token burden, embedding/output rows, and accumulated-forward optimization all move together. This file should be joined with 40k score deltas after the A01 collations finish and with A02's clean-vs-reinvest vector after s50_t19 delivers.")
    lines.append("\nJSON: `experiments/archive/frontier_consolidation/data/representation_allocation_frontier/representation_allocation_frontier.json`")
    lines.append("\nCSV: `experiments/archive/frontier_consolidation/data/representation_allocation_frontier/frontier_summary.csv`\n")
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": result["status"],
        "out_json": str(out_json),
        "out_md": str(out_md),
        "out_csv": str(out_csv),
        "elapsed_sec": result["elapsed_sec"],
        "frontier_rows": frontier_rows,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
