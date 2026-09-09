#!/usr/bin/env python3
"""research: build corpus-only static token mask priors for legal-tokenizer repair.

This script uses only the frozen research legal tokenizer, the allowed 10M
compact-view-reinvest pool, and the already constructed source/rewrite metadata.
It does not read BabyLM evaluation examples or labels.  The output is a small
vocabulary-sized set of token multipliers that can be used by a future trainer
wrapper to redistribute MLM prediction pressure while preserving the nominal
masking budget through batch renormalization.
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
import pathlib
import re
import statistics
import time
from typing import Any

from transformers import AutoTokenizer

def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
WORKSPACE = USER_ROOT / "experiments/archive/frontier_consolidation"
DEFAULT_POOL = WORKSPACE / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"
DEFAULT_ROW_META = WORKSPACE / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_changed_block_rows_meta.jsonl"
DEFAULT_PAIR_ROWS = WORKSPACE / "data/medium_compact_analysis/medium_compact_ws_rows.jsonl"
DEFAULT_TOKENIZER = WORKSPACE / "data/compliant_tokenizer"
DEFAULT_OUT = WORKSPACE / "data/static_token_mask_prior"
EXPECTED_POOL_SHA = "215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23"
EXPECTED_TOKENIZER_SHA = "91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9"
CHANGED_SOURCE = "cleanqwen_fineweb_compact_view_reinvest"

WORD_RE = re.compile(r"\S+")
STRIP_RE = re.compile(r"^[^\w]+|[^\w]+$", re.UNICODE)

SPATIAL = {
    "above", "across", "against", "along", "among", "around", "at", "behind", "below",
    "beneath", "beside", "between", "beyond", "by", "down", "from", "in", "inside",
    "into", "near", "off", "on", "onto", "opposite", "outside", "over", "through",
    "throughout", "to", "toward", "towards", "under", "underneath", "up", "within",
}
CAUSAL = {
    "because", "cause", "caused", "causes", "causing", "due", "effect", "effects",
    "impact", "impacts", "therefore", "thus", "hence", "result", "resulted", "results",
    "resulting", "lead", "leads", "led", "allow", "allows", "allowed", "prevent",
    "prevents", "prevented", "reduce", "reduces", "reduced", "increase", "increases",
    "increased", "risk", "risks", "so", "since", "thereby",
}
DYNAMIC = {
    "act", "acts", "acted", "acting", "arrive", "arrived", "begin", "began", "become",
    "became", "build", "built", "carry", "carried", "change", "changed", "changes",
    "collide", "collided", "come", "came", "create", "created", "develop", "developed",
    "drive", "drove", "enter", "entered", "fall", "fell", "falling", "flow", "flowed",
    "flows", "form", "formed", "forms", "grow", "grew", "hit", "hits", "leave", "left",
    "make", "made", "move", "moved", "moves", "moving", "produce", "produced",
    "reach", "reached", "rise", "rose", "run", "ran", "running", "send", "sent",
    "start", "started", "stop", "stopped", "turn", "turned", "use", "used", "work", "worked",
}
TEMPORAL = {
    "after", "before", "during", "while", "when", "whenever", "until", "since", "then",
    "later", "earlier", "first", "last", "next", "year", "years", "month", "months",
    "day", "days", "century", "centuries", "annual", "currently", "eventually",
}

SCHEME_PARAMS = {
    "relation_only_v1": {
        "spatial": 1.00,
        "causal": 0.70,
        "dynamic": 0.45,
        "temporal": 0.25,
        "numeric": 0.15,
        "entity": 0.05,
        "rewrite_relation": 0.25,
        "changed_domain": 0.10,
        "rare": 0.00,
    },
    "relation_info_v1": {
        "spatial": 0.85,
        "causal": 0.60,
        "dynamic": 0.40,
        "temporal": 0.20,
        "numeric": 0.30,
        "entity": 0.18,
        "rewrite_relation": 0.20,
        "changed_domain": 0.10,
        "rare": 0.25,
    },
}


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_word(w: str) -> str:
    return STRIP_RE.sub("", w).lower().replace("’", "'")


def word_spans(text: str) -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), norm_word(m.group(0))) for m in WORD_RE.finditer(text)]


def token_to_word_indices(text: str, offsets: list[tuple[int, int]]) -> tuple[list[int], list[tuple[int, int, str]]]:
    spans = word_spans(text)
    out: list[int] = []
    j = 0
    for s, e in offsets:
        if e <= s or not spans:
            out.append(-1)
            continue
        while j + 1 < len(spans) and spans[j][1] <= s:
            j += 1
        best = -1
        best_ov = 0
        for k in (j - 1, j, j + 1):
            if 0 <= k < len(spans):
                ws, we, _ = spans[k]
                ov = max(0, min(e, we) - max(s, ws))
                if ov > best_ov:
                    best = k
                    best_ov = ov
        out.append(best)
    return out, spans


def read_jsonl(path: pathlib.Path, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def load_selected_pair_ids(row_meta_path: pathlib.Path) -> tuple[set[str], dict[int, dict[str, Any]]]:
    selected: set[str] = set()
    by_example: dict[int, dict[str, Any]] = {}
    with row_meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            ex_id = int(obj["example_id"])
            pair_ids = [str(x).split(":", 1)[-1] for x in obj.get("pair_ids") or []]
            selected.update(pair_ids)
            by_example[ex_id] = {
                "row_index": int(obj.get("row_index", -1)),
                "pair_ids": pair_ids,
                "component_sources": obj.get("component_sources") or {},
                "words": int(obj.get("words", 0)),
            }
    return selected, by_example


def load_pair_segments(pair_rows_path: pathlib.Path, selected_ids: set[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with pair_rows_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            pid = str(obj.get("prompt_id") or obj.get("pair_id") or "")
            if pid in selected_ids:
                out[pid] = obj
    return out


def infer_roles_for_changed_row(text: str, meta: dict[str, Any] | None, pair_map: dict[str, dict[str, Any]]) -> list[str]:
    spans = word_spans(text)
    roles = ["unknown"] * len(spans)
    if not meta:
        return roles
    constructed_words: list[tuple[str, str]] = []
    for pid in meta.get("pair_ids") or []:
        p = pair_map.get(str(pid))
        if not p:
            continue
        for w in str(p.get("source_text") or "").split():
            constructed_words.append((norm_word(w), "source"))
        for w in str(p.get("rewrite_text") or "").split():
            constructed_words.append((norm_word(w), "rewrite"))
    if not constructed_words:
        return roles
    row_words = [w for _, _, w in spans]
    # Common path: exact packed prefix.  Some rows may contain a tiny top-up suffix;
    # keep those extra words as unknown rather than forcing an alignment.
    n = min(len(row_words), len(constructed_words))
    exact_prefix = all(row_words[i] == constructed_words[i][0] for i in range(n))
    if exact_prefix:
        for i in range(n):
            roles[i] = constructed_words[i][1]
        return roles
    # Fallback: monotone local alignment, used only for rare punctuation/normalization drift.
    j = 0
    for i, w in enumerate(row_words):
        found = -1
        for k in range(j, min(j + 8, len(constructed_words))):
            if constructed_words[k][0] == w:
                found = k
                break
        if found >= 0:
            roles[i] = constructed_words[found][1]
            j = found + 1
    return roles


def feature_flags(word: str, role: str, row_is_changed: bool, row_domains: dict[str, Any]) -> set[str]:
    flags: set[str] = set()
    if not word:
        return flags
    if word in SPATIAL:
        flags.add("spatial")
    if word in CAUSAL:
        flags.add("causal")
    if word in DYNAMIC:
        flags.add("dynamic")
    if word in TEMPORAL:
        flags.add("temporal")
    if any(ch.isdigit() for ch in word):
        flags.add("numeric")
    # Entity-like words are added later from raw span text only when not lowercase;
    # here we keep a placeholder-free lexical set.
    if role == "rewrite":
        flags.add("rewrite")
    elif role == "source":
        flags.add("paired_source")
    if row_is_changed:
        flags.add("changed_block")
        if any(k in row_domains for k in ("science_physical", "causal_relational", "geography_places", "quant_numeric")):
            flags.add("changed_domain")
    if flags.intersection({"spatial", "causal", "dynamic", "temporal", "numeric"}) and role == "rewrite":
        flags.add("rewrite_relation")
    return flags


def add_entity_flags(text: str, spans: list[tuple[int, int, str]], word_flags: list[set[str]]) -> None:
    raw_words = [m.group(0) for m in WORD_RE.finditer(text)]
    for i, raw in enumerate(raw_words[:len(word_flags)]):
        stripped = STRIP_RE.sub("", raw)
        if not stripped:
            continue
        # A deliberately simple corpus-only entity proxy: capitalized lexical tokens
        # away from one-letter sentence starts, acronyms, and tokens with digits.
        if any(ch.isdigit() for ch in stripped):
            continue
        if len(stripped) > 1 and (stripped[0].isupper() or stripped.isupper()):
            word_flags[i].add("entity")


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(DEFAULT_POOL))
    ap.add_argument("--row-meta", default=str(DEFAULT_ROW_META))
    ap.add_argument("--pair-rows", default=str(DEFAULT_PAIR_ROWS))
    ap.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--limit-rows", type=int, default=0)
    ap.add_argument("--max-length", type=int, default=256)
    args = ap.parse_args()

    t0 = time.time()
    pool = pathlib.Path(args.pool)
    row_meta = pathlib.Path(args.row_meta)
    pair_rows = pathlib.Path(args.pair_rows)
    tokenizer_dir = pathlib.Path(args.tokenizer)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pool_sha = sha256_file(pool)
    tok_sha = sha256_file(tokenizer_dir / "tokenizer.json")
    if args.limit_rows == 0 and pool_sha != EXPECTED_POOL_SHA:
        raise RuntimeError(f"pool SHA mismatch: {pool_sha} != {EXPECTED_POOL_SHA}")
    if tok_sha != EXPECTED_TOKENIZER_SHA:
        raise RuntimeError(f"tokenizer SHA mismatch: {tok_sha} != {EXPECTED_TOKENIZER_SHA}")

    selected_ids, meta_by_example = load_selected_pair_ids(row_meta)
    pair_map = load_pair_segments(pair_rows, selected_ids)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), use_fast=True)
    vocab_size = len(tokenizer)
    id_to_token = {i: tokenizer.convert_ids_to_tokens(i) for i in range(vocab_size)}
    special_ids = set(tokenizer.all_special_ids)

    counts = [0] * vocab_size
    feature_counts: dict[str, list[int]] = {name: [0] * vocab_size for name in [
        "spatial", "causal", "dynamic", "temporal", "numeric", "entity", "rewrite",
        "paired_source", "changed_block", "changed_domain", "rewrite_relation",
    ]}
    rows_seen = 0
    words_seen = 0
    token_occurrences = 0
    changed_rows = 0
    exact_role_rows = 0
    tokenized_rows_at_max_length = 0
    row_feature_hits = collections.Counter()

    with pool.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows_seen += 1
            if args.limit_rows and rows_seen > args.limit_rows:
                rows_seen -= 1
                break
            obj = json.loads(line)
            text = str(obj["text"])
            words = int(obj.get("words", len(text.split())))
            words_seen += words
            ex_id = int(obj.get("example_id", rows_seen - 1))
            source = str(obj.get("source", ""))
            row_is_changed = source == CHANGED_SOURCE
            meta = meta_by_example.get(ex_id)
            domains = meta.get("component_sources", {}) if meta else {}
            if row_is_changed:
                changed_rows += 1

            enc = tokenizer(text, add_special_tokens=False, truncation=True, max_length=args.max_length, return_offsets_mapping=True)
            ids = [int(x) for x in enc["input_ids"]]
            offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
            if len(ids) >= args.max_length:
                # This matches the trainer's truncation surface; records potential but
                # does not alter the prior construction.
                tokenized_rows_at_max_length += 1
            tok_word_idx, spans = token_to_word_indices(text, offsets)
            roles = infer_roles_for_changed_row(text, meta, pair_map) if row_is_changed else ["unknown"] * len(spans)
            if row_is_changed and roles and sum(1 for r in roles if r in ("source", "rewrite")) >= min(len(roles), words) * 0.95:
                exact_role_rows += 1
            word_flags: list[set[str]] = []
            for i, (_, _, w) in enumerate(spans):
                role = roles[i] if i < len(roles) else "unknown"
                word_flags.append(feature_flags(w, role, row_is_changed, domains))
            add_entity_flags(text, spans, word_flags)

            for tid, wi in zip(ids, tok_word_idx):
                if tid in special_ids:
                    continue
                counts[tid] += 1
                token_occurrences += 1
                if 0 <= wi < len(word_flags):
                    flags = word_flags[wi]
                    for flag in flags:
                        if flag in feature_counts:
                            feature_counts[flag][tid] += 1
                    for flag in flags:
                        row_feature_hits[flag] += 1

    if token_occurrences <= 0:
        raise RuntimeError("no token occurrences counted")

    nonzero_counts = [c for c in counts if c > 0]
    rarity_scores = [0.0] * vocab_size
    max_rarity = 0.0
    for tid, c in enumerate(counts):
        if c > 0:
            # Smooth inverse-frequency score.  The percentile normalization prevents
            # one-off byte/punctuation artifacts from dominating the map.
            rarity_scores[tid] = math.log((token_occurrences + vocab_size) / (c + 1.0))
            max_rarity = max(max_rarity, rarity_scores[tid])
    p95_rarity = percentile([x for x in rarity_scores if x > 0], 0.95) or max_rarity or 1.0

    schemes: dict[str, dict[str, Any]] = {}
    for name, params in SCHEME_PARAMS.items():
        raw = [1.0] * vocab_size
        for tid, c in enumerate(counts):
            if c <= 0 or tid in special_ids:
                raw[tid] = 1.0
                continue
            val = 1.0
            for feat in ("spatial", "causal", "dynamic", "temporal", "numeric", "entity", "rewrite_relation", "changed_domain"):
                coeff = float(params.get(feat, 0.0))
                if coeff:
                    val += coeff * (feature_counts[feat][tid] / c)
            rare_coeff = float(params.get("rare", 0.0))
            if rare_coeff:
                val += rare_coeff * min(rarity_scores[tid] / p95_rarity, 1.0)
            raw[tid] = val
        occ_mean = sum(raw[tid] * counts[tid] for tid in range(vocab_size)) / token_occurrences
        norm = [1.0] * vocab_size
        for tid, val in enumerate(raw):
            if counts[tid] > 0 and tid not in special_ids:
                norm[tid] = max(0.40, min(2.50, val / occ_mean))
            else:
                norm[tid] = 1.0
        post_mean = sum(norm[tid] * counts[tid] for tid in range(vocab_size)) / token_occurrences
        # Recenter after clamping; keep special/unseen at 1.0.
        final = [1.0] * vocab_size
        for tid, val in enumerate(norm):
            if counts[tid] > 0 and tid not in special_ids:
                final[tid] = max(0.35, min(2.75, val / post_mean))
        final_mean = sum(final[tid] * counts[tid] for tid in range(vocab_size)) / token_occurrences
        ranked = sorted(
            [
                {
                    "token_id": tid,
                    "token": id_to_token.get(tid),
                    "weight": round(final[tid], 6),
                    "count": counts[tid],
                    "feature_rates": {
                        feat: round(feature_counts[feat][tid] / counts[tid], 4)
                        for feat in feature_counts
                        if counts[tid] and feature_counts[feat][tid]
                    },
                }
                for tid in range(vocab_size)
                if counts[tid] > 0 and tid not in special_ids
            ],
            key=lambda x: (x["weight"], x["count"]),
            reverse=True,
        )
        schemes[name] = {
            "description": "occurrence-mean-normalized token multipliers derived only from training-pool lexical/domain structure",
            "params": params,
            "weights": [round(x, 6) for x in final],
            "occurrence_weighted_mean": final_mean,
            "min": min(final[tid] for tid in range(vocab_size) if counts[tid] > 0),
            "max": max(final[tid] for tid in range(vocab_size) if counts[tid] > 0),
            "top_tokens": ranked[:80],
            "bottom_tokens": sorted(ranked, key=lambda x: (x["weight"], -x["count"]))[:40],
        }

    feature_totals = {
        feat: {
            "token_occurrences": int(sum(vals)),
            "fraction_of_counted_tokens": (sum(vals) / token_occurrences),
            "distinct_token_ids": int(sum(1 for x in vals if x > 0)),
        }
        for feat, vals in feature_counts.items()
    }
    payload = {
        "status": "STATIC_TOKEN_MASK_PRIOR_BUILT",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Prepare a compliance-safe learning-signal repair: redistribute MLM mask pressure toward relation and information-bearing corpus tokens under the legal research tokenizer, without reading evaluation examples or changing the training corpus.",
        "inputs": {
            "pool_10m": str(pool),
            "pool_sha256": pool_sha,
            "tokenizer_dir": str(tokenizer_dir),
            "tokenizer_json_sha256": tok_sha,
            "row_meta": str(row_meta),
            "pair_rows": str(pair_rows),
            "limit_rows": args.limit_rows,
            "max_length": args.max_length,
        },
        "counts": {
            "rows_seen": rows_seen,
            "words_seen": words_seen,
            "token_occurrences_counted": token_occurrences,
            "vocab_size": vocab_size,
            "observed_token_ids": len(nonzero_counts),
            "changed_rows_seen": changed_rows,
            "changed_rows_with_high_role_alignment": exact_role_rows,
            "tokenized_rows_at_max_length": tokenized_rows_at_max_length,
        },
        "feature_totals": feature_totals,
        "feature_hit_counter": dict(row_feature_hits),
        "schemes": schemes,
        "interpretation": [
            "The maps are training-data-only token multipliers, not benchmark-derived vocabularies.",
            "Batch-level renormalization in the future trainer should keep the expected number of masked candidate tokens close to the nominal recipe while changing which tokens are preferentially predicted.",
            "This can test a learning-signal allocation repair only after the matched legal-tokenizer clean trajectory clarifies whether compact-view reinvestment survives at mature exposure."
        ],
        "elapsed_sec": round(time.time() - t0, 3),
    }
    out_json = out_dir / "static_token_mask_prior.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# research static token mask prior",
        "",
        payload["purpose"],
        "",
        "## Inputs",
        f"- 10M pool: `{pool}`",
        f"- pool SHA256: `{pool_sha}`",
        f"- tokenizer: `{tokenizer_dir}`",
        f"- tokenizer SHA256: `{tok_sha}`",
        f"- row limit: {args.limit_rows or 'full'}",
        "",
        "## Corpus surface",
        f"- rows seen: {rows_seen}",
        f"- words seen: {words_seen}",
        f"- counted token occurrences: {token_occurrences}",
        f"- observed token ids: {len(nonzero_counts)} / {vocab_size}",
        f"- changed rows seen: {changed_rows}; high source/rewrite role alignment: {exact_role_rows}",
        f"- rows reaching tokenizer max length 256 (possible trainer truncation): {tokenized_rows_at_max_length}",
        "",
        "## Feature occurrence fractions",
    ]
    for feat, rec in sorted(feature_totals.items()):
        lines.append(f"- {feat}: {rec['token_occurrences']} tokens ({rec['fraction_of_counted_tokens']:.4%}), distinct ids {rec['distinct_token_ids']}")
    for name, rec in schemes.items():
        lines += [
            "",
            f"## Scheme `{name}`",
            f"- occurrence-weighted mean: {rec['occurrence_weighted_mean']:.6f}",
            f"- min/max on observed ids: {rec['min']:.4f} / {rec['max']:.4f}",
            "- top weighted observed tokens:",
        ]
        for item in rec["top_tokens"][:25]:
            lines.append(f"  - id {item['token_id']} `{item['token']}` weight={item['weight']} count={item['count']} features={item['feature_rates']}")
    lines += ["", f"Full JSON: `{out_json}`"]
    out_md = out_dir / "static_token_mask_prior.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "out_json": str(out_json),
        "out_md": str(out_md),
        "rows_seen": rows_seen,
        "words_seen": words_seen,
        "token_occurrences_counted": token_occurrences,
        "schemes": list(schemes),
        "elapsed_sec": payload["elapsed_sec"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
