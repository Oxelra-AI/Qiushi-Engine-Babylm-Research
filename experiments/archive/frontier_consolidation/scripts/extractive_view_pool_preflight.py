#!/usr/bin/env python3
"""research: Source-wide extractive compact-view pool preflight.

Constructs two extractive view variants using ONLY source words:
  1. extractive_wide:    maximize content, source-wide coverage
  2. extractive_balanced: match compact content density (~65%), source-wide

Builds 10M pool JSONL for each, computes atlas stats matching research,
computes tokenizer-level properties, and writes quantitative preflight.

CPU/file-only. No model training, evaluation, upload, or leaderboard action.
"""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json, hashlib, re, math, statistics, time, os, sys
from pathlib import Path
from collections import Counter, OrderedDict

# ── Content/function classification (exact research atlas definitions) ──

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "so", "because", "as", "than",
    "to", "of", "in", "on", "for", "with", "without", "by", "from", "at", "into", "onto", "over",
    "under", "about", "between", "among", "through", "during", "before", "after", "above", "below",
    "is", "am", "are", "was", "were", "be", "been", "being", "do", "does", "did", "done", "doing",
    "have", "has", "had", "having", "can", "could", "may", "might", "must", "shall", "should",
    "will", "would", "this", "that", "these", "those", "there", "here", "it", "its", "they", "them",
    "their", "theirs", "he", "him", "his", "she", "her", "hers", "we", "us", "our", "ours", "you",
    "your", "yours", "i", "me", "my", "mine", "who", "whom", "whose", "which", "what", "where",
    "when", "why", "how", "not", "no", "nor", "only", "just", "also", "very", "more", "most", "less",
    "least", "much", "many", "some", "any", "all", "each", "every", "other", "another", "such", "own",
    "same", "too", "again", "still", "already", "yet", "up", "down", "out", "off", "back", "away",
    "into", "within", "across", "per", "via", "using", "used", "use", "uses", "become", "became",
}

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['\u2019-][A-Za-z0-9]+)?")


def split_words(text):
    return [w for w in text.split() if w]


def norm_word(word):
    parts = WORD_RE.findall(word)
    return "".join(parts).lower() if parts else ""


def is_content(norm):
    if not norm:
        return False
    if norm in STOPWORDS:
        return False
    if norm.isdigit():
        return True
    return len(norm) >= 4


# ── Extractive construction algorithms ──

def build_extractive_wide(source_words, target_count):
    """Content-maximizing, source-wide extractive view.
    Keeps content words from across the full source span,
    drops function words. Very high density, broad coverage.
    """
    N = len(source_words)
    V = target_count
    if V >= N:
        return list(source_words[:V])
    norms = [norm_word(w) for w in source_words]
    # Priority: content >> function; slight tail preference for ties
    priorities = []
    for i in range(N):
        p = 2.0 if is_content(norms[i]) else 0.0
        p += (i / max(N - 1, 1)) * 0.15
        p += i * 1e-8
        priorities.append(p)
    top = sorted(range(N), key=lambda i: priorities[i], reverse=True)[:V]
    return [source_words[i] for i in sorted(top)]


def build_extractive_balanced(source_words, target_count, target_content_frac=0.6474):
    """Density-matched, source-wide extractive view.
    Matches compact's ~65% content fraction while covering full source span.
    Keeps a controlled mix of content and function words, distributed.
    """
    N = len(source_words)
    V = target_count
    if V >= N:
        return list(source_words[:V])
    norms = [norm_word(w) for w in source_words]
    is_cont = [is_content(norms[i]) for i in range(N)]

    content_idx = [i for i in range(N) if is_cont[i]]
    function_idx = [i for i in range(N) if not is_cont[i]]

    target_c = min(len(content_idx), max(1, round(target_content_frac * V)))
    target_f = V - target_c
    if target_f > len(function_idx):
        target_f = len(function_idx)
        target_c = V - target_f
    if target_c > len(content_idx):
        target_c = len(content_idx)
        target_f = V - target_c

    def evenly_spaced(indices, k):
        if k <= 0:
            return []
        if k >= len(indices):
            return list(indices)
        step = len(indices) / k
        sel = []
        for j in range(k):
            idx = int(j * step + step / 2)
            idx = min(idx, len(indices) - 1)
            sel.append(indices[idx])
        sel = sorted(set(sel))
        # Fill any deduplication shortfall
        remaining = [i for i in indices if i not in set(sel)]
        while len(sel) < k and remaining:
            sel.append(remaining.pop(0))
            sel.sort()
        return sel[:k]

    kept_c = evenly_spaced(content_idx, target_c)
    kept_f = evenly_spaced(function_idx, target_f)
    kept = sorted(set(kept_c + kept_f))

    # Final adjustment
    while len(kept) > V:
        # Remove the earliest function word
        for i in list(kept):
            if not is_cont[i]:
                kept.remove(i)
                break
        else:
            kept.pop(0)
    while len(kept) < V:
        candidates = [i for i in range(N) if i not in set(kept)]
        if not candidates:
            break
        # Add later positions first
        kept.append(max(candidates))
        kept.sort()
    return [source_words[i] for i in kept]


# ── Atlas computation (matching research structure) ──

def compute_pair_atlas(source_text, view_text, source_words_list, view_words_list, tokenizer=None):
    """Compute atlas metrics for one source-view pair."""
    src_w = source_words_list
    vw_w = view_words_list
    N_s = len(src_w)
    N_v = len(vw_w)

    src_norms = [norm_word(w) for w in src_w]
    vw_norms = [norm_word(w) for w in vw_w]

    # Content positions in source
    content_positions = [i for i, n in enumerate(src_norms) if is_content(n)]
    N_content = len(content_positions)

    # Source content word multiset
    src_content_bag = Counter(n for n in src_norms if is_content(n))

    # View content words
    vw_content = [n for n in vw_norms if is_content(n)]
    vw_content_bag = Counter(vw_content)

    # Coverage: fraction of source content words found in view
    if N_content == 0:
        source_content_coverage = 0.0
    else:
        covered = sum(min(vw_content_bag.get(n, 0), src_content_bag[n]) for n in src_content_bag)
        source_content_coverage = covered / N_content

    # Content fraction
    content_fraction = len(vw_content) / N_v if N_v > 0 else 0.0

    # Source-absent content: content words in view not in source
    copied_content = []
    source_absent_content = []
    remaining_src = dict(src_content_bag)
    for n in vw_content:
        if remaining_src.get(n, 0) > 0:
            copied_content.append(n)
            remaining_src[n] -= 1
        else:
            source_absent_content.append(n)

    # Tail analysis: source positions beyond the repeat cutoff
    repeat_cutoff = N_v  # first N_v words = repeat
    tail_content_positions = [i for i in content_positions if i >= repeat_cutoff]
    if tail_content_positions:
        tail_content_norms = [src_norms[i] for i in tail_content_positions]
        tail_bag = Counter(tail_content_norms)
        tail_covered = sum(min(vw_content_bag.get(n, 0), tail_bag[n]) for n in tail_bag)
        tail_content_coverage = tail_covered / len(tail_content_positions)
    else:
        tail_content_coverage = None

    # Source position decile coverage
    decile_coverage = {}
    if N_content > 0:
        for pos in content_positions:
            d = min(int(pos / max(N_s, 1) * 10), 9)
            if d not in decile_coverage:
                decile_coverage[d] = {"total": 0, "covered": 0}
            decile_coverage[d]["total"] += 1
            if src_norms[pos] in vw_content_bag and vw_content_bag[src_norms[pos]] > 0:
                decile_coverage[d]["covered"] += 1

    # Token-level stats if tokenizer provided
    tok_stats = {}
    if tokenizer is not None:
        src_enc = tokenizer.encode(source_text, add_special_tokens=False)
        vw_enc = tokenizer.encode(" ".join(vw_w), add_special_tokens=False)
        tok_stats["source_active_tokens"] = len(src_enc)
        tok_stats["view_active_tokens"] = len(vw_enc)

    return {
        "source_words": N_s,
        "view_words": N_v,
        "ratio_view_to_source": N_v / N_s if N_s > 0 else 0,
        "source_content_coverage": source_content_coverage,
        "content_fraction": content_fraction,
        "compact_source_absent_content_words": len(source_absent_content),
        "compact_copied_content_words": len(copied_content),
        "compact_tail_content_coverage": tail_content_coverage,
        "compact_content_fraction": content_fraction,
        "compact_tail_only_content_words": len(tail_content_positions) if tail_content_positions else 0,
        "decile_coverage": decile_coverage,
        **tok_stats,
    }


def aggregate_atlas(pair_records):
    """Compute aggregate statistics matching research format."""
    def stat(values):
        clean = [v for v in values if v is not None and math.isfinite(v)]
        if not clean:
            return {"n": 0}
        return {
            "n": len(clean),
            "mean": statistics.mean(clean),
            "median": statistics.median(clean),
            "p10": sorted(clean)[max(0, int(len(clean) * 0.1))],
            "p25": sorted(clean)[max(0, int(len(clean) * 0.25))],
            "p75": sorted(clean)[min(len(clean) - 1, int(len(clean) * 0.75))],
            "p90": sorted(clean)[min(len(clean) - 1, int(len(clean) * 0.9))],
            "min": min(clean),
            "max": max(clean),
        }

    agg = {
        "pairs": len(pair_records),
        "total_source_words": sum(r["source_words"] for r in pair_records),
        "total_view_words": sum(r["view_words"] for r in pair_records),
        "stats": {
            "source_content_coverage": stat([r["source_content_coverage"] for r in pair_records]),
            "content_fraction": stat([r["content_fraction"] for r in pair_records]),
            "compact_source_absent_content_words": stat([r["compact_source_absent_content_words"] for r in pair_records]),
            "compact_tail_content_coverage": stat([r["compact_tail_content_coverage"] for r in pair_records]),
        },
        "fractions": {
            "pairs_with_tail_content_recovery": statistics.mean([1.0 if (r["compact_tail_content_coverage"] or 0) > 0 else 0.0 for r in pair_records]),
            "pairs_with_any_source_absent_content": statistics.mean([1.0 if r["compact_source_absent_content_words"] > 0 else 0.0 for r in pair_records]),
        },
        "content_word_totals": {
            "view_content_words": sum(r["compact_copied_content_words"] + r["compact_source_absent_content_words"] for r in pair_records),
            "copied_content_words": sum(r["compact_copied_content_words"] for r in pair_records),
            "source_absent_content_words": sum(r["compact_source_absent_content_words"] for r in pair_records),
        },
    }

    # Decile coverage aggregation
    decile_totals = {}
    for r in pair_records:
        for d_str, dc in r.get("decile_coverage", {}).items():
            d = int(d_str) if isinstance(d_str, str) else d_str
            if d not in decile_totals:
                decile_totals[d] = {"total": 0, "covered": 0}
            decile_totals[d]["total"] += dc["total"]
            decile_totals[d]["covered"] += dc["covered"]
    agg["source_position_decile_coverage"] = []
    for d in range(10):
        if d in decile_totals and decile_totals[d]["total"] > 0:
            cov = decile_totals[d]["covered"] / decile_totals[d]["total"]
        else:
            cov = 0.0
        agg["source_position_decile_coverage"].append({
            "decile": d,
            "source_content_positions": decile_totals.get(d, {}).get("total", 0),
            "coverage": cov,
        })

    # Token totals
    if "view_active_tokens" in pair_records[0]:
        agg["token_totals"] = {
            "total_source_active_tokens": sum(r.get("source_active_tokens", 0) for r in pair_records),
            "total_view_active_tokens": sum(r.get("view_active_tokens", 0) for r in pair_records),
        }

    return agg


# ── Pool construction ──

def build_pool_row(pairs_in_row, extractive_views):
    """Build a single pool row text from source + extractive view pairs."""
    parts = []
    for pair in pairs_in_row:
        parts.append(pair["source_text"])
        parts.append(extractive_views[pair["pair_id"]])
    return " ".join(parts)


# ── Main ──

def find_user_root():
    return _PUBLIC_ROOT


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def now_utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main():
    ROOT = find_user_root()
    WS = ROOT / "experiments/archive/frontier_consolidation"

    PAIRS_PATH = WS / "data/factorial_view_candidate_audit/compact_candidate_pairs.jsonl"
    ROW_META_PATH = WS / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_changed_block_rows_meta.jsonl"
    COMPACT_POOL_PATH = WS / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"
    REPEAT_POOL_PATH = WS / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_repeat_compact_reinvest_10M.jsonl"
    TOKENIZER_PATH = WS / "data/compliant_tokenizer"
    OUT_DIR = WS / "data/extractive_view_pool_preflight"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading pairs from {PAIRS_PATH} ...", flush=True)
    pairs = []
    pair_by_id = {}
    with open(PAIRS_PATH) as f:
        for line in f:
            p = json.loads(line)
            pairs.append(p)
            pair_by_id[p["pair_id"]] = p
    print(f"  {len(pairs)} pairs loaded", flush=True)

    print(f"Reading row metadata from {ROW_META_PATH} ...", flush=True)
    row_metas = []
    with open(ROW_META_PATH) as f:
        for line in f:
            row_metas.append(json.loads(line))
    print(f"  {len(row_metas)} rows", flush=True)

    # Load tokenizer
    tokenizer = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH))
        print(f"  Tokenizer loaded: vocab={tokenizer.vocab_size}", flush=True)
    except Exception as e:
        print(f"  Tokenizer load failed: {e}; skipping token analysis", flush=True)

    # ── Build extractive views ──
    variants = ["extractive_wide", "extractive_balanced"]
    extractive_views = {v: {} for v in variants}
    pair_atlases = {v: [] for v in variants}

    # Also compute repeat atlas for comparison
    repeat_atlases = []

    print("Building extractive views for all pairs ...", flush=True)
    for pi, pair in enumerate(pairs):
        src_words = split_words(pair["source_text"])
        compact_words = split_words(pair["view_text"])
        target_count = len(compact_words)
        pid = pair["pair_id"]

        # Repeat view: first target_count words
        repeat_words = src_words[:target_count]

        # Extractive wide
        wide_words = build_extractive_wide(src_words, target_count)
        extractive_views["extractive_wide"][pid] = " ".join(wide_words)

        # Extractive balanced
        balanced_words = build_extractive_balanced(src_words, target_count)
        extractive_views["extractive_balanced"][pid] = " ".join(balanced_words)

        # Atlas for each variant
        for vname, vw in [("extractive_wide", wide_words),
                          ("extractive_balanced", balanced_words)]:
            atlas = compute_pair_atlas(
                pair["source_text"], " ".join(vw),
                src_words, vw, tokenizer
            )
            atlas["pair_id"] = pid
            pair_atlases[vname].append(atlas)

        # Repeat atlas
        repeat_atlas = compute_pair_atlas(
            pair["source_text"], " ".join(repeat_words),
            src_words, repeat_words, tokenizer
        )
        repeat_atlases.append(repeat_atlas)

        # Also compute compact atlas
        if pi == 0:
            compact_atlases = []
        compact_atlas = compute_pair_atlas(
            pair["source_text"], pair["view_text"],
            src_words, compact_words, tokenizer
        )
        compact_atlases.append(compact_atlas)

        if (pi + 1) % 3000 == 0:
            print(f"  {pi + 1}/{len(pairs)} pairs processed", flush=True)

    print(f"  All {len(pairs)} pairs processed", flush=True)

    # ── Aggregate atlas stats ──
    agg_atlases = {}
    for vname in variants:
        agg_atlases[vname] = aggregate_atlas(pair_atlases[vname])
    agg_atlases["compact"] = aggregate_atlas(compact_atlases)
    agg_atlases["repeat"] = aggregate_atlas(repeat_atlases)

    # ── Build 10M pool files ──
    print("Reading compact pool for filler rows ...", flush=True)
    compact_rows = []
    with open(COMPACT_POOL_PATH) as f:
        for line in f:
            compact_rows.append(json.loads(line))
    n_changed = len(row_metas)
    filler_rows = compact_rows[n_changed:]
    print(f"  {len(filler_rows)} filler rows (unchanged)", flush=True)

    for vname in variants:
        pool_path = OUT_DIR / f"{vname}_10M.jsonl"
        print(f"Building {vname} 10M pool -> {pool_path} ...", flush=True)
        total_words = 0
        row_count = 0
        with open(pool_path, "w") as fout:
            # Changed block rows
            for ri, rmeta in enumerate(row_metas):
                if not rmeta["pair_ids"]:
                    # Topup row: use compact pool text unchanged
                    row_data = compact_rows[ri]
                    json.dump(row_data, fout, ensure_ascii=False)
                    fout.write("\n")
                    total_words += row_data["words"]
                    row_count += 1
                    continue

                pair_list = [pair_by_id[pid] for pid in rmeta["pair_ids"]]
                # Build row text
                parts = []
                word_sum = 0
                for pair in pair_list:
                    parts.append(pair["source_text"])
                    ev_text = extractive_views[vname][pair["pair_id"]]
                    parts.append(ev_text)
                    word_sum += pair["source_words"] + len(split_words(ev_text))
                row_text = " ".join(parts)
                actual_words = len(split_words(row_text))

                expected_words = rmeta["words"]
                if actual_words != expected_words:
                    # Try to diagnose
                    print(f"  WARNING: row {ri} word mismatch: expected {expected_words}, got {actual_words} (word_sum={word_sum})", flush=True)

                row_data = {
                    "text": row_text,
                    "words": actual_words,
                    "example_id": rmeta["example_id"],
                    "source": f"extractive_{vname}",
                }
                json.dump(row_data, fout, ensure_ascii=False)
                fout.write("\n")
                total_words += actual_words
                row_count += 1

            # Filler rows (unchanged)
            for frow in filler_rows:
                json.dump(frow, fout, ensure_ascii=False)
                fout.write("\n")
                total_words += frow["words"]
                row_count += 1

        pool_sha = sha256_file(pool_path)
        print(f"  {vname}: {row_count} rows, {total_words} words, SHA={pool_sha[:16]}...", flush=True)
        agg_atlases[vname]["pool"] = {
            "path": str(pool_path),
            "sha256": pool_sha,
            "rows": row_count,
            "total_words": total_words,
        }

    # ── Stream-level token analysis ──
    if tokenizer is not None:
        print("Computing stream-level tokenizer properties ...", flush=True)
        for vname in variants + ["compact", "repeat"]:
            if vname == "compact":
                pool_path = COMPACT_POOL_PATH
            elif vname == "repeat":
                pool_path = REPEAT_POOL_PATH
            else:
                pool_path = OUT_DIR / f"{vname}_10M.jsonl"
            total_active = 0
            total_wg = 0
            changed_active = 0
            changed_wg = 0
            with open(pool_path) as f:
                for li, line in enumerate(f):
                    row = json.loads(line)
                    enc = tokenizer.encode(row["text"], add_special_tokens=False)
                    wg = len(split_words(row["text"]))
                    total_active += len(enc)
                    total_wg += wg
                    if li < n_changed:
                        changed_active += len(enc)
                        changed_wg += wg
            agg_atlases[vname]["stream_tokens"] = {
                "total_active_tokens_10M": total_active,
                "total_word_groups_10M": total_wg,
                "changed_block_active_tokens": changed_active,
                "changed_block_word_groups": changed_wg,
            }
            print(f"  {vname}: active_tokens={total_active}, word_groups={total_wg}, changed_active={changed_active}", flush=True)

    # ── Comparison table ──
    comparison = {}
    ref_names = ["compact", "repeat", "extractive_wide", "extractive_balanced"]
    for metric in ["source_content_coverage", "content_fraction", "compact_tail_content_coverage"]:
        comparison[metric] = {}
        for vn in ref_names:
            s = agg_atlases[vn]["stats"].get(metric, {})
            comparison[metric][vn] = s.get("mean", None)

    comparison["source_absent_content_total"] = {}
    for vn in ref_names:
        comparison["source_absent_content_total"][vn] = agg_atlases[vn]["content_word_totals"].get("source_absent_content_words", 0)

    comparison["decile_coverage"] = {}
    for vn in ref_names:
        dc = agg_atlases[vn].get("source_position_decile_coverage", [])
        comparison["decile_coverage"][vn] = {d["decile"]: round(d["coverage"], 4) for d in dc}

    if "stream_tokens" in agg_atlases.get("compact", {}):
        comparison["stream_tokens_10M"] = {}
        for vn in ref_names:
            st = agg_atlases[vn].get("stream_tokens", {})
            comparison["stream_tokens_10M"][vn] = st.get("total_active_tokens_10M", None)
        comparison["changed_block_active_tokens"] = {}
        for vn in ref_names:
            st = agg_atlases[vn].get("stream_tokens", {})
            comparison["changed_block_active_tokens"][vn] = st.get("changed_block_active_tokens", None)

    # ── Example extractive views ──
    examples = []
    for pair in pairs[:5]:
        pid = pair["pair_id"]
        examples.append({
            "pair_id": pid,
            "source": pair["source_text"],
            "compact": pair["view_text"],
            "repeat": " ".join(split_words(pair["source_text"])[:pair["view_words"]]),
            "extractive_wide": extractive_views["extractive_wide"][pid],
            "extractive_balanced": extractive_views["extractive_balanced"][pid],
        })

    # ── Word count verification ──
    word_mismatches = {v: 0 for v in variants}
    for vname in variants:
        for pi, pair in enumerate(pairs):
            ev_words = len(split_words(extractive_views[vname][pair["pair_id"]]))
            if ev_words != pair["view_words"]:
                word_mismatches[vname] += 1

    # ── Write outputs ──
    result = {
        "status": "EXTRACTIVE_VIEW_POOL_PREFLIGHT",
        "created_utc": now_utc(),
        "pairs": len(pairs),
        "changed_block_rows": n_changed,
        "variants": variants,
        "word_count_mismatches": word_mismatches,
        "comparison": comparison,
        "aggregate_atlases": agg_atlases,
        "examples": examples,
        "interpretation": {
            "extractive_wide": "Content-maximizing source-wide extractive view. Uses only source words, drops function words preferentially. Expected: very high content density (~90%), broad source-position coverage matching compact.",
            "extractive_balanced": "Density-matched source-wide extractive view. Matches compact's ~65% content fraction by retaining selected function words. Expected: moderate density (~65%), broad coverage.",
            "experimental_design": {
                "comparison_arms": ["compact", "repeat", "extractive_wide", "extractive_balanced"],
                "outcomes": {
                    "extractive_approx_compact_gt_repeat": "Source-wide coverage/content selection is sufficient; generation is NOT necessary.",
                    "compact_gt_extractive_gt_repeat": "Coverage contributes, but contextual re-expression/abstraction adds downstream value.",
                    "compact_gt_extractive_approx_repeat": "Tail access alone insufficient; compact re-realization, density, or context recomposition is central.",
                    "extractive_gt_compact": "Original source wording/discourse continuity more useful than compression/paraphrase.",
                    "local_only_no_downstream": "Another local-learning/downstream split; do not promote.",
                },
            },
        },
        "no_training_official_eval_upload_aoa_or_leaderboard": True,
    }

    out_json = OUT_DIR / "extractive_view_pool_preflight.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Markdown summary
    md_lines = ["# research extractive view pool preflight\n"]
    md_lines.append(f"Pairs: {len(pairs)}, changed block rows: {n_changed}\n")
    md_lines.append(f"Word count mismatches: {word_mismatches}\n")

    md_lines.append("\n## Content density comparison (mean content_fraction)\n")
    for vn in ref_names:
        val = comparison["content_fraction"].get(vn)
        md_lines.append(f"- {vn}: {val:.4f}" if val else f"- {vn}: N/A")

    md_lines.append("\n## Source content coverage (mean)\n")
    for vn in ref_names:
        val = comparison["source_content_coverage"].get(vn)
        md_lines.append(f"- {vn}: {val:.4f}" if val else f"- {vn}: N/A")

    md_lines.append("\n## Tail content coverage (mean)\n")
    for vn in ref_names:
        val = comparison["compact_tail_content_coverage"].get(vn)
        md_lines.append(f"- {vn}: {val:.4f}" if val is not None else f"- {vn}: N/A")

    md_lines.append("\n## Source-absent content words (total)\n")
    for vn in ref_names:
        val = comparison["source_absent_content_total"].get(vn, 0)
        md_lines.append(f"- {vn}: {val}")

    md_lines.append("\n## Source position decile coverage\n")
    md_lines.append("| Decile | compact | repeat | ext_wide | ext_balanced |")
    md_lines.append("|--------|---------|--------|----------|--------------|")
    for d in range(10):
        vals = []
        for vn in ref_names:
            dc = comparison["decile_coverage"].get(vn, {})
            vals.append(f"{dc.get(d, 0):.4f}")
        md_lines.append(f"| {d} | {' | '.join(vals)} |")

    if "stream_tokens_10M" in comparison:
        md_lines.append("\n## Stream tokens (10M pool)\n")
        for vn in ref_names:
            val = comparison["stream_tokens_10M"].get(vn)
            md_lines.append(f"- {vn}: {val}")
        md_lines.append("\n## Changed block active tokens\n")
        for vn in ref_names:
            val = comparison["changed_block_active_tokens"].get(vn)
            md_lines.append(f"- {vn}: {val}")

    md_lines.append("\n## Examples\n")
    for ex in examples[:3]:
        md_lines.append(f"\n### {ex['pair_id']}")
        md_lines.append(f"- **Source**: {ex['source']}")
        md_lines.append(f"- **Compact**: {ex['compact']}")
        md_lines.append(f"- **Repeat**: {ex['repeat']}")
        md_lines.append(f"- **Extractive wide**: {ex['extractive_wide']}")
        md_lines.append(f"- **Extractive balanced**: {ex['extractive_balanced']}")

    md_lines.append(f"\n\nJSON: `{out_json}`")
    out_md = OUT_DIR / "extractive_view_pool_preflight.md"
    with open(out_md, "w") as f:
        f.write("\n".join(md_lines))

    print(json.dumps({
        "status": result["status"],
        "out_json": str(out_json),
        "out_md": str(out_md),
        "variants": variants,
        "word_count_mismatches": word_mismatches,
        "no_training_official_eval_upload_aoa_or_leaderboard": True,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
