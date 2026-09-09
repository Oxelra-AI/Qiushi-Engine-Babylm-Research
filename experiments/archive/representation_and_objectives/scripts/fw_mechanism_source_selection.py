#!/usr/bin/env python3
"""research: FW mechanism-family source selection and prompt preparation.

CPU-only script. No training, no evaluation, no GPU use.

Reads all cached FineWeb source pools from the representation and consolidation
studies, unions them by normalised text hash, identifies sources with existing compact
rewrites, selects sources up to the mechanism budget with domain
diversity, and prepares compact-rewrite prompts for sources that still
need generation.

Design principles:
  - Shared compliant tokenizer from ~9.33M common words across all arms
  - Three arms: compact_view, source_repeat, source_diversity
  - FineWeb source spans are COMMON to all arms (only the companion text differs)
  - Do not displace official BabyLM rows
  - Replace inherited Qwen paraphrase-pair rows before any official displacement
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import statistics
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Iterable

# ── Paths ──────────────────────────────────────────────────────────
ROOT = pathlib.Path("experiments/archive/representation_and_objectives")
A02  = pathlib.Path("experiments/archive/frontier_consolidation")

# Cached FineWeb source pools
V3_PATH     = ROOT / "data/live_fineweb_source_selector_v3/live_fineweb_selector_v3_stable.jsonl"
V5_PATH     = ROOT / "data/live_fineweb_selector_v5_strictstable/live_fineweb_selector_v5_kept.jsonl"
QV2_PATH    = ROOT / "data/live_fineweb_quality_tiers/live_fineweb_quality_v2_kept.jsonl"

# Compact reinvest pairs (already have compact rewrites)
A02_REINVEST = A02 / "data/density_core_reinvestment_medium_riskhard/selected_compact_reinvest_pairs.jsonl"

# Output
OUT_DIR   = ROOT / "data/fw_mechanism_source_selection"
NOTE_PATH = (ROOT.parents[2] / 'research/notes/representation_and_objectives/fw_mechanism_source_selection.md')

# ── Constants ──────────────────────────────────────────────────────
OFFICIAL_WORDS   = 7_919_680
TOTAL_BUDGET     = 10_000_000
NEUTRAL_TOPUP    = 9   # same neutral topup as existing pool
COMPACT_RATIO    = 0.618  # Historical compact rewrite/source ratio

# Compact prompt configuration (same as the validated generation recipe)
COMPACT_SYSTEM = (
    "You rewrite factual English sentences for a small masked language "
    "model. Keep the exact same facts and all names, numbers, dates, "
    "quantities, comparisons, negation, and causal relations. Prefer "
    "fewer, clearer words. Output only one sentence."
)


def read_jsonl(path: pathlib.Path, limit: int = 0) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def wc(text: str) -> int:
    return len((text or "").split())


def norm_hash(text: str) -> str:
    t = " ".join(text.lower().split())
    return hashlib.sha256(t.encode("utf-8")).hexdigest()[:32]


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def entity_word_count(entities: list[str]) -> int:
    return sum(len(str(e).split()) for e in entities if str(e).strip())


def compact_target_words(source_text: str, entities: list = None,
                         numbers: list = None) -> int:
    """Set a source-length target with an entity/number retention floor."""
    source_words = wc(source_text)
    ent_words = entity_word_count(entities or [])
    nums = len(numbers or [])
    floor = ent_words + nums + 5
    target = max(8, math.ceil(0.72 * source_words), floor)
    if target >= source_words:
        target = max(8, source_words - 1)
        if floor >= source_words:
            target = source_words
    return int(target)


def stat(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0}
    xs = sorted(vals)
    n = len(xs)
    return {
        "n": n, "sum": sum(xs),
        "mean": round(statistics.fmean(xs), 4),
        "median": round(statistics.median(xs), 1),
        "p05": xs[int(0.05 * (n - 1))],
        "p90": xs[int(0.90 * (n - 1))],
        "p95": xs[int(0.95 * (n - 1))],
        "min": xs[0], "max": xs[-1],
    }


# ── PHASE 1: Load and union all cached FineWeb source pools ───────

def load_source_pool(path: pathlib.Path, pool_name: str) -> list[dict]:
    """Load a source pool, normalise keys."""
    rows = read_jsonl(path)
    out = []
    for r in rows:
        text = str(r.get("source_text") or r.get("text") or
                   r.get("sentence_text") or r.get("core_clean_text") or "").strip()
        if not text or wc(text) < 5:
            continue
        out.append({
            "text": text,
            "words": wc(text),
            "doc_id": str(r.get("doc_id", "")),
            "domains": list(r.get("core_domain_hits") or
                           r.get("domain_hits") or r.get("domains") or
                           ["no_domain"]),
            "entities": list(r.get("core_clean_capitalized_phrases") or
                            r.get("source_entities") or
                            r.get("entities") or []),
            "numbers": list(r.get("core_numbers") or
                           r.get("source_numbers") or
                           r.get("numbers") or []),
            "norm_hash": norm_hash(text),
            "pool": pool_name,
            "sentence_id": r.get("sentence_id"),
        })
    return out


def load_a02_compact_pairs(path: pathlib.Path) -> dict[str, dict]:
    """Load compact reinvest pairs, indexed by normalised source hash."""
    rows = read_jsonl(path)
    pairs = {}
    for r in rows:
        src = str(r.get("source_text") or r.get("source") or "").strip()
        rw  = str(r.get("rewrite_text") or r.get("rewrite") or "").strip()
        if not src or not rw:
            continue
        nh = norm_hash(src)
        pairs[nh] = {
            "source_text": src,
            "rewrite_text": rw,
            "source_words": wc(src),
            "rewrite_words": wc(rw),
            "pair_id": r.get("pair_id", ""),
            "content_recall": r.get("content_recall"),
            "entity_recall": r.get("entity_recall"),
            "number_recall": r.get("number_recall"),
            "doc_id": str(r.get("doc_id", "")),
        }
    return pairs


# ── PHASE 2: Select sources up to budget ──────────────────────────

def select_sources(union: dict[str, dict], a02_pairs: dict[str, dict],
                   target_source_words: int) -> list[dict]:
    """Select sources prioritising:
    1. Sources with existing compact rewrites (no generation needed)
    2. Domain diversity
    3. Document diversity
    Stops when cumulative source words reach target."""

    # Partition into has_rewrite and needs_rewrite
    has_rw = []
    needs_rw = []
    for nh, info in union.items():
        rec = dict(info)
        rec["has_rewrite"] = nh in a02_pairs
        if rec["has_rewrite"]:
            has_rw.append(rec)
        else:
            needs_rw.append(rec)

    # Sort has_rewrite by domain diversity (least frequent domains first)
    domain_freq = Counter()
    for r in has_rw:
        for d in r["domains"]:
            domain_freq[d] += 1
    for r in has_rw:
        r["_domain_score"] = min(
            (domain_freq.get(d, 999999) for d in r["domains"]),
            default=999999
        )
    has_rw.sort(key=lambda r: (r["_domain_score"], r["doc_id"]))

    # Sort needs_rewrite similarly but with broader pool frequency
    domain_freq2 = Counter()
    for r in needs_rw:
        for d in r["domains"]:
            domain_freq2[d] += 1
    for r in needs_rw:
        r["_domain_score"] = min(
            (domain_freq2.get(d, 999999) for d in r["domains"]),
            default=999999
        )
    needs_rw.sort(key=lambda r: (r["_domain_score"], r["doc_id"]))

    # Select: first all has_rewrite, then needs_rewrite up to budget
    selected = []
    cumulative_words = 0
    seen_docs = set()

    for r in has_rw:
        if cumulative_words >= target_source_words:
            break
        selected.append(r)
        cumulative_words += r["words"]
        seen_docs.add(r["doc_id"])

    for r in needs_rw:
        if cumulative_words >= target_source_words:
            break
        selected.append(r)
        cumulative_words += r["words"]
        seen_docs.add(r["doc_id"])

    return selected


# ── PHASE 3: Compute arm budgets ──────────────────────────────────

def compute_budgets(selected: list[dict], a02_pairs: dict[str, dict]) -> dict:
    """Compute exact word budgets for all arms."""
    source_words_total = sum(r["words"] for r in selected)

    # Estimate rewrite words from existing + expected
    existing_rw_words = 0
    expected_new_rw_words = 0
    n_has_rw = 0
    n_needs_rw = 0
    for r in selected:
        if r["has_rewrite"]:
            pair = a02_pairs[r["norm_hash"]]
            existing_rw_words += pair["rewrite_words"]
            n_has_rw += 1
        else:
            # Estimate at target ratio
            tgt = compact_target_words(r["text"], r.get("entities"), r.get("numbers"))
            expected_new_rw_words += tgt
            n_needs_rw += 1

    estimated_rewrite_words = existing_rw_words + expected_new_rw_words
    estimated_pair_words = source_words_total + estimated_rewrite_words

    # Remaining budget for retained Qwen paraphrase block
    remaining_for_qwen = TOTAL_BUDGET - OFFICIAL_WORDS - estimated_pair_words - NEUTRAL_TOPUP
    if remaining_for_qwen < 0:
        # Need to reduce FineWeb block
        excess = -remaining_for_qwen
        remaining_for_qwen = 0
        estimated_pair_words -= excess
    
    return {
        "source_words_total": source_words_total,
        "existing_rewrite_words": existing_rw_words,
        "expected_new_rewrite_words": expected_new_rw_words,
        "estimated_total_rewrite_words": estimated_rewrite_words,
        "estimated_pair_words": estimated_pair_words,
        "n_with_existing_rewrite": n_has_rw,
        "n_needing_new_rewrite": n_needs_rw,
        "retained_qwen_words": max(0, remaining_for_qwen),
        "official_words": OFFICIAL_WORDS,
        "neutral_topup": NEUTRAL_TOPUP,
        "total_10M": OFFICIAL_WORDS + estimated_pair_words + max(0, remaining_for_qwen) + NEUTRAL_TOPUP,
        "shared_tokenizer_pool_words": (
            OFFICIAL_WORDS + max(0, remaining_for_qwen) + source_words_total + NEUTRAL_TOPUP
        ),
    }


# ── PHASE 4: Prepare compact rewrite prompts ─────────────────────

def make_compact_prompt(source: dict, idx: int) -> dict:
    """Create a compact rewrite prompt for Qwen generation."""
    text = source["text"]
    tgt = compact_target_words(text, source.get("entities"), source.get("numbers"))
    return {
        "prompt_id": f"representation_and_objectives_fwcompact_step098_{idx:06d}",
        "typ": "compression",
        "view_regime": "compact_faithful",
        "source_text": text,
        "source_words": source["words"],
        "compression_target_words": tgt,
        "target_ratio": round(tgt / max(1, source["words"]), 4),
        "system": COMPACT_SYSTEM,
        "prompt": (
            f"Rewrite the factual sentence in simpler plain English using "
            f"at most {tgt} words if the facts allow it. "
            "Do not drop any named entity, number, date, quantity, "
            "comparison, negation, or causal relation. "
            "Do not add background knowledge or explanations. "
            "Do not copy the original wording when a shorter faithful "
            "wording is possible. "
            "Output exactly one grammatical English sentence.\n\n"
            "SOURCE SENTENCE:\n" + text
        ),
        "doc_id": source.get("doc_id", ""),
        "domains": source.get("domains", []),
        "norm_hash": source["norm_hash"],
        "source_pool": source.get("pool", ""),
    }


# ── MAIN ──────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ─── Load source pools ────────────────────────────────────────
    print("Loading source pools...", flush=True)
    pools = {}
    pool_paths = {
        "v3_stable": V3_PATH,
        "v5_strictstable": V5_PATH,
        "quality_v2_kept": QV2_PATH,
    }
    for name, path in pool_paths.items():
        if path.exists():
            pools[name] = load_source_pool(path, name)
            print(f"  {name}: {len(pools[name])} rows", flush=True)
        else:
            print(f"  {name}: NOT FOUND at {path}", flush=True)

    # Load existing compact pairs
    print("Loading existing compact reinvest pairs...", flush=True)
    a02_pairs = load_a02_compact_pairs(A02_REINVEST) if A02_REINVEST.exists() else {}
    print(f"  Existing compact pairs: {len(a02_pairs)}", flush=True)

    # ─── Union by normalised hash ─────────────────────────────────
    print("Union by normalised hash...", flush=True)
    union: dict[str, dict] = {}

    # Add existing compact sources first (they have rewrites)
    for nh, pair in a02_pairs.items():
        if nh not in union:
            union[nh] = {
                "text": pair["source_text"],
                "words": pair["source_words"],
                "doc_id": pair.get("doc_id", ""),
                "domains": [],  # will be enriched from pools
                "entities": [],
                "numbers": [],
                "norm_hash": nh,
                "pool": "a02_compact_reinvest",
                "sentence_id": None,
            }

    # Add pool sources, enriching domains if already present
    for pool_name, pool_rows in pools.items():
        for r in pool_rows:
            nh = r["norm_hash"]
            if nh in union:
                # Enrich domains
                existing_domains = set(union[nh].get("domains", []))
                for d in r["domains"]:
                    existing_domains.add(d)
                union[nh]["domains"] = sorted(existing_domains)
                if not union[nh].get("entities") and r.get("entities"):
                    union[nh]["entities"] = r["entities"]
                if not union[nh].get("numbers") and r.get("numbers"):
                    union[nh]["numbers"] = r["numbers"]
            else:
                union[nh] = r

    print(f"  Union: {len(union)} unique sources, "
          f"{sum(r['words'] for r in union.values())} total source words",
          flush=True)

    # ─── Compute scale options ────────────────────────────────────
    total_cached_source_words = sum(r["words"] for r in union.values())
    
    # At natural compact ratio, how many pair words can we make?
    max_natural_pair = total_cached_source_words * (1 + COMPACT_RATIO)
    max_remaining_qwen = TOTAL_BUDGET - OFFICIAL_WORDS - max_natural_pair - NEUTRAL_TOPUP

    print(f"\n  Total cached source words: {total_cached_source_words:,}")
    print(f"  Max natural pair words (ratio {COMPACT_RATIO}): {max_natural_pair:,.0f}")
    print(f"  Max remaining Qwen at natural ratio: {max_remaining_qwen:,.0f}")

    # Target: use all cached sources to maximise FineWeb mechanism scale
    target_source_words = total_cached_source_words
    
    # ─── Select sources ───────────────────────────────────────────
    print("\nSelecting sources...", flush=True)
    selected = select_sources(union, a02_pairs, target_source_words)
    
    actual_source_words = sum(r["words"] for r in selected)
    print(f"  Selected: {len(selected)} sources, {actual_source_words:,} words", flush=True)

    # ─── Compute budgets ──────────────────────────────────────────
    budgets = compute_budgets(selected, a02_pairs)
    print(f"\n  Budget summary:")
    print(f"    Official BabyLM:     {budgets['official_words']:>10,}")
    print(f"    FineWeb pair block:  {budgets['estimated_pair_words']:>10,}")
    print(f"    Retained Qwen:       {budgets['retained_qwen_words']:>10,}")
    print(f"    Neutral topup:       {budgets['neutral_topup']:>10,}")
    print(f"    Total:               {budgets['total_10M']:>10,}")
    print(f"    Shared tok pool:     {budgets['shared_tokenizer_pool_words']:>10,}")
    print(f"    Sources w/ rewrite:  {budgets['n_with_existing_rewrite']:>10,}")
    print(f"    Sources need gen:    {budgets['n_needing_new_rewrite']:>10,}")

    # ─── Domain distribution of selected sources ──────────────────
    domain_counts = Counter()
    for r in selected:
        for d in r.get("domains") or ["no_domain"]:
            domain_counts[d] += 1
    
    doc_ids = {r.get("doc_id", "") for r in selected}

    # ─── Prepare compact prompts for sources without rewrites ─────
    print("\nPreparing compact rewrite prompts...", flush=True)
    prompts = []
    for idx, r in enumerate(selected):
        if not r.get("has_rewrite"):
            prompts.append(make_compact_prompt(r, idx))

    prompt_path = OUT_DIR / "fw_mechanism_compact_prompts.jsonl"
    with prompt_path.open("w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(prompts)} prompts to {prompt_path}", flush=True)

    prompt_target_words = sum(p["compression_target_words"] for p in prompts)
    prompt_source_words = sum(p["source_words"] for p in prompts)

    # ─── Save frozen source list ──────────────────────────────────
    source_list_path = OUT_DIR / "fw_mechanism_frozen_sources.jsonl"
    with source_list_path.open("w", encoding="utf-8") as f:
        for r in selected:
            rec = {
                "text": r["text"],
                "words": r["words"],
                "norm_hash": r["norm_hash"],
                "doc_id": r.get("doc_id", ""),
                "domains": r.get("domains", []),
                "pool": r.get("pool", ""),
                "has_rewrite": r.get("has_rewrite", False),
            }
            if r.get("has_rewrite"):
                pair = a02_pairs.get(r["norm_hash"], {})
                rec["rewrite_text"] = pair.get("rewrite_text", "")
                rec["rewrite_words"] = pair.get("rewrite_words", 0)
                rec["content_recall"] = pair.get("content_recall")
                rec["entity_recall"] = pair.get("entity_recall")
                rec["number_recall"] = pair.get("number_recall")
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Wrote frozen source list to {source_list_path}", flush=True)

    # ─── Arm specifications ───────────────────────────────────────
    arms = {
        "compact_view": {
            "description": "FineWeb source + compact rewrite pairs",
            "fineweb_block_content": "source_sentence \\n compact_rewrite",
            "companion_type": "compact_rewrite",
        },
        "source_repeat": {
            "description": "Same FineWeb sources, deterministic source "
                           "repetition filling same word budget",
            "fineweb_block_content": "source_sentence \\n repeated_source_prefix",
            "companion_type": "source_repeat",
        },
        "source_diversity": {
            "description": "Same first-span sources, different FineWeb "
                           "sources from held-out pool filling second span",
            "fineweb_block_content": "source_sentence \\n different_source_sentence",
            "companion_type": "diverse_source",
            "note": "construct if feasible; train after compact/repeat comparison",
        },
    }

    # Common across all arms
    common_spec = {
        "official_babylm_words": OFFICIAL_WORDS,
        "retained_qwen_words": budgets["retained_qwen_words"],
        "neutral_topup_words": NEUTRAL_TOPUP,
        "fineweb_source_words": actual_source_words,
        "note": "All arms share the same official rows, retained Qwen rows, "
                "neutral topup, and FineWeb source sentences. They differ only "
                "in the companion text accompanying each source sentence.",
    }

    shared_tokenizer_spec = {
        "description": "Shared compliant tokenizer trained on text common "
                       "to all arms (~9.33M words)",
        "components": {
            "official_babylm": OFFICIAL_WORDS,
            "retained_qwen_paraphrase": budgets["retained_qwen_words"],
            "fineweb_source_spans": actual_source_words,
            "neutral_topup": NEUTRAL_TOPUP,
        },
        "total_words": budgets["shared_tokenizer_pool_words"],
        "rationale": "separate arm-specific tokenizers "
                     "would entangle compact alignment with segmentation, labels, "
                     "and WWM geometry. First compare data effect under one "
                     "representation; only after a real effect emerges should a "
                     "treatment-specific tokenizer optimise the SOTA endpoint.",
    }

    # ─── Save master manifest ─────────────────────────────────────
    manifest = {
        "status": "FW_MECHANISM_SOURCE_SELECTION",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "CPU-only source selection and prompt preparation for "
                   "mechanism-scale FineWeb source-compact family",
        "no_training_or_eval_launched": True,

        "source_pool_inputs": {
            name: {"path": str(path), "rows": len(pools.get(name, [])),
                   "sha256": sha256_file(path) if path.exists() else "MISSING"}
            for name, path in pool_paths.items()
        },
        "a02_compact_pairs_input": {
            "path": str(A02_REINVEST),
            "pairs": len(a02_pairs),
            "sha256": sha256_file(A02_REINVEST) if A02_REINVEST.exists() else "MISSING",
        },

        "union_summary": {
            "unique_sources": len(union),
            "total_source_words": total_cached_source_words,
            "sources_with_a02_rewrites": sum(
                1 for nh in union if nh in a02_pairs),
        },

        "selection_summary": {
            "selected_sources": len(selected),
            "selected_source_words": actual_source_words,
            "source_word_stats": stat([float(r["words"]) for r in selected]),
            "unique_docs": len(doc_ids),
            "domain_counts": dict(domain_counts.most_common()),
            "n_with_existing_rewrite": budgets["n_with_existing_rewrite"],
            "n_needing_new_rewrite": budgets["n_needing_new_rewrite"],
        },

        "budgets": budgets,

        "prompt_summary": {
            "n_prompts": len(prompts),
            "prompt_source_words": prompt_source_words,
            "prompt_target_words_sum": prompt_target_words,
            "expected_ratio": round(prompt_target_words / max(1, prompt_source_words), 4),
            "prompt_file": str(prompt_path),
            "prompt_file_sha256": sha256_file(prompt_path),
        },

        "arm_specifications": arms,
        "common_across_arms": common_spec,
        "shared_tokenizer_specification": shared_tokenizer_spec,

        "outputs": {
            "manifest": str(OUT_DIR / "fw_mechanism_source_selection.json"),
            "frozen_sources": str(source_list_path),
            "compact_prompts": str(prompt_path),
            "note": str(NOTE_PATH),
        },

        "generation_next_steps": {
            "1_generate_compact_rewrites": (
                f"Run Qwen3.5-9B on {len(prompts)} prompts "
                f"({prompt_source_words:,} source words → "
                f"~{prompt_target_words:,} target words). "
                "Use research-style batch generation or training generate."
            ),
            "2_validate_rewrites": (
                "Check entity/number/content recall on new rewrites "
                "against source text. Reject/regenerate failures."
            ),
            "3_materialize_arms": (
                "Once rewrites are validated, build three matched 10M "
                "corpora (compact_view, source_repeat, source_diversity) "
                "with exact word counts."
            ),
            "4_train_shared_tokenizer": (
                f"Train one 16k BPE tokenizer on the "
                f"~{budgets['shared_tokenizer_pool_words']:,}-word "
                "common pool text."
            ),
        },

        "elapsed_sec": round(time.time() - t0, 1),
    }

    manifest_path = OUT_DIR / "fw_mechanism_source_selection.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")

    # ─── Save note ────────────────────────────────────────────────
    NOTE_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# research — FW mechanism source selection\n\n",
        "CPU-only source selection and compact-rewrite prompt preparation "
        "for mechanism-scale FineWeb source–compact family.\n\n",
        f"## Source union\n"
        f"- Unique sources: {len(union):,}\n"
        f"- Total source words: {total_cached_source_words:,}\n"
        f"- Sources with A02 rewrites: {budgets['n_with_existing_rewrite']:,}\n"
        f"- Sources needing new rewrites: {budgets['n_needing_new_rewrite']:,}\n\n",
        f"## Budget\n"
        f"- Official BabyLM: {OFFICIAL_WORDS:,} words\n"
        f"- FineWeb pair block: ~{budgets['estimated_pair_words']:,} words\n"
        f"- Retained Qwen: ~{budgets['retained_qwen_words']:,} words\n"
        f"- Neutral topup: {NEUTRAL_TOPUP} words\n"
        f"- Shared tokenizer pool: ~{budgets['shared_tokenizer_pool_words']:,} words\n\n",
        f"## Compact rewrite prompts\n"
        f"- {len(prompts):,} prompts prepared\n"
        f"- Source words: {prompt_source_words:,}\n"
        f"- Target rewrite words: ~{prompt_target_words:,}\n"
        f"- Expected ratio: {prompt_target_words / max(1, prompt_source_words):.3f}\n\n",
        "## Arm design (shared tokenizer)\n"
        "All arms share one compliant tokenizer trained on ~9.33M common "
        "words (official + retained Qwen + FineWeb source spans + neutral). "
        "Arms differ only in what accompanies each FineWeb source sentence:\n"
        "- **compact_view**: faithful shorter rewrite\n"
        "- **source_repeat**: repeated source prefix (same words)\n"
        "- **source_diversity**: different FineWeb source (new propositions)\n\n",
        f"## Artifacts\n"
        f"- Manifest: `{manifest_path}`\n"
        f"- Frozen sources: `{source_list_path}`\n"
        f"- Compact prompts: `{prompt_path}`\n",
    ]
    NOTE_PATH.write_text("".join(lines), encoding="utf-8")

    # Print compact summary
    print(json.dumps({
        "status": manifest["status"],
        "selected_sources": len(selected),
        "selected_source_words": actual_source_words,
        "estimated_pair_words": budgets["estimated_pair_words"],
        "retained_qwen_words": budgets["retained_qwen_words"],
        "shared_tokenizer_words": budgets["shared_tokenizer_pool_words"],
        "n_with_rewrite": budgets["n_with_existing_rewrite"],
        "n_need_generation": budgets["n_needing_new_rewrite"],
        "prompt_target_words": prompt_target_words,
        "elapsed_sec": manifest["elapsed_sec"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
