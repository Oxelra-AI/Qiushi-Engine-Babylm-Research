#!/usr/bin/env python3
"""research — Legal custom-corpus materializer for BabyLM Strict-Small.

Builds a 10M-word corpus mixing official BabyLM sources with additional
Simple Wikipedia text, designed to increase entity/knowledge density while
preserving developmental/conversational signals for Reading/AoA protection.

Sources (all legally accessible, no gated data, no LLM generation):
  - Official BabyLM 2026 Strict-Small corpus (partial retention)
  - Simple English Wikipedia via wikimedia/wikipedia streaming
  - GEM/wiki_auto_asset_turk simplified targets (meaning-simplified text)

Two arms for matched comparison:
  official_only: exact official 10M corpus (control)
  custom_mix: ~65% official + ~35% entity/knowledge-rich additions

All word counts use whitespace splitting, consistent with BabyLM accounting.
"""
from __future__ import annotations
import argparse, json, os, pathlib, random, sys, time
from collections import defaultdict

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
sys.path.insert(0, str(ROOT / "training/scripts"))


def build_official(out_dir: pathlib.Path, target_words: int | None = None) -> tuple[str, int, dict]:
    """Load official BabyLM corpus, optionally truncate to target_words."""
    import argparse as ap
    from babylm_masked_train_leadershape import download_dataset, TRAIN_FILES
    ns = ap.Namespace(
        dataset_id="BabyLM-community/BabyLM-2026-Strict-Small",
        dataset_revision="c92ab16b4f08858304b0815706065b3354d8fc0a",
    )
    raw_dir, manifest = download_dataset(ns, out_dir / "official_raw")
    
    # Read per-source
    source_texts = {}
    source_words = {}
    total = 0
    for f in TRAIN_FILES:
        text = (raw_dir / f).read_text(encoding="utf-8")
        wc = len(text.split())
        source_texts[f] = text
        source_words[f] = wc
        total += wc
    
    if target_words is None or target_words >= total:
        full_text = "\n".join(source_texts[f] for f in TRAIN_FILES)
        return full_text, total, {"sources": source_words, "total": total, "truncated": False}
    
    # Proportional truncation. Keep per-source word lists so we can top up exactly.
    ratio = target_words / total
    source_word_lists = {f: source_texts[f].split() for f in TRAIN_FILES}
    take = {}
    actual = 0
    for f in TRAIN_FILES:
        n_take = int(len(source_word_lists[f]) * ratio)
        take[f] = n_take
        actual += n_take

    # Top up (or trim) to hit the exact target by drawing remaining words from the
    # largest source that still has spare words. This actually changes the emitted
    # text, not just the counter.
    diff = target_words - actual
    if diff > 0:
        # add words: iterate sources by spare capacity, largest first
        order = sorted(TRAIN_FILES, key=lambda f: len(source_word_lists[f]) - take[f], reverse=True)
        for f in order:
            if diff <= 0:
                break
            spare = len(source_word_lists[f]) - take[f]
            add = min(spare, diff)
            take[f] += add
            diff -= add
    elif diff < 0:
        # remove words: trim from the largest taken source first
        order = sorted(TRAIN_FILES, key=lambda f: take[f], reverse=True)
        need = -diff
        for f in order:
            if need <= 0:
                break
            rem = min(take[f], need)
            take[f] -= rem
            need -= rem

    parts = [" ".join(source_word_lists[f][:take[f]]) for f in TRAIN_FILES]
    used = {f: take[f] for f in TRAIN_FILES}
    actual = sum(used.values())
    return "\n".join(parts), actual, {"sources": used, "total": actual, "truncated": True, "target": target_words}


def build_simple_wiki(target_words: int, seed: int = 217) -> tuple[str, int, dict]:
    """Stream Simple English Wikipedia articles, select diverse set."""
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train", streaming=True)
    
    rng = random.Random(seed)
    articles = []
    total = 0
    
    for item in ds:
        text = item["text"].strip()
        if not text:
            continue
        wc = len(text.split())
        if wc < 50:  # Skip stubs
            continue
        # Cap individual articles at 800 words for diversity
        if wc > 800:
            words = text.split()[:800]
            text = " ".join(words)
            wc = 800
        articles.append(text)
        total += wc
        if total >= target_words * 1.5:  # Collect surplus for selection
            break
    
    # Shuffle and select to target
    rng.shuffle(articles)
    selected = []
    used = 0
    for art in articles:
        wc = len(art.split())
        if used + wc > target_words:
            # Trim last article
            remaining = target_words - used
            if remaining > 20:
                selected.append(" ".join(art.split()[:remaining]))
                used += remaining
            break
        selected.append(art)
        used += wc
    
    corpus = "\n".join(selected)
    meta = {
        "source": "wikimedia/wikipedia:20231101.simple",
        "articles_available": len(articles),
        "articles_selected": len(selected),
        "words": used,
        "target_words": target_words,
        "seed": seed,
        "min_article_words": 50,
        "max_article_words": 800,
    }
    return corpus, used, meta


def build_gem_simplified(target_words: int, seed: int = 217) -> tuple[str, int, dict]:
    """Extract simplified target sentences from GEM/wiki_auto_asset_turk."""
    from datasets import load_dataset
    ds = load_dataset("GEM/wiki_auto_asset_turk", split="train", streaming=True)
    
    sentences = []
    total = 0
    for item in ds:
        # Use the simplified target text
        text = item.get("target", "").strip()
        if not text or len(text.split()) < 5:
            continue
        sentences.append(text)
        total += len(text.split())
        if total >= target_words * 1.3:
            break
    
    rng = random.Random(seed)
    rng.shuffle(sentences)
    selected = []
    used = 0
    for sent in sentences:
        wc = len(sent.split())
        if used + wc > target_words:
            remaining = target_words - used
            if remaining > 5:
                selected.append(" ".join(sent.split()[:remaining]))
                used += remaining
            break
        selected.append(sent)
        used += wc
    
    corpus = "\n".join(selected)
    meta = {
        "source": "GEM/wiki_auto_asset_turk:train:target",
        "sentences_available": len(sentences),
        "sentences_selected": len(selected),
        "words": used,
        "target_words": target_words,
        "seed": seed,
    }
    return corpus, used, meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True)
    p.add_argument("--total_words", type=int, default=10_000_000)
    p.add_argument("--custom_fraction", type=float, default=0.35,
                   help="Fraction of corpus from custom sources (SimpleWiki + GEM simplified)")
    p.add_argument("--wiki_fraction_of_custom", type=float, default=0.70,
                   help="What fraction of the custom portion comes from SimpleWiki vs GEM")
    p.add_argument("--seed", type=int, default=217)
    args = p.parse_args()
    
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    custom_words = int(args.total_words * args.custom_fraction)
    official_words = args.total_words - custom_words
    wiki_words = int(custom_words * args.wiki_fraction_of_custom)
    gem_words = custom_words - wiki_words
    
    print(f"Target: {args.total_words:,} total = {official_words:,} official + "
          f"{wiki_words:,} SimpleWiki + {gem_words:,} GEM simplified")
    
    # Build official portion
    print("\n[1/3] Building official portion...")
    off_text, off_wc, off_meta = build_official(out, official_words)
    print(f"  Official: {off_wc:,} words")
    
    # Build Simple Wikipedia portion
    print("\n[2/3] Building Simple Wikipedia portion...")
    wiki_text, wiki_wc, wiki_meta = build_simple_wiki(wiki_words, args.seed)
    print(f"  SimpleWiki: {wiki_wc:,} words ({wiki_meta['articles_selected']} articles)")
    
    # Build GEM simplified portion
    print("\n[3/3] Building GEM simplified portion...")
    gem_text, gem_wc, gem_meta = build_gem_simplified(gem_words, args.seed)
    print(f"  GEM simplified: {gem_wc:,} words ({gem_meta['sentences_selected']} sentences)")
    
    # Combine and verify
    combined = off_text + "\n" + wiki_text + "\n" + gem_text
    actual_total = len(combined.split())
    
    def enforce_exact_word_count(text: str, target: int, reservoir_texts: list[str]) -> tuple[str, int, dict]:
        """Make the final emitted text contain exactly target whitespace words.

        Internal counters can be off by a few words because component strings may
        contain blank/spacing artifacts. The on-disk word count is the authority.
        If short, append words from legal reservoir_texts; if long, trim.
        """
        words = text.split()
        before = len(words)
        if before < target:
            need = target - before
            added = []
            for res in reservoir_texts:
                if need <= 0:
                    break
                rwords = res.split()
                take = min(need, len(rwords))
                added.extend(rwords[:take])
                need -= take
            if need > 0:
                raise RuntimeError(f"Could not top up exact word count: still need {need}")
            words.extend(added)
        elif before > target:
            words = words[:target]
        after = len(words)
        return " ".join(words), after, {"before": before, "after": after, "target": target, "adjustment_words": after - before}

    # Save custom mix with exact final count enforced from legal source reservoirs
    mix_path = out / "custom_mix_10M.txt"
    combined_exact, actual_total_exact, mix_exact_meta = enforce_exact_word_count(
        combined, args.total_words, [wiki_text, gem_text, off_text]
    )
    mix_path.write_text(combined_exact, encoding="utf-8")
    actual_total = actual_total_exact
    
    # Save official-only control, exact final count enforced from its own official text
    print("\n[ctrl] Building official-only control...")
    ctrl_text, ctrl_wc, ctrl_meta = build_official(out, None)
    ctrl_path = out / "official_only_10M.txt"
    ctrl_exact, ctrl_wc_exact, ctrl_exact_meta = enforce_exact_word_count(
        ctrl_text, args.total_words, [ctrl_text]
    )
    ctrl_path.write_text(ctrl_exact, encoding="utf-8")
    ctrl_wc = ctrl_wc_exact
    
    # Manifest
    manifest = {
        "status": "CUSTOM_CORPUS",
        "total_target": args.total_words,
        "custom_fraction": args.custom_fraction,
        "actual_total_words": actual_total,
        "composition": {
            "official_retained": {"words": off_wc, "meta": off_meta},
            "simple_wiki_added": {"words": wiki_wc, "meta": wiki_meta},
            "gem_simplified_added": {"words": gem_wc, "meta": gem_meta},
        },
        "control": {"words": ctrl_wc, "meta": ctrl_meta},
        "files": {
            "custom_mix": str(mix_path),
            "official_only": str(ctrl_path),
        },
        "legality": {
            "all_sources_public": True,
            "no_gated_data": True,
            "no_llm_generation": True,
            "total_within_10M_budget": actual_total <= 10_000_000,
            "provenance": [
                "BabyLM-community/BabyLM-2026-Strict-Small (MIT license)",
                "wikimedia/wikipedia:20231101.simple (CC BY-SA 3.0)",
                "GEM/wiki_auto_asset_turk (CC BY-SA 4.0 via Wikipedia)",
            ],
        },
        "screen_spec": {
            "arms": ["official_only", "custom_mix"],
            "model": "protected DeBERTa-v2 8x480 baseline16k WWM",
            "exposure": "10M first, 20M if signal, 100M only after clear Overall gain",
            "columns_required": ["BLiMP", "Supplement", "Entity", "EWoK", "COMPS", "GlobalPIQA", "Reading"],
            "success_criterion": "Entity/EWoK movement OR broad BLiMP/COMPS/Reading gain; not GlobalPIQA alone",
            "protect": "Supplement (59.88) and Reading (7.62) must not collapse",
        },
    }
    (out / "corpus_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nDone. Actual total: {actual_total:,} words (budget: {args.total_words:,})")
    print(f"  Custom mix: {mix_path}")
    print(f"  Official control: {ctrl_path}")
    print(f"  Manifest: {out / 'corpus_manifest.json'}")


if __name__ == "__main__":
    main()
