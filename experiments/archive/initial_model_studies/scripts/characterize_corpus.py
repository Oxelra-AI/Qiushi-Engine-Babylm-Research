#!/usr/bin/env python3
"""research — Characterize the custom-mix corpus BEFORE training.

Pre-training validity requirement: do not launch two 10M runs on an arbitrary concatenation.
First measure whether the mix actually forms a higher-signal data mechanism that
could plausibly explain the leader's Entity/EWoK ability structure, or whether it
just glues official text + extra SimpleWiki + weakly-transferring GEM simplified.

Metrics per source stream (official-retained, SimpleWiki-added, GEM-simplified):
  1. Word/vocab statistics and type-token ratio.
  2. Source overlap vs official simple_wiki: n-gram (5-gram) Jaccard, to test whether
     added SimpleWiki is largely redundant with the official simple_wiki portion.
  3. Unique coverage: fraction of 5-grams in each added stream NOT present in official.
  4. Entity/relation density proxies:
     - capitalized-token fraction (named entities)
     - unique capitalized-type count per 1k words (entity diversity)
     - copula/relation cue frequency ("is a", "was a", "is the", "are", "has", "located",
       "part of", "member of", "known as") per 1k words
     - numeral/date density per 1k words
  5. Stylistic distribution:
     - mean sentence length, question fraction, first/second-person pronoun rate
       (conversational vs encyclopedic signal for Reading/AoA protection)

Also independently recounts total words from the final files.
"""
from __future__ import annotations
import json, pathlib, re, sys, random
from collections import Counter

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
CORPUS_DIR = ROOT / "data/custom_corpus"

RELATION_CUES = [
    "is a", "is an", "is the", "was a", "was an", "was the", "are the", "are a",
    "has", "have", "had", "located", "part of", "member of", "known as",
    "belongs to", "consists of", "made of", "used for", "called",
]
FIRST_SECOND_PRON = {"i", "you", "we", "me", "us", "my", "your", "our", "mine", "yours"}


def independent_word_count(path: pathlib.Path) -> int:
    n = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            n += len(line.split())
    return n


def ngrams(words, n=5):
    return set(tuple(words[i:i+n]) for i in range(len(words) - n + 1))


def profile_text(text: str, sample_words: int = 400_000, seed: int = 218) -> dict:
    words = text.split()
    total = len(words)
    # Sample a contiguous-ish subset for heavy metrics (deterministic)
    rng = random.Random(seed)
    if total > sample_words:
        start = rng.randint(0, total - sample_words)
        sample = words[start:start + sample_words]
    else:
        sample = words
    ns = len(sample)

    # vocab / TTR
    types = Counter(sample)
    ttr = len(types) / max(1, ns)

    # entity proxies
    cap = [w for w in sample if w[:1].isupper() and w[:1].isalpha()]
    cap_frac = len(cap) / max(1, ns)
    cap_types = len(set(cap))
    cap_type_per_1k = 1000 * cap_types / max(1, ns)

    # numerals/dates
    numer = sum(1 for w in sample if re.search(r"\d", w))
    numer_per_1k = 1000 * numer / max(1, ns)

    # relation cues (on lowercased joined text of sample)
    low = " ".join(sample).lower()
    rel_count = 0
    for cue in RELATION_CUES:
        rel_count += low.count(" " + cue + " ")
    rel_per_1k = 1000 * rel_count / max(1, ns)

    # pronoun rate
    pron = sum(1 for w in sample if w.lower().strip(".,!?;:") in FIRST_SECOND_PRON)
    pron_per_1k = 1000 * pron / max(1, ns)

    # sentence stats (approx by splitting on . ! ?)
    sents = re.split(r"[.!?]+", " ".join(sample))
    sents = [s for s in sents if s.strip()]
    mean_sent_len = ns / max(1, len(sents))
    q_frac = ("?" in " ".join(sample)) and (sum(1 for s in re.split(r"([.!?])", " ".join(sample)) if s == "?")) or 0
    q_frac = q_frac / max(1, len(sents))

    return {
        "total_words": total,
        "sample_words": ns,
        "type_token_ratio_sample": round(ttr, 5),
        "cap_frac": round(cap_frac, 4),
        "cap_type_per_1k": round(cap_type_per_1k, 3),
        "numeral_per_1k": round(numer_per_1k, 3),
        "relation_cue_per_1k": round(rel_per_1k, 3),
        "first_second_pron_per_1k": round(pron_per_1k, 3),
        "mean_sentence_len": round(mean_sent_len, 2),
        "question_frac": round(q_frac, 4),
        "_sample_words_list": sample,  # kept for overlap computations, removed before save
    }


def main():
    import argparse
    import argparse as ap
    sys.path.insert(0, str(ROOT / "training/scripts"))
    from babylm_masked_train_leadershape import download_dataset, TRAIN_FILES

    out = {"status": "CORPUS_CHARACTERIZATION"}

    # Independent recount
    off_path = CORPUS_DIR / "official_only_10M.txt"
    mix_path = CORPUS_DIR / "custom_mix_10M.txt"
    out["independent_word_counts"] = {
        "official_only_10M": independent_word_count(off_path),
        "custom_mix_10M": independent_word_count(mix_path),
    }

    # Load manifest to get the split boundaries
    manifest = json.loads((CORPUS_DIR / "corpus_manifest.json").read_text())
    comp = manifest["composition"]
    off_w = comp["official_retained"]["words"]
    wiki_w = comp["simple_wiki_added"]["words"]

    # Re-read the mix and split back into three streams by word offsets
    mix_words = mix_path.read_text(encoding="utf-8").split()
    official_stream = " ".join(mix_words[:off_w])
    wiki_stream = " ".join(mix_words[off_w:off_w + wiki_w])
    gem_stream = " ".join(mix_words[off_w + wiki_w:])

    # Also load the OFFICIAL simple_wiki portion separately to test redundancy
    ns2 = ap.Namespace(dataset_id="BabyLM-community/BabyLM-2026-Strict-Small",
                       dataset_revision="c92ab16b4f08858304b0815706065b3354d8fc0a")
    raw_dir, _ = download_dataset(ns2, CORPUS_DIR / "official_raw")
    official_simplewiki_text = (raw_dir / "simple_wiki.train.txt").read_text(encoding="utf-8")

    # Profiles
    prof = {}
    prof["official_retained"] = profile_text(official_stream)
    prof["simple_wiki_added"] = profile_text(wiki_stream)
    prof["gem_simplified_added"] = profile_text(gem_stream)
    prof["official_simplewiki_reference"] = profile_text(official_simplewiki_text)

    # Overlap: 5-gram Jaccard and unique coverage vs official corpus sample
    off_sample = prof["official_retained"]["_sample_words_list"]
    off_5 = ngrams(off_sample, 5)
    offwiki_5 = ngrams(prof["official_simplewiki_reference"]["_sample_words_list"], 5)

    overlaps = {}
    for key in ["simple_wiki_added", "gem_simplified_added"]:
        s = ngrams(prof[key]["_sample_words_list"], 5)
        jacc_off = len(s & off_5) / max(1, len(s | off_5))
        unique_vs_off = 1 - (len(s & off_5) / max(1, len(s)))
        jacc_offwiki = len(s & offwiki_5) / max(1, len(s | offwiki_5))
        unique_vs_offwiki = 1 - (len(s & offwiki_5) / max(1, len(s)))
        overlaps[key] = {
            "jaccard_5gram_vs_official_mix": round(jacc_off, 5),
            "unique_5gram_frac_vs_official_mix": round(unique_vs_off, 4),
            "jaccard_5gram_vs_official_simplewiki": round(jacc_offwiki, 5),
            "unique_5gram_frac_vs_official_simplewiki": round(unique_vs_offwiki, 4),
        }
    out["overlap_and_unique_coverage"] = overlaps

    # Remove heavy word lists before saving
    for k in prof:
        prof[k].pop("_sample_words_list", None)
    out["stream_profiles"] = prof

    # Simple verdict heuristic
    wiki = overlaps["simple_wiki_added"]
    gem = overlaps["gem_simplified_added"]
    rel_off = prof["official_retained"]["relation_cue_per_1k"]
    rel_wiki = prof["simple_wiki_added"]["relation_cue_per_1k"]
    ent_off = prof["official_retained"]["cap_type_per_1k"]
    ent_wiki = prof["simple_wiki_added"]["cap_type_per_1k"]

    notes = []
    if wiki["unique_5gram_frac_vs_official_simplewiki"] > 0.8:
        notes.append("SimpleWiki-added is largely NON-redundant with official simple_wiki (good unique coverage).")
    else:
        notes.append("SimpleWiki-added overlaps official simple_wiki substantially (redundancy risk).")
    if ent_wiki > 1.3 * ent_off:
        notes.append(f"SimpleWiki raises entity-type density ({ent_wiki:.2f} vs {ent_off:.2f} per 1k).")
    else:
        notes.append(f"SimpleWiki entity-type density NOT clearly higher ({ent_wiki:.2f} vs {ent_off:.2f}).")
    if rel_wiki > 1.3 * rel_off:
        notes.append(f"SimpleWiki raises relation-cue density ({rel_wiki:.2f} vs {rel_off:.2f} per 1k).")
    else:
        notes.append(f"SimpleWiki relation-cue density NOT clearly higher ({rel_wiki:.2f} vs {rel_off:.2f}).")
    out["verdict_notes"] = notes

    p = CORPUS_DIR / "characterization.json"
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "independent_word_counts": out["independent_word_counts"],
        "overlap": overlaps,
        "entity_density": {k: prof[k]["cap_type_per_1k"] for k in prof},
        "relation_density": {k: prof[k]["relation_cue_per_1k"] for k in prof},
        "pron_density": {k: prof[k]["first_second_pron_per_1k"] for k in prof},
        "verdict_notes": notes,
    }, indent=2))
    print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()
