#!/usr/bin/env python3
"""Create additional high-quality Qwen rewrite prompts for the clean paired corpus.

The first research prompt set was too small for a 25% complete-pair corpus and
included many fragments.  This script samples a larger, cleaner set of complete
sentences from the official BabyLM pool, avoiding sentences already sent to Qwen,
and shards the prompts so two H100s can generate in parallel.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import hashlib
import json
import pathlib
import random
import re
from dataclasses import dataclass

ROOT = _public_path('experiments/archive/compact_experience')
POOL_PATH = _public_path('experiments/archive/compact_experience/data/mixture/official_pool.jsonl')
OLD_SOURCES = _public_path('experiments/archive/compact_experience/data/qwen_aligned/source_sentences.jsonl')
OUT_DIR = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned')
OUT_DIR.mkdir(parents=True, exist_ok=True)
AI_DATA = _public_path('experiments/archive/compact_experience/training/data')
AI_DATA.mkdir(parents=True, exist_ok=True)

TARGET_TOTAL = 70000
N_SHARDS = 2
MIN_WORDS = 18
MAX_WORDS = 55
RNG_SEED = 28100

PROMPT_TEMPLATE = (
    "Paraphrase the sentence below with exactly the same meaning, using a different wording and structure.\n"
    "Keep every proper name, speaker label, number, date, quantity, and quoted word exactly unchanged.\n"
    "Do not add new facts, omit facts, explain, or continue the story. Output only one complete sentence.\n\n"
    "Sentence: {sentence}"
)

BAD_START = re.compile(r"^(?:and|but|or|because|which|that|while|when|if|though|although|unless|until|whereas|whose|whom|soon|of)\b", re.I)
TERMINAL_OK = re.compile(r"[.!?][\"'”’\)]*$")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

@dataclass
class Sent:
    text: str
    words: int
    source: str
    example_id: int
    sent_idx: int


def norm_text(t: str) -> str:
    return " ".join(t.replace("\u00a0", " ").split())


def split_sentences(text: str) -> list[str]:
    parts = SENT_SPLIT.split(text)
    out = []
    for p in parts:
        p = norm_text(p.strip())
        if p:
            out.append(p)
    return out


def fingerprint(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def mostly_weird(text: str) -> bool:
    letters = sum(ch.isalpha() for ch in text)
    if letters < 20:
        return True
    weird = sum((not ch.isalnum() and not ch.isspace() and ch not in "'\".,!?;:()[]-*%$/£€–—") for ch in text)
    return weird > max(4, letters // 20)


def is_complete_clean(sent: str, n: int) -> bool:
    if n < MIN_WORDS or n > MAX_WORDS:
        return False
    s = sent.strip()
    low = s.lower()
    if not TERMINAL_OK.search(s):
        return False
    if BAD_START.match(s.lstrip("\"'“”‘’([* ")):
        return False
    if s.count("...") > 0 or low.count(" er ") > 2 or low.count(" erm ") > 1:
        return False
    if "= =" in s or "http" in low or ".cha" in low or "/CHILDES" in s:
        return False
    if s.count("[") > 2 or s.count("]") > 2:
        return False
    if s.count(",") > max(5, n // 4):
        return False
    if mostly_weird(s):
        return False
    # Avoid very dialogue-fragment-like utterances; keep speaker-label childes lines if otherwise clean.
    first_word = s.lstrip("\"'“”‘’([ ").split()[0].strip(".,!?;:") if s.split() else ""
    if first_word and first_word[0].islower() and not first_word.startswith("*"):
        return False
    return True


def main() -> None:
    old_fp = set()
    if OLD_SOURCES.exists():
        with OLD_SOURCES.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    old_fp.add(fingerprint(json.loads(line)["text"]))

    candidates: list[Sent] = []
    source_rows = {}
    with POOL_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            source = row["source"]
            source_rows[source] = source_rows.get(source, 0) + 1
            for i, s in enumerate(split_sentences(row["text"])):
                n = len(s.split())
                fp = fingerprint(s)
                if fp in old_fp:
                    continue
                if is_complete_clean(s, n):
                    candidates.append(Sent(s, n, source, int(row["example_id"]), i))

    # Prefer sources whose research preliminary clean set looked reliable, but keep all sources present.
    # This is paired-signal material; the remaining 75% official filler preserves the full official mixture.
    weights = {
        "simple_wiki": 1.35,
        "gutenberg": 1.10,
        "bnc_spoken": 1.00,
        "open_subtitles": 0.85,
        "childes": 0.45,
        "switchboard": 0.50,
    }
    rng = random.Random(RNG_SEED)
    rng.shuffle(candidates)
    buckets: dict[str, list[Sent]] = {}
    for c in candidates:
        buckets.setdefault(c.source, []).append(c)

    weighted_total = sum(source_rows.get(src, 0) * weights.get(src, 1.0) for src in source_rows)
    target_per_source = {src: int(TARGET_TOTAL * source_rows[src] * weights.get(src, 1.0) / weighted_total) for src in source_rows}
    remainder = TARGET_TOTAL - sum(target_per_source.values())
    for src in sorted(source_rows, key=lambda s: source_rows[s] * weights.get(s, 1.0), reverse=True):
        if remainder <= 0:
            break
        target_per_source[src] += 1
        remainder -= 1

    selected: list[Sent] = []
    taken = {src: 0 for src in source_rows}
    for src, target in target_per_source.items():
        pool = buckets.get(src, [])
        take = min(target, len(pool))
        selected.extend(pool[:take])
        taken[src] = take
    # Fill any deficit with remaining candidates from all buckets.
    if len(selected) < TARGET_TOTAL:
        selected_fp = {fingerprint(s.text) for s in selected}
        rest = [c for c in candidates if fingerprint(c.text) not in selected_fp]
        rng.shuffle(rest)
        for c in rest[:TARGET_TOTAL - len(selected)]:
            selected.append(c)
            taken[c.source] = taken.get(c.source, 0) + 1
    rng.shuffle(selected)
    selected = selected[:TARGET_TOTAL]

    sources_path = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/extra_source_sentences.jsonl')
    prompts_path = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/extra_rewrite_prompts.jsonl')
    with sources_path.open("w", encoding="utf-8") as sf, prompts_path.open("w", encoding="utf-8") as pf:
        for i, s in enumerate(selected):
            pid = f"rw2_{i:06d}"
            sf.write(json.dumps({
                "id": pid, "text": s.text, "words": s.words, "source": s.source,
                "example_id": s.example_id, "sentence_idx": s.sent_idx,
            }, ensure_ascii=False) + "\n")
            pf.write(json.dumps({
                "id": pid, "prompt": PROMPT_TEMPLATE.format(sentence=s.text),
                "source_sentence": s.text, "source_words": s.words, "source_name": s.source,
                "source_example_id": s.example_id, "source_sentence_idx": s.sent_idx,
            }, ensure_ascii=False) + "\n")

    # Shard for two parallel generation jobs, preserving local index order for later merging.
    shard_paths = []
    shard_source_paths = []
    for shard in range(N_SHARDS):
        prompt_shard = AI_DATA / f"extra_rewrite_prompts_shard{shard}.jsonl"
        source_shard = OUT_DIR / f"extra_source_sentences_shard{shard}.jsonl"
        shard_paths.append(prompt_shard)
        shard_source_paths.append(source_shard)
        with prompt_shard.open("w", encoding="utf-8") as pf, source_shard.open("w", encoding="utf-8") as sf:
            for local_i, global_i in enumerate(range(shard, len(selected), N_SHARDS)):
                s = selected[global_i]
                pid = f"rw2s{shard}_{local_i:06d}"
                rec_src = {
                    "id": pid, "text": s.text, "words": s.words, "source": s.source,
                    "example_id": s.example_id, "sentence_idx": s.sent_idx,
                }
                sf.write(json.dumps(rec_src, ensure_ascii=False) + "\n")
                pf.write(json.dumps({
                    "id": pid, "prompt": PROMPT_TEMPLATE.format(sentence=s.text),
                    "source_sentence": s.text, "source_words": s.words, "source_name": s.source,
                    "source_example_id": s.example_id, "source_sentence_idx": s.sent_idx,
                }, ensure_ascii=False) + "\n")

    def sha(path: pathlib.Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            h.update(f.read())
        return h.hexdigest()

    meta = {
        "status": "EXTRA_PROMPTS_CREATED",
        "old_source_fingerprints_excluded": len(old_fp),
        "candidate_sentences": len(candidates),
        "selected": len(selected),
        "target": TARGET_TOTAL,
        "min_words": MIN_WORDS,
        "max_words": MAX_WORDS,
        "source_rows": source_rows,
        "source_weights": weights,
        "selected_per_source": taken,
        "word_stats": {
            "min": min(s.words for s in selected),
            "max": max(s.words for s in selected),
            "mean": round(sum(s.words for s in selected) / len(selected), 3),
            "total": sum(s.words for s in selected),
        },
        "files": {
            "sources": str(sources_path),
            "prompts": str(prompts_path),
            "prompt_shards": [str(p) for p in shard_paths],
            "source_shards": [str(p) for p in shard_source_paths],
        },
        "sha256": {"prompts": sha(prompts_path), **{p.name: sha(p) for p in shard_paths}},
    }
    (_public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/extra_prompt_metadata.json')).write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
