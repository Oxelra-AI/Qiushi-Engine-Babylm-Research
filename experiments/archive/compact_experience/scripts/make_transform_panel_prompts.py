#!/usr/bin/env python3
"""Create a small multi-family second-view transformation prompt panel.

research rebuilds the next route around information-efficient same-window second views.
This script samples one shared, source/length-stratified set of official BabyLM
sentences and writes prompts for several genuinely different transformation families.
The panel is deliberately small: it is for generation-quality and mechanism audit,
not a full training corpus.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = _public_path('experiments/archive/compact_experience')
POOL_PATH = _public_path('experiments/archive/compact_experience/data/mixture/official_pool.jsonl')
OUT_DIR = _public_path('experiments/archive/compact_experience/data/transform_panel')
AI_DATA = _public_path('experiments/archive/compact_experience/training/data')
OUT_DIR.mkdir(parents=True, exist_ok=True)
AI_DATA.mkdir(parents=True, exist_ok=True)

N_ORIGINALS = 3000
RNG_SEED = 61028
N_SHARDS = 2
MIN_WORDS = 12
MAX_WORDS = 55
LENGTH_BINS = [(12, 17), (18, 24), (25, 35), (36, 55)]
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
TERMINAL_OK = re.compile(r"[.!?][\"'”’\)]*$")
BAD_START = re.compile(r"^(?:and|but|or|which|that|whose|whom|soon|of)\b", re.I)

SOURCE_WORD_TARGET = {
    # Approximate official 10M source-word distribution from the recovered pool.
    "childes": 2_841_120,
    "gutenberg": 2_557_760,
    "open_subtitles": 2_282_880,
    "simple_wiki": 1_531_520,
    "bnc_spoken": 762_080,
    "switchboard": 24_640,
}

TRANSFORMS = {
    "near_paraphrase": {
        "target_len": "0.85-1.10x",
        "prompt": (
            "Paraphrase the sentence below with exactly the same meaning, using different wording and structure.\n"
            "Keep every proper name, speaker label, number, date, quantity, quoted word, negation, modality, comparison, and who-did-what-to-whom relation unchanged.\n"
            "Do not add new facts, omit facts, explain, or continue the story. Output only one complete sentence.\n\n"
            "Sentence: {sentence}"
        ),
    },
    "capsule_065": {
        "target_len": "0.60-0.70x",
        "prompt": (
            "Rewrite the sentence as one or two shorter complete natural English sentences. Aim for 60-70% of the original word count.\n"
            "Preserve every core proposition: participants, actions, recipients, locations, time, cause, negation, modality, comparisons, numbers, dates, quantities, and named entities.\n"
            "Delete only redundant modifiers, repeated wording, or non-core detail; never add facts or use outside knowledge. Output only the rewritten sentence(s).\n\n"
            "Sentence: {sentence}"
        ),
    },
    "capsule_050": {
        "target_len": "0.45-0.55x",
        "prompt": (
            "Rewrite the sentence as a compact proposition capsule: one or two short complete natural English sentences, about 45-55% of the original word count.\n"
            "Keep all core events and relations, including who acted on whom, negation, modality, comparisons, numbers, dates, quantities, and named entities.\n"
            "Remove only secondary wording and redundant description. Do not make a title, fragment, list, explanation, inference, or outside-fact summary. Output only the capsule.\n\n"
            "Sentence: {sentence}"
        ),
    },
    "typed_relation_qa": {
        "target_len": "8-18 words",
        "prompt": (
            "Write exactly one short question-answer pair about a relation explicitly stated in the sentence.\n"
            "Prefer a relation involving who did what to whom, time, place, cause, quantity, comparison, possession, event order, or a clear pronoun reference.\n"
            "The answer must be directly recoverable from the sentence; keep names, numbers, dates, quantities, negation, and polarity exactly right.\n"
            "Do not ask trivial first-name/title questions, do not use outside knowledge, and do not add facts.\n"
            "Output exactly: Question? — Answer.\n\n"
            "Sentence: {sentence}"
        ),
    },
}


def norm_ws(text: str) -> str:
    return " ".join((text or "").replace("\u00a0", " ").split())


def split_sentences(text: str) -> list[str]:
    return [norm_ws(x.strip()) for x in SENT_SPLIT.split(text or "") if norm_ws(x.strip())]


def word_count(text: str) -> int:
    return len((text or "").split())


def fingerprint(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def length_bin(n: int) -> str:
    for lo, hi in LENGTH_BINS:
        if lo <= n <= hi:
            return f"{lo}_{hi}"
    return "other"


def mostly_weird(text: str) -> bool:
    letters = sum(ch.isalpha() for ch in text)
    if letters < 20:
        return True
    weird = sum((not ch.isalnum() and not ch.isspace() and ch not in "'\".,!?;:()[]-*%$/£€–—_") for ch in text)
    return weird > max(4, letters // 20)


def candidate_ok(sent: str) -> bool:
    n = word_count(sent)
    if n < MIN_WORDS or n > MAX_WORDS:
        return False
    s = sent.strip()
    low = s.lower()
    if not TERMINAL_OK.search(s):
        return False
    if BAD_START.match(s.lstrip("\"'“”‘’([* ")):
        return False
    if "..." in s or "http" in low or ".cha" in low or "/childes" in low or "= =" in low:
        return False
    if "xxx" in low or low.count(" er ") > 2 or low.count(" erm ") > 1:
        return False
    if s.count(",") > max(6, n // 3):
        return False
    if mostly_weird(s):
        return False
    first = s.lstrip("\"'“”‘’([ ").split()[0].strip(".,!?;:") if s.split() else ""
    if first and first[0].islower() and not first.startswith("*"):
        return False
    return True


def load_candidates() -> list[dict]:
    candidates = []
    seen = set()
    with POOL_PATH.open("r", encoding="utf-8") as f:
        for row_i, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            source = row.get("source", "unknown")
            for sent_idx, sent in enumerate(split_sentences(row.get("text", ""))):
                sent = norm_ws(sent)
                fp = fingerprint(sent)
                if fp in seen:
                    continue
                if not candidate_ok(sent):
                    continue
                seen.add(fp)
                n = word_count(sent)
                candidates.append({
                    "text": sent,
                    "words": n,
                    "source": source,
                    "example_id": int(row.get("example_id", row_i)),
                    "source_row_index": row_i,
                    "sentence_idx": sent_idx,
                    "length_bin": length_bin(n),
                })
    return candidates


def target_source_counts(n: int, sources: list[str]) -> dict[str, int]:
    total = sum(SOURCE_WORD_TARGET.get(s, 0) for s in sources)
    raw = {s: n * SOURCE_WORD_TARGET.get(s, 0) / total for s in sources}
    out = {s: int(math.floor(raw[s])) for s in sources}
    rem = n - sum(out.values())
    for s in sorted(sources, key=lambda x: raw[x] - math.floor(raw[x]), reverse=True):
        if rem <= 0:
            break
        out[s] += 1
        rem -= 1
    return out


def stratified_sample(candidates: list[dict]) -> list[dict]:
    rng = random.Random(RNG_SEED)
    by_source_bin: dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_source: dict[str, list[dict]] = defaultdict(list)
    for c in candidates:
        by_source_bin[(c["source"], c["length_bin"])].append(c)
        by_source[c["source"]].append(c)
    for bucket in by_source_bin.values():
        rng.shuffle(bucket)
    for bucket in by_source.values():
        rng.shuffle(bucket)

    sources = sorted(by_source.keys())
    source_targets = target_source_counts(N_ORIGINALS, sources)
    selected = []
    selected_fp = set()
    for source in sources:
        source_pool = by_source[source]
        target = min(source_targets[source], len(source_pool))
        # Length bins are balanced within each source where possible, but not forced to shortest-first.
        bin_counts = Counter(c["length_bin"] for c in source_pool)
        bin_raw = {b: target * bin_counts[b] / len(source_pool) for b in bin_counts}
        bin_target = {b: int(math.floor(bin_raw[b])) for b in bin_counts}
        rem = target - sum(bin_target.values())
        for b in sorted(bin_counts, key=lambda x: bin_raw[x] - math.floor(bin_raw[x]), reverse=True):
            if rem <= 0:
                break
            bin_target[b] += 1
            rem -= 1
        for b, k in bin_target.items():
            for c in by_source_bin[(source, b)][:k]:
                fp = fingerprint(c["text"])
                if fp not in selected_fp:
                    selected.append(c); selected_fp.add(fp)
    # Fill shortfall without changing random seed.
    if len(selected) < N_ORIGINALS:
        rest = candidates[:]
        rng.shuffle(rest)
        for c in rest:
            if len(selected) >= N_ORIGINALS:
                break
            fp = fingerprint(c["text"])
            if fp not in selected_fp:
                selected.append(c); selected_fp.add(fp)
    rng.shuffle(selected)
    return selected[:N_ORIGINALS]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    candidates = load_candidates()
    selected = stratified_sample(candidates)

    originals_path = _public_path('experiments/archive/compact_experience/data/transform_panel/panel_originals.jsonl')
    prompts_path = _public_path('experiments/archive/compact_experience/data/transform_panel/panel_prompts.jsonl')
    shard_paths = [AI_DATA / f"transform_panel_prompts_shard{i}.jsonl" for i in range(N_SHARDS)]

    prompt_records = []
    with originals_path.open("w", encoding="utf-8") as of:
        for i, c in enumerate(selected):
            oid = f"s061_{i:05d}"
            c = dict(c)
            c["original_id"] = oid
            of.write(json.dumps(c, ensure_ascii=False) + "\n")
            for transform, spec in TRANSFORMS.items():
                prompt_records.append({
                    "id": f"{oid}__{transform}",
                    "original_id": oid,
                    "transform": transform,
                    "target_len": spec["target_len"],
                    "prompt": spec["prompt"].format(sentence=c["text"]),
                    "source_sentence": c["text"],
                    "source_words": c["words"],
                    "source_name": c["source"],
                    "source_example_id": c["example_id"],
                    "source_sentence_idx": c["sentence_idx"],
                    "length_bin": c["length_bin"],
                })

    with prompts_path.open("w", encoding="utf-8") as pf:
        for rec in prompt_records:
            pf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    for shard_i, sp in enumerate(shard_paths):
        with sp.open("w", encoding="utf-8") as sf:
            for j in range(shard_i, len(prompt_records), N_SHARDS):
                sf.write(json.dumps(prompt_records[j], ensure_ascii=False) + "\n")

    source_counts = Counter(c["source"] for c in selected)
    length_counts = Counter(c["length_bin"] for c in selected)
    word_counts = [c["words"] for c in selected]
    meta = {
        "status": "TRANSFORM_PANEL_PROMPTS_CREATED",
        "purpose": "small audit panel for information-efficient same-window second-view transformation families",
        "n_originals": len(selected),
        "n_transforms": len(TRANSFORMS),
        "n_prompts": len(prompt_records),
        "rng_seed": RNG_SEED,
        "min_words": MIN_WORDS,
        "max_words": MAX_WORDS,
        "transforms": {k: {"target_len": v["target_len"]} for k, v in TRANSFORMS.items()},
        "source_counts": dict(source_counts),
        "length_bin_counts": dict(length_counts),
        "word_stats": {
            "min": min(word_counts),
            "mean": round(sum(word_counts) / len(word_counts), 3),
            "max": max(word_counts),
            "total_original_words": sum(word_counts),
        },
        "files": {
            "originals": str(originals_path.relative_to(ROOT)),
            "prompts": str(prompts_path.relative_to(ROOT)),
            "prompt_shards": [str(p.relative_to(ROOT)) for p in shard_paths],
        },
        "sha256": {
            "originals": sha(originals_path),
            "prompts": sha(prompts_path),
            **{p.name: sha(p) for p in shard_paths},
        },
        "note": "This panel does not choose a final training arm. It supplies matched originals for generation-quality, length, fidelity, and relation-type audits before any full corpus materialization.",
    }
    (_public_path('experiments/archive/compact_experience/data/transform_panel/panel_prompt_metadata.json')).write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
