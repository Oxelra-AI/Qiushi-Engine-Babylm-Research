#!/usr/bin/env python3
"""Materialize research procedural entity-state story mixtures.

Purpose: test whether explicit entity-state transition structure transfers to
BabyLM Entity/EWoK without the 50/50 WikiAuto distribution damage.

For each seed, create two matched 1M JSONLs:
- state_coherent_90_10: 900k official BabyLM words + 100k coherent state stories.
- state_corrupted_90_10: identical official examples and same story word/target
  multiset, but the decisive object bindings are swapped so visible events no
  longer support the original final-state target.

The state story rows are encoded as `kind="pair_crossview"` so the existing
isolated `babylm_crossview_mask_train.py` can deterministically mask the recorded
source-side target word groups while official rows use standard WWM. This avoids
modifying the trusted WWM trainer.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import hashlib
import json
import os
import pathlib
import random
from dataclasses import dataclass
from typing import Any, Iterable

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

OFFICIAL_DATASET_ID = "BabyLM-community/BabyLM-2026-Strict-Small"
OFFICIAL_DATASET_REVISION = "c92ab16b4f08858304b0815706065b3354d8fc0a"
TRAIN_FILES = [
    "bnc_spoken.train.txt", "childes.train.txt", "gutenberg.train.txt",
    "open_subtitles.train.txt", "simple_wiki.train.txt", "switchboard.train.txt",
]
BASELINE_TOKENIZER_REPO = "BabyLM-community/BabyLM-2026-Baseline-GPT2-Strict"
ROOT = pathlib.Path("experiments/archive/initial_model_studies")
DEFAULT_OUT_DIR = ROOT / "data/state_revision_68"

NAMES = ["Ava", "Ben", "Cora", "Dima", "Eli", "Faye", "Gwen", "Hugo", "Iris", "Jules", "Kira", "Liam"]
OBJECTS = ["marble", "key", "coin", "shell", "cup", "map", "book", "ring", "card", "spoon", "button", "stone"]
LOCATIONS = ["basket", "drawer", "shelf", "tray", "box", "bowl", "bag", "cabinet", "desk", "closet", "cart", "crate"]
# Disjoint vocabulary for held-out procedural probe (analysis only, not training).
PROBE_NAMES = ["Nora", "Owen", "Pia", "Quinn", "Ravi", "Sana"]
PROBE_OBJECTS = ["badge", "ticket", "brush", "candle", "pencil", "cloth"]
PROBE_LOCATIONS = ["locker", "porch", "pantry", "garage", "garden", "cellar"]
TARGET_WORD_INDEX = 36  # in the 40-word template below
STORY_WORDS = 40


@dataclass
class Example:
    text: str
    words: int
    example_id: int = -1
    source: str = ""


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_words_in_file(path: pathlib.Path) -> int:
    total = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            total += len(line.split())
    return total


def iter_examples(files: list[pathlib.Path], max_words: int, words_per_example: int) -> Iterable[Example]:
    used = 0
    buf: list[str] = []
    buf_source = ""
    for fp in files:
        with fp.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                for w in line.split():
                    if used >= max_words:
                        break
                    if not buf:
                        buf_source = fp.name
                    buf.append(w); used += 1
                    if len(buf) >= words_per_example:
                        yield Example(" ".join(buf), len(buf), source=buf_source)
                        buf = []; buf_source = ""
                if used >= max_words:
                    break
        if used >= max_words:
            break
    if buf:
        yield Example(" ".join(buf), len(buf), source=buf_source)


def download_official(out_dir: pathlib.Path) -> tuple[pathlib.Path, list[dict[str, Any]]]:
    raw_dir = out_dir / "official_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    local = pathlib.Path(snapshot_download(
        repo_id=OFFICIAL_DATASET_ID,
        repo_type="dataset",
        revision=OFFICIAL_DATASET_REVISION,
        allow_patterns=TRAIN_FILES + ["README.md"],
        local_dir=raw_dir,
        local_dir_use_symlinks=False,
    ))
    manifest = []
    for name in TRAIN_FILES:
        p = local / name
        manifest.append({"name": name, "path": str(p), "bytes": p.stat().st_size,
                         "sha256": sha256_file(p), "whitespace_words": count_words_in_file(p)})
    return local, manifest


def select_official_examples(files: list[pathlib.Path], seed: int, pool_words: int, target_words: int, words_per_example: int) -> list[Example]:
    pool = list(iter_examples(files, pool_words, words_per_example))
    for i, ex in enumerate(pool):
        ex.example_id = i
    if sum(ex.words for ex in pool) != pool_words:
        raise RuntimeError("official pool did not reach requested word count")
    rng = random.Random(seed); rng.shuffle(pool)
    selected, total = [], 0
    for ex in pool:
        if total + ex.words <= target_words:
            selected.append(ex); total += ex.words
            if total == target_words:
                break
    if total != target_words:
        raise RuntimeError(f"official selected {total} words, target {target_words}")
    return selected


def story_pair(story_id: int, rng: random.Random, names=NAMES, objects=OBJECTS, locations=LOCATIONS) -> tuple[dict[str, Any], dict[str, Any]]:
    p1, p2 = rng.sample(names, 2)
    obj1, obj2 = rng.sample(objects, 2)
    loc1, loc2, loc3, loc4 = rng.sample(locations, 4)
    # 40 words exactly. Target state word loc3 is final word index 36.
    coherent = (
        f"{p1} put the {obj1} in the {loc1}. "
        f"{p2} put the {obj2} in the {loc2}. "
        f"Later {p1} moved the {obj1} to the {loc3}. "
        f"Meanwhile {p2} moved the {obj2} to the {loc4}. "
        f"The {obj1} is now in the {loc3} near the door."
    )
    # Same word multiset and final target, but the decisive second-step bindings are swapped.
    corrupted = (
        f"{p1} put the {obj1} in the {loc1}. "
        f"{p2} put the {obj2} in the {loc2}. "
        f"Later {p1} moved the {obj2} to the {loc3}. "
        f"Meanwhile {p2} moved the {obj1} to the {loc4}. "
        f"The {obj1} is now in the {loc3} near the door."
    )
    for txt in (coherent, corrupted):
        if len(txt.split()) != STORY_WORDS:
            raise RuntimeError(f"story word count {len(txt.split())}: {txt}")
        if txt.split()[TARGET_WORD_INDEX].rstrip(".") != loc3:
            raise RuntimeError("target index mismatch")
    base = {
        "kind": "pair_crossview",
        "words": STORY_WORDS,
        "source_text": None,  # filled below with the full story text
        "target_text": "",
        "source_pair_id": story_id,
        "target_pair_id": story_id,
        "true_target_pair_id": story_id,
        "mask_side": "source",
        "anchor_source_word_indices": [TARGET_WORD_INDEX],
        "anchor_norms": [loc3],
        "anchor_count": 1,
        "state_story_id": story_id,
        "state_target_word_index": TARGET_WORD_INDEX,
        "state_target_value": loc3,
        "state_actual_value_coherent": loc3,
        "state_actual_value_corrupted": loc4,
        "state_object": obj1,
        "distractor_object": obj2,
        "template_family": "two_object_location_overwrite_v1",
    }
    c = dict(base); c.update({"text": coherent, "source_text": coherent, "source": "procedural_state_coherent", "mode": "state_coherent"})
    k = dict(base); k.update({"text": corrupted, "source_text": corrupted, "source": "procedural_state_corrupted", "mode": "state_corrupted"})
    return c, k


def token_ids(tok, text: str) -> list[int]:
    return tok(text, add_special_tokens=False, truncation=False)["input_ids"]


def target_token_count(tok, text: str, idx: int) -> int:
    words = text.split(); piece = words[idx] if idx == 0 else " " + words[idx]
    return len(token_ids(tok, piece))


def write_jsonl(path: pathlib.Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def summarize(records: list[dict[str, Any]], tok, max_seq_length: int) -> dict[str, Any]:
    total_words = sum(int(r["words"]) for r in records)
    official = [r for r in records if r["kind"] == "official"]
    state = [r for r in records if r["kind"] == "pair_crossview"]
    kept = 0; untr = 0; lost = 0; over = 0; target_tok = 0
    src_words: dict[str, int] = {}
    target_values = []
    for r in records:
        ids = token_ids(tok, r["text"])
        untr += len(ids); kept += min(len(ids), max_seq_length)
        if len(ids) > max_seq_length:
            over += 1; lost += len(ids) - max_seq_length
        src_words[r["source"]] = src_words.get(r["source"], 0) + int(r["words"])
        if r["kind"] == "pair_crossview":
            target_tok += target_token_count(tok, r["text"], int(r["state_target_word_index"]))
            target_values.append(r["state_target_value"])
    return {
        "num_examples": len(records), "total_words": total_words,
        "official_examples": len(official), "state_examples": len(state),
        "total_untruncated_tokens": untr, "total_kept_tokens_at_max_seq_length": kept,
        "tokens_lost_to_truncation": lost, "truncated_examples_at_max_seq_length": over,
        "state_target_groups_total": len(state), "state_target_token_proxy_total": target_tok,
        "state_target_values_histogram": dict(sorted({v: target_values.count(v) for v in set(target_values)}.items())),
        "source_words": src_words,
    }


def materialize_seed(seed: int, args, official_files: list[pathlib.Path], tok) -> dict[str, Any]:
    official = select_official_examples(official_files, seed, args.official_pool_words, args.official_words, args.words_per_official_example)
    n_stories = args.state_words // STORY_WORDS
    if n_stories * STORY_WORDS != args.state_words:
        raise RuntimeError("state_words must be divisible by 40")
    rng = random.Random(args.story_seed_base + seed)
    coherent_stories, corrupted_stories = [], []
    for i in range(n_stories):
        c, k = story_pair(seed * 1_000_000 + i, rng)
        coherent_stories.append(c); corrupted_stories.append(k)
    official_records = []
    for j, ex in enumerate(official):
        official_records.append({"kind": "official", "slot_key": f"official:{j}", "source": f"official::{ex.source}",
                                 "text": ex.text, "words": ex.words, "official_example_id": ex.example_id,
                                 "official_source_file": ex.source, "mode": "official"})
    slots = [("official", i) for i in range(len(official_records))] + [("state", i) for i in range(n_stories)]
    random.Random(args.slot_seed_base + seed).shuffle(slots)
    coh_records, cor_records = [], []
    for out_i, (kind, i) in enumerate(slots):
        if kind == "official":
            rc = dict(official_records[i]); rk = dict(official_records[i])
        else:
            rc = dict(coherent_stories[i]); rk = dict(corrupted_stories[i])
        for r in (rc, rk):
            r["example_id"] = out_i; r["mixture_seed"] = seed
        coh_records.append(rc); cor_records.append(rk)
    coh_sum = summarize(coh_records, tok, args.max_seq_length)
    cor_sum = summarize(cor_records, tok, args.max_seq_length)
    validation = {
        "identical_example_count": len(coh_records) == len(cor_records),
        "identical_total_words": coh_sum["total_words"] == cor_sum["total_words"],
        "identical_official_records": [(r["example_id"], r["text"], r["words"], r["official_example_id"]) for r in coh_records if r["kind"] == "official"] == [(r["example_id"], r["text"], r["words"], r["official_example_id"]) for r in cor_records if r["kind"] == "official"],
        "identical_state_target_values": coh_sum["state_target_values_histogram"] == cor_sum["state_target_values_histogram"],
        "identical_state_target_groups": coh_sum["state_target_groups_total"] == cor_sum["state_target_groups_total"],
        "identical_state_target_token_proxy": coh_sum["state_target_token_proxy_total"] == cor_sum["state_target_token_proxy_total"],
        "combined_kept_token_delta_total": cor_sum["total_kept_tokens_at_max_seq_length"] - coh_sum["total_kept_tokens_at_max_seq_length"],
        "combined_truncation_delta": cor_sum["truncated_examples_at_max_seq_length"] - coh_sum["truncated_examples_at_max_seq_length"],
    }
    validation["hard_validation_ok"] = all(bool(validation[k]) for k in ["identical_example_count", "identical_total_words", "identical_official_records", "identical_state_target_values", "identical_state_target_groups", "identical_state_target_token_proxy"])
    label = f"seed{seed}_official{args.official_words}_state{args.state_words}_total{coh_sum['total_words']}_n{len(coh_records)}"
    coh_path = pathlib.Path(args.out_dir) / f"state_coherent_90_10_{label}.jsonl"
    cor_path = pathlib.Path(args.out_dir) / f"state_corrupted_90_10_{label}.jsonl"
    write_jsonl(coh_path, coh_records); write_jsonl(cor_path, cor_records)
    # Smoke subset: exact 10k = 56 official examples (8960 words) + 26 state stories (1040 words)
    smoke_off_n, smoke_state_n = 56, 26
    smoke_slots = [("official", i) for i in range(smoke_off_n)] + [("state", i) for i in range(smoke_state_n)]
    random.Random(args.slot_seed_base + seed + 999).shuffle(smoke_slots)
    smoke_coh, smoke_cor = [], []
    for out_i, (kind, i) in enumerate(smoke_slots):
        if kind == "official":
            rc = dict(official_records[i]); rk = dict(official_records[i])
        else:
            rc = dict(coherent_stories[i]); rk = dict(corrupted_stories[i])
        for r in (rc, rk):
            r["example_id"] = out_i; r["mixture_seed"] = seed; r["smoke_subset"] = True
        smoke_coh.append(rc); smoke_cor.append(rk)
    smoke_coh_path = pathlib.Path(args.out_dir) / f"state_smoke_coherent_seed{seed}_10000w.jsonl"
    smoke_cor_path = pathlib.Path(args.out_dir) / f"state_smoke_corrupted_seed{seed}_10000w.jsonl"
    write_jsonl(smoke_coh_path, smoke_coh); write_jsonl(smoke_cor_path, smoke_cor)
    return {
        "seed": seed, "official_words": args.official_words, "official_examples": len(official_records),
        "state_words": args.state_words, "state_examples": n_stories, "actual_total_words": coh_sum["total_words"],
        "combined_num_examples": len(coh_records),
        "coherent_path": str(coh_path), "corrupted_path": str(cor_path),
        "coherent_sha256": sha256_file(coh_path), "corrupted_sha256": sha256_file(cor_path),
        "smoke_coherent_path": str(smoke_coh_path), "smoke_corrupted_path": str(smoke_cor_path),
        "smoke_words": sum(r["words"] for r in smoke_coh), "smoke_examples": len(smoke_coh),
        "coherent_summary": coh_sum, "corrupted_summary": cor_sum, "validation": validation,
        "recommended_batch": {"batch_size": 83, "actual_steps": (len(coh_records) + 82) // 83, "target_steps": 98},
    }


def make_heldout_probe(out_dir: pathlib.Path, tok, n: int = 240) -> dict[str, Any]:
    rng = random.Random(680000)
    rows = []
    for i in range(n):
        c, k = story_pair(9_000_000 + i, rng, PROBE_NAMES, PROBE_OBJECTS, PROBE_LOCATIONS)
        for mode, r in [("coherent", c), ("corrupted", k)]:
            rows.append({"probe_id": f"{i}_{mode}", "mode": mode, "text": r["text"], "answer": r["state_target_value"],
                         "target_word_index": TARGET_WORD_INDEX, "object": r["state_object"],
                         "actual_value_corrupted": r["state_actual_value_corrupted"],
                         "tokens": len(token_ids(tok, r["text"]))})
    p = out_dir / "state_heldout_probe_disjoint_vocab.jsonl"
    write_jsonl(p, rows)
    return {"path": str(p), "num_rows": len(rows), "sha256": sha256_file(p)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--official_words", type=int, default=900000)
    p.add_argument("--state_words", type=int, default=100000)
    p.add_argument("--official_pool_words", type=int, default=10000000)
    p.add_argument("--words_per_official_example", type=int, default=160)
    p.add_argument("--max_seq_length", type=int, default=256)
    p.add_argument("--seeds", default="42,43")
    p.add_argument("--story_seed_base", type=int, default=680000)
    p.add_argument("--slot_seed_base", type=int, default=681000)
    p.add_argument("--hf_cache_dir", default="experiments/archive/initial_model_studies/training/hf_home")
    args = p.parse_args()
    out_dir = pathlib.Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(pathlib.Path(args.hf_cache_dir).resolve()))
    os.environ.setdefault("HF_HUB_CACHE", str((pathlib.Path(args.hf_cache_dir) / "hub").resolve()))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    tok = AutoTokenizer.from_pretrained(BASELINE_TOKENIZER_REPO, revision="main", use_fast=True)
    official_dir, official_manifest = download_official(out_dir)
    official_files = [official_dir / n for n in TRAIN_FILES]
    all_meta = {
        "official_dataset_id": OFFICIAL_DATASET_ID, "official_dataset_revision": OFFICIAL_DATASET_REVISION,
        "official_manifest": official_manifest,
        "procedural_generator": _public_path('experiments/archive/initial_model_studies/scripts/materialize_state_stories.py').name,
        "generator_description": "hand-coded 40-word two-object location overwrite stories; no external LM; generated words count toward budget",
        "story_words_each": STORY_WORDS, "target_word_index": TARGET_WORD_INDEX,
        "train_vocabulary": {"names": NAMES, "objects": OBJECTS, "locations": LOCATIONS},
        "probe_vocabulary": {"names": PROBE_NAMES, "objects": PROBE_OBJECTS, "locations": PROBE_LOCATIONS},
        "baseline_tokenizer_repo": BASELINE_TOKENIZER_REPO,
        "seeds": {},
    }
    for seed in [int(s) for s in args.seeds.split(",") if s.strip()]:
        sm = materialize_seed(seed, args, official_files, tok)
        all_meta["seeds"][str(seed)] = sm
    all_meta["heldout_probe"] = make_heldout_probe(out_dir, tok)
    meta_path = out_dir / "state_materialization_all_seeds.json"
    meta_path.write_text(json.dumps(all_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "STATE_STORY_JSONL_READY", "meta_path": str(meta_path),
                      "seeds": {s: {"coherent": m["coherent_path"], "corrupted": m["corrupted_path"],
                                     "total_words": m["actual_total_words"], "n": m["combined_num_examples"],
                                     "validation": m["validation"], "smoke_words": m["smoke_words"],
                                     "recommended_batch": m["recommended_batch"]}
                                for s, m in all_meta["seeds"].items()}}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
