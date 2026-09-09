#!/usr/bin/env python3
"""research: Official + Paired-Aligned Mixture Materializer.

Creates mixture training pools combining official BabyLM 10M corpus with
paired-aligned entity-coherent data at different ratios, plus exact 10-pass
training files.

Scientific rationale:
- 100% official: strong BLiMP/Supplement/COMPS/Reading, weak Entity (21.78)
- 100% aligned: strong Entity (29.16), weaker BLiMP/Supplement/COMPS
- Mixture hypothesis: there exists a sweet spot where Entity improves enough
  to increase Overall without proportional BLiMP/Supplement loss

Arms:
- mix_25pct: 75% official + 25% aligned = 10M words
- mix_50pct: 50% official + 50% aligned = 10M words  
- mix_75pct: 25% official + 75% aligned = 10M words

Each arm has exactly 62,500 examples × 160 words = 10,000,000 words.
Training = exactly 10 passes = 100,000,000 word exposure.
Legal under BabyLM Strict-Small: ≤10M words, ≤10 epochs.
"""
from __future__ import annotations
import json
import hashlib
import random
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List

# Paths
ROOT = Path("experiments/archive/compact_experience")
RAW_DIR = Path("experiments/archive/initial_model_studies/training/runs"
               "fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/raw_dataset")
ALIGNED_POOL = ROOT / "data/paired_alignment/aligned_pool.jsonl"
OUT_DIR = ROOT / "data/mixture"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILES = [
    "bnc_spoken.train.txt",
    "childes.train.txt",
    "gutenberg.train.txt",
    "open_subtitles.train.txt",
    "simple_wiki.train.txt",
    "switchboard.train.txt",
]

WORDS_PER_EXAMPLE = 160
POOL_WORDS = 10_000_000
SEED = 43

# Mixture configurations
MIXTURES = {
    "mix_25pct": {"official_frac": 0.75, "aligned_frac": 0.25},
    "mix_50pct": {"official_frac": 0.50, "aligned_frac": 0.50},
    "mix_75pct": {"official_frac": 0.25, "aligned_frac": 0.75},
}


@dataclass
class Example:
    text: str
    words: int
    example_id: int = -1
    source: str = ""


def materialize_official_pool() -> List[dict]:
    """Chunk raw text files into 62,500 × 160-word examples (same as INITIAL_MODEL_STUDIES trainer)."""
    pool_path = OUT_DIR / "official_pool.jsonl"
    if pool_path.exists():
        print(f"  Official pool already exists: {pool_path}")
        rows = []
        with pool_path.open() as f:
            for line in f:
                rows.append(json.loads(line))
        print(f"  Loaded {len(rows)} rows, {sum(r['words'] for r in rows)} words")
        return rows

    print("  Materializing official pool from raw text files...")
    files = [RAW_DIR / n for n in TRAIN_FILES]
    for f in files:
        if not f.exists():
            raise FileNotFoundError(f"Missing: {f}")

    examples = []
    used = 0
    buf: List[str] = []
    buf_source = ""

    for fp in files:
        with fp.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                for w in line.split():
                    if used >= POOL_WORDS:
                        break
                    if not buf:
                        buf_source = fp.stem.replace(".train", "")
                    buf.append(w)
                    if len(buf) == WORDS_PER_EXAMPLE:
                        examples.append({
                            "text": " ".join(buf),
                            "words": len(buf),
                            "example_id": len(examples),
                            "source": buf_source,
                        })
                        used += len(buf)
                        buf = []
                if used >= POOL_WORDS:
                    break
        if used >= POOL_WORDS:
            break

    print(f"  Official pool: {len(examples)} examples, {used} words")
    assert used == POOL_WORDS, f"Expected {POOL_WORDS}, got {used}"
    assert len(examples) == POOL_WORDS // WORDS_PER_EXAMPLE

    # Save
    with pool_path.open("w") as f:
        for row in examples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  Saved: {pool_path}")
    return examples


def load_aligned_pool() -> List[dict]:
    """Load the research ALIGNED pool."""
    rows = []
    with ALIGNED_POOL.open() as f:
        for line in f:
            rows.append(json.loads(line))
    print(f"  Aligned pool: {len(rows)} rows, {sum(r['words'] for r in rows)} words")
    return rows


def create_mixture_pool(
    official: List[dict],
    aligned: List[dict],
    n_official: int,
    n_aligned: int,
    mix_name: str,
    seed: int,
) -> Path:
    """Create a mixture pool JSONL by sampling from official and aligned pools."""
    pool_path = OUT_DIR / f"{mix_name}_pool.jsonl"
    
    rng = random.Random(seed)
    
    # Sample without replacement
    off_indices = rng.sample(range(len(official)), n_official)
    ali_indices = rng.sample(range(len(aligned)), n_aligned)
    
    # Build mixture pool
    pool = []
    for idx in off_indices:
        row = dict(official[idx])
        row["source"] = f"official::{row['source']}"
        row["example_id"] = len(pool)
        pool.append(row)
    
    for idx in ali_indices:
        row = dict(aligned[idx])
        row["source"] = "aligned_pair"
        row["example_id"] = len(pool)
        pool.append(row)
    
    # Shuffle the pool (so official and aligned are interleaved)
    rng.shuffle(pool)
    
    # Re-assign example_ids after shuffle
    for i, row in enumerate(pool):
        row["example_id"] = i
    
    total_words = sum(r["words"] for r in pool)
    print(f"  {mix_name}: {len(pool)} examples, {total_words} words "
          f"({n_official} official + {n_aligned} aligned)")
    
    with pool_path.open("w") as f:
        for row in pool:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    
    return pool_path


def expand_to_training(pool_path: Path, mix_name: str, seed: int) -> Path:
    """Expand pool to exact 10-pass training file."""
    train_dir = OUT_DIR / "training_files"
    train_dir.mkdir(parents=True, exist_ok=True)
    train_path = train_dir / f"{mix_name}_100M.jsonl"
    
    # Load pool
    pool = []
    with pool_path.open() as f:
        for line in f:
            pool.append(json.loads(line))
    
    pool_words = sum(r["words"] for r in pool)
    target_exposure = pool_words * 10  # Exactly 10 passes
    n_epochs = 10
    
    print(f"  Expanding {mix_name}: {pool_words} pool → {target_exposure} exposure ({n_epochs} passes)")
    
    with train_path.open("w") as f:
        total_written = 0
        for epoch in range(n_epochs):
            epoch_pool = list(pool)
            shuffle_seed = seed + 1000003 * epoch
            random.Random(shuffle_seed).shuffle(epoch_pool)
            for row in epoch_pool:
                row_out = dict(row)
                row_out["source"] = f"epoch{epoch+1}::{row['source']}"
                f.write(json.dumps(row_out, ensure_ascii=False) + "\n")
                total_written += row["words"]
    
    print(f"  Written: {train_path} ({total_written} words, {n_epochs} epochs)")
    assert total_written == target_exposure, f"Expected {target_exposure}, got {total_written}"
    return train_path


def sha256_prefix(path: Path, n=16) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def main():
    start = time.time()
    print("=" * 60)
    print("research: Official + Aligned Mixture Materializer")
    print("=" * 60)
    
    # 1. Materialize official pool
    print("\n1. Official pool:")
    official = materialize_official_pool()
    
    # 2. Load aligned pool
    print("\n2. Aligned pool:")
    aligned = load_aligned_pool()
    
    # 3. Create mixture pools
    print("\n3. Creating mixture pools:")
    mixture_info = {}
    
    for mix_name, cfg in MIXTURES.items():
        n_official = int(cfg["official_frac"] * (POOL_WORDS // WORDS_PER_EXAMPLE))
        n_aligned = (POOL_WORDS // WORDS_PER_EXAMPLE) - n_official
        
        # Verify word counts
        assert n_official * WORDS_PER_EXAMPLE + n_aligned * WORDS_PER_EXAMPLE == POOL_WORDS
        assert n_aligned <= len(aligned), f"Need {n_aligned} aligned but only have {len(aligned)}"
        assert n_official <= len(official), f"Need {n_official} official but only have {len(official)}"
        
        pool_path = create_mixture_pool(
            official, aligned, n_official, n_aligned, mix_name, SEED
        )
        
        mixture_info[mix_name] = {
            "n_official": n_official,
            "n_aligned": n_aligned,
            "official_words": n_official * WORDS_PER_EXAMPLE,
            "aligned_words": n_aligned * WORDS_PER_EXAMPLE,
            "total_words": POOL_WORDS,
            "pool_path": str(pool_path),
            "pool_sha256_prefix": sha256_prefix(pool_path),
        }
    
    # 4. Expand to training files
    print("\n4. Expanding to 10-pass training files:")
    for mix_name in MIXTURES:
        pool_path = OUT_DIR / f"{mix_name}_pool.jsonl"
        train_path = expand_to_training(pool_path, mix_name, SEED)
        mixture_info[mix_name]["training_path"] = str(train_path)
        mixture_info[mix_name]["training_sha256_prefix"] = sha256_prefix(train_path)
        mixture_info[mix_name]["training_exposure"] = POOL_WORDS * 10
        mixture_info[mix_name]["training_epochs"] = 10
    
    # 5. Generate training commands
    print("\n5. Training commands:")
    trainer = "experiments/archive/initial_model_studies/training/scripts/babylm_masked_train_fullcycle.py"
    tokenizer = ("experiments/archive/initial_model_studies/training/runs"
                 "fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model")
    
    commands = {}
    for mix_name in MIXTURES:
        train_path = mixture_info[mix_name]["training_path"]
        out_dir = f"experiments/archive/compact_experience/training/runs/{mix_name}_100M_seed43"
        cmd = (
            f"python {trainer} "
            f"--example_jsonl {train_path} "
            f"--output_dir {out_dir} "
            f"--tokenizer_path {tokenizer} --tokenizer_label baseline16k "
            f"--max_word_exposure 100000000 "
            f"--example_pool_words {POOL_WORDS} "
            f"--model_type deberta_v2 --n_layer 8 --hidden_size 480 --n_head 8 --ffn_mult 4 "
            f"--max_seq_length 256 --seq_length 256 "
            f"--batch_size 256 --learning_rate 1e-3 --weight_decay 0.01 --warmup_fraction 0.05 "
            f"--seed 43 --mask_prob 0.15 --mask_mode wwm "
            f"--num_workers 0 --checkpoint_words 10000000 --log_every 50"
        )
        commands[mix_name] = cmd
        mixture_info[mix_name]["train_command"] = cmd
        mixture_info[mix_name]["output_dir"] = out_dir
        print(f"  {mix_name}: {out_dir}")
    
    # 6. Save summary
    summary = {
        "status": "MIXTURE_MATERIALIZED",
        "design": {
            "hypothesis": "Optimal mix of official (syntax/diagnostic) + aligned (entity-coherent) "
                         "data can exceed either pure endpoint on Overall",
            "official_pool": f"{len(official)} examples, {POOL_WORDS} words",
            "aligned_pool": f"{len(aligned)} examples, {sum(r['words'] for r in aligned)} words",
            "mixture_total": f"{POOL_WORDS} words per arm",
            "training_exposure": f"{POOL_WORDS * 10} words (10 exact passes)",
            "recipe": "DeBERTa-v2 8×480, baseline16k, WWM 0.15, b256, seq256, LR 1e-3, seed 43",
        },
        "arms": mixture_info,
        "existing_endpoints": {
            "0pct_aligned": {
                "name": "INITIAL_MODEL_STUDIES research official baseline",
                "path": "experiments/archive/initial_model_studies/training/runs"
                       "fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model/chck_100M",
                "scores": {"Entity": 21.24, "BLiMP": 67.34, "Supplement": 65.2,
                          "EWoK": 49.64, "COMPS": 53.11, "GlobalPIQA_mean": 36.12,
                          "Reading": 7.33, "equal7_mean": 42.854},
            },
            "100pct_aligned": {
                "name": "research ALIGNED 100M",
                "path": "experiments/archive/compact_experience/training/runs"
                       "aligned_100M_seed43/hf_model/chck_100M",
                "scores": {"Entity_full": 29.16, "Entity_fast": 27.74,
                          "BLiMP": 64.12, "Supplement": 60.8,
                          "EWoK": 51.18, "COMPS": 51.42,
                          "GlobalPIQA_mean": 37.52, "Reading": 7.16,
                          "equal7_mean": 42.849},
            },
        },
        "elapsed_sec": round(time.time() - start, 1),
        "seed": SEED,
    }
    
    summary_path = OUT_DIR / "mixture_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{'=' * 60}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
