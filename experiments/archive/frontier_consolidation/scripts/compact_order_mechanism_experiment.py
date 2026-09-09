#!/usr/bin/env python3
"""research: Compact ordered-vs-scrambled mechanism experiment launcher.

Trains two stock DeBERTa-v2 8x480 arms that differ ONLY in whether compact-pair
view text retains its natural word order or is word-shuffled. Both arms use the
research pool scaffold, legal research tokenizer, 100M LR horizon, and identical seeds.

Usage:
  # Preflight check (no training):
  python compact_order_mechanism_experiment.py --preflight

  # Train ordered arm on GPU0:
  CUDA_VISIBLE_DEVICES=0 python compact_order_mechanism_experiment.py --arm ordered

  # Train scrambled arm on GPU1:
  CUDA_VISIBLE_DEVICES=1 python compact_order_mechanism_experiment.py --arm scrambled

  # Print commands without running:
  python compact_order_mechanism_experiment.py --dry
"""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# ── Paths ──
USER_ROOT = _public_path('experiments/archive/frontier_consolidation/scripts/compact_order_mechanism_experiment.py')
for _ in range(10):
    if (_public_path("experiments")).exists():
        break
    USER_ROOT = _public_path('experiments/archive/frontier_consolidation/scripts')
STUDY = _public_path('experiments/archive/frontier_consolidation')
WS = _public_path('experiments/archive/frontier_consolidation')

TRAINER = _public_path('experiments/archive/compact_experience/scripts/masking_curriculum_trainer.py')
TOKENIZER = _public_path('experiments/archive/frontier_consolidation/data/compliant_tokenizer')
SCAFFOLD = _public_path('experiments/archive/frontier_consolidation/data/compact_order_factorial_pool_scaffold')

POOLS = {
    "ordered": {
        "10m": _public_path('experiments/archive/frontier_consolidation/data/compact_order_factorial_pool_scaffold/compact_ordered_10M.jsonl'),
        "40m": _public_path('experiments/archive/frontier_consolidation/data/compact_order_factorial_pool_scaffold/compact_ordered_40M.jsonl'),
        "sha_10m": "215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23",
        "sha_40m": "48313173a4cd93e768f490c8a5eaaa850fc828fabc9cfb21ebad73651bef85dc",
    },
    "scrambled": {
        "10m": _public_path('experiments/archive/frontier_consolidation/data/compact_order_factorial_pool_scaffold/compact_scrambled_10M.jsonl'),
        "40m": _public_path('experiments/archive/frontier_consolidation/data/compact_order_factorial_pool_scaffold/compact_scrambled_40M.jsonl'),
        "sha_10m": "07101d1391fbc1090f1318ead2c780a8a0071f98e2da06b6be30e9a19ca14b1d",
        "sha_40m": "7b8dcd13655ab3208938ff1cdc633bb1a695cd2d1765700256b052c96fda26e3",
    },
}

# ── Training recipe (exact match to research legal reference) ──
RECIPE = {
    "hidden_size": 480,
    "n_layer": 8,
    "n_head": 8,
    "ffn_mult": 4,
    "seed": 43,
    "extra_init_seed": 43022,
    "train_rng_seed": 43023,
    "batch_size": 256,
    "seq_length": 256,
    "max_seq_length": 256,
    "learning_rate": 0.001,
    "warmup_fraction": 0.06,
    "weight_decay": 0.01,
    "masking_curriculum": "wwm_fixed",
    "mask_prob_start": 0.15,
    "mask_prob_end": 0.15,
    "max_word_exposure": 40_000_000,
    "checkpoint_words": 20_000_000,
    "lr_total_steps": 2529,  # 100M LR horizon
    "num_workers": 0,
    "log_every": 50,
    "dynamics_trace_every": 200,
}


def sha256_file(path: Path, max_bytes: int = 0) -> str:
    h = hashlib.sha256()
    read = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
            read += len(chunk)
            if max_bytes and read >= max_bytes:
                break
    return h.hexdigest()


def count_jsonl(path: Path) -> tuple:
    """Return (line_count, total_words)."""
    lines = 0
    words = 0
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            lines += 1
            words += d.get("words", len(d["text"].split()))
    return lines, words


def preflight():
    """Verify all inputs and print experiment summary."""
    print("=" * 70)
    print("research Compact Order Mechanism Experiment — Preflight")
    print("=" * 70)

    ok = True

    # Check trainer exists
    if TRAINER.exists():
        print(f"✓ Trainer: {TRAINER}")
    else:
        print(f"✗ Trainer missing: {TRAINER}")
        ok = False

    # Check tokenizer
    tok_json = _public_path('experiments/archive/frontier_consolidation/data/compliant_tokenizer/tokenizer.json')
    if tok_json.exists():
        print(f"✓ Tokenizer: {TOKENIZER}")
    else:
        print(f"✗ Tokenizer missing: {tok_json}")
        ok = False

    # Check pools and verify SHAs
    for arm_name, arm in POOLS.items():
        for size in ["10m", "40m"]:
            pool_path = arm[size]
            expected_sha = arm[f"sha_{size}"]
            if not pool_path.exists():
                print(f"✗ Pool missing: {pool_path}")
                ok = False
                continue
            actual_sha = sha256_file(pool_path)
            sha_match = actual_sha == expected_sha
            lines, words = count_jsonl(pool_path)
            status = "✓" if sha_match else "✗"
            print(f"{status} {arm_name} {size}: {lines} lines, {words:,} words, SHA {'OK' if sha_match else 'MISMATCH'}")
            if not sha_match:
                print(f"  expected: {expected_sha}")
                print(f"  actual:   {actual_sha}")
                ok = False

    # Verify full row-level equivalence between ordered and scrambled pools.
    print("\nFull row-level equivalence check...")
    ord_path = POOLS["ordered"]["10m"]
    scr_path = POOLS["scrambled"]["10m"]
    if ord_path.exists() and scr_path.exists():
        from collections import Counter
        diffs = 0
        total = 0
        bad_ids = 0
        bad_words = 0
        bad_multisets = 0
        diff_indices = []
        with open(ord_path) as fo, open(scr_path) as fs:
            for i, (lo, ls) in enumerate(zip(fo, fs)):
                do = json.loads(lo)
                ds = json.loads(ls)
                total += 1
                bad_ids += int(do["example_id"] != ds["example_id"])
                bad_words += int(do["words"] != ds["words"])
                if do["text"] != ds["text"]:
                    diffs += 1
                    diff_indices.append(i)
                    bad_multisets += int(Counter(do["text"].split()) != Counter(ds["text"].split()))
        contiguous = diff_indices == list(range(min(diff_indices), max(diff_indices) + 1)) if diff_indices else False
        row_ok = (total == 64740 and diffs == 3005 and bad_ids == 0 and
                  bad_words == 0 and bad_multisets == 0 and contiguous and
                  diff_indices[0] == 0 and diff_indices[-1] == 3004)
        print(f"  {'✓' if row_ok else '✗'} rows={total}, changed_text={diffs} (indices 0..3004 contiguous={contiguous})")
        print(f"  {'✓' if bad_ids == 0 else '✗'} example_id mismatches={bad_ids}")
        print(f"  {'✓' if bad_words == 0 else '✗'} word-count mismatches={bad_words}")
        print(f"  {'✓' if bad_multisets == 0 else '✗'} changed-row whitespace-token multiset mismatches={bad_multisets}")
        if not row_ok:
            ok = False

    # Print recipe
    print("\nTraining recipe:")
    for k, v in RECIPE.items():
        print(f"  {k}: {v}")

    # Print run directories
    for arm_name in POOLS:
        run_dir = _public_path('experiments/archive/frontier_consolidation/training/runs') / f"compact_order_{arm_name}_40M_seed43022"
        exists = "EXISTS" if run_dir.exists() else "will be created"
        print(f"\n  {arm_name} run dir: {run_dir.relative_to(USER_ROOT)} [{exists}]")

    print(f"\nPreflight {'PASSED' if ok else 'FAILED'}")
    return ok


def build_command(arm_name: str) -> list:
    """Build the training command for one arm."""
    pool_40m = POOLS[arm_name]["40m"]
    run_dir = _public_path('experiments/archive/frontier_consolidation/training/runs') / f"compact_order_{arm_name}_40M_seed43022"

    cmd = [
        sys.executable,
        str(TRAINER),
        "--example_jsonl", str(pool_40m),
        "--example_jsonl_label", f"compact_order_{arm_name}_40M",
        "--output_dir", str(run_dir),
        "--tokenizer_path", str(TOKENIZER),
        "--tokenizer_label", "compliant16k_reinvest10M",
    ]
    for k, v in RECIPE.items():
        cmd.extend([f"--{k}", str(v)])
    return cmd


def run_arm(arm_name: str):
    """Train one arm on its assigned GPU."""
    cmd = build_command(arm_name)
    run_dir = _public_path('experiments/archive/frontier_consolidation/training/runs') / f"compact_order_{arm_name}_40M_seed43022"
    run_dir.mkdir(parents=True, exist_ok=True)

    gpu = "0" if arm_name == "ordered" else "1"
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = gpu

    # Save command record
    record = {
        "experiment": "compact_order_mechanism",
        "arm": arm_name,
        "pool_sha": POOLS[arm_name]["sha_40m"],
        "recipe": RECIPE,
        "command": [str(c) for c in cmd],
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": gpu,
    }
    (run_dir / "train_command.json").write_text(json.dumps(record, indent=2))
    print(f"\nLaunching {arm_name} arm...")
    print(f"  Run dir: {run_dir}")
    print(f"  Pool: {POOLS[arm_name]['40m']}")
    print(f"  GPU: {gpu}")
    print(f"  Command: {' '.join(str(c) for c in cmd[:6])}...")
    print()

    proc = subprocess.run(cmd, cwd=str(USER_ROOT), env=env)
    record["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    record["returncode"] = proc.returncode
    (run_dir / "train_command.json").write_text(json.dumps(record, indent=2))

    if proc.returncode != 0:
        print(f"\n✗ {arm_name} arm failed with return code {proc.returncode}")
        sys.exit(proc.returncode)
    else:
        print(f"\n✓ {arm_name} arm completed successfully")


def main():
    p = argparse.ArgumentParser(description="research compact order mechanism experiment")
    p.add_argument("--preflight", action="store_true", help="Run preflight checks only")
    p.add_argument("--arm", choices=["ordered", "scrambled"], help="Train one arm")
    p.add_argument("--dry", action="store_true", help="Print commands without running")
    args = p.parse_args()

    if args.preflight:
        ok = preflight()
        sys.exit(0 if ok else 1)
    elif args.dry:
        for arm_name in POOLS:
            cmd = build_command(arm_name)
            run_dir = _public_path('experiments/archive/frontier_consolidation/training/runs') / f"compact_order_{arm_name}_40M_seed43022"
            gpu = "0" if arm_name == "ordered" else "1"
            print(f"\n# {arm_name} arm (GPU {gpu}):")
            print(f"CUDA_VISIBLE_DEVICES={gpu} {' '.join(str(c) for c in cmd)}")
            print(f"# Run dir: {run_dir.relative_to(USER_ROOT)}")
        print("\n# After both complete, evaluate:")
        for arm_name in POOLS:
            run_dir = _public_path('experiments/archive/frontier_consolidation/training/runs') / f"compact_order_{arm_name}_40M_seed43022"
            for chk in ["chck_20M", "chck_40M"]:
                print(f"# Score {arm_name} {chk}: {run_dir.relative_to(USER_ROOT)}/hf_model/{chk}")
        sys.exit(0)
    elif args.arm:
        # Quick preflight before training
        if not preflight():
            print("\nPreflight failed, aborting.")
            sys.exit(1)
        run_arm(args.arm)
    else:
        p.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
