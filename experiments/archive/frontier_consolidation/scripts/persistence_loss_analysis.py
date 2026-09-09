#!/usr/bin/env python3
"""research: Persistence loss analysis – why duplicate content stops paying.

Measures forward-only MLM loss on changed-block vs unchanged rows across the
existing view, repeat, breadth, and clean checkpoint ladders (all COMPLETE,
no new training). Tests whether duplicate rows become uninformative (low loss)
late while distinct rows retain error (higher loss) and learning signal.

Prediction:
- Repeat changed-block rows: loss drops fastest (text appears at 2 positions/pass)
- View changed-block rows: loss drops slower (unique surface form, same content)  
- Breadth changed-block rows: loss stays highest (completely novel FineWeb text)
- Unchanged rows: similar loss across all arms (same text in all pools)

If confirmed, this explains the persistence asymmetry:
  R-Cmax exEntity5 decays from +0.4361 to +0.0343 (common to late)
  V-Cmax and B-Cmax hold at +0.3853 and +0.3350 late

Runs on CPU to avoid GPU contention with training.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import argparse, json, os, pathlib, random, sys, time
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU only, before any torch import


def find_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_root()
WS = ROOT / "experiments/archive" / 'frontier_consolidation'
TOKENIZER_DIR = WS / "data" / "compliant_tokenizer"

# Pools: 10M-word single-pass JSONL (65,313 rows each)
POOL_FILES = {
    "view":    WS / "data" / "dose_2p64x_rowholdout_pools" / "compact_view_dose2p64x_10M.jsonl",
    "repeat":  WS / "data" / "dose_2p64x_rowholdout_pools" / "compact_repeat_dose2p64x_10M.jsonl",
    "breadth": WS / "data" / "dose_2p64x_breadth_rowholdout_pools" / "compact_breadth_dose2p64x_10M.jsonl",
    "clean":   WS / "data" / "dose_2p64x_rowholdout_pools" / "cleanqwen_lengthmatched_dose2p64x_10M.jsonl",
}

# Changed-block row metadata
CHANGED_META = {
    "view":    WS / "data" / "dose_2p64x_rowholdout_pools" / "compact_view_dose2p64x_changed_block_rows_meta.jsonl",
    "repeat":  WS / "data" / "dose_2p64x_rowholdout_pools" / "compact_repeat_dose2p64x_changed_block_rows_meta.jsonl",
    "breadth": WS / "data" / "dose_2p64x_breadth_rowholdout_pools" / "compact_breadth_dose2p64x_changed_block_rows_meta.jsonl",
    "clean":   WS / "data" / "dose_2p64x_rowholdout_pools" / "cleanqwen_lengthmatched_dose2p64x_changed_block_rows_meta.jsonl",
}

# Trained model runs (all COMPLETE with 10 checkpoints)
MODEL_RUNS = {
    "view":    WS / "training" / "runs" / "full_p2c_c2p_abs_view_dose2p64x_matched_rowholdout_deberta100M_seed43022",
    "repeat":  WS / "training" / "runs" / "full_p2c_c2p_abs_repeat_dose2p64x_matched_rowholdout_deberta100M_seed43022",
    "breadth": WS / "training" / "runs" / "full_p2c_c2p_abs_breadth_dose2p64x_matched_rowholdout_deberta100M_seed43022",
    "clean":   WS / "training" / "runs" / "full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43022",
}

CHECKPOINTS = ["chck_20M", "chck_40M", "chck_60M", "chck_80M", "chck_100M"]
N_CHANGED = 300   # sample size for changed-block rows
N_UNCHANGED = 300  # sample size for unchanged rows
MAX_SEQ_LEN = 256
MASK_PROB = 0.15
MASK_SEED = 29292  # fixed seed for consistent masking across checkpoints

OUT_DIR = WS / "data" / "persistence_loss_analysis"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def load_changed_indices(meta_path: pathlib.Path) -> set[int]:
    """Load row indices from changed-block metadata."""
    indices = set()
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            indices.add(row["row_index"])
    return indices


def sample_rows(pool_path: pathlib.Path, changed_indices: set[int],
                n_changed: int, n_unchanged: int, rng: random.Random) -> dict[str, list[str]]:
    """Read pool and sample changed/unchanged rows by text."""
    changed_texts: list[str] = []
    unchanged_texts: list[str] = []
    with open(pool_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row.get("text", "")
            if not text:
                continue
            if idx in changed_indices:
                changed_texts.append(text)
            else:
                unchanged_texts.append(text)

    # Sample
    if len(changed_texts) > n_changed:
        changed_texts = rng.sample(changed_texts, n_changed)
    if len(unchanged_texts) > n_unchanged:
        unchanged_texts = rng.sample(unchanged_texts, n_unchanged)

    return {"changed": changed_texts, "unchanged": unchanged_texts}


def compute_mlm_loss_on_texts(model, tokenizer, texts: list[str],
                              max_len: int, mask_prob: float,
                              mask_seed: int) -> dict[str, float]:
    """Compute mean per-token MLM loss on a list of texts with fixed masking."""
    import torch

    if not texts:
        return {"mean_loss": float("nan"), "n_rows": 0, "n_tokens_masked": 0}

    total_loss = 0.0
    total_masked = 0
    mask_rng = random.Random(mask_seed)

    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=max_len, padding=False)
            input_ids = enc["input_ids"].clone()
            labels = torch.full_like(input_ids, -100)

            # Fixed-seed masking: mask MASK_PROB of non-special tokens
            special_ids = {tokenizer.cls_token_id, tokenizer.sep_token_id,
                           tokenizer.pad_token_id}
            seq_len = input_ids.shape[1]
            maskable = [i for i in range(seq_len)
                        if input_ids[0, i].item() not in special_ids]
            n_mask = max(1, int(len(maskable) * mask_prob))
            positions = mask_rng.sample(maskable, min(n_mask, len(maskable)))

            for pos in positions:
                labels[0, pos] = input_ids[0, pos]
                input_ids[0, pos] = tokenizer.mask_token_id

            if labels[labels != -100].numel() == 0:
                continue

            outputs = model(input_ids=input_ids,
                            attention_mask=enc["attention_mask"],
                            labels=labels)
            # outputs.loss is mean over masked tokens
            n_masked = (labels != -100).sum().item()
            total_loss += outputs.loss.item() * n_masked
            total_masked += n_masked

    mean_loss = total_loss / total_masked if total_masked > 0 else float("nan")
    return {"mean_loss": round(mean_loss, 6), "n_rows": len(texts),
            "n_tokens_masked": total_masked}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="*", default=["view", "repeat", "breadth", "clean"])
    ap.add_argument("--checkpoints", nargs="*", default=CHECKPOINTS)
    ap.add_argument("--n-changed", type=int, default=N_CHANGED)
    ap.add_argument("--n-unchanged", type=int, default=N_UNCHANGED)
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Verify all paths
    missing = []
    for arm in args.arms:
        if not POOL_FILES[arm].exists():
            missing.append(f"pool:{arm}")
        if not CHANGED_META[arm].exists():
            missing.append(f"meta:{arm}")
        for ck in args.checkpoints:
            mp = MODEL_RUNS[arm] / "hf_model" / ck
            if not mp.exists():
                missing.append(f"model:{arm}/{ck}")

    plan = {
        "status": "PERSISTENCE_PLAN",
        "created_utc": now(),
        "arms": args.arms,
        "checkpoints": args.checkpoints,
        "n_changed": args.n_changed,
        "n_unchanged": args.n_unchanged,
        "total_measurements": len(args.arms) * len(args.checkpoints) * 2,
        "missing": missing,
        "cpu_only": True,
    }
    print(json.dumps(plan, indent=2), flush=True)
    if args.plan_only:
        return

    if missing:
        print(f"ERROR: {len(missing)} missing paths", flush=True)
        for m in missing:
            print(f"  - {m}", flush=True)
        # Continue with what's available

    # Load tokenizer once
    print(f"\nLoading tokenizer from {rel(TOKENIZER_DIR)}...", flush=True)
    import torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    print(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}", flush=True)

    # For each arm: sample rows once (same sample across checkpoints)
    arm_samples: dict[str, dict[str, list[str]]] = {}
    sample_rng = random.Random(42)
    for arm in args.arms:
        if not POOL_FILES[arm].exists() or not CHANGED_META[arm].exists():
            print(f"SKIP sampling {arm}: missing files", flush=True)
            continue
        print(f"\nSampling {arm} rows...", flush=True)
        changed_idx = load_changed_indices(CHANGED_META[arm])
        samples = sample_rows(POOL_FILES[arm], changed_idx,
                              args.n_changed, args.n_unchanged, sample_rng)
        arm_samples[arm] = samples
        print(f"  {arm}: {len(samples['changed'])} changed, "
              f"{len(samples['unchanged'])} unchanged rows "
              f"(from {len(changed_idx)} changed indices)", flush=True)

    # Measure loss across arms × checkpoints × block_types
    results: list[dict[str, Any]] = []
    total = sum(len(args.checkpoints) for a in args.arms if a in arm_samples)
    done = 0

    for arm in args.arms:
        if arm not in arm_samples:
            continue
        samples = arm_samples[arm]

        for ck in args.checkpoints:
            model_path = MODEL_RUNS[arm] / "hf_model" / ck
            if not model_path.exists():
                print(f"SKIP {arm}/{ck}: no model", flush=True)
                continue

            done += 1
            print(f"\n[{done}/{total}] {arm} {ck}: loading model...", flush=True)
            t0 = time.time()
            model = AutoModelForMaskedLM.from_pretrained(str(model_path))
            model.eval()
            load_time = round(time.time() - t0, 1)
            print(f"  Model loaded in {load_time}s", flush=True)

            for block_type in ["changed", "unchanged"]:
                texts = samples[block_type]
                if not texts:
                    continue
                t1 = time.time()
                loss_info = compute_mlm_loss_on_texts(
                    model, tokenizer, texts, MAX_SEQ_LEN, MASK_PROB, MASK_SEED)
                eval_time = round(time.time() - t1, 1)

                rec = {
                    "arm": arm, "checkpoint": ck, "block_type": block_type,
                    "mean_loss": loss_info["mean_loss"],
                    "n_rows": loss_info["n_rows"],
                    "n_tokens_masked": loss_info["n_tokens_masked"],
                    "eval_time_sec": eval_time,
                }
                results.append(rec)
                print(f"  {block_type}: loss={loss_info['mean_loss']:.4f} "
                      f"({loss_info['n_rows']} rows, {loss_info['n_tokens_masked']} tokens, "
                      f"{eval_time}s)", flush=True)

            # Free model memory
            del model
            import gc; gc.collect()

    # Save results
    output = {
        "status": "PERSISTENCE_DONE",
        "finished_utc": now(),
        "n_arms": len(arm_samples),
        "n_checkpoints": len(args.checkpoints),
        "n_results": len(results),
        "results": results,
        "sample_sizes": {arm: {k: len(v) for k, v in s.items()}
                         for arm, s in arm_samples.items()},
        "mask_prob": MASK_PROB,
        "mask_seed": MASK_SEED,
    }

    out_file = OUT_DIR / "persistence_loss_results.json"
    out_file.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(f"\nResults saved to {rel(out_file)}", flush=True)

    # Print summary table
    print("\n=== PERSISTENCE LOSS SUMMARY ===", flush=True)
    print(f"{'Arm':<10} {'Checkpoint':<12} {'Changed':>10} {'Unchanged':>10} {'Delta':>10}", flush=True)
    print("-" * 56, flush=True)
    by_arm_ck: dict[str, dict[str, float]] = {}
    for r in results:
        key = f"{r['arm']}_{r['checkpoint']}"
        by_arm_ck.setdefault(key, {})[r["block_type"]] = r["mean_loss"]
    for arm in args.arms:
        for ck in args.checkpoints:
            key = f"{arm}_{ck}"
            if key in by_arm_ck:
                ch = by_arm_ck[key].get("changed", float("nan"))
                unch = by_arm_ck[key].get("unchanged", float("nan"))
                delta = ch - unch if not (ch != ch or unch != unch) else float("nan")
                print(f"{arm:<10} {ck:<12} {ch:>10.4f} {unch:>10.4f} {delta:>+10.4f}", flush=True)

    print(json.dumps({"status": output["status"], "n_results": len(results)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
