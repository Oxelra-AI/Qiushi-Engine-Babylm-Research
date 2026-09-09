#!/usr/bin/env python3
"""research debug exact trainer masking reconstruction for compact-order streams."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

USER_ROOT = _public_path('experiments/archive/frontier_consolidation/scripts/mask_reconstruction_debug.py')
for _ in range(12):
    if (_public_path("experiments")).exists():
        break
    USER_ROOT = _public_path('experiments/archive/frontier_consolidation/scripts')
WS = _public_path('experiments/archive/frontier_consolidation')
TRAINER = _public_path('experiments/archive/compact_experience/scripts/masking_curriculum_trainer.py')
TOKENIZER = _public_path('experiments/archive/frontier_consolidation/data/compliant_tokenizer')
POOLS = {
    "ordered": _public_path('experiments/archive/frontier_consolidation/data/compact_order_factorial_pool_scaffold/compact_ordered_40M.jsonl'),
    "scrambled": _public_path('experiments/archive/frontier_consolidation/data/compact_order_factorial_pool_scaffold/compact_scrambled_40M.jsonl'),
}
RUNS = {
    "ordered": _public_path('experiments/archive/frontier_consolidation/training/runs/compact_order_ordered_40M_seed43022'),
    "scrambled": _public_path('experiments/archive/frontier_consolidation/training/runs/compact_order_scrambled_40M_seed43022'),
}


def import_trainer():
    spec = importlib.util.spec_from_file_location("masking_curriculum_trainer_step206", TRAINER)
    if spec is None or spec.loader is None:
        raise RuntimeError(TRAINER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_first_examples(path: Path, n_rows: int, mod):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            rows.append(mod.Example(
                text=str(obj["text"]),
                words=int(obj["words"]),
                example_id=int(obj.get("example_id", i)),
                source=str(obj.get("source", "example_jsonl")),
            ))
            if len(rows) >= n_rows:
                break
    return rows


def log_counts(arm: str, n: int) -> list[int]:
    p = RUNS[arm] / "training_log.jsonl"
    out: list[int] = []
    if not p.exists():
        return out
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            out.append(int(json.loads(line)["masked_tokens"]))
            if len(out) >= n:
                break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["ordered", "scrambled"], required=True)
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seq-length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=43023)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    mod = import_trainer()
    tok = mod.make_portable_tokenizer(str(TOKENIZER))
    examples = read_first_examples(POOLS[args.arm], args.batches * args.batch_size, mod)
    ds = mod.MaskedChunkDataset(examples, tok, args.seq_length)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=mod.collate, num_workers=0, pin_memory=False)
    state = mod.MaskingCurriculumState(curriculum="wwm_fixed", mask_prob_start=0.15, mask_prob_end=0.15, switch_frac=0.7, amlm_window=10, amlm_lambda=0.2)
    state.initialize(vocab_size=len(tok), total_steps=1012)
    dev = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device=dev)
    gen.manual_seed(args.seed)

    sim: list[int] = []
    cand: list[int] = []
    words: list[int] = []
    for step, batch in enumerate(loader, 1):
        input_ids = batch["input_ids"][:, :args.seq_length].contiguous().to(dev)
        attention_mask = batch["attention_mask"][:, :args.seq_length].contiguous().to(dev)
        word_group = batch["word_group"][:, :args.seq_length].contiguous().to(dev)
        state.current_step = step - 1
        _masked, labels = mod.apply_masking_curriculum(input_ids, attention_mask, word_group, tok, state, gen)
        sim.append(int((labels != -100).sum().item()))
        cand.append(int(attention_mask.bool().sum().item()))
        words.append(int(batch["words"].sum().item()))
        if step >= args.batches:
            break
    log = log_counts(args.arm, args.batches)
    diffs = [sim[i] - log[i] for i in range(min(len(sim), len(log)))]
    print(json.dumps({
        "arm": args.arm,
        "device": str(dev),
        "sim_counts": sim,
        "log_counts": log,
        "diffs": diffs,
        "exact_prefix_match": diffs == [0] * len(diffs) and len(sim) == len(log[:len(sim)]),
        "candidate_tokens_first_batches": cand,
        "batch_words_first_batches": words,
        "tokenizer_class": type(tok).__name__,
        "special_ids": list(map(int, tok.all_special_ids)),
    }, indent=2))


if __name__ == "__main__":
    main()
