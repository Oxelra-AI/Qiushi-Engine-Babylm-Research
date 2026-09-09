#!/usr/bin/env python3
"""Create one frozen untrained DeBERTa initialization for all research arms."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import sys
import time

import torch

USER_ROOT = pathlib.Path(".").resolve()
COMPACT_EXPERIENCE_SCRIPTS = USER_ROOT / "experiments/archive" / 'compact_experience' / "scripts"
if str(COMPACT_EXPERIENCE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(COMPACT_EXPERIENCE_SCRIPTS))

from masking_curriculum_trainer import make_portable_tokenizer, build_model, save_hf_checkpoint  # noqa: E402


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_seq_length", type=int, default=256)
    p.add_argument("--max_position_embeddings", type=int, default=512)
    p.add_argument("--hidden_size", type=int, default=480)
    p.add_argument("--n_layer", type=int, default=8)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--ffn_mult", type=int, default=4)
    p.add_argument("--position_buckets", type=int, default=256)
    p.add_argument("--max_relative_positions", type=int, default=256)
    p.add_argument("--deberta_pos_att_type", default="p2c,c2p")
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--extra_init_seed", type=int, default=43022)
    args = p.parse_args()
    t0 = time.time()
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = make_portable_tokenizer(args.tokenizer_path)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.extra_init_seed >= 0:
        random.seed(args.extra_init_seed)
        torch.manual_seed(args.extra_init_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.extra_init_seed)
    model = build_model(args, tokenizer)
    save_hf_checkpoint(model, tokenizer, out)
    weight_path = out / "model.safetensors"
    summary = {
        "status": "IDENTICAL_DEBERTA_INIT_DONE",
        "output_dir": str(out),
        "tokenizer_path": args.tokenizer_path,
        "vocab_size": len(tokenizer),
        "params": sum(x.numel() for x in model.parameters()),
        "model_safetensors_sha256": sha256_file(weight_path),
        "seed": args.seed,
        "extra_init_seed": args.extra_init_seed,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    (out / "identical_init_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
