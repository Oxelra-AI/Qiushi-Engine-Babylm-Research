#!/usr/bin/env python3
"""research: common-copied no-disentangle DeBERTa trainer wrapper.

Purpose
-------
The active research architecture-interaction experiment compares compact-minus-repeat
for stock DeBERTa (`pos_att_type=p2c,c2p`) against `no_disentangle_abs`
(`pos_att_type=[]`). research showed that simply omitting DeBERTa's optional
positional projection modules shifts the PyTorch construction RNG stream: many
same-named tensors differ at initialization. The four-cell research interaction is
still valid as an architecture-coordinate interaction, but if its result is
scientifically important and ambiguous, the clean follow-up is to train a
no-disentangle model whose every common tensor is copied from the exact same full
DeBERTa initialization.

This wrapper provides that follow-up without changing the base research/COMPACT_EXPERIENCE
training loop. It imports `masking_curriculum_trainer.py`, monkeypatches only
`build_model`, and then calls the trainer's `main()`. Inside the patched
`build_model`, after the base trainer has reset RNG to the normal initialization
seed and before it resets the train RNG, the wrapper:

  1. builds the full source model with `pos_att_type=p2c,c2p` from the current RNG
     state;
  2. builds the target no-disentangle model with `pos_att_type=[]`;
  3. copies every same-shaped/common tensor from the full source into the target;
  4. returns the target to the unchanged trainer loop.

The base trainer still constructs the dataset, masking, optimizer, scheduler,
training loop, checkpointing, metrics, and train-RNG reset exactly as before.

No training is launched by this file unless it is explicitly run without
`--common-copy-init-dry-run`. research uses only the dry-run verifier.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import os
import pathlib
import random
import statistics
import sys
import time
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
TRAINER_PATH = USER_ROOT / "experiments/archive" / 'compact_experience' / "scripts" / "masking_curriculum_trainer.py"
DEFAULT_RECORD_NAME = "common_copy_initialization.json"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def display_path(path: pathlib.Path) -> str:
    p = path if path.is_absolute() else USER_ROOT / path
    try:
        return str(p.resolve().relative_to(USER_ROOT))
    except ValueError:
        return str(p.resolve())


def parse_wrapper_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--common-copy-init-dry-run", action="store_true")
    parser.add_argument("--common-copy-source-pos-att-type", default="p2c,c2p")
    parser.add_argument("--common-copy-target-pos-att-type", default="")
    parser.add_argument("--common-copy-record-name", default=DEFAULT_RECORD_NAME)
    parser.add_argument("--common-copy-logit-samples", type=int, default=4)
    parser.add_argument("--common-copy-max-logit-length", type=int, default=64)
    return parser.parse_known_args(argv)


def load_trainer_module():
    spec = importlib.util.spec_from_file_location("compact_experience_masking_curriculum_trainer_commoncopy", str(TRAINER_PATH))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import base trainer from {TRAINER_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def normalize_pos_att_type(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    return [str(x).strip() for x in value if str(x).strip()]


def namespace_with(ns: argparse.Namespace, **updates: Any) -> argparse.Namespace:
    d = vars(ns).copy()
    d.update(updates)
    return argparse.Namespace(**d)


def tensor_group(key: str) -> str:
    if key.startswith("deberta.embeddings"):
        return "embeddings"
    if "attention.self.query_proj" in key or "attention.self.key_proj" in key or "attention.self.value_proj" in key:
        return "attention_qkv"
    if "pos_key_proj" in key or "pos_query_proj" in key:
        return "removed_pos_projection"
    if "attention.output" in key:
        return "attention_output"
    if ".intermediate." in key:
        return "intermediate"
    if ".output." in key and "attention.output" not in key:
        return "layer_output"
    if "rel_embeddings" in key:
        return "encoder_rel_embeddings"
    if key.startswith("cls."):
        return "mlm_head"
    return "other"


def compare_state_dicts(torch, left, right) -> dict[str, Any]:
    ls = left.state_dict()
    rs = right.state_dict()
    common = sorted(set(ls) & set(rs))
    only_left = sorted(set(ls) - set(rs))
    only_right = sorted(set(rs) - set(ls))
    exact = 0
    groups: dict[str, dict[str, Any]] = {}
    first_nonexact: list[dict[str, Any]] = []
    for k in common:
        lt = ls[k]
        rt = rs[k]
        g = tensor_group(k)
        rec = groups.setdefault(g, {"count": 0, "exact": 0, "nonexact": 0, "max_abs_values": [], "mean_abs_values": []})
        rec["count"] += 1
        if torch.equal(lt, rt):
            exact += 1
            rec["exact"] += 1
        else:
            rec["nonexact"] += 1
            diff = (lt - rt).abs().float()
            max_abs = float(diff.max().item()) if diff.numel() else 0.0
            mean_abs = float(diff.mean().item()) if diff.numel() else 0.0
            rec["max_abs_values"].append(max_abs)
            rec["mean_abs_values"].append(mean_abs)
            if len(first_nonexact) < 40:
                first_nonexact.append({"key": k, "group": g, "shape": list(lt.shape), "max_abs": max_abs, "mean_abs": mean_abs})
    for rec in groups.values():
        maxes = rec.pop("max_abs_values")
        means = rec.pop("mean_abs_values")
        rec["max_abs_max"] = max(maxes, default=0.0)
        rec["mean_abs_mean_over_tensors"] = float(statistics.mean(means)) if means else 0.0
    return {
        "left_key_count": len(ls),
        "right_key_count": len(rs),
        "common_key_count": len(common),
        "only_left_count": len(only_left),
        "only_right_count": len(only_right),
        "only_left_sample": only_left[:24],
        "only_right_sample": only_right[:24],
        "exact_common_count": exact,
        "nonexact_common_count": len(common) - exact,
        "groups": groups,
        "first_nonexact": first_nonexact,
    }


def copy_common_tensors(torch, source, target) -> dict[str, Any]:
    ss = source.state_dict()
    ts = target.state_dict()
    new_state = {}
    copied = 0
    missing_or_shape_mismatch: list[dict[str, Any]] = []
    for k, v in ts.items():
        if k in ss and tuple(ss[k].shape) == tuple(v.shape):
            new_state[k] = ss[k].detach().clone()
            copied += 1
        else:
            new_state[k] = v
            missing_or_shape_mismatch.append({
                "key": k,
                "target_shape": list(v.shape),
                "source_shape": list(ss[k].shape) if k in ss else None,
            })
    target.load_state_dict(new_state, strict=True)
    return {
        "copied_common_tensors": copied,
        "target_total_tensors": len(ts),
        "missing_or_shape_mismatch": missing_or_shape_mismatch,
        "all_target_tensors_copied": copied == len(ts) and not missing_or_shape_mismatch,
    }


def parameter_count(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def config_summary(model) -> dict[str, Any]:
    cfg = getattr(model, "config", None)
    if cfg is None:
        return {}
    return {
        "model_type": getattr(cfg, "model_type", None),
        "vocab_size": getattr(cfg, "vocab_size", None),
        "hidden_size": getattr(cfg, "hidden_size", None),
        "num_hidden_layers": getattr(cfg, "num_hidden_layers", None),
        "num_attention_heads": getattr(cfg, "num_attention_heads", None),
        "intermediate_size": getattr(cfg, "intermediate_size", None),
        "relative_attention": getattr(cfg, "relative_attention", None),
        "pos_att_type": getattr(cfg, "pos_att_type", None),
        "position_buckets": getattr(cfg, "position_buckets", None),
        "max_relative_positions": getattr(cfg, "max_relative_positions", None),
        "max_position_embeddings": getattr(cfg, "max_position_embeddings", None),
    }


def read_sample_texts(example_jsonl: str, n: int) -> list[str]:
    if n <= 0 or not example_jsonl:
        return []
    path = pathlib.Path(example_jsonl)
    if not path.is_absolute():
        path = USER_ROOT / path
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = str(obj.get("text", ""))
            if text:
                out.append(text)
            if len(out) >= n:
                break
    return out


def logit_diff(torch, tokenizer, source, target, texts: list[str], max_len: int) -> dict[str, Any]:
    if not texts:
        return {"n_texts": 0, "skipped": True}
    source.eval()
    target.eval()
    enc = tokenizer(texts, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
    with torch.no_grad():
        sl = source(**enc).logits.float()
        tl = target(**enc).logits.float()
    diff = (sl - tl).abs()
    return {
        "n_texts": len(texts),
        "shape": list(diff.shape),
        "mean_abs_logit_diff": float(diff.mean().item()),
        "rms_logit_diff": float(torch.sqrt((diff ** 2).mean()).item()),
        "max_abs_logit_diff": float(diff.max().item()),
    }


def build_common_copied_model(trainer, original_build_model, args: argparse.Namespace, tokenizer, wrapper_args: argparse.Namespace, *, dry_run: bool = False):
    import torch

    target_norm = normalize_pos_att_type(args.deberta_pos_att_type)
    requested_target_norm = normalize_pos_att_type(wrapper_args.common_copy_target_pos_att_type)
    source_norm = normalize_pos_att_type(wrapper_args.common_copy_source_pos_att_type)
    if target_norm != requested_target_norm:
        raise RuntimeError(
            "The base trainer argument --deberta_pos_att_type must equal the wrapper target pos_att_type. "
            f"got trainer={target_norm!r}, wrapper_target={requested_target_norm!r}"
        )
    if not source_norm:
        raise RuntimeError("The source pos_att_type must be nonempty; expected full p2c,c2p source.")

    source_args = namespace_with(args, deberta_pos_att_type=",".join(source_norm))
    target_args = namespace_with(args, deberta_pos_att_type=",".join(requested_target_norm))

    source_model = original_build_model(source_args, tokenizer)
    target_model = original_build_model(target_args, tokenizer)
    before_same_seed = compare_state_dicts(torch, source_model, target_model)
    copy_record = copy_common_tensors(torch, source_model, target_model)
    after_copy = compare_state_dicts(torch, source_model, target_model)

    texts = read_sample_texts(getattr(args, "example_jsonl", ""), int(wrapper_args.common_copy_logit_samples))
    logits = logit_diff(torch, tokenizer, source_model, target_model, texts, int(wrapper_args.common_copy_max_logit_length))

    record = {
        "status": "COMMON_COPIED_NODIS_INITIALIZATION",
        "created_utc": now(),
        "wrapper_path": display_path(_public_path('experiments/archive/frontier_consolidation/scripts/common_copy_nodis_trainer_wrapper.py')),
        "base_trainer_path": display_path(TRAINER_PATH),
        "mode": "dry_run_only" if dry_run else "training_build_model_patch",
        "source_pos_att_type": source_norm,
        "target_pos_att_type": requested_target_norm,
        "trainer_target_pos_att_type": target_norm,
        "target_is_common_copied_from_source_full_init": True,
        "source_parameter_count": parameter_count(source_model),
        "target_parameter_count": parameter_count(target_model),
        "source_config": config_summary(source_model),
        "target_config": config_summary(target_model),
        "before_copy_state_comparison": before_same_seed,
        "copy_record": copy_record,
        "after_copy_state_comparison": after_copy,
        "initial_logit_difference_source_full_vs_target_common_copied": logits,
        "expected_train_rng_reset_by_base_trainer_after_build_model": True,
        "interpretation": (
            "The returned target model has every same-shaped tensor copied from the same full DeBERTa initialization, "
            "while omitting the c2p/p2c positional projection tensors. The unchanged base trainer then resets train_rng_seed "
            "before masking/training, so the data order and masking stream remain the research coordinate."
        ),
        "boundary": "This record is initialization/build verification; it is not selected evaluation, SuperGLUE, AoA, upload, or leaderboard submission.",
    }

    out_dir = pathlib.Path(getattr(args, "output_dir", "."))
    if not out_dir.is_absolute():
        out_dir = USER_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    record_path = out_dir / wrapper_args.common_copy_record_name
    record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Source model is only an initialization donor; the returned model is the one trained.
    del source_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return target_model


def trainer_args_from_remaining(trainer, remaining: list[str]) -> argparse.Namespace:
    old_argv = sys.argv[:]
    try:
        sys.argv = [str(TRAINER_PATH)] + remaining
        return trainer.build_args()
    finally:
        sys.argv = old_argv


def dry_run(wrapper_args: argparse.Namespace, remaining: list[str]) -> None:
    import torch

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))
    trainer = load_trainer_module()
    args = trainer_args_from_remaining(trainer, remaining)
    tokenizer = trainer.make_portable_tokenizer(args.tokenizer_path) if getattr(args, "tokenizer_path", "") else None
    if tokenizer is None:
        raise RuntimeError("--tokenizer_path is required for common-copy dry-run verification")
    original_build_model = trainer.build_model
    target = build_common_copied_model(trainer, original_build_model, args, tokenizer, wrapper_args, dry_run=True)
    out_dir = pathlib.Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = USER_ROOT / out_dir
    record_path = out_dir / wrapper_args.common_copy_record_name
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    print(json.dumps({
        "status": "COMMON_COPY_DRY_RUN_OK",
        "record": display_path(record_path),
        "target_parameter_count": payload["target_parameter_count"],
        "after_copy_common": payload["after_copy_state_comparison"]["common_key_count"],
        "after_copy_nonexact_common": payload["after_copy_state_comparison"]["nonexact_common_count"],
        "initial_logit_mean_abs_full_vs_target": payload["initial_logit_difference_source_full_vs_target_common_copied"].get("mean_abs_logit_diff"),
    }, indent=2))
    del target


def train_mode(wrapper_args: argparse.Namespace, remaining: list[str]) -> None:
    trainer = load_trainer_module()
    original_build_model = trainer.build_model

    def patched_build_model(args: argparse.Namespace, tokenizer):
        return build_common_copied_model(trainer, original_build_model, args, tokenizer, wrapper_args, dry_run=False)

    trainer.build_model = patched_build_model
    sys.argv = [str(TRAINER_PATH)] + remaining
    trainer.main()


def main() -> None:
    wrapper_args, remaining = parse_wrapper_args(sys.argv[1:])
    if wrapper_args.common_copy_init_dry_run:
        dry_run(wrapper_args, remaining)
    else:
        train_mode(wrapper_args, remaining)


if __name__ == "__main__":
    main()
