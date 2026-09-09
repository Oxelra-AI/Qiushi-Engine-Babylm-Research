#!/usr/bin/env python3
"""CPU smoke tests for research word-mean MLM credit trainer."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import importlib.util
import json
import pathlib
import random
import sys
from types import SimpleNamespace

import torch


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
BASE_PATH = USER_ROOT / "experiments/archive/compact_experience/scripts/masking_curriculum_trainer.py"
WORDMEAN_PATH = USER_ROOT / "experiments/archive/frontier_consolidation/scripts/wordmean_mlm_trainer.py"
TOKENIZER = USER_ROOT / "experiments/archive/frontier_consolidation/data/compliant_tokenizer"
TRAIN_FILE = USER_ROOT / "experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl"
OUT = USER_ROOT / "experiments/archive/frontier_consolidation/data/wordmean_loss_smoke"


def import_path(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def first_examples(n: int):
    rows = []
    with TRAIN_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if len(rows) >= n:
                break
            obj = json.loads(line)
            rows.append(obj)
    return rows


def main() -> None:
    base = import_path("base_masking_for_smoke", BASE_PATH)
    wm = import_path("wordmean_for_smoke", WORDMEAN_PATH)
    tok = base.make_portable_tokenizer(str(TOKENIZER))
    rows = first_examples(8)
    examples = [base.Example(text=r["text"], words=int(r["words"]), example_id=int(r.get("example_id", i)), source=str(r.get("source", ""))) for i, r in enumerate(rows)]
    dataset = base.MaskedChunkDataset(examples, tok, 256)
    batch = base.collate([dataset[i] for i in range(len(dataset))])
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    word_group = batch["word_group"]

    state = base.MaskingCurriculumState(curriculum="wwm_fixed", mask_prob_start=0.15, mask_prob_end=0.15)
    state.initialize(vocab_size=len(tok), total_steps=1)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(43023)
    masked_inputs, labels = base.apply_masking_curriculum(input_ids, attention_mask, word_group, tok, state, gen)
    selected = labels != -100
    if int(selected.sum().item()) <= 0:
        raise RuntimeError("no selected tokens in smoke batch")
    if int(((word_group < 0) & selected).sum().item()) != 0:
        raise RuntimeError("selected tokens without WWM group")

    torch.manual_seed(12345)
    logits = torch.randn(input_ids.shape[0], input_ids.shape[1], len(tok), dtype=torch.float32)
    word_loss, word_stats = wm.word_mean_mlm_loss(logits, labels, word_group)
    token_loss, token_stats = wm.token_mean_mlm_loss(logits, labels)

    # Manual computation for word mean.
    token_ce = torch.nn.functional.cross_entropy(logits[selected], labels[selected], reduction="none")
    bsz, seq = labels.shape
    batch_ids = torch.arange(bsz).view(bsz, 1).expand(bsz, seq)
    keys = batch_ids[selected] * (seq + 1) + word_group[selected]
    unique_keys, inverse, counts = torch.unique(keys, sorted=False, return_inverse=True, return_counts=True)
    manual = (token_ce / counts[inverse].float()).sum() / unique_keys.numel()
    manual_close = bool(torch.allclose(word_loss, manual, atol=1e-7, rtol=1e-7))
    if not manual_close:
        raise RuntimeError(f"manual word-mean mismatch {word_loss.item()} vs {manual.item()}")

    # If every selected token is its own group, word-mean and token-mean must match.
    unique_group = torch.arange(labels.numel()).view_as(labels)
    unique_group = torch.where(word_group >= 0, unique_group, word_group)
    unique_loss, unique_stats = wm.word_mean_mlm_loss(logits, labels, unique_group)
    unique_token_close = bool(torch.allclose(unique_loss, token_loss, atol=1e-7, rtol=1e-7))
    if not unique_token_close:
        raise RuntimeError(f"unique-group word mean != token mean {unique_loss.item()} vs {token_loss.item()}")

    OUT.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "WORDMEAN_LOSS_SMOKE_OK",
        "rows": len(rows),
        "selected_tokens": int(selected.sum().item()),
        "selected_word_groups": word_stats["selected_word_groups"],
        "mean_tokens_per_selected_group": word_stats["mean_tokens_per_selected_group"],
        "word_mean_loss": float(word_loss.item()),
        "token_mean_loss": float(token_loss.item()),
        "loss_delta_word_minus_token": float(word_loss.item() - token_loss.item()),
        "manual_word_mean_close": manual_close,
        "unique_group_equals_token_mean": unique_token_close,
        "tokenizer_size": len(tok),
        "masking_path": "base.apply_masking_curriculum reused unchanged",
    }
    (OUT / "wordmean_loss_smoke.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "wordmean_loss_smoke.md").write_text(
        "# research word-mean MLM loss smoke\n\n" +
        f"Status: `{result['status']}`\n\n" +
        f"Selected tokens/groups: `{result['selected_tokens']}` / `{result['selected_word_groups']}`; " +
        f"mean tokens per selected group `{result['mean_tokens_per_selected_group']:.4f}`.\n\n" +
        f"Manual word-mean equality: `{manual_close}`. Unique-group equality to token-mean: `{unique_token_close}`.\n\n" +
        f"Word-mean loss `{result['word_mean_loss']:.6f}`, token-mean loss `{result['token_mean_loss']:.6f}` on random logits.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
