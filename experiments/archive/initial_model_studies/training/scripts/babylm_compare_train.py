#!/usr/bin/env python3
"""BabyLM 2026 Strict-Small same-exposure mechanism comparison trainer.

This extends the clean V0 pilot lifecycle without replacing it. It keeps the
same official corpus revision, word-exposure accounting, portable tokenizer save
path, HF checkpoint naming, and BabyLM evaluation compatibility, while adding
mechanism-distinct 1M candidates:

- dense_causal: GPT2LMHeadModel control; use shape args for stronger dense control.
- dense_untied_causal: GPT2LMHeadModel control with untied output head,
  used as a parameter/compute-adjacent control for custom untied candidates.
- sparse_causal: custom routed sparse causal model with router load/entropy logs.
- morph_side_causal: custom causal model with persistent gated token-form side
  features fused on every forward pass; a stronger representation test than
  one-time embedding initialization.
- morph_init_causal: legacy smoke-only GPT2 embedding initialization; do not use
  as a main 1M comparison candidate.

The script writes checkpoints, manifests and scientific metrics to --output_dir.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast, get_cosine_schedule_with_warmup

SCRIPT_DIR = _public_path('experiments/archive/initial_model_studies/training/scripts')
MODELING_DIR = _public_path('experiments/archive/initial_model_studies/training/modeling')
if str(MODELING_DIR) not in sys.path:
    sys.path.insert(0, str(MODELING_DIR))

from configuration_babylm_sparse import BabyLMSparseConfig
from modeling_babylm_sparse import BabyLMSparseForCausalLM
from configuration_babylm_morph import BabyLMMorphConfig
from modeling_babylm_morph import BabyLMMorphForCausalLM
from configuration_babylm_memory import BabyLMMemoryConfig
from modeling_babylm_memory import BabyLMMemoryForCausalLM
from configuration_babylm_surface import BabyLMSurfaceConfig
from modeling_babylm_surface import BabyLMSurfaceForCausalLM

TRAIN_FILES = [
    "bnc_spoken.train.txt",
    "childes.train.txt",
    "gutenberg.train.txt",
    "open_subtitles.train.txt",
    "simple_wiki.train.txt",
    "switchboard.train.txt",
]
BASELINE_TOKENIZER_REPO = "BabyLM-community/BabyLM-2026-Baseline-GPT2-Strict-Small"
CUSTOM_CODE_FILES = {
    "sparse": ["configuration_babylm_sparse.py", "modeling_babylm_sparse.py"],
    "morph": ["configuration_babylm_morph.py", "modeling_babylm_morph.py"],
    "memory": ["configuration_babylm_memory.py", "modeling_babylm_memory.py"],
    "surface": ["configuration_babylm_surface.py", "modeling_babylm_surface.py"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True)
    p.add_argument("--dataset_id", default="BabyLM-community/BabyLM-2026-Strict-Small")
    p.add_argument("--dataset_revision", default="c92ab16b4f08858304b0815706065b3354d8fc0a")
    p.add_argument("--variant", choices=["dense_causal", "dense_untied_causal", "sparse_causal", "morph_side_causal", "char_surface_causal", "lookup_adapter_causal", "memory_causal", "memory_nopersist_causal", "morph_init_causal"], default="dense_causal")
    p.add_argument("--tokenizer", choices=["baseline16k"], default="baseline16k")
    p.add_argument("--max_word_exposure", type=int, default=1_000_000)
    p.add_argument("--example_pool_words", type=int, default=0, help="Build and shuffle this many corpus words before consuming max_word_exposure; 0 means use max_word_exposure.")
    p.add_argument("--order_mode", choices=["random", "source_stage", "readability_interleave"], default="random", help="Reorder the already-selected consumed examples without changing their set or source mix.")
    p.add_argument("--checkpoint_words", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seq_length", type=int, default=256)
    p.add_argument("--words_per_example", type=int, default=160)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--learning_rate", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--warmup_fraction", type=float, default=0.05)
    p.add_argument("--lr_total_steps", type=int, default=0, help="Explicit LR schedule length; 0 means consumed-example loader length.")
    p.add_argument("--shared_core_seed", type=int, default=-1, help="If >=0, initialize GPT2-equivalent core weights from this seed for paired dense/memory runs.")
    p.add_argument("--extra_init_seed", type=int, default=-1, help="If >=0, seed architecture-specific extra weights before applying shared core init.")
    p.add_argument("--train_rng_seed", type=int, default=-1, help="If >=0, reset Python/Torch training RNG after model construction for paired dropout/noise streams.")
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=256)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--ffn_mult", type=int, default=4)
    p.add_argument("--n_experts", type=int, default=4)
    p.add_argument("--top_k", type=int, default=2)
    p.add_argument("--router_aux_coef", type=float, default=0.01)
    p.add_argument("--morph_init_scale", type=float, default=0.02)
    p.add_argument("--morph_dim", type=int, default=64)
    p.add_argument("--morph_feature_scale", type=float, default=1.0)
    p.add_argument("--memory_dim", type=int, default=128)
    p.add_argument("--memory_dropout", type=float, default=0.0)
    p.add_argument("--surface_dim", type=int, default=64)
    p.add_argument("--surface_feature_scale", type=float, default=1.0)
    p.add_argument("--ngram_vocab_size", type=int, default=1024)
    p.add_argument("--max_ngrams_per_token", type=int, default=16)
    p.add_argument("--lookup_rank", type=int, default=5)
    p.add_argument("--log_every", type=int, default=20)
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def count_words_in_file(path: Path) -> int:
    total = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            total += len(line.split())
    return total


def download_dataset(args: argparse.Namespace, out: Path) -> tuple[Path, list[dict]]:
    raw_dir = out / "raw_dataset"
    raw_dir.mkdir(parents=True, exist_ok=True)
    local = Path(
        snapshot_download(
            repo_id=args.dataset_id,
            repo_type="dataset",
            revision=args.dataset_revision,
            allow_patterns=TRAIN_FILES + ["README.md"],
            local_dir=raw_dir,
            local_dir_use_symlinks=False,
        )
    )
    manifest_files = []
    for name in TRAIN_FILES:
        p = local / name
        if not p.exists():
            raise FileNotFoundError(f"required training file missing after download: {p}")
        manifest_files.append({
            "path": str(p),
            "name": name,
            "bytes": p.stat().st_size,
            "sha256": sha256_file(p),
            "whitespace_words": count_words_in_file(p),
        })
    return local, manifest_files


@dataclass
class Example:
    text: str
    words: int
    example_id: int = -1
    source: str = ""

SOURCE_STAGE_RANK = {
    "childes.train.txt": 0,
    "bnc_spoken.train.txt": 1,
    "switchboard.train.txt": 1,
    "open_subtitles.train.txt": 2,
    "simple_wiki.train.txt": 3,
    "gutenberg.train.txt": 4,
}


def approx_syllables(word: str) -> int:
    word = "".join(ch.lower() for ch in word if ch.isalpha())
    if not word:
        return 1
    vowels = "aeiouy"
    groups = 0
    prev = False
    for ch in word:
        cur = ch in vowels
        if cur and not prev:
            groups += 1
        prev = cur
    if word.endswith("e") and groups > 1:
        groups -= 1
    return max(1, groups)


def example_features(ex: Example) -> dict[str, float]:
    words = ex.text.split()
    n_words = max(1, len(words))
    punct_sent = max(1, sum(ex.text.count(p) for p in ".!?"))
    avg_word_len = sum(len(w.strip(".,!?;:'\"()[]{}")) for w in words) / n_words
    avg_syllables = sum(approx_syllables(w) for w in words) / n_words
    avg_sent_len = n_words / punct_sent
    flesch = 206.835 - 1.015 * avg_sent_len - 84.6 * avg_syllables
    type_token = len({w.lower().strip(".,!?;:'\"()[]{}") for w in words}) / n_words
    return {
        "word_count": float(n_words),
        "sentence_proxy_count": float(punct_sent),
        "avg_word_len": float(avg_word_len),
        "avg_syllables": float(avg_syllables),
        "flesch_approx": float(flesch),
        "type_token_proxy": float(type_token),
        "source_stage_rank": float(SOURCE_STAGE_RANK.get(ex.source, 9)),
    }


def difficulty_score(ex: Example) -> float:
    f = example_features(ex)
    # Higher score means harder: lower readability, longer words, more lexical diversity.
    return (-f["flesch_approx"]) + 3.0 * f["avg_word_len"] + 10.0 * f["type_token_proxy"]


def reorder_examples(examples: list[Example], mode: str) -> tuple[list[Example], dict]:
    if mode == "random":
        return examples, {"order_mode": mode, "order_note": "selected shuffled order unchanged"}
    selected_ids = [ex.example_id for ex in examples]
    if mode == "source_stage":
        ordered = sorted(enumerate(examples), key=lambda ie: (SOURCE_STAGE_RANK.get(ie[1].source, 9), ie[0]))
        out = [ex for _, ex in ordered]
        return out, {
            "order_mode": mode,
            "order_note": "stable sort of the same selected examples by developmental source stage",
            "source_stage_rank": SOURCE_STAGE_RANK,
            "same_selected_id_set": sorted(selected_ids) == sorted(ex.example_id for ex in out),
        }
    if mode == "readability_interleave":
        scored = sorted([(difficulty_score(ex), i, ex) for i, ex in enumerate(examples)], key=lambda x: (x[0], x[1]))
        n_bins = 5
        bins = [[] for _ in range(n_bins)]
        for rank, (_, _, ex) in enumerate(scored):
            bins[min(n_bins - 1, rank * n_bins // max(1, len(scored)))].append(ex)
        out: list[Example] = []
        max_len = max((len(b) for b in bins), default=0)
        for j in range(max_len):
            for b in bins:
                if j < len(b):
                    out.append(b[j])
        feature_summary = {}
        for key in ["word_count", "avg_word_len", "flesch_approx", "type_token_proxy"]:
            vals = [example_features(ex)[key] for ex in examples]
            feature_summary[key] = {"min": min(vals), "mean": sum(vals) / len(vals), "max": max(vals)} if vals else None
        return out, {
            "order_mode": mode,
            "order_note": "same selected examples sorted easy-to-hard by approximate readability/length, then interleaved across five bins",
            "n_bins": n_bins,
            "same_selected_id_set": sorted(selected_ids) == sorted(ex.example_id for ex in out),
            "feature_summary": feature_summary,
        }
    raise ValueError(f"unknown order_mode {mode}")

def iter_examples(files: list[Path], max_words: int, words_per_example: int) -> Iterable[Example]:
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
                    buf.append(w)
                    used += 1
                    if len(buf) >= words_per_example:
                        yield Example(" ".join(buf), len(buf), source=buf_source)
                        buf = []
                        buf_source = ""
                if used >= max_words:
                    break
        if used >= max_words:
            break
    if buf:
        yield Example(" ".join(buf), len(buf), source=buf_source)


class WordChunkDataset(Dataset):
    def __init__(self, examples: list[Example], tokenizer, seq_length: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.seq_length = seq_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex.text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.seq_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "words": ex.words}


def collate(batch: list[dict]) -> dict:
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
        "words": torch.tensor([x["words"] for x in batch], dtype=torch.long),
    }


def make_portable_tokenizer():
    base = AutoTokenizer.from_pretrained(BASELINE_TOKENIZER_REPO, revision="main", use_fast=True)
    backend = getattr(base, "_tokenizer", None) or getattr(base, "backend_tokenizer", None)
    tok = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token=base.bos_token if base.bos_token else "<s>",
        eos_token=base.eos_token if base.eos_token else "</s>",
        unk_token=base.unk_token if base.unk_token else "<unk>",
        pad_token=base.pad_token if base.pad_token else "<pad>",
        mask_token=getattr(base, "mask_token", None) or "<mask>",
        model_max_length=1024,
    )
    tok.init_kwargs.pop("tokenizer_class", None)
    tok.init_kwargs["tokenizer_class"] = "PreTrainedTokenizerFast"
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "<pad>"})
    return tok


def force_portable_tokenizer_config(dst: Path) -> None:
    tok_cfg = dst / "tokenizer_config.json"
    if tok_cfg.exists():
        cfg = json.loads(tok_cfg.read_text(encoding="utf-8"))
        cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
        cfg.setdefault("bos_token", "<s>")
        cfg.setdefault("eos_token", "</s>")
        cfg.setdefault("unk_token", "<unk>")
        cfg.setdefault("pad_token", "<pad>")
        cfg.setdefault("mask_token", "<mask>")
        tok_cfg.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def copy_custom_code_if_needed(model, dst: Path) -> None:
    if isinstance(model, BabyLMSparseForCausalLM):
        for name in CUSTOM_CODE_FILES["sparse"]:
            shutil.copy2(MODELING_DIR / name, dst / name)
    if isinstance(model, BabyLMMorphForCausalLM):
        for name in CUSTOM_CODE_FILES["morph"]:
            shutil.copy2(MODELING_DIR / name, dst / name)
    if isinstance(model, BabyLMMemoryForCausalLM):
        for name in CUSTOM_CODE_FILES["memory"]:
            shutil.copy2(MODELING_DIR / name, dst / name)
    if isinstance(model, BabyLMSurfaceForCausalLM):
        for name in CUSTOM_CODE_FILES["surface"]:
            shutil.copy2(MODELING_DIR / name, dst / name)


def save_hf_checkpoint(model, tokenizer, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dst, safe_serialization=True)
    tokenizer.save_pretrained(dst)
    force_portable_tokenizer_config(dst)
    copy_custom_code_if_needed(model, dst)


def token_form_features(token: str) -> list[str]:
    s = token.replace("Ġ", " ").replace("▁", " ")
    s = s.strip().lower()
    if not s:
        s = "<blank>"
    feats = ["len=" + str(min(len(s), 12))]
    feats.extend("c=" + c for c in s[:24])
    for n in (2, 3):
        feats.extend(f"g{n}=" + s[i:i+n] for i in range(max(0, len(s) - n + 1))[:24])
    for k in (1, 2, 3, 4):
        if len(s) >= k:
            feats.append(f"suf{k}=" + s[-k:])
            feats.append(f"pre{k}=" + s[:k])
    return feats


def hashed_feature_vector(features: list[str], dim: int, scale: float) -> torch.Tensor:
    v = torch.zeros(dim)
    if not features:
        return v
    for feat in features:
        h = hashlib.sha256(feat.encode("utf-8")).digest()
        # Use 4 byte chunks to choose signed dimensions.
        for off in range(0, min(16, len(h)), 4):
            val = int.from_bytes(h[off:off+4], "little")
            idx = val % dim
            sign = 1.0 if ((val >> 31) & 1) == 0 else -1.0
            v[idx] += sign
    norm = v.norm().clamp_min(1.0)
    return scale * v / norm


def apply_morph_init(model: GPT2LMHeadModel, tokenizer, scale: float, out: Path) -> dict:
    emb = model.transformer.wte.weight
    dim = emb.shape[1]
    examples = []
    with torch.no_grad():
        for idx in range(len(tokenizer)):
            tok = tokenizer.convert_ids_to_tokens(idx)
            vec = hashed_feature_vector(token_form_features(str(tok)), dim, scale).to(emb.device, emb.dtype)
            emb[idx].add_(vec)
            if idx < 12:
                examples.append({"id": idx, "token": str(tok), "features": token_form_features(str(tok))[:12]})
    manifest = {"method": "deterministic character/prefix/suffix hashed additive embedding initialization", "scale": scale, "examples": examples}
    (out / "morph_init_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def build_morph_feature_matrix(tokenizer, morph_dim: int, scale: float, out: Path) -> tuple[torch.Tensor, dict]:
    """Build fixed token-form features used by the persistent morphology side channel."""
    mat = torch.zeros(len(tokenizer), morph_dim)
    examples = []
    for idx in range(len(tokenizer)):
        tok = str(tokenizer.convert_ids_to_tokens(idx))
        feats = token_form_features(tok)
        mat[idx] = hashed_feature_vector(feats, morph_dim, scale)
        if idx < 12:
            examples.append({"id": idx, "token": tok, "features": feats[:12]})
    manifest = {
        "method": "persistent fixed token-form features projected and gated on every forward pass",
        "morph_dim": morph_dim,
        "scale": scale,
        "feature_source": "token strings from the official 16k BabyLM baseline tokenizer; no external morphology analyzer or outside language data",
        "examples": examples,
    }
    (out / "morph_side_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return mat, manifest


def feature_to_ngram_id(feature: str, ngram_vocab_size: int) -> int:
    if ngram_vocab_size < 2:
        raise ValueError("ngram_vocab_size must be >= 2")
    h = hashlib.sha256(feature.encode("utf-8")).digest()
    return 1 + (int.from_bytes(h[:8], "little") % (ngram_vocab_size - 1))


def build_surface_ngram_matrices(tokenizer, ngram_vocab_size: int, max_ngrams: int, out: Path) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Map each tokenizer string to shared hashed character/n-gram feature IDs.

    The IDs are fixed, but the corresponding n-gram embeddings are learned and
    shared across all tokens. This is intentionally stronger than the old fixed
    hashed morph-side features.
    """
    ids = torch.zeros(len(tokenizer), max_ngrams, dtype=torch.long)
    mask = torch.zeros(len(tokenizer), max_ngrams, dtype=torch.float32)
    examples = []
    count_hist: dict[str, int] = {}
    length_hist: dict[str, int] = {}
    used_ids: set[int] = set()
    for idx in range(len(tokenizer)):
        tok = str(tokenizer.convert_ids_to_tokens(idx))
        feats = token_form_features(tok)
        chosen = feats[:max_ngrams]
        for j, feat in enumerate(chosen):
            fid = feature_to_ngram_id(feat, ngram_vocab_size)
            ids[idx, j] = fid
            mask[idx, j] = 1.0
            used_ids.add(fid)
        clean = tok.replace("Ġ", " ").replace("▁", " ").strip()
        length_hist[str(min(len(clean), 12))] = length_hist.get(str(min(len(clean), 12)), 0) + 1
        count_hist[str(len(chosen))] = count_hist.get(str(len(chosen)), 0) + 1
        if idx < 16:
            examples.append({"id": idx, "token": tok, "features": chosen, "feature_ids": [int(ids[idx, j]) for j in range(len(chosen))]})
    manifest = {
        "method": "char_surface_causal learned shared char/n-gram composition over official tokenizer strings",
        "ngram_vocab_size": ngram_vocab_size,
        "max_ngrams_per_token": max_ngrams,
        "unique_hashed_ngram_ids_used": len(used_ids),
        "feature_source": "official BabyLM baseline tokenizer vocabulary strings only; no external text, morphology analyzer, labels, or corpus exposure",
        "hash_scheme": "sha256(feature) mapped to 1..ngram_vocab_size-1; id 0 is reserved/padded and masked out",
        "length_histogram_capped12": length_hist,
        "ngram_count_histogram": count_hist,
        "examples": examples,
    }
    (out / "surface_char_ngram_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return ids, mask, manifest


def lookup_adapter_manifest(tokenizer, lookup_rank: int, out: Path) -> dict:
    examples = [{"id": idx, "token": str(tokenizer.convert_ids_to_tokens(idx))} for idx in range(min(16, len(tokenizer)))]
    manifest = {
        "method": "lookup_adapter_causal learned token-ID side adapter with same fusion path and no cross-token character/n-gram sharing",
        "lookup_rank": lookup_rank,
        "feature_source": "token IDs only; added-capacity control for learned char_ngram surface composition",
        "examples": examples,
    }
    (out / "surface_lookup_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def build_model(args: argparse.Namespace, tokenizer, out: Path):
    if args.variant in {"dense_causal", "dense_untied_causal", "morph_init_causal"}:
        tie_embeddings = args.variant != "dense_untied_causal"
        cfg = GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=args.seq_length,
            n_ctx=args.seq_length,
            n_embd=args.n_embd,
            n_layer=args.n_layer,
            n_head=args.n_head,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            tie_word_embeddings=tie_embeddings,
        )
        model = GPT2LMHeadModel(cfg)
        extra = {"model_family": "GPT2LMHeadModel", "tie_word_embeddings": tie_embeddings}
        if args.variant == "morph_init_causal":
            extra["morph_init"] = apply_morph_init(model, tokenizer, args.morph_init_scale, out)
        return model, extra
    if args.variant == "sparse_causal":
        cfg = BabyLMSparseConfig(
            vocab_size=len(tokenizer),
            n_positions=args.seq_length,
            n_embd=args.n_embd,
            n_layer=args.n_layer,
            n_head=args.n_head,
            ffn_mult=args.ffn_mult,
            n_experts=args.n_experts,
            top_k=args.top_k,
            router_aux_coef=args.router_aux_coef,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        return BabyLMSparseForCausalLM(cfg), {"model_family": "BabyLMSparseForCausalLM"}
    if args.variant == "morph_side_causal":
        cfg = BabyLMMorphConfig(
            vocab_size=len(tokenizer),
            n_positions=args.seq_length,
            n_embd=args.n_embd,
            n_layer=args.n_layer,
            n_head=args.n_head,
            ffn_mult=args.ffn_mult,
            morph_dim=args.morph_dim,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        return BabyLMMorphForCausalLM(cfg), {"model_family": "BabyLMMorphForCausalLM", "morph_dim": args.morph_dim}
    if args.variant in {"char_surface_causal", "lookup_adapter_causal"}:
        surface_mode = "char_ngram" if args.variant == "char_surface_causal" else "lookup"
        cfg = BabyLMSurfaceConfig(
            vocab_size=len(tokenizer),
            n_positions=args.seq_length,
            n_embd=args.n_embd,
            n_layer=args.n_layer,
            n_head=args.n_head,
            ffn_mult=args.ffn_mult,
            surface_mode=surface_mode,
            surface_dim=args.surface_dim,
            ngram_vocab_size=args.ngram_vocab_size,
            max_ngrams_per_token=args.max_ngrams_per_token,
            lookup_rank=args.lookup_rank,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        model = BabyLMSurfaceForCausalLM(cfg)
        extra = {
            "model_family": "BabyLMSurfaceForCausalLM",
            "surface_mode": surface_mode,
            "surface_dim": args.surface_dim,
            "ngram_vocab_size": args.ngram_vocab_size,
            "max_ngrams_per_token": args.max_ngrams_per_token,
            "lookup_rank": args.lookup_rank,
        }
        if surface_mode == "char_ngram":
            ids, mask, manifest = build_surface_ngram_matrices(tokenizer, args.ngram_vocab_size, args.max_ngrams_per_token, out)
            model.surface.surface_ngram_ids.copy_(ids)
            model.surface.surface_ngram_mask.copy_(mask)
            extra["surface_manifest"] = manifest
        else:
            extra["surface_manifest"] = lookup_adapter_manifest(tokenizer, args.lookup_rank, out)
        return model, extra
    if args.variant in {"memory_causal", "memory_nopersist_causal"}:
        memory_enabled = args.variant == "memory_causal"
        cfg = BabyLMMemoryConfig(
            vocab_size=len(tokenizer),
            n_positions=args.seq_length,
            n_embd=args.n_embd,
            n_layer=args.n_layer,
            n_head=args.n_head,
            ffn_mult=args.ffn_mult,
            memory_dim=args.memory_dim,
            memory_dropout=args.memory_dropout,
            memory_enabled=memory_enabled,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        return BabyLMMemoryForCausalLM(cfg), {"model_family": "BabyLMMemoryForCausalLM", "memory_dim": args.memory_dim, "memory_dropout": args.memory_dropout, "memory_enabled": memory_enabled}
    raise ValueError(args.variant)


def reset_all_rng(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def apply_shared_gpt2_core_init(model, args: argparse.Namespace, tokenizer) -> dict:
    """Copy an untied GPT-2 core initialization into compatible custom models.

    Adapter-specific parameters remain separately initialized by --extra_init_seed
    before construction. This keeps dense, memory, and surface runs paired on the
    same Transformer core while testing only the intended added mechanism.
    """
    if args.shared_core_seed < 0:
        return {"shared_core_seed": None, "shared_core_init_applied": False}
    cfg = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=args.seq_length,
        n_ctx=args.seq_length,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        tie_word_embeddings=False,
    )
    state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(args.shared_core_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.shared_core_seed)
    core = GPT2LMHeadModel(cfg)
    torch.random.set_rng_state(state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    if isinstance(model, GPT2LMHeadModel):
        model.load_state_dict(core.state_dict(), strict=True)
        return {"shared_core_seed": args.shared_core_seed, "shared_core_init_applied": True, "shared_core_target": "GPT2LMHeadModel"}
    if isinstance(model, (BabyLMMemoryForCausalLM, BabyLMSurfaceForCausalLM)):
        with torch.no_grad():
            model.wte.weight.copy_(core.transformer.wte.weight)
            model.wpe.weight.copy_(core.transformer.wpe.weight)
            model.ln_f.weight.copy_(core.transformer.ln_f.weight)
            model.ln_f.bias.copy_(core.transformer.ln_f.bias)
            model.lm_head.weight.copy_(core.lm_head.weight)
            for i, block in enumerate(model.h):
                g = core.transformer.h[i]
                block.ln_1.weight.copy_(g.ln_1.weight); block.ln_1.bias.copy_(g.ln_1.bias)
                block.ln_2.weight.copy_(g.ln_2.weight); block.ln_2.bias.copy_(g.ln_2.bias)
                block.attn.c_attn.weight.copy_(g.attn.c_attn.weight.t()); block.attn.c_attn.bias.copy_(g.attn.c_attn.bias)
                block.attn.c_proj.weight.copy_(g.attn.c_proj.weight.t()); block.attn.c_proj.bias.copy_(g.attn.c_proj.bias)
                block.ffn.net[0].weight.copy_(g.mlp.c_fc.weight.t()); block.ffn.net[0].bias.copy_(g.mlp.c_fc.bias)
                block.ffn.net[2].weight.copy_(g.mlp.c_proj.weight.t()); block.ffn.net[2].bias.copy_(g.mlp.c_proj.bias)
        return {"shared_core_seed": args.shared_core_seed, "shared_core_init_applied": True, "shared_core_target": model.__class__.__name__}
    return {"shared_core_seed": args.shared_core_seed, "shared_core_init_applied": False, "shared_core_target": model.__class__.__name__}


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(json.dumps({"event": "start", "run_dir": str(out), "args": vars(args)}, ensure_ascii=False), flush=True)

    raw_dir, manifest_files = download_dataset(args, out)
    total_words = sum(x["whitespace_words"] for x in manifest_files)
    selected_files = [raw_dir / name for name in TRAIN_FILES]
    pool_words_requested = args.example_pool_words if args.example_pool_words and args.example_pool_words > 0 else args.max_word_exposure
    pool_words = min(pool_words_requested, total_words)
    selected_words = min(args.max_word_exposure, pool_words)
    if args.max_word_exposure > pool_words:
        raise RuntimeError(f"max_word_exposure {args.max_word_exposure} exceeds example pool {pool_words}")

    tokenizer = make_portable_tokenizer()
    tokenizer_manifest = {
        "mode": args.tokenizer,
        "source_repo": BASELINE_TOKENIZER_REPO,
        "source_revision": "main",
        "vocab_size": len(tokenizer),
        "tokenizer_language_training_exposure_words": 0,
        "rule_note": "Uses official BabyLM 2026 baseline tokenizer, re-saved as PreTrainedTokenizerFast for portability.",
    }

    pool_examples = list(iter_examples(selected_files, pool_words, args.words_per_example))
    for i, ex in enumerate(pool_examples):
        ex.example_id = i
    pool_actual_words = sum(ex.words for ex in pool_examples)
    if pool_actual_words != pool_words:
        raise RuntimeError(f"pool word selection mismatch {pool_actual_words} vs {pool_words}")
    rng = random.Random(args.seed)
    rng.shuffle(pool_examples)
    examples: list[Example] = []
    actual_words = 0
    for ex in pool_examples:
        if actual_words >= selected_words:
            break
        if actual_words + ex.words <= selected_words:
            examples.append(ex)
            actual_words += ex.words
        else:
            take = selected_words - actual_words
            if take > 0:
                examples.append(Example(" ".join(ex.text.split()[:take]), take, example_id=ex.example_id, source=ex.source))
                actual_words += take
            break
    if actual_words != selected_words:
        raise RuntimeError(f"word selection mismatch {actual_words} vs {selected_words}")
    selected_example_ids_before_order = [ex.example_id for ex in examples]
    examples, order_info = reorder_examples(examples, args.order_mode)
    if sorted(selected_example_ids_before_order) != sorted(ex.example_id for ex in examples):
        raise RuntimeError("order_mode changed the selected example set; pure order experiment requires identical consumed examples")
    source_counts: dict[str, int] = {}
    source_words: dict[str, int] = {}
    for ex in examples:
        source_counts[ex.source] = source_counts.get(ex.source, 0) + 1
        source_words[ex.source] = source_words.get(ex.source, 0) + ex.words
    example_order_manifest = {
        "seed": args.seed,
        "example_pool_words_requested": pool_words_requested,
        "example_pool_words_actual": pool_words,
        "max_word_exposure_requested": args.max_word_exposure,
        "selected_for_training_words": actual_words,
        "words_per_example": args.words_per_example,
        "num_pool_examples": len(pool_examples),
        "num_consumed_examples": len(examples),
        "order_mode": args.order_mode,
        "order_info": order_info,
        "source_counts_consumed": source_counts,
        "source_words_consumed": source_words,
        "selected_example_ids_before_order": selected_example_ids_before_order,
        "consumed_example_ids_in_order": [ex.example_id for ex in examples],
        "consumed_example_ids_sorted": sorted(ex.example_id for ex in examples),
    }
    (out / "example_order_manifest.json").write_text(json.dumps(example_order_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    dataset = WordChunkDataset(examples, tokenizer, args.seq_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=2, pin_memory=torch.cuda.is_available())

    if args.extra_init_seed >= 0:
        reset_all_rng(args.extra_init_seed)
    model, model_extra = build_model(args, tokenizer, out)
    shared_init_info = apply_shared_gpt2_core_init(model, args, tokenizer)
    model_extra.update(shared_init_info)
    if args.train_rng_seed >= 0:
        reset_all_rng(args.train_rng_seed)
    param_count = sum(p.numel() for p in model.parameters())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    total_steps = len(loader)
    schedule_total_steps = args.lr_total_steps if args.lr_total_steps and args.lr_total_steps > 0 else total_steps
    if schedule_total_steps < total_steps:
        raise RuntimeError(f"lr_total_steps {schedule_total_steps} is shorter than actual training steps {total_steps}")
    warmup_steps = max(1, int(schedule_total_steps * args.warmup_fraction))
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=warmup_steps, num_training_steps=schedule_total_steps)

    log_path = out / "training_log.jsonl"
    cumulative_words = 0
    loss_values: list[float] = []
    saved_checkpoints: list[dict] = []
    next_checkpoint_words = args.checkpoint_words if args.checkpoint_words and args.checkpoint_words > 0 else None
    router_history: list[dict] = []
    morph_history: list[dict] = []
    memory_history: list[dict] = []
    surface_history: list[dict] = []
    model.train()
    with log_path.open("w", encoding="utf-8") as logf:
        for step, batch in enumerate(loader, 1):
            words = int(batch.pop("words").sum().item())
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            optim.zero_grad(set_to_none=True)
            outputs = model(**batch)
            loss = outputs.loss
            if loss is None:
                raise RuntimeError("model returned no loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            cumulative_words += words
            loss_float = float(loss.detach().cpu())
            loss_values.append(loss_float)
            router_stats = getattr(model, "_last_router_stats", {}) or {}
            morph_stats = getattr(model, "_last_morph_stats", {}) or {}
            memory_stats = getattr(model, "_last_memory_stats", {}) or {}
            surface_stats = getattr(model, "_last_surface_stats", {}) or {}
            if router_stats:
                router_history.append({"step": step, **router_stats})
            if morph_stats:
                morph_history.append({"step": step, **morph_stats})
            if memory_stats:
                memory_history.append({"step": step, **memory_stats})
            if surface_stats:
                surface_history.append({"step": step, **surface_stats})
            rec = {
                "step": step,
                "loss": loss_float,
                "lr": float(sched.get_last_lr()[0]),
                "batch_words": words,
                "cumulative_word_exposure": cumulative_words,
                "elapsed_sec": time.time() - start_time,
            }
            if router_stats:
                rec.update(router_stats)
            if morph_stats:
                rec.update(morph_stats)
            if memory_stats:
                rec.update(memory_stats)
            if surface_stats:
                rec.update(surface_stats)
            logf.write(json.dumps(rec) + "\n")
            if step == 1 or step % args.log_every == 0 or step == total_steps:
                print(json.dumps({"event": "train", **rec}), flush=True)
            while next_checkpoint_words is not None and cumulative_words >= next_checkpoint_words and next_checkpoint_words <= args.max_word_exposure:
                if next_checkpoint_words < 1_000_000:
                    ckpt_name = "chck_1M"
                elif next_checkpoint_words % 1_000_000 == 0:
                    ckpt_name = f"chck_{next_checkpoint_words // 1_000_000}M"
                else:
                    ckpt_name = f"chck_{next_checkpoint_words}w"
                ckpt_path = out / "hf_model" / ckpt_name
                save_hf_checkpoint(model, tokenizer, ckpt_path)
                rec_ckpt = {"name": ckpt_name, "target_word_exposure": next_checkpoint_words, "actual_cumulative_word_exposure": cumulative_words, "path": str(ckpt_path)}
                saved_checkpoints.append(rec_ckpt)
                print(json.dumps({"event": "checkpoint_saved", **rec_ckpt}), flush=True)
                next_checkpoint_words += args.checkpoint_words

    save_hf_checkpoint(model, tokenizer, out / "hf_model")
    if not saved_checkpoints:
        ckpt_path = out / "hf_model" / "chck_1M"
        save_hf_checkpoint(model, tokenizer, ckpt_path)
        saved_checkpoints.append({"name": "chck_1M", "target_word_exposure": args.checkpoint_words, "actual_cumulative_word_exposure": cumulative_words, "path": str(ckpt_path)})
    data_manifest = {
        "dataset_id": args.dataset_id,
        "dataset_revision": args.dataset_revision,
        "download_dir": str(raw_dir),
        "files": manifest_files,
        "total_official_dataset_whitespace_words_counted": total_words,
        "example_pool_words_requested": pool_words_requested,
        "example_pool_words_actual": pool_words,
        "selected_for_this_run_whitespace_words": actual_words,
        "max_word_exposure_requested": args.max_word_exposure,
        "example_order_manifest": str(out / "example_order_manifest.json"),
        "order_mode": args.order_mode,
        "word_exposure_count_method": "Python str.split() over selected training text; examples are built from example_pool_words, shuffled once by seed, consumed until max_word_exposure, then optionally reordered without changing the selected example set.",
    }
    (out / "data_manifest.json").write_text(json.dumps(data_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "tokenizer_manifest.json").write_text(json.dumps(tokenizer_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    router_summary = None
    if router_history:
        last = router_history[-1]
        router_summary = {k: last[k] for k in last if k != "step"}
    morph_summary = None
    if morph_history:
        last = morph_history[-1]
        morph_summary = {k: last[k] for k in last if k != "step"}
    memory_summary = None
    if memory_history:
        last = memory_history[-1]
        memory_summary = {k: last[k] for k in last if k != "step"}
    surface_summary = None
    if surface_history:
        last = surface_history[-1]
        surface_summary = {k: last[k] for k in last if k != "step"}
    metrics = {
        "variant": args.variant,
        "status": "completed",
        "parameter_count": param_count,
        "device": str(device),
        "cuda_device_count": torch.cuda.device_count(),
        "seq_length": args.seq_length,
        "words_per_example": args.words_per_example,
        "batch_size": args.batch_size,
        "steps": total_steps,
        "lr_schedule_total_steps": schedule_total_steps,
        "warmup_steps": warmup_steps,
        "word_exposure": cumulative_words,
        "requested_max_word_exposure": args.max_word_exposure,
        "example_pool_words_actual": pool_words,
        "order_mode": args.order_mode,
        "loss_first": loss_values[0] if loss_values else None,
        "loss_last": loss_values[-1] if loss_values else None,
        "loss_mean_last_20": sum(loss_values[-20:]) / min(20, len(loss_values)) if loss_values else None,
        "elapsed_sec": time.time() - start_time,
        "hf_model_dir": str(out / "hf_model"),
        "checkpoint_chck_1M_dir": str(out / "hf_model" / "chck_1M"),
        "saved_checkpoints": saved_checkpoints,
        "router_summary_last": router_summary,
        "morph_summary_last": morph_summary,
        "memory_summary_last": memory_summary,
        "surface_summary_last": surface_summary,
        **model_extra,
    }
    metrics_text = json.dumps(metrics, indent=2, ensure_ascii=False)
    (out / "metrics.json").write_text(metrics_text, encoding="utf-8")
    (out / "scientific_metrics.json").write_text(metrics_text, encoding="utf-8")
    print(json.dumps({"event": "completed", "metrics": metrics}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
