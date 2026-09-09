#!/usr/bin/env python3
"""Minimal official-compatible masked-LM trainer for BabyLM Strict-Small.

Separate from the verified causal trainer (`babylm_compare_train.py`) so the
masked-objective route cannot destabilize the causal path. Reuses the same
official corpus download, exact whitespace-word exposure accounting, portable
tokenizer, and chck_<N>M checkpoint conventions. Supports token-level random
masking and whole-word masking, plus a simple sequence-length schedule hook.

Model: BERT-style masked encoder via HF `BertForMaskedLM` (loads with
`AutoModelForMaskedLM`), evaluated with the official `mlm` backend.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from huggingface_hub import snapshot_download
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerFast,
    BertConfig,
    BertForMaskedLM,
    DebertaV2Config,
    DebertaV2ForMaskedLM,
    get_cosine_schedule_with_warmup,
)

BASELINE_TOKENIZER_REPO = "BabyLM-community/BabyLM-2026-Baseline-GPT2-Strict"
TRAIN_FILES = [
    "bnc_spoken.train.txt",
    "childes.train.txt",
    "gutenberg.train.txt",
    "open_subtitles.train.txt",
    "simple_wiki.train.txt",
    "switchboard.train.txt",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
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
            "path": str(p), "name": name, "bytes": p.stat().st_size,
            "sha256": sha256_file(p), "whitespace_words": count_words_in_file(p),
        })
    return local, manifest_files


@dataclass
class Example:
    text: str
    words: int
    example_id: int = -1
    source: str = ""


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


def load_examples_jsonl(path: Path, selected_words: int) -> tuple[list[Example], int, int, list[dict]]:
    """Load pre-materialized examples in exact order without shuffle or repacking.

    Used for controlled rewrite-pair experiments where pair-adjacent and
    pair-shuffled arms must retain the same sentence multiset and avoid partial
    examples. Each line must contain text and words; optional example_id/source
    are preserved in the manifest.
    """
    examples: list[Example] = []
    total_file_words = 0
    total_rows = 0
    sample_rows: list[dict] = []
    selected = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            total_rows += 1
            text = str(obj["text"])
            words = int(obj.get("words", len(text.split())))
            actual = len(text.split())
            if words != actual:
                raise RuntimeError(f"JSONL word-count mismatch at row {total_rows}: field={words} actual={actual}")
            total_file_words += words
            if selected < selected_words:
                if selected + words > selected_words:
                    raise RuntimeError(
                        f"JSONL selection would require partial example at row {total_rows}: "
                        f"selected={selected} words={words} target={selected_words}"
                    )
                ex_id = int(obj.get("example_id", total_rows - 1))
                source = str(obj.get("source", "example_jsonl"))
                examples.append(Example(text=text, words=words, example_id=ex_id, source=source))
                selected += words
                if len(sample_rows) < 10:
                    keep = {k: obj[k] for k in obj.keys() if k != "text"}
                    sample_rows.append(keep)
    if selected != selected_words:
        raise RuntimeError(f"JSONL selected words {selected} != target {selected_words}")
    return examples, total_file_words, total_rows, sample_rows


def reset_all_rng(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_portable_tokenizer(tokenizer_path: str = "") -> PreTrainedTokenizerFast:
    """Load the baseline tokenizer or a local tokenizer as a portable fast tokenizer."""
    if tokenizer_path:
        base = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    else:
        base = AutoTokenizer.from_pretrained(BASELINE_TOKENIZER_REPO, revision="main", use_fast=True)
    backend = getattr(base, "_tokenizer", None) or getattr(base, "backend_tokenizer", None)
    if backend is None:
        raise RuntimeError(f"Tokenizer at {tokenizer_path or BASELINE_TOKENIZER_REPO} has no fast backend")
    tok = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token=base.bos_token if base.bos_token else "<s>",
        eos_token=base.eos_token if base.eos_token else "</s>",
        unk_token=base.unk_token if base.unk_token else "<unk>",
        pad_token=base.pad_token if base.pad_token else "<pad>",
        mask_token=getattr(base, "mask_token", None) or "<mask>",
        model_max_length=getattr(base, "model_max_length", 1024) or 1024,
    )
    tok.init_kwargs.pop("tokenizer_class", None)
    tok.init_kwargs["tokenizer_class"] = "PreTrainedTokenizerFast"
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "<pad>"})
    if tok.mask_token is None:
        tok.add_special_tokens({"mask_token": "<mask>"})
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


def is_word_start(token_str: str) -> bool:
    # GPT-2 byte-level BPE marks a new word with the leading 'Ġ' (space) byte.
    return token_str.startswith("Ġ") or token_str.startswith("▁")


ENTITY_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "for", "to", "in", "on", "at", "by", "with", "from", "as",
    "is", "are", "was", "were", "be", "been", "being", "do", "does", "did", "have", "has", "had", "will", "would",
    "can", "could", "should", "may", "might", "must", "not", "no", "yes", "this", "that", "these", "those", "there",
    "here", "it", "its", "they", "them", "their", "he", "she", "his", "her", "we", "us", "you", "i", "me", "my",
    "your", "our", "what", "which", "who", "whom", "whose", "when", "where", "why", "how", "than", "then", "so",
    "if", "because", "about", "into", "up", "down", "out", "over", "under", "again", "very", "just", "also"
}


def normalize_entity_piece(piece: str) -> str:
    piece = str(piece).replace("Ġ", "").replace("▁", "").replace("Ċ", " ")
    piece = re.sub(r"[^A-Za-z0-9'-]+", "", piece)
    return piece.lower().strip("-'")


def entity_candidate_text(input_ids_b: torch.Tensor, positions: list[int], tokenizer) -> str:
    parts = [normalize_entity_piece(tokenizer.convert_ids_to_tokens(int(input_ids_b[p].item()))) for p in positions]
    return "".join(parts).strip("-'")


def is_entity_candidate_text(text: str) -> bool:
    if not text or text in ENTITY_STOPWORDS:
        return False
    if len(text) < 3:
        return False
    if not re.search(r"[a-z]", text):
        return False
    if text.isdigit():
        return False
    return True


def compute_entity_consistency_loss(
    hidden: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    word_group: torch.Tensor,
    labels: torch.Tensor,
    tokenizer,
    projection: torch.nn.Module,
    mode: str,
    lambda_weight: float,
    temperature: float,
    min_group_gap: int,
    max_pairs_per_example: int,
    max_pairs_per_batch: int,
    max_candidates_per_batch: int,
    rng_seed: int,
) -> tuple[torch.Tensor, dict]:
    """Entity mention consistency auxiliary loss over unmasked repeated word groups.

    The ordinary WWM loss remains unchanged. This auxiliary objective only reads
    last-layer states at unmasked exact repeated content-word groups and adds an
    InfoNCE term. `mode='shuffled_pair'` preserves pair count and target-group
    distribution but permutes targets away from their true anchors.
    """
    device = hidden.device
    zero = hidden.new_tensor(0.0)
    if mode == "none" or lambda_weight <= 0:
        return zero, {"entity_aux_active": 0, "entity_aux_pairs": 0, "entity_aux_candidates": 0, "entity_aux_loss": 0.0, "entity_aux_mean_gap": 0.0}

    input_cpu = input_ids.detach().cpu()
    group_cpu = word_group.detach().cpu()
    attn_cpu = attention_mask.detach().cpu()
    labels_cpu = labels.detach().cpu()
    bsz, seq = input_ids.shape
    candidates: list[dict] = []
    by_example_text: dict[tuple[int, str], list[int]] = defaultdict(list)

    for b in range(bsz):
        group_positions: dict[int, list[int]] = defaultdict(list)
        for p in range(seq):
            if int(attn_cpu[b, p].item()) == 0:
                continue
            g = int(group_cpu[b, p].item())
            if g < 0:
                continue
            group_positions[g].append(p)
        for g in sorted(group_positions):
            pos = group_positions[g]
            # Positive mention representations are intentionally not MLM targets.
            if any(int(labels_cpu[b, p].item()) != -100 for p in pos):
                continue
            txt = entity_candidate_text(input_cpu[b], pos, tokenizer)
            if not is_entity_candidate_text(txt):
                continue
            idx = len(candidates)
            candidates.append({"b": b, "gid": g, "positions": pos, "text": txt})
            by_example_text[(b, txt)].append(idx)

    raw_pairs: list[tuple[int, int, int]] = []
    active_examples = 0
    for b in range(bsz):
        used = 0
        for (bb, txt), idxs in list(by_example_text.items()):
            if bb != b or len(idxs) < 2:
                continue
            idxs = sorted(idxs, key=lambda i: candidates[i]["gid"])
            for i, a in enumerate(idxs[:-1]):
                if used >= max_pairs_per_example:
                    break
                for p in idxs[i + 1:]:
                    gap = int(candidates[p]["gid"] - candidates[a]["gid"])
                    if gap >= min_group_gap:
                        raw_pairs.append((a, p, gap))
                        used += 1
                        break
            if used >= max_pairs_per_example:
                break
        if used > 0:
            active_examples += 1

    rng = random.Random(rng_seed)
    if len(raw_pairs) > max_pairs_per_batch:
        rng.shuffle(raw_pairs)
        raw_pairs = raw_pairs[:max_pairs_per_batch]
    if not raw_pairs or len(candidates) < 2:
        return zero, {"entity_aux_active": 0, "entity_aux_pairs": 0, "entity_aux_candidates": len(candidates), "entity_aux_loss": 0.0, "entity_aux_mean_gap": 0.0}

    if mode == "shuffled_pair":
        targets = [p for _, p, _ in raw_pairs]
        rng.shuffle(targets)
        shuffled = []
        n = len(targets)
        for i, (a, p, gap) in enumerate(raw_pairs):
            t = targets[i]
            if n > 1:
                for off in range(n):
                    cand_t = targets[(i + off) % n]
                    if cand_t != a and candidates[cand_t]["text"] != candidates[a]["text"]:
                        t = cand_t
                        break
            if t != a:
                shuffled.append((a, t, gap))
        raw_pairs = shuffled
    elif mode != "consistency":
        raise ValueError(f"unknown entity_aux_mode {mode}")

    if not raw_pairs:
        return zero, {"entity_aux_active": 0, "entity_aux_pairs": 0, "entity_aux_candidates": len(candidates), "entity_aux_loss": 0.0, "entity_aux_mean_gap": 0.0}

    required = set()
    for a, p, _ in raw_pairs:
        required.add(a); required.add(p)
    extras = [i for i in range(len(candidates)) if i not in required]
    rng.shuffle(extras)
    keep = list(required)
    if len(keep) < max_candidates_per_batch:
        keep.extend(extras[:max_candidates_per_batch - len(keep)])
    keep = keep[:max_candidates_per_batch]
    keep_set = set(keep)
    raw_pairs = [(a, p, g) for a, p, g in raw_pairs if a in keep_set and p in keep_set]
    if not raw_pairs or len(keep) < 2:
        return zero, {"entity_aux_active": 0, "entity_aux_pairs": 0, "entity_aux_candidates": len(keep), "entity_aux_loss": 0.0, "entity_aux_mean_gap": 0.0}
    new_index = {old: i for i, old in enumerate(keep)}

    reps = []
    for old in keep:
        c = candidates[old]
        pos_t = torch.tensor(c["positions"], dtype=torch.long, device=device)
        reps.append(hidden[int(c["b"]), pos_t].mean(dim=0))
    z = torch.stack(reps, dim=0)
    z = F.normalize(projection(z), dim=-1)
    anchor_idx = torch.tensor([new_index[a] for a, _, _ in raw_pairs], dtype=torch.long, device=device)
    target_idx = torch.tensor([new_index[p] for _, p, _ in raw_pairs], dtype=torch.long, device=device)
    logits = torch.matmul(z[anchor_idx], z.t()) / max(1e-6, float(temperature))
    logits[torch.arange(logits.shape[0], device=device), anchor_idx] = -1e4
    aux_loss = F.cross_entropy(logits, target_idx)
    mean_gap = sum(g for _, _, g in raw_pairs) / max(1, len(raw_pairs))
    tele = {
        "entity_aux_active": 1,
        "entity_aux_pairs": len(raw_pairs),
        "entity_aux_candidates": len(keep),
        "entity_aux_loss": float(aux_loss.detach().cpu()),
        "entity_aux_mean_gap": float(mean_gap),
        "entity_aux_active_examples": active_examples,
    }
    return aux_loss, tele


class MaskedChunkDataset(Dataset):
    """Tokenizes each example and precomputes word-start groups for WWM."""

    def __init__(self, examples: list[Example], tokenizer, seq_length: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.special_ids = set(tokenizer.all_special_ids)
        # Precompute id -> is_word_start using token strings.
        self._word_start = {}

    def _word_start_flag(self, token_id: int) -> bool:
        v = self._word_start.get(token_id)
        if v is None:
            s = self.tokenizer.convert_ids_to_tokens(int(token_id))
            v = bool(s is not None and is_word_start(str(s)))
            self._word_start[token_id] = v
        return v

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex.text, add_special_tokens=False, truncation=True,
            max_length=self.seq_length, padding="max_length", return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        # Word-group id per position for WWM; each real token that is a word-start
        # opens a new group. Padding/specials get group -1.
        group = torch.full_like(input_ids, -1)
        gid = -1
        for i in range(input_ids.shape[0]):
            if attention_mask[i] == 0:
                continue
            tid = int(input_ids[i])
            if tid in self.special_ids:
                continue
            if gid < 0 or self._word_start_flag(tid) or i == 0:
                gid += 1
            group[i] = gid
        return {"input_ids": input_ids, "attention_mask": attention_mask, "word_group": group, "words": ex.words}


def collate(batch: list[dict]) -> dict:
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "word_group": torch.stack([x["word_group"] for x in batch]),
        "words": torch.tensor([x["words"] for x in batch], dtype=torch.long),
    }


def apply_masking(input_ids: torch.Tensor, attention_mask: torch.Tensor, word_group: torch.Tensor,
                  tokenizer, mask_mode: str, mask_prob: float, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (masked_input_ids, labels). labels=-100 where not predicted."""
    device = input_ids.device
    labels = input_ids.clone()
    bsz, seq = input_ids.shape
    mask_token_id = tokenizer.mask_token_id
    special_ids = torch.tensor(sorted(tokenizer.all_special_ids), device=device)
    candidate = attention_mask.bool() & ~torch.isin(input_ids, special_ids)
    select = torch.zeros_like(candidate)
    if mask_mode == "token":
        probs = torch.rand(bsz, seq, generator=gen, device=device)
        select = candidate & (probs < mask_prob)
    elif mask_mode == "wwm":
        for b in range(bsz):
            groups = word_group[b]
            valid_groups = torch.unique(groups[groups >= 0])
            if valid_groups.numel() == 0:
                continue
            gp = torch.rand(valid_groups.numel(), generator=gen, device=device)
            chosen = valid_groups[gp < mask_prob]
            if chosen.numel() == 0:
                continue
            sel_b = torch.isin(groups, chosen) & candidate[b]
            select[b] = sel_b
    else:
        raise ValueError(f"unknown mask_mode {mask_mode}")
    if select.sum() == 0:
        # ensure at least one prediction to avoid nan loss
        flat = candidate.view(-1).nonzero(as_tuple=False)
        if flat.numel() > 0:
            select.view(-1)[flat[0, 0]] = True
    labels[~select] = -100
    masked_inputs = input_ids.clone()
    # 80% [MASK], 10% random, 10% keep (standard BERT recipe)
    r = torch.rand(bsz, seq, generator=gen, device=device)
    mask_tok = select & (r < 0.8)
    rand_tok = select & (r >= 0.8) & (r < 0.9)
    masked_inputs[mask_tok] = mask_token_id
    if rand_tok.any():
        rand_ids = torch.randint(0, len(tokenizer), (int(rand_tok.sum()),), generator=gen, device=device)
        masked_inputs[rand_tok] = rand_ids
    return masked_inputs, labels


def seq_length_for_progress(frac: float, schedule: list[tuple[float, int]], default_len: int) -> int:
    length = default_len
    for thresh, L in schedule:
        if frac >= thresh:
            length = L
    return length


def summarize_tokenization_coupling(examples: list[Example], tokenizer, max_seq_length: int, sample_limit: int = 0) -> dict:
    """Record tokenizer-linked changes on the exact raw examples used for training.

    A tokenizer-granularity experiment changes sequence length, truncation,
    whole-word group structure, expected WWM supervised-token density, embedding
    capacity, and compute. This deterministic summary makes those linked changes
    visible instead of treating vocab size as an isolated scalar.
    """
    n = len(examples) if sample_limit <= 0 else min(len(examples), sample_limit)
    selected = examples[:n]
    total_words = 0
    total_untruncated_tokens = 0
    total_kept_tokens = 0
    total_groups_kept = 0
    total_truncated_tokens = 0
    truncated_examples = 0
    max_untruncated_tokens = 0
    max_kept_tokens = 0
    hist = {"<=64": 0, "65-128": 0, "129-256": 0, ">256": 0}
    ds = MaskedChunkDataset(selected, tokenizer, max_seq_length)
    for i, ex in enumerate(selected):
        enc_full = tokenizer(ex.text, add_special_tokens=False, truncation=False)
        full_len = len(enc_full["input_ids"])
        item = ds[i]
        kept = int(item["attention_mask"].sum().item())
        groups = item["word_group"]
        group_count = int(torch.unique(groups[groups >= 0]).numel())
        total_words += ex.words
        total_untruncated_tokens += full_len
        total_kept_tokens += kept
        total_groups_kept += group_count
        max_untruncated_tokens = max(max_untruncated_tokens, full_len)
        max_kept_tokens = max(max_kept_tokens, kept)
        if full_len > max_seq_length:
            truncated_examples += 1
            total_truncated_tokens += full_len - max_seq_length
        if full_len <= 64:
            hist["<=64"] += 1
        elif full_len <= 128:
            hist["65-128"] += 1
        elif full_len <= 256:
            hist["129-256"] += 1
        else:
            hist[">256"] += 1
    denom = max(1, n)
    word_denom = max(1, total_words)
    return {
        "num_examples_summarized": n,
        "total_words_summarized": total_words,
        "total_untruncated_tokens": total_untruncated_tokens,
        "total_kept_tokens_at_max_seq_length": total_kept_tokens,
        "total_word_groups_kept_at_max_seq_length": total_groups_kept,
        "mean_untruncated_tokens_per_example": total_untruncated_tokens / denom,
        "mean_kept_tokens_per_example": total_kept_tokens / denom,
        "mean_word_groups_kept_per_example": total_groups_kept / denom,
        "untruncated_tokens_per_whitespace_word": total_untruncated_tokens / word_denom,
        "kept_tokens_per_whitespace_word": total_kept_tokens / word_denom,
        "word_groups_kept_per_whitespace_word": total_groups_kept / word_denom,
        "expected_wwm_selected_word_groups_at_mask_prob_0p15": 0.15 * total_groups_kept,
        "expected_wwm_predicted_tokens_at_mask_prob_0p15_if_selected_groups_cover_tokens": 0.15 * total_kept_tokens,
        "truncated_examples_at_max_seq_length": truncated_examples,
        "truncated_example_fraction": truncated_examples / denom,
        "total_tokens_lost_to_truncation": total_truncated_tokens,
        "max_untruncated_tokens": max_untruncated_tokens,
        "max_kept_tokens": max_kept_tokens,
        "untruncated_length_histogram": hist,
    }


def build_model(args: argparse.Namespace, tokenizer):
    max_pos = max(args.max_position_embeddings, args.max_seq_length + 8)
    if args.model_type == "bert":
        cfg = BertConfig(
            vocab_size=len(tokenizer),
            hidden_size=args.hidden_size,
            num_hidden_layers=args.n_layer,
            num_attention_heads=args.n_head,
            intermediate_size=args.hidden_size * args.ffn_mult,
            max_position_embeddings=max_pos,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            position_embedding_type="absolute",
        )
        return BertForMaskedLM(cfg)
    if args.model_type == "deberta_v2":
        pos_att_type = [x.strip() for x in args.deberta_pos_att_type.split(",") if x.strip()]
        cfg = DebertaV2Config(
            vocab_size=len(tokenizer),
            hidden_size=args.hidden_size,
            num_hidden_layers=args.n_layer,
            num_attention_heads=args.n_head,
            intermediate_size=args.hidden_size * args.ffn_mult,
            max_position_embeddings=max_pos,
            max_relative_positions=args.max_relative_positions,
            position_buckets=args.position_buckets,
            relative_attention=(args.deberta_relative_attention.lower() == "true"),
            pos_att_type=pos_att_type,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        return DebertaV2ForMaskedLM(cfg)
    raise ValueError(f"unknown model_type {args.model_type}")


def save_hf_checkpoint(model, tokenizer, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dst, safe_serialization=True)
    tokenizer.save_pretrained(dst)
    force_portable_tokenizer_config(dst)


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_id", default="BabyLM-community/BabyLM-2026-Strict-Small")
    p.add_argument("--dataset_revision", default="c92ab16b4f08858304b0815706065b3354d8fc0a")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_word_exposure", type=int, default=1_000_000)
    p.add_argument("--example_pool_words", type=int, default=10_000_000)
    p.add_argument("--checkpoint_words", type=int, default=1_000_000)
    p.add_argument("--words_per_example", type=int, default=160)
    p.add_argument("--example_jsonl", default="", help="Optional pre-materialized examples JSONL consumed exactly in file order; bypasses official corpus selection")
    p.add_argument("--example_jsonl_label", default="", help="Human-readable label for JSONL data condition")
    p.add_argument("--example_jsonl_meta", default="", help="Optional metadata JSON file for pre-materialized examples")
    p.add_argument("--tokenizer_path", default="", help="Optional local/HF tokenizer path; empty uses BabyLM baseline 16k tokenizer")
    p.add_argument("--tokenizer_label", default="baseline16k", help="Human-readable tokenizer condition recorded in metrics")
    p.add_argument("--tokenization_summary_limit", type=int, default=0, help="0 summarizes all selected examples; positive limits for smokes")
    p.add_argument("--mask_mode", choices=["token", "wwm"], default="token")
    p.add_argument("--mask_prob", type=float, default=0.15)
    p.add_argument("--entity_aux_mode", choices=["none", "consistency", "shuffled_pair"], default="none")
    p.add_argument("--entity_aux_lambda", type=float, default=0.05)
    p.add_argument("--entity_aux_temperature", type=float, default=0.10)
    p.add_argument("--entity_projection_dim", type=int, default=128)
    p.add_argument("--entity_min_group_gap", type=int, default=4)
    p.add_argument("--entity_max_pairs_per_example", type=int, default=4)
    p.add_argument("--entity_max_pairs_per_batch", type=int, default=512)
    p.add_argument("--entity_max_candidates_per_batch", type=int, default=2048)
    p.add_argument("--seq_length", type=int, default=128)
    p.add_argument("--max_seq_length", type=int, default=256)
    p.add_argument("--max_position_embeddings", type=int, default=512, help="BERT config capacity for official evaluation inputs; decoupled from training truncation length")
    p.add_argument("--seq_len_schedule", default="", help="e.g. '0.0:64,0.5:128,0.8:256'")
    p.add_argument("--model_type", choices=["bert", "deberta_v2"], default="bert")
    p.add_argument("--position_buckets", type=int, default=256)
    p.add_argument("--max_relative_positions", type=int, default=256)
    p.add_argument("--deberta_pos_att_type", default="p2c,c2p")
    p.add_argument("--deberta_relative_attention", choices=["true", "false"], default="true")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_fraction", type=float, default=0.05)
    p.add_argument("--lr_total_steps", type=int, default=0)
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=8)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--ffn_mult", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--extra_init_seed", type=int, default=-1)
    p.add_argument("--train_rng_seed", type=int, default=-1)
    p.add_argument("--log_every", type=int, default=50)
    return p.parse_args()


def main() -> None:
    args = build_args()
    start_time = time.time()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pool_words = args.example_pool_words
    selected_words = args.max_word_exposure
    if pool_words < selected_words:
        pool_words = selected_words

    tokenizer = make_portable_tokenizer(args.tokenizer_path)
    example_jsonl_meta: dict = {}

    if args.example_jsonl:
        jsonl_path = Path(args.example_jsonl)
        examples, jsonl_total_words, jsonl_total_rows, sample_rows = load_examples_jsonl(jsonl_path, selected_words)
        actual_words = sum(ex.words for ex in examples)
        pool_words = jsonl_total_words
        manifest_files = [{
            "path": str(jsonl_path),
            "name": jsonl_path.name,
            "bytes": jsonl_path.stat().st_size,
            "sha256": sha256_file(jsonl_path),
            "whitespace_words": jsonl_total_words,
            "rows": jsonl_total_rows,
        }]
        total_words = jsonl_total_words
        if args.example_jsonl_meta:
            meta_path = Path(args.example_jsonl_meta)
            example_jsonl_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            example_jsonl_meta["meta_path"] = str(meta_path)
            example_jsonl_meta["meta_sha256"] = sha256_file(meta_path)
        example_selection_metadata = {
            "data_source_type": "example_jsonl",
            "example_jsonl": str(jsonl_path),
            "example_jsonl_label": args.example_jsonl_label,
            "example_jsonl_total_words": jsonl_total_words,
            "example_jsonl_total_rows": jsonl_total_rows,
            "example_jsonl_sample_rows_without_text": sample_rows,
            "example_jsonl_meta": example_jsonl_meta,
        }
    else:
        raw_dir, manifest_files = download_dataset(args, out)
        files = [raw_dir / n for n in TRAIN_FILES]
        total_words = sum(f["whitespace_words"] for f in manifest_files)

        # Full-cycle support: strict-small permits up to 10 epochs / 100M word exposures
        # over the 10M-word official corpus. The mechanism-test trainer sampled without
        # replacement from one pool and therefore could not represent exposures beyond
        # the unique corpus. This clone builds epoch-wise shuffled passes over the
        # official pool until selected_words is reached. Every repeated word exposure
        # is counted in the training budget and recorded in the manifest.
        if pool_words > total_words:
            pool_words = total_words
        pool_examples = list(iter_examples(files, pool_words, args.words_per_example))
        for i, ex in enumerate(pool_examples):
            ex.example_id = i
        pool_actual = sum(ex.words for ex in pool_examples)
        if pool_actual != pool_words:
            raise RuntimeError(f"pool word mismatch {pool_actual} vs {pool_words}")
        if selected_words > total_words * 10:
            raise RuntimeError(f"requested exposure {selected_words} exceeds 10 official epochs ({total_words * 10})")
        examples: list[Example] = []
        actual_words = 0
        epoch = 0
        epoch_metadata = []
        while actual_words < selected_words:
            epoch_examples = list(pool_examples)
            shuffle_seed = args.seed + 1000003 * epoch
            random.Random(shuffle_seed).shuffle(epoch_examples)
            before = actual_words
            epoch_take_examples = 0
            for ex in epoch_examples:
                if actual_words >= selected_words:
                    break
                source = f"epoch{epoch + 1}::{ex.source}"
                if actual_words + ex.words <= selected_words:
                    examples.append(Example(ex.text, ex.words, example_id=ex.example_id, source=source))
                    actual_words += ex.words
                    epoch_take_examples += 1
                else:
                    take = selected_words - actual_words
                    if take > 0:
                        examples.append(Example(" ".join(ex.text.split()[:take]), take, example_id=ex.example_id, source=source))
                        actual_words += take
                        epoch_take_examples += 1
                    break
            epoch_metadata.append({"epoch_index": epoch + 1, "shuffle_seed": shuffle_seed,
                                   "words_added": actual_words - before, "examples_added": epoch_take_examples})
            epoch += 1
        example_selection_metadata = {"data_source_type": "official_corpus_fullcycle",
                                      "selection_epochs": epoch_metadata,
                                      "unique_official_pool_words": pool_words,
                                      "total_official_corpus_words": total_words}

    if actual_words != selected_words:
        raise RuntimeError(f"word selection mismatch {actual_words} vs {selected_words}")

    source_words: dict[str, int] = {}
    for ex in examples:
        source_words[ex.source] = source_words.get(ex.source, 0) + ex.words
    (out / "example_order_manifest.json").write_text(json.dumps({
        "seed": args.seed,
        "example_pool_words_actual": pool_words,
        "selected_for_training_words": actual_words,
        "words_per_example": args.words_per_example,
        "num_consumed_examples": len(examples),
        "source_words_consumed": source_words,
        "consumed_example_ids_in_order": [ex.example_id for ex in examples],
        "mask_mode": args.mask_mode,
        "mask_prob": args.mask_prob,
        "tokenizer_label": args.tokenizer_label,
        "tokenizer_path": args.tokenizer_path,
        "tokenizer_vocab_size": len(tokenizer),
        **example_selection_metadata,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    tokenization_summary = summarize_tokenization_coupling(examples, tokenizer, args.max_seq_length, args.tokenization_summary_limit)
    (out / "tokenization_coupling_summary.json").write_text(json.dumps(tokenization_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    dataset = MaskedChunkDataset(examples, tokenizer, args.max_seq_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
                        num_workers=2, pin_memory=torch.cuda.is_available())

    if args.extra_init_seed >= 0:
        reset_all_rng(args.extra_init_seed)
    if args.max_position_embeddings < args.max_seq_length + 8:
        print(json.dumps({"event": "max_position_embeddings_raised", "requested": args.max_position_embeddings, "minimum": args.max_seq_length + 8}), flush=True)
        args.max_position_embeddings = args.max_seq_length + 8
    model = build_model(args, tokenizer)
    if args.train_rng_seed >= 0:
        reset_all_rng(args.train_rng_seed)
    param_count = sum(p.numel() for p in model.parameters())
    input_embedding_params = model.get_input_embeddings().weight.numel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    entity_projection = None
    entity_aux_params = 0
    if args.entity_aux_mode != "none":
        entity_projection = torch.nn.Sequential(
            torch.nn.Linear(args.hidden_size, args.entity_projection_dim),
            torch.nn.LayerNorm(args.entity_projection_dim),
        ).to(device)
        entity_aux_params = sum(p.numel() for p in entity_projection.parameters())
    optim_params = list(model.parameters()) + (list(entity_projection.parameters()) if entity_projection is not None else [])
    optim = torch.optim.AdamW(optim_params, lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    train_parameter_count = param_count + entity_aux_params
    total_steps = len(loader)
    schedule_total = args.lr_total_steps if args.lr_total_steps and args.lr_total_steps > 0 else total_steps
    if schedule_total < total_steps:
        raise RuntimeError(f"lr_total_steps {schedule_total} < actual steps {total_steps}")
    warmup = max(1, int(schedule_total * args.warmup_fraction))
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=warmup, num_training_steps=schedule_total)

    seq_schedule: list[tuple[float, int]] = []
    if args.seq_len_schedule:
        for part in args.seq_len_schedule.split(","):
            t, L = part.split(":")
            seq_schedule.append((float(t), int(L)))
        seq_schedule.sort()

    gen = torch.Generator(device=device)
    gen.manual_seed(args.train_rng_seed if args.train_rng_seed >= 0 else args.seed)

    log_path = out / "training_log.jsonl"
    cumulative_words = 0
    loss_values: list[float] = []
    masked_token_values: list[int] = []
    seq_len_values: list[int] = []
    saved_checkpoints: list[dict] = []
    entity_aux_totals: dict[str, float] = {"entity_aux_pairs": 0.0, "entity_aux_candidates": 0.0, "entity_aux_active": 0.0, "entity_aux_loss": 0.0, "entity_aux_active_examples": 0.0, "entity_aux_mean_gap_weighted_sum": 0.0}
    next_ckpt = args.checkpoint_words if args.checkpoint_words and args.checkpoint_words > 0 else None
    model.train()
    if entity_projection is not None:
        entity_projection.train()
    with log_path.open("w", encoding="utf-8") as logf:
        for step, batch in enumerate(loader, 1):
            words = int(batch.pop("words").sum().item())
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            word_group = batch["word_group"].to(device, non_blocking=True)
            frac = (step - 1) / max(1, schedule_total)
            cur_len = seq_length_for_progress(frac, seq_schedule, args.seq_length) if seq_schedule else args.seq_length
            cur_len = min(cur_len, args.max_seq_length)
            input_ids = input_ids[:, :cur_len].contiguous()
            attention_mask = attention_mask[:, :cur_len].contiguous()
            word_group = word_group[:, :cur_len].contiguous()
            masked_inputs, labels = apply_masking(input_ids, attention_mask, word_group, tokenizer, args.mask_mode, args.mask_prob, gen)
            optim.zero_grad(set_to_none=True)
            out_model = model(input_ids=masked_inputs, attention_mask=attention_mask, labels=labels, output_hidden_states=(entity_projection is not None))
            mlm_loss = out_model.loss
            if mlm_loss is None:
                raise RuntimeError("model returned no loss")
            entity_aux_loss = labels.new_tensor(0.0, dtype=torch.float32)
            entity_tele = {"entity_aux_active": 0, "entity_aux_pairs": 0, "entity_aux_candidates": 0, "entity_aux_loss": 0.0, "entity_aux_mean_gap": 0.0, "entity_aux_active_examples": 0}
            if entity_projection is not None:
                entity_aux_loss, entity_tele = compute_entity_consistency_loss(
                    out_model.hidden_states[-1], input_ids, attention_mask, word_group, labels, tokenizer,
                    entity_projection, args.entity_aux_mode, args.entity_aux_lambda, args.entity_aux_temperature,
                    args.entity_min_group_gap, args.entity_max_pairs_per_example,
                    args.entity_max_pairs_per_batch, args.entity_max_candidates_per_batch,
                    (args.train_rng_seed if args.train_rng_seed >= 0 else args.seed) + step * 1000003,
                )
            loss = mlm_loss + (args.entity_aux_lambda * entity_aux_loss if entity_projection is not None else entity_aux_loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(optim_params, 1.0)
            optim.step()
            sched.step()
            cumulative_words += words
            loss_float = float(loss.detach().cpu())
            loss_values.append(loss_float)
            n_pred = int((labels != -100).sum().item())
            masked_token_values.append(n_pred)
            seq_len_values.append(cur_len)
            rec = {"step": step, "loss": loss_float, "loss_mlm": float(mlm_loss.detach().cpu()),
                   "loss_entity_aux": float(entity_aux_loss.detach().cpu()) if entity_projection is not None else 0.0,
                   "lr": float(sched.get_last_lr()[0]),
                   "batch_words": words, "cumulative_word_exposure": cumulative_words,
                   "seq_len": cur_len, "masked_tokens": n_pred, "elapsed_sec": time.time() - start_time,
                   **entity_tele}
            entity_aux_totals["entity_aux_pairs"] += float(entity_tele.get("entity_aux_pairs", 0.0))
            entity_aux_totals["entity_aux_candidates"] += float(entity_tele.get("entity_aux_candidates", 0.0))
            entity_aux_totals["entity_aux_active"] += float(entity_tele.get("entity_aux_active", 0.0))
            entity_aux_totals["entity_aux_loss"] += float(entity_tele.get("entity_aux_loss", 0.0))
            entity_aux_totals["entity_aux_active_examples"] += float(entity_tele.get("entity_aux_active_examples", 0.0))
            entity_aux_totals["entity_aux_mean_gap_weighted_sum"] += float(entity_tele.get("entity_aux_mean_gap", 0.0)) * float(entity_tele.get("entity_aux_pairs", 0.0))
            logf.write(json.dumps(rec) + "\n")
            if step == 1 or step % args.log_every == 0 or step == total_steps:
                print(json.dumps({"event": "train", **rec}), flush=True)
            while next_ckpt is not None and cumulative_words >= next_ckpt and next_ckpt <= args.max_word_exposure:
                if next_ckpt < 1_000_000:
                    name = "chck_1M"
                elif next_ckpt % 1_000_000 == 0:
                    name = f"chck_{next_ckpt // 1_000_000}M"
                else:
                    name = f"chck_{next_ckpt}w"
                cp = out / "hf_model" / name
                save_hf_checkpoint(model, tokenizer, cp)
                saved_checkpoints.append({"name": name, "target_word_exposure": next_ckpt,
                                          "actual_cumulative_word_exposure": cumulative_words, "path": str(cp)})
                print(json.dumps({"event": "checkpoint_saved", "name": name, "cum_words": cumulative_words}), flush=True)
                next_ckpt += args.checkpoint_words

    save_hf_checkpoint(model, tokenizer, out / "hf_model")
    if entity_projection is not None:
        torch.save(entity_projection.state_dict(), out / "entity_projection_train_only.pt")
        (out / "entity_aux_config.json").write_text(json.dumps({
            "entity_aux_mode": args.entity_aux_mode,
            "entity_aux_lambda": args.entity_aux_lambda,
            "entity_aux_temperature": args.entity_aux_temperature,
            "entity_projection_dim": args.entity_projection_dim,
            "entity_min_group_gap": args.entity_min_group_gap,
            "entity_max_pairs_per_example": args.entity_max_pairs_per_example,
            "entity_max_pairs_per_batch": args.entity_max_pairs_per_batch,
            "entity_max_candidates_per_batch": args.entity_max_candidates_per_batch,
            "note": "Projection head is train-only and intentionally not saved inside hf_model for evaluation."
        }, indent=2, ensure_ascii=False), encoding="utf-8")
    if not saved_checkpoints:
        cp = out / "hf_model" / "chck_1M"
        save_hf_checkpoint(model, tokenizer, cp)
        saved_checkpoints.append({"name": "chck_1M", "target_word_exposure": args.checkpoint_words,
                                  "actual_cumulative_word_exposure": cumulative_words, "path": str(cp)})

    entity_aux_summary = {
        **entity_aux_totals,
        "entity_aux_mean_loss_per_active_step": (entity_aux_totals["entity_aux_loss"] / max(1.0, entity_aux_totals["entity_aux_active"])),
        "entity_aux_mean_gap_per_pair": (entity_aux_totals["entity_aux_mean_gap_weighted_sum"] / max(1.0, entity_aux_totals["entity_aux_pairs"])),
    }
    metrics = {
        "variant": f"masked_{args.mask_mode}",
        "backend": "mlm",
        "model_family": model.__class__.__name__,
        "model_type": args.model_type,
        "parameter_count": param_count,
        "train_parameter_count_including_aux_head": train_parameter_count,
        "entity_aux_parameter_count": entity_aux_params,
        "embedding_parameter_count": input_embedding_params,
        "non_embedding_parameter_count": param_count - input_embedding_params,
        "vocab_size": len(tokenizer),
        "tokenizer_label": args.tokenizer_label,
        "tokenizer_path": args.tokenizer_path,
        "word_exposure": cumulative_words,
        "example_pool_words_actual": pool_words,
        "selected_for_training_words": actual_words,
        "loss_first": loss_values[0] if loss_values else None,
        "loss_last": loss_values[-1] if loss_values else None,
        "lr_schedule_total_steps": schedule_total,
        "actual_training_steps": total_steps,
        "mask_mode": args.mask_mode,
        "mask_prob": args.mask_prob,
        "masked_tokens_total": sum(masked_token_values),
        "entity_aux_mode": args.entity_aux_mode,
        "entity_aux_lambda": args.entity_aux_lambda,
        "entity_aux_temperature": args.entity_aux_temperature,
        "entity_projection_dim": args.entity_projection_dim,
        "entity_min_group_gap": args.entity_min_group_gap,
        "entity_max_pairs_per_example": args.entity_max_pairs_per_example,
        "entity_max_pairs_per_batch": args.entity_max_pairs_per_batch,
        "entity_max_candidates_per_batch": args.entity_max_candidates_per_batch,
        "entity_aux_summary": entity_aux_summary,
        "masked_tokens_mean_per_step": (sum(masked_token_values) / len(masked_token_values)) if masked_token_values else None,
        "masked_tokens_per_whitespace_word": (sum(masked_token_values) / cumulative_words) if cumulative_words else None,
        "unique_training_seq_lengths": sorted(set(seq_len_values)),
        "tokenization_coupling_summary": tokenization_summary,
        "seq_length": args.seq_length,
        "max_seq_length": args.max_seq_length,
        "max_position_embeddings": max(args.max_position_embeddings, args.max_seq_length + 8),
        "seq_len_schedule": args.seq_len_schedule,
        "hidden_size": args.hidden_size,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "ffn_mult": args.ffn_mult,
        "intermediate_size": args.hidden_size * args.ffn_mult,
        "position_buckets": args.position_buckets,
        "max_relative_positions": args.max_relative_positions,
        "deberta_relative_attention": args.deberta_relative_attention,
        "deberta_pos_att_type": args.deberta_pos_att_type,
        "seed": args.seed,
        "extra_init_seed": args.extra_init_seed,
        "train_rng_seed": args.train_rng_seed,
        "saved_checkpoints": saved_checkpoints,
        "source_words_consumed": source_words,
        **example_selection_metadata,
    }
    (out / "scientific_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "data_manifest.json").write_text(json.dumps({
        "dataset_id": args.dataset_id,
        "dataset_revision": args.dataset_revision,
        "files": manifest_files,
        "total_dataset_whitespace_words_counted": total_words,
        "selected_for_this_run_whitespace_words": actual_words,
        "model_type": args.model_type,
        "tokenizer_label": args.tokenizer_label,
        "tokenizer_path": args.tokenizer_path,
        "tokenizer_vocab_size": len(tokenizer),
        "tokenization_coupling_summary_file": str(out / "tokenization_coupling_summary.json"),
        **example_selection_metadata,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"event": "done", "param_count": param_count, "loss_first": metrics["loss_first"],
                      "loss_last": metrics["loss_last"], "word_exposure": cumulative_words,
                      "checkpoints": [c["name"] for c in saved_checkpoints]}), flush=True)


if __name__ == "__main__":
    main()
