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
import collections
import hashlib
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
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


class Lamb(torch.optim.Optimizer):
    """Reference-style LAMB optimizer for isolated BabyLM component tests.

    This follows the staged torch_optimizer/cybertronai update form by default:
    exp_avg / sqrt(exp_avg_sq) with no debias correction, weight decay added to
    the Adam step before trust-ratio scaling, weight norm clamped to
    [0, trust_clip], and trust_ratio = weight_norm / adam_norm.  Optional
    ``debias=True`` matches the staged reference implementation's optional
    learning-rate debias factor, not a silent custom variant.  The default
    optimizer for the trainer remains AdamW unless ``--optimizer lamb`` is
    explicitly selected.
    """

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-6,
                 weight_decay: float = 0.0, trust_clip: float = 10.0, debias: bool = False):
        if lr <= 0:
            raise ValueError(f"invalid lr {lr}")
        if eps < 0:
            raise ValueError(f"invalid eps {eps}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"invalid betas {betas}")
        if weight_decay < 0:
            raise ValueError(f"invalid weight_decay {weight_decay}")
        if trust_clip < 0:
            raise ValueError(f"invalid trust_clip {trust_clip}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        trust_clip=trust_clip, debias=debias)
        super().__init__(params, defaults)
        self.last_trust_ratio_mean = None
        self.last_trust_ratio_min = None
        self.last_trust_ratio_max = None

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        trust_ratios: list[float] = []
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("LAMB does not support sparse gradients")
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                if group.get("debias", False):
                    bias_correction = (1.0 - beta2 ** state["step"]) ** 0.5
                    bias_correction /= (1.0 - beta1 ** state["step"])
                else:
                    bias_correction = 1.0
                step_size = group["lr"] * bias_correction
                weight_norm = torch.norm(p.data).clamp(0, group.get("trust_clip", 10.0))
                adam_step = exp_avg / exp_avg_sq.sqrt().add(group["eps"])
                if group["weight_decay"] != 0:
                    adam_step.add_(p.data, alpha=group["weight_decay"])
                adam_norm = torch.norm(adam_step)
                if weight_norm == 0 or adam_norm == 0:
                    trust_ratio = 1.0
                else:
                    trust_ratio = weight_norm / adam_norm
                state["weight_norm"] = weight_norm.clone() if torch.is_tensor(weight_norm) else weight_norm
                state["adam_norm"] = adam_norm.clone() if torch.is_tensor(adam_norm) else adam_norm
                state["trust_ratio"] = trust_ratio.clone() if torch.is_tensor(trust_ratio) else trust_ratio
                trust_ratio_float = float(trust_ratio.item() if torch.is_tensor(trust_ratio) else trust_ratio)
                trust_ratios.append(trust_ratio_float)
                p.data.add_(adam_step, alpha=-step_size * trust_ratio)
        if trust_ratios:
            self.last_trust_ratio_mean = sum(trust_ratios) / len(trust_ratios)
            self.last_trust_ratio_min = min(trust_ratios)
            self.last_trust_ratio_max = max(trust_ratios)
        else:
            self.last_trust_ratio_mean = None
            self.last_trust_ratio_min = None
            self.last_trust_ratio_max = None
        return loss


def build_optimizer(args: argparse.Namespace, model: torch.nn.Module) -> torch.optim.Optimizer:
    betas = (args.adam_beta1, args.adam_beta2)
    if args.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
                                 betas=betas, eps=args.optimizer_eps)
    if args.optimizer == "lamb":
        return Lamb(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
                    betas=betas, eps=args.optimizer_eps, trust_clip=args.lamb_trust_clip,
                    debias=args.lamb_debias)
    raise ValueError(f"unknown optimizer {args.optimizer}")


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


_CAP_RE = re.compile(r"^[A-Z][a-z]+")
_STOP_ANCHOR = {
    'a','an','the','and','or','but','if','then','else','when','while','of','to','in','on','at','by','for','from','with','without','as','is','are','was','were','be','been','being','am','do','does','did','done','have','has','had','having','can','could','will','would','shall','should','may','might','must','not','no','yes','than','that','this','these','those','it','its','he','she','they','them','his','her','their','we','us','you','your','i','me','my','mine','our','ours','who','whom','which','what','where','why','how','there','here','also','more','most','other','some','any','all','one','two','first','second','new','old','many','much','such','into','over','under','after','before','about','up','down','out','off','only','through','between','during','against','within','because'
}
_THIRD_PRON = {'he','she','it','they','him','her','them','his','hers','its','their','theirs','himself','herself','itself','themselves'}


def detect_anchors(input_ids, word_group, tokenizer, special_ids):
    """Detect repeated entity anchors and assign anchor roles to word groups.

    Returns anchor_role tensor: 0=no anchor, 1=source (keep visible), 2=target (prefer mask).
    An anchor is a word group whose lowercased surface text:
      - is a non-stop capitalized token that appears in 2+ groups, OR
      - is a content word (non-stop, non-number) that appears in 3+ groups, OR
      - is a third-person pronoun that co-occurs with a repeated capitalized entity.
    Source = first occurrence of each anchor type; target = later occurrences.
    """
    seq_len = input_ids.shape[0]
    # Map each word group to its lowercased surface string.
    group_to_text: dict[int, str] = {}
    for pos in range(seq_len):
        gid = int(word_group[pos].item())
        if gid < 0:
            continue
        tid = int(input_ids[pos].item())
        if tid in special_ids:
            continue
        tok_str = tokenizer.convert_ids_to_tokens(tid)
        if tok_str is None:
            continue
        # Strip BPE prefix for surface reconstruction.
        clean = tok_str.lstrip("Ġ").lstrip("▁")
        group_to_text.setdefault(gid, "")
        group_to_text[gid] += clean

    # Lowercase all group surfaces.
    group_surface = {gid: txt.lower() for gid, txt in group_to_text.items()}
    # Original case for capitalization check.
    group_surface_raw = group_to_text

    # Count surface occurrences across groups.
    surface_groups: dict[str, list[int]] = collections.defaultdict(list)
    for gid in sorted(group_surface.keys()):
        s = group_surface[gid]
        if s and s not in _STOP_ANCHOR and not re.fullmatch(r'\d+(\.\d+)?', s):
            surface_groups[s].append(gid)

    # Identify anchor types.
    anchor_source: set[int] = set()  # first mention groups (keep visible)
    anchor_target: set[int] = set()  # later mention groups (prefer mask)

    for surface, gids in surface_groups.items():
        if len(gids) < 2:
            continue
        # Check if it's a capitalized entity (threshold: 2+ occurrences).
        raw_first = group_surface_raw.get(gids[0], "")
        is_cap = bool(_CAP_RE.match(raw_first))
        # For non-capitalized content words, require 3+ to be anchor.
        if not is_cap and len(gids) < 3:
            continue
        anchor_source.add(gids[0])
        for gid in gids[1:]:
            anchor_target.add(gid)

    # Also mark third-person pronouns as targets if any capitalized entity anchor exists.
    if anchor_source:
        for gid, s in group_surface.items():
            if s in _THIRD_PRON and gid not in anchor_source and gid not in anchor_target:
                anchor_target.add(gid)

    # Build role tensor.
    anchor_role = torch.zeros(seq_len, dtype=torch.long)
    for pos in range(seq_len):
        gid = int(word_group[pos].item())
        if gid < 0:
            continue
        if gid in anchor_source:
            anchor_role[pos] = 1
        elif gid in anchor_target:
            anchor_role[pos] = 2
    return anchor_role


class MaskedChunkDataset(Dataset):
    """Tokenizes each example and precomputes word-start groups for WWM.

    Extended with anchor detection for entity-anchored masking modes.
    """

    def __init__(self, examples: list[Example], tokenizer, seq_length: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.special_ids = set(tokenizer.all_special_ids)
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
        # Compute anchor roles for entity-anchored masking.
        anchor_role = detect_anchors(input_ids, group, self.tokenizer, self.special_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask,
                "word_group": group, "anchor_role": anchor_role, "words": ex.words}


def collate(batch: list[dict]) -> dict:
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "word_group": torch.stack([x["word_group"] for x in batch]),
        "anchor_role": torch.stack([x["anchor_role"] for x in batch]),
        "words": torch.tensor([x["words"] for x in batch], dtype=torch.long),
    }


def apply_masking(input_ids: torch.Tensor, attention_mask: torch.Tensor, word_group: torch.Tensor,
                  tokenizer, mask_mode: str, mask_prob: float, gen: torch.Generator,
                  anchor_role: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Return (masked_input_ids, labels, telemetry). labels=-100 where not predicted.

    Supports mask_mode in {token, wwm, anchored_wwm, relation_broken_wwm}.
    anchored_wwm: preferentially mask anchor-target groups (role=2) while keeping
                  anchor-source groups (role=1) visible. Fills remaining budget with WWM.
    relation_broken_wwm: same total group budget as anchored_wwm, but selects groups
                         uniformly at random ignoring anchor roles (control).
    """
    device = input_ids.device
    labels = input_ids.clone()
    bsz, seq = input_ids.shape
    mask_token_id = tokenizer.mask_token_id
    special_ids_t = torch.tensor(sorted(tokenizer.all_special_ids), device=device)
    candidate = attention_mask.bool() & ~torch.isin(input_ids, special_ids_t)
    select = torch.zeros_like(candidate)
    telem = {"anchored_groups_masked": 0, "total_groups_masked": 0, "anchor_target_tokens_masked": 0}

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
    elif mask_mode == "anchored_wwm":
        for b in range(bsz):
            groups = word_group[b]
            roles = anchor_role[b] if anchor_role is not None else torch.zeros_like(groups)
            valid_groups = torch.unique(groups[groups >= 0])
            if valid_groups.numel() == 0:
                continue
            n_budget = max(1, int(valid_groups.numel() * mask_prob))
            # Force-select anchor-target groups (role=2).
            target_mask = torch.zeros(seq, dtype=torch.bool, device=device)
            for pos in range(seq):
                if int(roles[pos].item()) == 2:
                    target_mask[pos] = True
            target_groups = torch.unique(groups[target_mask & (groups >= 0)])
            # Force-keep anchor-source groups (role=1).
            source_mask = torch.zeros(seq, dtype=torch.bool, device=device)
            for pos in range(seq):
                if int(roles[pos].item()) == 1:
                    source_mask[pos] = True
            source_groups = torch.unique(groups[source_mask & (groups >= 0)])
            # Remove source groups from candidates for masking.
            non_source_valid = valid_groups[~torch.isin(valid_groups, source_groups)]
            # Start with all target groups.
            chosen_set = set(target_groups.tolist())
            # If under budget, fill with random non-source, non-target groups.
            remaining_budget = n_budget - len(chosen_set)
            if remaining_budget > 0:
                other_groups = non_source_valid[~torch.isin(non_source_valid, target_groups)]
                if other_groups.numel() > 0:
                    perm = torch.randperm(other_groups.numel(), generator=gen, device=device)
                    fill_n = min(remaining_budget, other_groups.numel())
                    for j in range(fill_n):
                        chosen_set.add(int(other_groups[perm[j]].item()))
            # If over budget (many anchors), subsample targets to fit using the torch Generator.
            if len(chosen_set) > n_budget:
                if target_groups.numel() > 0:
                    perm = torch.randperm(target_groups.numel(), generator=gen, device=device)
                    chosen_set = set(int(target_groups[perm[j]].item()) for j in range(min(n_budget, target_groups.numel())))
                else:
                    chosen_set = set()
            chosen = torch.tensor(sorted(chosen_set), dtype=torch.long, device=device)
            sel_b = torch.isin(groups, chosen) & candidate[b]
            select[b] = sel_b
            telem["anchored_groups_masked"] += int(target_groups.numel())
            telem["total_groups_masked"] += len(chosen_set)
            telem["anchor_target_tokens_masked"] += int((sel_b & target_mask).sum().item())
    elif mask_mode == "relation_broken_wwm":
        # Same total group budget as anchored would use, but random group selection.
        for b in range(bsz):
            groups = word_group[b]
            valid_groups = torch.unique(groups[groups >= 0])
            if valid_groups.numel() == 0:
                continue
            n_budget = max(1, int(valid_groups.numel() * mask_prob))
            perm = torch.randperm(valid_groups.numel(), generator=gen, device=device)
            chosen = valid_groups[perm[:n_budget]]
            sel_b = torch.isin(groups, chosen) & candidate[b]
            select[b] = sel_b
            telem["total_groups_masked"] += n_budget
    else:
        raise ValueError(f"unknown mask_mode {mask_mode}")

    if select.sum() == 0:
        flat = candidate.view(-1).nonzero(as_tuple=False)
        if flat.numel() > 0:
            select.view(-1)[flat[0, 0]] = True
    labels[~select] = -100
    masked_inputs = input_ids.clone()
    r = torch.rand(bsz, seq, generator=gen, device=device)
    mask_tok = select & (r < 0.8)
    rand_tok = select & (r >= 0.8) & (r < 0.9)
    masked_inputs[mask_tok] = mask_token_id
    if rand_tok.any():
        rand_ids = torch.randint(0, len(tokenizer), (int(rand_tok.sum()),), generator=gen, device=device)
        masked_inputs[rand_tok] = rand_ids
    return masked_inputs, labels, telem


def seq_length_for_progress(frac: float, schedule: list[tuple[float, int]], default_len: int) -> int:
    length = default_len
    for thresh, L in schedule:
        if frac >= thresh:
            length = L
    return length


def mask_mode_for_progress(frac: float, schedule: list[tuple[float, str]], default_mode: str) -> str:
    mode = default_mode
    for thresh, m in schedule:
        if frac >= thresh:
            mode = m
    return mode


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
            intermediate_size=(args.intermediate_size if getattr(args, "intermediate_size", 0) and args.intermediate_size > 0 else args.hidden_size * args.ffn_mult),
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
    p.add_argument("--mask_mode", choices=["token", "wwm", "anchored_wwm", "relation_broken_wwm"], default="token")
    p.add_argument("--mask_mode_schedule", default="", help="Optional curriculum schedule such as '0.0:wwm,0.7:token'; overrides --mask_mode by progress")
    p.add_argument("--curriculum_progress_basis", choices=["optimizer", "word"], default="optimizer", help="Progress variable for seq_len_schedule and mask_mode_schedule")
    p.add_argument("--curriculum_total_words", type=int, default=0, help="Denominator for word-basis curriculum; 0 uses selected training exposure")
    p.add_argument("--mask_prob", type=float, default=0.15)
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
    p.add_argument("--lr_total_steps", type=int, default=0, help="Cosine schedule total optimizer steps; 0 uses actual optimizer steps")
    p.add_argument("--optimizer", choices=["adamw", "lamb"], default="adamw")
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.98)
    p.add_argument("--optimizer_eps", type=float, default=1e-8)
    p.add_argument("--lamb_trust_clip", type=float, default=10.0)
    p.add_argument("--lamb_debias", action="store_true", help="Use staged torch_optimizer optional LAMB debias correction; default false matches reference default.")
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=8)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--ffn_mult", type=int, default=4)
    p.add_argument("--intermediate_size", type=int, default=0, help="Explicit FFN intermediate size; 0 uses hidden_size*ffn_mult")
    p.add_argument("--micro_batch_size", type=int, default=0, help="Forward/backward micro-batch size inside each effective DataLoader batch; 0 means no split")
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
        "mask_mode_schedule": args.mask_mode_schedule,
        "curriculum_progress_basis": args.curriculum_progress_basis,
        "curriculum_total_words": args.curriculum_total_words,
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
    optim = build_optimizer(args, model)
    total_steps = len(loader)
    optimizer_steps = total_steps
    schedule_total = args.lr_total_steps if args.lr_total_steps and args.lr_total_steps > 0 else optimizer_steps
    if schedule_total < optimizer_steps:
        raise RuntimeError(f"lr_total_steps {schedule_total} < actual optimizer steps {optimizer_steps}")
    micro_batch_size = int(getattr(args, "micro_batch_size", 0) or 0)
    warmup = max(1, int(schedule_total * args.warmup_fraction))
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=warmup, num_training_steps=schedule_total)

    seq_schedule: list[tuple[float, int]] = []
    if args.seq_len_schedule:
        for part in args.seq_len_schedule.split(","):
            t, L = part.split(":")
            seq_schedule.append((float(t), int(L)))
        seq_schedule.sort()
    mask_schedule: list[tuple[float, str]] = []
    if args.mask_mode_schedule:
        for part in args.mask_mode_schedule.split(","):
            t, mode = part.split(":")
            mode = mode.strip()
            if mode not in {"token", "wwm", "anchored_wwm", "relation_broken_wwm"}:
                raise ValueError(f"invalid mask mode in schedule: {mode}")
            mask_schedule.append((float(t), mode))
        mask_schedule.sort()
    curriculum_total_words = args.curriculum_total_words if args.curriculum_total_words and args.curriculum_total_words > 0 else actual_words

    gen = torch.Generator(device=device)
    gen.manual_seed(args.train_rng_seed if args.train_rng_seed >= 0 else args.seed)

    log_path = out / "training_log.jsonl"
    cumulative_words = 0
    loss_values: list[float] = []
    masked_token_values: list[int] = []
    seq_len_values: list[int] = []
    seq_len_step_counts: dict[str, int] = {}
    seq_len_word_counts: dict[str, int] = {}
    mask_mode_values: list[str] = []
    mask_mode_step_counts: dict[str, int] = {}
    mask_mode_word_counts: dict[str, int] = {}
    mask_mode_masked_token_counts: dict[str, int] = {}
    anchor_telem_totals = {"anchored_groups_masked": 0, "total_groups_masked": 0, "anchor_target_tokens_masked": 0}
    optimizer_lr_values: list[float] = []
    lamb_trust_ratio_mean_values: list[float] = []
    lamb_trust_ratio_min_values: list[float] = []
    lamb_trust_ratio_max_values: list[float] = []
    saved_checkpoints: list[dict] = []
    next_ckpt = args.checkpoint_words if args.checkpoint_words and args.checkpoint_words > 0 else None
    model.train()
    opt_step = 0
    with log_path.open("w", encoding="utf-8") as logf:
        for step, batch in enumerate(loader, 1):
            words = int(batch.pop("words").sum().item())
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            word_group = batch["word_group"].to(device, non_blocking=True)
            anchor_role = batch["anchor_role"].to(device, non_blocking=True)
            if args.curriculum_progress_basis == "word":
                frac = cumulative_words / max(1, curriculum_total_words)
            else:
                frac = opt_step / max(1, schedule_total)
            cur_len = seq_length_for_progress(frac, seq_schedule, args.seq_length) if seq_schedule else args.seq_length
            cur_len = min(cur_len, args.max_seq_length)
            cur_mask_mode = mask_mode_for_progress(frac, mask_schedule, args.mask_mode) if mask_schedule else args.mask_mode
            input_ids = input_ids[:, :cur_len].contiguous()
            attention_mask = attention_mask[:, :cur_len].contiguous()
            word_group = word_group[:, :cur_len].contiguous()
            anchor_role = anchor_role[:, :cur_len].contiguous()
            masked_inputs, labels, mask_telem = apply_masking(input_ids, attention_mask, word_group, tokenizer, cur_mask_mode, args.mask_prob, gen, anchor_role=anchor_role)
            n_pred = int((labels != -100).sum().item())
            optim.zero_grad(set_to_none=True)
            if micro_batch_size and 0 < micro_batch_size < masked_inputs.shape[0]:
                # Masking is done once on the full effective batch to preserve protected-trainer
                # RNG/WWM semantics; only activation-heavy forward/backward is micro-split.
                total_tok = max(1, n_pred)
                loss_sum = 0.0
                for lo in range(0, masked_inputs.shape[0], micro_batch_size):
                    hi = min(masked_inputs.shape[0], lo + micro_batch_size)
                    mb_labels = labels[lo:hi]
                    mb_tok = int((mb_labels != -100).sum().item())
                    out_model = model(input_ids=masked_inputs[lo:hi], attention_mask=attention_mask[lo:hi], labels=mb_labels)
                    loss_mb = out_model.loss
                    if loss_mb is None:
                        raise RuntimeError("model returned no loss")
                    (loss_mb * max(1, mb_tok)).backward()
                    loss_sum += float(loss_mb.detach().cpu()) * max(1, mb_tok)
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.div_(float(total_tok))
                loss_float = loss_sum / float(total_tok)
            else:
                out_model = model(input_ids=masked_inputs, attention_mask=attention_mask, labels=labels)
                loss = out_model.loss
                if loss is None:
                    raise RuntimeError("model returned no loss")
                loss.backward()
                loss_float = float(loss.detach().cpu())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            current_lr = float(sched.get_last_lr()[0])
            opt_step += 1
            cumulative_words += words
            loss_values.append(loss_float)
            masked_token_values.append(n_pred)
            seq_len_values.append(cur_len)
            seq_key = str(cur_len)
            seq_len_step_counts[seq_key] = seq_len_step_counts.get(seq_key, 0) + 1
            seq_len_word_counts[seq_key] = seq_len_word_counts.get(seq_key, 0) + words
            mask_mode_values.append(cur_mask_mode)
            mask_mode_step_counts[cur_mask_mode] = mask_mode_step_counts.get(cur_mask_mode, 0) + 1
            mask_mode_word_counts[cur_mask_mode] = mask_mode_word_counts.get(cur_mask_mode, 0) + words
            mask_mode_masked_token_counts[cur_mask_mode] = mask_mode_masked_token_counts.get(cur_mask_mode, 0) + n_pred
            for _k, _v in mask_telem.items():
                anchor_telem_totals[_k] = anchor_telem_totals.get(_k, 0) + int(_v)
            optimizer_lr_values.append(current_lr)
            rec = {"step": step, "opt_step": opt_step, "loss": loss_float, "lr": current_lr,
                   "batch_words": words, "cumulative_word_exposure": cumulative_words,
                   "seq_len": cur_len, "mask_mode": cur_mask_mode, "curriculum_frac": frac,
                   "masked_tokens": n_pred, "mask_telem": mask_telem, "elapsed_sec": time.time() - start_time}
            if args.optimizer == "lamb":
                rec["lamb_trust_ratio_mean"] = getattr(optim, "last_trust_ratio_mean", None)
                rec["lamb_trust_ratio_min"] = getattr(optim, "last_trust_ratio_min", None)
                rec["lamb_trust_ratio_max"] = getattr(optim, "last_trust_ratio_max", None)
                if rec["lamb_trust_ratio_mean"] is not None:
                    lamb_trust_ratio_mean_values.append(float(rec["lamb_trust_ratio_mean"]))
                    lamb_trust_ratio_min_values.append(float(rec["lamb_trust_ratio_min"]))
                    lamb_trust_ratio_max_values.append(float(rec["lamb_trust_ratio_max"]))
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
    if not saved_checkpoints:
        cp = out / "hf_model" / "chck_1M"
        save_hf_checkpoint(model, tokenizer, cp)
        saved_checkpoints.append({"name": "chck_1M", "target_word_exposure": args.checkpoint_words,
                                  "actual_cumulative_word_exposure": cumulative_words, "path": str(cp)})

    metrics = {
        "variant": (f"masked_{args.mask_mode}" if not args.mask_mode_schedule else "masked_curriculum"),
        "backend": "mlm",
        "model_family": model.__class__.__name__,
        "model_type": args.model_type,
        "parameter_count": param_count,
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
        "micro_batch_size": micro_batch_size,
        "optimizer_steps": opt_step,
        "effective_batch_size": args.batch_size,
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "adam_beta1": args.adam_beta1,
        "adam_beta2": args.adam_beta2,
        "optimizer_eps": args.optimizer_eps,
          "lamb_trust_clip": args.lamb_trust_clip,
          "lamb_debias": args.lamb_debias,
        "lr_first": optimizer_lr_values[0] if optimizer_lr_values else None,
        "lr_last": optimizer_lr_values[-1] if optimizer_lr_values else None,
        "lr_max_observed": max(optimizer_lr_values) if optimizer_lr_values else None,
        "lamb_trust_ratio_mean_first": lamb_trust_ratio_mean_values[0] if lamb_trust_ratio_mean_values else None,
        "lamb_trust_ratio_mean_last": lamb_trust_ratio_mean_values[-1] if lamb_trust_ratio_mean_values else None,
        "lamb_trust_ratio_mean_min": min(lamb_trust_ratio_mean_values) if lamb_trust_ratio_mean_values else None,
        "lamb_trust_ratio_mean_max": max(lamb_trust_ratio_mean_values) if lamb_trust_ratio_mean_values else None,
        "lamb_trust_ratio_min_observed": min(lamb_trust_ratio_min_values) if lamb_trust_ratio_min_values else None,
        "lamb_trust_ratio_max_observed": max(lamb_trust_ratio_max_values) if lamb_trust_ratio_max_values else None,
        "mask_mode": args.mask_mode,
        "mask_mode_schedule": args.mask_mode_schedule,
        "curriculum_progress_basis": args.curriculum_progress_basis,
        "curriculum_total_words": curriculum_total_words,
        "unique_mask_modes": sorted(set(mask_mode_values)),
        "mask_mode_step_counts": mask_mode_step_counts,
        "mask_mode_word_counts": mask_mode_word_counts,
        "mask_mode_masked_token_counts": mask_mode_masked_token_counts,
        "mask_prob": args.mask_prob,
        "masked_tokens_total": sum(masked_token_values),
        "masked_tokens_mean_per_step": (sum(masked_token_values) / len(masked_token_values)) if masked_token_values else None,
        "masked_tokens_per_whitespace_word": (sum(masked_token_values) / cumulative_words) if cumulative_words else None,
          "anchor_masking_telemetry_totals": anchor_telem_totals,
          "anchor_target_tokens_masked_per_predicted_token": (anchor_telem_totals.get("anchor_target_tokens_masked", 0) / max(1, sum(masked_token_values))),
        "unique_training_seq_lengths": sorted(set(seq_len_values)),
        "seq_len_step_counts": seq_len_step_counts,
        "seq_len_word_counts": seq_len_word_counts,
        "tokenization_coupling_summary": tokenization_summary,
        "seq_length": args.seq_length,
        "max_seq_length": args.max_seq_length,
        "max_position_embeddings": max(args.max_position_embeddings, args.max_seq_length + 8),
        "seq_len_schedule": args.seq_len_schedule,
        "hidden_size": args.hidden_size,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "ffn_mult": args.ffn_mult,
        "intermediate_size": (args.intermediate_size if getattr(args, "intermediate_size", 0) and args.intermediate_size > 0 else args.hidden_size * args.ffn_mult),
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
