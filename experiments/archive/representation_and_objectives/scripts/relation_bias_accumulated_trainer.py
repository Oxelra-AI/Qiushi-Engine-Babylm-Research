#!/usr/bin/env python3
"""research relation-biased accumulated trainer for post-40k BabyLM work.

This script is a local extension of the accumulated trainer.  It is
not used by the currently running legal-40k trainings.  It prepares a distinct
post-40k route: reallocate masked-LM prediction pressure toward broad
relation/predicate cue words already present in the exact allowed 10M training
pool, while preserving the same effective batch, word clock, sequence schedule,
optimizer schedule, and checkpoint accounting used by the research memory-safe
trainer.

The script also exposes --intermediate_size so a true 12x384 leader-shape
DeBERTa-v2 configuration can be trained later without relying on ffn_mult=4.
No official evaluation text is used here; relation cues come from the research
training-pair lexicon and the training examples themselves.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DebertaV2Config, DebertaV2ForMaskedLM, get_cosine_schedule_with_warmup

USER_ROOT = Path(".").resolve()
STUDY = USER_ROOT / "experiments/archive" / 'representation_and_objectives'
WORKSPACE = STUDY
COMPACT_EXPERIENCE_SCRIPTS = USER_ROOT / "experiments/archive" / 'compact_experience' / "scripts"
if str(COMPACT_EXPERIENCE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(COMPACT_EXPERIENCE_SCRIPTS))

import masking_curriculum_trainer as base  # noqa: E402

research = WORKSPACE / "scripts" / "relation_frame_retention_audit.py"
WORD_SPAN_RE = re.compile(r"[^\s]+")
WORD_TOKEN_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)?|\d+(?:\.\d+)?")


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def load_step54_lexicon() -> tuple[dict[str, list[str]], dict[str, list[tuple[str, ...]]]]:
    if not research.exists():
        raise RuntimeError(f"research relation lexicon script not found: {research}")
    spec = importlib.util.spec_from_file_location("relation_frame_retention_audit", research)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import research relation lexicon from {research}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.CUE_LEXICON, mod.MULTI_CUES


def norm_token(t: str) -> str:
    t = t.lower().strip()
    if len(t) > 5 and t.endswith("ing"):
        stem = t[:-3]
        if stem:
            return stem
    if len(t) > 4 and t.endswith("ed"):
        stem = t[:-2]
        if stem:
            return stem
    if len(t) > 4 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


def word_norms(word: str) -> list[str]:
    return [norm_token(m.group(0)) for m in WORD_TOKEN_RE.finditer(word)]


def build_cue_single(cue_lexicon: dict[str, list[str]]) -> dict[str, set[str]]:
    cue_single: dict[str, set[str]] = {}
    for cat, words in cue_lexicon.items():
        for w in words:
            n = norm_token(w)
            cue_single.setdefault(n, set()).add(cat)
    return cue_single


def relation_categories_by_word(
    text: str,
    cue_single: dict[str, set[str]],
    multi_cues: dict[str, list[tuple[str, ...]]],
) -> tuple[list[tuple[int, int, str]], list[set[str]]]:
    spans = [(m.start(), m.end(), m.group(0)) for m in WORD_SPAN_RE.finditer(text)]
    primary_norm: list[str] = []
    cats_by_word: list[set[str]] = []
    for _s, _e, raw in spans:
        norms = word_norms(raw)
        primary_norm.append(norms[0] if norms else "")
        cats: set[str] = set()
        for n in norms:
            cats.update(cue_single.get(n, set()))
        cats_by_word.append(cats)

    for cat, phrases in multi_cues.items():
        for raw_phrase in phrases:
            phrase = tuple(norm_token(x) for x in raw_phrase)
            L = len(phrase)
            if L == 0 or L > len(primary_norm):
                continue
            for i in range(len(primary_norm) - L + 1):
                if tuple(primary_norm[i:i + L]) == phrase:
                    for j in range(i, i + L):
                        cats_by_word[j].add(cat)
    return spans, cats_by_word


def word_index_for_offset(starts: list[int], spans: list[tuple[int, int, str]], start: int, end: int) -> int | None:
    if not spans:
        return None
    idx = bisect.bisect_right(starts, start) - 1
    if idx >= 0 and start < spans[idx][1] and end > spans[idx][0]:
        return idx
    mid = (start + end) // 2
    idx = bisect.bisect_right(starts, mid) - 1
    if idx >= 0 and mid < spans[idx][1]:
        return idx
    nxt = idx + 1
    if 0 <= nxt < len(spans) and end > spans[nxt][0]:
        return nxt
    return None


class RelationMaskedChunkDataset(Dataset):
    """Dataset matching base.MaskedChunkDataset plus a relation-token/group mask.

    word_group is constructed with the exact word-start logic used by the COMPACT_EXPERIENCE
    base trainer.  relation_group marks every visible non-special token in a word
    group as relation-bearing if any token in the group maps to a research cue word.
    """

    def __init__(
        self,
        examples: list[Any],
        tokenizer,
        seq_length: int,
        cue_single: dict[str, set[str]],
        multi_cues: dict[str, list[tuple[str, ...]]],
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.special_ids = set(tokenizer.all_special_ids)
        self._word_start: dict[int, bool] = {}
        self.cue_single = cue_single
        self.multi_cues = multi_cues

    def __len__(self) -> int:
        return len(self.examples)

    def _word_start_flag(self, token_id: int) -> bool:
        v = self._word_start.get(token_id)
        if v is None:
            s = self.tokenizer.convert_ids_to_tokens(int(token_id))
            v = bool(s is not None and base.is_word_start(str(s)))
            self._word_start[token_id] = v
        return v

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex.text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.seq_length,
            padding="max_length",
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        offsets = enc["offset_mapping"].squeeze(0)
        word_group = torch.full_like(input_ids, -1)
        relation_token = torch.zeros_like(input_ids, dtype=torch.bool)

        spans, cats_by_word = relation_categories_by_word(ex.text, self.cue_single, self.multi_cues)
        starts = [s for s, _e, _w in spans]
        gid = -1
        for i in range(input_ids.shape[0]):
            if int(attention_mask[i].item()) == 0:
                continue
            tid = int(input_ids[i].item())
            if tid in self.special_ids:
                continue
            if gid < 0 or self._word_start_flag(tid) or i == 0:
                gid += 1
            word_group[i] = gid
            start, end = offsets[i].tolist()
            wi = word_index_for_offset(starts, spans, int(start), int(end))
            if wi is not None and 0 <= wi < len(cats_by_word) and cats_by_word[wi]:
                relation_token[i] = True

        # Whole-word selection operates at word_group level; if one token in the
        # group maps to a relation cue, mark all candidate tokens in that group.
        valid_groups = torch.unique(word_group[word_group >= 0])
        for g in valid_groups.tolist():
            gm = word_group == int(g)
            if bool((relation_token & gm).any().item()):
                relation_token[gm] = True
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "word_group": word_group,
            "relation_group": relation_token,
            "words": torch.tensor(ex.words, dtype=torch.long),
        }


def relation_collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "word_group": torch.stack([x["word_group"] for x in batch]),
        "relation_group": torch.stack([x["relation_group"] for x in batch]),
        "words": torch.stack([x["words"] for x in batch]).long(),
    }


def combine_microbatches(micro_batches: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {
        "input_ids": torch.cat([b["input_ids"] for b in micro_batches], dim=0),
        "attention_mask": torch.cat([b["attention_mask"] for b in micro_batches], dim=0),
        "word_group": torch.cat([b["word_group"] for b in micro_batches], dim=0),
        "words": torch.cat([b["words"] for b in micro_batches], dim=0),
    }
    if "relation_group" in micro_batches[0]:
        out["relation_group"] = torch.cat([b["relation_group"] for b in micro_batches], dim=0)
    return out


def effective_batch_iterator(loader: DataLoader, accum_steps: int):
    buf: list[dict[str, torch.Tensor]] = []
    for batch in loader:
        buf.append(batch)
        if len(buf) == accum_steps:
            yield combine_microbatches(buf)
            buf = []
    if buf:
        yield combine_microbatches(buf)


def normalized_relation_probs(
    base_prob: float,
    boost: float,
    n_relation: int,
    n_nonrelation: int,
    prob_max: float,
) -> tuple[float, float, bool]:
    """Return relation/nonrelation probabilities preserving expected count when possible.

    For a row with n groups, expected selected groups equals base_prob*n unless a
    row has too many relation groups for the requested boost.  In that case the
    relation probability is reduced to the largest probability that preserves the
    same expectation with zero nonrelation probability.
    """
    n_total = n_relation + n_nonrelation
    if n_total <= 0:
        return base_prob, base_prob, False
    base_prob = float(max(0.0, min(1.0, base_prob)))
    rel_prob = float(max(0.0, min(prob_max, base_prob * boost)))
    clipped = rel_prob < base_prob * boost - 1e-12
    if n_relation == 0:
        return base_prob, base_prob, clipped
    if n_nonrelation == 0:
        # All groups are relation-bearing.  Preserve total expected group count.
        return base_prob, 0.0, clipped or rel_prob != base_prob
    nonrel_prob = (base_prob * n_total - rel_prob * n_relation) / n_nonrelation
    if nonrel_prob < 0.0:
        rel_prob = base_prob * n_total / n_relation
        nonrel_prob = 0.0
        clipped = True
    if nonrel_prob > 1.0:
        nonrel_prob = 1.0
        clipped = True
    return float(rel_prob), float(nonrel_prob), clipped


def summarize_selection(
    select: torch.Tensor,
    candidate: torch.Tensor,
    word_group: torch.Tensor,
    relation_group: torch.Tensor,
    prob_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    stats = {
        "candidate_tokens": int(candidate.sum().item()),
        "candidate_relation_tokens": int((candidate & relation_group).sum().item()),
        "selected_tokens": int(select.sum().item()),
        "selected_relation_tokens": int((select & relation_group).sum().item()),
        "candidate_groups": 0,
        "candidate_relation_groups": 0,
        "selected_groups": 0,
        "selected_relation_groups": 0,
        "probability_rows": len(prob_rows),
        "probability_rows_clipped": sum(1 for r in prob_rows if r.get("clipped")),
        "probability_rows_no_nonrelation": sum(1 for r in prob_rows if r.get("n_nonrelation_groups") == 0),
    }
    bsz = word_group.shape[0]
    for b in range(bsz):
        groups = word_group[b]
        cand_b = candidate[b]
        rel_b = relation_group[b]
        sel_b = select[b]
        valid = torch.unique(groups[(groups >= 0) & cand_b])
        selected = torch.unique(groups[(groups >= 0) & sel_b])
        stats["candidate_groups"] += int(valid.numel())
        stats["selected_groups"] += int(selected.numel())
        for gid_t in valid.tolist():
            gm = (groups == int(gid_t)) & cand_b
            if bool((gm & rel_b).any().item()):
                stats["candidate_relation_groups"] += 1
        for gid_t in selected.tolist():
            gm = (groups == int(gid_t)) & cand_b
            if bool((gm & rel_b).any().item()):
                stats["selected_relation_groups"] += 1
    for num, den, key in [
        (stats["candidate_relation_tokens"], stats["candidate_tokens"], "candidate_relation_token_fraction"),
        (stats["selected_relation_tokens"], stats["selected_tokens"], "selected_relation_token_fraction"),
        (stats["candidate_relation_groups"], stats["candidate_groups"], "candidate_relation_group_fraction"),
        (stats["selected_relation_groups"], stats["selected_groups"], "selected_relation_group_fraction"),
    ]:
        stats[key] = float(num) / float(den) if den else None
    if prob_rows:
        stats["mean_relation_group_prob"] = sum(r["relation_prob"] for r in prob_rows) / len(prob_rows)
        stats["mean_nonrelation_group_prob"] = sum(r["nonrelation_prob"] for r in prob_rows) / len(prob_rows)
    else:
        stats["mean_relation_group_prob"] = None
        stats["mean_nonrelation_group_prob"] = None
    return stats


def apply_masking_with_relation(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    word_group: torch.Tensor,
    relation_group: torch.Tensor | None,
    tokenizer,
    state: Any,
    gen: torch.Generator,
    *,
    relation_enabled: bool,
    relation_cue_boost: float,
    relation_prob_max: float = 0.6,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if (not relation_enabled) or relation_group is None or relation_cue_boost <= 1.0000001:
        masked_inputs, labels = base.apply_masking_curriculum(input_ids, attention_mask, word_group, tokenizer, state, gen)
        rel = relation_group if relation_group is not None else torch.zeros_like(input_ids, dtype=torch.bool)
        special_ids = torch.tensor(sorted(tokenizer.all_special_ids), device=input_ids.device)
        candidate = attention_mask.bool() & ~torch.isin(input_ids, special_ids)
        select = labels != -100
        stats = summarize_selection(select, candidate, word_group, rel.bool(), [])
        stats.update({
            "relation_enabled": False,
            "relation_cue_boost": relation_cue_boost,
            "mask_mode": state.get_current_mask_mode(),
        })
        return masked_inputs, labels, stats

    device = input_ids.device
    relation_group = relation_group.to(device).bool()
    labels = input_ids.clone()
    bsz, seq = input_ids.shape
    mask_token_id = tokenizer.mask_token_id
    special_ids = torch.tensor(sorted(tokenizer.all_special_ids), device=device)
    candidate = attention_mask.bool() & ~torch.isin(input_ids, special_ids)
    select = torch.zeros_like(candidate)
    mask_mode = state.get_current_mask_mode()
    base_prob = float(state.get_current_mask_prob())
    prob_rows: list[dict[str, Any]] = []

    if mask_mode == "token":
        n_relation = int((candidate & relation_group).sum().item())
        n_nonrelation = int((candidate & ~relation_group).sum().item())
        rel_prob, nonrel_prob, clipped = normalized_relation_probs(base_prob, relation_cue_boost, n_relation, n_nonrelation, relation_prob_max)
        probs = torch.rand(bsz, seq, generator=gen, device=device)
        select = candidate & (((relation_group & (probs < rel_prob)) | (~relation_group & (probs < nonrel_prob))))
        prob_rows.append({
            "row": "token_global",
            "n_relation_groups": n_relation,
            "n_nonrelation_groups": n_nonrelation,
            "relation_prob": rel_prob,
            "nonrelation_prob": nonrel_prob,
            "clipped": clipped,
        })
    elif mask_mode == "wwm":
        for b in range(bsz):
            groups = word_group[b]
            valid_groups = torch.unique(groups[(groups >= 0) & candidate[b]])
            if valid_groups.numel() == 0:
                continue
            rel_flags: list[bool] = []
            for gid_t in valid_groups.tolist():
                gm = (groups == int(gid_t)) & candidate[b]
                rel_flags.append(bool((gm & relation_group[b]).any().item()))
            n_relation = sum(1 for x in rel_flags if x)
            n_nonrelation = len(rel_flags) - n_relation
            rel_prob, nonrel_prob, clipped = normalized_relation_probs(base_prob, relation_cue_boost, n_relation, n_nonrelation, relation_prob_max)
            probs_by_group = torch.tensor([rel_prob if flag else nonrel_prob for flag in rel_flags], device=device)
            gp = torch.rand(valid_groups.numel(), generator=gen, device=device)
            chosen = valid_groups[gp < probs_by_group]
            if chosen.numel() > 0:
                select[b] = torch.isin(groups, chosen) & candidate[b]
            prob_rows.append({
                "row": int(b),
                "n_relation_groups": int(n_relation),
                "n_nonrelation_groups": int(n_nonrelation),
                "relation_prob": float(rel_prob),
                "nonrelation_prob": float(nonrel_prob),
                "clipped": bool(clipped),
            })
    else:
        raise RuntimeError(f"Unknown mask mode {mask_mode}")

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

    stats = summarize_selection(select, candidate, word_group, relation_group, prob_rows)
    stats.update({
        "relation_enabled": True,
        "relation_cue_boost": float(relation_cue_boost),
        "relation_prob_max": float(relation_prob_max),
        "base_mask_prob": float(base_prob),
        "mask_mode": mask_mode,
    })
    return masked_inputs, labels, stats


def build_model(args: argparse.Namespace, tokenizer):
    intermediate_size = int(getattr(args, "intermediate_size", 0) or 0)
    if intermediate_size <= 0:
        return base.build_model(args, tokenizer)
    max_pos = max(args.max_position_embeddings, args.max_seq_length + 8)
    pos_att_type = [x.strip() for x in args.deberta_pos_att_type.split(",") if x.strip()]
    cfg = DebertaV2Config(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.n_layer,
        num_attention_heads=args.n_head,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_pos,
        max_relative_positions=args.max_relative_positions,
        position_buckets=args.position_buckets,
        relative_attention=True,
        pos_att_type=pos_att_type,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return DebertaV2ForMaskedLM(cfg)


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Accumulated trainer with optional relation-biased masking")
    p.add_argument("--dataset_id", default="BabyLM-community/BabyLM-2026-Strict-Small")
    p.add_argument("--dataset_revision", default="c92ab16b4f08858304b0815706065b3354d8fc0a")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_word_exposure", type=int, default=4_000_000)
    p.add_argument("--example_pool_words", type=int, default=10_000_000)
    p.add_argument("--checkpoint_words", type=int, default=1_000_000)
    p.add_argument("--words_per_example", type=int, default=160)
    p.add_argument("--example_jsonl", default="")
    p.add_argument("--example_jsonl_label", default="")
    p.add_argument("--example_jsonl_meta", default="")
    p.add_argument("--tokenizer_path", default="")
    p.add_argument("--tokenizer_label", default="baseline16k")
    p.add_argument("--tokenization_summary_limit", type=int, default=500)

    p.add_argument("--masking_curriculum", choices=base.MASKING_CURRICULA, default="wwm_fixed")
    p.add_argument("--mask_prob_start", type=float, default=0.15)
    p.add_argument("--mask_prob_end", type=float, default=0.15)
    p.add_argument("--switch_frac", type=float, default=0.7)
    p.add_argument("--amlm_window", type=int, default=10)
    p.add_argument("--amlm_lambda", type=float, default=0.2)

    p.add_argument("--relation_masking", action="store_true")
    p.add_argument("--relation_cue_boost", type=float, default=1.0)
    p.add_argument("--relation_prob_max", type=float, default=0.6)

    p.add_argument("--seq_length", type=int, default=256)
    p.add_argument("--max_seq_length", type=int, default=256)
    p.add_argument("--max_position_embeddings", type=int, default=512)
    p.add_argument("--seq_len_schedule", default="")

    p.add_argument("--hidden_size", type=int, default=480)
    p.add_argument("--n_layer", type=int, default=8)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--ffn_mult", type=int, default=4)
    p.add_argument("--intermediate_size", type=int, default=0)
    p.add_argument("--position_buckets", type=int, default=256)
    p.add_argument("--max_relative_positions", type=int, default=256)
    p.add_argument("--deberta_pos_att_type", default="p2c,c2p")

    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--micro_batch_size", type=int, default=64)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_fraction", type=float, default=0.05)
    p.add_argument("--lr_total_steps", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=0)

    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--extra_init_seed", type=int, default=-1)
    p.add_argument("--train_rng_seed", type=int, default=-1)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--dynamics_trace_every", type=int, default=200)
    p.add_argument("--mask_stats_every", type=int, default=50)
    return p.parse_args()


def load_examples_and_metadata(args: argparse.Namespace) -> tuple[list[Any], int, int, dict[str, Any], list[dict], int]:
    selected_words = args.max_word_exposure
    if not args.example_jsonl:
        raise RuntimeError("This research trainer is intended for explicit example_jsonl runs only.")
    jsonl_path = Path(args.example_jsonl)
    examples, jsonl_total_words, jsonl_total_rows, sample_rows = base.load_examples_jsonl(jsonl_path, selected_words)
    actual_words = sum(ex.words for ex in examples)
    manifest_files = [{
        "path": str(jsonl_path),
        "name": jsonl_path.name,
        "bytes": jsonl_path.stat().st_size,
        "sha256": base.sha256_file(jsonl_path),
        "whitespace_words": jsonl_total_words,
        "rows": jsonl_total_rows,
    }]
    example_jsonl_meta = {}
    if args.example_jsonl_meta:
        meta_path = Path(args.example_jsonl_meta)
        example_jsonl_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    example_selection_metadata = {
        "data_source_type": "example_jsonl",
        "example_jsonl": str(jsonl_path),
        "example_jsonl_label": args.example_jsonl_label,
        "example_jsonl_meta": example_jsonl_meta,
        "manifest_files": manifest_files,
        "sample_rows": sample_rows,
    }
    if actual_words != selected_words:
        raise RuntimeError(f"word selection mismatch {actual_words} vs {selected_words}")
    return examples, actual_words, jsonl_total_words, example_selection_metadata, manifest_files, jsonl_total_rows


def reset_all_rng(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = build_args()
    if args.batch_size <= 0 or args.micro_batch_size <= 0:
        raise ValueError("batch_size and micro_batch_size must be positive")
    if args.batch_size % args.micro_batch_size != 0:
        raise ValueError("batch_size must be divisible by micro_batch_size")
    if args.relation_masking and args.masking_curriculum in ("amlm_hard", "amlm_hard_switch"):
        raise ValueError("relation_masking is not combined with AMLM in this memory-safe trainer")
    accum_steps = args.batch_size // args.micro_batch_size

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    start_time = time.time()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = base.make_portable_tokenizer(args.tokenizer_path)
    examples, actual_words, total_words, example_selection_metadata, manifest_files, jsonl_total_rows = load_examples_and_metadata(args)

    cue_lexicon, multi_cues = load_step54_lexicon()
    cue_single = build_cue_single(cue_lexicon)
    relation_lexicon_manifest = {
        "script": str(research),
        "script_sha256": sha256_file(research),
        "cue_categories": sorted(cue_lexicon),
        "cue_single_count": len(cue_single),
        "multi_phrase_count": sum(len(v) for v in multi_cues.values()),
        "cue_lexicon_hash": stable_hash({"single": cue_lexicon, "multi": {k: [list(x) for x in v] for k, v in multi_cues.items()}}),
    }

    if args.relation_masking:
        dataset = RelationMaskedChunkDataset(examples, tokenizer, args.max_seq_length, cue_single, multi_cues)
        collate_fn = relation_collate
    else:
        dataset = base.MaskedChunkDataset(examples, tokenizer, args.max_seq_length)
        collate_fn = base.collate
    loader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    total_steps = math.ceil(len(dataset) / args.batch_size)
    token_freq_ranks = base.compute_token_frequency_ranks(examples, tokenizer, args.max_seq_length)

    curriculum_state = base.MaskingCurriculumState(
        curriculum=args.masking_curriculum,
        mask_prob_start=args.mask_prob_start,
        mask_prob_end=args.mask_prob_end,
        switch_frac=args.switch_frac,
        amlm_window=args.amlm_window,
        amlm_lambda=args.amlm_lambda,
    )
    curriculum_state.initialize(vocab_size=len(tokenizer), total_steps=total_steps)

    reset_all_rng(args.seed)
    if args.extra_init_seed >= 0:
        reset_all_rng(args.extra_init_seed)
    model = build_model(args, tokenizer)
    if args.train_rng_seed >= 0:
        reset_all_rng(args.train_rng_seed)

    param_count = sum(p.numel() for p in model.parameters())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    schedule_total = args.lr_total_steps if args.lr_total_steps > 0 else total_steps
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
    dynamics_path = out / "dynamics_traces.jsonl"
    mask_stats_path = out / "mask_selection_stats.jsonl"
    cumulative_words = 0
    loss_values: list[float] = []
    saved_checkpoints: list[dict[str, Any]] = []
    dynamics_traces: list[dict[str, Any]] = []
    mask_stats_records: list[dict[str, Any]] = []
    next_ckpt = args.checkpoint_words if args.checkpoint_words > 0 else None

    source_words: dict[str, int] = {}
    for ex in examples:
        source_words[ex.source] = source_words.get(ex.source, 0) + ex.words

    (out / "example_order_manifest.json").write_text(json.dumps({
        "seed": args.seed,
        "selected_for_training_words": actual_words,
        "num_consumed_examples": len(examples),
        "masking_curriculum": args.masking_curriculum,
        "mask_prob_start": args.mask_prob_start,
        "mask_prob_end": args.mask_prob_end,
        "switch_frac": args.switch_frac,
        "relation_masking": bool(args.relation_masking),
        "relation_cue_boost": args.relation_cue_boost,
        "relation_prob_max": args.relation_prob_max,
        "relation_lexicon_manifest": relation_lexicon_manifest,
        "effective_batch_size": args.batch_size,
        "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation_steps": accum_steps,
        "source_words_consumed": source_words,
        **example_selection_metadata,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({
        "event": "relation_bias_accum_trainer_start",
        "created_utc": now_utc(),
        "output_dir": str(out),
        "device": str(device),
        "vocab_size": len(tokenizer),
        "param_count": param_count,
        "hidden_size": args.hidden_size,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "ffn_mult": args.ffn_mult,
        "intermediate_size": args.intermediate_size if args.intermediate_size > 0 else args.hidden_size * args.ffn_mult,
        "effective_batch_size": args.batch_size,
        "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation_steps": accum_steps,
        "total_effective_steps": total_steps,
        "word_exposure_target": args.max_word_exposure,
        "tokenizer_label": args.tokenizer_label,
        "relation_masking": bool(args.relation_masking),
        "relation_cue_boost": args.relation_cue_boost,
    }), flush=True)

    model.train()
    with log_path.open("w", encoding="utf-8") as logf, mask_stats_path.open("w", encoding="utf-8") as maskf:
        for step, batch in enumerate(effective_batch_iterator(loader, accum_steps), 1):
            words = int(batch.pop("words").sum().item())
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            word_group = batch["word_group"].to(device, non_blocking=True)
            relation_group = batch.get("relation_group")
            if relation_group is not None:
                relation_group = relation_group.to(device, non_blocking=True).bool()

            frac = (step - 1) / max(1, schedule_total)
            cur_len = base.seq_length_for_progress(frac, seq_schedule, args.seq_length) if seq_schedule else args.seq_length
            cur_len = min(cur_len, args.max_seq_length)
            input_ids = input_ids[:, :cur_len].contiguous()
            attention_mask = attention_mask[:, :cur_len].contiguous()
            word_group = word_group[:, :cur_len].contiguous()
            if relation_group is not None:
                relation_group = relation_group[:, :cur_len].contiguous()

            curriculum_state.current_step = step - 1
            masked_inputs, labels, mask_stats = apply_masking_with_relation(
                input_ids,
                attention_mask,
                word_group,
                relation_group,
                tokenizer,
                curriculum_state,
                gen,
                relation_enabled=bool(args.relation_masking),
                relation_cue_boost=float(args.relation_cue_boost),
                relation_prob_max=float(args.relation_prob_max),
            )
            n_pred_total_t = (labels != -100).sum()
            n_pred_total = int(n_pred_total_t.item())
            n_candidate = int(attention_mask.bool().sum().item())
            effective_mask_rate = n_pred_total / max(1, n_candidate)
            if n_pred_total <= 0:
                raise RuntimeError(f"no masked tokens in effective batch at step {step}")

            optim.zero_grad(set_to_none=True)
            weighted_loss_sum = 0.0
            active_microbatches = 0
            batch_rows = input_ids.shape[0]
            for start in range(0, batch_rows, args.micro_batch_size):
                end = min(start + args.micro_batch_size, batch_rows)
                sl_labels = labels[start:end]
                n_pred_i = int((sl_labels != -100).sum().item())
                if n_pred_i <= 0:
                    continue
                out_model = model(
                    input_ids=masked_inputs[start:end],
                    attention_mask=attention_mask[start:end],
                    labels=sl_labels,
                )
                loss_i = out_model.loss
                if loss_i is None:
                    raise RuntimeError("model returned no loss")
                scale = n_pred_i / n_pred_total
                (loss_i * scale).backward()
                weighted_loss_sum += float(loss_i.detach().cpu()) * scale
                active_microbatches += 1
                del out_model, loss_i

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()

            cumulative_words += words
            loss_float = float(weighted_loss_sum)
            loss_values.append(loss_float)
            if curriculum_state.uses_amlm:
                raise RuntimeError("research accumulated trainer does not support AMLM curricula")
            if step % args.dynamics_trace_every == 0:
                curriculum_state.trace_mask_rates.append(effective_mask_rate)

            rec = {
                "step": step,
                "loss": loss_float,
                "lr": float(sched.get_last_lr()[0]),
                "batch_words": words,
                "cumulative_word_exposure": cumulative_words,
                "seq_len": cur_len,
                "masked_tokens": n_pred_total,
                "effective_mask_rate": round(effective_mask_rate, 4),
                "mask_mode": curriculum_state.get_current_mask_mode(),
                "mask_prob_nominal": round(curriculum_state.get_current_mask_prob(), 4),
                "relation_masking": bool(args.relation_masking),
                "relation_selected_group_fraction": mask_stats.get("selected_relation_group_fraction"),
                "relation_selected_token_fraction": mask_stats.get("selected_relation_token_fraction"),
                "candidate_relation_group_fraction": mask_stats.get("candidate_relation_group_fraction"),
                "active_microbatches": active_microbatches,
                "micro_batch_size": args.micro_batch_size,
                "elapsed_sec": round(time.time() - start_time, 1),
            }
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            if step == 1 or step % args.mask_stats_every == 0 or step == total_steps:
                ms = {"step": step, "cumulative_word_exposure": cumulative_words, **mask_stats}
                mask_stats_records.append(ms)
                maskf.write(json.dumps(ms) + "\n")
                maskf.flush()
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
                base.save_hf_checkpoint(model, tokenizer, cp)
                trace = curriculum_state.flush_dynamics_trace(name)
                dynamics_traces.append(trace)
                saved_checkpoints.append({
                    "name": name,
                    "target_word_exposure": next_ckpt,
                    "actual_cumulative_word_exposure": cumulative_words,
                    "path": str(cp),
                })
                print(json.dumps({
                    "event": "checkpoint_saved",
                    "name": name,
                    "cum_words": cumulative_words,
                    "mask_mode": curriculum_state.get_current_mask_mode(),
                    "mask_prob": round(curriculum_state.get_current_mask_prob(), 4),
                }), flush=True)
                next_ckpt += args.checkpoint_words

            del input_ids, attention_mask, word_group, masked_inputs, labels
            if relation_group is not None:
                del relation_group

    base.save_hf_checkpoint(model, tokenizer, out / "hf_model")
    if not saved_checkpoints:
        cp = out / "hf_model" / "chck_1M"
        base.save_hf_checkpoint(model, tokenizer, cp)
        saved_checkpoints.append({
            "name": "chck_1M",
            "target_word_exposure": args.checkpoint_words,
            "actual_cumulative_word_exposure": cumulative_words,
            "path": str(cp),
        })

    with dynamics_path.open("w", encoding="utf-8") as f:
        for trace in dynamics_traces:
            f.write(json.dumps(trace) + "\n")

    metrics = {
        "variant": "relation_bias_accumulated" if args.relation_masking else "accumulated_no_relation_bias",
        "backend": "mlm",
        "model_family": "DebertaV2ForMaskedLM",
        "parameter_count": param_count,
        "vocab_size": len(tokenizer),
        "tokenizer_label": args.tokenizer_label,
        "word_exposure": cumulative_words,
        "loss_first": loss_values[0] if loss_values else None,
        "loss_last": loss_values[-1] if loss_values else None,
        "actual_training_steps": total_steps,
        "effective_batch_size": args.batch_size,
        "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation_steps": accum_steps,
        "masking_curriculum": args.masking_curriculum,
        "mask_prob_start": args.mask_prob_start,
        "mask_prob_end": args.mask_prob_end,
        "switch_frac": args.switch_frac,
        "relation_masking": bool(args.relation_masking),
        "relation_cue_boost": args.relation_cue_boost,
        "relation_prob_max": args.relation_prob_max,
        "relation_lexicon_manifest": relation_lexicon_manifest,
        "mask_selection_stats_file": str(mask_stats_path),
        "mask_selection_stats_sample": mask_stats_records[:10],
        "n_amlm_updates": curriculum_state.n_amlm_updates,
        "hidden_size": args.hidden_size,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "ffn_mult": args.ffn_mult,
        "intermediate_size": args.intermediate_size if args.intermediate_size > 0 else args.hidden_size * args.ffn_mult,
        "seed": args.seed,
        "extra_init_seed": args.extra_init_seed,
        "train_rng_seed": args.train_rng_seed,
        "seq_length": args.seq_length,
        "max_seq_length": args.max_seq_length,
        "seq_len_schedule": args.seq_len_schedule,
        "saved_checkpoints": saved_checkpoints,
        "dynamics_traces_file": str(dynamics_path),
        "source_words_consumed": source_words,
        **example_selection_metadata,
    }
    (out / "scientific_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({
        "event": "done",
        "param_count": param_count,
        "loss_first": metrics["loss_first"],
        "loss_last": metrics["loss_last"],
        "word_exposure": cumulative_words,
        "masking_curriculum": args.masking_curriculum,
        "relation_masking": bool(args.relation_masking),
        "relation_cue_boost": args.relation_cue_boost,
        "effective_batch_size": args.batch_size,
        "micro_batch_size": args.micro_batch_size,
        "checkpoints": [c["name"] for c in saved_checkpoints],
    }), flush=True)


if __name__ == "__main__":
    main()
