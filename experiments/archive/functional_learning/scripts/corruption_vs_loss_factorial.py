#!/usr/bin/env python3
"""research: Corruption-vs-loss factorial for recipient-only binding.

Scientific purpose
------------------
research showed answer-only training with CLEAN input succeeds (10/12 held)
while focused-answer + 15% background fails (0/12 held). But the clean
answer-only arm removed BOTH background losses AND background corruption
of relational evidence. This script separates those two factors:

Arm 1 (clean_answer_only):   Clean input, answer-only loss     [replication]
Arm 2 (corrupted_answer_only): 15% bg corruption on input,      answer-only loss
Arm 3 (corrupted_full_weight): 15% bg corruption on input,      answer_loss + bg_loss
                                 (answer kept at full weight,     not diluted by bg count)

Arms 2 and 3 see IDENTICAL corrupted inputs. The only difference is
whether background masked positions contribute to the loss gradient.

If Arm 2 ≈ Arm 1 → corruption does not destroy usable evidence
                     → background gradients are the problem
If Arm 2 << Arm 1 → evidence corruption matters regardless of loss source
If Arm 3 ≈ Arm 2  → bg gradients do not interfere at matched answer weight
If Arm 3 << Arm 2  → bg gradients actively interfere even at full answer weight
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import pathlib
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Controlled template substrate (identical to research)
# ---------------------------------------------------------------------------

TRAIN_ENTITY_PAIRS = [
    ("Alice", "Bob"), ("Maria", "James"), ("Elena", "Marcus"),
    ("Tom", "Sarah"), ("Maya", "Leo"), ("Grace", "Henry"),
    ("Oliver", "Emma"), ("Lucas", "Mia"), ("Ethan", "Lily"),
    ("Professor Chen", "Dr. Reyes"), ("Chef Kim", "Chef Park"),
    ("Captain Reed", "Captain Blake"), ("Rebecca", "Nathan"),
    ("David", "Sophia"), ("Dr. Foster", "Dr. Singh"),
    ("Nora", "Felix"), ("Clara", "Samuel"), ("Iris", "Julian"),
]

HELD_ENTITY_PAIRS = [
    ("Daniel", "Rachel"), ("Mr. Thompson", "Ms. Garcia"),
    ("Professor Wright", "Professor Lin"), ("Victor", "Hannah"),
    ("Priya", "Omar"), ("Yara", "Noah"),
]

STATE_WORDS = [
    "red", "blue", "green", "yellow", "purple", "orange",
    "silver", "gold", "white", "black", "brown", "pink",
]

TEMPLATES = [
    {
        "name": "marker",
        "source": "In the workshop, {E1} started with {S1}, while {E2} started with {S2}.",
        "update": "Later, {UE} changed to {NS}.",
        "final": "At inspection, {QE}'s marker was {STATE}.",
    },
    {
        "name": "badge",
        "source": "During orientation, {E1} received {S1} and {E2} received {S2}.",
        "update": "Before lunch, {UE} switched to {NS}.",
        "final": "In the roster, {QE}'s badge was {STATE}.",
    },
    {
        "name": "folder",
        "source": "For the filing task, {E1} used {S1} and {E2} used {S2}.",
        "update": "After the meeting, {UE} moved to {NS}.",
        "final": "On the shelf, {QE}'s folder was {STATE}.",
    },
    {
        "name": "ticket",
        "source": "At the station, {E1} held {S1} and {E2} held {S2}.",
        "update": "When plans changed, {UE} took {NS}.",
        "final": "At the gate, {QE}'s ticket was {STATE}.",
    },
]


@dataclass
class ContrastGroup:
    group_id: str
    template_name: str
    source_sentence: str
    update_target_sentence: str
    update_distractor_sentence: str
    final_frame: str
    target_entity: str
    distractor_entity: str
    query_side: str
    source_order: str
    source_state: str
    distractor_source_state: str
    new_state: str

    def update_text(self) -> str:
        return f"{self.source_sentence} {self.update_target_sentence} {self.final_frame.format(STATE=self.new_state)}"

    def retain_text(self) -> str:
        return f"{self.source_sentence} {self.update_distractor_sentence} {self.final_frame.format(STATE=self.source_state)}"

    def neutral_text(self) -> str:
        return f"{self.source_sentence} {self.final_frame.format(STATE=self.source_state)}"


def build_groups(entity_pairs, prefix, templates, state_words, start_offset: int = 0) -> List[ContrastGroup]:
    groups: List[ContrastGroup] = []
    n_states = len(state_words)
    n_templates = len(templates)
    idx = 0
    for ep_i, (a, b) in enumerate(entity_pairs):
        t_i = (ep_i + start_offset) % n_templates
        tpl = templates[t_i]
        s_base = (ep_i * 3 + start_offset) % n_states
        s1 = state_words[s_base % n_states]
        s2 = state_words[(s_base + 1) % n_states]
        ns = state_words[(s_base + 2) % n_states]
        q_side = "A" if (ep_i + start_offset) % 2 == 0 else "B"
        src_order = "AB" if (ep_i + start_offset) % 2 == 0 else "BA"
        if q_side == "A":
            target_entity, distractor_entity = a, b
            query_src, dist_src = s1, s2
        else:
            target_entity, distractor_entity = b, a
            query_src, dist_src = s2, s1
        if src_order == "AB":
            e1, e2, st1, st2 = a, b, s1, s2
        else:
            e1, e2, st1, st2 = b, a, s2, s1
        source_sent = tpl["source"].format(E1=e1, E2=e2, S1=st1, S2=st2)
        update_target = tpl["update"].format(UE=target_entity, NS=ns)
        update_distractor = tpl["update"].format(UE=distractor_entity, NS=ns)
        final_frame = tpl["final"].format(QE=target_entity, STATE="{STATE}")
        for dup in range(2):
            gid = f"{prefix}_{idx:03d}"
            groups.append(ContrastGroup(
                group_id=gid, template_name=tpl["name"],
                source_sentence=source_sent,
                update_target_sentence=update_target,
                update_distractor_sentence=update_distractor,
                final_frame=final_frame,
                target_entity=target_entity, distractor_entity=distractor_entity,
                query_side=q_side, source_order=src_order,
                source_state=query_src, distractor_source_state=dist_src, new_state=ns,
            ))
            idx += 1
    return groups


def groups_to_training_packets(groups: Sequence[ContrastGroup]) -> List[Dict[str, Any]]:
    packets: List[Dict[str, Any]] = []
    for g in groups:
        for ptype, text, exp_answer in [
            ("update", g.update_text(), g.new_state),
            ("retain", g.retain_text(), g.source_state),
        ]:
            mask_pos_text = g.final_frame.format(STATE="<mask>")
            full = text
            answer_start = full.rfind(exp_answer)
            if answer_start < 0:
                raise ValueError(f"Cannot find answer '{exp_answer}' in '{full}'")
            answer_end = answer_start + len(exp_answer)
            packets.append({
                "group_id": g.group_id, "packet_type": ptype, "full_text": full,
                "answer_text": exp_answer, "answer_char_start": answer_start,
                "answer_char_end": answer_end,
            })
    return packets


def is_word_start(token_str: str) -> bool:
    return token_str.startswith("Ġ") or token_str.startswith("▁")


def masked_final_text(source: str, update_sent: Optional[str], final_frame: str, tokenizer) -> str:
    if update_sent:
        return f"{source} {update_sent} {final_frame.format(STATE=tokenizer.mask_token)}"
    else:
        return f"{source} {final_frame.format(STATE=tokenizer.mask_token)}"


def one_token_candidate_id(tokenizer, masked_text: str, candidate: str) -> Tuple[int, Dict[str, Any]]:
    """Return the token id for candidate at the mask span, requiring exactly one token.

    The candidate token is determined by filling the same final mask position and
    inspecting offsets, so RoBERTa/DeBERTa space-prefix tokens are handled exactly.
    """
    mask_token = tokenizer.mask_token
    mask_pos = masked_text.index(mask_token)
    filled = masked_text[:mask_pos] + candidate + masked_text[mask_pos + len(mask_token):]
    enc = tokenizer(filled, return_offsets_mapping=True, add_special_tokens=True,
                    max_length=256, truncation=True)
    offsets = enc["offset_mapping"]
    pos = locate_span_token_positions(offsets, mask_pos, mask_pos + len(candidate))
    info = {
        "candidate": candidate,
        "mask_char_start": mask_pos,
        "candidate_token_positions": pos,
        "candidate_tokens": [tokenizer.convert_ids_to_tokens(int(enc["input_ids"][i])) for i in pos],
        "candidate_token_ids": [int(enc["input_ids"][i]) for i in pos],
        "filled_text": filled,
    }
    if len(pos) != 1:
        raise ValueError(f"Candidate {candidate!r} maps to {len(pos)} tokens in context: {info}")
    return int(enc["input_ids"][pos[0]]), info


def locate_span_token_positions(offsets: List[Tuple[int, int]], char_start: int, char_end: int) -> List[int]:
    positions = []
    for i, (cs, ce) in enumerate(offsets):
        if ce > char_start and cs < char_end and cs != ce:
            positions.append(i)
    return positions


# ---------------------------------------------------------------------------
# Evaluation (identical to research)
# ---------------------------------------------------------------------------

def evaluate_groups(model, tokenizer, groups: Sequence[ContrastGroup], device: torch.device,
                    tag: str = "") -> Dict[str, Any]:
    model.eval()
    results: List[Dict[str, Any]] = []
    for g in groups:
        scores: Dict[str, Dict[str, float]] = {}
        for condition_name, text, expected_answer in [
            ("update", g.update_text(), g.new_state),
            ("retain", g.retain_text(), g.source_state),
            ("neutral", g.neutral_text(), g.source_state),
        ]:
            update_sent = g.update_target_sentence if condition_name == "update" else (
                g.update_distractor_sentence if condition_name == "retain" else None
            )
            mt = masked_final_text(g.source_sentence, update_sent, g.final_frame, tokenizer)
            enc = tokenizer(mt, add_special_tokens=True, return_tensors="pt", max_length=256,
                           truncation=True, padding="max_length", return_offsets_mapping=True)
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"].squeeze(0).tolist()]
            mask_positions = [i for i, tid in enumerate(input_ids.squeeze(0).tolist())
                            if tid == tokenizer.mask_token_id]
            if len(mask_positions) != 1:
                raise ValueError(f"Expected 1 mask for {g.group_id} {condition_name}, found {len(mask_positions)}")
            mask_pos = mask_positions[0]
            new_id, _ = one_token_candidate_id(tokenizer, mt, g.new_state)
            src_id, _ = one_token_candidate_id(tokenizer, mt, g.source_state)
            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            log_probs = F.log_softmax(logits[0, mask_pos], dim=-1)
            lp_new = float(log_probs[new_id].item())
            lp_src = float(log_probs[src_id].item())
            top_id = int(logits[0, mask_pos].argmax().item())
            scores[condition_name] = {
                "lp_new": lp_new, "lp_source": lp_src,
                "margin_new_minus_source": lp_new - lp_src,
                "top_token": tokenizer.convert_ids_to_tokens(top_id),
            }
        u_margin = scores["update"]["margin_new_minus_source"]
        r_margin_src = -scores["retain"]["margin_new_minus_source"]
        n_margin_src = -scores["neutral"]["margin_new_minus_source"]
        results.append({
            "group_id": g.group_id,
            "update_new_minus_source": u_margin,
            "retain_source_minus_new": r_margin_src,
            "neutral_source_minus_new": n_margin_src,
            "retain_minus_neutral": r_margin_src - n_margin_src,
            "recipient_sensitivity": u_margin - r_margin_src,
            "update_correct": u_margin > 0,
            "retain_correct": r_margin_src > 0,
            "neutral_correct": n_margin_src > 0,
            "recipient_flip_correct": u_margin > 0 and r_margin_src > 0,
            "scores": scores,
        })
    n = len(results)
    return {
        "tag": tag,
        "n_groups": n,
        "mean_update_new_minus_source": sum(r["update_new_minus_source"] for r in results) / max(1, n),
        "mean_retain_source_minus_new": sum(r["retain_source_minus_new"] for r in results) / max(1, n),
        "mean_neutral_source_minus_new": sum(r["neutral_source_minus_new"] for r in results) / max(1, n),
        "mean_recipient_sensitivity": sum(r["recipient_sensitivity"] for r in results) / max(1, n),
        "mean_retain_minus_neutral_source_margin": sum(r["retain_minus_neutral"] for r in results) / max(1, n),
        "n_update_correct": sum(1 for r in results if r["update_correct"]),
        "n_retain_correct": sum(1 for r in results if r["retain_correct"]),
        "n_neutral_correct": sum(1 for r in results if r["neutral_correct"]),
        "n_recipient_flip_correct": sum(1 for r in results if r["recipient_flip_correct"]),
        "per_group": results,
    }


def compact_eval(ev: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "n_groups", "mean_update_new_minus_source", "mean_retain_source_minus_new",
        "mean_neutral_source_minus_new", "mean_recipient_sensitivity",
        "mean_retain_minus_neutral_source_margin", "n_update_correct", "n_retain_correct",
        "n_neutral_correct", "n_recipient_flip_correct",
    ]
    return {k: ev[k] for k in keys}


# ---------------------------------------------------------------------------
# Dataset with explicit masking separation
# ---------------------------------------------------------------------------

class RecipientPacketDataset(Dataset):
    def __init__(self, packets, tokenizer, seq_length: int = 256):
        self.tokenizer = tokenizer
        self.special_ids = set()
        for attr in ("cls_token_id", "sep_token_id", "pad_token_id", "mask_token_id"):
            v = getattr(tokenizer, attr, None)
            if v is not None:
                self.special_ids.add(int(v))
        self._word_start_cache: Dict[int, bool] = {}
        self.items: List[Dict[str, Any]] = []
        for pkt in packets:
            enc = tokenizer(pkt["full_text"], add_special_tokens=True, max_length=seq_length,
                           truncation=True, padding="max_length", return_offsets_mapping=True,
                           return_tensors="pt")
            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)
            offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"].squeeze(0).tolist()]
            answer_positions = locate_span_token_positions(offsets, int(pkt["answer_char_start"]),
                                                          int(pkt["answer_char_end"]))
            if len(answer_positions) != 1:
                raise ValueError(f"Answer span for {pkt['group_id']} {pkt['packet_type']} maps to "
                               f"{len(answer_positions)} tokens; text={pkt['full_text']}")
            answer_pos = int(answer_positions[0])
            word_group = torch.full((seq_length,), -1, dtype=torch.long)
            gid = -1
            for i in range(seq_length):
                if int(attention_mask[i]) == 0:
                    continue
                tid = int(input_ids[i])
                if tid in self.special_ids:
                    continue
                if gid < 0 or self._word_start_flag(tid) or i == 0:
                    gid += 1
                word_group[i] = gid
            answer_gid = int(word_group[answer_pos].item())
            if answer_gid < 0:
                raise ValueError(f"Answer token has no word group for {pkt['group_id']} {pkt['packet_type']}")
            self.items.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "word_group": word_group,
                "answer_pos": answer_pos,
                "answer_gid": answer_gid,
            })

    def _word_start_flag(self, tid: int) -> bool:
        v = self._word_start_cache.get(tid)
        if v is None:
            s = self.tokenizer.convert_ids_to_tokens(int(tid))
            v = bool(s is not None and is_word_start(str(s)))
            self._word_start_cache[tid] = v
        return v

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]


def collate_fn(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "word_group": torch.stack([b["word_group"] for b in batch]),
        "answer_pos": torch.tensor([int(b["answer_pos"]) for b in batch], dtype=torch.long),
        "answer_gid": torch.tensor([int(b["answer_gid"]) for b in batch], dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Masking: separates input corruption from label assignment
# ---------------------------------------------------------------------------

def apply_corruption_and_labels(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    word_group: torch.Tensor,
    answer_pos: torch.Tensor,
    answer_gid: torch.Tensor,
    tokenizer,
    bg_mask_prob: float,
    label_mode: str,  # "answer_only" or "answer_plus_bg"
    gen: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, int]]:
    """Apply background corruption to input and assign labels separately.

    This separates two things that research's masking conflated:
    1. Input corruption (which tokens are replaced/masked in the input)
    2. Label assignment (which positions contribute to loss)

    Background corruption (bg_mask_prob) determines which background tokens
    are corrupted in the input using 80/10/10 replacement.
    The forced answer position is ALWAYS replaced by [MASK].

    label_mode determines which positions get labels (contribute to loss):
    - "answer_only": only the forced answer position gets a label
    - "answer_plus_bg": both answer and bg corrupted positions get labels

    Returns: (masked_inputs, answer_labels, bg_labels, stats)
    - answer_labels: shape (bsz, seq), -100 everywhere except answer position
    - bg_labels: shape (bsz, seq), -100 everywhere except bg masked positions
    Both label tensors are always computed; the training loop decides which to use.
    """
    device = input_ids.device
    bsz, seq = input_ids.shape

    # --- research: Determine which background positions to corrupt ---
    bg_selected = torch.zeros((bsz, seq), dtype=torch.bool, device=device)
    if bg_mask_prob > 0:
        for b in range(bsz):
            max_gid = int(word_group[b].max().item())
            if max_gid < 0:
                continue
            group_mask = torch.rand(max_gid + 1, generator=gen, device=device) < bg_mask_prob
            for g in range(max_gid + 1):
                if bool(group_mask[g].item()):
                    bg_selected[b] |= (word_group[b] == g)
            # Ensure the answer word group is NOT selected as background
            agid = int(answer_gid[b].item())
            if agid >= 0:
                bg_selected[b] &= ~(word_group[b] == agid)

    # --- research: Apply input corruption ---
    masked_inputs = input_ids.clone()

    # Background: standard 80/10/10 replacement
    if bg_selected.any():
        replace_mask = (torch.rand(input_ids.shape, generator=gen, device=device) < 0.8) & bg_selected
        masked_inputs[replace_mask] = int(tokenizer.mask_token_id)
        rand_mask = (torch.rand(input_ids.shape, generator=gen, device=device) < 0.5) & bg_selected & ~replace_mask
        if int(rand_mask.sum().item()) > 0:
            rand_ids = torch.randint(0, int(tokenizer.vocab_size),
                                     (int(rand_mask.sum().item()),), generator=gen, device=device)
            masked_inputs[rand_mask] = rand_ids

    # Forced answer: always [MASK]
    forced_answer = torch.zeros((bsz, seq), dtype=torch.bool, device=device)
    for b in range(bsz):
        ap = int(answer_pos[b].item())
        if 0 <= ap < seq and int(attention_mask[b, ap].item()) == 1:
            forced_answer[b, ap] = True
            masked_inputs[b, ap] = int(tokenizer.mask_token_id)

    # --- research: Assign labels (separate from corruption) ---
    answer_labels = torch.full_like(input_ids, -100)
    bg_labels = torch.full_like(input_ids, -100)

    # Answer labels: always set for the forced answer position
    for b in range(bsz):
        ap = int(answer_pos[b].item())
        if 0 <= ap < seq:
            answer_labels[b, ap] = input_ids[b, ap]

    # Background labels: set for all bg-corrupted positions
    bg_labels[bg_selected] = input_ids[bg_selected]

    # --- Stats ---
    n_bg_corrupted = int(bg_selected.sum().item())
    n_answer_forced = int(forced_answer.sum().item())
    n_bg_replaced_mask = int(((masked_inputs == int(tokenizer.mask_token_id)) & bg_selected).sum().item())

    # Check: how many answer positions have their evidence corrupted?
    # (i.e., entity names, source states, etc. that are near the answer)
    stats = {
        "n_bg_corrupted_positions": n_bg_corrupted,
        "n_answer_forced": n_answer_forced,
        "n_bg_mask_replaced": n_bg_replaced_mask,
        "bg_mask_prob": bg_mask_prob,
        "label_mode": label_mode,
    }
    return masked_inputs, answer_labels, bg_labels, stats


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_private_model(model_path: pathlib.Path, private_bottleneck: int, private_scale: float,
                       device: torch.device):
    from transformers import DebertaV2Config
    from safetensors.torch import load_file
    sys.path.insert(0, str(pathlib.Path("experiments/archive/functional_learning/scripts")))
    from context_credit_trainer import FrozenSlowPrivateDebertaV2ForMaskedLM
    cfg = DebertaV2Config.from_pretrained(str(model_path), local_files_only=True)
    cfg.private_adapter_bottleneck = int(private_bottleneck)
    cfg.private_adapter_scale = float(private_scale)
    cfg.private_adapter_enabled = True
    model = FrozenSlowPrivateDebertaV2ForMaskedLM(cfg)
    sd = load_file(str(model_path / "model.safetensors"), device="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    private_missing = [x for x in missing if "private_adapter" in x]
    if private_missing:
        raise RuntimeError(f"Missing private adapter keys: {private_missing[:10]}")
    for m in model.modules():
        if hasattr(m, "private_adapter_enabled"):
            m.private_adapter_enabled = True
    model.to(device)
    return model, {"missing": list(missing), "unexpected": list(unexpected)}


def freeze_to_private_adapters(model) -> List[torch.nn.Parameter]:
    params: List[torch.nn.Parameter] = []
    for _, p in model.named_parameters():
        p.requires_grad_(False)
    for name, p in model.named_parameters():
        if "private_adapter" in name:
            p.requires_grad_(True)
            params.append(p)
    if not params:
        raise RuntimeError("No private_adapter parameters found")
    return params


# ---------------------------------------------------------------------------
# Training with explicit loss separation
# ---------------------------------------------------------------------------

def train_arm(arm_name: str, bg_mask_prob: float, use_bg_loss: bool,
              model_path: pathlib.Path, tokenizer,
              train_groups: Sequence[ContrastGroup], held_groups: Sequence[ContrastGroup],
              args, device: torch.device) -> Dict[str, Any]:
    """Train one arm of the factorial.

    bg_mask_prob: probability of background WWM corruption on input
    use_bg_loss: whether background labels contribute to loss gradient
    """
    model, load_info = load_private_model(model_path, args.private_bottleneck,
                                          args.private_scale, device)
    trainable = freeze_to_private_adapters(model)
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    packets = groups_to_training_packets(train_groups)
    ds = RecipientPacketDataset(packets, tokenizer, seq_length=args.seq_length)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    mask_gen = torch.Generator(device=device)
    mask_gen.manual_seed(args.seed + hash(arm_name) % 10000)

    trajectory: List[Dict[str, Any]] = []
    cum_stats = defaultdict(int)

    def record(epoch: int, loss_val: Optional[float] = None):
        ev_tr = evaluate_groups(model, tokenizer, train_groups, device, tag=f"{arm_name}_tr_e{epoch}")
        ev_he = evaluate_groups(model, tokenizer, held_groups, device, tag=f"{arm_name}_he_e{epoch}")
        entry = {"epoch": epoch, "train": compact_eval(ev_tr), "held": compact_eval(ev_he)}
        if loss_val is not None:
            entry["loss"] = loss_val
        trajectory.append(entry)
        print(
            f"[{arm_name}] e{epoch:04d}" + (f" L={loss_val:.4f}" if loss_val is not None else "") +
            f" | tr U={ev_tr['mean_update_new_minus_source']:+.3f} R={ev_tr['mean_retain_source_minus_new']:+.3f} "
            f"N={ev_tr['mean_neutral_source_minus_new']:+.3f} "
            f"flip={ev_tr['n_recipient_flip_correct']}/{ev_tr['n_groups']}"
            f" | he U={ev_he['mean_update_new_minus_source']:+.3f} R={ev_he['mean_retain_source_minus_new']:+.3f} "
            f"N={ev_he['mean_neutral_source_minus_new']:+.3f} "
            f"flip={ev_he['n_recipient_flip_correct']}/{ev_he['n_groups']}",
            flush=True,
        )

    record(0)
    model.train()
    label_mode = "answer_plus_bg" if use_bg_loss else "answer_only"

    for epoch in range(1, args.epochs + 1):
        epoch_answer_loss = 0.0
        epoch_bg_loss = 0.0
        epoch_answer_n = 0
        epoch_bg_n = 0

        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            word_group = batch["word_group"].to(device)
            answer_pos = batch["answer_pos"].to(device)
            answer_gid = batch["answer_gid"].to(device)

            masked_inputs, answer_labels, bg_labels, st = apply_corruption_and_labels(
                input_ids, attention_mask, word_group, answer_pos, answer_gid,
                tokenizer, bg_mask_prob=bg_mask_prob, label_mode=label_mode, gen=mask_gen,
            )

            # Forward pass (no built-in loss; compute manually)
            outputs = model(input_ids=masked_inputs, attention_mask=attention_mask)
            logits = outputs.logits  # (bsz, seq, vocab)

            # Answer loss: CE at the forced answer positions
            answer_mask = (answer_labels != -100)
            n_answer = int(answer_mask.sum().item())
            if n_answer > 0:
                answer_loss = F.cross_entropy(
                    logits[answer_mask].view(-1, logits.size(-1)),
                    answer_labels[answer_mask].view(-1),
                    reduction="mean",
                )
            else:
                answer_loss = torch.tensor(0.0, device=device)

            # Background loss: CE at bg-corrupted positions
            bg_mask = (bg_labels != -100)
            n_bg = int(bg_mask.sum().item())
            if use_bg_loss and n_bg > 0:
                bg_loss = F.cross_entropy(
                    logits[bg_mask].view(-1, logits.size(-1)),
                    bg_labels[bg_mask].view(-1),
                    reduction="mean",
                )
            else:
                bg_loss = torch.tensor(0.0, device=device)

            # Total loss: answer at full weight + bg at full weight
            # This keeps the answer gradient identical across arms with different bg settings
            loss = answer_loss + bg_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            opt.step()
            opt.zero_grad(set_to_none=True)

            epoch_answer_loss += float(answer_loss.detach()) * max(1, n_answer)
            epoch_bg_loss += float(bg_loss.detach()) * max(1, n_bg)
            epoch_answer_n += n_answer
            epoch_bg_n += n_bg

            for k, v in st.items():
                if isinstance(v, (int, float)):
                    cum_stats[k] += v

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            avg_a = epoch_answer_loss / max(1, epoch_answer_n)
            avg_b = epoch_bg_loss / max(1, epoch_bg_n) if epoch_bg_n > 0 else 0.0
            record(epoch, avg_a + avg_b)
            model.train()

    final_tr = evaluate_groups(model, tokenizer, train_groups, device, tag=f"{arm_name}_final_tr")
    final_he = evaluate_groups(model, tokenizer, held_groups, device, tag=f"{arm_name}_final_he")
    result = {
        "arm_name": arm_name,
        "bg_mask_prob": bg_mask_prob,
        "use_bg_loss": use_bg_loss,
        "label_mode": label_mode,
        "n_trainable": len(trainable),
        "epochs": args.epochs,
        "lr": args.lr,
        "cumulative_stats": dict(cum_stats),
        "trajectory": trajectory,
        "final_train_eval": final_tr,
        "final_held_eval": final_he,
    }
    del model, opt
    torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_groups_tokenization(tokenizer, groups: Sequence[ContrastGroup]) -> Dict[str, Any]:
    issues: List[Dict[str, Any]] = []
    for g in groups:
        for ctx, upd in [("target_update", g.update_target_sentence),
                         ("distractor_update", g.update_distractor_sentence),
                         ("neutral", None)]:
            mt = masked_final_text(g.source_sentence, upd, g.final_frame, tokenizer)
            try:
                one_token_candidate_id(tokenizer, mt, g.new_state)
                one_token_candidate_id(tokenizer, mt, g.source_state)
            except Exception as e:
                issues.append({"group_id": g.group_id, "context": ctx, "error": repr(e)})
    return {
        "n_groups": len(groups), "n_issues": len(issues), "issues": issues[:20],
        "query_side_counts": {s: sum(1 for g in groups if g.query_side == s) for s in ["A", "B"]},
        "source_order_counts": {s: sum(1 for g in groups if g.source_order == s) for s in ["AB", "BA"]},
    }


def write_jsonl(path: pathlib.Path, rows: Iterable[Dict[str, Any]]):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="models/frontier")
    ap.add_argument("--out-dir", default="experiments/archive/functional_learning/data/corruption_vs_loss_factorial")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--mask-prob", type=float, default=0.15)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--seq-length", type=int, default=256)
    ap.add_argument("--private-bottleneck", type=int, default=128)
    ap.add_argument("--private-scale", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=43034)
    ap.add_argument("--held-pattern-offset", type=int, default=7)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = pathlib.Path(args.model_path)
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, use_fast=True)
    print(f"Tokenizer loaded, vocab={tokenizer.vocab_size}, mask={tokenizer.mask_token}, device={device}",
          flush=True)

    train_groups = build_groups(TRAIN_ENTITY_PAIRS, "train", TEMPLATES, STATE_WORDS, start_offset=0)
    held_groups = build_groups(HELD_ENTITY_PAIRS, "held", TEMPLATES, STATE_WORDS,
                               start_offset=args.held_pattern_offset)

    # Validate tokenization
    tv = {
        "train": validate_groups_tokenization(tokenizer, train_groups),
        "heldout": validate_groups_tokenization(tokenizer, held_groups),
    }
    assert tv["train"]["n_issues"] == 0 and tv["heldout"]["n_issues"] == 0, \
        f"Tokenization issues: train={tv['train']['n_issues']}, held={tv['heldout']['n_issues']}"
    print(f"Built {len(train_groups)} train / {len(held_groups)} held groups; tokens valid", flush=True)

    # Save groups
    write_jsonl(out_dir / "train_groups.jsonl", [asdict(g) for g in train_groups])
    write_jsonl(out_dir / "heldout_groups.jsonl", [asdict(g) for g in held_groups])

    # Baseline
    base_model, _ = load_private_model(model_path, args.private_bottleneck, args.private_scale, device)
    base_model.eval()
    baseline_tr = evaluate_groups(base_model, tokenizer, train_groups, device, tag="baseline_tr")
    baseline_he = evaluate_groups(base_model, tokenizer, held_groups, device, tag="baseline_he")
    del base_model
    torch.cuda.empty_cache()
    print(f"Baseline: tr flip={baseline_tr['n_recipient_flip_correct']}/{baseline_tr['n_groups']} "
          f"he flip={baseline_he['n_recipient_flip_correct']}/{baseline_he['n_groups']}", flush=True)

    # === Three-arm factorial ===
    arms: List[Dict[str, Any]] = []

    # Arm 1: Clean input, answer-only loss (replication of research answer-only)
    print("\n=== ARM 1: clean_answer_only (bg_prob=0, answer-only loss) ===", flush=True)
    arms.append(train_arm(
        "clean_answer_only", bg_mask_prob=0.0, use_bg_loss=False,
        model_path=model_path, tokenizer=tokenizer,
        train_groups=train_groups, held_groups=held_groups, args=args, device=device,
    ))

    # Arm 2: Corrupted input, answer-only loss (separates evidence from gradients)
    print("\n=== ARM 2: corrupted_answer_only (bg_prob=0.15, answer-only loss) ===", flush=True)
    arms.append(train_arm(
        "corrupted_answer_only", bg_mask_prob=args.mask_prob, use_bg_loss=False,
        model_path=model_path, tokenizer=tokenizer,
        train_groups=train_groups, held_groups=held_groups, args=args, device=device,
    ))

    # Arm 3: Corrupted input, answer + bg loss (answer at full weight)
    print("\n=== ARM 3: corrupted_answer_plus_bg (bg_prob=0.15, answer + bg loss, explicit weight) ===", flush=True)
    arms.append(train_arm(
        "corrupted_answer_plus_bg", bg_mask_prob=args.mask_prob, use_bg_loss=True,
        model_path=model_path, tokenizer=tokenizer,
        train_groups=train_groups, held_groups=held_groups, args=args, device=device,
    ))

    # === Save results ===
    summary = {
        "status": "CORRUPTION_VS_LOSS_FACTORIAL",
        "design": {
            "purpose": "Separate evidence corruption from loss competition in recipient-only binding",
            "arm1_clean_answer_only": "bg_prob=0.0, answer-only loss (reference)",
            "arm2_corrupted_answer_only": "bg_prob=0.15 on input, answer-only loss (tests evidence sufficiency)",
            "arm3_corrupted_answer_plus_bg": "bg_prob=0.15 on input, answer + bg loss at full answer weight (tests bg gradient interference)",
            "answer_coefficient": "answer_loss computed separately at full weight; not diluted by bg target count",
            "key_comparison": "arm2 vs arm1 → evidence effect; arm3 vs arm2 → bg gradient effect",
        },
        "model_path": str(model_path),
        "device": str(device),
        "seed": args.seed,
        "held_pattern_offset": args.held_pattern_offset,
        "epochs": args.epochs,
        "lr": args.lr,
        "mask_prob": args.mask_prob,
        "n_train_groups": len(train_groups),
        "n_held_groups": len(held_groups),
        "tokenization_validation": tv,
        "baseline_train": compact_eval(baseline_tr),
        "baseline_held": compact_eval(baseline_he),
        "arms": [{
            "arm_name": a["arm_name"],
            "bg_mask_prob": a["bg_mask_prob"],
            "use_bg_loss": a["use_bg_loss"],
            "final_train": compact_eval(a["final_train_eval"]),
            "final_held": compact_eval(a["final_held_eval"]),
            "cumulative_stats": a["cumulative_stats"],
            "trajectory": a["trajectory"],
        } for a in arms],
    }

    # Interpretation
    a1_he = compact_eval(arms[0]["final_held_eval"])
    a2_he = compact_eval(arms[1]["final_held_eval"])
    a3_he = compact_eval(arms[2]["final_held_eval"])
    interp = []
    interp.append(f"Arm 1 (clean, answer-only): held flip {a1_he['n_recipient_flip_correct']}/{a1_he['n_groups']}, "
                  f"U={a1_he['mean_update_new_minus_source']:+.3f}, R={a1_he['mean_retain_source_minus_new']:+.3f}, "
                  f"N={a1_he['mean_neutral_source_minus_new']:+.3f}")
    interp.append(f"Arm 2 (corrupted, answer-only): held flip {a2_he['n_recipient_flip_correct']}/{a2_he['n_groups']}, "
                  f"U={a2_he['mean_update_new_minus_source']:+.3f}, R={a2_he['mean_retain_source_minus_new']:+.3f}, "
                  f"N={a2_he['mean_neutral_source_minus_new']:+.3f}")
    interp.append(f"Arm 3 (corrupted, answer+bg): held flip {a3_he['n_recipient_flip_correct']}/{a3_he['n_groups']}, "
                  f"U={a3_he['mean_update_new_minus_source']:+.3f}, R={a3_he['mean_retain_source_minus_new']:+.3f}, "
                  f"N={a3_he['mean_neutral_source_minus_new']:+.3f}")

    if a2_he["n_recipient_flip_correct"] >= a1_he["n_recipient_flip_correct"] - 2:
        interp.append("→ Evidence corruption does NOT destroy usable relational information under 15% WWM.")
        if a3_he["n_recipient_flip_correct"] < a2_he["n_recipient_flip_correct"] - 2:
            interp.append("→ Background GRADIENTS are the destructive factor, not input corruption.")
            interp.append("→ BabyLM design: preserve relational packet content, isolate answer credit from bg loss.")
        else:
            interp.append("→ Background gradients also do not interfere at matched answer weight.")
            interp.append("→ The research failure was loss DILUTION (answer weight diluted by bg target count).")
    else:
        interp.append("→ Evidence corruption IS a significant factor under 15% WWM.")
        interp.append("→ BabyLM design must protect relational evidence from masking, not just adjust loss weights.")

    summary["interpretation"] = "\n".join(interp)

    (out_dir / "factorial_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # Markdown summary
    md_lines = ["# research corruption-vs-loss factorial\n\n"]
    md_lines.append("## Design\n\n")
    md_lines.append("| Arm | Input corruption | Loss | Answer coefficient |\n")
    md_lines.append("|---|---|---|---|\n")
    md_lines.append("| 1. clean_answer_only | None | Answer only | Full |\n")
    md_lines.append("| 2. corrupted_answer_only | 15% WWM | Answer only | Full |\n")
    md_lines.append("| 3. corrupted_answer_plus_bg | 15% WWM | Answer + bg | Answer full, bg separate |\n\n")
    md_lines.append("## Results\n\n")
    md_lines.append("| Arm | Held flip | Held U | Held R | Held N | Held R-N |\n")
    md_lines.append("|---|---|---|---|---|---|\n")
    for i, a_he in enumerate([a1_he, a2_he, a3_he]):
        name = ["clean_answer_only", "corrupted_answer_only", "corrupted_answer_plus_bg"][i]
        md_lines.append(f"| {name} | {a_he['n_recipient_flip_correct']}/{a_he['n_groups']} "
                       f"| {a_he['mean_update_new_minus_source']:+.3f} "
                       f"| {a_he['mean_retain_source_minus_new']:+.3f} "
                       f"| {a_he['mean_neutral_source_minus_new']:+.3f} "
                       f"| {a_he['mean_retain_minus_neutral_source_margin']:+.3f} |\n")
    md_lines.append(f"\nBaseline: held flip {compact_eval(baseline_he)['n_recipient_flip_correct']}/{compact_eval(baseline_he)['n_groups']}\n\n")
    md_lines.append("## Interpretation\n\n")
    md_lines.append(summary["interpretation"])
    md_lines.append("\n")

    (out_dir / "factorial_summary.md").write_text("".join(md_lines), encoding="utf-8")

    print("\n" + "="*80, flush=True)
    print(json.dumps({
        "status": summary["status"],
        "out_dir": str(out_dir),
        "baseline_held_flip": f"{compact_eval(baseline_he)['n_recipient_flip_correct']}/{compact_eval(baseline_he)['n_groups']}",
        "arms": [
            {
                "arm": a["arm_name"],
                "held_flip": f"{compact_eval(a['final_held_eval'])['n_recipient_flip_correct']}/{compact_eval(a['final_held_eval'])['n_groups']}",
                "held_U": compact_eval(a["final_held_eval"])["mean_update_new_minus_source"],
                "held_R": compact_eval(a["final_held_eval"])["mean_retain_source_minus_new"],
                "held_N": compact_eval(a["final_held_eval"])["mean_neutral_source_minus_new"],
            }
            for a in arms
        ],
        "interpretation": summary["interpretation"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
