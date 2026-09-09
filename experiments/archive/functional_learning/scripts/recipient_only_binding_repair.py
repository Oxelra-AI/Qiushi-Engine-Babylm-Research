#!/usr/bin/env python3
"""research repaired recipient-only entity-binding pilot.

This repairs two flaws in the research held-out entity-binding experiment:

1. Recipient-only contrast: UPDATE and RETAIN conditions share the same source,
   same new state, and same final prediction frame; only the update recipient
   changes. The query entity and source order are counterbalanced.
2. Symmetric candidate scoring: evaluation uses an explicit one-mask final-answer
   span and compares token-length-matched candidates at the same mask position.
   No partial foil scoring or arbitrary length penalty is used.

The training comparison keeps the learning budget fixed across arms and keeps
background corruption controlled: both standard and focused arms use the same
15% whole-word background process; focused additionally forces the final answer
position to be a [MASK] target. This tests whether recipient-sensitive learning
survives a repaired task before any mixed-corpus BabyLM intervention.
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
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Controlled but natural-language-like template substrate
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

# One-token color/state words under the inherited compliant16k tokenizer in the
# cloze contexts used here. The first attempted list included navy/teal, which
# split into two tokens and invalidated symmetric candidate scoring; keep only
# validated one-token candidates for the repaired diagnostic.
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


def render_source(template: Dict[str, str], order: Tuple[str, str], state_by_entity: Dict[str, str]) -> str:
    e1, e2 = order
    return template["source"].format(E1=e1, E2=e2, S1=state_by_entity[e1], S2=state_by_entity[e2])


def build_groups(entity_pairs: Sequence[Tuple[str, str]], prefix: str, templates: Sequence[Dict[str, str]],
                 state_words: Sequence[str], start_offset: int = 0) -> List[ContrastGroup]:
    """Build counterbalanced recipient-only groups.

    Each group has a fixed source, fixed new state, fixed final prediction frame,
    and two update sentences differing only in recipient. Query side and textual
    source order are counterbalanced. State words cycle across source and new
    roles so a role-word heuristic is not stable.
    """
    groups: List[ContrastGroup] = []
    gi = 0
    n_state = len(state_words)
    for ep_idx, (a, b) in enumerate(entity_pairs):
        # Two templates per entity pair, with deterministic but varied selection.
        # Use start_offset for held-out construction so held groups are not just
        # the first train state/template pattern with new names.
        off_ep = ep_idx + start_offset
        template_indices = [(off_ep + 0) % len(templates), (off_ep + 2) % len(templates)]
        for local_ti, ti in enumerate(template_indices):
            tmpl = templates[ti]
            pattern_idx = gi + start_offset
            query_is_a = (pattern_idx % 2 == 0)
            target = a if query_is_a else b
            distractor = b if query_is_a else a
            source_order_normal = ((pattern_idx // 2) % 2 == 0)
            order = (a, b) if source_order_normal else (b, a)

            # Cycle state words through target-source, distractor-source, and new-state
            # roles with balanced offsets. An earlier research attempt used a stride-3
            # source index with 12 states, so query-source states collapsed to only four
            # colors and reintroduced a role-word shortcut. These offsets make every
            # state appear equally often in all three roles over 12-group blocks.
            source_state = state_words[pattern_idx % n_state]
            distractor_source_state = state_words[(pattern_idx + n_state // 3) % n_state]
            new_state = state_words[(pattern_idx + 2 * n_state // 3) % n_state]
            # Ensure all three are distinct; with the 12-state list and offsets 0/4/8
            # this is always true, but keep the guard for future edits.
            used = {source_state}
            while distractor_source_state in used:
                distractor_source_state = state_words[(state_words.index(distractor_source_state) + 1) % n_state]
            used.add(distractor_source_state)
            while new_state in used:
                new_state = state_words[(state_words.index(new_state) + 1) % n_state]

            state_by = {target: source_state, distractor: distractor_source_state}
            source = render_source(tmpl, order, state_by)
            update_target = tmpl["update"].format(UE=target, NS=new_state)
            update_distractor = tmpl["update"].format(UE=distractor, NS=new_state)
            final_frame = tmpl["final"].format(QE=target, STATE="{STATE}")

            groups.append(ContrastGroup(
                group_id=f"{prefix}_{gi:03d}",
                template_name=tmpl["name"],
                source_sentence=source,
                update_target_sentence=update_target,
                update_distractor_sentence=update_distractor,
                final_frame=final_frame,
                target_entity=target,
                distractor_entity=distractor,
                query_side="A" if query_is_a else "B",
                source_order="AB" if source_order_normal else "BA",
                source_state=source_state,
                distractor_source_state=distractor_source_state,
                new_state=new_state,
            ))
            gi += 1
    return groups


def groups_to_training_packets(groups: Sequence[ContrastGroup]) -> List[Dict[str, Any]]:
    packets: List[Dict[str, Any]] = []
    for g in groups:
        for ptype, update_sentence, answer, foil in [
            ("UPDATE", g.update_target_sentence, g.new_state, g.source_state),
            ("RETAIN", g.update_distractor_sentence, g.source_state, g.new_state),
        ]:
            final_sentence = g.final_frame.format(STATE=answer)
            full_text = f"{g.source_sentence} {update_sentence} {final_sentence}"
            # Locate final answer span by construction in the last sentence.
            final_start = full_text.rfind(final_sentence)
            ans_in_final = final_sentence.rfind(answer)
            if final_start < 0 or ans_in_final < 0:
                raise RuntimeError(f"Could not locate answer span for {g.group_id} {ptype}")
            answer_char_start = final_start + ans_in_final
            answer_char_end = answer_char_start + len(answer)
            packets.append({
                "group_id": g.group_id,
                "packet_type": ptype,
                "full_text": full_text,
                "source_sentence": g.source_sentence,
                "update_sentence": update_sentence,
                "final_sentence": final_sentence,
                "final_frame": g.final_frame,
                "answer_text": answer,
                "foil_text": foil,
                "answer_char_start": answer_char_start,
                "answer_char_end": answer_char_end,
                "target_entity": g.target_entity,
                "distractor_entity": g.distractor_entity,
                "source_state": g.source_state,
                "distractor_source_state": g.distractor_source_state,
                "new_state": g.new_state,
                "template_name": g.template_name,
                "query_side": g.query_side,
                "source_order": g.source_order,
            })
    return packets


# ---------------------------------------------------------------------------
# Tokenization and scoring
# ---------------------------------------------------------------------------


def is_word_start(token_str: str) -> bool:
    return token_str.startswith("Ġ") or token_str.startswith("▁")


def locate_span_token_positions(offset_mapping: Sequence[Tuple[int, int]], start: int, end: int) -> List[int]:
    positions: List[int] = []
    for i, (s, e) in enumerate(offset_mapping):
        if e <= s:  # special token or empty offset
            continue
        if s < end and e > start:
            positions.append(i)
    return positions


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


def score_new_minus_source(model, tokenizer, masked_text: str, new_state: str, source_state: str,
                           device: torch.device) -> Dict[str, Any]:
    """Score log p(new_state) - log p(source_state) at one explicit final mask."""
    enc = tokenizer(masked_text, return_tensors="pt", add_special_tokens=True,
                    max_length=256, truncation=True)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    mask_positions = (input_ids[0] == tokenizer.mask_token_id).nonzero(as_tuple=False).flatten().tolist()
    if len(mask_positions) != 1:
        raise ValueError(f"Expected one mask, found {mask_positions} in {masked_text}")
    mpos = int(mask_positions[0])

    new_id, new_info = one_token_candidate_id(tokenizer, masked_text, new_state)
    source_id, source_info = one_token_candidate_id(tokenizer, masked_text, source_state)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[0, mpos]
        lp = torch.log_softmax(logits, dim=-1)
    top_id = int(logits.argmax())
    return {
        "new_lp": float(lp[new_id]),
        "source_lp": float(lp[source_id]),
        "margin_new_minus_source": float(lp[new_id] - lp[source_id]),
        "new_token_id": new_id,
        "source_token_id": source_id,
        "new_token": tokenizer.convert_ids_to_tokens(new_id),
        "source_token": tokenizer.convert_ids_to_tokens(source_id),
        "top_id": top_id,
        "top_token": tokenizer.convert_ids_to_tokens(top_id),
        "mask_position": mpos,
        "new_info": new_info,
        "source_info": source_info,
    }


def masked_final_text(source: str, update: Optional[str], final_frame: str, tokenizer) -> str:
    final = final_frame.format(STATE=tokenizer.mask_token)
    if update is None:
        return f"{source} {final}"
    return f"{source} {update} {final}"


def evaluate_groups(model, tokenizer, groups: Sequence[ContrastGroup], device: torch.device,
                    tag: str = "eval") -> Dict[str, Any]:
    """Evaluate recipient-sensitive binding on groups.

    All margins are log p(new_state) - log p(source_state), except the aggregated
    source-vs-new margins which are negated for RETAIN/NEUTRAL readability.
    """
    model.eval()
    per_group: List[Dict[str, Any]] = []
    for g in groups:
        target_masked = masked_final_text(g.source_sentence, g.update_target_sentence, g.final_frame, tokenizer)
        distractor_masked = masked_final_text(g.source_sentence, g.update_distractor_sentence, g.final_frame, tokenizer)
        neutral_masked = masked_final_text(g.source_sentence, None, g.final_frame, tokenizer)
        su = score_new_minus_source(model, tokenizer, target_masked, g.new_state, g.source_state, device)
        sr = score_new_minus_source(model, tokenizer, distractor_masked, g.new_state, g.source_state, device)
        sn = score_new_minus_source(model, tokenizer, neutral_masked, g.new_state, g.source_state, device)
        u = su["margin_new_minus_source"]
        r_new_minus_source = sr["margin_new_minus_source"]
        n_new_minus_source = sn["margin_new_minus_source"]
        row = {
            "group_id": g.group_id,
            "template_name": g.template_name,
            "target_entity": g.target_entity,
            "distractor_entity": g.distractor_entity,
            "query_side": g.query_side,
            "source_order": g.source_order,
            "source_state": g.source_state,
            "distractor_source_state": g.distractor_source_state,
            "new_state": g.new_state,
            "target_update_new_minus_source": u,
            "distractor_update_new_minus_source": r_new_minus_source,
            "neutral_new_minus_source": n_new_minus_source,
            "update_margin_new_over_source": u,
            "retain_margin_source_over_new": -r_new_minus_source,
            "neutral_margin_source_over_new": -n_new_minus_source,
            "recipient_sensitivity": u - r_new_minus_source,
            "retain_minus_neutral_source_margin": (-r_new_minus_source) - (-n_new_minus_source),
            "update_correct": u > 0.0,
            "retain_correct": r_new_minus_source < 0.0,
            "neutral_correct": n_new_minus_source < 0.0,
            "recipient_flip_correct": (u > 0.0 and r_new_minus_source < 0.0),
            "scores": {"target_update": su, "distractor_update": sr, "neutral": sn},
        }
        per_group.append(row)
    model.train()

    n = max(1, len(per_group))
    def mean(key: str) -> float:
        return sum(float(r[key]) for r in per_group) / n
    def count(key: str) -> int:
        return sum(1 for r in per_group if r[key])

    return {
        "tag": tag,
        "n_groups": len(per_group),
        "mean_update_new_minus_source": mean("update_margin_new_over_source"),
        "mean_retain_source_minus_new": mean("retain_margin_source_over_new"),
        "mean_neutral_source_minus_new": mean("neutral_margin_source_over_new"),
        "mean_recipient_sensitivity": mean("recipient_sensitivity"),
        "mean_retain_minus_neutral_source_margin": mean("retain_minus_neutral_source_margin"),
        "n_update_correct": count("update_correct"),
        "n_retain_correct": count("retain_correct"),
        "n_neutral_correct": count("neutral_correct"),
        "n_recipient_flip_correct": count("recipient_flip_correct"),
        "per_group": per_group,
    }


# ---------------------------------------------------------------------------
# Training dataset and masking
# ---------------------------------------------------------------------------


class RecipientPacketDataset(Dataset):
    def __init__(self, packets: Sequence[Dict[str, Any]], tokenizer, seq_length: int = 256):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.special_ids = set(int(x) for x in tokenizer.all_special_ids)
        self._word_start_cache: Dict[int, bool] = {}
        self.items: List[Dict[str, Any]] = []
        for pkt in packets:
            enc = tokenizer(pkt["full_text"], add_special_tokens=True, max_length=seq_length,
                            truncation=True, padding="max_length", return_offsets_mapping=True,
                            return_tensors="pt")
            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)
            offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"].squeeze(0).tolist()]
            answer_positions = locate_span_token_positions(offsets, int(pkt["answer_char_start"]), int(pkt["answer_char_end"]))
            if len(answer_positions) != 1:
                raise ValueError(f"Answer span for {pkt['group_id']} {pkt['packet_type']} maps to {len(answer_positions)} tokens: {answer_positions}; text={pkt['full_text']}")
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
                "group_id": pkt["group_id"],
                "packet_type": pkt["packet_type"],
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


def apply_masking(input_ids: torch.Tensor, attention_mask: torch.Tensor, word_group: torch.Tensor,
                  answer_pos: torch.Tensor, answer_gid: torch.Tensor, tokenizer,
                  mode: str, mask_prob: float, gen: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
    """Apply controlled WWM.

    mode='standard': standard 15% WWM with 80/10/10 replacement.
    mode='focused': same background WWM plus forced final-answer [MASK] labels.

    Background corruption is controlled by using the same mask_prob for all word
    groups in both modes. Focused answer tokens are always replaced by [MASK] so
    the selected answer is hidden rather than left unchanged by 80/10/10.
    """
    device = input_ids.device
    bsz, seq = input_ids.shape
    labels = input_ids.clone()
    token_mask = torch.zeros((bsz, seq), dtype=torch.bool, device=device)
    forced_answer = torch.zeros((bsz, seq), dtype=torch.bool, device=device)

    for b in range(bsz):
        max_gid = int(word_group[b].max().item())
        if max_gid < 0:
            continue
        group_mask = torch.rand(max_gid + 1, generator=gen, device=device) < mask_prob
        for g in range(max_gid + 1):
            if bool(group_mask[g].item()):
                token_mask[b] |= (word_group[b] == g)
        if mode == "focused":
            ap = int(answer_pos[b].item())
            if 0 <= ap < seq and int(attention_mask[b, ap].item()) == 1:
                token_mask[b, ap] = True
                forced_answer[b, ap] = True
        elif mode != "standard":
            raise ValueError(f"Unknown masking mode {mode}")

    labels[~token_mask] = -100
    masked_inputs = input_ids.clone()

    # Non-forced selected tokens use inherited 80/10/10 replacement.
    stochastic_selected = token_mask & ~forced_answer
    replace_mask = (torch.rand(input_ids.shape, generator=gen, device=device) < 0.8) & stochastic_selected
    masked_inputs[replace_mask] = int(tokenizer.mask_token_id)
    rand_mask = (torch.rand(input_ids.shape, generator=gen, device=device) < 0.5) & stochastic_selected & ~replace_mask
    if int(rand_mask.sum().item()) > 0:
        rand_ids = torch.randint(0, int(tokenizer.vocab_size), (int(rand_mask.sum().item()),),
                                 generator=gen, device=device)
        masked_inputs[rand_mask] = rand_ids

    # Forced focused answer is always a real [MASK].
    masked_inputs[forced_answer] = int(tokenizer.mask_token_id)

    answer_labeled = 0
    answer_forced = 0
    answer_left_visible = 0
    for b in range(bsz):
        ap = int(answer_pos[b].item())
        if 0 <= ap < seq and labels[b, ap].item() != -100:
            answer_labeled += 1
            if bool(forced_answer[b, ap].item()):
                answer_forced += 1
            if int(masked_inputs[b, ap].item()) == int(input_ids[b, ap].item()):
                answer_left_visible += 1
    stats = {
        "total_targets": int((labels != -100).sum().item()),
        "answer_labeled": int(answer_labeled),
        "answer_forced": int(answer_forced),
        "answer_left_visible_when_labeled": int(answer_left_visible),
        "background_targets": int((labels != -100).sum().item()) - int(answer_labeled),
    }
    return masked_inputs, labels, stats


# ---------------------------------------------------------------------------
# Model loading and experiment driver
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


def train_arm(arm_name: str, mode: str, model_path: pathlib.Path, tokenizer, train_groups: Sequence[ContrastGroup],
              held_groups: Sequence[ContrastGroup], args, device: torch.device) -> Dict[str, Any]:
    model, load_info = load_private_model(model_path, args.private_bottleneck, args.private_scale, device)
    trainable = freeze_to_private_adapters(model)
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    packets = groups_to_training_packets(train_groups)
    ds = RecipientPacketDataset(packets, tokenizer, seq_length=args.seq_length)
    dl_gen = torch.Generator()
    dl_gen.manual_seed(args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    mask_gen = torch.Generator(device=device)
    mask_gen.manual_seed(args.seed + (17 if mode == "standard" else 29))

    trajectory: List[Dict[str, Any]] = []
    target_stats_totals = defaultdict(int)

    def record(epoch: int, loss_value: Optional[float] = None):
        ev_train = evaluate_groups(model, tokenizer, train_groups, device, tag=f"{arm_name}_train_e{epoch}")
        ev_held = evaluate_groups(model, tokenizer, held_groups, device, tag=f"{arm_name}_held_e{epoch}")
        entry = {"epoch": epoch, "train": compact_eval(ev_train), "heldout": compact_eval(ev_held)}
        if loss_value is not None:
            entry["loss"] = loss_value
        trajectory.append(entry)
        print(
            f"[{arm_name}] epoch {epoch:04d}" + (f" loss={loss_value:.4f}" if loss_value is not None else "") +
            f" | train U={ev_train['mean_update_new_minus_source']:+.2f} R={ev_train['mean_retain_source_minus_new']:+.2f} "
            f"N={ev_train['mean_neutral_source_minus_new']:+.2f} flip={ev_train['n_recipient_flip_correct']}/{ev_train['n_groups']}" +
            f" | held U={ev_held['mean_update_new_minus_source']:+.2f} R={ev_held['mean_retain_source_minus_new']:+.2f} "
            f"N={ev_held['mean_neutral_source_minus_new']:+.2f} flip={ev_held['n_recipient_flip_correct']}/{ev_held['n_groups']}",
            flush=True,
        )

    record(0)
    model.train()
    for epoch in range(1, args.epochs + 1):
        epoch_loss_num = 0.0
        epoch_targets = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            word_group = batch["word_group"].to(device)
            answer_pos = batch["answer_pos"].to(device)
            answer_gid = batch["answer_gid"].to(device)
            masked_inputs, labels, st = apply_masking(
                input_ids, attention_mask, word_group, answer_pos, answer_gid,
                tokenizer, mode=mode, mask_prob=args.mask_prob, gen=mask_gen,
            )
            n_targets = int(st["total_targets"])
            if n_targets <= 0:
                continue
            out = model(input_ids=masked_inputs, attention_mask=attention_mask, labels=labels)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            opt.step()
            opt.zero_grad(set_to_none=True)
            epoch_loss_num += float(loss.detach()) * n_targets
            epoch_targets += n_targets
            for k, v in st.items():
                target_stats_totals[k] += int(v)
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            record(epoch, epoch_loss_num / max(1, epoch_targets))
            model.train()

    final_train_full = evaluate_groups(model, tokenizer, train_groups, device, tag=f"{arm_name}_train_final_full")
    final_held_full = evaluate_groups(model, tokenizer, held_groups, device, tag=f"{arm_name}_held_final_full")
    result = {
        "arm_name": arm_name,
        "mode": mode,
        "load_info": load_info,
        "n_trainable_tensors": len(trainable),
        "n_train_packets": len(packets),
        "n_train_groups": len(train_groups),
        "n_held_groups": len(held_groups),
        "epochs": args.epochs,
        "lr": args.lr,
        "mask_prob": args.mask_prob,
        "target_stats_totals": dict(target_stats_totals),
        "trajectory": trajectory,
        "final_train_eval": final_train_full,
        "final_held_eval": final_held_full,
    }
    del model, opt
    torch.cuda.empty_cache()
    return result


def compact_eval(ev: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "n_groups", "mean_update_new_minus_source", "mean_retain_source_minus_new",
        "mean_neutral_source_minus_new", "mean_recipient_sensitivity",
        "mean_retain_minus_neutral_source_margin", "n_update_correct", "n_retain_correct",
        "n_neutral_correct", "n_recipient_flip_correct",
    ]
    return {k: ev[k] for k in keys}


def validate_groups_tokenization(tokenizer, groups: Sequence[ContrastGroup]) -> Dict[str, Any]:
    issues: List[Dict[str, Any]] = []
    token_pairs = []
    for g in groups:
        for context_name, update in [
            ("target_update", g.update_target_sentence),
            ("distractor_update", g.update_distractor_sentence),
            ("neutral", None),
        ]:
            mt = masked_final_text(g.source_sentence, update, g.final_frame, tokenizer)
            try:
                new_id, new_info = one_token_candidate_id(tokenizer, mt, g.new_state)
                src_id, src_info = one_token_candidate_id(tokenizer, mt, g.source_state)
                token_pairs.append({
                    "group_id": g.group_id,
                    "context": context_name,
                    "new_state": g.new_state,
                    "source_state": g.source_state,
                    "new_token": tokenizer.convert_ids_to_tokens(new_id),
                    "source_token": tokenizer.convert_ids_to_tokens(src_id),
                    "new_id": new_id,
                    "source_id": src_id,
                })
            except Exception as e:
                issues.append({"group_id": g.group_id, "context": context_name, "error": repr(e), "masked_text": mt})
    role_counts = defaultdict(lambda: defaultdict(int))
    for g in groups:
        role_counts[g.source_state]["query_source"] += 1
        role_counts[g.distractor_source_state]["distractor_source"] += 1
        role_counts[g.new_state]["new"] += 1
    return {
        "n_groups": len(groups),
        "n_issues": len(issues),
        "issues": issues[:50],
        "token_pairs_sample": token_pairs[:20],
        "state_role_counts": {k: dict(v) for k, v in sorted(role_counts.items())},
        "query_side_counts": dict(defaultdict(int, {s: sum(1 for g in groups if g.query_side == s) for s in ["A", "B"]})),
        "source_order_counts": dict(defaultdict(int, {s: sum(1 for g in groups if g.source_order == s) for s in ["AB", "BA"]})),
    }


def write_jsonl(path: pathlib.Path, rows: Iterable[Dict[str, Any]]):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_markdown(path: pathlib.Path, summary: Dict[str, Any]):
    lines: List[str] = []
    lines.append("# research repaired recipient-only binding pilot\n\n")
    lines.append("## Repair implemented\n\n")
    lines.append("- Recipient-only contrast: UPDATE and RETAIN share source, new state, and final prediction frame; only update recipient changes.\n")
    lines.append("- Query entity and source order are counterbalanced; state words rotate through query-source, distractor-source, and new-state roles.\n")
    lines.append("- Evaluation uses one explicit final `[MASK]` and compares one-token candidates at the same mask position using tokenizer offsets.\n")
    lines.append("- Training comparison uses the same number of update steps. Both arms use 15% WWM background corruption; focused additionally forces the final answer token to `[MASK]`.\n\n")
    lines.append("## Dataset/token validation\n\n")
    tv = summary["tokenization_validation"]
    lines.append(f"- Train groups: {summary['n_train_groups']} ({summary['n_train_packets']} training packets)\n")
    lines.append(f"- Held-out groups: {summary['n_held_groups']}\n")
    lines.append(f"- Tokenization issues: train={tv['train']['n_issues']}, held={tv['heldout']['n_issues']}\n")
    lines.append(f"- Train query-side counts: {tv['train']['query_side_counts']} ; source-order counts: {tv['train']['source_order_counts']}\n")
    lines.append(f"- Held query-side counts: {tv['heldout']['query_side_counts']} ; source-order counts: {tv['heldout']['source_order_counts']}\n\n")
    lines.append("## Baseline coherent86\n\n")
    btr = summary["baseline_train_eval"]
    bhe = summary["baseline_held_eval"]
    lines.append("| split | U new>source | R source>new | N source>new | R-N | sensitivity | update | retain | neutral | recipient flip |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    def row(label: str, ev: Dict[str, Any]):
        return (f"| {label} | {ev['mean_update_new_minus_source']:+.3f} | {ev['mean_retain_source_minus_new']:+.3f} "
                f"| {ev['mean_neutral_source_minus_new']:+.3f} | {ev['mean_retain_minus_neutral_source_margin']:+.3f} "
                f"| {ev['mean_recipient_sensitivity']:+.3f} | {ev['n_update_correct']}/{ev['n_groups']} "
                f"| {ev['n_retain_correct']}/{ev['n_groups']} | {ev['n_neutral_correct']}/{ev['n_groups']} "
                f"| {ev['n_recipient_flip_correct']}/{ev['n_groups']} |\n")
    lines.append(row("train", compact_eval(btr)))
    lines.append(row("heldout", compact_eval(bhe)))
    lines.append("\n## Training comparison\n\n")
    lines.append("| arm | split | U new>source | R source>new | N source>new | R-N | sensitivity | update | retain | neutral | recipient flip | answer labels | answer visible | background targets |\n")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for arm in summary["arms"]:
        st = arm.get("target_stats_totals", {})
        for split_name, ev_key in [("train", "final_train_eval"), ("heldout", "final_held_eval")]:
            ev = compact_eval(arm[ev_key])
            lines.append(
                f"| {arm['arm_name']} | {split_name} | {ev['mean_update_new_minus_source']:+.3f} "
                f"| {ev['mean_retain_source_minus_new']:+.3f} | {ev['mean_neutral_source_minus_new']:+.3f} "
                f"| {ev['mean_retain_minus_neutral_source_margin']:+.3f} | {ev['mean_recipient_sensitivity']:+.3f} "
                f"| {ev['n_update_correct']}/{ev['n_groups']} | {ev['n_retain_correct']}/{ev['n_groups']} "
                f"| {ev['n_neutral_correct']}/{ev['n_groups']} | {ev['n_recipient_flip_correct']}/{ev['n_groups']} "
                f"| {st.get('answer_labeled', 0)} | {st.get('answer_left_visible_when_labeled', 0)} | {st.get('background_targets', 0)} |\n"
            )
    lines.append("\n## Interpretation\n\n")
    lines.append(summary.get("interpretation", ""))
    lines.append("\n\n## Files\n\n")
    for k, v in summary.get("files", {}).items():
        lines.append(f"- {k}: `{v}`\n")
    path.write_text("".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="models/frontier")
    ap.add_argument("--out-dir", default="experiments/archive/functional_learning/data/recipient_only_binding_repair")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--eval-every", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--mask-prob", type=float, default=0.15)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--seq-length", type=int, default=256)
    ap.add_argument("--private-bottleneck", type=int, default=128)
    ap.add_argument("--private-scale", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=43033)
    ap.add_argument("--held-pattern-offset", type=int, default=0,
                    help="Offset held-out template/state/query pattern; nonzero avoids held groups reusing the first train patterns with only new names.")
    ap.add_argument("--baseline-only", action="store_true")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = pathlib.Path(args.model_path)
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, use_fast=True)
    if tokenizer.mask_token is None:
        raise RuntimeError("Tokenizer has no mask token")
    print(f"Tokenizer loaded from {model_path}, vocab={tokenizer.vocab_size}, mask={tokenizer.mask_token}, device={device}", flush=True)

    train_groups = build_groups(TRAIN_ENTITY_PAIRS, "train", TEMPLATES, STATE_WORDS, start_offset=0)
    held_groups = build_groups(HELD_ENTITY_PAIRS, "held", TEMPLATES, STATE_WORDS, start_offset=args.held_pattern_offset)
    train_packets = groups_to_training_packets(train_groups)
    held_packets = groups_to_training_packets(held_groups)

    write_jsonl(out_dir / "train_groups.jsonl", [asdict(g) for g in train_groups])
    write_jsonl(out_dir / "heldout_groups.jsonl", [asdict(g) for g in held_groups])
    write_jsonl(out_dir / "train_packets.jsonl", train_packets)
    write_jsonl(out_dir / "heldout_packets.jsonl", held_packets)

    tokenization_validation = {
        "train": validate_groups_tokenization(tokenizer, train_groups),
        "heldout": validate_groups_tokenization(tokenizer, held_groups),
    }
    if tokenization_validation["train"]["n_issues"] or tokenization_validation["heldout"]["n_issues"]:
        (out_dir / "tokenization_validation.json").write_text(json.dumps(tokenization_validation, indent=2, ensure_ascii=False), encoding="utf-8")
        raise RuntimeError("Tokenization validation failed; see tokenization_validation.json")
    (out_dir / "tokenization_validation.json").write_text(json.dumps(tokenization_validation, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Built {len(train_groups)} train groups / {len(train_packets)} packets and {len(held_groups)} held groups; token validation passed", flush=True)

    # Baseline evaluation.
    base_model, load_info = load_private_model(model_path, args.private_bottleneck, args.private_scale, device)
    base_model.eval()
    baseline_train = evaluate_groups(base_model, tokenizer, train_groups, device, tag="baseline_train")
    baseline_held = evaluate_groups(base_model, tokenizer, held_groups, device, tag="baseline_held")
    del base_model
    torch.cuda.empty_cache()
    print(
        f"Baseline train U={baseline_train['mean_update_new_minus_source']:+.2f} R={baseline_train['mean_retain_source_minus_new']:+.2f} "
        f"N={baseline_train['mean_neutral_source_minus_new']:+.2f} flip={baseline_train['n_recipient_flip_correct']}/{baseline_train['n_groups']} | "
        f"held U={baseline_held['mean_update_new_minus_source']:+.2f} R={baseline_held['mean_retain_source_minus_new']:+.2f} "
        f"N={baseline_held['mean_neutral_source_minus_new']:+.2f} flip={baseline_held['n_recipient_flip_correct']}/{baseline_held['n_groups']}",
        flush=True,
    )

    arms: List[Dict[str, Any]] = []
    if not args.baseline_only and args.epochs > 0:
        arms.append(train_arm("standard_wwm15", "standard", model_path, tokenizer, train_groups, held_groups, args, device))
        arms.append(train_arm("focused_answer_plus_same_bg", "focused", model_path, tokenizer, train_groups, held_groups, args, device))

    # Form careful interpretation from observed values without overclaiming.
    interp_lines: List[str] = []
    interp_lines.append(
        "This repaired run should be read as a recipient-sensitive control, not as proof of a general principle by itself. "
        "The same new-state word is used in the target-update and distractor-update contexts, so a successful recipient flip must follow the updated entity rather than the state word alone. "
        "Absolute UPDATE, RETAIN, and NEUTRAL margins are reported because a change in R-N alone can be caused by a movement of either term."
    )
    if arms:
        std = arms[0]
        foc = arms[1]
        std_h = compact_eval(std["final_held_eval"])
        foc_h = compact_eval(foc["final_held_eval"])
        base_h = compact_eval(baseline_held)
        interp_lines.append(
            f" On held-out repaired groups, coherent86 baseline recipient-flip correctness is "
            f"{base_h['n_recipient_flip_correct']}/{base_h['n_groups']} with U={base_h['mean_update_new_minus_source']:+.2f}, "
            f"R={base_h['mean_retain_source_minus_new']:+.2f}, N={base_h['mean_neutral_source_minus_new']:+.2f}. "
            f"After the fixed-budget standard arm it is {std_h['n_recipient_flip_correct']}/{std_h['n_groups']} "
            f"with U={std_h['mean_update_new_minus_source']:+.2f}, R={std_h['mean_retain_source_minus_new']:+.2f}, N={std_h['mean_neutral_source_minus_new']:+.2f}; "
            f"after focused answer-plus-same-background it is {foc_h['n_recipient_flip_correct']}/{foc_h['n_groups']} "
            f"with U={foc_h['mean_update_new_minus_source']:+.2f}, R={foc_h['mean_retain_source_minus_new']:+.2f}, N={foc_h['mean_neutral_source_minus_new']:+.2f}."
        )
        if foc_h["n_recipient_flip_correct"] > std_h["n_recipient_flip_correct"] and foc_h["mean_update_new_minus_source"] > 0 and foc_h["mean_retain_source_minus_new"] > 0:
            interp_lines.append(
                " The focused arm gives the stronger repaired recipient-sensitive learning signal, but the conclusion remains bounded to this controlled packet substrate and private-adapter short run. "
                "The next practical bridge should test whether an ALN-preserving mixed objective can keep this repaired readout while preserving cheap7 downstream competence."
            )
        elif foc_h["n_recipient_flip_correct"] <= std_h["n_recipient_flip_correct"]:
            interp_lines.append(
                " The focused arm does not clearly outperform standard on the repaired held-out contrast, so the research stronger conclusion should be withdrawn; the experience design or scoring frame still permits shortcuts or the intervention fails to install recipient-sensitive selection."
            )
        else:
            interp_lines.append(
                " The repaired results are mixed; they should guide another construction/measurement repair before any expensive mixed-corpus run."
            )

    summary = {
        "status": "RECIPIENT_ONLY_BINDING_REPAIR",
        "model_path": str(model_path),
        "device": str(device),
        "seed": args.seed,
        "held_pattern_offset": args.held_pattern_offset,
        "n_train_groups": len(train_groups),
        "n_held_groups": len(held_groups),
        "n_train_packets": len(train_packets),
        "n_held_packets": len(held_packets),
        "tokenization_validation": tokenization_validation,
        "baseline_load_info": load_info,
        "baseline_train_eval": baseline_train,
        "baseline_held_eval": baseline_held,
        "arms": arms,
        "interpretation": "".join(interp_lines),
        "files": {
            "summary_json": str(out_dir / "recipient_only_binding_repair_summary.json"),
            "summary_md": str(out_dir / "recipient_only_binding_repair_summary.md"),
            "train_groups": str(out_dir / "train_groups.jsonl"),
            "heldout_groups": str(out_dir / "heldout_groups.jsonl"),
            "train_packets": str(out_dir / "train_packets.jsonl"),
            "heldout_packets": str(out_dir / "heldout_packets.jsonl"),
            "tokenization_validation": str(out_dir / "tokenization_validation.json"),
        },
    }
    (out_dir / "recipient_only_binding_repair_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(out_dir / "recipient_only_binding_repair_summary.md", summary)
    print(json.dumps({
        "status": summary["status"],
        "summary_json": summary["files"]["summary_json"],
        "summary_md": summary["files"]["summary_md"],
        "baseline_held_flip": f"{baseline_held['n_recipient_flip_correct']}/{baseline_held['n_groups']}",
        "arms": [
            {
                "arm": a["arm_name"],
                "held_flip": f"{a['final_held_eval']['n_recipient_flip_correct']}/{a['final_held_eval']['n_groups']}",
                "held_U": a['final_held_eval']['mean_update_new_minus_source'],
                "held_R": a['final_held_eval']['mean_retain_source_minus_new'],
                "held_N": a['final_held_eval']['mean_neutral_source_minus_new'],
            }
            for a in arms
        ],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
