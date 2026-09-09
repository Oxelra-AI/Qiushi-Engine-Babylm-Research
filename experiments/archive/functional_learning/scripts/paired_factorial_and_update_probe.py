#!/usr/bin/env python3
"""research: paired short comparison for background-objective effects.

Scientific purpose
------------------
research suggested that adding background MLM loss can harm recipient-sensitive
binding, but the three arms did not receive a strictly paired corruption stream:
the mask generator used an arm-name-dependent seed and student dropout RNG was
not reset between arms. This script repairs that comparison.

Key repairs:
  * Arms use the same deterministic corruption seed for each epoch/batch.
  * Student dropout RNG is reset to the same deterministic seed before each
    forward pass in every arm.
  * The template groups are unique rather than duplicated.
  * Query side and source order are varied independently so the queried entity
    is not always first in the source sentence.

The endpoint comparison is still a controlled template experiment. It tests this
optimizer/substrate, not ordinary ALN learning. A compact one-step update probe
from a common trained snapshot additionally measures answer-gradient and
background-gradient norms, cosine, clipping coefficients, and immediate loss
changes under answer-only, bg-only, and combined updates.
"""

from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import copy
import json
import math
import pathlib
import random
import sys
from collections import defaultdict
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

SCRIPT_DIR = _public_path('experiments/archive/functional_learning/scripts')
sys.path.insert(0, str(SCRIPT_DIR))

from corruption_vs_loss_factorial import (  # noqa: E402
    TRAIN_ENTITY_PAIRS, HELD_ENTITY_PAIRS, STATE_WORDS, TEMPLATES,
    ContrastGroup, groups_to_training_packets, evaluate_groups, compact_eval,
    RecipientPacketDataset, collate_fn, apply_corruption_and_labels,
    load_private_model, freeze_to_private_adapters, validate_groups_tokenization,
    write_jsonl,
)


def seed_all(seed: int):
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def make_torch_generator(device: torch.device, seed: int) -> torch.Generator:
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    return gen


def step_seed(base: int, epoch: int, batch_idx: int, stream: int = 0) -> int:
    # Deterministic, independent of arm name and Python hash randomization.
    return int(base + 1000003 * epoch + 9176 * batch_idx + 104729 * stream)


def build_groups_decoupled_unique(entity_pairs, prefix: str, templates, state_words,
                                  start_offset: int = 0) -> List[ContrastGroup]:
    """Build unique recipient-only groups with independent query/source position.

    Two groups are produced per entity pair, but they are not duplicates: the
    template, state triple, queried side, and/or source order changes. The four
    (query_side, source_order) combinations are cycled so that the target entity
    appears first and second in the source about equally often.
    """
    combos = [("A", "AB"), ("A", "BA"), ("B", "AB"), ("B", "BA")]
    groups: List[ContrastGroup] = []
    n_states = len(state_words)
    n_templates = len(templates)
    idx = 0
    seen_texts = set()
    for ep_i, (a, b) in enumerate(entity_pairs):
        for variant in range(2):
            combo_i = (ep_i * 2 + variant + start_offset) % len(combos)
            q_side, src_order = combos[combo_i]
            tpl = templates[(ep_i * 2 + variant + start_offset) % n_templates]
            s_base = (ep_i * 5 + variant * 2 + start_offset) % n_states
            s1 = state_words[s_base % n_states]
            s2 = state_words[(s_base + 1) % n_states]
            ns = state_words[(s_base + 2) % n_states]
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
            gid = f"{prefix}_{idx:03d}"
            g = ContrastGroup(
                group_id=gid,
                template_name=tpl["name"],
                source_sentence=source_sent,
                update_target_sentence=update_target,
                update_distractor_sentence=update_distractor,
                final_frame=final_frame,
                target_entity=target_entity,
                distractor_entity=distractor_entity,
                query_side=q_side,
                source_order=src_order,
                source_state=query_src,
                distractor_source_state=dist_src,
                new_state=ns,
            )
            text_key = (g.source_sentence, g.update_target_sentence,
                        g.update_distractor_sentence, g.final_frame)
            if text_key in seen_texts:
                raise RuntimeError(f"Duplicate group text produced for {gid}")
            seen_texts.add(text_key)
            groups.append(g)
            idx += 1
    return groups


def group_balance(groups: Sequence[ContrastGroup]) -> Dict[str, Any]:
    def target_position(g: ContrastGroup) -> str:
        first_is_a = g.source_order == "AB"
        target_is_a = g.query_side == "A"
        return "target_first" if first_is_a == target_is_a else "target_second"
    return {
        "n_groups": len(groups),
        "query_side_counts": {s: sum(1 for g in groups if g.query_side == s) for s in ["A", "B"]},
        "source_order_counts": {s: sum(1 for g in groups if g.source_order == s) for s in ["AB", "BA"]},
        "target_source_position_counts": {
            s: sum(1 for g in groups if target_position(g) == s)
            for s in ["target_first", "target_second"]
        },
        "template_counts": {
            t["name"]: sum(1 for g in groups if g.template_name == t["name"])
            for t in TEMPLATES
        },
        "unique_text_keys": len({(g.source_sentence, g.update_target_sentence,
                                  g.update_distractor_sentence, g.final_frame) for g in groups}),
    }


def make_loader(train_groups: Sequence[ContrastGroup], tokenizer, seq_length: int, batch_size: int):
    packets = groups_to_training_packets(train_groups)
    ds = RecipientPacketDataset(packets, tokenizer, seq_length=seq_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return packets, ds, loader


def losses_from_logits(logits: torch.Tensor, answer_labels: torch.Tensor, bg_labels: torch.Tensor,
                       use_bg_loss: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    answer_mask = (answer_labels != -100)
    bg_mask = (bg_labels != -100)
    n_answer = int(answer_mask.sum().item())
    n_bg = int(bg_mask.sum().item())
    if n_answer > 0:
        answer_loss = F.cross_entropy(
            logits[answer_mask].view(-1, logits.size(-1)),
            answer_labels[answer_mask].view(-1),
            reduction="mean",
        )
    else:
        answer_loss = torch.tensor(0.0, device=logits.device)
    if use_bg_loss and n_bg > 0:
        bg_loss = F.cross_entropy(
            logits[bg_mask].view(-1, logits.size(-1)),
            bg_labels[bg_mask].view(-1),
            reduction="mean",
        )
    else:
        bg_loss = torch.tensor(0.0, device=logits.device)
    return answer_loss, bg_loss, answer_loss + bg_loss, n_answer, n_bg


def train_one_arm(arm_name: str, bg_mask_prob: float, use_bg_loss: bool,
                  model_path: pathlib.Path, tokenizer,
                  train_groups: Sequence[ContrastGroup], held_groups: Sequence[ContrastGroup],
                  args, device: torch.device,
                  return_state: bool = False) -> Dict[str, Any]:
    seed_all(args.seed)
    model, load_info = load_private_model(model_path, args.private_bottleneck,
                                          args.private_scale, device)
    trainable = freeze_to_private_adapters(model)
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    packets, ds, loader = make_loader(train_groups, tokenizer, args.seq_length, args.batch_size)

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
            f" | tr flip={ev_tr['n_recipient_flip_correct']}/{ev_tr['n_groups']}"
            f" U={ev_tr['mean_update_new_minus_source']:+.3f}"
            f" R={ev_tr['mean_retain_source_minus_new']:+.3f}"
            f" | he flip={ev_he['n_recipient_flip_correct']}/{ev_he['n_groups']}"
            f" U={ev_he['mean_update_new_minus_source']:+.3f}"
            f" R={ev_he['mean_retain_source_minus_new']:+.3f}"
            f" N={ev_he['mean_neutral_source_minus_new']:+.3f}",
            flush=True,
        )

    record(0)
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_answer_loss = 0.0
        epoch_bg_loss = 0.0
        epoch_answer_n = 0
        epoch_bg_n = 0
        for batch_idx, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            word_group = batch["word_group"].to(device)
            answer_pos = batch["answer_pos"].to(device)
            answer_gid = batch["answer_gid"].to(device)
            gen = make_torch_generator(device, step_seed(args.mask_seed, epoch, batch_idx))
            masked_inputs, answer_labels, bg_labels, st = apply_corruption_and_labels(
                input_ids, attention_mask, word_group, answer_pos, answer_gid,
                tokenizer, bg_mask_prob=bg_mask_prob,
                label_mode="answer_plus_bg" if use_bg_loss else "answer_only",
                gen=gen,
            )
            seed_all(step_seed(args.dropout_seed, epoch, batch_idx))
            outputs = model(input_ids=masked_inputs, attention_mask=attention_mask)
            answer_loss, bg_loss, loss, n_answer, n_bg = losses_from_logits(
                outputs.logits, answer_labels, bg_labels, use_bg_loss=use_bg_loss)
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
                    cum_stats[k] += int(v)
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            avg = epoch_answer_loss / max(1, epoch_answer_n)
            if use_bg_loss and epoch_bg_n > 0:
                avg += epoch_bg_loss / max(1, epoch_bg_n)
            record(epoch, avg)

    final_tr = evaluate_groups(model, tokenizer, train_groups, device, tag=f"{arm_name}_final_tr")
    final_he = evaluate_groups(model, tokenizer, held_groups, device, tag=f"{arm_name}_final_he")
    result = {
        "arm_name": arm_name,
        "bg_mask_prob": bg_mask_prob,
        "use_bg_loss": use_bg_loss,
        "paired_mask_seed_rule": "mask_seed + 1000003*epoch + 9176*batch_idx; identical across arms",
        "paired_dropout_seed_rule": "dropout_seed + 1000003*epoch + 9176*batch_idx; identical across arms",
        "cumulative_stats": dict(cum_stats),
        "trajectory": trajectory,
        "final_train": final_tr,
        "final_held": final_he,
    }
    if return_state:
        result["_model"] = model
        result["_opt"] = opt
        result["_trainable"] = trainable
    else:
        del model, opt
        torch.cuda.empty_cache()
    return result


def flatten_grads(params: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    chunks = []
    for p in params:
        if p.grad is None:
            chunks.append(torch.zeros_like(p.detach()).reshape(-1))
        else:
            chunks.append(p.grad.detach().reshape(-1).clone())
    if not chunks:
        return torch.empty(0)
    return torch.cat(chunks)


def fixed_first_batch(loader, device: torch.device) -> Dict[str, torch.Tensor]:
    batch = next(iter(loader))
    return {k: v.to(device) for k, v in batch.items()}


def make_corrupted_batch(batch: Dict[str, torch.Tensor], tokenizer, args, device: torch.device,
                         seed: int, bg_prob: float = 0.15):
    gen = make_torch_generator(device, seed)
    return apply_corruption_and_labels(
        batch["input_ids"], batch["attention_mask"], batch["word_group"],
        batch["answer_pos"], batch["answer_gid"], tokenizer,
        bg_mask_prob=bg_prob, label_mode="answer_plus_bg", gen=gen,
    )


def compute_loss_pair(model, batch, tokenizer, args, device, corruption_seed: int,
                      dropout_seed: int, train_mode: bool) -> Dict[str, float]:
    if train_mode:
        model.train()
        seed_all(dropout_seed)
    else:
        model.eval()
    masked_inputs, answer_labels, bg_labels, st = make_corrupted_batch(
        batch, tokenizer, args, device, seed=corruption_seed, bg_prob=args.mask_prob)
    with torch.set_grad_enabled(train_mode):
        outputs = model(input_ids=masked_inputs, attention_mask=batch["attention_mask"])
        a_loss, b_loss, _, n_a, n_b = losses_from_logits(
            outputs.logits, answer_labels, bg_labels, use_bg_loss=True)
    return {
        "answer_loss": float(a_loss.detach().item()),
        "bg_loss": float(b_loss.detach().item()),
        "n_answer": int(n_a),
        "n_bg": int(n_b),
        "n_bg_corrupted_positions": int(st["n_bg_corrupted_positions"]),
    }


def grad_for_kind(model, trainable, batch, tokenizer, args, device, kind: str,
                  corruption_seed: int, dropout_seed: int) -> Tuple[torch.Tensor, Dict[str, float]]:
    model.train()
    model.zero_grad(set_to_none=True)
    seed_all(dropout_seed)
    masked_inputs, answer_labels, bg_labels, st = make_corrupted_batch(
        batch, tokenizer, args, device, seed=corruption_seed, bg_prob=args.mask_prob)
    outputs = model(input_ids=masked_inputs, attention_mask=batch["attention_mask"])
    a_loss, b_loss, _, n_a, n_b = losses_from_logits(outputs.logits, answer_labels, bg_labels, use_bg_loss=True)
    if kind == "answer":
        loss = a_loss
    elif kind == "bg":
        loss = b_loss
    elif kind == "combined":
        loss = a_loss + b_loss
    else:
        raise ValueError(kind)
    loss.backward()
    grad = flatten_grads(trainable).detach().cpu()
    model.zero_grad(set_to_none=True)
    return grad, {
        "kind": kind,
        "answer_loss": float(a_loss.detach().item()),
        "bg_loss": float(b_loss.detach().item()),
        "loss_used": float(loss.detach().item()),
        "n_answer": int(n_a),
        "n_bg": int(n_b),
        "n_bg_corrupted_positions": int(st["n_bg_corrupted_positions"]),
    }


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    na = float(a.norm().item())
    nb = float(b.norm().item())
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(torch.dot(a, b).item() / (na * nb))


def clip_coef(norm: float, max_norm: float) -> float:
    if norm <= 0:
        return 1.0
    return float(min(1.0, max_norm / (norm + 1e-12)))


def optimizer_state_to_cpu(opt_state: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(opt_state)


def state_dict_to_cpu(model) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def restore_state(model, opt, model_state, opt_state):
    model.load_state_dict(model_state, strict=True)
    opt.load_state_dict(copy.deepcopy(opt_state))


def one_update_from_snapshot(model, opt, trainable, batch, tokenizer, args, device,
                             model_state, opt_state, kind: str,
                             corruption_seed: int, dropout_seed: int,
                             train_groups, held_groups) -> Dict[str, Any]:
    restore_state(model, opt, model_state, opt_state)
    before_losses = compute_loss_pair(model, batch, tokenizer, args, device,
                                      corruption_seed, dropout_seed, train_mode=False)
    before_held = compact_eval(evaluate_groups(model, tokenizer, held_groups, device, tag=f"probe_before_{kind}"))
    model.train()
    model.zero_grad(set_to_none=True)
    seed_all(dropout_seed)
    masked_inputs, answer_labels, bg_labels, st = make_corrupted_batch(
        batch, tokenizer, args, device, seed=corruption_seed, bg_prob=args.mask_prob)
    outputs = model(input_ids=masked_inputs, attention_mask=batch["attention_mask"])
    a_loss, b_loss, _, n_a, n_b = losses_from_logits(outputs.logits, answer_labels, bg_labels, use_bg_loss=True)
    if kind == "answer":
        loss = a_loss
    elif kind == "bg":
        loss = b_loss
    elif kind == "combined":
        loss = a_loss + b_loss
    else:
        raise ValueError(kind)
    loss.backward()
    raw_norm = float(torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm).item())
    opt.step()
    opt.zero_grad(set_to_none=True)
    after_losses = compute_loss_pair(model, batch, tokenizer, args, device,
                                     corruption_seed, dropout_seed, train_mode=False)
    after_held = compact_eval(evaluate_groups(model, tokenizer, held_groups, device, tag=f"probe_after_{kind}"))
    return {
        "kind": kind,
        "loss_used": float(loss.detach().item()),
        "answer_loss_train_forward": float(a_loss.detach().item()),
        "bg_loss_train_forward": float(b_loss.detach().item()),
        "n_answer": int(n_a),
        "n_bg": int(n_b),
        "raw_grad_norm_before_clip": raw_norm,
        "clip_coef": clip_coef(raw_norm, args.max_grad_norm),
        "before_losses_eval_mode": before_losses,
        "after_losses_eval_mode": after_losses,
        "delta_answer_loss_eval_mode": after_losses["answer_loss"] - before_losses["answer_loss"],
        "delta_bg_loss_eval_mode": after_losses["bg_loss"] - before_losses["bg_loss"],
        "before_held": before_held,
        "after_held": after_held,
        "delta_held_flip": after_held["n_recipient_flip_correct"] - before_held["n_recipient_flip_correct"],
        "delta_held_U": after_held["mean_update_new_minus_source"] - before_held["mean_update_new_minus_source"],
        "delta_held_R": after_held["mean_retain_source_minus_new"] - before_held["mean_retain_source_minus_new"],
    }


def run_update_probe(answer_arm_result: Dict[str, Any], tokenizer, train_groups, held_groups,
                     args, device: torch.device) -> Dict[str, Any]:
    model = answer_arm_result["_model"]
    opt = answer_arm_result["_opt"]
    trainable = answer_arm_result["_trainable"]
    packets, ds, loader = make_loader(train_groups, tokenizer, args.seq_length, args.batch_size)
    batch = fixed_first_batch(loader, device)
    corruption_seed = step_seed(args.mask_seed, args.epochs + 1, 0, stream=7)
    dropout_seed = step_seed(args.dropout_seed, args.epochs + 1, 0, stream=7)

    model_state = state_dict_to_cpu(model)
    opt_state = optimizer_state_to_cpu(opt.state_dict())

    g_answer, info_answer = grad_for_kind(model, trainable, batch, tokenizer, args, device,
                                          "answer", corruption_seed, dropout_seed)
    restore_state(model, opt, model_state, opt_state)
    g_bg, info_bg = grad_for_kind(model, trainable, batch, tokenizer, args, device,
                                  "bg", corruption_seed, dropout_seed)
    restore_state(model, opt, model_state, opt_state)
    g_combined, info_combined = grad_for_kind(model, trainable, batch, tokenizer, args, device,
                                              "combined", corruption_seed, dropout_seed)
    a_norm = float(g_answer.norm().item())
    b_norm = float(g_bg.norm().item())
    c_norm = float(g_combined.norm().item())
    grad_stats = {
        "snapshot": "final corrupted_answer_only arm",
        "fixed_batch_size": int(batch["input_ids"].shape[0]),
        "corruption_seed": int(corruption_seed),
        "dropout_seed": int(dropout_seed),
        "answer_grad_norm": a_norm,
        "bg_grad_norm": b_norm,
        "combined_grad_norm": c_norm,
        "answer_bg_cosine": cosine(g_answer, g_bg),
        "answer_combined_cosine": cosine(g_answer, g_combined),
        "bg_combined_cosine": cosine(g_bg, g_combined),
        "answer_grad_dot_bg_grad": float(torch.dot(g_answer, g_bg).item()),
        "answer_clip_coef_if_alone": clip_coef(a_norm, args.max_grad_norm),
        "bg_clip_coef_if_alone": clip_coef(b_norm, args.max_grad_norm),
        "combined_clip_coef": clip_coef(c_norm, args.max_grad_norm),
        "forward_info_answer_loss": info_answer,
        "forward_info_bg_loss": info_bg,
        "forward_info_combined_loss": info_combined,
    }
    one_step = []
    for kind in ["answer", "bg", "combined"]:
        one_step.append(one_update_from_snapshot(
            model, opt, trainable, batch, tokenizer, args, device,
            model_state, opt_state, kind, corruption_seed, dropout_seed,
            train_groups, held_groups,
        ))
    restore_state(model, opt, model_state, opt_state)
    return {"grad_stats": grad_stats, "one_step_updates": one_step}


def strip_private_objects(arm: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in arm.items() if not k.startswith("_")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="models/frontier")
    ap.add_argument("--out-dir", default="experiments/archive/functional_learning/data/paired_factorial_update_probe")
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
    ap.add_argument("--seed", type=int, default=43035)
    ap.add_argument("--mask-seed", type=int, default=935000)
    ap.add_argument("--dropout-seed", type=int, default=1935000)
    ap.add_argument("--held-pattern-offset", type=int, default=7)
    ap.add_argument("--include-clean", action="store_true")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = pathlib.Path(args.model_path)
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, use_fast=True)
    print(f"Tokenizer loaded, vocab={tokenizer.vocab_size}, mask={tokenizer.mask_token}, device={device}", flush=True)

    train_groups = build_groups_decoupled_unique(TRAIN_ENTITY_PAIRS, "train", TEMPLATES, STATE_WORDS, start_offset=0)
    held_groups = build_groups_decoupled_unique(HELD_ENTITY_PAIRS, "held", TEMPLATES, STATE_WORDS,
                                                start_offset=args.held_pattern_offset)
    tv = {"train": validate_groups_tokenization(tokenizer, train_groups),
          "heldout": validate_groups_tokenization(tokenizer, held_groups)}
    if tv["train"]["n_issues"] != 0 or tv["heldout"]["n_issues"] != 0:
        raise RuntimeError(f"Tokenization issues: train={tv['train']['n_issues']} held={tv['heldout']['n_issues']}")
    print(f"Built {len(train_groups)} unique train / {len(held_groups)} unique held groups", flush=True)
    print(f"Train balance: {json.dumps(group_balance(train_groups), ensure_ascii=False)}", flush=True)
    print(f"Held balance: {json.dumps(group_balance(held_groups), ensure_ascii=False)}", flush=True)
    write_jsonl(out_dir / "train_groups_decoupled_unique.jsonl", [asdict(g) for g in train_groups])
    write_jsonl(out_dir / "heldout_groups_decoupled_unique.jsonl", [asdict(g) for g in held_groups])

    base_model, _ = load_private_model(model_path, args.private_bottleneck, args.private_scale, device)
    baseline_train_full = evaluate_groups(base_model, tokenizer, train_groups, device, tag="baseline_train")
    baseline_held_full = evaluate_groups(base_model, tokenizer, held_groups, device, tag="baseline_held")
    baseline_train = compact_eval(baseline_train_full)
    baseline_held = compact_eval(baseline_held_full)
    del base_model
    torch.cuda.empty_cache()
    print(f"Baseline held flip={baseline_held['n_recipient_flip_correct']}/{baseline_held['n_groups']} "
          f"U={baseline_held['mean_update_new_minus_source']:+.3f} "
          f"R={baseline_held['mean_retain_source_minus_new']:+.3f} "
          f"N={baseline_held['mean_neutral_source_minus_new']:+.3f}", flush=True)

    arms: List[Dict[str, Any]] = []
    if args.include_clean:
        print("\n=== ARM 1: clean_answer_only (reference, no background corruption/loss) ===", flush=True)
        arms.append(train_one_arm(
            "clean_answer_only", 0.0, False, model_path, tokenizer,
            train_groups, held_groups, args, device, return_state=False))

    print("\n=== ARM 2: paired_corrupted_answer_only ===", flush=True)
    answer_arm = train_one_arm(
        "paired_corrupted_answer_only", args.mask_prob, False, model_path, tokenizer,
        train_groups, held_groups, args, device, return_state=True)
    arms.append(answer_arm)

    print("\n=== ARM 3: paired_corrupted_answer_plus_bg ===", flush=True)
    combined_arm = train_one_arm(
        "paired_corrupted_answer_plus_bg", args.mask_prob, True, model_path, tokenizer,
        train_groups, held_groups, args, device, return_state=False)
    arms.append(combined_arm)

    print("\n=== ONE-STEP UPDATE PROBE FROM ANSWER-ONLY SNAPSHOT ===", flush=True)
    update_probe = run_update_probe(answer_arm, tokenizer, train_groups, held_groups, args, device)

    clean = next((a for a in arms if a["arm_name"] == "clean_answer_only"), None)
    ans = answer_arm
    cmb = combined_arm
    ans_he = compact_eval(ans["final_held"])
    cmb_he = compact_eval(cmb["final_held"])
    interp_lines = []
    if clean is not None:
        cl_he = compact_eval(clean["final_held"])
        interp_lines.append(
            f"Clean answer-only: held flip={cl_he['n_recipient_flip_correct']}/{cl_he['n_groups']} "
            f"U={cl_he['mean_update_new_minus_source']:+.3f} R={cl_he['mean_retain_source_minus_new']:+.3f}")
    interp_lines.append(
        f"Paired corrupted answer-only: held flip={ans_he['n_recipient_flip_correct']}/{ans_he['n_groups']} "
        f"U={ans_he['mean_update_new_minus_source']:+.3f} R={ans_he['mean_retain_source_minus_new']:+.3f} "
        f"N={ans_he['mean_neutral_source_minus_new']:+.3f}")
    interp_lines.append(
        f"Paired corrupted answer+bg: held flip={cmb_he['n_recipient_flip_correct']}/{cmb_he['n_groups']} "
        f"U={cmb_he['mean_update_new_minus_source']:+.3f} R={cmb_he['mean_retain_source_minus_new']:+.3f} "
        f"N={cmb_he['mean_neutral_source_minus_new']:+.3f}")
    if ans["cumulative_stats"].get("n_bg_corrupted_positions") == cmb["cumulative_stats"].get("n_bg_corrupted_positions"):
        interp_lines.append("The two corrupted arms have matched cumulative background-corruption counts under the deterministic stream.")
    else:
        interp_lines.append("The two corrupted arms have different cumulative corruption counts; inspect stream implementation before interpreting endpoints.")
    interp_lines.append(
        "The paired endpoint tests whether adding background loss under identical corruption/dropout streams changes recipient-sensitive behavior in this controlled template optimizer; it does not by itself establish ordinary ALN dynamics or a universal gradient-direction law.")
    gs = update_probe["grad_stats"]
    interp_lines.append(
        f"Update probe at answer-only snapshot: ||g_answer||={gs['answer_grad_norm']:.4g}, "
        f"||g_bg||={gs['bg_grad_norm']:.4g}, cos(answer,bg)={gs['answer_bg_cosine']:+.4f}, "
        f"combined clip coef={gs['combined_clip_coef']:.4g}.")

    summary = {
        "status": "PAIRED_FACTORIAL_UPDATE_PROBE",
        "model_path": str(model_path),
        "device": str(device),
        "epochs": args.epochs,
        "lr": args.lr,
        "mask_prob": args.mask_prob,
        "seed": args.seed,
        "mask_seed": args.mask_seed,
        "dropout_seed": args.dropout_seed,
        "held_pattern_offset": args.held_pattern_offset,
        "group_design": {
            "unique_groups_no_duplicate_copies": True,
            "query_side_and_source_order_independently_cycled": True,
            "train_balance": group_balance(train_groups),
            "held_balance": group_balance(held_groups),
        },
        "tokenization_validation": tv,
        "baseline_train": baseline_train,
        "baseline_held": baseline_held,
        "arms": [
            {
                "arm_name": a["arm_name"],
                "bg_mask_prob": a["bg_mask_prob"],
                "use_bg_loss": a["use_bg_loss"],
                "cumulative_stats": a["cumulative_stats"],
                "trajectory": a["trajectory"],
                "final_train": compact_eval(a["final_train"]),
                "final_held": compact_eval(a["final_held"]),
            }
            for a in arms
        ],
        "update_probe": update_probe,
        "interpretation": "\n".join(interp_lines),
    }
    (out_dir / "paired_factorial_update_probe_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    md = ["# research paired factorial and update probe\n\n"]
    md.append("## Why this run exists\n\n")
    md.append("research used different arm-dependent corruption streams. This run pairs the corrupted arms by deterministic epoch/batch seeds and resets dropout seeds before each forward pass. It also removes duplicated groups and cycles query side independently from source order.\n\n")
    md.append("## Endpoint results\n\n")
    md.append("| Arm | Held flip | Held U | Held R | Held N | Train flip | bg corrupted positions | answer labels |\n")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for a in arms:
        he = compact_eval(a["final_held"])
        tr = compact_eval(a["final_train"])
        cs = a["cumulative_stats"]
        md.append(f"| {a['arm_name']} | {he['n_recipient_flip_correct']}/{he['n_groups']} | "
                  f"{he['mean_update_new_minus_source']:+.3f} | {he['mean_retain_source_minus_new']:+.3f} | "
                  f"{he['mean_neutral_source_minus_new']:+.3f} | {tr['n_recipient_flip_correct']}/{tr['n_groups']} | "
                  f"{cs.get('n_bg_corrupted_positions', 0)} | {cs.get('n_answer_forced', 0)} |\n")
    md.append("\n## Update probe\n\n")
    md.append(f"- Answer/bg gradient cosine at answer-only snapshot: {gs['answer_bg_cosine']:+.4f}\n")
    md.append(f"- Norms: answer {gs['answer_grad_norm']:.6g}, bg {gs['bg_grad_norm']:.6g}, combined {gs['combined_grad_norm']:.6g}\n")
    md.append(f"- Clip coefficients if max norm {args.max_grad_norm}: answer {gs['answer_clip_coef_if_alone']:.4g}, bg {gs['bg_clip_coef_if_alone']:.4g}, combined {gs['combined_clip_coef']:.4g}\n\n")
    md.append("One-step fixed-batch consequences:\n\n")
    md.append("| Update | Δ answer loss | Δ bg loss | Δ held flip | Δ held U | Δ held R |\n")
    md.append("|---|---:|---:|---:|---:|---:|\n")
    for r in update_probe["one_step_updates"]:
        md.append(f"| {r['kind']} | {r['delta_answer_loss_eval_mode']:+.6f} | {r['delta_bg_loss_eval_mode']:+.6f} | "
                  f"{r['delta_held_flip']} | {r['delta_held_U']:+.6f} | {r['delta_held_R']:+.6f} |\n")
    md.append("\n## Interpretation\n\n")
    md.append(summary["interpretation"] + "\n")
    (out_dir / "paired_factorial_update_probe_summary.md").write_text("".join(md), encoding="utf-8")

    # Remove large live objects before printing/exit.
    for key in ["_model", "_opt", "_trainable"]:
        answer_arm.pop(key, None)
    torch.cuda.empty_cache()

    print("\n" + "=" * 80, flush=True)
    print(json.dumps({
        "status": summary["status"],
        "out_dir": str(out_dir),
        "baseline_held_flip": f"{baseline_held['n_recipient_flip_correct']}/{baseline_held['n_groups']}",
        "paired_corrupted_answer_only_held_flip": f"{ans_he['n_recipient_flip_correct']}/{ans_he['n_groups']}",
        "paired_corrupted_answer_plus_bg_held_flip": f"{cmb_he['n_recipient_flip_correct']}/{cmb_he['n_groups']}",
        "answer_bg_grad_cosine": gs["answer_bg_cosine"],
        "interpretation": summary["interpretation"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
