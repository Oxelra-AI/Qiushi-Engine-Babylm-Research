#!/usr/bin/env python3
"""research frozen-anchor fast-path replay trainer.

This trainer repairs the ambiguity in the description of the earlier objective: the old research `neutral_only`
mode does *not* perform main-stream MLM acquisition; it only applies a neutrality
KL.  The present script performs true private-only main-MLM replay from the
verified scale1.75 chck_82M slow function.

Scientific contrast:
  coherent_replay    - normal legal suffix text, WWM targets from the suffix.
  spanbreak_replay   - same legal suffix rows and same WWM target-word groups, but
                       row-internal coherent spans are permuted.  Within-span token
                       order is preserved so local fluency and target/mask
                       multisets remain close to coherent replay, while cross-span
                       and cross-sentence dependencies are broken.

Only `.private_adapter.*` parameters are trainable.  Disabling the private adapters
recovers the frozen chck_82M slow function exactly by construction; the relevant
question is whether private-ON retains anchor-correct decisions while adding
reproducible new ones.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, DebertaV2Config

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/frontier_consolidation')
WORKSPACE = _public_path('experiments/archive/frontier_consolidation')
SCRIPTS = _public_path('experiments/archive/frontier_consolidation/scripts')
sys.path.insert(0, str(SCRIPTS))
from frozen82_private_modeling import FrozenSlowPrivateDebertaV2ForMaskedLM  # noqa: E402

DEFAULT_ENDPOINT = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M')
DEFAULT_STREAM = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl')

DEFAULTS = dict(
    initial_consumed_words=82_012_495,
    full_cap_words=100_000_000,
    # Match the truthful shuffled86 tail's charged suffix as closely as row-level
    # loading permits.  Main-only replay reaches 3,992,800 words (118 below this cap).
    max_tail_charged_words=3_992_918,
    skip_rows=530_944,
    batch_size=256,
    seq_length=256,
    learning_rate=0.001,
    warmup_fraction=0.06,
    lr_total_steps=455,
    weight_decay=0.01,
    mask_prob=0.15,
    checkpoint_words=1_000_000,
    log_every=25,
    private_adapter_bottleneck=128,
    private_adapter_scale=1.0,
    main_lambda=1.0,
    neutral_lambda=1.0,
    neutral_subsample=32,
    train_rng_seed=43023,
    span_shuffle_seed=43151,
    min_span_groups=5,
    max_span_groups=18,
    fallback_span_groups=12,
    max_updates=0,
)


@dataclass
class Ex:
    text: str
    words: int
    source: str
    example_id: int
    row_index: int


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def reset_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_word_start(token_str: str) -> bool:
    return token_str.startswith("Ġ") or token_str.startswith("▁")


def load_examples_tail(path: Path, skip_rows: int, max_main_words: int) -> list[Ex]:
    examples: list[Ex] = []
    selected = 0
    with path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < skip_rows:
                continue
            if not line.strip():
                continue
            obj = json.loads(line)
            text = str(obj["text"])
            words = int(obj.get("words", len(text.split())))
            if words != len(text.split()):
                raise RuntimeError(f"word mismatch row={idx} example_id={obj.get('example_id')}")
            if selected + words > max_main_words:
                break
            examples.append(Ex(text=text, words=words, source=str(obj.get("source", "")),
                               example_id=int(obj.get("example_id", -1)), row_index=idx))
            selected += words
    return examples


class TailDataset(Dataset):
    def __init__(self, examples: list[Ex], tokenizer, seq_length: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.special_ids = set(tokenizer.all_special_ids)
        self._word_start_cache: dict[int, bool] = {}

    def _word_start(self, tid: int) -> bool:
        v = self._word_start_cache.get(tid)
        if v is None:
            s = self.tokenizer.convert_ids_to_tokens(int(tid))
            v = bool(s is not None and is_word_start(str(s)))
            self._word_start_cache[tid] = v
        return v

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
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
        word_group = torch.full_like(input_ids, -1)
        gid = -1
        for i in range(input_ids.shape[0]):
            if attention_mask[i] == 0:
                continue
            tid = int(input_ids[i])
            if tid in self.special_ids:
                continue
            if gid < 0 or self._word_start(tid) or i == 0:
                gid += 1
            word_group[i] = gid
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "word_group": word_group,
            "words": ex.words,
            "example_id": ex.example_id,
            "row_index": ex.row_index,
            "source": ex.source,
            "text": ex.text,
        }


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "word_group": torch.stack([x["word_group"] for x in batch]),
        "words": torch.tensor([x["words"] for x in batch], dtype=torch.long),
        "example_id": torch.tensor([x["example_id"] for x in batch], dtype=torch.long),
        "row_index": torch.tensor([x["row_index"] for x in batch], dtype=torch.long),
        "source": [x["source"] for x in batch],
        "text": [x["text"] for x in batch],
    }


def apply_wwm(input_ids, attention_mask, word_group, tokenizer, mask_prob: float, gen: torch.Generator):
    device = input_ids.device
    B, S = input_ids.shape
    labels = input_ids.clone()
    mask_token_id = int(tokenizer.mask_token_id)
    special = torch.tensor(sorted(tokenizer.all_special_ids), device=device)
    candidate = attention_mask.bool() & ~torch.isin(input_ids, special)
    select = torch.zeros_like(candidate)
    selected_groups_by_row: list[list[int]] = []
    for b in range(B):
        groups = word_group[b]
        valid = torch.unique(groups[groups >= 0])
        row_selected: list[int] = []
        if valid.numel() > 0:
            r = torch.rand(valid.numel(), generator=gen, device=device)
            chosen = valid[r < mask_prob]
            if chosen.numel() > 0:
                select[b] = torch.isin(groups, chosen) & candidate[b]
                row_selected = [int(x) for x in chosen.detach().cpu().tolist()]
        selected_groups_by_row.append(row_selected)
    if select.sum() == 0:
        flat = candidate.view(-1).nonzero(as_tuple=False)
        if flat.numel() > 0:
            select.view(-1)[flat[0, 0]] = True
    labels[~select] = -100
    masked = input_ids.clone()
    r = torch.rand(B, S, generator=gen, device=device)
    masked[select & (r < 0.8)] = mask_token_id
    rand_tok = select & (r >= 0.8) & (r < 0.9)
    if rand_tok.any():
        masked[rand_tok] = torch.randint(0, len(tokenizer), (int(rand_tok.sum()),),
                                         generator=gen, device=device)
    return masked, labels, select, selected_groups_by_row


class TokenBoundaryCache:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.tok_cache: dict[int, str] = {}

    def tok(self, tid: int) -> str:
        v = self.tok_cache.get(int(tid))
        if v is None:
            v = str(self.tokenizer.convert_ids_to_tokens(int(tid)))
            self.tok_cache[int(tid)] = v
        return v

    def group_text(self, tids: list[int]) -> str:
        # Decode only short groups; this is used for punctuation boundaries, not training labels.
        try:
            return self.tokenizer.decode([int(x) for x in tids], clean_up_tokenization_spaces=False)
        except Exception:
            return "".join(self.tok(x) for x in tids)


def group_positions(input_ids_1d: torch.Tensor, attention_1d: torch.Tensor, word_group_1d: torch.Tensor) -> tuple[int, dict[int, list[int]], list[int]]:
    L = int(attention_1d.sum().item())
    pos_by_group: dict[int, list[int]] = {}
    for i in range(L):
        g = int(word_group_1d[i].item())
        if g >= 0:
            pos_by_group.setdefault(g, []).append(i)
    groups = sorted(pos_by_group)
    return L, pos_by_group, groups


def make_spans_for_row(input_ids_1d: torch.Tensor, attention_1d: torch.Tensor, word_group_1d: torch.Tensor,
                       tokenizer, cache: TokenBoundaryCache, args) -> list[list[int]]:
    L, pos_by_group, groups = group_positions(input_ids_1d, attention_1d, word_group_1d)
    if len(groups) <= 1:
        return [list(range(L))]
    spans: list[list[int]] = []
    cur_groups: list[int] = []
    for g in groups:
        cur_groups.append(g)
        tids = [int(input_ids_1d[p].item()) for p in pos_by_group[g]]
        text = cache.group_text(tids)
        boundary = any(ch in text for ch in [".", "?", "!", ";", ":"])
        if (boundary and len(cur_groups) >= int(args.min_span_groups)) or len(cur_groups) >= int(args.max_span_groups):
            positions: list[int] = []
            for gg in cur_groups:
                positions.extend(pos_by_group[gg])
            spans.append(positions)
            cur_groups = []
    if cur_groups:
        positions = []
        for gg in cur_groups:
            positions.extend(pos_by_group[gg])
        spans.append(positions)
    if len(spans) < 2 and len(groups) > int(args.fallback_span_groups):
        spans = []
        chunk = max(2, int(args.fallback_span_groups))
        for s in range(0, len(groups), chunk):
            positions = []
            for gg in groups[s:s + chunk]:
                positions.extend(pos_by_group[gg])
            spans.append(positions)
    return spans if spans else [list(range(L))]


def permute_spans(spans: list[list[int]], rng: random.Random) -> tuple[list[list[int]], bool]:
    if len(spans) <= 1:
        return spans, False
    order = list(range(len(spans)))
    rng.shuffle(order)
    if order == list(range(len(spans))):
        order = order[1:] + order[:1]
    return [spans[i] for i in order], True


def spanbreak_batch(input_ids: torch.Tensor, attention_mask: torch.Tensor, word_group: torch.Tensor,
                    tokenizer, args, loader_step: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    # CPU tensors in, CPU tensors out.  Word-group IDs travel with their original token groups so
    # the same selected group IDs yield the same target-token multiset as coherent replay.
    new_ids = input_ids.clone()
    new_wg = word_group.clone()
    cache = TokenBoundaryCache(tokenizer)
    rows_changed = 0
    total_rows = int(input_ids.shape[0])
    total_spans = 0
    total_groups = 0
    identity_rows = 0
    examples: list[dict[str, Any]] = []
    for b in range(total_rows):
        L, _pos_by_group, groups = group_positions(input_ids[b], attention_mask[b], word_group[b])
        total_groups += len(groups)
        spans = make_spans_for_row(input_ids[b], attention_mask[b], word_group[b], tokenizer, cache, args)
        total_spans += len(spans)
        rng = random.Random(int(args.span_shuffle_seed) + 1_000_003 * int(loader_step) + 9_176 * b)
        permuted, changed = permute_spans(spans, rng)
        if not changed:
            identity_rows += 1
            continue
        new_order = [p for span in permuted for p in span]
        if len(new_order) != L or sorted(new_order) != list(range(L)):
            raise RuntimeError(f"bad span permutation row={b} L={L} len={len(new_order)}")
        old_ids = input_ids[b, :L].clone()
        old_wg = word_group[b, :L].clone()
        new_ids[b, :L] = input_ids[b, new_order]
        new_wg[b, :L] = word_group[b, new_order]
        if not torch.equal(new_ids[b, :L].sort().values, old_ids.sort().values):
            raise RuntimeError(f"token multiset changed in row {b}")
        if not torch.equal(new_wg[b, :L].sort().values, old_wg.sort().values):
            raise RuntimeError(f"word-group multiset changed in row {b}")
        rows_changed += 1
        if len(examples) < 3:
            before = tokenizer.decode([int(x) for x in old_ids[:min(L, 96)].tolist()], clean_up_tokenization_spaces=False)
            after = tokenizer.decode([int(x) for x in new_ids[b, :min(L, 96)].tolist()], clean_up_tokenization_spaces=False)
            examples.append({"row_in_batch": b, "groups": len(groups), "spans": len(spans), "before_prefix": before, "after_prefix": after})
    stats = {
        "rows": total_rows,
        "rows_changed": rows_changed,
        "identity_rows": identity_rows,
        "mean_spans_per_row": total_spans / max(1, total_rows),
        "mean_groups_per_row": total_groups / max(1, total_rows),
        "examples": examples,
    }
    return new_ids, new_wg, stats


def maybe_transform_batch(input_ids_cpu: torch.Tensor, attention_cpu: torch.Tensor, word_group_cpu: torch.Tensor,
                          tokenizer, args, loader_step: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if args.replay_mode == "coherent_replay":
        return input_ids_cpu, word_group_cpu, {"rows": int(input_ids_cpu.shape[0]), "rows_changed": 0, "mean_spans_per_row": None}
    if args.replay_mode == "spanbreak_replay":
        return spanbreak_batch(input_ids_cpu, attention_cpu, word_group_cpu, tokenizer, args, loader_step)
    raise ValueError(args.replay_mode)


def lr_at_update(update_index0: int, schedule_total: int, warmup: int, peak_lr: float) -> float:
    if update_index0 < warmup:
        return peak_lr * update_index0 / max(1, warmup)
    p = (update_index0 - warmup) / max(1, schedule_total - warmup)
    p = min(max(p, 0.0), 1.0)
    return peak_lr * 0.5 * (1.0 + math.cos(math.pi * p))


def load_frozen_private_model(endpoint: Path, args):
    cfg = DebertaV2Config.from_pretrained(str(endpoint), local_files_only=True)
    cfg.private_adapter_bottleneck = int(args.private_adapter_bottleneck)
    cfg.private_adapter_scale = float(args.private_adapter_scale)
    cfg.private_adapter_enabled = True
    model = FrozenSlowPrivateDebertaV2ForMaskedLM(cfg)
    sd = load_file(str(endpoint / "model.safetensors"), device="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    tied_missing = {"cls.predictions.decoder.weight", "cls.predictions.decoder.bias"}
    bad_missing = [x for x in missing if "private_adapter" not in x and x not in tied_missing]
    if bad_missing or unexpected:
        raise RuntimeError(f"unexpected load mismatch: bad_missing={bad_missing[:10]} unexpected={unexpected[:10]}")
    model.tie_weights()
    model.register_for_auto_class("AutoModelForMaskedLM")
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model, missing, unexpected


def set_private_enabled(model, enabled: bool) -> None:
    model.set_private_enabled(enabled)


def save_checkpoint(model, tokenizer, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(dst), safe_serialization=True)
    tokenizer.save_pretrained(str(dst))
    src = _public_path('experiments/archive/frontier_consolidation/scripts/frozen82_private_modeling.py')
    d = dst / src.name
    if src.exists() and not d.exists():
        shutil.copy2(str(src), str(d))


def compute_neutral_loss(model, input_ids, attention_mask, neutral_subsample: int):
    n = min(int(neutral_subsample), input_ids.shape[0])
    if n <= 0:
        return None, 0.0
    was_training = model.training
    model.eval()
    ids = input_ids[:n]
    att = attention_mask[:n]
    set_private_enabled(model, False)
    with torch.no_grad():
        slow_logits = model(input_ids=ids, attention_mask=att).logits.detach()
    set_private_enabled(model, True)
    out = model(input_ids=ids, attention_mask=att)
    log_p_private = F.log_softmax(out.logits, dim=-1)
    p_slow = F.softmax(slow_logits, dim=-1)
    kl_tok = F.kl_div(log_p_private, p_slow, reduction="none").sum(-1)
    mask_float = att.float()
    loss = (kl_tok * mask_float).sum() / max(1.0, float(mask_float.sum()))
    if was_training:
        model.train()
    return loss, float(loss.detach().cpu())


def update_disruption_totals(totals: dict[str, Any], stats: dict[str, Any]) -> None:
    totals["batches"] += 1
    totals["rows"] += int(stats.get("rows") or 0)
    totals["rows_changed"] += int(stats.get("rows_changed") or 0)
    totals["identity_rows"] += int(stats.get("identity_rows") or 0)
    msp = stats.get("mean_spans_per_row")
    if msp is not None:
        totals["sum_mean_spans_per_row"] += float(msp)
    if not totals["examples"] and stats.get("examples"):
        totals["examples"] = stats["examples"]


def summarize_disruption(totals: dict[str, Any]) -> dict[str, Any]:
    rows = max(1, int(totals.get("rows") or 0))
    batches = max(1, int(totals.get("batches") or 0))
    return {
        "batches": totals.get("batches", 0),
        "rows": totals.get("rows", 0),
        "rows_changed": totals.get("rows_changed", 0),
        "row_changed_fraction": float(totals.get("rows_changed", 0)) / rows,
        "identity_rows": totals.get("identity_rows", 0),
        "mean_of_batch_mean_spans_per_row": float(totals.get("sum_mean_spans_per_row", 0.0)) / batches,
        "examples": totals.get("examples", []),
    }


def train(args) -> None:
    t0 = time.time()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    hf_cache = out / "hf_cache"
    (hf_cache / "modules").mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(hf_cache)
    os.environ["HF_MODULES_CACHE"] = str(hf_cache / "modules")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    endpoint = Path(args.endpoint)
    stream = Path(args.example_jsonl)

    print(json.dumps({"event": "start", "trainer": "FROZEN82_FASTPATH_REPLAY",
                      "device": str(device), "replay_mode": args.replay_mode,
                      "initial_consumed_words": args.initial_consumed_words,
                      "max_tail_charged_words": args.max_tail_charged_words,
                      "skip_rows": args.skip_rows,
                      "decision": "tests whether private-only coherent suffix replay adds reproducible private-ON competence beyond a short-range-plausible spanbreak control"}), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(str(endpoint), local_files_only=True)
    examples = load_examples_tail(stream, int(args.skip_rows), int(args.max_tail_charged_words))
    main_words_available = sum(e.words for e in examples)
    print(json.dumps({"event": "data_loaded", "tail_examples": len(examples),
                      "main_words_available": main_words_available,
                      "first_row_index": examples[0].row_index if examples else None,
                      "last_row_index": examples[-1].row_index if examples else None}), flush=True)

    dataset = TailDataset(examples, tokenizer, args.seq_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate, num_workers=0,
                        pin_memory=torch.cuda.is_available())

    reset_all(args.train_rng_seed)
    model, missing, unexpected = load_frozen_private_model(endpoint, args)
    model.to(device)
    if torch.cuda.is_available():
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()

    private_names = {n for n, _ in model.named_parameters() if ".private_adapter." in n}
    for n, p in model.named_parameters():
        p.requires_grad_(n in private_names)
    private_up_ids = {id(p) for n, p in model.named_parameters() if ".private_adapter.up." in n}
    private_normal = [p for n, p in model.named_parameters() if n in private_names and id(p) not in private_up_ids]
    private_zero_wd = [p for n, p in model.named_parameters() if n in private_names and id(p) in private_up_ids]
    optimizer = torch.optim.AdamW([
        {"params": private_normal, "weight_decay": args.weight_decay},
        {"params": private_zero_wd, "weight_decay": 0.0},
    ], lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-6)

    total_params = sum(p.numel() for p in model.parameters())
    private_params = sum(p.numel() for n, p in model.named_parameters() if n in private_names)
    print(json.dumps({"event": "model_loaded", "endpoint": rel(endpoint),
                      "total_params": total_params, "private_params": private_params,
                      "frozen_slow_params": total_params - private_params,
                      "missing_keys_count": len(missing), "unexpected_keys_count": len(unexpected)}), flush=True)

    schedule_total = int(args.lr_total_steps) if args.lr_total_steps > 0 else max(1, len(loader))
    warmup = max(1, int(schedule_total * args.warmup_fraction))
    gen = torch.Generator(device=device)
    gen.manual_seed(args.train_rng_seed)

    config = {
        "status": "FROZEN82_FASTPATH_REPLAY_CONFIG",
        "endpoint": rel(endpoint),
        "example_jsonl": rel(stream),
        "replay_mode": args.replay_mode,
        "initial_consumed_words": args.initial_consumed_words,
        "full_cap_words": args.full_cap_words,
        "max_tail_charged_words": args.max_tail_charged_words,
        "skip_rows": args.skip_rows,
        "tail_start_row_index_0based": args.skip_rows,
        "trainable": "private_adapter_only",
        "slow_path": "frozen_verified_scale1p75_chck_82M",
        "objective": "main_mlm_ce_plus_deterministic_private_on_vs_private_off_kl",
        "schedule_total": schedule_total,
        "warmup_steps": warmup,
        "total_params": total_params,
        "private_params": private_params,
        **{k: getattr(args, k) for k in DEFAULTS.keys()},
    }
    (out / "train_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    losses_main: list[float] = []
    losses_neutral: list[float] = []
    cum_main = 0
    updates = 0
    next_ckpt_tail = args.checkpoint_words if args.checkpoint_words > 0 else None
    stopped_before_cap = False
    disruption_totals = {"batches": 0, "rows": 0, "rows_changed": 0, "identity_rows": 0,
                         "sum_mean_spans_per_row": 0.0, "examples": []}
    log_path = out / "training_log.jsonl"

    model.train()
    with log_path.open("w", encoding="utf-8") as logf:
        for loader_step, batch in enumerate(loader, 1):
            if args.max_updates and updates >= args.max_updates:
                print(json.dumps({"event": "stop_after_max_updates", "updates": updates}), flush=True)
                break
            words = int(batch["words"].sum().item())
            tail_next = cum_main + words
            total_next = int(args.initial_consumed_words) + tail_next
            if tail_next > args.max_tail_charged_words or total_next > args.full_cap_words:
                stopped_before_cap = True
                print(json.dumps({"event": "stop_before_over_cap", "loader_step": loader_step,
                                  "tail_current_charged": cum_main,
                                  "next_main_words": words,
                                  "initial_consumed_words": args.initial_consumed_words,
                                  "total_next": total_next,
                                  "full_cap_words": args.full_cap_words}), flush=True)
                break

            replay_ids_cpu, replay_wg_cpu, disruption_stats = maybe_transform_batch(
                batch["input_ids"], batch["attention_mask"], batch["word_group"], tokenizer, args, loader_step)
            update_disruption_totals(disruption_totals, disruption_stats)

            input_ids = replay_ids_cpu.to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            word_group = replay_wg_cpu.to(device, non_blocking=True)

            masked_inputs, labels, _select, _selected_groups = apply_wwm(
                input_ids, attention_mask, word_group, tokenizer, args.mask_prob, gen)
            n_main_targets = int((labels != -100).sum().item())

            lr = lr_at_update(updates, schedule_total, warmup, args.learning_rate)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            optimizer.zero_grad(set_to_none=True)

            model.train()
            set_private_enabled(model, True)
            out_main = model(input_ids=masked_inputs, attention_mask=attention_mask)
            vocab = out_main.logits.shape[-1]
            main_loss_sum = F.cross_entropy(
                out_main.logits.reshape(-1, vocab), labels.reshape(-1),
                ignore_index=-100, reduction="sum")
            main_loss = main_loss_sum / max(1, n_main_targets)
            main_loss_value = float(main_loss.detach().cpu())
            (args.main_lambda * main_loss).backward()
            del out_main, main_loss_sum, main_loss
            losses_main.append(main_loss_value)

            neutral_loss_value = 0.0
            if args.neutral_lambda > 0:
                neutral_loss, neutral_loss_value = compute_neutral_loss(
                    model, masked_inputs, attention_mask, args.neutral_subsample)
                if neutral_loss is not None:
                    (args.neutral_lambda * neutral_loss).backward()
                    del neutral_loss
                losses_neutral.append(neutral_loss_value)

            private_params_list = private_normal + private_zero_wd
            torch.nn.utils.clip_grad_norm_(private_params_list, 1.0)
            optimizer.step()

            updates += 1
            cum_main += words
            tail_charged = cum_main
            total_consumed = int(args.initial_consumed_words) + tail_charged
            rec = {
                "update": updates,
                "loader_step": loader_step,
                "source_row_start": int(batch["row_index"][0].item()),
                "source_row_end": int(batch["row_index"][-1].item()),
                "lr": lr,
                "batch_words": words,
                "tail_main_words": cum_main,
                "tail_aux_words": 0,
                "tail_charged_words": tail_charged,
                "total_consumed_words": total_consumed,
                "main_targets": n_main_targets,
                "main_loss": main_loss_value,
                "neutral_loss": neutral_loss_value,
                "disruption_rows_changed": disruption_stats.get("rows_changed"),
                "disruption_mean_spans_per_row": disruption_stats.get("mean_spans_per_row"),
                "private_rms_max": max(model.private_adapter_rms()) if hasattr(model, "private_adapter_rms") else None,
                "elapsed_sec": round(time.time() - t0, 1),
            }
            logf.write(json.dumps(rec) + "\n")
            if updates == 1 or updates % args.log_every == 0:
                print(json.dumps({"event": "train", **rec}), flush=True)

            while next_ckpt_tail is not None and tail_charged >= next_ckpt_tail and next_ckpt_tail <= args.max_tail_charged_words:
                approx_total = int(args.initial_consumed_words) + next_ckpt_tail
                name = f"chck_{approx_total // 1_000_000}M" if approx_total % 1_000_000 == 0 else f"chck_total_{approx_total}w"
                save_checkpoint(model, tokenizer, out / "hf_model" / name)
                print(json.dumps({"event": "checkpoint", "name": name,
                                  "tail_charged_words": tail_charged,
                                  "total_consumed_words": total_consumed,
                                  "update": updates}), flush=True)
                next_ckpt_tail += args.checkpoint_words

            del input_ids, attention_mask, word_group, masked_inputs, labels
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    set_private_enabled(model, True)
    save_checkpoint(model, tokenizer, out / "hf_model" / "final")
    metrics = {
        "status": "FROZEN82_FASTPATH_REPLAY",
        "mode": args.replay_mode,
        "replay_mode": args.replay_mode,
        "endpoint": rel(endpoint),
        "initial_consumed_words": int(args.initial_consumed_words),
        "skip_rows": int(args.skip_rows),
        "max_tail_charged_words": int(args.max_tail_charged_words),
        "full_cap_words": int(args.full_cap_words),
        "stopped_before_cap": stopped_before_cap,
        "updates": updates,
        "schedule_total": schedule_total,
        "tail_main_word_exposure": cum_main,
        "tail_aux_word_exposure": 0,
        "tail_charged_words": cum_main,
        "total_consumed_words": int(args.initial_consumed_words) + cum_main,
        "total_params": total_params,
        "private_params": private_params,
        "frozen_slow_params": total_params - private_params,
        "first_main_loss": losses_main[0] if losses_main else None,
        "final_main_loss": losses_main[-1] if losses_main else None,
        "mean_main_loss": sum(losses_main) / len(losses_main) if losses_main else None,
        "main_loss_batches": len(losses_main),
        "first_aux_loss": None,
        "final_aux_loss": None,
        "mean_aux_loss": 0.0,
        "aux_loss_batches": 0,
        "first_neutral_loss": losses_neutral[0] if losses_neutral else None,
        "final_neutral_loss": losses_neutral[-1] if losses_neutral else None,
        "mean_neutral_loss": sum(losses_neutral) / len(losses_neutral) if losses_neutral else 0.0,
        "deterministic_neutrality_eval_mode": True,
        "trainable": "private_adapter_only",
        "disruption_summary": summarize_disruption(disruption_totals),
        "elapsed_sec": round(time.time() - t0, 2),
    }
    (out / "scientific_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": metrics["status"], "mode": args.replay_mode,
                      "updates": updates, "tail_charged_words": cum_main,
                      "total_consumed_words": metrics["total_consumed_words"],
                      "final_main_loss": metrics["final_main_loss"],
                      "final_neutral_loss": metrics["final_neutral_loss"],
                      "row_changed_fraction": metrics["disruption_summary"].get("row_changed_fraction"),
                      "out": rel(out), "elapsed_sec": metrics["elapsed_sec"]}, indent=2), flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", default=str(DEFAULT_ENDPOINT))
    p.add_argument("--example_jsonl", default=str(DEFAULT_STREAM))
    p.add_argument("--output_dir", required=True)
    p.add_argument("--replay_mode", choices=["coherent_replay", "spanbreak_replay"], required=True)
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            p.add_argument(f"--{k}", action="store_true" if not v else "store_false", default=v)
        elif isinstance(v, int):
            p.add_argument(f"--{k}", type=int, default=v)
        elif isinstance(v, float):
            p.add_argument(f"--{k}", type=float, default=v)
        else:
            p.add_argument(f"--{k}", default=v)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
