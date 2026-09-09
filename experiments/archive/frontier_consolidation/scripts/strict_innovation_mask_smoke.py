#!/usr/bin/env python3
"""research CPU smoke for strict content-innovation WWM reallocation.

The smoke verifies the repaired intervention before any H100 run:
  - strict content innovations are selected at elevated p_strict;
  - copyable rewrite groups are reduced just enough to fund that extra target mass;
  - source, duplicate/function-like rewrite-other, and filler/ordinary groups remain
    at the ordinary WWM rate, not raised as in the obsolete research map;
  - aggregate selected-token mass stays close to standard WWM.

CPU-only. No model update, no GPU, no official evaluation.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import importlib.util
import json
import pathlib
import statistics
import sys
import time
from collections import defaultdict
from typing import Any

import torch
from torch.utils.data import DataLoader


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
WORKSPACE = USER_ROOT / "experiments/archive/frontier_consolidation"
TRAINER_PATH = WORKSPACE / "scripts/strict_innovation_trainer.py"
STRICT_MAP_PATH = WORKSPACE / "data/strict_innovation_group_map/strict_innovation_group_map.json"
TRAIN_100M = WORKSPACE / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl"
TOKENIZER_DIR = WORKSPACE / "data/compliant_tokenizer"
OUT_DIR = WORKSPACE / "data/strict_innovation_mask_smoke"

N_CHANGED_ROWS = 768
N_UNCHANGED_ROWS = 2048
BATCH_SIZE = 64
MASK_PROB = 0.15
P_STRICT = 0.50
P_COPY_BASIS = "tokens"


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def collect_examples(base, changed_ids: set[int]) -> list[Any]:
    changed = []
    unchanged = []
    with TRAIN_100M.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            eid = int(obj.get("example_id", i))
            ex = base.Example(text=str(obj["text"]), words=int(obj.get("words", len(str(obj["text"]).split()))), example_id=eid)
            if eid in changed_ids and len(changed) < N_CHANGED_ROWS:
                changed.append(ex)
            elif eid not in changed_ids and len(unchanged) < N_UNCHANGED_ROWS:
                unchanged.append(ex)
            if len(changed) >= N_CHANGED_ROWS and len(unchanged) >= N_UNCHANGED_ROWS:
                break
    return changed + unchanged


def standard_wwm_labels(input_ids, attention_mask, word_group, tokenizer, gen, mask_prob: float) -> torch.Tensor:
    special_ids = torch.tensor(sorted(tokenizer.all_special_ids))
    candidate = attention_mask.bool() & ~torch.isin(input_ids, special_ids)
    select = torch.zeros_like(candidate)
    for b in range(input_ids.shape[0]):
        groups = word_group[b]
        valid = torch.unique(groups[groups >= 0])
        if valid.numel() == 0:
            continue
        gp = torch.rand(valid.numel(), generator=gen)
        chosen = valid[gp < mask_prob]
        if chosen.numel() > 0:
            select[b] = torch.isin(groups, chosen) & candidate[b]
    labels = input_ids.clone()
    labels[~select] = -100
    return labels


def rate(sel: int, avail: int) -> float | None:
    return None if avail <= 0 else sel / avail


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer = load_module("strict_trainer_for_smoke", TRAINER_PATH)
    base = trainer.base
    strict_map, totals = trainer.load_strict_map(STRICT_MAP_PATH)
    p_copy = trainer.compute_auto_p_copy(MASK_PROB, P_STRICT, totals, P_COPY_BASIS)
    tokenizer = base.make_portable_tokenizer(str(TOKENIZER_DIR))
    examples = collect_examples(base, set(strict_map))
    dataset = trainer.StrictInnovationMaskedChunkDataset(examples, tokenizer, 256)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=trainer.strict_innovation_collate, num_workers=0)

    std_stats = defaultdict(int)
    bias_stats = defaultdict(int)
    batch_rates = []
    for batch_idx, batch in enumerate(loader):
        words = batch.pop("words")
        example_ids = batch.pop("example_ids")
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        word_group = batch["word_group"]
        gen_std = torch.Generator(device="cpu"); gen_std.manual_seed(43023 + batch_idx * 1009)
        gen_bias = torch.Generator(device="cpu"); gen_bias.manual_seed(43023 + batch_idx * 1009)
        std_labels = standard_wwm_labels(input_ids, attention_mask, word_group, tokenizer, gen_std, MASK_PROB)
        _, bias_labels, ms = trainer.apply_strict_innovation_masking(
            input_ids, attention_mask, word_group, example_ids, tokenizer,
            MASK_PROB, P_STRICT, p_copy, strict_map, gen_bias,
        )
        std_selected = int((std_labels != -100).sum().item())
        bias_selected = int((bias_labels != -100).sum().item())
        std_stats["selected_tokens"] += std_selected
        std_stats["candidate_tokens"] += int(attention_mask.bool().sum().item())
        bias_stats["selected_tokens"] += bias_selected
        bias_stats["candidate_tokens"] += int(attention_mask.bool().sum().item())
        for k, v in ms.items():
            bias_stats[k] += int(v)
        batch_rates.append(bias_selected / max(1, std_selected))

    rates = {
        "strict_group_rate": rate(bias_stats["strict_selected_groups"], bias_stats["strict_available_groups"]),
        "copy_group_rate": rate(bias_stats["copy_selected_groups"], bias_stats["copy_available_groups"]),
        "ordinary_group_rate": rate(bias_stats["ordinary_selected_groups"], bias_stats["ordinary_available_groups"]),
        "source_group_rate": rate(bias_stats["source_selected_groups"], bias_stats["source_available_groups"]),
        "rewrite_other_group_rate": rate(bias_stats["rewrite_other_selected_groups"], bias_stats["rewrite_other_available_groups"]),
        "filler_group_rate": rate(bias_stats["filler_selected_groups"], bias_stats["filler_available_groups"]),
        "strict_token_rate": rate(bias_stats["strict_selected_tokens"], bias_stats["strict_available_tokens"]),
        "copy_token_rate": rate(bias_stats["copy_selected_tokens"], bias_stats["copy_available_tokens"]),
        "ordinary_token_rate": rate(bias_stats["ordinary_selected_tokens"], bias_stats["ordinary_available_tokens"]),
        "source_token_rate": rate(bias_stats["source_selected_tokens"], bias_stats["source_available_tokens"]),
        "rewrite_other_token_rate": rate(bias_stats["rewrite_other_selected_tokens"], bias_stats["rewrite_other_available_tokens"]),
        "filler_token_rate": rate(bias_stats["filler_selected_tokens"], bias_stats["filler_available_tokens"]),
    }
    mass_ratio = bias_stats["selected_tokens"] / max(1, std_stats["selected_tokens"])
    batch_ratio_summary = {
        "n": len(batch_rates),
        "mean": float(statistics.mean(batch_rates)) if batch_rates else None,
        "median": float(statistics.median(batch_rates)) if batch_rates else None,
        "min": float(min(batch_rates)) if batch_rates else None,
        "max": float(max(batch_rates)) if batch_rates else None,
    }
    checks = {
        "strict_rate_near_p_strict": abs((rates["strict_group_rate"] or 0) - P_STRICT) < 0.05,
        "copy_rate_near_p_copy": abs((rates["copy_group_rate"] or 0) - p_copy) < 0.035,
        "source_group_rate_near_baseline": abs((rates["source_group_rate"] or 0) - MASK_PROB) < 0.025,
        "rewrite_other_group_rate_near_baseline": abs((rates["rewrite_other_group_rate"] or 0) - MASK_PROB) < 0.05,
        "ordinary_group_rate_near_baseline": abs((rates["ordinary_group_rate"] or 0) - MASK_PROB) < 0.025,
        "mass_ratio_close_to_standard": 0.95 <= mass_ratio <= 1.05,
    }
    summary = {
        "status": "STRICT_INNOVATION_MASK_SMOKE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "changed_examples": N_CHANGED_ROWS,
            "unchanged_examples": N_UNCHANGED_ROWS,
            "batch_size": BATCH_SIZE,
            "mask_prob": MASK_PROB,
            "p_strict": P_STRICT,
            "p_copy_effective": p_copy,
            "p_copy_basis": P_COPY_BASIS,
        },
        "map_totals": {k: int(v) for k, v in totals.items()},
        "standard": dict(std_stats),
        "strict_biased": dict(bias_stats),
        "rates": rates,
        "mass_ratio_vs_standard_selected_tokens": mass_ratio,
        "batch_mass_ratio_summary": batch_ratio_summary,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "elapsed_sec": round(time.time() - t0, 3),
    }
    (OUT_DIR / "strict_innovation_mask_smoke.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# research — strict content-innovation mask smoke",
        "",
        "CPU-only; no model update, no GPU, no official evaluation.",
        "",
        f"- changed rows sampled: `{N_CHANGED_ROWS}`; unchanged rows sampled: `{N_UNCHANGED_ROWS}`",
        f"- p_strict: `{P_STRICT}`; p_copy_effective ({P_COPY_BASIS}-matched): `{p_copy:.8f}`; ordinary p: `{MASK_PROB}`",
        f"- selected-token mass ratio vs standard WWM: `{mass_ratio:.6f}`",
        f"- all checks passed: `{all(checks.values())}`",
        "",
        "## Empirical group rates",
    ]
    for k in ["strict_group_rate", "copy_group_rate", "source_group_rate", "rewrite_other_group_rate", "ordinary_group_rate", "filler_group_rate"]:
        lines.append(f"- {k}: `{rates[k]}`")
    lines += ["", "## Empirical token rates"]
    for k in ["strict_token_rate", "copy_token_rate", "source_token_rate", "rewrite_other_token_rate", "ordinary_token_rate", "filler_token_rate"]:
        lines.append(f"- {k}: `{rates[k]}`")
    lines += ["", "## Checks"]
    for k, v in checks.items():
        lines.append(f"- {k}: `{v}`")
    lines += ["", f"Full JSON: `{OUT_DIR / 'strict_innovation_mask_smoke.json'}`"]
    (OUT_DIR / "strict_innovation_mask_smoke.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": summary["status"],
        "p_copy_effective": p_copy,
        "mass_ratio": mass_ratio,
        "rates": {k: (None if v is None else round(v, 6)) for k, v in rates.items()},
        "checks": checks,
        "all_checks_passed": summary["all_checks_passed"],
        "elapsed_sec": summary["elapsed_sec"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
