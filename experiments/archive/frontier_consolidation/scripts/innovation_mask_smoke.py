#!/usr/bin/env python3
"""research: CPU smoke test for innovation-biased WWM masking.

Validates that the innovation-biased masking mechanism:
1. Correctly identifies changed vs unchanged rows
2. Biases selection toward innovation groups and away from copyable groups
3. Maintains total selected-group mass close to standard 0.15 baseline
4. Preserves expected mass budget on changed and unchanged rows

Runs on CPU using a representative sample of training batches.
No model training, no GPU required.
"""
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
import statistics
import time
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT

USER_ROOT = find_user_root()
WORKSPACE = USER_ROOT / "experiments/archive/frontier_consolidation"

# Reuse the innovation trainer module for its dataset/masking functions
TRAINER_PATH = WORKSPACE / "scripts/innovation_biased_trainer.py"
TRAIN_100M = WORKSPACE / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl"
TOKENIZER_DIR = WORKSPACE / "data/compliant_tokenizer"
INNOVATION_MAP_PATH = WORKSPACE / "data/innovation_group_map/innovation_group_map.json"
OUT_DIR = WORKSPACE / "data/innovation_mask_smoke"

N_BATCHES = 20  # Number of batches to test
BATCH_SIZE = 64


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    trainer = load_module("trainer_for_smoke", TRAINER_PATH)
    base = trainer.base

    # Load innovation map
    innovation_map = trainer.load_innovation_map(INNOVATION_MAP_PATH)

    # Load a small slice of examples (first N_BATCHES*BATCH_SIZE*2 rows for speed)
    tokenizer = base.make_portable_tokenizer(str(TOKENIZER_DIR))
    max_rows = N_BATCHES * BATCH_SIZE * 2
    examples: list[base.Example] = []
    with TRAIN_100M.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_rows:
                break
            obj = json.loads(line)
            text = str(obj["text"])
            words = int(obj.get("words", len(text.split())))
            eid = int(obj.get("example_id", i))
            examples.append(base.Example(text=text, words=words, example_id=eid))
    actual_words = sum(ex.words for ex in examples)
    print(json.dumps({"event": "data_loaded", "examples": len(examples),
                       "words": actual_words}), flush=True)

    # Build dataset and loader
    dataset = trainer.InnovationMaskedChunkDataset(examples, tokenizer, 256)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=trainer.innovation_collate, num_workers=0)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(43023)

    # Run masking on N_BATCHES batches with both standard and biased
    mask_prob = 0.15
    p_innov = 0.5
    p_copy = 0.0

    standard_stats = defaultdict(list)
    biased_stats = defaultdict(list)
    per_row_biased = []

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= N_BATCHES:
            break
        words = batch.pop("words")
        example_ids = batch.pop("example_ids")
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        word_group = batch["word_group"]

        bsz = input_ids.shape[0]
        special_ids_set = set(tokenizer.all_special_ids)
        special_ids_tensor = torch.tensor(sorted(special_ids_set))

        # Standard WWM for comparison
        gen_std = torch.Generator(device="cpu")
        gen_std.manual_seed(43023 + batch_idx * 1000)
        candidate = attention_mask.bool() & ~torch.isin(input_ids, special_ids_tensor)
        std_select = torch.zeros_like(candidate)
        for b in range(bsz):
            groups = word_group[b]
            valid = torch.unique(groups[groups >= 0])
            if valid.numel() == 0:
                continue
            gp = torch.rand(valid.numel(), generator=gen_std)
            chosen = valid[gp < mask_prob]
            if chosen.numel() > 0:
                std_select[b] = torch.isin(groups, chosen) & candidate[b]

        std_n_selected = int(std_select.sum().item())
        std_n_candidate = int(candidate.sum().item())
        standard_stats["selected"].append(std_n_selected)
        standard_stats["candidate"].append(std_n_candidate)
        standard_stats["rate"].append(std_n_selected / max(1, std_n_candidate))

        # Innovation-biased masking
        gen_bias = torch.Generator(device="cpu")
        gen_bias.manual_seed(43023 + batch_idx * 1000)

        _, labels, mask_stats = trainer.apply_innovation_biased_masking(
            input_ids, attention_mask, word_group, example_ids,
            tokenizer, mask_prob, p_innov, p_copy, innovation_map, gen_bias
        )
        bias_n_selected = int((labels != -100).sum().item())
        biased_stats["selected"].append(bias_n_selected)
        biased_stats["candidate"].append(std_n_candidate)
        biased_stats["rate"].append(bias_n_selected / max(1, std_n_candidate))
        biased_stats["changed_rows"].append(mask_stats.get("changed_rows", 0))
        biased_stats["unchanged_rows"].append(mask_stats.get("unchanged_rows", 0))
        biased_stats["innov_available"].append(mask_stats.get("innov_available", 0))
        biased_stats["innov_selected"].append(mask_stats.get("innov_selected", 0))
        biased_stats["copy_available"].append(mask_stats.get("copy_available", 0))
        biased_stats["copy_selected"].append(mask_stats.get("copy_selected", 0))
        biased_stats["other_selected"].append(mask_stats.get("other_selected", 0))

    def safe_stat(vals):
        if not vals:
            return {}
        return {"n": len(vals), "sum": sum(vals), "mean": round(statistics.mean(vals), 4),
                "min": min(vals), "max": max(vals)}

    innov_rate = (sum(biased_stats["innov_selected"]) /
                  max(1, sum(biased_stats["innov_available"])))
    copy_rate = (sum(biased_stats["copy_selected"]) /
                 max(1, sum(biased_stats["copy_available"])))

    summary = {
        "status": "INNOVATION_MASK_SMOKE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "n_batches": N_BATCHES,
            "batch_size": BATCH_SIZE,
            "mask_prob": mask_prob,
            "p_innov": p_innov,
            "p_copy": p_copy,
        },
        "standard_wwm": {
            "total_selected": sum(standard_stats["selected"]),
            "total_candidate": sum(standard_stats["candidate"]),
            "overall_rate": round(sum(standard_stats["selected"]) / max(1, sum(standard_stats["candidate"])), 6),
            "per_batch_rate": safe_stat(standard_stats["rate"]),
        },
        "innovation_biased_wwm": {
            "total_selected": sum(biased_stats["selected"]),
            "total_candidate": sum(biased_stats["candidate"]),
            "overall_rate": round(sum(biased_stats["selected"]) / max(1, sum(biased_stats["candidate"])), 6),
            "per_batch_rate": safe_stat(biased_stats["rate"]),
            "changed_rows_total": sum(biased_stats["changed_rows"]),
            "unchanged_rows_total": sum(biased_stats["unchanged_rows"]),
            "innovation_groups": {
                "available": sum(biased_stats["innov_available"]),
                "selected": sum(biased_stats["innov_selected"]),
                "empirical_rate": round(innov_rate, 4),
                "target_rate": p_innov,
            },
            "copyable_groups": {
                "available": sum(biased_stats["copy_available"]),
                "selected": sum(biased_stats["copy_selected"]),
                "empirical_rate": round(copy_rate, 4),
                "target_rate": p_copy,
            },
            "other_groups_selected": sum(biased_stats["other_selected"]),
        },
        "mass_comparison": {
            "standard_total_tokens": sum(standard_stats["selected"]),
            "biased_total_tokens": sum(biased_stats["selected"]),
            "ratio": round(sum(biased_stats["selected"]) / max(1, sum(standard_stats["selected"])), 4),
            "note": "Ratio near 1.0 means total selected mass is preserved",
        },
        "elapsed_sec": round(time.time() - t0, 2),
    }

    # Save
    (OUT_DIR / "innovation_mask_smoke.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    note = [
        "# research — Innovation-biased WWM mask smoke test",
        "",
        "CPU-only; no model training, no GPU.",
        "",
        f"- Batches tested: {N_BATCHES} × {BATCH_SIZE}",
        f"- p_innov={p_innov}, p_copy={p_copy}, mask_prob={mask_prob}",
        "",
        "## Mass comparison",
        f"- Standard WWM total selected tokens: {summary['standard_wwm']['total_selected']}",
        f"- Innovation-biased total selected tokens: {summary['innovation_biased_wwm']['total_selected']}",
        f"- Ratio (biased/standard): {summary['mass_comparison']['ratio']}",
        f"- Standard effective rate: {summary['standard_wwm']['overall_rate']}",
        f"- Biased effective rate: {summary['innovation_biased_wwm']['overall_rate']}",
        "",
        "## Innovation group selection",
        f"- Available: {summary['innovation_biased_wwm']['innovation_groups']['available']}",
        f"- Selected: {summary['innovation_biased_wwm']['innovation_groups']['selected']}",
        f"- Empirical rate: {summary['innovation_biased_wwm']['innovation_groups']['empirical_rate']} (target: {p_innov})",
        "",
        "## Copyable group selection",
        f"- Available: {summary['innovation_biased_wwm']['copyable_groups']['available']}",
        f"- Selected: {summary['innovation_biased_wwm']['copyable_groups']['selected']}",
        f"- Empirical rate: {summary['innovation_biased_wwm']['copyable_groups']['empirical_rate']} (target: {p_copy})",
        "",
        f"## Row counts ({N_BATCHES} batches)",
        f"- Changed rows: {summary['innovation_biased_wwm']['changed_rows_total']}",
        f"- Unchanged rows: {summary['innovation_biased_wwm']['unchanged_rows_total']}",
        "",
        f"Full JSON: `{OUT_DIR / 'innovation_mask_smoke.json'}`",
    ]
    (OUT_DIR / "innovation_mask_smoke.md").write_text("\n".join(note) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": summary["status"],
        "standard_rate": summary["standard_wwm"]["overall_rate"],
        "biased_rate": summary["innovation_biased_wwm"]["overall_rate"],
        "mass_ratio": summary["mass_comparison"]["ratio"],
        "innov_empirical_rate": innov_rate,
        "copy_empirical_rate": copy_rate,
        "changed_rows": summary["innovation_biased_wwm"]["changed_rows_total"],
        "elapsed_sec": summary["elapsed_sec"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
