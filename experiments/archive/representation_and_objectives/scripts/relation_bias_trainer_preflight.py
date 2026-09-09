#!/usr/bin/env python3
"""CPU-only preflight for research relation-biased accumulated trainer.

This script does not train a BabyLM model and does not use any official evaluation
text.  It checks that the research trainer extension preserves the research masking
path when disabled, identifies broad relation-cue words in the exact training
stream, increases relation-cue prediction pressure when enabled while preserving
roughly the expected selected-group budget, and can instantiate the true 12x384
leader-shape intermediate size needed for a later architecture route.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace
from typing import Any

import torch
from transformers import AutoTokenizer

ROOT = Path(".").resolve()
STUDY = ROOT / "experiments/archive" / 'representation_and_objectives'
WS = STUDY
SCRIPT = WS / "scripts" / "relation_bias_accumulated_trainer.py"
TRAIN_100M = ROOT / "experiments/archive" / 'frontier_consolidation' / "data" / "density_cleanqwen_overlay_medium_riskhard" / "cleanqwen_fineweb_compact_view_reinvest_100M.jsonl"
TOKENIZER_40K = WS / "data" / "legal_representation_route_map" / "tokenizers" / "legal_byte_bpe_40k"
OUT_DIR = WS / "data" / "relation_bias_preflight"
OUT_JSON = OUT_DIR / "relation_bias_trainer_preflight.json"
NOTE = (ROOT / 'research/notes/representation_and_objectives/relation_bias_trainer_preflight.md')

EXPECTED_LEADER_SHAPE_12X384_40K = 38_421_952


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def import_trainer():
    spec = importlib.util.spec_from_file_location("relation_bias_accumulated_trainer", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def load_examples(rb, n: int) -> list[Any]:
    out = []
    with TRAIN_100M.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            obj = json.loads(line)
            text = str(obj["text"])
            words = int(obj.get("words", len(text.split())))
            if len(text.split()) != words:
                raise RuntimeError(f"word mismatch at row {i}")
            out.append(rb.base.Example(text=text, words=words, example_id=int(obj.get("example_id", i)), source=str(obj.get("source", ""))))
    if len(out) != n:
        raise RuntimeError(f"loaded {len(out)} examples, expected {n}")
    return out


def collate_items(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([x["input_ids"] for x in items]),
        "attention_mask": torch.stack([x["attention_mask"] for x in items]),
        "word_group": torch.stack([x["word_group"] for x in items]),
        "relation_group": torch.stack([x["relation_group"] for x in items]),
        "words": torch.stack([x["words"] for x in items]).long(),
    }


def state(rb, curriculum: str = "wwm_fixed", current_step: int = 0, total_steps: int = 100) -> Any:
    st = rb.base.MaskingCurriculumState(
        curriculum=curriculum,
        mask_prob_start=0.15,
        mask_prob_end=0.15,
        switch_frac=0.7,
        amlm_window=10,
        amlm_lambda=0.2,
    )
    st.initialize(vocab_size=40_000, total_steps=total_steps)
    st.current_step = current_step
    return st


def mean(xs: list[float]) -> float | None:
    return float(sum(xs) / len(xs)) if xs else None


def stdev(xs: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    return float(statistics.pstdev(xs))


def q(xs: list[float], p: float) -> float | None:
    xs = sorted(float(x) for x in xs if math.isfinite(float(x)))
    if not xs:
        return None
    pos = p * (len(xs) - 1)
    lo = int(math.floor(pos)); hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "candidate_groups", "candidate_relation_groups", "selected_groups", "selected_relation_groups",
        "candidate_tokens", "candidate_relation_tokens", "selected_tokens", "selected_relation_tokens",
        "candidate_relation_group_fraction", "selected_relation_group_fraction",
        "candidate_relation_token_fraction", "selected_relation_token_fraction",
        "mean_relation_group_prob", "mean_nonrelation_group_prob",
        "probability_rows_clipped",
    ]
    out: dict[str, Any] = {"n_reps": len(records)}
    for k in keys:
        vals = [r.get(k) for r in records if isinstance(r.get(k), (int, float)) and r.get(k) is not None]
        if not vals:
            continue
        out[k + "_mean"] = mean([float(v) for v in vals])
        out[k + "_std"] = stdev([float(v) for v in vals])
        out[k + "_p10"] = q([float(v) for v in vals], 0.10)
        out[k + "_p90"] = q([float(v) for v in vals], 0.90)
    if records:
        cand_groups = records[0].get("candidate_groups")
        cand_rel_groups = records[0].get("candidate_relation_groups")
        cand_tokens = records[0].get("candidate_tokens")
        cand_rel_tokens = records[0].get("candidate_relation_tokens")
        out["candidate_groups_static"] = cand_groups
        out["candidate_relation_groups_static"] = cand_rel_groups
        out["candidate_relation_group_fraction_static"] = cand_rel_groups / cand_groups if cand_groups else None
        out["candidate_tokens_static"] = cand_tokens
        out["candidate_relation_tokens_static"] = cand_rel_tokens
        out["candidate_relation_token_fraction_static"] = cand_rel_tokens / cand_tokens if cand_tokens else None
        out["expected_uniform_selected_groups_at_p015"] = 0.15 * cand_groups if cand_groups else None
        if out.get("selected_groups_mean") and cand_groups:
            out["selected_groups_multiplier_vs_uniform_expectation"] = out["selected_groups_mean"] / (0.15 * cand_groups)
    return out


def sample_mask_stats(rb, batch: dict[str, torch.Tensor], tok, boost: float, mode: str = "wwm", reps: int = 60) -> dict[str, Any]:
    records = []
    for rep in range(reps):
        if mode == "wwm":
            st = state(rb, "wwm_fixed", current_step=0, total_steps=100)
        elif mode == "token":
            st = state(rb, "wwm_to_token", current_step=80, total_steps=100)
        else:
            raise ValueError(mode)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(10_000 + rep)
        _masked, _labels, stats = rb.apply_masking_with_relation(
            batch["input_ids"],
            batch["attention_mask"],
            batch["word_group"],
            batch["relation_group"],
            tok,
            st,
            gen,
            relation_enabled=True,
            relation_cue_boost=boost,
            relation_prob_max=0.6,
        )
        records.append(stats)
    out = aggregate(records)
    out["boost"] = boost
    out["mode"] = mode
    return out


def build_param_count(rb, tok, hidden_size: int, layers: int, heads: int, intermediate_size: int) -> int:
    args = SimpleNamespace(
        hidden_size=hidden_size,
        n_layer=layers,
        n_head=heads,
        ffn_mult=4,
        intermediate_size=intermediate_size,
        max_position_embeddings=512,
        max_seq_length=256,
        max_relative_positions=256,
        position_buckets=256,
        deberta_pos_att_type="p2c,c2p",
    )
    model = rb.build_model(args, tok)
    n = sum(p.numel() for p in model.parameters())
    del model
    return int(n)


def write_note(payload: dict[str, Any]) -> None:
    checks = payload["checks"]
    wwm2 = payload["boost_sampling"]["wwm_boost_2.0"]
    wwm3 = payload["boost_sampling"]["wwm_boost_3.0"]
    token2 = payload["boost_sampling"]["token_boost_2.0"]
    arch = payload["architecture"]
    lines = ["# research relation-biased trainer preflight\n\n"]
    lines.append("CPU-only test of `relation_bias_accumulated_trainer.py`; no BabyLM model training and no official evaluation text.\n\n")
    lines.append("## Checks\n\n")
    for k, v in checks.items():
        lines.append(f"- `{k}`: `{v}`\n")
    lines.append("\n## Real 100M-stream first-256-row masking sample\n\n")
    lines.append(f"- Candidate visible word groups: `{wwm2['candidate_groups_static']}`; relation-bearing groups: `{wwm2['candidate_relation_groups_static']}` (`{wwm2['candidate_relation_group_fraction_static']:.4f}`).\n")
    lines.append(f"- WWM boost 2.0 selected relation-group fraction mean: `{wwm2['selected_relation_group_fraction_mean']:.4f}`; selected groups multiplier vs uniform expectation: `{wwm2['selected_groups_multiplier_vs_uniform_expectation']:.4f}`.\n")
    lines.append(f"- WWM boost 3.0 selected relation-group fraction mean: `{wwm3['selected_relation_group_fraction_mean']:.4f}`; selected groups multiplier vs uniform expectation: `{wwm3['selected_groups_multiplier_vs_uniform_expectation']:.4f}`.\n")
    lines.append(f"- Token-mode boost 2.0 selected relation-token fraction mean: `{token2['selected_relation_token_fraction_mean']:.4f}` vs candidate `{token2['candidate_relation_token_fraction_static']:.4f}`.\n")
    lines.append("\n## Architecture extension\n\n")
    lines.append(f"- 12x384 legal40k with explicit intermediate_size=1280 has `{arch['legal40k_12x384_intermediate1280_param_count']}` parameters (expected `{EXPECTED_LEADER_SHAPE_12X384_40K}`).\n")
    lines.append("\n## Interpretation\n\n")
    lines.append("The extension preserves the original masking path when relation bias is disabled, so it can serve as a controlled post-40k objective route. When enabled, it raises relation-cue prediction pressure inside the existing legal corpus while keeping selected word-group count near the uniform 0.15 expectation. It remains only a prepared route until the current legal40k official vectors determine whether a relation/predicate learning-signal experiment is scientifically justified.\n\n")
    lines.append(f"JSON: `{rel(OUT_JSON)}`\n")
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rb = import_trainer()
    tok = rb.base.make_portable_tokenizer(str(TOKENIZER_40K))
    examples = load_examples(rb, 256)
    cue_lexicon, multi_cues = rb.load_step54_lexicon()
    cue_single = rb.build_cue_single(cue_lexicon)

    base_ds = rb.base.MaskedChunkDataset(examples, tok, 256)
    rel_ds = rb.RelationMaskedChunkDataset(examples, tok, 256, cue_single, multi_cues)

    mismatches = []
    for i in range(64):
        a = base_ds[i]
        b = rel_ds[i]
        if not torch.equal(a["input_ids"], b["input_ids"]):
            mismatches.append({"row": i, "field": "input_ids"})
        if not torch.equal(a["attention_mask"], b["attention_mask"]):
            mismatches.append({"row": i, "field": "attention_mask"})
        if not torch.equal(a["word_group"], b["word_group"]):
            mismatches.append({"row": i, "field": "word_group"})
    batch = collate_items([rel_ds[i] for i in range(256)])

    # Disabled wrapper should be exactly the base masking path given identical RNG state.
    st_base = state(rb, "wwm_fixed", current_step=0, total_steps=100)
    st_wrap = state(rb, "wwm_fixed", current_step=0, total_steps=100)
    gen_a = torch.Generator(device="cpu"); gen_a.manual_seed(12345)
    gen_b = torch.Generator(device="cpu"); gen_b.manual_seed(12345)
    masked_a, labels_a = rb.base.apply_masking_curriculum(
        batch["input_ids"], batch["attention_mask"], batch["word_group"], tok, st_base, gen_a,
    )
    masked_b, labels_b, disabled_stats = rb.apply_masking_with_relation(
        batch["input_ids"], batch["attention_mask"], batch["word_group"], batch["relation_group"], tok, st_wrap, gen_b,
        relation_enabled=False, relation_cue_boost=1.0,
    )
    disabled_equal = bool(torch.equal(masked_a, masked_b) and torch.equal(labels_a, labels_b))

    boosts = {
        "wwm_boost_1.5": sample_mask_stats(rb, batch, tok, 1.5, mode="wwm", reps=60),
        "wwm_boost_2.0": sample_mask_stats(rb, batch, tok, 2.0, mode="wwm", reps=60),
        "wwm_boost_3.0": sample_mask_stats(rb, batch, tok, 3.0, mode="wwm", reps=60),
        "token_boost_2.0": sample_mask_stats(rb, batch, tok, 2.0, mode="token", reps=40),
    }

    # Assert the intended scientific mechanics, but keep numerical tolerances broad
    # because this is a stochastic first-256-row sample from the actual stream.
    wwm2 = boosts["wwm_boost_2.0"]
    wwm3 = boosts["wwm_boost_3.0"]
    token2 = boosts["token_boost_2.0"]
    candidate_rel_frac = wwm2["candidate_relation_group_fraction_static"]
    checks = {
        "train_stream_exists": TRAIN_100M.exists(),
        "tokenizer_exists": TOKENIZER_40K.exists(),
        "word_group_identity_first64": len(mismatches) == 0,
        "disabled_relation_wrapper_exactly_matches_base_masking": disabled_equal,
        "boost2_raises_selected_relation_group_fraction": wwm2["selected_relation_group_fraction_mean"] > candidate_rel_frac + 0.04,
        "boost3_raises_more_than_boost2": wwm3["selected_relation_group_fraction_mean"] > wwm2["selected_relation_group_fraction_mean"] + 0.04,
        "boost2_preserves_selected_group_budget_roughly": 0.85 <= wwm2["selected_groups_multiplier_vs_uniform_expectation"] <= 1.15,
        "token_mode_boost2_raises_relation_token_fraction": token2["selected_relation_token_fraction_mean"] > token2["candidate_relation_token_fraction_static"] + 0.03,
    }

    arch_count = build_param_count(rb, tok, hidden_size=384, layers=12, heads=12, intermediate_size=1280)
    checks["leader_shape_12x384_40k_param_count_matches_step64"] = arch_count == EXPECTED_LEADER_SHAPE_12X384_40K

    payload = {
        "status": "RELATION_BIAS_TRAINER_PREFLIGHT",
        "script": rel(SCRIPT),
        "inputs": {
            "train_100m": rel(TRAIN_100M),
            "tokenizer_40k": rel(TOKENIZER_40K),
            "examples_from_real_stream": 256,
        },
        "checks": checks,
        "mismatches_first64": mismatches[:10],
        "disabled_wrapper_stats": disabled_stats,
        "boost_sampling": boosts,
        "architecture": {
            "legal40k_12x384_intermediate1280_param_count": arch_count,
            "expected_from_step64": EXPECTED_LEADER_SHAPE_12X384_40K,
        },
        "files": {"json": rel(OUT_JSON), "note": rel(NOTE)},
    }
    all_ok = all(bool(v) for v in checks.values())
    payload["all_checks_passed"] = all_ok
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_note(payload)
    print(json.dumps({
        "status": payload["status"],
        "all_checks_passed": all_ok,
        "candidate_relation_group_fraction": candidate_rel_frac,
        "wwm_boost2_selected_relation_group_fraction_mean": wwm2["selected_relation_group_fraction_mean"],
        "wwm_boost2_selected_group_multiplier": wwm2["selected_groups_multiplier_vs_uniform_expectation"],
        "param_count_12x384_legal40k_intermediate1280": arch_count,
        "out_json": rel(OUT_JSON),
        "note": rel(NOTE),
    }, indent=2), flush=True)
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
