#!/usr/bin/env python3
"""research: CPU preflight for a minimal DeBERTa positional-component ablation.

Scientific purpose
------------------
The compact-view data effect is strong in the legal stock DeBERTa-v2 coordinate
but did not transfer to GPT2 causal LM or stock RoBERTa under previously tested
coordinates. The remaining architecture question is narrow: which one DeBERTa
positional component is sufficient to preserve the compact-minus-repeat data
effect while the data, tokenizer, objective, masking, optimization, and legal
exposure are otherwise fixed.

This script only checks local files, config/math, and exact streams. It does
not launch GPU training, run official evaluation, upload, or submit.

Key source correction found during research: the existing mature repeat DeBERTa
run uses the non-legal inherited `baseline16k` tokenizer, while the legal
compact reference and research extractive arms use the legal research tokenizer.
Therefore any new compact-vs-repeat ablation must train both data arms under
the legal research tokenizer in the modified architecture; an old-tokenizer
repeat run cannot serve as the matched reference for a legal ablation family.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import hashlib
import json
import pathlib
from typing import Any

from transformers import AutoTokenizer, DebertaV2Config, DebertaV2ForMaskedLM


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
ROOT = USER_ROOT / "experiments/archive" / 'frontier_consolidation'
WORKSPACE = ROOT
DATA = WORKSPACE / "data"
A01 = USER_ROOT / "experiments/archive" / 'representation_and_objectives'

LEGAL_TOKENIZER = DATA / "compliant_tokenizer"
COMPACT_STREAM = DATA / "density_cleanqwen_overlay_medium_riskhard" / "cleanqwen_fineweb_compact_view_reinvest_100M.jsonl"
REPEAT_STREAM = DATA / "density_cleanqwen_overlay_medium_riskhard" / "cleanqwen_fineweb_repeat_compact_reinvest_100M.jsonl"
COMPACT_REFERENCE = WORKSPACE / "training" / "runs" / "complianttok_reinvest_seed43022_r2"
OLD_REPEAT_REFERENCE = A01 / "training" / "runs" / "gc_compact_repeat_reinvest_16k_seed43022_r2"

EXPECTED_TOK_SHA = "91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9"
EXPECTED_COMPACT_STREAM_SHA = "3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691"
EXPECTED_REPEAT_STREAM_SHA = "91e8817e25f234b87481fa6ca96d968d6c460b84317273629eda21ea3be24eaa"
EXPECTED_BASELINE = 34_467_424

VARIANTS: dict[str, dict[str, Any]] = {
    "c2p_only_abs": {
        "relative_attention": True,
        "pos_att_type": ["c2p"],
        "position_biased_input": True,
        "scientific_role": "retain content->position disentangled channel plus absolute input positions; removes p2c",
    },
    "p2c_only_abs": {
        "relative_attention": True,
        "pos_att_type": ["p2c"],
        "position_biased_input": True,
        "scientific_role": "retain position->content disentangled channel plus absolute input positions; removes c2p",
    },
    "no_disentangle_abs": {
        "relative_attention": True,
        "pos_att_type": [],
        "position_biased_input": True,
        "scientific_role": "retain rel_embeddings/absolute input positions but remove both attention score terms; isolates disentangled score contribution",
    },
}


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stream_counts(path: pathlib.Path, scan: bool) -> dict[str, Any]:
    out: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return out
    out["sha256"] = sha256_file(path)
    if not scan:
        return out
    rows = 0
    words = 0
    changed = 0
    unique_changed: dict[int, int] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rows += 1
            text = str(obj.get("text", ""))
            w = int(obj.get("words", len(text.split())))
            actual = len(text.split())
            if w != actual:
                raise RuntimeError(f"word mismatch {path} row {rows}: field={w} actual={actual}")
            words += w
            eid = obj.get("example_id")
            if isinstance(eid, int) and 950000 <= eid <= 953004:
                changed += 1
                unique_changed[eid] = unique_changed.get(eid, 0) + 1
    out.update({
        "rows": rows,
        "words": words,
        "changed_rows": changed,
        "unique_changed_ids": len(unique_changed),
        "bad_changed_repeat_counts": {str(k): v for k, v in unique_changed.items() if v != 10},
        "exact_100M": rows == 647400 and words == 100000000,
    })
    return out


def model_variant_info(tokenizer_len: int, tokenizer_ids: dict[str, int]) -> dict[str, Any]:
    base = dict(
        vocab_size=tokenizer_len,
        hidden_size=480,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=1920,
        max_position_embeddings=512,
        max_relative_positions=256,
        position_buckets=256,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        pad_token_id=tokenizer_ids["pad_token_id"],
        bos_token_id=tokenizer_ids["bos_token_id"],
        eos_token_id=tokenizer_ids["eos_token_id"],
    )
    rows = []
    for name, cfgadd in VARIANTS.items():
        cfg = DebertaV2Config(**base, **{k: v for k, v in cfgadd.items() if k != "scientific_role"})
        model = DebertaV2ForMaskedLM(cfg)
        params = sum(p.numel() for p in model.parameters())
        state_keys = list(model.state_dict().keys())
        rel_pos_keys = [k for k in state_keys if "rel" in k or "pos" in k]
        rows.append({
            "variant": name,
            "config": {k: cfgadd[k] for k in ("relative_attention", "pos_att_type", "position_biased_input")},
            "scientific_role": cfgadd["scientific_role"],
            "parameter_count": params,
            "delta_vs_baseline": params - EXPECTED_BASELINE,
            "rel_pos_tensor_count": len(rel_pos_keys),
            "has_encoder_rel_embeddings": any(k.endswith("encoder.rel_embeddings.weight") for k in state_keys),
            "has_position_embeddings": any(k.endswith("embeddings.position_embeddings.weight") for k in state_keys),
            "has_pos_key_proj": any("pos_key_proj" in k for k in state_keys),
            "has_pos_query_proj": any("pos_query_proj" in k for k in state_keys),
        })
    return {"variants": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(DATA / "deberta_positional_ablation_preflight"))
    ap.add_argument("--scan-streams", action="store_true", help="Scan full 100M JSONL streams (slower but exact)")
    args = ap.parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(str(LEGAL_TOKENIZER), use_fast=True)
    tok_sha = sha256_file(LEGAL_TOKENIZER / "tokenizer.json")
    compact_metrics = read_json(COMPACT_REFERENCE / "scientific_metrics.json")
    old_repeat_metrics = read_json(OLD_REPEAT_REFERENCE / "scientific_metrics.json")

    result: dict[str, Any] = {
        "status": "DEBERTA_POSITIONAL_ABLATION_PREFLIGHT",
        "no_gpu_training_launched": True,
        "no_official_eval_upload_aoa_or_leaderboard": True,
        "legal_tokenizer": {
            "path": str(LEGAL_TOKENIZER),
            "tokenizer_json_sha256": tok_sha,
            "matches_expected_legal_sha": tok_sha == EXPECTED_TOK_SHA,
            "vocab_size": len(tok),
            "special_tokens": tok.special_tokens_map,
        },
        "references": {
            "legal_compact_full_deberta": {
                "run_dir": str(COMPACT_REFERENCE),
                "tokenizer_label": compact_metrics.get("tokenizer_label"),
                "parameter_count": compact_metrics.get("parameter_count"),
                "word_exposure": compact_metrics.get("word_exposure"),
                "actual_training_steps": compact_metrics.get("actual_training_steps"),
                "loss_last": compact_metrics.get("loss_last"),
                "example_jsonl": compact_metrics.get("example_jsonl"),
            },
            "old_repeat_full_deberta_not_matched_for_new_ablation": {
                "run_dir": str(OLD_REPEAT_REFERENCE),
                "tokenizer_label": old_repeat_metrics.get("tokenizer_label"),
                "parameter_count": old_repeat_metrics.get("parameter_count"),
                "word_exposure": old_repeat_metrics.get("word_exposure"),
                "actual_training_steps": old_repeat_metrics.get("actual_training_steps"),
                "loss_last": old_repeat_metrics.get("loss_last"),
                "example_jsonl": old_repeat_metrics.get("example_jsonl"),
                "why_not_matched": "uses non-legal inherited baseline16k tokenizer, so a legal ablation family must train both compact and repeat under research tokenizer",
            },
        },
        "streams": {
            "compact": stream_counts(COMPACT_STREAM, scan=args.scan_streams),
            "repeat": stream_counts(REPEAT_STREAM, scan=args.scan_streams),
        },
        "model_variants": model_variant_info(
            tokenizer_len=len(tok),
            tokenizer_ids={"pad_token_id": tok.pad_token_id, "bos_token_id": tok.bos_token_id, "eos_token_id": tok.eos_token_id},
        ),
        "minimum_exact_contrast": {
            "data_arms": ["compact_view_reinvest_100M", "repeat_compact_reinvest_100M"],
            "tokenizer": "legal research compliant16k_reinvest10M for every arm",
            "architecture_family": "DebertaV2ForMaskedLM 8x480/8h, one positional component changed per arm",
            "learning_recipe": "research recipe: seeds 43/43022/43023, WWM p=0.15, AdamW lr=0.001, batch256/seq256, lr_total_steps=2529, exact 100M exposure",
            "interpretation": "the scientific estimand is compact-minus-repeat within each ablated architecture, not a single-arm gain and not a comparison to a different-tokenizer old repeat run",
        },
        "warnings": [
            "Do not use a 20M single-arm masking-rate gain to justify continuation; many early gains reversed in the recorded comparisons.",
            "The existing mature repeat DeBERTa run is tokenizer-mismatched to the legal coordinate; it is provenance/context, not the ablation reference.",
            "Parameter counts change under component ablation; the primary estimand is the data-treatment interaction, not equal total parameters.",
        ],
    }

    out_json = out_dir / "deberta_positional_ablation_preflight.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_md = out_dir / "deberta_positional_ablation_preflight.md"
    lines = [
        "# research DeBERTa positional ablation preflight",
        "",
        "No GPU training/evaluation/upload/leaderboard action occurred.",
        "",
        f"Legal tokenizer SHA matches expected: `{result['legal_tokenizer']['matches_expected_legal_sha']}`",
        f"Old mature repeat tokenizer label: `{old_repeat_metrics.get('tokenizer_label')}` (not legal-matched)",
        "",
        "| variant | params | delta vs full | rel_attention | pos_att_type | abs input | role |",
        "|---|---:|---:|---|---|---|---|",
    ]
    for row in result["model_variants"]["variants"]:
        cfg = row["config"]
        lines.append(
            f"| {row['variant']} | {row['parameter_count']} | {row['delta_vs_baseline']} | "
            f"{cfg['relative_attention']} | {','.join(cfg['pos_att_type']) or 'none'} | {cfg['position_biased_input']} | {row['scientific_role']} |"
        )
    lines.extend([
        "",
        "Required exact estimand: compact-minus-repeat within each ablated architecture under the legal research tokenizer.",
        "The old baseline16k repeat run cannot be reused as the legal reference for a new ablated architecture.",
    ])
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "out_json": str(out_json), "out_md": str(out_md), "scan_streams": args.scan_streams}, indent=2))


if __name__ == "__main__":
    main()
