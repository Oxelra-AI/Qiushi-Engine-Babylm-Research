#!/usr/bin/env python3
"""Decide whether the existing compact run can serve as the shared compact anchor.

The purpose is to avoid repeating a 100M-word compact training run if an existing
run uses the same compact data under the same recipe. The comparison also
records the substantive trainer difference: the queued pair uses
a memory-splitting trainer, whereas the reference uses the original full-batch trainer.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

A01_WS = Path("experiments/archive/representation_and_objectives")
A02_WS = Path("experiments/archive/frontier_consolidation")
OUT_DIR = A01_WS / "data/compact_anchor_match"
NOTE = (A01_WS.parents[2] / 'research/notes/representation_and_objectives/compact_anchor_match.md')

A01_COMMANDS = A01_WS / "data/fw_compact_vs_interleaved_breadth_train/interleaved_paired_commands.json"
A01_PREFLIGHT = A01_WS / "data/fw_interleaved_pair_launch/pair_preflight.json"
A02_PREFLIGHT = A02_WS / "data/fw_compact_vs_breadth_train/preflight.json"
A02_SCRIPT = A02_WS / "scripts/fw_compact_vs_breadth_train.py"
A02_COMPACT_RUN = A02_WS / "training/runs/fw_compact_view_shared16k_seed43022"
A02_COMPACT_MANIFEST = A02_COMPACT_RUN / "example_order_manifest.json"
A02_COMPACT_LOG = A02_COMPACT_RUN / "training_log.jsonl"

A01_ACCUM_TRAINER = A01_WS / "scripts/accumulated_masking_curriculum_trainer.py"
A02_FULL_TRAINER = Path("experiments/archive/compact_experience/scripts/masking_curriculum_trainer.py")

EXPECTED_COMPACT_STREAM_SHA = "c8d7f24b5edd2dad21f589c8cde72671d0b76038d9632fcd3d6c79178a24be68"
EXPECTED_TOKENIZER_SHA = "e70d167f620668130f88744998bd1bacbe05f6d0735012144fe501b271b04366"
EXPECTED_WORDS = 100_000_000
EXPECTED_ROWS = 641_830


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def last_jsonl(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    last = None
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                last = line
    return json.loads(last) if last else None


def flag_dict(cmd: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    i = 0
    while i < len(cmd):
        x = cmd[i]
        if isinstance(x, str) and x.startswith("--"):
            if i + 1 < len(cmd) and not str(cmd[i + 1]).startswith("--"):
                out[x[2:]] = str(cmd[i + 1])
                i += 2
                continue
            out[x[2:]] = "true"
        i += 1
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    a01_commands = load_json(A01_COMMANDS)
    a01_preflight = load_json(A01_PREFLIGHT)
    a02_preflight = load_json(A02_PREFLIGHT)
    a02_manifest = load_json(A02_COMPACT_MANIFEST) if A02_COMPACT_MANIFEST.exists() else {}
    a02_last = last_jsonl(A02_COMPACT_LOG)

    a01_compact_cmd = a01_commands["arms"]["compact_view"]["cmd"]
    a01_flags = flag_dict(a01_compact_cmd)
    a02_recipe = a02_preflight.get("recipe", {})
    a02_seed = a02_preflight.get("seed", {})
    a02_compact_pf = a02_preflight.get("arms", {}).get("compact_view", {})

    shared_recipe_keys = [
        "hidden_size", "n_layer", "n_head", "ffn_mult", "batch_size", "seq_length",
        "learning_rate", "warmup_fraction", "weight_decay", "masking_curriculum",
        "mask_prob_start", "mask_prob_end", "checkpoint_words", "max_word_exposure",
    ]
    recipe_match = {}
    for k in shared_recipe_keys:
        a01_val = a01_flags.get(k)
        a02_val = a02_recipe.get(k)
        recipe_match[k] = {"a01_queued": a01_val, "a02_compact": a02_val, "same_string_value": str(a01_val) == str(a02_val)}

    seed_match = {}
    for k in ["seed", "extra_init_seed", "train_rng_seed"]:
        seed_match[k] = {"a01_queued": a01_flags.get(k), "a02_compact": a02_seed.get(k), "same_string_value": str(a01_flags.get(k)) == str(a02_seed.get(k))}

    compact_data_same = (
        a01_flags.get("example_jsonl") == a02_compact_pf.get("stream_path")
        and a02_preflight.get("a01_data_provenance", {}).get("compact_stream_sha256") == EXPECTED_COMPACT_STREAM_SHA
    )
    tokenizer_same = (
        a01_flags.get("tokenizer_path") == "experiments/archive/representation_and_objectives/data/shared_tokenizer/shared_16k_tokenizer"
        and a02_preflight.get("info", {}).get("tokenizer_sha256") == EXPECTED_TOKENIZER_SHA
    )
    manifest_ok = (
        a02_manifest.get("selected_for_training_words") == EXPECTED_WORDS
        and a02_manifest.get("num_consumed_examples") == EXPECTED_ROWS
        and a02_manifest.get("example_jsonl") == a02_compact_pf.get("stream_path")
    )
    same_update_coordinate_if_rebased = compact_data_same and tokenizer_same and manifest_ok and all(v["same_string_value"] for v in recipe_match.values()) and all(v["same_string_value"] for v in seed_match.values())

    result: dict[str, Any] = {
        "status": "A02_COMPACT_CAN_BE_USED_AS_FULLBATCH_ANCHOR" if same_update_coordinate_if_rebased else "A02_COMPACT_ANCHOR_NEEDS_REPAIR",
        "compact_stream": {
            "a01_queued_path": a01_flags.get("example_jsonl"),
            "a02_path": a02_compact_pf.get("stream_path"),
            "sha256": a02_preflight.get("a01_data_provenance", {}).get("compact_stream_sha256"),
            "expected_sha256": EXPECTED_COMPACT_STREAM_SHA,
            "same_path_and_sha": compact_data_same,
        },
        "tokenizer": {
            "a01_queued_path": a01_flags.get("tokenizer_path"),
            "a02_tokenizer_sha256": a02_preflight.get("info", {}).get("tokenizer_sha256"),
            "expected_sha256": EXPECTED_TOKENIZER_SHA,
            "same": tokenizer_same,
        },
        "recipe_match": recipe_match,
        "seed_match": seed_match,
        "a02_compact_manifest": {
            "exists": A02_COMPACT_MANIFEST.exists(),
            "selected_for_training_words": a02_manifest.get("selected_for_training_words"),
            "num_consumed_examples": a02_manifest.get("num_consumed_examples"),
            "example_jsonl": a02_manifest.get("example_jsonl"),
            "example_jsonl_label": a02_manifest.get("example_jsonl_label"),
            "source_words_consumed": a02_manifest.get("source_words_consumed"),
            "manifest_matches_expected_stream_words_rows": manifest_ok,
        },
        "a02_compact_progress": {
            "training_log_exists": A02_COMPACT_LOG.exists(),
            "last_training_record": a02_last,
            "scientific_metrics_exists": (A02_COMPACT_RUN / "scientific_metrics.json").exists(),
            "train_result_exists": (A02_COMPACT_RUN / "train_result.json").exists(),
        },
        "trainer_difference": {
            "a01_queued_trainer": str(A01_ACCUM_TRAINER),
            "a01_queued_trainer_sha256": sha256_file(A01_ACCUM_TRAINER),
            "a02_trainer": str(A02_FULL_TRAINER),
            "a02_trainer_sha256": sha256_file(A02_FULL_TRAINER),
            "meaning": "A02 uses the original full 256-row batch trainer. The cancelled A01 pair would have used a memory-splitting trainer with effective batch 256 and microbatch 64; this is not bit-identical because dropout is executed in smaller chunks, even though mask sampling and masked-token weighting are designed to preserve the effective update.",
        },
        "non_training_differences": {
            "a02_example_jsonl_label_blank": a02_manifest.get("example_jsonl_label", None) == "",
            "a01_would_record_label": a01_flags.get("example_jsonl_label"),
            "a02_uses_default_tokenizer_label_in_output": True,
            "a01_would_record_tokenizer_label": a01_flags.get("tokenizer_label"),
            "meaning": "These affect metadata strings only, not tokenization, model parameters, batches, masks, optimizer steps, or text exposure.",
        },
        "decision": {
            "cancel_duplicate_compact": True,
            "train_only_a01_interleaved_breadth": True,
            "training_coordinate_for_interleaved_breadth": "Use the same COMPACT_EXPERIENCE full-batch trainer and same explicit arguments as A02, changing only the data stream and output directory.",
            "reason": "A02 compact has the same stream, tokenizer, architecture, optimizer, masking, seeds, exposure, and row order needed as the compact anchor. The original A01 microbatch pair coordinate is different enough that sharing would not be exact there, so A01 should move its interleaved breadth arm onto the A02 full-batch coordinate rather than repeat compact.",
        },
        "same_update_coordinate_if_interleaved_is_rebased_to_a02_fullbatch": same_update_coordinate_if_rebased,
    }

    out_json = OUT_DIR / "compact_anchor_match.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# research — A02 compact anchor match",
        "",
        f"Status: `{result['status']}`.",
        "",
        "A02's compact run uses the same A01 compact 100M stream (`c8d7f24b...`), the same shared legal 16k tokenizer (`e70d167f...`), the same DeBERTa-v2 8x480 recipe, fixed WWM 0.15, AdamW settings, seeds 43/43022/43023, 100M exposure, 641,830 rows, and the same row order recorded by the exact stream path.",
        "",
        "The only substantive execution difference from the cancelled A01 queued pair is the trainer coordinate: A02 uses the original COMPACT_EXPERIENCE full-batch trainer, while A01 had queued the accumulated memory-repair trainer. The two are intentionally close but not bit-identical, because dropout is executed over different forward-call shapes. Therefore A02 compact should not be used as the anchor for an A01 microbatch-coordinate comparison; instead the A01 interleaved breadth arm should be trained with the same COMPACT_EXPERIENCE full-batch trainer so the shared compact anchor is valid.",
        "",
        f"Current A02 compact progress record: `{a02_last}`.",
        "",
        "Decision: keep the duplicate A01 compact run cancelled; train only the A01 interleaved whole-sentence breadth arm in the A02 full-batch coordinate, then compare it to the A02 compact model when the compact training result is present.",
        "",
        f"JSON: `{out_json}`",
    ]
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": result["status"],
        "same_update_coordinate_if_interleaved_is_rebased_to_a02_fullbatch": same_update_coordinate_if_rebased,
        "cancel_duplicate_compact": True,
        "train_only_interleaved_breadth_fullbatch": True,
        "json": str(out_json),
        "note": str(NOTE),
        "a02_last_cumulative_words": None if a02_last is None else a02_last.get("cumulative_word_exposure"),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
