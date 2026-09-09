#!/usr/bin/env python3
"""Summarize the completed semantic-view treatment and packet-local control trainings.

This is a CPU-only record of training artifacts.  It deliberately does not infer
BabyLM competence: the downstream no-AoA trajectory is the first score evidence
for the semantic-view mechanism.
"""
from __future__ import annotations

import json
import pathlib
from typing import Any

ROOT = pathlib.Path("experiments/archive/representation_and_objectives")
RUNS = ROOT / "training" / "runs"
OUT_DIR = ROOT / "data" / "semantic_training_pair"
NOTE = (ROOT / 'notes'.parents[3] / 'research/notes/representation_and_objectives/semantic_training_pair_complete.md')

RUN_INFO = {
    "semantic_view_treatment": RUNS / "semantic_view_treatment_8x480_16k_wwm_seed43022",
    "original_packet_local": RUNS / "original_packet_local_8x480_16k_wwm_seed43022",
}
REQUIRED = [f"chck_{i}M" for i in range(1, 101)]
IDENTITY_KEYS = [
    "variant", "backend", "model_family", "parameter_count", "vocab_size",
    "tokenizer_label", "word_exposure", "actual_training_steps", "masking_curriculum",
    "mask_prob_start", "mask_prob_end", "switch_frac", "n_amlm_updates",
    "hidden_size", "n_layer", "n_head", "ffn_mult", "seed", "seq_length",
    "max_seq_length", "seq_len_schedule",
]


def load_metrics(run: pathlib.Path) -> dict[str, Any]:
    p = run / "scientific_metrics.json"
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf-8"))


def summarize_one(name: str, run: pathlib.Path, m: dict[str, Any]) -> dict[str, Any]:
    ckpts = [str(x.get("name")) for x in m.get("saved_checkpoints", []) if isinstance(x, dict)]
    missing = [x for x in REQUIRED if x not in set(ckpts)]
    unexpected = [x for x in ckpts if x not in set(REQUIRED)]
    return {
        "name": name,
        "run": str(run),
        "metrics": str(run / "scientific_metrics.json"),
        "example_jsonl": m.get("example_jsonl"),
        "example_jsonl_label": m.get("example_jsonl_label"),
        "word_exposure": m.get("word_exposure"),
        "actual_training_steps": m.get("actual_training_steps"),
        "loss_first": m.get("loss_first"),
        "loss_last": m.get("loss_last"),
        "parameter_count": m.get("parameter_count"),
        "vocab_size": m.get("vocab_size"),
        "tokenizer_label": m.get("tokenizer_label"),
        "model_shape": {"n_layer": m.get("n_layer"), "hidden_size": m.get("hidden_size"), "n_head": m.get("n_head"), "ffn_mult": m.get("ffn_mult")},
        "training_recipe": {"masking_curriculum": m.get("masking_curriculum"), "seq_length": m.get("seq_length"), "seed": m.get("seed"), "mask_prob_start": m.get("mask_prob_start"), "mask_prob_end": m.get("mask_prob_end")},
        "checkpoint_count": len(ckpts),
        "first_checkpoint": ckpts[0] if ckpts else None,
        "last_checkpoint": ckpts[-1] if ckpts else None,
        "missing_required_checkpoints": missing,
        "unexpected_checkpoints": unexpected,
        "complete_1M_to_100M_ladder": (not missing and not unexpected and len(ckpts) == len(REQUIRED)),
        "source_words_consumed": m.get("source_words_consumed", {}),
    }


def numeric(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {name: load_metrics(run) for name, run in RUN_INFO.items()}
    rows = {name: summarize_one(name, RUN_INFO[name], metrics[name]) for name in RUN_INFO}

    identity = {}
    for k in IDENTITY_KEYS:
        vals = {name: metrics[name].get(k) for name in RUN_INFO}
        identity[k] = {"values": vals, "same": len({json.dumps(v, sort_keys=True, default=str) for v in vals.values()}) == 1}
    differing_identity = {k: v for k, v in identity.items() if not v["same"]}

    t_loss = numeric(rows["semantic_view_treatment"].get("loss_last"))
    c_loss = numeric(rows["original_packet_local"].get("loss_last"))
    loss_delta = None if t_loss is None or c_loss is None else t_loss - c_loss

    payload = {
        "status": "SEMANTIC_TRAINING_PAIR_SUMMARIZED",
        "purpose": "Record completed matched training artifacts before downstream no-AoA evaluation; training loss is not used as BabyLM competence evidence.",
        "targets": rows,
        "identity_comparison": identity,
        "differing_identity_fields": differing_identity,
        "all_core_training_fields_same": not differing_identity,
        "loss_last_treatment_minus_packet_local": loss_delta,
        "source_difference": {
            "treatment_unique_changed_source_key": "simplewiki_semantic_view",
            "control_unique_changed_source_key": "simplewiki_original_packet_local",
            "matched_word_exposure_for_changed_block_each": rows["semantic_view_treatment"].get("source_words_consumed", {}).get("simplewiki_semantic_view") or rows["original_packet_local"].get("source_words_consumed", {}).get("simplewiki_original_packet_local"),
            "unchanged_interpretation": "Both runs share recipe, seed, shape, tokenizer, exposure, and official filler accounting; the planned downstream comparison tests generated source-conservative views against packet-local same-source repetition, not the isolated value of training loss.",
        },
        "pending_downstream_file": "experiments/archive/representation_and_objectives/data/semantic_view_noaoa_eval/semantic_view_packet_local_delta_summary.json",
    }
    out = OUT_DIR / "semantic_training_pair_summary.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = []
    lines.append("# research semantic-view training pair complete\n\n")
    lines.append("This records the matched training artifacts only. The scientific effect is still pending the official-compatible no-AoA trajectory.\n\n")
    lines.append("| target | word exposure | steps | checkpoints | last loss | example file |\n")
    lines.append("|---|---:|---:|---:|---:|---|\n")
    for name in ["semantic_view_treatment", "original_packet_local"]:
        r = rows[name]
        lines.append(f"| {name} | {r['word_exposure']:,} | {r['actual_training_steps']:,} | {r['checkpoint_count']} | {float(r['loss_last']):.6f} | `{r['example_jsonl']}` |\n")
    lines.append("\nCore training fields identical across arms: **{}**.\n".format(not differing_identity))
    if differing_identity:
        lines.append("Differing training fields: `{}`.\n".format(", ".join(sorted(differing_identity))))
    lines.append("\nTreatment last loss minus packet-local last loss: **{:.6f}**. This reflects the different input distribution/repetition structure and is not downstream competence evidence.\n".format(loss_delta if loss_delta is not None else float("nan")))
    lines.append("\nChanged block exposure: treatment `simplewiki_semantic_view` = {:,}; control `simplewiki_original_packet_local` = {:,}.\n".format(
        int(rows["semantic_view_treatment"]["source_words_consumed"].get("simplewiki_semantic_view", 0)),
        int(rows["original_packet_local"]["source_words_consumed"].get("simplewiki_original_packet_local", 0)),
    ))
    lines.append("\nDownstream result awaited: `experiments/archive/representation_and_objectives/data/semantic_view_noaoa_eval/semantic_view_packet_local_delta_summary.json`.\n")
    lines.append(f"\nJSON: `{out}`\n")
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text("".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": payload["status"],
        "out": str(out),
        "note": str(NOTE),
        "all_core_training_fields_same": payload["all_core_training_fields_same"],
        "loss_last_treatment_minus_packet_local": loss_delta,
        "treatment_loss_last": rows["semantic_view_treatment"]["loss_last"],
        "packet_loss_last": rows["original_packet_local"]["loss_last"],
        "checkpoint_ladders_complete": {name: rows[name]["complete_1M_to_100M_ladder"] for name in rows},
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
