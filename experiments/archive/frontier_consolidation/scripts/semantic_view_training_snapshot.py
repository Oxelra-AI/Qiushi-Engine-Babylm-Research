#!/usr/bin/env python3
"""Record representation_and_objectives semantic-view treatment/control training state from metrics.

This does not read the locked downstream evaluator directory and does not infer a
BabyLM score.  It preserves the matched-training identity so frontier_consolidation can decide
quickly when the no-AoA trajectory summary becomes available.
"""
from __future__ import annotations

import json
import pathlib
from typing import Any

ROOT = pathlib.Path("experiments/archive/representation_and_objectives")
OUT_DIR = pathlib.Path("experiments/archive/frontier_consolidation/data/semantic_view_pending")
NOTE = pathlib.Path("research/notes/frontier_consolidation/semantic_view_pending_and_training_snapshot.md")
RUNS = {
    "semantic_view_treatment": ROOT / "training/runs/semantic_view_treatment_8x480_16k_wwm_seed43022",
    "original_packet_local": ROOT / "training/runs/original_packet_local_8x480_16k_wwm_seed43022",
}
IDENTITY_KEYS = [
    "variant", "backend", "model_family", "parameter_count", "vocab_size", "tokenizer_label",
    "word_exposure", "actual_training_steps", "masking_curriculum", "mask_prob_start", "mask_prob_end",
    "hidden_size", "n_layer", "n_head", "ffn_mult", "seed", "seq_length", "max_seq_length",
]
REQUIRED_1_100 = {f"chck_{i}M" for i in range(1, 101)}
REQUIRED_NOAOA = {f"chck_{i}M" for i in range(10, 101, 10)}


def load(run: pathlib.Path) -> dict[str, Any]:
    p = run / "scientific_metrics.json"
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf-8"))


def summarize(name: str, run: pathlib.Path, m: dict[str, Any]) -> dict[str, Any]:
    ckpts = [x.get("name") for x in m.get("saved_checkpoints", []) if isinstance(x, dict)]
    ckset = set(str(x) for x in ckpts)
    source_words = m.get("source_words_consumed") or {}
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
        "training_recipe": {"masking_curriculum": m.get("masking_curriculum"), "seq_length": m.get("seq_length"), "seed": m.get("seed")},
        "checkpoint_count": len(ckpts),
        "has_complete_1M_to_100M_ladder": ckset == REQUIRED_1_100,
        "missing_1M_to_100M": sorted(REQUIRED_1_100 - ckset, key=lambda x: int(x.split('_')[1].rstrip('M'))),
        "has_noaoa_10M_spaced_ladder": REQUIRED_NOAOA <= ckset,
        "source_words_consumed": source_words,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {name: load(path) for name, path in RUNS.items()}
    summaries = {name: summarize(name, RUNS[name], metrics[name]) for name in RUNS}
    identity = {}
    for key in IDENTITY_KEYS:
        vals = {name: metrics[name].get(key) for name in RUNS}
        enc = {json.dumps(v, sort_keys=True, ensure_ascii=False, default=str) for v in vals.values()}
        identity[key] = {"same": len(enc) == 1, "values": vals}
    differing = {k: v for k, v in identity.items() if not v["same"]}
    t = summaries["semantic_view_treatment"]
    c = summaries["original_packet_local"]
    loss_delta = float(t["loss_last"]) - float(c["loss_last"])
    payload = {
        "status": "SEMANTIC_VIEW_TRAINING_SNAPSHOT",
        "purpose": "Matched training state before downstream no-AoA trajectory interpretation; this is not BabyLM score evidence.",
        "targets": summaries,
        "identity_comparison": identity,
        "differing_identity_fields": differing,
        "all_core_training_fields_same": not differing,
        "loss_last_treatment_minus_packet_local": loss_delta,
        "changed_block_words_per_10_passes": {
            "semantic_view_treatment_simplewiki_semantic_view": (t["source_words_consumed"].get("simplewiki_semantic_view") if isinstance(t["source_words_consumed"], dict) else None),
            "packet_local_control_simplewiki_original_packet_local": (c["source_words_consumed"].get("simplewiki_original_packet_local") if isinstance(c["source_words_consumed"], dict) else None),
        },
        "downstream_result_expected": "experiments/archive/representation_and_objectives/data/semantic_view_noaoa_eval/semantic_view_packet_local_delta_summary.json",
        "current_observation": "No treatment-control score delta was available at the time of this record.",
    }
    out = OUT_DIR / "semantic_view_training_snapshot.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = []
    lines.append("# research semantic-view training snapshot while downstream evaluation is pending\n\n")
    lines.append("This records the completed treatment/control training pair from representation_and_objectives without reading the locked downstream evaluation directory. It is not a BabyLM score.\n\n")
    lines.append("| arm | exposure | steps | checkpoints | last loss | changed block |\n")
    lines.append("|---|---:|---:|---:|---:|---|\n")
    for name in ["semantic_view_treatment", "original_packet_local"]:
        s = summaries[name]
        sw = s["source_words_consumed"] if isinstance(s["source_words_consumed"], dict) else {}
        changed = sw.get("simplewiki_semantic_view") or sw.get("simplewiki_original_packet_local")
        lines.append(f"| {name} | {s['word_exposure']:,} | {s['actual_training_steps']:,} | {s['checkpoint_count']} | {float(s['loss_last']):.6f} | {changed:,} words over ten passes |\n")
    lines.append(f"\nCore training fields identical: **{not differing}**. Treatment last loss minus packet-local last loss: **{loss_delta:.6f}**; this is distribution/repetition evidence, not downstream competence.\n\n")
    lines.append("The decisive file remains `experiments/archive/representation_and_objectives/data/semantic_view_noaoa_eval/semantic_view_packet_local_delta_summary.json`. The completed result remained unavailable.\n\n")
    lines.append(f"JSON: `{out}`\n")
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out": str(out), "note": str(NOTE), "all_core_training_fields_same": not differing, "loss_delta": loss_delta}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
