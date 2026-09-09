#!/usr/bin/env python3
"""research: construction check for the research context-vs-isolation coordinate.

For each saved sentence span, compare the tokenized target span inside the full row
(row_context) to the tokenized isolated sentence (isolation).  The research scorer
uses the same mask seed in both modes; this check verifies that this implies the
same target-coordinate mask subset and target-token labels, and that the coordinate
is not a degenerate whole-span mask.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import time
from collections import Counter
from typing import Any

ROOT = pathlib.Path(".").resolve()
WS = ROOT / "experiments/archive/relation_learning"
DEFAULT_RECORDS = WS / "data/context_isolation_screen_all100M/sentence_records.jsonl"
DEFAULT_AXIS = WS / "data/strict_complement_ngram_axis/strict_complement_ngram_axis_3000_rows.jsonl"
DEFAULT_OUT = WS / "data/context_isolation_mask_check"
MAX_LEN = 256
MASK_PROB = 0.15
TOKENIZERS = {
    "compact_experience_off43022": ROOT / "experiments/archive/compact_experience/training/runs/official_lengthmatched_16k_seed43022/hf_model/chck_100M",
    "representation_frontier_studies_crv_c43022": ROOT / "experiments/archive/frontier_consolidation/training/runs/full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43022/hf_model/chck_100M",
    "adapter_base43022": ROOT / "experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M",
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path | str) -> str:
    p = pathlib.Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def configure_cache(out_dir: pathlib.Path) -> None:
    base = out_dir / "hf_cache"
    for k, p in {
        "HF_HOME": base / "hf_home",
        "HF_HUB_CACHE": base / "hf_home" / "hub",
        "HUGGINGFACE_HUB_CACHE": base / "hf_home" / "hub",
        "TRANSFORMERS_CACHE": base / "transformers",
        "HF_MODULES_CACHE": base / "modules",
        "HF_DATASETS_CACHE": base / "datasets",
        "TMPDIR": base / "tmp",
    }.items():
        p.mkdir(parents=True, exist_ok=True)
        os.environ[k] = str(p.resolve())
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def special_id_set(tok) -> set[int]:
    out = set(int(x) for x in tok.all_special_ids if x is not None)
    for attr in ["bos_token_id", "eos_token_id", "pad_token_id", "cls_token_id", "sep_token_id", "mask_token_id"]:
        x = getattr(tok, attr, None)
        if x is not None:
            out.add(int(x))
    return out


def target_ids(tok, rec: dict[str, Any], mode: str) -> tuple[list[int], bool, list[tuple[int, int]]]:
    if mode == "row_context":
        text = str(rec["row_text"])
        c0, c1 = int(rec["char_start"]), int(rec["char_end"])
    elif mode == "isolation":
        text = str(rec["sentence_text"])
        c0, c1 = 0, len(text)
    else:
        raise ValueError(mode)
    enc = tok(text, max_length=MAX_LEN, truncation=True, padding=False, return_offsets_mapping=True)
    ids = [int(x) for x in enc["input_ids"]]
    offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
    special = special_id_set(tok)
    out: list[int] = []
    target_offsets: list[tuple[int, int]] = []
    for pos, (a, b) in enumerate(offsets):
        tid = int(ids[pos])
        if tid in special or (a == 0 and b == 0):
            continue
        if b > c0 and a < c1:
            out.append(tid)
            target_offsets.append((a, b))
    max_seen_end = max([int(b) for (a, b) in offsets if int(b) > 0], default=0)
    truncated = bool(mode == "row_context" and c1 > max_seen_end)
    return out, truncated, target_offsets


def choose_mask_indices(n_target: int, seed: int) -> list[int]:
    import numpy as np
    rng = np.random.RandomState(seed)
    mask = rng.random(n_target) < MASK_PROB
    if n_target > 0 and not bool(mask.any()):
        mask[0] = True
    # research consumes replace_decisions/random_draws after this, but the selected
    # target-coordinate subset is determined before those draws affect token values.
    return [i for i, m in enumerate(mask.tolist()) if bool(m)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=pathlib.Path, default=DEFAULT_RECORDS)
    ap.add_argument("--axis", type=pathlib.Path, default=DEFAULT_AXIS)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--max-examples", type=int, default=8)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    configure_cache(args.out_dir)

    from transformers import AutoTokenizer

    recs = read_jsonl(args.records)
    axis_rows = {int(r.get("axis_row_index", i)): r for i, r in enumerate(read_jsonl(args.axis))}
    for r in recs:
        if "row_text" not in r:
            axis = axis_rows.get(int(r.get("axis_row_index", -1)))
            if axis is None:
                raise KeyError(f"row_text missing and axis row not found for {r.get('sentence_uid')}")
            r["row_text"] = str(axis["text"])
    summary: dict[str, Any] = {
        "status": "CONTEXT_ISOLATION_MASK_CHECK",
        "created_utc": now(),
        "records": rel(args.records),
        "axis": rel(args.axis),
        "n_sentence_records": len(recs),
        "max_len": MAX_LEN,
        "mask_prob": MASK_PROB,
        "tokenizers": {},
    }
    all_examples: list[dict[str, Any]] = []
    for name, tok_path in TOKENIZERS.items():
        tok = AutoTokenizer.from_pretrained(str(tok_path), use_fast=True, local_files_only=True)
        counts = Counter()
        examples: list[dict[str, Any]] = []
        n_target_values: list[int] = []
        n_mask_values: list[int] = []
        for r in recs:
            row_ids, row_trunc, row_offsets = target_ids(tok, r, "row_context")
            iso_ids, iso_trunc, iso_offsets = target_ids(tok, r, "isolation")
            counts["records"] += 1
            if row_trunc:
                counts["row_target_truncated"] += 1
            if iso_trunc:
                counts["isolation_target_truncated"] += 1
            if len(row_ids) == len(iso_ids):
                counts["same_n_target_tokens"] += 1
            else:
                counts["different_n_target_tokens"] += 1
            if row_ids == iso_ids:
                counts["same_target_token_sequence"] += 1
            else:
                counts["different_target_token_sequence"] += 1
            mi_row = choose_mask_indices(len(row_ids), int(r["mask_seed"]))
            mi_iso = choose_mask_indices(len(iso_ids), int(r["mask_seed"]))
            masked_row = [row_ids[i] for i in mi_row]
            masked_iso = [iso_ids[i] for i in mi_iso]
            if mi_row == mi_iso:
                counts["same_target_coordinate_mask_indices"] += 1
            else:
                counts["different_target_coordinate_mask_indices"] += 1
            if masked_row == masked_iso:
                counts["same_masked_label_token_sequence"] += 1
            else:
                counts["different_masked_label_token_sequence"] += 1
            if len(mi_row) == len(row_ids) and row_ids:
                counts["row_context_whole_target_masked"] += 1
            if len(mi_iso) == len(iso_ids) and iso_ids:
                counts["isolation_whole_target_masked"] += 1
            n_target_values.append(len(row_ids))
            n_mask_values.append(len(mi_row))
            if (row_ids != iso_ids or mi_row != mi_iso or masked_row != masked_iso or row_trunc) and len(examples) < args.max_examples:
                ex = {
                    "tokenizer": name,
                    "sentence_uid": r.get("sentence_uid"),
                    "axis_row_index": r.get("axis_row_index"),
                    "source": r.get("source"),
                    "sentence_text": r.get("sentence_text"),
                    "row_n_target": len(row_ids),
                    "iso_n_target": len(iso_ids),
                    "row_target_token_prefix": tok.convert_ids_to_tokens(row_ids[:20]),
                    "iso_target_token_prefix": tok.convert_ids_to_tokens(iso_ids[:20]),
                    "row_mask_indices": mi_row[:20],
                    "iso_mask_indices": mi_iso[:20],
                    "row_masked_label_tokens": tok.convert_ids_to_tokens(masked_row[:20]),
                    "iso_masked_label_tokens": tok.convert_ids_to_tokens(masked_iso[:20]),
                    "row_truncated": row_trunc,
                    "row_offsets_prefix": row_offsets[:20],
                    "iso_offsets_prefix": iso_offsets[:20],
                }
                examples.append(ex)
                all_examples.append(ex)
        def avg(xs: list[int]) -> float:
            return float(sum(xs) / len(xs)) if xs else float("nan")
        summary["tokenizers"][name] = {
            "tokenizer_path": rel(tok_path),
            "counts": dict(counts),
            "same_target_token_sequence_fraction": counts["same_target_token_sequence"] / max(1, counts["records"]),
            "same_masked_label_token_sequence_fraction": counts["same_masked_label_token_sequence"] / max(1, counts["records"]),
            "row_context_whole_target_masked_fraction": counts["row_context_whole_target_masked"] / max(1, counts["records"]),
            "isolation_whole_target_masked_fraction": counts["isolation_whole_target_masked"] / max(1, counts["records"]),
            "mean_target_tokens": avg(n_target_values),
            "mean_masked_tokens": avg(n_mask_values),
            "examples": examples,
        }
    out_json = args.out_dir / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research context-isolation masking construction check",
        "",
        f"Records: `{rel(args.records)}`; n={len(recs)}; max_len={MAX_LEN}; mask_prob={MASK_PROB}.",
        "",
        "| tokenizer | same target seq | same masked labels | row whole-span masked | target trunc | mean target tokens | mean masked |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, rec in summary["tokenizers"].items():
        c = rec["counts"]
        lines.append(
            f"| {name} | {rec['same_target_token_sequence_fraction']:.4f} | {rec['same_masked_label_token_sequence_fraction']:.4f} | "
            f"{rec['row_context_whole_target_masked_fraction']:.4f} | {c.get('row_target_truncated', 0)} | {rec['mean_target_tokens']:.2f} | {rec['mean_masked_tokens']:.2f} |"
        )
    lines += ["", f"Summary JSON: `{rel(out_json)}`"]
    (args.out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "summary": rel(out_json), "tokenizers": {k: v["counts"] for k, v in summary["tokenizers"].items()}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
