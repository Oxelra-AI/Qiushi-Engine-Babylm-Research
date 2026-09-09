#!/usr/bin/env python3
"""Measure token-visibility behavior for density overlay pools.

Works for both official-only and clean-Qwen row-holdout overlay metadata. It reads
only training-corpus construction files, not BabyLM evaluation items.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import statistics
from typing import Any

from transformers import AutoTokenizer

ROOT = pathlib.Path("experiments/archive/frontier_consolidation")
DEFAULT_META = ROOT / "data/density_cleanqwen_overlay_rowholdout_highprecision/density_cleanqwen_rowholdout_overlay_metadata.json"
DEFAULT_OUT = ROOT / "data/density_overlay_token_visibility"
DEFAULT_NOTE = (ROOT.parents[2] / 'research/notes/frontier_consolidation/14_density_overlay_token_visibility.md')
TOKENIZERS = {
    "baseline16k": "experiments/archive/compact_experience/training/runs/qwen_clean_aligned_16k_seed43022/hf_model/chck_1M",
    "leader40k": "experiments/archive/compact_experience/data/shared_tokenizer/hf_tokenizer_40k_shared",
}
SEQ_LIMITS = [64, 128, 256]


def read_jsonl(path: pathlib.Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stat(vals: list[float]) -> dict[str, Any]:
    if not vals:
        return {"n": 0}
    xs = sorted(vals)
    return {
        "n": len(xs),
        "min": xs[0],
        "p01": xs[int(0.01 * (len(xs) - 1))],
        "p05": xs[int(0.05 * (len(xs) - 1))],
        "mean": statistics.fmean(xs),
        "median": statistics.median(xs),
        "p95": xs[int(0.95 * (len(xs) - 1))],
        "p99": xs[int(0.99 * (len(xs) - 1))],
        "max": xs[-1],
        "sum": sum(xs),
    }


def load_tokenizer(path: str):
    p = pathlib.Path(path)
    return AutoTokenizer.from_pretrained(str(p.resolve()) if p.exists() else path, use_fast=True, local_files_only=p.exists())


def token_lengths(tok, rows: list[dict[str, Any]]) -> list[int]:
    return [len(tok(str(r["text"]), add_special_tokens=False, truncation=False)["input_ids"]) for r in rows]


def compare(a: list[int], b: list[int]) -> dict[str, Any]:
    if len(a) != len(b):
        return {"same_n": False, "n_a": len(a), "n_b": len(b)}
    diffs = [x - y for x, y in zip(a, b)]
    absd = [abs(d) for d in diffs]
    return {
        "same_n": True,
        "diff_stats_a_minus_b": stat([float(d) for d in diffs]),
        "absdiff_stats": stat([float(d) for d in absd]),
        "fraction_equal": sum(1 for d in diffs if d == 0) / max(1, len(diffs)),
        "fraction_absdiff_le_4": sum(1 for d in absd if d <= 4) / max(1, len(absd)),
        "fraction_absdiff_le_16": sum(1 for d in absd if d <= 16) / max(1, len(absd)),
    }


def arm_names(fmeta: dict[str, Any]) -> tuple[str, str, str]:
    names = fmeta["arm_names"]
    control = names.get("official") or names.get("cleanqwen_lengthmatched")
    repeat = names["repeat"]
    view = names["view"]
    if not control:
        raise RuntimeError(f"no control arm in {names}")
    return control, repeat, view


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default=str(DEFAULT_META))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--note", default=str(DEFAULT_NOTE))
    args = ap.parse_args()

    meta_path = pathlib.Path(args.metadata)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    pools = meta["files"]["pools"]
    families = meta["families"]
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    row_measure: dict[str, Any] = {}
    for fam, fmeta in families.items():
        control, repeat, view = arm_names(fmeta)
        changed_rows = int(fmeta["changed_block_rows"])
        rows = {name: read_jsonl(pathlib.Path(pools[name]), limit=changed_rows) for name in [control, repeat, view]}
        word_lens = {name: [int(r.get("words") or len(str(r.get("text") or "").split())) for r in rs] for name, rs in rows.items()}
        row_measure[fam] = {
            "changed_rows": changed_rows,
            "arm_names": {"control": control, "repeat": repeat, "view": view},
            "word_length_sequence_equal": len({tuple(v) for v in word_lens.values()}) == 1,
            "word_length_stats": {name: stat([float(x) for x in lens]) for name, lens in word_lens.items()},
            "pool_sha256": {name: sha256_file(pathlib.Path(pools[name])) for name in [control, repeat, view]},
        }

    tokenizer_measure: dict[str, Any] = {}
    for tok_name, tok_path in TOKENIZERS.items():
        tok = load_tokenizer(tok_path)
        tok_payload = {"tokenizer_path": tok_path, "vocab_size": len(tok), "families": {}}
        for fam, fmeta in families.items():
            control, repeat, view = arm_names(fmeta)
            changed_rows = int(fmeta["changed_block_rows"])
            rows = {name: read_jsonl(pathlib.Path(pools[name]), limit=changed_rows) for name in [control, repeat, view]}
            lens = {name: token_lengths(tok, rs) for name, rs in rows.items()}
            tok_payload["families"][fam] = {
                "changed_rows": changed_rows,
                "token_length_stats": {name: stat([float(x) for x in lens[name]]) for name in [control, repeat, view]},
                "visibility_fraction_by_seq_limit": {
                    name: {str(L): sum(1 for x in lens[name] if x <= L) / max(1, len(lens[name])) for L in SEQ_LIMITS}
                    for name in [control, repeat, view]
                },
                "over_seq256_counts": {name: sum(1 for x in lens[name] if x > 256) for name in [control, repeat, view]},
                "token_count_comparisons": {
                    "view_minus_repeat": compare(lens[view], lens[repeat]),
                    "view_minus_control": compare(lens[view], lens[control]),
                    "repeat_minus_control": compare(lens[repeat], lens[control]),
                },
            }
        tokenizer_measure[tok_name] = tok_payload

    payload = {
        "status": "DENSITY_OVERLAY_TOKEN_VISIBILITY_MEASURED",
        "metadata": str(meta_path),
        "metadata_sha256": sha256_file(meta_path),
        "scientific_purpose": "Measure attention-window visibility and token-count shifts for the source-aligned density contrast on the chosen base.",
        "row_measure": row_measure,
        "tokenizers": tokenizer_measure,
        "interpretation": {
            "fixed_seq256": "Use fixed seq256 unless a separate pair-preserving short-window construction is made.",
            "view_repeat_token_shift": "Near view-vs-repeat has negligible mean shift; compact view-vs-repeat has a positive token-count shift under both candidate tokenizers and must be remembered when interpreting model-score deltas.",
        },
    }
    out_path = out_dir / "density_overlay_token_visibility.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["# research density overlay token visibility", ""]
    for tok_name, tp in tokenizer_measure.items():
        lines.append(f"## {tok_name}")
        for fam, ft in tp["families"].items():
            cmp_vr = ft["token_count_comparisons"]["view_minus_repeat"]
            mean_diff = cmp_vr.get("diff_stats_a_minus_b", {}).get("mean")
            frac_le4 = cmp_vr.get("fraction_absdiff_le_4")
            lines.append(f"- {fam}: over seq256 {ft['over_seq256_counts']}; view-repeat mean token diff {mean_diff}; abs diff <=4 fraction {frac_le4}.")
        lines.append("")
    lines.append(f"Full JSON: `{out_path}`")
    note_path = pathlib.Path(args.note)
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": payload["status"],
        "out": str(out_path),
        "over_seq256_counts": {
            tok: {fam: ft["over_seq256_counts"] for fam, ft in tp["families"].items()}
            for tok, tp in tokenizer_measure.items()
        },
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
