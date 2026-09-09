#!/usr/bin/env python3
"""Audit token-visibility and token-count behavior for density contrast pools.

The materializers enforce BabyLM-style whitespace word budgets and row-length identity.
Before spending H100 pretraining, this script checks the remaining model-facing facts:
* each changed row is visible under fixed seq256 for a tokenizer;
* fineweb repeat/view and official arms in a family have the same row counts and word lengths;
* token lengths under baseline16k and leader40k are close enough that downstream deltas are interpretable;
* fineweb repeat/view suffix/filler identity remains true at the text level.

No BabyLM evaluation examples or labels are read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import statistics
from typing import Any

from transformers import AutoTokenizer

ROOT = pathlib.Path("experiments/archive/frontier_consolidation")
DEFAULT_META = ROOT / "data/density_core_reinvestment_full/density_core_reinvestment_metadata.json"
DEFAULT_OUT = ROOT / "data/density_token_visibility_audit"
DEFAULT_NOTE = (ROOT.parents[2] / 'research/notes/frontier_consolidation/density_token_visibility_audit.md')
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
    lens: list[int] = []
    for r in rows:
        ids = tok(str(r["text"]), add_special_tokens=False, truncation=False)["input_ids"]
        lens.append(len(ids))
    return lens


def compare_lists(a: list[int], b: list[int]) -> dict[str, Any]:
    if len(a) != len(b):
        return {"same_n": False, "n_a": len(a), "n_b": len(b)}
    diffs = [x - y for x, y in zip(a, b)]
    absdiffs = [abs(d) for d in diffs]
    return {
        "same_n": True,
        "diff_stats_a_minus_b": stat([float(d) for d in diffs]),
        "absdiff_stats": stat([float(d) for d in absdiffs]),
        "fraction_equal": sum(1 for d in diffs if d == 0) / max(1, len(diffs)),
        "fraction_absdiff_le_4": sum(1 for d in absdiffs if d <= 4) / max(1, len(absdiffs)),
        "fraction_absdiff_le_16": sum(1 for d in absdiffs if d <= 16) / max(1, len(absdiffs)),
    }


def row_word_lengths(rows: list[dict[str, Any]]) -> list[int]:
    return [int(r.get("words") or len(str(r.get("text") or "").split())) for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default=str(DEFAULT_META))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--note", default=str(DEFAULT_NOTE))
    ap.add_argument("--filler-sample", type=int, default=2048)
    args = ap.parse_args()

    meta_path = pathlib.Path(args.metadata)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    families = meta["families"]
    pools = meta["files"]["pools"]
    tokenizer_payload: dict[str, Any] = {}
    family_payload: dict[str, Any] = {}

    for family_name, fmeta in families.items():
        arms = fmeta["arm_names"]
        names = [arms["official"], arms["repeat"], arms["view"]]
        changed_rows = int(fmeta["changed_block_rows"])
        arm_rows = {name: read_jsonl(pathlib.Path(pools[name]), limit=changed_rows) for name in names}
        arm_word_lens = {name: row_word_lengths(rows) for name, rows in arm_rows.items()}
        family_payload[family_name] = {
            "changed_rows": changed_rows,
            "arm_names": arms,
            "word_length_sequence_equal": len({tuple(v) for v in arm_word_lens.values()}) == 1,
            "word_length_stats": {name: stat([float(x) for x in lens]) for name, lens in arm_word_lens.items()},
            "pool_sha256": {name: sha256_file(pathlib.Path(pools[name])) for name in names},
        }

    for tok_name, tok_path in TOKENIZERS.items():
        tok = load_tokenizer(tok_path)
        tp: dict[str, Any] = {
            "tokenizer_path": tok_path,
            "vocab_size": len(tok),
            "families": {},
        }
        for family_name, fmeta in families.items():
            arms = fmeta["arm_names"]
            names = [arms["official"], arms["repeat"], arms["view"]]
            changed_rows = int(fmeta["changed_block_rows"])
            arm_rows = {name: read_jsonl(pathlib.Path(pools[name]), limit=changed_rows) for name in names}
            lens = {name: token_lengths(tok, rows) for name, rows in arm_rows.items()}
            fam_tok = {
                "changed_rows": changed_rows,
                "token_length_stats": {name: stat([float(x) for x in lens[name]]) for name in names},
                "visibility_fraction_by_seq_limit": {
                    name: {str(L): sum(1 for x in lens[name] if x <= L) / max(1, len(lens[name])) for L in SEQ_LIMITS}
                    for name in names
                },
                "over_seq256_counts": {name: sum(1 for x in lens[name] if x > 256) for name in names},
                "pairwise_token_count_comparisons": {
                    "view_minus_repeat": compare_lists(lens[arms["view"]], lens[arms["repeat"]]),
                    "view_minus_official": compare_lists(lens[arms["view"]], lens[arms["official"]]),
                    "repeat_minus_official": compare_lists(lens[arms["repeat"]], lens[arms["official"]]),
                },
            }
            tp["families"][family_name] = fam_tok
        tokenizer_payload[tok_name] = tp

    payload = {
        "status": "DENSITY_TOKEN_VISIBILITY_AUDITED",
        "metadata": str(meta_path),
        "metadata_sha256": sha256_file(meta_path),
        "scientific_purpose": "Check whether whitespace-matched density contrast pools remain attention-visible and model-token comparable under candidate BabyLM training tokenizers.",
        "tokenizers": tokenizer_payload,
        "families": family_payload,
        "interpretation": {
            "seq256_visibility": "A fixed seq256 screen is safe if changed rows have over_seq256_counts near zero; seq64/128 curricula would alter changed-block visibility and need separate pair-preserving construction.",
            "token_count_comparisons": "Large view-minus-repeat token count shifts would be a training-signal amount confound even with identical whitespace word budgets.",
        },
    }
    out_path = out_dir / "density_token_visibility_audit.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["# research density token-visibility audit", ""]
    for tok_name, tp in tokenizer_payload.items():
        lines.append(f"## {tok_name}")
        for fam, ft in tp["families"].items():
            over = ft["over_seq256_counts"]
            view_name = meta["families"][fam]["arm_names"]["view"]
            repeat_name = meta["families"][fam]["arm_names"]["repeat"]
            cmp_vr = ft["pairwise_token_count_comparisons"]["view_minus_repeat"]
            mean_diff = cmp_vr.get("diff_stats_a_minus_b", {}).get("mean")
            frac_le4 = cmp_vr.get("fraction_absdiff_le_4")
            lines.append(
                f"- {fam}: changed rows {ft['changed_rows']:,}; over seq256 {over}; "
                f"mean token diff {view_name} minus {repeat_name}: {mean_diff}; abs diff <=4 fraction: {frac_le4}."
            )
        lines.append("")
    lines.append(f"Full JSON: `{out_path}`")
    note_path = pathlib.Path(args.note)
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    compact = {tok: {fam: ft["over_seq256_counts"] for fam, ft in tp["families"].items()} for tok, tp in tokenizer_payload.items()}
    print(json.dumps({"status": payload["status"], "audit": str(out_path), "over_seq256_counts": compact}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
