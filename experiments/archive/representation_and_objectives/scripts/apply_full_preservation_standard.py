#!/usr/bin/env python3
"""Apply the research FW preservation standard to full Qwen3.5 outputs.

This is a small reusable wrapper around `fw_preservation_standard.py`.
It keeps the semantic standard fixed while allowing the same checker to be run on
pilot or full generation outputs.  It also writes the exact usable pair set that
the preservation-aware materializer should use for both compact_view and
source_repeat arms.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import pathlib
import sys
import time
from typing import Any, Iterable

SCRIPT_DIR = _public_path('experiments/archive/representation_and_objectives/scripts')
sys.path.insert(0, str(SCRIPT_DIR))
import fw_preservation_standard as std  # noqa: E402

WORKSPACE = _public_path('experiments/archive/representation_and_objectives')
DEFAULT_FROZEN = _public_path('experiments/archive/representation_and_objectives/data/fw_mechanism_source_selection/fw_mechanism_frozen_sources.jsonl')
DEFAULT_PROMPTS = _public_path('experiments/archive/representation_and_objectives/data/fw_mechanism_source_selection/fw_mechanism_compact_prompts.jsonl')
DEFAULT_FULL_OUTPUTS = _public_path('experiments/archive/representation_and_objectives/training/runs/qwen35_fw_compact_full_26015/outputs.jsonl')
DEFAULT_OUT_DIR = _public_path('experiments/archive/representation_and_objectives/data/fw_preservation_full')
DEFAULT_NOTE = _public_path('research/notes/representation_and_objectives/fw_preservation_full_application.md')


def write_jsonl(path: pathlib.Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def output_for_index(outputs: list[dict[str, Any]], idx: int) -> dict[str, Any]:
    if not outputs:
        return {}
    # Use the stable output `index` when present; retain positional fallback.
    if idx < len(outputs) and int(outputs[idx].get("index", idx)) == idx:
        return outputs[idx]
    for row in outputs:
        try:
            if int(row.get("index")) == idx:
                return row
        except Exception:
            pass
    return outputs[idx] if idx < len(outputs) else {}


def build_new_rows(prompts: list[dict[str, Any]], outputs: list[dict[str, Any]], source_kind: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, p in enumerate(prompts):
        out = output_for_index(outputs, i)
        raw = str(out.get("output") or out.get("generated") or out.get("generated_text") or out.get("text") or "")
        rec = dict(p)
        rec["raw_output"] = raw
        rec["rewrite_text"] = std.clean_output(raw)
        rec["model_id"] = out.get("model")
        rec["generated_tokens"] = out.get("generated_tokens")
        rows.append(std.evaluate_pair(rec, source_kind))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", default=str(DEFAULT_FROZEN))
    ap.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    ap.add_argument("--outputs", default=str(DEFAULT_FULL_OUTPUTS))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--note", default=str(DEFAULT_NOTE))
    ap.add_argument("--tag", default="full_qwen35")
    ap.add_argument("--source-kind", default="new_qwen35_full")
    ap.add_argument("--allow-short-outputs", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    frozen_path = pathlib.Path(args.frozen)
    prompts_path = pathlib.Path(args.prompts)
    outputs_path = pathlib.Path(args.outputs)
    out_dir = pathlib.Path(args.out_dir)
    note_path = pathlib.Path(args.note)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not outputs_path.exists():
        raise FileNotFoundError(f"outputs file not found: {outputs_path}")
    frozen = std.read_jsonl(frozen_path)
    prompts = std.read_jsonl(prompts_path)
    outputs = std.read_jsonl(outputs_path)
    if len(outputs) < len(prompts) and not args.allow_short_outputs:
        raise RuntimeError(f"outputs {len(outputs)} shorter than prompts {len(prompts)}")

    existing_rows = std.build_old_rows(frozen)
    new_rows = build_new_rows(prompts, outputs, args.source_kind)
    combined = existing_rows + new_rows
    usable = [r for r in combined if r["usable_for_compact_pair"]]
    summary_core = std.summarize(combined)
    arm_summary = std.arm_set_summary(usable)

    prefix = args.tag
    existing_path = out_dir / f"{prefix}_existing_a02_rows.jsonl"
    new_path = out_dir / f"{prefix}_new_rows.jsonl"
    all_path = out_dir / f"{prefix}_all_preservation_rows.jsonl"
    usable_path = out_dir / f"{prefix}_usable_pairs_for_materializer.jsonl"
    summary_path = out_dir / f"{prefix}_standard_summary.json"
    samples_path = out_dir / f"{prefix}_review_samples.json"

    write_jsonl(existing_path, existing_rows)
    write_jsonl(new_path, new_rows)
    write_jsonl(all_path, combined)
    write_jsonl(usable_path, usable)

    # Reuse the same text standard but record the real full-run inputs.
    standard = {
        "name": "fw_compact_preservation_standard_v1_step100",
        "inherited_standard_file": str(_public_path('experiments/archive/representation_and_objectives/data/fw_preservation_standard/preservation_standard_v1.json')),
        "hard_failure_rules": [
            "empty or malformed non-sentence output",
            "rewrite/source whitespace-word ratio <0.33 or >1.25",
            "exact copy of source text",
            "any missing source number or any new number",
            "any missing real named entity after sentence-starter false positives are trimmed",
            "content recall <0.28 for sources with at least 14 words or content overlap <0.10 for sources with at least 10 words",
            "negative polarity in source lost, or explicit negative polarity introduced when source has none",
            "explicit modality/hedge in source lost, or introduced when source has none",
            "causal relation in source lost unless a light causal bridge preserves both sides; new strong causal relation introduced when source has none",
            "comparison/order direction in source not retained; new up/down/before/after direction introduced when source has none",
        ],
    }
    payload = {
        "status": "FULL_PRESERVATION_STANDARD_APPLIED",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tag": args.tag,
        "standard": standard,
        "inputs": {
            "frozen_sources": str(frozen_path),
            "frozen_sources_sha256": std.sha256_file(frozen_path),
            "prompts": str(prompts_path),
            "prompts_sha256": std.sha256_file(prompts_path),
            "outputs": str(outputs_path),
            "outputs_sha256": std.sha256_file(outputs_path),
            "n_prompts": len(prompts),
            "n_outputs": len(outputs),
        },
        "summary": summary_core,
        "arm_source_set_comparison": arm_summary,
        "files": {
            "existing_rows": str(existing_path),
            "new_rows": str(new_path),
            "all_rows": str(all_path),
            "usable_pairs": str(usable_path),
            "summary": str(summary_path),
            "review_samples": str(samples_path),
            "note": str(note_path),
        },
        "elapsed_sec": round(time.time() - t0, 2),
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    rejected_new = [r for r in new_rows if not r["usable_for_compact_pair"]]
    usable_new = [r for r in new_rows if r["usable_for_compact_pair"]]
    samples = {
        "new_rejected_first50": rejected_new[:50],
        "new_usable_low_content_first50": sorted(usable_new, key=lambda r: (r["content_recall"], r["content_overlap"]))[:50],
        "all_hard_reason_prefixes": summary_core["hard_reason_prefixes"][:40],
    }
    samples_path.write_text(json.dumps(samples, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    overall = summary_core["overall"]
    by_kind = summary_core["by_source_kind"]
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(
        "# research full preservation-standard application\n\n"
        "This applies the frozen research semantic-preservation standard to the old A02 Qwen3.5 rewrites and the new Qwen3.5 outputs. "
        "The usable pair set is the only input that the preservation-aware arm materializer should use.\n\n"
        f"- Outputs: `{outputs_path}` ({len(outputs):,} rows for {len(prompts):,} prompts)\n"
        f"- Overall usable: {overall['usable']:,}/{overall['n']:,} ({overall['usable_rate']:.3f})\n"
        f"- Usable pair words: {overall['usable_pair_words']:,}\n"
        f"- compact_view/source_repeat pair totals match on retained source set: {arm_summary['word_totals_match']}\n\n"
        "## By source kind\n\n"
        + "".join(
            f"- {k}: {v['usable']:,}/{v['n']:,} usable ({v['usable_rate']:.3f}), pair words {v['usable_pair_words']:,}\n"
            for k, v in by_kind.items()
        )
        + "\n## Files\n\n"
        f"- Summary: `{summary_path}`\n"
        f"- Usable pair set: `{usable_path}`\n"
        f"- Review samples: `{samples_path}`\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "status": payload["status"],
        "tag": args.tag,
        "n_prompts": len(prompts),
        "n_outputs": len(outputs),
        "overall_usable": overall["usable"],
        "overall_n": overall["n"],
        "overall_rate": overall["usable_rate"],
        "usable_pair_words": overall["usable_pair_words"],
        "by_source_kind": {k: {"usable": v["usable"], "n": v["n"], "rate": v["usable_rate"], "pair_words": v["usable_pair_words"]} for k, v in by_kind.items()},
        "usable_pairs": str(usable_path),
        "summary": str(summary_path),
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
