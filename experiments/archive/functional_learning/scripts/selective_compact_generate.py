#!/usr/bin/env python3
"""research: selective semantic-preserving compact generation.

The research compact prompt forced a target ratio and produced semantic damage.  This
script prepares a safer prompt that treats the original source as authority, the
current rewrite as an inherited view, and allows KEEP_CURRENT when shortening would
remove teaching content or alter operator/order/role force.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

STUDY = Path("experiments/archive/functional_learning")
OUT_DIR = STUDY / "data/selective_compact_pilot"
PILOT_META = STUDY / "data/compact_pilot/pilot_enriched_metadata.jsonl"
MANIFEST = Path("experiments/archive/representation_and_objectives/data/qwen_internal_density_preflight/preferred_len90_r060_compaction_manifest.jsonl")
SELECTED_PAIRS = Path("experiments/archive/compact_experience/data/qwen_clean_aligned/selected_pairs.jsonl")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_pair_meta() -> Dict[str, Dict[str, Any]]:
    out = {}
    with SELECTED_PAIRS.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["pair_id"]] = r
    return out


def selective_prompt(rec: Dict[str, Any]) -> str:
    target = int(rec.get("target_compact_rewrite_words", rec.get("target_compact_words", 0)) or 0)
    # The target is deliberately a soft ceiling: semantic preservation beats savings.
    target_line = f"Aim for at most {target} whitespace-separated words only if fully faithful; otherwise use as many words as needed or KEEP_CURRENT." if target > 0 else "Shorten only if fully faithful; otherwise KEEP_CURRENT."
    return f"""Create a shorter faithful second view for BabyLM pretraining.

Original source is the authority. The current inherited rewrite is allowed as a starting point but may itself be imperfect.

{target_line}

Do NOT summarize away propositions. Preserve every load-bearing fact and relation: who did what to whom; speaker and addressee; quotation/question/command status; polarity and negation; modality, uncertainty, intention, prediction, future, counterfactual and conditional force; temporal order and event sequence; comparison/scalar force; all numbers with their units/owners/dates; and ordered attachments such as directions and landmarks.

If a shorter version would delete a substantive event/detail, change order or attachment, turn possibility/intention/condition into fact, collapse speakers, or otherwise alter meaning, output exactly:
KEEP_CURRENT

If faithful shortening is possible, output only the rewritten sentence, with no explanation.

ORIGINAL SOURCE:
{rec.get('original','')}

CURRENT INHERITED REWRITE:
{rec.get('current_rewrite','')}"""


def prepare(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = load_jsonl(Path(args.input))
    pair_meta = load_pair_meta()
    enriched = []
    for r in records:
        pid = r["pair_id"]
        meta = pair_meta.get(pid, {})
        # Pilot metadata already has these fields; manifest needs selected-pair enrich only for entities/numbers.
        rr = {**r, **{k: meta.get(k, r.get(k)) for k in ["entity_source", "entity_rewrite", "num_source", "num_rewrite", "content_overlap", "len_ratio"]}}
        enriched.append(rr)
    if args.max_records and args.max_records > 0:
        enriched = enriched[: int(args.max_records)]
    gen_rows = [{"prompt": selective_prompt(r), "pair_id": r["pair_id"]} for r in enriched]
    write_jsonl(out_dir / args.output_name, gen_rows)
    write_jsonl(out_dir / (Path(args.output_name).stem + "_metadata.jsonl"), enriched)
    summary = {
        "status": "SELECTIVE_PROMPTS_PREPARED",
        "n_prompts": len(gen_rows),
        "source_distribution": dict(Counter(r.get("source", "") for r in enriched)),
        "prompt_path": str(out_dir / args.output_name),
        "metadata_path": str(out_dir / (Path(args.output_name).stem + "_metadata.jsonl")),
        "policy": "semantic preservation dominates compression; KEEP_CURRENT allowed",
    }
    (out_dir / (Path(args.output_name).stem + "_summary.json")).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def clean_output(text: str) -> str:
    text = (text or "").strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1].strip()
    # If a chat model adds a label, strip only safe obvious wrappers.
    for p in ["Rewritten sentence:", "Faithful shortening:", "Output:"]:
        if text.lower().startswith(p.lower()):
            text = text[len(p):].strip()
    return text


def enrich(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    meta = load_jsonl(Path(args.metadata))
    gen = load_jsonl(Path(args.generation_output))
    joined = []
    keep = 0
    for i, g in enumerate(gen):
        idx = int(g.get("index", i))
        m = meta[idx] if idx < len(meta) else {}
        raw = clean_output(str(g.get("output", g.get("text", g.get("completion", "")))))
        keep_current = raw.strip().upper() == "KEEP_CURRENT"
        if keep_current:
            keep += 1
            compact = m.get("current_rewrite", "")
        else:
            compact = raw
        joined.append({
            "pair_id": m.get("pair_id", g.get("pair_id", f"idx_{idx}")),
            "source": m.get("source", ""),
            "example_id": m.get("example_id"),
            "original": m.get("original", ""),
            "current_rewrite": m.get("current_rewrite", ""),
            "compact_rewrite": compact,
            "selective_raw_output": raw,
            "selective_keep_current": keep_current,
            "target_compact_words": m.get("target_compact_rewrite_words", m.get("target_compact_words", 0)),
            "expected_saved_words": m.get("expected_saved_words", 0),
            "entity_source": m.get("entity_source", []),
            "num_source": m.get("num_source", []),
            "content_overlap": m.get("content_overlap", 0),
        })
    out_path = out_dir / args.joined_name
    write_jsonl(out_path, joined)
    summary = {
        "status": "SELECTIVE_GENERATION_JOINED",
        "n_outputs": len(gen),
        "n_joined": len(joined),
        "n_keep_current": keep,
        "n_candidate_shortening": len(joined) - keep,
        "joined_path": str(out_path),
    }
    (out_dir / (Path(args.joined_name).stem + "_summary.json")).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("prepare")
    p.add_argument("--input", default=str(PILOT_META))
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--output-name", default="selective_generation_input.jsonl")
    p.add_argument("--max-records", type=int, default=0)
    e = sub.add_parser("enrich")
    e.add_argument("--generation-output", required=True)
    e.add_argument("--metadata", default=str(OUT_DIR / "selective_generation_input_metadata.jsonl"))
    e.add_argument("--out-dir", default=str(OUT_DIR))
    e.add_argument("--joined-name", default="selective_generation_joined.jsonl")
    args = ap.parse_args()
    if args.cmd == "prepare":
        prepare(args)
    elif args.cmd == "enrich":
        enrich(args)
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
