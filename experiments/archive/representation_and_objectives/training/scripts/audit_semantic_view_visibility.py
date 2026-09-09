#!/usr/bin/env python3
"""Audit actual seq256 visibility of generated semantic views.

The protected trainer tokenizes each JSONL row with add_special_tokens=False,
truncation=True, max_length=256.  A row can preserve source+view words in the
corpus ledger yet hide appended generated views from the MLM objective if source
+ earlier text exceeds 256 baseline16k tokens.  This script reconstructs each
semantic-view packet, computes token spans of each generated view under the same
tokenizer interface, and records how much of the intervention is actually visible
before H100 training is launched.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import statistics
from typing import Any, Iterable

from transformers import AutoTokenizer

ROOT = pathlib.Path("experiments/archive/representation_and_objectives")
DEFAULT_DIR = ROOT / "training/data/semantic_view/full_contrast"
TOKENIZER = pathlib.Path("experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model")


def norm(text: str) -> str:
    return " ".join(str(text).replace("\u00a0", " ").split())


def stats(vals: Iterable[int | float]) -> dict[str, Any]:
    xs = list(vals)
    if not xs:
        return {"n": 0}
    ys = sorted(xs)
    def q(p: float):
        return ys[min(len(ys)-1, max(0, round((len(ys)-1)*p)))]
    return {"n": len(xs), "min": min(xs), "p05": q(0.05), "mean": round(statistics.mean(xs), 4), "median": statistics.median(xs), "p95": q(0.95), "p99": q(0.99), "max": max(xs), "sum": sum(xs)}


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows=[]
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_prompts(path: pathlib.Path) -> dict[str, dict[str, Any]]:
    out={}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o=json.loads(line)
            src=norm(o.get("source_text", ""))
            # Must match materializer sha256_text; import locally to avoid path surprises.
            import hashlib
            key=hashlib.sha256(" ".join(src.split()).encode("utf-8")).hexdigest()
            out[key]=o | {"source_text": src, "source_key": key}
    return out


def tok_len(tok, text: str) -> int:
    if not text:
        return 0
    return len(tok(text, add_special_tokens=False, truncation=False)["input_ids"])


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", default=str(DEFAULT_DIR))
    ap.add_argument("--tokenizer", default=str(TOKENIZER))
    ap.add_argument("--max-len", type=int, default=256)
    args=ap.parse_args()
    d=pathlib.Path(args.corpus_dir)
    meta=json.loads((d/"semantic_view_materialization_metadata.json").read_text(encoding="utf-8"))
    prompts=load_prompts(pathlib.Path(meta["prompts"]))
    accepted={str(o["prompt_id"]): o for o in read_jsonl(d/"accepted_views.jsonl")}
    packet_meta=read_jsonl(d/"semantic_packet_rows_meta.jsonl")
    treatment=read_jsonl(d/"semantic_view_treatment_10M.jsonl")[:len(packet_meta)]
    tok=AutoTokenizer.from_pretrained(str(pathlib.Path(args.tokenizer)), use_fast=True)

    row_reports=[]
    view_reports=[]
    bad_recon=[]
    for i, (pm, tr) in enumerate(zip(packet_meta, treatment)):
        key=str(pm["source_key"])
        p=prompts.get(key)
        if p is None:
            raise RuntimeError(f"missing prompt for row {i} key={key}")
        source=norm(p["source_text"])
        views=[]
        for pid in pm.get("prompt_ids") or []:
            v=accepted.get(str(pid))
            if v is None:
                raise RuntimeError(f"missing accepted view {pid} row {i}")
            views.append(v)
        views=sorted(views, key=lambda v: {"simplification":0,"paraphrase":1}.get(v.get("typ"),9))
        parts=[source]+[norm(v.get("output", "")) for v in views]
        actual=norm(tr["text"])
        expected=norm(" ".join(parts))
        if actual != expected and len(bad_recon) < 20:
            bad_recon.append({"row": i, "expected_prefix": expected[:220], "actual_prefix": actual[:220]})
        source_tokens=tok_len(tok, source)
        full_tokens=tok_len(tok, expected)
        row_visible_tokens=min(full_tokens, args.max_len)
        all_view_tokens=0
        visible_view_tokens=0
        full_views=0
        partial_views=0
        none_views=0
        view_detail=[]
        prefix=source
        prev_len=tok_len(tok, prefix)
        for v in views:
            view_text=norm(v.get("output", ""))
            new_prefix=norm(prefix + " " + view_text)
            end_len=tok_len(tok, new_prefix)
            view_tokens=max(0, end_len - prev_len)
            visible=max(0, min(end_len, args.max_len) - min(prev_len, args.max_len))
            visible=min(visible, view_tokens)
            fully=visible == view_tokens and view_tokens > 0
            partial=0 < visible < view_tokens
            none=visible == 0
            all_view_tokens += view_tokens
            visible_view_tokens += visible
            full_views += int(fully)
            partial_views += int(partial)
            none_views += int(none)
            vd={
                "row_index": i,
                "example_id": tr.get("example_id"),
                "source_article": pm.get("source_article"),
                "prompt_id": v.get("prompt_id"),
                "typ": v.get("typ"),
                "source_tokens_before_view": prev_len,
                "view_end_tokens": end_len,
                "view_tokens": view_tokens,
                "visible_tokens": visible,
                "visible_fraction": None if view_tokens == 0 else round(visible / view_tokens, 6),
                "fully_visible": fully,
                "partially_visible": partial,
                "not_visible": none,
                "source_words": p.get("source_words"),
                "view_words": v.get("output_words"),
                "row_words": tr.get("words"),
                "row_full_tokens": full_tokens,
            }
            view_reports.append(vd)
            view_detail.append(vd)
            prefix=new_prefix
            prev_len=end_len
        row_reports.append({
            "row_index": i,
            "example_id": tr.get("example_id"),
            "source_article": pm.get("source_article"),
            "source_words": p.get("source_words"),
            "row_words": tr.get("words"),
            "source_tokens": source_tokens,
            "row_full_tokens": full_tokens,
            "row_visible_tokens": row_visible_tokens,
            "views": len(views),
            "all_view_tokens": all_view_tokens,
            "visible_view_tokens": visible_view_tokens,
            "view_visible_fraction": None if all_view_tokens == 0 else round(visible_view_tokens / all_view_tokens, 6),
            "all_views_fully_visible": full_views == len(views),
            "any_view_partially_visible": partial_views > 0,
            "any_view_not_visible": none_views > 0,
            "full_views": full_views,
            "partial_views": partial_views,
            "none_views": none_views,
        })

    by_type={}
    for typ in ["simplification", "paraphrase"]:
        vs=[v for v in view_reports if v.get("typ")==typ]
        by_type[typ]={
            "views": len(vs),
            "fully_visible": sum(1 for v in vs if v["fully_visible"]),
            "partially_visible": sum(1 for v in vs if v["partially_visible"]),
            "not_visible": sum(1 for v in vs if v["not_visible"]),
            "visible_token_fraction": round(sum(v["visible_tokens"] for v in vs)/max(1,sum(v["view_tokens"] for v in vs)), 6),
            "view_token_stats": stats([v["view_tokens"] for v in vs]),
            "prefix_token_before_view_stats": stats([v["source_tokens_before_view"] for v in vs]),
        }
    rows_with_hidden=[r for r in row_reports if not r["all_views_fully_visible"]]
    rows_with_no_visible=[r for r in row_reports if r["visible_view_tokens"] == 0]
    payload={
        "status": "SEMANTIC_VIEW_VISIBILITY_AUDIT",
        "corpus_dir": str(d),
        "tokenizer": str(pathlib.Path(args.tokenizer)),
        "max_len_matches_trainer": args.max_len,
        "semantic_packet_rows": len(row_reports),
        "accepted_views": len(view_reports),
        "bad_reconstruction_count_sampled_until20": len(bad_recon),
        "bad_reconstruction_samples": bad_recon,
        "row_full_token_stats": stats([r["row_full_tokens"] for r in row_reports]),
        "source_token_stats": stats([r["source_tokens"] for r in row_reports]),
        "row_view_token_stats": stats([r["all_view_tokens"] for r in row_reports]),
        "view_visibility_by_type": by_type,
        "views_fully_visible": sum(1 for v in view_reports if v["fully_visible"]),
        "views_partially_visible": sum(1 for v in view_reports if v["partially_visible"]),
        "views_not_visible": sum(1 for v in view_reports if v["not_visible"]),
        "view_visible_token_fraction_overall": round(sum(v["visible_tokens"] for v in view_reports)/max(1,sum(v["view_tokens"] for v in view_reports)), 6),
        "rows_all_views_fully_visible": sum(1 for r in row_reports if r["all_views_fully_visible"]),
        "rows_with_any_hidden_or_partial_view": len(rows_with_hidden),
        "rows_with_zero_visible_view_tokens": len(rows_with_no_visible),
        "row_view_visible_fraction_stats": stats([r["view_visible_fraction"] for r in row_reports if r["view_visible_fraction"] is not None]),
        "rows_with_hidden_or_partial_view_examples": rows_with_hidden[:30],
        "views_hidden_or_partial_examples": [v for v in view_reports if not v["fully_visible"]][:40],
    }
    out=d/"semantic_view_seq256_visibility_audit.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    md=d/"semantic_view_seq256_visibility_audit.md"
    lines=["# research semantic-view seq256 visibility audit\n\n"]
    lines.append(f"Trainer-equivalent tokenizer max length: {args.max_len} baseline16k tokens, add_special_tokens=False.\n\n")
    lines.append(f"Semantic packet rows: {len(row_reports)}; accepted generated views: {len(view_reports)}.\n\n")
    lines.append(f"Views fully visible: {payload['views_fully_visible']} / {len(view_reports)}; partially visible: {payload['views_partially_visible']}; not visible: {payload['views_not_visible']}.\n\n")
    lines.append(f"Overall generated-view token visibility: {payload['view_visible_token_fraction_overall']:.4f}. Rows with any hidden/partial view: {len(rows_with_hidden)}; rows with zero visible view tokens: {len(rows_with_no_visible)}.\n\n")
    lines.append("By type:\n")
    for typ, s in by_type.items():
        lines.append(f"- {typ}: fully {s['fully_visible']}/{s['views']}, partial {s['partially_visible']}, none {s['not_visible']}, visible token fraction {s['visible_token_fraction']:.4f}.\n")
    lines.append(f"\nJSON: `{out}`\n")
    md.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "semantic_packet_rows": len(row_reports),
        "accepted_views": len(view_reports),
        "views_fully_visible": payload["views_fully_visible"],
        "views_partially_visible": payload["views_partially_visible"],
        "views_not_visible": payload["views_not_visible"],
        "view_visible_token_fraction_overall": payload["view_visible_token_fraction_overall"],
        "rows_with_any_hidden_or_partial_view": len(rows_with_hidden),
        "rows_with_zero_visible_view_tokens": len(rows_with_no_visible),
        "out_json": str(out),
        "out_md": str(md),
    }, indent=2))


if __name__ == "__main__":
    main()
