#!/usr/bin/env python3
"""research: fixed-tokenizer geometry audit for the dose-response pair pools.

The MAX dose arms hold the research tokenizer fixed to avoid refitting, but the
increment contains new FineWeb/WAR text. Before interpreting a dose curve, measure
whether old 1x, medium-unused increment, and WAR increment have materially
different tokenization burden under that fixed tokenizer.
"""
from __future__ import annotations

import collections
import json
import pathlib
import statistics
from typing import Any

from transformers import AutoTokenizer

ROOT = pathlib.Path("experiments/archive/frontier_consolidation")
TOKENIZER_DIR = ROOT / "data/compliant_tokenizer"
OLD_SELECTED = ROOT / "data/density_core_reinvestment_medium_riskhard/selected_compact_reinvest_pairs.jsonl"
MATCHED_MAX = ROOT / "data/dose_distribution_select/selected_matched_max_pairs.jsonl"
OUT_DIR = ROOT / "data/dose_tokenization_audit"


def wc(text: str) -> int:
    return len((text or "").split())


def norm_key(d: dict[str, Any]) -> str:
    k = str(d.get("key") or "").strip()
    if k.startswith("sid:") and "|doc:" in k:
        return k
    sid = str(d.get("sentence_id") or "").strip()
    doc = str(d.get("doc_id") or "").strip()
    if sid or doc:
        return f"sid:{sid}|doc:{doc}"
    return k or str(d.get("pair_id") or d.get("prompt_id") or "")


def read_pairs(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            src = " ".join(str(d.get("source_text") or "").split())
            rew = " ".join(str(d.get("rewrite_text") or "").split())
            rows.append({
                "key": norm_key(d),
                "origin": str(d.get("origin") or d.get("regime") or "old_selected_1x_medium"),
                "source_text": src,
                "rewrite_text": rew,
                "source_words": wc(src),
                "rewrite_words": wc(rew),
                "pair_words": wc(src) + wc(rew),
            })
    return rows


def stat(vals: list[float]) -> dict[str, Any]:
    if not vals:
        return {"n": 0}
    xs = sorted(vals)
    return {
        "n": len(vals),
        "mean": statistics.fmean(vals),
        "median": statistics.median(xs),
        "q10": xs[int(0.10*(len(xs)-1))],
        "q25": xs[int(0.25*(len(xs)-1))],
        "q75": xs[int(0.75*(len(xs)-1))],
        "q90": xs[int(0.90*(len(xs)-1))],
        "min": xs[0],
        "max": xs[-1],
        "sum": sum(vals),
    }


def encode_len(tok, text: str) -> tuple[int, int]:
    ids = tok(text, add_special_tokens=False)["input_ids"]
    unk = sum(1 for i in ids if tok.unk_token_id is not None and i == tok.unk_token_id)
    return len(ids), unk


def summarize_group(name: str, rows: list[dict[str, Any]], tok) -> dict[str, Any]:
    records = []
    token_counter: collections.Counter[int] = collections.Counter()
    for r in rows:
        st, su = encode_len(tok, r["source_text"])
        rt, ru = encode_len(tok, r["rewrite_text"])
        pt = st + rt
        token_counter.update(tok(r["source_text"] + " " + r["rewrite_text"], add_special_tokens=False)["input_ids"])
        records.append({
            "source_tokens": st,
            "rewrite_tokens": rt,
            "pair_tokens": pt,
            "source_tokens_per_word": st / max(1, r["source_words"]),
            "rewrite_tokens_per_word": rt / max(1, r["rewrite_words"]),
            "pair_tokens_per_word": pt / max(1, r["pair_words"]),
            "unk_tokens": su + ru,
            "source_words": r["source_words"],
            "rewrite_words": r["rewrite_words"],
            "pair_words": r["pair_words"],
        })
    total_words = sum(r["pair_words"] for r in rows)
    total_tokens = sum(r["pair_tokens"] for r in records)
    return {
        "name": name,
        "pairs": len(rows),
        "pair_words": total_words,
        "pair_tokens": total_tokens,
        "weighted_pair_tokens_per_word": total_tokens / max(1, total_words),
        "total_unk_tokens": sum(r["unk_tokens"] for r in records),
        "unique_token_ids": len(token_counter),
        "source_tokens_per_word": stat([r["source_tokens_per_word"] for r in records]),
        "rewrite_tokens_per_word": stat([r["rewrite_tokens_per_word"] for r in records]),
        "pair_tokens_per_word": stat([r["pair_tokens_per_word"] for r in records]),
        "pair_tokens": stat([r["pair_tokens"] for r in records]),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), use_fast=True)
    old = read_pairs(OLD_SELECTED)
    old_keys = {r["key"] for r in old}
    max_rows = read_pairs(MATCHED_MAX)
    by_group: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in max_rows:
        if r["key"] in old_keys:
            by_group["old_selected_1x_prefix"].append(r)
        else:
            by_group[r["origin"]].append(r)
    by_group["matched_max_all"].extend(max_rows)
    summaries = {name: summarize_group(name, rows, tok) for name, rows in sorted(by_group.items())}
    old_sum = summaries["old_selected_1x_prefix"]
    shifts = {}
    for name, s in summaries.items():
        if name == "old_selected_1x_prefix":
            continue
        shifts[name] = {
            "weighted_pair_tokens_per_word_shift": s["weighted_pair_tokens_per_word"] - old_sum["weighted_pair_tokens_per_word"],
            "weighted_pair_tokens_per_word_ratio": s["weighted_pair_tokens_per_word"] / old_sum["weighted_pair_tokens_per_word"],
            "pair_token_mean_shift": s["pair_tokens"]["mean"] - old_sum["pair_tokens"]["mean"],
            "pair_token_median_shift": s["pair_tokens"]["median"] - old_sum["pair_tokens"]["median"],
        }
    payload = {
        "status": "DOSE_TOKENIZATION_AUDIT_DONE",
        "tokenizer_dir": str(TOKENIZER_DIR),
        "tokenizer_vocab_size": len(tok),
        "groups": summaries,
        "shifts_vs_old_selected_1x_prefix": shifts,
        "interpretation": "Fixed research tokenizer prevents per-dose tokenizer refit confounding; these numbers measure whether the new matched increment still changes subword burden enough to qualify the dose result.",
    }
    out_json = OUT_DIR / "dose_tokenization_audit.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research dose tokenization audit", "", payload["interpretation"], ""]
    for name, s in summaries.items():
        lines.append(f"- {name}: pairs {s['pairs']}, words {s['pair_words']}, tokens {s['pair_tokens']}, weighted tokens/word {s['weighted_pair_tokens_per_word']:.4f}, unk {s['total_unk_tokens']}, unique token ids {s['unique_token_ids']}")
    lines.append("")
    lines.append("## Shifts vs old selected 1x prefix")
    for name, s in shifts.items():
        lines.append(f"- {name}: tokens/word shift {s['weighted_pair_tokens_per_word_shift']:+.4f} (ratio {s['weighted_pair_tokens_per_word_ratio']:.4f}), pair-token mean shift {s['pair_token_mean_shift']:+.3f}")
    lines.append("")
    lines.append(f"JSON: `{out_json}`")
    ((OUT_DIR.parents[4] / 'research/documents/frontier_consolidation/data/dose_tokenization_audit/dose_tokenization_audit.md')).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_json": str(out_json), "shifts_vs_old": shifts}, indent=2, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
