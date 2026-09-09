#!/usr/bin/env python3
"""research: tokenization and truncation audit for the MAX-register streams.

The seed43022 register result is negative for childspeech_removed - adultprose_removed.
This file-only audit checks whether the childspeech-removal stream has fewer
subword tokens and less truncation than its partner under the exact tokenizer and
seq256 loader.  If so, tokenization burden is not a natural explanation for its
lower score; if it has more tokens or more truncation, the contrast must be read
with that effective-exposure difference in mind.

No training, model loading, official evaluation, upload, or leaderboard action.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import json
import pathlib
import statistics
import time
from collections import defaultdict
from typing import Any


def find_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
TOKENIZER_DIR = WS / "data/compliant_tokenizer"
POOL = WS / "data/register_max_rowholdout_pools"
OUT = WS / "data/register_stream_effective_exposure_audit"
SEQ_LEN = 256
PASSES = 10
ARMS = {
    "childspeech_removed": POOL / "regmax_childspeech_samefw_10M.jsonl",
    "adultprose_removed": POOL / "regmax_adultprose_samefw_10M.jsonl",
}
OBSERVED = {
    "seed43022_child_minus_adult_exEntity4_80M": -0.4925,
    "seed43022_child_minus_adult_exEntity4_100M": -0.6125,
    "seed43022_child_minus_adult_cheap5_80M": -0.5120,
    "seed43022_child_minus_adult_cheap5_100M": -0.7140,
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def norm_source(row: dict[str, Any]) -> str:
    s = str(row.get("source", row.get("source_name", "unknown")))
    return s.split("::", 1)[1] if "::" in s else s


def nwords(row: dict[str, Any]) -> int:
    return int(row.get("words", len(str(row.get("text", "")).split())))


def new_acc() -> dict[str, Any]:
    return {
        "rows": 0,
        "words": 0,
        "chars": 0,
        "token_len_sum": 0,
        "token_after_trunc_sum": 0,
        "truncated_rows": 0,
        "truncated_tokens": 0,
        "max_token_len": 0,
        "token_lens": [],
    }


def add(acc: dict[str, Any], words: int, chars: int, token_len: int) -> None:
    acc["rows"] += 1
    acc["words"] += words
    acc["chars"] += chars
    acc["token_len_sum"] += token_len
    acc["token_after_trunc_sum"] += min(token_len, SEQ_LEN)
    acc["truncated_rows"] += int(token_len > SEQ_LEN)
    acc["truncated_tokens"] += max(0, token_len - SEQ_LEN)
    acc["max_token_len"] = max(acc["max_token_len"], token_len)
    acc["token_lens"].append(token_len)


def finalize(acc: dict[str, Any]) -> dict[str, Any]:
    rows = acc["rows"]
    words = acc["words"]
    token_lens = acc.pop("token_lens")
    mean = acc["token_len_sum"] / rows if rows else 0.0
    out = dict(acc)
    out.update({
        "mean_tokens_per_row": mean,
        "mean_tokens_per_word": acc["token_len_sum"] / words if words else 0.0,
        "mean_after_trunc_tokens_per_row": acc["token_after_trunc_sum"] / rows if rows else 0.0,
        "truncated_row_fraction": acc["truncated_rows"] / rows if rows else 0.0,
        "truncated_token_fraction": acc["truncated_tokens"] / acc["token_len_sum"] if acc["token_len_sum"] else 0.0,
        "p50_token_len": statistics.median(token_lens) if token_lens else None,
        "p90_token_len": sorted(token_lens)[int(0.9 * (len(token_lens)-1))] if token_lens else None,
        "p99_token_len": sorted(token_lens)[int(0.99 * (len(token_lens)-1))] if token_lens else None,
        "hundredM_token_len_sum": acc["token_len_sum"] * PASSES,
        "hundredM_token_after_trunc_sum": acc["token_after_trunc_sum"] * PASSES,
        "hundredM_truncated_tokens": acc["truncated_tokens"] * PASSES,
    })
    return out


def audit_arm(path: pathlib.Path, tokenizer) -> dict[str, Any]:
    totals = new_acc()
    by_kind: dict[str, dict[str, Any]] = defaultdict(new_acc)
    by_source: dict[str, dict[str, Any]] = defaultdict(new_acc)
    examples = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            text = str(row.get("text", ""))
            ids = tokenizer(text, add_special_tokens=False, truncation=False, return_attention_mask=False)["input_ids"]
            token_len = len(ids)
            words = nwords(row)
            chars = len(text)
            kind = str(row.get("regmax_row_kind", "unmarked"))
            src = norm_source(row)
            add(totals, words, chars, token_len)
            add(by_kind[kind], words, chars, token_len)
            add(by_source[src], words, chars, token_len)
            if token_len > SEQ_LEN and len(examples) < 8:
                examples.append({
                    "row_index": i,
                    "kind": kind,
                    "source": src,
                    "words": words,
                    "token_len": token_len,
                    "truncated_tokens": token_len - SEQ_LEN,
                    "text_prefix": text[:180],
                })
    return {
        "path": rel(path),
        "total": finalize(totals),
        "by_kind": {k: finalize(v) for k, v in sorted(by_kind.items())},
        "by_source": {k: finalize(v) for k, v in sorted(by_source.items())},
        "truncated_examples": examples,
    }


def subtract(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "rows", "words", "chars", "token_len_sum", "token_after_trunc_sum",
        "truncated_rows", "truncated_tokens", "hundredM_token_len_sum",
        "hundredM_token_after_trunc_sum", "hundredM_truncated_tokens",
        "mean_tokens_per_row", "mean_tokens_per_word", "mean_after_trunc_tokens_per_row",
        "truncated_row_fraction", "truncated_token_fraction",
    ]
    return {k: (a.get(k, 0) - b.get(k, 0)) for k in keys}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), use_fast=True)
    tok.model_max_length = 10**9
    arms = {name: audit_arm(path, tok) for name, path in ARMS.items()}
    child = arms["childspeech_removed"]["total"]
    adult = arms["adultprose_removed"]["total"]
    diff_total = subtract(child, adult)
    diff_by_kind: dict[str, Any] = {}
    all_kinds = sorted(set(arms["childspeech_removed"]["by_kind"]) | set(arms["adultprose_removed"]["by_kind"]))
    for k in all_kinds:
        diff_by_kind[k] = subtract(arms["childspeech_removed"]["by_kind"].get(k, {}), arms["adultprose_removed"]["by_kind"].get(k, {}))
    child_fewer_tokens = diff_total["token_len_sum"] < 0 and diff_total["token_after_trunc_sum"] < 0
    child_less_trunc = diff_total["truncated_rows"] < 0 and diff_total["truncated_tokens"] < 0
    payload = {
        "status": "REGISTER_STREAM_EFFECTIVE_EXPOSURE_AUDIT_DONE",
        "created_utc": now(),
        "purpose": "Check whether tokenization/truncation effective exposure can explain the seed43022 register reversal before using a seed43122 replication.",
        "tokenizer": rel(TOKENIZER_DIR),
        "seq_len": SEQ_LEN,
        "passes_in_100M_stream": PASSES,
        "arms": arms,
        "childspeech_removed_minus_adultprose_removed_total": diff_total,
        "childspeech_removed_minus_adultprose_removed_by_kind": diff_by_kind,
        "observed_seed43022_broad_sign": OBSERVED,
        "interpretation": {
            "childspeech_removed_has_fewer_tokens_and_less_truncation": bool(child_fewer_tokens and child_less_trunc),
            "childspeech_removed_has_fewer_tokens": bool(child_fewer_tokens),
            "childspeech_removed_has_less_truncation": bool(child_less_trunc),
            "reading": "If the flags are true, the childspeech-removal arm was not disadvantaged by greater token count or truncation burden despite scoring lower; if false, effective exposure remains a possible contributor that the seed replicate must absorb.",
        },
        "no_training_model_loading_official_eval_upload_or_leaderboard": True,
    }
    (OUT / "effective_exposure_audit.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # Compact CSVs for later comparison.
    with (OUT / "arm_totals.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["arm"] + list(next(iter(arms.values()))["total"].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for arm, rec in arms.items():
            w.writerow({"arm": arm, **rec["total"]})
    with (OUT / "kind_differences.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["kind"] + list(next(iter(diff_by_kind.values())).keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for k, rec in diff_by_kind.items():
            w.writerow({"kind": k, **rec})
    def fmt(x: Any) -> str:
        return f"{x:.6f}" if isinstance(x, float) else str(x)
    lines = [
        "# research register stream effective-exposure audit", "",
        "Tokenization/truncation comparison for the two fixed research MAX-register 10M streams under the exact compliant tokenizer and seq256 training loader. The 100M stream is ten repetitions, so 100M totals are 10x the 10M totals.", "",
        "## Total arm comparison", "",
        "| arm | words | tokens before trunc | tokens after trunc | truncated rows | truncated tokens | tokens/word | truncated row fraction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, rec in arms.items():
        t = rec["total"]
        lines.append(f"| {arm} | {t['words']} | {t['token_len_sum']} | {t['token_after_trunc_sum']} | {t['truncated_rows']} | {t['truncated_tokens']} | {t['mean_tokens_per_word']:.6f} | {t['truncated_row_fraction']:.6f} |")
    lines += ["", "## childspeech_removed minus adultprose_removed", ""]
    for k in ["token_len_sum", "token_after_trunc_sum", "truncated_rows", "truncated_tokens", "mean_tokens_per_word", "truncated_row_fraction"]:
        lines.append(f"- {k}: {fmt(diff_total[k])}")
    lines += ["", "## Reading", ""]
    if child_fewer_tokens and child_less_trunc:
        lines.append("The childspeech-removal stream has fewer effective tokens and less truncation than the adultprose-removal stream. Since the observed seed43022 register contrast is negative for childspeech_removed - adultprose_removed, tokenization/truncation burden runs opposite to the observed sign.")
    else:
        lines.append("The childspeech-removal stream does not have both fewer effective tokens and less truncation than the adultprose-removal stream. Treat tokenization/truncation as a possible contributor until the seed replicate is read.")
    lines += ["", f"JSON: `{rel(OUT / 'effective_exposure_audit.json')}`", f"CSV totals: `{rel(OUT / 'arm_totals.csv')}`", f"CSV by-kind differences: `{rel(OUT / 'kind_differences.csv')}`"]
    (OUT / "effective_exposure_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "summary_md": rel(OUT / "effective_exposure_audit.md"),
        "child_minus_adult_token_len_sum": diff_total["token_len_sum"],
        "child_minus_adult_token_after_trunc_sum": diff_total["token_after_trunc_sum"],
        "child_minus_adult_truncated_rows": diff_total["truncated_rows"],
        "child_minus_adult_truncated_tokens": diff_total["truncated_tokens"],
        "child_has_fewer_tokens_and_less_truncation": bool(child_fewer_tokens and child_less_trunc),
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
