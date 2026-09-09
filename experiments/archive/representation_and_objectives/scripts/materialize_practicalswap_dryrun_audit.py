#!/usr/bin/env python3
"""research: materialize and audit a no-training practical-swap dry corpus.

This CPU-only script takes the repaired research exact pair-word swap plan and
creates a 10M dry-run corpus by replacing each selected compact_view_reinvest
packet with an unused practical packet of identical pair_words at the same pair
position.  It preserves the original pair order length sequence, top-up row,
common filler, and total 10M pool words.  It also audits baseline16k tokenizer
exposure, seq256 joint visibility, and textual overlap between added packets and
GlobalPIQA prompts/options.

No 100M training stream is written, no GPU is used, and no managed research output
is read.
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import pathlib
import re
import statistics
import string
from typing import Any, Iterable

ROOT = pathlib.Path(".").resolve()
A01 = ROOT / "experiments/archive" / 'representation_and_objectives'
A02 = ROOT / "experiments/archive" / 'frontier_consolidation'
DENSITY_DIR = A02 / "data" / "density_core_reinvestment_medium_riskhard"
OVERLAY_DIR = A02 / "data" / "density_cleanqwen_overlay_medium_riskhard"
TOKENIZER = ROOT / "experiments/archive" / 'initial_model_studies' / "training" / "runs" / "fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256" / "hf_model"
SWAP_PLAN = A01 / "data" / "globalpiqa_practical_repair_blueprint" / "exact_wordlength_practical_swap_plan.jsonl"
GLOBALPIQA_FAST = ROOT / "experiments/archive" / 'initial_model_studies' / "repos" / "babylm-eval" / "strict" / "evaluation_data" / "fast_eval"
OUT_DIR = A01 / "data" / "practicalswap_dryrun_audit"
OUT_POOL = OUT_DIR / "cleanqwen_fineweb_compact_view_reinvest_practicalswap143_10M.jsonl"
OUT_ROW_META = OUT_DIR / "cleanqwen_fineweb_compact_view_reinvest_practicalswap143_changed_block_rows_meta.jsonl"
OUT_PAIR_LIST = OUT_DIR / "selected_compact_reinvest_practicalswap143_pairs.jsonl"
OUT_AUDIT = OUT_DIR / "practicalswap143_dryrun_audit.json"
OUT_NOTE = (ROOT / 'research/notes/representation_and_objectives/practicalswap143_dryrun_audit.md')
SEQ_LEN = 256
TOTAL_WORDS = 10_000_000
MAX_PACKET_WORDS = 160
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\-]*|\d+(?:\.\d+)?")
WORD_RE = re.compile(r"[A-Za-z0-9]+")
STOP = {
    "a", "an", "the", "and", "or", "but", "if", "then", "when", "what", "which", "who", "whom", "whose",
    "to", "of", "in", "on", "at", "by", "for", "from", "with", "without", "into", "onto", "over", "under",
    "is", "are", "was", "were", "be", "been", "being", "am", "do", "does", "did", "will", "would", "can", "could",
    "should", "may", "might", "must", "as", "it", "its", "this", "that", "these", "those", "you", "your", "i", "we",
    "they", "he", "she", "them", "him", "her", "his", "their", "our", "my", "me", "some", "any", "all", "not", "no",
    "one", "two", "three", "four", "there", "here", "after", "before", "than", "more", "less", "same", "other", "another",
}


def read_jsonl(path: pathlib.Path, limit: int | None = None) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: pathlib.Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def wc(text: str) -> int:
    return len(str(text or "").split())


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_for_copy(text: str) -> str:
    return " ".join(WORD_RE.findall(str(text or "").lower()))


def norm_tok(tok: str) -> str:
    tok = tok.lower().strip("'\"")
    if len(tok) > 4 and tok.endswith("ies"):
        tok = tok[:-3] + "y"
    elif len(tok) > 4 and tok.endswith("ves"):
        tok = tok[:-3] + "f"
    elif len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        tok = tok[:-1]
    return tok


def content_tokens(text: str) -> set[str]:
    out = set()
    for raw in TOKEN_RE.findall(str(text or "").lower().translate(str.maketrans("", "", string.punctuation.replace("'", "")))):
        tok = norm_tok(raw)
        if len(tok) < 3 or tok in STOP:
            continue
        out.add(tok)
    return out


def word_ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    toks = WORD_RE.findall(str(text or "").lower())
    if len(toks) < n:
        return set()
    return {tuple(toks[i:i+n]) for i in range(len(toks) - n + 1)}


def jaccard(a: set[Any], b: set[Any]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def quantiles(vals: list[float]) -> dict[str, Any]:
    if not vals:
        return {"n": 0}
    xs = sorted(float(v) for v in vals)
    def q(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        pos = p * (len(xs) - 1)
        lo = int(math.floor(pos)); hi = int(math.ceil(pos))
        if lo == hi:
            return xs[lo]
        return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)
    return {"n": len(xs), "min": xs[0], "p05": q(0.05), "p25": q(0.25), "mean": sum(xs)/len(xs), "median": q(0.5), "p75": q(0.75), "p95": q(0.95), "max": xs[-1], "sum": sum(xs)}


def pair_id_from_public(rec: dict[str, Any]) -> str:
    return str(rec.get("pair_id") or "")


def source_key_from_public(rec: dict[str, Any]) -> str:
    if rec.get("key"):
        return str(rec["key"])
    sid = str(rec.get("sentence_id") or "")
    doc = str(rec.get("doc_id") or "")
    if sid:
        return f"sid:{sid}|doc:{doc}"
    return "txt:" + hashlib.sha1(norm_for_copy(str(rec.get("source_text") or "")).encode("utf-8")).hexdigest()[:20]


def validate_pair(rec: dict[str, Any]) -> dict[str, Any]:
    source = " ".join(str(rec.get("source_text") or "").split())
    rewrite = " ".join(str(rec.get("rewrite_text") or "").split())
    out = dict(rec)
    out["source_text"] = source
    out["rewrite_text"] = rewrite
    out["source_words"] = int(rec.get("source_words") or wc(source))
    out["rewrite_words"] = int(rec.get("rewrite_words") or wc(rewrite))
    out["pair_words"] = int(rec.get("pair_words") or (out["source_words"] + out["rewrite_words"]))
    out["pair_id"] = pair_id_from_public(rec)
    out["key"] = source_key_from_public(rec)
    out["sentence_id"] = str(rec.get("sentence_id") or "")
    out["doc_id"] = str(rec.get("doc_id") or "")
    out["domain_hits"] = [str(x) for x in (rec.get("domain_hits") or [])]
    if out["source_words"] != wc(source) or out["rewrite_words"] != wc(rewrite):
        raise RuntimeError(f"word mismatch for {out['pair_id']}")
    if out["pair_words"] != out["source_words"] + out["rewrite_words"]:
        raise RuntimeError(f"pair word mismatch for {out['pair_id']}")
    if out["pair_words"] > MAX_PACKET_WORDS:
        raise RuntimeError(f"pair too long {out['pair_id']}")
    return out


def pack_pair_rows(pairs: list[dict[str, Any]], source_label: str, example_base: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cur_segments: list[str] = []
    cur_words = 0
    cur_pair_ids: list[str] = []
    cur_domains: collections.Counter[str] = collections.Counter()
    for p in pairs:
        text = f"{p['source_text']} {p['rewrite_text']}".strip()
        L = wc(text)
        if L != int(p["pair_words"]):
            raise RuntimeError(f"packed pair word mismatch {p['pair_id']}")
        if cur_words and cur_words + L > MAX_PACKET_WORDS:
            rows.append({
                "text": " ".join(cur_segments),
                "words": cur_words,
                "example_id": example_base + len(rows),
                "source": source_label,
                "pair_ids": list(cur_pair_ids),
                "component_sources": dict(cur_domains),
            })
            cur_segments, cur_words, cur_pair_ids, cur_domains = [], 0, [], collections.Counter()
        cur_segments.append(text)
        cur_words += L
        cur_pair_ids.append(str(p["pair_id"]))
        for d in p.get("domain_hits") or ["no_domain"]:
            cur_domains[str(d)] += L
    if cur_segments:
        rows.append({
            "text": " ".join(cur_segments),
            "words": cur_words,
            "example_id": example_base + len(rows),
            "source": source_label,
            "pair_ids": list(cur_pair_ids),
            "component_sources": dict(cur_domains),
        })
    for r in rows:
        if int(r["words"]) != wc(str(r["text"])) or int(r["words"]) > MAX_PACKET_WORDS:
            raise RuntimeError("row packing invariant failed")
    return rows


def row_record(r: dict[str, Any]) -> dict[str, Any]:
    return {"text": r["text"], "words": int(r["words"]), "example_id": int(r["example_id"]), "source": str(r["source"])}


def row_meta_record(i: int, r: dict[str, Any]) -> dict[str, Any]:
    return {"row_index": i, "example_id": int(r["example_id"]), "words": int(r["words"]), "pair_ids": r.get("pair_ids") or [], "component_sources": r.get("component_sources") or {}}


def enc_len(tok: Any, text: str, add_special_tokens: bool = False) -> int:
    return len(tok.encode(text, add_special_tokens=add_special_tokens))


def tokenizer_pair_stats(tok: Any, pairs: list[dict[str, Any]]) -> dict[str, Any]:
    pair_toks = []
    source_toks = []
    rewrite_toks = []
    ratios = []
    for p in pairs:
        st = enc_len(tok, p["source_text"], False)
        rt = enc_len(tok, p["rewrite_text"], False)
        pt = enc_len(tok, f"{p['source_text']} {p['rewrite_text']}", False)
        source_toks.append(st); rewrite_toks.append(rt); pair_toks.append(pt)
        ratios.append(rt / max(1, st))
    return {
        "pairs": len(pairs),
        "source_tokens_no_special": quantiles(source_toks),
        "rewrite_tokens_no_special": quantiles(rewrite_toks),
        "pair_tokens_no_special": quantiles(pair_toks),
        "rewrite_to_source_token_ratio": quantiles(ratios),
        "total_source_tokens": sum(source_toks),
        "total_rewrite_tokens": sum(rewrite_toks),
        "total_pair_tokens": sum(pair_toks),
    }


def compare_token_deltas(tok: Any, before: list[dict[str, Any]], after: list[dict[str, Any]]) -> dict[str, Any]:
    if len(before) != len(after):
        raise RuntimeError("pair lists differ in length")
    deltas_pair = []
    deltas_source = []
    deltas_rewrite = []
    changed = 0
    for b, a in zip(before, after):
        bp = enc_len(tok, f"{b['source_text']} {b['rewrite_text']}", False)
        ap = enc_len(tok, f"{a['source_text']} {a['rewrite_text']}", False)
        bs = enc_len(tok, b["source_text"], False); ass = enc_len(tok, a["source_text"], False)
        br = enc_len(tok, b["rewrite_text"], False); ar = enc_len(tok, a["rewrite_text"], False)
        deltas_pair.append(ap - bp); deltas_source.append(ass - bs); deltas_rewrite.append(ar - br)
        if b["key"] != a["key"]:
            changed += 1
    return {
        "changed_pair_positions": changed,
        "pair_token_delta_after_minus_before": quantiles(deltas_pair),
        "source_token_delta_after_minus_before": quantiles(deltas_source),
        "rewrite_token_delta_after_minus_before": quantiles(deltas_rewrite),
        "total_pair_token_delta": sum(deltas_pair),
        "total_source_token_delta": sum(deltas_source),
        "total_rewrite_token_delta": sum(deltas_rewrite),
    }


def visibility(tok: Any, pair_by_id: dict[str, dict[str, Any]], rows: list[dict[str, Any]], special_overhead: int) -> dict[str, Any]:
    cutoff = SEQ_LEN - special_overhead
    pair_records = []
    row_records = []
    for ridx, row in enumerate(rows):
        pids = row.get("pair_ids") or []
        prefix = ""
        full = partial = hidden = src_visible = 0
        for j, pid in enumerate(pids):
            p = pair_by_id[pid]
            source = str(p["source_text"])
            rewrite = str(p["rewrite_text"])
            unit = f"{source} {rewrite}".strip()
            source_prefix = prefix + (" " if prefix else "") + source
            pair_prefix = prefix + (" " if prefix else "") + unit
            start_tok = enc_len(tok, prefix, False) if prefix else 0
            source_end = enc_len(tok, source_prefix, False)
            pair_end = enc_len(tok, pair_prefix, False)
            visible_inside = max(0, min(pair_end, cutoff) - start_tok)
            source_ok = source_end <= cutoff
            full_ok = pair_end <= cutoff
            partial_ok = visible_inside > 0 and not full_ok
            if source_ok:
                src_visible += 1
            if full_ok:
                full += 1
            elif partial_ok:
                partial += 1
            else:
                hidden += 1
            pair_records.append({
                "row_index": ridx, "pair_position_in_row": j, "pair_id": pid,
                "start_tok_no_special": start_tok, "source_end_tok_no_special": source_end,
                "pair_end_tok_no_special": pair_end, "source_visible": source_ok,
                "pair_full_visible": full_ok, "pair_partial_visible": partial_ok,
            })
            prefix = pair_prefix
        row_records.append({
            "row_index": ridx, "pairs": len(pids), "row_words": int(row["words"]),
            "row_tokens_no_special": enc_len(tok, row["text"], False),
            "row_tokens_with_special": enc_len(tok, row["text"], True),
            "fully_visible_pairs": full, "partially_visible_pairs": partial, "hidden_pairs": hidden,
            "source_visible_pairs": src_visible,
        })
    total = len(pair_records)
    full = sum(1 for r in pair_records if r["pair_full_visible"])
    src = sum(1 for r in pair_records if r["source_visible"])
    partial = sum(1 for r in pair_records if r["pair_partial_visible"])
    hidden = total - full - partial
    trunc_rows = [r for r in row_records if r["row_tokens_with_special"] > SEQ_LEN]
    return {
        "total_pair_occurrences": total,
        "unique_pairs": len({r["pair_id"] for r in pair_records}),
        "source_visible_pairs": src,
        "full_source_plus_rewrite_visible_pairs": full,
        "partial_visible_pairs": partial,
        "hidden_pairs": hidden,
        "source_visible_rate": src / total if total else None,
        "full_pair_visible_rate": full / total if total else None,
        "partial_visible_rate": partial / total if total else None,
        "hidden_rate": hidden / total if total else None,
        "rows_over_seq256_with_special": len(trunc_rows),
        "truncated_rows_with_pair_loss": sum(1 for r in trunc_rows if r["fully_visible_pairs"] < r["pairs"]),
        "row_tokens_with_special": quantiles([r["row_tokens_with_special"] for r in row_records]),
        "lost_pairs_per_pair_row": quantiles([r["pairs"] - r["fully_visible_pairs"] for r in row_records if r["pairs"]]),
        "pair_position_loss_counts": dict(collections.Counter(r["pair_position_in_row"] for r in pair_records if not r["pair_full_visible"]).most_common()),
    }


def read_globalpiqa_examples() -> list[dict[str, Any]]:
    examples = []
    for split in ["global_piqa_parallel", "global_piqa_nonparallel"]:
        path = GLOBALPIQA_FAST / split / "eng_latn.jsonl"
        for row in read_jsonl(path):
            sols = []
            i = 0
            while f"solution{i}" in row:
                sols.append(str(row[f"solution{i}"]))
                i += 1
            text = " ".join([str(row.get("prompt", ""))] + sols)
            examples.append({
                "split": "parallel" if "parallel" in split and "nonparallel" not in split else "nonparallel",
                "example_id": str(row.get("example_id")),
                "prompt": str(row.get("prompt", "")),
                "solutions": sols,
                "categories": str(row.get("categories") or ""),
                "text": text,
                "content_tokens": content_tokens(text),
                "ngrams5": word_ngrams(text, 5),
                "ngrams7": word_ngrams(text, 7),
            })
    return examples


def contamination_overlap(additions: list[dict[str, Any]], removals: list[dict[str, Any]]) -> dict[str, Any]:
    examples = read_globalpiqa_examples()
    def scan(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
        pair_best = []
        all_scores = []
        exact_7 = 0
        exact_5 = 0
        for r in rows:
            text = f"{r['source_text']} {r['rewrite_text']}"
            toks = content_tokens(text)
            ng5 = word_ngrams(text, 5)
            ng7 = word_ngrams(text, 7)
            best = None
            for ex in examples:
                cj = jaccard(toks, ex["content_tokens"])
                n5 = len(ng5 & ex["ngrams5"])
                n7 = len(ng7 & ex["ngrams7"])
                score = (n7 > 0, n5 > 0, cj, n7, n5)
                rec = {
                    "pair_key": r["key"], "pair_id": r["pair_id"], "pair_text": text,
                    "example_id": ex["example_id"], "split": ex["split"], "categories": ex["categories"],
                    "content_jaccard": round(cj, 6), "shared_content_tokens": sorted(list(toks & ex["content_tokens"]))[:30],
                    "shared_5grams": n5, "shared_7grams": n7,
                    "example_prompt": ex["prompt"], "example_solutions": ex["solutions"],
                }
                if best is None or score > best[0]:
                    best = (score, rec)
            if best is not None:
                rec = best[1]
                pair_best.append(rec)
                all_scores.append(float(rec["content_jaccard"]))
                if int(rec["shared_7grams"]) > 0:
                    exact_7 += 1
                if int(rec["shared_5grams"]) > 0:
                    exact_5 += 1
        pair_best.sort(key=lambda x: (x["shared_7grams"], x["shared_5grams"], x["content_jaccard"]), reverse=True)
        return {
            "label": label,
            "rows": len(rows),
            "pairs_with_any_shared_7gram_to_globalpiqa": exact_7,
            "pairs_with_any_shared_5gram_to_globalpiqa": exact_5,
            "max_content_jaccard_stats": quantiles(all_scores),
            "highest_overlap_examples": pair_best[:20],
        }
    return {"globalpiqa_examples": len(examples), "additions": scan(additions, "additions"), "removals": scan(removals, "removals")}


def mean_field(rows: list[dict[str, Any]], field: str) -> float | None:
    vals = [float(r[field]) for r in rows if r.get(field) is not None]
    return sum(vals) / len(vals) if vals else None


def write_note(payload: dict[str, Any]) -> None:
    a = payload["audit"]
    v = payload["visibility"]
    c = payload["contamination_overlap"]
    lines: list[str] = []
    lines.append("# research practicalswap143 dry-run corpus and audit")
    lines.append("")
    lines.append("This CPU-only materialization replaces 143 selected compact-view packets with valid unused practical packets of identical pair-word length at the same pair positions. It writes a 10M dry-run pool and row metadata only; no 100M training stream, GPU job, or evaluation was launched.")
    lines.append("")
    lines.append("## Corpus invariants")
    lines.append("")
    lines.append(f"- Replacements: {a['replacement_count']} pair positions; added pair words {a['added_pair_words']}, removed pair words {a['removed_pair_words']}, net {a['net_pair_words']}.")
    lines.append(f"- Repaired selected pair list: {a['repaired_pair_count']} pairs / {a['repaired_pair_words']} words; original selected words {a['original_pair_words']}.")
    lines.append(f"- Pair-row count {a['pair_rows']} and changed-row count {a['changed_rows']} match the original; top-up rows copied from original changed block: {a['copied_topup_rows']}.")
    lines.append(f"- 10M pool words: {a['pool_words']}; row word-length sequence identical to original: {a['row_word_length_sequence_identical_to_original']}.")
    lines.append("")
    lines.append("## Tokenizer and visibility")
    lines.append("")
    td = payload["tokenizer_delta_after_minus_original"]
    lines.append(f"- Total pair-token delta after-minus-original over all 12,155 pair positions: {td['total_pair_token_delta']} (source {td['total_source_token_delta']}, rewrite {td['total_rewrite_token_delta']}).")
    lines.append(f"- Pair-token delta per position: mean {td['pair_token_delta_after_minus_before']['mean']:.3f}, p95 {td['pair_token_delta_after_minus_before']['p95']:.3f}, min {td['pair_token_delta_after_minus_before']['min']}, max {td['pair_token_delta_after_minus_before']['max']}.")
    lines.append(f"- Seq256 full source+rewrite visibility after swap: {v['full_source_plus_rewrite_visible_pairs']}/{v['total_pair_occurrences']} = {v['full_pair_visible_rate']:.6f}; source-visible {v['source_visible_rate']:.6f}; hidden pairs {v['hidden_pairs']}.")
    lines.append(f"- Rows over seq256 with special tokens: {v['rows_over_seq256_with_special']}; pair-position loss counts: {v['pair_position_loss_counts']}.")
    lines.append("")
    lines.append("## GlobalPIQA overlap screen")
    lines.append("")
    lines.append(f"- Added packets with any exact 7-gram shared with GlobalPIQA prompt/options: {c['additions']['pairs_with_any_shared_7gram_to_globalpiqa']} / {c['additions']['rows']}.")
    lines.append(f"- Added packets with any exact 5-gram shared with GlobalPIQA prompt/options: {c['additions']['pairs_with_any_shared_5gram_to_globalpiqa']} / {c['additions']['rows']}.")
    lines.append(f"- Added max content-Jaccard to a GlobalPIQA item: mean {c['additions']['max_content_jaccard_stats']['mean']:.3f}, p95 {c['additions']['max_content_jaccard_stats']['p95']:.3f}, max {c['additions']['max_content_jaccard_stats']['max']:.3f}.")
    lines.append("")
    lines.append("## Scientific reading")
    lines.append("")
    lines.append("The dry-run corpus demonstrates that the proposed practical repair can be represented as a very small, exact-whitespace-budget perturbation of the existing compact_view_reinvest stream while preserving row lengths and high joint visibility. The tokenizer-token delta is not zero, so any future trained comparison would still have a small lexical/token-exposure shift in addition to the semantic practical-action shift.")
    lines.append("The GlobalPIQA overlap screen should be read as a contamination safeguard, not as a selector. If high overlap examples appear in the JSON tail they must be inspected before training; if the pending endpoint evidence does not survive, this dry-run should remain unused.")
    lines.append("")
    lines.append(f"Pool: `{OUT_POOL}`")
    lines.append(f"Audit JSON: `{OUT_AUDIT}`")
    OUT_NOTE.parent.mkdir(parents=True, exist_ok=True)
    OUT_NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    from transformers import AutoTokenizer  # type: ignore
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True, use_fast=True)
    special_overhead = enc_len(tok, "hello", True) - enc_len(tok, "hello", False)

    overlay_meta = json.loads((OVERLAY_DIR / "density_cleanqwen_rowholdout_overlay_metadata.json").read_text(encoding="utf-8"))
    changed_rows = int(overlay_meta["families"]["compact_reinvest"]["changed_block_rows"])
    pair_rows_expected = int(overlay_meta["families"]["compact_reinvest"]["pair_rows"])

    original_pairs = [validate_pair(p) for p in read_jsonl(DENSITY_DIR / "selected_compact_reinvest_pairs.jsonl")]
    original_by_key = {p["key"]: p for p in original_pairs}
    original_by_id = {p["pair_id"]: p for p in original_pairs}
    swap_rows = list(read_jsonl(SWAP_PLAN))
    remove_to_add: dict[str, dict[str, Any]] = {}
    additions: list[dict[str, Any]] = []
    removals: list[dict[str, Any]] = []
    for s in swap_rows:
        add = validate_pair(s["add"])
        rem = validate_pair(s["remove"])
        if add["pair_words"] != rem["pair_words"]:
            raise RuntimeError(f"swap not exact length {add['key']} vs {rem['key']}")
        if rem["key"] not in original_by_key:
            raise RuntimeError(f"remove key not selected {rem['key']}")
        if add["key"] in original_by_key:
            raise RuntimeError(f"add key already selected {add['key']}")
        remove_to_add[rem["key"]] = add
        additions.append(add); removals.append(rem)

    repaired_pairs = [remove_to_add.get(p["key"], p) for p in original_pairs]
    if len({p["key"] for p in repaired_pairs}) != len(repaired_pairs):
        raise RuntimeError("duplicate key in repaired pair list")
    original_lengths = [int(p["pair_words"]) for p in original_pairs]
    repaired_lengths = [int(p["pair_words"]) for p in repaired_pairs]
    if original_lengths != repaired_lengths:
        raise RuntimeError("pair word length sequence changed")

    repaired_pair_rows = pack_pair_rows(repaired_pairs, "cleanqwen_fineweb_compact_view_reinvest_practicalswap143", 950000)
    original_pool_rows = list(read_jsonl(OVERLAY_DIR / "cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"))
    original_meta_rows = list(read_jsonl(OVERLAY_DIR / "cleanqwen_fineweb_compact_view_reinvest_changed_block_rows_meta.jsonl", limit=changed_rows))
    original_changed_rows = original_pool_rows[:changed_rows]
    original_filler_rows = original_pool_rows[changed_rows:]
    original_pair_row_count = sum(1 for m in original_meta_rows if m.get("pair_ids"))
    if len(repaired_pair_rows) != pair_rows_expected or original_pair_row_count != pair_rows_expected:
        raise RuntimeError(f"pair row count mismatch repaired {len(repaired_pair_rows)} expected {pair_rows_expected} original {original_pair_row_count}")
    copied_topup_rows = [dict(r) for r, m in zip(original_changed_rows, original_meta_rows) if not (m.get("pair_ids") or [])]
    if len(copied_topup_rows) != changed_rows - pair_rows_expected:
        raise RuntimeError("topup row count mismatch")
    new_changed_rows = [row_record(r) for r in repaired_pair_rows] + [row_record(r) for r in copied_topup_rows]
    new_pool_rows = new_changed_rows + [row_record(r) for r in original_filler_rows]
    pool_words = sum(int(r["words"]) for r in new_pool_rows)
    if pool_words != TOTAL_WORDS:
        raise RuntimeError(f"pool words {pool_words}")
    row_lengths_identical = [int(r["words"]) for r in new_pool_rows] == [int(r["words"]) for r in original_pool_rows]
    if not row_lengths_identical:
        raise RuntimeError("row word lengths changed")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT_POOL, new_pool_rows)
    row_meta = [row_meta_record(i, r) for i, r in enumerate(repaired_pair_rows)]
    for k, r in enumerate(copied_topup_rows, start=len(repaired_pair_rows)):
        # Preserve original top-up component/source metadata if present.
        om = original_meta_rows[k]
        row_meta.append({"row_index": k, "example_id": int(r["example_id"]), "words": int(r["words"]), "pair_ids": [], "component_sources": om.get("component_sources") or {}})
    write_jsonl(OUT_ROW_META, row_meta)
    write_jsonl(OUT_PAIR_LIST, repaired_pairs)

    repaired_by_id = {p["pair_id"]: p for p in repaired_pairs}
    vis = visibility(tok, repaired_by_id, repaired_pair_rows, special_overhead)
    token_delta = compare_token_deltas(tok, original_pairs, repaired_pairs)
    added_stats = tokenizer_pair_stats(tok, additions)
    removed_stats = tokenizer_pair_stats(tok, removals)
    original_stats = tokenizer_pair_stats(tok, original_pairs)
    repaired_stats = tokenizer_pair_stats(tok, repaired_pairs)
    cont = contamination_overlap(additions, removals)

    audit = {
        "status": "PRACTICALSWAP143_DRYRUN_AUDIT",
        "inputs": {
            "original_selected_pairs": str(DENSITY_DIR / "selected_compact_reinvest_pairs.jsonl"),
            "original_pool": str(OVERLAY_DIR / "cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"),
            "original_row_meta": str(OVERLAY_DIR / "cleanqwen_fineweb_compact_view_reinvest_changed_block_rows_meta.jsonl"),
            "overlay_metadata": str(OVERLAY_DIR / "density_cleanqwen_rowholdout_overlay_metadata.json"),
            "swap_plan": str(SWAP_PLAN),
            "tokenizer": str(TOKENIZER),
        },
        "audit": {
            "replacement_count": len(swap_rows),
            "original_pair_count": len(original_pairs),
            "repaired_pair_count": len(repaired_pairs),
            "original_pair_words": sum(int(p["pair_words"]) for p in original_pairs),
            "repaired_pair_words": sum(int(p["pair_words"]) for p in repaired_pairs),
            "added_pair_words": sum(int(p["pair_words"]) for p in additions),
            "removed_pair_words": sum(int(p["pair_words"]) for p in removals),
            "net_pair_words": sum(int(p["pair_words"]) for p in additions) - sum(int(p["pair_words"]) for p in removals),
            "pair_word_length_sequence_identical": original_lengths == repaired_lengths,
            "pair_rows": len(repaired_pair_rows),
            "changed_rows": len(new_changed_rows),
            "copied_topup_rows": len(copied_topup_rows),
            "pool_rows": len(new_pool_rows),
            "pool_words": pool_words,
            "row_word_length_sequence_identical_to_original": row_lengths_identical,
            "changed_block_words": sum(int(r["words"]) for r in new_changed_rows),
            "filler_rows_copied": len(original_filler_rows),
            "filler_words_copied": sum(int(r["words"]) for r in original_filler_rows),
            "added_means": {
                "content_recall": mean_field(additions, "content_recall"),
                "entity_recall": mean_field(additions, "entity_recall"),
                "number_recall": mean_field(additions, "number_recall"),
                "length_ratio": mean_field(additions, "length_ratio"),
            },
            "removed_means": {
                "content_recall": mean_field(removals, "content_recall"),
                "entity_recall": mean_field(removals, "entity_recall"),
                "number_recall": mean_field(removals, "number_recall"),
                "length_ratio": mean_field(removals, "length_ratio"),
            },
        },
        "tokenizer_stats": {
            "special_overhead_samples": {
                "empty": enc_len(tok, "", True) - enc_len(tok, "", False),
                "hello": special_overhead,
                "two_words": enc_len(tok, "hello world", True) - enc_len(tok, "hello world", False),
            },
            "original_selected_pairs": original_stats,
            "repaired_selected_pairs": repaired_stats,
            "added_pairs": added_stats,
            "removed_pairs": removed_stats,
        },
        "tokenizer_delta_after_minus_original": token_delta,
        "visibility": vis,
        "contamination_overlap": cont,
        "outputs": {
            "pool_10M": str(OUT_POOL),
            "row_meta": str(OUT_ROW_META),
            "pair_list": str(OUT_PAIR_LIST),
            "audit_json": str(OUT_AUDIT),
            "note": str(OUT_NOTE),
        },
        "sha256": {},
        "scientific_reading": {
            "main": "The practical swap is a small exact-whitespace-budget perturbation of compact_view_reinvest that preserves pair-word order lengths and row word lengths, but its tokenizer-token exposure and semantic distribution are measured rather than assumed identical.",
            "not_evidence_of_downstream_gain": "No training or official-compatible evaluation is run here; endpoint and seed evidence must decide whether this repair merits any GPU work.",
        },
    }
    OUT_AUDIT.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # Hash after audit path exists, then update with hashes.
    audit["sha256"] = {
        "pool_10M": sha256_file(OUT_POOL),
        "row_meta": sha256_file(OUT_ROW_META),
        "pair_list": sha256_file(OUT_PAIR_LIST),
    }
    OUT_AUDIT.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_note(audit)
    print(json.dumps({
        "status": audit["status"],
        "pool": str(OUT_POOL),
        "audit": str(OUT_AUDIT),
        "note": str(OUT_NOTE),
        "replacement_count": len(swap_rows),
        "pool_words": pool_words,
        "row_lengths_identical": row_lengths_identical,
        "total_pair_token_delta": token_delta["total_pair_token_delta"],
        "full_visibility_rate": vis["full_pair_visible_rate"],
        "additions_with_shared_7gram": cont["additions"]["pairs_with_any_shared_7gram_to_globalpiqa"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
