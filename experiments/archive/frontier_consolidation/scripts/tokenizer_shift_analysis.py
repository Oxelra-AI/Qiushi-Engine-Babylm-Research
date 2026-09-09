#!/usr/bin/env python3
"""research: analyze segmentation shift from old baseline16k to compliant16k.

This is a CPU diagnostic.  It does not train or evaluate a model.  It compares
how the old 100M-trained baseline tokenizer and the new 10M-trained compliant
tokenizer segment the actual reinvest and clean-Qwen 10M pools.  The goal is to
make later score shifts interpretable: extra subword fragmentation, seq256
truncation, and WWM group changes can move every official column even when the
text corpus and training recipe are frozen.
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import pathlib
import statistics
import time
from typing import Any, Iterable

from transformers import AutoTokenizer

ROOT = pathlib.Path("experiments/archive/frontier_consolidation")
OUT = ROOT / "data/tokenizer_shift_analysis"
OLD = pathlib.Path("experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model")
NEW = ROOT / "data/compliant_tokenizer"
POOLS = {
    "reinvest_10M": ROOT / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl",
    "clean_qwen_10M": pathlib.Path("experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl"),
}
SEQ = 256
RELATIONAL_WORDS = [
    "in", "on", "under", "over", "above", "below", "beside", "near", "between", "among", "inside", "outside", "through", "across", "around", "behind", "before", "after", "from", "to", "into", "onto", "within", "without",
    "because", "therefore", "causes", "caused", "cause", "causing", "effect", "result", "leads", "led", "during", "while", "when", "until", "then",
    "larger", "smaller", "heavier", "lighter", "higher", "lower", "faster", "slower", "more", "less", "same", "different",
]


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def is_word_start(tok: str) -> bool:
    return tok.startswith("Ġ") or tok.startswith("▁")


def group_count_from_tokens(tokens: list[str]) -> int:
    gid = -1
    for i, tok in enumerate(tokens):
        if gid < 0 or is_word_start(tok) or i == 0:
            gid += 1
    return max(0, gid + 1)


def q(vals: list[float], p: float) -> float:
    if not vals:
        return float("nan")
    vals = sorted(vals)
    idx = min(len(vals) - 1, max(0, round((len(vals) - 1) * p)))
    return float(vals[idx])


def summarize(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {}
    return {
        "n": len(vals),
        "mean": float(sum(vals) / len(vals)),
        "std": float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0,
        "min": float(min(vals)),
        "p05": q(vals, 0.05),
        "p50": q(vals, 0.50),
        "p95": q(vals, 0.95),
        "max": float(max(vals)),
    }


def iter_rows(path: pathlib.Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_vocab(tok) -> dict[str, int]:
    return tok.get_vocab()


def analyze_pool(name: str, path: pathlib.Path, old_tok, new_tok) -> dict[str, Any]:
    rows = 0
    words_total = 0
    row_words = []
    old_lens = []
    new_lens = []
    old_len_per_word = []
    new_len_per_word = []
    delta_len = []
    old_truncated = 0
    new_truncated = 0
    either_truncated = 0
    old_visible_groups = []
    new_visible_groups = []
    group_delta = []
    top_delta_examples: list[dict[str, Any]] = []
    source_counter = collections.Counter()

    for obj in iter_rows(path):
        rows += 1
        text = str(obj.get("text", ""))
        words = int(obj.get("words", len(text.split())))
        source_counter.update([str(obj.get("source", obj.get("source_name", "unknown"))).split("::")[-1]])
        words_total += words
        row_words.append(words)
        old_ids = old_tok(text, add_special_tokens=False, truncation=False)["input_ids"]
        new_ids = new_tok(text, add_special_tokens=False, truncation=False)["input_ids"]
        old_len = len(old_ids)
        new_len = len(new_ids)
        old_lens.append(old_len)
        new_lens.append(new_len)
        old_len_per_word.append(old_len / max(1, words))
        new_len_per_word.append(new_len / max(1, words))
        d = new_len - old_len
        delta_len.append(d)
        ot = old_len > SEQ
        nt = new_len > SEQ
        old_truncated += int(ot)
        new_truncated += int(nt)
        either_truncated += int(ot or nt)
        old_tokens_tr = old_tok.convert_ids_to_tokens(old_ids[:SEQ])
        new_tokens_tr = new_tok.convert_ids_to_tokens(new_ids[:SEQ])
        og = group_count_from_tokens([t for t in old_tokens_tr if t not in set(old_tok.all_special_tokens)])
        ng = group_count_from_tokens([t for t in new_tokens_tr if t not in set(new_tok.all_special_tokens)])
        old_visible_groups.append(og)
        new_visible_groups.append(ng)
        group_delta.append(ng - og)
        if rows <= 10 or abs(d) >= 20 or (nt and not ot) or (ng - og) <= -10:
            rec = {
                "row_index": rows - 1,
                "example_id": obj.get("example_id"),
                "source": obj.get("source"),
                "words": words,
                "old_tokens": old_len,
                "new_tokens": new_len,
                "delta_tokens": d,
                "old_visible_groups_256": og,
                "new_visible_groups_256": ng,
                "delta_visible_groups_256": ng - og,
                "old_truncated_256": ot,
                "new_truncated_256": nt,
                "text_prefix": text[:220],
            }
            top_delta_examples.append(rec)
            top_delta_examples = sorted(top_delta_examples, key=lambda r: (abs(r["delta_tokens"]), r["new_truncated_256"] and not r["old_truncated_256"], -r["new_visible_groups_256"]), reverse=True)[:40]
    return {
        "pool": name,
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": rows,
        "words_total": words_total,
        "source_counter_top": source_counter.most_common(20),
        "row_words": summarize([float(x) for x in row_words]),
        "old_tokens": summarize([float(x) for x in old_lens]),
        "new_tokens": summarize([float(x) for x in new_lens]),
        "new_minus_old_tokens": summarize([float(x) for x in delta_len]),
        "old_tokens_per_word": summarize(old_len_per_word),
        "new_tokens_per_word": summarize(new_len_per_word),
        "token_per_word_ratio_new_over_old": (sum(new_lens) / sum(old_lens)) if sum(old_lens) else None,
        "old_truncated_rows_seq256": old_truncated,
        "new_truncated_rows_seq256": new_truncated,
        "new_only_truncated_rows_seq256": sum(1 for o, n in zip(old_lens, new_lens) if n > SEQ and o <= SEQ),
        "old_only_truncated_rows_seq256": sum(1 for o, n in zip(old_lens, new_lens) if o > SEQ and n <= SEQ),
        "either_truncated_rows_seq256": either_truncated,
        "old_visible_word_groups_seq256": summarize([float(x) for x in old_visible_groups]),
        "new_visible_word_groups_seq256": summarize([float(x) for x in new_visible_groups]),
        "new_minus_old_visible_word_groups_seq256": summarize([float(x) for x in group_delta]),
        "total_old_tokens": int(sum(old_lens)),
        "total_new_tokens": int(sum(new_lens)),
        "total_delta_tokens": int(sum(delta_len)),
        "top_delta_examples": top_delta_examples,
    }


def analyze_rel_words(old_tok, new_tok) -> list[dict[str, Any]]:
    out = []
    for w in RELATIONAL_WORDS:
        forms = [w, " " + w, w.capitalize(), " " + w.capitalize()]
        for form in forms:
            old_ids = old_tok(form, add_special_tokens=False)["input_ids"]
            new_ids = new_tok(form, add_special_tokens=False)["input_ids"]
            out.append({
                "form": form,
                "word": w,
                "old_ids": old_ids,
                "new_ids": new_ids,
                "old_tokens": old_tok.convert_ids_to_tokens(old_ids),
                "new_tokens": new_tok.convert_ids_to_tokens(new_ids),
                "old_len": len(old_ids),
                "new_len": len(new_ids),
                "delta_len": len(new_ids) - len(old_ids),
            })
    return out


def vocab_overlap(old_tok, new_tok) -> dict[str, Any]:
    ov = load_vocab(old_tok)
    nv = load_vocab(new_tok)
    old_set = set(ov)
    new_set = set(nv)
    shared = old_set & new_set
    specials = {"<unk>", "<s>", "</s>", "<pad>", "<mask>"}
    def first_tokens(v: dict[str, int], n: int = 50) -> list[tuple[str, int]]:
        return sorted(v.items(), key=lambda kv: kv[1])[:n]
    return {
        "old_vocab_size": len(ov),
        "new_vocab_size": len(nv),
        "shared_token_strings": len(shared),
        "shared_fraction_old": len(shared) / len(old_set),
        "shared_fraction_new": len(shared) / len(new_set),
        "special_ids_old": {s: ov.get(s) for s in specials},
        "special_ids_new": {s: nv.get(s) for s in specials},
        "first_50_old_by_id": first_tokens(ov, 50),
        "first_50_new_by_id": first_tokens(nv, 50),
        "new_only_sample_by_id": sorted([(t, nv[t]) for t in new_set - old_set], key=lambda kv: kv[1])[:200],
        "old_only_sample_by_id": sorted([(t, ov[t]) for t in old_set - new_set], key=lambda kv: kv[1])[:200],
    }


def write_md(payload: dict[str, Any], path: pathlib.Path) -> None:
    lines = ["# research — tokenizer shift analysis\n\n"]
    lines.append("Compares old 100M-trained baseline16k tokenizer to new compliant16k trained only on the 10M compact_view_reinvest pool. This is a CPU segmentation diagnostic, not model evidence.\n\n")
    vo = payload["vocab_overlap"]
    lines.append("## Vocabulary overlap\n\n")
    lines.append(f"- old vocab size: {vo['old_vocab_size']}\n")
    lines.append(f"- new vocab size: {vo['new_vocab_size']}\n")
    lines.append(f"- shared token strings: {vo['shared_token_strings']} ({vo['shared_fraction_new']:.3f} of new)\n")
    lines.append(f"- special IDs old: `{vo['special_ids_old']}`\n")
    lines.append(f"- special IDs new: `{vo['special_ids_new']}`\n\n")
    lines.append("## Pool-level segmentation\n\n")
    lines.append("| pool | rows | words | old tok/word | new tok/word | new/old token ratio | old trunc@256 | new trunc@256 | new-only trunc | visible group Δ mean | total token Δ |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for name, r in payload["pools"].items():
        lines.append(f"| {name} | {r['rows']} | {r['words_total']} | {r['old_tokens_per_word']['mean']:.4f} | {r['new_tokens_per_word']['mean']:.4f} | {r['token_per_word_ratio_new_over_old']:.4f} | {r['old_truncated_rows_seq256']} | {r['new_truncated_rows_seq256']} | {r['new_only_truncated_rows_seq256']} | {r['new_minus_old_visible_word_groups_seq256']['mean']:.4f} | {r['total_delta_tokens']} |\n")
    lines.append("\n## Relational marker segmentation changes\n\n")
    changed = [x for x in payload["relational_words"] if x["delta_len"] != 0 or x["old_tokens"] != x["new_tokens"]]
    lines.append(f"Changed relational forms: {len(changed)} / {len(payload['relational_words'])}.\n\n")
    lines.append("| form | old tokens | new tokens | Δlen |\n|---|---|---|---:|\n")
    for x in changed[:80]:
        form = x["form"].replace(" ", "␠")
        lines.append(f"| `{form}` | `{x['old_tokens']}` | `{x['new_tokens']}` | {x['delta_len']} |\n")
    lines.append("\nMachine-readable JSON contains top token-delta example rows for each pool.\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    old_tok = AutoTokenizer.from_pretrained(str(OLD), use_fast=True)
    new_tok = AutoTokenizer.from_pretrained(str(NEW), use_fast=True)
    payload: dict[str, Any] = {
        "status": "TOKENIZER_SHIFT_ANALYSIS",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "old_tokenizer": str(OLD),
        "new_tokenizer": str(NEW),
        "seq_length": SEQ,
        "vocab_overlap": vocab_overlap(old_tok, new_tok),
        "pools": {},
        "relational_words": analyze_rel_words(old_tok, new_tok),
    }
    for name, path in POOLS.items():
        print(json.dumps({"event": "pool_start", "pool": name, "path": str(path)}), flush=True)
        payload["pools"][name] = analyze_pool(name, path, old_tok, new_tok)
        print(json.dumps({"event": "pool_done", "pool": name, "token_ratio": payload['pools'][name]['token_per_word_ratio_new_over_old'], "new_trunc": payload['pools'][name]['new_truncated_rows_seq256']}), flush=True)
    payload["elapsed_sec"] = round(time.time() - t0, 3)
    out_json = OUT / "tokenizer_shift_analysis.json"
    out_md = (OUT.parents[4] / 'research/documents/frontier_consolidation/data/tokenizer_shift_analysis/tokenizer_shift_analysis.md')
    payload["out_json"] = str(out_json)
    payload["out_md"] = str(out_md)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_md(payload, out_md)
    print(json.dumps({
        "status": payload["status"],
        "elapsed_sec": payload["elapsed_sec"],
        "out_json": str(out_json),
        "out_md": str(out_md),
        "pool_summaries": {k: {"ratio": v["token_per_word_ratio_new_over_old"], "old_trunc": v["old_truncated_rows_seq256"], "new_trunc": v["new_truncated_rows_seq256"], "visible_group_delta_mean": v["new_minus_old_visible_word_groups_seq256"]["mean"]} for k, v in payload["pools"].items()},
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
