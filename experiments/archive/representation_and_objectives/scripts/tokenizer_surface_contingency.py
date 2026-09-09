#!/usr/bin/env python3
"""Exact-tokenizer surface contingency for corrected results.

CPU-only. No model inference. Uses official evaluation text only to interpret
future score movement, not to choose training data or tokenizer vocabulary.
It compares the inherited out-of-budget 16k tokenizer against the exact
Strict-Small 10M-trained tokenizer (SHA 4a95a2a2...) on:
  * the allowed reinvest 10M pool used to train the tokenizer/model;
  * current official evaluation text surfaces by family and EWoK domain;
  * word-level fragmentation for high-frequency evaluation words.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import collections
import csv
import json
import math
import re
import statistics
import time
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoTokenizer

HERE = _public_path('experiments/archive/representation_and_objectives/scripts/tokenizer_surface_contingency.py')
A01 = _public_path('experiments/archive/representation_and_objectives')
USER_ROOT = _public_path('.')
WORKSPACE = _public_path('experiments/archive/representation_and_objectives')
OUT_DIR = _public_path('experiments/archive/representation_and_objectives/data/tokenizer_surface_contingency')
OUT_JSON = _public_path('experiments/archive/representation_and_objectives/data/tokenizer_surface_contingency/tokenizer_surface_contingency.json')
OUT_MD = _public_path('research/documents/representation_and_objectives/data/tokenizer_surface_contingency/tokenizer_surface_contingency.md')
OLD_TOKENIZER = _public_path('experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model')
NEW_TOKENIZER = _public_path('experiments/archive/representation_and_objectives/data/strictsmall_tokenizer_retrain/strictsmall_compact_reinvest_16k_tokenizer')
EXPECTED_NEW_SHA = "4a95a2a2ead21813a409c5ad7c2c008fbf418dcf3e0dc17bd970bd9ce25ea738"
REINVEST_POOL = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl')
PRISTINE_FULL = _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval')
GLOBALPIQA_FULL = _public_path('experiments/archive/representation_and_objectives/data/globalpiqa_official_lineage/official_dl_scratch/generated_by_current_official_dl/evaluation_data/full_eval')
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\-]{1,}")


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: Path | str) -> str:
    p = Path(p)
    try:
        return str(p.relative_to(USER_ROOT))
    except Exception:
        return str(p)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def token_strings(tok, text: str) -> list[str]:
    ids = tok(str(text), add_special_tokens=False, truncation=False)["input_ids"]
    return tok.convert_ids_to_tokens(ids)


def q(vals: list[float], p: float) -> float:
    if not vals:
        return float("nan")
    vals = sorted(vals)
    return float(vals[min(len(vals) - 1, max(0, round((len(vals) - 1) * p)))])


def summarize(vals: list[float]) -> dict[str, Any]:
    if not vals:
        return {"n": 0}
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


def source_group(obj: dict[str, Any]) -> str:
    src = str(obj.get("source", obj.get("source_name", "unknown")))
    if src == "cleanqwen_fineweb_compact_view_reinvest":
        return "reinvest_changed_block"
    if src.startswith("neutral_cleanqwen_topup_compact_reinvest"):
        return "reinvest_neutral_topup"
    if src == "qwen_pair_packed":
        return "inherited_qwen_pairs"
    if src in {"childes", "gutenberg", "open_subtitles", "simple_wiki", "bnc_spoken", "switchboard"}:
        return f"shared_filler::{src}"
    return f"other::{src}"


class Acc:
    def __init__(self, old_vocab: set[str], new_vocab: set[str]):
        self.shared = old_vocab & new_vocab
        self.old_only = old_vocab - new_vocab
        self.new_only = new_vocab - old_vocab
        self.groups: dict[str, dict[str, Any]] = {}

    def slot(self, group: str) -> dict[str, Any]:
        return self.groups.setdefault(group, {
            "group": group, "n_texts": 0, "total_words_field": 0,
            "old_total_tokens": 0, "new_total_tokens": 0,
            "old_shared_tokens": 0, "new_shared_tokens": 0,
            "old_only_tokens": 0, "new_only_tokens": 0,
            "delta_lens": [], "old_only_counter": collections.Counter(), "new_only_counter": collections.Counter(),
        })

    def add(self, group: str, old_tokens: list[str], new_tokens: list[str], words_field: int | None = None) -> None:
        s = self.slot(group)
        s["n_texts"] += 1
        if words_field is not None:
            s["total_words_field"] += int(words_field)
        s["old_total_tokens"] += len(old_tokens)
        s["new_total_tokens"] += len(new_tokens)
        s["old_shared_tokens"] += sum(1 for t in old_tokens if t in self.shared)
        s["new_shared_tokens"] += sum(1 for t in new_tokens if t in self.shared)
        oo = [t for t in old_tokens if t in self.old_only]
        no = [t for t in new_tokens if t in self.new_only]
        s["old_only_tokens"] += len(oo)
        s["new_only_tokens"] += len(no)
        s["old_only_counter"].update(oo)
        s["new_only_counter"].update(no)
        s["delta_lens"].append(len(new_tokens) - len(old_tokens))

    @staticmethod
    def finish_counter(c: collections.Counter, limit: int = 25) -> list[dict[str, Any]]:
        return [{"token": k, "count": int(v)} for k, v in c.most_common(limit)]

    def finish(self) -> list[dict[str, Any]]:
        rows = []
        for group, s in sorted(self.groups.items()):
            old_total = max(1, s["old_total_tokens"])
            new_total = max(1, s["new_total_tokens"])
            rows.append({
                "group": group,
                "n_texts": s["n_texts"],
                "total_words_field": s["total_words_field"],
                "old_total_tokens": s["old_total_tokens"],
                "new_total_tokens": s["new_total_tokens"],
                "new_over_old_token_ratio": s["new_total_tokens"] / old_total,
                "old_shared_occ_fraction": s["old_shared_tokens"] / old_total,
                "new_shared_occ_fraction": s["new_shared_tokens"] / new_total,
                "old_only_occ_fraction": s["old_only_tokens"] / old_total,
                "new_only_occ_fraction": s["new_only_tokens"] / new_total,
                "delta_len": summarize([float(x) for x in s["delta_lens"]]),
                "top_old_only_tokens": self.finish_counter(s["old_only_counter"]),
                "top_new_only_tokens": self.finish_counter(s["new_only_counter"]),
            })
        return rows


def update_words(counter: collections.Counter, text: str) -> None:
    for m in WORD_RE.finditer(text):
        w = m.group(0).lower()
        if len(w) > 1:
            counter[w] += 1


def collect_eval_records() -> Iterable[dict[str, Any]]:
    for family, dname in [("BLiMP", "blimp_filtered"), ("Supplement", "supplement_filtered")]:
        for p in sorted((PRISTINE_FULL / dname).glob("*.jsonl")):
            sub = p.stem
            for i, obj in enumerate(iter_jsonl(p)):
                for role, key in [("good", "sentence_good"), ("bad", "sentence_bad")]:
                    if key in obj:
                        yield {"family": family, "subtask": sub, "role": role, "row": i, "text": str(obj[key])}

    for p in sorted((_public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/ewok_filtered')).glob("*.jsonl")):
        sub = p.stem
        for i, obj in enumerate(iter_jsonl(p)):
            for key in ["Context1", "Context2", "Target1", "Target2"]:
                if key in obj:
                    yield {"family": "EWoK", "subtask": sub, "role": key, "row": i, "text": str(obj[key])}
            if all(k in obj for k in ["Context1", "Target1", "Context2", "Target2"]):
                yield {"family": "EWoK_concat", "subtask": sub, "role": "ctx1_target1", "row": i, "text": str(obj["Context1"]) + " " + str(obj["Target1"])}
                yield {"family": "EWoK_concat", "subtask": sub, "role": "ctx2_target2", "row": i, "text": str(obj["Context2"]) + " " + str(obj["Target2"])}

    for p in sorted((_public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/entity_tracking')).glob("*.jsonl")):
        sub = p.stem
        for i, obj in enumerate(iter_jsonl(p)):
            prefix = str(obj.get("input_prefix", ""))
            yield {"family": "Entity", "subtask": sub, "role": "prefix", "row": i, "text": prefix}
            for j, opt in enumerate(obj.get("options", [])):
                yield {"family": "Entity", "subtask": sub, "role": f"prefix_option{j}", "row": i, "text": prefix + str(opt)}

    for p in sorted((_public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/comps')).glob("*.jsonl")):
        sub = p.stem
        for i, obj in enumerate(iter_jsonl(p)):
            prop = str(obj.get("property_phrase", obj.get("property", "")))
            for role, key in [("acceptable", "prefix_acceptable"), ("unacceptable", "prefix_unacceptable")]:
                if key in obj:
                    yield {"family": "COMPS", "subtask": sub, "role": role, "row": i, "text": str(obj[key]) + " " + prop}

    for family, dname in [("GlobalPIQA_parallel", "global_piqa_parallel"), ("GlobalPIQA_nonparallel", "global_piqa_nonparallel")]:
        for p in sorted((GLOBALPIQA_FULL / dname).glob("*.jsonl")):
            for i, obj in enumerate(iter_jsonl(p)):
                prompt = str(obj.get("prompt", ""))
                yield {"family": family, "subtask": p.stem, "role": "prompt", "row": i, "text": prompt}
                for k, v in obj.items():
                    if k.startswith("solution") and isinstance(v, str):
                        yield {"family": family, "subtask": p.stem, "role": k, "row": i, "text": prompt + " " + v}

    rp = _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/reading/reading_data.csv')
    with rp.open("r", encoding="utf-8", errors="replace", newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            sent = str(row.get("sentence", ""))
            word = str(row.get("word", ""))
            if sent:
                yield {"family": "Reading_sentence", "subtask": "reading_data", "role": "sentence", "row": i, "text": sent}
            if word:
                yield {"family": "Reading_word", "subtask": "reading_data", "role": "word", "row": i, "text": word}

    for p in sorted((_public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/glue_filtered')).glob("*.jsonl")):
        parts = p.name.split(".")
        task = parts[0]
        split = parts[1] if len(parts) > 2 else "unknown"
        for i, obj in enumerate(iter_jsonl(p)):
            fields = [str(v) for k, v in obj.items() if k != "label" and isinstance(v, str)]
            if fields:
                yield {"family": "SuperGLUE", "subtask": task, "split": split, "role": "all_text_fields", "row": i, "text": " </s> ".join(fields)}


def analyze_pool(old_tok, new_tok, old_vocab: set[str], new_vocab: set[str]) -> tuple[list[dict[str, Any]], dict[str, collections.Counter]]:
    acc = Acc(old_vocab, new_vocab)
    words: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for n, obj in enumerate(iter_jsonl(REINVEST_POOL), 1):
        text = str(obj.get("text", ""))
        wf = int(obj.get("words", len(text.split())))
        group = source_group(obj)
        old = token_strings(old_tok, text)
        new = token_strings(new_tok, text)
        acc.add(group, old, new, wf)
        acc.add("ALL", old, new, wf)
        update_words(words[group], text)
        update_words(words["ALL"], text)
        if n % 10000 == 0:
            print(json.dumps({"event": "pool_progress", "rows": n}), flush=True)
    return acc.finish(), words


def analyze_eval(old_tok, new_tok, old_vocab: set[str], new_vocab: set[str]) -> tuple[list[dict[str, Any]], dict[str, collections.Counter], dict[str, collections.Counter]]:
    acc = Acc(old_vocab, new_vocab)
    words: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    ewok_domain_words: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for n, rec in enumerate(collect_eval_records(), 1):
        text = str(rec["text"])
        family = rec["family"]
        sub = rec["subtask"]
        old = token_strings(old_tok, text)
        new = token_strings(new_tok, text)
        acc.add(f"family::{family}", old, new)
        acc.add("ALL", old, new)
        update_words(words["ALL"], text)
        update_words(words[f"family::{family}"], text)
        if family in {"EWoK", "EWoK_concat"}:
            acc.add(f"ewok::{sub}", old, new)
            update_words(ewok_domain_words[sub], text)
        if n % 50000 == 0:
            print(json.dumps({"event": "eval_progress", "records": n}), flush=True)
    return acc.finish(), words, ewok_domain_words


def len_for_word(tok, word: str) -> int:
    return len(tok(" " + word, add_special_tokens=False, truncation=False)["input_ids"])


def toks_for_word(tok, word: str) -> list[str]:
    ids = tok(" " + word, add_special_tokens=False, truncation=False)["input_ids"]
    return tok.convert_ids_to_tokens(ids)


def word_fragmentation(old_tok, new_tok, train_counts: collections.Counter, eval_counts: collections.Counter, group: str, min_eval: int = 2, limit: int = 40) -> dict[str, Any]:
    rows = []
    for word, evc in eval_counts.items():
        if evc < min_eval or len(word) < 3:
            continue
        ol = len_for_word(old_tok, word)
        nl = len_for_word(new_tok, word)
        if ol == nl:
            continue
        rows.append({
            "word": word,
            "eval_count": int(evc),
            "train_count": int(train_counts.get(word, 0)),
            "old_len": ol,
            "new_len": nl,
            "delta_len_new_minus_old": nl - ol,
            "old_tokens": toks_for_word(old_tok, word),
            "new_tokens": toks_for_word(new_tok, word),
        })
    longer = sorted([r for r in rows if r["delta_len_new_minus_old"] > 0], key=lambda r: (r["delta_len_new_minus_old"] * r["eval_count"], r["eval_count"], -r["train_count"]), reverse=True)[:limit]
    shorter = sorted([r for r in rows if r["delta_len_new_minus_old"] < 0], key=lambda r: ((-r["delta_len_new_minus_old"]) * r["eval_count"], r["eval_count"], -r["train_count"]), reverse=True)[:limit]
    return {
        "group": group,
        "unique_eval_words": len(eval_counts),
        "total_eval_word_occurrences": int(sum(eval_counts.values())),
        "changed_word_types": len(rows),
        "eval_weighted_extra_pieces_new_longer": int(sum(r["eval_count"] * r["delta_len_new_minus_old"] for r in rows if r["delta_len_new_minus_old"] > 0)),
        "eval_weighted_extra_pieces_new_shorter": int(sum(r["eval_count"] * (-r["delta_len_new_minus_old"]) for r in rows if r["delta_len_new_minus_old"] < 0)),
        "top_new_longer_eval_words": longer,
        "top_new_shorter_eval_words": shorter,
    }


def row_by_group(rows: list[dict[str, Any]], group: str) -> dict[str, Any] | None:
    for r in rows:
        if r.get("group") == group:
            return r
    return None


def selected_occurrence_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keep = []
    for r in rows:
        g = r["group"]
        if g == "ALL" or g in {"reinvest_changed_block", "inherited_qwen_pairs", "reinvest_neutral_topup"} or g.startswith("shared_filler::") or g.startswith("family::") or g.startswith("ewok::"):
            rr = {k: v for k, v in r.items() if k not in {"top_old_only_tokens", "top_new_only_tokens"}}
            rr["top_old_only_tokens"] = r["top_old_only_tokens"][:12]
            rr["top_new_only_tokens"] = r["top_new_only_tokens"][:12]
            keep.append(rr)
    return keep


def write_md(payload: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# research — A01 exact tokenizer surface contingency\n\n")
    lines.append("CPU-only tokenizer comparison; no model inference. Official evaluation text is used only to interpret later corrected-tokenizer score movement, not to choose data or vocabulary.\n\n")
    v = payload["vocab"]
    lines.append(f"A01 compliant tokenizer SHA: `{payload['new_tokenizer_sha256']}`. Shared token strings with inherited tokenizer: {v['shared_token_strings']}/{v['new_vocab_size']} = {v['shared_fraction']:.6f}.\n\n")
    def table(title: str, rows: list[dict[str, Any]], groups: list[str]) -> None:
        by = {r["group"]: r for r in rows}
        lines.append(f"## {title}\n\n")
        lines.append("| group | texts | words | new/old tokens | old-only % | new-only % | mean token Δ | top old-only | top new-only |\n")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---|---|\n")
        for g in groups:
            r = by.get(g)
            if not r:
                continue
            old_top = ", ".join(f"`{x['token']}`:{x['count']}" for x in r["top_old_only_tokens"][:5])
            new_top = ", ".join(f"`{x['token']}`:{x['count']}" for x in r["top_new_only_tokens"][:5])
            lines.append(f"| {g} | {r['n_texts']} | {r['total_words_field']} | {r['new_over_old_token_ratio']:.4f} | {100*r['old_only_occ_fraction']:.2f} | {100*r['new_only_occ_fraction']:.2f} | {r['delta_len']['mean']:.3f} | {old_top} | {new_top} |\n")
        lines.append("\n")
    train_groups = ["ALL", "reinvest_changed_block", "inherited_qwen_pairs", "shared_filler::childes", "shared_filler::gutenberg", "shared_filler::open_subtitles", "shared_filler::simple_wiki", "shared_filler::bnc_spoken", "shared_filler::switchboard"]
    eval_groups = ["ALL", "family::BLiMP", "family::Supplement", "family::EWoK", "family::EWoK_concat", "family::Entity", "family::COMPS", "family::GlobalPIQA_parallel", "family::GlobalPIQA_nonparallel", "family::Reading_sentence", "family::Reading_word", "family::SuperGLUE"]
    table("Allowed reinvest 10M pool", payload["reinvest_pool_occurrence_rows"], train_groups)
    table("Current official evaluation text surface", payload["eval_occurrence_rows"], eval_groups)
    lines.append("## EWoK domain surface\n\n")
    lines.append("| domain | texts | new/old tokens | old-only % | new-only % | mean token Δ |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    for r in payload["eval_occurrence_rows"]:
        if r["group"].startswith("ewok::"):
            lines.append(f"| {r['group'].replace('ewok::','')} | {r['n_texts']} | {r['new_over_old_token_ratio']:.4f} | {100*r['old_only_occ_fraction']:.2f} | {100*r['new_only_occ_fraction']:.2f} | {r['delta_len']['mean']:.3f} |\n")
    lines.append("\n")
    lines.append("## Word-level fragmentation examples\n\n")
    for key, r in payload["word_fragmentation"].items():
        lines.append(f"### {key}\n\n")
        lines.append(f"Changed word types {r['changed_word_types']}/{r['unique_eval_words']}; weighted extra pieces when A01 tokenizer is longer: {r['eval_weighted_extra_pieces_new_longer']}; weighted saved pieces when shorter: {r['eval_weighted_extra_pieces_new_shorter']}.\n\n")
        lines.append("| word | eval count | train count | old len | A01 len | old tokens | A01 tokens |\n")
        lines.append("|---|---:|---:|---:|---:|---|---|\n")
        for x in r["top_new_longer_eval_words"][:10]:
            lines.append(f"| {x['word']} | {x['eval_count']} | {x['train_count']} | {x['old_len']} | {x['new_len']} | `{x['old_tokens']}` | `{x['new_tokens']}` |\n")
        lines.append("\n")
    lines.append("## Use after corrected full official scores arrive\n\n")
    lines.append("Compare actual score movement to this exact A01 tokenizer surface. Broad score losses with small tokenization pressure point to changed MLM target geometry or training dynamics under the legal vocabulary rather than crude truncation. Localized losses in families/domains with large old-only occurrence or concrete-word fragmentation make legal tokenizer-learning research plausible, but any such research must use only allowed 10M text and generic tokenizer objectives.\n")
    OUT_MD.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    old_tok = AutoTokenizer.from_pretrained(str(OLD_TOKENIZER), use_fast=True)
    new_tok = AutoTokenizer.from_pretrained(str(NEW_TOKENIZER), use_fast=True)
    old_vocab = set(old_tok.get_vocab())
    new_vocab = set(new_tok.get_vocab())
    print(json.dumps({"event": "loaded_tokenizers", "old_vocab": len(old_vocab), "new_vocab": len(new_vocab)}), flush=True)
    pool_rows, train_words = analyze_pool(old_tok, new_tok, old_vocab, new_vocab)
    print(json.dumps({"event": "pool_done", "groups": len(pool_rows)}), flush=True)
    eval_rows, eval_words, ewok_words = analyze_eval(old_tok, new_tok, old_vocab, new_vocab)
    print(json.dumps({"event": "eval_done", "groups": len(eval_rows)}), flush=True)
    train_all = train_words["ALL"]
    frag: dict[str, Any] = {}
    for group in ["ALL", "family::EWoK", "family::Supplement", "family::GlobalPIQA_parallel", "family::GlobalPIQA_nonparallel", "family::COMPS", "family::SuperGLUE"]:
        if group in eval_words:
            frag[group] = word_fragmentation(old_tok, new_tok, train_all, eval_words[group], group, min_eval=2)
    for domain in ["spatial-relations", "material-dynamics", "physical-dynamics", "physical-interactions", "physical-relations", "material-properties", "social-relations"]:
        if domain in ewok_words:
            frag[f"ewok_domain::{domain}"] = word_fragmentation(old_tok, new_tok, train_all, ewok_words[domain], f"ewok_domain::{domain}", min_eval=1, limit=30)
    payload = {
        "status": "A01_TOKENIZER_SURFACE_CONTINGENCY",
        "created_utc": now_utc(),
        "purpose": "Exact A01 tokenizer surface for interpreting corrected-tokenizer full official scores; no model inference and no benchmark-specific tokenizer design.",
        "old_tokenizer": rel(OLD_TOKENIZER),
        "new_tokenizer": rel(NEW_TOKENIZER),
        "new_tokenizer_sha256": EXPECTED_NEW_SHA,
        "reinvest_pool": rel(REINVEST_POOL),
        "official_eval_root": rel(PRISTINE_FULL),
        "globalpiqa_eval_root": rel(GLOBALPIQA_FULL),
        "vocab": {
            "old_vocab_size": len(old_vocab),
            "new_vocab_size": len(new_vocab),
            "shared_token_strings": len(old_vocab & new_vocab),
            "shared_fraction": len(old_vocab & new_vocab) / len(new_vocab) if new_vocab else math.nan,
            "old_only_token_strings": len(old_vocab - new_vocab),
            "new_only_token_strings": len(new_vocab - old_vocab),
        },
        "reinvest_pool_occurrence_rows": selected_occurrence_rows(pool_rows),
        "eval_occurrence_rows": selected_occurrence_rows(eval_rows),
        "word_fragmentation": frag,
        "elapsed_sec": round(time.time() - t0, 3),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_md(payload)
    key_groups = {g: row_by_group(payload["eval_occurrence_rows"], g) for g in ["ALL", "family::EWoK", "family::COMPS", "family::GlobalPIQA_parallel", "family::GlobalPIQA_nonparallel", "family::SuperGLUE"]}
    print(json.dumps({
        "status": payload["status"],
        "out_json": rel(OUT_JSON),
        "out_md": rel(OUT_MD),
        "elapsed_sec": payload["elapsed_sec"],
        "vocab_shared_fraction": payload["vocab"]["shared_fraction"],
        "key_eval_new_over_old": {k: (v or {}).get("new_over_old_token_ratio") for k, v in key_groups.items()},
        "key_eval_old_only_pct": {k: 100*(v or {}).get("old_only_occ_fraction", 0.0) for k, v in key_groups.items()},
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
