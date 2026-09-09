#!/usr/bin/env python3
"""research: tokenizer-learning contingency evidence for compliant retrain interpretation.

CPU-only evidence while the compliant-tokenizer model retrain is managed by the
runtime. This script does not train or evaluate a model and must not be used to
select benchmark-specific tokenizer vocabulary. It quantifies whether replacing
the old 100M-trained tokenizer by the legal 10M-trained tokenizer creates a
large, simple token-fragmentation pressure on the actual allowed pool or on the
official evaluation text surface. The scientific use is contingency planning:
if the compliant endpoint lands below the visible leader, decide whether a
legal tokenizer-learning repair is plausible, or whether the loss is more likely
from changed pretraining target geometry / model learning dynamics.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import collections
import csv
import json
import pathlib
import re
import statistics
import time
from typing import Any, Iterable

from transformers import AutoTokenizer

USER_ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/frontier_consolidation')
OUT = _public_path('experiments/archive/frontier_consolidation/data/tokenizer_learning_contingency')
OLD = _public_path('experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model')
NEW = _public_path('experiments/archive/frontier_consolidation/data/compliant_tokenizer')
REINVEST_POOL = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl')
CLEAN_POOL = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl')
PRISTINE_FULL = _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval')
GLOBALPIQA_FULL = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval')
SEQ_SENTENCE = 256
SEQ_FINETUNE = 512
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\-]{1,}")

EVAL_SENSITIVE_FAMILIES = {
    "BLiMP",
    "Supplement",
    "EWoK",
    "EWoK_concat",
    "Entity",
    "COMPS",
    "GlobalPIQA_parallel",
    "GlobalPIQA_nonparallel",
    "Reading_sentence",
    "Reading_word",
    "SuperGLUE",
}


def iter_jsonl(path: pathlib.Path) -> Iterable[dict[str, Any]]:
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


def summarize(vals: list[float]) -> dict[str, float]:
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


def collect_eval_records() -> Iterable[dict[str, Any]]:
    for family, dname in [("BLiMP", "blimp_filtered"), ("Supplement", "supplement_filtered")]:
        d = PRISTINE_FULL / dname
        for p in sorted(d.glob("*.jsonl")):
            sub = p.stem
            for i, obj in enumerate(iter_jsonl(p)):
                for role, key in [("good", "sentence_good"), ("bad", "sentence_bad")]:
                    if key in obj:
                        yield {"family": family, "subtask": sub, "split": "full_eval", "role": role, "row": i, "text": str(obj[key])}

    d = _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/ewok_filtered')
    for p in sorted(d.glob("*.jsonl")):
        sub = p.stem
        for i, obj in enumerate(iter_jsonl(p)):
            for key in ["Context1", "Context2", "Target1", "Target2"]:
                if key in obj:
                    yield {"family": "EWoK", "subtask": sub, "split": "full_eval", "role": key, "row": i, "text": str(obj[key])}
            if all(k in obj for k in ["Context1", "Target1", "Context2", "Target2"]):
                yield {"family": "EWoK_concat", "subtask": sub, "split": "full_eval", "role": "ctx1_target1", "row": i, "text": str(obj["Context1"]) + " " + str(obj["Target1"])}
                yield {"family": "EWoK_concat", "subtask": sub, "split": "full_eval", "role": "ctx2_target2", "row": i, "text": str(obj["Context2"]) + " " + str(obj["Target2"])}

    d = _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/entity_tracking')
    for p in sorted(d.glob("*.jsonl")):
        sub = p.stem
        for i, obj in enumerate(iter_jsonl(p)):
            prefix = str(obj.get("input_prefix", ""))
            yield {"family": "Entity", "subtask": sub, "split": "full_eval", "role": "input_prefix", "row": i, "text": prefix}
            for j, opt in enumerate(obj.get("options", [])):
                yield {"family": "Entity", "subtask": sub, "split": "full_eval", "role": f"prefix_option{j}", "row": i, "text": prefix + str(opt)}

    d = _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/comps')
    for p in sorted(d.glob("*.jsonl")):
        sub = p.stem
        for i, obj in enumerate(iter_jsonl(p)):
            prop = str(obj.get("property_phrase", obj.get("property", "")))
            for role, pk in [("acceptable", "prefix_acceptable"), ("unacceptable", "prefix_unacceptable")]:
                if pk in obj:
                    yield {"family": "COMPS", "subtask": sub, "split": "full_eval", "role": role, "row": i, "text": str(obj[pk]) + " " + prop}

    for family, dname in [("GlobalPIQA_parallel", "global_piqa_parallel"), ("GlobalPIQA_nonparallel", "global_piqa_nonparallel")]:
        d = GLOBALPIQA_FULL / dname
        for p in sorted(d.glob("*.jsonl")):
            for i, obj in enumerate(iter_jsonl(p)):
                prompt = str(obj.get("prompt", ""))
                yield {"family": family, "subtask": p.stem, "split": "full_eval", "role": "prompt", "row": i, "text": prompt}
                for k, v in obj.items():
                    if k.startswith("solution") and isinstance(v, str):
                        yield {"family": family, "subtask": p.stem, "split": "full_eval", "role": k, "row": i, "text": prompt + " " + v}

    rp = _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/reading/reading_data.csv')
    with rp.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            sent = str(row.get("sentence", ""))
            word = str(row.get("word", ""))
            if sent:
                yield {"family": "Reading_sentence", "subtask": "reading_data", "split": "full_eval", "role": "sentence", "row": i, "text": sent}
            if word:
                yield {"family": "Reading_word", "subtask": "reading_data", "split": "full_eval", "role": "word", "row": i, "text": word}

    gd = _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/glue_filtered')
    for p in sorted(gd.glob("*.jsonl")):
        parts = p.name.split(".")
        task = parts[0]
        split = parts[1] if len(parts) > 2 else "unknown"
        for i, obj in enumerate(iter_jsonl(p)):
            fields: list[str] = []
            for k, v in obj.items():
                if k == "label":
                    continue
                if isinstance(v, str):
                    fields.append(v)
            if fields:
                yield {"family": "SuperGLUE", "subtask": task, "split": split, "role": "all_text_fields", "row": i, "text": " </s> ".join(fields)}


class OccurrenceAccumulator:
    def __init__(self, old_vocab: set[str], new_vocab: set[str]):
        self.old_vocab = old_vocab
        self.new_vocab = new_vocab
        self.shared = old_vocab & new_vocab
        self.old_only = old_vocab - new_vocab
        self.new_only = new_vocab - old_vocab
        self.groups: dict[str, dict[str, Any]] = {}

    def _slot(self, group: str) -> dict[str, Any]:
        return self.groups.setdefault(group, {
            "group": group,
            "n_texts": 0,
            "total_words_field": 0,
            "old_total_tokens": 0,
            "new_total_tokens": 0,
            "old_shared_tokens": 0,
            "new_shared_tokens": 0,
            "old_only_tokens": 0,
            "new_only_tokens": 0,
            "delta_lens": [],
            "old_only_counter": collections.Counter(),
            "new_only_counter": collections.Counter(),
        })

    def add(self, group: str, text: str, old_tokens: list[str], new_tokens: list[str], words_field: int | None = None) -> None:
        s = self._slot(group)
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
        s["delta_lens"].append(len(new_tokens) - len(old_tokens))
        s["old_only_counter"].update(oo)
        s["new_only_counter"].update(no)

    @staticmethod
    def _finish_counter(c: collections.Counter, limit: int = 40) -> list[dict[str, Any]]:
        out = []
        for tok, n in c.most_common(limit):
            out.append({"token": tok, "count": int(n)})
        return out

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
                "top_old_only_tokens": self._finish_counter(s["old_only_counter"]),
                "top_new_only_tokens": self._finish_counter(s["new_only_counter"]),
            })
        return rows


def update_word_counter(counter: collections.Counter, text: str) -> None:
    for m in WORD_RE.finditer(text):
        w = m.group(0)
        if len(w) > 1:
            counter[w.lower()] += 1


def analyze_pool_occurrence(old_tok, new_tok, old_vocab: set[str], new_vocab: set[str], path: pathlib.Path, pool_label: str) -> dict[str, Any]:
    acc = OccurrenceAccumulator(old_vocab, new_vocab)
    word_counts_by_group: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for obj in iter_jsonl(path):
        text = str(obj.get("text", ""))
        group = source_group(obj) if pool_label == "reinvest" else f"clean_qwen::{str(obj.get('source', obj.get('source_name', 'unknown')))}"
        old_tokens = token_strings(old_tok, text)
        new_tokens = token_strings(new_tok, text)
        acc.add(group, text, old_tokens, new_tokens, int(obj.get("words", len(text.split()))))
        update_word_counter(word_counts_by_group[group], text)
        # Add a broad aggregate without re-encoding.
        acc.add("ALL", text, old_tokens, new_tokens, int(obj.get("words", len(text.split()))))
        update_word_counter(word_counts_by_group["ALL"], text)
    return {"occurrence_rows": acc.finish(), "word_counts_by_group": {g: c for g, c in word_counts_by_group.items()}}


def analyze_eval_occurrence(old_tok, new_tok, old_vocab: set[str], new_vocab: set[str]) -> dict[str, Any]:
    acc = OccurrenceAccumulator(old_vocab, new_vocab)
    word_counts_by_group: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    ewok_domain_words: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for rec in collect_eval_records():
        text = str(rec["text"])
        old_tokens = token_strings(old_tok, text)
        new_tokens = token_strings(new_tok, text)
        family = rec["family"]
        subtask = rec["subtask"]
        acc.add(f"family::{family}", text, old_tokens, new_tokens)
        acc.add("ALL", text, old_tokens, new_tokens)
        if family in {"EWoK", "EWoK_concat"}:
            acc.add(f"ewok::{subtask}", text, old_tokens, new_tokens)
        update_word_counter(word_counts_by_group["ALL"], text)
        update_word_counter(word_counts_by_group[f"family::{family}"], text)
        if family in {"EWoK", "EWoK_concat"}:
            update_word_counter(ewok_domain_words[subtask], text)
    return {
        "occurrence_rows": acc.finish(),
        "word_counts_by_group": {g: c for g, c in word_counts_by_group.items()},
        "ewok_domain_words": ewok_domain_words,
    }


def len_for_word(tok, word: str) -> int:
    return len(tok(" " + word, add_special_tokens=False, truncation=False)["input_ids"])


def toks_for_word(tok, word: str) -> list[str]:
    ids = tok(" " + word, add_special_tokens=False, truncation=False)["input_ids"]
    return tok.convert_ids_to_tokens(ids)


def word_fragmentation_table(old_tok, new_tok, train_counts: collections.Counter, eval_counts: collections.Counter, group_name: str, min_eval: int = 2, limit: int = 80) -> dict[str, Any]:
    rows = []
    for word, evc in eval_counts.items():
        if evc < min_eval or len(word) < 3:
            continue
        old_len = len_for_word(old_tok, word)
        new_len = len_for_word(new_tok, word)
        if old_len == new_len:
            continue
        tc = train_counts.get(word, 0)
        rows.append({
            "word": word,
            "eval_count": int(evc),
            "train_count": int(tc),
            "old_len": old_len,
            "new_len": new_len,
            "delta_len_new_minus_old": new_len - old_len,
            "old_tokens": toks_for_word(old_tok, word),
            "new_tokens": toks_for_word(new_tok, word),
        })
    longer_new = sorted([r for r in rows if r["delta_len_new_minus_old"] > 0], key=lambda r: (r["delta_len_new_minus_old"], r["eval_count"], r["train_count"]), reverse=True)[:limit]
    shorter_new = sorted([r for r in rows if r["delta_len_new_minus_old"] < 0], key=lambda r: (-r["delta_len_new_minus_old"], r["eval_count"], r["train_count"]), reverse=True)[:limit]
    all_weight = sum(eval_counts.values())
    longer_mass = sum(r["eval_count"] * r["delta_len_new_minus_old"] for r in rows if r["delta_len_new_minus_old"] > 0)
    shorter_mass = sum(r["eval_count"] * (-r["delta_len_new_minus_old"]) for r in rows if r["delta_len_new_minus_old"] < 0)
    return {
        "group": group_name,
        "unique_eval_words": len(eval_counts),
        "total_eval_word_occurrences": int(all_weight),
        "changed_word_types": len(rows),
        "eval_weighted_extra_pieces_new_longer": int(longer_mass),
        "eval_weighted_extra_pieces_new_shorter": int(shorter_mass),
        "top_new_longer_eval_words": longer_new,
        "top_new_shorter_eval_words": shorter_new,
    }


def assemble_word_fragmentation(old_tok, new_tok, train_word_counts: dict[str, collections.Counter], eval_word_counts: dict[str, collections.Counter], ewok_domain_words: dict[str, collections.Counter]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    train_all = train_word_counts["ALL"]
    for group in ["ALL", "family::EWoK", "family::EWoK_concat", "family::Supplement", "family::GlobalPIQA_parallel", "family::GlobalPIQA_nonparallel", "family::SuperGLUE"]:
        if group in eval_word_counts:
            out[group] = word_fragmentation_table(old_tok, new_tok, train_all, eval_word_counts[group], group, min_eval=2)
    for domain in ["spatial-relations", "material-dynamics", "physical-dynamics", "physical-interactions", "physical-relations", "material-properties", "social-relations"]:
        if domain in ewok_domain_words:
            out[f"ewok_domain::{domain}"] = word_fragmentation_table(old_tok, new_tok, train_all, ewok_domain_words[domain], f"ewok_domain::{domain}", min_eval=1, limit=50)
    return out


def compact_occurrence_rows(rows: list[dict[str, Any]], keep_prefixes: tuple[str, ...] = ("ALL", "reinvest_changed_block", "inherited_qwen_pairs", "shared_filler::", "family::", "ewok::")) -> list[dict[str, Any]]:
    compact = []
    for r in rows:
        g = r["group"]
        if g == "ALL" or g == "reinvest_changed_block" or g == "inherited_qwen_pairs" or any(g.startswith(p) for p in keep_prefixes if p.endswith("::") or p.endswith("::")):
            pass
        # Simpler explicit keep test.
        keep = g in {"ALL", "reinvest_changed_block", "inherited_qwen_pairs", "reinvest_neutral_topup"} or g.startswith("shared_filler::") or g.startswith("family::") or g.startswith("ewok::") or g.startswith("clean_qwen::")
        if keep:
            nr = {k: v for k, v in r.items() if k not in {"top_old_only_tokens", "top_new_only_tokens"}}
            nr["top_old_only_tokens"] = r["top_old_only_tokens"][:15]
            nr["top_new_only_tokens"] = r["top_new_only_tokens"][:15]
            compact.append(nr)
    return compact


def write_md(payload: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# research — tokenizer-learning contingency evidence\n\n")
    lines.append("CPU-only analysis while the compliant retrain is pending. It does not train or evaluate a model. Official evaluation text is used only to interpret possible score movement, not to choose pretraining or tokenizer vocabulary.\n\n")

    lines.append("## Vocabulary and occurrence mass\n\n")
    vo = payload["vocab"]
    lines.append(f"Old and compliant vocabularies each contain {vo['old_vocab_size']} tokens; shared token strings: {vo['shared_token_strings']} ({vo['shared_fraction']:.3f}).\n\n")
    lines.append("The occurrence mass is more informative than raw vocabulary overlap: if old-only token occurrences are tiny on the allowed pool and evaluation surface, a below-leader score should not be attributed to simple missing old-tokenizer surface pieces.\n\n")

    def table(title: str, rows: list[dict[str, Any]], groups: list[str]) -> None:
        lines.append(f"### {title}\n\n")
        lines.append("| group | texts/rows | words field | new/old token ratio | old-only occurrence % | new-only occurrence % | mean token Δ | top old-only tokens | top new-only tokens |\n")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---|---|\n")
        by = {r["group"]: r for r in rows}
        for g in groups:
            if g not in by:
                continue
            r = by[g]
            old_top = ", ".join(f"`{x['token']}`:{x['count']}" for x in r["top_old_only_tokens"][:5])
            new_top = ", ".join(f"`{x['token']}`:{x['count']}" for x in r["top_new_only_tokens"][:5])
            lines.append(f"| {g} | {r['n_texts']} | {r['total_words_field']} | {r['new_over_old_token_ratio']:.4f} | {100*r['old_only_occ_fraction']:.2f} | {100*r['new_only_occ_fraction']:.2f} | {r['delta_len']['mean']:.3f} | {old_top} | {new_top} |\n")
        lines.append("\n")

    train_groups = ["ALL", "reinvest_changed_block", "inherited_qwen_pairs", "shared_filler::childes", "shared_filler::gutenberg", "shared_filler::open_subtitles", "shared_filler::simple_wiki", "shared_filler::bnc_spoken", "shared_filler::switchboard"]
    table("Allowed reinvest 10M pretraining pool", payload["reinvest_pool_occurrence_rows"], train_groups)
    eval_groups = ["ALL", "family::BLiMP", "family::Supplement", "family::EWoK", "family::EWoK_concat", "family::Entity", "family::COMPS", "family::GlobalPIQA_parallel", "family::GlobalPIQA_nonparallel", "family::Reading_sentence", "family::Reading_word", "family::SuperGLUE"]
    table("Official evaluation text surface", payload["eval_occurrence_rows"], eval_groups)

    lines.append("## EWoK domain tokenization pressure\n\n")
    lines.append("| domain | texts | new/old token ratio | old-only occurrence % | new-only occurrence % | mean token Δ |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    ewok = [r for r in payload["eval_occurrence_rows"] if r["group"].startswith("ewok::")]
    for r in sorted(ewok, key=lambda x: x["group"]):
        lines.append(f"| {r['group'].replace('ewok::','')} | {r['n_texts']} | {r['new_over_old_token_ratio']:.4f} | {100*r['old_only_occ_fraction']:.2f} | {100*r['new_only_occ_fraction']:.2f} | {r['delta_len']['mean']:.3f} |\n")
    lines.append("\n")

    lines.append("## Word-level fragmentation readout\n\n")
    wf = payload["word_fragmentation"]
    for key in ["ALL", "family::EWoK", "family::Supplement", "family::GlobalPIQA_parallel", "family::SuperGLUE", "ewok_domain::spatial-relations", "ewok_domain::material-dynamics", "ewok_domain::physical-dynamics"]:
        if key not in wf:
            continue
        r = wf[key]
        lines.append(f"### {key}\n\n")
        lines.append(f"Changed word types: {r['changed_word_types']} / {r['unique_eval_words']}; weighted extra pieces when compliant tokenizer is longer: {r['eval_weighted_extra_pieces_new_longer']}; weighted saved pieces when compliant tokenizer is shorter: {r['eval_weighted_extra_pieces_new_shorter']}.\n\n")
        lines.append("Top words longer under the compliant tokenizer:\n\n")
        lines.append("| word | eval count | train count | old len | new len | old tokens | new tokens |\n")
        lines.append("|---|---:|---:|---:|---:|---|---|\n")
        for x in r["top_new_longer_eval_words"][:12]:
            lines.append(f"| {x['word']} | {x['eval_count']} | {x['train_count']} | {x['old_len']} | {x['new_len']} | `{x['old_tokens']}` | `{x['new_tokens']}` |\n")
        lines.append("\n")

    lines.append("## Interpretation for a possible below-leader compliant result\n\n")
    lines.append("If the compliant endpoint loses only modestly, the tokenization evidence suggests continuing to official collation rather than attributing the movement to crude length/truncation. If it falls well below the visible leader, the first fork should be: compare whether losses concentrate in families/domains where this analysis shows unusually high old-only occurrence mass or word fragmentation. A concentration there would make legal tokenizer learning worth studying using only the 10M pool and generic tokenizer priors. A broad loss or a loss in domains with small tokenization pressure would point instead to changed MLM target geometry, optimizer/initialization dynamics, or the data mechanism itself under the new vocabulary.\n")

    (_public_path('research/documents/frontier_consolidation/data/tokenizer_learning_contingency/tokenizer_learning_contingency.md')).write_text("".join(lines), encoding="utf-8")


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    old_tok = AutoTokenizer.from_pretrained(str(OLD), use_fast=True)
    new_tok = AutoTokenizer.from_pretrained(str(NEW), use_fast=True)
    old_vocab = set(old_tok.get_vocab())
    new_vocab = set(new_tok.get_vocab())

    print(json.dumps({"event": "analyze_reinvest_pool", "path": str(REINVEST_POOL)}), flush=True)
    reinvest = analyze_pool_occurrence(old_tok, new_tok, old_vocab, new_vocab, REINVEST_POOL, "reinvest")
    print(json.dumps({"event": "analyze_eval_text", "path": str(PRISTINE_FULL)}), flush=True)
    ev = analyze_eval_occurrence(old_tok, new_tok, old_vocab, new_vocab)
    word_frag = assemble_word_fragmentation(old_tok, new_tok, reinvest["word_counts_by_group"], ev["word_counts_by_group"], ev["ewok_domain_words"])

    payload: dict[str, Any] = {
        "status": "TOKENIZER_LEARNING_CONTINGENCY",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "old_tokenizer": str(OLD),
        "new_compliant_tokenizer": str(NEW),
        "reinvest_pool": str(REINVEST_POOL),
        "official_eval_root": str(PRISTINE_FULL),
        "globalpiqa_eval_root": str(GLOBALPIQA_FULL),
        "vocab": {
            "old_vocab_size": len(old_vocab),
            "new_vocab_size": len(new_vocab),
            "shared_token_strings": len(old_vocab & new_vocab),
            "shared_fraction": len(old_vocab & new_vocab) / len(new_vocab),
            "old_only_token_strings": len(old_vocab - new_vocab),
            "new_only_token_strings": len(new_vocab - old_vocab),
        },
        "reinvest_pool_occurrence_rows": compact_occurrence_rows(reinvest["occurrence_rows"]),
        "eval_occurrence_rows": compact_occurrence_rows(ev["occurrence_rows"]),
        "word_fragmentation": word_frag,
        "elapsed_sec": None,
    }
    payload["elapsed_sec"] = round(time.time() - t0, 3)
    out_json = _public_path('experiments/archive/frontier_consolidation/data/tokenizer_learning_contingency/tokenizer_learning_contingency.json')
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_md(payload)
    print(json.dumps({
        "status": payload["status"],
        "elapsed_sec": payload["elapsed_sec"],
        "out_json": str(out_json),
        "out_md": str(_public_path('research/documents/frontier_consolidation/data/tokenizer_learning_contingency/tokenizer_learning_contingency.md')),
        "vocab_shared_fraction": payload["vocab"]["shared_fraction"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
