#!/usr/bin/env python3
"""research official-evaluation representation atlas.

CPU-only evidence builder while the two legal-40k full evaluations are managed
asynchronously.  It compares the compliant 16k and 40k tokenizers on the exact
text units used by the Strict-Small official MLM zero-shot ranking surfaces, plus
surface maps for SuperGLUE finetuning and reading-time text.

The atlas is not a model score and uses official evaluation text only after both
tokenizers were already trained on the allowed 10M pool.  Its purpose is to make
future legal40k score movement interpretable: shorter target strings may improve
visible context and reduce per-word prediction burden, but larger vocabulary also
puts some target tokens onto lower-count embedding/output rows learned from the
same 10M pool.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import heapq
import json
import math
import re
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from tokenizers import Tokenizer

ROOT = Path(".").resolve()
STUDY = ROOT / "experiments/archive/representation_and_objectives"
WS = STUDY
OUT = WS / "data/official_eval_representation_atlas"
NOTE = (ROOT / 'research/notes/representation_and_objectives/71_official_eval_representation_atlas.md')
PRISTINE_FULL = WS / "data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval"
GLOBALPIQA_FULL = WS / "data/globalpiqa_official_lineage/official_dl_scratch/generated_by_current_official_dl/evaluation_data/full_eval"
LEGAL16_DIR = WS / "data/strictsmall_tokenizer_retrain/strictsmall_compact_reinvest_16k_tokenizer"
LEGAL40_DIR = WS / "data/legal_representation_route_map/tokenizers/legal_byte_bpe_40k"
SUPPORT_JSON = WS / "data/tokenizer_support_spectrum/tokenizer_support_spectrum.json"
FAST_MAP_JSON = WS / "data/fast_representation_map/fast_representation_map.json"
FRONTIER_JSON = WS / "data/frontier_refresh_and_wait_state/frontier_refresh_and_wait_state.json"
OVERLAP_JSON = WS / "data/generated_view_overlap_scan/generated_view_overlap_scan.json"
POOL_10M = ROOT / "experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"
EXPECTED_POOL_SHA = "215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23"
BATCH_SIZE = 512

EXPECTED_TOKENIZER_JSON_SHA = {
    "legal16": "4a95a2a2ead21813a409c5ad7c2c008fbf418dcf3e0dc17bd970bd9ce25ea738",
    "legal40": "94b44bcf5901d097e20294e8220740758eac29af3ec20d58d8da867c85197758",
}
SUPPORT_LABEL = {"legal16": "legal_a01_16k", "legal40": "legal_byte_bpe_40k"}
SPECIAL_TOKENS = ["<unk>", "<s>", "</s>", "<pad>", "<mask>"]
LOW_SUPPORT_CUTS = [20, 50, 100, 200]
ZERO_SHOT_FAMILIES = {"BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA"}
LENGTH_NORMALIZED_ZERO_SHOT = {"global_piqa_parallel", "global_piqa_nonparallel"}
WORD_RE = re.compile(r"\S+")


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_raw_tokenizer(path: Path) -> Tokenizer:
    tok = Tokenizer.from_file(str(path / "tokenizer.json"))
    try:
        tok.no_padding()
    except Exception:
        pass
    try:
        tok.no_truncation()
    except Exception:
        pass
    return tok


def is_word_start(token: str) -> bool:
    return token.startswith("Ġ") or token.startswith("▁")


def safe_div(a: float, b: float) -> float | None:
    return (a / b) if b else None


@dataclass
class Agg:
    items: int = 0
    candidates: int = 0
    full_text_tokens: int = 0
    full_with_special_tokens: int = 0
    target_tokens: int = 0
    target_groups: int = 0
    target_support_sum: int = 0
    target_support_min: int | None = None
    target_support_zero: int = 0
    unk_target_tokens: int = 0
    max_full_text_tokens: int = 0
    max_target_tokens: int = 0
    full_imbalance_sum: int = 0
    full_imbalance_max: int = 0
    target_imbalance_sum: int = 0
    target_imbalance_max: int = 0
    zero_target_items: int = 0
    low_support: dict[int, int] = field(default_factory=lambda: {c: 0 for c in LOW_SUPPORT_CUTS})

    def add_candidate(self, st: dict[str, Any]) -> None:
        self.candidates += 1
        self.full_text_tokens += int(st["full_text_tokens"])
        self.full_with_special_tokens += int(st["full_with_special_tokens"])
        self.target_tokens += int(st["target_tokens"])
        self.target_groups += int(st["target_groups"])
        self.unk_target_tokens += int(st["unk_target_tokens"])
        self.max_full_text_tokens = max(self.max_full_text_tokens, int(st["full_text_tokens"]))
        self.max_target_tokens = max(self.max_target_tokens, int(st["target_tokens"]))
        supports = st["target_support_values"]
        if supports:
            self.target_support_sum += int(sum(supports))
            local_min = int(min(supports))
            self.target_support_min = local_min if self.target_support_min is None else min(self.target_support_min, local_min)
            self.target_support_zero += sum(1 for x in supports if int(x) == 0)
            for cut in LOW_SUPPORT_CUTS:
                self.low_support[cut] += sum(1 for x in supports if int(x) < cut)

    def add_item(self, full_lens: list[int], target_lens: list[int]) -> None:
        self.items += 1
        full_imb = (max(full_lens) - min(full_lens)) if full_lens else 0
        target_imb = (max(target_lens) - min(target_lens)) if target_lens else 0
        self.full_imbalance_sum += full_imb
        self.full_imbalance_max = max(self.full_imbalance_max, full_imb)
        self.target_imbalance_sum += target_imb
        self.target_imbalance_max = max(self.target_imbalance_max, target_imb)
        if not target_lens or sum(target_lens) == 0:
            self.zero_target_items += 1

    def to_record(self) -> dict[str, Any]:
        rec = {
            "items": self.items,
            "candidates": self.candidates,
            "avg_full_text_tokens_per_candidate": safe_div(self.full_text_tokens, self.candidates),
            "avg_full_with_special_tokens_per_candidate": safe_div(self.full_with_special_tokens, self.candidates),
            "avg_target_tokens_per_candidate": safe_div(self.target_tokens, self.candidates),
            "avg_target_groups_per_candidate": safe_div(self.target_groups, self.candidates),
            "avg_target_tokens_per_group": safe_div(self.target_tokens, self.target_groups),
            "target_support_mean": safe_div(self.target_support_sum, self.target_tokens),
            "target_support_min": self.target_support_min,
            "target_support_zero_frac": safe_div(self.target_support_zero, self.target_tokens),
            "unk_target_tokens": self.unk_target_tokens,
            "max_full_text_tokens": self.max_full_text_tokens,
            "max_target_tokens": self.max_target_tokens,
            "avg_candidate_full_length_imbalance": safe_div(self.full_imbalance_sum, self.items),
            "max_candidate_full_length_imbalance": self.full_imbalance_max,
            "avg_candidate_target_length_imbalance": safe_div(self.target_imbalance_sum, self.items),
            "max_candidate_target_length_imbalance": self.target_imbalance_max,
            "zero_target_items": self.zero_target_items,
        }
        for cut in LOW_SUPPORT_CUTS:
            rec[f"target_frac_support_lt_{cut}"] = safe_div(self.low_support[cut], self.target_tokens)
        return rec


def count_groups_for_indices(tokens: list[str], indices: list[int]) -> int:
    if not indices:
        return 0
    idxset = set(indices)
    groups = 0
    prev_in = False
    for i in indices:
        t = str(tokens[i])
        if not prev_in or is_word_start(t):
            groups += 1
        prev_in = True
    # If indices are not contiguous, the first token after a gap should start a new group.
    # The current official target spans are contiguous suffixes or single reading words;
    # keep a correction for the rare non-contiguous case.
    if len(indices) > 1:
        corr = 0
        for a, b in zip(indices, indices[1:]):
            if b != a + 1:
                corr += 1
        groups += corr
    return groups


def encode_text_stats(
    tok: Tokenizer,
    support_counts: list[int],
    text: str,
    completion: str | None = None,
    target_span: tuple[int, int] | None = None,
) -> dict[str, Any]:
    enc = tok.encode(text, add_special_tokens=True)
    ids = list(enc.ids)
    tokens = list(enc.tokens)
    offsets = list(enc.offsets)
    text_indices = [i for i, (s, e) in enumerate(offsets) if int(e) > int(s)]
    if target_span is not None:
        a, b = target_span
        target_indices = [i for i, (s, e) in enumerate(offsets) if int(e) > a and int(s) < b and int(e) > int(s)]
    elif completion is None:
        target_indices = text_indices
    else:
        start_char = max(0, len(text) - len(completion))
        target_indices = [i for i, (s, e) in enumerate(offsets) if int(e) > start_char and int(e) > int(s)]
    full_groups = count_groups_for_indices(tokens, text_indices)
    target_groups = count_groups_for_indices(tokens, target_indices)
    target_ids = [int(ids[i]) for i in target_indices]
    target_support = [int(support_counts[x]) if 0 <= int(x) < len(support_counts) else 0 for x in target_ids]
    unk_id = tok.token_to_id("<unk>")
    return {
        "full_text_tokens": len(text_indices),
        "full_with_special_tokens": len(ids),
        "full_groups": full_groups,
        "target_tokens": len(target_indices),
        "target_groups": target_groups,
        "target_ids": target_ids,
        "target_token_strings": [tokens[i] for i in target_indices],
        "target_support_values": target_support,
        "unk_target_tokens": sum(1 for x in target_ids if unk_id is not None and x == int(unk_id)),
    }


def reading_target_span(sentence: str, context_length: int, target_word: str) -> tuple[int, int] | None:
    spans = list(WORD_RE.finditer(sentence))
    if 0 <= context_length < len(spans):
        m = spans[context_length]
        return (m.start(), m.end())
    # Fallback: first exact occurrence.
    idx = sentence.find(target_word)
    if idx >= 0:
        return (idx, idx + len(target_word))
    return None


def load_eval_items() -> Iterable[dict[str, Any]]:
    # BLiMP and Supplement share the BLiMP-style official decoder.
    for family, dname in [("BLiMP", "blimp_filtered"), ("Supplement", "supplement_filtered")]:
        d = PRISTINE_FULL / dname
        for p in sorted(d.glob("*.jsonl")):
            for line_idx, obj in enumerate(iter_jsonl(p), 1):
                yield {
                    "family": family,
                    "subtask": p.stem,
                    "source_file": str(p),
                    "line": line_idx,
                    "task_type": "mlm_zero_shot_rawsum",
                    "length_normalized": False,
                    "sentences": [str(obj["sentence_good"]), str(obj["sentence_bad"])],
                    "completions": [str(obj["sentence_good"]), str(obj["sentence_bad"])],
                    "target_spans": [None, None],
                    "metadata": {k: obj.get(k) for k in ["field", "UID", "linguistics_term", "template", "ambiguous"] if k in obj},
                }

    # EWoK official read_files.py scores Target1 under Context1 versus Context2.
    d = PRISTINE_FULL / "ewok_filtered"
    for p in sorted(d.glob("*.jsonl")):
        for line_idx, obj in enumerate(iter_jsonl(p), 1):
            t = str(obj["Target1"])
            yield {
                "family": "EWoK",
                "subtask": str(obj.get("Domain", p.stem)),
                "source_file": str(p),
                "line": line_idx,
                "task_type": "mlm_zero_shot_rawsum",
                "length_normalized": False,
                "sentences": [f"{obj['Context1']} {t}", f"{obj['Context2']} {t}"],
                "completions": [" " + t, " " + t],
                "target_spans": [None, None],
                "metadata": {
                    "context_type": obj.get("ContextType"),
                    "context_contrast": obj.get("ContextDiff"),
                    "target_contrast": obj.get("TargetDiff"),
                },
            }

    # Entity tracking official decoder skips rows with any 'nothing' answer option.
    d = PRISTINE_FULL / "entity_tracking"
    for p in sorted(d.glob("*.jsonl")):
        for line_idx, obj in enumerate(iter_jsonl(p), 1):
            opts = [str(x) for x in obj.get("options", [])]
            if any("nothing" in opt for opt in opts):
                continue
            prefix = str(obj.get("input_prefix", ""))
            yield {
                "family": "Entity",
                "subtask": f"{p.stem}_{obj.get('numops')}_ops",
                "source_file": str(p),
                "line": line_idx,
                "task_type": "mlm_zero_shot_rawsum",
                "length_normalized": False,
                "sentences": [prefix + opt for opt in opts],
                "completions": opts,
                "target_spans": [None for _ in opts],
                "metadata": {"numops": obj.get("numops"), "sample_id": obj.get("sample_id")},
            }

    # COMPS official decoder.
    comps_map = {
        "comps_base": "base",
        "comps_wugs": "wugs",
        "comps_wugs_dist-before": "wugs_dist_before",
        "comps_wugs_dist-in-between": "wugs_dist_in_between",
    }
    d = PRISTINE_FULL / "comps"
    for p in sorted(d.glob("*.jsonl")):
        for line_idx, obj in enumerate(iter_jsonl(p), 1):
            prop = str(obj["property_phrase"])
            acc = " ".join([str(obj["prefix_acceptable"]), prop])
            unacc = " ".join([str(obj["prefix_unacceptable"]), prop])
            yield {
                "family": "COMPS",
                "subtask": comps_map.get(p.stem, p.stem),
                "source_file": str(p),
                "line": line_idx,
                "task_type": "mlm_zero_shot_rawsum",
                "length_normalized": False,
                "sentences": [acc, unacc],
                "completions": [prop, prop],
                "target_spans": [None, None],
                "metadata": {"negative_sample_type": obj.get("negative_sample_type"), "property": obj.get("property")},
            }

    # GlobalPIQA official files generated in research; official scoring length-normalizes.
    for dname, nsol in [("global_piqa_parallel", 4), ("global_piqa_nonparallel", 2)]:
        d = GLOBALPIQA_FULL / dname
        for p in sorted(d.glob("*.jsonl")):
            for line_idx, obj in enumerate(iter_jsonl(p), 1):
                prompt = str(obj["prompt"])
                sols = [str(obj[f"solution{i}"]) for i in range(nsol)]
                yield {
                    "family": "GlobalPIQA",
                    "subtask": dname,
                    "source_file": str(p),
                    "line": line_idx,
                    "task_type": "mlm_zero_shot_length_normalized",
                    "length_normalized": True,
                    "sentences": [" ".join([prompt, s]) for s in sols],
                    "completions": [" " + s for s in sols],
                    "target_spans": [None for _ in sols],
                    "metadata": {"example_id": obj.get("example_id"), "label": obj.get("label")},
                }

    # Reading-time target words: record the target word span from context_length.
    rp = PRISTINE_FULL / "reading/reading_data.csv"
    if rp.exists():
        with rp.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            for line_idx, row in enumerate(reader, 2):
                sent = str(row.get("sentence") or "")
                word = str(row.get("word") or "")
                if not sent or not word:
                    continue
                try:
                    clen = int(float(row.get("context_length") or 0))
                except Exception:
                    clen = 0
                span = reading_target_span(sent, clen, word)
                yield {
                    "family": "Reading",
                    "subtask": "reading_word",
                    "source_file": str(rp),
                    "line": line_idx,
                    "task_type": "reading_surface_word",
                    "length_normalized": False,
                    "sentences": [sent],
                    "completions": [None],
                    "target_spans": [span],
                    "metadata": {"word": word, "context_length": clen, "sent_id": row.get("sent_id")},
                }

    # SuperGLUE surface map for finetuning text.  This is not zero-shot MLM target scoring.
    gd = PRISTINE_FULL / "glue_filtered"
    for p in sorted(gd.glob("*.jsonl")):
        task = p.name.split(".")[0]
        split = p.name.split(".")[1] if "." in p.name else "unknown"
        for line_idx, obj in enumerate(iter_jsonl(p), 1):
            fields = [str(v) for k, v in obj.items() if k != "label" and isinstance(v, str) and str(v).strip()]
            if not fields:
                continue
            text = " </s> ".join(fields)
            yield {
                "family": "SuperGLUE",
                "subtask": task,
                "source_file": str(p),
                "line": line_idx,
                "task_type": "finetune_surface_all_text",
                "length_normalized": False,
                "sentences": [text],
                "completions": [None],
                "target_spans": [None],
                "metadata": {"split": split},
            }


def compact_preview(text: str, n: int = 180) -> str:
    s = " ".join(text.split())
    return s[:n] + ("..." if len(s) > n else "")


def heap_add(heap: list[tuple[float, int, dict[str, Any]]], score: float, counter: int, rec: dict[str, Any], limit: int = 80) -> None:
    if not math.isfinite(score):
        return
    item = (score, counter, rec)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif score > heap[0][0]:
        heapq.heapreplace(heap, item)


def load_support_counts() -> dict[str, list[int]]:
    obj = json.loads(SUPPORT_JSON.read_text(encoding="utf-8"))
    out: dict[str, list[int]] = {}
    for short, label in SUPPORT_LABEL.items():
        out[short] = [int(x) for x in obj["pool_support"][label]["counts_by_id"]]
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    tokenizers = {"legal16": load_raw_tokenizer(LEGAL16_DIR), "legal40": load_raw_tokenizer(LEGAL40_DIR)}
    support_counts = load_support_counts()
    tokenizer_records = {}
    errors = []
    for label, d in [("legal16", LEGAL16_DIR), ("legal40", LEGAL40_DIR)]:
        tok_file = d / "tokenizer.json"
        sha = sha256_file(tok_file)
        tokenizer_records[label] = {
            "dir": str(d),
            "tokenizer_json_sha256": sha,
            "expected_tokenizer_json_sha256": EXPECTED_TOKENIZER_JSON_SHA[label],
            "sha_matches_expected": sha == EXPECTED_TOKENIZER_JSON_SHA[label],
            "vocab_size": tokenizers[label].get_vocab_size(),
            "special_ids": {s: tokenizers[label].token_to_id(s) for s in SPECIAL_TOKENS},
        }
        if sha != EXPECTED_TOKENIZER_JSON_SHA[label]:
            errors.append(f"{label} tokenizer SHA mismatch: {sha}")
    if errors:
        raise RuntimeError("; ".join(errors))

    family_aggs: dict[tuple[str, str], Agg] = collections.defaultdict(Agg)  # (tokenizer,family)
    subtask_aggs: dict[tuple[str, str, str], Agg] = collections.defaultdict(Agg)
    task_type_counts: collections.Counter[str] = collections.Counter()
    family_item_counts: collections.Counter[str] = collections.Counter()
    family_candidate_counts: collections.Counter[str] = collections.Counter()
    compare_family = collections.defaultdict(lambda: {
        "items": 0,
        "candidates": 0,
        "delta_full_text_tokens_sum": 0,
        "delta_target_tokens_sum": 0,
        "target_tokens_16_sum": 0,
        "target_tokens_40_sum": 0,
        "delta_target_imbalance_sum": 0,
        "delta_full_imbalance_sum": 0,
        "items_target_shorter_40": 0,
        "items_target_longer_40": 0,
        "items_target_same_40": 0,
        "items_imbalance_increased_40": 0,
        "items_imbalance_decreased_40": 0,
        "items_low50_frac_higher_40": 0,
    })
    compare_subtask = collections.defaultdict(lambda: {
        "items": 0,
        "candidates": 0,
        "delta_full_text_tokens_sum": 0,
        "delta_target_tokens_sum": 0,
        "target_tokens_16_sum": 0,
        "target_tokens_40_sum": 0,
        "delta_target_imbalance_sum": 0,
        "delta_full_imbalance_sum": 0,
        "items_target_shorter_40": 0,
        "items_target_longer_40": 0,
        "items_target_same_40": 0,
        "items_imbalance_increased_40": 0,
        "items_imbalance_decreased_40": 0,
        "items_low50_frac_higher_40": 0,
    })
    extremes = {
        "largest_target_token_reduction_40_vs_16": [],
        "largest_target_low50_fraction_legal40": [],
        "largest_target_imbalance_increase_legal40": [],
        "largest_full_length_reduction_40_vs_16": [],
    }

    item_counter = 0
    for item in load_eval_items():
        item_counter += 1
        family = item["family"]
        subtask = item["subtask"]
        task_type_counts[item["task_type"]] += 1
        family_item_counts[family] += 1
        family_candidate_counts[family] += len(item["sentences"])
        per_tok_candidate_stats: dict[str, list[dict[str, Any]]] = {"legal16": [], "legal40": []}
        for label, tok in tokenizers.items():
            fam_agg = family_aggs[(label, family)]
            sub_agg = subtask_aggs[(label, family, subtask)]
            full_lens: list[int] = []
            target_lens: list[int] = []
            for sent, comp, span in zip(item["sentences"], item["completions"], item["target_spans"]):
                st = encode_text_stats(tok, support_counts[label], sent, comp, span)
                per_tok_candidate_stats[label].append(st)
                fam_agg.add_candidate(st)
                sub_agg.add_candidate(st)
                full_lens.append(int(st["full_text_tokens"]))
                target_lens.append(int(st["target_tokens"]))
            fam_agg.add_item(full_lens, target_lens)
            sub_agg.add_item(full_lens, target_lens)

        s16 = per_tok_candidate_stats["legal16"]
        s40 = per_tok_candidate_stats["legal40"]
        full16 = sum(int(x["full_text_tokens"]) for x in s16)
        full40 = sum(int(x["full_text_tokens"]) for x in s40)
        tgt16 = sum(int(x["target_tokens"]) for x in s16)
        tgt40 = sum(int(x["target_tokens"]) for x in s40)
        imb16 = (max([int(x["target_tokens"]) for x in s16]) - min([int(x["target_tokens"]) for x in s16])) if s16 else 0
        imb40 = (max([int(x["target_tokens"]) for x in s40]) - min([int(x["target_tokens"]) for x in s40])) if s40 else 0
        fimb16 = (max([int(x["full_text_tokens"]) for x in s16]) - min([int(x["full_text_tokens"]) for x in s16])) if s16 else 0
        fimb40 = (max([int(x["full_text_tokens"]) for x in s40]) - min([int(x["full_text_tokens"]) for x in s40])) if s40 else 0
        low50_16 = sum(sum(1 for v in x["target_support_values"] if int(v) < 50) for x in s16)
        low50_40 = sum(sum(1 for v in x["target_support_values"] if int(v) < 50) for x in s40)
        low50_frac_16 = safe_div(low50_16, tgt16) or 0.0
        low50_frac_40 = safe_div(low50_40, tgt40) or 0.0
        for key in [(family,), (family, subtask)]:
            d = compare_family[key[0]] if len(key) == 1 else compare_subtask[(key[0], key[1])]
            d["items"] += 1
            d["candidates"] += len(item["sentences"])
            d["delta_full_text_tokens_sum"] += full40 - full16
            d["delta_target_tokens_sum"] += tgt40 - tgt16
            d["target_tokens_16_sum"] += tgt16
            d["target_tokens_40_sum"] += tgt40
            d["delta_target_imbalance_sum"] += imb40 - imb16
            d["delta_full_imbalance_sum"] += fimb40 - fimb16
            if tgt40 < tgt16:
                d["items_target_shorter_40"] += 1
            elif tgt40 > tgt16:
                d["items_target_longer_40"] += 1
            else:
                d["items_target_same_40"] += 1
            if imb40 > imb16:
                d["items_imbalance_increased_40"] += 1
            elif imb40 < imb16:
                d["items_imbalance_decreased_40"] += 1
            if low50_frac_40 > low50_frac_16:
                d["items_low50_frac_higher_40"] += 1

        base_rec = {
            "family": family,
            "subtask": subtask,
            "task_type": item["task_type"],
            "source_file": item["source_file"],
            "line": item["line"],
            "length_normalized": item["length_normalized"],
            "candidate_count": len(item["sentences"]),
            "text_preview": compact_preview(item["sentences"][0]),
            "metadata": json.dumps(item.get("metadata", {}), ensure_ascii=False, sort_keys=True),
            "target_tokens_legal16": tgt16,
            "target_tokens_legal40": tgt40,
            "delta_target_tokens_40_minus_16": tgt40 - tgt16,
            "full_text_tokens_legal16": full16,
            "full_text_tokens_legal40": full40,
            "delta_full_text_tokens_40_minus_16": full40 - full16,
            "target_imbalance_legal16": imb16,
            "target_imbalance_legal40": imb40,
            "delta_target_imbalance_40_minus_16": imb40 - imb16,
            "target_low50_frac_legal16": low50_frac_16,
            "target_low50_frac_legal40": low50_frac_40,
            "legal40_target_token_strings": " ".join(str(t) for x in s40 for t in x["target_token_strings"][:12])[:240],
            "legal40_target_support_values_head": " ".join(str(v) for x in s40 for v in x["target_support_values"][:12])[:240],
        }
        heap_add(extremes["largest_target_token_reduction_40_vs_16"], float(tgt16 - tgt40), item_counter, base_rec)
        heap_add(extremes["largest_target_low50_fraction_legal40"], float(low50_frac_40), item_counter, base_rec)
        heap_add(extremes["largest_target_imbalance_increase_legal40"], float(imb40 - imb16), item_counter, base_rec)
        heap_add(extremes["largest_full_length_reduction_40_vs_16"], float(full16 - full40), item_counter, base_rec)

    def agg_record(label: str, family: str, subtask: str | None, agg: Agg) -> dict[str, Any]:
        rec = {"tokenizer": label, "family": family}
        if subtask is not None:
            rec["subtask"] = subtask
        rec.update(agg.to_record())
        return rec

    family_rows = []
    for (label, family), agg in sorted(family_aggs.items()):
        family_rows.append(agg_record(label, family, None, agg))
    subtask_rows = []
    for (label, family, subtask), agg in sorted(subtask_aggs.items()):
        subtask_rows.append(agg_record(label, family, subtask, agg))

    def finish_compare(d: dict[str, Any]) -> dict[str, Any]:
        items = d["items"]
        out = dict(d)
        out["avg_delta_full_text_tokens_per_item"] = safe_div(d["delta_full_text_tokens_sum"], items)
        out["avg_delta_target_tokens_per_item"] = safe_div(d["delta_target_tokens_sum"], items)
        out["target_token_ratio_40_over_16"] = safe_div(d["target_tokens_40_sum"], d["target_tokens_16_sum"])
        out["avg_delta_target_imbalance_per_item"] = safe_div(d["delta_target_imbalance_sum"], items)
        out["avg_delta_full_imbalance_per_item"] = safe_div(d["delta_full_imbalance_sum"], items)
        out["frac_items_target_shorter_40"] = safe_div(d["items_target_shorter_40"], items)
        out["frac_items_target_longer_40"] = safe_div(d["items_target_longer_40"], items)
        out["frac_items_target_same_40"] = safe_div(d["items_target_same_40"], items)
        out["frac_items_imbalance_increased_40"] = safe_div(d["items_imbalance_increased_40"], items)
        out["frac_items_imbalance_decreased_40"] = safe_div(d["items_imbalance_decreased_40"], items)
        out["frac_items_low50_frac_higher_40"] = safe_div(d["items_low50_frac_higher_40"], items)
        return out

    compare_family_rows = [{"family": fam, **finish_compare(d)} for fam, d in sorted(compare_family.items())]
    compare_subtask_rows = [{"family": fam, "subtask": sub, **finish_compare(d)} for (fam, sub), d in sorted(compare_subtask.items())]
    extreme_rows = []
    for kind, heap in extremes.items():
        for score, _counter, rec in sorted(heap, key=lambda x: x[0], reverse=True):
            rr = {"extreme_kind": kind, "extreme_score": score}
            rr.update(rec)
            extreme_rows.append(rr)

    # Extract a small context from earlier map files so this result connects to the
    # active frontier and representation lineage without re-reading model outputs.
    lineages: dict[str, Any] = {
        "fast_map_path": str(FAST_MAP_JSON),
        "support_spectrum_path": str(SUPPORT_JSON),
        "frontier_path": str(FRONTIER_JSON),
        "overlap_path": str(OVERLAP_JSON),
    }
    if FAST_MAP_JSON.exists():
        fm = json.loads(FAST_MAP_JSON.read_text(encoding="utf-8"))
        lineages["pool_stats"] = {
            "legal_a01_16k": fm.get("pool_stats", {}).get("legal_a01_16k"),
            "legal_byte_bpe_40k": fm.get("pool_stats", {}).get("legal_byte_bpe_40k"),
        }
    if FRONTIER_JSON.exists():
        fr = json.loads(FRONTIER_JSON.read_text(encoding="utf-8"))
        lineages["frontier_anchor"] = fr.get("frontier_anchor") or fr.get("leader") or fr.get("current_target")

    payload = {
        "status": "OFFICIAL_EVAL_REPRESENTATION_ATLAS",
        "created_utc": now_utc(),
        "elapsed_sec": round(time.time() - t0, 3),
        "purpose": "Interpret legal16 versus legal40 representation changes on official Strict-Small text units before reading pending legal40 model scores.",
        "not_a_model_score": True,
        "pending_evaluation_tasks_not_inspected": ["s65_t22_tool1", "s66_t9_tool1"],
        "paths": {
            "pristine_full_eval": str(PRISTINE_FULL),
            "globalpiqa_full_eval": str(GLOBALPIQA_FULL),
            "legal16_tokenizer_dir": str(LEGAL16_DIR),
            "legal40_tokenizer_dir": str(LEGAL40_DIR),
            "support_json": str(SUPPORT_JSON),
        },
        "tokenizers": tokenizer_records,
        "item_counts_by_family": dict(sorted(family_item_counts.items())),
        "candidate_counts_by_family": dict(sorted(family_candidate_counts.items())),
        "task_type_counts": dict(sorted(task_type_counts.items())),
        "family_summary": family_rows,
        "subtask_summary": subtask_rows,
        "family_legal40_vs_legal16": compare_family_rows,
        "subtask_legal40_vs_legal16": compare_subtask_rows,
        "extreme_examples": extreme_rows,
        "lineage_context": lineages,
    }
    out_json = OUT / "official_eval_representation_atlas.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        keys = []
        for r in rows:
            for k in r.keys():
                if k not in keys:
                    keys.append(k)
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    write_csv(OUT / "family_representation_summary.csv", family_rows)
    write_csv(OUT / "subtask_representation_summary.csv", subtask_rows)
    write_csv(OUT / "family_legal40_vs_legal16.csv", compare_family_rows)
    write_csv(OUT / "subtask_legal40_vs_legal16.csv", compare_subtask_rows)
    write_csv(OUT / "extreme_official_text_examples.csv", extreme_rows)

    # Compose a compact research note with scientific quantities usable without
    # loading the whole JSON.
    rows_by_family_tok = {(r["family"], r["tokenizer"]): r for r in family_rows}
    cmp_by_family = {r["family"]: r for r in compare_family_rows}
    lines: list[str] = []
    lines.append("# research official-evaluation representation atlas\n\n")
    lines.append("This CPU-only atlas compares the compliant legal16 and legal40 tokenizers on official Strict-Small text units while the two legal40 full evaluations remain managed asynchronously. It is not a model score and does not read pending evaluation outputs.\n\n")
    lines.append("## Tokenizer identities\n\n")
    for label, rec in tokenizer_records.items():
        lines.append(f"- {label}: vocab {rec['vocab_size']}, tokenizer.json SHA `{rec['tokenizer_json_sha256']}`, expected match={rec['sha_matches_expected']}.\n")
    lines.append("\n## Family-level target geometry\n\n")
    lines.append("| family | items | candidates | target ratio 40/16 | avg Δ target tokens/item | legal40 target <50 support frac | legal16 target <50 support frac | legal40 avg target tokens/cand | legal16 avg target tokens/cand |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for family in sorted(family_item_counts):
        c = cmp_by_family[family]
        r16 = rows_by_family_tok[(family, "legal16")]
        r40 = rows_by_family_tok[(family, "legal40")]
        lines.append(
            f"| {family} | {c['items']} | {c['candidates']} | {c['target_token_ratio_40_over_16']:.4f} | {c['avg_delta_target_tokens_per_item']:.3f} | "
            f"{(r40.get('target_frac_support_lt_50') or 0):.4f} | {(r16.get('target_frac_support_lt_50') or 0):.4f} | "
            f"{(r40.get('avg_target_tokens_per_candidate') or 0):.3f} | {(r16.get('avg_target_tokens_per_candidate') or 0):.3f} |\n"
        )
    lines.append("\n## Interpretation for the pending legal40 vectors\n\n")
    lines.append("- Legal40 shortens official target spans on every family where the two tokenizers differ, but the larger vocabulary also increases the share of target tokens supported by fewer than 50 occurrences in the exact 10M tokenizer/model pool.\n")
    lines.append("- EWoK and Supplement are the most important coming columns for the legal16→legal40 question. If they recover, this atlas supports a representation-burden/context explanation; if they do not, the support-thinning side of the tradeoff remains a plausible contributor and the next route should alter architecture or learning signal rather than repeat vocabulary-size-only sweeps.\n")
    lines.append("- Entity target strings show much smaller support-thinning than EWoK/COMPS/GlobalPIQA; if Entity moves strongly, look beyond target segmentation alone, especially sequence/state tracking and fine-tune/generation dynamics.\n")
    lines.append("- GlobalPIQA official scoring length-normalizes completion log-probability, so target-token count changes are not interpreted the same way as raw-sum zero-shot families. Its low-support rise under legal40 is still useful for explaining uncertainty, not as a direct prediction.\n")
    lines.append("\n## Output files\n\n")
    lines.append(f"- JSON: `{out_json}`\n")
    for name in ["family_representation_summary.csv", "subtask_representation_summary.csv", "family_legal40_vs_legal16.csv", "subtask_legal40_vs_legal16.csv", "extreme_official_text_examples.csv"]:
        lines.append(f"- `{OUT / name}`\n")
    NOTE.write_text("".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": payload["status"],
        "elapsed_sec": payload["elapsed_sec"],
        "items": item_counter,
        "families": dict(sorted(family_item_counts.items())),
        "out_json": str(out_json),
        "note": str(NOTE),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
