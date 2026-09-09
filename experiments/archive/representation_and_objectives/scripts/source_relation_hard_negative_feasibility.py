#!/usr/bin/env python3
"""research source-grounded relation hard-negative measurement.

CPU-only work over the already selected compact-view-reinvest 10M corpus and
already trained checkpoints.  It does not train a model, run the BabyLM suite,
or touch managed SGCR jobs.

Purpose: before any H100 run, test whether an evaluation-independent set of
source-sentence relation alternatives has (a) enough precise-looking examples
inside the actual corpus and (b) weak/negative existing MLM margins, so a future
hard-negative relation objective would have a real target instead of merely
renaming prior RTD or relation-mask ideas.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = Path("experiments/archive/representation_and_objectives")
WS = ROOT
CORPUS = Path("experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl")
OUT = WS / "data/source_relation_hard_negative_feasibility"
NOTE = (WS.parents[2] / 'research/notes/representation_and_objectives/source_relation_hard_negative_feasibility.md')

DEFAULT_MODELS = {
    "legal40_depth_12x384_43022": WS / "training/runs/legal40k_12x384_depth_compact_view_reinvest_seed43022/hf_model/chck_100M",
    "legal40_8x480_43022": WS / "training/runs/legal40k_accum_compact_view_reinvest_seed43022/hf_model/chck_100M",
}

TOKENIZER_PATH = WS / "data/legal_tokenizer_scale/legal_byte_bpe_40k"
WORD_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)?|\d+(?:[.,:/-]\d+)*(?:%|st|nd|rd|th)?")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
BAD_RE = re.compile(r"https?://|www\.|@|<[^>]+>|\$[A-Za-z]|\{\}|\[\]|\b(mysql|javascript|click|subscribe|copyright|license|cookie|email|password|username|download)\b", re.I)

# Frozen, evaluation-independent, hand-written relation alternatives.  It is
# intentionally smaller and cleaner than the broad research corpus-count list:
# no in/out, on/off, left/right, like/dislike, know/ignore, can/cannot, or
# EWoK file-derived concepts.  Each entry is a same-word-count directed edit.
PAIR_SPECS = [
    ("temporal_order", "before", "after"),
    ("temporal_order", "after", "before"),
    ("temporal_order", "earlier", "later"),
    ("temporal_order", "later", "earlier"),
    ("spatial_vertical", "above", "below"),
    ("spatial_vertical", "below", "above"),
    ("spatial_vertical", "over", "under"),
    ("spatial_vertical", "under", "over"),
    ("spatial_containment", "inside", "outside"),
    ("spatial_containment", "outside", "inside"),
    ("quantity_comparison", "more", "less"),
    ("quantity_comparison", "less", "more"),
    ("quantity_comparison", "larger", "smaller"),
    ("quantity_comparison", "smaller", "larger"),
    ("quantity_comparison", "higher", "lower"),
    ("quantity_comparison", "lower", "higher"),
    ("quantity_comparison", "increase", "decrease"),
    ("quantity_comparison", "decrease", "increase"),
    ("quantity_comparison", "increases", "decreases"),
    ("quantity_comparison", "decreases", "increases"),
    ("quantity_comparison", "increased", "decreased"),
    ("quantity_comparison", "decreased", "increased"),
    ("quantity_comparison", "increasing", "decreasing"),
    ("quantity_comparison", "decreasing", "increasing"),
    ("identity_state", "same", "different"),
    ("identity_state", "different", "same"),
    ("truth_state", "true", "false"),
    ("truth_state", "false", "true"),
    ("containment_membership", "include", "exclude"),
    ("containment_membership", "exclude", "include"),
    ("containment_membership", "includes", "excludes"),
    ("containment_membership", "excludes", "includes"),
    ("containment_membership", "included", "excluded"),
    ("containment_membership", "excluded", "included"),
    ("containment_membership", "including", "excluding"),
    ("containment_membership", "excluding", "including"),
    ("affordance_block", "allows", "prevents"),
    ("affordance_block", "prevents", "allows"),
    ("affordance_block", "allow", "prevent"),
    ("affordance_block", "prevent", "allow"),
    ("affordance_block", "allowed", "prevented"),
    ("affordance_block", "prevented", "allowed"),
    ("affordance_block", "allowing", "preventing"),
    ("affordance_block", "preventing", "allowing"),
    ("thermal_state", "hot", "cold"),
    ("thermal_state", "cold", "hot"),
    ("access_state", "open", "closed"),
    ("access_state", "closed", "open"),
    ("possibility", "possible", "impossible"),
    ("possibility", "impossible", "possible"),
]

PAIR_PATTERNS = [(group, a, b, re.compile(rf"\b{re.escape(a)}\b", re.I)) for group, a, b in PAIR_SPECS]
ALL_TERMS = sorted({a for _, a, _ in PAIR_SPECS} | {b for _, _, b in PAIR_SPECS}, key=len, reverse=True)
ALL_TERM_RE = re.compile(r"\b(" + "|".join(re.escape(x) for x in ALL_TERMS) + r")\b", re.I)


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def wc(text: str) -> int:
    return len(WORD_RE.findall(text))


def apply_case(src: str, repl: str) -> str:
    if src.isupper():
        return repl.upper()
    if src[:1].isupper():
        return repl[:1].upper() + repl[1:]
    return repl


def clean_sentence(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def sentence_ok(s: str, min_words: int, max_words: int) -> bool:
    n = wc(s)
    if n < min_words or n > max_words:
        return False
    if len(s) < 35 or len(s) > 260:
        return False
    if BAD_RE.search(s):
        return False
    if s.count('"') >= 2 or s.count("'") >= 4:
        return False
    alpha = sum(ch.isalpha() for ch in s)
    if alpha < max(20, 0.55 * len(s)):
        return False
    return True


@dataclass
class Candidate:
    candidate_id: int
    row_index: int
    sentence_index: int
    example_id: Any
    source: str
    group: str
    original: str
    replacement: str
    start: int
    end: int
    sentence: str
    corrupted: str
    words: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "row_index": self.row_index,
            "sentence_index": self.sentence_index,
            "example_id": self.example_id,
            "source": self.source,
            "group": self.group,
            "original": self.original,
            "replacement": self.replacement,
            "start": self.start,
            "end": self.end,
            "words": self.words,
            "sentence": self.sentence,
            "corrupted": self.corrupted,
        }


def candidate_from_sentence(s: str, row_index: int, sentence_index: int, row: dict[str, Any]) -> Candidate | None:
    s = clean_sentence(s)
    if not sentence_ok(s, 7, 38):
        return None
    # Keep sentences with exactly one safe relation term occurrence.  This is a
    # precision-first source for future small measurements; larger work can relax it.
    all_hits = list(ALL_TERM_RE.finditer(s))
    if len(all_hits) != 1:
        return None
    hit = all_hits[0]
    surface = hit.group(0)
    for group, a, b, pat in PAIR_PATTERNS:
        if surface.lower() != a.lower():
            continue
        # Reject if the replacement side is already present elsewhere in the sentence.
        if re.search(rf"\b{re.escape(b)}\b", s, re.I):
            return None
        repl = apply_case(surface, b)
        corrupted = s[:hit.start()] + repl + s[hit.end():]
        return Candidate(
            candidate_id=-1,
            row_index=row_index,
            sentence_index=sentence_index,
            example_id=row.get("example_id"),
            source=str(row.get("source", "")),
            group=group,
            original=surface,
            replacement=repl,
            start=hit.start(),
            end=hit.end(),
            sentence=s,
            corrupted=corrupted,
            words=wc(s),
        )
    return None


def collect_candidates(max_rows: int, max_per_group: int, seed: int) -> tuple[list[Candidate], dict[str, Any]]:
    rng = random.Random(seed)
    by_group: dict[str, list[Candidate]] = defaultdict(list)
    raw_counts = Counter()
    row_count = 0
    word_count = 0
    sent_seen = 0
    with CORPUS.open("r", encoding="utf-8") as f:
        for row_index, line in enumerate(f):
            if max_rows and row_index >= max_rows:
                break
            row = json.loads(line)
            row_count += 1
            word_count += int(row.get("words", wc(row.get("text", ""))))
            text = clean_sentence(str(row.get("text", "")))
            # The compact-reinvest rows often contain several source/rewrite sentences.
            parts = [clean_sentence(x) for x in SENT_SPLIT_RE.split(text) if clean_sentence(x)]
            for sent_idx, sent in enumerate(parts):
                sent_seen += 1
                cand = candidate_from_sentence(sent, row_index, sent_idx, row)
                if cand is None:
                    continue
                raw_counts[cand.group] += 1
                if len(by_group[cand.group]) < max_per_group:
                    by_group[cand.group].append(cand)
            if by_group and all(len(by_group[g]) >= max_per_group for g in {p[0] for p in PAIR_SPECS}):
                break
    candidates = []
    for group in sorted(by_group):
        vals = by_group[group]
        rng.shuffle(vals)
        vals = sorted(vals[:max_per_group], key=lambda c: (c.row_index, c.sentence_index, c.start))
        candidates.extend(vals)
    candidates.sort(key=lambda c: (c.group, c.row_index, c.sentence_index, c.start))
    for i, c in enumerate(candidates):
        c.candidate_id = i
    meta = {
        "corpus": str(CORPUS),
        "rows_scanned": row_count,
        "words_scanned": word_count,
        "sentences_seen": sent_seen,
        "raw_candidate_counts_by_group": dict(raw_counts),
        "kept_by_group": dict(Counter(c.group for c in candidates)),
        "kept_total": len(candidates),
        "max_per_group": max_per_group,
        "seed": seed,
    }
    return candidates, meta


def phrase_indices_for_span(tokenizer, text: str, start: int, end: int) -> tuple[list[int], list[int], list[int]]:
    enc = tokenizer(text, return_offsets_mapping=True, return_tensors=None)
    ids = list(enc["input_ids"])
    offsets = list(enc["offset_mapping"])
    attn = list(enc["attention_mask"])
    idxs = []
    for i, (a, b) in enumerate(offsets):
        if b > start and a < end:
            idxs.append(i)
    return ids, attn, idxs


def score_candidate(model, tokenizer, cand: Candidate, device: torch.device) -> dict[str, Any]:
    orig_ids, orig_attn, orig_idxs = phrase_indices_for_span(tokenizer, cand.sentence, cand.start, cand.end)
    repl_end = cand.start + len(cand.replacement)
    cor_ids, _cor_attn, cor_idxs = phrase_indices_for_span(tokenizer, cand.corrupted, cand.start, repl_end)
    ok = True
    reason = ""
    if not orig_idxs or not cor_idxs:
        ok = False; reason = "no_span_tokens"
    elif len(orig_idxs) != len(cor_idxs):
        ok = False; reason = "different_span_token_count"
    elif len(orig_ids) != len(cor_ids):
        ok = False; reason = "different_sentence_token_count"
    elif orig_idxs != cor_idxs:
        ok = False; reason = "different_span_positions"
    else:
        outside_same = True
        span_set = set(orig_idxs)
        for i, (a, b) in enumerate(zip(orig_ids, cor_ids)):
            if i not in span_set and a != b:
                outside_same = False
                break
        if not outside_same:
            ok = False; reason = "outside_tokenization_changed"
    out = {
        "span_scoreable": ok,
        "span_skip_reason": reason,
        "orig_span_token_count": len(orig_idxs),
        "replacement_span_token_count": len(cor_idxs),
        "sentence_token_count": len(orig_ids),
    }
    if not ok:
        return out
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        raise RuntimeError("tokenizer has no mask token")
    masked = list(orig_ids)
    for idx in orig_idxs:
        masked[idx] = mask_id
    input_ids = torch.tensor([masked], dtype=torch.long, device=device)
    attention_mask = torch.tensor([orig_attn], dtype=torch.long, device=device)
    target_orig = torch.tensor([orig_ids[i] for i in orig_idxs], dtype=torch.long, device=device)
    target_cor = torch.tensor([cor_ids[i] for i in cor_idxs], dtype=torch.long, device=device)
    idx_t = torch.tensor(orig_idxs, dtype=torch.long, device=device)
    with torch.no_grad():
        res = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = res.logits if hasattr(res, "logits") else res[0]
        selected = logits[0, idx_t]
        lp = F.log_softmax(selected, dim=-1)
        orig_lp = lp.gather(-1, target_orig.unsqueeze(-1)).squeeze(-1)
        cor_lp = lp.gather(-1, target_cor.unsqueeze(-1)).squeeze(-1)
    orig_sum = float(orig_lp.sum().detach().cpu())
    cor_sum = float(cor_lp.sum().detach().cpu())
    out.update({
        "orig_logprob_sum": orig_sum,
        "replacement_logprob_sum": cor_sum,
        "margin_orig_minus_replacement": orig_sum - cor_sum,
        "margin_per_token": (orig_sum - cor_sum) / max(1, len(orig_idxs)),
    })
    return out


def summarize(vals: list[float]) -> dict[str, Any]:
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "mean": statistics.fmean(vals),
        "median": statistics.median(vals),
        "p10": sorted(vals)[max(0, int(0.10 * (len(vals)-1)))],
        "p90": sorted(vals)[min(len(vals)-1, int(0.90 * (len(vals)-1)))],
        "positive_frac": sum(v > 0 for v in vals) / len(vals),
        "nonpositive_frac": sum(v <= 0 for v in vals) / len(vals),
        "near0_abs_lt_0p25_frac": sum(abs(v) < 0.25 for v in vals) / len(vals),
        "weak_le_1p0_frac": sum(v <= 1.0 for v in vals) / len(vals),
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    scoreable = [r for r in records if r.get("span_scoreable")]
    out = {
        "records": len(records),
        "scoreable": len(scoreable),
        "scoreable_frac": len(scoreable) / len(records) if records else None,
        "skip_reasons": dict(Counter(r.get("span_skip_reason", "") for r in records if not r.get("span_scoreable"))),
        "margin": summarize([float(r["margin_orig_minus_replacement"]) for r in scoreable]),
        "margin_per_token": summarize([float(r["margin_per_token"]) for r in scoreable]),
        "by_group": {},
    }
    for group in sorted({r["group"] for r in records}):
        gr = [r for r in records if r["group"] == group and r.get("span_scoreable")]
        out["by_group"][group] = summarize([float(r["margin_orig_minus_replacement"]) for r in gr])
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                keys.append(k); seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_rows", type=int, default=64740)
    ap.add_argument("--max_per_group", type=int, default=40)
    ap.add_argument("--seed", type=int, default=91091)
    ap.add_argument("--models", nargs="*", default=[], choices=list(DEFAULT_MODELS))
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--candidate_only", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    candidates, meta = collect_candidates(args.max_rows, args.max_per_group, args.seed)
    cand_rows = [c.to_dict() for c in candidates]
    write_csv(OUT / "source_relation_hard_negative_candidates.csv", cand_rows)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    all_records: list[dict[str, Any]] = []
    model_summaries: dict[str, Any] = {}
    if not args.candidate_only:
        models = args.models or ["legal40_depth_12x384_43022", "legal40_8x480_43022"]
        for model_label in models:
            model_path = DEFAULT_MODELS[model_label]
            tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
            model = AutoModelForMaskedLM.from_pretrained(str(model_path), trust_remote_code=True)
            model.eval().to(device)
            records = []
            for cand in candidates:
                base = cand.to_dict()
                sc = score_candidate(model, tokenizer, cand, device)
                rec = {"model": model_label, **base, **sc}
                records.append(rec)
                all_records.append(rec)
            model_summaries[model_label] = summarize_records(records)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    write_csv(OUT / "source_relation_hard_negative_model_margins.csv", all_records)

    payload = {
        "status": "SOURCE_RELATION_HARD_NEGATIVE_FEASIBILITY",
        "created_utc": now_utc(),
        "purpose": "CPU-only source-corpus measurement for a possible relation hard-negative objective before any training commitment.",
        "managed_sgcr_tasks_touched": False,
        "training_launched": False,
        "official_eval_launched": False,
        "corpus_meta": meta,
        "pair_inventory_note": "Frozen hand-written general relation alternatives; not derived from BabyLM evaluation rows.",
        "excluded_high_polysemy_pairs": ["in/out", "on/off", "left/right", "like/dislike", "know/ignore", "can/cannot"],
        "candidate_csv": str(OUT / "source_relation_hard_negative_candidates.csv"),
        "margin_csv": str(OUT / "source_relation_hard_negative_model_margins.csv"),
        "model_summaries": model_summaries,
        "json_output": str(OUT / "source_relation_hard_negative_feasibility.json"),
        "note": str(NOTE),
    }
    (OUT / "source_relation_hard_negative_feasibility.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = []
    lines.append("# research — source-grounded relation hard-negative feasibility\n\n")
    lines.append("This CPU-only work used the existing compact-view-reinvest 10M corpus and already trained checkpoints. It did not train, run the BabyLM suite, or touch the managed SGCR tasks.\n\n")
    lines.append("## Source generator\n\n")
    lines.append("The relation-alternative inventory was frozen before reading any new evaluation rows. It excludes the high-polysemy pairs `in/out`, `on/off`, `left/right`, `like/dislike`, `know/ignore`, and `can/cannot`. Sentences are kept only when exactly one safe relation term appears and the opposite term is absent.\n\n")
    lines.append(f"Rows scanned `{meta['rows_scanned']}`, words scanned `{meta['words_scanned']}`, sentence fragments seen `{meta['sentences_seen']}`, kept candidates `{meta['kept_total']}`.\n\n")
    lines.append("Kept candidates by group:\n\n")
    lines.append("| group | kept | raw matches |\n|---|---:|---:|\n")
    kept = Counter(c.group for c in candidates)
    raw = meta["raw_candidate_counts_by_group"]
    for g in sorted(set(kept) | set(raw)):
        lines.append(f"| {g} | {kept.get(g,0)} | {raw.get(g,0)} |\n")
    lines.append("\n")
    if model_summaries:
        lines.append("## Existing-model margins\n\n")
        lines.append("Margin is log p(original relation word | masked source sentence) minus log p(replacement word | the same masked sentence), with only equal-token-span cases scored. Weak or negative margins mean the future objective would have a real prediction target; large positive margins mean the signal is probably redundant.\n\n")
        lines.append("| model | scoreable | positive | nonpositive | near zero | weak <=1 | median | mean |\n|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for m, s in model_summaries.items():
            mar = s.get("margin", {})
            lines.append(f"| {m} | {s.get('scoreable')}/{s.get('records')} | {mar.get('positive_frac')} | {mar.get('nonpositive_frac')} | {mar.get('near0_abs_lt_0p25_frac')} | {mar.get('weak_le_1p0_frac')} | {mar.get('median')} | {mar.get('mean')} |\n")
        lines.append("\nGroup-level margin summaries are in the JSON.\n\n")
    else:
        lines.append("Candidate-only mode: no model margins computed.\n\n")
    lines.append("## Scientific reading\n\n")
    lines.append("A future source-grounded hard-negative relation objective should use these examples only after human or teacher-assisted quality reading of sampled edits, because regex precision is uneven even after strict filters. The single-input hard-negative form is the lower-accounting form: encode the original corpus sentence once, mask the edited span, and compare the original label against the frozen opposite label. Full corrupted-sequence ranking would be a different and more expensive exposure design.\n\n")
    lines.append(f"JSON: `{payload['json_output']}`\n\nCandidates: `{payload['candidate_csv']}`\n\nMargins: `{payload['margin_csv']}`\n")
    NOTE.write_text("".join(lines), encoding="utf-8")

    compact = {
        "status": payload["status"],
        "candidates": meta["kept_total"],
        "groups": meta["kept_by_group"],
        "device_used": str(device),
        "models": list(model_summaries),
        "summary": {m: {
            "scoreable_frac": s.get("scoreable_frac"),
            "margin_positive_frac": s.get("margin", {}).get("positive_frac"),
            "margin_nonpositive_frac": s.get("margin", {}).get("nonpositive_frac"),
            "margin_weak_le_1p0_frac": s.get("margin", {}).get("weak_le_1p0_frac"),
            "margin_median": s.get("margin", {}).get("median"),
        } for m, s in model_summaries.items()},
        "out_json": payload["json_output"],
        "note": payload["note"],
    }
    print(json.dumps(compact, indent=2), flush=True)


if __name__ == "__main__":
    main()
