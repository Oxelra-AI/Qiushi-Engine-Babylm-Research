#!/usr/bin/env python3
"""research: held-out evaluation segmentation geometry vs legal-tokenizer deficit.

Scientific purpose
------------------
The legal-tokenizer deficit raises a segmentation question that does not
require GPU or pending SGCR/EWoK task output: is the old-tokenizer advantage aligned
with fine-grained segmentation geometry on held-out evaluation text, rather than a
simple optimizer/learning-rate effect?  research already showed that aggregate
held-out tokens-per-word does not explain the deficit.  This script tests sharper
predictions: productive-affix seam preservation, whole-word merging of content-like
morphological forms, within-tokenizer stem-token reuse, and old-vs-legal token-position
phase displacement.

The script only reads existing official-eval text, existing tokenizers, and the existing
research per-UID score decomposition.  It does not train, evaluate a model checkpoint,
fit a tokenizer, or use evaluation text to create a training artifact.
"""
from __future__ import annotations

import csv
import json
import math
import pathlib
import re
import statistics
from collections import Counter, defaultdict
from typing import Any, Iterable

from transformers import AutoTokenizer

ROOT_A01 = pathlib.Path("experiments/archive/representation_and_objectives")
ROOT_A02 = pathlib.Path("experiments/archive/frontier_consolidation")
OUT_DIR = ROOT_A01 / "data/eval_segmentation_geometry"
DEFICIT_JSON = ROOT_A02 / "data/legal_deficit_subtask_decomposition/legal_deficit_subtask_decomposition.json"
EVAL_ROOT = ROOT_A01 / "data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval"

TOKENIZER_PATHS = {
    # Old non-submittable tokenizer used by the above-frontier mechanism endpoint.
    "old_inherited_16k_nonlegal": ROOT_A02 / "training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/hf_model/chck_100M",
    # Legal same-pool 16k coordinate used in the old-vs-legal decomposition.
    "legal_step35_16k": ROOT_A02 / "data/compliant_tokenizer",
    # Independently built legal 16k for the same compact-reinvest pool.
    "legal_a01_16k": ROOT_A01 / "data/strictsmall_tokenizer_retrain/strictsmall_compact_reinvest_16k_tokenizer",
    # Legal 40k representation candidate with existing training/evaluation results.
    "legal_a01_40k": ROOT_A01 / "data/legal_representation_route_map/tokenizers/legal_byte_bpe_40k",
    # Support-floored 40k candidate; no local full endpoint, but useful geometry control.
    "legal_a01_40k_minfreq50": ROOT_A01 / "data/tokenizer_support_spectrum/tokenizers/legal_byte_bpe_40k_minfreq50",
}

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by", "can",
    "could", "did", "do", "does", "for", "from", "had", "has", "have", "he", "her",
    "hers", "him", "his", "i", "if", "in", "into", "is", "it", "its", "may", "might",
    "must", "not", "of", "on", "or", "our", "she", "should", "that", "the", "their",
    "them", "there", "these", "they", "this", "those", "to", "was", "we", "were",
    "what", "when", "where", "which", "who", "whom", "whose", "will", "with", "would",
    "you", "your", "than", "then", "so", "such", "very", "more", "most", "less", "least",
}

# Conservative productive-affix inventory.  The script records the exact boundary rules;
# we use them for relative old-vs-legal comparisons, not for a linguistic annotation claim.
PREFIXES = ("anti", "inter", "under", "over", "pre", "post", "non", "mis", "dis", "un", "re", "in", "im", "ir", "il")
SUFFIXES = (
    "ational", "fulness", "ousness", "ization", "isation", "iveness", "lessly",
    "ability", "ibility", "ments", "ness", "ment", "tion", "sion", "able", "ible",
    "ally", "less", "ful", "ous", "ive", "ity", "est", "ier", "ing", "ers",
    "ies", "ied", "ed", "er", "ly", "es", "s",
)
# Do not treat a trailing s as a productive suffix when the remaining stem would be too short
# or the form is a high-frequency function-like word.
S_SUFFIX_EXCLUDE = {"is", "was", "has", "his", "this", "thus", "us", "as", "does", "goes"}
SPECIAL_IDS = {0, 1, 2, 3, 4}


def read_jsonl(path: pathlib.Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def iter_eval_texts(column: str, uid: str) -> list[str]:
    texts: list[str] = []
    if column == "BLiMP":
        p = EVAL_ROOT / "blimp_filtered" / f"{uid}.jsonl"
        for d in read_jsonl(p):
            texts.append(d["sentence_good"])
            texts.append(d["sentence_bad"])
    elif column == "Supplement":
        p = EVAL_ROOT / "supplement_filtered" / f"{uid}.jsonl"
        for d in read_jsonl(p):
            texts.append(d["sentence_good"])
            texts.append(d["sentence_bad"])
    elif column == "EWoK":
        p = EVAL_ROOT / "ewok_filtered" / f"{uid}.jsonl"
        for d in read_jsonl(p):
            # Follow research/research convention: complete context+target strings.
            texts.append(" ".join([d["Context1"], d["Target1"]]))
            texts.append(" ".join([d["Context2"], d["Target2"]]))
    return texts


def word_spans(text: str) -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group(0)) for m in WORD_RE.finditer(text)]


def clean_word(w: str) -> str:
    return w.lower().strip("'")


def is_content_like(w: str) -> bool:
    wl = clean_word(w)
    return len(wl) >= 6 and wl not in STOPWORDS and any(c.isalpha() for c in wl)


def affix_boundaries(word: str) -> list[dict[str, Any]]:
    wl = clean_word(word)
    out: list[dict[str, Any]] = []
    if not wl.isalpha() or len(wl) < 4:
        return out
    for p in PREFIXES:
        if wl.startswith(p) and len(wl) - len(p) >= 3:
            out.append({"kind": "prefix", "affix": p, "boundary": len(p), "stem": wl[len(p):]})
    for s in SUFFIXES:
        if not wl.endswith(s):
            continue
        stem_len = len(wl) - len(s)
        if stem_len < 3:
            continue
        if s in {"s", "es"} and (wl in S_SUFFIX_EXCLUDE or stem_len < 4):
            continue
        # Avoid counting words ending in ss/us/ous as simple plural s.
        if s == "s" and (wl.endswith("ss") or wl.endswith("us") or wl.endswith("ous")):
            continue
        # Basic y->ies/ied stem recovery; recorded for stem-token reuse only.
        stem = wl[:stem_len]
        if s in {"ies", "ied"}:
            stem = stem + "y"
        out.append({"kind": "suffix", "affix": s, "boundary": stem_len, "stem": stem})
    # Unique by boundary/kind/affix.
    seen = set()
    uniq = []
    for b in out:
        key = (b["kind"], b["affix"], b["boundary"])
        if key not in seen:
            uniq.append(b)
            seen.add(key)
    return uniq


def token_pieces(tokenizer, text: str) -> tuple[list[int], list[tuple[int, int]], list[str]]:
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    offs = enc["offset_mapping"]
    toks = tokenizer.convert_ids_to_tokens(ids)
    return ids, offs, toks


def word_token_indices(offsets: list[tuple[int, int]], start: int, end: int) -> list[int]:
    inds = []
    for i, (a, b) in enumerate(offsets):
        if b <= start or a >= end:
            continue
        # Fast byte-level tokenizers can return zero-length artifacts; ignore them.
        if a == b:
            continue
        inds.append(i)
    return inds


def token_boundaries_for_word(offsets: list[tuple[int, int]], inds: list[int], start: int, end: int) -> set[int]:
    bounds: set[int] = set()
    for i in inds:
        a, b = offsets[i]
        # Clip to the word span; spaces/punctuation in byte-level tokens should not create false boundaries.
        aa = max(a, start)
        bb = min(b, end)
        if start < aa < end:
            bounds.add(aa - start)
        if start < bb < end:
            bounds.add(bb - start)
    return bounds


def norm_token_piece(tok: str) -> str:
    # Normalize common BPE markers; this is only used within-tokenizer for stem reuse.
    return tok.replace("Ġ", "").replace("▁", "").replace("Ċ", "").lower()


def isolated_pieces(tokenizer, word: str) -> set[str]:
    # Prefixing a space makes byte-level tokenizers expose word-initial tokens comparable to sentence use.
    ids = tokenizer(" " + word, add_special_tokens=False)["input_ids"]
    toks = tokenizer.convert_ids_to_tokens(ids)
    return {p for p in (norm_token_piece(t) for t in toks) if len(p) >= 2 and p.isalpha()}


def text_metrics_for_tokenizer(tokenizer, texts: list[str]) -> dict[str, float | int | None]:
    counters = Counter()
    stem_cache: dict[str, set[str]] = {}
    for text in texts:
        spans = word_spans(text)
        ids, offsets, toks = token_pieces(tokenizer, text)
        counters["texts"] += 1
        counters["tokens"] += sum(1 for i in ids if i not in SPECIAL_IDS)
        counters["chars"] += len(text)
        counters["words"] += len(spans)
        if len(ids) > 256:
            counters["texts_over_256"] += 1
        for start, end, raw in spans:
            wl = clean_word(raw)
            inds = word_token_indices(offsets, start, end)
            n_tok = len(inds)
            counters["alpha_words"] += 1
            counters["word_token_total"] += n_tok
            if n_tok == 1:
                counters["whole_alpha_words"] += 1
            if is_content_like(wl):
                counters["content_words"] += 1
                if n_tok == 1:
                    counters["whole_content_words"] += 1
            affs = affix_boundaries(wl)
            if affs:
                counters["morph_words"] += 1
                if n_tok == 1:
                    counters["whole_morph_words"] += 1
                bounds = token_boundaries_for_word(offsets, inds, start, end)
                if any(a["boundary"] in bounds for a in affs):
                    counters["morph_seam_preserved_words"] += 1
                if any(a["kind"] == "suffix" for a in affs):
                    counters["suffix_morph_words"] += 1
                    # Stem reuse: whether any isolated stem piece appears among sentence token pieces for the inflected word.
                    word_piece_set = {norm_token_piece(toks[i]) for i in inds}
                    ok = False
                    for a in affs:
                        if a["kind"] != "suffix":
                            continue
                        stem = a["stem"]
                        if stem not in stem_cache:
                            stem_cache[stem] = isolated_pieces(tokenizer, stem)
                        if stem_cache[stem] and (stem_cache[stem] & word_piece_set):
                            ok = True
                            break
                    if ok:
                        counters["suffix_stem_piece_reused_words"] += 1
    def frac(num: str, den: str) -> float | None:
        return counters[num] / counters[den] if counters[den] else None
    return {
        "texts": counters["texts"],
        "words": counters["words"],
        "tokens": counters["tokens"],
        "tokens_per_word": counters["tokens"] / counters["words"] if counters["words"] else None,
        "frac_texts_over_256": frac("texts_over_256", "texts"),
        "mean_word_tokens": counters["word_token_total"] / counters["alpha_words"] if counters["alpha_words"] else None,
        "whole_alpha_rate": frac("whole_alpha_words", "alpha_words"),
        "content_words": counters["content_words"],
        "whole_content_rate": frac("whole_content_words", "content_words"),
        "morph_words": counters["morph_words"],
        "whole_morph_rate": frac("whole_morph_words", "morph_words"),
        "morph_seam_preserved_rate": frac("morph_seam_preserved_words", "morph_words"),
        "suffix_morph_words": counters["suffix_morph_words"],
        "suffix_stem_piece_reuse_rate": frac("suffix_stem_piece_reused_words", "suffix_morph_words"),
    }


def phase_metrics(old_tok, legal_tok, texts: list[str]) -> dict[str, float | int | None]:
    vals_abs_mean = []
    vals_abs_max = []
    vals_final = []
    vals_word_delta_abs = []
    for text in texts:
        spans = word_spans(text)
        if not spans:
            continue
        old_ids, old_offsets, _ = token_pieces(old_tok, text)
        legal_ids, legal_offsets, _ = token_pieces(legal_tok, text)
        cum_old = 0
        cum_leg = 0
        per_abs = []
        for start, end, _ in spans:
            no = len(word_token_indices(old_offsets, start, end))
            nl = len(word_token_indices(legal_offsets, start, end))
            vals_word_delta_abs.append(abs(nl - no))
            cum_old += no
            cum_leg += nl
            per_abs.append(abs(cum_leg - cum_old))
        if per_abs:
            vals_abs_mean.append(statistics.fmean(per_abs))
            vals_abs_max.append(max(per_abs))
            vals_final.append(cum_leg - cum_old)
    def mean(xs: list[float]) -> float | None:
        return statistics.fmean(xs) if xs else None
    return {
        "phase_texts": len(vals_abs_mean),
        "mean_abs_cumulative_token_displacement": mean(vals_abs_mean),
        "mean_max_abs_cumulative_token_displacement": mean(vals_abs_max),
        "mean_final_token_delta": mean(vals_final),
        "mean_abs_word_token_delta": mean(vals_word_delta_abs),
    }


def pearson(xs: list[float | None], ys: list[float | None]) -> float | None:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if x is not None and y is not None and math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 3:
        return None
    xv, yv = zip(*pairs)
    mx, my = statistics.fmean(xv), statistics.fmean(yv)
    vx = sum((x - mx) ** 2 for x in xv)
    vy = sum((y - my) ** 2 for y in yv)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(vx * vy)


def ranks(vals: list[float]) -> list[float]:
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    out = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and vals[order[j]] == vals[order[i]]:
            j += 1
        avg = (i + 1 + j) / 2
        for k in range(i, j):
            out[order[k]] = avg
        i = j
    return out


def spearman(xs: list[float | None], ys: list[float | None]) -> float | None:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if x is not None and y is not None and math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 3:
        return None
    return pearson(ranks([x for x, _ in pairs]), ranks([y for _, y in pairs]))


def fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        if math.isnan(x):
            return "NaN"
        return f"{x:.{nd}f}"
    return str(x)


def safe_round(x: Any, nd: int = 6) -> Any:
    if x is None:
        return None
    if isinstance(x, float):
        if not math.isfinite(x):
            return None
        return round(x, nd)
    return x


def summarize_correlations(rows: list[dict[str, Any]], metrics: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col in ["BLiMP", "Supplement", "EWoK", "ALL"]:
        sub = rows if col == "ALL" else [r for r in rows if r["column"] == col]
        deltas = [r["score_delta_legal_step35_minus_old_ref"] for r in sub]
        out[col] = {"n_uid": len(sub)}
        for m in metrics:
            vals = [r.get(m) for r in sub]
            valid = [v for v in vals if v is not None and isinstance(v, (int, float)) and math.isfinite(float(v))]
            out[col][m] = {
                "mean": safe_round(statistics.fmean(valid), 6) if valid else None,
                "pearson_delta_vs_metric": safe_round(pearson(deltas, vals), 6),
                "spearman_delta_vs_metric": safe_round(spearman(deltas, vals), 6),
            }
    return out


def load_tokenizers() -> dict[str, Any]:
    toks = {}
    for label, path in TOKENIZER_PATHS.items():
        if not path.exists():
            continue
        toks[label] = AutoTokenizer.from_pretrained(str(path), use_fast=True, local_files_only=True)
    missing = [label for label, path in TOKENIZER_PATHS.items() if not path.exists()]
    if missing:
        print(json.dumps({"event": "missing_tokenizer_paths", "missing": missing}, indent=2), flush=True)
    return toks


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    deficit = json.loads(DEFICIT_JSON.read_text(encoding="utf-8"))
    tokenizers = load_tokenizers()
    required = {"old_inherited_16k_nonlegal", "legal_step35_16k"}
    if not required.issubset(tokenizers):
        raise RuntimeError(f"missing required tokenizers: {sorted(required - set(tokenizers))}")

    per_uid: list[dict[str, Any]] = []
    text_cache: dict[tuple[str, str], list[str]] = {}
    for column, drows in deficit["columns"].items():
        if column not in {"BLiMP", "Supplement", "EWoK"}:
            continue
        for d in drows:
            uid = d["uid"]
            delta = d.get("delta_legal_step35")
            if delta is None:
                continue
            texts = text_cache.setdefault((column, uid), iter_eval_texts(column, uid))
            row: dict[str, Any] = {
                "column": column,
                "uid": uid,
                "score_delta_legal_step35_minus_old_ref": float(delta),
                "n_texts": len(texts),
                "n_words_regex": sum(len(word_spans(t)) for t in texts),
            }
            for label, tok in tokenizers.items():
                tm = text_metrics_for_tokenizer(tok, texts)
                for k, v in tm.items():
                    row[f"{label}_{k}"] = v
            # Old-vs-legal research fine-grained differences. Positive seam_delta means legal preserves more seams.
            old = "old_inherited_16k_nonlegal"
            leg = "legal_step35_16k"
            for metric in [
                "tokens_per_word", "whole_alpha_rate", "whole_content_rate", "whole_morph_rate",
                "morph_seam_preserved_rate", "suffix_stem_piece_reuse_rate", "frac_texts_over_256",
            ]:
                row[f"minus_old_{metric}"] = (
                    row.get(f"{leg}_{metric}") - row.get(f"{old}_{metric}")
                    if row.get(f"{leg}_{metric}") is not None and row.get(f"{old}_{metric}") is not None else None
                )
            # Independent legal16k comparison; checks whether geometry is a budget-class property.
            if "legal_a01_16k" in tokenizers:
                for metric in ["tokens_per_word", "morph_seam_preserved_rate", "whole_morph_rate", "suffix_stem_piece_reuse_rate"]:
                    row[f"a01_16k_minus_step35_{metric}"] = (
                        row.get(f"legal_a01_16k_{metric}") - row.get(f"{leg}_{metric}")
                        if row.get(f"legal_a01_16k_{metric}") is not None and row.get(f"{leg}_{metric}") is not None else None
                    )
            # Does legal40k move geometry toward or away from the old tokenizer on the affected UID?
            if "legal_a01_40k" in tokenizers:
                for metric in ["tokens_per_word", "whole_morph_rate", "morph_seam_preserved_rate", "suffix_stem_piece_reuse_rate"]:
                    s35 = row.get(f"{leg}_{metric}")
                    oldv = row.get(f"{old}_{metric}")
                    k40 = row.get(f"legal_a01_40k_{metric}")
                    row[f"legal40k_minus_step35_{metric}"] = (k40 - s35) if k40 is not None and s35 is not None else None
                    denom = s35 - oldv if s35 is not None and oldv is not None else None
                    # Fraction of research-old signed gap closed by 40k. Values can be outside [0,1] when it overshoots.
                    row[f"legal40k_closes_step35_old_{metric}_frac"] = ((s35 - k40) / denom) if denom is not None and abs(denom) > 1e-9 else None
            # Phase displacement old vs legal research.
            pm = phase_metrics(tokenizers[old], tokenizers[leg], texts)
            for k, v in pm.items():
                row[f"vs_old_{k}"] = v
            per_uid.append(row)
            if len(per_uid) % 20 == 0:
                print(json.dumps({"event": "progress", "uids_done": len(per_uid)}), flush=True)

    metrics_to_correlate = [
        "minus_old_tokens_per_word",
        "minus_old_whole_morph_rate",
        "minus_old_morph_seam_preserved_rate",
        "minus_old_suffix_stem_piece_reuse_rate",
        "minus_old_whole_content_rate",
        "vs_old_mean_abs_cumulative_token_displacement",
        "vs_old_mean_max_abs_cumulative_token_displacement",
        "vs_old_mean_abs_word_token_delta",
        "legal40k_minus_step35_morph_seam_preserved_rate",
        "legal40k_minus_step35_whole_morph_rate",
        "legal40k_closes_step35_old_morph_seam_preserved_rate_frac",
        "legal40k_closes_step35_old_tokens_per_word_frac",
    ]
    correlations = summarize_correlations(per_uid, metrics_to_correlate)

    # Ranked subsets for interpretation.
    worst = sorted(per_uid, key=lambda r: r["score_delta_legal_step35_minus_old_ref"])[:20]
    best = sorted(per_uid, key=lambda r: r["score_delta_legal_step35_minus_old_ref"], reverse=True)[:20]
    focus_names = {
        "principle_A_reconstruction", "wh_questions_object_gap", "regular_plural_subject_verb_agreement_1",
        "anaphor_gender_agreement", "tough_vs_raising_1", "animate_subject_trans",
        "qa_congruence_easy", "qa_congruence_tricky", "turn_taking", "subject_aux_inversion", "hypernym",
        "physical-dynamics", "material-dynamics", "social-properties", "quantitative-properties",
        "spatial-relations", "material-properties", "social-interactions", "physical-relations",
    }
    focus = [r for r in per_uid if r["uid"] in focus_names]

    # Magnitude summaries: do worst losses actually have worse legal-vs-old seam geometry than gains?
    def avg(rows: list[dict[str, Any]], key: str) -> float | None:
        vals = [r.get(key) for r in rows if isinstance(r.get(key), (float, int)) and math.isfinite(float(r.get(key)))]
        return statistics.fmean(vals) if vals else None
    group_summary = {
        "worst20_mean_delta": avg(worst, "score_delta_legal_step35_minus_old_ref"),
        "best20_mean_delta": avg(best, "score_delta_legal_step35_minus_old_ref"),
    }
    for m in metrics_to_correlate:
        group_summary[f"worst20_mean_{m}"] = avg(worst, m)
        group_summary[f"best20_mean_{m}"] = avg(best, m)

    tokenizer_info = {
        label: {"path": str(TOKENIZER_PATHS[label]), "len": len(tok), "is_fast": tok.is_fast}
        for label, tok in tokenizers.items()
    }
    result = {
        "status": "EVAL_SEGMENTATION_GEOMETRY",
        "purpose": "CPU-only held-out segmentation geometry test of the old-vs-legal tokenizer deficit; no training, no model scoring, no use of pending SGCR/EWoK tasks.",
        "inputs": {
            "deficit_json": str(DEFICIT_JSON),
            "eval_root": str(EVAL_ROOT),
            "tokenizers": tokenizer_info,
            "affix_rules": {"prefixes": PREFIXES, "suffixes": SUFFIXES, "s_suffix_exclude": sorted(S_SUFFIX_EXCLUDE)},
        },
        "correlations": correlations,
        "group_summary": group_summary,
        "focus_rows": focus,
        "worst20_rows": worst,
        "best20_rows": best,
        "per_uid": per_uid,
    }
    out_json = OUT_DIR / "eval_segmentation_geometry.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    # Write compact CSV for later plotting/filtering.
    out_csv = OUT_DIR / "eval_segmentation_geometry_per_uid.csv"
    csv_fields = [
        "column", "uid", "score_delta_legal_step35_minus_old_ref", "n_texts", "n_words_regex",
        "old_inherited_16k_nonlegal_tokens_per_word", "legal_step35_16k_tokens_per_word", "minus_old_tokens_per_word",
        "old_inherited_16k_nonlegal_morph_words", "old_inherited_16k_nonlegal_morph_seam_preserved_rate",
        "legal_step35_16k_morph_seam_preserved_rate", "minus_old_morph_seam_preserved_rate",
        "old_inherited_16k_nonlegal_whole_morph_rate", "legal_step35_16k_whole_morph_rate", "minus_old_whole_morph_rate",
        "old_inherited_16k_nonlegal_suffix_stem_piece_reuse_rate", "legal_step35_16k_suffix_stem_piece_reuse_rate", "minus_old_suffix_stem_piece_reuse_rate",
        "vs_old_mean_abs_cumulative_token_displacement", "vs_old_mean_max_abs_cumulative_token_displacement", "vs_old_mean_abs_word_token_delta",
        "legal_a01_40k_morph_seam_preserved_rate", "legal40k_minus_step35_morph_seam_preserved_rate", "legal40k_closes_step35_old_morph_seam_preserved_rate_frac",
        "legal_a01_40k_tokens_per_word", "legal40k_minus_step35_tokens_per_word", "legal40k_closes_step35_old_tokens_per_word_frac",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        for r in per_uid:
            w.writerow({k: r.get(k) for k in csv_fields})

    # Human-readable note.
    out_md = (OUT_DIR.parents[4] / 'research/documents/representation_and_objectives/data/eval_segmentation_geometry/eval_segmentation_geometry.md')
    md: list[str] = []
    md.append("# research — held-out segmentation geometry vs legal-tokenizer deficit")
    md.append("")
    md.append("CPU-only analysis over existing official evaluation text, existing tokenizers, and the research old-vs-legal per-UID score table. It does not train, score model checkpoints, fit a tokenizer, or use pending SGCR/full-EWoK results.")
    md.append("")
    md.append("## Tokenizers")
    for label, info in tokenizer_info.items():
        md.append(f"- `{label}` len={info['len']}: `{info['path']}`")
    md.append("")
    md.append("## Correlations: research score delta vs fine-grained held-out tokenization geometry")
    md.append("`score_delta = legal_step35 - old_ref`; negative means the legal tokenizer endpoint lost. If a seam mechanism is primary, losses should align with lower legal-vs-old seam preservation or stem-piece reuse, not merely with total tokens per word.")
    md.append("")
    md.append("| column | n | metric | mean | Pearson(delta,metric) | Spearman |")
    md.append("|---|---:|---|---:|---:|---:|")
    for col in ["BLiMP", "Supplement", "EWoK", "ALL"]:
        for m in metrics_to_correlate:
            rec = correlations[col][m]
            md.append(f"| {col} | {correlations[col]['n_uid']} | `{m}` | {fmt(rec['mean'], 6)} | {fmt(rec['pearson_delta_vs_metric'], 4)} | {fmt(rec['spearman_delta_vs_metric'], 4)} |")
    md.append("")
    md.append("## Worst-vs-best UID contrast")
    md.append("| metric | worst20 mean | best20 mean |")
    md.append("|---|---:|---:|")
    md.append(f"| score delta | {fmt(group_summary['worst20_mean_delta'], 4)} | {fmt(group_summary['best20_mean_delta'], 4)} |")
    for m in metrics_to_correlate:
        md.append(f"| `{m}` | {fmt(group_summary.get('worst20_mean_' + m), 6)} | {fmt(group_summary.get('best20_mean_' + m), 6)} |")
    md.append("")
    md.append("## Focus rows")
    md.append("| column | uid | Δ score | research-old tpw | research-old seam | research-old whole morph | research-old stem reuse | phase mean abs | 40k-research seam | 40k seam gap closed |")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(focus, key=lambda x: (x["column"], x["score_delta_legal_step35_minus_old_ref"])):
        md.append(
            f"| {r['column']} | {r['uid']} | {fmt(r['score_delta_legal_step35_minus_old_ref'],2)} | "
            f"{fmt(r.get('minus_old_tokens_per_word'),4)} | {fmt(r.get('minus_old_morph_seam_preserved_rate'),4)} | "
            f"{fmt(r.get('minus_old_whole_morph_rate'),4)} | {fmt(r.get('minus_old_suffix_stem_piece_reuse_rate'),4)} | "
            f"{fmt(r.get('vs_old_mean_abs_cumulative_token_displacement'),4)} | {fmt(r.get('legal40k_minus_step35_morph_seam_preserved_rate'),4)} | "
            f"{fmt(r.get('legal40k_closes_step35_old_morph_seam_preserved_rate_frac'),3)} |"
        )
    md.append("")
    md.append("## Twenty largest legal research losses")
    md.append("| column | uid | Δ score | research-old tpw | research-old seam | research-old stem reuse | phase mean abs |")
    md.append("|---|---|---:|---:|---:|---:|---:|")
    for r in worst:
        md.append(
            f"| {r['column']} | {r['uid']} | {fmt(r['score_delta_legal_step35_minus_old_ref'],2)} | "
            f"{fmt(r.get('minus_old_tokens_per_word'),4)} | {fmt(r.get('minus_old_morph_seam_preserved_rate'),4)} | "
            f"{fmt(r.get('minus_old_suffix_stem_piece_reuse_rate'),4)} | {fmt(r.get('vs_old_mean_abs_cumulative_token_displacement'),4)} |"
        )
    md.append("")
    md.append("## Reading")
    md.append("This sharper geometry test should be read together with research: simple held-out tokens-per-word already failed as an explanation. A route-relevant positive result would require structured alignment of losses with seam/stem/phase metrics on the affected UIDs. If the correlations remain weak or sign-mixed, a pure tokenizer-seam story is not sufficient; the next route should emphasize either a different representation mechanism, optimization interaction, or the distinct compact-view/conditional signal rather than a type-dedup tokenizer run.")
    md.append("")
    md.append(f"Full JSON: `{out_json}`")
    md.append(f"Per-UID CSV: `{out_csv}`")
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    compact = {
        "status": result["status"],
        "out_json": str(out_json),
        "out_csv": str(out_csv),
        "out_md": str(out_md),
        "n_uid": len(per_uid),
        "correlations_all": correlations["ALL"],
        "group_summary_subset": {k: safe_round(v, 6) for k, v in group_summary.items() if k in {
            "worst20_mean_delta", "best20_mean_delta",
            "worst20_mean_step35_minus_old_morph_seam_preserved_rate", "best20_mean_step35_minus_old_morph_seam_preserved_rate",
            "worst20_mean_step35_minus_old_suffix_stem_piece_reuse_rate", "best20_mean_step35_minus_old_suffix_stem_piece_reuse_rate",
            "worst20_mean_step35_vs_old_mean_abs_cumulative_token_displacement", "best20_mean_step35_vs_old_mean_abs_cumulative_token_displacement",
        }},
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
