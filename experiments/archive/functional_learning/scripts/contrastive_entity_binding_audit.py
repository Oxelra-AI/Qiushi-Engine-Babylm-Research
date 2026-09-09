#!/usr/bin/env python3
"""research: audit and prototype a contrastive entity-binding packet family.

The state-update analysis shows that the existing state-update replacement arm
mostly strengthened source-state retention and did not improve neutral-adjusted
target-update use.  This script turns the next scientific requirement into a
concrete, inspectable data object before any H100 training:

  * preserve the inherited ALN source--rewrite ingredient as the supported
    baseline object;
  * add only from filler, rather than replacing ALN;
  * require paired same-source contrasts where one row updates the queried entity
    and the matched row updates an irrelevant entity while the queried entity
    should retain its source state;
  * inspect the actual MLM-masked learner input for target state spans so that
    unchanged-source copying, recency copying, use-sentence leakage, or cue words
    cannot solve the contrast without entity-specific binding.

The script is CPU-only.  It reads validated plain-use packets as candidate
natural-language material; it does not treat them as successful training evidence.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import collections
import csv
import hashlib
import json
import math
import os
import random
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = _public_path('.')
A01 = _public_path('experiments/archive/functional_learning')
A02 = _public_path('experiments/archive/relation_learning')
COMPACT_EXPERIENCE = _public_path('experiments/archive/compact_experience')

PLAIN_ALL = _public_path('experiments/archive/relation_learning/data/state_use_generation/validated_state_use_packets_plainuse_full_all.jsonl')
PLAIN_BALANCED = _public_path('experiments/archive/relation_learning/data/state_use_generation/validated_state_use_packets_plainuse_full_balanced.jsonl')
PLAIN_TRAIN = _public_path('experiments/archive/relation_learning/data/state_use_generation/validated_state_use_packets_plainuse_full_train.jsonl')
PLAIN_HELD = _public_path('experiments/archive/relation_learning/data/state_use_generation/validated_state_use_packets_plainuse_full_heldout.jsonl')
ALN_10M = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl')
ALN_META = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/qwen_pair_packed_rows_meta.jsonl')
OFFICIAL_POOL = _public_path('experiments/archive/compact_experience/data/mixture/official_pool.jsonl')
TOKENIZER_PATH = _public_path('experiments/archive/frontier_consolidation/data/compliant_tokenizer')
OUT_DIR = _public_path('experiments/archive/functional_learning/data/contrastive_entity_binding')
NOTE = _public_path('research/notes/functional_learning/contrastive_entity_binding_design.md')

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[\u2019'][A-Za-z0-9]+)?")
CUE_PATTERNS = [
    r"\bnow\b", r"\btoday\b", r"\bcurrently\b", r"\bnew\b", r"\bno longer\b",
    r"\bstill\b", r"\bremain(?:s|ed|ing)?\b", r"\bcontinue(?:s|d|ing)?\b",
    r"\bstay(?:s|ed|ing)?\b", r"\bkeep(?:s|ing)?\b", r"\bretain(?:s|ed|ing)?\b",
    r"\bunchanged\b", r"\bformer\b", r"\bagain later\b",
]

@dataclass
class Packet:
    pair_id: str
    packet_type: str
    rec: dict[str, Any]


def rel(p: Path | str) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip())


def toks(s: str) -> list[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(str(s or ""))]


def token_set(s: str) -> set[str]:
    return set(toks(s))


def lcs_words(a: str, b: str) -> int:
    aa, bb = toks(a), toks(b)
    best = 0
    # These strings are short; simple DP is fine.
    prev = [0] * (len(bb) + 1)
    for x in aa:
        cur = [0] * (len(bb) + 1)
        for j, y in enumerate(bb, start=1):
            if x == y:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def content_overlap(a: str, b: str) -> float:
    aa, bb = token_set(a), token_set(b)
    if not aa or not bb:
        return 0.0
    return len(aa & bb) / min(len(aa), len(bb))


def cue_hits(s: str) -> list[str]:
    low = str(s or "").lower()
    return [p.strip("\\b") for p in CUE_PATTERNS if re.search(p, low)]


def load_packets(path: Path) -> dict[str, dict[str, Packet]]:
    out: dict[str, dict[str, Packet]] = collections.defaultdict(dict)
    for rec in read_jsonl(path):
        pid = str(rec.get("pair_id"))
        typ = str(rec.get("packet_type"))
        if typ in {"UPDATED_USE", "UNCHANGED_DISTRACTOR_USE"}:
            out[pid][typ] = Packet(pid, typ, rec)
    return out


def summarize_counts(path: Path) -> dict[str, Any]:
    counts = collections.Counter()
    pair_counts = collections.defaultdict(set)
    n = 0
    for rec in read_jsonl(path):
        n += 1
        counts[rec.get("packet_type")] += 1
        pair_counts[rec.get("pair_id")].add(rec.get("packet_type"))
    both = sum(1 for s in pair_counts.values() if {"UPDATED_USE", "UNCHANGED_DISTRACTOR_USE"} <= set(s))
    return {"path": rel(path), "rows": n, "type_counts": dict(counts), "unique_pair_ids": len(pair_counts), "same_pair_both_types": both}


def span_piece_count(tokenizer, phrase: str) -> int:
    if tokenizer is None:
        return len(toks(phrase))
    ids = tokenizer(phrase, add_special_tokens=False).get("input_ids", [])
    return len(ids)


def mask_phrase_once(text: str, phrase: str) -> tuple[str, int]:
    """Replace the first exact phrase occurrence by [MASK] words; fallback token count."""
    phrase = norm(phrase)
    if not phrase:
        return text, 0
    # Try exact case-insensitive phrase with word-ish boundaries but preserve punctuation outside.
    pat = re.compile(re.escape(phrase), flags=re.IGNORECASE)
    m = pat.search(text)
    if not m:
        return text, 0
    n_words = max(1, len(phrase.split()))
    repl = " ".join(["[MASK]"] * n_words)
    return text[:m.start()] + repl + text[m.end():], n_words


def make_context(rec: dict[str, Any]) -> str:
    return norm(f"{rec.get('source_sentence','')} {rec.get('update_sentence','')} {rec.get('use_sentence','')}")


def make_masked(rec: dict[str, Any], answer_phrase: str) -> tuple[str, int]:
    # Mask only in the final use sentence, then reconstruct full context.
    source = norm(rec.get("source_sentence", ""))
    update = norm(rec.get("update_sentence", ""))
    use = norm(rec.get("use_sentence", ""))
    masked_use, nmask = mask_phrase_once(use, answer_phrase)
    return norm(f"{source} {update} {masked_use}"), nmask


def packet_quality(rec: dict[str, Any], tokenizer=None) -> dict[str, Any]:
    ptyp = rec.get("packet_type")
    source_state = norm(rec.get("source_state", ""))
    new_state = norm(rec.get("new_state", ""))
    use = norm(rec.get("use_sentence", ""))
    source = norm(rec.get("source_sentence", ""))
    update = norm(rec.get("update_sentence", ""))
    target_ent = " ".join(str(x) for x in rec.get("target_entity_terms") or [])
    updated_ent = " ".join(str(x) for x in rec.get("updated_entity_terms") or [])
    ans = new_state if ptyp == "UPDATED_USE" else source_state
    foil = source_state if ptyp == "UPDATED_USE" else new_state
    masked, nmask_words = make_masked(rec, ans)
    use_has_answer = bool(ans) and ans.lower() in use.lower()
    use_has_foil = bool(foil) and foil.lower() in use.lower()
    source_has_answer = bool(ans) and ans.lower() in source.lower()
    update_has_answer = bool(ans) and ans.lower() in update.lower()
    source_has_foil = bool(foil) and foil.lower() in source.lower()
    update_has_foil = bool(foil) and foil.lower() in update.lower()
    return {
        "pair_id": rec.get("pair_id"),
        "packet_type": ptyp,
        "target_entity": rec.get("target_entity"),
        "updated_entity": rec.get("updated_entity"),
        "target_updated_same_terms": sorted(set(rec.get("target_entity_terms") or []) & set(rec.get("updated_entity_terms") or [])),
        "target_updated_overlap": content_overlap(target_ent, updated_ent) if target_ent and updated_ent else 0.0,
        "answer_phrase": ans,
        "foil_phrase": foil,
        "answer_piece_count": span_piece_count(tokenizer, ans),
        "foil_piece_count": span_piece_count(tokenizer, foil),
        "masked_words_in_use": nmask_words,
        "use_has_answer_exact": use_has_answer,
        "use_has_foil_exact": use_has_foil,
        "source_has_answer_exact": source_has_answer,
        "update_has_answer_exact": update_has_answer,
        "source_has_foil_exact": source_has_foil,
        "update_has_foil_exact": update_has_foil,
        "use_source_lcs_words": lcs_words(use, source),
        "use_update_lcs_words": lcs_words(use, update),
        "use_source_content_overlap": content_overlap(use, source),
        "use_update_content_overlap": content_overlap(use, update),
        "cue_hits_use": cue_hits(use),
        "context_words": len(make_context(rec).split()),
        "masked_text": masked,
    }


def is_contrast_candidate(upd: dict[str, Any], ret: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if norm(upd.get("source_sentence")) != norm(ret.get("source_sentence")):
        reasons.append("source_sentence_mismatch")
    # Queried entity should be the same, otherwise the pair does not force same-source counterfactual assignment.
    if set(upd.get("target_entity_terms") or []) != set(ret.get("target_entity_terms") or []):
        reasons.append("target_entity_terms_mismatch")
    if not (set(upd.get("target_entity_terms") or []) & set(upd.get("updated_entity_terms") or [])):
        reasons.append("updated_row_update_not_target")
    if set(ret.get("target_entity_terms") or []) & set(ret.get("updated_entity_terms") or []):
        reasons.append("retention_row_updates_target")
    if not norm(upd.get("source_state")) or not norm(upd.get("new_state")) or not norm(ret.get("source_state")) or not norm(ret.get("new_state")):
        reasons.append("missing_state_phrase")
    if cue_hits(upd.get("use_sentence", "")) or cue_hits(ret.get("use_sentence", "")):
        reasons.append("use_sentence_cue")
    # The paired source state should be effectively the same for a clean source-retention alternative.
    if token_set(upd.get("source_state", "")) and token_set(ret.get("source_state", "")):
        src_state_j = len(token_set(upd.get("source_state", "")) & token_set(ret.get("source_state", ""))) / max(1, len(token_set(upd.get("source_state", "")) | token_set(ret.get("source_state", ""))))
        if src_state_j < 0.5:
            reasons.append(f"source_state_low_jaccard_{src_state_j:.2f}")
    return len(reasons) == 0, reasons


def maybe_load_tokenizer():
    try:
        from transformers import AutoTokenizer
        if TOKENIZER_PATH.exists():
            return AutoTokenizer.from_pretrained(str(TOKENIZER_PATH), local_files_only=True)
    except Exception as exc:
        print(f"TOKENIZER_LOAD_FAILED: {exc}", flush=True)
    return None


def load_aln_pool_sample_and_stats(max_rows: int = 200000) -> dict[str, Any]:
    stats = {"path": rel(ALN_10M), "exists": ALN_10M.exists()}
    if not ALN_10M.exists():
        return stats
    n = 0; words = 0; sources = collections.Counter(); aln_rows = 0; filler_rows = 0
    sample = []
    with ALN_10M.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            n += 1; words += int(rec.get("words", len(rec.get("text", "").split())))
            sources[rec.get("source", "unknown")] += 1
            if rec.get("source") == "qwen_aligned":
                aln_rows += 1
                if len(sample) < 3:
                    sample.append(rec.get("text", "")[:350])
            else:
                filler_rows += 1
    stats.update({"rows": n, "words": words, "source_counts": dict(sources), "qwen_aligned_rows": aln_rows, "filler_rows": filler_rows, "sha256": sha256_file(ALN_10M), "sample_aln_text": sample})
    if ALN_META.exists():
        meta_n = 0; meta_words = 0; pairs = 0
        with ALN_META.open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line); meta_n += 1; meta_words += int(r.get("words", 0)); pairs += int(r.get("n_pairs", 0))
        stats.update({"packed_meta_rows": meta_n, "packed_meta_words": meta_words, "packed_meta_pairs": pairs, "packed_meta_path": rel(ALN_META)})
    return stats


def prototype_replacement_budget(aln_stats: dict[str, Any], n_contrast_pairs: int, pair_records: list[dict[str, Any]]) -> dict[str, Any]:
    # A paired contrast contributes two rows. To preserve ALN, it should displace official filler rows only.
    # For a pilot materialization, use exact 160-word rows padded with neutral official snippets later; here record
    # the raw context lengths and the maximum possible filler replacement under the matched candidates found.
    ctx_words = [r["update_context_words"] + r["retain_context_words"] for r in pair_records]
    raw_words = sum(ctx_words)
    return {
        "n_same_source_contrast_pairs": n_contrast_pairs,
        "raw_two_row_words_total": raw_words,
        "raw_words_mean_per_pair": round(statistics.mean(ctx_words), 2) if ctx_words else None,
        "raw_words_p95_per_pair": round(sorted(ctx_words)[int(0.95 * (len(ctx_words)-1))], 2) if ctx_words else None,
        "suggested_next_materialization": "replace official filler rows in the 10M ALN pool with 160-word rows containing one contrast packet per row; do not remove qwen_aligned rows until ALN expansion comparator has been separated.",
        "aln_rows_available": aln_stats.get("qwen_aligned_rows"),
        "official_filler_rows_available": aln_stats.get("filler_rows"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-sample", type=int, default=60)
    ap.add_argument("--min-pair-candidates", type=int, default=1)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _public_path('research/notes/functional_learning').mkdir(parents=True, exist_ok=True)

    tokenizer = maybe_load_tokenizer()
    counts = [summarize_counts(p) for p in [PLAIN_ALL, PLAIN_BALANCED, PLAIN_TRAIN, PLAIN_HELD] if p.exists()]
    packets = load_packets(PLAIN_ALL)

    same_pair_candidates = []
    rejection_counts = collections.Counter()
    quality_rows = []
    for pid, d in packets.items():
        if {"UPDATED_USE", "UNCHANGED_DISTRACTOR_USE"} <= set(d):
            upd = d["UPDATED_USE"].rec
            ret = d["UNCHANGED_DISTRACTOR_USE"].rec
            ok, reasons = is_contrast_candidate(upd, ret)
            if ok:
                q_upd = packet_quality(upd, tokenizer)
                q_ret = packet_quality(ret, tokenizer)
                rec = {
                    "pair_id": pid,
                    "source_sentence": norm(upd.get("source_sentence", "")),
                    "target_entity": upd.get("target_entity"),
                    "update_updated_entity": upd.get("updated_entity"),
                    "retain_updated_entity": ret.get("updated_entity"),
                    "update_answer": q_upd["answer_phrase"],
                    "retain_answer": q_ret["answer_phrase"],
                    "update_foil": q_upd["foil_phrase"],
                    "retain_foil": q_ret["foil_phrase"],
                    "update_context_words": q_upd["context_words"],
                    "retain_context_words": q_ret["context_words"],
                    "update_answer_piece_count": q_upd["answer_piece_count"],
                    "retain_answer_piece_count": q_ret["answer_piece_count"],
                    "update_masked_words": q_upd["masked_words_in_use"],
                    "retain_masked_words": q_ret["masked_words_in_use"],
                    "update_use_update_lcs": q_upd["use_update_lcs_words"],
                    "retain_use_update_lcs": q_ret["use_update_lcs_words"],
                    "update_use_source_lcs": q_upd["use_source_lcs_words"],
                    "retain_use_source_lcs": q_ret["use_source_lcs_words"],
                    "update_masked_text": q_upd["masked_text"],
                    "retain_masked_text": q_ret["masked_text"],
                    "update_raw": upd,
                    "retain_raw": ret,
                }
                same_pair_candidates.append(rec)
                quality_rows.append({k: v for k, v in rec.items() if not k.endswith("_raw") and not k.endswith("masked_text") and k != "source_sentence"})
            else:
                for r in reasons:
                    rejection_counts[r] += 1

    # Also record a less strict availability view: same source and same target missing because generated packets often
    # generated only one accepted type per pair. This tells whether new generation needs paired prompts.
    same_source_both_count = sum(1 for pid, d in packets.items() if {"UPDATED_USE", "UNCHANGED_DISTRACTOR_USE"} <= set(d))

    rng = random.Random(31031)
    sample = same_pair_candidates[:]
    rng.shuffle(sample)
    sample = sample[: args.max_sample]
    with (_public_path('experiments/archive/functional_learning/data/contrastive_entity_binding/same_pair_contrast_candidates_sample.jsonl')).open("w", encoding="utf-8") as f:
        for r in sample:
            out = dict(r)
            # Keep raw records for sample inspection, but they are large; JSONL is still manageable.
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    if quality_rows:
        with (_public_path('experiments/archive/functional_learning/data/contrastive_entity_binding/same_pair_contrast_candidates_quality.csv')).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(quality_rows[0].keys()))
            w.writeheader(); w.writerows(quality_rows)

    aln_stats = load_aln_pool_sample_and_stats()
    budget = prototype_replacement_budget(aln_stats, len(same_pair_candidates), same_pair_candidates)

    # Summaries of the accepted single-packet set: how often an always-source or always-recent strategy solves each type.
    single_quality = []
    for d in packets.values():
        for typ in ["UPDATED_USE", "UNCHANGED_DISTRACTOR_USE"]:
            if typ in d:
                q = packet_quality(d[typ].rec, tokenizer)
                single_quality.append(q)
    by_type: dict[str, dict[str, Any]] = {}
    for typ in ["UPDATED_USE", "UNCHANGED_DISTRACTOR_USE"]:
        rows = [q for q in single_quality if q["packet_type"] == typ]
        if not rows:
            continue
        by_type[typ] = {
            "n": len(rows),
            "answer_in_source_frac": sum(bool(q["source_has_answer_exact"]) for q in rows) / len(rows),
            "answer_in_update_frac": sum(bool(q["update_has_answer_exact"]) for q in rows) / len(rows),
            "foil_in_source_frac": sum(bool(q["source_has_foil_exact"]) for q in rows) / len(rows),
            "foil_in_update_frac": sum(bool(q["update_has_foil_exact"]) for q in rows) / len(rows),
            "masked_words_missing_frac": sum(q["masked_words_in_use"] == 0 for q in rows) / len(rows),
            "cue_hit_frac": sum(bool(q["cue_hits_use"]) for q in rows) / len(rows),
            "mean_answer_piece_count": statistics.mean(q["answer_piece_count"] for q in rows),
            "mean_use_source_lcs": statistics.mean(q["use_source_lcs_words"] for q in rows),
            "mean_use_update_lcs": statistics.mean(q["use_update_lcs_words"] for q in rows),
        }

    design = {
        "status": "CONTRASTIVE_ENTITY_BINDING_AUDIT",
        "created_by": "functional_learning",
        "scientific_question": "Can we construct natural-language experience where only entity-specific update binding, not source retention or recency copying, solves both update and retention targets?",
        "source_packet_files": [rel(p) for p in [PLAIN_ALL, PLAIN_BALANCED, PLAIN_TRAIN, PLAIN_HELD] if p.exists()],
        "count_summaries": counts,
        "same_pair_both_types": same_source_both_count,
        "strict_same_source_contrast_candidates": len(same_pair_candidates),
        "strict_rejection_counts": dict(rejection_counts.most_common(20)),
        "single_packet_actual_mask_summary": by_type,
        "aln_pool_stats": aln_stats,
        "budget_prototype": budget,
        "outputs": {
            "summary_json": rel(_public_path('experiments/archive/functional_learning/data/contrastive_entity_binding/contrastive_binding_audit_summary.json')),
            "candidate_sample_jsonl": rel(_public_path('experiments/archive/functional_learning/data/contrastive_entity_binding/same_pair_contrast_candidates_sample.jsonl')),
            "quality_csv": rel(_public_path('experiments/archive/functional_learning/data/contrastive_entity_binding/same_pair_contrast_candidates_quality.csv')) if quality_rows else None,
            "note": rel(NOTE),
        },
        "interpretation": {
            "existing_plainuse_training_status": "research shows the existing replacement arm is not a direct route: it improved retention-like margins while target-update use was neutral/negative.",
            "if_strict_candidates_few": "Generate paired prompts explicitly rather than recombining independent updated/distractor generations; the pair must share source and target entity so that changing only updated_entity changes the correct masked state.",
            "first_training_readout": "neutral-adjusted update margin and neutral-adjusted retention margin on held-out same-source paired packets; both must improve, because an average trading update against retention repeats the research failure.",
        },
    }
    write_json(_public_path('experiments/archive/functional_learning/data/contrastive_entity_binding/contrastive_binding_audit_summary.json'), design)

    def fmt(x: Any) -> str:
        if x is None:
            return "NA"
        if isinstance(x, float):
            return f"{x:.4f}"
        return str(x)

    lines = []
    lines.append("# research contrastive entity-binding intervention design\n")
    lines.append("This note records a CPU audit and prototype for the next natural-language intervention after the state-update route weakened. It is not a training result and it does not use the pending pa87 official evaluations.\n")
    lines.append("## Why the old state-update packet cannot be bundled with parent anchoring\n")
    lines.append("The state-update synthesis shows that the replacement packet family strengthened source-state retention while neutral-adjusted target-update use was negative in both seeds. It also displaced the inherited ALN qwen-pair ingredient and was slightly under 100M words. Parent anchoring could preserve a parent function, but it cannot make a packet require entity-specific updating when a source-retention or recency shortcut solves much of the exposure.\n")
    lines.append("## What this audit measured\n")
    lines.append("The script read the validated research plain-use packets and looked for same-pair UPDATED_USE and UNCHANGED_DISTRACTOR_USE examples with identical source sentence and target entity. It then built masked contexts by masking only the target state phrase in the final use sentence, using the actual compliant16k tokenizer when available.\n")
    lines.append("\nPacket file counts:\n\n")
    lines.append("| file | rows | UPDATED | RETAIN | unique pair IDs | same-pair both types |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    for c in counts:
        tc = c.get("type_counts", {})
        lines.append(f"| `{c['path']}` | {c['rows']} | {tc.get('UPDATED_USE',0)} | {tc.get('UNCHANGED_DISTRACTOR_USE',0)} | {c['unique_pair_ids']} | {c['same_pair_both_types']} |\n")
    lines.append("\n")
    lines.append(f"Strict same-source/same-target contrast candidates found: **{len(same_pair_candidates)}**. Rejection counts among same-pair opportunities: `{dict(rejection_counts.most_common(8))}`.\n\n")
    lines.append("Single-packet masked-input audit:\n\n")
    lines.append("| type | n | answer in source | answer in update | foil in source | foil in update | missing mask | cue hits | mean answer pieces | mean use-source LCS | mean use-update LCS |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for typ, s in by_type.items():
        lines.append(f"| `{typ}` | {s['n']} | {fmt(s['answer_in_source_frac'])} | {fmt(s['answer_in_update_frac'])} | {fmt(s['foil_in_source_frac'])} | {fmt(s['foil_in_update_frac'])} | {fmt(s['masked_words_missing_frac'])} | {fmt(s['cue_hit_frac'])} | {fmt(s['mean_answer_piece_count'])} | {fmt(s['mean_use_source_lcs'])} | {fmt(s['mean_use_update_lcs'])} |\n")
    lines.append("\n")
    lines.append("This table makes the scientific risk concrete. In the intended UPDATED row the correct state is normally present in the update sentence, so a recency/copy rule can help. In the intended RETAIN row the correct state is present in the original source while an irrelevant entity's new state is present in the update sentence, so a source-retention rule can help. A useful intervention must improve both rows in held-out pair combinations; one-sided gains are not evidence of operation binding.\n")
    lines.append("## ALN preservation and budget direction\n")
    lines.append(f"The inherited ALN pool exists at `{rel(ALN_10M)}` with {aln_stats.get('rows')} rows and {aln_stats.get('words')} words; the audit counted {aln_stats.get('qwen_aligned_rows')} `qwen_aligned` rows and {aln_stats.get('filler_rows')} filler rows. Its SHA256 is `{aln_stats.get('sha256')}`. The next intervention should add contrastive packets from filler rows, not remove this ALN structure.\n")
    lines.append("## Required packet structure for the next GPU-worthy arm\n")
    lines.append("For each source and queried entity, create a paired contrast with the same source sentence and same target entity:\n\n")
    lines.append("- UPDATE row: source state S(e) is followed by an update to the target entity e, and the masked use sentence should prefer the new state U(e).\n")
    lines.append("- RETAIN row: the same source state S(e) is followed by an update to a different entity d, and the masked use sentence should prefer S(e), not U(d).\n")
    lines.append("- The use sentence must contain no temporal/persistence cues such as now/still/remain/current/new, and the masked state span must be the only occurrence of the answer in the use sentence.\n")
    lines.append("- The two rows should be held out by source/entity combination, not merely by sentence text, so that held-out scoring tests recombination of entity selector and update operation.\n\n")
    lines.append("The first result should be a vector, not a scalar: neutral-adjusted UPDATE margin and neutral-adjusted RETAIN margin on held-out same-source pairs. Both must move in the useful direction; an average that trades update for retention repeats the research failure.\n")
    lines.append("## Files\n")
    for k, v in design["outputs"].items():
        if v:
            lines.append(f"- {k}: `{v}`\n")
    NOTE.write_text("".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": design["status"],
        "same_pair_both_types": same_source_both_count,
        "strict_candidates": len(same_pair_candidates),
        "summary_json": rel(_public_path('experiments/archive/functional_learning/data/contrastive_entity_binding/contrastive_binding_audit_summary.json')),
        "note": rel(NOTE),
        "sample": rel(_public_path('experiments/archive/functional_learning/data/contrastive_entity_binding/same_pair_contrast_candidates_sample.jsonl')),
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
