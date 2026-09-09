#!/usr/bin/env python3
"""research: Analyze expansion compact-view generation outputs and merge with existing accepted pairs.

Reuses research acceptance criteria exactly. Merges GPU0+GPU1 outputs, applies acceptance,
combines with the existing medium accepted pool, and reports dose capacity.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import pathlib
import random
import re
import statistics
from typing import Any, Iterable

ROOT = pathlib.Path("experiments/archive/frontier_consolidation")

# Existing accepted pairs
EXISTING_ACCEPTED = ROOT / "data/medium_compact_analysis/medium_compact_ws_accepted_rewrites.jsonl"
EXISTING_ROWS = ROOT / "data/medium_compact_analysis/medium_compact_ws_rows.jsonl"

# Expansion generation outputs
GPU0_PROMPTS = ROOT / "data/dose_expansion_prompts/expansion_compact_prompts_gpu0.jsonl"
GPU1_PROMPTS = ROOT / "data/dose_expansion_prompts/expansion_compact_prompts_gpu1.jsonl"
GPU0_OUTPUTS = ROOT / "training/runs/expansion_compact_qwen_gpu0/outputs.jsonl"
GPU1_OUTPUTS = ROOT / "training/runs/expansion_compact_qwen_gpu1/outputs.jsonl"

OUT_DIR = ROOT / "data/expansion_analysis"

# === research acceptance criteria (exact copy) ===
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[''][A-Za-z0-9]+)?|[£$€]\s*\d+(?:\.\d+)?|\d+(?:[.,:/\-]\d+)*(?:%|st|nd|rd|th)?")
NUM_RE = re.compile(r"(?:[£$€]\s*)?\d+(?:[.,:/\-]\d+)*(?:%|st|nd|rd|th)?", re.I)
CAP_SEQ_RE = re.compile(r"\b(?:[A-Z][A-Za-z0-9.&''\-]+(?:\s+|$)){1,7}")
STOPWORDS = {
    "the","a","an","and","or","but","if","then","of","to","in","on","for","with","as","by","from","at",
    "is","are","was","were","be","been","being","it","its","this","that","these","those","he","she","they","we",
    "you","i","his","her","their","our","your","not","no","so","than","into","about","can","could","may","might",
    "will","would","should","has","have","had","do","does","did","who","which","what","when","where","why","how",
    "one","two","new","old","more","most","many","much","some","such","very","only","also","between","during","after",
    "study","studies","research","show","shows","using","used","use","made","make","called","known",
}
ENTITY_STOP = {
    "The","A","An","This","That","These","Those","In","On","For","At","By","From","To","And","But","Or","If",
    "When","While","Because","SOURCE SENTENCE","FineWeb","Sentence","English","Study","Research","Both","Although",
}
BAD_OUTPUT_PATTERNS = [
    re.compile(p, re.I) for p in [
        r"^\s*(sure|here('| i)s|certainly|of course)\b",
        r"as an ai", r"i (cannot|can't)", r"please provide", r"output only", r"rewrite the factual", r"source sentence",
        r"simplified sentence\s*:", r"the original sentence", r"not mentioned in the sentence", r"cannot determine",
        r"\[.*\]", r"^\s*[-*•]", r"\n\s*[-*•]",
    ]
]

def lexical_tokens(text):
    return [m.group(0).lower() for m in WORD_RE.finditer(text or "")]

def content_tokens(text):
    return [t for t in lexical_tokens(text) if len(t) > 2 and t not in STOPWORDS and not NUM_RE.fullmatch(t)]

def normalized_phrase(text):
    return " ".join(re.findall(r"[a-z0-9]+", (text or "").lower()))

def capital_entities(text):
    ents = set()
    for m in CAP_SEQ_RE.finditer(text or ""):
        e = " ".join(m.group(0).split()).strip(" .,:;!?()[]{}\"'")
        if not e or e in ENTITY_STOP or len(e) <= 2:
            continue
        ents.add(normalized_phrase(e))
    return ents

def sig_entity_tokens(text):
    return {t for t in content_tokens(text) if len(t) > 2}

def normalize_num(s):
    s = str(s).lower().replace(",", "")
    s = re.sub(r"\s+", "", s)
    s = s.replace("percent", "%")
    return s

def numbers(text):
    return {normalize_num(m.group(0)) for m in NUM_RE.finditer(text or "")}

def source_entity_preserved(entity, output):
    ent_norm = normalized_phrase(entity)
    out_norm = normalized_phrase(output)
    if not ent_norm: return True
    if ent_norm in out_norm: return True
    ent_toks = sig_entity_tokens(entity)
    if not ent_toks: return True
    out_toks = set(content_tokens(output))
    kept = len(ent_toks & out_toks)
    if len(ent_toks) <= 2: return kept == len(ent_toks)
    return kept / len(ent_toks) >= 0.75

def content_recall_fn(source, output):
    src = set(content_tokens(source))
    out = set(content_tokens(output))
    return len(src & out) / len(src) if src else 1.0

def overlap_fn(a, b):
    aa, bb = set(content_tokens(a)), set(content_tokens(b))
    if not aa and not bb: return 1.0
    if not aa or not bb: return 0.0
    return len(aa & bb) / len(aa | bb)

def wc(text): return len((text or "").split())

def clean_output(text):
    text = (text or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\r\n\"'")

def structure_flags(raw_output, cleaned):
    flags = []
    if not cleaned:
        flags.append("empty_output"); return flags
    if any(rx.search(raw_output or "") for rx in BAD_OUTPUT_PATTERNS):
        flags.append("bad_output_pattern")
    if "\n" in (raw_output or "").strip():
        flags.append("multi_line_output")
    if not (cleaned[0].isupper() or cleaned[0].isdigit() or cleaned[0] in "\"'"):
        flags.append("bad_start")
    if cleaned[-1] not in ".!?\"'":
        flags.append("bad_end")
    terminal_marks = len(re.findall(r"[.!?](?:\s|$)", cleaned))
    if terminal_marks >= 3: flags.append("many_sentences")
    if re.search(r"\b(first|second|third|for example)\b", cleaned, re.I) and terminal_marks >= 2:
        flags.append("explanatory_expansion")
    return flags

def analyze_one(prompt, raw_output, idx):
    source = " ".join(str(prompt.get("source_text") or "").split())
    out = clean_output(raw_output)
    src_ws = wc(source)
    out_ws = wc(out)
    ratio_ws = out_ws / max(1, src_ws)
    src_nums = set(map(normalize_num, prompt.get("source_numbers") or [])) | numbers(source)
    out_nums = numbers(out)
    src_entities = [str(x) for x in (prompt.get("source_entities") or []) if str(x).strip()]
    preserved = [e for e in src_entities if source_entity_preserved(e, out)]
    missing = [e for e in src_entities if e not in preserved]
    nrec = len(src_nums & out_nums) / max(1, len(src_nums)) if src_nums else 1.0
    erec = len(preserved) / max(1, len(src_entities)) if src_entities else 1.0
    ov = overlap_fn(source, out)
    crec = content_recall_fn(source, out)
    out_caps = capital_entities(out)
    src_caps = capital_entities(source) | {normalized_phrase(e) for e in src_entities if normalized_phrase(e)}
    new_ent = sorted(e for e in out_caps if e not in src_caps and not any(e in s or s in e for s in src_caps))

    hard = []
    soft = []
    hard.extend(structure_flags(raw_output, out))
    if ratio_ws < 0.35 or ratio_ws > 1.45: hard.append(f"length_ratio_{ratio_ws:.2f}")
    if nrec < 1.0: hard.append(f"number_recall_{nrec:.2f}")
    if out_nums - src_nums: hard.append(f"new_numbers_{len(out_nums - src_nums)}")
    if src_entities and erec < (1.0 if len(src_entities) <= 2 else 0.75): hard.append(f"entity_recall_{erec:.2f}")
    if len(new_ent) > 3: hard.append(f"new_entity_like_{len(new_ent)}")
    if ov < 0.10 and src_ws >= 10: hard.append(f"low_overlap_{ov:.2f}")
    if crec < 0.28 and src_ws >= 14: hard.append(f"low_content_recall_{crec:.2f}")
    if ov > 0.92 and ratio_ws > 0.80 and src_ws >= 14: soft.append("near_copy_view")
    if ratio_ws > 1.05: soft.append("not_shorter_than_source")

    return {
        "prompt_id": prompt.get("prompt_id"),
        "sentence_id": prompt.get("sentence_id"),
        "doc_id": prompt.get("doc_id"),
        "source_text": source,
        "rewrite_text": out,
        "source_words": src_ws,
        "rewrite_words": out_ws,
        "pair_words": src_ws + out_ws,
        "length_ratio": ratio_ws,
        "content_recall": crec,
        "content_overlap": ov,
        "entity_recall": erec,
        "number_recall": nrec,
        "domain_hits": prompt.get("domain_hits") or [],
        "soft_flags": soft,
        "hard_reasons": hard,
        "accepted_for_next_construction": not hard,
    }

def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Analyze expansion outputs ----
    all_expansion_rows = []
    for gpu_label, ppath, opath in [("gpu0", GPU0_PROMPTS, GPU0_OUTPUTS), ("gpu1", GPU1_PROMPTS, GPU1_OUTPUTS)]:
        if not opath.exists():
            print(f"WARNING: {opath} not found, skipping {gpu_label}", flush=True)
            continue
        prompts = read_jsonl(ppath)
        outputs = read_jsonl(opath)
        if len(outputs) < len(prompts):
            raise RuntimeError(f"{gpu_label}: outputs {len(outputs)} < prompts {len(prompts)}")
        for i, pr in enumerate(prompts):
            raw = str(outputs[i].get("output") or outputs[i].get("generated_text") or outputs[i].get("text") or outputs[i].get("completion") or "")
            row = analyze_one(pr, raw, len(all_expansion_rows))
            row["gpu_source"] = gpu_label
            all_expansion_rows.append(row)

    expansion_accepted = [r for r in all_expansion_rows if r["accepted_for_next_construction"]]
    expansion_rejected = [r for r in all_expansion_rows if not r["accepted_for_next_construction"]]

    # Save expansion analysis
    expansion_rows_path = OUT_DIR / "expansion_generation_rows.jsonl"
    expansion_accepted_path = OUT_DIR / "expansion_accepted_rewrites.jsonl"
    with expansion_rows_path.open("w", encoding="utf-8") as f:
        for r in all_expansion_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with expansion_accepted_path.open("w", encoding="utf-8") as f:
        for r in expansion_accepted:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- Merge with existing accepted ----
    existing = read_jsonl(EXISTING_ACCEPTED)
    existing_ids = {r.get("prompt_id") or r.get("sentence_id") for r in existing}

    # Combined pool: existing first, then expansion
    combined = list(existing)
    new_added = 0
    for r in expansion_accepted:
        pid = r.get("prompt_id") or r.get("sentence_id")
        if pid not in existing_ids:
            combined.append(r)
            existing_ids.add(pid)
            new_added += 1

    combined_path = OUT_DIR / "combined_all_accepted_pairs.jsonl"
    with combined_path.open("w", encoding="utf-8") as f:
        for r in combined:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- Dose capacity ----
    total_pair_words = sum(r.get("pair_words", 0) for r in combined)
    base_changed_block = 423520  # 1x dose
    max_dose = total_pair_words / base_changed_block

    exp_pair_words = sum(r.get("pair_words", 0) for r in expansion_accepted)

    reasons = collections.Counter()
    for r in expansion_rejected:
        for hr in r.get("hard_reasons", []):
            for pref in ["length_ratio","number_recall","new_numbers","entity_recall","new_entity_like","low_overlap","low_content_recall"]:
                if hr.startswith(pref):
                    reasons[pref] += 1
                    break
            else:
                reasons[hr] += 1

    meta = {
        "status": "EXPANSION_ANALYZED_AND_MERGED",
        "expansion_total": len(all_expansion_rows),
        "expansion_accepted": len(expansion_accepted),
        "expansion_rejected": len(expansion_rejected),
        "expansion_acceptance_rate": len(expansion_accepted) / max(1, len(all_expansion_rows)),
        "expansion_pair_words": exp_pair_words,
        "top_rejection_reasons": reasons.most_common(20),
        "existing_accepted": len(existing),
        "combined_total": len(combined),
        "combined_pair_words": total_pair_words,
        "base_changed_block_words": base_changed_block,
        "max_dose_multiple": round(max_dose, 3),
        "dose_capacity": {
            "1x": {"pairs_needed": 12155, "pair_words": base_changed_block, "feasible": True},
            "2x": {"pairs_needed": "~24k", "pair_words": base_changed_block * 2, "feasible": total_pair_words >= base_changed_block * 2},
            "3x": {"pairs_needed": "~37k", "pair_words": base_changed_block * 3, "feasible": total_pair_words >= base_changed_block * 3},
        },
        "files": {
            "expansion_rows": str(expansion_rows_path),
            "expansion_accepted": str(expansion_accepted_path),
            "combined_accepted": str(combined_path),
        },
    }
    meta_path = OUT_DIR / "expansion_analysis_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({k: v for k, v in meta.items() if k != "files"}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
