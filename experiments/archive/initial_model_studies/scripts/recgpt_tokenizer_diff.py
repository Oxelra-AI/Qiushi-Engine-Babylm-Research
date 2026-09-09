#!/usr/bin/env python3
"""research: compare public RecGPT vs official-corpus RecGPT tokenizers on BabyLM probe words.

This is exploratory evidence for the RecGPT phenotype differential. It does not
change any model or evaluation artifact.
"""
from __future__ import annotations
import csv, json, pathlib, statistics
from transformers import AutoTokenizer

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
PUBLIC_TOK = ROOT / "data/recgpt_local/patched_model"
OFFICIAL_TOK = ROOT / "data/recgpt_official_tokenizer"
READING = ROOT / "repos/babylm-eval/strict/evaluation_data/full_eval/reading/reading_data.csv"
AOA = ROOT / "repos/babylm-eval/strict/evaluation_data/full_eval/aoa/cdi_childes.json"
OUT = ROOT / "data/recgpt_tokenizer_diff.json"


def reading_words():
    with READING.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    candidates = ["word", "target", "target_word", "Word", "item"]
    col = next((c for c in candidates if c in rows[0]), None)
    if col is None:
        # Fall back to the first short alphabetic-ish column.
        for c in rows[0]:
            vals = [r.get(c, "") for r in rows[:50]]
            if vals and sum(1 for v in vals if v and len(v.split()) == 1) > 30:
                col = c; break
    return [r[col].strip() for r in rows if r.get(col, '').strip()]


def aoa_words():
    data = json.loads(AOA.read_text(encoding='utf-8'))
    words = []
    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k in {"word", "target_word", "target"} and isinstance(v, str):
                    words.append(v.strip())
                else:
                    walk(v)
        elif isinstance(x, list):
            for y in x: walk(y)
    walk(data)
    if not words and isinstance(data, dict):
        words.extend([str(k) for k in data.keys()])
    return [w for w in words if w]


def stats_for(tok, words):
    no_lens=[]; sp_lens=[]; no_single=0; sp_single=0; differ=0
    examples=[]
    for w in words:
        ids0 = tok.encode(w, add_special_tokens=False)
        ids1 = tok.encode(' ' + w, add_special_tokens=False)
        no_lens.append(len(ids0)); sp_lens.append(len(ids1))
        no_single += (len(ids0) == 1); sp_single += (len(ids1) == 1); differ += (ids0 != ids1)
        if len(examples) < 12 and (len(ids0) != len(ids1) or len(ids1) > 1):
            examples.append({"word": w, "no_space": ids0, "with_space": ids1, "no_tokens": tok.convert_ids_to_tokens(ids0), "space_tokens": tok.convert_ids_to_tokens(ids1)})
    n=len(words)
    return {
        "n": n,
        "no_space_len_mean": statistics.mean(no_lens),
        "with_space_len_mean": statistics.mean(sp_lens),
        "no_space_single_frac": no_single/n,
        "with_space_single_frac": sp_single/n,
        "space_ids_differ_frac": differ/n,
        "p95_no_space_len": sorted(no_lens)[int(0.95*(n-1))],
        "p95_with_space_len": sorted(sp_lens)[int(0.95*(n-1))],
        "examples": examples,
    }


def compare_tokenizers(words):
    pub = AutoTokenizer.from_pretrained(str(PUBLIC_TOK), trust_remote_code=True, use_fast=True)
    off = AutoTokenizer.from_pretrained(str(OFFICIAL_TOK), use_fast=True)
    rows=[]
    for w in words:
        rows.append({
            "word": w,
            "public_with_space_len": len(pub.encode(' '+w, add_special_tokens=False)),
            "official_with_space_len": len(off.encode(' '+w, add_special_tokens=False)),
            "public_no_space_len": len(pub.encode(w, add_special_tokens=False)),
            "official_no_space_len": len(off.encode(w, add_special_tokens=False)),
        })
    return {
        "public_shorter_with_space_frac": sum(r["public_with_space_len"] < r["official_with_space_len"] for r in rows)/len(rows),
        "official_shorter_with_space_frac": sum(r["official_with_space_len"] < r["public_with_space_len"] for r in rows)/len(rows),
        "equal_with_space_frac": sum(r["official_with_space_len"] == r["public_with_space_len"] for r in rows)/len(rows),
        "mean_public_minus_official_with_space_len": statistics.mean(r["public_with_space_len"] - r["official_with_space_len"] for r in rows),
        "largest_public_advantage": sorted(rows, key=lambda r: r["public_with_space_len"] - r["official_with_space_len"])[:20],
        "largest_official_advantage": sorted(rows, key=lambda r: r["official_with_space_len"] - r["public_with_space_len"])[:20],
    }


def main():
    rw = reading_words()
    aw = aoa_words()
    wordsets = {
        "reading_targets": rw,
        "aoa_words": aw,
        "union": sorted(set(rw) | set(aw)),
    }
    pub = AutoTokenizer.from_pretrained(str(PUBLIC_TOK), trust_remote_code=True, use_fast=True)
    off = AutoTokenizer.from_pretrained(str(OFFICIAL_TOK), use_fast=True)
    out = {"public_tokenizer": str(PUBLIC_TOK), "official_tokenizer": str(OFFICIAL_TOK), "sets": {}}
    for name, words in wordsets.items():
        out["sets"][name] = {
            "public": stats_for(pub, words),
            "official": stats_for(off, words),
            "comparison": compare_tokenizers(words),
        }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT), "summary": {k: {"n": v["public"]["n"], "public_space_single": v["public"]["with_space_single_frac"], "official_space_single": v["official"]["with_space_single_frac"], "mean_pub_minus_official_space_len": v["comparison"]["mean_public_minus_official_with_space_len"]} for k, v in out["sets"].items()}}, indent=2))

if __name__ == "__main__":
    main()
