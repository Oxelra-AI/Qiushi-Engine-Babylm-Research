#!/usr/bin/env python3
"""research: fast tokenizer representation map.

Reuses the pre-trained 24k/32k/40k tokenizers from research plus existing 16k
tokenizers.  Computes ONLY aggregate tokenization statistics (tokens/word,
WWM-groups/word) on the 10M training pool and official eval text families.
Skips the slow per-row visible_prefix_stats binary search that caused research
timeout.  Produces a JSON/CSV comparison table.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import math
import pathlib
import time
from typing import Any, Iterable

from transformers import AutoTokenizer

STUDY = pathlib.Path("experiments/archive/representation_and_objectives")
WS = STUDY
OUT = WS / "data/fast_representation_map"
NOTE = (WS.parents[2] / 'research/notes/representation_and_objectives/fast_representation_map.md')

# Pre-trained tokenizers
TOK_DIRS = {
    "old_inherited_16k": pathlib.Path("experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model"),
    "legal_a01_16k": pathlib.Path("experiments/archive/representation_and_objectives/data/strictsmall_tokenizer_retrain/strictsmall_compact_reinvest_16k_tokenizer"),
    "legal_byte_bpe_24k": pathlib.Path("experiments/archive/representation_and_objectives/data/legal_representation_route_map/tokenizers/legal_byte_bpe_24k"),
    "legal_byte_bpe_32k": pathlib.Path("experiments/archive/representation_and_objectives/data/legal_representation_route_map/tokenizers/legal_byte_bpe_32k"),
    "legal_byte_bpe_40k": pathlib.Path("experiments/archive/representation_and_objectives/data/legal_representation_route_map/tokenizers/legal_byte_bpe_40k"),
}

# Byte-alphabet 16k if available
A02_BYTE16 = pathlib.Path("experiments/archive/frontier_consolidation/data/compliant_tokenizer_bytealphabet")
if A02_BYTE16.exists():
    TOK_DIRS["legal_a02_bytealpha_16k"] = A02_BYTE16

POOL_10M = pathlib.Path("experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl")
PRISTINE_FULL = pathlib.Path("experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval")
GLOBALPIQA_FULL = pathlib.Path("experiments/archive/representation_and_objectives/data/globalpiqa_official_lineage/official_dl_scratch/generated_by_current_official_dl/evaluation_data/full_eval")

SEQ256 = 256


def iter_jsonl(path: pathlib.Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def is_word_start(t: str) -> bool:
    return t.startswith("Ġ") or t.startswith("▁")


def token_stats(tok, text: str) -> tuple[int, int, int]:
    """Returns (n_tokens, n_wwm_groups, n_whitespace_words)."""
    ids = tok(text, add_special_tokens=False, truncation=False)["input_ids"]
    n_tok = len(ids)
    toks_str = tok.convert_ids_to_tokens(ids) if ids else []
    n_groups = 0
    for i, t in enumerate(toks_str):
        if i == 0 or is_word_start(str(t)):
            n_groups += 1
    n_words = len(text.split())
    return n_tok, n_groups, n_words


def collect_eval_texts() -> dict[str, list[str]]:
    """Collect representative eval texts grouped by family. Keep it lean."""
    families: dict[str, list[str]] = collections.defaultdict(list)

    # BLiMP + Supplement
    for family, dname in [("BLiMP", "blimp_filtered"), ("Supplement", "supplement_filtered")]:
        d = PRISTINE_FULL / dname
        if d.exists():
            for p in sorted(d.glob("*.jsonl")):
                for obj in iter_jsonl(p):
                    for key in ["sentence_good", "sentence_bad"]:
                        if key in obj:
                            families[family].append(str(obj[key]))

    # EWoK
    d = PRISTINE_FULL / "ewok_filtered"
    if d.exists():
        for p in sorted(d.glob("*.jsonl")):
            for obj in iter_jsonl(p):
                for key in ["Context1", "Context2", "Target1", "Target2"]:
                    if key in obj:
                        families["EWoK"].append(str(obj[key]))
                # Also add context+target concatenations (what the model sees)
                if all(k in obj for k in ["Context1", "Target1"]):
                    families["EWoK_scored"].append(str(obj["Context1"]) + " " + str(obj["Target1"]))
                if all(k in obj for k in ["Context2", "Target2"]):
                    families["EWoK_scored"].append(str(obj["Context2"]) + " " + str(obj["Target2"]))

    # Entity
    d = PRISTINE_FULL / "entity_tracking"
    if d.exists():
        for p in sorted(d.glob("*.jsonl")):
            for obj in iter_jsonl(p):
                prefix = str(obj.get("input_prefix", ""))
                if prefix:
                    families["Entity"].append(prefix)

    # COMPS
    d = PRISTINE_FULL / "comps"
    if d.exists():
        for p in sorted(d.glob("*.jsonl")):
            for obj in iter_jsonl(p):
                prop = str(obj.get("property_phrase", obj.get("property", "")))
                for pk in ["prefix_acceptable", "prefix_unacceptable"]:
                    if pk in obj:
                        families["COMPS"].append(str(obj[pk]) + " " + prop)

    # GlobalPIQA
    for dname in ["global_piqa_parallel", "global_piqa_nonparallel"]:
        d = GLOBALPIQA_FULL / dname
        if d.exists():
            for p in sorted(d.glob("*.jsonl")):
                for obj in iter_jsonl(p):
                    prompt = str(obj.get("prompt", ""))
                    if prompt:
                        families[f"GlobalPIQA"].append(prompt)
                    for k, v in obj.items():
                        if k.startswith("solution") and isinstance(v, str):
                            families["GlobalPIQA"].append(prompt + " " + v)

    # SuperGLUE (glue_filtered)
    gd = PRISTINE_FULL / "glue_filtered"
    if gd.exists():
        for p in sorted(gd.glob("*.jsonl")):
            task = p.name.split(".")[0]
            for obj in iter_jsonl(p):
                fields = [str(v) for k, v in obj.items() if k != "label" and isinstance(v, str)]
                if fields:
                    families[f"SuperGLUE_{task}"].append(" </s> ".join(fields))
                    families["SuperGLUE"].append(" </s> ".join(fields))

    return dict(families)


def process_texts(tok, texts: list[str], batch_size: int = 512) -> dict[str, Any]:
    """Aggregate stats with batched fast-tokenizer calls."""
    total_tokens = 0
    total_groups = 0
    total_words = 0
    n = 0
    over256 = 0
    unk_id = tok.unk_token_id
    unk_count = 0
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        enc = tok(batch, add_special_tokens=False, truncation=False, padding=False)
        ids_batch = enc["input_ids"]
        fast_encs = getattr(enc, "encodings", None)
        for j, (text, ids) in enumerate(zip(batch, ids_batch)):
            if fast_encs is not None:
                toks_str = fast_encs[j].tokens
            else:
                toks_str = tok.convert_ids_to_tokens(ids) if ids else []
            n_groups = sum(1 for i, t in enumerate(toks_str) if i == 0 or is_word_start(str(t)))
            total_tokens += len(ids)
            total_groups += n_groups
            total_words += len(text.split())
            over256 += int(len(ids) > SEQ256)
            if unk_id is not None:
                unk_count += sum(1 for x in ids if int(x) == int(unk_id))
            n += 1
    return {
        "n_texts": n,
        "total_tokens": total_tokens,
        "total_words": total_words,
        "tokens_per_word": total_tokens / max(1, total_words),
        "wwm_groups": total_groups,
        "groups_per_word": total_groups / max(1, total_words),
        "mean_tokens_per_group": total_tokens / max(1, total_groups),
        "over_seq256": over256,
        "unk_count": unk_count,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Load tokenizers
    tokenizers = {}
    for label, path in sorted(TOK_DIRS.items()):
        if path.exists():
            tokenizers[label] = AutoTokenizer.from_pretrained(str(path), use_fast=True)
            print(f"  loaded {label}: vocab={tokenizers[label].vocab_size}", flush=True)
        else:
            print(f"  SKIP {label}: path not found", flush=True)

    if not tokenizers:
        print(json.dumps({"status": "ERROR", "message": "No tokenizers loaded"}))
        return

    # 1. Training pool statistics (sample first 5000 rows for speed, then full)
    print("  Processing 10M pool...", flush=True)
    pool_stats = {}
    pool_rows = []
    for row in iter_jsonl(POOL_10M):
        pool_rows.append(str(row.get("text", "")))
    print(f"    {len(pool_rows)} rows loaded", flush=True)

    for label, tok in tokenizers.items():
        print(f"    tokenizing pool with {label}...", flush=True)
        pool_stats[label] = process_texts(tok, pool_rows)

    # 2. Evaluation text statistics
    print("  Collecting eval texts...", flush=True)
    eval_families = collect_eval_texts()
    print(f"    {len(eval_families)} families, total texts: {sum(len(v) for v in eval_families.values())}", flush=True)

    eval_stats = {}
    for label, tok in tokenizers.items():
        eval_stats[label] = {}
        print(f"    eval with {label}...", flush=True)
        for fam, texts in sorted(eval_families.items()):
            eval_stats[label][fam] = process_texts(tok, texts)

    # 3. Build comparison table relative to reference (legal_a01_16k)
    ref = "legal_a01_16k"
    comparison = {}
    for label in tokenizers:
        if label == ref:
            continue
        comp = {"pool": {}, "eval": {}}
        # Pool comparison
        r_pool = pool_stats[ref]
        o_pool = pool_stats[label]
        comp["pool"] = {
            "new_tokens_per_word": o_pool["tokens_per_word"],
            "ref_tokens_per_word": r_pool["tokens_per_word"],
            "ratio_tokens_per_word": o_pool["tokens_per_word"] / max(1e-9, r_pool["tokens_per_word"]),
            "new_groups_per_word": o_pool["groups_per_word"],
            "ref_groups_per_word": r_pool["groups_per_word"],
            "ratio_groups_per_word": o_pool["groups_per_word"] / max(1e-9, r_pool["groups_per_word"]),
            "new_over_seq256": o_pool["over_seq256"],
            "ref_over_seq256": r_pool["over_seq256"],
            "new_unk": o_pool["unk_count"],
        }
        # Eval comparison per family
        for fam in sorted(eval_families):
            r_ev = eval_stats[ref].get(fam, {})
            o_ev = eval_stats[label].get(fam, {})
            if r_ev and o_ev:
                comp["eval"][fam] = {
                    "new_tokens_per_word": o_ev["tokens_per_word"],
                    "ref_tokens_per_word": r_ev["tokens_per_word"],
                    "ratio": o_ev["tokens_per_word"] / max(1e-9, r_ev["tokens_per_word"]),
                    "new_groups_per_word": o_ev["groups_per_word"],
                    "ref_groups_per_word": r_ev["groups_per_word"],
                    "groups_ratio": o_ev["groups_per_word"] / max(1e-9, r_ev["groups_per_word"]),
                    "new_mean_tok_per_group": o_ev["mean_tokens_per_group"],
                    "ref_mean_tok_per_group": r_ev["mean_tokens_per_group"],
                    "new_unk": o_ev["unk_count"],
                    "new_over256": o_ev["over_seq256"],
                    "ref_over256": r_ev["over_seq256"],
                }
        comparison[f"{label}_vs_{ref}"] = comp

    # Also compare old_inherited vs legal_a01 to relate to known score movements
    if "old_inherited_16k" in tokenizers:
        old_ref = "old_inherited_16k"
        comp_old = {"pool": {}, "eval": {}}
        r_pool = pool_stats[old_ref]
        o_pool = pool_stats[ref]
        comp_old["pool"] = {
            "legal16k_tokens_per_word": o_pool["tokens_per_word"],
            "inherited16k_tokens_per_word": r_pool["tokens_per_word"],
            "ratio": o_pool["tokens_per_word"] / max(1e-9, r_pool["tokens_per_word"]),
        }
        for fam in sorted(eval_families):
            r_ev = eval_stats[old_ref].get(fam, {})
            o_ev = eval_stats[ref].get(fam, {})
            if r_ev and o_ev:
                comp_old["eval"][fam] = {
                    "legal16k_tpw": o_ev["tokens_per_word"],
                    "inherited16k_tpw": r_ev["tokens_per_word"],
                    "ratio": o_ev["tokens_per_word"] / max(1e-9, r_ev["tokens_per_word"]),
                }
        comparison["legal_a01_16k_vs_old_inherited_16k"] = comp_old

    elapsed = round(time.time() - t0, 1)

    payload = {
        "status": "FAST_REPRESENTATION_MAP",
        "elapsed_sec": elapsed,
        "tokenizers_loaded": list(tokenizers.keys()),
        "vocab_sizes": {label: tok.vocab_size for label, tok in tokenizers.items()},
        "pool_stats": pool_stats,
        "eval_stats": eval_stats,
        "comparison_vs_legal_a01_16k": comparison,
        "out_json": str(OUT / "fast_representation_map.json"),
        "note": str(NOTE),
    }

    (OUT / "fast_representation_map.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    # Write CSV comparison
    csv_path = OUT / "tokenizer_comparison.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tokenizer", "vocab_size", "pool_tpw", "pool_gpw", "pool_tpg", "pool_over256", "pool_unk"])
        for label in sorted(tokenizers):
            ps = pool_stats[label]
            writer.writerow([
                label,
                tokenizers[label].vocab_size,
                f"{ps['tokens_per_word']:.4f}",
                f"{ps['groups_per_word']:.4f}",
                f"{ps['mean_tokens_per_group']:.4f}",
                ps["over_seq256"],
                ps["unk_count"],
            ])

    # Write note
    lines = ["# research: Fast representation map\n\n"]
    lines.append("## Tokenizer training pool statistics\n\n")
    lines.append("| Tokenizer | Vocab | tokens/word | groups/word | tokens/group | over_seq256 | unk |\n")
    lines.append("|-----------|-------|-------------|-------------|--------------|-------------|-----|\n")
    for label in sorted(tokenizers):
        ps = pool_stats[label]
        lines.append(f"| {label} | {tokenizers[label].vocab_size} | {ps['tokens_per_word']:.4f} | {ps['groups_per_word']:.4f} | {ps['mean_tokens_per_group']:.4f} | {ps['over_seq256']} | {ps['unk_count']} |\n")

    lines.append("\n## Key eval family token ratios (new / legal_a01_16k)\n\n")
    key_fams = ["BLiMP", "Supplement", "EWoK", "EWoK_scored", "Entity", "COMPS", "GlobalPIQA", "SuperGLUE"]
    for cand in ["legal_byte_bpe_24k", "legal_byte_bpe_32k", "legal_byte_bpe_40k", "old_inherited_16k"]:
        ckey = f"{cand}_vs_{ref}" if cand != "old_inherited_16k" else "legal_a01_16k_vs_old_inherited_16k"
        if ckey in comparison:
            lines.append(f"\n### {cand} vs {ref}\n")
            for fam in key_fams:
                ev = comparison[ckey]["eval"].get(fam, {})
                ratio = ev.get("ratio", ev.get("ratio", "N/A"))
                lines.append(f"- {fam}: token ratio {ratio:.4f}\n" if isinstance(ratio, float) else f"- {fam}: N/A\n")

    lines.append(f"\n## Artifacts\n- JSON: `{OUT / 'fast_representation_map.json'}`\n- CSV: `{csv_path}`\n- note: `{NOTE}`\n")
    lines.append(f"\nElapsed: {elapsed}s\n")

    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text("".join(lines))

    print(json.dumps({
        "status": "FAST_REPRESENTATION_MAP_READY",
        "elapsed_sec": elapsed,
        "out_json": str(OUT / "fast_representation_map.json"),
        "csv": str(csv_path),
        "note": str(NOTE),
        "vocab_sizes": {label: tok.vocab_size for label, tok in tokenizers.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
