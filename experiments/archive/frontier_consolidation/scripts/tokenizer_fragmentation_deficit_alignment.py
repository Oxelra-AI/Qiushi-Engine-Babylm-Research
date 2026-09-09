#!/usr/bin/env python3
"""research: relate legal-tokenizer score losses to tokenization geometry.

Scientific purpose
------------------
research showed that the legal 16k tokenizer deficit is concentrated in syntax,
QA-congruence, and EWoK dynamics, and that the existing relation_only masking prior is
not aligned with those losses.  Before spending GPU on a new objective, this CPU-only
analysis asks whether the score losses are explained by tokenizer fragmentation / visible
string compression and whether candidate legal representation tokenizers (24k/32k/40k and
support-floored variants already trained only on the allowed 10M pool by representation_and_objectives) move
the affected subtasks toward the old non-submittable representation.

It DOES NOT train a tokenizer, train a model, run official evaluation, or use evaluation
text to design training data.  It only tokenizes official evaluation strings with existing
tokenizers and correlates these geometry metrics with already-measured UID score deltas.
"""
from __future__ import annotations

import json
import math
import pathlib
import statistics
from typing import Iterable, Any

from transformers import AutoTokenizer

ROOT = pathlib.Path("experiments/archive/frontier_consolidation")
DEFICIT_JSON = ROOT / "data/legal_deficit_subtask_decomposition/legal_deficit_subtask_decomposition.json"
EVAL_ROOT = pathlib.Path("experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval")
OUT_DIR = ROOT / "data/tokenizer_fragmentation_deficit_alignment"

TOKENIZERS = {
    # Non-submittable reference tokenizer from the old above-leader endpoint.
    "old_inherited_16k_nonlegal": "experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/hf_model/chck_100M",
    # Best complete legal endpoint coordinate.
    "legal_step35_16k": "experiments/archive/frontier_consolidation/data/compliant_tokenizer",
    # Legal full-byte coordinate, nearly identical to strictsmalltok in observed scores.
    "legal_bytealpha_16k": "experiments/archive/frontier_consolidation/data/compliant_tokenizer_bytealphabet",
    # Legal representation candidates trained only on the same allowed 10M pool by representation_and_objectives.
    "legal_24k": "experiments/archive/representation_and_objectives/data/legal_representation_route_map/tokenizers/legal_byte_bpe_24k",
    "legal_32k": "experiments/archive/representation_and_objectives/data/legal_representation_route_map/tokenizers/legal_byte_bpe_32k",
    "legal_40k": "experiments/archive/representation_and_objectives/data/legal_representation_route_map/tokenizers/legal_byte_bpe_40k",
    "legal_40k_minfreq25": "experiments/archive/representation_and_objectives/data/tokenizer_support_spectrum/tokenizers/legal_byte_bpe_40k_minfreq25",
    "legal_40k_minfreq50": "experiments/archive/representation_and_objectives/data/tokenizer_support_spectrum/tokenizers/legal_byte_bpe_40k_minfreq50",
}

SPECIAL_IDS = {0, 1, 2, 3, 4}


def read_jsonl(path: pathlib.Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def whitespace_words(text: str) -> int:
    return len(text.split())


def iter_texts(column: str, uid: str) -> list[str]:
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
            # Match research's convention: score the complete context+target string.
            texts.append(" ".join([d["Context1"], d["Target1"]]))
            texts.append(" ".join([d["Context2"], d["Target2"]]))
    return texts


def safe_mean(xs: Iterable[float]) -> float | None:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def pearson(xs: list[float | None], ys: list[float | None]) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None and math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return None
    xvals, yvals = zip(*pairs)
    mx, my = statistics.mean(xvals), statistics.mean(yvals)
    vx = sum((x - mx) ** 2 for x in xvals)
    vy = sum((y - my) ** 2 for y in yvals)
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
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            out[order[k]] = avg
        i = j
    return out


def spearman(xs: list[float | None], ys: list[float | None]) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None and math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return None
    return pearson(ranks([x for x, _ in pairs]), ranks([y for _, y in pairs]))


def tokenize_lengths(tokenizer, texts: list[str]) -> dict[str, float]:
    total_tokens = 0
    total_nonspecial_tokens = 0
    total_chars = 0
    max_tokens = 0
    over_256 = 0
    # Batch tokenization keeps this fast while preserving exact tokenizer behavior.
    batch_size = 512
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, add_special_tokens=False, padding=False, truncation=False)
        for ids, text in zip(enc["input_ids"], batch):
            n = len(ids)
            ns = sum(1 for t in ids if t not in SPECIAL_IDS)
            total_tokens += n
            total_nonspecial_tokens += ns
            total_chars += len(text)
            if n > max_tokens:
                max_tokens = n
            if n > 256:
                over_256 += 1
    total_words = sum(whitespace_words(t) for t in texts)
    return {
        "strings": len(texts),
        "words": total_words,
        "chars": total_chars,
        "tokens": total_tokens,
        "nonspecial_tokens": total_nonspecial_tokens,
        "tokens_per_word": total_tokens / total_words if total_words else None,
        "nonspecial_tokens_per_word": total_nonspecial_tokens / total_words if total_words else None,
        "tokens_per_char": total_tokens / total_chars if total_chars else None,
        "max_tokens_per_string": max_tokens,
        "frac_strings_over_256": over_256 / len(texts) if texts else None,
    }


def fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    deficit = json.loads(DEFICIT_JSON.read_text(encoding="utf-8"))

    tokenizers = {}
    tok_info = {}
    for label, p in TOKENIZERS.items():
        path = pathlib.Path(p)
        tok = AutoTokenizer.from_pretrained(str(path), use_fast=True, local_files_only=True)
        tokenizers[label] = tok
        tok_info[label] = {"path": str(path), "len_tokenizer": len(tok), "is_fast": tok.is_fast}

    per_uid = []
    text_cache: dict[tuple[str, str], list[str]] = {}
    for column, rows in deficit["columns"].items():
        for row in rows:
            uid = row["uid"]
            score_delta = row.get("delta_legal_step35")
            if score_delta is None:
                continue
            texts = text_cache.setdefault((column, uid), iter_texts(column, uid))
            item: dict[str, Any] = {
                "column": column,
                "uid": uid,
                "score_delta_legal_step35_minus_old_ref": score_delta,
                "n_eval_strings": len(texts),
                "whitespace_words": sum(whitespace_words(t) for t in texts),
            }
            for label, tok in tokenizers.items():
                m = tokenize_lengths(tok, texts)
                for k, v in m.items():
                    item[f"{label}_{k}"] = v

            # Pairwise representation movement metrics.
            old_tpw = item["old_inherited_16k_nonlegal_tokens_per_word"]
            s35_tpw = item["legal_step35_16k_tokens_per_word"]
            byte_tpw = item["legal_bytealpha_16k_tokens_per_word"]
            k40_tpw = item["legal_40k_tokens_per_word"]
            mf50_tpw = item["legal_40k_minfreq50_tokens_per_word"]
            item["minus_old_tpw"] = s35_tpw - old_tpw
            item["bytealpha_minus_step35_tpw"] = byte_tpw - s35_tpw
            item["legal40k_minus_step35_tpw"] = k40_tpw - s35_tpw
            item["legal40k_minfreq50_minus_step35_tpw"] = mf50_tpw - s35_tpw
            # Fraction of research-vs-old length gap closed by 40k; positive means candidate moves toward old token count.
            denom = s35_tpw - old_tpw
            item["legal40k_closes_step35_old_tpw_gap_frac"] = ((s35_tpw - k40_tpw) / denom) if abs(denom) > 1e-9 else None
            per_uid.append(item)

    # Correlations between score delta and representation metrics.
    metrics = [
        "minus_old_tpw",
        "bytealpha_minus_step35_tpw",
        "legal40k_minus_step35_tpw",
        "legal40k_minfreq50_minus_step35_tpw",
        "legal40k_closes_step35_old_tpw_gap_frac",
        "legal_step35_16k_frac_strings_over_256",
        "old_inherited_16k_nonlegal_frac_strings_over_256",
        "legal_40k_frac_strings_over_256",
    ]
    corr = {}
    for column in ["BLiMP", "Supplement", "EWoK", "ALL"]:
        rows = per_uid if column == "ALL" else [r for r in per_uid if r["column"] == column]
        corr[column] = {"n_uid": len(rows)}
        deltas = [r["score_delta_legal_step35_minus_old_ref"] for r in rows]
        for metric in metrics:
            vals = [r.get(metric) for r in rows]
            corr[column][metric] = {
                "pearson_delta_vs_metric": round(pearson(deltas, vals), 6) if pearson(deltas, vals) is not None else None,
                "spearman_delta_vs_metric": round(spearman(deltas, vals), 6) if spearman(deltas, vals) is not None else None,
                "mean_metric": round(safe_mean([v for v in vals if v is not None]), 6) if vals else None,
            }

    focus_uids = {
        "principle_A_reconstruction", "wh_questions_object_gap", "animate_subject_trans",
        "tough_vs_raising_1", "anaphor_gender_agreement", "regular_plural_subject_verb_agreement_1",
        "qa_congruence_easy", "qa_congruence_tricky", "turn_taking", "subject_aux_inversion", "hypernym",
        "physical-dynamics", "material-dynamics", "social-properties", "quantitative-properties",
        "spatial-relations", "material-properties", "social-interactions", "physical-relations",
    }
    focus = [r for r in per_uid if r["uid"] in focus_uids]
    worst = sorted(per_uid, key=lambda r: r["score_delta_legal_step35_minus_old_ref"])[:20]

    result = {
        "status": "TOKENIZER_FRAGMENTATION_DEFICIT_ALIGNMENT",
        "purpose": "CPU-only diagnostic: does tokenization geometry explain the legal research deficit and do legal larger-vocab candidates move affected subtasks toward the old reference?",
        "inputs": {
            "deficit_json": str(DEFICIT_JSON),
            "eval_root": str(EVAL_ROOT),
            "tokenizers": tok_info,
        },
        "correlations": corr,
        "focus_rows": focus,
        "worst_20_rows": worst,
        "per_uid": per_uid,
    }
    out_json = OUT_DIR / "tokenizer_fragmentation_deficit_alignment.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    md = []
    md.append("# research tokenizer-fragmentation / legal-deficit alignment\n")
    md.append("CPU-only analysis of existing tokenizers and existing eval strings. It does not train, evaluate, or tune a tokenizer from evaluation data.\n")
    md.append("## Tokenizers read\n")
    for label, info in tok_info.items():
        md.append(f"- `{label}` len={info['len_tokenizer']}: `{info['path']}`")
    md.append("")
    md.append("## Correlations with score delta (legal_step35 minus old_ref)\n")
    md.append("Negative delta means the legal research endpoint lost. A useful fragmentation explanation would show worse losses where `minus_old_tpw` or overflow increased; a useful representation-candidate signal would show candidate movement specifically on losing subtasks.\n")
    md.append("| column | n | metric | mean metric | Pearson(delta,metric) | Spearman |")
    md.append("|---|---:|---|---:|---:|---:|")
    for column in ["BLiMP", "Supplement", "EWoK", "ALL"]:
        for metric in metrics:
            rec = corr[column][metric]
            md.append(f"| {column} | {corr[column]['n_uid']} | `{metric}` | {fmt(rec['mean_metric'], 6)} | {fmt(rec['pearson_delta_vs_metric'], 4)} | {fmt(rec['spearman_delta_vs_metric'], 4)} |")
    md.append("")
    md.append("## Focus rows\n")
    md.append("| column | uid | Δ score | old tpw | research tpw | research-old | 40k tpw | 40k-research | 40k gap-closed | minfreq50-research |")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(focus, key=lambda r: (r["column"], r["score_delta_legal_step35_minus_old_ref"])):
        md.append("| {column} | {uid} | {delta} | {old} | {s35} | {dold} | {k40} | {dk40} | {closed} | {dmf50} |".format(
            column=r["column"],
            uid=r["uid"],
            delta=fmt(r["score_delta_legal_step35_minus_old_ref"], 2),
            old=fmt(r["old_inherited_16k_nonlegal_tokens_per_word"], 4),
            s35=fmt(r["legal_step35_16k_tokens_per_word"], 4),
            dold=fmt(r["minus_old_tpw"], 4),
            k40=fmt(r["legal_40k_tokens_per_word"], 4),
            dk40=fmt(r["legal40k_minus_step35_tpw"], 4),
            closed=fmt(r["legal40k_closes_step35_old_tpw_gap_frac"], 3),
            dmf50=fmt(r["legal40k_minfreq50_minus_step35_tpw"], 4),
        ))
    md.append("")
    md.append("## Twenty largest research losses\n")
    md.append("| column | uid | Δ score | research-old tpw | 40k-research tpw | 40k gap-closed |")
    md.append("|---|---|---:|---:|---:|---:|")
    for r in worst:
        md.append(f"| {r['column']} | {r['uid']} | {fmt(r['score_delta_legal_step35_minus_old_ref'],2)} | {fmt(r['minus_old_tpw'],4)} | {fmt(r['legal40k_minus_step35_tpw'],4)} | {fmt(r['legal40k_closes_step35_old_tpw_gap_frac'],3)} |")
    md.append("")
    md.append("## Interpretation guard\n")
    md.append("- This file can falsify simple fragmentation stories if score losses do not align with research-vs-old token inflation or overflow.\n")
    md.append("- A legal 40k candidate that reduces tokens per word on losing subtasks is still only a representation hypothesis until A01's full official 40k evaluations finish; low-support vocabulary and the accumulated-trainer path remain separate evidence.\n")
    md.append(f"\nFull JSON: `{out_json}`")
    out_md = (OUT_DIR.parents[4] / 'research/documents/frontier_consolidation/data/tokenizer_fragmentation_deficit_alignment/tokenizer_fragmentation_deficit_alignment.md')
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": result["status"],
        "out_json": str(out_json),
        "out_md": str(out_md),
        "n_uid": len(per_uid),
        "correlations_all": corr["ALL"],
        "focus_count": len(focus),
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
