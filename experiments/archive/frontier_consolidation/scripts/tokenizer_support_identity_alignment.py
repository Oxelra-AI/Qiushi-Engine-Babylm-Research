#!/usr/bin/env python3
"""research: tokenizer support and token-identity alignment with legal endpoint losses.

This CPU-only analysis strengthens the route choice before another H100 run.  It uses
only existing tokenizers and already-scored endpoint deltas.  It counts how often each
subword token appears in the allowed 10M compact-view-reinvest pool, then asks whether
evaluation subtasks that lost under the legal research tokenizer are especially exposed to
low-support tokens or to token-identity changes relative to the old non-submittable
reference tokenizer.  It also characterizes the legal 40k and support-floored candidate
interfaces already prepared, without training or evaluating a model.

No tokenizer is trained here; official evaluation text is used only for post-training
interpretation of existing tokenizers and score vectors.
"""
from __future__ import annotations

import collections
import csv
import json
import math
import pathlib
import statistics
import time
from typing import Any, Iterable

from transformers import AutoTokenizer

STUDY = pathlib.Path("experiments/archive/frontier_consolidation")
POOL = STUDY / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"
DEFICIT_JSON = STUDY / "data/legal_deficit_subtask_decomposition/legal_deficit_subtask_decomposition.json"
EVAL_ROOT = pathlib.Path("experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval")
OUT_DIR = STUDY / "data/tokenizer_support_identity_alignment"

TOKENIZERS = {
    "old_inherited_16k_nonlegal": "experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/hf_model/chck_100M",
    "legal_step35_16k": "experiments/archive/frontier_consolidation/data/compliant_tokenizer",
    "legal_bytealpha_16k": "experiments/archive/frontier_consolidation/data/compliant_tokenizer_bytealphabet",
    "legal_24k": "experiments/archive/representation_and_objectives/data/legal_representation_route_map/tokenizers/legal_byte_bpe_24k",
    "legal_32k": "experiments/archive/representation_and_objectives/data/legal_representation_route_map/tokenizers/legal_byte_bpe_32k",
    "legal_40k": "experiments/archive/representation_and_objectives/data/legal_representation_route_map/tokenizers/legal_byte_bpe_40k",
    "legal_40k_minfreq25": "experiments/archive/representation_and_objectives/data/tokenizer_support_spectrum/tokenizers/legal_byte_bpe_40k_minfreq25",
    "legal_40k_minfreq50": "experiments/archive/representation_and_objectives/data/tokenizer_support_spectrum/tokenizers/legal_byte_bpe_40k_minfreq50",
}
SPECIAL_IDS = {0, 1, 2, 3, 4}
LOW_THRESHOLDS = [5, 10, 20, 50, 100, 200, 500]


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_jsonl(path: pathlib.Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def iter_pool_texts() -> Iterable[str]:
    for d in read_jsonl(POOL):
        yield d["text"]


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
            texts.append(" ".join([d["Context1"], d["Target1"]]))
            texts.append(" ".join([d["Context2"], d["Target2"]]))
    return texts


def non_special_ids(ids: list[int]) -> list[int]:
    return [int(i) for i in ids if int(i) not in SPECIAL_IDS]


def count_pool_tokens(label: str, tokenizer, batch_size: int = 128) -> dict[str, Any]:
    counts = [0] * len(tokenizer)
    rows = 0
    words = 0
    token_total = 0
    unk_id = tokenizer.unk_token_id
    unk = 0
    batch: list[str] = []
    for text in iter_pool_texts():
        batch.append(text)
        words += len(text.split())
        if len(batch) >= batch_size:
            rows, token_total, unk = consume_batch(tokenizer, counts, batch, rows, token_total, unk_id, unk)
            batch = []
    if batch:
        rows, token_total, unk = consume_batch(tokenizer, counts, batch, rows, token_total, unk_id, unk)
    used = [c for i, c in enumerate(counts) if i not in SPECIAL_IDS and c > 0]
    return {
        "label": label,
        "vocab_size": len(tokenizer),
        "rows": rows,
        "words": words,
        "tokens": token_total,
        "tokens_per_word": token_total / words if words else None,
        "unk_count": unk,
        "counts": counts,
        "used_nonspecial_vocab": len(used),
        "unused_nonspecial_vocab": max(0, len(tokenizer) - len(SPECIAL_IDS) - len(used)),
        "median_used_count": statistics.median(used) if used else None,
        "p10_used_count": percentile(used, 0.10) if used else None,
        "mass_below": {str(t): sum(c for i, c in enumerate(counts) if i not in SPECIAL_IDS and 0 < c < t) for t in LOW_THRESHOLDS},
        "vocab_below": {str(t): sum(1 for i, c in enumerate(counts) if i not in SPECIAL_IDS and 0 < c < t) for t in LOW_THRESHOLDS},
    }


def consume_batch(tokenizer, counts: list[int], batch: list[str], rows: int, token_total: int, unk_id: int | None, unk: int):
    enc = tokenizer(batch, add_special_tokens=False, padding=False, truncation=False)
    for ids in enc["input_ids"]:
        ns = non_special_ids(ids)
        rows += 1
        token_total += len(ns)
        for tid in ns:
            if tid < len(counts):
                counts[tid] += 1
            if unk_id is not None and tid == unk_id:
                unk += 1
    return rows, token_total, unk


def percentile(xs: list[int], q: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    pos = q * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(xs[lo])
    return float(xs[lo] * (hi - pos) + xs[hi] * (pos - lo))


def token_strings(tokenizer, ids: list[int]) -> list[str]:
    return tokenizer.convert_ids_to_tokens(ids)


def eval_support_metrics(tokenizer, counts: list[int], texts: list[str]) -> dict[str, Any]:
    all_counts: list[int] = []
    total_tokens = 0
    batch_size = 512
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, add_special_tokens=False, padding=False, truncation=False)
        for ids in enc["input_ids"]:
            ns = non_special_ids(ids)
            total_tokens += len(ns)
            for tid in ns:
                all_counts.append(counts[tid] if tid < len(counts) else 0)
    if not all_counts:
        return {"eval_tokens": 0}
    out: dict[str, Any] = {
        "eval_tokens": total_tokens,
        "mean_pool_count": sum(all_counts) / len(all_counts),
        "median_pool_count": percentile(all_counts, 0.50),
        "p10_pool_count": percentile(all_counts, 0.10),
        "mean_log10_pool_count_plus1": sum(math.log10(c + 1) for c in all_counts) / len(all_counts),
    }
    for t in LOW_THRESHOLDS:
        out[f"frac_pool_count_lt_{t}"] = sum(c < t for c in all_counts) / len(all_counts)
    return out


def eval_token_counter(tokenizer, texts: list[str]) -> collections.Counter[str]:
    c: collections.Counter[str] = collections.Counter()
    batch_size = 512
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i:i + batch_size], add_special_tokens=False, padding=False, truncation=False)
        for ids in enc["input_ids"]:
            c.update(token_strings(tokenizer, non_special_ids(ids)))
    return c


def multiset_overlap(a: collections.Counter[str], b: collections.Counter[str]) -> tuple[int, int, int]:
    keys = set(a) | set(b)
    common = sum(min(a[k], b[k]) for k in keys)
    maxsum = sum(max(a[k], b[k]) for k in keys)
    return common, sum(a.values()), sum(b.values()) if maxsum else (0, 0, 0)


def identity_metrics(a: collections.Counter[str], b: collections.Counter[str]) -> dict[str, float | None]:
    keys = set(a) | set(b)
    common = sum(min(a[k], b[k]) for k in keys)
    union = sum(max(a[k], b[k]) for k in keys)
    atot = sum(a.values())
    btot = sum(b.values())
    return {
        "token_multiset_jaccard": common / union if union else None,
        "a_common_frac": common / atot if atot else None,
        "b_common_frac": common / btot if btot else None,
        "b_not_in_a_frac": 1.0 - (common / btot) if btot else None,
        "a_not_in_b_frac": 1.0 - (common / atot) if atot else None,
    }


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


def safe_round(x: Any, nd: int = 6) -> Any:
    if isinstance(x, float):
        return round(x, nd)
    return x


def fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def main() -> None:
    start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    deficit = json.loads(DEFICIT_JSON.read_text(encoding="utf-8"))

    tokenizers = {}
    pool_stats = {}
    for label, path in TOKENIZERS.items():
        tok = AutoTokenizer.from_pretrained(path, use_fast=True, local_files_only=True)
        tokenizers[label] = tok
        pool_stats[label] = count_pool_tokens(label, tok)

    per_uid: list[dict[str, Any]] = []
    for column, rows in deficit["columns"].items():
        for row in rows:
            uid = row["uid"]
            delta = row.get("delta_legal_step35")
            if delta is None:
                continue
            texts = iter_eval_texts(column, uid)
            item: dict[str, Any] = {
                "column": column,
                "uid": uid,
                "score_delta_legal_step35_minus_old_ref": delta,
                "eval_strings": len(texts),
                "eval_words": sum(len(t.split()) for t in texts),
            }
            counters = {}
            for label, tok in tokenizers.items():
                support = eval_support_metrics(tok, pool_stats[label]["counts"], texts)
                for k, v in support.items():
                    item[f"{label}_{k}"] = v
                counters[label] = eval_token_counter(tok, texts)
            for label in ["legal_step35_16k", "legal_bytealpha_16k", "legal_24k", "legal_32k", "legal_40k", "legal_40k_minfreq25", "legal_40k_minfreq50"]:
                im = identity_metrics(counters["old_inherited_16k_nonlegal"], counters[label])
                prefix = f"{label}_vs_old"
                for k, v in im.items():
                    item[f"{prefix}_{k}"] = v
            im_40_vs_16 = identity_metrics(counters["legal_step35_16k"], counters["legal_40k"])
            for k, v in im_40_vs_16.items():
                item[f"legal_40k_vs_step35_{k}"] = v
            per_uid.append(item)

    metrics = []
    for label in TOKENIZERS:
        for key in ["mean_log10_pool_count_plus1", "frac_pool_count_lt_50", "frac_pool_count_lt_100", "frac_pool_count_lt_200"]:
            metrics.append(f"{label}_{key}")
    for label in ["legal_step35_16k", "legal_bytealpha_16k", "legal_24k", "legal_32k", "legal_40k", "legal_40k_minfreq25", "legal_40k_minfreq50"]:
        metrics.append(f"{label}_vs_old_token_multiset_jaccard")
        metrics.append(f"{label}_vs_old_b_not_in_a_frac")
    metrics.append("legal_40k_vs_step35_b_not_in_a_frac")

    correlations: dict[str, Any] = {}
    for column in ["BLiMP", "Supplement", "EWoK", "ALL"]:
        rows = per_uid if column == "ALL" else [r for r in per_uid if r["column"] == column]
        deltas = [r["score_delta_legal_step35_minus_old_ref"] for r in rows]
        correlations[column] = {"n_uid": len(rows)}
        for metric in metrics:
            vals = [r.get(metric) for r in rows]
            p = pearson(deltas, vals)
            s = spearman(deltas, vals)
            correlations[column][metric] = {
                "pearson_delta_vs_metric": safe_round(p),
                "spearman_delta_vs_metric": safe_round(s),
                "mean_metric": safe_round(sum(v for v in vals if isinstance(v, (int, float))) / len([v for v in vals if isinstance(v, (int, float))]) if any(isinstance(v, (int, float)) for v in vals) else None),
            }

    focus_uids = {
        "principle_A_reconstruction", "wh_questions_object_gap", "animate_subject_trans",
        "regular_plural_subject_verb_agreement_1", "tough_vs_raising_1", "anaphor_gender_agreement",
        "matrix_question_npi_licensor_present", "qa_congruence_easy", "qa_congruence_tricky",
        "physical-dynamics", "material-dynamics", "social-properties", "quantitative-properties",
        "spatial-relations", "material-properties", "social-interactions",
    }
    focus = [r for r in per_uid if r["uid"] in focus_uids]
    worst = sorted(per_uid, key=lambda r: r["score_delta_legal_step35_minus_old_ref"])[:20]

    # Strip raw count arrays from JSON summary but save compact pool table.
    pool_compact = {}
    for label, st in pool_stats.items():
        pool_compact[label] = {k: v for k, v in st.items() if k != "counts"}
        for k in ["mass_below", "vocab_below"]:
            pool_compact[label][k] = st[k]

    result = {
        "status": "TOKENIZER_SUPPORT_IDENTITY_ALIGNMENT",
        "created_utc": now_utc(),
        "elapsed_sec": round(time.time() - start, 3),
        "purpose": "CPU-only alignment of legal research score losses with tokenizer support and token-identity geometry across existing legal representation candidates.",
        "inputs": {
            "pool": str(POOL),
            "deficit_json": str(DEFICIT_JSON),
            "eval_root": str(EVAL_ROOT),
            "tokenizers": TOKENIZERS,
        },
        "pool_support_compact": pool_compact,
        "correlations": correlations,
        "focus_rows": focus,
        "worst_20_rows": worst,
        "per_uid": per_uid,
    }
    out_json = OUT_DIR / "tokenizer_support_identity_alignment.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = OUT_DIR / "per_uid_support_identity.csv"
    key_cols = [
        "column", "uid", "score_delta_legal_step35_minus_old_ref",
        "legal_step35_16k_mean_log10_pool_count_plus1", "legal_step35_16k_frac_pool_count_lt_50", "legal_step35_16k_frac_pool_count_lt_100",
        "legal_40k_mean_log10_pool_count_plus1", "legal_40k_frac_pool_count_lt_50", "legal_40k_frac_pool_count_lt_100",
        "legal_40k_minfreq50_mean_log10_pool_count_plus1", "legal_40k_minfreq50_frac_pool_count_lt_50", "legal_40k_minfreq50_frac_pool_count_lt_100",
        "legal_step35_16k_vs_old_token_multiset_jaccard", "legal_step35_16k_vs_old_b_not_in_a_frac",
        "legal_40k_vs_old_token_multiset_jaccard", "legal_40k_vs_old_b_not_in_a_frac", "legal_40k_vs_step35_b_not_in_a_frac",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=key_cols)
        w.writeheader()
        for r in per_uid:
            w.writerow({k: r.get(k) for k in key_cols})

    md = []
    md.append("# research tokenizer support and identity alignment\n")
    md.append("CPU-only analysis over existing tokenizers and the allowed 10M pool. Evaluation text is used only to interpret already-trained tokenizers and already-scored endpoint deltas.\n")
    md.append("## Pool support summary\n")
    md.append("| tokenizer | vocab | tok/word | used non-special | median count | p10 count | frac mass <50 | frac vocab <50 |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for label, st in pool_compact.items():
        tokens = st["tokens"]
        mass_lt50 = st["mass_below"]["50"] / tokens if tokens else 0
        vocab_lt50 = st["vocab_below"]["50"] / max(1, st["used_nonspecial_vocab"])
        md.append(f"| {label} | {st['vocab_size']} | {st['tokens_per_word']:.4f} | {st['used_nonspecial_vocab']} | {fmt(st['median_used_count'],1)} | {fmt(st['p10_used_count'],1)} | {mass_lt50:.4f} | {vocab_lt50:.4f} |")
    md.append("")
    md.append("## Selected correlations with legal research score delta\n")
    md.append("Negative score delta means legal research lost vs the old non-submittable reference. If low support explains losses, `frac_pool_count_lt_*` should correlate negatively and log-count should correlate positively with score delta. If token identity shift explains losses, legal-vs-old new-token fraction should correlate negatively.\n")
    selected_metrics = [
        "legal_step35_16k_mean_log10_pool_count_plus1", "legal_step35_16k_frac_pool_count_lt_50", "legal_step35_16k_frac_pool_count_lt_100",
        "legal_step35_16k_vs_old_b_not_in_a_frac", "legal_step35_16k_vs_old_token_multiset_jaccard",
        "legal_40k_frac_pool_count_lt_50", "legal_40k_vs_old_b_not_in_a_frac", "legal_40k_vs_step35_b_not_in_a_frac",
        "legal_40k_minfreq50_frac_pool_count_lt_50",
    ]
    md.append("| column | n | metric | mean | Pearson(delta,metric) | Spearman |")
    md.append("|---|---:|---|---:|---:|---:|")
    for column in ["BLiMP", "Supplement", "EWoK", "ALL"]:
        for metric in selected_metrics:
            rec = correlations[column][metric]
            md.append(f"| {column} | {correlations[column]['n_uid']} | `{metric}` | {fmt(rec['mean_metric'],4)} | {fmt(rec['pearson_delta_vs_metric'],4)} | {fmt(rec['spearman_delta_vs_metric'],4)} |")
    md.append("")
    md.append("## Focus rows\n")
    md.append("| column | uid | Δ score | research frac<50 | research new-vs-old frac | research-old Jaccard | 40k frac<50 | 40k new-vs-old frac | 40k-vs-research new frac | minfreq50 frac<50 |")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(focus, key=lambda x: (x["column"], x["score_delta_legal_step35_minus_old_ref"])):
        md.append("| {column} | {uid} | {delta} | {s35lt50} | {s35new} | {s35jac} | {k40lt50} | {k40new} | {k40s35new} | {mf50lt50} |".format(
            column=r["column"], uid=r["uid"], delta=fmt(r["score_delta_legal_step35_minus_old_ref"],2),
            s35lt50=fmt(r.get("legal_step35_16k_frac_pool_count_lt_50"),4),
            s35new=fmt(r.get("legal_step35_16k_vs_old_b_not_in_a_frac"),4),
            s35jac=fmt(r.get("legal_step35_16k_vs_old_token_multiset_jaccard"),4),
            k40lt50=fmt(r.get("legal_40k_frac_pool_count_lt_50"),4),
            k40new=fmt(r.get("legal_40k_vs_old_b_not_in_a_frac"),4),
            k40s35new=fmt(r.get("legal_40k_vs_step35_b_not_in_a_frac"),4),
            mf50lt50=fmt(r.get("legal_40k_minfreq50_frac_pool_count_lt_50"),4),
        ))
    md.append("")
    md.append("## Largest research losses\n")
    md.append("| column | uid | Δ score | research frac<50 | research new-vs-old frac | 40k frac<50 | 40k-vs-research new frac |")
    md.append("|---|---|---:|---:|---:|---:|---:|")
    for r in worst:
        md.append(f"| {r['column']} | {r['uid']} | {fmt(r['score_delta_legal_step35_minus_old_ref'],2)} | {fmt(r.get('legal_step35_16k_frac_pool_count_lt_50'),4)} | {fmt(r.get('legal_step35_16k_vs_old_b_not_in_a_frac'),4)} | {fmt(r.get('legal_40k_frac_pool_count_lt_50'),4)} | {fmt(r.get('legal_40k_vs_step35_b_not_in_a_frac'),4)} |")
    md.append("")
    md.append("## Scientific reading\n")
    md.append("- Use this file to distinguish simple low-support or token-identity stories from broader legal representation/optimization effects.\n")
    md.append("- A 40k run can recover syntax/QA only if its larger merge inventory helps more than its added low-support embeddings hurt; this table shows which subtasks would be exposed to each side of that trade.\n")
    md.append("- Do not turn these correlations into a new route without the pending mature clean-vs-reinvest comparison and A01's full official 40k results.\n")
    md.append(f"\nJSON: `{out_json}`\nCSV: `{csv_path}`")
    out_md = (OUT_DIR.parents[4] / 'research/documents/frontier_consolidation/data/tokenizer_support_identity_alignment/tokenizer_support_identity_alignment.md')
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": result["status"],
        "out_json": str(out_json),
        "out_md": str(out_md),
        "csv": str(csv_path),
        "elapsed_sec": result["elapsed_sec"],
        "n_uid": len(per_uid),
        "selected_all_correlations": {m: correlations["ALL"][m] for m in selected_metrics},
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
