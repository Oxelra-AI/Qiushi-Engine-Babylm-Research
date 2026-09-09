#!/usr/bin/env python3
"""research: measure row-internal near-duplicate adjacency in the CLEAN stream.

Scientific purpose
------------------
The current relation-typed composition result is based on designed same-window
relations.  This script measures the natural dose already present in the
CLEAN 10M stream used by the MAX-dose BabyLM mechanism arms.  The 100M training
stream is a 10x repeat of this 10M stream, so unique row-internal adjacency is
measured on the 10M file and exposure counts are reported after multiplying by
10.

The audit is intentionally lexical and transparent: split each row into
sentence/utterance-like spans, remove high-frequency function words, require a
minimum number of content tokens, and count row-internal span pairs whose
content overlap coefficient |A∩B|/min(|A|,|B|) exceeds a sweep of thresholds.
A fixed-window sensitivity view is also written so the conclusion does not rest
entirely on sentence splitting.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import hashlib
import json
import math
import pathlib
import re
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

ROOT = _public_path('experiments/archive/relation_learning/scripts/clean_nearduplicate_adjacency_audit.py')
ROOT = _PUBLIC_ROOT

WS = _public_path('experiments/archive/relation_learning')
CLEAN_10M = _public_path('experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/cleanqwen_lengthmatched_dose2p64x_10M.jsonl')
CLEAN_100M_MAT = _public_path('experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/cleanqwen_lengthmatched_dose2p64x_100M.materialization.json')
META = _public_path('experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/dose2p64x_rowholdout_metadata.json')
OUT = _public_path('experiments/archive/relation_learning/data/clean_natural_adjacency_audit')
FIG = _public_path('experiments/archive/relation_learning/figures/clean_natural_adjacency_audit')
NOTE = _public_path('research/notes/relation_learning/clean_natural_nearduplicate_adjacency.md')

STOPWORDS = {
    # compact English function-word list; keep names/numbers/content words.
    "a","about","above","after","again","against","all","am","an","and","any","are","aren","as","at",
    "be","because","been","before","being","below","between","both","but","by","can","cannot","could","couldn",
    "did","didn","do","does","doesn","doing","don","down","during","each","few","for","from","further",
    "had","hadn","has","hasn","have","haven","having","he","her","here","hers","herself","him","himself",
    "his","how","i","if","in","into","is","isn","it","its","itself","just","ll","m","ma","me","might",
    "more","most","mustn","my","myself","no","nor","not","now","o","of","off","on","once","only","or",
    "other","our","ours","ourselves","out","over","own","re","s","same","shan","she","should","shouldn",
    "so","some","such","t","than","that","the","their","theirs","them","themselves","then","there","these",
    "they","this","those","through","to","too","under","until","up","ve","very","was","wasn","we","were",
    "weren","what","when","where","which","while","who","whom","why","will","with","won","would","wouldn",
    "you","your","yours","yourself","yourselves","yes","yeah","yep","uh","um","erm","er","huh","oh","okay",
    "ok","mm","mmm","mhm","xxx"
}
TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:[.,:/-]\d+)*")
SPEAKER_RE = re.compile(r"\s+(?=\*[A-Z]{2,4}:)")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+|\s+(?==\s*=)|\n+")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_id(*parts: Any) -> str:
    h = hashlib.sha1("\u241f".join(str(x) for x in parts).encode("utf-8")).hexdigest()
    return h[:20]


def parse_source(row: dict[str, Any]) -> str:
    s = str(row.get("source") or "unknown")
    if "::" in s:
        s = s.split("::", 1)[1]
    return s or "unknown"


def source_family(src: str) -> str:
    if src == "qwen_pair_packed":
        return "designed_qwen_pair_packed"
    if src in {"childes", "open_subtitles", "bnc_spoken", "gutenberg", "simple_wiki", "switchboard"}:
        return "natural_babylm_subcorpus"
    return "other_or_unknown"


def norm_token(tok: str) -> str:
    t = tok.lower().strip("'’")
    if t.endswith("'s"):
        t = t[:-2]
    return t


def content_tokens(text: str) -> list[str]:
    out: list[str] = []
    for m in TOKEN_RE.finditer(text):
        t = norm_token(m.group(0))
        if not t:
            continue
        if t in STOPWORDS:
            continue
        if t.isalpha() and len(t) < 3:
            continue
        out.append(t)
    return out


def split_sentence_spans(text: str, min_content: int) -> list[dict[str, Any]]:
    # Speaker turns are sentence-like units in CHILDES; headings often mark
    # unrelated article boundaries in SimpleWiki/Gutenberg-like rows.
    tmp = SPEAKER_RE.sub("\n", text)
    parts = [p.strip() for p in SENT_SPLIT_RE.split(tmp) if p.strip()]
    # Further split long packed segments on colon speaker/dialogue boundaries,
    # but preserve enough text for content overlap.
    refined: list[str] = []
    for p in parts:
        if len(p.split()) > 55:
            # split on strong punctuation/heading markers inside very long chunks
            qs = [q.strip() for q in re.split(r"(?<=[.!?])\s+|\s+=\s+=\s+", p) if q.strip()]
            refined.extend(qs if len(qs) > 1 else [p])
        else:
            refined.append(p)
    spans: list[dict[str, Any]] = []
    for idx, p in enumerate(refined):
        toks = content_tokens(p)
        if len(toks) >= min_content:
            spans.append({"span_index": idx, "text": p, "tokens": toks, "token_set": set(toks), "n_content": len(toks)})
    return spans


def fixed_window_spans(text: str, win: int, stride: int) -> list[dict[str, Any]]:
    toks = content_tokens(text)
    spans: list[dict[str, Any]] = []
    if len(toks) < 2 * win:
        return spans
    for i, start in enumerate(range(0, len(toks) - win + 1, stride)):
        part = toks[start:start + win]
        spans.append({"span_index": i, "start_content_token": start, "tokens": part, "token_set": set(part), "n_content": len(part), "text": " ".join(part)})
    return spans


def pair_metrics(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
    sa, sb = a["token_set"], b["token_set"]
    inter = len(sa & sb)
    union = len(sa | sb)
    min_den = max(1, min(len(sa), len(sb)))
    max_den = max(1, max(len(sa), len(sb)))
    return {
        "shared_types": float(inter),
        "overlap_coef": inter / min_den,
        "jaccard": inter / max(1, union),
        "recall_a_in_b": inter / max(1, len(sa)),
        "recall_b_in_a": inter / max(1, len(sb)),
        "containment_max": inter / max_den,
    }


def quantile(xs: list[float], q: float) -> float:
    xs = sorted(x for x in xs if math.isfinite(x))
    if not xs:
        return float("nan")
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos)); hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


@dataclass
class ScanResult:
    candidate_rows: list[dict[str, Any]]
    threshold_rows: list[dict[str, Any]]
    example_rows: list[dict[str, Any]]
    span_rows: list[dict[str, Any]]
    global_stats: dict[str, Any]


def scan_stream(path: pathlib.Path, min_content: int, thresholds: list[float], max_examples_per_source: int, fixed_window: int, fixed_stride: int) -> ScanResult:
    t0 = time.time()
    candidate_stats: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(lambda: {
        "rows": 0, "rows_with_ge2_spans": 0, "span_count": 0, "candidate_pairs": 0,
        "overlaps": [], "jaccards": [], "max_overlap_rows": [], "words": 0,
    })
    threshold_stats: dict[tuple[str, str, str, float], dict[str, Any]] = defaultdict(lambda: {
        "hit_pairs": 0, "rows_with_hit": set(), "hit_overlap_values": [], "hit_jaccard_values": []
    })
    examples: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    span_summary: dict[tuple[str, str], dict[str, Any]] = defaultdict(lambda: {"rows": 0, "spans": 0, "words": 0})
    total_rows = 0; total_words = 0
    row_hit_ids_by_thr: dict[float, set[int]] = {thr: set() for thr in thresholds}
    sha = hashlib.sha256()

    modes = ["sentence", "fixed_window"]
    with path.open(encoding="utf-8") as f:
        for line in f:
            b = line.encode("utf-8")
            sha.update(b)
            row = json.loads(line)
            row_idx = int(row.get("example_id", total_rows))
            text = str(row.get("text") or "")
            words = int(row.get("words") or len(text.split()))
            src = parse_source(row)
            fam = source_family(src)
            total_rows += 1; total_words += words
            for mode in modes:
                if mode == "sentence":
                    spans = split_sentence_spans(text, min_content=min_content)
                else:
                    spans = fixed_window_spans(text, win=fixed_window, stride=fixed_stride)
                ss = span_summary[(mode, src)]
                ss["rows"] += 1; ss["spans"] += len(spans); ss["words"] += words
                key = (mode, src, fam)
                cs = candidate_stats[key]
                cs["rows"] += 1; cs["words"] += words; cs["span_count"] += len(spans)
                if len(spans) >= 2:
                    cs["rows_with_ge2_spans"] += 1
                row_max = 0.0
                for i in range(len(spans)):
                    for j in range(i + 1, len(spans)):
                        # fixed-window spans overlap by construction if their content-token
                        # starts are too close; only compare non-overlapping windows.
                        if mode == "fixed_window":
                            si = int(spans[i].get("start_content_token", i * fixed_stride))
                            sj = int(spans[j].get("start_content_token", j * fixed_stride))
                            if abs(sj - si) < fixed_window:
                                continue
                        m = pair_metrics(spans[i], spans[j])
                        ov = float(m["overlap_coef"]); jc = float(m["jaccard"])
                        cs["candidate_pairs"] += 1
                        cs["overlaps"].append(ov); cs["jaccards"].append(jc)
                        row_max = max(row_max, ov)
                        for thr in thresholds:
                            if ov >= thr:
                                ts = threshold_stats[(mode, src, fam, thr)]
                                ts["hit_pairs"] += 1
                                ts["rows_with_hit"].add(row_idx)
                                ts["hit_overlap_values"].append(ov)
                                ts["hit_jaccard_values"].append(jc)
                                row_hit_ids_by_thr[thr].add(row_idx)
                                ekey = (mode, src, thr)
                                if len(examples[ekey]) < max_examples_per_source:
                                    examples[ekey].append({
                                        "mode": mode,
                                        "threshold": thr,
                                        "row_example_id": row_idx,
                                        "source": src,
                                        "source_family": fam,
                                        "words": words,
                                        "span_i": spans[i].get("span_index"),
                                        "span_j": spans[j].get("span_index"),
                                        "n_content_i": spans[i].get("n_content"),
                                        "n_content_j": spans[j].get("n_content"),
                                        "overlap_coef": round(ov, 4),
                                        "jaccard": round(jc, 4),
                                        "shared_types": int(m["shared_types"]),
                                        "span_i_text": spans[i].get("text", "")[:500],
                                        "span_j_text": spans[j].get("text", "")[:500],
                                    })
                if len(spans) >= 2:
                    cs["max_overlap_rows"].append(row_max)

    candidate_rows: list[dict[str, Any]] = []
    # Per-source/mode rows.
    for (mode, src, fam), cs in sorted(candidate_stats.items()):
        ovs = cs["overlaps"]; jcs = cs["jaccards"]; maxovs = cs["max_overlap_rows"]
        candidate_rows.append({
            "mode": mode,
            "source": src,
            "source_family": fam,
            "rows": cs["rows"],
            "words_10M_stream": cs["words"],
            "rows_with_ge2_spans": cs["rows_with_ge2_spans"],
            "span_count": cs["span_count"],
            "candidate_pairs": cs["candidate_pairs"],
            "candidate_pairs_per_1M_words": cs["candidate_pairs"] / max(1, cs["words"]) * 1_000_000,
            "overlap_mean": statistics.mean(ovs) if ovs else float("nan"),
            "overlap_p50": quantile(ovs, 0.50),
            "overlap_p90": quantile(ovs, 0.90),
            "overlap_p95": quantile(ovs, 0.95),
            "overlap_p99": quantile(ovs, 0.99),
            "jaccard_mean": statistics.mean(jcs) if jcs else float("nan"),
            "row_max_overlap_p50": quantile(maxovs, 0.50),
            "row_max_overlap_p90": quantile(maxovs, 0.90),
            "row_max_overlap_p95": quantile(maxovs, 0.95),
            "row_max_overlap_p99": quantile(maxovs, 0.99),
        })

    # Aggregate natural/all/designed summaries as additional rows.
    def aggregate(label: str, mode: str, pred) -> dict[str, Any]:
        keys = [k for k in candidate_stats if k[0] == mode and pred(k)]
        rows = sum(candidate_stats[k]["rows"] for k in keys)
        words = sum(candidate_stats[k]["words"] for k in keys)
        ge2 = sum(candidate_stats[k]["rows_with_ge2_spans"] for k in keys)
        spans = sum(candidate_stats[k]["span_count"] for k in keys)
        pairs = sum(candidate_stats[k]["candidate_pairs"] for k in keys)
        ovs: list[float] = [] ; jcs: list[float] = [] ; maxovs: list[float] = []
        for k in keys:
            ovs.extend(candidate_stats[k]["overlaps"]); jcs.extend(candidate_stats[k]["jaccards"]); maxovs.extend(candidate_stats[k]["max_overlap_rows"])
        return {
            "mode": mode, "source": label, "source_family": label, "rows": rows, "words_10M_stream": words,
            "rows_with_ge2_spans": ge2, "span_count": spans, "candidate_pairs": pairs,
            "candidate_pairs_per_1M_words": pairs / max(1, words) * 1_000_000,
            "overlap_mean": statistics.mean(ovs) if ovs else float("nan"),
            "overlap_p50": quantile(ovs, 0.50), "overlap_p90": quantile(ovs, 0.90),
            "overlap_p95": quantile(ovs, 0.95), "overlap_p99": quantile(ovs, 0.99),
            "jaccard_mean": statistics.mean(jcs) if jcs else float("nan"),
            "row_max_overlap_p50": quantile(maxovs, 0.50), "row_max_overlap_p90": quantile(maxovs, 0.90),
            "row_max_overlap_p95": quantile(maxovs, 0.95), "row_max_overlap_p99": quantile(maxovs, 0.99),
        }
    for mode in modes:
        candidate_rows.append(aggregate("ALL", mode, lambda k: True))
        candidate_rows.append(aggregate("NATURAL_ONLY", mode, lambda k: k[2] == "natural_babylm_subcorpus"))
        candidate_rows.append(aggregate("DESIGNED_QWEN_PAIR_PACKED", mode, lambda k: k[1] == "qwen_pair_packed"))

    threshold_rows: list[dict[str, Any]] = []
    for (mode, src, fam, thr), ts in sorted(threshold_stats.items()):
        words = candidate_stats[(mode, src, fam)]["words"]
        threshold_rows.append({
            "mode": mode,
            "source": src,
            "source_family": fam,
            "threshold_overlap_coef": thr,
            "hit_pairs_unique_10M": ts["hit_pairs"],
            "hit_pair_exposures_100M_repeat10": ts["hit_pairs"] * 10,
            "rows_with_hit_unique_10M": len(ts["rows_with_hit"]),
            "hit_pairs_per_1M_words_unique": ts["hit_pairs"] / max(1, words) * 1_000_000,
            "hit_pair_exposures_per_1M_training_words_100M": (ts["hit_pairs"] * 10) / 100_000_000 * 1_000_000,
            "hit_overlap_mean": statistics.mean(ts["hit_overlap_values"]) if ts["hit_overlap_values"] else float("nan"),
            "hit_jaccard_mean": statistics.mean(ts["hit_jaccard_values"]) if ts["hit_jaccard_values"] else float("nan"),
        })

    # Add aggregate threshold rows.
    for mode in modes:
        for agg_label, pred in [
            ("ALL", lambda k: True),
            ("NATURAL_ONLY", lambda k: k[2] == "natural_babylm_subcorpus"),
            ("DESIGNED_QWEN_PAIR_PACKED", lambda k: k[1] == "qwen_pair_packed"),
        ]:
            agg_words = sum(candidate_stats[k]["words"] for k in candidate_stats if k[0] == mode and pred(k))
            for thr in thresholds:
                hit_pairs = 0; rowset: set[int] = set(); ovs = []; jcs = []
                for (m, src, fam, t), ts in threshold_stats.items():
                    if m == mode and abs(t - thr) < 1e-12 and pred((m, src, fam)):
                        hit_pairs += ts["hit_pairs"]; rowset |= ts["rows_with_hit"]; ovs.extend(ts["hit_overlap_values"]); jcs.extend(ts["hit_jaccard_values"])
                threshold_rows.append({
                    "mode": mode, "source": agg_label, "source_family": agg_label,
                    "threshold_overlap_coef": thr,
                    "hit_pairs_unique_10M": hit_pairs,
                    "hit_pair_exposures_100M_repeat10": hit_pairs * 10,
                    "rows_with_hit_unique_10M": len(rowset),
                    "hit_pairs_per_1M_words_unique": hit_pairs / max(1, agg_words) * 1_000_000,
                    "hit_pair_exposures_per_1M_training_words_100M": (hit_pairs * 10) / 100_000_000 * 1_000_000,
                    "hit_overlap_mean": statistics.mean(ovs) if ovs else float("nan"),
                    "hit_jaccard_mean": statistics.mean(jcs) if jcs else float("nan"),
                })

    ex_rows = [r for rows in examples.values() for r in rows]
    span_rows = [
        {"mode": m, "source": s, "rows": v["rows"], "words": v["words"], "eligible_spans": v["spans"], "eligible_spans_per_1M_words": v["spans"] / max(1, v["words"]) * 1_000_000}
        for (m, s), v in sorted(span_summary.items())
    ]
    global_stats = {
        "status": "CLEAN_NEARDUP_ADJACENCY_AUDIT_DONE",
        "created_utc": now(),
        "elapsed_sec": round(time.time() - t0, 2),
        "input_10M": rel(path),
        "input_10M_sha256": sha.hexdigest(),
        "clean_100M_materialization": rel(CLEAN_100M_MAT),
        "metadata": rel(META),
        "total_rows_10M": total_rows,
        "total_words_10M": total_words,
        "training_100M_is_repeat10": True,
        "min_content_tokens_per_span": min_content,
        "thresholds_overlap_coef": thresholds,
        "fixed_window_content_tokens": fixed_window,
        "fixed_window_stride_content_tokens": fixed_stride,
        "designed_selected_pair_dose_10M": (read_json(META).get("pair_summary") or {}).get("pairs"),
        "designed_selected_pair_word_budget_10M": (read_json(META).get("dose") or {}).get("selected_pair_words"),
        "definition": "near-duplicate hit: row-internal pair of eligible spans with |content-token type intersection| / min(type-counts) >= threshold; content tokens remove an explicit stopword list and very short alphabetic tokens.",
    }
    return ScanResult(candidate_rows, threshold_rows, ex_rows, span_rows, global_stats)


def make_figures(threshold_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]], thresholds: list[float]) -> list[pathlib.Path]:
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    made: list[pathlib.Path] = []
    # Sentence-mode natural by subcorpus threshold bars.
    src_order = ["childes", "open_subtitles", "bnc_spoken", "gutenberg", "simple_wiki", "switchboard", "qwen_pair_packed", "NATURAL_ONLY", "ALL"]
    for mode in ["sentence", "fixed_window"]:
        rows = [r for r in threshold_rows if r["mode"] == mode and r["source"] in src_order]
        fig, ax = plt.subplots(figsize=(10, 4.8))
        width = 0.18
        xs = list(range(len(src_order)))
        for ti, thr in enumerate(thresholds):
            vals = []
            for src in src_order:
                rr = [r for r in rows if r["source"] == src and abs(float(r["threshold_overlap_coef"]) - thr) < 1e-12]
                vals.append(float(rr[0]["hit_pairs_per_1M_words_unique"]) if rr else 0.0)
            ax.bar([x + (ti - (len(thresholds)-1)/2)*width for x in xs], vals, width=width, label=f">={thr:.2f}")
        ax.set_xticks(xs); ax.set_xticklabels(src_order, rotation=35, ha="right")
        ax.set_ylabel("row-internal near-duplicate span pairs / 1M words (unique 10M)")
        ax.set_title(f"CLEAN stream near-duplicate adjacency dose ({mode} spans)")
        ax.legend(title="overlap coef")
        fig.tight_layout()
        p = FIG / f"clean_neardup_dose_by_source_{mode}.png"
        fig.savefig(p, dpi=220)
        fig.savefig(p.with_suffix(".pdf"))
        plt.close(fig)
        made.extend([p, p.with_suffix(".pdf")])
    return made


def fmt(x: Any, nd: int = 3) -> str:
    try:
        y = float(x)
        if not math.isfinite(y):
            return "NA"
        return f"{y:.{nd}f}"
    except Exception:
        return str(x)


def pick(rows: list[dict[str, Any]], mode: str, source: str, thr: float) -> dict[str, Any]:
    for r in rows:
        if r.get("mode") == mode and r.get("source") == source and abs(float(r.get("threshold_overlap_coef")) - thr) < 1e-12:
            return r
    return {}


def make_note(result: ScanResult, figures: list[pathlib.Path], thresholds: list[float]) -> None:
    _public_path('research/notes/relation_learning').mkdir(parents=True, exist_ok=True)
    meta = read_json(META)
    pair_dose = (meta.get("pair_summary") or {}).get("pairs")
    pair_words = (meta.get("dose") or {}).get("selected_pair_words")
    natural_thr05 = pick(result.threshold_rows, "sentence", "NATURAL_ONLY", 0.50)
    natural_thr07 = pick(result.threshold_rows, "sentence", "NATURAL_ONLY", 0.70)
    qwen_thr05 = pick(result.threshold_rows, "sentence", "DESIGNED_QWEN_PAIR_PACKED", 0.50)
    all_thr05 = pick(result.threshold_rows, "sentence", "ALL", 0.50)
    fw_nat_thr05 = pick(result.threshold_rows, "fixed_window", "NATURAL_ONLY", 0.50)
    lines: list[str] = []
    lines.append("# research CLEAN row-internal near-duplicate adjacency audit")
    lines.append("")
    lines.append("This Execute note measures the natural dose of the relation-typed composition phenomenon in the CLEAN stream before any new natural-composition training is designed. The 100M training stream is a 10x repeat of the 10M stream, so unique row-internal relation opportunities are counted on the 10M file and exposure counts multiply by ten.")
    lines.append("")
    lines.append("## Definition")
    lines.append("")
    lines.append(f"Rows are split into sentence/utterance-like spans. A span is eligible when it has at least `{result.global_stats['min_content_tokens_per_span']}` content tokens after stopword removal. A near-duplicate hit at threshold `τ` is a pair of eligible spans in the same row with content-overlap coefficient `|A∩B|/min(|A|,|B|) >= τ`. I also wrote a fixed-window sensitivity scan with `{result.global_stats['fixed_window_content_tokens']}` content-token windows and stride `{result.global_stats['fixed_window_stride_content_tokens']}`. This is a lexical dose measurement, not a trained-model result.")
    lines.append("")
    lines.append("## Main measurement")
    lines.append("")
    lines.append(f"Input stream: `{result.global_stats['input_10M']}`; SHA256 `{result.global_stats['input_10M_sha256'][:16]}...`. It contains `{result.global_stats['total_rows_10M']}` rows and `{result.global_stats['total_words_10M']:,}` words. The designed MAX intervention dose used for the causal mechanism arms is `{pair_dose}` selected source/companion relations occupying `{pair_words:,}` words per 10M stream before 10x repetition.")
    lines.append("")
    lines.append("Sentence-span hits:")
    lines.append("")
    lines.append("| subset | τ=0.50 unique hits / 10M | τ=0.50 hits / 1M words | τ=0.70 unique hits / 10M | rows with τ=0.50 hit | 100M exposure count at τ=0.50 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for label, r05, r07 in [
        ("natural BabyLM subcorpora only", natural_thr05, natural_thr07),
        ("designed qwen_pair_packed block", qwen_thr05, pick(result.threshold_rows, "sentence", "DESIGNED_QWEN_PAIR_PACKED", 0.70)),
        ("all CLEAN rows", all_thr05, pick(result.threshold_rows, "sentence", "ALL", 0.70)),
    ]:
        lines.append(f"| {label} | {int(r05.get('hit_pairs_unique_10M',0))} | {fmt(r05.get('hit_pairs_per_1M_words_unique'))} | {int(r07.get('hit_pairs_unique_10M',0))} | {int(r05.get('rows_with_hit_unique_10M',0))} | {int(r05.get('hit_pair_exposures_100M_repeat10',0))} |")
    lines.append("")
    lines.append(f"Fixed-window sensitivity for natural-only τ=0.50 finds `{int(fw_nat_thr05.get('hit_pairs_unique_10M',0))}` unique hits, `{fmt(fw_nat_thr05.get('hit_pairs_per_1M_words_unique'))}` per 1M words. The two views differ because sentence splitting catches local restatements/utterance repetition, while fixed windows catch repeated lexical blocks independent of punctuation.")
    lines.append("")
    lines.append("## By subcorpus")
    lines.append("")
    lines.append("| source | τ=0.50 sentence hits / 10M | hits / 1M source words | rows with hit | mean hit overlap |")
    lines.append("|---|---:|---:|---:|---:|")
    for src in ["childes", "open_subtitles", "bnc_spoken", "gutenberg", "simple_wiki", "switchboard", "qwen_pair_packed"]:
        r = pick(result.threshold_rows, "sentence", src, 0.50)
        lines.append(f"| {src} | {int(r.get('hit_pairs_unique_10M',0))} | {fmt(r.get('hit_pairs_per_1M_words_unique'))} | {int(r.get('rows_with_hit_unique_10M',0))} | {fmt(r.get('hit_overlap_mean'))} |")
    lines.append("")
    lines.append("## Scientific reading")
    lines.append("")
    natural_hits = int(natural_thr05.get('hit_pairs_unique_10M', 0) or 0)
    qwen_hits = int(qwen_thr05.get('hit_pairs_unique_10M', 0) or 0)
    lines.append(f"At τ=0.50, the natural BabyLM subcorpora contribute `{natural_hits}` row-internal lexical near-duplicate span pairs per 10M stream, whereas the protected `qwen_pair_packed` block contributes `{qwen_hits}` under the same sentence-span definition. Compare this with the designed causal arms' `{pair_dose}` selected source/companion relations per 10M stream. Thus the practical claim should distinguish (i) the deliberately installed compact relation dose, (ii) the existing designed Qwen-pair block already in CLEAN, and (iii) the smaller or larger natural BabyLM row-internal dose measured here. The paper should not imply that the natural stream necessarily has the same relation dose as the intervention without this table.")
    lines.append("")
    lines.append("If the natural-only τ=0.50 hit count is large enough for a training intervention, the next construction is a CLEAN_NATURAL_SPLIT arm that preserves row texts and word budget as much as possible while moving one member of each naturally co-occurring high-overlap span pair to a separate row. Its readouts should be pre-stated against CLEAN: lower exact copy gain and improved true-source use on source-absent changed/substituted targets would indicate that baseline adjacency itself pays the exact-recurrence cost. If the measured natural dose is too low relative to the 1.82x/2.64x designed doses, the stronger extent test is an ICLM-style composed-window manipulation with near-duplicate filtering on versus off.")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append(f"- Candidate distribution table: `{rel(_public_path('experiments/archive/relation_learning/data/clean_natural_adjacency_audit/candidate_span_pair_distribution.csv'))}`")
    lines.append(f"- Threshold hit table: `{rel(_public_path('experiments/archive/relation_learning/data/clean_natural_adjacency_audit/threshold_hit_summary.csv'))}`")
    lines.append(f"- Examples: `{rel(_public_path('experiments/archive/relation_learning/data/clean_natural_adjacency_audit/near_duplicate_examples.jsonl'))}`")
    lines.append(f"- Global JSON: `{rel(_public_path('experiments/archive/relation_learning/data/clean_natural_adjacency_audit/clean_nearduplicate_adjacency_result.json'))}`")
    for p in figures:
        lines.append(f"- Figure: `{rel(p)}`")
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-content", type=int, default=8)
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.50, 0.60, 0.70, 0.80])
    ap.add_argument("--max-examples-per-source", type=int, default=8)
    ap.add_argument("--fixed-window", type=int, default=12)
    ap.add_argument("--fixed-stride", type=int, default=12)
    args = ap.parse_args()
    for p in [CLEAN_10M, CLEAN_100M_MAT, META]:
        if not p.exists():
            raise FileNotFoundError(p)
    mat = read_json(CLEAN_100M_MAT)
    if int(mat.get("repeats", 0)) != 10:
        raise RuntimeError(f"CLEAN 100M materialization is not repeat10: {mat}")
    OUT.mkdir(parents=True, exist_ok=True)
    result = scan_stream(CLEAN_10M, args.min_content, args.thresholds, args.max_examples_per_source, args.fixed_window, args.fixed_stride)
    write_csv(_public_path('experiments/archive/relation_learning/data/clean_natural_adjacency_audit/candidate_span_pair_distribution.csv'), result.candidate_rows)
    write_csv(_public_path('experiments/archive/relation_learning/data/clean_natural_adjacency_audit/threshold_hit_summary.csv'), result.threshold_rows)
    write_csv(_public_path('experiments/archive/relation_learning/data/clean_natural_adjacency_audit/eligible_span_summary.csv'), result.span_rows)
    with (_public_path('experiments/archive/relation_learning/data/clean_natural_adjacency_audit/near_duplicate_examples.jsonl')).open("w", encoding="utf-8") as f:
        for r in result.example_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    figures = make_figures(result.threshold_rows, result.candidate_rows, args.thresholds)
    result.global_stats["files"] = {
        "candidate_distribution": rel(_public_path('experiments/archive/relation_learning/data/clean_natural_adjacency_audit/candidate_span_pair_distribution.csv')),
        "threshold_summary": rel(_public_path('experiments/archive/relation_learning/data/clean_natural_adjacency_audit/threshold_hit_summary.csv')),
        "eligible_span_summary": rel(_public_path('experiments/archive/relation_learning/data/clean_natural_adjacency_audit/eligible_span_summary.csv')),
        "examples": rel(_public_path('experiments/archive/relation_learning/data/clean_natural_adjacency_audit/near_duplicate_examples.jsonl')),
        "note": rel(NOTE),
        "figures": [rel(p) for p in figures],
    }
    (_public_path('experiments/archive/relation_learning/data/clean_natural_adjacency_audit/clean_nearduplicate_adjacency_result.json')).write_text(json.dumps(result.global_stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    make_note(result, figures, args.thresholds)
    print(json.dumps({
        "status": result.global_stats["status"],
        "note": rel(NOTE),
        "threshold_summary": rel(_public_path('experiments/archive/relation_learning/data/clean_natural_adjacency_audit/threshold_hit_summary.csv')),
        "examples": rel(_public_path('experiments/archive/relation_learning/data/clean_natural_adjacency_audit/near_duplicate_examples.jsonl')),
        "figures": [rel(p) for p in figures],
        "natural_sentence_tau050_hits": pick(result.threshold_rows, "sentence", "NATURAL_ONLY", 0.50).get("hit_pairs_unique_10M"),
        "qwen_sentence_tau050_hits": pick(result.threshold_rows, "sentence", "DESIGNED_QWEN_PAIR_PACKED", 0.50).get("hit_pairs_unique_10M"),
        "elapsed_sec": result.global_stats["elapsed_sec"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
