#!/usr/bin/env python3
"""research refined CLEAN near-duplicate audit stripping annotation scaffolding.

The first research audit measured model-visible lexical overlap.  This refined
pass estimates the natural-language relation dose by removing CHILDES/CHAT style
speaker labels and bracketed annotation/action spans before tokenization.  It
also separates exact-normalized repetition from nonidentical high-overlap pairs.
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
from collections import defaultdict
from typing import Any

ROOT = _public_path('experiments/archive/relation_learning/scripts/refined_clean_nearduplicate_audit.py')
ROOT = _PUBLIC_ROOT

WS = _public_path('experiments/archive/relation_learning')
CLEAN_10M = _public_path('experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/cleanqwen_lengthmatched_dose2p64x_10M.jsonl')
META = _public_path('experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/dose2p64x_rowholdout_metadata.json')
FIRST = _public_path('experiments/archive/relation_learning/data/clean_natural_adjacency_audit')
OUT = _public_path('experiments/archive/relation_learning/data/clean_natural_adjacency_refined')
FIG = _public_path('experiments/archive/relation_learning/figures/clean_natural_adjacency_refined')
NOTE = _public_path('research/notes/relation_learning/clean_natural_nearduplicate_adjacency_refined.md')

STOPWORDS = {
    "a","about","above","after","again","against","all","am","an","and","any","are","aren","as","at","be","because","been","before","being","below","between","both","but","by","can","cannot","could","couldn","did","didn","do","does","doesn","doing","don","down","during","each","few","for","from","further","had","hadn","has","hasn","have","haven","having","he","her","here","hers","herself","him","himself","his","how","i","if","in","into","is","isn","it","its","itself","just","ll","m","ma","me","might","more","most","mustn","my","myself","no","nor","not","now","o","of","off","on","once","only","or","other","our","ours","ourselves","out","over","own","re","s","same","shan","she","should","shouldn","so","some","such","t","than","that","the","their","theirs","them","themselves","then","there","these","they","this","those","through","to","too","under","until","up","ve","very","was","wasn","we","were","weren","what","when","where","which","while","who","whom","why","will","with","won","would","wouldn","you","your","yours","yourself","yourselves","yes","yeah","yep","uh","um","erm","er","huh","oh","okay","ok","mm","mmm","mhm","xxx",
    # frequent corpus scaffolding and speaker codes when they survive stripping
    "mot","chi","fat","bro","sis","exp","inv","int","add","act","com","gra","cod","tim","spa","eng","hun","urs","mar"
}
TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:[.,:/-]\d+)*")
SPEAKER_SPLIT_RE = re.compile(r"\s+(?=\*[A-Z]{2,4}:)")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+|\s+(?==\s*=)|\n+")
CHAT_TIER_LABEL_RE = re.compile(r"(?:^|\s)[*%][A-Za-z]{2,6}:\s*")
BRACKET_RE = re.compile(r"\[[^\]]*\]")
ANGLE_RE = re.compile(r"<[^>]*>")
PAREN_TRACK_RE = re.compile(r"\((?:tracks?|track)[^)]+\)", re.IGNORECASE)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def strip_scaffold(text: str) -> str:
    # Keep utterance content but remove CHAT speaker/tier labels; remove bracketed
    # action/annotation spans because they generated many false semantic hits in
    # the model-visible audit.
    x = text.replace("\uFFFD", " ")
    x = CHAT_TIER_LABEL_RE.sub(" ", x)
    x = BRACKET_RE.sub(" ", x)
    x = ANGLE_RE.sub(" ", x)
    x = PAREN_TRACK_RE.sub(" ", x)
    x = re.sub(r"%[a-z]+:\s*", " ", x, flags=re.IGNORECASE)
    return x


def norm_token(tok: str) -> str:
    t = tok.lower().strip("'’")
    if t.endswith("'s"):
        t = t[:-2]
    return t


def tokens_visible(text: str) -> list[str]:
    return tokens_from_text(text)


def tokens_refined(text: str) -> list[str]:
    return tokens_from_text(strip_scaffold(text))


def tokens_from_text(text: str) -> list[str]:
    out = []
    for m in TOKEN_RE.finditer(text):
        t = norm_token(m.group(0))
        if not t or t in STOPWORDS:
            continue
        if t.isalpha() and len(t) < 3:
            continue
        out.append(t)
    return out


def normalize_sequence(toks: list[str]) -> tuple[str, ...]:
    return tuple(toks)


def split_spans(text: str, mode: str, min_content: int) -> list[dict[str, Any]]:
    tmp = SPEAKER_SPLIT_RE.sub("\n", text)
    parts = [p.strip() for p in SENT_SPLIT_RE.split(tmp) if p.strip()]
    spans: list[dict[str, Any]] = []
    for idx, p in enumerate(parts):
        toks = tokens_refined(p) if mode == "refined" else tokens_visible(p)
        if len(toks) >= min_content:
            spans.append({"span_index": idx, "raw_text": p, "clean_text": strip_scaffold(p), "tokens": toks, "token_set": set(toks), "n_content": len(toks), "seq": normalize_sequence(toks)})
    return spans


def pair_metrics(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    sa, sb = a["token_set"], b["token_set"]
    inter = len(sa & sb)
    union = len(sa | sb)
    min_den = max(1, min(len(sa), len(sb)))
    overlap = inter / min_den
    jac = inter / max(1, union)
    seq_exact = a["seq"] == b["seq"]
    set_exact = sa == sb
    return {"shared_types": inter, "overlap_coef": overlap, "jaccard": jac, "seq_exact": seq_exact, "set_exact": set_exact}


def quantile(xs: list[float], q: float) -> float:
    xs = sorted(x for x in xs if math.isfinite(x))
    if not xs:
        return float("nan")
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


def scan(min_content: int, thresholds: list[float], max_examples: int) -> dict[str, Any]:
    t0 = time.time()
    stats = defaultdict(lambda: {"rows": 0, "words": 0, "spans": 0, "rows_ge2": 0, "candidate_pairs": 0, "overlaps": [], "max_row_overlap": []})
    hits = defaultdict(lambda: {"pairs": 0, "rows": set(), "ovs": [], "jacs": [], "seq_exact": 0, "set_exact": 0, "nonidentical": 0})
    examples: dict[tuple[str, str, float, str], list[dict[str, Any]]] = defaultdict(list)
    total_rows = 0; total_words = 0
    sha = hashlib.sha256()
    with CLEAN_10M.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            sha.update(line.encode("utf-8"))
            row = json.loads(line)
            row_id = int(row.get("example_id", line_no))
            text = str(row.get("text") or "")
            words = int(row.get("words") or len(text.split()))
            src = parse_source(row); fam = source_family(src)
            total_rows += 1; total_words += words
            for mode in ["visible", "refined"]:
                spans = split_spans(text, mode, min_content)
                key = (mode, src, fam)
                s = stats[key]
                s["rows"] += 1; s["words"] += words; s["spans"] += len(spans)
                if len(spans) >= 2:
                    s["rows_ge2"] += 1
                row_max = 0.0
                for i in range(len(spans)):
                    for j in range(i + 1, len(spans)):
                        m = pair_metrics(spans[i], spans[j])
                        ov = float(m["overlap_coef"]); jac = float(m["jaccard"])
                        s["candidate_pairs"] += 1; s["overlaps"].append(ov); row_max = max(row_max, ov)
                        for thr in thresholds:
                            if ov >= thr:
                                h = hits[(mode, src, fam, thr)]
                                h["pairs"] += 1; h["rows"].add(row_id); h["ovs"].append(ov); h["jacs"].append(jac)
                                relation_kind = "exact_sequence" if m["seq_exact"] else ("exact_set" if m["set_exact"] else "nonidentical_high_overlap")
                                h["seq_exact"] += int(bool(m["seq_exact"])); h["set_exact"] += int(bool(m["set_exact"])); h["nonidentical"] += int(relation_kind == "nonidentical_high_overlap")
                                ekey = (mode, src, thr, relation_kind)
                                if len(examples[ekey]) < max_examples:
                                    examples[ekey].append({
                                        "mode": mode, "source": src, "source_family": fam, "threshold": thr, "relation_kind": relation_kind,
                                        "row_example_id": row_id, "row_words": words,
                                        "span_i": spans[i]["span_index"], "span_j": spans[j]["span_index"],
                                        "n_content_i": spans[i]["n_content"], "n_content_j": spans[j]["n_content"],
                                        "overlap_coef": round(ov, 4), "jaccard": round(jac, 4), "shared_types": m["shared_types"],
                                        "span_i_clean": spans[i]["clean_text"][:600], "span_j_clean": spans[j]["clean_text"][:600],
                                        "span_i_raw": spans[i]["raw_text"][:600], "span_j_raw": spans[j]["raw_text"][:600],
                                    })
                if len(spans) >= 2:
                    s["max_row_overlap"].append(row_max)
    cand_rows = []
    for (mode, src, fam), s in sorted(stats.items()):
        ovs = s["overlaps"]; maxovs = s["max_row_overlap"]
        cand_rows.append({
            "mode": mode, "source": src, "source_family": fam, "rows": s["rows"], "words_10M": s["words"],
            "eligible_spans": s["spans"], "rows_with_ge2_spans": s["rows_ge2"], "candidate_pairs": s["candidate_pairs"],
            "candidate_pairs_per_1M_words": s["candidate_pairs"] / max(1, s["words"]) * 1_000_000,
            "overlap_mean": statistics.mean(ovs) if ovs else float("nan"), "overlap_p90": quantile(ovs, 0.90),
            "overlap_p95": quantile(ovs, 0.95), "overlap_p99": quantile(ovs, 0.99),
            "row_max_overlap_p90": quantile(maxovs, 0.90), "row_max_overlap_p95": quantile(maxovs, 0.95), "row_max_overlap_p99": quantile(maxovs, 0.99),
        })
    def pred_all(k): return True
    def pred_nat(k): return k[2] == "natural_babylm_subcorpus"
    def pred_qwen(k): return k[1] == "qwen_pair_packed"
    for mode in ["visible", "refined"]:
        for label, pred in [("ALL", pred_all), ("NATURAL_ONLY", pred_nat), ("DESIGNED_QWEN_PAIR_PACKED", pred_qwen)]:
            keys = [k for k in stats if k[0] == mode and pred(k)]
            words = sum(stats[k]["words"] for k in keys); rows = sum(stats[k]["rows"] for k in keys)
            spans = sum(stats[k]["spans"] for k in keys); ge2 = sum(stats[k]["rows_ge2"] for k in keys); pairs = sum(stats[k]["candidate_pairs"] for k in keys)
            ovs: list[float] = []; maxovs: list[float] = []
            for k in keys:
                ovs.extend(stats[k]["overlaps"]); maxovs.extend(stats[k]["max_row_overlap"])
            cand_rows.append({"mode": mode, "source": label, "source_family": label, "rows": rows, "words_10M": words, "eligible_spans": spans, "rows_with_ge2_spans": ge2, "candidate_pairs": pairs, "candidate_pairs_per_1M_words": pairs / max(1, words) * 1_000_000, "overlap_mean": statistics.mean(ovs) if ovs else float("nan"), "overlap_p90": quantile(ovs, 0.90), "overlap_p95": quantile(ovs, 0.95), "overlap_p99": quantile(ovs, 0.99), "row_max_overlap_p90": quantile(maxovs, 0.90), "row_max_overlap_p95": quantile(maxovs, 0.95), "row_max_overlap_p99": quantile(maxovs, 0.99)})
    hit_rows = []
    for (mode, src, fam, thr), h in sorted(hits.items()):
        words = stats[(mode, src, fam)]["words"]
        hit_rows.append({"mode": mode, "source": src, "source_family": fam, "threshold_overlap_coef": thr, "hit_pairs_unique_10M": h["pairs"], "hit_pair_exposures_100M_repeat10": h["pairs"] * 10, "rows_with_hit_unique_10M": len(h["rows"]), "hit_pairs_per_1M_words": h["pairs"] / max(1, words) * 1_000_000, "seq_exact_pairs": h["seq_exact"], "set_exact_pairs": h["set_exact"], "nonidentical_high_overlap_pairs": h["nonidentical"], "nonidentical_fraction": h["nonidentical"] / max(1, h["pairs"]), "hit_overlap_mean": statistics.mean(h["ovs"]) if h["ovs"] else float("nan"), "hit_jaccard_mean": statistics.mean(h["jacs"]) if h["jacs"] else float("nan")})
    for mode in ["visible", "refined"]:
        for label, pred in [("ALL", pred_all), ("NATURAL_ONLY", pred_nat), ("DESIGNED_QWEN_PAIR_PACKED", pred_qwen)]:
            words = sum(stats[k]["words"] for k in stats if k[0] == mode and pred(k))
            for thr in thresholds:
                pairs = seq = setx = nonid = 0; rowset: set[int] = set(); ovs=[]; jacs=[]
                for (m, src, fam, t), h in hits.items():
                    if m == mode and abs(t-thr) < 1e-12 and pred((m, src, fam)):
                        pairs += h["pairs"]; seq += h["seq_exact"]; setx += h["set_exact"]; nonid += h["nonidentical"]; rowset |= h["rows"]; ovs.extend(h["ovs"]); jacs.extend(h["jacs"])
                hit_rows.append({"mode": mode, "source": label, "source_family": label, "threshold_overlap_coef": thr, "hit_pairs_unique_10M": pairs, "hit_pair_exposures_100M_repeat10": pairs*10, "rows_with_hit_unique_10M": len(rowset), "hit_pairs_per_1M_words": pairs / max(1, words) * 1_000_000, "seq_exact_pairs": seq, "set_exact_pairs": setx, "nonidentical_high_overlap_pairs": nonid, "nonidentical_fraction": nonid / max(1,pairs), "hit_overlap_mean": statistics.mean(ovs) if ovs else float("nan"), "hit_jaccard_mean": statistics.mean(jacs) if jacs else float("nan")})
    ex_rows = [r for rs in examples.values() for r in rs]
    global_json = {"status": "REFINED_CLEAN_NEARDUP_ADJACENCY_DONE", "created_utc": now(), "elapsed_sec": round(time.time()-t0,2), "input_10M": rel(CLEAN_10M), "input_sha256": sha.hexdigest(), "min_content": min_content, "thresholds": thresholds, "definition": "visible mode uses model-visible content tokens; refined mode removes CHAT speaker/tier labels and bracketed/action annotations before counting content-overlap coefficient.", "total_rows": total_rows, "total_words": total_words}
    return {"candidate_rows": cand_rows, "hit_rows": hit_rows, "example_rows": ex_rows, "global": global_json}


def pick(rows: list[dict[str, Any]], mode: str, source: str, thr: float) -> dict[str, Any]:
    for r in rows:
        if r.get("mode") == mode and r.get("source") == source and abs(float(r.get("threshold_overlap_coef"))-thr)<1e-12:
            return r
    return {}


def make_fig(hit_rows: list[dict[str, Any]], thresholds: list[float]) -> list[pathlib.Path]:
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    made=[]
    sources=["childes","open_subtitles","bnc_spoken","gutenberg","simple_wiki","switchboard","NATURAL_ONLY","DESIGNED_QWEN_PAIR_PACKED"]
    for mode in ["visible","refined"]:
        fig, ax = plt.subplots(figsize=(10,4.8))
        xs=list(range(len(sources))); width=0.18
        for ti,thr in enumerate(thresholds):
            vals=[]
            for src in sources:
                r=pick(hit_rows, mode, src, thr); vals.append(float(r.get("hit_pairs_per_1M_words",0) or 0))
            ax.bar([x+(ti-(len(thresholds)-1)/2)*width for x in xs], vals, width, label=f">={thr:.2f}")
        ax.set_xticks(xs); ax.set_xticklabels(sources, rotation=35, ha="right")
        ax.set_ylabel("near-duplicate sentence-span pairs / 1M words")
        ax.set_title(f"CLEAN natural adjacency dose ({mode} tokens)")
        ax.legend(title="overlap coef")
        fig.tight_layout(); p=FIG/f"clean_sentence_neardup_{mode}.png"; fig.savefig(p,dpi=220); fig.savefig(p.with_suffix('.pdf')); plt.close(fig); made += [p,p.with_suffix('.pdf')]
    return made


def fmt(x: Any, nd:int=3) -> str:
    try:
        y=float(x); return "NA" if not math.isfinite(y) else f"{y:.{nd}f}"
    except Exception: return str(x)


def make_note(res: dict[str, Any], figs: list[pathlib.Path]) -> None:
    hit_rows=res["hit_rows"]; meta=read_json(META); pair_dose=(meta.get("pair_summary") or {}).get("pairs")
    lines=[]
    lines.append("# research refined CLEAN natural near-duplicate adjacency audit")
    lines.append("")
    lines.append("The first research audit measured model-visible lexical overlap and intentionally left corpus scaffolding visible. Inspecting examples showed that CHILDES speaker labels and bracketed action annotations inflate some high-overlap counts. This refined pass estimates language-content adjacency by stripping `*MOT:`/`*CHI:`-style labels, `%` tier labels, and bracketed action/annotation spans before tokenization. The model-visible table remains useful as an upper-bound exposure measurement because those tokens are in the stream; the refined table is the better guide for a natural-language composition intervention.")
    lines.append("")
    lines.append("## Refined sentence-span dose")
    lines.append("")
    lines.append("| subset | τ | unique hits / 10M | hits / 1M words | rows with hit | exact-sequence hits | nonidentical high-overlap hits | nonidentical fraction |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for mode in ["visible","refined"]:
        for subset in ["NATURAL_ONLY","DESIGNED_QWEN_PAIR_PACKED","ALL"]:
            for thr in [0.50,0.70,0.80]:
                r=pick(hit_rows, mode, subset, thr)
                label=f"{mode}:{subset}"
                lines.append(f"| {label} | {thr:.2f} | {int(r.get('hit_pairs_unique_10M',0))} | {fmt(r.get('hit_pairs_per_1M_words'))} | {int(r.get('rows_with_hit_unique_10M',0))} | {int(r.get('seq_exact_pairs',0))} | {int(r.get('nonidentical_high_overlap_pairs',0))} | {fmt(r.get('nonidentical_fraction'))} |")
    lines.append("")
    lines.append("## By natural subcorpus after stripping")
    lines.append("")
    lines.append("| source | τ=0.50 hits / 10M | hits / 1M words | rows with hit | exact-sequence | nonidentical | τ=0.70 hits |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for src in ["childes","open_subtitles","bnc_spoken","gutenberg","simple_wiki","switchboard"]:
        r05=pick(hit_rows,"refined",src,0.50); r07=pick(hit_rows,"refined",src,0.70)
        lines.append(f"| {src} | {int(r05.get('hit_pairs_unique_10M',0))} | {fmt(r05.get('hit_pairs_per_1M_words'))} | {int(r05.get('rows_with_hit_unique_10M',0))} | {int(r05.get('seq_exact_pairs',0))} | {int(r05.get('nonidentical_high_overlap_pairs',0))} | {int(r07.get('hit_pairs_unique_10M',0))} |")
    nat05=pick(hit_rows,"refined","NATURAL_ONLY",0.50); nat07=pick(hit_rows,"refined","NATURAL_ONLY",0.70); qwen05=pick(hit_rows,"refined","DESIGNED_QWEN_PAIR_PACKED",0.50)
    lines.append("")
    lines.append("## Scientific reading")
    lines.append("")
    lines.append(f"After stripping annotation scaffolding, natural BabyLM subcorpora still contain `{int(nat05.get('hit_pairs_unique_10M',0))}` high-overlap sentence-span pairs at τ=0.50 and `{int(nat07.get('hit_pairs_unique_10M',0))}` at τ=0.70 per unique 10M stream. These become `{int(nat05.get('hit_pair_exposures_100M_repeat10',0))}` and `{int(nat07.get('hit_pair_exposures_100M_repeat10',0))}` pair exposures in the repeated 100M training stream. The designed causal intervention used `{pair_dose}` selected same-window relations per 10M stream, so the natural language-content dose is material but several-fold smaller than the deliberate compact relation dose. The `qwen_pair_packed` block in CLEAN remains much denser (`{int(qwen05.get('hit_pairs_unique_10M',0))}` refined τ=0.50 hits), so the baseline already contains a large designed restatement/repetition block in addition to natural BabyLM adjacency.")
    lines.append("")
    lines.append("The next natural-composition construction should not claim it is pure BabyLM if it includes `qwen_pair_packed`; it should either (a) split only refined NATURAL_ONLY high-overlap sentence pairs to test whether ordinary row composition contributes a smaller but measurable copy/restatement pressure, or (b) explicitly split the qwen-pair block as a baseline-designed-adjacency control. Because the refined natural dose is far below the 2.64x designed dose, any training readout should be framed as a sensitivity/extent measurement, not as a prerequisite for the already established DeBERTa compact locality result.")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append(f"- Refined threshold table: `{rel(_public_path('experiments/archive/relation_learning/data/clean_natural_adjacency_refined/threshold_hit_summary_refined.csv'))}`")
    lines.append(f"- Candidate distribution: `{rel(_public_path('experiments/archive/relation_learning/data/clean_natural_adjacency_refined/candidate_distribution_refined.csv'))}`")
    lines.append(f"- Examples: `{rel(_public_path('experiments/archive/relation_learning/data/clean_natural_adjacency_refined/near_duplicate_examples_refined.jsonl'))}`")
    lines.append(f"- Result JSON: `{rel(_public_path('experiments/archive/relation_learning/data/clean_natural_adjacency_refined/refined_clean_nearduplicate_adjacency_result.json'))}`")
    lines.append(f"- First model-visible audit: `{rel(_public_path('experiments/archive/relation_learning/data/clean_natural_adjacency_audit/threshold_hit_summary.csv'))}`")
    for p in figs: lines.append(f"- Figure: `{rel(p)}`")
    NOTE.write_text("\n".join(lines)+"\n", encoding="utf-8")


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-content", type=int, default=8)
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.50,0.60,0.70,0.80])
    ap.add_argument("--max-examples-per-bucket", type=int, default=6)
    args=ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    res=scan(args.min_content,args.thresholds,args.max_examples_per_bucket)
    write_csv(_public_path('experiments/archive/relation_learning/data/clean_natural_adjacency_refined/candidate_distribution_refined.csv'), res["candidate_rows"])
    write_csv(_public_path('experiments/archive/relation_learning/data/clean_natural_adjacency_refined/threshold_hit_summary_refined.csv'), res["hit_rows"])
    with (_public_path('experiments/archive/relation_learning/data/clean_natural_adjacency_refined/near_duplicate_examples_refined.jsonl')).open("w", encoding="utf-8") as f:
        for r in res["example_rows"]: f.write(json.dumps(r, ensure_ascii=False)+"\n")
    figs=make_fig(res["hit_rows"], args.thresholds)
    res["global"]["files"]={"threshold": rel(_public_path('experiments/archive/relation_learning/data/clean_natural_adjacency_refined/threshold_hit_summary_refined.csv')), "candidate": rel(_public_path('experiments/archive/relation_learning/data/clean_natural_adjacency_refined/candidate_distribution_refined.csv')), "examples": rel(_public_path('experiments/archive/relation_learning/data/clean_natural_adjacency_refined/near_duplicate_examples_refined.jsonl')), "note": rel(NOTE), "figures":[rel(p) for p in figs]}
    (_public_path('experiments/archive/relation_learning/data/clean_natural_adjacency_refined/refined_clean_nearduplicate_adjacency_result.json')).write_text(json.dumps(res["global"], indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    make_note(res, figs)
    nat05=pick(res["hit_rows"],"refined","NATURAL_ONLY",0.50); nat07=pick(res["hit_rows"],"refined","NATURAL_ONLY",0.70); qwen05=pick(res["hit_rows"],"refined","DESIGNED_QWEN_PAIR_PACKED",0.50)
    print(json.dumps({"status":res["global"]["status"],"note":rel(NOTE),"refined_nat_tau050_hits":nat05.get("hit_pairs_unique_10M"),"refined_nat_tau070_hits":nat07.get("hit_pairs_unique_10M"),"refined_qwen_tau050_hits":qwen05.get("hit_pairs_unique_10M"),"figures":[rel(p) for p in figs],"elapsed_sec":res["global"]["elapsed_sec"]}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
