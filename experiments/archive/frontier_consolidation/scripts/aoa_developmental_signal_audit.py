#!/usr/bin/env python3
"""research: AoA developmental-signal audit before any new curriculum training.

Purpose
-------
The research proposal was scientifically/rule risky because it would use the exact
504 official AoA evaluation words as a pretraining mask list.  This audit treats the
AoA file only as an *evaluation-analysis* object and asks what the existing legal
corpora and already-trained trajectories actually show.

It quantifies:
  1. frequency of official AoA target words by human age group and by training source;
  2. their exposure timing under the baseline clean-Qwen order and the prior safe
     first-pass developmental order;
  3. group surprisal trajectories for already-trained clean-Qwen and first-pass
     source-order models;
  4. whether corpus-only developmental ordering was already tested and what it cost
     on the broad BabyLM surface.

No output of this script is a training corpus, token mask list, or model input.  It is
an analysis record to decide whether a new legal, transferable developmental schedule
is scientifically justified.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import json
import math
import pathlib
import re
import statistics
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Tuple

ROOT = _public_path('experiments/archive/frontier_consolidation')
OUT_DIR = _public_path('experiments/archive/frontier_consolidation/data/aoa_developmental_audit')
OUT_JSON = _public_path('experiments/archive/frontier_consolidation/data/aoa_developmental_audit/aoa_developmental_signal_audit.json')
OUT_NOTE = _public_path('research/notes/frontier_consolidation/aoa_developmental_signal_audit.md')

CDI_HUMAN = pathlib.Path("experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/aoa/cdi_human.csv")
CLEAN_QWEN_10M = pathlib.Path("experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl")
OFFICIAL_POOL_10M = pathlib.Path("experiments/archive/compact_experience/data/mixture/official_pool.jsonl")
DEVCURR_10M = pathlib.Path("experiments/archive/compact_experience/data/aoa_developmental_order/qwen_aligned_devcurr_firstpass_10M.jsonl")
DEVCURR_META = pathlib.Path("experiments/archive/compact_experience/data/aoa_developmental_order/devcurr_materialization_metadata.json")
SOURCE_BALANCED_META = pathlib.Path("experiments/archive/compact_experience/data/source_balanced_devcurr/source_balanced_devcurr_metadata.json")

SURPRISAL_PATHS = {
    "clean_qwen_seed43022": pathlib.Path("experiments/archive/compact_experience/data/full_eval/aoa_outputs/qwen_clean_aligned/hf_model_local_ckpts/main/zero_shot/mlm/AoA_word/surprisal.json"),
    "clean_qwen_seed43122": pathlib.Path("experiments/archive/compact_experience/data/full_eval/aoa_outputs/qwen_clean_aligned_seed43122/hf_model_local_ckpts/main/zero_shot/mlm/AoA_word/surprisal.json"),
    "devcurr_firstpass_seed43022": pathlib.Path("experiments/archive/compact_experience/data/devcurr_eval/aoa_outputs/qwen_devcurr_firstpass_seed43022/hf_model_local_ckpts/main/zero_shot/mlm/AoA_word/surprisal.json"),
    "devcurr_firstpass_seed43122": pathlib.Path("experiments/archive/compact_experience/data/devcurr_eval/aoa_outputs/qwen_devcurr_firstpass_seed43122/hf_model_local_ckpts/main/zero_shot/mlm/AoA_word/surprisal.json"),
}
AOA_LOCAL_SUMMARIES = {
    "clean_qwen_seed43022": pathlib.Path("experiments/archive/compact_experience/data/full_eval/aoa_outputs/qwen_clean_aligned/aoa_local_ckpts.json"),
    "clean_qwen_seed43122": pathlib.Path("experiments/archive/compact_experience/data/full_eval/aoa_outputs/qwen_clean_aligned_seed43122/aoa_local_ckpts.json"),
    "devcurr_firstpass_seed43022": pathlib.Path("experiments/archive/compact_experience/data/devcurr_eval/aoa_outputs/qwen_devcurr_firstpass_seed43022/aoa_local_ckpts.json"),
    "devcurr_firstpass_seed43122": pathlib.Path("experiments/archive/compact_experience/data/devcurr_eval/aoa_outputs/qwen_devcurr_firstpass_seed43122/aoa_local_ckpts.json"),
}
FULL_EVAL_CLEAN = pathlib.Path("experiments/archive/compact_experience/data/full_eval/full_eval_summary.json")
DEVCURR_STDOUT = pathlib.Path("experiments/archive/compact_experience/data/devcurr_eval_seed43022_stdout.log")

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+")
STEPS = [f"chck_{i}M" for i in range(1, 10)] + [f"chck_{i}M" for i in range(10, 101, 10)]
STEP_WORDS = {f"chck_{i}M": i * 1_000_000 for i in range(1, 10)} | {f"chck_{i}M": i * 1_000_000 for i in range(10, 101, 10)}


def norm_word(w: str) -> str:
    return w.strip().lower()


def iter_words(text: str) -> Iterable[str]:
    for m in WORD_RE.finditer(text):
        yield m.group(0).lower()


def age_bin(age: float) -> str:
    if age <= 20:
        return "early_le20"
    if age <= 25:
        return "mid_21_25"
    if age <= 30:
        return "late_26_30"
    return "very_late_gt30"


def stats(vals: List[float]) -> Dict[str, Any]:
    if not vals:
        return {"n": 0, "min": None, "mean": None, "median": None, "max": None}
    xs = sorted(vals)
    return {"n": len(xs), "min": xs[0], "mean": sum(xs) / len(xs), "median": statistics.median(xs), "max": xs[-1]}


def load_cdi() -> Dict[str, Dict[str, Any]]:
    words: Dict[str, Dict[str, Any]] = {}
    with CDI_HUMAN.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        month_cols = [str(i) for i in range(16, 31)]
        for row in reader:
            w = norm_word(row["word"])
            probs = {int(m): float(row[m]) for m in month_cols if row.get(m, "") != ""}
            # Match the usual 50% acquisition convention; if not crossed by 30, put
            # it just beyond the table for coarse grouping.
            acquired = [m for m, p in probs.items() if p >= 0.5]
            aoa = float(min(acquired) if acquired else 31)
            words[w] = {"word": w, "human_aoa_month": aoa, "bin": age_bin(aoa), "p16": probs.get(16), "p30": probs.get(30)}
    return words


def pearson(xs: List[float], ys: List[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def ranks(vals: List[float]) -> List[float]:
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    r = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j + 2) / 2.0  # 1-indexed ranks
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(xs: List[float], ys: List[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    return pearson(ranks(xs), ranks(ys))


def count_cdi_in_corpus(path: pathlib.Path, cdi: Dict[str, Dict[str, Any]], bin_words: int = 1_000_000) -> Dict[str, Any]:
    cdi_set = set(cdi)
    counts_by_word: Counter[str] = Counter()
    counts_by_source: Dict[str, Counter[str]] = defaultdict(Counter)
    source_words: Counter[str] = Counter()
    bin_records: List[Dict[str, Any]] = []
    cur_bin_idx = 1
    cur_words = 0
    cur_source_words: Counter[str] = Counter()
    cur_cdi_counts: Counter[str] = Counter()
    total_words = 0
    row_count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = obj["text"]
            words = int(obj["words"])
            source = str(obj.get("source", "unknown"))
            row_count += 1
            total_words += words
            source_words[source] += words
            cur_words += words
            cur_source_words[source] += words
            row_counter = Counter(w for w in iter_words(text) if w in cdi_set)
            if row_counter:
                counts_by_word.update(row_counter)
                counts_by_source[source].update(row_counter)
                cur_cdi_counts.update(row_counter)
            while cur_words >= bin_words:
                bin_records.append({
                    "bin_index": cur_bin_idx,
                    "row_atomic_words": cur_words,
                    "source_words": dict(sorted(cur_source_words.items())),
                    "cdi_counts_by_age_bin": aggregate_counts_by_age(cur_cdi_counts, cdi),
                    "top_cdi_words": cur_cdi_counts.most_common(12),
                })
                cur_bin_idx += 1
                cur_words = 0
                cur_source_words = Counter()
                cur_cdi_counts = Counter()
    by_age = aggregate_counts_by_age(counts_by_word, cdi)
    by_source_age = {src: aggregate_counts_by_age(cnt, cdi) for src, cnt in sorted(counts_by_source.items())}
    rows = []
    for w, meta in cdi.items():
        rows.append((meta["human_aoa_month"], math.log1p(counts_by_word[w]), counts_by_word[w], w))
    corr_aoa_logfreq = pearson([r[0] for r in rows], [r[1] for r in rows])
    spear_aoa_logfreq = spearman([r[0] for r in rows], [r[1] for r in rows])
    return {
        "path": str(path),
        "row_count": row_count,
        "total_words": total_words,
        "source_words": dict(sorted(source_words.items())),
        "cdi_total_token_hits": int(sum(counts_by_word.values())),
        "cdi_word_types_with_hits": int(sum(1 for w in cdi if counts_by_word[w] > 0)),
        "cdi_counts_by_age_bin": by_age,
        "cdi_counts_by_source_and_age_bin": by_source_age,
        "cdi_top_words": counts_by_word.most_common(30),
        "cdi_bottom_hit_words": sorted([(w, counts_by_word[w], cdi[w]["human_aoa_month"]) for w in cdi], key=lambda x: (x[1], x[2], x[0]))[:30],
        "correlation_human_aoa_vs_log1p_corpus_count": corr_aoa_logfreq,
        "spearman_human_aoa_vs_log1p_corpus_count": spear_aoa_logfreq,
        "first_1M_bins": bin_records[:10],
    }


def aggregate_counts_by_age(counts: Counter[str], cdi: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for b in ["early_le20", "mid_21_25", "late_26_30", "very_late_gt30"]:
        ws = [w for w, m in cdi.items() if m["bin"] == b]
        vals = [counts[w] for w in ws]
        out[b] = {
            "word_types": len(ws),
            "token_hits": int(sum(vals)),
            "types_with_hits": int(sum(1 for v in vals if v > 0)),
            "mean_hits_per_type": (sum(vals) / len(vals)) if vals else None,
            "median_hits_per_type": statistics.median(vals) if vals else None,
        }
    return out


def load_surprisal(path: pathlib.Path, cdi: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    d = json.loads(path.read_text(encoding="utf-8"))
    rows = d["results"]
    by_step_word: Dict[str, Dict[str, List[float]]] = {s: defaultdict(list) for s in STEPS}
    for r in rows:
        step = r["step"]
        w = norm_word(r["target_word"])
        if step in by_step_word and w in cdi:
            by_step_word[step][w].append(float(r["surprisal"]))
    mean_by_step_word: Dict[str, Dict[str, float]] = {}
    for step, wd in by_step_word.items():
        mean_by_step_word[step] = {w: sum(v) / len(v) for w, v in wd.items() if v}
    represented = sorted(set().union(*[set(x) for x in mean_by_step_word.values()]))
    group_step: Dict[str, Dict[str, Any]] = {}
    for step in STEPS:
        group_step[step] = {}
        for b in ["early_le20", "mid_21_25", "late_26_30", "very_late_gt30"]:
            vals = [mean_by_step_word[step][w] for w in represented if cdi[w]["bin"] == b and w in mean_by_step_word[step]]
            group_step[step][b] = stats(vals)
    per_word_summary = []
    for w in represented:
        s1 = mean_by_step_word.get("chck_1M", {}).get(w)
        s10 = mean_by_step_word.get("chck_10M", {}).get(w)
        s100 = mean_by_step_word.get("chck_100M", {}).get(w)
        if s1 is None or s100 is None:
            continue
        total_drop = s1 - s100
        threshold = s1 - 0.5 * total_drop
        half_cross_words = None
        if total_drop > 0:
            for step in STEPS:
                sv = mean_by_step_word[step].get(w)
                if sv is not None and sv <= threshold:
                    half_cross_words = STEP_WORDS[step]
                    break
        per_word_summary.append({
            "word": w,
            "human_aoa_month": cdi[w]["human_aoa_month"],
            "bin": cdi[w]["bin"],
            "s1": s1,
            "s10": s10,
            "s100": s100,
            "drop_1M_to_100M": total_drop,
            "half_drop_cross_words": half_cross_words,
        })
    aoa = [x["human_aoa_month"] for x in per_word_summary]
    final_s = [x["s100"] for x in per_word_summary]
    drop = [x["drop_1M_to_100M"] for x in per_word_summary]
    half_pairs = [(x["human_aoa_month"], math.log10(x["half_drop_cross_words"])) for x in per_word_summary if x["half_drop_cross_words"]]
    return {
        "path": str(path),
        "rows": len(rows),
        "represented_cdi_words": len(represented),
        "group_step_mean_surprisal": group_step,
        "correlation_human_aoa_vs_final_surprisal": pearson(aoa, final_s),
        "spearman_human_aoa_vs_final_surprisal": spearman(aoa, final_s),
        "correlation_human_aoa_vs_total_drop": pearson(aoa, drop),
        "spearman_human_aoa_vs_total_drop": spearman(aoa, drop),
        "correlation_human_aoa_vs_log_half_drop_time": pearson([a for a, _ in half_pairs], [h for _, h in half_pairs]) if len(half_pairs) >= 3 else None,
        "spearman_human_aoa_vs_log_half_drop_time": spearman([a for a, _ in half_pairs], [h for _, h in half_pairs]) if len(half_pairs) >= 3 else None,
        "n_words_with_positive_drop_and_half_cross": len(half_pairs),
        "per_word_extremes": {
            "largest_improvements": sorted(per_word_summary, key=lambda x: x["drop_1M_to_100M"], reverse=True)[:20],
            "worse_by_100M": sorted(per_word_summary, key=lambda x: x["drop_1M_to_100M"])[:20],
            "earliest_half_cross": sorted([x for x in per_word_summary if x["half_drop_cross_words"]], key=lambda x: (x["half_drop_cross_words"], x["human_aoa_month"], x["word"]))[:30],
            "latest_half_cross": sorted([x for x in per_word_summary if x["half_drop_cross_words"]], key=lambda x: (-x["half_drop_cross_words"], x["human_aoa_month"], x["word"]))[:30],
        },
    }


def read_aoa_summaries() -> Dict[str, Any]:
    out = {}
    for name, path in AOA_LOCAL_SUMMARIES.items():
        if path.exists():
            d = json.loads(path.read_text(encoding="utf-8"))
            out[name] = {
                "aoa": d.get("aoa"),
                "num_rows": d.get("num_rows"),
                "num_steps": d.get("num_steps"),
                "step_counts_unique": sorted(set((d.get("step_counts") or {}).values())),
                "step_mean_surprisal": d.get("step_mean_surprisal"),
            }
    return out


def read_broad_scores() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if FULL_EVAL_CLEAN.exists():
        d = json.loads(FULL_EVAL_CLEAN.read_text(encoding="utf-8"))
        # Keep compact; exact file is already canonical.
        out["clean_qwen_seed43022"] = {
            "source": str(FULL_EVAL_CLEAN),
            "overall": d.get("Overall") or d.get("overall") or d.get("scores", {}).get("Overall"),
            "raw": d,
        }
    if DEVCURR_STDOUT.exists():
        final = None
        with DEVCURR_STDOUT.open("r", encoding="utf-8") as f:
            for line in f:
                if '"event": "target_finalized"' in line:
                    final = json.loads(line)
        if final:
            out["devcurr_firstpass_seed43022"] = {
                "source": str(DEVCURR_STDOUT),
                "overall": final.get("Overall"),
                "scores": final.get("scores"),
            }
    return out


def make_note(payload: Dict[str, Any]) -> str:
    clean = payload["corpus_frequency"]["clean_qwen_10M"]
    official = payload["corpus_frequency"]["official_pool_10M"]
    dev = payload["corpus_frequency"]["devcurr_firstpass_order_10M"]
    sur = payload["surprisal_trajectory"]
    broad = payload["broad_scores"]
    def fmt(x: Any, nd: int = 4) -> str:
        if x is None:
            return "NA"
        if isinstance(x, float):
            return f"{x:.{nd}f}"
        return str(x)
    lines: List[str] = []
    lines.append("# research — AoA developmental signal audit before training")
    lines.append("")
    lines.append("## Why this audit was needed")
    lines.append("The research training proposal is not acceptable as written: it would use the exact official AoA words as a pretraining mask list, and it described ten epoch-specific 10M files rather than one documented ≤10M corpus repeated within the epoch limit. No training was launched. It treats the AoA file only as an evaluation-analysis object and asks whether a transferable, legal developmental schedule is supported by existing evidence.")
    lines.append("")
    lines.append("## Corpus frequency pattern of official AoA words")
    for name, obj in [("clean-Qwen 10M", clean), ("official pool 10M", official), ("prior source-block first-pass order", dev)]:
        lines.append(f"### {name}")
        lines.append(f"- total words: {obj['total_words']:,}; CDI token hits: {obj['cdi_total_token_hits']:,}; CDI types with hits: {obj['cdi_word_types_with_hits']}")
        lines.append(f"- Pearson human AoA vs log corpus count: {fmt(obj['correlation_human_aoa_vs_log1p_corpus_count'])}; Spearman: {fmt(obj['spearman_human_aoa_vs_log1p_corpus_count'])}")
        for b, rec in obj["cdi_counts_by_age_bin"].items():
            lines.append(f"  - {b}: types {rec['word_types']}, hits {rec['token_hits']}, mean/type {fmt(rec['mean_hits_per_type'],2)}, median/type {fmt(rec['median_hits_per_type'],2)}")
    lines.append("")
    lines.append("Interpretation: the post-hoc benchmark words are not a clean source-only developmental axis. In clean-Qwen, later-acquired groups have many more target-token hits per type than the earliest group, so a source/difficulty curriculum can easily make the evaluated late words easy early unless the corpus itself is changed for broad reasons. AoA arithmetic alone is not enough.")
    lines.append("")
    lines.append("## Existing trajectory pattern")
    for name in ["clean_qwen_seed43022", "clean_qwen_seed43122", "devcurr_firstpass_seed43022", "devcurr_firstpass_seed43122"]:
        if name not in sur:
            continue
        obj = sur[name]
        lines.append(f"### {name}")
        lines.append(f"- represented CDI words in surprisal file: {obj['represented_cdi_words']}; rows: {obj['rows']}")
        lines.append(f"- corr human AoA vs final surprisal: Pearson {fmt(obj['correlation_human_aoa_vs_final_surprisal'])}, Spearman {fmt(obj['spearman_human_aoa_vs_final_surprisal'])}")
        lines.append(f"- corr human AoA vs total drop: Pearson {fmt(obj['correlation_human_aoa_vs_total_drop'])}, Spearman {fmt(obj['spearman_human_aoa_vs_total_drop'])}")
        lines.append(f"- corr human AoA vs log half-drop time: Pearson {fmt(obj['correlation_human_aoa_vs_log_half_drop_time'])}, Spearman {fmt(obj['spearman_human_aoa_vs_log_half_drop_time'])}, n={obj['n_words_with_positive_drop_and_half_cross']}")
        for step in ["chck_1M", "chck_10M", "chck_100M"]:
            gs = obj["group_step_mean_surprisal"].get(step, {})
            compact = ", ".join(f"{b}:{fmt(v.get('mean'),2)}" for b, v in gs.items())
            lines.append(f"  - {step}: {compact}")
    lines.append("")
    lines.append("The already-trained safe first-pass source-order intervention produced AoA=0.0 for both seeds. For seed43022 it also lowered the broad official-like Overall from the inherited clean-Qwen 41.3443 to 40.6475, with Entity dropping 25.76→23.72 and GlobalPIQA 36.62→34.225. Thus source-block timing has already been falsified as a useful SOTA lever in this form.")
    lines.append("")
    lines.append("## Consequence for next execution")
    lines.append("Do not train any curriculum that uses the official AoA target list as a mask or sampling signal. A legal next candidate must be a permutation or schedule over one ≤10M documented corpus, with each row appearing at most once per epoch, and must be justified by training-only or external developmental evidence such as source type, lexical/conceptual difficulty, dialogue/concreteness proxies, or independent norms not identical to the official target file. Before a 100M run, build a small auditable corpus-order or source-reweighting design and predict its effect on both AoA trajectory and the broad task surface; if it mainly attacks AoA while risking Entity/GlobalPIQA/SuperGLUE, it is not the main route. The semantic-view contrast remains more directly connected to the 41.8 gap; wait for its score evidence rather than consuming H100 time on an unsupported AoA curriculum.")
    lines.append("")
    lines.append(f"Machine-readable audit: `{OUT_JSON}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cdi = load_cdi()
    age_counts = Counter(m["bin"] for m in cdi.values())
    payload: Dict[str, Any] = {
        "status": "AOA_DEVELOPMENTAL_SIGNAL_AUDIT",
        "non_leakage_and_compliance_position": {
            "did_not_train": True,
            "official_aoa_words_used_only_for_posthoc_analysis": True,
            "forbidden_intervention_rejected": "Do not use the exact official AoA target words as a pretraining mask/sampling list.",
            "legal_curriculum_requirement": "A candidate must be a schedule/permutation/reweighting over one documented <=10M-word corpus, with epoch exposure consistent with the <=10 epoch rule and no official evaluation target conditioning."
        },
        "cdi_age_bins": dict(sorted(age_counts.items())),
        "corpus_frequency": {},
        "surprisal_trajectory": {},
        "aoa_local_summaries": read_aoa_summaries(),
        "prior_safe_curricula": {},
        "broad_scores": read_broad_scores(),
    }
    for name, path in [
        ("clean_qwen_10M", CLEAN_QWEN_10M),
        ("official_pool_10M", OFFICIAL_POOL_10M),
        ("devcurr_firstpass_order_10M", DEVCURR_10M),
    ]:
        payload["corpus_frequency"][name] = count_cdi_in_corpus(path, cdi)
    for name, path in SURPRISAL_PATHS.items():
        if path.exists():
            payload["surprisal_trajectory"][name] = load_surprisal(path, cdi)
    if DEVCURR_META.exists():
        payload["prior_safe_curricula"]["source_block_firstpass"] = json.loads(DEVCURR_META.read_text(encoding="utf-8"))
    if SOURCE_BALANCED_META.exists():
        payload["prior_safe_curricula"]["source_balanced_untrained_variants"] = json.loads(SOURCE_BALANCED_META.read_text(encoding="utf-8"))
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    OUT_NOTE.write_text(make_note(payload), encoding="utf-8")
    print(json.dumps({
        "wrote": str(OUT_JSON),
        "note": str(OUT_NOTE),
        "cdi_age_bins": payload["cdi_age_bins"],
        "clean_qwen_cdi_hits": payload["corpus_frequency"]["clean_qwen_10M"]["cdi_counts_by_age_bin"],
        "clean_qwen_aoa_logfreq_pearson": payload["corpus_frequency"]["clean_qwen_10M"]["correlation_human_aoa_vs_log1p_corpus_count"],
        "trajectory_corrs": {k: {
            "aoa_vs_final_surprisal": v["correlation_human_aoa_vs_final_surprisal"],
            "aoa_vs_drop": v["correlation_human_aoa_vs_total_drop"],
            "aoa_vs_half_time": v["correlation_human_aoa_vs_log_half_drop_time"],
        } for k, v in payload["surprisal_trajectory"].items()},
        "broad_scores_keys": list(payload["broad_scores"].keys()),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
