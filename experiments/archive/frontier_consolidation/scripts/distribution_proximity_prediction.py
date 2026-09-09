#!/usr/bin/env python3
"""research CPU-only preregistered distribution-proximity prediction.

Scientific role
---------------
Before MAX-register scores are read, this script commits a quantitative
prediction for the contrast

    childspeech_removed - adultprose_removed

where both arms admit the identical research MAX FineWeb compact-view block.  The
model fitted here is intentionally simple:

    score_delta(family, arm vs clean) = beta * [d(family, removed_prop_clean)
                                               - d(family, admitted_arm)]

Positive score_delta means the admitted block is closer to the evaluation family
than the proportional clean block it displaced.  The future register contrast
cancels the admitted block and becomes

    beta * [d(family, removed_childspeech) - d(family, removed_adultprose)].

The script computes two independent distance views:
  * word_js: lowercase word unigram JS with numbers collapsed;
  * profile_js: closed-class/function + punctuation/shape profile JS, excluding
    open-class lexical identities except through abstract shapes.

No model loading, training, official evaluation, GPU work, GlobalPIQA,
SuperGLUE, AoA, upload, or leaderboard action.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import collections
import csv
import json
import math
import pathlib
import re
import statistics
import time
from typing import Any, Iterable


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
OUT = WS / "data/distribution_proximity_prediction"
NOTE = WS / "notes/distribution_proximity_prediction.md"
BASE_POOL = ROOT / "experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl"
POOL256 = WS / "data/dose_2p64x_rowholdout_pools"
VIEW10 = POOL256 / "compact_view_dose2p64x_10M.jsonl"
REPEAT10 = POOL256 / "compact_repeat_dose2p64x_10M.jsonl"
HELDOUT_PROP = POOL256 / "heldout_cleanqwen_rows.jsonl"
BREADTH10 = WS / "data/dose_2p64x_breadth_rowholdout_pools/compact_breadth_dose2p64x_10M.jsonl"
REGSEL = WS / "data/register_max_rowholdout_pools/regmax_selection_records.jsonl"
REGMETA = WS / "data/register_max_rowholdout_pools/register_max_rowholdout_metadata.json"
CONTRAST_CSV = WS / "data/reference_decomposition_readout/contrast_window_summaries.csv"
EVAL_ROOT = ROOT / "experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval"
PAIR_ROWS = 7923

FAMILIES = ["BLiMP", "Supplement", "EWoK", "COMPS", "Reading", "Entity"]
EX_ENTITY = ["BLiMP", "Supplement", "EWoK", "COMPS", "Reading"]
ARM_BLOCK = {"V": "admitted_view", "B": "admitted_breadth", "R": "admitted_repeat"}
ARM_CONTRAST = {"V": "D1_VminusCmax", "B": "D1_BminusCmax", "R": "D1_RminusCmax"}
PRIMARY_MODEL_NAME = "common10_80_all3_exEntity"
PRIMARY_METRIC = "word_js"
PROFILE_METRIC = "profile_js"
SEED_SPREAD_REFERENCE = {
    "source": "research interpretation-scale note / research_record: same-coordinate mature spread estimates",
    "cheap6_pointwise_spread": 0.4199,
    "exEntity5_mature_spread": 0.1765,
}

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:[.,]\d+)*", re.UNICODE)
TOK_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:[.,]\d+)*|[^\w\s]", re.UNICODE)
SPEAKER_RE = re.compile(r"\*[A-Z]{2,5}:")
TIER_RE = re.compile(r"%[a-z]{2,5}:")
BRACKET_RE = re.compile(r"\[[^\]]{0,80}\]")
URL_RE = re.compile(r"https?://|www\.", re.I)

FUNCTION_CATS: dict[str, str] = {}

def _add(cat: str, words: str) -> None:
    for w in words.split():
        FUNCTION_CATS[w] = cat

_add("determiner", "a an the this that these those another each every either neither no some any both all much many few fewer several such what whatever which whichever")
_add("pronoun", "i me my mine myself we us our ours ourselves you your yours yourself yourselves he him his himself she her hers herself it its itself they them their theirs themselves one oneself who whom whose whoever whomever")
_add("auxiliary", "am is are was were be been being do does did have has had having can could may might must shall should will would ought")
_add("preposition", "about above across after against along amid among around as at before behind below beneath beside besides between beyond by concerning despite down during except for from in inside into like near of off on onto out outside over past per regarding since through throughout to toward towards under underneath until unto up upon versus via with within without")
_add("conjunction", "and or but nor so yet for although because before since unless until while whereas whether if though than once whenever where wherever after")
_add("negation", "not n't never none nobody nothing nowhere neither nor")
_add("wh", "who whom whose what which when where why how")
_add("quantifier", "all any both each every few many most much no several some enough more less least fewer half twice once")
_add("discourse", "yes yeah yep no okay ok well oh ah uh um hmm hey hello please thanks thank sorry")
_add("degree", "very too quite rather really just almost already still even only also again ever")
_add("deictic", "here there now then today tomorrow yesterday")
_add("particle", "up down out off on over back away around apart")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def iter_jsonl(path: pathlib.Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def clean_text(s: Any) -> str:
    return " ".join(str(s).replace("\u0000", " ").split())


def row_words(row: dict[str, Any]) -> int:
    try:
        return int(row.get("words", len(clean_text(row.get("text", "")).split())))
    except Exception:
        return len(clean_text(row.get("text", "")).split())


def word_tokens(text: str) -> list[str]:
    out: list[str] = []
    for m in WORD_RE.finditer(text):
        t = m.group(0).lower()
        if any(ch.isdigit() for ch in t):
            out.append("<num>")
        else:
            out.append(t)
    return out


def word_counter_from_texts(texts: Iterable[str]) -> collections.Counter[str]:
    c: collections.Counter[str] = collections.Counter()
    for text in texts:
        c.update(word_tokens(text))
    return c


def len_bin(n: int) -> str:
    if n <= 2:
        return "1-2"
    if n <= 4:
        return "3-4"
    if n <= 7:
        return "5-7"
    if n <= 11:
        return "8-11"
    return "12+"


def sent_len_bin(n: int) -> str:
    if n <= 5:
        return "1-5"
    if n <= 10:
        return "6-10"
    if n <= 20:
        return "11-20"
    if n <= 35:
        return "21-35"
    return "36+"


def shape_of(tok: str) -> str:
    if any(ch.isdigit() for ch in tok):
        return "num"
    if tok.isupper():
        return "upper"
    if tok[:1].isupper():
        return "title"
    if tok.islower():
        return "lower"
    return "mixed"


def punct_cat(tok: str) -> str:
    if tok in ".!?":
        return f"sentence_{tok}"
    if tok in ",;:":
        return "clause_punct"
    if tok in "\"'“”‘’":
        return "quote"
    if tok in "-–—":
        return "dash"
    if tok in "()[]{}":
        return "bracket"
    if tok in "*/\\|":
        return "transcript_or_markup"
    return "other_punct"


def profile_counter_from_texts(texts: Iterable[str]) -> collections.Counter[str]:
    c: collections.Counter[str] = collections.Counter()
    for text0 in texts:
        text = clean_text(text0)
        if not text:
            continue
        for _ in SPEAKER_RE.finditer(text):
            c["format:speaker_marker"] += 1
        for _ in TIER_RE.finditer(text):
            c["format:transcript_tier"] += 1
        for _ in BRACKET_RE.finditer(text):
            c["format:bracketed_stage_or_note"] += 1
        if URL_RE.search(text):
            c["format:url_like"] += 1
        sents = [s for s in re.split(r"[.!?]+", text) if s.strip()]
        for s in sents:
            c[f"sent_len:{sent_len_bin(len(word_tokens(s)))}"] += 1
        for tok in TOK_RE.findall(text):
            if not tok.strip():
                continue
            if not WORD_RE.fullmatch(tok):
                c[f"punct:{punct_cat(tok)}"] += 1
                continue
            lw = tok.lower()
            if any(ch.isdigit() for ch in lw):
                c["tok:number"] += 1
                c[f"tok_len:{len_bin(len(lw))}"] += 1
                continue
            cat = FUNCTION_CATS.get(lw)
            if cat:
                # Closed-class exact identities are retained only for function words;
                # open-class lexical identities are deliberately not counted here.
                c[f"func_cat:{cat}"] += 1
                c[f"func_word:{lw}"] += 1
            else:
                c[f"open_shape:{shape_of(tok)}"] += 1
                c[f"open_len:{len_bin(len(lw))}"] += 1
                if "'" in lw:
                    c["open_has_apostrophe"] += 1
                for suf in ["ing", "ed", "ly", "ion", "ment", "ness", "s"]:
                    if lw.endswith(suf) and len(lw) > len(suf) + 2:
                        c[f"open_suffix:{suf}"] += 1
                        break
    return c


def counters_from_texts(texts: list[str]) -> dict[str, collections.Counter[str]]:
    return {"word_js": word_counter_from_texts(texts), "profile_js": profile_counter_from_texts(texts)}


def first_n_texts(path: pathlib.Path, n: int) -> tuple[list[str], dict[str, Any]]:
    texts: list[str] = []
    source_words: collections.Counter[str] = collections.Counter()
    word_total = 0
    rows = 0
    for i, row in enumerate(iter_jsonl(path)):
        if i >= n:
            break
        text = clean_text(row.get("text", ""))
        texts.append(text)
        rows += 1
        w = row_words(row)
        word_total += w
        source_words[str(row.get("source", "unknown")).split("::")[-1]] += w
    return texts, {"path": rel(path), "rows": rows, "row_words": word_total, "source_words": dict(source_words.most_common())}


def all_jsonl_texts(path: pathlib.Path) -> tuple[list[str], dict[str, Any]]:
    texts: list[str] = []
    source_words: collections.Counter[str] = collections.Counter()
    word_total = 0
    rows = 0
    for row in iter_jsonl(path):
        text = clean_text(row.get("text", ""))
        texts.append(text)
        rows += 1
        w = row_words(row)
        word_total += w
        source_words[str(row.get("source", "unknown")).split("::")[-1]] += w
    return texts, {"path": rel(path), "rows": rows, "row_words": word_total, "source_words": dict(source_words.most_common())}


def selected_base_texts(child_idx: set[int], adult_idx: set[int]) -> tuple[dict[str, list[str]], dict[str, Any]]:
    texts = {"removed_childspeech": [], "removed_adultprose": []}
    stats: dict[str, Any] = {
        "removed_childspeech": {"rows": 0, "row_words": 0, "source_words": collections.Counter()},
        "removed_adultprose": {"rows": 0, "row_words": 0, "source_words": collections.Counter()},
    }
    all_idx = child_idx | adult_idx
    max_idx = max(all_idx) if all_idx else -1
    with BASE_POOL.open(encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i > max_idx and len(texts["removed_childspeech"]) == len(child_idx) and len(texts["removed_adultprose"]) == len(adult_idx):
                break
            if i not in all_idx:
                continue
            row = json.loads(line)
            for name, idxset in [("removed_childspeech", child_idx), ("removed_adultprose", adult_idx)]:
                if i in idxset:
                    t = clean_text(row.get("text", ""))
                    texts[name].append(t)
                    w = row_words(row)
                    stats[name]["rows"] += 1
                    stats[name]["row_words"] += w
                    stats[name]["source_words"][str(row.get("source", "unknown")).split("::")[-1]] += w
    for name in stats:
        stats[name]["source_words"] = dict(stats[name]["source_words"].most_common())
        stats[name]["path_basis"] = rel(BASE_POOL)
    return texts, stats


def load_selection_indices() -> tuple[set[int], set[int], dict[str, Any]]:
    child: set[int] = set()
    adult: set[int] = set()
    distances_child: list[int] = []
    distances_adult: list[int] = []
    distances_pair: list[int] = []
    with REGSEL.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            child.add(int(r["childspeech_base_index"]))
            adult.add(int(r["adultprose_base_index"]))
            distances_child.append(int(r.get("childspeech_distance_to_target", 0)))
            distances_adult.append(int(r.get("adultprose_distance_to_target", 0)))
            distances_pair.append(int(r.get("child_adult_abs_distance", 0)))
    def dist_summary(xs: list[int]) -> dict[str, Any]:
        xs2 = sorted(xs)
        return {
            "n": len(xs),
            "mean": statistics.mean(xs) if xs else None,
            "median": statistics.median(xs) if xs else None,
            "p90": xs2[int(0.9 * (len(xs2) - 1))] if xs2 else None,
            "max": max(xs2) if xs2 else None,
        }
    return child, adult, {
        "selection_records": rel(REGSEL),
        "child_unique_indices": len(child),
        "adult_unique_indices": len(adult),
        "child_distance_to_prop_target": dist_summary(distances_child),
        "adult_distance_to_prop_target": dist_summary(distances_adult),
        "child_adult_sorted_pair_distance": dist_summary(distances_pair),
    }


def add_if_string(texts: list[str], value: Any) -> None:
    if isinstance(value, str):
        s = clean_text(value)
        if s:
            texts.append(s)
    elif isinstance(value, list):
        for v in value:
            add_if_string(texts, v)
    elif isinstance(value, dict):
        for v in value.values():
            add_if_string(texts, v)


def extract_jsonl_record_texts(family: str, rec: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    if family in {"BLiMP", "Supplement"}:
        for k in ["sentence_good", "sentence_bad", "one_prefix_word_good", "one_prefix_word_bad"]:
            add_if_string(texts, rec.get(k))
        return texts
    if family == "EWoK":
        for k, v in rec.items():
            kl = str(k).lower()
            if kl.startswith("context") or kl.startswith("target") or kl.startswith("concept") or kl in {"question", "answer"}:
                add_if_string(texts, v)
        return texts
    if family == "Entity":
        for k, v in rec.items():
            kl = str(k).lower()
            if any(s in kl for s in ["input", "prefix", "target", "answer", "option", "candidate", "choice"]):
                add_if_string(texts, v)
        return texts
    if family == "COMPS":
        for k, v in rec.items():
            kl = str(k).lower()
            if any(s in kl for s in ["property", "concept", "prefix", "phrase", "sentence", "acceptable", "unacceptable"]):
                add_if_string(texts, v)
        return texts
    # Conservative fallback for future small schema variations.
    for k, v in rec.items():
        if isinstance(v, str) and len(v.split()) >= 2:
            add_if_string(texts, v)
    return texts


def eval_family_texts(family: str) -> tuple[list[str], dict[str, Any]]:
    texts: list[str] = []
    files: list[str] = []
    rows = 0
    if family == "Reading":
        p = EVAL_ROOT / "reading/reading_data.csv"
        seen: set[str] = set()
        with p.open(encoding="utf-8", errors="replace", newline="") as f:
            for row in csv.DictReader(f):
                rows += 1
                s = clean_text(row.get("sentence", ""))
                if s and s not in seen:
                    seen.add(s)
                    texts.append(s)
        files.append(rel(p))
        return texts, {"family": family, "files": files, "rows": rows, "texts": len(texts), "reading_unique_sentences": True}
    subdir = {
        "BLiMP": "blimp_filtered",
        "Supplement": "supplement_filtered",
        "EWoK": "ewok_filtered",
        "Entity": "entity_tracking",
        "COMPS": "comps",
    }[family]
    d = EVAL_ROOT / subdir
    for p in sorted(d.glob("*.jsonl")):
        files.append(rel(p))
        for rec in iter_jsonl(p):
            rows += 1
            texts.extend(extract_jsonl_record_texts(family, rec))
    return texts, {"family": family, "dir": rel(d), "files": files, "rows": rows, "texts": len(texts)}


def js_divergence(p_counts: collections.Counter[str], q_counts: collections.Counter[str]) -> float:
    np = float(sum(p_counts.values()))
    nq = float(sum(q_counts.values()))
    if np <= 0 or nq <= 0:
        return float("nan")
    out = 0.0
    keys = set(p_counts) | set(q_counts)
    for k in keys:
        p = p_counts.get(k, 0) / np
        q = q_counts.get(k, 0) / nq
        m = 0.5 * (p + q)
        if p > 0:
            out += 0.5 * p * math.log(p / m, 2)
        if q > 0:
            out += 0.5 * q * math.log(q / m, 2)
    return out


def top_items(c: collections.Counter[str], n: int = 20) -> list[dict[str, Any]]:
    total = sum(c.values()) or 1
    return [{"item": k, "count": int(v), "p": v / total} for k, v in c.most_common(n)]


def counter_stats(name: str, meta: dict[str, Any], counters: dict[str, collections.Counter[str]]) -> dict[str, Any]:
    return {
        "block_or_family": name,
        **meta,
        "word_unigram_events": int(sum(counters["word_js"].values())),
        "word_unigram_types": int(len(counters["word_js"])),
        "profile_events": int(sum(counters["profile_js"].values())),
        "profile_types": int(len(counters["profile_js"])),
        "word_top20": top_items(counters["word_js"], 20),
        "profile_top20": top_items(counters["profile_js"], 20),
    }


def read_score_summaries() -> dict[tuple[str, str, str], dict[str, Any]]:
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    with CONTRAST_CSV.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                mean = float(row["mean"])
            except Exception:
                continue
            row2 = dict(row)
            row2["mean"] = mean
            row2["n"] = int(row.get("n") or 0)
            out[(row["contrast"], row["quantity"], row["window"])] = row2
    return out


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def ranks(vals: list[float]) -> list[float]:
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    out = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and vals[order[j]] == vals[order[i]]:
            j += 1
        r = (i + 1 + j) / 2.0
        for k in range(i, j):
            out[order[k]] = r
        i = j
    return out


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    return pearson(ranks(xs), ranks(ys))


def fit_beta(rows: list[dict[str, Any]]) -> dict[str, Any]:
    xs = [float(r["x_distance_advantage"]) for r in rows]
    ys = [float(r["score_delta_points"]) for r in rows]
    denom = sum(x * x for x in xs)
    beta = sum(x * y for x, y in zip(xs, ys)) / denom if denom > 0 else float("nan")
    preds = [beta * x for x in xs]
    sse = sum((y - p) ** 2 for y, p in zip(ys, preds))
    syy0 = sum(y * y for y in ys)
    syyc = sum((y - statistics.mean(ys)) ** 2 for y in ys) if len(ys) > 1 else float("nan")
    return {
        "beta_score_points_per_js_bit": beta,
        "n": len(rows),
        "x_min": min(xs) if xs else None,
        "x_max": max(xs) if xs else None,
        "x_mean": statistics.mean(xs) if xs else None,
        "y_mean": statistics.mean(ys) if ys else None,
        "through_origin_r2_vs_zero": (1 - sse / syy0) if syy0 > 0 else None,
        "centered_r2_reference": (1 - sse / syyc) if syyc > 0 else None,
        "pearson_x_y": pearson(xs, ys),
        "spearman_x_y": spearman(xs, ys),
        "rmse_points": math.sqrt(sse / len(rows)) if rows else None,
        "rows": rows,
    }


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    NOTE.parent.mkdir(parents=True, exist_ok=True)

    # Exact training blocks.
    block_texts: dict[str, list[str]] = {}
    block_meta: dict[str, dict[str, Any]] = {}
    for name, path in [("admitted_view", VIEW10), ("admitted_repeat", REPEAT10), ("admitted_breadth", BREADTH10)]:
        block_texts[name], block_meta[name] = first_n_texts(path, PAIR_ROWS)
        block_meta[name]["definition"] = f"first {PAIR_ROWS} changed-block rows; 133-word clean topup omitted"
    block_texts["removed_proportional"], block_meta["removed_proportional"] = all_jsonl_texts(HELDOUT_PROP)
    block_meta["removed_proportional"]["definition"] = "original research proportional clean rows displaced by V/B/R/Cmax contrast"
    child_idx, adult_idx, selection_meta = load_selection_indices()
    selected_texts, selected_stats = selected_base_texts(child_idx, adult_idx)
    for name in ["removed_childspeech", "removed_adultprose"]:
        block_texts[name] = selected_texts[name]
        block_meta[name] = selected_stats[name]
        block_meta[name]["definition"] = "research MAX-register selected clean rows removed from that arm; 133-word arm-specific topup not subtracted"
    block_meta["removed_childspeech"].update({"register_selection": "CHILDES/OpenSubtitles/BNC Spoken/Switchboard"})
    block_meta["removed_adultprose"].update({"register_selection": "Gutenberg/SimpleWiki"})

    all_counters: dict[str, dict[str, collections.Counter[str]]] = {}
    stats_rows: list[dict[str, Any]] = []
    for name, texts in block_texts.items():
        c = counters_from_texts(texts)
        all_counters[name] = c
        stats_rows.append(counter_stats(name, block_meta[name], c))

    eval_meta: dict[str, dict[str, Any]] = {}
    for fam in FAMILIES:
        texts, meta = eval_family_texts(fam)
        eval_meta[fam] = meta
        c = counters_from_texts(texts)
        all_counters[f"eval_{fam}"] = c
        stats_rows.append(counter_stats(f"eval_{fam}", meta, c))

    # Distance table.
    distance_rows: list[dict[str, Any]] = []
    distances: dict[str, dict[str, dict[str, float]]] = {m: {fam: {} for fam in FAMILIES} for m in [PRIMARY_METRIC, PROFILE_METRIC]}
    blocks = ["admitted_view", "admitted_breadth", "admitted_repeat", "removed_proportional", "removed_childspeech", "removed_adultprose"]
    for metric in [PRIMARY_METRIC, PROFILE_METRIC]:
        for fam in FAMILIES:
            ec = all_counters[f"eval_{fam}"][metric]
            for block in blocks:
                d = js_divergence(ec, all_counters[block][metric])
                distances[metric][fam][block] = d
                distance_rows.append({"metric": metric, "family": fam, "block": block, "js_bits": d})

    scores = read_score_summaries()
    fit_rows_by_spec: dict[str, dict[str, Any]] = {}
    calibration_long_rows: list[dict[str, Any]] = []
    fit_specs = [
        {"name": "common10_80_all3_exEntity", "window": "common10_80", "arms": ["V", "B", "R"], "families": EX_ENTITY, "interpretation": "primary common-window calibration requested for the first register readout"},
        {"name": "common10_80_distinct_VB_exEntity", "window": "common10_80", "arms": ["V", "B"], "families": EX_ENTITY, "interpretation": "distinct admitted-content calibration, excluding the duplicate arm"},
        {"name": "late80_100_all3_exEntity", "window": "late80_100", "arms": ["V", "B", "R"], "families": EX_ENTITY, "interpretation": "late common calibration; shows duplicate recurrence decay"},
        {"name": "late80_100_distinct_VB_exEntity", "window": "late80_100", "arms": ["V", "B"], "families": EX_ENTITY, "interpretation": "persistent distinct-content calibration"},
        {"name": "common10_80_all3_withEntity", "window": "common10_80", "arms": ["V", "B", "R"], "families": FAMILIES, "interpretation": "secondary view including Entity, whose operation mixture is known to behave differently"},
    ]
    for metric in [PRIMARY_METRIC, PROFILE_METRIC]:
        for spec in fit_specs:
            rows: list[dict[str, Any]] = []
            for arm in spec["arms"]:
                for fam in spec["families"]:
                    sc = scores.get((ARM_CONTRAST[arm], fam, spec["window"]))
                    if sc is None:
                        continue
                    x = distances[metric][fam]["removed_proportional"] - distances[metric][fam][ARM_BLOCK[arm]]
                    row = {
                        "metric": metric,
                        "fit_spec": spec["name"],
                        "window": spec["window"],
                        "arm": arm,
                        "contrast": ARM_CONTRAST[arm],
                        "family": fam,
                        "x_distance_advantage": x,
                        "score_delta_points": sc["mean"],
                        "score_n_checkpoints": sc["n"],
                        "admitted_block": ARM_BLOCK[arm],
                        "removed_block": "removed_proportional",
                    }
                    rows.append(row)
                    calibration_long_rows.append(row)
            fit = fit_beta(rows)
            fit.update({k: spec[k] for k in ["name", "window", "arms", "families", "interpretation"]})
            fit["metric"] = metric
            fit_rows_by_spec[f"{metric}:{spec['name']}"] = fit

    # Register predictions from each fitted beta.
    prediction_rows: list[dict[str, Any]] = []
    aggregate_prediction_rows: list[dict[str, Any]] = []
    for fit_key, fit in fit_rows_by_spec.items():
        metric = fit["metric"]
        beta = float(fit["beta_score_points_per_js_bit"])
        per_family: dict[str, float] = {}
        for fam in FAMILIES:
            xreg = distances[metric][fam]["removed_childspeech"] - distances[metric][fam]["removed_adultprose"]
            pred = beta * xreg
            per_family[fam] = pred
            row = {
                "metric": metric,
                "fit_spec": fit["name"],
                "window_calibrated": fit["window"],
                "family": fam,
                "x_child_minus_adult_distance": xreg,
                "pred_childspeech_removed_minus_adultprose_removed_points": pred,
                "sign": "positive_childspeech_arm_higher" if pred > 0 else ("negative_adultprose_arm_higher" if pred < 0 else "zero"),
                "beta_score_points_per_js_bit": beta,
                "fit_n": fit["n"],
                "fit_pearson_x_y": fit.get("pearson_x_y"),
                "fit_rmse_points": fit.get("rmse_points"),
            }
            prediction_rows.append(row)
        for quantity, fams in [("exEntity5", EX_ENTITY), ("cheap6", FAMILIES)]:
            vals = [per_family[f] for f in fams]
            aggregate_prediction_rows.append({
                "metric": metric,
                "fit_spec": fit["name"],
                "quantity": quantity,
                "pred_childspeech_removed_minus_adultprose_removed_points": statistics.mean(vals),
                "families": ";".join(fams),
                "min_family_pred": min(vals),
                "max_family_pred": max(vals),
                "positive_family_count": sum(1 for v in vals if v > 0),
                "negative_family_count": sum(1 for v in vals if v < 0),
                "vs_exEntity5_mature_spread_multiple": (statistics.mean(vals) / SEED_SPREAD_REFERENCE["exEntity5_mature_spread"]) if quantity == "exEntity5" else None,
                "vs_cheap6_pointwise_spread_multiple": (statistics.mean(vals) / SEED_SPREAD_REFERENCE["cheap6_pointwise_spread"]) if quantity == "cheap6" else None,
            })

    # Metric agreement for the register distance ordering and fitted predictions.
    agreement_rows: list[dict[str, Any]] = []
    for spec_name in {fit["name"] for fit in fit_rows_by_spec.values()}:
        for fams_name, fams in [("exEntity", EX_ENTITY), ("all6", FAMILIES)]:
            word_x = [distances[PRIMARY_METRIC][f]["removed_childspeech"] - distances[PRIMARY_METRIC][f]["removed_adultprose"] for f in fams]
            prof_x = [distances[PROFILE_METRIC][f]["removed_childspeech"] - distances[PROFILE_METRIC][f]["removed_adultprose"] for f in fams]
            word_pred = [r["pred_childspeech_removed_minus_adultprose_removed_points"] for r in prediction_rows if r["metric"] == PRIMARY_METRIC and r["fit_spec"] == spec_name and r["family"] in fams]
            prof_pred = [r["pred_childspeech_removed_minus_adultprose_removed_points"] for r in prediction_rows if r["metric"] == PROFILE_METRIC and r["fit_spec"] == spec_name and r["family"] in fams]
            agreement_rows.append({
                "fit_spec": spec_name,
                "family_set": fams_name,
                "distance_pearson_word_profile": pearson(word_x, prof_x),
                "distance_spearman_word_profile": spearman(word_x, prof_x),
                "prediction_pearson_word_profile": pearson(word_pred, prof_pred) if len(word_pred) == len(prof_pred) else None,
                "prediction_spearman_word_profile": spearman(word_pred, prof_pred) if len(word_pred) == len(prof_pred) else None,
                "sign_agreement_count": sum((wx > 0) == (px > 0) for wx, px in zip(word_x, prof_x)),
                "n": len(fams),
            })

    # Duplicate recurrence decay constraint in the already scored vectors.
    def score_mean(contrast: str, quantity: str, window: str) -> float | None:
        row = scores.get((contrast, quantity, window))
        return float(row["mean"]) if row else None

    def safe_ratio(a: float | None, b: float | None) -> float | None:
        return (a / b) if a is not None and b not in (None, 0) else None

    common_v = score_mean("D1_VminusCmax", "exEntity5", "common10_80")
    common_b = score_mean("D1_BminusCmax", "exEntity5", "common10_80")
    common_r = score_mean("D1_RminusCmax", "exEntity5", "common10_80")
    late_v = score_mean("D1_VminusCmax", "exEntity5", "late80_100")
    late_b = score_mean("D1_BminusCmax", "exEntity5", "late80_100")
    late_r = score_mean("D1_RminusCmax", "exEntity5", "late80_100")
    duplicate_decay = {
        "quantity": "exEntity5",
        "common10_80": {"VminusCmax": common_v, "BminusCmax": common_b, "RminusCmax": common_r, "R_over_mean_VB": safe_ratio(common_r, statistics.mean([common_v, common_b]) if common_v is not None and common_b is not None else None)},
        "late80_100": {"VminusCmax": late_v, "BminusCmax": late_b, "RminusCmax": late_r, "R_over_mean_VB": safe_ratio(late_r, statistics.mean([late_v, late_b]) if late_v is not None and late_b is not None else None)},
        "interpretation": "The duplicate repeat arm initially moves with the FineWeb replacement cluster but its broad ex-Entity advantage nearly vanishes late while distinct-content V/B remain positive; a distribution-proximity account must include an independent-content/persistence factor rather than treating repeated text as equally useful indefinitely.",
    }

    # Primary committed rows are the all-three common-window word fit,
    # with profile companion to test whether ordering survives without open-class lexical identities.
    primary_predictions = [r for r in prediction_rows if r["metric"] == PRIMARY_METRIC and r["fit_spec"] == PRIMARY_MODEL_NAME]
    profile_predictions = [r for r in prediction_rows if r["metric"] == PROFILE_METRIC and r["fit_spec"] == PRIMARY_MODEL_NAME]
    primary_aggregates = [r for r in aggregate_prediction_rows if r["metric"] == PRIMARY_METRIC and r["fit_spec"] == PRIMARY_MODEL_NAME]
    profile_aggregates = [r for r in aggregate_prediction_rows if r["metric"] == PROFILE_METRIC and r["fit_spec"] == PRIMARY_MODEL_NAME]

    # Write flat tables.
    def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        # Keep nested columns JSON-encoded.
        keys: list[str] = []
        for r in rows:
            for k in r.keys():
                if k not in keys:
                    keys.append(k)
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                rr = {}
                for k in keys:
                    v = r.get(k)
                    if isinstance(v, (dict, list)):
                        rr[k] = json.dumps(v, ensure_ascii=False, sort_keys=True)
                    else:
                        rr[k] = v
                w.writerow(rr)

    write_csv(OUT / "block_and_eval_stats.csv", stats_rows)
    write_csv(OUT / "distance_rows.csv", distance_rows)
    write_csv(OUT / "calibration_rows.csv", calibration_long_rows)
    # Flatten fit summary without embedding the long row list.
    fit_summary_rows = []
    for k, fit in fit_rows_by_spec.items():
        fit_summary_rows.append({kk: vv for kk, vv in fit.items() if kk != "rows"})
    write_csv(OUT / "fit_summary_rows.csv", fit_summary_rows)
    write_csv(OUT / "prediction_rows.csv", prediction_rows)
    write_csv(OUT / "aggregate_prediction_rows.csv", aggregate_prediction_rows)
    write_csv(OUT / "metric_agreement_rows.csv", agreement_rows)

    summary = {
        "status": "DISTRIBUTION_PROXIMITY_PREDICTION_COMPLETE",
        "created_utc": now(),
        "purpose": "CPU-only pre-score quantitative prediction for MAX-register childspeech_removed - adultprose_removed under a fixed-budget distribution-proximity model.",
        "inputs": {
            "view10": rel(VIEW10),
            "repeat10": rel(REPEAT10),
            "breadth10": rel(BREADTH10),
            "heldout_proportional": rel(HELDOUT_PROP),
            "selection_records": rel(REGSEL),
            "register_metadata": rel(REGMETA),
            "contrast_window_summaries": rel(CONTRAST_CSV),
            "eval_root": rel(EVAL_ROOT),
        },
        "selection_meta": selection_meta,
        "block_meta": block_meta,
        "eval_meta": eval_meta,
        "metrics": {
            "word_js": "lowercase word unigram JS in bits with digit strings collapsed to <num>",
            "profile_js": "closed-class/function words plus punctuation, transcript markers, sentence-length, token-shape and suffix profiles; open-class identities removed",
        },
        "fit_model": "score_delta_points = beta * (JS(eval, removed_prop_clean) - JS(eval, admitted_arm)); beta fitted through origin",
        "fit_summaries": fit_rows_by_spec,
        "primary_model": {
            "metric": PRIMARY_METRIC,
            "fit_spec": PRIMARY_MODEL_NAME,
            "fit": {kk: vv for kk, vv in fit_rows_by_spec[f"{PRIMARY_METRIC}:{PRIMARY_MODEL_NAME}"].items() if kk != "rows"},
            "family_predictions": primary_predictions,
            "aggregate_predictions": primary_aggregates,
        },
        "profile_companion_same_spec": {
            "metric": PROFILE_METRIC,
            "fit_spec": PRIMARY_MODEL_NAME,
            "fit": {kk: vv for kk, vv in fit_rows_by_spec[f"{PROFILE_METRIC}:{PRIMARY_MODEL_NAME}"].items() if kk != "rows"},
            "family_predictions": profile_predictions,
            "aggregate_predictions": profile_aggregates,
        },
        "metric_agreement": agreement_rows,
        "duplicate_decay_constraint": duplicate_decay,
        "seed_spread_reference": SEED_SPREAD_REFERENCE,
        "future_score_reading": {
            "contrast_definition": "childspeech_removed - adultprose_removed; positive means the arm sacrificing developmental/speech rows scores higher than the arm sacrificing Gutenberg/SimpleWiki rows",
            "adult_prose_removal_cost_prediction": "The proximity model predicts positive values for families whose evaluation distribution is closer to the adult-prose block than to the developmental/speech block.",
            "observations_that_weaken_the_account": [
                "near-zero aggregate exEntity5 contrast when the committed prediction is substantially larger than the measured seed-spread scale",
                "negative childspeech_removed - adultprose_removed on most adult/editable-English-close families despite positive distance differences",
                "a family ordering unrelated to both word_js and profile_js distance-difference ordering",
                "a result explainable only by Entity operation allocation while ex-Entity families remain flat",
            ],
        },
        "files": {
            "summary_json": rel(OUT / "distribution_proximity_prediction_summary.json"),
            "summary_md": rel(OUT / "distribution_proximity_prediction_summary.md"),
            "note_md": rel(NOTE),
            "block_and_eval_stats_csv": rel(OUT / "block_and_eval_stats.csv"),
            "distance_rows_csv": rel(OUT / "distance_rows.csv"),
            "calibration_rows_csv": rel(OUT / "calibration_rows.csv"),
            "fit_summary_rows_csv": rel(OUT / "fit_summary_rows.csv"),
            "prediction_rows_csv": rel(OUT / "prediction_rows.csv"),
            "aggregate_prediction_rows_csv": rel(OUT / "aggregate_prediction_rows.csv"),
            "metric_agreement_rows_csv": rel(OUT / "metric_agreement_rows.csv"),
        },
        "no_model_loading_training_official_evaluation_gpu_globalpiqa_superglue_aoa_upload_or_leaderboard": True,
        "elapsed_sec": round(time.time() - t0, 3),
    }
    # Avoid storing huge Counter objects; fit rows are plain dicts already but nested row lists are long and useful.
    (OUT / "distribution_proximity_prediction_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Concise scientific note.
    def fmt(x: Any, nd: int = 4) -> str:
        if x is None:
            return "NA"
        try:
            return f"{float(x):.{nd}f}"
        except Exception:
            return str(x)

    primary_fit = summary["primary_model"]["fit"]
    profile_fit = summary["profile_companion_same_spec"]["fit"]
    prim_agg = {r["quantity"]: r for r in primary_aggregates}
    prof_agg = {r["quantity"]: r for r in profile_aggregates}
    primary_lines = []
    profile_lines = []
    for fam in FAMILIES:
        pr = next(r for r in primary_predictions if r["family"] == fam)
        qr = next(r for r in profile_predictions if r["family"] == fam)
        primary_lines.append(f"| {fam} | {fmt(pr['x_child_minus_adult_distance'], 5)} | {fmt(pr['pred_childspeech_removed_minus_adultprose_removed_points'], 3)} | {pr['sign']} |")
        profile_lines.append(f"| {fam} | {fmt(qr['x_child_minus_adult_distance'], 5)} | {fmt(qr['pred_childspeech_removed_minus_adultprose_removed_points'], 3)} | {qr['sign']} |")
    agree_primary = [r for r in agreement_rows if r["fit_spec"] == PRIMARY_MODEL_NAME and r["family_set"] == "exEntity"][0]
    md = [
        "# research distribution-proximity prediction before MAX-register scores",
        "",
        "This CPU-only result was written before reading any MAX-register score. It commits a quantitative prediction for `childspeech_removed - adultprose_removed`, where positive means the arm that sacrifices CHILDES/OpenSubtitles/BNC/Switchboard scores higher than the arm that sacrifices Gutenberg/SimpleWiki. Both arms admit the identical research MAX FineWeb compact-view block, so the admitted side cancels in the contrast.",
        "",
        "## Exact text blocks used",
        "",
        f"- admitted view block: first {PAIR_ROWS} rows of `{rel(VIEW10)}`, {block_meta['admitted_view']['row_words']} row words.",
        f"- admitted breadth block: first {PAIR_ROWS} rows of `{rel(BREADTH10)}`, {block_meta['admitted_breadth']['row_words']} row words.",
        f"- admitted repeat block: first {PAIR_ROWS} rows of `{rel(REPEAT10)}`, {block_meta['admitted_repeat']['row_words']} row words.",
        f"- proportional removed clean block: `{rel(HELDOUT_PROP)}`, {block_meta['removed_proportional']['row_words']} row words.",
        f"- childspeech removed block: {block_meta['removed_childspeech']['row_words']} row words, sources {block_meta['removed_childspeech']['source_words']}.",
        f"- adultprose removed block: {block_meta['removed_adultprose']['row_words']} row words, sources {block_meta['removed_adultprose']['source_words']}.",
        "- The 133-word arm-specific topup is reported in research metadata and is not used as the removed block; it is 0.0119% of the MAX substitution words.",
        "",
        "## Calibration equation",
        "",
        "For each evaluation family and V/B/R arm, the fitted scalar is",
        "",
        "`score_delta_points = beta * (JS(eval_family, removed_proportional_clean) - JS(eval_family, admitted_arm))`.",
        "",
        "The primary calibration uses the three matched-geometry vectors V-Cmax, B-Cmax, and R-Cmax on BLiMP, Supplement, EWoK, COMPS, and Reading over common10_80, because the queued register scorer targets chck_10M..chck_80M.",
        "",
        f"Primary word-JS beta: **{fmt(primary_fit['beta_score_points_per_js_bit'], 3)} score points / JS bit** over n={primary_fit['n']}; Pearson x-y={fmt(primary_fit['pearson_x_y'], 3)}, Spearman={fmt(primary_fit['spearman_x_y'], 3)}, RMSE={fmt(primary_fit['rmse_points'], 3)} points.",
        f"Profile-JS companion beta on the same rows: **{fmt(profile_fit['beta_score_points_per_js_bit'], 3)} score points / JS bit**; Pearson x-y={fmt(profile_fit['pearson_x_y'], 3)}, Spearman={fmt(profile_fit['spearman_x_y'], 3)}, RMSE={fmt(profile_fit['rmse_points'], 3)} points.",
        "",
        "## Committed primary word-JS prediction",
        "",
        "| family | JS(child removed)-JS(adult removed) | predicted score points | sign |",
        "|---|---:|---:|---|",
        *primary_lines,
        "",
        f"Primary word-JS aggregate exEntity5 prediction: **{fmt(prim_agg['exEntity5']['pred_childspeech_removed_minus_adultprose_removed_points'], 3)} points**, {fmt(prim_agg['exEntity5']['vs_exEntity5_mature_spread_multiple'], 2)}x the research exEntity mature seed-spread reference {SEED_SPREAD_REFERENCE['exEntity5_mature_spread']}. Primary cheap6 prediction including Entity: **{fmt(prim_agg['cheap6']['pred_childspeech_removed_minus_adultprose_removed_points'], 3)} points**, {fmt(prim_agg['cheap6']['vs_cheap6_pointwise_spread_multiple'], 2)}x the research cheap6 pointwise spread reference {SEED_SPREAD_REFERENCE['cheap6_pointwise_spread']}.",
        "",
        "## Non-open-lexical profile companion",
        "",
        "The profile metric removes open-class identities and retains closed-class/function words, punctuation, transcript markers, sentence length, token shape, and suffix categories. It tests whether the expected family ordering survives beyond lexical frequency.",
        "",
        "| family | profile distance child-adult | predicted score points | sign |",
        "|---|---:|---:|---|",
        *profile_lines,
        "",
        f"Profile aggregate exEntity5 prediction: **{fmt(prof_agg['exEntity5']['pred_childspeech_removed_minus_adultprose_removed_points'], 3)} points**; cheap6 prediction: **{fmt(prof_agg['cheap6']['pred_childspeech_removed_minus_adultprose_removed_points'], 3)} points**. Word/profile exEntity distance-order Pearson={fmt(agree_primary['distance_pearson_word_profile'], 3)}, Spearman={fmt(agree_primary['distance_spearman_word_profile'], 3)}, sign agreement {agree_primary['sign_agreement_count']}/{agree_primary['n']}.",
        "",
        "## Required persistence clause from the existing matched-geometry vectors",
        "",
        f"Under matched geometry, R-Cmax exEntity5 falls from {fmt(common_r, 4)} common10_80 to {fmt(late_r, 4)} late80_100, while V-Cmax and B-Cmax remain {fmt(late_v, 4)} and {fmt(late_b, 4)} late. R/mean(V,B) is {fmt(duplicate_decay['common10_80']['R_over_mean_VB'], 3)} common and {fmt(duplicate_decay['late80_100']['R_over_mean_VB'], 3)} late. Therefore a viable principle is not just distance-to-target; repeated exposure to the same admitted text loses broad ex-Entity value late, while distinct admitted content persists. The register pair should be read as a test of the sacrificed-register term under distinct admitted content, not as support for unlimited duplication.",
        "",
        "## How future MAX-register scores bear on this account",
        "",
        "If the observed childspeech_removed - adultprose_removed vector is positive and roughly follows the committed family ordering on ex-Entity families, the fixed-budget proximity account gains quantitative support: scarce words buy downstream competence when they are spent near the target distribution and when the admitted content remains distinct. If the contrast is near zero or reversed on the adult/editable-English-close families, or if only Entity moves while ex-Entity remains flat, the account loses force and the interpretation should shift toward admitted-content idiosyncrasy, DeBERTa coordinate effects, or structure beyond these two distance views.",
        "",
        "## Files",
        "",
        f"- summary JSON: `{rel(OUT / 'distribution_proximity_prediction_summary.json')}`",
        f"- distance rows: `{rel(OUT / 'distance_rows.csv')}`",
        f"- calibration rows: `{rel(OUT / 'calibration_rows.csv')}`",
        f"- family predictions: `{rel(OUT / 'prediction_rows.csv')}`",
        f"- aggregate predictions: `{rel(OUT / 'aggregate_prediction_rows.csv')}`",
        f"- metric agreement: `{rel(OUT / 'metric_agreement_rows.csv')}`",
    ]
    md_text = "\n".join(md) + "\n"
    (OUT / "distribution_proximity_prediction_summary.md").write_text(md_text, encoding="utf-8")
    NOTE.write_text(md_text, encoding="utf-8")

    print(json.dumps({
        "status": summary["status"],
        "summary_json": rel(OUT / "distribution_proximity_prediction_summary.json"),
        "summary_md": rel(OUT / "distribution_proximity_prediction_summary.md"),
        "note_md": rel(NOTE),
        "primary_beta": primary_fit["beta_score_points_per_js_bit"],
        "primary_exEntity5_pred": prim_agg["exEntity5"]["pred_childspeech_removed_minus_adultprose_removed_points"],
        "primary_cheap6_pred": prim_agg["cheap6"]["pred_childspeech_removed_minus_adultprose_removed_points"],
        "profile_exEntity5_pred": prof_agg["exEntity5"]["pred_childspeech_removed_minus_adultprose_removed_points"],
        "profile_cheap6_pred": prof_agg["cheap6"]["pred_childspeech_removed_minus_adultprose_removed_points"],
        "word_profile_exEntity_spearman": agree_primary["distance_spearman_word_profile"],
        "duplicate_decay": duplicate_decay,
        "no_model_loading_training_official_evaluation_gpu_globalpiqa_superglue_aoa_upload_or_leaderboard": True,
        "elapsed_sec": summary["elapsed_sec"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
