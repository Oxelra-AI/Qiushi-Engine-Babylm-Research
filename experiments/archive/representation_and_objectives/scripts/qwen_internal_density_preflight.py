#!/usr/bin/env python3
"""research: CPU-only preflight for Qwen-internal compact-view density redesign.

This script does not train, evaluate, or generate text.  It converts the compact-view
failure evidence into a safer next-corpus design question: can the already-successful
COMPACT_EXPERIENCE clean-Qwen paired block be compacted internally, preserving all official filler
and original-side evidence, so the recovered word budget can add official rows with
broad relational/action-outcome content?

Selection of new top-up rows uses only the official training pool and a broad,
hand-written relation/action lexicon.  Official evaluation examples are not used for
selection.  AoA target words are loaded only for an after-the-fact exposure audit.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import collections
import hashlib
import json
import math
import pathlib
import re
import statistics
import time
from typing import Any, Iterable

ROOT = _public_path('.')
A01_WS = _public_path('experiments/archive/representation_and_objectives')
COMPACT_EXPERIENCE_QWEN_DIR = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned')
SELECTED_PAIRS = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/selected_pairs.jsonl')
QWEN_POOL = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl')
QWEN_META = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/clean_materialization_metadata.json')
OFFICIAL_POOL = _public_path('experiments/archive/compact_experience/data/mixture/official_pool.jsonl')
A02_LOSS_NOTE = _public_path('research/notes/frontier_consolidation/compact_core_full_loss_anatomy.md')
CDI_CHILDES = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/aoa/cdi_childes.json')
OUT_DIR = _public_path('experiments/archive/representation_and_objectives/data/qwen_internal_density_preflight')
OUT_NOTE = _public_path('research/notes/representation_and_objectives/qwen_internal_density_preflight.md')
OUT_JSON = _public_path('experiments/archive/representation_and_objectives/data/qwen_internal_density_preflight/qwen_internal_density_preflight.json')
PREFERRED_MANIFEST = _public_path('experiments/archive/representation_and_objectives/data/qwen_internal_density_preflight/preferred_len90_r060_compaction_manifest.jsonl')
PREFERRED_TOPUP = _public_path('experiments/archive/representation_and_objectives/data/qwen_internal_density_preflight/preferred_len90_r060_affordance_topup_rows.jsonl')
PREFERRED_NEUTRAL = _public_path('experiments/archive/representation_and_objectives/data/qwen_internal_density_preflight/preferred_len90_r060_neutral_source_matched_rows.jsonl')
PROMPT_SLICE = _public_path('experiments/archive/representation_and_objectives/data/qwen_internal_density_preflight/preferred_len90_r060_compaction_prompt_slice512.jsonl')

TOTAL_WORDS = 10_000_000
OFFICIAL_ROW_WORDS = 160
MIN_COMPACT_REWRITE_WORDS = 6

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[\u2019'][A-Za-z0-9]+)?|\*[A-Za-z]+:?|[£$€]\s*\d+(?:\.\d+)?|\d+(?:[.,:/-]\d+)*(?:%|st|nd|rd|th)?", re.I)

# Broad independent lexicon for training-pool row selection.  It is not derived from
# GlobalPIQA or any other evaluation examples.
ACTION_WORDS = {
    "act", "add", "ask", "bake", "balance", "bend", "bite", "blow", "boil", "break", "bring",
    "build", "burn", "buy", "carry", "catch", "change", "clean", "climb", "close", "collect",
    "come", "cook", "cool", "cover", "crash", "cut", "dig", "do", "draw", "drink", "drive",
    "drop", "dry", "eat", "empty", "fall", "feed", "fill", "find", "fit", "fix", "float", "fold",
    "freeze", "give", "go", "grab", "grow", "hang", "heat", "help", "hide", "hit", "hold", "keep",
    "kick", "lift", "light", "lock", "look", "make", "melt", "mix", "move", "open", "paint", "pay",
    "pick", "place", "play", "pour", "press", "pull", "push", "put", "read", "remove", "repair",
    "ride", "roll", "run", "send", "shake", "show", "shut", "sit", "slide", "spill", "stand", "start",
    "stay", "stop", "take", "tear", "throw", "tie", "touch", "turn", "use", "walk", "wash", "wear",
    "wipe", "work", "write",
}
CAUSAL_TEMPORAL_WORDS = {
    "after", "afterward", "again", "although", "because", "before", "cause", "caused", "causes",
    "change", "changed", "consequence", "continue", "during", "effect", "eventually", "fail", "failed",
    "finally", "first", "happen", "happened", "if", "later", "next", "prevent", "result", "results",
    "since", "so", "therefore", "then", "until", "unless", "when", "whenever", "while",
}
PHYSICAL_OBJECT_WORDS = {
    "air", "bag", "ball", "bed", "bike", "bird", "boat", "book", "bottle", "box", "bread", "brush",
    "building", "car", "chair", "clothes", "coat", "cup", "door", "drink", "egg", "fire", "floor",
    "food", "fork", "glass", "ground", "hand", "hat", "house", "ice", "key", "knife", "light", "milk",
    "money", "paper", "pen", "pencil", "plate", "rock", "room", "rope", "shoe", "spoon", "stairs",
    "stick", "stone", "table", "toy", "tree", "wall", "water", "wheel", "window", "wood",
}
SPATIAL_STATE_WORDS = {
    "above", "across", "around", "back", "behind", "below", "bottom", "down", "far", "front", "high",
    "inside", "left", "low", "near", "next", "off", "on", "outside", "over", "right", "side", "through",
    "top", "under", "up", "wet", "dry", "hot", "cold", "full", "empty", "open", "closed", "broken",
    "clean", "dirty", "heavy", "light", "small", "large",
}
SOCIAL_ROLE_WORDS = {
    "baby", "boy", "child", "children", "dad", "daddy", "doctor", "family", "friend", "girl", "man",
    "mom", "mommy", "mother", "person", "people", "teacher", "woman", "you", "i", "he", "she", "they",
}
LEXICON_VERSION = "broad_action_relation_affordance_v1_not_eval_derived"


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: pathlib.Path, rows: Iterable[dict[str, Any]]) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_tokens(text: str) -> list[str]:
    toks: list[str] = []
    for m in WORD_RE.finditer(text):
        t = m.group(0).strip("'\"“”‘’.,!?;:()[]{}")
        if t:
            toks.append(t.lower())
    return toks


def count_aoa_targets(text: str, aoa_words: set[str]) -> int:
    return sum(1 for t in norm_tokens(text) if t in aoa_words)


def load_aoa_words() -> tuple[set[str], dict[str, str]]:
    data = json.loads(CDI_CHILDES.read_text(encoding="utf-8"))
    words = set(data.keys())
    bins: dict[str, str] = {}
    # Approximate child bin from the mean age records, only for exposure reporting.
    for w, records in data.items():
        ages: list[float] = []
        for r in records:
            for k in ("age", "month", "age_month", "months"):
                if k in r:
                    try:
                        ages.append(float(r[k]))
                    except Exception:
                        pass
                    break
        mean_age = statistics.mean(ages) if ages else math.nan
        if not math.isfinite(mean_age):
            bins[w] = "unknown"
        elif mean_age <= 20:
            bins[w] = "early_le20"
        elif mean_age <= 25:
            bins[w] = "middle_20to25"
        else:
            bins[w] = "late_gt25"
    return words, bins


def quantiles(vals: Iterable[float]) -> dict[str, Any]:
    xs = list(vals)
    if not xs:
        return {"n": 0}
    xs_sorted = sorted(xs)
    def q(p: float) -> float:
        if len(xs_sorted) == 1:
            return xs_sorted[0]
        idx = p * (len(xs_sorted) - 1)
        lo = int(math.floor(idx)); hi = int(math.ceil(idx))
        if lo == hi:
            return xs_sorted[lo]
        return xs_sorted[lo] * (hi - idx) + xs_sorted[hi] * (idx - lo)
    return {
        "n": len(xs),
        "min": min(xs),
        "p05": q(0.05),
        "mean": statistics.mean(xs),
        "median": statistics.median(xs),
        "p95": q(0.95),
        "max": max(xs),
        "sum": sum(xs),
    }


def compact_target_len(original_words: int, current_rewrite_words: int, ratio: float) -> int:
    # Use ceil so the target does not over-promise savings; enforce a readable minimum.
    return min(current_rewrite_words, max(MIN_COMPACT_REWRITE_WORDS, int(math.ceil(original_words * ratio))))


def compaction_candidates(pairs: list[dict[str, Any]], min_len_ratio: float | None, min_content_overlap: float | None, ratio: float, limit: int | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in pairs:
        if min_len_ratio is not None and float(p["len_ratio"]) < min_len_ratio:
            continue
        if min_content_overlap is not None and float(p["content_overlap"]) < min_content_overlap:
            continue
        target = compact_target_len(int(p["original_words"]), int(p["rewrite_words"]), ratio)
        save = int(p["rewrite_words"]) - target
        if save <= 0:
            continue
        o = dict(p)
        o["target_compact_rewrite_words"] = target
        o["expected_saved_words"] = save
        o["target_rewrite_source_ratio"] = round(target / max(1, int(p["original_words"])), 4)
        out.append(o)
    out.sort(key=lambda r: (r["expected_saved_words"], float(r.get("content_overlap", 0.0)), float(r.get("entity_recall", 0.0))), reverse=True)
    if limit is not None:
        out = out[:limit]
    # Stable materialization order by original pair_id after selecting high-save subset.
    out.sort(key=lambda r: str(r["pair_id"]))
    return out


def summarize_compaction(rows: list[dict[str, Any]], ratio: float) -> dict[str, Any]:
    saved = sum(int(r["expected_saved_words"]) for r in rows)
    return {
        "pairs_to_compact": len(rows),
        "target_ratio": ratio,
        "current_rewrite_words_compacted_subset": sum(int(r["rewrite_words"]) for r in rows),
        "target_compact_rewrite_words_subset": sum(int(r["target_compact_rewrite_words"]) for r in rows),
        "expected_saved_words": saved,
        "addable_full_official_160w_rows": saved // OFFICIAL_ROW_WORDS,
        "slack_words_for_exact_materialization": saved % OFFICIAL_ROW_WORDS,
        "expected_saved_fraction_of_10M": saved / TOTAL_WORDS,
        "source_words_compacted_subset": sum(int(r["original_words"]) for r in rows),
        "source_counts_by_pair_words": dict(collections.Counter({})),
        "rewrite_len_ratio_stats_before": quantiles(float(r["len_ratio"]) for r in rows),
        "content_overlap_stats": quantiles(float(r["content_overlap"]) for r in rows),
        "entity_recall_stats": quantiles(float(r.get("entity_recall", 1.0)) for r in rows),
        "expected_save_stats_per_pair": quantiles(int(r["expected_saved_words"]) for r in rows),
    }


def row_relation_features(text: str) -> dict[str, Any]:
    toks = norm_tokens(text)
    cnt = collections.Counter(toks)
    action = sum(cnt[w] for w in ACTION_WORDS)
    causal = sum(cnt[w] for w in CAUSAL_TEMPORAL_WORDS)
    physical = sum(cnt[w] for w in PHYSICAL_OBJECT_WORDS)
    spatial = sum(cnt[w] for w in SPATIAL_STATE_WORDS)
    social = sum(cnt[w] for w in SOCIAL_ROLE_WORDS)
    # Pattern bonuses encourage rows that actually link action, state, and consequence.
    tokset = set(toks)
    if_then = int("if" in tokset and "then" in tokset)
    because_so = int("because" in tokset or "so" in tokset or "cause" in tokset or "result" in tokset)
    action_physical = int(action > 0 and physical > 0)
    action_state = int(action > 0 and (spatial > 0 or causal > 0))
    # Normalize gently; all official rows are nominally 160 words, but some tokenization noise exists.
    denom = max(1, len(toks))
    score = (1.4 * action + 1.2 * causal + 0.9 * physical + 0.7 * spatial + 0.5 * social + 4.0 * if_then + 2.5 * because_so + 2.0 * action_physical + 2.0 * action_state) / denom
    return {
        "tokens": len(toks),
        "action_count": action,
        "causal_temporal_count": causal,
        "physical_object_count": physical,
        "spatial_state_count": spatial,
        "social_role_count": social,
        "if_then_bonus": if_then,
        "because_so_bonus": because_so,
        "action_physical_bonus": action_physical,
        "action_state_bonus": action_state,
        "relation_affordance_score": score,
    }


def source_counter(rows: Iterable[dict[str, Any]], weight_key: str = "words") -> dict[str, int]:
    c: collections.Counter[str] = collections.Counter()
    for r in rows:
        c[str(r["source"])] += int(r.get(weight_key, 0))
    return dict(c)


def normalize_weights(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    if total <= 0:
        return {k: 0.0 for k in counts}
    return {k: v / total for k, v in counts.items()}


def allocate_quotas(n: int, target_weights: dict[str, float], availability: dict[str, int]) -> dict[str, int]:
    raw: dict[str, float] = {s: n * target_weights.get(s, 0.0) for s in availability}
    quotas = {s: min(availability[s], int(math.floor(raw.get(s, 0.0)))) for s in availability}
    remaining = n - sum(quotas.values())
    # First distribute by largest fractional remainder where available.
    while remaining > 0:
        candidates = [s for s in availability if quotas[s] < availability[s]]
        if not candidates:
            break
        candidates.sort(key=lambda s: (raw.get(s, 0.0) - math.floor(raw.get(s, 0.0)), target_weights.get(s, 0.0), availability[s] - quotas[s]), reverse=True)
        s = candidates[0]
        quotas[s] += 1
        remaining -= 1
    return {s: q for s, q in quotas.items() if q > 0}


def select_by_quota(rows: list[dict[str, Any]], quotas: dict[str, int], high: bool = True, used: set[int] | None = None) -> list[dict[str, Any]]:
    used = used if used is not None else set()
    by_source: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        if int(r["example_id"]) in used:
            continue
        by_source[str(r["source"])].append(r)
    selected: list[dict[str, Any]] = []
    for src, q in quotas.items():
        cand = by_source.get(src, [])
        cand.sort(key=lambda r: (float(r["features"]["relation_affordance_score"]), int(r["features"]["action_count"]), int(r["features"]["causal_temporal_count"]), int(r["example_id"])), reverse=high)
        take = cand[:q]
        selected.extend(take)
        used.update(int(r["example_id"]) for r in take)
    # If some quota could not be filled because of used rows, fill globally with the same high/low order.
    missing = sum(quotas.values()) - len(selected)
    if missing > 0:
        rest = [r for r in rows if int(r["example_id"]) not in used]
        rest.sort(key=lambda r: (float(r["features"]["relation_affordance_score"]), int(r["features"]["action_count"]), int(r["features"]["causal_temporal_count"]), int(r["example_id"])), reverse=high)
        selected.extend(rest[:missing])
    selected.sort(key=lambda r: (str(r["source"]), int(r["example_id"])))
    return selected


def rows_summary(rows: list[dict[str, Any]], aoa_words: set[str]) -> dict[str, Any]:
    if not rows:
        return {"rows": 0}
    feats = [r["features"] for r in rows]
    return {
        "rows": len(rows),
        "words": sum(int(r["words"]) for r in rows),
        "source_words": source_counter(rows),
        "source_rows": dict(collections.Counter(str(r["source"]) for r in rows)),
        "relation_affordance_score_stats": quantiles(float(f["relation_affordance_score"]) for f in feats),
        "action_count_stats": quantiles(int(f["action_count"]) for f in feats),
        "causal_temporal_count_stats": quantiles(int(f["causal_temporal_count"]) for f in feats),
        "physical_object_count_stats": quantiles(int(f["physical_object_count"]) for f in feats),
        "spatial_state_count_stats": quantiles(int(f["spatial_state_count"]) for f in feats),
        "aoa_target_occurrences": sum(count_aoa_targets(str(r["text"]), aoa_words) for r in rows),
        "top_examples": [
            {
                "example_id": int(r["example_id"]),
                "source": str(r["source"]),
                "score": round(float(r["features"]["relation_affordance_score"]), 4),
                "feature_counts": {k: r["features"][k] for k in ["action_count", "causal_temporal_count", "physical_object_count", "spatial_state_count", "social_role_count"]},
                "text_prefix": str(r["text"])[:260],
            }
            for r in sorted(rows, key=lambda x: float(x["features"]["relation_affordance_score"]), reverse=True)[:12]
        ],
    }


def removed_suffix_surrogate_summary(compact_rows: list[dict[str, Any]], aoa_words: set[str]) -> dict[str, Any]:
    removed_word_total = 0
    removed_aoa = 0
    removed_by_source: collections.Counter[str] = collections.Counter()
    examples: list[dict[str, Any]] = []
    for r in compact_rows:
        rewrite_words = str(r["rewrite"]).split()
        target = int(r["target_compact_rewrite_words"])
        removed = rewrite_words[target:]
        if not removed:
            continue
        removed_text = " ".join(removed)
        removed_word_total += len(removed)
        removed_aoa += count_aoa_targets(removed_text, aoa_words)
        removed_by_source[str(r["source"])] += len(removed)
        if len(examples) < 10 and int(r["expected_saved_words"]) >= 8:
            examples.append({
                "pair_id": str(r["pair_id"]),
                "source": str(r["source"]),
                "example_id": int(r["example_id"]),
                "current_rewrite_words": int(r["rewrite_words"]),
                "target_compact_rewrite_words": target,
                "removed_suffix_words": len(removed),
                "rewrite_prefix_kept": " ".join(rewrite_words[:target])[:220],
                "removed_suffix_prefix": removed_text[:220],
            })
    return {
        "surrogate": "current-rewrite suffix after target length; actual generated compact views must be audited before training",
        "removed_words": removed_word_total,
        "removed_aoa_target_occurrences": removed_aoa,
        "removed_words_by_source": dict(removed_by_source),
        "examples": examples,
    }


def make_prompt(pair: dict[str, Any]) -> str:
    target = int(pair["target_compact_rewrite_words"])
    return (
        "Rewrite the sentence as a shorter faithful second view for BabyLM pretraining. "
        f"Use at most {target} whitespace-separated words. Preserve all named entities, numbers, and the main relation/action outcome. "
        "Do not add facts. Output only the rewritten sentence.\n"
        f"Sentence: {pair['original']}"
    )


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = read_jsonl(SELECTED_PAIRS)
    qwen_pool = read_jsonl(QWEN_POOL)
    official_pool = read_jsonl(OFFICIAL_POOL)
    qwen_meta = json.loads(QWEN_META.read_text(encoding="utf-8"))
    aoa_words, _aoa_bins = load_aoa_words()

    selected_ids = {int(p["example_id"]) for p in pairs}
    filler_rows = [r for r in qwen_pool if str(r.get("source")) != "qwen_pair_packed"]
    filler_ids = {int(r["example_id"]) for r in filler_rows}
    missing_selected_ids = selected_ids - filler_ids
    official_by_id = {int(r["example_id"]): r for r in official_pool}
    missing_rows: list[dict[str, Any]] = []
    for eid in sorted(missing_selected_ids):
        r = dict(official_by_id[eid])
        r["features"] = row_relation_features(str(r["text"]))
        r["aoa_target_occurrences"] = count_aoa_targets(str(r["text"]), aoa_words)
        missing_rows.append(r)

    official_source_counts = source_counter(official_pool)
    missing_source_counts = source_counter(missing_rows)
    candidate_availability_rows = dict(collections.Counter(str(r["source"]) for r in missing_rows))
    official_w = normalize_weights(official_source_counts)
    missing_w = normalize_weights(missing_source_counts)
    mixed_weights = {s: 0.5 * official_w.get(s, 0.0) + 0.5 * missing_w.get(s, 0.0) for s in set(official_w) | set(missing_w)}
    # Normalize after averaging over union.
    sw = sum(mixed_weights.values())
    mixed_weights = {s: v / sw for s, v in mixed_weights.items() if v > 0}

    scenario_specs = {
        "all_pairs_r060": {"min_len_ratio": None, "min_content_overlap": None, "ratio": 0.60, "limit": None},
        "len90_r060": {"min_len_ratio": 0.90, "min_content_overlap": None, "ratio": 0.60, "limit": None},
        "len90_r065": {"min_len_ratio": 0.90, "min_content_overlap": None, "ratio": 0.65, "limit": None},
        "len90_content80_r060": {"min_len_ratio": 0.90, "min_content_overlap": 0.80, "ratio": 0.60, "limit": None},
        "pilot4096_len90_r060": {"min_len_ratio": 0.90, "min_content_overlap": None, "ratio": 0.60, "limit": 4096},
    }
    scenarios: dict[str, Any] = {}
    compaction_rows_by_scenario: dict[str, list[dict[str, Any]]] = {}
    for name, spec in scenario_specs.items():
        rows = compaction_candidates(pairs, spec["min_len_ratio"], spec["min_content_overlap"], spec["ratio"], spec["limit"])
        compaction_rows_by_scenario[name] = rows
        summ = summarize_compaction(rows, spec["ratio"])
        src_words: collections.Counter[str] = collections.Counter()
        for r in rows:
            src_words[str(r["source"])] += int(r["expected_saved_words"])
        summ["saved_words_by_pair_source"] = dict(src_words)
        add_rows = int(summ["addable_full_official_160w_rows"])
        quotas = allocate_quotas(add_rows, mixed_weights, candidate_availability_rows) if add_rows > 0 else {}
        selected_topup = select_by_quota(missing_rows, quotas, high=True, used=set()) if add_rows > 0 else []
        neutral = select_by_quota(missing_rows, dict(collections.Counter(str(r["source"]) for r in selected_topup)), high=False, used={int(r["example_id"]) for r in selected_topup}) if selected_topup else []
        summ["topup_design"] = {
            "selection_source": "official_pool rows whose official example_id appears in selected Qwen pairs but whose full 160-word row is not already in the clean-Qwen filler",
            "selection_lexicon_version": LEXICON_VERSION,
            "eval_examples_used_for_selection": False,
            "target_weighting": "0.5*official_source_distribution + 0.5*available_missing_selected_source_distribution, clipped to availability; rows sorted by broad relation/action-affordance score within source",
            "quotas_rows": quotas,
            "affordance_topup_summary": rows_summary(selected_topup, aoa_words),
            "neutral_source_matched_summary": rows_summary(neutral, aoa_words),
            "removed_rewrite_suffix_surrogate_summary": removed_suffix_surrogate_summary(rows, aoa_words),
        }
        if selected_topup:
            summ["topup_minus_removed_suffix_surrogate"] = {
                "aoa_target_occurrence_delta": summ["topup_design"]["affordance_topup_summary"]["aoa_target_occurrences"] - summ["topup_design"]["removed_rewrite_suffix_surrogate_summary"]["removed_aoa_target_occurrences"],
                "official_words_added_minus_rewrite_suffix_words_removed": summ["topup_design"]["affordance_topup_summary"]["words"] - summ["topup_design"]["removed_rewrite_suffix_surrogate_summary"]["removed_words"],
                "note": "Delta uses a suffix-removal surrogate for the old rewrite; actual generated compact text must be audited. Positive AoA occurrence delta means added official rows contain more CDI target occurrences than the removed rewrite suffix surrogate.",
            }
        scenarios[name] = summ

    preferred_rows = compaction_rows_by_scenario["len90_r060"]
    preferred_add_rows = scenarios["len90_r060"]["addable_full_official_160w_rows"]
    preferred_quotas = allocate_quotas(preferred_add_rows, mixed_weights, candidate_availability_rows)
    preferred_topup = select_by_quota(missing_rows, preferred_quotas, high=True, used=set())
    preferred_neutral = select_by_quota(missing_rows, dict(collections.Counter(str(r["source"]) for r in preferred_topup)), high=False, used={int(r["example_id"]) for r in preferred_topup})

    # Compact manifest writes enough exact information for a future generator/materializer, but no generated text.
    manifest_rows = []
    for r in preferred_rows:
        manifest_rows.append({
            "pair_id": r["pair_id"],
            "source": r["source"],
            "example_id": r["example_id"],
            "original": r["original"],
            "current_rewrite": r["rewrite"],
            "original_words": r["original_words"],
            "current_rewrite_words": r["rewrite_words"],
            "target_compact_rewrite_words": r["target_compact_rewrite_words"],
            "expected_saved_words": r["expected_saved_words"],
            "content_overlap": r["content_overlap"],
            "entity_recall": r["entity_recall"],
            "num_source": r.get("num_source", []),
            "entity_source": r.get("entity_source", []),
        })
    write_jsonl(PREFERRED_MANIFEST, manifest_rows)
    write_jsonl(PREFERRED_TOPUP, [{k: v for k, v in r.items() if k != "features"} | {"features": r["features"]} for r in preferred_topup])
    write_jsonl(PREFERRED_NEUTRAL, [{k: v for k, v in r.items() if k != "features"} | {"features": r["features"]} for r in preferred_neutral])
    prompt_rows = []
    # Stratified prompt slice: highest-save, high-content, and source coverage in one bounded file.
    by_source: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in preferred_rows:
        by_source[str(r["source"])].append(r)
    for src, rows in by_source.items():
        rows.sort(key=lambda x: (int(x["expected_saved_words"]), float(x["content_overlap"])), reverse=True)
        for r in rows[:max(8, 512 // max(1, len(by_source)))]:
            prompt_rows.append(r)
    if len(prompt_rows) < 512:
        already = {str(r["pair_id"]) for r in prompt_rows}
        rest = [r for r in preferred_rows if str(r["pair_id"]) not in already]
        rest.sort(key=lambda x: (float(x["content_overlap"]), int(x["expected_saved_words"])), reverse=True)
        prompt_rows.extend(rest[:512 - len(prompt_rows)])
    prompt_rows = prompt_rows[:512]
    write_jsonl(PROMPT_SLICE, [
        {
            "pair_id": r["pair_id"],
            "source": r["source"],
            "example_id": r["example_id"],
            "target_compact_rewrite_words": r["target_compact_rewrite_words"],
            "expected_saved_words": r["expected_saved_words"],
            "original": r["original"],
            "current_rewrite": r["rewrite"],
            "prompt": make_prompt(r),
        }
        for r in prompt_rows
    ])

    current_qwen_counts = source_counter(qwen_pool)
    qwen_pair_ids_by_source = collections.Counter(str(p["source"]) for p in pairs)
    current_pair_words_by_source = collections.Counter()
    for p in pairs:
        current_pair_words_by_source[str(p["source"])] += int(p["pair_words"])

    payload = {
        "status": "QWEN_INTERNAL_DENSITY_PREFLIGHT",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "CPU-only redesign preflight after compact FineWeb replacement core failed full evaluation: preserve the COMPACT_EXPERIENCE clean-Qwen developmental allocation, compact redundant generated rewrites inside the already-successful Qwen pair block, and use saved words for official relational/action-outcome rows selected without evaluation examples.",
        "expensive_work_status": "no GPU, no generation, no evaluation, no 100M training launched; pending research reinvest results remain decisive before any expensive work",
        "input_files": {
            "selected_pairs": str(SELECTED_PAIRS.relative_to(ROOT)),
            "qwen_pool": str(QWEN_POOL.relative_to(ROOT)),
            "official_pool": str(OFFICIAL_POOL.relative_to(ROOT)),
            "qwen_metadata": str(QWEN_META.relative_to(ROOT)),
            "a02_loss_note": str(A02_LOSS_NOTE.relative_to(ROOT)),
            "aoa_words_for_audit_only": str(CDI_CHILDES.relative_to(ROOT)),
        },
        "input_sha256": {
            "selected_pairs.jsonl": sha256_file(SELECTED_PAIRS),
            "qwen_aligned_10M.jsonl": sha256_file(QWEN_POOL),
            "official_pool.jsonl": sha256_file(OFFICIAL_POOL),
        },
        "clean_qwen_baseline": {
            "selected_pairs": len(pairs),
            "unique_selected_official_example_ids": len(selected_ids),
            "pair_words": int(qwen_meta["selected_pair_words"]),
            "pair_word_fraction": qwen_meta["selected_pair_word_fraction"],
            "qwen_pool_rows": len(qwen_pool),
            "qwen_pool_words": sum(int(r["words"]) for r in qwen_pool),
            "qwen_pair_packed_words": current_qwen_counts.get("qwen_pair_packed", 0),
            "official_filler_words_preserved_by_redesign": sum(int(r["words"]) for r in filler_rows),
            "current_qwen_source_words": current_qwen_counts,
            "current_pair_words_by_original_source": dict(current_pair_words_by_source),
            "current_pair_count_by_original_source": dict(qwen_pair_ids_by_source),
        },
        "missing_full_official_row_pool_for_reinvestment": {
            "definition": "selected Qwen source example_ids whose full 160-word official row is not already present as an official filler row in qwen_aligned_10M",
            "selected_ids": len(selected_ids),
            "filler_ids": len(filler_ids),
            "missing_selected_ids": len(missing_selected_ids),
            "candidate_rows": len(missing_rows),
            "candidate_words": sum(int(r["words"]) for r in missing_rows),
            "source_words": missing_source_counts,
            "source_rows": candidate_availability_rows,
            "relation_affordance_summary": rows_summary(missing_rows, aoa_words),
        },
        "source_weighting": {
            "official_source_weights": official_w,
            "missing_candidate_source_weights": missing_w,
            "mixed_preferred_weights": mixed_weights,
            "reason": "avoid replacing the clean-Qwen substrate; if saved words are reinvested, keep broad official-source coverage rather than selecting only one high-score source",
        },
        "scenarios": scenarios,
        "preferred_design": {
            "name": "len90_r060",
            "why_preferred_if_pending_endpoint_fails": "Compacts only current Qwen rewrites at or above near-parity length, leaves low-ratio already-short rewrites untouched, preserves all original sides and all existing official filler, saves a learning-relevant ~2%+ corpus budget, and reinvests into full official rows rather than FineWeb replacement.",
            "not_authorized_until": "read the completed reinvest full vector and seed43122 result; if reinvest has substantial full-NLP gain with neutral AoA, this design may be superseded; if it lacks full-NLP/AoA, this becomes the safer density continuation candidate",
            "manifest_files": {
                "compaction_manifest": str(PREFERRED_MANIFEST.relative_to(ROOT)),
                "affordance_topup_rows": str(PREFERRED_TOPUP.relative_to(ROOT)),
                "neutral_source_matched_rows": str(PREFERRED_NEUTRAL.relative_to(ROOT)),
                "prompt_slice512": str(PROMPT_SLICE.relative_to(ROOT)),
            },
        },
        "next_low_cost_checks_before_any_100M_run": [
            "Generate only the 512-prompt slice or a similarly small stratified slice to measure whether Qwen can produce faithful <=target compact rewrites inside the COMPACT_EXPERIENCE pair distribution; audit entity/number/content retention and realized saved words.",
            "Materialize a 10M dry-run only after real compact outputs exist; audit exact word accounting, row packing, seq256 source+compact-view visibility, token geometry, source distribution, AoA target exposure, and no phrase-level overlap with official evaluation data as a contamination safeguard, not a selector.",
            "Train only if pending reinvest results show the FineWeb replacement endpoint lacks complete-eval value and the compact-slice audit shows high-fidelity compaction; use a matched neutral-topup control or endpoint comparison that can predict full evaluation, not fast seven-column surface alone.",
        ],
        "elapsed_sec": round(time.time() - t0, 3),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    pref = scenarios["len90_r060"]
    top = pref["topup_design"]["affordance_topup_summary"]
    neut = pref["topup_design"]["neutral_source_matched_summary"]
    delta = pref.get("topup_minus_removed_suffix_surrogate", {})
    lines = [
        "# research Qwen-internal density preflight",
        "",
        "CPU-only redesign work. No GPU, generation, official evaluation, or training was launched; the pending research reinvest results remain the next decisive evidence.",
        "",
        "## Why this redesign is different from the failed compact FineWeb replacement",
        "- Full compact_view_core did not merely have an AoA problem: replacing its AoA by zero would lift Overall only to about 41.40 because its full NLP average is essentially tied to clean-Qwen while it loses Supplement, GlobalPIQA, and SuperGLUE.",
        "- The safer density question is therefore internal to the already-successful COMPACT_EXPERIENCE clean-Qwen allocation: shorten redundant generated rewrites while preserving all original sides and all official filler rows, then use the recovered words for additional official rows rather than a new FineWeb replacement block.",
        "",
        "## Clean-Qwen substrate facts",
        f"- Selected Qwen pairs: {len(pairs)} pairs / {qwen_meta['selected_pair_words']:,} words ({100*qwen_meta['selected_pair_word_fraction']:.3f}% of the pool).",
        f"- Existing official filler that this redesign preserves: {sum(int(r['words']) for r in filler_rows):,} words.",
        f"- Selected official example IDs: {len(selected_ids):,}; full official rows not already in filler: {len(missing_selected_ids):,} rows / {sum(int(r['words']) for r in missing_rows):,} words.",
        "",
        "## Preferred if FineWeb-reinvest full vector fails: len_ratio>=0.90, target rewrite/source ratio 0.60",
        f"- Pairs to compact: {pref['pairs_to_compact']:,}; expected saved words: {pref['expected_saved_words']:,} ({100*pref['expected_saved_fraction_of_10M']:.3f}% of 10M).",
        f"- Addable full official 160-word rows: {pref['addable_full_official_160w_rows']:,}; exact-materialization slack to absorb by length adjustment: {pref['slack_words_for_exact_materialization']} words.",
        f"- Compacted-subset current rewrite words -> target compact rewrite words: {pref['current_rewrite_words_compacted_subset']:,} -> {pref['target_compact_rewrite_words_subset']:,}.",
        f"- Top-up row source counts: {top['source_rows']}.",
        f"- Affordance top-up score mean/p95: {top['relation_affordance_score_stats']['mean']:.4f}/{top['relation_affordance_score_stats']['p95']:.4f}; neutral source-matched mean/p95: {neut['relation_affordance_score_stats']['mean']:.4f}/{neut['relation_affordance_score_stats']['p95']:.4f}.",
        f"- After-the-fact AoA-target exposure audit using a rewrite-suffix surrogate: added top-up minus removed suffix = {delta.get('aoa_target_occurrence_delta')} target occurrences; this is not a selection signal and must be repeated after actual compact generation.",
        "",
        "## Other CPU-only scenarios",
    ]
    for name, s in scenarios.items():
        lines.append(f"- {name}: compact {s['pairs_to_compact']:,} pairs, save {s['expected_saved_words']:,} words, add {s['addable_full_official_160w_rows']:,} official rows, slack {s['slack_words_for_exact_materialization']}.")
    lines += [
        "",
        "## Files written for future low-cost checks",
        f"- Preferred compaction manifest: `{PREFERRED_MANIFEST.relative_to(ROOT)}`",
        f"- Preferred relation/action top-up rows: `{PREFERRED_TOPUP.relative_to(ROOT)}`",
        f"- Source-matched neutral rows for a future matched comparison: `{PREFERRED_NEUTRAL.relative_to(ROOT)}`",
        f"- 512-prompt slice for a future compact-generation audit: `{PROMPT_SLICE.relative_to(ROOT)}`",
        f"- Machine-readable summary: `{OUT_JSON.relative_to(ROOT)}`",
        "",
        "## Scientific use",
        "This preflight is a design asset, not a training result. It should be used only after the pending reinvest full vector is read. If reinvest lacks a substantial full-NLP gain or inherits negative AoA, the FineWeb-replacement endpoint should be retired and this Qwen-internal density route can be tested first by a small compact-generation slice and a 10M dry-run audit. If reinvest unexpectedly has strong full-NLP with neutral AoA, preserve that endpoint instead of diverting to this redesign.",
    ]
    OUT_NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": payload["status"],
        "out_json": str(OUT_JSON.relative_to(ROOT)),
        "out_note": str(OUT_NOTE.relative_to(ROOT)),
        "preferred_pairs": pref["pairs_to_compact"],
        "preferred_saved_words": pref["expected_saved_words"],
        "preferred_add_rows": pref["addable_full_official_160w_rows"],
        "missing_full_rows": len(missing_rows),
        "preferred_topup_score_mean": top["relation_affordance_score_stats"]["mean"],
        "neutral_score_mean": neut["relation_affordance_score_stats"]["mean"],
        "aoa_delta_surrogate": delta.get("aoa_target_occurrence_delta"),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
