#!/usr/bin/env python3
"""research fast matched neutral-control builder for transition substrate.

The first matched-control builder was too expensive because it rescanned every
reservoir with the full strict feature extractor.  This repair keeps the scientific
object but lowers cost: it uses the already-built treatment slices and collects
neutral sentences from the same allowed reservoirs with a lightweight low-transition
screen, stopping once each requested source/length budget is filled.

No official evaluation item text is read.  No training is launched.
"""
from __future__ import annotations

import csv
import json
import re
import statistics
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

USER_ROOT = Path.cwd()
A01_WS = USER_ROOT / "experiments/archive/representation_and_objectives"
OUT = A01_WS / "data/transition_matched_controls_fast"
NOTE = A01_WS / "notes/transition_matched_controls_fast.md"
BALANCED_200K = A01_WS / "data/strict_transition_substrate/strict_transition_balanced_200k_slice.jsonl"
BUDGETS = [50_000, 100_000, 200_000]

RESERVOIRS: dict[str, dict[str, Any]] = {
    "compact_experience_aligned_10m_rows": {
        "path": USER_ROOT / "experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl",
        "fields": ["text"],
        "kind": "current_lineage_10M_pool",
    },
    "fw_frozen_sources_38167": {
        "path": A01_WS / "data/fw_mechanism_source_selection/fw_mechanism_frozen_sources.jsonl",
        "fields": ["text"],
        "kind": "fineweb_candidate_source_reservoir",
    },
    "fw_compact_rewrites": {
        "path": A01_WS / "data/fw_full_preservation/full26k_usable_pairs_for_materializer.jsonl",
        "fields": ["rewrite_text"],
        "kind": "selected_qwen35_compact_rewrites",
    },
    "fw_breadth_whole_sentence_companions": {
        "path": A01_WS / "data/fw_source_breadth_wholesentence_arm/source_breadth_wholesentence_companion_sources.jsonl",
        "fields": ["text"],
        "kind": "selected_fineweb_breadth_companions",
    },
}

WORD_RE = re.compile(r"\b[\w]+(?:['’-][\w]+)?\b", re.UNICODE)
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“‘\[])|\n+")
# Lightweight exclusion: many allowed neutral sentences still contain common nouns;
# exclude only obvious causal/conditional/change/contrast/procedure vocabulary.
TRANSITION_RE = re.compile(
    r"\b(if|when|whenever|unless|because|therefore|result|results|resulted|cause|causes|caused|leads?|prevent|allows?|so that|in order to|"
    r"become|became|change|changed|changes|turns? into|from\b.{0,35}\bto|increase|decrease|more|less|fewer|full|empty|hot|cold|wet|dry|"
    r"push|pull|throw|drop|fall|fell|bounce|break|broke|broken|bend|melt|freeze|dissolve|open|close|up|down|above|below|inside|outside|"
    r"before|after|first|last|next|earlier|later|how to|best|safe|careful)\b",
    re.I,
)
NOISE_RE = re.compile(r"https?://|www\.|</?\w+|\[[^\]]*(illustration|edit|citation)[^\]]*\]", re.I)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def wc(text: str) -> int:
    return len(WORD_RE.findall(text))


def split_sentences(text: str) -> list[str]:
    out = []
    for s in SENT_SPLIT_RE.split(text):
        s = norm(s)
        if not s:
            continue
        for sub in re.split(r"\s+(?=\*[A-Z]{2,5}:)|\s+(?=Q:)|\s+(?=A:)", s):
            sub = norm(sub)
            if sub:
                out.append(sub)
    return out


def low_transition_score(text: str) -> int:
    # Count broad transition tokens for summary; accepted neutral rows usually have zero.
    return len(TRANSITION_RE.findall(text))


def is_neutral(text: str, words: int) -> bool:
    if words < 8 or words > 85:
        return False
    if NOISE_RE.search(text):
        return False
    if low_transition_score(text) > 0:
        return False
    # Avoid dense proper-name/news rows as neutral controls for physical relation probes.
    if len(re.findall(r"\b[A-Z][a-z]{2,}\b", text)) >= 8:
        return False
    return True


def treatment_subset(balanced: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    rows = sorted(balanced, key=lambda r: (r.get("selected_route_bucket", ""), -float(r.get("score", 0)), abs(int(r["words"]) - 32), r.get("text_sha256", "")))
    selected = []
    total = 0
    for r in rows:
        w = int(r["words"])
        if total + w <= budget:
            selected.append(r); total += w
        if total == budget:
            return selected
    # exact repair with one row from unselected if possible
    rem = budget - total
    for r in rows:
        if r in selected:
            continue
        if int(r["words"]) == rem:
            selected.append(r); total += rem; return selected
    return selected


def source_word_targets(treatment: list[dict[str, Any]]) -> dict[str, int]:
    out = Counter()
    for r in treatment:
        out[str(r.get("source_label", ""))] += int(r["words"])
    return dict(out)


def collect_neutral(target_words_by_source: dict[str, int], max_extra_per_source: int = 2000) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = []
    seen_text = set()
    got = Counter()
    scanned = {}
    # We over-collect each source by a small margin so exact-ish repair has options.
    for label, target_words in target_words_by_source.items():
        spec = RESERVOIRS.get(label)
        if spec is None:
            # Some treatment sources such as fw_compact_pair_sources can be controlled by frozen sources.
            spec = RESERVOIRS.get("fw_frozen_sources_38167")
        if spec is None:
            continue
        path = Path(spec["path"])
        sent_scanned = 0; word_scanned = 0; accepted = 0
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for row_index, line in enumerate(f):
                if got[label] >= target_words + max_extra_per_source:
                    break
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for field in spec["fields"]:
                    text = str(obj.get(field, "") or "")
                    if not text:
                        continue
                    for sentence_index, sent in enumerate(split_sentences(text)):
                        words = wc(sent)
                        sent_scanned += 1; word_scanned += words
                        if not is_neutral(sent, words):
                            continue
                        key = re.sub(r"\s+", " ", sent.lower()).strip()
                        if key in seen_text:
                            continue
                        seen_text.add(key)
                        rec = {
                            "source_label": label,
                            "source_kind": spec["kind"],
                            "input_path": str(path),
                            "row_index": row_index,
                            "sentence_index": sentence_index,
                            "field": field,
                            "text": sent,
                            "words": words,
                            "neutral_transition_score": low_transition_score(sent),
                            "origin_source": obj.get("source") or obj.get("pool") or obj.get("source_kind") or obj.get("origin_source"),
                            "doc_id": obj.get("doc_id"),
                            "norm_hash": obj.get("norm_hash"),
                        }
                        selected.append(rec); got[label] += words; accepted += 1
                        if got[label] >= target_words + max_extra_per_source:
                            break
                    if got[label] >= target_words + max_extra_per_source:
                        break
        scanned[label] = {"target_words": target_words, "collected_words": got[label], "accepted_sentences": accepted, "sentences_scanned": sent_scanned, "words_scanned": word_scanned, "path": str(path)}
    return selected, scanned


def pick_exact_neutral(pool: list[dict[str, Any]], target_words_by_source: dict[str, int], target_total: int) -> list[dict[str, Any]]:
    # Source-matched greedy; exact total repair can trade source balance at the end.
    by_source = defaultdict(list)
    for r in pool:
        by_source[str(r.get("source_label", ""))].append(r)
    for src, rows in by_source.items():
        rows.sort(key=lambda r: (abs(int(r["words"]) - 30), str(r.get("origin_source") or ""), r["text"]))
    chosen = []
    used_ids = set()
    total = 0
    for src, target in target_words_by_source.items():
        sw = 0
        for r in by_source.get(src, []):
            if id(r) in used_ids:
                continue
            w = int(r["words"])
            if sw + w <= target and total + w <= target_total:
                chosen.append(r); used_ids.add(id(r)); sw += w; total += w
            if sw == target:
                break
    # Fill exact residual if possible from any source.
    pool_left = [r for r in pool if id(r) not in used_ids]
    pool_left.sort(key=lambda r: (abs(int(r["words"]) - max(1, target_total - total)), int(r["words"]), r["text"]))
    while total < target_total:
        rem = target_total - total
        exact = next((r for r in pool_left if int(r["words"]) == rem), None)
        if exact is not None:
            chosen.append(exact); total += int(exact["words"]); break
        candidates = [r for r in pool_left if int(r["words"]) <= rem]
        if not candidates:
            break
        r = min(candidates, key=lambda x: (abs(int(x["words"]) - min(30, rem)), x["text"]))
        chosen.append(r); total += int(r["words"]); pool_left.remove(r)
    return chosen


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"sentences": 0, "words": 0}
    return {
        "sentences": len(rows),
        "words": sum(int(r["words"]) for r in rows),
        "mean_words": statistics.fmean(int(r["words"]) for r in rows),
        "median_words": statistics.median(int(r["words"]) for r in rows),
        "source_words": dict(Counter({k: sum(int(r["words"]) for r in rows if str(r.get("source_label", "")) == k) for k in {str(r.get("source_label", "")) for r in rows}})),
        "source_counts": dict(Counter(str(r.get("source_label", "")) for r in rows)),
    }


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    balanced = read_jsonl(BALANCED_200K)
    summary = {
        "status": "TRANSITION_MATCHED_CONTROLS_FAST_BUILT",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "boundary": "Allowed reservoirs only; no official evaluation text; no training.",
        "treatment_source": str(BALANCED_200K),
        "budgets": {},
    }
    csv_rows = []
    for budget in BUDGETS:
        treatment = treatment_subset(balanced, budget)
        target_total = sum(int(r["words"]) for r in treatment)
        by_source_target = source_word_targets(treatment)
        neutral_pool, scanned = collect_neutral(by_source_target, max_extra_per_source=1500)
        neutral = pick_exact_neutral(neutral_pool, by_source_target, target_total)
        tpath = OUT / f"transition_treatment_{budget//1000}k.jsonl"
        npath = OUT / f"neutral_control_{budget//1000}k.jsonl"
        write_jsonl(tpath, treatment)
        write_jsonl(npath, neutral)
        trec = summarize(treatment); nrec = summarize(neutral)
        rec = {
            "requested_budget": budget,
            "treatment_path": str(tpath),
            "neutral_path": str(npath),
            "treatment": trec,
            "neutral": nrec,
            "word_difference_treatment_minus_neutral": trec["words"] - nrec["words"],
            "neutral_collection": scanned,
            "scientific_status": "future small-probe substrate only",
        }
        summary["budgets"][f"{budget//1000}k"] = rec
        for arm, s in [("treatment", trec), ("neutral", nrec)]:
            csv_rows.append({"budget": f"{budget//1000}k", "arm": arm, "sentences": s["sentences"], "words": s["words"], "mean_words": s.get("mean_words"), "source_words": json.dumps(s.get("source_words", {}), ensure_ascii=False)})
    csv_path = OUT / "transition_matched_control_fast_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader(); w.writerows(csv_rows)
    summary["summary_csv"] = str(csv_path)
    summary["elapsed_sec"] = round(time.time() - t0, 2)
    out_json = OUT / "transition_matched_control_fast_builder.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research — Fast transition matched controls\n\n",
        "Built low-transition neutral controls matched to the existing transition treatment slices from the same allowed reservoirs. No official evaluation text is used and no training is launched.\n\n",
        "| budget | treatment words | neutral words | difference | treatment sentences | neutral sentences |\n",
        "|---:|---:|---:|---:|---:|---:|\n",
    ]
    for b, rec in summary["budgets"].items():
        lines.append(f"| {b} | {rec['treatment']['words']} | {rec['neutral']['words']} | {rec['word_difference_treatment_minus_neutral']} | {rec['treatment']['sentences']} | {rec['neutral']['sentences']} |\n")
    lines += [
        "\nScientific use: these are matched-control substrate assets for a later low-cost shared-checkpoint factorial only if FW compact/breadth endpoints fail to move EWoK conditional-reversal failures and GlobalPIQA hard-row margins. The treatment still needs semantic filtering before any full 10M materialization.\n\n",
        f"Summary JSON: `{out_json}`\n",
        f"Summary CSV: `{csv_path}`\n",
    ]
    NOTE.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "budgets": {k: {"treatment_words": v["treatment"]["words"], "neutral_words": v["neutral"]["words"], "diff": v["word_difference_treatment_minus_neutral"]} for k, v in summary["budgets"].items()},
        "json": str(out_json),
        "csv": str(csv_path),
        "note": str(NOTE),
        "elapsed_sec": summary["elapsed_sec"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
