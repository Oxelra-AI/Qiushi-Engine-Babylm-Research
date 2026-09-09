#!/usr/bin/env python3
"""research: corpus-scale structure census for post-fast-path route comparison.

This is a cheap, no-training audit over the current legal compact-view 10M corpus.
It estimates whether the corpus already contains enough evaluation-independent
structure for three orthogonal routes:
  1. learning experience: procedural / ordered action-outcome passages;
  2. entity-event-time computation: entity/state/update cues in multi-sentence rows;
  3. cross-context signal: contrast / alternative / same-row source+compact pair opportunities.

The counts are regex-based route scouts, not training data and not claims of clean labels.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT = Path(".")
DEFAULT_CORPUS = Path("experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl")
DEFAULT_META = Path("experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_changed_block_rows_meta.jsonl")
DEFAULT_OUT = Path("experiments/archive/representation_and_objectives/data/corpus_structure_census")

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")
WORD_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)?|\d+(?:[.,]\d+)?")

PATTERNS = {
    "procedural_order": re.compile(r"\b(first|second|third|then|next|after|before|finally|once|when|while|until|during|step|steps|procedure|process|instructions?|recipe|method|guide|how to)\b", re.I),
    "temporal_update": re.compile(r"\b(after|before|then|next|finally|once|when|while|until|during|later|earlier|subsequently|initially|eventually|meanwhile)\b", re.I),
    "state_change": re.compile(r"\b(becomes?|became|become|turns?|turned|changes?|changed|converts?|converted|transforms?|transformed|forms?|formed|creates?|created|produces?|produced|removes?|removed|destroys?|destroyed|opens?|opened|closes?|closed|fills?|filled|empties?|emptied|melts?|melted|freezes?|frozen|cools?|cooled|heats?|heated|dries?|dried|breaks?|broken|dissolves?|dissolved|absorbs?|absorbed|releases?|released|moves?|moved|places?|placed|puts?|put|adds?|added|mixes?|mixed|separates?|separated)\b", re.I),
    "explicit_result": re.compile(r"\b(causes?|caused|leads? to|led to|results? in|resulting in|therefore|so that|as a result|because of|due to|from\s+\w+\s+to\s+\w+)\b", re.I),
    "contrast_alternative": re.compile(r"\b(but|however|whereas|while|instead|rather than|unlike|compared with|compared to|in contrast|on the other hand|if|otherwise|unless|although|though|despite)\b", re.I),
    "entity_ref": re.compile(r"\b(it|its|they|their|them|this|that|these|those|former|latter|same|another|other)\b", re.I),
    "containment_location": re.compile(r"\b(in|inside|into|onto|on|under|over|above|below|near|from|to|between|within|through|across|around|container|box|bag|room|house|body|system|region|area|surface|layer)\b", re.I),
    "physical_affordance": re.compile(r"\b(rigid|flexible|liquid|solid|gas|hot|cold|wet|dry|heavy|light|empty|full|open|closed|strong|weak|hard|soft|rough|smooth|clean|dirty|safe|dangerous|fragile|stable|unstable)\b", re.I),
}

# route-specific conjunctions; each route needs several weak signals together.
def row_flags(text: str, words: int) -> Dict[str, bool]:
    hits = {name: bool(pat.search(text)) for name, pat in PATTERNS.items()}
    sents = [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]
    nsent = len(sents)
    # Many compact-view rows concatenate original/simplified sentence pairs. Repeated concepts plus multiple sentence boundaries are important.
    learning_experience = (
        nsent >= 3
        and hits["procedural_order"]
        and (hits["state_change"] or hits["explicit_result"])
    )
    event_state_computation = (
        nsent >= 2
        and hits["state_change"]
        and hits["entity_ref"]
        and (hits["temporal_update"] or hits["containment_location"])
    )
    cross_context_signal = (
        nsent >= 2
        and hits["contrast_alternative"]
        and (hits["physical_affordance"] or hits["explicit_result"] or hits["state_change"])
    )
    dense_binding_candidate = learning_experience and event_state_computation and cross_context_signal
    return {
        **hits,
        "learning_experience_candidate": learning_experience,
        "event_state_computation_candidate": event_state_computation,
        "cross_context_signal_candidate": cross_context_signal,
        "dense_binding_candidate": dense_binding_candidate,
        "multi_sentence": nsent >= 2,
        "long_context": words >= 80,
    }


def compact_word_count(text: str) -> int:
    # The corpus includes an explicit words field. This backup is used only if absent.
    return len(WORD_RE.findall(text))


def top_component_sources(meta_path: Path) -> Dict[str, int]:
    out = collections.Counter()
    if not meta_path.exists():
        return {}
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            for k, v in obj.get("component_sources", {}).items():
                try:
                    out[k] += int(v)
                except Exception:
                    pass
    return dict(out.most_common())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    ap.add_argument("--meta", default=str(DEFAULT_META))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--max-samples", type=int, default=12)
    args = ap.parse_args()

    corpus_path = Path(args.corpus)
    meta_path = Path(args.meta)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = collections.Counter()
    word_counts = collections.Counter()
    samples: Dict[str, List[dict]] = collections.defaultdict(list)
    route_cooccur = collections.Counter()
    per_10k_bins = []
    sha = hashlib.sha256()
    total_rows = 0
    total_words = 0
    sentence_hist = collections.Counter()
    words_hist = collections.Counter()

    with corpus_path.open("rb") as raw:
        for bline in raw:
            sha.update(bline)
            total_rows += 1
            try:
                obj = json.loads(bline.decode("utf-8"))
            except Exception as e:
                counts["json_decode_error"] += 1
                if len(samples["json_decode_error"]) < args.max_samples:
                    samples["json_decode_error"].append({"row": total_rows - 1, "error": repr(e)})
                continue
            text = obj.get("text", "")
            words = int(obj.get("words") or compact_word_count(text))
            total_words += words
            sents = [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]
            sentence_hist[min(len(sents), 20)] += 1
            words_hist[min((words // 20) * 20, 300)] += 1
            flags = row_flags(text, words)
            active_routes = []
            for name, val in flags.items():
                if val:
                    counts[name] += 1
                    word_counts[name] += words
                    if name.endswith("_candidate") and len(samples[name]) < args.max_samples:
                        samples[name].append({
                            "row_index": total_rows - 1,
                            "example_id": obj.get("example_id"),
                            "words": words,
                            "source": obj.get("source"),
                            "text_preview": text[:700],
                        })
                    if name.endswith("_candidate"):
                        active_routes.append(name)
            for i, a in enumerate(active_routes):
                for b in active_routes[i+1:]:
                    route_cooccur[tuple(sorted((a, b)))] += 1
            if total_rows % 10000 == 0:
                per_10k_bins.append({"rows_seen": total_rows, "words_seen": total_words, **{k: counts[k] for k in ["learning_experience_candidate", "event_state_computation_candidate", "cross_context_signal_candidate", "dense_binding_candidate"]}})

    route_names = ["learning_experience_candidate", "event_state_computation_candidate", "cross_context_signal_candidate", "dense_binding_candidate"]
    pattern_names = list(PATTERNS.keys())
    summary = {
        "status": "CORPUS_STRUCTURE_CENSUS",
        "corpus": str(corpus_path),
        "meta": str(meta_path),
        "corpus_sha256": sha.hexdigest(),
        "total_rows": total_rows,
        "total_words": total_words,
        "counts": dict(counts),
        "word_counts": dict(word_counts),
        "route_rates": {k: {"rows": counts[k], "row_frac": counts[k] / total_rows if total_rows else None, "words": word_counts[k], "word_frac": word_counts[k] / total_words if total_words else None} for k in route_names},
        "pattern_rates": {k: {"rows": counts[k], "row_frac": counts[k] / total_rows if total_rows else None, "words": word_counts[k], "word_frac": word_counts[k] / total_words if total_words else None} for k in pattern_names},
        "route_cooccurrence": {" & ".join(k): v for k, v in route_cooccur.items()},
        "sentence_hist_capped20": dict(sorted(sentence_hist.items())),
        "words_hist_bin20_capped300": dict(sorted(words_hist.items())),
        "component_sources_word_counts": top_component_sources(meta_path),
        "per_10k_bins": per_10k_bins,
        "samples": samples,
        "interpretation": {
            "caution": "Regex census only estimates source-structure availability; it does not certify clean labels or target alternatives.",
            "use": "If a route cannot find many candidates here or in a cleaner extractor, it should not receive H100 training. If counts are ample, the next step is a no-training model-behavior screen on existing checkpoints before training.",
        },
    }
    out_json = out_dir / "corpus_structure_census.json"
    out_md = out_dir / "corpus_structure_census.md"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = []
    lines.append("# research corpus-scale structure census")
    lines.append("")
    lines.append(f"Corpus: `{corpus_path}`")
    lines.append(f"SHA256: `{sha.hexdigest()}`")
    lines.append(f"Rows: `{total_rows}`; words: `{total_words}`")
    lines.append("")
    lines.append("## Route-scale weak-signal counts")
    lines.append("")
    lines.append("| route candidate | rows | row % | words | word % |")
    lines.append("|---|---:|---:|---:|---:|")
    for k in route_names:
        rate = summary["route_rates"][k]
        lines.append(f"| {k} | {rate['rows']} | {100*rate['row_frac']:.2f} | {rate['words']} | {100*rate['word_frac']:.2f} |")
    lines.append("")
    lines.append("## Primitive pattern counts")
    lines.append("")
    lines.append("| pattern | rows | row % | words | word % |")
    lines.append("|---|---:|---:|---:|---:|")
    for k in pattern_names:
        rate = summary["pattern_rates"][k]
        lines.append(f"| {k} | {rate['rows']} | {100*rate['row_frac']:.2f} | {rate['words']} | {100*rate['word_frac']:.2f} |")
    lines.append("")
    lines.append("## Component-source word counts in changed compact block")
    lines.append("")
    for k, v in list(summary["component_sources_word_counts"].items())[:12]:
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Samples")
    for k in route_names:
        lines.append("")
        lines.append(f"### {k}")
        for s in samples.get(k, [])[: args.max_samples]:
            preview = s["text_preview"].replace("\n", " ")
            lines.append(f"- row {s['row_index']} words={s['words']} id={s.get('example_id')}: {preview}")
    lines.append("")
    lines.append(f"JSON: `{out_json}`")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "out_json": str(out_json), "out_md": str(out_md), "total_rows": total_rows, "total_words": total_words, "route_rates": summary["route_rates"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
