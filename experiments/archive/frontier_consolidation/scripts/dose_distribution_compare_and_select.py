#!/usr/bin/env python3
"""research: compare old and expansion compact-view pair distributions and select a matched MAX increment.

The dose experiment is only interpretable if increasing the restructured fraction
mostly changes dose, not the source population. The inherited 1x block came from
the medium FineWeb tier, while the new increment came from the broader
web-artifact-removed tier. This script measures that difference and builds a
largest practical increment whose coarse feature distribution follows the
inherited selected 1x block.
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import json
import math
import pathlib
import re
import statistics
from typing import Any, Iterable

ROOT = pathlib.Path("experiments/archive/frontier_consolidation")
OLD_SELECTED_DEFAULT = ROOT / "data/density_core_reinvestment_medium_riskhard/selected_compact_reinvest_pairs.jsonl"
MEDIUM_ACCEPTED_DEFAULT = ROOT / "data/medium_compact_analysis/medium_compact_ws_accepted_rewrites.jsonl"
EXPANSION_ACCEPTED_DEFAULT = ROOT / "data/expansion_analysis/expansion_accepted_rewrites.jsonl"
OUT_DIR_DEFAULT = ROOT / "data/dose_distribution_select"
BASE_CHANGED_BLOCK_WORDS = 423_520
MAX_PACKET_WORDS = 160

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[''][A-Za-z0-9]+)?|[£$€]\s*\d+(?:\.\d+)?|\d+(?:[.,:/\-]\d+)*(?:%|st|nd|rd|th)?")
NUM_RE = re.compile(r"(?:[£$€]\s*)?\d+(?:[.,:/\-]\d+)*(?:%|st|nd|rd|th)?", re.I)
STOPWORDS = {
    "the","a","an","and","or","but","if","then","of","to","in","on","for","with","as","by","from","at",
    "is","are","was","were","be","been","being","it","its","this","that","these","those","he","she","they","we",
    "you","i","his","her","their","our","your","not","no","so","than","into","about","can","could","may","might",
    "will","would","should","has","have","had","do","does","did","who","which","what","when","where","why","how",
    "one","two","new","old","more","most","many","much","some","such","very","only","also","between","during","after",
    "study","studies","research","show","shows","using","used","use","made","make","called","known",
}

@dataclasses.dataclass
class Pair:
    pair_id: str
    key: str
    source_text: str
    rewrite_text: str
    source_words: int
    rewrite_words: int
    pair_words: int
    sentence_id: str = ""
    doc_id: str = ""
    domain_hits: list[str] = dataclasses.field(default_factory=list)
    origin: str = ""
    content_recall: float | None = None
    content_overlap: float | None = None
    entity_recall: float | None = None
    number_recall: float | None = None
    soft_flags: list[str] = dataclasses.field(default_factory=list)
    source_risks: list[str] = dataclasses.field(default_factory=list)
    length_ratio: float = 0.0
    pair_content_density: float = 0.0
    rewrite_content_density: float = 0.0
    novel_content_fraction: float = 0.0


def wc(text: str) -> int:
    return len((text or "").split())


def lexical_tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(text or "")]


def content_tokens(text: str) -> list[str]:
    return [t for t in lexical_tokens(text) if len(t) > 2 and t not in STOPWORDS and not NUM_RE.fullmatch(t)]


def norm_key(d: dict[str, Any]) -> str:
    k = str(d.get("key") or "").strip()
    if k.startswith("sid:") and "|doc:" in k:
        return k
    sid = str(d.get("sentence_id") or "").strip()
    doc = str(d.get("doc_id") or "").strip()
    if sid or doc:
        return f"sid:{sid}|doc:{doc}"
    return k or str(d.get("pair_id") or d.get("prompt_id") or "")


def norm_pair_id(d: dict[str, Any], key: str, origin: str) -> str:
    pid = str(d.get("pair_id") or "").strip()
    if pid:
        return pid
    pr = str(d.get("prompt_id") or "").strip()
    if pr:
        return f"compact:{pr}"
    return f"compact:{origin}:{hashlib.sha1(key.encode('utf-8')).hexdigest()[:12]}"


def pair_from_dict(d: dict[str, Any], origin: str) -> Pair:
    src = " ".join(str(d.get("source_text") or "").split())
    rew = " ".join(str(d.get("rewrite_text") or "").split())
    sw = wc(src)
    rw = wc(rew)
    pw = sw + rw
    key = norm_key(d)
    pid = norm_pair_id(d, key, origin)
    src_cont = content_tokens(src)
    rew_cont = content_tokens(rew)
    src_set = set(src_cont)
    novel = [t for t in rew_cont if t not in src_set]
    p = Pair(
        pair_id=pid,
        key=key,
        source_text=src,
        rewrite_text=rew,
        source_words=sw,
        rewrite_words=rw,
        pair_words=pw,
        sentence_id=str(d.get("sentence_id") or ""),
        doc_id=str(d.get("doc_id") or ""),
        domain_hits=[str(x) for x in (d.get("domain_hits") or [])],
        origin=origin,
        content_recall=float(d["content_recall"]) if d.get("content_recall") is not None else None,
        content_overlap=float(d["content_overlap"]) if d.get("content_overlap") is not None else None,
        entity_recall=float(d["entity_recall"]) if d.get("entity_recall") is not None else None,
        number_recall=float(d["number_recall"]) if d.get("number_recall") is not None else None,
        soft_flags=[str(x) for x in (d.get("soft_flags") or [])],
        source_risks=[str(x) for x in (d.get("source_risks") or [])],
    )
    p.length_ratio = rw / max(1, sw)
    p.pair_content_density = (len(src_cont) + len(rew_cont)) / max(1, pw)
    p.rewrite_content_density = len(rew_cont) / max(1, rw)
    p.novel_content_fraction = len(novel) / max(1, len(rew_cont))
    return p


def read_pairs(path: pathlib.Path, origin: str) -> list[Pair]:
    rows: list[Pair] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            p = pair_from_dict(json.loads(line), origin)
            if p.source_text and p.rewrite_text and p.pair_words > 0 and p.pair_words <= MAX_PACKET_WORDS:
                rows.append(p)
    return rows


def stat(vals: list[float], weights: list[float] | None = None) -> dict[str, Any]:
    if not vals:
        return {"n": 0}
    xs = sorted(vals)
    if weights is None:
        wmean = statistics.fmean(vals)
    else:
        s = sum(weights)
        wmean = sum(v*w for v, w in zip(vals, weights)) / s if s else None
    return {
        "n": len(vals),
        "mean": statistics.fmean(vals),
        "weighted_mean": wmean,
        "median": statistics.median(xs),
        "q10": xs[int(0.10*(len(xs)-1))],
        "q25": xs[int(0.25*(len(xs)-1))],
        "q75": xs[int(0.75*(len(xs)-1))],
        "q90": xs[int(0.90*(len(xs)-1))],
        "min": xs[0],
        "max": xs[-1],
        "sum": sum(vals),
    }


def feature_values(pairs: list[Pair], name: str) -> list[float]:
    return [float(getattr(p, name)) for p in pairs]


def summarize(name: str, pairs: list[Pair]) -> dict[str, Any]:
    weights = [float(p.pair_words) for p in pairs]
    out: dict[str, Any] = {
        "name": name,
        "pairs": len(pairs),
        "pair_words": sum(p.pair_words for p in pairs),
        "source_words": sum(p.source_words for p in pairs),
        "rewrite_words": sum(p.rewrite_words for p in pairs),
        "unique_keys": len({p.key for p in pairs}),
        "unique_docs": len({p.doc_id for p in pairs}),
        "origin_pair_words": dict(collections.Counter({}).most_common()),
    }
    c = collections.Counter()
    for p in pairs:
        c[p.origin] += p.pair_words
    out["origin_pair_words"] = dict(c.most_common())
    for feat in ["source_words", "rewrite_words", "pair_words", "length_ratio", "pair_content_density", "rewrite_content_density", "novel_content_fraction"]:
        out[feat] = stat(feature_values(pairs, feat), weights)
    vals = [p.content_recall for p in pairs if p.content_recall is not None]
    out["content_recall"] = stat([float(x) for x in vals])
    vals = [p.content_overlap for p in pairs if p.content_overlap is not None]
    out["content_overlap"] = stat([float(x) for x in vals])
    out["domain_hit_counts"] = dict(collections.Counter(d for p in pairs for d in (p.domain_hits or ["no_domain"])).most_common(20))
    return out


def std_old(pairs: list[Pair], feat: str) -> float:
    vals = feature_values(pairs, feat)
    if len(vals) < 2:
        return 1.0
    sd = statistics.pstdev(vals)
    return sd if sd > 1e-9 else 1.0


def divergence(old: list[Pair], other: list[Pair]) -> dict[str, Any]:
    out = {}
    for feat in ["source_words", "rewrite_words", "pair_words", "length_ratio", "pair_content_density", "rewrite_content_density", "novel_content_fraction"]:
        ov = feature_values(old, feat)
        xv = feature_values(other, feat)
        if not xv:
            continue
        sd = std_old(old, feat)
        out[feat] = {
            "old_mean": statistics.fmean(ov),
            "other_mean": statistics.fmean(xv),
            "mean_shift_in_old_sd": (statistics.fmean(xv) - statistics.fmean(ov)) / sd,
            "old_median": statistics.median(sorted(ov)),
            "other_median": statistics.median(sorted(xv)),
        }
    return out


def quantile_edges(vals: list[float], probs: list[float]) -> list[float]:
    xs = sorted(vals)
    return [xs[int(p*(len(xs)-1))] for p in probs]


def bin_index(x: float, edges: list[float]) -> int:
    for i, e in enumerate(edges):
        if x <= e:
            return i
    return len(edges)


def make_bin(p: Pair, edges: dict[str, list[float]]) -> tuple[int, int, int, int]:
    return (
        bin_index(p.source_words, edges["source_words"]),
        bin_index(p.length_ratio, edges["length_ratio"]),
        bin_index(p.pair_content_density, edges["pair_content_density"]),
        bin_index(p.novel_content_fraction, edges["novel_content_fraction"]),
    )


def pair_distance(p: Pair, old: list[Pair], means: dict[str, float], sds: dict[str, float]) -> float:
    feats = ["source_words", "length_ratio", "pair_content_density", "novel_content_fraction", "content_recall"]
    total = 0.0
    for feat in feats:
        if feat == "content_recall":
            val = p.content_recall if p.content_recall is not None else means.get(feat, 0.0)
        else:
            val = float(getattr(p, feat))
        total += ((float(val) - means[feat]) / max(1e-9, sds[feat])) ** 2
    return math.sqrt(total / len(feats))


def select_matched_increment(old: list[Pair], candidates: list[Pair], bin_slack: float) -> tuple[list[Pair], dict[str, Any]]:
    edges = {
        "source_words": quantile_edges(feature_values(old, "source_words"), [0.20, 0.40, 0.60, 0.80]),
        "length_ratio": quantile_edges(feature_values(old, "length_ratio"), [0.25, 0.50, 0.75]),
        "pair_content_density": quantile_edges(feature_values(old, "pair_content_density"), [0.33, 0.66]),
        "novel_content_fraction": quantile_edges(feature_values(old, "novel_content_fraction"), [0.33, 0.66]),
    }
    old_bin_words: collections.Counter[tuple[int, int, int, int]] = collections.Counter()
    for p in old:
        old_bin_words[make_bin(p, edges)] += p.pair_words
    cand_bins: dict[tuple[int, int, int, int], list[Pair]] = collections.defaultdict(list)
    means = {
        "source_words": statistics.fmean(feature_values(old, "source_words")),
        "length_ratio": statistics.fmean(feature_values(old, "length_ratio")),
        "pair_content_density": statistics.fmean(feature_values(old, "pair_content_density")),
        "novel_content_fraction": statistics.fmean(feature_values(old, "novel_content_fraction")),
        "content_recall": statistics.fmean([p.content_recall for p in old if p.content_recall is not None]),
    }
    sds = {
        "source_words": std_old(old, "source_words"),
        "length_ratio": std_old(old, "length_ratio"),
        "pair_content_density": std_old(old, "pair_content_density"),
        "novel_content_fraction": std_old(old, "novel_content_fraction"),
        "content_recall": statistics.pstdev([float(p.content_recall) for p in old if p.content_recall is not None]) or 1.0,
    }
    for p in candidates:
        cand_bins[make_bin(p, edges)].append(p)
    for b in cand_bins:
        cand_bins[b].sort(key=lambda p: (pair_distance(p, old, means, sds), p.pair_id))

    candidate_total = sum(p.pair_words for p in candidates)
    old_total = sum(p.pair_words for p in old)
    selected: list[Pair] = []
    selected_bin_words: collections.Counter[tuple[int, int, int, int]] = collections.Counter()

    # Word quota per old bin at the candidate-total increment, with slack to avoid
    # throwing away many near-matched rows because of coarse bin edges.
    for b, rows in sorted(cand_bins.items(), key=lambda kv: (-old_bin_words.get(kv[0], 0), kv[0])):
        target = (old_bin_words.get(b, 0) / old_total) * candidate_total if old_total else 0.0
        cap = max(0, int(round(target * bin_slack)))
        used = 0
        for p in rows:
            if used + p.pair_words <= cap:
                selected.append(p)
                used += p.pair_words
        selected_bin_words[b] = used

    # A second pass fills with globally close candidates while keeping each bin no
    # more than 1.5x its old-proportional share. This makes MAX as large as possible
    # without letting the shorter WAR tail dominate.
    chosen = {p.key for p in selected}
    loose_cap = {}
    for b in set(list(old_bin_words) + list(cand_bins)):
        target = (old_bin_words.get(b, 0) / old_total) * candidate_total if old_total else 0.0
        loose_cap[b] = max(int(round(target * max(1.5, bin_slack))), selected_bin_words.get(b, 0))
    unused = [p for p in candidates if p.key not in chosen]
    unused.sort(key=lambda p: (pair_distance(p, old, means, sds), p.pair_id))
    for p in unused:
        b = make_bin(p, edges)
        if selected_bin_words[b] + p.pair_words <= loose_cap.get(b, 0):
            selected.append(p)
            selected_bin_words[b] += p.pair_words
            chosen.add(p.key)

    meta = {
        "edges": edges,
        "bin_slack": bin_slack,
        "old_bin_words": {str(k): v for k, v in old_bin_words.items()},
        "candidate_bin_words": {str(k): sum(p.pair_words for p in rows) for k, rows in cand_bins.items()},
        "selected_bin_words": {str(k): v for k, v in selected_bin_words.items()},
        "candidate_increment_pair_words": candidate_total,
        "selected_increment_pair_words": sum(p.pair_words for p in selected),
        "selected_increment_fraction_of_candidates": sum(p.pair_words for p in selected) / max(1, candidate_total),
    }
    return selected, meta


def write_pairs(path: pathlib.Path, pairs: Iterable[Pair]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for p in pairs:
            rec = dataclasses.asdict(p)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-selected", default=str(OLD_SELECTED_DEFAULT))
    ap.add_argument("--medium-accepted", default=str(MEDIUM_ACCEPTED_DEFAULT))
    ap.add_argument("--expansion-accepted", default=str(EXPANSION_ACCEPTED_DEFAULT))
    ap.add_argument("--out-dir", default=str(OUT_DIR_DEFAULT))
    ap.add_argument("--bin-slack", type=float, default=1.12)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    old = read_pairs(pathlib.Path(args.old_selected), "old_selected_1x_medium")
    medium_all = read_pairs(pathlib.Path(args.medium_accepted), "medium_unused_accepted")
    expansion = read_pairs(pathlib.Path(args.expansion_accepted), "war_expansion_accepted")
    old_keys = {p.key for p in old}
    old_ids = {p.pair_id for p in old}
    medium_remaining = [p for p in medium_all if p.key not in old_keys]
    candidates = medium_remaining + expansion

    selected_increment, select_meta = select_matched_increment(old, candidates, args.bin_slack)
    selected_max = old + selected_increment

    all_increment = candidates
    all_max = old + all_increment

    selected_path = out_dir / "selected_matched_max_pairs.jsonl"
    increment_path = out_dir / "selected_matched_increment_pairs.jsonl"
    write_pairs(selected_path, selected_max)
    write_pairs(increment_path, selected_increment)

    groups = {
        "old_selected_1x": old,
        "medium_all_accepted": medium_all,
        "medium_remaining_after_1x": medium_remaining,
        "war_expansion_accepted": expansion,
        "all_available_increment": all_increment,
        "all_available_max_old_plus_increment": all_max,
        "matched_increment": selected_increment,
        "matched_max_old_plus_increment": selected_max,
    }
    summaries = {name: summarize(name, pairs) for name, pairs in groups.items()}
    divergences = {name: divergence(old, pairs) for name, pairs in groups.items() if name != "old_selected_1x"}

    meta = {
        "status": "DOSE_DISTRIBUTION_COMPARED_AND_MATCHED_MAX_SELECTED",
        "scientific_purpose": "Ensure that the MAX dose changes the fraction of budget restructured into source+compact packets without letting the web-artifact-removed increment become a hidden source-population change.",
        "inputs": {
            "old_selected": str(args.old_selected),
            "medium_accepted": str(args.medium_accepted),
            "expansion_accepted": str(args.expansion_accepted),
        },
        "base_changed_block_words": BASE_CHANGED_BLOCK_WORDS,
        "old_keys_in_medium_all": len(old_keys & {p.key for p in medium_all}),
        "old_pair_ids_in_medium_all": len(old_ids & {p.pair_id for p in medium_all}),
        "groups": summaries,
        "divergence_vs_old_selected": divergences,
        "selection": select_meta,
        "selected_files": {
            "matched_increment": str(increment_path),
            "matched_max": str(selected_path),
        },
        "matched_max_pair_words": sum(p.pair_words for p in selected_max),
        "matched_max_dose_multiple": sum(p.pair_words for p in selected_max) / BASE_CHANGED_BLOCK_WORDS,
        "matched_increment_pair_words": sum(p.pair_words for p in selected_increment),
        "matched_increment_origin_pair_words": summaries["matched_increment"]["origin_pair_words"],
        "all_available_max_pair_words": sum(p.pair_words for p in all_max),
        "all_available_max_dose_multiple": sum(p.pair_words for p in all_max) / BASE_CHANGED_BLOCK_WORDS,
        "selected_max_sha256": None,
        "old_selected_is_inner_prefix": [p.key for p in selected_max[:len(old)]] == [p.key for p in old],
        "interpretation_note": "The tokenizer must remain the research compliant16k_reinvest10M tokenizer for all dose arms; MAX arms are mechanism instruments, not new end-to-end legal submissions, because their documented text spans the old tokenizer-fitting pool plus additional generated FineWeb views.",
    }
    meta["selected_max_sha256"] = sha256_file(selected_path)
    meta_path = out_dir / "dose_distribution_comparison_and_selection.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# research dose distribution comparison and matched MAX selection",
        "",
        "The inherited 1x block is preserved as the inner set. The increment is selected from unused medium accepted pairs plus the web-artifact-removed expansion so that source length, compression ratio, content density, and novel rewrite content stay close to the inherited block.",
        "",
        f"Old 1x: {summaries['old_selected_1x']['pairs']} pairs, {summaries['old_selected_1x']['pair_words']} pair words.",
        f"All available MAX: {summaries['all_available_max_old_plus_increment']['pairs']} pairs, {summaries['all_available_max_old_plus_increment']['pair_words']} pair words, dose {meta['all_available_max_dose_multiple']:.3f}x.",
        f"Matched MAX: {summaries['matched_max_old_plus_increment']['pairs']} pairs, {summaries['matched_max_old_plus_increment']['pair_words']} pair words, dose {meta['matched_max_dose_multiple']:.3f}x.",
        f"Matched increment origins: {json.dumps(meta['matched_increment_origin_pair_words'], ensure_ascii=False)}.",
        "",
        "## Mean shifts vs inherited 1x selected block (in old-block standard deviations)",
    ]
    for group in ["war_expansion_accepted", "all_available_increment", "matched_increment", "matched_max_old_plus_increment"]:
        lines.append(f"### {group}")
        for feat, rec in divergences[group].items():
            lines.append(f"- {feat}: {rec['mean_shift_in_old_sd']:+.3f} (old {rec['old_mean']:.4f}, group {rec['other_mean']:.4f})")
    lines += ["", f"JSON: `{meta_path}`", f"Selected MAX pairs: `{selected_path}`"]
    (out_dir / "dose_distribution_comparison_and_selection.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": meta["status"],
        "old_keys_in_medium_all": meta["old_keys_in_medium_all"],
        "all_available_max_dose_multiple": meta["all_available_max_dose_multiple"],
        "matched_max_dose_multiple": meta["matched_max_dose_multiple"],
        "matched_max_pair_words": meta["matched_max_pair_words"],
        "matched_increment_origin_pair_words": meta["matched_increment_origin_pair_words"],
        "old_selected_is_inner_prefix": meta["old_selected_is_inner_prefix"],
        "meta_path": str(meta_path),
        "selected_max_pairs": str(selected_path),
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
