#!/usr/bin/env python3
"""Find natural entity-relation evidence clusters in official BabyLM training text.

This script is a metadata-only pre-training asset. It reads only official Strict-Small
training text and extracts local sentence clusters that share an anchor string while
carrying complementary context words. It writes samples and statistics for deciding
whether a low-dose cluster materializer is worth building. It does not read official
AoA/CDI items, child curves, AoA scores, SuperGLUE labels, or downstream evaluation
outputs.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import collections
import hashlib
import json
import math
import pathlib
import random
import re
import statistics
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Tuple

ROOT = _public_path('experiments/archive/compact_experience')
RAW = pathlib.Path("experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/raw_dataset")
DEFAULT_OUT = _public_path('experiments/archive/compact_experience/data/natural_entity_relation_clusters')
TRAIN_FILES = [
    "simple_wiki.train.txt",
    "gutenberg.train.txt",
    "childes.train.txt",
    "bnc_spoken.train.txt",
    "open_subtitles.train.txt",
    "switchboard.train.txt",
]
TOTAL_WORDS = 10_000_000

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=(?:[\"'“‘(\[])?[A-Z0-9*])")
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[\u2019'][A-Za-z0-9]+)?|[£$€]\s*\d+(?:\.\d+)?|\d+(?:[.,:/-]\d+)*(?:%|st|nd|rd|th)?")
NUM_RE = re.compile(r"(?:[£$€]\s*)?\d+(?:[.,:/-]\d+)*(?:%|st|nd|rd|th)?", re.I)
STOP = {
    "the","a","an","and","or","but","if","then","than","that","this","these","those","there","here","where","when","while","who","whom","whose","which","what","why","how",
    "is","are","was","were","be","been","being","am","do","does","did","done","have","has","had","having","can","could","will","would","shall","should","may","might","must",
    "to","of","in","on","for","from","with","without","into","onto","by","as","at","about","over","under","after","before","between","through","during","within","around",
    "it","its","he","his","him","she","her","hers","they","their","them","we","our","us","you","your","i","me","my","mine","one","ones","someone","something","anything",
    "not","no","yes","so","very","just","only","also","more","most","many","much","some","all","any","both","each","every","other","same","new","old","good","great",
}
FIRST_WORD_NOT_ENTITY = {"The","A","An","And","But","Or","In","On","At","For","From","To","It","This","That","These","Those","There","Here","When","While","If","Then","He","She","They","We","You","I"}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def norm_ws(s: str) -> str:
    return " ".join(s.replace("\u00a0", " ").split())


def word_tokens(text: str) -> List[str]:
    return [m.group(0) for m in WORD_RE.finditer(text)]


def content_tokens(text: str) -> List[str]:
    out = []
    for w in word_tokens(text):
        t = w.strip("'\"“”‘’.,!?;:()[]{}").lower()
        if len(t) >= 3 and t not in STOP and not NUM_RE.fullmatch(t):
            out.append(t)
    return out


def anchors(text: str) -> List[str]:
    out: List[str] = []
    words = text.split()
    # Numbers are strong local anchors for comparative and world facts.
    for m in NUM_RE.finditer(text):
        out.append(re.sub(r"\s+", "", m.group(0).lower()))
    # Quoted titles and capitalized multiword phrases.
    for q in re.finditer(r"[\"“]([^\"”]{3,60})[\"”]", text):
        toks = content_tokens(q.group(1))
        if 1 <= len(toks) <= 6:
            out.append("quote:" + "_".join(toks[:6]))
    cap_run: List[str] = []
    def flush_run() -> None:
        nonlocal cap_run
        if len(cap_run) >= 2:
            out.append("cap:" + "_".join(x.lower() for x in cap_run[:6]))
        cap_run = []
    for i, raw in enumerate(words):
        w = raw.strip("'\"“”‘’.,!?;:()[]{}").replace("&amp;", "&")
        letters = re.sub(r"[^A-Za-z]", "", w)
        if not letters:
            flush_run(); continue
        is_cap = letters[0].isupper() and len(letters) > 1
        internal = any(c.isupper() for c in letters[1:])
        allcaps = len(letters) >= 2 and letters.isupper()
        if is_cap or internal or allcaps:
            if i == 0 and w in FIRST_WORD_NOT_ENTITY and not internal and not allcaps:
                flush_run(); continue
            low = w.lower()
            if low in STOP:
                flush_run(); continue
            cap_run.append(w)
        else:
            flush_run()
    flush_run()
    # Repeated salient content terms can anchor common-noun relation clusters.
    cnt = collections.Counter(content_tokens(text))
    for t, c in cnt.items():
        if c >= 2 and len(t) >= 5:
            out.append("rep:" + t)
    seen = set(); dedup = []
    for a in out:
        if a not in seen:
            seen.add(a); dedup.append(a)
    return dedup


def split_sentences(line: str) -> List[str]:
    s = norm_ws(line)
    if not s:
        return []
    if s.startswith("=") and s.endswith("="):
        return []
    # CHILDES/BNC speaker tags remain useful as local utterance text after tag removal.
    s = re.sub(r"^\*[A-Z]+:\s*", "", s)
    s = re.sub(r"^\[[^\]]+\]\s*", "", s)
    parts = SENT_SPLIT.split(s)
    out = []
    for p in parts:
        p = norm_ws(p.strip())
        if not p:
            continue
        wc = len(p.split())
        if 5 <= wc <= 80:
            out.append(p)
    return out


@dataclass
class Sent:
    sid: int
    source: str
    block_id: str
    local_i: int
    text: str
    words: int
    content: List[str]
    anchors: List[str]


@dataclass
class Cluster:
    cluster_id: str
    source: str
    block_id: str
    anchor: str
    sentence_ids: List[int]
    heldout_sentence_id: int | None
    words: int
    pairwise_content_jaccard_mean: float
    unique_content_words: int
    anchor_count_in_block: int
    texts: List[str]
    heldout_text: str | None


def source_blocks(path: pathlib.Path, max_lines: int | None) -> Iterable[Tuple[str, List[str]]]:
    name = path.name
    block: List[str] = []
    block_id = None
    lines_seen = 0
    chunk_size = 48 if name in {"gutenberg.train.txt", "bnc_spoken.train.txt", "open_subtitles.train.txt", "switchboard.train.txt"} else 80
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            lines_seen += 1
            if max_lines is not None and lines_seen > max_lines:
                break
            s = line.rstrip("\n")
            if name == "simple_wiki.train.txt" and s.startswith("= = ="):
                if block:
                    yield (block_id or f"{name}:before_{line_no}", block)
                block_id = f"{name}:heading_{line_no}:" + norm_ws(s).strip("= ")[:80]
                block = []
                continue
            block.append(s)
            if name != "simple_wiki.train.txt" and len(block) >= chunk_size:
                yield (f"{name}:lines_{line_no-len(block)+1}_{line_no}", block)
                block = []
                block_id = None
    if block:
        yield (block_id or f"{name}:tail", block)


def read_sentences(raw: pathlib.Path, files: List[str], max_lines_per_file: int | None) -> List[Sent]:
    out: List[Sent] = []
    sid = 0
    for fn in files:
        path = raw / fn
        if not path.exists():
            raise FileNotFoundError(path)
        for bid, lines in source_blocks(path, max_lines_per_file):
            local = 0
            for line in lines:
                for sent in split_sentences(line):
                    a = anchors(sent)
                    c = content_tokens(sent)
                    if not a or len(c) < 3:
                        continue
                    out.append(Sent(sid=sid, source=fn, block_id=bid, local_i=local, text=sent, words=len(sent.split()), content=c, anchors=a))
                    sid += 1; local += 1
    return out


def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def find_clusters(sents: List[Sent], max_clusters: int, seed: int) -> List[Cluster]:
    by_block_anchor: Dict[Tuple[str, str, str], List[Sent]] = collections.defaultdict(list)
    for s in sents:
        for a in s.anchors:
            by_block_anchor[(s.source, s.block_id, a)].append(s)
    candidates: List[Cluster] = []
    for (source, block_id, a), xs in by_block_anchor.items():
        xs = sorted(xs, key=lambda z: z.local_i)
        if len(xs) < 2:
            continue
        # Pick two or three nearby-but-not-identical sentences. Prefer content diversity.
        best = None
        for i in range(len(xs)):
            for j in range(i + 1, min(len(xs), i + 6)):
                group = [xs[i], xs[j]]
                if j + 1 < len(xs) and xs[j + 1].local_i - xs[i].local_i <= 12:
                    group.append(xs[j + 1])
                if len({g.text for g in group}) < len(group):
                    continue
                total_words = sum(g.words for g in group)
                if not (18 <= total_words <= 140):
                    continue
                pair_js = []
                for u in range(len(group)):
                    for v in range(u + 1, len(group)):
                        pair_js.append(jaccard(group[u].content, group[v].content))
                mean_j = statistics.mean(pair_js) if pair_js else 1.0
                uniq = len(set(t for g in group for t in g.content))
                # Avoid near duplicates but require some lexical connection beyond the anchor.
                if mean_j > 0.72 or mean_j < 0.02:
                    continue
                score = uniq - 20.0 * mean_j + min(len(group), 3)
                if best is None or score > best[0]:
                    best = (score, group, mean_j, uniq)
        if best is None:
            continue
        _, group, mean_j, uniq = best
        group_ids = {g.sid for g in group}
        held = next((z for z in xs if z.sid not in group_ids and abs(z.local_i - group[-1].local_i) <= 16), None)
        cid_base = f"{source}|{block_id}|{a}|" + ",".join(str(g.sid) for g in group)
        cid = hashlib.sha1(cid_base.encode("utf-8")).hexdigest()[:16]
        candidates.append(Cluster(
            cluster_id=cid,
            source=source,
            block_id=block_id,
            anchor=a,
            sentence_ids=[g.sid for g in group],
            heldout_sentence_id=held.sid if held else None,
            words=sum(g.words for g in group),
            pairwise_content_jaccard_mean=float(mean_j),
            unique_content_words=int(uniq),
            anchor_count_in_block=len(xs),
            texts=[g.text for g in group],
            heldout_text=held.text if held else None,
        ))
    rng = random.Random(seed)
    rng.shuffle(candidates)
    # Rank after a shuffle to avoid deterministic source prefix dominance among ties.
    candidates.sort(key=lambda c: (c.heldout_sentence_id is not None, c.unique_content_words, -c.pairwise_content_jaccard_mean, -c.words), reverse=True)
    return candidates[:max_clusters]


def summarize_clusters(clusters: List[Cluster], sents: List[Sent], args: argparse.Namespace) -> Dict[str, Any]:
    source_counts = collections.Counter(c.source for c in clusters)
    anchor_prefix_counts = collections.Counter(c.anchor.split(":", 1)[0] if ":" in c.anchor else "num" for c in clusters)
    words = [c.words for c in clusters]
    held = sum(1 for c in clusters if c.heldout_sentence_id is not None)
    estimated_2pct_clusters = 0
    total = 0
    for c in clusters:
        if total + c.words > int(TOTAL_WORDS * 0.02):
            break
        total += c.words; estimated_2pct_clusters += 1
    estimated_4pct_clusters = 0
    total4 = 0
    for c in clusters:
        if total4 + c.words > int(TOTAL_WORDS * 0.04):
            break
        total4 += c.words; estimated_4pct_clusters += 1
    return {
        "status": "NATURAL_ENTITY_RELATION_CLUSTER_PREFLIGHT",
        "created_utc": now(),
        "non_leakage_statement": "Reads only official Strict-Small training text. No official AoA/CDI words, child curves, AoA outputs, SuperGLUE labels, or downstream evaluation results are read or used.",
        "raw_dataset": str(args.raw_dataset),
        "files": args.files,
        "max_lines_per_file": args.max_lines_per_file,
        "sentences_with_anchors": len(sents),
        "clusters_found": len(clusters),
        "source_counts": dict(source_counts.most_common()),
        "anchor_prefix_counts": dict(anchor_prefix_counts.most_common()),
        "cluster_words": {
            "mean": statistics.mean(words) if words else None,
            "median": statistics.median(words) if words else None,
            "min": min(words) if words else None,
            "max": max(words) if words else None,
        },
        "heldout_third_sentence_clusters": held,
        "estimated_low_dose_counts": {
            "clusters_for_2pct_words": estimated_2pct_clusters,
            "words_for_2pct_prefix": sum(c.words for c in clusters[:estimated_2pct_clusters]),
            "clusters_for_4pct_words": estimated_4pct_clusters,
            "words_for_4pct_prefix": sum(c.words for c in clusters[:estimated_4pct_clusters]),
        },
        "intended_controls": {
            "true_cluster": "same-block shared-anchor complementary official sentences packed into one local row",
            "anchor_only_shuffle": "keep source, length and anchor frequency but replace one sentence with a same-anchor sentence from another block when possible",
            "original_repeat": "repeat one anchor sentence or an equal-length excerpt to match anchor/mask exposure",
            "untouched_order": "same sentence inventory left in its original corpus positions",
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dataset", default=str(RAW))
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT))
    ap.add_argument("--files", nargs="*", default=TRAIN_FILES)
    ap.add_argument("--max_lines_per_file", type=int, default=None)
    ap.add_argument("--max_clusters", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42042)
    args = ap.parse_args()
    raw = pathlib.Path(args.raw_dataset)
    sents = read_sentences(raw, args.files, args.max_lines_per_file)
    clusters = find_clusters(sents, args.max_clusters, args.seed)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_clusters(clusters, sents, args)
    (out_dir / "cluster_preflight_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (out_dir / "clusters_sample.jsonl").open("w", encoding="utf-8") as f:
        for c in clusters[: min(len(clusters), args.max_clusters)]:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    note = out_dir / "cluster_preflight_summary.md"
    lines = [
        "# research natural entity-relation cluster preflight",
        "",
        summary["non_leakage_statement"],
        "",
        f"Sentences with anchors: {summary['sentences_with_anchors']}",
        f"Clusters found: {summary['clusters_found']}",
        f"Clusters with held-out same-block sentence: {summary['heldout_third_sentence_clusters']}",
        f"2% prefix: {summary['estimated_low_dose_counts']['clusters_for_2pct_words']} clusters / {summary['estimated_low_dose_counts']['words_for_2pct_prefix']} words",
        f"4% prefix: {summary['estimated_low_dose_counts']['clusters_for_4pct_words']} clusters / {summary['estimated_low_dose_counts']['words_for_4pct_prefix']} words",
        "",
        "## Top source counts",
    ]
    for k, v in list(summary["source_counts"].items())[:10]:
        lines.append(f"- {k}: {v}")
    lines += ["", "## First samples"]
    for c in clusters[:5]:
        lines.append(f"- `{c.cluster_id}` {c.source} anchor={c.anchor} words={c.words} heldout={c.heldout_sentence_id is not None}")
        for t in c.texts:
            lines.append(f"  - {t}")
    note.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(out_dir / "cluster_preflight_summary.json"), "sample": str(out_dir / "clusters_sample.jsonl"), "note": str(note), "clusters_found": len(clusters)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
