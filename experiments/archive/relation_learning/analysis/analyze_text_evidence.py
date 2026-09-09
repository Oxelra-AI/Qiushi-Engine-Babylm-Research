#!/usr/bin/env python3
"""CPU-only audit of VIEW, REPEAT, and CLEAN changed-block text.

The analysis is deliberately lexical and transparent.  It reconstructs the
hash-rotated REPEAT continuation from the original materializer, validates it
against the materialized rows, and compares equal-length added segments as well
as complete changed blocks.  No model inference or GPU work is performed.
"""
from __future__ import annotations

import collections
import csv
import difflib
import hashlib
import json
import math
import pathlib
import random
import re
from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np

ROOT = pathlib.Path("/workspace")
PAIR_PATH = ROOT / "experiments/archive/frontier_consolidation/data/dose_distribution_select/selected_matched_max_pairs.jsonl"
POOL_DIR = ROOT / "experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools"
META_PATH = POOL_DIR / "dose2p64x_rowholdout_metadata.json"
VIEW_POOL = POOL_DIR / "compact_view_dose2p64x_10M.jsonl"
REPEAT_POOL = POOL_DIR / "compact_repeat_dose2p64x_10M.jsonl"
CLEAN_POOL = POOL_DIR / "cleanqwen_lengthmatched_dose2p64x_10M.jsonl"
VIEW_ROW_META = POOL_DIR / "compact_view_dose2p64x_changed_block_rows_meta.jsonl"
REPEAT_ROW_META = POOL_DIR / "compact_repeat_dose2p64x_changed_block_rows_meta.jsonl"
CLEAN_ROW_META = POOL_DIR / "cleanqwen_lengthmatched_dose2p64x_changed_block_rows_meta.jsonl"
EVAL_DIR = ROOT / "experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval"
OUT = ROOT / "experiments/archive/relation_learning/analysis/work"

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:[.,]\d+)*")
SENT_END_RE = re.compile(r"[.!?][\"')\]]*\s*$")
PRONOUNS = {
    "he", "she", "it", "they", "him", "her", "them", "his", "hers", "its",
    "their", "theirs", "this", "that", "these", "those", "which", "who", "whom",
}
STOP = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "of", "to", "in",
    "on", "at", "for", "from", "with", "by", "as", "is", "are", "was", "were", "be",
    "been", "being", "has", "have", "had", "do", "does", "did", "not", "no", "this",
    "that", "these", "those", "it", "its", "he", "she", "they", "them", "their", "we",
    "you", "i", "will", "would", "can", "could", "may", "might", "must", "should",
}

# Broad event classes.  These are regexes over lowercase alphabetic tokens, not
# parser labels.  Keeping the categories explicit makes false positives auditable.
CATEGORY_PATTERNS = {
    "movement": re.compile(
        r"^(?:move|moves|moved|moving|relocate|relocates|relocated|relocating|"
        r"migrate|migrates|migrated|migrating|travel|travels|traveled|travelled|traveling|travelling|"
        r"arrive|arrives|arrived|arriving|depart|departs|departed|departing|"
        r"enter|enters|entered|entering|leave|leaves|leaving|left|return|returns|returned|returning)$"
    ),
    "transfer": re.compile(
        r"^(?:give|gives|gave|given|giving|receive|receives|received|receiving|"
        r"transfer|transfers|transferred|transferring|send|sends|sent|sending|"
        r"bring|brings|brought|bringing|carry|carries|carried|carrying|"
        r"buy|buys|bought|buying|sell|sells|sold|selling|acquire|acquires|acquired|acquiring|"
        r"obtain|obtains|obtained|obtaining|donate|donates|donated|donating|"
        r"exchange|exchanges|exchanged|exchanging|hand|hands|handed|handing)$"
    ),
    "change": re.compile(
        r"^(?:become|becomes|became|becoming|change|changes|changed|changing|"
        r"transform|transforms|transformed|transforming|convert|converts|converted|converting|"
        r"increase|increases|increased|increasing|decrease|decreases|decreased|decreasing|"
        r"rise|rises|rose|risen|rising|fall|falls|fell|fallen|falling|"
        r"start|starts|started|starting|stop|stops|stopped|stopping|"
        r"begin|begins|began|begun|beginning|end|ends|ended|ending|"
        r"create|creates|created|creating|destroy|destroys|destroyed|destroying|"
        r"open|opens|opened|opening|close|closes|closed|closing|"
        r"develop|develops|developed|developing|break|breaks|broke|broken|breaking|"
        r"join|joins|joined|joining|split|splits|splitting|replace|replaces|replaced|replacing|"
        r"add|adds|added|adding|remove|removes|removed|removing)$"
    ),
    "containment": re.compile(
        r"^(?:contain|contains|contained|containing|include|includes|included|including|"
        r"hold|holds|held|holding|store|stores|stored|storing|keep|keeps|kept|keeping|"
        r"locate|locates|located|locating|place|places|placed|placing|put|puts|putting|"
        r"insert|inserts|inserted|inserting)$"
    ),
}
STATE_CATEGORIES = tuple(CATEGORY_PATTERNS)
PROPERTY_MARKERS = {
    "property_modal": {"can", "cannot"},
    "property_possession": {"has", "have", "had"},
    "property_copula": {"is", "are", "was", "were", "be", "been", "being"},
}
PREPOSITIONS = {"from", "to", "into", "in", "inside", "within", "out", "of", "by", "with"}
ENTITY_SIGNATURE = {"box", "contains", "contents", "move", "remove", "put", "from", "to", "into"}
ENTITY_PREDICATE_SIGNATURE = {"box", "contains", "contents", "move", "remove", "put"}


def lex_tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(text)]


def raw_words(text: str) -> list[str]:
    return text.split()


def ngrams(xs: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(xs[i:i+n]) for i in range(max(0, len(xs)-n+1))]


def safe_div(a: float, b: float) -> float:
    return a / b if b else float("nan")


def repeat_source_words(source_text: str, n_words: int, salt: str) -> str:
    toks = source_text.split()
    if not toks or n_words <= 0:
        return ""
    start = int(hashlib.sha1(salt.encode("utf-8")).hexdigest()[:8], 16) % len(toks)
    rot = toks[start:] + toks[:start]
    out: list[str] = []
    while len(out) < n_words:
        out.extend(rot[: n_words - len(out)])
    return " ".join(out)


def category_markers(text: str) -> dict[str, list[str]]:
    ts = lex_tokens(text)
    return {name: [t for t in ts if pat.match(t)] for name, pat in CATEGORY_PATTERNS.items()}


def generic_property_hits(text: str, comps_heads: set[str]) -> dict[str, int]:
    ts = lex_tokens(text)
    counts = collections.Counter(ts)
    out = {name: sum(counts[t] for t in vals) for name, vals in PROPERTY_MARKERS.items()}
    out["comps_head_any"] = sum(counts[t] for t in comps_heads)
    # Higher-precision multiword frames seen in COMPS properties.
    low = " " + " ".join(ts) + " "
    frames = [" made of ", " used for ", " used to ", " found in ", " lives in ", " live in ",
              " consists of ", " composed of ", " belongs to "]
    out["property_frame"] = sum(low.count(p) for p in frames)
    return out


def command_frames(text: str) -> dict[str, int]:
    # At most 18 tokens between command slots; this avoids paragraph-spanning matches.
    s = " ".join(lex_tokens(text))
    mid = r"(?:\s+\w+){0,18}?\s+"
    return {
        "entity_move_from_to": len(re.findall(r"\bmove" + mid + r"from" + mid + r"to\b", s)),
        "entity_remove_from": len(re.findall(r"\bremove" + mid + r"from\b", s)),
        "entity_put_into": len(re.findall(r"\bput" + mid + r"into\b", s)),
        "entity_contains": len(re.findall(r"\bcontains\b", s)),
    }


def passive_heuristic(text: str) -> bool:
    # Surface cue only: a BE auxiliary followed within two tokens by an -ed/-en form.
    s = " ".join(lex_tokens(text))
    return bool(re.search(r"\b(?:is|are|was|were|be|been|being)(?:\s+\w+){0,2}\s+\w+(?:ed|en)\b", s))


def anchor_tokens(text: str) -> list[str]:
    # Capitalized non-sentence-initial words, numbers, and mixed-case identifiers.
    raw = raw_words(text)
    anchors: list[str] = []
    for i, tok in enumerate(raw):
        clean = re.sub(r"^[^A-Za-z0-9$]+|[^A-Za-z0-9']+$", "", tok)
        if not clean:
            continue
        if re.search(r"\d", clean) or (i > 0 and clean[0].isupper()) or (any(c.islower() for c in clean) and any(c.isupper() for c in clean[1:])):
            anchors.append(clean.lower())
    return anchors


def content_tokens(text: str) -> list[str]:
    return [t for t in lex_tokens(text) if t not in STOP and len(t) > 1]


def overlap_recall(source: str, second: str, token_fn=content_tokens) -> float:
    src = collections.Counter(token_fn(source))
    dst = collections.Counter(token_fn(second))
    return safe_div(sum((src & dst).values()), sum(dst.values()))


def source_ngram_coverage(source: str, second: str, n: int) -> float:
    a = set(ngrams(lex_tokens(source), n))
    b = ngrams(lex_tokens(second), n)
    return safe_div(sum(g in a for g in b), len(b))


def shared_anchor_order_inversion(source: str, second: str) -> bool:
    a, b = anchor_tokens(source), anchor_tokens(second)
    ca, cb = collections.Counter(a), collections.Counter(b)
    shared = [x for x in a if ca[x] == 1 and cb[x] == 1]
    if len(shared) < 2:
        return False
    order_a = {x: i for i, x in enumerate(a)}
    order_b = {x: i for i, x in enumerate(b)}
    for i, x in enumerate(shared):
        for y in shared[i+1:]:
            if (order_a[x] - order_a[y]) * (order_b[x] - order_b[y]) < 0:
                return True
    return False


def sentence_complete(text: str) -> bool:
    return bool(SENT_END_RE.search(text.strip()))


def load_jsonl(path: pathlib.Path, limit: int | None = None) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
                if limit is not None and len(out) >= limit:
                    break
    return out


def write_csv(path: pathlib.Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if not rows:
        return
    fields = fields or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def official_signatures() -> tuple[set[str], dict]:
    comps_heads = collections.Counter()
    comps_props = set()
    comps_rows = 0
    with (EVAL_DIR / "comps/comps_base.jsonl").open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            toks = lex_tokens(d["property"])
            if toks:
                comps_heads[toks[0]] += 1
            comps_props.add(d["property"].lower())
            comps_rows += 1
    entity_counts = collections.Counter()
    entity_rows = 0
    by_depth = collections.Counter()
    for p in sorted((EVAL_DIR / "entity_tracking").glob("*.jsonl")):
        with p.open(encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                entity_rows += 1
                by_depth[int(d["numops"])] += 1
                entity_counts.update(t for t in lex_tokens(d["input_prefix"]) if t in ENTITY_SIGNATURE)
    info = {
        "comps_base_rows": comps_rows,
        "comps_unique_properties": len(comps_props),
        "comps_unique_head_tokens": len(comps_heads),
        "comps_head_counts": dict(comps_heads.most_common()),
        "entity_rows": entity_rows,
        "entity_rows_by_numops": dict(sorted(by_depth.items())),
        "entity_signature_counts": dict(entity_counts.most_common()),
    }
    return set(comps_heads), info


@dataclass
class PairMetric:
    pair_id: str
    origin: str
    domains: str
    source_words: int
    second_words: int
    view_content_recall: float
    repeat_content_recall: float
    supplied_content_recall: float
    supplied_entity_recall: float
    view_anchor_recall: float
    repeat_anchor_recall: float
    view_bigram_coverage: float
    repeat_bigram_coverage: float
    view_trigram_coverage: float
    repeat_trigram_coverage: float
    view_sequence_ratio: float
    repeat_sequence_ratio: float
    view_complete: int
    repeat_complete: int
    view_pronoun_shift: int
    repeat_pronoun_shift: int
    view_voice_shift: int
    repeat_voice_shift: int
    view_anchor_order_inversion: int
    repeat_anchor_order_inversion: int
    view_prep_shift: int
    repeat_prep_shift: int
    view_event_reexpression: int
    repeat_event_reexpression: int
    view_identity_variation: int
    repeat_identity_variation: int
    view_role_reexpression: int
    repeat_role_reexpression: int
    source_state_hits: int
    view_state_hits: int
    repeat_state_hits: int
    source_property_hits: int
    view_property_hits: int
    repeat_property_hits: int
    view_entity_signature_hits: int
    repeat_entity_signature_hits: int


def count_signals(text: str, comps_heads: set[str]) -> dict[str, int]:
    toks = lex_tokens(text)
    counts = collections.Counter(toks)
    cats = category_markers(text)
    props = generic_property_hits(text, comps_heads)
    frames = command_frames(text)
    out: dict[str, int] = {f"state_{k}": len(v) for k, v in cats.items()}
    out["state_total"] = sum(len(v) for v in cats.values())
    out.update(props)
    out["property_narrow"] = props["property_modal"] + props["property_possession"] + props["property_frame"]
    out["entity_signature"] = sum(counts[x] for x in ENTITY_SIGNATURE)
    out["entity_predicate_signature"] = sum(counts[x] for x in ENTITY_PREDICATE_SIGNATURE)
    out.update(frames)
    out["entity_command_total"] = sum(frames.values())
    out["pronoun"] = sum(counts[x] for x in PRONOUNS)
    return out


def pair_metrics(pair: dict, comps_heads: set[str]) -> tuple[PairMetric, str]:
    src = " ".join(pair["source_text"].split())
    view = " ".join(pair["rewrite_text"].split())
    repeat = repeat_source_words(src, len(view.split()), pair["pair_id"])
    src_cat, view_cat, rep_cat = category_markers(src), category_markers(view), category_markers(repeat)
    src_prop, view_prop, rep_prop = (count_signals(x, comps_heads) for x in (src, view, repeat))

    def changed_set(a: Iterable[str], b: Iterable[str]) -> bool:
        return set(a) != set(b)

    # A re-expression requires a same-category marker in both views and at least
    # one surface marker in the second view not present in the source.  Merely
    # truncating the source marker set (as REPEAT does) does not qualify.
    view_event_reexpr = any(src_cat[c] and view_cat[c] and (set(view_cat[c]) - set(src_cat[c])) for c in STATE_CATEGORIES)
    rep_event_reexpr = any(src_cat[c] and rep_cat[c] and (set(rep_cat[c]) - set(src_cat[c])) for c in STATE_CATEGORIES)
    src_pron = {t for t in lex_tokens(src) if t in PRONOUNS}
    src_prep = {t for t in lex_tokens(src) if t in PREPOSITIONS}
    view_pron_shift = src_pron != {t for t in lex_tokens(view) if t in PRONOUNS}
    rep_pron_shift = src_pron != {t for t in lex_tokens(repeat) if t in PRONOUNS}
    view_prep_shift = src_prep != {t for t in lex_tokens(view) if t in PREPOSITIONS}
    rep_prep_shift = src_prep != {t for t in lex_tokens(repeat) if t in PREPOSITIONS}
    view_voice_shift = passive_heuristic(src) != passive_heuristic(view)
    rep_voice_shift = passive_heuristic(src) != passive_heuristic(repeat)
    view_inv = shared_anchor_order_inversion(src, view)
    rep_inv = shared_anchor_order_inversion(src, repeat)
    supplied_entity = float(pair.get("entity_recall", float("nan")))
    supplied_content = float(pair.get("content_recall", float("nan")))
    va = overlap_recall(src, view, anchor_tokens)
    ra = overlap_recall(src, repeat, anchor_tokens)
    vb = source_ngram_coverage(src, view, 2)
    rb = source_ngram_coverage(src, repeat, 2)
    vt = source_ngram_coverage(src, view, 3)
    rt = source_ngram_coverage(src, repeat, 3)
    vcr = overlap_recall(src, view)
    rcr = overlap_recall(src, repeat)
    identity_v = (vcr >= .50 and (math.isnan(va) or va >= .80) and vb < .80)
    identity_r = (rcr >= .50 and (math.isnan(ra) or ra >= .80) and rb < .80)
    role_v = identity_v and (view_event_reexpr or view_pron_shift or view_voice_shift or view_inv or view_prep_shift)
    role_r = identity_r and (rep_event_reexpr or rep_pron_shift or rep_voice_shift or rep_inv or rep_prep_shift)
    m = PairMetric(
        pair_id=pair["pair_id"], origin=str(pair.get("origin", "")),
        domains="|".join(pair.get("domain_hits") or ["no_domain"]),
        source_words=len(src.split()), second_words=len(view.split()),
        view_content_recall=vcr, repeat_content_recall=rcr,
        supplied_content_recall=supplied_content, supplied_entity_recall=supplied_entity,
        view_anchor_recall=va, repeat_anchor_recall=ra,
        view_bigram_coverage=vb, repeat_bigram_coverage=rb,
        view_trigram_coverage=vt, repeat_trigram_coverage=rt,
        view_sequence_ratio=difflib.SequenceMatcher(None, lex_tokens(src), lex_tokens(view), autojunk=False).ratio(),
        repeat_sequence_ratio=difflib.SequenceMatcher(None, lex_tokens(src), lex_tokens(repeat), autojunk=False).ratio(),
        view_complete=int(sentence_complete(view)), repeat_complete=int(sentence_complete(repeat)),
        view_pronoun_shift=int(view_pron_shift), repeat_pronoun_shift=int(rep_pron_shift),
        view_voice_shift=int(view_voice_shift), repeat_voice_shift=int(rep_voice_shift),
        view_anchor_order_inversion=int(view_inv), repeat_anchor_order_inversion=int(rep_inv),
        view_prep_shift=int(view_prep_shift), repeat_prep_shift=int(rep_prep_shift),
        view_event_reexpression=int(view_event_reexpr), repeat_event_reexpression=int(rep_event_reexpr),
        view_identity_variation=int(identity_v), repeat_identity_variation=int(identity_r),
        view_role_reexpression=int(role_v), repeat_role_reexpression=int(role_r),
        source_state_hits=src_prop["state_total"], view_state_hits=view_prop["state_total"], repeat_state_hits=rep_prop["state_total"],
        source_property_hits=src_prop["property_narrow"], view_property_hits=view_prop["property_narrow"], repeat_property_hits=rep_prop["property_narrow"],
        view_entity_signature_hits=view_prop["entity_signature"], repeat_entity_signature_hits=rep_prop["entity_signature"],
    )
    return m, repeat


def aggregate_segment(name: str, texts: list[str], comps_heads: set[str], unit: str) -> tuple[dict, list[dict]]:
    totals = collections.Counter()
    word_total = 0
    complete = 0
    any_counts = collections.Counter()
    for text in texts:
        words = len(text.split())
        word_total += words
        sig = count_signals(text, comps_heads)
        totals.update(sig)
        complete += int(sentence_complete(text))
        any_counts.update(k for k, v in sig.items() if v > 0)
    summary = {"segment": name, "unit": unit, "units": len(texts), "words": word_total,
               "complete_units": complete, "complete_unit_pct": 100 * safe_div(complete, len(texts))}
    rows = []
    for signal in sorted(totals):
        rows.append({
            "segment": name, "unit": unit, "signal": signal, "count": totals[signal],
            "per_1000_words": 1000 * safe_div(totals[signal], word_total),
            "units_any": any_counts[signal], "units_any_pct": 100 * safe_div(any_counts[signal], len(texts)),
        })
    return summary, rows


def paired_bootstrap(metrics: list[PairMetric], seed: int = 90602, reps: int = 2000) -> list[dict]:
    rng = np.random.default_rng(seed)
    fields = [
        ("state_hits_per_1000_words", "view_state_hits", "repeat_state_hits", "second_words", 1000),
        ("property_hits_per_1000_words", "view_property_hits", "repeat_property_hits", "second_words", 1000),
        ("entity_signature_per_1000_words", "view_entity_signature_hits", "repeat_entity_signature_hits", "second_words", 1000),
        ("complete_segment_percentage_points", "view_complete", "repeat_complete", None, 100),
        ("identity_variation_percentage_points", "view_identity_variation", "repeat_identity_variation", None, 100),
        ("role_reexpression_percentage_points", "view_role_reexpression", "repeat_role_reexpression", None, 100),
        ("event_reexpression_percentage_points", "view_event_reexpression", "repeat_event_reexpression", None, 100),
    ]
    n = len(metrics)
    out = []
    for label, vf, rf, denomf, scale in fields:
        diff = np.asarray([getattr(x, vf) - getattr(x, rf) for x in metrics], dtype=np.float64)
        denom = np.asarray([getattr(x, denomf) for x in metrics], dtype=np.float64) if denomf else None
        obs = scale * diff.sum() / (denom.sum() if denomf else n)
        chunks: list[np.ndarray] = []
        done = 0
        while done < reps:
            batch = min(100, reps - done)
            idx = rng.integers(0, n, size=(batch, n), dtype=np.int32)
            nums = diff[idx].sum(axis=1)
            vals_b = scale * nums / (denom[idx].sum(axis=1) if denomf else n)
            chunks.append(vals_b)
            done += batch
        vals = np.sort(np.concatenate(chunks))
        out.append({"contrast": label, "view_minus_repeat": obs,
                    "paired_bootstrap_p2_5": float(vals[int(.025 * reps)]),
                    "paired_bootstrap_p97_5": float(vals[int(.975 * reps)]),
                    "bootstrap_units": n, "bootstrap_reps": reps, "seed": seed})
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
    comps_heads, official = official_signatures()
    (OUT / "official_benchmark_signatures.json").write_text(json.dumps(official, indent=2), encoding="utf-8")
    pairs = load_jsonl(PAIR_PATH)
    metrics: list[PairMetric] = []
    repeats: list[str] = []
    sources: list[str] = []
    views: list[str] = []
    pair_lookup: dict[str, dict] = {}
    for p in pairs:
        m, rep = pair_metrics(p, comps_heads)
        metrics.append(m); repeats.append(rep)
        sources.append(" ".join(p["source_text"].split())); views.append(" ".join(p["rewrite_text"].split()))
        pair_lookup[p["pair_id"]] = p
    write_csv(OUT / "pair_metrics.csv", [asdict(m) for m in metrics])

    # Validate reconstruction against the first pair rows in the actual pools.
    view_meta = load_jsonl(VIEW_ROW_META)
    repeat_meta = load_jsonl(REPEAT_ROW_META)
    clean_meta = load_jsonl(CLEAN_ROW_META)
    n_changed_rows = len(view_meta)
    view_rows = load_jsonl(VIEW_POOL, n_changed_rows)
    repeat_rows = load_jsonl(REPEAT_POOL, n_changed_rows)
    clean_rows = load_jsonl(CLEAN_POOL, n_changed_rows)
    rebuilt_v, rebuilt_r, row_words, row_pairids = [], [], [], []
    curv: list[str] = []; curr: list[str] = []; curw = 0; curids: list[str] = []
    for p, rep in zip(pairs, repeats):
        L = len(p["source_text"].split()) + len(p["rewrite_text"].split())
        if curw and curw + L > 160:
            rebuilt_v.append(" ".join(curv)); rebuilt_r.append(" ".join(curr)); row_words.append(curw); row_pairids.append(curids)
            curv, curr, curw, curids = [], [], 0, []
        curv.append(p["source_text"] + " " + p["rewrite_text"])
        curr.append(p["source_text"] + " " + rep)
        curw += L; curids.append(p["pair_id"])
    if curv:
        rebuilt_v.append(" ".join(curv)); rebuilt_r.append(" ".join(curr)); row_words.append(curw); row_pairids.append(curids)
    pair_row_count = len(rebuilt_v)
    validation = {
        "pair_count": len(pairs), "pair_row_count": pair_row_count,
        "changed_meta_rows_each": [len(view_meta), len(repeat_meta), len(clean_meta)],
        "pair_words_recomputed": sum(len(x.split()) + len(y.split()) for x, y in zip(sources, views)),
        "source_words_recomputed": sum(len(x.split()) for x in sources),
        "rewrite_words_recomputed": sum(len(x.split()) for x in views),
        "view_pair_rows_exact_match": all(rebuilt_v[i] == view_rows[i]["text"] for i in range(pair_row_count)),
        "repeat_pair_rows_exact_match": all(rebuilt_r[i] == repeat_rows[i]["text"] for i in range(pair_row_count)),
        "view_repeat_meta_identical": view_meta == repeat_meta,
        "row_pairids_match_meta": all(row_pairids[i] == view_meta[i]["pair_ids"] for i in range(pair_row_count)),
        "full_changed_words": [sum(r["words"] for r in x) for x in (view_rows, repeat_rows, clean_rows)],
        "metadata_expected": metadata["dose"],
    }
    (OUT / "reconstruction_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")

    # Segment-level signal tables.  Full blocks include the shared source text and
    # the 133-word neutral top-up; pair-added comparisons are exactly length matched.
    segment_specs = [
        ("pair_source_shared", sources, "pair"), ("view_added_rewrite", views, "pair"),
        ("repeat_added_fragment", repeats, "pair"),
        ("view_full_changed_block", [r["text"] for r in view_rows], "row"),
        ("repeat_full_changed_block", [r["text"] for r in repeat_rows], "row"),
        ("clean_full_changed_block", [r["text"] for r in clean_rows], "row"),
    ]
    structure, signal_rows = [], []
    for name, texts, unit in segment_specs:
        s, rr = aggregate_segment(name, texts, comps_heads, unit)
        structure.append(s); signal_rows.extend(rr)
    write_csv(OUT / "corpus_structure.csv", structure)
    write_csv(OUT / "signal_rates.csv", signal_rows)

    # Surface/transition summary.
    float_fields = ["view_content_recall", "repeat_content_recall", "supplied_content_recall", "supplied_entity_recall",
                    "view_anchor_recall", "repeat_anchor_recall", "view_bigram_coverage", "repeat_bigram_coverage",
                    "view_trigram_coverage", "repeat_trigram_coverage", "view_sequence_ratio", "repeat_sequence_ratio"]
    binary_fields = ["view_complete", "repeat_complete", "view_pronoun_shift", "repeat_pronoun_shift",
                     "view_voice_shift", "repeat_voice_shift", "view_anchor_order_inversion", "repeat_anchor_order_inversion",
                     "view_prep_shift", "repeat_prep_shift", "view_event_reexpression", "repeat_event_reexpression",
                     "view_identity_variation", "repeat_identity_variation", "view_role_reexpression", "repeat_role_reexpression"]
    transition_rows = []
    for f in float_fields:
        vals = [getattr(m, f) for m in metrics if not math.isnan(getattr(m, f))]
        vals.sort()
        transition_rows.append({"metric": f, "n": len(vals), "mean": sum(vals)/len(vals),
                                "median": vals[len(vals)//2], "p10": vals[int(.1*len(vals))], "p90": vals[int(.9*len(vals))]})
    for f in binary_fields:
        vals = [getattr(m, f) for m in metrics]
        transition_rows.append({"metric": f, "n": len(vals), "mean": sum(vals)/len(vals),
                                "median": "", "p10": "", "p90": ""})
    write_csv(OUT / "pair_transition_summary.csv", transition_rows)
    write_csv(OUT / "paired_bootstrap_contrasts.csv", paired_bootstrap(metrics))

    # Per-category preservation/loss makes clear whether VIEW merely adds more
    # marker tokens or restates source events in a changed surface form.
    category_transition_rows = []
    for category in STATE_CATEGORIES:
        pat = CATEGORY_PATTERNS[category]
        src_sets = [{t for t in lex_tokens(s) if pat.match(t)} for s in sources]
        for arm, seconds in [("view", views), ("repeat", repeats)]:
            sec_sets = [{t for t in lex_tokens(s) if pat.match(t)} for s in seconds]
            src_any = sum(bool(x) for x in src_sets)
            sec_any = sum(bool(x) for x in sec_sets)
            both = sum(bool(a) and bool(b) for a, b in zip(src_sets, sec_sets))
            changed = sum(bool(a) and bool(b) and a != b for a, b in zip(src_sets, sec_sets))
            novel = sum(bool(a) and bool(b) and bool(b - a) for a, b in zip(src_sets, sec_sets))
            same = sum(bool(a) and bool(b) and a == b for a, b in zip(src_sets, sec_sets))
            lost = sum(bool(a) and not b for a, b in zip(src_sets, sec_sets))
            gained = sum(not a and bool(b) for a, b in zip(src_sets, sec_sets))
            category_transition_rows.append({
                "category": category, "arm": arm, "pairs": len(pairs),
                "source_any": src_any, "second_any": sec_any, "both_any": both,
                "retention_pct_of_source_any": 100 * safe_div(both, src_any),
                "same_marker_set": same, "changed_marker_set": changed,
                "changed_marker_pct_of_both": 100 * safe_div(changed, both),
                "novel_second_marker": novel,
                "novel_second_marker_pct_of_both": 100 * safe_div(novel, both),
                "lost": lost, "gained": gained,
            })
    write_csv(OUT / "category_transitions.csv", category_transition_rows)

    # Domain/origin stratification for the main per-token contrasts.
    groups: dict[tuple[str, str], list[int]] = collections.defaultdict(list)
    for i, (p, m) in enumerate(zip(pairs, metrics)):
        for d in p.get("domain_hits") or ["no_domain"]:
            groups[("domain", d)].append(i)
        groups[("origin", m.origin)].append(i)
    strat = []
    for (group_type, group), ids in sorted(groups.items()):
        words = sum(metrics[i].second_words for i in ids)
        for arm, sf, pf in [("view", "view_state_hits", "view_property_hits"), ("repeat", "repeat_state_hits", "repeat_property_hits")]:
            sh = sum(getattr(metrics[i], sf) for i in ids); ph = sum(getattr(metrics[i], pf) for i in ids)
            strat.append({"group_type": group_type, "group": group, "arm": arm, "pairs": len(ids), "words": words,
                          "state_hits": sh, "state_per_1000": 1000*sh/words,
                          "property_hits": ph, "property_per_1000": 1000*ph/words,
                          "state_to_property_ratio": safe_div(sh, ph)})
    write_csv(OUT / "stratified_rates.csv", strat)

    # Ranked, auditable examples: state/event re-expression, role shift, property,
    # and deliberately selected low-scoring counterexamples.
    def score_state(i: int) -> tuple:
        m = metrics[i]
        return (m.view_event_reexpression, m.view_role_reexpression, m.view_state_hits-m.repeat_state_hits,
                -m.view_bigram_coverage, m.supplied_entity_recall)
    def score_property(i: int) -> tuple:
        m = metrics[i]
        return (m.view_property_hits-m.repeat_property_hits, m.view_identity_variation, -m.view_bigram_coverage)
    state_ids = sorted(range(len(metrics)), key=score_state, reverse=True)[:12]
    prop_ids = sorted(range(len(metrics)), key=score_property, reverse=True)[:8]
    counter_ids = sorted(range(len(metrics)), key=lambda i: (metrics[i].repeat_state_hits-metrics[i].view_state_hits,
                                                              metrics[i].view_bigram_coverage), reverse=True)[:8]
    picked = [("state_or_role_reexpression", i) for i in state_ids] + [("property_dense", i) for i in prop_ids] + [("counterexample_repeat_retains_more_state", i) for i in counter_ids]
    with (OUT / "representative_examples.jsonl").open("w", encoding="utf-8") as f:
        for kind, i in picked:
            p, m = pairs[i], metrics[i]
            f.write(json.dumps({"example_kind": kind, "pair_id": m.pair_id,
                                "source": p["source_text"], "view": p["rewrite_text"], "repeat": repeats[i],
                                "metrics": asdict(m)}, ensure_ascii=False) + "\n")

    # Human-readable compact extracts.
    lines = ["# Representative pair extracts", "",
             "These examples are heuristic-ranked, not hand-picked proof. Counts are in `pair_metrics.csv`.", ""]
    for kind, ids in [("State/event and role re-expression", state_ids[:6]),
                      ("Property-dense rewrites", prop_ids[:4]),
                      ("Counterexamples: repeat fragment retains more state markers", counter_ids[:4])]:
        lines += [f"## {kind}", ""]
        for i in ids:
            p, m = pairs[i], metrics[i]
            lines += [f"- `{m.pair_id}`", f"  - SOURCE: {p['source_text']}",
                      f"  - VIEW: {p['rewrite_text']}", f"  - REPEAT: {repeats[i]}",
                      f"  - state hits (V/R)={m.view_state_hits}/{m.repeat_state_hits}; "
                      f"bigram coverage (V/R)={m.view_bigram_coverage:.3f}/{m.repeat_bigram_coverage:.3f}; "
                      f"event re-expression (V/R)={m.view_event_reexpression}/{m.repeat_event_reexpression}", ""]
    (OUT / "representative_examples.md").write_text("\n".join(lines), encoding="utf-8")

    # Machine-readable run synopsis.
    synopsis = {"validation": validation, "outputs": sorted(p.name for p in OUT.iterdir()),
                "heuristic_notes": {
                    "identity_variation": "content-token recall >=0.50, anchor recall >=0.80 (or no anchors), and source bigram coverage <0.80",
                    "role_reexpression": "identity_variation plus any event-marker, pronoun-set, passive-cue, anchor-order, or preposition-set shift",
                    "property_narrow": "can/cannot + has/have/had + selected multiword COMPS-like frames; excludes generic copula from the narrow total",
                    "state_total": "sum of explicit movement, transfer, change, and containment verb-form lexicons",
                    "command_frames": "literal Entity-like move...from...to, remove...from, put...into, and contains within limited windows",
                }}
    (OUT / "analysis_synopsis.json").write_text(json.dumps(synopsis, indent=2), encoding="utf-8")
    print(json.dumps(synopsis, indent=2))


if __name__ == "__main__":
    main()
