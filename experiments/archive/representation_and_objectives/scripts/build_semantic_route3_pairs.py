#!/usr/bin/env python3
"""research: rebuild Route 3 as semantically matched two-context pairs.

The research pair file established that target-shared cross-context scoring is
unsaturated, but the examples were often noisy: unrelated domains, dialogue
transcripts, pronouns, and metaphorical/abstract uses.  This script rebuilds a
smaller, cleaner object from the same legal 10M pool with three purposes:

1. Prefer sentence pairs with the same physical-property contrast and matched
   entity category/head, so the two contexts are semantically comparable.
2. Save correspondence-destroying controls that preserve target/family/length
   distributions as much as possible.
3. Keep all outputs as evaluation-independent research objects for deciding
   whether a tiny antisymmetric training experiment is worth launching.

No model weights are changed.  The extraction is deterministic and uses only the
legal compact-view pool (SHA 215944...).
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = _public_path('experiments/archive/representation_and_objectives/scripts')
WORKSPACE = _public_path('experiments/archive/representation_and_objectives')
STUDY = _public_path('experiments/archive/representation_and_objectives')
USER_ROOT = _public_path('.')
CORPUS = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl')
OUT_DIR = _public_path('experiments/archive/representation_and_objectives/data/semantic_route3')
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 19503
rng = random.Random(SEED)

ANTONYM_PAIRS = [
    ("hot", "cold"), ("warm", "cool"), ("wet", "dry"),
    ("soft", "hard"), ("open", "closed"), ("full", "empty"),
    # broken/intact excluded because intact is multi-token under legal16k.
    ("bright", "dark"), ("clean", "dirty"), ("strong", "weak"),
    ("new", "old"), ("deep", "shallow"), ("thick", "thin"),
    ("heavy", "light"), ("large", "small"), ("tight", "loose"),
    ("solid", "liquid"), ("frozen", "melted"), ("flat", "sharp"),
    ("smooth", "rough"),
]
PHYS_WORDS = {w for pair in ANTONYM_PAIRS for w in pair}

COPULAS = r"(?:is|was|are|were|became|becomes|turned|turns|gets|got|remains|remained|stays|stayed|feels|felt|looks|looked|seems|seemed)"
DET = r"(?:[Tt]he|[Aa]n?|[Tt]his|[Tt]hat|[Ii]ts|[Tt]heir|[Oo]ur|[Mm]y|[Hh]is|[Hh]er|[Tt]hese|[Tt]hose)"
PAT_ENTITY_STATE = re.compile(
    rf'\b({DET}\s+[A-Za-z][A-Za-z\'-]*(?:\s+[A-Za-z][A-Za-z\'-]*)?)\s+({COPULAS})\s+(?:(?:very|quite|extremely|rather|still|now|also|often|usually|always|so|too)\s+)?([A-Za-z][A-Za-z\'-]*)\b',
    re.IGNORECASE,
)
SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z*\"“])')
WORD_RE = re.compile(r"[A-Za-z]+")

BAD_MARKERS = [
    "*CHI", "*MOT", "*FAT", "*INV", "@", "http", "www.", "<BR", "&lt;", "&gt;",
    "[exits]", "[enters", "[aft", "\t",
]
PRONOUN_HEADS = {"it", "they", "he", "she", "we", "you", "i", "them", "him", "her"}
ABSTRACT_HEADS = {
    "life", "lives", "policy", "plan", "plans", "rumor", "rumors", "sign", "signs",
    "reason", "reasons", "thought", "thoughts", "idea", "ideas", "way", "ways",
    "place", "places", "world", "time", "times", "case", "cases", "issue", "issues",
    "rate", "rates", "traffic", "document", "documents", "news", "story", "stories",
    "heart", "hearts", "condition", "conditions", "feeling", "feelings",
}
STOP_CONTENT = {
    "the", "a", "an", "this", "that", "these", "those", "its", "their", "our", "my", "his", "her",
    "is", "was", "are", "were", "be", "been", "being", "became", "becomes", "turned", "turns", "gets", "got",
    "remains", "remained", "stays", "stayed", "feels", "felt", "looks", "looked", "seems", "seemed",
    "very", "quite", "extremely", "rather", "still", "now", "also", "often", "usually", "always", "so", "too",
    "and", "or", "but", "because", "with", "without", "from", "into", "onto", "over", "under", "before", "after",
    "then", "when", "while", "where", "who", "what", "which", "as", "of", "to", "in", "on", "for", "by", "at",
}

CATEGORY_LEXICON: dict[str, set[str]] = {
    "body": set("eye eyes hand hands arm arms leg legs foot feet face head hair skin throat neck mouth ear ears cheek cheeks brow heart body bodies".split()),
    "container_building": set("box boxes bag bags door doors window windows drawer drawers room rooms house houses home cottage bottle bottles cup cups glass glasses jar jars trunk trunks cabinet cabinets bed beds bureau shelf shelves building buildings tower towers".split()),
    "natural_environment": set("ground road roads river rivers sea ocean water waters air sky skies cloud clouds sun moon star stars mountain mountains hill hills land soil mud snow ice fire forest forests weather morning night".split()),
    "artifact_material": set("book books telephone paper papers silk wood furniture cloth clothes clothing metal stone surface surfaces arch ink document documents page pages wall walls floor floors roof roofs car cars train trains".split()),
    "animal_plant": set("tree trees flower flowers leaf leaves frog frogs bird birds animal animals dog dogs cat cats horse horses fish".split()),
    "food_substance": set("rice sugar fruit blackberries macaroni cheese milk tea coffee soup bread meat oil liquid liquids material materials".split()),
}

COPULA_CLASS = {
    "is": "be", "was": "be", "are": "be", "were": "be",
    "remains": "persist", "remained": "persist", "stays": "persist", "stayed": "persist",
    "became": "change", "becomes": "change", "turned": "change", "turns": "change", "gets": "change", "got": "change",
    "feels": "percept", "felt": "percept", "looks": "percept", "looked": "percept", "seems": "percept", "seemed": "percept",
}


def rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(USER_ROOT))
    except Exception:
        return str(p)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sentence_quality(sent: str) -> tuple[bool, str]:
    s = sent.strip()
    if len(s) < 30 or len(s) > 260:
        return False, "length_char"
    words = WORD_RE.findall(s)
    if len(words) < 6 or len(words) > 42:
        return False, "length_word"
    low = s.lower()
    for marker in BAD_MARKERS:
        if marker.lower() in low:
            return False, "bad_marker"
    if s.count("*") or s.count("[") or s.count("]"):
        return False, "transcript_or_stage"
    if s.count('"') + s.count("“") + s.count("”") + s.count("'") > 4:
        return False, "too_quoted"
    if " - " in s or s.startswith("-"):
        return False, "dialogue_dash"
    alpha = sum(ch.isalpha() for ch in s)
    if alpha / max(1, len(s)) < 0.55:
        return False, "non_alpha"
    return True, "ok"


def norm_entity(entity_raw: str) -> tuple[str, str, str]:
    toks = [w.lower() for w in WORD_RE.findall(entity_raw)]
    while toks and toks[0] in {"the", "a", "an", "this", "that", "these", "those", "its", "their", "our", "my", "his", "her"}:
        toks = toks[1:]
    ent = " ".join(toks)
    head = toks[-1] if toks else ""
    cat = "abstract_or_unknown"
    for cname, heads in CATEGORY_LEXICON.items():
        if head in heads:
            cat = cname
            break
    if head in PRONOUN_HEADS:
        cat = "pronoun"
    elif head in ABSTRACT_HEADS:
        cat = "abstract_or_unknown"
    return ent, head, cat


def content_signature(sent: str, target: str, entity_norm: str) -> set[str]:
    banned = set(WORD_RE.findall(target.lower())) | set(WORD_RE.findall(entity_norm.lower())) | STOP_CONTENT
    return {w.lower() for w in WORD_RE.findall(sent) if len(w) >= 3 and w.lower() not in banned and w.lower() not in PHYS_WORDS}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def process_corpus() -> tuple[dict[str, list[dict[str, Any]]], Counter, Counter]:
    by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    reject = Counter()
    rows_seen = 0
    with CORPUS.open("r", encoding="utf-8") as f:
        for row_idx, line in enumerate(f):
            if not line.strip():
                continue
            rows_seen += 1
            row = json.loads(line)
            text = row.get("text", "")
            for sent in SENT_SPLIT.split(text):
                sent = sent.strip()
                ok, reason = sentence_quality(sent)
                if not ok:
                    reject[reason] += 1
                    continue
                for m in PAT_ENTITY_STATE.finditer(sent):
                    state = m.group(3).lower().strip("'-")
                    if state not in PHYS_WORDS:
                        continue
                    # Exclude sentences that contain the opposite property too; the span-level task becomes ambiguous.
                    fams = [p for p in ANTONYM_PAIRS if state in p]
                    if fams:
                        opp = fams[0][1] if fams[0][0] == state else fams[0][0]
                        if re.search(rf"\b{re.escape(opp)}\b", sent, flags=re.I):
                            reject["contains_opposite"] += 1
                            continue
                    ent_norm, head, cat = norm_entity(m.group(1))
                    if not ent_norm or cat == "pronoun":
                        reject["pronoun_entity"] += 1
                        continue
                    cop = m.group(2).lower()
                    rec = {
                        "sentence": sent,
                        "row_idx": row_idx,
                        "example_id": row.get("example_id"),
                        "words": len(WORD_RE.findall(sent)),
                        "char_len": len(sent),
                        "entity_raw": m.group(1),
                        "entity_norm": ent_norm,
                        "entity_head": head,
                        "entity_category": cat,
                        "copula": cop,
                        "copula_class": COPULA_CLASS.get(cop, cop),
                        "state": state,
                        "state_start": m.start(3),
                        "state_end": m.end(3),
                        "content_sig": sorted(content_signature(sent, state, ent_norm)),
                    }
                    by_state[state].append(rec)
    return by_state, reject, Counter({"rows_seen": rows_seen})


def pair_score(a: dict[str, Any], b: dict[str, Any]) -> float:
    same_head = a["entity_head"] == b["entity_head"] and a["entity_head"]
    same_cat = a["entity_category"] == b["entity_category"] and a["entity_category"] != "abstract_or_unknown"
    len_diff = abs(a["words"] - b["words"])
    char_ratio = abs(a["char_len"] - b["char_len"]) / max(a["char_len"], b["char_len"], 1)
    cop_mismatch = 0 if a["copula_class"] == b["copula_class"] else 1
    overlap = jaccard(set(a["content_sig"]), set(b["content_sig"]))
    # Lower is better.  Same head is strongest, same broad physical category next.
    return (
        (0 if same_head else (1.2 if same_cat else 5.0))
        + 0.06 * len_diff
        + 1.5 * char_ratio
        + 0.45 * cop_mismatch
        + 0.25 * (1.0 - overlap)
        + (0.0 if a["entity_category"] != "abstract_or_unknown" else 1.0)
        + (0.0 if b["entity_category"] != "abstract_or_unknown" else 1.0)
    )


def make_pair(pa: str, pb: str, a: dict[str, Any], b: dict[str, Any], rank: int, score: float, control: str = "true_semantic") -> dict[str, Any]:
    return {
        "pair_type": "semantic_antonym",
        "control_type": control,
        "antonym_pair": f"{pa}/{pb}",
        "target_a": pa,
        "foil_a": pb,
        "target_b": pb,
        "foil_b": pa,
        "context_a": a["sentence"],
        "context_b": b["sentence"],
        "row_a": a["row_idx"],
        "row_b": b["row_idx"],
        "state_start_a": a["state_start"],
        "state_end_a": a["state_end"],
        "state_start_b": b["state_start"],
        "state_end_b": b["state_end"],
        "entity_a": a["entity_raw"],
        "entity_b": b["entity_raw"],
        "entity_head_a": a["entity_head"],
        "entity_head_b": b["entity_head"],
        "entity_category_a": a["entity_category"],
        "entity_category_b": b["entity_category"],
        "copula_class_a": a["copula_class"],
        "copula_class_b": b["copula_class"],
        "words_a": a["words"],
        "words_b": b["words"],
        "semantic_pair_rank": rank,
        "semantic_pair_score": score,
        "same_entity_head": a["entity_head"] == b["entity_head"],
        "same_entity_category": a["entity_category"] == b["entity_category"],
    }


def build_true_pairs(by_state: dict[str, list[dict[str, Any]]], max_per_family: int = 80) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}
    # Two passes: first require same head or same physical category; if the pool
    # is too small for a meaningful matched pilot, keep the same head/category
    # preference but fill remaining slots with best-scoring broad cross-category
    # matches.  This preserves the semantic matching direction while avoiding an
    # artificially tiny object caused only by a six-label coarse lexicon.
    for pa, pb in ANTONYM_PAIRS:
        la = by_state.get(pa, [])
        lb = by_state.get(pb, [])
        cand: list[tuple[float, dict[str, Any], dict[str, Any], bool]] = []
        for a in la:
            for b in lb:
                if a["row_idx"] == b["row_idx"]:
                    continue
                same_head = a["entity_head"] == b["entity_head"] and a["entity_head"]
                same_cat = a["entity_category"] == b["entity_category"] and a["entity_category"] != "abstract_or_unknown"
                if abs(a["words"] - b["words"]) > 22:
                    continue
                matched = bool(same_head or same_cat)
                # Cross-category fill is allowed but receives a large score penalty.
                score = pair_score(a, b) + (0.0 if matched else 4.0)
                cand.append((score, a, b, matched))
        cand.sort(key=lambda x: x[0])
        used_a: Counter[int] = Counter()
        used_b: Counter[int] = Counter()
        kept = 0
        kept_semantic = 0
        kept_cross_fill = 0
        for sc, a, b, matched in cand:
            ka = (a["row_idx"], a["state_start"], a["state_end"])
            kb = (b["row_idx"], b["state_start"], b["state_end"])
            if used_a[ka] >= 3 or used_b[kb] >= 3:
                continue
            kept += 1
            if matched:
                kept_semantic += 1
                ctype = "true_semantic"
            else:
                kept_cross_fill += 1
                ctype = "true_cross_category_fill"
            used_a[ka] += 1
            used_b[kb] += 1
            pairs.append(make_pair(pa, pb, a, b, kept, sc, ctype))
            if kept >= max_per_family:
                break
        stats[f"{pa}/{pb}"] = {
            "state_a_candidates": len(la),
            "state_b_candidates": len(lb),
            "candidate_edges": len(cand),
            "kept": kept,
            "kept_semantic": kept_semantic,
            "kept_cross_category_fill": kept_cross_fill,
            "same_head_kept": sum(1 for p in pairs if p["antonym_pair"] == f"{pa}/{pb}" and p["same_entity_head"]),
            "same_category_kept": sum(1 for p in pairs if p["antonym_pair"] == f"{pa}/{pb}" and p["same_entity_category"]),
        }
    return pairs, stats


def permute_b_within_family(true_pairs: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    """Destroy pair correspondence while preserving exact A/B context pools.

    mode=target_family_shuffle: random derangement within each antonym family.
    mode=family_length_matched: greedily pair each A with a B of same family and close length but different category/head where possible.
    """
    out: list[dict[str, Any]] = []
    byfam: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in true_pairs:
        byfam[p["antonym_pair"]].append(p)
    for fam, ps in sorted(byfam.items()):
        if len(ps) < 2:
            continue
        bpool = [p for p in ps]
        if mode == "target_family_shuffle":
            perm = list(range(len(ps)))
            for _ in range(1000):
                rng.shuffle(perm)
                if all(perm[i] != i for i in range(len(ps))):
                    break
            else:
                perm = perm[1:] + perm[:1]
            for i, j in enumerate(perm):
                a = ps[i]
                b = ps[j]
                q = dict(a)
                for key in ["context_b", "row_b", "state_start_b", "state_end_b", "entity_b", "entity_head_b", "entity_category_b", "copula_class_b", "words_b"]:
                    q[key] = b[key]
                q["semantic_pair_score"] = None
                q["semantic_pair_rank"] = a["semantic_pair_rank"]
                q["same_entity_head"] = q["entity_head_a"] == q["entity_head_b"]
                q["same_entity_category"] = q["entity_category_a"] == q["entity_category_b"]
                q["control_type"] = mode
                q["control_source_true_rank_b"] = b["semantic_pair_rank"]
                out.append(q)
        elif mode == "family_length_matched":
            remaining = bpool[:]
            for a in sorted(ps, key=lambda p: p["words_a"]):
                def cost(b: dict[str, Any]) -> tuple[float, float]:
                    same = (a["semantic_pair_rank"] == b["semantic_pair_rank"])
                    same_head = a["entity_head_a"] == b["entity_head_b"]
                    same_cat = a["entity_category_a"] == b["entity_category_b"]
                    # Prefer different head/category to break semantic correspondence, but keep length close.
                    return (
                        (1000 if same and len(ps) > 1 else 0)
                        + (8 if same_head else 0)
                        + (3 if same_cat else 0)
                        + abs(a["words_a"] - b["words_b"]),
                        rng.random(),
                    )
                b = min(remaining, key=cost)
                remaining.remove(b)
                q = dict(a)
                for key in ["context_b", "row_b", "state_start_b", "state_end_b", "entity_b", "entity_head_b", "entity_category_b", "copula_class_b", "words_b"]:
                    q[key] = b[key]
                q["semantic_pair_score"] = None
                q["control_type"] = mode
                q["same_entity_head"] = q["entity_head_a"] == q["entity_head_b"]
                q["same_entity_category"] = q["entity_category_a"] == q["entity_category_b"]
                q["control_source_true_rank_b"] = b["semantic_pair_rank"]
                out.append(q)
        else:
            raise ValueError(mode)
    return out


def swap_targets(true_pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for p in true_pairs:
        q = dict(p)
        # Keep contexts/spans, swap the correct/foil target assignment.  This is
        # a target-swapped null and should invert the antisymmetric score.
        q["target_a"], q["target_b"] = p["target_b"], p["target_a"]
        q["foil_a"], q["foil_b"] = p["foil_b"], p["foil_a"]
        q["control_type"] = "target_swapped_null"
        out.append(q)
    return out


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            # content_sig not present here; all values are json-serializable.
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def summarize_pairs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    fam = Counter(p["antonym_pair"] for p in rows)
    cat = Counter((p.get("entity_category_a"), p.get("entity_category_b")) for p in rows)
    len_diffs = [abs(int(p["words_a"]) - int(p["words_b"])) for p in rows]
    return {
        "n": len(rows),
        "families": dict(sorted(fam.items())),
        "same_head_frac": sum(1 for p in rows if p.get("same_entity_head")) / len(rows),
        "same_category_frac": sum(1 for p in rows if p.get("same_entity_category")) / len(rows),
        "len_diff_mean": statistics.fmean(len_diffs),
        "len_diff_median": statistics.median(len_diffs),
        "category_pair_top10": [{"category_a": a, "category_b": b, "n": n} for (a, b), n in cat.most_common(10)],
    }


def main() -> None:
    print(json.dumps({"event": "start", "corpus": rel(CORPUS), "seed": SEED}), flush=True)
    by_state, reject, corpus_counter = process_corpus()
    true_pairs, fam_stats = build_true_pairs(by_state, max_per_family=80)
    ctrl_target = permute_b_within_family(true_pairs, "target_family_shuffle")
    ctrl_len = permute_b_within_family(true_pairs, "family_length_matched")
    ctrl_swap = swap_targets(true_pairs)

    # Shuffle rows deterministically for train/test convenience while retaining rank metadata.
    for rows in [true_pairs, ctrl_target, ctrl_len, ctrl_swap]:
        rng.shuffle(rows)

    paths = {
        "true_semantic": _public_path('experiments/archive/representation_and_objectives/data/semantic_route3/route3_semantic_true_pairs.jsonl'),
        "target_family_shuffle": _public_path('experiments/archive/representation_and_objectives/data/semantic_route3/route3_control_target_family_shuffle.jsonl'),
        "family_length_matched": _public_path('experiments/archive/representation_and_objectives/data/semantic_route3/route3_control_family_length_matched.jsonl'),
        "target_swapped_null": _public_path('experiments/archive/representation_and_objectives/data/semantic_route3/route3_control_target_swapped_null.jsonl'),
    }
    write_jsonl(paths["true_semantic"], true_pairs)
    write_jsonl(paths["target_family_shuffle"], ctrl_target)
    write_jsonl(paths["family_length_matched"], ctrl_len)
    write_jsonl(paths["target_swapped_null"], ctrl_swap)

    # Save candidate sentence counts without dumping all sentence text.
    candidate_counts = {state: len(rows) for state, rows in sorted(by_state.items())}
    summary = {
        "status": "SEMANTIC_ROUTE3_PAIRS",
        "scientific_purpose": "clean semantically matched two-context property pairs and correspondence-destroying controls before any Route 3 training",
        "corpus": rel(CORPUS),
        "corpus_sha256": sha256_file(CORPUS),
        "expected_corpus_sha256": "215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23",
        "seed": SEED,
        "candidate_counts_by_state": candidate_counts,
        "reject_counts": dict(reject),
        "rows_seen": corpus_counter["rows_seen"],
        "family_stats": fam_stats,
        "summaries": {
            "true_semantic": summarize_pairs(true_pairs),
            "target_family_shuffle": summarize_pairs(ctrl_target),
            "family_length_matched": summarize_pairs(ctrl_len),
            "target_swapped_null": summarize_pairs(ctrl_swap),
        },
        "files": {k: rel(v) for k, v in paths.items()},
    }
    out_json = _public_path('experiments/archive/representation_and_objectives/data/semantic_route3/semantic_route3_summary.json')
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    note_lines = [
        "# research semantic Route 3 pair rebuild",
        "",
        f"Corpus: `{rel(CORPUS)}`",
        f"Corpus SHA256: `{summary['corpus_sha256']}`",
        f"True semantic pairs: {len(true_pairs)}",
        f"Target/family shuffle controls: {len(ctrl_target)}",
        f"Family/length matched controls: {len(ctrl_len)}",
        f"Target-swapped nulls: {len(ctrl_swap)}",
        "",
        "## True-pair quality summary",
        f"- same entity head fraction: {summary['summaries']['true_semantic']['same_head_frac']:.3f}",
        f"- same entity category fraction: {summary['summaries']['true_semantic']['same_category_frac']:.3f}",
        f"- mean word-length difference: {summary['summaries']['true_semantic']['len_diff_mean']:.2f}",
        f"- median word-length difference: {summary['summaries']['true_semantic']['len_diff_median']:.2f}",
        "",
        "## Family yields",
    ]
    for fam, st in sorted(fam_stats.items()):
        if st["kept"]:
            note_lines.append(f"- {fam}: kept {st['kept']} / candidate_edges {st['candidate_edges']} (semantic={st['kept_semantic']}, cross_fill={st['kept_cross_category_fill']}, same_head_kept={st['same_head_kept']})")
    note_lines += [
        "",
        "## Files",
    ]
    for k, v in paths.items():
        note_lines.append(f"- {k}: `{rel(v)}`")
    note_lines.append(f"- JSON summary: `{rel(out_json)}`")
    note = _public_path('research/notes/representation_and_objectives/semantic_route3_pair_rebuild.md')
    note.write_text("\n".join(note_lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": summary["status"],
        "true_pairs": len(true_pairs),
        "target_family_shuffle": len(ctrl_target),
        "family_length_matched": len(ctrl_len),
        "same_head_frac": summary["summaries"]["true_semantic"]["same_head_frac"],
        "same_category_frac": summary["summaries"]["true_semantic"]["same_category_frac"],
        "out_json": rel(out_json),
        "note": rel(note),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
