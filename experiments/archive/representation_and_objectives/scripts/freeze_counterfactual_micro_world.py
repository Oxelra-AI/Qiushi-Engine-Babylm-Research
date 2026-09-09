#!/usr/bin/env python3
"""research: freeze an executable counterfactual micro-world frame set.

Scientific purpose
------------------
After the sparse20 coupled dual-view route closed, the research needs a fixed,
legal-pool-derived measurement of context-conditioned alternative binding that cannot
be satisfied by margin flattening or row turnover.  The structural contrast v1
measured coherent-vs-corrupted text margins and failed the 82M ranking while containing
surface artifacts.  This script freezes a different object before any checkpoint
scoring: crossed-sign, single-mask alternative-binding frames with matched target
lengths and explicit controls.

No model scoring or training occurs here.  The output frame set is the object that later
GPU inference may score.
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
import os
import random
import re
import statistics
import string
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from transformers import AutoTokenizer

USER_ROOT = _public_path('.')
os.chdir(USER_ROOT)
A01_WS = _public_path('experiments/archive/representation_and_objectives')
A02_WS = _public_path('experiments/archive/frontier_consolidation')
POOL_PATH = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl')
LEGAL16K_TOK = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M')
LEGAL40K_TOK = _public_path('experiments/archive/representation_and_objectives/training/runs/legal40k_accum_compact_view_reinvest_seed43022/hf_model/chck_100M')
OUT_ROOT = _public_path('experiments/archive/representation_and_objectives/data/counterfactual_micro_world')
NOTE_PATH = _public_path('research/notes/representation_and_objectives/counterfactual_micro_world_freeze.md')

WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’-]*")
PUNCT = string.punctuation + "“”‘’«»‹›"

# Tiny controlled lexicon, filtered by legal-pool corpus occurrence and both tokenizers.
NAMES_CAND = [
    "Alice", "Bob", "Sam", "Tom", "Kate", "Amy", "Ben", "Nora", "Lucy", "Jack",
    "Emma", "Liam", "Mia", "Noah", "Anna", "John", "Mary", "Seth", "Rose", "Max",
]
OBJECTS_CAND = [
    "book", "key", "coin", "cup", "hat", "bag", "ball", "card", "sock", "shoe",
    "box", "bottle", "letter", "toy", "apple", "glass", "map", "ring", "pen", "paper",
]
LOCATIONS_CAND = [
    "table", "box", "basket", "bed", "desk", "shelf", "floor", "chair", "closet", "cupboard",
    "counter", "drawer", "room", "garden", "kitchen", "school", "house", "park", "shop", "bag",
]
# Renaming control: use common single-token names/objects distinct from originals, not invented strings.
RENAME_NAMES = ["Mia", "Noah", "Anna", "John", "Mary", "Max", "Rose", "Liam"]
RENAME_OBJECTS = ["map", "ring", "pen", "paper", "toy", "cup", "coin", "key"]
RENAME_LOCATIONS = ["room", "garden", "kitchen", "school", "house", "park", "shop", "desk"]

SPATIAL_PAIRS = [("in", "outside"), ("inside", "outside"), ("on", "under"), ("above", "below"), ("near", "far")]
STATE_PAIRS = [("open", "closed"), ("locked", "unlocked"), ("empty", "full"), ("clean", "dirty")]
TRANSFER_VERBS = ["gave", "handed", "passed", "sent", "offered", "showed"]
REPORT_VERBS = ["said", "thought", "believed", "claimed"]


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: Path | str) -> str:
    pp = Path(p)
    try:
        return str(pp.resolve().relative_to(USER_ROOT))
    except Exception:
        return str(pp)


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_obj(obj: Any) -> str:
    b = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def norm_word(w: str) -> str:
    return w.strip(PUNCT).lower().replace("’", "'")


def corpus_counts(path: Path, max_rows: int = 0) -> tuple[collections.Counter[str], dict[str, Any]]:
    cnt: collections.Counter[str] = collections.Counter()
    rows = 0
    words = 0
    sources: collections.Counter[str] = collections.Counter()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = obj.get("text", "")
            rows += 1
            src = obj.get("source") or obj.get("meta", {}).get("source") or "unknown"
            sources[src] += 1
            for m in WORD_RE.finditer(text):
                w = norm_word(m.group(0))
                if w:
                    cnt[w] += 1
                    words += 1
            if max_rows and rows >= max_rows:
                break
    meta = {"rows": rows, "word_tokens_counted": words, "source_counts": dict(sources.most_common())}
    return cnt, meta


def enc_ids(tok, text: str) -> list[int]:
    return list(tok(text, add_special_tokens=False)["input_ids"])


def token_len_in_context(tok, prefix: str, target: str, suffix: str) -> int:
    # Token length of the target span as it actually appears in the rendered sentence.
    sent = prefix + target + suffix
    start = len(prefix); end = start + len(target)
    enc = tok(sent, return_offsets_mapping=True, add_special_tokens=True)
    offsets = enc["offset_mapping"]
    ids = enc["input_ids"]
    chosen = []
    for tid, (a, b) in zip(ids, offsets):
        if b > start and a < end and not (a == b == 0):
            chosen.append(tid)
    return len(chosen)


def is_single_target(tok16, tok40, prefix: str, target: str, suffix: str) -> bool:
    return token_len_in_context(tok16, prefix, target, suffix) == 1 and token_len_in_context(tok40, prefix, target, suffix) == 1


def word_ok(w: str, counts: collections.Counter[str], tok16, tok40, min_count: int = 2) -> bool:
    wl = w.lower()
    if counts[wl] < min_count:
        return False
    # Require bare and leading-space contexts to be compact; name/object may appear after spaces.
    return len(enc_ids(tok16, " " + w)) <= 2 and len(enc_ids(tok40, " " + w)) <= 2


def target_prior(counts: collections.Counter[str], a: str, b: str) -> dict[str, Any]:
    ca = counts[a.lower()]
    cb = counts[b.lower()]
    total = ca + cb
    ratio = (max(ca, cb) / max(1, min(ca, cb))) if min(ca, cb) > 0 else math.inf
    return {"a_count": ca, "b_count": cb, "total": total, "max_min_ratio": ratio, "balanced_ratio_le_4": bool(ratio <= 4.0)}


def make_record(frame_id: str, family: str, depth: float, context1: str, context2: str, query_template: str,
                alt_a: str, alt_b: str, correct_c1: str, correct_c2: str, slots: dict[str, str], counts: collections.Counter[str],
                source: str, subtype: str) -> dict[str, Any]:
    assert correct_c1 in {"a", "b"} and correct_c2 in {"a", "b"} and correct_c1 != correct_c2
    return {
        "frame_id": frame_id,
        "family": family,
        "subtype": subtype,
        "depth": depth,
        "source": source,
        "context1": context1,
        "context2": context2,
        "query_template": query_template,
        "alt_a": alt_a,
        "alt_b": alt_b,
        "correct_c1": correct_c1,
        "correct_c2": correct_c2,
        "slots": slots,
        "target_prior": target_prior(counts, alt_a, alt_b),
    }


def render_query(query_template: str, target: str) -> tuple[str, tuple[int, int]]:
    marker = "{target}"
    if marker not in query_template:
        raise ValueError(query_template)
    prefix, suffix = query_template.split(marker, 1)
    sent = prefix + target + suffix
    return sent, (len(prefix), len(prefix) + len(target))


def with_context(context: str, query_template: str, target: str) -> tuple[str, tuple[int, int]]:
    q, (s, e) = render_query(query_template, target)
    prefix = (context.strip() + " ") if context.strip() else ""
    return prefix + q, (len(prefix) + s, len(prefix) + e)


def frame_token_quality(rec: dict[str, Any], tok16, tok40) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for context_key in ["context1", "context2", ""]:
        context = rec.get(context_key, "") if context_key else ""
        ck = context_key or "no_context"
        for alt_key in ["alt_a", "alt_b"]:
            target = rec[alt_key]
            sent, span = with_context(context, rec["query_template"], target)
            prefix = sent[:span[0]]; suffix = sent[span[1]:]
            out[f"{ck}_{alt_key}_tok16_len"] = token_len_in_context(tok16, prefix, target, suffix)
            out[f"{ck}_{alt_key}_tok40_len"] = token_len_in_context(tok40, prefix, target, suffix)
    return out


def apply_renaming(rec: dict[str, Any], name_map: dict[str, str], obj_map: dict[str, str], loc_map: dict[str, str]) -> dict[str, Any]:
    # Whole-word, case-sensitive replacements. For Type B target alternatives, update alt_a/alt_b too.
    repl = {**name_map, **obj_map, **loc_map}
    def rep_text(text: str) -> str:
        out = text
        # Longest first avoids nested replacements.
        for old, new in sorted(repl.items(), key=lambda kv: -len(kv[0])):
            out = re.sub(rf"\b{re.escape(old)}\b", new, out)
        return out
    r = json.loads(json.dumps(rec))
    for key in ["context1", "context2", "query_template", "alt_a", "alt_b"]:
        r[key] = rep_text(r[key])
    r["frame_id"] = rec["frame_id"] + "__renamed"
    r["source"] = rec.get("source", "") + "+renaming_control"
    r["renaming_of"] = rec["frame_id"]
    r["renaming_map"] = repl
    return r


def choose_vocab(counts, tok16, tok40) -> dict[str, list[str]]:
    names = [w for w in NAMES_CAND if word_ok(w, counts, tok16, tok40, min_count=1)]
    objects = [w for w in OBJECTS_CAND if word_ok(w, counts, tok16, tok40, min_count=2)]
    locations = [w for w in LOCATIONS_CAND if word_ok(w, counts, tok16, tok40, min_count=2)]
    rename_names = [w for w in RENAME_NAMES if word_ok(w, counts, tok16, tok40, min_count=1)]
    rename_objects = [w for w in RENAME_OBJECTS if word_ok(w, counts, tok16, tok40, min_count=2)]
    rename_locations = [w for w in RENAME_LOCATIONS if word_ok(w, counts, tok16, tok40, min_count=2)]
    return {
        "names": names, "objects": objects, "locations": locations,
        "rename_names": rename_names, "rename_objects": rename_objects, "rename_locations": rename_locations,
    }


def build_frames(counts, tok16, tok40, seed: int, per_family: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    vocab = choose_vocab(counts, tok16, tok40)
    frames: list[dict[str, Any]] = []
    rejects: collections.Counter[str] = collections.Counter()

    def can_score_query(qt: str, a: str, b: str, contexts: list[str]) -> bool:
        for c in contexts + [""]:
            for t in [a, b]:
                sent, span = with_context(c, qt, t)
                if not is_single_target(tok16, tok40, sent[:span[0]], t, sent[span[1]:]):
                    return False
        return True

    # Type A: direct spatial relation, depth 1.
    i = 0
    attempts = 0
    while i < per_family and attempts < 20000:
        attempts += 1
        obj = rng.choice(vocab["objects"]); loc = rng.choice([x for x in vocab["locations"] if x != obj])
        a, b = rng.choice(SPATIAL_PAIRS)
        # Avoid "far the table"; use a single target word and fixed phrase.
        if (a, b) == ("near", "far"):
            qt = f"The {obj} is {{target}} from the {loc}."
            c1 = f"The {obj} is near the {loc}."
            c2 = f"The {obj} is far from the {loc}."
        else:
            qt = f"The {obj} is {{target}} the {loc}."
            c1 = f"The {obj} is {a} the {loc}."
            c2 = f"The {obj} is {b} the {loc}."
        if not can_score_query(qt, a, b, [c1, c2]):
            rejects["A_token_len"] += 1; continue
        rec = make_record(f"A_{i+1:04d}", "A_spatial_direct", 1.0, c1, c2, qt, a, b, "a", "b", {"object": obj, "location": loc}, counts, "template_pool_vocab", "relation_antonym")
        rec["token_quality"] = frame_token_quality(rec, tok16, tok40)
        frames.append(rec); i += 1

    # Type B: transfer/recipient role, depth 1.
    i = 0; attempts = 0
    while i < per_family and attempts < 20000:
        attempts += 1
        n1, n2 = rng.sample(vocab["names"], 2)
        obj = rng.choice(vocab["objects"])
        verb = rng.choice(TRANSFER_VERBS)
        qt = f"The {obj} was given to {{target}}."
        c1 = f"{n1} {verb} the {obj} to {n2}."
        c2 = f"{n2} {verb} the {obj} to {n1}."
        a, b = n2, n1
        if not can_score_query(qt, a, b, [c1, c2]):
            rejects["B_token_len"] += 1; continue
        rec = make_record(f"B_{i+1:04d}", "B_transfer_role", 1.0, c1, c2, qt, a, b, "a", "b", {"giver1": n1, "giver2": n2, "object": obj, "verb": verb}, counts, "template_pool_vocab", "argument_swap")
        rec["token_quality"] = frame_token_quality(rec, tok16, tok40)
        frames.append(rec); i += 1

    # Type C: attributed report of a binary spatial state, depth 1.5.
    i = 0; attempts = 0
    while i < per_family and attempts < 20000:
        attempts += 1
        name = rng.choice(vocab["names"])
        obj = rng.choice(vocab["objects"])
        loc = rng.choice([x for x in vocab["locations"] if x != obj])
        # Use inside/outside/in/on-like targets. Keep query target single.
        a, b = rng.choice([("inside", "outside"), ("on", "under"), ("above", "below")])
        verb = rng.choice(REPORT_VERBS)
        if (a, b) == ("inside", "outside"):
            c1 = f"{name} {verb} that the {obj} is inside the {loc}."
            c2 = f"{name} {verb} that the {obj} is outside the {loc}."
            qt = f"According to {name}, the {obj} is {{target}} the {loc}."
        else:
            c1 = f"{name} {verb} that the {obj} is {a} the {loc}."
            c2 = f"{name} {verb} that the {obj} is {b} the {loc}."
            qt = f"According to {name}, the {obj} is {{target}} the {loc}."
        if not can_score_query(qt, a, b, [c1, c2]):
            rejects["C_token_len"] += 1; continue
        rec = make_record(f"C_{i+1:04d}", "C_reported_relation", 1.5, c1, c2, qt, a, b, "a", "b", {"speaker": name, "object": obj, "location": loc, "verb": verb}, counts, "template_pool_vocab", "reported_binary_relation")
        rec["token_quality"] = frame_token_quality(rec, tok16, tok40)
        frames.append(rec); i += 1

    # Type D: final state after last event, depth 2.
    i = 0; attempts = 0
    state_event = {
        ("open", "closed"): ("opened", "closed"),
        ("locked", "unlocked"): ("locked", "unlocked"),
        ("empty", "full"): ("emptied", "filled"),
        ("clean", "dirty"): ("cleaned", "dirtied"),
    }
    while i < per_family and attempts < 20000:
        attempts += 1
        name = rng.choice(vocab["names"])
        obj = rng.choice([x for x in vocab["objects"] if x not in {"paper"}])
        a, b = rng.choice(STATE_PAIRS)  # alt_a, alt_b
        ev_a, ev_b = state_event[(a, b)]
        c1 = f"The {obj} was {b}. {name} {ev_a} the {obj}. Then {name} {ev_b} the {obj}."
        # last event is ev_b => state b. To keep C1 correct=a, instead set C1 last ev_a and C2 last ev_b.
        c1 = f"The {obj} was {b}. {name} {ev_b} the {obj}. Then {name} {ev_a} the {obj}."
        c2 = f"The {obj} was {a}. {name} {ev_a} the {obj}. Then {name} {ev_b} the {obj}."
        qt = f"The {obj} is now {{target}}."
        if not can_score_query(qt, a, b, [c1, c2]):
            rejects["D_token_len"] += 1; continue
        rec = make_record(f"D_{i+1:04d}", "D_state_update_order", 2.0, c1, c2, qt, a, b, "a", "b", {"actor": name, "object": obj, "event_a": ev_a, "event_b": ev_b}, counts, "template_pool_vocab", "last_event_final_state")
        rec["token_quality"] = frame_token_quality(rec, tok16, tok40)
        frames.append(rec); i += 1

    meta = {"vocab": vocab, "rejects": dict(rejects)}
    return frames, meta


def attach_controls(frames: list[dict[str, Any]], seed: int, counts, tok16, tok40) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed + 101)
    # Shuffled contexts: deterministic derangement within all frames, preferring different family.
    indices = list(range(len(frames)))
    shuffled = indices[:]
    for _ in range(1000):
        rng.shuffle(shuffled)
        if all(i != j for i, j in zip(indices, shuffled)):
            break
    for i, j in zip(indices, shuffled):
        donor = frames[j]
        frames[i]["controls"] = {
            "shuffled_context_source": donor["frame_id"],
            "shuffled_context1": donor["context1"],
            "shuffled_context2": donor["context2"],
        }
    vocab = choose_vocab(counts, tok16, tok40)
    renamed: list[dict[str, Any]] = []
    rn = vocab["rename_names"] or RENAME_NAMES
    ro = vocab["rename_objects"] or RENAME_OBJECTS
    rl = vocab["rename_locations"] or RENAME_LOCATIONS
    for rec in frames:
        slots = rec.get("slots", {})
        names = [v for k, v in slots.items() if k in {"speaker", "giver1", "giver2", "actor"}]
        objs = [v for k, v in slots.items() if k == "object"]
        locs = [v for k, v in slots.items() if k == "location"]
        name_map = {old: rn[(hash(rec["frame_id"] + old) % len(rn))] for old in names}
        # Ensure distinct renamed names for role targets if two names are present.
        if len(names) >= 2 and len(rn) >= 2:
            name_map[names[0]] = rn[0]
            name_map[names[1]] = rn[1] if rn[1] != rn[0] else rn[2 % len(rn)]
        obj_map = {old: ro[(hash("obj" + rec["frame_id"] + old) % len(ro))] for old in objs}
        loc_map = {old: rl[(hash("loc" + rec["frame_id"] + old) % len(rl))] for old in locs}
        r = apply_renaming(rec, name_map, obj_map, loc_map)
        r["token_quality"] = frame_token_quality(r, tok16, tok40)
        renamed.append(r)
        rec.setdefault("controls", {})["renamed_frame_id"] = r["frame_id"]
    return renamed, {"renamed_count": len(renamed)}


def summarize(frames: list[dict[str, Any]], renamed: list[dict[str, Any]], corpus_meta: dict[str, Any], build_meta: dict[str, Any]) -> dict[str, Any]:
    fam = collections.Counter(r["family"] for r in frames)
    depths = collections.Counter(str(r["depth"]) for r in frames)
    prior_ratios = [r["target_prior"]["max_min_ratio"] for r in frames if math.isfinite(r["target_prior"]["max_min_ratio"])]
    balanced = sum(1 for r in frames if r["target_prior"].get("balanced_ratio_le_4"))
    examples = {k: [] for k in fam}
    for r in frames:
        if len(examples[r["family"]]) < 3:
            examples[r["family"]].append({k: r[k] for k in ["frame_id", "context1", "context2", "query_template", "alt_a", "alt_b"]})
    return {
        "status": "FROZEN",
        "created_utc": now_utc(),
        "scientific_object": "crossed-sign context-conditioned alternative binding micro-world; no model scores used in construction",
        "pool_path": rel(POOL_PATH),
        "pool_sha256": sha256_path(POOL_PATH),
        "legal16k_tokenizer": rel(LEGAL16K_TOK),
        "legal40k_tokenizer": rel(LEGAL40K_TOK),
        "corpus_meta": corpus_meta,
        "build_meta": build_meta,
        "n_frames": len(frames),
        "n_renamed_controls": len(renamed),
        "family_counts": dict(fam),
        "depth_counts": dict(depths),
        "target_prior_balanced_ratio_le_4": {"n": balanced, "fraction": balanced / len(frames) if frames else None},
        "target_prior_ratio_summary": {
            "n": len(prior_ratios),
            "median": statistics.median(prior_ratios) if prior_ratios else None,
            "mean": statistics.fmean(prior_ratios) if prior_ratios else None,
            "max": max(prior_ratios) if prior_ratios else None,
        },
        "example_frames": examples,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def write_md(summary: dict[str, Any], frame_path: Path, renamed_path: Path, manifest_path: Path, content_sha: str) -> None:
    lines = []
    lines.append("# research frozen counterfactual micro-world")
    lines.append("")
    lines.append(f"Status: **{summary['status']}**")
    lines.append("")
    lines.append("This file freezes a no-training diagnostic object before checkpoint scoring. It is designed to measure crossed-sign context-conditioned alternative binding rather than coherent-vs-corrupted text plausibility.")
    lines.append("")
    lines.append("## Construction")
    lines.append(f"- Pool: `{summary['pool_path']}`")
    lines.append(f"- Pool SHA256: `{summary['pool_sha256']}`")
    lines.append(f"- Legal16k tokenizer: `{summary['legal16k_tokenizer']}`")
    lines.append(f"- Legal40k tokenizer: `{summary['legal40k_tokenizer']}`")
    lines.append(f"- Frames: **{summary['n_frames']}**")
    lines.append(f"- Renaming controls: **{summary['n_renamed_controls']}**")
    lines.append(f"- Family counts: `{summary['family_counts']}`")
    lines.append(f"- Depth counts: `{summary['depth_counts']}`")
    lines.append(f"- Content SHA256: `{content_sha}`")
    lines.append("")
    lines.append("## Target-prior balance")
    lines.append(f"- Ratio ≤4: {summary['target_prior_balanced_ratio_le_4']['n']} / {summary['n_frames']} ({summary['target_prior_balanced_ratio_le_4']['fraction']:.3f})")
    lines.append(f"- Ratio summary: `{summary['target_prior_ratio_summary']}`")
    lines.append("")
    lines.append("## Why A02 v1 is not duplicated")
    lines.append("A02's structural contrast v1 scored coherent-vs-perturbed text margins, ranked chck_77M rather than chck_82M, and contained punctuation/template artifacts. This object instead renders two contexts, one query, two alternatives, and requires Δ1 and Δ2 to have opposite correct signs. It includes no model results in construction.")
    lines.append("")
    lines.append("## Example frames")
    for fam, exs in summary["example_frames"].items():
        lines.append(f"### {fam}")
        for ex in exs:
            lines.append(f"- `{ex['frame_id']}` C1: {ex['context1']} C2: {ex['context2']} Query: {ex['query_template']} Alts: {ex['alt_a']} / {ex['alt_b']}")
    lines.append("")
    lines.append("## Files")
    lines.append(f"- Frames JSONL: `{rel(frame_path)}`")
    lines.append(f"- Renamed controls JSONL: `{rel(renamed_path)}`")
    lines.append(f"- Manifest JSON: `{rel(manifest_path)}`")
    _public_path('research/notes/representation_and_objectives').mkdir(parents=True, exist_ok=True)
    NOTE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=184)
    ap.add_argument("--per-family", type=int, default=120)
    ap.add_argument("--max-pool-rows", type=int, default=0)
    args = ap.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"event": "load_tokenizers", "tok16": rel(LEGAL16K_TOK), "tok40": rel(LEGAL40K_TOK)}), flush=True)
    tok16 = AutoTokenizer.from_pretrained(str(LEGAL16K_TOK), use_fast=True, trust_remote_code=True)
    tok40 = AutoTokenizer.from_pretrained(str(LEGAL40K_TOK), use_fast=True, trust_remote_code=True)
    print(json.dumps({"event": "count_corpus", "pool": rel(POOL_PATH)}), flush=True)
    counts, corpus_meta = corpus_counts(POOL_PATH, max_rows=args.max_pool_rows)
    frames, build_meta = build_frames(counts, tok16, tok40, seed=args.seed, per_family=args.per_family)
    renamed, control_meta = attach_controls(frames, seed=args.seed, counts=counts, tok16=tok16, tok40=tok40)
    build_meta["control_meta"] = control_meta
    # Final deterministic ordering by frame_id; do not randomize after controls are attached.
    frames = sorted(frames, key=lambda r: r["frame_id"])
    renamed = sorted(renamed, key=lambda r: r["frame_id"])
    frame_path = _public_path('experiments/archive/representation_and_objectives/data/counterfactual_micro_world/counterfactual_micro_world_frames.jsonl')
    renamed_path = _public_path('experiments/archive/representation_and_objectives/data/counterfactual_micro_world/counterfactual_micro_world_renamed_controls.jsonl')
    manifest_path = _public_path('experiments/archive/representation_and_objectives/data/counterfactual_micro_world/counterfactual_micro_world_manifest.json')
    write_jsonl(frame_path, frames)
    write_jsonl(renamed_path, renamed)
    content = {"frames": frames, "renamed_controls": renamed}
    content_sha = sha256_obj(content)
    summary = summarize(frames, renamed, corpus_meta, build_meta)
    summary["content_sha256"] = content_sha
    summary["frame_path"] = rel(frame_path)
    summary["renamed_control_path"] = rel(renamed_path)
    summary["manifest_path"] = rel(manifest_path)
    summary["frame_file_sha256"] = sha256_path(frame_path)
    summary["renamed_file_sha256"] = sha256_path(renamed_path)
    manifest_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    write_md(summary, frame_path, renamed_path, manifest_path, content_sha)
    print(json.dumps({"status": "FROZEN", "n_frames": len(frames), "n_renamed": len(renamed), "manifest": rel(manifest_path), "note": rel(NOTE_PATH), "content_sha256": content_sha}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
