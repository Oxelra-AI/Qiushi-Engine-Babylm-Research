#!/usr/bin/env python3
"""research: static repair of the research counterfactual micro-world.

The research v2 object was frozen before any checkpoint scoring.  research static
inspection found a remaining surface artifact in v2: the near/far subtype used a
single query template "{target} from", which makes the near alternative
ungrammatical ("near from").  This script creates v3 BEFORE any checkpoint is
scored.  The repair is purely linguistic/static: remove near/far, keep only
alternative pairs that fit one identical query frame, and make renamed controls
respect subtype-compatible slot classes.

No model scores are read or produced by this script.
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
TOK16 = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M')
TOK40 = _public_path('experiments/archive/representation_and_objectives/training/runs/legal40k_accum_compact_view_reinvest_seed43022/hf_model/chck_100M')
OUT_ROOT = _public_path('experiments/archive/representation_and_objectives/data/counterfactual_micro_world_v3')
NOTE_PATH = _public_path('research/notes/representation_and_objectives/counterfactual_micro_world_v3_static_repair.md')
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’-]*")
PUNCT = string.punctuation + "“”‘’«»‹›"

NAMES = ["Alice", "Bob", "Sam", "Tom", "Kate", "Amy", "Ben", "Lucy", "Jack", "Emma", "Anna", "John", "Mary", "Seth", "Rose", "Max"]
PORTABLE_OBJECTS = ["book", "key", "coin", "cup", "hat", "bag", "ball", "card", "sock", "shoe", "box", "bottle", "letter", "toy", "apple", "glass", "map", "ring", "pen", "paper"]
SURFACES = ["table", "bed", "desk", "shelf", "chair", "counter"]
ABOVE_BELOW_REFERENTS = ["table", "bed", "desk", "shelf", "floor", "chair", "counter"]
CONTAINERS = ["box", "basket", "closet", "cupboard", "drawer", "bag", "room", "house", "shop"]
TRANSFER_OBJECTS = ["book", "key", "coin", "cup", "hat", "bag", "ball", "card", "letter", "toy", "apple", "map", "ring", "pen", "paper"]
TRANSFER_VERBS = ["gave", "handed", "passed", "sent"]
REPORT_VERBS = ["said", "thought", "believed", "claimed"]
STATE_OBJECTS = {
    ("open", "closed"): ["box", "door", "window", "drawer", "bag"],
    ("locked", "unlocked"): ["door", "gate", "box", "drawer"],
    ("empty", "full"): ["cup", "bottle", "box", "bag", "glass"],
}
STATE_EVENTS = {
    ("open", "closed"): ("opened", "closed"),
    ("locked", "unlocked"): ("locked", "unlocked"),
    ("empty", "full"): ("emptied", "filled"),
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: Path | str) -> str:
    pp = Path(p)
    try:
        return str(pp.resolve().relative_to(USER_ROOT))
    except Exception:
        return str(pp)


def sha_path(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_obj(o: Any) -> str:
    return hashlib.sha256(json.dumps(o, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def norm(w: str) -> str:
    return w.strip(PUNCT).lower().replace("’", "'")


def count_pool(path: Path) -> tuple[collections.Counter[str], dict[str, Any]]:
    c: collections.Counter[str] = collections.Counter()
    src = collections.Counter()
    rows = 0
    words = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rows += 1
            src[obj.get("source") or "unknown"] += 1
            for m in WORD_RE.finditer(obj.get("text", "")):
                w = norm(m.group(0))
                if w:
                    c[w] += 1
                    words += 1
    return c, {"rows": rows, "word_tokens_counted": words, "source_counts": dict(src.most_common())}


def token_span_len(tok, sentence: str, span: tuple[int, int]) -> int:
    enc = tok(sentence, return_offsets_mapping=True, add_special_tokens=True)
    s, e = span
    n = 0
    for a, b in enc["offset_mapping"]:
        if a == b == 0:
            continue
        if b > s and a < e:
            n += 1
    return n


def render(context: str, query_template: str, target: str) -> tuple[str, tuple[int, int]]:
    pre, post = query_template.split("{target}", 1)
    q = pre + target + post
    lead = (context.strip() + " ") if context.strip() else ""
    sent = lead + q
    return sent, (len(lead) + len(pre), len(lead) + len(pre) + len(target))


def scoreable(tok16, tok40, query: str, a: str, b: str, c1: str, c2: str) -> tuple[bool, dict[str, int]]:
    q: dict[str, int] = {}
    for ck, c in [("c1", c1), ("c2", c2), ("none", "")]:
        for ak, t in [("a", a), ("b", b)]:
            sent, span = render(c, query, t)
            q[f"{ck}_{ak}_tok16"] = token_span_len(tok16, sent, span)
            q[f"{ck}_{ak}_tok40"] = token_span_len(tok40, sent, span)
    ok = all(v == 1 for v in q.values())
    return ok, q


def word_ok(w: str, counts: collections.Counter[str], tok16, tok40, min_count: int = 1) -> bool:
    return counts[w.lower()] >= min_count and len(tok16(" " + w, add_special_tokens=False)["input_ids"]) <= 2 and len(tok40(" " + w, add_special_tokens=False)["input_ids"]) <= 2


def filtered(xs: list[str], counts, tok16, tok40, min_count=1) -> list[str]:
    return [x for x in xs if word_ok(x, counts, tok16, tok40, min_count)]


def prior(counts, a: str, b: str) -> dict[str, Any]:
    ca, cb = counts[a.lower()], counts[b.lower()]
    ratio = math.inf if min(ca, cb) == 0 else max(ca, cb) / min(ca, cb)
    return {"a_count": ca, "b_count": cb, "max_min_ratio": ratio, "balanced_ratio_le_4": bool(ratio <= 4.0)}


def rec(fid: str, fam: str, subtype: str, depth: float, c1: str, c2: str, query: str, a: str, b: str, slots: dict[str, str], counts, tq: dict[str, int]) -> dict[str, Any]:
    return {
        "frame_id": fid,
        "family": fam,
        "subtype": subtype,
        "depth": depth,
        "context1": c1,
        "context2": c2,
        "query_template": query,
        "alt_a": a,
        "alt_b": b,
        "correct_c1": "a",
        "correct_c2": "b",
        "slots": slots,
        "target_prior": prior(counts, a, b),
        "token_quality": tq,
        "source": "hand_cleaned_template_pool_vocab_v3_static_repair",
        "construction_note": "slot words occur in the legal pool; alternatives fit one identical query syntax; target spans are one token under legal16k and legal40k",
    }


def build(counts, tok16, tok40, seed: int, per_family: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    names = filtered(NAMES, counts, tok16, tok40, 1)
    portable = filtered(PORTABLE_OBJECTS, counts, tok16, tok40, 2)
    surfaces = filtered(SURFACES, counts, tok16, tok40, 2)
    above_refs = filtered(ABOVE_BELOW_REFERENTS, counts, tok16, tok40, 2)
    containers = filtered(CONTAINERS, counts, tok16, tok40, 2)
    transfer_objs = filtered(TRANSFER_OBJECTS, counts, tok16, tok40, 2)
    state_objs = {k: filtered(v, counts, tok16, tok40, 1) for k, v in STATE_OBJECTS.items()}
    vocab = {
        "names": names,
        "portable_objects": portable,
        "surfaces_for_on_under": surfaces,
        "referents_for_above_below": above_refs,
        "containers_for_inside_outside": containers,
        "transfer_objects": transfer_objs,
        "state_objects": {str(k): v for k, v in state_objs.items()},
    }
    frames: list[dict[str, Any]] = []
    rej = collections.Counter()
    seen_text = set()

    def add(r: dict[str, Any]) -> bool:
        sig = (r["family"], r["context1"], r["context2"], r["query_template"], r["alt_a"], r["alt_b"])
        if sig in seen_text:
            rej["duplicate"] += 1
            return False
        seen_text.add(sig)
        frames.append(r)
        return True

    # A: direct relation.  v3 removes near/far because it cannot share a grammatical
    # one-token slot with one query template ("near from" was a static artifact).
    i = 0
    attempts = 0
    while i < per_family and attempts < 50000:
        attempts += 1
        obj = rng.choice(portable)
        kind = rng.choice(["inside_outside", "on_under", "above_below"])
        if kind == "inside_outside":
            loc = rng.choice([x for x in containers if x != obj])
            a, b = "inside", "outside"
            c1 = f"The {obj} is inside the {loc}."
            c2 = f"The {obj} is outside the {loc}."
            q = f"The {obj} is {{target}} the {loc}."
        elif kind == "on_under":
            loc = rng.choice([x for x in surfaces if x != obj])
            a, b = "on", "under"
            c1 = f"The {obj} is on the {loc}."
            c2 = f"The {obj} is under the {loc}."
            q = f"The {obj} is {{target}} the {loc}."
        else:
            loc = rng.choice([x for x in above_refs if x != obj])
            a, b = "above", "below"
            c1 = f"The {obj} is above the {loc}."
            c2 = f"The {obj} is below the {loc}."
            q = f"The {obj} is {{target}} the {loc}."
        ok, tq = scoreable(tok16, tok40, q, a, b, c1, c2)
        if not ok:
            rej["A_token"] += 1
            continue
        if add(rec(f"A_{i+1:04d}", "A_spatial_direct", kind, 1.0, c1, c2, q, a, b, {"object": obj, "place": loc}, counts, tq)):
            i += 1

    # B: transfer recipient; query compatible with all transfer verbs.
    i = 0
    attempts = 0
    while i < per_family and attempts < 50000:
        attempts += 1
        n1, n2 = rng.sample(names, 2)
        obj = rng.choice(transfer_objs)
        verb = rng.choice(TRANSFER_VERBS)
        c1 = f"{n1} {verb} the {obj} to {n2}."
        c2 = f"{n2} {verb} the {obj} to {n1}."
        q = f"The recipient of the {obj} was {{target}}."
        a, b = n2, n1
        ok, tq = scoreable(tok16, tok40, q, a, b, c1, c2)
        if not ok:
            rej["B_token"] += 1
            continue
        if add(rec(f"B_{i+1:04d}", "B_transfer_role", verb, 1.0, c1, c2, q, a, b, {"agent_c1": n1, "recipient_c1": n2, "object": obj, "verb": verb}, counts, tq)):
            i += 1

    # C: report/attribution; relation must be retrieved through the reported speaker.
    i = 0
    attempts = 0
    while i < per_family and attempts < 50000:
        attempts += 1
        name = rng.choice(names)
        obj = rng.choice(portable)
        kind = rng.choice(["inside_outside", "on_under", "above_below"])
        if kind == "inside_outside":
            loc = rng.choice([x for x in containers if x != obj])
            a, b = "inside", "outside"
        elif kind == "on_under":
            loc = rng.choice([x for x in surfaces if x != obj])
            a, b = "on", "under"
        else:
            loc = rng.choice([x for x in above_refs if x != obj])
            a, b = "above", "below"
        verb = rng.choice(REPORT_VERBS)
        c1 = f"{name} {verb} that the {obj} is {a} the {loc}."
        c2 = f"{name} {verb} that the {obj} is {b} the {loc}."
        q = f"According to {name}, the {obj} is {{target}} the {loc}."
        ok, tq = scoreable(tok16, tok40, q, a, b, c1, c2)
        if not ok:
            rej["C_token"] += 1
            continue
        if add(rec(f"C_{i+1:04d}", "C_reported_relation", kind, 1.5, c1, c2, q, a, b, {"speaker": name, "object": obj, "place": loc, "verb": verb}, counts, tq)):
            i += 1

    # D: noncommuting final-state update.
    i = 0
    attempts = 0
    state_keys = [k for k, xs in state_objs.items() if xs]
    while i < per_family and attempts < 50000:
        attempts += 1
        name = rng.choice(names)
        a, b = rng.choice(state_keys)
        obj = rng.choice(state_objs[(a, b)])
        ev_a, ev_b = STATE_EVENTS[(a, b)]
        c1 = f"The {obj} was {b}. {name} {ev_b} the {obj}. Then {name} {ev_a} the {obj}."
        c2 = f"The {obj} was {a}. {name} {ev_a} the {obj}. Then {name} {ev_b} the {obj}."
        q = f"The {obj} is now {{target}}."
        ok, tq = scoreable(tok16, tok40, q, a, b, c1, c2)
        if not ok:
            rej["D_token"] += 1
            continue
        if add(rec(f"D_{i+1:04d}", "D_state_update_order", f"{a}_{b}", 2.0, c1, c2, q, a, b, {"actor": name, "object": obj, "state_a": a, "state_b": b, "event_a": ev_a, "event_b": ev_b}, counts, tq)):
            i += 1

    got = collections.Counter(r["family"] for r in frames)
    want = {"A_spatial_direct": per_family, "B_transfer_role": per_family, "C_reported_relation": per_family, "D_state_update_order": per_family}
    if got != want:
        raise RuntimeError(f"failed to build balanced frame set: got={got} want={want} rejects={rej}")
    return frames, {"vocab": vocab, "rejects": dict(rej), "seen_text_signatures": len(seen_text), "static_repairs_vs_v2": ["removed near_far subtype; no shared grammatical one-slot query exists for near/far", "renaming controls preserve subtype-compatible slot classes"]}


def compatible_place_choices(r: dict[str, Any], counts, tok16, tok40) -> list[str]:
    subtype = r.get("subtype")
    old = r.get("slots", {}).get("place")
    obj = r.get("slots", {}).get("object")
    if subtype == "inside_outside":
        raw = CONTAINERS
    elif subtype == "on_under":
        raw = SURFACES
    elif subtype == "above_below":
        raw = ABOVE_BELOW_REFERENTS
    else:
        raw = SURFACES + CONTAINERS
    xs = filtered(raw, counts, tok16, tok40, 1)
    return [x for x in xs if x != old and x != obj] or xs


def compatible_object_choices(r: dict[str, Any], counts, tok16, tok40) -> list[str]:
    old = r.get("slots", {}).get("object")
    fam = r.get("family")
    if fam == "D_state_update_order":
        state_a = r.get("slots", {}).get("state_a")
        state_b = r.get("slots", {}).get("state_b")
        raw = STATE_OBJECTS.get((state_a, state_b), PORTABLE_OBJECTS)
    elif fam == "B_transfer_role":
        raw = TRANSFER_OBJECTS
    else:
        raw = PORTABLE_OBJECTS
    xs = filtered(raw, counts, tok16, tok40, 1)
    return [x for x in xs if x != old] or xs


def rename_frames(frames: list[dict[str, Any]], counts, tok16, tok40, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed + 700)
    rn = filtered(NAMES, counts, tok16, tok40, 1)
    out = []
    for r in frames:
        slots = r.get("slots", {})
        mapping: dict[str, str] = {}
        old_names: list[str] = []
        for k in ["agent_c1", "recipient_c1", "speaker", "actor"]:
            if slots.get(k):
                old_names.append(slots[k])
        unique_names = list(dict.fromkeys(old_names))
        name_choices = [x for x in rn if x not in unique_names]
        if len(name_choices) < len(unique_names):
            name_choices = rn
        for old, new in zip(unique_names, rng.sample(name_choices, min(len(unique_names), len(name_choices)))):
            mapping[old] = new
        if slots.get("object"):
            choices = compatible_object_choices(r, counts, tok16, tok40)
            mapping[slots["object"]] = rng.choice(choices)
        if slots.get("place"):
            choices = [x for x in compatible_place_choices(r, counts, tok16, tok40) if x not in mapping.values()]
            if not choices:
                choices = compatible_place_choices(r, counts, tok16, tok40)
            mapping[slots["place"]] = rng.choice(choices)

        def sub(text: str) -> str:
            z = text
            for old, new in sorted(mapping.items(), key=lambda kv: -len(kv[0])):
                z = re.sub(rf"\b{re.escape(old)}\b", new, z)
            return z

        rr = json.loads(json.dumps(r))
        rr["frame_id"] = r["frame_id"] + "__renamed"
        rr["renaming_of"] = r["frame_id"]
        rr["renaming_map"] = mapping
        for key in ["context1", "context2", "query_template", "alt_a", "alt_b"]:
            rr[key] = sub(rr[key])
        ok, tq = scoreable(tok16, tok40, rr["query_template"], rr["alt_a"], rr["alt_b"], rr["context1"], rr["context2"])
        if not ok:
            raise RuntimeError(f"renamed frame not scoreable: {rr['frame_id']} {tq}")
        # Minimal additional static guards against the exact v2 bug and category-breaking renames.
        bad = ["near from", "far the ", "open ring", "closed ring", "full door", "empty door"]
        text_join = " ".join([rr["context1"], rr["context2"], rr["query_template"]]).lower()
        if any(b in text_join for b in bad):
            raise RuntimeError(f"bad renamed frame surface: {rr['frame_id']} {text_join}")
        rr["token_quality"] = tq
        rr["source"] = r["source"] + "+renaming_control_subtype_preserving"
        out.append(rr)
    return out


def attach_shuffled(frames: list[dict[str, Any]], seed: int) -> None:
    fam_order = ["A_spatial_direct", "B_transfer_role", "C_reported_relation", "D_state_update_order"]
    by_fam = {fam: [i for i, r in enumerate(frames) if r["family"] == fam] for fam in fam_order}
    sizes = {fam: len(v) for fam, v in by_fam.items()}
    if len(set(sizes.values())) != 1 or next(iter(sizes.values())) == 0:
        raise RuntimeError(f"family-balanced frames required for shuffled control: {sizes}")
    rng = random.Random(seed + 1300)
    for fam in fam_order:
        rng.shuffle(by_fam[fam])
    donor_map: dict[int, int] = {}
    for fi, fam in enumerate(fam_order):
        donor_fam = fam_order[(fi + 1) % len(fam_order)]
        for i, j in zip(by_fam[fam], by_fam[donor_fam]):
            donor_map[i] = j
    if set(donor_map) != set(range(len(frames))):
        raise RuntimeError("incomplete shuffled donor map")
    for i, j in sorted(donor_map.items()):
        frames[i]["controls"] = {
            "shuffled_context_source": frames[j]["frame_id"],
            "shuffled_context_family": frames[j]["family"],
            "shuffled_context1": frames[j]["context1"],
            "shuffled_context2": frames[j]["context2"],
        }


def static_quality(frames: list[dict[str, Any]], renamed: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    fam = collections.Counter(r["family"] for r in frames)
    subtype = collections.Counter(r["subtype"] for r in frames)
    depth = collections.Counter(str(r["depth"]) for r in frames)
    pri = [r["target_prior"]["max_min_ratio"] for r in frames if math.isfinite(r["target_prior"]["max_min_ratio"])]
    balanced = sum(1 for r in frames if r["target_prior"].get("balanced_ratio_le_4"))
    tq_bad = [r["frame_id"] for r in frames if any(v != 1 for v in r["token_quality"].values())]
    all_text = "\n".join(" ".join([r["context1"], r["context2"], r["query_template"]]) for r in frames + (renamed or []))
    artifact_phrases = ["near from", "far the ", "near to", "far to"]
    artifacts = {p: all_text.lower().count(p) for p in artifact_phrases if all_text.lower().count(p)}
    return {
        "n_frames": len(frames),
        "family_counts": dict(fam),
        "subtype_counts": dict(subtype),
        "depth_counts": dict(depth),
        "token_quality_all_single": len(tq_bad) == 0,
        "token_quality_bad_ids": tq_bad[:20],
        "target_prior_balanced_le4": {"n": balanced, "fraction": balanced / len(frames) if frames else None},
        "target_prior_ratio": {"n": len(pri), "median": statistics.median(pri) if pri else None, "mean": statistics.fmean(pri) if pri else None, "max": max(pri) if pri else None},
        "surface_artifact_phrase_counts": artifacts,
        "example_frames_by_family": {f: [{k: r[k] for k in ["frame_id", "context1", "context2", "query_template", "alt_a", "alt_b", "subtype"]} for r in frames if r["family"] == f][:4] for f in fam},
    }


def write_jsonl(p: Path, rows: list[dict[str, Any]]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n")


def write_note(summary: dict[str, Any]) -> None:
    q = summary["static_quality"]
    lines = [
        "# research counterfactual micro-world v3 static repair",
        "",
        "Status: **FROZEN_V3_STATIC_REPAIR_BEFORE_SCORING**",
        "",
        "This replaces the research v2 object before any checkpoint scoring. Static inspection found that the v2 `near_far` subtype used a single query frame with `{target} from`, which makes the `near` alternative ungrammatical (`near from`). v3 removes near/far and also makes renamed controls preserve subtype-compatible slot classes, so the measure is less likely to score surface artifacts rather than crossed-sign context-conditioned alternative binding.",
        "",
        "## Files",
        f"- Frames: `{summary['frame_path']}`",
        f"- Renamed controls: `{summary['renamed_path']}`",
        f"- Manifest: `{summary['manifest_path']}`",
        f"- Content SHA256: `{summary['content_sha256']}`",
        f"- Frame file SHA256: `{summary['frame_file_sha256']}`",
        f"- Renamed file SHA256: `{summary['renamed_file_sha256']}`",
        "",
        "## Static quality",
        f"- Frames: {q['n_frames']}",
        f"- Family counts: `{q['family_counts']}`",
        f"- Depth counts: `{q['depth_counts']}`",
        f"- Subtype counts: `{q['subtype_counts']}`",
        f"- All target spans one token under both tokenizers: {q['token_quality_all_single']}",
        f"- Target prior ratio ≤4: {q['target_prior_balanced_le4']['n']} / {q['n_frames']} ({q['target_prior_balanced_le4']['fraction']:.3f})",
        f"- Target prior ratio summary: `{q['target_prior_ratio']}`",
        f"- Surface artifact phrase counts: `{q['surface_artifact_phrase_counts']}`",
        "",
        "## Examples",
    ]
    for fam, exs in q["example_frames_by_family"].items():
        lines.append(f"### {fam}")
        for e in exs:
            lines.append(f"- `{e['frame_id']}` ({e['subtype']}): C1={e['context1']} C2={e['context2']} Query={e['query_template']} Alts={e['alt_a']}/{e['alt_b']}")
    lines.extend([
        "",
        "## Scoring interpretation fixed before model runs",
        "For every frame, the scorer computes Δ1=s(C1,Q,a)-s(C1,Q,b), Δ2=s(C2,Q,a)-s(C2,Q,b). Crossed-sign success requires Δ1>0 and Δ2<0. Context-removed and shuffled-context controls use the same frozen alternatives and query; renamed controls are separate frozen records. Candidate-prior-balanced and imbalanced subsets must be reported separately. this measure must not be tuned to rank chck_82M highly; the first two-checkpoint run is only a mechanical coherence/control check, and interpretation requires extension to distinct legal trajectories plus the MLM-only/coupled aligned/coupled shuffled panel.",
    ])
    _public_path('research/notes/representation_and_objectives').mkdir(parents=True, exist_ok=True)
    NOTE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1842)
    ap.add_argument("--per-family", type=int, default=100)
    args = ap.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"event": "load_tokenizers", "tok16": rel(TOK16), "tok40": rel(TOK40)}), flush=True)
    tok16 = AutoTokenizer.from_pretrained(str(TOK16), use_fast=True, trust_remote_code=True)
    tok40 = AutoTokenizer.from_pretrained(str(TOK40), use_fast=True, trust_remote_code=True)
    print(json.dumps({"event": "count_pool", "pool": rel(POOL_PATH)}), flush=True)
    counts, corpus_meta = count_pool(POOL_PATH)
    frames, build_meta = build(counts, tok16, tok40, args.seed, args.per_family)
    frames.sort(key=lambda r: r["frame_id"])
    attach_shuffled(frames, args.seed)
    renamed = rename_frames(frames, counts, tok16, tok40, args.seed)
    renamed.sort(key=lambda r: r["frame_id"])
    frame_path = _public_path('experiments/archive/representation_and_objectives/data/counterfactual_micro_world_v3/counterfactual_micro_world_v3_frames.jsonl')
    ren_path = _public_path('experiments/archive/representation_and_objectives/data/counterfactual_micro_world_v3/counterfactual_micro_world_v3_renamed_controls.jsonl')
    man_path = _public_path('experiments/archive/representation_and_objectives/data/counterfactual_micro_world_v3/counterfactual_micro_world_v3_manifest.json')
    write_jsonl(frame_path, frames)
    write_jsonl(ren_path, renamed)
    content_sha = sha_obj({"frames": frames, "renamed": renamed})
    summary = {
        "status": "FROZEN_V3_STATIC_REPAIR_BEFORE_SCORING",
        "created_utc": now(),
        "seed": args.seed,
        "per_family": args.per_family,
        "scientific_object": "crossed-sign context-conditioned alternative binding micro-world",
        "reason_v3_replaced_v2": "static inspection found v2 near/far query artifact ('near from') before any checkpoint scoring; renamed controls also needed subtype-compatible slot classes",
        "pool_path": rel(POOL_PATH),
        "pool_sha256": sha_path(POOL_PATH),
        "legal16k_tokenizer": rel(TOK16),
        "legal40k_tokenizer": rel(TOK40),
        "corpus_meta": corpus_meta,
        "build_meta": build_meta,
        "static_quality": static_quality(frames, renamed),
        "n_renamed_controls": len(renamed),
        "content_sha256": content_sha,
        "frame_path": rel(frame_path),
        "renamed_path": rel(ren_path),
        "manifest_path": rel(man_path),
        "frame_file_sha256": sha_path(frame_path),
        "renamed_file_sha256": sha_path(ren_path),
    }
    man_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    write_note(summary)
    print(json.dumps({"status": summary["status"], "frames": len(frames), "renamed": len(renamed), "content_sha256": content_sha, "manifest": rel(man_path), "note": rel(NOTE_PATH)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
