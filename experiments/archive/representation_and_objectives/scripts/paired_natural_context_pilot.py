#!/usr/bin/env python3
"""research: high-precision paired natural context / causal-transform pilot.

This is an Explore-stage readout, not training. It answers the strategist's
concern about the research cue pool: alternatives here are not chosen from local
scale1.75 logits and the sample is drawn after scanning the full legal corpus.

Two related natural objects are built from the legal compact-view 10M corpus:

1. entity_exchange: one-token alternatives A/B from natural comparative or
   spatial sentences.  Contexts are paired causal transformations, e.g.
      [MASK] are more sensitive than adults -> children
      [MASK] are less sensitive than children -> adults
   The fixed alternatives A,B should reverse ordering between the two contexts.

2. relation_word: fixed inverse relation alternatives from the same natural
   sentence, e.g.
      X is [MASK] Y : above vs below
      Y is [MASK] X : below vs above
   This is easier and mainly checks whether relation-token ordering is already
   saturated.

The script compares scale1.75 80M with its matched legal16k base at 80M using
single-mask logits. It writes row-level CSVs and a summary suitable for deciding
whether this source contains a real natural competition signal before any H100
training is considered.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import json
import os
import pathlib
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
OUT = ROOT / "experiments/archive/representation_and_objectives/data/paired_natural_context_pilot"
NOTE = ROOT / "research/notes/representation_and_objectives/paired_natural_context_pilot.md"
CACHE = OUT / "hf_cache"
for key, sub in {
    "HF_HOME": "hf_home",
    "HF_HUB_CACHE": "hf_home/hub",
    "HUGGINGFACE_HUB_CACHE": "hf_home/hub",
    "TRANSFORMERS_CACHE": "transformers",
    "HF_MODULES_CACHE": "modules",
    "HF_DATASETS_CACHE": "datasets",
}.items():
    p = CACHE / sub
    p.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(p.resolve())
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
from transformers import AutoModelForMaskedLM, AutoTokenizer  # noqa: E402

CORPUS = ROOT / "experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"
SCALE80 = ROOT / "experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M80M_seed43022/hf_model/chck_80M"
BASE80 = ROOT / "experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_80M"

WORD = r"[A-Za-z][A-Za-z'\-]{2,}"
WORD_RE = re.compile(WORD)
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
STOP = set("""
a an the and or but if then than as at by for from in into is it its of on onto over under to with without within across after before during about above below between through this that these those there here when where while who whom whose which what why how be been being are was were will would should could can may might must do does did doing done have has had having you your we our they their he his she her i my me mine us them him himself herself itself themselves not no nor so such very more most less least other another same just only own out up down off back much many any each every some all both either neither because although though since until like also well just actually really probably maybe there here now then yes yeah okay ok uh um www xxx gonna wanna gotta
""".split())
PRONOUNS = set("i you he she we they me him her us them it this that these those someone somebody anyone anybody everyone everybody".split())

INVERSE = {
    "more": "less", "less": "more",
    "higher": "lower", "lower": "higher",
    "larger": "smaller", "smaller": "larger",
    "bigger": "smaller", "shorter": "longer", "longer": "shorter",
    "older": "younger", "younger": "older",
    "heavier": "lighter", "lighter": "heavier",
    "hotter": "colder", "colder": "hotter", "warmer": "cooler", "cooler": "warmer",
    "above": "below", "below": "above",
    "left": "right", "right": "left",
    "before": "after", "after": "before",
    "behind": "front", "front": "behind",
}
SCALAR_RELS = "more|less|higher|lower|larger|smaller|bigger|shorter|longer|older|younger|heavier|lighter|hotter|colder|warmer|cooler"
SPATIAL_RELS = "above|below"
LATERAL_RELS = "left|right"
TEMPORAL_RELS = "before|after"

# Comparatives: A verb REL property than B. A and B are deliberately restricted to
# one lexical word to keep candidate scoring exact; property may contain several
# words up to the explicit 'than'.
COMPARATIVE_RE = re.compile(
    rf"\b(?P<a>{WORD})\s+(?P<verb>is|are|was|were|becomes?|became|seems?|appears?)\s+(?P<rel>{SCALAR_RELS})\s+(?P<prop>[A-Za-z][A-Za-z'\-]*(?:\s+(?!than\b)[A-Za-z][A-Za-z'\-]*){{0,5}})\s+than\s+(?P<b>{WORD})\b",
    re.I,
)
# Spatial subject/object one-token alternatives.
SPATIAL_ENTITY_RE = re.compile(
    rf"\b(?P<a>{WORD})\s+(?P<verb>is|are|was|were|lies|lie|sits|sit|stands|stand|stood|stays|stay|located)\s+(?P<rel>{SPATIAL_RELS})\s+(?P<b>{WORD})\b",
    re.I,
)
# Lateral with explicit 'of': A is left/right of B.
LATERAL_ENTITY_RE = re.compile(
    rf"\b(?P<a>{WORD})\s+(?P<verb>is|are|was|were|lies|lie|sits|sit|stands|stand|located)\s+(?P<rel>{LATERAL_RELS})\s+of\s+(?P<b>{WORD})\b",
    re.I,
)
# Temporal simple A before/after B, used as held-out/noisy family; no training recommendation will depend on it.
TEMPORAL_ENTITY_RE = re.compile(rf"\b(?P<a>{WORD})\s+(?P<rel>{TEMPORAL_RELS})\s+(?P<b>{WORD})\b", re.I)

# Relation-word variants allow short phrases rather than one-token A/B. These are
# easier surfaces, useful mainly as saturation controls.
SHORT_PHRASE = rf"(?:{WORD})(?:\s+(?:of|the|a|an|and|for|to|in|on|with|{WORD})){{0,5}}"
COMPARATIVE_REL_RE = re.compile(
    rf"\b(?P<a>{SHORT_PHRASE})\s+(?P<verb>is|are|was|were|becomes?|became|seems?|appears?)\s+(?P<rel>more|less)\s+(?P<prop>[A-Za-z][A-Za-z'\-]*(?:\s+(?!than\b)[A-Za-z][A-Za-z'\-]*){{0,5}})\s+than\s+(?P<b>{SHORT_PHRASE})\b",
    re.I,
)
SPATIAL_REL_RE = re.compile(
    rf"\b(?P<a>{SHORT_PHRASE})\s+(?P<verb>is|are|was|were|lies|lie|sits|sit|stands|stand|stood|stays|stay|located)\s+(?P<rel>above|below)\s+(?P<b>{SHORT_PHRASE})\b",
    re.I,
)
LATERAL_REL_RE = re.compile(
    rf"\b(?P<a>{SHORT_PHRASE})\s+(?P<verb>is|are|was|were|lies|lie|sits|sit|stands|stand|located)\s+(?P<rel>left|right)\s+of\s+(?P<b>{SHORT_PHRASE})\b",
    re.I,
)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def clean_word(w: str) -> str | None:
    s = w.lower().strip("-'’‘`\".,;:!?()[]{}")
    if len(s) < 4 or len(s) > 16:
        return None
    if s in STOP or s in PRONOUNS:
        return None
    if not re.fullmatch(r"[a-z][a-z'\-]{3,15}", s):
        return None
    return s


def clean_phrase(p: str) -> str | None:
    p = re.sub(r"\s+", " ", p.strip(" \t\n\r.,;:!?()[]{}\"'“”‘’"))
    words = [w.strip("-'’‘`\".,;:!?()[]{}") for w in p.split()]
    words = [w for w in words if w]
    while words and words[0].lower() in {"and", "or", "but", "because", "while", "when", "where", "that", "which", "who"}:
        words.pop(0)
    while words and words[-1].lower() in {"and", "or", "but", "of", "to", "for", "with", "in", "on", "the", "a", "an"}:
        words.pop()
    if not (1 <= len(words) <= 7):
        return None
    joined = " ".join(words)
    if len(joined) < 4 or len(joined) > 70:
        return None
    lows = [w.lower() for w in words]
    if all(w in STOP or w in PRONOUNS for w in lows):
        return None
    if any(w in {"www", "xxx"} for w in lows):
        return None
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9'\-]*(?:\s+[A-Za-z0-9][A-Za-z0-9'\-]*){0,6}", joined):
        return None
    return joined


def one_token_id(tokenizer, word: str) -> int | None:
    # We score lower-case candidates because legal corpora/eval are largely cased
    # but relation/entity common nouns should have stable lower-case ids. Reject if
    # tokenization is not exactly one non-special id.
    enc = tokenizer(word.lower(), add_special_tokens=False)
    ids = enc.get("input_ids") or []
    if len(ids) != 1:
        return None
    tid = int(ids[0])
    if tid in set(tokenizer.all_special_ids):
        return None
    return tid


def sent_split(text: str) -> list[str]:
    parts = []
    for s in SENT_SPLIT_RE.split(text):
        s = re.sub(r"\s+", " ", s).strip()
        if 35 <= len(s) <= 260:
            parts.append(s)
    return parts


def add_entity(rows: list[dict[str, Any]], *, line_no: int, example_id: Any, source: str, sent: str, family: str, a: str, b: str, verb: str, relword: str, prop: str = "") -> None:
    aw = clean_word(a); bw = clean_word(b)
    if aw is None or bw is None or aw == bw:
        return
    r = relword.lower()
    inv = INVERSE.get(r)
    if inv is None:
        return
    prop = re.sub(r"\s+", " ", prop.strip())
    verb_l = verb.lower()
    if family.startswith("comp"):
        if not prop or "than" in prop.lower() or len(prop.split()) > 6:
            return
        ctx1 = f"{tokenizer_mask_placeholder()} {verb_l} {r} {prop} than {bw}."
        ctx2 = f"{tokenizer_mask_placeholder()} {verb_l} {inv} {prop} than {aw}."
    elif family.startswith("spatial"):
        ctx1 = f"{tokenizer_mask_placeholder()} {verb_l} {r} {bw}."
        ctx2 = f"{tokenizer_mask_placeholder()} {verb_l} {inv} {aw}."
    elif family.startswith("lateral"):
        ctx1 = f"{tokenizer_mask_placeholder()} {verb_l} {r} of {bw}."
        ctx2 = f"{tokenizer_mask_placeholder()} {verb_l} {inv} of {aw}."
    elif family.startswith("temporal"):
        ctx1 = f"{tokenizer_mask_placeholder()} happened {r} {bw}."
        ctx2 = f"{tokenizer_mask_placeholder()} happened {inv} {aw}."
    else:
        return
    rows.append({
        "object": "entity_exchange",
        "family": family,
        "line_no": line_no,
        "example_id": example_id,
        "source": source,
        "a_word": aw,
        "b_word": bw,
        "rel": r,
        "inv_rel": inv,
        "property": prop,
        "context1": ctx1,
        "expect1": aw,
        "alt1": bw,
        "context2": ctx2,
        "expect2": bw,
        "alt2": aw,
        "sentence": sent,
    })


def tokenizer_mask_placeholder() -> str:
    # Replaced with actual tokenizer mask after tokenizer load; `[MASK]` is used
    # by the BabyLM DeBERTa tokenizers and keeps rows readable before scoring.
    return "[MASK]"


def add_relation(rows: list[dict[str, Any]], *, line_no: int, example_id: Any, source: str, sent: str, family: str, a: str, b: str, verb: str, relword: str, prop: str = "") -> None:
    aa = clean_phrase(a); bb = clean_phrase(b)
    if aa is None or bb is None or aa.lower() == bb.lower():
        return
    r = relword.lower(); inv = INVERSE.get(r)
    if inv is None:
        return
    verb_l = verb.lower()
    prop = re.sub(r"\s+", " ", prop.strip())
    if family.startswith("comp"):
        if not prop or len(prop.split()) > 6:
            return
        ctx1 = f"{aa} {verb_l} [MASK] {prop} than {bb}."
        ctx2 = f"{bb} {verb_l} [MASK] {prop} than {aa}."
    elif family.startswith("spatial"):
        ctx1 = f"{aa} {verb_l} [MASK] {bb}."
        ctx2 = f"{bb} {verb_l} [MASK] {aa}."
    elif family.startswith("lateral"):
        ctx1 = f"{aa} {verb_l} [MASK] of {bb}."
        ctx2 = f"{bb} {verb_l} [MASK] of {aa}."
    else:
        return
    rows.append({
        "object": "relation_word",
        "family": family,
        "line_no": line_no,
        "example_id": example_id,
        "source": source,
        "a_phrase": aa,
        "b_phrase": bb,
        "rel": r,
        "inv_rel": inv,
        "property": prop,
        "context1": ctx1,
        "expect1": r,
        "alt1": inv,
        "context2": ctx2,
        "expect2": inv,
        "alt2": r,
        "sentence": sent,
    })


def mine_rows(max_rows: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    ent: list[dict[str, Any]] = []
    relrows: list[dict[str, Any]] = []
    line_hits = 0
    source_counts = Counter()
    with CORPUS.open("r", encoding="utf-8") as f:
        for z, line in enumerate(f, start=1):
            if max_rows and z > max_rows:
                break
            rec = json.loads(line)
            text = rec["text"]
            source = rec.get("source", "")
            any_hit = False
            for sent in sent_split(text):
                for m in COMPARATIVE_RE.finditer(sent):
                    add_entity(ent, line_no=z, example_id=rec.get("example_id"), source=source, sent=sent, family="comparative_entity", a=m.group("a"), b=m.group("b"), verb=m.group("verb"), relword=m.group("rel"), prop=m.group("prop"))
                    any_hit = True
                for m in SPATIAL_ENTITY_RE.finditer(sent):
                    add_entity(ent, line_no=z, example_id=rec.get("example_id"), source=source, sent=sent, family="spatial_vertical_entity", a=m.group("a"), b=m.group("b"), verb=m.group("verb"), relword=m.group("rel"))
                    any_hit = True
                for m in LATERAL_ENTITY_RE.finditer(sent):
                    add_entity(ent, line_no=z, example_id=rec.get("example_id"), source=source, sent=sent, family="spatial_lateral_entity", a=m.group("a"), b=m.group("b"), verb=m.group("verb"), relword=m.group("rel"))
                    any_hit = True
                # Temporal family is collected but treated separately/noisy.
                for m in TEMPORAL_ENTITY_RE.finditer(sent):
                    add_entity(ent, line_no=z, example_id=rec.get("example_id"), source=source, sent=sent, family="temporal_entity_heldout", a=m.group("a"), b=m.group("b"), verb="", relword=m.group("rel"))
                    any_hit = True
                for m in COMPARATIVE_REL_RE.finditer(sent):
                    add_relation(relrows, line_no=z, example_id=rec.get("example_id"), source=source, sent=sent, family="comparative_relation", a=m.group("a"), b=m.group("b"), verb=m.group("verb"), relword=m.group("rel"), prop=m.group("prop"))
                    any_hit = True
                for m in SPATIAL_REL_RE.finditer(sent):
                    add_relation(relrows, line_no=z, example_id=rec.get("example_id"), source=source, sent=sent, family="spatial_vertical_relation", a=m.group("a"), b=m.group("b"), verb=m.group("verb"), relword=m.group("rel"))
                    any_hit = True
                for m in LATERAL_REL_RE.finditer(sent):
                    add_relation(relrows, line_no=z, example_id=rec.get("example_id"), source=source, sent=sent, family="spatial_lateral_relation", a=m.group("a"), b=m.group("b"), verb=m.group("verb"), relword=m.group("rel"))
                    any_hit = True
            if any_hit:
                line_hits += 1
                source_counts[source] += 1
    return ent, relrows, {"line_hits": line_hits, "source_counts": dict(source_counts)}


def stratified_sample(rows: list[dict[str, Any]], max_per_family: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    fams: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        fams[r["family"]].append(r)
    sampled: list[dict[str, Any]] = []
    for fam, rs in sorted(fams.items()):
        rng.shuffle(rs)
        sampled.extend(rs[:max_per_family])
    rng.shuffle(sampled)
    for i, r in enumerate(sampled):
        r["split"] = "heldout" if i % 5 == 0 else "development"
    return sampled


def tokenize_and_filter(rows: list[dict[str, Any]], tokenizer) -> tuple[list[dict[str, Any]], dict[str, int]]:
    counts = Counter()
    out = []
    mask_token = tokenizer.mask_token or "[MASK]"
    for r in rows:
        r = dict(r)
        # Substitute actual mask token in case a future tokenizer differs.
        r["context1"] = r["context1"].replace("[MASK]", mask_token)
        r["context2"] = r["context2"].replace("[MASK]", mask_token)
        ids = {}
        ok = True
        for key in ["expect1", "alt1", "expect2", "alt2"]:
            tid = one_token_id(tokenizer, str(r[key]))
            if tid is None:
                counts[f"reject_multitoken_{key}"] += 1
                ok = False
                break
            ids[key + "_id"] = tid
        if not ok:
            continue
        enc1 = tokenizer(r["context1"], add_special_tokens=True, truncation=True, max_length=96)
        enc2 = tokenizer(r["context2"], add_special_tokens=True, truncation=True, max_length=96)
        mask_id = tokenizer.mask_token_id
        if mask_id is None or enc1["input_ids"].count(mask_id) != 1 or enc2["input_ids"].count(mask_id) != 1:
            counts["reject_bad_mask"] += 1
            continue
        r.update(ids)
        out.append(r)
    return out, dict(counts)


def batch_score(model, tokenizer, rows: list[dict[str, Any]], device: torch.device, batch_size: int) -> list[dict[str, float]]:
    model.eval()
    mask_id = tokenizer.mask_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    scored: list[dict[str, float]] = []
    contexts = []
    meta = []
    for ri, r in enumerate(rows):
        contexts.append(r["context1"]); meta.append((ri, 1, int(r["expect1_id"]), int(r["alt1_id"])))
        contexts.append(r["context2"]); meta.append((ri, 2, int(r["expect2_id"]), int(r["alt2_id"])))
    temp = [None] * len(contexts)
    with torch.no_grad():
        for off in range(0, len(contexts), batch_size):
            batch_ctx = contexts[off:off+batch_size]
            enc = tokenizer(batch_ctx, padding=True, truncation=True, max_length=96, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            attn = enc.get("attention_mask")
            if attn is not None:
                attn = attn.to(device)
            out = model(input_ids=input_ids, attention_mask=attn)
            logits = out.logits
            for j in range(input_ids.shape[0]):
                mask_pos = (input_ids[j] == mask_id).nonzero(as_tuple=False).view(-1)
                if len(mask_pos) != 1:
                    temp[off+j] = {"bad_mask": 1.0}
                    continue
                _, side, exp_id, alt_id = meta[off+j]
                lv = logits[j, int(mask_pos.item())]
                exp_logit = float(lv[exp_id].item())
                alt_logit = float(lv[alt_id].item())
                temp[off+j] = {
                    f"margin{side}": exp_logit - alt_logit,
                    f"expect_logit{side}": exp_logit,
                    f"alt_logit{side}": alt_logit,
                    f"expect_rank{side}": rank_of(lv, exp_id),
                    f"alt_rank{side}": rank_of(lv, alt_id),
                }
    for i in range(len(rows)):
        a = temp[2*i] or {}; b = temp[2*i+1] or {}
        rec = {}
        rec.update(a); rec.update(b)
        rec["M_pair"] = float(rec.get("margin1", 0.0) + rec.get("margin2", 0.0))
        rec["both_correct"] = bool(rec.get("margin1", -1e9) > 0 and rec.get("margin2", -1e9) > 0)
        rec["either_wrong"] = not rec["both_correct"]
        scored.append(rec)
    return scored


def rank_of(logits: torch.Tensor, token_id: int) -> int:
    val = logits[token_id]
    return int((logits > val).sum().item()) + 1


def summarize(vals: list[float]) -> dict[str, Any]:
    if not vals:
        return {"n": 0}
    xs = sorted(float(v) for v in vals)
    def q(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        k = int(round(p * (len(xs)-1)))
        return xs[max(0, min(len(xs)-1, k))]
    return {"n": len(xs), "mean": statistics.fmean(xs), "median": statistics.median(xs), "p10": q(0.10), "p90": q(0.90), "min": xs[0], "max": xs[-1]}


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    # Preserve any keys introduced later.
    seen = set(keys)
    for r in rows:
        for k in r:
            if k not in seen:
                keys.append(k); seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v) for k, v in r.items()})


def summarize_rows(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not rows:
        return {"n": 0}
    for group_name, filt in [("all", lambda r: True), ("development", lambda r: r.get("split") == "development"), ("heldout", lambda r: r.get("split") == "heldout")]:
        gr = [r for r in rows if filt(r)]
        if not gr:
            out[group_name] = {"n": 0}
            continue
        out[group_name] = {
            "n": len(gr),
            "both_correct_frac": sum(bool(r[f"{prefix}_both_correct"]) for r in gr) / len(gr),
            "either_wrong": sum(bool(r[f"{prefix}_either_wrong"]) for r in gr),
            "M_pair": summarize([float(r[f"{prefix}_M_pair"]) for r in gr]),
            "margin1": summarize([float(r[f"{prefix}_margin1"]) for r in gr]),
            "margin2": summarize([float(r[f"{prefix}_margin2"]) for r in gr]),
        }
    fams = {}
    for fam in sorted(set(r["family"] for r in rows)):
        fr = [r for r in rows if r["family"] == fam]
        fams[fam] = {
            "n": len(fr),
            "both_correct_frac": sum(bool(r[f"{prefix}_both_correct"]) for r in fr) / len(fr),
            "M_pair": summarize([float(r[f"{prefix}_M_pair"]) for r in fr]),
        }
    out["by_family"] = fams
    return out


def line_quantiles(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    xs = sorted(int(r["line_no"]) for r in rows)
    def q(p: float) -> int:
        return xs[int(round(p*(len(xs)-1)))]
    return {"n": len(xs), "min": xs[0], "p10": q(0.1), "median": q(0.5), "p90": q(0.9), "max": xs[-1]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--max-rows", type=int, default=64740)
    ap.add_argument("--max-per-family", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"event": "load_tokenizer", "model": rel(SCALE80), "utc": now()}), flush=True)
    tok = AutoTokenizer.from_pretrained(SCALE80, trust_remote_code=True)
    print(json.dumps({"event": "mine_start", "corpus": rel(CORPUS), "max_rows": args.max_rows, "utc": now()}), flush=True)
    ent_all, rel_all, mine_meta = mine_rows(args.max_rows)
    print(json.dumps({"event": "mine_done", "entity_raw": len(ent_all), "relation_raw": len(rel_all), **mine_meta, "utc": now()}), flush=True)
    ent_sample = stratified_sample(ent_all, args.max_per_family, seed=15801)
    rel_sample = stratified_sample(rel_all, args.max_per_family, seed=15802)
    ent, ent_reject = tokenize_and_filter(ent_sample, tok)
    relrows, rel_reject = tokenize_and_filter(rel_sample, tok)
    rows = ent + relrows
    print(json.dumps({"event": "sample_filter_done", "entity": len(ent), "relation": len(relrows), "entity_reject": ent_reject, "relation_reject": rel_reject, "utc": now()}), flush=True)
    if not rows:
        raise RuntimeError("no rows after filtering")
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(json.dumps({"event": "load_models", "device": str(device), "utc": now()}), flush=True)
    scale = AutoModelForMaskedLM.from_pretrained(SCALE80, trust_remote_code=True).to(device).eval()
    base = AutoModelForMaskedLM.from_pretrained(BASE80, trust_remote_code=True).to(device).eval()
    print(json.dumps({"event": "score_scale", "n": len(rows), "utc": now()}), flush=True)
    s_scores = batch_score(scale, tok, rows, device, args.batch_size)
    print(json.dumps({"event": "score_base", "n": len(rows), "utc": now()}), flush=True)
    b_scores = batch_score(base, tok, rows, device, args.batch_size)
    for r, ss, bs in zip(rows, s_scores, b_scores):
        for k, v in ss.items():
            r["scale_" + k] = v
        for k, v in bs.items():
            r["base_" + k] = v
        r["scale_minus_base_M_pair"] = float(r["scale_M_pair"] - r["base_M_pair"])
        r["scale_minus_base_both"] = int(bool(r["scale_both_correct"])) - int(bool(r["base_both_correct"]))
        r["scale_specific_failure"] = bool((not r["scale_both_correct"]) and r["base_both_correct"])
        r["base_specific_failure"] = bool(r["scale_both_correct"] and (not r["base_both_correct"]))
    row_path = OUT / "paired_natural_context_rows.csv"
    hard_path = OUT / "paired_natural_context_hard_examples.csv"
    write_csv(row_path, rows)
    hard = [r for r in rows if r["object"] == "entity_exchange" and (r["scale_specific_failure"] or r["scale_M_pair"] < 1.0 or r["scale_minus_base_M_pair"] < -1.0)]
    hard = sorted(hard, key=lambda r: (not r["scale_specific_failure"], r["scale_minus_base_M_pair"], r["scale_M_pair"]))[:120]
    write_csv(hard_path, hard)
    obj_summary = {}
    for obj in sorted(set(r["object"] for r in rows)):
        rr = [r for r in rows if r["object"] == obj]
        obj_summary[obj] = {
            "n": len(rr),
            "line_distribution": line_quantiles(rr),
            "source_counts": dict(Counter(r["source"] for r in rr)),
            "scale": summarize_rows(rr, "scale"),
            "base": summarize_rows(rr, "base"),
            "scale_minus_base_M_pair": summarize([float(r["scale_minus_base_M_pair"]) for r in rr]),
            "scale_specific_failures": sum(bool(r["scale_specific_failure"]) for r in rr),
            "base_specific_failures": sum(bool(r["base_specific_failure"]) for r in rr),
        }
    summary = {
        "status": "PAIRED_NATURAL_CONTEXT_PILOT_DONE",
        "created_utc": now(),
        "description": "Full-corpus sampled fixed-alternative paired natural/causal-transform readout; no local-logit alternative selection, no official eval rows, no training.",
        "corpus": rel(CORPUS),
        "scale_model": rel(SCALE80),
        "base_model": rel(BASE80),
        "params": vars(args),
        "mining": {"raw_entity": len(ent_all), "raw_relation": len(rel_all), "sampled_entity": len(ent_sample), "sampled_relation": len(rel_sample), "filtered_entity": len(ent), "filtered_relation": len(relrows), "entity_reject": ent_reject, "relation_reject": rel_reject, **mine_meta},
        "objects": obj_summary,
        "rows_csv": rel(row_path),
        "hard_examples_csv": rel(hard_path),
        "interpretation_guide": "entity_exchange is the intended hard object; relation_word is a relation-token saturation control. A useful training source would need entity_exchange to be precise, unsaturated, sampled across corpus, and to show model differences not reducible to generic relation-word reconstruction.",
    }
    summary_path = OUT / "paired_natural_context_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join([
        "# research paired natural context / causal-transform pilot",
        "",
        f"Summary JSON: `{rel(summary_path)}`",
        f"Rows CSV: `{rel(row_path)}`",
        f"Hard examples CSV: `{rel(hard_path)}`",
        "",
        "This pilot was built because the research cue-filtered pool used scale1.75 local logits and many nearby cues did not govern the masked word. Here alternatives are fixed semantic inverses or fixed A/B entities from paired causal transformations, and the miner scans the whole legal 10M corpus before stratified sampling.",
        "",
        "## Object summary",
        json.dumps(obj_summary, indent=2, ensure_ascii=False),
        "",
        "## Immediate interpretation",
        summary["interpretation_guide"],
        ""
    ]), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "summary": rel(summary_path), "rows": len(rows), "entity_rows": len(ent), "relation_rows": len(relrows), "hard_examples": len(hard), "note": rel(NOTE)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
