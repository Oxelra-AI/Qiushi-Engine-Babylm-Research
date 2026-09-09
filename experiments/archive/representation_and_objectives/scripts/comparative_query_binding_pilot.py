#!/usr/bin/env python3
"""research: well-formed comparative query-binding pilot.

This readout repairs the malformed research entity_exchange object. It uses natural
comparative sentences from the legal 10M corpus but keeps both alternatives
visible in the same sentence and masks an answer slot whose category is licensed:

    <natural sentence>. In this comparison, the more P option is [MASK].
    <natural sentence>. In this comparison, the less P option is [MASK].

The two fixed one-token alternatives A/B are mined from the sentence "A is more P
than B". They are not selected from model logits. The query relation flips while
the evidence sentence stays fixed, so unigram frequency is largely cancelled by
M_pair = margin(query_rel) + margin(query_inverse). This is an evaluation-
independent readout only; it is not a training source.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import json
import math
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
OUT = ROOT / "experiments/archive/representation_and_objectives/data/comparative_query_binding_pilot"
NOTE = ROOT / "research/notes/representation_and_objectives/comparative_query_binding_pilot.md"
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
SCALAR_RELS = "more|less|higher|lower|larger|smaller|bigger|shorter|longer|older|younger|heavier|lighter|hotter|colder|warmer|cooler|better|worse|faster|slower|stronger|weaker"
INV = {
    "more": "less", "less": "more",
    "higher": "lower", "lower": "higher",
    "larger": "smaller", "smaller": "larger", "bigger": "smaller",
    "shorter": "longer", "longer": "shorter",
    "older": "younger", "younger": "older",
    "heavier": "lighter", "lighter": "heavier",
    "hotter": "colder", "colder": "hotter", "warmer": "cooler", "cooler": "warmer",
    "better": "worse", "worse": "better", "faster": "slower", "slower": "faster", "stronger": "weaker", "weaker": "stronger",
}
COMP_RE = re.compile(
    rf"\b(?P<a>{WORD})\s+(?P<verb>is|are|was|were|becomes?|became|seems?|appears?)\s+(?P<rel>{SCALAR_RELS})\s+(?P<prop>[A-Za-z][A-Za-z'\-]*(?:\s+(?!than\b)[A-Za-z][A-Za-z'\-]*){{0,4}})\s+than\s+(?P<b>{WORD})\b",
    re.I,
)
STOP = set("""
a an the and or but if then than as at by for from in into is it its of on onto over under to with without within across after before during about above below between through this that these those there here when where while who whom whose which what why how be been being are was were will would should could can may might must do does did doing done have has had having you your we our they their he his she her i my me mine us them him himself herself itself themselves not no nor so such very more most less least other another same just only own out up down off back much many any each every some all both either neither because although though since until like also well just actually really probably maybe there here now then yes yeah okay ok uh um www xxx gonna wanna gotta thought initially ever never once theirs ours yours mine someone anything everything nothing nobody anybody everybody however therefore whereas percent number numbers amount amounts kind kinds type types way ways thing things things
""".split())
BAD = set("thought initially ever never once theirs yours ours mine itself himself herself themselves someone somebody anyone anybody everyone everybody something anything everything nothing none same different other others rather former latter current earlier later first last next previous soon immediately already".split())


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
    if s in STOP or s in BAD:
        return None
    if s.endswith("ly"):
        return None
    if not re.fullmatch(r"[a-z][a-z'\-]{3,15}", s):
        return None
    return s

def clean_prop(p: str) -> str | None:
    p = re.sub(r"\s+", " ", p.lower().strip(" -,'\".;:!?()[]{}"))
    if not p or len(p) > 60:
        return None
    toks = p.split()
    if not (1 <= len(toks) <= 5):
        return None
    if toks[0] in {"and", "or", "but", "than"}:
        return None
    if any(t in {"than"} for t in toks):
        return None
    if all(t in STOP for t in toks):
        return None
    return p

def one_token_id(tok, word: str) -> int | None:
    ids = tok(word.lower(), add_special_tokens=False).get("input_ids") or []
    if len(ids) != 1:
        return None
    tid = int(ids[0])
    if tid in set(tok.all_special_ids):
        return None
    return tid

def sent_split(text: str) -> list[str]:
    out = []
    for s in SENT_SPLIT_RE.split(text):
        s = re.sub(r"\s+", " ", s).strip()
        if 30 <= len(s) <= 220:
            out.append(s)
    return out

def word_counts(max_rows: int = 0) -> Counter[str]:
    c = Counter()
    with CORPUS.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if max_rows and i > max_rows:
                break
            text = json.loads(line)["text"].lower()
            for m in WORD_RE.finditer(text):
                w = clean_word(m.group(0))
                if w:
                    c[w] += 1
    return c

def mine(tok, counts: Counter[str], max_rows: int, max_raw_per_family: int, freq_ratio: float, max_sentence_chars: int) -> list[dict[str, Any]]:
    raw = []
    with CORPUS.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if max_rows and line_no > max_rows:
                break
            rec = json.loads(line)
            for sent in sent_split(rec["text"]):
                if len(sent) > max_sentence_chars:
                    continue
                for m in COMP_RE.finditer(sent):
                    a = clean_word(m.group("a")); b = clean_word(m.group("b")); prop = clean_prop(m.group("prop")); r = m.group("rel").lower()
                    inv = INV.get(r)
                    if not a or not b or not prop or not inv or a == b:
                        continue
                    fa, fb = counts[a], counts[b]
                    if fa < 10 or fb < 10:
                        continue
                    if max(fa, fb) / max(1, min(fa, fb)) > freq_ratio:
                        continue
                    aid = one_token_id(tok, a); bid = one_token_id(tok, b)
                    if aid is None or bid is None or aid == bid:
                        continue
                    raw.append({
                        "line_no": line_no,
                        "example_id": rec.get("example_id"),
                        "source": rec.get("source"),
                        "sentence": sent,
                        "a_word": a,
                        "b_word": b,
                        "a_freq": fa,
                        "b_freq": fb,
                        "rel": r,
                        "inv_rel": inv,
                        "property": prop,
                        "a_id": aid,
                        "b_id": bid,
                    })
    # Stratify by relation and source over full corpus after scanning.
    rng = random.Random(15831)
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in raw:
        buckets[(r["rel"], str(r["source"]))].append(r)
    sampled = []
    per_bucket = max(2, max_raw_per_family // max(1, len(buckets)))
    for _, rs in sorted(buckets.items()):
        rng.shuffle(rs)
        sampled.extend(rs[:per_bucket])
    if len(sampled) < max_raw_per_family:
        remaining = [r for r in raw if r not in sampled]
        rng.shuffle(remaining)
        sampled.extend(remaining[:max_raw_per_family-len(sampled)])
    rng.shuffle(sampled)
    sampled = sampled[:max_raw_per_family]
    for i, r in enumerate(sampled):
        mask = tok.mask_token or "[MASK]"
        # Both alternatives and the natural sentence are visible. Only the query adjective/relation changes.
        pre = f"{r['sentence']} In this comparison, the {r['rel']} {r['property']} option is {mask}."
        post = f"{r['sentence']} In this comparison, the {r['inv_rel']} {r['property']} option is {mask}."
        # Reject contexts that tokenize without exactly one mask.
        e1 = tok(pre, add_special_tokens=True, truncation=True, max_length=128)
        e2 = tok(post, add_special_tokens=True, truncation=True, max_length=128)
        if e1["input_ids"].count(tok.mask_token_id) != 1 or e2["input_ids"].count(tok.mask_token_id) != 1:
            continue
        r["context1"] = pre
        r["expect1"] = r["a_word"]; r["alt1"] = r["b_word"]; r["expect1_id"] = r["a_id"]; r["alt1_id"] = r["b_id"]
        r["context2"] = post
        r["expect2"] = r["b_word"]; r["alt2"] = r["a_word"]; r["expect2_id"] = r["b_id"]; r["alt2_id"] = r["a_id"]
        r["split"] = "heldout" if i % 5 == 0 else "development"
    return sampled

def rank_of(logits: torch.Tensor, tid: int) -> int:
    return int((logits > logits[tid]).sum().item()) + 1

def score_model(model, tok, rows: list[dict[str, Any]], device: torch.device, batch_size: int) -> list[dict[str, Any]]:
    contexts = []
    meta = []
    for i, r in enumerate(rows):
        contexts.append(r["context1"]); meta.append((i, 1, int(r["expect1_id"]), int(r["alt1_id"])))
        contexts.append(r["context2"]); meta.append((i, 2, int(r["expect2_id"]), int(r["alt2_id"])))
    temp = [None] * len(contexts)
    mask_id = tok.mask_token_id
    with torch.no_grad():
        for off in range(0, len(contexts), batch_size):
            enc = tok(contexts[off:off+batch_size], padding=True, truncation=True, max_length=128, return_tensors="pt")
            ids = enc["input_ids"].to(device)
            attn = enc.get("attention_mask")
            if attn is not None:
                attn = attn.to(device)
            logits = model(input_ids=ids, attention_mask=attn).logits
            for j in range(ids.shape[0]):
                pos = (ids[j] == mask_id).nonzero(as_tuple=False).view(-1)
                if len(pos) != 1:
                    temp[off+j] = {"bad_mask": True}; continue
                _, side, eid, aid = meta[off+j]
                lv = logits[j, int(pos.item())]
                temp[off+j] = {
                    f"margin{side}": float(lv[eid].item() - lv[aid].item()),
                    f"expect_rank{side}": rank_of(lv, eid),
                    f"alt_rank{side}": rank_of(lv, aid),
                    f"expect_logit{side}": float(lv[eid].item()),
                    f"alt_logit{side}": float(lv[aid].item()),
                }
    out = []
    for i in range(len(rows)):
        d = {}
        d.update(temp[2*i] or {}); d.update(temp[2*i+1] or {})
        d["M_pair"] = float(d.get("margin1", 0.0) + d.get("margin2", 0.0))
        d["both_correct"] = bool(d.get("margin1", -1e9) > 0 and d.get("margin2", -1e9) > 0)
        d["either_wrong"] = not d["both_correct"]
        out.append(d)
    return out

def qstats(vals):
    xs = sorted(float(v) for v in vals if math.isfinite(float(v)))
    if not xs:
        return {"n": 0}
    def q(p):
        if len(xs) == 1: return xs[0]
        return xs[int(round(p*(len(xs)-1)))]
    return {"n": len(xs), "mean": statistics.fmean(xs), "median": statistics.median(xs), "p10": q(0.1), "p90": q(0.9), "min": xs[0], "max": xs[-1]}

def summarize(rows, prefix):
    out = {}
    for name, filt in [("all", lambda r: True), ("development", lambda r: r.get("split")=="development"), ("heldout", lambda r: r.get("split")=="heldout")]:
        rr = [r for r in rows if filt(r)]
        out[name] = {"n": len(rr), "both_correct_frac": sum(bool(r[f"{prefix}_both_correct"]) for r in rr)/len(rr) if rr else None, "M_pair": qstats([r[f"{prefix}_M_pair"] for r in rr]) if rr else {"n":0}, "margin1": qstats([r[f"{prefix}_margin1"] for r in rr]) if rr else {"n":0}, "margin2": qstats([r[f"{prefix}_margin2"] for r in rr]) if rr else {"n":0}}
    rels = {}
    for relname in sorted(set(r["rel"] for r in rows)):
        rr = [r for r in rows if r["rel"] == relname]
        rels[relname] = {"n": len(rr), "both_correct_frac": sum(bool(r[f"{prefix}_both_correct"]) for r in rr)/len(rr), "M_pair": qstats([r[f"{prefix}_M_pair"] for r in rr])}
    out["by_rel"] = rels
    return out

def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8"); return
    keys = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                keys.append(k); seen.add(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--max-rows", type=int, default=64740)
    ap.add_argument("--max-samples", type=int, default=240)
    ap.add_argument("--freq-ratio", type=float, default=20.0)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--max-sentence-chars", type=int, default=180)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"event":"load_tokenizer", "utc":now()}), flush=True)
    tok = AutoTokenizer.from_pretrained(SCALE80, trust_remote_code=True)
    print(json.dumps({"event":"count_start", "utc":now()}), flush=True)
    counts = word_counts(args.max_rows)
    print(json.dumps({"event":"count_done", "types":len(counts), "utc":now()}), flush=True)
    print(json.dumps({"event":"mine_start", "utc":now()}), flush=True)
    rows = mine(tok, counts, args.max_rows, args.max_samples, args.freq_ratio, args.max_sentence_chars)
    print(json.dumps({"event":"mine_done", "rows":len(rows), "utc":now()}), flush=True)
    if not rows:
        raise RuntimeError("no rows")
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    scale = AutoModelForMaskedLM.from_pretrained(SCALE80, trust_remote_code=True).to(device).eval()
    base = AutoModelForMaskedLM.from_pretrained(BASE80, trust_remote_code=True).to(device).eval()
    print(json.dumps({"event":"score_scale", "n":len(rows), "device":str(device), "utc":now()}), flush=True)
    ss = score_model(scale, tok, rows, device, args.batch_size)
    print(json.dumps({"event":"score_base", "n":len(rows), "device":str(device), "utc":now()}), flush=True)
    bs = score_model(base, tok, rows, device, args.batch_size)
    for r, s, b in zip(rows, ss, bs):
        for k, v in s.items(): r["scale_"+k] = v
        for k, v in b.items(): r["base_"+k] = v
        r["scale_minus_base_M_pair"] = float(r["scale_M_pair"] - r["base_M_pair"])
        r["scale_specific_failure"] = bool((not r["scale_both_correct"]) and r["base_both_correct"])
        r["base_specific_failure"] = bool(r["scale_both_correct"] and (not r["base_both_correct"]))
    row_path = OUT / "comparative_query_binding_rows.csv"
    hard_path = OUT / "comparative_query_binding_hard_examples.csv"
    write_csv(row_path, rows)
    hard = [r for r in rows if r["scale_specific_failure"] or r["scale_M_pair"] < 1.0 or r["scale_minus_base_M_pair"] < -1.0]
    hard = sorted(hard, key=lambda r:(not r["scale_specific_failure"], r["scale_minus_base_M_pair"], r["scale_M_pair"]))[:120]
    write_csv(hard_path, hard)
    summary = {
        "status":"COMPARATIVE_QUERY_BINDING_PILOT_DONE",
        "created_utc": now(),
        "description":"Well-formed natural comparative query-binding readout with both alternatives visible; fixed A/B alternatives from full-corpus natural comparatives; no local-logit mining and no training.",
        "corpus": rel(CORPUS), "scale_model": rel(SCALE80), "base_model": rel(BASE80), "params": vars(args),
        "rows": len(rows),
        "source_counts": dict(Counter(r["source"] for r in rows)),
        "line_distribution": qstats([r["line_no"] for r in rows]),
        "scale": summarize(rows, "scale"),
        "base": summarize(rows, "base"),
        "scale_minus_base_M_pair": qstats([r["scale_minus_base_M_pair"] for r in rows]),
        "scale_specific_failures": sum(r["scale_specific_failure"] for r in rows),
        "base_specific_failures": sum(r["base_specific_failure"] for r in rows),
        "rows_csv": rel(row_path),
        "hard_examples_csv": rel(hard_path),
        "interpretation":"This readout is informative only if base both_correct is nonzero and there are corpus-distributed scale-specific failures with well-formed examples. It is a probe, not a training object by itself.",
    }
    sp = OUT / "comparative_query_binding_summary.json"
    sp.write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join([
        "# research comparative query-binding pilot",
        "",
        f"Summary JSON: `{rel(sp)}`",
        f"Rows CSV: `{rel(row_path)}`",
        f"Hard examples CSV: `{rel(hard_path)}`",
        "",
        "This pilot repairs the prior malformed entity-exchange frames by keeping the natural comparative sentence and both alternatives visible, then asking which alternative is the more/less (or inverse) property-bearing option.",
        "",
        "## Counts",
        json.dumps({"rows": summary["rows"], "scale_specific_failures": summary["scale_specific_failures"], "base_specific_failures": summary["base_specific_failures"], "source_counts": summary["source_counts"]}, indent=2),
        "",
        "## Scale",
        json.dumps(summary["scale"], indent=2),
        "",
        "## Base",
        json.dumps(summary["base"], indent=2),
        "",
        "## Scale minus base M_pair",
        json.dumps(summary["scale_minus_base_M_pair"], indent=2),
        "",
        summary["interpretation"],
        ""
    ]), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "summary": rel(sp), "rows": len(rows), "scale_specific_failures": summary["scale_specific_failures"], "base_specific_failures": summary["base_specific_failures"], "note": rel(NOTE)}, indent=2), flush=True)

if __name__ == "__main__":
    main()
