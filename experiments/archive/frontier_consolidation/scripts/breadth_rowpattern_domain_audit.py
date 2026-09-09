#!/usr/bin/env python3
"""research companion audit: rowwise WWM patterns and domain/quality of MAX breadth.

This complements `breadth_arm_token_wwm_junction_audit.py` by measuring
things that aggregate totals can hide:

* exact row-count sequence hashes for legal16k tokens, visible tokens, WWM groups;
* exact equality fractions of those row sequences;
* visible word-start flag pattern distance and WWM group-size distribution;
* common coarse text-domain and quality/flag distributions for compact rewrites
  versus selected breadth sentences.

CPU/file-only.  No training, GPU evaluation, SuperGLUE, AoA, upload, or
leaderboard action.
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
import re
import statistics
import time
from typing import Any, Iterable

from transformers import AutoTokenizer


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
TOKENIZER_DIR = WS / "data/compliant_tokenizer"
VIEW_POOL = WS / "data/dose_2p64x_rowholdout_pools/compact_view_dose2p64x_10M.jsonl"
REPEAT_POOL = WS / "data/dose_2p64x_rowholdout_pools/compact_repeat_dose2p64x_10M.jsonl"
BREADTH_POOL = WS / "data/dose_2p64x_breadth_rowholdout_pools/compact_breadth_dose2p64x_10M.jsonl"
PAIRS = WS / "data/dose_distribution_select/selected_matched_max_pairs.jsonl"
BREADTH_COMPANIONS = WS / "data/dose_2p64x_breadth_rowholdout_pools/compact_breadth_dose2p64x_selected_companion_sources.jsonl"
OUT_DIR = WS / "data/breadth_arm_geometry_audit"
SEQ_LENGTH = 256
CHANGED_ROWS = 7923
REWRITE_KEYS = ("rewrite_text", "view_text", "compact_text", "target_text", "rewrite")

DOMAIN_WORDS = {
    "quant_numeric": re.compile(r"\b\d|percent|million|billion|year|years|km|kg|\$|\d+", re.I),
    "science_technical": re.compile(r"science|energy|chemical|temperature|species|software|system|protein|medical|physics|planet|cell|data|algorithm|technology|engineer", re.I),
    "causal_relational": re.compile(r"because|therefore|caused|causes|effect|response|result|after|before|during|if|when|while|due to|led to|so that", re.I),
    "institutions_society": re.compile(r"government|university|foundation|company|school|court|law|policy|program|institute|society|department", re.I),
    "geography_places": re.compile(r"\b(city|country|river|island|mountain|state|county|region|village|town|lake)\b", re.I),
    "media_culture": re.compile(r"film|book|music|story|novel|art|game|television|song|culture|movie", re.I),
    "people_history": re.compile(r"history|historical|king|queen|war|century|president|ancient|born|died", re.I),
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def read_jsonl(path: pathlib.Path, limit: int | None = None) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                return
            if line.strip():
                yield json.loads(line)


def stats(xs: list[float]) -> dict[str, Any]:
    if not xs:
        return {"n": 0}
    s = sorted(xs)
    def q(p: float) -> float:
        if len(s) == 1:
            return float(s[0])
        return float(s[min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))] )
    return {"n": len(s), "mean": float(statistics.mean(s)), "sd": float(statistics.pstdev(s)) if len(s) > 1 else 0.0, "min": float(s[0]), "p05": q(0.05), "median": q(0.5), "p95": q(0.95), "max": float(s[-1]), "sum": float(sum(s))}


def sequence_sha(xs: list[int]) -> str:
    h = hashlib.sha256()
    for x in xs:
        h.update(int(x).to_bytes(8, "little", signed=True))
    return h.hexdigest()


def categorical_js(a: collections.Counter[str], b: collections.Counter[str]) -> float:
    na = sum(a.values()) or 1
    nb = sum(b.values()) or 1
    out = 0.0
    for k in set(a) | set(b):
        p = a.get(k, 0) / na
        q = b.get(k, 0) / nb
        m = 0.5 * (p + q)
        if p:
            out += 0.5 * p * math.log(p / m, 2)
        if q:
            out += 0.5 * q * math.log(q / m, 2)
    return float(out)


class Patterner:
    def __init__(self, tok, seq_len: int) -> None:
        self.tok = tok
        self.seq = seq_len
        self.special = set(tok.all_special_ids)
        self._ws: dict[int, bool] = {}

    def word_start(self, tid: int) -> bool:
        v = self._ws.get(tid)
        if v is None:
            s = self.tok.convert_ids_to_tokens(int(tid))
            v = bool(s is not None and (str(s).startswith("\u0120") or str(s).startswith("\u2581")))
            self._ws[tid] = v
        return v

    def row(self, text: str) -> dict[str, Any]:
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        vis = ids[: self.seq]
        starts: list[int] = []
        group_sizes: list[int] = []
        cur_size = 0
        gid = -1
        for i, tid in enumerate(vis):
            if tid in self.special:
                continue
            ws = bool(gid < 0 or self.word_start(int(tid)) or i == 0)
            starts.append(1 if ws else 0)
            if ws:
                if gid >= 0:
                    group_sizes.append(cur_size)
                gid += 1
                cur_size = 1
            else:
                cur_size += 1
        if gid >= 0:
            group_sizes.append(cur_size)
        return {"tokens_full": len(ids), "tokens_visible": len(vis), "wwm_groups": len(group_sizes), "word_start_pattern": starts, "group_sizes": group_sizes}


def pattern_distance(a: list[int], b: list[int]) -> float:
    n = max(len(a), len(b))
    if n == 0:
        return 0.0
    m = min(len(a), len(b))
    diff = sum(1 for i in range(m) if a[i] != b[i]) + abs(len(a) - len(b))
    return diff / n


def rewrite_text(pair: dict[str, Any]) -> str:
    for k in REWRITE_KEYS:
        v = pair.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def common_text_domains(text: str) -> list[str]:
    hits = [k for k, rgx in DOMAIN_WORDS.items() if rgx.search(text)]
    return hits or ["no_domain"]


def pair_domains(pair: dict[str, Any]) -> list[str]:
    hits = list(pair.get("domain_hits") or [])
    if not hits:
        return ["no_domain"]
    mapping = {"science_physical": "science_technical", "causal_relational": "causal_relational", "quant_numeric": "quant_numeric", "institutions_society": "institutions_society", "geography_places": "geography_places", "media_culture": "media_culture", "people_history": "people_history"}
    return [mapping.get(str(h), str(h)) for h in hits]


def add_multilabel(counter: collections.Counter[str], labels: list[str], weight: int = 1) -> None:
    for lab in labels:
        counter[str(lab)] += weight


def domain_quality() -> dict[str, Any]:
    rewrite_domain_meta: collections.Counter[str] = collections.Counter()
    rewrite_domain_text: collections.Counter[str] = collections.Counter()
    source_domain_text: collections.Counter[str] = collections.Counter()
    rewrite_word_domain_meta: collections.Counter[str] = collections.Counter()
    rewrite_flags: collections.Counter[str] = collections.Counter()
    source_risks: collections.Counter[str] = collections.Counter()
    rw_words = 0
    src_words = 0
    for p in read_jsonl(PAIRS):
        rw = rewrite_text(p)
        sw = str(p.get("source_text") or "")
        rww = len(rw.split())
        sww = len(sw.split())
        rw_words += rww; src_words += sww
        add_multilabel(rewrite_domain_meta, pair_domains(p))
        add_multilabel(rewrite_word_domain_meta, pair_domains(p), rww)
        add_multilabel(rewrite_domain_text, common_text_domains(rw))
        add_multilabel(source_domain_text, common_text_domains(sw))
        for f in list(p.get("soft_flags") or []) or ["no_soft_flags"]:
            rewrite_flags[str(f)] += 1
        for f in list(p.get("source_risks") or []) or ["no_source_risks"]:
            source_risks[str(f)] += 1

    breadth_domain_meta: collections.Counter[str] = collections.Counter()
    breadth_domain_text: collections.Counter[str] = collections.Counter()
    breadth_word_domain_meta: collections.Counter[str] = collections.Counter()
    breadth_flags: collections.Counter[str] = collections.Counter()
    bw_words = 0
    for b in read_jsonl(BREADTH_COMPANIONS):
        text = str(b.get("text") or "")
        w = len(text.split())
        bw_words += w
        dom = [str(b.get("primary_domain") or "no_domain")]
        dom = ["causal_relational" if x == "causal_temporal" else x for x in dom]
        add_multilabel(breadth_domain_meta, dom)
        add_multilabel(breadth_word_domain_meta, dom, w)
        add_multilabel(breadth_domain_text, common_text_domains(text))
        flags = list(b.get("source_row_quality_flags") or [])
        breadth_flags["flagged" if flags else "no_flags"] += 1
        for f in flags:
            breadth_flags[str(f)] += 1

    return {
        "compact_rewrite_words": rw_words,
        "max_source_words": src_words,
        "breadth_words": bw_words,
        "compact_rewrite_domain_from_pair_metadata_rows": dict(rewrite_domain_meta.most_common()),
        "compact_rewrite_domain_from_pair_metadata_word_weighted": dict(rewrite_word_domain_meta.most_common()),
        "compact_rewrite_domain_common_text_classifier_rows": dict(rewrite_domain_text.most_common()),
        "max_source_domain_common_text_classifier_rows": dict(source_domain_text.most_common()),
        "breadth_domain_from_source_metadata_rows": dict(breadth_domain_meta.most_common()),
        "breadth_domain_from_source_metadata_word_weighted": dict(breadth_word_domain_meta.most_common()),
        "breadth_domain_common_text_classifier_rows": dict(breadth_domain_text.most_common()),
        "compact_rewrite_soft_flags": dict(rewrite_flags.most_common()),
        "compact_source_risks": dict(source_risks.most_common()),
        "breadth_quality_flags": dict(breadth_flags.most_common()),
        "domain_js_bits_common_text_rewrite_vs_breadth": categorical_js(rewrite_domain_text, breadth_domain_text),
        "domain_js_bits_metadata_rewrite_vs_breadth": categorical_js(rewrite_domain_meta, breadth_domain_meta),
        "note": "Pair metadata domains are inherited from MAX source/rewrite pair fields and can be multi-label; breadth metadata domains are a single heuristic label from the FineWeb source side. The common text classifier is crude but applied symmetrically to rewrite, source, and breadth text.",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=CHANGED_ROWS)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()
    out_dir = pathlib.Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), use_fast=True)
    pat = Patterner(tok, SEQ_LENGTH)
    rows: dict[str, list[dict[str, Any]]] = {}
    for name, path in {"view": VIEW_POOL, "repeat": REPEAT_POOL, "breadth": BREADTH_POOL}.items():
        arr = []
        for obj in read_jsonl(path, args.rows):
            arr.append(pat.row(str(obj.get("text") or "")))
        rows[name] = arr
        print(json.dumps({"loaded": name, "rows": len(arr)}, ensure_ascii=False), flush=True)
    if not (len(rows["view"]) == len(rows["repeat"]) == len(rows["breadth"]) == args.rows):
        raise RuntimeError({k: len(v) for k, v in rows.items()})

    seqs: dict[str, dict[str, list[int]]] = {}
    group_sizes: dict[str, list[int]] = {}
    for name, arr in rows.items():
        seqs[name] = {k: [int(r[k]) for r in arr] for k in ["tokens_full", "tokens_visible", "wwm_groups"]}
        gs: list[int] = []
        for r in arr:
            gs.extend(int(x) for x in r["group_sizes"])
        group_sizes[name] = gs

    count_sequence = {}
    for name in rows:
        count_sequence[name] = {k: {"sha256": sequence_sha(v), "stats": stats([float(x) for x in v])} for k, v in seqs[name].items()}
    equality = {}
    for other in ["repeat", "breadth"]:
        equality[other + "_vs_view"] = {}
        for k in ["tokens_full", "tokens_visible", "wwm_groups"]:
            equality[other + "_vs_view"][k] = {
                "exact_equal_rows": sum(1 for a, b in zip(seqs["view"][k], seqs[other][k]) if a == b),
                "fraction": sum(1 for a, b in zip(seqs["view"][k], seqs[other][k]) if a == b) / args.rows,
            }

    pattern = {}
    for other in ["repeat", "breadth"]:
        dists: list[float] = []
        exact = 0
        for a, b in zip(rows["view"], rows[other]):
            ap = a["word_start_pattern"]; bp = b["word_start_pattern"]
            if ap == bp:
                exact += 1
            dists.append(pattern_distance(ap, bp))
        pattern[other + "_vs_view"] = {"exact_same_visible_word_start_pattern_rows": exact, "fraction_exact": exact / args.rows, "normalized_hamming_plus_length_distance": stats(dists)}

    group_size_out = {name: stats([float(x) for x in gs]) for name, gs in group_sizes.items()}
    group_size_out["breadth_minus_view_mean_group_size"] = group_size_out["breadth"].get("mean", 0) - group_size_out["view"].get("mean", 0)
    group_size_out["repeat_minus_view_mean_group_size"] = group_size_out["repeat"].get("mean", 0) - group_size_out["view"].get("mean", 0)

    dq = domain_quality()
    conditions = []
    bpat = pattern["breadth_vs_view"]
    if bpat["fraction_exact"] < 0.5:
        conditions.append(f"visible word-start pattern exact match is only {bpat['fraction_exact']:.4f}; rowwise subword-boundary geometry differs even though WWM group totals match")
    bg = group_size_out.get("breadth", {}).get("mean")
    vg = group_size_out.get("view", {}).get("mean")
    if bg and vg and abs(bg - vg) / vg > 0.01:
        conditions.append(f"mean WWM group token span differs by {100*(bg-vg)/vg:+.3f}%: selected groups expose different subword span lengths")
    if dq.get("domain_js_bits_common_text_rewrite_vs_breadth", 0) > 0.02:
        conditions.append(f"common text-domain distribution differs (JS={dq['domain_js_bits_common_text_rewrite_vs_breadth']:.4f} bits)")

    payload = {
        "status": "BREADTH_ROWPATTERN_DOMAIN_AUDIT_DONE",
        "created_utc": now(),
        "changed_rows": args.rows,
        "seq_length": SEQ_LENGTH,
        "inputs": {"view": rel(VIEW_POOL), "repeat": rel(REPEAT_POOL), "breadth": rel(BREADTH_POOL), "pairs": rel(PAIRS), "breadth_companions": rel(BREADTH_COMPANIONS)},
        "count_sequence_hashes_and_stats": count_sequence,
        "rowwise_count_sequence_equality_vs_view": equality,
        "visible_word_start_pattern_vs_view": pattern,
        "wwm_group_size_distribution_visible": group_size_out,
        "domain_and_quality": dq,
        "carried_conditions": conditions,
        "no_training_gpu_eval_superglue_aoa_upload_or_leaderboard": True,
    }
    (out_dir / "breadth_rowpattern_domain_audit.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# research MAX breadth row-pattern and domain audit",
        "",
        f"Changed rows: {args.rows}; seq_length {SEQ_LENGTH}.",
        "",
        "## Sequence hashes",
        "",
        "| arm | tokens_full_sha | tokens_visible_sha | wwm_groups_sha |",
        "|---|---|---|---|",
    ]
    for name in ["view", "repeat", "breadth"]:
        c = count_sequence[name]
        lines.append(f"| {name} | {c['tokens_full']['sha256']} | {c['tokens_visible']['sha256']} | {c['wwm_groups']['sha256']} |")
    lines += ["", "## Equality and pattern distance vs view", "", "| comparison | token-count exact | visible-count exact | WWM-count exact | word-start pattern exact | mean pattern distance |", "|---|---|---|---|---|---|"]
    for comp in ["repeat_vs_view", "breadth_vs_view"]:
        e = equality[comp]
        p = pattern[comp]
        lines.append(f"| {comp} | {e['tokens_full']['fraction']:.4f} | {e['tokens_visible']['fraction']:.4f} | {e['wwm_groups']['fraction']:.4f} | {p['fraction_exact']:.4f} | {p['normalized_hamming_plus_length_distance']['mean']:.4f} |")
    lines += ["", "## Visible WWM group size", "", "| arm | mean | sd | p05 | median | p95 |", "|---|---|---|---|---|---|"]
    for name in ["view", "repeat", "breadth"]:
        s = group_size_out[name]
        lines.append(f"| {name} | {s['mean']:.5f} | {s['sd']:.5f} | {s['p05']} | {s['median']} | {s['p95']} |")
    lines += ["", "## Domain and quality", "", f"Common text-domain JS rewrite vs breadth: `{dq['domain_js_bits_common_text_rewrite_vs_breadth']}`", f"Metadata-domain JS rewrite vs breadth: `{dq['domain_js_bits_metadata_rewrite_vs_breadth']}`", "", f"Rewrite domains by pair metadata: `{dq['compact_rewrite_domain_from_pair_metadata_rows']}`", f"Breadth domains by source metadata: `{dq['breadth_domain_from_source_metadata_rows']}`", f"Rewrite flags: `{dq['compact_rewrite_soft_flags']}`", f"Breadth flags: `{dq['breadth_quality_flags']}`", "", "## Conditions to carry", ""]
    lines.extend(f"- {c}" for c in conditions)
    (out_dir / "breadth_rowpattern_domain_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "equality": equality, "pattern": pattern, "group_size_mean": {k: v.get('mean') for k, v in group_size_out.items() if isinstance(v, dict)}, "conditions": conditions, "out": rel(out_dir)}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
