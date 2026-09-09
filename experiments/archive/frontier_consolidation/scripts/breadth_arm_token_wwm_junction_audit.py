#!/usr/bin/env python3
"""research: CPU audit of the MAX breadth allocation arm against the MAX view arm.

Whitespace row-length equality is insufficient evidence that the breadth arm
is geometry-matched.  Replacing one compact rewrite with 1-5
independent whole FineWeb sentences can change:

  * legal16k token counts per row (fertility),
  * how much of each row is visible after truncation to seq_length=256,
  * whole-word-masking group counts, i.e. actual mask opportunity,
  * sentence-junction/punctuation structure inside changed rows,
  * the lexical population of the replaced companion text.

This script measures all of those under the exact trainer semantics used by
`experiments/archive/compact_experience/scripts/masking_curriculum_trainer.py`:
tokenizer(add_special_tokens=False, truncation=True, max_length=seq_length,
padding="max_length"), and WWM groups incremented at word-start tokens
("Ġ"/"▁") over non-special visible positions.

It performs no training, GPU work, evaluation, SuperGLUE, AoA, upload, or
leaderboard action.  Output is a JSON/MD record that either clears the breadth
arm for one H100 run or records the exact altered condition.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import collections
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
BREADTH_META = WS / "data/dose_2p64x_breadth_rowholdout_pools/max_breadth_rowholdout_metadata.json"
BREADTH_COMPANIONS = WS / "data/dose_2p64x_breadth_rowholdout_pools/compact_breadth_dose2p64x_selected_companion_sources.jsonl"
PAIRS = WS / "data/dose_distribution_select/selected_matched_max_pairs.jsonl"
OUT_DIR = WS / "data/breadth_arm_geometry_audit"
SEQ_LENGTH = 256
CHANGED_ROWS = 7923
SENT_END = re.compile(r"[.!?]['\"\u201d\u2019)\]]*(\s|$)")
REWRITE_KEYS = ("rewrite_text", "view_text", "compact_text", "target_text", "rewrite")


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
            line = line.strip()
            if line:
                yield json.loads(line)


def stats(vals: list[float]) -> dict[str, Any]:
    if not vals:
        return {"n": 0}
    s = sorted(vals)

    def q(p: float) -> float:
        if len(s) == 1:
            return float(s[0])
        idx = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
        return float(s[idx])

    return {
        "n": len(s),
        "sum": float(sum(s)),
        "mean": float(statistics.mean(s)),
        "sd": float(statistics.pstdev(s)) if len(s) > 1 else 0.0,
        "min": float(s[0]),
        "p05": q(0.05),
        "median": q(0.50),
        "p95": q(0.95),
        "max": float(s[-1]),
    }


def paired_delta_stats(a: list[float], b: list[float]) -> dict[str, Any]:
    """b - a, paired by index."""
    if len(a) != len(b):
        return {"error": f"length mismatch {len(a)} vs {len(b)}"}
    d = [float(y) - float(x) for x, y in zip(a, b)]
    st = stats(d)
    st["frac_zero"] = sum(1 for x in d if abs(x) < 1e-12) / len(d) if d else None
    st["frac_pos"] = sum(1 for x in d if x > 0) / len(d) if d else None
    st["frac_neg"] = sum(1 for x in d if x < 0) / len(d) if d else None
    st["mean_abs"] = float(statistics.mean([abs(x) for x in d])) if d else None
    return st


class RowGeometry:
    def __init__(self, tokenizer, seq_length: int) -> None:
        self.tok = tokenizer
        self.seq_length = seq_length
        self.special_ids = set(tokenizer.all_special_ids)
        self._ws: dict[int, bool] = {}

    def word_start(self, tid: int) -> bool:
        v = self._ws.get(tid)
        if v is None:
            s = self.tok.convert_ids_to_tokens(int(tid))
            v = bool(s is not None and (str(s).startswith("\u0120") or str(s).startswith("\u2581")))
            self._ws[tid] = v
        return v

    def measure(self, text: str) -> dict[str, Any]:
        full = self.tok(text, add_special_tokens=False)["input_ids"]
        visible = full[: self.seq_length]
        gid = -1
        groups = 0
        maskable = 0
        for i, tid in enumerate(visible):
            if tid in self.special_ids:
                continue
            maskable += 1
            if gid < 0 or self.word_start(int(tid)) or i == 0:
                gid += 1
                groups += 1
        words = len(text.split())
        return {
            "words": words,
            "tokens_full": len(full),
            "tokens_visible": len(visible),
            "truncated": len(full) > self.seq_length,
            "tokens_dropped": max(0, len(full) - self.seq_length),
            "wwm_groups_visible": groups,
            "maskable_visible": maskable,
            "fertility_tokens_per_word": (len(full) / words) if words else None,
        }


def junction_profile(text: str) -> dict[str, Any]:
    return {
        "sentence_end_marks": len(SENT_END.findall(text)),
        "period": text.count("."),
        "comma": text.count(","),
        "question": text.count("?"),
        "exclam": text.count("!"),
        "semicolon_colon": text.count(";") + text.count(":"),
        "quote": text.count('"') + text.count("\u201c") + text.count("\u201d"),
        "digits": sum(ch.isdigit() for ch in text),
        "cap_words": sum(1 for w in text.split() if w[:1].isupper()),
    }


def lexical_profile(tokenizer, texts: list[str], label: str) -> dict[str, Any]:
    counts: collections.Counter[str] = collections.Counter()
    words_total = 0
    tokens_total = 0
    word_len: list[float] = []
    digits = 0
    caps = 0
    chars = 0
    seg_words: list[float] = []
    for t in texts:
        ws = t.split()
        words_total += len(ws)
        seg_words.append(float(len(ws)))
        counts.update(w.strip(".,;:!?\"'()[]").lower() for w in ws if w.strip(".,;:!?\"'()[]"))
        word_len.extend(float(len(w)) for w in ws)
        digits += sum(ch.isdigit() for ch in t)
        caps += sum(1 for w in ws if w[:1].isupper())
        chars += len(t)
        tokens_total += len(tokenizer(t, add_special_tokens=False)["input_ids"])
    return {
        "label": label,
        "segments": len(texts),
        "words": words_total,
        "tokens_legal16k": tokens_total,
        "fertility_tokens_per_word": (tokens_total / words_total) if words_total else None,
        "unique_word_types": len(counts),
        "type_token_ratio": (len(counts) / words_total) if words_total else None,
        "mean_word_chars": float(statistics.mean(word_len)) if word_len else None,
        "digit_chars_per_1k_chars": (1000.0 * digits / chars) if chars else None,
        "capitalized_word_fraction": (caps / words_total) if words_total else None,
        "segment_word_stats": stats(seg_words),
        "_counts": counts,
    }


def js_divergence(a: collections.Counter, b: collections.Counter) -> float:
    na = sum(a.values()) or 1
    nb = sum(b.values()) or 1
    keys = set(a) | set(b)
    out = 0.0
    for k in keys:
        p = a.get(k, 0) / na
        q = b.get(k, 0) / nb
        m = 0.5 * (p + q)
        if p > 0:
            out += 0.5 * p * math.log(p / m, 2)
        if q > 0:
            out += 0.5 * q * math.log(q / m, 2)
    return float(out)


def rewrite_text_of(pair: dict[str, Any]) -> str | None:
    for k in REWRITE_KEYS:
        v = pair.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=CHANGED_ROWS)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    missing = [rel(p) for p in [TOKENIZER_DIR, VIEW_POOL, REPEAT_POOL, BREADTH_POOL, BREADTH_META, BREADTH_COMPANIONS, PAIRS] if not p.exists()]
    plan = {
        "status": "BREADTH_GEOMETRY_AUDIT_PLAN",
        "created_utc": now(),
        "purpose": "Measure legal16k token, visible-window, WWM mask-opportunity, junction, and lexical-population differences between the MAX view arm and the MAX breadth allocation arm before any H100 launch.",
        "changed_rows_audited": args.rows,
        "seq_length": SEQ_LENGTH,
        "missing_inputs": missing,
        "no_training_gpu_eval_superglue_aoa_upload_or_leaderboard": True,
    }
    (out_dir / "breadth_geometry_audit_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if args.plan_only:
        return
    if missing:
        raise FileNotFoundError(f"missing inputs: {missing}")

    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), use_fast=True)
    geo = RowGeometry(tok, SEQ_LENGTH)

    view_rows = list(read_jsonl(VIEW_POOL, args.rows))
    breadth_rows = list(read_jsonl(BREADTH_POOL, args.rows))
    repeat_rows = list(read_jsonl(REPEAT_POOL, args.rows))
    if not (len(view_rows) == len(breadth_rows) == len(repeat_rows) == args.rows):
        raise RuntimeError(f"row count mismatch: view={len(view_rows)} breadth={len(breadth_rows)} repeat={len(repeat_rows)}")

    per_arm: dict[str, dict[str, list[float]]] = {}
    junction: dict[str, dict[str, list[float]]] = {}
    for name, rows in [("view", view_rows), ("repeat", repeat_rows), ("breadth", breadth_rows)]:
        acc: dict[str, list[float]] = collections.defaultdict(list)
        jacc: dict[str, list[float]] = collections.defaultdict(list)
        for r in rows:
            text = str(r.get("text") or "")
            m = geo.measure(text)
            for k in ["words", "tokens_full", "tokens_visible", "tokens_dropped", "wwm_groups_visible", "maskable_visible"]:
                acc[k].append(float(m[k]))
            acc["truncated"].append(1.0 if m["truncated"] else 0.0)
            if m["fertility_tokens_per_word"] is not None:
                acc["fertility_tokens_per_word"].append(float(m["fertility_tokens_per_word"]))
            for k, v in junction_profile(text).items():
                jacc[k].append(float(v))
        per_arm[name] = dict(acc)
        junction[name] = dict(jacc)

    word_equal_view_breadth = per_arm["view"]["words"] == per_arm["breadth"]["words"]

    geometry = {
        "per_arm_row_stats": {arm: {k: stats(v) for k, v in acc.items()} for arm, acc in per_arm.items()},
        "paired_breadth_minus_view": {k: paired_delta_stats(per_arm["view"][k], per_arm["breadth"][k]) for k in ["words", "tokens_full", "tokens_visible", "tokens_dropped", "wwm_groups_visible", "maskable_visible", "truncated"]},
        "paired_repeat_minus_view": {k: paired_delta_stats(per_arm["view"][k], per_arm["repeat"][k]) for k in ["tokens_full", "tokens_visible", "wwm_groups_visible", "maskable_visible"]},
        "whitespace_word_sequence_identical_view_vs_breadth": bool(word_equal_view_breadth),
        "changed_block_token_totals": {arm: float(sum(per_arm[arm]["tokens_full"])) for arm in per_arm},
        "changed_block_visible_token_totals": {arm: float(sum(per_arm[arm]["tokens_visible"])) for arm in per_arm},
        "changed_block_wwm_group_totals": {arm: float(sum(per_arm[arm]["wwm_groups_visible"])) for arm in per_arm},
    }
    for arm in ["repeat", "breadth"]:
        vt = geometry["changed_block_token_totals"]["view"]
        vg = geometry["changed_block_wwm_group_totals"]["view"]
        geometry.setdefault("relative_changed_block_shift_vs_view", {})[arm] = {
            "tokens_full_pct": 100.0 * (geometry["changed_block_token_totals"][arm] - vt) / vt if vt else None,
            "wwm_groups_pct": 100.0 * (geometry["changed_block_wwm_group_totals"][arm] - vg) / vg if vg else None,
            "visible_tokens_pct": 100.0 * (geometry["changed_block_visible_token_totals"][arm] - geometry["changed_block_visible_token_totals"]["view"]) / geometry["changed_block_visible_token_totals"]["view"] if geometry["changed_block_visible_token_totals"]["view"] else None,
        }

    junction_out = {
        "per_arm": {arm: {k: stats(v) for k, v in acc.items()} for arm, acc in junction.items()},
        "paired_breadth_minus_view": {k: paired_delta_stats(junction["view"][k], junction["breadth"][k]) for k in junction["view"]},
    }

    # Replaced-companion lexical population: compact rewrites vs selected breadth sentences.
    rewrites: list[str] = []
    rewrite_key_used = None
    missing_rewrite = 0
    for p in read_jsonl(PAIRS):
        t = rewrite_text_of(p)
        if t is None:
            missing_rewrite += 1
            continue
        if rewrite_key_used is None:
            for k in REWRITE_KEYS:
                if isinstance(p.get(k), str) and p[k].strip():
                    rewrite_key_used = k
                    break
        rewrites.append(t)
    breadth_sents = [str(r.get("text") or "") for r in read_jsonl(BREADTH_COMPANIONS)]
    sources = [str(p.get("source_text") or "") for p in read_jsonl(PAIRS)]

    lex_rewrite = lexical_profile(tok, rewrites, "max_compact_rewrites")
    lex_breadth = lexical_profile(tok, breadth_sents, "max_selected_breadth_sentences")
    lex_source = lexical_profile(tok, sources, "max_source_sentences")
    cr, cb, cs = lex_rewrite.pop("_counts"), lex_breadth.pop("_counts"), lex_source.pop("_counts")
    lexical = {
        "profiles": [lex_rewrite, lex_breadth, lex_source],
        "unigram_js_divergence_bits": {
            "rewrite_vs_breadth": js_divergence(cr, cb),
            "rewrite_vs_source": js_divergence(cr, cs),
            "breadth_vs_source": js_divergence(cb, cs),
        },
        "companion_word_budget_match": {
            "rewrite_words": lex_rewrite["words"],
            "breadth_words": lex_breadth["words"],
            "equal": lex_rewrite["words"] == lex_breadth["words"],
        },
        "rewrite_key_used": rewrite_key_used,
        "pairs_without_rewrite_text": missing_rewrite,
    }

    meta = json.loads(BREADTH_META.read_text(encoding="utf-8"))
    sel = meta.get("selection") or {}
    population = {
        "selected_doc_overlap_fraction": sel.get("selected_doc_overlap_fraction"),
        "selected_unique_docs": sel.get("selected_unique_docs"),
        "doc_cap": sel.get("doc_cap"),
        "selected_flag_rows": sel.get("selected_flag_rows"),
        "candidate_flag_rows": (meta.get("candidate_summary") or {}).get("flag_rows"),
        "selected_sentence_word_stats": sel.get("selected_sentence_word_stats"),
        "candidate_word_stats": (meta.get("candidate_summary") or {}).get("word_stats"),
        "selection_is_length_biased": None,
        "interpretation": "Breadth is sentence-level additional experience drawn largely from the same FineWeb document population, not new-document breadth.",
    }
    cw = population["candidate_word_stats"] or {}
    sw = population["selected_sentence_word_stats"] or {}
    if cw.get("mean") and sw.get("mean"):
        population["selection_is_length_biased"] = {
            "candidate_mean_words": cw["mean"],
            "selected_mean_words": sw["mean"],
            "selected_minus_candidate_mean_words": float(sw["mean"]) - float(cw["mean"]),
            "note": "Row budgets force longer sentences; the breadth arm therefore samples the longer tail of the same population.",
        }
    if population["candidate_flag_rows"] and population["selected_flag_rows"]:
        c = population["candidate_flag_rows"]
        s = population["selected_flag_rows"]
        ctot = sum(c.values()) or 1
        stot = sum(s.values()) or 1
        population["flagged_fraction"] = {"candidate": c.get("flagged", 0) / ctot, "selected": s.get("flagged", 0) / stot}

    # Condition judgment: what must be carried explicitly if the arm is trained.
    tok_shift = (geometry.get("relative_changed_block_shift_vs_view") or {}).get("breadth", {})
    conditions: list[str] = []
    for key, label, tol in [("tokens_full_pct", "changed-block legal16k tokens", 1.0), ("wwm_groups_pct", "changed-block WWM mask opportunity", 1.0), ("visible_tokens_pct", "changed-block visible tokens after seq256", 1.0)]:
        v = tok_shift.get(key)
        if v is None:
            conditions.append(f"{label}: not measured")
        elif abs(float(v)) > tol:
            conditions.append(f"{label} differs by {float(v):+.3f}% (> {tol}% tolerance): carry as an explicit condition of view-minus-breadth")
    jd = junction_out["paired_breadth_minus_view"].get("sentence_end_marks", {})
    if isinstance(jd.get("mean"), float) and abs(jd["mean"]) > 0.25:
        conditions.append(f"changed rows contain {jd['mean']:+.3f} sentence-final marks per row versus view: boundary geometry differs")
    if population.get("selected_doc_overlap_fraction") and float(population["selected_doc_overlap_fraction"]) > 0.5:
        conditions.append("breadth material shares documents with MAX sources for a large fraction of selected sentences: this is same-population sentence breadth, not new-document breadth")
    lexjs = lexical["unigram_js_divergence_bits"]["rewrite_vs_breadth"]
    if lexjs > 0.05:
        conditions.append(f"replaced companion unigram distribution differs from compact rewrites (JS={lexjs:.4f} bits): allocation contrast includes a lexical-population change")

    payload = {
        "status": "BREADTH_GEOMETRY_AUDIT_DONE",
        "created_utc": now(),
        "inputs": {"tokenizer": rel(TOKENIZER_DIR), "view_pool": rel(VIEW_POOL), "repeat_pool": rel(REPEAT_POOL), "breadth_pool": rel(BREADTH_POOL), "pairs": rel(PAIRS), "breadth_companions": rel(BREADTH_COMPANIONS)},
        "changed_rows_audited": args.rows,
        "seq_length": SEQ_LENGTH,
        "geometry": geometry,
        "junction": junction_out,
        "lexical_population": lexical,
        "source_population": population,
        "carried_conditions": conditions,
        "arm_is_word_and_row_matched": bool(word_equal_view_breadth),
        "no_training_gpu_eval_superglue_aoa_upload_or_leaderboard": True,
    }
    (out_dir / "breadth_geometry_audit.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def g(path: list[str], d: Any = None) -> Any:
        cur: Any = payload
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                return d
            cur = cur[k]
        return cur

    lines = [
        "# research MAX breadth arm geometry audit (view vs breadth, changed rows only)",
        "",
        f"Rows audited: {args.rows}; seq_length {SEQ_LENGTH}; fixed research legal16k tokenizer.",
        f"Whitespace word sequence identical view vs breadth: `{word_equal_view_breadth}`",
        "",
        "## Changed-block totals",
        "",
        "| quantity | view | repeat | breadth | breadth vs view |",
        "|---|---|---|---|---|",
    ]
    for label, key, pct in [("legal16k tokens", "changed_block_token_totals", "tokens_full_pct"), ("visible tokens (seq256)", "changed_block_visible_token_totals", "visible_tokens_pct"), ("WWM groups", "changed_block_wwm_group_totals", "wwm_groups_pct")]:
        row = g(["geometry", key], {}) or {}
        shift = tok_shift.get(pct)
        lines.append(f"| {label} | {row.get('view')} | {row.get('repeat')} | {row.get('breadth')} | {f'{float(shift):+.3f}%' if shift is not None else 'n/a'} |")
    lines += [
        "",
        "## Paired per-row differences (breadth minus view)",
        "",
        "| quantity | mean | mean_abs | sd | frac_zero | p05 | p95 |",
        "|---|---|---|---|---|---|---|",
    ]
    for k in ["tokens_full", "tokens_visible", "wwm_groups_visible", "maskable_visible", "tokens_dropped"]:
        d = g(["geometry", "paired_breadth_minus_view", k], {}) or {}
        lines.append(f"| {k} | {d.get('mean')} | {d.get('mean_abs')} | {d.get('sd')} | {d.get('frac_zero')} | {d.get('p05')} | {d.get('p95')} |")
    lines += ["", "## Junction / punctuation (breadth minus view, per changed row)", "", "| mark | mean delta | view mean | breadth mean |", "|---|---|---|---|"]
    for k in ["sentence_end_marks", "period", "comma", "digits", "cap_words"]:
        d = g(["junction", "paired_breadth_minus_view", k], {}) or {}
        vm = (g(["junction", "per_arm", "view", k], {}) or {}).get("mean")
        bm = (g(["junction", "per_arm", "breadth", k], {}) or {}).get("mean")
        lines.append(f"| {k} | {d.get('mean')} | {vm} | {bm} |")
    lines += ["", "## Replaced companion population", ""]
    for prof in lexical["profiles"]:
        lines.append(f"- `{prof['label']}`: segments {prof['segments']}, words {prof['words']}, tokens {prof['tokens_legal16k']}, fertility {prof['fertility_tokens_per_word']}, TTR {prof['type_token_ratio']}, mean word chars {prof['mean_word_chars']}, capitalized fraction {prof['capitalized_word_fraction']}")
    lines += [
        "",
        f"Unigram JS divergence (bits): rewrite vs breadth {lexical['unigram_js_divergence_bits']['rewrite_vs_breadth']:.4f}; rewrite vs source {lexical['unigram_js_divergence_bits']['rewrite_vs_source']:.4f}; breadth vs source {lexical['unigram_js_divergence_bits']['breadth_vs_source']:.4f}",
        f"Selected breadth document overlap with MAX sources: {population.get('selected_doc_overlap_fraction')}",
        "",
        "## Conditions to carry with any view-minus-breadth reading",
        "",
    ]
    lines += [f"- {c}" for c in conditions] or ["- none above tolerance"]
    (out_dir / "breadth_geometry_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k not in {"geometry", "junction", "lexical_population"}}, indent=2, ensure_ascii=False), flush=True)
    print(json.dumps({"changed_block_totals": geometry["changed_block_token_totals"], "wwm_totals": geometry["changed_block_wwm_group_totals"], "relative_shift_vs_view": geometry.get("relative_changed_block_shift_vs_view")}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
