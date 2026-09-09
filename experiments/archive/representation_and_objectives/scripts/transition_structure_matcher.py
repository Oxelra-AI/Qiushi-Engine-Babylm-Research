#!/usr/bin/env python3
"""research: re-derive compact-contained transition/control contrast.

Scientific purpose
------------------
research showed that a legal warm-start relative-exposure probe is feasible only on
sentences already exact-contained in the compact 10M pool, but the inherited
research 30k treatment/control files are not a ready matched pair.  This script
asks whether the compact-contained subset can isolate transition structure from
source/style well enough to justify any later GPU continuation.

It performs CPU-only source/style matching.  It does not launch training and does
not read official evaluation item text.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path.cwd()
WS = ROOT / "experiments/archive/representation_and_objectives"
IN_T = WS / "data/transition_containment/transition_exact_in_compact.jsonl"
IN_C = WS / "data/transition_containment/anchor_control_exact_in_compact.jsonl"
COMPACT_10M = WS / "data/fw_full_arms/fw_preserved_compact_view_10M.jsonl"
OUT = WS / "data/transition_structure_match"
NOTE = WS / "notes/transition_structure_match.md"

WORD_RE = re.compile(r"\b[\w]+(?:['’-][\w]+)?\b", re.UNICODE)
PROPER_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")
DIGIT_RE = re.compile(r"\b\d{1,4}\b")

TRANSITION_PATTERNS = [
    r"\bif\b", r"\bwhen\b", r"\bwhenever\b", r"\bunless\b", r"\bonce\b",
    r"\bafter\b", r"\bbefore\b", r"\bbecause\b", r"\bsince\b", r"\bdue to\b",
    r"\bas a result\b", r"\btherefore\b", r"\bso that\b", r"\bin order to\b",
    r"\bleads? to\b", r"\bcaus(?:e|es|ed|ing)\b", r"\bresult(?:s|ed|ing)?\b",
    r"\bprevent(?:s|ed|ing)?\b", r"\ballow(?:s|ed|ing)?\b",
    r"\bbecome(?:s|ing)?\b", r"\bbecame\b", r"\bturn(?:s|ed|ing)? into\b",
    r"\bchange(?:s|d|ing)?\b", r"\bgo(?:es|ing)? from\b", r"\bwent from\b",
    r"\bfrom\b.{0,40}\bto\b", r"\bincreas(?:e|es|ed|ing)\b", r"\bdecreas(?:e|es|ed|ing)\b",
    r"\bfill(?:s|ed|ing)?\b", r"\bempty(?:ies|ied|ing)?\b", r"\bopen(?:s|ed|ing)?\b",
    r"\bclos(?:e|es|ed|ing)\b", r"\bheat(?:s|ed|ing)?\b", r"\bcool(?:s|ed|ing)?\b",
    r"\bfreez(?:e|es|ing)\b", r"\bmelt(?:s|ed|ing)?\b", r"\bdissolv(?:e|es|ed|ing)\b",
    r"\bbreak(?:s|ing|s up)?\b", r"\bbroke\b", r"\bbroken\b", r"\bbend(?:s|ing|ed)?\b",
    r"\bstretch(?:es|ed|ing)?\b", r"\btear(?:s|ing|tore)?\b", r"\bwet(?:s|ted|ting)?\b",
    r"\bdry(?:ies|ied|ing)?\b",
]
PHYSICAL_PATTERNS = [
    r"\bpush(?:ed|es|ing)?\b", r"\bpull(?:ed|s|ing)?\b", r"\bthrow(?:s|ing|n)?\b", r"\bthrew\b",
    r"\bdrop(?:ped|s|ping)?\b", r"\bfall(?:s|ing|en)?\b", r"\bfell\b", r"\bbounce(?:s|d|ing)?\b",
    r"\broll(?:s|ed|ing)?\b", r"\bfloat(?:s|ed|ing)?\b", r"\bsink(?:s|ing|sank)?\b", r"\bslide(?:s|d|ing)?\b",
    r"\bpour(?:s|ed|ing)?\b", r"\bspill(?:s|ed|ing)?\b", r"\bcover(?:s|ed|ing)?\b", r"\bblock(?:s|ed|ing)?\b",
    r"\bwater\b", r"\bair\b", r"\bglass\b", r"\bmetal\b", r"\bplastic\b", r"\bwood(?:en)?\b", r"\bpaper\b",
    r"\bfabric\b", r"\bcloth\b", r"\brubber\b", r"\bliquid\b", r"\bsolid\b", r"\bgas\b", r"\bice\b",
    r"\bsteam\b", r"\bfire\b", r"\blight\b", r"\bshadow\b", r"\bsoil\b", r"\bstone\b", r"\bsand\b",
    r"\bbag\b", r"\bcup\b", r"\bbox\b", r"\bcontainer\b", r"\bbottle\b", r"\bwindow\b", r"\bdoor\b",
    r"\bball\b", r"\bpipe\b", r"\brope\b", r"\bstring\b", r"\bwire\b", r"\bwheel\b", r"\btool\b",
    r"\bhot\b", r"\bcold\b", r"\bwet\b", r"\bdry\b", r"\bsoft\b", r"\bhard\b", r"\bheavy\b", r"\blight\b",
]
SPATIAL_PATTERNS = [
    r"\bleft\b", r"\bright\b", r"\bup\b", r"\bdown\b", r"\babove\b", r"\bbelow\b", r"\bunder\b", r"\bover\b",
    r"\binside\b", r"\boutside\b", r"\bin front of\b", r"\bbehind\b", r"\bbetween\b", r"\bthrough\b", r"\bacross\b",
    r"\baround\b", r"\btowards?\b", r"\baway from\b", r"\bnear\b", r"\bfar\b", r"\bcloser\b", r"\bfurther\b",
]
QUANTITY_PATTERNS = [
    r"\bmore\b", r"\bless\b", r"\bfewer\b", r"\badd(?:s|ed|ing)?\b", r"\bsubtract(?:s|ed|ing)?\b",
    r"\bmultiply\b", r"\bdivide(?:s|d|ing)?\b", r"\btotal\b", r"\bevery\b", r"\beach\b", r"\b\d{1,4}\b",
]
PRONOUNS = {"i", "me", "my", "mine", "we", "us", "our", "you", "your", "he", "him", "his", "she", "her", "hers", "they", "them", "their", "it", "its"}

REGIMES = {
    "coarse_source_style": ["source_label", "origin_source", "length_bin", "required_capability"],
    "source_origin_len_cap_digit": ["source_label", "origin_source", "length_bin", "required_capability", "digit_bin"],
    "source_origin_len_cap_quote": ["source_label", "origin_source", "length_bin", "required_capability", "quote_bin"],
    "source_origin_len_cap_proper": ["source_label", "origin_source", "length_bin", "required_capability", "proper_bin"],
    "source_origin_len_cap_pronoun": ["source_label", "origin_source", "length_bin", "required_capability", "pronoun_bin"],
    "source_origin_len_cap_comma": ["source_label", "origin_source", "length_bin", "required_capability", "comma_bin"],
    "medium_style": ["source_label", "origin_source", "length_bin", "required_capability", "quote_bin", "proper_bin", "digit_bin"],
    "surface_no_comma": ["source_label", "origin_source", "length_bin", "required_capability", "quote_bin", "proper_bin", "pronoun_bin", "digit_bin"],
    "surface_no_quote": ["source_label", "origin_source", "length_bin", "required_capability", "proper_bin", "pronoun_bin", "digit_bin", "comma_bin"],
    "strict_style": ["source_label", "origin_source", "length_bin", "required_capability", "quote_bin", "proper_bin", "pronoun_bin", "digit_bin", "comma_bin"],
}


def regex_count(patterns: Iterable[str], text: str) -> int:
    return sum(len(re.findall(p, text, flags=re.I)) for p in patterns)


def norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def stable_key(text: str) -> str:
    return hashlib.sha256(norm_text(text).lower().encode("utf-8")).hexdigest()


def bin_count(n: int, cuts: tuple[int, ...]) -> str:
    if n <= cuts[0]:
        return f"0_{cuts[0]}"
    lo = cuts[0] + 1
    for c in cuts[1:]:
        if n <= c:
            return f"{lo}_{c}"
        lo = c + 1
    return f"{lo}_plus"


def style_features(text: str) -> dict[str, Any]:
    toks = WORD_RE.findall(text)
    low_toks = [t.lower() for t in toks]
    quote_chars = sum(text.count(ch) for ch in ['"', "'", "“", "”", "‘", "’"])
    proper = len(PROPER_RE.findall(text))
    digits = len(DIGIT_RE.findall(text))
    pronouns = sum(1 for t in low_toks if t in PRONOUNS)
    commas = text.count(",")
    transition_markers = regex_count(TRANSITION_PATTERNS, text)
    physical = regex_count(PHYSICAL_PATTERNS, text)
    spatial = regex_count(SPATIAL_PATTERNS, text)
    quantity = regex_count(QUANTITY_PATTERNS, text)
    return {
        "token_words": len(toks),
        "quote_chars": quote_chars,
        "quote_bin": bin_count(quote_chars, (0, 2, 6)),
        "proper_count": proper,
        "proper_bin": bin_count(proper, (0, 3, 7)),
        "digit_count": digits,
        "digit_bin": bin_count(digits, (0, 1, 3)),
        "pronoun_count": pronouns,
        "pronoun_bin": bin_count(pronouns, (0, 3, 8)),
        "comma_count": commas,
        "comma_bin": bin_count(commas, (0, 3, 8)),
        "transition_marker_count_recomputed": transition_markers,
        "physical_anchor_count": physical,
        "spatial_anchor_count": spatial,
        "quantity_anchor_count": quantity,
    }


def load_rows(path: Path, arm: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for local_index, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            text = norm_text(str(obj["text"]))
            actual = len(text.split())
            field = int(obj.get("words", actual))
            if actual <= 0:
                continue
            out = dict(obj)
            out["arm"] = arm
            out["local_index"] = local_index
            out["text"] = text
            out["words"] = actual
            out["words_field"] = field
            out["word_field_matches_actual"] = (field == actual)
            loc = out.get("compact_location") or {}
            out["compact_row_index"] = loc.get("row_index")
            out["compact_line_index"] = loc.get("line_index")
            out["compact_row_source"] = loc.get("row_source")
            out["text_sha256"] = out.get("text_sha256") or stable_key(text)
            out.update(style_features(text))
            rows.append(out)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k); keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def key_for(row: dict[str, Any], fields: list[str]) -> tuple[Any, ...]:
    return tuple(row.get(f) for f in fields)


def reachable_dp(rows: list[dict[str, Any]], cap: int) -> dict[int, tuple[int, int] | None]:
    """Return reachable sums <= cap with predecessor (previous_sum, row_index)."""
    dp: dict[int, tuple[int, int] | None] = {0: None}
    for i, row in enumerate(rows):
        w = int(row["words"])
        # Iterate over a snapshot so every row is used at most once.
        for s in sorted(list(dp.keys()), reverse=True):
            ns = s + w
            if ns <= cap and ns not in dp:
                dp[ns] = (s, i)
    return dp


def reconstruct(dp: dict[int, tuple[int, int] | None], target: int, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    inds: list[int] = []
    s = target
    while s:
        prev = dp.get(s)
        if prev is None:
            raise RuntimeError(f"cannot reconstruct sum {target}")
        ps, i = prev
        inds.append(i)
        s = ps
    return [rows[i] for i in reversed(inds)]


def group_rows(rows: list[dict[str, Any]], fields: list[str]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[key_for(r, fields)].append(r)
    for vals in grouped.values():
        vals.sort(key=lambda r: (r["text_sha256"], r["local_index"]))
    return grouped


def matched_by_regime(t_rows: list[dict[str, Any]], c_rows: list[dict[str, Any]], regime: str, fields: list[str]) -> dict[str, Any]:
    gt = group_rows(t_rows, fields)
    gc = group_rows(c_rows, fields)
    keys = sorted(set(gt) & set(gc), key=lambda x: tuple(str(v) for v in x))
    selected_t: list[dict[str, Any]] = []
    selected_c: list[dict[str, Any]] = []
    strata_rows: list[dict[str, Any]] = []
    skipped = 0
    for k in keys:
        rt = gt[k]
        rc = gc[k]
        cap = min(sum(int(r["words"]) for r in rt), sum(int(r["words"]) for r in rc))
        if cap <= 0:
            continue
        dpt = reachable_dp(rt, cap)
        dpc = reachable_dp(rc, cap)
        common_sums = sorted(set(dpt) & set(dpc))
        target = common_sums[-1] if common_sums else 0
        if target <= 0:
            skipped += 1
            continue
        st = reconstruct(dpt, target, rt)
        sc = reconstruct(dpc, target, rc)
        for r in st:
            q = dict(r); q["match_regime"] = regime; q["match_key"] = repr(k); selected_t.append(q)
        for r in sc:
            q = dict(r); q["match_regime"] = regime; q["match_key"] = repr(k); selected_c.append(q)
        strata_rows.append({
            "regime": regime,
            "match_key": repr(k),
            "target_words_each_arm": target,
            "transition_rows_available": len(rt),
            "control_rows_available": len(rc),
            "transition_words_available": sum(int(r["words"]) for r in rt),
            "control_words_available": sum(int(r["words"]) for r in rc),
            "transition_rows_selected": len(st),
            "control_rows_selected": len(sc),
        })
    return {
        "regime": regime,
        "fields": fields,
        "selected_transition": selected_t,
        "selected_control": selected_c,
        "strata_rows": strata_rows,
        "common_strata": len(keys),
        "skipped_common_strata": skipped,
    }


def counter_words(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    c = Counter()
    for r in rows:
        c[str(r.get(key))] += int(r["words"])
    return dict(c)


def counter_count(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    c = Counter()
    for r in rows:
        c[str(r.get(key))] += 1
    return dict(c)


def mean(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [float(r.get(key, 0.0) or 0.0) for r in rows]
    return statistics.fmean(vals) if vals else None


def l1_from_word_counters(a: dict[str, int], b: dict[str, int]) -> float | None:
    sa, sb = sum(a.values()), sum(b.values())
    if not sa or not sb:
        return None
    keys = set(a) | set(b)
    return sum(abs(a.get(k, 0) / sa - b.get(k, 0) / sb) for k in keys)


def summarize_pair(name: str, st: list[dict[str, Any]], sc: list[dict[str, Any]], fields: list[str]) -> dict[str, Any]:
    tw = sum(int(r["words"]) for r in st)
    cw = sum(int(r["words"]) for r in sc)
    rows_t = {r.get("compact_row_index") for r in st if r.get("compact_row_index") is not None}
    rows_c = {r.get("compact_row_index") for r in sc if r.get("compact_row_index") is not None}
    style_l1 = {}
    for f in ["source_label", "origin_source", "length_bin", "required_capability", "quote_bin", "proper_bin", "pronoun_bin", "digit_bin", "comma_bin"]:
        style_l1[f] = l1_from_word_counters(counter_words(st, f), counter_words(sc, f))
    return {
        "regime": name,
        "fields": fields,
        "transition_sentences": len(st),
        "control_sentences": len(sc),
        "transition_words": tw,
        "control_words": cw,
        "exact_word_parity": tw == cw,
        "compact_row_overlap_selected": len(rows_t & rows_c),
        "compact_rows_transition": len(rows_t),
        "compact_rows_control": len(rows_c),
        "mean_transition_markers_transition": mean(st, "transition_marker_count_recomputed"),
        "mean_transition_markers_control": mean(sc, "transition_marker_count_recomputed"),
        "mean_physical_anchor_transition": mean(st, "physical_anchor_count"),
        "mean_physical_anchor_control": mean(sc, "physical_anchor_count"),
        "mean_spatial_anchor_transition": mean(st, "spatial_anchor_count"),
        "mean_spatial_anchor_control": mean(sc, "spatial_anchor_count"),
        "mean_quantity_anchor_transition": mean(st, "quantity_anchor_count"),
        "mean_quantity_anchor_control": mean(sc, "quantity_anchor_count"),
        "words_by_source_label_transition": counter_words(st, "source_label"),
        "words_by_source_label_control": counter_words(sc, "source_label"),
        "words_by_origin_source_transition": counter_words(st, "origin_source"),
        "words_by_origin_source_control": counter_words(sc, "origin_source"),
        "words_by_route_bucket_transition": counter_words(st, "selected_route_bucket"),
        "sentences_by_route_bucket_transition": counter_count(st, "selected_route_bucket"),
        "style_l1_by_words": style_l1,
    }


def qsummary(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    return {
        "arm": arm,
        "sentences": len(rows),
        "words": sum(int(r["words"]) for r in rows),
        "field_mismatches": sum(1 for r in rows if not r.get("word_field_matches_actual")),
        "by_source_label_words": counter_words(rows, "source_label"),
        "by_origin_source_words": counter_words(rows, "origin_source"),
        "by_length_bin_words": counter_words(rows, "length_bin"),
        "by_required_capability_words": counter_words(rows, "required_capability"),
        "by_quote_bin_words": counter_words(rows, "quote_bin"),
        "by_proper_bin_words": counter_words(rows, "proper_bin"),
        "by_pronoun_bin_words": counter_words(rows, "pronoun_bin"),
        "by_digit_bin_words": counter_words(rows, "digit_bin"),
        "mean_transition_markers": mean(rows, "transition_marker_count_recomputed"),
        "mean_physical_anchor": mean(rows, "physical_anchor_count"),
        "mean_spatial_anchor": mean(rows, "spatial_anchor_count"),
        "mean_quantity_anchor": mean(rows, "quantity_anchor_count"),
    }


def compact_row_lengths_and_exclusions(selected_for_exclusion: list[dict[str, Any]], target_words: int) -> dict[str, Any]:
    """Find an equal-word whole-row filler removal for one added extra subset.

    The exclusion set contains rows holding either arm's selected sentences so a
    common filler deletion does not remove examples that one branch is trying to
    add.  The word target is per arm, not transition+control combined.
    """
    target = int(target_words)
    selected_rows = {int(r["compact_row_index"]) for r in selected_for_exclusion if r.get("compact_row_index") is not None}
    rows: list[dict[str, Any]] = []
    with COMPACT_10M.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            text = norm_text(str(obj["text"]))
            w = len(text.split())
            if idx in selected_rows:
                continue
            rows.append({"row_index": idx, "line_index": obj.get("example_id"), "source": obj.get("source"), "words": w, "text_sha256": stable_key(text)})
    # Prefer whole 160-word rows where possible; DP only needs to reach about 25k.
    rows.sort(key=lambda r: (abs(int(r["words"]) - 160), str(r.get("source")), r["row_index"]))
    dp = reachable_dp(rows, target)
    if target in dp:
        filler = reconstruct(dp, target, rows)
    else:
        best = max(dp.keys())
        filler = reconstruct(dp, best, rows)
    return {
        "target_words": target,
        "exact_filler_possible": target in dp,
        "filler_words": sum(int(r["words"]) for r in filler),
        "filler_rows": len(filler),
        "excluded_selected_compact_rows": len(selected_rows),
        "candidate_rows_scanned": len(rows),
        "filler_rows_manifest": filler[:300],
    }


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    t_rows = load_rows(IN_T, "transition")
    c_rows = load_rows(IN_C, "anchor_control")

    regime_payloads = []
    summary_rows = []
    pair_summaries = {}
    preferred = None

    for name, fields in REGIMES.items():
        res = matched_by_regime(t_rows, c_rows, name, fields)
        st = res["selected_transition"]
        sc = res["selected_control"]
        write_jsonl(OUT / f"{name}_transition.jsonl", st)
        write_jsonl(OUT / f"{name}_anchor_control.jsonl", sc)
        write_csv(OUT / f"{name}_strata.csv", res["strata_rows"])
        ps = summarize_pair(name, st, sc, fields)
        ps.update({
            "transition_jsonl": str(OUT / f"{name}_transition.jsonl"),
            "anchor_control_jsonl": str(OUT / f"{name}_anchor_control.jsonl"),
            "strata_csv": str(OUT / f"{name}_strata.csv"),
            "common_strata": res["common_strata"],
            "skipped_common_strata": res["skipped_common_strata"],
        })
        pair_summaries[name] = ps
        regime_payloads.append({k: v for k, v in res.items() if k not in {"selected_transition", "selected_control"}})
        row = {
            "regime": name,
            "fields": ";".join(fields),
            "transition_sentences": ps["transition_sentences"],
            "control_sentences": ps["control_sentences"],
            "words_each_arm": ps["transition_words"],
            "exact_word_parity": ps["exact_word_parity"],
            "source_label_l1": ps["style_l1_by_words"]["source_label"],
            "origin_source_l1": ps["style_l1_by_words"]["origin_source"],
            "length_bin_l1": ps["style_l1_by_words"]["length_bin"],
            "quote_bin_l1": ps["style_l1_by_words"]["quote_bin"],
            "proper_bin_l1": ps["style_l1_by_words"]["proper_bin"],
            "pronoun_bin_l1": ps["style_l1_by_words"]["pronoun_bin"],
            "digit_bin_l1": ps["style_l1_by_words"]["digit_bin"],
            "comma_bin_l1": ps["style_l1_by_words"]["comma_bin"],
            "mean_transition_markers_transition": ps["mean_transition_markers_transition"],
            "mean_transition_markers_control": ps["mean_transition_markers_control"],
        }
        summary_rows.append(row)

    # Scientific preference favors source/style isolation over raw volume.  Coarse
    # matching is useful as an upper-volume reference but still leaves quote,
    # pronoun, comma, and digit imbalance.  The medium pair is the main defensible
    # mechanism contrast; the stricter pairs are too small for a standalone run but
    # useful for interpretation.
    if pair_summaries["medium_style"]["transition_words"] >= 10_000:
        preferred_name = "medium_style"
    elif pair_summaries["surface_no_comma"]["transition_words"] >= 8_000:
        preferred_name = "surface_no_comma"
    else:
        preferred_name = "strict_style"
    preferred = pair_summaries[preferred_name]
    preferred_transition = [json.loads(line) for line in (OUT / f"{preferred_name}_transition.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    preferred_control = [json.loads(line) for line in (OUT / f"{preferred_name}_anchor_control.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    filler = compact_row_lengths_and_exclusions(preferred_transition + preferred_control, preferred["transition_words"])

    # Reduce large text samples in summary while keeping examples inspectable through JSONL files.
    examples = {
        "preferred_transition_head": [{k: r.get(k) for k in ["text", "words", "source_label", "origin_source", "length_bin", "required_capability", "selected_route_bucket", "transition_marker_count_recomputed", "physical_anchor_count", "spatial_anchor_count", "quantity_anchor_count"]} for r in preferred_transition[:8]],
        "preferred_control_head": [{k: r.get(k) for k in ["text", "words", "source_label", "origin_source", "length_bin", "required_capability", "selected_route_bucket", "transition_marker_count_recomputed", "physical_anchor_count", "spatial_anchor_count", "quantity_anchor_count"]} for r in preferred_control[:8]],
    }

    write_csv(OUT / "match_regime_summary.csv", summary_rows)
    write_jsonl(OUT / "preferred_transition_structure_extra.jsonl", preferred_transition)
    write_jsonl(OUT / "preferred_anchor_control_extra.jsonl", preferred_control)
    write_csv(OUT / "preferred_filler_rows_manifest.csv", filler["filler_rows_manifest"])

    payload = {
        "status": "TRANSITION_STRUCTURE_MATCHED",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "boundary": "CPU-only re-match of compact-contained corpus-derived sentences; no training launched and no official evaluation item text read.",
        "inputs": {"transition": str(IN_T), "anchor_control": str(IN_C), "compact_10m": str(COMPACT_10M)},
        "input_summary": {"transition": qsummary(t_rows, "transition"), "anchor_control": qsummary(c_rows, "anchor_control")},
        "pair_summaries": pair_summaries,
        "preferred_regime": preferred_name,
        "preferred_summary": preferred,
        "whole_row_common_filler_for_one_extra_pass": filler,
        "examples": examples,
        "scientific_reading": {
            "can_isolate_source_and_style_at_probe_scale": bool(preferred["transition_words"] >= 10_000 and preferred["exact_word_parity"]),
            "reading": "The compact-contained subset supports only a modest relative-exposure mechanism contrast if the preferred matched pair is used: transition/control have exact word parity and match source/origin/length/capability plus several surface-style bins. The contrast remains narrative-heavy and should be read as a mechanism probe for transition structure, not as a direct SOTA data recipe. Coarse matching gives more words but is more style-confounded; fully strict matching is much smaller.",
            "not_yet_a_training_launch": "A GPU continuation still requires pending FW readouts or their absence to show that FW allocation does not already provide compact-level broad strength with relation movement. Existing full-batch trainers do not expose a true checkpoint-resume CLI; a future continuation should load the same intermediate weights for both arms and use identical continuation hyperparameters, with the reset optimizer treated as part of the controlled probe coordinate unless optimizer state is recovered elsewhere.",
        },
        "files": {
            "summary_json": str(OUT / "transition_structure_matcher.json"),
            "match_regime_summary_csv": str(OUT / "match_regime_summary.csv"),
            "preferred_transition_jsonl": str(OUT / "preferred_transition_structure_extra.jsonl"),
            "preferred_anchor_control_jsonl": str(OUT / "preferred_anchor_control_extra.jsonl"),
            "preferred_filler_rows_manifest_csv": str(OUT / "preferred_filler_rows_manifest.csv"),
            "note": str(NOTE),
        },
        "elapsed_sec": round(time.time() - t0, 2),
    }
    (OUT / "transition_structure_matcher.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines: list[str] = []
    lines.append("# research — compact-contained transition-structure contrast\n\n")
    lines.append("## Purpose\n\n")
    lines.append("research made a legal warm-start relative-exposure probe feasible but also showed that the research 30k files are not a ready pair. This analysis re-derived treatment/control subsets only from exact-contained compact-pool sentences and matched source/style before any GPU work.\n\n")
    lines.append("## Input subset\n\n")
    lines.append(f"- Transition contained subset: {len(t_rows)} sentences / {sum(r['words'] for r in t_rows)} actual whitespace words.\n")
    lines.append(f"- Anchor-control contained subset: {len(c_rows)} sentences / {sum(r['words'] for r in c_rows)} actual whitespace words.\n")
    lines.append("- Both are still dominated by the COMPACT_EXPERIENCE-aligned narrative pool; this remains a transfer risk rather than a reason to train blindly.\n\n")
    lines.append("## Matching results\n\n")
    lines.append("| regime | fields | transition sentences | control sentences | words each arm | source L1 | origin L1 | length L1 | quote L1 | proper L1 | pronoun L1 | digit L1 | comma L1 | transition markers T/C |\n")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in summary_rows:
        lines.append(
            f"| `{r['regime']}` | `{r['fields']}` | {r['transition_sentences']} | {r['control_sentences']} | {r['words_each_arm']} | "
            f"{r['source_label_l1']:.4f} | {r['origin_source_l1']:.4f} | {r['length_bin_l1']:.4f} | {r['quote_bin_l1']:.4f} | {r['proper_bin_l1']:.4f} | {r['pronoun_bin_l1']:.4f} | {r['digit_bin_l1']:.4f} | {r['comma_bin_l1']:.4f} | "
            f"{r['mean_transition_markers_transition']:.2f}/{r['mean_transition_markers_control']:.2f} |\n"
        )
    lines.append("\n")
    lines.append(f"Preferred regime: `{preferred_name}`. It gives exact word parity at **{preferred['transition_words']} words per arm** ({preferred['transition_sentences']} transition sentences, {preferred['control_sentences']} controls).\n\n")
    lines.append("## Scientific reading\n\n")
    if preferred["transition_words"] >= 10_000:
        lines.append("The compact-contained subset is sufficient only for a modest small mechanism probe: the preferred pair matches source/origin, length, required capability, quote bins, proper-name bins, and digit bins while preserving a large transition-marker separation. It can test whether extra exposure to transition structure moves EWoK conditional compatibility and GlobalPIQA hard-row ranks beyond an equally narrative-rich anchor control, but it is not strong enough to justify a full route by itself.\n\n")
    else:
        lines.append("The compact-contained subset is too small after source/style matching for a useful mechanism probe; the substrate should be rebuilt before any GPU continuation.\n\n")
    lines.append("This remains a probe, not a SOTA data recipe. The material is narrative-heavy and mostly from the COMPACT_EXPERIENCE-aligned pool, so success must mean coupled movement in EWoK four-cell structure and GlobalPIQA hard ranks while preserving broad compact-arm strength. An EWoK-only shift or a broad-column drop would repeat the earlier relation-data failure mode.\n\n")
    lines.append("## Continuation stream implication\n\n")
    lines.append(f"A common whole-row filler removal of {filler['filler_words']} words is {'available exactly' if filler['exact_filler_possible'] else 'not exact'} for one extra-subset insertion pass, excluding rows that contain preferred treatment/control sentences. This target is the per-arm extra subset, not treatment plus control combined; both arms can remove the same filler and add either the matched transition subset or matched anchor-control subset, keeping per-pass word exposure equal.\n")
    lines.append("Existing trainers do not expose a true resume flag in their command-line interface; a later continuation trainer must load the same intermediate model weights for both arms and record that optimizer state is either recovered or deliberately reset symmetrically.\n\n")
    lines.append("## Files\n\n")
    for k, p in payload["files"].items():
        lines.append(f"- {k}: `{p}`\n")
    NOTE.write_text("".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": payload["status"],
        "preferred_regime": preferred_name,
        "preferred_words_each_arm": preferred["transition_words"],
        "preferred_transition_sentences": preferred["transition_sentences"],
        "preferred_control_sentences": preferred["control_sentences"],
        "strict_style_words_each_arm": pair_summaries["strict_style"]["transition_words"],
        "medium_style_words_each_arm": pair_summaries["medium_style"]["transition_words"],
        "coarse_words_each_arm": pair_summaries["coarse_source_style"]["transition_words"],
        "filler_exact": filler["exact_filler_possible"],
        "filler_words": filler["filler_words"],
        "summary_json": payload["files"]["summary_json"],
        "note": str(NOTE),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
