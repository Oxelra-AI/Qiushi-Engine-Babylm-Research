#!/usr/bin/env python3
"""research v2 anchor-capability matched controls.

This repairs the first anchor-control builder, whose single priority label caused a
large physical->spatial mismatch.  Here a neutral sentence can satisfy multiple
anchor capabilities.  Each treatment row is matched to a no-causal/no-change/no-
contrast control with the same source and length bin when possible and with the
required capability implied by the treatment bucket.

No official evaluation item text is read and no training is launched.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

USER_ROOT = Path.cwd()
A01_WS = USER_ROOT / "experiments/archive/representation_and_objectives"
CORE_SCRIPT = A01_WS / "scripts/core_transition_filter_and_controls.py"
IN_DIR = A01_WS / "data/core_transition_filter"
OUT = A01_WS / "data/anchor_matched_controls_v2"
NOTE = A01_WS / "notes/anchor_matched_controls_v2.md"
BUDGETS = [50_000, 100_000, 200_000]

spec = importlib.util.spec_from_file_location("core", CORE_SCRIPT)
core = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(core)  # type: ignore[union-attr]

QUANTITY_RE = re.compile(
    r"\b(one|two|three|four|five|six|seven|eight|nine|ten|half|quarter|twice|once|many|few|several|some|all|both|each|every|number|amount|part|piece|hour|minute|day|week|month|year|age|old|young|small|large|short|long|high|low|deep|wide|narrow|temperature|weight|size|inch|foot|feet|meter|metre|mile|pound|gram|degree)\b|\b\d+(?:\.\d+)?\b",
    re.I,
)
ABSTRACT_EXTRA_RE = re.compile(
    r"\b(government|policy|policies|election|parliament|minister|kingdom|princess|lord|market|company|business|rights|law|court|argument|opinion|belief|idea|story|movie|religion|campaign|internet|website|software|data|research|study|studies|competition|winning|love|happiness|music|dance|agency|report|environmental|political|communist|army|party|finance|stock|school|student|teacher|university)\b",
    re.I,
)

# Explicit relation structure to suppress in controls.  Include some lexical forms
# that the first anchor-control run allowed and that made samples abstractly
# relational rather than merely anchored.
RELATION_CONTROL_RE = re.compile(
    r"\b(if|when|whenever|unless|because|since|therefore|thus|so that|in order to|as a result|result|results|resulted|cause|causes|caused|leads?|led to|prevent|prevents|prevented|allows?|allowed|"
    r"become|became|becomes|turns? into|changed?|changes?|increase[sd]?|decrease[sd]?|more|less|fewer|rather than|instead|whereas|although|but|while|before|after|first|then|next|finally|from\b.{0,45}\bto|"
    r"open(?:ed|s)?|close[sd]?|sealed?|unsealed|broken?|breaks?|melt(?:ed|s)?|freez(?:e|es|ing|en)|dissolv(?:e|es|ed)|fall(?:s|en|ing)?|fell|drop(?:s|ped)?|rise[sn]?|rose|lower(?:ed|s)?|lift(?:ed|s)?|move[sd]?|shifts?|spread(?:s|ing)?|mix(?:ed|es)?|separate[sd]?|strengthen(?:ed|s)?|weaken(?:ed|s)?|need(?:ed|s)? to|must|should|how to|safe|careful|make sure)\b",
    re.I,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def relation_markers(text: str) -> int:
    return len(RELATION_CONTROL_RE.findall(text))


def anchor_counts(text: str) -> dict[str, int]:
    toks = core.words(text)
    return {
        "concrete": sum(1 for t in toks if t in core.CONCRETE_WORDS),
        "action": len(core.ACTION_RE.findall(text)),
        "spatial": len(core.SPATIAL_RE.findall(text)),
        "affordance": len(core.AFFORDANCE_RE.findall(text)),
        "quantity": len(QUANTITY_RE.findall(text)),
        "abstract_noise": sum(1 for t in toks if t in core.ABSTRACT_NOISE_WORDS) + len(ABSTRACT_EXTRA_RE.findall(text)),
    }


def capabilities(text: str) -> list[str]:
    c = anchor_counts(text)
    caps = []
    concrete_or_action = c["concrete"] >= 1 or c["action"] >= 1
    # Treat spatial/quantity/procedure as additional capabilities; a sentence with
    # concrete+spatial also remains a valid physical anchor.
    if c["concrete"] >= 2 or (c["concrete"] >= 1 and c["action"] >= 1):
        caps.append("physical_anchor")
    if c["spatial"] >= 1 and concrete_or_action:
        caps.append("spatial_anchor")
    if c["quantity"] >= 1 and concrete_or_action:
        caps.append("quantity_anchor")
    if c["affordance"] >= 1 and concrete_or_action:
        caps.append("procedure_anchor")
    return caps


def required_capability(row: dict[str, Any]) -> str:
    b = str(row.get("selected_route_bucket") or "")
    if b == "spatial_transition":
        return "spatial_anchor"
    if b == "temporal_quantity_transition":
        return "quantity_anchor"
    if b == "affordance_procedure_transition":
        return "procedure_anchor"
    # explicit contrast treatment is nearly absent in the current core slices; when
    # present, concrete/action anchored controls are the safest counterpart.
    return "physical_anchor"


def is_anchor_control(text: str, n: int) -> bool:
    if n < 8 or n > 90:
        return False
    if core.NOISE_RE.search(text) or core.TRANSCRIPT_NOISE_RE.search(text):
        return False
    if relation_markers(text) > 0:
        return False
    c = anchor_counts(text)
    if c["abstract_noise"] > 0:
        return False
    if not capabilities(text):
        return False
    proper = len(re.findall(r"\b[A-Z][a-z]{2,}\b", text))
    if proper >= max(5, n // 7):
        return False
    return True


def collect_pool(multiplier: float = 8.0) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # Estimate source needs from the largest treatment slice.  Over-collect all
    # capabilities because same-source physical controls are the scarcest part.
    largest = load_jsonl(IN_DIR / "core_transition_treatment_200k.jsonl")
    need_by_source: Counter[str] = Counter()
    for r in largest:
        need_by_source[str(r.get("source_label", ""))] += int(r["words"])
    need_by_source = Counter({src: int(math.ceil(w * multiplier + 10000)) for src, w in need_by_source.items()})

    pool: list[dict[str, Any]] = []
    seen: set[str] = set()
    scan: dict[str, Any] = {}
    for label, need in need_by_source.items():
        spec = core.RESERVOIRS.get(label) or core.RESERVOIRS.get("fw_frozen_sources_38167")
        if spec is None:
            continue
        path = Path(spec["path"])
        got = 0; accepted = 0; sent_scanned = 0; word_scanned = 0
        cap_words: Counter[str] = Counter()
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for row_index, line in enumerate(f):
                if got >= need:
                    break
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for field in spec["fields"]:
                    raw = str(obj.get(field, "") or "")
                    if not raw:
                        continue
                    for sent_i, sent in enumerate(core.split_sentences(raw)):
                        sent = core.norm(sent)
                        n = core.wc(sent)
                        sent_scanned += 1; word_scanned += n
                        if not is_anchor_control(sent, n):
                            continue
                        key = sent.lower()
                        if key in seen:
                            continue
                        caps = capabilities(sent)
                        if not caps:
                            continue
                        seen.add(key)
                        c = anchor_counts(sent)
                        rec = {
                            "source_label": label,
                            "source_kind": spec["kind"],
                            "input_path": str(path),
                            "row_index": row_index,
                            "sentence_index": sent_i,
                            "field": field,
                            "text": sent,
                            "words": n,
                            "length_bin": core.len_bin(n),
                            "anchor_capabilities": caps,
                            "anchor_counts": c,
                            "relation_marker_count": relation_markers(sent),
                            "origin_source": obj.get("source") or obj.get("pool") or obj.get("source_kind") or obj.get("origin_source"),
                            "doc_id": obj.get("doc_id"),
                            "norm_hash": obj.get("norm_hash"),
                        }
                        pool.append(rec); got += n; accepted += 1
                        for cap in caps:
                            cap_words[cap] += n
                        if got >= need:
                            break
                    if got >= need:
                        break
        scan[label] = {
            "target_from_200k_slice_words": int((need - 10000) / multiplier) if multiplier else None,
            "collection_need_words": need,
            "collected_words": got,
            "accepted_sentences": accepted,
            "accepted_by_capability_words": dict(cap_words),
            "sentences_scanned": sent_scanned,
            "words_scanned": word_scanned,
            "path": str(path),
        }
    return pool, scan


def build_indices(pool: list[dict[str, Any]]):
    idx: defaultdict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    idx_sl: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    idx_sc: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    idx_lc: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    idx_s: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    idx_c: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in pool:
        src = str(r["source_label"]); lb = str(r["length_bin"])
        for cap in r.get("anchor_capabilities", []):
            idx[(src, lb, cap)].append(r); idx_sc[(src, cap)].append(r); idx_lc[(lb, cap)].append(r); idx_c[cap].append(r)
        idx_sl[(src, lb)].append(r); idx_s[src].append(r)
    collections = list(idx.values()) + list(idx_sl.values()) + list(idx_sc.values()) + list(idx_lc.values()) + list(idx_s.values()) + list(idx_c.values())
    for rows in collections:
        rows.sort(key=lambda r: (int(r["words"]), -int(r["anchor_counts"].get("concrete", 0)), -len(r.get("anchor_capabilities", [])), r["text"]))
    pool.sort(key=lambda r: (int(r["words"]), -int(r["anchor_counts"].get("concrete", 0)), r["text"]))
    return idx, idx_sl, idx_sc, idx_lc, idx_s, idx_c


def pick_controls(treatment: list[dict[str, Any]], pool: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    idx, idx_sl, idx_sc, idx_lc, idx_s, idx_c = build_indices(pool)
    used: set[int] = set()
    modes = Counter()
    chosen: list[dict[str, Any]] = []

    def choose(rows: list[dict[str, Any]], target_len: int, req: str) -> dict[str, Any] | None:
        best = None; best_key = None
        for r in rows:
            if id(r) in used:
                continue
            caps = set(r.get("anchor_capabilities", []))
            key = (
                0 if req in caps else 1,
                abs(int(r["words"]) - target_len),
                -int(r.get("anchor_counts", {}).get("concrete", 0)),
                -len(caps),
                r.get("text", ""),
            )
            if best_key is None or key < best_key:
                best = r; best_key = key
        if best is None:
            return None
        used.add(id(best)); return best

    for tr in sorted(treatment, key=lambda r: (-int(r["words"]), required_capability(r), str(r.get("text_sha256", "")))):
        src = str(tr.get("source_label", "")); lb = core.len_bin(int(tr["words"])); req = required_capability(tr); tw = int(tr["words"])
        ladders = [
            ("same_source_lenbin_requiredcap", idx[(src, lb, req)]),
            ("same_source_requiredcap", idx_sc[(src, req)]),
            ("same_lenbin_requiredcap", idx_lc[(lb, req)]),
            ("any_source_requiredcap", idx_c[req]),
            ("same_source_lenbin_anycap", idx_sl[(src, lb)]),
            ("same_source_anycap", idx_s[src]),
            ("any_anchor_control", pool),
        ]
        selected = None; mode = "unmatched"
        for mode_i, rows in ladders:
            selected = choose(rows, tw, req)
            if selected is not None:
                mode = mode_i
                break
        if selected is not None:
            rr = dict(selected)
            rr["required_capability"] = req
            rr["matched_treatment_words"] = tw
            rr["matched_treatment_source_label"] = src
            rr["matched_treatment_length_bin"] = lb
            rr["matched_treatment_bucket"] = str(tr.get("selected_route_bucket"))
            rr["match_mode"] = mode
            chosen.append(rr); modes[mode] += 1
        else:
            modes["unmatched"] += 1
    return chosen, {"match_modes": dict(modes)}


def wcounter(rows: list[dict[str, Any]], keyfunc) -> Counter[str]:
    c: Counter[str] = Counter()
    for r in rows:
        c[str(keyfunc(r))] += int(r["words"])
    return c


def capability_word_counter_controls(rows: list[dict[str, Any]], required_only: bool = False) -> Counter[str]:
    c: Counter[str] = Counter()
    for r in rows:
        w = int(r["words"])
        if required_only and r.get("required_capability"):
            c[str(r["required_capability"])] += w
        else:
            for cap in r.get("anchor_capabilities", []):
                c[str(cap)] += w
    return c


def capability_word_counter_treatment(rows: list[dict[str, Any]]) -> Counter[str]:
    c: Counter[str] = Counter()
    for r in rows:
        c[required_capability(r)] += int(r["words"])
    return c


def l1(a: Counter[str], b: Counter[str]) -> float:
    keys = set(a) | set(b)
    ta = sum(a.values()) or 1; tb = sum(b.values()) or 1
    return sum(abs(a.get(k,0)/ta - b.get(k,0)/tb) for k in keys)


def summarize_treatment(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = sum(int(r["words"]) for r in rows)
    return {
        "sentences": len(rows),
        "words": total,
        "mean_words": total/len(rows) if rows else 0.0,
        "median_words": statistics.median([int(r["words"]) for r in rows]) if rows else 0.0,
        "required_capability_words": dict(capability_word_counter_treatment(rows)),
        "source_words": dict(wcounter(rows, lambda r: r.get("source_label", ""))),
        "length_bin_words": dict(wcounter(rows, lambda r: core.len_bin(int(r["words"])))),
    }


def summarize_controls(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = sum(int(r["words"]) for r in rows)
    return {
        "sentences": len(rows),
        "words": total,
        "mean_words": total/len(rows) if rows else 0.0,
        "median_words": statistics.median([int(r["words"]) for r in rows]) if rows else 0.0,
        "required_capability_words": dict(capability_word_counter_controls(rows, required_only=True)),
        "all_capability_words": dict(capability_word_counter_controls(rows, required_only=False)),
        "source_words": dict(wcounter(rows, lambda r: r.get("source_label", ""))),
        "length_bin_words": dict(wcounter(rows, lambda r: r.get("length_bin", ""))),
    }


def metrics(treatment: list[dict[str, Any]], controls: list[dict[str, Any]]) -> dict[str, Any]:
    t_src = wcounter(treatment, lambda r: r.get("source_label", "")); c_src = wcounter(controls, lambda r: r.get("source_label", ""))
    t_lb = wcounter(treatment, lambda r: core.len_bin(int(r["words"]))); c_lb = wcounter(controls, lambda r: r.get("length_bin", ""))
    t_req = capability_word_counter_treatment(treatment); c_req = capability_word_counter_controls(controls, required_only=True)
    diffs = [int(r["words"]) - int(r.get("matched_treatment_words", 0)) for r in controls]
    req_hit = [1 if r.get("required_capability") in set(r.get("anchor_capabilities", [])) else 0 for r in controls]
    return {
        "word_diff_control_minus_treatment": sum(int(r["words"]) for r in controls) - sum(int(r["words"]) for r in treatment),
        "source_l1": l1(t_src, c_src),
        "length_bin_l1": l1(t_lb, c_lb),
        "required_capability_l1": l1(t_req, c_req),
        "row_abs_length_diff_mean": statistics.mean([abs(x) for x in diffs]) if diffs else None,
        "row_abs_length_diff_median": statistics.median([abs(x) for x in diffs]) if diffs else None,
        "required_capability_hit_fraction": sum(req_hit)/len(req_hit) if req_hit else 0.0,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True); NOTE.parent.mkdir(parents=True, exist_ok=True)
    pool, scan = collect_pool()
    write_jsonl(OUT / "anchor_control_candidate_pool_v2.jsonl", pool)
    summary_rows = []
    match_rows = []
    detailed = {}
    for budget in BUDGETS:
        treatment = load_jsonl(IN_DIR / f"core_transition_treatment_{budget//1000}k.jsonl")
        controls, info = pick_controls(treatment, pool)
        ctrl_path = OUT / f"core_transition_anchor_control_v2_{budget//1000}k.jsonl"
        write_jsonl(ctrl_path, controls)
        st = summarize_treatment(treatment); sc = summarize_controls(controls); mm = metrics(treatment, controls)
        detailed[str(budget)] = {"treatment_summary": st, "anchor_control_summary": sc, "match_metrics": mm, "match_info": info, "control_path": str(ctrl_path)}
        for arm, sm in [("treatment", st), ("anchor_control_v2", sc)]:
            summary_rows.append({
                "budget": f"{budget//1000}k",
                "arm": arm,
                "sentences": sm["sentences"],
                "words": sm["words"],
                "mean_words": round(sm["mean_words"], 3),
                "median_words": sm["median_words"],
                "required_capability_words": json.dumps(sm["required_capability_words"], sort_keys=True),
                "source_words": json.dumps(sm["source_words"], sort_keys=True),
                "length_bin_words": json.dumps(sm["length_bin_words"], sort_keys=True),
                "all_control_capability_words": json.dumps(sm.get("all_capability_words", {}), sort_keys=True),
            })
        match_rows.append({
            "budget": f"{budget//1000}k",
            "word_diff_control_minus_treatment": mm["word_diff_control_minus_treatment"],
            "source_l1": round(mm["source_l1"], 6),
            "length_bin_l1": round(mm["length_bin_l1"], 6),
            "required_capability_l1": round(mm["required_capability_l1"], 6),
            "row_abs_length_diff_mean": round(float(mm["row_abs_length_diff_mean"] or 0.0), 3),
            "row_abs_length_diff_median": round(float(mm["row_abs_length_diff_median"] or 0.0), 3),
            "required_capability_hit_fraction": round(mm["required_capability_hit_fraction"], 6),
            "match_modes": json.dumps(info["match_modes"], sort_keys=True),
        })
    write_csv(OUT / "anchor_control_v2_summary.csv", summary_rows, ["budget","arm","sentences","words","mean_words","median_words","required_capability_words","source_words","length_bin_words","all_control_capability_words"])
    write_csv(OUT / "anchor_control_v2_match_summary.csv", match_rows, ["budget","word_diff_control_minus_treatment","source_l1","length_bin_l1","required_capability_l1","row_abs_length_diff_mean","row_abs_length_diff_median","required_capability_hit_fraction","match_modes"])
    summary = {
        "status": "ANCHOR_MATCHED_CONTROLS_V2_BUILT",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repair_over_v1": "capability-based matching; a control may be both physical and spatial, and each treatment row requires its route-implied capability",
        "anchor_pool_count": len(pool),
        "anchor_pool_words": sum(int(r["words"]) for r in pool),
        "scan": scan,
        "budgets": detailed,
        "files": {
            "anchor_pool": str(OUT / "anchor_control_candidate_pool_v2.jsonl"),
            "summary_csv": str(OUT / "anchor_control_v2_summary.csv"),
            "match_summary_csv": str(OUT / "anchor_control_v2_match_summary.csv"),
            "note": str(NOTE),
        },
        "elapsed_sec": round(time.time() - t0, 3),
    }
    (OUT / "anchor_matched_controls_v2.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    note = [
        "# research — anchor-matched controls v2",
        "",
        "## Purpose",
        "",
        "The first anchor-control builder kept length/source matching but its priority label shifted much of the control word mass from physical to spatial anchors. This v2 repair uses capability matching: a no-relation control can satisfy multiple anchors, and every treatment row is matched to a control with the required physical/spatial/quantity/procedure capability whenever possible.",
        "",
        "## Main counts",
        "",
        f"- Anchor-control candidate pool: {summary['anchor_pool_count']} sentences / {summary['anchor_pool_words']} words.",
        "- Match measurements are in `anchor_control_v2_match_summary.csv`.",
        "",
        "## Scientific reading",
        "",
        "This is now the preferred neutral comparator for any later small transition-substrate probe. It still does not justify a 100M route: first read the FW compact/breadth endpoints through official scores, EWoK interaction failures, and GlobalPIQA hard-row margins.",
        "",
        "## Files",
        "",
        f"- summary JSON: `{OUT / 'anchor_matched_controls_v2.json'}`",
        f"- candidate pool: `{OUT / 'anchor_control_candidate_pool_v2.jsonl'}`",
        f"- controls: `{OUT / 'core_transition_anchor_control_v2_50k.jsonl'}`, `{OUT / 'core_transition_anchor_control_v2_100k.jsonl'}`, `{OUT / 'core_transition_anchor_control_v2_200k.jsonl'}`",
        f"- summary CSV: `{OUT / 'anchor_control_v2_summary.csv'}`",
        f"- match summary CSV: `{OUT / 'anchor_control_v2_match_summary.csv'}`",
    ]
    NOTE.write_text("\n".join(note) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "anchor_pool_count": summary["anchor_pool_count"],
        "anchor_pool_words": summary["anchor_pool_words"],
        "summary_csv": str(OUT / "anchor_control_v2_summary.csv"),
        "match_summary_csv": str(OUT / "anchor_control_v2_match_summary.csv"),
        "json": str(OUT / "anchor_matched_controls_v2.json"),
        "note": str(NOTE),
        "elapsed_sec": summary["elapsed_sec"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
