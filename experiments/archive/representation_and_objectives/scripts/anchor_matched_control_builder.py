#!/usr/bin/env python3
"""research: build anchor-matched low-relation controls for the core transition slices.

The research length-matched controls remove relation markers but can be physically or
semantically unlike the transition treatment.  This CPU-only repair builds controls
that still contain concrete/object/action/spatial/quantity/procedure anchors while
suppressing explicit causal/change/contrast markers.  It is only a future probe
asset; no official evaluation item text is read and no training is launched.
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
OUT = A01_WS / "data/anchor_matched_controls"
NOTE = A01_WS / "notes/anchor_matched_controls.md"
BUDGETS = [50_000, 100_000, 200_000]

spec = importlib.util.spec_from_file_location("core", CORE_SCRIPT)
core = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(core)  # type: ignore[union-attr]

QUANTITY_RE = re.compile(
    r"\b(one|two|three|four|five|six|seven|eight|nine|ten|half|quarter|twice|once|many|few|several|some|all|both|each|every|number|amount|part|piece|hour|minute|day|week|month|year|age|old|young|small|large|short|long|high|low|deep|wide|narrow|temperature|weight|size)\b|\b\d+(?:\.\d+)?\b",
    re.I,
)
SOCIAL_ABSTRACT_RE = re.compile(r"\b(government|policy|election|market|company|business|rights|law|argument|opinion|belief|idea|story|movie|religion|campaign|internet|website|software|data|research|study|competition|winning|love|happiness|music|dance)\b", re.I)


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
    return len(core.CAUSAL_RE.findall(text)) + len(core.CHANGE_RE.findall(text)) + len(core.CONTRAST_RE.findall(text))


def anchor_counts(text: str) -> dict[str, int]:
    toks = core.words(text)
    return {
        "concrete": sum(1 for t in toks if t in core.CONCRETE_WORDS),
        "action": len(core.ACTION_RE.findall(text)),
        "spatial": len(core.SPATIAL_RE.findall(text)),
        "affordance": len(core.AFFORDANCE_RE.findall(text)),
        "quantity": len(QUANTITY_RE.findall(text)),
        "abstract_noise": sum(1 for t in toks if t in core.ABSTRACT_NOISE_WORDS) + len(SOCIAL_ABSTRACT_RE.findall(text)),
    }


def route_anchor_class(row: dict[str, Any]) -> str:
    b = str(row.get("selected_route_bucket") or "")
    if b == "spatial_transition":
        return "spatial_anchor"
    if b == "temporal_quantity_transition":
        return "quantity_anchor"
    if b == "affordance_procedure_transition":
        return "procedure_anchor"
    if b == "explicit_contrast_transition":
        return "contrast_content_anchor"
    return "physical_anchor"


def sentence_anchor_class(text: str) -> str | None:
    c = anchor_counts(text)
    # Prefer the most specific non-relational anchor.  A sentence can be physical and
    # spatial; row matching will use this single class plus source/length, so the
    # exact anchor counts remain saved for later inspection.
    if c["spatial"] >= 1 and (c["concrete"] >= 1 or c["action"] >= 1):
        return "spatial_anchor"
    if c["quantity"] >= 1 and (c["concrete"] >= 1 or c["action"] >= 1):
        return "quantity_anchor"
    if c["affordance"] >= 1 and (c["concrete"] >= 1 or c["action"] >= 1):
        return "procedure_anchor"
    if c["concrete"] >= 1 or c["action"] >= 1:
        return "physical_anchor"
    return None


def is_anchor_control(text: str, n: int) -> bool:
    if n < 8 or n > 90:
        return False
    if core.NOISE_RE.search(text) or core.TRANSCRIPT_NOISE_RE.search(text):
        return False
    if relation_markers(text) > 0:
        return False
    c = anchor_counts(text)
    if c["abstract_noise"] >= 2 and c["abstract_noise"] > c["concrete"] + c["action"]:
        return False
    if sentence_anchor_class(text) is None:
        return False
    if len(re.findall(r"\b[A-Z][a-z]{2,}\b", text)) >= max(8, n // 5):
        return False
    return True


def treatment_targets(treatment: list[dict[str, Any]]) -> dict[tuple[str, str, str], int]:
    out: Counter[tuple[str, str, str]] = Counter()
    for r in treatment:
        out[(str(r.get("source_label", "")), core.len_bin(int(r["words"])), route_anchor_class(r))] += int(r["words"])
    return dict(out)


def collect_anchor_pool(targets: dict[tuple[str, str, str], int], multiplier: float = 3.0, slack: int = 6000) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_source: Counter[str] = Counter()
    for (src, _lb, _ac), w in targets.items():
        by_source[src] += w
    need_by_source = {src: int(math.ceil(w * multiplier + slack)) for src, w in by_source.items()}
    pool: list[dict[str, Any]] = []
    seen: set[str] = set()
    scan: dict[str, Any] = {}
    for label, need in need_by_source.items():
        spec = core.RESERVOIRS.get(label) or core.RESERVOIRS.get("fw_frozen_sources_38167")
        if spec is None:
            continue
        path = Path(spec["path"])
        got = 0; accepted = 0; sent_scanned = 0; word_scanned = 0
        got_by_class: Counter[str] = Counter()
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
                        ac = sentence_anchor_class(sent)
                        if ac is None:
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
                            "anchor_class": ac,
                            "anchor_counts": c,
                            "relation_marker_count": relation_markers(sent),
                            "origin_source": obj.get("source") or obj.get("pool") or obj.get("source_kind") or obj.get("origin_source"),
                            "doc_id": obj.get("doc_id"),
                            "norm_hash": obj.get("norm_hash"),
                        }
                        pool.append(rec); got += n; accepted += 1; got_by_class[ac] += n
                        if got >= need:
                            break
                    if got >= need:
                        break
        scan[label] = {
            "target_words": by_source[label],
            "collection_need_words": need,
            "collected_words": got,
            "accepted_sentences": accepted,
            "accepted_by_anchor_class_words": dict(got_by_class),
            "sentences_scanned": sent_scanned,
            "words_scanned": word_scanned,
            "path": str(path),
        }
    return pool, scan


def build_indices(pool: list[dict[str, Any]]):
    by_sla: defaultdict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    by_sl: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_sa: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_la: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_s: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    by_l: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    by_a: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in pool:
        src = str(r["source_label"]); lb = str(r["length_bin"]); ac = str(r["anchor_class"])
        by_sla[(src, lb, ac)].append(r); by_sl[(src, lb)].append(r); by_sa[(src, ac)].append(r)
        by_la[(lb, ac)].append(r); by_s[src].append(r); by_l[lb].append(r); by_a[ac].append(r)
    for coll in [by_sla, by_sl, by_sa, by_la, by_s, by_l, by_a]:
        for rows in coll.values():
            rows.sort(key=lambda r: (int(r["words"]), -int(r["anchor_counts"].get("concrete",0)), r["text"]))
    pool.sort(key=lambda r: (int(r["words"]), str(r.get("anchor_class", "")), r["text"]))
    return by_sla, by_sl, by_sa, by_la, by_s, by_l, by_a


def pick_controls(treatment: list[dict[str, Any]], pool: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_sla, by_sl, by_sa, by_la, by_s, by_l, by_a = build_indices(pool)
    used: set[int] = set()
    modes = Counter()
    chosen: list[dict[str, Any]] = []

    def choose(rows: list[dict[str, Any]], target_len: int, target_ac: str) -> dict[str, Any] | None:
        best = None; best_key = None
        for r in rows:
            if id(r) in used:
                continue
            key = (
                abs(int(r["words"]) - target_len),
                0 if str(r.get("anchor_class")) == target_ac else 1,
                -int(r.get("anchor_counts", {}).get("concrete", 0)),
                r.get("text", ""),
            )
            if best_key is None or key < best_key:
                best = r; best_key = key
        if best is None:
            return None
        used.add(id(best))
        return best

    for tr in sorted(treatment, key=lambda r: (-int(r["words"]), route_anchor_class(r), str(r.get("text_sha256", "")))):
        src = str(tr.get("source_label", "")); lb = core.len_bin(int(tr["words"])); ac = route_anchor_class(tr); tw = int(tr["words"])
        ladders = [
            ("same_source_lenbin_anchor", by_sla[(src, lb, ac)]),
            ("same_source_lenbin", by_sl[(src, lb)]),
            ("same_source_anchor", by_sa[(src, ac)]),
            ("same_lenbin_anchor", by_la[(lb, ac)]),
            ("same_source", by_s[src]),
            ("same_lenbin", by_l[lb]),
            ("same_anchor", by_a[ac]),
            ("any_anchor_control", pool),
        ]
        selected = None; mode = "unmatched"
        for mode_i, rows in ladders:
            selected = choose(rows, tw, ac)
            if selected is not None:
                mode = mode_i
                break
        if selected is not None:
            rr = dict(selected)
            rr["matched_treatment_words"] = tw
            rr["matched_treatment_source_label"] = src
            rr["matched_treatment_length_bin"] = lb
            rr["matched_treatment_anchor_class"] = ac
            rr["match_mode"] = mode
            chosen.append(rr); modes[mode] += 1
        else:
            modes["unmatched"] += 1
    return chosen, {"match_modes": dict(modes)}


def word_counter(rows: list[dict[str, Any]], keyfunc) -> Counter[str]:
    c: Counter[str] = Counter()
    for r in rows:
        c[str(keyfunc(r))] += int(r["words"])
    return c


def l1(a: Counter[str], b: Counter[str]) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    ta = sum(a.values()) or 1
    tb = sum(b.values()) or 1
    return sum(abs(a.get(k,0)/ta - b.get(k,0)/tb) for k in keys)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = sum(int(r["words"]) for r in rows)
    ac_words = word_counter(rows, lambda r: r.get("anchor_class") or route_anchor_class(r))
    src_words = word_counter(rows, lambda r: r.get("source_label", ""))
    lb_words = word_counter(rows, lambda r: r.get("length_bin") or core.len_bin(int(r["words"])))
    return {
        "sentences": len(rows),
        "words": total,
        "mean_words": total / len(rows) if rows else 0.0,
        "median_words": statistics.median([int(r["words"]) for r in rows]) if rows else 0.0,
        "anchor_class_words": dict(ac_words),
        "source_words": dict(src_words),
        "length_bin_words": dict(lb_words),
    }


def match_metrics(treatment: list[dict[str, Any]], controls: list[dict[str, Any]]) -> dict[str, Any]:
    t_src = word_counter(treatment, lambda r: r.get("source_label", "")); c_src = word_counter(controls, lambda r: r.get("source_label", ""))
    t_lb = word_counter(treatment, lambda r: core.len_bin(int(r["words"]))); c_lb = word_counter(controls, lambda r: r.get("length_bin", ""))
    t_ac = word_counter(treatment, route_anchor_class); c_ac = word_counter(controls, lambda r: r.get("anchor_class", ""))
    diffs = [int(c.get("words", 0)) - int(c.get("matched_treatment_words", 0)) for c in controls if "matched_treatment_words" in c]
    return {
        "source_l1": l1(t_src, c_src),
        "length_bin_l1": l1(t_lb, c_lb),
        "anchor_class_l1": l1(t_ac, c_ac),
        "word_diff": sum(int(r["words"]) for r in controls) - sum(int(r["words"]) for r in treatment),
        "row_abs_length_diff_mean": statistics.mean([abs(d) for d in diffs]) if diffs else None,
        "row_abs_length_diff_median": statistics.median([abs(d) for d in diffs]) if diffs else None,
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
    treatments: dict[int, list[dict[str, Any]]] = {}
    merged_targets: Counter[tuple[str, str, str]] = Counter()
    for budget in BUDGETS:
        p = IN_DIR / f"core_transition_treatment_{budget//1000}k.jsonl"
        rows = load_jsonl(p)
        treatments[budget] = rows
        for k, v in treatment_targets(rows).items():
            merged_targets[k] = max(merged_targets[k], v)
    pool, scan = collect_anchor_pool(dict(merged_targets))
    write_jsonl(OUT / "anchor_control_candidate_pool.jsonl", pool)

    summary_rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    detailed: dict[str, Any] = {}
    for budget, treatment in treatments.items():
        controls, info = pick_controls(treatment, pool)
        outp = OUT / f"core_transition_anchor_control_{budget//1000}k.jsonl"
        write_jsonl(outp, controls)
        sm_t = summarize(treatment); sm_c = summarize(controls); mm = match_metrics(treatment, controls)
        detailed[str(budget)] = {"treatment_summary": sm_t, "anchor_control_summary": sm_c, "match_metrics": mm, "match_info": info, "control_path": str(outp)}
        for arm, sm in [("treatment", sm_t), ("anchor_control", sm_c)]:
            summary_rows.append({
                "budget": f"{budget//1000}k",
                "arm": arm,
                "sentences": sm["sentences"],
                "words": sm["words"],
                "mean_words": round(sm["mean_words"], 3),
                "median_words": sm["median_words"],
                "anchor_class_words": json.dumps(sm["anchor_class_words"], sort_keys=True),
                "source_words": json.dumps(sm["source_words"], sort_keys=True),
                "length_bin_words": json.dumps(sm["length_bin_words"], sort_keys=True),
            })
        match_rows.append({
            "budget": f"{budget//1000}k",
            "word_diff_anchor_control_minus_treatment": mm["word_diff"],
            "source_l1": round(mm["source_l1"], 6),
            "length_bin_l1": round(mm["length_bin_l1"], 6),
            "anchor_class_l1": round(mm["anchor_class_l1"], 6),
            "row_abs_length_diff_mean": round(float(mm["row_abs_length_diff_mean"] or 0.0), 3),
            "row_abs_length_diff_median": round(float(mm["row_abs_length_diff_median"] or 0.0), 3),
            "match_modes": json.dumps(info["match_modes"], sort_keys=True),
        })
    write_csv(OUT / "anchor_control_summary.csv", summary_rows, ["budget","arm","sentences","words","mean_words","median_words","anchor_class_words","source_words","length_bin_words"])
    write_csv(OUT / "anchor_control_match_summary.csv", match_rows, ["budget","word_diff_anchor_control_minus_treatment","source_l1","length_bin_l1","anchor_class_l1","row_abs_length_diff_mean","row_abs_length_diff_median","match_modes"])
    summary = {
        "status": "ANCHOR_MATCHED_CONTROLS_BUILT",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "controls with concrete/action/spatial/quantity/procedure anchors but without explicit causal/change/contrast markers",
        "treatment_dir": str(IN_DIR),
        "anchor_pool_count": len(pool),
        "anchor_pool_words": sum(int(r["words"]) for r in pool),
        "scan": scan,
        "budgets": detailed,
        "files": {
            "anchor_pool": str(OUT / "anchor_control_candidate_pool.jsonl"),
            "summary_csv": str(OUT / "anchor_control_summary.csv"),
            "match_summary_csv": str(OUT / "anchor_control_match_summary.csv"),
            "note": str(NOTE),
        },
        "elapsed_sec": round(time.time() - t0, 3),
    }
    (OUT / "anchor_matched_controls.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    note = [
        "# research — anchor-matched low-relation controls",
        "",
        "## Purpose",
        "",
        "The first research controls matched source and length but often differed in concrete content. This CPU-only repair collects controls that retain object/action/spatial/quantity/procedure anchors while suppressing explicit causal/change/contrast markers. The intended future comparison is transition structure versus anchored non-transition content, not merely concrete vocabulary versus unrelated text.",
        "",
        "## Main counts",
        "",
        f"- Anchor-control candidate pool: {summary['anchor_pool_count']} sentences / {summary['anchor_pool_words']} words.",
        "- Matching measurements are in `anchor_control_match_summary.csv`.",
        "",
        "## Scientific reading",
        "",
        "These controls make a later small probe cleaner, but they still do not justify any 100M run. The running FW compact/breadth endpoints must be read first. If they do not move the shared relation weakness, a small shared-coordinate probe can compare core transition slices against these anchor-matched controls before any larger route is considered.",
        "",
        "## Files",
        "",
        f"- summary JSON: `{OUT / 'anchor_matched_controls.json'}`",
        f"- anchor pool: `{OUT / 'anchor_control_candidate_pool.jsonl'}`",
        f"- controls: `{OUT / 'core_transition_anchor_control_50k.jsonl'}`, `{OUT / 'core_transition_anchor_control_100k.jsonl'}`, `{OUT / 'core_transition_anchor_control_200k.jsonl'}`",
        f"- summary CSV: `{OUT / 'anchor_control_summary.csv'}`",
        f"- match summary CSV: `{OUT / 'anchor_control_match_summary.csv'}`",
    ]
    NOTE.write_text("\n".join(note) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "anchor_pool_count": summary["anchor_pool_count"],
        "anchor_pool_words": summary["anchor_pool_words"],
        "summary_csv": str(OUT / "anchor_control_summary.csv"),
        "match_summary_csv": str(OUT / "anchor_control_match_summary.csv"),
        "json": str(OUT / "anchor_matched_controls.json"),
        "note": str(NOTE),
        "elapsed_sec": summary["elapsed_sec"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
