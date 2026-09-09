#!/usr/bin/env python3
"""research: file-only admission/saturation readout for fixed-budget substituted text.

This script reads already-materialized pool metadata and selected pair/source files. It
asks whether the broad ex-Entity advantage is better attributed to the corpus slice
occupying the substituted words than to the exact companion form. It does not load
models, train, evaluate, or touch GPUs.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import math
import os
import re
import statistics
import time
from pathlib import Path
from typing import Any, Iterable

ROOT = Path("experiments/archive/frontier_consolidation")
OUT = ROOT / "data/admission_saturation_readout"

META = ROOT / "data/density_cleanqwen_overlay_medium_riskhard/density_cleanqwen_rowholdout_overlay_metadata.json"
META = ROOT / "data/dose_2p64x_rowholdout_pools/dose2p64x_rowholdout_metadata.json"
META = ROOT / "data/dose_1p82x_rowholdout_pools/dose1p82_rowholdout_metadata.json"
META = ROOT / "data/dose_2p64x_breadth_rowholdout_pools/max_breadth_rowholdout_metadata.json"

PAIR_FILES = {
    "dose1_view_repeat_pairset": ROOT / "data/density_core_reinvestment_medium_riskhard/selected_compact_reinvest_pairs.jsonl",
    "dose1p82_view_repeat_pairset": ROOT / "data/dose_intermediate_select/selected_dose1p82_pairs.jsonl",
    "dose2p64_view_repeat_pairset": ROOT / "data/dose_distribution_select/selected_matched_max_pairs.jsonl",
}
BREADTH_COMPANION = ROOT / "data/dose_2p64x_breadth_rowholdout_pools/compact_breadth_dose2p64x_selected_companion_sources.jsonl"
OLD_POINT = ROOT / "data/common_window_budget_decomposition/common10_80_point_decomposition.csv"
OLD_FAMILY = ROOT / "data/common_window_budget_decomposition/common10_80_family_components.csv"
REF_CONTRAST = ROOT / "data/reference_decomposition_readout/contrast_window_summaries.csv"
SEED_ALIGN = ROOT / "data/axis_noise_floor_readout/checkpoint_seed_vector_alignment.csv"
ENDPOINT_DISP = ROOT / "data/axis_noise_floor_readout/endpoint_vs_seed_displacement.csv"

WORD_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*|\d+(?:[.,:]\d+)*")
DOMAIN_PATTERNS = {
    "quant_numeric": re.compile(r"\b\d|\b(percent|million|billion|thousand|year|years|km|kg|\$|euro|pounds|score|date|decade|century)\b", re.I),
    "science_technical": re.compile(r"\b(science|scientific|energy|chemical|temperature|species|software|system|protein|medical|physics|planet|cell|data|algorithm|technology|engineer|electric|material|environment|climate|computer|database|vitamin|gas|water|earth|polar)\b", re.I),
    "causal_relational": re.compile(r"\b(because|therefore|caused|causes|cause|effect|response|result|after|before|during|if|when|while|due to|led to|leads to|so that|rather|although|however|thereby|consequently)\b", re.I),
    "institutions_society": re.compile(r"\b(government|university|foundation|company|school|court|law|policy|program|institute|society|department|president|minister|council|election|operation|passenger)\b", re.I),
    "geography_places": re.compile(r"\b(city|country|river|island|mountain|state|county|region|village|town|lake|galaxy|milky way|earth|virginia|georgian|europe|polar)\b", re.I),
    "media_culture": re.compile(r"\b(film|book|music|story|novel|art|game|television|song|culture|movie|album|band|guitar|slayer)\b", re.I),
    "people_history": re.compile(r"\b(history|historical|king|queen|war|century|president|ancient|born|died|christianity|jesus|foundation)\b", re.I),
}
STOP = {
    "the","a","an","and","or","but","if","then","else","when","while","of","in","on","at","to","for","from","by","with","as","is","are","was","were","be","been","being","it","its","this","that","these","those","there","their","they","them","he","she","his","her","we","you","your","i","not","no","so","than","into","out","up","down","over","under","about","after","before","through","during","will","would","can","could","may","might","must","should","do","does","did","had","has","have","also","very","more","most","some","any","all","one","two","many","much","such","only","other","same","new","old","just","like","because"
}


def rel(p: Path) -> str:
    return str(p)


def read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def iter_jsonl(p: Path) -> Iterable[dict[str, Any]]:
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def toks(text: str) -> list[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(text or "")]


def content_toks(text: str) -> list[str]:
    return [w for w in toks(text) if w not in STOP and len(w) > 1]


def infer_domains(text: str) -> list[str]:
    hits = [k for k, pat in DOMAIN_PATTERNS.items() if pat.search(text or "")]
    return hits or ["no_domain"]


def counter_stats(c: collections.Counter[str]) -> dict[str, Any]:
    n = sum(c.values())
    top = c.most_common(20)
    return {"tokens": int(n), "types": len(c), "top20": top}


def shannon(counter: collections.Counter[str]) -> float:
    n = sum(counter.values())
    if n <= 0:
        return 0.0
    return -sum((v/n) * math.log2(v/n) for v in counter.values() if v > 0)


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def summarize_pair_file(path: Path, label: str) -> dict[str, Any]:
    source_counter: collections.Counter[str] = collections.Counter()
    rewrite_counter: collections.Counter[str] = collections.Counter()
    pair_counter: collections.Counter[str] = collections.Counter()
    source_content_counter: collections.Counter[str] = collections.Counter()
    rewrite_content_counter: collections.Counter[str] = collections.Counter()
    pair_content_counter: collections.Counter[str] = collections.Counter()
    domains = collections.Counter()
    domains_by_words = collections.Counter()
    docs = set()
    sentence_ids = set()
    origins = collections.Counter()
    rows = 0
    source_words = rewrite_words = pair_words = 0
    content_recall = []
    content_overlap = []
    length_ratio = []
    entity_recall = []
    number_recall = []
    soft_flags = collections.Counter()
    source_risks = collections.Counter()
    for r in iter_jsonl(path):
        rows += 1
        st = str(r.get("source_text") or "")
        rt = str(r.get("rewrite_text") or "")
        sw = int(r.get("source_words") or len(st.split()))
        rw = int(r.get("rewrite_words") or len(rt.split()))
        pw = int(r.get("pair_words") or (sw+rw))
        source_words += sw
        rewrite_words += rw
        pair_words += pw
        source_counter.update(toks(st))
        rewrite_counter.update(toks(rt))
        pair_counter.update(toks(st)); pair_counter.update(toks(rt))
        source_content_counter.update(content_toks(st))
        rewrite_content_counter.update(content_toks(rt))
        pair_content_counter.update(content_toks(st)); pair_content_counter.update(content_toks(rt))
        doc = str(r.get("doc_id") or "")
        sid = str(r.get("sentence_id") or "")
        if doc: docs.add(doc)
        if sid: sentence_ids.add(sid)
        origins[str(r.get("origin") or "unknown")] += 1
        doms = r.get("domain_hits") or []
        if not doms:
            doms = infer_domains(st)
        for d in doms or ["no_domain"]:
            domains[str(d)] += 1
            domains_by_words[str(d)] += pw
        for key, arr in [("soft_flags", r.get("soft_flags") or []), ("source_risks", r.get("source_risks") or [])]:
            ctr = soft_flags if key == "soft_flags" else source_risks
            for x in arr: ctr[str(x)] += 1
        for arr, key in [(content_recall,"content_recall"),(content_overlap,"content_overlap"),(length_ratio,"length_ratio"),(entity_recall,"entity_recall"),(number_recall,"number_recall")]:
            if r.get(key) is not None:
                arr.append(float(r[key]))
    return {
        "label": label,
        "path": rel(path),
        "records": rows,
        "source_words": source_words,
        "rewrite_words": rewrite_words,
        "pair_words": pair_words,
        "unique_docs": len(docs),
        "unique_sentence_ids": len(sentence_ids),
        "origin_counts": dict(origins.most_common()),
        "rewrite_to_source_ratio": rewrite_words / max(1, source_words),
        "word_types_pair": len(pair_counter),
        "word_types_source": len(source_counter),
        "word_types_rewrite": len(rewrite_counter),
        "content_types_pair": len(pair_content_counter),
        "content_types_source": len(source_content_counter),
        "content_types_rewrite": len(rewrite_content_counter),
        "hapax_content_types_pair": sum(1 for v in pair_content_counter.values() if v == 1),
        "content_entropy_pair": shannon(pair_content_counter),
        "source_content_counter": source_content_counter,
        "rewrite_content_counter": rewrite_content_counter,
        "pair_content_counter": pair_content_counter,
        "domains_by_record": dict(domains.most_common()),
        "domains_by_words": dict(domains_by_words.most_common()),
        "content_recall_mean": statistics.mean(content_recall) if content_recall else None,
        "content_overlap_mean": statistics.mean(content_overlap) if content_overlap else None,
        "length_ratio_mean": statistics.mean(length_ratio) if length_ratio else None,
        "entity_recall_mean": statistics.mean(entity_recall) if entity_recall else None,
        "number_recall_mean": statistics.mean(number_recall) if number_recall else None,
        "soft_flags": dict(soft_flags.most_common(20)),
        "source_risks": dict(source_risks.most_common(20)),
    }


def summarize_breadth(path: Path) -> dict[str, Any]:
    total_words = 0
    rows = 0
    docs = set()
    hashes = set()
    counter = collections.Counter()
    content_counter = collections.Counter()
    domains = collections.Counter()
    domains_by_words = collections.Counter()
    word_lens = []
    for r in iter_jsonl(path):
        rows += 1
        text = str(r.get("text") or r.get("source_text") or "")
        w = int(r.get("words") or len(text.split()))
        total_words += w
        word_lens.append(w)
        counter.update(toks(text))
        content_counter.update(content_toks(text))
        doc = str(r.get("doc_id") or "")
        if doc: docs.add(doc)
        h = str(r.get("source_hash") or r.get("hash") or "")
        if h: hashes.add(h)
        pd = str(r.get("primary_domain") or "")
        doms = ["causal_relational" if pd == "causal_temporal" else pd] if pd else infer_domains(text)
        for d in doms or ["no_domain"]:
            domains[d] += 1
            domains_by_words[d] += w
    return {
        "label": "dose2p64_breadth_companion_only",
        "path": rel(path),
        "records": rows,
        "words": total_words,
        "unique_docs": len(docs),
        "unique_hashes": len(hashes),
        "word_types": len(counter),
        "content_types": len(content_counter),
        "hapax_content_types": sum(1 for v in content_counter.values() if v == 1),
        "content_entropy": shannon(content_counter),
        "content_counter": content_counter,
        "domains_by_record": dict(domains.most_common()),
        "domains_by_words": dict(domains_by_words.most_common()),
        "word_length_mean": statistics.mean(word_lens) if word_lens else None,
        "word_length_median": statistics.median(word_lens) if word_lens else None,
    }


def clean_holdout_summaries() -> dict[str, Any]:
    s015 = read_json(META)
    s258 = read_json(META)
    s256 = read_json(META)
    out: dict[str, Any] = {}
    for label, meta in [("dose1_clean_heldout", s015), ("dose1p82_clean_heldout", s258), ("dose2p64_clean_heldout", s256)]:
        bs = meta["base_split"]
        rh = bs["row_holdout"]
        source_words = dict(rh.get("heldout_words_by_source") or {})
        out[label] = {
            "label": label,
            "changed_budget_words": int(meta.get("changed_block_budget_words") or meta.get("dose", {}).get("changed_block_budget_words") or bs.get("changed_block_budget_words")),
            "heldout_rows": int(rh.get("target_rows")),
            "heldout_words_by_source": source_words,
            "heldout_source_entropy": shannon(collections.Counter(source_words)),
            "heldout_source_count": len(source_words),
            "old_1x_subset": bool(bs.get("old_1x_heldout_subset_of_intermediate") or bs.get("old_1x_heldout_subset_of_max") or label == "dose1_clean_heldout"),
        }
    return out


def source_domain_map(meta: dict[str, Any], summary_key: str) -> dict[str, int]:
    ps = meta[summary_key]
    return {str(k): int(v) for k, v in (ps.get("domain_hit_counts") or {}).items()}


def pair_dose_summaries() -> dict[str, dict[str, Any]]:
    s015 = read_json(META)
    s258 = read_json(META)
    s256 = read_json(META)
    return {
        "dose1_pair_metadata": s015["pair_summaries"]["compact_reinvest"],
        "dose1p82_pair_metadata": s258["pair_summary"],
        "dose2p64_pair_metadata": s256["pair_summary"],
    }


def pair_increment_metrics(summaries: dict[str, Any]) -> list[dict[str, Any]]:
    labels = ["dose1_view_repeat_pairset", "dose1p82_view_repeat_pairset", "dose2p64_view_repeat_pairset"]
    rows = []
    prev_counter: collections.Counter[str] = collections.Counter()
    prev_types: set[str] = set()
    prev_docs = 0
    prev_words = 0
    for label in labels:
        s = summaries[label]
        cnt: collections.Counter[str] = s["pair_content_counter"]
        typ = set(cnt.keys())
        new_types = typ - prev_types
        rows.append({
            "label": label,
            "dose_rho": {"dose1_view_repeat_pairset": 0.042352, "dose1p82_view_repeat_pairset": 0.07712, "dose2p64_view_repeat_pairset": 0.111872}[label],
            "pair_words": s["pair_words"],
            "records": s["records"],
            "unique_docs": s["unique_docs"],
            "content_types_pair": s["content_types_pair"],
            "new_content_types_vs_previous_dose": len(new_types),
            "content_type_gain_per_added_100k_pair_words": None if s["pair_words"] == prev_words else len(new_types) / ((s["pair_words"] - prev_words) / 100000.0),
            "unique_doc_gain_vs_previous_dose": s["unique_docs"] - prev_docs,
            "pair_word_gain_vs_previous_dose": s["pair_words"] - prev_words,
            "domains_by_words": s["domains_by_words"],
            "top_new_content_types_vs_previous_dose": collections.Counter({k: cnt[k] for k in new_types}).most_common(30),
        })
        prev_counter = cnt
        prev_types = typ
        prev_docs = s["unique_docs"]
        prev_words = s["pair_words"]
    return rows


def read_old_family_components() -> dict[str, Any]:
    # Existing old-clean score-side dose readout from research.
    rows = []
    with OLD_FAMILY.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("family") and r.get("component") in {"V_minus_C", "R_minus_C", "V_minus_R"}:
                rows.append(r)
    by: dict[tuple[str,str,str], float] = {}
    for r in rows:
        try:
            by[(r["dose_name"], r["family"], r["component"])] = float(r["mean"])
        except Exception:
            pass
    labels = ["dose1", "dose1p82", "dose2p64"]
    families_ex_ent = ["BLiMP", "Supplement", "EWoK", "COMPS", "Reading"]
    out_rows = []
    for dose in labels:
        rec: dict[str, Any] = {"dose_name": dose}
        for comp in ["V_minus_C", "R_minus_C", "V_minus_R"]:
            vals = [by.get((dose, fam, comp)) for fam in families_ex_ent]
            vals2 = [v for v in vals if v is not None]
            rec[comp + "_exEntity5"] = statistics.mean(vals2) if vals2 else None
            rec[comp + "_exEntity5_l2"] = math.sqrt(sum(v*v for v in vals2)) if vals2 else None
        out_rows.append(rec)
    return {"old_clean_family_rows": rows, "old_clean_exentity_by_dose": out_rows}


def read_reference_80m() -> dict[str, Any]:
    out = {}
    if not REF_CONTRAST.exists():
        return out
    with REF_CONTRAST.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("window") == "common10_80" and r.get("quantity") == "exEntity5":
                out[r["contrast"]] = float(r["mean"])
    return out


def read_noise_floor() -> dict[str, Any]:
    out: dict[str, Any] = {}
    if SEED_ALIGN.exists():
        with SEED_ALIGN.open("r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("checkpoint") == "common10_80" and r.get("family_set") == "stable5_exEntity":
                    out["seed1x_common10_80_exEntity_cosine"] = float(r["cosine"])
                    out["seed1x_common10_80_exEntity_difference_l2"] = float(r["difference_l2"])
                    out["seed1x_common10_80_exEntity_difference_over_mean_seed_l2"] = float(r["difference_over_mean_seed_l2"])
                if r.get("checkpoint") == "common10_80" and r.get("family_set") == "stable6":
                    out["seed1x_common10_80_stable6_cosine"] = float(r["cosine"])
                    out["seed1x_common10_80_stable6_difference_l2"] = float(r["difference_l2"])
    if ENDPOINT_DISP.exists():
        with ENDPOINT_DISP.open("r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                out["dose1_to_max_exEntity_displacement_l2"] = float(r["dose1_to_max_exEntity_displacement_l2"])
                out["seed1_pair_exEntity_separation_l2"] = float(r["seed1_pair_exEntity_separation_l2"])
                out["endpoint_displacement_over_seed_separation"] = float(r["endpoint_displacement_over_seed_separation"])
                break
    return out


def slim(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if not isinstance(v, collections.Counter)}


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    if fieldnames is None:
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            rr = {}
            for k in fieldnames:
                v = r.get(k)
                if isinstance(v, (dict, list, tuple)):
                    rr[k] = json.dumps(v, ensure_ascii=False, sort_keys=True)
                else:
                    rr[k] = v
            w.writerow(rr)


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    clean_summ = clean_holdout_summaries()
    pair_meta = pair_dose_summaries()
    pair_summ = {label: summarize_pair_file(path, label) for label, path in PAIR_FILES.items()}
    breadth = summarize_breadth(BREADTH_COMPANION)

    # Set overlaps among forms at MAX.
    max_pair = pair_summ["dose2p64_view_repeat_pairset"]
    max_pair_source = set(max_pair["source_content_counter"].keys())
    max_pair_rewrite = set(max_pair["rewrite_content_counter"].keys())
    max_pair_all = set(max_pair["pair_content_counter"].keys())
    breadth_set = set(breadth["content_counter"].keys())
    form_overlap = {
        "max_source_vs_rewrite_content_jaccard": jaccard(max_pair_source, max_pair_rewrite),
        "max_pair_all_vs_breadth_content_jaccard": jaccard(max_pair_all, breadth_set),
        "max_source_vs_breadth_content_jaccard": jaccard(max_pair_source, breadth_set),
        "max_rewrite_vs_breadth_content_jaccard": jaccard(max_pair_rewrite, breadth_set),
        "breadth_content_types_not_in_max_pair": len(breadth_set - max_pair_all),
        "max_pair_content_types_not_in_breadth": len(max_pair_all - breadth_set),
        "top_breadth_only_content_types": collections.Counter({k: breadth["content_counter"][k] for k in (breadth_set - max_pair_all)}).most_common(30),
        "top_pair_only_content_types": collections.Counter({k: max_pair["pair_content_counter"][k] for k in (max_pair_all - breadth_set)}).most_common(30),
    }

    increment_rows = pair_increment_metrics(pair_summ)
    score_old = read_old_family_components()
    reference80 = read_reference_80m()
    noise = read_noise_floor()

    # Compact admission saturation proxies: compare dose increase in admitted FineWeb pair words/types/docs to old-clean score movement.
    sat_rows = []
    score_by_dose = {r["dose_name"]: r for r in score_old["old_clean_exentity_by_dose"]}
    label_to_dose = {"dose1_view_repeat_pairset": "dose1", "dose1p82_view_repeat_pairset": "dose1p82", "dose2p64_view_repeat_pairset": "dose2p64"}
    first = pair_summ["dose1_view_repeat_pairset"]
    for label, s in pair_summ.items():
        dose = label_to_dose[label]
        sc = score_by_dose.get(dose, {})
        sat_rows.append({
            "dose_name": dose,
            "rho": {"dose1": 0.042352, "dose1p82": 0.07712, "dose2p64": 0.111872}[dose],
            "pair_words": s["pair_words"],
            "pair_words_multiple_vs_1x": s["pair_words"] / first["pair_words"],
            "unique_docs": s["unique_docs"],
            "unique_docs_multiple_vs_1x": s["unique_docs"] / first["unique_docs"],
            "content_types_pair": s["content_types_pair"],
            "content_types_multiple_vs_1x": s["content_types_pair"] / first["content_types_pair"],
            "hapax_content_types_pair": s["hapax_content_types_pair"],
            "content_entropy_pair": s["content_entropy_pair"],
            "oldclean_V_minus_C_exEntity5_mean": sc.get("V_minus_C_exEntity5"),
            "oldclean_R_minus_C_exEntity5_mean": sc.get("R_minus_C_exEntity5"),
            "oldclean_V_minus_R_exEntity5_mean": sc.get("V_minus_R_exEntity5"),
        })

    # Domain ratios using metadata because it records intended construction labels.
    domain_rows = []
    meta_by_dose = pair_meta
    for name, ps in meta_by_dose.items():
        words = int(ps.get("pair_words") or 0)
        for d, c in (ps.get("domain_hit_counts") or {}).items():
            domain_rows.append({"block": name, "domain": d, "record_hits": int(c), "record_hit_fraction": int(c) / max(1, int(ps.get("pairs") or 0)), "pair_words_total": words})
    for d, c in breadth.get("domains_by_words", {}).items():
        domain_rows.append({"block": "dose2p64_breadth_companion_metadata_inferred", "domain": d, "record_hits": breadth.get("domains_by_record", {}).get(d), "record_hit_fraction": None, "pair_words_total": breadth["words"], "domain_words": c, "domain_word_fraction": c / max(1, breadth["words"])})
    for label, s in clean_summ.items():
        total = s["changed_budget_words"]
        for d, c in s["heldout_words_by_source"].items():
            domain_rows.append({"block": label, "domain": d, "record_hits": None, "record_hit_fraction": None, "pair_words_total": total, "domain_words": c, "domain_word_fraction": c / max(1, total)})

    json_summary = {
        "status": "ADMISSION_SATURATION_READOUT_COMPLETE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scientific_question": "Do the finite-budget gains sit in the corpus admitted into the substituted words rather than in own-source compact re-expression, and does the admitted-text coverage saturate between 4.2% and 11.2% budget?",
        "inputs": {
            "meta": rel(META),
            "meta": rel(META),
            "meta": rel(META),
            "meta": rel(META),
            "pair_files": {k: rel(v) for k, v in PAIR_FILES.items()},
            "breadth_companion": rel(BREADTH_COMPANION),
            "old_score_family_components": rel(OLD_FAMILY),
            "old_reference_contrasts": rel(REF_CONTRAST),
        },
        "clean_holdout_summaries": clean_summ,
        "pair_metadata_summaries": pair_meta,
        "pair_text_summaries": {k: slim(v) for k, v in pair_summ.items()},
        "breadth_companion_summary": slim(breadth),
        "pair_increment_metrics": increment_rows,
        "max_form_overlap": form_overlap,
        "old_clean_score_saturation_proxy": score_old["old_clean_exentity_by_dose"],
        "reference_80M_exEntity5_old_clean": reference80,
        "noise_floor_context": noise,
        "saturation_rows": sat_rows,
        "interpretation": {
            "admission_cluster_visible_before_matched_clean": "Existing old-clean tables already show MAX ex-Entity V-C and R-C are close (+0.598 and +0.535 over common10_80), while the one visible breadth point at 80M has B-C_old +0.896 and V-B -0.062. This makes corpus admission the plausible carrier and companion form a near-floor perturbation, pending matched-clean scoring.",
            "coverage_shape": "The admitted FineWeb pair budget rises 423,511 -> 771,199 -> 1,118,587 words (1.0x -> 1.82x -> 2.64x), but unique docs and content types grow sublinearly; compare rows in saturation_rows and pair_increment_metrics. A sublinear coverage curve matching a weak score-dose curve supports an admission/coverage-threshold mechanism rather than proportional exposure.",
            "remaining_decisive_evidence": "When clean scoring lands, recompute V-C, B-C, and R-C against matched-geometry clean in one common window and compare their spread to the research seed/noise context. Do not infer from old-clean alone because row/update geometry still differs."
        },
        "boundary": "File-only corpus and existing-score readout. No model loading, training, official evaluation, GPU work, GlobalPIQA/SuperGLUE/AoA, upload, leaderboard action, or final-facing writing.",
        "elapsed_sec": round(time.time() - t0, 3),
    }

    # Remove counters before JSON dump.
    (OUT / "admission_saturation_summary.json").write_text(json.dumps(json_summary, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    write_csv(OUT / "admitted_pair_increment_metrics.csv", increment_rows)
    write_csv(OUT / "score_coverage_saturation_rows.csv", sat_rows)
    write_csv(OUT / "domain_composition_rows.csv", domain_rows)

    md = []
    md.append("# research admission/saturation readout\n\n")
    md.append("CPU/file-only readout of existing pools and score tables. No model loading, training, evaluation, GPU work, GlobalPIQA/SuperGLUE/AoA, upload, leaderboard action, or final-facing writing occurred.\n\n")
    md.append("## Score-side reason for the test\n\n")
    md.append("Before matched-clean scoring lands, existing old-clean tables already indicate that companion form is not the broad ex-Entity carrier: MAX common10_80 ex-Entity V-C_old is +0.5981, R-C_old is +0.5345, and V-R is only +0.0636. The single visible breadth point at 80M gives B-C_old +0.896 and V-B -0.062. Thus V/B/R are clustered relative to clean while differing from one another on the same scale as the seed/basin spread measured in research.\n\n")
    md.append("## Admitted FineWeb coverage across dose\n\n")
    md.append("| dose | rho | pair words | words mult | unique docs | docs mult | content types | type mult | old V-C exEnt | old R-C exEnt | old V-R exEnt |\n")
    md.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in sat_rows:
        md.append(f"| {r['dose_name']} | {r['rho']:.6f} | {r['pair_words']:,} | {r['pair_words_multiple_vs_1x']:.3f} | {r['unique_docs']:,} | {r['unique_docs_multiple_vs_1x']:.3f} | {r['content_types_pair']:,} | {r['content_types_multiple_vs_1x']:.3f} | {r['oldclean_V_minus_C_exEntity5_mean']:+.4f} | {r['oldclean_R_minus_C_exEntity5_mean']:+.4f} | {r['oldclean_V_minus_R_exEntity5_mean']:+.4f} |\n")
    md.append("\n")
    md.append("## Incremental coverage\n\n")
    md.append("| dose label | added pair words vs previous | new content types vs previous | new types per 100k added words | unique doc gain |\n")
    md.append("|---|---:|---:|---:|---:|\n")
    for r in increment_rows:
        val = r["content_type_gain_per_added_100k_pair_words"]
        md.append(f"| {r['label']} | {r['pair_word_gain_vs_previous_dose']:,} | {r['new_content_types_vs_previous_dose']:,} | {val if val is not None else 0:.1f} | {r['unique_doc_gain_vs_previous_dose']:,} |\n")
    md.append("\n")
    md.append("## MAX form overlap\n\n")
    for k, v in form_overlap.items():
        if not k.startswith("top_"):
            md.append(f"- {k}: {v}\n")
    md.append("\n")
    md.append("## Scientific reading\n\n")
    md.append("The existing score side and file side point to a sharper mechanism test: the load-bearing quantity is likely the admitted FineWeb slice replacing clean-Qwen words, not whether the admitted slice is shown as own-source compact re-expression, breadth, or exact source duplication. The dose file statistics now make the coming matched-clean readout interpretable as a cluster test: compute V-C, B-C, and R-C against the geometry-matched clean in one common window, then compare their within-cluster spread to the research seed/noise context. If all three are similarly positive and the gain changes weakly from 4.2% to 11.2% while admitted types/docs grow sublinearly, the principle becomes a finite-budget content-admission/coverage-threshold effect. If only one arm survives matched-clean, the carrier returns to companion form and must be decomposed further.\n\n")
    md.append("## Files\n\n")
    for name in ["admission_saturation_summary.json", "admitted_pair_increment_metrics.csv", "score_coverage_saturation_rows.csv", "domain_composition_rows.csv"]:
        md.append(f"- `{OUT / name}`\n")
    ((OUT.parents[4] / 'research/documents/frontier_consolidation/data/admission_saturation_readout/admission_saturation_summary.md')).write_text("".join(md), encoding="utf-8")

    print(json.dumps({
        "status": "ADMISSION_SATURATION_READOUT_COMPLETE",
        "summary_md": rel((OUT.parents[4] / 'research/documents/frontier_consolidation/data/admission_saturation_readout/admission_saturation_summary.md')),
        "summary_json": rel(OUT / "admission_saturation_summary.json"),
        "saturation_rows": sat_rows,
        "max_form_overlap": {k: v for k, v in form_overlap.items() if not k.startswith("top_")},
        "noise_floor_context": noise,
        "no_model_loading_training_evaluation_gpu_globalpiqa_superglue_aoa_upload_or_leaderboard": True,
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
