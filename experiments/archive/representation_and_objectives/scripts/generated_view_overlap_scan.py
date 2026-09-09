#!/usr/bin/env python3
"""research CPU-only scored-text overlap scan for generated-view corpus components.

This script reuses the research/research official evaluation coordinate and scans the
legal compact_view_reinvest 10M corpus, with special separation of generated rewrites
from their original source counterparts. It is designed to make any future legal40k
endpoint easier to defend scientifically: if a score jump appears, the record will
show whether exact scored-text spans are concentrated in original public/official
source material or in teacher-generated rewrites.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import hashlib
import json
import pathlib
import re
import time
from collections import Counter, defaultdict
from typing import Any, Iterable

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/representation_and_objectives')
PEER = _public_path('experiments/archive/frontier_consolidation')
OUT_DIR = _public_path('experiments/archive/representation_and_objectives/data/generated_view_overlap_scan')
OUT_JSON = _public_path('experiments/archive/representation_and_objectives/data/generated_view_overlap_scan/generated_view_overlap_scan.json')
OUT_COMPONENT_CSV = _public_path('experiments/archive/representation_and_objectives/data/generated_view_overlap_scan/component_kind_overlap_summary.csv')
OUT_ROW_CSV = _public_path('experiments/archive/representation_and_objectives/data/generated_view_overlap_scan/row_source_overlap_summary.csv')
OUT_NGRAM_CSV = _public_path('experiments/archive/representation_and_objectives/data/generated_view_overlap_scan/matched_ngram_summary.csv')
OUT_DETAIL_CSV = _public_path('experiments/archive/representation_and_objectives/data/generated_view_overlap_scan/generated_rewrite_overlap_details.csv')
NOTE = _public_path('research/notes/representation_and_objectives/generated_view_overlap_scan.md')

PRISTINE_EVAL_ROOT = _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval')
GLOBALPIQA_EVAL_ROOT = _public_path('experiments/archive/representation_and_objectives/data/globalpiqa_official_lineage/official_dl_scratch/generated_by_current_official_dl/evaluation_data/full_eval')
ACTIVE_10M = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl')
CHANGED_META = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_changed_block_rows_meta.jsonl')
COMPACT_PAIRS = _public_path('experiments/archive/frontier_consolidation/data/density_core_reinvestment_medium_riskhard/selected_compact_reinvest_pairs.jsonl')
QWEN_PACKED_META = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/qwen_pair_packed_rows_meta.jsonl')
QWEN_SELECTED_PAIRS = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/selected_pairs.jsonl')

EXPECTED_SHA = {
    "active_10m": "215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23",
    "compact_pairs": "d2a3110c110e216180b632ce7a181acdf348b804ea19f172e16b5afeb0d8c9fc",
    "qwen_selected_pairs": "5edec9a76470036f3ae06f9ed3c53f060ad7811b20697fddf0034cc24f11b53c",
}

NS = (7, 8, 10)
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
TEXT_KEYS = {
    "sentence", "sentence_good", "sentence_bad", "good_sentence", "bad_sentence",
    "context", "question", "passage", "paragraph", "premise", "hypothesis",
    "text", "text_a", "text_b", "input", "target", "option1", "option2",
    "answer", "correct", "incorrect", "choice1", "choice2", "query", "title",
    "article", "summary", "word", "stem", "ending0", "ending1", "ending2", "ending3",
    "prompt", "solution0", "solution1", "solution2", "solution3",
}
SKIP_KEYS = {"id", "idx", "uid", "guid", "label", "labels", "metadata", "meta", "path", "file", "filename", "source", "source_file", "example_id"}
STRICT_SCORE_TOP_DIRS = {
    "aoa", "blimp_filtered", "supplement_filtered", "ewok_filtered", "entity_tracking",
    "comps", "reading", "global_piqa_parallel", "global_piqa_nonparallel",
}
STRICT_NON_SCORE_TOP_DIRS = {"vqa_filtered", "winoground_filtered"}


def toks(text: str) -> list[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(text or "")]


def ngrams(ts: list[str], n: int) -> set[str]:
    if len(ts) < n:
        return set()
    return {" ".join(ts[i:i+n]) for i in range(len(ts) - n + 1)}


def iter_texts(obj: Any, parent_key: str = "") -> Iterable[str]:
    if isinstance(obj, str):
        pk = parent_key.lower()
        if pk in SKIP_KEYS:
            return
        if pk in TEXT_KEYS or len(toks(obj)) >= 5:
            yield obj
    elif isinstance(obj, list):
        for x in obj:
            yield from iter_texts(x, parent_key)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in SKIP_KEYS:
                continue
            yield from iter_texts(v, str(k))


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonl_count(path: pathlib.Path) -> int:
    c = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                c += 1
    return c


def classify_eval_file(rel: pathlib.Path) -> dict[str, Any]:
    top = rel.parts[0] if rel.parts else rel.name
    name = rel.name
    if top == "glue_filtered":
        if ".valid." in name:
            return {"top_dir": top, "file_role": "superglue_validation", "strict_small_score_text": True}
        if ".train." in name:
            return {"top_dir": top, "file_role": "superglue_finetune_train", "strict_small_score_text": False}
        return {"top_dir": top, "file_role": "superglue_unknown", "strict_small_score_text": False}
    if top in STRICT_NON_SCORE_TOP_DIRS:
        return {"top_dir": top, "file_role": "multimodal_not_used_for_strict_small_text_score", "strict_small_score_text": False}
    if top in STRICT_SCORE_TOP_DIRS:
        return {"top_dir": top, "file_role": "strict_small_score_text", "strict_small_score_text": True}
    return {"top_dir": top, "file_role": "not_in_strict_small_text_overall", "strict_small_score_text": False}


def eval_sources() -> list[tuple[pathlib.Path, pathlib.Path]]:
    srcs: list[tuple[pathlib.Path, pathlib.Path]] = []
    for p in sorted(PRISTINE_EVAL_ROOT.rglob("*")):
        if p.is_file() and p.suffix.lower() in {".json", ".jsonl", ".csv", ".txt"}:
            srcs.append((p, p.relative_to(PRISTINE_EVAL_ROOT)))
    for task_dir in ["global_piqa_parallel", "global_piqa_nonparallel"]:
        p = GLOBALPIQA_EVAL_ROOT / task_dir / "eng_latn.jsonl"
        if p.exists():
            srcs.append((p, pathlib.Path(task_dir) / "eng_latn.jsonl"))
    return srcs


def build_strict_eval_index() -> tuple[list[dict[str, Any]], dict[int, dict[str, list[int]]], dict[str, int]]:
    records: list[dict[str, Any]] = []
    ng_to_ids: dict[int, dict[str, list[int]]] = {n: defaultdict(list) for n in NS}  # type: ignore[assignment]
    counts: Counter[str] = Counter()

    def add_record(rel: pathlib.Path, locator: str, text: str, cls: dict[str, Any]) -> None:
        if not cls.get("strict_small_score_text"):
            return
        ts = toks(text)
        if len(ts) < min(NS):
            return
        rid = len(records)
        rec = {
            "id": rid,
            "file": str(rel),
            "locator": locator,
            "text_prefix": text[:320],
            "num_tokens": len(ts),
            "top_dir": cls["top_dir"],
            "file_role": cls["file_role"],
        }
        records.append(rec)
        counts[cls["top_dir"]] += 1
        for n in NS:
            for ng in ngrams(ts, n):
                ng_to_ids[n][ng].append(rid)

    for p, rel in eval_sources():
        cls = classify_eval_file(rel)
        try:
            suffix = p.suffix.lower()
            if suffix == ".jsonl":
                with p.open(encoding="utf-8") as f:
                    for li, line in enumerate(f, 1):
                        if not line.strip():
                            continue
                        try:
                            obj = json.loads(line)
                            for text in iter_texts(obj):
                                add_record(rel, f"line:{li}", text, cls)
                        except json.JSONDecodeError:
                            add_record(rel, f"line:{li}", line, cls)
            elif suffix == ".json":
                obj = json.loads(p.read_text(encoding="utf-8"))
                for text in iter_texts(obj):
                    add_record(rel, "json", text, cls)
            elif suffix == ".csv":
                with p.open(encoding="utf-8", newline="") as f:
                    for ri, row in enumerate(csv.DictReader(f), 1):
                        for key, val in row.items():
                            if val and isinstance(val, str):
                                add_record(rel, f"row:{ri}:col:{key}", val, cls)
            else:
                add_record(rel, "txt", p.read_text(encoding="utf-8", errors="ignore"), cls)
        except Exception as exc:
            # Preserve read errors in the summary rather than silently changing the score-text coordinate.
            counts[f"_read_error::{rel}"] += 1
    return records, ng_to_ids, dict(counts)


def load_jsonl_by(path: pathlib.Path, key: str) -> dict[Any, dict[str, Any]]:
    out: dict[Any, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            out[obj[key]] = obj
    return out


def load_changed_meta_by_row() -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    with CHANGED_META.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                out[int(obj["row_index"])] = obj
    return out


def load_qwen_meta_by_example() -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    with QWEN_PACKED_META.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                out[int(obj["example_id"])] = obj
    return out


def init_kind_summary() -> dict[str, Any]:
    return {
        "components": 0,
        "words_declared": 0,
        "tokens_lexical": 0,
        "components_with_any_match_n7": 0,
        "components_with_any_match_n8": 0,
        "components_with_any_match_n10": 0,
        "unique_matched_ngrams_n7": set(),
        "unique_matched_ngrams_n8": set(),
        "unique_matched_ngrams_n10": set(),
        "component_ngram_hit_count_n7": 0,
        "component_ngram_hit_count_n8": 0,
        "component_ngram_hit_count_n10": 0,
        "eval_record_hit_count_n7": 0,
        "eval_record_hit_count_n8": 0,
        "eval_record_hit_count_n10": 0,
        "top_dir_hits_n7": Counter(),
        "top_dir_hits_n8": Counter(),
        "top_dir_hits_n10": Counter(),
    }


def row_source_class(source: str) -> str:
    if source == "cleanqwen_fineweb_compact_view_reinvest":
        return "fineweb_source_qwen_compact_rewrite_pair_row"
    if source == "qwen_pair_packed":
        return "inherited_official_source_qwen_paraphrase_pair_row"
    if source.startswith("neutral_cleanqwen_topup"):
        return "neutral_topup_from_heldout_official_row"
    return "official_babylm_source_row"


def scan_component(
    *,
    component_kind: str,
    branch: str,
    pair_id: str,
    text: str,
    words: int,
    source_label: str,
    component_meta: dict[str, Any],
    eval_records: list[dict[str, Any]],
    ng_to_ids: dict[int, dict[str, list[int]]],
    kind_summary: dict[str, dict[str, Any]],
    ngram_summary: dict[tuple[int, str], dict[str, Any]],
    detail_rows: list[dict[str, Any]],
    detail_cap: int = 20000,
) -> None:
    ks = kind_summary.setdefault(component_kind, init_kind_summary())
    ts = toks(text)
    ks["components"] += 1
    ks["words_declared"] += int(words or 0)
    ks["tokens_lexical"] += len(ts)
    for n in NS:
        matched_ngrams: list[str] = []
        for ng in ngrams(ts, n):
            ids = ng_to_ids[n].get(ng)
            if not ids:
                continue
            matched_ngrams.append(ng)
            ks[f"unique_matched_ngrams_n{n}"].add(ng)
            ks[f"component_ngram_hit_count_n{n}"] += 1
            ks[f"eval_record_hit_count_n{n}"] += len(set(ids))
            top_dirs = Counter(eval_records[rid]["top_dir"] for rid in set(ids))
            ks[f"top_dir_hits_n{n}"].update(top_dirs)
            ngs = ngram_summary.setdefault((n, ng), {
                "n": n,
                "ngram": ng,
                "component_kinds": Counter(),
                "branches": Counter(),
                "pairs": set(),
                "eval_records": set(),
                "top_dirs": Counter(),
            })
            ngs["component_kinds"].update([component_kind])
            ngs["branches"].update([branch])
            ngs["pairs"].add(pair_id)
            ngs["eval_records"].update(set(ids))
            ngs["top_dirs"].update(top_dirs)
            if "rewrite" in component_kind and len(detail_rows) < detail_cap:
                for rid in sorted(set(ids))[:8]:
                    rec = eval_records[rid]
                    detail_rows.append({
                        "n": n,
                        "ngram": ng,
                        "component_kind": component_kind,
                        "branch": branch,
                        "pair_id": pair_id,
                        "source_label": source_label,
                        "words": int(words or 0),
                        "lexical_tokens": len(ts),
                        "eval_top_dir": rec["top_dir"],
                        "eval_file": rec["file"],
                        "eval_locator": rec["locator"],
                        "component_text_prefix": text[:240].replace("\n", " "),
                        "eval_text_prefix": rec["text_prefix"].replace("\n", " "),
                        "meta": json.dumps(component_meta, ensure_ascii=False, sort_keys=True)[:500],
                    })
        if matched_ngrams:
            ks[f"components_with_any_match_n{n}"] += 1


def finalize_kind_summary(kind_summary: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for kind, d in sorted(kind_summary.items()):
        row: dict[str, Any] = {"component_kind": kind}
        for k, v in d.items():
            if isinstance(v, set):
                row[k] = len(v)
            elif isinstance(v, Counter):
                row[k] = dict(v)
            else:
                row[k] = v
        rows.append(row)
    return rows


def scan_generated_components(eval_records: list[dict[str, Any]], ng_to_ids: dict[int, dict[str, list[int]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    kind_summary: dict[str, dict[str, Any]] = {}
    ngram_summary: dict[tuple[int, str], dict[str, Any]] = {}
    detail_rows: list[dict[str, Any]] = []

    compact_pairs = load_jsonl_by(COMPACT_PAIRS, "pair_id")
    for pid, p in sorted(compact_pairs.items()):
        meta = {"doc_id": p.get("doc_id"), "sentence_id": p.get("sentence_id"), "domain_hits": p.get("domain_hits")}
        scan_component(
            component_kind="fineweb_compact_original_source",
            branch="compact_fineweb",
            pair_id=str(pid),
            text=p.get("source_text", ""),
            words=int(p.get("source_words") or 0),
            source_label="fineweb",
            component_meta=meta,
            eval_records=eval_records,
            ng_to_ids=ng_to_ids,
            kind_summary=kind_summary,
            ngram_summary=ngram_summary,
            detail_rows=detail_rows,
        )
        scan_component(
            component_kind="fineweb_compact_qwen_rewrite",
            branch="compact_fineweb",
            pair_id=str(pid),
            text=p.get("rewrite_text", ""),
            words=int(p.get("rewrite_words") or 0),
            source_label="qwen3.5-9b compact rewrite of fineweb source",
            component_meta=meta,
            eval_records=eval_records,
            ng_to_ids=ng_to_ids,
            kind_summary=kind_summary,
            ngram_summary=ngram_summary,
            detail_rows=detail_rows,
        )

    qwen_pairs = load_jsonl_by(QWEN_SELECTED_PAIRS, "pair_id")
    for pid, p in sorted(qwen_pairs.items()):
        meta = {"source": p.get("source"), "example_id": p.get("example_id"), "cohort": p.get("cohort")}
        scan_component(
            component_kind="official_source_for_qwen_paraphrase",
            branch="clean_qwen_official",
            pair_id=str(pid),
            text=p.get("original", ""),
            words=int(p.get("original_words") or 0),
            source_label=str(p.get("source")),
            component_meta=meta,
            eval_records=eval_records,
            ng_to_ids=ng_to_ids,
            kind_summary=kind_summary,
            ngram_summary=ngram_summary,
            detail_rows=detail_rows,
        )
        scan_component(
            component_kind="official_source_qwen_paraphrase_rewrite",
            branch="clean_qwen_official",
            pair_id=str(pid),
            text=p.get("rewrite", ""),
            words=int(p.get("rewrite_words") or 0),
            source_label=f"qwen3.5-9b paraphrase of {p.get('source')}",
            component_meta=meta,
            eval_records=eval_records,
            ng_to_ids=ng_to_ids,
            kind_summary=kind_summary,
            ngram_summary=ngram_summary,
            detail_rows=detail_rows,
        )

    component_rows = finalize_kind_summary(kind_summary)
    ngram_rows: list[dict[str, Any]] = []
    for (_n, _ng), d in ngram_summary.items():
        ngram_rows.append({
            "n": d["n"],
            "ngram": d["ngram"],
            "component_kind_counts": dict(d["component_kinds"]),
            "branch_counts": dict(d["branches"]),
            "pair_count": len(d["pairs"]),
            "eval_record_count": len(d["eval_records"]),
            "top_dir_counts": dict(d["top_dirs"]),
        })
    ngram_rows.sort(key=lambda r: (r["n"], r["pair_count"], r["eval_record_count"]), reverse=True)
    detail_rows.sort(key=lambda r: (r["n"], r["component_kind"], r["pair_id"], r["eval_top_dir"]), reverse=True)
    return component_rows, ngram_rows, detail_rows


def scan_active_rows(eval_records: list[dict[str, Any]], ng_to_ids: dict[int, dict[str, list[int]]]) -> list[dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    with ACTIVE_10M.open(encoding="utf-8") as f:
        for row_index, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            source = str(obj.get("source", ""))
            klass = row_source_class(source)
            d = summary.setdefault(klass, {
                "row_source_class": klass,
                "rows": 0,
                "words": 0,
                "rows_with_any_match_n7": 0,
                "rows_with_any_match_n8": 0,
                "rows_with_any_match_n10": 0,
                "row_ngram_hit_count_n7": 0,
                "row_ngram_hit_count_n8": 0,
                "row_ngram_hit_count_n10": 0,
                "eval_record_hit_count_n7": 0,
                "eval_record_hit_count_n8": 0,
                "eval_record_hit_count_n10": 0,
                "unique_matched_ngrams_n7": set(),
                "unique_matched_ngrams_n8": set(),
                "unique_matched_ngrams_n10": set(),
                "top_dir_hits_n7": Counter(),
                "top_dir_hits_n8": Counter(),
                "top_dir_hits_n10": Counter(),
            })
            d["rows"] += 1
            d["words"] += int(obj.get("words") or 0)
            ts = toks(obj.get("text", ""))
            for n in NS:
                row_hit = False
                for ng in ngrams(ts, n):
                    ids = ng_to_ids[n].get(ng)
                    if not ids:
                        continue
                    row_hit = True
                    d[f"unique_matched_ngrams_n{n}"].add(ng)
                    d[f"row_ngram_hit_count_n{n}"] += 1
                    d[f"eval_record_hit_count_n{n}"] += len(set(ids))
                    d[f"top_dir_hits_n{n}"].update(Counter(eval_records[rid]["top_dir"] for rid in set(ids)))
                if row_hit:
                    d[f"rows_with_any_match_n{n}"] += 1
    out: list[dict[str, Any]] = []
    for _, d in sorted(summary.items()):
        row: dict[str, Any] = {}
        for k, v in d.items():
            if isinstance(v, set):
                row[k] = len(v)
            elif isinstance(v, Counter):
                row[k] = dict(v)
            else:
                row[k] = v
        out.append(row)
    return out


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]], preferred: list[str] | None = None) -> None:
    if preferred is None:
        keys: list[str] = []
        for r in rows:
            for k in r.keys():
                if k not in keys:
                    keys.append(k)
    else:
        extra = []
        for r in rows:
            for k in r.keys():
                if k not in preferred and k not in extra:
                    extra.append(k)
        keys = preferred + extra
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            rr = {}
            for k in keys:
                v = r.get(k, "")
                if isinstance(v, (dict, list, tuple)):
                    rr[k] = json.dumps(v, ensure_ascii=False, sort_keys=True)
                else:
                    rr[k] = v
            w.writerow(rr)


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _public_path('research/notes/representation_and_objectives').mkdir(parents=True, exist_ok=True)

    eval_records, ng_to_ids, eval_counts_by_top = build_strict_eval_index()
    component_rows, ngram_rows, detail_rows = scan_generated_components(eval_records, ng_to_ids)
    row_rows = scan_active_rows(eval_records, ng_to_ids)

    artifact_fingerprints = {
        "active_10m": {"path": str(ACTIVE_10M), "sha256": sha256_file(ACTIVE_10M), "rows": jsonl_count(ACTIVE_10M), "expected_sha256": EXPECTED_SHA["active_10m"]},
        "compact_pairs": {"path": str(COMPACT_PAIRS), "sha256": sha256_file(COMPACT_PAIRS), "rows": jsonl_count(COMPACT_PAIRS), "expected_sha256": EXPECTED_SHA["compact_pairs"]},
        "qwen_selected_pairs": {"path": str(QWEN_SELECTED_PAIRS), "sha256": sha256_file(QWEN_SELECTED_PAIRS), "rows": jsonl_count(QWEN_SELECTED_PAIRS), "expected_sha256": EXPECTED_SHA["qwen_selected_pairs"]},
    }
    for d in artifact_fingerprints.values():
        d["sha256_matches_expected"] = d["sha256"] == d["expected_sha256"]

    rewrite_kinds = {"fineweb_compact_qwen_rewrite", "official_source_qwen_paraphrase_rewrite"}
    rewrite_summary = {r["component_kind"]: r for r in component_rows if r["component_kind"] in rewrite_kinds}
    original_summary = {r["component_kind"]: r for r in component_rows if r["component_kind"] not in rewrite_kinds}

    payload = {
        "status": "GENERATED_VIEW_OVERLAP_SCAN",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "CPU-only exact scored-text span scan for legal compact_view_reinvest training components while legal40k official evaluations run asynchronously.",
        "n_values": list(NS),
        "official_coordinate": {
            "pristine_eval_root": str(PRISTINE_EVAL_ROOT),
            "official_generated_globalpiqa_root": str(GLOBALPIQA_EVAL_ROOT),
            "strict_score_text_definition": "Strict-Small text Overall columns: BLiMP, Supplement, EWoK, Entity, COMPS, SuperGLUE validation, GlobalPIQA, Reading, AoA.",
            "strict_eval_record_counts_by_top": eval_counts_by_top,
            "strict_eval_records_total": len(eval_records),
            "strict_eval_unique_ngrams": {str(n): len(ng_to_ids[n]) for n in NS},
        },
        "artifact_fingerprints": artifact_fingerprints,
        "component_kind_summary": component_rows,
        "row_source_summary": row_rows,
        "rewrite_component_summary": rewrite_summary,
        "original_component_summary": original_summary,
        "top_matched_ngrams": ngram_rows[:200],
        "generated_rewrite_detail_rows_written": len(detail_rows),
        "outputs": {
            "json": str(OUT_JSON),
            "component_csv": str(OUT_COMPONENT_CSV),
            "row_csv": str(OUT_ROW_CSV),
            "ngram_csv": str(OUT_NGRAM_CSV),
            "generated_rewrite_detail_csv": str(OUT_DETAIL_CSV),
            "note": str(NOTE),
        },
        "interpretation_limits": [
            "Exact n-gram sharing is a provenance and defensibility signal, not a causal estimate of benchmark score movement.",
            "Official BabyLM source rows are allowed input material for all Strict-Small submissions, so exact overlaps in official source rows are interpreted separately from generated rewrites.",
            "The scan does not use any model predictions and does not interact with the running legal40k evaluations.",
        ],
        "elapsed_sec": round(time.time() - t0, 3),
    }

    write_csv(OUT_COMPONENT_CSV, component_rows)
    write_csv(OUT_ROW_CSV, row_rows)
    write_csv(OUT_NGRAM_CSV, ngram_rows[:5000])
    write_csv(OUT_DETAIL_CSV, detail_rows)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def comp_line(kind: str) -> str:
        r = rewrite_summary.get(kind) or original_summary.get(kind) or {}
        return (
            f"- `{kind}`: components={r.get('components')}, words={r.get('words_declared')}, "
            f"n7 components with score-text span={r.get('components_with_any_match_n7')}, "
            f"n8={r.get('components_with_any_match_n8')}, n10={r.get('components_with_any_match_n10')}, "
            f"unique n7 spans={r.get('unique_matched_ngrams_n7')}"
        )

    note_lines = [
        "# research generated-view scored-text overlap scan",
        "",
        "This CPU-only scan ran while the two legal40k official evaluations remained managed asynchronously. It did not read task status, train a model, or evaluate a model.",
        "",
        "## Official coordinate and files",
        f"- Strict score text came from `{PRISTINE_EVAL_ROOT}` plus research GlobalPIQA files at `{GLOBALPIQA_EVAL_ROOT}`.",
        f"- Active 10M corpus: `{ACTIVE_10M}`; SHA match: `{artifact_fingerprints['active_10m']['sha256_matches_expected']}`.",
        f"- Compact FineWeb pairs: `{COMPACT_PAIRS}`; SHA match: `{artifact_fingerprints['compact_pairs']['sha256_matches_expected']}`.",
        f"- Clean-Qwen official-source pairs: `{QWEN_SELECTED_PAIRS}`; SHA match: `{artifact_fingerprints['qwen_selected_pairs']['sha256_matches_expected']}`.",
        f"- Strict eval text records: {len(eval_records)}; unique score-text ngrams: {payload['official_coordinate']['strict_eval_unique_ngrams']}.",
        "",
        "## Generated rewrite components",
        comp_line("fineweb_compact_qwen_rewrite"),
        comp_line("official_source_qwen_paraphrase_rewrite"),
        "",
        "## Original/source components",
        comp_line("fineweb_compact_original_source"),
        comp_line("official_source_for_qwen_paraphrase"),
        "",
        "## Row-level source summary",
    ]
    for r in row_rows:
        note_lines.append(
            f"- `{r['row_source_class']}`: rows={r['rows']}, words={r['words']}, "
            f"rows with n7/n8/n10 exact score spans={r['rows_with_any_match_n7']}/{r['rows_with_any_match_n8']}/{r['rows_with_any_match_n10']}, "
            f"unique n7 spans={r['unique_matched_ngrams_n7']}"
        )
    note_lines += [
        "",
        "## Interpretation",
        "Exact overlap is recorded to protect future endpoint interpretation. It does not estimate why a model scores well. The important separation here is between allowed original/source text and teacher-generated rewrites; if generated-rewrite exact spans are sparse and generic, a future high score is less likely to be explained by direct teacher production of score strings.",
        "",
        f"JSON: `{OUT_JSON}`",
        f"Component CSV: `{OUT_COMPONENT_CSV}`",
        f"Row CSV: `{OUT_ROW_CSV}`",
        f"Ngram CSV: `{OUT_NGRAM_CSV}`",
        f"Generated rewrite detail CSV: `{OUT_DETAIL_CSV}`",
    ]
    NOTE.write_text("\n".join(note_lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": payload["status"],
        "elapsed_sec": payload["elapsed_sec"],
        "strict_eval_records_total": len(eval_records),
        "rewrite_component_summary": rewrite_summary,
        "row_source_summary": row_rows,
        "out_json": str(OUT_JSON),
        "note": str(NOTE),
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
