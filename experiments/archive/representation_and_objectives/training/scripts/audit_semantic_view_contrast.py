#!/usr/bin/env python3
"""Audit research semantic-view matched contrast corpora.

Checks exact word accounting, row-length identity, identical official filler,
training-file exposure totals if present, and baseline16k token-length/truncation
statistics.  It audits all source-only comparison arms: the stream-matched
control, the packet-local cyclic-offset control preserving
row identity and boundaries, and the packet-exact prefix-repetition forensic
artifact when present.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import collections
import hashlib
import importlib.util
import json
import pathlib
import statistics
import sys
from typing import Iterable, Any

from transformers import AutoTokenizer

ROOT = pathlib.Path("experiments/archive/representation_and_objectives")
DEFAULT_DIR = ROOT / "training/data/semantic_view/full_contrast"
TOKENIZER = pathlib.Path("experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model")
MAX_SEQ = 256


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stats(vals: Iterable[int | float]) -> dict:
    xs = list(vals)
    if not xs:
        return {"n": 0}
    ys = sorted(xs)
    def q(p: float):
        return ys[min(len(ys)-1, max(0, round((len(ys)-1)*p)))]
    return {"n": len(xs), "min": min(xs), "p05": q(0.05), "mean": round(statistics.mean(xs), 4), "median": statistics.median(xs), "p95": q(0.95), "p99": q(0.99), "max": max(xs), "sum": sum(xs)}


def read_rows(path: pathlib.Path, expected_words: int | None = None) -> list[dict]:
    rows = []
    total = 0
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            o = json.loads(line)
            text = str(o["text"])
            w = int(o.get("words", len(text.split())))
            actual = len(text.split())
            if w != actual:
                raise RuntimeError(f"{path} row {i} word mismatch field={w} actual={actual}")
            rows.append({"text": text, "words": w, "source": str(o.get("source", "")), "example_id": int(o.get("example_id", i))})
            total += w
    if expected_words is not None and total != expected_words:
        raise RuntimeError(f"{path} words {total} != expected {expected_words}")
    return rows


def source_words(rows: list[dict]) -> dict[str, int]:
    c = collections.Counter()
    for r in rows:
        c[r["source"]] += int(r["words"])
    return dict(c)


def token_audit(rows: list[dict], tok, sample_limit: int = 0) -> dict:
    lens = []
    over = 0
    excess = 0
    examples = []
    iterable = rows if sample_limit <= 0 else rows[:sample_limit]
    for i, r in enumerate(iterable):
        ids = tok(r["text"], add_special_tokens=True, truncation=False)["input_ids"]
        L = len(ids)
        lens.append(L)
        if L > MAX_SEQ:
            over += 1
            excess += L - MAX_SEQ
            if len(examples) < 20:
                examples.append({"row": i, "source": r["source"], "words": r["words"], "tokens": L, "text_prefix": r["text"][:260]})
    return {"rows_audited": len(iterable), "token_length_stats": stats(lens), "rows_over_seq256": over, "fraction_rows_over_seq256": round(over / max(1, len(iterable)), 6), "excess_tokens_if_truncated_at_256": excess, "over_seq256_examples": examples}


def compare_after_prefix(treat: list[dict], ctrl: list[dict], prefix_rows: int) -> dict:
    if len(treat) != len(ctrl):
        return {"identical_after_prefix": False, "reason": "row_count_mismatch"}
    mismatches = []
    for i in range(prefix_rows, len(treat)):
        a = treat[i]
        b = ctrl[i]
        if a != b:
            mismatches.append({"row": i, "treatment": {k: a[k] for k in ["words", "source", "example_id"]}, "control": {k: b[k] for k in ["words", "source", "example_id"]}})
            if len(mismatches) >= 20:
                break
    return {"identical_after_prefix": len(mismatches) == 0, "mismatch_count_sampled_until20": len(mismatches), "mismatch_samples": mismatches}


def training_info(path: pathlib.Path, expected_words: int) -> dict:
    if not path.exists():
        return {"exists": False}
    rows = 0
    words = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            rows += 1
            words += int(o["words"])
    return {"exists": True, "rows": rows, "words": words, "expected_words_match": words == expected_words, "sha256": sha256_file(path)}


def read_meta_rows(path: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def compare_packet_identity(semantic_meta: list[dict], packet_meta: list[dict], prefix_rows: int) -> dict:
    """Verify packet-local control preserves treatment source identity/boundary."""
    if len(semantic_meta) != prefix_rows or len(packet_meta) != prefix_rows:
        return {
            "same_packet_count": False,
            "semantic_meta_rows": len(semantic_meta),
            "packet_meta_rows": len(packet_meta),
            "expected_prefix_rows": prefix_rows,
        }
    mismatches = []
    for i, (a, b) in enumerate(zip(semantic_meta, packet_meta)):
        fields = ["row_index", "example_id", "words", "source_key", "source_article", "view_types", "prompt_ids"]
        diffs = {k: (a.get(k), b.get(k)) for k in fields if a.get(k) != b.get(k)}
        if diffs:
            mismatches.append({"row": i, "diffs": diffs})
            if len(mismatches) >= 20:
                break
    return {
        "same_packet_count": True,
        "source_identity_and_row_fields_identical": len(mismatches) == 0,
        "mismatch_count_sampled_until20": len(mismatches),
        "mismatch_samples": mismatches,
    }


def load_materializer_module():
    path = _public_path('experiments/archive/representation_and_objectives/training/scripts/audit_semantic_view_contrast.py').with_name("materialize_semantic_view_contrast.py")
    spec = importlib.util.spec_from_file_location("materialize_semantic_view_contrast_for_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load materializer module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def deep_prefix_construction_check(d: pathlib.Path, meta: dict[str, Any], treat: list[dict], packet_local: list[dict], semantic_meta: list[dict], prefix_rows: int) -> dict:
    """Reconstruct the intervention prefix from prompts/accepted views and compare actual strings."""
    if not packet_local:
        return {"ok": False, "reason": "missing_packet_local"}
    mat = load_materializer_module()
    prompts_path = pathlib.Path(meta.get("prompts", ""))
    if not prompts_path.exists():
        return {"ok": False, "reason": f"prompts_missing:{prompts_path}"}
    prompts = mat.load_prompts(prompts_path)
    prompt_by_key = {p.source_key: p for p in prompts}
    accepted_by_prompt_id: dict[str, dict[str, Any]] = {}
    acc_path = d / "accepted_views.jsonl"
    if not acc_path.exists():
        return {"ok": False, "reason": f"accepted_views_missing:{acc_path}"}
    with acc_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                o = json.loads(line)
                accepted_by_prompt_id[str(o["prompt_id"])] = o
    mismatches = []
    prefix_fail = []
    control_vocab_fail = []
    checked = 0
    for i in range(prefix_rows):
        sm = semantic_meta[i]
        source_key = sm.get("source_key")
        p = prompt_by_key.get(source_key)
        if p is None:
            mismatches.append({"row": i, "kind": "missing_prompt", "source_key": source_key})
            continue
        prompt_ids = list(sm.get("prompt_ids") or [])
        views = []
        for pid in prompt_ids:
            v = accepted_by_prompt_id.get(str(pid))
            if v is None:
                mismatches.append({"row": i, "kind": "missing_accepted_view", "prompt_id": pid})
            else:
                views.append(v)
        views_sorted = sorted(views, key=lambda v: {"simplification": 0, "paraphrase": 1}.get(v.get("typ"), 9))
        expected_treat = mat.normalize_ws(" ".join([p.source_text] + [str(v.get("output", "")) for v in views_sorted]))
        expected_control = mat.balanced_packet_local_original_to_length(p.source_text, int(treat[i]["words"]), offset=i * 3 + 1)
        actual_treat = mat.normalize_ws(str(treat[i]["text"]))
        actual_control = mat.normalize_ws(str(packet_local[i]["text"]))
        if actual_treat != expected_treat:
            mismatches.append({"row": i, "kind": "treatment_text_mismatch", "expected_prefix": expected_treat[:220], "actual_prefix": actual_treat[:220]})
        if actual_control != expected_control:
            mismatches.append({"row": i, "kind": "packet_local_text_mismatch", "expected_prefix": expected_control[:220], "actual_prefix": actual_control[:220]})
        src_norm = mat.normalize_ws(p.source_text)
        if not actual_treat.startswith(src_norm) or not actual_control.startswith(src_norm):
            prefix_fail.append({"row": i, "source_prefix": src_norm[:160], "treatment_prefix": actual_treat[:160], "packet_prefix": actual_control[:160]})
        source_vocab = set(p.source_text.split())
        if any(w not in source_vocab for w in actual_control.split()):
            bad = [w for w in actual_control.split() if w not in source_vocab][:20]
            control_vocab_fail.append({"row": i, "bad_words": bad})
        checked += 1
        if len(mismatches) >= 20 or len(prefix_fail) >= 20 or len(control_vocab_fail) >= 20:
            break
    return {
        "ok": not mismatches and not prefix_fail and not control_vocab_fail and checked == prefix_rows,
        "rows_checked": checked,
        "expected_rows": prefix_rows,
        "text_mismatch_count_sampled_until20": len(mismatches),
        "text_mismatch_samples": mismatches[:20],
        "prefix_fail_count_sampled_until20": len(prefix_fail),
        "prefix_fail_samples": prefix_fail[:20],
        "control_words_outside_source_vocab_count_sampled_until20": len(control_vocab_fail),
        "control_words_outside_source_vocab_samples": control_vocab_fail[:20],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", default=str(DEFAULT_DIR))
    ap.add_argument("--total-words", type=int, default=10_000_000)
    ap.add_argument("--passes", type=int, default=10)
    ap.add_argument("--tokenizer", default=str(TOKENIZER))
    ap.add_argument("--token-sample-limit", type=int, default=0, help="0 audits all pool rows; positive audits prefix only")
    args = ap.parse_args()
    d = pathlib.Path(args.corpus_dir)
    meta_path = d / "semantic_view_materialization_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    treat_path = d / "semantic_view_treatment_10M.jsonl"
    ctrl_path = d / "original_stream_matched_10M.jsonl"
    packet_local_path = d / "original_packet_local_10M.jsonl"
    packet_path = d / "original_packet_exact_matched_10M.jsonl"
    treat = read_rows(treat_path, args.total_words)
    ctrl = read_rows(ctrl_path, args.total_words)
    packet_local = read_rows(packet_local_path, args.total_words) if packet_local_path.exists() else []
    packet = read_rows(packet_path, args.total_words) if packet_path.exists() else []
    if [r["words"] for r in treat] != [r["words"] for r in ctrl]:
        raise RuntimeError("treatment/stream-control row length sequences differ")
    if packet_local and [r["words"] for r in treat] != [r["words"] for r in packet_local]:
        raise RuntimeError("treatment/packet-local row length sequences differ")
    if packet and [r["words"] for r in treat] != [r["words"] for r in packet]:
        raise RuntimeError("treatment/packet-exact row length sequences differ")
    prefix_rows = int(meta.get("semantic_packet_rows", 0))
    filler_compare = compare_after_prefix(treat, ctrl, prefix_rows)
    packet_local_filler_compare = compare_after_prefix(treat, packet_local, prefix_rows) if packet_local else {"identical_after_prefix": False, "reason": "missing_packet_local"}
    if not filler_compare["identical_after_prefix"]:
        raise RuntimeError(f"official filler mismatch after semantic prefix for stream control: {filler_compare}")
    if not packet_local_filler_compare["identical_after_prefix"]:
        raise RuntimeError(f"official filler mismatch after semantic prefix for packet-local control: {packet_local_filler_compare}")
    semantic_meta = read_meta_rows(d / "semantic_packet_rows_meta.jsonl")
    packet_local_meta = read_meta_rows(d / "packet_local_rows_meta.jsonl")
    packet_local_identity = compare_packet_identity(semantic_meta, packet_local_meta, prefix_rows) if packet_local else {"source_identity_and_row_fields_identical": False, "reason": "missing_packet_local"}
    if not packet_local_identity.get("source_identity_and_row_fields_identical", False):
        raise RuntimeError(f"packet-local control does not preserve semantic packet identity: {packet_local_identity}")
    prefix_construction = deep_prefix_construction_check(d, meta, treat, packet_local, semantic_meta, prefix_rows)
    if not prefix_construction.get("ok", False):
        raise RuntimeError(f"deep prefix construction check failed: {prefix_construction}")
    tok = AutoTokenizer.from_pretrained(str(pathlib.Path(args.tokenizer)), use_fast=True)
    report = {
        "status": "SEMANTIC_VIEW_CONTRAST_AUDITED",
        "corpus_dir": str(d),
        "metadata_path": str(meta_path),
        "materialization_status": meta.get("status"),
        "total_words_per_pool": args.total_words,
        "passes": args.passes,
        "semantic_packet_rows": prefix_rows,
        "semantic_packet_words": meta.get("semantic_packet_words"),
        "row_counts": {"treatment": len(treat), "stream_control": len(ctrl), "packet_local_control": len(packet_local) if packet_local else None, "packet_exact_control": len(packet) if packet else None},
        "row_length_sequence_identical": True,
        "official_filler_compare_stream_control": filler_compare,
        "official_filler_compare_packet_local_control": packet_local_filler_compare,
        "packet_local_identity_compare": packet_local_identity,
        "deep_prefix_construction_check": prefix_construction,
        "source_word_counts_treatment": source_words(treat),
        "source_word_counts_stream_control": source_words(ctrl),
        "source_word_counts_packet_local_control": source_words(packet_local) if packet_local else None,
        "source_word_counts_packet_exact_control": source_words(packet) if packet else None,
        "pool_sha256": {"treatment": sha256_file(treat_path), "stream_control": sha256_file(ctrl_path), "packet_local_control": sha256_file(packet_local_path) if packet_local_path.exists() else None, "packet_exact_control": sha256_file(packet_path) if packet_path.exists() else None},
        "tokenizer": str(pathlib.Path(args.tokenizer)),
        "tokenizer_vocab_size": len(tok),
        "token_audit_treatment": token_audit(treat, tok, args.token_sample_limit),
        "token_audit_stream_control": token_audit(ctrl, tok, args.token_sample_limit),
        "token_audit_packet_local_control": token_audit(packet_local, tok, args.token_sample_limit) if packet_local else None,
        "token_audit_packet_exact_control": token_audit(packet, tok, args.token_sample_limit) if packet else None,
        "token_audit_semantic_prefix_treatment": token_audit(treat[:prefix_rows], tok, 0),
        "token_audit_semantic_prefix_stream_control": token_audit(ctrl[:prefix_rows], tok, 0),
        "token_audit_semantic_prefix_packet_local_control": token_audit(packet_local[:prefix_rows], tok, 0) if packet_local else None,
        "training_files": {
            "semantic_view_treatment_100M": training_info(d / "semantic_view_treatment_100M.jsonl", args.total_words * args.passes),
            "original_stream_matched_100M": training_info(d / "original_stream_matched_100M.jsonl", args.total_words * args.passes),
            "original_packet_local_100M": training_info(d / "original_packet_local_100M.jsonl", args.total_words * args.passes),
            "original_packet_exact_matched_100M": training_info(d / "original_packet_exact_matched_100M.jsonl", args.total_words * args.passes),
        },
    }
    report["token_audit_delta_stream_control_minus_treatment"] = {
        "mean_tokens": round(report["token_audit_stream_control"]["token_length_stats"].get("mean", 0) - report["token_audit_treatment"]["token_length_stats"].get("mean", 0), 4),
        "rows_over_seq256": report["token_audit_stream_control"]["rows_over_seq256"] - report["token_audit_treatment"]["rows_over_seq256"],
        "excess_tokens": report["token_audit_stream_control"]["excess_tokens_if_truncated_at_256"] - report["token_audit_treatment"]["excess_tokens_if_truncated_at_256"],
    }
    if report["token_audit_packet_local_control"] is not None:
        report["token_audit_delta_packet_local_minus_treatment"] = {
            "mean_tokens": round(report["token_audit_packet_local_control"]["token_length_stats"].get("mean", 0) - report["token_audit_treatment"]["token_length_stats"].get("mean", 0), 4),
            "rows_over_seq256": report["token_audit_packet_local_control"]["rows_over_seq256"] - report["token_audit_treatment"]["rows_over_seq256"],
            "excess_tokens": report["token_audit_packet_local_control"]["excess_tokens_if_truncated_at_256"] - report["token_audit_treatment"]["excess_tokens_if_truncated_at_256"],
        }
    else:
        report["token_audit_delta_packet_local_minus_treatment"] = None
    out = d / "semantic_view_contrast_audit.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "corpus_dir": str(d),
        "semantic_packet_words": report["semantic_packet_words"],
        "row_counts": report["row_counts"],
        "filler_identical_after_prefix_stream": filler_compare["identical_after_prefix"],
        "filler_identical_after_prefix_packet_local": packet_local_filler_compare["identical_after_prefix"],
        "packet_local_identity": packet_local_identity.get("source_identity_and_row_fields_identical"),
        "token_delta_stream_minus_treatment": report["token_audit_delta_stream_control_minus_treatment"],
        "token_delta_packet_local_minus_treatment": report["token_audit_delta_packet_local_minus_treatment"],
        "training_files": {k: v.get("exists") for k, v in report["training_files"].items()},
        "out": str(out),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
