#!/usr/bin/env python3
"""Audit research frozen stratified local-response events.

Checks span reconstruction, tokenizer/input consistency, category agreement with the
research-compatible definitions in roberta_pair_stratified_response_probe.py,
feature/bin agreement with the research atlas, and basic special-token safety.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path
from typing import Any


def find_user_root() -> Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
PROBE = WS / "scripts/roberta_pair_stratified_response_probe.py"
DEFAULT_EVENTS = WS / "data/roberta_pair_stratified_response_probe/frozen_events.jsonl"
DEFAULT_MANIFEST = WS / "data/roberta_pair_stratified_response_probe/stratum_manifest.json"
DEFAULT_OUT = WS / "data/roberta_pair_stratified_event_audit"
DEFAULT_TOKENIZER = WS / "data/compliant_tokenizer"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_probe_module():
    spec = importlib.util.spec_from_file_location("probe", PROBE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {PROBE}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def close(a: Any, b: Any, tol: float = 1e-12) -> bool:
    try:
        return math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a) - float(b)) <= tol
    except Exception:
        return a == b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--max_examples", type=int, default=12)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    mod = load_probe_module()
    tok = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True, use_fast=True)
    pairs = mod.read_pairs()
    atlas = mod.read_atlas()
    thresholds = mod.build_thresholds(atlas)
    events = read_jsonl(args.events)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8")) if args.manifest.exists() else None
    special = set(tok.all_special_ids)

    counts = {
        "events": len(events),
        "missing_pair": 0,
        "missing_atlas": 0,
        "input_ids_mismatch": 0,
        "span_mismatch": 0,
        "decoded_span_norm_mismatch": 0,
        "category_mismatch": 0,
        "feature_mismatch": 0,
        "feature_bin_mismatch": 0,
        "special_token_in_target_span": 0,
        "bad_span": 0,
    }
    examples: dict[str, list[dict[str, Any]]] = {k: [] for k in counts if k != "events"}

    for e in events:
        pid = e.get("pair_id")
        if pid not in pairs:
            counts["missing_pair"] += 1
            if len(examples["missing_pair"]) < args.max_examples:
                examples["missing_pair"].append({"event_index": e.get("event_index"), "pair_id": pid})
            continue
        if pid not in atlas:
            counts["missing_atlas"] += 1
            if len(examples["missing_atlas"]) < args.max_examples:
                examples["missing_atlas"].append({"event_index": e.get("event_index"), "pair_id": pid})
            continue
        row = pairs[pid]
        a = atlas[pid]
        src = str(row["source_text"])
        side = mod.side_text_for(row, e["view_type"])
        side_words = side.split()
        recomputed_ids = tok(src + " " + side, add_special_tokens=False, truncation=True, max_length=mod.SEQ_LEN)["input_ids"]
        if [int(x) for x in e.get("input_ids", [])] != [int(x) for x in recomputed_ids]:
            counts["input_ids_mismatch"] += 1
            if len(examples["input_ids_mismatch"]) < args.max_examples:
                examples["input_ids_mismatch"].append({"event_index": e.get("event_index"), "pair_id": pid, "view_type": e.get("view_type")})
        wi = int(e["word_index"])
        recomputed_span = mod.target_span(tok, src, side_words, wi)
        span = tuple(int(z) for z in e["span"])
        if recomputed_span != span:
            counts["span_mismatch"] += 1
            if len(examples["span_mismatch"]) < args.max_examples:
                examples["span_mismatch"].append({"event_index": e.get("event_index"), "pair_id": pid, "span": span, "recomputed_span": recomputed_span})
        st, en = span
        ids = [int(x) for x in e.get("input_ids", [])]
        if not (0 <= st < en <= len(ids)):
            counts["bad_span"] += 1
            if len(examples["bad_span"]) < args.max_examples:
                examples["bad_span"].append({"event_index": e.get("event_index"), "pair_id": pid, "span": span, "input_len": len(ids)})
            continue
        if any(t in special for t in ids[st:en]):
            counts["special_token_in_target_span"] += 1
            if len(examples["special_token_in_target_span"]) < args.max_examples:
                examples["special_token_in_target_span"].append({"event_index": e.get("event_index"), "pair_id": pid, "span": span})
        decoded_norm = mod.norm(tok.decode(ids[st:en], clean_up_tokenization_spaces=False))
        event_norm = mod.norm(str(e.get("word", "")))
        if decoded_norm != event_norm:
            counts["decoded_span_norm_mismatch"] += 1
            if len(examples["decoded_span_norm_mismatch"]) < args.max_examples:
                examples["decoded_span_norm_mismatch"].append({"event_index": e.get("event_index"), "pair_id": pid, "word": e.get("word"), "decoded": tok.decode(ids[st:en], clean_up_tokenization_spaces=False), "decoded_norm": decoded_norm, "event_norm": event_norm})
        source_norms = {mod.norm(w) for w in src.split() if mod.norm(w)}
        recomputed_cat = mod.category_for_word(str(e.get("word")), source_norms, str(e.get("view_type")))
        if recomputed_cat != e.get("category"):
            counts["category_mismatch"] += 1
            if len(examples["category_mismatch"]) < args.max_examples:
                examples["category_mismatch"].append({"event_index": e.get("event_index"), "pair_id": pid, "word": e.get("word"), "category": e.get("category"), "recomputed": recomputed_cat})
        bins = mod.assign_bins(a, thresholds)
        for feature in mod.FEATURES:
            if not close(e.get("features", {}).get(feature), a.get(feature)):
                counts["feature_mismatch"] += 1
                if len(examples["feature_mismatch"]) < args.max_examples:
                    examples["feature_mismatch"].append({"event_index": e.get("event_index"), "pair_id": pid, "feature": feature, "event": e.get("features", {}).get(feature), "atlas": a.get(feature)})
                break
        for feature, b in bins.items():
            if e.get("feature_bins", {}).get(feature) != b:
                counts["feature_bin_mismatch"] += 1
                if len(examples["feature_bin_mismatch"]) < args.max_examples:
                    examples["feature_bin_mismatch"].append({"event_index": e.get("event_index"), "pair_id": pid, "feature": feature, "event": e.get("feature_bins", {}).get(feature), "recomputed": b})
                break

    problems = {k: v for k, v in counts.items() if k != "events" and v}
    payload = {
        "status": "STRATIFIED_EVENT_AUDIT",
        "created_utc": now(),
        "events_path": str(args.events),
        "events_sha256": sha256_file(args.events),
        "manifest_sha256": manifest.get("frozen_events_sha256") if manifest else None,
        "event_sha_matches_manifest": (manifest is not None and sha256_file(args.events) == manifest.get("frozen_events_sha256")),
        "tokenizer_json_sha256": sha256_file(args.tokenizer / "tokenizer.json"),
        "counts": counts,
        "problems": problems,
        "examples": {k: v for k, v in examples.items() if v},
        "mechanically_sound": len(problems) == 0 and (manifest is not None and sha256_file(args.events) == manifest.get("frozen_events_sha256")),
        "meaning": "Span/input/category/feature agreement audit for the frozen event set; it does not train, evaluate official tasks, upload, or submit.",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "event_audit.json", payload)
    md = [
        "# research stratified-event audit",
        "",
        f"Created: `{payload['created_utc']}`",
        f"Events: `{payload['counts']['events']}`",
        f"Event SHA matches manifest: `{payload['event_sha_matches_manifest']}`",
        f"Mechanically sound: `{payload['mechanically_sound']}`",
        "",
        "Problem counts:",
    ]
    for k, v in counts.items():
        if k != "events":
            md.append(f"- `{k}`: {v}")
    (args.out_dir / "event_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "mechanically_sound": payload["mechanically_sound"], "problems": problems, "out": str(args.out_dir / "event_audit.json")}, indent=2), flush=True)
    if not payload["mechanically_sound"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
