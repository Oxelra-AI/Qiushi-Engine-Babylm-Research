#!/usr/bin/env python3
"""research: ALN-preserving legal-tail overlay builder for relation-first packets.

This is a data-construction asset for the BabyLM bridge, not a training result.
It starts from the exact coherent86 remaining-tail accounting used in research and
materializes a tail stream in which repeated repaired relation-first packet rows
replace only non-`qwen_pair_packed` material.  The purpose is to let the next
training comparison test whether the acquired state-selection operation can be
installed while inherited ALN/Qwen-pair material and total legal exposure are
preserved.

Key scientific constraints preserved in the manifest:
  * parent is exact coherent86 at 86,005,295 charged words;
  * target tail word count is the row-level count selected from the same research
    suffix stream, not an invented budget;
  * all `qwen_pair_packed` rows inside that selected tail are preserved verbatim;
  * relation rows are built from research repaired train UPDATE/RETAIN rows only
    (neutral rows excluded), with answer spans recorded for future answer-focused
    masking;
  * non-Qwen tail words are displaced to make room for relation rows, with a
    split last filler row used only to match total charged words exactly.

The produced overlay is a candidate input to a future trainer that can apply
answer-span supervision on relation rows and ordinary WWM on ordinary rows.  It
is not by itself evidence that the intervention improves BabyLM competence.
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
import pathlib
import random
import time
from typing import Any, Dict, Iterable, List, Tuple

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
DEFAULT_BASE_STREAM = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl')
DEFAULT_REL_ROWS = _public_path('experiments/archive/functional_learning/data/relation_first_repaired/repaired_scoring_rows.jsonl')
DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/relation_tail_overlay')
DEFAULT_INITIAL_WORDS = 86_005_295
DEFAULT_FULL_CAP_WORDS = 100_000_000
DEFAULT_SKIP_ROWS = 556_791
DEFAULT_MAX_TAIL_WORDS = DEFAULT_FULL_CAP_WORDS - DEFAULT_INITIAL_WORDS


def rel(path: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: pathlib.Path, rows: Iterable[Dict[str, Any]]) -> Tuple[int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = words = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
            words += int(r.get("words", len(str(r.get("text", "")).split())))
    return n, words


def load_tail_rows(path: pathlib.Path, skip_rows: int, max_tail_words: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    selected = 0
    with path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < skip_rows:
                continue
            if not line.strip():
                continue
            obj = json.loads(line)
            text = str(obj["text"])
            words = int(obj.get("words", len(text.split())))
            if words != len(text.split()):
                raise RuntimeError(f"word mismatch in base stream row {idx}: declared={words} actual={len(text.split())}")
            if selected + words > max_tail_words:
                break
            out.append({
                "text": text,
                "words": words,
                "example_id": int(obj.get("example_id", -1)),
                "source": str(obj.get("source", "")),
                "orig_row_index": idx,
                "bridge_kind": "ordinary_tail",
            })
            selected += words
    return out


def full_relation_text(row: Dict[str, Any]) -> Tuple[str, int, int]:
    frame = str(row["use_sentence_frame"])
    answer = str(row["answer_text"])
    if frame.count("{STATE}") != 1:
        raise ValueError(f"bad frame for {row.get('pair_id')} {row.get('row_type')}: {frame[:120]!r}")
    before, after = frame.split("{STATE}")
    full = before + answer + after
    return full, len(before), len(before) + len(answer)


def relation_packet_rows(path: pathlib.Path, repeat_epochs: int, shuffle_each_epoch: bool, seed: int) -> List[Dict[str, Any]]:
    base = []
    for row in read_jsonl(path):
        if row.get("split") != "train" or row.get("role") == "NEUTRAL":
            continue
        text, a0, a1 = full_relation_text(row)
        words = len(text.split())
        if words <= 0:
            raise ValueError(f"empty relation row {row.get('pair_id')} {row.get('row_type')}")
        base.append({
            "text": text,
            "words": words,
            "example_id": -1,
            "source": "relation_first_repaired_packet",
            "bridge_kind": "relation_answer_packet",
            "pair_id": row.get("pair_id"),
            "relation": row.get("relation"),
            "row_type": row.get("row_type"),
            "role": row.get("role"),
            "query_orientation": row.get("query_orientation"),
            "updated_entity": row.get("updated_entity"),
            "query_entity": row.get("query_entity"),
            "entity_a": row.get("entity_a"),
            "entity_b": row.get("entity_b"),
            "value_a": row.get("value_a"),
            "value_b": row.get("value_b"),
            "shared_new_value": row.get("shared_new_value"),
            "answer_text": row.get("answer_text"),
            "foil_text": row.get("foil_text"),
            "answer_char_span": [a0, a1],
            "contract_version": row.get("contract_version"),
        })
    if not base:
        raise RuntimeError(f"No relation train rows loaded from {path}")
    out: List[Dict[str, Any]] = []
    for ep in range(1, int(repeat_epochs) + 1):
        rows = [dict(r) for r in base]
        if shuffle_each_epoch:
            random.Random(seed + ep).shuffle(rows)
        for j, r in enumerate(rows):
            r["relation_epoch"] = ep
            r["relation_epoch_index"] = j
            r["example_id"] = 9_000_000 + (ep - 1) * len(base) + j
            out.append(r)
    return out


def split_row_to_words(row: Dict[str, Any], take_words: int, part_id: int) -> Dict[str, Any]:
    words = str(row["text"]).split()
    if not (0 < take_words <= len(words)):
        raise ValueError((take_words, len(words)))
    q = dict(row)
    q["text"] = " ".join(words[:take_words])
    q["words"] = take_words
    q["source"] = str(row.get("source", "")) + f"::bridge_split_part{part_id}"
    q["bridge_split_from_orig_row_index"] = row.get("orig_row_index")
    q["bridge_split_original_words"] = len(words)
    return q


def choose_ordinary_fill(tail_rows: List[Dict[str, Any]], target_words: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Choose ordinary rows to fill target_words while preserving all qwen_pair rows.

    If all qwen_pair rows plus needed non-qwen exceed the target at row granularity,
    split the last selected non-qwen row to achieve exact word matching.
    """
    qwen = [r for r in tail_rows if r.get("source") == "qwen_pair_packed"]
    nonq = [r for r in tail_rows if r.get("source") != "qwen_pair_packed"]
    qwen_words = sum(int(r["words"]) for r in qwen)
    if qwen_words > target_words:
        raise RuntimeError(f"Cannot preserve qwen tail words {qwen_words} inside ordinary target {target_words}")
    needed_nonq = target_words - qwen_words
    selected_nonq: List[Dict[str, Any]] = []
    selected_nonq_words = 0
    split_used = None
    part_id = 0
    for r in nonq:
        w = int(r["words"])
        if selected_nonq_words + w < needed_nonq:
            selected_nonq.append(r)
            selected_nonq_words += w
        elif selected_nonq_words + w == needed_nonq:
            selected_nonq.append(r)
            selected_nonq_words += w
            break
        else:
            take = needed_nonq - selected_nonq_words
            if take > 0:
                selected_nonq.append(split_row_to_words(r, take, part_id))
                selected_nonq_words += take
                split_used = {"orig_row_index": r.get("orig_row_index"), "orig_words": w, "taken_words": take, "source": r.get("source")}
            break
    if selected_nonq_words != needed_nonq:
        raise RuntimeError(f"Could not fill ordinary target: needed_nonq={needed_nonq} selected={selected_nonq_words}")
    # Restore approximate original order among preserved rows by orig index. Relation rows
    # can later be interleaved by the trainer; here all ordinary rows retain their order.
    ordinary = sorted(qwen + selected_nonq, key=lambda r: int(r.get("orig_row_index", 10**12)))
    audit = {
        "ordinary_target_words": target_words,
        "preserved_qwen_pair_rows": len(qwen),
        "preserved_qwen_pair_words": qwen_words,
        "selected_nonq_rows": len(selected_nonq),
        "selected_nonq_words": selected_nonq_words,
        "split_used": split_used,
        "dropped_nonq_rows_from_reference_tail": len(nonq) - len([r for r in selected_nonq if "bridge_split_from_orig_row_index" not in r]) - (1 if split_used else 0),
        "dropped_nonq_words_from_reference_tail": sum(int(r["words"]) for r in nonq) - selected_nonq_words,
    }
    return ordinary, audit


def source_counts(rows: Iterable[Dict[str, Any]]) -> Tuple[Dict[str, int], Dict[str, int]]:
    rc, wc = collections.Counter(), collections.Counter()
    for r in rows:
        s = str(r.get("source", ""))
        rc[s] += 1
        wc[s] += int(r.get("words", len(str(r.get("text", "")).split())))
    return dict(rc), dict(wc)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-stream", default=str(DEFAULT_BASE_STREAM))
    ap.add_argument("--relation-rows", default=str(DEFAULT_REL_ROWS))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--skip-rows", type=int, default=DEFAULT_SKIP_ROWS)
    ap.add_argument("--initial-words", type=int, default=DEFAULT_INITIAL_WORDS)
    ap.add_argument("--full-cap-words", type=int, default=DEFAULT_FULL_CAP_WORDS)
    ap.add_argument("--max-tail-words", type=int, default=DEFAULT_MAX_TAIL_WORDS)
    ap.add_argument("--relation-repeat-epochs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=43043)
    ap.add_argument("--shuffle-relation-each-epoch", action="store_true")
    ap.add_argument("--write-jsonl", action="store_true")
    args = ap.parse_args()

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base_stream = pathlib.Path(args.base_stream)
    rel_rows_path = pathlib.Path(args.relation_rows)

    tail = load_tail_rows(base_stream, int(args.skip_rows), int(args.max_tail_words))
    tail_words = sum(int(r["words"]) for r in tail)
    relation = relation_packet_rows(rel_rows_path, int(args.relation_repeat_epochs), bool(args.shuffle_relation_each_epoch), int(args.seed))
    relation_words = sum(int(r["words"]) for r in relation)
    if relation_words >= tail_words:
        raise RuntimeError(f"relation_words {relation_words} >= tail_words {tail_words}")
    ordinary_target = tail_words - relation_words
    ordinary, ordinary_audit = choose_ordinary_fill(tail, ordinary_target)

    # Put relation rows first for transparent construction. A future trainer can choose
    # its own interleaving schedule from bridge_kind/source fields.
    overlay = relation + ordinary
    overlay_words = sum(int(r["words"]) for r in overlay)
    if overlay_words != tail_words:
        raise RuntimeError(f"overlay words {overlay_words} != reference tail words {tail_words}")

    tail_rc, tail_wc = source_counts(tail)
    ov_rc, ov_wc = source_counts(overlay)
    rel_rc, rel_wc = source_counts(relation)
    manifest = {
        "status": "RELATION_TAIL_OVERLAY_BUILT",
        "created_utc": now_utc(),
        "scientific_purpose": "Prepare an ALN-preserving legal-tail corpus in which repaired relation-first packets replace only non-qwen filler-like tail material for a future coherent86 continuation comparison.",
        "inputs": {
            "base_stream": rel(base_stream),
            "base_stream_sha256": sha256_file(base_stream) if base_stream.exists() else None,
            "relation_rows": rel(rel_rows_path),
            "relation_rows_sha256": sha256_file(rel_rows_path) if rel_rows_path.exists() else None,
        },
        "accounting": {
            "initial_consumed_words": int(args.initial_words),
            "full_cap_words": int(args.full_cap_words),
            "nominal_max_tail_words": int(args.max_tail_words),
            "skip_rows": int(args.skip_rows),
            "reference_tail_rows": len(tail),
            "reference_tail_words_row_level": tail_words,
            "final_total_words_if_trained_from_parent": int(args.initial_words) + tail_words,
            "relation_repeat_epochs": int(args.relation_repeat_epochs),
            "relation_packet_rows": len(relation),
            "relation_packet_words": relation_words,
            "relation_fraction_of_tail_words": relation_words / tail_words,
            "ordinary_tail_rows_in_overlay": len(ordinary),
            "ordinary_tail_words_in_overlay": ordinary_target,
            "overlay_rows": len(overlay),
            "overlay_words": overlay_words,
            "exact_word_match_to_reference_tail": overlay_words == tail_words,
        },
        "ordinary_preservation_audit": ordinary_audit,
        "reference_tail_source_rows": tail_rc,
        "reference_tail_source_words": tail_wc,
        "relation_source_rows": rel_rc,
        "relation_source_words": rel_wc,
        "overlay_source_rows": ov_rc,
        "overlay_source_words": ov_wc,
        "bridge_training_note": "Use bridge_kind=relation_answer_packet rows for explicit answer-span masking using answer_char_span; ordinary_tail rows should receive ordinary WWM or preservation objective. This construction alone is not a training result.",
    }
    (out / "overlay_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    relation_out = out / ("relation_packet_train_rows_e%03d.jsonl" % int(args.relation_repeat_epochs))
    write_jsonl(relation_out, relation)
    if args.write_jsonl:
        overlay_out = out / ("overlay_tail_relation_e%03d.jsonl" % int(args.relation_repeat_epochs))
        n, w = write_jsonl(overlay_out, overlay)
        manifest["outputs"] = {
            "relation_rows": rel(relation_out),
            "overlay_tail": rel(overlay_out),
            "overlay_written_rows": n,
            "overlay_written_words": w,
        }
        (out / "overlay_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": manifest["status"],
        "manifest": rel(out / "overlay_manifest.json"),
        "tail_words": tail_words,
        "relation_words": relation_words,
        "relation_fraction": relation_words / tail_words,
        "preserved_qwen_pair_words": ordinary_audit["preserved_qwen_pair_words"],
        "overlay_written": bool(args.write_jsonl),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
