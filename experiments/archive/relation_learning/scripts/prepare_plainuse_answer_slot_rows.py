#!/usr/bin/env python3
"""research: convert validated plain-use packets into answer-slot rows.

The research packet files are single natural packets rather than paired UPDATE/RETAIN
rows.  This script creates a compact training/evaluation schema for the missing
acquisition cell: source sentence + update sentence + final use sentence with exactly
one supervised state slot.  Source and update tokens remain visible; only the final
state phrase is replaced by masks in training.

For UPDATED_USE, answer_text is the new_state phrase expressed in the final use.
For UNCHANGED_DISTRACTOR_USE, answer_text is the source_state phrase expressed in the
final use; foil_text is the competing state phrase.  Rows are kept only when an exact
case-insensitive phrase occurrence can be found in the final use sentence and replaced
by {STATE}.  This preserves an explicit final-span contract for answer-only
harnesses rather than relying on token search elsewhere in the context.
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
import re
import time
from typing import Any

ROOT0 = _public_path('experiments/archive/relation_learning/scripts/prepare_plainuse_answer_slot_rows.py')
ROOT = _PUBLIC_ROOT

DATA = ROOT / "experiments/archive/relation_learning/data/state_use_generation"
TRAIN = DATA / "validated_state_use_packets_plainuse_full_step047_train.jsonl"
HELD = DATA / "validated_state_use_packets_plainuse_full_step047_heldout.jsonl"
OUT_DEFAULT = ROOT / "experiments/archive/relation_learning/data/plainuse_answer_slot_rows"
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[\u2019'][A-Za-z0-9]+)?")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def norm_space(s: str) -> str:
    return " ".join(str(s or "").split())


def exact_phrase_frame(use_sentence: str, phrase: str) -> tuple[str, int, int] | None:
    use = norm_space(use_sentence)
    ph = norm_space(phrase)
    if not use or not ph:
        return None
    pattern = re.compile(r"(?<![A-Za-z0-9])" + re.escape(ph) + r"(?![A-Za-z0-9])", flags=re.IGNORECASE)
    hits = list(pattern.finditer(use))
    if len(hits) != 1:
        return None
    m = hits[0]
    frame = use[:m.start()] + "{STATE}" + use[m.end():]
    return frame, m.start(), m.end()


def content_tokens(s: str) -> set[str]:
    return {m.group(0).lower() for m in WORD_RE.finditer(s) if len(m.group(0)) >= 3}


def choose_answer_phrase(packet: dict[str, Any]) -> tuple[str, list[str], str] | None:
    ptype = str(packet.get("packet_type"))
    if ptype == "UPDATED_USE":
        terms = [str(x) for x in packet.get("use_new_state_hits") or []]
        full = str(packet.get("new_state", ""))
        fallback = [str(x) for x in packet.get("new_specific_terms") or []]
        kind = "new_state"
    elif ptype == "UNCHANGED_DISTRACTOR_USE":
        terms = [str(x) for x in packet.get("use_source_state_hits") or []]
        full = str(packet.get("source_state", ""))
        fallback = [str(x) for x in packet.get("source_specific_terms") or []]
        kind = "source_state"
    else:
        return None
    candidates: list[str] = []
    if full:
        candidates.append(full)
    # Prefer longest lexical hits present in the final use if the full state phrase is paraphrased.
    for t in sorted(set(terms + fallback), key=lambda x: (-len(x.split()), -len(x), x)):
        if len(t.strip()) >= 3 and t.strip().lower() not in {"the", "and", "with"}:
            candidates.append(t)
    seen = set()
    uniq = []
    for c in candidates:
        k = norm_space(c).lower()
        if k and k not in seen:
            seen.add(k); uniq.append(norm_space(c))
    return kind, uniq, full


def convert_packet(packet: dict[str, Any], split: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    ptype = str(packet.get("packet_type"))
    ans_info = choose_answer_phrase(packet)
    if ans_info is None:
        return None, {"pair_id": packet.get("pair_id"), "split": split, "reason": "unknown_packet_type", "packet_type": ptype}
    answer_kind, candidates, declared_full = ans_info
    use = norm_space(str(packet.get("use_sentence", "")))
    selected = None
    for cand in candidates:
        fr = exact_phrase_frame(use, cand)
        if fr is not None:
            selected = (cand, *fr)
            break
    if selected is None:
        return None, {
            "pair_id": packet.get("pair_id"), "split": split, "packet_type": ptype,
            "reason": "no_unique_answer_phrase_occurrence_in_use", "use_sentence": use,
            "candidate_phrases": candidates[:8], "declared_full_state": declared_full,
        }
    answer, frame, start, end = selected
    foil = norm_space(str(packet.get("source_state" if ptype == "UPDATED_USE" else "new_state", "")))
    if not foil:
        return None, {"pair_id": packet.get("pair_id"), "split": split, "packet_type": ptype, "reason": "empty_foil"}
    # Avoid useless rows where answer and foil share all content words.
    ac = content_tokens(answer)
    fc = content_tokens(foil)
    if ac and fc and ac == fc:
        return None, {"pair_id": packet.get("pair_id"), "split": split, "packet_type": ptype, "reason": "answer_foil_same_content", "answer": answer, "foil": foil}
    row = {
        "row_id": f"{split}:{packet.get('pair_id')}:{ptype}",
        "pair_id": str(packet.get("pair_id")),
        "split": split,
        "packet_type": ptype,
        "source_sentence": norm_space(str(packet.get("source_sentence", ""))),
        "update_sentence": norm_space(str(packet.get("update_sentence", ""))),
        "use_sentence": use,
        "use_sentence_frame": frame,
        "answer_text": answer,
        "foil_text": foil,
        "answer_state_kind": answer_kind,
        "foil_state_kind": "source_state" if answer_kind == "new_state" else "new_state",
        "target_entity": packet.get("target_entity"),
        "updated_entity": packet.get("updated_entity"),
        "source_state": packet.get("source_state"),
        "new_state": packet.get("new_state"),
        "answer_char_start_in_use": start,
        "answer_char_end_in_use": end,
        "source_specific_terms": packet.get("source_specific_terms"),
        "new_specific_terms": packet.get("new_specific_terms"),
        "use_new_state_hits": packet.get("use_new_state_hits"),
        "use_source_state_hits": packet.get("use_source_state_hits"),
        "plainuse_cue_hits": packet.get("plainuse_cue_hits"),
        "original_generation_source": packet.get("generation_source"),
        "training_contract": "source+update visible; final answer phrase only is supervised/masked",
    }
    return row, None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--train", default=str(TRAIN))
    ap.add_argument("--heldout", default=str(HELD))
    args = ap.parse_args()
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    srcs = {"train": pathlib.Path(args.train), "heldout": pathlib.Path(args.heldout)}
    split_packet_counts = {}
    for split, path in srcs.items():
        packets = read_jsonl(path)
        split_packet_counts[split] = len(packets)
        split_rows: list[dict[str, Any]] = []
        for packet in packets:
            row, rej = convert_packet(packet, split)
            if row is not None:
                split_rows.append(row)
                all_rows.append(row)
            if rej is not None:
                rejects.append(rej)
        write_jsonl(out / f"answer_slot_rows_{split}.jsonl", split_rows)
    write_jsonl(out / "answer_slot_rows_all.jsonl", all_rows)
    write_jsonl(out / "answer_slot_rejects.jsonl", rejects)
    counts = collections.Counter((r["split"], r["packet_type"]) for r in all_rows)
    rej_counts = collections.Counter(r["reason"] for r in rejects)
    summary = {
        "status": "PLAINUSE_ANSWER_SLOT_ROWS_DONE",
        "created_utc": now(),
        "source_paths": {k: rel(v) for k, v in srcs.items()},
        "source_sha256": {k: sha256_file(v) for k, v in srcs.items()},
        "source_packet_counts": split_packet_counts,
        "rows_all": len(all_rows),
        "rows_by_split_type": {f"{k[0]}:{k[1]}": v for k, v in sorted(counts.items())},
        "rejects": len(rejects),
        "reject_reasons": dict(rej_counts),
        "paths": {
            "all": rel(out / "answer_slot_rows_all.jsonl"),
            "train": rel(out / "answer_slot_rows_train.jsonl"),
            "heldout": rel(out / "answer_slot_rows_heldout.jsonl"),
            "rejects": rel(out / "answer_slot_rejects.jsonl"),
        },
        "row_schema": "source_sentence, update_sentence, use_sentence_frame with one {STATE}, answer_text, foil_text; final answer span only should be masked/labeled.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
