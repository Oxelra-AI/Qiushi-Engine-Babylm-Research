#!/usr/bin/env python3
"""Build explicit cross-view masking JSONLs from research mixture materialization.

Matched control: anchors/masked side are determined once
from the original true pair, then copied unchanged to the shuffled arm. The
shuffled arm only permutes the visible other side. Thus predicted source-side
word groups are identical per pair example in adjacent and shuffled conditions.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
from collections import Counter
from typing import Any

import pyarrow.parquet as pq
from transformers import AutoTokenizer

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
research = ROOT / "data/mixture_revision_61/mixture_materialization_all_seeds.json"
OUT_DIR = ROOT / "data/crossview_revision_64"
BASELINE_TOKENIZER_REPO = "BabyLM-community/BabyLM-2026-Baseline-GPT2-Strict"
STOP = set("""
a an the and or but if while of in on at by for from to with without into onto over under before after during as is are was were be been being am do does did have has had this that these those it its his her their our your my i you he she they we not no yes than then there here who whom whose which what when where why how
""".split())
WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")


def norm_token(w: str) -> str:
    m = WORD_RE.search(w)
    return m.group(0).lower().strip("'-") if m else ""


def is_anchor_word(raw: str, norm: str) -> bool:
    if not norm or norm in STOP:
        return False
    if any(ch.isdigit() for ch in norm):
        return True
    if raw[:1].isupper() and len(norm) >= 2:
        return True
    if len(norm) >= 5:
        return True
    return False


def anchors_from_true_pair(source: str, target: str, max_anchors: int = 8) -> dict[str, Any]:
    src_words = source.split()
    tgt_norms = {norm_token(w) for w in target.split()}
    tgt_norms.discard("")
    idxs, norms = [], []
    seen = set()
    for i, raw in enumerate(src_words):
        n = norm_token(raw)
        if n in tgt_norms and is_anchor_word(raw, n) and n not in seen:
            idxs.append(i); norms.append(n); seen.add(n)
        if len(idxs) >= max_anchors:
            break
    return {"mask_side": "source", "anchor_source_word_indices": idxs, "anchor_norms": norms}


def load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def token_ids(tok, text: str) -> list[int]:
    return tok(text, add_special_tokens=False, truncation=False)["input_ids"]


def source_anchor_token_count(tok, source_text: str, anchor_indices: list[int]) -> int:
    words = source_text.split()
    total = 0
    for i in anchor_indices:
        if 0 <= i < len(words):
            # GPT2 byte-level BPE tokenization depends on leading-space status.
            # Counting isolated pieces is only a supervision-budget proxy; the
            # trainer validates real group equality from source-side groups.
            piece = words[i] if i == 0 else " " + words[i]
            total += len(token_ids(tok, piece))
    return total


def explicit_pair_record(base: dict[str, Any], gem_rows: list[dict[str, Any]], target_pair_id: int, mode: str, true_anchor: dict[str, Any], tok) -> dict[str, Any]:
    sid = int(base["source_pair_id"])
    tid = int(target_pair_id)
    source_text = " ".join(str(gem_rows[sid]["source"]).replace("\n", " ").replace("[SEP]", "SEP").split())
    visible_target = " ".join(str(gem_rows[tid]["target"]).replace("\n", " ").replace("[SEP]", "SEP").split())
    true_target = " ".join(str(gem_rows[sid]["target"]).replace("\n", " ").replace("[SEP]", "SEP").split())
    text = source_text + " " + visible_target
    rec = {
        "example_id": int(base["example_id"]),
        "kind": "pair_crossview",
        "source": f"crossview_{mode}",
        "mode": mode,
        "text": text,
        "words": len(text.split()),
        "source_text": source_text,
        "target_text": visible_target,
        "true_target_text": true_target,
        "source_pair_id": sid,
        "target_pair_id": tid,
        "true_target_pair_id": sid,
        "mask_side": true_anchor["mask_side"],
        "anchor_source_word_indices": true_anchor["anchor_source_word_indices"],
        "anchor_norms": true_anchor["anchor_norms"],
        "anchor_count": len(true_anchor["anchor_source_word_indices"]),
        "anchor_token_count_proxy": source_anchor_token_count(tok, source_text, true_anchor["anchor_source_word_indices"]),
        "visible_other_side_is_true_pair": tid == sid,
    }
    return rec


def convert_seed(seed: str, meta: dict[str, Any], gem_rows: list[dict[str, Any]], tok, smoke_words: int) -> dict[str, Any]:
    sm = meta["seeds"][seed]
    adj_in = load_jsonl(pathlib.Path(sm["mixture_adjacent_path"]))
    shuf_in = load_jsonl(pathlib.Path(sm["mixture_shuffled_path"]))
    if len(adj_in) != len(shuf_in):
        raise RuntimeError(f"seed {seed}: arm length mismatch")
    full_adj, full_shuf = [], []
    pair_supervision_pairs = []
    for a, s in zip(adj_in, shuf_in):
        if int(a["example_id"]) != int(s["example_id"]):
            raise RuntimeError(f"seed {seed}: example_id mismatch")
        if a["kind"] == "official":
            for src, dst, mode in [(a, full_adj, "adjacent"), (s, full_shuf, "shuffled")]:
                dst.append({
                    "example_id": int(src["example_id"]),
                    "kind": "official",
                    "source": src["source"],
                    "mode": mode,
                    "text": src["text"],
                    "words": int(src["words"]),
                    "official_example_id": int(src["official_example_id"]),
                    "official_source_file": src["official_source_file"],
                })
        else:
            sid = int(a["source_pair_id"])
            if sid != int(s["source_pair_id"]):
                raise RuntimeError(f"seed {seed}: source pair mismatch")
            source_text = " ".join(str(gem_rows[sid]["source"]).replace("\n", " ").replace("[SEP]", "SEP").split())
            true_target = " ".join(str(gem_rows[sid]["target"]).replace("\n", " ").replace("[SEP]", "SEP").split())
            anchor = anchors_from_true_pair(source_text, true_target)
            ra = explicit_pair_record(a, gem_rows, int(a["target_pair_id"]), "adjacent", anchor, tok)
            rs = explicit_pair_record(s, gem_rows, int(s["target_pair_id"]), "shuffled", anchor, tok)
            if ra["anchor_source_word_indices"] != rs["anchor_source_word_indices"] or ra["anchor_norms"] != rs["anchor_norms"]:
                raise RuntimeError("anchor mismatch after copy")
            full_adj.append(ra); full_shuf.append(rs)
            pair_supervision_pairs.append((ra["anchor_count"], ra["anchor_token_count_proxy"], rs["anchor_count"], rs["anchor_token_count_proxy"]))
    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        pair = [r for r in rows if r["kind"] == "pair_crossview"]
        official = [r for r in rows if r["kind"] == "official"]
        over = [r["example_id"] for r in rows if len(token_ids(tok, r["text"])) > 256]
        return {
            "num_examples": len(rows),
            "total_words": sum(int(r["words"]) for r in rows),
            "official_examples": len(official),
            "pair_examples": len(pair),
            "pair_examples_with_anchors": sum(1 for r in pair if r["anchor_count"] > 0),
            "pair_anchor_groups_total": sum(int(r.get("anchor_count", 0)) for r in pair),
            "pair_anchor_token_proxy_total": sum(int(r.get("anchor_token_count_proxy", 0)) for r in pair),
            "over_256_examples": len(over),
            "over_256_first20": over[:20],
        }
    full_adj_path = OUT_DIR / f"crossview_full_adjacent_seed{seed}.jsonl"
    full_shuf_path = OUT_DIR / f"crossview_full_shuffled_seed{seed}.jsonl"
    write_jsonl(full_adj_path, full_adj); write_jsonl(full_shuf_path, full_shuf)
    # Smoke subset: choose same example positions with equal per-example words to keep arm totals equal.
    smoke_adj, smoke_shuf, total = [], [], 0
    for ra, rs in zip(full_adj, full_shuf):
        if int(ra["words"]) != int(rs["words"]):
            continue
        smoke_adj.append(ra); smoke_shuf.append(rs); total += int(ra["words"])
        if total >= smoke_words:
            break
    smoke_adj_path = OUT_DIR / f"crossview_smoke_adjacent_seed{seed}_{total}w.jsonl"
    smoke_shuf_path = OUT_DIR / f"crossview_smoke_shuffled_seed{seed}_{total}w.jsonl"
    write_jsonl(smoke_adj_path, smoke_adj); write_jsonl(smoke_shuf_path, smoke_shuf)
    sup_ok = all(a == c and b == d for a, b, c, d in pair_supervision_pairs)
    smoke_sup_ok = all(int(a.get("anchor_count", 0)) == int(s.get("anchor_count", 0)) and int(a.get("anchor_token_count_proxy", 0)) == int(s.get("anchor_token_count_proxy", 0)) for a, s in zip(smoke_adj, smoke_shuf) if a["kind"] == "pair_crossview")
    return {
        "seed": int(seed),
        "full_adjacent_path": str(full_adj_path),
        "full_shuffled_path": str(full_shuf_path),
        "smoke_adjacent_path": str(smoke_adj_path),
        "smoke_shuffled_path": str(smoke_shuf_path),
        "full_adjacent_summary": summarize(full_adj),
        "full_shuffled_summary": summarize(full_shuf),
        "smoke_adjacent_summary": summarize(smoke_adj),
        "smoke_shuffled_summary": summarize(smoke_shuf),
        "per_pair_supervision_proxy_identical_full": sup_ok,
        "per_pair_supervision_proxy_identical_smoke": smoke_sup_ok,
        "anchor_count_histogram_full": dict(Counter(r.get("anchor_count", 0) for r in full_adj if r["kind"] == "pair_crossview")),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="42")
    p.add_argument("--smoke_words", type=int, default=10000)
    args = p.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = json.loads(research.read_text(encoding="utf-8"))
    gem_path = pathlib.Path(meta["gem_dataset_file_cache_path"])
    gem_rows = pq.read_table(gem_path).to_pylist()
    tok = AutoTokenizer.from_pretrained(BASELINE_TOKENIZER_REPO, revision="main", use_fast=True)
    out = {
        "source_step61_meta": str(research),
        "control_statement": "mask_side and anchor_source_word_indices are computed from the true source-target pair once, copied unchanged to shuffled; shuffled only changes visible target_text.",
        "seeds": {},
    }
    for seed in [s.strip() for s in args.seeds.split(",") if s.strip()]:
        out["seeds"][seed] = convert_seed(seed, meta, gem_rows, tok, args.smoke_words)
    meta_path = OUT_DIR / "crossview_materialization_revision_64.json"
    meta_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "CROSSVIEW_JSONL_READY", "meta_path": str(meta_path), "seeds": out["seeds"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
