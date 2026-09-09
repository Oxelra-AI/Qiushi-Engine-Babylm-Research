#!/usr/bin/env python3
"""research: tokenizer shift on official BabyLM evaluation texts.

CPU-only diagnostic: compare the old 100M-trained baseline16k tokenizer and the
new 10M-trained compliant16k tokenizer on the official evaluation text surface.
It does not run model inference.  It estimates how much tokenizer replacement can
change scoring/finetuning geometry before the compliant retrain endpoints exist.
"""
from __future__ import annotations

import collections
import csv
import json
import pathlib
import statistics
import time
from typing import Any, Iterable

from transformers import AutoTokenizer

USER_ROOT = pathlib.Path(".").resolve()
STUDY = USER_ROOT / "experiments/archive/frontier_consolidation"
OUT = STUDY / "data/eval_text_tokenizer_shift"
OLD = USER_ROOT / "experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model"
NEW = STUDY / "data/compliant_tokenizer"
PRISTINE_FULL = USER_ROOT / "experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval"
GLOBALPIQA_FULL = USER_ROOT / "experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval"

SEQ_SENTENCE = 256
SEQ_FINETUNE = 512


def q(vals: list[float], p: float) -> float:
    if not vals:
        return float("nan")
    vals = sorted(vals)
    return float(vals[min(len(vals) - 1, max(0, round((len(vals) - 1) * p)))])


def summarize(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "mean": float(sum(vals) / len(vals)),
        "std": float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0,
        "min": float(min(vals)),
        "p05": q(vals, 0.05),
        "p50": q(vals, 0.50),
        "p95": q(vals, 0.95),
        "max": float(max(vals)),
    }


def iter_jsonl(path: pathlib.Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def token_len(tok, text: str) -> int:
    return len(tok(str(text), add_special_tokens=False, truncation=False)["input_ids"])


def collect_records() -> Iterable[dict[str, Any]]:
    # BLiMP and Supplement: lexical/syntactic minimal-pair sentences.
    for family, dname in [("BLiMP", "blimp_filtered"), ("Supplement", "supplement_filtered")]:
        d = PRISTINE_FULL / dname
        for p in sorted(d.glob("*.jsonl")):
            sub = p.stem
            for i, obj in enumerate(iter_jsonl(p)):
                for role, key in [("good", "sentence_good"), ("bad", "sentence_bad")]:
                    if key in obj:
                        yield {"family": family, "subtask": sub, "split": "full_eval", "role": role, "row": i, "text": str(obj[key])}

    # EWoK: direct contexts and targets are the scored relational text pieces.
    d = PRISTINE_FULL / "ewok_filtered"
    for p in sorted(d.glob("*.jsonl")):
        sub = p.stem
        for i, obj in enumerate(iter_jsonl(p)):
            for key in ["Context1", "Context2", "Target1", "Target2"]:
                if key in obj:
                    yield {"family": "EWoK", "subtask": sub, "split": "full_eval", "role": key, "row": i, "text": str(obj[key])}
            # Also approximate the scored concatenated alternatives.
            if all(k in obj for k in ["Context1", "Target1", "Context2", "Target2"]):
                yield {"family": "EWoK_concat", "subtask": sub, "split": "full_eval", "role": "ctx1_target1", "row": i, "text": str(obj["Context1"]) + " " + str(obj["Target1"])}
                yield {"family": "EWoK_concat", "subtask": sub, "split": "full_eval", "role": "ctx2_target2", "row": i, "text": str(obj["Context2"]) + " " + str(obj["Target2"])}

    # Entity Tracking: input prefix plus options determine pseudo-likelihood choice.
    d = PRISTINE_FULL / "entity_tracking"
    for p in sorted(d.glob("*.jsonl")):
        sub = p.stem
        for i, obj in enumerate(iter_jsonl(p)):
            prefix = str(obj.get("input_prefix", ""))
            yield {"family": "Entity", "subtask": sub, "split": "full_eval", "role": "input_prefix", "row": i, "text": prefix}
            for j, opt in enumerate(obj.get("options", [])):
                yield {"family": "Entity", "subtask": sub, "split": "full_eval", "role": f"prefix_option{j}", "row": i, "text": prefix + str(opt)}

    # COMPS: official rows contain acceptable/unacceptable prefixes and property phrase.
    d = PRISTINE_FULL / "comps"
    for p in sorted(d.glob("*.jsonl")):
        sub = p.stem
        for i, obj in enumerate(iter_jsonl(p)):
            prop = str(obj.get("property_phrase", obj.get("property", "")))
            for role, pk in [("acceptable", "prefix_acceptable"), ("unacceptable", "prefix_unacceptable")]:
                if pk in obj:
                    yield {"family": "COMPS", "subtask": sub, "split": "full_eval", "role": role, "row": i, "text": str(obj[pk]) + " " + prop}

    # GlobalPIQA data is generated in the local INITIAL_MODEL_STUDIES tree and was used in research collation.
    for family, dname in [("GlobalPIQA_parallel", "global_piqa_parallel"), ("GlobalPIQA_nonparallel", "global_piqa_nonparallel")]:
        d = GLOBALPIQA_FULL / dname
        for p in sorted(d.glob("*.jsonl")):
            for i, obj in enumerate(iter_jsonl(p)):
                prompt = str(obj.get("prompt", ""))
                yield {"family": family, "subtask": p.stem, "split": "full_eval", "role": "prompt", "row": i, "text": prompt}
                for k, v in obj.items():
                    if k.startswith("solution") and isinstance(v, str):
                        yield {"family": family, "subtask": p.stem, "split": "full_eval", "role": k, "row": i, "text": prompt + " " + v}

    # Reading: sentence-level geometry and individual target-word geometry.
    rp = PRISTINE_FULL / "reading/reading_data.csv"
    with rp.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            sent = str(row.get("sentence", ""))
            word = str(row.get("word", ""))
            if sent:
                yield {"family": "Reading_sentence", "subtask": "reading_data", "split": "full_eval", "role": "sentence", "row": i, "text": sent}
            if word:
                yield {"family": "Reading_word", "subtask": "reading_data", "split": "full_eval", "role": "word", "row": i, "text": word}

    # SuperGLUE finetuning train+valid/predict files; sequence length is 512.
    gd = PRISTINE_FULL / "glue_filtered"
    for p in sorted(gd.glob("*.jsonl")):
        # names like boolq.train.jsonl or boolq.valid.jsonl
        stem_parts = p.name.split(".")
        task = stem_parts[0]
        split = stem_parts[1] if len(stem_parts) > 2 else "unknown"
        for i, obj in enumerate(iter_jsonl(p)):
            text_fields = []
            for k, v in obj.items():
                if k == "label":
                    continue
                if isinstance(v, str):
                    text_fields.append(v)
            text = " </s> ".join(text_fields)
            if text.strip():
                yield {"family": "SuperGLUE", "subtask": task, "split": split, "role": "all_text_fields", "row": i, "text": text}


def analyze(old_tok, new_tok) -> dict[str, Any]:
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    top_abs_delta: list[dict[str, Any]] = []
    total = 0
    for rec in collect_records():
        total += 1
        text = rec["text"]
        old = token_len(old_tok, text)
        new = token_len(new_tok, text)
        delta = new - old
        seq = SEQ_FINETUNE if rec["family"] == "SuperGLUE" else SEQ_SENTENCE
        key = (rec["family"], rec["subtask"], rec["split"])
        slot = by_key.setdefault(key, {
            "family": rec["family"], "subtask": rec["subtask"], "split": rec["split"],
            "n_texts": 0, "old_lens": [], "new_lens": [], "deltas": [],
            "old_trunc": 0, "new_trunc": 0, "new_only_trunc": 0, "old_only_trunc": 0,
            "roles": collections.Counter(), "seq_limit": seq,
        })
        slot["n_texts"] += 1
        slot["old_lens"].append(old)
        slot["new_lens"].append(new)
        slot["deltas"].append(delta)
        slot["old_trunc"] += int(old > seq)
        slot["new_trunc"] += int(new > seq)
        slot["new_only_trunc"] += int(new > seq and old <= seq)
        slot["old_only_trunc"] += int(old > seq and new <= seq)
        slot["roles"].update([rec["role"]])
        if len(top_abs_delta) < 80 or abs(delta) > abs(top_abs_delta[-1]["delta_tokens"]):
            x = {k: rec[k] for k in ["family", "subtask", "split", "role", "row"]}
            x.update({"old_tokens": old, "new_tokens": new, "delta_tokens": delta, "seq_limit": seq, "old_trunc": old > seq, "new_trunc": new > seq, "text_prefix": text[:240]})
            top_abs_delta.append(x)
            top_abs_delta.sort(key=lambda r: (abs(r["delta_tokens"]), r["new_trunc"] and not r["old_trunc"]), reverse=True)
            top_abs_delta = top_abs_delta[:80]
    rows = []
    for key, s in sorted(by_key.items()):
        n = s["n_texts"]
        row = {
            "family": s["family"], "subtask": s["subtask"], "split": s["split"], "n_texts": n,
            "roles": dict(s["roles"]), "seq_limit": s["seq_limit"],
            "old_len": summarize([float(x) for x in s["old_lens"]]),
            "new_len": summarize([float(x) for x in s["new_lens"]]),
            "delta_len": summarize([float(x) for x in s["deltas"]]),
            "token_ratio_new_over_old": (sum(s["new_lens"]) / sum(s["old_lens"])) if sum(s["old_lens"]) else None,
            "old_truncated": s["old_trunc"],
            "new_truncated": s["new_trunc"],
            "new_only_truncated": s["new_only_trunc"],
            "old_only_truncated": s["old_only_trunc"],
            "old_truncated_pct": 100.0 * s["old_trunc"] / n,
            "new_truncated_pct": 100.0 * s["new_trunc"] / n,
            "total_old_tokens": int(sum(s["old_lens"])),
            "total_new_tokens": int(sum(s["new_lens"])),
            "total_delta_tokens": int(sum(s["deltas"])),
        }
        rows.append(row)
    # Family aggregation from subtask rows.
    fam: dict[str, dict[str, Any]] = {}
    for r in rows:
        f = r["family"]
        a = fam.setdefault(f, {"family": f, "n_texts": 0, "total_old_tokens": 0, "total_new_tokens": 0, "total_delta_tokens": 0, "old_truncated": 0, "new_truncated": 0, "new_only_truncated": 0, "old_only_truncated": 0, "seq_limit": r["seq_limit"]})
        for k in ["n_texts", "total_old_tokens", "total_new_tokens", "total_delta_tokens", "old_truncated", "new_truncated", "new_only_truncated", "old_only_truncated"]:
            a[k] += r[k]
    fam_rows = []
    for f, a in sorted(fam.items()):
        n = a["n_texts"]
        a = dict(a)
        a["token_ratio_new_over_old"] = a["total_new_tokens"] / a["total_old_tokens"] if a["total_old_tokens"] else None
        a["old_truncated_pct"] = 100.0 * a["old_truncated"] / n if n else 0.0
        a["new_truncated_pct"] = 100.0 * a["new_truncated"] / n if n else 0.0
        fam_rows.append(a)
    return {"subtask_rows": rows, "family_rows": fam_rows, "top_abs_delta_examples": top_abs_delta, "total_text_units": total}


def write_md(payload: dict[str, Any]) -> None:
    lines = ["# research — tokenizer shift on official evaluation texts\n\n"]
    lines.append("CPU-only comparison of the old 100M-trained baseline16k tokenizer and the new 10M-trained compliant16k tokenizer on official evaluation text units. This is geometry evidence, not model performance.\n\n")
    lines.append("## Family aggregate\n\n")
    lines.append("| family | texts | seq | old tokens | new tokens | new/old | total Δ | old trunc % | new trunc % | new-only trunc | old-only trunc |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in payload["family_rows"]:
        lines.append(f"| {r['family']} | {r['n_texts']} | {r['seq_limit']} | {r['total_old_tokens']} | {r['total_new_tokens']} | {r['token_ratio_new_over_old']:.4f} | {r['total_delta_tokens']} | {r['old_truncated_pct']:.3f} | {r['new_truncated_pct']:.3f} | {r['new_only_truncated']} | {r['old_only_truncated']} |\n")
    lines.append("\n## Largest token-count shifts\n\n")
    lines.append("| family | subtask | split | role | row | old | new | Δ | trunc old→new | text prefix |\n")
    lines.append("|---|---|---|---|---:|---:|---:|---:|---|---|\n")
    for x in payload["top_abs_delta_examples"][:40]:
        prefix = x["text_prefix"].replace("|", "¦").replace("\n", " ")[:120]
        lines.append(f"| {x['family']} | {x['subtask']} | {x['split']} | {x['role']} | {x['row']} | {x['old_tokens']} | {x['new_tokens']} | {x['delta_tokens']} | {int(x['old_trunc'])}→{int(x['new_trunc'])} | {prefix} |\n")
    ((USER_ROOT / 'research/documents/frontier_consolidation/data/eval_text_tokenizer_shift/eval_text_tokenizer_shift.md')).write_text("".join(lines), encoding="utf-8")


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    old_tok = AutoTokenizer.from_pretrained(str(OLD), use_fast=True)
    new_tok = AutoTokenizer.from_pretrained(str(NEW), use_fast=True)
    payload = {
        "status": "EVAL_TEXT_TOKENIZER_SHIFT",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "old_tokenizer": str(OLD),
        "new_tokenizer": str(NEW),
        "sentence_seq_limit": SEQ_SENTENCE,
        "finetune_seq_limit": SEQ_FINETUNE,
        "pristine_full_eval": str(PRISTINE_FULL),
        "globalpiqa_full_eval": str(GLOBALPIQA_FULL),
    }
    payload.update(analyze(old_tok, new_tok))
    payload["elapsed_sec"] = round(time.time() - t0, 3)
    out_json = OUT / "eval_text_tokenizer_shift.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_md(payload)
    print(json.dumps({
        "status": payload["status"],
        "elapsed_sec": payload["elapsed_sec"],
        "total_text_units": payload["total_text_units"],
        "out_json": str(out_json),
        "out_md": str((USER_ROOT / 'research/documents/frontier_consolidation/data/eval_text_tokenizer_shift/eval_text_tokenizer_shift.md')),
        "family_rows": payload["family_rows"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
