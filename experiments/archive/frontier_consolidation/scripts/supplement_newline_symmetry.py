#!/usr/bin/env python3
"""research: Supplement newline/<unk> symmetry under the research tokenizer.

CPU-only.  This does not run any model.  It measures whether the research
same-pool tokenizer's Supplement <unk> targets are matched between good and bad
alternatives, and how much of the official Supplement column is structurally
exposed to this tokenizer surface.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import hashlib
import importlib.util
import json
import pathlib
import statistics
import time
from collections import Counter, defaultdict
from typing import Any

from tokenizers import Tokenizer


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
STUDY = USER_ROOT / "experiments/archive/frontier_consolidation"
STRICT_REPO = USER_ROOT / "experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict"
SUPP_DIR = STRICT_REPO / "evaluation_data/full_eval/supplement_filtered"
READ_FILES_PATH = STRICT_REPO / "evaluation_pipeline/sentence_zero_shot/read_files.py"
TOK = STUDY / "data/compliant_tokenizer/tokenizer.json"
BYTEALPHA_TOK = STUDY / "data/compliant_tokenizer_bytealphabet/tokenizer.json"
OUT_DIR = STUDY / "data/supplement_newline_symmetry"

SHA_EXPECTED = "91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9"
BYTEALPHA_SHA_EXPECTED = "b2b317e655a96f2c14eb559ce9fa7573f180904d84819cacfb5f61174cf355cf"


spec = importlib.util.spec_from_file_location("strict_sentence_read_files_step043", READ_FILES_PATH)
read_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(read_mod)  # type: ignore[arg-type]


def sha256_file(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def encode(tok: Tokenizer, text: str) -> dict[str, Any]:
    enc = tok.encode(text, add_special_tokens=True)
    return {"ids": enc.ids, "tokens": enc.tokens, "offsets": enc.offsets}


def target_indices(text: str, completion: str, offsets: list[tuple[int, int]]) -> list[int]:
    start_char_idx = len(text) - len(completion)
    return [i for i, (_start, end) in enumerate(offsets) if end > start_char_idx]


def token_records(tok: Tokenizer, text: str, completion: str) -> dict[str, Any]:
    enc = encode(tok, text)
    idxs = target_indices(text, completion, enc["offsets"])
    target_set = set(idxs)
    records = []
    for i, (token, (start, end), tid) in enumerate(zip(enc["tokens"], enc["offsets"], enc["ids"])):
        rec = {
            "index": i,
            "id": tid,
            "token": token,
            "start": start,
            "end": end,
            "span": text[start:end],
            "is_target": i in target_set,
            "is_unk": token == "<unk>",
        }
        records.append(rec)
    unk_target_records = [r for r in records if r["is_target"] and r["is_unk"]]
    return {
        "token_count": len(records),
        "target_count": len(idxs),
        "unk_count": sum(1 for r in records if r["is_unk"]),
        "target_unk_count": len(unk_target_records),
        "target_records": [records[i] for i in idxs],
        "unk_target_records": unk_target_records,
        "records": records,
    }


def multiset(values: list[Any]) -> dict[str, int]:
    return dict(Counter(json.dumps(v, ensure_ascii=False, sort_keys=True) for v in values))


def window(text: str, start: int, end: int, k: int) -> dict[str, str]:
    return {
        "left": text[max(0, start - k):start],
        "right": text[end:end + k],
    }


def bytealpha_tokens_for_span(tok: Tokenizer, text_span: str) -> list[str]:
    if text_span == "":
        return []
    return tok.encode(text_span, add_special_tokens=False).tokens


def read_jsonl(path: pathlib.Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            raw = line.rstrip("\n")
            if raw:
                yield line_no, raw, json.loads(raw)


def analyze_pair(task_name: str, path: pathlib.Path, line_no: int, raw_line: str, tok_step35: Tokenizer, tok_ba: Tokenizer) -> dict[str, Any]:
    decoded = read_mod.decode(raw_line, path, "blimp", False, None)[0]
    sentences = decoded["sentences"]
    completions = decoded["completions"]
    cand = []
    for label, text, completion in zip(["good", "bad"], sentences, completions):
        s35 = token_records(tok_step35, text, completion)
        ba = token_records(tok_ba, text, completion)
        unk_spans = [{
            "start": r["start"],
            "end": r["end"],
            "span": r["span"],
            "repr": repr(r["span"]),
            "ordinals": [ord(ch) for ch in r["span"]],
            "bytealpha_tokens_for_span": bytealpha_tokens_for_span(tok_ba, r["span"]),
            "local8": window(text, r["start"], r["end"], 8),
            "local24": window(text, r["start"], r["end"], 24),
        } for r in s35["unk_target_records"]]
        cand.append({
            "which": label,
            "text": text,
            "token_count_step35": s35["token_count"],
            "target_count_step35": s35["target_count"],
            "target_unk_count_step35": s35["target_unk_count"],
            "token_count_bytealpha": ba["token_count"],
            "target_count_bytealpha": ba["target_count"],
            "target_unk_count_bytealpha": ba["target_unk_count"],
            "unk_spans": unk_spans,
        })
    good, bad = cand
    good_spans = [u["span"] for u in good["unk_spans"]]
    bad_spans = [u["span"] for u in bad["unk_spans"]]
    good_offsets = [(u["start"], u["end"]) for u in good["unk_spans"]]
    bad_offsets = [(u["start"], u["end"]) for u in bad["unk_spans"]]
    affected = bool(good_spans or bad_spans)
    both = bool(good_spans and bad_spans)
    same_count = len(good_spans) == len(bad_spans)
    same_spans = multiset(good_spans) == multiset(bad_spans)
    same_offsets = Counter(good_offsets) == Counter(bad_offsets)
    same_span_offsets = same_count and Counter(zip(good_spans, good_offsets)) == Counter(zip(bad_spans, bad_offsets))
    all_spans = good_spans + bad_spans
    all_include_newline = affected and all("\n" in span for span in all_spans)
    all_pure_or_prefix_newline = affected and all(span.startswith("\n") for span in all_spans)
    local8_same = True
    local24_same = True
    if same_count and good["unk_spans"] and bad["unk_spans"]:
        for gu, bu in zip(good["unk_spans"], bad["unk_spans"]):
            if gu["local8"] != bu["local8"]:
                local8_same = False
            if gu["local24"] != bu["local24"]:
                local24_same = False
    elif affected:
        local8_same = False
        local24_same = False
    text_before_same = True
    text_after_same = True
    if same_count and good["unk_spans"] and bad["unk_spans"]:
        for gu, bu in zip(good["unk_spans"], bad["unk_spans"]):
            if good["text"][:gu["start"]] != bad["text"][:bu["start"]]:
                text_before_same = False
            if good["text"][gu["end"]:] != bad["text"][bu["end"]:]:
                text_after_same = False
    elif affected:
        text_before_same = False
        text_after_same = False
    total_s35_targets = good["target_count_step35"] + bad["target_count_step35"]
    total_ba_targets = good["target_count_bytealpha"] + bad["target_count_bytealpha"]
    total_target_unk = good["target_unk_count_step35"] + bad["target_unk_count_step35"]
    return {
        "task": task_name,
        "path": str(path),
        "line": line_no,
        "uid": decoded.get("UID"),
        "affected": affected,
        "both_candidates_affected": both,
        "same_unk_count_good_bad": same_count,
        "same_unk_span_multiset": same_spans,
        "same_unk_offset_multiset": same_offsets,
        "same_unk_span_and_offset": same_span_offsets,
        "all_unk_spans_include_newline": all_include_newline,
        "all_unk_spans_start_with_newline": all_pure_or_prefix_newline,
        "local8_same_at_unk": local8_same,
        "local24_same_at_unk": local24_same,
        "prefix_before_unk_same": text_before_same,
        "suffix_after_unk_same": text_after_same,
        "total_target_tokens_pair": total_s35_targets,
        "bytealpha_total_target_tokens_pair": total_ba_targets,
        "target_unk_tokens_pair": total_target_unk,
        "target_unk_fraction_step35_pair": total_target_unk / total_s35_targets if total_s35_targets else 0.0,
        "bytealpha_minus_step35_target_tokens_pair": total_ba_targets - total_s35_targets,
        "good": {k: v for k, v in good.items() if k != "text"},
        "bad": {k: v for k, v in bad.items() if k != "text"},
        "good_text_preview": good["text"][:240].replace("\n", "\\n"),
        "bad_text_preview": bad["text"][:240].replace("\n", "\\n"),
    }


def add_bool(counter: Counter, key: str, value: bool):
    if value:
        counter[key] += 1


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, Counter] = defaultdict(Counter)
    spans_by_task: dict[str, Counter] = defaultdict(Counter)
    byte_repairs_by_task: dict[str, Counter] = defaultdict(Counter)
    frac_values_by_task: dict[str, list[float]] = defaultdict(list)
    output_examples = []
    symmetry_keys = [
        "both_candidates_affected", "same_unk_count_good_bad", "same_unk_span_multiset",
        "same_unk_offset_multiset", "same_unk_span_and_offset", "all_unk_spans_include_newline",
        "all_unk_spans_start_with_newline", "local8_same_at_unk", "local24_same_at_unk",
        "prefix_before_unk_same", "suffix_after_unk_same",
    ]
    for r in rows:
        c = by_task[r["task"]]
        c["rows"] += 1
        if r["affected"]:
            c["affected_rows"] += 1
            frac_values_by_task[r["task"]].append(r["target_unk_fraction_step35_pair"])
            if len(output_examples) < 14:
                output_examples.append(r)
            # Empty-span equality is true for unaffected rows; only count these
            # structure/symmetry facts among rows that actually contain research
            # target <unk> exposure.
            for key in symmetry_keys:
                add_bool(c, key, r[key])
        c["target_tokens_pair"] += r["total_target_tokens_pair"]
        c["bytealpha_target_tokens_pair"] += r["bytealpha_total_target_tokens_pair"]
        c["target_unk_tokens_pair"] += r["target_unk_tokens_pair"]
        c["bytealpha_minus_step35_target_tokens_pair"] += r["bytealpha_minus_step35_target_tokens_pair"]
        for side in ["good", "bad"]:
            for unk in r[side]["unk_spans"]:
                spans_by_task[r["task"]][repr(unk["span"])] += 1
                for tok in unk["bytealpha_tokens_for_span"]:
                    byte_repairs_by_task[r["task"]][tok] += 1
    summary_by_task: dict[str, Any] = {}
    supplement_macro_at_risk_points = 0.0
    row_flip_coefficients = {}
    for task, c in sorted(by_task.items()):
        rows_n = c["rows"]
        affected = c["affected_rows"]
        affected_rate = affected / rows_n if rows_n else 0.0
        supplement_points_max = 20.0 * affected_rate
        supplement_macro_at_risk_points += supplement_points_max
        row_flip_coefficients[task] = {
            "rows": rows_n,
            "supplement_column_points_per_row_flip": 20.0 / rows_n if rows_n else 0.0,
            "overall_points_per_row_flip": 20.0 / (9.0 * rows_n) if rows_n else 0.0,
        }
        frac_values = frac_values_by_task.get(task, [])
        summary_by_task[task] = {
            **dict(c),
            "affected_rate": affected_rate,
            "target_unk_fraction": c["target_unk_tokens_pair"] / c["target_tokens_pair"] if c["target_tokens_pair"] else 0.0,
            "bytealpha_over_step35_target_token_ratio": c["bytealpha_target_tokens_pair"] / c["target_tokens_pair"] if c["target_tokens_pair"] else 0.0,
            "max_supplement_column_points_if_all_affected_rows_flip": supplement_points_max,
            "affected_row_target_unk_fraction_mean": statistics.mean(frac_values) if frac_values else 0.0,
            "affected_row_target_unk_fraction_max": max(frac_values) if frac_values else 0.0,
            "unk_span_counts": spans_by_task[task].most_common(30),
            "bytealpha_repair_token_counts_from_step35_unk_spans": byte_repairs_by_task[task].most_common(30),
        }
    overall = Counter()
    for r in rows:
        overall["rows"] += 1
        if r["affected"]:
            overall["affected_rows"] += 1
            for key in symmetry_keys:
                add_bool(overall, key, r[key])
        overall["target_tokens_pair"] += r["total_target_tokens_pair"]
        overall["bytealpha_target_tokens_pair"] += r["bytealpha_total_target_tokens_pair"]
        overall["target_unk_tokens_pair"] += r["target_unk_tokens_pair"]
        overall["bytealpha_minus_step35_target_tokens_pair"] += r["bytealpha_minus_step35_target_tokens_pair"]
    overall_dict = dict(overall)
    overall_dict.update({
        "affected_rate_rows": overall["affected_rows"] / overall["rows"] if overall["rows"] else 0.0,
        "target_unk_fraction": overall["target_unk_tokens_pair"] / overall["target_tokens_pair"] if overall["target_tokens_pair"] else 0.0,
        "bytealpha_over_step35_target_token_ratio": overall["bytealpha_target_tokens_pair"] / overall["target_tokens_pair"] if overall["target_tokens_pair"] else 0.0,
        "supplement_macro_points_structurally_exposed": supplement_macro_at_risk_points,
        "overall_points_structurally_exposed_if_every_affected_supplement_row_flips": supplement_macro_at_risk_points / 9.0,
    })
    return {
        "overall": overall_dict,
        "by_task": summary_by_task,
        "row_flip_coefficients": row_flip_coefficients,
        "examples": output_examples,
    }


def write_md(payload: dict[str, Any], out_md: pathlib.Path) -> None:
    summ = payload["summary"]
    overall = summ["overall"]
    lines = []
    lines.append("# research Supplement newline symmetry under the research tokenizer")
    lines.append("")
    lines.append("This CPU-only analysis asks whether the research same-pool tokenizer's Supplement `<unk>` targets are matched between the good and bad alternatives in the official MLM scoring path. It does not run model inference and does not alter either compliant tokenizer.")
    lines.append("")
    lines.append(f"research tokenizer SHA: `{payload['tokenizers']['sha']}`")
    lines.append(f"Byte-alphabet tokenizer SHA: `{payload['tokenizers']['bytealpha_sha']}`")
    lines.append("")
    lines.append("## Aggregate Supplement surface")
    lines.append("")
    lines.append(f"Rows scanned: {overall['rows']}; rows with research target `<unk>`: {overall['affected_rows']} ({overall['affected_rate_rows']:.6f}).")
    lines.append(f"research target `<unk>` tokens: {overall['target_unk_tokens_pair']} out of {overall['target_tokens_pair']} paired candidate target tokens ({overall['target_unk_fraction']:.6f}).")
    lines.append(f"Byte-alphabet / research target-token ratio: {overall['bytealpha_over_step35_target_token_ratio']:.6f}.")
    lines.append(f"Affected rows all have both candidates affected: {overall.get('both_candidates_affected', 0)}/{overall['affected_rows']}.")
    lines.append(f"Affected rows with same `<unk>` count in good and bad: {overall.get('same_unk_count_good_bad', 0)}/{overall['affected_rows']}.")
    lines.append(f"Affected rows with same `<unk>` span multiset: {overall.get('same_unk_span_multiset', 0)}/{overall['affected_rows']}.")
    lines.append(f"Affected rows with same span+offset: {overall.get('same_unk_span_and_offset', 0)}/{overall['affected_rows']}.")
    lines.append(f"Affected rows where every research `<unk>` span starts with newline: {overall.get('all_unk_spans_start_with_newline', 0)}/{overall['affected_rows']}.")
    lines.append(f"Affected rows with identical 8-character local windows around the research `<unk>` target: {overall.get('local8_same_at_unk', 0)}/{overall['affected_rows']}.")
    lines.append(f"Affected rows with identical 24-character local windows around the research `<unk>` target: {overall.get('local24_same_at_unk', 0)}/{overall['affected_rows']}.")
    lines.append("")
    lines.append(f"If every affected Supplement row flipped solely because of this tokenizer surface, the maximum exposed Supplement movement would be {overall['supplement_macro_points_structurally_exposed']:.3f} column points, or {overall['overall_points_structurally_exposed_if_every_affected_supplement_row_flips']:.3f} Overall points. This is only a structural bound; symmetry means the realized effect could be much smaller and must be measured by official scoring.")
    lines.append("")
    lines.append("## By Supplement subtask")
    lines.append("")
    lines.append("| subtask | rows | affected rows | affected rate | target `<unk>` frac | same count | same span | same span+offset | local8 same | top research span | max Supplement pts | Overall per row flip |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|")
    for task, d in summ["by_task"].items():
        top_span = d["unk_span_counts"][0][0] if d["unk_span_counts"] else ""
        coeff = summ["row_flip_coefficients"][task]
        lines.append(
            f"| {task} | {d['rows']} | {d.get('affected_rows', 0)} | {d['affected_rate']:.6f} | {d['target_unk_fraction']:.6f} | {d.get('same_unk_count_good_bad', 0)} | {d.get('same_unk_span_multiset', 0)} | {d.get('same_unk_span_and_offset', 0)} | {d.get('local8_same_at_unk', 0)} | `{top_span}` | {d['max_supplement_column_points_if_all_affected_rows_flip']:.3f} | {coeff['overall_points_per_row_flip']:.6f} |"
        )
    lines.append("")
    lines.append("## Interpretation for endpoint comparison")
    lines.append("")
    lines.append("The research tokenizer weakness is real but highly structured. In all affected rows the good and bad candidates both contain a research target `<unk>`; in the affected QA rows the unknown span and offset are identical, while turn-taking contains the same newline-starting target in both candidates but local context can differ because the contrast changes speaker/pronoun material before or after the line break. Thus the newline/speaker `<unk>` is not a simple one-sided poison token, and its model-score effect cannot be inferred from tokenization alone.")
    lines.append("")
    lines.append("For later pristine results, a research-versus-byte-alphabet difference concentrated in `qa_congruence_easy`, `qa_congruence_tricky`, or `turn_taking` should be read as tokenizer-surface interaction. A difference in hypernym, subject-aux inversion, BLiMP, EWoK, Entity, COMPS, or GlobalPIQA cannot be explained by the research newline `<unk>` exposure shown here.")
    lines.append("")
    lines.append(f"Full JSON: `{payload['out_json']}`")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    s35_sha = sha256_file(TOK)
    ba_sha = sha256_file(BYTEALPHA_TOK)
    tok_s35 = Tokenizer.from_file(str(TOK))
    tok_ba = Tokenizer.from_file(str(BYTEALPHA_TOK))
    rows = []
    for path in sorted(SUPP_DIR.glob("*.jsonl")):
        task_name = path.stem
        for line_no, raw_line, _row in read_jsonl(path):
            rows.append(analyze_pair(task_name, path, line_no, raw_line, tok_s35, tok_ba))
    summary = summarize(rows)
    out_json = OUT_DIR / "supplement_newline_symmetry.json"
    out_md = OUT_DIR / "supplement_newline_symmetry.md"
    payload = {
        "status": "SUPPLEMENT_NEWLINE_SYMMETRY",
        "scientific_question": "whether research-tokenizer Supplement <unk> targets are symmetric between official good/bad alternatives and how much score surface they can affect",
        "official_paths": {
            "strict_repo": str(STRICT_REPO),
            "supplement_dir": str(SUPP_DIR),
            "read_files_py": str(READ_FILES_PATH),
        },
        "tokenizers": {
            "path": str(TOK),
            "sha": s35_sha,
            "sha_expected": SHA_EXPECTED,
            "sha_matches_expected": s35_sha == SHA_EXPECTED,
            "bytealpha_path": str(BYTEALPHA_TOK),
            "bytealpha_sha": ba_sha,
            "bytealpha_sha_expected": BYTEALPHA_SHA_EXPECTED,
            "bytealpha_sha_matches_expected": ba_sha == BYTEALPHA_SHA_EXPECTED,
        },
        "summary": summary,
        "rows": rows,
        "elapsed_sec": round(time.time() - t0, 3),
        "out_json": str(out_json),
        "out_md": str(out_md),
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_md(payload, out_md)
    print(json.dumps({
        "status": payload["status"],
        "out_json": str(out_json),
        "out_md": str(out_md),
        "rows": summary["overall"]["rows"],
        "affected_rows": summary["overall"]["affected_rows"],
        "same_count": summary["overall"].get("same_unk_count_good_bad", 0),
        "same_span": summary["overall"].get("same_unk_span_multiset", 0),
        "same_span_offset": summary["overall"].get("same_unk_span_and_offset", 0),
        "supplement_points_exposed": summary["overall"]["supplement_macro_points_structurally_exposed"],
        "elapsed_sec": payload["elapsed_sec"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
