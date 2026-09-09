#!/usr/bin/env python3
"""research: row-level exposure of research-tokenizer <unk> events on official scored text.

CPU-only. Uses the official strict sentence_zero_shot decode functions and MLM
offset-span scoring logic to quantify where the research same-pool tokenizer's
missing byte alphabet creates <unk> tokens. This clarifies that the research
endpoint is rule-compliant but empirically different from the research byte-
alphabet repair; endpoint choice must be made by official evaluation results.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import hashlib
import importlib.util
import json
import pathlib
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Iterable

from tokenizers import Tokenizer


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
STUDY = USER_ROOT / "experiments/archive/frontier_consolidation"
STRICT_REPO = USER_ROOT / "experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict"
EVAL_ROOT = STRICT_REPO / "evaluation_data/full_eval"
GLOBALPIQA_ROOT = USER_ROOT / "experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval"
TOK = STUDY / "data/compliant_tokenizer/tokenizer.json"
BYTEALPHA_TOK = STUDY / "data/compliant_tokenizer_bytealphabet/tokenizer.json"
OUT_DIR = STUDY / "data/unk_scored_row_exposure"

# Import official decode utilities by file path to avoid modifying the repo.
READ_FILES_PATH = STRICT_REPO / "evaluation_pipeline/sentence_zero_shot/read_files.py"
spec = importlib.util.spec_from_file_location("strict_sentence_read_files", READ_FILES_PATH)
read_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(read_mod)  # type: ignore[arg-type]

ZERO_SHOT_SPECS = [
    ("BLiMP", "blimp", EVAL_ROOT / "blimp_filtered"),
    ("Supplement", "blimp", EVAL_ROOT / "supplement_filtered"),
    ("EWoK", "ewok", EVAL_ROOT / "ewok_filtered"),
    ("Entity", "entity_tracking", EVAL_ROOT / "entity_tracking"),
    ("COMPS", "comps", EVAL_ROOT / "comps"),
    ("GlobalPIQA_parallel", "global_piqa_parallel", GLOBALPIQA_ROOT / "global_piqa_parallel"),
    ("GlobalPIQA_nonparallel", "global_piqa_nonparallel", GLOBALPIQA_ROOT / "global_piqa_nonparallel"),
]
SUPERGLUE_ROOT = EVAL_ROOT / "glue_filtered"
READING_FILE = EVAL_ROOT / "reading/reading_data.csv"
AOA_FILE = EVAL_ROOT / "aoa/cdi_childes.json"


def sha256_file(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_jsonl(path: pathlib.Path) -> Iterable[tuple[int, dict[str, Any], str]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            raw = line.rstrip("\n")
            if not raw:
                continue
            yield line_no, json.loads(raw), raw


def encode(tok: Tokenizer, text: str) -> tuple[list[int], list[str], list[tuple[int, int]]]:
    enc = tok.encode(text, add_special_tokens=True)
    return enc.ids, enc.tokens, enc.offsets


def completion_indices(text: str, completion: str, offsets: list[tuple[int, int]]) -> list[int]:
    start_char_idx = len(text) - len(completion)
    return [i for i, (start, end) in enumerate(offsets) if end > start_char_idx]


def count_sequence(tok: Tokenizer, text: str) -> dict[str, Any]:
    ids, toks, offs = encode(tok, text)
    u = sum(1 for t in toks if t == "<unk>")
    return {"tokens": len(toks), "unk": u, "toks": toks, "offsets": offs}


def scan_zero_shot_family(family: str, task: str, root: pathlib.Path, tok_a: Tokenizer, tok_b: Tokenizer) -> dict[str, Any]:
    fam = Counter(); by_task: dict[str, Counter] = defaultdict(Counter); examples = []
    files = sorted(root.glob("*.jsonl")) if root.is_dir() else []
    for path in files:
        task_name = path.stem
        for line_no, raw_row, raw_line in load_jsonl(path):
            decoded_rows = read_mod.decode(raw_line, path, task, False, None)
            if not decoded_rows:
                continue
            for decoded_idx, dec in enumerate(decoded_rows):
                fam["rows"] += 1; by_task[task_name]["rows"] += 1
                row_a = row_b = 0
                row_target_a = row_target_b = 0
                row_context_a = row_context_b = 0
                sentences = dec["sentences"]
                completions = dec["completions"]
                for cand_i, (text, completion) in enumerate(zip(sentences, completions)):
                    ca = count_sequence(tok_a, text)
                    cb = count_sequence(tok_b, text)
                    idx_a = completion_indices(text, completion, ca["offsets"])
                    idx_b = completion_indices(text, completion, cb["offsets"])
                    targ_a = sum(1 for i in idx_a if ca["toks"][i] == "<unk>")
                    targ_b = sum(1 for i in idx_b if cb["toks"][i] == "<unk>")
                    ctx_a = ca["unk"] - targ_a
                    ctx_b = cb["unk"] - targ_b
                    fam["candidate_strings"] += 1; fam["tokens_a"] += ca["tokens"]; fam["tokens_b"] += cb["tokens"]
                    fam["unk_a"] += ca["unk"]; fam["unk_b"] += cb["unk"]; fam["target_unk_a"] += targ_a; fam["target_unk_b"] += targ_b; fam["context_unk_a"] += ctx_a; fam["context_unk_b"] += ctx_b
                    by_task[task_name]["candidate_strings"] += 1; by_task[task_name]["tokens_a"] += ca["tokens"]; by_task[task_name]["tokens_b"] += cb["tokens"]
                    by_task[task_name]["unk_a"] += ca["unk"]; by_task[task_name]["unk_b"] += cb["unk"]; by_task[task_name]["target_unk_a"] += targ_a; by_task[task_name]["target_unk_b"] += targ_b; by_task[task_name]["context_unk_a"] += ctx_a; by_task[task_name]["context_unk_b"] += ctx_b
                    row_a += ca["unk"]; row_b += cb["unk"]; row_target_a += targ_a; row_target_b += targ_b; row_context_a += ctx_a; row_context_b += ctx_b
                    if ca["unk"]:
                        by_task[task_name]["candidate_strings_with_unk_a"] += 1
                    if cb["unk"]:
                        by_task[task_name]["candidate_strings_with_unk_b"] += 1
                    if ca["unk"] and len(examples) < 16:
                        try:
                            first = ca["toks"].index("<unk>")
                        except ValueError:
                            first = 0
                        examples.append({
                            "family": family, "task": task_name, "path": str(path), "line": line_no, "decoded_index": decoded_idx,
                            "candidate_index": cand_i, "uid": dec.get("UID"), "label": dec.get("label"),
                            "unk": ca["unk"], "target_unk": targ_a, "context_unk": ctx_a,
                            "bytealpha_unk": cb["unk"], "completion_prefix": completion[:120].replace("\n", "\\n"),
                            "text_prefix": text[:240].replace("\n", "\\n"),
                            "tokens_around_first_unk": ca["toks"][max(0, first-5):first+6],
                            "non_ascii_chars": [[repr(ch), ord(ch)] for ch in sorted(set(text)) if ord(ch) > 127][:30],
                        })
                if row_a:
                    fam["rows_with_unk_a"] += 1; by_task[task_name]["rows_with_unk_a"] += 1
                if row_b:
                    fam["rows_with_unk_b"] += 1; by_task[task_name]["rows_with_unk_b"] += 1
                if row_target_a:
                    fam["rows_with_target_unk_a"] += 1; by_task[task_name]["rows_with_target_unk_a"] += 1
                if row_target_b:
                    fam["rows_with_target_unk_b"] += 1; by_task[task_name]["rows_with_target_unk_b"] += 1
                if row_context_a:
                    fam["rows_with_context_unk_a"] += 1; by_task[task_name]["rows_with_context_unk_a"] += 1
                if row_context_b:
                    fam["rows_with_context_unk_b"] += 1; by_task[task_name]["rows_with_context_unk_b"] += 1
    return {"family": family, "task": task, "root": str(root), "summary": dict(fam), "by_task": {k: dict(v) for k, v in sorted(by_task.items())}, "examples": examples}


def superglue_text_fields(row: dict[str, Any]) -> list[tuple[str, str]]:
    return [(k, v) for k, v in row.items() if isinstance(v, str) and k not in {"label", "idx"}]


def scan_superglue(tok_a: Tokenizer, tok_b: Tokenizer) -> dict[str, Any]:
    fam = Counter(); by_file: dict[str, Counter] = defaultdict(Counter); examples = []
    for path in sorted(SUPERGLUE_ROOT.glob("*.jsonl")):
        parts = path.name.split(".")
        task = parts[0]
        split = parts[1] if len(parts) > 2 else "unknown"
        key = f"{task}.{split}"
        for line_no, row, _ in load_jsonl(path):
            fam["rows"] += 1; by_file[key]["rows"] += 1
            row_a = row_b = 0
            for field, text in superglue_text_fields(row):
                ca = count_sequence(tok_a, text)
                cb = count_sequence(tok_b, text)
                fam["strings"] += 1; fam["tokens_a"] += ca["tokens"]; fam["tokens_b"] += cb["tokens"]; fam["unk_a"] += ca["unk"]; fam["unk_b"] += cb["unk"]
                by_file[key]["strings"] += 1; by_file[key]["tokens_a"] += ca["tokens"]; by_file[key]["tokens_b"] += cb["tokens"]; by_file[key]["unk_a"] += ca["unk"]; by_file[key]["unk_b"] += cb["unk"]
                row_a += ca["unk"]; row_b += cb["unk"]
                if ca["unk"]:
                    by_file[key]["strings_with_unk_a"] += 1
                    if len(examples) < 12:
                        try:
                            first = ca["toks"].index("<unk>")
                        except ValueError:
                            first = 0
                        examples.append({"task_split": key, "path": str(path), "line": line_no, "field": field, "unk": ca["unk"], "bytealpha_unk": cb["unk"], "text_prefix": text[:240].replace("\n", "\\n"), "tokens_around_first_unk": ca["toks"][max(0, first-5):first+6], "non_ascii_chars": [[repr(ch), ord(ch)] for ch in sorted(set(text)) if ord(ch) > 127][:20]})
                if cb["unk"]:
                    by_file[key]["strings_with_unk_b"] += 1
            if row_a:
                fam["rows_with_unk_a"] += 1; by_file[key]["rows_with_unk_a"] += 1
            if row_b:
                fam["rows_with_unk_b"] += 1; by_file[key]["rows_with_unk_b"] += 1
    return {"family": "SuperGLUE", "root": str(SUPERGLUE_ROOT), "summary": dict(fam), "by_task_split": {k: dict(v) for k, v in sorted(by_file.items())}, "examples": examples}


def scan_reading_aoa(tok_a: Tokenizer, tok_b: Tokenizer) -> dict[str, Any]:
    # Light input-surface scan; Reading/AoA official functions score word-level
    # surprisals, so this is not a pair-ranking target/context split.
    fam = Counter(); examples = []
    if READING_FILE.exists():
        with READING_FILE.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for line_no, row in enumerate(reader, 2):
                for field, text in row.items():
                    if not isinstance(text, str) or not text:
                        continue
                    if field.lower() not in {"sentence", "text", "word", "context", "fulltext", "fulltextmarked"} and not any(x in field.lower() for x in ["text", "sentence", "word"]):
                        continue
                    ca = count_sequence(tok_a, text); cb = count_sequence(tok_b, text)
                    fam["strings"] += 1; fam["tokens_a"] += ca["tokens"]; fam["tokens_b"] += cb["tokens"]; fam["unk_a"] += ca["unk"]; fam["unk_b"] += cb["unk"]
                    if ca["unk"] and len(examples) < 6:
                        first = ca["toks"].index("<unk>")
                        examples.append({"source": "reading", "line": line_no, "field": field, "unk": ca["unk"], "bytealpha_unk": cb["unk"], "text_prefix": text[:240].replace("\n", "\\n"), "tokens_around_first_unk": ca["toks"][max(0, first-5):first+6]})
                    if ca["unk"]: fam["strings_with_unk_a"] += 1
                    if cb["unk"]: fam["strings_with_unk_b"] += 1
    if AOA_FILE.exists():
        data = json.loads(AOA_FILE.read_text(encoding="utf-8"))
        for word, recs in data.items() if isinstance(data, dict) else []:
            if not isinstance(recs, list):
                continue
            for i, rec in enumerate(recs):
                if not isinstance(rec, dict):
                    continue
                for field, text in rec.items():
                    if isinstance(text, str) and field in {"context", "target", "sentence", "text"}:
                        ca = count_sequence(tok_a, text); cb = count_sequence(tok_b, text)
                        fam["strings"] += 1; fam["tokens_a"] += ca["tokens"]; fam["tokens_b"] += cb["tokens"]; fam["unk_a"] += ca["unk"]; fam["unk_b"] += cb["unk"]
                        if ca["unk"] and len(examples) < 12:
                            first = ca["toks"].index("<unk>")
                            examples.append({"source": "aoa", "word": word, "record": i, "field": field, "unk": ca["unk"], "bytealpha_unk": cb["unk"], "text_prefix": text[:240].replace("\n", "\\n"), "tokens_around_first_unk": ca["toks"][max(0, first-5):first+6]})
                        if ca["unk"]: fam["strings_with_unk_a"] += 1
                        if cb["unk"]: fam["strings_with_unk_b"] += 1
    return {"family": "Reading_AoA_input_surface", "summary": dict(fam), "examples": examples}


def add_rates(d0: dict[str, Any]) -> dict[str, Any]:
    d = dict(d0)
    rows = d.get("rows", 0) or 0
    strings = d.get("candidate_strings", d.get("strings", 0)) or 0
    for prefix in ["a", "b"]:
        toks = d.get(f"tokens_{prefix}", 0) or 0
        d[f"unk_rate_{prefix}"] = d.get(f"unk_{prefix}", 0) / toks if toks else 0.0
        d[f"target_unk_rate_{prefix}"] = d.get(f"target_unk_{prefix}", 0) / toks if toks else 0.0
        d[f"row_unk_rate_{prefix}"] = d.get(f"rows_with_unk_{prefix}", 0) / rows if rows else 0.0
        d[f"row_target_unk_rate_{prefix}"] = d.get(f"rows_with_target_unk_{prefix}", 0) / rows if rows else 0.0
        d[f"string_unk_rate_{prefix}"] = d.get(f"candidate_strings_with_unk_{prefix}", d.get(f"strings_with_unk_{prefix}", 0)) / strings if strings else 0.0
    d["token_ratio_b_over_a"] = d.get("tokens_b", 0) / d["tokens_a"] if d.get("tokens_a") else 0.0
    return d


def enrich(obj: Any) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "summary" and isinstance(v, dict):
                out[k] = add_rates(v)
            elif k in {"by_task", "by_task_split"} and isinstance(v, dict):
                out[k] = {kk: add_rates(vv) if isinstance(vv, dict) else vv for kk, vv in v.items()}
            else:
                out[k] = enrich(v)
        return out
    if isinstance(obj, list):
        return [enrich(x) for x in obj]
    return obj


def write_md(payload: dict[str, Any], out_md: pathlib.Path) -> None:
    lines = ["# research row-level research-tokenizer `<unk>` exposure", ""]
    lines.append("The research tokenizer is rule-compliant because it was fitted only on the allowed 10M reinvest pool. This report quantifies where its missing byte-level alphabet entries create `<unk>` tokens under the official strict decode/tokenization path. The byte-alphabet tokenizer is the only structural repair now under study; no benchmark-informed vocabulary tuning is allowed.")
    lines.append("")
    lines.append(f"research tokenizer SHA: `{payload['tokenizers']['sha']}`")
    lines.append(f"Byte-alphabet tokenizer SHA: `{payload['tokenizers']['bytealpha_sha']}`")
    lines.append("")
    lines.append("## Zero-shot official-decode families")
    lines.append("")
    lines.append("| family | decoded rows | research rows with `<unk>` | research rows with target-span `<unk>` | target row rate | research `<unk>` tokens | ByteAlpha `<unk>` tokens | token ratio B/A |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for fam in payload["zero_shot_families"]:
        s = fam["summary"]
        lines.append(f"| {fam['family']} | {s.get('rows',0)} | {s.get('rows_with_unk_a',0)} | {s.get('rows_with_target_unk_a',0)} | {s.get('row_target_unk_rate_a',0):.6f} | {s.get('unk_a',0)} | {s.get('unk_b',0)} | {s.get('token_ratio_b_over_a',0):.6f} |")
    lines.append("")
    lines.append("## Supplement task localization")
    lines.append("")
    supp = next((f for f in payload["zero_shot_families"] if f["family"] == "Supplement"), None)
    if supp:
        lines.append("| task | rows | rows with research `<unk>` | rows with target-span `<unk>` | target row rate | research `<unk>` tokens | target `<unk>` tokens | token ratio B/A |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for task, s in supp["by_task"].items():
            lines.append(f"| {task} | {s.get('rows',0)} | {s.get('rows_with_unk_a',0)} | {s.get('rows_with_target_unk_a',0)} | {s.get('row_target_unk_rate_a',0):.6f} | {s.get('unk_a',0)} | {s.get('target_unk_a',0)} | {s.get('token_ratio_b_over_a',0):.6f} |")
    lines.append("")
    lines.append("For MLM zero-shot, tokens whose offsets overlap the completion span are individually masked and scored. In Supplement, the newline is part of the whole sentence/completion, so the affected rows above are direct target-span exposure, not only context exposure.")
    lines.append("")
    lines.append("## SuperGLUE finetuning input exposure")
    lines.append("")
    sg = payload["superglue"]
    s = sg["summary"]
    lines.append(f"SuperGLUE train/valid text rows scanned: {s.get('rows',0)}; rows with research `<unk>`: {s.get('rows_with_unk_a',0)} ({s.get('row_unk_rate_a',0):.6f}); research `<unk>` tokens: {s.get('unk_a',0)}; byte-alphabet `<unk>` tokens: {s.get('unk_b',0)}; token ratio B/A {s.get('token_ratio_b_over_a',0):.6f}.")
    lines.append("")
    lines.append("| task.split | rows | rows with research `<unk>` | row rate | research `<unk>` tokens | token ratio B/A |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for key, d in sorted(sg["by_task_split"].items(), key=lambda kv: kv[1].get("rows_with_unk_a",0), reverse=True)[:12]:
        lines.append(f"| {key} | {d.get('rows',0)} | {d.get('rows_with_unk_a',0)} | {d.get('row_unk_rate_a',0):.6f} | {d.get('unk_a',0)} | {d.get('token_ratio_b_over_a',0):.6f} |")
    lines.append("")
    ra = payload["reading_aoa_surface"]["summary"]
    lines.append("## Reading/AoA input surface")
    lines.append("")
    lines.append(f"Input strings scanned: {ra.get('strings',0)}; research `<unk>` tokens: {ra.get('unk_a',0)}; byte-alphabet `<unk>` tokens: {ra.get('unk_b',0)}; token ratio B/A {ra.get('token_ratio_b_over_a',0):.6f}.")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("research remains a valid compliant endpoint and should not be discarded without its official score. The `<unk>` exposure is sharply localized: Supplement has 473/5218 rows with direct target-span `<unk>` tokens, because the completion equals the full string and contains dialogue newlines in QA/turn-taking; SuperGLUE exposure is sparse and mostly BoolQ/QQP input text; BLiMP/EWoK/Entity/COMPS/GlobalPIQA show no research `<unk>` under the official decode scan. The byte-alphabet repair removes the coverage defect with almost unchanged length geometry, but its newline/rare-byte tokens are structurally present rather than necessarily well-trained from the 10M stream. Therefore pristine official evaluation, not speculation, must decide between the two compliant endpoints if both finish without blocking resource priority.")
    lines.append("")
    lines.append(f"Full JSON: `{out_md.with_suffix('.json')}`")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tok_a = Tokenizer.from_file(str(TOK))
    tok_b = Tokenizer.from_file(str(BYTEALPHA_TOK))
    zero = []
    for family, task, root in ZERO_SHOT_SPECS:
        zero.append(scan_zero_shot_family(family, task, root, tok_a, tok_b))
    payload = {
        "status": "STEP35_UNK_SCORED_ROW_EXPOSURE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Quantify row-level official-decode exposure of research same-pool tokenizer <unk> events; no model inference.",
        "official_decode_file": str(READ_FILES_PATH),
        "eval_root": str(EVAL_ROOT),
        "tokenizers": {
            "path": str(TOK),
            "sha": sha256_file(TOK),
            "bytealpha_path": str(BYTEALPHA_TOK),
            "bytealpha_sha": sha256_file(BYTEALPHA_TOK),
        },
        "zero_shot_families": zero,
        "superglue": scan_superglue(tok_a, tok_b),
        "reading_aoa_surface": scan_reading_aoa(tok_a, tok_b),
        "elapsed_sec": round(time.time() - t0, 3),
    }
    payload = enrich(payload)
    out_json = OUT_DIR / "unk_scored_row_exposure.json"
    out_md = OUT_DIR / "unk_scored_row_exposure.md"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_md(payload, out_md)
    print(json.dumps({
        "status": payload["status"],
        "out_json": str(out_json),
        "out_md": str(out_md),
        "zero_shot_rows_with_unk": {f["family"]: f["summary"].get("rows_with_unk_a", 0) for f in payload["zero_shot_families"]},
        "zero_shot_rows_with_target_unk": {f["family"]: f["summary"].get("rows_with_target_unk_a", 0) for f in payload["zero_shot_families"]},
        "superglue_rows_with_unk": payload["superglue"]["summary"].get("rows_with_unk_a", 0),
        "elapsed_sec": payload["elapsed_sec"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
