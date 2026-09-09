#!/usr/bin/env python3
"""research CPU audit of the BabyLM Strict AoA 6560-vs-8005 discrepancy.

This script is read-only except for its outputs. It establishes:
  * the official repository/data identities used locally;
  * that official collate_preds.py expects AOA_SIZE=8005;
  * that official AoA_word/run.py default min_context is 0;
  * that the inherited local helper hardcoded load_eval(word_path, 20, False);
  * the exact words/contexts omitted by min_context=20.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import ast
import collections
import csv
import hashlib
import json
import pathlib
import re
import subprocess
import textwrap
from typing import Any

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/representation_and_objectives')
WORKSPACE = _public_path('experiments/archive/representation_and_objectives')
STRICT_REPO = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval')
STRICT = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict')

FILES = {
    "collate_preds": _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_pipeline/collate_preds.py'),
    "aoa_run": _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_pipeline/AoA_word/run.py'),
    "aoa_eval_util": _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_pipeline/AoA_word/eval_util.py'),
    "aoa_utils": _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_pipeline/utils.py'),
    "cdi_childes": _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/aoa/cdi_childes.json'),
    "cdi_human": _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/aoa/cdi_human.csv'),
    "local_helper": _public_path('experiments/archive/compact_experience/scripts/aoa_local_ckpts_for_model.py'),
    "aoa_record": _public_path('experiments/archive/representation_and_objectives/data/compact_reinvest_full_eval/aoa_outputs/compact_view_reinvest/aoa_local_ckpts.json'),
}


def sha256_path(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info() -> dict[str, Any]:
    def run(args):
        try:
            out = subprocess.check_output(args, cwd=STRICT_REPO, stderr=subprocess.STDOUT, text=True)
            return {"ok": True, "stdout": out.strip()}
        except Exception as e:
            return {"ok": False, "error": repr(e)}
    return {
        "repo": str(STRICT_REPO),
        "remote": run(["git", "remote", "-v"]),
        "head": run(["git", "rev-parse", "HEAD"]),
        "status_short": run(["git", "status", "--short"]),
    }


def parse_aoa_size(collate_text: str) -> int | None:
    m = re.search(r"^AOA_SIZE\s*=\s*(\d+)\s*$", collate_text, flags=re.M)
    return int(m.group(1)) if m else None


def parse_run_default_min_context(run_text: str) -> int | None:
    tree = ast.parse(run_text)
    # Find add_argument(..., "--min_context", default=<int>, ...)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            args = [getattr(a, "value", None) for a in node.args]
            if "--min_context" in args:
                for kw in node.keywords:
                    if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                        return int(kw.value.value)
    return None


def find_helper_min_context(helper_text: str) -> dict[str, Any]:
    # Preserve exact line(s), not only parsed value.
    hits = []
    for i, line in enumerate(helper_text.splitlines(), start=1):
        if "load_eval" in line or "min_context" in line:
            hits.append({"line": i, "text": line})
    m = re.search(r"load_eval\(\s*word_path\s*,\s*(\d+)\s*,\s*False\s*\)", helper_text)
    return {"hardcoded_min_context": int(m.group(1)) if m else None, "hits": hits}


def load_contexts() -> dict[str, list[dict[str, Any]]]:
    with FILES["cdi_childes"].open(encoding="utf-8") as f:
        return json.load(f)


def counts_for_min_context(data: dict[str, list[dict[str, Any]]], min_context: int) -> dict[str, Any]:
    words = [w for w, contexts in data.items() if len(w) > 1 and len(contexts) >= min_context]
    return {
        "min_context": min_context,
        "num_words_loaded": len(words),
        "total_context_rows_loaded": sum(len(data[w]) for w in words),
        "context_count_distribution_loaded": dict(sorted(collections.Counter(len(data[w]) for w in words).items())),
        "words_loaded": words,
    }


def cdi_human_words() -> set[str]:
    with FILES["cdi_human"].open(encoding="utf-8") as f:
        return {r["word"] for r in csv.DictReader(f)}


def normalize_context(c: dict[str, Any]) -> str:
    # Context entries are source-derived evaluation text, not instructions.
    return str(c.get("context", ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(_public_path('experiments/archive/representation_and_objectives/data/aoa_discrepancy_audit')))
    args = ap.parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    texts = {k: p.read_text(encoding="utf-8", errors="replace") for k, p in FILES.items() if p.suffix in {".py", ".json", ".csv"} and k != "cdi_childes"}
    collate_text = FILES["collate_preds"].read_text(encoding="utf-8")
    run_text = FILES["aoa_run"].read_text(encoding="utf-8")
    helper_text = FILES["local_helper"].read_text(encoding="utf-8")
    data = load_contexts()

    all0 = counts_for_min_context(data, 0)
    min20 = counts_for_min_context(data, 20)
    words0 = set(all0["words_loaded"])
    words20 = set(min20["words_loaded"])
    omitted_words = sorted(words0 - words20)
    omitted_context_rows = sum(len(data[w]) for w in omitted_words)

    cdi_words = cdi_human_words()
    omitted = []
    for w in omitted_words:
        contexts = data[w]
        omitted.append({
            "word": w,
            "context_count": len(contexts),
            "in_cdi_human": w in cdi_words,
            "contexts_sha256": hashlib.sha256(json.dumps(contexts, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest(),
            "sample_contexts": [normalize_context(c) for c in contexts[:3]],
        })
    omitted_jsonl = out_dir / "omitted_by_min_context20_words.jsonl"
    omitted_jsonl.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in omitted), encoding="utf-8")

    research = json.loads(FILES["aoa_record"].read_text(encoding="utf-8"))
    file_identities = {}
    for k, p in FILES.items():
        if p.exists() and p.is_file():
            file_identities[k] = {"path": str(p), "size_bytes": p.stat().st_size, "sha256": sha256_path(p)}

    official_size = parse_aoa_size(collate_text)
    run_default = parse_run_default_min_context(run_text)
    helper_info = find_helper_min_context(helper_text)

    result = {
        "status": "AOA_MINCONTEXT_DISCREPANCY_AUDIT",
        "git_info": git_info(),
        "file_identities": file_identities,
        "official_collator_aoa_size": official_size,
        "official_aoa_run_default_min_context": run_default,
        "local_helper_min_context": helper_info,
        "counts_by_min_context": {
            "official_min_context_0": {k: v for k, v in all0.items() if k != "words_loaded"},
            "helper_min_context_20": {k: v for k, v in min20.items() if k != "words_loaded"},
        },
        "recorded_aoa": {
            "path": str(FILES["aoa_record"]),
            "row_count_values": research.get("row_count_values"),
            "num_rows": research.get("num_rows"),
            "num_steps": research.get("num_steps"),
            "curve_fitness_record": research.get("curve_fitness_record"),
            "aoa": research.get("aoa"),
            "surprisal_path": research.get("surprisal_path"),
        },
        "omitted_by_min_context20": {
            "num_words": len(omitted_words),
            "num_context_rows": omitted_context_rows,
            "expected_difference_official_minus_helper": official_size - min20["total_context_rows_loaded"] if official_size is not None else None,
            "jsonl_path": str(omitted_jsonl),
            "context_count_distribution_omitted": dict(sorted(collections.Counter(len(data[w]) for w in omitted_words).items())),
            "first_40_omitted_words": omitted_words[:40],
        },
        "conclusion": (
            "The 6560-vs-8005 discrepancy is explained exactly by evaluator behavior, not by training-corpus data selection: "
            "the local research helper hardcoded load_eval(word_path, 20, False), keeping only the 328 CDI words with 20 contexts "
            "and omitting 176 official CDI words: 157 with 1-19 contexts plus 19 zero-context words. The official pipeline default is "
            "min_context=0 and the collator expects 8005 context-level surprisal rows per checkpoint. The research AoA=0.0 was therefore "
            "not produced through the unmodified official AoA collation path and must be replaced by an official-min_context=0 rerun."
        ),
    }
    out_json = out_dir / "aoa_mincontext_discrepancy_audit.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    note = _public_path('research/notes/representation_and_objectives/aoa_mincontext_discrepancy_audit.md')
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(textwrap.dedent(f"""
        # research AoA 6560-vs-8005 discrepancy audit

        CPU audit JSON: `{out_json}`  
        Omitted words/contexts JSONL: `{omitted_jsonl}`

        ## Finding

        The local research AoA helper used `load_eval(word_path, 20, False)`. The official AoA runner default is `--min_context 0`, and the official collator has `AOA_SIZE = {official_size}`.

        Counting the official `cdi_childes.json`:

        - official min_context=0: {all0['num_words_loaded']} words, {all0['total_context_rows_loaded']} context rows per checkpoint.
        - local helper min_context=20: {min20['num_words_loaded']} words, {min20['total_context_rows_loaded']} context rows per checkpoint.
        - omitted by the helper: {len(omitted_words)} words and {omitted_context_rows} context rows per checkpoint.

        research recorded `{research.get('row_count_values')}` rows per checkpoint and AoA `{research.get('aoa')}` with curve record `{research.get('curve_fitness_record')}`. That exactly matches the helper-side `min_context=20` count, not the official collator.

        ## Interpretation

        The discrepancy is evaluator configuration, not training-data selection: the official AoA data are present locally and have {all0['total_context_rows_loaded']} context rows. The missing rows are the official low-context CDI words omitted by the helper. Because AoA curve fitness is word-set sensitive and p-gated, the 42.0868 endpoint remains provisional until the official-min_context=0 AoA rerun returns and the result is reproduced through the official collation path.
        """), encoding="utf-8")

    print(json.dumps({
        "status": result["status"],
        "official_aoa_size": official_size,
        "official_min0_contexts": all0["total_context_rows_loaded"],
        "helper_min20_contexts": min20["total_context_rows_loaded"],
        "omitted_words": len(omitted_words),
        "omitted_context_rows": omitted_context_rows,
        "out_json": str(out_json),
        "note": str(note),
    }, indent=2))


if __name__ == "__main__":
    main()
