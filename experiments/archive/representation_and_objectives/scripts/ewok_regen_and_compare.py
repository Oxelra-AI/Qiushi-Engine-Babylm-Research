#!/usr/bin/env python3
"""Regenerate official EWoK filtered files from the pristine coordinate and compare
counts against local vendored data, collator constants, and raw source counts.

This isolates whether the EWoK size mismatch is local corruption or an upstream
code/data inconsistency in the pinned official release. No official code or data is
patched; generation uses the pristine checkout's own vocab and dl_and_filter logic.
"""

from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import ast
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = _public_path('.')
OUT = _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate')
STRICT = _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict')
NLTK = _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/nltk_data')
RAW_PARQUET = _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/raw_ewok_download/repo/data/test/ewok-core-1.0.parquet')
LOCAL_EWOK = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/ewok_filtered')


def norm(d: dict) -> dict:
    return dict(sorted(d.items()))


def parse_const(py: Path, name: str) -> dict:
    tree = ast.parse(py.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tg in node.targets:
                if isinstance(tg, ast.Name) and tg.id == name:
                    return {k: int(v) for k, v in ast.literal_eval(node.value).items()}
    return {}


def dict_equal(a: dict, b: dict) -> bool:
    return all(a.get(k) == b.get(k) for k in set(a) | set(b))


def main() -> None:
    import nltk
    nltk.data.path.insert(0, str(NLTK))
    from nltk.tokenize import word_tokenize
    import pandas as pd

    vocab_path = _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_pipeline/ewok/vocab.txt')
    vocab = {l.strip() for l in vocab_path.read_text(encoding="utf-8").splitlines() if l.strip()}

    df = pd.read_parquet(RAW_PARQUET)
    raw: Counter = Counter()
    filt: Counter = Counter()
    items_per_domain: dict[str, list] = {}
    for _, ex in df.iterrows():
        domain = ex["Domain"]
        raw[domain] += 2
        skip = False
        for key in ("Context1", "Context2", "Target1", "Target2"):
            for w in word_tokenize(str(ex[key]).lower()):
                if w not in vocab:
                    skip = True
                    break
            if skip:
                break
        if not skip:
            items_per_domain.setdefault(domain, []).append(ex)
            filt[domain] += 2

    raw = norm(dict(raw))
    filt = norm(dict(filt))

    gen_dir = _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/ewok_filtered')
    gen_dir.mkdir(parents=True, exist_ok=True)
    gen_counts: dict[str, int] = {}
    for domain, items in items_per_domain.items():
        p = gen_dir / f"{domain}.jsonl"
        with p.open("w", encoding="utf-8") as f:
            for item in items:
                rec = {k: item[k] for k in df.columns}
                f.write(json.dumps(rec, default=str) + "\n")
                sw = dict(rec)
                sw["Context1"], sw["Context2"] = sw["Context2"], sw["Context1"]
                sw["Target1"], sw["Target2"] = sw["Target2"], sw["Target1"]
                f.write(json.dumps(sw, default=str) + "\n")
        gen_counts[domain] = sum(1 for _ in p.open())
    gen_counts = norm(gen_counts)

    local_counts = norm({p.stem: sum(1 for _ in p.open()) for p in sorted(LOCAL_EWOK.glob("*.jsonl"))})
    const = norm(parse_const(_public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_pipeline/collate_preds.py'), "EWOK_SIZES"))

    keys = sorted(set(const) | set(gen_counts) | set(local_counts) | set(raw) | set(filt))
    rows = [
        {
            "domain": k,
            "collator_const": const.get(k),
            "pristine_gen": gen_counts.get(k),
            "local": local_counts.get(k),
            "raw_x2": raw.get(k),
            "filtered_x2": filt.get(k),
        }
        for k in keys
    ]

    result = {
        "status": "EWOK_REGEN_AND_COMPARE",
        "ewok_raw_parquet_sha256": hashlib.sha256(RAW_PARQUET.read_bytes()).hexdigest(),
        "vocab_size": len(vocab),
        "collator_EWOK_SIZES": const,
        "pristine_generated_counts": gen_counts,
        "local_counts": local_counts,
        "raw_source_x2": raw,
        "filtered_source_x2_recomputed": filt,
        "rows": rows,
        "pristine_gen_equals_local": dict_equal(gen_counts, local_counts),
        "pristine_gen_equals_filtered_recomputed": dict_equal(gen_counts, filt),
        "collator_equals_raw": dict_equal(const, raw),
        "collator_equals_filtered": dict_equal(const, filt),
        "collator_equals_pristine_gen": dict_equal(const, gen_counts),
        "total_pristine_gen": sum(gen_counts.values()),
        "total_local": sum(local_counts.values()),
        "total_collator_const": sum(const.values()),
    }
    outp = _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/ewok_regen_and_compare.json')
    outp.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in [
        "pristine_gen_equals_local",
        "pristine_gen_equals_filtered_recomputed",
        "collator_equals_raw",
        "collator_equals_filtered",
        "collator_equals_pristine_gen",
        "total_pristine_gen",
        "total_local",
        "total_collator_const",
    ]}, indent=2))
    for r in rows:
        print(r)
    print("out", str(outp.relative_to(ROOT)))


if __name__ == "__main__":
    main()
