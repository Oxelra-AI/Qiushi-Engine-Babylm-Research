#!/usr/bin/env python3
"""Materialize separate coherent leash/readout sets for research format replay.

The two sets are fixed coherent rows from the same coherent86 private suffix but are
kept disjoint so the preservation number is not computed on the same examples used
for the KL leash.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json, pathlib, time

ROOT = _public_path('.')
BASE = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl')
OUT = _public_path('experiments/archive/relation_learning/data/held_coherent_sets')
SKIP_ROWS = 530944
N_LEASH = 256
N_READOUT = 256

def rel(p):
    try: return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception: return str(p)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    with BASE.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < SKIP_ROWS:
                continue
            if not line.strip():
                continue
            obj = json.loads(line)
            rows.append(obj)
            if len(rows) >= N_LEASH + N_READOUT:
                break
    if len(rows) < N_LEASH + N_READOUT:
        raise RuntimeError(f"not enough rows: {len(rows)}")
    paths = {
        "leash": _public_path('experiments/archive/relation_learning/data/held_coherent_sets/held_coherent_leash_256rows.jsonl'),
        "readout": _public_path('experiments/archive/relation_learning/data/held_coherent_sets/held_coherent_readout_256rows.jsonl'),
    }
    for key, subset in [("leash", rows[:N_LEASH]), ("readout", rows[N_LEASH:N_LEASH+N_READOUT])]:
        with paths[key].open("w", encoding="utf-8") as g:
            for obj in subset:
                g.write(json.dumps(obj, ensure_ascii=False) + "\n")
    manifest = {
        "status": "HELD_COHERENT_SETS_READY",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_stream": rel(BASE),
        "skip_rows": SKIP_ROWS,
        "sets": {},
        "reason": "leash examples receive gradient through the neutral KL; readout examples are disjoint no-gradient preservation measurement"
    }
    for key, subset in [("leash", rows[:N_LEASH]), ("readout", rows[N_LEASH:N_LEASH+N_READOUT])]:
        manifest["sets"][key] = {
            "path": rel(paths[key]),
            "rows": len(subset),
            "words": sum(int(o.get("words", len(str(o.get("text", "")).split()))) for o in subset),
            "mean_words": sum(int(o.get("words", len(str(o.get("text", "")).split()))) for o in subset)/len(subset),
            "first_example_id": subset[0].get("example_id"),
            "last_example_id": subset[-1].get("example_id"),
        }
    (_public_path('experiments/archive/relation_learning/data/held_coherent_sets/manifest.json')).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
