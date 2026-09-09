#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json, pathlib, time
from statistics import mean
ROOT = _public_path('.')
CHEAP = ["BLiMP","Supplement","EWoK","Entity","COMPS","GlobalPIQA","Reading"]
paths = {
    "anchor80": _public_path('experiments/archive/representation_and_objectives/data/frozen80_eval/per_target/anchor80.json'),
    "coherent80": _public_path('experiments/archive/representation_and_objectives/data/frozen80_eval/per_target/coherent80.json'),
}
outdir = _public_path('experiments/archive/representation_and_objectives/data/fastpath80_pending_status')
outdir.mkdir(parents=True, exist_ok=True)
rows = {}
for name,path in paths.items():
    p=json.loads(path.read_text())
    scores=p["official_overall"]["scores"]
    cheap7=mean(float(scores[c]) for c in CHEAP)
    rows[name]={"path":str(path.relative_to(ROOT)),"scores":{c:scores[c] for c in CHEAP},"cheap7":cheap7}
deltas={c: rows["coherent80"]["scores"][c]-rows["anchor80"]["scores"][c] for c in CHEAP}
deltas["cheap7"] = rows["coherent80"]["cheap7"]-rows["anchor80"]["cheap7"]
summary={"status":"FASTPATH80_PENDING_TWO_ARM_STATUS","created_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),"rows":rows,"coherent_minus_anchor":deltas,"reading":"coherent80 does not beat its own same-trajectory frozen chck_80M anchor on cheap7; four-way interpretation awaits shuffled80 and ordinary84 payloads."}
(_public_path('experiments/archive/representation_and_objectives/data/fastpath80_pending_status/fastpath80_pending_status.json')).write_text(json.dumps(summary, indent=2)+"\n")
md=["# research pending status: frozen80 fast-path", "", f"coherent80 cheap7 {rows['coherent80']['cheap7']:.6f}; anchor80 cheap7 {rows['anchor80']['cheap7']:.6f}; delta {deltas['cheap7']:+.6f}.", "", "| Column | anchor80 | coherent80 | delta |", "|---|---:|---:|---:|"]
for c in CHEAP:
    md.append(f"| {c} | {rows['anchor80']['scores'][c]:.6f} | {rows['coherent80']['scores'][c]:.6f} | {deltas[c]:+.6f} |")
md.append("")
md.append(summary["reading"])
(_public_path('research/documents/representation_and_objectives/data/fastpath80_pending_status/fastpath80_pending_status.md')).write_text("\n".join(md)+"\n")
print(json.dumps({"status":summary["status"],"coherent_minus_anchor_cheap7":deltas["cheap7"],"out":str((_public_path('experiments/archive/representation_and_objectives/data/fastpath80_pending_status/fastpath80_pending_status.json')).relative_to(ROOT))}, indent=2))
