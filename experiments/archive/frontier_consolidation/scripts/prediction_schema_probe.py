#!/usr/bin/env python3
"""Inspect official predictions.json schemas for margin/stratum extraction."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json, pathlib, time
from typing import Any

ROOT = _public_path('experiments/archive/frontier_consolidation/scripts/prediction_schema_probe.py')
ROOT = _PUBLIC_ROOT
WS = _public_path('experiments/archive/frontier_consolidation')
OUT = _public_path('experiments/archive/frontier_consolidation/data/prediction_schema_probe')
PATHS = {
 'clean_blimp': _public_path('experiments/archive/frontier_consolidation/data/deberta_maxgeom_clean_stable_eval/eval/official_outputs/deberta_maxgeom_clean_seed43022_chck_100M/BLiMP/chck_100M/full_deberta_maxgeom_clean_seed43022_chck_100M_BLiMP/zero_shot/mlm/blimp/blimp_filtered/predictions.json'),
 'clean_supplement': _public_path('experiments/archive/frontier_consolidation/data/deberta_maxgeom_clean_stable_eval/eval/official_outputs/deberta_maxgeom_clean_seed43022_chck_100M/Supplement/chck_100M/full_deberta_maxgeom_clean_seed43022_chck_100M_Supplement/zero_shot/mlm/blimp/supplement_filtered/predictions.json'),
 'clean_ewok': _public_path('experiments/archive/frontier_consolidation/data/deberta_maxgeom_clean_stable_eval/eval/official_outputs/deberta_maxgeom_clean_seed43022_chck_100M/EWoK/chck_100M/full_deberta_maxgeom_clean_seed43022_chck_100M_EWoK/zero_shot/mlm/ewok/ewok_filtered/predictions.json'),
 'clean_comps': _public_path('experiments/archive/frontier_consolidation/data/deberta_maxgeom_clean_stable_eval/eval/official_outputs/deberta_maxgeom_clean_seed43022_chck_100M/COMPS/chck_100M/full_deberta_maxgeom_clean_seed43022_chck_100M_COMPS/zero_shot/mlm/comps/comps/predictions.json'),
 'clean_entity': _public_path('experiments/archive/frontier_consolidation/data/deberta_maxgeom_clean_stable_eval/eval/official_outputs/deberta_maxgeom_clean_seed43022_chck_100M/Entity/chck_100M/full_deberta_maxgeom_clean_seed43022_chck_100M_Entity/zero_shot/mlm/entity_tracking/entity_tracking/predictions.json'),
 'adult_ewok_partial': _public_path('experiments/archive/frontier_consolidation/data/gpu_priority_eval/regmax_adultprose_chck_40M/official_outputs/EWoK/chck_40M/EWoK/zero_shot/mlm/ewok/ewok_filtered/predictions.json'),
}

def rel(p: pathlib.Path) -> str:
    try: return str(p.relative_to(ROOT))
    except Exception: return str(p)

def short(v: Any) -> Any:
    if isinstance(v, (int,float,str,bool)) or v is None:
        s = repr(v)
        return s[:300] + ('...' if len(s)>300 else '')
    if isinstance(v, list):
        return {"type":"list", "len":len(v), "first": short(v[0]) if v else None}
    if isinstance(v, dict):
        return {"type":"dict", "keys": list(v.keys())[:30], "sample": {k: short(v[k]) for k in list(v)[:8]}}
    return str(type(v))

def numeric_paths(obj: Any, prefix: str="") -> list[str]:
    out=[]
    if isinstance(obj, dict):
        for k,v in obj.items():
            out += numeric_paths(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        if obj:
            out += numeric_paths(obj[0], prefix+"[0]")
    elif isinstance(obj, (int,float)) and not isinstance(obj, bool):
        out.append(prefix)
    return out

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summaries={"created_utc": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), "paths": {}}
    lines=["# research prediction schema probe", ""]
    for name,path in PATHS.items():
        entry={"path": rel(path), "exists": path.exists()}
        lines += [f"## {name}", f"- path: `{rel(path)}`", f"- exists: {path.exists()}"]
        if path.exists():
            data=json.loads(path.read_text(encoding='utf-8'))
            entry["top_type"] = type(data).__name__
            entry["top_len"] = len(data) if hasattr(data,'__len__') else None
            if isinstance(data, list) and data:
                item=data[0]
            elif isinstance(data, dict) and data:
                item=next(iter(data.values()))
            else:
                item=data
            entry["item_type"] = type(item).__name__
            entry["item_summary"] = short(item)
            entry["numeric_paths_first_item"] = numeric_paths(item)[:100]
            if isinstance(item, dict):
                entry["item_keys"] = list(item.keys())
            lines.append(f"- top: {entry['top_type']} len={entry.get('top_len')}; item: {entry['item_type']}")
            lines.append(f"- numeric paths first item: `{entry['numeric_paths_first_item'][:30]}`")
            lines.append("```json")
            lines.append(json.dumps(entry["item_summary"], indent=2, ensure_ascii=False)[:4000])
            lines.append("```")
        lines.append("")
        summaries["paths"][name]=entry
    (_public_path('experiments/archive/frontier_consolidation/data/prediction_schema_probe/prediction_schema_probe.json')).write_text(json.dumps(summaries, indent=2, ensure_ascii=False)+"\n", encoding='utf-8')
    (_public_path('research/documents/frontier_consolidation/data/prediction_schema_probe/prediction_schema_probe.md')).write_text("\n".join(lines)+"\n", encoding='utf-8')
    print(json.dumps({"status":"SCHEMA_PROBE_DONE", "summary_json": rel(_public_path('experiments/archive/frontier_consolidation/data/prediction_schema_probe/prediction_schema_probe.json')), "summary_md": rel(_public_path('research/documents/frontier_consolidation/data/prediction_schema_probe/prediction_schema_probe.md'))}, indent=2), flush=True)

if __name__ == '__main__': main()
