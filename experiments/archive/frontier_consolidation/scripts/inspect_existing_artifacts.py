#!/usr/bin/env python3
"""research: quick structural inspection of BabyLM eval data and saved prediction files.
This script is read-only and writes a compact JSON/MD summary for choosing the
next no-new-training analyses."""
from __future__ import annotations

import csv
import json
import pathlib
from collections import OrderedDict

ROOT = pathlib.Path('.').resolve()
OUT_DIR = pathlib.Path('experiments/archive/frontier_consolidation/data/existing_artifact_inspection')
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLES = OrderedDict([
    ('eval_ewok', pathlib.Path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/ewok_filtered/physical-dynamics.jsonl')),
    ('eval_blimp', pathlib.Path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/blimp_filtered/wh_island.jsonl')),
    ('eval_supplement', pathlib.Path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/supplement_filtered/qa_congruence_tricky.jsonl')),
    ('eval_entity', pathlib.Path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/entity_tracking/regular.jsonl')),
    ('eval_comps', pathlib.Path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/comps/comps_base.jsonl')),
    ('eval_globalpiqa_parallel', pathlib.Path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/global_piqa_parallel/eng_latn.jsonl')),
    ('eval_globalpiqa_nonparallel', pathlib.Path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/global_piqa_nonparallel/eng_latn.jsonl')),
    ('eval_reading', pathlib.Path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/reading/reading_data.csv')),
    ('pred_blimp70', pathlib.Path('experiments/archive/frontier_consolidation/data/legal_mature_treatment_effect_eval/official_outputs/complianttok_reinvest_seed43022_70M/BLiMP/chck_70M/full_complianttok_reinvest_seed43022_70M_BLiMP/zero_shot/mlm/blimp/blimp_filtered/predictions.json')),
    ('pred_ewok70', pathlib.Path('experiments/archive/frontier_consolidation/data/legal_mature_treatment_effect_eval/official_outputs/complianttok_reinvest_seed43022_70M/EWoK/chck_70M/full_complianttok_reinvest_seed43022_70M_EWoK/zero_shot/mlm/ewok/ewok_filtered/predictions.json')),
    ('pred_entity70', pathlib.Path('experiments/archive/frontier_consolidation/data/legal_mature_treatment_effect_eval/official_outputs/complianttok_reinvest_seed43022_70M/Entity/chck_70M/full_complianttok_reinvest_seed43022_70M_Entity/zero_shot/mlm/entity_tracking/entity_tracking/predictions.json')),
    ('pred_comps70', pathlib.Path('experiments/archive/frontier_consolidation/data/legal_mature_treatment_effect_eval/official_outputs/complianttok_reinvest_seed43022_70M/COMPS/chck_70M/full_complianttok_reinvest_seed43022_70M_COMPS/zero_shot/mlm/comps/comps/predictions.json')),
    ('pred_globalpiqa70', pathlib.Path('experiments/archive/frontier_consolidation/data/legal_mature_treatment_effect_eval/official_outputs/complianttok_reinvest_seed43022_70M/GlobalPIQA_parallel/chck_70M/full_complianttok_reinvest_seed43022_70M_GlobalPIQA_parallel/zero_shot/mlm/global_piqa_parallel/global_piqa_parallel/predictions.json')),
    ('pred_reading70', pathlib.Path('experiments/archive/frontier_consolidation/data/legal_mature_treatment_effect_eval/official_outputs/complianttok_reinvest_seed43022_70M/Reading/chck_70M/full_complianttok_reinvest_seed43022_70M_Reading/zero_shot/mlm/reading/predictions.json')),
    ('pred_blimp100', pathlib.Path('experiments/archive/frontier_consolidation/data/compliant_full_eval/official_outputs/complianttok_reinvest_seed43022/BLiMP/chck_100M/full_complianttok_reinvest_seed43022_BLiMP/zero_shot/mlm/blimp/blimp_filtered/predictions.json')),
    ('aoa_score100', pathlib.Path('experiments/archive/frontier_consolidation/data/compliant_full_eval/aoa_outputs/complianttok_reinvest_seed43022/hf_model_local_ckpts/main/zero_shot/mlm/AoA_word/aoa_score.json')),
    ('aoa_surprisal', pathlib.Path('experiments/archive/frontier_consolidation/data/compliant_full_eval/aoa_outputs/complianttok_reinvest_seed43022/hf_model_local_ckpts/main/zero_shot/mlm/AoA_word/surprisal.json')),
])

def summarize_obj(obj, depth=0):
    if depth >= 2:
        return repr(obj)[:300]
    if isinstance(obj, dict):
        out = {'type': 'dict', 'len': len(obj), 'keys': list(obj.keys())[:20]}
        samples = []
        for k, v in list(obj.items())[:3]:
            samples.append({'key': k, 'value': summarize_obj(v, depth+1)})
        out['samples'] = samples
        return out
    if isinstance(obj, list):
        return {'type': 'list', 'len': len(obj), 'first': summarize_obj(obj[0], depth+1) if obj else None}
    return {'type': type(obj).__name__, 'repr': repr(obj)[:300]}

def read_jsonl_sample(path, n=2):
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            rows.append(json.loads(line))
    return {'row_count_first_counted': i+1 if 'i' in locals() else 0, 'samples': [summarize_obj(r) for r in rows]}

def read_csv_sample(path, n=2):
    with path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        rows = []
        for i, row in enumerate(reader):
            if i >= n:
                break
            rows.append(row)
    return {'fieldnames': reader.fieldnames, 'samples': [summarize_obj(r) for r in rows]}

def main():
    result = OrderedDict()
    for name, path in SAMPLES.items():
        rec = OrderedDict(path=str(path), exists=path.exists(), size=path.stat().st_size if path.exists() else None)
        if path.exists():
            try:
                if path.suffix == '.jsonl':
                    rec['kind'] = 'jsonl'
                    rec['summary'] = read_jsonl_sample(path)
                elif path.suffix == '.csv':
                    rec['kind'] = 'csv'
                    rec['summary'] = read_csv_sample(path)
                elif path.suffix == '.json':
                    rec['kind'] = 'json'
                    with path.open('r', encoding='utf-8') as f:
                        rec['summary'] = summarize_obj(json.load(f))
                else:
                    rec['kind'] = path.suffix
            except Exception as e:
                rec['error'] = repr(e)
        result[name] = rec
    out_json = OUT_DIR / 'artifact_structure_summary.json'
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    md = ['# research existing artifact structure inspection', '']
    for name, rec in result.items():
        md.append(f'## {name}')
        md.append(f"- path: `{rec['path']}`")
        md.append(f"- exists: {rec['exists']} size: {rec['size']}")
        if 'error' in rec:
            md.append(f"- error: `{rec['error']}`")
        else:
            md.append('```json')
            md.append(json.dumps(rec.get('summary'), indent=2, ensure_ascii=False)[:2500])
            md.append('```')
        md.append('')
    out_md = (ROOT / 'research/documents/frontier_consolidation/data/existing_artifact_inspection/artifact_structure_summary.md')
    out_md.write_text('\n'.join(md), encoding='utf-8')
    print(json.dumps({'status':'ok','out_json':str(out_json),'out_md':str(out_md)}, indent=2))

if __name__ == '__main__':
    main()
