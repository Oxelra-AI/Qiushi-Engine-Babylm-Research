#!/usr/bin/env python3
"""Summarize the research/research natural-packet binding factorial arms."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import argparse, json, pathlib, time
ROOT0=_public_path('experiments/archive/relation_learning/scripts/summarize_binding_factorial.py')
ROOT = _PUBLIC_ROOT
ARMS=['answer_clean','uniform_wwm','answer_corrupt_update_state']
def rel(p:pathlib.Path)->str:
    try: return str(p.relative_to(ROOT))
    except Exception: return str(p)
def load(path): return json.loads(path.read_text(encoding='utf-8'))
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out-root', type=pathlib.Path, default=ROOT/'experiments/archive/relation_learning/data/binding_factorial')
    args=ap.parse_args(); out=args.out_root
    if not out.is_absolute(): out=ROOT/out
    rows=[]
    for arm in ARMS:
        sp=out/arm/'summary.json'
        row={'arm':arm,'summary':rel(sp),'present':sp.exists()}
        if sp.exists():
            s=load(sp)
            row['status']=s.get('status')
            row['checkpoint_dir']=s.get('checkpoint_dir')
            row['updates_seen']=s.get('updates_seen')
            row['n_train_tokenized']=s.get('n_train_tokenized')
            row['n_heldout_tokenized']=s.get('n_heldout_tokenized')
            row['adapter_params_loaded']=(s.get('model_identity') or {}).get('adapter_params_loaded')
            traj=s.get('trajectory') or []
            row['trajectory_epochs']=[x.get('epoch') for x in traj]
            if traj:
                ev=traj[-1].get('eval') or {}
                row.update({
                    'final_epoch':traj[-1].get('epoch'),
                    'binding_joint_correct':ev.get('binding_joint_correct'),
                    'binding_joint_accuracy':ev.get('binding_joint_accuracy'),
                    'binding_mean_joint_min_margin':ev.get('binding_mean_joint_min_margin'),
                    'unchanged_correct':((ev.get('by_role') or {}).get('unchanged_entity') or {}).get('correct_by_margin'),
                    'unchanged_accuracy':((ev.get('by_role') or {}).get('unchanged_entity') or {}).get('accuracy_by_margin'),
                    'unchanged_mean_margin':((ev.get('by_role') or {}).get('unchanged_entity') or {}).get('mean_correct_margin'),
                    'updated_correct':((ev.get('by_role') or {}).get('updated_entity') or {}).get('correct_by_margin'),
                    'updated_accuracy':((ev.get('by_role') or {}).get('updated_entity') or {}).get('accuracy_by_margin'),
                    'updated_mean_margin':((ev.get('by_role') or {}).get('updated_entity') or {}).get('mean_correct_margin'),
                })
        rows.append(row)
    idx={r['arm']:r for r in rows}
    contrasts=[]
    for a,b in [('answer_clean','uniform_wwm'),('answer_clean','answer_corrupt_update_state'),('uniform_wwm','answer_corrupt_update_state')]:
        if idx[a].get('binding_joint_accuracy') is not None and idx[b].get('binding_joint_accuracy') is not None:
            contrasts.append({'contrast':f'{a}-minus-{b}','binding_joint_accuracy_delta':idx[a]['binding_joint_accuracy']-idx[b]['binding_joint_accuracy'],'binding_joint_correct_delta':idx[a]['binding_joint_correct']-idx[b]['binding_joint_correct'],'unchanged_accuracy_delta':idx[a]['unchanged_accuracy']-idx[b]['unchanged_accuracy'],'updated_accuracy_delta':idx[a]['updated_accuracy']-idx[b]['updated_accuracy']})
    summary={'status':'BINDING_FACTORIAL_SUMMARY','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),'out_root':rel(out),'rows':rows,'contrasts':contrasts,'scientific_read':'The interpretable signal is whether answer_clean improves held-out binding-pair joint margins over both uniform_wwm and answer_corrupt_update_state, not whether any arm memorizes the in-format answer slot.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    lines=['# research binding factorial summary','','| arm | joint | pairs | unchanged | updated | updates | checkpoint |','|---|---:|---:|---:|---:|---:|---|']
    for r in rows:
        if not r.get('present'):
            lines.append(f"| {r['arm']} | MISSING | | | | | |")
        else:
            lines.append(f"| {r['arm']} | {r.get('binding_joint_accuracy')} | {r.get('binding_joint_correct')} | {r.get('unchanged_accuracy')} | {r.get('updated_accuracy')} | {r.get('updates_seen')} | {r.get('checkpoint_dir')} |")
    lines += ['','## Contrasts','']+[json.dumps(c,ensure_ascii=False) for c in contrasts]
    (out/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':summary['status'],'summary':rel(out/'summary.json'),'md':rel(out/'summary.md'),'rows':rows,'contrasts':contrasts},indent=2,ensure_ascii=False),flush=True)
if __name__=='__main__': main()
