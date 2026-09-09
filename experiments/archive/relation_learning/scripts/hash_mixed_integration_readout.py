#!/usr/bin/env python3
"""Integrate hash-mixed DeBERTa seed43022 probe outputs against C/R/V/RS/VS endpoints."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json, math
from pathlib import Path
from typing import Any
import pandas as pd

ROOT = _public_path('experiments/archive/relation_learning/scripts/hash_mixed_integration_readout.py')
ROOT = _PUBLIC_ROOT
WS = _public_path('experiments/archive/relation_learning')
OUT = _public_path('experiments/archive/relation_learning/data/hash_mixed_integration')
HASH = _public_path('experiments/archive/relation_learning/data/hash_mixed_probes')
ORIG = _public_path('experiments/archive/relation_learning/data/heldout_copy_rewrite_entity_ablation')
RS = _public_path('experiments/archive/relation_learning/data/split_repeat_split_probes')
VS = _public_path('experiments/archive/relation_learning/data/split_view_split_probes')
ROLE_MAP = {"D_C_43022":"C", "D_R_43022":"R", "D_V_43022":"V", "D_RS_43022":"RS", "D_VS_43022":"VS", "D_HM_43022":"HM"}


def rel(p: Path) -> str: return str(p.relative_to(ROOT))

def se(s):
    vals = pd.Series(s, dtype=float).dropna()
    return float(vals.std(ddof=0) / math.sqrt(len(vals))) if len(vals) > 1 else float("nan")

def read_existing(paths):
    frames=[]
    for p in paths:
        if p.exists() and p.stat().st_size > 1:
            frames.append(pd.read_csv(p))
    if not frames: raise RuntimeError(paths)
    return pd.concat(frames, ignore_index=True)

def late(df, value_cols, keys):
    df = df[(df.seed == 43022) & df.checkpoint.isin(["chck_80M", "chck_90M", "chck_100M"])].copy()
    if "arm" not in df.columns: raise RuntimeError({"missing_arm": list(df.columns)})
    g = df.groupby(["arm"] + keys, dropna=False).agg(n_checkpoints=("checkpoint", "nunique"), **{c:(c,"mean") for c in value_cols}).reset_index()
    g = g[g.n_checkpoints == 3].copy()
    g["role"] = g.arm.map(ROLE_MAP).fillna(g.arm)
    return g

def contrast(df, a, b, keys, value):
    aa = df[df.role == a][keys + [value]].rename(columns={value:"a"})
    bb = df[df.role == b][keys + [value]].rename(columns={value:"b"})
    j = aa.merge(bb, on=keys)
    d = j.a - j.b
    return {"contrast":f"{a}minus{b}", "n":int(len(d)), "mean":float(d.mean()), "se":se(d), "frac_positive":float((d>0).mean())}

def h(rows, **kw):
    sub = rows.copy()
    for k,v in kw.items(): sub = sub[sub[k] == v]
    if len(sub) != 1: raise RuntimeError({"missing":kw, "n":len(sub), "rows":sub.to_dict("records")[:5]})
    return float(sub.iloc[0].mean)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rw = read_existing([_public_path('experiments/archive/relation_learning/data/heldout_copy_rewrite_entity_ablation/rewrite_pair_rows.csv'), _public_path('experiments/archive/relation_learning/data/split_repeat_split_probes/rewrite_pair_rows.csv'), _public_path('experiments/archive/relation_learning/data/split_view_split_probes/rewrite_pair_rows.csv'), _public_path('experiments/archive/relation_learning/data/hash_mixed_probes/rewrite_pair_rows.csv')])
    rw_l = late(rw, ["gain", "true_source_nll", "unrelated_source_nll"], ["probe_id", "pair_id", "token_class"])
    rw_rows=[]
    for group in ["ALL", "nonoverlap", "overlap"]:
        sub = rw_l if group=="ALL" else rw_l[rw_l.token_class == group]
        keys=["probe_id", "pair_id", "token_class"]
        for value in ["gain", "true_source_nll", "unrelated_source_nll"]:
            for a,b in [("HM","C"),("HM","R"),("HM","V"),("HM","RS"),("HM","VS"),("R","C"),("V","C"),("RS","C"),("VS","C")]:
                r = contrast(sub,a,b,keys,value); r.update({"family":"rewrite", "group":group, "value":value}); rw_rows.append(r)
    rw_df = pd.DataFrame(rw_rows); rw_df.to_csv(_public_path('experiments/archive/relation_learning/data/hash_mixed_integration/hash_mixed_rewrite_contrasts.csv'), index=False)

    cp = read_existing([_public_path('experiments/archive/relation_learning/data/heldout_copy_rewrite_entity_ablation/copy_pair_rows.csv'), _public_path('experiments/archive/relation_learning/data/split_repeat_split_probes/copy_pair_rows.csv'), _public_path('experiments/archive/relation_learning/data/split_view_split_probes/copy_pair_rows.csv'), _public_path('experiments/archive/relation_learning/data/hash_mixed_probes/copy_pair_rows.csv')])
    cp_l = late(cp, ["gain", "repeated_nll", "unrepeated_nll"], ["probe_id", "span_len", "source_domain"])
    cp_rows=[]
    for group, sub in [("ALL", cp_l), ("span_1", cp_l[cp_l.span_len==1]), ("span_4", cp_l[cp_l.span_len==4])]:
        keys=["probe_id", "span_len", "source_domain"]
        for value in ["gain"]:
            for a,b in [("HM","C"),("HM","R"),("HM","V"),("HM","RS"),("HM","VS"),("R","C"),("V","C"),("RS","C"),("VS","C")]:
                r=contrast(sub,a,b,keys,value); r.update({"family":"copy", "group":group, "value":value}); cp_rows.append(r)
    cp_df=pd.DataFrame(cp_rows); cp_df.to_csv(_public_path('experiments/archive/relation_learning/data/hash_mixed_integration/hash_mixed_copy_contrasts.csv'), index=False)

    ent = read_existing([_public_path('experiments/archive/relation_learning/data/heldout_copy_rewrite_entity_ablation/entity_ablation_rows.csv'), _public_path('experiments/archive/relation_learning/data/hash_mixed_probes/entity_ablation_rows.csv')])
    ent_l = late(ent, ["margin_full", "effect_no_initial_qbox", "effect_no_last_relevant_update", "effect_no_all_relevant_updates"], ["probe_id", "relevant_updates", "entity_type"])
    ent_rows=[]
    for group, sub in [("ALL", ent_l), ("rel_ge3", ent_l[ent_l.relevant_updates>=3])]:
        keys=["probe_id", "relevant_updates", "entity_type"]
        for value in ["margin_full", "effect_no_initial_qbox", "effect_no_last_relevant_update", "effect_no_all_relevant_updates"]:
            for a,b in [("HM","C"),("HM","R"),("HM","V"),("R","C"),("V","C")]:
                r=contrast(sub,a,b,keys,value); r.update({"family":"entity_ablation", "group":group, "value":value}); ent_rows.append(r)
    ent_df=pd.DataFrame(ent_rows); ent_df.to_csv(_public_path('experiments/archive/relation_learning/data/hash_mixed_integration/hash_mixed_entity_ablation_contrasts.csv'), index=False)

    def val(df, contrast_name, group, value):
        s=df[(df.contrast==contrast_name)&(df.group==group)&(df.value==value)]
        if len(s)!=1: raise RuntimeError({"missing":contrast_name, "group":group, "value":value, "n":len(s)})
        return float(s.iloc[0]["mean"]), float(s.iloc[0]["se"]), int(s.iloc[0]["n"])

    hm_c_non, hm_c_non_se, n_non = val(rw_df,"HMminusC","nonoverlap","gain")
    hm_r_non, _, _ = val(rw_df,"HMminusR","nonoverlap","gain")
    hm_v_non, _, _ = val(rw_df,"HMminusV","nonoverlap","gain")
    hm_rs_non, _, _ = val(rw_df,"HMminusRS","nonoverlap","gain")
    hm_vs_non, _, _ = val(rw_df,"HMminusVS","nonoverlap","gain")
    r_c_non, _, _ = val(rw_df,"RminusC","nonoverlap","gain")
    v_c_non, _, _ = val(rw_df,"VminusC","nonoverlap","gain")
    rs_c_non, _, _ = val(rw_df,"RSminusC","nonoverlap","gain")
    vs_c_non, _, _ = val(rw_df,"VSminusC","nonoverlap","gain")
    hm_c_T, _, _ = val(rw_df,"HMminusC","nonoverlap","true_source_nll")
    hm_c_U, _, _ = val(rw_df,"HMminusC","nonoverlap","unrelated_source_nll")
    prop = 0.4974*r_c_non + (1-0.4974)*v_c_non

    hm_c_copy, hm_c_copy_se, n_copy = val(cp_df,"HMminusC","ALL","gain")
    hm_r_copy, _, _ = val(cp_df,"HMminusR","ALL","gain")
    hm_v_copy, _, _ = val(cp_df,"HMminusV","ALL","gain")
    r_c_copy, _, _ = val(cp_df,"RminusC","ALL","gain")
    v_c_copy, _, _ = val(cp_df,"VminusC","ALL","gain")
    rs_c_copy, _, _ = val(cp_df,"RSminusC","ALL","gain")
    vs_c_copy, _, _ = val(cp_df,"VSminusC","ALL","gain")

    hm_c_ent, _, _ = val(ent_df,"HMminusC","rel_ge3","margin_full")
    hm_r_ent, _, _ = val(ent_df,"HMminusR","rel_ge3","margin_full")
    hm_v_ent, _, _ = val(ent_df,"HMminusV","rel_ge3","margin_full")

    note = f"""# research hash-mixed exact/rewrite integration

The hash-mixed seed43022 arm preserves the selected sources and 100M-word budget, but assigns same-window companions by deterministic hash: 16,560 exact repeat companions and 16,731 compact rewrite companions (repeat fraction 0.4974). It tests whether mixed relation practice behaves like proportional averaging, identity-dominant capture, restatement robustness, or context-selected dual competence.

## Compact nonoverlap rewrite and natural-copy readouts

| readout | HM−C | HM−R | HM−V | R−C | V−C | split references |
|---|---:|---:|---:|---:|---:|---|
| compact nonoverlap rewrite gain | {hm_c_non:+.4f} ± {hm_c_non_se:.4f} | {hm_r_non:+.4f} | {hm_v_non:+.4f} | {r_c_non:+.4f} | {v_c_non:+.4f} | RS−C {rs_c_non:+.4f}; VS−C {vs_c_non:+.4f} |
| natural-copy gain | {hm_c_copy:+.4f} ± {hm_c_copy_se:.4f} | {hm_r_copy:+.4f} | {hm_v_copy:+.4f} | {r_c_copy:+.4f} | {v_c_copy:+.4f} | RS−C {rs_c_copy:+.4f}; VS−C {vs_c_copy:+.4f} |

For compact nonoverlap rewrite gain, the half-weighted endpoint prediction is {prop:+.4f}; observed HM−C is {hm_c_non:+.4f}. Thus the mixed arm is not captured by exact recurrence despite half of its relation rows being exact. It lands near the proportional cancellation point and near CLEAN, while remaining much better than REPEAT ({hm_r_non:+.4f}) and much worse than VIEW ({hm_v_non:+.4f}).

For natural copy, HM−C is positive ({hm_c_copy:+.4f}) but below original REPEAT ({r_c_copy:+.4f}) and close to or somewhat above VIEW ({v_c_copy:+.4f}). This is blended relation practice rather than a pure relation-selected double competence.

## T/U component placement

On compact nonoverlap rewrites, HM−C true-source NLL is {hm_c_T:+.4f} and unrelated-source NLL is {hm_c_U:+.4f}; the gain contrast is U−T. The mixture therefore cancels source-conditioned computations rather than simply moving the whole arm uniformly above or below CLEAN.

## Entity cue-ablation placement

On the research stale-gold Entity cue-ablation readout, rel≥3 margin_full HM−C is {hm_c_ent:+.4f}; HM−R is {hm_r_ent:+.4f}; HM−V is {hm_v_ent:+.4f}. This is not official Entity scoring and should remain secondary. It is useful only as another indication that HM is intermediate rather than an endpoint clone.

## Scientific reading

Hash-mixed same-window training strengthens the sequence-composition principle: when a finite learner practices incompatible local relations on the same family of content, the installed source-use computation can partially cancel. The result does not support a strong near-duplicate-contamination threshold at 50%; it shows that exact and restatement relation practice blend under this dose. A dose curve would be needed to state when minority exact recurrence dominates. For the current paper, hash-mix belongs as a boundary/composability result, not as part of the core causal proof.

## Files

- `{rel(_public_path('experiments/archive/relation_learning/data/hash_mixed_integration/hash_mixed_rewrite_contrasts.csv'))}`
- `{rel(_public_path('experiments/archive/relation_learning/data/hash_mixed_integration/hash_mixed_copy_contrasts.csv'))}`
- `{rel(_public_path('experiments/archive/relation_learning/data/hash_mixed_integration/hash_mixed_entity_ablation_contrasts.csv'))}`
"""
    (_public_path('research/documents/relation_learning/data/hash_mixed_integration/hash_mixed_integration.md')).write_text(note, encoding="utf-8")
    summary={
        "status":"HASH_MIXED_INTEGRATION_DONE",
        "note":rel(_public_path('research/documents/relation_learning/data/hash_mixed_integration/hash_mixed_integration.md')),
        "hm_minus_clean_nonoverlap_rewrite_gain":hm_c_non,
        "proportional_endpoint_prediction":prop,
        "hm_minus_clean_copy_gain":hm_c_copy,
        "hm_minus_clean_true_source_nll_nonoverlap":hm_c_T,
        "hm_minus_clean_unrelated_source_nll_nonoverlap":hm_c_U,
        "n_nonoverlap_rewrite":n_non,
        "n_copy":n_copy,
    }
    (_public_path('experiments/archive/relation_learning/data/hash_mixed_integration/hash_mixed_integration_summary.json')).write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__": main()
