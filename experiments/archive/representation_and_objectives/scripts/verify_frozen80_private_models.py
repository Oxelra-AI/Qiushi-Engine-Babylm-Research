#!/usr/bin/env python3
"""Verify frozen-80M private replay model identity and exposure accounting."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import hashlib, json, pathlib, time
from safetensors import safe_open

ROOT = _public_path('.')
ANCHOR = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_80M/model.safetensors')
ARMS = {
    "coherent80": _public_path('experiments/archive/representation_and_objectives/training/runs/frozen80_fastpath4M_coherent_seed43022'),
    "shuffled80": _public_path('experiments/archive/representation_and_objectives/training/runs/frozen80_fastpath4M_shuffled_seed43022'),
}
OUT = _public_path('experiments/archive/representation_and_objectives/data/frozen80_identity/frozen80_identity.json')


def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8<<20), b''): h.update(b)
    return h.hexdigest()


def rel(p):
    try: return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception: return str(p)


def tensor_map(path):
    out={}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for k in f.keys(): out[k]=f.get_tensor(k)
    return out


def main():
    import torch
    anchor=tensor_map(ANCHOR)
    result={"status":"PASS","created_utc":time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            "anchor":{"path":rel(ANCHOR),"sha256":sha256(ANCHOR),"tensor_count":len(anchor)},"arms":{}}
    for name, run in ARMS.items():
        model=run/"hf_model/final/model.safetensors"
        metrics=json.loads((run/"scientific_metrics.json").read_text())
        cand=tensor_map(model)
        private=[k for k in cand if ".private_adapter." in k]
        nonprivate=[k for k in cand if k not in private]
        missing=sorted(set(anchor)-set(nonprivate))
        extra=sorted(set(nonprivate)-set(anchor))
        unequal=[]
        maxdiff=0.0
        for k in sorted(set(anchor)&set(nonprivate)):
            if not torch.equal(anchor[k],cand[k]):
                unequal.append(k)
                maxdiff=max(maxdiff,float((anchor[k].float()-cand[k].float()).abs().max()))
        ok=not missing and not extra and not unequal and len(private)>0
        result["arms"][name]={
            "model_path":rel(model),"sha256":sha256(model),"private_tensor_count":len(private),
            "private_param_count":sum(cand[k].numel() for k in private),
            "nonprivate_tensor_count":len(nonprivate),"missing_nonprivate":missing,"extra_nonprivate":extra,
            "unequal_nonprivate":unequal,"nonprivate_max_abs_diff":maxdiff,"nonprivate_exact_equal":ok,
            "training_metrics":{k:metrics.get(k) for k in ["mode","updates","tail_charged_words","total_consumed_words","first_main_loss","final_main_loss","first_neutral_loss","final_neutral_loss","mean_neutral_loss"]}}
        if not ok: result["status"]="FAIL"
    a=result["arms"]["coherent80"]["training_metrics"]
    b=result["arms"]["shuffled80"]["training_metrics"]
    result["matched_accounting"]={
        "same_updates":a["updates"]==b["updates"],"same_tail_words":a["tail_charged_words"]==b["tail_charged_words"],
        "same_total_words":a["total_consumed_words"]==b["total_consumed_words"],
        "ordinary84_actual_exposure":84028405,
        "exposure_gap_vs_ordinary84":84028405-int(a["total_consumed_words"])}
    if not all([result["matched_accounting"][k] for k in ["same_updates","same_tail_words","same_total_words"]]): result["status"]="FAIL"
    _public_path('experiments/archive/representation_and_objectives/data/frozen80_identity').mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"status":result["status"],"out":rel(OUT),"matched_accounting":result["matched_accounting"],
                      "coherent_sha":result["arms"]["coherent80"]["sha256"],"shuffled_sha":result["arms"]["shuffled80"]["sha256"]},indent=2))

if __name__=="__main__": main()
