#!/usr/bin/env python3
"""research R1 train-vocab vs heldout-vocab MLM-head readout.

The first research curve used held-out vocabulary only and gave pair-summed margins
~0. This script distinguishes two explanations:
  1. no learning of dynamic state dependence even on the generated vocabulary;
  2. learning on seen generated words but no transfer to held-out words/templates.

For each split, it scores the same frozen MLM-head true-vs-counterfactual metric
on protected reference and ordered/static checkpoints.
"""
from __future__ import annotations
import argparse, importlib.util, json, pathlib, sys, time
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
GEN = ROOT / "scripts/r1_generator_v3.py"
BASE_EVAL = ROOT / "scripts/eval_r1_mlm_counterfactual.py"
spec = importlib.util.spec_from_file_location("r1gen", GEN)
r1gen = importlib.util.module_from_spec(spec); sys.modules[spec.name] = r1gen; spec.loader.exec_module(r1gen)
spec2 = importlib.util.spec_from_file_location("baseeval", BASE_EVAL)
baseeval = importlib.util.module_from_spec(spec2); sys.modules[spec2.name] = baseeval; spec2.loader.exec_module(baseeval)

PROTECTED = ROOT / "training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/hf_model/chck_10M"
ORDERED_ROOT = ROOT / "training/runs/r1_ordered_dynamic_5M/hf_model"
STATIC_ROOT = ROOT / "training/runs/r1_static_indirect_control_5M/hf_model"
OUT = ROOT / "data/r1_vocab_split_learning_curve.json"

TRAIN_LOCS = ["the kitchen", "the garage", "the attic", "the cellar", "the study", "the porch", "the closet"]
TRAIN_ITEMS = ["the lamp", "the clock", "the vase", "the mirror", "the rug", "the painting", "the cushion", "the blanket", "the candle", "the plant", "the photo", "the trophy", "the statue", "the basket"]
HELDOUT_LOCS = ["the shed", "the vault", "the loft", "the hallway", "the pantry"]
HELDOUT_ITEMS = ["the fan", "the radio", "the stool", "the hammer", "the broom", "the kettle"]


def set_vocab(split: str):
    if split == "train_vocab":
        r1gen.LOCATIONS = list(TRAIN_LOCS); r1gen.ITEMS = list(TRAIN_ITEMS)
    elif split == "heldout_vocab":
        r1gen.LOCATIONS = list(HELDOUT_LOCS); r1gen.ITEMS = list(HELDOUT_ITEMS)
    else:
        raise ValueError(split)


def generate_equal_len_records(split: str, tokenizer, n_pairs: int, seed: int):
    set_vocab(split)
    recs = []
    cur_seed = seed
    batches = 0
    while len(recs) < n_pairs and batches < 80:
        batches += 1
        pairs = r1gen.generate_paired_corpus(n_pairs=max(n_pairs, 100), seed=cur_seed)
        cur_seed += 7919
        for a, b in pairs:
            if a.answer == b.answer:
                continue
            a_ids = tokenizer(a.answer, add_special_tokens=False)["input_ids"]
            b_ids = tokenizer(b.answer, add_special_tokens=False)["input_ids"]
            if len(a_ids) != len(b_ids):
                continue
            recs.append({
                "passage_a": a.passage, "passage_b": b.passage,
                "answer_a": a.answer, "answer_b": b.answer,
                "answer_token_len": len(a_ids), "query_obj": a.query_obj,
                "word_count_a": a.word_count, "word_count_b": b.word_count,
            })
            if len(recs) >= n_pairs: break
    if len(recs) < n_pairs:
        raise RuntimeError(f"{split}: only {len(recs)} records")
    return recs


def score_model(model_path: pathlib.Path, tokenizer, records, device):
    model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True).to(device).eval()
    margins=[]; pair_sums=[]; both_correct=0; one_correct=0; correct_cases=0; rows=[]
    t0=time.time()
    for i,r in enumerate(records):
        mt_a,_=baseeval.make_masked(r["passage_a"], r["answer_a"], tokenizer)
        mt_b,_=baseeval.make_masked(r["passage_b"], r["answer_b"], tokenizer)
        a_true=baseeval.answer_logprob(model,tokenizer,mt_a,r["answer_a"],device)
        a_cf=baseeval.answer_logprob(model,tokenizer,mt_a,r["answer_b"],device)
        b_true=baseeval.answer_logprob(model,tokenizer,mt_b,r["answer_b"],device)
        b_cf=baseeval.answer_logprob(model,tokenizer,mt_b,r["answer_a"],device)
        ma=a_true-a_cf; mb=b_true-b_cf
        ca=ma>0; cb=mb>0
        correct_cases += int(ca)+int(cb)
        both_correct += int(ca and cb)
        one_correct += int(ca)+int(cb)==1
        margins.extend([ma,mb]); pair_sums.append(ma+mb)
        if i<8: rows.append({"i":i,"answer_a":r["answer_a"],"answer_b":r["answer_b"],"ma":ma,"mb":mb,"pair_sum":ma+mb,"ca":ca,"cb":cb})
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    n=len(records); nc=2*n
    return {
        "model_path": str(model_path), "n_pairs": n, "n_cases": nc,
        "case_accuracy": correct_cases/max(1,nc),
        "both_contexts_correct_pair_fraction": both_correct/max(1,n),
        "exactly_one_context_correct_pair_fraction": one_correct/max(1,n),
        "mean_case_margin": sum(margins)/max(1,nc),
        "mean_pair_sum_margin": sum(pair_sums)/max(1,n),
        "median_pair_sum_margin": sorted(pair_sums)[n//2] if pair_sums else None,
        "positive_pair_sum_fraction": sum(1 for x in pair_sums if x>0)/max(1,n),
        "elapsed_sec": round(time.time()-t0,1), "sample_rows": rows,
    }


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--n_pairs",type=int,default=200); ap.add_argument("--seed",type=int,default=2000); ap.add_argument("--output_json",default=str(OUT)); args=ap.parse_args()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer=AutoTokenizer.from_pretrained(PROTECTED,use_fast=True)
    if tokenizer.mask_token is None: tokenizer.mask_token="<mask>"
    payload={"status":"R1_VOCAB_SPLIT_CURVE","readout":"pair-summed frozen MLM-head margin; both contexts must be correct for state-sensitive success","device":str(device),"n_pairs":args.n_pairs,"seed":args.seed,"splits":{}}
    out=pathlib.Path(args.output_json); out.parent.mkdir(parents=True,exist_ok=True)
    for split in ["train_vocab","heldout_vocab"]:
        records=generate_equal_len_records(split,tokenizer,args.n_pairs,args.seed)
        payload["splits"][split]={"records_sample":records[:5],"protected":score_model(PROTECTED,tokenizer,records,device),"ordered_dynamic":{},"static_indirect_control":{},"deltas_ordered_minus_static":{}}
        out.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
        for ck in [f"chck_{i}M" for i in range(1,6)]:
            od=score_model(ORDERED_ROOT/ck,tokenizer,records,device)
            st=score_model(STATIC_ROOT/ck,tokenizer,records,device)
            payload["splits"][split]["ordered_dynamic"][ck]=od
            payload["splits"][split]["static_indirect_control"][ck]=st
            payload["splits"][split]["deltas_ordered_minus_static"][ck]={
                "both_contexts_correct_pair_fraction": od["both_contexts_correct_pair_fraction"]-st["both_contexts_correct_pair_fraction"],
                "mean_pair_sum_margin": od["mean_pair_sum_margin"]-st["mean_pair_sum_margin"],
                "positive_pair_sum_fraction": od["positive_pair_sum_fraction"]-st["positive_pair_sum_fraction"],
            }
            out.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
            print(json.dumps({"split":split,"ck":ck,"ordered_pair_sum":od["mean_pair_sum_margin"],"static_pair_sum":st["mean_pair_sum_margin"],"delta":payload["splits"][split]["deltas_ordered_minus_static"][ck]},indent=2),flush=True)
    print(json.dumps({"out":str(out)},indent=2))
if __name__=="__main__": main()
