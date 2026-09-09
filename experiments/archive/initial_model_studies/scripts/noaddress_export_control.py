#!/usr/bin/env python3
"""research: train/export no-address auxiliary control for research base-transfer screen.

The research screen compared plain_mixed vs wess_aux_exported_base. To interpret official
base-model gains, we need a matched no_address control: same gold spans, recurrent memory,
fusion path, synthetic schedule, and exported plain DeBERTa backbone, but no persistent
entity addressing.
"""
from __future__ import annotations
import importlib.util, json, pathlib, sys, time, torch
from transformers import AutoTokenizer

ROOT=pathlib.Path('experiments/archive/initial_model_studies')
SRC=ROOT/'scripts/wess_aux_base_transfer_screen.py'
OUT_JSON=ROOT/'data/noaddress_export_control.json'
RUN_DIR=ROOT/'training/runs/wess_aux_base_transfer_screen/smoke_384x4_anneal'

spec=importlib.util.spec_from_file_location('screen',SRC)
screen=importlib.util.module_from_spec(spec); sys.modules[spec.name]=screen; spec.loader.exec_module(screen)
bridge=screen.bridge; dyn=screen.dyn

class Args:
    hidden_size=384; n_layer=4; n_head=12; intermediate_size=1280
    steps=1000; batch_size=32; episode_ratio=0.5; anneal=True
    encoder_lr=2e-4; slot_lr=1e-3; gate_init=0.03; probe_every=250; seed=42
    train_pairs=1200; eval_pairs=200; official_examples=5000; run_name='smoke_384x4_anneal'


def main():
    args=Args(); t0=time.time(); bridge.setup_env(); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tok=AutoTokenizer.from_pretrained(str(bridge.TOK_PATH),use_fast=True)
    if tok.pad_token is None: tok.pad_token=tok.mask_token
    train_eps=dyn.build_pairs_tmpl(tok,args.train_pairs,args.seed*10+1,dyn.TRAIN_TEMPLATES,dyn.TRAIN_QUERY)
    eval_eps=dyn.build_pairs_tmpl(tok,args.eval_pairs,9990,dyn.HELDOUT_TEMPLATES,dyn.HELDOUT_QUERY)
    official=screen.load_official(tok,args.official_examples,args.seed*10+3)
    audit=bridge.shortcut_audit(eval_eps)
    model,meta=screen.train(tok,'no_address',args,train_eps,official,device)
    rec={'train_meta':meta,
         'official_loss_base_only':screen.eval_official_loss(tok,model,official,device,wess_on=False),
         'synthetic_base_only':screen.eval_binding(tok,model,eval_eps,device,wess_on=False),
         'synthetic_noaddress_on':screen.eval_binding(tok,model,eval_eps,device,wess_on=True),
         'interventions_noaddress_on':screen.eval_interventions(tok,model,eval_eps[:120],device),
         'params_plain_hf':sum(p.numel() for p in model.mlm.parameters()),
         'params_with_aux':sum(p.numel() for p in model.parameters())}
    export_dir=RUN_DIR/'no_address/hf_model'
    model.export_plain(tok,export_dir)
    rec['exported_plain_hf_model']=str(export_dir)
    payload={'status':'NOADDRESS_EXPORT_CONTROL','description':'Matched no-address auxiliary control for research exported-base screen; same data/schedule as smoke_384x4_anneal.','device':str(device),'args':{k:v for k,v in Args.__dict__.items() if not k.startswith('_') and not callable(v)},'shortcut_audit':audit,'no_address':rec,'elapsed_sec':round(time.time()-t0,1)}
    OUT_JSON.write_text(json.dumps(payload,indent=2)+'\n')
    (RUN_DIR/'no_address/screen_result.json').parent.mkdir(parents=True,exist_ok=True)
    (RUN_DIR/'no_address/screen_result.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps({'out':str(OUT_JSON),'export':str(export_dir),'metrics':rec},indent=2)[:4000],flush=True)

if __name__=='__main__': main()
