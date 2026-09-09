#!/usr/bin/env python3
"""research: probe Phase-2 sequence-curriculum dynamics and feasible batch sizes.

Runs a few forward/backward updates for selected (seq_len, batch_size) pairs using
the actual 12x384/40k DeBERTa-v2 + LAMB recipe. It measures memory, visible tokens,
visible word groups, masked tokens, and time/update. This is a scientific guard
before long training because fixed batch256 with early truncation both miscounts
visible experience and OOMs at seq256.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import argparse, json, time, sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

# Import the research trainer objects rather than copying implementation.
sys.path.insert(0, str(_public_path('experiments/archive/compact_experience/scripts')))
import phase2_sota_trainer as tr  # type: ignore


def make_args(tokenizer_path: str, data_path: str):
    class A: pass
    a = A()
    a.model_type = "deberta_v2"
    a.n_layer = 12; a.hidden_size = 384; a.n_head = 12; a.ffn_mult = 4; a.intermediate_size = 1280
    a.max_seq_length = 256
    a.max_position_embeddings = 512; a.max_relative_positions = -1; a.position_buckets = 256
    a.deberta_relative_attention = "true"; a.deberta_pos_att_type = "p2c,c2p"; a.share_att_key = "true"; a.position_biased_input = "false"
    a.tokenizer_path = tokenizer_path; a.example_jsonl = data_path
    return a


def probe_pair(args, tokenizer, examples, seq_len: int, batch_size: int, updates: int, device: torch.device):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    dataset = tr.MaskedChunkDataset(examples, tokenizer, seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=tr.collate, num_workers=0)
    margs = make_args(args.tokenizer_path, args.example_jsonl)
    tr.reset_all_rng(args.extra_init_seed)
    model = tr.build_model(margs, tokenizer).to(device)
    opt = tr.make_lamb_optimizer(model, lr=0.007, weight_decay=0.01)
    gen = torch.Generator(device=device); gen.manual_seed(args.train_rng_seed)
    model.train()
    t0=time.time(); losses=[]; vis_tokens=[]; vis_groups=[]; masked=[]; words=[]
    ok=True; err=""
    try:
        for i,batch in enumerate(loader,1):
            if i>updates: break
            input_ids=batch["input_ids"].to(device); attention_mask=batch["attention_mask"].to(device); word_group=batch["word_group"].to(device)
            labels_in, labels = tr.apply_masking(input_ids, attention_mask, word_group, tokenizer, "wwm", 0.15, gen)
            opt.zero_grad(set_to_none=True)
            out=model(input_ids=labels_in, attention_mask=attention_mask, labels=labels)
            loss=out.loss; loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            losses.append(float(loss.detach().cpu()))
            vis_tokens.append(int(attention_mask.sum().item()))
            vis_groups.append(int(((word_group >= 0) & attention_mask.bool()).sum().item()))
            masked.append(int((labels != -100).sum().item()))
            words.append(int(batch["words"].sum().item()))
        torch.cuda.synchronize(device)
    except Exception as e:
        ok=False; err=repr(e)
    elapsed=time.time()-t0
    peak=torch.cuda.max_memory_allocated(device)/(1024**3)
    del model, opt, dataset, loader
    torch.cuda.empty_cache()
    return {
        "seq_len": seq_len, "batch_size": batch_size, "ok": ok, "error": err,
        "updates_completed": len(losses), "loss_first": losses[0] if losses else None, "loss_last": losses[-1] if losses else None,
        "mean_visible_tokens_per_update": sum(vis_tokens)/len(vis_tokens) if vis_tokens else None,
        "mean_visible_positions_per_word_counted": (sum(vis_tokens)/sum(words)) if words else None,
        "mean_visible_wordgroup_positions_per_word_counted": (sum(vis_groups)/sum(words)) if words else None,
        "mean_masked_tokens_per_update": sum(masked)/len(masked) if masked else None,
        "peak_mem_gb": round(peak,3), "elapsed_sec": round(elapsed,2),
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--example_jsonl", default="experiments/archive/compact_experience/data/fixedinit_replication/official_100M.jsonl")
    p.add_argument("--tokenizer_path", default="experiments/archive/compact_experience/data/shared_tokenizer/hf_tokenizer_40k_shared")
    p.add_argument("--max_words", type=int, default=500000)
    p.add_argument("--updates", type=int, default=2)
    p.add_argument("--extra_init_seed", type=int, default=43200)
    p.add_argument("--train_rng_seed", type=int, default=43201)
    p.add_argument("--out", default="experiments/archive/compact_experience/data/phase2_dynamics/dynamics_probe.json")
    p.add_argument("--pairs", nargs="*", default=["64:512", "64:768", "64:1024", "128:256", "128:384", "128:512", "256:64", "256:96", "256:128", "256:160"])
    args=p.parse_args()
    tokenizer=tr.make_portable_tokenizer(args.tokenizer_path)
    examples=tr.load_examples_jsonl(Path(args.example_jsonl), args.max_words)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results=[]
    for spec in args.pairs:
        L,B=map(int,spec.split(":"))
        print(json.dumps({"event":"probe_start","seq_len":L,"batch_size":B}), flush=True)
        rec=probe_pair(args, tokenizer, examples, L, B, args.updates, device)
        print(json.dumps({"event":"probe_done", **rec}), flush=True)
        results.append(rec)
    payload={"status":"PHASE2_DYNAMICS_PROBE_DONE", "example_jsonl":args.example_jsonl, "tokenizer_path":args.tokenizer_path, "max_words":args.max_words, "updates":args.updates, "results":results}
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,indent=2),encoding="utf-8")

if __name__=="__main__": main()
