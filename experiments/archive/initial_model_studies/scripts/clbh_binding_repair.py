#!/usr/bin/env python3
"""research repaired CLBH binding discrimination test.

Repairs the token-lookup bug and candidate-comparison confound:
- Do NOT compare only the first token of a multi-token candidate.
- Use candidates verified to consist of a standalone space marker plus exactly one
  semantic word token under the BabyLM baseline16k tokenizer.
- Compare the semantic word token IDs (e.g. Ġhat vs Ġball), and verify those IDs
  occur in the context sequence at the intended property mentions.

The CLBH pointer is trained on a frozen pretrained DeBERTa encoder for 100k words,
then tested on entity-property bindings where both candidate property tokens occur
in the same context but only one is bound to the queried entity.
"""
from __future__ import annotations
import argparse, json, os, pathlib, random, sys, time, math
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer, get_cosine_schedule_with_warmup

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
sys.path.insert(0, str((ROOT/'training/scripts').resolve()))
from babylm_masked_train import TRAIN_FILES, Example, apply_masking, collate, download_dataset, iter_examples, MaskedChunkDataset

FROZEN_CKPT = ROOT/'training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/hf_model/chck_100M'
OUT_DEFAULT = ROOT/'training/runs/clbh_binding_repair_100k'

class CLBHPointer(nn.Module):
    def __init__(self, hidden_size: int, ptr_dim: int, vocab_size: int, word_embeddings: torch.Tensor | None = None):
        super().__init__()
        self.ptr_dim = ptr_dim
        self.vocab_size = vocab_size
        self.query_proj = nn.Linear(hidden_size, ptr_dim, bias=False)
        self.key_proj = nn.Linear(hidden_size, ptr_dim, bias=False)
        self.gate_proj = nn.Linear(hidden_size, 1, bias=True)
        nn.init.constant_(self.gate_proj.bias, -2.0)
        if word_embeddings is not None:
            with torch.no_grad():
                # Low-rank embedding projection is a structural warm-start only;
                # the binding test below is the real scientific measurement.
                U, S, V = torch.svd_lowrank(word_embeddings.float(), q=ptr_dim)
                self.query_proj.weight.copy_(V.T)
                self.key_proj.weight.copy_(V.T)
        self.register_buffer('shuffle_perm', torch.randperm(vocab_size))

    def forward(self, hidden_states, input_ids, attention_mask, mask_token_id, mode='clbh'):
        B, L, H = hidden_states.shape
        Q = self.query_proj(hidden_states)
        K = self.key_proj(hidden_states)
        raw = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.ptr_dim)
        is_visible = (input_ids != mask_token_id) & (attention_mask == 1)
        vis = is_visible.unsqueeze(1).expand(B, L, L).float()
        self_mask = torch.eye(L, device=hidden_states.device).unsqueeze(0)
        vis = vis * (1.0 - self_mask)
        scores = F.softplus(raw) * vis
        gate = torch.sigmoid(self.gate_proj(hidden_states))
        if mode == 'shuffled':
            scatter_ids = self.shuffle_perm[input_ids.clamp(0, self.vocab_size - 1)]
        else:
            scatter_ids = input_ids
        copy_logits = torch.zeros(B, L, self.vocab_size, device=hidden_states.device)
        scatter_expanded = scatter_ids.unsqueeze(1).expand(B, L, L)
        copy_logits.scatter_add_(2, scatter_expanded, scores)
        diag = {
            'gate_mean': float(gate.mean().detach().cpu()),
            'max_copy': float(copy_logits.max().detach().cpu()),
            'mean_positive_score': float(scores[scores > 0].mean().detach().cpu()) if (scores > 0).any() else 0.0,
            'n_visible_mean': float(is_visible.float().sum(1).mean().detach().cpu()),
        }
        return copy_logits, gate, diag

def env_setup():
    hf = ROOT/'training/hf_home'
    os.environ['HF_HOME'] = str(hf.resolve())
    os.environ['HF_HUB_CACHE'] = str((hf/'hub').resolve())
    os.environ['HF_DATASETS_CACHE'] = str((hf/'datasets').resolve())
    os.environ['TRANSFORMERS_CACHE'] = str((hf/'transformers').resolve())
    os.environ['HF_MODULES_CACHE'] = str((ROOT/'training/hf_modules_cache').resolve())
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

def candidate_semantic_token(tokenizer, word: str):
    """Return a single semantic token id for a word if tokenizer pattern is [space_marker, word_token]."""
    ids = tokenizer(' ' + word, add_special_tokens=False)['input_ids']
    toks = tokenizer.convert_ids_to_tokens(ids)
    # BabyLM baseline tokenizer often returns standalone space marker then word token.
    # Accept only exactly two tokens where the first decodes to whitespace/space marker
    # and the second decodes to the intended word surface.
    if len(ids) == 2 and toks[0] in ('Ġ', '▁'):
        surf = toks[1].replace('Ġ', '').replace('▁', '').lower()
        if surf == word.lower():
            return ids[1], toks[1], {'raw_ids': ids, 'raw_tokens': toks, 'mode': 'space_marker_plus_single_word'}
    # Also accept a single token directly matching the word.
    if len(ids) == 1:
        surf = toks[0].replace('Ġ', '').replace('▁', '').lower()
        if surf == word.lower():
            return ids[0], toks[0], {'raw_ids': ids, 'raw_tokens': toks, 'mode': 'single_token'}
    return None, None, {'raw_ids': ids, 'raw_tokens': toks, 'mode': 'rejected'}

def encode_with_mask(tokenizer, text: str, mask_id: int):
    parts = text.split('[MASK]')
    ids = []
    for i, part in enumerate(parts):
        # Do NOT strip: preserve boundary tokenization around the mask.
        ids.extend(tokenizer(part, add_special_tokens=False)['input_ids'])
        if i < len(parts) - 1:
            ids.append(mask_id)
    return ids

def build_cases(tokenizer):
    raw_cases = [
        ("Alice found a ball. Bob found a hat.", "Bob lost the [MASK] yesterday.", "hat", "ball"),
        ("Alice found a ball. Bob found a hat.", "Alice lost the [MASK] yesterday.", "ball", "hat"),
        ("The cat ate fish. The dog ate bones.", "The dog wanted more [MASK] today.", "bones", "fish"),
        ("The cat ate fish. The dog ate bones.", "The cat wanted more [MASK] today.", "fish", "bones"),
        ("John lives in Paris. Mary lives in London.", "John returned to [MASK] last week.", "Paris", "London"),
        ("John lives in Paris. Mary lives in London.", "Mary returned to [MASK] last week.", "London", "Paris"),
        ("The red car is fast. The blue car is slow.", "The blue car drove [MASK] down the road.", "slow", "fast"),
        ("The red car is fast. The blue car is slow.", "The red car drove [MASK] down the road.", "fast", "slow"),
        ("Sarah brought milk. Tom brought cake.", "Sarah shared the [MASK] later.", "milk", "cake"),
        ("Sarah brought milk. Tom brought cake.", "Tom shared the [MASK] later.", "cake", "milk"),
        ("The north door was open. The south door was closed.", "The north door stayed [MASK].", "open", "closed"),
        ("The north door was open. The south door was closed.", "The south door stayed [MASK].", "closed", "open"),
    ]
    cases = []
    for context, query, correct, distractor in raw_cases:
        ctid, ctok, cmeta = candidate_semantic_token(tokenizer, correct)
        dtid, dtok, dmeta = candidate_semantic_token(tokenizer, distractor)
        if ctid is None or dtid is None:
            continue
        ids = encode_with_mask(tokenizer, context + ' ' + query, tokenizer.mask_token_id)
        toks = tokenizer.convert_ids_to_tokens(ids)
        # verify both semantic candidate IDs occur in context before the mask
        mask_positions = [i for i,x in enumerate(ids) if x == tokenizer.mask_token_id]
        if not mask_positions:
            continue
        mp = mask_positions[0]
        cpos = [i for i,x in enumerate(ids[:mp]) if x == ctid]
        dpos = [i for i,x in enumerate(ids[:mp]) if x == dtid]
        if not cpos or not dpos:
            continue
        cases.append({
            'context': context, 'query': query, 'correct': correct, 'distractor': distractor,
            'correct_tid': ctid, 'distractor_tid': dtid, 'correct_token': ctok, 'distractor_token': dtok,
            'correct_tokenization': cmeta, 'distractor_tokenization': dmeta,
            'input_ids': ids, 'input_tokens': toks, 'mask_pos': mp,
            'correct_context_positions': cpos, 'distractor_context_positions': dpos,
        })
    return cases

def binding_test(model, pointer, tokenizer, device, cases, mode='clbh'):
    model.eval(); pointer.eval()
    out = []
    mask_id = tokenizer.mask_token_id
    for c in cases:
        inp = torch.tensor([c['input_ids']], dtype=torch.long, device=device)
        attn = torch.ones_like(inp)
        mp = c['mask_pos']; ctid = c['correct_tid']; dtid = c['distractor_tid']
        with torch.no_grad():
            enc = model.deberta(input_ids=inp, attention_mask=attn)
            hidden = enc.last_hidden_state
            base_logits = model.cls(hidden)
            copy_logits, gate, diag = pointer(hidden, inp, attn, mask_id, mode=mode)
            combined = base_logits + gate * copy_logits
        row = {
            'context': c['context'], 'query': c['query'], 'correct': c['correct'], 'distractor': c['distractor'],
            'correct_tid': ctid, 'distractor_tid': dtid, 'correct_token': c['correct_token'], 'distractor_token': c['distractor_token'],
            'correct_context_positions': c['correct_context_positions'], 'distractor_context_positions': c['distractor_context_positions'],
            'mask_pos': mp, 'tokens': c['input_tokens'],
            'base_correct': float(base_logits[0, mp, ctid].item()),
            'base_distractor': float(base_logits[0, mp, dtid].item()),
            'copy_correct': float(copy_logits[0, mp, ctid].item()),
            'copy_distractor': float(copy_logits[0, mp, dtid].item()),
            'combined_correct': float(combined[0, mp, ctid].item()),
            'combined_distractor': float(combined[0, mp, dtid].item()),
            'gate_at_mask': float(gate[0, mp, 0].item()),
        }
        row['base_margin'] = row['base_correct'] - row['base_distractor']
        row['copy_margin'] = row['copy_correct'] - row['copy_distractor']
        row['combined_margin'] = row['combined_correct'] - row['combined_distractor']
        out.append(row)
    if out:
        aggregate = {
            'n_cases': len(out),
            'copy_discrimination_frac': sum(r['copy_margin'] > 0 for r in out) / len(out),
            'combined_discrimination_frac': sum(r['combined_margin'] > 0 for r in out) / len(out),
            'base_discrimination_frac': sum(r['base_margin'] > 0 for r in out) / len(out),
            'mean_copy_margin': sum(r['copy_margin'] for r in out) / len(out),
            'mean_base_margin': sum(r['base_margin'] for r in out) / len(out),
            'mean_combined_margin': sum(r['combined_margin'] for r in out) / len(out),
            'mean_gate_at_mask': sum(r['gate_at_mask'] for r in out) / len(out),
        }
    else:
        aggregate = {'n_cases': 0, 'copy_discrimination_frac': 0, 'combined_discrimination_frac': 0, 'base_discrimination_frac': 0, 'mean_copy_margin': 0, 'mean_base_margin': 0, 'mean_combined_margin': 0, 'mean_gate_at_mask': 0}
    model.train(); pointer.train()
    return {'aggregate': aggregate, 'cases': out}

def train_pointer(args, model, pointer, tokenizer, device, outdir):
    # data
    raw_dir, _ = download_dataset(args, outdir)
    files = [raw_dir / n for n in TRAIN_FILES]
    pool = list(iter_examples(files, args.max_word_exposure, 160))
    for i, ex in enumerate(pool): ex.example_id = i
    rng = random.Random(args.seed); rng.shuffle(pool)
    examples=[]; actual=0
    for ex in pool:
        if actual >= args.max_word_exposure: break
        if actual + ex.words <= args.max_word_exposure:
            examples.append(ex); actual += ex.words
        else:
            take=args.max_word_exposure-actual
            examples.append(Example(' '.join(ex.text.split()[:take]), take, ex.example_id, ex.source)); actual += take
    ds=MaskedChunkDataset(examples, tokenizer, 256)
    loader=DataLoader(ds,batch_size=args.batch_size,shuffle=False,collate_fn=collate,num_workers=2,pin_memory=torch.cuda.is_available())
    opt=torch.optim.AdamW(pointer.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps=len(loader)
    sched=get_cosine_schedule_with_warmup(opt, max(1,total_steps//20), total_steps)
    gen=torch.Generator(device=device); gen.manual_seed(args.seed)
    logs=[]; t0=time.time(); cum=0; mask_id=tokenizer.mask_token_id
    pointer.train(); model.eval()
    for step,batch in enumerate(loader,1):
        words=int(batch.pop('words').sum().item()); cum += words
        input_ids=batch['input_ids'].to(device)[:,:256]
        attn=batch['attention_mask'].to(device)[:,:256]
        wg=batch['word_group'].to(device)[:,:256]
        masked_inputs, labels=apply_masking(input_ids,attn,wg,tokenizer,'wwm',0.15,gen)
        with torch.no_grad():
            enc=model.deberta(input_ids=masked_inputs,attention_mask=attn)
            hidden=enc.last_hidden_state
            base_logits=model.cls(hidden)
        copy_logits, gate, diag=pointer(hidden.detach(), masked_inputs, attn, mask_id, mode=args.mode)
        combined=base_logits.detach()+gate*copy_logits
        loss=F.cross_entropy(combined.view(-1,len(tokenizer)), labels.view(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(pointer.parameters(),1.0); opt.step(); sched.step()
        rec={'step':step,'cum':cum,'loss':float(loss.detach().cpu()),'lr':float(sched.get_last_lr()[0]),'elapsed':time.time()-t0,**diag}
        logs.append(rec)
        if step<=5 or step%args.log_every==0 or step==total_steps:
            print(json.dumps({'event':'train',**rec}), flush=True)
    return logs, cum, total_steps

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output_dir', default=str(OUT_DEFAULT))
    ap.add_argument('--mode', choices=['clbh','shuffled'], default='clbh')
    ap.add_argument('--ptr_dim', type=int, default=64)
    ap.add_argument('--max_word_exposure', type=int, default=100000)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--learning_rate', type=float, default=5e-4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--log_every', type=int, default=20)
    ap.add_argument('--dataset_id', default='BabyLM-community/BabyLM-2026-Strict-Small')
    ap.add_argument('--dataset_revision', default='c92ab16b4f08858304b0815706065b3354d8fc0a')
    args=ap.parse_args()
    env_setup(); outdir=pathlib.Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(json.dumps({'event':'load_frozen_model','ckpt':str(FROZEN_CKPT)}), flush=True)
    model=AutoModelForMaskedLM.from_pretrained(str(FROZEN_CKPT.resolve()), trust_remote_code=True).to(device)
    tokenizer=AutoTokenizer.from_pretrained(str(FROZEN_CKPT.resolve()), use_fast=True)
    for p in model.parameters(): p.requires_grad=False
    model.eval()
    word_emb=model.deberta.embeddings.word_embeddings.weight.detach()
    pointer=CLBHPointer(model.config.hidden_size,args.ptr_dim,len(tokenizer),word_emb).to(device)
    pointer_params=sum(p.numel() for p in pointer.parameters())
    cases=build_cases(tokenizer)
    print(json.dumps({'event':'cases_built','n_cases':len(cases),'case_tokens':[{k:c[k] for k in ['correct','distractor','correct_token','distractor_token','correct_tid','distractor_tid','correct_context_positions','distractor_context_positions','mask_pos']} for c in cases]}, ensure_ascii=False), flush=True)
    pre= binding_test(model,pointer,tokenizer,device,cases,mode=args.mode)
    pre_shuf=binding_test(model,pointer,tokenizer,device,cases,mode='shuffled')
    print(json.dumps({'event':'pre_binding','clbh':pre['aggregate'],'shuffled':pre_shuf['aggregate']}), flush=True)
    logs,cum,total_steps=train_pointer(args,model,pointer,tokenizer,device,outdir)
    post=binding_test(model,pointer,tokenizer,device,cases,mode=args.mode)
    post_shuf=binding_test(model,pointer,tokenizer,device,cases,mode='shuffled')
    print(json.dumps({'event':'post_binding','clbh':post['aggregate'],'shuffled':post_shuf['aggregate']}), flush=True)
    payload={
        'status':'CLBH_BINDING_REPAIR_SINGLE_SEMANTIC_TOKEN',
        'frozen_ckpt':str(FROZEN_CKPT),'mode':args.mode,'ptr_dim':args.ptr_dim,'pointer_params':pointer_params,
        'word_exposure':cum,'total_steps':total_steps,'loss_first':logs[0]['loss'] if logs else None,'loss_last':logs[-1]['loss'] if logs else None,
        'tokenization_rule':'candidate accepted only if tokenizer(" "+word) is [standalone space marker, single semantic word token] or one matching token; margins use semantic word token id, not space marker or first arbitrary subword',
        'pre_clbh':pre,'pre_shuffled':pre_shuf,'post_clbh':post,'post_shuffled':post_shuf,
        'interpretation':{
            'binding_passes': post['aggregate']['copy_discrimination_frac']>0.6 and post['aggregate']['mean_copy_margin']>0.05,
            'copy_margin_exceeds_shuffled': post['aggregate']['mean_copy_margin'] > post_shuf['aggregate']['mean_copy_margin'] + 0.02,
            'combined_exceeds_base': post['aggregate']['mean_combined_margin'] > post['aggregate']['mean_base_margin'] + 0.02,
        },
    }
    (outdir/'binding_repair_results.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    torch.save(pointer.state_dict(), outdir/'clbh_pointer.pt')
    print(json.dumps({'event':'done','out':str(outdir/'binding_repair_results.json'),'interpretation':payload['interpretation'],'post_clbh':post['aggregate'],'post_shuffled':post_shuf['aggregate']}, indent=2), flush=True)
if __name__=='__main__': main()
