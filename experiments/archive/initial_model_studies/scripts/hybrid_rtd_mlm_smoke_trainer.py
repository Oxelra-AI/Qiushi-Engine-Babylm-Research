#!/usr/bin/env python3
"""research corrected hybrid RTD+MLM smoke trainer.

Purpose: verify the RTD+MLM implementation only, before any large run.
It reuses the verified BabyLM masked trainer's official/jsonl data selection,
collate, WWM masking, DeBERTa-v2 config, checkpoint saving, and tokenization
summaries. It adds a small generator and an RTD head; the saved HF checkpoint is
only the discriminator, preserving official MLM evaluation compatibility.
"""
from __future__ import annotations
import argparse, json, pathlib, random, sys, time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

sys.path.insert(0, str(pathlib.Path('experiments/archive/initial_model_studies/training/scripts').resolve()))
from babylm_masked_train import (  # noqa: E402
    TRAIN_FILES, Example, apply_masking, build_model, collate, download_dataset,
    iter_examples, load_examples_jsonl, make_portable_tokenizer, MaskedChunkDataset,
    reset_all_rng, save_hf_checkpoint, seq_length_for_progress, sha256_file,
    summarize_tokenization_coupling,
)

class SmallGenerator(nn.Module):
    def __init__(self, vocab_size: int, disc_embed: nn.Embedding, hidden: int = 128,
                 layers: int = 2, heads: int = 4, max_position: int = 512):
        super().__init__()
        # Hold the discriminator embedding as a NON-registered reference so its
        # parameters stay owned by the discriminator only (GDES stop-gradient in forward).
        object.__setattr__(self, '_disc_embed', disc_embed)
        self.proj = nn.Linear(disc_embed.embedding_dim, hidden, bias=False)
        self.pos = nn.Embedding(max_position, hidden)
        self.norm = nn.LayerNorm(hidden)
        enc = nn.TransformerEncoderLayer(hidden, heads, hidden * 4, dropout=0.1,
                                         activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Linear(hidden, vocab_size))
    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            e = self._disc_embed(input_ids).detach()
        x = self.proj(e) + self.pos(torch.arange(input_ids.size(1), device=input_ids.device)).unsqueeze(0)
        x = self.norm(x)
        h = self.encoder(x, src_key_padding_mask=(attention_mask == 0))
        return self.head(h)

class RTDHead(nn.Module):
    def __init__(self, hidden: int):
        super().__init__(); self.net = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
    def forward(self, h): return self.net(h).squeeze(-1)

def save_aux_checkpoint(gen: nn.Module, rtd: nn.Module, dst: pathlib.Path, args, step: int, cum_words: int) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    torch.save({
        'generator_state_dict': gen.state_dict(),
        'rtd_head_state_dict': rtd.state_dict(),
        'step': step,
        'cumulative_word_exposure': int(cum_words),
        'gen_hidden': args.gen_hidden,
        'gen_layers': args.gen_layers,
        'gen_heads': args.gen_heads,
        'hidden_size': args.hidden_size,
        'lambda_rtd': args.lambda_rtd,
        'note': 'Auxiliary generator/RTD head state for context-ablation probes; discriminator is saved as HF model in same checkpoint directory.'
    }, dst / 'hybrid_aux.pt')

def build_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset_id', default='BabyLM-community/BabyLM-2026-Strict-Small')
    p.add_argument('--dataset_revision', default='c92ab16b4f08858304b0815706065b3354d8fc0a')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--max_word_exposure', type=int, default=10000)
    p.add_argument('--example_pool_words', type=int, default=10000)
    p.add_argument('--checkpoint_words', type=int, default=10000)
    p.add_argument('--words_per_example', type=int, default=160)
    p.add_argument('--example_jsonl', default=''); p.add_argument('--example_jsonl_label', default=''); p.add_argument('--example_jsonl_meta', default='')
    p.add_argument('--tokenizer_path', default=''); p.add_argument('--tokenizer_label', default='baseline16k'); p.add_argument('--tokenization_summary_limit', type=int, default=200)
    p.add_argument('--mask_mode', choices=['token','wwm'], default='wwm'); p.add_argument('--mask_prob', type=float, default=0.15)
    p.add_argument('--seq_length', type=int, default=256); p.add_argument('--max_seq_length', type=int, default=256); p.add_argument('--max_position_embeddings', type=int, default=512)
    p.add_argument('--seq_len_schedule', default='')
    p.add_argument('--model_type', choices=['bert','deberta_v2'], default='deberta_v2')
    p.add_argument('--position_buckets', type=int, default=256); p.add_argument('--max_relative_positions', type=int, default=256)
    p.add_argument('--deberta_pos_att_type', default='p2c,c2p'); p.add_argument('--deberta_relative_attention', choices=['true','false'], default='true')
    p.add_argument('--hidden_size', type=int, default=480); p.add_argument('--n_layer', type=int, default=8); p.add_argument('--n_head', type=int, default=8); p.add_argument('--ffn_mult', type=int, default=4)
    p.add_argument('--gen_hidden', type=int, default=128); p.add_argument('--gen_layers', type=int, default=2); p.add_argument('--gen_heads', type=int, default=4); p.add_argument('--lambda_rtd', type=float, default=50.0)
    p.add_argument('--batch_size', type=int, default=32); p.add_argument('--learning_rate', type=float, default=1e-3); p.add_argument('--weight_decay', type=float, default=0.01); p.add_argument('--warmup_fraction', type=float, default=0.05); p.add_argument('--lr_total_steps', type=int, default=0)
    p.add_argument('--seed', type=int, default=42); p.add_argument('--extra_init_seed', type=int, default=456); p.add_argument('--train_rng_seed', type=int, default=789); p.add_argument('--log_every', type=int, default=5)
    return p.parse_args()

def load_examples(args, out, tokenizer):
    pool_words = max(args.example_pool_words, args.max_word_exposure); selected_words = args.max_word_exposure; example_jsonl_meta = {}
    if args.example_jsonl:
        p = pathlib.Path(args.example_jsonl); examples, total_words, total_rows, sample_rows = load_examples_jsonl(p, selected_words)
        manifest_files = [{'path': str(p), 'name': p.name, 'bytes': p.stat().st_size, 'sha256': sha256_file(p), 'whitespace_words': total_words, 'rows': total_rows}]
        if args.example_jsonl_meta:
            mp = pathlib.Path(args.example_jsonl_meta); example_jsonl_meta = json.loads(mp.read_text(encoding='utf-8'))
        meta = {'data_source_type':'example_jsonl','example_jsonl':str(p),'example_jsonl_label':args.example_jsonl_label,'example_jsonl_total_words':total_words,'example_jsonl_total_rows':total_rows,'example_jsonl_sample_rows_without_text':sample_rows,'example_jsonl_meta':example_jsonl_meta}
    else:
        raw_dir, manifest_files = download_dataset(args, out); files = [raw_dir / n for n in TRAIN_FILES]
        total_words = sum(f['whitespace_words'] for f in manifest_files)
        pool = list(iter_examples(files, pool_words, args.words_per_example))
        for i, ex in enumerate(pool): ex.example_id = i
        rng = random.Random(args.seed); rng.shuffle(pool)
        examples=[]; actual=0
        for ex in pool:
            if actual >= selected_words: break
            if actual + ex.words <= selected_words:
                examples.append(ex); actual += ex.words
            else:
                take = selected_words - actual; examples.append(Example(' '.join(ex.text.split()[:take]), take, ex.example_id, ex.source)); actual += take
        meta = {'data_source_type':'official_corpus'}
    actual_words = sum(ex.words for ex in examples)
    if actual_words != selected_words: raise RuntimeError(f'word mismatch {actual_words} vs {selected_words}')
    source_words={}
    for ex in examples: source_words[ex.source]=source_words.get(ex.source,0)+ex.words
    (out/'example_order_manifest.json').write_text(json.dumps({'seed':args.seed,'example_pool_words_actual':pool_words,'selected_for_training_words':actual_words,'words_per_example':args.words_per_example,'num_consumed_examples':len(examples),'source_words_consumed':source_words,'mask_mode':args.mask_mode,'mask_prob':args.mask_prob,'tokenizer_label':args.tokenizer_label,'tokenizer_path':args.tokenizer_path,'tokenizer_vocab_size':len(tokenizer),**meta}, indent=2, ensure_ascii=False), encoding='utf-8')
    (out/'data_manifest.json').write_text(json.dumps({'dataset_id':args.dataset_id,'dataset_revision':args.dataset_revision,'files':manifest_files,'selected_for_this_run_whitespace_words':actual_words,**meta}, indent=2, ensure_ascii=False), encoding='utf-8')
    return examples, actual_words, pool_words, source_words

def main():
    args=build_args(); start=time.time(); out=pathlib.Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    tokenizer=make_portable_tokenizer(args.tokenizer_path); examples, actual_words, pool_words, source_words = load_examples(args, out, tokenizer)
    tok_sum=summarize_tokenization_coupling(examples, tokenizer, args.max_seq_length, args.tokenization_summary_limit); (out/'tokenization_coupling_summary.json').write_text(json.dumps(tok_sum, indent=2), encoding='utf-8')
    ds=MaskedChunkDataset(examples, tokenizer, args.max_seq_length); loader=DataLoader(ds,batch_size=args.batch_size,shuffle=False,collate_fn=collate,num_workers=2,pin_memory=torch.cuda.is_available())
    if args.extra_init_seed >= 0: reset_all_rng(args.extra_init_seed)
    disc=build_model(args, tokenizer); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); disc.to(device)
    if args.train_rng_seed >= 0: reset_all_rng(args.train_rng_seed)
    gen=SmallGenerator(len(tokenizer), disc.get_input_embeddings(), args.gen_hidden, args.gen_layers, args.gen_heads, args.max_position_embeddings).to(device)
    rtd=RTDHead(args.hidden_size).to(device)
    # Dedupe by id: the GDES-shared discriminator embedding is referenced inside the
    # generator (disc_embed) but its gradient is stopped there; it must appear once.
    seen=set(); params=[]
    for p in list(disc.parameters())+list(gen.parameters())+list(rtd.parameters()):
        if id(p) in seen: continue
        seen.add(id(p)); params.append(p)
    opt=torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9,0.98))
    total_steps=len(loader); sched_total=args.lr_total_steps if args.lr_total_steps>0 else total_steps
    if sched_total < total_steps: raise RuntimeError('lr_total_steps < actual steps')
    sched=get_cosine_schedule_with_warmup(opt, max(1,int(sched_total*args.warmup_fraction)), sched_total)
    mask_gen=torch.Generator(device=device); mask_gen.manual_seed(args.train_rng_seed if args.train_rng_seed>=0 else args.seed)
    next_ckpt=args.checkpoint_words if args.checkpoint_words>0 else None; cum=0; logs=[]; saved=[]; masked_total=0; active_total=0; replaced_total=0
    seq_schedule=[]
    if args.seq_len_schedule:
        for part in args.seq_len_schedule.split(','):
            t,L=part.split(':'); seq_schedule.append((float(t),int(L)))
        seq_schedule.sort()
    disc.train(); gen.train(); rtd.train(); logf=(out/'training_log.jsonl').open('w',encoding='utf-8')
    for step,batch in enumerate(loader,1):
        words=int(batch.pop('words').sum().item()); input_ids=batch['input_ids'].to(device); attn=batch['attention_mask'].to(device); wg=batch['word_group'].to(device)
        cur_len=min(seq_length_for_progress((research)/max(1,sched_total),seq_schedule,args.seq_length) if seq_schedule else args.seq_length, args.max_seq_length)
        input_ids=input_ids[:,:cur_len].contiguous(); attn=attn[:,:cur_len].contiguous(); wg=wg[:,:cur_len].contiguous()
        masked_inputs, labels = apply_masking(input_ids, attn, wg, tokenizer, args.mask_mode, args.mask_prob, mask_gen)
        masked_pos = labels != -100; original = input_ids.clone()
        gen_logits=gen(masked_inputs, attn); gen_loss=F.cross_entropy(gen_logits.view(-1,len(tokenizer)), labels.view(-1), ignore_index=-100)
        with torch.no_grad():
            samples=torch.multinomial(F.softmax(gen_logits[masked_pos], dim=-1), 1).squeeze(-1) if masked_pos.any() else torch.empty(0,dtype=torch.long,device=device)
        rtd_in=original.clone(); rtd_in[masked_pos]=samples
        rtd_labels=(rtd_in == original).float()
        outd=disc(input_ids=rtd_in, attention_mask=attn, output_hidden_states=True)
        mlm_targets=original.clone(); mlm_targets[~masked_pos]=-100
        disc_mlm=F.cross_entropy(outd.logits.view(-1,outd.logits.size(-1)), mlm_targets.view(-1), ignore_index=-100)
        rtd_logits=rtd(outd.hidden_states[-1]); active=attn.bool().view(-1)
        rtd_loss=F.binary_cross_entropy_with_logits(rtd_logits.view(-1)[active], rtd_labels.view(-1)[active])
        loss=gen_loss + disc_mlm + args.lambda_rtd*rtd_loss
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params,1.0); opt.step(); sched.step()
        cum += words; n_mask=int(masked_pos.sum().item()); n_active=int(attn.sum().item()); n_repl=int(((rtd_in != original) & attn.bool()).sum().item())
        masked_total += n_mask; active_total += n_active; replaced_total += n_repl
        with torch.no_grad():
            lab=rtd_labels.view(-1)[active]; pred_orig=(torch.sigmoid(rtd_logits.view(-1)[active])>0.5).float()
            acc=(pred_orig==lab).float().mean().item(); pred_orig_rate=pred_orig.mean().item(); orig_rate=lab.mean().item()
            # Shortcut-exclusion metrics: recall on the REPLACED class (label==0) is the real
            # signal; the all-original majority shortcut has replaced_recall==0. Above-majority
            # accuracy = acc - max(orig_rate, 1-orig_rate).
            repl_mask=(lab==0); orig_mask=(lab==1)
            replaced_recall=((pred_orig==0)&repl_mask).float().sum().item()/max(1.0,repl_mask.float().sum().item())
            original_recall=((pred_orig==1)&orig_mask).float().sum().item()/max(1.0,orig_mask.float().sum().item())
            majority=max(orig_rate,1.0-orig_rate); above_majority=acc-majority
        rec={'step':step,'cumulative_word_exposure':cum,'batch_words':words,'gen_loss':float(gen_loss.detach().cpu()),'disc_mlm_loss':float(disc_mlm.detach().cpu()),'rtd_loss':float(rtd_loss.detach().cpu()),'total_loss':float(loss.detach().cpu()),'rtd_acc':acc,'rtd_pred_original_rate':pred_orig_rate,'rtd_label_original_rate':orig_rate,'rtd_replaced_recall':replaced_recall,'rtd_original_recall':original_recall,'rtd_above_majority':above_majority,'masked_tokens':n_mask,'replaced_tokens':n_repl,'active_tokens':n_active,'lr':float(sched.get_last_lr()[0]),'elapsed_sec':time.time()-start}
        logs.append(rec); logf.write(json.dumps(rec)+'\n'); logf.flush()
        if step==1 or step%args.log_every==0 or step==total_steps: print(json.dumps({'event':'train',**rec}), flush=True)
        while next_ckpt is not None and cum >= next_ckpt and next_ckpt <= args.max_word_exposure:
            name=f"chck_{next_ckpt//1_000_000}M" if next_ckpt>=1_000_000 and next_ckpt%1_000_000==0 else ("chck_1M" if next_ckpt<1_000_000 else f"chck_{next_ckpt}w")
            cp=out/'hf_model'/name; save_hf_checkpoint(disc, tokenizer, cp); save_aux_checkpoint(gen, rtd, cp, args, step, cum); saved.append({'name':name,'target_word_exposure':next_ckpt,'actual_cumulative_word_exposure':cum,'path':str(cp),'aux_path':str(cp/'hybrid_aux.pt')}); next_ckpt += args.checkpoint_words
    logf.close(); save_hf_checkpoint(disc, tokenizer, out/'hf_model'); save_aux_checkpoint(gen, rtd, out/'hf_model', args, step, cum)
    metrics={'variant':'hybrid_rtd_mlm_smoke','backend':'mlm','discriminator_parameter_count':sum(p.numel() for p in disc.parameters()),'generator_parameter_count':sum(p.numel() for p in gen.parameters()),'rtd_head_parameter_count':sum(p.numel() for p in rtd.parameters()),'word_exposure':cum,'example_pool_words_actual':pool_words,'selected_for_training_words':actual_words,'actual_training_steps':total_steps,'loss_first':logs[0]['total_loss'] if logs else None,'loss_last':logs[-1]['total_loss'] if logs else None,'gen_loss_last':logs[-1]['gen_loss'] if logs else None,'disc_mlm_loss_last':logs[-1]['disc_mlm_loss'] if logs else None,'rtd_loss_last':logs[-1]['rtd_loss'] if logs else None,'rtd_acc_last':logs[-1]['rtd_acc'] if logs else None,'rtd_pred_original_rate_last':logs[-1]['rtd_pred_original_rate'] if logs else None,'rtd_label_original_rate_last':logs[-1]['rtd_label_original_rate'] if logs else None,'rtd_replaced_recall_last':logs[-1]['rtd_replaced_recall'] if logs else None,'rtd_original_recall_last':logs[-1]['rtd_original_recall'] if logs else None,'rtd_above_majority_last':logs[-1]['rtd_above_majority'] if logs else None,'masked_tokens_total':masked_total,'active_tokens_total':active_total,'replaced_tokens_total':replaced_total,'replaced_token_fraction_active':replaced_total/active_total if active_total else None,'masked_tokens_per_whitespace_word':masked_total/cum if cum else None,'lambda_rtd':args.lambda_rtd,'mask_mode':args.mask_mode,'mask_prob':args.mask_prob,'batch_size':args.batch_size,'seed':args.seed,'extra_init_seed':args.extra_init_seed,'train_rng_seed':args.train_rng_seed,'hidden_size':args.hidden_size,'n_layer':args.n_layer,'n_head':args.n_head,'position_buckets':args.position_buckets,'max_relative_positions':args.max_relative_positions,'deberta_relative_attention':args.deberta_relative_attention,'deberta_pos_att_type':args.deberta_pos_att_type,'tokenization_coupling_summary':tok_sum,'source_words_consumed':source_words,'saved_checkpoints':saved}
    (out/'scientific_metrics.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'event':'done','metrics':str(out/'scientific_metrics.json'),'disc_params':metrics['discriminator_parameter_count'],'word_exposure':cum,'rtd_acc_last':metrics['rtd_acc_last'],'rtd_pred_original_rate_last':metrics['rtd_pred_original_rate_last'],'rtd_label_original_rate_last':metrics['rtd_label_original_rate_last'],'rtd_replaced_recall_last':metrics['rtd_replaced_recall_last'],'rtd_above_majority_last':metrics['rtd_above_majority_last']}), flush=True)

if __name__ == '__main__':
    main()
