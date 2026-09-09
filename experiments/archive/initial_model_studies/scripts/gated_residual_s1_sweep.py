#!/usr/bin/env python3
"""research — Fix S1-depth WESS scaling failure with gated residual fusion.

Root cause of 384×12 failure: the fusion completely REPLACES the encoder hidden
state at the mask position. For a deep model this destroys the contextual MLM
representation. Fix: use h + gate * proj(slot) where gate starts near 0.

Also tests lower LR for better deep-model stability.
"""
from __future__ import annotations
import importlib.util, json, pathlib, random, statistics, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, DebertaV2Config, DebertaV2Model, get_cosine_schedule_with_warmup

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
SRC = ROOT / 'scripts/wess_mlm_transfer_pilot.py'
OUT = ROOT / 'data/gated_residual_s1_sweep.json'

spec = importlib.util.spec_from_file_location('bridge', SRC)
bridge = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bridge
spec.loader.exec_module(bridge)


def load_official(tok, n, seed, max_len=192):
    rng = random.Random(seed); rows = []
    for fn in ['childes.train.txt','bnc_spoken.train.txt','simple_wiki.train.txt','open_subtitles.train.txt','qed.train.txt']:
        p = bridge.RAW / fn
        if not p.exists(): continue
        with p.open('r', encoding='utf-8', errors='replace') as f:
            for line in f:
                text = line.strip(); wc = len(text.split())
                if 6 <= wc <= 120: rows.append(text)
                if len(rows) >= n * 8: break
    rng.shuffle(rows); examples = []
    for text in rows:
        enc = tok(text, add_special_tokens=False, truncation=True, max_length=max_len)
        ids = list(enc['input_ids'])
        if len(ids) < 8: continue
        labels = [-100] * len(ids)
        cand = [i for i, t in enumerate(ids) if t not in {tok.pad_token_id, tok.mask_token_id}]
        if not cand: continue
        mpos = rng.choice(cand); labels[mpos] = ids[mpos]; ids[mpos] = tok.mask_token_id
        examples.append({'input_ids': ids, 'labels': labels})
        if len(examples) >= n: break
    return examples


class GatedResidualWESS(nn.Module):
    """WESS with gated residual fusion: h_mask + gate * proj(slot).
    
    gate is a learnable scalar initialized to init_alpha (default 0.0),
    so the model starts as a pure MLM and gradually learns to use slots.
    """
    def __init__(self, tok, arm: str, hidden: int, layers: int, heads: int,
                 intermediate: int, init_alpha: float = 0.0):
        super().__init__()
        self.arm = arm; self.tok = tok; self.hidden = hidden
        cfg = DebertaV2Config(
            vocab_size=len(tok), hidden_size=hidden, num_hidden_layers=layers,
            num_attention_heads=heads, intermediate_size=intermediate,
            max_position_embeddings=512, position_buckets=256,
            relative_attention=True, pos_att_type=['p2c', 'c2p'],
            pad_token_id=tok.pad_token_id)
        self.enc = DebertaV2Model(cfg)
        self.bias = nn.Parameter(torch.zeros(len(tok)))
        # Slot components
        self.slot_init = nn.Linear(hidden, hidden)
        self.event_mlp = nn.Sequential(nn.Linear(2*hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.gru = nn.GRUCell(hidden, hidden)
        # Gated residual fusion: slot_proj maps slot to hidden, gate scales it
        self.slot_proj = nn.Linear(hidden, hidden)
        self.fusion_gate = nn.Parameter(torch.tensor(init_alpha))
    
    def route_slot(self, ep, ei: int, t: int):
        if self.arm == 'wess_gold': return ei
        if self.arm == 'wess_eventwise_random': return random.Random(ep.pair_id*1000+t*17+3).randrange(2)
        if self.arm == 'wess_wrong_entity': return 1 - ei
        if self.arm == 'no_address': return 0
        return ei
    
    def run_memory(self, h, ep, skip_last_query=False, swap=False):
        if self.arm == 'plain_mlm': return None
        ent0 = h[ep.event_entity_pos[0]]; ent1 = None
        for idx, (ei, _) in enumerate(ep.events):
            if ei == 1: ent1 = h[ep.event_entity_pos[idx]]; break
        if ent1 is None: ent1 = ent0
        slot_list = [self.slot_init(ent0), self.slot_init(ent1)]
        if self.arm == 'no_address': slot_list[1] = slot_list[0]
        last_q = max(i for i, (ei, _) in enumerate(ep.events) if ei == ep.query_entity)
        for t, (ei, _) in enumerate(ep.events):
            if skip_last_query and t == last_q: continue
            e_repr = h[ep.event_entity_pos[t]]; s_repr = h[ep.event_state_pos[t]]
            upd = self.event_mlp(torch.cat([e_repr, s_repr]))
            si = self.route_slot(ep, ei, t)
            new_val = self.gru(upd.unsqueeze(0), slot_list[si].unsqueeze(0)).squeeze(0)
            slot_list[si] = new_val
            if self.arm == 'no_address': slot_list[1] = slot_list[0]
        if swap and self.arm in {'wess_gold', 'wess_eventwise_random', 'wess_wrong_entity'}:
            slot_list[0], slot_list[1] = slot_list[1], slot_list[0]
        return slot_list[ep.query_entity if self.arm != 'no_address' else 0]
    
    def forward(self, input_ids, attention_mask, labels=None, eps=None, intervention=None):
        h = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if eps is not None and self.arm != 'plain_mlm':
            h_new = h.clone()
            for b, ep in enumerate(eps):
                if ep is None: continue
                slot = self.run_memory(h[b], ep,
                                       skip_last_query=(intervention == 'ablate'),
                                       swap=(intervention == 'swap'))
                if slot is not None:
                    m = ep.mask_pos
                    # GATED RESIDUAL: h + gate * proj(slot)
                    slot_contribution = self.slot_proj(slot)
                    h_new = h_new.clone()
                    h_new[b, m] = h[b, m] + self.fusion_gate * slot_contribution
            h = h_new
        logits = torch.matmul(h, self.enc.embeddings.word_embeddings.weight.t()) + self.bias
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)
        return logits, loss


def train_arm(tok, arm, train_eps, official, steps, batch_size, lr, seed, device, log_every=100):
    torch.manual_seed(seed); random.seed(seed)
    model = GatedResidualWESS(tok, arm, hidden=384, layers=12, heads=12,
                               intermediate=1280, init_alpha=0.0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    warmup = max(1, int(steps * 0.05))
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=warmup, num_training_steps=steps)
    rng = random.Random(seed + 31)
    losses = []; n_ep = max(1, round(batch_size * 0.25))  # 25% episode ratio for stronger signal
    for step in range(steps):
        ep_batch = [rng.choice(train_eps) for _ in range(n_ep)]
        off_batch = [rng.choice(official) for _ in range(batch_size - len(ep_batch))]
        items = ep_batch + off_batch; rng.shuffle(items)
        x, a, l, _, eps = bridge.pad_features(items, tok.pad_token_id, device)
        _, loss = model(x, a, l, eps)
        assert torch.isfinite(loss), f'nonfinite {arm} step {step}'
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        losses.append(float(loss.detach()))
        if log_every and (step + 1) % log_every == 0:
            gate_val = float(model.fusion_gate.detach()) if hasattr(model, 'fusion_gate') else 0
            print(f'  {arm} step {step+1}/{steps} loss={losses[-1]:.4f} gate={gate_val:.4f}', flush=True)
    return model, losses


def main():
    bridge.setup_env(); t0 = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tok = AutoTokenizer.from_pretrained(str(bridge.TOK_PATH), use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token is not None else tok.mask_token
    
    train_eps = bridge.build_pairs(tok, 1200, 421)
    eval_eps = bridge.build_pairs(tok, 240, 422)
    official = load_official(tok, 5000, 423)
    audit = bridge.shortcut_audit(eval_eps)
    print(f'Data: {len(train_eps)} train eps, {len(eval_eps)} eval eps, {len(official)} official', flush=True)
    print(f'Shortcut audit: {json.dumps(audit)}', flush=True)
    
    # Sweep: test gated residual at two LRs
    configs = [
        {'name': 's1_gated_lr1e3_500', 'lr': 1e-3, 'steps': 500},
        {'name': 's1_gated_lr5e4_500', 'lr': 5e-4, 'steps': 500},
        {'name': 's1_gated_lr5e4_1000', 'lr': 5e-4, 'steps': 1000},
    ]
    arms = ['plain_mlm', 'no_address', 'wess_gold', 'wess_eventwise_random']
    
    all_results = {}
    for cfg in configs:
        print(f"\n{'='*60}\nConfig: {cfg['name']} (lr={cfg['lr']}, steps={cfg['steps']})\n{'='*60}", flush=True)
        cfg_results = {}
        for arm in arms:
            print(f'ARM {arm}', flush=True)
            model, losses = train_arm(tok, arm, train_eps, official,
                                       steps=cfg['steps'], batch_size=32,
                                       lr=cfg['lr'], seed=42, device=device,
                                       log_every=cfg['steps']//5)
            bind = bridge.eval_binding(tok, model, eval_eps, device)
            off_loss = bridge.eval_official_loss(tok, model, official[:256], device)
            rec = {'binding': bind, 'official_mlm_loss': off_loss,
                   'loss_last': losses[-1] if losses else None,
                   'loss_mean_last20': sum(losses[-20:])/max(1, len(losses[-20:])),
                   'params': sum(p.numel() for p in model.parameters())}
            if arm != 'plain_mlm':
                rec['interventions'] = bridge.eval_interventions(tok, model, eval_eps[:160], device)
                if hasattr(model, 'fusion_gate'):
                    rec['learned_gate'] = float(model.fusion_gate.detach())
            cfg_results[arm] = rec
            print(f'  {arm}: pair_acc={bind["pair_acc"]:.4f} logodds={bind["mean_logodds"]:.4f} '
                  f'off_loss={off_loss:.4f}', flush=True)
            if 'interventions' in rec:
                print(f'    swap={rec["interventions"]["swap_logodds_delta_other_minus_answer"]:.4f} '
                      f'ablate={rec["interventions"]["ablation_logodds_delta_prev_minus_answer"]:.4f}', flush=True)
            if 'learned_gate' in rec:
                print(f'    learned_gate={rec["learned_gate"]:.4f}', flush=True)
            del model
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        all_results[cfg['name']] = cfg_results
    
    payload = {
        'status': 'GATED_RESIDUAL_S1_SWEEP',
        'description': 'S1-depth (12×384) WESS with gated residual fusion (h + gate*proj(slot), gate init 0) '
                      'and 25% episode ratio. Tests whether preserving MLM representation fixes S1 scaling failure.',
        'configs': configs,
        'arms': arms,
        'shortcut_audit': audit,
        'results': all_results,
        'elapsed_sec': round(time.time() - t0, 1),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + '\n')
    print(f'\nSaved {OUT}', flush=True)


if __name__ == '__main__':
    main()
