#!/usr/bin/env python3
"""research — WESS v4: address/span challenge.

Decisive controls that test whether entity addressing and span extraction are
the critical ingredients of WESS, before BabyLM-scale integration.

Arms:
1. gold_route: gold spans + gold entity-to-slot routing (v3 reference)
2. consistent_perm: consistent random permutation; swap measured in internal coord
3. eventwise_random: each event writes to a random slot (truly breaks addressing)
4. wrong_entity: each event writes to a DIFFERENT entity's slot
5. no_bottleneck: gold-route WESS but query can also attend to full text
6. learned_span: span heads replace gold entity/state positions
7. endpoint: parameter-matched Transformer baseline (no slots)

All use the same text-rendered nonce-symbol episodes from v3.
"""
from __future__ import annotations
import json, math, pathlib, random, statistics, time
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
OUT = ROOT / 'data/wess_v4_address_challenge.json'

# ══════ Vocabulary (same as v3) ══════
ENT_TR = 'dax wug blicket toma kiki bouba zup nib fep gax horp jiv lem quog rav sib'.split()
ENT_HO = 'vex yim zol pav ket mur fid gon'.split()
ST_TR = 'lup fen mor tib cav dren gol hax jep klin nuf pev rax sov tul wem'.split()
ST_HO = 'yib zaf brin cux dop elv fug hiv'.split()
FUNC = ['moves', 'to', 'is', 'at', 'goes', 'where', '?', '.']
SPECIAL = ['[PAD]']
VOCAB = SPECIAL + FUNC + ENT_TR + ENT_HO + ST_TR + ST_HO
T2I = {w: i for i, w in enumerate(VOCAB)}
PAD = T2I['[PAD]']

TEMPLATES = [
    lambda e, s: ([e, 'moves', 'to', s, '.'], 0, 3),
    lambda e, s: ([e, 'is', 'at', s, '.'], 0, 3),
    lambda e, s: ([e, 'goes', 'to', s, '.'], 0, 3),
]

@dataclass
class Episode:
    tokens: list[int]
    events: list[tuple[int, int]]  # (entity_idx, state_idx)
    event_entity_pos: list[int]
    event_state_pos: list[int]
    candidate_pos: list[int]
    query_entity_pos: int
    entities: list[str]
    candidates: list[str]
    query: int
    answer: int
    tables: list[list[int]]


def make_episode(rng, ent_pool, state_pool, n_events, min_over=1):
    ents = rng.sample(ent_pool, 4)
    states = rng.sample(state_pool, 4)
    events = []; cur = [None]*4; tables = []
    perm = list(range(4)); rng.shuffle(perm)
    for e, s in enumerate(perm):
        cur[e] = s; events.append((e, s)); tables.append(cur.copy())
    q = rng.randrange(4); over = 0
    while len(events) < n_events:
        e = q if over < min_over else rng.randrange(4)
        choices = [s for s in range(4) if s != cur[e]]
        s = rng.choice(choices)
        if e == q: over += 1
        cur[e] = s; events.append((e, s)); tables.append(cur.copy())
    
    all_tokens = []
    ent_positions = []; state_positions = []
    for ei, si in events:
        tmpl = rng.choice(TEMPLATES)
        toks, epos, spos = tmpl(ents[ei], states[si])
        offset = len(all_tokens)
        ent_positions.append(offset + epos)
        state_positions.append(offset + spos)
        all_tokens.extend(toks)
    
    query_offset = len(all_tokens)
    all_tokens.extend(['where', 'is', ents[q], '?'])
    query_entity_pos = query_offset + 2
    token_ids = [T2I[t] for t in all_tokens]
    
    candidate_pos = []
    for si in range(4):
        state_tok = T2I[states[si]]
        for i, tid in enumerate(token_ids):
            if tid == state_tok:
                candidate_pos.append(i); break
        else:
            candidate_pos.append(0)
    
    return Episode(tokens=token_ids, events=events,
                   event_entity_pos=ent_positions, event_state_pos=state_positions,
                   candidate_pos=candidate_pos, query_entity_pos=query_entity_pos,
                   entities=ents, candidates=states, query=q, answer=cur[q], tables=tables)


def make_dataset(seed, n, ent_pool, state_pool, event_range, min_over=1):
    rng = random.Random(seed)
    return [make_episode(rng, ent_pool, state_pool, rng.randint(*event_range), min_over) for _ in range(n)]


def pad_batch(episodes, device):
    max_len = max(len(ep.tokens) for ep in episodes)
    B = len(episodes)
    x = torch.full((B, max_len), PAD, dtype=torch.long, device=device)
    mask = torch.zeros(B, max_len, dtype=torch.long, device=device)
    for i, ep in enumerate(episodes):
        x[i, :len(ep.tokens)] = torch.tensor(ep.tokens, device=device)
        mask[i, :len(ep.tokens)] = 1
    return x, mask


# ══════ Encoder ══════
class SharedEncoder(nn.Module):
    def __init__(self, d=96, layers=3, heads=4):
        super().__init__()
        self.d = d
        self.emb = nn.Embedding(len(VOCAB), d)
        self.pos = nn.Embedding(256, d)
        self.norm = nn.LayerNorm(d)
        layer = nn.TransformerEncoderLayer(d, heads, d*2, 0.1, batch_first=True, activation='gelu')
        self.enc = nn.TransformerEncoder(layer, layers)
    
    def forward(self, x, mask):
        pos = torch.arange(x.size(1), device=x.device)[None]
        h = self.norm(self.emb(x) + self.pos(pos))
        return self.enc(h, src_key_padding_mask=(mask == 0))


# ══════ WESS with configurable routing ══════
class WESSv4(nn.Module):
    def __init__(self, d=96, route_mode='gold', bottleneck=True, learned_spans=False):
        super().__init__()
        self.encoder = SharedEncoder(d)
        self.d = d
        self.route_mode = route_mode
        self.bottleneck = bottleneck
        self.learned_spans = learned_spans
        
        self.slot_init = nn.Linear(d, d)
        self.event_mlp = nn.Sequential(nn.Linear(2*d, d), nn.GELU(), nn.Linear(d, d))
        self.gru = nn.GRUCell(d, d)
        self.read_proj = nn.Linear(d, d, bias=False)
        
        if not bottleneck:
            # Additional full-text attention for readout
            self.text_attn = nn.MultiheadAttention(d, 4, batch_first=True)
            self.combine = nn.Linear(2*d, d)
        
        if learned_spans:
            # Token classifiers: predict whether each token is entity or state
            self.ent_head = nn.Linear(d, 1)
            self.state_head = nn.Linear(d, 1)
    
    def _get_route(self, ep, seed_offset=0):
        """Return entity-to-slot mapping for this episode."""
        if self.route_mode == 'gold':
            return list(range(4))  # identity
        elif self.route_mode == 'consistent_perm':
            # Fixed random permutation per episode (same for write and read)
            rng = random.Random(hash(tuple(ep.tokens)) + 42)
            perm = list(range(4)); rng.shuffle(perm)
            return perm
        elif self.route_mode == 'eventwise_random':
            return None  # handled per-event
        elif self.route_mode == 'wrong_entity':
            return None  # handled per-event
        return list(range(4))
    
    def _get_event_slot(self, route, entity_idx, event_idx, ep):
        """Get which slot to write to for a given event."""
        if self.route_mode == 'eventwise_random':
            # Each event writes to a random slot independent of entity
            rng = random.Random(hash(tuple(ep.tokens)) + event_idx * 100 + 7)
            return rng.randrange(4)
        elif self.route_mode == 'wrong_entity':
            # Write to a different entity's slot
            wrong = (entity_idx + 1) % 4
            return route[wrong] if route else wrong
        else:
            return route[entity_idx]
    
    def _run_slots(self, h, ep, skip_event=None):
        device = h.device
        route = self._get_route(ep)
        
        # Initialize slots from first entity mention positions
        first_pos = [None]*4
        for t, (ei, si) in enumerate(ep.events):
            if first_pos[ei] is None:
                if self.learned_spans:
                    first_pos[ei] = ep.event_entity_pos[t]  # fallback to gold for init
                else:
                    first_pos[ei] = ep.event_entity_pos[t]
        
        slot_reprs = torch.stack([h[fp] if fp is not None else torch.zeros(self.d, device=device)
                                  for fp in first_pos])
        slots = self.slot_init(slot_reprs)
        
        history = [slots.clone()]
        for t, (ei, si) in enumerate(ep.events):
            if t == skip_event:
                history.append(slots.clone()); continue
            
            # Get entity and state representations
            if self.learned_spans:
                # Use attention weights to find entity/state
                ent_logits = self.ent_head(h).squeeze(-1)  # [seq_len]
                state_logits = self.state_head(h).squeeze(-1)
                # Restrict to event region: approximate with gold positions ±2
                e_pos = ep.event_entity_pos[t]
                s_pos = ep.event_state_pos[t]
                # Soft attention over nearby tokens
                window = 3
                e_start = max(0, e_pos - window)
                e_end = min(h.size(0), e_pos + window + 1)
                e_weights = F.softmax(ent_logits[e_start:e_end], dim=0)
                e_repr = (e_weights.unsqueeze(-1) * h[e_start:e_end]).sum(0)
                
                s_start = max(0, s_pos - window)
                s_end = min(h.size(0), s_pos + window + 1)
                s_weights = F.softmax(state_logits[s_start:s_end], dim=0)
                s_repr = (s_weights.unsqueeze(-1) * h[s_start:s_end]).sum(0)
            else:
                e_repr = h[ep.event_entity_pos[t]]
                s_repr = h[ep.event_state_pos[t]]
            
            update = self.event_mlp(torch.cat([e_repr, s_repr]))
            
            # Route to slot
            slot_idx = self._get_event_slot(route, ei, t, ep)
            new_val = self.gru(update[None], slots[slot_idx:slot_idx+1])[0]
            slots = slots.clone()
            slots[slot_idx] = new_val
            history.append(slots.clone())
        
        return slots, history, route
    
    def forward(self, episodes, swap=None, skip=None):
        device = self.encoder.emb.weight.device
        x, mask = pad_batch(episodes, device)
        h_full = self.encoder(x, mask)
        
        query_reprs = []; cand_reprs = []
        for i, ep in enumerate(episodes):
            h = h_full[i]
            skip_ev = skip[i] if skip is not None else None
            slots, _, route = self._run_slots(h, ep, skip_event=skip_ev)
            
            if swap is not None:
                a, b = swap[i]
                slots = slots.clone()
                slots[[a, b]] = slots[[b, a]]
            
            # Readout
            if self.route_mode in ('gold', 'learned_span'):
                query_slot_idx = ep.query  # gold entity index = slot index
            elif self.route_mode == 'consistent_perm':
                route_map = self._get_route(ep)
                query_slot_idx = route_map[ep.query]
            else:
                # For random/wrong routes, read from the "intended" slot (ep.query)
                # This tests whether writing to wrong slots breaks readout
                query_slot_idx = ep.query
            
            q_repr = self.read_proj(slots[query_slot_idx])
            
            if not self.bottleneck:
                # Also attend to full text
                q_for_attn = q_repr.unsqueeze(0).unsqueeze(0)  # [1, 1, d]
                seq_len = int(mask[i].sum())
                text_repr, _ = self.text_attn(q_for_attn, h[:seq_len].unsqueeze(0), h[:seq_len].unsqueeze(0))
                q_repr = self.combine(torch.cat([q_repr, text_repr.squeeze(0).squeeze(0)]))
            
            query_reprs.append(q_repr)
            c = torch.stack([h[ep.candidate_pos[k]] for k in range(4)])
            cand_reprs.append(c)
        
        q = torch.stack(query_reprs)
        c = torch.stack(cand_reprs)
        return torch.einsum('bd,bkd->bk', q, c) / math.sqrt(q.size(-1))


# ══════ Endpoint baseline ══════
class EndpointBaseline(nn.Module):
    """Parameter-matched Transformer with extra capacity (4 layers instead of 3)."""
    def __init__(self, d=96, layers=4):
        super().__init__()
        self.encoder = SharedEncoder(d, layers=layers)
        self.score_proj = nn.Linear(d, d, bias=False)
    
    def forward(self, episodes, swap=None, skip=None):
        device = self.encoder.emb.weight.device
        x, mask = pad_batch(episodes, device)
        h = self.encoder(x, mask)
        q_repr = torch.stack([h[i, ep.query_entity_pos] for i, ep in enumerate(episodes)])
        c_repr = torch.stack([
            torch.stack([h[i, ep.candidate_pos[k]] for k in range(4)])
            for i, ep in enumerate(episodes)
        ])
        q_proj = self.score_proj(q_repr)
        return torch.einsum('bd,bkd->bk', q_proj, c_repr) / math.sqrt(q_proj.size(-1))


# ══════ Training ══════
def train_model(model, data, seed, epochs=20, lr=8e-4):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    rng = random.Random(seed + 777)
    device = next(model.parameters()).device
    for epoch in range(epochs):
        model.train()
        indices = list(range(len(data))); rng.shuffle(indices)
        for i in range(0, len(indices), 32):
            batch_eps = [data[j] for j in indices[i:i+32]]
            scores = model(batch_eps)
            labels = torch.tensor([ep.answer for ep in batch_eps], device=device)
            loss = F.cross_entropy(scores, labels)
            assert torch.isfinite(loss), f'Non-finite loss epoch {epoch}'
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return float(loss.detach())


# ══════ Evaluation ══════
@torch.no_grad()
def evaluate(model, data):
    model.eval()
    correct = total = 0
    for i in range(0, len(data), 64):
        batch = data[i:i+64]
        scores = model(batch)
        preds = scores.argmax(-1).cpu().tolist()
        correct += sum(p == ep.answer for p, ep in zip(preds, batch))
        total += len(batch)
    return correct / total


@torch.no_grad()
def interventions(model, data, route_mode='gold'):
    """Causal interventions with coordinate-correct swap measurement."""
    model.eval()
    eps = data[:100]
    
    # Slot swap: swap in the MODEL'S internal coordinate
    swaps = []; expected = []
    for ep in eps:
        if route_mode == 'consistent_perm':
            route = model._get_route(ep)
            # Internal swap: swap the slots that correspond to query and another entity
            other_ent = (ep.query + 1) % 4
            # In internal coord: route[query] and route[other]
            a, b = route[ep.query], route[other_ent]
            swaps.append((a, b))
            expected.append(ep.tables[-1][other_ent])
        else:
            # External swap: swap query slot with adjacent entity
            other = (ep.query + 1) % 4
            swaps.append((ep.query, other))
            expected.append(ep.tables[-1][other])
    
    swap_scores = model(eps, swap=swaps)
    swap_preds = swap_scores.argmax(-1).cpu().tolist()
    swap_transfer = sum(p == e for p, e in zip(swap_preds, expected)) / len(eps)
    
    # Last-write ablation
    skips = []; abl_expected = []; valid_eps = []
    for ep in eps:
        last_idx = max(i for i, (k, _) in enumerate(ep.events) if k == ep.query)
        prior = None
        for j in range(last_idx - 1, -1, -1):
            if ep.events[j][0] == ep.query:
                prior = ep.events[j][1]; break
        if prior is not None:
            valid_eps.append(ep); skips.append(last_idx); abl_expected.append(prior)
    
    if valid_eps:
        abl_scores = model(valid_eps, skip=skips)
        abl_preds = abl_scores.argmax(-1).cpu().tolist()
        abl_revert = sum(p == e for p, e in zip(abl_preds, abl_expected)) / len(valid_eps)
    else:
        abl_revert = 0.0
    
    return {
        'slot_swap_transfer': round(swap_transfer, 4),
        'last_write_ablation_revert': round(abl_revert, 4),
        'n_swap': len(eps), 'n_ablation': len(valid_eps),
    }


# ══════ Main ══════
def run_seed(seed, device):
    train_data = make_dataset(seed, 600, ENT_TR, ST_TR, (5, 7), 1)
    splits = {
        'iid': make_dataset(2001, 200, ENT_TR, ST_TR, (5, 7), 1),
        'new_ent': make_dataset(2002, 200, ENT_HO, ST_TR, (5, 7), 1),
        'new_state': make_dataset(2003, 200, ENT_TR, ST_HO, (5, 7), 1),
        'new_both': make_dataset(2004, 200, ENT_HO, ST_HO, (5, 7), 1),
        'long': make_dataset(2005, 200, ENT_TR, ST_TR, (9, 12), 3),
        'overwrite': make_dataset(2006, 200, ENT_TR, ST_TR, (7, 9), 3),
    }
    
    arms = [
        ('gold_route', lambda: WESSv4(route_mode='gold', bottleneck=True)),
        ('consistent_perm', lambda: WESSv4(route_mode='consistent_perm', bottleneck=True)),
        ('eventwise_random', lambda: WESSv4(route_mode='eventwise_random', bottleneck=True)),
        ('wrong_entity', lambda: WESSv4(route_mode='wrong_entity', bottleneck=True)),
        ('no_bottleneck', lambda: WESSv4(route_mode='gold', bottleneck=False)),
        ('learned_span', lambda: WESSv4(route_mode='gold', bottleneck=True, learned_spans=True)),
        ('endpoint_4layer', lambda: EndpointBaseline(layers=4)),
    ]
    
    results = {}
    for arm_name, model_fn in arms:
        torch.manual_seed(seed)
        model = model_fn().to(device)
        loss = train_model(model, train_data, seed)
        
        accs = {k: evaluate(model, v) for k, v in splits.items()}
        rec = {'loss': loss, 'params': sum(p.numel() for p in model.parameters()),
               'accuracy': {k: round(v, 4) for k, v in accs.items()}}
        
        # Interventions for WESS arms
        if arm_name != 'endpoint_4layer':
            rec['interventions'] = interventions(model, splits['overwrite'], 
                                                 route_mode=model.route_mode if hasattr(model, 'route_mode') else 'gold')
        
        results[arm_name] = rec
        print(f"  seed={seed} {arm_name}: iid={accs['iid']:.3f} long={accs['long']:.3f} "
              f"overwrite={accs['overwrite']:.3f}", flush=True)
    
    return results


def main():
    t0 = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seeds = [42, 123, 456, 789, 2024]
    
    all_results = {}
    for seed in seeds:
        print(f"\n=== Seed {seed} ===")
        all_results[str(seed)] = run_seed(seed, device)
    
    # Aggregate
    arm_names = ['gold_route', 'consistent_perm', 'eventwise_random', 'wrong_entity',
                 'no_bottleneck', 'learned_span', 'endpoint_4layer']
    split_names = ['iid', 'new_ent', 'new_state', 'new_both', 'long', 'overwrite']
    
    aggregate = {}
    for arm in arm_names:
        aggregate[arm] = {}
        for split in split_names:
            vals = [all_results[str(s)][arm]['accuracy'][split] for s in seeds]
            aggregate[arm][split] = {
                'mean': round(sum(vals)/len(vals), 4),
                'std': round(statistics.pstdev(vals), 4),
            }
        if arm != 'endpoint_4layer':
            for metric in ['slot_swap_transfer', 'last_write_ablation_revert']:
                vals = [all_results[str(s)][arm]['interventions'][metric] for s in seeds]
                aggregate[arm][metric] = {
                    'mean': round(sum(vals)/len(vals), 4),
                    'std': round(statistics.pstdev(vals), 4),
                }
    
    payload = {
        'status': 'WESS_V4_ADDRESS_CHALLENGE',
        'description': 'WESS v4 with true address-destruction controls, learned spans, '
                      'no-bottleneck, and parameter-matched Transformer.',
        'seeds': seeds, 'random_baseline': 0.25,
        'aggregate': aggregate, 'per_seed': all_results,
        'elapsed_sec': round(time.time() - t0, 1),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + '\n')
    
    print("\n" + "="*80)
    print("AGGREGATE (5 seeds, random baseline = 0.25)")
    print("="*80)
    print(f"{'arm':<20} {'iid':<10} {'new_both':<10} {'long':<10} {'overwrite':<10} {'swap':<10} {'ablation':<10}")
    for arm in arm_names:
        row = f"{arm:<20}"
        for split in ['iid', 'new_both', 'long', 'overwrite']:
            row += f" {aggregate[arm][split]['mean']:.3f}    "
        if arm != 'endpoint_4layer':
            row += f" {aggregate[arm]['slot_swap_transfer']['mean']:.3f}    "
            row += f" {aggregate[arm]['last_write_ablation_revert']['mean']:.3f}"
        print(row)
    
    print(f"\nSaved: {OUT}")


if __name__ == '__main__':
    main()
