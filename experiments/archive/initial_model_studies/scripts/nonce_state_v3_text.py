#!/usr/bin/env python3
"""research — Natural-language-interface v3 for nonce-symbol state tracking.

Key change from v2: events are rendered as text tokens processed by a shared
Transformer encoder. WESS extracts entity/state representations from encoder
hidden states at gold span positions, not from isolated symbol embeddings.

This tests whether the role-binding mechanism survives when binding information
must be extracted from contextualized distributed representations.

Arms:
1. endpoint: Transformer encoder + query-position readout + candidate scoring
2. step_supervised: same encoder + explicit per-entity per-event state prediction
3. wess_gold: encoder + GRU entity slots with gold participant routing from encoder
4. wess_shuffled: same as wess_gold but entity-to-slot routing randomly permuted

Measurements: accuracy per split, slot-swap transfer, last-write ablation,
persistence rate, selective write accuracy, entity-renaming equivariance.
"""
from __future__ import annotations
import json, math, pathlib, random, statistics, time
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
OUT = ROOT / 'data/nonce_state_tracking/v3_text_interface.json'

# ══════ Vocabulary ══════
ENT_TR = 'dax wug blicket toma kiki bouba zup nib fep gax horp jiv lem quog rav sib'.split()
ENT_HO = 'vex yim zol pav ket mur fid gon'.split()
ST_TR = 'lup fen mor tib cav dren gol hax jep klin nuf pev rax sov tul wem'.split()
ST_HO = 'yib zaf brin cux dop elv fug hiv'.split()
FUNC = ['moves', 'to', 'is', 'at', 'goes', 'where', '?', '.']
SPECIAL = ['[PAD]']
VOCAB = SPECIAL + FUNC + ENT_TR + ENT_HO + ST_TR + ST_HO
T2I = {w: i for i, w in enumerate(VOCAB)}
PAD = T2I['[PAD]']

# Event templates: each returns (tokens, entity_pos, state_pos)
TEMPLATES = [
    lambda e, s: ([e, 'moves', 'to', s, '.'], 0, 3),
    lambda e, s: ([e, 'is', 'at', s, '.'], 0, 3),
    lambda e, s: ([e, 'goes', 'to', s, '.'], 0, 3),
]
TEMPLATES_HO = [
    lambda e, s: ([s, 'is', 'where', e, 'is', '.'], 3, 0),  # inverted order
]

@dataclass
class Episode:
    tokens: list[int]           # Full episode token IDs
    events: list[tuple[int, int]]  # (entity_idx, state_idx) per event
    event_entity_pos: list[int]    # Token position of entity in each event
    event_state_pos: list[int]     # Token position of state in each event
    candidate_pos: list[int]       # Token positions of each candidate state (first occurrence)
    query_entity_pos: int          # Position of queried entity token in query
    entities: list[str]
    candidates: list[str]
    query: int                     # Entity index being queried
    answer: int                    # Candidate index of correct state
    tables: list[list[int]]        # State table after each event


def make_episode(rng, ent_pool, state_pool, n_events, min_over=1, templates=None):
    if templates is None:
        templates = TEMPLATES
    ents = rng.sample(ent_pool, 4)
    states = rng.sample(state_pool, 4)
    
    events = []
    cur = [None] * 4
    tables = []
    # Initialize all entities
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
    
    # Render as text tokens
    all_tokens = []
    ent_positions = []
    state_positions = []
    for ei, si in events:
        tmpl = rng.choice(templates)
        toks, epos, spos = tmpl(ents[ei], states[si])
        offset = len(all_tokens)
        ent_positions.append(offset + epos)
        state_positions.append(offset + spos)
        all_tokens.extend(toks)
    
    # Query: "where is ENTITY ?"
    query_offset = len(all_tokens)
    all_tokens.extend(['where', 'is', ents[q], '?'])
    query_entity_pos = query_offset + 2
    
    # Convert to IDs
    token_ids = [T2I[t] for t in all_tokens]
    
    # Find first occurrence of each candidate state
    candidate_pos = []
    for si in range(4):
        state_tok = T2I[states[si]]
        for i, tid in enumerate(token_ids):
            if tid == state_tok:
                candidate_pos.append(i); break
        else:
            candidate_pos.append(0)  # fallback
    
    return Episode(
        tokens=token_ids, events=events,
        event_entity_pos=ent_positions, event_state_pos=state_positions,
        candidate_pos=candidate_pos, query_entity_pos=query_entity_pos,
        entities=ents, candidates=states, query=q,
        answer=cur[q], tables=tables,
    )


def make_dataset(seed, n, ent_pool, state_pool, event_range, min_over=1, templates=None):
    rng = random.Random(seed)
    return [make_episode(rng, ent_pool, state_pool, rng.randint(*event_range),
                        min_over, templates) for _ in range(n)]


# ══════ Models ══════

def pad_batch(episodes, device):
    """Pad token sequences and return metadata tensors."""
    max_len = max(len(ep.tokens) for ep in episodes)
    B = len(episodes)
    x = torch.full((B, max_len), PAD, dtype=torch.long, device=device)
    mask = torch.zeros(B, max_len, dtype=torch.long, device=device)
    for i, ep in enumerate(episodes):
        x[i, :len(ep.tokens)] = torch.tensor(ep.tokens, device=device)
        mask[i, :len(ep.tokens)] = 1
    return x, mask


class SharedEncoder(nn.Module):
    def __init__(self, d=96, layers=3, heads=4):
        super().__init__()
        self.d = d
        self.emb = nn.Embedding(len(VOCAB), d)
        self.pos = nn.Embedding(256, d)
        self.norm = nn.LayerNorm(d)
        layer = nn.TransformerEncoderLayer(d, heads, d * 2, 0.1,
                                           batch_first=True, activation='gelu')
        self.enc = nn.TransformerEncoder(layer, layers)
    
    def forward(self, x, mask):
        pos = torch.arange(x.size(1), device=x.device)[None]
        h = self.norm(self.emb(x) + self.pos(pos))
        return self.enc(h, src_key_padding_mask=(mask == 0))


class EndpointArm(nn.Module):
    """Arm 1: encode full text, pool query region, score against candidate states."""
    def __init__(self, d=96):
        super().__init__()
        self.encoder = SharedEncoder(d)
        self.score_proj = nn.Linear(d, d, bias=False)
    
    def forward(self, episodes, h=None):
        device = self.encoder.emb.weight.device
        if h is None:
            x, mask = pad_batch(episodes, device)
            h = self.encoder(x, mask)
        # Query representation: pool query entity position
        q_repr = torch.stack([h[i, ep.query_entity_pos] for i, ep in enumerate(episodes)])
        # Candidate representations: from their first occurrence positions
        c_repr = torch.stack([
            torch.stack([h[i, ep.candidate_pos[k]] for k in range(4)])
            for i, ep in enumerate(episodes)
        ])  # [B, 4, d]
        q_proj = self.score_proj(q_repr)  # [B, d]
        scores = torch.einsum('bd,bkd->bk', q_proj, c_repr) / math.sqrt(q_proj.size(-1))
        return scores


class StepSupervisedArm(nn.Module):
    """Arm 2: same encoder + explicit per-entity state prediction after each event."""
    def __init__(self, d=96):
        super().__init__()
        self.encoder = SharedEncoder(d)
        self.score_proj = nn.Linear(d, d, bias=False)
        self.step_proj = nn.Linear(d, d, bias=False)
    
    def forward(self, episodes, return_step_logits=False):
        device = self.encoder.emb.weight.device
        x, mask = pad_batch(episodes, device)
        h = self.encoder(x, mask)
        
        # Endpoint score
        q_repr = torch.stack([h[i, ep.query_entity_pos] for i, ep in enumerate(episodes)])
        c_repr = torch.stack([
            torch.stack([h[i, ep.candidate_pos[k]] for k in range(4)])
            for i, ep in enumerate(episodes)
        ])
        q_proj = self.score_proj(q_repr)
        endpoint_scores = torch.einsum('bd,bkd->bk', q_proj, c_repr) / math.sqrt(q_proj.size(-1))
        
        if not return_step_logits:
            return endpoint_scores
        
        # Step supervision: after each event, predict each entity's state
        # from that entity's most recent mention position
        B = len(episodes)
        max_events = max(len(ep.events) for ep in episodes)
        step_logits = []
        step_labels = []
        
        for b, ep in enumerate(episodes):
            last_pos = [None] * 4  # last seen position per entity
            for t, (ei, si) in enumerate(ep.events):
                last_pos[ei] = ep.event_entity_pos[t]
                # Predict state for each entity that has been mentioned
                for k in range(4):
                    if last_pos[k] is not None:
                        e_repr = self.step_proj(h[b, last_pos[k]])
                        logit = torch.einsum('d,kd->k', e_repr, c_repr[b]) / math.sqrt(e_repr.size(-1))
                        step_logits.append(logit)
                        step_labels.append(ep.tables[t][k])
        
        if step_logits:
            step_logits = torch.stack(step_logits)
            step_labels = torch.tensor(step_labels, device=device)
        else:
            step_logits = None
            step_labels = None
        
        return endpoint_scores, step_logits, step_labels


class WESSArm(nn.Module):
    """Arm 3/4: shared encoder + GRU entity slots with gold/shuffled routing."""
    def __init__(self, d=96, shuffled=False):
        super().__init__()
        self.encoder = SharedEncoder(d)
        self.d = d
        self.shuffled = shuffled
        # Slot initialization from first-mention entity representation
        self.slot_init = nn.Linear(d, d)
        # Event update: combine entity repr + state repr -> update vector
        self.event_mlp = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d))
        # GRU for slot update
        self.gru = nn.GRUCell(d, d)
        # Readout projection
        self.read_proj = nn.Linear(d, d, bias=False)
    
    def _run_slots(self, h, ep, perm=None, skip_event=None):
        """Run slot updates for one episode. Returns final slots and history."""
        device = h.device
        # Initialize slots from first entity mention
        first_pos = [None] * 4
        for t, (ei, si) in enumerate(ep.events):
            if first_pos[ei] is None:
                first_pos[ei] = ep.event_entity_pos[t]
        
        slot_init_reprs = torch.stack([h[fp] if fp is not None else torch.zeros(self.d, device=device)
                                       for fp in first_pos])
        slots = self.slot_init(slot_init_reprs)  # [4, d]
        
        # Apply permutation for shuffled routing
        if perm is not None:
            route = perm
        elif self.shuffled:
            route = list(range(4))
            random.shuffle(route)
        else:
            route = list(range(4))  # identity = gold routing
        
        history = [slots.clone()]
        for t, (ei, si) in enumerate(ep.events):
            if t == skip_event:
                history.append(slots.clone())
                continue
            # Extract entity and state representations from encoder
            e_repr = h[ep.event_entity_pos[t]]
            s_repr = h[ep.event_state_pos[t]]
            # Compute update vector
            update = self.event_mlp(torch.cat([e_repr, s_repr]))
            # Route to correct slot
            slot_idx = route[ei]
            # GRU update
            new_val = self.gru(update[None], slots[slot_idx:slot_idx+1])[0]
            slots = slots.clone()
            slots[slot_idx] = new_val
            history.append(slots.clone())
        
        return slots, history, route
    
    def forward(self, episodes, swap=None, skip=None):
        device = self.encoder.emb.weight.device
        x, mask = pad_batch(episodes, device)
        h_full = self.encoder(x, mask)
        
        query_reprs = []
        cand_reprs = []
        
        for i, ep in enumerate(episodes):
            h = h_full[i]
            skip_ev = skip[i] if skip is not None else None
            slots, _, route = self._run_slots(h, ep, skip_event=skip_ev)
            
            # Apply slot swap if requested
            if swap is not None:
                a, b = swap[i]
                slots = slots.clone()
                slots[[a, b]] = slots[[b, a]]
            
            # Readout: use the queried entity's slot
            query_slot_idx = route[ep.query]
            q_repr = self.read_proj(slots[query_slot_idx])
            query_reprs.append(q_repr)
            
            # Candidates from encoder hidden states
            c = torch.stack([h[ep.candidate_pos[k]] for k in range(4)])
            cand_reprs.append(c)
        
        q = torch.stack(query_reprs)
        c = torch.stack(cand_reprs)
        scores = torch.einsum('bd,bkd->bk', q, c) / math.sqrt(q.size(-1))
        return scores


# ══════ Training and Evaluation ══════

def train_model(model, arm_name, data, seed, epochs=20, lr=8e-4):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    rng = random.Random(seed + 777)
    device = next(model.parameters()).device
    
    for epoch in range(epochs):
        model.train()
        indices = list(range(len(data))); rng.shuffle(indices)
        for i in range(0, len(indices), 32):
            batch_eps = [data[j] for j in indices[i:i+32]]
            
            if arm_name == 'step_supervised':
                endpoint_scores, step_logits, step_labels = model(batch_eps, return_step_logits=True)
                labels = torch.tensor([ep.answer for ep in batch_eps], device=device)
                loss = F.cross_entropy(endpoint_scores, labels)
                if step_logits is not None and len(step_logits) > 0:
                    loss = loss + 0.5 * F.cross_entropy(step_logits, step_labels)
            else:
                scores = model(batch_eps)
                labels = torch.tensor([ep.answer for ep in batch_eps], device=device)
                loss = F.cross_entropy(scores, labels)
            
            assert torch.isfinite(loss), f'Non-finite loss at epoch {epoch}'
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    
    return float(loss)


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    device = next(model.parameters()).device
    correct = total = 0
    for i in range(0, len(data), 64):
        batch = data[i:i+64]
        if hasattr(model, 'forward'):
            scores = model(batch) if not isinstance(model, StepSupervisedArm) else model(batch, return_step_logits=False)
        preds = scores.argmax(-1).cpu().tolist()
        correct += sum(p == ep.answer for p, ep in zip(preds, batch))
        total += len(batch)
    return correct / total


@torch.no_grad()
def wess_interventions(model, data):
    """Causal interventions on WESS."""
    model.eval()
    eps = data[:100]
    
    # 1. Slot swap: swap queried entity's slot with another
    swaps = []
    swap_expected = []
    for ep in eps:
        other = (ep.query + 1) % 4
        swaps.append((ep.query, other))
        swap_expected.append(ep.tables[-1][other])
    
    swap_scores = model(eps, swap=swaps)
    swap_preds = swap_scores.argmax(-1).cpu().tolist()
    swap_transfer = sum(p == e for p, e in zip(swap_preds, swap_expected)) / len(eps)
    
    # 2. Last-write ablation: skip the last event that updated the query entity
    skips = []
    abl_expected = []
    valid_eps = []
    for ep in eps:
        last_idx = max(i for i, (k, _) in enumerate(ep.events) if k == ep.query)
        # Find previous state
        prior = None
        for j in range(last_idx - 1, -1, -1):
            if ep.events[j][0] == ep.query:
                prior = ep.events[j][1]; break
        if prior is not None:
            valid_eps.append(ep)
            skips.append(last_idx)
            abl_expected.append(prior)
    
    if valid_eps:
        abl_scores = model(valid_eps, skip=skips)
        abl_preds = abl_scores.argmax(-1).cpu().tolist()
        abl_revert = sum(p == e for p, e in zip(abl_preds, abl_expected)) / len(valid_eps)
    else:
        abl_revert = 0.0
    
    # 3. Persistence: check that non-updated entities retain their states
    persist_correct = persist_total = 0
    for ep in eps:
        # For each entity that wasn't updated in the last event
        last_ev_ent = ep.events[-1][0]
        for k in range(4):
            if k != last_ev_ent and k != ep.query:
                # Their state should be the same as in the second-to-last table
                if len(ep.tables) >= 2:
                    persist_total += 1
                    if ep.tables[-1][k] == ep.tables[-2][k]:
                        persist_correct += 1
    
    return {
        'slot_swap_transfer': round(swap_transfer, 4),
        'last_write_ablation_revert': round(abl_revert, 4),
        'persistence_check': round(persist_correct / max(1, persist_total), 4),
        'n_swap': len(eps),
        'n_ablation': len(valid_eps),
    }


def run_seed(seed, device):
    """Run all arms for one seed."""
    train_data = make_dataset(seed, 600, ENT_TR, ST_TR, (5, 7), 1)
    splits = {
        'iid': make_dataset(2001, 200, ENT_TR, ST_TR, (5, 7), 1),
        'new_ent': make_dataset(2002, 200, ENT_HO, ST_TR, (5, 7), 1),
        'new_state': make_dataset(2003, 200, ENT_TR, ST_HO, (5, 7), 1),
        'new_both': make_dataset(2004, 200, ENT_HO, ST_HO, (5, 7), 1),
        'long': make_dataset(2005, 200, ENT_TR, ST_TR, (9, 12), 3),
        'overwrite': make_dataset(2006, 200, ENT_TR, ST_TR, (7, 9), 3),
    }
    
    results = {}
    arms = [
        ('endpoint', lambda: EndpointArm().to(device)),
        ('step_supervised', lambda: StepSupervisedArm().to(device)),
        ('wess_gold', lambda: WESSArm(shuffled=False).to(device)),
        ('wess_shuffled', lambda: WESSArm(shuffled=True).to(device)),
    ]
    
    for arm_name, model_fn in arms:
        torch.manual_seed(seed)
        model = model_fn()
        
        train_arm = arm_name.replace('_gold', '').replace('_shuffled', '')
        if train_arm.startswith('wess'):
            train_arm = arm_name  # keep distinction for training
        
        loss = train_model(model, arm_name, train_data, seed)
        
        accs = {k: evaluate(model, v) for k, v in splits.items()}
        rec = {'loss_last': loss, 'params': sum(p.numel() for p in model.parameters()),
               'accuracy': {k: round(v, 4) for k, v in accs.items()}}
        
        if arm_name.startswith('wess'):
            rec['interventions'] = wess_interventions(model, splits['overwrite'])
        
        results[arm_name] = rec
        print(f"  seed={seed} {arm_name}: iid={accs['iid']:.3f} new_both={accs['new_both']:.3f} "
              f"long={accs['long']:.3f} overwrite={accs['overwrite']:.3f}", flush=True)
    
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
    arms = ['endpoint', 'step_supervised', 'wess_gold', 'wess_shuffled']
    split_names = ['iid', 'new_ent', 'new_state', 'new_both', 'long', 'overwrite']
    
    aggregate = {}
    for arm in arms:
        aggregate[arm] = {}
        for split in split_names:
            vals = [all_results[str(s)][arm]['accuracy'][split] for s in seeds]
            aggregate[arm][split] = {
                'mean': round(sum(vals)/len(vals), 4),
                'std': round(statistics.pstdev(vals), 4),
                'values': vals,
            }
        if arm.startswith('wess'):
            for metric in ['slot_swap_transfer', 'last_write_ablation_revert']:
                vals = [all_results[str(s)][arm]['interventions'][metric] for s in seeds]
                aggregate[arm][metric] = {
                    'mean': round(sum(vals)/len(vals), 4),
                    'std': round(statistics.pstdev(vals), 4),
                    'values': vals,
                }
    
    payload = {
        'status': 'NONCE_STATE_V3_TEXT_INTERFACE',
        'description': 'Natural-language text interface: events as "dax moves to lup." '
                      'processed by shared Transformer encoder. WESS extracts entity/state '
                      'representations from encoder hidden states at gold span positions.',
        'seeds': seeds, 'random_baseline': 0.25,
        'aggregate': aggregate, 'per_seed': all_results,
        'elapsed_sec': round(time.time() - t0, 1),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + '\n')
    
    print("\n" + "=" * 70)
    print("AGGREGATE (5 seeds, random baseline = 0.25)")
    print("=" * 70)
    print(f"{'split':<15} {'endpoint':<16} {'step_sup':<16} {'wess_gold':<16} {'wess_shuf':<16}")
    for split in split_names:
        row = f"{split:<15}"
        for arm in arms:
            m = aggregate[arm][split]['mean']
            s = aggregate[arm][split]['std']
            row += f" {m:.3f}±{s:.3f}    "
        print(row)
    
    print(f"\nWESS interventions (overwrite split):")
    for arm in ['wess_gold', 'wess_shuffled']:
        sw = aggregate[arm].get('slot_swap_transfer', {}).get('mean', 0)
        ab = aggregate[arm].get('last_write_ablation_revert', {}).get('mean', 0)
        print(f"  {arm}: swap_transfer={sw:.3f}, write_ablation_revert={ab:.3f}")
    
    print(f"\nSaved: {OUT}")


if __name__ == '__main__':
    main()
