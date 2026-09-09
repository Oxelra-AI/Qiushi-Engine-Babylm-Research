#!/usr/bin/env python3
"""research — Nonce-symbol entity-state update micro-world experiment.

Three-arm comparison to determine whether entity-state binding is learnable:
1. Endpoint-only: standard Transformer, only supervise final query answer
2. Step-state supervision: same Transformer, supervise full state table after each event
3. WESS: persistent entity slots with gated updates + bottlenecked query readout

Data: nonce-symbol episodes with 4 entities, 4 states, controlled overwrite,
interference, and anti-shortcut extrapolation axes.

Random baseline: 25% (K=4 states).
"""
from __future__ import annotations
import json, os, pathlib, random, time, copy, math
from dataclasses import dataclass, field
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
OUT_DIR = ROOT / 'data/nonce_state_tracking'
RESULTS_FILE = OUT_DIR / 'results.json'
NOTE_FILE = (ROOT.parents[2] / 'research/notes/initial_model_studies/179_nonce_state_tracking_results.md')

# ═══════════════════════════════════════════════════════════════
# PART 1: DATA GENERATOR
# ═══════════════════════════════════════════════════════════════

# Nonce vocabulary pools (no real-word overlap to prevent co-occurrence shortcuts)
NONCE_ENTITIES_TRAIN = [
    "dax", "wug", "blicket", "toma", "kiki", "bouba", "zup", "nib",
    "fep", "gax", "horp", "jiv", "lem", "quog", "rav", "sib",
]
NONCE_ENTITIES_HELDOUT = [
    "vex", "yim", "zol", "pav", "ket", "mur", "fid", "gon",
]
NONCE_STATES_TRAIN = [
    "lup", "fen", "mor", "tib", "cav", "dren", "gol", "hax",
    "jep", "klin", "nuf", "pev", "rax", "sov", "tul", "wem",
]
NONCE_STATES_HELDOUT = [
    "yib", "zaf", "brin", "cux", "dop", "elv", "fug", "hiv",
]

EVENT_TEMPLATES = [
    "{E} is at {S}.",
    "{E} moves to {S}.",
    "{E} goes to {S}.",
    "{E} is now at {S}.",
]
EVENT_TEMPLATES_HELDOUT = [
    "{E} travels to {S}.",
    "{E} arrives at {S}.",
]

QUERY_TEMPLATES = ["Where is {E}?"]


@dataclass
class Episode:
    """A single nonce-symbol state-tracking episode."""
    text: str                    # Full episode text (events + query)
    events: list[dict]           # [{entity, state, template_idx, event_idx}]
    query_entity: str            # Which entity is queried
    answer_state: str            # Correct final state of queried entity
    entities: list[str]          # All entities in this episode
    states: list[str]            # All states used
    state_table: list[dict]      # [{entity: state}] after each event
    n_events: int
    n_overwrites: int            # How many times the queried entity was overwritten
    n_distractors: int           # Entities other than the queried one
    has_interference: bool       # Distractor updates after last target update
    metadata: dict = field(default_factory=dict)


def generate_episode(
    n_entities: int = 4,
    n_states: int = 4,
    n_events: int = 4,
    min_overwrites: int = 1,
    entity_pool: list[str] = None,
    state_pool: list[str] = None,
    templates: list[str] = None,
    rng: random.Random = None,
) -> Episode:
    """Generate one nonce-symbol state-tracking episode."""
    if rng is None:
        rng = random.Random()
    if entity_pool is None:
        entity_pool = NONCE_ENTITIES_TRAIN
    if state_pool is None:
        state_pool = NONCE_STATES_TRAIN
    if templates is None:
        templates = EVENT_TEMPLATES
    
    # Sample entities and states
    entities = rng.sample(entity_pool, min(n_entities, len(entity_pool)))
    states = rng.sample(state_pool, min(n_states, len(state_pool)))
    
    # Generate events ensuring at least min_overwrites for the query entity
    events = []
    current_state = {}  # entity -> current state
    state_history = []  # list of full state tables after each event
    
    # First, assign initial states to all entities
    for i, ent in enumerate(entities):
        state = states[i % len(states)]
        current_state[ent] = state
        events.append({'entity': ent, 'state': state, 'template_idx': rng.randrange(len(templates))})
        state_history.append(dict(current_state))
    
    # Choose query entity
    query_entity = rng.choice(entities)
    
    # Generate additional events to reach n_events, ensuring overwrites
    overwrites_so_far = 0
    remaining_events = n_events - len(entities)
    
    for _ in range(max(0, remaining_events)):
        if overwrites_so_far < min_overwrites:
            # Force an overwrite of the query entity
            ent = query_entity
            overwrites_so_far += 1
        else:
            ent = rng.choice(entities)
            if ent == query_entity:
                overwrites_so_far += 1
        
        # Choose a NEW state (different from current)
        available = [s for s in states if s != current_state.get(ent)]
        if not available:
            available = states
        state = rng.choice(available)
        current_state[ent] = state
        events.append({'entity': ent, 'state': state, 'template_idx': rng.randrange(len(templates))})
        state_history.append(dict(current_state))
    
    # Optionally add interference (distractor updates after last target update)
    # Find last query entity event
    last_query_idx = max(i for i, e in enumerate(events) if e['entity'] == query_entity)
    has_interference = last_query_idx < len(events) - 1
    
    # Build text
    lines = []
    for ev in events:
        tmpl = templates[ev['template_idx'] % len(templates)]
        lines.append(tmpl.format(E=ev['entity'], S=ev['state']))
    
    query_tmpl = rng.choice(QUERY_TEMPLATES)
    query_line = query_tmpl.format(E=query_entity)
    lines.append(query_line)
    
    text = " ".join(lines)
    answer = current_state[query_entity]
    
    n_distractors = len(entities) - 1
    
    return Episode(
        text=text,
        events=[{**e, 'event_idx': i} for i, e in enumerate(events)],
        query_entity=query_entity,
        answer_state=answer,
        entities=entities,
        states=states,
        state_table=state_history,
        n_events=len(events),
        n_overwrites=overwrites_so_far,
        n_distractors=n_distractors,
        has_interference=has_interference,
    )


def generate_dataset(
    n_episodes: int,
    n_events_range: tuple[int, int],
    min_overwrites: int,
    entity_pool: list[str],
    state_pool: list[str],
    templates: list[str],
    seed: int,
) -> list[Episode]:
    """Generate a dataset of episodes."""
    rng = random.Random(seed)
    episodes = []
    for _ in range(n_episodes):
        n_events = rng.randint(*n_events_range)
        ep = generate_episode(
            n_entities=4, n_states=4, n_events=n_events,
            min_overwrites=min_overwrites,
            entity_pool=entity_pool, state_pool=state_pool,
            templates=templates, rng=rng,
        )
        episodes.append(ep)
    return episodes


def generate_all_splits(seed: int = 179):
    """Generate train and multiple held-out evaluation splits."""
    splits = {}
    
    # Training: seen entities, seen states, 4-6 events, 1+ overwrites
    splits['train'] = generate_dataset(
        n_episodes=500, n_events_range=(4, 6), min_overwrites=1,
        entity_pool=NONCE_ENTITIES_TRAIN, state_pool=NONCE_STATES_TRAIN,
        templates=EVENT_TEMPLATES, seed=seed)
    
    # Eval: same distribution as train
    splits['eval_iid'] = generate_dataset(
        n_episodes=200, n_events_range=(4, 6), min_overwrites=1,
        entity_pool=NONCE_ENTITIES_TRAIN, state_pool=NONCE_STATES_TRAIN,
        templates=EVENT_TEMPLATES, seed=seed + 1000)
    
    # Eval: unseen entities, seen states
    splits['eval_new_ent'] = generate_dataset(
        n_episodes=200, n_events_range=(4, 6), min_overwrites=1,
        entity_pool=NONCE_ENTITIES_HELDOUT, state_pool=NONCE_STATES_TRAIN,
        templates=EVENT_TEMPLATES, seed=seed + 2000)
    
    # Eval: seen entities, unseen states
    splits['eval_new_state'] = generate_dataset(
        n_episodes=200, n_events_range=(4, 6), min_overwrites=1,
        entity_pool=NONCE_ENTITIES_TRAIN, state_pool=NONCE_STATES_HELDOUT,
        templates=EVENT_TEMPLATES, seed=seed + 3000)
    
    # Eval: unseen entities AND unseen states
    splits['eval_new_both'] = generate_dataset(
        n_episodes=200, n_events_range=(4, 6), min_overwrites=1,
        entity_pool=NONCE_ENTITIES_HELDOUT, state_pool=NONCE_STATES_HELDOUT,
        templates=EVENT_TEMPLATES, seed=seed + 4000)
    
    # Eval: length extrapolation (7-10 events, more overwrites)
    splits['eval_long'] = generate_dataset(
        n_episodes=200, n_events_range=(7, 10), min_overwrites=2,
        entity_pool=NONCE_ENTITIES_TRAIN, state_pool=NONCE_STATES_TRAIN,
        templates=EVENT_TEMPLATES, seed=seed + 5000)
    
    # Eval: unseen template family
    splits['eval_new_template'] = generate_dataset(
        n_episodes=200, n_events_range=(4, 6), min_overwrites=1,
        entity_pool=NONCE_ENTITIES_TRAIN, state_pool=NONCE_STATES_TRAIN,
        templates=EVENT_TEMPLATES_HELDOUT, seed=seed + 6000)
    
    # Eval: heavy overwrite (same entity overwritten 3-4 times)
    splits['eval_heavy_overwrite'] = generate_dataset(
        n_episodes=200, n_events_range=(6, 8), min_overwrites=3,
        entity_pool=NONCE_ENTITIES_TRAIN, state_pool=NONCE_STATES_TRAIN,
        templates=EVENT_TEMPLATES, seed=seed + 7000)
    
    return splits


# ═══════════════════════════════════════════════════════════════
# PART 2: MODELS
# ═══════════════════════════════════════════════════════════════

class SimpleEncoder(nn.Module):
    """Small Transformer encoder for the micro-world experiment."""
    def __init__(self, vocab_size, hidden_size=128, n_layers=3, n_heads=4,
                 intermediate_size=256, max_pos=256, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.pos_embeddings = nn.Embedding(max_pos, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=n_heads,
            dim_feedforward=intermediate_size, dropout=dropout,
            batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
    
    def forward(self, input_ids, attention_mask=None):
        seq_len = input_ids.shape[1]
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.embeddings(input_ids) + self.pos_embeddings(positions)
        x = self.layer_norm(self.dropout(x))
        
        if attention_mask is not None:
            # Convert 0/1 mask to bool mask for nn.TransformerEncoder
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None
        
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return x


class EndpointModel(nn.Module):
    """Arm 1: Standard Transformer, predict masked query answer from full context."""
    def __init__(self, vocab_size, n_states, hidden_size=128, **kwargs):
        super().__init__()
        self.encoder = SimpleEncoder(vocab_size, hidden_size, **kwargs)
        self.state_head = nn.Linear(hidden_size, n_states)
    
    def forward(self, input_ids, attention_mask, query_mask_pos):
        """query_mask_pos: position of the [MASK] token for the answer."""
        h = self.encoder(input_ids, attention_mask)
        # Extract representation at query mask position
        mask_repr = h[torch.arange(h.size(0)), query_mask_pos]
        logits = self.state_head(mask_repr)
        return logits


class StepStateModel(nn.Module):
    """Arm 2: Same Transformer + per-event state supervision head."""
    def __init__(self, vocab_size, n_states, n_entities, hidden_size=128, **kwargs):
        super().__init__()
        self.encoder = SimpleEncoder(vocab_size, hidden_size, **kwargs)
        self.state_head = nn.Linear(hidden_size, n_states)  # endpoint
        self.step_head = nn.Linear(hidden_size, n_states)   # per-event per-entity
        self.n_entities = n_entities
        self.n_states = n_states
    
    def forward(self, input_ids, attention_mask, query_mask_pos,
                entity_positions=None):
        """
        entity_positions: [batch, n_entities, n_events] — token position of each
                          entity mention at each event (for step supervision)
        """
        h = self.encoder(input_ids, attention_mask)
        # Endpoint prediction
        mask_repr = h[torch.arange(h.size(0)), query_mask_pos]
        endpoint_logits = self.state_head(mask_repr)
        
        # Per-step per-entity state prediction
        step_logits = None
        if entity_positions is not None:
            B, K, T = entity_positions.shape
            # Gather entity representations at each event
            flat_pos = entity_positions.reshape(B, -1)  # [B, K*T]
            flat_pos = flat_pos.clamp(0, h.size(1) - 1)
            gathered = h.gather(1, flat_pos.unsqueeze(-1).expand(-1, -1, h.size(-1)))
            gathered = gathered.reshape(B, K, T, -1)
            step_logits = self.step_head(gathered)  # [B, K, T, n_states]
        
        return endpoint_logits, step_logits


class EntitySlotModule(nn.Module):
    """WESS: Writable Entity-State Slots with GRU update and gated persistence."""
    def __init__(self, hidden_size, n_heads=4):
        super().__init__()
        self.hidden_size = hidden_size
        # Write attention: slot queries event tokens
        self.write_attn = nn.MultiheadAttention(hidden_size, n_heads, batch_first=True)
        # GRU for slot update
        self.gru = nn.GRUCell(hidden_size, hidden_size)
        # Gate
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )
        # Read attention: query reads from slots
        self.read_attn = nn.MultiheadAttention(hidden_size, n_heads, batch_first=True)
    
    def update_slots(self, slots, event_hidden, event_mask=None):
        """
        slots: [B, K, H] — current entity slots
        event_hidden: [B, T_event, H] — encoder hidden states for this event
        Returns: updated slots [B, K, H], gates [B, K, 1]
        """
        B, K, H = slots.shape
        # Write attention: each slot attends to event tokens
        slots_flat = slots.reshape(B * K, 1, H)
        event_expanded = event_hidden.unsqueeze(1).expand(-1, K, -1, -1).reshape(B * K, -1, H)
        
        if event_mask is not None:
            mask_expanded = event_mask.unsqueeze(1).expand(-1, K, -1).reshape(B * K, -1)
            key_padding_mask = (mask_expanded == 0)
        else:
            key_padding_mask = None
        
        z, _ = self.write_attn(slots_flat, event_expanded, event_expanded,
                               key_padding_mask=key_padding_mask)
        z = z.reshape(B, K, H)
        
        # GRU update
        slots_in = slots.reshape(B * K, H)
        z_in = z.reshape(B * K, H)
        m_new = self.gru(z_in, slots_in).reshape(B, K, H)
        
        # Gate: decide whether to update
        gate_input = torch.cat([slots, z], dim=-1)
        g = self.gate(gate_input)  # [B, K, 1]
        
        # Gated update
        updated = (1 - g) * slots + g * m_new
        return updated, g
    
    def read_slots(self, query_repr, slots):
        """
        query_repr: [B, 1, H] — representation at query mask position
        slots: [B, K, H] — entity slots
        Returns: enhanced query repr [B, H]
        """
        out, attn_weights = self.read_attn(query_repr, slots, slots)
        return out.squeeze(1), attn_weights.squeeze(1)


class WESSModel(nn.Module):
    """Arm 3: WESS — entity slots + bottlenecked readout."""
    def __init__(self, vocab_size, n_states, n_entities, hidden_size=128, **kwargs):
        super().__init__()
        self.encoder = SimpleEncoder(vocab_size, hidden_size, **kwargs)
        self.slot_module = EntitySlotModule(hidden_size)
        self.state_head = nn.Linear(hidden_size, n_states)
        self.step_head = nn.Linear(hidden_size, n_states)  # per-slot prediction
        self.n_entities = n_entities
        self.hidden_size = hidden_size
    
    def forward(self, input_ids, attention_mask, query_mask_pos,
                entity_first_positions=None, event_boundaries=None,
                event_entity_mask=None):
        """
        entity_first_positions: [B, K] — position of first mention of each entity
        event_boundaries: [B, n_events, 2] — (start, end) token positions per event
        event_entity_mask: [B, K, n_events] — 1 if entity participates in event
        """
        h = self.encoder(input_ids, attention_mask)
        B = h.size(0)
        
        # Initialize slots from first-mention spans
        if entity_first_positions is not None:
            pos = entity_first_positions.clamp(0, h.size(1) - 1)
            slots = h.gather(1, pos.unsqueeze(-1).expand(-1, -1, self.hidden_size))
        else:
            slots = torch.zeros(B, self.n_entities, self.hidden_size, device=h.device)
        
        # Process events sequentially
        all_gates = []
        all_slot_states = [slots.clone()]
        step_logits_list = []
        
        if event_boundaries is not None:
            n_events = event_boundaries.size(1)
            for t in range(n_events):
                start = event_boundaries[:, t, 0]  # [B]
                end = event_boundaries[:, t, 1]      # [B]
                
                # Extract event tokens (variable length — use max and mask)
                max_len = int((end - start).max().item())
                if max_len <= 0:
                    max_len = 1
                
                event_h = torch.zeros(B, max_len, self.hidden_size, device=h.device)
                event_mask = torch.zeros(B, max_len, device=h.device)
                for b in range(B):
                    s, e = int(start[b]), int(end[b])
                    length = min(e - s, max_len)
                    if length > 0:
                        event_h[b, :length] = h[b, s:s+length]
                        event_mask[b, :length] = 1.0
                
                # Update slots
                updated_slots, gates = self.slot_module.update_slots(slots, event_h, event_mask)
                
                # Apply participation mask if available
                if event_entity_mask is not None:
                    participation = event_entity_mask[:, :, t].unsqueeze(-1)  # [B, K, 1]
                    slots = participation * updated_slots + (1 - participation) * slots
                else:
                    slots = updated_slots
                
                all_gates.append(gates)
                all_slot_states.append(slots.clone())
                
                # Per-slot state prediction at this step
                step_logits_list.append(self.step_head(slots))  # [B, K, n_states]
        
        # BOTTLENECKED READOUT: query can only access slots, not full text
        query_repr = h[torch.arange(B), query_mask_pos].unsqueeze(1)  # [B, 1, H]
        readout, read_attn = self.slot_module.read_slots(query_repr, slots)
        
        # Final prediction from slot readout only (bottleneck!)
        endpoint_logits = self.state_head(readout)
        
        # Stack step logits
        step_logits = torch.stack(step_logits_list, dim=2) if step_logits_list else None
        # step_logits: [B, K, T, n_states]
        
        return endpoint_logits, step_logits, all_gates, read_attn, all_slot_states


# ═══════════════════════════════════════════════════════════════
# PART 3: TOKENIZATION AND BATCH PREPARATION
# ═══════════════════════════════════════════════════════════════

class NonceLMTokenizer:
    """Simple word-level tokenizer for the nonce micro-world."""
    def __init__(self):
        # Build vocabulary from all possible tokens
        special = ['[PAD]', '[MASK]', '[UNK]', '.', '?']
        keywords = ['is', 'at', 'moves', 'to', 'goes', 'now', 'travels',
                   'arrives', 'Where']
        all_nonce = (NONCE_ENTITIES_TRAIN + NONCE_ENTITIES_HELDOUT +
                    NONCE_STATES_TRAIN + NONCE_STATES_HELDOUT)
        vocab_list = special + keywords + sorted(set(all_nonce))
        self.token2id = {t: i for i, t in enumerate(vocab_list)}
        self.id2token = {i: t for t, i in self.token2id.items()}
        self.pad_id = self.token2id['[PAD]']
        self.mask_id = self.token2id['[MASK]']
        self.vocab_size = len(self.token2id)
    
    def encode(self, text: str) -> list[int]:
        tokens = text.replace('.', ' .').replace('?', ' ?').split()
        return [self.token2id.get(t, self.token2id['[UNK]']) for t in tokens]
    
    def decode(self, ids: list[int]) -> str:
        return ' '.join(self.id2token.get(i, '[UNK]') for i in ids)


def prepare_batch(episodes: list[Episode], tokenizer: NonceLMTokenizer,
                  max_len: int = 128, state2idx: dict = None):
    """Prepare a batch with all metadata needed for three arms."""
    B = len(episodes)
    
    # Build state vocabulary mapping
    if state2idx is None:
        all_states = sorted(set(s for ep in episodes for s in ep.states))
        state2idx = {s: i for i, s in enumerate(all_states)}
    n_states = max(len(state2idx), 4)
    
    # Encode episodes
    input_ids = torch.full((B, max_len), tokenizer.pad_id, dtype=torch.long)
    attention_mask = torch.zeros(B, max_len, dtype=torch.long)
    query_mask_pos = torch.zeros(B, dtype=torch.long)
    labels = torch.zeros(B, dtype=torch.long)  # endpoint answer
    
    # Step-state supervision labels: [B, K, n_events]
    K = 4  # max entities
    max_events = max(ep.n_events for ep in episodes)
    step_labels = torch.full((B, K, max_events), -1, dtype=torch.long)
    entity_positions = torch.zeros(B, K, max_events, dtype=torch.long)
    
    # WESS metadata
    entity_first_pos = torch.zeros(B, K, dtype=torch.long)
    event_boundaries = torch.zeros(B, max_events, 2, dtype=torch.long)
    event_entity_mask = torch.zeros(B, K, max_events)
    
    for b, ep in enumerate(episodes):
        # Tokenize full text with answer replaced by [MASK]
        answer_token = ep.answer_state
        text_with_mask = ep.text.replace(
            f"? {answer_token}.", f"? [MASK] .")
        ids = tokenizer.encode(text_with_mask)[:max_len]
        input_ids[b, :len(ids)] = torch.tensor(ids)
        attention_mask[b, :len(ids)] = 1
        
        # Find mask position
        if tokenizer.mask_id in ids:
            query_mask_pos[b] = ids.index(tokenizer.mask_id)
        
        # Answer label
        labels[b] = state2idx.get(ep.answer_state, 0)
        
        # Entity-to-index mapping for this episode
        ent2idx = {e: i for i, e in enumerate(ep.entities[:K])}
        
        # Process events for metadata
        # Tokenize event by event to find boundaries
        text_tokens = ep.text.replace('.', ' .').replace('?', ' ?').split()
        token_pos = 0
        event_start = 0
        
        for t, ev in enumerate(ep.events):
            if t >= max_events:
                break
            # Find event text tokens
            tmpl = EVENT_TEMPLATES[ev['template_idx'] % len(EVENT_TEMPLATES)]
            ev_text = tmpl.format(E=ev['entity'], S=ev['state'])
            ev_tokens = ev_text.replace('.', ' .').split()
            ev_len = len(ev_tokens)
            
            event_boundaries[b, t, 0] = event_start
            event_boundaries[b, t, 1] = min(event_start + ev_len, max_len)
            
            # Entity position in this event
            eidx = ent2idx.get(ev['entity'], -1)
            if eidx >= 0 and eidx < K:
                # Find entity token position
                for i, tok in enumerate(ev_tokens):
                    if tok == ev['entity']:
                        entity_positions[b, eidx, t] = event_start + i
                        break
                event_entity_mask[b, eidx, t] = 1.0
                
                # Step-state label: state after this event
                if t < len(ep.state_table):
                    current = ep.state_table[t].get(ev['entity'])
                    if current in state2idx:
                        step_labels[b, eidx, t] = state2idx[current]
            
            event_start += ev_len
        
        # Entity first positions
        for ent, eidx in ent2idx.items():
            if eidx < K:
                # First event where this entity appears
                for t, ev in enumerate(ep.events):
                    if ev['entity'] == ent:
                        entity_first_pos[b, eidx] = event_boundaries[b, t, 0]
                        break
        
        # Fill step_labels for ALL entities at each event (state table)
        for t in range(min(len(ep.state_table), max_events)):
            for ent, state in ep.state_table[t].items():
                eidx = ent2idx.get(ent, -1)
                if 0 <= eidx < K and state in state2idx:
                    step_labels[b, eidx, t] = state2idx[state]
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'query_mask_pos': query_mask_pos,
        'labels': labels,
        'step_labels': step_labels,
        'entity_positions': entity_positions,
        'entity_first_pos': entity_first_pos,
        'event_boundaries': event_boundaries,
        'event_entity_mask': event_entity_mask,
        'state2idx': state2idx,
        'n_states': n_states,
    }


# ═══════════════════════════════════════════════════════════════
# PART 4: TRAINING AND EVALUATION
# ═══════════════════════════════════════════════════════════════

def train_one_epoch(model, arm_name, train_episodes, tokenizer, state2idx,
                    optimizer, device, batch_size=32):
    """Train one epoch for a given arm."""
    model.train()
    random.shuffle(train_episodes)
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    
    for i in range(0, len(train_episodes), batch_size):
        batch_eps = train_episodes[i:i+batch_size]
        batch = prepare_batch(batch_eps, tokenizer, state2idx=state2idx)
        
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        
        optimizer.zero_grad()
        
        if arm_name == 'endpoint':
            logits = model(batch['input_ids'], batch['attention_mask'],
                          batch['query_mask_pos'])
            loss = F.cross_entropy(logits, batch['labels'])
        
        elif arm_name == 'step_state':
            endpoint_logits, step_logits = model(
                batch['input_ids'], batch['attention_mask'],
                batch['query_mask_pos'], batch['entity_positions'])
            
            # Endpoint loss
            loss_endpoint = F.cross_entropy(endpoint_logits, batch['labels'])
            
            # Step-state loss (only where labels are valid)
            loss_step = torch.tensor(0.0, device=device)
            if step_logits is not None:
                valid = batch['step_labels'] >= 0
                if valid.any():
                    flat_logits = step_logits[valid]
                    flat_labels = batch['step_labels'][valid]
                    loss_step = F.cross_entropy(flat_logits, flat_labels)
            
            loss = loss_endpoint + 0.5 * loss_step
        
        elif arm_name == 'wess':
            endpoint_logits, step_logits, gates, read_attn, _ = model(
                batch['input_ids'], batch['attention_mask'],
                batch['query_mask_pos'], batch['entity_first_pos'],
                batch['event_boundaries'], batch['event_entity_mask'])
            
            # Endpoint loss
            loss_endpoint = F.cross_entropy(endpoint_logits, batch['labels'])
            
            # Step-state (slot) loss
            loss_step = torch.tensor(0.0, device=device)
            if step_logits is not None:
                valid = batch['step_labels'] >= 0
                if valid.any():
                    flat_logits = step_logits[valid]
                    flat_labels = batch['step_labels'][valid]
                    loss_step = F.cross_entropy(flat_logits, flat_labels)
            
            # Persistence loss (non-participating slots should not change)
            loss_persist = torch.tensor(0.0, device=device)
            # (simplified: already handled by event_entity_mask in forward)
            
            loss = loss_endpoint + 0.3 * loss_step + 0.1 * loss_persist
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * len(batch_eps)
        
        # Accuracy
        if arm_name == 'endpoint':
            preds = logits.argmax(dim=-1)
        else:
            preds = endpoint_logits.argmax(dim=-1)
        total_correct += (preds == batch['labels']).sum().item()
        total_count += len(batch_eps)
    
    return total_loss / max(1, total_count), total_correct / max(1, total_count)


@torch.no_grad()
def evaluate(model, arm_name, episodes, tokenizer, state2idx, device, batch_size=64):
    """Evaluate endpoint accuracy on a split."""
    model.eval()
    total_correct = 0
    total_count = 0
    
    for i in range(0, len(episodes), batch_size):
        batch_eps = episodes[i:i+batch_size]
        batch = prepare_batch(batch_eps, tokenizer, state2idx=state2idx)
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        
        if arm_name == 'endpoint':
            logits = model(batch['input_ids'], batch['attention_mask'],
                          batch['query_mask_pos'])
        elif arm_name == 'step_state':
            logits, _ = model(batch['input_ids'], batch['attention_mask'],
                             batch['query_mask_pos'], batch['entity_positions'])
        elif arm_name == 'wess':
            logits, _, _, _, _ = model(
                batch['input_ids'], batch['attention_mask'],
                batch['query_mask_pos'], batch['entity_first_pos'],
                batch['event_boundaries'], batch['event_entity_mask'])
        
        preds = logits.argmax(dim=-1)
        total_correct += (preds == batch['labels']).sum().item()
        total_count += len(batch_eps)
    
    return total_correct / max(1, total_count)


def run_experiment(seed: int, device):
    """Run the full three-arm experiment for one seed."""
    torch.manual_seed(seed)
    random.seed(seed)
    
    tokenizer = NonceLMTokenizer()
    splits = generate_all_splits(seed=seed)
    
    # Build state2idx from training data
    all_states = sorted(set(s for ep in splits['train'] for s in ep.states))
    state2idx = {s: i for i, s in enumerate(all_states)}
    n_states = len(state2idx)
    
    # Model configs
    model_kwargs = dict(n_layers=3, n_heads=4, intermediate_size=256, max_pos=128)
    hidden_size = 128
    K = 4
    
    results = {}
    
    for arm_name in ['endpoint', 'step_state', 'wess']:
        print(f"\n{'='*50}\nArm: {arm_name} | Seed: {seed}\n{'='*50}")
        
        torch.manual_seed(seed)  # Same init for all arms
        
        if arm_name == 'endpoint':
            model = EndpointModel(tokenizer.vocab_size, n_states, hidden_size, **model_kwargs)
        elif arm_name == 'step_state':
            model = StepStateModel(tokenizer.vocab_size, n_states, K, hidden_size, **model_kwargs)
        elif arm_name == 'wess':
            model = WESSModel(tokenizer.vocab_size, n_states, K, hidden_size, **model_kwargs)
        
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
        
        # Train for 30 epochs
        n_epochs = 30
        for epoch in range(n_epochs):
            loss, train_acc = train_one_epoch(
                model, arm_name, splits['train'], tokenizer, state2idx,
                optimizer, device, batch_size=32)
            if (epoch + 1) % 10 == 0:
                eval_acc = evaluate(model, arm_name, splits['eval_iid'],
                                   tokenizer, state2idx, device)
                print(f"  Epoch {epoch+1}: loss={loss:.4f}, train_acc={train_acc:.3f}, eval_iid={eval_acc:.3f}")
        
        # Final evaluation on all splits
        arm_results = {}
        for split_name, split_eps in splits.items():
            if split_name == 'train':
                continue
            acc = evaluate(model, arm_name, split_eps, tokenizer, state2idx, device)
            arm_results[split_name] = round(acc, 4)
            print(f"  {split_name}: {acc:.3f}")
        
        arm_results['train_acc'] = round(train_acc, 4)
        arm_results['n_params'] = sum(p.numel() for p in model.parameters())
        results[arm_name] = arm_results
    
    return results


def main():
    setup_env()
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Run with 5 seeds
    SEEDS = [42, 123, 456, 789, 2024]
    all_seed_results = {}
    
    for seed in SEEDS:
        print(f"\n{'#'*60}\nSEED {seed}\n{'#'*60}")
        results = run_experiment(seed, device)
        all_seed_results[seed] = results
    
    # Aggregate across seeds
    arms = ['endpoint', 'step_state', 'wess']
    splits_eval = ['eval_iid', 'eval_new_ent', 'eval_new_state', 'eval_new_both',
                   'eval_long', 'eval_new_template', 'eval_heavy_overwrite']
    
    aggregate = {}
    for arm in arms:
        arm_agg = {}
        for split in splits_eval:
            accs = [all_seed_results[s][arm].get(split, 0) for s in SEEDS]
            arm_agg[split] = {
                'mean': round(sum(accs) / len(accs), 4),
                'std': round((sum((a - sum(accs)/len(accs))**2 for a in accs) / len(accs))**0.5, 4),
                'per_seed': accs,
            }
        aggregate[arm] = arm_agg
    
    # Save results
    payload = {
        'status': 'NONCE_STATE_TRACKING',
        'description': 'Three-arm nonce-symbol entity-state tracking experiment',
        'arms': arms,
        'seeds': SEEDS,
        'n_train': 500,
        'n_eval_per_split': 200,
        'model': {'hidden': 128, 'layers': 3, 'heads': 4},
        'training': {'epochs': 30, 'batch_size': 32, 'lr': 3e-4},
        'random_baseline': 0.25,
        'aggregate': aggregate,
        'per_seed': all_seed_results,
        'elapsed_sec': round(time.time() - t0, 1),
    }
    RESULTS_FILE.write_text(json.dumps(payload, indent=2) + '\n')
    
    # Print summary
    print("\n" + "="*70)
    print("AGGREGATE RESULTS (mean ± std across 5 seeds, random baseline = 0.25)")
    print("="*70)
    print(f"{'split':<25} {'endpoint':<18} {'step_state':<18} {'wess':<18}")
    for split in splits_eval:
        row = f"{split:<25}"
        for arm in arms:
            m = aggregate[arm][split]['mean']
            s = aggregate[arm][split]['std']
            row += f" {m:.3f}±{s:.3f}      "
        print(row)
    
    print(f"\nParam counts: endpoint={all_seed_results[SEEDS[0]]['endpoint']['n_params']}, "
          f"step_state={all_seed_results[SEEDS[0]]['step_state']['n_params']}, "
          f"wess={all_seed_results[SEEDS[0]]['wess']['n_params']}")
    
    print(f"\nResults saved to {RESULTS_FILE}")


def setup_env():
    hf = ROOT / 'training/hf_home'
    os.environ['HF_HOME'] = str(hf.resolve())
    os.environ['HF_HUB_CACHE'] = str((hf / 'hub').resolve())
    os.environ['TRANSFORMERS_CACHE'] = str((hf / 'transformers').resolve())
    os.environ['HF_MODULES_CACHE'] = str((ROOT / 'training/hf_modules_cache').resolve())
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


if __name__ == '__main__':
    main()
