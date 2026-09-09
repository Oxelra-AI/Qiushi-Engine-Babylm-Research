#!/usr/bin/env python3
"""research: Paired contrastive scoring and wording-cross evaluation.

Scientific design:
- Training is structurally identical to research (same data, arms, procedure)
- Evaluation adds three key changes:
  1. Paired AB-vs-BA contrastive scoring: for each (world, query, template),
     compare logit(context, hyp_AB) vs logit(context, hyp_BA) instead of
     independent binary classification.  This eliminates per-template
     calibration bias.
  2. Held hypothesis wordings: train hyp = "X won the match against Y" /
     "X held the higher ranking than Y"; held hyp = "X claimed the victory
     over Y" / "X outranked Y".  Crossed with held/train context wordings
     gives a 2×2 surface.
  3. Independent semantic grounding: before fine-tuning, measure pretrained
     cosine similarity between train-wording and held-wording representations
     of the same facts.  High similarity means wordings are equivalent in the
     pretrained space; any fine-tuned failure is calibration, not representation.

If paired contrastive scoring recovers held-wording accuracy, the role
coordinate generalizes but binary head calibration is template-specific.
If both fail, the representation does not generalize.

Note on scientific object: contemporaneous ATP state is independently attested
and decorrelated from the event outcome.  Success here establishes role-
coordinate REUSE under interference, NOT event-derived state update.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, math, os, random, re, time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, DebertaV2Config, DebertaV2Model

STUDY = Path("experiments/archive/representation_and_objectives")
WS = STUDY
ATP_DIR = WS / "data/upset_balanced_state_substrate"
PAIR_DIR = WS / "data/paired_world_pilot"
DEFAULT_MODEL = Path("experiments/archive/frontier_consolidation/training/runs"
                     "cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/hf_model/chck_100M")
DEFAULT_OUT = WS / "data/contrastive_wording_cross"

ALIAS_POOL = [
    "Alice","Ben","Clara","David","Emma","Frank","Grace","Henry",
    "Iris","Jack","Kate","Liam","Mia","Noah","Olivia","Paul",
    "Quinn","Rose","Sam","Tina","Uma","Victor","Wendy","Xavier",
    "Yara","Zoe","Aaron","Bella","Caleb","Diana","Ethan","Fiona",
    "Gavin","Hannah","Isaac","Julia","Kevin","Laura","Mason","Nora",
    "Owen","Paula","Riley","Sarah","Thomas","Vera","Will","Nina",
]

# --- Context wording families ---
# Train: used in base anchor + sparse evidence + some eval
# Held: used ONLY in evaluation
EVENT_CTX = {
    "anchor": ["{W} defeated {L} in the match.",
               "{L} lost to {W} in the match."],
    "train":  ["{W} prevailed over {L} in the match.",
               "{L} was beaten by {W} in the match."],
    "held":   ["{W} edged out {L} in the match.",
               "{L} went down to {W} in the match."],
}
STATE_CTX = {
    "anchor": ["The weekly ranking record put {H} above {Lo}.",
               "That week, {Lo} was ranked below {H}."],
    "train":  ["According to the rankings, {H} stood higher than {Lo}.",
               "The ATP list placed {Lo} under {H}."],
    "held":   ["{H} held the better ranking position than {Lo}.",
               "{Lo} trailed {H} in the weekly rankings."],
}

# --- Hypothesis wording families ---
HYP_EVENT = {
    "train": "{X} won the match against {Y}.",
    "held":  "{X} claimed the victory over {Y}.",
}
HYP_STATE = {
    "train": "{X} held the higher ranking than {Y}.",
    "held":  "{X} outranked {Y}.",
}

# ===================================================================
# Data loading
# ===================================================================
@dataclass(frozen=True)
class World:
    source: str; family_id: str; world_id: str; split: str; sport: str
    participant_a: str; participant_b: str
    winner_name: str; loser_name: str; winner_label: str
    higher_name: str | None = None; lower_name: str | None = None
    winner_higher_at_time: bool | None = None

def _rj(p: Path):
    return [json.loads(x) for x in p.read_text("utf-8").splitlines() if x.strip()]

def load_atp() -> tuple[list[World], list[World]]:
    out: dict[str, list[World]] = {"train": [], "held": []}
    for sp in ["train", "held"]:
        for f in _rj(ATP_DIR / f"families_{sp}.jsonl"):
            for wk in ["context1", "context2"]:
                c = f[wk]; w = c["winner"]; l = c["loser"]
                a, b = f["participant_a"], f["participant_b"]
                wl = "A" if w == a else "B"
                h, lo = (w, l) if c["winner_higher_at_time"] else (l, w)
                out[sp].append(World("atp", f["family_id"], f"{f['family_id']}::{wk}",
                    sp, "tennis_atp", a, b, w, l, wl, h, lo, bool(c["winner_higher_at_time"])))
    return out["train"], out["held"]

def load_pairs() -> tuple[list[World], list[World]]:
    out: dict[str, list[World]] = {"train": [], "held": []}
    for sp in ["train", "held"]:
        for f in _rj(PAIR_DIR / f"families_{sp}.jsonl"):
            for wk in ["context1", "context2"]:
                c = f[wk]; w = c["winner_name"]; l = c["loser_name"]
                a, b = f["participant_a"], f["participant_b"]
                wl = "A" if w == a else "B"
                out[sp].append(World("pair", f["family_id"], f"{f['family_id']}::{wk}",
                    sp, str(f.get("sport") or f.get("source_type","sport")),
                    a, b, w, l, wl))
    return out["train"], out["held"]

def select(ws: list[World], n: int, seed: int, state: bool = False) -> list[World]:
    pool = [w for w in ws if (not state or w.higher_name)] if state else ws[:]
    rng = random.Random(seed); rng.shuffle(pool)
    return pool[:min(n, len(pool))]

# ===================================================================
# Alias management (same as research)
# ===================================================================
def _si(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:16], 16)

def aliases(key: str, n: int) -> list[str]:
    return random.Random(_si("aliases|" + key)).sample(ALIAS_POOL, n)

def other_aliases(used: set[str], key: str, n: int = 2) -> list[str]:
    cands = [x for x in ALIAS_POOL if x not in used]
    return random.Random(_si("other|" + key)).sample(cands, n)

def amap2(w: World, ns: str) -> dict[str, str]:
    a, b = aliases(f"two|{ns}|{w.world_id}", 2)
    return {w.participant_a: a, w.participant_b: b}

def amap4(p: World, s: World, ns: str) -> dict[str, str]:
    al = aliases(f"four|{ns}|{p.world_id}|{s.world_id}", 4)
    return {p.participant_a: al[0], p.participant_b: al[1],
            s.participant_a: al[2], s.participant_b: al[3]}

def an(am: dict[str, str], name: str) -> str:
    return am[name]

# ===================================================================
# Sentence builders
# ===================================================================
def ev_sent(w: World, am: dict, grp: str, var: int) -> str:
    return EVENT_CTX[grp][var % len(EVENT_CTX[grp])].format(
        W=an(am, w.winner_name), L=an(am, w.loser_name))

def st_sent(w: World, am: dict, grp: str, var: int) -> str:
    assert w.higher_name and w.lower_name
    return STATE_CTX[grp][var % len(STATE_CTX[grp])].format(
        H=an(am, w.higher_name), Lo=an(am, w.lower_name))

def ev_label(w: World, hd: str) -> int:
    return int((hd == "AB" and w.winner_label == "A") or
               (hd == "BA" and w.winner_label == "B"))

def st_label(w: World, hd: str) -> int:
    hl = "A" if w.higher_name == w.participant_a else "B"
    return int((hd == "AB" and hl == "A") or (hd == "BA" and hl == "B"))

def hyp_ev(am: dict, w: World, hd: str, wording: str = "train") -> str:
    x = an(am, w.participant_a if hd == "AB" else w.participant_b)
    y = an(am, w.participant_b if hd == "AB" else w.participant_a)
    return HYP_EVENT[wording].format(X=x, Y=y)

def hyp_st(am: dict, w: World, hd: str, wording: str = "train") -> str:
    x = an(am, w.participant_a if hd == "AB" else w.participant_b)
    y = an(am, w.participant_b if hd == "AB" else w.participant_a)
    return HYP_STATE[wording].format(X=x, Y=y)

# ===================================================================
# Row construction
# ===================================================================
def mk(*, row_id, context, hypothesis, label, split, eval_set, arm, source,
        sport, query_family, primary_world, secondary_world=None,
        template_group="", label_mode="true", conflict=None, train_kind="",
        hyp_dir="", pair_key="", hyp_wording="train", ctx_wording="train"):
    return dict(id=row_id, text=context+" [SEP] "+hypothesis,
        context=context, hypothesis=hypothesis, label=int(label),
        split=split, eval_set=eval_set, arm=arm, source=source, sport=sport,
        query_family=query_family, primary_world=primary_world,
        secondary_world=secondary_world, template_group=template_group,
        label_mode=label_mode, conflict=bool(conflict) if conflict is not None else None,
        train_kind=train_kind, hyp_dir=hyp_dir, pair_key=pair_key,
        hyp_wording=hyp_wording, ctx_wording=ctx_wording)

def compound_rows(prims, secs, *, ev_grp, st_grp, ns, split, ev_set, arm,
                  lm: Literal["true","flip"]="true", tk="compound",
                  hyp_w="train"):
    rows = []
    for i, p in enumerate(prims):
        s = secs[(i*7+3) % len(secs)]
        if s.world_id == p.world_id: s = secs[(i*7+4) % len(secs)]
        am = amap4(p, s, ns)
        for ev in range(len(EVENT_CTX[ev_grp])):
            for sv in range(len(STATE_CTX[st_grp])):
                ctx = (f"Match record: {ev_sent(p, am, ev_grp, ev)} "
                       f"Ranking record: {st_sent(p, am, st_grp, sv)} "
                       f"Separate ranking record: {st_sent(s, am, st_grp, (sv+1))}")
                tpl = f"e={ev_grp}:{ev}|s={st_grp}:{sv}"
                for hd in ["AB","BA"]:
                    base_pk = f"{ns}|{p.world_id}|{s.world_id}|{tpl}"
                    lab = ev_label(p, hd)
                    if lm == "flip": lab = 1 - lab
                    rows.append(mk(row_id=f"{base_pk}|event|{hd}|h={hyp_w}",
                        context=ctx, hypothesis=hyp_ev(am, p, hd, hyp_w),
                        label=lab, split=split, eval_set=ev_set, arm=arm,
                        source=p.source, sport=p.sport, query_family="event_role",
                        primary_world=p.world_id, secondary_world=s.world_id,
                        template_group=tpl, label_mode=lm,
                        conflict=not bool(p.winner_higher_at_time),
                        train_kind=tk, hyp_dir=hd, pair_key=f"{base_pk}|event|h={hyp_w}",
                        hyp_wording=hyp_w, ctx_wording=ev_grp))
                    lab = st_label(p, hd)
                    if lm == "flip": lab = 1 - lab
                    rows.append(mk(row_id=f"{base_pk}|focal|{hd}|h={hyp_w}",
                        context=ctx, hypothesis=hyp_st(am, p, hd, hyp_w),
                        label=lab, split=split, eval_set=ev_set, arm=arm,
                        source=p.source, sport=p.sport, query_family="focal_state",
                        primary_world=p.world_id, secondary_world=s.world_id,
                        template_group=tpl, label_mode=lm,
                        conflict=not bool(p.winner_higher_at_time),
                        train_kind=tk, hyp_dir=hd, pair_key=f"{base_pk}|focal|h={hyp_w}",
                        hyp_wording=hyp_w, ctx_wording=ev_grp))
                    lab = st_label(s, hd)
                    if lm == "flip": lab = 1 - lab
                    rows.append(mk(row_id=f"{base_pk}|untouched|{hd}|h={hyp_w}",
                        context=ctx, hypothesis=hyp_st(am, s, hd, hyp_w),
                        label=lab, split=split, eval_set=ev_set, arm=arm,
                        source=s.source, sport=s.sport, query_family="untouched_state",
                        primary_world=p.world_id, secondary_world=s.world_id,
                        template_group=tpl, label_mode=lm,
                        conflict=not bool(p.winner_higher_at_time),
                        train_kind=tk, hyp_dir=hd, pair_key=f"{base_pk}|untouched|h={hyp_w}",
                        hyp_wording=hyp_w, ctx_wording=ev_grp))
    return rows

def event_only_rows(worlds, *, ev_grp, ns, split, ev_set, arm,
                    lm: Literal["true","flip"]="true", tk="event_only",
                    hyp_w="train"):
    rows = []
    for w in worlds:
        am = amap2(w, ns)
        for ev in range(len(EVENT_CTX[ev_grp])):
            ctx = f"Match record: {ev_sent(w, am, ev_grp, ev)}"
            tpl = f"e={ev_grp}:{ev}"
            for hd in ["AB","BA"]:
                base_pk = f"{ns}|{w.world_id}|{tpl}"
                lab = ev_label(w, hd)
                if lm == "flip": lab = 1 - lab
                rows.append(mk(row_id=f"{base_pk}|event|{hd}|h={hyp_w}",
                    context=ctx, hypothesis=hyp_ev(am, w, hd, hyp_w),
                    label=lab, split=split, eval_set=ev_set, arm=arm,
                    source=w.source, sport=w.sport, query_family="event_role",
                    primary_world=w.world_id, template_group=tpl, label_mode=lm,
                    train_kind=tk, hyp_dir=hd, pair_key=f"{base_pk}|event|h={hyp_w}",
                    hyp_wording=hyp_w, ctx_wording=ev_grp))
    return rows

def exposure_rows(prims, secs, *, ev_grp, st_grp, ns, split, ev_set, arm):
    rows = []
    for i, p in enumerate(prims):
        s = secs[(i*7+3) % len(secs)]
        if s.world_id == p.world_id: s = secs[(i*7+4) % len(secs)]
        am = amap4(p, s, ns)
        used = set(am.values())
        x, y, z = other_aliases(used, f"{ns}|{p.world_id}|{s.world_id}", 3)
        vals = {"A": am[p.participant_a], "B": am[p.participant_b],
                "C": am[s.participant_a], "D": am[s.participant_b], "X": x, "Y": y, "Z": z}
        for ev in range(len(EVENT_CTX[ev_grp])):
            for sv in range(len(STATE_CTX[st_grp])):
                ctx = (f"Match record: {ev_sent(p, am, ev_grp, ev)} "
                       f"Ranking record: {st_sent(p, am, st_grp, sv)} "
                       f"Separate ranking record: {st_sent(s, am, st_grp, (sv+1))}")
                tpl = f"e={ev_grp}:{ev}|s={st_grp}:{sv}"
                mention_pairs = [
                    (f"The passage mentions {vals['A']}.", 1),
                    (f"The passage mentions {vals['B']}.", 1),
                    (f"The passage mentions {vals['C']}.", 1),
                    (f"The passage mentions {vals['X']}.", 0),
                    (f"The passage mentions {vals['Y']}.", 0),
                    (f"The passage mentions {vals['Z']}.", 0),
                ]
                for j, (hyp, lab) in enumerate(mention_pairs):
                    rows.append(mk(row_id=f"{ns}|{p.world_id}|{s.world_id}|{tpl}|mention|{j}",
                        context=ctx, hypothesis=hyp, label=lab,
                        split=split, eval_set=ev_set, arm=arm, source=p.source,
                        sport=p.sport, query_family="mention_exposure",
                        primary_world=p.world_id, secondary_world=s.world_id,
                        template_group=tpl, label_mode="mention",
                        train_kind="sparse_exposure"))
    return rows

def paired_exposure_event(worlds, *, ev_grp, ns, split, ev_set, arm):
    rows = []
    for w in worlds:
        am = amap2(w, ns)
        used = set(am.values())
        x, y = other_aliases(used, f"paired_exp|{ns}|{w.world_id}", 2)
        for ev in range(len(EVENT_CTX[ev_grp])):
            ctx = f"Match record: {ev_sent(w, am, ev_grp, ev)}"
            tpl = f"e={ev_grp}:{ev}"
            for j, (hyp, lab) in enumerate([(f"The passage mentions {am[w.participant_a]}.", 1),
                                             (f"The passage mentions {x}.", 0)]):
                rows.append(mk(row_id=f"{ns}|{w.world_id}|{tpl}|mention|{j}",
                    context=ctx, hypothesis=hyp, label=lab,
                    split=split, eval_set=ev_set, arm=arm, source=w.source,
                    sport=w.sport, query_family="mention_exposure",
                    primary_world=w.world_id, template_group=tpl,
                    label_mode="mention", train_kind="sparse_exposure"))
    return rows

# ===================================================================
# Semantic grounding check
# ===================================================================
def semantic_grounding(model_path: Path, tokenizer, worlds: list[World],
                       device: str, max_len: int = 128) -> dict[str, Any]:
    """Pretrained-model cosine similarity between train vs held wordings."""
    enc = DebertaV2Model.from_pretrained(str(model_path)).to(device).eval()
    def pool(text):
        tok = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
        with torch.no_grad():
            h = enc(tok["input_ids"].to(device), tok["attention_mask"].to(device)).last_hidden_state
            m = tok["attention_mask"].to(device).unsqueeze(-1).float()
            return (h * m).sum(1) / m.sum(1).clamp_min(1)

    cos_ev, cos_st, cos_hyp_ev, cos_hyp_st = [], [], [], []
    for w in worlds[:50]:  # bounded sample
        am = amap2(w, "grounding")
        # Event context
        for tv in range(len(EVENT_CTX["train"])):
            for hv in range(len(EVENT_CTX["held"])):
                r1 = pool(ev_sent(w, am, "train", tv))
                r2 = pool(ev_sent(w, am, "held", hv))
                cos_ev.append(float(F.cosine_similarity(r1, r2).item()))
        # State context (ATP only)
        if w.higher_name:
            for tv in range(len(STATE_CTX["train"])):
                for hv in range(len(STATE_CTX["held"])):
                    r1 = pool(st_sent(w, am, "train", tv))
                    r2 = pool(st_sent(w, am, "held", hv))
                    cos_st.append(float(F.cosine_similarity(r1, r2).item()))
        # Event hypothesis
        for hd in ["AB"]:
            r1 = pool(hyp_ev(am, w, hd, "train"))
            r2 = pool(hyp_ev(am, w, hd, "held"))
            cos_hyp_ev.append(float(F.cosine_similarity(r1, r2).item()))
        # State hypothesis (ATP only)
        if w.higher_name:
            for hd in ["AB"]:
                r1 = pool(hyp_st(am, w, hd, "train"))
                r2 = pool(hyp_st(am, w, hd, "held"))
                cos_hyp_st.append(float(F.cosine_similarity(r1, r2).item()))
    del enc; torch.cuda.empty_cache()
    def stats(v):
        if not v: return {}
        return {"mean": float(np.mean(v)), "std": float(np.std(v)),
                "min": float(np.min(v)), "max": float(np.max(v)), "n": len(v)}
    return {"event_ctx": stats(cos_ev), "state_ctx": stats(cos_st),
            "event_hyp": stats(cos_hyp_ev), "state_hyp": stats(cos_hyp_st)}

# ===================================================================
# Model
# ===================================================================
class DebertaEntailment(nn.Module):
    def __init__(self, model_path: str | Path, regime: str, init: str):
        super().__init__()
        if init == "pretrained":
            self.encoder = DebertaV2Model.from_pretrained(str(model_path))
        else:
            self.encoder = DebertaV2Model(DebertaV2Config.from_pretrained(str(model_path)))
        h = int(self.encoder.config.hidden_size)
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(h, h//2),
                                  nn.GELU(), nn.Dropout(0.1), nn.Linear(h//2, 2))
        if regime == "full":
            for p in self.encoder.parameters(): p.requires_grad_(True)
        else:
            raise ValueError(regime)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(out.dtype)
        return self.head((out * m).sum(1) / m.sum(1).clamp_min(1))

def encode_rows(tok, rows, ml, dev):
    texts = [r["text"] for r in rows]
    enc = tok(texts, padding=True, truncation=True, max_length=ml, return_tensors="pt")
    labels = torch.tensor([int(r["label"]) for r in rows], dtype=torch.long)
    return enc["input_ids"].to(dev), enc["attention_mask"].to(dev), labels.to(dev)

# ===================================================================
# Evaluation: standard + paired contrastive
# ===================================================================
def get_all_logits(model, tok, rows, ml, dev, bs):
    model.eval()
    logits_list = []
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            batch = rows[i:i+bs]
            x, m, y = encode_rows(tok, batch, ml, dev)
            logits_list.append(model(x, m).detach().cpu())
    return torch.cat(logits_list, 0) if logits_list else torch.zeros(0, 2)

def standard_eval(logits, rows):
    if not rows: return {}
    pred = logits.argmax(1).numpy()
    lab = np.array([r["label"] for r in rows])
    out = {"n": len(rows), "acc": float(np.mean(pred == lab))}
    for key in ["query_family", "conflict"]:
        groups = collections.defaultdict(list)
        for i, r in enumerate(rows): groups[str(r.get(key, ""))].append(i)
        out[f"by_{key}"] = {g: float(np.mean(pred[idx] == lab[idx]))
                            for g, idx in groups.items() if idx}
    out["key_subsets"] = {}
    for qf in ["event_role", "focal_state", "untouched_state"]:
        idx = [i for i, r in enumerate(rows) if r.get("query_family") == qf]
        if idx: out["key_subsets"][qf] = float(np.mean(pred[idx] == lab[idx]))
    for qf in ["focal_state", "untouched_state"]:
        idx = [i for i, r in enumerate(rows) if r.get("query_family") == qf
               and r.get("conflict") is True]
        if idx: out["key_subsets"][f"{qf}_conflict"] = float(np.mean(pred[idx] == lab[idx]))
    return out

def contrastive_eval(logits, rows):
    """Paired AB-vs-BA contrastive scoring."""
    if not rows: return {}
    # Group into pairs
    pairs: dict[str, dict[str, tuple[int, dict]]] = collections.defaultdict(dict)
    for i, r in enumerate(rows):
        pk = r.get("pair_key", "")
        hd = r.get("hyp_dir", "")
        if pk and hd: pairs[pk][hd] = (i, r)

    correct_total = [0, 0]
    by_query: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    by_conflict: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])

    for pk, dirs in pairs.items():
        if "AB" not in dirs or "BA" not in dirs: continue
        i_ab, r_ab = dirs["AB"]
        i_ba, r_ba = dirs["BA"]
        # Belief: logit[1] - logit[0]
        belief_ab = float(logits[i_ab, 1] - logits[i_ab, 0])
        belief_ba = float(logits[i_ba, 1] - logits[i_ba, 0])
        pred_ab = belief_ab > belief_ba
        true_ab = r_ab["label"] == 1
        ok = int(pred_ab == true_ab)
        correct_total[0] += ok; correct_total[1] += 1
        qf = r_ab.get("query_family", "")
        by_query[qf][0] += ok; by_query[qf][1] += 1
        cf = r_ab.get("conflict")
        if cf is not None:
            ck = f"{qf}_{'conflict' if cf else 'nonconflict'}"
            by_conflict[ck][0] += ok; by_conflict[ck][1] += 1

    n = correct_total[1]
    return {
        "contrastive_acc": correct_total[0] / n if n else float("nan"),
        "n_pairs": n,
        "by_query": {k: v[0]/v[1] if v[1] else float("nan") for k, v in by_query.items()},
        "by_conflict": {k: v[0]/v[1] if v[1] else float("nan") for k, v in by_conflict.items()},
    }

def full_eval(model, tok, eval_sets, ml, dev, bs):
    results = {}
    for name, rows in eval_sets.items():
        logits = get_all_logits(model, tok, rows, ml, dev, bs)
        results[name] = {
            "standard": standard_eval(logits, rows),
            "contrastive": contrastive_eval(logits, rows),
        }
    return results

# ===================================================================
# Training
# ===================================================================
def train_one(*, model_path, tok, arm, seed, train_rows, evals, args, dev):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = DebertaEntailment(model_path, "full", "pretrained").to(dev)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    head_p = [p for n, p in model.named_parameters() if n.startswith("head") and p.requires_grad]
    enc_p = [p for n, p in model.named_parameters() if not n.startswith("head") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": head_p, "lr": args.head_lr},
                              {"params": enc_p, "lr": args.encoder_lr}],
                             weight_decay=args.weight_decay)
    x, m, y = encode_rows(tok, train_rows, args.max_len, dev)
    ds = TensorDataset(x, m, y)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    best_state, best_train = None, -1.0
    history = []
    for ep in range(1, args.epochs + 1):
        model.train(); losses = []
        for xb, mb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb, mb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            preds = []
            for i in range(0, len(train_rows), args.eval_batch_size):
                preds.append(model(x[i:i+args.eval_batch_size], m[i:i+args.eval_batch_size]).detach())
            ta = float((torch.cat(preds).argmax(1) == y).float().mean())
        history.append({"epoch": ep, "loss": float(np.mean(losses)), "train_acc": ta})
        if ta > best_train:
            best_train = ta
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state: model.load_state_dict(best_state)
    eval_out = full_eval(model, tok, evals, args.max_len, dev, args.eval_batch_size)
    return {"arm": arm, "seed": seed, "epochs": args.epochs,
            "trainable_params": int(trainable), "history": history,
            "best_train_acc": float(best_train), "evals": eval_out}

# ===================================================================
# Dataset building
# ===================================================================
def build_datasets(args):
    atp_tr, atp_he = load_atp()
    pair_tr, pair_he = load_pairs()
    atp_train = select(atp_tr, args.train_atp, 26201, True)
    atp_train2 = select(atp_tr, args.train_atp + 17, 26202, True)
    atp_sparse = select(atp_tr, args.sparse_atp, 26203, True)
    atp_sparse2 = select(atp_tr, args.sparse_atp + 17, 26204, True)
    atp_eval = select(atp_he, args.eval_atp, 26205, True)
    atp_eval2 = select(atp_he, args.eval_atp + 17, 26206, True)
    pair_train = select(pair_tr, args.train_pair, 26207)
    pair_sparse = select(pair_tr, args.sparse_pair, 26208)
    pair_eval = select(pair_he, args.eval_pair, 26209)

    # Base anchor: train wordings, same for all arms
    base = []
    base += compound_rows(atp_train, atp_train2, ev_grp="anchor", st_grp="anchor",
        ns="base_anchor", split="train", ev_set="base", arm="base", tk="base_compound")
    base += event_only_rows(pair_train, ev_grp="anchor", ns="base_pair",
        split="train", ev_set="base", arm="base", tk="base_pair_event")

    # Evaluation sets: crossed 2×2 wording design + anchor baseline
    def make_evals(arm_name):
        ev = {}
        # 1. Train context + train hypothesis (baseline, like research "probe")
        ev["atp_trainCtx_trainHyp"] = compound_rows(atp_eval, atp_eval2,
            ev_grp="train", st_grp="train", ns="eval_tt", split="held",
            ev_set="atp_trainCtx_trainHyp", arm=arm_name, hyp_w="train")
        # 2. Held context + train hypothesis
        ev["atp_heldCtx_trainHyp"] = compound_rows(atp_eval, atp_eval2,
            ev_grp="held", st_grp="held", ns="eval_ht", split="held",
            ev_set="atp_heldCtx_trainHyp", arm=arm_name, hyp_w="train")
        # 3. Train context + held hypothesis
        ev["atp_trainCtx_heldHyp"] = compound_rows(atp_eval, atp_eval2,
            ev_grp="train", st_grp="train", ns="eval_th", split="held",
            ev_set="atp_trainCtx_heldHyp", arm=arm_name, hyp_w="held")
        # 4. Held context + held hypothesis (hardest)
        ev["atp_heldCtx_heldHyp"] = compound_rows(atp_eval, atp_eval2,
            ev_grp="held", st_grp="held", ns="eval_hh", split="held",
            ev_set="atp_heldCtx_heldHyp", arm=arm_name, hyp_w="held")
        # 5. Anchor context + train hypothesis (sanity)
        ev["atp_anchorCtx_trainHyp"] = compound_rows(atp_eval, atp_eval2,
            ev_grp="anchor", st_grp="anchor", ns="eval_at", split="held",
            ev_set="atp_anchorCtx_trainHyp", arm=arm_name, hyp_w="train")
        # Paired-world event: train and held context/hypothesis
        ev["pair_trainCtx_trainHyp"] = event_only_rows(pair_eval, ev_grp="train",
            ns="ev_pair_tt", split="held", ev_set="pair_trainCtx_trainHyp",
            arm=arm_name, hyp_w="train")
        ev["pair_heldCtx_trainHyp"] = event_only_rows(pair_eval, ev_grp="held",
            ns="ev_pair_ht", split="held", ev_set="pair_heldCtx_trainHyp",
            arm=arm_name, hyp_w="train")
        ev["pair_trainCtx_heldHyp"] = event_only_rows(pair_eval, ev_grp="train",
            ns="ev_pair_th", split="held", ev_set="pair_trainCtx_heldHyp",
            arm=arm_name, hyp_w="held")
        ev["pair_heldCtx_heldHyp"] = event_only_rows(pair_eval, ev_grp="held",
            ns="ev_pair_hh", split="held", ev_set="pair_heldCtx_heldHyp",
            arm=arm_name, hyp_w="held")
        return ev

    arms = {}
    for arm in ["exposure", "aligned", "flipped"]:
        train = list(base)
        if arm == "exposure":
            train += exposure_rows(atp_sparse, atp_sparse2, ev_grp="train",
                st_grp="train", ns="sp_exp", split="train", ev_set="sparse", arm=arm)
            train += paired_exposure_event(pair_sparse, ev_grp="train",
                ns="sp_pair_exp", split="train", ev_set="sparse", arm=arm)
        else:
            mode = "true" if arm == "aligned" else "flip"
            train += compound_rows(atp_sparse, atp_sparse2, ev_grp="train",
                st_grp="train", ns="sp_rel", split="train", ev_set="sparse",
                arm=arm, lm=mode, tk=f"sparse_{arm}")
            train += event_only_rows(pair_sparse, ev_grp="train", ns="sp_pair",
                split="train", ev_set="sparse", arm=arm, lm=mode,
                tk=f"sparse_{arm}_pair")
        arms[arm] = {"train": train, "evals": make_evals(arm)}
    return arms, atp_eval, pair_eval

# ===================================================================
# Aggregation
# ===================================================================
def aggregate(results):
    by_arm = collections.defaultdict(list)
    for r in results: by_arm[r["arm"]].append(r)
    agg = {}
    for arm, rs in by_arm.items():
        item = {"n_seeds": len(rs),
                "train_acc_mean": float(np.mean([r["best_train_acc"] for r in rs])),
                "evals": {}}
        eval_names = sorted(rs[0]["evals"].keys())
        for en in eval_names:
            # Standard
            std_accs = [r["evals"][en]["standard"].get("acc", float("nan")) for r in rs]
            std_ks = {}
            for qf in ["event_role", "focal_state", "untouched_state",
                        "focal_state_conflict", "untouched_state_conflict"]:
                vals = [r["evals"][en]["standard"].get("key_subsets", {}).get(qf)
                        for r in rs]
                vals = [v for v in vals if v is not None]
                if vals: std_ks[qf] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
            # Contrastive
            con_accs = [r["evals"][en]["contrastive"].get("contrastive_acc", float("nan")) for r in rs]
            con_qs = {}
            for qf in ["event_role", "focal_state", "untouched_state"]:
                vals = [r["evals"][en]["contrastive"].get("by_query", {}).get(qf)
                        for r in rs]
                vals = [v for v in vals if v is not None]
                if vals: con_qs[qf] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
            con_cfs = {}
            for ck in ["focal_state_conflict", "focal_state_nonconflict",
                        "untouched_state_conflict", "untouched_state_nonconflict"]:
                vals = [r["evals"][en]["contrastive"].get("by_conflict", {}).get(ck)
                        for r in rs]
                vals = [v for v in vals if v is not None]
                if vals: con_cfs[ck] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
            item["evals"][en] = {
                "standard_acc": {"mean": float(np.nanmean(std_accs)), "std": float(np.nanstd(std_accs))},
                "standard_subsets": std_ks,
                "contrastive_acc": {"mean": float(np.nanmean(con_accs)), "std": float(np.nanstd(con_accs))},
                "contrastive_by_query": con_qs,
                "contrastive_by_conflict": con_cfs,
            }
        agg[arm] = item
    return agg

# ===================================================================
# Reporting
# ===================================================================
def write_md(agg, grounding, out_dir):
    lines = ["# research contrastive wording-cross probe\n"]
    if grounding:
        lines.append("## Semantic grounding (pretrained model, cosine similarity)\n")
        for k, v in grounding.items():
            if v: lines.append(f"- **{k}**: mean={v['mean']:.4f} std={v['std']:.4f} min={v['min']:.4f} n={v['n']}\n")
        lines.append("\n")
    for arm, item in sorted(agg.items()):
        lines.append(f"## Arm: {arm} (train acc {item['train_acc_mean']:.3f}, {item['n_seeds']} seeds)\n\n")
        # Table header
        lines.append("| eval_set | std_acc | std_event | std_focal | std_untouched "
                      "| con_acc | con_event | con_focal | con_untouched |\n")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for en, ev in sorted(item["evals"].items()):
            sa = ev["standard_acc"]["mean"]
            se = ev["standard_subsets"].get("event_role", {}).get("mean", float("nan"))
            sf = ev["standard_subsets"].get("focal_state", {}).get("mean", float("nan"))
            su = ev["standard_subsets"].get("untouched_state", {}).get("mean", float("nan"))
            ca = ev["contrastive_acc"]["mean"]
            ce = ev["contrastive_by_query"].get("event_role", {}).get("mean", float("nan"))
            cf = ev["contrastive_by_query"].get("focal_state", {}).get("mean", float("nan"))
            cu = ev["contrastive_by_query"].get("untouched_state", {}).get("mean", float("nan"))
            lines.append(f"| {en} | {sa:.3f} | {se:.3f} | {sf:.3f} | {su:.3f} "
                          f"| {ca:.3f} | {ce:.3f} | {cf:.3f} | {cu:.3f} |\n")
        lines.append("\n")
    (out_dir / "contrastive_wording_cross_summary.md").write_text("".join(lines), "utf-8")

# ===================================================================
# Main
# ===================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL))
    ap.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--train_atp", type=int, default=120)
    ap.add_argument("--sparse_atp", type=int, default=24)
    ap.add_argument("--eval_atp", type=int, default=120)
    ap.add_argument("--train_pair", type=int, default=120)
    ap.add_argument("--sparse_pair", type=int, default=24)
    ap.add_argument("--eval_pair", type=int, default=160)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--eval_batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--encoder_lr", type=float, default=8e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--skip_grounding", action="store_true")
    args = ap.parse_args()

    dev = args.device
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model_path)
    print(json.dumps({"status": "START", "args": vars(args),
                       "device": dev, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}), flush=True)

    # Semantic grounding check
    grounding = {}
    if not args.skip_grounding:
        atp_all = load_atp()[1]  # held set
        grounding = semantic_grounding(Path(args.model_path), tok, atp_all, dev)
        print(json.dumps({"event": "grounding", **grounding}), flush=True)

    # Build datasets
    arms, atp_eval_worlds, pair_eval_worlds = build_datasets(args)
    for arm_name, obj in arms.items():
        print(json.dumps({"event": "arm_built", "arm": arm_name,
                           "train_rows": len(obj["train"]),
                           "eval_sets": {k: len(v) for k, v in obj["evals"].items()}}), flush=True)

    # Train and evaluate
    results = []
    for arm_name, obj in arms.items():
        for seed_idx in range(args.n_seeds):
            seed = 26200 + seed_idx
            r = train_one(model_path=Path(args.model_path), tok=tok, arm=arm_name,
                seed=seed, train_rows=obj["train"], evals=obj["evals"], args=args, dev=dev)
            results.append(r)
            ta = r["best_train_acc"]
            # Quick print key evals
            for en in ["atp_trainCtx_trainHyp", "atp_heldCtx_trainHyp",
                        "atp_trainCtx_heldHyp", "atp_heldCtx_heldHyp"]:
                ev = r["evals"].get(en, {})
                std = ev.get("standard", {})
                con = ev.get("contrastive", {})
                print(json.dumps({"event": "eval", "arm": arm_name, "seed": seed,
                    "train_acc": ta, "eval_set": en,
                    "std_acc": std.get("acc"),
                    "std_event": std.get("key_subsets", {}).get("event_role"),
                    "std_focal": std.get("key_subsets", {}).get("focal_state"),
                    "std_untouched": std.get("key_subsets", {}).get("untouched_state"),
                    "con_acc": con.get("contrastive_acc"),
                    "con_event": con.get("by_query", {}).get("event_role"),
                    "con_focal": con.get("by_query", {}).get("focal_state"),
                    "con_untouched": con.get("by_query", {}).get("untouched_state"),
                }), flush=True)

    # Aggregate
    agg = aggregate(results)
    summary = {"status": "CONTRASTIVE_WORDING_CROSS",
               "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "args": vars(args), "grounding": grounding, "aggregate": agg,
               "per_seed": results}
    (out / "contrastive_wording_cross_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", "utf-8")
    write_md(agg, grounding, out)
    print(json.dumps({"status": "DONE",
        "summary_json": str(out / "contrastive_wording_cross_summary.json")}), flush=True)

if __name__ == "__main__":
    main()
