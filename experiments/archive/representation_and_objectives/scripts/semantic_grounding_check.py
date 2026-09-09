#!/usr/bin/env python3
"""research: semantic grounding check (CPU-only, no training needed).

Measures pretrained BabyLM DeBERTa cosine similarity between train-wording
and held-wording representations of the SAME facts.  High similarity means
the pretrained model considers them semantically equivalent; any fine-tuned
failure on held wordings would then be calibration, not representation.

This runs independently of GPU availability.
"""
import json, hashlib, random, time, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, DebertaV2Model

STUDY = Path("experiments/archive/representation_and_objectives")
ATP_DIR = STUDY / "data/upset_balanced_state_substrate"
MODEL = Path("experiments/archive/frontier_consolidation/training/runs"
             "cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/hf_model/chck_100M")
OUT = STUDY / "data/contrastive_wording_cross"

ALIAS_POOL = [
    "Alice","Ben","Clara","David","Emma","Frank","Grace","Henry",
    "Iris","Jack","Kate","Liam","Mia","Noah","Olivia","Paul",
    "Quinn","Rose","Sam","Tina","Uma","Victor","Wendy","Xavier",
    "Yara","Zoe","Aaron","Bella","Caleb","Diana","Ethan","Fiona",
    "Gavin","Hannah","Isaac","Julia","Kevin","Laura","Mason","Nora",
    "Owen","Paula","Riley","Sarah","Thomas","Vera","Will","Nina",
]

EVENT_CTX = {
    "train":  ["{W} prevailed over {L} in the match.",
               "{L} was beaten by {W} in the match."],
    "held":   ["{W} edged out {L} in the match.",
               "{L} went down to {W} in the match."],
}
STATE_CTX = {
    "train":  ["According to the rankings, {H} stood higher than {Lo}.",
               "The ATP list placed {Lo} under {H}."],
    "held":   ["{H} held the better ranking position than {Lo}.",
               "{Lo} trailed {H} in the weekly rankings."],
}
HYP_EVENT = {"train": "{X} won the match against {Y}.",
             "held": "{X} claimed the victory over {Y}."}
HYP_STATE = {"train": "{X} held the higher ranking than {Y}.",
             "held": "{X} outranked {Y}."}

def _si(s): return int(hashlib.sha256(s.encode()).hexdigest()[:16], 16)
def aliases(key, n): return random.Random(_si("aliases|"+key)).sample(ALIAS_POOL, n)

def load_atp_held():
    worlds = []
    for f_raw in Path(ATP_DIR / "families_held.jsonl").read_text("utf-8").splitlines():
        if not f_raw.strip(): continue
        f = json.loads(f_raw)
        for wk in ["context1", "context2"]:
            c = f[wk]; w = c["winner"]; l = c["loser"]
            a, b = f["participant_a"], f["participant_b"]
            wl = "A" if w == a else "B"
            h, lo = (w, l) if c["winner_higher_at_time"] else (l, w)
            worlds.append({"world_id": f"{f['family_id']}::{wk}",
                           "winner": w, "loser": l, "a": a, "b": b,
                           "higher": h, "lower": lo, "wl": wl})
    return worlds

def main():
    dev = "cpu"
    print(json.dumps({"status": "GROUNDING_START", "device": dev}), flush=True)
    tok = AutoTokenizer.from_pretrained(str(MODEL))
    enc = DebertaV2Model.from_pretrained(str(MODEL)).to(dev).eval()
    worlds = load_atp_held()[:60]  # bounded sample

    def pool(text):
        t = tok(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            h = enc(t["input_ids"], t["attention_mask"]).last_hidden_state
            m = t["attention_mask"].unsqueeze(-1).float()
            return (h * m).sum(1) / m.sum(1).clamp_min(1)

    cos_ev_ctx, cos_st_ctx, cos_hyp_ev, cos_hyp_st = [], [], [], []
    cos_compound = []
    t0 = time.time()

    for i, w in enumerate(worlds):
        al = aliases(f"two|grounding|{w['world_id']}", 2)
        am = {w['a']: al[0], w['b']: al[1]}
        W, L = am[w['winner']], am[w['loser']]
        H, Lo = am[w['higher']], am[w['lower']]

        # Event context: each train variant vs each held variant
        for tv in range(len(EVENT_CTX["train"])):
            r1 = pool(EVENT_CTX["train"][tv].format(W=W, L=L))
            for hv in range(len(EVENT_CTX["held"])):
                r2 = pool(EVENT_CTX["held"][hv].format(W=W, L=L))
                cos_ev_ctx.append(float(F.cosine_similarity(r1, r2).item()))

        # State context
        for tv in range(len(STATE_CTX["train"])):
            r1 = pool(STATE_CTX["train"][tv].format(H=H, Lo=Lo))
            for hv in range(len(STATE_CTX["held"])):
                r2 = pool(STATE_CTX["held"][hv].format(H=H, Lo=Lo))
                cos_st_ctx.append(float(F.cosine_similarity(r1, r2).item()))

        # Event hypothesis
        X, Y = al[0], al[1]
        r1 = pool(HYP_EVENT["train"].format(X=X, Y=Y))
        r2 = pool(HYP_EVENT["held"].format(X=X, Y=Y))
        cos_hyp_ev.append(float(F.cosine_similarity(r1, r2).item()))

        # State hypothesis
        r1 = pool(HYP_STATE["train"].format(X=X, Y=Y))
        r2 = pool(HYP_STATE["held"].format(X=X, Y=Y))
        cos_hyp_st.append(float(F.cosine_similarity(r1, r2).item()))

        # Full compound: same facts, train vs held wording for entire context
        train_ctx = (f"Match record: {EVENT_CTX['train'][0].format(W=W, L=L)} "
                     f"Ranking record: {STATE_CTX['train'][0].format(H=H, Lo=Lo)}")
        held_ctx = (f"Match record: {EVENT_CTX['held'][0].format(W=W, L=L)} "
                    f"Ranking record: {STATE_CTX['held'][0].format(H=H, Lo=Lo)}")
        r1 = pool(train_ctx)
        r2 = pool(held_ctx)
        cos_compound.append(float(F.cosine_similarity(r1, r2).item()))

        if (i + 1) % 20 == 0:
            print(json.dumps({"event": "progress", "worlds": i+1,
                               "elapsed": round(time.time()-t0, 1)}), flush=True)

    def stats(v):
        if not v: return {}
        return {"mean": round(float(np.mean(v)), 4),
                "std": round(float(np.std(v)), 4),
                "min": round(float(np.min(v)), 4),
                "max": round(float(np.max(v)), 4),
                "median": round(float(np.median(v)), 4),
                "n": len(v)}

    result = {
        "status": "SEMANTIC_GROUNDING",
        "n_worlds": len(worlds),
        "elapsed_seconds": round(time.time() - t0, 2),
        "event_context_cosine": stats(cos_ev_ctx),
        "state_context_cosine": stats(cos_st_ctx),
        "event_hypothesis_cosine": stats(cos_hyp_ev),
        "state_hypothesis_cosine": stats(cos_hyp_st),
        "compound_context_cosine": stats(cos_compound),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "semantic_grounding.json").write_text(
        json.dumps(result, indent=2) + "\n", "utf-8")

    # Readable summary
    lines = ["# research Semantic Grounding (pretrained BabyLM DeBERTa)\n\n"]
    lines.append("Cosine similarity between train-wording and held-wording representations\n")
    lines.append("of the same facts, using the pretrained model BEFORE fine-tuning.\n\n")
    for k in ["event_context_cosine", "state_context_cosine",
              "event_hypothesis_cosine", "state_hypothesis_cosine",
              "compound_context_cosine"]:
        s = result[k]
        if s:
            lines.append(f"- **{k}**: mean={s['mean']:.4f} ± {s['std']:.4f}, "
                         f"min={s['min']:.4f}, max={s['max']:.4f}, n={s['n']}\n")
    lines.append(f"\nElapsed: {result['elapsed_seconds']}s on {len(worlds)} worlds.\n")
    lines.append("\n## Interpretation\n\n")
    lines.append("If cosine similarity is high (>0.9), the pretrained model considers\n")
    lines.append("train and held wordings as semantically equivalent representations.\n")
    lines.append("Any fine-tuned held-wording failure would then be attributable to\n")
    lines.append("template-specific calibration in the classification head, not to\n")
    lines.append("representational non-equivalence in the pretrained encoder.\n")
    ((OUT.parents[4] / 'research/documents/representation_and_objectives/data/contrastive_wording_cross/semantic_grounding.md')).write_text("".join(lines), "utf-8")

    print(json.dumps(result), flush=True)

if __name__ == "__main__":
    main()
