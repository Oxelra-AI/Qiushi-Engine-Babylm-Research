#!/usr/bin/env python3
"""research: causal donor-query activation intervention.

Tests whether the query-match component at context attribute positions causally
determines the answer by donor-query redirection.  For the same context, run
with query A (donor) to extract attribute-position hidden states, then run with
query B but patch those states at an intermediate layer.  If the answer redirects
from B's attribute to A's attribute, the component carries usable entity-selection
information—not merely novelty, scale, or correlated content.

Three intervention types at each layer (0,1,2):
  full     – replace entire hidden state at ATTR_POS from donor
  d_only   – replace only the query-match direction component
  orth_only – replace only the orthogonal complement (negative control)

Tested on preparation (bound), direct_full (collapsed), and static_1over17
(preserved) models.  Cross-model transplant tests whether the collapsed model's
downstream readout can use the preserving model's query-match signal.
"""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import argparse, copy, json, math, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = _public_path('experiments/archive/functional_learning/scripts')
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import loss_allocation_binding as Base
import query_first_binding as S16
import binding_branching as S17
import budget_matched_full_objective as S18
import revision_019b_embedding_role_decomposition as S19b

SL = 18; IS_POS = 15; VOCAB = S16.VOCAB; PAD = S16.PAD
RWT = S16.RWT; RWT_TOKS = S16.RWT_TOKS
ATTR_POS = [5, 8, 11, 14]
K = Base.K; N_ATTR = Base.N_ATTR
TRAIN_E = Base.TRAIN_E; HELD_E = Base.HELD_E
DEFAULT_PREP = {42: 300, 43: 500, 100: 400}

ARMS = {
    "direct_full":    dict(target="bound", kind="static", lr=3e-4, ctx_weight=1.0),
    "static_1over17": dict(target="bound", kind="static", lr=3e-4, ctx_weight=1.0/17.0),
}


# ═══════════════════ continuation training ═══════════════════
def custom_seqs(rows, target):
    seqs = []
    for ce, ca, qi, bag_ti in rows:
        tgt = RWT(ca[qi]) if target == "bound" else RWT(ca[bag_ti])
        seqs.append(S16.mk_seq("query_first", ce, ca, qi, tgt))
    return seqs


def train_epoch(model, opt, seqs, order_idx, dev, bs, mask, kind, ctx_weight):
    st = torch.tensor(seqs, dtype=torch.long)[torch.tensor(order_idx, dtype=torch.long)]
    L = SL - 1; model.train(); tot = den = 0.0
    for i in range(0, st.size(0), bs):
        b = st[i:i+bs].to(dev); inp, tgt = b[:, :L], b[:, 1:]
        logits = model(inp, mask)
        ce = F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1),
                             reduction='none').view(tgt.shape)
        nonpad = (tgt != PAD).float()
        if kind == "context_only":
            wt = nonpad.clone(); wt[:, IS_POS] = 0.0; wt[:, -1] = 0.0
        else:
            pw = torch.full((L,), float(ctx_weight), device=dev)
            pw[IS_POS] = 1.0; pw[-1] = 0.0
            wt = nonpad * pw.unsqueeze(0)
        loss = (ce * wt).sum() / wt.sum().clamp_min(1.0)
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            tot += float((ce * wt).sum()); den += float(wt.sum())
    return tot / max(den, 1)


def train_continuation(tied_state, seed, P, arm_name, cfg, dev, cm, n_train, epochs):
    arm = ARMS[arm_name]
    model = S19b.make_untied_from_tied_state(tied_state, cfg, dev)
    torch.manual_seed(seed + 250000 + abs(hash(arm_name)) % 9999)
    opt = torch.optim.AdamW(model.parameters(), lr=arm['lr'], weight_decay=cfg['wd'])
    for be in range(1, epochs + 1):
        ep = P + be
        rows = S16.make_epoch_rows(seed, ep, n_train)
        seqs = custom_seqs(rows, arm['target'])
        idx = S16.common_order(seed, ep, n_train)
        train_epoch(model, opt, seqs, idx, dev, cfg['bs'], cm, arm['kind'],
                    arm['ctx_weight'])
    return model


# ═══════════════════ hook-based forward passes ═══════════════════
def forward_extract(model, inp, mask, layers):
    """Return (logits, {layer: (B,4,d) at ATTR_POS})."""
    stored = {}; hooks = []
    for li in layers:
        def _mk(l):
            def hook(mod, inp_arg, out):
                stored[l] = out[:, ATTR_POS, :].detach().clone()
            return hook
        hooks.append(model.blks[li].register_forward_hook(_mk(li)))
    model.eval()
    with torch.no_grad():
        logits = model(inp, mask)
    for h in hooks:
        h.remove()
    return logits, stored


def forward_patch(model, inp, mask, layer, donor, mode='full', direction=None):
    """Return logits with ATTR_POS patched after block `layer`."""
    def _hook(mod, inp_arg, out):
        p = out.clone()
        h = p[:, ATTR_POS, :]
        if mode == 'full':
            p[:, ATTR_POS, :] = donor
        elif mode == 'd_only':
            d = direction.to(out.device).view(1, 1, -1)
            proj_c = (h * d).sum(-1, keepdim=True) * d
            proj_d = (donor * d).sum(-1, keepdim=True) * d
            p[:, ATTR_POS, :] = h - proj_c + proj_d
        elif mode == 'orth_only':
            d = direction.to(out.device).view(1, 1, -1)
            proj_c = (h * d).sum(-1, keepdim=True) * d
            proj_d = (donor * d).sum(-1, keepdim=True) * d
            p[:, ATTR_POS, :] = proj_c + (donor - proj_d)
        return p
    hk = model.blks[layer].register_forward_hook(_hook)
    model.eval()
    with torch.no_grad():
        logits = model(inp, mask)
    hk.remove()
    return logits


# ═══════════════════ direction learning ═══════════════════
def _gen_dir_seqs(rng, n):
    seqs, qis = [], []
    for _ in range(n):
        ce = rng.choice(TRAIN_E, K, replace=False).tolist()
        ca = rng.choice(N_ATTR, K, replace=False).tolist()
        qi = int(rng.integers(K))
        seqs.append(S16.mk_seq("query_first", ce, ca, qi, PAD))
        qis.append(qi)
    return seqs, qis


def _extract_feats(model, seqs, dev, cm, layers, bs=128):
    feats = {l: [] for l in layers}
    for i in range(0, len(seqs), bs):
        inp = torch.tensor(seqs[i:i+bs], dtype=torch.long, device=dev)[:, :SL-1]
        _, stored = forward_extract(model, inp, cm, layers)
        for l in layers:
            feats[l].append(stored[l].cpu())
    return {l: torch.cat(feats[l], 0) for l in layers}


def learn_directions(model, dev, cm, seed=250250, n_train=1500, n_test=500):
    """Learn query-match directions from the preparation model at each layer."""
    rng = np.random.default_rng(seed)
    layers = list(range(len(model.blks)))
    tr_seqs, tr_qi = _gen_dir_seqs(rng, n_train)
    te_seqs, te_qi = _gen_dir_seqs(rng, n_test)
    tr_feats = _extract_feats(model, tr_seqs, dev, cm, layers)
    te_feats = _extract_feats(model, te_seqs, dev, cm, layers)
    tr_lab = torch.tensor(tr_qi, dtype=torch.long)
    te_lab = torch.tensor(te_qi, dtype=torch.long)

    directions, tr_acc, te_acc = {}, {}, {}
    for l in layers:
        f = tr_feats[l]                     # (N, 4, d)
        N, ns, d = f.shape
        matched = torch.stack([f[j, int(tr_lab[j])] for j in range(N)])
        um = [f[j, s] for j in range(N) for s in range(ns) if s != int(tr_lab[j])]
        unmatched = torch.stack(um)
        diff = matched.mean(0) - unmatched.mean(0)
        direction = diff / diff.norm().clamp_min(1e-8)
        directions[l] = direction
        sc = (f * direction.view(1, 1, -1)).sum(-1)
        tr_acc[l] = float((sc.argmax(1) == tr_lab).float().mean())
        ste = te_feats[l]
        sc2 = (ste * direction.view(1, 1, -1)).sum(-1)
        te_acc[l] = float((sc2.argmax(1) == te_lab).float().mean())
    return directions, tr_acc, te_acc


def test_direction(model, directions, dev, cm, seed=250251, n=500):
    """Test preparation-learned directions on another model."""
    rng = np.random.default_rng(seed)
    seqs, qis = _gen_dir_seqs(rng, n)
    layers = list(directions.keys())
    feats = _extract_feats(model, seqs, dev, cm, layers)
    lab = torch.tensor(qis, dtype=torch.long)
    accs = {}
    for l in layers:
        d = directions[l]
        sc = (feats[l] * d.view(1, 1, -1)).sum(-1)
        accs[l] = float((sc.argmax(1) == lab).float().mean())
    return accs


# ═══════════════════ pair generation ═══════════════════
def gen_train_pairs(seed, n):
    rng = np.random.default_rng(seed)
    pairs = []
    for _ in range(n):
        ce = rng.choice(TRAIN_E, K, replace=False).tolist()
        ca = rng.choice(N_ATTR, K, replace=False).tolist()
        qi_a = int(rng.integers(K)); qi_b = qi_a
        while qi_b == qi_a:
            qi_b = int(rng.integers(K))
        pairs.append(dict(
            seq_a=S16.mk_seq("query_first", ce, ca, qi_a, PAD),
            seq_b=S16.mk_seq("query_first", ce, ca, qi_b, PAD),
            target_a=ca[qi_a], target_b=ca[qi_b]))
    return pairs


def gen_held_pairs(seed, n):
    """Return dict with 'held_donor' and 'train_donor' pair lists."""
    rng = np.random.default_rng(seed)
    out = {'held_donor': [], 'train_donor': []}
    for _ in range(n):
        he = int(rng.choice(HELD_E))
        others = rng.choice(TRAIN_E, K-1, replace=False).tolist()
        ce = list(rng.permutation(others + [he]))
        ca = rng.choice(N_ATTR, K, replace=False).tolist()
        hi = ce.index(he)
        ti = hi
        while ti == hi:
            ti = int(rng.integers(K))
        seq_h = S16.mk_seq("query_first", ce, ca, hi, PAD)
        seq_t = S16.mk_seq("query_first", ce, ca, ti, PAD)
        out['held_donor'].append(dict(
            seq_a=seq_h, seq_b=seq_t,
            target_a=ca[hi], target_b=ca[ti]))
        out['train_donor'].append(dict(
            seq_a=seq_t, seq_b=seq_h,
            target_a=ca[ti], target_b=ca[hi]))
    return out


# ═══════════════════ intervention evaluation ═══════════════════
def _eval_pairs(model, pairs, directions, dev, cm, layers,
                modes=('full', 'd_only', 'orth_only'), bs=64):
    """Core intervention loop.  Returns (metrics_dict, {layer: donor_acts})."""
    all_logits_a = []; all_logits_b = []; donor_all = {l: [] for l in layers}

    for i in range(0, len(pairs), bs):
        batch = pairs[i:i+bs]
        inp_a = torch.tensor([p['seq_a'] for p in batch], dtype=torch.long, device=dev)[:, :SL-1]
        inp_b = torch.tensor([p['seq_b'] for p in batch], dtype=torch.long, device=dev)[:, :SL-1]
        logits_a, stored_a = forward_extract(model, inp_a, cm, layers)
        all_logits_a.append(logits_a[:, IS_POS].cpu())
        for l in layers:
            donor_all[l].append(stored_a[l].cpu())
        model.eval()
        with torch.no_grad():
            all_logits_b.append(model(inp_b, cm)[:, IS_POS].cpu())

    all_logits_a = torch.cat(all_logits_a, 0)
    all_logits_b = torch.cat(all_logits_b, 0)
    for l in layers:
        donor_all[l] = torch.cat(donor_all[l], 0)

    # clean accuracy
    ca = sum(1 for j, p in enumerate(pairs) if int(all_logits_a[j].argmax()) == RWT(p['target_a']))
    cb = sum(1 for j, p in enumerate(pairs) if int(all_logits_b[j].argmax()) == RWT(p['target_b']))
    N = len(pairs)
    res = dict(clean_a_acc=round(ca / max(N, 1), 4), clean_b_acc=round(cb / max(N, 1), 4), n=N)

    for l in layers:
        don = donor_all[l]
        for mode in modes:
            redir = 0; nat = 0; rnlls = []; nnlls = []
            d = directions.get(l) if directions else None
            for i in range(0, N, bs):
                batch = pairs[i:i+bs]; bsz = len(batch)
                inp_b = torch.tensor([p['seq_b'] for p in batch],
                                     dtype=torch.long, device=dev)[:, :SL-1]
                donor = don[i:i+bsz].to(dev)
                logits_p = forward_patch(model, inp_b, cm, l, donor, mode, d)
                for j, p in enumerate(batch):
                    ta = RWT(p['target_a']); tb = RWT(p['target_b'])
                    lp = F.log_softmax(logits_p[j, IS_POS], dim=-1)
                    pred = int(logits_p[j, IS_POS].argmax())
                    if pred == ta: redir += 1
                    if pred == tb: nat += 1
                    rnlls.append(float(-lp[ta])); nnlls.append(float(-lp[tb]))
            key = f"L{l}_{mode}"
            res[key] = dict(
                redirect_rate=round(redir / max(N, 1), 4),
                natural_rate=round(nat / max(N, 1), 4),
                redirect_nll=round(float(np.mean(rnlls)), 4),
                natural_nll=round(float(np.mean(nnlls)), 4),
                redirect_margin=round(float(np.mean(nnlls)) - float(np.mean(rnlls)), 4))
    return res, donor_all


def run_cross_model(model_recv, pairs, ext_donor, directions, dev, cm,
                    layers, modes=('full', 'd_only'), bs=64):
    """Run intervention on model_recv using donor activations from another model."""
    N = len(pairs); res = {}
    for l in layers:
        don = ext_donor[l]
        for mode in modes:
            redir = 0; nat = 0; rnlls = []; nnlls = []
            d = directions.get(l) if directions else None
            for i in range(0, N, bs):
                batch = pairs[i:i+bs]; bsz = len(batch)
                inp_b = torch.tensor([p['seq_b'] for p in batch],
                                     dtype=torch.long, device=dev)[:, :SL-1]
                donor = don[i:i+bsz].to(dev)
                logits_p = forward_patch(model_recv, inp_b, cm, l, donor, mode, d)
                for j, p in enumerate(batch):
                    ta = RWT(p['target_a']); tb = RWT(p['target_b'])
                    lp = F.log_softmax(logits_p[j, IS_POS], dim=-1)
                    pred = int(logits_p[j, IS_POS].argmax())
                    if pred == ta: redir += 1
                    if pred == tb: nat += 1
                    rnlls.append(float(-lp[ta])); nnlls.append(float(-lp[tb]))
            res[f"L{l}_{mode}"] = dict(
                redirect_rate=round(redir / max(N, 1), 4),
                natural_rate=round(nat / max(N, 1), 4),
                redirect_nll=round(float(np.mean(rnlls)), 4),
                natural_nll=round(float(np.mean(nnlls)), 4),
                redirect_margin=round(float(np.mean(nnlls)) - float(np.mean(rnlls)), 4))
    return res


# ═══════════════════ quick behavior ═══════════════════
def quick_behavior(model, probes, dev, cm_blk, prep_tok, prep_out):
    pack = S19b.eval_pack(model, probes, dev, cm_blk, prep_tok, prep_out)
    sm = S19b.metric_summary(pack)
    return {k: round(sm.get(k, 0.0), 4) for k in
            ['held_top4', 'held_b', 'held_sel', 'train_top4']}


# ═══════════════════ output helpers ═══════════════════
class NpEnc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating, float)): return round(float(o), 6)
        if isinstance(o, np.ndarray): return o.tolist()
        return super().default(o)


def write_note(data, path):
    L = ["# research: causal donor-query activation intervention", "",
         "## Design", "",
         "Same context, two queries A and B. Donor run (query A) extracts hidden",
         "states at ATTR_POS after each layer. Recipient run (query B) patches those",
         "states. If the answer redirects from B's attribute to A's attribute, the",
         "component carries usable entity-selection information.", "",
         "- **full**: replace entire hidden state at ATTR_POS",
         "- **d_only**: replace only the prep-learned query-match direction component",
         "- **orth_only**: replace only the orthogonal complement (control)", ""]

    for sk, sd in data.get('seeds', {}).items():
        L.append(f"## Seed {sk}"); L.append("")
        # behavior
        L.append("### Behavior after continuation")
        L.append("| model | held_top4 | held_b | held_sel | train_top4 |")
        L.append("|---|---:|---:|---:|---:|")
        for m in ['prep', 'direct_full', 'static_1over17']:
            b = sd.get('behavior', {}).get(m, {})
            L.append(f"| {m} | {b.get('held_top4',0):.3f} | {b.get('held_b',0):+.3f} | "
                     f"{b.get('held_sel',0):+.3f} | {b.get('train_top4',0):.3f} |")
        L.append("")
        # direction
        L.append("### Direction accuracy (prep-learned, tested per model)")
        L.append("| model | L0 | L1 | L2 |")
        L.append("|---|---:|---:|---:|")
        for m in ['prep', 'direct_full', 'static_1over17']:
            da = sd.get('dir_acc', {}).get(m, {})
            vals = [f"{da.get(l, da.get(str(l), 0)):.3f}" for l in range(3)]
            L.append(f"| {m} | {vals[0]} | {vals[1]} | {vals[2]} |")
        L.append("")
        # train redirect
        L.append("### Train-entity donor-query redirection")
        for m in ['prep', 'direct_full', 'static_1over17']:
            iv = sd.get('iv_train', {}).get(m, {})
            if not iv: continue
            L.append(f"**{m}** clean: A={iv.get('clean_a_acc',0):.3f}, B={iv.get('clean_b_acc',0):.3f}")
            L.append("| layer_mode | redirect | natural | margin |")
            L.append("|---|---:|---:|---:|")
            for k in sorted(k for k in iv if k.startswith('L')):
                v = iv[k]
                L.append(f"| {k} | {v['redirect_rate']:.3f} | {v['natural_rate']:.3f} | "
                         f"{v['redirect_margin']:+.3f} |")
            L.append("")
        # held
        for case_name in ['held_donor', 'train_donor']:
            L.append(f"### Held-entity redirection ({case_name})")
            for m in ['prep', 'direct_full', 'static_1over17']:
                iv = sd.get(f'iv_{case_name}', {}).get(m, {})
                if not iv: continue
                L.append(f"**{m}**")
                L.append("| layer_mode | redirect | natural | margin |")
                L.append("|---|---:|---:|---:|")
                for k in sorted(k for k in iv if k.startswith('L')):
                    v = iv[k]
                    L.append(f"| {k} | {v['redirect_rate']:.3f} | {v['natural_rate']:.3f} | "
                             f"{v['redirect_margin']:+.3f} |")
                L.append("")
        # cross-model
        xm = sd.get('cross_model', {})
        if xm:
            L.append("### Cross-model transplant (preserving → collapsed)")
            L.append("| layer_mode | redirect | natural | margin |")
            L.append("|---|---:|---:|---:|")
            for k in sorted(k for k in xm if k.startswith('L')):
                v = xm[k]
                L.append(f"| {k} | {v['redirect_rate']:.3f} | {v['natural_rate']:.3f} | "
                         f"{v['redirect_margin']:+.3f} |")
            L.append("")
    L += ["## Interpretation", "",
          "Redirect rate measures whether the model's answer follows the donor query",
          "rather than the natural query. High redirect with d_only but not orth_only",
          "establishes that the query-match direction causally determines the answer.",
          "Collapsed models with moderate direction accuracy but low redirect rate",
          "indicate the signal exists but is no longer functionally used.",
          "Cross-model transplant tests whether the collapsed model's downstream",
          "readout can still process the preserving model's query-match signal.", ""]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(L))


# ═══════════════════ main ═══════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint-dir',
                    default='experiments/archive/functional_learning/data'
                            'revision_019b_embedding_role_decomposition/checkpoints')
    ap.add_argument('--data', default='experiments/archive/functional_learning/data'
                                      'causal_intervention')
    ap.add_argument('--note', default='research/notes/functional_learning'
                                      'causal_intervention.md')
    ap.add_argument('--seeds', default='43,100')
    ap.add_argument('--cont-epochs', type=int, default=100)
    ap.add_argument('--n-train', type=int, default=500)
    ap.add_argument('--n-dir', type=int, default=2000)
    ap.add_argument('--n-pairs-train', type=int, default=500)
    ap.add_argument('--n-pairs-held', type=int, default=200)
    ap.add_argument('--smoke', action='store_true')
    A = ap.parse_args()
    if A.smoke:
        A.seeds = '100'; A.cont_epochs = 10; A.n_train = 64
        A.n_dir = 200; A.n_pairs_train = 50; A.n_pairs_held = 25
        n_probe_std, n_probe_sm = 96, 48
    else:
        n_probe_std, n_probe_sm = 256, 128

    seeds = [int(s) for s in A.seeds.split(',') if s.strip()]
    cfg = dict(d=64, nh=2, nl=3, lr=3e-4, wd=0.01, bs=64)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cm = S17.standard_causal_mask(SL-1, dev)
    cm_blk = S17.block_query_ctx_mask(SL-1, dev)
    probes = S18.make_probes(seed=180018, n_std=n_probe_std, n_small=n_probe_sm)
    ckpt_dir = Path(A.checkpoint_dir)
    layers = list(range(3))
    modes = ('full', 'd_only', 'orth_only')

    train_pairs = gen_train_pairs(250350, A.n_pairs_train)
    held_pair_dict = gen_held_pairs(250360, A.n_pairs_held)

    all_data = dict(config=dict(seeds=seeds, cont_epochs=A.cont_epochs,
                                n_train=A.n_train, n_dir=A.n_dir,
                                n_pairs_train=A.n_pairs_train,
                                n_pairs_held=A.n_pairs_held), seeds={})
    t0 = time.time()
    print(f"Device: {dev}; seeds={seeds}; cont_epochs={A.cont_epochs}", flush=True)

    for sd in seeds:
        P = DEFAULT_PREP[sd]
        tied = torch.load(ckpt_dir / f'seed{sd}_prep_tied.pt', map_location=dev)
        prep_tok = tied['tok.weight'].detach().clone().to(dev)

        SD = dict(behavior={}, dir_acc={}, iv_train={},
                  iv_held_donor={}, iv_train_donor={}, cross_model={})

        # ── preparation model ──
        m_prep = S19b.make_untied_from_tied_state(tied, cfg, dev)
        prep_out = m_prep.out.weight.detach().clone()
        beh = quick_behavior(m_prep, probes, dev, cm_blk, prep_tok, prep_out)
        SD['behavior']['prep'] = beh
        print(f"seed={sd} prep  h4={beh['held_top4']:.3f} hB={beh['held_b']:+.3f}", flush=True)

        dirs, dtr, dte = learn_directions(m_prep, dev, cm, seed=250250+sd,
                                          n_train=int(A.n_dir*0.75),
                                          n_test=A.n_dir - int(A.n_dir*0.75))
        SD['dir_acc']['prep'] = dte
        print(f"  prep dir test acc: {dte}", flush=True)

        # ── continuations ──
        models = dict(prep=m_prep)
        for aname in ['direct_full', 'static_1over17']:
            m = train_continuation(tied, sd, P, aname, cfg, dev, cm,
                                   A.n_train, A.cont_epochs)
            models[aname] = m
            beh = quick_behavior(m, probes, dev, cm_blk, prep_tok, prep_out)
            SD['behavior'][aname] = beh
            da = test_direction(m, dirs, dev, cm, seed=250251+sd,
                                n=A.n_dir - int(A.n_dir*0.75))
            SD['dir_acc'][aname] = da
            print(f"  {aname:20s} h4={beh['held_top4']:.3f} hB={beh['held_b']:+.3f} "
                  f"dir={da}", flush=True)

        # ── donor-query interventions ──
        preserving_donor = None
        for mname, model in models.items():
            print(f"  intervention {mname} ...", end=' ', flush=True)
            iv_tr, don_tr = _eval_pairs(model, train_pairs, dirs, dev, cm,
                                        layers, modes)
            SD['iv_train'][mname] = iv_tr
            # held — two cases
            iv_hd, _ = _eval_pairs(model, held_pair_dict['held_donor'],
                                   dirs, dev, cm, layers, modes)
            SD['iv_held_donor'][mname] = iv_hd
            iv_td, _ = _eval_pairs(model, held_pair_dict['train_donor'],
                                   dirs, dev, cm, layers, modes)
            SD['iv_train_donor'][mname] = iv_td
            if mname == 'static_1over17':
                preserving_donor = don_tr
            # compact print
            best_l = max(layers, key=lambda l: iv_tr.get(f'L{l}_full', {}).get('redirect_rate', 0))
            bf = iv_tr.get(f'L{best_l}_full', {})
            print(f"best L{best_l}_full redir={bf.get('redirect_rate',0):.3f} "
                  f"nat={bf.get('natural_rate',0):.3f}", flush=True)

        # ── cross-model transplant ──
        if preserving_donor is not None and 'direct_full' in models:
            print(f"  cross-model transplant ...", flush=True)
            xm = run_cross_model(models['direct_full'], train_pairs,
                                 preserving_donor, dirs, dev, cm, layers,
                                 modes=('full', 'd_only'))
            SD['cross_model'] = xm

        all_data['seeds'][str(sd)] = SD

    out_dir = Path(A.data); out_dir.mkdir(parents=True, exist_ok=True)
    outp = out_dir / 'results.json'
    outp.write_text(json.dumps(all_data, indent=2, cls=NpEnc))
    write_note(all_data, A.note)
    elapsed = round(time.time() - t0, 1)
    print(json.dumps(dict(status='ok', out=str(outp), note=A.note, elapsed=elapsed),
                     indent=2))


if __name__ == '__main__':
    main()
