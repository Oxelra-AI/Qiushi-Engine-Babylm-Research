#!/usr/bin/env python3
"""research: Matched source-trigger experiment.

Scientific question: When does identity practice create a source-triggered
competing prediction (BabyLM-like, positive D_m) versus broad target-prior
harm (Step8b-like, D_m ≈ 0)?

Key design:
  - Common correspondence backbone (100 seqs/epoch) for ALL arms
  - Shared context draws for IDENT and UNPAIRED arms
  - IDENT_FULL and IDENT_BLOCKED use identical token sequences
  - UNPAIRED_SRC uses same contexts but deranged targets
  - Corrected T/U probes with matched token multisets (entity-attribute swap)

Arms (400 extra seqs/epoch each):
  IDENT_FULL:    SRC(a)->SRC(a), full causal attention
  IDENT_BLOCKED: SRC(a)->SRC(a), query segment blocked from source event
  UNPAIRED_SRC:  SRC(a)->SRC(b) b!=a, full attention (main comparator)

Source-specific interaction:
  D_m(L,Q) = (m_IDENT,T - m_UP,T) - (m_IDENT,U - m_UP,U)
  Positive D_rwt_nll = identity pairing causes T-specific rewrite damage
  Positive D_p_src_a = identity pairing causes T-specific source-token elevation
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, json, time, argparse
from pathlib import Path

# ─── Vocabulary ────────────────────────────────────────────────────
BOS, PAD, SEP, HAS, IS = 0, 1, 2, 3, 4
N_ENT, N_ATTR = 8, 10
ENT  = lambda e: 5 + e       # 5..12
SRC  = lambda a: 13 + a      # 13..22
RWT  = lambda a: 23 + a      # 23..32
COPY_CUE, REWRITE_CUE, NEUTRAL_CUE, CONST_CUE = 33, 34, 35, 36
VOCAB = 37
TRAIN_E = list(range(6)); HELD_E = [6, 7]; ALL_E = list(range(8))
N_CTX = 4
SRC_TOKS = list(range(13, 23))
RWT_TOKS = list(range(23, 33))
SL  = 19  # BOS 4*(E HAS S) SEP E CUE IS TGT PAD
IS_POS  = 16  # input position whose logits predict TGT
CUE_POS = 15  # sequence position of CUE token

def mk_seq(ce, ca, qi, cue, tgt):
    s = [BOS]
    for i in range(N_CTX): s += [ENT(ce[i]), HAS, SRC(ca[i])]
    s += [SEP, ENT(ce[qi]), cue, IS, tgt, PAD]
    return s[:SL]

# ─── Data generation ───────────────────────────────────────────────
class DG:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def _ctx(self):
        while True:
            ce = self.rng.choice(ALL_E, N_CTX, replace=False).tolist()
            c = [i for i, e in enumerate(ce) if e in TRAIN_E]
            if c:
                qi = int(self.rng.choice(c))
                ca = [int(self.rng.integers(N_ATTR)) for _ in range(N_CTX)]
                return ce, ca, qi

    def base_draws(self, n):
        """Context draws shared across IDENT and UNPAIRED arms."""
        return [dict(ce=ce, ca=ca, qi=qi, a=ca[qi])
                for ce, ca, qi in (self._ctx() for _ in range(n))]

    def backbone(self, n, cue_mode):
        cue = REWRITE_CUE if cue_mode == "typed" else CONST_CUE
        draws = self.base_draws(n)
        return [mk_seq(d["ce"], d["ca"], d["qi"], cue, RWT(d["a"])) for d in draws]

    @staticmethod
    def ident_seqs(draws, cue_mode):
        cue = COPY_CUE if cue_mode == "typed" else CONST_CUE
        return [mk_seq(d["ce"], d["ca"], d["qi"], cue, SRC(d["a"])) for d in draws]

    @staticmethod
    def unpaired_seqs(draws, cue_mode, drng):
        """Deranged targets: SRC(b) with b != a, same context."""
        cue = COPY_CUE if cue_mode == "typed" else CONST_CUE
        seqs = []
        for d in draws:
            b = d["a"]
            while b == d["a"]: b = int(drng.integers(N_ATTR))
            seqs.append(mk_seq(d["ce"], d["ca"], d["qi"], cue, SRC(b)))
        return seqs

    def probes_tu(self, n_per=60):
        """Corrected T/U/NS probes.  CUE_POS = PAD placeholder (filled at eval).
        T: query entity has correct attribute in context.
        U: swap query's attribute with another entity → same token multiset.
        NS: query entity absent from context."""
        probes = []
        for he in HELD_E:
            for _ in range(n_per):
                a = int(self.rng.integers(N_ATTR))
                oth = self.rng.choice(TRAIN_E, N_CTX - 1, replace=False).tolist()
                ce = list(self.rng.permutation(oth + [he]))
                qi = ce.index(he)
                ca = [a if e == he else int(self.rng.integers(N_ATTR))
                      for _, e in enumerate(ce)]
                # T probe
                s_t = mk_seq(ce, ca, qi, PAD, PAD)
                # U probe: swap he's attr with another entity
                oi = [i for i in range(N_CTX) if i != qi]
                si = int(self.rng.choice(oi))
                ca_u = list(ca); ca_u[qi], ca_u[si] = ca[si], a
                s_u = mk_seq(ce, ca_u, qi, PAD, PAD)
                # NS probe: he not in context
                ns_ce = self.rng.choice(TRAIN_E, N_CTX, replace=False).tolist()
                ns_ca = [int(self.rng.integers(N_ATTR)) for _ in range(N_CTX)]
                s_ns = [BOS]
                for i in range(N_CTX):
                    s_ns += [ENT(ns_ce[i]), HAS, SRC(ns_ca[i])]
                s_ns += [SEP, ENT(he), PAD, IS, PAD, PAD]
                s_ns = (s_ns + [PAD] * SL)[:SL]
                probes.append(dict(s_t=s_t, s_u=s_u, s_ns=s_ns,
                                   ct=RWT(a), cp=SRC(a), a=a))
        return probes

# ─── Model (identical to Step10b) ─────────────────────────────────
class Blk(nn.Module):
    def __init__(self, d, nh):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, nh, batch_first=True, dropout=0)
        self.ln1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
        self.ln2 = nn.LayerNorm(d)
    def forward(self, x, m):
        h = self.ln1(x); h, _ = self.attn(h, h, h, attn_mask=m)
        x = x + h; return x + self.ff(self.ln2(x))

class CLM(nn.Module):
    def __init__(self, V, d, nh, nl, ml):
        super().__init__()
        self.tok = nn.Embedding(V, d); self.pos = nn.Embedding(ml, d)
        self.blks = nn.ModuleList([Blk(d, nh) for _ in range(nl)])
        self.ln = nn.LayerNorm(d)
    def forward(self, x, m=None):
        B, L = x.shape
        h = self.tok(x) + self.pos(torch.arange(L, device=x.device))
        for b in self.blks: h = b(h, m)
        return self.ln(h) @ self.tok.weight.T

# ─── Attention masks ──────────────────────────────────────────────
def causal(L, dev="cpu"):
    return torch.triu(torch.ones(L, L, dtype=torch.bool, device=dev), diagonal=1)

def build_blocked_masks(sub_draws, n_bb, L):
    """3D masks: causal for backbone; causal+blocked for identity seqs.
    Block positions [14,15,16] from query entity's context event
    (positions qi*3+1, qi*3+2, qi*3+3)."""
    N = n_bb + len(sub_draws)
    cm = causal(L)
    masks = cm.unsqueeze(0).expand(N, -1, -1).clone()
    for i, d in enumerate(sub_draws):
        idx = n_bb + i
        qi = d["qi"]
        for bp in [14, 15, 16]:
            for sp in [qi*3 + 1, qi*3 + 2, qi*3 + 3]:
                if bp < L and sp < L:
                    masks[idx, bp, sp] = True
    return masks

# ─── Training ─────────────────────────────────────────────────────
def train_ep(model, opt, seqs_t, m3d, dev, nh, bs=64):
    model.train()
    N = seqs_t.size(0); L = seqs_t.size(1) - 1
    idx = torch.randperm(N); seqs_t = seqs_t[idx]
    if m3d is not None: m3d = m3d[idx]
    tl, tt = 0., 0
    for i in range(0, N, bs):
        batch = seqs_t[i:i+bs].to(dev)
        inp, tgt = batch[:, :-1], batch[:, 1:]
        bsz = inp.size(0)
        if m3d is not None:
            m = m3d[i:i+bsz, :L, :L].to(dev)
            m = m.unsqueeze(1).expand(-1, nh, -1, -1).reshape(bsz*nh, L, L)
        else:
            m = causal(L, dev)
        logits = model(inp, m)
        lm = (tgt != PAD).float()
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1), reduction='none')
        loss = (loss.view(tgt.shape) * lm).sum() / max(lm.sum(), 1)
        opt.zero_grad(); loss.backward(); opt.step()
        tl += loss.item() * lm.sum().item(); tt += lm.sum().item()
    return tl / max(tt, 1)

# ─── Evaluation ────────────────────────────────────────────────────
MKEYS = ["rwt_nll", "rwt_fam", "rwt_fnll", "rwt_wnll", "rwt_top1", "rwt_mrr",
         "p_src_a", "src_fam", "src_wnll", "copy_nll", "decomp"]

def eval_probes(model, probes, pkey, cue_tok, dev):
    """Comprehensive probe evaluation with NLL decomposition."""
    model.eval()
    seqs = [list(p[pkey]) for p in probes]
    for s in seqs: s[CUE_POS] = cue_tok
    st = torch.tensor(seqs, dtype=torch.long, device=dev)
    L = st.size(1) - 1; cm = causal(L, dev)
    M = {k: [] for k in MKEYS}
    with torch.no_grad():
        for i in range(0, len(seqs), 256):
            b = st[i:i+256, :L]; logits = model(b, cm)
            p = F.softmax(logits[:, IS_POS, :], dim=-1)
            lp = F.log_softmax(logits[:, IS_POS, :], dim=-1)
            for j in range(b.size(0)):
                ix = i + j
                ct, cp, a = probes[ix]["ct"], probes[ix]["cp"], probes[ix]["a"]
                # RWT metrics
                rnll = -lp[j, ct].item()
                prw  = p[j, RWT_TOKS].sum().item()
                rfnll = -np.log(max(prw, 1e-30))
                rwnll = -np.log(max(p[j, ct].item() / max(prw, 1e-30), 1e-30))
                rp = p[j, RWT_TOKS]
                t1 = int(rp.argmax().item() == a)
                rk = (rp.argsort(descending=True) == a).nonzero(as_tuple=True)[0].item() + 1
                # SRC metrics
                psa = p[j, cp].item()
                psf = p[j, SRC_TOKS].sum().item()
                swnll = -np.log(max(psa / max(psf, 1e-30), 1e-30))
                cnll = -lp[j, cp].item()
                M["rwt_nll"].append(rnll); M["rwt_fam"].append(prw)
                M["rwt_fnll"].append(rfnll); M["rwt_wnll"].append(rwnll)
                M["rwt_top1"].append(t1); M["rwt_mrr"].append(1./rk)
                M["p_src_a"].append(psa); M["src_fam"].append(psf)
                M["src_wnll"].append(swnll); M["copy_nll"].append(cnll)
                M["decomp"].append(abs(rnll - rfnll - rwnll))
    return {k: round(float(np.mean(v)), 5) for k, v in M.items()}

# ─── Summary computation ──────────────────────────────────────────
ARMS = ["ident_full", "ident_blocked", "unpaired_src"]
REPORT_M = ["rwt_nll", "rwt_fnll", "rwt_wnll", "rwt_mrr", "rwt_top1",
            "p_src_a", "src_fam", "copy_nll"]

def compute_summary(data, seeds):
    S = {}
    for mkey in [k for k in data if k.startswith("cue_")]:
        sm = {}
        for arm in ARMS:
            aa = {}
            for cond in ["T", "U", "NS"]:
                ca = {}
                for m in REPORT_M:
                    vs = [data[mkey][f"seed{s}"][arm]["curves"][-1][cond][m]
                          for s in seeds]
                    ca[m] = dict(mean=round(np.mean(vs), 5),
                                 std=round(np.std(vs), 5))
                aa[cond] = ca
            sm[arm] = aa
        # Source-specific interactions
        ints = {}
        for ia in ["ident_full", "ident_blocked"]:
            dk = f"D_{ia}"
            ints[dk] = {}
            for m in REPORT_M:
                ds = []
                for s in seeds:
                    it = data[mkey][f"seed{s}"][ia]["curves"][-1]["T"][m]
                    iu = data[mkey][f"seed{s}"][ia]["curves"][-1]["U"][m]
                    ut = data[mkey][f"seed{s}"]["unpaired_src"]["curves"][-1]["T"][m]
                    uu = data[mkey][f"seed{s}"]["unpaired_src"]["curves"][-1]["U"][m]
                    ds.append((it - ut) - (iu - uu))
                ints[dk][m] = dict(mean=round(np.mean(ds), 5),
                                   std=round(np.std(ds), 5),
                                   vals=[round(d, 5) for d in ds])
        ints["local_access"] = {}
        for m in REPORT_M:
            df = ints["D_ident_full"][m]["vals"]
            db = ints["D_ident_blocked"][m]["vals"]
            dd = [f - b for f, b in zip(df, db)]
            ints["local_access"][m] = dict(mean=round(np.mean(dd), 5),
                                            std=round(np.std(dd), 5))
        sm["interactions"] = ints
        S[mkey] = sm
    # Task-specification contribution
    if "cue_typed" in S and "cue_uninformative" in S:
        ts = {}
        for ia in ["ident_full", "ident_blocked"]:
            dk = f"D_{ia}"
            ts[dk] = {}
            for m in REPORT_M:
                dt = S["cue_typed"]["interactions"][dk][m]["vals"]
                du = S["cue_uninformative"]["interactions"][dk][m]["vals"]
                dd = [t - u for t, u in zip(dt, du)]
                ts[dk][m] = dict(mean=round(np.mean(dd), 5),
                                  std=round(np.std(dd), 5))
        S["task_spec"] = ts
    return S

def print_summary(S):
    print(f"\n{'='*70}\nSOURCE-TRIGGER SUMMARY\n{'='*70}")
    for mkey in sorted(k for k in S if k.startswith("cue_")):
        print(f"\n=== {mkey} ===")
        for arm in ARMS:
            if arm not in S[mkey]: continue
            a = S[mkey][arm]
            print(f"  {arm:15s}  T:rnll={a['T']['rwt_nll']['mean']:.3f}±{a['T']['rwt_nll']['std']:.3f}"
                  f"  U:rnll={a['U']['rwt_nll']['mean']:.3f}±{a['U']['rwt_nll']['std']:.3f}"
                  f"  T:psrc={a['T']['p_src_a']['mean']:.4f}"
                  f"  U:psrc={a['U']['p_src_a']['mean']:.4f}"
                  f"  T:mrr={a['T']['rwt_mrr']['mean']:.3f}")
        I = S[mkey].get("interactions", {})
        for dk in ["D_ident_full", "D_ident_blocked"]:
            if dk not in I: continue
            print(f"\n  {dk}:")
            for m in ["rwt_nll", "rwt_wnll", "p_src_a", "copy_nll", "rwt_mrr"]:
                if m in I[dk]:
                    print(f"    {m:12s}: {I[dk][m]['mean']:+.5f} ± {I[dk][m]['std']:.5f}")
        la = I.get("local_access", {})
        if la:
            print(f"\n  Local access (FULL - BLOCKED):")
            for m in ["rwt_nll", "rwt_wnll", "p_src_a", "copy_nll", "rwt_mrr"]:
                if m in la:
                    print(f"    {m:12s}: {la[m]['mean']:+.5f} ± {la[m]['std']:.5f}")
    ts = S.get("task_spec", {})
    if ts:
        print(f"\n=== Task specification (TYPED - UNINFORMATIVE) ===")
        for dk in ["D_ident_full", "D_ident_blocked"]:
            if dk not in ts: continue
            print(f"\n  {dk}:")
            for m in ["rwt_nll", "rwt_wnll", "p_src_a", "copy_nll", "rwt_mrr"]:
                if m in ts[dk]:
                    print(f"    {m:12s}: {ts[dk][m]['mean']:+.5f} ± {ts[dk][m]['std']:.5f}")

# ─── Main ──────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 100])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--backbone_n", type=int, default=100)
    ap.add_argument("--sub_n", type=int, default=400)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--nh", type=int, default=2)
    ap.add_argument("--nl", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=str, default="data/source_trigger")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cue-modes", nargs="+", default=["typed", "uninformative"])
    A = ap.parse_args()
    if A.smoke:
        A.seeds = [42]; A.epochs = 20; A.eval_every = 10
        A.cue_modes = ["typed"]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    od = Path(A.out); od.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    all_data = {"config": vars(A)}

    for cm in A.cue_modes:
        mkey = f"cue_{cm}"; all_data[mkey] = {}
        for seed in A.seeds:
            print(f"\n{'='*60}\nSeed {seed}  cue={cm}\n{'='*60}", flush=True)

            # Shared backbone
            dg_bb = DG(seed)
            bb_seqs = dg_bb.backbone(A.backbone_n, cm)

            # Shared sub draws for IDENT and UNPAIRED
            dg_sub = DG(seed * 1000 + 1)
            sub_draws = dg_sub.base_draws(A.sub_n)

            # Arm-specific sequences
            id_seqs = DG.ident_seqs(sub_draws, cm)
            drng = np.random.default_rng(seed * 1000 + 5)
            up_seqs = DG.unpaired_seqs(sub_draws, cm, drng)

            # Probes (same for all arms within seed)
            dg_probe = DG(seed * 2000)
            probes = dg_probe.probes_tu(60)

            # Blocked masks
            L = SL - 1
            bm = build_blocked_masks(sub_draws, A.backbone_n, L)

            # Shared model init
            torch.manual_seed(seed)
            model = CLM(VOCAB, A.d, A.nh, A.nl, SL).to(dev)
            init_sd = {k: v.clone() for k, v in model.state_dict().items()}
            if seed == A.seeds[0] and cm == A.cue_modes[0]:
                print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

            eval_cue = REWRITE_CUE if cm == "typed" else CONST_CUE
            seed_res = {}

            for arm in ARMS:
                t1 = time.time()
                arm_seqs = id_seqs if arm in ("ident_full", "ident_blocked") else up_seqs
                all_seqs = bb_seqs + arm_seqs
                seqs_t = torch.tensor(all_seqs, dtype=torch.long)
                m3d = bm if arm == "ident_blocked" else None

                model.load_state_dict(init_sd)
                opt = torch.optim.AdamW(model.parameters(), lr=A.lr)
                curves = []

                for ep in range(1, A.epochs + 1):
                    tl = train_ep(model, opt, seqs_t, m3d, dev, A.nh)
                    if ep % A.eval_every == 0 or ep == 1 or ep == A.epochs:
                        mt  = eval_probes(model, probes, "s_t",  eval_cue, dev)
                        mu  = eval_probes(model, probes, "s_u",  eval_cue, dev)
                        mns = eval_probes(model, probes, "s_ns", eval_cue, dev)
                        row = dict(e=ep, tl=round(tl, 5), T=mt, U=mu, NS=mns)

                        # Cue swap at final epoch (typed only)
                        if ep == A.epochs and cm == "typed":
                            swaps = {}
                            for sn, stk in [("copy", COPY_CUE), ("const", CONST_CUE)]:
                                swaps[sn] = dict(
                                    T=eval_probes(model, probes, "s_t", stk, dev),
                                    U=eval_probes(model, probes, "s_u", stk, dev))
                            row["cue_swap"] = swaps
                        curves.append(row)

                elapsed = time.time() - t1
                seed_res[arm] = dict(curves=curves, elapsed=round(elapsed, 1))
                ft, fu = curves[-1]["T"], curves[-1]["U"]
                print(f"  {arm:15s}  T_rnll={ft['rwt_nll']:.3f} U_rnll={fu['rwt_nll']:.3f} "
                      f"T_psrc={ft['p_src_a']:.4f} U_psrc={fu['p_src_a']:.4f} "
                      f"T_mrr={ft['rwt_mrr']:.3f} U_mrr={fu['rwt_mrr']:.3f} "
                      f"({elapsed:.1f}s)", flush=True)

                # Save final model for follow-up interventions
                md = od / f"models_{cm}_s{seed}"; md.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), md / f"{arm}.pt")

            all_data[mkey][f"seed{seed}"] = seed_res

    # ─── Summary ───────────────────────────────────────────────────
    summary = compute_summary(all_data, A.seeds)
    all_data["summary"] = summary
    with open(od / "results.json", "w") as f:
        json.dump(all_data, f, indent=2)
    print_summary(summary)

    # Quick decomposition check
    for mkey in [k for k in all_data if k.startswith("cue_")]:
        max_d = max(all_data[mkey][f"seed{s}"][arm]["curves"][-1][c]["decomp"]
                    for s in A.seeds for arm in ARMS for c in ["T", "U", "NS"])
        print(f"  {mkey} max decomp error: {max_d:.6f}")

    print(json.dumps({"status": "DONE", "out": str(od),
                       "elapsed": round(time.time() - t0, 1)}))

if __name__ == "__main__":
    main()
