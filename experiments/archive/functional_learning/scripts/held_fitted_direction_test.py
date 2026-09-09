#!/usr/bin/env python3
"""research: is held transfer lost or merely rotated into a held-specific direction?

research/027 used the preparation-learned direction.  This script tests a key
alternative: after direct-full continuation, held-query selection might still be
present but encoded in a new direction not aligned with the preparation/train
interface.

For each endpoint model we fit L1 matched-minus-unmatched directions from:
  - train: trained-query contexts
  - held_one: contexts with exactly one held entity, querying that held entity
  - two_held: contexts with both held entities present, querying one of them

Then we test each direction on slot classification and donor-query d-only
redirection for train, one-held-vs-train, and two-held-vs-held pairs.  The two-held
case is important: it distinguishes query-conditioned selection among held tokens
from simply marking the unique unseen/held slot.
"""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = _public_path('experiments/archive/functional_learning/scripts')
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import query_first_binding as S16
import binding_branching as S17
import budget_matched_full_objective as S18
import revision_019b_embedding_role_decomposition as S19b
import causal_intervention as S25
import causal_interface_trajectory as S26

SL = S25.SL; IS_POS = S25.IS_POS; PAD = S25.PAD; RWT = S25.RWT
K = S25.K; N_ATTR = S25.N_ATTR; TRAIN_E = S25.TRAIN_E; HELD_E = S25.HELD_E
DEFAULT_PREP = S25.DEFAULT_PREP
ARMS = S26.ARMS


class NpEnc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating, float)): return round(float(o), 6)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, torch.Tensor): return o.detach().cpu().tolist()
        return super().default(o)


def r6(x): return round(float(x), 6)


def gen_dir_bank(seed, n, kind):
    rng = np.random.default_rng(seed)
    seqs, labels = [], []
    for _ in range(n):
        if kind == "train":
            ce = rng.choice(TRAIN_E, K, replace=False).tolist()
            qi = int(rng.integers(K))
        elif kind == "held_one":
            he = int(rng.choice(HELD_E))
            others = rng.choice(TRAIN_E, K-1, replace=False).tolist()
            ce = list(rng.permutation(others + [he]))
            qi = ce.index(he)
        elif kind == "two_held":
            others = rng.choice(TRAIN_E, K-2, replace=False).tolist()
            ce = list(rng.permutation(list(HELD_E) + others))
            held_slots = [i for i, e in enumerate(ce) if e in HELD_E]
            qi = int(rng.choice(held_slots))
        elif kind == "train_in_twoheld":
            others = rng.choice(TRAIN_E, K-2, replace=False).tolist()
            ce = list(rng.permutation(list(HELD_E) + others))
            train_slots = [i for i, e in enumerate(ce) if e in TRAIN_E]
            qi = int(rng.choice(train_slots))
        else:
            raise ValueError(kind)
        ca = rng.choice(N_ATTR, K, replace=False).tolist()
        seqs.append(S16.mk_seq("query_first", ce, ca, qi, PAD))
        labels.append(qi)
    return seqs, torch.tensor(labels, dtype=torch.long)


def gen_two_held_pairs(seed, n, donor_kind="held_vs_held"):
    rng = np.random.default_rng(seed)
    pairs = []
    for _ in range(n):
        others = rng.choice(TRAIN_E, K-2, replace=False).tolist()
        ce = list(rng.permutation(list(HELD_E) + others))
        ca = rng.choice(N_ATTR, K, replace=False).tolist()
        held_slots = [i for i, e in enumerate(ce) if e in HELD_E]
        train_slots = [i for i, e in enumerate(ce) if e in TRAIN_E]
        if donor_kind == "held_vs_held":
            qi_a = int(rng.choice(held_slots))
            qi_b = held_slots[0] if held_slots[1] == qi_a else held_slots[1]
        elif donor_kind == "held_vs_train":
            qi_a = int(rng.choice(held_slots))
            qi_b = int(rng.choice(train_slots))
        elif donor_kind == "train_vs_held":
            qi_a = int(rng.choice(train_slots))
            qi_b = int(rng.choice(held_slots))
        else:
            raise ValueError(donor_kind)
        pairs.append(dict(seq_a=S16.mk_seq("query_first", ce, ca, qi_a, PAD),
                          seq_b=S16.mk_seq("query_first", ce, ca, qi_b, PAD),
                          target_a=ca[qi_a], target_b=ca[qi_b]))
    return pairs


def train_to_endpoint(seed, arm, epochs, prep_state, cfg, dev, cm, n_train):
    if arm == "prep":
        return S19b.make_untied_from_tied_state(prep_state, cfg, dev)
    model = S19b.make_untied_from_tied_state(prep_state, cfg, dev)
    opt = torch.optim.AdamW(model.parameters(), lr=ARMS[arm]["lr"], weight_decay=cfg["wd"])
    P = DEFAULT_PREP[seed]
    for be in range(1, epochs + 1):
        rows = S16.make_epoch_rows(seed, P + be, n_train)
        seqs = S16.rows_to_seqs(rows, "query_first", "bound")
        idx = S16.common_order(seed, P + be, n_train)
        cw = S26.ctx_weight_for(ARMS[arm], be)
        S25.train_epoch(model, opt, seqs, idx, dev, cfg["bs"], cm, "static", cw)
    return model


@torch.no_grad()
def extract_l1(model, seqs, dev, cm, bs=128):
    fs = []
    for i in range(0, len(seqs), bs):
        inp = torch.tensor(seqs[i:i+bs], dtype=torch.long, device=dev)[:, :SL-1]
        _, stored = S25.forward_extract(model, inp, cm, [1])
        fs.append(stored[1].cpu())
    return torch.cat(fs, 0)


def fit_direction(model, source_kind, dev, cm, seed, n_fit, n_test):
    tr_s, tr_l = gen_dir_bank(seed, n_fit, source_kind)
    te_s, te_l = gen_dir_bank(seed + 1, n_test, source_kind)
    tr = extract_l1(model, tr_s, dev, cm)
    te = extract_l1(model, te_s, dev, cm)
    matched = torch.stack([tr[i, int(tr_l[i])] for i in range(tr.size(0))])
    unmatched = torch.stack([tr[i, j] for i in range(tr.size(0)) for j in range(K) if j != int(tr_l[i])])
    d = matched.mean(0) - unmatched.mean(0)
    d = d / d.norm().clamp_min(1e-8)
    def score(feats, lab):
        s = (feats * d.view(1, 1, -1)).sum(-1)
        pred = s.argmax(1)
        true = s[torch.arange(s.size(0)), lab]
        tmp = s.clone(); tmp[torch.arange(s.size(0)), lab] = -1e9
        return dict(acc=r6((pred == lab).float().mean()), margin=r6((true - tmp.max(1).values).mean()))
    return d, dict(fit=score(tr, tr_l), source_test=score(te, te_l))


def eval_direction_on_banks(model, d, banks, dev, cm):
    out = {}
    for name, (seqs, lab) in banks.items():
        feats = extract_l1(model, seqs, dev, cm)
        s = (feats * d.view(1, 1, -1)).sum(-1)
        pred = s.argmax(1)
        true = s[torch.arange(s.size(0)), lab]
        tmp = s.clone(); tmp[torch.arange(s.size(0)), lab] = -1e9
        out[name] = dict(acc=r6((pred == lab).float().mean()), margin=r6((true - tmp.max(1).values).mean()))
    return out


def eval_direction_patch(model, d, pairs_by_case, dev, cm):
    out = {}
    for cname, pairs in pairs_by_case.items():
        donor, clean_a = S26.extract_donor_l1(model, pairs, dev, cm)
        out[cname] = dict(
            clean_a=clean_a,
            clean_b=S26.clean_b_acc(model, pairs, dev, cm),
            d_only=S26.patch_eval(model, pairs, donor, dev, cm, direction=d, mode="d_only"),
            orth=S26.patch_eval(model, pairs, donor, dev, cm, direction=d, mode="orth_only"),
            full=S26.patch_eval(model, pairs, donor, dev, cm, mode="full"),
        )
    return out


def direction_cosines(dirs):
    keys = list(dirs.keys())
    out = {}
    for i, a in enumerate(keys):
        for b in keys[i+1:]:
            out[f"{a}__{b}"] = r6(torch.dot(dirs[a].cpu(), dirs[b].cpu()).item())
    return out


def quick_behavior(model, probes, dev, cm_blk, prep_tok, prep_out):
    pack = S19b.eval_pack(model, probes, dev, cm_blk, prep_tok, prep_out)
    sm = S19b.metric_summary(pack)
    return {k: r6(sm.get(k, 0.0)) for k in ["train_top4", "held_top4", "held_b", "held_sel", "blocked_train_top4"]}


def summarize(data):
    out = {}
    for arm in data["config"]["arms"]:
        out[arm] = {}
        rs = [r for r in data["records"] if r["arm"] == arm]
        if not rs: continue
        for dsrc in data["config"]["direction_sources"]:
            def mean(fn): return r6(np.mean([fn(r) for r in rs]))
            out[arm][dsrc] = dict(
                train_acc=mean(lambda r: r["directions"][dsrc]["bank_eval"]["train"]["acc"]),
                held_one_acc=mean(lambda r: r["directions"][dsrc]["bank_eval"]["held_one"]["acc"]),
                two_held_acc=mean(lambda r: r["directions"][dsrc]["bank_eval"]["two_held"]["acc"]),
                redir_train=mean(lambda r: r["directions"][dsrc]["patch_eval"]["train_pairs"]["d_only"]["redirect"]),
                redir_held_one=mean(lambda r: r["directions"][dsrc]["patch_eval"]["oneheld_held_vs_train"]["d_only"]["redirect"]),
                redir_two_held=mean(lambda r: r["directions"][dsrc]["patch_eval"]["twoheld_held_vs_held"]["d_only"]["redirect"]),
                orth_two_held=mean(lambda r: r["directions"][dsrc]["patch_eval"]["twoheld_held_vs_held"]["orth"]["redirect"]),
            )
    data["summary"] = out
    return out


def write_note(data, path):
    lines = [
        "# research held-fitted direction test",
        "",
        "## Purpose",
        "",
        "This tests whether direct-full held failure is only a rotation away from the preparation/train direction. Directions are fitted from train, one-held-query, or two-held-query L1 states in the current endpoint model, then evaluated by classification and d-only donor-query redirection. The two-held condition separates query-conditioned selection among held tokens from simply finding the unique held slot.",
        "",
        "## Mean results across seeds",
        "",
    ]
    for arm, bysrc in data.get("summary", {}).items():
        lines += [f"### {arm}", "", "| fitted direction | cls train | cls one-held | cls two-held | redir train | redir one-held | redir two-held | orth two-held |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for src, s in bysrc.items():
            lines.append(f"| {src} | {s['train_acc']:.3f} | {s['held_one_acc']:.3f} | {s['two_held_acc']:.3f} | {s['redir_train']:.3f} | {s['redir_held_one']:.3f} | {s['redir_two_held']:.3f} | {s['orth_two_held']:.3f} |")
        lines.append("")
    lines += [
        "## Interpretation",
        "",
        "If direct-full held failure were only a hidden rotation, a direction fitted from held or two-held states should recover high held/two-held redirection even when the train/preparation direction fails. If held-fitted directions do not rescue two-held redirection while train redirection is high, the endpoint has narrowed the functional interface to trained query symbols rather than preserving a rotated held selector.",
        "",
        "## Files",
        "",
        f"- Data: `{data['paths']['data']}`",
        f"- Script: `{data['paths']['script']}`",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", default="experiments/archive/functional_learning/data/revision_019b_embedding_role_decomposition/checkpoints")
    ap.add_argument("--data", default="experiments/archive/functional_learning/data/held_fitted_direction_test/results.json")
    ap.add_argument("--note", default="research/notes/functional_learning/held_fitted_direction_test.md")
    ap.add_argument("--seeds", default="43,100")
    ap.add_argument("--arms", default="prep,direct_full,static_1over17,interleaved_ans_full")
    ap.add_argument("--cont-epochs", type=int, default=500)
    ap.add_argument("--n-train", type=int, default=500)
    ap.add_argument("--n-fit", type=int, default=900)
    ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--n-pairs", type=int, default=256)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.seeds = "100"; args.arms = "prep,direct_full"; args.cont_epochs = 10
        args.n_train = 64; args.n_fit = 200; args.n_test = 80; args.n_pairs = 80
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    arms = [x.strip() for x in args.arms.split(",") if x.strip()]
    direction_sources = ["train", "held_one", "two_held"]
    bank_names = ["train", "held_one", "two_held", "train_in_twoheld"]
    cfg = dict(d=64, nh=2, nl=3, wd=0.01, bs=64)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cm = S17.standard_causal_mask(SL-1, dev)
    cm_blk = S17.block_query_ctx_mask(SL-1, dev)
    probes = S18.make_probes(seed=180018, n_std=256 if not args.smoke else 96, n_small=128 if not args.smoke else 48)
    ckpt_dir = Path(args.checkpoint_dir)
    eval_banks = {name: gen_dir_bank(280000 + i, args.n_test, name) for i, name in enumerate(bank_names)}
    oneheld = S25.gen_held_pairs(280100, args.n_pairs)
    pairs_by_case = {
        "train_pairs": S25.gen_train_pairs(280101, args.n_pairs),
        "oneheld_held_vs_train": oneheld["held_donor"],
        "oneheld_train_vs_held": oneheld["train_donor"],
        "twoheld_held_vs_held": gen_two_held_pairs(280102, args.n_pairs, "held_vs_held"),
        "twoheld_held_vs_train": gen_two_held_pairs(280103, args.n_pairs, "held_vs_train"),
        "twoheld_train_vs_held": gen_two_held_pairs(280104, args.n_pairs, "train_vs_held"),
    }
    data = dict(config=dict(seeds=seeds, arms=arms, cont_epochs=args.cont_epochs, n_fit=args.n_fit,
                            n_test=args.n_test, n_pairs=args.n_pairs, direction_sources=direction_sources,
                            banks=bank_names, cases=list(pairs_by_case.keys())),
                paths=dict(script="experiments/archive/functional_learning/scripts/held_fitted_direction_test.py",
                           data=args.data, note=args.note),
                records=[], summary={})
    t0 = time.time()
    print(f"Device: {dev}; seeds={seeds}; arms={arms}; cont_epochs={args.cont_epochs}", flush=True)
    for sd in seeds:
        prep_state = torch.load(ckpt_dir / f"seed{sd}_prep_tied.pt", map_location=dev)
        prep_tok = prep_state["tok.weight"].detach().clone().to(dev)
        prep_base = S19b.make_untied_from_tied_state(prep_state, cfg, dev)
        prep_out = prep_base.out.weight.detach().clone()
        print("\n" + "="*88)
        print(f"seed={sd}")
        print("="*88, flush=True)
        for arm in arms:
            if arm != "prep" and arm not in ARMS:
                raise ValueError(arm)
            model = train_to_endpoint(sd, arm, args.cont_epochs, prep_state, cfg, dev, cm, args.n_train)
            rec = dict(seed=sd, arm=arm, branch_epoch=0 if arm == "prep" else args.cont_epochs,
                       behavior=quick_behavior(model, probes, dev, cm_blk, prep_tok, prep_out),
                       directions={}, direction_cosines={})
            dirs = {}
            for src in direction_sources:
                dvec, fit_info = fit_direction(model, src, dev, cm, seed=281000 + sd*37 + direction_sources.index(src)*101,
                                               n_fit=args.n_fit, n_test=args.n_test)
                dirs[src] = dvec
                rec["directions"][src] = dict(fit=fit_info,
                                               bank_eval=eval_direction_on_banks(model, dvec, eval_banks, dev, cm),
                                               patch_eval=eval_direction_patch(model, dvec, pairs_by_case, dev, cm))
            rec["direction_cosines"] = direction_cosines(dirs)
            data["records"].append(rec)
            b = rec["behavior"]
            print(f"  {arm:<22s} tr4={b['train_top4']:.3f} h4={b['held_top4']:.3f} hB={b['held_b']:+.2f}")
            for src in direction_sources:
                de = rec["directions"][src]
                print(f"    dir {src:<9s} cls train/one/two={de['bank_eval']['train']['acc']:.3f}/{de['bank_eval']['held_one']['acc']:.3f}/{de['bank_eval']['two_held']['acc']:.3f} "
                      f"redir train/one/two={de['patch_eval']['train_pairs']['d_only']['redirect']:.3f}/{de['patch_eval']['oneheld_held_vs_train']['d_only']['redirect']:.3f}/{de['patch_eval']['twoheld_held_vs_held']['d_only']['redirect']:.3f}", flush=True)
    summarize(data)
    outp = Path(args.data); outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(data, indent=2, cls=NpEnc))
    write_note(data, args.note)
    print(json.dumps(dict(status="ok", out=args.data, note=args.note, elapsed=round(time.time() - t0, 1)), indent=2))


if __name__ == "__main__":
    main()
