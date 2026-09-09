#!/usr/bin/env python3
"""research: necessity-style clean-run interventions on the L1 d-component.

research/026 showed that donor d-component patches can redirect answers.  This
script tests the converse on ordinary clean evaluations: if we erase or permute
the natural L1 query-match d-pattern across attribute positions, does behavior
collapse or redirect?

Interventions at layer 1 attribute positions:
  clean       no edit
  center_d    set every attribute slot's d projection to the per-example mean
  zero_d      remove every attribute slot's d projection
  rotate_d    cyclically rotate d projections across the K slots while leaving
              orthogonal components unchanged.  For each example we count both
              natural correctness and whether the answer follows the slot whose
              d projection came from the originally queried slot.

The test is run for preparation, direct_full continuation, static_1over17, and
interleaved_ans_full at 500 continuation epochs (plus smoke shorter runs).  It
separates familiar-query recovery from held-query transfer and avoids using
familiar donor-redirection as the endpoint.
"""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import argparse, copy, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = _public_path('experiments/archive/functional_learning/scripts')
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import query_first_binding as S16
import binding_branching as S17
import budget_matched_full_objective as S18
import revision_019b_embedding_role_decomposition as S19b
import causal_intervention as S25
import causal_interface_trajectory as S26

SL = S25.SL
IS_POS = S25.IS_POS
VOCAB = S25.VOCAB
PAD = S25.PAD
RWT = S25.RWT
ATTR_POS = S25.ATTR_POS
K = S25.K
N_ATTR = S25.N_ATTR
TRAIN_E = S25.TRAIN_E
HELD_E = S25.HELD_E
DEFAULT_PREP = S25.DEFAULT_PREP

ARMS = S26.ARMS


class NpEnc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating, float)):
            return round(float(o), 6)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, torch.Tensor):
            return o.detach().cpu().tolist()
        return super().default(o)


def r6(x):
    return round(float(x), 6)


def make_eval_bank(seed, n, kind):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        if kind == "train":
            ce = rng.choice(TRAIN_E, K, replace=False).tolist()
            ca = rng.choice(N_ATTR, K, replace=False).tolist()
            qi = int(rng.integers(K))
        elif kind == "held_query":
            he = int(rng.choice(HELD_E))
            others = rng.choice(TRAIN_E, K-1, replace=False).tolist()
            ce = list(rng.permutation(others + [he]))
            ca = rng.choice(N_ATTR, K, replace=False).tolist()
            qi = ce.index(he)
        elif kind == "train_in_held_ctx":
            he = int(rng.choice(HELD_E))
            others = rng.choice(TRAIN_E, K-1, replace=False).tolist()
            ce = list(rng.permutation(others + [he]))
            ca = rng.choice(N_ATTR, K, replace=False).tolist()
            hi = ce.index(he)
            qi = int(rng.choice([j for j in range(K) if j != hi]))
        else:
            raise ValueError(kind)
        # The rotated d-pattern moves the original query slot marker to rot_idx.
        rot_idx = (qi + 1) % K
        rows.append(dict(seq=S16.mk_seq("query_first", ce, ca, qi, RWT(ca[qi])),
                         ce=ce, ca=ca, qi=qi, correct=ca[qi], rot_idx=rot_idx,
                         rot_target=ca[rot_idx]))
    return rows


def train_to_epoch(seed, arm_name, epochs, prep_state, cfg, dev, cm, n_train):
    if arm_name == "prep" or epochs == 0:
        return S19b.make_untied_from_tied_state(prep_state, cfg, dev)
    arm = ARMS[arm_name]
    P = DEFAULT_PREP[seed]
    model = S19b.make_untied_from_tied_state(prep_state, cfg, dev)
    opt = torch.optim.AdamW(model.parameters(), lr=arm["lr"], weight_decay=cfg["wd"])
    for be in range(1, epochs + 1):
        rows = S16.make_epoch_rows(seed, P + be, n_train)
        seqs = S16.rows_to_seqs(rows, "query_first", "bound")
        idx = S16.common_order(seed, P + be, n_train)
        cw = S26.ctx_weight_for(arm, be)
        S25.train_epoch(model, opt, seqs, idx, dev, cfg["bs"], cm, "static", cw)
    return model


def forward_intervene(model, inp, mask, direction, mode):
    if mode == "clean":
        model.eval()
        with torch.no_grad():
            return model(inp, mask)
    d = direction.to(inp.device).view(1, 1, -1)
    def hook(mod, inp_arg, out):
        p = out.clone()
        h = p[:, ATTR_POS, :]
        proj = (h * d).sum(-1, keepdim=True) * d
        orth = h - proj
        coeff = (h * d).sum(-1, keepdim=True)
        if mode == "center_d":
            c = coeff.mean(dim=1, keepdim=True)
            p[:, ATTR_POS, :] = orth + c * d
        elif mode == "zero_d":
            p[:, ATTR_POS, :] = orth
        elif mode == "rotate_d":
            # Left roll slots: slot j receives slot j-1's d coefficient, so the
            # original query slot qi moves to (qi+1) modulo K.
            coeff_r = torch.roll(coeff, shifts=1, dims=1)
            p[:, ATTR_POS, :] = orth + coeff_r * d
        else:
            raise ValueError(mode)
        return p
    hk = model.blks[1].register_forward_hook(hook)
    model.eval()
    with torch.no_grad():
        logits = model(inp, mask)
    hk.remove()
    return logits


def eval_interventions(model, rows, direction, dev, cm, bs=128):
    modes = ["clean", "center_d", "zero_d", "rotate_d"]
    out = {m: dict(correct=0, rot=0, correct_nll=[], rot_nll=[], margin_rot_minus_correct=[]) for m in modes}
    # Direction slot stats in the clean hidden states.
    dir_acc = 0; margins = []
    for i in range(0, len(rows), bs):
        batch = rows[i:i+bs]
        inp = torch.tensor([r["seq"] for r in batch], dtype=torch.long, device=dev)[:, :SL-1]
        # clean L1 projections for slot accuracy and margin
        _, stored = S25.forward_extract(model, inp, cm, [1])
        feats = stored[1].cpu()
        d_cpu = direction.cpu().view(1, 1, -1)
        score = (feats * d_cpu).sum(-1)
        labs = torch.tensor([r["qi"] for r in batch], dtype=torch.long)
        pred_slot = score.argmax(1)
        dir_acc += int((pred_slot == labs).sum())
        true = score[torch.arange(score.size(0)), labs]
        tmp = score.clone(); tmp[torch.arange(score.size(0)), labs] = -1e9
        margins.extend((true - tmp.max(1).values).tolist())
        for mode in modes:
            logits = forward_intervene(model, inp, cm, direction, mode)[:, IS_POS].cpu()
            lp = F.log_softmax(logits, dim=-1)
            pred = logits.argmax(1)
            for j, r in enumerate(batch):
                ct = RWT(r["correct"]); rt = RWT(r["rot_target"])
                out[mode]["correct"] += int(int(pred[j]) == ct)
                out[mode]["rot"] += int(int(pred[j]) == rt)
                cn = float(-lp[j, ct]); rn = float(-lp[j, rt])
                out[mode]["correct_nll"].append(cn)
                out[mode]["rot_nll"].append(rn)
                out[mode]["margin_rot_minus_correct"].append(cn - rn)
    N = max(len(rows), 1)
    res = dict(dir_acc=r6(dir_acc / N), dir_margin=r6(np.mean(margins)))
    for mode, v in out.items():
        res[mode] = dict(correct_acc=r6(v["correct"] / N), rot_acc=r6(v["rot"] / N),
                         correct_nll=r6(np.mean(v["correct_nll"])),
                         rot_nll=r6(np.mean(v["rot_nll"])),
                         rot_margin=r6(np.mean(v["margin_rot_minus_correct"])))
        res[mode]["delta_correct_vs_clean"] = None
    c0 = res["clean"]["correct_acc"]
    n0 = res["clean"]["correct_nll"]
    for mode in modes:
        res[mode]["delta_acc_vs_clean"] = r6(res[mode]["correct_acc"] - c0)
        res[mode]["delta_nll_vs_clean"] = r6(res[mode]["correct_nll"] - n0)
    return res


def quick_behavior(model, probes, dev, cm_blk, prep_tok, prep_out):
    pack = S19b.eval_pack(model, probes, dev, cm_blk, prep_tok, prep_out)
    sm = S19b.metric_summary(pack)
    return {k: r6(sm.get(k, 0.0)) for k in ["train_top4", "held_top4", "held_b", "held_sel", "blocked_train_top4"]}


def summarize(data):
    summary = {}
    for arm in data["config"]["arms"]:
        rs = [r for r in data["records"] if r["arm"] == arm]
        if not rs:
            continue
        summary[arm] = {}
        for bank in data["config"]["banks"]:
            br = [r["banks"][bank] for r in rs]
            def mean(path):
                vals = []
                for x in br:
                    cur = x
                    for p in path:
                        cur = cur[p]
                    vals.append(float(cur))
                return r6(np.mean(vals))
            summary[arm][bank] = dict(
                clean=mean(["clean", "correct_acc"]),
                center=mean(["center_d", "correct_acc"]),
                center_drop=mean(["center_d", "delta_acc_vs_clean"]),
                zero=mean(["zero_d", "correct_acc"]),
                zero_drop=mean(["zero_d", "delta_acc_vs_clean"]),
                rotate_correct=mean(["rotate_d", "correct_acc"]),
                rotate_target=mean(["rotate_d", "rot_acc"]),
                rotate_margin=mean(["rotate_d", "rot_margin"]),
                dir_acc=mean(["dir_acc"]),
            )
    data["summary"] = summary
    return summary


def write_note(data, path):
    lines = [
        "# research clean L1 d-component intervention",
        "",
        "## Purpose",
        "",
        "research/026 showed that donor d-component patches can redirect answers. This experiment edits the natural clean forward pass: center or zero the layer-1 d projection across attribute slots, or rotate the d pattern so the original query slot's scalar component moves to a decoy slot. This tests whether ordinary behavior requires the component and whether direct-full familiar recovery uses the same interface.",
        "",
        "## Mean results across seeds",
        "",
    ]
    for arm, banks in data.get("summary", {}).items():
        lines += [f"### {arm}", "", "| bank | clean | center | center drop | zero | zero drop | rotate correct | rotate target | rotate margin | dir acc |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for bank, s in banks.items():
            lines.append(f"| {bank} | {s['clean']:.3f} | {s['center']:.3f} | {s['center_drop']:+.3f} | {s['zero']:.3f} | {s['zero_drop']:+.3f} | {s['rotate_correct']:.3f} | {s['rotate_target']:.3f} | {s['rotate_margin']:+.3f} | {s['dir_acc']:.3f} |")
        lines.append("")
    lines += [
        "## Interpretation",
        "",
        "A large center/zero drop means the natural slot-specific d pattern is needed for the answer. A high rotated-target rate means the answer follows the moved d marker rather than the original query. Comparing `train`, `held_query`, and `train_in_held_ctx` shows whether failure is query-token-specific or due to held-containing contexts. If direct-full at epoch 500 has high train drop/rotation but weak held drop/rotation, familiar recovery is using the interface while held transfer is not.",
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
    ap.add_argument("--data", default="experiments/archive/functional_learning/data/clean_d_component_ablation/results.json")
    ap.add_argument("--note", default="research/notes/functional_learning/clean_d_component_ablation.md")
    ap.add_argument("--seeds", default="43,100")
    ap.add_argument("--arms", default="prep,direct_full,static_1over17,interleaved_ans_full")
    ap.add_argument("--cont-epochs", type=int, default=500)
    ap.add_argument("--n-train", type=int, default=500)
    ap.add_argument("--n-bank", type=int, default=384)
    ap.add_argument("--n-dir", type=int, default=1000)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.seeds = "100"; args.arms = "prep,direct_full,static_1over17"; args.cont_epochs = 10
        args.n_train = 64; args.n_bank = 80; args.n_dir = 250
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    arms = [x.strip() for x in args.arms.split(",") if x.strip()]
    banks = ["train", "held_query", "train_in_held_ctx"]
    cfg = dict(d=64, nh=2, nl=3, wd=0.01, bs=64)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cm = S17.standard_causal_mask(SL-1, dev)
    cm_blk = S17.block_query_ctx_mask(SL-1, dev)
    probes = S18.make_probes(seed=180018, n_std=256 if not args.smoke else 96, n_small=128 if not args.smoke else 48)
    ckpt_dir = Path(args.checkpoint_dir)
    eval_banks = {b: make_eval_bank(270000 + i, args.n_bank, b) for i, b in enumerate(banks)}
    data = dict(config=dict(seeds=seeds, arms=arms, cont_epochs=args.cont_epochs,
                            n_train=args.n_train, n_bank=args.n_bank, n_dir=args.n_dir, banks=banks),
                paths=dict(script="experiments/archive/functional_learning/scripts/clean_d_component_ablation.py",
                           data=args.data, note=args.note),
                records=[], summary={})
    t0 = time.time()
    print(f"Device: {dev}; seeds={seeds}; arms={arms}; cont_epochs={args.cont_epochs}", flush=True)
    for sd in seeds:
        prep_state = torch.load(ckpt_dir / f"seed{sd}_prep_tied.pt", map_location=dev)
        prep_tok = prep_state["tok.weight"].detach().clone().to(dev)
        prep_model = S19b.make_untied_from_tied_state(prep_state, cfg, dev)
        prep_out = prep_model.out.weight.detach().clone()
        prep_d, prep_dir = S26.learn_l1_direction(prep_model, dev, cm, seed=370000 + sd,
                                                  n_train=args.n_dir, n_test=max(120, args.n_dir//3))
        print("\n" + "="*80)
        print(f"seed={sd}; prep_dir_test={prep_dir['test']['acc']:.3f}/{prep_dir['test']['margin']:+.3f}")
        print("="*80, flush=True)
        for arm in arms:
            model = train_to_epoch(sd, arm, 0 if arm == "prep" else args.cont_epochs,
                                   prep_state, cfg, dev, cm, args.n_train)
            rec = dict(seed=sd, arm=arm, branch_epoch=0 if arm == "prep" else args.cont_epochs,
                       behavior=quick_behavior(model, probes, dev, cm_blk, prep_tok, prep_out), banks={})
            for bank_name, rows in eval_banks.items():
                rec["banks"][bank_name] = eval_interventions(model, rows, prep_d, dev, cm)
            data["records"].append(rec)
            b = rec["behavior"]
            train = rec["banks"]["train"]; held = rec["banks"]["held_query"]; tih = rec["banks"]["train_in_held_ctx"]
            print(f"  {arm:<22s} tr4={b['train_top4']:.3f} h4={b['held_top4']:.3f} "
                  f"train clean/center/rotT={train['clean']['correct_acc']:.3f}/{train['center_d']['correct_acc']:.3f}/{train['rotate_d']['rot_acc']:.3f} "
                  f"held={held['clean']['correct_acc']:.3f}/{held['center_d']['correct_acc']:.3f}/{held['rotate_d']['rot_acc']:.3f} "
                  f"trainHeld={tih['clean']['correct_acc']:.3f}/{tih['center_d']['correct_acc']:.3f}/{tih['rotate_d']['rot_acc']:.3f}", flush=True)
    summarize(data)
    outp = Path(args.data); outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(data, indent=2, cls=NpEnc))
    write_note(data, args.note)
    print(json.dumps(dict(status="ok", out=args.data, note=args.note, elapsed=round(time.time() - t0, 1)), indent=2))


if __name__ == "__main__":
    main()
