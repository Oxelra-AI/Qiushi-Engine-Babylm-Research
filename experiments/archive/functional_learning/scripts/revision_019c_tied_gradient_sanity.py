#!/usr/bin/env python3
"""Step019c: exact gradient sanity check for tied input/output rows.

This is a small, independent check while the long Step019b causal run is pending.
It verifies the implementation-level fact behind the embedding-role hypothesis:
held entity tokens that never occur as continuation inputs or correct targets still
receive full-objective gradients when the input embedding matrix is tied to the
softmax output classifier. In an untied copy the same gradient goes to out.weight,
while the held input rows have zero data gradient (apart from AdamW decay during a
step).
"""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json, math, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = _public_path('experiments/archive/functional_learning/scripts')
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import loss_allocation_binding as Base
import query_first_binding as S16
import binding_branching as S17
import revision_019b_embedding_role_decomposition as S19b

K, N_ATTR, VOCAB = Base.K, Base.N_ATTR, Base.VOCAB
TRAIN_E, HELD_E = Base.TRAIN_E, Base.HELD_E
ENT, RWT = Base.ENT, Base.RWT
SL, IS_POS = 18, 15
HELD_ROWS = [ENT(e) for e in HELD_E]


def model_features_tied(model, x, mask):
    B, L = x.shape
    h = model.tok(x) + model.pos(torch.arange(L, device=x.device))
    for b in model.blks:
        h = b(h, mask)
    return model.ln(h)


def model_features_untied(model, x, mask):
    B, L = x.shape
    h = model.tok(x) + model.pos(torch.arange(L, device=x.device))
    for b in model.blks:
        h = b(h, mask)
    return model.ln(h)


def weighted_full_loss_and_manual_output_grad(model, x, tgt, mask, rows):
    """Return full-objective loss and manual non-target output gradient for rows."""
    feats = model_features_tied(model, x, mask)
    logits = feats @ model.tok.weight.T
    ce = F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1), reduction="none").view(tgt.shape)
    nonpad = (tgt != Base.PAD).float()
    pos_w = S17.weight_vec("full", 1.0, x.device)
    wt = nonpad * pos_w.view(1, -1)
    denom = wt.sum().clamp_min(1.0)
    loss = (ce * wt).sum() / denom
    p = F.softmax(logits, dim=-1)
    manual = []
    for r in rows:
        # Held entity rows are never targets in this sampled batch; if that invariant
        # fails, the formula below is no longer output-only non-target pressure.
        g = (p[:, :, r].unsqueeze(-1) * feats * wt.unsqueeze(-1)).sum(dim=(0, 1)) / denom
        manual.append(g)
    return loss, torch.stack(manual, dim=0), logits.detach(), wt.detach(), feats.detach()


def weighted_full_loss_untied(model, x, tgt, mask):
    logits = model(x, mask)
    ce = F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1), reduction="none").view(tgt.shape)
    nonpad = (tgt != Base.PAD).float()
    pos_w = S17.weight_vec("full", 1.0, x.device)
    wt = nonpad * pos_w.view(1, -1)
    denom = wt.sum().clamp_min(1.0)
    return (ce * wt).sum() / denom


def held_absence(seqs):
    arr = np.array(seqs, dtype=np.int64)
    x = arr[:, :-1]
    tgt = arr[:, 1:]
    held = set(HELD_ROWS)
    return {
        "held_input_count": int(sum(int(v in held) for v in x.reshape(-1).tolist())),
        "held_target_count": int(sum(int(v in held) for v in tgt.reshape(-1).tolist())),
        "held_rows": HELD_ROWS,
    }


def adamw_one_step_delta(model, loss_fn, rows, lr=3e-4, wd=0.0):
    rows_t = torch.tensor(rows, device=next(model.parameters()).device, dtype=torch.long)
    before = model.tok.weight.detach()[rows_t].clone()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss = loss_fn(model)
    opt.zero_grad(); loss.backward(); opt.step()
    after = model.tok.weight.detach()[rows_t].clone()
    return torch.linalg.norm(after - before, dim=1).detach().cpu().tolist(), torch.linalg.norm(before, dim=1).detach().cpu().tolist()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/archive/functional_learning/data/revision_019c_tied_gradient_sanity/results.json")
    ap.add_argument("--note", default="research/notes/functional_learning/019c_tied_gradient_sanity.md")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--epoch", type=int, default=401)
    ap.add_argument("--n", type=int, default=128)
    A = ap.parse_args()

    torch.manual_seed(A.seed)
    np.random.seed(A.seed)
    dev = torch.device("cpu")
    cfg = dict(d=64, nh=2, nl=3, lr=3e-4, wd=0.01, bs=64)
    cm = S17.standard_causal_mask(SL - 1, dev)

    rows = S16.make_epoch_rows(A.seed, A.epoch, A.n)
    seqs = S16.rows_to_seqs(rows, "query_first", "bound")
    idx = S16.common_order(A.seed, A.epoch, A.n)
    seqs = np.asarray(seqs, dtype=np.int64)[idx]
    x = torch.tensor(seqs[:, :-1], dtype=torch.long, device=dev)
    tgt = torch.tensor(seqs[:, 1:], dtype=torch.long, device=dev)
    held_rows_t = torch.tensor(HELD_ROWS, device=dev, dtype=torch.long)

    absence = held_absence(seqs)
    tied = Base.CLM(VOCAB, cfg["d"], cfg["nh"], cfg["nl"], SL).to(dev)
    tied.eval()
    loss, manual, logits, wt, feats = weighted_full_loss_and_manual_output_grad(tied, x, tgt, cm, HELD_ROWS)
    tied.zero_grad(set_to_none=True)
    loss.backward()
    autograd_tied = tied.tok.weight.grad[held_rows_t].detach().clone()
    manual_err = (autograd_tied - manual).abs().max().item()

    untied = S19b.make_untied_from_tied_state(tied.state_dict(), cfg, dev)
    untied.eval()
    loss_u = weighted_full_loss_untied(untied, x, tgt, cm)
    untied.zero_grad(set_to_none=True)
    loss_u.backward()
    untied_input_grad = untied.tok.weight.grad[held_rows_t].detach().clone()
    untied_output_grad = untied.out.weight.grad[held_rows_t].detach().clone()
    output_match_err = (untied_output_grad - autograd_tied).abs().max().item()

    # One-step displacement checks, using identical initial tied state.  With wd=0,
    # an untied held input row should not move because it has no data gradient.
    state = tied.state_dict()
    def make_tied_clone():
        m = Base.CLM(VOCAB, cfg["d"], cfg["nh"], cfg["nl"], SL).to(dev); m.load_state_dict(state); return m
    def make_untied_clone():
        return S19b.make_untied_from_tied_state(state, cfg, dev)
    def loss_t(m):
        return weighted_full_loss_and_manual_output_grad(m, x, tgt, cm, HELD_ROWS)[0]
    def loss_u_fn(m):
        return weighted_full_loss_untied(m, x, tgt, cm)

    tied_delta_wd0, tied_norms = adamw_one_step_delta(make_tied_clone(), loss_t, HELD_ROWS, lr=cfg["lr"], wd=0.0)
    untied_delta_wd0, untied_norms = adamw_one_step_delta(make_untied_clone(), loss_u_fn, HELD_ROWS, lr=cfg["lr"], wd=0.0)
    tied_delta_wd, _ = adamw_one_step_delta(make_tied_clone(), loss_t, HELD_ROWS, lr=cfg["lr"], wd=cfg["wd"])
    untied_delta_wd, _ = adamw_one_step_delta(make_untied_clone(), loss_u_fn, HELD_ROWS, lr=cfg["lr"], wd=cfg["wd"])
    expected_decay_delta = [cfg["lr"] * cfg["wd"] * n for n in untied_norms]

    # Category gradient comparison for scale: held output rows vs trained entity and RWT rows.
    tied.zero_grad(set_to_none=True)
    loss, _, _, _, _ = weighted_full_loss_and_manual_output_grad(tied, x, tgt, cm, HELD_ROWS)
    loss.backward()
    grad = tied.tok.weight.grad.detach()
    cat = {
        "held_ent_grad_l2_mean": float(torch.linalg.norm(grad[held_rows_t], dim=1).mean().item()),
        "train_ent_grad_l2_mean": float(torch.linalg.norm(grad[torch.tensor([ENT(e) for e in TRAIN_E], device=dev)], dim=1).mean().item()),
        "rwt_grad_l2_mean": float(torch.linalg.norm(grad[torch.tensor([RWT(a) for a in range(N_ATTR)], device=dev)], dim=1).mean().item()),
    }

    result = {
        "config": {"seed": A.seed, "epoch": A.epoch, "n": A.n, **cfg},
        "absence": absence,
        "loss_tied": float(loss.item()),
        "loss_untied_copy": float(loss_u.item()),
        "manual_tied_grad_max_abs_err": manual_err,
        "untied_output_vs_tied_grad_max_abs_err": output_match_err,
        "tied_held_grad_l2": torch.linalg.norm(autograd_tied, dim=1).detach().cpu().tolist(),
        "manual_held_grad_l2": torch.linalg.norm(manual, dim=1).detach().cpu().tolist(),
        "untied_held_input_grad_l2": torch.linalg.norm(untied_input_grad, dim=1).detach().cpu().tolist(),
        "untied_held_output_grad_l2": torch.linalg.norm(untied_output_grad, dim=1).detach().cpu().tolist(),
        "adamw_one_step_input_delta": {
            "tied_wd0": tied_delta_wd0,
            "untied_wd0": untied_delta_wd0,
            "tied_wd001": tied_delta_wd,
            "untied_wd001": untied_delta_wd,
            "expected_untied_decay_only_delta": expected_decay_delta,
            "held_row_norms": tied_norms,
        },
        "category_grad_l2": cat,
    }
    outp = Path(A.out); outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(result, indent=2))

    lines = []
    lines.append("# Step019c tied-gradient sanity check")
    lines.append("")
    lines.append(f"Sampled query-first bound batch: seed={A.seed}, epoch={A.epoch}, n={A.n}.")
    lines.append(f"Held entity token rows: {HELD_ROWS}; held input count={absence['held_input_count']}, held target count={absence['held_target_count']}.")
    lines.append("")
    lines.append("## Main implementation facts")
    lines.append("")
    lines.append(f"- Tied held-row gradient L2: {[round(x, 8) for x in result['tied_held_grad_l2']]}")
    lines.append(f"- Manual non-target softmax gradient matches tied autograd: max |diff|={manual_err:.3e}.")
    lines.append(f"- Untied held input gradient L2: {[round(x, 12) for x in result['untied_held_input_grad_l2']]}")
    lines.append(f"- Untied held output gradient L2: {[round(x, 8) for x in result['untied_held_output_grad_l2']]}; matches tied held-row gradient with max |diff|={output_match_err:.3e}.")
    lines.append("")
    lines.append("Thus, in this implementation, a held entity absent from the continuation input and target stream is still trained as a negative output class when embeddings are tied. Untying routes that same gradient to `out.weight` and leaves the held input embedding with zero data gradient.")
    lines.append("")
    lines.append("## One AdamW step")
    lines.append("")
    lines.append(f"- wd=0 tied held input displacement: {[round(x, 8) for x in tied_delta_wd0]}; untied held input displacement: {[round(x, 12) for x in untied_delta_wd0]}.")
    lines.append(f"- wd=0.01 tied held input displacement: {[round(x, 8) for x in tied_delta_wd]}; untied held input displacement: {[round(x, 8) for x in untied_delta_wd]}.")
    lines.append(f"- Expected untied decay-only displacement lr*wd*||row||: {[round(x, 8) for x in expected_decay_delta]}.")
    lines.append("")
    lines.append("This does not by itself show that row drift causes held-transfer loss; Step019b must establish that by row restoration/freezing and hybrid evaluations. It does verify the concrete gradient path that makes the hypothesis possible.")
    lines.append("")
    lines.append(f"Full JSON: `{outp}`")
    note = Path(A.note); note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "ok", "out": str(outp), "note": str(note), "manual_err": manual_err, "output_match_err": output_match_err}, indent=2))


if __name__ == "__main__":
    main()
