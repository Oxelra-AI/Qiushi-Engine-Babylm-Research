#!/usr/bin/env python3
"""Step025b: per-model direction learning.

Tests whether collapsed models have reorganized their query-match signal into
a different direction, rather than losing it entirely.  Learns a fresh direction
from each model's own activations and tests within-model donor-query redirection
using that model-specific direction.
"""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json, sys, time
from pathlib import Path
import numpy as np, torch

SCRIPT_DIR = _public_path('experiments/archive/functional_learning/scripts')
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import causal_intervention as S25

SL = S25.SL; IS_POS = S25.IS_POS; ATTR_POS = S25.ATTR_POS
DEFAULT_PREP = S25.DEFAULT_PREP

def main():
    import binding_branching as S17
    import budget_matched_full_objective as S18
    import revision_019b_embedding_role_decomposition as S19b

    seeds = [43, 100]
    cfg = dict(d=64, nh=2, nl=3, lr=3e-4, wd=0.01, bs=64)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cm = S17.standard_causal_mask(SL-1, dev)
    cm_blk = S17.block_query_ctx_mask(SL-1, dev)
    probes = S18.make_probes(seed=180018, n_std=256, n_small=128)
    ckpt_dir = Path('experiments/archive/functional_learning/data'
                    'revision_019b_embedding_role_decomposition/checkpoints')
    layers = [0, 1, 2]
    train_pairs = S25.gen_train_pairs(250350, 500)
    held_pairs = S25.gen_held_pairs(250360, 200)

    results = {}; t0 = time.time()
    for sd in seeds:
        P = DEFAULT_PREP[sd]
        tied = torch.load(ckpt_dir / f'seed{sd}_prep_tied.pt', map_location=dev)
        prep_tok = tied['tok.weight'].detach().clone().to(dev)
        m_prep = S19b.make_untied_from_tied_state(tied, cfg, dev)
        prep_out = m_prep.out.weight.detach().clone()

        # Learn prep direction (reference)
        dirs_prep, _, prep_te = S25.learn_directions(m_prep, dev, cm,
            seed=250250+sd, n_train=1500, n_test=500)

        models = {'prep': m_prep}
        for arm_name in ['direct_full', 'static_1over17']:
            models[arm_name] = S25.train_continuation(
                tied, sd, P, arm_name, cfg, dev, cm, 500, 100)

        sd_res = {}
        for mname, model in models.items():
            beh = S25.quick_behavior(model, probes, dev, cm_blk, prep_tok, prep_out)
            # Learn this model's own direction
            own_dirs, _, own_te = S25.learn_directions(model, dev, cm,
                seed=260260+sd, n_train=1500, n_test=500)
            # Test with prep direction
            prep_acc = S25.test_direction(model, dirs_prep, dev, cm,
                seed=250251+sd, n=500)
            # Run intervention with own direction
            iv_own, _ = S25._eval_pairs(model, train_pairs, own_dirs, dev, cm,
                                        layers, ('full', 'd_only', 'orth_only'))
            # Run intervention with prep direction (same as research)
            iv_prep, _ = S25._eval_pairs(model, train_pairs, dirs_prep, dev, cm,
                                         layers, ('d_only',))
            
            sd_res[mname] = dict(
                behavior=beh,
                own_dir_acc={str(l): round(own_te[l], 4) for l in layers},
                prep_dir_acc={str(l): round(prep_acc[l], 4) for l in layers},
                iv_own_L1_full=iv_own.get('L1_full', {}),
                iv_own_L1_d_only=iv_own.get('L1_d_only', {}),
                iv_own_L1_orth=iv_own.get('L1_orth_only', {}),
                iv_prep_L1_d_only=iv_prep.get('L1_d_only', {}),
            )
            print(f"sd={sd} {mname:20s} h4={beh['held_top4']:.3f} "
                  f"own_L1={own_te[1]:.3f} prep_L1={prep_acc[1]:.3f} "
                  f"own_d_redir={iv_own.get('L1_d_only',{}).get('redirect_rate',0):.3f} "
                  f"prep_d_redir={iv_prep.get('L1_d_only',{}).get('redirect_rate',0):.3f}",
                  flush=True)

        results[str(sd)] = sd_res

    outp = Path('experiments/archive/functional_learning/data/causal_intervention'
                'per_model_directions.json')
    outp.write_text(json.dumps(results, indent=2, cls=S25.NpEnc))
    print(json.dumps(dict(status='ok', out=str(outp),
                          elapsed=round(time.time()-t0, 1)), indent=2))

if __name__ == '__main__':
    main()
