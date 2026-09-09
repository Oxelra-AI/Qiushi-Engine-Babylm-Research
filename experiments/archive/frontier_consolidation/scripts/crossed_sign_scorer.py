#!/usr/bin/env python3
"""research: Score crossed-sign interaction probe on BabyLM checkpoints.

Computes focused pseudo-log-likelihood (PLL) for each alternative under
each context, the 2×2 interaction Δ, crossed-sign accuracy, and
correlation with the known scale1.75 cheap7 trajectory.

Usage:
  python crossed_sign_scorer.py \
      --probe-json .../crossed_sign_probe.json \
      --run-dir .../step106_.../hf_model \
      --checkpoints chck_77M chck_78M chck_79M chck_80M chck_81M chck_82M chck_83M chck_100M \
      --gpu 0 --tag scale1p75_late_panel [--smoke 5]
"""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse, json, os, sys, time, pathlib, math

# Set writable HF caches BEFORE importing transformers
_ws = _public_path('experiments/archive/frontier_consolidation')
_cache = str(_public_path('experiments/archive/frontier_consolidation/.hf_cache_scorer'))
os.makedirs(_cache, exist_ok=True)
os.environ.setdefault("HF_HOME", _cache)
os.environ.setdefault("HF_MODULES_CACHE", os.path.join(_cache, "modules"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(_cache, "transformers"))

from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Known scale1.75 cheap7 for correlation
CHEAP7 = {
    "chck_77M": 43.2821428571, "chck_78M": 43.7021428571,
    "chck_79M": 43.5785714286, "chck_80M": 43.8121428571,
    "chck_81M": 43.6492857143, "chck_82M": 43.9594498765,
    "chck_83M": 43.8078571429, "chck_100M": 43.5431599193,
}


def compute_focused_pll(model, tokenizer, full_text, focus_text, device,
                        max_length=256):
    """Compute focused PLL of focus_text within full_text via masked scoring."""
    enc = tokenizer(full_text, return_tensors="pt", max_length=max_length,
                    truncation=True, return_offsets_mapping=True)
    input_ids = enc["input_ids"][0]
    attn_mask = enc["attention_mask"][0]
    offsets = enc["offset_mapping"][0].tolist()

    # Locate focus character span
    idx = full_text.rfind(focus_text)
    if idx < 0:
        return None, 0
    fstart, fend = idx, idx + len(focus_text)

    # Token positions whose start falls within the focus span
    positions = [i for i, (os, oe) in enumerate(offsets)
                 if os >= fstart and os < fend and os < oe]
    if not positions:
        return None, 0

    n = len(positions)
    mask_id = tokenizer.mask_token_id

    # Batch: one masked copy per focus token
    batch_ids = input_ids.unsqueeze(0).expand(n, -1).clone().to(device)
    batch_attn = attn_mask.unsqueeze(0).expand(n, -1).to(device)
    target_ids = []
    for j, pos in enumerate(positions):
        target_ids.append(batch_ids[j, pos].item())
        batch_ids[j, pos] = mask_id

    with torch.no_grad():
        logits = model(input_ids=batch_ids, attention_mask=batch_attn).logits

    total_ll = 0.0
    for j, pos in enumerate(positions):
        lp = F.log_softmax(logits[j, pos], dim=-1)
        total_ll += lp[target_ids[j]].item()
    return total_ll, n


def score_item(model, tokenizer, item, device):
    """Score one crossed-sign item: 4 contextual + 2 context-free PLLs."""
    c1, c2 = item["context_1"], item["context_2"]
    altA, altB = item["alt_A"], item["alt_B"]
    r = {"id": item["id"], "family": item["family"]}

    for ctx_lbl, ctx in [("C1", c1), ("C2", c2), ("free", "")]:
        for alt_lbl, alt in [("A", altA), ("B", altB)]:
            full = (ctx + " " + alt) if ctx else alt
            pll, nt = compute_focused_pll(model, tokenizer, full, alt, device)
            r[f"pll_{alt_lbl}_{ctx_lbl}"] = pll
            r[f"ntok_{alt_lbl}_{ctx_lbl}"] = nt

    pAC1 = r.get("pll_A_C1"); pBC1 = r.get("pll_B_C1")
    pAC2 = r.get("pll_A_C2"); pBC2 = r.get("pll_B_C2")
    pAf  = r.get("pll_A_free"); pBf = r.get("pll_B_free")

    if all(v is not None for v in [pAC1, pBC1, pAC2, pBC2]):
        m1 = pAC1 - pBC1          # margin under C1
        m2 = pAC2 - pBC2          # margin under C2
        r["margin_C1"] = m1
        r["margin_C2"] = m2
        r["interaction"] = m1 - m2 # the crossed-sign interaction
        r["crossed_sign"] = int(m1 > 0 and m2 < 0)
    else:
        r["margin_C1"] = r["margin_C2"] = r["interaction"] = r["crossed_sign"] = None

    r["bias"] = (pAf - pBf) if (pAf is not None and pBf is not None) else None
    return r


def _pearson(x, y):
    if len(x) < 3: return None
    x, y = np.asarray(x, float), np.asarray(y, float)
    dx, dy = x - x.mean(), y - y.mean()
    d = math.sqrt(float((dx**2).sum() * (dy**2).sum()))
    return float((dx * dy).sum() / d) if d > 0 else None

def _spearman(x, y):
    if len(x) < 3: return None
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).statistic)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-json", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--smoke", type=int, default=0,
                    help="first N items per family (0=all)")
    args = ap.parse_args()

    ws = _public_path('experiments/archive/frontier_consolidation')
    OUT = ws / "data" / "crossed_sign_scores"
    OUT.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # load probe
    probe = json.load(open(args.probe_json))
    items = probe["items"]
    probe_sha = probe["content_sha256"]

    if args.smoke > 0:
        by_fam = defaultdict(list)
        for it in items: by_fam[it["family"]].append(it)
        items = [it for f in sorted(by_fam) for it in by_fam[f][:args.smoke]]
        print(json.dumps({"event": "smoke_filter", "n_items": len(items)}))

    run_dir = pathlib.Path(args.run_dir)
    all_ckpt = {}
    comp_ser, cs_ser = {}, {}

    for ckpt in args.checkpoints:
        cp = run_dir / ckpt
        if not cp.exists():
            print(json.dumps({"event": "skip", "checkpoint": ckpt})); continue
        print(json.dumps({"event": "load_model", "checkpoint": ckpt,
                          "path": str(cp)}))
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(str(cp), trust_remote_code=True)
        mdl = AutoModelForMaskedLM.from_pretrained(str(cp), trust_remote_code=True)
        mdl.eval().to(device)

        ir = []
        for i, item in enumerate(items):
            ir.append(score_item(mdl, tok, item, device))
            if (i+1) % 50 == 0:
                print(json.dumps({"event": "progress", "checkpoint": ckpt,
                                  "scored": i+1, "total": len(items)}))

        # per-family
        by_fam = defaultdict(list)
        for r in ir: by_fam[r["family"]].append(r)
        pf = {}
        for f in sorted(by_fam):
            ints = [r["interaction"] for r in by_fam[f] if r["interaction"] is not None]
            crs  = [r["crossed_sign"] for r in by_fam[f] if r["crossed_sign"] is not None]
            bis  = [r["bias"] for r in by_fam[f] if r["bias"] is not None]
            pf[f] = {
                "mean_interaction": float(np.mean(ints)) if ints else None,
                "std_interaction": float(np.std(ints)) if ints else None,
                "crossed_sign_accuracy": float(np.mean(crs)) if crs else None,
                "mean_bias": float(np.mean(bis)) if bis else None,
                "n_scored": len(ints), "n_items": len(by_fam[f]),
            }
        # depth sub-groups for entity_state
        es = [r for r in ir if r["family"] == "entity_state"]
        depth_groups = defaultdict(list)
        for r in es:
            d = next((it.get("depth") for it in items if it["id"] == r["id"]), None)
            if d is not None: depth_groups[d].append(r)
        depth_summary = {}
        for d in sorted(depth_groups):
            di = [r["interaction"] for r in depth_groups[d] if r["interaction"] is not None]
            dc = [r["crossed_sign"] for r in depth_groups[d] if r["crossed_sign"] is not None]
            depth_summary[f"depth_{d}"] = {
                "mean_interaction": float(np.mean(di)) if di else None,
                "crossed_sign_accuracy": float(np.mean(dc)) if dc else None,
                "n": len(di),
            }

        fam_means = [v["mean_interaction"] for v in pf.values()
                     if v["mean_interaction"] is not None]
        composite = float(np.mean(fam_means)) if fam_means else None
        all_cs = [r["crossed_sign"] for r in ir if r["crossed_sign"] is not None]
        ocs = float(np.mean(all_cs)) if all_cs else None
        el = round(time.time() - t0, 2)

        all_ckpt[ckpt] = {"per_family": pf, "depth_summary": depth_summary,
                          "composite": composite, "overall_crossed_sign": ocs,
                          "n_scored": sum(1 for r in ir if r["interaction"] is not None),
                          "elapsed_sec": el, "item_results": ir}
        comp_ser[ckpt] = composite
        cs_ser[ckpt] = ocs
        print(json.dumps({"event": "checkpoint_done", "checkpoint": ckpt,
                          "composite": composite, "crossed_sign": ocs,
                          "elapsed_sec": el}))
        del mdl; torch.cuda.empty_cache()

    # correlations
    shared = sorted(set(comp_ser) & set(CHEAP7))
    cv = [(comp_ser[c], CHEAP7[c]) for c in shared if comp_ser[c] is not None]
    sv = [(cs_ser[c], CHEAP7[c]) for c in shared if cs_ser[c] is not None]
    corr = {
        "composite_vs_cheap7_pearson":  _pearson([a for a,_ in cv],[b for _,b in cv]),
        "composite_vs_cheap7_spearman": _spearman([a for a,_ in cv],[b for _,b in cv]),
        "crossed_sign_vs_cheap7_pearson":  _pearson([a for a,_ in sv],[b for _,b in sv]),
        "crossed_sign_vs_cheap7_spearman": _spearman([a for a,_ in sv],[b for _,b in sv]),
        "n_shared": len(shared),
    }

    best_comp = max(comp_ser, key=lambda k: comp_ser[k] or -1e9) if comp_ser else None
    best_cs   = max(cs_ser, key=lambda k: cs_ser[k] or -1e9) if cs_ser else None

    # per-family correlations with cheap7
    fam_corr = {}
    all_fams = sorted(set(it["family"] for it in items))
    for fam in all_fams:
        fv = [(all_ckpt[c]["per_family"].get(fam,{}).get("mean_interaction"), CHEAP7[c])
              for c in shared
              if all_ckpt[c]["per_family"].get(fam,{}).get("mean_interaction") is not None]
        fam_corr[fam] = _pearson([a for a,_ in fv], [b for _,b in fv])

    output = {
        "status": "CROSSED_SIGN_SCORED", "tag": args.tag,
        "probe_sha256": probe_sha, "n_items_used": len(items),
        "smoke": args.smoke,
        "composite_series": comp_ser, "crossed_sign_series": cs_ser,
        "cheap7_series": {k: CHEAP7[k] for k in shared},
        "correlations": corr, "family_correlations": fam_corr,
        "best_composite_checkpoint": best_comp,
        "best_crossed_sign_checkpoint": best_cs,
        "official_best_checkpoint": "chck_82M",
        "checkpoints": {k: {kk: vv for kk, vv in v.items() if kk != "item_results"}
                        for k, v in all_ckpt.items()},
    }
    oj = OUT / f"{args.tag}.json"
    json.dump(output, open(oj, "w"), indent=2)

    # detail
    dj = OUT / f"{args.tag}_detail.json"
    json.dump({k: v["item_results"] for k, v in all_ckpt.items()},
              open(dj, "w"), indent=2)

    # markdown
    ln = [f"# Crossed-sign scores: {args.tag}", "",
          f"Probe SHA: `{probe_sha[:16]}...`  Items: {len(items)}",
          f"Best composite ckpt: **{best_comp}** | Best CS ckpt: **{best_cs}** | Official: chck_82M", ""]
    ln += ["## Correlations with cheap7", ""]
    for k, v in corr.items():
        ln.append(f"- {k}: **{v:.4f}**" if v is not None else f"- {k}: N/A")
    ln += ["", "## Per-checkpoint", ""]
    for ckpt in args.checkpoints:
        if ckpt not in all_ckpt: continue
        cr = all_ckpt[ckpt]
        ch = CHEAP7.get(ckpt, "?")
        ln.append(f"### {ckpt} (cheap7={ch})")
        ln.append(f"  composite={cr['composite']:.4f}  crossed_sign={cr['overall_crossed_sign']:.4f}"
                  if cr['composite'] is not None else "  composite=N/A")
        for f2 in sorted(cr["per_family"]):
            p2 = cr["per_family"][f2]
            mi = f"{p2['mean_interaction']:.4f}" if p2["mean_interaction"] is not None else "N/A"
            cs2 = f"{p2['crossed_sign_accuracy']:.3f}" if p2["crossed_sign_accuracy"] is not None else "N/A"
            bi = f"{p2['mean_bias']:.3f}" if isinstance(p2.get('mean_bias'),(int,float)) else "N/A"
            ln.append(f"  - {f2}: Δ={mi} cs={cs2} bias={bi}")
        if cr.get("depth_summary"):
            for dk, dv in sorted(cr["depth_summary"].items()):
                di2 = f"{dv['mean_interaction']:.4f}" if dv["mean_interaction"] is not None else "N/A"
                dc2 = f"{dv['crossed_sign_accuracy']:.3f}" if dv["crossed_sign_accuracy"] is not None else "N/A"
                ln.append(f"  - entity_state {dk}: Δ={di2} cs={dc2} n={dv['n']}")
        ln.append("")
    ln += ["## Per-family correlations", ""]
    for f3, fc in fam_corr.items():
        ln.append(f"- {f3}: {fc:.4f}" if fc is not None else f"- {f3}: N/A")
    om = OUT / f"{args.tag}.md"
    om.write_text("\n".join(ln))

    print(json.dumps({
        "status": "CROSSED_SIGN_SCORED", "tag": args.tag,
        "composite_vs_cheap7_pearson": corr["composite_vs_cheap7_pearson"],
        "best_composite_checkpoint": best_comp,
        "out_json": str(oj), "out_md": str(om),
    }, indent=2))


if __name__ == "__main__":
    main()
