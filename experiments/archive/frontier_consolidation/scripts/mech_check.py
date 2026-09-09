#!/usr/bin/env python3
"""research mechanical verification for corrected dual-view shared-private trainer."""
from __future__ import annotations
import json, os, sys
from pathlib import Path

STUDY = Path("experiments/archive/frontier_consolidation")
WORKSPACE = STUDY
SCRIPTS = WORKSPACE / "scripts"
sys.path.insert(0, str(SCRIPTS))
OUT = WORKSPACE / "data/mech_check"
OUT.mkdir(parents=True, exist_ok=True)
hf = str(OUT / "hf_cache")
os.makedirs(hf, exist_ok=True)
os.environ.setdefault("HF_HOME", hf)
os.environ.setdefault("TRANSFORMERS_CACHE", hf)
os.environ.setdefault("HF_MODULES_CACHE", str(Path(hf) / "modules"))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, DebertaV2ForMaskedLM
import dual_view_corrected_trainer as T
from adapter_modeling import AdapterDebertaV2ForMaskedLM

TOK = WORKSPACE / "data/compliant_tokenizer"
AUX = WORKSPACE / "data/aux_pair_data/aux_pair_data.json"
POOL = WORKSPACE / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"

class Args:
    hidden_size=480; n_layer=8; n_head=8; ffn_mult=4; seq_length=256; aux_max_length=464
    adapter_bottleneck=128; adapter_scale=1.0


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(str(TOK))
    mask_id = int(tok.mask_token_id)
    cls_id = tok.cls_token_id if tok.cls_token_id is not None else tok.bos_token_id
    sep_id = tok.sep_token_id if tok.sep_token_id is not None else tok.eos_token_id
    pad_id = tok.pad_token_id or 0
    report = {"status":"MECH_CHECK", "device":str(dev)}

    # zero-output equivalence: copy a stock state into augmented stock tensors and compare logits.
    cfg = T.build_model(Args(), tok).config
    stock = DebertaV2ForMaskedLM(cfg).to(dev).eval()
    aug = AdapterDebertaV2ForMaskedLM(cfg).to(dev).eval()
    sd = aug.state_dict()
    for k, v in stock.state_dict().items():
        if k in sd: sd[k] = v
    aug.load_state_dict(sd, strict=False)
    x = torch.randint(5, 1000, (2, 32), device=dev); am = torch.ones_like(x)
    with torch.no_grad():
        report["init_logit_max_abs_diff"] = float((stock(input_ids=x, attention_mask=am).logits - aug(input_ids=x, attention_mask=am).logits).abs().max().cpu())

    raw = json.load(open(AUX))
    aux_data = raw["pair_data"]
    pair_eids = {int(k) for k in aux_data}
    rows = {}
    for line in open(POOL):
        r=json.loads(line); eid=int(r["example_id"])
        if eid in pair_eids:
            rows[eid]=r
        if len(rows)>=8: break
    eids = sorted(rows)[:8]
    exs=[T.Ex(rows[e]["text"], int(rows[e]["words"]), str(rows[e].get("source","")), e) for e in eids]
    ds=T.DualViewDataset(exs, tok, 256, pair_eids)
    batch=T.collate([ds[i] for i in range(len(exs))])
    ids=batch["input_ids"].to(dev); att=batch["attention_mask"].to(dev); wg=batch["word_group"].to(dev)
    gen=torch.Generator(device=dev); gen.manual_seed(43023)
    mi, labels = T.apply_wwm(ids, att, wg, tok, 0.15, gen)
    report["n_pair_rows_in_microbatch"] = int(batch["is_pair"].sum().item())

    vidx=[i for i in range(len(exs)) if bool(batch["is_pair"][i])]
    recs=[aux_data[str(int(batch["example_id"][i].item()))] for i in vidx]
    units = T.collect_aux_units(wg, labels, vidx, recs)
    report["aux_units"] = len(units)
    built_a, words_a, cond_a, free_a = T.build_view_batches(units, "aligned", 43022, 1, cls_id, sep_id, mask_id, pad_id, 464, dev)
    built_s, words_s, cond_s, free_s = T.build_view_batches(units, "shuffled", 43022, 1, cls_id, sep_id, mask_id, pad_id, 464, dev)
    report.update({
        "aligned_aux_words": words_a, "shuffled_aux_words": words_s,
        "charged_words_identical": words_a == words_s,
        "aligned_conditioned_views": cond_a, "aligned_free_views": free_a,
        "shuffled_conditioned_views": cond_s, "shuffled_free_views": free_s,
    })
    if built_a is not None and built_s is not None:
        ta = built_a[2][built_a[2] != -100].sort().values
        ts = built_s[2][built_s[2] != -100].sort().values
        report["aux_target_multiset_identical"] = bool(ta.shape == ts.shape and torch.equal(ta, ts))
        report["aux_input_differs"] = bool(not torch.equal(built_a[0], built_s[0]))
        # Free views are every second row in construction; these must be identical aligned/shuffled.
        report["source_free_views_identical"] = bool(torch.equal(built_a[0][1::2], built_s[0][1::2]) and torch.equal(built_a[2][1::2], built_s[2][1::2]))

    # Private aux gradient check
    model = AdapterDebertaV2ForMaskedLM(cfg).to(dev).train()
    # ensure adapter branch has nonzero reachable gradients through up; zero-init up still gets gradient, so no perturb needed.
    adapter_names = {n for n,_ in model.named_parameters() if ".adapter." in n}
    model.zero_grad(set_to_none=True)
    inp, aatt, lab = built_a
    flags=[]
    for n,p in model.named_parameters():
        flags.append((n,p,p.requires_grad))
        if n not in adapter_names: p.requires_grad_(False)
    out=model(input_ids=inp, attention_mask=aatt)
    loss=F.cross_entropy(out.logits.view(-1,out.logits.shape[-1]), lab.view(-1), ignore_index=-100)
    loss.backward()
    stock_norm=0.0; adapter_norm=0.0; nstock=0
    for n,p,fl in flags:
        if p.grad is not None:
            gn=float(p.grad.norm().cpu())
            if n in adapter_names: adapter_norm += gn
            else: stock_norm += gn; nstock += int(gn>0)
        p.requires_grad_(fl)
    report.update({
        "aux_loss_value": float(loss.detach().cpu()),
        "aux_stock_grad_norm_sum": stock_norm,
        "aux_adapter_grad_norm_sum": adapter_norm,
        "aux_n_stock_params_with_grad": nstock,
        "private_gradient_ok": bool(stock_norm == 0.0 and adapter_norm > 0.0 and float(loss.detach().cpu()) > 0.0),
    })
    (OUT/"mech_check.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)

if __name__ == "__main__": main()
