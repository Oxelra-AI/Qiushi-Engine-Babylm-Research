# consistency decoy diagnostic and route correction predictor global decoy probe

Broad exact-changed-row test of whether a ridge source->rewrite predictor improves held-out true-vs-same-row-decoy separation over identity residuals.

CPU-only. No model update, official evaluation, corpus/tokenizer change, or H100 work.

## Inputs
- changed rows loaded: `768`; center_mode `row_centered`; dims `64,128,480`
- ridge_alpha `10.0`, train_frac `0.6`, splits `5`
- train SHA: `3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691`
- tokenizer SHA: `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`

## Results
### tokenmean_80M (pairs=3093, rows=768)
- d64 identity all: true-decoy-mean=0.9731, true-decoy-max=0.7542, top1=0.9997
  ridge holdout: pred true-decoy-mean=0.9800411059938094, identity true-decoy-mean=0.9754535282011018, pred top1=0.9998377939983779, identity top1=0.9995116518992188, adv_mean=0.004587577792707753, adv_top1=0.0003261420991591768
- d128 identity all: true-decoy-mean=0.9929, true-decoy-max=0.7881, top1=0.9997
  ridge holdout: pred true-decoy-mean=1.0148344235251954, identity true-decoy-mean=0.994664446597654, pred top1=0.999350750853362, identity top1=0.9998383185125304, adv_mean=0.02016997692754141, adv_top1=-0.0004875676591683531
- d480 identity all: true-decoy-mean=0.9894, true-decoy-max=0.7935, top1=1.0000
  ridge holdout: pred true-decoy-mean=1.0394479296907573, identity true-decoy-mean=0.9911394792638833, pred top1=0.9998383185125304, identity top1=1.0, adv_mean=0.04830845042687393, adv_top1=-0.00016168148746968923
### tokenmean_100M (pairs=3093, rows=768)
- d64 identity all: true-decoy-mean=0.9763, true-decoy-max=0.7606, top1=1.0000
  ridge holdout: pred true-decoy-mean=0.9839455028932301, identity true-decoy-mean=0.9782058793683078, pred top1=0.9998377939983779, identity top1=1.0, adv_mean=0.005739623524922255, adv_top1=-0.00016220600162206723
- d128 identity all: true-decoy-mean=0.9967, true-decoy-max=0.7939, top1=1.0000
  ridge holdout: pred true-decoy-mean=1.017768046165809, identity true-decoy-mean=0.9981222873996909, pred top1=0.999350750853362, identity top1=1.0, adv_mean=0.019645758766118092, adv_top1=-0.0006492491466380424
- d480 identity all: true-decoy-mean=0.9929, true-decoy-max=0.7997, top1=1.0000
  ridge holdout: pred true-decoy-mean=1.0417361620961294, identity true-decoy-mean=0.9942002757404509, pred top1=0.9998383185125304, identity top1=1.0, adv_mean=0.047535886355678535, adv_top1=-0.00016168148746968923

## Interpretation
- This broad predictor diagnostic uses stable example_id sampling from the changed-row span map, avoiding the sparse/front-stream issue in the first predictor probe.
- A predictor is useful for pair-specificity only if its held-out true-minus-same-row-decoy margin or top1 exceeds the identity source residual. If identity is already stronger, pair-specific separation is already present in the residual geometry and the predictor is not justified as a correspondence extractor.
- This is still only representation evidence. A future trainer must decide whether gentle agreement improves downstream learning without erasing private details; it cannot claim expected score gain from this probe alone.

Full JSON: `experiments/archive/frontier_consolidation/data/predictor_global_decoy_probe/predictor_global_decoy_probe.json`
