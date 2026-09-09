# successor route comparison while minfreq runs source-view consistency hidden-gradient probe

CPU-only actual-checkpoint probe. No parameter update, official evaluation, corpus change, or GPU use.

## Sample
- stream sample mode: `front`
- rows loaded: `512`
- batch size: `32`; max batches per checkpoint: `6`
- train SHA: `3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691`
- tokenizer SHA: `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`

## Gradient summaries
- tokenmean_80M: usable_batches=5, mlm_loss_mean=2.4388, aux_cos_loss_mean=0.0969, aux/MLM hidden-L2 λ=1 mean=0.093, aux-span ratio=0.350, grad cosine all=0.0024, suggested λ for 5% all-hidden L2≈0.5361
- wordmean_80M: usable_batches=5, mlm_loss_mean=2.6077, aux_cos_loss_mean=0.0986, aux/MLM hidden-L2 λ=1 mean=0.094, aux-span ratio=0.346, grad cosine all=0.0029, suggested λ for 5% all-hidden L2≈0.5324

## Interpretation
- The probe compares gradient pressure at the final hidden tensor, not full parameter updates. It is a scale and compatibility measurement for future construction, not score evidence.
- If source-view consistency is later selected, λ should be chosen so its hidden-state gradient is a small fraction of MLM on ordinary mixed batches, because actual paired representations are already close and full-vector forcing risks damaging source-specific syntax/entities.
- Word-mean versus token-mean on the same probe rows has aux cosine loss delta +0.0017 and aux/MLM hidden-gradient-ratio delta +0.001; combine this with successor route comparison while minfreq runs representation alignment and earlier analysis scores before attributing GlobalPIQA/COMPS movement to pair abstraction.

Full JSON: `experiments/archive/frontier_consolidation/data/consistency_gradient_probe/consistency_gradient_probe.json`
