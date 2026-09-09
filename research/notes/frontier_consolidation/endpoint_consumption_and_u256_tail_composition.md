# endpoint consumption and u256 tail composition — endpoint-consumption assets and U256 tail-composition interpretation

CPU/file-only analysis; no GPU was used. Scale1.75 100M training was complete. Its full official evaluation and U256 faithful-visibility 100M training remained pending.

## 1. scale1.75 100M endpoint loads through the official package path (CPU)
`scripts/scale1p75_hf_load_smoke.py` →
`data/scale1p75_hf_load_smoke/scale1p75_hf_load_smoke.json` (ok=true),
note `notes/scale1p75_hf_load_smoke.md`.

- The first run exposed a cache-permission issue: `AutoModelForMaskedLM.from_pretrained(..., trust_remote_code=True)`
  tried to write the dynamic-module cache under a read-only default HF cache
  directory. Fixed by pointing HF_HOME / HF_MODULES_CACHE /
  HUGGINGFACE_HUB_CACHE to writable directories inside the isolated output. The evaluator
  path must set a writable module cache; this is a packaging/runtime detail, not a model defect.
- After the cache fix the custom `AdapterDebertaV2ForMaskedLM` loads on CPU, total params 35,463,008,
  adapter params 995,584 (matched by `.adapter.` naming), vocab 16,384, mask id 4, finite logits, tiny
  deterministic masked-LM probe loss 4.3965. Endpoint tokenizer vocab equals training tokenizer vocab
  (endpoint tokenizer.json SHA differs only from HF reserialization, as in dual mechanism 20m overlap).
- Two initial `ok=false` checks were smoke-script assumptions (raw tokenizer-JSON SHA equality; adapter
  name substring), corrected to semantic vocab equality and `.adapter.` naming. This is not a BabyLM score.

## 2. U256 recovered-tail composition vs row256-visible prefix (CPU)
`scripts/u256_tail_balance_profile.py` →
`data/u256_tail_balance_profile/u256_tail_balance_profile.json`,
`tail_balance_by_source.csv`, `top_tail_enriched_rows.csv`, note `notes/u256_tail_balance_profile.md`.

Exact legal substrate verified (pool SHA `2159...5a23`, tokenizer JSON SHA `91b7...e8f9`).
Recovered tokens 369,626, active ratio 1.025857, recovered words 197,806, boundary pieces 10,461 —
identical to the active endpoint consumption and u256 interpretation profile, so the two independent scans agree.

Decisive new finding: the U256-recovered suffix mass is NOT a neutral extension of the visible prefix.
Global suffix-minus-prefix shifts:
- Row-type: CHILDES-like **+0.4041**, subtitle-like −0.1151, list/TOC-like −0.1615, cleanish −0.1515,
  allcaps/glued **+0.0639**, encoding-noise −0.0012.
- Content-cue per word: quantitative **+0.0166**, physical +0.0009, action +0.0001, material −0.0000,
  spatial **−0.0057**, mental/social −0.0030.
- Source share of recovered tokens: CHILDES 60.56%, SimpleWiki 20.28%, OpenSubtitles 10.51%,
  Gutenberg 8.11%, qwen_pair_packed 0.35%, compact FineWeb reinvest 0.17%.

Interpretation for the pending U256 100M endpoint:
- The U256 gain, if it matures, is dominated by **short CHILDES utterance-suffix credit** plus SimpleWiki
  tails, with a mild quantitative-cue enrichment; it is almost disjoint from the compact-view reinvest block.
- The enriched allcaps/glued fraction and depleted cleanish fraction flag a real risk that some recovered
  mass is noisy subtitle/Gutenberg-list tail text (see `top_recovered_tail_noise_rows.csv` in active endpoint consumption and u256 interpretation and
  `top_tail_enriched_rows.csv` here). If the U256 endpoint reverses on EWoK material/quantitative or Reading,
  read it against this suffix mix, not as evidence against compact-view reinvestment itself.
- Because the recovered content-cue shift is near-zero except quantitative, U256 is unlikely to repair the
  same EWoK material-dynamics deficit that fixed scale1.75 leaves; the two routes are mechanistically distinct
  (residual capacity redirection vs stream-object visibility over official-source tails).

## 3. When endpoints deliver
- scale1.75: after `data/scale1p75_100M_full_eval_hardened/summary/...summary.json` appears,
  run `scripts/scale1p75_post_eval_analyzer.py` (payload backfill + item-family flips vs spatial repair route status 100M).
  If Overall ≥ 41.8, route to legality/reproducibility/packaging verification using dual mechanism 20m overlap readiness/provenance
  and endpoint consumption and u256 tail composition HF-load smoke (including the writable-module-cache requirement), then submission/HF readiness.
- U256: after its endpoint exists and no conflicting eval runs, submit
  `scripts/full_eval_u256_100M_hardened.py`, then
  `scripts/u256_post_eval_analyzer.py`, and interpret with active endpoint consumption and u256 interpretation + endpoint consumption and u256 tail composition U256 profiles.

Do not launch another expensive route until one endpoint score changes the route judgment or frees resources.
