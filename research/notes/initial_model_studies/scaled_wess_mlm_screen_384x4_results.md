# scaled wess mlm screen 384x4 results — Scaled WESS-MLM screen, 384×4 intermediate run

## Evidence

- Script: `scripts/scaled_wess_mlm_screen.py`
- Smoke result: `data/scaled_wess_mlm_screen_smoke.json`
- Intermediate 384×4 result: `data/scaled_wess_mlm_screen_384x4_240.json`
- Contract: `plans/scaled_wess_mlm_screen_contract.md`

## What was run

The scaled wess mlm screen 384x4 results script turns the earlier analysis/187 bridge into a configurable screen with:

- DeBERTa-v2 relative/disentangled attention backbone;
- configurable hidden size/layers/heads/intermediate size;
- matched arms: `plain_mlm`, `no_address`, `wess_gold`, `wess_eventwise_random`;
- official BabyLM text plus counterfactual paired entity-state episodes;
- explicit accounting of examples, approximate word exposure, token exposure, and masked targets;
- fixed shortcut audit on the paired evaluation suite;
- binding example accuracy, pair accuracy, correct-vs-counterfactual log-odds, official-text MLM loss, slot swap and last-write ablation effects.

A tiny 2-layer smoke run completed first and confirmed the scalable script, accounting, WESS fusion, and result writing.

The main intermediate run used:

- hidden size 384;
- 4 DeBERTa-v2 layers;
- 12 heads;
- intermediate size 1280;
- 240 optimizer steps;
- batch size 32;
- episode ratio 0.10;
- 1200 train pairs, 240 eval pairs;
- 5000 official text examples;
- four matched arms.

Per arm, the run saw about 72,368 whitespace-word exposure and 111,308 token exposure, with 7,680 masked targets. This is still far below the planned 10M S1-shaped screen, so it is an intermediate mechanism-scale result, not an official BabyLM candidate.

## Shortcut audit on the fixed eval suite

All position/recency shortcuts fail on pair accuracy:

| shortcut | example acc | pair acc |
|---|---:|---:|
| first_state | 0.000 | 0.000 |
| last_state | 0.500 | 0.000 |
| nearest_state | 0.500 | 0.000 |
| majority_fixed_first_visible | 0.250 | 0.000 |
| prev_query_state | 0.000 | 0.000 |

The paired target still isolates entity-to-state assignment: solving one member by recency/position does not solve the counterfactual pair.

## Results

| arm | example acc | pair acc | mean log-odds | official MLM loss | swap delta | ablation delta |
|---|---:|---:|---:|---:|---:|---:|
| plain_mlm | 0.1125 | 0.0000 | -0.00004 | 3.8994 | — | — |
| no_address | 0.1438 | 0.0250 | -0.00069 | 3.9290 | 0.0000 | +0.0033 |
| **wess_gold** | **0.7938** | **0.6125** | **+5.9045** | 3.9112 | **+11.2003** | **+6.5474** |
| wess_eventwise_random | 0.1229 | 0.0042 | +0.00009 | 3.9177 | -0.0034 | -0.0099 |

## Scientific interpretation

### WESS converts the wess mlm transfer multiseed results log-odds signal into top-1 binding

At the same backbone shape/training schedule/data mixture, only `wess_gold` learns the paired entity-state task:

- example accuracy 0.7938 vs 0.1125–0.1438 for controls;
- pair accuracy 0.6125 vs 0.0000–0.0250 for controls;
- mean correct-vs-counterfactual log-odds +5.9045 vs approximately zero in controls.

This is the first run where the WESS bridge moves from probability direction to strong top-1 behavior.

### The effect remains address-specific

The matched `no_address` arm and the `wess_eventwise_random` address-destruction arm do not learn the pair task. They also show no meaningful slot intervention effect. Therefore the result is not explained by:

- synthetic episode text alone;
- extra parameters;
- gold entity/state spans;
- direct state-token access;
- the recurrent/fusion pathway without persistent entity addresses.

The large WESS-only intervention deltas show that slot state still causally steers the masked token:

- slot swap shifts probability toward the other entity's state by +11.20 nats;
- last-write ablation shifts probability toward the previous state by +6.55 nats.

### Official-text MLM loss is comparable in this short run

Official-text validation loss is similar across arms:

- plain_mlm 3.8994;
- wess_gold 3.9112;
- random route 3.9177;
- no_address 3.9290.

The WESS result does not come with an obvious immediate ordinary-MLM collapse in this intermediate setting.

## Important limits

This run is not yet a BabyLM SOTA candidate and not yet the planned S1 10M screen.

Limitations:

1. It uses 4 layers, not the S1 12-layer model.
2. It uses only ~72k word exposure per arm, not 10M cumulative exposure.
3. It evaluates controlled paired episodes and official-text MLM loss, not official Entity/EWoK/GlobalPIQA/BLiMP/Supplement/Reading columns.
4. The episode suite still uses gold span metadata and controlled location words.
5. The script does not yet save Hugging Face checkpoints for official evaluator use.

## Consequence for the route

The route should now move to an official-compatible S1-shaped WESS trainer and short-budget evaluation, not more micro-world work.

The next implementation should integrate this WESS/fusion mechanism into the real BabyLM trainer so that it can:

- train S1-shaped arms with proper 10M word accounting;
- save HF checkpoints;
- run official-compatible Entity, EWoK, GlobalPIQA, BLiMP/Supplement/COMPS and Reading checks;
- update the BabyLM scoreboard if a candidate becomes complete enough.

The 384×4 result strongly supports spending that engineering effort: persistent entity-indexed WESS has now shown (i) causal micro-world validity, (ii) DeBERTa MLM log-odds transfer across seeds, and (iii) top-1 paired binding at larger hidden size while controls fail.
