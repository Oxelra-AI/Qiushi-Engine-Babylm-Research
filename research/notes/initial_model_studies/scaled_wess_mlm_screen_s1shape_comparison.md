# scaled wess mlm screen 384x4 results — Scaled WESS-MLM screen: 384×4 success vs S1-depth failure

## Evidence

- Configurable screen script: `scripts/scaled_wess_mlm_screen.py`
- Smoke result: `data/scaled_wess_mlm_screen_smoke.json`
- 384×4 intermediate result: `data/scaled_wess_mlm_screen_384x4_240.json`
- S1-shaped 12×384 short result: `data/scaled_wess_mlm_screen_s1shape_240.json`
- Prior route contract: `plans/scaled_wess_mlm_screen_contract.md`

## What scaled wess mlm screen 384x4 results established

scaled wess mlm screen 384x4 results built a configurable WESS-MLM screen that reuses the earlier analysis/187 counterfactual paired suite and WESS/no-address/destroyed-routing logic, but allows larger DeBERTa-v2 shapes and records approximate word exposure, token exposure, masked targets, official-text validation loss, and mechanism metrics.

The script compiles and runs. The smoke run validated all four matched arms and accounting. Two nontrivial screens were then run at the same data mixture and training length:

- 384 hidden, 4 layers, 12 heads, intermediate 1280, 240 steps, batch 32, 10% episode ratio;
- S1-shaped 384 hidden, 12 layers, 12 heads, intermediate 1280, 240 steps, batch 32, 10% episode ratio.

Both used about 72,368 whitespace-word exposure, 111,308 token exposure, and 7,680 masked targets per arm — far below the intended 10M S1 screen.

## Result comparison

| run / arm | example acc | pair acc | log-odds | official MLM loss | swap delta | ablation delta |
|---|---:|---:|---:|---:|---:|---:|
| 384×4 plain | 0.1125 | 0.0000 | -0.0000 | 3.8994 | — | — |
| 384×4 no_address | 0.1438 | 0.0250 | -0.0007 | 3.9290 | 0.0000 | +0.0033 |
| **384×4 WESS** | **0.7938** | **0.6125** | **+5.9045** | 3.9112 | **+11.2003** | **+6.5474** |
| 384×4 random-route | 0.1229 | 0.0042 | +0.0001 | 3.9177 | -0.0034 | -0.0099 |
| S1 plain | 0.1125 | 0.0000 | -0.0000 | 4.1571 | — | — |
| S1 no_address | 0.1063 | 0.0000 | +0.0039 | 4.6498 | 0.0000 | +0.0231 |
| **S1 WESS** | 0.1063 | 0.0000 | +0.0018 | 4.3685 | +0.0015 | +0.0080 |
| S1 random-route | 0.1146 | 0.0000 | -0.0051 | 5.3654 | -0.0157 | -0.0209 |

## Scientific interpretation

### 1. WESS can scale from log-odds to top-1 at larger hidden size

The 384×4 run is a strong mechanism result: WESS reaches 79.4% example accuracy and 61.3% pair accuracy while plain/no-address/random controls remain near baseline. Slot-swap and last-write-ablation deltas are very large and address-specific. This confirms the wess mlm transfer multiseed results short bridge was not merely a tiny-model artifact; with enough hidden size and moderate depth, WESS converts the address-specific log-odds signal into top-1 paired binding.

### 2. The effect does not appear automatically in the 12-layer S1 shape under the same short schedule

The 12-layer S1-shaped run failed to learn the paired binding task at 240 steps / ~72k word exposure: WESS log-odds and interventions are essentially zero, and pair accuracy is zero. This is not a negative result for S1-scale WESS overall because the run is extremely short and the deeper model had worse early official-text MLM loss than the 4-layer screen. It is evidence about training dynamics: the target S1 depth may need more updates, different learning-rate/warmup dynamics, higher episode ratio, stronger binding-target masking, layer placement changes, or initialization/fusion adjustments before the slot pathway takes hold.

### 3. Do not blindly launch the full S1 10M run from the current implementation

The 384×4 result justifies continued scaling, but the S1-depth failure says the full 12-layer 10M screen should be preceded by an optimization/geometry bridge rather than launched exactly as-is. Otherwise a failed 10M run might only measure poor early slot-path optimization, not the mechanism's value.

## Next execution options

The most efficient next work is a focused scaling-dynamics sweep before the expensive official-compatible screen:

1. **Depth ladder at fixed exposure:** 4, 6, 8, 12 layers with hidden 384, 240–500 steps, same 10% episode ratio. Determine where WESS learning breaks.
2. **Episode ratio / target frequency:** for S1 depth, compare 10%, 25%, 50% episode examples or force every synthetic example's binding target to be supervised; the current S1 run saw only 720 episode examples.
3. **Longer S1 warm-up screen:** S1-shaped WESS/plain/no-address/random at 1000–2000 steps, still far below 10M, to see whether the slot pathway is merely slower.
4. **Fusion/layer placement:** inject slot fusion at a middle or final subset of layers, or use a gated residual initialized near zero so the deep model can preserve MLM while learning slot use.
5. **Checkpoint/evaluator integration:** once S1-depth WESS shows nonzero pair accuracy/interventions, add HF checkpoint saving and official Entity/EWoK/GlobalPIQA evaluation.

The immediate conclusion is not to return to micro-world work. The mechanism works. The open problem is scaling and optimization of the WESS pathway inside the target BabyLM DeBERTa training regime while preserving official-score strengths.
