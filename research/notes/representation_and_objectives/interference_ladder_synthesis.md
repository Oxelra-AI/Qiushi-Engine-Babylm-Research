# ewok domain link — Interference ladder: synthesis and route implications

Status: **MEASURED**

## Frozen object
- 800 frames = 80 base (50 open_closed + 30 empty_full) × 10 interference conditions
- 800 renamed controls
- Content SHA256: `4135fafc13c729cdda18ef27074a5be5e3776cd1acd2152a36abafea283dcbff`
- All targets one-token under both legal16k and legal40k tokenizers
- Manifest: `experiments/archive/representation_and_objectives/data/interference_ladder/interference_ladder_manifest.json`

## Scored panel (19 checkpoints)
- Panel JSON: `experiments/archive/representation_and_objectives/data/interference_ladder_scores/interference_ladder_panel_summary.json`
- Panel note: `research/notes/representation_and_objectives/interference_ladder_panel.md`

## Key findings

### 1. Three universally saturated conditions
- `explicit_final` (direct statement): crossed=1.0 for all mature models
- `consistent_prior` (prior agrees + event): crossed=1.0 for all mature models
- `no_context` (pure prior): crossed=0.0 for all (by construction)

### 2. Three universally failed conditions
- `contradict_bare`: crossed=0.0 for all
- `contradict_temporal` (adding "Then"): crossed=0.0 for all — temporal markers do NOT help
- `contradict_reinforced` (repeated prior): crossed=0.0 for all

### 3. Three trajectory-discriminating conditions (the science)
- `last_event` (single action, no prior): legal16k_100M=0.988, scale1.75=0.34-0.61, legal40k_8x480=0.36, legal40k_depth12=0.038, fw_compact=0.063 — 26× variation
- `distractor_event` (unrelated distractor + action): similar ranking but uniformly lower — distractors hurt even without contradiction
- `reported_event` (third-party report): same ordering but further reduced

### 4. explicit_override: the asymmetry
- `explicit_override` (contradiction + explicit final statement): 0.5-1.0 for mature models
- This means models CAN override contradictory prior state IF the override is explicit, but NOT if the override requires inferring state from action
- The deficit is specifically action-to-state inference under interference, not general context failure

### 5. D-state consistency check
- `last_event` and `contradict_bare` match counterfactual micro world v3 static repair D-state ablation within expected variation (100 D frames vs 80 ladder frames)
- `explicit_final` matches exactly

### 6. Coupled sparse20 models fail everything
- Both coupled_aligned and coupled_shuffled fail even `explicit_final` — broadly damaged, not interference-specific

## What this establishes scientifically

The interference ladder separates two distinct capabilities:
1. **Reading explicit state** — all mature models succeed (explicit_final=1.0, consistent_prior=1.0, explicit_override≈1.0)
2. **Inferring state from action under interference** — catastrophically fails when ANY contradictory prior exists (last_event→0.0 with any prior contradiction)

The 26× trajectory variation in `last_event` (action-to-state WITHOUT interference) is the unsaturated discriminating signal. The catastrophic collapse at `contradict_bare` is the interference boundary.

## What it does NOT establish
- The aggregate EWoK correlations are weak (last_event vs EWoK accuracy Pearson=-0.13)
- The ladder does not directly predict GlobalPIQA hard52 or overall scores
- It is not a training selector and should not be used as one
- The measure does not tell us HOW to fix the interference; only WHERE the boundary is

## Route implications
- The measurement is clean and reproducible but does not predict official surfaces well enough to guide training alone
- This measure motivates further representation-theory work, not a training recipe
- The scientific contribution is the precise characterization: small MLMs fail at action-mediated state update under interference, not at explicit state reading or consistent-evidence tracking
- This suggests any training intervention should target the action→state inference pathway specifically, not general context or general state tracking

## Existing artifacts
- Freezer: `Scripts/freeze_interference_ladder.py`
- Scorer: `Scripts/score_interference_ladder.py`
- Domain link: `Scripts/ewok_domain_link.py`
- Per-target CSVs: `data/interference_ladder_scores/targets/*/interference_ladder_records.csv`

## Protected endpoint
chck_82M untouched. No training launched.
