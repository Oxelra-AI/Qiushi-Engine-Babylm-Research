# BabyLM Strict-Small — Official Rules & Landscape (official rules and landscape grounding)

Historical literature and evaluation landscape, recorded on 2026-08-13. Sources
are the official CfP/Findings papers cited below; this note is not a current rules re-verification.

## 1. Official constraints (2026 cycle, Strict-Small) — MUST obey

Source: 2026 CfP `\cite{choshen2026babylm}` (BabyLM Turns 4), consistent with 2025 `\cite{charpentier2025findings}` and 2024 `\cite{hu2024findings}`.

- **Data budget**: train on **≤10M words** (whitespace-separated word count). Official
  detoxified 10M Strict-Small corpus is provided; custom/swapped data allowed with a datasheet,
  still ≤10M words.
- **Training duration**: **≤10 epochs**, quantified as **≤100M words seen** (repeated exposures
  counted). This is the operative compute cap, not FLOPs.
- **Tokenizer / any learned-on-language tool counts toward the 10M budget** (train tokenizer,
  parser, LM etc. on the same/parts of the 10M; the *sum of all text seen by all training*
  ≤ 10M). This is a subtle, important rule.
- **Synthetic data** allowed only as a *closed system*: an augmenter's training data counts toward
  the budget. Generating >10M words from a model trained within-budget is legit.
- **External model (merged Interaction/Multimodal)**: allowed as caregiver/peer, but (a) submission
  model exposure incl. generated text ≤10M words; (b) **distillation revealing external tokenizer,
  weights, hidden states, or output distribution is NOT allowed** unless the external model's
  training word count is charged to the budget. Scalar rewards ≤10M.
- **Intermediate checkpoints REQUIRED**: every 1M words until 10M seen. Needed for the pipeline.
- **Scoring interface**: model must score a sequence of words (pseudo-)log-likelihood WITHOUT
  additional fine-tuning (zero-shot probability comparisons dominate eval).
- No epoch/hyperparameter search limits beyond the above; report findings.

## 2. Official evaluation (Overall metric)

- 2023: BLiMP + BLiMP-Supplement + (Super)GLUE + MSGS (50/30/20 weighting) `\cite{warstadt2023findings}`.
- 2024: BLiMP, BLiMP-Supplement, (Super)GLUE (fine-tune), EWoK (hidden); Overall = avg of task scores `\cite{hu2024findings}`.
- 2025/2026: **(Super)GLUE, BLiMP, EWoK + human-likeness tasks** (morphological generalization,
  entity tracking, concept/COMPS, reading-time / eye-tracking / self-paced reading prediction,
  word learning / AoA, adjective nominalization, past-tense). **Overall = macro average of
  human-likeness score and NLP-task score** `\cite{charpentier2025findings}`.
- Pipeline is open-source, builds on 2025 repo (github.com/babylm). Hidden tasks released near
  deadline. 2023 eval pipeline already acquired: `staging/acquired/babylm_evaluation-pipeline-2023_*`.

## 3. Current SOTA / winners to beat (Strict-Small)

- **2025 Strict-Small winner: MoEP**, macro average **42.3** `\cite{charpentier2025findings}`.
- Official 2025/2026 baseline: **GPT-BERT** (Charpentier & Samuel 2024) — hybrid causal+masked on
  LTG-BERT backbone; 2024 GPT-BERT Strict-Small Text Average **70.4** `\cite{hu2024findings}`.
  (NB: score *scales differ across years* — 2024 "Text Average ~70" vs 2025 "macro avg ~42" because
  2025 added harder human-likeness tasks and re-weighted. Must confirm exact 2026 aggregation before
  claiming SOTA.)
- 2023 winner: **ELC-BERT** (LTG-BERT + weighted sum of previous layers), trained >2000 epochs
  (now illegal under 10-epoch rule) `\cite{warstadt2023findings}`.
- GPT-2 Small = naive autoregressive baseline; SimPO preference baseline added 2026.

## 4. Candidate mechanisms already in the literature (to inherit / beat / avoid duplicating)

- **GPT-BERT hybrid CLM+MLM** on LTG-BERT backbone — current baseline & strong; the thing to beat.
- **AntLM**: alternate CLM/MLM objective schedules; +1–2.2% macro over baselines `\cite{yu2024antlm}`.
- **Adaptive MLM (AMLM)** + n-hot sub-token embeddings; 2025 score 41.9 `\cite{edman2025mask}`.
- **BabyHGRN**: HGRN2 linear-recurrent (RNN) backbone + distillation; competitive at low resource `\cite{haller2024babyhgrn}`.
- **Co4 LM**: single-layer, 8M-param, triadic (two-point-neuron) modulation, O(N); claims beat GPT-2/GPT-BERT baselines `\cite{zain2025single}`.
- **Variation Sets** (child-directed repetition-with-variation) augmentation — mixed effects `\cite{haga2024babylm}`.
- Text-complexity / simplification of pretraining data helps linguistic tasks `\cite{charpentier2025babylm}`.
- Curriculum learning: historically *largely unsuccessful* in 2023 meta-analysis — high bar.

## 5. Recorded compute configuration

- Recorded hardware: 2× H100. Single-GPU + 2-GPU DDP supported.
- BabyLM models are tiny (8M–125M typical); training is fast → enables heavy parallel ablation.

## 6. Candidate mechanisms (not yet selected)

The scientific question is which *mechanism* could improve sample efficiency and Overall,
rather than which hyperparameter setting scores highest. Candidate directions (NOT yet chosen):
- objective design (CLM/MLM hybridization schedule, adaptive masking) — crowded but improvable.
- backbone innovation (linear-recurrent / two-point-neuron / state-space) for 10M-word regime.
- data representation & tokenizer under the "tokenizer counts toward budget" rule — underexplored lever.
- the *human-likeness* sub-metric (reading-time, word-learning, morphology) — where Overall is
  now decided and where cognitive-plausibility mechanisms could give real, non-tuning gains.

Open question: which sub-scores dominate the 2026 macro average, and where is
the largest under-served headroom for a small model? That determines the breakthrough target.
