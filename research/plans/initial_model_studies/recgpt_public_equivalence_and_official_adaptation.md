# earlier analysis — RecGPT public-source equivalence and official-corpus adaptation plan

## Why the current earlier analysis trainer must not launch as-is

Inspecting the public source at
`data/public_recgpt_source/serdardoesml-bblm26-recgpt-178566e/` confirms
the local `scripts/recgpt_official_train.py` is NOT numerically equivalent
to the public RecGPT learning system:

1. **Aurora optimizer differs materially.**
   - Public `optimizer.py`: `polar()` uses constants (a,b,c)=(2,-1.5,0.5), transposes when rows>cols.
   - Rectangular path uses a row-balancing preconditioner: `pp_iterations=2`, `pp_beta=0.5`,
     with `target_row_sq = n/m` and per-row scaling `D`, then final `update *= max(1, m/n)**0.5`.
   - Local earlier analysis `newton_schulz_5` uses constants (3.4445,-4.7750,2.0315) and NO row-balancing
     preconditioner. This changes update geometry for the 16x MLP matrices, which is exactly where
     Aurora is claimed to matter. Not equivalent.
   - Public momentum: `momentum.lerp_(grad, 1-mu)` then `grad.lerp_(momentum, mu)` (nesterov).
     Local uses a different `buf.mul_(mom).add_(grad); update=grad+mom*buf`. Not equivalent.
   - Public Cautious Weight Decay: `mask=(update*p).ge(0); p.add_(p*mask, alpha=-lr*wd)`.
     Local mixes grad-based masks. Not equivalent.

2. **NextLat auxiliary differs.**
   - Public `nl_aux_model.py`: bias-free `input_proj(2H->H)`, 3-layer MLP up/mid/down with GeLU,
     `down` zero-init, `mid` ones-init, residual + RMSNorm with learned `final_norm`, and a
     depth rollout using `embeddings[:, k:]` with segment-valid transition masking.
   - It consumes the model's pre-`h_to_e` normalized hidden `x` AND the factorized embedding `e`
     via `return_hidden_and_embed=True` (NOT recomputed embeddings). Local earlier analysis recomputes
     `e_to_h(embed_tokens(input_ids[:,1:]))` and uses a different MLP init. Not equivalent.

3. **Data/exposure semantics differ.**
   - Public dataloader streams packed doc-segments from parquet, trains on shifted targets
     (`x=s[:-1], y=s[1:]`), counts training tokens as sum(len-1), and uses a token clock.
   - Public strict-small checkpoint marks: `epoch_tokens*mark/10` — i.e. it assumes the dataset
     is ~10M words so "chck_kM" ≈ k million WORDS only because words≈epoch_tokens/10 by construction.
   - Local earlier analysis used a homemade packer and a token→word proxy. For a valid AoA trajectory the
     checkpoints must correspond to true WORD exposure marks (chck_1M..chck_9M), which requires the
     token clock to be tied to word counts.

## Official-corpus public-format data (ready)

`scripts/prepare_official_recgpt_public_format.py` produced
`data/recgpt_official_public_format/`:
- `official_10M_sentence_docs.jsonl`: 205,291 docs, exactly 10,000,000 words.
- `official_10M_sentence_docs.parquet`: list<int32> input_ids per doc.
- token_sum 14,075,099; public training-token count (sum len-1) 13,869,808.
- **words_per_training_token = 0.7210**, training_tokens_per_1M_words = 1,386,981.

Word-exposure checkpoint marks (true AoA-relevant):
- chck_kM should fire at k * 1,386,981 training tokens (k=1..9), because 1M words ≈ 1.387M training tokens.
- This is essentially what public `checkpoint_schedule("strict-small")` computes as
  `epoch_tokens*k/10` when epoch_tokens≈13.87M. So using the public code directly with our
  10M-word parquet yields word-faithful chck_1M..chck_9M automatically. Good.

## Correct next action (adapt public code, do not reimplement)

Rather than fix the simplified local optimizer/aux, IMPORT the public modules and run a thin driver:

1. Add a package path shim so `import main.model / main.optimizer / main.nl_aux_model / main.dataloader`
   resolve from `data/public_recgpt_source/serdardoesml-bblm26-recgpt-178566e/`.
   The model.py there is the same architecture as the patched loader; use the public model.py for training
   (it defines RecGPTConfig/RecGPTForCausalLM with segment_ids + return_hidden_and_embed).
2. Build a `get_base_dir()`-compatible layout (the public `common.py` derives paths); either
   set the expected env/base dir or write a small driver that constructs `TrainConfig` and calls
   `main.train.train(cfg)` with our parquet + tokenizer placed where it expects (`data/`, `tokenizers/`).
3. Numeric equivalence pre-check BEFORE the long run:
   - Load public `RecGPTForCausalLM` with the PUBLIC pretrained weights, run one packed batch through
     the public forward, and confirm loss matches the earlier analysis/230 evaluation-time behavior.
   - Instantiate the public `SingleDeviceAuroraWithAuxAdam` and public `NL_Aux_Model`, run 1-2 steps on
     a tiny batch from our parquet, confirm finite CE + NextLat losses and that param groups match the
     public `build_optimizer` split exactly (embed/lm_head/norm/e_to_h/h_to_e → Adam; 2D block matrices → Aurora).
   - Record losses; this is the "public-source equivalence" criterion.
4. Only after the equivalence pre-check passes, launch the full 10-epoch official-corpus run with:
   - `--checkpoint-strict-small` so chck_1M..chck_9M fire on the token clock ≈ word exposure.
   - microbatch_tok=32768, sequence_len=512, epochs=10, torch_compile on.
   - Save HF-compatible checkpoints + copy modeling_recgpt.py for evaluator `--backend causal`.

## Interpretation constraints

- The first full run tests whether the COMPLETE public RecGPT learning system reproduces its public
  phenotype on legal official data. Success must NOT be attributed to recursion alone.
- Only after establishing this reference do we run same-corpus, same-budget ablations that separate:
  causal interface, recursive weight sharing (depth), Aurora vs AdamW, and NextLat on/off.
- Continue tracking the unresolved evaluator-coordinate gaps (Reading -4.89 after space-fix,
  GlobalPIQA -1.97) when interpreting any Overall movement; do not fold them silently into model-card values.

## Files
- Public source: `data/public_recgpt_source/serdardoesml-bblm26-recgpt-178566e/`
- Official public-format data: `data/recgpt_official_public_format/`
- Data prep script: `scripts/prepare_official_recgpt_public_format.py`
- Superseded simplified trainer (do NOT launch): `scripts/recgpt_official_train.py`
