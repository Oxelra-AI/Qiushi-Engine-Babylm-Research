# throughput fix execution note: Throughput bottleneck resolved — execution note

## State at step close

### Training Completions
All four DeBERTa dose arms are now fully trained:
- MAX view: 100M, 2,552 steps, 34,467,424 params, loss_last 2.4930 ✓
- MAX repeat: 100M, 2,552 steps, 34,467,424 params, loss_last 2.3379 ✓  
- 1.82x midpoint view: 100M, 2,541 steps, 34,467,424 params, loss_last 2.5445 ✓
- 1.82x midpoint repeat: 100M, 2,541 steps, 34,467,424 params, loss_last 2.5047 ✓

### Evaluation Throughput
The serialized GPU evaluator completed only 3/25 evaluations in 2.6 hours while contending with training for GPU resources. `cpu_parallel_dose_eval.py` was constructed with the following configuration; the projected parallel throughput is not a completed timing result:
- Forces CUDA_VISIBLE_DEVICES="" on all subprocesses for CPU-only evaluation.
- Runs 16 workers in parallel, with an estimate of approximately 48 evaluations in 60-90 minutes.
- Reuses 20 existing results from prior evaluations.
- Evaluates all 7 dose arms: clean0(10-80M), dose1 v/r(10-100M), dose1p82 v/r(10-100M), MAX v/r(10-100M).

### Standalone readout
Built `dose_readout.py` — reads the unified CSV and seed spread data to produce:
- Semantic profile (V-R at each dose)
- Clean-free growth (V_dose - V_1x)
- Total treatment (V-C at matched checkpoints)
- Human-readable summary with seed-spread context

AST-checked. Ready to run the moment the CSV is written.

### GPU utilization
Both H100s productively used:
- GPU 0: RoBERTa MAX view training, at earlier analysis / ~2552 (~31%), ~60 min remaining
- GPU 1: RoBERTa MAX clean training, just launched (~90 min)

### Pending Measurements
- MAX ladder evaluator (still running, redundant with CPU eval)
- midpoint evaluator (waiting / running, partially redundant)
- curve integrator (waiting for evaluator output)
- RoBERTa MAX view training (~60 min remaining)
- RoBERTa MAX clean training (~90 min remaining)
- RoBERTa evaluator (waiting for training)

### Next work (immediate)
1. Wait for the CPU evaluation to complete → run dose_readout.py
2. Read the dose profile in the frozen reading order from pre result reading order and replication plan
3. RoBERTa view/clean should finish concurrently
4. Read RoBERTa evaluator results when available
5. Based on dose curve shape, determine next H100 action (second-seed MAX replication of carrier leg)

### Unfinished Analysis Checks
- Separate trajectory means from pointwise seed spread.
- Add family-concentration and leave-one-family-out summaries.
- Include token/WWM tables in the integrator.
These checks remain pending and should precede final synthesis.
