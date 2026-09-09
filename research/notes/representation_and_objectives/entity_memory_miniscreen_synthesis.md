# entity memory miniscreen synthesis entity-keyed event-memory miniscreen synthesis

> **Withdrawn interpretation (R36).** The corpus assigned the answer from the action outcome regardless of which entity the query named. The high accuracies below therefore do not establish query-conditioned entity-state binding or compositional generalization. Write-key permutation sensitivity and concentrated slot attention do not repair that invalid inference.
>
> The original measurements are retained unchanged as a historical record. Statements below that claim successful compositional binding, selective addressing of the queried entity, or satisfaction of the binding criteria are superseded by this notice and must not be cited as supported findings.
>
> A corrected paired-query corpus and revised training script were constructed, but they do not establish a completed, successful corrected-model experiment. This withdrawal is specific to this miniscreen; it does not withdraw the separate supplied-address experiments or the final BabyLM model evaluations. See [research catalog, R36](../../catalog.md).

## Experimental design

Three matched arms trained from scratch on 218K words of frozen synthetic two-entity state-update stories (12K training records, 8 epochs). Decisive evaluation withholds affected-entity × transition recombinations while ensuring every entity, action, and state occurs in acquisition examples. Two seeds (43022, 43123).

- **Vanilla**: 4-layer Transformer MLM, 192d, no memory block. 4,975,296 params.
- **Shared-key**: Same + entity-keyed event memory block inserted after layer 1. Write and read share one address projection. 5,271,745 params.
- **Independent-key**: Same + entity-keyed event memory block with separate write and read address projections. 5,271,745 params (matched to shared-key).

Tokenizer: legal 16k BPE from the chck_82M endpoint. Evaluation: first discriminative BPE piece (after space prefix) at the [MASK] position.

## Cross-seed results

### Held recombination accuracy (decisive test — entity×transition never co-occurred in training binding)

| Arm | Seed 43022 | Seed 43123 | Mean |
|-----|-----------|-----------|------|
| Vanilla | 76.6% | 75.0% | 75.8% |
| Shared-key | 96.4% | 100.0% | 98.2% |
| Independent-key | 100.0% | 97.9% | 99.0% |

Both memory arms outperform vanilla by +22–24 pp on held recombinations. Effect replicated across seeds. Train binding is 100% for all arms (memorized).

### Write-key permutation (causal intervention — flips which entity slot receives the event update)

| Arm | Seed 43022 held_recomb | Seed 43123 held_recomb |
|-----|----------------------|----------------------|
| Shared-key | 65.6% (Δ −30.7pp) | 60.4% (Δ −39.6pp) |
| Independent-key | 56.2% (Δ −43.8pp) | 64.6% (Δ −33.3pp) |

Permuting write-to-entity correspondence at inference drops held recomb by 31–44 pp. This proves the entity address is the operative mechanism for binding events to entities, not merely extra capacity.

### Multi-event order (held interleaving/reversal sequences)

| Arm | Seed 43022 | Seed 43123 | Mean |
|-----|-----------|-----------|------|
| Vanilla | 89.6% | 62.5% | 76.0% |
| Shared-key | 62.5% | 62.5% | 62.5% |
| Independent-key | 52.1% | 62.5% | 57.3% |

Vanilla has an unstable advantage (one seed only). All arms struggle with multi-event sequential processing. The gate-based write mechanism does not reliably implement sequential overwrite.

### Read/write transport

| Arm | Seed | Read slot-0 | Read H | Write H |
|-----|------|------------|--------|---------|
| Shared-key | 43022 | 0.974 | 0.046 | 0.286 |
| Shared-key | 43123 | 0.990 | 0.014 | 0.288 |
| Independent-key | 43022 | 0.641 | 0.014 | 0.172 |
| Independent-key | 43123 | 0.027 | 0.026 | 0.076 |

Shared-key reads almost exclusively from one slot (0.97–0.99 for slot 0) — highly selective entity addressing. Independent-key has variable and less consistent read routing.

## Interpretation

1. **The entity-keyed memory creates the missing compositional operation.** route2 factorial causal review showed BabyLM models can represent affected-entity identity but cannot compose it with action-result polarity into an output-usable final state. The memory block produces exactly this composition on held-out entity×transition recombinations (98–99% vs 76% vanilla).

2. **The entity address is causally operative.** Write-key permutation disrupts held recomb by 31–44 pp while keeping all parameters and network structure unchanged. This is not a capacity artifact.

3. **Shared-key creates more precise entity addressing.** Read slot-0 concentration 0.97–0.99 (shared) vs 0.03–0.64 (independent). Despite this, both achieve similar held_recomb accuracy, suggesting the compositional binding can form through either precise or distributed routing.

4. **Multi-event sequential overwrite remains weak.** The gate-based write mechanism does not reliably handle multiple events in sequence. The vanilla transformer may handle this through bidirectional attention rather than sequential state maintenance. This is a known limitation for the next design iteration.

5. **Vanilla achieves 76% held_recomb.** Not zero — ordinary transformers partially learn entity×transition binding from the synthetic data. The memory block pushes this 23 pp higher with causal entity-address dependence.

## Status Against the Scientific Criteria

The scientific criterion is: "A shared-key advantage that survives independent-key and write-permutation comparisons and causally updates only the queried slot would identify the missing operation."

- ✓ Both memory arms dramatically outperform vanilla on held recombinations
- ✓ Write-key permutation causally disrupts both arms (31–44 pp)
- ✓ Shared-key has concentrated read attention on the queried entity (0.97–0.99)
- ✓ Shared-key vs independent-key: both achieve ~98–99%, but shared has more precise routing
- ⚠ Multi-event overwrite is weak — architectural improvement needed before legal-stream deployment

## Data provenance

- Corpus: `experiments/archive/representation_and_objectives/training/data/entity_memory_corpus`
  - train SHA: `6f4f6f5dd7d93cce83d685a46fe61d90ed1edb85e4a3597ace666bfdde482b2f`
  - eval SHA: `6ad12a592ae799753f48eb9262ed9fbfe8e7a9958c0c14c41837abefb21893c1`
- Results: `experiments/archive/representation_and_objectives/training/data/{arm}_seed{seed}/result.json`
- Scripts: `experiments/archive/representation_and_objectives/training/scripts/generate_entity_memory_corpus.py`, `train_entity_memory_miniscreen.py`
- Synthesis: `experiments/archive/representation_and_objectives/data/miniscreen_synthesis.json`

## Next steps

The mechanism screen is positive: entity-keyed event memory creates compositional binding that generalizes to held-out recombinations and is causally dependent on entity addressing.

Before entering the legal-stream mixture:
1. Strengthen the multi-event gate or replace it with hard-attention overwrite
2. Integrate the memory block into DeBERTa-v2 8×480 architecture
3. Design a small mixed run: legal 10M compact-view stream + counted synthetic entity-state supplement
4. Evaluate unchanged natural interaction surfaces (EWoK transitions, GlobalPIQA hard) as readouts
