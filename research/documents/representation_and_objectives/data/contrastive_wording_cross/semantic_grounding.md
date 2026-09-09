# contrastive wording cross and grounding Semantic Grounding (pretrained BabyLM DeBERTa)

Cosine similarity between train-wording and held-wording representations
of the same facts, using the pretrained model BEFORE fine-tuning.

- **event_context_cosine**: mean=0.8207 ± 0.0414, min=0.7442, max=0.9171, n=240
- **state_context_cosine**: mean=0.6903 ± 0.0873, min=0.5403, max=0.8299, n=240
- **event_hypothesis_cosine**: mean=0.8561 ± 0.0166, min=0.8057, max=0.8904, n=60
- **state_hypothesis_cosine**: mean=0.6733 ± 0.0435, min=0.5815, max=0.7632, n=60
- **compound_context_cosine**: mean=0.9208 ± 0.0066, min=0.9084, max=0.9407, n=60

Elapsed: 16.88s on 60 worlds.

## Interpretation

If cosine similarity is high (>0.9), the pretrained model considers
train and held wordings as semantically equivalent representations.
Any fine-tuned held-wording failure would then be attributable to
template-specific calibration in the classification head, not to
representational non-equivalence in the pretrained encoder.
