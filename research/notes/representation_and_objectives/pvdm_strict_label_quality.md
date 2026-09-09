# pvdm compliance and control design — strict deterministic PVDM label quality

The first deterministic label pass was legally clean but scientifically too permissive: nearly every row was eligible and top targets included speaker tags. This stricter training-facing pass excludes learned NLP tools and uses only fixed lexical/window rules plus frequency bins from the allowed 10M compact pool.

## Main facts

- rows/words: 192,549 rows / 30,000,000 words in the 70M→100M tail
- eligible rows: 184,689 (0.959)
- accepted events: 628,344; unique dependent targets: 628,344
- target positions per 1000 words: 20.94
- identical dependent-target sequence for treatment/control: `fd78f6c82db80dcdac3e3b3397c1252a76c9ebc2918470999a6df329cef16e14`
- labels: `experiments/archive/representation_and_objectives/data/pvdm_strict_labels/pvdm_strict_tail_70M_100M_labels.jsonl`
- tail JSONL for continuation: `experiments/archive/representation_and_objectives/data/pvdm_strict_labels/compact_tail_70M_100M.jsonl`
- summary: `experiments/archive/representation_and_objectives/data/pvdm_strict_labels/pvdm_strict_label_summary.json`
- samples: `experiments/archive/representation_and_objectives/data/pvdm_strict_labels/pvdm_strict_event_samples.jsonl`

Use these strict labels, not the permissive `pvdm_deterministic_labels`, for any training-facing PVDM experiment. The trainer must still normalize masking so PVDM and control have identical expected/actual masked mass per batch and must run a standard-masking parity check before any 30M GPU continuation.
