# compliance control interpretation and eval harness — clean-Qwen own-tokenizer diagnostic
Diagnostic tokenizer trained on the clean-Qwen 10M pool to quantify the fixed-tokenizer control interpretation. No model retraining is launched.

- Clean-own tokenizer SHA256: `2fac71fd07fb67d2de800eceb1b1b1ba657a7bb434dffcf378ded60e6297b7d3`; vocab 16384; specials {'<unk>': 0, '<s>': 1, '</s>': 2, '<pad>': 3, '<mask>': 4}.
- Vocab overlap with reinvest-trained compliant tokenizer: 15,605/16,384 = 0.9525 of reinvest vocab and 0.9525 of clean-own vocab.
- Clean pool total token ratio clean-own/reinvest-tokenizer: 0.999595; seq256 truncation 16756 -> 16730.
- Reinvest pool total token ratio clean-own/reinvest-tokenizer: 1.000517; seq256 truncation 15967 -> 15980.

Interpretation: the running clean-Qwen model is a fixed-reinvest-tokenizer scientific control only. This diagnostic says how large the tokenizer-corpus mismatch is in token geometry; it cannot replace a model result.
