# clean d component ablation parent/load validation

Parent SHA `e14d757ae51b41e33bf0813f841248fecd1eefeb9e040f520c4c6203343b15c8`; expected match: `True`.
Parent tensor file identical to earlier analysis alpha1 source tensor file: `True`.
Config private scale `0.75`; executed layer scales `[0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75]`.
Private key count `48`; missing keys `['cls.predictions.decoder.weight', 'cls.predictions.decoder.bias']`; unexpected keys `[]`.
Missing private keys: `[]`.

Interpretation: clean d component ablation can rely on the exact coherent86 alpha0.75 tensor/config parent only if the expected SHA, private-key presence, and executed scale checks are true.
