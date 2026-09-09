# effective input repaired contrast result effective-input audit for mechanism macro convergence repetition-vs-variation

PosAlign hard matching replaces each matched surface name by the same candidate/other parameter vectors before the GRU. Raw name identity therefore does not reach the relational learner when matching succeeds. The audit below canonicalizes each event by replacing the two names with `<C>` and `<O>`.

## Pattern counts after name replacement

| set | events | unique effective patterns | unique patterns after object abstraction | voices | objects |
|---|---:|---:|---:|---|---:|
| fixed_train_comparison_events | 384 | 200 | 14 | active:192, passive:192 | 16 |
| vary_context_e1_events | 384 | 161 | 8 | active:194, passive:190 | 24 |
| vary_context_e1_e3_events | 1152 | 190 | 8 | active:560, passive:592 | 24 |
| vary_noise_e1_events | 384 | 383 | 369 | active:192, passive:192 | 16 |
| vary_noise_e1_e3_events | 1152 | 1144 | 1024 | active:576, passive:576 | 16 |
| train_changed_state_events | 288 | 48 | 36 | active:144, passive:144 | 16 |
| eval_changed_state_events | 384 | 80 | 8 | active:192, passive:192 | 16 |

## Set overlap with eval and with fixed train

| comparison | space | A | B | intersection | A minus B | B minus A | Jaccard |
|---|---|---:|---:|---:|---:|---:|---:|
| fixed_train_comparison_events__vs__eval_changed_state_events | effective | 200 | 80 | 0 | 200 | 80 | 0.000 |
| fixed_train_comparison_events__vs__eval_changed_state_events | abstract_object | 14 | 8 | 8 | 6 | 0 | 0.571 |
| fixed_train_comparison_events__vs__vary_context_e1_events | effective | 200 | 161 | 0 | 200 | 161 | 0.000 |
| fixed_train_comparison_events__vs__vary_context_e1_events | abstract_object | 14 | 8 | 7 | 7 | 1 | 0.467 |
| fixed_train_comparison_events__vs__vary_noise_e1_events | effective | 200 | 383 | 0 | 200 | 383 | 0.000 |
| fixed_train_comparison_events__vs__vary_noise_e1_events | abstract_object | 14 | 369 | 0 | 14 | 369 | 0.000 |
| vary_context_e1_events__vs__eval_changed_state_events | effective | 161 | 80 | 2 | 159 | 78 | 0.008 |
| vary_context_e1_events__vs__eval_changed_state_events | abstract_object | 8 | 8 | 4 | 4 | 4 | 0.333 |
| vary_noise_e1_events__vs__eval_changed_state_events | effective | 383 | 80 | 0 | 383 | 80 | 0.000 |
| vary_noise_e1_events__vs__eval_changed_state_events | abstract_object | 369 | 8 | 0 | 369 | 8 | 0.000 |

## Top fixed comparison effective patterns

- `<bos> during the notebook episode , <C> daxed <O> . <eos>` ×4; raw example: During the notebook episode, Lena daxed Tomas.
- `<bos> during the notebook episode , <C> was daxed by <O> . <eos>` ×4; raw example: During the notebook episode, Tomas was daxed by Lena.
- `<bos> during the teapot episode , <C> daxed <O> . <eos>` ×4; raw example: During the teapot episode, Pavel daxed Keira.
- `<bos> during the teapot episode , <C> was daxed by <O> . <eos>` ×4; raw example: During the teapot episode, Keira was daxed by Pavel.
- `<bos> during the camera episode , <C> daxed <O> . <eos>` ×4; raw example: During the camera episode, Rina daxed Noel.
- `<bos> during the camera episode , <C> was daxed by <O> . <eos>` ×4; raw example: During the camera episode, Noel was daxed by Rina.
- `<bos> during the blanket episode , <C> daxed <O> . <eos>` ×4; raw example: During the blanket episode, Tomas daxed Rina.
- `<bos> during the blanket episode , <C> was daxed by <O> . <eos>` ×4; raw example: During the blanket episode, Rina was daxed by Tomas.
- `<bos> during the basket episode , <C> daxed <O> . <eos>` ×4; raw example: During the basket episode, Nia daxed Jonas.
- `<bos> during the basket episode , <C> was daxed by <O> . <eos>` ×4; raw example: During the basket episode, Jonas was daxed by Nia.
- `<bos> during the tablet episode , <C> daxed <O> . <eos>` ×4; raw example: During the tablet episode, Felix daxed Mira.
- `<bos> during the tablet episode , <C> was daxed by <O> . <eos>` ×4; raw example: During the tablet episode, Mira was daxed by Felix.

## Top eval changed-state effective patterns

- `<bos> during the hammer episode , <C> meped <O> . <eos>` ×8; raw example: During the hammer episode, Tara meped Ben.
- `<bos> during the hammer episode , <C> was meped by <O> . <eos>` ×8; raw example: During the hammer episode, Ben was meped by Tara.
- `<bos> during the shell episode , <C> meped <O> . <eos>` ×8; raw example: During the shell episode, Simon meped Vera.
- `<bos> during the shell episode , <C> was meped by <O> . <eos>` ×8; raw example: During the shell episode, Vera was meped by Simon.
- `<bos> during the drum episode , <C> meped <O> . <eos>` ×8; raw example: During the drum episode, Yara meped Arun.
- `<bos> during the drum episode , <C> was meped by <O> . <eos>` ×8; raw example: During the drum episode, Arun was meped by Yara.
- `<bos> during the booklet episode , <C> meped <O> . <eos>` ×8; raw example: During the booklet episode, Ben meped Yara.
- `<bos> during the booklet episode , <C> was meped by <O> . <eos>` ×8; raw example: During the booklet episode, Yara was meped by Ben.
- `<bos> during the goblet episode , <C> norped <O> . <eos>` ×8; raw example: During the goblet episode, Caleb norped Zara.
- `<bos> during the goblet episode , <C> was norped by <O> . <eos>` ×8; raw example: During the goblet episode, Zara was norped by Caleb.
- `<bos> during the key episode , <C> norped <O> . <eos>` ×8; raw example: During the key episode, June norped Eli.
- `<bos> during the key episode , <C> was norped by <O> . <eos>` ×8; raw example: During the key episode, Eli was norped by June.

## Consequence for repaired experiment

Name variation in VARY_CONTEXT is not an effective-input manipulation under hard PosAlign. The remaining visible changes are object lexical identity, voice/order, and added filler tokens; the existing base comparison set already contains many objects and both active/passive voices. A stronger contrast must manipulate visible nuisance tokens that can actually compete with the relational solution, or remove the supplied name-invariance and compare learning the matcher with using a frozen matcher.
