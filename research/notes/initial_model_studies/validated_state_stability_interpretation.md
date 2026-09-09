# validated state stability interpretation validated-state stability smoke — interpretation

## Purpose

Selecting FineWeb passages by `R_extra > 0` under the current best checkpoint could simply amplify old model preferences: exact filler repetition, static templates, and lexical co-occurrence. The validated state stability interpretation smoke added two stronger requirements before any 1M materialization:

1. compare `R_extra` against a filler-only repeat effect;
2. require direction stability across WWM seed/stage checkpoints, not only `wwm43_80M`.

## Evidence files

- Script: `scripts/validated_state_stability_smoke.py`
- 1.5k pilot: `data/validated_state_stability_smoke_pilot.json`, `notes/validated_state_stability_smoke_pilot.md`
- 5k run: `data/validated_state_stability_smoke_5k.json`, `notes/validated_state_stability_smoke_5k.md`

## 1.5k pilot

- 1,500 streamed docs; 1,064 basic-quality docs.
- Only 3 thread cases extracted.
- All-model `R_extra > 0`: 2/3.
- All-model `relation_over_filler > 0`: 1/3.
- The stable positive case was `english / letters`, a florin/letter template rather than an entity state transition.

## 5k run

- 5,000 streamed docs; 3,554 basic-quality docs.
- 19 cases extracted; 18 common across all scored checkpoints.
- All-model `R_extra > 0`: 10/18 = 0.556.
- All-model `relation_over_filler > 0`: 7/18 = 0.389.

Per-model means over 18 used cases:

| model | R_extra mean | R_extra frac>0 | filler repeat mean | relation_over_filler mean | relation_over_filler frac>0 |
|---|---:|---:|---:|---:|---:|
| wwm43_40M | +0.7419 | 0.667 | +0.0409 | +0.7010 | 0.500 |
| wwm43_80M | +1.8795 | 0.667 | +0.2091 | +1.6703 | 0.556 |
| wwm43_100M | +1.8310 | 0.611 | +0.2747 | +1.5563 | 0.556 |
| wwm42_100M | +1.3862 | 0.556 | +0.1885 | +1.1977 | 0.556 |

## Why this does not justify 1M validated-state materialization

The numeric sign is not enough. The yielded examples show that the extractor still treats many static or spurious spans as entities/states:

- `english / letters`: English florins called because letters were omitted — strong but template/lexical, not narrative entity state.
- `parts / causing`: a geological phrase where “Parts ... dropped off, causing ...” is event wording but not trackable entity state.
- `christ / right`: religious panel text with repeated right/left and Christ, likely layout/template language.
- `theobald / involved`: family became involved in Spiritualism; closer to a relation, but still not an object-location/holder state update.
- `standing / rail`, `isometric / rail`, `streets / center`, `avenue / center`, `march / sites`: clear failures of entity typing.
- Several zero-effect rows (`nasa/northern`, `africa/lake`, `cockroaches/head`) show static descriptive facts rather than predictive state use.

Thus the relation-over-filler test is useful and stricter than `R_extra`, but the current extractor is not producing enough high-quality narrative state examples. Scaling it to 1M would mostly amplify old model sensitivity to lexical/template repetition and static factual patterns, not construct the missing Entity Tracking ability.

## Scientific conclusion

Do not materialize `validated_state_1M` from the current validated state stability interpretation extractor. The route needs either:

1. a much stronger event/thread representation with real entity typing and alias/pronoun handling, followed by the same relation-over-filler and cross-model stability test; or
2. a different data mechanism, such as controlled official-legal generated state-transition microstories or paired simplification where state variables are explicit and can be verified, with all generated words counted under the BabyLM budget.

The fineweb relation vs random 1m profile static-relation filter remains a real EWoK signal (+3.27 over same-source random FineWeb), but validated state stability interpretation shows that the current attempt to turn it into an Entity mechanism by model-selected repeated-relation passages is not ready for training.
