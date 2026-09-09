# binding content trade predeclared predictions interim mechanism readout

Pre-declared mechanism note: `research/notes/frontier_consolidation/binding_content_trade_predeclared_predictions.md`
CPU-safe worker: `experiments/archive/frontier_consolidation/scripts/cpu_safe_scoring_worker.py`; probe status CPU_PROBE_OK, cpu_assert_seen=True

## MAX 100M EWoK now measured
- MAX view EWoK 100M: 50.15
- MAX repeat EWoK 100M: 49.88
- V-R: 0.269999999999996
- Current repeat-ready columns: ['BLiMP', 'Supplement', 'EWoK']; missing ['Entity', 'COMPS', 'Reading']

## 80M sampled margin pilot
- dose1 EWoK: n=55 mean=0.33854358875065704 median=0.6189367789775133 frac_positive=0.6 binary_delta_mean=0.10909090909090909
- dose1 Entity: n=90 mean=0.867337383302058 median=0.36170252662850544 frac_positive=0.5111111111111111 binary_delta_mean=-0.022222222222222223
- dose1p82 EWoK: n=55 mean=0.011078408611303365 median=-0.3559688590466976 frac_positive=0.45454545454545453 binary_delta_mean=-0.10909090909090909
- dose1p82 Entity: n=90 mean=0.2795949275411355 median=0.21994262095540762 frac_positive=0.5333333333333333 binary_delta_mean=0.03333333333333333
- dose2p64 EWoK: n=55 mean=-0.29596345308289695 median=0.1289599807932973 frac_positive=0.5272727272727272 binary_delta_mean=0.07272727272727272
- dose2p64 Entity: n=90 mean=0.9792498404043727 median=0.5343762714765035 frac_positive=0.5555555555555556 binary_delta_mean=0.08888888888888889

In the small 80M balanced sample, Entity V-R margins are positive at all three doses and largest at MAX, while EWoK V-R margin moves from positive at 1x to near zero at 1.82x and negative at MAX. This supports using a larger sampled trajectory as a graded signed-family test, but it is not a full-dataset result.

JSON: `experiments/archive/frontier_consolidation/data/interim_mechanism_readout/interim_mechanism_readout_summary.json`
