# ewok domain link interference-controlled state-update measurement

Status: **FROZEN**

Systematic ladder of 10 interference conditions applied to state-change scenarios.
Each base frame has one object, one actor, one state-change pair (open/closed or empty/full).
Conditions vary prior-state interference while holding the final update fixed.

- Base frames: 80 (50 open_closed, 30 empty_full)
- Conditions: 10
- Total frames: 800
- Renamed controls: 800
- Frame SHA256: `72761e94641db0367cb88a9351b46a93f76f8db41d7967329e7f4f32c768731f`
- Renamed SHA256: `e8ef52edd9fef689e3da2a1c50f71ff1d736f5af1cc88b30e829d8f92335aa24`
- Content SHA256: `4135fafc13c729cdda18ef27074a5be5e3776cd1acd2152a36abafea283dcbff`

## Conditions

| # | Condition | Description | Example (open_closed, box, Alice) |
|---|---|---|---|
| 1 | `no_context` | Pure target prior (empty context) | c1: "" / c2: "" |
| 2 | `explicit_final` | Direct statement of final state | c1: "The box is now open." / c2: "The box is now closed." |
| 3 | `last_event` | Single event, no prior state | c1: "Alice opened the box." / c2: "Alice closed the box." |
| 4 | `consistent_prior` | Prior agrees with final + event | c1: "The box was already open. Then Alice opened the box." / c2: "The box was already closed. Then Alice closed the box." |
| 5 | `distractor_event` | Unrelated distractor + final event | c1: "The ball fell on the floor. Alice opened the box." / c2: "The ball fell on the floor. Alice closed the box." |
| 6 | `contradict_bare` | Contradictory prior + final event | c1: "The box was closed. Alice opened the box." / c2: "The box was open. Alice closed the box." |
| 7 | `contradict_temporal` | Contradictory prior + 'Then' + final event | c1: "The box was closed. Then Alice opened the box." / c2: "The box was open. Then Alice closed the box." |
| 8 | `contradict_reinforced` | Contradictory prior + reinforcement + final event | c1: "The box was closed. The box stayed closed. Then Alice opened the box." / c2: "The box was open. The box stayed open. Then Alice closed the box." |
| 9 | `explicit_override` | Contradictory prior + explicit final statement | c1: "The box was closed. The box is now open." / c2: "The box was open. The box is now closed." |
| 10 | `reported_event` | Third-party report of final event | c1: "Sam said that Alice opened the box." / c2: "Sam said that Alice closed the box." |

## Interference hypothesis

counterfactual micro world v3 static repair D-state ablation found:
- `explicit_final` crossed=1.0 for all mature models
- `last_event_only` crossed=0.03 to 0.89, trajectory-dependent
- `initial_plus_last` (= contradict_bare) crossed=0.0 for all mature models
- `full_v3` (= contradict_reinforced approx) crossed=0.0 for all

The interference boundary is between no-prior-state and any-prior-state conditions.
This ladder tests whether specific interference mitigations (temporal markers,
explicit overrides, consistent evidence, reported framing, distractors) change
the boundary, and whether the pattern varies across training trajectories.

## Next: score existing checkpoints before any training

Frames: `experiments/archive/representation_and_objectives/data/interference_ladder/interference_ladder_frames.jsonl`
Renamed: `experiments/archive/representation_and_objectives/data/interference_ladder/interference_ladder_renamed_controls.jsonl`
Manifest: `experiments/archive/representation_and_objectives/data/interference_ladder/interference_ladder_manifest.json`
