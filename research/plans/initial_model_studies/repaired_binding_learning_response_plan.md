# earlier analysis — repaired binding learning-response plan after binding learning response results code review

## What changed

The binding learning response results four-arm result cannot be used to decide whether downstream same-word-bag binding learning works.

Source inspection found that the script generated texts as:

```text
Alice placed a stone in the box. Bob placed a shell in the basket. Later Alice returned to the
```

rather than:

```text
Alice placed a stone in the box. Bob placed a shell in the basket. Later Alice returned to the box.
```

Because the target word was not appended to the query, the training and evaluation code searched for the last occurrence of the target and masked the setup occurrence, e.g.

```text
Alice placed a stone in the [MASK]. Bob placed a shell in the basket. Later Alice returned to the
```

The intended earlier analysis measurement was different:

```text
Alice placed a stone in the box. Bob placed a shell in the basket. Later Alice returned to the [MASK].
```

So binding learning response results measured setup-token reconstruction under swapped entity assignment, not downstream state retrieval from a visible setup.

## What remains true from binding learning response results

- The `true_binding` arm increased training-set directionality on the task it actually ran: training margin 0.0188, training accuracy 0.8906.
- That directionality failed when both names and state words changed: held-out margin -0.0143, accuracy 0.375.
- This supports the narrower statement that the current naive loss and data construction can memorize surface/template associations.

It does **not** establish that the intended downstream contrastive objective failed, because that objective was not actually run.

## Why no BabyLM-scale run is justified yet

The `wwm_correct` held-out change from 0.4688 to 0.5938 is too fragile to scale:

- 2-layer toy model, hidden 128;
- only 64 training pairs and 32 main held-out pairs;
- single seed;
- held-out mean margin only +0.0021;
- pre-training accuracy was below random;
- train-entity/new-location subset moved the opposite way: accuracy 0.2917, margin -0.0012;
- the measured target was the wrong setup occurrence;
- this weak signal conflicts with stronger controlled probes from structure density conclusion and objective pivot, 167, and 173.

Thus the route must not jump to a 10M binding-rich WWM mixture.

## Correct minimal rerun

Build `binding_learning_response_v2.py` or repair binding learning response results with these changes.

### Sample construction

For every pair, construct and store:

- `correct_prefix`: setup + query prefix;
- `swapped_prefix`: entity-assignment-swapped setup + the same query prefix;
- `target`: state/location/property token;
- `correct_text = correct_prefix + " " + target + "."`;
- `swapped_text = swapped_prefix + " " + target + "."`.

The target must appear in the downstream query in both texts. The setup target remains visible; only the query target is masked for the measurement and contrastive loss.

### Span handling

Do not locate the target by whole-sequence last-match. Record the query target span explicitly.

A robust method:

1. Use a fast tokenizer with offset mappings.
2. Store the character start/end of the appended query target.
3. Convert that character span to token positions.
4. Verify the target span belongs to the query, not setup.
5. Verify correct/swapped query target token IDs and positions match.
6. Verify correct/swapped tokenizer-ID multisets match after construction.
7. Verify truncation never removes the target span.

Print several human-readable masked examples in the output.

### Matched arms

Run exactly the same four arms:

1. `wwm_correct`: WWM on correct passages only.
2. `wwm_both`: WWM on equal counts of correct and swapped passages, no labels.
3. `random_pair`: WWM plus the same margin loss with random orientation.
4. `true_binding`: WWM plus the margin loss with correct context positive.

Use matched update count, masked-token exposure, tokenizer, initialization, and pre-generated batch/mask schedules across arms. Remove Python `hash(arm)` from seeds; use explicit integer seeds.

### Replication before scaling

Run at least 3 seeds on the repaired downstream-query task.

Increase held-out size substantially relative to binding learning response results, with separate splits:

- train entities + train states in new combinations;
- new entities + train states;
- train entities + new states;
- new entities + new states;
- held-out template families;
- location templates and property templates separately;
- overwritten-state and distractor-entity templates.

Save per-sample margins, not only aggregates.

### Needed result before any 10M work

Only proceed if the true-binding arm produces a stable increase in downstream query-slot margin and binding-choice accuracy across seeds and held-out splits, and this increase is clearly larger than `wwm_correct`, `wwm_both`, and `random_pair`.

If plain WWM also shows stable improvement over `wwm_both`, that supports a data-exposure route. If neither arm separates reliably, move to an architecture/state route rather than more data replacement.

## Current route judgment

- Do not scale binding learning response results to BabyLM 10M.
- Do not declare the intended contrastive objective failed.
- Do not accept the `wwm_correct` improvement as evidence for a large data route.
- The immediate next work is the repaired multi-seed downstream-query learning-response experiment, with attributable per-sample outputs.
