# earlier analysis frame-varied recombination rows

Only the final query sentence is rephrased; source/update evidence, entities, source_state and new_state are preserved. Train-seen frames allow multi-form practice; eval-unseen frames test whether the readout escapes a single query wording.

| file | rows | pairs | words |
|---|---:|---:|---:|
| train_seen_frames | 16660 | 8330 | 789834 |
| train_all_frames | 26656 | 13328 | 1263068 |
| heldout_all_frames | 3200 | 1600 | 151344 |
| heldout_seen_frames | 2000 | 1000 | 94640 |
| heldout_unseen_frames | 1200 | 600 | 56704 |

## Frames

| frame | split | template |
|---|---|---|
| f00_original_relevant_state | train_seen | The relevant state of {entity} is {answer}. |
| f01_at_this_point | train_seen | At this point, {entity} is {answer}. |
| f02_for_current_state | train_seen | For {entity}, the current state is {answer}. |
| f03_currently_associated | train_seen | The state currently associated with {entity} is {answer}. |
| f04_given_context | train_seen | Given this context, {entity} should be described as {answer}. |
| f05_after_descriptions | eval_unseen | After these descriptions, {entity} is {answer}. |
| f06_by_the_end | eval_unseen | By the end of the passage, {entity} is {answer}. |
| f07_context_indicates | eval_unseen | The context indicates that {entity} is {answer}. |
