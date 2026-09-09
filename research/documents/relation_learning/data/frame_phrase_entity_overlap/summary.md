# binding transfer and composition prediction frame phrase / Entity wording comparison

The binding frames use state/context/passage phrasings and no box/contains/move/remove/put/nothing wording; they are not near copies of the official Entity container query surface.

| frame | split | container-keyword? | Entity tail overlap | template |
|---|---|---:|---|---|
| f00_original_relevant_state | train_seen | 0 | none | The relevant state of {entity} is {answer}. |
| f01_at_this_point | train_seen | 0 | none | At this point, {entity} is {answer}. |
| f02_for_current_state | train_seen | 0 | none | For {entity}, the current state is {answer}. |
| f03_currently_associated | train_seen | 0 | none | The state currently associated with {entity} is {answer}. |
| f04_given_context | train_seen | 0 | none | Given this context, {entity} should be described as {answer}. |
| f05_after_descriptions | eval_unseen | 0 | none | After these descriptions, {entity} is {answer}. |
| f06_by_the_end | eval_unseen | 0 | none | By the end of the passage, {entity} is {answer}. |
| f07_context_indicates | eval_unseen | 0 | none | The context indicates that {entity} is {answer}. |

Official Entity tail prompt examples:
- `Box 1 contains`
- `Box 3 contains`
- `Box 0 contains`
- `Box 2 contains`
- `Box 5 contains`
- `Box 6 contains`
- `Box 4 contains`
