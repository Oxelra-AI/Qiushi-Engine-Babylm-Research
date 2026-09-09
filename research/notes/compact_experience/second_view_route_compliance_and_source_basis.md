# second view route compliance and source basis — compliance/source basis for generated second-view route

The information-efficient second-view route still uses local Qwen-generated text, so compliance remains a live research condition rather than an assumption to hide until final submission.

Current local official-source evidence:

- `data/external/FAQs.md`, lines 114-116: BabyLM FAQ says any training objective/regime is permitted as long as data restrictions are followed, and if ancillary models are used for reranking or data augmentation, the training data for these models is counted toward the 100M word budget.
- Same source, lines 126-128: external tools are allowed; if learned on language, their tokens count toward the 100M. The FAQ explicitly says synthetic data is allowed under restrictions and gives examples of generated text from models trained within the word budget.
- Same source, lines 139-141: approved teacher families include Qwen 2.5, Qwen 3, and Qwen 3.5 up to 9B parameters.

Recorded interpretation for this experiment:

1. The small second view route compliance and source basis panel is a research audit of a generated-data mechanism, not a submission claim.
2. If a Qwen-generated transformation family becomes a candidate for official submission, the final data accounting must state the generated words inside the 10M training corpus and the 100M exposure budget, and must preserve Qwen model identity, prompts, decoding settings, input/output logs, filtering code, accepted/rejected records, and corpus hashes.
3. The submission model must not receive Qwen weights, hidden states, output distributions, or any signal beyond generated text included in the counted training corpus.
4. No official AoA/CDI words, per-item scores, child curves, checkpoint acquisition trajectories, or predictions may be used for data generation, filtering, training, or route selection. AoA remains only a final aggregate measurement on frozen candidates.
5. The FAQ text supports continued research on Qwen-generated transformations but does not by itself finish final submission compliance: before final packaging, re-read the current call/guidelines/FAQ pages and archive exact versions, including any track-specific wording about synthetic data, external models, distillation, and epoch counting.
