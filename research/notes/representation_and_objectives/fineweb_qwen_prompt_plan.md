# cached fineweb quality audit cached FineWeb Qwen rewrite prompt plan

Prepared prompts only; no Qwen generation was launched because both H100s are reserved for the semantic-view treatment/control contrast.

- Source candidate: `experiments/archive/representation_and_objectives/training/data/cached_fineweb_seqsafe96_candidate/cleanqwen_seqsafe_fineweb_single_doc_10M.jsonl`
- Selected sources: 4,096 across 4,096 docs, 365,696 words.
- Prompts: 8,192 = simplification + paraphrase for each source.
- Scientific purpose: if raw cached FineWeb source replacement is promising or semantic-view is weak, this prepares a legal reconstruction closer to the leader phenotype: FineWeb-derived text plus Qwen-generated simpler/restated views, with source consistency screening before training.
- It must remain distinct from the exact gated leader dataset; generated text counts toward BabyLM word exposure if used.

Metadata: `experiments/archive/representation_and_objectives/training/data/fineweb_qwen_prompts/prompt_metadata.json`

Prompts: `experiments/archive/representation_and_objectives/training/data/fineweb_qwen_prompts/fineweb_simplification_paraphrase_prompts.jsonl`
