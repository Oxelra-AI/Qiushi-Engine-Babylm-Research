# sequence curriculum loop measurement unassigned space-token probe
CPU-only characterization of offset-unassigned tokens in the sequence curriculum loop measurement sequence/pair chunking audits.

Pool SHA matched: `True`.

| tokenizer | scope | rows | tokens | unassigned | fraction | top token | top substring |
|---|---|---:|---:|---:|---:|---|---|
| legal16k | full_pool | 64740 | 14664519 | 86333 | 0.0059 | Ġ:86333 | ' ':86333 |
| legal16k | changed_rows | 3005 | 607898 | 5197 | 0.0085 | Ġ:5197 | ' ':5197 |
| minfreq50_supportfloor | full_pool | 64740 | 14483600 | 85864 | 0.0059 | Ġ:85864 | ' ':85864 |
| minfreq50_supportfloor | changed_rows | 3005 | 592605 | 5161 | 0.0087 | Ġ:5161 | ' ':5161 |

## Scientific reading
The unassigned offsets are overwhelmingly standalone separator-space byte-level tokens (`Ġ`, substring `' '`). They enter the trainer's token stream but are not semantic word content. Earlier offset-based word/pair chunk lengths are therefore slightly optimistic by about 0.6% on the full pool and about 0.8% on changed compact-pair rows. This does not change the qualitative conclusion that prefix slicing hides large suffix fractions or that pair-aware chunking is needed, but an eventual faithful sequence trainer should chunk the tokenizer group stream directly rather than reconstruct chunks only from whitespace words.

Full JSON: `experiments/archive/frontier_consolidation/data/unassigned_space_token_probe/unassigned_space_token_probe.json`
