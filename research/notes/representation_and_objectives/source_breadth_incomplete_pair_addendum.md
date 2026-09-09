# An Incomplete Source-Breadth Pair Is Not a Negative Control Result

Status: initial pair invalidated by incomplete training; repaired comparison prepared, with no matched downstream result in this record.

The intended comparison replaced 17.5% of the corpus with cached FineWeb source-only text and used an official length-matched control. The first treatment produced only 32 training-log rows and a 1M checkpoint. The control produced no training-log rows because of an out-of-memory failure. Neither an adverse source-breadth effect nor a successful treatment effect can be inferred from this unmatched pair.

The repaired comparison retained the same corpus files, tokenizer, model architecture, optimizer, seeds, exposure and WWM recipe. Fresh output locations separated it from partial artifacts. A trainer-exact token/masking audit was required before interpreting the repeated comparison, followed by matched 10M-to-100M no-AoA trajectories.

The scientific question remained whether source-only breadth generated sufficient component-level movement to justify a four-arm family: natural text, independent breadth, same-source repetition and generated views. An execution failure before matched measurements did not answer that question and was not a reason to classify the source route as scientifically refuted.
