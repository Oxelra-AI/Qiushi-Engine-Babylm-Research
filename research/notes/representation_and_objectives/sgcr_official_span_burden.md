# sgcr official span burden — exact-prefix SGCR residual burden in official answer text

CPU-only measurement. It uses the repaired exact BPE-prefix legal40k→legal16k decomposition and K=50 rho weights from the SGCR training code. It does not read or query the running SGCR training/evaluation tasks.

The quantity `mean residual` is average `(1-rho_t)` over token occurrences in a span. For K=50, tokens with legal40k pool count below 50 receive more component-path weight than standard-row weight. `component>=50 among low<50` asks whether low-count legal40k tokens are decomposed into legal16k components each seen at least 50 times in the same allowed 10M pool.

| family | items | disc token share | shared mean residual | disc mean residual | residual ratio | shared frac<50 | disc frac<50 | frac<50 ratio | disc share of low<50 residual | disc low<50 component>=50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EWoK | 7618 | 0.1050 | 0.1180 | 0.1341 | 1.1363 | 0.0878 | 0.0865 | 0.9845 | 0.1089 | 0.9232 |
| GlobalPIQA | 203 | 0.2243 | 0.0769 | 0.1169 | 1.5195 | 0.0382 | 0.0713 | 1.8669 | 0.3545 | 0.9826 |
| GlobalPIQA_parallel | 103 | 0.3174 | 0.0522 | 0.1015 | 1.9445 | 0.0206 | 0.0552 | 2.6791 | 0.5607 | 0.9859 |
| GlobalPIQA_nonparallel | 100 | 0.1044 | 0.1013 | 0.1773 | 1.7513 | 0.0554 | 0.1341 | 2.4207 | 0.2217 | 0.9773 |
| COMPS | 91028 | 0.2187 | 0.0895 | 0.2326 | 2.5988 | 0.0640 | 0.1871 | 2.9225 | 0.4510 | 0.9847 |
| Entity | 6780 | 0.0727 | 0.0872 | 0.0897 | 1.0287 | 0.0030 | 0.0159 | 5.2339 | 0.2909 | 1.0000 |

## Reading for the pending endpoint

- The exact-prefix SGCR K=50 run is most directly aimed at candidate-varying tokens when the discriminating spans carry much higher residual pressure than shared spans and their low-count tokens are usually component-supported.
- COMPS has the strongest measured alignment: discriminating spans have about 2.6× the mean residual of shared spans, while their below-50 support fraction is about 2.92× the shared fraction; almost all low-count discriminating occurrences decompose into well-supported legal16k components. A real endpoint should therefore move COMPS if the mechanism is effective.
- GlobalPIQA is weaker but still aligned: candidate solutions carry about 1.5× the shared residual pressure in aggregate, with both parallel and nonparallel aligned. Parallel and nonparallel should be read separately because prior vectors showed they can move differently, not because the burden map predicts nonparallel should benefit more.
- Entity has a high low<50 enrichment in answer options after the official skip, but its mean residual ratio is modest because many shared prefix tokens sit in the 50–99 support band. A positive Entity movement would support the component-sharing story; a flat Entity vector would not by itself refute SGCR.
- EWoK remains poorly aligned with this mechanism: candidate-varying context tokens are not meaningfully more low-supported or residual-heavy than shared tokens. Large EWoK recovery should not be attributed to simple rare-token sharing without further evidence.

JSON: `experiments/archive/representation_and_objectives/data/sgcr_official_span_burden/sgcr_official_span_burden.json`
CSV: `experiments/archive/representation_and_objectives/data/sgcr_official_span_burden/sgcr_official_span_burden_by_group.csv`
