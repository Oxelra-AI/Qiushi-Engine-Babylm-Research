# adjusted source use residual: adjusted source-use residual — compact does not win by copied-token retrieval

## What this step tested
After the earlier analysis closure of the compact causal-reciprocity route, the surviving mechanism object is masked-denoising cross-view visibility under budget pressure (note 217). Before committing H100 time to a boundary-blocked MLM visibility intervention, this step used existing hypothesis comparison frozen chck82 source-use records (44,995 target rows, three families: compact, prefix-fluent, sourcewide-onegap, each vs its scrambled control) to answer a cheap, decisive question:

> Does compact have a hidden stronger source-conditioned ordering interaction (I_f) once copy opportunity is matched, or is its raw lower I_f simply composition (compact has many source-absent targets)?

I_f = [NLL(V_scr|S) − NLL(V_ord|S)] − [NLL(V_scr) − NLL(V_ord)]: positive means ordered structure specifically helps source-conditioned reconstruction beyond generic fluency.

Artifacts:
- `scripts/adjusted_source_use_probe_analysis.py`
- `scripts/adjusted_source_use_fast_bootstrap.py`
- `data/adjusted_source_use_probe/adjusted_source_use_probe.json` / `.md`
- `data/adjusted_source_use_probe/adjusted_source_use_fast_bootstrap.json` / `.md`

## Composition explains the raw gap
Raw family mean I_f: compact 2.748, prefix 3.465, onegap 3.595. But compact's target set is very different:
- absent-target fraction: compact 0.217 vs prefix 0.005 vs onegap 0.005
- complete-BPE-copy fraction: compact 0.647 vs prefix/onegap 0.994

Restricting to copied targets already shrinks the gap (copied mean I_f: compact 3.257, prefix 3.451, onegap 3.582).

## Adjusted result: compact copied-target residual is negative, tight
Common-support direct standardization on copied targets, matched over copy zone, source-match multiplicity, lexical class, source-position decile, min-visible-gap, and ordinal position, with a fixed-support pair-cluster bootstrap (1000 resamples, ~92% retain full support):

- compact − mean(extractives): observed −0.495, median −0.491, 95% [−0.605, −0.384], p>0 = 0.000
- compact − prefix: median −0.622, 95% [−0.741, −0.505], p>0 = 0.000
- compact − onegap: median −0.362, 95% [−0.504, −0.235], p>0 = 0.000
- Using compact's own copied-target distribution as the standard: compact − mean(extractives) median −0.425, 95% [−0.547, −0.313], p>0 = 0.000
- Coarser strata (drop gap/ord-pos): compact − mean(extractives) median −0.567, 95% [−0.670, −0.464], p>0 = 0.000

All three strata resolutions agree: on matched copied targets, compact's ordered source-use interaction is *lower* than the extractive families, not higher, and the interval is comfortably away from zero.

Compact source-absent targets (n=3251) keep only weak positive interaction: mean I_f 0.908 (source_effect_ord 0.872, source_effect_scr −0.036) — consistent with weak semantic/context association, not direct retrieval.

## Scientific meaning
This is the strongest cheap evidence yet against a "compact wins by stronger ordered copied-token retrieval" story:
1. The frozen chck82 model retrieves copied source content *less* effectively per matched target from compact views than from fluent/onegap extractive views.
2. Compact's distinctive material is its source-absent rewrite content (21.7% of targets), which carries only weak reconstruction interaction.
3. Therefore the downstream compact-view training advantage (earlier analysis +2.4164 view−repeat; correctness transition analysis Component B relational/Supplement gains) cannot be explained as the model learning to copy ordered source tokens better. If anything the reconstruction-shortcut channel is weaker in compact.

This directly supports the note-217 abstraction hypothesis over the pure ordered-retrieval hypothesis, but as frozen-probe evidence it bounds, not proves, the training mechanism.

## Consequence for the next intervention
The prospective MLM boundary-blocked cross-view visibility intervention (note 217) should:
- Separate copied vs source-absent (noncopied) target losses explicitly. The prediction geometry now differs from earlier retrieval framing: if compact's benefit is source-absent semantic supervision, blocking cross-view attention should hurt source-absent target *learning transfer* more than copied-target reconstruction, and the extractive (copied-only) arms should not reproduce compact's relational/Supplement transitions even with matched tail coverage.
- Not assume copied-token retrieval is the load-bearing channel; the frozen probe says copied retrieval is if anything a disadvantage for compact.

## Caveats
- Frozen chck82 reconstruction probe, single model; measures reconstruction ability, not the causal training signal.
- Copied targets are matched; source-absent compact targets have no extractive-family common support, so the source-absent comparison is within-compact only.
- This does not by itself justify H100 spend; it sharpens the design and the copied/noncopied readout for evaluating the next mechanism route.
