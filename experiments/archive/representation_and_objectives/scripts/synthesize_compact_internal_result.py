#!/usr/bin/env python3
"""Synthesize research compact directional internal results into a decision note."""
from __future__ import annotations
import json, pathlib, time

PAIR = pathlib.Path('experiments/archive/representation_and_objectives/data/compact_pair_objective/pair_objective_probe.json')
NOEVAL = pathlib.Path('experiments/archive/representation_and_objectives/data/compact_directional_internal/directional_interaction_readout_noeval.json')
OUT_JSON = pathlib.Path('experiments/archive/representation_and_objectives/data/compact_directional_internal/internal_synthesis.json')
OUT_MD = pathlib.Path('research/notes/representation_and_objectives/compact_directional_internal_result.md')

pair = json.loads(PAIR.read_text())
noeval = json.loads(NOEVAL.read_text())
inter = pair['interactions']
learn = pair['learning_changes']
train_inter = noeval['training']['training_interactions']
integ = noeval['training']['integrity']

def g(path, *keys):
    x = path
    for k in keys:
        x = x[k]
    return x

summary = {
    'status': 'COMPACT_DIRECTIONAL_INTERNAL_SYNTHESIS',
    'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'integrity': integ,
    'data_meta': pair['data_meta'],
    'training_epoch2_interactions_loss_lower_is_better': train_inter,
    'pair_probe_interactions_loss_lower_is_better': inter,
    'central_numbers': {
        'forward_pair_side_I_loss': inter['forward']['side_loss']['mixed_minus_oneway_loss'],
        'forward_pair_copied_I_loss': inter['forward']['copied_loss']['mixed_minus_oneway_loss'],
        'forward_pair_noncopied_I_loss': inter['forward']['noncopied_loss']['mixed_minus_oneway_loss'],
        'reverse_pair_side_I_loss': inter['reverse']['side_loss']['mixed_minus_oneway_loss'],
        'reverse_pair_copied_I_loss': inter['reverse']['copied_loss']['mixed_minus_oneway_loss'],
        'reverse_pair_noncopied_I_loss': inter['reverse']['noncopied_loss']['mixed_minus_oneway_loss'],
        'forward_noncopied_minus_copied_I': inter['copied_vs_noncopied_interaction']['forward'],
        'reverse_noncopied_minus_copied_I': inter['copied_vs_noncopied_interaction']['reverse'],
        'epoch_summary_loss_I': train_inter['loss']['reciprocal_minus_oneway'],
        'epoch_summary_copied_I': train_inter['copied_loss']['reciprocal_minus_oneway'],
        'epoch_summary_noncopied_I': train_inter['noncopied_loss']['reciprocal_minus_oneway'],
    },
    'learning_change_ranges': learn,
    'interpretation': {
        'paired_objective_learned': 'Yes: every second-epoch continuation lowers the directly evaluated side-target loss in its trained direction by roughly 0.319-0.440 nats from the corresponding prefix; both copied and noncopied losses move down.',
        'mixed_direction_internal_effect': 'Small. Cross-evaluated pair losses give mixed-minus-oneway loss I=-0.0060 on forward targets and -0.0104 on reverse targets; noncopied targets improve more than copied targets, especially reverse orientation (noncopy-copy differential -0.0305).',
        'important_caution': 'The epoch-training summaries themselves do not show a favorable reciprocal effect (overall +0.00064, copied +0.00874, noncopied +0.02048 loss; lower is better). The favorable signal appears only in direct cross-orientation pair probing and is small, so it should be treated as an internal trajectory clue, not a mechanism result.',
        'scientific_read': 'The compact causal screen learned the source/side prediction objective and preserves some cross-direction knowledge, including a small noncopied-target interaction. This is enough to justify waiting for endpoint cheap7/EWoK-domain transfer and, if transfer is not null, copy-matched semantic/random controls. It is not enough to claim compact-specific reciprocal semantic exchange.'
    },
    'next_decision_rules': {
        'if_endpoint_transfer_absent': 'If cheap7/EWoK-domain directional interaction is absent or negative, close this 20M compact reciprocal-causal instantiation: internal pair learning did not transfer to the relevant BabyLM surface strongly enough.',
        'if_endpoint_relational_signal_positive': 'If research relational EWoK domains or cheap7 show a meaningful positive I, run the same v4 fork protocol for semantic_extract and random_extract controls before attributing specificity to compact faithful rewrites.',
        'if_endpoint_mixed_or_weak_positive': 'If endpoint is weak but internal noncopied I remains the only positive clue, do not launch broad controls immediately; consider a bounded continuation/checkpointed extension only if it can distinguish delayed transfer from null more cheaply than two full copy-matched controls.',
        'never_interpret_alone': 'Do not use FR-FF, RF-RR, or training loss alone as the estimand; use I=0.5*(FR+RF)-0.5*(FF+RR) and compare to copy-matched controls for compact specificity.'
    },
    'relation_context_constraint': {
          'a02_step183_source_conditioned_ordering': 'A02 found large source-conditioned ordering interactions in frozen DeBERTa for compact, prefix, and onegap families, with compact the smallest I_f because 21.7% of compact targets are source-absent. This strengthens the alternative explanation that compact-view gains come from general ordered source retrieval plus tail coverage, budget efficiency, and content density, not a compact-specific semantic recoding or reciprocal mechanism.'
      },
      'paths': {'pair_probe': str(PAIR), 'noeval_readout': str(NOEVAL), 'endpoint_eval_task_failed_path_mode': 's215_t10_tool1', 'endpoint_eval_task_rerun': 's215_t13_tool1', 'endpoint_eval_root_rerun': 'experiments/archive/representation_and_objectives/data/compact_directional_cheap7_eval_r2'}
}
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

cn = summary['central_numbers']
md = f"""# research compact directional internal result\n\n## What was checked\n\nBoth compact fork branches completed with exact shared prefixes within branch pairs. The internal readout uses the intended directional interaction\n\n`I_loss = 0.5*(FR + RF) - 0.5*(FF + RR)`\n\nwhere lower loss is better. The pair-objective probe directly evaluated every trained model on the 12,155 compact pairs in both causal orientations, separated into target tokens that were copied from the context segment and target tokens not present in that context.\n\n## Main internal numbers\n\n- Training epoch-summary interaction (loss, lower better): overall `{cn['epoch_summary_loss_I']:.6f}`, copied `{cn['epoch_summary_copied_I']:.6f}`, noncopied `{cn['epoch_summary_noncopied_I']:.6f}`. This is not favorable.\n- Direct pair-probe forward target I: side `{cn['forward_pair_side_I_loss']:.6f}`, copied `{cn['forward_pair_copied_I_loss']:.6f}`, noncopied `{cn['forward_pair_noncopied_I_loss']:.6f}`.\n- Direct pair-probe reverse target I: side `{cn['reverse_pair_side_I_loss']:.6f}`, copied `{cn['reverse_pair_copied_I_loss']:.6f}`, noncopied `{cn['reverse_pair_noncopied_I_loss']:.6f}`.\n- Noncopied-minus-copied I: forward `{cn['forward_noncopied_minus_copied_I']:.6f}`, reverse `{cn['reverse_noncopied_minus_copied_I']:.6f}`. Negative means the mixed schedule helps noncopied targets more than copied targets.\n\n## Scientific interpretation\n\nThe paired causal objective was learned: all trained second epochs lower the directly evaluated side-target losses in their trained direction by large amounts relative to the corresponding prefix, including noncopied targets. The mixed-direction signal is real enough to track but small. It appears most clearly in direct cross-orientation pair probing, especially reverse noncopied targets; it is not visible as a favorable aggregate in the raw epoch training summaries. Therefore the compact-only 20M run is an internal trajectory clue, not a compact-specific mechanism result. A02's research source-conditioned ordering result further weakens any compact-specific reading: ordered source conditioning is large for compact, prefix, and onegap views, and compact has the smallest frozen I_f because many rewrite targets are source-absent. Any positive endpoint transfer must therefore be tested against copy-matched extractive controls before being interpreted as compact semantic exchange.\n\n## Decision rule for the pending endpoint readout\n\nEndpoint scoring was first attempted as `s215_t10_tool1` but failed before execution because the managed runtime started inside a session path while the command used full `Sessions/...` paths. It was relaunched from explicit root cwd as `s215_t13_tool1` and will write under `experiments/archive/representation_and_objectives/data/compact_directional_cheap7_eval_r2`.\n\n- If cheap7/EWoK-domain transfer is absent, close this 20M compact reciprocal-causal instantiation.\n- If the research relational domains or cheap7 show a meaningful positive directional interaction, run copy-matched `semantic_extract` and `random_extract` v4 fork controls before interpreting compact specificity.\n- If endpoint transfer is weak but the internal noncopied effect is the only favorable sign, consider only a bounded continuation if it is cheaper and more decisive than full controls for distinguishing delayed transfer from null.\n\nJSON evidence: `{OUT_JSON}`. Pair-probe evidence: `{PAIR}`.\n"""
OUT_MD.parent.mkdir(parents=True, exist_ok=True)
OUT_MD.write_text(md, encoding='utf-8')
print(json.dumps({'status': summary['status'], 'json': str(OUT_JSON), 'md': str(OUT_MD), 'central_numbers': cn}, indent=2), flush=True)
