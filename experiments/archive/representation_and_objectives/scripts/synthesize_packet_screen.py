#!/usr/bin/env python3
"""Synthesize research role-switch packet screen evidence."""
import json
import pathlib

ROOT = pathlib.Path('.')
PKT = ROOT / 'experiments/archive/representation_and_objectives/data/role_switch_expanded_packets/manifest.json'
LEARN = ROOT / 'experiments/archive/representation_and_objectives/data/disposable_packet_learning_screen/learning_screen_summary.json'
EWOK = ROOT / 'experiments/archive/representation_and_objectives/data/screen_ewok_interaction_reader/screen_ewok_interaction_summary.json'
GP_DIR = ROOT / 'experiments/archive/representation_and_objectives/data/screen_globalpiqa_margin_reader'
OUT_JSON = ROOT / 'experiments/archive/representation_and_objectives/data/packet_screen_synthesis/packet_screen_synthesis.json'
OUT_NOTE = ROOT / 'research/notes/representation_and_objectives/role_switch_packet_screen_synthesis.md'


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def gp_summary(target):
    obj = load(GP_DIR / f'{target}_margins.json')
    return {mode: md['summary'] for mode, md in obj['modes'].items()}


def gp_delta(base, target):
    out = {}
    for mode in sorted(set(base) & set(target)):
        b, t = base[mode], target[mode]
        d = {
            'accuracy': t['accuracy'] - b['accuracy'],
            'chance_adjusted_accuracy': t['chance_adjusted_accuracy'] - b['chance_adjusted_accuracy'],
            'all_rows_mean_top_minus_correct': t['all_rows_margin_summary']['mean_top_minus_correct'] - b['all_rows_margin_summary']['mean_top_minus_correct'],
            'all_rows_small_wrong_le_0p50': t['all_rows_margin_summary']['small_wrong_margin_le_0p50_nats'] - b['all_rows_margin_summary']['small_wrong_margin_le_0p50_nats'],
        }
        if b.get('always_wrong_subset') is not None and t.get('always_wrong_subset') is not None:
            d['always_wrong_subset_accuracy'] = t['always_wrong_subset']['accuracy'] - b['always_wrong_subset']['accuracy']
            d['always_wrong_subset_mean_top_minus_correct'] = t['always_wrong_subset']['mean_top_minus_correct'] - b['always_wrong_subset']['mean_top_minus_correct']
            d['always_wrong_subset_small_wrong_le_0p50'] = t['always_wrong_subset']['small_wrong_margin_le_0p50_nats'] - b['always_wrong_subset']['small_wrong_margin_le_0p50_nats']
        out[mode] = d
    return out


def main():
    pkt = load(PKT)
    learn = load(LEARN)
    ewok = load(EWOK)
    gp_base = gp_summary('base_reheat')
    gp_treat = gp_summary('screen_treatment')
    gp_fixed = gp_summary('screen_role_fixed')

    def lsum(label):
        return learn['summaries'][label]
    synth = {
        'status': 'PACKET_SCREEN_SYNTHESIS',
        'legal_status': 'No submission-facing endpoint: disposable screen starts from a 100M checkpoint and adds packet text after corpus consumption. Any legal route must train tokenizer/model from a revised <=10M corpus from the beginning.',
        'expanded_packet_suite': {
            'train_words_per_arm': pkt['treatment_train']['words'],
            'train_texts_per_arm': pkt['treatment_train']['n_texts'],
            'train_families': pkt['train_families'],
            'excluded_train_families': pkt['excluded_train_families'],
            'provenance_overlap_7_8_10': {k: pkt['provenance'][k]['overlap_count'] for k in ['7gram','8gram','10gram']},
            'treatment_control_word_match': pkt['treatment_train']['words'] == pkt['role_fixed_control_train']['words'],
            'target_counts_treatment': pkt['target_counts_treatment'],
            'target_counts_role_fixed': pkt['target_counts_role_fixed'],
        },
        'packet_exchange_screen': {
            'base_overall_both_correct': lsum('base_reheat')['overall']['both_correct'],
            'treatment_overall_both_correct': lsum('treatment')['overall']['both_correct'],
            'role_fixed_overall_both_correct': lsum('role_fixed')['overall']['both_correct'],
            'treatment_minus_role_fixed_overall_both_correct': learn['deltas']['treatment_minus_role_fixed']['overall']['both_correct'],
            'treatment_minus_role_fixed_M_mean': learn['deltas']['treatment_minus_role_fixed']['overall']['M_mean'],
            'heldout_style_both_correct': {
                'base': lsum('base_reheat')['by_style']['held_out']['both_correct'],
                'treatment': lsum('treatment')['by_style']['held_out']['both_correct'],
                'role_fixed': lsum('role_fixed')['by_style']['held_out']['both_correct'],
                'treatment_minus_role_fixed': learn['deltas']['treatment_minus_role_fixed']['by_style_both_correct']['held_out'],
            },
            'container_family_both_correct': {
                'base': lsum('base_reheat')['by_family']['container']['both_correct'],
                'treatment': lsum('treatment')['by_family']['container']['both_correct'],
                'role_fixed': lsum('role_fixed')['by_family']['container']['both_correct'],
                'treatment_minus_role_fixed': learn['deltas']['treatment_minus_role_fixed']['by_family_both_correct']['container'],
            },
            'temporal_family_both_correct': {
                'base': lsum('base_reheat')['by_family']['temporal']['both_correct'],
                'treatment': lsum('treatment')['by_family']['temporal']['both_correct'],
                'role_fixed': lsum('role_fixed')['by_family']['temporal']['both_correct'],
                'treatment_minus_role_fixed': learn['deltas']['treatment_minus_role_fixed']['by_family_both_correct']['temporal'],
            },
        },
        'globalpiqa_hard_surface': {
            'base': {m: {'accuracy': gp_base[m]['accuracy'], 'always_wrong_subset_accuracy': gp_base[m].get('always_wrong_subset', {}).get('accuracy') if gp_base[m].get('always_wrong_subset') else None, 'mean_top_minus_correct': gp_base[m]['all_rows_margin_summary']['mean_top_minus_correct']} for m in gp_base},
            'treatment': {m: {'accuracy': gp_treat[m]['accuracy'], 'always_wrong_subset_accuracy': gp_treat[m].get('always_wrong_subset', {}).get('accuracy') if gp_treat[m].get('always_wrong_subset') else None, 'mean_top_minus_correct': gp_treat[m]['all_rows_margin_summary']['mean_top_minus_correct']} for m in gp_treat},
            'role_fixed': {m: {'accuracy': gp_fixed[m]['accuracy'], 'always_wrong_subset_accuracy': gp_fixed[m].get('always_wrong_subset', {}).get('accuracy') if gp_fixed[m].get('always_wrong_subset') else None, 'mean_top_minus_correct': gp_fixed[m]['all_rows_margin_summary']['mean_top_minus_correct']} for m in gp_fixed},
            'treatment_minus_base': gp_delta(gp_base, gp_treat),
            'role_fixed_minus_base': gp_delta(gp_base, gp_fixed),
            'treatment_minus_role_fixed': gp_delta(gp_fixed, gp_treat),
        },
        'ewok_hard_surface': {
            'base_summary': ewok['targets']['base_reheat']['summary'],
            'treatment_summary': ewok['targets']['screen_treatment']['summary'],
            'role_fixed_summary': ewok['targets']['screen_role_fixed']['summary'],
            'treatment_minus_base': ewok['deltas_vs_base_reheat']['screen_treatment'],
            'role_fixed_minus_base': ewok['deltas_vs_base_reheat']['screen_role_fixed'],
            'treatment_minus_role_fixed': ewok['treatment_minus_role_fixed'],
        },
        'interpretation': {
            'positive': 'Role-exchange packets are learnable and transfer within the synthetic grammar: treatment beats role-fixed by +0.324 both_correct overall, +0.181 on held-out styles, and +0.344 on the wholly unseen container family after only 176,640 packet-word exposure.',
            'caution': 'The official hard surfaces do not yet show a decisive role-exchange-specific repair. GlobalPIQA_parallel rises only +1.94 points over base and +1.94 over role_fixed, with no improvement on the 52-row always-wrong subset accuracy over base and a slightly worse hard-subset mean wrong margin. EWoK stable failures fall by 126 for treatment but also by 111 for role_fixed; treatment-specific EWoK improvement is only -15 stable failures and essentially zero accuracy difference vs role_fixed.',
            'route_implication': 'Route B survives as a mechanism worth engineering, but not as authorization for two full blind retrains. The next scientific work is a legal from-beginning corpus/tokenizer design and possibly a smaller from-scratch training screen that embeds packets from the start, with role-fixed matched control and official hard-surface readouts.'
        },
        'files': {
            'packet_manifest': str(PKT),
            'learning_screen_summary': str(LEARN),
            'globalpiqa_individual_margins_dir': str(GP_DIR),
            'ewok_summary': str(EWOK),
            'synthesis_json': str(OUT_JSON),
            'synthesis_note': str(OUT_NOTE),
        }
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(synth, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    md = f"""# research — role-switch packet disposable learning screen synthesis\n\n## Legal/provenance status\n\nThis screen is not a submission-facing endpoint. It starts from the completed reheat 100M checkpoint and adds generated packet exposure after the old corpus has already been consumed. A legal BabyLM route must build a revised <=10M corpus and train tokenizer/model from that corpus from the beginning.\n\nExpanded suite: {pkt['treatment_train']['n_texts']} train texts / {pkt['treatment_train']['words']} words per arm; train families {pkt['train_families']}; temporal and container excluded from training. Official-text {7,8,10}-gram overlaps are {synth['expanded_packet_suite']['provenance_overlap_7_8_10']}. Treatment/control are word matched and target-count matched.\n\n## Synthetic packet exchange result\n\nBase reheat both_correct = {synth['packet_exchange_screen']['base_overall_both_correct']:.4f}; role-switch treatment = {synth['packet_exchange_screen']['treatment_overall_both_correct']:.4f}; role-fixed control = {synth['packet_exchange_screen']['role_fixed_overall_both_correct']:.4f}. Treatment minus role-fixed = {synth['packet_exchange_screen']['treatment_minus_role_fixed_overall_both_correct']:+.4f} both_correct and {synth['packet_exchange_screen']['treatment_minus_role_fixed_M_mean']:+.3f} four-cell M.\n\nHeld-out-style both_correct: {synth['packet_exchange_screen']['heldout_style_both_correct']}. The wholly unseen container family moves from base {synth['packet_exchange_screen']['container_family_both_correct']['base']:.4f} to treatment {synth['packet_exchange_screen']['container_family_both_correct']['treatment']:.4f} versus role-fixed {synth['packet_exchange_screen']['container_family_both_correct']['role_fixed']:.4f}. Temporal remains bad and should stay excluded/rebuilt: {synth['packet_exchange_screen']['temporal_family_both_correct']}.\n\n## Official hard-surface readout\n\nGlobalPIQA_parallel: base {gp_base['parallel']['accuracy']:.2f}, treatment {gp_treat['parallel']['accuracy']:.2f}, role-fixed {gp_fixed['parallel']['accuracy']:.2f}. Treatment minus role-fixed is {synth['globalpiqa_hard_surface']['treatment_minus_role_fixed']['parallel']['accuracy']:+.2f} points. The 52-row hard subset accuracy is base {gp_base['parallel']['always_wrong_subset']['accuracy']:.2f}, treatment {gp_treat['parallel']['always_wrong_subset']['accuracy']:.2f}, role-fixed {gp_fixed['parallel']['always_wrong_subset']['accuracy']:.2f}; hard-subset mean top-minus-correct does not improve relative to base. GlobalPIQA_nonparallel is base {gp_base['nonparallel']['accuracy']:.2f}, treatment {gp_treat['nonparallel']['accuracy']:.2f}, role-fixed {gp_fixed['nonparallel']['accuracy']:.2f}.\n\nEWoK four-cell: base accuracy {ewok['targets']['base_reheat']['summary']['accuracy']:.4f}, stable failures {ewok['targets']['base_reheat']['summary']['stable_failure']}; treatment accuracy {ewok['targets']['screen_treatment']['summary']['accuracy']:.4f}, stable failures {ewok['targets']['screen_treatment']['summary']['stable_failure']}; role-fixed accuracy {ewok['targets']['screen_role_fixed']['summary']['accuracy']:.4f}, stable failures {ewok['targets']['screen_role_fixed']['summary']['stable_failure']}. Treatment reduces stable failures by {ewok['deltas_vs_base_reheat']['screen_treatment']['stable_failure']} vs base, but role-fixed also reduces them by {ewok['deltas_vs_base_reheat']['screen_role_fixed']['stable_failure']}; treatment-specific difference is only {ewok['treatment_minus_role_fixed']['stable_failure']}.\n\n## Scientific interpretation\n\nRole-switch text is not merely memorized local fit: after only 176,640 packet-word exposure, it transfers to held-out templates and the wholly unseen container family inside the synthetic grammar, strongly beyond the role-fixed control. This is the first clean low-cost evidence that balanced role exchange can teach a context-conditioned binding operation rather than only reinforce target priors.\n\nHowever, the original BabyLM hard surfaces barely move in a role-exchange-specific way. GlobalPIQA_parallel gets only a small +1.94 point movement and the hard 52-row rank deficit remains deep. EWoK improves similarly under both treatment and role-fixed packet exposure, so most of that change is not specific to exchange. Route B should therefore be continued as a mechanism-engineering route, but not by immediately launching two full blind 100M retrains.\n\nNext work should design a compliant from-beginning replacement corpus/tokenizer: packets must be embedded inside a revised <=10M pool before tokenizer training and model training, with a role-fixed matched corpus as the scientific control. A smaller from-scratch screen can test whether starting with the packets preserves broad learning and moves official hard surfaces before committing full H100 endpoints.\n"""
    OUT_NOTE.parent.mkdir(parents=True, exist_ok=True)
    OUT_NOTE.write_text(md, encoding='utf-8')
    print(json.dumps({'event': 'PACKET_SCREEN_SYNTHESIS_DONE', 'json': str(OUT_JSON), 'note': str(OUT_NOTE)}, indent=2), flush=True)

if __name__ == '__main__':
    main()
