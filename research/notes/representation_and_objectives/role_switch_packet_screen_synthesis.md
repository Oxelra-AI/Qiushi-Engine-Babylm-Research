# role switch packet screen synthesis — role-switch packet disposable learning screen synthesis

## Legal/provenance status

This screen is not a submission-facing endpoint. It starts from the completed reheat 100M checkpoint and adds generated packet exposure after the old corpus has already been consumed. A legal BabyLM route must build a revised <=10M corpus and train tokenizer/model from that corpus from the beginning.

Expanded suite: 1560 train texts / 22080 words per arm; train families ['comparative', 'spatial', 'state_change', 'transfer']; temporal and container excluded from training. Official-text (7, 8, 10)-gram overlaps are {'7gram': 0, '8gram': 0, '10gram': 0}. Treatment/control are word matched and target-count matched.

## Synthetic packet exchange result

Base reheat both_correct = 0.2981; role-switch treatment = 0.6987; role-fixed control = 0.3750. Treatment minus role-fixed = +0.3237 both_correct and +4.950 four-cell M.

Held-out-style both_correct: {'base': 0.37592592592592594, 'treatment': 0.5222222222222223, 'role_fixed': 0.34074074074074073, 'treatment_minus_role_fixed': 0.18148148148148152}. The wholly unseen container family moves from base 0.0278 to treatment 0.5611 versus role-fixed 0.2167. Temporal remains bad and should stay excluded/rebuilt: {'base': 0.31, 'treatment': 0.21666666666666667, 'role_fixed': 0.32, 'treatment_minus_role_fixed': -0.10333333333333333}.

## Official hard-surface readout

GlobalPIQA_parallel: base 24.27, treatment 26.21, role-fixed 24.27. Treatment minus role-fixed is +1.94 points. The 52-row hard subset accuracy is base 3.85, treatment 3.85, role-fixed 1.92; hard-subset mean top-minus-correct does not improve relative to base. GlobalPIQA_nonparallel is base 51.00, treatment 51.00, role-fixed 48.00.

EWoK four-cell: base accuracy 0.5056, stable failures 2587; treatment accuracy 0.5127, stable failures 2461; role-fixed accuracy 0.5130, stable failures 2476. Treatment reduces stable failures by -126 vs base, but role-fixed also reduces them by -111; treatment-specific difference is only -15.

## Scientific interpretation

Role-switch text is not merely memorized local fit: after only 176,640 packet-word exposure, it transfers to held-out templates and the wholly unseen container family inside the synthetic grammar, strongly beyond the role-fixed control. This is the first clean low-cost evidence that balanced role exchange can teach a context-conditioned binding operation rather than only reinforce target priors.

However, the original BabyLM hard surfaces barely move in a role-exchange-specific way. GlobalPIQA_parallel gets only a small +1.94 point movement and the hard 52-row rank deficit remains deep. EWoK improves similarly under both treatment and role-fixed packet exposure, so most of that change is not specific to exchange. Route B should therefore be continued as a mechanism-engineering route, but not by immediately launching two full blind 100M retrains.

Next work should design a compliant from-beginning replacement corpus/tokenizer: packets must be embedded inside a revised <=10M pool before tokenizer training and model training, with a role-fixed matched corpus as the scientific control. A smaller from-scratch screen can test whether starting with the packets preserves broad learning and moves official hard surfaces before committing full H100 endpoints.
