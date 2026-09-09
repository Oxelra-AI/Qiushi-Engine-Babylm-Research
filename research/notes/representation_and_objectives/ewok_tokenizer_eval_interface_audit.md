# ewok tokenizer eval interface audit — EWoK tokenizer evaluation-interface audit

JSON: `experiments/archive/representation_and_objectives/data/ewok_tokenizer_eval_interface_audit/ewok_tokenizer_eval_interface_audit.json`
Rows CSV: `experiments/archive/representation_and_objectives/data/ewok_tokenizer_eval_interface_audit/ewok_tokenizer_eval_interface_rows.csv`

This compares the inherited Strict-100M tokenizer and the new 10M-trained Strict-Small tokenizer on the official EWoK MLM scoring interface. It measures target/completion token counts and sentence token counts; it is not a model score.

## Overall EWoK geometry
- rows: 7618
- mean completion tokens: old 7.891, new 8.128, delta 0.237
- completion token count changed in 0.306 of rows; increased 0.256, decreased 0.050
- mean candidate sentence tokens: old 20.193, new 20.779, delta 0.586

## Focused unstable subset geometry
- focus_all: n=553, completion delta mean=0.349, changed_frac=0.306, sentence delta mean=0.659
- focus_negative: n=318, completion delta mean=0.340, changed_frac=0.296, sentence delta mean=0.668
- focus_0110: n=168, completion delta mean=0.357, changed_frac=0.321, sentence delta mean=0.750

## Worst relation domains
- material-dynamics: n=770, completion delta mean=0.027, changed_frac=0.262, sentence delta mean=0.186
- physical-dynamics: n=120, completion delta mean=0.292, changed_frac=0.292, sentence delta mean=0.375
- spatial-relations: n=490, completion delta mean=0.306, changed_frac=0.233, sentence delta mean=0.743
- physical-interactions: n=556, completion delta mean=0.399, changed_frac=0.288, sentence delta mean=0.662
- social-relations: n=1548, completion delta mean=0.471, changed_frac=0.461, sentence delta mean=0.910
- agent-properties: n=2210, completion delta mean=0.136, changed_frac=0.331, sentence delta mean=0.664
- material-properties: n=170, completion delta mean=0.382, changed_frac=0.394, sentence delta mean=0.459
- social-interactions: n=294, completion delta mean=0.367, changed_frac=0.367, sentence delta mean=0.667

## Selected concept tokenizations
- subordinate: old(3)=['Ġsub', 'ordin', 'ate'] ; new(3)=['Ġsub', 'ord', 'inate']
- boss: old(1)=['Ġboss'] ; new(1)=['Ġboss']
- parent: old(1)=['Ġparent'] ; new(1)=['Ġparent']
- child: old(1)=['Ġchild'] ; new(1)=['Ġchild']
- teacher: old(1)=['Ġteacher'] ; new(1)=['Ġteacher']
- student: old(1)=['Ġstudent'] ; new(1)=['Ġstudent']
- above: old(1)=['Ġabove'] ; new(1)=['Ġabove']
- below: old(1)=['Ġbelow'] ; new(1)=['Ġbelow']
- kick: old(1)=['Ġkick'] ; new(1)=['Ġkick']
- drop: old(1)=['Ġdrop'] ; new(1)=['Ġdrop']
- touch: old(1)=['Ġtouch'] ; new(1)=['Ġtouch']
- break: old(1)=['Ġbreak'] ; new(1)=['Ġbreak']
- stir: old(1)=['Ġstir'] ; new(1)=['Ġstir']
- accelerate: old(2)=['Ġacceler', 'ate'] ; new(2)=['Ġacceler', 'ate']
- slow down: old(2)=['Ġslow', 'Ġdown'] ; new(2)=['Ġslow', 'Ġdown']

## Reading
The corrected tokenizer should be interpreted as a representation change at both training and evaluation time. If later official EWoK movement differs from the inherited-tokenizer endpoint, this audit tells whether target-token fragmentation is large enough to be a plausible contributor. The actual endpoint decision still requires completed retraining and official collation.
