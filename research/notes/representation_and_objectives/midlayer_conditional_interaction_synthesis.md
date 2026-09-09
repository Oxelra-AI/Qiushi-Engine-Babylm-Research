# midlayer base80 midlayer conditional-interaction synthesis

This synthesis compares matched legal16k base, scale1.75 live, and scale1.75-disabled layerwise residual-state readouts. The readout applies the final MLM head to each layer, so positive midlayer recovery is meaningful, while negative decodability is not proof of absence.

Summary JSON: `experiments/archive/representation_and_objectives/data/midlayer_synthesis/midlayer_synthesis.json`

## Final-layer reproduction

- `matched_base_80M`: EWoK selected-final t1=0.43487621097954793 vs prior full EWoK=0.5091887634549751; selected-final stable=0.43918191603875134 vs prior full stable=0.3260698346022578; GP final all=28.155339805825243 vs prior parallel=None; GP final hard52=3.8461538461538463 vs prior hard52=None.
- `scale1p75_live_80M`: EWoK selected-final t1=0.4294940796555436 vs prior full EWoK=0.4927802572853767; selected-final stable=0.43487621097954793 vs prior full stable=0.3514045681281176; GP final all=26.21359223300971 vs prior parallel=26.21359223300971; GP final hard52=1.9230769230769231 vs prior hard52=1.9230769230769231.
- `scale1p75_disabled_80M`: EWoK selected-final t1=0.4402583423035522 vs prior full EWoK=0.49448674192701497; selected-final stable=0.40365984930032295 vs prior full stable=0.3409031241795747; GP final all=24.271844660194176 vs prior parallel=24.271844660194176; GP final hard52=1.9230769230769231 vs prior hard52=1.9230769230769231.

## Best-layer pattern

- `matched_base_80M`: EWoK best both layer 4, best interaction layer 4, lowest stable layer 2; GlobalPIQA best all layer 8, best hard52 layer 0.
- `scale1p75_live_80M`: EWoK best both layer 7, best interaction layer 6, lowest stable layer 1; GlobalPIQA best all layer 4, best hard52 layer 1.
- `scale1p75_disabled_80M`: EWoK best both layer 7, best interaction layer 6, lowest stable layer 1; GlobalPIQA best all layer 4, best hard52 layer 1.

## Paired base-scale deltas

### scale_live_minus_base
- EWoK L0: Δinteraction=0.0056, Δt1=0.0308, net both=-9, net stable=-14.
- EWoK L1: Δinteraction=0.0009, Δt1=0.0118, net both=-7, net stable=-34.
- EWoK L2: Δinteraction=-0.0181, Δt1=0.0366, net both=-31, net stable=40.
- EWoK L3: Δinteraction=-0.0160, Δt1=0.0379, net both=-9, net stable=27.
- EWoK L4: Δinteraction=-0.0365, Δt1=0.0193, net both=-40, net stable=14.
- EWoK L5: Δinteraction=0.0016, Δt1=0.0213, net both=-11, net stable=5.
- EWoK L6: Δinteraction=-0.0031, Δt1=0.0014, net both=-12, net stable=-6.
- EWoK L7: Δinteraction=-0.5895, Δt1=-0.3836, net both=9, net stable=84.
- EWoK L8: Δinteraction=-0.0913, Δt1=-0.0079, net both=18, net stable=-4.
- GP hard52 L0: Δtop_minus_correct=-0.0048 (negative is better for b), net correct=-1, Δrank=-0.0962.
- GP hard52 L1: Δtop_minus_correct=0.0674 (negative is better for b), net correct=1, Δrank=-0.1154.
- GP hard52 L2: Δtop_minus_correct=0.1080 (negative is better for b), net correct=2, Δrank=-0.0769.
- GP hard52 L3: Δtop_minus_correct=0.0448 (negative is better for b), net correct=5, Δrank=-0.0577.
- GP hard52 L4: Δtop_minus_correct=0.0444 (negative is better for b), net correct=3, Δrank=0.0000.
- GP hard52 L5: Δtop_minus_correct=0.0161 (negative is better for b), net correct=1, Δrank=0.0000.
- GP hard52 L6: Δtop_minus_correct=0.0004 (negative is better for b), net correct=-1, Δrank=-0.1346.
- GP hard52 L7: Δtop_minus_correct=0.1705 (negative is better for b), net correct=-2, Δrank=0.0962.
- GP hard52 L8: Δtop_minus_correct=0.1084 (negative is better for b), net correct=-1, Δrank=0.1346.

### scale_disabled_minus_base
- EWoK L0: Δinteraction=0.0056, Δt1=0.0308, net both=-9, net stable=-14.
- EWoK L1: Δinteraction=0.0001, Δt1=0.0109, net both=-15, net stable=-35.
- EWoK L2: Δinteraction=-0.0184, Δt1=0.0354, net both=-37, net stable=41.
- EWoK L3: Δinteraction=-0.0186, Δt1=0.0338, net both=-4, net stable=17.
- EWoK L4: Δinteraction=-0.0365, Δt1=0.0210, net both=-27, net stable=5.
- EWoK L5: Δinteraction=-0.0047, Δt1=0.0143, net both=-8, net stable=-9.
- EWoK L6: Δinteraction=0.0053, Δt1=0.0124, net both=-5, net stable=-20.
- EWoK L7: Δinteraction=-0.4450, Δt1=-0.3419, net both=6, net stable=53.
- EWoK L8: Δinteraction=0.0126, Δt1=-0.0439, net both=12, net stable=-33.
- GP hard52 L0: Δtop_minus_correct=-0.0048 (negative is better for b), net correct=-1, Δrank=-0.0962.
- GP hard52 L1: Δtop_minus_correct=0.0683 (negative is better for b), net correct=1, Δrank=-0.0962.
- GP hard52 L2: Δtop_minus_correct=0.1107 (negative is better for b), net correct=3, Δrank=-0.1154.
- GP hard52 L3: Δtop_minus_correct=0.0502 (negative is better for b), net correct=5, Δrank=-0.0577.
- GP hard52 L4: Δtop_minus_correct=0.0283 (negative is better for b), net correct=3, Δrank=-0.0385.
- GP hard52 L5: Δtop_minus_correct=0.0151 (negative is better for b), net correct=2, Δrank=-0.0769.
- GP hard52 L6: Δtop_minus_correct=-0.0201 (negative is better for b), net correct=0, Δrank=-0.1154.
- GP hard52 L7: Δtop_minus_correct=0.1781 (negative is better for b), net correct=-1, Δrank=0.1154.
- GP hard52 L8: Δtop_minus_correct=0.0857 (negative is better for b), net correct=-1, Δrank=0.1346.

### scale_live_minus_disabled
- EWoK L0: Δinteraction=0.0000, Δt1=0.0000, net both=0, net stable=0.
- EWoK L1: Δinteraction=0.0008, Δt1=0.0010, net both=8, net stable=1.
- EWoK L2: Δinteraction=0.0003, Δt1=0.0011, net both=6, net stable=-1.
- EWoK L3: Δinteraction=0.0026, Δt1=0.0042, net both=-5, net stable=10.
- EWoK L4: Δinteraction=0.0000, Δt1=-0.0018, net both=-13, net stable=9.
- EWoK L5: Δinteraction=0.0063, Δt1=0.0070, net both=-3, net stable=14.
- EWoK L6: Δinteraction=-0.0085, Δt1=-0.0110, net both=-7, net stable=14.
- EWoK L7: Δinteraction=-0.1445, Δt1=-0.0418, net both=3, net stable=31.
- EWoK L8: Δinteraction=-0.1039, Δt1=0.0360, net both=6, net stable=29.
- GP hard52 L0: Δtop_minus_correct=0.0000 (negative is better for b), net correct=0, Δrank=0.0000.
- GP hard52 L1: Δtop_minus_correct=-0.0009 (negative is better for b), net correct=0, Δrank=-0.0192.
- GP hard52 L2: Δtop_minus_correct=-0.0027 (negative is better for b), net correct=-1, Δrank=0.0385.
- GP hard52 L3: Δtop_minus_correct=-0.0055 (negative is better for b), net correct=0, Δrank=0.0000.
- GP hard52 L4: Δtop_minus_correct=0.0162 (negative is better for b), net correct=0, Δrank=0.0385.
- GP hard52 L5: Δtop_minus_correct=0.0010 (negative is better for b), net correct=-1, Δrank=0.0769.
- GP hard52 L6: Δtop_minus_correct=0.0205 (negative is better for b), net correct=-1, Δrank=-0.0192.
- GP hard52 L7: Δtop_minus_correct=-0.0076 (negative is better for b), net correct=-1, Δrank=-0.0192.
- GP hard52 L8: Δtop_minus_correct=0.0227 (negative is better for b), net correct=0, Δrank=0.0000.

## Interpretation

The direct frozen-head readout finds frequent intermediate decodability on final-wrong rows, especially GlobalPIQA hard52 rank/correctness and EWoK interaction positivity, but this is not yet a clean late-erasure mechanism: the same phenomenon appears in both matched base and scale1.75, and paired EWoK deltas show scale1.75 is already worse than the matched base across most middle/final layers rather than losing a uniquely correct late signal. The live adapter contributes some final GlobalPIQA improvement over disabled inference, but its EWoK effect is small and mixed; disabling does not restore the matched base trajectory. This supports representation/trajectory formation as the main deficit unless a later decoder-robust alignment check shows persistent, adjacent-layer correct signals erased at a specific late boundary.
