# causal interface trajectory causal interface trajectory

## Purpose

This experiment follows the causal intervention donor-query causal interface through continuation. It does not treat familiar-symbol redirection as the endpoint. The measurements separate three questions: whether the model forms a query-conditioned L1 signal, whether later layers/readout remain sensitive to a supplied preparation signal, and whether the effect narrows from train symbols to held symbols.

## Mean trajectory across seeds

### direct_full

| be | train4 | held4 | heldB | ctxCE | dir train/held | self d train/held | prep→cur d train/held | cur→prep d train/held |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.000 | 0.898 | +6.398 | 32.258 | 0.973/0.827 | 0.996/0.856 | 0.996/0.856 | 0.996/0.856 |
| 1 | 0.771 | 0.668 | +3.552 | 21.384 | 0.790/0.570 | 0.779/0.616 | 0.996/0.866 | 0.699/0.584 |
| 5 | 0.299 | 0.270 | -0.051 | 5.738 | 0.343/0.213 | 0.283/0.222 | 0.807/0.619 | 0.230/0.241 |
| 25 | 0.336 | 0.352 | +0.072 | 1.423 | 0.297/0.272 | 0.312/0.306 | 0.600/0.466 | 0.266/0.259 |
| 100 | 0.383 | 0.379 | +0.780 | 1.338 | 0.275/0.255 | 0.387/0.328 | 0.697/0.584 | 0.277/0.294 |
| 150 | 0.432 | 0.371 | +0.668 | 1.294 | 0.350/0.292 | 0.418/0.350 | 0.697/0.566 | 0.311/0.306 |
| 250 | 0.793 | 0.402 | +2.020 | 1.250 | 0.530/0.373 | 0.760/0.359 | 0.814/0.581 | 0.553/0.381 |
| 500 | 0.889 | 0.359 | +2.585 | 1.236 | 0.508/0.372 | 0.838/0.391 | 0.746/0.522 | 0.566/0.372 |

### static_1over17

| be | train4 | held4 | heldB | ctxCE | dir train/held | self d train/held | prep→cur d train/held | cur→prep d train/held |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.000 | 0.898 | +6.398 | 32.258 | 0.973/0.827 | 0.996/0.856 | 0.996/0.856 | 0.996/0.856 |
| 1 | 0.938 | 0.770 | +4.517 | 21.546 | 0.895/0.673 | 0.902/0.713 | 0.996/0.859 | 0.830/0.700 |
| 5 | 0.982 | 0.910 | +3.898 | 6.281 | 0.960/0.880 | 0.975/0.869 | 0.947/0.741 | 0.504/0.519 |
| 25 | 1.000 | 0.777 | +5.299 | 1.420 | 0.993/0.723 | 1.000/0.775 | 0.969/0.781 | 0.990/0.700 |
| 100 | 1.000 | 0.672 | +5.656 | 1.337 | 1.000/0.658 | 1.000/0.653 | 0.975/0.791 | 1.000/0.631 |
| 150 | 1.000 | 0.746 | +7.564 | 1.306 | 1.000/0.755 | 0.998/0.734 | 0.965/0.828 | 0.996/0.744 |
| 250 | 1.000 | 0.859 | +9.530 | 1.260 | 1.000/0.807 | 1.000/0.812 | 0.965/0.812 | 1.000/0.787 |
| 500 | 1.000 | 0.777 | +8.692 | 1.238 | 0.903/0.605 | 1.000/0.719 | 0.969/0.844 | 0.955/0.622 |

### interleaved_ans_full

| be | train4 | held4 | heldB | ctxCE | dir train/held | self d train/held | prep→cur d train/held | cur→prep d train/held |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.000 | 0.898 | +6.398 | 32.258 | 0.973/0.827 | 0.996/0.856 | 0.996/0.856 | 0.996/0.856 |
| 1 | 0.998 | 0.879 | +6.870 | 32.178 | 0.973/0.825 | 0.996/0.869 | 0.994/0.866 | 0.992/0.878 |
| 5 | 0.953 | 0.941 | +4.661 | 11.369 | 0.960/0.898 | 0.955/0.925 | 0.988/0.825 | 0.527/0.616 |
| 25 | 1.000 | 0.871 | +5.426 | 1.736 | 0.997/0.795 | 1.000/0.831 | 0.959/0.781 | 0.975/0.762 |
| 100 | 1.000 | 0.797 | +6.191 | 1.373 | 0.995/0.723 | 1.000/0.753 | 0.963/0.800 | 0.984/0.675 |
| 150 | 1.000 | 0.738 | +6.280 | 1.364 | 0.993/0.682 | 1.000/0.691 | 0.961/0.791 | 0.992/0.653 |
| 250 | 1.000 | 0.742 | +6.375 | 1.332 | 0.995/0.722 | 1.000/0.716 | 0.953/0.762 | 0.998/0.716 |
| 500 | 1.000 | 0.875 | +8.814 | 1.256 | 1.000/0.760 | 1.000/0.825 | 0.934/0.809 | 0.998/0.787 |

## Reading guide

- `dir train/held` is L1 slot classification by the preparation-learned direction in the current model. It measures separability, not causal use.
- `self d train/held` patches the current model's own donor activation into itself along the preparation direction. It measures formation plus current downstream use.
- `prep→cur d` patches the preparation donor signal into the current recipient. High values mean current downstream layers can still use a well-formed preparation-era signal.
- `cur→prep d` patches the current donor signal into the preparation recipient. Low values mean the current activation no longer carries the preparation-readable query-selection component.
- Held columns use held-donor pairs: the donor query is a held entity in a context containing one held entity and three train entities; the recipient query is a train entity from the same context.

## Main interpretation

The causal interface should be read against behavioral held transfer. If familiar `self d` remains high while held `self d` falls, the continuation has narrowed formation of the reusable selector rather than globally preserving it. If `prep→cur d` remains high when `self d` is low, the recipient can use a supplied signal, so loss is upstream of or at signal formation rather than a complete failure of downstream readout. If `cur→prep d` is low, the current donor activation itself lacks the preparation-readable signal. Seed-level details in `results.json` are necessary for the subtle seed100 direct-full case, where direction classification can be near chance while d-only redirection remains functional.

## Files

- Data: `experiments/archive/functional_learning/data/causal_interface_trajectory/results.json`
- Figure: `experiments/archive/functional_learning/figures/causal_interface_trajectory.png`
- Script: `experiments/archive/functional_learning/scripts/causal_interface_trajectory.py`
