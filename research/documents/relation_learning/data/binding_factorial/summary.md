# seed43022 dose practical predictions binding factorial summary

| arm | joint | pairs | unchanged | updated | updates | checkpoint |
|---|---:|---:|---:|---:|---:|---|
| answer_clean | 0.205 | 41 | 0.37 | 0.83 | 6260 | experiments/archive/relation_learning/data/binding_factorial/answer_clean/checkpoint |
| uniform_wwm | 0.035 | 7 | 0.295 | 0.68 | 6260 | experiments/archive/relation_learning/data/binding_factorial/uniform_wwm/checkpoint |
| answer_corrupt_update_state | 0.155 | 31 | 0.61 | 0.5475 | 6260 | experiments/archive/relation_learning/data/binding_factorial/answer_corrupt_update_state/checkpoint |

## Contrasts

{"contrast": "answer_clean-minus-uniform_wwm", "binding_joint_accuracy_delta": 0.16999999999999998, "binding_joint_correct_delta": 34, "unchanged_accuracy_delta": 0.07500000000000001, "updated_accuracy_delta": 0.1499999999999999}
{"contrast": "answer_clean-minus-answer_corrupt_update_state", "binding_joint_accuracy_delta": 0.04999999999999999, "binding_joint_correct_delta": 10, "unchanged_accuracy_delta": -0.24, "updated_accuracy_delta": 0.2825}
{"contrast": "uniform_wwm-minus-answer_corrupt_update_state", "binding_joint_accuracy_delta": -0.12, "binding_joint_correct_delta": -24, "unchanged_accuracy_delta": -0.315, "updated_accuracy_delta": 0.13250000000000006}
