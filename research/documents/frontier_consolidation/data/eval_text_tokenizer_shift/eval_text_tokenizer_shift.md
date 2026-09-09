# compliant tokenizer shift and eval readiness — tokenizer shift on official evaluation texts

CPU-only comparison of the old 100M-trained baseline16k tokenizer and the new 10M-trained compliant16k tokenizer on official evaluation text units. This is geometry evidence, not model performance.

## Family aggregate

| family | texts | seq | old tokens | new tokens | new/old | total Δ | old trunc % | new trunc % | new-only trunc | old-only trunc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BLiMP | 119750 | 256 | 1223251 | 1228373 | 1.0042 | 5122 | 0.000 | 0.000 | 0 | 0 |
| COMPS | 182056 | 256 | 2647164 | 2673488 | 1.0099 | 26324 | 0.000 | 0.000 | 0 | 0 |
| EWoK | 30472 | 256 | 277182 | 280718 | 1.0128 | 3536 | 0.000 | 0.000 | 0 | 0 |
| EWoK_concat | 15236 | 256 | 277182 | 280718 | 1.0128 | 3536 | 0.000 | 0.000 | 0 | 0 |
| Entity | 56898 | 256 | 9267436 | 9281512 | 1.0015 | 14076 | 0.053 | 0.053 | 0 | 0 |
| GlobalPIQA_nonparallel | 300 | 256 | 7402 | 7469 | 1.0091 | 67 | 0.000 | 0.000 | 0 | 0 |
| GlobalPIQA_parallel | 515 | 256 | 15317 | 15411 | 1.0061 | 94 | 0.000 | 0.000 | 0 | 0 |
| Reading_sentence | 1726 | 256 | 20102 | 20220 | 1.0059 | 118 | 0.000 | 0.000 | 0 | 0 |
| Reading_word | 1726 | 256 | 1996 | 2003 | 1.0035 | 7 | 0.000 | 0.000 | 0 | 0 |
| SuperGLUE | 92959 | 512 | 15899832 | 15826798 | 0.9954 | -73034 | 8.937 | 8.840 | 10 | 100 |
| Supplement | 10436 | 256 | 172467 | 172736 | 1.0016 | 269 | 0.000 | 0.000 | 0 | 0 |

## Largest token-count shifts

| family | subtask | split | role | row | old | new | Δ | trunc old→new | text prefix |
|---|---|---|---|---:|---:|---:|---:|---|---|
| SuperGLUE | boolq | train | all_text_fields | 3872 | 574 | 618 | 44 | 1→1 | was dwight the father of angela's baby </s> Angela Martin -- Dwight is the only person that Angela likes in the office.  |
| SuperGLUE | multirc | train | all_text_fields | 9802 | 735 | 692 | -43 | 1→1 | In chapters 3 and 4 we described how the U.S. government adjusted its existing agencies and capacities to address the em |
| SuperGLUE | multirc | train | all_text_fields | 9704 | 711 | 670 | -41 | 1→1 | In chapters 3 and 4 we described how the U.S. government adjusted its existing agencies and capacities to address the em |
| SuperGLUE | multirc | train | all_text_fields | 9757 | 712 | 671 | -41 | 1→1 | In chapters 3 and 4 we described how the U.S. government adjusted its existing agencies and capacities to address the em |
| SuperGLUE | multirc | train | all_text_fields | 9701 | 713 | 673 | -40 | 1→1 | In chapters 3 and 4 we described how the U.S. government adjusted its existing agencies and capacities to address the em |
| SuperGLUE | multirc | train | all_text_fields | 9754 | 714 | 674 | -40 | 1→1 | In chapters 3 and 4 we described how the U.S. government adjusted its existing agencies and capacities to address the em |
| SuperGLUE | multirc | train | all_text_fields | 9797 | 716 | 676 | -40 | 1→1 | In chapters 3 and 4 we described how the U.S. government adjusted its existing agencies and capacities to address the em |
| SuperGLUE | multirc | train | all_text_fields | 9801 | 739 | 699 | -40 | 1→1 | In chapters 3 and 4 we described how the U.S. government adjusted its existing agencies and capacities to address the em |
| SuperGLUE | multirc | train | all_text_fields | 9803 | 735 | 695 | -40 | 1→1 | In chapters 3 and 4 we described how the U.S. government adjusted its existing agencies and capacities to address the em |
| SuperGLUE | multirc | train | all_text_fields | 9806 | 736 | 696 | -40 | 1→1 | In chapters 3 and 4 we described how the U.S. government adjusted its existing agencies and capacities to address the em |
| SuperGLUE | multirc | train | all_text_fields | 9810 | 733 | 693 | -40 | 1→1 | In chapters 3 and 4 we described how the U.S. government adjusted its existing agencies and capacities to address the em |
| SuperGLUE | multirc | train | all_text_fields | 9811 | 729 | 689 | -40 | 1→1 | In chapters 3 and 4 we described how the U.S. government adjusted its existing agencies and capacities to address the em |
| SuperGLUE | multirc | train | all_text_fields | 9700 | 712 | 673 | -39 | 1→1 | In chapters 3 and 4 we described how the U.S. government adjusted its existing agencies and capacities to address the em |
| SuperGLUE | multirc | train | all_text_fields | 9702 | 713 | 674 | -39 | 1→1 | In chapters 3 and 4 we described how the U.S. government adjusted its existing agencies and capacities to address the em |
| SuperGLUE | multirc | train | all_text_fields | 9703 | 708 | 669 | -39 | 1→1 | In chapters 3 and 4 we described how the U.S. government adjusted its existing agencies and capacities to address the em |
| SuperGLUE | multirc | train | all_text_fields | 9705 | 710 | 671 | -39 | 1→1 | In chapters 3 and 4 we described how the U.S. government adjusted its existing agencies and capacities to address the em |
| SuperGLUE | multirc | train | all_text_fields | 9706 | 708 | 669 | -39 | 1→1 | In chapters 3 and 4 we described how the U.S. government adjusted its existing agencies and capacities to address the em |
| SuperGLUE | multirc | train | all_text_fields | 9707 | 710 | 671 | -39 | 1→1 | In chapters 3 and 4 we described how the U.S. government adjusted its existing agencies and capacities to address the em |
| SuperGLUE | multirc | train | all_text_fields | 9753 | 713 | 674 | -39 | 1→1 | In chapters 3 and 4 we described how the U.S. government adjusted its existing agencies and capacities to address the em |
| SuperGLUE | multirc | train | all_text_fields | 9755 | 714 | 675 | -39 | 1→1 | In chapters 3 and 4 we described how the U.S. government adjusted its existing agencies and capacities to address the em |
| SuperGLUE | multirc | train | all_text_fields | 9756 | 709 | 670 | -39 | 1→1 | In chapters 3 and 4 we described how the U.S. government adjusted its existing agencies and capacities to address the em |
| SuperGLUE | multirc | train | all_text_fields | 9758 | 711 | 672 | -39 | 1→1 | In chapters 3 and 4 we described how the U.S. government adjusted its existing agencies and capacities to address the em |
| SuperGLUE | multirc | train | all_text_fields | 9759 | 709 | 670 | -39 | 1→1 | In chapters 3 and 4 we described how the U.S. government adjusted its existing agencies and capacities to address the em |
| SuperGLUE | multirc | train | all_text_fields | 9760 | 711 | 672 | -39 | 1→1 | In chapters 3 and 4 we described how the U.S. government adjusted its existing agencies and capacities to address the em |
| SuperGLUE | multirc | train | all_text_fields | 9761 | 717 | 678 | -39 | 1→1 | In chapters 3 and 4 we described how the U.S. government adjusted its existing agencies and capacities to address the em |
| SuperGLUE | multirc | train | all_text_fields | 9795 | 718 | 679 | -39 | 1→1 | In chapters 3 and 4 we described how the U.S. government adjusted its existing agencies and capacities to address the em |
| SuperGLUE | multirc | train | all_text_fields | 9804 | 729 | 690 | -39 | 1→1 | In chapters 3 and 4 we described how the U.S. government adjusted its existing agencies and capacities to address the em |
| SuperGLUE | multirc | train | all_text_fields | 9812 | 738 | 699 | -39 | 1→1 | In chapters 3 and 4 we described how the U.S. government adjusted its existing agencies and capacities to address the em |
| SuperGLUE | boolq | valid | all_text_fields | 181 | 456 | 494 | 38 | 0→0 | did rita die in season 4 of dexter </s> Rita Bennett -- In the fourth season opener, Rita and Dexter are living happily  |
| SuperGLUE | multirc | train | all_text_fields | 9712 | 737 | 699 | -38 | 1→1 | In chapters 3 and 4 we described how the U.S. government adjusted its existing agencies and capacities to address the em |
| SuperGLUE | multirc | train | all_text_fields | 9715 | 733 | 695 | -38 | 1→1 | In chapters 3 and 4 we described how the U.S. government adjusted its existing agencies and capacities to address the em |
| SuperGLUE | multirc | train | all_text_fields | 9741 | 748 | 710 | -38 | 1→1 | In chapters 3 and 4 we described how the U.S. government adjusted its existing agencies and capacities to address the em |
| SuperGLUE | multirc | train | all_text_fields | 9762 | 716 | 678 | -38 | 1→1 | In chapters 3 and 4 we described how the U.S. government adjusted its existing agencies and capacities to address the em |
| SuperGLUE | multirc | train | all_text_fields | 9769 | 724 | 686 | -38 | 1→1 | In chapters 3 and 4 we described how the U.S. government adjusted its existing agencies and capacities to address the em |
| SuperGLUE | multirc | train | all_text_fields | 9770 | 730 | 692 | -38 | 1→1 | In chapters 3 and 4 we described how the U.S. government adjusted its existing agencies and capacities to address the em |
| SuperGLUE | multirc | train | all_text_fields | 9781 | 763 | 725 | -38 | 1→1 | In chapters 3 and 4 we described how the U.S. government adjusted its existing agencies and capacities to address the em |
| SuperGLUE | multirc | train | all_text_fields | 9784 | 751 | 713 | -38 | 1→1 | In chapters 3 and 4 we described how the U.S. government adjusted its existing agencies and capacities to address the em |
| SuperGLUE | multirc | train | all_text_fields | 9785 | 730 | 692 | -38 | 1→1 | In chapters 3 and 4 we described how the U.S. government adjusted its existing agencies and capacities to address the em |
| SuperGLUE | multirc | train | all_text_fields | 9789 | 741 | 703 | -38 | 1→1 | In chapters 3 and 4 we described how the U.S. government adjusted its existing agencies and capacities to address the em |
| SuperGLUE | multirc | train | all_text_fields | 9793 | 715 | 677 | -38 | 1→1 | In chapters 3 and 4 we described how the U.S. government adjusted its existing agencies and capacities to address the em |
