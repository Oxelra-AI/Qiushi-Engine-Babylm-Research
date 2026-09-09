# paired world stress teacher and route teacher failure postanalysis

Pairs: 3200

State counts: {'both_correct': 2813, 'qwen_correct_llama_wrong': 209, 'llama_correct_qwen_wrong': 178}

## Template both-correct rates

| template | n | both-correct | states |
|---|---:|---:|---|
| 01:defeated:w:train | 144 | 1.000 | {'both_correct': 144} |
| 02:beat:w:train | 144 | 0.903 | {'both_correct': 130, 'qwen_correct_llama_wrong': 13, 'llama_correct_qwen_wrong': 1} |
| 03:won against:w:train | 208 | 0.885 | {'qwen_correct_llama_wrong': 23, 'both_correct': 184, 'llama_correct_qwen_wrong': 1} |
| 04:overcame:w:train | 232 | 0.922 | {'qwen_correct_llama_wrong': 14, 'both_correct': 214, 'llama_correct_qwen_wrong': 4} |
| 05:proved too strong for:w:train | 120 | 0.983 | {'both_correct': 118, 'qwen_correct_llama_wrong': 2} |
| 06:lost to:l:train | 128 | 0.992 | {'both_correct': 127, 'llama_correct_qwen_wrong': 1} |
| 07:fell to:l:train | 192 | 0.911 | {'both_correct': 175, 'qwen_correct_llama_wrong': 16, 'llama_correct_qwen_wrong': 1} |
| 08:was defeated by:l:train | 136 | 1.000 | {'both_correct': 136} |
| 09:was beaten by:l:train | 168 | 0.994 | {'both_correct': 167, 'llama_correct_qwen_wrong': 1} |
| 10:was unable to overcome:l:train | 160 | 0.600 | {'both_correct': 96, 'qwen_correct_llama_wrong': 14, 'llama_correct_qwen_wrong': 50} |
| 11:saw triumph over:m:train | 136 | 0.985 | {'both_correct': 134, 'llama_correct_qwen_wrong': 1, 'qwen_correct_llama_wrong': 1} |
| 12:emerged victorious over:m:train | 168 | 0.952 | {'both_correct': 160, 'llama_correct_qwen_wrong': 5, 'qwen_correct_llama_wrong': 3} |
| 13:prevailed against:m:train | 200 | 0.955 | {'both_correct': 191, 'llama_correct_qwen_wrong': 4, 'qwen_correct_llama_wrong': 5} |
| 14:came out on top:m:train | 128 | 0.969 | {'both_correct': 124, 'llama_correct_qwen_wrong': 4} |
| 15:ended with victorious over:m:train | 120 | 0.950 | {'both_correct': 114, 'qwen_correct_llama_wrong': 4, 'llama_correct_qwen_wrong': 2} |
| 16:edged out:w:held | 160 | 0.950 | {'both_correct': 152, 'qwen_correct_llama_wrong': 2, 'llama_correct_qwen_wrong': 6} |
| 17:succumbed to:l:held | 88 | 0.716 | {'llama_correct_qwen_wrong': 2, 'qwen_correct_llama_wrong': 23, 'both_correct': 63} |
| 18:victory for W over L:m:held | 136 | 0.971 | {'both_correct': 132, 'llama_correct_qwen_wrong': 4} |
| 19:claimed the win:w:held | 176 | 0.932 | {'both_correct': 164, 'qwen_correct_llama_wrong': 1, 'llama_correct_qwen_wrong': 11} |
| 20:went down to:l:held | 256 | 0.344 | {'qwen_correct_llama_wrong': 88, 'llama_correct_qwen_wrong': 80, 'both_correct': 88} |

## By template-first role

| first role | n | both-correct | states |
|---|---:|---:|---|
| l | 1128 | 0.755 | {'both_correct': 852, 'qwen_correct_llama_wrong': 141, 'llama_correct_qwen_wrong': 135} |
| m | 888 | 0.963 | {'both_correct': 855, 'llama_correct_qwen_wrong': 20, 'qwen_correct_llama_wrong': 13} |
| w | 1184 | 0.934 | {'qwen_correct_llama_wrong': 55, 'both_correct': 1106, 'llama_correct_qwen_wrong': 23} |

## Interpretation

Failures are disagreements rather than shared wrong labels if both_same_wrong is zero. Concentration by held templates or by lost_to hypotheses indicates realization/predicate brittleness, not a stable teacher-supervised substrate.
