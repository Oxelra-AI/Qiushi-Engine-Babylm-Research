# earlier analysis DeBERTa pair attention mask smoke

Pass: True

Full-visible custom vs public forward max logit delta: 0.000e+00; loss delta: 0.000e+00.

Blocked vs visible max logit delta: 7.668e-03; masked-position delta: 4.486e-03.

Gradient finite: True; grad norm: 8.84318.

A pair-aware MLM trainer requires a custom forward path; sending a square mask to the stock MLM forward does not implement the intended comparison.
