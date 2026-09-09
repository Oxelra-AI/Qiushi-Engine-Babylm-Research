# Distinct Post-40k Factor Proposals

Status: proposed alternatives with completed preflight only; no training outcome is supplied.

Two alternatives were prepared on the identical legal-40k tokenizer and compact-view-reinvestment stream for seeds 43022 and 43122.

The relation-boost alternative approximately doubled selection probability for relation-cue WWM groups while reducing nonrelation probability to preserve the expected 0.15 group budget. In the first 256 real stream rows, the candidate relation fraction was 0.1101, selected relation fraction 0.2208 and selected-group budget multiplier 1.0024. The disabled intervention reproduced baseline masking exactly. This was a learning-signal allocation test, not a vocabulary sweep.

The depth alternative used 12 layers, hidden size 384 and intermediate size 1280, giving 38,421,952 parameters instead of 45,826,720 in the 8x480 model. It tested depth-over-width on the same representation and data.

The proposed decision sequence first required the full legal-40k result. A strong endpoint would favor reproduction; broad failure would motivate a genuinely different factor. The prepared alternatives did not themselves establish that either relation weighting or depth improved downstream learning.
