# MAX-Register Opportunity Cost

Status: matched MAX-register pools passed preflight; half-dose training completed, but register scores were not yet measured.

The half-dose run completed 100,000,000 words in 2,552 updates, with 34,467,424 parameters, vocabulary 16,384, first/last losses 9.815051/2.425119 and ten checkpoints. The quarter-dose run was still partial at 50,948,297 words and update 1302, with loss 2.820576. These training observations did not establish the dose-response curve.

Both MAX-register arms admitted the identical FineWeb block and preserved 653,130 rows and 100,000,000 exposure words. One displaced 1,118,720 developmental/speech words: CHILDES 578,080, OpenSubtitles 411,040, BNC Spoken 124,160 and Switchboard 5,440. The other displaced 1,118,720 adult-prose words: Gutenberg 710,560 and SimpleWiki 408,160. Tokenizer and model-size checks matched.

The primary future contrast was childspeech-removed minus adultprose-removed on common-window exEntity5 and family components. Comparing both arms with proportional MAX-view and two matched clean basins would separate register skew from general admission and seed variation.

If childspeech removal scored lower, developmental/speech experience had higher opportunity cost under this budget; the opposite sign favored adult prose. Similar arms would favor the admitted content or architecture-conditioned mixture over sacrificed-register identity. Entity required separate operation-stratified analysis. No claim about learned role structure followed from register divergence alone.
