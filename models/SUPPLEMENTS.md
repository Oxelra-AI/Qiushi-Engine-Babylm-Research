# Published Evaluation Supplements

The original supplements were checked against the immutable Hugging Face revisions, Git/LFS identities and published checksums.

275 files are byte-identical copies; 6 retain all scientific values with a public-facing estimator provenance label. Original and distributed digests are recorded separately.

| Model | Additional scientific files | Local directories |
| --- | ---: | --- |
| Frontier | 140 | [Evaluation](frontier/evaluation/) and [submission](frontier/submission/) |
| Principle Guided | 141 | [Evaluation](principle_guided/evaluation/) and [submission](principle_guided/submission/) |

The supplements include complete endpoint predictions, Fast checkpoint trajectories, acquisition-order measurements and official-format submission JSON files. The principle-guided package also includes its continuation record.

The original 18 loading, weight and model-documentation files in each package are unchanged. File identities, publication revisions and scientific roles are in the [material manifest](../evidence/materials_manifest.json).

Files containing benchmark text, and the submission bundles, remain local-only pending source-specific redistribution review. They are present in the local engineering tree but excluded from the source distribution. No model was retrained or rescored during this import.
