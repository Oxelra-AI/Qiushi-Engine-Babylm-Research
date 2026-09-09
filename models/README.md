# Two Generations of Language Models

Each model generation has a separate local directory containing actual weights, configuration, tokenizer, custom loading code and published documentation. Pinned Hugging Face revisions are also retained.

| Local directory | Released name | Role |
| --- | --- | --- |
| [frontier](frontier/README.md) | Qiushi-Engine-Frontier-Advancement | Representative Stage I model and shared parent for later studies |
| [principle_guided](principle_guided/README.md) | Qiushi-Engine-Principle-Guided-Frontier-Advancement | Representative model for the full Stage III strategy |

Both are masked language models, not conversational generation models. Their encoders contain two residual paths that must also be retained in downstream use.

The tokenizers have identical vocabularies, merge rules and special tokens, but their exported default padding/truncation lengths differ. Model comparisons must explicitly use the same task-appropriate length, padding and truncation settings; export defaults are not a common evaluation protocol.

## Local Use

Install the dependencies recorded in the model directory in an isolated environment, then run this from the repository root:

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

path = "models/principle_guided"
tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
model = AutoModelForMaskedLM.from_pretrained(
    path, trust_remote_code=True, local_files_only=True
)
```

`trust_remote_code=True` loads the custom model class from that directory. Inspect the adjacent Python file before use. This example does not update remote models. For encoder fine-tuning, use the `AutoModel` entry point from the same directory.

For a code package without weights, `make models` downloads the same pinned revisions. Each model is approximately 146 MB; GitHub weight distribution uses Git LFS. Ordinary report and source packages need not include weights. `make package-with-models` creates an offline package with local weights.

## Supporting Materials

Each directory retains loading files, the published model card, methods, data documentation, training records, environment records and checkpoint catalog. All 281 scientific supplements from the pinned releases are now present: 140 for the first generation and 141 for the second. Item-level predictions, Fast checkpoint results, AoA measurements and official-format submissions are in each model's `evaluation/` and `submission/` directories. The second generation also includes `CONTINUATION.json`.

Of these supplements, 275 are byte-identical to the original releases. The other 6 differ only in estimator provenance labels, which use public method names; scientific values, text and ordering are unchanged. See the [supplement guide](SUPPLEMENTS.md) for file correspondence. Distribution follows the source-specific permissions in the [data guide](../data/README.md) and [third-party notices](../reproducibility/THIRD_PARTY.md); items marked for local use only are excluded from public packages.

Original pinned releases:

- [Complete Frontier materials](https://huggingface.co/leslie721007/Qiushi-Engine-Frontier-Advancement/tree/5eb20f9c5088f40183269bb2c97381711aea0143)
- [Complete principle-guided materials](https://huggingface.co/leslie721007/Qiushi-Engine-Principle-Guided-Frontier-Advancement/tree/ce7eabf0dfbd3d1393670f41f610bdccbdbe45d1)

`VALIDATION.json` is the existing validation record accompanying each public model, not a report of training or evaluation rerun during this preparation. Tools maintain pinned file versions; readers generally need not inspect each one individually. Each model retains its existing Apache-2.0 license and notices.
