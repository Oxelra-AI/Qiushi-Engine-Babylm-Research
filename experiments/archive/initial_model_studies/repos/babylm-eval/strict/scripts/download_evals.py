from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="BabyLM-community/BabyLM-2026-Strict-Evals",
    repo_type="dataset",
    local_dir=".",
)
