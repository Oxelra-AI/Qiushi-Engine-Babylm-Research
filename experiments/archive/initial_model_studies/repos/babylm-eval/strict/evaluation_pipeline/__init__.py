
# --- compat shim: load checkpoints saved by transformers 5.x on 4.x ---
# 5.x writes tokenizer_class "TokenizersBackend" into tokenizer_config.json;
# 4.x has no such class and AutoTokenizer raises. AutoTokenizer's resolver
# falls back to getattr(transformers, class_name), so aliasing it to
# PreTrainedTokenizerFast (which loads tokenizer.json directly) fixes all
# call sites in this package at once.
try:
    import transformers
    if not hasattr(transformers, "TokenizersBackend"):
        transformers.TokenizersBackend = transformers.PreTrainedTokenizerFast
except Exception:
    pass
