# Version-pinned vLLM dependency for dense DFlash2

Use [the dense inference guide](../../docs/cached_dflash2/VLLM_INFERENCE.md) and `scripts/vllm_dflash2_runtime.py`. This directory is source overlay data, not an installable vLLM wheel.

The base is `bbf5893213eff1e1dc7bbe76ce410b31eaca1507`; all 22 overlay files and `source-provenance.json` are preserved byte-for-byte from the recorded runtime. Some vendor modules implement MTP/hybrid internals. They remain solely to preserve the complete pinned dependency; this branch's serving/evaluation wrappers accept only target/dense and do not enable those paths.

Do not format vendor files, crop the dependency tree, or change recorded hashes to hide a check failure. Use a personal checkout and matching compiled extensions. Historical output equivalence has not been established; a successful load or acceptance measurement does not prove lossless acceleration.
