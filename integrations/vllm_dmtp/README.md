# Version-pinned D-MTP vLLM overlay

See [the inference guide](../../docs/cached_dflash2/VLLM_INFERENCE.md) for installation, serving, evaluation and limitations. This directory is not an installable vLLM package.

Base commit: `bbf5893213eff1e1dc7bbe76ce410b31eaca1507` of the original private Qwen-capable fork. The model-specific base and matching CUDA/PyTorch environment must be supplied separately. An upstream checkout or matching version string alone is insufficient.

A [complete Git patch](patches/README.md) is also provided: 14 modified files and 8 added files, with installation commands and SHA-256 metadata. It was applied to the exact base and verified to reproduce all 22 overlay files byte for byte. Use either that patch or the overlay installer, not both. Historical `online/*/vllm.patch` logs omit untracked added files and are not standalone installers.

`source-provenance.json` lists 22 exact file hashes. Nineteen files came from the downloaded server overlay archive; three residual-fusion updates were checked against the final server audit. The overlay also contains earlier DFlash2/n-gram plumbing needed by that checkout; inclusion does not mean n-gram is enabled in these experiments. Existing SPDX notices are retained. The training integration and launch/eval wrappers are separate from these byte-preserved files.

Do not format or apply this tree blindly. `scripts/vllm_dmtp_runtime.py` checks the base, local modifications and hashes before application. The historical runtime has unresolved cross-GPU D-MTP output differences; see the inference guide before interpreting acceptance results.
