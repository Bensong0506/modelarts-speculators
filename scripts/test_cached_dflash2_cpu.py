"""Isolated component tests without importing unrelated Transformers model families.

This does NOT validate the installed Speculators CLI or CUDA/vLLM execution.
Run pytest tests/unit/train/cached_dflash2 in the supported Linux dev environment
for the normal package-import path. This runner also works on Intel macOS with
torch 2.2.2, numpy<2, pydantic>=2, PyYAML and safetensors.
"""

import sys
import types
import unittest
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    for name, directory in (
        ("speculators", root / "src/speculators"),
        ("speculators.train", root / "src/speculators/train"),
    ):
        package = types.ModuleType(name)
        package.__path__ = [str(directory)]
        sys.modules[name] = package
    suite = unittest.defaultTestLoader.discover(
        str(root / "tests/unit/train/cached_dflash2"), top_level_dir=str(root)
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())


if __name__ == "__main__":
    main()
