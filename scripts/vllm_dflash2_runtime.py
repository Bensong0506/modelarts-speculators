"""Install/check the version-pinned experimental vLLM overlay in a private checkout."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

INTEGRATION = Path(__file__).resolve().parents[1] / "integrations" / "vllm_dflash2"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(root, *args):
    return subprocess.run(  # noqa: S603
        ["git", "-C", str(root), *args],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout.strip()


def manifest():
    audit = json.loads((INTEGRATION / "source-provenance.json").read_text())
    for item in audit["files"]:
        source = INTEGRATION / "overlay" / item["path"]
        if sha256(source) != item["sha256"]:
            raise ValueError(f"Bundled overlay hash mismatch: {item['path']}")
    return audit


def check_runtime(root):
    root = Path(root).resolve()
    audit = manifest()
    head = git(root, "rev-parse", "HEAD")
    if head != audit["base_commit"]:
        raise ValueError(
            f"Expected private vLLM base {audit['base_commit']}; found {head}"
        )
    for item in audit["files"]:
        file = root / item["path"]
        if not file.is_file() or sha256(file) != item["sha256"]:
            raise ValueError(f"Missing or changed runtime file: {item['path']}")
    # Changes outside the archived files would invalidate the version claim.
    changed = set(git(root, "diff", "HEAD", "--name-only").splitlines())
    changed.update(git(root, "ls-files", "--others", "--exclude-standard").splitlines())
    allowed = {item["path"] for item in audit["files"]}
    if changed - allowed:
        raise ValueError(f"Unreviewed runtime changes: {sorted(changed - allowed)}")
    return {"base_commit": head, "verified_overlay_files": len(audit["files"])}


def apply_overlay(root, apply=False):
    root = Path(root).resolve()
    if root == Path("/opt/b300") or Path("/opt/b300") in root.parents:
        raise ValueError(
            "Use a personal checkout; shared /opt/b300 is not an install target"
        )
    audit = manifest()
    if git(root, "rev-parse", "HEAD") != audit["base_commit"]:
        raise ValueError(
            "Wrong vLLM base commit; do not overlay arbitrary pip/upstream vLLM"
        )
    if git(root, "status", "--porcelain", "--untracked-files=normal"):
        raise ValueError("Overlay application requires a clean private checkout")
    if apply:
        for item in audit["files"]:
            target = root / item["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(INTEGRATION / "overlay" / item["path"], target)
        return check_runtime(root)
    return {
        "dry_run": True,
        "base_commit": audit["base_commit"],
        "files": len(audit["files"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = (
        check_runtime(args.runtime_root)
        if args.check
        else apply_overlay(args.runtime_root, args.apply)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
