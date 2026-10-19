"""Step 01: download the PARIKSHA source repositories at pinned commits and verify them.

For each source in config/sources.json this script:
  1. fetches exactly the pinned commit (a shallow fetch, so only that snapshot is downloaded);
  2. checks it out with line-ending conversion switched off, so the files are byte-identical
     on Windows, macOS and Linux;
  3. checks every file the pipeline uses against its expected SHA-256 checksum;
  4. writes data/raw/MANIFEST.json, recording what was downloaded and when.

If a commit or a checksum does not match, the script stops with an error, so an analysis can
never run on data that differs from the data described in the report.

Usage:  python scripts/01_download_data.py
Needs:  git on the PATH and an internet connection. (MuRIL weights are downloaded later, in step 05.)
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "sources.json"
RAW = ROOT / "data" / "raw"


def git(args: list[str], cwd: Path) -> str:
    """Run one git command in `cwd` and return its output; stop with a clear message if it fails."""
    env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}  # the pipeline never uses the repos' Git LFS files
    result = subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"git {' '.join(args)} failed in {cwd}:\n{result.stderr.strip()}")
    return result.stdout.strip()


def sha256(path: Path) -> str:
    """SHA-256 checksum of a file, read in 1 MB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(name: str, source: dict) -> Path:
    """Make data/raw/<name> an exact checkout of the pinned commit and return its path."""
    dest = RAW / name
    if not (dest / ".git").exists():
        dest.mkdir(parents=True, exist_ok=True)
        git(["init", "-q"], dest)
        git(["remote", "add", "origin", source["url"]], dest)
    # Keep LF line endings, otherwise Git on Windows changes the bytes and the checksums fail.
    git(["config", "core.autocrlf", "false"], dest)
    git(["config", "core.eol", "lf"], dest)
    # Some sources (Microsoft's PARIKSHA) track *.csv with Git LFS, but the pipeline never uses
    # those LFS files (every checksum-verified file is JSON/TSV, or a plain non-LFS CSV from Karya).
    # Neutralise the LFS filter locally so checkout works even when the git-lfs binary is absent:
    # a globally-registered "required" LFS filter would otherwise abort checkout with
    # "git-lfs: command not found". LFS-tracked files are left as their small pointer text, which is
    # harmless because none of them are read or verified by this pipeline.
    git(["config", "filter.lfs.required", "false"], dest)
    git(["config", "filter.lfs.smudge", "cat"], dest)
    git(["config", "filter.lfs.clean", "cat"], dest)
    git(["config", "filter.lfs.process", ""], dest)
    git(["fetch", "-q", "--depth", "1", "origin", source["commit"]], dest)
    git(["checkout", "-q", "--force", "--detach", source["commit"]], dest)
    head = git(["rev-parse", "HEAD"], dest)
    if head != source["commit"]:
        sys.exit(f"[{name}] expected commit {source['commit']}, but got {head}")
    return dest


def verify(name: str, source: dict, dest: Path) -> dict:
    """Check each listed file against its expected SHA-256; return the details for the manifest."""
    details, problems = {}, []
    for rel_path, expected in source["files"].items():
        path = dest / rel_path
        if not path.is_file():
            problems.append(f"missing file: {rel_path}")
            continue
        actual = sha256(path)
        details[rel_path] = {"sha256": actual, "bytes": path.stat().st_size}
        if actual != expected:
            problems.append(f"checksum mismatch: {rel_path}")
    if problems:
        sys.exit(f"[{name}] verification failed:\n  " + "\n  ".join(problems))
    return details


def main() -> None:
    sources = json.loads(CONFIG.read_text(encoding="utf-8"))
    manifest = {
        "downloaded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "sources": {},
    }
    for name, source in sources.items():
        print(f"[{name}] fetching {source['url']} at commit {source['commit'][:7]} ...")
        dest = fetch(name, source)
        files = verify(name, source, dest)
        print(f"[{name}] OK: commit verified and {len(files)} files match their SHA-256 checksums")
        manifest["sources"][name] = {
            "url": source["url"],
            "commit": source["commit"],
            "licence": source["licence"],
            "files": files,
        }
    out = RAW / "MANIFEST.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
