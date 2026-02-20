#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE = REPO_ROOT / ".github" / "RELEASE_TEMPLATE.md"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"


def _run(command: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess:
    print("$", " ".join(command))
    return subprocess.run(command, cwd=REPO_ROOT, check=check)


def _ensure_clean_worktree() -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise RuntimeError("Worktree is not clean. Commit/stash changes first or pass --allow-dirty.")


def _validate_version(version: str) -> None:
    if not version.startswith("v"):
        raise ValueError("Version tag must start with 'v' (example: v1.0.0-rc1).")


def cmd_gate(args: argparse.Namespace) -> None:
    _run([sys.executable, "-m", "ruff", "check", "multimodal_lm_eap_ig", "tests", "scripts"])
    _run([sys.executable, "-m", "mypy"])
    _run([sys.executable, "-m", "pytest", "-q"])
    if not args.skip_build:
        cmd_build(args)
        cmd_check(args)


def cmd_build(args: argparse.Namespace) -> None:
    if args.clean_dist:
        shutil.rmtree(REPO_ROOT / "dist", ignore_errors=True)
    _run([sys.executable, "-m", "build"])


def cmd_check(args: argparse.Namespace) -> None:
    del args
    dist_dir = REPO_ROOT / "dist"
    artifacts = sorted(dist_dir.glob("*"))
    if not artifacts:
        raise RuntimeError("No artifacts found under dist/. Run release.py build first.")
    _run([sys.executable, "-m", "twine", "check", *[str(path) for path in artifacts]])


def cmd_notes(args: argparse.Namespace) -> None:
    _validate_version(args.version)
    template_path = Path(args.template).resolve()
    if not template_path.exists():
        raise FileNotFoundError(f"Release template not found: {template_path}")

    output_path = Path(args.output).resolve() if args.output else (REPO_ROOT / "releases" / f"{args.version}.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    template = template_path.read_text(encoding="utf-8")
    rendered = (
        template.replace("{{VERSION}}", args.version)
        .replace("{{DATE_UTC}}", datetime.now(timezone.utc).date().isoformat())
        .replace("{{TAG}}", args.version)
    )
    output_path.write_text(rendered, encoding="utf-8")
    print(f"Wrote release notes draft to: {output_path}")


def cmd_tag(args: argparse.Namespace) -> None:
    _validate_version(args.version)
    if not args.allow_dirty:
        _ensure_clean_worktree()

    existing = subprocess.run(
        ["git", "tag", "--list", args.version],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if existing and not args.force:
        raise RuntimeError(f"Tag already exists: {args.version}. Use --force to recreate.")
    if existing and args.force:
        _run(["git", "tag", "-d", args.version])

    message = args.message or args.version
    _run(["git", "tag", "-a", args.version, "-m", message])
    if args.push:
        if args.force:
            _run(["git", "push", "--force", "origin", f"refs/tags/{args.version}"])
        else:
            _run(["git", "push", "origin", args.version])


def cmd_set_version(args: argparse.Namespace) -> None:
    if not PYPROJECT_PATH.exists():
        raise FileNotFoundError(f"Missing pyproject.toml at {PYPROJECT_PATH}")
    raw = PYPROJECT_PATH.read_text(encoding="utf-8")
    updated, count = re.subn(
        r'(?m)^version\s*=\s*"[^"]+"',
        f'version = "{args.version}"',
        raw,
        count=1,
    )
    if count != 1:
        raise RuntimeError("Could not find unique `version = ...` line in pyproject.toml")
    PYPROJECT_PATH.write_text(updated, encoding="utf-8")
    print(f"Updated pyproject version to {args.version}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Release engineering utilities for mm-eap.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gate = subparsers.add_parser("gate", help="Run release gate: ruff + mypy + pytest (+ build/check).")
    gate.add_argument("--skip-build", action="store_true", help="Skip build/twine checks.")
    gate.add_argument("--clean-dist", action="store_true", help="Clean dist/ before build.")
    gate.set_defaults(func=cmd_gate)

    build = subparsers.add_parser("build", help="Build sdist/wheel.")
    build.add_argument("--clean-dist", action="store_true", help="Clean dist/ before build.")
    build.set_defaults(func=cmd_build)

    check = subparsers.add_parser("check", help="Run twine metadata checks on dist/*.")
    check.set_defaults(func=cmd_check)

    notes = subparsers.add_parser("notes", help="Render release notes from template.")
    notes.add_argument("--version", required=True, help="Release version tag (example: v1.0.0-rc1).")
    notes.add_argument("--template", default=str(DEFAULT_TEMPLATE), help="Template path.")
    notes.add_argument("--output", default="", help="Output markdown path.")
    notes.set_defaults(func=cmd_notes)

    tag = subparsers.add_parser("tag", help="Create annotated git tag (optionally push).")
    tag.add_argument("--version", required=True, help="Release version tag (example: v1.0.0-rc1).")
    tag.add_argument("--message", default="", help="Annotated tag message.")
    tag.add_argument("--push", action="store_true", help="Push tag to origin.")
    tag.add_argument("--force", action="store_true", help="Recreate tag if it already exists.")
    tag.add_argument("--allow-dirty", action="store_true", help="Allow tagging with non-clean worktree.")
    tag.set_defaults(func=cmd_tag)

    set_version = subparsers.add_parser("set-version", help="Update version in pyproject.toml.")
    set_version.add_argument(
        "--version",
        required=True,
        help='PEP 440 version (example: 1.0.0rc1 or 1.0.0). For git tags use command "tag".',
    )
    set_version.set_defaults(func=cmd_set_version)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
