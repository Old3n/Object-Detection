"""Build and optionally publish a Windows release without GitHub Actions."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "Old3n/Object-Detection"


def run(*args: str, capture: bool = False) -> str:
    result = subprocess.run(
        [str(arg) for arg in args], cwd=ROOT, check=True,
        stdout=subprocess.PIPE if capture else None, text=True,
    )
    return result.stdout.strip() if capture else ""


def version_tag(value: str) -> str:
    if not re.fullmatch(r"v\d+\.\d+\.\d+(?:-[A-Za-z0-9]+(?:[.-][A-Za-z0-9]+)*)?", value):
        raise argparse.ArgumentTypeError("Use a version such as v1.0.0 or v1.0.0-beta.1.")
    return value


def clean_commit() -> str:
    if run("git", "status", "--porcelain", capture=True):
        raise RuntimeError("Commit or stash local changes before publishing. Release files must match the GitHub commit.")
    return run("git", "rev-parse", "HEAD", capture=True)


def prepare_publish(version: str) -> str:
    for command in ("git", "gh"):
        if shutil.which(command) is None:
            raise RuntimeError(f"Install {command} and reopen your terminal before publishing.")
    commit = clean_commit()
    try:
        run("gh", "auth", "status", "--hostname", "github.com")
    except subprocess.CalledProcessError:
        run("gh", "auth", "login", "--hostname", "github.com", "--git-protocol", "https", "--web")
    permission = run("gh", "repo", "view", REPOSITORY, "--json", "viewerPermission", "--jq", ".viewerPermission", capture=True)
    if permission not in {"ADMIN", "MAINTAIN", "WRITE"}:
        raise RuntimeError(f"The signed-in GitHub account cannot publish to {REPOSITORY}.")
    remote_commit = run("gh", "api", f"repos/{REPOSITORY}/commits/{commit}", "--jq", ".sha", capture=True)
    if remote_commit != commit:
        raise RuntimeError("Push this commit to GitHub before publishing.")
    refs = json.loads(run("gh", "api", f"repos/{REPOSITORY}/git/matching-refs/tags/{version}", capture=True))
    if any(ref["ref"] == f"refs/tags/{version}" for ref in refs):
        raise RuntimeError(f"Tag {version} already exists. Choose a new version; existing releases will not be overwritten.")
    releases = json.loads(run("gh", "release", "list", "--repo", REPOSITORY, "--limit", "1000", "--json", "tagName", capture=True))
    if any(release["tagName"] == version for release in releases):
        raise RuntimeError(f"Release {version} already exists, possibly as a draft. Choose a new version or finish that draft manually.")
    return commit


def build(python: Path) -> list[Path]:
    run(python, "-m", "pip", "install", "--upgrade", "pip")
    run(python, "-m", "pip", "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu")
    run(python, "-m", "pip", "install", "-r", "requirements.txt", "pyinstaller>=6.11,<7")
    run(python, "-m", "unittest", "discover", "-s", "tests", "-v")
    run(python, "-c", "from ultralytics import YOLO; YOLO('yolo26n.pt')")
    run(python, "-m", "PyInstaller", "--noconfirm", "--clean", "AutoPilot.spec")
    executable = ROOT / "dist" / "AutoPilot.exe"
    run(executable, "--self-test")
    run(executable, "--tool", "risk", "--help")
    run(executable, "--tool", "surround", "--help")
    run(python, "tools/package_pwa.py")
    inventory = ROOT / "dist" / "build-dependencies.txt"
    inventory.write_text(run(python, "-m", "pip", "freeze", capture=True) + "\n", encoding="utf-8")
    assets = [executable, ROOT / "dist" / "AutoPilot-pwa.zip", inventory]
    checksums = ROOT / "dist" / "SHA256SUMS.txt"
    lines = []
    for path in assets:
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        lines.append(f"{digest}  {path.name}")
    checksums.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return assets + [checksums]


def publish(version: str, commit: str, assets: list[Path]) -> None:
    if clean_commit() != commit:
        raise RuntimeError("The checkout changed during the build. Release upload cancelled.")
    notes = ROOT / "dist" / "release-notes.md"
    notes.write_text(
        f"Built locally on Windows from `{commit}`. No GitHub Actions runner was used.\n\n"
        "- AutoPilot.exe: single-file Windows x64 CPU build with the default YOLO26n model.\n"
        "- AutoPilot-pwa.zip: Android-installable web UI and Python backend source. "
        "Android requires a running backend over trusted HTTPS; this is not an APK.\n"
        "- SHA256SUMS.txt and build-dependencies.txt: checksums and resolved build dependencies.\n\n"
        "The unit tests and executable inference/CLI self-tests passed before upload. "
        "Physical camera and Android device behavior are not validated by those tests.\n",
        encoding="utf-8",
    )
    args = ["gh", "release", "create", version, *map(str, assets),
            "--repo", REPOSITORY, "--target", commit, "--title", f"AutoPilot {version}",
            "--notes-file", str(notes)]
    if "-" in version:
        args.append("--prerelease")
    run(*args)
    print(f"Published: https://github.com/{REPOSITORY}/releases/tag/{version}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true", help="Upload tested assets to a new GitHub Release.")
    parser.add_argument("--version", type=version_tag, help="New release tag, required with --publish.")
    args = parser.parse_args()
    if args.publish and not args.version:
        parser.error("--publish requires --version vX.Y.Z")
    if sys.platform != "win32" or platform.machine().lower() not in {"amd64", "x86_64"}:
        parser.error("Run this script on Windows x64. Windows executables cannot be built by this script on Linux/macOS.")
    if sys.version_info[:2] != (3, 11) or sys.maxsize <= 2**32:
        parser.error("Run with 64-bit Python 3.11: py -3.11 tools/build_release.py")
    try:
        commit = prepare_publish(args.version) if args.publish else None
        with tempfile.TemporaryDirectory(prefix="autopilot-build-") as directory:
            environment = Path(directory) / "venv"
            run(sys.executable, "-m", "venv", environment)
            assets = build(environment / "Scripts" / "python.exe")
        print(f"Verified release files are in {ROOT / 'dist'}")
        if args.publish:
            publish(args.version, commit, assets)
        return 0
    except (OSError, RuntimeError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        print(f"Release stopped: {exc}", file=sys.stderr)
        print("Existing releases are never overwritten. If an upload was interrupted, inspect GitHub Releases for a partial draft.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
