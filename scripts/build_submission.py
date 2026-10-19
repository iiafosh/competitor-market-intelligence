"""Build the submission ZIP: one parent folder, no secrets, all mandatory files.

Usage (from the project root):
    python scripts/build_submission.py

The archive is written next to the project folder as ``<FolderName>.zip`` and is re-opened
and checked before the script reports success.
"""

from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
FOLDER_NAME = PROJECT_DIR.name
ZIP_PATH = PROJECT_DIR.parent / f"{FOLDER_NAME}.zip"

NAME_PATTERN = re.compile(r"^[A-Za-z]+_Module1_2_Capstone\.zip$")
EXCLUDED_DIRS = {".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache", "cache", ".git", ".idea", ".vscode", "dist"}
EXCLUDED_FILES = {".env"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".zip", ".tmp"}
MANDATORY = (
    "README.md",
    "requirements.txt",
    ".env.example",
    "src/main.py",
    "outputs/enriched_products.csv",
    "outputs/full_analysis.json",
)
# OpenAI/OpenRouter-style secret keys (sk-..., sk-proj-..., sk-svcacct-..., sk-admin-..., sk-or-v1-...).
SECRET_PATTERN = re.compile(r"sk-(?:proj-|svcacct-|admin-)?[A-Za-z0-9_\-]{20,}")
ALLOWLIST_MARKER = "pragma: allowlist secret"  # detect-secrets convention for deliberate fake test keys
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".csv", ".ini", ".log", ".example", ".gitignore", ".toml", ".cfg", ".html"}


def iter_project_files() -> list[Path]:
    files = []
    for path in sorted(PROJECT_DIR.rglob("*")):
        relative = path.relative_to(PROJECT_DIR)
        if any(part in EXCLUDED_DIRS for part in relative.parts):
            continue
        if path.is_dir() or path.name in EXCLUDED_FILES or path.suffix in EXCLUDED_SUFFIXES:
            continue
        if path.name.startswith(".env") and path.name != ".env.example":
            continue
        files.append(path)
    return files


def scan_for_secrets(files: list[Path]) -> list[str]:
    findings = []
    for path in files:
        if path.suffix not in TEXT_SUFFIXES and path.name not in {".env.example", ".gitignore"}:
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
            if ALLOWLIST_MARKER in line:
                continue
            for match in SECRET_PATTERN.finditer(line):
                findings.append(f"{path.relative_to(PROJECT_DIR)}:{number}: possible API key '{match.group()[:8]}...'")
    return findings


def main() -> int:
    if not NAME_PATTERN.match(ZIP_PATH.name):
        print(f"ERROR: '{ZIP_PATH.name}' does not match [FullName]_Module1_2_Capstone.zip")
        return 1
    files = iter_project_files()
    relative_names = {path.relative_to(PROJECT_DIR).as_posix() for path in files}
    missing = [name for name in MANDATORY if name not in relative_names]
    if missing:
        print(f"ERROR: mandatory files missing: {missing}")
        return 1
    secrets = scan_for_secrets(files)
    if secrets:
        print("ERROR: refusing to package - possible secrets found:\n  " + "\n  ".join(secrets))
        return 1

    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, f"{FOLDER_NAME}/{path.relative_to(PROJECT_DIR).as_posix()}")

    with zipfile.ZipFile(ZIP_PATH) as archive:
        broken = archive.testzip()
        names = archive.namelist()
    top_level = {name.split("/", 1)[0] for name in names}
    problems = []
    if broken:
        problems.append(f"corrupt member: {broken}")
    if top_level != {FOLDER_NAME}:
        problems.append(f"expected a single parent folder, found {sorted(top_level)}")
    if any(name.endswith("/.env") for name in names):
        problems.append(".env found inside the archive")
    if problems:
        print("ERROR: " + "; ".join(problems))
        return 1

    size_kb = ZIP_PATH.stat().st_size / 1024
    print(f"Created {ZIP_PATH} ({len(names)} files, {size_kb:.0f} KB); single parent folder '{FOLDER_NAME}/', no .env, no keys found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
