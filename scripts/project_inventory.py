#!/usr/bin/env python
"""Read-only inventory of the SENTINEL 2.0 project."""

from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone

EXCLUDED_DIRS = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", "node_modules",
}
REPORT_DIR = Path("reports") / "project_audit"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"Project root does not exist: {root}")

    files = []
    directories = []
    errors = []

    for current, dirnames, filenames in os.walk(root, onerror=lambda e: errors.append(str(e))):
        current_path = Path(current)
        dirnames[:] = sorted(
            name for name in dirnames
            if name not in EXCLUDED_DIRS
        )

        relative_dir = current_path.relative_to(root)
        if str(relative_dir) != ".":
            directories.append(relative_dir.as_posix())

        for filename in sorted(filenames):
            path = current_path / filename
            relative = path.relative_to(root)

            # Avoid including this script's own generated reports in inventory.
            if relative.parts[:2] == ("reports", "project_audit"):
                continue

            try:
                stat = path.stat()
                record = {
                    "path": relative.as_posix(),
                    "extension": path.suffix.lower() or "[no extension]",
                    "size_bytes": stat.st_size,
                }
                if path.suffix.lower() in {".py", ".toml", ".yml", ".yaml", ".json"}:
                    record["sha256"] = sha256_file(path)
                files.append(record)
            except OSError as exc:
                errors.append(f"{relative.as_posix()}: {exc}")

    files.sort(key=lambda item: item["path"].lower())
    directories.sort(key=str.lower)

    extension_counts = {}
    for item in files:
        extension_counts[item["extension"]] = extension_counts.get(item["extension"], 0) + 1

    root_files = sorted(
        item["path"] for item in files
        if "/" not in item["path"]
    )

    report = {
        "audit_type": "read_only_project_inventory",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "python_version": sys.version,
        "platform": platform.platform(),
        "summary": {
            "directory_count": len(directories),
            "file_count": len(files),
            "total_file_size_bytes": sum(item["size_bytes"] for item in files),
            "extension_counts": dict(sorted(extension_counts.items())),
            "root_files": root_files,
        },
        "directories": directories,
        "files": files,
        "errors": errors,
        "excluded_directory_names": sorted(EXCLUDED_DIRS),
    }

    report_dir = root / REPORT_DIR
    report_dir.mkdir(parents=True, exist_ok=True)

    json_path = report_dir / "project_inventory.json"
    text_path = report_dir / "project_inventory.txt"

    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "SENTINEL 2.0 — READ-ONLY PROJECT INVENTORY",
        f"Project root: {root}",
        f"Generated UTC: {report['generated_at_utc']}",
        "",
        "SUMMARY",
        f"Directories: {len(directories)}",
        f"Files: {len(files)}",
        f"Total file size: {report['summary']['total_file_size_bytes']:,} bytes",
        "",
        "ROOT FILES",
        *[f"  {name}" for name in root_files],
        "",
        "DIRECTORIES",
        *[f"  {name}/" for name in directories],
        "",
        "FILES",
        *[
            f"  {item['path']}  |  {item['size_bytes']:,} bytes"
            for item in files
        ],
        "",
        "ERRORS",
        *([f"  {error}" for error in errors] or ["  None"]),
        "",
        "Note: .git, virtual environments, caches, and node_modules were excluded.",
    ]
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("SENTINEL 2.0 inventory complete")
    print(f"Project root: {root}")
    print(f"Directories: {len(directories)}")
    print(f"Files: {len(files)}")
    print(f"Read errors: {len(errors)}")
    print(f"JSON report: {json_path}")
    print(f"Text report: {text_path}")

    if errors:
        print("WARNING: Some files could not be inventoried. Inspect the report.")


if __name__ == "__main__":
    main()
