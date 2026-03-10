#!/usr/bin/env python3
"""
Tidy the Mastering-RL project root: archive large zips, move artifacts, delete caches.

Why this exists
---------------
The repo root accumulates:
- zip archives (deployment/env/assets)
- artifact folders (figures/videos)
- python caches (__pycache__)
- temp conda export files (condaenv.*.requirements.txt)
- migrated old script folders (past_scripts/)

This tool makes the root cleaner without changing Python source layout.

Default is dry-run. Use --apply to actually perform changes.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass
class Action:
    kind: str  # move|delete|symlink|mkdir|skip
    src: str
    dest: str
    note: str = ""


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _move(src: Path, dest: Path) -> None:
    _safe_mkdir(dest.parent)
    shutil.move(str(src), str(dest))


def _rm_rf(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path, ignore_errors=True)


def _make_symlink(link_path: Path, target: Path) -> None:
    # link_path -> target
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    link_path.symlink_to(target)


def _iter_conda_req_files(repo_root: Path) -> Iterable[Path]:
    for p in repo_root.glob("condaenv.*.requirements.txt"):
        if p.is_file():
            yield p


def plan(repo_root: Path, keep_symlinks: bool) -> List[Action]:
    actions: List[Action] = []

    archives = repo_root / "archives"
    artifacts = repo_root / "artifacts"

    actions.append(Action("mkdir", str(archives), str(archives), "create archives/"))
    actions.append(Action("mkdir", str(artifacts), str(artifacts), "create artifacts/"))

    # 1) Delete caches
    pycache = repo_root / "__pycache__"
    if pycache.exists():
        actions.append(Action("delete", str(pycache), "", "delete python cache (regenerates automatically)"))

    # 2) Archive zip files (keep folder `deployment_package/` as-is)
    for zip_name in ["deployment_package.zip", "FetchPickAndPlace-v4.zip"]:
        z = repo_root / zip_name
        if z.exists():
            actions.append(Action("move", str(z), str(archives / zip_name), "archive zip"))

    # 3) Move temp conda export files
    conda_dir = archives / "conda"
    for f in _iter_conda_req_files(repo_root):
        actions.append(Action("move", str(f), str(conda_dir / f.name), "archive conda export requirements"))

    # 4) Move artifact folders (and optionally keep symlink for compatibility)
    for dname in ["figures", "videos"]:
        d = repo_root / dname
        if not d.exists():
            continue

        dest = artifacts / dname
        # If already a symlink pointing to artifacts/, skip
        if d.is_symlink():
            actions.append(Action("skip", str(d), str(dest), "already a symlink; leaving as-is"))
            continue

        actions.append(Action("move", str(d), str(dest), "move artifacts folder"))
        if keep_symlinks:
            actions.append(Action("symlink", str(d), str(dest), "create symlink to keep old path working"))

    # 5) past_scripts is migrated; archive it (optional symlink not needed)
    past = repo_root / "past_scripts"
    if past.exists():
        actions.append(Action("move", str(past), str(archives / "past_scripts_migrated"), "archive migrated past_scripts/"))

    return actions


def apply_actions(actions: List[Action], repo_root: Path, manifest_path: Path, do_apply: bool) -> None:
    # Print plan
    for a in actions:
        if a.kind == "mkdir":
            print(f"[mkdir] {a.src}")
        elif a.kind == "delete":
            print(f"[delete] {a.src}  # {a.note}")
        elif a.kind == "move":
            print(f"[move] {a.src} -> {a.dest}  # {a.note}")
        elif a.kind == "symlink":
            print(f"[symlink] {a.src} -> {a.dest}  # {a.note}")
        else:
            print(f"[skip] {a.src}  # {a.note}")

    # Write manifest (always)
    _safe_mkdir(manifest_path.parent)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["kind", "src", "dest", "note"])
        w.writeheader()
        for a in actions:
            w.writerow({"kind": a.kind, "src": a.src, "dest": a.dest, "note": a.note})

    print(f"\n[manifest] wrote: {manifest_path}")

    if not do_apply:
        print("\n[dry-run] no changes applied. Re-run with --apply to execute.")
        return

    # Execute actions in a safe order:
    # - mkdirs first
    # - moves next
    # - deletes
    # - symlinks last
    for a in actions:
        if a.kind == "mkdir":
            _safe_mkdir(Path(a.src))

    for a in actions:
        if a.kind == "move":
            src = Path(a.src)
            dest = Path(a.dest)
            if not src.exists():
                continue
            _move(src, dest)

    for a in actions:
        if a.kind == "delete":
            _rm_rf(Path(a.src))

    for a in actions:
        if a.kind == "symlink":
            link_path = Path(a.src)
            target = Path(a.dest)
            # Make relative symlink if possible (nicer when moving repo)
            try:
                rel_target = Path(target).relative_to(link_path.parent)
                _make_symlink(link_path, rel_target)
            except Exception:
                _make_symlink(link_path, target)

    print("\n[apply] done.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="Actually perform changes (default: dry-run).")
    ap.add_argument(
        "--repo-root",
        type=str,
        default=str(Path(__file__).resolve().parents[1]),
        help="Path to Mastering-RL directory (default: inferred).",
    )
    ap.add_argument(
        "--no-symlinks",
        action="store_true",
        help="Do not create symlinks for moved artifact dirs (figures/videos).",
    )
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    actions = plan(repo_root=repo_root, keep_symlinks=(not args.no_symlinks))
    manifest_path = repo_root / "archives" / "tidy_manifest.csv"
    apply_actions(actions, repo_root=repo_root, manifest_path=manifest_path, do_apply=args.apply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



