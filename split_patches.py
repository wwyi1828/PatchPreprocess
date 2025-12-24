#!/usr/bin/env python3
"""Utility to split a slide patch directory into evenly sized shards via symlinks."""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path


def split_evenly(items, parts):
    """Yield ``parts`` slices whose lengths differ by at most one item."""
    base = len(items) // parts
    remainder = len(items) % parts
    start = 0
    for idx in range(parts):
        extra = 1 if idx < remainder else 0
        end = start + base + extra
        yield items[start:end]
        start = end


def prepare_dest_dir(dest: Path, *, force: bool):
    if dest.exists():
        if not dest.is_dir():
            raise ValueError(f"{dest} exists and is not a directory")
        if force:
            shutil.rmtree(dest)
        elif any(dest.iterdir()):
            raise ValueError(
                f"{dest} already exists and is not empty. Pass --force to overwrite."
            )
    dest.mkdir(parents=True, exist_ok=True)


def symlink_chunk(chunk, dest: Path):
    for slide in chunk:
        link_path = dest / slide.name
        if link_path.exists() or link_path.is_symlink():
            if link_path.resolve() == slide.resolve():
                continue
            raise FileExistsError(f"{link_path} already exists and points elsewhere")
        link_path.symlink_to(slide)


def shard_slides(src: Path, *, parts: int, out_root: Path | None, prefix: str, force: bool):
    if parts < 1:
        raise ValueError("parts must be >= 1")
    if not src.is_dir():
        raise FileNotFoundError(f"{src} is not a directory")

    slides = sorted(p for p in src.iterdir() if p.is_dir())
    if not slides:
        raise RuntimeError(f"{src} contains no slide directories to split")

    root = out_root if out_root is not None else src.parent
    root.mkdir(parents=True, exist_ok=True)

    shards = list(split_evenly(slides, parts))
    for idx, chunk in enumerate(shards):
        shard_dir = root / f"{prefix}{idx+1}"
        prepare_dest_dir(shard_dir, force=force)
        symlink_chunk(chunk, shard_dir)

    return [root / f"{prefix}{idx+1}" for idx in range(parts)]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src_folder", type=Path, help="Path to the original patches directory")
    parser.add_argument(
        "-p", "--parts", type=int, default=2, help="Number of shards to create (default: 2)"
    )
    parser.add_argument(
        "-o",
        "--output-root",
        type=Path,
        default=None,
        help="Directory under which shards will be created (default: src parent)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Prefix for shard directories (default: <src_name>_part)",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Remove existing shard directories before recreating them",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    prefix = args.prefix if args.prefix else f"{args.src_folder.name}_part"
    shard_dirs = shard_slides(
        args.src_folder.resolve(),
        parts=args.parts,
        out_root=args.output_root.resolve() if args.output_root else None,
        prefix=prefix,
        force=args.force,
    )
    print("Created shards:")
    for shard_path in shard_dirs:
        # Print relative paths if under cwd for readability
        try:
            display = shard_path.relative_to(Path.cwd())
        except ValueError:
            display = shard_path
        print(f" - {display}")


if __name__ == "__main__":
    main()
