#!/usr/bin/env python3
"""Create shard directories and per-shard config files derived from a base config."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

from split_patches import shard_slides


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "configs",
        nargs="+",
        type=Path,
        help="Base YAML config files to shard",
    )
    parser.add_argument("-p", "--parts", type=int, default=3, help="Number of shards per config")
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Recreate shard directories even if they already exist",
    )
    return parser.parse_args()


def shard_config(config_path: Path, *, parts: int, force: bool):
    data = yaml.safe_load(config_path.read_text())
    src_folder = Path(data["src_folder"]).resolve()
    dst_file = data["dst_file"]

    shard_dirs = shard_slides(
        src_folder,
        parts=parts,
        out_root=src_folder.parent,
        prefix=f"{src_folder.name}_part",
        force=force,
    )

    new_configs = []
    for idx, shard in enumerate(shard_dirs, start=1):
        shard_cfg = copy.deepcopy(data)
        shard_cfg["src_folder"] = str(shard)
        shard_cfg["dst_file"] = f"{dst_file}_part{idx}"
        out_path = config_path.with_name(f"{config_path.stem}_part{idx}{config_path.suffix}")
        out_path.write_text(yaml.safe_dump(shard_cfg, sort_keys=False))
        new_configs.append(out_path)

    return new_configs


def main():
    args = parse_args()
    for cfg in args.configs:
        shards = shard_config(cfg, parts=args.parts, force=args.force)
        print(f"Generated {len(shards)} configs from {cfg}:")
        for shard_cfg in shards:
            print(f" - {shard_cfg}")


if __name__ == "__main__":
    main()
