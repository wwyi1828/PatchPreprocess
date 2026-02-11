#!/usr/bin/env python3
"""Create separate directories with symlinks for 20x and 40x files"""

from pathlib import Path
import os

source_dir = Path('/data/data/SlideBench/wsi_stage1_seed42_n1000')
output_base = Path('/data/data/SlideBench_split')

# Create output directories
dir_20x = output_base / 'SlideBench_20x_files'
dir_40x = output_base / 'SlideBench_40x_files'

dir_20x.mkdir(parents=True, exist_ok=True)
dir_40x.mkdir(parents=True, exist_ok=True)

print(f"Creating symlink directories...")
print(f"Source: {source_dir}")
print(f"20x output: {dir_20x}")
print(f"40x output: {dir_40x}")

# Read file lists
with open('file_lists/20x_files.txt', 'r') as f:
    files_20x = [line.strip() for line in f if line.strip()]

with open('file_lists/40x_files.txt', 'r') as f:
    files_40x = [line.strip() for line in f if line.strip()]

# Create symlinks for 20x files
print(f"\nCreating {len(files_20x)} symlinks for 20x files...")
for filename in files_20x:
    src_file = source_dir / filename
    dst_file = dir_20x / filename

    if dst_file.exists() or dst_file.is_symlink():
        dst_file.unlink()

    if src_file.exists():
        dst_file.symlink_to(src_file)
    else:
        print(f"Warning: {src_file} not found")

print(f"✓ Created {len(list(dir_20x.iterdir()))} symlinks in {dir_20x}")

# Create symlinks for 40x files
print(f"\nCreating {len(files_40x)} symlinks for 40x files...")
for filename in files_40x:
    src_file = source_dir / filename
    dst_file = dir_40x / filename

    if dst_file.exists() or dst_file.is_symlink():
        dst_file.unlink()

    if src_file.exists():
        dst_file.symlink_to(src_file)
    else:
        print(f"Warning: {src_file} not found")

print(f"✓ Created {len(list(dir_40x.iterdir()))} symlinks in {dir_40x}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"20x files directory: {dir_20x}")
print(f"  Symlinks created: {len(list(dir_20x.iterdir()))}")
print(f"\n40x files directory: {dir_40x}")
print(f"  Symlinks created: {len(list(dir_40x.iterdir()))}")
print("\nYou can now run CLAM patching on these directories separately.")
