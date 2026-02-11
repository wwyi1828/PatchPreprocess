#!/bin/bash
# Clean up temporary symlink directories for SlideBench preprocessing

SPLIT_BASE="/data/data/SlideBench_split"

echo "================================================================================"
echo "SlideBench Symlink Cleanup"
echo "================================================================================"
echo ""

if [ ! -d "$SPLIT_BASE" ]; then
    echo "✓ No symlink directories found. Already clean!"
    exit 0
fi

echo "Found symlink directories:"
echo "  $SPLIT_BASE/"
if [ -d "$SPLIT_BASE/SlideBench_20x_files" ]; then
    echo "    - SlideBench_20x_files/ ($(ls $SPLIT_BASE/SlideBench_20x_files 2>/dev/null | wc -l) symlinks)"
fi
if [ -d "$SPLIT_BASE/SlideBench_40x_files" ]; then
    echo "    - SlideBench_40x_files/ ($(ls $SPLIT_BASE/SlideBench_40x_files 2>/dev/null | wc -l) symlinks)"
fi
echo ""
echo "These are only symlinks (shortcuts), not the actual SVS files."
echo "Deleting them will NOT delete your original data."
echo ""
read -p "Delete these symlink directories? [y/N] " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf "$SPLIT_BASE"
    echo ""
    echo "✓ Deleted $SPLIT_BASE"
    echo "✓ Your original SVS files in /data/data/SlideBench/wsi_stage1_seed42_n1000 are untouched"
else
    echo ""
    echo "Kept symlink directories."
fi

echo ""
echo "================================================================================"
