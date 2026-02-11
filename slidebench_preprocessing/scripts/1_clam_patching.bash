#!/bin/bash
# Step 1: CLAM Patching for SlideBench (20x and 40x separately)
# This script creates patches from SVS files

set -e  # Exit on error

echo "================================================================================"
echo "SlideBench CLAM Patching - Magnification-Separated Processing"
echo "================================================================================"
echo ""

# Configuration
CLAM_DIR="/data/Projects/CLAM_pre"
PREPROCESS_DIR="/data/Projects/preprocess"
SOURCE_DIR="/data/data/SlideBench/wsi_stage1_seed42_n1000"
SPLIT_BASE="/data/data/SlideBench_split"
EXTRACT_BASE="/data/data/Extracted_IMAGES"

# Directories for split files (symlinks)
DIR_20X="$SPLIT_BASE/SlideBench_20x_files"
DIR_40X="$SPLIT_BASE/SlideBench_40x_files"

# Output directories for patches
PATCHES_20X="$EXTRACT_BASE/SlideBench_20x"
PATCHES_40X="$EXTRACT_BASE/SlideBench_40x"

echo "Step 1: Create symlink directories for 20x and 40x files"
echo "--------------------------------------------------------------------------------"

# Check if symlink directories already exist
if [ -d "$DIR_20X" ] && [ "$(ls -A $DIR_20X 2>/dev/null | wc -l)" -gt 0 ] && \
   [ -d "$DIR_40X" ] && [ "$(ls -A $DIR_40X 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "✓ Symlink directories already exist, skipping creation"
    echo "  20x: $DIR_20X ($(ls $DIR_20X | wc -l) files)"
    echo "  40x: $DIR_40X ($(ls $DIR_40X | wc -l) files)"
else
    echo "Creating symlink directories..."
    cd "$PREPROCESS_DIR/slidebench_preprocessing/scripts"
    python create_symlink_dirs.py
fi
echo ""

echo "Step 2: Run CLAM patching for 20x files (113 slides)"
echo "--------------------------------------------------------------------------------"
echo "Source: $DIR_20X"
echo "Output: $PATCHES_20X"
echo ""

cd "$CLAM_DIR"
python create_patches_modified.py \
    --source "$DIR_20X" \
    --save_dir "$PATCHES_20X" \
    --patch_size 224 \
    --step_size 224 \
    --patch_level 0 \
    --seg \
    --patch

echo ""
echo "✓ 20x patching completed"
echo ""

echo "Step 3: Run CLAM patching for 40x+ files (879 slides)"
echo "--------------------------------------------------------------------------------"
echo "Source: $DIR_40X"
echo "Output: $PATCHES_40X"
echo ""

python create_patches_modified.py \
    --source "$DIR_40X" \
    --save_dir "$PATCHES_40X" \
    --patch_size 224 \
    --step_size 224 \
    --patch_level 0 \
    --custom_downsample 2 \
    --seg \
    --patch

echo ""
echo "✓ 40x patching completed"
echo ""

echo "================================================================================"
echo "CLAM PATCHING COMPLETED SUCCESSFULLY"
echo "================================================================================"
echo ""
echo "Output directories:"
echo "  20x patches: $PATCHES_20X/patches/"
echo "  40x patches: $PATCHES_40X/patches/"
echo ""
echo "Summary:"
echo "  20x slides processed: 113"
echo "  40x slides processed: 879"
echo "  Total: 992 (8 unknown files skipped)"
echo ""
echo "Next step:"
echo "  Run feature extraction: bash 2_extract_features.bash"
echo ""
echo "================================================================================"
echo ""
echo "Cleanup: The symlink directories are no longer needed."
echo "  $DIR_20X"
echo "  $DIR_40X"
echo ""
read -p "Delete symlink directories now? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf "$SPLIT_BASE"
    echo "✓ Deleted $SPLIT_BASE"
else
    echo "Kept symlink directories. To delete later, run:"
    echo "  rm -rf $SPLIT_BASE"
fi
echo ""
echo "================================================================================"
