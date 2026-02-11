#!/bin/bash
# Parallel CLAM Patching for SlideBench (32 cores optimized)
# Runs 9 processes in parallel: 1x 20x + 8x 40x batches

set -e  # Exit on error

echo "================================================================================"
echo "SlideBench CLAM Patching - PARALLEL MODE (9 processes)"
echo "================================================================================"
echo ""
echo "Hardware: 32 CPU cores detected"
echo "Strategy: 1x 20x (113 files) + 8x 40x batches (~110 files each)"
echo "Expected speedup: 3-4x faster than sequential"
echo ""

# Configuration
CLAM_DIR="/data/Projects/CLAM_pre"
PREPROCESS_DIR="/data/Projects/preprocess"
SOURCE_DIR="/data/data/SlideBench/wsi_stage1_seed42_n1000"
SPLIT_BASE="/data/data/SlideBench_split"
EXTRACT_BASE="/data/data/Extracted_IMAGES"

# Output directories
PATCHES_20X="$EXTRACT_BASE/SlideBench_20x"
PATCHES_40X="$EXTRACT_BASE/SlideBench_40x"

echo "Step 1: Create symlink directories"
echo "--------------------------------------------------------------------------------"

# Create base directories
mkdir -p "$SPLIT_BASE"

# 20x symlinks
DIR_20X="$SPLIT_BASE/SlideBench_20x_files"
if [ ! -d "$DIR_20X" ] || [ "$(ls -A $DIR_20X 2>/dev/null | wc -l)" -eq 0 ]; then
    echo "Creating 20x symlinks..."
    mkdir -p "$DIR_20X"
    cd "$PREPROCESS_DIR/slidebench_preprocessing/file_lists"
    while IFS= read -r filename; do
        [ -z "$filename" ] && continue
        ln -sf "$SOURCE_DIR/$filename" "$DIR_20X/$filename"
    done < 20x_files.txt
    echo "✓ Created $(ls $DIR_20X | wc -l) 20x symlinks"
else
    echo "✓ 20x symlinks already exist ($(ls $DIR_20X | wc -l) files)"
fi

# 40x batch symlinks
echo ""
echo "Creating 40x batch symlinks..."
for i in {1..8}; do
    DIR_40X_BATCH="$SPLIT_BASE/SlideBench_40x_batch${i}"
    if [ ! -d "$DIR_40X_BATCH" ] || [ "$(ls -A $DIR_40X_BATCH 2>/dev/null | wc -l)" -eq 0 ]; then
        mkdir -p "$DIR_40X_BATCH"
        cd "$PREPROCESS_DIR/slidebench_preprocessing/file_lists"
        while IFS= read -r filename; do
            [ -z "$filename" ] && continue
            ln -sf "$SOURCE_DIR/$filename" "$DIR_40X_BATCH/$filename"
        done < "40x_batch${i}.txt"
        echo "  ✓ Batch $i: $(ls $DIR_40X_BATCH | wc -l) files"
    else
        echo "  ✓ Batch $i: already exists ($(ls $DIR_40X_BATCH | wc -l) files)"
    fi
done

echo ""
echo "Step 2: Launch parallel CLAM patching (9 processes)"
echo "--------------------------------------------------------------------------------"
echo "Starting at $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

cd "$CLAM_DIR"

# Launch 20x processing
echo "[1/9] Starting 20x processing (113 files)..."
python create_patches_modified.py \
    --source "$DIR_20X" \
    --save_dir "$PATCHES_20X" \
    --patch_size 224 \
    --step_size 224 \
    --patch_level 0 \
    --seg \
    --patch > "$EXTRACT_BASE/clam_20x.log" 2>&1 &
PID_20X=$!

# Launch 40x batch processing
for i in {1..8}; do
    DIR_40X_BATCH="$SPLIT_BASE/SlideBench_40x_batch${i}"
    echo "[$((i+1))/9] Starting 40x batch $i ($(ls $DIR_40X_BATCH | wc -l) files)..."

    python create_patches_modified.py \
        --source "$DIR_40X_BATCH" \
        --save_dir "$PATCHES_40X" \
        --patch_size 224 \
        --step_size 224 \
        --patch_level 0 \
        --custom_downsample 2 \
        --seg \
        --patch > "$EXTRACT_BASE/clam_40x_batch${i}.log" 2>&1 &

    eval "PID_40X_$i=$!"
done

echo ""
echo "All 9 processes launched! PIDs:"
echo "  20x: $PID_20X"
for i in {1..8}; do
    eval "echo \"  40x batch $i: \$PID_40X_$i\""
done

echo ""
echo "Monitoring progress (logs in $EXTRACT_BASE/)..."
echo "  tail -f $EXTRACT_BASE/clam_*.log"
echo ""
echo "Waiting for all processes to complete..."
echo "(This will take ~3-5 hours depending on CPU)"
echo ""

# Wait for all processes
wait $PID_20X
echo "✓ 20x completed"

for i in {1..8}; do
    eval "wait \$PID_40X_$i"
    echo "✓ 40x batch $i completed"
done

echo ""
echo "Completed at $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "================================================================================"
echo "PARALLEL CLAM PATCHING COMPLETED SUCCESSFULLY"
echo "================================================================================"
echo ""
echo "Output directories:"
echo "  20x patches: $PATCHES_20X/patches/ (113 slides)"
echo "  40x patches: $PATCHES_40X/patches/ (879 slides)"
echo ""
echo "Logs:"
for log in "$EXTRACT_BASE"/clam_*.log; do
    [ -f "$log" ] && echo "  $(basename $log)"
done
echo ""
echo "Next step:"
echo "  bash 2_extract_features.bash"
echo ""
echo "================================================================================"
echo ""
echo "Cleanup: The symlink directories are no longer needed."
read -p "Delete symlink directories now? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf "$SPLIT_BASE"
    echo "✓ Deleted $SPLIT_BASE"
else
    echo "Kept symlink directories. To delete later, run:"
    echo "  bash cleanup_symlinks.bash"
fi
echo ""
echo "================================================================================"
