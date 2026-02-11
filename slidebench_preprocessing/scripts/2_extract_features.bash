#!/bin/bash
# Step 2: CONCH Feature Extraction for SlideBench (20x and 40x separately)
# This script extracts features from patches created by CLAM

set -e  # Exit on error

echo "================================================================================"
echo "SlideBench CONCH Feature Extraction - Parallel Processing"
echo "================================================================================"
echo ""

# Configuration
PREPROCESS_DIR="/data/Projects/preprocess"
PATCHES_20X="/data/data/Extracted_IMAGES/SlideBench_20x/patches"
PATCHES_40X="/data/data/Extracted_IMAGES/SlideBench_40x/patches"

# GPU selection (modify as needed)
GPU_20X=0
GPU_40X=1

# Check if patches exist
echo "Checking prerequisites..."
if [ ! -d "$PATCHES_20X" ]; then
    echo "ERROR: 20x patches not found at $PATCHES_20X"
    echo "Please run 1_clam_patching.bash first!"
    exit 1
fi

if [ ! -d "$PATCHES_40X" ]; then
    echo "ERROR: 40x patches not found at $PATCHES_40X"
    echo "Please run 1_clam_patching.bash first!"
    exit 1
fi

echo "✓ Patches found"
echo ""

echo "Step 1: Extract CONCH features for 20x data"
echo "--------------------------------------------------------------------------------"
echo "Patches: $PATCHES_20X"
echo "GPU: $GPU_20X"
echo ""

cd "$PREPROCESS_DIR"

CUDA_VISIBLE_DEVICES=$GPU_20X python 1_extract_pretrain_feats.py \
    --config='slidebench_preprocessing/configs/SlideBench_20x.yaml' \
    --batch_size 256 \
    --model_type CONCH &

PID_20X=$!
echo "Started 20x feature extraction (PID: $PID_20X) on GPU $GPU_20X"
echo ""

echo "Step 2: Extract CONCH features for 40x+ data (running in parallel)"
echo "--------------------------------------------------------------------------------"
echo "Patches: $PATCHES_40X"
echo "GPU: $GPU_40X"
echo ""

CUDA_VISIBLE_DEVICES=$GPU_40X python 1_extract_pretrain_feats.py \
    --config='slidebench_preprocessing/configs/SlideBench_40x.yaml' \
    --batch_size 256 \
    --model_type CONCH &

PID_40X=$!
echo "Started 40x+ feature extraction (PID: $PID_40X) on GPU $GPU_40X"
echo ""

echo "Waiting for both feature extraction processes to complete..."
echo "(This may take 8-15 hours)"
echo ""

wait $PID_20X
echo "✓ 20x feature extraction completed"

wait $PID_40X
echo "✓ 40x+ feature extraction completed"

echo ""
echo "================================================================================"
echo "FEATURE EXTRACTION COMPLETED SUCCESSFULLY"
echo "================================================================================"
echo ""
echo "Output directories:"
echo "  20x features: /data/data/SlideBench_20x_CONCH/"
echo "  40x features: /data/data/SlideBench_40x_CONCH/"
echo ""
echo "Summary:"
echo "  20x slides: 113 .h5 files"
echo "  40x slides: 879 .h5 files"
echo ""
echo "You can now use these features for training!"
echo ""
echo "================================================================================"
