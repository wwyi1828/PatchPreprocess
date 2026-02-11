# SlideBench Processing Guide - Magnification-Separated Approach

## Overview

SlideBench dataset contains slides scanned at different magnifications:
- **113 slides (11.3%)**: True 20x (MPP ≈ 0.50)
- **879 slides (87.9%)**: Scanned at 40x+ (MPP ≈ 0.25)
- **8 slides (0.8%)**: Unknown (no MPP metadata)

To maintain resolution consistency, we process them separately.

## Files Created

### 1. Analysis Scripts
- `check_slidebench_properties.py` - Sample checker (first 10 slides)
- `check_all_slidebench.py` - Full dataset analyzer
- `slidebench_mpp_report.csv` - Complete MPP report for all 1000 slides

### 2. File Lists
- `file_lists/20x_files.txt` - List of 113 true 20x slides
- `file_lists/40x_files.txt` - List of 879 high-res slides

### 3. Label Files (need updating with actual labels)
- `labels/SlideBench_20x.csv` - Labels for 20x slides
- `labels/SlideBench_40x.csv` - Labels for 40x slides
- `labels/SlideBench_all.csv` - Combined labels

### 4. Configuration Files
- `configs/SlideBench_20x.yaml` - Config for 20x processing
- `configs/SlideBench_40x.yaml` - Config for 40x processing

### 5. Processing Scripts
- `split_by_magnification.py` - Generates file lists and label CSVs
- `create_symlink_dirs.py` - Creates symlink directories
- `process_slidebench_by_mag.bash` - Main processing pipeline

## Usage

### Quick Start (All-in-One)

```bash
cd /data/Projects/preprocess

# Make the script executable
chmod +x process_slidebench_by_mag.bash

# Run the complete pipeline
./process_slidebench_by_mag.bash
```

This will:
1. Create symlink directories separating 20x and 40x files
2. Run CLAM patching for 20x files (patch_level=0, ~20x)
3. Run CLAM patching for 40x files (patch_level=0, ~40x)
4. Extract CONCH features for both categories in parallel

### Step-by-Step (Manual)

#### Step 1: Analyze the dataset (already done)
```bash
python check_all_slidebench.py
```

#### Step 2: Generate file lists
```bash
python split_by_magnification.py
```

#### Step 3: Create symlink directories
```bash
python create_symlink_dirs.py
```

#### Step 4: Run CLAM patching for 20x files
```bash
cd /data/Projects/CLAM_pre

python create_patches_modified.py \
    --source /data/data/SlideBench_split/SlideBench_20x_files \
    --save_dir /data/data/Extracted_IMAGES/SlideBench_20x \
    --patch_size 224 \
    --step_size 224 \
    --patch_level 0 \
    --seg \
    --patch
```

#### Step 5: Run CLAM patching for 40x files
```bash
python create_patches_modified.py \
    --source /data/data/SlideBench_split/SlideBench_40x_files \
    --save_dir /data/data/Extracted_IMAGES/SlideBench_40x \
    --patch_size 224 \
    --step_size 224 \
    --patch_level 0 \
    --seg \
    --patch
```

#### Step 6: Extract CONCH features (can run in parallel)
```bash
cd /data/Projects/preprocess

# On GPU 0 - 20x files
CUDA_VISIBLE_DEVICES=0 python 1_extract_pretrain_feats.py \
    --config='configs/SlideBench_20x.yaml' \
    --batch_size 256 \
    --model_type CONCH &

# On GPU 1 - 40x files
CUDA_VISIBLE_DEVICES=1 python 1_extract_pretrain_feats.py \
    --config='configs/SlideBench_40x.yaml' \
    --batch_size 256 \
    --model_type CONCH &

wait
```

## Output Structure

```
/data/data/
├── SlideBench_split/
│   ├── SlideBench_20x_files/          # Symlinks to 113 true 20x slides
│   └── SlideBench_40x_files/          # Symlinks to 879 high-res slides
│
├── Extracted_IMAGES/
│   ├── SlideBench_20x/
│   │   ├── patches/                   # PNG patches from 20x slides
│   │   ├── masks/                     # Segmentation masks
│   │   └── process_list_autogen.csv
│   └── SlideBench_40x/
│       ├── patches/                   # PNG patches from 40x slides
│       ├── masks/
│       └── process_list_autogen.csv
│
├── SlideBench_20x_CONCH/              # CONCH features for 20x (113 .h5 files)
└── SlideBench_40x_CONCH/              # CONCH features for 40x (879 .h5 files)
```

## Important Notes

### Label Files
The generated label CSV files contain placeholder labels (`type: unknown`). You need to update them with actual labels if available:
- `labels/SlideBench_20x.csv`
- `labels/SlideBench_40x.csv`

If you don't have labels, the feature extraction will still work, but you'll need to handle the labels later.

### GPU Configuration
By default, the pipeline uses:
- GPU 0 for 20x processing
- GPU 1 for 40x processing

Modify `process_slidebench_by_mag.bash` variables `GPU_20X` and `GPU_40X` if needed.

### Disk Space Requirements
Estimated space needed:
- Patches (20x): ~50-100 GB
- Patches (40x): ~400-800 GB
- Features (20x): ~5-10 GB
- Features (40x): ~40-80 GB

### Processing Time Estimates
- CLAM patching (20x): 1-3 hours
- CLAM patching (40x): 10-20 hours
- CONCH features (20x): 1-2 hours
- CONCH features (40x): 8-15 hours

## Troubleshooting

### If patching fails for some slides
Check `process_list_autogen.csv` in the output directories to see which slides failed.

### If running out of memory during feature extraction
Reduce `--batch_size` parameter (try 128 or 64).

### If you want to process sequentially instead of parallel
In `process_slidebench_by_mag.bash`, remove the `&` and `wait` commands.

## Next Steps After Processing

Once features are extracted, you can:
1. Combine features from both categories if needed
2. Train models separately for each resolution
3. Analyze performance differences between 20x and 40x data
