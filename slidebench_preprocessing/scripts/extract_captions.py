#!/usr/bin/env python3
"""
Extract captions from SlideBench JSON and create CSV files split by magnification
Maps JSON slide IDs to actual SVS filenames
"""

import json
import pandas as pd
from pathlib import Path

# Paths
caption_json = '/data/data/SlideBench/data/SlideInstruct_train_stage1_caption.stratified_seed42_n1000.json'
mpp_report = '../reports/slidebench_mpp_report.csv'
output_dir = Path('../captions')
output_dir.mkdir(exist_ok=True)

print("="*80)
print("Extracting Captions from SlideBench JSON")
print("="*80)

# Load caption JSON
print(f"\nLoading captions from: {caption_json}")
with open(caption_json, 'r') as f:
    caption_data = json.load(f)
print(f"✓ Loaded {len(caption_data)} caption entries")

# Load MPP report for filename mapping
print(f"\nLoading MPP report: {mpp_report}")
mpp_df = pd.read_csv(mpp_report)
print(f"✓ Loaded {len(mpp_df)} SVS file records")

# Create mapping from slide ID to SVS filename
print("\nBuilding slide ID → SVS filename mapping...")
id_to_svs = {}
svs_to_category = {}

for _, row in mpp_df.iterrows():
    svs_filename = row['filename']
    category = row['category']

    # Extract slide ID (remove UUID suffix)
    # Example: TCGA-A2-A1FX-01Z-00-DX1.UUID.svs → TCGA-A2-A1FX-01Z-00-DX1
    slide_id = svs_filename.split('.')[0]

    id_to_svs[slide_id] = svs_filename
    svs_to_category[svs_filename] = category

print(f"✓ Created mapping for {len(id_to_svs)} slide IDs")

# Extract captions and map to SVS files
print("\nExtracting captions and matching to SVS files...")
caption_records = []
unmatched = []

for entry in caption_data:
    # Extract slide ID from image path
    # Example: ./BRCA/TCGA-A2-A1FX-01Z-00-DX1.csv → TCGA-A2-A1FX-01Z-00-DX1
    image_path = entry['image'][0]
    slide_id = image_path.split('/')[-1].replace('.csv', '')

    # Extract caption (from GPT response)
    caption = entry['conversations'][1]['value']

    # Match to SVS filename
    if slide_id in id_to_svs:
        svs_filename = id_to_svs[slide_id]
        category = svs_to_category[svs_filename]

        caption_records.append({
            'filename': svs_filename,
            'slide_id': slide_id,
            'category': category,
            'caption': caption,
            'caption_length': len(caption)
        })
    else:
        unmatched.append(slide_id)

print(f"✓ Matched {len(caption_records)} captions to SVS files")
if unmatched:
    print(f"⚠ Warning: {len(unmatched)} captions could not be matched to SVS files")
    print(f"  First few unmatched: {unmatched[:5]}")

# Create DataFrame
df_all = pd.DataFrame(caption_records)

print("\n" + "="*80)
print("Caption Statistics")
print("="*80)
print(f"\nTotal captions: {len(df_all)}")
print(f"\nBy category:")
print(df_all['category'].value_counts())
print(f"\nCaption length statistics:")
print(df_all['caption_length'].describe())

# Split by magnification category
df_20x = df_all[df_all['category'] == '20x'].copy()
df_40x = df_all[df_all['category'] == '40x+'].copy()
df_unknown = df_all[df_all['category'] == 'unknown'].copy()

print(f"\n20x captions: {len(df_20x)}")
print(f"40x+ captions: {len(df_40x)}")
print(f"Unknown captions: {len(df_unknown)}")

# Save CSV files (simple format for easy loading during training)
print("\n" + "="*80)
print("Saving Caption Files")
print("="*80)

# 20x captions
output_20x = output_dir / 'SlideBench_20x_captions.csv'
df_20x[['filename', 'caption']].to_csv(output_20x, index=False)
print(f"✓ Saved 20x captions: {output_20x}")
print(f"  Entries: {len(df_20x)}")

# 40x captions
output_40x = output_dir / 'SlideBench_40x_captions.csv'
df_40x[['filename', 'caption']].to_csv(output_40x, index=False)
print(f"✓ Saved 40x captions: {output_40x}")
print(f"  Entries: {len(df_40x)}")

# All captions (for reference)
output_all = output_dir / 'SlideBench_all_captions.csv'
df_all[['filename', 'slide_id', 'category', 'caption']].to_csv(output_all, index=False)
print(f"✓ Saved all captions: {output_all}")
print(f"  Entries: {len(df_all)}")

# Also save detailed version with metadata
output_detailed = output_dir / 'SlideBench_captions_detailed.csv'
df_all.to_csv(output_detailed, index=False)
print(f"✓ Saved detailed captions: {output_detailed}")

print("\n" + "="*80)
print("Sample Captions")
print("="*80)

print("\n20x sample:")
if len(df_20x) > 0:
    sample = df_20x.iloc[0]
    print(f"  File: {sample['filename']}")
    print(f"  Caption preview: {sample['caption'][:200]}...")

print("\n40x sample:")
if len(df_40x) > 0:
    sample = df_40x.iloc[0]
    print(f"  File: {sample['filename']}")
    print(f"  Caption preview: {sample['caption'][:200]}...")

print("\n" + "="*80)
print("✓ Caption extraction complete!")
print("="*80)
print("\nGenerated files:")
print(f"  {output_20x}")
print(f"  {output_40x}")
print(f"  {output_all}")
print(f"  {output_detailed}")
print("\nYou can now use these CSV files directly in your training code.")
