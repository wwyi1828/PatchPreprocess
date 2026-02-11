#!/usr/bin/env python3
"""Split SlideBench files by actual magnification (20x vs 40x+)"""

import pandas as pd
from pathlib import Path
import shutil

# Read the MPP report
report_file = 'slidebench_mpp_report.csv'
df = pd.read_csv(report_file)

print(f"Total files: {len(df)}")
print(f"\nCategory distribution:")
print(df['category'].value_counts())

# Split by category
df_20x = df[df['category'] == '20x'].copy()
df_40x = df[df['category'] == '40x+'].copy()
df_unknown = df[df['category'] == 'unknown'].copy()

print(f"\n20x files: {len(df_20x)}")
print(f"40x+ files: {len(df_40x)}")
print(f"Unknown files: {len(df_unknown)} (will be skipped)")

# Create file lists (just filenames for CLAM processing)
output_dir = Path('file_lists')
output_dir.mkdir(exist_ok=True)

# Save 20x file list
with open(output_dir / '20x_files.txt', 'w') as f:
    for filename in df_20x['filename']:
        f.write(f"{filename}\n")
print(f"\n✓ Saved: {output_dir / '20x_files.txt'}")

# Save 40x file list
with open(output_dir / '40x_files.txt', 'w') as f:
    for filename in df_40x['filename']:
        f.write(f"{filename}\n")
print(f"✓ Saved: {output_dir / '40x_files.txt'}")

# Create label CSVs (assuming we need to generate labels)
# For SlideBench, we might need to get labels from another source
# For now, create placeholder CSVs with dummy labels
labels_dir = Path('labels')
labels_dir.mkdir(exist_ok=True)

# Create 20x label CSV (placeholder - you'll need to fill in actual labels)
df_20x_labels = pd.DataFrame({
    'filename': df_20x['filename'],
    'type': 'unknown'  # Replace with actual labels if available
})
df_20x_labels.to_csv(labels_dir / 'SlideBench_20x.csv', index=False)
print(f"✓ Saved: {labels_dir / 'SlideBench_20x.csv'} (needs label update)")

# Create 40x label CSV
df_40x_labels = pd.DataFrame({
    'filename': df_40x['filename'],
    'type': 'unknown'  # Replace with actual labels if available
})
df_40x_labels.to_csv(labels_dir / 'SlideBench_40x.csv', index=False)
print(f"✓ Saved: {labels_dir / 'SlideBench_40x.csv'} (needs label update)")

# Create combined label CSV (for reference)
df_all_labels = pd.concat([df_20x_labels, df_40x_labels], ignore_index=True)
df_all_labels.to_csv(labels_dir / 'SlideBench_all.csv', index=False)
print(f"✓ Saved: {labels_dir / 'SlideBench_all.csv'} (needs label update)")

# Print statistics
print("\n" + "="*80)
print("STATISTICS")
print("="*80)
print(f"\n20x slides:")
print(f"  Count: {len(df_20x)}")
print(f"  Average MPP: {df_20x['mpp_x'].mean():.4f}")
print(f"  Average equiv mag: {df_20x['equiv_mag'].mean():.2f}x")

print(f"\n40x+ slides:")
print(f"  Count: {len(df_40x)}")
print(f"  Average MPP: {df_40x['mpp_x'].mean():.4f}")
print(f"  Average equiv mag: {df_40x['equiv_mag'].mean():.2f}x")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("1. Update label CSV files with actual labels (if available)")
print("2. Run CLAM patching separately for 20x and 40x files")
print("3. Run CONCH feature extraction for each category")
print("\nSee the generated shell script: process_slidebench_by_mag.bash")
