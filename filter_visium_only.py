#!/usr/bin/env python3
import json
import pandas as pd
from collections import defaultdict

# Load CSV
print("Loading CSV file...")
csv_path = "/data/data/ST/HEST1K/HEST_v1_1_0.csv"
df = pd.read_csv(csv_path)

# Create mapping of id -> st_technology
id_to_tech = dict(zip(df['id'], df['st_technology']))

# Load JSON
print("Loading JSON file...")
json_path = "/data/Projects/preprocess/hest_human_visium_subtypes.json"
with open(json_path, 'r') as f:
    data = json.load(f)

# Track excluded samples
excluded_by_tech = defaultdict(list)
total_samples = 0
kept_samples = 0

# Filter the data
filtered_data = {}
for organ_key, subtypes in data.items():
    filtered_subtypes = {}

    for subtype, sample_ids in subtypes.items():
        filtered_ids = []

        for sample_id in sample_ids:
            total_samples += 1
            tech = id_to_tech.get(sample_id, "UNKNOWN")

            if tech == "Visium":
                filtered_ids.append(sample_id)
                kept_samples += 1
            else:
                excluded_by_tech[tech].append(sample_id)

        if filtered_ids:  # Only keep subtypes that have remaining samples
            filtered_subtypes[subtype] = filtered_ids

    if filtered_subtypes:  # Only keep organs that have remaining subtypes
        filtered_data[organ_key] = filtered_subtypes

# Print summary
print("\n" + "="*60)
print("EXCLUSION SUMMARY")
print("="*60)
print(f"Total samples in original JSON: {total_samples}")
print(f"Samples kept (Visium): {kept_samples}")
print(f"Samples excluded: {total_samples - kept_samples}")
print()

for tech, samples in sorted(excluded_by_tech.items()):
    print(f"\n{tech}: {len(samples)} samples excluded")
    print(f"Sample IDs: {', '.join(sorted(samples)[:10])}", end="")
    if len(samples) > 10:
        print(f" ... and {len(samples) - 10} more")
    else:
        print()

# Save filtered JSON
output_path = "/data/Projects/preprocess/hest_human_visium_subtypes_cleaned.json"
with open(output_path, 'w') as f:
    json.dump(filtered_data, f, indent=4)

print(f"\n{'='*60}")
print(f"Cleaned JSON saved to: {output_path}")
print(f"Organs in cleaned data: {len(filtered_data)}")
print(f"{'='*60}")
