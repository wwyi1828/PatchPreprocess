#!/usr/bin/env python3
"""Create tumor subtype label CSV from SlideInstruct caption JSON."""

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd


DEFAULT_INPUT_JSON = Path("examples/data/slideinstruct_caption.example.json")
DEFAULT_OUTPUT_CSV = Path("outputs/slidebench_labels.csv")


def extract_label_record(entry):
    """Extract (slide_id, subtype) from one JSON entry."""
    image_field = entry.get("image")
    if isinstance(image_field, list):
        if not image_field:
            return None
        image_path = str(image_field[0])
    elif isinstance(image_field, str):
        image_path = image_field
    else:
        return None

    # Example image path:
    # ./BRCA/TCGA-A2-A1FX-01Z-00-DX1.csv
    path_parts = [part for part in image_path.replace("\\", "/").split("/") if part and part != "."]
    if len(path_parts) < 2:
        return None

    subtype = path_parts[0].strip()
    slide_id = Path(path_parts[-1]).stem.strip()
    if not subtype or not slide_id:
        return None

    return slide_id, subtype


def create_tumor_subtype_labels(input_json: Path, output_csv: Path) -> None:
    with input_json.open("r", encoding="utf-8") as f:
        entries = json.load(f)

    if not isinstance(entries, list):
        raise ValueError(f"Expected a JSON list in {input_json}, got {type(entries).__name__}.")

    records = {}
    invalid_count = 0
    conflicting = []

    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            invalid_count += 1
            continue

        parsed = extract_label_record(entry)
        if parsed is None:
            invalid_count += 1
            continue

        slide_id, subtype = parsed
        old_subtype = records.get(slide_id)
        if old_subtype is not None and old_subtype != subtype:
            conflicting.append((idx, slide_id, old_subtype, subtype))
            continue

        records[slide_id] = subtype

    if conflicting:
        conflict_preview = "\n".join(
            f"  idx={idx}, slide={sid}, old={old}, new={new}"
            for idx, sid, old, new in conflicting[:10]
        )
        raise ValueError(
            "Found conflicting subtype labels for the same slide_id:\n"
            f"{conflict_preview}\n"
            f"Total conflicts: {len(conflicting)}"
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [{"filename": slide_id, "type": subtype} for slide_id, subtype in sorted(records.items())]
    )
    df.to_csv(output_csv, index=False)

    print(f"Input JSON: {input_json}")
    print(f"Output CSV: {output_csv}")
    print(f"Total JSON entries: {len(entries)}")
    print(f"Valid label rows: {len(df)}")
    print(f"Invalid entries skipped: {invalid_count}")
    print("Subtype counts:")
    for subtype, count in Counter(df["type"]).most_common():
        print(f"  {subtype}: {count}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create SlideBench tumor subtype label CSV from SlideInstruct caption JSON."
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=DEFAULT_INPUT_JSON,
        help=f"Path to SlideInstruct JSON (default: {DEFAULT_INPUT_JSON})",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Path to output CSV (default: {DEFAULT_OUTPUT_CSV})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    create_tumor_subtype_labels(args.input_json, args.output_csv)


if __name__ == "__main__":
    main()
