#!/usr/bin/env python3
"""
Convert a single .pkl file (containing multiple slides) to a folder of .h5 files.
Each slide will be saved as a separate .h5 file.
"""

import argparse
import pickle
import h5py
from pathlib import Path
from tqdm import tqdm


def convert_pkl_to_h5_folder(pkl_path, output_folder):
    """
    Convert a pickle file containing slide data to individual H5 files.

    Args:
        pkl_path: Path to the input .pkl file
        output_folder: Path to the output directory where .h5 files will be saved
    """
    # Load the pickle file
    print(f"Loading pickle file: {pkl_path}", flush=True)
    import os
    file_size_mb = os.path.getsize(pkl_path) / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB", flush=True)

    with open(pkl_path, 'rb') as f:
        target_objs = pickle.load(f)

    print(f"Loaded {len(target_objs)} slides from pickle file", flush=True)

    # Create output directory
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_folder}")

    # Convert each slide to an H5 file
    for a_graph in tqdm(target_objs, desc="Converting slides"):
        slide_name = a_graph.slide_index
        h5_file_path = output_folder / f"{slide_name}.h5"

        with h5py.File(str(h5_file_path), 'w') as f:
            # Save features
            f.create_dataset('feats', data=a_graph.x.cpu())
            # Save coordinates
            f.create_dataset('cords', data=a_graph.pos.cpu())
            # Save ratios if they exist
            if hasattr(a_graph, 'y') and a_graph.y is not None:
                f.create_dataset('ratios', data=a_graph.y.cpu())
            # Save slide-level labels if they exist (support both old graph_y and new slide_y)
            slide_label = getattr(a_graph, 'slide_y', None) or getattr(a_graph, 'graph_y', None)
            if slide_label is not None:
                f.create_dataset('label', data=slide_label.cpu())

    print(f"Conversion complete! Saved {len(target_objs)} H5 files to {output_folder}")

    # Return the count for the bash script to capture
    return len(target_objs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a .pkl file to a folder of .h5 files"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input .pkl file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output directory for .h5 files"
    )

    args = parser.parse_args()

    convert_pkl_to_h5_folder(args.input, args.output)
