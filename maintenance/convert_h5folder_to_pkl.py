#!/usr/bin/env python3
"""
Convert a folder of .h5 files (one slide per file) back into a single .pkl file.
"""

import argparse
import pickle
from pathlib import Path
from typing import Iterable, List

import h5py
import numpy as np
import torch
from types import SimpleNamespace
from tqdm import tqdm


def _read_dataset(handle, key: str):
    if key not in handle:
        return None
    return handle[key][()]


def _decode_caption(caption_data):
    if caption_data is None:
        return None

    if hasattr(caption_data, "shape") and caption_data.shape == ():
        caption_data = caption_data.item()

    if isinstance(caption_data, (bytes, bytearray)):
        return caption_data.decode("utf-8")

    if isinstance(caption_data, np.ndarray):
        if caption_data.shape == ():
            caption_data = caption_data.item()
        elif caption_data.size == 1:
            caption_data = caption_data.reshape(-1)[0]

    if isinstance(caption_data, np.bytes_):
        return caption_data.decode("utf-8")
    return caption_data


def _to_tensor(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value
    return torch.from_numpy(np.asarray(value))


def convert_h5_folder_to_pkl(h5_folder: str, output_pkl: str) -> int:
    """
    Convert a folder of .h5 files to a single pickle file.

    Args:
        h5_folder: Directory containing slide-level .h5 files
        output_pkl: Path to output .pkl file
    """
    h5_folder = Path(h5_folder)
    output_pkl = Path(output_pkl)
    if not h5_folder.exists():
        raise FileNotFoundError(f"Input folder not found: {h5_folder}")
    output_pkl.parent.mkdir(parents=True, exist_ok=True)

    h5_files: Iterable[Path] = sorted(h5_folder.glob("*.h5"))
    h5_files = list(h5_files)
    print(f"Found {len(h5_files)} H5 files in {h5_folder}", flush=True)

    target_objs: List[SimpleNamespace] = []
    for h5_file in tqdm(h5_files, desc="Converting H5 files"):
        with h5py.File(str(h5_file), "r") as f:
            feats = _read_dataset(f, "feats")
            cords = _read_dataset(f, "cords")
            if feats is None or cords is None:
                raise ValueError(f"Missing required dataset 'feats' or 'cords' in {h5_file}")

            ratios = _read_dataset(f, "ratios")
            label = _read_dataset(f, "label")
            caption = _decode_caption(_read_dataset(f, "caption"))

        obj = SimpleNamespace()
        obj.x = _to_tensor(feats).cpu()
        obj.pos = _to_tensor(cords).cpu()
        obj.y = _to_tensor(ratios).cpu() if ratios is not None else None
        obj.slide_index = h5_file.stem

        if caption is not None:
            obj.slide_caption = caption

        if label is not None:
            slide_label = _to_tensor(label).cpu()
            obj.slide_y = slide_label
            obj.graph_y = slide_label

        target_objs.append(obj)

    print(f"Conversion complete! Writing {len(target_objs)} objects to {output_pkl}", flush=True)
    with open(output_pkl, "wb") as f:
        pickle.dump(target_objs, f)
    print(f"Saved {len(target_objs)} objects to {output_pkl}", flush=True)
    return len(target_objs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a folder of .h5 files back to a .pkl file."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to directory containing .h5 files"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output .pkl file"
    )

    args = parser.parse_args()
    convert_h5_folder_to_pkl(args.input, args.output)
