"""
Extract MacenkoUNI features from raw image patches stored in imge_RAW/ h5 files.

Each input h5 file contains:
  mor_feats: (N, 224, 224, 3) uint8  -- raw image patches

Each output h5 file contains:
  mor_feats: (N, 1024) float32       -- Macenko-normalized UNI features

Usage:
    python pipeline/extract_macenkouni_from_raw.py \
        --input_dir /path/to/imge_RAW \
        --batch_size 64 --gpu 0
"""

import argparse
import glob
import os
import sys

import h5py
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.image_encoder import extract_h5_features_in_batches, load_image_encoder
from core.gene_utils import save_mor_h5


def main():
    parser = argparse.ArgumentParser(description="Extract MacenkoUNI features from imge_RAW h5 files")
    parser.add_argument("--input_dir", required=True, help="Directory containing imge_RAW .h5 files")
    parser.add_argument("--output_dir", default=None, help="Output directory (default: sibling imge_MACENKOUNI/)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--uni_ckpt_dir", default=os.environ.get("UNI_CKPT_DIR"),
                        help="Directory containing UNI pytorch_model.bin. Can also be set with UNI_CKPT_DIR.")
    args = parser.parse_args()

    input_dir = args.input_dir.rstrip("/")
    if args.output_dir is None:
        output_dir = os.path.join(os.path.dirname(input_dir), "imge_MACENKOUNI")
    else:
        output_dir = args.output_dir

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")

    model_type, model, image_processor = load_image_encoder(
        "MACENKOUNI",
        device=device,
        uni_ckpt_dir=args.uni_ckpt_dir,
    )
    model.to(device).eval()

    h5_files = sorted(glob.glob(os.path.join(input_dir, "*.h5")))
    if not h5_files:
        print(f"No .h5 files found in {input_dir}")
        return

    print(f"Found {len(h5_files)} files\n")
    for h5_path in h5_files:
        name = os.path.basename(h5_path)
        out_path = os.path.join(output_dir, name)
        if os.path.exists(out_path):
            print(f"[skip] {name}")
            continue
        with h5py.File(h5_path, "r") as f:
            images = f["mor_feats"][:]  # (N, 224, 224, 3) uint8
        feats = extract_h5_features_in_batches(
            images,
            model_type=model_type,
            model=model,
            image_processor=image_processor,
            batch_size=args.batch_size,
            device=device,
        )
        save_mor_h5(out_path, feats.cpu().numpy())
        print(f"[done] {name}: {images.shape[0]} patches -> {feats.shape}")


if __name__ == "__main__":
    main()
