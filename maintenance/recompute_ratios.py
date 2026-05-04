
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# python maintenance/recompute_ratios.py --input /path/to/features --xml-folder /path/to/annotations

import argparse
import os
import pickle
import random
import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch
import xml.etree.ElementTree as ET
import yaml
from shapely.affinity import scale
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union
from shapely.prepared import prep
from rtree import index
from lxml import etree


def xml_position(xml_path, ds_factor=2):
    """Parse XML annotations and return positive / negative unions."""
    root = ET.parse(xml_path).getroot()
    rtree_index = index.Index()
    polygons = []

    for annotation in root.findall('.//Annotation'):
        coords = []
        coordinate_elements = annotation.findall('.//Coordinate')
        if not coordinate_elements:
            coordinate_elements = annotation.findall('.//Vertex')

        for coordinate in coordinate_elements:
            x_val = float(coordinate.get('X')) / ds_factor
            y_val = float(coordinate.get('Y')) / ds_factor
            coords.append((x_val, y_val))

        if len(coords) < 3:
            continue

        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(1e-6)

        polygons.append(poly)
        rtree_index.insert(len(polygons) - 1, poly.bounds)

    positive_polygons = []
    negative_polygons = []

    for pos, poly in enumerate(polygons):
        possible_overlaps = list(rtree_index.intersection(poly.bounds))
        is_negative = any(polygons[other_pos].contains(poly) and other_pos != pos for other_pos in possible_overlaps)
        if is_negative:
            negative_polygons.append(poly)
        else:
            positive_polygons.append(poly)

    positive_union = unary_union(positive_polygons) if positive_polygons else MultiPolygon()
    negative_union = unary_union(negative_polygons) if negative_polygons else MultiPolygon()
    return positive_union, negative_union


def safe_xml_position(xml_path, ds_factor=2):
    try:
        return xml_position(xml_path, ds_factor)
    except ET.ParseError as exc:
        print(f"XML parse error for {xml_path}: {exc}. Attempting recovery.")

        with open(xml_path, 'r', encoding='utf-8') as xml_file:
            content = xml_file.read()

        fixed_content = f'<root>\n{content}\n</root>'
        parser = etree.XMLParser(recover=True)
        root = etree.fromstring(fixed_content.encode('utf-8'), parser)

        with tempfile.NamedTemporaryFile(suffix='.xml', delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(etree.tostring(root, encoding='utf-8'))

        try:
            return xml_position(temp_path, ds_factor)
        finally:
            os.unlink(temp_path)


def infer_ds_factor_from_coords(coords, annotations, *, patch_size=224, candidate_factors=None, sample_limit=2000):
    if candidate_factors is None:
        candidate_factors = (1, 2, 4, 8, 16)

    if annotations is None or annotations.is_empty or len(coords) == 0:
        return 1

    coords_array = np.asarray(coords)
    if sample_limit is not None and len(coords_array) > sample_limit:
        indices = np.random.choice(len(coords_array), sample_limit, replace=False)
        coords_array = coords_array[indices]

    patch_boxes = [box(float(x), float(y), float(x) + patch_size, float(y) + patch_size) for x, y in coords_array]

    best_factor = 1
    best_hits = -1

    for factor in candidate_factors:
        if factor <= 0:
            continue

        scaled_annotations = scale(annotations, xfact=1.0 / factor, yfact=1.0 / factor, origin=(0, 0))
        if scaled_annotations.is_empty:
            hits = 0
        else:
            prepared = prep(scaled_annotations)
            hits = sum(1 for patch_box in patch_boxes if prepared.intersects(patch_box))

        if hits > best_hits:
            best_hits = hits
            best_factor = factor

    return best_factor if best_factor > 0 else 1


def compute_patch_ratios(coords, annotations, patch_size=224):
    patch_area = float(patch_size * patch_size)
    if annotations is None or annotations.is_empty or patch_area == 0:
        return np.zeros(len(coords), dtype=np.float32)

    prepared = prep(annotations)
    ratios = []
    for x_val, y_val in coords:
        patch_box = box(float(x_val), float(y_val), float(x_val) + patch_size, float(y_val) + patch_size)
        if not prepared.intersects(patch_box):
            ratios.append(0.0)
            continue
        overlap_area = patch_box.intersection(annotations).area
        ratios.append(overlap_area / patch_area)

    return np.asarray(ratios, dtype=np.float32)


def update_pkl(
    input_path,
    xml_folder,
    *,
    output_path=None,
    patch_size=224,
    candidate_factors=None,
    sample_limit=2000,
    missing_xml='ignore',
):
    target_path = Path(output_path) if output_path else Path(input_path)

    with open(input_path, 'rb') as file:
        objects = pickle.load(file)

    xml_folder_path = Path(xml_folder)
    ratios_summary = []

    for obj in objects:
        slide_index = getattr(obj, 'slide_index', None)
        if slide_index is None:
            continue

        coords_tensor = getattr(obj, 'pos', None)
        if coords_tensor is None:
            continue

        coords = coords_tensor.detach().cpu().numpy()

        xml_path = xml_folder_path / f"{slide_index}.xml"
        if not xml_path.exists():
            if missing_xml == 'zero':
                ratios = np.zeros(len(coords), dtype=np.float32)
                ds_factor = 1
                print(f"{slide_index}: annotation missing, ratios set to zero.")
                ratios_summary.append(0.0)
                if getattr(obj, 'y', None) is not None:
                    dtype = obj.y.dtype
                    device = obj.y.device
                else:
                    dtype = torch.float32
                    device = torch.device('cpu')
                obj.y = torch.from_numpy(ratios).to(dtype=dtype, device=device)
            else:
                print(f"{slide_index}: annotation missing, ratios left unchanged.")
            continue
        else:
            positive_union, negative_union = safe_xml_position(str(xml_path), ds_factor=1)
            annotations = positive_union.difference(negative_union)

            ds_factor = infer_ds_factor_from_coords(
                coords,
                annotations,
                patch_size=patch_size,
                candidate_factors=candidate_factors,
                sample_limit=sample_limit,
            )

            scaled_annotations = scale(annotations, xfact=1.0 / ds_factor, yfact=1.0 / ds_factor, origin=(0, 0))
            ratios = compute_patch_ratios(coords, scaled_annotations, patch_size=patch_size)
            print(f"{slide_index}: ds_factor={ds_factor}, mean_ratio={ratios.mean():.4f}")

        ratios_summary.append(ratios.mean() if len(ratios) else 0.0)

        if getattr(obj, 'y', None) is not None:
            dtype = obj.y.dtype
            device = obj.y.device
        else:
            dtype = torch.float32
            device = torch.device('cpu')
        obj.y = torch.from_numpy(ratios).to(dtype=dtype, device=device)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, 'wb') as file:
        pickle.dump(objects, file)

    if output_path and Path(output_path) != Path(input_path):
        print(f"Saved updated PKL to {output_path}")
    else:
        print(f"Updated PKL in place: {input_path}")

    return ratios_summary


def update_h5_directory(
    input_dir,
    xml_folder,
    *,
    output_dir=None,
    patch_size=224,
    candidate_factors=None,
    sample_limit=2000,
    missing_xml='ignore',
):
    input_dir_path = Path(input_dir)
    xml_folder_path = Path(xml_folder)
    if output_dir:
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
    else:
        output_dir_path = None

    h5_files = sorted(input_dir_path.glob('*.h5'))
    if not h5_files:
        print(f"No .h5 files found in {input_dir}")
        return []

    ratios_summary = []

    for h5_file in h5_files:
        slide_index = h5_file.stem
        if output_dir_path:
            target_file = output_dir_path / h5_file.name
            shutil.copy2(h5_file, target_file)
        else:
            target_file = h5_file

        with h5py.File(target_file, 'r+') as handle:
            coords = np.array(handle['cords'])

            xml_path = xml_folder_path / f"{slide_index}.xml"
            if not xml_path.exists():
                if missing_xml == 'zero':
                    ratios = np.zeros(len(coords), dtype=np.float32)
                    ds_factor = 1
                    print(f"{slide_index}: annotation missing, ratios set to zero.")
                    if 'ratios' in handle:
                        del handle['ratios']
                    handle.create_dataset('ratios', data=ratios, dtype=np.float32)
                    ratios_summary.append(0.0)
                else:
                    print(f"{slide_index}: annotation missing, ratios left unchanged.")
                continue
            else:
                positive_union, negative_union = safe_xml_position(str(xml_path), ds_factor=1)
                annotations = positive_union.difference(negative_union)

                ds_factor = infer_ds_factor_from_coords(
                    coords,
                    annotations,
                    patch_size=patch_size,
                    candidate_factors=candidate_factors,
                    sample_limit=sample_limit,
                )

                scaled_annotations = scale(annotations, xfact=1.0 / ds_factor, yfact=1.0 / ds_factor, origin=(0, 0))
                ratios = compute_patch_ratios(coords, scaled_annotations, patch_size=patch_size)
                print(f"{slide_index}: ds_factor={ds_factor}, mean_ratio={ratios.mean():.4f}")

            if 'ratios' in handle:
                del handle['ratios']
            handle.create_dataset('ratios', data=ratios, dtype=np.float32)

            ratios_summary.append(ratios.mean() if len(ratios) else 0.0)

    if output_dir_path:
        print(f"Updated H5 files written to {output_dir_path}")
    else:
        print(f"Updated H5 files in place under {input_dir}")

    return ratios_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Recompute patch overlap ratios using XML annotations.")
    parser.add_argument('--input', required=True, help="Path to .pkl file or directory containing .h5 files.")
    parser.add_argument('--xml-folder', help="Directory holding annotation XML files (overrides config).")
    parser.add_argument('--config', help="Optional YAML config to pull defaults such as xml_folder or ds_factor_candidates.")
    parser.add_argument('--output', help="Optional output path (.pkl) or directory for updated H5 files.")
    parser.add_argument('--patch-size', type=int, default=224, help="Patch size used during extraction (default: 224).")
    parser.add_argument('--candidate-factors', nargs='*', type=int, default=[1, 2, 4, 8, 16], help="Candidate ds factors to evaluate.")
    parser.add_argument('--sample-limit', type=int, default=2000, help="Maximum patches sampled per slide for inference (default: 2000).")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for subsampling patches.")
    parser.add_argument(
        '--missing-xml-action',
        choices=['ignore', 'zero'],
        default='ignore',
        help="Behaviour when an annotation XML is missing: leave ratios untouched ('ignore') or set to zero ('zero').",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    config_data = {}
    if args.config:
        with open(args.config, 'r') as cfg_file:
            config_data = yaml.safe_load(cfg_file) or {}

    xml_folder = args.xml_folder or config_data.get('xml_folder')
    if not xml_folder:
        raise ValueError("XML folder must be provided via --xml-folder or config['xml_folder']")

    patch_size = args.patch_size
    if 'patch_size' in config_data and args.patch_size == 224:
        try:
            patch_size = int(config_data['patch_size'])
        except (TypeError, ValueError):
            pass

    candidate_factors = args.candidate_factors
    if candidate_factors == [1, 2, 4, 8, 16] and config_data.get('ds_factor_candidates') is not None:
        candidate_factors = config_data['ds_factor_candidates']

    if not isinstance(candidate_factors, (list, tuple)):
        candidate_factors = [candidate_factors]
    parsed_candidate_factors = []
    for factor in candidate_factors:
        try:
            value = int(factor)
        except (TypeError, ValueError):
            continue
        if value > 0:
            parsed_candidate_factors.append(value)
    candidate_factors = parsed_candidate_factors or [1]

    input_path = Path(args.input)

    if input_path.suffix == '.pkl':
        update_pkl(
            input_path,
            xml_folder,
            output_path=args.output,
            patch_size=patch_size,
            candidate_factors=candidate_factors,
            sample_limit=args.sample_limit,
            missing_xml=args.missing_xml_action,
        )
    elif input_path.is_dir():
        update_h5_directory(
            input_path,
            xml_folder,
            output_dir=args.output,
            patch_size=patch_size,
            candidate_factors=candidate_factors,
            sample_limit=args.sample_limit,
            missing_xml=args.missing_xml_action,
        )
    else:
        raise ValueError("--input must be a .pkl file or a directory containing .h5 files")


if __name__ == '__main__':
    main()
