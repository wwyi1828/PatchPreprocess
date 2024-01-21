import numpy as np
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.ops import unary_union
import xml.etree.ElementTree as ET
import os
from pathlib import Path
from tqdm import tqdm
from torch_geometric.data import Data as gData
import pickle
import torch


def xml_position(xml_path, ds_factor=2):
    root = ET.parse(xml_path).getroot()
    polygons_coords = []
    for annotation in root.findall('.//Annotation'):
        coords = []
        for coordinate in annotation.findall('.//Coordinate'):
            x = float(coordinate.get('X'))
            y = float(coordinate.get('Y'))
            coords.append((x/ds_factor, y/ds_factor))
        if len(coords) < 4:
            print(f"Skipping invalid coordinates: {coords}")
            continue  # skip to the next annotation
        coords = Polygon(coords) # Convert coordinates to polygon objs
        if not coords.is_valid: # Fix self-intersection
            coords = coords.buffer(1e-6)

        polygons_coords.append(coords)
    return polygons_coords

xml_folder = '/media/weiyi/My Passport/Data/CAMELYON16/training/lesion_annotations'
xml_folder = '/media/weiyi/My Passport/Data/CAMELYON16/testing/lesion_annotations'
ds_factor = 2
patch_size = 224
xml_folder = Path(xml_folder)

target_objs = pickle.load(open('/media/weiyi/My Passport/Data/C16_test.pkl','rb'))
print('Loaded')

for target_obj in target_objs:
    xml_item = xml_folder.joinpath(f'{target_obj.slide_index}.xml')
    if xml_item.exists():
        annot_coordinates = xml_position(xml_item)
        # Eliminates overlapping
        annot_coordinates = unary_union(annot_coordinates)  # it may return Polygon or MultiPolygon
        annot_coordinates = MultiPolygon([annot_coordinates]) if annot_coordinates.geom_type == 'Polygon' else annot_coordinates
    else:
        annot_coordinates = MultiPolygon()

    overlap_ratios = []
    for pcoord in target_obj.pos:
        # 5 points for coordinates
        patch_box = box(pcoord[0], pcoord[1], pcoord[0] + patch_size, pcoord[1] + patch_size)

        intersection = patch_box.intersection(annot_coordinates)
        ratio = intersection.area / patch_box.area
        overlap_ratios.append(ratio)
    target_obj.y = torch.stack([torch.tensor(_) for _ in overlap_ratios])
    if target_obj.graph_y==1:
        print(target_obj.slide_index, np.array(overlap_ratios).max(), (np.array(overlap_ratios)>0.08).sum())


pickle.dump(target_objs, open('/media/weiyi/My Passport/Data/C16_test.pkl','wb'))