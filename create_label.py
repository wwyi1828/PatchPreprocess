import numpy as np
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.ops import unary_union
import xml.etree.ElementTree as ET
import h5py
import os
from pathlib import Path
from tqdm import tqdm

xml_folder = '/media/weiyi/My Passport/Data/CAMELYON16/training/lesion_annotations'
patch_folder = '/media/weiyi/My Passport/Data/Extracted_IMAGES/Camelyon16/patches'
ds_factor = 2
patch_size = 224

xml_folder = Path(xml_folder)
patch_folder = Path(patch_folder)


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

def h5_position(h5_path):
    h5_obj = patch_folder.joinpath(h5_path)
    with h5py.File(h5_obj, 'r') as f:
        # data = list(f['imgs'])
        cord = list(f['coords'])
    return cord


target_slide = 'tumor_003'

xml_item = xml_folder.joinpath(f'{target_slide}.xml')
h5_item = patch_folder.joinpath(f'{target_slide}.h5')

annot_coordinates = xml_position(xml_item)
patch_coordinates = h5_position(h5_item)

# Eliminates overlapping
annot_coordinates = unary_union(annot_coordinates)  # it may return Polygon or MultiPolygon
annot_coordinates = MultiPolygon([annot_coordinates]) if annot_coordinates.geom_type == 'Polygon' else annot_coordinates

overlap_ratios = []
for pcoord in patch_coordinates:
    # 5 points for coordinates
    patch_box = box(pcoord[0], pcoord[1], pcoord[0] + patch_size, pcoord[1] + patch_size)

    intersection = patch_box.intersection(annot_coordinates)
    ratio = intersection.area / patch_box.area
    overlap_ratios.append(ratio)
print(np.array(overlap_ratios).mean(), np.array(overlap_ratios).max())


"""
Done, just think about how to save the patch-label here.
in the original h5 file or a new csv file with extracted feature csv?
"""

target_slides = [f'tumor_{str(i).zfill(3)}' for i in range(1, 112)]
# target_slide = 'tumor_055'
for target_slide in target_slides:
    xml_item = xml_folder.joinpath(f'{target_slide}.xml')
    h5_item = patch_folder.joinpath(f'{target_slide}.h5')

    annot_coordinates = xml_position(xml_item)
    patch_coordinates = h5_position(h5_item)

    # Eliminates overlapping
    annot_coordinates = unary_union(annot_coordinates)  # it may return Polygon or MultiPolygon
    annot_coordinates = MultiPolygon([annot_coordinates]) if annot_coordinates.geom_type == 'Polygon' else annot_coordinates

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from shapely.geometry import box

    # Initialize bounding box
    bbox = box(0, 0, 1, 1)

    # Update bounding box with MultiPolygon
    if annot_coordinates:
        annot_bounds = annot_coordinates.bounds
        bbox = bbox.union(box(annot_bounds[0], annot_bounds[1], annot_bounds[2], annot_bounds[3]))


    # Update bounding box with all boxes
    for pcoord in patch_coordinates:
        
        box_coords = [pcoord[0], pcoord[1], pcoord[0] + patch_size, pcoord[1] + patch_size]
        bbox = bbox.union(box(*box_coords))

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))
    minx, miny, maxx, maxy = bbox.bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(maxy, miny)  # Inverted Y-axis to treat (0,0) as top-left
    ax.set_aspect('equal', adjustable='box') 

    # Draw boxes
    for pcoord in patch_coordinates:
        rect = patches.Rectangle((pcoord[0], pcoord[1]), patch_size, patch_size, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    # Draw MultiPolygon
    if annot_coordinates:
        for poly in annot_coordinates.geoms:
            x, y = poly.exterior.xy
            # x = [i / 2 for i in x]
            # y = [i / 2 for i in y]
            ax.fill(x, y, alpha=0.5, fc='b', label="MultiPolygon")

    plt.savefig(f'check_preprocessing/{target_slide}.png')
