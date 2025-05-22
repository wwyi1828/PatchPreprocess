import h5py
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models, transforms
import numpy as np
import pandas as pd
import os
import pickle
import torch_geometric
from torch_geometric.data import Data as gData
from torch.utils.data import DataLoader, Dataset
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.ops import unary_union
import xml.etree.ElementTree as ET
from pathlib import Path
from transformers import AutoImageProcessor
from transformers.image_processing_utils import BatchFeature
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import yaml
import argparse

from utils import ModifiedResNet, GaussianBlur

from collections import OrderedDict

import tempfile
tempfile.tempdir = '/data/temp/'


parser = argparse.ArgumentParser(description='Read Configuration File')
parser.add_argument('--config', type=str, default='configs/TCGA_Adrenal.yaml', 
                    help='Path to the configuration file.')
parser.add_argument('--batch_size', type=int, default=2048, 
                    help='Batch size for preprocessing.')

args = parser.parse_args()
# args.config = 'configs/C16_test.yaml'

with open(args.config, 'r') as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

class H5Dataset(Dataset):
    def __init__(self, file_path):
        with h5py.File(file_path, 'r') as f:
            self.images = np.stack(f['imgs'])
            self.coords = np.array(f['coords'])

        self.images = np.transpose(self.images, (0, 3, 1, 2))
        self.images = torch.from_numpy(self.images).float()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.coords[idx]

class PNGDataset(Dataset):
    def __init__(self, file_path):
        self.png_paths = list(Path(file_path).rglob('*.png'))

    def __len__(self):
        return len(self.png_paths)
    
    def __getitem__(self, index):
        with Image.open(self.png_paths[index]).convert("RGB") as image:
            image = np.array(image)
            image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image).float()
            coord = np.array(self.png_paths[index].stem.split('_')).astype(int)

            return image, coord

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

# Create a TensorDataset and DataLoader
slide_labels = pd.read_csv(config['label_file'])
slide_labels = dict(zip(slide_labels['filename'], slide_labels['type']))
# slide_ext = list(slide_labels.keys())[0].split('.')[-1] # svs or tif
# label_convert_dict = {'Group_BT': 0, 'Group_AT': 1, 'Group_MT': 2}
# label_convert_dict = {'TCGA_LUAD': 0, 'TCGA_LUSC':1} 
if config['dataset_name'] == 'Camelyon16':
    label_convert_dict = {'Normal': 0, 'Tumor': 1} #Camelyon16
elif config['dataset_name'] == 'BRACS':
    label_convert_dict = {'Group_BT': 0, 'Group_AT': 1, 'Group_MT': 2}
elif config['dataset_name'] == 'TCGA_LUNG':
    label_convert_dict = {'TCGA_LUAD': 0, 'TCGA_LUSC':1} 
# slide_labels = {key.split('.')[0]: {'Normal': 0, 'Tumor': 1}[value] for key, value in slide_labels.items()}
# slide_labels = {key.split(f'.')[0]: {'TCGA_ACC': 0, 'TCGA_PCPG': 1}[value] for key, value in slide_labels.items()}
slide_labels = {key.split(f'.')[0]: label_convert_dict[value] for key, value in slide_labels.items()}

target_objs = []
# src_folder = '/pool1/data/Extracted_IMAGES/Camelyon16_test/patches/'
src_folder = config['src_folder']
output_folder = config['dst_folder']
os.makedirs(output_folder, exist_ok=True)

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


# CAMELYON16
# NORMALIZATION_MEAN = [0.6932, 0.5440, 0.7006]
# NORMALIZATION_STD = [0.2498, 0.2916, 0.2135]

# BRACS
# NORMALIZATION_MEAN = [0.7954, 0.6322, 0.7400]
# NORMALIZATION_STD = [0.1598, 0.2119, 0.1574]

# ImageNet
NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
NORMALIZATION_STD = [0.229, 0.224, 0.225]

normalize = transforms.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD)
image_processor = transforms.Compose([
    transforms.ToTensor(),
    normalize
])


for slide_h5 in tqdm(os.listdir(src_folder)):
    slide_name = slide_h5.split('.')[0]

    dataset = PNGDataset(f'{src_folder}{slide_h5}')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=12, pin_memory=True)

    all_images = []
    all_coords = []
    overlap_ratios = []
    wsi_label = None

    if config['dataset_name'] == 'Camelyon16':
        xml_folder = config['xml_folder']
        ds_factor = 2
        patch_size = 224
        xml_folder = Path(xml_folder)

        xml_item = xml_folder.joinpath(f'{slide_name}.xml')
        annot_coordinates = MultiPolygon()
        if xml_item.exists():
            polygons_coords = xml_position(xml_item)
            annot_coordinates = unary_union(polygons_coords)
            annot_coordinates = MultiPolygon([annot_coordinates]) if annot_coordinates.geom_type == 'Polygon' else annot_coordinates

        for images, coords in dataloader:
            all_images.append(images.numpy())
            all_coords.append(coords)
            for coord in coords:
                patch_box = box(coord[0], coord[1], coord[0] + patch_size, coord[1] + patch_size)
                intersection = patch_box.intersection(annot_coordinates)
                ratio = intersection.area / patch_box.area
                overlap_ratios.append(ratio)
    else:
        for images, coords in dataloader:
            all_images.append(images.numpy())
            all_coords.append(coords)

    all_images = np.concatenate(all_images, axis=0)
    all_coords = np.concatenate(all_coords, axis=0)
    overlap_ratios = np.array(overlap_ratios)

    try:
        wsi_label = F.one_hot(torch.tensor(slide_labels[slide_name]), num_classes=len(label_convert_dict)).float().unsqueeze(0)
    except KeyError:
        print(f"Slide name {slide_name} not found, skipping...")
        with open('error_log.txt', 'a') as f:
            f.write(f"Slide name {slide_name} not found\n")


    file_path = os.path.join(output_folder, f"{slide_name}.h5")
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('images', data=all_images)
        f.create_dataset('coords', data=all_coords)
        f.create_dataset('overlap_ratios', data=overlap_ratios)
        f.create_dataset('wsi_label', data=wsi_label)

    
    print(f"Data for {slide_name} saved in {file_path}")






# import h5py

# # 指定你的HDF5文件路径
# file_path = '/data/data/C16_raw/train/tumor_038.h5'

# # 打开HDF5文件
# with h5py.File(file_path, 'r') as file:
#     print("Datasets in the file:")
#     print(list(file.keys()))  # 打印所有数据集的名称

#     # 可以选择查看每个数据集的具体信息
#     for dataset_name in file.keys():
#         dataset = file[dataset_name]
#         print(f"\nDataset '{dataset_name}':")
#         print("Shape:", dataset.shape)
#         print("Type:", dataset.dtype)
