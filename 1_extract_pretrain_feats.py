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
    def __init__(self, file_path, augument=None):
        self.png_paths = list(Path(file_path).rglob('*.png'))
        self.augument = augument

    def __len__(self):
        return len(self.png_paths)
    
    def __getitem__(self, index):
        with Image.open(self.png_paths[index]) as image:
            if self.augument is None:
                image = np.array(image)
                image = np.transpose(image, (2, 0, 1))
                image = torch.from_numpy(image).float()
            else:
                image = self.augument(image)
            coord = np.array(self.png_paths[index].stem.split('_')).astype(int)

            return image, coord


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
# model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
# model.state_dict().keys()

if 'ckpt_path' in config:
    ckpt_path = config['ckpt_path']
    ckpt = torch.load(ckpt_path)['state_dict']
    print(f'{ckpt_path} loaded')

    updated_ckpt = {key.replace("module.encoder.", ""): value for key, value in ckpt.items()}
    model.load_state_dict(updated_ckpt, strict=False)
else:
    print("Load ImageNet weights")

if False:
    model = nn.Sequential(*list(model.children())[:-1])
else:
    model = ModifiedResNet(model)

model.eval()
model.cuda()
model.to(device)

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
# src_folder = '/media/weiyi/My Passport/Data/Extracted_IMAGES/Camelyon16_test/patches/'
src_folder = config['src_folder']


# Newly computed 
# CAMELYON_NORMALIZATION_MEAN = [0.6932, 0.5440, 0.7006]
# CAMELYON_NORMALIZATION_STD = [0.2498, 0.2916, 0.2135]

# BRACS
# CAMELYON_NORMALIZATION_MEAN = [0.7954, 0.6322, 0.7400]
# CAMELYON_NORMALIZATION_STD = [0.1598, 0.2119, 0.1574]

# ImageNet
CAMELYON_NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
CAMELYON_NORMALIZATION_STD = [0.229, 0.224, 0.225]

NORMALIZATION_MEAN = CAMELYON_NORMALIZATION_MEAN
NORMALIZATION_STD = CAMELYON_NORMALIZATION_STD
normalize = transforms.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD)
augmentation = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])


# processed_slides = {obj.slide_index for obj in target_objs}
for slide_h5 in tqdm(os.listdir(src_folder)):
    slide_name = slide_h5.split('.')[0]

    # if slide_name in processed_slides:
    #     continue
    if False:
        dataset = H5Dataset(f'{src_folder}{slide_h5}')
    else:
        dataset = PNGDataset(f'{src_folder}{slide_h5}', augmentation)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=12, pin_memory=True)

    features_list = []
    position_list = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0].to(device) #inputs := batch_size, 224, 224, 3
            mean = torch.tensor(NORMALIZATION_MEAN).view(1, 3, 1, 1).to(device)
            std = torch.tensor(NORMALIZATION_STD).view(1, 3, 1, 1).to(device)
            # inputs = (inputs / 255.0 - mean) / std
            # inputs = augmentation(inputs).to(device)
            inputs = inputs.to(device)

            features = model(inputs)
            # raise ValueError
            features = features.cpu().squeeze()
            if len(features.shape) == 1:  # if it's a 1D tensor
                features = features.unsqueeze(0)
            features_list.append(features)
            position_list.append(batch[1])

    # Wrap information
    graph_y = None
    try:
        graph_y = F.one_hot(torch.tensor(slide_labels[slide_name]), num_classes=len(label_convert_dict)).float().unsqueeze(0)
    except KeyError:
        print(f"Slide name {slide_name} not found, skipping...")
        with open('error_log.txt', 'a') as f:
            f.write(f"Slide name {slide_name} not found\n")

    a_graph = gData(
        x = torch.concat(features_list),
        pos = torch.concat(position_list),
        y = None, 
        graph_y = graph_y,
        slide_index = slide_name)
    
    target_objs.append(a_graph)

if config['dataset_name'] != 'Camelyon16':
    # pickle.dump(target_objs, open('/media/weiyi/My Passport/Data/C16_test.pkl','wb'))
    pickle.dump(target_objs, open(config['dst_file'],'wb'))
else:

    from shapely.geometry import Polygon, MultiPolygon, box
    from shapely.ops import unary_union
    import xml.etree.ElementTree as ET
    from pathlib import Path


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

    # xml_folder = '/media/weiyi/My Passport/Data/CAMELYON16/training/lesion_annotations'
    # xml_folder = '/media/weiyi/My Passport/Data/CAMELYON16/testing/lesion_annotations'
    xml_folder = config['xml_folder']
    ds_factor = 2
    patch_size = 224
    xml_folder = Path(xml_folder)


    for target_obj in tqdm(target_objs):
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
        # if target_obj.graph_y==1:
        #     print(target_obj.slide_index, np.array(overlap_ratios).max(), (np.array(overlap_ratios)>0.08).sum())


    # pickle.dump(target_objs, open('/media/weiyi/My Passport/Data/C16_test.pkl','wb'))
    pickle.dump(target_objs, open(config['dst_file'],'wb'))