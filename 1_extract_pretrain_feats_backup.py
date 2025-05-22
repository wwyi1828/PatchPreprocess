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
from transformers import ViTMAEModel, AutoModel
from transformers import AutoImageProcessor, AutoModelForZeroShotImageClassification, AutoProcessor
from transformers import pipeline
from transformers.image_processing_utils import BatchFeature
import transformers
import timm
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
parser.add_argument('--model_type', type=str, default='ResNet50')
parser.add_argument('--save_mode', type=str, default='single') # separate/single

args = parser.parse_args()

# args.config = 'configs/C17.yaml'

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
        # self.png_paths = sorted(list(Path(file_path).rglob('*.png'))) # Force fixed order
        self.augument = augument

    def __len__(self):
        return len(self.png_paths)
    
    def __getitem__(self, index):
        with Image.open(self.png_paths[index]).convert("RGB") as image:
            if self.augument is None:
                image = np.array(image)
                image = np.transpose(image, (2, 0, 1))
                image = torch.from_numpy(image).float()
            elif isinstance(self.augument, transformers.models.clip.processing_clip.CLIPProcessor):
                image = self.augument(images=image, return_tensors="pt")
            else:
                image = self.augument(image)
                if isinstance(image, BatchFeature):
                    image = torch.from_numpy(image.pixel_values[0])
            coord = np.array(self.png_paths[index].stem.split('_')).astype(int)

            return image, coord



# Create a TensorDataset and DataLoader
slide_labels = pd.read_csv(config['label_file'])
slide_labels = dict(zip(slide_labels['filename'], slide_labels['type']))
# slide_ext = list(slide_labels.keys())[0].split('.')[-1] # svs or tif
# label_convert_dict = {'Group_BT': 0, 'Group_AT': 1, 'Group_MT': 2}
# label_convert_dict = {'TCGA_LUAD': 0, 'TCGA_LUSC':1} 
if config['dataset_name'] == 'Camelyon16':
    label_convert_dict = {'Normal': 0, 'Tumor': 1} #Camelyon16
if config['dataset_name'] == 'Camelyon17':
    label_convert_dict = {'negative': 0, 'itc': 1, 'micro':2, 'macro':3}
elif config['dataset_name'] == 'BRACS':
    label_convert_dict = {'Group_BT': 0, 'Group_AT': 1, 'Group_MT': 2}
elif config['dataset_name'] == 'TCGA_LUNG':
    label_convert_dict = {'TCGA_LUAD': 0, 'TCGA_LUSC':1} 
elif config['dataset_name'] == 'TCGA_LGG':
    label_convert_dict = {'Astrocytoma': 0, 'Oligodendroglioma': 1, 'Oligoastrocytoma': 2, 'Low-Grade Glioma': 3} 
# slide_labels = {key.split('.')[0]: {'Normal': 0, 'Tumor': 1}[value] for key, value in slide_labels.items()}
# slide_labels = {key.split(f'.')[0]: {'TCGA_ACC': 0, 'TCGA_PCPG': 1}[value] for key, value in slide_labels.items()}
slide_labels = {key.split(f'.')[0]: label_convert_dict[value] for key, value in slide_labels.items()}

target_objs = []
# src_folder = '/pool1/data/Extracted_IMAGES/Camelyon16_test/patches/'
src_folder = config['src_folder']


torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.model_type == 'ResNet50':
    model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
    # model = models.resnet50(weights=None)
if args.model_type == 'ResNet18':
    model = models.resnet18(weights='ResNet18_Weights.DEFAULT')

if 'ResNet' in args.model_type:
    if 'ckpt_path' in config:
        ckpt_path = config['ckpt_path']
        ckpt = torch.load(ckpt_path)['state_dict']
        print(f'{ckpt_path} loaded')

        updated_ckpt = {key.replace("module.encoder.", ""): value for key, value in ckpt.items()}
        model.load_state_dict(updated_ckpt, strict=False)
    else:
        print("Load ImageNet weights")

    if args.model_type == 'ResNet18':
        model = nn.Sequential(*list(model.children())[:-1])
    elif args.model_type == 'ResNet50':
        model = ModifiedResNet(model)

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
        # transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        # transforms.RandomGrayscale(p=0.2),
        # transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

if args.model_type == 'MAE':
    model_config = 'facebook/vit-mae-large'
    model = ViTMAEModel.from_pretrained(model_config)
    print("Load MAE weights")
    image_processor = AutoImageProcessor.from_pretrained(model_config)
if args.model_type == 'DINOv2':
    model_config = 'facebook/dinov2-large'
    model = AutoModel.from_pretrained(model_config)
    print("Load DINO weights")
    image_processor = AutoImageProcessor.from_pretrained(model_config)
if args.model_type == 'PLIP':
    print("Load PLIP weights")
    pipe = pipeline("zero-shot-image-classification", model="vinid/plip")# Load model directly
    model = AutoModelForZeroShotImageClassification.from_pretrained("vinid/plip")
    image_processor = AutoProcessor.from_pretrained("vinid/plip")
if args.model_type == 'UNI':
    print("Load UNI weights")
    model = timm.create_model(
        "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
    )
    local_dir = "/data/checkpoints/UNI/"
    model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)
    image_processor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
if args.model_type == 'CONCH':
    print('Load CONCH weights')
    from conch.open_clip_custom import create_model_from_pretrained
    model, image_processor = create_model_from_pretrained('conch_ViT-B-16', "/data/checkpoints/CONCH/pytorch_model.bin")

model.eval()
model.cuda()
model.to(device)


# processed_slides = {obj.slide_index for obj in target_objs}
for slide_h5 in tqdm(os.listdir(src_folder)):
    slide_name = slide_h5.split('.')[0]

    # if slide_name in processed_slides:
    #     continue
    if False:
        dataset = H5Dataset(f'{src_folder}{slide_h5}')
    else:
        dataset = PNGDataset(f'{src_folder}{slide_h5}', image_processor)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=12, pin_memory=True)

    features_list = []
    position_list = []
    # with torch.no_grad():
    with torch.inference_mode():
        for batch in dataloader:
            inputs = batch[0].to(device) #inputs := batch_size, 224, 224, 3
            inputs = inputs.to(device)

            if args.model_type == 'PLIP':
                features = model.vision_model(pixel_values=inputs['pixel_values'].squeeze(1)).pooler_output
            elif args.model_type == 'CONCH':
                features = model.encode_image(inputs, proj_contrast=False, normalize=False)
            else:
                features = model(inputs)
                if args.model_type == 'MAE':
                    features = features.last_hidden_state
                    features = torch.mean(features, dim=1)
                if args.model_type == 'DINOv2':
                    features = features.last_hidden_state
                    features = features[:, 0, :]
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

    if graph_y is not None:
        a_graph = gData(
            x = torch.concat(features_list),
            pos = torch.concat(position_list),
            y = None, 
            graph_y = graph_y,
            slide_index = slide_name)
        target_objs.append(a_graph)

dst_file = Path(config['dst_file'])
# if config['dataset_name'] != 'Camelyon16':
if 'xml_folder' not in config:
    print('no xml folder')
    # pickle.dump(target_objs, open('/media/weiyi/My Passport/Data/C16_test.pkl','wb'))
    if dst_file.suffix == '.pkl':
        pickle.dump(target_objs, open(str(dst_file),'wb'))
        # pickle.dump(target_objs, open(config['dst_file'],'wb'))
    else:
        dst_file.mkdir(parents=True, exist_ok=True)
        for target_obj in target_objs:
            slide_name = target_obj.slide_index
            h5_file_path = dst_file / f"{slide_name}.h5"
            
            with h5py.File(str(h5_file_path), 'w') as f:
                f.create_dataset('feats', data=target_obj.x.cpu())
                f.create_dataset('cords', data=target_obj.pos.cpu())
                if target_obj.graph_y is not None:
                    f.create_dataset('label', data=target_obj.graph_y.cpu())
else:
    print("xml_folder is defined and its value is:", config['xml_folder'])
    from shapely.geometry import Polygon, MultiPolygon, box
    from shapely.ops import unary_union
    import xml.etree.ElementTree as ET
    from pathlib import Path
    from rtree import index


    def xml_position(xml_path, ds_factor=2):
        root = ET.parse(xml_path).getroot()
        idx = index.Index()  # 空间索引
        polygons = []
        
        for annotation in root.findall('.//Annotation'):
            coords = []
            for coordinate in annotation.findall('.//Coordinate'):
                x = float(coordinate.get('X'))
                y = float(coordinate.get('Y'))
                coords.append((x/ds_factor, y/ds_factor))
            if len(coords) < 4:
                print(f"Skipping invalid coordinates: {coords}")
                continue  # skip to the next annotation
            poly = Polygon(coords)
            if not poly.is_valid:
                poly = poly.buffer(1e-6)
            polygons.append(poly)
            idx.insert(len(polygons)-1, poly.bounds)  # 将多边形添加到索引中

        positive_polygons = []
        negative_polygons = []

        for pos, poly in enumerate(polygons):
            possible_overlaps = list(idx.intersection(poly.bounds))
            is_negative = any(polygons[other_pos].contains(poly) and other_pos != pos for other_pos in possible_overlaps)
            if is_negative:
                negative_polygons.append(poly)
            else:
                positive_polygons.append(poly)

        positive_union = unary_union(positive_polygons)
        negative_union = unary_union(negative_polygons)
        
        return positive_union, negative_union

    # xml_folder = '/media/weiyi/My Passport/Data/CAMELYON16/training/lesion_annotations'
    # xml_folder = '/media/weiyi/My Passport/Data/CAMELYON16/testing/lesion_annotations'
    xml_folder = config['xml_folder']
    ds_factor = 2
    patch_size = 224
    xml_folder = Path(xml_folder)

    for target_obj in tqdm(target_objs):
        xml_item = xml_folder.joinpath(f'{target_obj.slide_index}.xml')
        if xml_item.exists():
            positive_union, negative_union = xml_position(xml_item)
            annot_coordinates = positive_union.difference(negative_union)  # 计算正多边形与负多边形的差集
        else:
            annot_coordinates = MultiPolygon()

        overlap_ratios = []
        for pcoord in target_obj.pos:
            patch_box = box(pcoord[0], pcoord[1], pcoord[0] + patch_size, pcoord[1] + patch_size)
            intersection = patch_box.intersection(annot_coordinates)
            ratio = intersection.area / patch_box.area
            overlap_ratios.append(ratio)
        target_obj.y = torch.stack([torch.tensor(_) for _ in overlap_ratios])
        # if target_obj.graph_y==1:
        #     print(target_obj.slide_index, np.array(overlap_ratios).max(), (np.array(overlap_ratios)>0.08).sum())


    # pickle.dump(target_objs, open('/media/weiyi/My Passport/Data/C16_test.pkl','wb'))
    # pickle.dump(target_objs, open(config['dst_file'],'wb'))
    if dst_file.suffix == '.pkl':
        pickle.dump(target_objs, open(str(dst_file),'wb'))
        # pickle.dump(target_objs, open(config['dst_file'],'wb'))
    else:
        dst_file.mkdir(parents=True, exist_ok=True)
        for target_obj in target_objs:
            slide_name = target_obj.slide_index
            h5_file_path = dst_file / f"{slide_name}.h5"
            
            with h5py.File(str(h5_file_path), 'w') as f:
                f.create_dataset('feats', data=target_obj.x.cpu())
                f.create_dataset('cords', data=target_obj.pos.cpu())
                if target_obj.y is not None:
                    f.create_dataset('ratios', data=target_obj.y.cpu())
                if target_obj.graph_y is not None:
                    f.create_dataset('label', data=target_obj.graph_y.cpu())