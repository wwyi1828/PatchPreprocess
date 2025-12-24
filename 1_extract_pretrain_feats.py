import h5py
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models, transforms
import numpy as np
import pandas as pd
import os
import pickle
from types import SimpleNamespace
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


from shapely.geometry import Polygon, MultiPolygon, box
from shapely.ops import unary_union
from shapely.affinity import scale
from shapely.prepared import prep
import xml.etree.ElementTree as ET
from pathlib import Path
from rtree import index
from lxml import etree
import tempfile
import random

def xml_position(xml_path, ds_factor=2):
    root = ET.parse(xml_path).getroot()
    idx = index.Index()  # 空间索引
    polygons = []
    
    for annotation in root.findall('.//Annotation'):
        coords = []
        coordinate_elements = annotation.findall('.//Coordinate')
        if not coordinate_elements:
            coordinate_elements = annotation.findall('.//Vertex')

        # for coordinate in annotation.findall('.//Coordinate'):
        for coordinate in coordinate_elements:
            x = float(coordinate.get('X'))
            y = float(coordinate.get('Y'))
            coords.append((x/ds_factor, y/ds_factor))
        # if len(coords) < 4:
        if len(coords) < 3: # was 4
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

def safe_xml_position(xml_path, ds_factor=2):
    try:
        # 尝试直接调用原始函数
        return xml_position(xml_path, ds_factor)
    except ET.ParseError as e:
        print(f"捕获XML解析错误: {e}，尝试自动修复...")
        
        try:
            # 读取原始内容
            with open(xml_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 添加根元素包装内容
            fixed_content = f'<root>\n{content}\n</root>'
            
            # 使用lxml解析修复后的内容
            parser = etree.XMLParser(recover=True)
            root = etree.fromstring(fixed_content.encode('utf-8'), parser)
            
            # 创建临时文件
            with tempfile.NamedTemporaryFile(suffix='.xml', delete=False) as temp:
                temp_path = temp.name
                temp.write(etree.tostring(root, encoding='utf-8'))
            
            print(f"使用临时修复文件: {temp_path}")
            
            try:
                # 使用临时文件调用原始函数
                result = xml_position(temp_path, ds_factor)
                return result
            finally:
                # 无论成功与否，都删除临时文件
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    print(f"已删除临时文件: {temp_path}")
        
        except Exception as inner_e:
            print(f"修复尝试失败: {inner_e}")
            raise e  # 重新抛出原始错误


def infer_ds_factor(png_paths, annotations, *, patch_size=224, candidate_factors=None, sample_limit=2000):
    """Infer the downsample factor by comparing patch boxes against annotations."""
    if candidate_factors is None:
        candidate_factors = (1, 2, 4, 8, 16)

    # Ensure we have annotations to compare against
    if annotations is None or annotations.is_empty:
        return 1

    # Gather patch coordinates from filenames
    coords = []
    for path in png_paths:
        parts = path.stem.split('_')
        if len(parts) != 2:
            continue
        try:
            x_val = int(parts[0])
            y_val = int(parts[1])
        except ValueError:
            continue
        coords.append((x_val, y_val))

    if not coords:
        return 1

    # Limit the number of patches to keep the check lightweight
    if sample_limit is not None and len(coords) > sample_limit:
        coords = random.sample(coords, sample_limit)

    patch_boxes = [box(x, y, x + patch_size, y + patch_size) for x, y in coords]

    best_factor = candidate_factors[0]
    best_hits = -1

    for factor in candidate_factors:
        # Skip nonsensical factors
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
elif config['dataset_name'] == 'TCGA_BRAC':
    label_convert_dict = {'Infiltrating Ductal Carcinoma': 0, 'Infiltrating Lobular Carcinoma': 1, 'Other': 2} 
elif config['dataset_name'] == 'TCGA_COAD':
    label_convert_dict = {'Colon Adenocarcinoma': 0, 'Colon Adenocarcinoma, Mucinous Type': 1, 'Unknown': 2} 
elif config['dataset_name'] == 'BACH':
    label_convert_dict = {'negative': 0, 'positive': 1} 
elif config['dataset_name'] == 'trastuzumab':
    label_convert_dict = {'nonresponder': 0, 'responder': 1}
elif config['dataset_name'] == 'Yale_HER2':
    label_convert_dict = {'negative': 0, 'positive': 1} 
# slide_labels = {key.split('.')[0]: {'Normal': 0, 'Tumor': 1}[value] for key, value in slide_labels.items()}
# slide_labels = {key.split(f'.')[0]: {'TCGA_ACC': 0, 'TCGA_PCPG': 1}[value] for key, value in slide_labels.items()}
slide_labels = {key.split(f'.')[0]: label_convert_dict[value] for key, value in slide_labels.items()}

dst_file = Path(config['dst_file'])
if dst_file.suffix == '.pkl':
    target_objs = []

# src_folder = '/pool1/data/Extracted_IMAGES/Camelyon16_test/patches/'
src_folder = config['src_folder']


torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
random.seed(42)
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
if args.model_type == 'UNIv2':
    # from huggingface_hub import hf_hub_download
    local_dir = "/data/checkpoints/UNIv2"
    # os.makedirs(local_dir, exist_ok=True)  # create directory if it does not exist
    # hf_hub_download("MahmoodLab/UNI2-h", filename="pytorch_model.bin", local_dir=local_dir, force_download=True)
    timm_kwargs = {
                'model_name': 'vit_giant_patch14_224',
                'img_size': 224, 
                'patch_size': 14, 
                'depth': 24,
                'num_heads': 24,
                'init_values': 1e-5, 
                'embed_dim': 1536,
                'mlp_ratio': 2.66667*2,
                'num_classes': 0, 
                'no_embed_class': True,
                'mlp_layer': timm.layers.SwiGLUPacked, 
                'act_layer': torch.nn.SiLU, 
                'reg_tokens': 8, 
                'dynamic_img_size': True
            }
    model = timm.create_model(
        pretrained=False, **timm_kwargs
    )
    model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)
    image_processor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    model.eval()
if args.model_type == 'CONCH':
    print('Load CONCH weights')
    from conch.open_clip_custom import create_model_from_pretrained
    model, image_processor = create_model_from_pretrained('conch_ViT-B-16', "/data/checkpoints/CONCH/pytorch_model.bin")


if args.model_type == 'Virchow2':
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    from timm.layers import SwiGLUPacked
    print('Load Virchow2 weights')
    model = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
    image_processor = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

if args.model_type == 'GigaPath':
    print('Load GigaPath weights')
    model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
    image_processor = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
if args.model_type == 'Raw':
    print('Extract Raw images (no encoding)')
    model = nn.Identity()
    # Use standard ImageNet normalization or raw tensor depending on requirement. 
    # Usually standard normalization is expected for downstream DL.
    image_processor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

model.eval()
model.cuda()
model.to(device)

# processed_slides = {obj.slide_index for obj in target_objs}
for slide_h5 in tqdm(os.listdir(src_folder)):
    slide_name = slide_h5.split('.')[0]
    # if slide_name in processed_slides:
    #     continue
    # Build slide path robustly regardless of trailing slash in src_folder
    slide_path = os.path.join(src_folder, slide_h5)
    # Skip non-directories (e.g., stray files)
    if not os.path.isdir(slide_path):
        continue
    if False:
        dataset = H5Dataset(slide_path)
    else:
        dataset = PNGDataset(slide_path, image_processor)
    # print(slide_name, len(dataset), f'{src_folder}{slide_h5}')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=12, pin_memory=True, prefetch_factor=4)

    features_list = []
    position_list = []

    # If the dataset is empty (no PNGs), skip to avoid concat error
    if len(dataset) == 0:
        print(f"Warning: no patches found under {slide_path}; skipping slide {slide_name}")
        with open('error_log.txt', 'a') as f:
            f.write(f"Empty slide directory: {slide_path}\n")
        continue
    # with torch.no_grad():
    with torch.inference_mode():
        for batch in dataloader:
            inputs = batch[0] # inputs := batch_size, 3, 224, 224
            
            if args.model_type != 'Raw':
                inputs = inputs.to(device)

            if args.model_type == 'PLIP':
                features = model.vision_model(pixel_values=inputs['pixel_values'].squeeze(1)).pooler_output
            elif args.model_type == 'CONCH':
                features = model.encode_image(inputs, proj_contrast=False, normalize=False)
            elif args.model_type == 'Raw':
                # No model forward pass needed for identity, inputs are already the features
                features = inputs
            else:
                features = model(inputs)
                if args.model_type == 'MAE':
                    features = features.last_hidden_state
                    features = torch.mean(features, dim=1)
                elif args.model_type == 'DINOv2':
                    features = features.last_hidden_state
                    features = features[:, 0, :]
                elif args.model_type == 'Virchow2':
                    # Virchow2 returns [batch_size, seq_len, embed_dim]
                    # Use CLS token (first token) for classification-like features
                    features = features[:, 0, :]
            # raise ValueError
            features = features.cpu()
            # Ensure features is always 2D: [batch_size, feature_dim]
            if len(features.shape) == 1:  # Single sample case
                features = features.unsqueeze(0)
            elif len(features.shape) == 3:  # [batch_size, 1, feature_dim] case
                features = features.squeeze(1)
            # Now features should be [batch_size, feature_dim]
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
        a_graph = SimpleNamespace(
            x = torch.concat(features_list),
            pos = torch.concat(position_list),
            y = None,
            graph_y = graph_y,
            slide_index = slide_name)

        if 'xml_folder' in config:
            print("xml_folder is defined and its value is:", config['xml_folder'], slide_name)

            xml_folder = Path(config['xml_folder'])
            patch_size = 224
            xml_item = xml_folder.joinpath(f'{a_graph.slide_index}.xml')

            annot_coordinates = MultiPolygon()
            inferred_ds_factor = 1

            if xml_item.exists():
                positive_union, negative_union = safe_xml_position(xml_item, ds_factor=1)
                annot_coordinates_ds1 = positive_union.difference(negative_union)

                candidate_factors = config.get('ds_factor_candidates', [1, 2, 4, 8, 16])
                if not isinstance(candidate_factors, (list, tuple)):
                    candidate_factors = [candidate_factors]
                parsed_factors = []
                for factor in candidate_factors:
                    try:
                        value = int(factor)
                    except (TypeError, ValueError):
                        continue
                    if value > 0:
                        parsed_factors.append(value)
                candidate_factors = parsed_factors
                # Ensure 1 is always considered to avoid empty candidate lists
                if 1 not in candidate_factors:
                    candidate_factors = [1] + candidate_factors
                if not candidate_factors:
                    candidate_factors = [1, 2, 4, 8, 16]

                inferred_ds_factor = infer_ds_factor(
                    getattr(dataset, 'png_paths', []),
                    annot_coordinates_ds1,
                    patch_size=patch_size,
                    candidate_factors=candidate_factors
                )
                print(f"Inferred ds_factor for {a_graph.slide_index}: {inferred_ds_factor}")

                annot_coordinates = scale(
                    annot_coordinates_ds1,
                    xfact=1.0 / inferred_ds_factor,
                    yfact=1.0 / inferred_ds_factor,
                    origin=(0, 0)
                )
            else:
                print(f"Annotation file not found for {a_graph.slide_index}, using empty annotations.")

            overlap_ratios = []
            for pcoord in a_graph.pos:
                patch_box = box(pcoord[0], pcoord[1], pcoord[0] + patch_size, pcoord[1] + patch_size)
                intersection = patch_box.intersection(annot_coordinates)
                ratio = intersection.area / patch_box.area if patch_box.area else 0
                overlap_ratios.append(ratio)
            a_graph.y = torch.stack([torch.tensor(_) for _ in overlap_ratios])
        
        if dst_file.suffix == '.pkl':
            target_objs.append(a_graph)
        else:
            dst_file.mkdir(parents=True, exist_ok=True)
            slide_name = a_graph.slide_index
            h5_file_path = dst_file / f"{slide_name}.h5"

            with h5py.File(str(h5_file_path), 'w') as f:
                f.create_dataset('feats', data=a_graph.x.cpu())
                f.create_dataset('cords', data=a_graph.pos.cpu())
                if a_graph.y is not None:
                    f.create_dataset('ratios', data=a_graph.y.cpu())
                if a_graph.graph_y is not None:
                    f.create_dataset('label', data=a_graph.graph_y.cpu())

if dst_file.suffix == '.pkl':
    pickle.dump(target_objs, open(str(dst_file),'wb'))
