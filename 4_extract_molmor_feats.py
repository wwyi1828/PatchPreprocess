import h5py
import torch
import copy
import scanpy as sc
import numpy as np
import scipy.sparse as sp
import pandas as pd
import anndata as ad
from transformers import AutoImageProcessor, AutoModelForZeroShotImageClassification, AutoProcessor, CLIPImageProcessor
from transformers import pipeline
from tqdm import tqdm
import os
import timm
from pathlib import Path
import argparse
from utils_gene import *

def read_h5_data(file_path):
    with h5py.File(file_path, 'r') as f:
        barcodes = f['barcode'][()]
        coords = f['coords'][()]
        images = f['img'][()]
        return barcodes, coords, images

parser = argparse.ArgumentParser(description='Read Configuration File')
parser.add_argument('--dataset', type=str, default='NCBI_SKIN')
parser.add_argument('--batch_size', type=int, default=512, 
                    help='Batch size for preprocessing.')
parser.add_argument('--gene_prc', action='store_true')
parser.add_argument('--imge_prc', type=str, default=None)
parser.add_argument('--dst_file', default="/pool2/data/Mor2Mol")
args = parser.parse_args()
dst_file = Path(f"{args.dst_file}/{args.dataset}")

dataset_name = 'SPA_breast_2'
dataset_name = args.dataset
if dataset_name == 'MISC_brain':
    target_dataset = [f'MISC{idx}' for idx in range(1,13)] #All 20x, Visium
elif dataset_name == 'MISC_lung':
    target_dataset = [f'MISC{idx}' for idx in range(13,33)] #All 40x, have other available visium
elif dataset_name == 'MISC_bowel':
    target_dataset = [f'MISC{idx}' for idx in range(33,74)] #40x, Visium
elif dataset_name == 'INT_Lymphnode':
    target_dataset = [f'INT{idx}' for idx in range(1,25)]
elif dataset_name == 'Xenium_Lung':
    target_dataset = [f'NCBI{idx}' for idx in range(856,885)]
elif dataset_name == 'ColonMap':
    target_dataset = [f'NCBI{idx}' for idx in range(33,74)]
elif dataset_name == 'NCBI_SKIN':
    target_dataset = [f'NCBI{idx}' for idx in range(469,523)] # Visium, 20x, FFPE
elif dataset_name == 'NCBI_brain':
    target_dataset = [f'NCBI{idx}' for idx in range(336,411)]
elif dataset_name == 'NCBI_Spinal':
    target_dataset = [f'NCBI{idx}' for idx in range(11,336)] # ST, Non Visium
elif dataset_name == 'SPA_breast_1':
    target_dataset = [f'SPA{idx}' for idx in range(51,155)][0:68] # All 20x
elif dataset_name == 'SPA_breast_2':
    target_dataset = [f'SPA{idx}' for idx in range(51,155)][68:] # All 20x
elif dataset_name == 'SPA_breast':
    target_dataset = [f'SPA{idx}' for idx in range(51,155)] # All 20x
elif dataset_name == 'TENX_Xenium_breast':
    target_dataset = [f'TENX{idx}' for idx in range(94, 100)]



[f'NCBI{idx}' for idx in range(759,771)] # Human + Skin + ST

datastem = Path(f"/data/data/ST/HEST1K/")
dst_file.mkdir(parents=True, exist_ok=True)
top_k = 10000

if args.gene_prc:
    sorted_hvg, sorted_heg = dataset_topgenes(datastem, target_dataset, top_k=top_k, verbose=True)

    for item in tqdm(target_dataset):
        mol_path = datastem.joinpath(f'st/{item}.h5ad')
        if not mol_path.exists():
            tqdm.write(f"File not found: {mol_path}, skipping...")
            continue
        adata = sc.read_h5ad(mol_path)
        adata = adata[:, ~adata.var_names.str.startswith('__ambiguous')]
        adata = avg_duplicate_geneexpr(adata)
        adata = avg_duplicate_cells(adata)

        #sc.pp.normalize_total(adata, target_sum=1e4) 
        sc.pp.log1p(adata)
        ### Match
        mor_path = datastem.joinpath(f'patches/{item}.h5')
        barcodes, coords, images = read_h5_data(mor_path)
        barcode_list = [b.item().decode() for b in barcodes]
        adata_barcodes = list(adata.obs.index)
        barcode_to_image_idx = {barcode: idx for idx, barcode in enumerate(barcode_list)}

        valid_indices = [] 
        for i, barcode in enumerate(adata_barcodes):
            if barcode in barcode_to_image_idx:
                valid_indices.append(i)
        adata = adata[valid_indices].copy()
        ###
        if sp.issparse(adata.X):
            X_hvg = adata[:, sorted_hvg].X.toarray().astype(np.float16)
            X_heg = adata[:, sorted_heg].X.toarray().astype(np.float16)
        else:
            X_hvg = adata[:, sorted_hvg].X.astype(np.float16)
            X_heg = adata[:, sorted_heg].X.astype(np.float16)
        X_heg = torch.from_numpy(X_heg).float()
        X_hvg = torch.from_numpy(X_hvg).float()

        # array_rows = adata.obs['array_row'].values
        # array_cols = adata.obs['array_col'].values
        # shifted_rows = array_rows - array_rows.min()
        # shifted_cols = array_cols - array_cols.min()
        # array_coordinates = torch.stack([
        #     torch.tensor(shifted_cols, dtype=torch.int),
        #     torch.tensor(shifted_rows, dtype=torch.int)
        # ], dim=1)

        valid_image_indices = []
        for i, barcode in enumerate(adata_barcodes):
            if barcode in barcode_to_image_idx:
                valid_image_indices.append(barcode_to_image_idx[barcode])

        x_coords = coords[:, 0]
        y_coords = coords[:, 1]
        x_min, y_min = np.min(x_coords), np.min(y_coords)
        normalized_x = (x_coords - x_min) / 224
        normalized_y = (y_coords - y_min) / 224
        normalized_coords = torch.from_numpy(np.column_stack((normalized_x, normalized_y))).float()
        normalized_coords = normalized_coords[valid_image_indices]
        
        remap_coords = map_to_integer_grid(coords)
        # unique_positions = set(tuple(coord) for coord in map_to_integer_grid(coords))
        # assert len(unique_positions) == len(coords), f'AssertionError: origin: {len(coords)} remap: {len(unique_positions)}'

        coordinates = remap_coords
        coordinates = coordinates[valid_image_indices]
        # adata.obs.index
        # barcodes[valid_image_indices]

        # x_coords = coords[:, 0]
        # y_coords = coords[:, 1]
        # x_min, y_min = np.min(x_coords), np.min(y_coords)
        # normalized_x = (x_coords - x_min) / 224
        # normalized_y = (y_coords - y_min) / 224
        # normalized_coords = torch.from_numpy(np.column_stack((normalized_x, normalized_y))).float()
        # print(((normalized_coords[valid_image_indices]-array_coordinates.float())**2).mean(), ((normalized_coords[valid_image_indices]-coordinates.float())**2).mean())

        gene_folder = dst_file / "gene"
        gene_folder.mkdir(parents=True, exist_ok=True)
        h5_file_path = gene_folder / f"{item}.h5"

        with h5py.File(str(h5_file_path), 'w') as f:
            f.create_dataset('cords', data=coordinates)
            f.create_dataset('float_cords', data=normalized_coords)
            f.create_dataset('hvg', data=X_hvg)
            f.create_dataset('heg', data=X_heg)
            # Save the gene names as string arrays
            hvg_gene_names = np.array(sorted_hvg, dtype='S')
            heg_gene_names = np.array(sorted_heg, dtype='S')
            f.create_dataset('hvg_gene_names', data=hvg_gene_names)
            f.create_dataset('heg_gene_names', data=heg_gene_names)


    check_list1 = copy.deepcopy(list(adata.obs.index))
    print("GENE extraction done")
###############
if args.imge_prc is None:
    raise SystemExit

# model_type = 'UNI'
model_type = args.imge_prc
if model_type == 'PLIP':
    print("Load PLIP weights")
    pipe = pipeline("zero-shot-image-classification", model="vinid/plip")# Load model directly
    model = AutoModelForZeroShotImageClassification.from_pretrained("vinid/plip")
    image_processor = AutoProcessor.from_pretrained("vinid/plip")
elif model_type == 'UNI':
    print("Load UNI weights")
    model = timm.create_model(
        "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
    )
    local_dir = "/data/checkpoints/UNI/"
    model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)
    image_processor = CLIPImageProcessor(
        do_resize=False,
        do_center_crop=False,
        do_normalize=True,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225]
    )
elif model_type == 'R50':
    from torchvision import models, transforms
    from utils import ModifiedResNet
    model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
    model = ModifiedResNet(model)
    image_processor = CLIPImageProcessor(
        do_resize=False,
        do_center_crop=False,
        do_normalize=True,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225]
    )


# batch_size = 512
batch_size = args.batch_size
if model_type != 'Raw':
    model = model.cuda()
for item in tqdm(target_dataset):
    mor_path = datastem.joinpath(f'patches/{item}.h5')
    if not mor_path.exists():
        tqdm.write(f"File not found: {mor_path}, skipping...")
        continue
    barcodes, coords, images = read_h5_data(mor_path)

    ### Match
    mol_path = datastem.joinpath(f'st/{item}.h5ad')
    adata = sc.read_h5ad(mol_path)
    adata = adata[:, ~adata.var_names.str.startswith('__ambiguous')]
    adata = avg_duplicate_geneexpr(adata)
    adata = avg_duplicate_cells(adata)
    barcode_list = [b.item().decode() for b in barcodes]
    adata_barcodes = list(adata.obs.index)
    barcode_to_image_idx = {barcode: idx for idx, barcode in enumerate(barcode_list)}

    valid_image_indices = []
    for i, barcode in enumerate(adata_barcodes):
        if barcode in barcode_to_image_idx:
            valid_image_indices.append(barcode_to_image_idx[barcode])
    images = images[valid_image_indices]
    coords = coords[valid_image_indices]
    barcodes = barcodes[valid_image_indices]
    barcode_list = [b.item().decode() for b in barcodes]
    ###

    num_images = images.shape[0]
    num_batches = int(np.ceil(num_images / batch_size))
    features_list = []

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_images)
        batch_images = images[start_idx:end_idx]

        with torch.no_grad():
            if model_type == 'PLIP':
                inputs = image_processor(images=list(batch_images), return_tensors="pt")
                outputs = model.vision_model(pixel_values=inputs['pixel_values'].cuda().squeeze(1)).pooler_output
            elif model_type == 'Raw':
                outputs = torch.tensor(batch_images)
            else:
                inputs = image_processor(images=list(batch_images), return_tensors="pt")
                outputs = model(inputs['pixel_values'].cuda())
            if len(outputs.shape) == 1:
                outputs = outputs.unsqueeze(0)
            features_list.append(outputs.cpu())
    
    img_feats = torch.concat(features_list)

    gene_folder = dst_file / f"imge_{model_type}"
    gene_folder.mkdir(parents=True, exist_ok=True)
    h5_file_path = gene_folder / f"{item}.h5"

    with h5py.File(str(h5_file_path), 'w') as f:
        f.create_dataset('img_feats', data=img_feats)
check_list2 = copy.deepcopy(barcode_list)


# print(f"adata.obs.index 长度: {len(check_list1)}")
# print(f"barcode_list 长度: {len(check_list2)}")

# if len(check_list1) == len(check_list2):
#     are_identical = sorted(check_list1) == sorted(check_list2)
#     print(f"排序后列表是否完全相同: {are_identical}")
    
#     set_equal = set(check_list1) == set(check_list2)
#     print(f"集合是否相同: {set_equal}")
    
#     if set_equal and not are_identical:
#         print("两个列表包含相同的元素，但顺序不同")
# else:
#     common = set(check_list1) & set(check_list2)
#     print(f"共有 {len(common)} 个相同元素")
#     print(f"adata.obs.index 独有元素: {len(set(check_list1) - set(check_list2))}")
#     print(f"barcode_list 独有元素: {len(set(check_list2) - set(check_list1))}")