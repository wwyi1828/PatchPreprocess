import torch
import copy
import scanpy as sc
import numpy as np
import scipy.sparse as sp
import anndata as ad
import pandas as pd
from transformers import AutoModelForZeroShotImageClassification, AutoProcessor, CLIPImageProcessor
from transformers import pipeline
from tqdm import tqdm
import pybiomart as pbm
import os
import timm
import json
from pathlib import Path
import argparse
from PIL import Image
from utils_gene import *

def is_xenium_dataset(dataset_name):
    """Check if the dataset is a Xenium dataset"""
    return dataset_name in ['Xenium_Lung', 'Xenium112_breast']

def filter_blank_genes(adata, verbose=False):
    """Filter out BLANK_ genes from Xenium data"""
    if verbose:
        print(f"Before filtering BLANK_ genes: {adata.n_vars} genes")
    
    # Filter out genes starting with 'BLANK_'
    blank_mask = (
        adata.var_names.str.startswith('BLANK_') |
        adata.var_names.str.startswith('NegControlProbe') |
        adata.var_names.str.startswith('NegControlCodeword')
    )
    n_blank_genes = blank_mask.sum()
    
    if n_blank_genes > 0:
        adata = adata[:, ~blank_mask].copy()
        if verbose:
            print(f"Filtered out {n_blank_genes} BLANK_ genes")
            print(f"After filtering BLANK_ genes: {adata.n_vars} genes")
    else:
        if verbose:
            print("No BLANK_ genes found")
    
    return adata

def pause_if_nonfinite(array, name, enabled):
    if not enabled:
        return
    arr = array if isinstance(array, np.ndarray) else np.asarray(array)
    if not np.isfinite(arr).all():
        num_nan = int(np.isnan(arr).sum())
        num_inf = int(np.isinf(arr).sum())
        print(f"[Warning] Non-finite values detected in {name}: nan={num_nan}, inf={num_inf}")
        try:
            input("Press Enter to continue, or Ctrl+C to abort...")
        except KeyboardInterrupt:
            raise

def ensure_pil_image(image):
    if isinstance(image, Image.Image):
        return image
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[0] != arr.shape[-1]:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
    if arr.dtype != np.uint8:
        max_val = float(arr.max()) if arr.size else 0.0
        if max_val <= 1.0:
            arr = (arr * 255.0).clip(0, 255)
        else:
            arr = arr.clip(0, 255)
        arr = arr.astype(np.uint8)
    return Image.fromarray(np.ascontiguousarray(arr))

parser = argparse.ArgumentParser(description='Read Configuration File')
parser.add_argument('--dataset', type=str, default='NCBI_SKIN')
parser.add_argument('--batch_size', type=int, default=512, 
                    help='Batch size for preprocessing.')
parser.add_argument('--gene_prc', action='store_true')
parser.add_argument('--imge_prc', type=str, default=None)
parser.add_argument('--dst_pixelsize', type=float, default=55)
parser.add_argument('--dst_file', default="/pool2/data/Mor2Mol")
parser.add_argument('--normalize', action='store_true', 
                    help='Whether to apply normalize_total in individual dataset processing')
parser.add_argument('--log1p', action='store_true', 
                    help='Whether to apply log1p in individual dataset processing')
parser.add_argument('--pause_on_overflow', action='store_false',
                    help='Pause and wait for input if overflow/NaN/Inf detected')
args = parser.parse_args()
dst_file = Path(f"{args.dst_file}/{args.dataset}")

# dataset_name = 'SPA_breast'
dataset_name = args.dataset
if dataset_name == 'MISC_brain':
    target_dataset = [f'MISC{idx}' for idx in range(1,13)] #All 20x, Visium
elif dataset_name == 'MISC_lung':
    target_dataset = [f'MISC{idx}' for idx in range(13,33)] #All 40x, have other available visium
elif dataset_name == 'MISC_bowel':
    target_dataset = [f'MISC{idx}' for idx in range(33,74)] #40x, Visium
elif dataset_name == 'MISC_heart':
    target_dataset = [f'MISC{idx}' for idx in range(101,143)] #40x, Visium (119~127 20X)
elif dataset_name == 'INT_kidney':
    target_dataset = [f'INT{idx}' for idx in range(1,25)]
elif dataset_name == 'Xenium_Lung':
    target_dataset = [f'NCBI{idx}' for idx in range(856,885)]
elif dataset_name == 'NCBI_skin':
    target_dataset = [f'NCBI{idx}' for idx in range(469,523)] # Visium, 20x, FFPE
elif dataset_name == 'NCBI_brain':
    target_dataset = [f'NCBI{idx}' for idx in range(336,411)]
elif dataset_name == 'NCBI_spinal':
    target_dataset = [f'NCBI{idx}' for idx in range(11,336)] # ST, Non Visium
elif dataset_name == 'SPA_breast_1':
    target_dataset = [f'SPA{idx}' for idx in range(51,155)][0:68] # All 20x
elif dataset_name == 'SPA_breast_2':
    target_dataset = [f'SPA{idx}' for idx in range(51,155)][68:] # All 20x
elif dataset_name == 'SPA_breast':
    target_dataset = [f'SPA{idx}' for idx in range(51,155)] # All 20x
elif dataset_name == 'Xenium112_breast':
    target_dataset = [f'TENX{idx}' for idx in range(94, 100)]
elif dataset_name == 'HD':
    target_dataset = [f'HD{idx}' for idx in range(1,7)]

data_dir = '/data/data/ST/HEST1K/st'
existing_datasets = []
missing_datasets = []
for dataset_id in target_dataset:
    file_path = os.path.join(data_dir, f'{dataset_id}.h5ad')
    if os.path.exists(file_path):
        existing_datasets.append(dataset_id)
    else:
        missing_datasets.append(dataset_id)
print(f"# Total files: {len(target_dataset)}")
print(f"# Existing files: {len(existing_datasets)}")
print(f"# Missing files: {len(missing_datasets)}")
if missing_datasets:
    print(f"Missing files: {missing_datasets}")
target_dataset = existing_datasets

datastem = Path(f"/data/data/ST/HEST1K/")
dst_file.mkdir(parents=True, exist_ok=True)
top_k = 2000

if args.gene_prc:
    # Step 1: Process all datasets and combine to find HVG/HEG
    print("Step 1: Processing datasets to find HVG/HEG...")
    verbose = True
    all_adata = []
    for item in target_dataset:
        mol_path = datastem.joinpath(f'st/{item}.h5ad')
        adata = sc.read_h5ad(mol_path)
        
        # Standardize gene names and aggregate expression (Sum)
        adata = clean_var_names(adata, verbose=verbose)
        adata = aggregate_adata(adata, verbose=verbose)
        
        # Filter BLANK_ genes for Xenium datasets
        if is_xenium_dataset(dataset_name):
            adata = filter_blank_genes(adata, verbose)
        
        # Filter to only include spots that have corresponding morphology data
        mor_path = datastem.joinpath(f'patches/{item}.h5')
        if mor_path.exists():
            barcodes, coords, images = read_h5_data(mor_path)
            gene_barcodes = list(adata.obs.index)
            valid_gene_indices, valid_image_indices = create_barcode_mapping(gene_barcodes, barcodes)
            adata = adata[valid_gene_indices].copy()
            print(f"Dataset {item}: Filtered to {len(valid_gene_indices)} spots with morphology data")
        else:
            print(f"Warning: No morphology data found for {item}, using all spots")
        
        all_adata.append(adata)

    # Combine all datasets for HVG/HEG selection
    print("Combining datasets for HVG/HEG selection...")
    union_genes, hvg_indices, heg_indices, sorted_hvg, sorted_heg, orig_shape, orig_n_vars = compute_hvg_heg_union(
        all_adata,
        top_k=top_k,
        flavor='seurat',
        min_cells=3,
    )

    print(f"Combined dataset shape: {orig_shape}")
    print(f"Number of common genes after inner join: {orig_n_vars}")
    print(f"HVG genes: {len(sorted_hvg)}, HEG genes: {len(sorted_heg)}")
    print(f"Union genes: {len(union_genes)}, Overlap: {len(sorted_hvg) + len(sorted_heg) - len(union_genes)}")

if args.imge_prc is not None:
    model_type = args.imge_prc.strip()
    model = None
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
    elif model_type == 'V2':
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform
        from timm.layers import SwiGLUPacked
        print("Load Virchow2 weights")
        model = timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
        image_processor = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
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

    if model_type != 'Raw':
        model.eval()

##########################
for item in tqdm(target_dataset):
    mol_path = datastem.joinpath(f'st/{item}.h5ad')
    mor_path = datastem.joinpath(f'patches/{item}.h5')
    json_path = datastem.joinpath(f'metadata/{item}.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    src_pixelsize = meta['pixel_size_um_estimated']
    dst_pixelsize = args.dst_pixelsize / 224
    patch_size_src = 224 * (dst_pixelsize / src_pixelsize)

    if not mol_path.exists():
        tqdm.write(f"Gene file not found: {mol_path}, skipping...")
        continue
    if not mor_path.exists():
        tqdm.write(f"Image file not found: {mor_path}, skipping...")
        continue

    adata = sc.read_h5ad(mol_path)
    
    # Standardize gene names and aggregate expression (Sum)
    adata = clean_var_names(adata, verbose=True)
    adata = aggregate_adata(adata, verbose=True)
    
    # Filter BLANK_ genes for Xenium datasets
    if is_xenium_dataset(dataset_name):
        adata = filter_blank_genes(adata, verbose=True)

    barcodes, coords, images = read_h5_data(mor_path)
    gene_barcodes = list(adata.obs.index)
    valid_gene_indices, valid_image_indices = create_barcode_mapping(gene_barcodes, barcodes)
    
    if args.gene_prc:
        # Check if we have any matching indices
        if len(valid_gene_indices) == 0:
            tqdm.write(f"Warning: No matching barcodes found for {item}, skipping gene processing...")
            continue
            
        if args.normalize:
            sc.pp.normalize_total(adata, target_sum=1e4)
        if args.log1p:
            sc.pp.log1p(adata)

        # Filter adata to matched spots
        adata = adata[valid_gene_indices].copy()
        tqdm.write(f"Dataset {item}: Matched {len(valid_gene_indices)} spots with morphology data")

        # Calculate total UMI counts for each spot
        if sp.issparse(adata.X):
            total_umi_counts = np.array(adata.X.sum(axis=1)).flatten().astype(np.float32)
        else:
            total_umi_counts = adata.X.sum(axis=1).astype(np.float32)

        # Extract union gene expression matrix
        if sp.issparse(adata.X):
            X_union = adata[:, union_genes].X.toarray().astype(np.float32)
        else:
            X_union = adata[:, union_genes].X.astype(np.float32)

        pause_if_nonfinite(X_union, "X_union", args.pause_on_overflow)
        pause_if_nonfinite(total_umi_counts, "total_umi_counts", args.pause_on_overflow)

        # Process coordinates using valid indices
        coords = coords[valid_image_indices]
        x_coords = coords[:, 0]
        y_coords = coords[:, 1]
        x_min, y_min = np.min(x_coords), np.min(y_coords)
        normalized_x = (x_coords - x_min) / patch_size_src
        normalized_y = (y_coords - y_min) / patch_size_src
        normalized_coords = np.column_stack((normalized_x, normalized_y)).astype(np.float32)

        # coords = coords[valid_image_indices].astype(np.float32)
        # coords = coords / patch_size_src
        # coords -= coords.min(axis=0, keepdims=True)
        # normalized_coords = torch.from_numpy(coords)
        
        remap_coords = map_to_integer_grid(coords, patch_size_src)
        coordinates = remap_coords

        # Save processed data
        gene_folder = dst_file / "gene"
        gene_folder.mkdir(parents=True, exist_ok=True)
        h5_file_path = gene_folder / f"{item}.h5"

        save_gene_h5(
            h5_file_path,
            mol_feats=X_union,
            cords=coordinates,
            float_cords=normalized_coords,
            total_umi_counts=total_umi_counts,
            orig_cords=coords,
            gene_names=union_genes,
            hvg_indices=hvg_indices,
            heg_indices=heg_indices,
        )

        check_list1 = copy.deepcopy(list(adata.obs.index))
        print("GENE extraction done")
        # print("\nUsage instructions:")
        # print("- Load 'union' dataset containing all union genes")
        # print("- Use 'hvg_indices' to select top-k HVG: union[:, hvg_indices[:k]]")
        # print("- Use 'heg_indices' to select top-k HEG: union[:, heg_indices[:k]]")
        # print("- 'union_gene_names' contains the gene names corresponding to union columns")
        # print("- Get HVG gene names: union_gene_names[hvg_indices[:k]]")
        # print("- Get HEG gene names: union_gene_names[heg_indices[:k]]")
###############
    if args.imge_prc is not None:
        # batch_size = 512
        batch_size = args.batch_size
        if model_type != 'Raw':
            model = model.cuda()
            model.eval()
        for item in tqdm(target_dataset):
            mor_path = datastem.joinpath(f'patches/{item}.h5')
            if not mor_path.exists():
                tqdm.write(f"File not found: {mor_path}, skipping...")
                continue
            barcodes, coords, images = read_h5_data(mor_path)

            ### Match
            mol_path = datastem.joinpath(f'st/{item}.h5ad')
            adata = sc.read_h5ad(mol_path)
            
            # Standardize and Aggregate
            adata = clean_var_names(adata, verbose=False)
            adata = aggregate_adata(adata, verbose=False)
            
            # Filter BLANK_ genes for Xenium datasets
            if is_xenium_dataset(dataset_name):
                adata = filter_blank_genes(adata, verbose=False)
                
            barcode_list = [b.item().decode() for b in barcodes]
            adata_barcodes = list(adata.obs.index)
            barcode_to_image_idx = {barcode: idx for idx, barcode in enumerate(barcode_list)}

            valid_image_indices = []
            for i, barcode in enumerate(adata_barcodes):
                if barcode in barcode_to_image_idx:
                    valid_image_indices.append(barcode_to_image_idx[barcode])
            
            # Check if we have any matching barcodes
            if len(valid_image_indices) == 0:
                tqdm.write(f"Warning: No matching barcodes found for {item}, skipping image processing...")
                continue
                
            images = images[valid_image_indices]
            coords = coords[valid_image_indices]
            barcodes = barcodes[valid_image_indices]
            barcode_list = [b.item().decode() for b in barcodes]
            tqdm.write(f"Dataset {item}: Processing {len(valid_image_indices)} images with matching barcodes")
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
                    elif model_type == 'V2':
                        processed = [image_processor(ensure_pil_image(img)) for img in batch_images]
                        inputs = torch.stack(processed).cuda()
                        outputs = model(inputs)
                        outputs = outputs[:, 0, :]
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

            save_mor_h5(h5_file_path, img_feats)
