import h5py
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
import pickle
import timm
import json
from pathlib import Path
import argparse
from PIL import Image
from utils_gene import *

def read_h5_data(file_path):
    with h5py.File(file_path, 'r') as f:
        barcodes = f['barcode'][()]
        coords = f['coords'][()]
        images = f['img'][()]
        return barcodes, coords, images

def create_barcode_mapping(gene_barcodes, image_barcodes):
    image_barcode_list = [b.item().decode() if hasattr(b, 'item') else str(b) for b in image_barcodes]
    barcode_to_image_idx = {barcode: idx for idx, barcode in enumerate(image_barcode_list)}
    
    valid_gene_indices = []
    valid_image_indices = []
    
    for i, barcode in enumerate(gene_barcodes):
        if barcode in barcode_to_image_idx:
            valid_gene_indices.append(i)
            valid_image_indices.append(barcode_to_image_idx[barcode])
    return valid_gene_indices, valid_image_indices

def create_ensembl_mapping(all_ensembl_ids, save_path, batch_size=100):
    """
    Create Ensembl ID to Gene Symbol mapping in batches and save to file
    """
    dataset = pbm.Dataset(name='hsapiens_gene_ensembl', host='http://www.ensembl.org')
    
    all_mappings = {}
    unique_ids = list(set(all_ensembl_ids))  # Remove duplicates
    
    print(f"Creating mapping for {len(unique_ids)} unique Ensembl IDs...")
    
    for i in range(0, len(unique_ids), batch_size):
        batch_ids = unique_ids[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(unique_ids)-1)//batch_size + 1} ({len(batch_ids)} IDs)")
        
        try:
            results = dataset.query(
                attributes=['ensembl_gene_id', 'hgnc_symbol', 'description'],
                filters={'link_ensembl_gene_id': batch_ids}
            )
            
            if i == 0:
                print(f"Query result columns: {results.columns.tolist()}")
            for _, row in results.iterrows():
                if pd.notna(row['HGNC symbol']) and row['HGNC symbol'] != '':
                    all_mappings[row['Gene stable ID']] = row['HGNC symbol']
            
            print(f"  Added {len(results)} mappings")
            
        except Exception as e:
            print(f"  Error in batch {i//batch_size + 1}: {e}")
            continue
    
    # Save mapping to file
    with open(save_path, 'wb') as f:
        pickle.dump(all_mappings, f)
    
    print(f"Saved {len(all_mappings)} mappings to {save_path}")
    return all_mappings

def convert_gene_names_to_symbols(adata, id_to_symbol):
    """Convert Ensembl IDs to Gene Symbols for an AnnData object"""
    new_var_names = []
    for ensembl_id in adata.var_names:
        if ensembl_id in id_to_symbol:
            new_var_names.append(id_to_symbol[ensembl_id])
        else:
            new_var_names.append(ensembl_id)
    
    adata.var_names = new_var_names
    converted_count = len([x for x in new_var_names if x in id_to_symbol.values()])
    return adata, converted_count

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
parser.add_argument('--dataset', type=str, default='SPA_breast')
parser.add_argument('--batch_size', type=int, default=512, 
                    help='Batch size for preprocessing.')
parser.add_argument('--gene_prc', action='store_true')
parser.add_argument('--imge_prc', type=str, default=None)
parser.add_argument('--dst_pixelsize', type=float, default=100)
parser.add_argument('--dst_file', default="/pool2/data/Mor2Mol")
parser.add_argument('--normalize', action='store_true', 
                    help='Whether to apply normalize_total in individual dataset processing')
parser.add_argument('--log1p', action='store_true', 
                    help='Whether to apply log1p in individual dataset processing')
parser.add_argument('--pause_on_overflow', action='store_false',
                    help='Pause and wait for input if overflow/NaN/Inf detected')
args = parser.parse_args()
dst_file = Path(f"{args.dst_file}/{args.dataset}")

dataset_name = 'SPA_breast'
# dataset_name = args.dataset
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

datastem = Path(f"/data/data/ST/HEST1K/")
target_dataset_1 = [f'SPA{idx}' for idx in range(51,155)][0:68] # Ensemble
target_dataset_2 = [f'SPA{idx}' for idx in range(51,155)][68:] # Symbol
dst_file.mkdir(parents=True, exist_ok=True)


# Load or create Ensembl to Symbol mapping
mapping_file = datastem / "ensembl_to_symbol_mapping.pkl"
if os.path.exists(mapping_file):
    print("Loading existing Ensembl to Symbol mapping...")
    with open(mapping_file, 'rb') as f:
        id_to_symbol = pickle.load(f)
    print(f"Loaded {len(id_to_symbol)} mappings")
else:
    print("Creating new Ensembl to Symbol mapping...")
    all_ensembl_ids = set()
    for item in target_dataset_1:
        mol_path = datastem.joinpath(f'st/{item}.h5ad')
        adata = sc.read_h5ad(mol_path)
        adata = adata[:, ~adata.var_names.str.startswith('__ambiguous')]
        all_ensembl_ids.update(adata.var_names.tolist())
    id_to_symbol = create_ensembl_mapping(list(all_ensembl_ids), mapping_file)

top_k = 2000

if args.gene_prc:
    # Step 1: Process all datasets and combine to find HVG/HEG
    print("Step 1: Processing datasets to find HVG/HEG...")
    verbose = True
    all_adata_1 = []
    all_adata_2 = []
    
    # Process dataset_1 (Ensembl IDs -> Gene Symbols)
    print("Processing target_dataset_1 (Ensembl -> Symbol conversion)...")
    for item in target_dataset_1:
        mol_path = datastem.joinpath(f'st/{item}.h5ad')
        adata = sc.read_h5ad(mol_path)
        
        # 1. Standardize IDs (remove versions) then Map
        adata = clean_var_names(adata, verbose=False)
        adata = convert_ensembl_to_symbol(adata, id_to_symbol)
        
        # 2. Aggregate Duplicates (Sum)
        adata = aggregate_adata(adata, verbose=verbose)
        
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
        
        all_adata_1.append(adata)

    # Process dataset_2 (already has Gene Symbols)
    print("Processing target_dataset_2 (already Gene Symbols)...")
    for item in target_dataset_2:
        mol_path = datastem.joinpath(f'st/{item}.h5ad')
        adata = sc.read_h5ad(mol_path)
        
        # Standardize and Aggregate
        adata = clean_var_names(adata, verbose=verbose)
        adata = aggregate_adata(adata, verbose=verbose)
        
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
        
        all_adata_2.append(adata)

    # Combine all datasets for HVG/HEG selection
    print("Combining datasets for HVG/HEG selection...")
    all_adata = all_adata_1 + all_adata_2
    combined_adata = ad.concat(all_adata, join='inner')
    combined_adata.obs_names_make_unique()

    print(f"Combined dataset shape: {combined_adata.shape}")
    print(f"Number of common genes after inner join: {combined_adata.n_vars}")

    sc.pp.filter_genes(combined_adata, min_cells=3)
    # n_before = combined_adata.shape[0]
    # mask = (combined_adata.X.sum(axis=1) > 0)
    # combined_adata = combined_adata[mask, :].copy()
    # n_after = combined_adata.shape[0]
    # print(f"Removed {n_before - n_after} spots/barcodes with zero expression.")

    # Apply full preprocessing for HVG/HEG selection
    sc.pp.normalize_total(combined_adata, target_sum=1e4)
    sc.pp.log1p(combined_adata)

    # Find HVG
    sc.pp.highly_variable_genes(
        combined_adata,
        n_top_genes=top_k,
        flavor='seurat',
        subset=False
    )
    hvg_mask = combined_adata.var['highly_variable']
    sorted_hvg = combined_adata.var_names[hvg_mask][
        np.argsort(combined_adata.var.loc[hvg_mask, 'dispersions_norm'].values)[::-1]
    ]

    # Find HEG
    means = np.array(combined_adata.X.mean(axis=0)).flatten()
    combined_adata.var['means'] = means
    sorted_heg = combined_adata.var_names[np.argsort(means)[::-1][:top_k]]

    # Create union genes and indices
    union_genes = []
    gene_set = set()
    for gene in sorted_hvg:
        if gene not in gene_set:
            union_genes.append(gene)
            gene_set.add(gene)
    for gene in sorted_heg:
        if gene not in gene_set:
            union_genes.append(gene)
            gene_set.add(gene)
    
    gene_to_idx = {gene: idx for idx, gene in enumerate(union_genes)}
    hvg_indices = [gene_to_idx[gene] for gene in sorted_hvg]
    heg_indices = [gene_to_idx[gene] for gene in sorted_heg]

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
    elif model_type == 'Raw':
        pass
    else:
        raise ValueError(f"Unsupported image processor type: {model_type}")


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
    
    # 1. Clean and Map (if applicable)
    adata = clean_var_names(adata, verbose=True)
    if item in target_dataset_1:
        adata = convert_ensembl_to_symbol(adata, id_to_symbol)
    
    # 2. Aggregate (Sum)
    adata = aggregate_adata(adata, verbose=True)

    barcodes, coords, images = read_h5_data(mor_path)
    gene_barcodes = list(adata.obs.index)
    valid_gene_indices, valid_image_indices = create_barcode_mapping(gene_barcodes, barcodes)
    
    if args.gene_prc:

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
