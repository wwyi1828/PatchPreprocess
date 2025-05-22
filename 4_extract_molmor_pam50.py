import h5py
import torch
import copy
import pickle
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
parser.add_argument('--batch_size', type=int, default=512, 
                    help='Batch size for preprocessing.')
parser.add_argument('--gene_prc', action='store_true')
parser.add_argument('--imge_prc', type=str, default=None)
args = parser.parse_args()


dataset_name = 'SPA_beast_2'
target_dataset = [f'SPA{idx}' for idx in range(51,155)][68:] # All 20x


datastem = Path(f"/data/data/ST/HEST1K/")
dst_file = Path(f"/pool2/data/Mor2Mol/{dataset_name}")
dst_file.mkdir(parents=True, exist_ok=True)

pam50_genes = ['ACTR3B', 'ANLN', 'BAG1', 'BCL2', 'BIRC5', 'BLVRA', 'CCNB1', 
               'CCNE1', 'CDC20', 'CDC6', 'NUF2', 'CDH3', 'CENPF', 'CEP55', 
               'CXXC5', 'EGFR', 'ERBB2', 'ESR1', 'EXO1', 'FGFR4', 'FOXA1', 
               'FOXC1', 'GPR160', 'GRB7', 'KIF2C', 'NDC80', 'KRT14', 'KRT17', 
               'KRT5', 'MAPT', 'MDM2', 'MELK', 'MIA', 'MKI67', 'MLPH', 'MMP11', 
               'MYBL2', 'MYC', 'NAT1', 'ORC6', 'PGR', 'PHGDH', 'PTTG1', 'RRM2', 
               'SFRP1', 'SLC39A6', 'TMEM45B', 'TYMS', 'UBE2C', 'UBE2T']


if args.gene_prc:
    all_adata = []
    for item in target_dataset:
        mol_path = datastem.joinpath(f'st/{item}.h5ad')
        if not mol_path.exists():
            print(f"File not found: {mol_path}, skipping...")
            continue
        adata = sc.read_h5ad(mol_path)
        adata = adata[:, ~adata.var_names.str.startswith('__ambiguous')]
        missing_genes = [gene for gene in pam50_genes if gene not in adata.var_names]
        print(missing_genes)
        if missing_genes:
            zero_df = pd.DataFrame(
                0, 
                index=adata.obs_names, 
                columns=missing_genes
            )
            adata = ad.concat([adata, ad.AnnData(zero_df)], axis=1)

        var_names_clean = adata.var_names#.str.replace(r"\.\d+$", "", regex=True)
        missing_genes = [gene for gene in pam50_genes if gene not in var_names_clean]
        print(f"adata {item}: Missing {len(missing_genes)} genes -> {missing_genes}")

        sc.pp.log1p(adata)
        gene_to_idx = {gene: idx for idx, gene in enumerate(adata.var_names)}
        mg_idx = [gene_to_idx[gene] for gene in pam50_genes if gene in gene_to_idx]
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
            X_mg = adata[:, mg_idx].X.toarray().astype(np.float16)
        else:
            X_mg = adata[:, mg_idx].X.astype(np.float16)
        X_mg = torch.from_numpy(X_mg).float()

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

        gene_folder = dst_file / "gene_mg"
        gene_folder.mkdir(parents=True, exist_ok=True)
        h5_file_path = gene_folder / f"{item}.h5"

        with h5py.File(str(h5_file_path), 'w') as f:
            f.create_dataset('cords', data=coordinates)
            f.create_dataset('float_cords', data=normalized_coords)
            f.create_dataset('heg', data=X_mg)
            f.create_dataset('hvg', data=X_mg)

    print("GENE extraction done")
