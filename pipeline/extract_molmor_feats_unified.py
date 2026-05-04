#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import h5py
import torch
import scanpy as sc
import numpy as np
import scipy.sparse as sp
import anndata as ad
import pandas as pd
from tqdm import tqdm
import os
import pickle
import json
from pathlib import Path
import argparse
import re
from core.gene_utils import *
from core.image_encoder import load_image_encoder, extract_h5_features_in_batches

def format_folder_name(json_key):
    """Human_Brain_visium_MISC -> MISC_brain"""
    parts = json_key.split('_')
    if len(parts) < 4: return json_key.lower()
    return f"{parts[-1]}_{parts[1].lower()}"

def main():
    parser = argparse.ArgumentParser(description="Unified HEST Preprocessing Pipeline")
    parser.add_argument('--data_root', required=True, help="HEST-style root containing st/, patches/, and metadata/.")
    parser.add_argument('--subtype_json', required=True, help="JSON mapping organ/subtype groups to sample IDs.")
    parser.add_argument('--dst_file', required=True, help="Output root directory.")
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--gene_prc', action='store_true')
    parser.add_argument('--imge_prc', type=str, default=None, help="UNI, PLIP, V2, R50, or Raw")
    parser.add_argument('--dst_pixelsize', type=float, default=55)
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--log1p', action='store_true')
    parser.add_argument('--ensembl_cache', default='.ensembl_to_symbol_human.json',
                        help="Cache path for Ensembl-to-symbol mapping.")
    parser.add_argument('--uni_ckpt_dir', default=os.environ.get("UNI_CKPT_DIR"),
                        help="Directory containing UNI pytorch_model.bin. Can also be set with UNI_CKPT_DIR.")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    st_dir = data_root / 'st'
    patch_dir = data_root / 'patches'
    json_path = Path(args.subtype_json)
    e2s_json_path = Path(args.ensembl_cache)

    # 1. 资源准备
    if not json_path.exists():
        raise FileNotFoundError(f"JSON config not found: {json_path}")
    organ_map = json.load(open(json_path))
    e2s = get_ensembl_to_symbol_mapping(organism="human", use_cache=True, cache_file=e2s_json_path)
    global_genes = compute_global_gene_intersection(
        organ_map, st_dir, mapping=e2s, clean_names=True, remove_internal=True
    )
    print(f"Global Universe established: {len(global_genes)} genes.")

    # 2. 模型加载逻辑
    model, image_processor = None, None
    model_type = None
    if args.imge_prc:
        model_type, model, image_processor = load_image_encoder(
            args.imge_prc,
            input_source="h5",
            uni_ckpt_dir=args.uni_ckpt_dir,
        )

    # 3. 分组循环
    for group_key, subtypes in organ_map.items():
        folder_name = format_folder_name(group_key)
        group_dst = Path(args.dst_file) / folder_name
        group_dst.mkdir(parents=True, exist_ok=True)

        all_ids = []
        for ids in subtypes.values(): all_ids.extend(ids)
        print(f"\n>>> Processing {folder_name} ({len(all_ids)} slides)")

        # 采样计算该组的 HVG/HEG 排名 (基于对齐后的矩阵)
        hvg_indices, heg_indices = None, None
        if args.gene_prc:
            print("Calculating group rankings...")
            rank_adatas = build_rank_adatas_unified(
                all_ids,
                ST_DIR,
                PATCH_DIR,
                global_genes=global_genes,
                e2s=e2s,
            )

            if rank_adatas:
                hvg_indices, heg_indices = compute_hvg_heg_rankings_unified(
                    rank_adatas, top_k=2000, flavor="seurat"
                )
            else:
                hvg_indices = np.array([], dtype=np.int64)
                heg_indices = np.array([], dtype=np.int64)

        # 4. 逐个 Slide 处理
        for ds_id in tqdm(all_ids):
            mol_path, mor_path = st_dir / f"{ds_id}.h5ad", patch_dir / f"{ds_id}.h5"
            json_meta = data_root / "metadata" / f"{ds_id}.json"
            if not (mol_path.exists() and mor_path.exists()): continue

            # A. 基因标准化与对齐
            gene_barcodes = []
            adata = None

            if args.gene_prc:
                adata = sc.read_h5ad(mol_path)
                adata = clean_var_names(adata, verbose=False)
                adata = convert_ensembl_to_symbol(adata, e2s)
                adata = aggregate_adata(adata)

                # 物理对齐
                X_df = pd.DataFrame(adata.X.toarray() if sp.issparse(adata.X) else adata.X, columns=adata.var_names)
                X_aligned = X_df.reindex(columns=global_genes, fill_value=0).values
                adata = ad.AnnData(X=X_aligned, obs=adata.obs, var=pd.DataFrame(index=global_genes))

                if args.normalize: sc.pp.normalize_total(adata, target_sum=1e4)
                if args.log1p: sc.pp.log1p(adata)
                gene_barcodes = list(adata.obs.index)
            else:
                # 仅读取 Barcode，不加载矩阵，极大加速
                # 必须复现 aggregate_adata 中的去重逻辑以保证对齐一致性
                # utils_gene.aggregate_adata 逻辑：如有重复索引，groupby(level=0).sum() -> 导致排序
                try:
                    ad_backed = sc.read_h5ad(mol_path, backed='r')
                    try:
                        obs_names = ad_backed.obs_names.astype(str)
                        if obs_names.duplicated().any():
                            gene_barcodes = sorted(list(set(obs_names)))
                        else:
                            gene_barcodes = list(obs_names)
                    finally:
                        file_handle = getattr(ad_backed, "file", None)
                        if file_handle is not None:
                            file_handle.close()
                except Exception as e:
                    # Fallback (rare case)
                    try:
                        ad_tmp = sc.read_h5ad(mol_path)
                        obs_names = ad_tmp.obs_names.astype(str)
                        if obs_names.duplicated().any():
                            gene_barcodes = sorted(list(set(obs_names)))
                        else:
                            gene_barcodes = list(obs_names)
                    except:
                        continue

            # B. 匹配 Barcode & 坐标计算
            barcodes, coords, images = read_h5_data(mor_path)
            v_gene, v_img = create_barcode_mapping(gene_barcodes, barcodes)
            if not v_gene: continue

            if adata is not None:
                adata = adata[v_gene].copy()
            matched_coords = coords[v_img]
            meta = json.load(open(json_meta))
            src_pixelsize = meta['pixel_size_um_estimated']
            patch_size_src = 224 * ((args.dst_pixelsize/224) / src_pixelsize)

            cords_int = map_to_integer_grid(matched_coords, patch_size_src)
            x_c, y_c = matched_coords[:, 0], matched_coords[:, 1]
            float_cords = np.column_stack(((x_c - x_c.min())/patch_size_src, (y_c - y_c.min())/patch_size_src))

            # C. 保存特征
            if args.gene_prc:
                gene_out = group_dst / "gene"
                gene_out.mkdir(exist_ok=True)
                mol_feats = adata.X.toarray() if sp.issparse(adata.X) else adata.X
                total_umi_counts = np.array(adata.X.sum(axis=1)).flatten().astype(np.float32)
                hvg_to_save = hvg_indices if hvg_indices is not None else np.array([], dtype=np.int64)
                heg_to_save = heg_indices if heg_indices is not None else np.array([], dtype=np.int64)
                save_gene_h5(
                    gene_out / f"{ds_id}.h5",
                    mol_feats=mol_feats,
                    cords=cords_int,
                    float_cords=float_cords.astype(np.float32),
                    total_umi_counts=total_umi_counts,
                    orig_cords=matched_coords,
                    gene_names=global_genes,
                    hvg_indices=hvg_to_save,
                    heg_indices=heg_to_save,
                )

            if args.imge_prc:
                img_out = group_dst / f"imge_{model_type}"
                img_out.mkdir(exist_ok=True)
                m_images = images[v_img]
                feats = extract_h5_features_in_batches(
                    m_images,
                    model_type=model_type,
                    model=model,
                    image_processor=image_processor,
                    batch_size=args.batch_size,
                )
                save_mor_h5(img_out / f"{ds_id}.h5", feats.numpy())

if __name__ == "__main__":
    main()
