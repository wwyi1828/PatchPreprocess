#!/usr/bin/env python3
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
from PIL import Image
import re
from utils_gene import *

# --- 核心配置 ---
E2S_JSON_PATH = Path('.ensembl_to_symbol_human.json')
DATA_ROOT = Path('/data/data/ST/HEST1K')
ST_DIR = DATA_ROOT / 'st'
PATCH_DIR = DATA_ROOT / 'patches'
JSON_PATH = Path('hest_human_visium_subtypes.json')

def format_folder_name(json_key):
    """Human_Brain_visium_MISC -> MISC_brain"""
    parts = json_key.split('_')
    if len(parts) < 4: return json_key.lower()
    return f"{parts[-1]}_{parts[1].lower()}"

def main():
    parser = argparse.ArgumentParser(description="Unified HEST Preprocessing Pipeline")
    parser.add_argument('--dst_file', default="/pool2/data/Mor2Mol_Final")
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--gene_prc', action='store_true')
    parser.add_argument('--imge_prc', type=str, default=None, help="UNI, PLIP, V2, R50, or Raw")
    parser.add_argument('--dst_pixelsize', type=float, default=55)
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--log1p', action='store_true')
    args = parser.parse_args()

    # 1. 资源准备
    if not JSON_PATH.exists(): raise FileNotFoundError(f"JSON config not found: {JSON_PATH}")
    organ_map = json.load(open(JSON_PATH))
    e2s = get_ensembl_to_symbol_mapping(organism="human", use_cache=True, cache_file=E2S_JSON_PATH)
    global_genes = compute_global_gene_intersection(
        organ_map, ST_DIR, mapping=e2s, clean_names=True, remove_internal=True
    )
    print(f"Global Universe established: {len(global_genes)} genes.")

    # 2. 模型加载逻辑
    model, image_processor = None, None
    if args.imge_prc and args.imge_prc.upper() != 'RAW':
        from transformers import AutoModelForZeroShotImageClassification, AutoProcessor, CLIPImageProcessor
        import timm
        m_type = args.imge_prc.upper()
        print(f"Loading {m_type} weights...")
        if m_type == 'UNI':
            model = timm.create_model("vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True)
            model.load_state_dict(torch.load("/data/checkpoints/UNI/pytorch_model.bin", map_location="cpu"))
            image_processor = CLIPImageProcessor(do_resize=False, do_center_crop=False, do_normalize=True)
        elif m_type == 'PLIP':
            model = AutoModelForZeroShotImageClassification.from_pretrained("vinid/plip")
            image_processor = AutoProcessor.from_pretrained("vinid/plip")
        elif m_type == 'V2':
            from timm.layers import SwiGLUPacked
            model = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
            from timm.data import resolve_data_config, create_transform
            image_processor = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        elif m_type == 'R50':
            from torchvision import models
            from utils import ModifiedResNet
            model = ModifiedResNet(models.resnet50(weights='ResNet50_Weights.DEFAULT'))
            image_processor = CLIPImageProcessor(do_resize=False, do_center_crop=False, do_normalize=True,
                                                 image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225])
        if model: model = model.cuda().eval()

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
            rank_adatas = []
            for ds_id in all_ids[:5]:
                p = ST_DIR / f"{ds_id}.h5ad"
                if not p.exists(): continue
                ad_tmp = sc.read_h5ad(p)
                ad_tmp = clean_var_names(ad_tmp, verbose=False)
                ad_tmp = convert_ensembl_to_symbol(ad_tmp, e2s)
                ad_tmp = aggregate_adata(ad_tmp)
                # Reindex to global universe
                X_df = pd.DataFrame(ad_tmp.X.toarray() if sp.issparse(ad_tmp.X) else ad_tmp.X, columns=ad_tmp.var_names)
                X_aligned = X_df.reindex(columns=global_genes, fill_value=0).values
                rank_adatas.append(ad.AnnData(X=X_aligned, obs=ad_tmp.obs, var=pd.DataFrame(index=global_genes)))
            
            if rank_adatas:
                combined = ad.concat(rank_adatas, join='inner')
                combined.obs_names_make_unique()
                # Ranking only: drop all-zero spots so HVG/HEG stats are not skewed (rows kept in downstream saving)
                sums = np.array(combined.X.sum(axis=1)).flatten()
                rank_base = combined[sums > 0, :].copy()
                sc.pp.normalize_total(rank_base, target_sum=1e4)
                sc.pp.log1p(rank_base)
                sc.pp.highly_variable_genes(rank_base, n_top_genes=2000, flavor='seurat')
                hvg_indices = np.argsort(rank_base.var['dispersions_norm'].values)[::-1]
                heg_indices = np.argsort(np.array(rank_base.X.mean(axis=0)).flatten())[::-1]

        # 4. 逐个 Slide 处理
        for ds_id in tqdm(all_ids):
            mol_path, mor_path = ST_DIR / f"{ds_id}.h5ad", PATCH_DIR / f"{ds_id}.h5"
            json_meta = DATA_ROOT / "metadata" / f"{ds_id}.json"
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
                    obs_names = ad_backed.obs_names.astype(str)
                    if obs_names.duplicated().any():
                        gene_barcodes = sorted(list(set(obs_names)))
                    else:
                        gene_barcodes = list(obs_names)
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
                with h5py.File(gene_out / f"{ds_id}.h5", 'w') as f:
                    f.create_dataset('mol_feats', data=adata.X.astype(np.float32))
                    f.create_dataset('cords', data=cords_int)
                    f.create_dataset('float_cords', data=float_cords.astype(np.float32))
                    f.create_dataset('total_umi_counts', data=np.array(adata.X.sum(axis=1)).flatten().astype(np.float32))
                    f.create_dataset('orig_cords', data=matched_coords)
                    f.create_dataset('union_gene_names', data=np.array(global_genes, dtype='S'))
                    f.create_dataset('hvg_indices', data=hvg_indices)
                    f.create_dataset('heg_indices', data=heg_indices)

            if args.imge_prc:
                img_out = group_dst / f"imge_{args.imge_prc.upper()}"
                img_out.mkdir(exist_ok=True)
                m_images = images[v_img]
                feats = []
                for i in range(0, len(m_images), args.batch_size):
                    batch = m_images[i:i+args.batch_size]
                    if args.imge_prc.upper() == 'RAW':
                        feats.append(torch.tensor(batch))
                    else:
                        inputs = image_processor(images=list(batch), return_tensors="pt")['pixel_values'].cuda()
                        with torch.no_grad():
                            out = model(inputs)
                            if hasattr(out, 'pooler_output'): out = out.pooler_output
                            if len(out.shape) > 2: out = out[:, 0, :]
                            feats.append(out.cpu())
                with h5py.File(img_out / f"{ds_id}.h5", 'w') as f:
                    f.create_dataset('mor_feats', data=torch.cat(feats).numpy())

if __name__ == "__main__":
    main()
