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

def create_barcode_mapping(gene_barcodes, image_barcodes):
    """
    创建基因和图像数据之间的索引映射
    返回匹配的索引对
    """
    # 将image barcodes转换为字符串并创建映射
    image_barcode_list = [b.item().decode() if hasattr(b, 'item') else str(b) for b in image_barcodes]
    barcode_to_image_idx = {barcode: idx for idx, barcode in enumerate(image_barcode_list)}
    
    # 找到匹配的索引
    valid_gene_indices = []
    valid_image_indices = []
    
    for i, barcode in enumerate(gene_barcodes):
        if barcode in barcode_to_image_idx:
            valid_gene_indices.append(i)
            valid_image_indices.append(barcode_to_image_idx[barcode])
    
    return valid_gene_indices, valid_image_indices

def process_coordinates(coords, valid_indices):
    """处理坐标数据"""
    coords_subset = coords[valid_indices]
    
    # 标准化坐标
    x_coords = coords_subset[:, 0]
    y_coords = coords_subset[:, 1]
    x_min, y_min = np.min(x_coords), np.min(y_coords)
    normalized_x = (x_coords - x_min) / 224
    normalized_y = (y_coords - y_min) / 224
    normalized_coords = torch.from_numpy(np.column_stack((normalized_x, normalized_y))).float()
    
    # 重映射坐标
    remap_coords = map_to_integer_grid(coords)
    coordinates = remap_coords[valid_indices]
    
    return coordinates, normalized_coords

def extract_image_features(images, model, image_processor, model_type, batch_size):
    """提取图像特征"""
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
    
    return torch.concat(features_list)

parser = argparse.ArgumentParser(description='Read Configuration File')
parser.add_argument('--dataset', type=str, default='NCBI_SKIN')
parser.add_argument('--batch_size', type=int, default=512, 
                    help='Batch size for preprocessing.')
parser.add_argument('--gene_prc', action='store_true')
parser.add_argument('--imge_prc', type=str, default=None)
parser.add_argument('--dst_file', default="/pool2/data/Mor2Mol")
args = parser.parse_args()

dst_file = Path(f"{args.dst_file}/{args.dataset}")
datastem = Path(f"/data/data/ST/HEST1K/")
dst_file.mkdir(parents=True, exist_ok=True)
top_k = 10000

# 数据集配置
dataset_name = args.dataset
if dataset_name == 'MISC_brain':
    target_dataset = [f'MISC{idx}' for idx in range(1,13)]
elif dataset_name == 'MISC_lung':
    target_dataset = [f'MISC{idx}' for idx in range(13,33)]
elif dataset_name == 'MISC_bowel':
    target_dataset = [f'MISC{idx}' for idx in range(33,74)]
elif dataset_name == 'INT_Lymphnode':
    target_dataset = [f'INT{idx}' for idx in range(1,25)]
elif dataset_name == 'Xenium_Lung':
    target_dataset = [f'NCBI{idx}' for idx in range(856,885)]
elif dataset_name == 'ColonMap':
    target_dataset = [f'NCBI{idx}' for idx in range(33,74)]
elif dataset_name == 'NCBI_SKIN':
    target_dataset = [f'NCBI{idx}' for idx in range(469,523)]
elif dataset_name == 'NCBI_brain':
    target_dataset = [f'NCBI{idx}' for idx in range(336,411)]
elif dataset_name == 'NCBI_Spinal':
    target_dataset = [f'NCBI{idx}' for idx in range(11,336)]
elif dataset_name == 'SPA_breast_1':
    target_dataset = [f'SPA{idx}' for idx in range(51,155)][0:68]
elif dataset_name == 'SPA_breast_2':
    target_dataset = [f'SPA{idx}' for idx in range(51,155)][68:]
elif dataset_name == 'SPA_breast':
    target_dataset = [f'SPA{idx}' for idx in range(51,155)]
elif dataset_name == 'TENX_Xenium_breast':
    target_dataset = [f'TENX{idx}' for idx in range(94, 100)]

# 如果需要处理基因数据，先获取顶级基因
if args.gene_prc:
    # Option 1: Use original method (current)
    sorted_hvg, sorted_heg = dataset_topgenes(datastem, target_dataset, top_k=top_k, verbose=True)
    
    # Option 2: Use modern method (recommended - uncomment to switch)
    # from utils_gene_optimized import dataset_topgenes_modern
    # sorted_hvg, sorted_heg = dataset_topgenes_modern(
    #     datastem, target_dataset, top_k=top_k, hvg_method='seurat_v3', verbose=True
    # )

# 如果需要处理图像数据，加载模型
model = None
image_processor = None
if args.imge_prc is not None:
    model_type = args.imge_prc
    if model_type == 'PLIP':
        print("Load PLIP weights")
        pipe = pipeline("zero-shot-image-classification", model="vinid/plip")
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
    
    if model_type != 'Raw' and model is not None:
        model = model.cuda()

# 主要处理循环 - 统一处理每个数据集
# 用于最后验证匹配结果的列表
final_gene_barcodes = []
final_image_barcodes = []
# 用于记录每个数据集的匹配信息
dataset_match_info = []

for item in tqdm(target_dataset):
    # 检查文件是否存在
    mol_path = datastem.joinpath(f'st/{item}.h5ad')
    mor_path = datastem.joinpath(f'patches/{item}.h5')
    
    if not mol_path.exists():
        tqdm.write(f"Gene file not found: {mol_path}, skipping...")
        continue
    if not mor_path.exists():
        tqdm.write(f"Image file not found: {mor_path}, skipping...")
        continue
    
    # 加载数据（一次性加载）
    adata = sc.read_h5ad(mol_path)
    adata = adata[:, ~adata.var_names.str.startswith('__ambiguous')]
    adata = avg_duplicate_geneexpr(adata)
    adata = avg_duplicate_cells(adata)
    
    barcodes, coords, images = read_h5_data(mor_path)
    
    # 创建索引映射（核心优化点）
    gene_barcodes = list(adata.obs.index)
    valid_gene_indices, valid_image_indices = create_barcode_mapping(gene_barcodes, barcodes)
    
    if len(valid_gene_indices) == 0:
        tqdm.write(f"No matching barcodes found for {item}, skipping...")
        continue
    
    # 筛选匹配的数据
    adata_matched = adata[valid_gene_indices].copy()
    images_matched = images[valid_image_indices]
    coords_matched = coords[valid_image_indices]
    barcodes_matched = barcodes[valid_image_indices]
    
    # 收集匹配的barcodes用于最后验证
    matched_gene_barcodes = [gene_barcodes[i] for i in valid_gene_indices]
    matched_image_barcodes = [barcodes[i].item().decode() if hasattr(barcodes[i], 'item') else str(barcodes[i]) for i in valid_image_indices]
    
    final_gene_barcodes.extend(matched_gene_barcodes)
    final_image_barcodes.extend(matched_image_barcodes)
    
    # 记录每个数据集的匹配信息
    total_gene_spots = len(gene_barcodes)
    total_image_patches = len(barcodes)
    matched_count = len(matched_gene_barcodes)
    match_rate = matched_count / max(total_gene_spots, total_image_patches) * 100 if max(total_gene_spots, total_image_patches) > 0 else 0
    
    dataset_match_info.append({
        'dataset': item,
        'total_gene_spots': total_gene_spots,
        'total_image_patches': total_image_patches,
        'matched_count': matched_count,
        'match_rate': match_rate
    })
    
    # 处理基因数据
    if args.gene_prc:
        sc.pp.log1p(adata_matched)
        
        if sp.issparse(adata_matched.X):
            X_hvg = adata_matched[:, sorted_hvg].X.toarray().astype(np.float16)
            X_heg = adata_matched[:, sorted_heg].X.toarray().astype(np.float16)
        else:
            X_hvg = adata_matched[:, sorted_hvg].X.astype(np.float16)
            X_heg = adata_matched[:, sorted_heg].X.astype(np.float16)
        
        X_heg = torch.from_numpy(X_heg).float()
        X_hvg = torch.from_numpy(X_hvg).float()
        
        # 处理坐标
        coordinates, normalized_coords = process_coordinates(coords, valid_image_indices)
        
        # 保存基因数据
        gene_folder = dst_file / "gene"
        gene_folder.mkdir(parents=True, exist_ok=True)
        h5_file_path = gene_folder / f"{item}.h5"

        with h5py.File(str(h5_file_path), 'w') as f:
            f.create_dataset('cords', data=coordinates)
            f.create_dataset('float_cords', data=normalized_coords)
            f.create_dataset('hvg', data=X_hvg)
            f.create_dataset('heg', data=X_heg)
            hvg_gene_names = np.array(sorted_hvg, dtype='S')
            heg_gene_names = np.array(sorted_heg, dtype='S')
            f.create_dataset('hvg_gene_names', data=hvg_gene_names)
            f.create_dataset('heg_gene_names', data=heg_gene_names)
    
    # 处理图像数据
    if args.imge_prc is not None and model is not None:
        img_feats = extract_image_features(
            images_matched, model, image_processor, model_type, args.batch_size
        )
        
        # 保存图像特征
        img_folder = dst_file / f"imge_{model_type}"
        img_folder.mkdir(parents=True, exist_ok=True)
        h5_file_path = img_folder / f"{item}.h5"

        with h5py.File(str(h5_file_path), 'w') as f:
            f.create_dataset('img_feats', data=img_feats)

if args.gene_prc:
    print("GENE extraction done")
if args.imge_prc is not None:
    print("IMAGE extraction done")

# Validation of Matching Results
print("\n" + "="*70)
print("Matching Results Validation")
print("="*70)
# Display matching info for each dataset
if dataset_match_info:
    print("\nDetails of Each Dataset Matching:")
    print("-" * 70)
    print(f"{'Dataset':<15} {'Gene Spots':<10} {'Image Patches':<12} {'Matched':<10} {'Match Rate':<10}")
    print("-" * 70)
    
    for info in dataset_match_info:
        status = "✅" if info['match_rate'] >= 90 else "⚠️" if info['match_rate'] >= 70 else "❌"
        print(f"{info['dataset']:<15} {info['total_gene_spots']:<10} {info['total_image_patches']:<12} {info['matched_count']:<10} {info['match_rate']:.1f}% {status}")
    
    print("-" * 70)
    
    # Summary statistics
    total_processed = len(dataset_match_info)
    high_match = sum(1 for info in dataset_match_info if info['match_rate'] >= 90)
    med_match = sum(1 for info in dataset_match_info if 70 <= info['match_rate'] < 90)
    low_match = sum(1 for info in dataset_match_info if info['match_rate'] < 70)
    
    print(f"\nSummary:")
    print(f"  Total datasets processed: {total_processed}")
    print(f"  High match rate (≥90%): {high_match}")
    print(f"  Medium match rate (70-90%): {med_match}")
    print(f"  Low match rate (<70%): {low_match}")

print("\nOverall Matching Validation:")

if len(final_gene_barcodes) > 0 and len(final_image_barcodes) > 0:
    print(f"Total gene barcodes: {len(final_gene_barcodes)}")
    print(f"Total image barcodes: {len(final_image_barcodes)}")
    
    if len(final_gene_barcodes) == len(final_image_barcodes):
        are_identical = sorted(final_gene_barcodes) == sorted(final_image_barcodes)
        print(f"Are the sorted lists identical: {are_identical}")
        
        set_equal = set(final_gene_barcodes) == set(final_image_barcodes)
        print(f"Are the sets identical: {set_equal}")
        
        if set_equal and not are_identical:
            print("Both lists contain the same elements, but the order is different.")
        elif are_identical:
            print("✅ Perfect match: All barcodes are identical and in the same order.")
        elif set_equal:
            print("✅ Successful match: All barcodes are identical, but order differs.")
        else:
            print("⚠️ Partial match: Same length but contents are not identical.")
    else:
        common = set(final_gene_barcodes) & set(final_image_barcodes)
        print(f"Number of common barcodes: {len(common)}")
        print(f"Gene-only barcodes: {len(set(final_gene_barcodes) - set(final_image_barcodes))}")
        print(f"Image-only barcodes: {len(set(final_image_barcodes) - set(final_gene_barcodes))}")
        
        if len(common) > 0:
            match_rate = len(common) / max(len(final_gene_barcodes), len(final_image_barcodes)) * 100
            print(f"Match rate: {match_rate:.2f}%")
            if match_rate >= 90:
                print("✅ High match rate: Good alignment.")
            elif match_rate >= 70:
                print("⚠️ Medium match rate: Please check your data.")
            else:
                print("❌ Low match rate: Data alignment issue.")
        else:
            print("❌ No match: Gene and image data are completely unmatched.")
else:
    if len(final_gene_barcodes) == 0 and len(final_image_barcodes) == 0:
        print("⚠️ No data processed.")
    elif len(final_gene_barcodes) == 0:
        print("⚠️ No gene data processed.")
    elif len(final_image_barcodes) == 0:
        print("⚠️ No image data processed.")

print("="*70)
