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
from utils_gene_optimized import *

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

def process_coordinates(coords, valid_indices):
    coords_subset = coords[valid_indices]
    
    x_coords = coords_subset[:, 0]
    y_coords = coords_subset[:, 1]
    x_min, y_min = np.min(x_coords), np.min(y_coords)
    normalized_x = (x_coords - x_min) / 224
    normalized_y = (y_coords - y_min) / 224
    normalized_coords = torch.from_numpy(np.column_stack((normalized_x, normalized_y))).float()
    
    remap_coords = map_to_integer_grid(coords)
    coordinates = remap_coords[valid_indices]
    
    return coordinates, normalized_coords

def extract_image_features(images, model, image_processor, model_type, batch_size):
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

parser = argparse.ArgumentParser(description='Modern Spatial Transcriptomics Processing Pipeline')
parser.add_argument('--dataset', type=str, default='NCBI_SKIN', 
                    help='Dataset to process')
parser.add_argument('--batch_size', type=int, default=512, 
                    help='Batch size for preprocessing')
parser.add_argument('--gene_prc', action='store_true',
                    help='Enable gene processing with modern pipeline')
parser.add_argument('--imge_prc', type=str, default=None,
                    help='Image processing model type (PLIP, UNI, R50, Raw)')
parser.add_argument('--dst_file', default="/pool2/data/Mor2Mol",
                    help='Destination directory for processed data')

# Modern preprocessing parameters
parser.add_argument('--hvg_method', type=str, default='seurat_v3',
                    choices=['seurat', 'seurat_v3', 'cell_ranger'],
                    help='Method for highly variable gene selection')
parser.add_argument('--min_genes', type=int, default=200,
                    help='Minimum genes per cell')
parser.add_argument('--min_cells', type=int, default=3,
                    help='Minimum cells per gene')
parser.add_argument('--max_genes', type=int, default=5000,
                    help='Maximum genes per cell (doublet detection)')
parser.add_argument('--mt_percentage', type=float, default=20.0,
                    help='Maximum mitochondrial gene percentage')
parser.add_argument('--target_sum', type=float, default=1e4,
                    help='Target sum for normalization')
parser.add_argument('--gene_selection_method', type=str, default='hvg',
                    choices=['hvg', 'heg', 'combined'],
                    help='Gene selection method')

args = parser.parse_args()

dst_file = Path(f"{args.dst_file}/{args.dataset}")
datastem = Path(f"/data/data/ST/HEST1K/")
dst_file.mkdir(parents=True, exist_ok=True)
top_k = 10000

print("=" * 60)
print("MODERN SPATIAL TRANSCRIPTOMICS PROCESSING PIPELINE")
print("=" * 60)
print(f"Dataset: {args.dataset}")
print(f"Gene processing: {args.gene_prc}")
print(f"Image processing: {args.imge_prc}")
print(f"HVG method: {args.hvg_method}")
print(f"Gene selection: {args.gene_selection_method}")
print("=" * 60)

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

print(f"Processing {len(target_dataset)} samples: {target_dataset[:3]}{'...' if len(target_dataset) > 3 else ''}")


if args.gene_prc:
    print("\n" + "=" * 40)
    print("MODERN GENE PROCESSING")
    print("=" * 40)
    
    sorted_hvg, sorted_heg = dataset_topgenes_modern(
        datastem=datastem,
        dataset_list=target_dataset,
        top_k=top_k,
        hvg_method=args.hvg_method,
        min_cells=args.min_cells,
        min_genes=args.min_genes,
        mt_percentage=args.mt_percentage,
        target_sum=args.target_sum,
        verbose=True
    )
    
    print(f"\nGene selection completed:")
    print(f"  HVGs: {len(sorted_hvg)}")
    print(f"  HEGs: {len(sorted_heg)}")

# 如果需要处理图像数据，加载模型
model = None
image_processor = None
if args.imge_prc is not None:
    print("\n" + "=" * 40)
    print("IMAGE MODEL LOADING")
    print("=" * 40)
    
    model_type = args.imge_prc
    if model_type == 'PLIP':
        print("Loading PLIP weights...")
        pipe = pipeline("zero-shot-image-classification", model="vinid/plip")
        model = AutoModelForZeroShotImageClassification.from_pretrained("vinid/plip")
        image_processor = AutoProcessor.from_pretrained("vinid/plip")
    elif model_type == 'UNI':
        print("Loading UNI weights...")
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
        print("Loading ResNet50 weights...")
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
        print("Model loaded successfully!")

# 主要处理循环 - 统一处理每个数据集
print("\n" + "=" * 40)
print("DATASET PROCESSING")
print("=" * 40)

# 用于最后验证匹配结果的列表
final_gene_barcodes = []
final_image_barcodes = []
# 用于记录每个数据集的匹配信息
dataset_match_info = []

for item in tqdm(target_dataset, desc="Processing datasets"):
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
    
    # 应用现代预处理（如果启用基因处理）
    if args.gene_prc:
        original_shape = adata.shape
        modern_spatial_preprocessing(
            adata,
            min_genes=args.min_genes,
            min_cells=args.min_cells,
            max_genes=args.max_genes,
            mt_percentage=args.mt_percentage,
            target_sum=args.target_sum,
            n_top_genes=top_k,
            flavor=args.hvg_method,
            copy=False
        )
        
        # 选择基因
        if args.gene_selection_method == 'hvg':
            selected_genes = sorted_hvg
        elif args.gene_selection_method == 'heg':
            selected_genes = sorted_heg
        else:
            combined_genes = list(dict.fromkeys(sorted_hvg + sorted_heg))
            selected_genes = combined_genes[:top_k]
        
        # 过滤基因
        available_genes = [gene for gene in selected_genes if gene in adata.var_names]
        adata = adata[:, available_genes].copy()
        
        tqdm.write(f"  {item}: {original_shape} -> {adata.shape} after modern preprocessing")
    
    # 获取基因表达矩阵和条形码
    gene_expression = adata.X.toarray() if sp.issparse(adata.X) else adata.X
    gene_barcodes = adata.obs_names.tolist()
    
    # 读取图像数据
    image_barcodes, coordinates, images = read_h5_data(mor_path)
    
    # 创建映射
    valid_gene_indices, valid_image_indices = create_barcode_mapping(gene_barcodes, image_barcodes)
    
    # 检查是否有匹配的数据
    if len(valid_gene_indices) == 0:
        tqdm.write(f"No matching barcodes found for {item}, skipping...")
        continue
    
    # 提取匹配的数据
    matched_gene_expression = gene_expression[valid_gene_indices]
    matched_gene_barcodes = [gene_barcodes[i] for i in valid_gene_indices]
    
    matched_images = images[valid_image_indices]
    matched_image_barcodes = [image_barcodes[i] for i in valid_image_indices]
    
    # 处理坐标
    matched_coordinates, normalized_coords = process_coordinates(coordinates, valid_image_indices)
    
    # 提取图像特征（如果需要）
    if args.imge_prc is not None:
        image_features = extract_image_features(
            matched_images, model, image_processor, model_type, args.batch_size
        )
    else:
        image_features = torch.tensor(matched_images)
    
    # 保存数据
    output_path = dst_file.joinpath(f'{item}.h5')
    
    with h5py.File(output_path, 'w') as f:
        # 基因数据
        f.create_dataset('gene_expression', data=matched_gene_expression)
        f.create_dataset('gene_barcodes', data=[b.encode() for b in matched_gene_barcodes])
        if args.gene_prc:
            f.create_dataset('gene_names', data=[g.encode() for g in available_genes])
        
        # 图像数据
        f.create_dataset('image_features', data=image_features.numpy())
        f.create_dataset('image_barcodes', data=[b.encode() if isinstance(b, str) else b for b in matched_image_barcodes])
        f.create_dataset('coordinates', data=matched_coordinates)
        f.create_dataset('normalized_coordinates', data=normalized_coords.numpy())
        
        # 元数据
        f.attrs['dataset'] = item
        f.attrs['n_spots'] = len(matched_gene_barcodes)
        f.attrs['n_genes'] = matched_gene_expression.shape[1]
        f.attrs['hvg_method'] = args.hvg_method if args.gene_prc else 'none'
        f.attrs['gene_selection_method'] = args.gene_selection_method if args.gene_prc else 'none'
        f.attrs['image_model'] = args.imge_prc if args.imge_prc else 'raw'
        f.attrs['preprocessing_version'] = 'modern_v1.0'
    
    # 记录匹配信息
    dataset_match_info.append({
        'dataset': item,
        'total_spots': len(matched_gene_barcodes),
        'gene_shape': matched_gene_expression.shape,
        'image_shape': image_features.shape
    })
    
    # 收集所有条形码用于最终验证
    final_gene_barcodes.extend(matched_gene_barcodes)
    final_image_barcodes.extend(matched_image_barcodes)

# 打印最终统计信息
print("\n" + "=" * 60)
print("PROCESSING SUMMARY")
print("=" * 60)

total_spots = sum([info['total_spots'] for info in dataset_match_info])
print(f"Successfully processed {len(dataset_match_info)} datasets")
print(f"Total spots processed: {total_spots}")

if args.gene_prc:
    print(f"Gene processing: {args.hvg_method} method")
    print(f"Gene selection: {args.gene_selection_method}")
    print(f"Final gene count: {len(available_genes) if 'available_genes' in locals() else 'N/A'}")

if args.imge_prc:
    print(f"Image processing: {args.imge_prc}")

print(f"Output directory: {dst_file}")

# 验证所有条形码都匹配
print(f"\nBarcode matching verification:")
print(f"  Gene barcodes: {len(final_gene_barcodes)}")
print(f"  Image barcodes: {len(final_image_barcodes)}")
print(f"  All matched: {set(final_gene_barcodes) == set([b.decode() if isinstance(b, bytes) else str(b) for b in final_image_barcodes])}")

print("\n" + "=" * 60)
print("PROCESSING COMPLETED SUCCESSFULLY!")
print("=" * 60) 