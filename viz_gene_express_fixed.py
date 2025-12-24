import scanpy as sc
import openslide
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import json
import h5py
import anndata as ad
from scipy.sparse import csr_matrix

def save_spatial_plot(adata, save_path: str, name: str, key: str, pl_kwargs={}):
    """
    从 AnnData 对象保存空间表达谱图。
    
    Args:
        adata (sc.AnnData): AnnData 对象. 
        save_path (str): 保存图像的目录路径。
        name (str): 图像文件名前缀。
        key (str): 在图中着色的特征 (来自 adata.obs 或 adata.var_names)。
        pl_kwargs(dict): 传递给 sc.pl.spatial 的其他参数。
    """
    # 在非交互式环境中关闭图像显示
    plt.ioff()

    # 确保保存目录存在
    os.makedirs(save_path, exist_ok=True)
    
    # 使用 scanpy 绘制空间图
    fig = sc.pl.spatial(
        adata,
        show=False,
        img_key="downscaled_fullres",
        color=[key],
        title=f"Spots colored by {key}",
        return_fig=True,
        **pl_kwargs
    )
    
    filename = f"{name}_spatial_plot.png"
    
    # 保存图像
    fig.savefig(os.path.join(save_path, filename), dpi=400, bbox_inches='tight')
    plt.close(fig)
    print(f"Spatial plot saved to {os.path.join(save_path, filename)}")

def load_adata_from_processed_h5(h5_path: str):
    """
    从预处理后的 h5 文件加载 AnnData 对象。

    Args:
        h5_path (str): 预处理 h5 文件的路径。

    Returns:
        sc.AnnData: 包含基因表达数据的 AnnData 对象。
    """
    print(f"Loading processed H5 file from: {h5_path}")

    with h5py.File(h5_path, 'r') as f:
        # 读取基因表达数据
        mol_feats = f['mol_feats'][:]
        print(f"Gene expression matrix shape: {mol_feats.shape}")

        # 读取基因名称
        union_gene_names_bytes = f['union_gene_names'][:]
        union_gene_names = [name.decode('utf-8') for name in union_gene_names_bytes]
        print(f"Number of genes: {len(union_gene_names)}")

        # 读取坐标信息
        orig_cords = f['orig_cords'][:]
        cords = f['cords'][:]

        # 读取 HVG/HEG 索引（如果存在）
        hvg_indices = f.get('hvg_indices', None)
        heg_indices = f.get('heg_indices', None)

        if hvg_indices is not None:
            hvg_indices = hvg_indices[:]
        if heg_indices is not None:
            heg_indices = heg_indices[:]

    # 创建 AnnData 对象
    # 使用稀疏矩阵存储基因表达数据以节省内存
    X_sparse = csr_matrix(mol_feats)
    adata = ad.AnnData(X=X_sparse)

    # 设置变量（基因）名称
    adata.var_names = union_gene_names

    # 设置观测（细胞/spot）名称
    n_spots = mol_feats.shape[0]
    adata.obs_names = [f"spot_{i}" for i in range(n_spots)]

    # 添加坐标信息到 obsm（spatial 坐标）
    adata.obsm['spatial'] = orig_cords
    print(f"Spatial coordinates shape: {orig_cords.shape}")

    # 添加 HVG/HEG 信息到 var（如果存在）
    if hvg_indices is not None:
        # 过滤掉超出当前基因矩阵范围的 HVG 索引
        valid_hvg_indices = hvg_indices[hvg_indices < len(union_gene_names)]
        if len(valid_hvg_indices) > 0:
            adata.var['is_hvg'] = False
            adata.var.loc[adata.var.index[valid_hvg_indices], 'is_hvg'] = True
            print(f"Added {len(valid_hvg_indices)} HVG annotations (filtered from {len(hvg_indices)})")
        else:
            print("Warning: No valid HVG indices found within current gene set")

    if heg_indices is not None:
        # 过滤掉超出当前基因矩阵范围的 HEG 索引
        valid_heg_indices = heg_indices[heg_indices < len(union_gene_names)]
        if len(valid_heg_indices) > 0:
            adata.var['is_heg'] = False
            adata.var.loc[adata.var.index[valid_heg_indices], 'is_heg'] = True
            print(f"Added {len(valid_heg_indices)} HEG annotations (filtered from {len(heg_indices)})")
        else:
            print("Warning: No valid HEG indices found within current gene set")

    # 计算并添加基本的 QC 指标
    adata.obs['total_counts'] = np.array(adata.X.sum(axis=1)).flatten()
    adata.obs['n_genes_by_counts'] = np.array((adata.X > 0).sum(axis=1)).flatten()

    print(f"Created AnnData object with {adata.n_obs} spots and {adata.n_vars} genes")
    return adata

def main():
    """主函数，用于解析命令行参数并执行绘图流程。"""
    parser = argparse.ArgumentParser(
        description="Generate and save a spatial plot for a given gene or QC metric from H5AD/processed H5 and WSI files.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("data_path", type=str, help="Path to the data file (.h5ad or processed .h5).")
    parser.add_argument("wsi_path", type=str, help="Path to the WSI file (e.g., .tif, .svs).")
    parser.add_argument(
        "--use_processed_h5",
        action="store_true",
        help="Use processed H5 file instead of raw H5AD file. Data path should point to .h5 file."
    )
    parser.add_argument(
        "--gene", 
        type=str, 
        default="total_counts", 
        help="Gene name to plot (must be in adata.var_names). "
             "Defaults to 'total_counts' (a QC metric)."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./output", 
        help="Directory to save the plot. Defaults to './output'."
    )
    parser.add_argument(
        "--output_name", 
        type=str, 
        default=None, 
        help="Output file name prefix. "
             "Defaults to the gene name or 'total_counts'."
    )
    parser.add_argument(
        "--width",
        type=int,
        default=2000,
        help="Width of the downscaled image in pixels. Defaults to 2000."
    )
    parser.add_argument(
        "--spot_diameter_um",
        type=float,
        default=55.0,
        help="Spot diameter in micrometers. Defaults to 55.0 um (typical for 10x Visium)."
    )
    parser.add_argument(
        "--pixel_size_um",
        type=float,
        default=None,
        help="Pixel size in um/pixel. If not provided, will try to load from corresponding JSON metadata file or use 0.5 as default."
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default=None,
        help="Path to JSON metadata file containing pixel_size_um_estimated. If not provided, will try to infer from h5ad path."
    )
    parser.add_argument(
        "--round_v",
        action="store_true",
        help="Round expression values to nearest integers for better readability."
    )
    parser.add_argument(
        "--no_log1p",
        action="store_true",
        help="Disable log1p transformation of expression values (enabled by default)."
    )

    args = parser.parse_args()

    # --- 数据加载 ---
    print(f"Loading data file from: {args.data_path}")
    print(f"Loading WSI file from: {args.wsi_path}")
    try:
        if args.use_processed_h5:
            # 从预处理 H5 文件加载
            adata = load_adata_from_processed_h5(args.data_path)
        else:
            # 从原始 H5AD 文件加载
            adata = sc.read_h5ad(args.data_path)
        wsi = openslide.OpenSlide(args.wsi_path)
    except Exception as e:
        print(f"\nError loading files: {e}")
        print("Please ensure the paths are correct and files are not corrupted.")
        return

    # --- 表达值log1p变换 ---
    if not args.no_log1p:
        print("Applying log1p transformation to expression values...")
        if hasattr(adata.X, 'toarray'):  # 稀疏矩阵
            # 先将负值设为0，再对稀疏矩阵应用log1p
            print(f"Original min value: {adata.X.data.min():.6f}")
            negative_count = (adata.X.data < 0).sum()
            if negative_count > 0:
                print(f"Warning: Found {negative_count} negative values, setting them to 0 before log1p")
                adata.X.data = np.maximum(adata.X.data, 0)
            adata.X.data = np.log1p(adata.X.data)
            print("log1p transformation applied to sparse matrix")
        else:  # 密集矩阵
            print(f"Original min value: {adata.X.min():.6f}")
            negative_count = (adata.X < 0).sum()
            if negative_count > 0:
                print(f"Warning: Found {negative_count} negative values, setting them to 0 before log1p")
                adata.X = np.maximum(adata.X, 0)
            adata.X = np.log1p(adata.X)
            print("log1p transformation applied to dense matrix")
        
        # 重新计算QC指标（因为表达值发生了变化）
        print("Recalculating QC metrics after log1p transformation...")
        adata.obs['total_counts'] = np.array(adata.X.sum(axis=1)).flatten()
        adata.obs['n_genes_by_counts'] = np.array((adata.X > 0).sum(axis=1)).flatten()

    # --- 表达值取整处理 ---
    if args.round_v:
        print("Rounding expression values to nearest integers...")
        if hasattr(adata.X, 'toarray'):  # 稀疏矩阵
            original_dtype = adata.X.dtype
            # 转换为密集数组，进行取整，然后转回稀疏矩阵
            dense_data = adata.X.toarray()
            rounded_data = np.round(dense_data).astype(original_dtype)
            adata.X = csr_matrix(rounded_data)
            print(f"Expression values rounded for sparse matrix (dtype: {original_dtype})")
        else:  # 密集矩阵
            original_dtype = adata.X.dtype
            adata.X = np.round(adata.X).astype(original_dtype)
            print(f"Expression values rounded for dense matrix (dtype: {original_dtype})")

        # 重新计算QC指标（因为表达值发生了变化）
        print("Recalculating QC metrics after rounding...")
        adata.obs['total_counts'] = np.array(adata.X.sum(axis=1)).flatten()
        adata.obs['n_genes_by_counts'] = np.array((adata.X > 0).sum(axis=1)).flatten()

    # --- 图像处理 ---
    print("Downscaling WSI...")
    downscaled_width = args.width
    wsi_width, wsi_height = wsi.dimensions
    downscaled_height = int(wsi_height * (downscaled_width / wsi_width))
    thumbnail = wsi.get_thumbnail((downscaled_width, downscaled_height))
    thumbnail_np = np.array(thumbnail)

    scale_factor = downscaled_width / wsi_width

    # --- 确定像素尺寸 ---
    pixel_size_um = args.pixel_size_um
    if pixel_size_um is None:
        # 尝试从 JSON metadata 文件中加载
        metadata_path = args.metadata_path
        if metadata_path is None:
            # 尝试基于数据路径推断 metadata 路径
            data_dir = os.path.dirname(args.data_path)
            data_basename = os.path.basename(args.data_path)
            if args.use_processed_h5:
                data_basename = data_basename.replace('.h5', '')
            else:
                data_basename = data_basename.replace('.h5ad', '')
            potential_metadata_path = os.path.join(os.path.dirname(data_dir), 'metadata', f'{data_basename}.json')
            if os.path.exists(potential_metadata_path):
                metadata_path = potential_metadata_path
                print(f"Found metadata file: {metadata_path}")
        
        if metadata_path and os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    pixel_size_um = metadata.get('pixel_size_um_estimated')
                    if pixel_size_um is not None:
                        print(f"Loaded pixel size from metadata: {pixel_size_um} um/pixel")
                        
                        # 同时从 metadata 更新 spot 直径（如果存在的话）
                        metadata_spot_diameter = metadata.get('spot_diameter')
                        if metadata_spot_diameter is not None and args.spot_diameter_um == 55.0:  # 只在使用默认值时更新
                            args.spot_diameter_um = metadata_spot_diameter
                            print(f"Updated spot diameter from metadata: {metadata_spot_diameter} um")
                            
                        # 可选：显示其他有用信息
                        inter_spot_dist = metadata.get('inter_spot_dist')
                        if inter_spot_dist is not None:
                            print(f"Inter-spot distance from metadata: {inter_spot_dist} um")
                    else:
                        print("pixel_size_um_estimated not found in metadata file")
            except Exception as e:
                print(f"Error reading metadata file: {e}")
        
        # 如果仍然没有找到，尝试从 adata 的现有 spatial 信息中推断
        if pixel_size_um is None:
            if 'spatial' in adata.uns and 'ST' in adata.uns['spatial']:
                if 'scalefactors' in adata.uns['spatial']['ST']:
                    existing_spot_diameter = adata.uns['spatial']['ST']['scalefactors'].get('spot_diameter_fullres')
                    if existing_spot_diameter is not None:
                        pixel_size_um = args.spot_diameter_um / existing_spot_diameter
                        print(f"Estimated pixel size from existing spatial data: {pixel_size_um:.3f} um/pixel")
        
        # 最后的默认值
        if pixel_size_um is None:
            pixel_size_um = 0.5  # 默认值
            print(f"Using default pixel size: {pixel_size_um} um/pixel")
    else:
        print(f"Using provided pixel size: {pixel_size_um} um/pixel")

    # 计算正确的 spot_diameter_fullres (物理尺寸 / 像素尺寸)
    spot_diameter_fullres = args.spot_diameter_um / pixel_size_um
    print(f"Calculated spot_diameter_fullres: {spot_diameter_fullres:.1f} pixels "
          f"({args.spot_diameter_um} um / {pixel_size_um} um/pixel)")

    # --- 更新 AnnData 对象 ---
    print("Adding spatial information to AnnData object...")
    library_id = 'ST' 
    if 'spatial' not in adata.uns:
        adata.uns['spatial'] = {}
    if library_id not in adata.uns['spatial']:
        adata.uns['spatial'][library_id] = {}
    
    # 添加图像和缩放因子信息 (按照 HEST 的标准格式)
    adata.uns['spatial'][library_id]['images'] = {'downscaled_fullres': thumbnail_np}
    adata.uns['spatial'][library_id]['scalefactors'] = {
        'tissue_downscaled_fullres_scalef': scale_factor,
        'spot_diameter_fullres': spot_diameter_fullres  # 正确计算的 spot 直径
    }

    # --- 准备绘图 ---
    plot_key = args.gene
    
    # 检查指定的基因是否存在
    if plot_key != 'total_counts' and plot_key not in adata.var_names:
        print(f"\nError: Gene '{plot_key}' not found in adata.var_names.")
        print("Please check the gene name for typos.")
        available_genes = adata.var_names.tolist()
        if len(available_genes) > 10:
            print("First 10 available genes:", available_genes[:10])
        else:
            print("Available genes:", available_genes)
        return
    
    # 如果需要，计算 QC 指标
    if plot_key == 'total_counts' and plot_key not in adata.obs:
        print(f"'{plot_key}' not found in adata.obs. Calculating QC metrics...")
        sc.pp.calculate_qc_metrics(adata, inplace=True)

    # 设置输出文件名
    output_name = args.output_name if args.output_name is not None else plot_key.replace('/', '_')

    # --- 生成并保存图像 ---
    print(f"Generating plot for '{plot_key}'...")
    save_spatial_plot(adata, save_path=args.output_dir, name=output_name, key=plot_key)

if __name__ == "__main__":
    main()

# 使用示例：
"""
使用原始 H5AD 文件：
python viz_gene_express_fixed.py /data/data/ST/HEST1K/st/MISC10.h5ad /data/data/ST/HEST1K/wsis/MISC10.tif --metadata_path /data/data/ST/HEST1K/metadata/MISC10.json --gene CACNG2 --output_name real

python viz_gene_express_fixed.py /data/data/SAPPHIRE/MISC_brain/gene/MISC10.h5 /data/data/ST/HEST1K/wsis/MISC10.tif --use_processed_h5 --metadata_path /data/data/ST/HEST1K/metadata/MISC10.json --gene CACNG2 --output_name h5_real
python viz_gene_express_fixed.py /data/Projects/SAPPHIRE/predictions/MISC_brain/predictions/MISC10.h5 /data/data/ST/HEST1K/wsis/MISC10.tif --use_processed_h5 --metadata_path /data/data/ST/HEST1K/metadata/MISC10.json --gene CACNG2 --output_name h5_pred

查看帮助：
python viz_gene_express_fixed.py --help
"""
