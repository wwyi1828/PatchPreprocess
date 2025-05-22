import scipy
import scanpy as sc
import scipy.sparse as sp
import cupy as cp
import numpy as np
import pandas as pd
import re
import anndata as ad

def clean_gene_versions(adata, verbose=True):
    gene_dict = {}  # 存储基因 ID -> 最高版本号
    original_to_cleaned = {}  # 记录映射关系（原始ID -> 处理后的ID）
    changes = []  # 记录发生的变更

    # 第一次遍历：找出每个基因的最高版本
    for gene in adata.var_names:
        match = re.match(r"^(.+)\.(\d+)$", gene)
        if match:
            base_id, version = match.groups()
            version = int(version)
            if base_id not in gene_dict or (gene_dict[base_id] is not None and gene_dict[base_id] < version):
                gene_dict[base_id] = version
            original_to_cleaned[gene] = base_id  # 记录映射关系
            
            # 记录变更
            changes.append((gene, base_id))
        else:
            if gene not in gene_dict:  # 仅当该基因未出现在字典中时才加入
                gene_dict[gene] = None
            original_to_cleaned[gene] = gene  # 记录映射关系

    # 打印变更信息
    if verbose and changes:  # 只有当有变更时才打印
        for original, cleaned in changes:
            print(f"{original} --> {cleaned}")

    # 第二次遍历：保留最高版本，去除其他版本
    new_var_names = []
    for gene in adata.var_names:
        match = re.match(r"^(.+)\.(\d+)$", gene)
        if match:
            base_id, version = match.groups()
            if int(version) == gene_dict[base_id]:
                new_var_names.append(base_id)
            else:
                new_var_names.append(None)
        else:
            new_var_names.append(gene)

    # 过滤并重命名
    mask = np.array([name is not None for name in new_var_names])
    adata = adata[:, mask].copy()
    adata.var_names = np.array([name for name in new_var_names if name is not None])

    return adata

def map_to_integer_grid(coords, patch_size=224, initial_radius=10, expansion_factor=1.1, max_attempts=1000):
    """
    Maps coordinates to a [0,0], [0,1], etc. grid system where each patch
    has a unique position that's as close as possible to the normalized coordinates.
    
    Args:
        coords: numpy array of shape (n, 2) containing the original coordinates
        patch_size: size of each patch (default: 224)
        initial_radius: initial search radius for finding grid points (default: 15)
        expansion_factor: factor to expand search radius if needed (default: 1.5)
        
    Returns:
        numpy array of shape (n, 2) containing the mapped integer grid coordinates
    """
    # Extract x and y coordinates
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    
    # Normalize coordinates by subtracting minimum and dividing by patch size
    x_min, y_min = np.min(x_coords), np.min(y_coords)
    normalized_x = (x_coords - x_min) / patch_size
    normalized_y = (y_coords - y_min) / patch_size
    normalized_coords = np.column_stack((normalized_x, normalized_y))
    
    # Calculate distance to the nearest integer grid point
    distances = np.sum((normalized_coords - np.round(normalized_coords))**2, axis=1)
    
    # Sort indices by distance (closest to integer grid points first)
    sorted_indices = np.argsort(distances)
    
    # Assign coordinates to integer grid points
    integer_coords = np.zeros((len(coords), 2), dtype=int)
    used_coords = set()
    
    for idx in sorted_indices:
        norm_coord = normalized_coords[idx]
        nearest_grid = (int(round(norm_coord[0])), int(round(norm_coord[1])))
        
        if nearest_grid not in used_coords:
            # If the nearest grid point is available, assign it
            integer_coords[idx] = nearest_grid
            used_coords.add(nearest_grid)
        else:
            # Start with the initial search radius
            search_radius = initial_radius
            
            for attempt in range(max_attempts):
                # Find the next best unused integer grid point
                candidates = []
                
                # Use a spiral search pattern for efficiency
                for radius in range(1, search_radius + 1):
                    # Check points at current radius around the nearest grid
                    for dx in range(-radius, radius + 1):
                        for dy in [-radius, radius]:  # Top and bottom edges
                            test_coord = (nearest_grid[0] + dx, nearest_grid[1] + dy)
                            if test_coord not in used_coords:
                                dist = (norm_coord[0] - test_coord[0])**2 + (norm_coord[1] - test_coord[1])**2
                                candidates.append((test_coord, dist))
                    
                    for dy in range(-radius + 1, radius):  # Left and right edges (excluding corners)
                        for dx in [-radius, radius]:
                            test_coord = (nearest_grid[0] + dx, nearest_grid[1] + dy)
                            if test_coord not in used_coords:
                                dist = (norm_coord[0] - test_coord[0])**2 + (norm_coord[1] - test_coord[1])**2
                                candidates.append((test_coord, dist))
                
                if candidates:
                    # Sort by distance and pick the closest
                    candidates.sort(key=lambda x: x[1])
                    best_coord = candidates[0][0]
                    integer_coords[idx] = best_coord
                    used_coords.add(best_coord)
                    break
                else:
                    # Expand the search radius and try again
                    search_radius = int(search_radius * expansion_factor)
                    # print(f"Expanding search radius to {search_radius} for point {idx}")
            
            if attempt == max_attempts - 1 and not candidates:
                # If we've exhausted all attempts and still can't find a point,
                # find any unused coordinate regardless of distance
                min_x, max_x = int(min(normalized_coords[:, 0])) - 10, int(max(normalized_coords[:, 0])) + 10
                min_y, max_y = int(min(normalized_coords[:, 1])) - 10, int(max(normalized_coords[:, 1])) + 10
                
                # Find any unused grid point in the extended range
                for x in range(min_x, max_x + 1):
                    for y in range(min_y, max_y + 1):
                        if (x, y) not in used_coords:
                            integer_coords[idx] = [x, y]
                            used_coords.add((x, y))
                            # print(f"Assigned distant point ({x}, {y}) to coordinate {idx}")
                            break
                    if (x, y) not in used_coords:
                        break
                
                # If we still can't find a point, create a new one outside the range
                if tuple(integer_coords[idx]) not in used_coords:
                    new_x, new_y = max_x + 1, min_y
                    integer_coords[idx] = [new_x, new_y]
                    used_coords.add((new_x, new_y))
                    # print(f"Created new point outside range: ({new_x}, {new_y}) for coordinate {idx}")

    min_grid_x = np.min(integer_coords[:, 0])
    min_grid_y = np.min(integer_coords[:, 1])
    integer_coords[:, 0] -= min_grid_x
    integer_coords[:, 1] -= min_grid_y
    return integer_coords
        
def avg_duplicate_geneexpr(adata, verbose=False):

    gene_names = adata.var_names
    duplicated_genes = pd.Index(gene_names)[pd.Index(gene_names).duplicated(keep=False)]

    if len(duplicated_genes) == 0:
        # if verbose:
        #     print("No duplicated genes found.")
        return adata

    X = adata.X.toarray() if sp.issparse(adata.X) else adata.X
    expr_df = pd.DataFrame(X, index=adata.obs_names, columns=gene_names)

    expr_df_avg = expr_df.groupby(expr_df.columns, axis=1).max()

    if verbose:
        for gene in duplicated_genes.unique():
            duplicate_indices = np.where(gene_names == gene)[0]
            print(f"Gene name: {gene}")
            for idx in duplicate_indices:
                expr_values = expr_df.iloc[:, idx].values
                print(f"  Original expression (column {idx}): {expr_values}")
            print(f"  Avg. Expression: {expr_df_avg[gene].values}\n")

    new_adata = sc.AnnData(
        X=expr_df_avg.values, 
        obs=adata.obs.copy(), 
        var=pd.DataFrame(index=expr_df_avg.columns)
    )

    return new_adata

def avg_duplicate_cells(adata, verbose=False):
    cell_names = adata.obs_names
    duplicated_cells = pd.Index(cell_names)[pd.Index(cell_names).duplicated(keep=False)]
    
    if len(duplicated_cells) == 0:
        if verbose:
            print("No duplicated cells/barcodes found.")
        return adata
        
    # Convert to array for easier manipulation
    X = adata.X.toarray() if sp.issparse(adata.X) else adata.X
    expr_df = pd.DataFrame(X, index=cell_names, columns=adata.var_names)
    
    # Group by cell names (rows) and average
    expr_df_avg = expr_df.groupby(level=0).sum()
    
    if verbose:
        for cell in duplicated_cells.unique():
            duplicate_indices = np.where(cell_names == cell)[0]
            print(f"Cell barcode: {cell}")
            print(f"  Number of duplicates: {len(duplicate_indices)}")
            # Optional: print some statistics about the merged cells
            cell_expressions = expr_df.loc[cell]
            print(f"  Expression correlation: {cell_expressions.T.corr().mean().mean():.4f}")
            print(f"  Merged {len(duplicate_indices)} duplicate entries into one\n")
    
    unique_obs = adata.obs.loc[~adata.obs_names.duplicated(keep='first')]
    
    new_adata = sc.AnnData(
        X=expr_df_avg.values,
        obs=unique_obs.loc[expr_df_avg.index],  # Keep only obs for remaining cells
        var=adata.var.copy()
    )
    
    return new_adata


def select_top_genes(adata, k=50, log_transform=True):
    if scipy.sparse.issparse(adata.X):
        X_np = adata.X.toarray()
        X = cp.array(X_np)
    else:
        X = cp.array(adata.X)
    
    if log_transform:
        X = cp.log1p(X)
    
    means = cp.mean(X, axis=0)
    variances = cp.var(X, axis=0)
    
    dispersion = variances / (means + 1e-12)
    
    top_genes_idx = cp.argsort(dispersion)[-k:][::-1].get()
    top_variable_genes = adata.var_names[top_genes_idx]

    top_mean_idx = cp.argsort(means)[-k:][::-1].get()
    top_mean_genes = adata.var_names[top_mean_idx]
    
    return top_variable_genes, top_mean_genes

def dataset_topgenes(datastem, dataset_list, top_k=3000, normalize_total=False, verbose=False):
    """
    从数据集列表中选择高变异基因(HVG)和高表达基因(HEG)
    
    参数:
        dataset_list: 数据集ID列表
        top_k: 选择的基因数量
        verbose: 是否显示详细信息
        
    返回:
        sorted_hvg: 排序后的高变异基因名称列表
        sorted_heg: 排序后的高表达基因名称列表
    """
    if verbose:
        print(f"从{len(dataset_list)}个数据集中选择{top_k}个高变异和高表达基因...")
    
    # 读取并预处理所有数据集
    all_adata = []
    for item in dataset_list:
        mol_path = datastem.joinpath(f'st/{item}.h5ad')
        if not mol_path.exists():
            if verbose:
                print(f"File not found: {mol_path}, skipping...")
            continue
        adata = sc.read_h5ad(mol_path)
        adata = adata[:, ~adata.var_names.str.startswith('__ambiguous')]
        adata = avg_duplicate_geneexpr(adata, verbose=verbose)
        adata = avg_duplicate_cells(adata)
        all_adata.append(adata)
    
    # 合并数据集
    combined_adata = ad.concat(all_adata, join='inner')
    combined_adata.obs_names_make_unique()
    
    # 归一化和对数转换
    if normalize_total:
        sc.pp.normalize_total(combined_adata, target_sum=1e4)
    sc.pp.log1p(combined_adata)
    
    # 找出高变异基因
    sc.pp.highly_variable_genes(
        combined_adata, 
        n_top_genes=top_k,
        flavor='seurat',
        subset=False
    )
    
    # 排序高变异基因
    hvg_mask = combined_adata.var['highly_variable']
    sorted_hvg = combined_adata.var_names[hvg_mask][
        np.argsort(combined_adata.var.loc[hvg_mask, 'dispersions_norm'].values)[::-1]
    ]
    
    # 计算平均表达并找出高表达基因
    means = np.array(combined_adata.X.mean(axis=0)).flatten()
    combined_adata.var['means'] = means
    sorted_heg = combined_adata.var_names[np.argsort(means)[::-1][:top_k]]
    
    if verbose:
        print(f"选择了{len(sorted_hvg)}个高变异基因和{len(sorted_heg)}个高表达基因")
    
    return sorted_hvg, sorted_heg