common_genes = set.intersection(*all_genes)
unique_genes = set.union(*all_genes)
print(f"Common genes: {len(common_genes)}")
print(f"Total unique genes: {len(unique_genes)}")

for i, adata in enumerate(all_adata):
    print(f"Dataset {i}:")
    print(f"  - Unique genes: {adata.var_names.is_unique}")  # 这里去掉括号
    print(f"  - Unique cells: {adata.obs_names.is_unique}")  # 这里也去掉括号

def check_duplicate_genes_in_dataset(adata, dataset_idx, max_genes=5, max_cells=5):
    """
    检查单个数据集中的重复基因及其UMI计数
    
    参数:
    adata: AnnData对象
    dataset_idx: 数据集索引
    max_genes: 最多显示的重复基因数量
    max_cells: 每个基因最多显示的细胞数量
    """
    import pandas as pd
    import numpy as np
    import scipy.sparse as sp
    
    print(f"\n数据集 {dataset_idx} 分析:")
    print(f"形状: {adata.shape[0]} 细胞 x {adata.shape[1]} 基因")
    
    # 检查重复基因
    var_names = pd.Series(adata.var_names)
    duplicated_mask = var_names.duplicated(keep=False)
    duplicate_genes = var_names[duplicated_mask].unique()
    
    print(f"重复基因数量: {len(duplicate_genes)} / {len(var_names)}")
    
    if len(duplicate_genes) == 0:
        print("没有重复基因\n")
        return
    
    # 显示前几个重复基因及其表达值
    genes_to_display = min(max_genes, len(duplicate_genes))
    print(f"\n显示前 {genes_to_display} 个重复基因的UMI计数:")
    
    for gene in duplicate_genes[:genes_to_display]:
        print(f"\n基因: {gene}")
        
        # 获取这个基因名在var_names中的所有位置
        gene_indices = np.where(adata.var_names == gene)[0]
        print(f"  在基因矩阵中出现 {len(gene_indices)} 次，索引位置: {gene_indices}")
        
        # 对每个重复基因位置，显示一些细胞的表达值
        for idx in gene_indices:
            # 获取这个基因位置的表达向量
            if sp.issparse(adata.X):
                gene_expr = adata.X[:, idx].toarray().flatten()
            else:
                gene_expr = adata.X[:, idx].flatten()
            
            # 计算基本统计数据
            non_zero = np.count_nonzero(gene_expr)
            percent_expressed = (non_zero / len(gene_expr)) * 100
            
            print(f"  位置 {idx} 统计: ", end="")
            print(f"表达细胞: {non_zero}/{len(gene_expr)} ({percent_expressed:.2f}%), ", end="")
            print(f"平均UMI: {np.mean(gene_expr):.4f}, 最大UMI: {np.max(gene_expr):.1f}")
            
            # 显示前几个有表达的细胞
            if non_zero > 0:
                # 找到表达最高的几个细胞
                top_cells = np.argsort(gene_expr)[-max_cells:][::-1]
                print(f"    表达最高的 {len(top_cells)} 个细胞 (索引:UMI):")
                for cell_idx in top_cells:
                    print(f"      细胞 {cell_idx}: {gene_expr[cell_idx]:.1f}")
            else:
                print("    所有细胞UMI计数均为0")
    
    print("\n")  # 添加额外空行作为分隔

# 对每个数据集运行分析
for i, adata in enumerate(all_adata):
    check_duplicate_genes_in_dataset(adata, i)

# 输出总体统计信息
print("总体统计:")
for i, adata in enumerate(all_adata):
    var_names = pd.Series(adata.var_names)
    duplicated_mask = var_names.duplicated(keep=False)
    duplicate_genes = var_names[duplicated_mask].unique()
    
    print(f"数据集 {i}: 总基因 {len(var_names)}, 重复基因 {len(duplicate_genes)} ({len(duplicate_genes)/len(var_names)*100:.2f}%)")