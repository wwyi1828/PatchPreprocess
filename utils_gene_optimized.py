import scipy
import scanpy as sc
import scipy.sparse as sp
import numpy as np
import pandas as pd
import re
import anndata as ad
from typing import Optional, Tuple, List, Union
import warnings

# Configure scanpy settings for spatial data
sc.settings.verbosity = 2  # verbosity level
sc.settings.set_figure_params(dpi=80, facecolor='white')

def modern_spatial_preprocessing(
    adata: ad.AnnData,
    min_genes: int = 200,          # Filter out cells with fewer genes
    min_cells: int = 3,            # Filter out genes expressed in fewer cells  
    max_genes: int = 5000,         # Filter out cells with too many genes (potential doublets)
    mt_percentage: float = 20.0,   # Max mitochondrial gene percentage
    target_sum: Optional[float] = 1e4,  # Target sum for normalization
    n_top_genes: int = 2000,       # Number of HVGs to select
    flavor: str = 'seurat_v3',     # HVG selection method
    log_transform: bool = True,
    copy: bool = False
) -> Optional[ad.AnnData]:
    """
    Modern spatial transcriptomics preprocessing pipeline following current best practices.
    
    This pipeline implements the standard workflow used in top spatial transcriptomics papers:
    1. Basic filtering (genes, cells, mitochondrial content)
    2. Normalization (size factor correction)
    3. Log transformation
    4. Highly variable gene selection using established methods
    5. Quality control metrics calculation
    
    Parameters:
    -----------
    adata : AnnData
        The spatial transcriptomics data
    min_genes : int, default 200
        Minimum number of genes expressed per cell
    min_cells : int, default 3
        Minimum number of cells expressing a gene
    max_genes : int, default 5000
        Maximum number of genes per cell (doublet detection)
    mt_percentage : float, default 20.0
        Maximum mitochondrial gene percentage
    target_sum : float, default 1e4
        Target sum for normalization (use None to skip normalization)
    n_top_genes : int, default 2000
        Number of highly variable genes to select
    flavor : str, default 'seurat_v3'
        Method for HVG selection ('seurat', 'seurat_v3', 'cell_ranger')
    log_transform : bool, default True
        Whether to apply log transformation
    copy : bool, default False
        Return a copy instead of writing to adata
        
    Returns:
    --------
    AnnData or None
        Processed AnnData object if copy=True, else None
    """
    
    if copy:
        adata = adata.copy()
    
    print("Starting modern spatial transcriptomics preprocessing...")
    print(f"Initial data shape: {adata.shape}")
    
    # Store raw counts
    adata.raw = adata
    
    # 1. Basic filtering
    print("\n1. Quality control and filtering...")
    
    # Calculate QC metrics
    adata.var['mt'] = adata.var_names.str.startswith('MT-') | adata.var_names.str.startswith('mt-')
    adata.var['ribo'] = adata.var_names.str.startswith('RPS') | adata.var_names.str.startswith('RPL')
    
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    
    # Add mitochondrial gene percentage
    adata.obs['pct_counts_mt'] = adata.obs['pct_counts_mt'] if 'pct_counts_mt' in adata.obs.columns else 0
    # Filter cells and genes
    print(f"  Filtering cells with < {min_genes} or > {max_genes} genes...")
    sc.pp.filter_cells(adata, min_genes=min_genes)
    
    print(f"  Filtering genes expressed in < {min_cells} cells...")
    sc.pp.filter_genes(adata, min_cells=min_cells)
    
    # Filter cells with high mitochondrial content
    if mt_percentage > 0:
        print(f"  Filtering cells with > {mt_percentage}% mitochondrial genes...")
        adata = adata[adata.obs.pct_counts_mt < mt_percentage, :].copy()
    
    # Filter potential doublets
    if max_genes > 0:
        adata = adata[adata.obs.n_genes_by_counts < max_genes, :].copy()
    
    print(f"  After filtering: {adata.shape}")
    
    # 2. Normalization
    if target_sum is not None:
        print(f"\n2. Normalizing to {target_sum} total counts per cell...")
        sc.pp.normalize_total(adata, target_sum=target_sum)
    
    # 3. Log transformation
    if log_transform:
        print("3. Log transformation...")
        sc.pp.log1p(adata)
    
    # 4. Highly variable gene selection
    print(f"\n4. Selecting {n_top_genes} highly variable genes using {flavor} method...")
    
    try:
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_top_genes,
            flavor=flavor,
            subset=False,  # Don't subset yet, keep all genes
            batch_key=None  # Can add batch correction if needed
        )
        
        # Store HVG information
        n_hvgs = adata.var['highly_variable'].sum()
        print(f"  Selected {n_hvgs} highly variable genes")
        
    except Exception as e:
        print(f"  Warning: HVG selection failed with {flavor}, falling back to seurat")
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_top_genes,
            flavor='seurat',
            subset=False
        )
    
    # 5. Calculate additional metrics for spatial data
    print("\n5. Calculating spatial-specific metrics...")
    
    # Calculate total UMI counts per spot
    adata.obs['total_counts'] = np.array(adata.X.sum(axis=1)).flatten()
    
    # Calculate number of detected genes per spot
    adata.obs['n_genes'] = np.array((adata.X > 0).sum(axis=1)).flatten()
    
    print("Preprocessing completed!")
    print(f"Final shape: {adata.shape}")
    print(f"Number of HVGs: {adata.var['highly_variable'].sum()}")
    
    if copy:
        return adata


def select_genes_modern(
    adata: ad.AnnData,
    method: str = 'hvg',
    n_genes: int = 2000,
    use_hvg_info: bool = True
) -> List[str]:
    """
    Select genes using modern established methods.
    
    Parameters:
    -----------
    adata : AnnData
        Processed AnnData object with HVG information
    method : str
        'hvg' for highly variable genes, 'heg' for highly expressed genes, or 'combined'
    n_genes : int
        Number of genes to select
    use_hvg_info : bool
        Whether to use pre-calculated HVG information
        
    Returns:
    --------
    List[str]
        Selected gene names
    """
    
    if method == 'hvg':
        if use_hvg_info and 'highly_variable' in adata.var.columns:
            # Use pre-calculated HVGs
            hvg_genes = adata.var_names[adata.var['highly_variable']].tolist()
            return hvg_genes[:n_genes]
        else:
            # Calculate HVGs on the fly
            temp_adata = adata.copy()
            sc.pp.highly_variable_genes(temp_adata, n_top_genes=n_genes, subset=False)
            return temp_adata.var_names[temp_adata.var['highly_variable']].tolist()
    
    elif method == 'heg':
        # Highly expressed genes based on mean expression
        mean_expr = np.array(adata.X.mean(axis=0)).flatten()
        top_indices = np.argsort(mean_expr)[::-1][:n_genes]
        return adata.var_names[top_indices].tolist()
    
    elif method == 'combined':
        # Combine HVG and HEG (50-50 split)
        n_hvg = n_genes // 2
        n_heg = n_genes - n_hvg
        
        hvg_genes = select_genes_modern(adata, 'hvg', n_hvg, use_hvg_info)
        heg_genes = select_genes_modern(adata, 'heg', n_heg, use_hvg_info)
        
        # Remove overlap and ensure we get the requested number
        combined_genes = list(dict.fromkeys(hvg_genes + heg_genes))  # Remove duplicates while preserving order
        return combined_genes[:n_genes]
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'hvg', 'heg', or 'combined'")


def dataset_topgenes_modern(
    datastem,
    dataset_list: List[str],
    top_k: int = 3000,
    hvg_method: str = 'seurat_v3',
    min_cells: int = 3,
    min_genes: int = 200,
    mt_percentage: float = 20.0,
    target_sum: float = 1e4,
    verbose: bool = False
) -> Tuple[List[str], List[str]]:
    """
    Modern approach to select top genes across multiple spatial datasets.
    
    This function implements current best practices for spatial transcriptomics:
    - Proper quality control and filtering
    - Size factor normalization
    - Established HVG selection methods
    - Batch-aware processing when needed
    
    Parameters:
    -----------
    datastem : Path
        Path to dataset directory
    dataset_list : List[str]
        List of dataset identifiers
    top_k : int, default 3000
        Number of top genes to select
    hvg_method : str, default 'seurat_v3'
        Method for HVG selection
    min_cells : int, default 3
        Minimum cells expressing a gene
    min_genes : int, default 200
        Minimum genes per cell
    mt_percentage : float, default 20.0
        Maximum mitochondrial percentage
    target_sum : float, default 1e4
        Target sum for normalization
    verbose : bool, default False
        Print detailed information
        
    Returns:
    --------
    Tuple[List[str], List[str]]
        (highly_variable_genes, highly_expressed_genes)
    """
    
    if verbose:
        print(f"Processing {len(dataset_list)} datasets with modern pipeline...")
        print(f"Target genes: {top_k}, HVG method: {hvg_method}")
    
    # Read and preprocess all datasets
    processed_datasets = []
    
    for item in dataset_list:
        mol_path = datastem.joinpath(f'st/{item}.h5ad')
        if not mol_path.exists():
            if verbose:
                print(f"File not found: {mol_path}, skipping...")
            continue
        
        if verbose:
            print(f"Processing {item}...")
        
        # Load and clean data
        adata = sc.read_h5ad(mol_path)
        adata = adata[:, ~adata.var_names.str.startswith('__ambiguous')]
        adata = avg_duplicate_geneexpr(adata, verbose=False)
        adata = avg_duplicate_cells(adata, verbose=False)
        
        # Apply modern preprocessing
        modern_spatial_preprocessing(
            adata,
            min_genes=min_genes,
            min_cells=min_cells,
            mt_percentage=mt_percentage,
            target_sum=target_sum,
            n_top_genes=top_k,
            flavor=hvg_method,
            copy=False
        )
        
        # Add dataset identifier for batch effects
        adata.obs['dataset'] = item
        processed_datasets.append(adata)
    
    if not processed_datasets:
        raise ValueError("No valid datasets found!")
    
    # Combine datasets
    if verbose:
        print("Combining datasets...")
    
    combined_adata = ad.concat(processed_datasets, join='inner', merge='same')
    combined_adata.obs_names_make_unique()
    
    # Re-run HVG selection on combined data for better consistency
    if verbose:
        print(f"Running final HVG selection on combined data...")
    
    # Use batch_key if multiple datasets for batch-aware HVG selection
    batch_key = 'dataset' if len(processed_datasets) > 1 else None
    
    sc.pp.highly_variable_genes(
        combined_adata,
        n_top_genes=top_k,
        flavor=hvg_method,
        subset=False,
        batch_key=batch_key
    )
    
    # Extract highly variable genes
    hvg_mask = combined_adata.var['highly_variable']
    
    if 'dispersions_norm' in combined_adata.var.columns:
        # Sort by normalized dispersion
        sorted_hvg = combined_adata.var_names[hvg_mask][
            np.argsort(combined_adata.var.loc[hvg_mask, 'dispersions_norm'].values)[::-1]
        ].tolist()
    else:
        # Fallback to mean expression
        means = np.array(combined_adata.X.mean(axis=0)).flatten()
        hvg_indices = np.where(hvg_mask)[0]
        sorted_indices = np.argsort(means[hvg_indices])[::-1]
        sorted_hvg = combined_adata.var_names[hvg_indices[sorted_indices]].tolist()
    
    # Calculate highly expressed genes
    means = np.array(combined_adata.X.mean(axis=0)).flatten()
    combined_adata.var['mean_expression'] = means
    sorted_heg = combined_adata.var_names[np.argsort(means)[::-1][:top_k]].tolist()
    
    if verbose:
        print(f"Selected {len(sorted_hvg)} highly variable genes")
        print(f"Selected {len(sorted_heg)} highly expressed genes")
        
        # Print overlap statistics
        overlap = set(sorted_hvg) & set(sorted_heg)
        print(f"Overlap between HVG and HEG: {len(overlap)} genes ({len(overlap)/top_k*100:.1f}%)")
    
    return sorted_hvg, sorted_heg


# Keep existing utility functions with improvements
def clean_gene_versions(adata, verbose=True):
    """Clean gene version numbers (e.g., ENSG00000000003.15 -> ENSG00000000003)"""
    gene_dict = {}
    original_to_cleaned = {}
    changes = []

    # First pass: find highest version for each gene
    for gene in adata.var_names:
        match = re.match(r"^(.+)\.(\d+)$", gene)
        if match:
            base_id, version = match.groups()
            version = int(version)
            if base_id not in gene_dict or (gene_dict[base_id] is not None and gene_dict[base_id] < version):
                gene_dict[base_id] = version
            original_to_cleaned[gene] = base_id
            changes.append((gene, base_id))
        else:
            if gene not in gene_dict:
                gene_dict[gene] = None
            original_to_cleaned[gene] = gene

    if verbose and changes:
        print(f"Cleaned {len(changes)} gene version numbers")

    # Second pass: keep only highest versions
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

    # Filter and rename
    mask = np.array([name is not None for name in new_var_names])
    adata = adata[:, mask].copy()
    adata.var_names = np.array([name for name in new_var_names if name is not None])

    return adata


def avg_duplicate_geneexpr(adata, verbose=False):
    """Average expression of duplicate genes"""
    gene_names = adata.var_names
    duplicated_genes = pd.Index(gene_names)[pd.Index(gene_names).duplicated(keep=False)]

    if len(duplicated_genes) == 0:
        return adata

    X = adata.X.toarray() if sp.issparse(adata.X) else adata.X
    expr_df = pd.DataFrame(X, index=adata.obs_names, columns=gene_names)
    expr_df_avg = expr_df.groupby(expr_df.columns, axis=1).mean()  # Use mean instead of max

    if verbose:
        print(f"Averaged {len(duplicated_genes.unique())} duplicate genes")

    new_adata = sc.AnnData(
        X=expr_df_avg.values,
        obs=adata.obs.copy(),
        var=pd.DataFrame(index=expr_df_avg.columns)
    )

    return new_adata


def avg_duplicate_cells(adata, verbose=False):
    """Average expression of duplicate cells/barcodes"""
    cell_names = adata.obs_names
    duplicated_cells = pd.Index(cell_names)[pd.Index(cell_names).duplicated(keep=False)]
    
    if len(duplicated_cells) == 0:
        return adata
        
    X = adata.X.toarray() if sp.issparse(adata.X) else adata.X
    expr_df = pd.DataFrame(X, index=cell_names, columns=adata.var_names)
    expr_df_avg = expr_df.groupby(level=0).mean()  # Use mean instead of sum
    
    if verbose:
        print(f"Averaged {len(duplicated_cells.unique())} duplicate cells")
    
    unique_obs = adata.obs.loc[~adata.obs_names.duplicated(keep='first')]
    
    new_adata = sc.AnnData(
        X=expr_df_avg.values,
        obs=unique_obs.loc[expr_df_avg.index],
        var=adata.var.copy()
    )
    
    return new_adata 