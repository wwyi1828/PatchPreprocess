import scipy
import scanpy as sc
import scipy.sparse as sp
import numpy as np
import pandas as pd
import re
import anndata as ad
import json
from typing import Optional, List, Mapping, Sequence, Dict
import h5py
from pathlib import Path

GENE_PREFIX_RE = re.compile(r'^(GRCh38|GRCm39|human|mouse|HG38|MM10|GRCh37)[_|-]+', flags=re.IGNORECASE)

def argsort_desc_nan_last(values: Sequence[float]) -> np.ndarray:
    """Argsort descending, treating NaN as -inf so they end up last after sorting."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return np.array([], dtype=np.int64)
    arr = np.nan_to_num(arr, nan=-np.inf)
    return np.argsort(arr, kind="mergesort")[::-1]

def get_ensembl_to_symbol_mapping(
    organism: str = "human",
    use_cache: bool = True,
    cache_file: Optional[Path] = None,
) -> Dict[str, str]:
    """Fetch or load Ensembl→Symbol mapping with on-disk caching."""
    cache_path = cache_file or Path(f".ensembl_to_symbol_{organism}.json")

    if use_cache and cache_path.exists():
        try:
            with cache_path.open() as f:
                return json.load(f)
        except Exception:
            pass

    try:
        from pybiomart import Server

        server = Server(host="http://www.ensembl.org")
        if organism.lower() == "human":
            dataset_name = "hsapiens_gene_ensembl"
        elif organism.lower() == "mouse":
            dataset_name = "mmusculus_gene_ensembl"
        else:
            raise ValueError(f"Unsupported organism: {organism}")

        dataset = server.marts["ENSEMBL_MART_ENSEMBL"].datasets[dataset_name]
        result = dataset.query(
            attributes=["ensembl_gene_id", "external_gene_name"],
            use_attr_names=True
        )

        mapping: Dict[str, str] = {}
        for _, row in result.iterrows():
            ensembl_id = row["ensembl_gene_id"]
            symbol = row["external_gene_name"]
            if ensembl_id and symbol:
                mapping[ensembl_id] = symbol
                base_id = ensembl_id.split(".")[0]
                if base_id != ensembl_id:
                    mapping[base_id] = symbol

        if use_cache:
            try:
                with cache_path.open("w") as f:
                    json.dump(mapping, f, indent=2)
            except Exception:
                pass

        return mapping

    except ImportError:
        print("Warning: pybiomart not installed. Install with: pip install pybiomart")
        return {}
    except Exception as exc:
        print(f"Warning: Failed to fetch Ensembl mapping: {exc}")
        return {}

def standardize_gene_name(name: str) -> str:
    """String-level gene cleaning (mirror clean_var_names): drop prefixes/version, uppercase."""
    cleaned = GENE_PREFIX_RE.sub('', str(name))
    return cleaned.split('.')[0].upper()

def load_gene_names(
    h5ad_path: Path,
    clean_names: bool = False,
    mapping: Optional[Mapping[str, str]] = None,
    remove_internal: bool = False,
) -> List[str]:
    """Read gene names with optional mapping/cleaning, without loading matrices."""
    adata = ad.read_h5ad(h5ad_path, backed="r")

    if mapping:
        adata = convert_ensembl_to_symbol(adata, mapping)
    if clean_names:
        adata = clean_var_names(adata, verbose=False)

    names = [str(n) for n in adata.var_names]
    if remove_internal:
        names = [n for n in names if n and not n.startswith('__') and n.lower() != 'nan']

    file_handle = getattr(adata, "file", None)
    if file_handle is not None:
        file_handle.close()
    return names

def compute_global_gene_intersection(
    organ_map: Mapping,
    data_root: Path,
    mapping: Optional[Mapping[str, str]] = None,
    clean_names: bool = True,
    remove_internal: bool = True,
) -> List[str]:
    """Compute intersection of gene sets across all datasets in organ_map."""
    all_ids = []
    for organ_datasets in organ_map.values():
        for ids in organ_datasets.values():
            all_ids.extend(ids)

    global_set = None
    for ds_id in all_ids:
        path = data_root / f"{ds_id}.h5ad"
        if not path.exists():
            continue
        try:
            genes = set(load_gene_names(path, clean_names=clean_names, mapping=mapping, remove_internal=remove_internal))
        except Exception:
            continue
        if global_set is None:
            global_set = genes
        else:
            global_set &= genes
    return sorted(global_set) if global_set else []

def clean_var_names(adata: ad.AnnData, verbose: bool = True) -> ad.AnnData:
    """
    统一基因命名规范：移除前缀、移除版本号、转大写。
    """
    if adata.n_vars == 0:
        return adata
    original_names = adata.var_names.astype(str)
    # 移除前缀 (如 GRCh38__, human--) 和 版本号 (如 .13)
    cleaned = [re.sub(r'^(GRCh38|GRCm39|human|mouse|HG38|MM10|GRCh37)[_|-]+', '', n, flags=re.IGNORECASE).split('.')[0].upper() for n in original_names]

    diff_count = sum(1 for i in range(len(cleaned)) if cleaned[i] != original_names[i])
    if verbose and diff_count > 0:
        print(f"Standardized {diff_count} gene names.")

    adata.var_names = cleaned
    return adata

def aggregate_adata(adata: ad.AnnData, strategy: str = 'sum', verbose: bool = False) -> ad.AnnData:
    """
    对 adata 进行去重：
    1. 聚合重复基因 (columns)
    2. 聚合重复条形码 (rows)
    默认均使用 Sum 策略。
    """
    # 1. 聚合重复基因
    if adata.var_names.duplicated().any():
        old_names = adata.var_names.astype(str)
        unique_names = []
        seen = set()
        for name in old_names:
            if name not in seen:
                unique_names.append(name); seen.add(name)

        name_to_idx = {name: i for i, name in enumerate(unique_names)}
        if strategy == 'sum':
            rows, cols = range(len(old_names)), [name_to_idx[n] for n in old_names]
            P = sp.csr_matrix((np.ones(len(old_names)), (rows, cols)), shape=(len(old_names), len(unique_names)))
            new_X = adata.X @ P
        else: # Max fallback
            X_dense = adata.X.toarray() if sp.issparse(adata.X) else adata.X
            new_X_dense = np.zeros((adata.shape[0], len(unique_names)), dtype=X_dense.dtype)
            for i, name in enumerate(unique_names):
                indices = [j for j, n in enumerate(old_names) if n == name]
                new_X_dense[:, i] = np.max(X_dense[:, indices], axis=1)
            new_X = sp.csr_matrix(new_X_dense) if sp.issparse(adata.X) else new_X_dense

        adata = ad.AnnData(X=new_X, obs=adata.obs, var=pd.DataFrame(index=unique_names), uns=adata.uns, obsm=adata.obsm)

    # 2. 聚合重复细胞 (Barcodes)
    if adata.obs_names.duplicated().any():
        X = adata.X.toarray() if sp.issparse(adata.X) else adata.X
        df = pd.DataFrame(X, index=adata.obs_names)
        # 细胞聚合始终使用 sum 以保证 UMI 计数正确
        df_sum = df.groupby(level=0).sum()
        unique_obs = adata.obs.loc[~adata.obs_names.duplicated(keep='first')]
        adata = ad.AnnData(X=df_sum.values, obs=unique_obs.loc[df_sum.index], var=adata.var)

    return adata

def map_to_integer_grid(coords, patch_size):
    """将物理坐标映射到整数网格 [0,0], [0,1]..."""
    x_min, y_min = np.min(coords, axis=0)
    norm_coords = (coords - [x_min, y_min]) / patch_size
    integer_coords = np.zeros_like(norm_coords, dtype=int)
    used = set()
    # 简单的四舍五入映射，冲突时进行螺旋搜索（简化版逻辑保留）
    for i, pos in enumerate(norm_coords):
        target = tuple(np.round(pos).astype(int))
        if target not in used:
            integer_coords[i] = target
            used.add(target)
        else:
            # 冲突处理：寻找最近的空位
            found = False
            for r in range(1, 20):
                for dx in range(-r, r+1):
                    for dy in range(-r, r+1):
                        new_target = (target[0]+dx, target[1]+dy)
                        if new_target not in used:
                            integer_coords[i] = new_target
                            used.add(new_target); found = True; break
                    if found: break
                if found: break
    return integer_coords - np.min(integer_coords, axis=0)

# --- 常用 IO 与 映射 ---
def convert_ensembl_to_symbol(adata, mapping):
    """根据映射表将 Ensembl ID 转换为 Symbol"""
    new_names = []
    for n in adata.var_names:
        mapped = mapping.get(n, n)
        # 如果映射到 nan，保留原始名称
        if pd.isna(mapped):
            new_names.append(n)
        else:
            new_names.append(mapped)
    adata.var_names = new_names
    return adata

def read_h5_data(file_path):
    import h5py
    with h5py.File(file_path, 'r') as f:
        return f['barcode'][()], f['coords'][()], f['img'][()]

def create_barcode_mapping(gene_barcodes, image_barcodes):
    image_barcode_list = [b.item().decode() if hasattr(b, 'item') else str(b) for b in image_barcodes]
    barcode_to_image_idx = {barcode: idx for idx, barcode in enumerate(image_barcode_list)}
    v_gene, v_img = [], []
    for i, barcode in enumerate(gene_barcodes):
        if barcode in barcode_to_image_idx:
            v_gene.append(i); v_img.append(barcode_to_image_idx[barcode])
    return v_gene, v_img


def build_rank_adatas_unified(
    all_ids: Sequence[str],
    st_dir: Path,
    patch_dir: Path,
    *,
    global_genes: Sequence[str],
    e2s: Mapping[str, str],
) -> List[ad.AnnData]:
    rank_adatas: List[ad.AnnData] = []
    for ds_id in all_ids:
        p = st_dir / f"{ds_id}.h5ad"
        mor_path = patch_dir / f"{ds_id}.h5"
        if not p.exists() or not mor_path.exists():
            continue
        ad_tmp = sc.read_h5ad(p)
        ad_tmp = clean_var_names(ad_tmp, verbose=False)
        ad_tmp = convert_ensembl_to_symbol(ad_tmp, e2s)
        ad_tmp = aggregate_adata(ad_tmp)
        barcodes, coords, images = read_h5_data(mor_path)
        gene_barcodes = list(ad_tmp.obs.index)
        v_gene, v_img = create_barcode_mapping(gene_barcodes, barcodes)
        if not v_gene:
            continue
        ad_tmp = ad_tmp[v_gene].copy()
        X_df = pd.DataFrame(
            ad_tmp.X.toarray() if sp.issparse(ad_tmp.X) else ad_tmp.X,
            columns=ad_tmp.var_names,
        )
        X_aligned = X_df.reindex(columns=global_genes, fill_value=0).values
        rank_adatas.append(
            ad.AnnData(X=X_aligned, obs=ad_tmp.obs, var=pd.DataFrame(index=global_genes))
        )
    return rank_adatas


def compute_hvg_heg_rankings_unified(
    rank_adatas: Sequence[ad.AnnData],
    *,
    top_k: int = 2000,
    flavor: str = "seurat",
):
    combined = ad.concat(list(rank_adatas), join="inner")
    combined.obs_names_make_unique()
    sums = np.array(combined.X.sum(axis=1)).flatten()
    rank_base = combined[sums > 0, :].copy()
    sc.pp.normalize_total(rank_base, target_sum=1e4)
    sc.pp.log1p(rank_base)
    sc.pp.highly_variable_genes(rank_base, n_top_genes=top_k, flavor=flavor)
    hvg_indices = argsort_desc_nan_last(rank_base.var["dispersions_norm"].values)
    heg_indices = argsort_desc_nan_last(np.array(rank_base.X.mean(axis=0)).flatten())
    return hvg_indices, heg_indices


def compute_hvg_heg_union(
    adatas: Sequence[ad.AnnData],
    *,
    top_k: int = 2000,
    flavor: str = "seurat",
    min_cells: int = 3,
):
    combined_adata = ad.concat(list(adatas), join="inner")
    combined_adata.obs_names_make_unique()
    orig_shape = combined_adata.shape
    orig_n_vars = combined_adata.n_vars

    sc.pp.filter_genes(combined_adata, min_cells=min_cells)
    sc.pp.normalize_total(combined_adata, target_sum=1e4)
    sc.pp.log1p(combined_adata)

    sc.pp.highly_variable_genes(
        combined_adata,
        n_top_genes=top_k,
        flavor=flavor,
        subset=False,
    )
    hvg_mask = combined_adata.var["highly_variable"]
    sorted_hvg = combined_adata.var_names[hvg_mask][
        argsort_desc_nan_last(combined_adata.var.loc[hvg_mask, "dispersions_norm"].values)
    ]

    means = np.array(combined_adata.X.mean(axis=0)).flatten()
    combined_adata.var["means"] = means
    sorted_heg = combined_adata.var_names[np.argsort(means)[::-1][:top_k]]

    union_genes: List[str] = []
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

    return union_genes, hvg_indices, heg_indices, sorted_hvg, sorted_heg, orig_shape, orig_n_vars


# --- Standardized H5 writers to keep output schema consistent ---
def save_gene_h5(
    out_path,
    mol_feats,
    cords,
    float_cords,
    total_umi_counts,
    orig_cords,
    gene_names,
    hvg_indices,
    heg_indices,
    global_hvg_indices=None,
    global_heg_indices=None,
):
    """Save gene features with a consistent schema and dtypes."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mol_feats = np.asarray(mol_feats, dtype=np.float32)
    cords = np.asarray(cords, dtype=np.int64)
    float_cords = np.asarray(float_cords, dtype=np.float32)
    total_umi_counts = np.asarray(total_umi_counts, dtype=np.float32)
    orig_cords = np.asarray(orig_cords, dtype=np.float32)
    gene_names = np.asarray(gene_names, dtype="S")
    hvg_indices = np.asarray(hvg_indices, dtype=np.int64)
    heg_indices = np.asarray(heg_indices, dtype=np.int64)
    if global_hvg_indices is not None:
        global_hvg_indices = np.asarray(global_hvg_indices, dtype=np.int64)
    if global_heg_indices is not None:
        global_heg_indices = np.asarray(global_heg_indices, dtype=np.int64)

    with h5py.File(out_path, "w") as f:
        f.create_dataset("mol_feats", data=mol_feats)
        f.create_dataset("cords", data=cords)
        f.create_dataset("float_cords", data=float_cords)
        f.create_dataset("total_umi_counts", data=total_umi_counts)
        f.create_dataset("orig_cords", data=orig_cords)
        f.create_dataset("union_gene_names", data=gene_names)
        f.create_dataset("hvg_indices", data=hvg_indices)
        f.create_dataset("heg_indices", data=heg_indices)
        if global_hvg_indices is not None:
            f.create_dataset("global_hvg_indices", data=global_hvg_indices)
        if global_heg_indices is not None:
            f.create_dataset("global_heg_indices", data=global_heg_indices)


def save_mor_h5(out_path, mor_feats):
    """Save morphology features with a consistent schema and dtypes."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mor_feats = np.asarray(mor_feats, dtype=np.float32)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("mor_feats", data=mor_feats)
