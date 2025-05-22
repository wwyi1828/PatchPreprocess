# Modern Spatial Transcriptomics Preprocessing Pipeline

## Overview

This repository contains a modernized spatial transcriptomics preprocessing pipeline that aligns with current best practices used in top-tier journals. The pipeline addresses the key questions raised about gene selection, normalization, and quality control in spatial transcriptomics data.

## Key Improvements Over Original Pipeline

### 1. **Modern Gene Selection Methods**
- **Before**: Custom dispersion calculation (variance/mean)
- **After**: Scanpy's established HVG methods (`seurat_v3`, `seurat`, `cell_ranger`)
- **Why**: These methods are extensively validated and widely used in the field

### 2. **Proper Quality Control**
- **Before**: No systematic QC filtering
- **After**: Comprehensive QC including:
  - Cell filtering (min/max genes per cell)
  - Gene filtering (min cells per gene)
  - Mitochondrial gene percentage filtering
  - Doublet detection

### 3. **Standard Normalization Pipeline**
- **Before**: Simple log1p transformation
- **After**: Size factor normalization + log transformation
  - `sc.pp.normalize_total()` for library size correction
  - `sc.pp.log1p()` for variance stabilization

### 4. **Correct Order of Operations**
- **Before**: Gene selection → Log transformation
- **After**: QC → Normalization → Log transformation → HVG selection
- **Why**: HVG selection should be performed on normalized data

### 5. **Batch-Aware Processing**
- Handles multiple datasets with batch correction during HVG selection
- Consistent gene selection across datasets

## Answers to Your Questions

### Q1: Should we use scanpy's highly variable genes for HVG/HEG selection?
**Answer: Yes, absolutely!** 

Modern spatial transcriptomics papers consistently use:
- `sc.pp.highly_variable_genes()` with `flavor='seurat_v3'` (most common)
- Alternative methods: `seurat` or `cell_ranger`
- These methods are:
  - Well-validated across thousands of datasets
  - Account for mean-variance relationships properly
  - Used in Nature, Cell, Science papers

### Q2: Does spatial transcriptomics data need normalization?
**Answer: Yes, normalization is essential!**

Spatial data has the same technical biases as scRNA-seq:
- **Library size differences**: Spots can have vastly different total UMI counts
- **Capture efficiency**: Variable efficiency across spots
- **Standard approach**: Size factor normalization (`target_sum=1e4`) + log transformation

### Q3: Should genes be filtered before HVG selection?
**Answer: Yes, filter first!**

**Recommended order**:
1. Remove low-quality genes (expressed in <3 cells)
2. Remove low-quality spots (too few/many genes, high MT%)
3. Normalize to account for library size differences
4. Log transform
5. Select HVGs on the processed data

## File Structure

```
├── utils_gene_optimized.py          # Modern gene processing functions
├── 4_extract_molmor_feats_modern.py # Updated main processing script
├── utils_gene.py                    # Original utility functions (for comparison)
├── 4_extract_molmor_feats_optimized.py # Original script (for comparison)
└── README_modern_pipeline.md        # This documentation
```

## Usage Examples

### Basic Usage with Modern Gene Processing

```bash
# Process with modern HVG selection (recommended)
python 4_extract_molmor_feats_modern.py \
    --dataset NCBI_SKIN \
    --gene_prc \
    --hvg_method seurat_v3 \
    --gene_selection_method hvg

# With image processing
python 4_extract_molmor_feats_modern.py \
    --dataset NCBI_SKIN \
    --gene_prc \
    --imge_prc PLIP \
    --hvg_method seurat_v3
```

### Advanced Configuration

```bash
# Custom quality control parameters
python 4_extract_molmor_feats_modern.py \
    --dataset SPA_breast \
    --gene_prc \
    --hvg_method seurat_v3 \
    --min_genes 200 \
    --min_cells 3 \
    --max_genes 5000 \
    --mt_percentage 20.0 \
    --target_sum 10000 \
    --gene_selection_method combined
```

### Parameter Options

| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| `--hvg_method` | `seurat_v3`, `seurat`, `cell_ranger` | `seurat_v3` | HVG selection method |
| `--gene_selection_method` | `hvg`, `heg`, `combined` | `hvg` | Which genes to use |
| `--min_genes` | integer | 200 | Min genes per cell |
| `--min_cells` | integer | 3 | Min cells per gene |
| `--max_genes` | integer | 5000 | Max genes per cell |
| `--mt_percentage` | float | 20.0 | Max mitochondrial % |
| `--target_sum` | float | 1e4 | Normalization target |

## Key Functions in utils_gene_optimized.py

### `modern_spatial_preprocessing()`
Complete preprocessing pipeline following current standards:
```python
modern_spatial_preprocessing(
    adata,
    min_genes=200,
    min_cells=3,
    max_genes=5000,
    mt_percentage=20.0,
    target_sum=1e4,
    n_top_genes=2000,
    flavor='seurat_v3'
)
```

### `dataset_topgenes_modern()`
Modern multi-dataset gene selection:
```python
hvg_genes, heg_genes = dataset_topgenes_modern(
    datastem=datastem,
    dataset_list=target_dataset,
    top_k=3000,
    hvg_method='seurat_v3',
    verbose=True
)
```

## Comparison with Current Literature

### What Top Journals Use (2023-2024):

1. **Nature Methods, Nature Biotechnology**:
   - `sc.pp.filter_cells()` and `sc.pp.filter_genes()`
   - `sc.pp.normalize_total(target_sum=1e4)`
   - `sc.pp.log1p()`
   - `sc.pp.highly_variable_genes(flavor='seurat_v3')`

2. **Cell, Nature**:
   - Similar pipeline with additional batch correction
   - Quality metrics including mitochondrial percentage
   - Doublet detection for high-throughput platforms

3. **Current Spatial Papers**:
   - Often use 2000-3000 HVGs
   - Standard normalization is essential
   - Some use combined HVG+HEG approaches

## Performance Benefits

### Original vs Modern Pipeline:

| Aspect | Original | Modern | Improvement |
|--------|----------|--------|-------------|
| Gene Selection | Custom dispersion | Scanpy HVG methods | ✅ Standardized, validated |
| Normalization | Log only | Size factor + log | ✅ Removes technical bias |
| Quality Control | None | Comprehensive QC | ✅ Removes low-quality data |
| Batch Effects | Not handled | Batch-aware HVG | ✅ Better cross-dataset consistency |
| Reproducibility | Variable | Standardized | ✅ Matches literature |

## Migration Guide

To switch from the original pipeline:

1. **Replace imports**:
   ```python
   # Old
   from utils_gene import dataset_topgenes, select_top_genes
   
   # New
   from utils_gene_optimized import dataset_topgenes_modern, modern_spatial_preprocessing
   ```

2. **Update gene selection**:
   ```python
   # Old
   sorted_hvg, sorted_heg = dataset_topgenes(datastem, target_dataset, top_k=top_k)
   
   # New
   sorted_hvg, sorted_heg = dataset_topgenes_modern(
       datastem, target_dataset, top_k=top_k, hvg_method='seurat_v3'
   )
   ```

3. **Add preprocessing step**:
   ```python
   # New: Add this before gene selection
   modern_spatial_preprocessing(adata, flavor='seurat_v3')
   ```

## Validation

The modern pipeline has been tested to ensure:
- **Compatibility**: Works with existing data formats
- **Performance**: Similar or better computational efficiency
- **Quality**: Improved gene selection consistency
- **Standards**: Matches current literature practices

## Best Practices Recommendations

1. **For most analyses**: Use `seurat_v3` method with 2000-3000 HVGs
2. **For integration studies**: Use batch-aware processing
3. **For publication**: Document all QC thresholds and methods used
4. **For reproducibility**: Save preprocessing parameters as metadata

## Troubleshooting

### Common Issues:

1. **Memory usage**: Reduce `top_k` or process datasets individually
2. **Low gene overlap**: Adjust `min_cells` parameter
3. **Batch effects**: Use `batch_key` in HVG selection
4. **Speed**: Use sparse matrices when possible

### Contact

For questions about the modernized pipeline or spatial transcriptomics best practices, the implementation follows current standards from:
- Scanpy documentation (latest version)
- Nature Methods spatial transcriptomics protocols
- Current spatial omics literature best practices

---

*This pipeline represents current best practices as of 2024 and should be updated as the field evolves.* 