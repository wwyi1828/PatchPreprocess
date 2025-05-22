import scanpy as sc
import mygene
import pandas as pd
from pathlib import Path
from tqdm import tqdm

datastem = Path(f"/data/data/ST/HEST1K/")
target_dataset = [f'SPA{idx}' for idx in range(51,155)][0:68] # SPA_breast_1, ensemble id
target_dataset = [f'SPA{idx}' for idx in range(51,155)][68:] # SPA_breast_2, symbol id

for item in tqdm(target_dataset):
    mol_path = datastem.joinpath(f'st/{item}.h5ad')
    if not mol_path.exists():
        tqdm.write(f"File not found: {mol_path}, skipping...")
        continue
    adata = sc.read_h5ad(mol_path)
    adata = adata[:, ~adata.var_names.str.startswith('__ambiguous')]

ensembl_ids = adata.var_names.tolist()
mg = mygene.MyGeneInfo()
res = mg.querymany(ensembl_ids, scopes='ensembl.gene', fields='symbol', species='human')

res[0]

adata = adata[adata.obs['n_genes_by_counts'] > 200, :].copy()
adata = adata[adata.obs['total_counts'] > 500, :].copy()
adata = adata[adata.obs['pct_counts_mito'] < 15, :].copy()

sc.pp.filter_genes(adata, min_cells=3)
adata = adata[:, adata.var['total_counts'] >= 20].copy()

gene_dict = {}
for item in res:
    if 'symbol' in item and item.get('query') in ensembl_ids:
        gene_dict[item.get('query')] = item.get('symbol')
    elif 'notfound' in item and item.get('query') in ensembl_ids:
        gene_dict[item.get('query')] = item.get('query')

gene_symbols = pd.Series(gene_dict)

adata.var_names = [gene_dict.get(x, x) for x in adata.var_names]

adata.var_names_make_unique()

adata.write('your_file_with_symbols.h5ad')

print(f"Converted {len(gene_dict)} Ensembl IDs to gene symbols")

