# import biomart
# import pickle
# def get_ensembl_mappings(dataset='mmusculus_gene_ensembl', host='http://www.ensembl.org'):
#     server = biomart.BiomartServer(host + '/biomart')
#     mart = server.datasets[dataset]

#     # 根据物种不同，Symbol 字段的名称也可能不同
#     # 小鼠常用 'mgi_symbol'，人类常用 'hgnc_symbol'。也有 'external_gene_name' 等别名。
#     if 'mmusculus' in dataset:
#         symbol_attr = 'mgi_symbol'
#     else:
#         symbol_attr = 'hgnc_symbol'  # 对人类

#     attributes = [
#         'ensembl_transcript_id',
#         symbol_attr,
#         'ensembl_gene_id',
#         'ensembl_peptide_id'
#     ]

#     response = mart.search({'attributes': attributes})
#     data = response.raw.data.decode('ascii')

#     ensembl_to_genesymbol = {}
#     for line in data.splitlines():
#         cols = line.split('\t')
#         transcript_id = cols[0]
#         gene_symbol   = cols[1]
#         ensembl_gene  = cols[2]
#         ensembl_pep   = cols[3]

#         # 避免空字符串作为 key
#         if transcript_id:
#             ensembl_to_genesymbol[transcript_id] = gene_symbol
#         if ensembl_gene:
#             ensembl_to_genesymbol[ensembl_gene]  = gene_symbol
#         if ensembl_pep:
#             ensembl_to_genesymbol[ensembl_pep]   = gene_symbol

#     return ensembl_to_genesymbol

# human_dict = get_ensembl_mappings(dataset='hsapiens_gene_ensembl')
# mouse_dict = get_ensembl_mappings(dataset='mmusculus_gene_ensembl')
# ensembl2symbol = {}
# ensembl2symbol.update(mouse_dict)
# ensembl2symbol.update(human_dict)
# with open('/data/data/ensembl2symbol.pkl', 'wb') as f:
#     pickle.dump(ensembl2symbol, f)



import pickle
import scanpy as sc
from pathlib import Path
import re
import os
from utils_gene import clean_gene_versions


with open('/data/data/ensembl2symbol.pkl', 'rb') as f:
    ensembl2symbol = pickle.load(f)
symbol2ensemb = dict(zip(ensembl2symbol.values(), ensembl2symbol.keys()))

def remove_version(ensembl_id: str) -> str:
    return re.sub(r'\.\d+$', '', ensembl_id)

def convert_ensembl_to_symbol(adata, mapping_dict):
    new_var_names = []
    for gid in adata.var_names:
        if gid in mapping_dict:
            symbol = mapping_dict[gid]
            new_var_names.append(symbol if symbol else gid)
        else:
            new_var_names.append(gid)
    adata.var_names = new_var_names


datastem = Path("/data/data/ST/HEST1K/")
target_dataset = [f"SPA{idx}" for idx in range(51, 155)]

for item in target_dataset:
    src_path = datastem.joinpath(f"backup/{item}.h5ad")
    if not src_path.exists():
        print(f"File not found: {src_path}, skipping...")
        continue

    adata = sc.read_h5ad(src_path)
    adata = adata[:, ~adata.var_names.str.startswith('__ambiguous')]
    # convert_ensembl_to_symbol(adata, ensembl2symbol)
    convert_ensembl_to_symbol(adata, symbol2ensemb)
    adata = clean_gene_versions(adata)
    dst_path = datastem.joinpath(f"st/{item}.h5ad")
    adata.write_h5ad(dst_path)
    print(f"Saved updated file as: {dst_path}\n")

