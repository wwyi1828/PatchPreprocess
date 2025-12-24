import torch
from tqdm import tqdm
from utils_gene import *

def reference_normalize(target_spots, refer_spots, refer_patches, dst_path):

    dst_adata = sc.read_h5ad(target_spots)
    adata = sc.read_h5ad(refer_spots)
    barcodes, coords, images = read_h5_data(refer_patches)

    gene_barcodes = list(adata.obs.index)
    valid_gene_indices, valid_image_indices = create_barcode_mapping(gene_barcodes, barcodes)

    coords = coords[valid_image_indices]
    ref_x = coords[:, 0]
    ref_y = coords[:, 1]
    x_min, y_min = np.min(ref_x), np.min(ref_y)

    # test.obsm['spatial']
    dst_coords = dst_adata.obsm['patch_spatial']
    x_coords = dst_coords[:, 0]
    y_coords = dst_coords[:, 1]
    normalized_x = (x_coords - x_min) / 224
    normalized_y = (y_coords - y_min) / 224
    normalized_coords = torch.from_numpy(np.column_stack((normalized_x, normalized_y))).float()
    
    remap_coords = map_to_integer_grid(dst_coords)
    coordinates = remap_coords

    dst_adata.obsm['float_coords'] = normalized_coords.numpy()
    dst_adata.obsm['coords'] = coordinates

    dst_adata.write_h5ad(dst_path)


target_dataset = [f'HD{idx}' for idx in range(1,7)]
for item in tqdm(target_dataset):
    src_path = f"/data/data/ST/temp_pcoords/{item}.h5ad"
    ref_sath = f"/data/data/ST/HEST1K/st/{item}.h5ad"
    ref_path = f"/data/data/ST/HEST1K/patches/{item}.h5"
    dst_path = f"/data/data/ST/aligned_adata/{item}.h5ad"
    reference_normalize(src_path, ref_sath, ref_path, dst_path)