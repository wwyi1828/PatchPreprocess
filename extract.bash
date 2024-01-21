# python 1_extract_feats.py --config='configs/TCGA_Lung_20x.yaml'
# CUDA_VISIBLE_DEVICES=1 python 1_extract_feats.py --config='configs/C16.yaml'



# CUDA_VISIBLE_DEVICES=1 python 1_extract_pretrain_feats.py --config='configs/C16_clubyol.yaml' &&
# CUDA_VISIBLE_DEVICES=0 python 1_extract_pretrain_feats.py --config='configs/BRACS_train.yaml' &&
# CUDA_VISIBLE_DEVICES=1 python 1_extract_pretrain_feats.py --config='configs/C16_clubyol_test.yaml'
# CUDA_VISIBLE_DEVICES=0 python 1_extract_pretrain_feats.py --config='configs/BRACS_test.yaml' &&
# CUDA_VISIBLE_DEVICES=0 python 1_extract_pretrain_feats.py --config='configs/BRACS_val.yaml'


# CUDA_VISIBLE_DEVICES=1 python 3_KNN.py && 
# CUDA_VISIBLE_DEVICES=1 python 3_KNN.py
# CUDA_VISIBLE_DEVICES=2 python 1_extract_pretrain_feats.py --config='configs/TCGA_Lung.yaml'

# CUDA_VISIBLE_DEVICES=2 python 1_extract_pretrain_feats.py --config='configs/BRACS_train.yaml'
# CUDA_VISIBLE_DEVICES=2 python 1_extract_pretrain_feats.py --config='configs/BRACS_val.yaml'
# CUDA_VISIBLE_DEVICES=2 python 1_extract_pretrain_feats.py --config='configs/BRACS_test.yaml'


CUDA_VISIBLE_DEVICES=2 python 1_extract_pretrain_feats.py --config='configs/C16.yaml' --batch_size 2048&
CUDA_VISIBLE_DEVICES=1 python 1_extract_pretrain_feats.py --config='configs/C16_test.yaml' --batch_size 1536&
wait