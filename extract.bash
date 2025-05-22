# python 1_extract_feats.py --config='configs/TCGA_Lung_20x.yaml'
# CUDA_VISIBLE_DEVICES=1 python 1_extract_feats.py --config='configs/C16.yaml'



# CUDA_VISIBLE_DEVICES=1 python 1_extract_pretrain_feats.py --config='configs/C16_clubyol.yaml' &&
# CUDA_VISIBLE_DEVICES=0 python 1_extract_pretrain_feats.py --config='configs/BRACS_train.yaml' &&
# CUDA_VISIBLE_DEVICES=1 python 1_extract_pretrain_feats.py --config='configs/C16_clubyol_test.yaml'
# CUDA_VISIBLE_DEVICES=0 python 1_extract_pretrain_feats.py --config='configs/BRACS_test.yaml' &&
# CUDA_VISIBLE_DEVICES=0 python 1_extract_pretrain_feats.py --config='configs/BRACS_val.yaml'


# CUDA_VISIBLE_DEVICES=1 python 3_KNN.py && 
# CUDA_VISIBLE_DEVICES=1 python 3_KNN.py
# CUDA_VISIBLE_DEVICES=1 python 1_extract_pretrain_feats.py --config='configs/TCGA_Lung.yaml'

# CUDA_VISIBLE_DEVICES=0 python 1_extract_pretrain_feats.py --config='configs/BRACS_train.yaml' --batch_size 3072
# CUDA_VISIBLE_DEVICES=0 python 1_extract_pretrain_feats.py --config='configs/BRACS_val.yaml' --batch_size 3072
# CUDA_VISIBLE_DEVICES=0 python 1_extract_pretrain_feats.py --config='configs/BRACS_test.yaml' --batch_size 3072


# CUDA_VISIBLE_DEVICES=1 python 1_extract_pretrain_feats.py --config='configs/C16.yaml' --batch_size 3072&
# CUDA_VISIBLE_DEVICES=1 python 1_extract_pretrain_feats.py --config='configs/C16_test.yaml' --batch_size 3072&
# wait


# CUDA_VISIBLE_DEVICES=1 python 1_extract_pretrain_feats.py --config='configs/C16.yaml' --batch_size 16 --model_type 'MAE'
# CUDA_VISIBLE_DEVICES=1 python 1_extract_pretrain_feats.py --config='configs/C16_test.yaml' --batch_size 16 --model_type 'MAE'
# CUDA_VISIBLE_DEVICES=1 python 1_extract_pretrain_feats.py --config='configs/C16_clusiam.yaml' --batch_size 16 --model_type 'DINOv2'
# CUDA_VISIBLE_DEVICES=1 python 1_extract_pretrain_feats.py --config='configs/C16_clusiam_test.yaml' --batch_size 16 --model_type 'DINOv2'

# python 1_extract_raw_feats.py --config='configs/BRACS_train.yaml' --batch_size 100 &
# python 1_extract_raw_feats.py --config='configs/BRACS_val.yaml' --batch_size 100 &
# python 1_extract_raw_feats.py --config='configs/BRACS_test.yaml' --batch_size 100


#!/bin/bash

# # GPU 0
# CUDA_VISIBLE_DEVICES=0 python 1_extract_pretrain_feats.py --config='configs/TCGA_Lung_20x.yaml' --batch_size 2048 --model_type ResNet50 && \
# CUDA_VISIBLE_DEVICES=0 python 1_extract_pretrain_feats.py --config='configs/BRACS_train.yaml' --batch_size 2048 --model_type PLIP &

# # GPU 1
# # CUDA_VISIBLE_DEVICES=1 python 1_extract_pretrain_feats.py --config='configs/BRACS_test.yaml' --batch_size 3072 --model_type PLIP && \
# CUDA_VISIBLE_DEVICES=1 python 1_extract_pretrain_feats.py --config='configs/BRACS_val.yaml' --batch_size 2048 --model_type PLIP &

# # GPU 2
# CUDA_VISIBLE_DEVICES=2 python 1_extract_pretrain_feats.py --config='configs/TCGA_Lung.yaml' --batch_size 2048 --model_type ResNet50 &

# wait

# CUDA_VISIBLE_DEVICES=0 python 1_extract_pretrain_feats.py --config='configs/BRACS_val.yaml' --batch_size 32 --model_type UNI
# CUDA_VISIBLE_DEVICES=0 python 1_extract_pretrain_feats.py --config='configs/BRACS_test.yaml' --batch_size 32 --model_type UNI
# (
#     CUDA_VISIBLE_DEVICES=0 python 1_extract_pretrain_feats.py --config='configs/TCGA_Lung.yaml' --batch_size 256 --model_type CONCH
# ) &
# (
#     CUDA_VISIBLE_DEVICES=1 python 1_extract_pretrain_feats.py --config='configs/BRACS_val.yaml' --batch_size 256 --model_type CONCH
#     CUDA_VISIBLE_DEVICES=1 python 1_extract_pretrain_feats.py --config='configs/BRACS_test.yaml' --batch_size 256 --model_type CONCH
# ) &
# (
#     CUDA_VISIBLE_DEVICES=2 python 1_extract_pretrain_feats.py --config='configs/BRACS_train.yaml' --batch_size 256 --model_type CONCH
#     CUDA_VISIBLE_DEVICES=2 python 1_extract_pretrain_feats.py --config='configs/TCGA_Lung_20x.yaml' --batch_size 256 --model_type CONCH
# )

# CUDA_VISIBLE_DEVICES=1 python 1_extract_pretrain_feats.py --config='configs/BRACS_val.yaml' --batch_size 1024 &
# CUDA_VISIBLE_DEVICES=0 python 1_extract_pretrain_feats.py --config='configs/BRACS_train.yaml' --batch_size 1024 &
# CUDA_VISIBLE_DEVICES=2 python 1_extract_pretrain_feats.py --config='configs/BRACS_test.yaml' --batch_size 1024 &
# wait


# CUDA_VISIBLE_DEVICES=2 python 1_extract_pretrain_feats.py --config='configs/TCGA_Lung.yaml' --batch_size 1024 --model_type PLIP&
# CUDA_VISIBLE_DEVICES=1 python 1_extract_pretrain_feats.py --config='configs/TCGA_Lung_20x.yaml' --batch_size 1024 --model_type PLIP&
# wait

# CUDA_VISIBLE_DEVICES=1 python 1_extract_pretrain_feats.py --config='configs/TCGA_BRAC.yaml' --batch_size 1024 --model_type CTrans
# CUDA_VISIBLE_DEVICES=0 python 1_extract_pretrain_feats.py --config='configs/TCGA_Lung.yaml' --batch_size 384 --model_type CTrans &
# CUDA_VISIBLE_DEVICES=2 python 1_extract_pretrain_feats.py --config='configs/TCGA_Lung_20x.yaml' --batch_size 384 --model_type CTrans &
# wait
# CUDA_VISIBLE_DEVICES=1 python 1_extract_pretrain_feats.py --config='configs/BRACS_val.yaml' --batch_size 384 --model_type CTrans

# CUDA_VISIBLE_DEVICES=2 python 1_extract_pretrain_feats.py --config='configs/BACH.yaml' --batch_size 1024&
# CUDA_VISIBLE_DEVICES=0 python 1_extract_pretrain_feats.py --config='configs/TCGA_Lung_20x.yaml' --batch_size 1024&
# wait



# CUDA_VISIBLE_DEVICES=1 python 1_extract_pretrain_feats.py --config='configs/TCGA_COAD.yaml' --batch_size 1024 --model_type ResNet50
CUDA_VISIBLE_DEVICES=1 python 1_extract_pretrain_feats.py --config='configs/BACH.yaml' --batch_size 1024 --model_type UNI

# CUDA_VISIBLE_DEVICES=1 python 1_extract_pretrain_feats.py --config='configs/trastuzumab.yaml' --batch_size 1024 --model_type PLIP &
# CUDA_VISIBLE_DEVICES=2 python 1_extract_pretrain_feats.py --config='configs/Yale_HER2.yaml' --batch_size 1024 --model_type UNI &
# wait


CUDA_VISIBLE_DEVICES=0 python 1_extract_pretrain_feats.py --config='configs/BRACS_train.yaml' --batch_size 1024 --model_type UNIv2
CUDA_VISIBLE_DEVICES=0 python 1_extract_pretrain_feats.py --config='configs/BRACS_val.yaml' --batch_size 1024 --model_type UNIv2
CUDA_VISIBLE_DEVICES=0 python 1_extract_pretrain_feats.py --config='configs/BRACS_test.yaml' --batch_size 1024 --model_type UNIv2
wait


CUDA_VISIBLE_DEVICES=2 python 1_extract_pretrain_feats.py --config='configs/C16_test.yaml' --batch_size 1024 --model_type ResNet50