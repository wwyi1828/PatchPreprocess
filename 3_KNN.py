import torch
import pickle
import os
from pathlib import Path
from torch.nn import functional as F
from tqdm import trange, tqdm
from sklearn.metrics import *
import pandas as pd
from torch_geometric.data import Data as gData
import numpy as np

missing = ['test_114']
# feats_path = '/data/data/Old_CluSiam_R18'
feats_path = '/data/data/CluSiam_A09R18'
print(feats_path)
# feats_path = '/pool1/data/C16_R50'
training_data = pickle.load(open(f'{feats_path}.pkl', 'rb'))
testing_data = pickle.load(open(f'{feats_path}_test.pkl', 'rb'))

# feats_folder = Path('/data/data/Camelyon_precompute/c100_b512_20x_byol_pair_alpha09_concat_49/0')
# label_folder = Path('/data/data/Camelyon_precompute/c100_b512_20x_byol_pair_alpha09_concat_49/3')
# training_data = []
# for slide_name in tqdm(os.listdir(feats_folder)):
#     slide_name = slide_name.split('.csv')[0]
#     if slide_name not in missing:
#         feats_path = feats_folder.joinpath(f'{slide_name}.csv')
#         labels_path = label_folder.joinpath(f'{slide_name}.csv')
#         bag_feats = torch.tensor(pd.read_csv(feats_path).to_numpy(),dtype=torch.float16)
#         bag_labels = torch.tensor(pd.read_csv(labels_path).to_numpy(),dtype=torch.long).squeeze()
#         a_graph = gData(
#             x = bag_feats,
#             y = bag_labels, 
#             slide_index = slide_name)
#         training_data.append(a_graph)

# feats_folder = Path('/data/data/Camelyon_precompute_test/c100_b512_20x_byol_pair_alpha09_concat_49/0')
# label_folder = Path('/data/data/Camelyon_precompute_test/c100_b512_20x_byol_pair_alpha09_concat_49/3')
# testing_data = []
# for slide_name in tqdm(os.listdir(feats_folder)):
#     slide_name = slide_name.split('.csv')[0]
#     if slide_name not in missing:
#         feats_path = feats_folder.joinpath(f'{slide_name}.csv')
#         labels_path = label_folder.joinpath(f'{slide_name}.csv')
#         bag_feats = torch.tensor(pd.read_csv(feats_path).to_numpy(),dtype=torch.float16)
#         bag_labels = torch.tensor(pd.read_csv(labels_path).to_numpy(),dtype=torch.long).squeeze()
#         a_graph = gData(
#             x = bag_feats,
#             y = bag_labels, 
#             slide_index = slide_name)
#         testing_data.append(a_graph)

# Exclude slides without annnotations
testing_data = [_ for _ in testing_data if _.slide_index not in missing]

train_labels = torch.cat([_.y for _ in training_data], dim=0).long()
test_labels = torch.cat([_.y for _ in testing_data], dim=0).long()

train_feats = torch.cat([_.x for _ in training_data], dim=0)
test_feats = torch.cat([_.x for _ in testing_data], dim=0)

# train_batch_size = int(32768/0.8)
# test_batch_size = int(49152/0.8)

train_batch_size = int(32768/1)
test_batch_size = int(49152/1)

device = torch.device('cpu')

def compute_topk_knn(test_feats, train_feats, test_labels, train_labels,
                     k, train_batch_size, test_batch_size, device):
    num_test_batches = (len(test_feats) + test_batch_size - 1) // test_batch_size
    num_train_batches = (len(train_feats) + train_batch_size - 1) // train_batch_size

    all_predicted_labels = []

    for i in trange(num_test_batches):
        test_batch = test_feats[i * test_batch_size:(i + 1) * test_batch_size].to(device)
        test_batch = F.normalize(test_batch, dim=1)
        #Initialize global topk
        global_topk_indices = torch.full((test_batch.size(0), k), -1, dtype=torch.long, device=device)
        global_topk_values = torch.full((test_batch.size(0), k), -float('inf'), dtype=torch.float, device=device)

        for j in range(num_train_batches):
            train_batch_start_index = j * train_batch_size

            train_batch = train_feats[j * train_batch_size:(j + 1) * train_batch_size].to(device)
            train_batch = F.normalize(train_batch, dim=1)
            similarity = torch.matmul(test_batch, train_batch.T)

            topk_values, topk_indices = torch.topk(similarity, k, largest=True, sorted=True) #Current topk
            topk_indices = topk_indices + train_batch_start_index  # 转换为全局索引

            combine_vals = torch.cat([global_topk_values, topk_values], dim=1) # 2*k in total
            combine_indices = torch.cat([global_topk_indices, topk_indices], dim=1)
            _, new_indices = torch.topk(combine_vals, k, dim=1, largest=True, sorted=True)
            global_topk_indices = torch.gather(combine_indices, 1, new_indices)
            global_topk_values = torch.gather(combine_vals, 1, new_indices)

        predicted_labels = torch.mode(train_labels[global_topk_indices.cpu()], dim=1).values.cpu()
        all_predicted_labels.append(predicted_labels)
    
    return torch.cat(all_predicted_labels)



k = 1
device = torch.device("cuda")
test_preds = compute_topk_knn(test_feats, train_feats, test_labels, train_labels,
                     k, train_batch_size, test_batch_size, device)

print(feats_path)
print(accuracy_score(test_labels, test_preds), f1_score(test_labels, test_preds))