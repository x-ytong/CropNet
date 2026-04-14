import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import TensorDataset
from util.label import label_mapping



def get_file(folder, featype, country):
    if featype == "harmonic":
        return os.path.join(folder, f"harmonic_{country}_month5-11.npz")
    elif featype == "median":
        return os.path.join(folder, f"median_S2_{country}_month5-11_day5.npz")
    elif featype == "median_CropNet":
        return os.path.join(folder, f"median_S2_{country}_month1-12_day5.npz")
    elif featype == "presto":
        return os.path.join(folder, f"presto_embedding_S2_{country}.npz")
    elif featype == "presto_MayNov":
        return os.path.join(folder, f"presto_embedding_S2_{country}_MayNov.npz")
    elif featype == "alphaearth":
        return os.path.join(folder, f"alphaearth_embedding_{country}.npz")
    else:
        raise ValueError(f"Unsupported feature type: {featype}")


def create_dataset(data, old2new):
    data_lbl = pd.Series(data['lbl']).map(old2new)
    mask = data_lbl.notna().tolist()

    data_fea = data['fea'][mask]
    data_lbl = data_lbl[mask].astype(int).to_numpy()

    data_fea, unique_id = np.unique(data_fea, axis=0, return_index=True)
    data_lbl = data_lbl[unique_id]

    data_fea = torch.tensor(data_fea, dtype=torch.float32).unsqueeze(1)
    data_lbl = torch.tensor(data_lbl, dtype=torch.long)
    dataset = TensorDataset(data_fea, data_lbl)
    return dataset


def data_load(folder, featype, source, targets, crop_index):
    info_str = ''
    train_path = get_file(folder, featype, source)
    train_data = np.load(train_path)

    for lbl, crop_num in pd.Series(train_data['lbl']).value_counts().sort_index().to_dict().items():
        info_str += f'Train: {source}, crop type: {crop_index[lbl]}, sample num: {crop_num}.\n'

    train_index, old2new = label_mapping(train_data, crop_index)
    sr_dataset = create_dataset(train_data, old2new)

    tg_datasets = {}
    for target in targets:
        test_path = get_file(folder, featype, target)
        test_data = np.load(test_path)

        for lbl, crop_num in pd.Series(test_data['lbl']).value_counts().sort_index().to_dict().items():
            info_str += f'Test: {target}, crop type: {crop_index[lbl]}, sample num: {crop_num}.\n'

        tg_datasets[target] = create_dataset(test_data, old2new)
    return sr_dataset, tg_datasets, train_index, info_str
