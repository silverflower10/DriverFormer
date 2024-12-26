#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 13:55:50 2024

@author: silverflo
"""


import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, chrom_data, features, prob,
                 sequence_length=99, stride=1,
                 is_train=True, chrom_to_id=None):
        """
        Args:
            chrom_data (pd.DataFrame): DataFrame containing genomic data.
            features (list): Columns used as input features.
            prob (list): Columns for probability values, e.g., ['prob', 'fold_sample_count'].
            sequence_length (int): Length of the sequence window.
            stride (int): Stride for the window.
            is_train (bool): Training mode if True, else test mode.
            chrom_to_id (dict): Mapping chromosome -> ID.
        """
        self.chrom_data = chrom_data
        self.features = features
        self.prob = prob
        self.is_train = is_train
        self.sequence_length = sequence_length
        self.stride = stride
        self.chrom_to_id = chrom_to_id

        # chrom_id를 얻기 위해 첫 행 확인 (빈 DataFrame일 경우 대비 예외처리 권장)
        if len(chrom_data) > 0 and chrom_to_id is not None:
            first_chrom = chrom_data.iloc[0]['chrom']
            self.chrom_id = chrom_to_id[first_chrom]
        else:
            self.chrom_id = 0

    def __len__(self):
        N = len(self.chrom_data)
        window_size = self.sequence_length + 1
        if self.is_train and self.sequence_length > 0:
            return max(0, (N - window_size) // self.stride + 1)
        else:
            return N

    def __getitem__(self, idx):
        if self.is_train and self.sequence_length > 0:
            start_idx = idx * self.stride
            end_idx = start_idx + self.sequence_length + 1
            if end_idx > len(self.chrom_data):
                end_idx = len(self.chrom_data)
            rows = self.chrom_data.iloc[start_idx:end_idx]
        else:
            rows = self.chrom_data.iloc[[idx]]

        feature_values = rows[self.features].values.astype('float32')
        prob_values = rows[self.prob].values.astype('float32')
        position_values = rows['x_pos'].values.astype('float32')

        if self.chrom_to_id is not None:
            chrom_ids = rows['chrom'].map(self.chrom_to_id).values
            chrom_id_tensor = torch.tensor(chrom_ids, dtype=torch.long)
        else:
            chrom_id_tensor = torch.zeros((len(rows),), dtype=torch.long)

        return feature_values, prob_values, position_values, chrom_id_tensor


def create_dataloader(chrom_data, features, prob,
                      batch_size=128, sequence_length=99, stride=1,
                      is_train=True, chrom_to_id=None):
    """
    Custom DataLoader 생성
    """
    dataset = CustomDataset(
        chrom_data, features, prob,
        sequence_length=sequence_length,
        stride=stride,
        is_train=is_train,
        chrom_to_id=chrom_to_id
    )
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=is_train, drop_last=False)
    return dataloader
