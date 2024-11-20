#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 13:55:50 2024

@author: silverflo
"""
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    """
    Custom Dataset class for handling genomic data with optional overlapping.
    """
    def __init__(self, chrom_data, features, stft_features, count_feature, overlap_size=0, is_train=True):
        """
        Initialize the dataset.

        Args:
            chrom_data (pd.DataFrame): DataFrame containing genomic data.
            features (list): List of feature column names.
            stft_features (list): List of STFT feature column names.
            count_feature (str): Name of the count column.
            overlap_size (int, optional): Size of the overlap between windows. Defaults to 0.
            is_train (bool, optional): Whether the dataset is for training. Defaults to True.
        """
        self.chrom_data = chrom_data
        self.features = features
        self.stft_features = stft_features
        self.count_feature = count_feature  # Column for count values
        self.is_train = is_train
        self.overlap_size = overlap_size

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        if self.is_train and self.overlap_size > 0:
            return len(self.chrom_data) - self.overlap_size
        else:
            return len(self.chrom_data)

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.

        Args:
            idx (int): Index of the data.

        Returns:
            tuple: A tuple containing feature values, STFT values, position, and count.
        """
        if self.is_train and self.overlap_size > 0:
            end_idx = idx + 1 + self.overlap_size if idx + 1 + self.overlap_size < len(self.chrom_data) else len(self.chrom_data)
            rows = self.chrom_data.iloc[idx:end_idx]
        else:
            rows = self.chrom_data.iloc[[idx]]

        feature_values = rows[self.features].values.astype('float32')
        stft_values = rows[self.stft_features].values.astype('float32')
        position_value = rows['x_pos'].values.astype('float32')
        count_value = rows[self.count_feature].values.astype('float32')

        # Ensure consistent shape by taking the last token in case of overlap
        feature_values = feature_values[-1:]
        stft_values = stft_values[-1:]
        position_value = position_value[-1:]
        count_value = count_value[-1:]

        return feature_values, stft_values, position_value, count_value


def create_dataloader(chrom_data, features, stft_features, count_feature, batch_size=128, overlap_size=0, is_train=True):
    """
    Create a DataLoader for the CustomDataset.

    Args:
        chrom_data (pd.DataFrame): DataFrame containing genomic data.
        features (list): List of feature column names.
        stft_features (list): List of STFT feature column names.
        count_feature (str): Name of the count column.
        batch_size (int, optional): Batch size for DataLoader. Defaults to 128.
        overlap_size (int, optional): Size of the overlap between windows. Defaults to 0.
        is_train (bool, optional): Whether the dataset is for training. Defaults to True.

    Returns:
        DataLoader: A DataLoader for the CustomDataset.
    """
    dataset = CustomDataset(
        chrom_data,
        features,
        stft_features,
        count_feature,
        overlap_size=overlap_size,
        is_train=is_train
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
