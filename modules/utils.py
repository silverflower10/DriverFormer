#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 13:33:22 2024

@author: silverflo
"""

import gc
import torch
import torch.nn.functional as F
import numpy as np
import json
import random



def pad_to_batch_size(tensor, batch_size):
    """
    Pads a tensor to the specified batch size.
    """
    current_size = tensor.size(0)
    padding_size = batch_size - current_size

    if current_size < batch_size:
        if tensor.dim() == 1:
            tensor = F.pad(tensor, (0, padding_size))
        elif tensor.dim() == 2:
            tensor = F.pad(tensor, (0, 0, 0, padding_size))
        elif tensor.dim() == 3:
            tensor = F.pad(tensor, (0, 0, 0, 0, 0, padding_size))
        else:
            raise ValueError(f"Unhandled tensor dimension: {tensor.dim()}")

        padding_mask = torch.cat([torch.ones(current_size), torch.zeros(padding_size)], dim=0)
    else:
        padding_mask = torch.ones(current_size)

    return tensor, padding_mask


def clear_memory():
    """
    Clears memory for both CPU and GPU.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_sinusoidal_position_encoding(position, d_model):
    if position.dim() == 1:
        position = position.unsqueeze(1)
    
    seq_len = position.size(1)
    angle_rates = 1 / torch.pow(10000, (2 * (torch.arange(d_model) // 2)) / d_model).to(position.device)
    position = position.unsqueeze(2)
    angle_rads = position * angle_rates.unsqueeze(0).unsqueeze(0)
    pos_encoding = torch.zeros_like(angle_rads).to(position.device)
    pos_encoding[:, :, 0::2] = torch.sin(angle_rads[:, :, 0::2])
    pos_encoding[:, :, 1::2] = torch.cos(angle_rads[:, :, 1::2])
    return pos_encoding


def load_config(config_path):
    """
    Load a configuration file in JSON format.

    Args:
        config_path (str): Path to the JSON configuration file.

    Returns:
        dict: Configuration parameters as a dictionary.
    """
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON configuration file: {e}")
        
def set_seed(seed):
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): Seed value to initialize random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

