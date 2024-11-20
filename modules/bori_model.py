#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 17:50:21 2024

@author: silverflo
"""

import torch.nn as nn
from BORI import BayesianTransformerEncoderModel, BayesianTransformerDecoderModel
from BORI import BasicTransformerEncoderModel, BasicTransformerDecoderModel

def create_models(config, device):
    """
    Initialize encoder and decoder models based on config.
    Supports both Bayesian and Basic Transformer versions.
    """
    input_dim = len(config["FEATURES"])  # Calculate input_dim dynamically
    stft_features_dim = len(config["STFT_FEATURES"])  # Calculate stft_features_dim dynamically

    # 모델 선택
    if config["MODEL_TYPE"] == "bayesian":
        # Bayesian Transformer 모델 초기화
        encoder = BayesianTransformerEncoderModel(
            input_dim=input_dim,
            e_model=config["E_MODEL"],
            nhead=config["NHEAD"],
            num_encoder_layers=config["NUM_ENCODER_LAYERS"],
            dim_feedforward=config["DIM_FEEDFORWARD"],
            dropout=config["DROPOUT"]
        ).to(device)

        decoder = BayesianTransformerDecoderModel(
            input_dim=stft_features_dim,
            feature_dim=config["D_MODEL"],
            d_model=config["D_MODEL"],
            nhead=config["NHEAD"],
            num_encoder_layers=config["NUM_ENCODER_LAYERS"],
            num_decoder_layers=config["NUM_DECODER_LAYERS"],
            dim_feedforward=config["DIM_FEEDFORWARD"],
            dropout=config["DROPOUT"]
        ).to(device)

    elif config["MODEL_TYPE"] == "basic":
        # Basic Transformer 모델 초기화
        encoder = BasicTransformerEncoderModel(
            input_dim=input_dim,
            e_model=config["E_MODEL"],
            nhead=config["NHEAD"],
            num_encoder_layers=config["NUM_ENCODER_LAYERS"],
            dim_feedforward=config["DIM_FEEDFORWARD"],
            dropout=config["DROPOUT"]
        ).to(device)

        decoder = BasicTransformerDecoderModel(
            input_dim=stft_features_dim,
            feature_dim=config["D_MODEL"],
            d_model=config["D_MODEL"],
            nhead=config["NHEAD"],
            num_encoder_layers=config["NUM_ENCODER_LAYERS"],
            num_decoder_layers=config["NUM_DECODER_LAYERS"],
            dim_feedforward=config["DIM_FEEDFORWARD"],
            dropout=config["DROPOUT"]
        ).to(device)

    else:
        raise ValueError(f"Unsupported MODEL_TYPE: {config['MODEL_TYPE']}")

    return encoder, decoder