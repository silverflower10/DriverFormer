#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 17:50:21 2024

@author: silverflo
"""


from modules.model import TransformerEncoderModel, TransformerDecoderModel
from modules.model import Variational_formerEncoderModel, Variational_formerDecoderModel

def create_models(config, device):
    """
    Initialize encoder and decoder models based on config.
    Supports both Bayesian and Basic Transformer versions.
    """
    input_dim = len(config["FEATURES"])  # Calculate input_dim dynamically
    num_chroms = config.get("NUM_CHROMS", 1)  
   
    # 모델 선택
    if  config["MODEL_TYPE"] == "basic":
        # Basic Transformer 모델 초기화
        encoder = TransformerEncoderModel(
            input_dim=input_dim,
            e_model=config["E_MODEL"],
            nhead=config["NHEAD"],
            num_encoder_layers=config["NUM_ENCODER_LAYERS"],
            dim_feedforward=config["DIM_FEEDFORWARD"],
            dropout=config["DROPOUT"]
        ).to(device)

        decoder = TransformerDecoderModel(
            input_dim=1,
            feature_dim=config["D_MODEL"],
            d_model=config["D_MODEL"],
            nhead=config["NHEAD"],
            num_encoder_layers=config["NUM_ENCODER_LAYERS"],
            num_decoder_layers=config["NUM_DECODER_LAYERS"],
            dim_feedforward=config["DIM_FEEDFORWARD"],
            dropout=config["DROPOUT"]
        ).to(device)
        
    elif config["MODEL_TYPE"] == "variational":
        
        # Variational Transformer 모델 초기화
        encoder = Variational_formerEncoderModel(
            input_dim=input_dim,
            e_model=config["E_MODEL"],
            nhead=config["NHEAD"],
            num_encoder_layers=config["NUM_ENCODER_LAYERS"],
            dim_feedforward=config["E_FEEDFORWARD"],
            dropout=config["DROPOUT"],
            num_chroms=num_chroms  
        ).to(device)

        decoder = Variational_formerDecoderModel(
            input_dim=1,
            feature_dim=config["D_MODEL"],
            d_model=config["D_MODEL"],
            nhead=config["NHEAD"],
            num_encoder_layers=config["NUM_ENCODER_LAYERS"],
            num_decoder_layers=config["NUM_DECODER_LAYERS"],
            dim_feedforward=config["D_FEEDFORWARD"],
            dropout=config["DROPOUT"],
        ).to(device)

    else:
        raise ValueError(f"Unsupported MODEL_TYPE: {config['MODEL_TYPE']}")

    return encoder, decoder