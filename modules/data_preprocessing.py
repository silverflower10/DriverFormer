#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 17:49:18 2024

@author: silverflo
"""



import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from modules import integrate_genomic_tiles, integrate_folds





def preprocess_data(config):
    """
    1) integrate_genomic_tiles: tile+mutation+covariates+eligible
    2) integrate_folds: fold 정보 + merges
    3) MinMaxScaler on selected columns
    4) x_pos 추가
    """
    print("Step 1: Integrating genomic tiles and covariates...")
    integrated_df = integrate_genomic_tiles(
        fasta_file=config["FASTA_FILE"],
        mutation_file=config["MUTATION_FILE"],
        covariate_paths=config["COVARIATE_PATHS"],
        eligible_path=config["ELIGIBLE_PATH"],
        tile_start=config["TILE_START"],
        tile_end=config["TILE_END"],
        idcap=config["IDCAP"],
        max_workers=config["MAX_WORKERS"]
    )
    print("Step 1 completed: Genomic tiles integrated with covariates.")

    print("Step 2: Integrating folds...")
    output = integrate_folds(
        fasta_file=config["FASTA_FILE"],
        mutation_file=config["MUTATION_FILE"],
        integrated_df=integrated_df,
        tile_start=config["TILE_START"],
        tile_end=config["TILE_END"],
        n=config["N_FOLDS"],
        idcap=config["IDCAP"],
        max_workers=config["MAX_WORKERS"]
    )
    print("Step 2 completed: Fold integration done.")

    columns_to_normalize = config.get("COLUMNS_TO_NORMALIZE", [])
    if columns_to_normalize:
        scaler = MinMaxScaler()
        output[columns_to_normalize] = scaler.fit_transform(output[columns_to_normalize])
        print(f"Normalization completed for columns: {columns_to_normalize}")
    else:
        print("No columns specified for normalization.")

    # x_pos 추가
    output = output.sort_values(["chrom","start"])
    output["x_pos"] = output.groupby(["chrom","start","end"]).ngroup()
    print("Data preprocessing completed successfully.")

    return output