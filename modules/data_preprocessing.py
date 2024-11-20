#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 17:49:18 2024

@author: silverflo
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from BORI import integrate_genomic_tiles, run_stft_analysis

def preprocess_data(config):
    """Load, process, and normalize data."""
    # Step 1: Integrate genomic tiles
    output = integrate_genomic_tiles(
        fasta_file=config["FASTA_FILE"],
        mutation_file=config["MUTATION_FILE"],
        covariate_paths=config["COVARIATE_PATHS"],
        eligible_path=config["ELIGIBLE_PATH"],
        tile_start=config["TILE_START"],
        tile_end=config["TILE_END"],
        idcap=config["IDCAP"],
        max_workers=config["MAX_WORKERS"]
    )

    # Step 2: Run STFT analysis
    tiles_df = run_stft_analysis(
        output,
        config["MUTATION_FILE"],
        window_size=config["WINDOW_SIZE"],
        overlap=config["OVERLAP"],
        fs=config["FS"],
        nperseg=config["NPERSEG"],
        noverlap=config["NOVERLAP"]
    )

    # Step 3: Normalize covariates
    columns_to_normalize = config["COLUMNS_TO_NORMALIZE"]
    scaler = MinMaxScaler()
    tiles_df[columns_to_normalize] = scaler.fit_transform(tiles_df[columns_to_normalize])

    # Add positional index for transformer input
    tiles_df = tiles_df.sort_values(["chrom", "start"])
    tiles_df["x_pos"] = range(len(tiles_df))

    return tiles_df
