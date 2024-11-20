#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 15:39:20 2024

@author: silverflo
"""

# Import submodules from `modules`
from .modules import (
    BayesianTransformerEncoderModel,
    BayesianTransformerDecoderModel,
    BasicTransformerEncoderModel,
    BasicTransformerDecoderModel,
    
    integrate_genomic_tiles,
    load_mutations,
    create_tiles,
    process_chromosome,
    parallel_process,
    change_dtypes,
    tile_creation,
    process_covariate,
    integrate_covariates,
    

    generate_binary_signal,
    compute_chromosome_spectrum,
    aggregate_to_tiles,
    process_and_aggregate_chromosome,
    process_and_aggregate_parallel,
    run_stft_analysis,
    
    CustomDataset,
    create_dataloader,
    
    pad_to_batch_size,
    clear_memory,
    get_sinusoidal_position_encoding,
    
    analyze_posterior
    
)

# Define the public API of the package
__all__ = [
    # Models
    "BayesianTransformerEncoderModel",
    "BayesianTransformerDecoderModel",
    "BasicTransformerEncoderModel",
    "BasicTransformerDecoderModel",
    
    # Genomic Tile Preprocessing
    "integrate_genomic_tiles",
    "load_mutations",
    "create_tiles",
    "process_chromosome",
    "parallel_process",
    "change_dtypes",
    "tile_creation",
    "process_covariate",
    "integrate_covariates",
    
    # STFT Analysis
    "generate_binary_signal",
    "compute_chromosome_spectrum",
    "aggregate_to_tiles",
    "process_and_aggregate_chromosome",
    "process_and_aggregate_parallel",
    "run_stft_analysis",
    
    
    
    # DataLoader
    "CustomDataset",
    "create_dataloader",
    
    # Utilities
    "pad_to_batch_size",
    "clear_memory",
    "get_sinusoidal_position_encoding",
    
    "analyze_posterior"
]