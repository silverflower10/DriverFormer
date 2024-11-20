#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 14:06:15 2024

@author: silverflo
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 14:00:00 2024

@author: silverflo
"""

# Import from submodules within the modules directory
from .model import (
    BayesianTransformerEncoderModel,
    BayesianTransformerDecoderModel,
    BasicTransformerEncoderModel,
    BasicTransformerDecoderModel
)

# Import functions from genomic_tile_preprocessing module
from .genomic_tile_preprocessing import (
    integrate_genomic_tiles,
    load_mutations,
    create_tiles,
    process_chromosome,
    parallel_process,
    change_dtypes,
    tile_creation,
    process_covariate,
    integrate_covariates
)

from .stft_module import (
    generate_binary_signal,
    compute_chromosome_spectrum,
    aggregate_to_tiles,
    process_and_aggregate_chromosome,
    process_and_aggregate_parallel,
    run_stft_analysis
)


from .data_loader import (
    CustomDataset,
    create_dataloader
)

from .utils import (
    pad_to_batch_size,
    clear_memory,
    get_sinusoidal_position_encoding
)

from .posterior_analysis import analyze_posterior


# Specify all the modules that will be accessible when the package is imported
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
