#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 17:47:21 2024

@author: silverflo
"""


import argparse
from modules.utils import load_config, set_seed
from modules.running import run_BORI
import pandas as pd
import numpy as np

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train Bayesian Transformer Model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--data", type=str, required=True, help="Path to preprocessed data file")
    parser.add_argument("--output", type=str, required=True, help="Path to save the evaluation results")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set random seed
    set_seed(config["SEED"])

    # Load preprocessed data
    print(f"Loading preprocessed data from {args.data}...")
    data = pd.read_pickle(args.data)
    data.sort_values(['chrom','start'])
    data['x_pos'] = np.arange(len(data))
    print("Data loaded successfully.")

    # Update output path in config
    config["OUTPUT_PATH"] = args.output

    # Train and evaluate model

    run_BORI(config, data, args.output)

    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
