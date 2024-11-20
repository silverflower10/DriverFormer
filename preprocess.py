#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 19:06:08 2024

@author: silverflo
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Preprocess genomic data for Bayesian Transformer model.
"""

import argparse
from modules.utils import load_config
from modules.data_preprocessing import preprocess_data

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Preprocess genomic data")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--output", type=str, required=True, help="Path to save preprocessed data")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Preprocess data
    data = preprocess_data(config)

    # Save preprocessed data
    data.to_pickle(args.output)
    print(f"Preprocessed data saved to {args.output}")

if __name__ == "__main__":
    main()
