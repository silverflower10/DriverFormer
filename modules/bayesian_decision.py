#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 11:49:46 2024

@author: silverflo
"""

import pandas as pd
import numpy as np

def calculate_bayesian_fdr(results, posterior_col='posterior_probability', cost_fp=1.0, cost_fn=1.0):
    """
    Calculate Bayesian FDR and determine optimal threshold using Bayesian Decision Rule.

    Parameters:
    - results: DataFrame containing posterior probabilities and other information.
    - posterior_col: Name of the column containing posterior probabilities of H1.
    - cost_fp: Cost for False Positives.
    - cost_fn: Cost for False Negatives.

    Returns:
    - optimal_threshold: The threshold minimizing the expected loss.
    - fdr: The Bayesian False Discovery Rate value.
    - selected_results: DataFrame containing rows selected based on the optimal threshold.
    """
    # Ensure the posterior probability column exists
    if posterior_col not in results.columns:
        raise ValueError(f"Column '{posterior_col}' not found in results DataFrame.")

    # Sort by posterior probabilities
    sorted_results = results.sort_values(by=posterior_col, ascending=True)

    # Calculate cumulative sums for expected losses
    cumsum_fp = np.cumsum(1 - sorted_results[posterior_col])  # False Positive Costs
    cumsum_fn = np.cumsum(sorted_results[posterior_col][::-1])[::-1]  # False Negative Costs

    # Calculate total expected loss for each threshold
    loss = cost_fp * cumsum_fp + cost_fn * cumsum_fn
    optimal_index = np.argmin(loss)

    # Determine the optimal threshold
    optimal_threshold = sorted_results[posterior_col].iloc[optimal_index]
    print(f"Optimal threshold based on Bayesian Decision Rule: {optimal_threshold:.4f}")

    # Calculate posterior probabilities of H0 (null hypothesis)
    results['P_H0_given_zi'] = 1 - results[posterior_col]

    # Select rows based on the optimal threshold
    selected_results = results[results[posterior_col] < optimal_threshold]

    # Compute Bayesian FDR
    if not selected_results.empty:
        fdr = selected_results['P_H0_given_zi'].sum() / len(selected_results)
    else:
        fdr = 0.0  # If no rows are selected, FDR is 0 by definition

    print(f"Bayesian FDR (threshold={optimal_threshold}): {fdr:.4f}")
    return optimal_threshold, fdr, selected_results

# Example usage
if __name__ == "__main__":
    # Load results DataFrame (replace 'results.pkl' with your actual file)
    output_path = "results.pkl"  # Path to the results file
    results = pd.read_pickle(output_path)  # Load the DataFrame

    # Specify costs for Bayesian Decision Rule
    cost_fp = 1.0  # Cost for False Positives
    cost_fn = 3.0  # Cost for False Negatives

    # Calculate Bayesian FDR and optimal threshold
    optimal_threshold, fdr, selected_results = calculate_bayesian_fdr(
        results,
        posterior_col='posterior_probability',
        cost_fp=cost_fp,
        cost_fn=cost_fn
    )