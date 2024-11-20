#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 12:50:42 2024

@author: silverflo
"""
# stft_module.py

import pandas as pd
import numpy as np
from scipy.signal import stft
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.preprocessing import MinMaxScaler

def generate_binary_signal(df, start, end):
    grouped = (
        df[(df['position'] >= start) & (df['position'] < end)]
        .groupby('sample')['position']
        .mean()
        .round().astype(int)
    )
    signal = np.zeros(end - start)
    if not grouped.empty:
        positions = grouped - start
        signal[positions] = 1
    else:
        signal[:] = 0.01  # Default value to prevent empty signal
    return signal

def compute_chromosome_spectrum(chrom, tiles_group, df, window_size, overlap, fs, nperseg, noverlap):
    chrom_start = tiles_group["start"].min()
    chrom_end = tiles_group["end"].max()
    binned_data = []

    start = chrom_start
    while start < chrom_end:
        end = min(start + window_size, chrom_end)

        binary_signal = generate_binary_signal(df[df['chrom'] == chrom], start, end)
  
        # STFT
        f, t, Zxx = stft(binary_signal, fs=fs,
                         nperseg=min(nperseg, len(binary_signal)),
                         noverlap=min(min(nperseg - 1, len(binary_signal) - 1), noverlap))
        power_spectrum = np.abs(Zxx) ** 2

        if np.all(power_spectrum == 0):
            max_energy_freq = 0
            variability = 0
            mean_power = 0
        else:
            non_dc_indices = f > 0
            power_spectrum_mean = power_spectrum.mean(axis=1)
            if non_dc_indices.sum() > 0:
                max_energy_freq = f[non_dc_indices][np.argmax(power_spectrum_mean[non_dc_indices])]
            else:
                max_energy_freq = 0  # All energy in DC component
            variability = power_spectrum.std()
            mean_power = power_spectrum_mean.mean()

        binned_data.append({
            "chrom": chrom,
            "start": start,
            "end": end,
            "mean_power": mean_power,
            "max_energy_freq": max_energy_freq,
            "variability": variability
        })

        start += window_size - overlap

    return chrom, binned_data

def aggregate_to_tiles(tiles, binned_df):
    aggregated_data = []
    for _, tile in tiles.iterrows():
        tile_start = tile['start']
        tile_end = tile['end']
        tile_chrom = tile['chrom']

        overlapping_bins = binned_df[
            (binned_df['chrom'] == tile_chrom) & 
            (binned_df['start'] < tile_end) & 
            (binned_df['end'] > tile_start)
        ]

        if overlapping_bins.empty:
            aggregated_data.append({
                "chrom": tile_chrom,
                "start": tile_start,
                "end": tile_end,
                "mean_power": 0,
                "max_energy_freq": 0,
                "variability": 0
            })
        else:
            overlapping_bins = overlapping_bins.copy()
            overlapping_bins['overlap_start'] = overlapping_bins['start'].clip(lower=tile_start)
            overlapping_bins['overlap_end'] = overlapping_bins['end'].clip(upper=tile_end)
            overlapping_bins['overlap_length'] = overlapping_bins['overlap_end'] - overlapping_bins['overlap_start']

            total_overlap_length = overlapping_bins['overlap_length'].sum()
            mean_power = (
                (overlapping_bins['mean_power'] * overlapping_bins['overlap_length']).sum() / total_overlap_length
                if total_overlap_length > 0 else 0
            )
            max_energy_freq = (
                (overlapping_bins['max_energy_freq'] * overlapping_bins['overlap_length']).sum() / total_overlap_length
                if total_overlap_length > 0 else 0
            )
            variability = (
                (overlapping_bins['variability'] * overlapping_bins['overlap_length']).sum() / total_overlap_length
                if total_overlap_length > 0 else 0
            )

            aggregated_data.append({
                "chrom": tile_chrom,
                "start": tile_start,
                "end": tile_end,
                "mean_power": mean_power,
                "max_energy_freq": max_energy_freq,
                "variability": variability
            })

    return pd.DataFrame(aggregated_data)

def process_and_aggregate_chromosome(chrom, group, df, window_size, overlap, fs, nperseg, noverlap):
    chrom_result, binned_data = compute_chromosome_spectrum(chrom, group, df, window_size, overlap, fs, nperseg, noverlap)
    binned_df = pd.DataFrame(binned_data)
    aggregated_data = aggregate_to_tiles(group, binned_df)
    return chrom, aggregated_data

def process_and_aggregate_parallel(tiles, df, window_size, overlap, fs, nperseg, noverlap):
    all_aggregated_data = []
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(process_and_aggregate_chromosome, chrom, group, df, window_size, overlap, fs, nperseg, noverlap): chrom
            for chrom, group in tiles.groupby("chrom")
        }
        for future in as_completed(futures):
            chrom = futures[future]
            try:
                chrom_result, aggregated_data = future.result()
                all_aggregated_data.extend(aggregated_data.to_dict('records'))
                print(f"Chromosome {chrom_result} processing and aggregation completed.")
            except Exception as e:
                print(f"Chromosome {chrom} processing failed with error: {e}")
    return pd.DataFrame(all_aggregated_data)

def run_stft_analysis(output, mutation_path, window_size, overlap, fs, nperseg, noverlap):
    tiles = output[['chrom', 'start', 'end']]   
    mut = pd.read_pickle(mutation_path)
    mut.columns = ['chrom', 'start', 'end', 'UUID','variantTypes', 'sample']
    mut['position'] = (mut['start'] + mut['end']) // 2
    adjusted_binned_df = process_and_aggregate_parallel(tiles, mut, window_size, overlap, fs, nperseg, noverlap)
    
    # Normalize values
    columns_to_scale = ['max_energy_freq', 'mean_power', 'variability']
    scaler = MinMaxScaler()
    adjusted_binned_df[columns_to_scale] = scaler.fit_transform(adjusted_binned_df[columns_to_scale])
    
    # Merge with original output
    merged_output = output.merge(
        adjusted_binned_df[['chrom', 'start', 'end', 'mean_power', 'max_energy_freq', 'variability']],
        on=['chrom', 'start', 'end'],
        how='left'
    )
    merged_output['position'] = (merged_output['start'] + merged_output['end']) // 2
    return merged_output
