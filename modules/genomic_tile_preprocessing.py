#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 10:27:50 2024

@author: silverflo
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
genomic_tile_preprocessing.py

Genomic tile creation and covariate integration module.
"""



import pandas as pd
from Bio import SeqIO
from concurrent.futures import ProcessPoolExecutor
import pybedtools

# Step 1: Load Mutations
def load_mutations(filepath):
    """
    Load mutation data from a pickle file and preprocess columns.
    """
    mut = pd.read_pickle(filepath)
    mut.sort_values(['Chromosome', 'Start'], inplace=True)
    mut.columns = ['chrom', 'start', 'end', 'UUID', 'variantTypes', 'sample']
    mut['position'] = (mut['start'] + mut['end']) // 2
    return mut

# Step 2: Create Tiles
def create_tiles(fasta_file, tile_start, tile_end):
    """
    Generate genomic tiles from a reference genome.
    """
    dna_string_set = SeqIO.to_dict(SeqIO.parse(fasta_file, "fasta"))
    for record_id in list(dna_string_set.keys()):
        dna_string_set['chr' + record_id] = dna_string_set.pop(record_id)
    seq_names = [name for name in dna_string_set.keys() if name != 'chrMT']
    seq_lengths = {seq: len(dna_string_set[seq].seq) for seq in seq_names}
    tiles = []
    for chrom in seq_names:
        chr_length = seq_lengths[chrom]
        tile_starts = range(1, chr_length, tile_start)
        tile_ends = [min(start + tile_end, chr_length) for start in tile_starts]
        for start, end in zip(tile_starts, tile_ends):
            tiles.append([chrom, start, end])
    print("Genomic tiles created successfully.")
    return pd.DataFrame(tiles, columns=['chrom', 'start', 'end'])

# Step 3: Process Chromosome Mutations
def process_chromosome(chrom_data):
    chrom, target_chrom, mut_chrom, idcap = chrom_data
    result = []
    for _, target in target_chrom.iterrows():
        overlapping_mut = mut_chrom[(mut_chrom['position'] >= target['start']) & 
                                    (mut_chrom['position'] <= target['end'])]
        capped_counts = overlapping_mut.groupby('sample').size().apply(lambda x: min(x, idcap))
        total_count = capped_counts.sum()
        result.append({
            'chrom': target['chrom'],
            'start': target['start'],
            'end': target['end'],
            'count': total_count
        })
    return result

def parallel_process(tiles_df, mut_df, idcap=1, max_workers=4):
    chrom_data_list = [
        (chrom, tiles_df[tiles_df['chrom'] == chrom], mut_df[mut_df['chrom'] == chrom], idcap)
        for chrom in tiles_df['chrom'].unique()
    ]
    result = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(process_chromosome, chrom_data_list)
        for res in results:
            result.extend(res)
    print("Mutation data aggregation completed.")
    return pd.DataFrame(result)

# Step 4: Change Data Types
def change_dtypes(df):
    df['chrom'] = df['chrom'].astype(str)
    df['start'] = df['start'].astype('int64')
    df['end'] = df['end'].astype('int64')
    df.sort_values(['chrom', 'start'], inplace=True)
    columns_to_keep = ['chrom', 'start', 'end']
    if 'score' in df.columns:
        df['score'] = df['score'].astype('float64')
        columns_to_keep.append('score')
    return df[columns_to_keep]

def tile_creation(fasta_file, mutation_file, tile_start, tile_end, idcap=1, max_workers=4):
    """
    High-level function to load mutations, create tiles, aggregate mutation data, and change data types.
    """
    # Load mutations
    mut_df = load_mutations(mutation_file)
    
    # Create tiles
    tiles_df = create_tiles(fasta_file, tile_start, tile_end)
    
    # Aggregate mutation data
    result_df = parallel_process(tiles_df, mut_df, idcap=idcap, max_workers=max_workers)
    
    # Change data types and format
    result_df = change_dtypes(result_df)
    
    return result_df

# Step 5: Process Covariate
def process_covariate(hypotheses, covariate_df, covariate_name, eligible_df):
    hypotheses_bed = pybedtools.BedTool.from_dataframe(hypotheses)
    covariate_bed = pybedtools.BedTool.from_dataframe(covariate_df)
    eligible_bed = pybedtools.BedTool.from_dataframe(eligible_df)
    intersected = hypotheses_bed.intersect(covariate_bed, wao=True)
    columns = ['chrom', 'start', 'end', 'chrom_c', 'start_c', 'end_c', 'overlap']
    if 'score' in covariate_df.columns:
        columns.insert(6, 'score')
    intersected_df = intersected.to_dataframe(names=columns)
    intersected_df = intersected_df[intersected_df['overlap'] > 0]
    intersected_df['cov_length'] = intersected_df['end_c'] - intersected_df['start_c']
    intersected_df = intersected_df[intersected_df['cov_length'] > 0]
    intersected_eligible = pybedtools.BedTool.from_dataframe(intersected_df[['chrom', 'start', 'end']]).intersect(eligible_bed, wao=True)
    intersected_eligible_df = intersected_eligible.to_dataframe(names=['chrom', 'start', 'end', 'chrom_e', 'start_e', 'end_e', 'eligible_overlap'])
    intersected_eligible_df = intersected_eligible_df[intersected_eligible_df['eligible_overlap'] > 0]
    intersected_df = intersected_df.merge(intersected_eligible_df, on=['chrom', 'start', 'end'], how='left')
    intersected_df['eligible_overlap'] = intersected_df['eligible_overlap'].fillna(0)
    intersected_df['eligible_bin_sum'] = intersected_df.groupby(['chrom', 'start', 'end'])['eligible_overlap'].transform('sum')
    intersected_df['eligible_weight'] = intersected_df['eligible_overlap'] / intersected_df['eligible_bin_sum']
    if 'score' in columns:
        intersected_df['score'] = pd.to_numeric(intersected_df['score'], errors='coerce').fillna(0)
        intersected_df['overlap_ratio'] = intersected_df['overlap'] / intersected_df['cov_length']
        intersected_df['weighted_score'] = intersected_df['overlap_ratio'] * intersected_df['score'] * intersected_df['eligible_weight']
        weighted_averages = intersected_df.groupby(['chrom', 'start', 'end'], group_keys=False).agg(
            weighted_score_sum=('weighted_score', 'sum')
        ).reset_index().rename(columns={'weighted_score_sum': covariate_name})
    else:
        intersected_df['covariate'] = intersected_df['overlap']
        intersected_df['weighted_covariate'] = intersected_df['covariate'] * intersected_df['eligible_weight']
        weighted_averages = intersected_df.groupby(['chrom', 'start', 'end'], group_keys=False).agg(
            weighted_covariate_sum=('weighted_covariate', 'sum')
        ).reset_index().rename(columns={'weighted_covariate_sum': covariate_name})
    print(f"Processing of covariate '{covariate_name}' completed.")
    return weighted_averages


def integrate_covariates(hypotheses_df, covariate_paths, eligible_path):
    # covariate_paths의 파일들을 읽어와 데이터프레임으로 저장
    covariate_dfs = {name: pd.read_pickle(path) for name, path in covariate_paths.items()}
    eligible = pd.read_pickle(eligible_path)
    eligible = change_dtypes(eligible)
    hypotheses_df = change_dtypes(hypotheses_df)
    
    # Process each covariate in parallel
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                process_covariate,
                hypotheses_df,
                change_dtypes(covariate_df),  # 직접 데이터프레임을 넘겨줌
                covariate_name,
                eligible
            )
            for covariate_name, covariate_df in covariate_dfs.items()  # covariate_df 사용
        ]
        results = [future.result() for future in futures]
    
    # Merge each covariate's results with the hypotheses_df
    for weighted_averages, covariate_name in zip(results, covariate_dfs.keys()):
        hypotheses_df = hypotheses_df.merge(weighted_averages, on=['chrom', 'start', 'end'], how='left')
        hypotheses_df[covariate_name] = hypotheses_df[covariate_name].fillna(0)
    
    return hypotheses_df


# High-Level Function
def integrate_genomic_tiles(fasta_file, mutation_file, covariate_paths, eligible_path, tile_start, tile_end, idcap=1, max_workers=4):
    hypotheses_df = tile_creation(fasta_file, mutation_file, tile_start, tile_end, idcap=idcap, max_workers=max_workers)
    output = integrate_covariates(hypotheses_df, covariate_paths, eligible_path)
    return output

