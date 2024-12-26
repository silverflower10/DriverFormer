#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 19:06:08 2024

@author: silverflo
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
preprocess_data.py

모든 로직:
1) integrate_genomic_tiles (-> returns integrated hypotheses & mut_df)
2) integrate_folds (takes integrated_hypotheses & mut_df with fold)
3) preprocess_data (MinMaxScaler, x_pos)
Now with a main() that parses command-line args:
 --config /path/to/config.json
 --output /path/to/output.pkl

@author: silverflo
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from Bio import SeqIO
import pybedtools
from sklearn.preprocessing import MinMaxScaler

########################
# 0. 유틸리티
########################

def load_config(config_path):
    """
    JSON 형식의 설정 파일을 로드해서 dict로 반환.
    """
    with open(config_path, "r") as f:
        cfg = json.load(f)
    return cfg

def ensure_bed_format(df):
    """
    df에 (chrom, start, end) 열이 있는지, dtype이 int인지 등 확인/보정하는 예시 함수.
    """
    if 'chrom' not in df.columns or 'start' not in df.columns or 'end' not in df.columns:
        raise ValueError("DataFrame must have columns: chrom, start, end.")
    df['chrom'] = df['chrom'].astype(str)
    df['start'] = df['start'].astype(int)
    df['end'] = df['end'].astype(int)
    return df

def load_mutations(filepath):
    """
    Load mutation data from a pickle file and preprocess columns -> (chrom, start, end, ...).
    여기에 fold 할당도 할 수 있음.
    """
    mut = pd.read_pickle(filepath)
    mut.sort_values(['Chromosome','Start'], inplace=True)
    mut.columns = ['chrom','start','end','UUID','variantTypes','sample']
    mut['position'] = (mut['start']+mut['end'])//2
    mut = ensure_bed_format(mut)
    return mut

########################
# 1. integrate_genomic_tiles
########################

def filter_hypotheses_by_eligible(hypotheses, eligible):
    """
    예시: hypotheses에서 'chrom', 'start', 'end'가 eligible과 아예 겹치지 않는 것 제거
    (실제 구현은 필요에 따라 다를 수 있음)
    """
    return hypotheses

def process_covariate(hypotheses_df, covariate_df, covariate_name):
    """
    hypotheses vs. covariate overlap 계산 -> weighted average 생성
    예: overlap 비례로 가중합, etc.
    """
    hypotheses_bed = pybedtools.BedTool.from_dataframe(hypotheses_df)
    covariate_bed = pybedtools.BedTool.from_dataframe(covariate_df)

    intersected = hypotheses_bed.intersect(covariate_bed, wao=True)

    columns = ['chrom','start','end','chrom_c','start_c','end_c','overlap']
    if 'score' in covariate_df.columns:
        columns.insert(6,'score')

    intersected_df = intersected.to_dataframe(names=columns)
    intersected_df = intersected_df[intersected_df['overlap']>0]

    if 'score' in columns:
        intersected_df['cov_length'] = intersected_df['end_c'] - intersected_df['start_c']
        intersected_df['score'] = pd.to_numeric(intersected_df['score'], errors='coerce').fillna(0)
        intersected_df['overlap_ratio'] = intersected_df['overlap'] / intersected_df['cov_length']
        intersected_df['weighted_score'] = intersected_df['score'] * intersected_df['overlap_ratio']
        weighted_averages = intersected_df.groupby(['chrom','start','end'])['weighted_score'].sum().reset_index()
        weighted_averages = weighted_averages.rename(columns={'weighted_score': covariate_name})
    else:
        overlap_sum = intersected_df.groupby(['chrom','start','end'])['overlap'].sum().reset_index()
        overlap_sum = overlap_sum.rename(columns={'overlap': covariate_name})
        weighted_averages = overlap_sum

    return ensure_bed_format(weighted_averages), covariate_name

def create_tiles(fasta_file, tile_start, tile_end):
    """
    참조 게놈 fasta파일 로드 -> 'chrMT' 제외 -> tile_start, tile_end 간격으로 tiles 생성
    """
    dna_string_set = SeqIO.to_dict(SeqIO.parse(fasta_file, "fasta"))
    for record_id in list(dna_string_set.keys()):
        dna_string_set['chr'+record_id] = dna_string_set.pop(record_id)
    seq_names = [name for name in dna_string_set.keys() if name!='chrMT']
    seq_lengths = {seq: len(dna_string_set[seq].seq) for seq in seq_names}

    tiles = []
    for chrom in seq_names:
        chr_length = seq_lengths[chrom]
        tile_starts = range(1, chr_length, tile_start)
        tile_ends = [min(start+tile_end, chr_length) for start in tile_starts]
        for s,e in zip(tile_starts, tile_ends):
            tiles.append([chrom, s, e])
    tiles_df = pd.DataFrame(tiles, columns=['chrom','start','end'])
    return ensure_bed_format(tiles_df)

def integrate_genomic_tiles(
    fasta_file, mutation_file, covariate_paths, eligible_path,
    tile_start=5000, tile_end=9999, idcap=1, max_workers=4
):
    tiles_df = create_tiles(fasta_file, tile_start, tile_end)
    mut_df = load_mutations(mutation_file)
    hypotheses_df = tiles_df.copy()

    # eligible
    eligible = pd.read_pickle(eligible_path)
    eligible = ensure_bed_format(eligible)
    filtered_hypotheses = filter_hypotheses_by_eligible(hypotheses_df, eligible)

    # eligible overlap
    hypotheses_bed = pybedtools.BedTool.from_dataframe(filtered_hypotheses)
    eligible_bed = pybedtools.BedTool.from_dataframe(eligible)
    intersected_eligible = hypotheses_bed.intersect(eligible_bed, wao=True)
    intersected_eligible_df = intersected_eligible.to_dataframe(
        names=['chrom','start','end','chrom_e','start_e','end_e','score','overlap'],
        dtype={'score':'float64'}
    )
    intersected_eligible_df = intersected_eligible_df[intersected_eligible_df['overlap']>0]
    eligible_sum = intersected_eligible_df.groupby(['chrom','start','end'])['overlap'].sum().reset_index(name='eligible')
    filtered_hypotheses = filtered_hypotheses.merge(eligible_sum, on=['chrom','start','end'], how='left')
    filtered_hypotheses['eligible'] = filtered_hypotheses['eligible'].fillna(0)

    # covariates
    covariate_dfs = {name: pd.read_pickle(path) for name, path in covariate_paths.items()}
    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = []
        for covariate_name, covariate_df_path in covariate_paths.items():
            covariate_df = pd.read_pickle(covariate_df_path)
            covariate_df = ensure_bed_format(covariate_df)
            futures.append(exe.submit(process_covariate, filtered_hypotheses, covariate_df, covariate_name))
        results = [f.result() for f in futures]

    for (weighted_averages, covariate_name) in results:
        filtered_hypotheses = filtered_hypotheses.merge(weighted_averages, on=['chrom','start','end'], how='left')
        filtered_hypotheses[covariate_name] = filtered_hypotheses[covariate_name].fillna(0)
    
    return filtered_hypotheses, mut_df


########################
# 2. integrate_folds
########################

def process_chromosome_fold(chrom_data):
    chrom, target_chrom, mut_chrom, idcap, fold_sample_nums_total, n = chrom_data
    result = []
    all_folds = np.arange(n)

    for idx, target in target_chrom.iterrows():
        overlapping_mut = mut_chrom[
            (mut_chrom['position']>=target['start']) &
            (mut_chrom['position']<=target['end'])
        ]
        if 'fold' not in overlapping_mut.columns:
            raise ValueError("mut 데이터프레임에 fold 컬럼이 없습니다. fold 할당 후 시도하세요.")

        fold_sample_counts = overlapping_mut.groupby(['fold','sample']).size()

        def cap_func(group):
            return group.apply(lambda x: min(x, idcap))
        fold_sample_counts_capped = fold_sample_counts.groupby(level='fold', group_keys=False).apply(cap_func)

        fold_counts = fold_sample_counts_capped.groupby(level='fold').sum().reset_index(name='count')

        fold_counts = fold_counts.set_index('fold').reindex(all_folds, fill_value=0).reset_index()
        fold_counts = pd.merge(fold_counts, fold_sample_nums_total, on='fold', how='left')
        fold_counts['fold_sample_count'] = fold_counts['fold_sample_count'].fillna(0)

        fold_counts['prob'] = 0.0
        valid_sample_mask = (fold_counts['fold_sample_count']>0)
        fold_counts.loc[valid_sample_mask, 'prob'] = (
            fold_counts.loc[valid_sample_mask, 'count'] / fold_counts.loc[valid_sample_mask,'fold_sample_count']
        )

        for _, row_fold in fold_counts.iterrows():
            result.append({
                'chrom': target['chrom'],
                'start': target['start'],
                'end': target['end'],
                'fold': row_fold['fold'],
                'count': row_fold['count'],
                'fold_sample_count': row_fold['fold_sample_count'],
                'prob': row_fold['prob']
            })
    return result

def parallel_process_fold(tiles_df, mut_df, idcap, n, max_workers=4):
    if 'fold' not in mut_df.columns:
        raise ValueError("mut 데이터프레임에 fold 컬럼이 없습니다. fold 할당 후 시도하세요.")
    
    fold_sample_nums_total = mut_df.groupby('fold')['sample'].nunique().reset_index(name='fold_sample_count')

    chrom_data_list = []
    for chrom_id in tiles_df['chrom'].unique():
        target_chrom = tiles_df[tiles_df['chrom']==chrom_id]
        mut_chrom = mut_df[mut_df['chrom']==chrom_id]
        chrom_data_list.append((chrom_id, target_chrom, mut_chrom, idcap, fold_sample_nums_total, n))

    result = []
    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        results = exe.map(process_chromosome_fold, chrom_data_list)
        for res in results:
            result.extend(res)
    return pd.DataFrame(result)

def integrate_folds(
    filtered_hypotheses, mut_df,
    fasta_file, tile_start, tile_end,
    n=5, idcap=1, max_workers=4
):
    tiles_df = create_tiles(fasta_file, tile_start, tile_end)
    fold_result_df = parallel_process_fold(tiles_df, mut_df, idcap, n, max_workers)

    merged_df = pd.merge(
        filtered_hypotheses,
        fold_result_df[['chrom','start','end','fold','count','fold_sample_count','prob']],
        on=['chrom','start','end'],
        how='left'
    )
    merged_df['count'] = merged_df['count'].fillna(0)
    merged_df = merged_df.dropna(subset=['fold'])

    return merged_df


########################
# 3. preprocess_data
########################

def preprocess_data(config):
    """
    1) integrate_genomic_tiles -> returns (filtered_hypotheses, mut_df)
    2) fold 할당 (단 한 번)
    3) integrate_folds -> merge fold info
    4) MinMaxScaler
    5) x_pos
    6) (옵션) covariate=0 or fold 없는 chrom 제거
    """
    print("Step 1: Integrating genomic tiles and covariates...")
    filtered_hypotheses, mut_df = integrate_genomic_tiles(
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

    # 2) 여기서 fold 할당
    print("Assigning fold to mutation df...")
    n = config["N_FOLDS"]
    samples = mut_df['sample'].unique()
    np.random.shuffle(samples)
    fold_ids = np.random.randint(low=0, high=n, size=len(samples))
    sample_to_fold = dict(zip(samples, fold_ids))
    mut_df['fold'] = mut_df['sample'].map(sample_to_fold)
    print(f"Fold assignment done. total folds={n}")

    # 3) integrate folds
    print("Step 2: Integrating folds...")
    output = integrate_folds(
        filtered_hypotheses=filtered_hypotheses,
        mut_df=mut_df,
        fasta_file=config["FASTA_FILE"],
        tile_start=config["TILE_START"],
        tile_end=config["TILE_END"],
        n=n,
        idcap=config["IDCAP"],
        max_workers=config["MAX_WORKERS"]
    )
    print("Step 2 completed: Fold integration done.")

    # 4) MinMaxScaler
    columns_to_normalize = config.get("COLUMNS_TO_NORMALIZE", [])
    if columns_to_normalize:
        scaler = MinMaxScaler()
        output[columns_to_normalize] = scaler.fit_transform(output[columns_to_normalize])
        print(f"Normalization completed for columns: {columns_to_normalize}")
    else:
        print("No columns specified for normalization.")

    # 5) x_pos
    output = output.sort_values(["chrom","start"])
    output["x_pos"] = output.groupby(["chrom","start","end"]).ngroup()
    print("Data preprocessing completed successfully.")

 

    return output


def main():
    parser = argparse.ArgumentParser(description="Preprocess genomic data")
    parser.add_argument("--config", type=str, required=True, help="Path to config file (JSON)")
    parser.add_argument("--output", type=str, required=True, help="Path to save preprocessed data")
    args = parser.parse_args()

    config = load_config(args.config)
    data = preprocess_data(config)

    data.to_pickle(args.output)
    print(f"Preprocessed data saved to {args.output}")


if __name__ == "__main__":
    main()
