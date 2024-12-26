#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 15:02:19 2024

@author: silverflo
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
fold_preprocessing.py

1) Load mutation data (all_mutations.pkl) & feature (feature50000.pkl)
2) Assign fold randomly (n=5)
3) Create tiles from reference fasta
4) For each tile, count mutation per fold
5) Merge with feature, drop rows with fold=NaN
6) MinMaxScaler columns and save to all_df50000.pkl

Run: python fold_preprocessing.py
"""

import os
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from Bio import SeqIO
from sklearn.preprocessing import MinMaxScaler

def create_tiles(fasta_file, tile_start, tile_end):
    """
    참조 게놈 fasta 파일 로드 후, tile_start ~ tile_end 범위로 타일 생성.
    chrMT 제외, 'chr' 접두어 추가.
    """
    dna_string_set = SeqIO.to_dict(SeqIO.parse(fasta_file, "fasta"))
    
    # 염색체 이름 앞에 'chr' 추가
    for record_id in list(dna_string_set.keys()):
        dna_string_set['chr' + record_id] = dna_string_set.pop(record_id)

    # chrMT 제거
    seq_names = [name for name in dna_string_set.keys() if name not in ['chrMT']]

    # 염색체 길이 정보
    seq_lengths = {seq: len(dna_string_set[seq].seq) for seq in seq_names}

    # 타일 생성
    tiles = []
    for chr_id in seq_names:
        chr_length = seq_lengths[chr_id]
        tile_starts = range(1, chr_length, tile_start)
        tile_ends = [min(start + tile_end, chr_length) for start in tile_starts]
        for start, end in zip(tile_starts, tile_ends):
            tiles.append([chr_id, start, end])

    tiles_df = pd.DataFrame(tiles, columns=['chrom', 'start', 'end'])
    return tiles_df


def process_chromosome(chrom_data):
    """
    염색체별로 overlapping mutation을 fold별로 세어 'count', 'fold_sample_count', 'prob' 계산.
    """
    chrom, target_chrom, mut_chrom, idcap, fold_sample_nums_total, n = chrom_data
    result = []
    all_folds = np.arange(n)

    for idx, target in target_chrom.iterrows():
        overlapping_mut = mut_chrom[
            (mut_chrom['position'] >= target['start']) &
            (mut_chrom['position'] <= target['end'])
        ]
        
        if 'fold' not in overlapping_mut.columns:
            raise ValueError("mut 데이터프레임에 fold 컬럼이 없습니다. fold 할당 후 시도하세요.")

        # fold, sample 별 변이 카운트
        fold_sample_counts = overlapping_mut.groupby(['fold', 'sample']).size()

        # idcap 적용
        def cap_func(group):
            return group.apply(lambda x: min(x, idcap))
        fold_sample_counts_capped = fold_sample_counts.groupby(level='fold', group_keys=False).apply(cap_func)

        # fold별 총 변이 수
        fold_counts = fold_sample_counts_capped.groupby(level='fold').sum().reset_index(name='count')

        # 모든 fold 포함하도록 reindex
        fold_counts = fold_counts.set_index('fold').reindex(all_folds, fill_value=0).reset_index()

        # fold별 전체 샘플 수 merge
        fold_counts = pd.merge(fold_counts, fold_sample_nums_total, on='fold', how='left')
        fold_counts['fold_sample_count'] = fold_counts['fold_sample_count'].fillna(0)

        # prob 계산 = count / fold_sample_count
        fold_counts['prob'] = 0.0
        valid_sample_mask = (fold_counts['fold_sample_count'] > 0)
        fold_counts.loc[valid_sample_mask, 'prob'] = (
            fold_counts.loc[valid_sample_mask, 'count'] /
            fold_counts.loc[valid_sample_mask, 'fold_sample_count']
        )

        for _, row in fold_counts.iterrows():
            result.append({
                'chrom': target['chrom'],
                'start': target['start'],
                'end': target['end'],
                'fold': row['fold'],
                'count': row['count'],
                'fold_sample_count': row['fold_sample_count'],
                'prob': row['prob']
            })
    return result


def parallel_process(tiles_df, mut_df, idcap, n, max_workers=4):
    """
    모든 염색체에 대해 process_chromosome을 병렬 실행 -> 결과 DataFrame 반환.
    """
    if 'fold' not in mut_df.columns:
        raise ValueError("mutation df에 fold 컬럼이 없습니다. fold 할당 후 시도하세요.")

    # fold별 sample 수
    fold_sample_nums_total = mut_df.groupby('fold')['sample'].nunique().reset_index(name='fold_sample_count')

    # 염색체별 데이터 준비
    chrom_data_list = []
    for chrom_id in tiles_df['chrom'].unique():
        target_chrom = tiles_df[tiles_df['chrom'] == chrom_id]
        mut_chrom = mut_df[mut_df['chrom'] == chrom_id]
        chrom_data_list.append((chrom_id, target_chrom, mut_chrom, idcap, fold_sample_nums_total, n))

    result = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(process_chromosome, chrom_data_list)
        for res in results:
            result.extend(res)

    return pd.DataFrame(result)
