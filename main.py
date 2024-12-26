#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 17:47:21 2024

@author: silverflo
"""

"""

메인 스크립트 예시:
1. 명령줄 인자로 config.json, data.pkl, output 경로 받기
2. config.json 불러오기 & Data 로드
3. 모델 초기화
4. 학습 루프
5. 결과 평가
6. 결과 저장 (선택)
"""

import os
import gc
import random
import numpy as np
import pandas as pd
import torch
import argparse
import json

from modules.data_loader import create_dataloader, CustomDataset
from modules.models import TransformerEncoderModel, TransformerDecoderModel
from modules.running import train_one_epoch
from modules.posterior_analysis import evaluate_model_with_posterior
from modules.utils import clear_memory


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Transformer VAE training with external config")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to the model config JSON file."
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the preprocessed data pickle (pkl) file."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        default="results.pkl",
        help="Path to save the final results (default: results.pkl)."
    )
    return parser.parse_args()


def main():
    ########################################
    # 0. argparse로 외부 인자 받기
    ########################################
    args = parse_arguments()
    config_path = args.config
    data_path = args.data
    output_path = args.output

    ########################################
    # 1. config.json 불러오기
    ########################################
    with open(config_path, "r") as f:
        config = json.load(f)

    # 예: config JSON 구조 (예시):
    # {
    #   "MODEL": {
    #       "E_MODEL": 64,
    #       "D_MODEL": 64,
    #       "NHEAD": 8,
    #       "NUM_ENCODER_LAYERS": 6,
    #       "NUM_DECODER_LAYERS": 6,
    #       "DIM_FEEDFORWARD": 256,
    #       "DROPOUT": 0.1
    #   },
    #   "TRAINING": {
    #       "LEARNING_RATE": 0.0001,
    #       "NUM_EPOCHS": 20,
    #       "BATCH_SIZE": 128,
    #       "RECON_LOSS_WEIGHT": 200,
    #       "KL_LOSS_WEIGHT": 0.05,
    #       "NB_LOSS_WEIGHT": 10,
    #       "CE_LOSS_WEIGHT": 200
    #   },
    #   "FEATURES": ["activchrom","hetchrom",...],
    #   "PROB_COLUMNS": ["prob","fold_sample_count"],
    #   "SEQUENCE_LENGTH": 99,
    #   "STRIDE": 50
    # }

    # 필요한 파라미터 꺼내오기
    e_model = config["MODEL"]["E_MODEL"]
    d_model = config["MODEL"]["D_MODEL"]
    nhead = config["MODEL"]["NHEAD"]
    num_encoder_layers = config["MODEL"]["NUM_ENCODER_LAYERS"]
    num_decoder_layers = config["MODEL"]["NUM_DECODER_LAYERS"]
    dim_feedforward = config["MODEL"]["DIM_FEEDFORWARD"]
    dropout = config["MODEL"]["DROPOUT"]

    learning_rate = config["TRAINING"]["LEARNING_RATE"]
    num_epochs = config["TRAINING"]["NUM_EPOCHS"]
    batch_size = config["TRAINING"]["BATCH_SIZE"]
    recon_loss_weight = config["TRAINING"]["RECON_LOSS_WEIGHT"]
    kl_loss_weight = config["TRAINING"]["KL_LOSS_WEIGHT"]
    nb_loss_weight = config["TRAINING"]["NB_LOSS_WEIGHT"]
    ce_loss_weight = config["TRAINING"]["CE_LOSS_WEIGHT"]

    features = config["FEATURES"]
    prob = config["PROB_COLUMNS"]
    sequence_length = config["SEQUENCE_LENGTH"]
    stride = config["STRIDE"]

    # (Optional) random seed
    seed_val = config.get("SEED", 42)
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)

    ########################################
    # 2. 데이터 불러오기
    ########################################
    df = pd.read_pickle(data_path)
    df.sort_values(['chrom', 'start'], inplace=True)


    # 환경변수 세팅 (필요 시)
    os.environ["OMP_NUM_THREADS"] = "30"
    os.environ["MKL_NUM_THREADS"] = "30"
    os.environ["NUMEXPR_NUM_THREADS"] = "30"
    os.environ["OPENBLAS_NUM_THREADS"] = "30"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "30"

    torch.set_num_threads(30)

    # GPU or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    unique_chroms = df['chrom'].unique()
    chrom_to_id = {c: i for i, c in enumerate(unique_chroms)}

    ########################################
    # 3. 모델 초기화
    ########################################
    transformer_encoder = TransformerEncoderModel(
        input_dim=len(features),
        e_model=e_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        num_chroms=len(chrom_to_id)
    ).to(device)

    transformer_decoder = TransformerDecoderModel(
        input_dim=1,
        feature_dim=d_model,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout
    ).to(device)

    optimizer = torch.optim.Adam(
        list(transformer_encoder.parameters()) + list(transformer_decoder.parameters()),
        lr=learning_rate
    )

    torch.autograd.set_detect_anomaly(True)

    ########################################
    # 4. fold별 DataLoader 구성
    ########################################
    fold_ids = df['fold'].unique()
    fold_dataloaders = {}
    group_cols = ['chrom','start','end']

    from torch.utils.data import ConcatDataset
    for fold_id in fold_ids:
        fold_data = df[df['fold'] == fold_id].sort_values(group_cols)
        datasets_per_chrom = []

        for chrom, chrom_data in fold_data.groupby('chrom'):
            distinct = chrom_data[group_cols].drop_duplicates()
            indices = []
            for _, grp_row in distinct.iterrows():
                grp_df = chrom_data[
                    (chrom_data['chrom'] == grp_row['chrom']) &
                    (chrom_data['start'] == grp_row['start']) &
                    (chrom_data['end'] == grp_row['end'])
                ]
                if len(grp_df) > 0:
                    idx = grp_df.sample(1).index[0]
                    indices.append(idx)
            chrom_data_unique = chrom_data.loc[indices].reset_index(drop=True)

            dataset = CustomDataset(
                chrom_data_unique, features, prob,
                sequence_length=sequence_length, stride=stride,
                is_train=True, chrom_to_id=chrom_to_id
            )
            datasets_per_chrom.append(dataset)

        if datasets_per_chrom:
            fold_dataset = ConcatDataset(datasets_per_chrom)
        else:
            fold_dataset = CustomDataset(
                pd.DataFrame(columns=fold_data.columns),
                features, prob,
                sequence_length=sequence_length, 
                stride=stride,
                is_train=True, chrom_to_id=chrom_to_id
            )

        fold_dataloader = torch.utils.data.DataLoader(
            fold_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False
        )
        fold_dataloaders[fold_id] = fold_dataloader

    ########################################
    # 5. 학습 루프
    ########################################
    for epoch in range(num_epochs):
        print(f"\n===== epoch {epoch+1}/{num_epochs} 시작 =====")
        avg_loss = train_one_epoch(
            fold_dataloaders,
            transformer_encoder,
            transformer_decoder,
            optimizer,
            device,
            nb_loss_weight=nb_loss_weight,
            kl_loss_weight=kl_loss_weight,
            recon_loss_weight=recon_loss_weight,
            ce_loss_weight=ce_loss_weight
        )
        print(f"Average Loss for Epoch {epoch+1}: {avg_loss:.4f}")
        clear_memory()

    clear_memory()

    ########################################
    # 6. 평가를 위한 df_combined 생성
    ########################################
    df_combined = (df.groupby(['chrom','start','end'], as_index=False)
                     .agg({'prob':'mean',
                           'fold_sample_count':'sum',
                           'activchrom':'mean','hetchrom':'mean',
                           'LINE':'mean','SINE':'mean','LTR':'mean',
                           'GC':'mean','Mapability':'mean','MCF7_DNAase':'mean'}))
    df_combined.sort_values(['chrom','start'], inplace=True)
    df_combined['x_pos'] = np.arange(len(df_combined))

    # 테스트 DataLoader
    test_dataset = CustomDataset(
        df_combined, features, prob,
        sequence_length=sequence_length, 
        stride=stride,
        is_train=False, chrom_to_id=chrom_to_id
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )
    test_dataloaders = {'all_data': test_dataloader}

    ########################################
    # 7. 모델 평가
    ########################################
    final_results = evaluate_model_with_posterior(
        transformer_encoder,
        transformer_decoder,
        test_dataloaders,
        device,
        chrom_to_id=chrom_to_id,
        batch_size=batch_size
    )
    print("Final results head:")
    print(final_results.head())

    # 병합 예시
    merged_df = pd.merge(
        df_combined,
        final_results[['x_pos','expected_count','actual_counts','actual_prob','pred_probability','pred_variance','posterior_probability']],
        on='x_pos', how='left'
    )
    print("Merged result:\n", merged_df.head())

    ########################################
    # 8. 결과 저장 (선택)
    ########################################
    # 예: output_path에 저장
    merged_df.to_pickle(output_path)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
