#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 17:50:21 2024

@author: silverflo
"""

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from BORI import CustomDataset, analyze_posterior
from modules.bori_model import create_models
from BORI import pad_to_batch_size, clear_memory

def run_BORI(config, data, output_path):
    """Train and evaluate the model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare datasets and dataloaders
    features = config["FEATURES"]
    stft_features = config["STFT_FEATURES"]
    count = config["COUNT_COLUMN"]
    
    

    datasets = {
        chrom: CustomDataset(chrom_data, features, stft_features, count, overlap_size=config["BATCH_OVERLAP"], is_train=True)
        for chrom, chrom_data in data.groupby("chrom")
    }
    train_dataloaders = {
        chrom: DataLoader(dataset, batch_size=config["BATCH_SIZE"], shuffle=True)
        for chrom, dataset in datasets.items()
    }

    # Initialize models
    encoder, decoder = create_models(config, device)


    # Define optimizer
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=config["LEARNING_RATE"]
    )

    # Training loop
    for epoch in range(config["NUM_EPOCHS"]):
        print(f"\n===== epoch {epoch + 1}/{config['NUM_EPOCHS']} START =====")
        
        all_counts_tensors = []
        all_padding_masks = [] 
        all_feature = []
        all_position = []
        all_loss = []
    
        for chrom, dataloader in train_dataloaders.items():
            counts_per_chrom = []
            loss_per_chrom = 0
            feature_per_chrom = []
            position_per_chrom = []
            padding_masks_per_chrom = []
            
            for batch_all, batch_stft, batch_positions, batch_counts in dataloader:
                batch_all = batch_all.to(device)
                batch_stft = batch_stft.to(device)
                batch_counts = batch_counts.to(device)
                batch_positions = batch_positions.to(device)
        
              
                batch_all, padding_mask_all = pad_to_batch_size(batch_all, config['BATCH_SIZE'])
                batch_stft, padding_mask_stft = pad_to_batch_size(batch_stft, config['BATCH_SIZE'])
                batch_counts, padding_mask_counts = pad_to_batch_size(batch_counts, config['BATCH_SIZE'])
                batch_positions, padding_mask_positions = pad_to_batch_size(batch_positions, config['BATCH_SIZE'])

                valid_indices = padding_mask_all.bool()
                if valid_indices.sum() == 0:
                    continue 
        
                batch_all = batch_all[valid_indices]
                batch_stft = batch_stft[valid_indices]
                batch_counts = batch_counts[valid_indices]
                batch_positions = batch_positions[valid_indices]
        
                feature_per_chrom.append(batch_all.cpu().detach())
                position_per_chrom.append(batch_positions.cpu().detach())
                padding_masks_per_chrom.append(padding_mask_all.cpu())
        
                batch_counts = batch_counts.float()
                batch_all = batch_all.float()
                batch_stft = batch_stft.float().to(device)
                batch_positions = batch_positions.float()
   
                optimizer.zero_grad()
  
                # Encoder 출력 계산
                encoder_output = encoder(batch_all, batch_positions, device)

                # Decoder를 통해 pred_lambda, pred_alpha 예측
                pred_lambda, pred_alpha, class_probs = decoder(batch_stft.unsqueeze(-1), encoder_output, batch_positions, device)

                # Lambda와 Alpha 값 클램핑으로 안정성 확보
                pred_lambda = torch.clamp(pred_lambda, min=1e-6, max=1e6)
                pred_alpha = torch.clamp(pred_alpha, min=1e-6, max=1e6)
        
                # 3. 이진 분류 손실 계산 (count == 0)
                classification_loss = F.binary_cross_entropy(
                class_probs, (batch_counts == 0).float()
                ) * config["CLASSIFICATION_LOSS_WEIGHT"]

        
                # 데이터 포인트별로 0인 데이터를 범주형으로 처리
                zero_indices = (batch_counts == 0)
                nonzero_indices = (batch_counts != 0)


                # 데이터 포인트별로 과산포 여부 판단 후 적절한 분포 선택
                sampled_counts_list = []
                nb_loss_list = []
        
                for i in range(batch_counts.size(0)):  # 배치의 각 데이터 포인트에 대해 반복
                    count_value = batch_counts[i].item()  # 실제 count 값
                    lambda_value = pred_lambda[i].item()  # 예측된 lambda 값

           
                    if count_value > lambda_value:
                        # Gamma-Poisson(Negative Binomial) 사용
                        nb_dist = torch.distributions.NegativeBinomial(
                            total_count=torch.exp(pred_alpha[i]),  # Gamma-Poisson (Negative Binomial)
                            probs=torch.sigmoid(pred_lambda[i])
                        )
                    else:
                        # Poisson 분포 사용
                        nb_dist = torch.distributions.Poisson(
                            rate=torch.sigmoid(pred_lambda[i])  # Poisson
                        )
                        pred_alpha[i] = torch.zeros_like(pred_alpha[i])

                    # 각 데이터 포인트별 샘플링 및 log_prob 계산
                    sampled_counts = nb_dist.sample()
                    sampled_counts = sampled_counts.unsqueeze(0) 
                    sampled_counts_list.append(sampled_counts)
            


                    # log_prob에서 NaN 발생 여부 확인 및 처리
                    log_prob = nb_dist.log_prob(batch_counts[i])
                    if torch.isnan(log_prob).any():
                        log_prob = torch.zeros_like(log_prob)  # NaN 발생 시 0으로 처리
                    nb_loss_list.append(-log_prob.mean() * config['NB_LOSS_WEIGHT'])

                # 전체 배치에 대해 Reconstruction Loss 및 총 손실 계산
                sampled_counts = torch.stack(sampled_counts_list).to(device)
        

                # Reconstruction Loss 계산 (NaN 방지)
                recon_loss = F.smooth_l1_loss(sampled_counts, batch_counts)
                if torch.isnan(recon_loss).any():
                    recon_loss = torch.zeros_like(recon_loss)  # NaN 발생 시 0으로 처리
                recon_loss = recon_loss * config['RECON_LOSS_WEIGHT']

                nb_loss = torch.stack(nb_loss_list).mean() 
                if torch.isnan(nb_loss).any():
                    nb_loss = torch.zeros_like(nb_loss)

                loss_per_chrom += recon_loss.item()

                # 베이지안 정규화 및 KL 손실
                log_prior = decoder.transformer_decoder.log_prior()
                log_variational_posterior = decoder.transformer_decoder.log_variational_posterior()

                # KL divergence-like term for Bayesian regularization
                kl_loss = ((log_variational_posterior - log_prior) / len(dataloader)) * config['KL_LOSS_WEIGHT']
                if torch.isnan(kl_loss).any():
                    kl_loss = torch.zeros_like(kl_loss)  # NaN 발생 시 처리

                # Total loss combines reconstruction loss, negative binomial loss, and KL divergence
                total_loss = recon_loss + nb_loss + kl_loss + classification_loss
                total_loss.backward()

                # CPU로 데이터 이동 후 저장
                batch_counts_cpu = batch_counts.detach().cpu()
                counts_per_chrom.append(batch_counts_cpu)

                # Gradient Clipping 적용
                torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), max_norm=5.0)

                # optimizer step
                optimizer.step()

            # 크로모좀 단위로 에러와 재구성된 값 저장
            all_counts_tensors.append(torch.cat(counts_per_chrom, dim=0))
            all_loss.append(loss_per_chrom)

            clear_memory()
            torch.cuda.empty_cache()

        avg_loss_per_epoch = sum(all_loss) / len(all_loss)
        print(f"Average Loss for Epoch {epoch + 1}: {avg_loss_per_epoch:.4f}")


        clear_memory()
        torch.cuda.empty_cache()
    



    # Evaluation
    test_datasets = {
        chrom: CustomDataset(chrom_data, features, stft_features, count, overlap_size=0, is_train=False)
        for chrom, chrom_data in data.groupby("chrom")
    }
    test_dataloaders = {
        chrom: DataLoader(dataset, batch_size=config["BATCH_SIZE"])
        for chrom, dataset in test_datasets.items()
    }
    results = analyze_posterior(encoder, decoder, test_dataloaders, device)
    results.to_pickle(output_path)


    print("BORI is completed,,,")
    print() 
    print("Evaluation results:")
    print(results.head())

