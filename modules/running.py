#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 08:34:01 2024

@author: silverflo
"""

import torch
import torch.nn.functional as F
import torch.distributions as dist
import random

from .utils import pad_to_batch_size

def train_one_epoch(
    fold_dataloaders, transformer_encoder, transformer_decoder, optimizer,
    device, nb_loss_weight, kl_loss_weight, recon_loss_weight, ce_loss_weight,
    eps_val=1e-8, max_norm=5.0
):
    """
    한 epoch 동안 모든 fold를 학습
    """
    fold_order = list(fold_dataloaders.keys())
    random.shuffle(fold_order)
    epoch_loss = 0.0
    total_steps = 0

    for fid in fold_order:
        dataloader = fold_dataloaders[fid]

        for batch_all, batch_prob, batch_positions, batch_chrom_ids in dataloader:
            batch_all = batch_all.to(device)
            batch_prob = batch_prob.to(device)
            batch_positions = batch_positions.to(device)
            batch_chrom_ids = batch_chrom_ids.to(device)

            # 패딩
            batch_all, padding_mask_all = pad_to_batch_size(batch_all, dataloader.batch_size)
            batch_prob, padding_mask_prob = pad_to_batch_size(batch_prob, dataloader.batch_size)
            batch_positions, padding_mask_positions = pad_to_batch_size(batch_positions, dataloader.batch_size)
            batch_chrom_ids, padding_mask_chrom = pad_to_batch_size(batch_chrom_ids, dataloader.batch_size)

            valid_indices = padding_mask_all.bool()
            if valid_indices.sum() == 0:
                continue

            batch_all = batch_all[valid_indices]
            batch_prob = batch_prob[valid_indices]
            batch_positions = batch_positions[valid_indices]
            batch_chrom_ids = batch_chrom_ids[valid_indices]

            batch_all = batch_all.float()
            batch_prob = batch_prob.float()
            batch_positions = batch_positions.float()

            optimizer.zero_grad()

            prob_values = batch_prob[:, :, 0]    # (B, S)
            fold_sample_count = batch_prob[:, :, 1]  # (B, S)
            B, S, F_dim = batch_all.size()

            # Encoder
            encoder_output = transformer_encoder(batch_all, batch_positions, batch_chrom_ids, device)
            decoder_input = prob_values.unsqueeze(-1)  # (B, S, 1)

            # Decoder
            pred_probability, pred_variance, mu, logvar = transformer_decoder(decoder_input, encoder_output, batch_positions, device)
            pred_probability = pred_probability.squeeze(-1)  # (B, S)
            pred_variance = pred_variance.squeeze(-1)        # (B, S)

            observed_count = torch.round(prob_values * fold_sample_count).long()

            mu_val = fold_sample_count * pred_probability
            sigma2_val = pred_variance

            poisson_mask = (sigma2_val < mu_val)
            nb_mask = ~poisson_mask

            mu_nb = mu_val[nb_mask]
            sigma2_nb = sigma2_val[nb_mask]

            r_val = (mu_nb**2) / (sigma2_nb - mu_nb + eps_val)
            p_val = r_val / (r_val + mu_nb + eps_val)

            r_flat = torch.zeros_like(mu_val.view(-1))
            p_flat = torch.zeros_like(mu_val.view(-1))

            r_flat[nb_mask.view(-1)] = r_val.view(-1)
            p_flat[nb_mask.view(-1)] = p_val.view(-1)

            observed_count_flat = observed_count.view(-1)

            nb_dist = dist.NegativeBinomial(
                total_count=r_flat[nb_mask.view(-1)],
                probs=p_flat[nb_mask.view(-1)]
            )
            log_prob_nb = nb_dist.log_prob(observed_count_flat[nb_mask.view(-1)])

            poisson_dist = dist.Poisson(mu_val[poisson_mask])
            log_prob_poisson = poisson_dist.log_prob(observed_count_flat[poisson_mask.view(-1)])

            log_prob_combined = torch.zeros_like(observed_count_flat, dtype=torch.float, device=device)
            log_prob_combined[nb_mask.view(-1)] = log_prob_nb
            log_prob_combined[poisson_mask.view(-1)] = log_prob_poisson

            log_prob_combined[torch.isnan(log_prob_combined)] = 0.0

            nb_loss = -log_prob_combined.mean() * nb_loss_weight
            if torch.isnan(nb_loss):
                nb_loss = torch.zeros_like(nb_loss)

            mu_flat = mu.view(-1, mu.size(-1))
            logvar_flat = logvar.view(-1, logvar.size(-1))

            kl_div = -0.5 * torch.mean(
                torch.sum(1 + logvar_flat - mu_flat.pow(2) - logvar_flat.exp(), dim=1)
            )
            kl_loss = kl_div * kl_loss_weight
            if torch.isnan(kl_loss):
                kl_loss = torch.zeros_like(kl_loss)

            mean = mu_val  # NB나 Poisson 모두 mean = mu_val
            recon_loss = F.mse_loss((mean+1).log(), (observed_count.float()+1).log()) * recon_loss_weight
            if torch.isnan(recon_loss):
                recon_loss = torch.zeros_like(recon_loss)

            prob_ce_loss = F.binary_cross_entropy(pred_probability, prob_values) * ce_loss_weight

            total_loss = nb_loss + kl_loss + recon_loss + prob_ce_loss
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                list(transformer_encoder.parameters()) + list(transformer_decoder.parameters()),
                max_norm=max_norm
            )
            optimizer.step()

            epoch_loss += total_loss.item()
            total_steps += 1

    return epoch_loss / total_steps if total_steps > 0 else 0.0

