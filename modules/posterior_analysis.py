
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 14:18:50 2024

@author: silverflo
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 14:30:00 2024

@author: silverflo
"""
import torch
import pandas as pd
import torch.distributions as dist
import torch.nn.functional as F

from .utils import pad_to_batch_size

def evaluate_model_with_posterior(
    transformer_encoder, transformer_decoder, test_dataloaders,
    device, chrom_to_id, batch_size=128
):
    transformer_encoder.eval()
    transformer_decoder.eval()

    total_results = []
    id_to_chrom = {v: k for k, v in chrom_to_id.items()}

    eps_val = 1e-8

    with torch.no_grad():
        for name, dataloader in test_dataloaders.items():
            total_positions = []
            total_expected_counts = []
            total_actual_counts = []
            total_posterior_probs = []
            total_chrom_ids_list = []
            total_pred_probability = []
            total_pred_variance = []  # pred_variance를 담을 리스트
            total_actual_prob = []

            for batch_all, batch_prob, batch_positions, batch_chrom_ids in dataloader:
                batch_all = batch_all.to(device)
                batch_prob = batch_prob.to(device)
                batch_positions = batch_positions.to(device)
                batch_chrom_ids = batch_chrom_ids.to(device)

                # 패딩
                batch_all, padding_mask_all = pad_to_batch_size(batch_all, batch_size)
                batch_prob, padding_mask_prob = pad_to_batch_size(batch_prob, batch_size)
                batch_positions, padding_mask_positions = pad_to_batch_size(batch_positions, batch_size)
                batch_chrom_ids, padding_mask_chrom = pad_to_batch_size(batch_chrom_ids, batch_size)

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

                prob_values = batch_prob[:, :, 0]
                fold_sample_count = batch_prob[:, :, 1]

                # 인코더
                encoder_output = transformer_encoder(
                    batch_all, batch_positions, batch_chrom_ids, device
                )
                decoder_input = prob_values.unsqueeze(-1)

                # 디코더
                pred_probability, pred_variance, mu, logvar = transformer_decoder(
                    decoder_input, encoder_output, batch_positions, device
                )
                pred_probability = pred_probability.squeeze(-1)
                pred_variance = pred_variance.squeeze(-1)

                observed_count = torch.round(prob_values * fold_sample_count).long()

                mu_val = fold_sample_count * pred_probability
                sigma2_val = pred_variance

                poisson_mask = (sigma2_val < mu_val)
                nb_mask = ~poisson_mask

                mu_nb = mu_val[nb_mask]
                sigma2_nb = sigma2_val[nb_mask]
                r_val = (mu_nb**2) / (sigma2_nb - mu_nb + eps_val)
                p_val = r_val / (r_val + mu_nb + eps_val)

                r_full = torch.zeros_like(mu_val)
                p_full = torch.zeros_like(mu_val)

                r_full[nb_mask] = r_val
                p_full[nb_mask] = p_val

                observed_count_flat = observed_count.view(-1)
                mu_flat = mu_val.view(-1)

                nb_indices = nb_mask.view(-1).nonzero(as_tuple=True)[0]
                poisson_indices = poisson_mask.view(-1).nonzero(as_tuple=True)[0]

                log_prob_combined = torch.zeros_like(
                    observed_count_flat, dtype=torch.float, device=device
                )

                if nb_indices.numel() > 0:
                    r_nb_flat = r_full.view(-1)[nb_indices]
                    p_nb_flat = p_full.view(-1)[nb_indices]
                    obs_nb = observed_count_flat[nb_indices]
                    nb_dist = dist.NegativeBinomial(total_count=r_nb_flat, probs=p_nb_flat)
                    log_prob_nb = nb_dist.log_prob(obs_nb)
                    log_prob_nb[torch.isnan(log_prob_nb)] = 0.0
                    log_prob_combined[nb_indices] = log_prob_nb

                if poisson_indices.numel() > 0:
                    obs_poisson = observed_count_flat[poisson_indices]
                    mu_poisson = mu_flat[poisson_indices]
                    poisson_dist = dist.Poisson(mu_poisson)
                    log_prob_poisson = poisson_dist.log_prob(obs_poisson)
                    log_prob_poisson[torch.isnan(log_prob_poisson)] = 0.0
                    log_prob_combined[poisson_indices] = log_prob_poisson

                posterior_prob = torch.exp(log_prob_combined).view(mu_val.size())

                # 예측 결과들을 리스트에 저장
                total_expected_counts.append(mu_val.cpu())
                total_posterior_probs.append(posterior_prob.cpu())
                total_positions.append(batch_positions.cpu())
                total_actual_counts.append(observed_count.cpu())
                total_chrom_ids_list.append(batch_chrom_ids.cpu())

                total_pred_probability.append(pred_probability.cpu())
                total_pred_variance.append(pred_variance.cpu())  # variance 추가
                total_actual_prob.append(prob_values.cpu())

            # 각 fold나 전체 결과 취합
            if len(total_positions) == 0:
                continue

            total_positions = torch.cat(total_positions, dim=0)
            total_actual_counts = torch.cat(total_actual_counts, dim=0)
            total_expected_counts = torch.cat(total_expected_counts, dim=0)
            total_posterior_probs = torch.cat(total_posterior_probs, dim=0)
            total_chrom_ids = torch.cat(total_chrom_ids_list, dim=0)
            total_pred_probability = torch.cat(total_pred_probability, dim=0)
            total_pred_variance = torch.cat(total_pred_variance, dim=0)  # 합치기
            total_actual_prob = torch.cat(total_actual_prob, dim=0)

            chrom_ids_np = total_chrom_ids.numpy().squeeze()
            if chrom_ids_np.ndim == 0:
                chrom_ids_np = chrom_ids_np[None]
            chrom_series = pd.Series(chrom_ids_np.flatten()).map(id_to_chrom)

            chrom_results = pd.DataFrame({
                'chrom': chrom_series.values,
                'x_pos': total_positions.cpu().numpy().squeeze(),
                'expected_count': total_expected_counts.numpy().squeeze(),
                'actual_counts': total_actual_counts.numpy().squeeze(),
                'posterior_probability': total_posterior_probs.numpy().squeeze(),
                'pred_probability': total_pred_probability.numpy().squeeze(),
                'pred_variance': total_pred_variance.numpy().squeeze(),  # DataFrame에 추가
                'actual_prob': total_actual_prob.numpy().squeeze()
            })

            total_results.append(chrom_results)

    final_results = pd.concat(total_results, ignore_index=True)
    return final_results
