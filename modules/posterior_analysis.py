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


def analyze_posterior(transformer_encoder, transformer_decoder, dataloaders, device):
    """
    Evaluate the model by predicting counts and posterior probabilities.

    Args:
        transformer_encoder: The encoder model.
        transformer_decoder: The decoder model.
        dataloaders (dict): Dictionary of dataloaders, one for each chromosome.
        device (torch.device): The device to run the model on.

    Returns:
        pd.DataFrame: Evaluation results with posterior probabilities and expected counts.
    """
    total_results = []

    # Set the model to evaluation mode
    transformer_encoder.eval()
    transformer_decoder.eval()

    with torch.no_grad():
        for chrom, dataloader in dataloaders.items():
            total_positions = []
            total_actual_counts = []
            total_expected_counts = []
            total_posterior_probs = []
            total_class_predictions = []

            for batch_all, batch_stft, batch_positions, batch_counts in dataloader:
                # Move data to the specified device
                batch_all = batch_all.to(device)
                batch_stft = batch_stft.to(device)
                batch_positions = batch_positions.to(device)
                batch_counts = batch_counts.to(device)

                # Compute encoder output
                encoder_output = transformer_encoder(batch_all, batch_positions, device)

                # Decode lambda, alpha, and class probabilities
                pred_lambda, pred_alpha, class_probs = transformer_decoder(
                    batch_stft.unsqueeze(-1), encoder_output, batch_positions, device
                )

                # Binary classification: Predict whether count == 0
                predicted_classes = (class_probs > 0.5).long()
                total_class_predictions.append(predicted_classes.cpu())

                # Separate data for count == 0 and count != 0
                zero_indices = (predicted_classes == 1).squeeze()
                nonzero_indices = (predicted_classes == 0).squeeze()

                # Compute posterior probabilities and expected counts
                posterior_prob_list = []
                expected_count_list = []

                for i in range(len(pred_lambda)):
                    if zero_indices[i]:  # Predicted as count == 0
                        poisson_dist = torch.distributions.Poisson(rate=torch.exp(pred_lambda[i]))
                        expected_count = torch.tensor(0.0, device=device)
                        posterior_prob = torch.exp(poisson_dist.log_prob(batch_counts[i]))
                    else:  # Predicted as count != 0
                        if pred_alpha[i].item() > 0:
                            # Gamma-Poisson (Negative Binomial) distribution
                            nb_dist = torch.distributions.NegativeBinomial(
                                total_count=torch.exp(pred_alpha[i]),
                                probs=torch.sigmoid(pred_lambda[i])
                            )
                            expected_count = nb_dist.mean
                            posterior_prob = torch.exp(nb_dist.log_prob(batch_counts[i]))
                        else:
                            # Poisson distribution
                            poisson_dist = torch.distributions.Poisson(rate=torch.exp(pred_lambda[i]))
                            expected_count = poisson_dist.mean
                            posterior_prob = torch.exp(poisson_dist.log_prob(batch_counts[i]))

                    expected_count_list.append(expected_count.cpu())
                    posterior_prob_list.append(posterior_prob.cpu())

                # Store results for each position
                total_expected_counts.append(torch.stack(expected_count_list))
                total_positions.append(batch_positions.cpu())
                total_actual_counts.append(batch_counts.cpu())
                total_posterior_probs.append(torch.stack(posterior_prob_list))

            # Concatenate results across batches
            total_positions = torch.cat(total_positions, dim=0)
            total_actual_counts = torch.cat(total_actual_counts, dim=0)
            total_expected_counts = torch.cat(total_expected_counts, dim=0)
            total_posterior_probs = torch.cat(total_posterior_probs, dim=0)
            total_class_predictions = torch.cat(total_class_predictions, dim=0)

            # Combine results into a DataFrame
            chrom_results = pd.DataFrame({
                'chrom': chrom,
                'x_pos': total_positions.cpu().squeeze().numpy(),
                'expected_count': total_expected_counts.numpy(),
                'actual_counts': total_actual_counts.squeeze().numpy(),
                'posterior_probability': total_posterior_probs.squeeze().numpy(),
                'predicted_class': total_class_predictions.squeeze().numpy(),
            })

            total_results.append(chrom_results)

    # Combine results across all chromosomes
    final_results = pd.concat(total_results, ignore_index=True)

    return final_results
