

  #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 13:29:39 2024

@author: silverflo
"""




import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import get_sinusoidal_position_encoding

class TransformerEncoderModel(nn.Module):
    def __init__(self, input_dim, e_model, nhead,
                 num_encoder_layers, dim_feedforward,
                 dropout, num_chroms):
        super().__init__()
        self.e_model = e_model

        # 임베딩 레이어
        self.feature_embedding = nn.Linear(input_dim, e_model)
        self.chrom_embedding = nn.Embedding(num_chroms, e_model)
        self.activation = nn.ReLU()

        # Transformer Encoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=e_model, nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True
            ),
            num_layers=num_encoder_layers
        )

    def forward(self, feature_matrix, positions, chrom_ids, device):
        feature_matrix = feature_matrix.to(device)
        positions = positions.to(device)
        chrom_ids = chrom_ids.to(device)

        # Linear embedding
        embedded_feature = self.feature_embedding(feature_matrix)
        embedded_feature = self.activation(embedded_feature)

        # Chrom embedding
        chrom_emb = self.chrom_embedding(chrom_ids)

        # Positional encoding
        pos_enc = get_sinusoidal_position_encoding(positions, self.e_model).to(device)

        # 합산
        encoder_input = embedded_feature + chrom_emb + pos_enc

        # Transformer Encoder
        encoder_output = self.transformer_encoder(encoder_input)  # (B, S, e_model)
        return encoder_output


class TransformerDecoderModel(nn.Module):
    def __init__(self, input_dim, feature_dim, d_model,
                 nhead, num_encoder_layers, num_decoder_layers,
                 dim_feedforward, dropout):
        super().__init__()
        self.d_model = d_model
        self.feature_embedding = nn.Linear(input_dim, d_model)
        self.activation = nn.ReLU()

        # Transformer Decoder
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True
            ),
            num_layers=num_decoder_layers
        )

        # Variational parameters
        self.fc_mu = nn.Linear(d_model, d_model)
        self.fc_logvar = nn.Linear(d_model, d_model)

        # Probability & Variance
        self.fc_probability = nn.Linear(d_model, 1)
        self.fc_variance = nn.Linear(d_model, 1)

    def forward(self, count_input, encoder_output, positions, device):
        # (B, S, 1) → (B*S, 1)
        B, S, C = count_input.size()
        count_input = count_input.view(B*S, C)
        count_input = count_input.to(device)
        encoder_output = encoder_output.to(device)
        positions = positions.to(device)

        embedded_count = self.feature_embedding(count_input)
        embedded_count = self.activation(embedded_count)

        # 다시 (B, S, d_model)
        embedded_count = embedded_count.view(B, S, -1)

        # 혹시 encoder_output이 (B, d_model) 형태라면 unsqueeze(1)
        if encoder_output.dim() == 2:
            encoder_output = encoder_output.unsqueeze(1)

        # position embedding
        pos_enc = get_sinusoidal_position_encoding(positions, self.d_model).to(device)
        decoder_input = embedded_count + pos_enc

        # Cross-attention (TransformerDecoder)
        decoder_output = self.transformer_decoder(
            tgt=decoder_input, memory=encoder_output
        )

        # Variational
        mu = self.fc_mu(decoder_output)
        logvar = self.fc_logvar(decoder_output)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        # Predictions
        pred_probability = torch.sigmoid(self.fc_probability(z))
        pred_variance = F.softplus(self.fc_variance(z))

        return pred_probability, pred_variance, mu, logvar
