#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 13:29:39 2024

@author: silverflo
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoderModel(nn.Module):
    def __init__(self, input_dim, e_model, nhead, num_encoder_layers, dim_feedforward, dropout):
        super(TransformerEncoderModel, self).__init__()
        
        # Feature 값을 임베딩하기 위한 Linear layer (입력: feature matrix)
        self.feature_embedding = nn.Linear(input_dim, e_model)
        self.activation = nn.ReLU()
        
        # Transformer 인코더 정의
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(e_model, nhead, dim_feedforward, dropout, batch_first=True),
            num_encoder_layers
        )

    def forward(self, feature_matrix, positions, device):
        # Feature 임베딩
        feature_matrix = feature_matrix.to(device)
        embedded_feature = self.feature_embedding(feature_matrix)
        embedded_feature = self.activation(embedded_feature)

        # Transformer 인코더 출력
        encoder_output = self.transformer_encoder(embedded_feature)
        return encoder_output  # [batch_size, seq_len, d_model]
    
class TransformerDecoderModel(nn.Module):
    def __init__(self, input_dim, feature_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout):
        super(TransformerDecoderModel, self).__init__()

        # Feature 값을 받아 임베딩
        self.stft_embedding = nn.Linear(input_dim, d_model)  # input_dim을 받아 d_model로 임베딩
        self.activation = nn.ReLU()

        # Transformer Decoder: Cross-attention을 수행하는 디코더
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True),
            num_decoder_layers
        )

        # 최종적으로 expected count 값을 예측하기 위한 MLP
        # Lambda와 Alpha를 예측하기 위해 두 개의 출력 (pred_lambda, pred_alpha)
        self.fc_output = nn.Linear(d_model, 2)
        self.classifier = nn.Linear(d_model, 1)
        self.d_model = d_model

    def forward(self, stft_input, encoder_output, positions, device):
        # 모든 입력을 GPU/CPU로 이동
        stft_input = stft_input.to(device)
        encoder_output = encoder_output.to(device)
        positions = positions.to(device)

        # Counts 임베딩
        stft_input = stft_input.view(stft_input.size(0), -1)  # (batch_size, 1) -> (batch_size, d_model)
        embedded_stft = self.stft_embedding(stft_input)
        embedded_stft = self.activation(embedded_stft)

        # `encoder_output` 차원이 (batch_size, seq_len, d_model)이 되어야 하므로, 필요 시 차원 추가
        if encoder_output.dim() == 2:
            encoder_output = encoder_output.unsqueeze(1)  # (batch_size, d_model) -> (batch_size, 1, d_model)

        # Cross-attention 수행
        decoder_output = self.transformer_decoder(tgt=embedded_stft.unsqueeze(1), memory=encoder_output)

        # 최종적으로 MLP를 통해 pred_lambda와 pred_alpha 예측
        pred_output = self.fc_output(decoder_output.mean(dim=1))  # (batch_size, 2)

        # pred_lambda와 pred_alpha로 분리
        pred_lambda = F.softplus(pred_output[:, 0])  # Lambda는 양수로 제한
        pred_alpha = F.softplus(pred_output[:, 1])  # Alpha도 양수로 제한
        
        class_logits = self.classifier(decoder_output.mean(dim=1))  # (batch_size, 1)
        class_probs = torch.sigmoid(class_logits)  # Class probabilities

        return pred_lambda, pred_alpha, class_probs
