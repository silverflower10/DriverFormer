#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 11:33:13 2024

@author: silverflo
"""

import pandas as pd
import torch
import sys
sys.path.append('/home/silverflo/BayeFormers')
from modules.utils import load_config, set_seed
from modules.running_withkl import run_BORI_withkl
from modules.data_preprocessing import preprocess_data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.gam.api import GLMGam, BSplines
import os
#%%
import sys

# 명령줄 인자 강제 설정
sys.argv = [
    "preprocess.py",  # 스크립트 이름
    "--config", "/home/silverflo/BORI/experiment/configs/data_config1_11_20.json",
    "--output", "/home/silverflo/BORI/experiment/data/input1.pkl"
]

# preprocess.py 실행
from preprocess import main

main()
#%%SFTF Plot

data = pd.read_pickle("/home/silverflo/BORI/experiment/data/input1.pkl")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# 데이터프레임의 컬럼 이름
columns_to_plot = ['max_energy_freq', 'mean_power', 'variability']
count_column = 'count'

# 데이터 로드
df = data

# count 데이터를 MinMaxScaler로 정규화
scaler = MinMaxScaler()
df[f'{count_column}_scaled'] = scaler.fit_transform(df[[count_column]])

# 각 컬럼과 count_scaled의 관계를 나타내는 산점도와 회귀선 그리기
for col in columns_to_plot:
    # 산점도와 회귀선
    plt.figure(figsize=(8, 6))
    sns.regplot(x=df[f'{count_column}_scaled'], y=df[col], scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'})
    
    # 회귀 모델 계산
    x = df[f'{count_column}_scaled'].dropna()
    y = df[col].dropna()
    coeff = np.polyfit(x, y, deg=1)  # 1차 회귀 (선형 회귀)
    regression_eq = f'y = {coeff[0]:.2f}x + {coeff[1]:.2f}'
    
    # 텍스트로 회귀선 방정식 추가
    plt.text(0.05, 0.95, regression_eq, fontsize=12, color='blue', transform=plt.gca().transAxes, 
             verticalalignment='top')
    
    # y축 범위 고정
    plt.ylim(0, 1.5)
    
    # 제목 및 축 레이블
    plt.title(f'Relationship between {col} and Scaled {count_column} with Regression Line', fontsize=14)
    plt.xlabel(f'{count_column} (scaled)', fontsize=12)
    plt.ylabel(col, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


#%%
file_name = "config9_11_29"
#%%
# JSON 설정 파일 경로
config_path = f"/home/silverflo/BORI/experiment/configs/model_{file_name}.json"
config = load_config(config_path)


# 데이터 로드
data_path = "/home/silverflo/BORI/experiment/data/input1.pkl"
data = pd.read_pickle(data_path)
data.sort_values(['chrom', 'start'], inplace=True)
data['x_pos'] = range(len(data))  # 추가적 필요한 컬럼

# 모델 실행
output_path = f"/home/silverflo/BORI/experiment/results/{file_name}_op.pkl"
set_seed(config["SEED"]) 
results = run_BORI_withkl(config, data, output_path)

#%%
results = pd.read_pickle(f"/home/silverflo/BORI/experiment/results/{file_name}_op.pkl")
#%%# 실제 QQ Plot
plt.figure(figsize=(10, 6))
plt.scatter(results['actual_counts'], results['expected_count'], alpha=0.3, s=8)
plt.plot([results['actual_counts'].min(), results['actual_counts'].max()], 
         [results['actual_counts'].min(), results['actual_counts'].max()], 
         color='red', linestyle='--')  # y=x 라인을 추가하여 비교 기준 제공
plt.title('Comparison of Actual Counts and Expected Counts')
plt.xlabel('Actual Counts')
plt.ylabel('Expected Counts')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))

# 실제 actual_count와 예상 expected_count 값에 노이즈 추가
noise_actual = np.random.normal(0, 0.5, size=len(results['actual_counts']))  # 실제 count에 대한 노이즈
noise_expected = np.random.normal(0, 0.5, size=len(results['expected_count']))  # 예상 count에 대한 노이즈

noisy_actual_counts = sorted(results['actual_counts'] + noise_actual)
noisy_expected_counts = sorted(results['expected_count'] + noise_expected)

plt.scatter(noisy_expected_counts, noisy_actual_counts, alpha=0.5, s=5)
plt.plot([min(noisy_expected_counts), max(noisy_expected_counts)], 
         [min(noisy_expected_counts), max(noisy_expected_counts)], 
         color='red', linestyle='--')  # y=x 라인을 추가하여 비교 기준 제공
plt.title('QQ Plot')
plt.xlabel('Expected Counts (sorted)')
plt.ylabel('Actual Counts (sorted)')
plt.grid(True)
plt.show()



#%%

results = results[results['chrom'] != 'chrY']

# 스무딩된 Chromosome Plot
chromosome_order = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 
                    'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 
                    'chr20', 'chr21', 'chr22', 'chrX', 'chrY']

results['chrom'] = pd.Categorical(results['chrom'], categories=chromosome_order, ordered=True)
results = results.sort_values(by=['chrom', 'start']).reset_index(drop=True)

results['x_pos'] = np.arange(len(results))

plt.figure(figsize=(25, 10))

x_pos_with_data = results['x_pos'].dropna()
x_min, x_max = x_pos_with_data.min(), x_pos_with_data.max()
plt.xlim(x_min, x_max)
plt.ylim(0, 100)

x = results['x_pos'].values
y_expected = results['expected_count'].values
y_actual_sum = results['actual_counts'].values

bs_expected = BSplines(x[:, np.newaxis], df=[500], degree=[5])
bs_actual_sum = BSplines(x[:, np.newaxis], df=[500], degree=[5])

model_expected = GLMGam(y_expected, smoother=bs_expected, alpha=100).fit()
model_actual_sum = GLMGam(y_actual_sum, smoother=bs_actual_sum, alpha=100).fit()

smoothed_expected = model_expected.predict(bs_expected.basis)
smoothed_actual_sum = model_actual_sum.predict(bs_actual_sum.basis)

plt.plot(x, smoothed_expected, color='blue', linewidth=1, label='Smoothed Expected Count')
plt.plot(x, smoothed_actual_sum, color='red', linewidth=1, label='Smoothed Actual Count Sum')

group_ends = results.groupby('chrom', observed=True)['x_pos'].max()
for end in group_ends:
    plt.axvline(x=end, color='grey', linestyle='--', linewidth=0.5)

chromosome_positions = results.groupby('chrom', observed=True)['x_pos'].mean()
for chrom, pos in chromosome_positions.items():
    plt.text(pos, plt.ylim()[0], chrom, horizontalalignment='center', verticalalignment='top', fontsize=10, rotation=45, transform=plt.gca().get_xaxis_transform())

plt.gca().set_xticks([])
plt.ylabel('Counts')
plt.title('Chromosome Plot with Smoothed Counts')
plt.legend()
plt.grid(True)
plt.show()
#%%manhaten plot
# Manhattan Plot with Highlight for Low Posterior Probability
plt.figure(figsize=(25, 10))

# 염색체 순서와 정렬
chromosome_order = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10',
                    'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19',
                    'chr20', 'chr21', 'chr22', 'chrX']

results['chrom'] = pd.Categorical(results['chrom'], categories=chromosome_order, ordered=True)
results = results.sort_values(by=['chrom', 'start']).reset_index(drop=True)

# x_pos 계산
results['x_pos'] = np.arange(len(results))

# 맨하탄 플롯
highlight_threshold = 0.01
highlight_color = 'red'
default_color = 'gray'

# 데이터 분리
highlight_data = results[results['posterior_probability'] < highlight_threshold]
normal_data = results[results['posterior_probability'] >= highlight_threshold]

# Normal data scatter
plt.scatter(
    normal_data['x_pos'],
    -np.log10(normal_data['posterior_probability']),
    color=default_color,
    s=5,
    alpha=0.7,
    label=f'Posterior >= {highlight_threshold}'
)

# Highlighted data scatter
plt.scatter(
    highlight_data['x_pos'],
    -np.log10(highlight_data['posterior_probability']),
    color=highlight_color,
    s=5,
    alpha=0.7,
    label=f'Posterior < {highlight_threshold}'
)

# x축 기준선과 염색체 경계선
group_ends = results.groupby('chrom', observed=True)['x_pos'].max()
for end in group_ends:
    plt.axvline(x=end, color='grey', linestyle='--', linewidth=0.5)

# x축 레이블(염색체 이름)
chromosome_positions = results.groupby('chrom', observed=True)['x_pos'].mean()
for chrom, pos in chromosome_positions.items():
    plt.text(pos, plt.ylim()[0] + 0.1, chrom, horizontalalignment='center', verticalalignment='bottom', fontsize=10)

# 축과 제목
plt.title('Manhattan Plot with Highlighted Posterior Probabilities', fontsize=16)
plt.xlabel('Chromosomes', fontsize=12)
plt.ylabel('-log10(Posterior Probability)', fontsize=12)
plt.legend(loc='upper right', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.xticks([])
plt.tight_layout()
plt.show()


