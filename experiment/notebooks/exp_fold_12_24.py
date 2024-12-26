#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 11:33:13 2024

@author: silverflo
"""

#%%
from modules.utils import load_config, set_seed
import sys

# 명령줄 인자 강제 설정
sys.argv = [
    "preprocess.py",  # 스크립트 이름
    "--config", "/home/silverflo/BORI/experiment/configs/data_config.json",
    "--output", "/home/silverflo/BORI/experiment/data/input1.pkl"
]

# preprocess.py 실행
from preprocess import main

main()
#%%
import pandas as pd

data = pd.read_pickle("/home/silverflo/BORI/experiment/data/input1.pkl")

data = data[data['chrom'] != 'chrY']
data.to_pickle("/home/silverflo/BORI/experiment/data/input1.pkl")
#%%
import sys
from modules.utils import load_config

# 원하는 file_name
file_name = "config_fold1_12_24"

# JSON 설정 파일 경로
config_path = f"/home/silverflo/BORI/experiment/configs/model_{file_name}.json"
# 실제로 사용할 데이터 (전처리 결과)
data_path = "/home/silverflo/BORI/experiment/data/input1.pkl"
# 최종 결과(merged_df 등)를 저장할 경로
output_path = f"/home/silverflo/BORI/experiment/results/{file_name}_final.pkl"

# 이제 main.py의 argparse를 "모의"로 실행하기 위해 sys.argv 설정
sys.argv = [
    "main.py",  # 스크립트 이름
    "--config", config_path,
    "--data", data_path,
    "--output", output_path
]

# main.py 실행
# 주의: main.py 내부에 "if __name__ == '__main__': main()" 구조가 있다고 가정
import main
main.main()  # 직접 main() 함수 호출

#%%
results = pd.read_pickle(output_path)

#%%# 실제 QQ Plot
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.gam.api import GLMGam, BSplines
import pandas as pd

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
highlight_threshold = 0.001
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


#%%
import torch
import torch.distributions as dist
import pandas as pd

def compute_p_value_per_row(row, eps_val=1e-8):
    """
    각 row(하나의 구간)에 대해:
      1) 포아송 또는 음이항 분포를 생성
      2) 관측된 actual_counts까지의 CDF를 계산
      3) p-value = 1 - CDF(actual_counts)
    를 반환합니다.
    """
    mu_val = float(row['expected_count'])   # 예측 평균
    var_val = float(row['pred_variance'])   # 예측 분산
    obs_count = int(row['actual_counts'])   # 실제 관측 카운트

    # 음이항 vs 포아송 분기
    if var_val < mu_val:
        # -------- Poisson --------
        pois_dist = dist.Poisson(torch.tensor(mu_val, dtype=torch.float))
        cdf_val = 0.0
        for k in range(obs_count + 1):
            pmf_k = torch.exp(pois_dist.log_prob(torch.tensor(k, dtype=torch.float)))
            cdf_val += pmf_k.item()
        p_value = 1.0 - cdf_val
    else:
        # -------- Negative Binomial --------
        # r = mu^2 / (sigma^2 - mu)
        # p = r / (r + mu)
        r_val = (mu_val**2) / (var_val - mu_val + eps_val)
        r_val = max(r_val, 1e-8)

        p_val = r_val / (r_val + mu_val + eps_val)
        p_val = min(max(p_val, 1e-8), 1.0 - 1e-8)

        nb_dist = dist.NegativeBinomial(
            total_count=torch.tensor(r_val, dtype=torch.float),
            probs=torch.tensor(p_val, dtype=torch.float)
        )
        cdf_val = 0.0
        for k in range(obs_count + 1):
            pmf_k = torch.exp(nb_dist.log_prob(torch.tensor(k, dtype=torch.float)))
            cdf_val += pmf_k.item()
        p_value = 1.0 - cdf_val

    p_value = max(p_value, 0.0)  # 음수가 되지 않도록 보정
    return p_value


def compute_p_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    주어진 DataFrame에서 각 row에 대해 p-value를 계산하고,
    'p_value' 컬럼으로 추가하여 반환합니다.
    """
    p_values = df.apply(compute_p_value_per_row, axis=1)
    df['p_value'] = p_values  # 새로운 컬럼으로 추가
    return df

results_with_p = compute_p_values(results)

#%%
def plot_qq_log(df, pval_col='p_value',
                random_epsilon=True,
                random_range=(9, 11)):
    """
    df[pval_col] 에 있는 p-values를 -log10 변환하여,
    이론적 균등분포(Uniform(0,1))에도 -log10 변환을 적용한 뒤
    QQ plot을 그립니다.

    Parameters
    ----------
    df : pd.DataFrame
        p-values가 담긴 DataFrame
    pval_col : str
        df 내부에 존재하는 p-value 컬럼 이름
    random_epsilon : bool
        True이면, p-value가 0 근처에서 inf가 되는 문제를 방지하기 위해
        eps를 무작위로 생성해서 p-value에 더해줍니다.
    random_range : tuple
        random_epsilon=True인 경우 사용.
        eps = 10^(-U),  U ~ Uniform(random_range[0], random_range[1]).
        예: (9, 11) => eps ~ 10^(-9) ~ 10^(-11)

    Returns
    -------
    None
        QQ Plot을 그립니다.
    """

    # 1) p-values 정렬
    pvals = np.sort(df[pval_col].values)
    n = len(pvals)

    if random_epsilon:
        # (1) U ~ Uniform(a, b)
        # (2) eps_i = 10^(-U_i)
        # => pvals_i += eps_i
        u = np.random.uniform(random_range[0], random_range[1], size=n)
        eps_arr = 10 ** (-u)
        pvals = pvals + eps_arr
    else:
        # 고정 eps 사용
        eps = 1e-10
        pvals = pvals + eps

    # 2) pvals가 [0,1] 범위 벗어나지 않도록 클램핑 (특히 1보다 커지는 경우 방지)
    #    1e-300은 매우 작은 수치로, log10 계산 시 -log10(1e-300)=300
    pvals = np.clip(pvals, 1e-300, 1.0)

    # 3) 이론적 quantile (0~1) 계산 및 클램핑
    theoretical = (np.arange(1, n+1)) / (n+1)
    theoretical = np.clip(theoretical, 1e-300, 1.0)

    # 4) -log10 변환
    observed = -np.log10(pvals)
    expected = -np.log10(theoretical)

    # 5) QQ plot
    plt.figure(figsize=(10,10))
    plt.plot(expected, observed, 'o', label='Observed -log10 p-values')

    # y = x 대각선 그리기
    max_val = max(expected.max(), observed.max())

    plt.plot([0, max_val], [0, max_val], 'r--', label='y = x')
    plt.xlabel('Expected -log10 p-values (Uniform(0,1))')
    plt.ylabel('Observed -log10 p-values')
    plt.title('-log10 QQ Plot of p-values')
    plt.legend()
    plt.show()
    
plot_qq_log(results_with_p, pval_col='p_value', random_epsilon=True)

