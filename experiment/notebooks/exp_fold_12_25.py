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
file_name = "config_fold1_12_25"

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
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 08:34:01 2024

@author: silverflo (Modified to use SciPy)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import poisson, nbinom


def compute_p_value_per_row(row, eps_val=1e-8, min_p=1e-50):
    """
    각 row(하나의 구간)에 대해:
      1) Poisson 또는 Negative Binomial 분포(CDF)를 SciPy로 계산
      2) 관측된 actual_counts까지의 CDF를 얻어
      3) p-value = 1 - CDF(actual_counts)
    를 반환합니다.

    Parameters
    ----------
    row : pd.Series
        'expected_count', 'pred_variance', 'actual_counts' 등을 포함한 한 행
    eps_val : float
        수치 안정성용 작은 값 (분모 보호)
    min_p : float
        p-value가 너무 작아 0으로 깎이는 것을 방지하기 위한 최소값 클램핑
        예) 1e-50, 1e-300, etc.
    """
    mu_val = float(row['expected_count'])   # 예측 평균
    var_val = float(row['pred_variance'])   # 예측 분산
    obs_count = int(row['actual_counts'])   # 실제 관측 카운트

    # Poisson vs Negative Binomial 분기
    if var_val < mu_val:
        # ----- Poisson -----
        # mu = mu_val
        cdf_val = poisson.cdf(obs_count, mu_val)
        p_value = 1.0 - cdf_val
    else:
        # ----- Negative Binomial -----
        # r = mu^2 / (sigma^2 - mu)
        r_val = (mu_val**2) / (var_val - mu_val + eps_val)
        r_val = max(r_val, 1e-8)

        # p = r / (r + mu)
        p_dist = r_val / (r_val + mu_val + eps_val)
        p_dist = min(max(p_dist, 1e-8), 1.0 - 1e-8)

        # SciPy nbinom: shape = r, prob = p
        cdf_val = nbinom.cdf(obs_count, r_val, p_dist)
        p_value = 1.0 - cdf_val

    # 0 이하인 경우 보정
    if p_value < 0.0:
        p_value = 0.0
    # 매우 작은 p-value를 min_p로 클램핑
    if p_value < min_p:
        p_value = min_p

    return p_value


def compute_p_values(df: pd.DataFrame,
                     expected_col='expected_count',
                     var_col='pred_variance',
                     count_col='actual_counts',
                     min_p=1e-50) -> pd.DataFrame:
    """
    주어진 DataFrame에서 각 row에 대해 p-value를 계산하고,
    'p_value' 컬럼으로 추가하여 반환합니다.

    Parameters
    ----------
    df : pd.DataFrame
        'expected_count', 'pred_variance', 'actual_counts' 등을 포함한 DataFrame
    expected_col : str
        예측 평균값 컬럼 이름
    var_col : str
        예측 분산값 컬럼 이름
    count_col : str
        실제 관측 카운트 컬럼 이름
    min_p : float
        p-value 최소값 클램핑

    Returns
    -------
    pd.DataFrame
        p_value 컬럼이 추가된 DataFrame
    """
    def row_func(row):
        row_dict = {
            'expected_count': row[expected_col],
            'pred_variance': row[var_col],
            'actual_counts': row[count_col]
        }
        return compute_p_value_per_row(row_dict, min_p=min_p)

    p_values = df.apply(row_func, axis=1)
    df['p_value'] = p_values
    return df


def plot_qq_log(df: pd.DataFrame,
                pval_col='p_value',
                random_epsilon=True,
                random_range=(9, 11),
                eps_fixed=1e-10):
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
    eps_fixed : float
        random_epsilon=False일 때 사용할 고정 eps 값.

    Returns
    -------
    None
        QQ Plot을 그립니다.
    """
    # p-values
    pvals = df[pval_col].values.copy()
    pvals = np.sort(pvals)
    n = len(pvals)

    # 무작위 eps 또는 고정 eps 추가
    if random_epsilon:
        u = np.random.uniform(random_range[0], random_range[1], size=n)
        eps_arr = 10.0 ** (-u)
        pvals += eps_arr
    else:
        pvals += eps_fixed

    # [1e-300, 1] 범위로 클램핑
    pvals = np.clip(pvals, 1e-300, 1.0)

    # 이론적 균등분포 (0~1) quantile
    uniform_q = (np.arange(1, n+1)) / (n+1)
    uniform_q = np.clip(uniform_q, 1e-300, 1.0)

    # -log10 변환
    observed = -np.log10(pvals)
    expected = -np.log10(uniform_q)

    # QQ plot
    plt.figure(figsize=(6,6))
    plt.scatter(expected, observed, s=8, alpha=0.7, label='Observed -log10 p-values')

    # y = x 대각선
    max_val = max(observed.max(), expected.max())
    plt.plot([0, max_val], [0, max_val], 'r--', label='y = x')

    plt.xlabel('Expected -log10(p) (Uniform(0,1))')
    plt.ylabel('Observed -log10(p)')
    plt.title('QQ Plot with SciPy-based p-values')
    plt.legend()
    plt.show()



results_with_p = compute_p_values(results, 
                                  expected_col='expected_count', 
                                  var_col='pred_variance', 
                                  count_col='actual_counts', 
                                  min_p=1e-50)

plot_qq_log(results_with_p, pval_col='p_value',
            random_epsilon=True, random_range=(9,100),
            eps_fixed=1e-10)

