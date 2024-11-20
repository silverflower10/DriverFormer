#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 15:05:08 2024

@author: silverflo
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 15:05:08 2024

@author: silverflo
"""

from setuptools import setup, find_packages

setup(
    name="BORI",  # 패키지 이름
    version="0.1.0",  # 패키지 버전
    author="silverflo",
    author_email="your_email@example.com",  # 이메일 주소
    description="A Python package for Bayesian modeling and genomic data processing",
    long_description=open("README.md").read(),  # 프로젝트 설명 파일 (README.md)
    long_description_content_type="text/markdown",
    url="https://github.com/silverflower10/BORI",  # 프로젝트 GitHub URL
    packages=find_packages(),  # 패키지 자동 탐색
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.2.0",
        "torch>=1.8.0",
        "scipy>=1.6.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "statsmodels>=0.12.0",
        "hdbscan>=0.8.0",
        "pybedtools>=0.8.0",
        "biopython>=1.78",
        "scikit-learn>=0.24.0",
    ],
    extras_require={
        "dev": ["pytest", "flake8"],  # 개발 환경에 필요한 의존성
        "bayeformers": []  # 빈 리스트로 두고 아래 지침을 따름
    },
    python_requires=">=3.8",  # 최소 Python 버전
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
)
