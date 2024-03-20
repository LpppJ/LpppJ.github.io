---
layout: post
related_posts:
  _
title: 
description: >
  [IJCAI 2023](https://arxiv.org/pdf/2202.07125.pdf)
sitemap:
    changefreq: daily
    priority: 1.0
hide_last_modified: true
---

# (Survey paper) Transformers in Time Series: A Survey (IJCAI 2023)

## Abstract
- Transformer는 long-range dependencies and interactions를 학습할 수 있다.
- 본 논문에서는 Network structure 관점에서 Transformer를 TS forecasting에 사용하기 위해 어떤 adaptaion and modification을 했는지 알아보고
- Application 관점에서 forecasting, anomaly detection, and classification을 포함한 task에 대해 얼마나 잘 작동하는지 알아본다.
- 마지막으로 future direction을 제시한다.

## 1. Introduction
- Transformer : ability for long-range dependencies and interactions in sequential data
- Time series : How to effectively model long-range and short-range temporal dependency and capture seasonality simultaneously ?
- Network modification 관점 : low-level(i.e. module)부터 high-level(i.e. architecture)
- Application 관점 : summarize Transformer for forecasting, anomaly detection, and classification

## 2. Preliminaries of the Transformer
### 2.1. Vanilla Transformer
- Encoder : a multi- head self-attention module and a position-wise feed-forward network
- Decoder : cross-attention models between the multi-head self-attention module and the position-wise feed-forward network
### 2.2. Input Encoding and Positional Encoding
- No recurrence, instead positional encoding
- $$PE(t)_i= \begin{cases}\sin \left(\omega_i t\right) & i \% 2=0 \\ \cos \left(\omega_i t\right) & i \% 2=1\end{cases}$$, $$\omega_i$$ is the hand-crafted frequency for each dim