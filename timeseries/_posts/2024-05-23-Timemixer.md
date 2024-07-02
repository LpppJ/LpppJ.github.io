---
layout: post
related_posts:
  _
title: 
description: >
  [ICLR 2024](https://openreview.net/pdf?id=7oLshfEIC2)
sitemap:
    changefreq: daily
    priority: 1.0
hide_last_modified: true
---

# TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting

## Abstract
- 지금까지는 plain decomposition and multiperiodicity analysis
- 본 논문에서는 novel view of multiscale-mixing (intuition : distinct patterns in different sampling scales)
- **TimeMixer** :  fully MLP-based architecture
  - Past-Decomposable-Mixing (PDM) : past extraction
    -  1) decomposition to multiscale series and 2) mixes the decomposed seasonal and trend components
  - Future-Multipredictor-Mixing (FMM) : future prediction
    - ensembles multiple predictors

## 1. Introduction
- challenge: complex and non-stationary nature of the real world system
- CNN : MICN(ICLR 2023), TimesNet(ICLR 2023), TCN(Soft Computing 2020)
- Transformer : Informer(AAAI 2021), Autoformer(NeurIPS 2021), 