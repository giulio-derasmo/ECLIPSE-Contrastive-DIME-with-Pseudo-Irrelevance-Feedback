# ECLIPSE-Contrastive-DIME-with-Pseudo-Irrelevance-Feedback


## Overview
This repository contains the implementation and experimental setup for the paper "Eclipse: Contrastive Dimension Importance Estimation with Pseudo-Irrelevance Feedback". The paper introduces Eclipse, a novel method that enhances dense retrieval models by leveraging both relevant and irrelevant documents to improve retrieval performance.

Eclipse is built upon the Manifold Clustering Hypothesis, aiming to reduce noise in high-dimensional embeddings by estimating noisy dimensions from irrelevant documents, effectively filtering them to highlight relevant signals. The method improves upon traditional Dimension Importance Estimators (DIMEs) by incorporating pseudo-irrelevant feedback.

## Benchmarks
Eclipse has been tested on four benchmark datasets: TREC Deep Learning 2019, TREC Deep Learning 2020, DL-HARD 2021, TREC Robust 2004; and three different retrieaval models ANCE, Contriever, TAS-B. 

## Results

<img src="images/comparison-eclipse-baselines.jpeg" alt="RQ: Can pseudo-irrelevance feedback be leveraged to improve state-of-the-
art DIME approaches?" width="400"/>
