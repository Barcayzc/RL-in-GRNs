# Reinforcement Learning for Control in Gene Regulatory Networks

This repository contains a reimplementation of the paper:

> **Deep Reinforcement Learning for Stabilization of Large-Scale Probabilistic Boolean Networks**  
> Sotiris Moschoyiannis, Evangelos Chatzaroulas, Vytenis Šliogeris, Yuhu Wu  
> [IEEE TNNLS, 2022](https://ieeexplore.ieee.org/document/9999487)

---

##  Project Overview

This project applies **Deep Reinforcement Learning (DRL)** to the problem of **controlling gene regulatory networks (GRNs)** modeled as **Probabilistic Boolean Networks (PBNs)**. The task is to guide a biological system from an undesirable state (e.g., cancer-like) to a desirable target attractor via minimal gene interventions.

Our approach uses:
- **Double Deep Q-Networks (DDQN)**  
- **Prioritized Experience Replay (PER)**
- A **model-free RL setting**: no need for full transition matrices

---

##  Dataset

We use real single-cell RNA-seq data from mouse pancreatic epithelial cells:

- **GSE132188**: ~11,000 cells × ~28,000 genes  
- Preprocessing pipeline includes:
  - Quality filtering, log normalization
  - Gene selection by stage-discriminative variance
  - Binarization via k-means (k=2) per gene
  - Boolean function inference using COD
  - Construction of a Gym-compatible PBN environment

 Data source: [GEO Accession GSE132188](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE132188)

---