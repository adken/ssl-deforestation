# Self-Supervised Learning for Deforestation Mapping

Multi-modal fusion of **optical and radar satellite image time series** for deforestation mapping in the Amazon rainforest.

This repository contains the implementation developed for my **Master’s thesis**, which investigated self-supervised representation learning for Earth observation using a **TempCNN** backbone and **VICReg (Variance–Invariance–Covariance Regularization)**.

## Thesis Focus

The thesis explored whether self-supervised learning could be used to learn useful representations from multi-modal satellite image time series and transfer those representations to a downstream deforestation-mapping task.

The main components include:

* Data preprocessing
* Multi-modal satellite time-series dataset preparation
* TempCNN backbone
* VICReg self-supervised pretraining
* Optical and radar data fusion
* Downstream training and evaluation

## Repository Structure

```text
ssl-deforestation/
├── models/             # Models used during self-supervised pretraining
├── utils/              # Pretraining utilities
├── downstream/
│   ├── models/         # Models for the downstream task
│   ├── utils/          # Downstream utilities
│   └── train.py        # Downstream training
├── pretrain.py         # VICReg pretraining
└── README.md
```

## Workflow

### 1. Self-Supervised Pretraining

Multi-modal satellite image time series are encoded using a TempCNN backbone and trained with the VICReg objective to learn representations without relying on downstream class labels.

### 2. Downstream Deforestation Mapping

The learned representations are evaluated on a supervised deforestation-mapping task using the scripts contained in `downstream/`.

## Research Topics

* Self-supervised learning
* Earth observation
* Optical–radar data fusion
* Satellite image time-series analysis
* Deforestation mapping
* Representation learning
* VICReg
* TempCNN


**Adriko Kennedy**

