# Self-Supervised Learning for Deforestation Mapping

Multi-modal self-supervised representation learning from **Sentinel-1 radar** and **Sentinel-2 optical** satellite time series for downstream deforestation mapping.

This repository contains a corrected and refactored implementation of the code developed for a Master's thesis. The model uses separate **TempCNN** encoders for Sentinel-1 and Sentinel-2 and aligns their representations during pretraining with **VICReg (Variance-Invariance-Covariance Regularization)**.

## What this repository contains

- paired Sentinel-1/Sentinel-2 time-series loading;
- modality-specific TempCNN encoders;
- VICReg self-supervised pretraining;
- transfer of pretrained encoder weights to a downstream classifier;
- supervised, frozen-encoder, and end-to-end fine-tuning evaluation modes;
- deterministic train/validation/test splitting;
- checkpointing and TensorBoard logging;
- smoke tests for tensor shapes, pairing, freezing, and checkpoint transfer.

## Repository structure

```text
ssl-deforestation/
├── models/
│   ├── tempCNN.py          # shared TempCNN backbone
│   └── vicreg.py           # dual-encoder VICReg model
├── utils/
│   ├── dataset.py          # SSL paired S1/S2 dataset
│   ├── loss.py             # VICReg losses
│   └── utils.py            # LARS + learning-rate schedule
├── downstream/
│   ├── models/
│   │   └── tempCNN.py      # downstream fusion/classification model
│   ├── utils/
│   │   └── dataset.py      # labelled downstream dataset
│   └── train.py
├── tests/
├── pretrain.py
├── requirements.txt
└── README.md
```

The same `models.TempCNN` implementation is used during pretraining and downstream transfer so encoder definitions cannot silently drift apart.

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For development/testing:

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Expected data layout

### SSL pretraining

The pretraining directory must contain:

```text
/path/to/pretrain-data/
├── s1_stack.npy    # (N, T, 2)
└── s2_stack.npy    # (N, T, 10)
```

Each index is treated as a matched cross-modal observation:

```text
S1[i] <-> S2[i]
```

The loader converts samples to channels-first tensors expected by `Conv1d`:

```text
S1: (2, T)
S2: (10, T)
```

Sentinel-1 is z-score normalized per band over time. Sentinel-2 values are divided by 10,000.

### Downstream classification

```text
/path/to/downstream-data/
├── s1.npy          # (N, T, 2)
├── s2.npy          # (N, T, 10)
└── labels.npy      # (N,)
```

Downstream preprocessing matches pretraining.

## 1. VICReg pretraining

```bash
python pretrain.py \
  --path /path/to/pretrain-data \
  --batch_size 1024 \
  --epochs 100 \
  --device cuda
```

Important outputs include:

```text
checkpoints/best_checkpoint.pth
checkpoints/final_checkpoint.pth
logs/
```

The checkpoint contains the complete VICReg state, including `encoder_s1` and `encoder_s2`. Only those encoder weights are transferred to the downstream task.

## 2. Downstream evaluation

Run downstream training as a module from the repository root.

### Fully supervised baseline

```bash
python -m downstream.train \
  --mode supervised \
  --datapath /path/to/downstream-data
```

This initializes both encoders randomly and trains the full classifier end-to-end.

### Frozen SSL encoders

```bash
python -m downstream.train \
  --mode freeze \
  --checkpoint checkpoints/best_checkpoint.pth \
  --datapath /path/to/downstream-data
```

The pretrained S1/S2 encoders are loaded and kept fully frozen. Their BatchNorm statistics and dropout behavior are also held in evaluation mode; only the downstream head is optimized.

### Fine-tuned SSL encoders

```bash
python -m downstream.train \
  --mode fine-tuning \
  --checkpoint checkpoints/best_checkpoint.pth \
  --datapath /path/to/downstream-data
```

The same pretrained encoder weights are loaded, after which both encoders and the downstream head are optimized end-to-end.

## Evaluation modes

| Mode | Encoder initialization | Encoders trainable | Head trainable |
|---|---|---:|---:|
| `supervised` | Random | Yes | Yes |
| `freeze` | VICReg | No | Yes |
| `fine-tuning` | VICReg | Yes | Yes |

The default downstream split in this corrected implementation is **60% train / 20% validation / 20% test**, using a deterministic random seed.

## Compatibility with thesis-era checkpoints

The shared TempCNN intentionally retains the historical layer names:

```text
conv_bn_relu1
conv_bn_relu2
conv_bn_relu3
```

This keeps the encoder state-dict keys compatible with VICReg checkpoints produced by the original implementation, provided the checkpoint contains the corresponding `encoder_s1.*` and `encoder_s2.*` weights.

## Important reproducibility note

This repository is a **corrected implementation**, not a bit-for-bit reproduction of the original thesis code. Several issues in the original implementation affected the SSL/downstream comparison, including S1/S2 sample pairing, checkpoint transfer, frozen-encoder behavior, tensor orientation, and downstream loss/model selection.

The corrected implementation also standardizes preprocessing, uses shuffled SSL batches, and defaults to a 60/20/20 downstream split. Consequently, newly generated metrics should be treated as results from the corrected pipeline rather than assumed to reproduce previously reported numbers exactly.

## Research topics

- self-supervised learning;
- Earth observation;
- optical-radar fusion;
- satellite image time-series analysis;
- deforestation mapping;
- representation learning;
- VICReg;
- TempCNN.
