# MSAR

[![arXiv](https://img.shields.io/badge/arXiv-2611.20714-b31b1b.svg)](https://arxiv.org/abs/2611.20714)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Next-Scale Autoregressive Forecasting for Time Series via Modular Multi-Scale Decoupling** (NeurIPS 2026)

## Overview

**MSAR** is a modular framework for time series forecasting that generates
predictions from coarse to fine temporal scales. A forecasting module first
predicts the coarsest scale, then imputation modules progressively fill in
finer-scale values using the observed history and coarser predictions.

- **Aligned temporal scales.** Each module operates on inputs and outputs at the same resolution.
- **Coarse-to-fine prediction.** Predictions at one scale guide the next scale.
- **Modular backbones.** Forecasting and imputation modules share a backbone family and can use different hidden widths.

<p align="center">
  <img src="assets/framework.png" alt="MSAR: scale-aligned forecasting followed by coarse-to-fine imputation" width="96%">
</p>

**Supported backbones:** PatchTST, DLinear, iTransformer, TimeMixer, TimesNet,
SimpleTM, and DUET. Start with the **PatchTST on ETTh1** example below.

## Getting Started

### 1. Installation

The experiment environment uses Python 3.12 and PyTorch 2.8.0. A CUDA-capable GPU
is recommended for training.

```bash
git clone https://github.com/Oryxo/MSAR.git
cd MSAR

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For a specific CUDA build, install the matching PyTorch 2.8.0 package using the
[official installation instructions](https://pytorch.org/get-started/previous-versions/)
before installing `requirements.txt`. Check GPU availability with:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### 2. Prepare the data

Download the preprocessed benchmark datasets provided by
[Time-Series-Library](https://github.com/thuml/Time-Series-Library) from
[Google Drive](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing),
[Baidu Drive](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy), or
[Hugging Face](https://huggingface.co/datasets/thuml/Time-Series-Library).
Extract the datasets into `./dataset/`, preserving their subdirectories:

```text
dataset/
├── ETT-small/
│   ├── ETTh1.csv
│   ├── ETTh2.csv
│   ├── ETTm1.csv
│   └── ETTm2.csv
├── electricity/electricity.csv
├── exchange_rate/exchange_rate.csv
├── traffic/traffic.csv
├── weather/weather.csv
└── illness/national_illness.csv
```

Use the matching `--data` name for ETT datasets. For Electricity, Exchange,
Traffic, Weather, ILI, or other CSV datasets, use `--data custom` and specify
`--root_path`, `--data_path`, `--target`, and `--freq`. CSV files should contain
a `date` column and numeric time series columns. Use `--freq t` for ETTm data.
Only download the datasets needed for your experiments.

### 3. Train and evaluate

Run the PatchTST + MSAR example for a prediction length of 96:

```bash
PRED_LENS=96 bash scripts/PatchTST/ETTh1.sh
```

Run all four prediction lengths (`96`, `192`, `336`, and `720`):

```bash
bash scripts/PatchTST/ETTh1.sh
```

The script trains the scale-specific modules, performs joint training with the
coarse forecasting module frozen, and evaluates the final model on the test set.
It uses **compact imputation by default**: `d_model=16` and `d_ff=32`. The
forecasting module retains `d_model=512` and `d_ff=2048`.

You can select a GPU, dataset directory, prediction lengths, or random seed:

```bash
GPU=0 ROOT_PATH=./dataset/ETT-small PRED_LENS="96 192" SEED=2021 \
  bash scripts/PatchTST/ETTh1.sh
```

### 4. Run a baseline

Use the independent Time-Series-Library forecasting pipeline for a baseline:

```bash
python run.py --task_name long_term_forecast \
  --model PatchTST --data ETTh1 \
  --root_path ./dataset/ETT-small --data_path ETTh1.csv \
  --seq_len 336 --label_len 0 --pred_len 96
```

This path trains one forecasting model with **MSE by default**, without MSAR
teacher-forcing or joint training. Pass `--loss MAE` for a comparison with the
MAE objective used by the MSAR example. The original TSLib experiment class is
retained; MAE and the tuple-output DUET/SimpleTM backbones use a separate adapter.

> **DUET + MSAR:** We recommend `--full_imputation` so that imputation uses the
> same `d_model` and `d_ff` as forecasting. Other MSAR runs default to compact
> imputation (`16/32`). DLinear does not use these hidden-width parameters.

```bash
python run.py --task_name msar_forecasting --model DUET --full_imputation
```

Use `python run.py --help` for additional options.

## Outputs

| Location | Contents |
|---|---|
| `results.txt` | Test MSE and MAE, appended after each run |
| `logs/PatchTST/ETTh1/` | Training and evaluation logs from the example script |
| `checkpoints/model_<index>/<setting>/checkpoint.pth` | MSAR weights for each scale |
| `checkpoints/<setting>/checkpoint.pth` | Baseline weights |
| `results/<setting>/` | Baseline predictions, labels, and metrics saved by TSLib |

Metrics are computed on normalized data. Change `--result_file` and
`--checkpoints` to use different output locations. The training pipeline also
writes `result_long_term_forecast.txt`. Data, logs, and generated outputs are
excluded from version control.

## Repository Structure

```text
MSAR/
├── assets/                   # Paper PDF and framework illustration
├── data_provider/            # Dataset loaders and data splits
├── exp/                      # Independent TSLib and MSAR experiments
├── layers/                   # Shared neural network layers
├── models/                   # Forecasting and imputation backbones
├── scripts/PatchTST/ETTh1.sh  # Initial training example
├── utils/                    # Metrics and training utilities
├── requirements.txt
└── run.py                    # Command-line entry point
```

## Citation

If you use MSAR in your research, please cite:

```bibtex
@misc{fan2026msarnextscaleautoregressiveforecasting,
  title         = {MSAR: Next-Scale Autoregressive Forecasting for Time Series via Modular Multi-Scale Decoupling},
  author        = {Fanda Fan and Kuiye Ding and Liming Mao and Bingsong Wang and Yao Wang and Xiaorui Wang and Ruijie Jian and Zhipeng Liu and Luqi Gong and Zhenghua Lu and Chunjie Luo and Jianfeng Zhan},
  year          = {2026},
  eprint        = {2611.20714},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2611.20714},
}
```

## Acknowledgments

We thank the authors and maintainers of
[Time-Series-Library](https://github.com/thuml/Time-Series-Library),
[PatchTST](https://github.com/yuqinie98/PatchTST), and
[iTransformer](https://github.com/thuml/iTransformer) for their code and datasets,
and the authors of the other backbones included in this project.

## License

MSAR is released under the [MIT License](LICENSE). Third-party code remains
subject to its original licenses; see [licenses/](licenses/) and the notices
in the source files.
