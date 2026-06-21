# OLMo SAE Builder

Readable sparse autoencoder tooling for `allenai/Olmo-3-7B-Think`.

The core flow is:

```text
download_olmo_model.py
-> collect_data.py
-> compute_pre_bias.py
-> train.py
```

The scripts are intentionally plain PyTorch. They are meant to be easy to read,
modify, and run on a single GPU.

## Core Files

- `sparse_autoencoder/model.py`: SAE architecture, tied decoder, TopK activation.
- `sparse_autoencoder/loss.py`: normalized reconstruction loss and optional L1 loss.
- `sparse_autoencoder/download_olmo_model.py`: downloads OLMo to the local model folder.
- `sparse_autoencoder/collect_data.py`: streams text, runs OLMo, and saves activation chunks.
- `sparse_autoencoder/compute_pre_bias.py`: computes the activation mean used as `pre_bias`.
- `sparse_autoencoder/train.py`: trains one layer SAE from saved activation chunks.

## Install

```bash
pip install -e .
```

## 1. Download OLMo

```bash
python -m sparse_autoencoder.download_olmo_model
```

By default the model is saved to:

```text
models/Olmo-3-7B-Think
```

You can override this with:

```bash
OLMO_MODEL_PATH=/path/to/Olmo-3-7B-Think \
python -m sparse_autoencoder.download_olmo_model
```

## 2. Collect Activations

```bash
SAE_LAYER=7 python -m sparse_autoencoder.collect_data
```

The collector:

- downloads Dolma `v1_6-sample` and Dolci-Instruct-SFT shards locally
- streams text rows from disk instead of loading the whole dataset
- tokenizes mixed-length chunks up to `8192` tokens
- runs OLMo in `bfloat16` with `torch.no_grad()`
- captures post-layer residual activations with forward hooks
- saves activation chunks shaped `[num_real_tokens, 4096]`

Activation files are written by default to:

```text
data/olmo_acts/layer_07/acts_000000.pt
```

## 3. Compute Pre-Bias

```bash
SAE_LAYER=7 python -m sparse_autoencoder.compute_pre_bias
```

This scans the saved activation chunks once and caches the mean activation
vector. `train.py` loads this cache instead of recomputing it on every run.

## 4. Train

```bash
SAE_LAYER=7 \
SAE_N_LATENTS=131072 \
SAE_TOP_K=64 \
SAE_BATCH_SIZE=256 \
SAE_LR=3e-4 \
python -m sparse_autoencoder.train
```

Checkpoints are saved by default to:

```text
data/sae_checkpoints/layer_07/
```

## Path Overrides

All default paths are anchored to the repository root, so the scripts work from
any current working directory.

Useful overrides:

```bash
SAE_DATA_ROOT=/path/to/data
SAE_ACTS_ROOT=/path/to/olmo_acts
SAE_ACTS_DIR=/path/to/one/layer
SAE_CHECKPOINT_ROOT=/path/to/checkpoints
SAE_CHECKPOINT_DIR=/path/to/one/checkpoint_dir
SAE_PRE_BIAS_ROOT=/path/to/pre_bias
SAE_PRE_BIAS_CACHE=/path/to/pre_bias.pt
SAE_MODELS_ROOT=/path/to/models
SAE_MODEL_DIR=/path/to/Olmo-3-7B-Think
OLMO_MODEL_PATH=/path/to/Olmo-3-7B-Think
```

## Training Knobs

```bash
SAE_LAYER=7
SAE_N_INPUTS=4096
SAE_N_LATENTS=131072
SAE_TOP_K=64
SAE_BATCH_SIZE=256
SAE_EPOCHS=1
SAE_LR=3e-4
SAE_NORMALIZE_INPUTS=0
SAE_SAVE_EVERY_FILES=25
SAE_LOG_EVERY_STEPS=500
```

## Notes

The old distributed Triton trainer, GPT-2 checkpoint path helpers, feature
explanation helpers, and SAE viewer were removed from this fork. This repo is
now centered on the single-GPU OLMo SAE pipeline above.
