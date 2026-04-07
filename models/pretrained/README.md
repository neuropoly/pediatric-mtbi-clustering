# Pretrained Models

This directory contains pretrained autoencoder checkpoints used for latent representation analysis of pediatric brain MRI.

## Available checkpoints

* **`epoch-200.pth`**
* **`epoch-338.pth`**

Both checkpoints correspond to the **same model architecture** trained under the same configuration, saved at different training epochs.

## Training data

These models were trained on **normative (healthy) pediatric brain MRI data**, covering:

* **Age range:** 1–7 years
* **Population:** typically developing children (no mTBI)

The goal of training on normative data is to learn a **baseline representation of typical brain structure**, enabling downstream analysis (e.g., clustering or deviation detection) in mTBI subjects.

## Training setup

* **Model:** AutoencoderKL (VAE-style)
* **Input:** 3D T1-weighted MRI volumes
* **Preprocessing:** intensity normalization and spatial alignment (see main repo for details)
* **Objective:** reconstruction + KL regularization (and optional perceptual/adversarial components depending on experiment), see `scripts/train/train_autoencoder_all.py` to know how the network was trained.

Both checkpoints differ only by the number of training epochs:

* `epoch-200.pth`: earlier checkpoint
* `epoch-338.pth`: later checkpoint (typically better convergence)

## Original training location

These checkpoints were originally generated on the cluster at:

```
/home/andim/scratch/bgp/ae_all_output/
```

## Usage

These models are intended for:

* extracting **latent representations** from MRI scans
* reconstructing input images for quality control
* analyzing latent space structure (PCA, UMAP, clustering)
* detecting deviations from normative patterns (e.g., mTBI)

Example usage:

```bash
python scripts/analyze/extract_latents.py \
    --model-path models/pretrained/epoch-338.pth \
    --config configs/analyze/extract_latents.yaml
```

## Notes

* These models are trained on **healthy data only** and are not classifiers.
* Any mTBI-related findings come from **downstream analysis in latent space**, not from supervised prediction.
* Ensure that input data follows the same preprocessing pipeline as used during training.
