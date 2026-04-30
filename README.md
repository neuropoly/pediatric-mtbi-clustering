# pediatric-mtbi-clustering
Latent-space analysis of pediatric brain MRI for mTBI detection using autoencoder-based representations

This project explores whether **mild traumatic brain injury (mTBI)** in children can be identified through **latent representations of brain MRI** learned by a pretrained AutoencoderKL model.

The repository provides:

* pretrained models (in `models/pretrained/`)
* dataset format
* environment setup

Your task is to implement the **analysis pipeline** on top of these.

---

## 🚀 Project Workflow

You are expected to implement the following three steps:

### 1. Extract latent representations

Implement:

```bash
scripts/analyze/extract_latents.py
```

This script should:

* load a pretrained model (`.pth`)
* read a dataset CSV
* extract latent representations from each MRI
* save them to disk (`.npz`)
* use the `latent_path` column to determine output locations

---

### 2. Cluster latent representations

Implement:

```bash
scripts/analyze/cluster_latents.py
```

This script should:

* load latent vectors (`.npz`)
* perform clustering (e.g., KMeans, hierarchical, etc.)
* evaluate separation between diagnostic groups (control vs mTBI)

---

### 3. Visualize latent space

Implement:

```bash
scripts/analyze/visualize_latents.py
```

This script should:

* reduce dimensionality (PCA, UMAP, t-SNE)
* generate plots
* visualize separation between groups

---

## 🧠 Pretrained Models

Pretrained AutoencoderKL models are provided in:

```text
models/pretrained/
```

See:

```text
models/pretrained/README.md
```

for details on:

* training data (normative children, 1–7 years)
* architecture
* checkpoint differences

These models are used to extract latent representations.
They are **not classifiers**.

---

## 🧪 Datasets

The data used in this project comes from:

### KOALA & DAUFIN datasets

* Pediatric mTBI cohorts from the team of Miriam Beauchamp
* Reference: Pediatric mild traumatic brain injury study

### KAOUENN dataset (EMPENN – local dataset)

* Collected by Fanny Dégeilh

These datasets include:

* control subjects
* children with orthopedic injuries (considered as controls)
* children with mTBI
* longitudinal and/or cross-sectional scans

---

## 📦 Installation

The project uses a `pyproject.toml` configuration.

### Create environment

```bash
python -m venv ~/venvs/env_name
source ~/venvs/env_name/bin/activate
python -m pip install --upgrade pip
```

### Install the project

```bash
pip install -e .
```

---

## 📄 Dataset Format

The input to `extract_latents.py` is a CSV file with the following structure:

```csv
subject_id,image_uid,split,sex,age,diagnosis,time_post_injury_days,image_path,segm_path,latent_path,head_size,accumbens_area,amygdala,brain_stem,caudate,cerebellum_cortex,cerebellum_white_matter,cerebral_cortex,cerebral_white_matter,csf,fourth_ventricle,hippocampus,inferior_lateral_ventricle,lateral_ventricle,pallidum,putamen,thalamus,third_ventricle,ventral_dc
sub-1007,sub-1007,train,0.0,0.9883,CC,,/home/.../sub-1007_brain.nii.gz,/home/.../sub-1007_segm.nii.gz,/home/.../sub-1007_latent.npz,1403942,0.2954,0.4607,0.1811,0.5138,0.5999,0.0,0.8166,0.4874,0.5338,0.7263,0.4665,0.4773,0.4192,0.0,0.3160,0.2246,0.2905,0.3425
```

### Key columns

* `subject_id`: Identifier of the participant (in the case of cross-sectional datasets, subject_id=image_uid)
* `split`: Split for further analyses (train, validation and test)
* `age`: age of the participant (max-min normalized, min: 12 months, max: 84 months) 
* `sex`: sex of the participant (0: Male, 1: Female)
* `image_path`: path to MRI image (input)
* `segm_path`: path to MRI image segmentation
* `latent_path`: where latent representation should be saved
* `diagnosis`: typically `control` or `mtbi` or `orthopedic_injury` (may appear as `CC`, etc.)
* `time_post_injury_days`: useful for studying clustering based on this variable for `mtbi` subjects
* additional columns: regional brain measures based on `segm_path` (optional for downstream analysis)

---

## ⚠️ Important Notes on Data

* The CSV file and all generated outputs (latents, results) **must not be committed to Git**
* Add them to `.gitignore`, for example:

```text
data/
outputs/
*.csv
*.npz
```

This ensures:

* compliance with data privacy
* reproducibility without sharing sensitive data

---

## 📤 Expected Output of `extract_latents.py`

For each row in the CSV:

* Input:

  * MRI image (`image_path`)
  * pretrained model (`.pth`)
* Output:

  * latent vector saved at `latent_path` (`.npz`)

---

## 💡 Suggested Pipeline

```bash
# 1. Extract latents
python scripts/analyze/extract_latents.py \
    --dataset_csv data/dataset.csv \
    --model_path models/pretrained/epoch-338.pth

# 2. Cluster
python scripts/analyze/cluster_latents.py \
    --dataset_csv data/dataset.csv

# 3. Visualize
python scripts/analyze/visualize_latents.py \
    --dataset_csv data/dataset.csv


# 4. Dataset CSV
python scripts/analyze/prepare_dataset_csv.py \
    --root_path /home/USERNAME/dataset_root \
    --output_csv data/dataset.csv \
    --latent_path outputs/latents
```

---

## 📚 Summary

```text
MRI → AutoencoderKL → Latent space → Clustering → Visualization
```

Goal:

> Determine whether **mTBI subjects separate from controls in latent space**

---

## ⚠️ Final Remarks

* Models are trained on **normative data only**
* This is **not a diagnostic model**
* All findings rely on **latent space analysis**

