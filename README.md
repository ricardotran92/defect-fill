# Defect Fill

---

The paper-protocol path supports the standard MVTec AD layout, creates a deterministic one-third reference split, and reserves the remaining two-thirds for generation and evaluation.
-> Codex fix several problems and adapt to the mvtec datasets. Index still has a distance behind the paper.




<img width="792" height="1920" alt="download" src="https://github.com/user-attachments/assets/174cd6e0-5935-442b-8a5e-05402afa31ff" />
<img width="468" height="289" alt="b4ea2043690fb2a7e0b83317cf63cc8a" src="https://github.com/user-attachments/assets/15f4e9a4-1004-4eff-ad44-e9971d220cd8" />

## Problem 
The generated results tend to be similar black circles/irregular holes, with slight yellow-green halos appearing on some edges, and the slender mask branches sometimes not fully converted into defects. This nicely explains the combination of 'KID being good and IC-LPIPS being low'.
## Overview

DefectFill fine-tunes a Stable Diffusion 2 inpainting model with LoRA to learn a specific defect concept from a small set of reference images. Three complementary loss terms drive training:

| Loss | Weight | Purpose |
|------|--------|---------|
| **Defect loss** `L_def` | 0.5 | Precisely captures intrinsic defect features |
| **Object loss** `L_obj` | 0.2 | Learns the semantic relationship between defect and object |
| **Attention loss** `L_attn` | 0.05 | Ensures [V*] token attends to the defect region |

After training, **Low-Fidelity Selection (LFS)** generates 8 candidates per (image, mask) pair and selects the one with the highest LPIPS score inside the masked region — the most "realistic" defect.

---

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/ricardotran92/defect-fill.gits
    cd defectfill
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

**Requirements include:** `torch`, `diffusers`, `transformers`, `peft`, `lpips`, and `albumentations`.

---

## Verified Environment

The current Windows setup was verified with the `ldd` Conda environment, Python 3.10, PyTorch 2.7.0 + CUDA 12.8, Diffusers 0.37.1, Transformers 5.6.2, PEFT 0.20.0, and Albumentations 2.0.8.

## Usage

### 1. Training

The defaults use paper-aligned learning rates, 2000 optimizer steps, batch size 1 with four-step gradient accumulation (effective batch size 4), LoRA dropout 0.1, and checkpoints every 100 optimizer steps.

```bash
conda activate ldd
python train.py \
  --data_dir "D:\Novels_Code\open_datasets\MVTec" \
  --object_class hazelnut \
  --defect_type hole \
  --output_dir "runs\hazelnut_hole_paper_2000step" \
  --base_model_path "D:\Novels_Code\stablediffusion\sd2_inpainting"
```

The run directory contains `split_manifest.json`, `train_config.json`, TensorBoard logs, and atomic checkpoints. `--resume_from auto` is the default; rerunning the same command resumes from the newest complete compatible checkpoint. A damaged latest file is skipped automatically.

Intermediate checkpoints (`checkpoint_<step>.pt`, about 37MB each) exist only so an unfinished run can resume. When training completes, `checkpoint_final.pt` is written and the intermediate files are deleted; it holds the same LoRA weights, optimizer and RNG state as the last step. Pass `--keep_intermediate_checkpoints` to keep them. An interrupted run keeps all of its checkpoints.

Key training details (from paper):
- **Base model**: `sd2-community/stable-diffusion-2-inpainting`
- **LoRA** on UNet attention layers + text encoder projection matrices
- **Warmup**: linear 0 → LR over first 100 steps
- **Augmentation**: random resize ×[1.0, 1.125] + random crop
- **Random masks** M_rand: 30 boxes, sides 3–25% of image size
- **[V*] token**: the word `sks`

### 2. Inference
Generate new synthetic defects on healthy images. The script uses LPIPS to pick the best generation from a batch of candidates.
```bash
python inference.py \
  --checkpoint "runs\hazelnut_hole_paper_2000step\checkpoints\checkpoint_final.pt" \
  --output_dir "runs\hazelnut_hole_paper_generated_100" \
  --base_model_path "D:\Novels_Code\stablediffusion\sd2_inpainting" \
  --data_dir "D:\Novels_Code\open_datasets\MVTec" \
  --object_class hazelnut \
  --defect_type hole \
  --total_images 100 \
  --num_samples 8 \
  --steps 50 \
  --guidance_scale 2.0
```

While the code is still changing, generate **100 images** per run for quick comparisons (the paper uses 1000; pass `--total_images 1000` for a final run). Use a new `--output_dir` for each image count: evaluation scores every `*_generated.png` in the directory, and resuming into a folder that already holds 1000 images would mix the two runs.

To check how thin mask branches are filled, favour chosen held-out masks: `--priority_masks 006_mask.png,010_mask.png,011_mask.png,014_mask.png,016_mask.png --priority_ratio 3` spends 3 of every 4 outputs on those masks (75 of 100 images, 15 per mask) and 1 of every 4 on the remaining masks. Seeds and normal images depend only on the output index, so runs with the same settings compare identical image/mask/seed triples. Metrics of a priority plan are only comparable with other runs that use the same plan; write it to its own output folder (for example a `_100p` suffix).

Inference uses only held-out masks from the saved split manifest, keeps the full 512-by-512 view, applies persistent-noise latent background preservation, and updates `inference_log.json` after every output so generation can also resume.

### 3. Evaluation

```bash
python evaluate.py \
  --generated_dir "runs\hazelnut_hole_paper_generated_100" \
  --split_manifest "runs\hazelnut_hole_paper_2000step\split_manifest.json" \
  --output_csv "runs\paper_protocol_metrics.csv" \
  --object_class hazelnut \
  --defect_type hole \
  --config_name paper_2000step_100img \
  --category_type object
```

Evaluation validates the protocol fingerprint and full-view inference log. It reports raw KID, paper-table `KID x 1000`, and reference-clustered intra-cluster LPIPS. With 100 generated images the metrics are noisier than with 1000, so compare runs only at the same image count.

---

## Method Details

### Defect Loss (Eq. 5)

```
L_def = E[ || M ⊙ (ε − ε_θ(x_t^def, t, c^def)) ||² ]
```

Background image: `B_def = (1 − M) ⊙ I`  
Input: `x_t^def = concat(x_t, b_def, M)`  
Prompt `P_def = "A photo of sks"`

### Object Loss (Eq. 7)

```
L_obj = E[ || M' ⊙ (ε − ε_θ(x_t^obj, t, c^obj)) ||² ]
M' = M + α·(1 − M),   α = 0.3
```

Random box mask M_rand (30 boxes), `B_rand = (1 − M_rand) ⊙ I`  
Input: `x_t^obj = concat(x_t, b_rand, M_rand)`  
Prompt `P_obj = "A <object> with sks"`

### Attention Loss (Eq. 8)

```
L_attn = E[ || A_t^[V*] − M ||² ]
```

Cross-attention maps from UNet **decoder** (up_blocks) only, averaged over layers and resized to latent resolution.

### Combined Loss (Eq. 9)

```
L_ours = 0.5·L_def + 0.2·L_obj + 0.05·L_attn
```

---



