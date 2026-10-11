# DefectFill Paper-Protocol Repair Design

## Objective

Repair the unofficial DefectFill implementation so that training, generation,
and evaluation follow the CVPR 2025 paper closely enough for a meaningful
MVTec AD comparison. The repaired workflow must survive workstation power
losses without corrupting the most recent valid checkpoint.

The existing `runs/` outputs and legacy checkpoints remain untouched. New
experiments use a separate output directory and a protocol version recorded in
their configuration, split manifest, checkpoints, and inference logs.

## Scope

The first validated target is MVTec AD `hazelnut/hole` using:

- Dataset root: `D:\Novels_Code\open_datasets\MVTec`
- Base model: `D:\Novels_Code\stablediffusion\sd2_inpainting`
- Conda environment: `ldd`

The implementation remains reusable for other standard MVTec object and
defect categories. Downstream anomaly detection and localization are outside
this repair; the deliverable covers DefectFill training, defect generation,
KID, and IC-LPIPS.

## Protocol Decisions

### Deterministic MVTec split

For each defect category, sort image-mask pairs by filename, shuffle them with
an explicit split seed, and assign `max(1, floor(N / 3))` pairs to training.
The remaining pairs are held out. Training can only read training pairs;
inference can only use held-out masks; KID uses held-out real defect images.

The exact filenames, seed, source dataset, and split counts are written to
`split_manifest.json`. Inference and evaluation consume this manifest rather
than independently rebuilding a split.

### Full-view preprocessing

Images and masks are first resized together to 512 x 512. Training then applies
the paper augmentation: random scaling from 1.0 to 1.125 followed by a random
512 x 512 crop. Masks use nearest-neighbor interpolation. This keeps the full
object in the learning signal while allowing the small translation introduced
by the paper augmentation.

The defect and object branches consume the same augmented image. `M_rand` is
created only for the object branch. Both branches reuse the same clean latent,
noise, timestep, and noisy latent. The object loss uses the true defect mask to
construct `M' = M + alpha * (1 - M)`.

Inference resizes the complete normal image and held-out mask to 512 x 512. It
does not perform defect-centered smart cropping.

### Text token and LoRA

Paper mode uses the existing tokenizer text `sks`; it does not add a new token
or claim to train a textual-inversion embedding. The implementation locates the
token position used by the prompt for attention supervision and fails with an
actionable error if the configured identifier cannot be located.

UNet and text-encoder attention projection matrices use LoRA rank 8, scaling
factor 16, and dropout 0.1. Only LoRA parameters are optimized. New checkpoints
record the identifier and reject incompatible legacy `<defect>` checkpoints
instead of silently loading a mismatched embedding.

### Loss and attention behavior

The repaired implementation keeps the paper loss weights:

- `lambda_defect = 0.5`
- `lambda_object = 0.2`
- `lambda_attention = 0.05`
- `alpha = 0.3`

Masked noise losses are accumulated in FP32 with an empty-mask guard. Attention
maps are collected only from decoder cross-attention layers, remain attached to
the graph, and are supervised by the true defect mask. Tests verify that the
attention loss produces nonzero gradients in both text-encoder and UNet LoRA
parameters.

### Standard inpainting inference

Generation uses the standard nine-channel Stable Diffusion 2 inpainting path
with DDIM. The custom per-step hard replacement of background latents is
removed. In particular, no new background noise is sampled at every denoising
step.

Paper evaluation defaults are 50 DDIM steps, guidance scale 2.0, and eight LFS
candidates. LFS computes spatial LPIPS inside the requested mask and selects
the candidate with the highest masked distance from the normal input.

### Evaluation

KID compares complete 512 x 512 generated images with complete held-out real
defect images under one preprocessing implementation. Both raw KID and
`KID x 1000` are reported. The evaluator records real/generated counts and
refuses to compare cropped and full-view protocol artifacts.

IC-LPIPS follows the reference intra-cluster procedure used by the cited
few-shot generation work: generated images are assigned to their nearest real
reference sample by LPIPS distance, then diversity is aggregated within the
resulting clusters. It is not the all-pairs distance across unrelated normal
backgrounds used by the current repository.

Paper-scale reporting uses 1,000 generated images per defect category and then
averages category scores for an object. A short validation run may use fewer
images, but its output is labeled `smoke` and is never presented as a paper
score.

## Power-Loss-Safe Checkpoints

Every numbered checkpoint is written to a temporary file in the checkpoint
directory, flushed and synchronized, then atomically moved into place. A power
loss can therefore leave either the previous valid checkpoint or the new valid
checkpoint, but not a partially overwritten final file.

Each checkpoint contains:

- protocol version and configuration fingerprint;
- global optimizer step;
- UNet and text-encoder LoRA weights;
- optimizer state;
- Python, NumPy, Torch CPU, and Torch CUDA random states;
- split-manifest fingerprint and prompt identifier.

`--resume_from auto` scans numbered checkpoints from newest to oldest, validates
each candidate, skips truncated or incompatible files with a warning, and
loads the latest valid one. `--save_steps` defaults to 100. SIGINT and SIGTERM
request an emergency checkpoint at the next safe batch boundary. Abrupt power
loss falls back to the latest atomic periodic checkpoint.

Exact batch ordering after an abrupt restart is not guaranteed, but model,
optimizer, global step, data split, and random states are restored. This is
sufficient to continue the experiment without resetting training progress.

## Components

- `data_loader.py`: deterministic split consumption and full-view transforms.
- `model.py`: paper token, LoRA dropout, attention collection, and standard
  inpainting generation.
- `train.py`: manifest creation, effective batch handling, protocol logging,
  safe periodic checkpointing, and automatic resume.
- `inference.py`: held-out manifest consumption, standard full-view generation,
  and eight-candidate LFS.
- `evaluate.py`: protocol-compatible KID and clustered IC-LPIPS.
- `utils.py`: atomic checkpoint serialization, validation, discovery, and RNG
  restoration.
- `tests/`: regression tests for each repaired behavior.

## Error Handling

Training stops before allocating the diffusion model if the selected category
has fewer than two valid image-mask pairs, the manifest overlaps train and
held-out sets, or a mask is empty after full-view preprocessing. Resume fails
clearly when the protocol/configuration fingerprint differs. Corrupt newer
checkpoints are skipped only when an older valid checkpoint is available.

Inference refuses training masks, missing manifest entries, incompatible
checkpoint identifiers, and mixed crop/full-view artifacts. Evaluation refuses
fewer than two real or generated images and records when sample counts are too
small for a paper-scale claim.

## Verification Strategy

Development follows red-green TDD for:

1. deterministic, disjoint one-third/two-thirds splitting;
2. full-view resize and synchronized image-mask augmentation;
3. `sks` use without tokenizer expansion and LoRA dropout 0.1;
4. object-branch-only `M_rand` with shared noise/timestep/noisy latent;
5. differentiable attention loss using the true defect mask;
6. standard inpainting generation without per-step background resampling;
7. clustered IC-LPIPS on a small synthetic example;
8. full-view KID input validation;
9. atomic checkpoint save, corrupt-latest fallback, compatibility checks, and
   optimizer/RNG restoration.

After unit tests pass, run these stages in order:

1. one-batch tensor and gradient diagnostic;
2. 20-step training plus two-image inference smoke test;
3. interrupted/resumed training test across a checkpoint boundary;
4. 200-step, effective-batch-four validation with 20 generated images;
5. paper-parameter 2,000-step training with checkpoints every 100 steps;
6. 50-step, eight-candidate generation and paper-scale evaluation when 1,000
   images per defect category are available.

## Acceptance Criteria

- All regression tests pass in conda environment `ldd`.
- No NaN or infinite training losses occur in the validation run.
- Training automatically resumes from the newest valid checkpoint after a
  simulated corrupt latest checkpoint.
- Train and held-out filenames are disjoint and match the saved manifest.
- Generated and real KID inputs use the same full-view preprocessing.
- IC-LPIPS uses real-reference clustering rather than global all-pairs LPIPS.
- The 200-step validation produces complete logs, checkpoint metadata,
  generated images, metrics, and a visual comparison without the previous
  cyan-blue boundary artifact being introduced by custom latent replacement.
- Paper-scale results are reported separately from smoke and validation runs.
