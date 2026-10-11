# DefectFill Paper-Protocol Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the unofficial DefectFill repository train, resume, generate, and evaluate a paper-aligned MVTec AD baseline with power-loss-safe checkpoints.

**Architecture:** Add a small protocol module as the source of truth for deterministic MVTec splits and artifact metadata. Keep model training in the existing files, but replace the frozen pseudo-textual-inversion token and custom denoising loop with paper token/LoRA settings and the standard Diffusers inpainting pipeline. Centralize atomic checkpoint discovery and restoration in `utils.py`, and make evaluation explicitly consume the split manifest.

**Tech Stack:** Python 3.10, PyTorch 2.7, Diffusers 0.37, Transformers 5.6, PEFT 0.20, Albumentations 2.0, LPIPS, torchvision, pytest.

---

## File Structure

- Create `protocol.py`: protocol version, deterministic MVTec split manifest, fingerprints, and artifact metadata validation.
- Modify `data_loader.py`: consume explicit training pairs and use full-view synchronized transforms.
- Modify `model.py`: use `sks`, LoRA dropout 0.1, and standard inpainting pipeline generation.
- Modify `utils.py`: atomic checkpoints, compatibility metadata, latest-valid discovery, and RNG state restoration.
- Modify `train.py`: create/load manifest, auto-resume, safe interrupts, and effective batch defaults.
- Modify `inference.py`: held-out manifest masks, full-view resizing, standard DDIM, and LFS defaults.
- Modify `evaluate.py`: protocol checks, KID raw/scaled output, and reference-clustered IC-LPIPS.
- Create `tests/test_protocol.py`, `tests/test_checkpoint_resume.py`, and `tests/test_evaluation_protocol.py`.
- Extend `tests/test_defectfill_regressions.py` for token, dropout, generation, and branch invariants.

### Task 1: Deterministic Split And Full-View Data

**Files:**
- Create: `protocol.py`
- Modify: `data_loader.py`
- Create: `tests/test_protocol.py`

- [ ] **Step 1: Write failing split and transform tests**

```python
def test_split_is_reproducible_disjoint_and_one_third(tmp_path):
    pairs = make_fake_mvtec_category(tmp_path, count=8)
    manifest = build_split_manifest(tmp_path, "hazelnut", "hole", seed=42)
    assert len(manifest["train"]) == 2
    assert len(manifest["heldout"]) == 6
    assert set(manifest["train"]).isdisjoint(manifest["heldout"])
    assert manifest == build_split_manifest(tmp_path, "hazelnut", "hole", seed=42)

def test_training_sample_is_full_view_512_and_mask_remains_nonempty(tmp_path):
    dataset = make_dataset_with_corner_defect(tmp_path)
    sample = dataset[0]
    assert sample["image"].shape == (3, 512, 512)
    assert sample["mask"].shape == (512, 512)
    assert sample["mask"].sum() > 0
```

- [ ] **Step 2: Run tests and verify expected failures**

Run: `conda run -n ldd python -B -m pytest tests/test_protocol.py -q`

Expected: import failure for missing `protocol.py` or missing manifest-aware dataset API.

- [ ] **Step 3: Implement protocol manifest**

```python
PROTOCOL_VERSION = "defectfill-paper-v1"

def build_split_manifest(data_dir, object_class, defect_type, seed=42):
    pairs = collect_standard_mvtec_pairs(data_dir, object_class, defect_type)
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    train_count = max(1, len(shuffled) // 3)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "seed": seed,
        "object_class": object_class,
        "defect_type": defect_type,
        "train": shuffled[:train_count],
        "heldout": shuffled[train_count:],
    }
```

Add canonical JSON fingerprinting and atomic manifest saving. Validate that pair paths exist, masks match images, both partitions are nonempty, and partitions do not overlap.

- [ ] **Step 4: Implement full-view data path**

Resize source image and mask to 512 x 512 before applying `RandomScale(scale_limit=(0.0, 0.125))` and `RandomCrop(512, 512)`. Use nearest-neighbor mask interpolation and reject transformed empty masks by retrying the augmentation a bounded number of times, then falling back to the full-view resize.

- [ ] **Step 5: Run protocol tests**

Run: `conda run -n ldd python -B -m pytest tests/test_protocol.py -q`

Expected: all protocol tests pass.

- [ ] **Step 6: Commit**

```powershell
git add protocol.py data_loader.py tests/test_protocol.py
git commit -m "fix: align MVTec split and preprocessing with paper"
```

### Task 2: Paper Token, LoRA, Attention, And Generation

**Files:**
- Modify: `model.py`
- Modify: `tests/test_defectfill_regressions.py`

- [ ] **Step 1: Write failing model configuration tests**

```python
def test_paper_mode_uses_existing_sks_token_without_vocab_growth():
    before = len(fake_pipeline.tokenizer)
    model = build_model(fake_pipeline, placeholder_token="sks", lora_dropout=0.1)
    assert len(fake_pipeline.tokenizer) == before
    assert model.placeholder_token == "sks"

def test_lora_dropout_is_point_one():
    model = build_real_or_config_only_model()
    assert model.pipeline.unet.peft_config["default"].lora_dropout == 0.1
    assert model.pipeline.text_encoder.peft_config["default"].lora_dropout == 0.1

def test_generate_delegates_to_standard_inpaint_pipeline():
    model.generate(image=image, mask=mask, prompt="A hazelnut with sks")
    fake_pipeline.assert_called_once()
```

- [ ] **Step 2: Run focused tests and verify failures**

Run: `conda run -n ldd python -B -m pytest tests/test_defectfill_regressions.py -q`

Expected: failures because vocabulary grows, dropout is 0, and generation uses a custom loop.

- [ ] **Step 3: Implement paper model configuration**

Set defaults to `placeholder_token="sks"` and `lora_dropout=0.1`. Remove tokenizer expansion and learned-embedding initialization. Resolve all token positions in the prompt and use the first identifier position consistently for attention supervision. Pass `lora_dropout` into both PEFT `LoraConfig` objects.

- [ ] **Step 4: Replace custom denoising with pipeline generation**

```python
result = self.pipeline(
    prompt=[prompt] * batch_size,
    image=image,
    mask_image=mask,
    num_inference_steps=num_inference_steps,
    guidance_scale=guidance_scale,
    generator=generator,
    output_type="pt",
)
return result.images
```

Temporarily disable attention-map storage during inference and restore it after generation. Keep the differentiable training processor active for training forwards.

- [ ] **Step 5: Run focused and numeric tests**

Run: `conda run -n ldd python -B -m pytest tests/test_defectfill_regressions.py tests/test_loss_numerics.py -q`

Expected: all focused tests pass.

- [ ] **Step 6: Commit**

```powershell
git add model.py tests/test_defectfill_regressions.py
git commit -m "fix: use paper prompt token and standard inpainting"
```

### Task 3: Atomic Checkpoint And Auto Resume

**Files:**
- Modify: `utils.py`
- Create: `tests/test_checkpoint_resume.py`

- [ ] **Step 1: Write failing checkpoint tests**

```python
def test_atomic_checkpoint_has_no_temp_file_after_success(tmp_path):
    path = tmp_path / "checkpoint_100.pt"
    save_checkpoint_atomic(model, optimizer, 100, path, metadata)
    assert path.exists()
    assert not list(tmp_path.glob("*.tmp"))

def test_find_latest_valid_skips_corrupt_newer_file(tmp_path):
    save_valid_checkpoint(tmp_path / "checkpoint_100.pt", step=100)
    (tmp_path / "checkpoint_200.pt").write_bytes(b"truncated")
    assert find_latest_valid_checkpoint(tmp_path, expected_metadata)[0].name == "checkpoint_100.pt"

def test_resume_restores_optimizer_step_and_rng(tmp_path):
    expected = capture_rng_state()
    save_checkpoint_atomic(model, optimizer, 100, path, metadata)
    perturb_all_rngs()
    step = load_checkpoint(model, optimizer, path, expected_metadata, restore_rng=True)
    assert step == 100
    assert_rng_state_matches(expected)
```

- [ ] **Step 2: Run tests and verify failures**

Run: `conda run -n ldd python -B -m pytest tests/test_checkpoint_resume.py -q`

Expected: missing atomic/discovery APIs.

- [ ] **Step 3: Implement atomic serialization**

Write to `path.with_suffix(path.suffix + ".tmp")`, flush and `os.fsync`, then use `os.replace`. Store protocol/config/split fingerprints, prompt token, RNG states, LoRA state dictionaries, optimizer state, and step.

- [ ] **Step 4: Implement validation and latest-valid discovery**

Sort `checkpoint_<step>.pt` numerically descending. Load each on CPU, validate required keys and expected metadata, and return the first valid candidate. Warn for corruption; raise when no compatible checkpoint exists and explicit resume was requested.

- [ ] **Step 5: Run checkpoint tests**

Run: `conda run -n ldd python -B -m pytest tests/test_checkpoint_resume.py -q`

Expected: all checkpoint tests pass.

- [ ] **Step 6: Commit**

```powershell
git add utils.py tests/test_checkpoint_resume.py
git commit -m "feat: add power-loss-safe checkpoint resume"
```

### Task 4: Training Integration

**Files:**
- Modify: `train.py`
- Extend: `tests/test_protocol.py`
- Extend: `tests/test_checkpoint_resume.py`

- [ ] **Step 1: Write failing training wiring tests**

Assert that training creates/loads `split_manifest.json`, defaults to auto resume and save interval 100, passes only manifest training pairs to the loader, uses effective batch four, and includes protocol fingerprints in every save call.

- [ ] **Step 2: Run focused tests and verify failures**

Run: `conda run -n ldd python -B -m pytest tests/test_protocol.py tests/test_checkpoint_resume.py -q`

Expected: parser/default/wiring assertions fail.

- [ ] **Step 3: Integrate manifest and resume**

Create the manifest before model allocation. If it already exists, validate and reuse it. Compute configuration fingerprint from model, optimizer, loss, preprocessing, and split settings. Resolve `--resume_from auto` through latest-valid discovery.

- [ ] **Step 4: Add interruption-safe save behavior**

Default `--save_steps 100`, `--batch_size 1`, and `--gradient_accumulation_steps 4`. Register SIGINT/SIGTERM flags and atomically save an emergency numbered checkpoint at the next optimizer-step boundary before exiting.

- [ ] **Step 5: Run focused tests**

Run: `conda run -n ldd python -B -m pytest tests/test_protocol.py tests/test_checkpoint_resume.py tests/test_defectfill_regressions.py -q`

Expected: all focused tests pass.

- [ ] **Step 6: Commit**

```powershell
git add train.py tests/test_protocol.py tests/test_checkpoint_resume.py
git commit -m "feat: wire paper protocol and automatic resume into training"
```

### Task 5: Held-Out Full-View Inference

**Files:**
- Modify: `inference.py`
- Extend: `tests/test_protocol.py`
- Extend: `tests/test_defectfill_regressions.py`

- [ ] **Step 1: Write failing inference protocol tests**

Assert that inference reads held-out masks from the manifest, never selects a training mask, resizes complete image and mask to 512 x 512, defaults to 8 candidates/50 steps/guidance 2.0, and records `view_mode="full"` plus manifest fingerprint.

- [ ] **Step 2: Run tests and verify failures**

Run: `conda run -n ldd python -B -m pytest tests/test_protocol.py tests/test_defectfill_regressions.py -q`

Expected: smart-crop and default assertions fail.

- [ ] **Step 3: Implement manifest-driven inference**

Remove `smart_crop_dynamic` from paper mode. Load complete RGB image and held-out mask, resize both to 512 x 512, invoke `model.generate`, perform masked spatial LPIPS LFS, and write protocol metadata to `inference_log.json`.

- [ ] **Step 4: Run focused tests**

Run: `conda run -n ldd python -B -m pytest tests/test_protocol.py tests/test_defectfill_regressions.py -q`

Expected: all focused tests pass.

- [ ] **Step 5: Commit**

```powershell
git add inference.py tests/test_protocol.py tests/test_defectfill_regressions.py
git commit -m "fix: generate from held-out full-view MVTec masks"
```

### Task 6: Protocol-Compatible Metrics

**Files:**
- Modify: `evaluate.py`
- Create: `tests/test_evaluation_protocol.py`

- [ ] **Step 1: Write failing evaluation tests**

```python
def test_clustered_ic_lpips_assigns_each_generated_sample_to_nearest_reference():
    distances = torch.tensor([[0.1, 0.9], [0.2, 0.8], [0.9, 0.1]])
    assert assign_clusters(distances).tolist() == [0, 0, 1]

def test_evaluation_rejects_mixed_view_modes(tmp_path):
    write_inference_log(tmp_path, view_mode="crop")
    with pytest.raises(ValueError, match="full-view"):
        validate_evaluation_protocol(tmp_path, manifest)

def test_kid_reports_raw_and_scaled_values():
    result = format_kid(0.00199, 0.0002)
    assert result["kid_raw"] == pytest.approx(0.00199)
    assert result["kid_x1000"] == pytest.approx(1.99)
```

- [ ] **Step 2: Run tests and verify failures**

Run: `conda run -n ldd python -B -m pytest tests/test_evaluation_protocol.py -q`

Expected: missing clustered metric and protocol validation APIs.

- [ ] **Step 3: Implement reference-clustered IC-LPIPS**

Compute LPIPS distances from every generated image to every real reference,
assign each generated image to its nearest reference, and average generated-to-
reference LPIPS within nonempty clusters before averaging clusters. Report the
number and size of nonempty clusters.

- [ ] **Step 4: Implement KID protocol reporting**

Use manifest-held-out real images and full-view generated images, apply one
Inception preprocessing path, and report both raw KID and KID x 1000 with sample
counts. Validate inference metadata before feature extraction.

- [ ] **Step 5: Run evaluation and full unit suites**

Run: `conda run -n ldd python -B -m pytest tests -q`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add evaluate.py tests/test_evaluation_protocol.py
git commit -m "fix: implement paper-compatible DefectFill metrics"
```

### Task 7: End-To-End Verification And Resumable Runs

**Files:**
- Modify: `README.md`
- Create through commands: `runs/hazelnut_hole_paper_smoke/`, `runs/hazelnut_hole_paper_200step/`, and `runs/hazelnut_hole_paper_2000step/`

- [ ] **Step 1: Run static and unit verification**

Run: `conda run -n ldd python -B -m pytest tests -q`

Expected: zero failures.

- [ ] **Step 2: Run one-batch tensor/gradient diagnostic**

Verify 9 input channels, shared noise/timestep/noisy latent, finite three losses,
and nonzero attention gradients for both LoRA parameter groups.

- [ ] **Step 3: Run 20-step smoke training and inference**

Use output `runs/hazelnut_hole_paper_smoke`, save every 10 steps, generate two
images with two candidates and five DDIM steps, and verify logs/metadata.

- [ ] **Step 4: Simulate interruption and resume**

Resume the smoke output with a larger target step. Confirm logs show the prior
step, optimizer state loads, and training continues rather than restarting.

- [ ] **Step 5: Run 200-step validation**

Use effective batch four, save every 100 steps, generate 20 full-view images
with 50 DDIM steps and eight LFS candidates, calculate KID and clustered
IC-LPIPS, and inspect representative outputs for boundary artifacts.

- [ ] **Step 6: Start and complete the resumable 2,000-step run**

Use `runs/hazelnut_hole_paper_2000step`, auto resume, effective batch four,
paper learning rates/loss weights, and atomic checkpoints every 100 optimizer
steps. Continue from the latest checkpoint after any interruption.

- [ ] **Step 7: Document exact commands and protocol labels**

Update README with the local-model command, auto-resume behavior, split
manifest, paper-scale versus smoke labels, and evaluation command.

- [ ] **Step 8: Run final verification**

Run unit tests again, inspect `git diff --check`, verify the latest checkpoint
loads, and compare generated/real input dimensions and manifest fingerprints.

- [ ] **Step 9: Commit documentation**

```powershell
git add README.md
git commit -m "docs: add paper protocol training and resume workflow"
```
