import os
import pickle
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F


CHECKPOINT_FORMAT_VERSION = 2


def _to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def capture_rng_state() -> Dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": None,
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Optional[Dict[str, Any]]) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") and torch.cuda.is_available():
        for device_index, device_state in enumerate(state["cuda"][:torch.cuda.device_count()]):
            torch.cuda.set_rng_state(device_state, device=device_index)


def _validate_checkpoint(checkpoint, path, expected_metadata=None):
    required = {"step", "text_encoder_lora", "unet_lora"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint {path} is missing fields: {sorted(missing)}")

    metadata = checkpoint.get("metadata", {})
    for key, expected_value in (expected_metadata or {}).items():
        actual_value = metadata.get(key)
        if actual_value != expected_value:
            raise ValueError(
                f"Checkpoint metadata mismatch for {key}: "
                f"expected {expected_value!r}, got {actual_value!r}"
            )
    return checkpoint


def _read_checkpoint(path, expected_metadata=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return _validate_checkpoint(checkpoint, path, expected_metadata)


def save_checkpoint(model, optimizer, step, path, metadata=None):
    """Atomically save both LoRA branches and all state needed to resume training."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")

    checkpoint = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "step": int(step),
        "metadata": dict(metadata or {}),
        "placeholder_token": getattr(model, "placeholder_token", "sks"),
        "text_encoder_lora": _to_cpu({
            key: value
            for key, value in model.pipeline.text_encoder.state_dict().items()
            if "lora" in key
        }),
        "unet_lora": _to_cpu({
            key: value
            for key, value in model.pipeline.unet.state_dict().items()
            if "lora" in key
        }),
        "optimizer": _to_cpu(optimizer.state_dict()) if optimizer is not None else None,
        "rng_state": capture_rng_state(),
    }

    try:
        with temporary_path.open("wb") as checkpoint_file:
            torch.save(checkpoint, checkpoint_file)
            checkpoint_file.flush()
            os.fsync(checkpoint_file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    print(f"Checkpoint saved to {path}")


def _load_lora_branch(module, saved_state, branch_name):
    current_keys = set(module.state_dict())
    recognized_state = {
        key: value for key, value in saved_state.items() if key in current_keys
    }
    if saved_state and not recognized_state:
        raise ValueError(f"No {branch_name} LoRA keys in the checkpoint match this model")
    module.load_state_dict(recognized_state, strict=False)


def load_checkpoint(
    model,
    optimizer,
    path,
    expected_metadata=None,
    restore_rng=False,
) -> int:
    """Restore a validated checkpoint without allocating a second model on the GPU."""
    if not os.path.exists(path):
        print(f"Checkpoint {path} not found, starting from scratch")
        return 0

    checkpoint = _read_checkpoint(path, expected_metadata=expected_metadata)
    _load_lora_branch(
        model.pipeline.text_encoder,
        checkpoint["text_encoder_lora"],
        "text encoder",
    )
    _load_lora_branch(model.pipeline.unet, checkpoint["unet_lora"], "UNet")

    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if restore_rng:
        restore_rng_state(checkpoint.get("rng_state"))

    print(f"Checkpoint loaded from {path}")
    return int(checkpoint["step"])


def find_latest_valid_checkpoint(checkpoints_dir, expected_metadata=None):
    """Return the newest readable compatible checkpoint, ignoring partial files."""
    checkpoints_dir = Path(checkpoints_dir)
    if not checkpoints_dir.exists():
        return None

    latest_path = None
    latest_step = -1
    for candidate in checkpoints_dir.glob("checkpoint_*.pt"):
        try:
            checkpoint = _read_checkpoint(candidate, expected_metadata=expected_metadata)
            step = int(checkpoint["step"])
            if step > latest_step or (
                step == latest_step and candidate.name == "checkpoint_final.pt"
            ):
                latest_path = candidate
                latest_step = step
        except (OSError, RuntimeError, EOFError, ValueError, KeyError, pickle.UnpicklingError) as error:
            print(f"[Checkpoint] Skipping invalid {candidate}: {error}")
    return str(latest_path) if latest_path is not None else None


def compute_spatial_lpips(lpips_model, img1, img2, mask, smooth_boundary=False):
    """
    Calculates the Perceptual Distance specifically within the masked region using Spatial LPIPS
    
    Args:
        lpips_model: LPIPS model instance initialized with spatial=True
        img1: Reference image [B, 3, H, W], range [-1, 1]
        img2: Comparison image [B, 3, H, W], range [-1, 1]
        mask: Defect mask [B, 1, H, W], range [0, 1]
        smooth_boundary: Blur the mask edges (off by default; the paper scores only inside the mask)
    
    Returns:
        lpips_score: LPIPS score for the masked region (scalar)
    """
    # 1. Compute spatial LPIPS map (pixel-wise perceptual distance)
    lpips_map = lpips_model(img1, img2)  # Output shape: [B, 1, H', W']
    
    # 2. Resize the mask to match the LPIPS output resolution
    mask_resized = F.interpolate(
        mask, 
        size=lpips_map.shape[-2:], 
        mode='bilinear',
        align_corners=False
    )
    
    # 3. Optional: Smooth boundary edges (Gaussian-like blur via AvgPool)
    if smooth_boundary:
        mask_smoothed = F.avg_pool2d(
            F.pad(mask_resized, (2, 2, 2, 2), mode='replicate'),
            kernel_size=5, stride=1
        )
    else:
        mask_smoothed = mask_resized
    
    # 4. Mask-weighted summation
    weighted_sum = (lpips_map * mask_smoothed).sum(dim=(2, 3))
    mask_sum = mask_smoothed.sum(dim=(2, 3)) + 1e-8
    
    # 5. Return normalized LPIPS score
    return (weighted_sum / mask_sum).mean()


def compute_spatial_lpips_batch(lpips_model, reference, samples, mask, smooth_boundary=False):
    """
    Batch calculation of Spatial LPIPS - Evaluates all generated samples at once (FP16 compatible)
    
    This utilizes the "paired batch" mode of LPIPS by expanding the reference image
    to match the number of samples, allowing parallel evaluation in a single forward pass.
    
    Args:
        lpips_model: Spatial LPIPS model instance (spatial=True)
        reference: Single reference image [1, 3, H, W], range [-1, 1]
        samples: Multiple generated samples [N, 3, H, W], range [-1, 1]
        mask: Single defect mask [1, 1, H, W], range [0, 1]
        smooth_boundary: Blur the mask edges (off by default; the paper scores only inside the mask)
    
    Returns:
        lpips_scores: A tensor of [N] LPIPS scores
    """
    num_samples = samples.shape[0]
    
    # LPIPS model expects FP32 input; cast based on model parameters
    lpips_dtype = next(lpips_model.parameters()).dtype
    
    # Expand reference and mask to match sample count N
    reference_expanded = reference.repeat(num_samples, 1, 1, 1).to(dtype=lpips_dtype)
    samples_for_lpips = samples.to(dtype=lpips_dtype)
    mask_expanded = mask.repeat(num_samples, 1, 1, 1)
    
    # Parallel forward pass for all N pairs
    lpips_maps = lpips_model(reference_expanded, samples_for_lpips)  # [N, 1, H', W']
    
    # Match mask size to LPIPS feature map resolution
    mask_resized = F.interpolate(
        mask_expanded.to(dtype=lpips_maps.dtype), 
        size=lpips_maps.shape[-2:], 
        mode='bilinear',
        align_corners=False
    )
    
    # Boundary smoothing
    if smooth_boundary:
        mask_smoothed = F.avg_pool2d(
            F.pad(mask_resized, (2, 2, 2, 2), mode='replicate'),
            kernel_size=5, stride=1
        )
    else:
        mask_smoothed = mask_resized
    
    # Calculate weighted scores per sample
    weighted_sum = (lpips_maps * mask_smoothed).sum(dim=(2, 3))  # [N, 1]
    mask_sum = mask_smoothed.sum(dim=(2, 3)) + 1e-8             # [N, 1]
    
    lpips_scores = (weighted_sum / mask_sum).squeeze(1)  # [N]
    
    return lpips_scores
