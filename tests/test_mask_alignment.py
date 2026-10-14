"""Thin mask branches must survive the trip to latent resolution (v3), without turning the
inpainted area into 8x8 blocks (v4), and the background must stay untouched."""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import composite_background, mask_to_latent  # noqa: E402


def _thin_line_mask():
    mask = torch.zeros(1, 1, 64, 64)
    mask[..., 10:13, 3:60] = 1.0  # 3-pixel-wide horizontal branch, misses the nearest-sampling grid
    mask[..., 40:56, 40:56] = 1.0  # thick blob
    return mask


def test_maxpool_latent_mask_covers_every_defect_pixel():
    mask = _thin_line_mask()

    latent_mask = mask_to_latent(mask, (8, 8), "maxpool")

    assert latent_mask.shape == (1, 1, 8, 8)
    covered = F.interpolate(latent_mask, size=mask.shape[-2:])
    assert torch.all(covered[mask > 0] == 1)  # the thin branch is inside the latent mask
    # a cell-aligned pixel mask is resized back to exactly this latent mask (what the pipeline does)
    assert torch.equal(F.interpolate(covered, size=(8, 8)), latent_mask)


def test_nearest_mode_reproduces_the_original_code_and_drops_the_thin_branch():
    mask = _thin_line_mask()

    latent_mask = mask_to_latent(mask, (8, 8), "nearest")

    assert torch.equal(latent_mask, F.interpolate(mask, size=(8, 8)))
    assert latent_mask[..., 1, :].sum() == 0  # the 3-pixel branch vanishes, as diagnosed


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        mask_to_latent(_thin_line_mask(), (8, 8), "bilinear")


def test_composite_follows_the_defect_shape_not_the_latent_grid():
    original = torch.zeros(1, 3, 64, 64)
    generated = torch.ones(1, 3, 64, 64)
    mask = _thin_line_mask()

    out = composite_background(generated, original, mask, grow=3, feather=2)

    assert torch.all(out[..., 10:13, 3:60] == 1)            # defect kept
    assert torch.all(out[..., 7:16, 10:50] == 1)             # grown by 3 px around the branch
    assert torch.all(out[..., 20:35, :] == 0)                # rest of the latent cell row stays original
    assert torch.all(out[..., :4, :] == 0)
    assert 0 < out[0, 0, 16, 30] < 1                         # feathered edge


def test_training_blanks_the_background_with_the_full_resolution_mask():
    train = (ROOT / "train.py").read_text(encoding="utf-8")
    model = (ROOT / "model.py").read_text(encoding="utf-8")

    assert "masked_images = defect_images * (1 - defect_masks)" in train
    assert "random_masked_images = defect_images * (1 - random_masks)" in train
    assert train.count("mask_to_latent(") == 2
    assert train.count('"latent_mask_mode": args.latent_mask_mode') == 2
    assert train.count('"background_mask": BACKGROUND_MASK') == 2
    assert "F.interpolate(defect_masks" not in train
    assert "masked_image_latents=background_image" in model and "mask_image=cell_mask.float()" in model


def test_inference_records_the_mask_settings():
    inference = (ROOT / "inference.py").read_text(encoding="utf-8")

    assert '"latent_mask_mode": args.latent_mask_mode' in inference and '"composite": args.composite' in inference
