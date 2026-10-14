"""v3: thin mask branches must survive the trip to latent resolution, and the background must stay untouched."""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import composite_background, prepare_inpaint_masks  # noqa: E402


def _thin_line_mask():
    mask = torch.zeros(1, 1, 64, 64)
    mask[..., 10:13, 3:60] = 1.0  # 3-pixel-wide horizontal branch, misses the nearest-sampling grid
    mask[..., 40:56, 40:56] = 1.0  # thick blob
    return mask


def test_maxpool_keeps_every_defect_pixel_inside_the_latent_mask():
    mask = _thin_line_mask()

    pixel_mask, latent_mask = prepare_inpaint_masks(mask, (8, 8), "maxpool")

    assert latent_mask.shape == (1, 1, 8, 8)
    assert torch.all(pixel_mask[mask > 0] == 1)                       # covers the thin branch too
    assert torch.equal(F.interpolate(pixel_mask, size=(8, 8)), latent_mask)  # nearest resize is consistent


def test_nearest_mode_reproduces_the_original_code_and_drops_the_thin_branch():
    mask = _thin_line_mask()

    pixel_mask, latent_mask = prepare_inpaint_masks(mask, (8, 8), "nearest")

    assert torch.equal(pixel_mask, mask)
    assert torch.equal(latent_mask, F.interpolate(mask, size=(8, 8)))
    assert latent_mask[..., 1, :].sum() == 0  # the 3-pixel branch vanishes, as diagnosed


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        prepare_inpaint_masks(_thin_line_mask(), (8, 8), "bilinear")


def test_composite_keeps_original_pixels_away_from_the_mask_and_generated_ones_inside():
    original = torch.zeros(1, 3, 32, 32)
    generated = torch.ones(1, 3, 32, 32)
    mask = torch.zeros(1, 1, 32, 32)
    mask[..., 8:16, 8:16] = 1.0

    out = composite_background(generated, original, mask, feather=2)

    assert torch.all(out[..., 8:16, 8:16] == 1)
    assert torch.all(out[..., :5, :] == 0) and torch.all(out[..., 19:, :] == 0)
    edge = out[0, 0, 12, 6]
    assert 0 < edge < 1  # feathered transition just outside the mask


def test_train_and_inference_use_the_configurable_latent_mask():
    train = (ROOT / "train.py").read_text(encoding="utf-8")
    inference = (ROOT / "inference.py").read_text(encoding="utf-8")
    model = (ROOT / "model.py").read_text(encoding="utf-8")

    assert train.count("prepare_inpaint_masks(") == 2
    assert train.count('"latent_mask_mode": args.latent_mask_mode') == 2
    assert "F.interpolate(defect_masks" not in train
    assert '"latent_mask_mode": args.latent_mask_mode' in inference and '"composite": args.composite' in inference
    assert "mask_image=pixel_mask.float()" in model
