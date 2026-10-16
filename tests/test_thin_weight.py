"""v7: thin mask branches can be up-weighted in L_def."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import DefectFillModel, mask_to_latent, thin_mask_part  # noqa: E402


def _hole_with_crack():
    mask = torch.zeros(1, 1, 64, 64)
    mask[..., 30:46, 30:46] = 1.0   # 16x16 hole: survives a 7x7 opening
    mask[..., 37:39, 2:30] = 1.0    # 2-pixel crack: removed by the opening
    return mask


def test_thin_part_is_the_crack_not_the_hole():
    mask = _hole_with_crack()
    thin = thin_mask_part(mask)

    assert torch.all(thin[..., 37:39, 2:30] == 1)
    assert torch.all(thin[..., 32:44, 32:44] == 0)
    assert torch.all(thin <= mask)


def test_thin_weight_raises_the_share_of_the_crack_in_the_defect_loss():
    mask = _hole_with_crack()
    latent_mask = mask_to_latent(mask, (16, 16), "maxpool")
    thin_latent = mask_to_latent(thin_mask_part(mask), (16, 16), "maxpool")
    noise = torch.zeros(1, 4, 16, 16)
    error_on_crack = thin_latent.expand(1, 4, 16, 16) * (1 - latent_mask.expand(1, 4, 16, 16) + thin_latent.expand(1, 4, 16, 16)).clamp(max=1)

    plain = DefectFillModel.compute_masked_mse(error_on_crack, noise, latent_mask, "mask")
    weighted = DefectFillModel.compute_masked_mse(error_on_crack, noise, latent_mask + 1.5 * thin_latent, "mask")

    assert weighted > plain  # an error on the crack now costs more


def test_training_wires_the_thin_weight_into_l_def_only():
    train = (ROOT / "train.py").read_text(encoding="utf-8")

    assert "defect_weight_latents = mask_latents + (args.thin_weight - 1.0) * thin_latents" in train
    assert "model.compute_defect_loss(noise_pred, noise, defect_weight_latents" in train
    assert "mask_latents=mask_latents" in train  # the UNet mask channel stays binary
    assert train.count('"thin_weight": args.thin_weight') == 2
