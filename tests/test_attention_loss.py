"""v6: the attention loss must not shrink with the defect area."""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import attention_mask_loss  # noqa: E402


def _small_defect():
    mask = torch.zeros(64, 64)
    mask[30:36, 30:36] = 1.0  # ~0.9% of the map, like a hazelnut hole at latent size
    return mask


def test_total_mode_is_the_original_whole_map_mean():
    attention = torch.rand(64, 64)
    mask = _small_defect()

    assert torch.isclose(attention_mask_loss(attention, mask, "total"), F.mse_loss(attention, mask))


def test_region_mode_keeps_full_strength_for_a_small_defect():
    attention = torch.zeros(64, 64)  # [V*] attends nowhere
    mask = _small_defect()

    total = attention_mask_loss(attention, mask, "total")
    region = attention_mask_loss(attention, mask, "region")

    assert total < 0.01                       # whole-map mean is diluted by the area
    assert torch.isclose(region, torch.tensor(1.0))  # inside term (0 - 1)^2 = 1, outside term 0


def test_region_mode_is_zero_for_a_perfect_map_and_penalises_attention_outside():
    mask = _small_defect()

    assert attention_mask_loss(mask.clone(), mask, "region") == 0
    leaking = mask.clone()
    leaking[0:10, 0:10] = 1.0
    outside_share = 100 / (64 * 64 - 36)
    assert torch.isclose(attention_mask_loss(leaking, mask, "region"), torch.tensor(outside_share))


def test_region_mode_backpropagates():
    attention = torch.full((64, 64), 0.1, requires_grad=True)
    attention_mask_loss(attention, _small_defect(), "region").backward()

    assert attention.grad is not None and attention.grad.abs().sum() > 0


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        attention_mask_loss(torch.zeros(4, 4), torch.zeros(4, 4), "mean")


def test_training_passes_and_fingerprints_the_attention_loss_mode():
    train = (ROOT / "train.py").read_text(encoding="utf-8")
    model = (ROOT / "model.py").read_text(encoding="utf-8")

    assert "attention_loss_norm=args.attention_loss_norm" in train
    assert train.count('"attention_loss_norm": args.attention_loss_norm') == 2
    assert "attention_mask_loss(avg_attn_map, mask" in model
