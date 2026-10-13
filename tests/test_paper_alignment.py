"""Regression tests for the places where the code was aligned with the DefectFill paper."""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_loader import (  # noqa: E402
    RANDOM_BOX_COUNT,
    RANDOM_BOX_MAX_FRAC,
    RANDOM_BOX_MIN_FRAC,
    generate_random_box_mask,
)
from model import DefectFillModel  # noqa: E402
from utils import compute_spatial_lpips_batch  # noqa: E402


def test_object_loss_weights_background_by_alpha_squared():
    noise = torch.zeros(1, 1, 1, 2)
    noise_pred = torch.ones(1, 1, 1, 2)
    defect_mask = torch.tensor([[[[1.0, 0.0]]]])
    host = SimpleNamespace(compute_masked_mse=DefectFillModel.compute_masked_mse)

    loss = DefectFillModel.compute_object_loss(host, noise_pred, noise, defect_mask, alpha=0.3)

    # M' = [1, 0.3] sits inside the squared norm: (1² + 0.3²) / 2
    assert torch.isclose(loss, torch.tensor((1.0 + 0.09) / 2))


def test_random_box_mask_is_reproducible_and_follows_paper_box_sizes():
    first = generate_random_box_mask(512, 512, generator=torch.Generator().manual_seed(0))
    second = generate_random_box_mask(512, 512, generator=torch.Generator().manual_seed(0))

    assert torch.equal(first, second)
    assert set(first.unique().tolist()) <= {0.0, 1.0}
    assert (RANDOM_BOX_COUNT, RANDOM_BOX_MIN_FRAC, RANDOM_BOX_MAX_FRAC) == (30, 0.03, 0.25)
    max_side = int(512 * RANDOM_BOX_MAX_FRAC)
    assert 0 < first.mean().item() <= RANDOM_BOX_COUNT * max_side**2 / 512**2


class _AbsDifferenceLPIPS(nn.Module):
    """Stand-in for spatial LPIPS: per-pixel absolute difference."""

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))

    def forward(self, first, second):
        return (first - second).abs().mean(dim=1, keepdim=True)


def test_low_fidelity_selection_scores_only_inside_the_mask_by_default():
    reference = torch.zeros(1, 3, 32, 32)
    mask = torch.zeros(1, 1, 32, 32)
    mask[..., 8:24, 8:24] = 1.0
    outside_change = torch.ones(1, 3, 32, 32) * (1 - mask)
    inside_change = torch.ones(1, 3, 32, 32) * 0.5 * mask
    samples = torch.cat([outside_change, inside_change])

    scores = compute_spatial_lpips_batch(_AbsDifferenceLPIPS(), reference, samples, mask)

    assert scores[0].item() == 0.0
    assert int(scores.argmax()) == 1


def test_ic_lpips_clusters_around_training_references():
    source = (ROOT / "evaluate.py").read_text(encoding="utf-8")

    assert 'for entry in manifest["train"]]' in source
    assert "cluster_references," in source
    assert '"ic_lpips_reference": "train"' in source


def test_masked_loss_version_is_part_of_the_resume_fingerprint():
    source = (ROOT / "train.py").read_text(encoding="utf-8")

    assert source.count('"masked_loss": MASKED_LOSS_VERSION') == 2


def test_training_scales_the_fp16_loss_and_fingerprints_it():
    source = (ROOT / "train.py").read_text(encoding="utf-8")

    assert "grad_scaler.scale(total_loss).backward()" in source
    assert "grad_scaler.step(optimizer)" in source and "grad_scaler.update()" in source
    assert "total_loss.backward()" not in source
    assert source.count('"loss_scaling": LOSS_SCALING_VERSION') == 2
    assert source.count("grad_scaler=grad_scaler") == 4  # resume + three checkpoint saves


def test_checkpoint_round_trips_the_grad_scaler_state(tmp_path):
    from utils import load_checkpoint, save_checkpoint

    class _TinyBranch(nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_weight = nn.Parameter(torch.zeros(1))

    model = SimpleNamespace(placeholder_token="sks", pipeline=SimpleNamespace(text_encoder=_TinyBranch(), unet=_TinyBranch()))
    saved = torch.amp.GradScaler("cpu", init_scale=1024.0)
    save_checkpoint(model, None, 7, tmp_path / "checkpoint_7.pt", grad_scaler=saved)

    restored = torch.amp.GradScaler("cpu", init_scale=1.0)
    step = load_checkpoint(model, None, tmp_path / "checkpoint_7.pt", grad_scaler=restored)

    assert step == 7
    assert restored.get_scale() == 1024.0
