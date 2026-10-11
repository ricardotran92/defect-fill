import torch

from model import DefectFillModel


def test_masked_mse_is_finite_for_empty_fp16_mask():
    noise_pred = torch.zeros(1, 4, 64, 64, dtype=torch.float16)
    noise = torch.zeros_like(noise_pred)
    mask = torch.zeros(1, 1, 64, 64, dtype=torch.float16)

    loss = DefectFillModel.compute_masked_mse(noise_pred, noise, mask)

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    assert loss.item() == 0.0


def test_masked_mse_matches_masked_average_in_fp32():
    noise_pred = torch.tensor([[[[2.0, 4.0]]]], dtype=torch.float16)
    noise = torch.tensor([[[[1.0, 1.0]]]], dtype=torch.float16)
    mask = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float16)

    loss = DefectFillModel.compute_masked_mse(noise_pred, noise, mask)

    assert torch.isclose(loss, torch.tensor(1.0))
