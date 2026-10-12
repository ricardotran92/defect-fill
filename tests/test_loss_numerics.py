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


def test_masked_mse_is_paper_squared_norm_averaged_over_all_elements():
    noise_pred = torch.tensor([[[[2.0, 4.0]]]], dtype=torch.float16)
    noise = torch.tensor([[[[1.0, 1.0]]]], dtype=torch.float16)
    mask = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float16)

    loss = DefectFillModel.compute_masked_mse(noise_pred, noise, mask)

    # ||M ⊙ (ε − ε̂)||² = 1² + 0² over 2 elements
    assert torch.isclose(loss, torch.tensor(0.5))
