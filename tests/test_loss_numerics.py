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


def test_masked_mse_total_norm_averages_over_all_elements():
    noise_pred = torch.tensor([[[[2.0, 4.0]]]], dtype=torch.float16)
    noise = torch.tensor([[[[1.0, 1.0]]]], dtype=torch.float16)
    mask = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float16)

    loss = DefectFillModel.compute_masked_mse(noise_pred, noise, mask, normalize="total")

    # ||M ⊙ (ε − ε̂)||² = 1² + 0² over 2 elements
    assert torch.isclose(loss, torch.tensor(0.5))


def test_masked_mse_mask_norm_is_a_weighted_mean_over_the_mask_and_all_channels():
    noise = torch.zeros(1, 4, 1, 4)
    noise_pred = torch.ones(1, 4, 1, 4) * torch.tensor([1.0, 2.0, 3.0, 3.0])
    mask = torch.tensor([[[[1.0, 1.0, 0.0, 0.0]]]])  # one channel, broadcast over 4 latent channels

    loss = DefectFillModel.compute_masked_mse(noise_pred, noise, mask, normalize="mask")

    # mean of squared error over the masked elements only: (1 + 4) / 2, no factor of 4 from the channels
    assert torch.isclose(loss, torch.tensor(2.5))
