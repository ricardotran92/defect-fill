"""token_attention_map: the [V*] attention shared by the attention loss and tools/attention_probe.py."""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import DefectFillModel, attention_mask_loss  # noqa: E402


def _host(token_index=4):
    torch.manual_seed(0)
    maps = {
        "down_blocks.0.attentions.0.attn2": torch.rand(1, 2, 16, 77),  # encoder layer: must be ignored
        "up_blocks.1.attentions.0.attn2": torch.rand(1, 2, 16, 77),    # 4x4
        "up_blocks.2.attentions.0.attn2": torch.rand(1, 2, 64, 77),    # 8x8
    }
    host = SimpleNamespace(attention_maps=maps, defect_token_indices=[token_index], attention_loss_norm="region")
    host.token_attention_map = lambda index, size: DefectFillModel.token_attention_map(host, index, size)
    return host


def test_averages_heads_then_decoder_layers_only():
    host = _host()
    expected = torch.stack([
        F.interpolate(host.attention_maps[name][0, :, :, 4].mean(0).reshape(1, 1, side, side), size=(8, 8),
                      mode="bilinear", align_corners=False).squeeze()
        for name, side in (("up_blocks.1.attentions.0.attn2", 4), ("up_blocks.2.attentions.0.attn2", 8))
    ]).mean(0)

    assert torch.allclose(DefectFillModel.token_attention_map(host, 0, (8, 8)), expected)


def test_missing_token_or_maps_give_none():
    assert DefectFillModel.token_attention_map(_host(token_index=-1), 0, (8, 8)) is None
    empty = SimpleNamespace(attention_maps={}, defect_token_indices=[4])
    assert DefectFillModel.token_attention_map(empty, 0, (8, 8)) is None


def test_attention_loss_uses_the_same_map():
    host = _host()
    mask = torch.zeros(1, 1, 8, 8)
    mask[..., 2:5, 2:5] = 1.0

    loss = DefectFillModel.get_attention_loss(host, mask)

    assert torch.isclose(loss, attention_mask_loss(DefectFillModel.token_attention_map(host, 0, (8, 8)), mask[0, 0], "region"))
