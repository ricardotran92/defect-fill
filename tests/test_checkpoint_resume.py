import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils import find_latest_valid_checkpoint, load_checkpoint, save_checkpoint  # noqa: E402


class _TinyBranch(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.lora_weight = torch.nn.Parameter(torch.tensor([value], dtype=torch.float32))


def _make_model():
    return SimpleNamespace(
        placeholder_token="sks",
        pipeline=SimpleNamespace(
            text_encoder=_TinyBranch(1.0),
            unet=_TinyBranch(2.0),
        ),
    )


def test_checkpoint_round_trip_restores_weights_optimizer_rng_and_metadata(tmp_path):
    model = _make_model()
    optimizer = torch.optim.AdamW(
        list(model.pipeline.text_encoder.parameters()) + list(model.pipeline.unet.parameters()),
        lr=1e-3,
    )
    metadata = {"protocol_fingerprint": "abc123", "object_class": "hazelnut"}
    path = tmp_path / "checkpoint_12.pt"

    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    save_checkpoint(model, optimizer, 12, path, metadata=metadata)
    expected_rng_values = (random.random(), np.random.rand(), torch.rand(1))

    with torch.no_grad():
        model.pipeline.text_encoder.lora_weight.fill_(99)
        model.pipeline.unet.lora_weight.fill_(99)
    random.random()
    np.random.rand()
    torch.rand(1)

    step = load_checkpoint(
        model,
        optimizer,
        path,
        expected_metadata=metadata,
        restore_rng=True,
    )

    assert step == 12
    assert model.pipeline.text_encoder.lora_weight.item() == pytest.approx(1.0)
    assert model.pipeline.unet.lora_weight.item() == pytest.approx(2.0)
    assert random.random() == pytest.approx(expected_rng_values[0])
    assert np.random.rand() == pytest.approx(expected_rng_values[1])
    assert torch.equal(torch.rand(1), expected_rng_values[2])
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert checkpoint["metadata"] == metadata
    assert "learned_embedding" not in checkpoint
    assert not list(tmp_path.glob("*.tmp"))


def test_checkpoint_metadata_mismatch_is_rejected(tmp_path):
    model = _make_model()
    path = tmp_path / "checkpoint_3.pt"
    save_checkpoint(model, None, 3, path, metadata={"protocol_fingerprint": "first"})

    with pytest.raises(ValueError, match="protocol_fingerprint"):
        load_checkpoint(
            model,
            None,
            path,
            expected_metadata={"protocol_fingerprint": "second"},
        )


def test_latest_valid_checkpoint_skips_a_corrupt_newer_file(tmp_path):
    model = _make_model()
    metadata = {"protocol_fingerprint": "abc123"}
    older = tmp_path / "checkpoint_10.pt"
    corrupt_newer = tmp_path / "checkpoint_20.pt"
    save_checkpoint(model, None, 10, older, metadata=metadata)
    corrupt_newer.write_bytes(b"incomplete checkpoint")

    selected = find_latest_valid_checkpoint(tmp_path, expected_metadata=metadata)

    assert selected == str(older)


def test_latest_valid_checkpoint_uses_internal_step_not_final_filename(tmp_path):
    model = _make_model()
    metadata = {"protocol_fingerprint": "abc123"}
    stale_final = tmp_path / "checkpoint_final.pt"
    newer_periodic = tmp_path / "checkpoint_20.pt"
    save_checkpoint(model, None, 10, stale_final, metadata=metadata)
    save_checkpoint(model, None, 20, newer_periodic, metadata=metadata)

    selected = find_latest_valid_checkpoint(tmp_path, expected_metadata=metadata)

    assert selected == str(newer_periodic)
