import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference import (  # noqa: E402
    _load_existing_log,
    infer_split_manifest_path,
    prepare_full_view_inputs,
)


def test_manifest_path_is_inferred_from_checkpoint_run_directory(tmp_path):
    checkpoint = tmp_path / "run" / "checkpoints" / "checkpoint_100.pt"

    assert infer_split_manifest_path(checkpoint) == tmp_path / "run" / "split_manifest.json"


def test_inference_resizes_the_whole_image_and_mask_without_cropping(tmp_path):
    image = np.zeros((1024, 1024, 3), dtype=np.uint8)
    image[:, :512] = (255, 0, 0)
    mask = np.zeros((1024, 1024), dtype=np.uint8)
    mask[32:160, 32:160] = 255
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    Image.fromarray(image).save(image_path)
    Image.fromarray(mask).save(mask_path)

    image_tensor, mask_tensor = prepare_full_view_inputs(
        image_path,
        mask_path,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert tuple(image_tensor.shape) == (1, 3, 512, 512)
    assert tuple(mask_tensor.shape) == (1, 1, 512, 512)
    assert float(mask_tensor.sum()) > 0
    assert image_tensor[0, 0, 256, 32] > 0.9
    assert image_tensor[0, 0, 256, 480] < -0.9


def test_inference_defaults_and_metadata_match_the_paper_protocol():
    source = (ROOT / "inference.py").read_text(encoding="utf-8")

    assert 'parser.add_argument("--num_samples", type=int, default=8' in source
    assert 'parser.add_argument("--steps", type=int, default=50' in source
    assert 'parser.add_argument("--guidance_scale", type=float, default=2.0)' in source
    assert 'parser.add_argument("--total_images", type=int, default=1000' in source
    assert '"view_mode": "full"' in source
    assert '"split_manifest_fingerprint"' in source
    assert "expected_metadata=checkpoint_metadata" in source


def test_inference_resume_rejects_changed_sampling_parameters(tmp_path):
    log_path = tmp_path / "inference_log.json"
    log_path.write_text(
        '{"split_manifest_fingerprint":"abc","view_mode":"full","steps":2}',
        encoding="utf-8",
    )

    assert _load_existing_log(
        log_path,
        {"split_manifest_fingerprint": "abc", "view_mode": "full", "steps": 2},
    ) is not None
    assert _load_existing_log(
        log_path,
        {"split_manifest_fingerprint": "abc", "view_mode": "full", "steps": 50},
    ) is None
