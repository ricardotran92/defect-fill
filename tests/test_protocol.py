import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_loader import MVTecDefectDataset, build_train_transform  # noqa: E402
from protocol import (  # noqa: E402
    PROTOCOL_VERSION,
    build_split_manifest,
    configuration_fingerprint,
    load_split_manifest,
    save_split_manifest,
    split_manifest_fingerprint,
)


def _write_rgb(path: Path, value: int = 128):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((1024, 1024, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def _write_mask(path: Path, index: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((1024, 1024), dtype=np.uint8)
    start = 8 + index
    mask[start : start + 48, start : start + 48] = 255
    assert cv2.imwrite(str(path), mask)


def _make_mvtec_category(root: Path, count: int = 8):
    for index in range(count):
        _write_rgb(root / "hazelnut" / "test" / "hole" / f"{index:03d}.png", 80 + index)
        _write_mask(
            root / "hazelnut" / "ground_truth" / "hole" / f"{index:03d}_mask.png",
            index,
        )


def test_split_is_reproducible_disjoint_and_one_third(tmp_path):
    root = tmp_path / "MVTec"
    _make_mvtec_category(root, count=8)

    first = build_split_manifest(root, "hazelnut", "hole", seed=42)
    second = build_split_manifest(root, "hazelnut", "hole", seed=42)

    assert first == second
    assert first["protocol_version"] == PROTOCOL_VERSION
    assert len(first["train"]) == 2
    assert len(first["heldout"]) == 6

    train_images = {entry["image"] for entry in first["train"]}
    heldout_images = {entry["image"] for entry in first["heldout"]}
    assert train_images.isdisjoint(heldout_images)


def test_split_manifest_round_trip_preserves_fingerprint(tmp_path):
    root = tmp_path / "MVTec"
    _make_mvtec_category(root, count=6)
    manifest = build_split_manifest(root, "hazelnut", "hole", seed=7)
    path = tmp_path / "split_manifest.json"

    save_split_manifest(manifest, path)
    loaded = load_split_manifest(path, expected_data_dir=root)

    assert loaded == manifest
    assert split_manifest_fingerprint(loaded) == split_manifest_fingerprint(manifest)
    assert json.loads(path.read_text(encoding="utf-8"))["protocol_version"] == PROTOCOL_VERSION


def test_training_sample_is_full_view_512_and_mask_remains_nonempty(tmp_path):
    root = tmp_path / "MVTec"
    _make_mvtec_category(root, count=3)
    manifest = build_split_manifest(root, "hazelnut", "hole", seed=42)
    dataset = MVTecDefectDataset(
        root_dir=str(root),
        object_class="hazelnut",
        split="train",
        defect_type="hole",
        transform=build_train_transform(),
        pairs=manifest["train"],
    )

    for _ in range(20):
        sample = dataset[0]
        assert tuple(sample["image"].shape) == (3, 512, 512)
        assert tuple(sample["mask"].shape) == (512, 512)
        assert float(sample["mask"].sum()) > 0
        assert sample["view_mode"] == "full"


def test_manifest_rejects_too_few_pairs(tmp_path):
    root = tmp_path / "MVTec"
    _make_mvtec_category(root, count=1)

    try:
        build_split_manifest(root, "hazelnut", "hole", seed=42)
    except ValueError as exc:
        assert "at least two" in str(exc)
    else:
        raise AssertionError("expected a category with one pair to be rejected")


def test_configuration_fingerprint_is_order_independent():
    first = {"rank": 8, "loss": {"object": 0.2, "defect": 0.5}}
    second = {"loss": {"defect": 0.5, "object": 0.2}, "rank": 8}

    assert configuration_fingerprint(first) == configuration_fingerprint(second)
