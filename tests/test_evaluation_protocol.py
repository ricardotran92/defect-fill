import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate import (  # noqa: E402
    collect_generated_images,
    reference_cluster_statistics,
    validate_inference_log,
)


def test_reference_cluster_ic_lpips_averages_pairwise_diversity_per_cluster():
    generated_to_reference = np.array(
        [
            [0.1, 0.9],
            [0.2, 0.8],
            [0.8, 0.2],
            [0.9, 0.1],
        ]
    )
    generated_pairwise = np.array(
        [
            [0.0, 0.4, 0.7, 0.7],
            [0.4, 0.0, 0.7, 0.7],
            [0.7, 0.7, 0.0, 0.8],
            [0.7, 0.7, 0.8, 0.0],
        ]
    )

    result = reference_cluster_statistics(generated_to_reference, generated_pairwise)

    assert result["cluster_sizes"] == [2, 2]
    assert result["cluster_scores"] == pytest.approx([0.4, 0.8])
    assert result["mean"] == pytest.approx(0.6)
    assert result["std"] == pytest.approx(0.2)


def test_reference_cluster_ic_lpips_ignores_singleton_clusters():
    generated_to_reference = np.array([[0.1, 0.9], [0.8, 0.2], [0.7, 0.3]])
    generated_pairwise = np.array(
        [[0.0, 0.6, 0.5], [0.6, 0.0, 0.2], [0.5, 0.2, 0.0]]
    )

    result = reference_cluster_statistics(generated_to_reference, generated_pairwise)

    assert result["cluster_sizes"] == [1, 2]
    assert result["cluster_scores"] == pytest.approx([0.2])
    assert result["mean"] == pytest.approx(0.2)


def test_generated_collection_is_sorted_and_excludes_support_files(tmp_path):
    (tmp_path / "0002_generated.png").write_bytes(b"x")
    (tmp_path / "0001_generated.png").write_bytes(b"x")
    (tmp_path / "0001_mask.png").write_bytes(b"x")

    assert [Path(path).name for path in collect_generated_images(tmp_path)] == [
        "0001_generated.png",
        "0002_generated.png",
    ]


def test_evaluation_rejects_mismatched_or_cropped_inference_log():
    log = {"split_manifest_fingerprint": "abc", "view_mode": "crop"}

    with pytest.raises(ValueError, match="view_mode"):
        validate_inference_log(log, expected_fingerprint="abc")

    log["view_mode"] = "full"
    with pytest.raises(ValueError, match="fingerprint"):
        validate_inference_log(log, expected_fingerprint="different")


def test_evaluation_reports_raw_and_paper_scaled_kid():
    source = (ROOT / "evaluate.py").read_text(encoding="utf-8")

    assert '"kid_x1000"' in source
    assert '"kid_raw"' in source
    assert 'manifest["heldout"]' in source
