"""Generation plan that favours chosen held-out masks (e.g. thin-branch masks) in a fixed ratio."""
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference import calculate_generation_plan, resolve_priority_masks  # noqa: E402

PRIORITY = [1, 2, 5, 7, 0]  # positions of 006/010/011/014/016 in the hazelnut/hole held-out list


def test_without_priority_masks_the_plan_is_unchanged():
    plan = calculate_generation_plan(40, 12, target_total=100)

    assert plan == [(i % 40, i % 12, i) for i in range(100)]


def test_three_to_one_plan_puts_75_of_100_outputs_on_the_priority_masks():
    plan = calculate_generation_plan(40, 12, target_total=100, priority_indices=PRIORITY, priority_ratio=3)
    counts = Counter(mask for _, mask, _ in plan)

    assert sum(counts[m] for m in PRIORITY) == 75
    assert all(counts[m] == 15 for m in PRIORITY)
    others = [m for m in range(12) if m not in PRIORITY]
    assert sum(counts[m] for m in others) == 25
    assert all(counts[m] in (3, 4) for m in others)
    assert [mask for _, mask, _ in plan[:4]] == [1, 2, 5, 3]  # three priority, then one other
    assert [good for good, _, _ in plan] == [i % 40 for i in range(100)]


def test_plan_is_identical_for_every_run_with_the_same_settings():
    first = calculate_generation_plan(40, 12, 100, PRIORITY, 3)
    second = calculate_generation_plan(40, 12, 100, PRIORITY, 3)

    assert first == second


def test_priority_masks_are_resolved_by_file_name_and_unknown_names_fail():
    pairs = [{"mask": f"/data/hazelnut/ground_truth/hole/{name}"} for name in ("016_mask.png", "006_mask.png", "010_mask.png")]

    assert resolve_priority_masks(pairs, ["006_mask.png", "016_mask.png"]) == [1, 0]
    assert resolve_priority_masks(pairs, []) == []
    with pytest.raises(ValueError):
        resolve_priority_masks(pairs, ["999_mask.png"])


def test_priority_settings_are_part_of_the_inference_signature():
    source = (ROOT / "inference.py").read_text(encoding="utf-8")

    assert '"priority_masks": args.priority_masks' in source
    assert '"priority_ratio": args.priority_ratio if args.priority_masks else None' in source
