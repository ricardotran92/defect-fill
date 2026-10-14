"""Measure how much of each mask a generation run actually paints, and how much the background drifts.

Targets the two weaknesses noted in the README: thin mask branches left unpainted, and colour
changes outside the mask. For every <i>_generated.png with its <i>_original.png and <i>_mask.png:

- fill_thick / fill_thin: share of mask pixels whose colour changed by more than THRESHOLD (0-255)
  in the thick part (morphological opening with a 7x7 kernel) and the thin part (mask minus opening)
- drift_ring: mean absolute change in an 8-pixel band just outside the mask
- drift_far: mean absolute change in the background farther than 32 pixels from the mask (ideal: 0)

Usage: python tools/mask_fill.py <generated_dir> [--limit N]   (prints JSON)
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

THRESHOLD = 25
THICK_KERNEL = 7
RING_RADIUS = 8
FAR_RADIUS = 32
MIN_THIN_PIXELS = 30


def _image_stats(generated_path):
    stem = generated_path.name[: -len("_generated.png")]
    folder = generated_path.parent
    generated = cv2.imread(str(generated_path)).astype(np.float32)
    original = cv2.imread(str(folder / f"{stem}_original.png")).astype(np.float32)
    mask = cv2.imread(str(folder / f"{stem}_mask.png"), cv2.IMREAD_GRAYSCALE) > 127
    if not mask.any():
        return None

    mask_u8 = mask.astype(np.uint8)
    thick = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, np.ones((THICK_KERNEL, THICK_KERNEL), np.uint8)) > 0
    thin = mask & ~thick
    ring = (cv2.dilate(mask_u8, np.ones((2 * RING_RADIUS + 1,) * 2, np.uint8)) > 0) & ~mask
    far = ~(cv2.dilate(mask_u8, np.ones((2 * FAR_RADIUS + 1,) * 2, np.uint8)) > 0)
    change = np.abs(generated - original).mean(axis=2)
    changed = change > THRESHOLD
    return {
        "fill_thick": float(changed[thick].mean()) if thick.any() else None,
        "fill_thin": float(changed[thin].mean()) if thin.sum() >= MIN_THIN_PIXELS else None,
        "drift_ring": float(change[ring].mean()),
        "drift_far": float(change[far].mean()),
    }


def mask_fill_stats(generated_dir, limit=None):
    paths = sorted(Path(generated_dir).rglob("*_generated.png"))[:limit]
    per_image = [stats for stats in map(_image_stats, paths) if stats is not None]
    summary = {"n_images": len(per_image), "threshold": THRESHOLD}
    for key in ("fill_thick", "fill_thin", "drift_ring", "drift_far"):
        values = [stats[key] for stats in per_image if stats[key] is not None]
        summary[key] = float(np.mean(values)) if values else None
        if key == "fill_thin":
            summary["n_images_with_thin_part"] = len(values)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("generated_dir")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(mask_fill_stats(args.generated_dir, args.limit), indent=2))
