import hashlib
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional


PROTOCOL_VERSION = "defectfill-paper-v1"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def _canonical_json(value: Dict) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def split_manifest_fingerprint(manifest: Dict) -> str:
    return hashlib.sha256(_canonical_json(manifest).encode("utf-8")).hexdigest()


def configuration_fingerprint(configuration: Dict) -> str:
    return hashlib.sha256(_canonical_json(configuration).encode("utf-8")).hexdigest()


def collect_standard_mvtec_pairs(
    data_dir,
    object_class: str,
    defect_type: str,
) -> List[Dict[str, str]]:
    root = Path(data_dir).resolve()
    image_dir = root / object_class / "test" / defect_type
    mask_dir = root / object_class / "ground_truth" / defect_type
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(
            f"Missing standard MVTec directories: {image_dir} and {mask_dir}"
        )

    pairs = []
    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        mask_path = mask_dir / f"{image_path.stem}_mask.png"
        if not mask_path.is_file():
            raise FileNotFoundError(f"Missing mask for {image_path.name}: {mask_path}")
        pairs.append(
            {
                "image": image_path.relative_to(root).as_posix(),
                "mask": mask_path.relative_to(root).as_posix(),
            }
        )
    return pairs


def validate_split_manifest(manifest: Dict, expected_data_dir: Optional[Path] = None) -> Dict:
    required = {
        "protocol_version",
        "data_dir",
        "object_class",
        "defect_type",
        "seed",
        "train",
        "heldout",
    }
    missing = required.difference(manifest)
    if missing:
        raise ValueError(f"Split manifest is missing keys: {sorted(missing)}")
    if manifest["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError(
            f"Unsupported protocol version: {manifest['protocol_version']}"
        )
    if expected_data_dir is not None:
        expected = os.path.normcase(str(Path(expected_data_dir).resolve()))
        actual = os.path.normcase(str(Path(manifest["data_dir"]).resolve()))
        if expected != actual:
            raise ValueError(f"Manifest data root mismatch: expected {expected}, got {actual}")

    train_images = {entry["image"] for entry in manifest["train"]}
    heldout_images = {entry["image"] for entry in manifest["heldout"]}
    if not manifest["train"] or not manifest["heldout"]:
        raise ValueError("Split manifest requires nonempty train and heldout partitions")
    if train_images.intersection(heldout_images):
        raise ValueError("Split manifest train and heldout partitions overlap")

    root = Path(manifest["data_dir"])
    for partition in ("train", "heldout"):
        for entry in manifest[partition]:
            image_path = root / Path(entry["image"])
            mask_path = root / Path(entry["mask"])
            if not image_path.is_file() or not mask_path.is_file():
                raise FileNotFoundError(
                    f"Manifest entry is missing files: {image_path}, {mask_path}"
                )
    return manifest


def build_split_manifest(
    data_dir,
    object_class: str,
    defect_type: str,
    seed: int = 42,
) -> Dict:
    root = Path(data_dir).resolve()
    pairs = collect_standard_mvtec_pairs(root, object_class, defect_type)
    if len(pairs) < 2:
        raise ValueError(
            f"Defect category requires at least two image-mask pairs, found {len(pairs)}"
        )

    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    train_count = max(1, len(shuffled) // 3)
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "data_dir": str(root),
        "object_class": object_class,
        "defect_type": defect_type,
        "seed": int(seed),
        "train": shuffled[:train_count],
        "heldout": shuffled[train_count:],
    }
    return validate_split_manifest(manifest, expected_data_dir=root)


def save_split_manifest(manifest: Dict, path) -> None:
    validate_split_manifest(manifest)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def load_split_manifest(path, expected_data_dir=None) -> Dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    return validate_split_manifest(manifest, expected_data_dir=expected_data_dir)


def resolve_manifest_pairs(manifest: Dict, partition: str) -> List[Dict[str, str]]:
    if partition not in {"train", "heldout"}:
        raise ValueError(f"Unknown manifest partition: {partition}")
    root = Path(manifest["data_dir"])
    return [
        {
            "image": str(root / Path(entry["image"])),
            "mask": str(root / Path(entry["mask"])),
        }
        for entry in manifest[partition]
    ]
