"""Paper-protocol KID and reference-clustered IC-LPIPS evaluation."""

import argparse
import csv
import json
import os
from datetime import datetime
from itertools import combinations
from pathlib import Path

import lpips
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm

from protocol import PROTOCOL_VERSION, load_split_manifest, split_manifest_fingerprint


class KIDEvaluator:
    def __init__(self, device="cuda"):
        self.device = device
        self.inception = models.inception_v3(
            weights=models.Inception_V3_Weights.IMAGENET1K_V1,
            transform_input=False,
        )
        self.inception.fc = nn.Identity()
        self.inception = self.inception.to(device).eval()
        self.preprocess = transforms.Compose(
            [
                transforms.Resize((299, 299), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    @torch.no_grad()
    def extract_features(self, images, batch_size=32):
        features = []
        for start in tqdm(range(0, len(images), batch_size), desc="Inception features"):
            batch_paths = images[start : start + batch_size]
            batch = torch.stack(
                [self.preprocess(Image.open(path).convert("RGB")) for path in batch_paths]
            ).to(self.device)
            features.append(self.inception(batch).cpu())
        return torch.cat(features, dim=0)

    @staticmethod
    def polynomial_kernel(x, y, degree=3, gamma=None, coef0=1.0):
        gamma = 1.0 / x.shape[1] if gamma is None else gamma
        return (gamma * torch.mm(x, y.t()) + coef0) ** degree

    @classmethod
    def compute_mmd(cls, x, y):
        if len(x) < 2 or len(y) < 2:
            raise ValueError("Unbiased KID requires at least two real and two generated images")
        k_xx = cls.polynomial_kernel(x, x)
        k_yy = cls.polynomial_kernel(y, y)
        k_xy = cls.polynomial_kernel(x, y)
        n, m = len(x), len(y)
        return (
            (k_xx.sum() - k_xx.trace()) / (n * (n - 1))
            + (k_yy.sum() - k_yy.trace()) / (m * (m - 1))
            - 2.0 * k_xy.mean()
        )

    def compute_kid(self, real_images, generated_images, num_subsets=100, subset_size=None, seed=42):
        real_features = self.extract_features(real_images)
        generated_features = self.extract_features(generated_images)
        subset_size = subset_size or min(len(real_features), len(generated_features), 1000)
        if subset_size < 2:
            raise ValueError("KID subset size must be at least two")

        rng = np.random.default_rng(seed)
        scores = []
        for _ in range(num_subsets):
            real_indices = rng.choice(len(real_features), subset_size, replace=False)
            generated_indices = rng.choice(len(generated_features), subset_size, replace=False)
            scores.append(
                self.compute_mmd(
                    real_features[real_indices],
                    generated_features[generated_indices],
                ).item()
            )
        return float(np.mean(scores)), float(np.std(scores))


def reference_cluster_statistics(generated_to_reference, generated_pairwise):
    """Compute equal-weight mean intra-cluster pairwise LPIPS over valid clusters."""
    generated_to_reference = np.asarray(generated_to_reference, dtype=np.float64)
    generated_pairwise = np.asarray(generated_pairwise, dtype=np.float64)
    if generated_to_reference.ndim != 2:
        raise ValueError("generated_to_reference must be a 2D distance matrix")
    if generated_pairwise.shape != (len(generated_to_reference), len(generated_to_reference)):
        raise ValueError("generated_pairwise must be a square generated-image distance matrix")

    assignments = generated_to_reference.argmin(axis=1)
    cluster_sizes = []
    cluster_scores = []
    for reference_index in range(generated_to_reference.shape[1]):
        members = np.flatnonzero(assignments == reference_index)
        cluster_sizes.append(int(len(members)))
        if len(members) < 2:
            continue
        cluster_matrix = generated_pairwise[np.ix_(members, members)]
        pair_scores = cluster_matrix[np.triu_indices(len(members), k=1)]
        pair_scores = pair_scores[np.isfinite(pair_scores)]
        if len(pair_scores):
            cluster_scores.append(float(pair_scores.mean()))

    return {
        "mean": float(np.mean(cluster_scores)) if cluster_scores else float("nan"),
        "std": float(np.std(cluster_scores)) if cluster_scores else float("nan"),
        "assignments": assignments.tolist(),
        "cluster_sizes": cluster_sizes,
        "cluster_scores": cluster_scores,
        "valid_clusters": len(cluster_scores),
    }


class ICLPIPSEvaluator:
    def __init__(self, net="vgg", device="cuda"):
        self.device = torch.device(device)
        self.lpips_net = lpips.LPIPS(net=net, spatial=False).to(self.device).eval()
        self.preprocess = transforms.Compose(
            [
                transforms.Resize((256, 256), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )

    def _load_images(self, paths):
        return [self.preprocess(Image.open(path).convert("RGB")) for path in paths]

    @torch.no_grad()
    def _score_pairs(self, first_images, second_images, pairs, batch_size=16):
        scores = []
        for start in tqdm(range(0, len(pairs), batch_size), desc="LPIPS pairs", leave=False):
            batch_pairs = pairs[start : start + batch_size]
            first = torch.stack([first_images[i] for i, _ in batch_pairs]).to(self.device)
            second = torch.stack([second_images[j] for _, j in batch_pairs]).to(self.device)
            batch_scores = self.lpips_net(first, second).reshape(-1).float().cpu().numpy()
            scores.extend(batch_scores.tolist())
        return scores

    def compute_ic_lpips(
        self,
        generated_images,
        reference_images,
        max_pairs_per_cluster=None,
        seed=42,
    ):
        if len(generated_images) < 2 or not reference_images:
            raise ValueError("IC-LPIPS requires at least two generated images and one reference")

        generated = self._load_images(generated_images)
        references = self._load_images(reference_images)
        assignment_pairs = [
            (generated_index, reference_index)
            for generated_index in range(len(generated))
            for reference_index in range(len(references))
        ]
        assignment_scores = self._score_pairs(generated, references, assignment_pairs)
        generated_to_reference = np.asarray(assignment_scores).reshape(
            len(generated), len(references)
        )
        assignments = generated_to_reference.argmin(axis=1)

        pairwise = np.full((len(generated), len(generated)), np.nan, dtype=np.float64)
        np.fill_diagonal(pairwise, 0.0)
        rng = np.random.default_rng(seed)
        for reference_index in range(len(references)):
            members = np.flatnonzero(assignments == reference_index).tolist()
            cluster_pairs = list(combinations(members, 2))
            if max_pairs_per_cluster and len(cluster_pairs) > max_pairs_per_cluster:
                selected = rng.choice(len(cluster_pairs), max_pairs_per_cluster, replace=False)
                cluster_pairs = [cluster_pairs[index] for index in selected]
            if not cluster_pairs:
                continue
            scores = self._score_pairs(generated, generated, cluster_pairs)
            for (first_index, second_index), score in zip(cluster_pairs, scores):
                pairwise[first_index, second_index] = score
                pairwise[second_index, first_index] = score

        return reference_cluster_statistics(generated_to_reference, pairwise)


def collect_generated_images(directory):
    return sorted(
        str(path)
        for path in Path(directory).rglob("*_generated.png")
        if path.is_file()
    )


def validate_inference_log(log, expected_fingerprint):
    if log.get("view_mode") != "full":
        raise ValueError(f"Inference view_mode must be 'full', got {log.get('view_mode')!r}")
    if log.get("split_manifest_fingerprint") != expected_fingerprint:
        raise ValueError("Inference log split manifest fingerprint does not match evaluation")
    if log.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Inference log protocol version does not match evaluation")
    return log


def _find_inference_log(generated_dir, explicit_path=None):
    if explicit_path:
        return Path(explicit_path)
    generated_dir = Path(generated_dir).resolve()
    candidates = [
        generated_dir / "inference_log.json",
        generated_dir.parent / "inference_log.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Could not find inference_log.json beside generated images")


def _append_csv(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(result)
    file_exists = path.is_file()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = load_split_manifest(args.split_manifest)
    if manifest["object_class"] != args.object_class or manifest["defect_type"] != args.defect_type:
        raise ValueError("Manifest object class or defect type does not match evaluation arguments")
    manifest_fingerprint = split_manifest_fingerprint(manifest)

    inference_log_path = _find_inference_log(args.generated_dir, args.inference_log)
    with inference_log_path.open("r", encoding="utf-8") as handle:
        inference_log = validate_inference_log(json.load(handle), manifest_fingerprint)

    root = Path(manifest["data_dir"])
    real_images = [str(root / Path(entry["image"])) for entry in manifest["heldout"]]
    generated_images = collect_generated_images(args.generated_dir)
    if len(real_images) < 2 or len(generated_images) < 2:
        raise ValueError("Evaluation requires at least two held-out real and generated images")

    kid_evaluator = KIDEvaluator(device=device)
    kid_raw, kid_raw_std = kid_evaluator.compute_kid(
        real_images,
        generated_images,
        num_subsets=args.kid_subsets,
        subset_size=min(len(real_images), len(generated_images), args.kid_subset_size),
        seed=args.seed,
    )
    del kid_evaluator
    if device.type == "cuda":
        torch.cuda.empty_cache()

    ic_evaluator = ICLPIPSEvaluator(device=device)
    ic_result = ic_evaluator.compute_ic_lpips(
        generated_images,
        real_images,
        max_pairs_per_cluster=args.max_pairs_per_cluster or None,
        seed=args.seed,
    )

    result = {
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "object_class": args.object_class,
        "defect_type": args.defect_type,
        "config": args.config_name,
        "category_type": args.category_type,
        "protocol_version": PROTOCOL_VERSION,
        "split_manifest_fingerprint": manifest_fingerprint,
        "view_mode": "full",
        "real_count": len(real_images),
        "generated_count": len(generated_images),
        "kid_raw": kid_raw,
        "kid_raw_std": kid_raw_std,
        "kid_x1000": kid_raw * 1000.0,
        "kid_x1000_std": kid_raw_std * 1000.0,
        "ic_lpips": ic_result["mean"],
        "ic_lpips_cluster_std": ic_result["std"],
        "ic_lpips_valid_clusters": ic_result["valid_clusters"],
        "ic_lpips_cluster_sizes": json.dumps(ic_result["cluster_sizes"]),
        "inference_target_images": inference_log.get("target_images"),
    }
    _append_csv(args.output_csv, result)
    print(
        f"KID: {result['kid_x1000']:.3f} x10^-3 "
        f"(raw {kid_raw:.6f}); IC-LPIPS: {ic_result['mean']:.4f}"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DefectFill with the paper protocol")
    parser.add_argument("--generated_dir", type=str, required=True)
    parser.add_argument("--split_manifest", type=str, required=True)
    parser.add_argument("--inference_log", type=str, default=None)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--object_class", type=str, required=True)
    parser.add_argument("--defect_type", type=str, required=True)
    parser.add_argument("--config_name", type=str, default="paper")
    parser.add_argument(
        "--category_type",
        type=str,
        default="unknown",
        choices=["object", "texture", "unknown"],
    )
    parser.add_argument("--kid_subsets", type=int, default=100)
    parser.add_argument("--kid_subset_size", type=int, default=1000)
    parser.add_argument("--max_pairs_per_cluster", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    evaluate(parser.parse_args())
