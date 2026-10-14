import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from model import LATENT_MASK_MODES, DefectFillModel
from protocol import PROTOCOL_VERSION, load_split_manifest, split_manifest_fingerprint
from utils import compute_spatial_lpips_batch, load_checkpoint


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def infer_split_manifest_path(checkpoint) -> Path:
    checkpoint = Path(checkpoint).resolve()
    return checkpoint.parent.parent / "split_manifest.json"


def prepare_full_view_inputs(
    image_path,
    mask_path,
    device,
    dtype,
    dilate_mask=False,
    mask_kernel_size=3,
):
    """Resize the complete image and mask to 512 without defect-centered cropping."""
    image = Image.open(image_path).convert("RGB")
    mask = np.asarray(Image.open(mask_path).convert("L"))

    if dilate_mask:
        kernel_size = mask_kernel_size if mask_kernel_size % 2 == 1 else mask_kernel_size + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

    image = image.resize((512, 512), resample=Image.Resampling.BICUBIC)
    mask_image = Image.fromarray(mask).resize((512, 512), resample=Image.Resampling.NEAREST)

    image_tensor = transforms.ToTensor()(image)
    image_tensor = transforms.Normalize([0.5] * 3, [0.5] * 3)(image_tensor)
    mask_tensor = torch.from_numpy((np.asarray(mask_image) > 0).astype(np.float32)).unsqueeze(0)
    return (
        image_tensor.unsqueeze(0).to(device=device, dtype=dtype),
        mask_tensor.unsqueeze(0).to(device=device, dtype=dtype),
    )


def calculate_generation_plan(num_good_images, num_masks, target_total=1000):
    if num_good_images <= 0 or num_masks <= 0:
        return []
    return [
        (output_index % num_good_images, output_index % num_masks, output_index)
        for output_index in range(target_total)
    ]


def write_json_atomic(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)


def generate_and_select_batch(
    model,
    clean_image,
    mask,
    object_class,
    num_samples=8,
    steps=50,
    guidance_scale=2.0,
    batch_size=4,
    seed=42,
    latent_mask_mode="maxpool",
    composite=True,
):
    """Generate paper-style candidates and retain the strongest masked LPIPS change."""
    prompt = f"A {object_class} with {model.placeholder_token}"
    all_samples = []
    for start_index in range(0, num_samples, batch_size):
        current_batch_size = min(batch_size, num_samples - start_index)
        batch_clean = clean_image.repeat(current_batch_size, 1, 1, 1)
        batch_mask = mask.repeat(current_batch_size, 1, 1, 1)
        generator = torch.Generator(device=clean_image.device).manual_seed(seed + start_index)
        generated = model.generate(
            image=batch_clean,
            mask=batch_mask,
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            latent_mask_mode=latent_mask_mode,
            composite=composite,
        )
        all_samples.append(generated * 2.0 - 1.0)

    samples = torch.cat(all_samples, dim=0)
    lpips_scores = compute_spatial_lpips_batch(
        model.get_lpips_model(),
        clean_image,
        samples,
        mask,
        smooth_boundary=False,  # paper: LPIPS "measured only within the masked region"
    )
    best_index = int(lpips_scores.argmax().item())
    return samples[best_index].clone(), float(lpips_scores[best_index].item())


def _load_existing_log(log_path, expected_fields):
    if not log_path.is_file():
        return None
    try:
        with log_path.open("r", encoding="utf-8") as handle:
            log = json.load(handle)
        if all(log.get(key) == value for key, value in expected_fields.items()):
            return log
    except (OSError, ValueError, TypeError):
        pass
    return None


def inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    manifest_path = Path(args.split_manifest) if args.split_manifest else infer_split_manifest_path(args.checkpoint)
    manifest = load_split_manifest(manifest_path, expected_data_dir=args.data_dir)
    if manifest["object_class"] != args.object_class or manifest["defect_type"] != args.defect_type:
        raise ValueError("Manifest object class or defect type does not match inference arguments")
    manifest_fingerprint = split_manifest_fingerprint(manifest)
    checkpoint_metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "split_manifest_fingerprint": manifest_fingerprint,
        "object_class": args.object_class,
        "defect_type": args.defect_type,
        "view_mode": "full",
    }

    model = DefectFillModel(
        device=device,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        base_model_path=args.base_model_path,
    )
    load_checkpoint(
        model,
        None,
        args.checkpoint,
        expected_metadata=checkpoint_metadata,
    )
    model.pipeline.unet.eval()
    model.pipeline.text_encoder.eval()
    model.pipeline.vae.to(dtype=dtype)

    root = Path(manifest["data_dir"])
    good_dir = root / args.object_class / "test" / "good"
    good_files = sorted(
        path for path in good_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    heldout_pairs = [
        {
            "image": root / Path(entry["image"]),
            "mask": root / Path(entry["mask"]),
        }
        for entry in manifest["heldout"]
    ]
    generation_plan = calculate_generation_plan(
        len(good_files),
        len(heldout_pairs),
        target_total=args.total_images,
    )
    if not generation_plan:
        raise ValueError("No normal images or held-out masks are available for generation")

    output_root = Path(args.output_dir)
    defect_output_dir = output_root / args.defect_type
    defect_output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "inference_log.json"
    resume_signature = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "protocol_version": PROTOCOL_VERSION,
        "split_manifest_fingerprint": manifest_fingerprint,
        "view_mode": "full",
        "object_class": args.object_class,
        "defect_type": args.defect_type,
        "num_samples": args.num_samples,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "dilate_mask": args.dilate_mask,
        "mask_kernel_size": args.mask_kernel_size,
        "background_preservation": "persistent_latent_noise_v1",
        "lfs_mask": "binary",
        "latent_mask_mode": args.latent_mask_mode,
        "composite": args.composite,
    }
    inference_log = _load_existing_log(log_path, resume_signature) or {
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "split_manifest": str(manifest_path.resolve()),
        "heldout_reference_count": len(heldout_pairs),
        "results": [],
        **resume_signature,
    }
    inference_log["target_images"] = args.total_images
    completed = {int(result["output_idx"]): result for result in inference_log["results"]}

    for good_index, mask_index, output_index in tqdm(generation_plan, desc=f"Generating {args.defect_type}"):
        generated_path = defect_output_dir / f"{output_index:04d}_generated.png"
        mask_output_path = defect_output_dir / f"{output_index:04d}_mask.png"
        original_output_path = defect_output_dir / f"{output_index:04d}_original.png"
        if (
            output_index in completed
            and generated_path.is_file()
            and mask_output_path.is_file()
            and original_output_path.is_file()
        ):
            continue

        good_path = good_files[good_index]
        heldout = heldout_pairs[mask_index]
        image_tensor, mask_tensor = prepare_full_view_inputs(
            good_path,
            heldout["mask"],
            device=device,
            dtype=dtype,
            dilate_mask=args.dilate_mask,
            mask_kernel_size=args.mask_kernel_size,
        )
        with torch.no_grad():
            defect_image, lpips_score = generate_and_select_batch(
                model,
                image_tensor,
                mask_tensor,
                args.object_class,
                num_samples=args.num_samples,
                steps=args.steps,
                guidance_scale=args.guidance_scale,
                batch_size=args.batch_size,
                seed=args.seed + output_index * args.num_samples,
                latent_mask_mode=args.latent_mask_mode,
                composite=args.composite,
            )

        save_image((defect_image.float() + 1.0) / 2.0, generated_path)
        save_image(mask_tensor.float(), mask_output_path)
        save_image((image_tensor.float() + 1.0) / 2.0, original_output_path)
        result = {
            "output_idx": output_index,
            "input_image": str(good_path),
            "source_mask": str(heldout["mask"]),
            "heldout_reference": str(heldout["image"]),
            "lpips_score": lpips_score,
        }
        completed[output_index] = result
        inference_log["results"] = [completed[index] for index in sorted(completed)]
        write_json_atomic(inference_log, log_path)
        if device.type == "cuda" and output_index % 10 == 0:
            torch.cuda.empty_cache()

    write_json_atomic(inference_log, log_path)
    print(f"Inference log saved to: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate DefectFill images with the paper protocol")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--base_model_path",
        type=str,
        default=r"D:\Novels_Code\stablediffusion\sd2_inpainting",
        help="Local Diffusers SD2 inpainting directory",
    )
    parser.add_argument("--data_dir", type=str, required=True, help="MVTec root")
    parser.add_argument("--object_class", type=str, required=True, help="Object class")
    parser.add_argument("--defect_type", type=str, required=True, help="Defect category")
    parser.add_argument("--split_manifest", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=8, help="Candidates per image")
    parser.add_argument("--steps", type=int, default=50, help="DDIM diffusion steps")
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--total_images", type=int, default=1000, help="Synthetic images to create")
    parser.add_argument("--batch_size", type=int, default=4, help="Parallel candidate batch size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--dilate_mask", type=str, default="False")
    parser.add_argument("--mask_kernel_size", type=int, default=3)
    parser.add_argument(
        "--latent_mask_mode",
        choices=LATENT_MASK_MODES,
        default="maxpool",
        help="Mask downsampling to latent resolution: maxpool keeps thin mask branches; nearest = original code",
    )
    parser.add_argument(
        "--composite",
        choices=("on", "off"),
        default="on",
        help="on: keep original pixels outside the inpainted area (no VAE colour drift); off = original code",
    )
    args = parser.parse_args()
    args.dilate_mask = args.dilate_mask.lower() == "true"
    args.composite = args.composite == "on"
    inference(args)
