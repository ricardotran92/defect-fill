"""Where does the [V*] token look? Decoder cross-attention of `sks` on thin vs thick mask parts (server, GPU).

For each checkpoint, each (normal image, held-out mask) pair and each timestep, run one 9-channel UNet
forward with the object prompt "A <object> with sks" (as at generation time), take the [V*] attention
averaged over heads and decoder layers (DefectFillModel.token_attention_map), and resize it to 512 px.
Report the mean attention on the thick part of the mask (7x7 opening), on the thin part (mask minus
opening) and on the background farther than 32 px from the mask.

  python tools/attention_probe.py --checkpoint v6=runs/.../checkpoint_final.pt --checkpoint v5=... \
    --base_model_path $SHARED/models/sd2_inpainting --data_dir $SHARED/datasets/mvtec_ad \
    --split_manifest runs/hazelnut_hole_paper_2000step_v6/split_manifest.json \
    --mask_names 006_mask.png,011_mask.png --output_dir runs/diagnostics/attention_probe

Writes summary.json (per case and averaged per checkpoint) and attention_grid.jpg
(rows: cases; columns: image with mask outline, then one heatmap per checkpoint, each scaled to its own max).
"""
import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference import prepare_full_view_inputs  # noqa: E402
from model import DefectFillModel, mask_to_latent  # noqa: E402
from protocol import load_split_manifest  # noqa: E402
from utils import load_checkpoint  # noqa: E402

THICK_KERNEL = 7
FAR_RADIUS = 32
TILE = 256


def mask_regions(mask):
    mask_u8 = mask.astype(np.uint8)
    thick = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, np.ones((THICK_KERNEL, THICK_KERNEL), np.uint8)) > 0
    far = ~(cv2.dilate(mask_u8, np.ones((2 * FAR_RADIUS + 1,) * 2, np.uint8)) > 0)
    return thick, mask & ~thick, far


@torch.no_grad()
def probe_case(model, image, mask, prompt, timesteps, seed):
    vae = model.pipeline.vae
    scale = vae.config.scaling_factor
    latents = vae.encode(image).latent_dist.mode() * scale
    background = vae.encode(image * (1 - mask)).latent_dist.mode() * scale
    mask_latents = mask_to_latent(mask, latents.shape[-2:], "maxpool")
    embeddings = model.get_text_embeddings([prompt], enable_grad=False)
    maps = []
    for t in timesteps:
        noise = torch.randn(latents.shape, generator=torch.Generator(device=latents.device).manual_seed(seed + t),
                            device=latents.device, dtype=latents.dtype)
        timestep = torch.tensor([t], device=latents.device)
        noisy = model.scheduler.add_noise(latents, noise, timestep)
        model(noisy, background, mask_latents, timestep, embeddings)
        attention = model.token_attention_map(0, latents.shape[-2:])
        maps.append(attention.float())
    attention = torch.stack(maps).mean(0)[None, None]
    return torch.nn.functional.interpolate(attention, size=mask.shape[-2:], mode="bilinear", align_corners=False)[0, 0].cpu().numpy()


def region_stats(attention, mask):
    thick, thin, far = mask_regions(mask)
    stats = {"thick": float(attention[thick].mean()) if thick.any() else None,
             "thin": float(attention[thin].mean()) if thin.sum() >= 30 else None,
             "background": float(attention[far].mean())}
    if stats["thick"] and stats["thin"] is not None:
        stats["thin_over_thick"] = stats["thin"] / stats["thick"]
    stats["mask_over_background"] = float(attention[mask].mean()) / max(stats["background"], 1e-8)
    return stats


def tile_image(image_rgb, mask):
    outline = cv2.drawContours(image_rgb.copy(), cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_NONE)[0], -1, (255, 255, 255), 2)
    return cv2.resize(outline, (TILE, TILE))


def tile_heatmap(attention, mask):
    heat = cv2.applyColorMap((255 * attention / max(attention.max(), 1e-8)).astype(np.uint8), cv2.COLORMAP_INFERNO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    heat = cv2.drawContours(heat, cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_NONE)[0], -1, (0, 255, 255), 1)
    return cv2.resize(heat, (TILE, TILE))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", action="append", required=True, help="LABEL=PATH, repeatable")
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split_manifest", required=True)
    parser.add_argument("--object_class", default="hazelnut")
    parser.add_argument("--defect_type", default="hole")
    parser.add_argument("--mask_names", required=True, help="Comma-separated held-out mask file names")
    parser.add_argument("--good_indices", default="0,1", help="Indices into sorted test/good images")
    parser.add_argument("--timesteps", default="200,500,800")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    device = torch.device("cuda")
    manifest = load_split_manifest(args.split_manifest, expected_data_dir=args.data_dir)
    root = Path(manifest["data_dir"])
    heldout = {Path(e["mask"]).name: root / e["mask"] for e in manifest["heldout"]}
    good = sorted((root / args.object_class / "test" / "good").glob("*.png"))
    masks = [name.strip() for name in args.mask_names.split(",") if name.strip()]
    missing = [name for name in masks if name not in heldout]
    if missing:
        raise SystemExit(f"Not held-out masks: {missing}")
    cases = [(gi, name) for name in masks for gi in (int(i) for i in args.good_indices.split(","))]
    timesteps = [int(t) for t in args.timesteps.split(",")]
    checkpoints = [spec.split("=", 1) for spec in args.checkpoint]
    prompt = f"A {args.object_class} with sks"

    with contextlib.redirect_stdout(io.StringIO()):
        model = DefectFillModel(device=device, base_model_path=args.base_model_path)
    model.pipeline.unet.eval()
    model.pipeline.text_encoder.eval()

    inputs = [prepare_full_view_inputs(good[gi], heldout[name], device=device, dtype=torch.float16) for gi, name in cases]
    rows = [[tile_image(((image[0].float().permute(1, 2, 0).cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8),
                        mask[0, 0].cpu().numpy() > 0.5)] for image, mask in inputs]
    summary = {"prompt": prompt, "timesteps": timesteps, "cases": [f"good{gi}_{name}" for gi, name in cases],
               "regions": {"thick": f"{THICK_KERNEL}x{THICK_KERNEL} opening of the mask", "thin": "mask minus thick",
                           "background": f"farther than {FAR_RADIUS} px from the mask"},
               "checkpoints": {}}
    for label, path in checkpoints:
        with contextlib.redirect_stdout(io.StringIO()):
            load_checkpoint(model, None, path)
        per_case = []
        for row, (image, mask) in zip(rows, inputs):
            fine_mask = mask[0, 0].cpu().numpy() > 0.5
            attention = probe_case(model, image, mask, prompt, timesteps, args.seed)
            per_case.append(region_stats(attention, fine_mask))
            row.append(tile_heatmap(attention, fine_mask))
        mean = {key: float(np.mean([c[key] for c in per_case if c.get(key) is not None]))
                for key in ("thick", "thin", "background", "thin_over_thick", "mask_over_background")}
        summary["checkpoints"][label] = {"path": path, "mean": mean, "per_case": per_case}
        print(f"{label:14s} thick={mean['thick']:.4f} thin={mean['thin']:.4f} background={mean['background']:.4f} "
              f"thin/thick={mean['thin_over_thick']:.2f} mask/background={mean['mask_over_background']:.1f}")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    grid = np.concatenate([np.concatenate(row, axis=1) for row in rows], axis=0)
    cv2.imwrite(str(output / "attention_grid.jpg"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("columns: image | " + " | ".join(label for label, _ in checkpoints), "->", output)


if __name__ == "__main__":
    main()
