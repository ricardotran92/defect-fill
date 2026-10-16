import os
import json
import torch
import argparse
import torch.nn.functional as F
import random
import signal
import time
import numpy as np
from torch.optim import AdamW
from tqdm import tqdm
from model import LATENT_MASK_MODES, MASKED_LOSS_NORMS, DefectFillModel, mask_to_latent
from data_loader import generate_random_box_mask, get_data_loaders
from protocol import (
    PROTOCOL_VERSION,
    build_split_manifest,
    configuration_fingerprint,
    load_split_manifest,
    save_split_manifest,
    split_manifest_fingerprint,
)
from utils import (
    find_latest_valid_checkpoint,
    load_checkpoint,
    prune_intermediate_checkpoints,
    save_checkpoint,
)
# TensorBoard support
from torch.utils.tensorboard import SummaryWriter
import datetime


# Masked noise losses follow paper Eq. 5/7: weight inside the squared norm, mean over all elements.
# Recorded in the resume fingerprint so checkpoints trained with the old normalization are not resumed.
MASKED_LOSS_VERSION = "paper_eq5_eq7_mean"
# The UNet runs in fp16 without autocast, so backward produces fp16 gradients. The paper-normalized
# losses are ~100x smaller than before, which made those gradients underflow; dynamic loss scaling
# keeps them representable. Part of the resume fingerprint like MASKED_LOSS_VERSION.
LOSS_SCALING_VERSION = "dynamic_grad_scaler_v1"
# B = (1 - M) * I uses the full-resolution mask (paper Eq. 4); v3 blanked whole latent cells ("cell").
BACKGROUND_MASK = "pixel"


def generate_seed_from_timestamp():
    """Generates a random seed based on the current timestamp"""
    return int(time.time() * 1000) % (2**31)

def train(args):
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.defect_type:
        raise ValueError("Paper protocol training requires one explicit --defect_type")

    # Create output directory structure
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoints_dir = os.path.join(args.output_dir, "checkpoints")
    tensorboard_dir = os.path.join(args.output_dir, "tensorboard")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    # Freeze the one-third train / two-thirds held-out split before allocating the model.
    manifest_path = os.path.join(args.output_dir, "split_manifest.json")
    if os.path.exists(manifest_path):
        manifest = load_split_manifest(manifest_path, expected_data_dir=args.data_dir)
        expected_manifest_fields = {
            "object_class": args.object_class,
            "defect_type": args.defect_type,
            "seed": args.split_seed,
        }
        for key, expected_value in expected_manifest_fields.items():
            if manifest[key] != expected_value:
                raise ValueError(
                    f"Existing split manifest {key} mismatch: "
                    f"expected {expected_value!r}, got {manifest[key]!r}"
                )
    else:
        manifest = build_split_manifest(
            args.data_dir,
            args.object_class,
            args.defect_type,
            seed=args.split_seed,
        )
        save_split_manifest(manifest, manifest_path)

    manifest_fingerprint = split_manifest_fingerprint(manifest)
    resume_critical_config = {
        "protocol_version": PROTOCOL_VERSION,
        "split_manifest_fingerprint": manifest_fingerprint,
        "base_model_path": os.path.normcase(os.path.abspath(args.base_model_path)),
        "placeholder_token": "sks",
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "text_encoder_lr": args.text_encoder_lr,
        "unet_lr": args.unet_lr,
        "lambda_defect": args.lambda_defect,
        "lambda_obj": args.lambda_obj,
        "lambda_attn": args.lambda_attn,
        "alpha": args.alpha,
        "masked_loss": MASKED_LOSS_VERSION,
        "masked_loss_norm": args.masked_loss_norm,
        "loss_scaling": LOSS_SCALING_VERSION,
        "latent_mask_mode": args.latent_mask_mode,
        "background_mask": BACKGROUND_MASK,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "lr_warmup_steps": args.lr_warmup_steps,
        "seed": args.seed,
    }
    checkpoint_metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "split_manifest_fingerprint": manifest_fingerprint,
        "configuration_fingerprint": configuration_fingerprint(resume_critical_config),
        "object_class": args.object_class,
        "defect_type": args.defect_type,
        "view_mode": "full",
    }

    # Create log file in the output directory
    log_file_path = os.path.join(args.output_dir, "train_log.txt")
    log_file = open(log_file_path, "a")
    log_file.write(f"\n\n{'='*60}\n")
    log_file.write(f"Training started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"Object class: {args.object_class}\n")
    log_file.write(f"Defect type: {args.defect_type if args.defect_type else 'all'}\n")
    log_file.write(f"Base model path: {args.base_model_path}\n")
    log_file.write(f"Config name: {args.config_name}\n")
    log_file.write(f"Lambda defect: {args.lambda_defect}\n")
    log_file.write(f"Lambda obj: {args.lambda_obj}\n")
    log_file.write(f"Lambda attn: {args.lambda_attn}\n")
    log_file.write(f"Alpha (obj branch bg weight): {args.alpha}\n")
    log_file.write(f"Gradient accumulation steps: {args.gradient_accumulation_steps}\n")
    log_file.write(f"Random seed: {args.seed}\n")
    log_file.write(f"{'='*60}\n\n")
    
    # Save training configuration to JSON
    train_config = {
        "timestamp": datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
        "object_class": args.object_class,
        "defect_type": args.defect_type if args.defect_type else "all",
        "base_model_path": args.base_model_path,
        "config_name": args.config_name,
        "lambda_defect": args.lambda_defect,
        "lambda_obj": args.lambda_obj,
        "lambda_attn": args.lambda_attn,
        "alpha": args.alpha,
        "masked_loss": MASKED_LOSS_VERSION,
        "masked_loss_norm": args.masked_loss_norm,
        "loss_scaling": LOSS_SCALING_VERSION,
        "latent_mask_mode": args.latent_mask_mode,
        "background_mask": BACKGROUND_MASK,
        "batch_size": args.batch_size,
        "max_train_steps": args.max_train_steps,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "text_encoder_lr": args.text_encoder_lr,
        "unet_lr": args.unet_lr,
        "lr_warmup_steps": args.lr_warmup_steps,
        "save_steps": args.save_steps,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "protocol_version": PROTOCOL_VERSION,
        "split_manifest": os.path.abspath(manifest_path),
        "split_manifest_fingerprint": manifest_fingerprint,
        "configuration_fingerprint": checkpoint_metadata["configuration_fingerprint"],
        "view_mode": "full",
        "train_images": len(manifest["train"]),
        "heldout_images": len(manifest["heldout"]),
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "lora_dropout": args.lora_dropout,
    }
    config_path = os.path.join(args.output_dir, "train_config.json")
    with open(config_path, "w") as f:
        json.dump(train_config, f, indent=4)
    print(f"Training config saved to: {config_path}")
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(tensorboard_dir)
    
    # Load data
    train_loader, _ = get_data_loaders(
        root_dir=args.data_dir,
        object_class=args.object_class,
        batch_size=args.batch_size,
        defect_type=args.defect_type,
        dilate_mask=args.dilate_mask,
        mask_kernel_size=args.mask_kernel_size,
        train_pairs=manifest["train"],
        num_workers=args.num_workers,
    )
    
    # Initialize model
    model = DefectFillModel(
        device=device,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        seed=args.seed,
        base_model_path=args.base_model_path,
    )
    
    # Set up optimizers with specific learning rates for Text Encoder and UNet
    text_encoder_params = [p for n, p in model.pipeline.text_encoder.named_parameters() if "lora" in n]
    unet_params = [p for n, p in model.pipeline.unet.named_parameters() if "lora" in n]
    
    optimizer = AdamW([
        {"params": text_encoder_params, "lr": args.text_encoder_lr},
        {"params": unet_params, "lr": args.unet_lr}
    ])
    
    grad_scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    # Save original LRs for warmup calculation
    base_lrs = [args.text_encoder_lr, args.unet_lr]
    
    # Use the scheduler loaded with the selected base model.
    noise_scheduler = model.scheduler
    
    # Auto-resume from the newest complete, protocol-compatible optimizer boundary.
    start_step = 0
    resume_path = None
    if args.resume_from == "auto":
        resume_path = find_latest_valid_checkpoint(
            checkpoints_dir,
            expected_metadata=checkpoint_metadata,
        )
    elif args.resume_from:
        resume_path = args.resume_from
    if resume_path:
        start_step = load_checkpoint(
            model,
            optimizer,
            resume_path,
            expected_metadata=checkpoint_metadata,
            restore_rng=True,
            grad_scaler=grad_scaler,
        )
        print(f"Resuming from step {start_step}")
        log_file.write(f"Resuming from {resume_path} at step {start_step}\n")
    
    # Set models to training mode
    model.pipeline.unet.train()
    model.pipeline.text_encoder.train()
    
    total_steps = args.max_train_steps
    progress_bar = tqdm(range(start_step, total_steps), desc="Training Progress")
    
    global_step = start_step
    accumulation_step = 0
    optimizer.zero_grad(set_to_none=True)

    stop_requested = False

    def request_stop(signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        print(f"Stop signal {signum} received; finishing the current operation safely...")

    previous_signal_handlers = {}
    for stop_signal in (signal.SIGINT, signal.SIGTERM):
        previous_signal_handlers[stop_signal] = signal.getsignal(stop_signal)
        signal.signal(stop_signal, request_stop)

    while global_step < total_steps and not stop_requested:
        for batch in train_loader:
            if global_step >= total_steps or stop_requested:
                break
                
            # Move data to device
            images = batch["image"].to(device, dtype=torch.float16)
            masks = batch["mask"].to(device, dtype=torch.float16)
            backgrounds = batch["background"].to(device, dtype=torch.float16)
            is_defect = batch["is_defect"]
            
            # Ensure we only process defective samples
            defect_samples = torch.nonzero(is_defect).squeeze(1)
            if len(defect_samples) == 0:
                continue  # Skip batch if no defects present
                
            # Extract defect-only samples
            defect_images = images[defect_samples]
            defect_masks = masks[defect_samples]
            defect_backgrounds = backgrounds[defect_samples]
            object_classes = [batch["object_class"][int(i)] for i in defect_samples]
            
            # Learning rate warmup
            if global_step < args.lr_warmup_steps:
                lr_scale = min(1.0, (global_step + 1) / args.lr_warmup_steps)
                for i, param_group in enumerate(optimizer.param_groups):
                    param_group["lr"] = base_lrs[i] * lr_scale
            
            # Reset attention maps
            if hasattr(model, 'attention_maps'):
                model.attention_maps = {}
            
            # ========== PHASE 1: Defect Branch (Defect Texture Learning) ==========
            # The fixed "sks" identifier is optimized through the text-encoder LoRA.
            defect_prompts = [f"A photo of {model.placeholder_token}" for _ in range(len(defect_samples))]
            text_embeddings = model.get_text_embeddings(defect_prompts, enable_grad=True)
            
            # Latent space processing
            with torch.no_grad():
                latents = model.pipeline.vae.encode(defect_images).latent_dist.sample()
                latents = latents * model.pipeline.vae.config.scaling_factor  # z_0
            
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)  # z_t
            
            if len(defect_masks.shape) == 3:
                defect_masks = defect_masks.unsqueeze(1)
            
            # Mask at latent resolution; b = E(I * (1-M)) uses the full-resolution mask (Eq. 4)
            latent_size = latents.shape[-2:]
            mask_latents = mask_to_latent(defect_masks, latent_size, args.latent_mask_mode)
            with torch.no_grad():
                masked_images = defect_images * (1 - defect_masks)
                masked_image_latents = model.pipeline.vae.encode(masked_images).latent_dist.sample()
                masked_image_latents = masked_image_latents * model.pipeline.vae.config.scaling_factor
            
            # Forward pass (9-channel input)
            outputs = model(
                noisy_latents=noisy_latents,
                masked_image_latents=masked_image_latents,
                mask_latents=mask_latents,
                timesteps=timesteps,
                encoder_hidden_states=text_embeddings
            )
            
            # Compute losses
            noise_pred = outputs["noise_pred"]
            defect_loss = model.compute_defect_loss(noise_pred, noise, mask_latents, normalize=args.masked_loss_norm)
            
            # ========== PHASE 2: Object Branch (Object Integrity Learning) ==========
            # Random box masks M_rand (Eq. 6) for structural context learning
            height, width = defect_images.shape[-2:]
            random_masks = torch.stack(
                [generate_random_box_mask(height, width) for _ in range(defect_images.shape[0])]
            ).unsqueeze(1).to(device, dtype=defect_images.dtype)
            
            # Prompts linking object class with the defect token
            obj_prompts = [f"A {obj_class} with {model.placeholder_token}" for obj_class in object_classes]
            obj_text_embeddings = model.get_text_embeddings(obj_prompts, enable_grad=True)
            
            random_mask_latents = mask_to_latent(random_masks, latent_size, args.latent_mask_mode)
            object_weight_latents = mask_latents  # M' and the attention target use the true defect mask M
            with torch.no_grad():
                random_masked_images = defect_images * (1 - random_masks)
                random_masked_image_latents = model.pipeline.vae.encode(random_masked_images).latent_dist.sample()
                random_masked_image_latents = random_masked_image_latents * model.pipeline.vae.config.scaling_factor

            obj_outputs = model(
                noisy_latents=noisy_latents,
                masked_image_latents=random_masked_image_latents,
                mask_latents=random_mask_latents,
                timesteps=timesteps,
                encoder_hidden_states=obj_text_embeddings,
                attention_mask_latents=object_weight_latents
            )

            object_loss = model.compute_object_loss(obj_outputs["noise_pred"], noise, object_weight_latents, alpha=args.alpha,
                                                    normalize=args.masked_loss_norm)
            attention_loss = obj_outputs.get("attention_loss", torch.tensor(0.0, device=device))
            
            # Total Loss Calculation
            total_loss = args.lambda_defect * defect_loss + args.lambda_obj * object_loss + args.lambda_attn * attention_loss
            total_loss = total_loss / args.gradient_accumulation_steps
            
            # NaN Check
            if not torch.isfinite(total_loss):
                print(f"Warning: NaN loss detected at step {global_step}")
                log_file.write(f"Warning: NaN loss at step {global_step}\n")
                optimizer.zero_grad(set_to_none=True)
                accumulation_step = 0
                continue
            
            grad_scaler.scale(total_loss).backward()
            accumulation_step += 1
            
            # Optimization step
            if accumulation_step >= args.gradient_accumulation_steps:
                grad_scaler.step(optimizer)  # unscales, skips the step if any gradient is inf/NaN
                grad_scaler.update()
                optimizer.zero_grad(set_to_none=True)
                accumulation_step = 0
                
                progress_bar.update(1)
                global_step += 1
                
                # Logging and TensorBoard updates
                writer.add_scalar("Loss/Defect", defect_loss.item(), global_step)
                writer.add_scalar("Loss/Object", object_loss.item(), global_step)
                writer.add_scalar("Loss/Attention", attention_loss.item(), global_step)
                writer.add_scalar("Loss/Total", total_loss.item() * args.gradient_accumulation_steps, global_step)
                writer.add_scalar("LossScale", grad_scaler.get_scale(), global_step)
                
                if global_step % 10 == 0:
                    for i, param_group in enumerate(optimizer.param_groups):
                        writer.add_scalar(f"LearningRate/group{i}", param_group["lr"], global_step)
                
                # Periodic checkpointing
                if global_step % args.save_steps == 0 or global_step == total_steps:
                    checkpoint_path = os.path.join(checkpoints_dir, f"checkpoint_{global_step}.pt")
                    save_checkpoint(
                        model,
                        optimizer,
                        global_step,
                        checkpoint_path,
                        metadata=checkpoint_metadata,
                        grad_scaler=grad_scaler,
                    )
                    log_file.write(f"Checkpoint saved at step {global_step}\n")

    for stop_signal, previous_handler in previous_signal_handlers.items():
        signal.signal(stop_signal, previous_handler)

    if stop_requested:
        optimizer.zero_grad(set_to_none=True)
        checkpoint_path = os.path.join(checkpoints_dir, f"checkpoint_{global_step}.pt")
        save_checkpoint(
            model,
            optimizer,
            global_step,
            checkpoint_path,
            metadata=checkpoint_metadata,
            grad_scaler=grad_scaler,
        )
        log_file.write(f"Interrupted; checkpoint saved at step {global_step}\n")
        log_file.flush()
        writer.close()
        log_file.close()
        print(f"Training interrupted safely at step {global_step}: {checkpoint_path}")
        return model

    # Save final model
    final_checkpoint_path = os.path.join(checkpoints_dir, "checkpoint_final.pt")
    save_checkpoint(
        model,
        optimizer,
        global_step,
        final_checkpoint_path,
        metadata=checkpoint_metadata,
        grad_scaler=grad_scaler,
    )
    print(f"Final model saved to: {final_checkpoint_path}")

    if not args.keep_intermediate_checkpoints:
        removed = prune_intermediate_checkpoints(checkpoints_dir)
        print(f"Removed {len(removed)} intermediate checkpoints; kept {final_checkpoint_path}")
        log_file.write(f"Removed {len(removed)} intermediate checkpoints after completion\n")

    writer.close()
    log_file.close()
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DefectFill model")
    
    # Paths
    parser.add_argument("--data_dir", type=str, required=True, help="Path to MVTec AD dataset")
    parser.add_argument("--object_class", type=str, required=True, help="Object class to train on")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save models")
    parser.add_argument(
        "--base_model_path",
        type=str,
        default=r"D:\Novels_Code\stablediffusion\sd2_inpainting",
        help="Local Diffusers SD2 inpainting directory, or a HuggingFace model id",
    )
    
    # Loss Weights
    parser.add_argument("--lambda_defect", type=float, default=0.5, help="Defect loss weight (L_def)")
    parser.add_argument("--lambda_obj", type=float, default=0.2, help="Object integrity loss weight (L_obj)")
    parser.add_argument("--lambda_attn", type=float, default=0.05, help="Attention loss weight (L_attn)")
    parser.add_argument("--alpha", type=float, default=0.3, help="Background weight for object branch")
    parser.add_argument(
        "--masked_loss_norm",
        choices=MASKED_LOSS_NORMS,
        default="mask",
        help="Reduction of L_def/L_obj: mask = weighted mean over the mask (v5); total = mean over all elements (v2-v4)",
    )
    
    # Training Config
    parser.add_argument("--config_name", type=str, default="base", help="Experiment name (base/tex/obj)")
    parser.add_argument("--defect_type", type=str, required=True, help="Specific defect type to train on")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--text_encoder_lr", type=float, default=4e-5)
    parser.add_argument("--unet_lr", type=float, default=2e-4)
    parser.add_argument("--max_train_steps", type=int, default=2000)
    parser.add_argument("--lr_warmup_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42, help="Training seed")
    parser.add_argument("--split_seed", type=int, default=42, help="Fixed one-third split seed")
    parser.add_argument("--resume_from", type=str, default="auto")
    parser.add_argument(
        "--latent_mask_mode",
        choices=LATENT_MASK_MODES,
        default="maxpool",
        help="Mask downsampling to latent resolution: maxpool keeps thin mask branches; nearest = original code",
    )
    parser.add_argument(
        "--keep_intermediate_checkpoints",
        action="store_true",
        help="Keep checkpoint_<step>.pt files after training completes (default: delete them, keep checkpoint_final.pt)",
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--dilate_mask", type=str, default="False", help="Whether to dilate masks (True/False)")
    parser.add_argument("--mask_kernel_size", type=int, default=3, help="Size of dilation kernel (must be odd, e.g. 3, 5, 7)")
    
    args = parser.parse_args()

    # Helper to convert string "True" to boolean
    args.dilate_mask = args.dilate_mask.lower() == "true"
    
    # Seed setup
    if args.seed == -1:
        args.seed = generate_seed_from_timestamp()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    train(args)
