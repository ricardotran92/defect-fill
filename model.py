import os
# Hugging Face Mirror Configuration
# Option 1: hf-mirror.com (Available in some regions)
# Option 2: Use ModelScope as an alternative
USE_MODELSCOPE = False  # Set to True for ModelScope, False for HuggingFace

import torch
import torch.nn as nn
from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel
from peft import LoraConfig, get_peft_model
import lpips
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math
from diffusers.models.attention_processor import Attention, AttnProcessor


class AttentionStoreProcessor(AttnProcessor):
    """Attention Processor used to store cross-attention maps for steering"""
    def __init__(self, model=None):
        super().__init__()
        self.model = model  # Reference to the main model instance

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        
        query = attn.to_q(hidden_states)
        
        is_cross_attention = encoder_hidden_states is not None
        
        if not is_cross_attention:
            # For Self-Attention, use standard processing
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
        else:
            # For Cross-Attention, we need to capture and store the maps
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
        
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        
        # Calculate attention scores
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores * attn.scale
        
        # Apply softmax to get attention probabilities
        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
        
        # Store cross-attention probabilities in the model instance
        if is_cross_attention and self.model is not None:
            # Get the name of the current block
            for name, module in self.model.pipeline.unet.named_modules():
                if module == attn and "attn2" in name and "up_blocks" in name:
                    try:
                        num_heads = attn.heads
                        total_elements = attention_probs.numel()
                        query_len = hidden_states.shape[1]
                        
                        # Key length is usually 77 (CLIP text encoder output length)
                        key_len = encoder_hidden_states.shape[1] if encoder_hidden_states is not None else query_len
                        
                        # Safe Reshape - verify shape compatibility
                        expected_size = batch_size * num_heads * query_len * key_len
                        if total_elements == expected_size:
                            reshaped_probs = attention_probs.reshape(
                                batch_size, 
                                num_heads, 
                                query_len,
                                key_len
                            )
                            if not hasattr(self.model, "attention_maps"):
                                self.model.attention_maps = {}
                            self.model.attention_maps[name] = reshaped_probs
                        else:
                            print(f"Warning: Attention shape mismatch - Elements: {total_elements}, Expected: {expected_size}")
                    except Exception as e:
                        print(f"Attention Map Processing Error: {e}, Shape: {attention_probs.shape}")
                    break
        
        hidden_states = torch.matmul(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        
        return hidden_states

LATENT_MASK_MODES = ("maxpool", "nearest")
MASKED_LOSS_NORMS = ("mask", "total")


def mask_to_latent(mask: torch.Tensor, latent_size, mode: str = "maxpool") -> torch.Tensor:
    """Pixel mask (B, 1, H, W) -> latent mask (B, 1, h, w).

    The latent mask is the UNet mask channel, weights L_def/L_obj, is the attention target, and
    selects where background latents are preserved during sampling.
    "nearest" (original code) samples one pixel per 8x8 cell and drops thin mask branches: on
    hazelnut/hole held-out masks it keeps only ~39% of thin-branch pixels.
    "maxpool" marks a cell whenever any of its pixels is defective.
    The background image B = (1 - M) * I always uses the full-resolution mask M (paper Eq. 4);
    blanking whole 8x8 cells there made v3 paint block-shaped holes.
    """
    if mode == "nearest":
        return F.interpolate(mask, size=tuple(latent_size))
    if mode == "maxpool":
        return F.adaptive_max_pool2d(mask, tuple(latent_size))
    raise ValueError(f"Unknown latent mask mode {mode!r}; choose from {LATENT_MASK_MODES}")


def composite_background(generated: torch.Tensor, original: torch.Tensor, mask: torch.Tensor,
                         grow: int = 3, feather: int = 2) -> torch.Tensor:
    """Keep the original pixels outside the defect (all tensors in [0, 1]).

    Decoding the whole latent replaces every background pixel with its VAE reconstruction, which
    shifts colours across the image. The generated pixels are kept on the defect mask grown by
    `grow` pixels (so the defect edge is not cut), fading out over `feather` more pixels. The seam
    follows the defect shape instead of the 8x8 latent grid.
    """
    weight = mask.float()
    if grow > 0:
        weight = F.max_pool2d(weight, 2 * grow + 1, stride=1, padding=grow)
    if feather > 0:
        size = 2 * feather + 1
        blurred = F.avg_pool2d(F.pad(weight, (feather,) * 4, mode="replicate"), size, stride=1)
        weight = torch.maximum(weight, blurred)
    weight = weight.to(generated.dtype)
    return weight * generated + (1.0 - weight) * original.to(generated.dtype)


class DefectFillModel(nn.Module):
    def __init__(
        self,
        device="cuda",
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.1,
        seed=42,
        placeholder_token="sks",
        base_model_path=None,
    ):
        super().__init__()
        torch.manual_seed(seed)
        self.device = device

        # Base Model ID
        hf_model_id = "sd2-community/stable-diffusion-2-inpainting"
        model_source = base_model_path or hf_model_id

        # Prefer a user-provided local Diffusers directory when available.
        if base_model_path is not None:
            model_source = os.path.normpath(base_model_path)
            if not os.path.exists(model_source):
                raise FileNotFoundError(f"Base model path does not exist: {model_source}")

            print(f"[Local Model] Loading base model from: {model_source}")
            self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                model_source,
                torch_dtype=torch.float16,
                local_files_only=True,
            ).to(device)
            self.scheduler = DDIMScheduler.from_pretrained(
                model_source,
                subfolder="scheduler",
                local_files_only=True,
            )
        elif USE_MODELSCOPE:
            try:
                from modelscope import snapshot_download
                print(f"[ModelScope] Downloading model: {hf_model_id}")
                local_model_path = snapshot_download(hf_model_id)
                print(f"[ModelScope] Model downloaded to: {local_model_path}")
                
                self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                    local_model_path,
                    torch_dtype=torch.float16
                ).to(device)
                
                self.scheduler = DDIMScheduler.from_pretrained(
                    local_model_path,
                    subfolder="scheduler"
                )
            except ImportError:
                print("[Warning] modelscope not installed. Try: pip install modelscope")
                print("[Info] Attempting HuggingFace fallback...")
                self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                    hf_model_id, torch_dtype=torch.float16
                ).to(device)
                self.scheduler = DDIMScheduler.from_pretrained(hf_model_id, subfolder="scheduler")
        else:
            self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                model_source, torch_dtype=torch.float16
            ).to(device)
            self.scheduler = DDIMScheduler.from_pretrained(model_source, subfolder="scheduler")

        self.pipeline.set_progress_bar_config(disable=True)
        self.pipeline.scheduler = self.scheduler

        # The paper uses the tokenizer's existing "sks" identifier, not textual inversion.
        self.placeholder_token = placeholder_token
        placeholder_token_ids = self.pipeline.tokenizer.encode(
            self.placeholder_token,
            add_special_tokens=False,
        )
        if len(placeholder_token_ids) != 1:
            raise ValueError(
                f"Identifier '{self.placeholder_token}' must map to exactly one CLIP token; "
                f"got {placeholder_token_ids}"
            )
        self.placeholder_token_id = placeholder_token_ids[0]
        print(f"[Prompt Identifier] {self.placeholder_token!r} token id = {self.placeholder_token_id}")
        
        # LoRA Configuration
        unet_lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            init_lora_weights="gaussian"
        )
        
        text_encoder_lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            init_lora_weights="gaussian"
        )
        
        # Apply LoRA adapters
        self.pipeline.unet = get_peft_model(self.pipeline.unet, unet_lora_config)
        self.pipeline.text_encoder = get_peft_model(self.pipeline.text_encoder, text_encoder_lora_config)
        
        # Freeze VAE parameters
        for param in self.pipeline.vae.parameters():
            param.requires_grad = False
        
        # LPIPS is only needed during sample selection in inference, so load it lazily.
        self.lpips_model = None
        
        self.attention_maps = {}
        self.register_attention_processor()
        self.defect_token_indices = []

    def register_attention_processor(self):
        """Replace standard UNet attention processors with custom ones"""
        self.attention_maps = {}
        for name, module in self.pipeline.unet.named_modules():
            if isinstance(module, Attention) and "attn2" in name:  # Target Cross-Attention only
                module.processor = AttentionStoreProcessor(model=self)

    def get_lpips_model(self):
        """Load LPIPS only when inference needs perceptual sample selection."""
        if self.lpips_model is None:
            self.lpips_model = lpips.LPIPS(net='vgg', spatial=True).to(self.device)
        return self.lpips_model

    def get_attention_loss(self, mask_latents: torch.Tensor) -> torch.Tensor:
        """
        Calculates Attention Loss - forces <defect> token attention maps to align with the defect mask.
        """
        if not self.attention_maps:
            return torch.tensor(0.0, device=mask_latents.device)
        
        if len(mask_latents.shape) == 3:
            mask_latents = mask_latents.unsqueeze(1)
        
        batch_size = mask_latents.shape[0]
        attention_loss = torch.tensor(0.0, device=mask_latents.device)
        
        # Use only decoder (up_blocks) attention maps
        decoder_attention_maps = {
            name: attn_map for name, attn_map in self.attention_maps.items()
            if "up_blocks" in name
        }
        
        if not decoder_attention_maps:
            return torch.tensor(0.0, device=mask_latents.device)
        
        for b in range(batch_size):
            token_idx = self.defect_token_indices[b] if b < len(self.defect_token_indices) else -1
            if token_idx < 0:
                continue
            
            mask = mask_latents[b].squeeze(0)  # (H, W)
            resized_attention_maps = []
            
            for name, attn_map in decoder_attention_maps.items():
                try:
                    if b < attn_map.shape[0]:
                        # Average attention across all heads for the specific token
                        defect_attn = attn_map[b, :, :, token_idx].mean(dim=0)
                        
                        seq_len = defect_attn.shape[0]
                        h = int(math.sqrt(seq_len))
                        if h * h == seq_len:
                            defect_attn = defect_attn.reshape(h, h)
                            resized_attn = F.interpolate(
                                defect_attn.unsqueeze(0).unsqueeze(0),
                                size=mask.shape,
                                mode='bilinear',
                                align_corners=False
                            ).squeeze()
                            resized_attention_maps.append(resized_attn)
                except Exception:
                    continue
            
            if resized_attention_maps:
                avg_attn_map = torch.stack(resized_attention_maps).mean(dim=0)
                # L2 Loss: ||AttentionMap - Mask||^2
                sample_loss = F.mse_loss(avg_attn_map, mask)
                attention_loss += sample_loss
        
        return attention_loss / batch_size if batch_size > 0 else attention_loss

    def get_text_embeddings(self, prompts, enable_grad=True):
        """Encodes prompts and locates the precise index of the <defect> token"""
        if not hasattr(self, 'pipeline') or self.pipeline is None:
            raise ValueError("Pipeline not initialized")
            
        if isinstance(prompts, str):
            prompts = [prompts]
        
        text_inputs = self.pipeline.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.pipeline.device)
        
        input_ids = text_inputs.input_ids
        
        # Locate the <defect> token position in each prompt
        self.defect_token_indices = []
        for ids in input_ids:
            positions = (ids == self.placeholder_token_id).nonzero(as_tuple=True)[0]
            self.defect_token_indices.append(positions[0].item() if len(positions) > 0 else -1)
        
        if enable_grad:
            text_embeddings = self.pipeline.text_encoder(input_ids)[0]
        else:
            with torch.no_grad():
                text_embeddings = self.pipeline.text_encoder(input_ids)[0]
            
        return text_embeddings
    
    def forward(
        self,
        noisy_latents: torch.Tensor,
        masked_image_latents: torch.Tensor,
        mask_latents: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask_latents: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Training Forward Pass - Implements 9-channel input.
        Input format: [noisy_latents(4), masked_background(4), mask(1)]
        """
        self.attention_maps = {}
        concat_latents = torch.cat([noisy_latents, masked_image_latents, mask_latents], dim=1)
        
        noise_pred = self.pipeline.unet(
            concat_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
        ).sample
        
        attention_target = attention_mask_latents if attention_mask_latents is not None else mask_latents
        attention_loss = self.get_attention_loss(attention_target)
        
        return {
            "noise_pred": noise_pred,
            "attention_loss": attention_loss
        }

    @staticmethod
    def compute_masked_mse(noise_pred: torch.Tensor, noise: torch.Tensor, mask: torch.Tensor,
                           normalize: str = "mask") -> torch.Tensor:
        """Paper Eq. 5/7: ||W ⊙ (ε − ε_θ)||², computed in FP32.

        The weight sits inside the squared norm, so a soft weight w contributes w² per element.
        The paper does not state the reduction:
        - "mask": divide by the weight energy sum(W²) over all channels, i.e. a weighted mean over
          the masked elements. A defect covering ~1% of the image still gives a full-strength
          L_def signal.
        - "total": divide by the number of elements (v2-v4). L_def then shrinks with the defect
          area (~1% for hazelnut/hole), the object branch dominates and the model reconstructs
          instead of drawing defects, as in paper Fig. 7 without L_def.
        """
        residual_sq = (mask.float() * (noise.float() - noise_pred.float())).pow(2)
        if normalize == "total":
            return residual_sq.mean()
        if normalize == "mask":
            energy = mask.float().pow(2).expand_as(residual_sq).sum()
            return residual_sq.sum() / energy.clamp(min=1e-6)
        raise ValueError(f"Unknown masked-loss normalization {normalize!r}; choose from {MASKED_LOSS_NORMS}")

    def compute_defect_loss(self, noise_pred: torch.Tensor, noise: torch.Tensor, mask_latents: torch.Tensor,
                            normalize: str = "mask") -> torch.Tensor:
        """L_def (Eq. 5): squared error restricted to the defect mask M"""
        return self.compute_masked_mse(noise_pred, noise, mask_latents, normalize)

    def compute_object_loss(self, noise_pred: torch.Tensor, noise: torch.Tensor, mask_latents: torch.Tensor,
                            alpha: float = 0.3, normalize: str = "mask") -> torch.Tensor:
        """L_obj (Eq. 7): ||M' ⊙ (ε − ε_θ)||² with M' = M + alpha*(1-M), so background weighs alpha²"""
        weighted_mask = mask_latents + alpha * (1 - mask_latents)
        return self.compute_masked_mse(noise_pred, noise, weighted_mask, normalize)
    
    def generate(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        prompt: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        generator: Optional[torch.Generator] = None,
        latent_mask_mode: str = "maxpool",
        composite: bool = True,
    ) -> torch.Tensor:
        """Generate with the standard Diffusers 9-channel inpainting path.

        latent_mask_mode: how the mask reaches latent resolution (see mask_to_latent).
        composite: keep the original pixels outside the (slightly grown) defect mask.
        """
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)

        image_for_pipeline = image.float()
        if image_for_pipeline.min() < 0:
            image_for_pipeline = (image_for_pipeline + 1.0) / 2.0
        image_for_pipeline = image_for_pipeline.clamp(0.0, 1.0)

        mask_for_pipeline = mask.float()
        if mask_for_pipeline.max() > 1:
            mask_for_pipeline = mask_for_pipeline / 255.0
        mask_for_pipeline = mask_for_pipeline.clamp(0.0, 1.0)

        vae_dtype = next(self.pipeline.vae.parameters()).dtype
        clean_for_vae = image_for_pipeline.to(dtype=vae_dtype) * 2.0 - 1.0
        with torch.no_grad():
            image_latents = self.pipeline.vae.encode(clean_for_vae).latent_dist.mode()
            image_latents = image_latents * self.pipeline.vae.config.scaling_factor

        latent_height, latent_width = image_latents.shape[-2:]
        initial_noise = torch.randn(
            image_latents.shape,
            generator=generator,
            device=image_latents.device,
            dtype=image_latents.dtype,
        )
        pixel_mask = mask_for_pipeline.to(device=image_latents.device, dtype=image_latents.dtype)
        mask_latents = mask_to_latent(pixel_mask, (latent_height, latent_width), latent_mask_mode)
        # The pipeline resizes mask_image with nearest; a cell-aligned mask reproduces mask_latents exactly.
        cell_mask = F.interpolate(mask_latents, size=pixel_mask.shape[-2:])
        # B = (1 - M) * I with the full-resolution mask (paper Eq. 4), in the pipeline's [-1, 1] range.
        # A 3-channel tensor here is VAE-encoded by the pipeline without re-masking it by mask_image.
        background_image = (image_for_pipeline.to(pixel_mask.device) * 2.0 - 1.0) * (1.0 - pixel_mask.float())

        def preserve_background(pipe, step_index, _timestep, callback_kwargs):
            latents = callback_kwargs["latents"]
            if step_index + 1 < len(pipe.scheduler.timesteps):
                next_timestep = pipe.scheduler.timesteps[step_index + 1]
                timestep_batch = next_timestep.expand(latents.shape[0])
                background_latents = pipe.scheduler.add_noise(
                    image_latents,
                    initial_noise,
                    timestep_batch,
                )
            else:
                background_latents = image_latents
            callback_kwargs["latents"] = (
                mask_latents * latents + (1.0 - mask_latents) * background_latents
            )
            return callback_kwargs

        result = self.pipeline(
            prompt=[prompt] * image_for_pipeline.shape[0],
            image=image_for_pipeline,
            mask_image=cell_mask.float(),
            masked_image_latents=background_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            latents=initial_noise,
            callback_on_step_end=preserve_background,
            callback_on_step_end_tensor_inputs=["latents"],
            output_type="pt",
        )
        if not composite:
            return result.images
        original = image_for_pipeline.to(device=result.images.device)
        return composite_background(result.images, original, pixel_mask.to(result.images.device))
