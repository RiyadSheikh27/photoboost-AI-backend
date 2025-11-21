"""
Image Enhancer using Real-ESRGAN + Stable Diffusion (Diffusers Backend)
Compatible with Django startup.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
import numpy as np

from diffusers import StableDiffusionUpscalePipeline, DDIMScheduler
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan import GFPGANer  # Optional: Face enhancement post-process


@dataclass(kw_only=True)
class ESRGANUpscalerCheckpoints:
    esrgan_model: str = "RealESRGAN_x4plus"
    sd_model: str = "runwayml/stable-diffusion-v1-5"


class ESRGANUpscaler:
    def __init__(
        self,
        checkpoints: ESRGANUpscalerCheckpoints,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.checkpoints = checkpoints

        # Step 1: Load Real-ESRGAN for initial x4 tiling upscale
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        self.esrgan = RealESRGANer(
            scale=4,
            model_path=checkpoints.esrgan_model,  # Or local .pth path
            model=model,
            tile=400,  # Tile size for large images (memory efficient)
            tile_pad=10,
            pre_pad=0,
            half=(dtype == torch.float16),  # FP16 if supported
            gpu_id=0 if device.type == "cuda" else None,
        )

        # Optional: GFPGAN for face restoration after ESRGAN
        self.face_enhancer = GFPGANer(model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth', upscale=4)

        # Step 2: Load SD Upscale Pipeline for latent refinement
        self.sd_pipeline = StableDiffusionUpscalePipeline.from_pretrained(
            checkpoints.sd_model,
            torch_dtype=dtype,
        )
        self.sd_pipeline.scheduler = DDIMScheduler.from_config(self.sd_pipeline.scheduler.config)
        self.sd_pipeline = self.sd_pipeline.to(device)

    def to(self, device: torch.device, dtype: torch.dtype):
        self.esrgan = None  # Re-init if needed, but ESRGAN doesn't have .to()
        self.sd_pipeline = self.sd_pipeline.to(device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype

    def pre_upscale(
        self,
        image: Image.Image,
        upscale_factor: float,
        prompt: str = "high quality, detailed, sharp",
        strength: float = 0.3,  # Denoising strength for SD
        num_inference_steps: int = 20,
        **_: Any
    ) -> Image.Image:
        """
        Chained Upscale:
        1. ESRGAN x4 tiling (handles large images without OOM).
        2. SD latent upscale + refinement (for total factor >4).
        """
        # Convert to RGB
        img_array = np.array(image.convert("RGB"))

        # Step 1: ESRGAN Pre-Upscale (x4)
        output_esrgan, _ = self.esrgan.enhance(img_array, outscale=4)
        esrgan_image = Image.fromarray(output_esrgan)

        # Optional: Face enhancement
        if self.face_enhancer:
            _, _, enhanced_esrgan = self.face_enhancer.enhance(output_esrgan, has_aligned=False, only_center_face=False, paste_back=True)
            esrgan_image = Image.fromarray(enhanced_esrgan)

        remaining_factor = upscale_factor / 4.0
        if remaining_factor <= 1.0:
            return esrgan_image  # No further upscale needed

        # Step 2: SD Upscale Refinement
        upscaled = self.sd_pipeline(
            prompt=prompt,
            image=esrgan_image,  # Low-res input (after ESRGAN)
            num_inference_steps=num_inference_steps,
            denoising_strength=strength,
            guidance_scale=7.5,
            generator=torch.Generator(device=self.device).manual_seed(42),  # Reproducible
        ).images[0]

        return upscaled


# Example Usage (in model_loader.py)
# from .esrgan_upscaler import ESRGANUpscaler, ESRGANUpscalerCheckpoints
# checkpoints = ESRGANUpscalerCheckpoints()
# upscaler = ESRGANUpscaler(checkpoints, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), dtype=torch.float16)
# enhanced = upscaler.pre_upscale(image, upscale_factor=8.0, prompt="detailed photo")