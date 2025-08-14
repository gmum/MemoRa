import torch
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)

from .base import BaseModel


class ESD(BaseModel):
    def __init__(
        self,
        base_model="CompVis/stable-diffusion-v1-4",
        unet_weights="./ckpt/ESD/diffusers-nudity-ESDu1-UNET.pt",
        feature_extractor=None,
        name="ESD",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.base_model = base_model
        self.unet_weights = unet_weights
        self.feature_extractor = feature_extractor

    def load_pipeline(self):
        vae = AutoencoderKL.from_pretrained(
            self.base_model, subfolder="vae", torch_dtype=self.torch_dtype
        ).to(self.device)
        unet = UNet2DConditionModel.from_pretrained(
            self.base_model, subfolder="unet", torch_dtype=self.torch_dtype
        )
        unet.load_state_dict(torch.load(self.unet_weights, map_location="cpu"))
        unet = unet.to(self.device, dtype=self.torch_dtype)
        tokenizer = CLIPTokenizer.from_pretrained(
            self.base_model, subfolder="tokenizer"
        )
        text_encoder = CLIPTextModel.from_pretrained(
            self.base_model, subfolder="text_encoder", torch_dtype=self.torch_dtype
        ).to(self.device)
        scheduler = DDIMScheduler.from_pretrained(
            self.base_model, subfolder="scheduler"
        )

        self.pipeline = StableDiffusionPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=self.safety_checker,
            feature_extractor=self.feature_extractor,
        ).to(self.device)
        self.pipeline = self.pipeline.to(dtype=self.torch_dtype)

    def load_lora(self, lora_dir, lora_scale, lora_weights):
        self.pipeline.load_lora_weights(lora_dir, weight_name=lora_weights)
        self.pipeline._lora_scale = lora_scale
