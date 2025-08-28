import torch
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    LMSDiscreteScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)

from .base import BaseModel


class BaseUnet(BaseModel):
    def __init__(
        self,
        name,
        base_model="CompVis/stable-diffusion-v1-4",
        unet_weights=None,
        feature_extractor=None,
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
        scheduler = LMSDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            steps_offset=1
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
