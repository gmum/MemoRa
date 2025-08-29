import torch
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    LMSDiscreteScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)

from .base import BaseModel


class AdvUnlearn(BaseModel):
    def __init__(
        self,
        name="AdvUnlearn",
        base_model="CompVis/stable-diffusion-v1-4",
        text_encoder_weights=None,
        feature_extractor=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.base_model = base_model
        self.text_encoder_weights = text_encoder_weights
        self.feature_extractor = feature_extractor

    def load_pipeline(self):
        vae = AutoencoderKL.from_pretrained(
            self.base_model, subfolder="vae", torch_dtype=self.torch_dtype
        ).to(self.device)
        unet = UNet2DConditionModel.from_pretrained(
            self.base_model, subfolder="unet", torch_dtype=self.torch_dtype
        )
        tokenizer = CLIPTokenizer.from_pretrained(
            self.base_model, subfolder="tokenizer"
        )
        text_encoder = CLIPTextModel.from_pretrained(
            self.base_model, subfolder="text_encoder", torch_dtype=self.torch_dtype
        ).to(self.device)
        text_encoder.load_state_dict(self.extract_text_encoder_ckpt(), strict=False)
        scheduler = LMSDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            steps_offset=1,
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

    def extract_text_encoder_ckpt(self):
        full_ckpt = torch.load(self.text_encoder_weights)
        new_ckpt = {}
        for key in full_ckpt.keys():
            if "text_encoder.text_model" in key:
                new_ckpt[key.replace("text_encoder.", "")] = full_ckpt[key]
        return new_ckpt
