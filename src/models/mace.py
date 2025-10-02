import torch
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import StableDiffusionPipeline

from .base import BaseModel


class MACE(BaseModel):
    def __init__(
        self,
        base_model="CompVis/stable-diffusion-v1-4",
        name="MACE",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.base_model = base_model

    def load_pipeline(self):
        self.pipeline = StableDiffusionPipeline.from_pretrained(
        self.base_model,
        torch_dtype=torch.float16,
        ).to("cuda")