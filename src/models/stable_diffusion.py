from diffusers import StableDiffusionPipeline

from .base import BaseModel


class StableDiffusion(BaseModel):
    def __init__(
        self,
        base_model="CompVis/stable-diffusion-v1-4",
        name="StableDiffusion",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.base_model = base_model

    def load_pipeline(self):
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.base_model,
            safety_checker=self.safety_checker,
            torch_dtype=self.torch_dtype,
        ).to(self.device)
