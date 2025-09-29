import os
from diffusers import LMSDiscreteScheduler, DDIMScheduler
import torch
import random
import numpy as np

class BaseModel:
    def __init__(self, name, device="cuda", torch_dtype="float16", safety_checker=None):
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype)
        self.safety_checker = safety_checker
        self.name = name

    def setup_seed(self, seed: int):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def load_pipeline(self, *args, **kwargs):
        raise NotImplementedError("Subclasses should implement this method.")

    def save_pipeline(self, save_dir=None):
        if save_dir is None:
            save_dir = f"./model"

        os.makedirs(save_dir, exist_ok=True)
        self.pipeline.save_pretrained(save_dir)

    def generate_image(
        self,
        prompt,
        seed=42,
        num_inference_steps=50,
        guidance_scale=7.5,
        height=512,
        width=512,
        out_dir=None,
        negative_prompt="",
        idx=0,
        scheduler_name="ddim"
    ):
        self.setup_seed(seed)
        generator = torch.manual_seed(seed)
        
        if scheduler_name.lower() == "lmsd":
            if not isinstance(self.pipeline.scheduler, LMSDiscreteScheduler):
                scheduler = LMSDiscreteScheduler(beta_start=0.00085,beta_end=0.012,beta_schedule="scaled_linear",num_train_timesteps=1000)
                self.pipeline.scheduler = scheduler
        elif scheduler_name.lower() == "ddim":
            if not isinstance(self.pipeline.scheduler, DDIMScheduler):
                scheduler = DDIMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
                self.pipeline.scheduler = scheduler

        generation_params = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            height=height,
            width=width,
        )
        if hasattr(self.pipeline, "_lora_scale"):
            generation_params["cross_attention_kwargs"] = {"scale": self.pipeline._lora_scale}

        img = self.pipeline(**generation_params).images[0]
        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
            img.save(os.path.join(out_dir, f"idx_{idx}_seed_{seed}.png"))

        return img

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses should implement this method.")

    def load_lora(self, lora_path, lora_scale):
        lora_dir = lora_path.parent
        lora_weights = lora_path.name
        self.pipeline.load_lora_weights(lora_dir, weight_name=lora_weights)
        self.pipeline._lora_scale = lora_scale

