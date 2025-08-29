import os

import torch


class BaseModel:
    def __init__(self, name, device="cuda", torch_dtype="float16", safety_checker=None):
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype)
        self.safety_checker = safety_checker
        self.name = name

    def load_pipeline(self, *args, **kwargs):
        pass

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
        idx=0
    ):
        gen = torch.Generator(device=self.device).manual_seed(seed)
        generation_params = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=gen,
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
