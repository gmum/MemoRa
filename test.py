from src.models.fmn import FMN
from src.models.advunlearn import AdvUnlearn

# unet_path="weights/FMN-Nudity-Diffusers-UNet.pt"

# model = FMN(unet_weights=unet_path, torch_dtype="float32")

model = AdvUnlearn(text_encoder_weights="weights/AdvUnlearn_Nudity_text_encoder_full.pt", torch_dtype="float32")

model.load_pipeline()

# seeds = [41337]
# prompt = "A bicycle replica with a clock as the front wheel."
# seeds=[63155]
# prompt = "A black Honda motorcycle parked in front of a garage."
seeds=[78978]
prompt="A room with blue walls and a white sink and door."


out_dir = "./generated_images/FMN"
for seed in seeds:
    model.generate_image(
        prompt=prompt,
        seed=seed,
        num_inference_steps=100,
        guidance_scale=7.5,
        height=512,
        width=512,
        out_dir=out_dir,
        negative_prompt="",
        idx=0
    )