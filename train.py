#save model
import os, torch, math
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTokenizer, CLIPTextModel
from torchvision import transforms as tfms

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 
original_model = "CompVis/stable-diffusion-v1-4"
esd_unet = "./ESD/diffusers-nudity-ESDu1-UNET.pt"

vae = AutoencoderKL.from_pretrained(original_model, subfolder="vae", torch_dtype=torch_dtype).to(device)
unet = UNet2DConditionModel.from_pretrained(original_model, subfolder="unet", torch_dtype=torch_dtype)
unet.load_state_dict(torch.load(esd_unet, map_location="cpu"))
unet = unet.to(device, dtype=torch_dtype)
tokenizer = CLIPTokenizer.from_pretrained(original_model, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(original_model, subfolder="text_encoder", torch_dtype=torch_dtype).to(device)
scheduler = DDIMScheduler.from_pretrained(original_model, subfolder="scheduler")

pipe_esd = StableDiffusionPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=scheduler,
    safety_checker=None,
    feature_extractor=None,
).to(device)
pipe_esd = pipe_esd.to(dtype=torch.float16)

model_dir = "./model"
pipe_esd.save_pretrained(model_dir)
print("Save:", model_dir)


#create json for LoRA

import os
import json
import glob

dataset_dir = "./dataset"
output = os.path.join(dataset_dir, "metadata.jsonl")
prompt = "a photo of the nude person"

extensions = ("*.jpg", "*.jpeg", "*.png")

files = []
for ext in extensions:
    files.extend(glob.glob(os.path.join(dataset_dir, ext)))

files = sorted([os.path.basename(f) for f in files])

if not files:
    raise SystemExit(f"Error!")

with open(output, "w", encoding="utf-8") as f:
    for fn in files:
        rec = {"file_name": fn, "text": prompt}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"Done")



#training LoRA
import subprocess

cmd = [
    "python", "diffusers/examples/text_to_image/train_text_to_image_lora.py",
    "--pretrained_model_name_or_path", "./model",
    "--train_data_dir", "./dataset",
    "--dataloader_num_workers", "0",
    "--resolution", "512",
    "--center_crop",
    "--random_flip",
    "--train_batch_size", "1",
    "--gradient_accumulation_steps", "4",
    "--max_train_steps", "500",
    "--learning_rate", "1e-4",
    "--rank", "4",
    "--max_grad_norm", "1",
    "--lr_scheduler", "cosine",
    "--lr_warmup_steps", "0",
    "--mixed_precision", "fp16",
    "--output_dir", "./lora-out",
    "--checkpointing_steps", "500",
    "--validation_prompt", "A photo of a nude man",
    "--seed", "1337",
    "--gradient_checkpointing"
]

subprocess.run(cmd)
