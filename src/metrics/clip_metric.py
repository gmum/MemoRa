import os
import pandas as pd
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel


def calculate_clip(csv_path, out_dir_images, out_dir, file_name, device="cuda:0"):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
  
    df = pd.read_csv(csv_path, index_col="index")

    clip_scores = []

    count = 0
    for idx, row in df.iterrows():
        prompt = row["prompt"]
        seed = int(row.get("evaluation_seed") or 2024)
        img_filename = f"idx_{idx}_seed_{seed}.png"
        img_path = os.path.join(out_dir_images, img_filename)

        image = Image.open(img_path)

        inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True).to(device)

        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image

        clip_scores.append(logits_per_image.item())

    average_clip_score = sum(clip_scores) / (len(clip_scores))
    content = f'Mean CLIP Score = {average_clip_score}'
    file_path = out_dir + "/" + file_name
    
    print(content)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)
    return average_clip_score, file_path