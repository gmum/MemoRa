import os
import json
import glob
import subprocess


def create_json_metadata(dataset_dir, prompt, output_name = "metadata.jsonl"):
    output = os.path.join(dataset_dir, output_name)
    extensions = ("*.jpg", "*.jpeg", "*.png")

    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(dataset_dir, ext)))
    files = sorted([os.path.basename(f) for f in files])

    if not files:
        raise SystemExit(f"Eror!")

    with open(output, "w", encoding="utf-8") as f:
        for fn in files:
            rec = {"file_name": fn, "text": prompt}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Saved metadata: {output}!")
    return output


def train_lora(
    model_dir,
    dataset_dir,
    output_dir,
    validation_prompt,
    steps,
    rank,
    batch_size,
    learning_rate,
    seed
):
    cmd = [
    "python", "diffusers/examples/text_to_image/train_text_to_image_lora.py",
    "--pretrained_model_name_or_path", str(model_dir),
    "--train_data_dir", str(dataset_dir),
    "--dataloader_num_workers", "0",
    "--resolution", "512",
    "--center_crop",
    "--random_flip",
    "--train_batch_size", str(batch_size),
    "--gradient_accumulation_steps", "4",
    "--max_train_steps", str(steps),
    "--learning_rate", str(learning_rate),
    "--rank", str(rank),
    "--max_grad_norm", "1",
    "--lr_scheduler", "cosine",
    "--lr_warmup_steps", "0",
    "--mixed_precision", "fp16",
    "--output_dir", str(output_dir),
    "--checkpointing_steps", "500",
    "--validation_prompt", str(validation_prompt),
    "--seed", str(seed),
    "--gradient_checkpointing"
    ]

    subprocess.run(cmd, check=True)
