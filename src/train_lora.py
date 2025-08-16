import csv
import os

import hydra
import tqdm

from utils.lora_utils import create_json_metadata, train_lora


def run_experiment(config):
    create_json_metadata(config.dataset.path, config.dataset.prompt)
    
    train_lora(
        model_dir=config.model.path,
        dataset_dir=config.dataset.path,
        output_dir=config.exp.output_dir,
        validation_prompt=config.exp.validation_prompt,
        steps=config.exp.steps,
        rank=config.exp.rank,
        batch_size=config.exp.batch_size,
        learning_rate=config.exp.learning_rate,
        seed=config.exp.seed,
    )


@hydra.main(config_path="../configs", config_name="config_lora", version_base=None)
def main(config):
    run_experiment(config)


if __name__ == "__main__":
    main()