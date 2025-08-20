import csv
import os

import hydra
import tqdm
from pathlib import Path

from metrics.nudenet_detector import nudenet_classify


def run_experiment(config):
    folder = config.generate_images.out_dir
    result_path = Path(config.nudenet_classifier.out_dir) / config.nudenet_classifier.file_name
    stats = nudenet_classify(folder, result_path)
    print(stats)


@hydra.main(config_path="../configs", config_name="reunlearning_explicit_content", version_base=None)
def main(config):
    run_experiment(config)


if __name__ == "__main__":
    main()