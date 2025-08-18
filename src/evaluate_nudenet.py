import csv
import os

import hydra
import tqdm

from utils.nudity_eval import nudent_classify


def run_experiment(config):
    stats = nudent_classify(config.exp.out_dir)
    print(stats)


@hydra.main(config_path="../configs", config_name="config_generate", version_base=None)
def main(config):
    run_experiment(config)


if __name__ == "__main__":
    main()