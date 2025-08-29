import csv
import os

import hydra
import tqdm
from pathlib import Path

from cleanfid import fid

def calculate_fid(config):

    score = fid.compute_fid(config.fid.first_dir, config.fid.second_dir, num_workers=0) ## only for Windows: num_workers=0
    print(f'FID score: {score}')

    result_path = Path(config.fid.out_dir) / config.fid.file_name
    result_path.parent.mkdir(parents=True, exist_ok=True)

    with open(result_path, "w", encoding="utf-8") as f:
        f.write(f"Fid score for folders: \n")
        f.write(f"-first folder: {config.fid.first_dir}  \n")
        f.write(f"-second folder: {config.fid.second_dir}  \n\n")
        f.write(str(score))


@hydra.main(config_path="../configs", config_name="reunlearning_explicit_content", version_base=None)
def main(config):
    calculate_fid(config)


if __name__ == "__main__":
    main()
