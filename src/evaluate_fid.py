import hydra
from T2IBenchmark import calculate_fid
import os

@hydra.main(config_path="../configs", config_name="reunlearning_nudity", version_base=None)
def fid(config):
    result, _ = calculate_fid(config.fid.first_dir, config.fid.second_dir)
    print(result)
    
    file_path = config.fid.out_dir + "/" + config.fid.file_name
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(str(result))

if __name__ == "__main__":
    fid()