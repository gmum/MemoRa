# Instrukcja

1. Pobranie diffusers:
```bash
conda create -n reunlearning python=3.13
conda activate reunlearning
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```
2. Poranie wag z ESD, umieszczamy je w katalogu ESD
https://erasing.baulab.info/weights/esd_models/NSFW/
3. Uruchomienie kodu python src/nude_detection.py, config jest domyślnie ustawiony na poprawny ale można zmieniać używając --config-name <config>
