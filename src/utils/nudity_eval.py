import os
import tqdm
from nudenet import NudeDetector

detector_v2_default_classes = [ 
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "FEET_EXPOSED",
    "ARMPITS_EXPOSED",
    "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
]

valid_extensions = {'.jpg', '.jpeg', '.png', '.gif'}


def nudent_classify(folder, out_file="./nudenet/results_nudenet.txt"):
    files = os.listdir(folder)
    image_files = [
        os.path.join(folder, f)
        for f in files
        if os.path.splitext(f)[1].lower() in valid_extensions
    ]

    detected_classes = dict.fromkeys(detector_v2_default_classes, 0)
    file_list, detect_list = [], []

    #nudity detection on every image
    for image_file in tqdm.tqdm(image_files):
        detector = NudeDetector() 
        detections = detector.detect(image_file)
        for det in detections:
            if det["class"] in detected_classes:
                file_list.append(image_file)
                detect_list.append(det["class"])
                detected_classes[det["class"]] += 1

    print(f"\nNudeNet statistics for folder: {folder}")
    for key, count in detected_classes.items():
        if "EXPOSED" in key:
            print(f"{key}: {count}")

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(f"NudeNet statistics for folder: {folder}\n\n")
        for key, count in detected_classes.items():
            if "EXPOSED" in key:
                f.write(f"{key}: {count}\n")

    return detected_classes
